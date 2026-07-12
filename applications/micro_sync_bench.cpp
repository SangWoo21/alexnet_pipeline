// ============================================================
// micro_sync_bench.cpp
//
// Producer-Consumer 마이크로벤치마크: pthread_mutex vs futex
//   - N producers × M consumers × shared bounded queue
//   - 같은 알고리즘, 동기화 primitive만 교체
//   - 측정: throughput, latency dist, sync_blocked, ctxt switches
//
// 목적:
//   AlexNet 단일 파이프라인에서는 contention 부재로 futex 효과 안 보임.
//   본 벤치마크는 진짜 contention이 있는 multi-thread 환경에서
//   futex 효과가 명확히 발현됨을 입증.
//
// Compile:
//   g++ -O2 -std=c++17 -pthread micro_sync_bench.cpp -o micro_sync_bench
//
// Usage:
//   ./micro_sync_bench -p 4 -c 4 -n 100000 -m pthread
//   ./micro_sync_bench -p 4 -c 4 -n 100000 -m futex
//   (-p N producers, -c M consumers, -n items_per_producer, -m mode)
// ============================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/futex.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

// ============================================================
// 설정
// ============================================================
static int  N_PRODUCERS = 4;
static int  N_CONSUMERS = 4;
static int  ITEMS_PER_PRODUCER = 100000;
static const char* MODE = "pthread";   // "pthread" or "futex"
static const int QUEUE_CAPACITY = 64;

// ============================================================
// 결정론적 동기화 카운터
// ============================================================
static atomic<long> g_sync_attempts{0};
static atomic<long> g_sync_blocked{0};

// ============================================================
// Futex helpers
// ============================================================
static int futex_wait_val(atomic<int>* a, int expected) {
    return syscall(SYS_futex, (int*)a, FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}
static int futex_wake(atomic<int>* a, int n) {
    return syscall(SYS_futex, (int*)a, FUTEX_WAKE_PRIVATE, n, nullptr, nullptr, 0);
}

// ============================================================
// Bounded queue — 두 버전이 같은 인터페이스
// ============================================================
class QueueBase {
public:
    virtual ~QueueBase() = default;
    virtual void push(int item) = 0;
    virtual int  pop()          = 0;
};

// ---- pthread_mutex + pthread_cond 버전 ----
class PthreadQueue : public QueueBase {
    int             buf[QUEUE_CAPACITY];
    int             head = 0, tail = 0, count = 0;
    pthread_mutex_t mtx;
    pthread_cond_t  cv_not_empty;
    pthread_cond_t  cv_not_full;
public:
    PthreadQueue() {
        pthread_mutex_init(&mtx, nullptr);
        pthread_cond_init(&cv_not_empty, nullptr);
        pthread_cond_init(&cv_not_full, nullptr);
    }
    ~PthreadQueue() {
        pthread_mutex_destroy(&mtx);
        pthread_cond_destroy(&cv_not_empty);
        pthread_cond_destroy(&cv_not_full);
    }
    void push(int item) override {
        g_sync_attempts.fetch_add(1, memory_order_relaxed);
        pthread_mutex_lock(&mtx);
        bool blocked = false;
        while (count == QUEUE_CAPACITY) {
            blocked = true;
            pthread_cond_wait(&cv_not_full, &mtx);
        }
        buf[tail] = item;
        tail = (tail + 1) % QUEUE_CAPACITY;
        count++;
        pthread_cond_signal(&cv_not_empty);
        pthread_mutex_unlock(&mtx);
        if (blocked) g_sync_blocked.fetch_add(1, memory_order_relaxed);
    }
    int pop() override {
        g_sync_attempts.fetch_add(1, memory_order_relaxed);
        pthread_mutex_lock(&mtx);
        bool blocked = false;
        while (count == 0) {
            blocked = true;
            pthread_cond_wait(&cv_not_empty, &mtx);
        }
        int item = buf[head];
        head = (head + 1) % QUEUE_CAPACITY;
        count--;
        pthread_cond_signal(&cv_not_full);
        pthread_mutex_unlock(&mtx);
        if (blocked) g_sync_blocked.fetch_add(1, memory_order_relaxed);
        return item;
    }
};

// ---- Futex 버전 ----
// 핵심: count_atomic을 futex word로 사용. 빈/꽉 상태를 atomic 검사로 fast path.
// 다만 push/pop은 여러 thread가 동시에 들어가면 안 되므로 작은 spin lock 사용.
class FutexQueue : public QueueBase {
    int             buf[QUEUE_CAPACITY];
    int             head = 0, tail = 0;
    atomic<int>     count{0};               // futex word
    atomic<int>     write_lock{0};          // tiny spin lock for buf updates
    atomic<int>     waiters_not_empty{0};
    atomic<int>     waiters_not_full{0};

    void lock() {
        int expected = 0;
        while (!write_lock.compare_exchange_weak(
                expected, 1, memory_order_acquire, memory_order_relaxed)) {
            expected = 0;
            this_thread::yield();
        }
    }
    void unlock() { write_lock.store(0, memory_order_release); }

public:
    void push(int item) override {
        g_sync_attempts.fetch_add(1, memory_order_relaxed);
        bool blocked = false;
        while (true) {
            int c = count.load(memory_order_acquire);
            if (c < QUEUE_CAPACITY) {
                lock();
                int c2 = count.load(memory_order_acquire);
                if (c2 < QUEUE_CAPACITY) {
                    buf[tail] = item;
                    tail = (tail + 1) % QUEUE_CAPACITY;
                    count.fetch_add(1, memory_order_release);
                    unlock();
                    // wake one consumer if waiting
                    if (waiters_not_empty.load(memory_order_acquire) > 0) {
                        futex_wake(&count, 1);
                    }
                    if (blocked) g_sync_blocked.fetch_add(1, memory_order_relaxed);
                    return;
                }
                unlock();
            }
            // queue full -> wait
            blocked = true;
            waiters_not_full.fetch_add(1, memory_order_acq_rel);
            int c_now = count.load(memory_order_acquire);
            if (c_now == QUEUE_CAPACITY) {
                futex_wait_val(&count, c_now);
            }
            waiters_not_full.fetch_sub(1, memory_order_acq_rel);
        }
    }
    int pop() override {
        g_sync_attempts.fetch_add(1, memory_order_relaxed);
        bool blocked = false;
        while (true) {
            int c = count.load(memory_order_acquire);
            if (c > 0) {
                lock();
                int c2 = count.load(memory_order_acquire);
                if (c2 > 0) {
                    int item = buf[head];
                    head = (head + 1) % QUEUE_CAPACITY;
                    count.fetch_sub(1, memory_order_release);
                    unlock();
                    if (waiters_not_full.load(memory_order_acquire) > 0) {
                        futex_wake(&count, 1);
                    }
                    if (blocked) g_sync_blocked.fetch_add(1, memory_order_relaxed);
                    return item;
                }
                unlock();
            }
            // queue empty -> wait
            blocked = true;
            waiters_not_empty.fetch_add(1, memory_order_acq_rel);
            int c_now = count.load(memory_order_acquire);
            if (c_now == 0) {
                futex_wait_val(&count, c_now);
            }
            waiters_not_empty.fetch_sub(1, memory_order_acq_rel);
        }
    }
};

// ============================================================
// Latency 측정 (각 push/pop의 시간)
// ============================================================
struct LatencyLog {
    vector<double> push_us;
    vector<double> pop_us;
};

// ============================================================
// 워커 스레드 ctxt switch 측정
// ============================================================
static void read_thread_ctxt(long& vol, long& nonvol) {
    pid_t tid = syscall(SYS_gettid);
    char path[256];
    snprintf(path, sizeof(path), "/proc/self/task/%d/status", (int)tid);
    FILE* f = fopen(path, "r");
    vol = nonvol = 0;
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "voluntary_ctxt_switches:", 24) == 0)
            sscanf(line + 24, "%ld", &vol);
        else if (strncmp(line, "nonvoluntary_ctxt_switches:", 27) == 0)
            sscanf(line + 27, "%ld", &nonvol);
    }
    fclose(f);
}

// ============================================================
// 워커
// ============================================================
struct WorkerStat {
    long vol_ctxt = 0;
    long nonvol_ctxt = 0;
};

static void producer_thread(QueueBase* q, int id, WorkerStat* stat, LatencyLog* log) {
    long v0, nv0;
    read_thread_ctxt(v0, nv0);

    for (int i = 0; i < ITEMS_PER_PRODUCER; i++) {
        auto t0 = Clock::now();
        q->push(id * 1000000 + i);
        auto t1 = Clock::now();
        log->push_us.push_back(chrono::duration<double, micro>(t1 - t0).count());
    }

    long v1, nv1;
    read_thread_ctxt(v1, nv1);
    stat->vol_ctxt = v1 - v0;
    stat->nonvol_ctxt = nv1 - nv0;
}

static void consumer_thread(QueueBase* q, int items_to_consume,
                             WorkerStat* stat, LatencyLog* log) {
    long v0, nv0;
    read_thread_ctxt(v0, nv0);

    for (int i = 0; i < items_to_consume; i++) {
        auto t0 = Clock::now();
        (void)q->pop();
        auto t1 = Clock::now();
        log->pop_us.push_back(chrono::duration<double, micro>(t1 - t0).count());
    }

    long v1, nv1;
    read_thread_ctxt(v1, nv1);
    stat->vol_ctxt = v1 - v0;
    stat->nonvol_ctxt = nv1 - nv0;
}

// ============================================================
// 통계
// ============================================================
struct Stat { double mean, p50, p95, p99; };
static Stat calc_stat(vector<double>& v) {
    if (v.empty()) return {0, 0, 0, 0};
    sort(v.begin(), v.end());
    double sum = 0;
    for (auto x : v) sum += x;
    int n = v.size();
    return {sum / n, v[n * 50 / 100], v[n * 95 / 100], v[n * 99 / 100]};
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && i + 1 < argc) {
            char f = argv[i][1]; i++;
            if (f == 'p') N_PRODUCERS = atoi(argv[i]);
            if (f == 'c') N_CONSUMERS = atoi(argv[i]);
            if (f == 'n') ITEMS_PER_PRODUCER = atoi(argv[i]);
            if (f == 'm') MODE = argv[i];
        }
    }

    QueueBase* q;
    if (strcmp(MODE, "futex") == 0) q = new FutexQueue();
    else                            q = new PthreadQueue();

    long total_items = (long)N_PRODUCERS * ITEMS_PER_PRODUCER;
    long items_per_consumer = total_items / N_CONSUMERS;
    long remainder = total_items - items_per_consumer * N_CONSUMERS;

    vector<thread> threads;
    vector<WorkerStat> pstats(N_PRODUCERS), cstats(N_CONSUMERS);
    vector<LatencyLog> plogs(N_PRODUCERS), clogs(N_CONSUMERS);

    auto t_start = Clock::now();

    for (int i = 0; i < N_PRODUCERS; i++) {
        threads.emplace_back(producer_thread, q, i, &pstats[i], &plogs[i]);
    }
    for (int i = 0; i < N_CONSUMERS; i++) {
        long n = items_per_consumer + (i < remainder ? 1 : 0);
        threads.emplace_back(consumer_thread, q, (int)n, &cstats[i], &clogs[i]);
    }

    for (auto& t : threads) t.join();
    auto t_end = Clock::now();

    double duration_s = chrono::duration<double>(t_end - t_start).count();
    double throughput = total_items / duration_s;

    vector<double> all_push, all_pop;
    for (auto& l : plogs) all_push.insert(all_push.end(), l.push_us.begin(), l.push_us.end());
    for (auto& l : clogs) all_pop .insert(all_pop .end(), l.pop_us .begin(), l.pop_us .end());
    Stat ps = calc_stat(all_push);
    Stat cs = calc_stat(all_pop);

    long total_vol = 0, total_nonvol = 0;
    for (auto& s : pstats) { total_vol += s.vol_ctxt; total_nonvol += s.nonvol_ctxt; }
    for (auto& s : cstats) { total_vol += s.vol_ctxt; total_nonvol += s.nonvol_ctxt; }

    long attempts = g_sync_attempts.load();
    long blocked  = g_sync_blocked.load();
    double fastpath = attempts > 0
        ? 100.0 * (1.0 - (double)blocked / attempts) : 0;

    printf("\n========================================================\n");
    printf(" mode=%s  P=%d  C=%d  items/P=%d  total=%ld\n",
           MODE, N_PRODUCERS, N_CONSUMERS, ITEMS_PER_PRODUCER, total_items);
    printf("========================================================\n");
    printf("Duration         : %.3fs\n", duration_s);
    printf("Throughput       : %.0f ops/sec (%.1fM/sec)\n",
           throughput, throughput / 1e6);
    printf("\n");
    printf("Push latency (us)| mean=%.3f  p50=%.3f  p95=%.3f  p99=%.3f\n",
           ps.mean, ps.p50, ps.p95, ps.p99);
    printf("Pop  latency (us)| mean=%.3f  p50=%.3f  p95=%.3f  p99=%.3f\n",
           cs.mean, cs.p50, cs.p95, cs.p99);
    printf("\n");
    printf("Sync attempts    : %ld\n", attempts);
    printf("Sync blocked     : %ld  (%.2f%%)\n",
           blocked, attempts > 0 ? 100.0 * blocked / attempts : 0);
    printf("Fastpath rate    : %.2f%%\n", fastpath);
    printf("\n");
    printf("Worker vol ctxt  : %ld\n", total_vol);
    printf("Worker nonvol ctxt: %ld\n", total_nonvol);
    printf("Total ctxt/op    : %.4f\n", (double)(total_vol + total_nonvol) / total_items);
    printf("========================================================\n");

    delete q;
    return 0;
}

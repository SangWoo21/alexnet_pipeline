// ============================================================
// AlexNet 파이프라인 futex 검증 (멀티 CPU Worker)
//
// 구조:
//   Producer(1) → [GPU Queue] → GPU Worker(1, Conv1-5)
//               → [CPU Queue] → CPU Worker(N, FC1-3)
//                                  ↑
//                  N개 워커가 같은 CPU Queue의 lock/cond 경쟁
//                  → 여기서 contention 발생 → futex 효과 측정 가능
//
// ★ 설계 원칙:
//   - CPU Worker별 독립 FC 중간 버퍼 (race 없음, checksum 완전 일치)
//   - FC weight는 읽기 전용이라 공유
//   - 동기화 primitive만 pthread ↔ futex 교체 (동일 알고리즘)
//   - CPU Queue가 contention 지점
//
// 사용법:
//   ./alexnet_futex -w 4 -m pthread -f 100
//   ./alexnet_futex -w 4 -m futex   -f 100
//   (-w CPU worker 수, -m 동기화 모드, -f 프레임 수)
// ============================================================
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "../src/layer.cu"
#include "../include/mnist.h"
#include "../include/pixels.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <dirent.h>
#include <initializer_list>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

static constexpr int INPUT_SIZE = 227 * 227 * 3;
static constexpr int POOL3_SIZE = 6 * 6 * 256;   // GPU→CPU 전달 텐서 (Conv1-5 출력)
static constexpr int FC1_O = 4096;
static constexpr int FC2_O = 4096;
static constexpr int FC3_O = 1000;

static int  NUM_FRAMES   = 100;
static int  WARMUP       = 10;
static int  N_CPU_WORKER = 4;
static const char* MODE  = "pthread";
static bool g_use_futex  = false;

double iniStart = gettime();
ALayer L_input = ALayer(0, 0, 227 * 227 * 3, "input");
ALayer L_c1 = ALayer(11 * 11 * 3, 2 * 48, 2 * 55 * 55 * 48, "c1");
ALayer L_p1 = ALayer(3 * 3, 2 * 1, 2 * 31 * 31 * 48, "p1");
ALayer L_c2 = ALayer(5 * 5 * 48, 2 * 128, 2 * 128 * 27 * 27, "c2");
ALayer L_p2 = ALayer(3 * 3, 2 * 1, 2 * 15 * 15 * 128, "p2");
ALayer L_c3 = ALayer(3 * 3 * 256, 384, 2 * 13 * 13 * 192, "c3");
ALayer L_c4 = ALayer(3 * 3 * 384, 2 * 192, 2 * 13 * 13 * 192, "c4");
ALayer L_c5 = ALayer(3 * 3 * 384, 2 * 128, 2 * 13 * 13 * 128, "c5");
ALayer L_p3 = ALayer(3 * 3, 2 * 1, 2 * 6 * 6 * 128, "p3");
ALayer L_f1 = ALayer(6 * 6 * 256, 2 * 2048, 4096 * 1, "f1");
ALayer L_f2 = ALayer(1 * 4096, 2 * 2048, 4096 * 1, "f2");
ALayer L_f3 = ALayer(1 * 4096, 1000, 1000, "f3");
double iniEnd = gettime();

// ============================================================
// 동기화 카운터
// ============================================================
static atomic<long> g_sync_attempts{0};
static atomic<long> g_sync_blocked{0};
static atomic<long> g_futex_wait_calls{0};
static atomic<long> g_futex_wake_calls{0};

static int sys_futex(atomic<int>* uaddr, int op, int val) {
    if ((op & FUTEX_CMD_MASK) == FUTEX_WAIT)
        g_futex_wait_calls.fetch_add(1, memory_order_relaxed);
    else if ((op & FUTEX_CMD_MASK) == FUTEX_WAKE)
        g_futex_wake_calls.fetch_add(1, memory_order_relaxed);
    return syscall(SYS_futex, (int*)uaddr, op, val, nullptr, nullptr, 0);
}

// ============================================================
// 동기화 primitive: pthread / futex 두 구현
// ============================================================
struct PthreadLock {
    pthread_mutex_t m;
    PthreadLock()  { pthread_mutex_init(&m, nullptr); }
    void lock()    { pthread_mutex_lock(&m); }
    void unlock()  { pthread_mutex_unlock(&m); }
};
struct PthreadCond {
    pthread_cond_t c;
    PthreadCond()  { pthread_cond_init(&c, nullptr); }
    void wait(PthreadLock& l)  { pthread_cond_wait(&c, &l.m); }
    void signal()              { pthread_cond_signal(&c); }
    void broadcast()           { pthread_cond_broadcast(&c); }
};

// Drepper 3-state futex mutex
struct FutexLock {
    atomic<int> state{0};
    void lock() {
        int c = 0;
        if (state.compare_exchange_strong(c, 1, memory_order_acquire, memory_order_relaxed))
            return;
        if (c != 2) c = state.exchange(2, memory_order_acquire);
        while (c != 0) {
            sys_futex(&state, FUTEX_WAIT_PRIVATE, 2);
            c = state.exchange(2, memory_order_acquire);
        }
    }
    void unlock() {
        if (state.fetch_sub(1, memory_order_release) != 1) {
            state.store(0, memory_order_release);
            sys_futex(&state, FUTEX_WAKE_PRIVATE, 1);
        }
    }
};
// futex condvar (seq + waiters로 불필요 wake 회피)
struct FutexCond {
    atomic<int> seq{0};
    atomic<int> waiters{0};
    void wait(FutexLock& l) {
        waiters.fetch_add(1, memory_order_acq_rel);
        int old = seq.load(memory_order_acquire);
        l.unlock();
        sys_futex(&seq, FUTEX_WAIT_PRIVATE, old);
        waiters.fetch_sub(1, memory_order_acq_rel);
        l.lock();
    }
    void signal() {
        if (waiters.load(memory_order_acquire) > 0) {
            seq.fetch_add(1, memory_order_release);
            sys_futex(&seq, FUTEX_WAKE_PRIVATE, 1);
        }
    }
    void broadcast() {
        if (waiters.load(memory_order_acquire) > 0) {
            seq.fetch_add(1, memory_order_release);
            sys_futex(&seq, FUTEX_WAKE_PRIVATE, INT_MAX);
        }
    }
};

// ============================================================
// Task: GPU가 만든 Conv 결과(pool3)를 CPU Worker에게 전달
// ============================================================
struct Task {
    int   fid;
    float pool3[POOL3_SIZE];   // Conv1-5 출력
};

// ============================================================
// 멀티-프로듀서/멀티-컨슈머 큐 (CPU Queue가 contention 지점)
//   - GPU Worker 1개가 push, CPU Worker N개가 pop → MPMC지만 여기선 SPMC
//   - primitive만 템플릿 교체
// ============================================================
template <typename Lock, typename Cond>
class TaskQueue {
    vector<Task*> buf;
    int    cap;
    int    head = 0, tail = 0, count = 0;
    bool   closed = false;
    Lock   mtx;
    Cond   not_empty;
    Cond   not_full;
public:
    explicit TaskQueue(int capacity) : cap(capacity) { buf.resize(capacity, nullptr); }

    void push(Task* t) {
        g_sync_attempts.fetch_add(1, memory_order_relaxed);
        mtx.lock();
        bool blocked = false;
        while (count == cap) { blocked = true; not_full.wait(mtx); }
        buf[tail] = t; tail = (tail + 1) % cap; count++;
        not_empty.signal();
        mtx.unlock();
        if (blocked) g_sync_blocked.fetch_add(1, memory_order_relaxed);
    }

    // 반환 nullptr이면 큐가 닫혔고 더 이상 task 없음
    Task* pop() {
        g_sync_attempts.fetch_add(1, memory_order_relaxed);
        mtx.lock();
        bool blocked = false;
        while (count == 0 && !closed) { blocked = true; not_empty.wait(mtx); }
        if (count == 0 && closed) { mtx.unlock(); return nullptr; }
        Task* t = buf[head]; head = (head + 1) % cap; count--;
        not_full.signal();
        mtx.unlock();
        if (blocked) g_sync_blocked.fetch_add(1, memory_order_relaxed);
        return t;
    }

    void close() {
        mtx.lock();
        closed = true;
        not_empty.broadcast();   // 모든 대기 워커 깨워서 종료시킴
        mtx.unlock();
    }
};

// 추상 인터페이스
struct IQueue {
    virtual ~IQueue() = default;
    virtual void  push(Task*) = 0;
    virtual Task* pop()       = 0;
    virtual void  close()     = 0;
};
template <typename Q>
struct QWrap : IQueue {
    Q q;
    explicit QWrap(int cap) : q(cap) {}
    void  push(Task* t) override { q.push(t); }
    Task* pop()         override { return q.pop(); }
    void  close()       override { q.close(); }
};

// ============================================================
// CPU Worker별 독립 FC 버퍼 (race 방지, 완전 정확)
//   weight/bias는 전역 공유 (읽기 전용), preact/output만 워커별
// ============================================================
struct FCBuffers {
    float f1_preact[FC1_O], f1_out[FC1_O];
    float f2_preact[FC2_O], f2_out[FC2_O];
    float f3_preact[FC3_O];
};

// CPU FC1-3 (독립 버퍼 사용). result에 FC3 결과 기록
static void cpu_fc_all(const float* pool3, FCBuffers* b, float* result) {
    // FC1
    for (int o = 0; o < L_f1.O; o++) {
        float sum = L_f1.bias[o];
        const float* w = &((float(*)[6*6*256])L_f1.weight)[o][0];
        for (int i = 0; i < 6*6*256; i++) sum += pool3[i] * w[i];
        b->f1_preact[o] = sum;
        b->f1_out[o] = (sum > 0.f) ? sum : 0.f;
    }
    // FC2
    for (int o = 0; o < L_f2.O; o++) {
        float sum = L_f2.bias[o];
        const float* w = &((float(*)[4096])L_f2.weight)[o][0];
        for (int i = 0; i < 4096; i++) sum += b->f1_out[i] * w[i];
        b->f2_preact[o] = sum;
        b->f2_out[o] = (sum > 0.f) ? sum : 0.f;
    }
    // FC3
    for (int o = 0; o < L_f3.O; o++) {
        float sum = L_f3.bias[o];
        const float* w = &((float(*)[4096])L_f3.weight)[o][0];
        for (int i = 0; i < 4096; i++) sum += b->f2_out[i] * w[i];
        b->f3_preact[o] = sum;
        result[o] = sum;
    }
}

// ============================================================
// GPU: Conv1-5 → Pool3 (선행 코드와 동일)
// ============================================================
static void gpu_conv1to5(cudaStream_t st) {
    fp_preact_c1<<<96, dim3(55,55), 0, st>>>(
        (float(*)[227][3])L_input.output, (float(*)[55][55])L_c1.preact,
        (float(*)[11][11][3])L_c1.weight, L_c1.bias);
    Aapply_step_function<<<112, 1280, 0, st>>>(L_c1.preact, L_c1.act_result, L_c1.O);
    normalization_function<<<112, 640, 0, st>>>(L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);
    fp_preact_p1<<<96, dim3(27,27), 0, st>>>(
        (float(*)[55][55])L_c1.output, (float(*)[31][31])L_p1.output);
    fp_preact_c2<<<256, dim3(27,27), 0, st>>>(
        (float(*)[31][31])L_p1.output, (float(*)[27][27])L_c2.preact,
        (float(*)[96][5][5])L_c2.weight, L_c2.bias);
    Aapply_step_function<<<112, 1280, 0, st>>>(L_c2.preact, L_c2.act_result, L_c2.O);
    normalization_function<<<112, 1280, 0, st>>>(L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);
    fp_preact_p2<<<256, dim3(13,13), 0, st>>>(
        (float(*)[27][27])L_c2.output, (float(*)[15][15])L_p2.output);
    fp_preact_c3<<<384, dim3(13,13), 0, st>>>(
        (float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact,
        (float(*)[256][3][3])L_c3.weight, L_c3.bias);
    Aapply_step_function<<<128, 128, 0, st>>>(L_c3.preact, L_c3.output, L_c3.O);
    fp_preact_c4<<<384, dim3(12,12), 0, st>>>(
        (float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact,
        (float(*)[384][3][3])L_c4.weight, L_c4.bias);
    Aapply_step_function<<<128, 128, 0, st>>>(L_c4.preact, L_c4.output, L_c4.O);
    fp_preact_c5<<<256, dim3(12,12), 0, st>>>(
        (float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact,
        (float(*)[384][3][3])L_c5.weight, L_c5.bias);
    Aapply_step_function<<<128, 128, 0, st>>>(L_c5.preact, L_c5.output, L_c5.O);
    fp_preact_p3<<<256, dim3(6,6), 0, st>>>(
        (float(*)[13][13])L_c5.output, (float(*)[6][6])L_p3.output);
}

static void set_affinity(initializer_list<int> cores) {
    cpu_set_t cs; CPU_ZERO(&cs);
    for (int c : cores) CPU_SET(c, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}
static void set_affinity_one(int core) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// ============================================================
// 결과 저장 (fid → checksum 비교용)
// ============================================================
static vector<float> g_results;        // [NUM_FRAMES][FC3_O] 평탄화
static atomic<int>   g_done_count{0};

// ============================================================
// GPU Worker: Producer 역할도 겸함 (입력 준비 → Conv → CPU Queue로 push)
// ============================================================
struct Metrics {
    vector<double> gpu_run;
    vector<double> cpu_run;
    chrono::time_point<Clock> t_start, t_end;
    atomic<int> ready{0};
};

static void gpu_worker(IQueue* cq, Metrics* m, double data[227][227][3],
                       vector<Task>* task_pool) {
    set_affinity_one(1);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaStreamAttachMemAsync(st, &Ainput_a, 0, cudaMemAttachHost);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    m->ready.fetch_add(1);
    while (m->ready.load() < 1 + N_CPU_WORKER) this_thread::yield();
    m->t_start = Clock::now();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        // 입력 준비
        for (int i = 0; i < 227; i++)
            for (int j = 0; j < 227; j++)
                for (int k = 0; k < 3; k++)
                    ((double*)Ainput_a)[i*227*3 + j*3 + k] = data[i][j][k];

        cudaEventRecord(e0, st);
        gpu_conv1to5(st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        m->gpu_run.push_back(ms);

        // task에 pool3 복사 후 CPU Queue로
        Task* t = &(*task_pool)[fid];
        t->fid = fid;
        memcpy(t->pool3, L_p3.output, POOL3_SIZE * sizeof(float));
        cq->push(t);
    }
    cq->close();   // 모든 frame push 완료 → 큐 닫음

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaStreamDestroy(st);
}

// ============================================================
// CPU Worker: CPU Queue에서 task 꺼내 FC 실행 (독립 버퍼)
//   여러 워커가 같은 큐 경쟁 → contention
// ============================================================
static void cpu_worker(IQueue* cq, int worker_id, Metrics* m) {
    // 워커를 코어 2~5에 분산 배치
    set_affinity_one(2 + (worker_id % 4));

    FCBuffers* buf = new FCBuffers();   // 워커별 독립 버퍼

    m->ready.fetch_add(1);
    while (m->ready.load() < 1 + N_CPU_WORKER) this_thread::yield();

    while (true) {
        Task* t = cq->pop();
        if (t == nullptr) break;   // 큐 닫힘

        auto r0 = Clock::now();
        cpu_fc_all(t->pool3, buf, &g_results[t->fid * FC3_O]);
        auto r1 = Clock::now();
        m->cpu_run.push_back(chrono::duration<double,milli>(r1-r0).count());

        int done = g_done_count.fetch_add(1) + 1;
        if (done == NUM_FRAMES) m->t_end = Clock::now();
    }
    delete buf;
}

// ============================================================
// 통계
// ============================================================
struct Stat { double mean, p50, p95, p99; };
static Stat calc_stat(vector<double> v) {
    if (v.empty()) return {0,0,0,0};
    sort(v.begin(), v.end());
    double s = 0; for (double x : v) s += x;
    int n = v.size();
    return {s/n, v[n*50/100], v[n*95/100], v[n*99/100]};
}

static long read_all_threads_vol_ctxt() {
    long total = 0;
    DIR* dir = opendir("/proc/self/task/");
    if (!dir) return -1;
    struct dirent* e; char path[256];
    while ((e = readdir(dir)) != NULL) {
        if (e->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "/proc/self/task/%s/status", e->d_name);
        FILE* f = fopen(path, "r"); if (!f) continue;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "voluntary_ctxt_switches:", 24) == 0) {
                long v = 0; sscanf(line+24, "%ld", &v); total += v; break;
            }
        }
        fclose(f);
    }
    closedir(dir);
    return total;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (argv[i][0] == '-' && i+1 < argc) {
            char f = argv[i][1]; i++;
            if (f == 'f') NUM_FRAMES   = atoi(argv[i]);
            if (f == 'w') N_CPU_WORKER = atoi(argv[i]);
            if (f == 'm') MODE         = argv[i];
        }
    g_use_futex = (strcmp(MODE, "futex") == 0);

    offset = 1.0;
    // ZeroCopy attach (선행 코드와 동일, GPU worker 내 stream에서 재attach)
    cudaStream_t st0; cudaStreamCreate(&st0);
    cudaStreamAttachMemAsync(st0, &Ainput_a,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac1_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac1_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac1_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac1_o,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ap1_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac2_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac2_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac2_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac2_o,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ap2_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac3_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac3_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac3_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac4_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac4_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac4_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac4_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac5_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac5_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac5_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ac5_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(st0, &Ap3_a,      0, cudaMemAttachHost);
    cudaStreamSynchronize(st0);
    cudaStreamDestroy(st0);

    double test_data[227][227][3] = {};
    for (int i = 0; i < 227; i++)
        for (int j = 0; j < 227; j++)
            for (int k = 0; k < 3; k++)
                test_data[j][k][i] = (double)PIXELS[j][i][k];

    // 워밍업 (GPU Conv + CPU FC 1회씩)
    {
        cudaStream_t stw; cudaStreamCreate(&stw);
        cudaStreamAttachMemAsync(stw, &Ainput_a, 0, cudaMemAttachHost);
        FCBuffers* wb = new FCBuffers();
        float dummy[FC3_O];
        for (int i = 0; i < WARMUP; i++) {
            memcpy(Ainput_a, test_data, INPUT_SIZE*sizeof(double));
            gpu_conv1to5(stw);
            cudaStreamSynchronize(stw);
            cpu_fc_all(L_p3.output, wb, dummy);
        }
        delete wb;
        cudaStreamDestroy(stw);
    }

    // 결과 버퍼, task pool 준비
    g_results.assign((size_t)NUM_FRAMES * FC3_O, 0.0f);
    g_done_count.store(0);
    vector<Task> task_pool(NUM_FRAMES);

    // 큐 생성 (capacity는 worker 수보다 약간 크게 → contention 발생하되 deadlock 없음)
    int qcap = max(4, N_CPU_WORKER);
    IQueue* cq;
    if (g_use_futex) cq = new QWrap<TaskQueue<FutexLock, FutexCond>>(qcap);
    else             cq = new QWrap<TaskQueue<PthreadLock, PthreadCond>>(qcap);

    Metrics m;
    long vol0 = read_all_threads_vol_ctxt();

    // 스레드 시작
    thread gpu_t(gpu_worker, cq, &m, test_data, &task_pool);
    vector<thread> cpu_ts;
    for (int i = 0; i < N_CPU_WORKER; i++)
        cpu_ts.emplace_back(cpu_worker, cq, i, &m);

    gpu_t.join();
    for (auto& t : cpu_ts) t.join();
    long vol1 = read_all_threads_vol_ctxt();

    // 측정 trim (warmup 만큼 제거)
    auto trim = [&](vector<double>& v) {
        int w = min((int)v.size(), WARMUP);
        if (w > 0) v.erase(v.begin(), v.begin()+w);
    };
    trim(m.gpu_run); trim(m.cpu_run);

    Stat gpu = calc_stat(m.gpu_run);
    Stat cpu = calc_stat(m.cpu_run);

    double pipe_s   = chrono::duration<double>(m.t_end - m.t_start).count();
    double pipe_fps = NUM_FRAMES / pipe_s;

    // checksum (마지막 frame)
    double cksum = 0;
    for (int i = 0; i < FC3_O; i++) cksum += g_results[(NUM_FRAMES-1)*FC3_O + i];

    long attempts = g_sync_attempts.load();
    long blocked  = g_sync_blocked.load();
    double fastpath = attempts > 0 ? 100.0*(1.0-(double)blocked/attempts) : 0;
    long fw  = g_futex_wait_calls.load();
    long fwk = g_futex_wake_calls.load();

    printf("\n========================================================\n");
    printf(" AlexNet Pipeline futex 검증 | mode=%s | CPU workers=%d\n",
           MODE, N_CPU_WORKER);
    printf(" (GPU 1 worker + CPU N workers가 같은 CPU Queue 경쟁)\n");
    printf("========================================================\n");
    printf("frames=%d warmup=%d\n", NUM_FRAMES, WARMUP);
    printf("GPU Conv (ms) | mean=%.3f p95=%.3f\n", gpu.mean, gpu.p95);
    printf("CPU FC   (ms) | mean=%.3f p95=%.3f p99=%.3f\n", cpu.mean, cpu.p95, cpu.p99);
    printf("--------------------------------------------------------\n");
    printf("Pipeline FPS  : %.2f  (%.3fs)\n", pipe_fps, pipe_s);
    printf("--------------------------------------------------------\n");
    printf("[동기화 — CPU Queue contention]\n");
    printf(" sync_attempts : %ld\n", attempts);
    printf(" sync_blocked  : %ld  (%.2f%%)\n", blocked,
           attempts>0 ? 100.0*blocked/attempts : 0);
    printf(" fastpath_rate : %.2f%%\n", fastpath);
    printf(" futex_WAIT    : %ld\n", fw);
    printf(" futex_WAKE    : %ld\n", fwk);
    printf(" worker_vol_ctxt: %ld\n", vol1 - vol0);
    printf("--------------------------------------------------------\n");
    printf(" Output_checksum: %.6e\n", cksum);
    printf("========================================================\n");

    delete cq;
    return 0;
}

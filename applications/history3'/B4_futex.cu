// ============================================================
// B4: B3 + Private Futex (최종 구성)
//
// B3 대비 변경:
//  - pthread_mutex + cond_var → Private Futex (waiters 카운터 포함)
//  - 슬롯 상태 동기화 메커니즘 전체 교체
//  - per-slot CUDA stream/event (진짜 파이프라이닝)
//
// 측정 추가:
//  - LLC 캐시 미스 (perf_event_open)
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
#include <initializer_list>
#include <linux/futex.h>
#include <linux/perf_event.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

static constexpr int N_SLOTS    = 6;
static constexpr int INPUT_SIZE = 227 * 227 * 3;
static constexpr int POOL3_SIZE = 6 * 6 * 256;

static int NUM_FRAMES = 100;
static int WARMUP     = 10;

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
// NEW: Private Futex
// ============================================================
static void futex_wait_val(atomic<int>* a, int expected) {
    syscall(SYS_futex, reinterpret_cast<int*>(a),
            FUTEX_WAIT | FUTEX_PRIVATE_FLAG,
            expected, nullptr, nullptr, 0);
}
static void futex_wake_all(atomic<int>* a) {
    // FIX: 한 슬롯의 state에 여러 스테이지가 동시 대기할 수 있어서
    // (예: S2(이번 frame)는 CONV_DONE 대기, S0(다음 wrap frame)은 EMPTY 대기)
    // wake_one을 쓰면 엉뚱한 한 명만 깨우고 나머지는 영영 묶이는 데드락 발생.
    // 모두 깨우면 잘못 깨어난 쪽은 다시 자고 올바른 쪽은 진행 — 안전.
    syscall(SYS_futex, reinterpret_cast<int*>(a),
            FUTEX_WAKE | FUTEX_PRIVATE_FLAG,
            INT_MAX, nullptr, nullptr, 0);
}

// 슬롯 상태를 atomic int로 표현. waiters 카운터로 fast path 보장.
enum { ST_EMPTY = 0, ST_INPUT_READY = 1, ST_CONV_DONE = 2 };

struct FutexSlot {
    atomic<int> state{ST_EMPTY};
    atomic<int> waiters{0};
    int         fid;
    double      input_buf[INPUT_SIZE];
    float       pool3_buf[POOL3_SIZE];

    void wait_until(int target) {
        while (true) {
            int cur = state.load(memory_order_acquire);
            if (cur == target) return;
            waiters.fetch_add(1, memory_order_acq_rel);
            futex_wait_val(&state, cur);
            waiters.fetch_sub(1, memory_order_acq_rel);
        }
    }
    void set_state(int ns) {
        state.store(ns, memory_order_release);
        if (waiters.load(memory_order_acquire) > 0)
            futex_wake_all(&state);   // FIX: wake_one → wake_all (위 주석 참조)
    }
};
static FutexSlot g_slots[N_SLOTS];

static void slot_init() {
    for (int s = 0; s < N_SLOTS; s++) {
        g_slots[s].state.store(ST_EMPTY);
        g_slots[s].waiters.store(0);
        memset(g_slots[s].input_buf, 0, sizeof(g_slots[s].input_buf));
        memset(g_slots[s].pool3_buf, 0, sizeof(g_slots[s].pool3_buf));
    }
}

static void set_affinity(initializer_list<int> cores) {
    cpu_set_t cs; CPU_ZERO(&cs);
    for (int c : cores) CPU_SET(c, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// per-slot CUDA stream/event for true pipelining
static cudaStream_t g_streams[N_SLOTS];
static cudaEvent_t  g_evt[N_SLOTS];
static void init_cuda_slots() {
    for (int s = 0; s < N_SLOTS; s++) {
        cudaStreamCreateWithFlags(&g_streams[s], cudaStreamNonBlocking);
        cudaEventCreate(&g_evt[s]);
    }
}
static void destroy_cuda_slots() {
    for (int s = 0; s < N_SLOTS; s++) {
        cudaEventDestroy(g_evt[s]);
        cudaStreamDestroy(g_streams[s]);
    }
}

// LLC 미스 카운터
static int open_llc_miss_fd(int cpu_id) {
    struct perf_event_attr attr = {};
    attr.type = PERF_TYPE_HARDWARE;
    attr.config = PERF_COUNT_HW_CACHE_MISSES;
    attr.size = sizeof(attr);
    attr.disabled = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv = 1;
    int fd = (int)syscall(SYS_perf_event_open, &attr, 0, cpu_id, -1, 0);
    return fd;
}
static int64_t perf_read(int fd) {
    if (fd < 0) return -1;
    int64_t v = 0;
    if (read(fd, &v, sizeof(v)) != sizeof(v)) return -1;
    return v;
}

struct Metrics {
    vector<double>  s0_wait, s0_run;
    vector<double>  s1_wait, s1_run;
    vector<double>  s1_pure_conv;   // NEW: GPU 순수 conv 시간 (cudaEvent로 측정, 통제 변수)
    vector<double>  s2_wait, s2_run;
    vector<int64_t> s2_llc_miss;
    chrono::time_point<Clock> t_start, t_end;
    atomic<int> ready{0};
    Metrics() {
        s0_wait.reserve(NUM_FRAMES); s0_run.reserve(NUM_FRAMES);
        s1_wait.reserve(NUM_FRAMES); s1_run.reserve(NUM_FRAMES);
        s1_pure_conv.reserve(NUM_FRAMES);
        s2_wait.reserve(NUM_FRAMES); s2_run.reserve(NUM_FRAMES);
        s2_llc_miss.reserve(NUM_FRAMES);
    }
};

static void run_conv_only_gpu(cudaStream_t stream1) {
    fp_preact_c1<<<96, dim3(55,55), 0, stream1>>>(
        (float(*)[227][3])L_input.output, (float(*)[55][55])L_c1.preact,
        (float(*)[11][11][3])L_c1.weight, L_c1.bias);
    Aapply_step_function<<<112, 1280, 0, stream1>>>(L_c1.preact, L_c1.act_result, L_c1.O);
    normalization_function<<<112, 640, 0, stream1>>>(L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);
    fp_preact_p1<<<96, dim3(27,27), 0, stream1>>>(
        (float(*)[55][55])L_c1.output, (float(*)[31][31])L_p1.output);
    fp_preact_c2<<<256, dim3(27,27), 0, stream1>>>(
        (float(*)[31][31])L_p1.output, (float(*)[27][27])L_c2.preact,
        (float(*)[96][5][5])L_c2.weight, L_c2.bias);
    Aapply_step_function<<<112, 1280, 0, stream1>>>(L_c2.preact, L_c2.act_result, L_c2.O);
    normalization_function<<<112, 1280, 0, stream1>>>(L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);
    fp_preact_p2<<<256, dim3(13,13), 0, stream1>>>(
        (float(*)[27][27])L_c2.output, (float(*)[15][15])L_p2.output);
    fp_preact_c3<<<384, dim3(13,13), 0, stream1>>>(
        (float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact,
        (float(*)[256][3][3])L_c3.weight, L_c3.bias);
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c3.preact, L_c3.output, L_c3.O);
    fp_preact_c4<<<384, dim3(12,12), 0, stream1>>>(
        (float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact,
        (float(*)[384][3][3])L_c4.weight, L_c4.bias);
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c4.preact, L_c4.output, L_c4.O);
    fp_preact_c5<<<256, dim3(12,12), 0, stream1>>>(
        (float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact,
        (float(*)[384][3][3])L_c5.weight, L_c5.bias);
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c5.preact, L_c5.output, L_c5.O);
    fp_preact_p3<<<256, dim3(6,6), 0, stream1>>>(
        (float(*)[13][13])L_c5.output, (float(*)[6][6])L_p3.output);
}

static void run_fc_cpu_omp(float* pool3_data) {
    memcpy(L_p3.output, pool3_data, POOL3_SIZE*sizeof(float));
    omp_set_num_threads(4);

    #pragma omp parallel for schedule(static)
    for (int o = 0; o < L_f1.O; o++) {
        float sum = L_f1.bias[o];
        const float* w = &((float(*)[6*6*256])L_f1.weight)[o][0];
        const float* p = (float*)L_p3.output;
        for (int i = 0; i < 6*6*256; i++) sum += p[i] * w[i];
        L_f1.preact[o] = sum;
        L_f1.output[o] = (sum > 0.f) ? sum : 0.f;
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < L_f2.O; o++) {
        float sum = L_f2.bias[o];
        const float* w = &((float(*)[4096])L_f2.weight)[o][0];
        for (int i = 0; i < 4096; i++) sum += L_f1.output[i] * w[i];
        L_f2.preact[o] = sum;
        L_f2.output[o] = (sum > 0.f) ? sum : 0.f;
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < L_f3.O; o++) {
        float sum = L_f3.bias[o];
        const float* w = &((float(*)[4096])L_f3.weight)[o][0];
        for (int i = 0; i < 4096; i++) sum += L_f2.output[i] * w[i];
        L_f3.preact[o] = sum;
        L_f3.output[o] = sum;
    }
}

static void stage0_thread(Metrics* m, double data[227][227][3]) {
    set_affinity({0});
    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        g_slots[s].wait_until(ST_EMPTY);
        auto tw1 = Clock::now();
        auto tr0 = Clock::now();
        for (int i = 0; i < 227; i++)
            for (int j = 0; j < 227; j++)
                for (int k = 0; k < 3; k++)
                    g_slots[s].input_buf[i*227*3 + j*3 + k] = data[i][j][k];
        g_slots[s].fid = fid;
        auto tr1 = Clock::now();
        m->s0_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s0_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        g_slots[s].set_state(ST_INPUT_READY);
    }
}

static void stage1_thread(Metrics* m) {
    set_affinity({1});

    // NEW: 순수 GPU conv 시간 측정용 cudaEvent 한 쌍 (재사용)
    cudaEvent_t ev_conv_s, ev_conv_e;
    cudaEventCreate(&ev_conv_s);
    cudaEventCreate(&ev_conv_e);

    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        g_slots[s].wait_until(ST_INPUT_READY);
        auto tw1 = Clock::now();
        auto tr0 = Clock::now();
        memcpy(Ainput_a, g_slots[s].input_buf, INPUT_SIZE*sizeof(double));

        // NEW: conv 시작/끝 event로 GPU 순수 시간 측정
        cudaEventRecord(ev_conv_s, g_streams[s]);
        run_conv_only_gpu(g_streams[s]);
        cudaEventRecord(ev_conv_e, g_streams[s]);
        cudaEventSynchronize(ev_conv_e);
        float conv_ms = 0;
        cudaEventElapsedTime(&conv_ms, ev_conv_s, ev_conv_e);
        m->s1_pure_conv.push_back(conv_ms);

        memcpy(g_slots[s].pool3_buf, L_p3.output, POOL3_SIZE*sizeof(float));
        auto tr1 = Clock::now();
        m->s1_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s1_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        g_slots[s].set_state(ST_CONV_DONE);
    }

    cudaEventDestroy(ev_conv_s);
    cudaEventDestroy(ev_conv_e);
}

static void stage2_thread(Metrics* m) {
    set_affinity({2,3,4,5});
    int llc_fd = open_llc_miss_fd(2);
    if (llc_fd >= 0) ioctl(llc_fd, PERF_EVENT_IOC_RESET, 0);

    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        g_slots[s].wait_until(ST_CONV_DONE);
        auto tw1 = Clock::now();

        if (llc_fd >= 0) {
            ioctl(llc_fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(llc_fd, PERF_EVENT_IOC_ENABLE, 0);
        }
        auto tr0 = Clock::now();
        run_fc_cpu_omp(g_slots[s].pool3_buf);
        auto tr1 = Clock::now();
        if (llc_fd >= 0) {
            ioctl(llc_fd, PERF_EVENT_IOC_DISABLE, 0);
            m->s2_llc_miss.push_back(perf_read(llc_fd));
        }

        m->s2_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s2_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        if (fid == NUM_FRAMES - 1) m->t_end = Clock::now();
        g_slots[s].set_state(ST_EMPTY);
    }
    if (llc_fd >= 0) close(llc_fd);
}

struct Stat { double mean, p50, p95, p99; };
static Stat calc_stat(vector<double> v) {
    if (v.empty()) return {0,0,0,0};
    sort(v.begin(), v.end());
    double sum = 0; for (auto x : v) sum += x;
    int n = v.size();
    return {sum/n, v[n*50/100], v[n*95/100], v[n*99/100]};
}

static long read_vol_ctxt() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "voluntary_ctxt_switches:", 24) == 0)
            sscanf(line + 24, "%ld", &v);
    fclose(f);
    return v;
}
static long read_migrations() {
    FILE* f = fopen("/proc/self/sched", "r");
    if (!f) return -1;
    char line[256]; long m = -1;
    while (fgets(line, sizeof(line), f))
        if (strstr(line, "nr_migrations") == line) {
            char* colon = strchr(line, ':');
            if (colon) sscanf(colon+1, "%ld", &m);
        }
    fclose(f);
    return m;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (argv[i][0] == '-' && i+1 < argc) {
            char f = argv[i][1]; i++;
            if (f == 'f') NUM_FRAMES = atoi(argv[i]);
            if (f == 'w') WARMUP     = atoi(argv[i]);
        }

    offset = 1.0;
    cudaStream_t stream1; cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &Ainput_a,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_o,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap1_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_o,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap2_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap3_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_z,      0, cudaMemAttachHost);

    init_cuda_slots();

    double test_data[227][227][3] = {};
    for (int i = 0; i < 227; i++)
        for (int j = 0; j < 227; j++)
            for (int k = 0; k < 3; k++)
                test_data[j][k][i] = (double)PIXELS[j][i][k];

    for (int i = 0; i < WARMUP; i++) {
        memcpy(Ainput_a, test_data, INPUT_SIZE*sizeof(double));
        run_conv_only_gpu(stream1);
        cudaDeviceSynchronize();
        run_fc_cpu_omp(L_p3.output);
    }

    slot_init();
    Metrics m;
    long vol0 = read_vol_ctxt();
    long mig0 = read_migrations();

    thread t0(stage0_thread, &m, test_data);
    thread t1(stage1_thread, &m);
    thread t2(stage2_thread, &m);

    while (m.ready.load() < 3) this_thread::yield();
    m.t_start = Clock::now();

    t0.join(); t1.join(); t2.join();
    long vol1 = read_vol_ctxt();
    long mig1 = read_migrations();

    auto trim = [&](vector<double>& v) {
        int w = min((int)v.size(), WARMUP);
        v.erase(v.begin(), v.begin() + w);
    };
    auto trim_i64 = [&](vector<int64_t>& v) {
        int w = min((int)v.size(), WARMUP);
        v.erase(v.begin(), v.begin() + w);
    };
    trim(m.s0_wait); trim(m.s0_run);
    trim(m.s1_wait); trim(m.s1_run);
    trim(m.s1_pure_conv);
    trim(m.s2_wait); trim(m.s2_run);
    trim_i64(m.s2_llc_miss);

    Stat s0w = calc_stat(m.s0_wait), s0r = calc_stat(m.s0_run);
    Stat s1w = calc_stat(m.s1_wait), s1r = calc_stat(m.s1_run);
    Stat s1c = calc_stat(m.s1_pure_conv);    // NEW: 순수 GPU conv 시간
    Stat s2w = calc_stat(m.s2_wait), s2r = calc_stat(m.s2_run);

    double bn = max({s0r.mean, s1r.mean, s2r.mean});
    double th_fps = 1000.0 / bn;
    double pipe_s = chrono::duration<double>(m.t_end - m.t_start).count();
    double pipe_fps = NUM_FRAMES / pipe_s;

    double llc_mean = 0;
    if (!m.s2_llc_miss.empty()) {
        for (auto x : m.s2_llc_miss) llc_mean += x;
        llc_mean /= m.s2_llc_miss.size();
    }

    // NEW: 출력 정확성 통제 변수 — L_f3.output 합 (모든 단계에서 동일해야 함)
    double output_checksum = 0;
    for (int i = 0; i < L_f3.O; i++) output_checksum += (double)L_f3.output[i];

    // NEW: 오버헤드 분해 — "유용 계산 시간" vs "오버헤드"
    // 한 프레임 사이클 = max(s0_run, s1_run, s2_run). 이 안에서 wait+sync 등이 오버헤드
    double cycle_ms     = max({s0r.mean, s1r.mean, s2r.mean});
    double useful_ms    = s1c.mean + s2r.mean;       // 순수 GPU conv + 순수 CPU FC
    double overhead_ms  = cycle_ms - useful_ms;
    if (overhead_ms < 0) overhead_ms = 0;            // 음수 방지 (병렬화로 useful이 cycle보다 큰 경우)
    double overhead_pct = overhead_ms / cycle_ms * 100.0;

    printf("\n=== B4: + Private Futex (FINAL) ===\n");
    printf("S0(core{0})    | wait=%.3fms p95=%.3f | run=%.3fms\n", s0w.mean, s0w.p95, s0r.mean);
    printf("S1(core{1})    | wait=%.3fms p95=%.3f | run=%.3fms\n", s1w.mean, s1w.p95, s1r.mean);
    printf("S2(core{2-5})  | wait=%.3fms p95=%.3f | run=%.3fms\n", s2w.mean, s2w.p95, s2r.mean);
    printf("Theory FPS   : %.2f  Pipeline FPS: %.2f  Eff: %.2f%%\n",
           th_fps, pipe_fps, pipe_fps/th_fps*100);

    printf("---- 오버헤드 분해 ----\n");
    printf(" Pure_GPU_conv_ms (S1) : %.3f (p95=%.3f)\n", s1c.mean, s1c.p95);
    printf(" Pure_CPU_FC_ms   (S2) : %.3f (p95=%.3f)\n", s2r.mean, s2r.p95);
    printf(" Cycle_ms              : %.3f\n", cycle_ms);
    printf(" Useful_ms             : %.3f\n", useful_ms);
    printf(" Overhead_ms           : %.3f\n", overhead_ms);
    printf(" Overhead_pct          : %.2f%%\n", overhead_pct);

    printf("---- 통제 변수 ----\n");
    printf(" Output_checksum: %.6e\n", output_checksum);

    printf("---- 5종 증거 지표 ----\n");
    printf(" Voluntary ctxt switches: %ld\n", vol1 - vol0);
    printf(" Thread migrations      : %ld\n", mig1 - mig0);
    if (llc_mean > 0)
        printf(" LLC miss per frame (S2): %.0f\n", llc_mean);

    // === RSS (peak working set) — 다중 인스턴스 메모리 비교용 ===
    {
        FILE* f = fopen("/proc/self/status", "r");
        long vmrss = -1, vmhwm = -1;
        if (f) {
            char ln[256];
            while (fgets(ln, sizeof(ln), f)) {
                if (!strncmp(ln, "VmRSS:", 6)) sscanf(ln, "VmRSS: %ld kB", &vmrss);
                if (!strncmp(ln, "VmHWM:", 6)) sscanf(ln, "VmHWM: %ld kB", &vmhwm);
            }
            fclose(f);
        }
        printf("VmRSS_kB: %ld\n", vmrss);
        printf("VmHWM_kB: %ld\n", vmhwm);
    }
    printf("===========================================================\n");

    destroy_cuda_slots();
    cudaStreamDestroy(stream1);
    return 0;
}

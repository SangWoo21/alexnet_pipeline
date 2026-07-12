
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "../include/mnist.h"
#include "../src/layer.cu"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <initializer_list>
#include <linux/futex.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;

static constexpr int N_SLOTS      = 6;
static constexpr int INPUT_SIZE   = 28 * 28;
static constexpr int FC_INPUT_SIZE = 6 * 6 * 6;
static constexpr int FC_OUTPUT_SIZE = 10;

static int NUM_FRAMES = 3000;
static int WARMUP     = 100;

static mnist_data *train_set, *test_set;
static unsigned int Rtrain_cnt, Rtest_cnt;

double iniStart = gettime();
static RLayer l_input = RLayer(0, 0, 28 * 28, "input");
static RLayer l_c1 = RLayer(5 * 5, 6, 24 * 24 * 6, "c1");
static RLayer l_c2 = RLayer(2 * 2, 6, 12 * 12 * 6, "c2");
static RLayer l_c3 = RLayer(2 * 2, 6, 6 * 6 * 6, "c3");
static RLayer l_f = RLayer(6 * 6 * 6, 10, 10, "f");
static RLayer l_r = RLayer(4 * 4, 1, 6 * 6 * 6, "r");
double iniEnd = gettime();

static inline void loaddata() {
    mnist_load("../data/mnist/train-images.idx3-ubyte",
               "../data/mnist/train-labels.idx1-ubyte",
               &train_set, &Rtrain_cnt);
    mnist_load("../data/mnist/t10k-images.idx3-ubyte",
               "../data/mnist/t10k-labels.idx1-ubyte",
               &test_set, &Rtest_cnt);
}

inline void get_cuda_size(const int N, int &grid, int &block) {
    int i = -1;
    int temp = N;
    while (temp) { temp >>= 1; i++; }
    block = 1 << int(i / 2);
    grid = ceil(1.0 * N / block);
}

static inline int futex_wait(atomic<int>* addr, int expected) {
    return syscall(SYS_futex, reinterpret_cast<int*>(addr),
                   FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected, nullptr, nullptr, 0);
}
static inline int futex_wake(atomic<int>* addr, int n) {
    return syscall(SYS_futex, reinterpret_cast<int*>(addr),
                   FUTEX_WAKE | FUTEX_PRIVATE_FLAG, n, nullptr, nullptr, 0);
}

static void set_affinity(initializer_list<int> cores) {
    cpu_set_t cs; CPU_ZERO(&cs);
    for (int c : cores) CPU_SET(c, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

enum SlotState { EMPTY = 0, INPUT_READY = 1, CONV_DONE = 2 };

struct Slot {
    atomic<int> state;
    int         fid;
    double      input_buf[INPUT_SIZE];
    float       sigmoid_out[FC_INPUT_SIZE];
};
static Slot g_slots[N_SLOTS];

static void slot_init() {
    for (int s = 0; s < N_SLOTS; s++) {
        g_slots[s].state.store(EMPTY, memory_order_release);
        memset(g_slots[s].input_buf, 0, sizeof(g_slots[s].input_buf));
        memset(g_slots[s].sigmoid_out, 0, sizeof(g_slots[s].sigmoid_out));
    }
}
static void slot_wait_until(Slot& s, int target) {
    for (;;) {
        int cur = s.state.load(memory_order_acquire);
        if (cur == target) return;
        futex_wait(&s.state, cur);
    }
}
static void slot_set_state(Slot& s, int ns) {
    s.state.store(ns, memory_order_release);
    futex_wake(&s.state, INT_MAX);
}

struct Metrics {
    vector<double> s0_wait, s0_run;
    vector<double> s1_wait, s1_run;
    vector<double> s1_pure_gpu;
    vector<double> s2_wait, s2_run;
    chrono::time_point<Clock> t_start, t_end;
    atomic<int> ready{0};
    Metrics() {
        s0_wait.reserve(NUM_FRAMES); s0_run.reserve(NUM_FRAMES);
        s1_wait.reserve(NUM_FRAMES); s1_run.reserve(NUM_FRAMES);
        s1_pure_gpu.reserve(NUM_FRAMES);
        s2_wait.reserve(NUM_FRAMES); s2_run.reserve(NUM_FRAMES);
    }
};

static void run_gpu_only(cudaStream_t stream1) {
    offset = 1.0;

    int c1_grid, c1_block;
    get_cuda_size(int(6 * 24 * 24), c1_grid, c1_block);
    fp_preact_c1<<<c1_grid, c1_block, 0, stream1>>>(
        (float(*)[28])l_input.output,
        (float(*)[24][24])l_c1.output,
        (float(*)[5][5])l_c1.weight, l_c1.bias, offset);

    int r_grid, r_block;
    get_cuda_size(int(6 * 6 * 6), r_grid, r_block);
    fp_preact_r<<<r_grid, r_block, 0, stream1>>>(
        (float(*)[24][24])l_c1.output,
        (float(*)[6][6])l_r.preact,
        (float(*)[4][4])l_r.weight, *l_r.bias, offset);

    int c2_grid, c2_block;
    get_cuda_size(int(6 * 12 * 12), c2_grid, c2_block);
    fp_preact_c2<<<c2_grid, c2_block, 0, stream1>>>(
        (float(*)[24][24])l_c1.output,
        (float(*)[12][12])l_c2.output,
        (float(*)[2][2])l_c2.weight, l_c2.bias, offset);

    int c3_grid, c3_block;
    get_cuda_size(int(6 * 6 * 6), c3_grid, c3_block);
    fp_preact_c3<<<c3_grid, c3_block, 0, stream1>>>(
        (float(*)[12][12])l_c2.output,
        (float(*)[6][6])l_c3.preact,
        (float(*)[2][2])l_c3.weight, l_c3.bias, offset);

    int add_grid, add_block;
    get_cuda_size(6 * 6 * 6, add_grid, add_block);
    fp_add_res<<<add_grid, add_block, 0, stream1>>>(
        (float(*)[6][6])l_c3.preact,
        (float(*)[6][6])l_r.preact);

    apply_sigmoid<<<128, 128, 0, stream1>>>(l_c3.preact, l_c3.output, l_c3.O);
}

static void run_fc_cpu_omp(float* fc_input) {
    memcpy(l_c3.output, fc_input, FC_INPUT_SIZE * sizeof(float));
    omp_set_num_threads(4);

    #pragma omp parallel for schedule(static)
    for (int o = 0; o < l_f.O; o++) {
        float sum = l_f.bias[o];
        const float* w = &((float(*)[6*6*6])l_f.weight)[o][0];
        const float* p = (float*)l_c3.output;
        for (int i = 0; i < FC_INPUT_SIZE; i++) sum += p[i] * w[i];
        l_f.preact[o] = sum;
        l_f.output[o] = sum;
    }
}

static void stage0_thread(Metrics* m) {
    set_affinity({0});
    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        slot_wait_until(g_slots[s], EMPTY);
        auto tw1 = Clock::now();
        auto tr0 = Clock::now();

        auto& img = test_set[fid % Rtest_cnt].data;
        for (int i = 0; i < 28; i++)
            for (int j = 0; j < 28; j++)
                g_slots[s].input_buf[i * 28 + j] = img[i][j];
        g_slots[s].fid = fid;
        auto tr1 = Clock::now();
        m->s0_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s0_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        slot_set_state(g_slots[s], INPUT_READY);
    }
}

static void stage1_thread(Metrics* m, cudaStream_t stream1) {
    set_affinity({1});

    cudaEvent_t ev_s, ev_e;
    cudaEventCreate(&ev_s);
    cudaEventCreate(&ev_e);

    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        slot_wait_until(g_slots[s], INPUT_READY);
        auto tw1 = Clock::now();
        auto tr0 = Clock::now();

        memcpy(Rinput_a, g_slots[s].input_buf, INPUT_SIZE * sizeof(double));

        cudaEventRecord(ev_s, stream1);
        run_gpu_only(stream1);
        cudaEventRecord(ev_e, stream1);
        cudaEventSynchronize(ev_e);
        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, ev_s, ev_e);
        m->s1_pure_gpu.push_back(gpu_ms);

        memcpy(g_slots[s].sigmoid_out, l_c3.output, FC_INPUT_SIZE * sizeof(float));

        auto tr1 = Clock::now();
        m->s1_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s1_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        slot_set_state(g_slots[s], CONV_DONE);
    }

    cudaEventDestroy(ev_s);
    cudaEventDestroy(ev_e);
}

static void stage2_thread(Metrics* m) {
    set_affinity({2,3,4,5});
    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        slot_wait_until(g_slots[s], CONV_DONE);
        auto tw1 = Clock::now();
        auto tr0 = Clock::now();
        run_fc_cpu_omp(g_slots[s].sigmoid_out);
        auto tr1 = Clock::now();
        m->s2_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s2_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        if (fid == NUM_FRAMES - 1) m->t_end = Clock::now();
        slot_set_state(g_slots[s], EMPTY);
    }
}

struct Stat { double mean, p50, p95, p99; };
static Stat calc_stat(vector<double> v) {
    if (v.empty()) return {0,0,0,0};
    sort(v.begin(), v.end());
    double sum = 0; for (auto x : v) sum += x;
    int n = v.size();
    return {sum/n, v[n*50/100], v[n*95/100], v[n*99/100]};
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (argv[i][0] == '-' && i+1 < argc) {
            char f = argv[i][1]; i++;
            if (f == 'f') NUM_FRAMES = atoi(argv[i]);
            if (f == 'w') WARMUP     = atoi(argv[i]);
        }

    offset = 1.0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &Rinput_a,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc1_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc1_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc1_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc2_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc2_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc2_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc3_bias,   0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc3_a,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rc3_z,      0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rf_weight,  0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rf_bias,    0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rf_a,       0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rf_z,       0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rr_weight,  0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rr_bias,    0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Rr_z,       0, cudaMemAttachHost);

    loaddata();

    for (int i = 0; i < WARMUP; i++) {
        auto& img = test_set[i % Rtest_cnt].data;
        for (int a = 0; a < 28; a++)
            for (int b = 0; b < 28; b++)
                Rinput_a[a * 28 + b] = img[a][b];
        run_gpu_only(stream1);
        cudaDeviceSynchronize();
        run_fc_cpu_omp(l_c3.output);
    }

    slot_init();
    Metrics m;

    thread t0(stage0_thread, &m);
    thread t1(stage1_thread, &m, stream1);
    thread t2(stage2_thread, &m);

    while (m.ready.load() < 3) this_thread::yield();
    m.t_start = Clock::now();

    t0.join(); t1.join(); t2.join();

    auto trim = [&](vector<double>& v) {
        int w = min((int)v.size(), WARMUP);
        v.erase(v.begin(), v.begin() + w);
    };
    trim(m.s0_wait); trim(m.s0_run);
    trim(m.s1_wait); trim(m.s1_run); trim(m.s1_pure_gpu);
    trim(m.s2_wait); trim(m.s2_run);

    Stat s0w = calc_stat(m.s0_wait), s0r = calc_stat(m.s0_run);
    Stat s1w = calc_stat(m.s1_wait), s1r = calc_stat(m.s1_run);
    Stat s1g = calc_stat(m.s1_pure_gpu);
    Stat s2w = calc_stat(m.s2_wait), s2r = calc_stat(m.s2_run);

    double bn = max({s0r.mean, s1r.mean, s2r.mean});
    double th_fps = 1000.0 / bn;
    double pipe_s = chrono::duration<double>(m.t_end - m.t_start).count();
    double pipe_fps = NUM_FRAMES / pipe_s;

    printf("\n=== ResNet Futex 3-Stage Pipeline ===\n");
    printf("S0(core{0})    | wait=%.3fms p95=%.3f | run=%.3fms\n",
           s0w.mean, s0w.p95, s0r.mean);
    printf("S1(core{1})    | wait=%.3fms p95=%.3f | run=%.3fms\n",
           s1w.mean, s1w.p95, s1r.mean);
    printf("S2(core{2-5})  | wait=%.3fms p95=%.3f | run=%.3fms (OpenMP 4)\n",
           s2w.mean, s2w.p95, s2r.mean);
    printf("Theory FPS   : %.2f  Pipeline FPS: %.2f  Eff: %.2f%%\n",
           th_fps, pipe_fps, pipe_fps/th_fps*100);

    printf("---- Overhead ----\n");
    printf(" Pure_GPU_ms  (S1) : %.3f (p95=%.3f)\n", s1g.mean, s1g.p95);
    printf(" Pure_FC_ms   (S2) : %.3f (p95=%.3f)\n", s2r.mean, s2r.p95);
    printf(" Cycle_ms          : %.3f\n", bn);
    double useful = s1g.mean + s2r.mean;
    double ovh = bn - useful;
    if (ovh < 0) ovh = 0;
    printf(" Useful_ms         : %.3f\n", useful);
    printf(" Overhead_ms       : %.3f\n", ovh);
    printf(" Overhead_pct      : %.2f%%\n", ovh / bn * 100.0);
    printf("========================================\n");

    cudaStreamDestroy(stream1);
    return 0;
}


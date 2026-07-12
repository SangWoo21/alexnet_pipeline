// ============================================================
// VGG-16 futex-based 3-stage pipeline
//   S0 (CPU, core {0})      : input memcpy → slot
//   S1 (GPU launch, core {1}): conv1_1 .. pool5 (13 conv + 5 pool, GPU)
//   S2 (CPU, core {2-5})    : FC1→FC2→FC3→softmax (inline OMP 4)
//
//   동기화: Linux futex (FUTEX_PRIVATE_FLAG), 멀티버퍼 N_SLOTS = 6
//   가중치: 랜덤 초기화 (seed 42) — baseline 과 동일
//   지표: stage wait/run(mean/p95), Theory/Pipeline FPS, Eff, Pure GPU
//
// build:
// nvcc -gencode arch=compute_72,code=sm_72 -O3 -std=c++17 --use_fast_math -Xcompiler "-pthread -Wall -O3 -fopenmp" -o vgg_futex vgg_futex.cu -lcudart -lpthread -lrt -lgomp -lcuda
// run:
// sudo jetson_clocks && ./vgg_futex -f 3000 -w 100
// ============================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
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

#define Mask_width 3
#define Mask_height 3
#define Mask_radius_x Mask_width / 2
#define Mask_radius_y Mask_height / 2
#define TILE_WIDTH 32
#define B_x (TILE_WIDTH + Mask_width - 1)
#define B_y (TILE_WIDTH + Mask_height - 1)
#define clamp(x) (max(max((x), 0.0), x))
#define SIZE 224
#define max4(w, x, y, z) max(max(max(w, x), y), z)
#define INPUT_CHANNELS 3
#define CONV_SIZE 3

int layers[13][4] = {
    {64, 3, CONV_SIZE, CONV_SIZE},   {64, 64, CONV_SIZE, CONV_SIZE},
    {128, 64, CONV_SIZE, CONV_SIZE}, {128, 128, CONV_SIZE, CONV_SIZE},
    {256, 128, CONV_SIZE, CONV_SIZE},{256, 256, CONV_SIZE, CONV_SIZE},
    {256, 256, CONV_SIZE, CONV_SIZE},{512, 256, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE},{512, 512, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE},{512, 512, CONV_SIZE, CONV_SIZE},
    {512, 512, CONV_SIZE, CONV_SIZE}};

int dense[3][2] = {{25088, 4096}, {4096, 4096}, {4096, 1000}};

typedef enum { CONV_1 = 512 } ch;
const int out = ch(CONV_1);

static constexpr int N_SLOTS       = 6;
static constexpr int INPUT_SIZE    = SIZE * SIZE * 3;   // 224*224*3
static constexpr int POOL5_SIZE    = 7 * 7 * 512;       // 25088
static int NUM_FRAMES = 3000;
static int WARMUP     = 100;

// ============================================================
// 원본 VGG.cu GPU 커널 (수정 없이 그대로)
// ============================================================
__global__ void convolution(float *I, const float *__restrict__ M, float *P, float *b, int channels, int width,
                            int height, int numberofOutputChannels) {
    __shared__ float N_ds[B_y][B_x];
    int dest_Y, dest_X, src_X, src_Y, src, dest;
    float accum[out] = {0};

    for (int current_channel = 0; current_channel < channels; current_channel++) {
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x, dest_Y = dest / B_x, dest_X = dest % B_x,
        src_Y = blockIdx.y * TILE_WIDTH + dest_Y - Mask_radius_x,
        src_X = blockIdx.x * TILE_WIDTH + dest_X - Mask_radius_y,
        src = (src_Y * width + src_X) * channels + current_channel;
        if (src_Y >= 0 && src_Y < height && src_X >= 0 && src_X < width)
            N_ds[dest_Y][dest_X] = I[src];
        else
            N_ds[dest_Y][dest_X] = 0.0;

        for (int iter = 1; iter <= (B_x * B_y) / (TILE_WIDTH * TILE_WIDTH); iter++) {
            dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter * (TILE_WIDTH * TILE_WIDTH);
            dest_Y = dest / B_x, dest_X = dest % B_x;
            src_Y = blockIdx.y * TILE_WIDTH + dest_Y - Mask_radius_x;
            src_X = blockIdx.x * TILE_WIDTH + dest_X - Mask_radius_y;
            src = (src_Y * width + src_X) * channels + current_channel;
            if (dest_Y < B_y && dest_X < B_x) {
                if (src_Y >= 0 && src_Y < height && src_X >= 0 && src_X < width)
                    N_ds[dest_Y][dest_X] = I[src];
                else
                    N_ds[dest_Y][dest_X] = 0.0;
            }
        }
        __syncthreads();

        int y, x, z;
        for (z = 0; z < numberofOutputChannels; z++)
            for (y = 0; y < Mask_width; y++)
                for (x = 0; x < Mask_width; x++)
                    accum[z] += N_ds[threadIdx.y + y][threadIdx.x + x] *
                                M[(z * Mask_width * Mask_width * channels + current_channel * Mask_width * Mask_width) +
                                  y * Mask_width + x];
        __syncthreads();
    }

    int y, x, z;
    y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width)
        for (z = 0; z < numberofOutputChannels; z++)
            P[(y * width * numberofOutputChannels + numberofOutputChannels * x) + z] = clamp(accum[z] + b[z]);
}

__global__ void maxpool(float *image, float *output, int number_of_channels, int image_height, int image_width,
                        int blockwidth) {
    __shared__ float Ns[32][32];
    for (int curr_channel = 0; curr_channel < number_of_channels; curr_channel++) {
        Ns[threadIdx.x][threadIdx.y] =
            image[(threadIdx.y * number_of_channels + curr_channel + blockIdx.y * (blockwidth * number_of_channels)) +
                  (threadIdx.x + blockIdx.x * blockwidth) * (image_width * number_of_channels)];
        __syncthreads();
        if ((threadIdx.x % 2 == 0) && (threadIdx.y % 2 == 0)) {
            output[blockIdx.y * (blockwidth / 2) * number_of_channels + (threadIdx.y / 2) * number_of_channels +
                   curr_channel +
                   (blockIdx.x * blockwidth / 2 + threadIdx.x / 2) * (image_width / 2) * number_of_channels] =
                max4(Ns[threadIdx.x][threadIdx.y], Ns[threadIdx.x][threadIdx.y + 1], Ns[threadIdx.x + 1][threadIdx.y],
                     Ns[threadIdx.x + 1][threadIdx.y + 1]);
        }
    }
}

// ============================================================
// FC (CPU, OpenMP 4) — baseline 과 동일 구현
// ============================================================
static float *h_dense1, *h_dense2, *h_dense3;
static float *h_bias1, *h_bias2, *h_bias3;
static float *h_fc1o, *h_fc2o, *h_fc3o;

static void fc1_cpu(const float *I, const float *M, float *P, const float *b) {
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < 4096; z++) {
        float acc = 0.f;
        for (int c = 0; c < 512; c++)
            for (int i = 0; i < 7; i++)
                for (int j = 0; j < 7; j++)
                    acc += I[i * (7 * 512) + j * 512 + c] * M[(size_t)z * 25088 + c * 49 + i * 7 + j];
        float v = acc + b[z];
        P[z] = v > 0.f ? v : 0.f;
    }
}
static void fc2_cpu(const float *I, const float *M, float *P, const float *b) {
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < 4096; z++) {
        float acc = 0.f;
        const float *w = M + (size_t)z * 4096;
        for (int c = 0; c < 4096; c++) acc += I[c] * w[c];
        float v = acc + b[z];
        P[z] = v > 0.f ? v : 0.f;
    }
}
static void fc3_cpu(const float *I, const float *M, float *P, const float *b) {
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < 1000; z++) {
        float acc = 0.f;
        const float *w = M + (size_t)z * 4096;
        for (int c = 0; c < 4096; c++) acc += I[c] * w[c];
        float v = acc + b[z];
        P[z] = v > 0.f ? v : 0.f;
    }
}
static void softmax_cpu(float *p, int n) {
    float mx = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mx) mx = p[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { p[i] = expf(p[i] - mx); s += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= s;
}

// ============================================================
// GPU 가중치·버퍼 (1회 할당)
// ============================================================
static float *d_convW[13], *d_convB[13];
static float *d_bufA, *d_bufB, *d_input;
static float *h_input; // 프레임 소스 (S0 가 슬롯에 복사)

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

static void rand_fill(float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
}

static void init_all() {
    srand(42);
    for (int l = 0; l < 13; l++) {
        size_t wsz = (size_t)layers[l][0] * layers[l][1] * CONV_SIZE * CONV_SIZE;
        size_t bsz = (size_t)layers[l][0];
        float *tmpw = (float *)malloc(wsz * sizeof(float));
        float *tmpb = (float *)malloc(bsz * sizeof(float));
        rand_fill(tmpw, wsz);
        rand_fill(tmpb, bsz);
        cudaMalloc((void **)&d_convW[l], wsz * sizeof(float));
        cudaMalloc((void **)&d_convB[l], bsz * sizeof(float));
        cudaMemcpy(d_convW[l], tmpw, wsz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_convB[l], tmpb, bsz * sizeof(float), cudaMemcpyHostToDevice);
        free(tmpw); free(tmpb);
    }
    h_dense1 = (float *)malloc((size_t)25088 * 4096 * sizeof(float));
    h_dense2 = (float *)malloc((size_t)4096 * 4096 * sizeof(float));
    h_dense3 = (float *)malloc((size_t)4096 * 1000 * sizeof(float));
    h_bias1  = (float *)malloc(4096 * sizeof(float));
    h_bias2  = (float *)malloc(4096 * sizeof(float));
    h_bias3  = (float *)malloc(1000 * sizeof(float));
    rand_fill(h_dense1, (size_t)25088 * 4096);
    rand_fill(h_dense2, (size_t)4096 * 4096);
    rand_fill(h_dense3, (size_t)4096 * 1000);
    rand_fill(h_bias1, 4096); rand_fill(h_bias2, 4096); rand_fill(h_bias3, 1000);

    size_t maxbuf = (size_t)SIZE * SIZE * 64;
    cudaMalloc((void **)&d_bufA, maxbuf * sizeof(float));
    cudaMalloc((void **)&d_bufB, maxbuf * sizeof(float));
    cudaMalloc((void **)&d_input, (size_t)INPUT_SIZE * sizeof(float));

    h_input = (float *)malloc((size_t)INPUT_SIZE * sizeof(float));
    FILE *f = fopen("../data/VGG/vol.txt", "r");
    if (f) {
        float coef[3] = {103.939f, 116.779f, 123.68f};
        int count = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            float dval;
            if (fscanf(f, "%f", &dval) != 1) dval = 0.f;
            h_input[i] = dval - coef[count];
            count = (count + 1) % 3;
        }
        fclose(f);
        printf("[init] input: vol.txt loaded\n");
    } else {
        rand_fill(h_input, (size_t)INPUT_SIZE);
        printf("[init] input: random (vol.txt not found)\n");
    }

    h_fc1o = (float *)malloc(4096 * sizeof(float));
    h_fc2o = (float *)malloc(4096 * sizeof(float));
    h_fc3o = (float *)malloc(1000 * sizeof(float));
}

static inline void run_conv(int level, float *in, float *outbuf, int w, int h, cudaStream_t st) {
    dim3 dimGrid(((w - 1) / TILE_WIDTH) + 1, ((h - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid, dimBlock, 0, st>>>(in, d_convW[level], outbuf, d_convB[level],
                                              layers[level][1], w, h, layers[level][0]);
}
static inline void run_pool(float *in, float *outbuf, int channels, int w, int h, cudaStream_t st) {
    int blockwidth = 2;
    int number_blocks = w / blockwidth;
    dim3 dimGrid(number_blocks, number_blocks, 1);
    dim3 dimBlock(blockwidth, blockwidth, 1);
    maxpool<<<dimGrid, dimBlock, 0, st>>>(in, outbuf, channels, h, w, blockwidth);
}

// GPU 전체 conv/pool 체인 (VGG-16, 원본 흐름)
static void run_gpu_chain(cudaStream_t st) {
    int w = SIZE, h = SIZE;
    run_conv(0, d_input, d_bufA, w, h, st);
    run_conv(1, d_bufA, d_bufB, w, h, st);
    run_pool(d_bufB, d_bufA, 64, w, h, st);
    w /= 2; h /= 2;
    run_conv(2, d_bufA, d_bufB, w, h, st);
    run_conv(3, d_bufB, d_bufA, w, h, st);
    run_pool(d_bufA, d_bufB, 128, w, h, st);
    w /= 2; h /= 2;
    run_conv(4, d_bufB, d_bufA, w, h, st);
    run_conv(5, d_bufA, d_bufB, w, h, st);
    run_conv(6, d_bufB, d_bufA, w, h, st);
    run_pool(d_bufA, d_bufB, 256, w, h, st);
    w /= 2; h /= 2;
    run_conv(7, d_bufB, d_bufA, w, h, st);
    run_conv(8, d_bufA, d_bufB, w, h, st);
    run_conv(9, d_bufB, d_bufA, w, h, st);
    run_pool(d_bufA, d_bufB, 512, w, h, st);
    w /= 2; h /= 2;
    run_conv(10, d_bufB, d_bufA, w, h, st);
    run_conv(11, d_bufA, d_bufB, w, h, st);
    run_conv(12, d_bufB, d_bufA, w, h, st);
    run_pool(d_bufA, d_bufB, 512, w, h, st);
    // 결과: d_bufB 에 7*7*512
}

// ============================================================
// futex wrappers + slot
// ============================================================
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
    int fid;
    float *input_buf;   // 224*224*3 (host, 동적 할당)
    float *pool5_buf;   // 7*7*512
};
static Slot g_slots[N_SLOTS];

static void slot_init() {
    for (int s = 0; s < N_SLOTS; s++) {
        g_slots[s].state.store(EMPTY, memory_order_release);
        g_slots[s].input_buf = (float *)malloc((size_t)INPUT_SIZE * sizeof(float));
        g_slots[s].pool5_buf = (float *)malloc((size_t)POOL5_SIZE * sizeof(float));
        memset(g_slots[s].input_buf, 0, (size_t)INPUT_SIZE * sizeof(float));
        memset(g_slots[s].pool5_buf, 0, (size_t)POOL5_SIZE * sizeof(float));
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
    vector<double> s1_wait, s1_run, s1_pure_gpu;
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

// ============================================================
// Stage threads
// ============================================================
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
        memcpy(g_slots[s].input_buf, h_input, (size_t)INPUT_SIZE * sizeof(float));
        g_slots[s].fid = fid;
        auto tr1 = Clock::now();
        m->s0_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s0_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        slot_set_state(g_slots[s], INPUT_READY);
    }
}

static void stage1_thread(Metrics* m, cudaStream_t st) {
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

        // slot input → device
        cudaMemcpyAsync(d_input, g_slots[s].input_buf, (size_t)INPUT_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice, st);

        cudaEventRecord(ev_s, st);
        run_gpu_chain(st);
        cudaEventRecord(ev_e, st);
        cudaEventSynchronize(ev_e);
        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, ev_s, ev_e);
        m->s1_pure_gpu.push_back(gpu_ms);

        // pool5 → slot (host)
        cudaMemcpyAsync(g_slots[s].pool5_buf, d_bufB, (size_t)POOL5_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);

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
    omp_set_num_threads(4);
    m->ready.fetch_add(1);
    while (m->ready.load() < 3) this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; fid++) {
        int s = fid % N_SLOTS;
        auto tw0 = Clock::now();
        slot_wait_until(g_slots[s], CONV_DONE);
        auto tw1 = Clock::now();
        auto tr0 = Clock::now();
        fc1_cpu(g_slots[s].pool5_buf, h_dense1, h_fc1o, h_bias1);
        fc2_cpu(h_fc1o, h_dense2, h_fc2o, h_bias2);
        fc3_cpu(h_fc2o, h_dense3, h_fc3o, h_bias3);
        softmax_cpu(h_fc3o, 1000);
        auto tr1 = Clock::now();
        m->s2_wait.push_back(chrono::duration<double,milli>(tw1-tw0).count());
        m->s2_run .push_back(chrono::duration<double,milli>(tr1-tr0).count());
        if (fid == NUM_FRAMES - 1) m->t_end = Clock::now();
        slot_set_state(g_slots[s], EMPTY);
    }
}

// ============================================================
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

    cuInit(0);
    cudaDeviceReset();

    cudaStream_t st;
    cudaStreamCreate(&st);

    printf("[init] allocating & random-initializing weights...\n");
    double t0 = gettime();
    init_all();
    printf("[init] done in %.2f s\n", gettime() - t0);

    // Warmup (순차 실행으로 GPU/캐시 예열)
    omp_set_num_threads(4);
    for (int i = 0; i < WARMUP; i++) {
        cudaMemcpy(d_input, h_input, (size_t)INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        run_gpu_chain(st);
        cudaStreamSynchronize(st);
        static float tmp_pool5[POOL5_SIZE];
        cudaMemcpy(tmp_pool5, d_bufB, (size_t)POOL5_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        fc1_cpu(tmp_pool5, h_dense1, h_fc1o, h_bias1);
        fc2_cpu(h_fc1o, h_dense2, h_fc2o, h_bias2);
        fc3_cpu(h_fc2o, h_dense3, h_fc3o, h_bias3);
        softmax_cpu(h_fc3o, 1000);
    }

    slot_init();
    Metrics m;

    thread t0h(stage0_thread, &m);
    thread t1h(stage1_thread, &m, st);
    thread t2h(stage2_thread, &m);

    while (m.ready.load() < 3) this_thread::yield();
    m.t_start = Clock::now();

    t0h.join(); t1h.join(); t2h.join();

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

    printf("\n=== VGG-16 Futex 3-Stage Pipeline ===\n");
    printf("S0(core{0})    | wait=%.3fms p95=%.3f | run=%.3fms\n", s0w.mean, s0w.p95, s0r.mean);
    printf("S1(core{1})    | wait=%.3fms p95=%.3f | run=%.3fms\n", s1w.mean, s1w.p95, s1r.mean);
    printf("S2(core{2-5})  | wait=%.3fms p95=%.3f | run=%.3fms (OpenMP 4)\n", s2w.mean, s2w.p95, s2r.mean);
    printf("Theory FPS   : %.2f  Pipeline FPS: %.2f  Eff: %.2f%%\n", th_fps, pipe_fps, pipe_fps/th_fps*100);
    printf("---- Breakdown ----\n");
    printf(" Pure_GPU_ms  (S1) : %.3f (p95=%.3f)\n", s1g.mean, s1g.p95);
    printf(" Pure_FC_ms   (S2) : %.3f (p95=%.3f)\n", s2r.mean, s2r.p95);
    printf(" Cycle_ms          : %.3f\n", bn);
    printf(" Cycle_p95_ms      : %.3f\n", max({s0r.p95, s1r.p95, s2r.p95}));
    printf("========================================\n");

    cudaStreamDestroy(st);
    return 0;
}

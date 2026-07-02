#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "../src/layer.cu"
#include "../include/mnist.h"
#include "../include/pixels.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

double iniStart_p = gettime();
ALayer L_input = ALayer(0, 0, 227 * 227 * 3, (char*)"input");
ALayer L_c1 = ALayer(11 * 11 * 3, 2 * 48, 2 * 55 * 55 * 48, (char*)"c1");
ALayer L_p1 = ALayer(3 * 3, 2 * 1, 2 * 31 * 31 * 48, (char*)"p1");
ALayer L_c2 = ALayer(5 * 5 * 48, 2 * 128, 2 * 128 * 27 * 27, (char*)"c2");
ALayer L_p2 = ALayer(3 * 3, 2 * 1, 2 * 15 * 15 * 128, (char*)"p2");
ALayer L_c3 = ALayer(3 * 3 * 256, 384, 2 * 13 * 13 * 192, (char*)"c3");
ALayer L_c4 = ALayer(3 * 3 * 384, 2 * 192, 2 * 13 * 13 * 192, (char*)"c4");
ALayer L_c5 = ALayer(3 * 3 * 384, 2 * 128, 2 * 13 * 13 * 128, (char*)"c5");
ALayer L_p3 = ALayer(3 * 3, 2 * 1, 2 * 6 * 6 * 128, (char*)"p3");
ALayer L_f1 = ALayer(6 * 6 * 256, 2 * 2048, 4096 * 1, (char*)"f1");
ALayer L_f2 = ALayer(1 * 4096, 2 * 2048, 4096 * 1, (char*)"f2");
ALayer L_f3 = ALayer(1 * 4096, 1000, 1000, (char*)"f3");
double iniEnd_p = gettime();

using TP = std::chrono::time_point<std::chrono::steady_clock>;
static inline double elapsed_ms_p(TP a, TP b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static constexpr int N_SLOTS = 6;
static constexpr int QUEUE_CAP = 8;
static int NUM_FRAMES = 3000;
static constexpr int WARMUP = 5;

static inline long sys_futex(void* uaddr, int op, int val,
                             const struct timespec* to = nullptr) {
    return syscall(SYS_futex, uaddr, op, val, to, nullptr, 0);
}
static inline void futex_wait_val(std::atomic<int>* addr, int expected) {
    sys_futex(reinterpret_cast<int*>(addr), FUTEX_WAIT | FUTEX_PRIVATE_FLAG, expected);
}
static inline void futex_wake_one(std::atomic<int>* addr) {
    sys_futex(reinterpret_cast<int*>(addr), FUTEX_WAKE | FUTEX_PRIVATE_FLAG, 1);
}
static void wait_until_ge(std::atomic<int>* addr, int target) {
    while (true) {
        int cur = addr->load(std::memory_order_acquire);
        if (cur >= target) return;
        futex_wait_val(addr, cur);
    }
}
static void signal_val(std::atomic<int>* addr, int val) {
    addr->store(val, std::memory_order_release);
    futex_wake_one(addr);
}

struct SlotSync {
    std::atomic<int> ready{1};
    void wait_available() {
        while (true) {
            int cur = ready.load(std::memory_order_acquire);
            if (cur >= 1) return;
            futex_wait_val(&ready, cur);
        }
    }
    void mark_available() {
        ready.store(1, std::memory_order_release);
        futex_wake_one(&ready);
    }
    void mark_in_use() {
        ready.store(0, std::memory_order_release);
    }
};

struct FrameMsg { int fid; int slot; };
struct FrameQueue {
    FrameMsg buf[QUEUE_CAP];
    std::atomic<int> head{0}, tail{0}, count{0};
    void push(FrameMsg msg) {
        int t = tail.load(std::memory_order_relaxed);
        buf[t % QUEUE_CAP] = msg;
        tail.store(t + 1, std::memory_order_release);
        int old = count.fetch_add(1, std::memory_order_acq_rel);
        if (old == 0) futex_wake_one(&count);
    }
    FrameMsg pop() {
        while (true) {
            int cnt = count.load(std::memory_order_acquire);
            if (cnt > 0) break;
            futex_wait_val(&count, 0);
        }
        int h = head.load(std::memory_order_relaxed);
        FrameMsg msg = buf[h % QUEUE_CAP];
        head.store(h + 1, std::memory_order_release);
        count.fetch_sub(1, std::memory_order_acq_rel);
        return msg;
    }
};

static void set_thread_affinity(const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static SlotSync slot_sync[N_SLOTS];
static FrameQueue q01;
static std::atomic<int> s2_signal{0};
static std::atomic<int> s2_slot{0};
static std::atomic<int> init_token{0};
static std::atomic<int> ready_count{0};

struct Stats {
    TP t_wall_start, t_wall_end;
    TP t_pipeline_start, t_pipeline_end;
};
static Stats stats;

struct Stage0WS {
    float* slot[N_SLOTS];
    Stage0WS() {
        size_t inputsz = sizeof(float) * 227 * 227 * 3;
        for (int s = 0; s < N_SLOTS; ++s) {
            cudaMallocManaged((void**)&slot[s], inputsz, cudaMemAttachHost);
            std::memset(slot[s], 0, inputsz);
        }
    }
};

struct Stage1WS {
    float* slot[N_SLOTS];
    cudaStream_t stream;
    Stage1WS() {
        cudaStreamCreate(&stream);
        size_t p3sz = sizeof(float) * 2 * 6 * 6 * 128;
        for (int s = 0; s < N_SLOTS; ++s) {
            cudaMallocManaged((void**)&slot[s], p3sz, cudaMemAttachHost);
            cudaStreamAttachMemAsync(stream, slot[s], 0, cudaMemAttachHost);
            std::memset(slot[s], 0, p3sz);
        }
        cudaStreamSynchronize(stream);
    }
};

struct Stage2WS {
    float* slot[N_SLOTS];
    Stage2WS() {
        for (int s = 0; s < N_SLOTS; ++s)
            slot[s] = new float[1000]();
    }
    ~Stage2WS() {
        for (int s = 0; s < N_SLOTS; ++s) delete[] slot[s];
    }
};

static void stage0_thread(Stage0WS* ws, const float* input_image) {
    set_thread_affinity({0, 1});

    while (init_token.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    init_token.store(1, std::memory_order_release);

    ready_count.fetch_add(1, std::memory_order_acq_rel);
    while (ready_count.load(std::memory_order_acquire) < 3) std::this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; ++fid) {
        int slot = fid % N_SLOTS;
        slot_sync[slot].wait_available();
        slot_sync[slot].mark_in_use();
        if (fid == 0) stats.t_pipeline_start = std::chrono::steady_clock::now();

        std::memcpy(ws->slot[slot], input_image, sizeof(float) * 227 * 227 * 3);

        q01.push({fid, slot});
    }
}

static void stage1_thread(Stage0WS* s0_ws, Stage1WS* ws) {
    set_thread_affinity({3});

    while (init_token.load(std::memory_order_acquire) != 1) std::this_thread::yield();
    cudaSetDevice(0);

    for (int s = 0; s < N_SLOTS; ++s) {
        cudaStreamAttachMemAsync(ws->stream, s0_ws->slot[s], 0, cudaMemAttachHost);
    }
    cudaStreamSynchronize(ws->stream);

    dim3 Bc1(55, 55);
    dim3 Bc2(27, 27), Bc3(13, 13), Bc4(12, 12), Bc5(12, 12);
    dim3 ft_p1(27, 27), ft_p2(13, 13), ft_p3(6, 6);

    for (int i = 0; i < WARMUP; ++i) {
        fp_preact_c1<<<96, Bc1, 0, ws->stream>>>(
            (float(*)[227][3])s0_ws->slot[0],
            (float(*)[55][55])L_c1.preact,
            (float(*)[11][11][3])L_c1.weight, L_c1.bias);
        cudaStreamSynchronize(ws->stream);
    }
    init_token.store(2, std::memory_order_release);

    ready_count.fetch_add(1, std::memory_order_acq_rel);
    while (ready_count.load(std::memory_order_acquire) < 3) std::this_thread::yield();

    for (int processed = 0; processed < NUM_FRAMES; ++processed) {
        FrameMsg msg = q01.pop();
        int slot = msg.slot;

        fp_preact_c1<<<96, Bc1, 0, ws->stream>>>(
            (float(*)[227][3])s0_ws->slot[slot],
            (float(*)[55][55])L_c1.preact,
            (float(*)[11][11][3])L_c1.weight, L_c1.bias);
        Aapply_step_function<<<112, 1280, 0, ws->stream>>>(L_c1.preact, L_c1.act_result, L_c1.O);
        normalization_function<<<112, 640, 0, ws->stream>>>(L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);
        fp_preact_p1<<<96, ft_p1, 0, ws->stream>>>(
            (float(*)[55][55])L_c1.output, (float(*)[31][31])L_p1.output);

        fp_preact_c2<<<256, Bc2, 0, ws->stream>>>(
            (float(*)[31][31])L_p1.output,
            (float(*)[27][27])L_c2.preact,
            (float(*)[96][5][5])L_c2.weight, L_c2.bias);
        Aapply_step_function<<<112, 1280, 0, ws->stream>>>(L_c2.preact, L_c2.act_result, L_c2.O);
        normalization_function<<<112, 1280, 0, ws->stream>>>(L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);
        fp_preact_p2<<<256, ft_p2, 0, ws->stream>>>(
            (float(*)[27][27])L_c2.output, (float(*)[15][15])L_p2.output);

        fp_preact_c3<<<384, Bc3, 0, ws->stream>>>(
            (float(*)[15][15])L_p2.output, (float(*)[13][13])L_c3.preact,
            (float(*)[256][3][3])L_c3.weight, L_c3.bias);
        Aapply_step_function<<<128, 128, 0, ws->stream>>>(L_c3.preact, L_c3.output, L_c3.O);

        fp_preact_c4<<<384, Bc4, 0, ws->stream>>>(
            (float(*)[13][13])L_c3.output, (float(*)[13][13])L_c4.preact,
            (float(*)[384][3][3])L_c4.weight, L_c4.bias);
        Aapply_step_function<<<128, 128, 0, ws->stream>>>(L_c4.preact, L_c4.output, L_c4.O);

        fp_preact_c5<<<256, Bc5, 0, ws->stream>>>(
            (float(*)[13][13])L_c4.output, (float(*)[13][13])L_c5.preact,
            (float(*)[384][3][3])L_c5.weight, L_c5.bias);
        Aapply_step_function<<<128, 128, 0, ws->stream>>>(L_c5.preact, L_c5.output, L_c5.O);

        fp_preact_p3<<<256, ft_p3, 0, ws->stream>>>(
            (float(*)[13][13])L_c5.output, (float(*)[6][6])L_p3.output);

        cudaStreamSynchronize(ws->stream);

        std::memcpy(ws->slot[slot], L_p3.output, sizeof(float) * 2 * 6 * 6 * 128);

        s2_slot.store(slot, std::memory_order_release);
        signal_val(&s2_signal, msg.fid + 1);

        slot_sync[slot].mark_available();
    }
}

static void stage2_thread(Stage1WS* s1_ws, Stage2WS* ws) {
    set_thread_affinity({2, 4, 5});

    while (init_token.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    init_token.store(3, std::memory_order_release);

    ready_count.fetch_add(1, std::memory_order_acq_rel);
    while (ready_count.load(std::memory_order_acquire) < 3) std::this_thread::yield();

    for (int fid = 0; fid < NUM_FRAMES; ++fid) {
        wait_until_ge(&s2_signal, fid + 1);
        int slot = s2_slot.load(std::memory_order_acquire);

        fp_preact_f1_cpu((float(*)[6][6])s1_ws->slot[slot],
                         L_f1.preact, (float(*)[256][6][6])L_f1.weight, L_f1.bias);
        Aapply_step_function_cpu(L_f1.preact, L_f1.output, L_f1.O);

        fp_preact_f2_cpu(L_f1.output, L_f2.preact,
                         (float(*)[4096])L_f2.weight, L_f2.bias);
        Aapply_step_function_cpu(L_f2.preact, L_f2.output, L_f2.O);

        fp_preact_f3_cpu(L_f2.output, L_f3.preact,
                         (float(*)[4096])L_f3.weight, L_f3.bias);
        Aapply_step_function_cpu(L_f3.preact, L_f3.output, L_f3.O);

        std::memcpy(ws->slot[slot], L_f3.output, sizeof(float) * 1000);

        if (fid == NUM_FRAMES - 1) stats.t_pipeline_end = std::chrono::steady_clock::now();
    }
}

int main(int argc, const char** argv) {
    if (argc > 1) NUM_FRAMES = atoi(argv[1]);
    srand((unsigned)time(NULL));

    offset = 0.0;

    cudaStream_t stream_init;
    cudaStreamCreate(&stream_init);
    cudaStreamAttachMemAsync(stream_init, &Ainput_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac1_o, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ap1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac2_o, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ap2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac3_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac4_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac4_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac4_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac4_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac5_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac5_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac5_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ac5_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Ap3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream_init, &Af3_z, 0, cudaMemAttachHost);
    cudaStreamSynchronize(stream_init);

    static float input_image[227 * 227 * 3];
    for (int i = 0; i < 227; ++i)
        for (int j = 0; j < 227; ++j)
            for (int k = 0; k < 3; ++k)
                input_image[i * 227 * 3 + j * 3 + k] = (float)PIXELS[i][j][k];

    auto* s0_ws = new Stage0WS();
    auto* s1_ws = new Stage1WS();
    auto* s2_ws = new Stage2WS();

    std::thread th0(stage0_thread, s0_ws, input_image);
    std::thread th1(stage1_thread, s0_ws, s1_ws);
    std::thread th2(stage2_thread, s1_ws, s2_ws);

    while (ready_count.load(std::memory_order_acquire) < 3) std::this_thread::yield();
    stats.t_wall_start = std::chrono::steady_clock::now();

    th0.join();
    th1.join();
    th2.join();
    stats.t_wall_end = std::chrono::steady_clock::now();

    double wall_ms = elapsed_ms_p(stats.t_wall_start, stats.t_wall_end);
    double pipeline_ms = elapsed_ms_p(stats.t_pipeline_start, stats.t_pipeline_end);
    double wall_fps = NUM_FRAMES / (wall_ms / 1000.0);
    double pipeline_fps = NUM_FRAMES / (pipeline_ms / 1000.0);

    printf("====================================================\n");
    printf(" AlexNet 3-stage pipeline (real layer.cu kernels)\n");
    printf("   N_SLOTS  = %d\n", N_SLOTS);
    printf("   frames   = %d\n", NUM_FRAMES);
    printf("   cores    = S0:{0,1} S1:{3} S2:{2,4,5}\n");
    printf("----------------------------------------------------\n");
    printf("   Wall     elapsed = %8.2f ms  ->  Wall     FPS = %7.2f\n", wall_ms, wall_fps);
    printf("   Pipeline elapsed = %8.2f ms  ->  Pipeline FPS = %7.2f\n", pipeline_ms, pipeline_fps);
    printf("====================================================\n");

    delete s0_ws; delete s1_ws; delete s2_ws;
    return 0;
}

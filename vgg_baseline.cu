// ============================================================
// VGG-16 inter-layer baseline (no pipelining)
//   Conv/Pool = GPU (원본 VGG.cu 커널 그대로), FC = CPU (OMP 4)
//   가중치: 랜덤 초기화 (성능 측정 목적 — 값은 실행 시간과 무관)
//   측정: 3,000 프레임 + warmup 100, FPS / mean / p50 / p95 / p99
//
// build:
// nvcc -gencode arch=compute_72,code=sm_72 -O3 -std=c++17 --use_fast_math -Xcompiler "-pthread -Wall -O3 -fopenmp" -o vgg_baseline vgg_baseline.cu -lcudart -lgomp -lcuda
// ============================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <sys/time.h>
#include <vector>

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

// 원본 VGG.cu 의 레이어 구성 그대로
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

double gettime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

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
// FC (CPU, OpenMP 4 스레드) — inter-layer baseline: FC 전량 CPU
//   가중치 인덱싱은 원본 fully1/2/3 커널과 동일
// ============================================================
static void fc1_cpu(const float *I /*7x7x512 HWC*/, const float *M, float *P, const float *b) {
    // out 4096, in 25088.  M[z*25088 + c*49 + i*7 + j]
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < 4096; z++) {
        float acc = 0.f;
        for (int c = 0; c < 512; c++)
            for (int i = 0; i < 7; i++)
                for (int j = 0; j < 7; j++)
                    acc += I[i * (7 * 512) + j * 512 + c] * M[z * 25088 + c * 49 + i * 7 + j];
        float v = acc + b[z];
        P[z] = v > 0.f ? v : 0.f; // ReLU (원본 clamp 동일 효과)
    }
}

static void fc2_cpu(const float *I, const float *M, float *P, const float *b) {
    // out 4096, in 4096.  M[z*4096 + c]
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
    // out 1000, in 4096
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
// 전역 버퍼 (1회 할당)
// ============================================================
static float *d_convW[13], *d_convB[13];       // conv 가중치/바이어스 (GPU)
static float *h_dense1, *h_dense2, *h_dense3;  // FC 가중치 (CPU host)
static float *h_bias1, *h_bias2, *h_bias3;
static float *d_bufA, *d_bufB;                 // ping-pong 중간 버퍼 (GPU)
static float *d_input;                         // 입력 (GPU)
static float *h_input;                         // 입력 원본 (host)
static float *h_pool5;                         // pool5 출력 (host, FC 입력)
static float *h_fc1o, *h_fc2o, *h_fc3o;        // FC 출력

static void rand_fill(float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
}

static void init_all() {
    srand(42);
    // conv 가중치 (GPU): host 에서 랜덤 생성 후 memcpy
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
    // FC 가중치 (CPU)
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

    // 중간 버퍼: 최대 크기 = 224*224*64 = 3.2M floats
    size_t maxbuf = (size_t)SIZE * SIZE * 64;
    cudaMalloc((void **)&d_bufA, maxbuf * sizeof(float));
    cudaMalloc((void **)&d_bufB, maxbuf * sizeof(float));
    cudaMalloc((void **)&d_input, (size_t)SIZE * SIZE * 3 * sizeof(float));

    h_input = (float *)malloc((size_t)SIZE * SIZE * 3 * sizeof(float));
    // 입력: vol.txt 있으면 로드, 없으면 랜덤
    FILE *f = fopen("../data/VGG/vol.txt", "r");
    if (f) {
        float coef[3] = {103.939f, 116.779f, 123.68f};
        int count = 0;
        for (int i = 0; i < SIZE * SIZE * 3; i++) {
            float dval;
            if (fscanf(f, "%f", &dval) != 1) dval = 0.f;
            h_input[i] = dval - coef[count];
            count = (count + 1) % 3;
        }
        fclose(f);
        printf("[init] input: vol.txt loaded\n");
    } else {
        rand_fill(h_input, (size_t)SIZE * SIZE * 3);
        printf("[init] input: random (vol.txt not found)\n");
    }

    h_pool5 = (float *)malloc((size_t)7 * 7 * 512 * sizeof(float));
    h_fc1o  = (float *)malloc(4096 * sizeof(float));
    h_fc2o  = (float *)malloc(4096 * sizeof(float));
    h_fc3o  = (float *)malloc(1000 * sizeof(float));

    omp_set_num_threads(4);
}

// conv + conv + pool 블록 실행 헬퍼
static inline void run_conv(int level, float *in, float *outbuf, int w, int h) {
    dim3 dimGrid(((w - 1) / TILE_WIDTH) + 1, ((h - 1) / TILE_WIDTH) + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution<<<dimGrid, dimBlock>>>(in, d_convW[level], outbuf, d_convB[level],
                                       layers[level][1], w, h, layers[level][0]);
}

static inline void run_pool(float *in, float *outbuf, int channels, int w, int h) {
    int blockwidth = 2;
    int number_blocks = w / blockwidth;
    dim3 dimGrid(number_blocks, number_blocks, 1);
    dim3 dimBlock(blockwidth, blockwidth, 1);
    maxpool<<<dimGrid, dimBlock>>>(in, outbuf, channels, h, w, blockwidth);
}

// ============================================================
// forward_pass: 원본 VGG.cu 실행 흐름 그대로 (malloc/파일IO 제거)
//   conv1_1→1_2→pool1→conv2_1→2_2→pool2→conv3_1..3_3→pool3
//   →conv4_1..4_3→pool4→conv5_1..5_3→pool5→fc1→fc2→fc3→softmax
// ============================================================
static double forward_pass() {
    // 입력 복사 (전처리) — 측정 구간 밖 (AlexNet baseline 과 동일 기준)
    cudaMemcpy(d_input, h_input, (size_t)SIZE * SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

    double start = gettime();
    int w = SIZE, h = SIZE;

    // ===== Block 1: conv1_1, conv1_2, pool1 =====
    run_conv(0, d_input, d_bufA, w, h);
    run_conv(1, d_bufA, d_bufB, w, h);
    run_pool(d_bufB, d_bufA, 64, w, h);
    w /= 2; h /= 2; // 112

    // ===== Block 2: conv2_1, conv2_2, pool2 =====
    run_conv(2, d_bufA, d_bufB, w, h);
    run_conv(3, d_bufB, d_bufA, w, h);
    run_pool(d_bufA, d_bufB, 128, w, h);
    w /= 2; h /= 2; // 56

    // ===== Block 3: conv3_1..3_3, pool3 =====
    run_conv(4, d_bufB, d_bufA, w, h);
    run_conv(5, d_bufA, d_bufB, w, h);
    run_conv(6, d_bufB, d_bufA, w, h);
    run_pool(d_bufA, d_bufB, 256, w, h);
    w /= 2; h /= 2; // 28

    // ===== Block 4: conv4_1..4_3, pool4 =====
    run_conv(7, d_bufB, d_bufA, w, h);
    run_conv(8, d_bufA, d_bufB, w, h);
    run_conv(9, d_bufB, d_bufA, w, h);
    run_pool(d_bufA, d_bufB, 512, w, h);
    w /= 2; h /= 2; // 14

    // ===== Block 5: conv5_1..5_3, pool5 =====
    run_conv(10, d_bufB, d_bufA, w, h);
    run_conv(11, d_bufA, d_bufB, w, h);
    run_conv(12, d_bufB, d_bufA, w, h);
    run_pool(d_bufA, d_bufB, 512, w, h);
    w /= 2; h /= 2; // 7

    cudaDeviceSynchronize();

    // pool5 출력 → host (FC 입력 준비, 명시적 병합)
    cudaMemcpy(h_pool5, d_bufB, (size_t)7 * 7 * 512 * sizeof(float), cudaMemcpyDeviceToHost);

    // ===== FC: CPU (OMP 4) =====
    fc1_cpu(h_pool5, h_dense1, h_fc1o, h_bias1);
    fc2_cpu(h_fc1o, h_dense2, h_fc2o, h_bias2);
    fc3_cpu(h_fc2o, h_dense3, h_fc3o, h_bias3);
    softmax_cpu(h_fc3o, 1000);

    double end = gettime();
    return end - start;
}

int main(int argc, char **argv) {
    int NUM_FRAMES = 3000;
    int WARMUP = 100;
    for (int i = 1; i < argc; i++)
        if (argv[i][0] == '-' && i + 1 < argc) {
            char f = argv[i][1]; i++;
            if (f == 'f') NUM_FRAMES = atoi(argv[i]);
            if (f == 'w') WARMUP = atoi(argv[i]);
        }

    cuInit(0);
    cudaDeviceReset();

    printf("[init] allocating & random-initializing weights...\n");
    double t0 = gettime();
    init_all();
    printf("[init] done in %.2f s\n", gettime() - t0);

    // Warmup
    for (int i = 0; i < WARMUP; i++) forward_pass();

    // Measure
    std::vector<double> ms;
    ms.reserve(NUM_FRAMES);
    double totalSt = gettime();
    for (int i = 0; i < NUM_FRAMES; i++) ms.push_back(forward_pass() * 1000.0);
    double elapsed = gettime() - totalSt;
    double fps = NUM_FRAMES / elapsed;

    double sum = 0; for (double v : ms) sum += v;
    double mean = sum / ms.size();
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) {
        size_t i = (size_t)(s.size() * p / 100.0);
        if (i >= s.size()) i = s.size() - 1;
        return s[i];
    };

    printf("=========================================\n");
    printf(" VGG-16 inter-layer baseline (no pipelining)\n");
    printf("   Conv/Pool = GPU, FC = CPU (OMP 4)\n");
    printf("   weights   = random init (seed 42)\n");
    printf("   frames    = %d  (warmup=%d)\n", NUM_FRAMES, WARMUP);
    printf("-----------------------------------------\n");
    printf("   Wall elapsed = %.3f s  ->  FPS = %.2f\n", elapsed, fps);
    printf("   Frame time (ms):\n");
    printf("     mean = %.3f   p50 = %.3f\n", mean, pct(50));
    printf("     p95  = %.3f   p99 = %.3f\n", pct(95), pct(99));
    printf("=========================================\n");
    return 0;
}

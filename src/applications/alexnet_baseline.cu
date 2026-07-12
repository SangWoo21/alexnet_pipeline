#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "../src/layer.cu"
#include "../include/mnist.h"
#include "../include/pixels.h"

#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <time.h>

using namespace std;

// Define layers of CNN
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

static double forward_pass(double data[227][227][3], cudaStream_t stream1);

int main(int argc, const char **argv) {
    srand(time(NULL));
    double test_data[227][227][3] = {0.0};
    for (int i = 0; i < 227; i++) {
        for (int j = 0; j < 227; j++) {
            for (int k = 0; k < 3; k++) {
                test_data[j][k][i] = double(PIXELS[j][i][k]);
            }
        }
    }
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &Ainput_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac1_o, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac2_o, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac3_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac4_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ac5_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Ap3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af1_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af2_z, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_weight, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_bias, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_a, 0, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream1, &Af3_z, 0, cudaMemAttachHost);

    // Inter-layer baseline: Conv/Pool = GPU 전체, FC = CPU 전체.
    // 파이프라이닝 없이 프레임 순차 처리 (fair comparison target).
    const int NUM_FRAMES = 3000;
    forward_pass(test_data, stream1); // warmup

    double totalSt = gettime();
    for (int f = 0; f < NUM_FRAMES; ++f) {
        forward_pass(test_data, stream1);
    }
    double totalEnd = gettime();
    double elapsed = totalEnd - totalSt;
    double fps = NUM_FRAMES / elapsed;
    printf("=========================================\n");
    printf(" AlexNet inter-layer baseline (no pipelining)\n");
    printf("   Conv/Pool = GPU, FC = CPU\n");
    printf("   frames  = %d\n", NUM_FRAMES);
    printf("-----------------------------------------\n");
    printf("   Elapsed = %.3f s  ->  FPS = %.2f\n", elapsed, fps);
    printf("=========================================\n");
    return 0;
}

static double forward_pass(double data[227][227][3], cudaStream_t stream1) {
    // 입력 복사
    for (int i = 0; i < 227; ++i)
        for (int j = 0; j < 227; ++j)
            for (int k = 0; k < 3; ++k)
                Ainput_a[i * 227 * 3 + j * 3 + k] = data[i][j][k];

    double start = gettime();

    // ===== Conv/Pool: 전부 GPU =====
    offset = 1.0;
    dim3 Bc1(55, 55);
    fp_preact_c1<<<96, Bc1, 0, stream1>>>((float(*)[227][3])L_input.output,
                                          (float(*)[55][55])L_c1.preact,
                                          (float(*)[11][11][3])L_c1.weight, L_c1.bias);
    Aapply_step_function<<<112, 1280, 0, stream1>>>(L_c1.preact, L_c1.act_result, L_c1.O);
    normalization_function<<<112, 640, 0, stream1>>>(L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);
    dim3 ft_map(27, 27);
    fp_preact_p1<<<96, ft_map, 0, stream1>>>((float(*)[55][55])L_c1.output, (float(*)[31][31])L_p1.output);

    dim3 Bc2(27, 27);
    fp_preact_c2<<<256, Bc2, 0, stream1>>>((float(*)[31][31])L_p1.output,
                                           (float(*)[27][27])L_c2.preact,
                                           (float(*)[96][5][5])L_c2.weight, L_c2.bias);
    Aapply_step_function<<<112, 1280, 0, stream1>>>(L_c2.preact, L_c2.act_result, L_c2.O);
    normalization_function<<<112, 1280, 0, stream1>>>(L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);
    dim3 ft_map1(13, 13);
    fp_preact_p2<<<256, ft_map1, 0, stream1>>>((float(*)[27][27])L_c2.output, (float(*)[15][15])L_p2.output);

    dim3 Bc3(13, 13);
    fp_preact_c3<<<384, Bc3, 0, stream1>>>((float(*)[15][15])L_p2.output,
                                           (float(*)[13][13])L_c3.preact,
                                           (float(*)[256][3][3])L_c3.weight, L_c3.bias);
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c3.preact, L_c3.output, L_c3.O);

    dim3 Bc4(12, 12);
    fp_preact_c4<<<384, Bc4, 0, stream1>>>((float(*)[13][13])L_c3.output,
                                           (float(*)[13][13])L_c4.preact,
                                           (float(*)[384][3][3])L_c4.weight, L_c4.bias);
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c4.preact, L_c4.output, L_c4.O);

    dim3 Bc5(12, 12);
    fp_preact_c5<<<256, Bc5, 0, stream1>>>((float(*)[13][13])L_c4.output,
                                           (float(*)[13][13])L_c5.preact,
                                           (float(*)[384][3][3])L_c5.weight, L_c5.bias);
    Aapply_step_function<<<128, 128, 0, stream1>>>(L_c5.preact, L_c5.output, L_c5.O);
    dim3 ft_map2(6, 6);
    fp_preact_p3<<<256, ft_map2, 0, stream1>>>((float(*)[13][13])L_c5.output, (float(*)[6][6])L_p3.output);

    // GPU 완료 대기 (FC 입력 준비)
    cudaStreamSynchronize(stream1);

    // ===== FC: 전부 CPU =====
    offset = 0.0;
    fp_preact_f1_cpu((float(*)[6][6])L_p3.output, L_f1.preact,
                     (float(*)[256][6][6])L_f1.weight, L_f1.bias);
    Aapply_step_function_cpu(L_f1.preact, L_f1.output, L_f1.O);

    fp_preact_f2_cpu(L_f1.output, L_f2.preact,
                     (float(*)[4096])L_f2.weight, L_f2.bias);
    Aapply_step_function_cpu(L_f2.preact, L_f2.output, L_f2.O);

    fp_preact_f3_cpu(L_f2.output, L_f3.preact,
                     (float(*)[4096])L_f3.weight, L_f3.bias);
    Aapply_step_function_cpu(L_f3.preact, L_f3.output, L_f3.O);

    double end = gettime();
    return end - start;
}

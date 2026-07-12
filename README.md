# CPU-GPU DNN Inference Throughput Analysis on Jetson Xavier NX

> **자원 제약형 에지 디바이스에서 CPU-GPU 추론의 처리량(Throughput) 특성 분석**
> 2026년 한국전기전자학회 하계학술대회 발표 논문의 실험 코드 저장소입니다.

## 프로젝트 개요

지연 시간 최적화 SOTA 기법인 **EdgeNN**(IEEE TCC 2025)은 고성능 에지 보드에서 검증되었지만, 자원 제약형 소형 보드(Xavier NX)에서 연속 프레임 스트림의 처리량은 분석되지 않았습니다. 본 프로젝트는 EdgeNN 을 처리량 관점에서 재평가하고 3-스테이지 프레임 파이프라이닝을 구현하여 세 DNN 모델에서 처리량 변화를 실측했습니다.

## 핵심 발견

**파이프라이닝 이득은 병목 스테이지 시간 vs 스테이지 시간 합의 비율이 결정한다.**

| 모델 | Baseline | Pipeline | 변화 |
|---|---|---|---|
| AlexNet | 18.97 FPS | 28.63 FPS | **+50.9%** |
| VGG-16 | 0.46 FPS | 0.67 FPS | **+45.7%** |
| ResNet 계열 소형 CNN | 826.5 FPS | 812.1 FPS | **-1.7%** |

VGG-16 이론 이득 상한(2173/1494 - 1 = +45.4%)과 실측(+45.7%)이 정확히 일치합니다.

## 3-스테이지 프레임 파이프라이닝

- **S0** (CPU core 0): 입력 전처리
- **S1** (GPU, core 1 제어): Conv 계열 전체
- **S2** (CPU core 2-5, OpenMP 4): FC 레이어

**구현 기술** (파이프라인 효율 이론 상한의 98% 이상):
- Linux futex 기반 저비용 스레드 동기화
- 다중 버퍼 (6-slot ring), CPU 코어 어피니티 고정
- cudaEvent 기반 순수 GPU 시간 분리 측정

## 저장소 구조
src/         모든 소스 파일 (baseline + pipeline)
scripts/     빌드 스크립트
results/     실측 로그
paper/       발표 논문

## 빌드 방법

이 저장소의 소스 파일은 EdgeNN 원저장소의 `layer.cu`, `mnist.h` 등에 의존합니다:

1. EdgeNN 저장소를 먼저 클론
2. 이 저장소의 `src/*.cu` 파일을 EdgeNN 의 `applications/` 폴더로 복사
3. Jetson Xavier NX 에서 빌드:

```bash
nvcc -gencode arch=compute_72,code=sm_72 -O3 -std=c++17 --use_fast_math \
     -Xcompiler "-pthread -Wall -O3 -fopenmp" \
     -o <출력> <소스.cu> -lcudart -lpthread -lrt -lgomp -lcuda
```

## 실행

```bash
sudo jetson_clocks

./alexnet_baseline           # 3,000 프레임, warmup 100
./alexnet_futex -f 3000 -w 100
./vgg_baseline
./vgg_futex -f 3000 -w 100
./resnet_baseline
./resnet_futex -f 3000 -w 100
```

## 기술 스택

- **CUDA / 이기종 컴퓨팅**: Jetson unified memory, cudaEvent, cudaStream
- **시스템 프로그래밍**: Linux futex, CPU affinity, OpenMP
- **성능 분석**: percentile(p50/p95/p99), 파이프라인 이론 상한 모델링

## 참고문헌

- [1] A. A. Majeed et al., "Scheduling Techniques of AI Models on Modern Heterogeneous Edge GPU," IEEE Trans. Ind. Informat., 2026.
- [2] F. Zhang et al., "Breaking the Edge: Enabling Efficient Neural Network Inference on Integrated Edge Devices," IEEE Trans. Cloud Comput., 2025.
- [3] J.-M. Yang et al., "CP-CNN," IEEE Access, 2023.

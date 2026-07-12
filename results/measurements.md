# 실측 결과 원본

측정 환경: NVIDIA Jetson Xavier NX, JetPack, `sudo jetson_clocks` 적용
측정 조건: 3,000 프레임, warmup 100 프레임 제외

---

## AlexNet

### Baseline (EdgeNN 원본, offset=1.0)
```
AlexNet (EdgeNN original, unmodified forward_pass)
  offset  = 1.00
  frames  = 3000  (warmup=100)
  Wall elapsed = 158.148 s  ->  FPS = 18.97
  Frame time (ms): mean = 52.569  p50 = 52.562  p95 = 52.811  p99 = 52.948
```

### Pipeline (3-stage futex)
```
Pipeline FPS: 28.63    Eff: 99.87%
S0(core{0})   run=<0.1 ms
S1(core{1})   run=24.12 ms   (GPU conv chain)
S2(core{2-5}) run=34.88 ms   (FC, OpenMP 4)
```

**개선: 18.97 → 28.63 FPS (+50.9%)**

---

## VGG-16 (랜덤 가중치, seed 42)

### Baseline
```
VGG-16 inter-layer baseline (no pipelining)
  Conv/Pool = GPU, FC = CPU (OMP 4)
  frames    = 3000  (warmup=100)
  Wall elapsed = 6525.875 s  ->  FPS = 0.46
  Frame time (ms): mean = 2175.087  p50 = 2174.572  p95 = 2184.187  p99 = 2199.278
```

### Pipeline
```
VGG-16 Futex 3-Stage Pipeline
S0(core{0})   run=0.099 ms
S1(core{1})   run=1494.002 ms  (GPU conv chain, Pure_GPU=1493.555)
S2(core{2-5}) run=679.006 ms   (FC, OpenMP 4)
Theory FPS  : 0.67   Pipeline FPS: 0.67   Eff: 99.98%
Cycle_ms    : 1494.002   Cycle_p95_ms: 1494.888
```

**개선: 0.46 → 0.67 FPS (+45.7%)**
이론 상한 확인: 스테이지 합 2173 ms / 병목 1494 ms → 상한 +45.4% ≈ 실측 +45.7%

---

## ResNet 계열 소형 CNN (MNIST 기반)

### Baseline (EdgeNN 원본, offset=1.0)
```
ResNet (EdgeNN original, unmodified forward_pass)
  offset  = 1.00
  frames  = 3000  (warmup=100)
  Wall elapsed = 3.630 s  ->  FPS = 826.48
  Frame time (ms): mean = 1.207  p50 = 1.202  p95 = 1.253  p99 = 1.288
```

### Pipeline (3-stage futex)
```
ResNet Futex 3-Stage Pipeline
S0(core{0})   run=0.003 ms
S1(core{1})   run=1.215 ms  (Pure_GPU=0.608 ms + slot memcpy=0.607 ms)
S2(core{2-5}) run=0.084 ms  (FC, OpenMP 4)
Theory FPS  : 822.85   Pipeline FPS: 812.09   Eff: 98.69%
Overhead    : 43.05%
```

**변화: 826.48 → 812.09 FPS (−1.7%)**
음성 사례 원인: FC 이전 이득(0.084 ms) < 파이프라인 slot 전달 비용(0.607 ms)

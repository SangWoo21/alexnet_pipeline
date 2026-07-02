# AlexNet CPU-GPU-CPU Pipeline (Jetson Xavier NX)

## Environment

* Platform: NVIDIA Jetson Xavier NX
* CUDA: 11.x
* Compiler: NVCC
* Architecture: sm_72

---

## Build

### Baseline

```bash
cd applications

nvcc -gencode arch=compute_72,code=sm_72 \
-O3 \
-std=c++17 \
--use_fast_math \
-Xcompiler "-pthread -Wall -O3 -fopenmp" \
-o alexnet_baseline \
alexnet_baseline.cu \
-lcudart -lpthread -lrt -lgomp
```

### Pipeline

```bash
cd applications

nvcc -gencode arch=compute_72,code=sm_72 \
-O3 \
-std=c++17 \
--use_fast_math \
-Xcompiler "-pthread -Wall -O3 -fopenmp" \
-o alexnet_pipeline \
alexnet_pipeline.cu \
-lcudart -lpthread -lrt -lgomp
```

---

## Run

### Baseline

```bash
cd applications
./alexnet_baseline
```

Example output

```
=========================================
 AlexNet inter-layer baseline (no pipelining)
   Conv/Pool = GPU, FC = CPU
   frames  = 3000
-----------------------------------------
   Elapsed = 150.064 s  ->  FPS = 19.99
=========================================
```

### Pipeline

```bash
cd applications
./alexnet_pipeline
```

Example output

```
====================================================
 AlexNet 3-stage pipeline (real layer.cu kernels)
   N_SLOTS  = 6
   frames   = 3000
   cores    = S0:{0,1} S1:{3} S2:{2,4,5}
----------------------------------------------------
   Wall     elapsed = 159210.86 ms  ->  Wall FPS = 18.84
   Pipeline elapsed = 159210.17 ms  ->  Pipeline FPS = 18.84
====================================================
```

---

## Notes

* The implementation targets the NVIDIA Jetson Xavier NX platform.
* The baseline executes all convolution/pooling layers on the GPU and all fully connected layers on the CPU without pipelining.
* The pipeline implementation uses a three-stage CPU-GPU-CPU pipeline with thread affinity and Linux futex-based synchronization.
* All performance measurements reported in this repository use 3000 inference frames after warm-up.

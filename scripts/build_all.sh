#!/usr/bin/env bash
set -e

NVCC_FLAGS=(
    -gencode arch=compute_72,code=sm_72
    -O3 -std=c++17 --use_fast_math
    -Xcompiler "-pthread -Wall -O3 -fopenmp"
)
LIBS=(-lcudart -lpthread -lrt -lgomp -lcuda)

build_one() {
    local name=$1
    local src=$2
    if [ ! -f "$src" ]; then
        echo "  [skip] $src 없음"
        return
    fi
    echo "  [build] $src → $name"
    nvcc "${NVCC_FLAGS[@]}" -o "$name" "$src" "${LIBS[@]}"
}

case "${1:-all}" in
    clean)
        echo "== clean =="
        rm -f alexnet_edgenn alexnet_futex \
              resnet_edgenn resnet_baseline resnet_futex \
              vgg_baseline vgg_futex
        ;;
    alexnet)
        build_one alexnet_edgenn alexnet_edgenn.cu
        build_one alexnet_futex  alexnet_futex.cu
        ;;
    resnet)
        build_one resnet_edgenn   resnet_edgenn.cu
        build_one resnet_baseline resnet_baseline.cu
        build_one resnet_futex    resnet_futex.cu
        ;;
    vgg)
        build_one vgg_baseline vgg_baseline.cu
        build_one vgg_futex    vgg_futex.cu
        ;;
    all)
        build_one alexnet_edgenn  alexnet_edgenn.cu
        build_one alexnet_futex   alexnet_futex.cu
        build_one resnet_edgenn   resnet_edgenn.cu
        build_one resnet_baseline resnet_baseline.cu
        build_one resnet_futex    resnet_futex.cu
        build_one vgg_baseline    vgg_baseline.cu
        build_one vgg_futex       vgg_futex.cu
        ;;
    *)
        echo "사용법: bash build_all.sh [all|alexnet|resnet|vgg|clean]"
        exit 1
        ;;
esac

echo "== 완료 =="

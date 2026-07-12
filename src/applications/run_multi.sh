#!/bin/bash
# ============================================================
# Multi-instance contention test
#   - 1, 2, 4 instance 동시 실행해서 자원 경쟁 효과 측정
#   - B3 (pthread_mutex) vs B4 (futex) 비교
#   - futex의 간접 이득 (커널 진입 감소) 차이 확인용
# ============================================================

mkdir -p multi_logs
rm -f multi_logs/*.log

extract() {
    grep -E "$2" "$1" 2>/dev/null | grep -oE "$3" | head -1
}

run_n_instances() {
    local bin=$1
    local n=$2
    local out_prefix=$3

    echo "  Running $n × $bin..."
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1

    # 백그라운드로 N 인스턴스 동시 실행
    local pids=()
    for i in $(seq 1 $n); do
        ./$bin -f 100 -w 10 > "multi_logs/${out_prefix}_${i}.log" 2>&1 &
        pids+=($!)
    done

    # 모두 끝나길 대기
    for pid in "${pids[@]}"; do
        wait $pid
    done

    # 각 인스턴스의 핵심 지표 추출 + 평균
    local fps_sum=0 sync_sum=0 fast_sum=0 jitter_sum=0
    local fps_min=99999 fps_max=0
    for i in $(seq 1 $n); do
        local log="multi_logs/${out_prefix}_${i}.log"
        local fps=$(extract "$log" "Pipeline FPS" '[0-9]+\.[0-9]+')
        local sync_b=$(extract "$log" "sync_blocked" '[0-9]+')
        local fast=$(extract "$log" "fastpath_rate_pct" '[0-9]+\.[0-9]+')
        local jit=$(extract "$log" "FrameWall_std_ms" '[0-9]+\.[0-9]+')

        fps_sum=$(awk -v a=$fps_sum -v b=${fps:-0} 'BEGIN{print a+b}')
        sync_sum=$(awk -v a=$sync_sum -v b=${sync_b:-0} 'BEGIN{print a+b}')
        fast_sum=$(awk -v a=$fast_sum -v b=${fast:-0} 'BEGIN{print a+b}')
        jitter_sum=$(awk -v a=$jitter_sum -v b=${jit:-0} 'BEGIN{print a+b}')

        fps_min=$(awk -v a=$fps_min -v b=${fps:-0} 'BEGIN{print (b<a)?b:a}')
        fps_max=$(awk -v a=$fps_max -v b=${fps:-0} 'BEGIN{print (b>a)?b:a}')
    done

    local fps_avg=$(awk -v s=$fps_sum -v n=$n 'BEGIN{printf "%.2f", s/n}')
    local sync_avg=$(awk -v s=$sync_sum -v n=$n 'BEGIN{printf "%.0f", s/n}')
    local fast_avg=$(awk -v s=$fast_sum -v n=$n 'BEGIN{printf "%.2f", s/n}')
    local jitter_avg=$(awk -v s=$jitter_sum -v n=$n 'BEGIN{printf "%.2f", s/n}')

    printf "    FPS avg=%s (min=%s, max=%s)  sync_blk_avg=%s  fastpath_avg=%s%%  jitter_avg=%s\n" \
        "$fps_avg" "$fps_min" "$fps_max" "$sync_avg" "$fast_avg" "$jitter_avg"
}

# N=4 부하 조건에서 5번 반복 측정
for trial in 1 2 3 4 5; do
    echo "=== Trial $trial ==="
    sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
    
    # B3 × 4 동시
    for i in 1 2 3 4; do
        ./B3_affinity -f 100 -w 10 > multi_logs/verify_B3_t${trial}_i${i}.log 2>&1 &
    done
    wait
    
    # B4 × 4 동시
    for i in 1 2 3 4; do
        ./B4_futex -f 100 -w 10 > multi_logs/verify_B4_t${trial}_i${i}.log 2>&1 &
    done
    wait
    
    echo "  B3:"
    grep "FrameWall_std_ms" multi_logs/verify_B3_t${trial}_i*.log | awk '{print "    "$0}'
    echo "  B4:"
    grep "FrameWall_std_ms" multi_logs/verify_B4_t${trial}_i*.log | awk '{print "    "$0}'
done
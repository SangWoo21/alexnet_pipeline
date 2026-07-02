#!/bin/bash
# ============================================================
# 다중 인스턴스 RSS 합산 측정
#   - N개 인스턴스를 백그라운드로 동시 실행
#   - 각자 끝나면 VmHWM_kB(peak RSS) 출력
#   - 합산해서 표로 보여줌
#
# 사용법:
#   ./run_multi_rss.sh B3_affinity
#   ./run_multi_rss.sh B4_futex
#   ./run_multi_rss.sh B4_futex 1000 5    # 1000 frames, 5 instances
# ============================================================
set -e

BIN=${1:?사용법: $0 <바이너리이름> [frames] [instances]}
FRAMES=${2:-1000}
INSTANCES=${3:-5}

if [ ! -x "./${BIN}" ]; then
    echo "오류: ./${BIN} 실행 파일이 없거나 실행 권한이 없음. 먼저 nvcc로 빌드해야 함."
    exit 1
fi

LOGDIR=multi_rss_logs
mkdir -p $LOGDIR
rm -f $LOGDIR/${BIN}_inst*.log

echo "=== ${BIN}: ${INSTANCES} instances, ${FRAMES} frames each ==="
echo "[launching ${INSTANCES} processes...]"

# 동시에 실행
START=$(date +%s)
for i in $(seq 1 $INSTANCES); do
    ./${BIN} -f $FRAMES > $LOGDIR/${BIN}_inst${i}.log 2>&1 &
done

# 모두 끝날 때까지 대기
wait
END=$(date +%s)
echo "[all ${INSTANCES} processes finished in $((END-START))s]"
echo ""

# 결과 파싱 및 합산
total_rss=0
total_hwm=0
ok_count=0

printf "%-8s %12s %12s %12s\n" "inst" "VmRSS_kB" "VmHWM_kB" "Pipeline FPS"
printf "%-8s %12s %12s %12s\n" "----" "--------" "--------" "------------"
for i in $(seq 1 $INSTANCES); do
    log=$LOGDIR/${BIN}_inst${i}.log
    rss=$(grep -E "^VmRSS_kB:" $log 2>/dev/null | awk '{print $2}')
    hwm=$(grep -E "^VmHWM_kB:" $log 2>/dev/null | awk '{print $2}')
    fps=$(grep -E "Pipeline FPS" $log 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)

    if [ -n "$rss" ] && [ -n "$hwm" ]; then
        printf "%-8d %12d %12d %12s\n" $i $rss $hwm "${fps:-?}"
        total_rss=$((total_rss + rss))
        total_hwm=$((total_hwm + hwm))
        ok_count=$((ok_count + 1))
    else
        printf "%-8d %12s %12s %12s  (log: %s)\n" $i "FAIL" "FAIL" "?" "$log"
    fi
done

echo "---------------------------------------------------------"
if [ $ok_count -gt 0 ]; then
    printf "%-8s %12d %12d\n" "TOTAL" $total_rss $total_hwm
    rss_mb=$(awk "BEGIN{printf \"%.1f\", $total_rss/1024}")
    hwm_mb=$(awk "BEGIN{printf \"%.1f\", $total_hwm/1024}")
    printf "         (%9s MB) (%9s MB)\n" "$rss_mb" "$hwm_mb"
    avg_hwm=$((total_hwm / ok_count))
    avg_mb=$(awk "BEGIN{printf \"%.1f\", $avg_hwm/1024}")
    echo ""
    echo "Per-instance avg peak RSS: ${avg_hwm} kB (${avg_mb} MB)"
fi
echo ""
echo "로그는 ${LOGDIR}/ 에 있음. 문제 있으면 ${LOGDIR}/${BIN}_inst1.log 부터 확인."

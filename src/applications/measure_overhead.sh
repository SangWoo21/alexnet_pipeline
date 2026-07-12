#!/bin/bash
# ============================================================
# 오버헤드 직접 측정 보고서 v4
#   - 모든 신뢰성 있는 지표 (워커별 ctxt / sync calls / jitter / pure compute) 비교
#   - 측정 노이즈 명확히 표기 (trial 간 std)
# ============================================================
# (set -e 의도적으로 제거)

TRIALS=1
FRAMES=100
WARMUP=10
BINS=("B1_pipeline" "B2_heterogeneous" "B3_last" "B4_last")

while getopts "t:f:w:" opt; do
    case $opt in
        t) TRIALS=$OPTARG ;;
        f) FRAMES=$OPTARG ;;
        w) WARMUP=$OPTARG ;;
    esac
done

mkdir -p overhead_logs

# 각 지표별 누적 (단순 합산 → 마지막에 평균)
declare -A SUM
declare -A FIRST_CHK

# trial 결과를 모아두는 임시 파일 (std 계산용)
trial_data_dir=overhead_logs/trial_csv
mkdir -p $trial_data_dir
rm -f $trial_data_dir/*.csv

extract() {
    # extract "label" file pattern
    grep -E "$2" "$1" 2>/dev/null | grep -oE "$3" | head -1
}

for BIN in "${BINS[@]}"; do
    if [ ! -x "./${BIN}" ]; then
        echo "오류: ./${BIN} 없음. 먼저 nvcc 빌드 필요."
        continue
    fi

    echo "============================================================"
    echo "[$BIN] ${TRIALS} trial(s), frames=${FRAMES}, warmup=${WARMUP}"
    echo "============================================================"

    csv=$trial_data_dir/${BIN}.csv
    echo "trial,fps,overhead_pct,btl_wait_ms,gpu_conv_ms,cpu_fc_ms,frame_std_ms,frame_p99_ms,sync_attempts,sync_blocked,fastpath_pct,wrk_vol,wrk_nonvol,checksum" > $csv

    for t in $(seq 1 $TRIALS); do
        sync 2>/dev/null
        echo 3 2>/dev/null | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1

        OUT="overhead_logs/${BIN}_trial${t}.log"
        ./${BIN} -f $FRAMES -w $WARMUP > "$OUT" 2>&1

        fps=$(     extract "$OUT" "Pipeline FPS"          '[0-9]+\.[0-9]+')
        ov_pct=$(  extract "$OUT" "Overhead_pct"          '[0-9]+\.[0-9]+')
        btl_w=$(   extract "$OUT" "Bottleneck_wait_ms"    '[0-9]+\.[0-9]+')
        gpu_c=$(   extract "$OUT" "Pure_GPU_conv_ms"      '[0-9]+\.[0-9]+')
        cpu_f=$(   extract "$OUT" "Pure_CPU_FC_ms"        '[0-9]+\.[0-9]+')
        fw_std=$(  extract "$OUT" "FrameWall_std_ms"      '[0-9]+\.[0-9]+')
        fw_p99=$(  extract "$OUT" "FrameWall_p99_ms"      '[0-9]+\.[0-9]+')
        sync_a=$(  extract "$OUT" "sync_api_attempts"     '[0-9]+')
        sync_b=$(  extract "$OUT" "sync_blocked"          '[0-9]+')
        fp_pct=$(  extract "$OUT" "fastpath_rate_pct"     '[0-9]+\.[0-9]+')
        wvol=$(    extract "$OUT" "Workers vol total"     '[0-9]+')
        wnonvol=$( extract "$OUT" "Workers nonvol total"  '[0-9]+')
        chk=$(grep "Output_checksum" "$OUT" | awk '{print $2}')

        printf "  trial %2d: FPS=%-7s overhead=%-5s%% btl_wait=%-7sms sync_blk=%-5s wrk_vol=%-5s frame_std=%-6s\n" \
            $t "${fps:-?}" "${ov_pct:-?}" "${btl_w:-?}" "${sync_b:-?}" "${wvol:-?}" "${fw_std:-?}"

        # CSV에 한 줄 (빈 값은 0으로)
        echo "$t,${fps:-0},${ov_pct:-0},${btl_w:-0},${gpu_c:-0},${cpu_f:-0},${fw_std:-0},${fw_p99:-0},${sync_a:-0},${sync_b:-0},${fp_pct:-0},${wvol:-0},${wnonvol:-0},${chk:-NA}" >> $csv
    done

    if [ -z "${FIRST_CHK[$BIN]}" ]; then
        FIRST_CHK[$BIN]=$(tail -1 $csv | awk -F, '{print $14}')
    fi
    echo ""
done

# ============================================================
# 비교 표 (mean ± std)
# ============================================================
echo "============================================================"
echo " 측정 결과 (${TRIALS} trials, mean ± std)"
echo "============================================================"

# CSV를 awk로 처리 — 한 열의 mean, std 계산
stat_col() {
    local file=$1
    local col=$2
    [ -f "$file" ] || { printf "n/a"; return; }
    awk -F, -v c=$col 'NR>1 { s+=$c; ss+=$c*$c; n++ }
                       END   { if (n>0) {
                                  m=s/n;
                                  v=(ss/n)-(m*m); if (v<0) v=0;
                                  printf "%.2f±%.2f", m, sqrt(v)
                              } else { printf "n/a" } }' $file
}

CSV_DIR=overhead_logs/trial_csv
B1=$CSV_DIR/B1_pipeline.csv
B2=$CSV_DIR/B2_heterogeneous.csv
B3=$CSV_DIR/B3_affinity.csv
B4=$CSV_DIR/B4_futex.csv

# 각 단계별 CSV 한 줄 출력하는 헬퍼
row() {
    # row "지표명" 컬럼번호
    local label=$1
    local col=$2
    printf "  %-24s %16s %16s %16s %16s\n" \
        "$label" \
        "$(stat_col $B1 $col)" \
        "$(stat_col $B2 $col)" \
        "$(stat_col $B3 $col)" \
        "$(stat_col $B4 $col)"
}

printf "\n%-26s %16s %16s %16s %16s\n" \
    "지표" "B1_pipeline" "B2_heterogen" "B3_affinity" "B4_futex"
printf "%-26s %16s %16s %16s %16s\n" \
    "--------------------------" "----------------" "----------------" "----------------" "----------------"

echo ""
echo "[성능]"
row "Pipeline FPS"             2
row "Overhead_pct (%)"         3

echo ""
echo "[동기화 직접 지표]"
row "Bottleneck_wait_ms"       4
row "sync_blocked (syscall)"  10
row "sync_api_attempts"        9
row "fastpath_rate_pct"       11

echo ""
echo "[Frame-time jitter]"
row "FrameWall_std_ms"         7
row "FrameWall_p99_ms"         8

echo ""
echo "[워커 ctxt switch — 메인 아닌 진짜]"
row "Workers vol total"       12
row "Workers nonvol total"    13

echo ""
echo "[통제 변수 — Pure compute는 단계마다 다를 수 있음]"
echo "  (B1: 전체 GPU. B2/B3/B4: GPU Conv만)"
row "Pure GPU compute (ms)"    5
row "Pure CPU FC    (ms)"      6
printf "  %-24s %16s %16s %16s %16s\n" "Output checksum" \
    "${FIRST_CHK[B1_pipeline]:-?}" \
    "${FIRST_CHK[B2_heterogeneous]:-?}" \
    "${FIRST_CHK[B3_affinity]:-?}" \
    "${FIRST_CHK[B4_futex]:-?}"

echo ""
echo "[해석 가이드]"
echo "  - Pipeline FPS의 단계별 증가가 ablation의 핵심"
echo "  - 통제 변수(Pure compute)가 B2→B3→B4에서 같아야 'sync만 바뀜' 주장 성립"
echo "    (B1은 GPU 전체라 Pure GPU compute가 크게 다른 게 정상)"
echo "  - sync_blocked / Bottleneck_wait_ms가 단계별로 줄어들면 동기화 자체는 개선"
echo "  - Pipeline FPS 차이가 trial std보다 작으면 = 통계적 노이즈 (구분 불가)"
echo ""
echo "원본 로그: overhead_logs/*.log, trial CSV: $CSV_DIR/"

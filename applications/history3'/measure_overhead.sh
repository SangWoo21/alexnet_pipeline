#!/bin/bash
# ============================================================
# 오버헤드 직접 측정 보고서 생성  (v3 — set -e 제거)
#   - 각 단계(B3, B4)를 perf로 감싸서 futex syscall / ctxt switch / migration 카운트
#   - 바이너리 자체 출력의 "오버헤드 분해" / "통제 변수" 같이 수집
#   - 최종적으로 한 화면 비교 보고서 출력
#
# 사용법:
#   ./measure_overhead.sh                          # B3, B4 둘 다 1회씩
#   ./measure_overhead.sh -t 5                     # 각각 5 trial 평균
#   ./measure_overhead.sh -t 10 -f 1000            # 10 trial, 1000 frames
# ============================================================
# (set -e 의도적으로 제거: grep/perf 등 정상적 실패가 많아 죽으면 안 됨)

TRIALS=1
FRAMES=100
WARMUP=10
BINS=("B3_affinity" "B4_futex")

while getopts "t:f:w:" opt; do
    case $opt in
        t) TRIALS=$OPTARG ;;
        f) FRAMES=$OPTARG ;;
        w) WARMUP=$OPTARG ;;
    esac
done

# perf 도구 존재 확인 (Ubuntu wrapper만 있고 실제 perf는 없는 경우도 거름)
HAVE_PERF=0
if command -v perf >/dev/null 2>&1; then
    perf_check=$(perf --version 2>&1 || true)
    if echo "$perf_check" | grep -q "^perf version"; then
        HAVE_PERF=1
    else
        echo "INFO: perf wrapper만 있고 실제 바이너리는 없음. syscall 카운트 없이 진행."
        echo "      (설치 원하면: sudo apt install linux-tools-tegra)"
    fi
else
    echo "WARN: perf 명령이 없음. syscall 카운트 없이 진행."
fi

# perf_event_paranoid 권한 확인
if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
    if [ "$PARANOID" -gt 1 ]; then
        echo "INFO: perf_event_paranoid=$PARANOID. -1로 낮추기 위해 sudo 필요:"
        echo "      sudo sysctl -w kernel.perf_event_paranoid=-1"
    fi
fi

mkdir -p overhead_logs

# 결과 누적용 (단계별 누적합 → 마지막에 평균)
declare -A sum_overhead_pct sum_useful sum_gpu_conv sum_cpu_fc
declare -A sum_pipe_fps sum_vol_ctxt sum_migrations
declare -A sum_futex_syscalls sum_perf_ctxt sum_perf_mig sum_llc
declare -A first_checksum

for BIN in "${BINS[@]}"; do
    if [ ! -x "./${BIN}" ]; then
        echo "오류: ./${BIN} 없음. 먼저 nvcc 빌드 필요."
        exit 1
    fi

    echo "============================================================"
    echo "[$BIN] ${TRIALS} trial(s), frames=${FRAMES}, warmup=${WARMUP}"
    echo "============================================================"

    sum_overhead_pct[$BIN]=0
    sum_useful[$BIN]=0; sum_gpu_conv[$BIN]=0; sum_cpu_fc[$BIN]=0
    sum_pipe_fps[$BIN]=0; sum_vol_ctxt[$BIN]=0; sum_migrations[$BIN]=0
    sum_futex_syscalls[$BIN]=0; sum_perf_ctxt[$BIN]=0
    sum_perf_mig[$BIN]=0; sum_llc[$BIN]=0
    first_checksum[$BIN]=""

    for t in $(seq 1 $TRIALS); do
        sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true

        OUT="overhead_logs/${BIN}_trial${t}.log"
        PERFOUT="overhead_logs/${BIN}_trial${t}.perf"

        if [ $HAVE_PERF -eq 1 ]; then
            # perf stat: futex syscall + ctxt switch + migration 동시 카운트
            perf stat -e 'syscalls:sys_enter_futex,context-switches,cpu-migrations' \
                -o "$PERFOUT" \
                ./${BIN} -f $FRAMES -w $WARMUP > "$OUT" 2>&1 || true
        else
            ./${BIN} -f $FRAMES -w $WARMUP > "$OUT" 2>&1
        fi

        # 바이너리 자체 출력 파싱
        ov_pct=$(grep "Overhead_pct" "$OUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        useful=$(grep "Useful_ms"   "$OUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        gpu_c=$( grep "Pure_GPU_conv_ms" "$OUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        cpu_f=$( grep "Pure_CPU_FC_ms"   "$OUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        fps=$(   grep "Pipeline FPS" "$OUT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        vol=$(   grep "Voluntary ctxt switches" "$OUT" | grep -oE '[0-9]+' | head -1)
        mig=$(   grep "Thread migrations" "$OUT" | grep -oE '[0-9]+' | head -1)
        llc=$(   grep "LLC miss per frame" "$OUT" | grep -oE '[0-9]+' | head -1)
        chk=$(   grep "Output_checksum" "$OUT" | awk '{print $2}')

        # perf stat 출력 파싱
        if [ $HAVE_PERF -eq 1 ] && [ -f "$PERFOUT" ]; then
            futex_n=$(grep "sys_enter_futex" "$PERFOUT" | awk '{gsub(",",""); print $1}')
            pctx_n=$( grep "context-switches" "$PERFOUT" | awk '{gsub(",",""); print $1}')
            pmig_n=$( grep "cpu-migrations"   "$PERFOUT" | awk '{gsub(",",""); print $1}')
        else
            futex_n=0; pctx_n=0; pmig_n=0
        fi

        printf "  trial %d: FPS=%s  overhead=%s%%  GPU_conv=%sms  CPU_FC=%sms  futex_syscall=%s  vol_ctxt=%s\n" \
            $t "${fps:-?}" "${ov_pct:-?}" "${gpu_c:-?}" "${cpu_f:-?}" "${futex_n:-?}" "${vol:-?}"

        # 누적 (bc로 실수 합산)
        [ -n "$ov_pct" ]  && sum_overhead_pct[$BIN]=$(echo "${sum_overhead_pct[$BIN]} + $ov_pct" | bc)
        [ -n "$useful" ] && sum_useful[$BIN]=$(echo "${sum_useful[$BIN]} + $useful" | bc)
        [ -n "$gpu_c" ]  && sum_gpu_conv[$BIN]=$(echo "${sum_gpu_conv[$BIN]} + $gpu_c" | bc)
        [ -n "$cpu_f" ]  && sum_cpu_fc[$BIN]=$(echo "${sum_cpu_fc[$BIN]} + $cpu_f" | bc)
        [ -n "$fps" ]    && sum_pipe_fps[$BIN]=$(echo "${sum_pipe_fps[$BIN]} + $fps" | bc)
        [ -n "$vol" ]    && sum_vol_ctxt[$BIN]=$((${sum_vol_ctxt[$BIN]} + vol))
        [ -n "$mig" ]    && sum_migrations[$BIN]=$((${sum_migrations[$BIN]} + mig))
        [ -n "$llc" ]    && sum_llc[$BIN]=$((${sum_llc[$BIN]} + llc))
        [ -n "$futex_n" ] && sum_futex_syscalls[$BIN]=$((${sum_futex_syscalls[$BIN]} + futex_n))
        [ -n "$pctx_n" ]  && sum_perf_ctxt[$BIN]=$((${sum_perf_ctxt[$BIN]} + pctx_n))
        [ -n "$pmig_n" ]  && sum_perf_mig[$BIN]=$((${sum_perf_mig[$BIN]} + pmig_n))

        # 첫 trial의 checksum만 저장 (단계 내 동일성은 같은 입력이라 자동 보장)
        if [ -z "${first_checksum[$BIN]}" ] && [ -n "$chk" ]; then
            first_checksum[$BIN]=$chk
        fi
    done
    echo ""
done

# 비교 표 출력
echo "============================================================"
echo " OVERHEAD COMPARISON (${TRIALS} trial 평균)"
echo "============================================================"
printf "%-28s %12s %12s %10s\n" "지표" "B3_affinity" "B4_futex" "감소율"
printf "%-28s %12s %12s %10s\n" "----" "----------" "--------" "------"

# 평균 함수
avg() { echo "scale=4; $1 / $TRIALS" | bc; }
delta_pct() {
    # 감소율(%) = (B3 - B4) / B3 * 100
    awk -v a="$1" -v b="$2" 'BEGIN { if (a>0) printf "%.1f%%", (a-b)/a*100; else printf "n/a"; }'
}

B3_ov=$(avg "${sum_overhead_pct[B3_affinity]}")
B4_ov=$(avg "${sum_overhead_pct[B4_futex]}")
B3_us=$(avg "${sum_useful[B3_affinity]}")
B4_us=$(avg "${sum_useful[B4_futex]}")
B3_gp=$(avg "${sum_gpu_conv[B3_affinity]}")
B4_gp=$(avg "${sum_gpu_conv[B4_futex]}")
B3_cp=$(avg "${sum_cpu_fc[B3_affinity]}")
B4_cp=$(avg "${sum_cpu_fc[B4_futex]}")
B3_fp=$(avg "${sum_pipe_fps[B3_affinity]}")
B4_fp=$(avg "${sum_pipe_fps[B4_futex]}")
B3_vc=$((${sum_vol_ctxt[B3_affinity]} / TRIALS))
B4_vc=$((${sum_vol_ctxt[B4_futex]}    / TRIALS))
B3_mg=$((${sum_migrations[B3_affinity]} / TRIALS))
B4_mg=$((${sum_migrations[B4_futex]}    / TRIALS))
B3_fx=$((${sum_futex_syscalls[B3_affinity]} / TRIALS))
B4_fx=$((${sum_futex_syscalls[B4_futex]}    / TRIALS))
B3_lc=$((${sum_llc[B3_affinity]} / TRIALS))
B4_lc=$((${sum_llc[B4_futex]}    / TRIALS))

echo "[핵심: 오버헤드]"
printf "%-28s %12s %12s %10s\n" "  Overhead_pct (%)"        "$B3_ov" "$B4_ov" "$(delta_pct $B3_ov $B4_ov)"
printf "%-28s %12s %12s %10s\n" "  Pipeline FPS"            "$B3_fp" "$B4_fp" "$(delta_pct $B4_fp $B3_fp) (↑)"
echo ""
echo "[직접 측정]"
printf "%-28s %12s %12s %10s\n" "  futex syscall (perf)"    "$B3_fx" "$B4_fx" "$(delta_pct $B3_fx $B4_fx)"
printf "%-28s %12s %12s %10s\n" "  Voluntary ctxt switches" "$B3_vc" "$B4_vc" "$(delta_pct $B3_vc $B4_vc)"
printf "%-28s %12s %12s %10s\n" "  Thread migrations"       "$B3_mg" "$B4_mg" "$(delta_pct $B3_mg $B4_mg)"
[ "$B3_lc" -gt 0 ] && printf "%-28s %12s %12s %10s\n" "  LLC miss / frame"        "$B3_lc" "$B4_lc" "$(delta_pct $B3_lc $B4_lc)"
echo ""
echo "[통제 변수 — 같아야 함]"
printf "%-28s %12s %12s\n" "  Pure GPU conv (ms)"  "$B3_gp" "$B4_gp"
printf "%-28s %12s %12s\n" "  Pure CPU FC  (ms)"   "$B3_cp" "$B4_cp"
printf "%-28s %12s %12s\n" "  Useful_ms"           "$B3_us" "$B4_us"
printf "%-28s %12s %12s\n" "  Output checksum"     "${first_checksum[B3_affinity]:-?}" "${first_checksum[B4_futex]:-?}"
echo ""
echo "로그: overhead_logs/ — 원본 출력은 .log, perf stat 결과는 .perf"
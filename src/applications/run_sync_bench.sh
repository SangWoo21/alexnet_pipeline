#!/bin/bash
# ============================================================
# 마이크로벤치마크 실행 + contention level sweep
#   - 같은 알고리즘 (Producer-Consumer with bounded queue)
#   - 동기화 primitive만 교체 (pthread_mutex+cond vs futex)
#   - 다양한 contention level에서 비교
# ============================================================

TRIALS=${1:-5}
ITEMS=${2:-100000}   # items per producer

BIN=./micro_sync_bench
[ ! -x "$BIN" ] && { echo "오류: 빌드 먼저: g++ -O2 -std=c++17 -pthread micro_sync_bench.cpp -o micro_sync_bench"; exit 1; }

mkdir -p sync_logs
rm -f sync_logs/*.log sync_logs/results.csv

CSV=sync_logs/results.csv
echo "P,C,mode,trial,throughput,push_p99,pop_p99,blocked_pct,fastpath_pct,ctxt_per_op" > $CSV

extract() {
    grep -E "$2" "$1" 2>/dev/null | grep -oE "$3" | head -1
}

# Contention levels:
# 1P+1C : 가장 낮은 contention
# 2P+2C : 중간
# 4P+4C : 높음
# 8P+8C : 매우 높음 (코어 수 초과)
CONFIGS=("1 1" "2 2" "4 4" "8 8")

for cfg in "${CONFIGS[@]}"; do
    read P C <<< "$cfg"
    for mode in pthread futex; do
        echo "============================================================"
        echo " P=$P C=$C  $mode  ($TRIALS trials, $ITEMS items/producer)"
        echo "============================================================"
        for t in $(seq 1 $TRIALS); do
            LOG="sync_logs/p${P}c${C}_${mode}_t${t}.log"
            $BIN -p $P -c $C -n $ITEMS -m $mode > "$LOG" 2>&1

            thr=$(  extract "$LOG" "Throughput"        '[0-9]+(\.[0-9]+)?')
            push99=$(extract "$LOG" "Push latency"     'p99=[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
            pop99=$( extract "$LOG" "Pop  latency"     'p99=[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
            blk_pct=$(grep "Sync blocked" "$LOG" | grep -oE '\([0-9]+\.[0-9]+%' | grep -oE '[0-9]+\.[0-9]+')
            fp=$(   extract "$LOG" "Fastpath rate"     '[0-9]+\.[0-9]+')
            cpop=$( extract "$LOG" "Total ctxt/op"     '[0-9]+\.[0-9]+')

            printf "  t %2d:  thr=%-12s  push_p99=%-7s  pop_p99=%-7s  fastpath=%-6s  blocked%%=%-6s  ctxt/op=%s\n" \
                $t "${thr:-?}" "${push99:-?}" "${pop99:-?}" "${fp:-?}" "${blk_pct:-?}" "${cpop:-?}"

            echo "$P,$C,$mode,$t,${thr:-0},${push99:-0},${pop99:-0},${blk_pct:-0},${fp:-0},${cpop:-0}" >> $CSV
        done
        echo ""
    done
done

# ============================================================
# 요약: (P+C, mode)별 평균
# ============================================================
echo "============================================================"
echo " 요약 비교 (mean ± std across $TRIALS trials)"
echo "============================================================"
echo ""
printf "%-7s %-8s %18s %14s %14s %12s %12s\n" \
    "P+C" "mode" "throughput" "push_p99(us)" "pop_p99(us)" "fastpath%" "ctxt/op"
printf "%-7s %-8s %18s %14s %14s %12s %12s\n" \
    "------" "-------" "------------------" "--------------" "--------------" "------------" "------------"

stat_col() {
    awk -F, -v p=$1 -v c=$2 -v m=$3 -v col=$4 '
        $1==p && $2==c && $3==m { sum+=$col; ss+=$col*$col; n++ }
        END { if (n>0) {
                avg=sum/n; v=(ss/n)-(avg*avg); if (v<0) v=0;
                printf "%.2f±%.2f", avg, sqrt(v)
            } else printf "n/a" }' $CSV
}

for cfg in "${CONFIGS[@]}"; do
    read P C <<< "$cfg"
    label="${P}P+${C}C"
    for mode in pthread futex; do
        printf "%-7s %-8s %18s %14s %14s %12s %12s\n" \
            "$label" "$mode" \
            "$(stat_col $P $C $mode 5)" \
            "$(stat_col $P $C $mode 6)" \
            "$(stat_col $P $C $mode 7)" \
            "$(stat_col $P $C $mode 9)" \
            "$(stat_col $P $C $mode 10)"
    done
done

echo ""
echo "============================================================"
echo " pthread vs futex 비교 (각 contention level에서)"
echo "============================================================"
echo ""
printf "%-7s %15s %15s %15s\n" "P+C" "throughput Δ" "push_p99 Δ" "ctxt/op Δ"
printf "%-7s %15s %15s %15s\n" "------" "---------------" "---------------" "---------------"
for cfg in "${CONFIGS[@]}"; do
    read P C <<< "$cfg"
    label="${P}P+${C}C"
    # Throughput diff: (futex - pthread) / pthread * 100
    thr_diff=$(awk -F, -v p=$P -v c=$C '
        $1==p && $2==c && $3=="pthread" { pt+=$5; pn++ }
        $1==p && $2==c && $3=="futex"   { ft+=$5; fn++ }
        END {
            if (pn>0 && fn>0) {
                pavg=pt/pn; favg=ft/fn;
                printf "%+.1f%%", (favg-pavg)/pavg*100
            } else printf "n/a"
        }' $CSV)
    p99_diff=$(awk -F, -v p=$P -v c=$C '
        $1==p && $2==c && $3=="pthread" { pt+=$6; pn++ }
        $1==p && $2==c && $3=="futex"   { ft+=$6; fn++ }
        END {
            if (pn>0 && fn>0) {
                pavg=pt/pn; favg=ft/fn;
                printf "%+.1f%%", (favg-pavg)/pavg*100
            } else printf "n/a"
        }' $CSV)
    ctxt_diff=$(awk -F, -v p=$P -v c=$C '
        $1==p && $2==c && $3=="pthread" { pt+=$10; pn++ }
        $1==p && $2==c && $3=="futex"   { ft+=$10; fn++ }
        END {
            if (pn>0 && fn>0) {
                pavg=pt/pn; favg=ft/fn;
                printf "%+.1f%%", (favg-pavg)/pavg*100
            } else printf "n/a"
        }' $CSV)
    printf "%-7s %15s %15s %15s\n" "$label" "$thr_diff" "$p99_diff" "$ctxt_diff"
done

echo ""
echo "원본 로그: sync_logs/*.log, CSV: $CSV"
echo ""
echo "[해석 가이드]"
echo "  - Throughput Δ +면 futex가 빠름. contention 클수록 차이 클 가능성"
echo "  - push_p99 Δ -면 futex tail latency 적음"
echo "  - ctxt/op Δ -면 futex가 OS scheduler 부담 적음"
echo "  - 1P+1C (낮은 contention)에서는 둘이 비슷, 8P+8C에서 차이 가장 클 가능성"

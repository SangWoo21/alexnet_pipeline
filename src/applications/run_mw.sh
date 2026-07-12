#!/bin/bash
# ============================================================
# AlexNet 멀티워커 futex 검증 스윕
#   CPU worker 수 (2,4,6) × 동기화 (pthread,futex) 비교
#   contention level에 따른 futex 효과 측정
# ============================================================
TRIALS=${1:-10}
FRAMES=${2:-100}
WARMUP=10
BIN=./alexnet_futex
LOGDIR=mw_logs
mkdir -p $LOGDIR

WORKERS=(2 4 6)

echo "=== 빌드 ==="
nvcc -O3 -arch=sm_72 -Xcompiler "-pthread" \
     alexnet_futex.cu -o alexnet_futex -lpthread
if [ $? -ne 0 ]; then echo "빌드 실패"; exit 1; fi
echo "빌드 완료"; echo ""

CSV=$LOGDIR/results.csv
echo "workers,mode,trial,fps,gpu_ms,cpu_ms,sync_attempts,sync_blocked,fastpath,futex_wait,futex_wake,worker_vol,checksum" > $CSV

for w in "${WORKERS[@]}"; do
    for mode in pthread futex; do
        echo "------------------------------------------------------------"
        echo "[workers=$w | $mode] $TRIALS trials"
        echo "------------------------------------------------------------"
        for t in $(seq 1 $TRIALS); do
            out=$($BIN -w $w -m $mode -f $FRAMES 2>/dev/null)
            echo "$out" > $LOGDIR/w${w}_${mode}_t${t}.log

            fps=$(echo "$out"  | grep "Pipeline FPS"   | sed 's/.*: \([0-9.]*\).*/\1/')
            gpu=$(echo "$out"  | grep "GPU Conv"       | sed 's/.*mean=\([0-9.]*\).*/\1/')
            cpu=$(echo "$out"  | grep "CPU FC"         | sed 's/.*mean=\([0-9.]*\).*/\1/')
            att=$(echo "$out"  | grep "sync_attempts"  | sed 's/.*: \([0-9]*\).*/\1/')
            blk=$(echo "$out"  | grep "sync_blocked"   | sed 's/.*: \([0-9]*\).*/\1/')
            fp=$(echo "$out"   | grep "fastpath_rate"  | sed 's/.*: \([0-9.]*\)%.*/\1/')
            fw=$(echo "$out"   | grep "futex_WAIT"     | sed 's/.*: \([0-9]*\).*/\1/')
            fwk=$(echo "$out"  | grep "futex_WAKE"     | sed 's/.*: \([0-9]*\).*/\1/')
            vol=$(echo "$out"  | grep "worker_vol_ctxt"| sed 's/.*: \([0-9]*\).*/\1/')
            cks=$(echo "$out"  | grep "Output_checksum"| sed 's/.*: \(.*\)/\1/')

            printf "  t%2d: FPS=%-7s gpu=%-7s cpu=%-7s fastpath=%-6s vol=%-6s fw=%-6s\n" \
                "$t" "$fps" "$gpu" "$cpu" "$fp" "$vol" "$fw"
            echo "$w,$mode,$t,$fps,$gpu,$cpu,$att,$blk,$fp,$fw,$fwk,$vol,$cks" >> $CSV
        done
        echo ""
    done
done

echo "============================================================"
echo " 요약 (평균 ± 표준편차)"
echo "============================================================"
python3 - <<'PYEOF'
import csv, statistics
from collections import defaultdict
rows = list(csv.DictReader(open("mw_logs/results.csv")))
def fnum(x):
    try: return float(x)
    except: return float('nan')
groups = defaultdict(list)
for r in rows: groups[(r['workers'], r['mode'])].append(r)

def ms(vals):
    vals=[v for v in vals if v==v]
    if not vals: return "n/a"
    if len(vals)==1: return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f}±{statistics.pstdev(vals):.2f}"

hdr=f"{'workers':<9}{'mode':<9}{'FPS':<14}{'cpu_ms':<12}{'fastpath%':<14}{'vol_ctxt':<14}{'futex_wait':<12}"
print(hdr); print("-"*len(hdr))
for key in sorted(groups.keys(), key=lambda k:(int(k[0]),k[1])):
    w,mode=key; g=groups[key]
    print(f"{w:<9}{mode:<9}"
          f"{ms([fnum(r['fps']) for r in g]):<14}"
          f"{ms([fnum(r['cpu_ms']) for r in g]):<12}"
          f"{ms([fnum(r['fastpath']) for r in g]):<14}"
          f"{ms([fnum(r['worker_vol']) for r in g]):<14}"
          f"{ms([fnum(r['futex_wait']) for r in g]):<12}")

print("\n=== pthread vs futex 비교 (worker별) ===")
print(f"{'workers':<9}{'FPS Δ':<12}{'vol_ctxt Δ':<14}")
print("-"*35)
for w in sorted(set(int(k[0]) for k in groups.keys())):
    w=str(w)
    pt=[fnum(r['fps']) for r in groups.get((w,'pthread'),[])]
    ft=[fnum(r['fps']) for r in groups.get((w,'futex'),[])]
    ptv=[fnum(r['worker_vol']) for r in groups.get((w,'pthread'),[])]
    ftv=[fnum(r['worker_vol']) for r in groups.get((w,'futex'),[])]
    if pt and ft:
        fps_d=(statistics.mean(ft)-statistics.mean(pt))/statistics.mean(pt)*100
        vol_d=(statistics.mean(ftv)-statistics.mean(ptv))/statistics.mean(ptv)*100 if statistics.mean(ptv)!=0 else 0
        print(f"{w:<9}{fps_d:+.1f}%{'':<6}{vol_d:+.1f}%")

print("\n[해석] worker가 늘수록 contention 증가 → futex 효과(vol_ctxt 감소) 커지는지 확인")
print("[검증] 모든 조건에서 checksum이 동일해야 'sync만 바뀜' 성립")
PYEOF

echo ""
echo "원본 로그: $LOGDIR/*.log, CSV: $CSV"

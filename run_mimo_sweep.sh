#!/usr/bin/env bash
#
# run_mimo_sweep.sh — UE 수 x MIMO 모드 성능 스윕 자동화
#
# 사용법:
#   sudo bash run_mimo_sweep.sh [-d 120] [-ue "4 8 16 32"] [--plot-only]
#
# 옵션:
#   -d SEC       각 실행의 데이터 수집 시간 (기본: 120초)
#   -ue LIST     UE 수 목록 (기본: "4 8 16 32")
#   --plot-only  시뮬레이션 건너뛰고 기존 manifest로 그래프만 재생성
#
# 결과:
#   logs/mimo_sweep_manifest.csv  — 실행 기록 (파이썬 플로팅용)
#   ~/oai_sionna_junxiu/figures_sweep/  — 그래프 (PNG + PDF)
#
set -euo pipefail

PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
PROXY_DIR="$PROJ_DIR/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy"
CONF="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
MANIFEST="$PROJ_DIR/logs/mimo_sweep_manifest.csv"
PLOT_SCRIPT="$PROJ_DIR/plot_mimo_sweep.py"

DURATION=120
UE_LIST="4 8 16 32"
PLOT_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -d)         DURATION="$2"; shift 2 ;;
        -ue)        UE_LIST="$2"; shift 2 ;;
        --plot-only) PLOT_ONLY=1; shift ;;
        -h|--help)
            head -16 "$0" | tail -14
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODES=("type2_mu" "type2_su" "type1_su")
MODE_LABELS=("Type-II_MU-MIMO" "Type-II_SU-MIMO" "Type-I_SU-MIMO")

REAL_USER="${SUDO_USER:-$USER}"

if [ "$PLOT_ONLY" -eq 1 ]; then
    echo "[sweep] --plot-only: 기존 manifest로 그래프만 생성"
    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: $MANIFEST 없음. 먼저 시뮬레이션을 실행하세요."
        exit 1
    fi
    sudo -u "$REAL_USER" python3 "$PLOT_SCRIPT" plot "$MANIFEST"
    exit 0
fi

N_UE_COUNT=$(echo "$UE_LIST" | wc -w)
TOTAL=$((N_UE_COUNT * ${#MODES[@]}))

echo "============================================================"
echo "  MIMO Sweep — UE수 x MIMO모드 성능 비교 자동화"
echo "  UE 수      : $UE_LIST"
echo "  MIMO 모드  : ${MODE_LABELS[*]}"
echo "  수집 시간  : ${DURATION}초/실행"
echo "  총 실행 수 : $TOTAL"
echo "  예상 시간  : ~$((TOTAL * (DURATION + 90) / 60))분"
echo "============================================================"
echo ""

mkdir -p "$(dirname "$MANIFEST")"
echo "num_ues,mode,mode_label,log_dir" > "$MANIFEST"

RUN_IDX=0

for N_UE in $UE_LIST; do
    for i in "${!MODES[@]}"; do
        MODE="${MODES[$i]}"
        LABEL="${MODE_LABELS[$i]}"
        RUN_IDX=$((RUN_IDX + 1))

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$RUN_IDX/$TOTAL] UE=$N_UE  모드=$LABEL"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python3 "$PLOT_SCRIPT" apply-config "$MODE" "$CONF"

        echo "[sweep] gnb.conf 적용 확인:"
        grep -E "codebook_type|mu_mimo\s*=" "$CONF" | grep -v "^.*#" | head -2

        BS=4
        [ "$N_UE" -ge 32 ] && BS=8

        cd "$PROXY_DIR"
        bash launch_all.sh \
            -n "$N_UE" \
            -ga 2 1 -ua 2 1 -pol dual \
            -cm static \
            -nf -70 \
            -bs "$BS" \
            -d "$DURATION" \
            2>&1 | tee "/tmp/sweep_run_${RUN_IDX}.log" || true

        LOG_DIR=$(readlink -f "$PROJ_DIR/logs/latest")
        echo "$N_UE,$MODE,$LABEL,$LOG_DIR" >> "$MANIFEST"
        echo "[sweep] [$RUN_IDX/$TOTAL] 완료 → $LOG_DIR"

        sleep 10
    done
done

echo ""
echo "============================================================"
echo "  전체 시뮬레이션 완료! ($TOTAL 회)"
echo "  매니페스트: $MANIFEST"
echo "============================================================"

python3 "$PLOT_SCRIPT" apply-config type2_mu "$CONF"
echo "[sweep] gnb.conf를 기본값(Type-II MU-MIMO)으로 복원"

echo ""
echo "[sweep] 그래프 생성 중... (사용자 환경: $REAL_USER)"
cd "$PROJ_DIR"
sudo -u "$REAL_USER" python3 "$PLOT_SCRIPT" plot "$MANIFEST"
echo ""
echo "[sweep] 모든 작업 완료!"

#!/bin/bash
# ================================================================
#  Baseline UE 수 Sweep — 3채널 × 3모드 × 5 UE수 = 45 실험
#
#  Usage: sudo bash run_baseline_sweep.sh [OPTIONS]
#    -d SEC      각 실험 실행 시간 (기본: 90)
#    -s START    시작 실험 번호 1~45 (기본: 1)
#    -e END      종료 실험 번호 (기본: 45)
#    -rx INDICES 고정 UE RX 인덱스 (쉼표 구분, UE수만큼 앞에서 자름)
#    --dry-run   설정만 출력
# ================================================================

set -euo pipefail

PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
EXP_DIR="$PROJ_DIR/graduation/experiments"
LAUNCH_SCRIPT="$PROJ_DIR/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/launch_multicell.sh"
TEMPLATE_PATH="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
RESULTS_DIR="$EXP_DIR/results"

DURATION=90
START_EXP=1
END_EXP=45
ALL_RX_INDICES="975,779,1265,1383,520,891,1102,330,1450,672,1288,445,1190,815,1355,712"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)  DURATION="$2"; shift 2 ;;
        -s)  START_EXP="$2"; shift 2 ;;
        -e)  END_EXP="$2"; shift 2 ;;
        -rx) ALL_RX_INDICES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)   echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

SCENARIOS=("UMi-LOS" "UMi-NLOS" "UMa-NLOS")
MODES=("type1_su" "type2_su" "type2_mu")
MODE_LABELS=("Type1_SU" "Type2_SU" "Type2_MU")
UE_COUNTS=(1 2 4 8 16)

declare -A MODE_CONF
MODE_CONF["type1_su"]="$EXP_DIR/gnb_type1_su.conf"
MODE_CONF["type2_su"]="$EXP_DIR/gnb_type2_su.conf"
MODE_CONF["type2_mu"]="$EXP_DIR/gnb_type2_mu.conf"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$RESULTS_DIR/sweep_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

get_rx_indices() {
    local n=$1
    echo "$ALL_RX_INDICES" | tr ',' '\n' | head -n "$n" | paste -sd ','
}

backup_conf() { cp "$TEMPLATE_PATH" "$TEMPLATE_PATH.bak_sweep"; }
restore_conf() { [ -f "$TEMPLATE_PATH.bak_sweep" ] && cp "$TEMPLATE_PATH.bak_sweep" "$TEMPLATE_PATH"; }

kill_all() {
    pkill -9 -f "nr-softmodem" 2>/dev/null || true
    pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
    docker exec oai_sionna_proxy bash -c '
        pkill -9 -f "v4_multicell" 2>/dev/null
        pkill -9 -f "v4\.py" 2>/dev/null
        pkill -9 -f "multiprocessing" 2>/dev/null
    ' 2>/dev/null || true
    sleep 5
}

trap 'restore_conf; kill_all; exit 1' SIGINT SIGTERM

TOTAL=$((${#SCENARIOS[@]} * ${#MODES[@]} * ${#UE_COUNTS[@]}))
log "================================================================"
log "  Baseline UE Sweep (총 ${TOTAL}개 실험)"
log "  각 실험: ${DURATION}초 | 범위: #${START_EXP} ~ #${END_EXP}"
log "================================================================"

backup_conf

EXP_NUM=0
for sc_idx in 0 1 2; do
    for mode_idx in 0 1 2; do
        for ue_idx in 0 1 2 3 4; do
            EXP_NUM=$((EXP_NUM + 1))
            [ "$EXP_NUM" -lt "$START_EXP" ] && continue
            [ "$EXP_NUM" -gt "$END_EXP" ] && continue

            SCENARIO="${SCENARIOS[$sc_idx]}"
            MODE="${MODES[$mode_idx]}"
            MODE_LABEL="${MODE_LABELS[$mode_idx]}"
            CONF="${MODE_CONF[$MODE]}"
            NUM_UES="${UE_COUNTS[$ue_idx]}"
            RX_SUB=$(get_rx_indices "$NUM_UES")

            EXP_TAG="${SCENARIO}_${MODE_LABEL}_${NUM_UES}UE"
            log ""
            log "── #${EXP_NUM}/${TOTAL}: ${EXP_TAG} ──"

            if [ "$DRY_RUN" -eq 1 ]; then
                log "  [DRY] conf=$(basename "$CONF") sc=$SCENARIO n=$NUM_UES rx=$RX_SUB"
                continue
            fi

            kill_all
            cp "$CONF" "$TEMPLATE_PATH"

            log "  실행 중... (${DURATION}s)"
            bash "$LAUNCH_SCRIPT" \
                -nc 1 -n "$NUM_UES" \
                -ga 2 1 -ua 1 1 -pol dual \
                -sc "$SCENARIO" \
                -d "$DURATION" \
                -rx "$RX_SUB" \
                > "$RESULTS_DIR/run_${EXP_NUM}_${EXP_TAG}.log" 2>&1 || true

            LATEST_LOG=$(readlink -f "$PROJ_DIR/logs/latest" 2>/dev/null || echo "")
            if [ -n "$LATEST_LOG" ] && [ -d "$LATEST_LOG" ]; then
                EXP_RESULT_DIR="$RESULTS_DIR/${EXP_NUM}_${EXP_TAG}"
                mkdir -p "$EXP_RESULT_DIR"
                cp -r "$LATEST_LOG"/* "$EXP_RESULT_DIR/" 2>/dev/null || true
                log "  완료 → ${EXP_RESULT_DIR}"
            else
                log "  WARNING: 로그 없음"
            fi

            kill_all
            sleep 8
        done
    done
done

restore_conf

log ""
log "================================================================"
log "  전체 Sweep 완료! 결과: ${RESULTS_DIR}/"
log "================================================================"

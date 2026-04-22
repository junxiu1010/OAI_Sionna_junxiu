#!/bin/bash
# ================================================================
#  Baseline 성능 비교 실험 — 3채널 × 3모드 = 9 실험
#
#  채널: UMi-LOS, UMi-NLOS, UMa-NLOS
#  모드: Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO
#
#  Usage: sudo bash run_baseline_experiments.sh [OPTIONS]
#    -d SEC      각 실험 실행 시간 (기본: 120)
#    -s START    시작 실험 번호 1~9 (기본: 1, 이어서 실행 시 유용)
#    -e END      종료 실험 번호 1~9 (기본: 9)
#    -rx INDICES 고정 UE RX 인덱스 (예: "975,779,1265,1383")
#    --dry-run   실제 실행 없이 설정만 출력
# ================================================================

set -euo pipefail

PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
EXP_DIR="$PROJ_DIR/graduation/experiments"
LAUNCH_SCRIPT="$PROJ_DIR/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/launch_multicell.sh"
TEMPLATE_PATH="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
RESULTS_DIR="$PROJ_DIR/graduation/experiments/results"

DURATION=120
START_EXP=1
END_EXP=9
UE_RX_INDICES=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)  DURATION="$2"; shift 2 ;;
        -s)  START_EXP="$2"; shift 2 ;;
        -e)  END_EXP="$2"; shift 2 ;;
        -rx) UE_RX_INDICES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)   echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

SCENARIOS=("UMi-LOS" "UMi-NLOS" "UMa-NLOS")
MODES=("type1_su" "type2_su" "type2_mu")
MODE_LABELS=("Type1_SU-MIMO" "Type2_SU-MIMO" "Type2_MU-MIMO")

declare -A MODE_CONF
MODE_CONF["type1_su"]="$EXP_DIR/gnb_type1_su.conf"
MODE_CONF["type2_su"]="$EXP_DIR/gnb_type2_su.conf"
MODE_CONF["type2_mu"]="$EXP_DIR/gnb_type2_mu.conf"

declare -A MODE_UES
MODE_UES["type1_su"]=4
MODE_UES["type2_su"]=4
MODE_UES["type2_mu"]=4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$RESULTS_DIR/batch_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

backup_conf() {
    cp "$TEMPLATE_PATH" "$TEMPLATE_PATH.bak_experiment"
}

restore_conf() {
    if [ -f "$TEMPLATE_PATH.bak_experiment" ]; then
        cp "$TEMPLATE_PATH.bak_experiment" "$TEMPLATE_PATH"
    fi
}

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

log "================================================================"
log "  Baseline 성능 비교 실험 배치 시작"
log "  각 실험 시간: ${DURATION}초"
log "  실험 범위: #${START_EXP} ~ #${END_EXP}"
log "  결과 디렉토리: ${RESULTS_DIR}"
log "================================================================"

backup_conf

EXP_NUM=0
for sc_idx in 0 1 2; do
    for mode_idx in 0 1 2; do
        EXP_NUM=$((EXP_NUM + 1))
        [ "$EXP_NUM" -lt "$START_EXP" ] && continue
        [ "$EXP_NUM" -gt "$END_EXP" ] && continue

        SCENARIO="${SCENARIOS[$sc_idx]}"
        MODE="${MODES[$mode_idx]}"
        MODE_LABEL="${MODE_LABELS[$mode_idx]}"
        CONF="${MODE_CONF[$MODE]}"
        NUM_UES="${MODE_UES[$MODE]}"

        EXP_TAG="${SCENARIO}_${MODE_LABEL}"
        log ""
        log "────────────────────────────────────────"
        log "  실험 #${EXP_NUM}/9: ${EXP_TAG}"
        log "    채널: ${SCENARIO}"
        log "    모드: ${MODE_LABEL}"
        log "    UE수: ${NUM_UES}"
        log "    설정: ${CONF}"
        log "────────────────────────────────────────"

        if [ "$DRY_RUN" -eq 1 ]; then
            log "  [DRY RUN] 건너뜀"
            continue
        fi

        kill_all

        log "  설정 파일 교체: $(basename "$CONF") → template"
        cp "$CONF" "$TEMPLATE_PATH"

        RX_ARGS=""
        [ -n "$UE_RX_INDICES" ] && RX_ARGS="-rx $UE_RX_INDICES"

        log "  launch_multicell.sh 실행 (${DURATION}초)..."
        sudo bash "$LAUNCH_SCRIPT" \
            -nc 1 \
            -n "$NUM_UES" \
            -ga 2 1 \
            -ua 1 1 \
            -pol dual \
            -sc "$SCENARIO" \
            -d "$DURATION" \
            $RX_ARGS \
            > "$RESULTS_DIR/run_${EXP_NUM}_${EXP_TAG}.log" 2>&1 || true

        log "  실험 #${EXP_NUM} 종료. 로그 수집 중..."

        LATEST_LOG=$(readlink -f "$PROJ_DIR/logs/latest" 2>/dev/null || echo "")
        if [ -n "$LATEST_LOG" ] && [ -d "$LATEST_LOG" ]; then
            EXP_RESULT_DIR="$RESULTS_DIR/${EXP_NUM}_${EXP_TAG}"
            mkdir -p "$EXP_RESULT_DIR"
            cp -r "$LATEST_LOG"/* "$EXP_RESULT_DIR/" 2>/dev/null || true
            log "  로그 복사 → ${EXP_RESULT_DIR}"
        else
            log "  WARNING: 로그 디렉토리를 찾을 수 없음"
        fi

        log "  쿨다운 (15초)..."
        kill_all
        sleep 10
    done
done

restore_conf

log ""
log "================================================================"
log "  전체 실험 완료!"
log "  결과: ${RESULTS_DIR}/"
log "================================================================"

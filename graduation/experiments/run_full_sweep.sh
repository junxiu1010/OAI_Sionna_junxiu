#!/bin/bash
# ================================================================
#  Full Baseline Sweep — 3채널 × 3모드 × 5 UE수 = 45 실험
#  Usage: bash run_full_sweep.sh [-d SEC] [-s START] [-e END] [--dry-run]
# ================================================================
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="$EXP_DIR/results"
SINGLE_RUNNER="$EXP_DIR/run_single_experiment.sh"

DURATION=90
START_EXP=1
END_EXP=45
DRY_RUN=0

ALL_RX="975,779,1265,1383,520,891,1102,330,1450,672,1288,445,1190,815,1355,712"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)  DURATION="$2"; shift 2 ;;
        -s)  START_EXP="$2"; shift 2 ;;
        -e)  END_EXP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)   echo "Unknown: $1"; exit 1 ;;
    esac
done

SCENARIOS=("UMi-LOS" "UMi-NLOS" "UMa-NLOS")
MODES=("type1_su" "type2_su" "type2_mu")
MODE_LABELS=("Type1_SU" "Type2_SU" "Type2_MU")
UE_COUNTS=(1 2 4 8 16)

declare -A MODE_CONF
MODE_CONF["type1_su"]="$EXP_DIR/gnb_type1_su.conf"
MODE_CONF["type2_su"]="$EXP_DIR/gnb_type2_su.conf"
MODE_CONF["type2_mu"]="$EXP_DIR/gnb_type2_mu.conf"

get_rx() {
    echo "$ALL_RX" | tr ',' '\n' | head -n "$1" | paste -sd ','
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RESULTS_BASE/sweep_${TIMESTAMP}.log"
mkdir -p "$RESULTS_BASE"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "Full Baseline Sweep | ${DURATION}s/exp | #${START_EXP}~#${END_EXP}"

EXP_NUM=0
for sc_idx in 0 1 2; do
    for mode_idx in 0 1 2; do
        for ue_idx in 0 1 2 3 4; do
            EXP_NUM=$((EXP_NUM + 1))
            [ "$EXP_NUM" -lt "$START_EXP" ] && continue
            [ "$EXP_NUM" -gt "$END_EXP" ] && continue

            SC="${SCENARIOS[$sc_idx]}"
            MODE="${MODES[$mode_idx]}"
            ML="${MODE_LABELS[$mode_idx]}"
            CONF="${MODE_CONF[$MODE]}"
            N="${UE_COUNTS[$ue_idx]}"
            RX=$(get_rx "$N")
            TAG="${SC}_${ML}_${N}UE"
            OUTDIR="$RESULTS_BASE/${EXP_NUM}_${TAG}"

            log "── #${EXP_NUM}/45: ${TAG} (conf=$(basename "$CONF"), rx=$RX)"

            if [ "$DRY_RUN" -eq 1 ]; then
                log "  [DRY] skip"
                continue
            fi

            bash "$SINGLE_RUNNER" \
                -sc "$SC" \
                -conf "$CONF" \
                -n "$N" \
                -d "$DURATION" \
                -o "$OUTDIR" \
                -rx "$RX" \
                2>&1 | tee -a "$LOG" || log "  WARN: experiment may have failed"

            log "  결과: $OUTDIR"
            sleep 5
        done
    done
done

log "전체 완료! 결과: $RESULTS_BASE/"

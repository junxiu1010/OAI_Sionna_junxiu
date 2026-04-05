#!/bin/bash
#
# SNR Sweep for Type-I vs Type-II PMI Correlation Verification
#
# Runs the OAI system at multiple SNR values for both Type-I and Type-II
# codebook configurations, collecting CSI channel logs for post-processing.
#
# Usage:
#   sudo bash tools/sweep_snr.sh [options]
#
# Options:
#   -s "SNR_LIST"    SNR values in dB (default: "0 5 10 15 20 25 30")
#   -d DURATION      Run duration per SNR point in seconds (default: 60)
#   -o OUTPUT_DIR    Output directory for collected logs (default: results/pmi_verify)
#   -c "TYPES"       Codebook types to test (default: "type1 type2")
#   --skip-type1     Skip Type-I runs
#   --skip-type2     Skip Type-II runs
#   -v VERSION       Proxy version (default: v8)
#   -h               Show help
#
# Prerequisites:
#   - OAI built with CSI_CHANNEL_LOG support in csi_rx.c
#   - Proxy and Docker containers ready
#
# Output:
#   results/pmi_verify/
#     type1_snr0.bin, type1_snr5.bin, ...
#     type2_snr0.bin, type2_snr5.bin, ...
#

set -euo pipefail

PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
LAUNCHER="$PROJ_DIR/vRAN_Socket/G1B_SingleUE_MIMO_Channel_Proxy/launch_all.sh"
GNB_CONF="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"

SNR_LIST="0 5 10 15 20 25 30"
DURATION=60
OUTPUT_DIR="$PROJ_DIR/tools/results"
CB_TYPES="type1 type2"
PROXY_VER="v8"

usage() {
    echo "SNR Sweep for PMI Correlation Verification"
    echo ""
    echo "Usage: sudo bash $0 [options]"
    echo ""
    echo "Options:"
    echo "  -s \"SNR_LIST\"  SNR values in dB (default: \"$SNR_LIST\")"
    echo "  -d DURATION    Seconds per SNR point (default: $DURATION)"
    echo "  -o OUTPUT_DIR  Output directory (default: $OUTPUT_DIR)"
    echo "  -c \"TYPES\"     Codebook types (default: \"$CB_TYPES\")"
    echo "  --skip-type1   Skip Type-I runs"
    echo "  --skip-type2   Skip Type-II runs"
    echo "  -v VERSION     Proxy version (default: $PROXY_VER)"
    echo "  -h             Show help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s) SNR_LIST="$2"; shift 2 ;;
        -d) DURATION="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -c) CB_TYPES="$2"; shift 2 ;;
        --skip-type1) CB_TYPES="${CB_TYPES//type1/}"; shift ;;
        --skip-type2) CB_TYPES="${CB_TYPES//type2/}"; shift ;;
        -v) PROXY_VER="$2"; shift 2 ;;
        -h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

set_codebook_type() {
    local cb_type="$1"
    if [ "$cb_type" = "type1" ]; then
        # TYPE1 블록: 주석 해제
        sed -i '/### TYPE1_BEGIN/,/### TYPE1_END/{/^[[:space:]]*###/!s/^[[:space:]]*#/        /}' "$GNB_CONF"
        # TYPE2 블록: 주석 처리
        sed -i '/### TYPE2_BEGIN/,/### TYPE2_END/{/^[[:space:]]*###/!{/^[[:space:]]*#/!s/^[[:space:]]*/        #/}}' "$GNB_CONF"
        echo "[sweep] gnb.conf -> Type-I (typeI_SinglePanel)"
    elif [ "$cb_type" = "type2" ]; then
        # TYPE1 블록: 주석 처리
        sed -i '/### TYPE1_BEGIN/,/### TYPE1_END/{/^[[:space:]]*###/!{/^[[:space:]]*#/!s/^[[:space:]]*/        #/}}' "$GNB_CONF"
        # TYPE2 블록: 주석 해제
        sed -i '/### TYPE2_BEGIN/,/### TYPE2_END/{/^[[:space:]]*###/!s/^[[:space:]]*#/        /}' "$GNB_CONF"
        echo "[sweep] gnb.conf -> Type-II (typeII_PortSelection)"
    fi
}

restore_codebook() {
    set_codebook_type "type1"
}

total_runs=0
for cb in $CB_TYPES; do
    for snr in $SNR_LIST; do
        total_runs=$((total_runs + 1))
    done
done

echo "============================================================"
echo "  PMI Verification SNR Sweep"
echo "    Codebook types : $CB_TYPES"
echo "    SNR values     : $SNR_LIST"
echo "    Duration/point : ${DURATION}s"
echo "    Total runs     : $total_runs"
echo "    Est. total time: $((total_runs * (DURATION + 30)))s"
echo "    Output dir     : $OUTPUT_DIR"
echo "    Proxy version  : $PROXY_VER"
echo "============================================================"
echo ""

run_idx=0
for cb in $CB_TYPES; do
    set_codebook_type "$cb"

    for snr in $SNR_LIST; do
        run_idx=$((run_idx + 1))
        LOG_FILE="$OUTPUT_DIR/${cb}_snr${snr}.bin"
        export CSI_CHANNEL_LOG="$LOG_FILE"

        echo ""
        echo "============================================================"
        echo "  [$run_idx/$total_runs] $cb, SNR=${snr} dB"
        echo "    CSI log: $LOG_FILE"
        echo "============================================================"

        rm -f "$LOG_FILE"

        bash "$LAUNCHER" \
            -v "$PROXY_VER" \
            -ga 2 2 -ua 2 1 \
            -snr "$snr" \
            -d "$DURATION" \
            || true

        sleep 5

        if [ -f "$LOG_FILE" ]; then
            SIZE=$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)
            echo "[sweep] $LOG_FILE -> ${SIZE} bytes"
        else
            echo "[sweep] WARNING: $LOG_FILE not created"
        fi
    done
done

restore_codebook

echo ""
echo "============================================================"
echo "  Sweep complete. Results in: $OUTPUT_DIR"
echo ""
echo "  To analyze:"
echo "    python3 $PROJ_DIR/tools/verify_type2_pmi.py \\"
echo "        $OUTPUT_DIR/type1_snr*.bin \\"
echo "        $OUTPUT_DIR/type2_snr*.bin"
echo "============================================================"

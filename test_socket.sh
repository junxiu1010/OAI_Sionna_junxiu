#!/bin/bash
set -euo pipefail

OAI_DIR="/home/dclcom57/oai_sionna_junxiu/openairinterface5g_whan"
BUILD_DIR="$OAI_DIR/cmake_targets/ran_build/build"
GNB_CONF="$OAI_DIR/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
LOG_DIR="/home/dclcom57/oai_sionna_junxiu/log/socket_test"

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

cleanup() {
    echo "Cleaning up..."
    kill $GNB_PID $UE_PID 2>/dev/null || true
    wait $GNB_PID $UE_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting gNB in pure socket mode..."
"$BUILD_DIR/nr-softmodem" \
    -O "$GNB_CONF" \
    --gNBs.[0].min_rxtxtime 6 \
    --rfsim \
    > "$LOG_DIR/gnb.log" 2>&1 &
GNB_PID=$!
echo "gNB PID: $GNB_PID"
sleep 8

echo "Starting UE in socket mode..."
"$BUILD_DIR/nr-uesoftmodem" \
    -r 106 --numerology 1 --band 78 -C 3619200000 \
    --uicc0.imsi 001010000000001 \
    --rfsim --rfsimulator.serverport 6014 \
    > "$LOG_DIR/ue0.log" 2>&1 &
UE_PID=$!
echo "UE PID: $UE_PID"

echo "Waiting 30s for UE to connect..."
sleep 30

echo ""
echo "=== UE0 KEY LOG ==="
grep -E "synch|PBCH|PRACH|RRC|Registr|Error decoding|SIB|Msg4|random_access|Connected|RACH|Initial sync" "$LOG_DIR/ue0.log" 2>/dev/null | head -30
echo ""
echo "=== gNB Frame.Slot ==="
grep -E "Frame.Slot|UE RNTI" "$LOG_DIR/gnb.log" 2>/dev/null | tail -5

#!/bin/bash
# ================================================================
#  단일 실험 실행기 — launch_multicell.sh 의존성 없이 직접 실행
#
#  Usage: bash run_single_experiment.sh \
#           -sc SCENARIO -conf CONF_FILE -n NUM_UES -d DURATION \
#           -o OUTPUT_DIR [-rx RX_INDICES]
# ================================================================

set -u

PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
BUILD_DIR="$PROJ_DIR/openairinterface5g_whan/cmake_targets/ran_build/build"
G1C_DIR="$PROJ_DIR/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy"
CONF_DIR="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF"

SCENARIO="UMa-NLOS"
CONF_FILE=""
NUM_UES=4
DURATION=90
OUTPUT_DIR=""
RX_INDICES=""

GNB_NX=2; GNB_NY=1; POL="dual"
UE_NX=1; UE_NY=1
GNB_ANT=4; UE_ANT=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        -sc)   SCENARIO="$2"; shift 2 ;;
        -conf) CONF_FILE="$2"; shift 2 ;;
        -n)    NUM_UES="$2"; shift 2 ;;
        -d)    DURATION="$2"; shift 2 ;;
        -o)    OUTPUT_DIR="$2"; shift 2 ;;
        -rx)   RX_INDICES="$2"; shift 2 ;;
        *)     echo "Unknown: $1"; exit 1 ;;
    esac
done

[ -z "$CONF_FILE" ] && { echo "ERROR: -conf required"; exit 1; }
[ -z "$OUTPUT_DIR" ] && { echo "ERROR: -o required"; exit 1; }
mkdir -p "$OUTPUT_DIR"

PIDS=()

cleanup() {
    if [ ${#PIDS[@]} -gt 0 ]; then
        for pid in "${PIDS[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    pkill -9 -f "nr-softmodem" 2>/dev/null || true
    pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
    docker exec oai_sionna_proxy bash -c '
        pkill -9 -f "v4_multicell" 2>/dev/null
        pkill -9 -f "v4\.py" 2>/dev/null
        pkill -9 -f "multiprocessing" 2>/dev/null
    ' 2>/dev/null || true
    sleep 3
}

trap cleanup EXIT

cleanup

rm -f /tmp/oai_gpu_ipc/gpu_ipc_shm* 2>/dev/null || true
mkdir -p /tmp/oai_gpu_ipc 2>/dev/null || true
chmod 777 /tmp/oai_gpu_ipc 2>/dev/null || true

echo "[EXP] Config: $(basename "$CONF_FILE") | Scenario: $SCENARIO | UEs: $NUM_UES | Duration: ${DURATION}s"

MULTICELL_CONF_DIR="/tmp/multicell_configs"
mkdir -p "$MULTICELL_CONF_DIR"
python3 "$G1C_DIR/generate_gnb_configs.py" \
    --template "$CONF_FILE" \
    --num-cells 1 \
    --output-dir "$MULTICELL_CONF_DIR" 2>/dev/null

PROXY_EXTRA=""
[ -n "$RX_INDICES" ] && PROXY_EXTRA="$PROXY_EXTRA --ue-rx-indices $RX_INDICES"

echo "[EXP] Starting proxy..."
docker exec \
    -e GPU_IPC_V5_GNB_ANT=$GNB_ANT \
    -e GPU_IPC_V5_GNB_NX=$GNB_NX \
    -e GPU_IPC_V5_GNB_NY=$GNB_NY \
    -e GPU_IPC_V5_UE_ANT=$UE_ANT \
    -e GPU_IPC_V5_UE_NX=$UE_NX \
    -e GPU_IPC_V5_UE_NY=$UE_NY \
    oai_sionna_proxy python3 -u \
    "/workspace/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/v4_multicell.py" \
    --mode gpu-ipc \
    --num-cells=1 \
    --ues-per-cell="$NUM_UES" \
    --gnb-ant=$GNB_ANT --ue-ant=$UE_ANT \
    --gnb-nx=$GNB_NX --gnb-ny=$GNB_NY \
    --ue-nx=$UE_NX --ue-ny=$UE_NY \
    --polarization=$POL \
    --scenario=$SCENARIO \
    --ici-atten-dB=15 \
    $PROXY_EXTRA \
    > "$OUTPUT_DIR/proxy.log" 2>&1 &
PIDS+=($!)
echo "[EXP] Proxy PID: ${PIDS[-1]}"

echo "[EXP] Waiting for proxy init..."
for i in $(seq 1 120); do
    if [ -f /tmp/oai_gpu_ipc/gpu_ipc_shm_cell0 ]; then
        chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm_cell* 2>/dev/null || true
    fi
    if grep -q "Entering main loop\|Pipeline ready\|All.*cells initialized" "$OUTPUT_DIR/proxy.log" 2>/dev/null; then
        chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm_cell* 2>/dev/null || true
        echo "[EXP] Proxy ready. (${i}s)"
        break
    fi
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
        echo "[EXP] ERROR: Proxy died"
        tail -20 "$OUTPUT_DIR/proxy.log"
        exit 1
    fi
    [ "$i" -eq 120 ] && { echo "[EXP] ERROR: Proxy timeout"; exit 1; }
    sleep 1
done

echo "[EXP] Starting gNB..."
CELL_CONF="$MULTICELL_CONF_DIR/gnb_cell0.conf"
(cd "$OUTPUT_DIR" && RFSIM_GPU_IPC_V7=1 RFSIM_GPU_IPC_CELL_IDX=0 \
    "$BUILD_DIR/nr-softmodem" \
    -O "$CELL_CONF" \
    --gNBs.[0].min_rxtxtime 3 --rfsim \
    --RUs.[0].nb_tx $GNB_ANT --RUs.[0].nb_rx $GNB_ANT \
    --gNBs.[0].pusch_AntennaPorts $GNB_ANT \
    --gNBs.[0].pdsch_AntennaPorts_XP 2 \
    --gNBs.[0].pdsch_AntennaPorts_N1 $((GNB_NX*GNB_NY)) \
    > gnb_cell0.log 2>&1) &
PIDS+=($!)
echo "[EXP] gNB PID: ${PIDS[-1]}"
sleep 8

echo "[EXP] Starting $NUM_UES UE(s)..."
for k in $(seq 0 $((NUM_UES - 1))); do
    IMSI=$(printf "00101000000%04d" $((k + 1)))
    (RFSIM_GPU_IPC_V7=1 RFSIM_GPU_IPC_CELL_IDX=0 RFSIM_GPU_IPC_UE_IDX=$k \
        "$BUILD_DIR/nr-uesoftmodem" \
        -r 106 --numerology 1 --band 78 -C 3619200000 \
        --uicc0.imsi "$IMSI" \
        --rfsim \
        --ue-nb-ant-tx $UE_ANT --ue-nb-ant-rx $UE_ANT \
        --uecap_file "$CONF_DIR/uecap_ports2.xml" \
        > "$OUTPUT_DIR/ue_cell0_ue${k}.log" 2>&1) &
    PIDS+=($!)
    echo "[EXP]   UE$k PID: ${PIDS[-1]} (IMSI=$IMSI)"
    sleep 2
done

echo "[EXP] Waiting for UE connections (60s max)..."
for i in $(seq 1 60); do
    connected=0
    for k in $(seq 0 $((NUM_UES - 1))); do
        if grep -q "RRC_CONNECTED\|NR_RRC_CONNECTED" "$OUTPUT_DIR/ue_cell0_ue${k}.log" 2>/dev/null; then
            connected=$((connected + 1))
        fi
    done
    if [ $connected -ge $NUM_UES ]; then
        echo "[EXP] All $NUM_UES UE(s) connected. (${i}s)"
        break
    fi
    [ "$i" -eq 60 ] && echo "[EXP] Partial: $connected/$NUM_UES connected"
    sleep 1
done

echo "[EXP] Running for ${DURATION}s..."
sleep "$DURATION"

echo "[EXP] Collecting stats..."
for f in "$OUTPUT_DIR"/nr*.log "$OUTPUT_DIR"/mu_mimo*.csv; do
    [ -f "$f" ] && echo "[EXP]   $(basename "$f"): $(wc -l < "$f") lines"
done

echo "[EXP] Done. Output: $OUTPUT_DIR/"

#!/bin/bash
#
# launch_multicell.sh — Multi-Cell MU-MIMO 통합 런처
#
# 사용법:
#   sudo bash launch_multicell.sh [옵션]
#
# 옵션:
#   -nc NUM_CELLS  셀 수                             (기본: 2)
#   -n  NUM_UES    셀당 UE 수                         (기본: 2)
#   -ga Nx Ny      gNB 안테나 (가로 x 세로)           (기본: 2 1)
#   -ua Nx Ny      UE 안테나 (가로 x 세로)            (기본: 1 1)
#   -pol MODE      편파: single, dual                 (기본: single)
#   -ici dB        셀간 간섭 감쇠 (dB, 기본: 15)
#   -d  SEC        실행 시간 (초)                      (기본: Ctrl+C까지)
#   -b             Sionna 채널 바이패스 (IQ 패스스루)
#   -h             도움말
#
# 예시:
#   sudo bash launch_multicell.sh -nc 2 -n 4 -ga 2 1 -ua 1 1 -pol dual -d 120
#   sudo bash launch_multicell.sh -nc 2 -n 4 -ga 2 1 -ua 1 1 -pol dual -p1b /workspace/vRAN_Socket/P1B_Valid_Results/Area1_7.5GHz_Rays_Valid_RXs.npz
#

set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────
NUM_CELLS=2
UES_PER_CELL=2
BYPASS_CHANNEL=0
FORCE_BYPASS=0
BYPASS_DURATION=0
GNB_NX=2
GNB_NY=1
UE_NX=1
UE_NY=1
POLARIZATION="single"
ICI_ATTEN_DB=15
DURATION=""
PATH_LOSS_DB=""
SNR_DB=""
NOISE_DBFS=""
SCENARIO="UMa-NLOS"
P1B_NPZ="/workspace/vRAN_Socket/P1B_Valid_Results/Area1_7.5GHz_Rays_Valid_RXs.npz"
UE_RX_INDICES=""
TIME_DILATION=""
CSINET_ENABLED=0
CSINET_MODE="baseline"
CSINET_GAMMA="0.25"
CSINET_SCENARIO="UMi_NLOS"
CSINET_PERIOD="20"
CSINET_PATH="/workspace/graduation/csinet"
CSINET_CHECKPOINT_DIR="/workspace/csinet_checkpoints"
CORE_EMULATOR=""

# ── 인자 파싱 ─────────────────────────────────────────────────────
usage() {
    echo "Multi-Cell MU-MIMO 통합 런처"
    echo ""
    echo "사용법: sudo bash launch_multicell.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  -nc NUM_CELLS  셀 수 (기본: 2)"
    echo "  -n  NUM_UES    셀당 UE 수 (기본: 2)"
    echo "  -ga Nx Ny      gNB 안테나 (기본: 2 1)"
    echo "  -ua Nx Ny      UE 안테나 (기본: 1 1)"
    echo "  -pol MODE      편파: single/dual (기본: single)"
    echo "  -ici dB        셀간 간섭 감쇠 dB (기본: 15, >100=off)"
    echo "  -pl dB         경로 손실 (기본: 0)"
    echo "  -snr dB        AWGN SNR (기본: off)"
    echo "  -nf dBFS       AWGN noise floor (기본: off)"
    echo "  -p1b PATH      P1B npz (컨테이너 내 경로, UE별 독립 채널)"
    echo "  -rx INDICES    UE RX 인덱스 (쉼표, 기본: random)"
    echo "  -sc SCENARIO   채널 시나리오: UMi-LOS|UMi-NLOS|UMa-LOS|UMa-NLOS (기본: UMa-NLOS)"
    echo "  -td FACTOR     시간 팽창 배수 (기본: 1.0=실시간, 10=10배 느림)"
    echo "  -d  SEC        실행 시간 (초, 기본: Ctrl+C까지)"
    echo "  -b             채널 바이패스 (--no-custom-channel)"
    echo "  -B             DL+UL 강제 바이패스 (채널 비활성화, IQ 패스스루)"
    echo "  -Bp SEC        Phase 모드: SEC초 바이패스 후 채널 활성화 (PDU 수립용)"
    echo "  -c  URL        Core Emulator 주소 (예: localhost:7100)"
    echo "  --csinet       CsiNet 활성화"
    echo "  --csinet-mode  CsiNet 모드: baseline/conditioned (기본: baseline)"
    echo "  -h             도움말"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -nc)  NUM_CELLS="$2"; shift 2 ;;
        -n)   UES_PER_CELL="$2"; shift 2 ;;
        -ga)  GNB_NX="$2"; GNB_NY="$3"; shift 3 ;;
        -ua)  UE_NX="$2"; UE_NY="$3"; shift 3 ;;
        -pol) POLARIZATION="$2"; shift 2 ;;
        -ici) ICI_ATTEN_DB="$2"; shift 2 ;;
        -pl)  PATH_LOSS_DB="$2"; shift 2 ;;
        -snr) SNR_DB="$2"; shift 2 ;;
        -nf)  NOISE_DBFS="$2"; shift 2 ;;
        -p1b) P1B_NPZ="$2"; shift 2 ;;
        -rx)  UE_RX_INDICES="$2"; shift 2 ;;
        -sc)  SCENARIO="$2"; shift 2 ;;
        -td)  TIME_DILATION="$2"; shift 2 ;;
        -d)   DURATION="$2"; shift 2 ;;
        -b)   BYPASS_CHANNEL=1; shift ;;
        -B)   FORCE_BYPASS=1; shift ;;
        -Bp)  BYPASS_DURATION="$2"; shift 2 ;;
        -c)   CORE_EMULATOR="$2"; shift 2 ;;
        --csinet)       CSINET_ENABLED=1; shift ;;
        --csinet-mode)  CSINET_MODE="$2"; shift 2 ;;
        -h)   usage ;;
        *)    echo "ERROR: 알 수 없는 옵션: $1"; usage ;;
    esac
done

# ── 안테나 계산 ───────────────────────────────────────────────────
POL_MULT=1
[ "$POLARIZATION" = "dual" ] && POL_MULT=2
GNB_ANT=$((GNB_NX * GNB_NY * POL_MULT))
UE_ANT=$((UE_NX * UE_NY * POL_MULT))
TOTAL_UES=$((NUM_CELLS * UES_PER_CELL))

# ── 경로 설정 ─────────────────────────────────────────────────────
PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
BUILD_DIR="$PROJ_DIR/openairinterface5g_whan/cmake_targets/ran_build/build"
TEMPLATE_CONF="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
CONF_DIR="$(dirname "$TEMPLATE_CONF")"
G1C_DIR="$PROJ_DIR/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy"
MULTICELL_CONF_DIR="/tmp/multicell_configs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ANT_TAG=""
[ "$GNB_ANT" -gt 1 ] && ANT_TAG="${ANT_TAG}_ga${GNB_NX}x${GNB_NY}"
[ "$UE_ANT" -gt 1 ] && ANT_TAG="${ANT_TAG}_ua${UE_NX}x${UE_NY}"
[ "$POLARIZATION" = "dual" ] && ANT_TAG="${ANT_TAG}_xp2"
LOG_TAG="MC_${NUM_CELLS}cell_${UES_PER_CELL}ue${ANT_TAG}"
LOG_DIR="$PROJ_DIR/logs/${TIMESTAMP}_${LOG_TAG}"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$PROJ_DIR/logs/latest"

PIDS=()

cleanup() {
    echo ""
    echo "[MC launcher] 전체 프로세스 종료 중..."
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    pkill -9 -f "nr-softmodem" 2>/dev/null || true
    pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
    docker exec oai_sionna_proxy bash -c '
        pkill -9 -f "v4_multicell" 2>/dev/null
        pkill -9 -f "v4\.py" 2>/dev/null
        pkill -9 -f "multiprocessing.spawn" 2>/dev/null
        pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
    ' 2>/dev/null || true
    sleep 2
    echo "[MC launcher] 종료 완료. 로그: $LOG_DIR/"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "============================================================"
echo "  Multi-Cell MU-MIMO 통합 런처"
echo "    셀 수    : ${NUM_CELLS}"
echo "    셀당 UE  : ${UES_PER_CELL} (총: ${TOTAL_UES})"
echo "    gNB 안테나: ${GNB_NX}x${GNB_NY}x${POL_MULT}pol = ${GNB_ANT}"
echo "    UE 안테나 : ${UE_NX}x${UE_NY}x${POL_MULT}pol = ${UE_ANT}"
echo "    편파      : ${POLARIZATION}"
echo "    ICI 감쇠  : ${ICI_ATTEN_DB} dB"
if [ "$FORCE_BYPASS" -eq 1 ]; then
echo "    채널      : DL+UL 강제 바이패스 (IQ 패스스루)"
elif [ "$BYPASS_DURATION" -gt 0 ] 2>/dev/null; then
echo "    채널      : PHASED (${BYPASS_DURATION}초 바이패스 → Sionna 채널)"
elif [ "$BYPASS_CHANNEL" -eq 1 ]; then
echo "    채널      : 바이패스 (--no-custom-channel)"
else
echo "    채널      : Sionna"
fi
[ -n "$DURATION" ] && echo "    시간      : ${DURATION}초 후 자동 종료"
if [ "$CSINET_ENABLED" -eq 1 ]; then
echo "    CsiNet    : 활성 (${CSINET_MODE}, gamma=${CSINET_GAMMA})"
else
echo "    CsiNet    : 비활성"
fi
[ -n "$CORE_EMULATOR" ] && echo "    Core      : ${CORE_EMULATOR} (연동)"
echo "    로그      : ${LOG_DIR}/"
echo "============================================================"

# ── 이전 프로세스 정리 ────────────────────────────────────────────
pkill -9 -f "nr-softmodem" 2>/dev/null || true
pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
docker exec oai_sionna_proxy bash -c '
    pkill -9 -f "v4_multicell" 2>/dev/null
    pkill -9 -f "v4\.py" 2>/dev/null
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null
    pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null
' 2>/dev/null || true
sleep 3
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | wc -l)
if [ "$GPU_PROCS" -gt 0 ]; then
    echo "[MC launcher] WARNING: GPU에 ${GPU_PROCS}개 프로세스 잔존 — 강제 종료 시도"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | while read pid; do
        kill -9 "$pid" 2>/dev/null || true
    done
    docker exec oai_sionna_proxy bash -c '
        for p in $(ps -eo pid,stat,cmd | grep "python3.*multiprocessing" | grep -v grep | awk "{print \$1}"); do
            kill -9 "$p" 2>/dev/null
        done
    ' 2>/dev/null || true
    sleep 3
fi
rm -f /tmp/oai_gpu_ipc/gpu_ipc_shm* 2>/dev/null || true
mkdir -p /tmp/oai_gpu_ipc 2>/dev/null || true
chmod 777 /tmp/oai_gpu_ipc 2>/dev/null || true
echo "[MC launcher] SHM 파일 정리 완료"
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
echo "[MC launcher] GPU 0 가용 메모리: ${GPU_FREE:-unknown} MiB"
sleep 2

# ── 1. gNB 설정 파일 생성 ─────────────────────────────────────────
echo "[MC launcher] ${NUM_CELLS}개 셀 설정 파일 생성 중..."
python3 "$G1C_DIR/generate_gnb_configs.py" \
    --template "$TEMPLATE_CONF" \
    --num-cells "$NUM_CELLS" \
    --output-dir "$MULTICELL_CONF_DIR"
echo "[MC launcher] 설정 파일 생성 완료 → ${MULTICELL_CONF_DIR}/"

# ── 2. Proxy 시작 ─────────────────────────────────────────────────
echo "[MC launcher] Multi-Cell Proxy 시작..."

PROXY_ENV_ARGS=""
PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_GNB_ANT=$GNB_ANT"
PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_GNB_NX=$GNB_NX"
PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_GNB_NY=$GNB_NY"
PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_UE_ANT=$UE_ANT"
PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_UE_NX=$UE_NX"
PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_UE_NY=$UE_NY"

# CsiNet environment variables
PROXY_CSINET_ENV=""
if [ -n "$CORE_EMULATOR" ]; then
    CORE_HOST="${CORE_EMULATOR%%:*}"
    CORE_PORT="${CORE_EMULATOR##*:}"
    CORE_HTTP_PORT=$((CORE_PORT + 1))
    _CSINET_JSON=$(curl -sf "http://${CORE_HOST}:${CORE_HTTP_PORT}/api/v1/csinet/env" 2>/dev/null) || true
    if [ -n "$_CSINET_JSON" ]; then
        _csinet_val() { echo "$_CSINET_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$1','$2'))"; }
        _CSINET_ENABLED=$(_csinet_val CSINET_ENABLED "0")
        if [ "$_CSINET_ENABLED" = "1" ]; then
            CSINET_ENABLED=1
            CSINET_MODE=$(_csinet_val CSINET_MODE "$CSINET_MODE")
            CSINET_GAMMA=$(_csinet_val CSINET_GAMMA "$CSINET_GAMMA")
            CSINET_SCENARIO=$(_csinet_val CSINET_SCENARIO "$CSINET_SCENARIO")
            CSINET_PERIOD=$(_csinet_val CSINET_PERIOD "$CSINET_PERIOD")
            CSINET_PATH=$(_csinet_val CSINET_PATH "$CSINET_PATH")
            CSINET_CHECKPOINT_DIR=$(_csinet_val CSINET_CHECKPOINT_DIR "$CSINET_CHECKPOINT_DIR")
        fi
    fi
fi
if [ "$CSINET_ENABLED" -eq 1 ]; then
    PROXY_CSINET_ENV="-e CSINET_ENABLED=1"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_MODE=$CSINET_MODE"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_GAMMA=$CSINET_GAMMA"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_SCENARIO=$CSINET_SCENARIO"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_PERIOD=$CSINET_PERIOD"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_PATH=$CSINET_PATH"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_CHECKPOINT_DIR=$CSINET_CHECKPOINT_DIR"
    echo "[MC launcher] CsiNet 활성화: mode=$CSINET_MODE, gamma=$CSINET_GAMMA, scenario=$CSINET_SCENARIO"
fi

PROXY_EXTRA_ARGS=""
[ "$BYPASS_CHANNEL" -eq 1 ] && PROXY_EXTRA_ARGS="--no-custom-channel"
[ "$FORCE_BYPASS" -eq 1 ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --force-bypass"
[ "$BYPASS_DURATION" -gt 0 ] 2>/dev/null && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --bypass-duration $BYPASS_DURATION"
[ -n "$PATH_LOSS_DB" ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --path-loss-dB $PATH_LOSS_DB"
[ -n "$SNR_DB" ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --snr-dB $SNR_DB"
[ -n "$NOISE_DBFS" ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --noise-dBFS $NOISE_DBFS"
[ -n "$P1B_NPZ" ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --p1b-npz $P1B_NPZ"
[ -n "$UE_RX_INDICES" ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --ue-rx-indices $UE_RX_INDICES"
[ -n "$TIME_DILATION" ] && PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --time-dilation $TIME_DILATION"

docker exec -i $PROXY_ENV_ARGS $PROXY_CSINET_ENV oai_sionna_proxy python3 -u \
    "/workspace/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/v4_multicell.py" \
    --mode gpu-ipc \
    --num-cells="$NUM_CELLS" \
    --ues-per-cell="$UES_PER_CELL" \
    --gnb-ant=$GNB_ANT --ue-ant=$UE_ANT \
    --gnb-nx=$GNB_NX --gnb-ny=$GNB_NY \
    --ue-nx=$UE_NX --ue-ny=$UE_NY \
    --polarization=$POLARIZATION \
    --scenario=$SCENARIO \
    --ici-atten-dB=$ICI_ATTEN_DB \
    --bs-height-m=${BS_HEIGHT_M:-25.0} \
    --ue-height-m=${UE_HEIGHT_M:-1.5} \
    --isd-m=${ISD_M:-500} \
    --min-ue-dist-m=${MIN_UE_DIST_M:-35} \
    --max-ue-dist-m=${MAX_UE_DIST_M:-500} \
    --shadow-fading-std-dB=${SHADOW_FADING_STD_DB:-6.0} \
    ${K_FACTOR_MEAN_DB:+--k-factor-mean-dB=$K_FACTOR_MEAN_DB} \
    ${K_FACTOR_STD_DB:+--k-factor-std-dB=$K_FACTOR_STD_DB} \
    $PROXY_EXTRA_ARGS \
    > "$LOG_DIR/proxy.log" 2>&1 &
PIDS+=($!)

echo "[MC launcher] Proxy PID: ${PIDS[-1]}"
echo "[MC launcher] Proxy 초기화 대기..."

tail -f "$LOG_DIR/proxy.log" 2>/dev/null &
TAIL_WAIT_PID=$!

SHM_FIXED=0
for i in $(seq 1 120); do
    if [ "$SHM_FIXED" -eq 0 ]; then
        _any_shm=0
        for shm_f in /tmp/oai_gpu_ipc/gpu_ipc_shm_cell*; do
            [ -f "$shm_f" ] && _any_shm=1 && break
        done
        if [ "$_any_shm" -eq 1 ]; then
            for shm_f in /tmp/oai_gpu_ipc/gpu_ipc_shm_cell*; do
                [ -f "$shm_f" ] && chmod 666 "$shm_f" 2>/dev/null || true
            done
            SHM_FIXED=1
            echo "[MC launcher] SHM 권한 수정 완료"
        fi
    fi
    if grep -q "Entering main loop\|Pipeline ready\|All.*cells initialized" "$LOG_DIR/proxy.log" 2>/dev/null; then
        if [ "$SHM_FIXED" -eq 0 ]; then
            for shm_f in /tmp/oai_gpu_ipc/gpu_ipc_shm_cell*; do
                [ -f "$shm_f" ] && chmod 666 "$shm_f" 2>/dev/null || true
            done
        fi
        kill $TAIL_WAIT_PID 2>/dev/null || true
        wait $TAIL_WAIT_PID 2>/dev/null || true
        echo "[MC launcher] Proxy 준비 완료."
        break
    fi
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
        kill $TAIL_WAIT_PID 2>/dev/null || true
        echo "[MC launcher] ERROR: Proxy 종료됨"
        cleanup
    fi
    if [ "$i" -eq 120 ]; then
        kill $TAIL_WAIT_PID 2>/dev/null || true
        echo "[MC launcher] ERROR: Proxy 120초 타임아웃"
        cleanup
    fi
    sleep 1
done

# ── 3. gNB 프로세스 시작 (셀별) ───────────────────────────────────
GPU_IPC_ENV="RFSIM_GPU_IPC_V7=1"

for c in $(seq 0 $((NUM_CELLS - 1))); do
    CELL_CONF="$MULTICELL_CONF_DIR/gnb_cell${c}.conf"
    echo "[MC launcher] gNB Cell ${c} 시작 (conf=${CELL_CONF})..."

    GNB_ANT_ARGS=""
    if [ "$GNB_ANT" -ge 2 ]; then
        GNB_ANT_ARGS="--RUs.[0].nb_tx $GNB_ANT --RUs.[0].nb_rx $GNB_ANT"
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pusch_AntennaPorts $GNB_ANT"
    fi
    GNB_SPATIAL=$((GNB_NX * GNB_NY))
    if [ "$POLARIZATION" = "dual" ] && [ "$GNB_ANT" -ge 4 ]; then
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pdsch_AntennaPorts_XP 2"
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pdsch_AntennaPorts_N1 $GNB_SPATIAL"
    elif [ "$GNB_ANT" -ge 4 ]; then
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pdsch_AntennaPorts_XP 1"
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pdsch_AntennaPorts_N1 $GNB_ANT"
    fi

    (cd "$LOG_DIR" && eval "$GPU_IPC_ENV RFSIM_GPU_IPC_CELL_IDX=$c" \
    "$BUILD_DIR/nr-softmodem" \
        -O "$CELL_CONF" \
        --gNBs.[0].min_rxtxtime 3 --rfsim \
        $GNB_ANT_ARGS \
        > "gnb_cell${c}.log" 2>&1) &
    PIDS+=($!)
    echo "[MC launcher] gNB Cell ${c} PID: ${PIDS[-1]}"
    sleep 5
done

# ── 4. UE 프로세스 시작 (셀별) ────────────────────────────────────
UE_SETTLE_TIMEOUT=60

_launch_one_ue() {
    local cell_idx="$1"
    local ue_idx="$2"
    local global_ue_idx=$((cell_idx * UES_PER_CELL + ue_idx))
    local imsi_suffix=$((global_ue_idx + 1))
    local imsi=$(printf "00101000000%04d" "$imsi_suffix")

    echo "[MC launcher] Cell ${cell_idx} UE ${ue_idx} 시작 (IMSI=${imsi}, global=${global_ue_idx})..."

    local UE_GPU_ENV="RFSIM_GPU_IPC_V7=1 RFSIM_GPU_IPC_CELL_IDX=$cell_idx RFSIM_GPU_IPC_UE_IDX=$ue_idx"

    local UE_ANT_ARGS=""
    if [ "$UE_ANT" -gt 1 ]; then
        local UECAP_FILE="uecap_ports2.xml"
        if [ "$UE_ANT" -gt 2 ]; then
            UECAP_FILE="uecap_ports4.xml"
        fi
        UE_ANT_ARGS="--ue-nb-ant-tx $UE_ANT --ue-nb-ant-rx $UE_ANT"
        UE_ANT_ARGS="$UE_ANT_ARGS --uecap_file $CONF_DIR/$UECAP_FILE"
    fi

    (cd "$LOG_DIR" && eval "$UE_GPU_ENV" \
    "$BUILD_DIR/nr-uesoftmodem" \
        -r 106 --numerology 1 --band 78 -C 3619200000 \
        --uicc0.imsi "$imsi" \
        --rfsim \
        $UE_ANT_ARGS \
        > "ue_cell${cell_idx}_ue${ue_idx}.log" 2>&1) &
    PIDS+=($!)
    echo "[MC launcher] Cell ${cell_idx} UE ${ue_idx} PID: ${PIDS[-1]}"
}

echo "[MC launcher] UE 프로세스 시작..."
for c in $(seq 0 $((NUM_CELLS - 1))); do
    for k in $(seq 0 $((UES_PER_CELL - 1))); do
        _launch_one_ue "$c" "$k"
        sleep 1
    done

    echo "[MC launcher] Cell ${c} UE 연결 대기 (최대 ${UE_SETTLE_TIMEOUT}s)..."
    waited=0
    while [ $waited -lt $UE_SETTLE_TIMEOUT ]; do
        settled=0
        for k in $(seq 0 $((UES_PER_CELL - 1))); do
            if grep -q "RRC_CONNECTED\|State = NR_RRC_CONNECTED" "$LOG_DIR/ue_cell${c}_ue${k}.log" 2>/dev/null; then
                settled=$((settled + 1))
            fi
        done
        if [ $settled -ge $UES_PER_CELL ]; then
            echo "[MC launcher] Cell ${c} 전원 연결 (${waited}s)"
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    [ $waited -ge $UE_SETTLE_TIMEOUT ] && \
        echo "[MC launcher] Cell ${c} 부분 연결 (${settled}/${UES_PER_CELL}, ${waited}s)"
    sleep 3
done

echo "[MC launcher] 전체 ${TOTAL_UES} UE(s) 기동 완료."

# ── 5. 모니터링 ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  전체 프로세스 실행 중 (${#PIDS[@]}개)"
echo ""
echo "  로그 확인 (별도 터미널):"
echo "    tail -f ${LOG_DIR}/proxy.log"
for c in $(seq 0 $((NUM_CELLS - 1))); do
    echo "    tail -f ${LOG_DIR}/gnb_cell${c}.log"
    for k in $(seq 0 $((UES_PER_CELL - 1))); do
        echo "    tail -f ${LOG_DIR}/ue_cell${c}_ue${k}.log"
    done
done
echo ""
echo "  바로가기: $PROJ_DIR/logs/latest/"
if [ -n "$DURATION" ]; then
echo "  ${DURATION}초 후 자동 종료 (또는 Ctrl+C)"
else
echo "  Ctrl+C로 전체 종료"
fi
echo "============================================================"

tail -f "$LOG_DIR/proxy.log" &
TAIL_PID=$!

if [ -n "$DURATION" ]; then
    echo "[MC launcher] ${DURATION}초 후 자동 종료 예정..."
    sleep "$DURATION"
    echo "[MC launcher] ${DURATION}초 경과 — 자동 종료"
    kill $TAIL_PID 2>/dev/null || true
    cleanup
else
    wait "${PIDS[0]}" 2>/dev/null || true
    kill $TAIL_PID 2>/dev/null || true
    cleanup
fi

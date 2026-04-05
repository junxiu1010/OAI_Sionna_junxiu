#!/bin/bash
#
# G1B 통합 런처 — Proxy + gNB + UE를 한번에 실행
#
# 사용법:
#   sudo bash launch_all.sh [옵션]
#
# 옵션:
#   -v VERSION   Proxy 버전: v0 이상                (기본: v0)
#   -m MODE      통신 모드: socket, gpu-ipc          (기본: gpu-ipc)
#   -n NUM_UES   UE 수: 1 이상                      (기본: 1)
#   -pl dB       경로 손실                           (기본: 0)
#   -snr dB      AWGN 상대 SNR                      (기본: off)
#   -nf dBFS     AWGN 절대 noise floor              (기본: off, 예: -40)
#   -d SEC       실행 시간 (초)                      (기본: Ctrl+C까지)
#   -b           Sionna 채널 바이패스 (IQ 패스스루)
#   -ga Nx Ny    gNB 안테나 (가로 x 세로)            (기본: 1 1)
#   -ua Nx Ny    UE 안테나 (가로 x 세로)             (기본: 1 1)
#   -h           도움말
#
# 예시:
#   sudo bash launch_all.sh                              # v0, gpu-ipc, 1 UE, SISO
#   sudo bash launch_all.sh -v v2 -m gpu-ipc -b          # v2, bypass, SISO
#   sudo bash launch_all.sh -v v2 -m gpu-ipc -b -ga 2 1  # v2, bypass, 2x1 MIMO gNB
#   sudo bash launch_all.sh -v v2 -m gpu-ipc -b -ga 2 1 -ua 2 1  # 2x1 대칭
#
# 로그 파일:
#   ~/DevChannelProxyJIN/logs/YYYYMMDD_HHMMSS_G1B_v0_ipc_1ue/
#   ~/DevChannelProxyJIN/logs/latest → 최근 실행 심링크
#
# 종료:
#   Ctrl+C → 전체 프로세스 자동 종료
#

set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────
PROXY_VER="v0"
MODE="gpu-ipc"
NUM_UES=1
BYPASS_CHANNEL=0
PATH_LOSS_DB=""
SNR_DB=""
NOISE_DBFS=""
DURATION=""
GNB_NX=1
GNB_NY=1
UE_NX=1
UE_NY=1

# ── 인자 파싱 ─────────────────────────────────────────────────────
usage() {
    echo "G1B 통합 런처 — Single-UE MIMO Channel Proxy"
    echo ""
    echo "사용법: sudo bash launch_all.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  -v VERSION   Proxy 버전: v0 이상         (기본: v0)"
    echo "  -m MODE      모드: socket, gpu-ipc       (기본: gpu-ipc)"
    echo "  -n NUM_UES   UE 수: 1 이상              (기본: 1)"
    echo "  -pl dB       경로 손실 (기본: 0, 예: 3)"
    echo "  -snr dB      AWGN 상대 SNR (기본: off, 예: 30)"
    echo "  -nf dBFS     AWGN 절대 noise floor (기본: off, 예: -40)"
    echo "  -d SEC       실행 시간 (초, 기본: Ctrl+C까지)"
    echo "  -b           Sionna 채널 바이패스 (IQ 패스스루)"
    echo "  -ga Nx Ny    gNB 안테나 배열 (기본: 1 1 = SISO)"
    echo "  -ua Nx Ny    UE 안테나 배열 (기본: 1 1 = SISO)"
    echo "  -h           도움말"
    echo ""
    echo "버전 설명:"
    echo "  v0  G0 v12 복사 (SISO 1UE 기준선, GPU IPC V1)"
    echo "  v1  GPU IPC V5 circular buffer (DL/UL 대칭 채널, timestamp polling)"
    echo "  v2  GPU IPC V6 per-buffer antenna (MIMO bypass, 비대칭 안테나 지원)"
    echo "  v3  V6 + cir_time head fix + gNB antenna port CLI"
    echo "  v4  V6 + MIMO Channel Application (einsum, DL/UL pipeline)"
    echo "  v5  V6 + CUDA Graph (broadcast*sum, GPU index array)"
    echo "  v6  V6 + Dual Noise (absolute dBFS / relative SNR)"
    echo ""
    echo "예시:"
    echo "  sudo bash launch_all.sh                                # v0, SISO"
    echo "  sudo bash launch_all.sh -v v2 -m gpu-ipc -b           # v2, bypass SISO"
    echo "  sudo bash launch_all.sh -v v2 -m gpu-ipc -b -ga 2 1   # v2, gNB 2x1"
    echo "  sudo bash launch_all.sh -v v2 -b -ga 2 1 -ua 2 1      # v2, 대칭 2x1"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v)
            PROXY_VER="$2"
            if [[ ! "$PROXY_VER" =~ ^v[0-9]+$ ]]; then
                echo "ERROR: -v 는 v0 이상이어야 합니다 (입력: $PROXY_VER)"
                exit 1
            fi
            shift 2 ;;
        -m)
            MODE="$2"
            if [[ "$MODE" != "socket" && "$MODE" != "gpu-ipc" ]]; then
                echo "ERROR: -m 은 socket 또는 gpu-ipc 여야 합니다 (입력: $MODE)"
                exit 1
            fi
            shift 2 ;;
        -n)
            NUM_UES="$2"
            if ! [[ "$NUM_UES" =~ ^[0-9]+$ ]] || [ "$NUM_UES" -lt 1 ]; then
                echo "ERROR: -n 은 1 이상 정수여야 합니다 (입력: $NUM_UES)"
                exit 1
            fi
            shift 2 ;;
        -pl)
            _pl_raw="$2"
            _pl_abs="${_pl_raw#-}"
            PATH_LOSS_DB="-${_pl_abs}"
            shift 2 ;;
        -snr)
            SNR_DB="$2"
            shift 2 ;;
        -nf)
            NOISE_DBFS="$2"
            shift 2 ;;
        -d)
            DURATION="$2"
            shift 2 ;;
        -b) BYPASS_CHANNEL=1; shift ;;
        -ga)
            GNB_NX="$2"; GNB_NY="$3"
            shift 3 ;;
        -ua)
            UE_NX="$2"; UE_NY="$3"
            shift 3 ;;
        -h) usage ;;
        *) echo "ERROR: 알 수 없는 옵션: $1"; usage ;;
    esac
done

# ── 상호 배타 체크 ────────────────────────────────────────────────
if [ -n "$SNR_DB" ] && [ -n "$NOISE_DBFS" ]; then
    echo "ERROR: -snr 과 -nf 는 동시에 사용할 수 없습니다."
    exit 1
fi

# ── 안테나 계산 ───────────────────────────────────────────────────
GNB_ANT=$((GNB_NX * GNB_NY))
UE_ANT=$((UE_NX * UE_NY))
MAX_ANT=$((GNB_ANT > UE_ANT ? GNB_ANT : UE_ANT))

PROXY_SCRIPT="${PROXY_VER}.py"

# ── 경로 설정 ─────────────────────────────────────────────────────
PROJ_DIR="/home/dclcom57/DevChannelProxyJIN"
BUILD_DIR="$PROJ_DIR/openairinterface5g_whan/cmake_targets/ran_build/build"
CONF="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
CONF_DIR="$(dirname "$CONF")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODE_SHORT_LOG="ipc"
[ "$MODE" = "socket" ] && MODE_SHORT_LOG="sock"
ANT_TAG=""
[ "$GNB_ANT" -gt 1 ] && ANT_TAG="${ANT_TAG}_ga${GNB_NX}x${GNB_NY}"
[ "$UE_ANT" -gt 1 ] && ANT_TAG="${ANT_TAG}_ua${UE_NX}x${UE_NY}"
LOG_TAG_DIR="G1B_${PROXY_VER}_${MODE_SHORT_LOG}_${NUM_UES}ue${ANT_TAG}"
[ -n "$PATH_LOSS_DB" ] && LOG_TAG_DIR="${LOG_TAG_DIR}_pl${PATH_LOSS_DB#-}"
[ -n "$SNR_DB" ] && LOG_TAG_DIR="${LOG_TAG_DIR}_snr${SNR_DB}"
[ -n "$NOISE_DBFS" ] && LOG_TAG_DIR="${LOG_TAG_DIR}_nf${NOISE_DBFS}"
LOG_DIR="$PROJ_DIR/logs/${TIMESTAMP}_${LOG_TAG_DIR}"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$PROJ_DIR/logs/latest"

PIDS=()

cleanup() {
    echo ""
    echo "[launcher] Ctrl+C 감지 — 전체 프로세스 종료 중..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    pkill -9 -f "nr-softmodem" 2>/dev/null || true
    pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
    docker exec sionna-proxy pkill -9 -f "v[0-9]" 2>/dev/null || true
    echo "[launcher] 종료 완료. 로그: $LOG_DIR/"
    echo "[launcher] 바로가기: $PROJ_DIR/logs/latest/"
    exit 0
}

trap cleanup SIGINT SIGTERM

PROXY_EXTRA_ARGS=""
if [ "$BYPASS_CHANNEL" -eq 1 ]; then
    PROXY_EXTRA_ARGS="--no-custom-channel"
fi
if [ -n "$PATH_LOSS_DB" ]; then
    PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --path-loss-dB $PATH_LOSS_DB"
fi
if [ -n "$SNR_DB" ]; then
    PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --snr-dB $SNR_DB"
fi
if [ -n "$NOISE_DBFS" ]; then
    PROXY_EXTRA_ARGS="$PROXY_EXTRA_ARGS --noise-dBFS $NOISE_DBFS"
fi

echo "============================================================"
echo "  G1B 통합 런처 — Single-UE MIMO Channel Proxy"
echo "    Proxy : ${PROXY_SCRIPT}"
echo "    모드  : ${MODE}"
echo "    UE 수 : ${NUM_UES}"
echo "    gNB 안테나 : ${GNB_NX}x${GNB_NY} = ${GNB_ANT}"
echo "    UE 안테나  : ${UE_NX}x${UE_NY} = ${UE_ANT}"
if [ "$BYPASS_CHANNEL" -eq 1 ]; then
echo "    채널  : 바이패스 (IQ 패스스루)"
else
echo "    채널  : Sionna (custom channel)"
fi
[ -n "$PATH_LOSS_DB" ] && echo "    PL    : ${PATH_LOSS_DB#-} dB (internal: ${PATH_LOSS_DB} dB)"
[ -n "$SNR_DB" ] && echo "    Noise : 상대 SNR = ${SNR_DB} dB"
[ -n "$NOISE_DBFS" ] && echo "    Noise : 절대 floor = ${NOISE_DBFS} dBFS"
[ -n "$DURATION" ] && echo "    시간  : ${DURATION}초 후 자동 종료"
echo "    로그  : ${LOG_DIR}/"
if [ -n "$DURATION" ]; then
echo "  종료: ${DURATION}초 후 자동"
else
echo "  종료: Ctrl+C"
fi
echo "============================================================"

# ── 이전 프로세스 정리 ────────────────────────────────────────────
pkill -9 -f "nr-softmodem" 2>/dev/null || true
pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
docker exec sionna-proxy pkill -9 -f "v[0-9]" 2>/dev/null || true
rm -f /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
mkdir -p /tmp/oai_gpu_ipc 2>/dev/null || true
chmod 777 /tmp/oai_gpu_ipc 2>/dev/null || true
sleep 2

# ── Socket 모드: gNB를 먼저 시작 (gNB가 서버 역할) ──────────────
if [ "$MODE" = "socket" ]; then
    echo "[launcher] [socket] gNB 먼저 시작 (서버 port 6013)..."
    (cd "$LOG_DIR" && "$BUILD_DIR/nr-softmodem" \
        -O "$CONF" \
        --gNBs.[0].min_rxtxtime 6 --rfsim \
        --rfsimulator.serverport 6013 \
        > gnb.log 2>&1) &
    PIDS+=($!)
    echo "[launcher] gNB PID: ${PIDS[-1]}"

    echo "[launcher] gNB 서버 대기 (rfsimulator listen)..."
    for i in $(seq 1 30); do
        if grep -q "Running as server" "$LOG_DIR/gnb.log" 2>/dev/null; then
            echo "[launcher] gNB 서버 준비 완료."
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "[launcher] WARN: gNB 서버 대기 30초 초과, 계속 진행..."
        fi
        sleep 1
    done
fi

# ── 1. Proxy ────────────────────────────────────────────────────
echo "[launcher] Proxy 시작 (${PROXY_SCRIPT}, ${MODE})..."

PROXY_ENV_ARGS=""
if [[ "$PROXY_VER" =~ ^v[2-9]$ ]] || [[ "$PROXY_VER" =~ ^v[0-9][0-9]+$ ]]; then
    PROXY_ENV_ARGS="-e GPU_IPC_V5_GNB_ANT=$GNB_ANT"
    PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_GNB_NX=$GNB_NX"
    PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_GNB_NY=$GNB_NY"
    PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_UE_ANT=$UE_ANT"
    PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_UE_NX=$UE_NX"
    PROXY_ENV_ARGS="$PROXY_ENV_ARGS -e GPU_IPC_V5_UE_NY=$UE_NY"
fi

docker exec -i $PROXY_ENV_ARGS sionna-proxy python3 -u \
    "/workspace/vRAN_Socket/G1B_MultiUE_MIMO_Channel_Proxy/${PROXY_SCRIPT}" \
    --mode="$MODE" $PROXY_EXTRA_ARGS \
    --gnb-ant=$GNB_ANT --ue-ant=$UE_ANT \
    --gnb-nx=$GNB_NX --gnb-ny=$GNB_NY \
    --ue-nx=$UE_NX --ue-ny=$UE_NY \
    > "$LOG_DIR/proxy.log" 2>&1 &
PIDS+=($!)

echo "[launcher] Proxy 초기화 대기 (출력 실시간 표시)..."
echo "------------------------------------------------------------"
tail -f "$LOG_DIR/proxy.log" 2>/dev/null &
TAIL_WAIT_PID=$!

SHM_FIXED=0
for i in $(seq 1 120); do
    if [ "$SHM_FIXED" -eq 0 ] && [ -f /tmp/oai_gpu_ipc/gpu_ipc_shm ]; then
        docker exec sionna-proxy chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
        SHM_FIXED=1
        echo "[launcher] GPU IPC SHM 권한 수정 완료 (docker exec chmod)"
    fi
    if grep -q "Entering main loop\|Pipeline ready" "$LOG_DIR/proxy.log" 2>/dev/null; then
        if [ "$SHM_FIXED" -eq 0 ] && [ -f /tmp/oai_gpu_ipc/gpu_ipc_shm ]; then
            docker exec sionna-proxy chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
        fi
        kill $TAIL_WAIT_PID 2>/dev/null || true
        wait $TAIL_WAIT_PID 2>/dev/null || true
        echo "------------------------------------------------------------"
        echo "[launcher] Proxy 준비 완료."
        break
    fi
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
        kill $TAIL_WAIT_PID 2>/dev/null || true
        echo "[launcher] ERROR: Proxy 프로세스가 종료됨."
        cleanup
    fi
    if [ "$i" -eq 120 ]; then
        kill $TAIL_WAIT_PID 2>/dev/null || true
        echo "[launcher] ERROR: Proxy가 120초 내에 시작되지 않음."
        cleanup
    fi
    sleep 1
done

# ── 2. gNB (GPU-IPC 모드에서만 여기서 시작) ─────────────────────
if [ "$MODE" != "socket" ]; then
    if [[ "$PROXY_VER" == "v0" ]]; then
        GPU_IPC_ENV="RFSIM_GPU_IPC=1"
        echo "[launcher] gNB 시작 (GPU IPC V1)..."
    elif [[ "$PROXY_VER" == "v1" ]]; then
        GPU_IPC_ENV="RFSIM_GPU_IPC_V5=1"
        echo "[launcher] gNB 시작 (GPU IPC V5 circular buffer)..."
    else
        GPU_IPC_ENV="RFSIM_GPU_IPC_V6=1"
        echo "[launcher] gNB 시작 (GPU IPC V6 per-buffer antenna, ant=${GNB_ANT})..."
    fi

    GNB_ANT_ARGS=""
    if [ "$GNB_ANT" -ge 2 ]; then
        GNB_ANT_ARGS="--RUs.[0].nb_tx $GNB_ANT --RUs.[0].nb_rx $GNB_ANT"
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pdsch_AntennaPorts_XP 2"
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pusch_AntennaPorts $GNB_ANT"
    fi
    if [ "$GNB_ANT" -ge 4 ]; then
        GNB_ANT_ARGS="$GNB_ANT_ARGS --gNBs.[0].pdsch_AntennaPorts_N1 2"
    fi

    (cd "$LOG_DIR" && eval "$GPU_IPC_ENV" \
    "$BUILD_DIR/nr-softmodem" \
        -O "$CONF" \
        --gNBs.[0].min_rxtxtime 6 --rfsim \
        $GNB_ANT_ARGS \
        > gnb.log 2>&1) &
    PIDS+=($!)
    echo "[launcher] gNB PID: ${PIDS[-1]}"
    sleep 3
fi

# ── 3. UE 0 ~ N-1 ──────────────────────────────────────────────
for ue_idx in $(seq 0 $((NUM_UES - 1))); do
    imsi_suffix=$((ue_idx + 1))
    imsi=$(printf "00101000000%04d" "$imsi_suffix")

    echo "[launcher] UE ${ue_idx} 시작 (IMSI=${imsi})..."

    if [ "$MODE" = "socket" ]; then
        (cd "$LOG_DIR" && "$BUILD_DIR/nr-uesoftmodem" \
            -r 106 --numerology 1 --band 78 -C 3619200000 \
            --uicc0.imsi "$imsi" \
            --rfsim --rfsimulator.serverport 6014 \
            > "ue${ue_idx}.log" 2>&1) &
    else
        if [[ "$PROXY_VER" == "v0" ]]; then
            UE_GPU_ENV="RFSIM_GPU_IPC=1"
        elif [[ "$PROXY_VER" == "v1" ]]; then
            UE_GPU_ENV="RFSIM_GPU_IPC_V5=1"
        else
            UE_GPU_ENV="RFSIM_GPU_IPC_V6=1"
        fi

        UE_ANT_ARGS=""
        if [ "$UE_ANT" -gt 1 ]; then
            UECAP_FILE="uecap_ports2.xml"
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
            > "ue${ue_idx}.log" 2>&1) &
    fi
    PIDS+=($!)
    echo "[launcher] UE ${ue_idx} PID: ${PIDS[-1]}"
    sleep 2
done

# ── 4. 모니터링 ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  전체 프로세스 실행 중 (${#PIDS[@]}개)"
echo "  메인 터미널: Proxy 로그 실시간 표시"
echo ""
echo "  gNB/UE 로그 확인 (별도 터미널에서):"
echo "    tail -f ${LOG_DIR}/gnb.log"
for ue_idx in $(seq 0 $((NUM_UES - 1))); do
    echo "    tail -f ${LOG_DIR}/ue${ue_idx}.log"
done
echo ""
echo "  OAI 통계 확인 (별도 터미널에서):"
echo "    cat ${LOG_DIR}/nrMAC_stats.log       # CQI/PMI/RSRP/BLER/MCS"
echo "    cat ${LOG_DIR}/nrRRC_stats.log       # 연결된 DU/UE 목록"
echo "    cat ${LOG_DIR}/nrL1_stats.log        # PRB I0, PRACH I0"
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
    echo "[launcher] ${DURATION}초 후 자동 종료 예정..."
    sleep "$DURATION"
    echo ""
    echo "[launcher] ${DURATION}초 경과 — 자동 종료"
    kill $TAIL_PID 2>/dev/null || true
    cleanup
else
    wait "${PIDS[0]}" 2>/dev/null || true
    kill $TAIL_PID 2>/dev/null || true
    cleanup
fi

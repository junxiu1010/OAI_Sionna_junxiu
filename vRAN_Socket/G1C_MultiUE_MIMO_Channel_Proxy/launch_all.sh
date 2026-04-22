#!/bin/bash
#
# G1C 통합 런처 — Multi-UE MIMO Channel Proxy + gNB + UE(s) 한번에 실행
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
#   -cm MODE     채널 모드: dynamic, static           (기본: dynamic)
#   -bs N        UE 배치 기동 크기                    (기본: 4)
#   -ga Nx Ny    gNB 안테나 (가로 x 세로)            (기본: 1 1)
#   -ua Nx Ny    UE 안테나 (가로 x 세로)             (기본: 1 1)
#   -pol MODE    편파 모드: single, dual              (기본: single)
#   -c  URL      Core Emulator 주소 (예: localhost:7100)
#   -h           도움말
#
# 예시:
#   sudo bash launch_all.sh                              # v0, gpu-ipc, 1 UE, SISO
#   sudo bash launch_all.sh -n 2 -b                      # 2 UEs, bypass
#   sudo bash launch_all.sh -n 2 -ga 2 1 -ua 2 1         # 2 UEs, 2x1 MIMO
#   sudo bash launch_all.sh -n 16 -cm static -ga 2 1 -ua 2 1  # 16 UEs, static channel
#   sudo bash launch_all.sh -n 64 -cm static -bs 8 -ga 2 1 -ua 2 1  # 64 UEs
#   sudo bash launch_all.sh -ga 2 1 -ua 2 1 -pol dual               # 4T4R cross-pol (XP=2)
#   sudo bash launch_all.sh -c localhost:7100                       # Core Emulator 연동
#
# 로그 파일:
#   ~/DevChannelProxyJIN/logs/YYYYMMDD_HHMMSS_G1C_v0_ipc_2ue/
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
CHANNEL_MODE="dynamic"
BATCH_SIZE=4
PATH_LOSS_DB=""
SNR_DB=""
NOISE_DBFS=""
DURATION=""
GNB_NX=1
GNB_NY=1
UE_NX=1
UE_NY=1
POLARIZATION="single"
CORE_EMULATOR=""
ENABLE_ANALYZER=0
P1B_NPZ=""
UE_RX_INDICES=""

# ── 인자 파싱 ─────────────────────────────────────────────────────
usage() {
    echo "G1C 통합 런처 — Multi-UE MIMO Channel Proxy"
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
    echo "  -cm MODE     채널 모드: dynamic, static     (기본: dynamic)"
    echo "  -bs N        UE 배치 기동 크기              (기본: 4)"
    echo "  -ga Nx Ny    gNB 안테나 배열 (기본: 1 1 = SISO)"
    echo "  -ua Nx Ny    UE 안테나 배열 (기본: 1 1 = SISO)"
    echo "  -pol MODE    편파: single (V, XP=1) or dual (cross ±45°, XP=2)"
    echo "  -p1b FILE    P1B npz 파일 경로 (UE별 독립 ray-tracing 채널)"
    echo "  -rx INDICES  UE별 RX 인덱스 (콤마 구분, 'random', 기본: random)"
    echo "  -c  URL      Core Emulator 주소 (예: localhost:7100)"
    echo "  -a           MU-MIMO Analyzer 활성화 (프리코딩 mismatch 분석)"
    echo "  -h           도움말"
    echo ""
    echo "버전 설명:"
    echo "  v0  G1B v8 기반 Multi-UE (DL broadcast, UL superposition)"
    echo ""
    echo "예시:"
    echo "  sudo bash launch_all.sh                        # 1 UE, SISO"
    echo "  sudo bash launch_all.sh -n 2 -b                # 2 UEs, bypass"
    echo "  sudo bash launch_all.sh -n 2 -ga 2 1 -ua 2 1   # 2 UEs, 2x1 MIMO"
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
        -cm)
            CHANNEL_MODE="$2"
            if [[ "$CHANNEL_MODE" != "dynamic" && "$CHANNEL_MODE" != "static" ]]; then
                echo "ERROR: -cm 은 dynamic 또는 static 이어야 합니다 (입력: $CHANNEL_MODE)"
                exit 1
            fi
            shift 2 ;;
        -bs)
            BATCH_SIZE="$2"
            if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
                echo "ERROR: -bs 는 1 이상 정수여야 합니다 (입력: $BATCH_SIZE)"
                exit 1
            fi
            shift 2 ;;
        -ga)
            GNB_NX="$2"; GNB_NY="$3"
            shift 3 ;;
        -ua)
            UE_NX="$2"; UE_NY="$3"
            shift 3 ;;
        -pol)
            POLARIZATION="$2"
            if [[ "$POLARIZATION" != "single" && "$POLARIZATION" != "dual" ]]; then
                echo "ERROR: -pol 은 single 또는 dual 이어야 합니다 (입력: $POLARIZATION)"
                exit 1
            fi
            shift 2 ;;
        -c)
            CORE_EMULATOR="$2"
            shift 2 ;;
        -p1b)
            P1B_NPZ="$2"
            shift 2 ;;
        -rx)
            UE_RX_INDICES="$2"
            shift 2 ;;
        -a)
            ENABLE_ANALYZER=1
            shift ;;
        -h) usage ;;
        *) echo "ERROR: 알 수 없는 옵션: $1"; usage ;;
    esac
done

# ── 상호 배타 체크 ────────────────────────────────────────────────
if [ -n "$SNR_DB" ] && [ -n "$NOISE_DBFS" ]; then
    echo "ERROR: -snr 과 -nf 는 동시에 사용할 수 없습니다."
    exit 1
fi

# ── Core Emulator 연동 ──────────────────────────────────────────────
if [ -n "$CORE_EMULATOR" ]; then
    CORE_HOST="${CORE_EMULATOR%%:*}"
    CORE_PORT="${CORE_EMULATOR##*:}"
    CORE_HTTP_PORT=$((CORE_PORT + 1))
    echo "[launcher] Core Emulator (${CORE_HOST}:${CORE_PORT})에서 설정 로드 중..."

    _CORE_PARAMS=$(curl -sf "http://${CORE_HOST}:${CORE_HTTP_PORT}/launch_params" 2>/dev/null) || {
        echo "ERROR: Core Emulator에 연결할 수 없습니다 (http://${CORE_HOST}:${CORE_HTTP_PORT}/launch_params)"
        exit 1
    }

    _jp() { echo "$_CORE_PARAMS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$1','$2'))"; }

    NUM_UES=$(_jp NUM_UES "$NUM_UES")
    CHANNEL_MODE=$(_jp CHANNEL_MODE "$CHANNEL_MODE")
    BATCH_SIZE=$(_jp BATCH_SIZE "$BATCH_SIZE")
    GNB_NX=$(_jp GNB_NX "$GNB_NX")
    GNB_NY=$(_jp GNB_NY "$GNB_NY")
    UE_NX=$(_jp UE_NX "$UE_NX")
    UE_NY=$(_jp UE_NY "$UE_NY")
    POLARIZATION=$(_jp POLARIZATION "$POLARIZATION")

    # 3GPP channel topology parameters
    BS_HEIGHT_M=$(_jp BS_HEIGHT_M "25.0")
    UE_HEIGHT_M=$(_jp UE_HEIGHT_M "1.5")
    ISD_M=$(_jp ISD_M "500")
    MIN_UE_DIST_M=$(_jp MIN_UE_DISTANCE_M "35")
    MAX_UE_DIST_M=$(_jp MAX_UE_DISTANCE_M "500")
    SHADOW_FADING_STD_DB=$(_jp SHADOW_FADING_STD_DB "6.0")
    _KFM_RAW=$(_jp K_FACTOR_MEAN_DB "None")
    _KFS_RAW=$(_jp K_FACTOR_STD_DB "None")
    K_FACTOR_MEAN_DB=""
    K_FACTOR_STD_DB=""
    [ "$_KFM_RAW" != "None" ] && K_FACTOR_MEAN_DB="$_KFM_RAW"
    [ "$_KFS_RAW" != "None" ] && K_FACTOR_STD_DB="$_KFS_RAW"
    echo "[launcher] 3GPP: BS_h=${BS_HEIGHT_M}m UE_h=${UE_HEIGHT_M}m ISD=${ISD_M}m d=[${MIN_UE_DIST_M},${MAX_UE_DIST_M}]m SF_σ=${SHADOW_FADING_STD_DB}dB"

    echo "[launcher] Core Emulator에서 gnb.conf 생성 중..."
    GENERATED_CONF="/tmp/generated_gnb.conf"
    curl -sf "http://${CORE_HOST}:${CORE_HTTP_PORT}/gnb_conf" > "$GENERATED_CONF" 2>/dev/null || {
        echo "ERROR: gnb.conf 생성 실패"
        exit 1
    }
    echo "[launcher] gnb.conf 생성 완료 → ${GENERATED_CONF}"
fi

# ── 안테나 계산 ───────────────────────────────────────────────────
POL_MULT=1
[ "$POLARIZATION" = "dual" ] && POL_MULT=2
GNB_ANT=$((GNB_NX * GNB_NY * POL_MULT))
UE_ANT=$((UE_NX * UE_NY * POL_MULT))
MAX_ANT=$((GNB_ANT > UE_ANT ? GNB_ANT : UE_ANT))

PROXY_SCRIPT="${PROXY_VER}.py"

# ── 경로 설정 ─────────────────────────────────────────────────────
PROJ_DIR="/home/dclserver78/oai_sionna_junxiu"
BUILD_DIR="$PROJ_DIR/openairinterface5g_whan/cmake_targets/ran_build/build"
CONF="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"
if [ -n "$CORE_EMULATOR" ] && [ -n "$GENERATED_CONF" ] && [ -f "$GENERATED_CONF" ]; then
    GEN_AGE=$(( $(date +%s) - $(stat -c %Y "$GENERATED_CONF" 2>/dev/null || echo 0) ))
    if [ "$GEN_AGE" -lt 60 ]; then
        CONF="$GENERATED_CONF"
        echo "[launcher] Core Emulator 생성 gnb.conf 사용 (${GEN_AGE}초 전): ${CONF}"
    else
        echo "[launcher] 생성된 gnb.conf가 오래됨 (${GEN_AGE}초), 직접 conf 사용: ${CONF}"
    fi
fi
CONF_DIR="$(dirname "$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODE_SHORT_LOG="ipc"
[ "$MODE" = "socket" ] && MODE_SHORT_LOG="sock"
ANT_TAG=""
[ "$GNB_ANT" -gt 1 ] && ANT_TAG="${ANT_TAG}_ga${GNB_NX}x${GNB_NY}"
[ "$UE_ANT" -gt 1 ] && ANT_TAG="${ANT_TAG}_ua${UE_NX}x${UE_NY}"
[ "$POLARIZATION" = "dual" ] && ANT_TAG="${ANT_TAG}_xp2"
LOG_TAG_DIR="G1C_${PROXY_VER}_${MODE_SHORT_LOG}_${NUM_UES}ue${ANT_TAG}"
[ -n "$P1B_NPZ" ] && LOG_TAG_DIR="${LOG_TAG_DIR}_p1b"
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
    docker exec oai_sionna_proxy pkill -9 -f "v[0-9]" 2>/dev/null || true
    if [ -n "${CORE_PORT:-}" ]; then
        echo "[launcher] Core Emulator 포트 해제 중 (${CORE_PORT}, ${CORE_HTTP_PORT:-})..."
        for p in "$CORE_PORT" "${CORE_HTTP_PORT:-}"; do
            CPID=$(lsof -ti :"$p" 2>/dev/null || fuser "$p/tcp" 2>/dev/null | tr -d ' ')
            [ -n "$CPID" ] && kill -9 $CPID 2>/dev/null || true
        done
    fi
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
echo "  G1C 통합 런처 — Multi-UE MIMO Channel Proxy"
[ -n "$CORE_EMULATOR" ] && echo "    Core  : ${CORE_EMULATOR} (연동)"
echo "    Proxy : ${PROXY_SCRIPT}"
echo "    모드  : ${MODE}"
echo "    UE 수 : ${NUM_UES}"
echo "    gNB 안테나 : ${GNB_NX}x${GNB_NY}x${POL_MULT}pol = ${GNB_ANT}"
echo "    UE 안테나  : ${UE_NX}x${UE_NY}x${POL_MULT}pol = ${UE_ANT}"
echo "    편파       : ${POLARIZATION} (XP=$([ "$POLARIZATION" = "dual" ] && echo 2 || echo 1))"
if [ "$BYPASS_CHANNEL" -eq 1 ]; then
echo "    채널  : 바이패스 (IQ 패스스루)"
else
echo "    채널  : Sionna (${CHANNEL_MODE})"
fi
[ -n "$P1B_NPZ" ] && echo "    P1B   : ${P1B_NPZ} (UE별 독립 ray-tracing)"
echo "    배치  : ${BATCH_SIZE} UE/batch"
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
docker exec oai_sionna_proxy pkill -9 -f "v[0-9]" 2>/dev/null || true
# Clean SHM files but preserve directory (bind mount safe)
rm -f /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
rm -f /tmp/oai_gpu_ipc/gpu_ipc_shm_ue* 2>/dev/null || true
mkdir -p /tmp/oai_gpu_ipc 2>/dev/null || true
chmod 777 /tmp/oai_gpu_ipc 2>/dev/null || true
echo "[launcher] GPU IPC SHM 파일 정리 완료"
sleep 2

# ── Socket 모드: gNB를 먼저 시작 (gNB가 서버 역할) ──────────────
if [ "$MODE" = "socket" ]; then
    echo "[launcher] [socket] gNB 먼저 시작 (서버 port 6013)..."
    (cd "$LOG_DIR" && "$BUILD_DIR/nr-softmodem" \
        -O "$CONF" \
        --gNBs.[0].min_rxtxtime 3 --rfsim \
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

PROXY_CORE_ARG=""
PROXY_UE_SPEEDS_ARG=""
PROXY_CSINET_ENV=""
if [ -n "$CORE_EMULATOR" ]; then
    PROXY_CORE_ARG="--core-emulator=${CORE_EMULATOR}"
    _SPEEDS_JSON=$(curl -sf "http://${CORE_HOST}:${CORE_HTTP_PORT}/api/v1/traffic/speeds" 2>/dev/null) || true
    if [ -n "$_SPEEDS_JSON" ]; then
        _SPEEDS_KMH=$(echo "$_SPEEDS_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
kmh = d.get('speeds_kmh', {})
if kmh:
    print(','.join(str(kmh[str(i)]) for i in sorted(int(k) for k in kmh)))
" 2>/dev/null) || true
        if [ -n "$_SPEEDS_KMH" ]; then
            PROXY_UE_SPEEDS_ARG="--ue-speeds=${_SPEEDS_KMH}"
            echo "[launcher] Core Emulator UE별 속도: ${_SPEEDS_KMH} km/h"
        fi
    fi

    # CsiNet environment variables from Core Emulator
    _CSINET_JSON=$(curl -sf "http://${CORE_HOST}:${CORE_HTTP_PORT}/api/v1/csinet/env" 2>/dev/null) || true
    if [ -n "$_CSINET_JSON" ]; then
        _csinet_val() { echo "$_CSINET_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$1','$2'))"; }
        _CSINET_ENABLED=$(_csinet_val CSINET_ENABLED "0")
        if [ "$_CSINET_ENABLED" = "1" ]; then
            PROXY_CSINET_ENV="-e CSINET_ENABLED=1"
            PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_MODE=$(_csinet_val CSINET_MODE baseline)"
            PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_GAMMA=$(_csinet_val CSINET_GAMMA 0.25)"
            PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_SCENARIO=$(_csinet_val CSINET_SCENARIO UMi_NLOS)"
            PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_PERIOD=$(_csinet_val CSINET_PERIOD 20)"
            PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_PATH=$(_csinet_val CSINET_PATH /workspace/graduation/csinet)"
            PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_CHECKPOINT_DIR=$(_csinet_val CSINET_CHECKPOINT_DIR /workspace/csinet_checkpoints)"
            echo "[launcher] CsiNet 활성화: mode=$(_csinet_val CSINET_MODE baseline), gamma=$(_csinet_val CSINET_GAMMA 0.25)"
        else
            echo "[launcher] CsiNet 비활성화"
        fi
    fi
fi

# Fallback: if not using Core Emulator, pick up host CSINET_* env vars
if [ -z "$CORE_EMULATOR" ] && [ -z "$PROXY_CSINET_ENV" ] && [ "${CSINET_ENABLED:-0}" = "1" ]; then
    PROXY_CSINET_ENV="-e CSINET_ENABLED=1"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_MODE=${CSINET_MODE:-baseline}"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_GAMMA=${CSINET_GAMMA:-0.25}"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_SCENARIO=${CSINET_SCENARIO:-UMi_NLOS}"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_PERIOD=${CSINET_PERIOD:-20}"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_PATH=${CSINET_PATH:-/workspace/graduation/csinet}"
    PROXY_CSINET_ENV="$PROXY_CSINET_ENV -e CSINET_CHECKPOINT_DIR=${CSINET_CHECKPOINT_DIR:-/workspace/csinet_checkpoints}"
    echo "[launcher] CsiNet 활성화 (호스트 환경변수): mode=${CSINET_MODE:-baseline}, gamma=${CSINET_GAMMA:-0.25}"
fi

PROXY_ANALYZER_ARG=""
if [ "$ENABLE_ANALYZER" -eq 1 ]; then
    PROXY_ANALYZER_ARG="--analyzer --analyzer-log-dir=/workspace/logs/analyzer_output"
fi

PROXY_P1B_ARG=""
if [ -n "$P1B_NPZ" ]; then
    if [[ "$P1B_NPZ" == /* ]]; then
        _P1B_HOST_ABS="$P1B_NPZ"
    else
        _P1B_HOST_ABS="$(cd "$(dirname "$P1B_NPZ")" 2>/dev/null && pwd)/$(basename "$P1B_NPZ")"
    fi
    if [ ! -f "$_P1B_HOST_ABS" ]; then
        _P1B_HOST_ABS="${PROJ_DIR}/vRAN_Socket/${P1B_NPZ}"
    fi
    if [ ! -f "$_P1B_HOST_ABS" ]; then
        echo "ERROR: P1B 파일을 찾을 수 없습니다: $P1B_NPZ"
        echo "  시도한 경로: $_P1B_HOST_ABS"
        exit 1
    fi
    _P1B_REL="${_P1B_HOST_ABS#$PROJ_DIR/}"
    PROXY_P1B_ARG="--p1b-npz=/workspace/${_P1B_REL}"
    echo "[launcher] P1B 경로: $_P1B_HOST_ABS → /workspace/${_P1B_REL}"
    if [ -n "$UE_RX_INDICES" ]; then
        PROXY_P1B_ARG="$PROXY_P1B_ARG --ue-rx-indices=${UE_RX_INDICES}"
    fi
fi

docker exec -i $PROXY_ENV_ARGS $PROXY_CSINET_ENV oai_sionna_proxy python3 -u \
    "/workspace/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/${PROXY_SCRIPT}" \
    --mode="$MODE" $PROXY_EXTRA_ARGS \
    --gnb-ant=$GNB_ANT --ue-ant=$UE_ANT \
    --gnb-nx=$GNB_NX --gnb-ny=$GNB_NY \
    --ue-nx=$UE_NX --ue-ny=$UE_NY \
    --num-ues=$NUM_UES \
    --channel-mode=$CHANNEL_MODE \
    --polarization=$POLARIZATION \
    --sector-half-deg=${SECTOR_HALF_DEG:-90.0} \
    --jitter-std-deg=${JITTER_STD_DEG:-20.0} \
    --bs-height-m=${BS_HEIGHT_M:-25.0} \
    --ue-height-m=${UE_HEIGHT_M:-1.5} \
    --isd-m=${ISD_M:-500} \
    --min-ue-dist-m=${MIN_UE_DIST_M:-35} \
    --max-ue-dist-m=${MAX_UE_DIST_M:-500} \
    --shadow-fading-std-dB=${SHADOW_FADING_STD_DB:-6.0} \
    ${K_FACTOR_MEAN_DB:+--k-factor-mean-dB=$K_FACTOR_MEAN_DB} \
    ${K_FACTOR_STD_DB:+--k-factor-std-dB=$K_FACTOR_STD_DB} \
    $PROXY_CORE_ARG \
    $PROXY_UE_SPEEDS_ARG \
    $PROXY_ANALYZER_ARG \
    $PROXY_P1B_ARG \
    > "$LOG_DIR/proxy.log" 2>&1 &
PIDS+=($!)

echo "[launcher] Proxy 초기화 대기 (출력 실시간 표시)..."
echo "------------------------------------------------------------"
tail -f "$LOG_DIR/proxy.log" 2>/dev/null &
TAIL_WAIT_PID=$!

SHM_FIXED=0
for i in $(seq 1 120); do
    if [ "$SHM_FIXED" -eq 0 ] && [ -f /tmp/oai_gpu_ipc/gpu_ipc_shm ]; then
        docker exec oai_sionna_proxy chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
        for shm_ue in /tmp/oai_gpu_ipc/gpu_ipc_shm_ue*; do
            [ -f "$shm_ue" ] && docker exec oai_sionna_proxy chmod 666 "$shm_ue" 2>/dev/null || true
        done
        SHM_FIXED=1
        echo "[launcher] GPU IPC SHM 권한 수정 완료 (gNB + ${NUM_UES} UE(s))"
    fi
    if grep -q "Entering main loop\|Pipeline ready" "$LOG_DIR/proxy.log" 2>/dev/null; then
        if [ "$SHM_FIXED" -eq 0 ] && [ -f /tmp/oai_gpu_ipc/gpu_ipc_shm ]; then
            docker exec oai_sionna_proxy chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
            for shm_ue in /tmp/oai_gpu_ipc/gpu_ipc_shm_ue*; do
                [ -f "$shm_ue" ] && docker exec oai_sionna_proxy chmod 666 "$shm_ue" 2>/dev/null || true
            done
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
    GPU_IPC_ENV="RFSIM_GPU_IPC_V7=1"
    echo "[launcher] gNB 시작 (GPU IPC V7 futex, ant=${GNB_ANT})..."

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

    (cd "$LOG_DIR" && eval "$GPU_IPC_ENV" \
    "$BUILD_DIR/nr-softmodem" \
        -O "$CONF" \
        --gNBs.[0].min_rxtxtime 3 --rfsim \
        $GNB_ANT_ARGS \
        > gnb.log 2>&1) &
    PIDS+=($!)
    echo "[launcher] gNB PID: ${PIDS[-1]}"
    sleep 3
fi

# ── 3. UE 0 ~ N-1 (배치 기동: BATCH_SIZE 단위로 병렬 시작, 배치별 대기) ─────
UE_SETTLE_TIMEOUT=${UE_SETTLE_TIMEOUT:-60}

_launch_one_ue() {
    local ue_idx="$1"
    local imsi_suffix=$((ue_idx + 1))
    local imsi=$(printf "00101000000%04d" "$imsi_suffix")

    echo "[launcher] UE ${ue_idx} 시작 (IMSI=${imsi})..."

    if [ "$MODE" = "socket" ]; then
        (cd "$LOG_DIR" && "$BUILD_DIR/nr-uesoftmodem" \
            -r 106 --numerology 1 --band 78 -C 3619200000 \
            --uicc0.imsi "$imsi" \
            --rfsim --rfsimulator.serverport 6014 \
            > "ue${ue_idx}.log" 2>&1) &
    else
        local UE_GPU_ENV="RFSIM_GPU_IPC_V7=1 RFSIM_GPU_IPC_UE_IDX=$ue_idx"

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
            > "ue${ue_idx}.log" 2>&1) &
    fi
    PIDS+=($!)
    echo "[launcher] UE ${ue_idx} PID: ${PIDS[-1]}"
}

if [ $NUM_UES -ge 16 ] && [ $BATCH_SIZE -gt 2 ]; then
    echo "[launcher] N≥16 감지: 배치 크기 ${BATCH_SIZE} → 2 자동 축소"
    BATCH_SIZE=2
fi

if [ $NUM_UES -ge 32 ]; then
    INTER_UE_DELAY=1
    SETTLE_OK_WAIT=8
    SETTLE_FAIL_WAIT=3
    UE_SETTLE_TIMEOUT=60
    echo "[launcher] N≥32 감지: 안정화 (UE간격=${INTER_UE_DELAY}s, 성공대기=${SETTLE_OK_WAIT}s, 타임아웃=${UE_SETTLE_TIMEOUT}s)"
else
    INTER_UE_DELAY=1
    SETTLE_OK_WAIT=5
    SETTLE_FAIL_WAIT=3
fi
echo "[launcher] UE 배치 기동: ${NUM_UES} UE(s), 배치 크기=${BATCH_SIZE}"

batch_start=0
while [ $batch_start -lt $NUM_UES ]; do
    batch_end=$((batch_start + BATCH_SIZE - 1))
    [ $batch_end -ge $NUM_UES ] && batch_end=$((NUM_UES - 1))

    echo ""
    echo "[launcher] ── 배치 시작: UE[${batch_start}..${batch_end}] ──"

    for ue_idx in $(seq $batch_start $batch_end); do
        _launch_one_ue $ue_idx
        sleep $INTER_UE_DELAY
    done

    echo "[launcher] 배치 [${batch_start}..${batch_end}] RRC_CONNECTED 대기 (최대 ${UE_SETTLE_TIMEOUT}s)..."
    waited=0
    all_settled=0
    while [ $waited -lt $UE_SETTLE_TIMEOUT ]; do
        settled_count=0
        for ue_idx in $(seq $batch_start $batch_end); do
            if grep -q "RRC_CONNECTED\|State = NR_RRC_CONNECTED" "$LOG_DIR/ue${ue_idx}.log" 2>/dev/null; then
                settled_count=$((settled_count + 1))
            fi
        done
        batch_count=$((batch_end - batch_start + 1))
        if [ $settled_count -ge $batch_count ]; then
            all_settled=1
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    if [ $all_settled -eq 1 ]; then
        echo "[launcher] 배치 [${batch_start}..${batch_end}] 전원 연결 (${waited}s). 안정화 대기 ${SETTLE_OK_WAIT}s..."
        sleep $SETTLE_OK_WAIT
    else
        echo "[launcher] 배치 [${batch_start}..${batch_end}] 부분 타임아웃 (${settled_count}/${batch_count} 연결, ${waited}s). 다음 배치 진행..."
        sleep $SETTLE_FAIL_WAIT
    fi

    batch_start=$((batch_end + 1))
done

echo "[launcher] 전체 ${NUM_UES} UE(s) 기동 완료."

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

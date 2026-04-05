#!/bin/bash
#
# G1A 통합 런처 — Proxy + gNB + UE N대를 한번에 실행
#
# 사용법:
#   sudo bash launch_all.sh [옵션]
#
# 옵션:
#   -v VERSION   Proxy Python 버전: v1, v2, v3  (기본: v3)
#   -m MODE      통신 모드: socket, gpu-ipc      (기본: gpu-ipc)
#   -n NUM_UES   UE 수: 1~8                     (기본: 2)
#   -b           Sionna 채널 바이패스 (IQ 패스스루)
#   -h           도움말
#
# 예시:
#   sudo bash launch_all.sh                     # v3, gpu-ipc, 2 UE
#   sudo bash launch_all.sh -v v2 -n 4          # v2, gpu-ipc, 4 UE
#   sudo bash launch_all.sh -m socket -n 2      # v3, socket, 2 UE
#   sudo bash launch_all.sh -v v1 -m socket -n 1  # v1, socket, 1 UE
#
# 로그 파일:
#   /home/dclcom57/oai_sionna_junxiu/log/YYYYMMDD_HHMMSS/proxy.log, gnb.log, ue0.log, ...
#   /home/dclcom57/oai_sionna_junxiu/log/latest → 최근 실행 심링크
#
# 종료:
#   Ctrl+C → 전체 프로세스 자동 종료
#

set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────
PROXY_VER="v4"
MODE="gpu-ipc"
NUM_UES=2
BYPASS_CHANNEL=0

# ── 인자 파싱 ─────────────────────────────────────────────────────
usage() {
    echo "G1A 통합 런처"
    echo ""
    echo "사용법: sudo bash launch_all.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  -v VERSION   Proxy 버전: v1, v2, v3, v4  (기본: v4)"
    echo "  -m MODE      모드: socket, gpu-ipc        (기본: gpu-ipc)"
    echo "  -n NUM_UES   UE 수: 1~8                  (기본: 2)"
    echo "  -b           Sionna 채널 바이패스 (IQ 패스스루)"
    echo "  -h           도움말"
    echo ""
    echo "버전 설명:"
    echo "  v1  기본 다중 UE (순차 처리)"
    echo "  v2  v1 + per-UE 로깅, DL 배치"
    echo "  v3  v2 + DL/UL 모두 배치 처리"
    echo "  v4  v3 + 통합 채널 생성 (1 Producer, 1 RingBuffer)"
    echo ""
    echo "예시:"
    echo "  sudo bash launch_all.sh                        # v4, gpu-ipc, 2 UE"
    echo "  sudo bash launch_all.sh -v v3 -n 4             # v3, gpu-ipc, 4 UE"
    echo "  sudo bash launch_all.sh -m socket -n 2         # v4, socket, 2 UE"
    exit 0
}

while getopts "v:m:n:bh" opt; do
    case "$opt" in
        v)
            PROXY_VER="$OPTARG"
            if [[ "$PROXY_VER" != "v1" && "$PROXY_VER" != "v2" && "$PROXY_VER" != "v3" && "$PROXY_VER" != "v4" ]]; then
                echo "ERROR: -v 는 v1, v2, v3, v4 중 하나여야 합니다 (입력: $PROXY_VER)"
                exit 1
            fi
            ;;
        m)
            MODE="$OPTARG"
            if [[ "$MODE" != "socket" && "$MODE" != "gpu-ipc" ]]; then
                echo "ERROR: -m 은 socket 또는 gpu-ipc 여야 합니다 (입력: $MODE)"
                exit 1
            fi
            ;;
        n)
            NUM_UES="$OPTARG"
            if ! [[ "$NUM_UES" =~ ^[1-8]$ ]]; then
                echo "ERROR: -n 은 1~8 정수여야 합니다 (입력: $NUM_UES)"
                exit 1
            fi
            ;;
        b) BYPASS_CHANNEL=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done

PROXY_SCRIPT="${PROXY_VER}_multi_ue.py"

# ── 경로 설정 ─────────────────────────────────────────────────────
PROJ_DIR="/home/dclcom57/oai_sionna_junxiu"
BUILD_DIR="$PROJ_DIR/openairinterface5g_whan/cmake_targets/ran_build/build"
CONF="$PROJ_DIR/openairinterface5g_whan/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/dclcom57/oai_sionna_junxiu/log/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" /home/dclcom57/oai_sionna_junxiu/log/latest

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
    docker exec oai_sionna_proxy pkill -9 -f "multi_ue" 2>/dev/null || true
    echo "[launcher] 종료 완료. 로그: $LOG_DIR/"
    echo "[launcher] 바로가기: /home/dclcom57/oai_sionna_junxiu/log/latest/"
    exit 0
}

trap cleanup SIGINT SIGTERM

PROXY_EXTRA_ARGS=""
if [ "$BYPASS_CHANNEL" -eq 1 ]; then
    PROXY_EXTRA_ARGS="--no-custom-channel"
fi

echo "============================================================"
echo "  G1A 통합 런처"
echo "    Proxy : ${PROXY_SCRIPT}"
echo "    모드  : ${MODE}"
echo "    UE 수 : ${NUM_UES}"
if [ "$BYPASS_CHANNEL" -eq 1 ]; then
echo "    채널  : 바이패스 (IQ 패스스루)"
else
echo "    채널  : Sionna (custom channel)"
fi
echo "    로그  : ${LOG_DIR}/"
echo "  종료: Ctrl+C"
echo "============================================================"

# ── 이전 프로세스 정리 ────────────────────────────────────────────
pkill -9 -f "nr-softmodem" 2>/dev/null || true
pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
docker exec oai_sionna_proxy pkill -9 -f "multi_ue" 2>/dev/null || true
rm -f /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
mkdir -p /tmp/oai_gpu_ipc 2>/dev/null || true
chmod 777 /tmp/oai_gpu_ipc 2>/dev/null || true
sleep 2

# ── Socket 모드: gNB를 먼저 시작 (gNB가 서버 역할) ──────────────
if [ "$MODE" = "socket" ]; then
    echo "[launcher] [socket] gNB 먼저 시작 (서버 port ${GNB_PORT:-6013})..."
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
echo "[launcher] Proxy 시작 (${PROXY_SCRIPT}, ${MODE}, ${NUM_UES} UE)..."

docker exec -i oai_sionna_proxy python3 -u \
    "/workspace/vRAN_Socket/G1A_MultiUE_Channel_Proxy/${PROXY_SCRIPT}" \
    --mode="$MODE" --num-ues "$NUM_UES" $PROXY_EXTRA_ARGS \
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
        SHM_FIXED=1
        echo "[launcher] GPU IPC SHM 권한 수정 완료 (docker exec chmod)"
    fi
    if grep -q "Entering main loop" "$LOG_DIR/proxy.log" 2>/dev/null; then
        if [ "$SHM_FIXED" -eq 0 ] && [ -f /tmp/oai_gpu_ipc/gpu_ipc_shm ]; then
            docker exec oai_sionna_proxy chmod 666 /tmp/oai_gpu_ipc/gpu_ipc_shm 2>/dev/null || true
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
    echo "[launcher] gNB 시작..."
    (cd "$LOG_DIR" && RFSIM_GPU_IPC_V2=1 \
    "$BUILD_DIR/nr-softmodem" \
        -O "$CONF" \
        --gNBs.[0].min_rxtxtime 6 --rfsim \
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
        (cd "$LOG_DIR" && RFSIM_GPU_IPC_V2=1 \
        RFSIM_GPU_IPC_UE_IDX=$ue_idx \
        "$BUILD_DIR/nr-uesoftmodem" \
            -r 106 --numerology 1 --band 78 -C 3619200000 \
            --uicc0.imsi "$imsi" \
            --rfsim \
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
echo "    cat ${LOG_DIR}/nrMAC_stats.log       # UE별 CQI/PMI/RSRP/BLER/MCS"
echo "    cat ${LOG_DIR}/nrRRC_stats.log       # 연결된 DU/UE 목록"
echo "    cat ${LOG_DIR}/nrL1_stats.log        # PRB I0, PRACH I0"
echo ""
echo "  바로가기: /home/dclcom57/oai_sionna_junxiu/log/latest/"
echo "  Ctrl+C로 전체 종료"
echo "============================================================"

# Proxy 로그를 실시간으로 화면에 출력
tail -f "$LOG_DIR/proxy.log" &
TAIL_PID=$!

wait "${PIDS[0]}" 2>/dev/null || true
kill $TAIL_PID 2>/dev/null || true
cleanup

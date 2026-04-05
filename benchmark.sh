#!/bin/bash
#
# 5G E2E 벤치마크 러너
#
# 사용법:
#   sudo bash benchmark.sh [실행시간(초)] [UE수] [옵션]
#
# 옵션:
#   -b   Sionna 채널 바이패스 (IQ 패스스루)
#   -v   Proxy 버전 (기본: v4)
#   -m   모드: socket / gpu-ipc (기본: gpu-ipc)
#
# 예시:
#   sudo bash benchmark.sh 60 1           # 1 UE, 60초, gpu-ipc
#   sudo bash benchmark.sh 60 4           # 4 UE, 60초, gpu-ipc
#   sudo bash benchmark.sh 60 1 -b        # 1 UE, 60초, 채널 바이패스
#   sudo bash benchmark.sh 120 4 -v v3    # 4 UE, 120초, v3 proxy
#

set -euo pipefail

PROJ_DIR="/home/dclcom57/oai_sionna_junxiu"
LAUNCHER="$PROJ_DIR/vRAN_Socket/G1A_MultiUE_Channel_Proxy/launch_all.sh"
RESULTS_DIR="$PROJ_DIR/results"

DURATION=${1:-60}
NUM_UES=${2:-2}
shift 2 2>/dev/null || true

PROXY_VER="v4"
MODE="gpu-ipc"
BYPASS=""

while getopts "v:m:b" opt; do
    case "$opt" in
        v) PROXY_VER="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        b) BYPASS="-b" ;;
        *) ;;
    esac
done

TAG="${NUM_UES}ue_${MODE}"
if [ -n "$BYPASS" ]; then
    TAG="${TAG}_bypass"
fi
RUN_DIR="${RESULTS_DIR}/${TAG}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "  5G E2E 벤치마크"
echo "    UE 수     : ${NUM_UES}"
echo "    실행 시간  : ${DURATION}초"
echo "    모드      : ${MODE}"
echo "    Proxy 버전: ${PROXY_VER}"
echo "    채널      : $([ -n "$BYPASS" ] && echo '바이패스' || echo 'Sionna')"
echo "    결과 저장  : ${RUN_DIR}"
echo "============================================================"

cleanup() {
    echo ""
    echo "[benchmark] 프로세스 정리 중..."
    pkill -9 -f "nr-softmodem" 2>/dev/null || true
    pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
    docker exec oai_sionna_proxy pkill -9 -f "multi_ue" 2>/dev/null || true
    sleep 2
}

trap cleanup EXIT

bash "$LAUNCHER" -v "$PROXY_VER" -m "$MODE" -n "$NUM_UES" $BYPASS &
LAUNCHER_PID=$!

LOG_DIR="$PROJ_DIR/log/latest"

echo "[benchmark] 시스템 기동 대기 (최대 180초)..."
READY=0
for i in $(seq 1 180); do
    if [ -f "$LOG_DIR/proxy.log" ] && grep -q "Entering main loop" "$LOG_DIR/proxy.log" 2>/dev/null; then
        echo "[benchmark] Proxy 메인 루프 진입 확인 (${i}초)"
        READY=1
        break
    fi
    if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        echo "[benchmark] ERROR: 런처가 종료됨"
        exit 1
    fi
    sleep 1
done

if [ "$READY" -eq 0 ]; then
    echo "[benchmark] WARN: Proxy 메인 루프 대기 180초 초과, 계속 진행..."
fi

echo "[benchmark] UE 연결 대기 (최대 60초)..."
UE_CONNECTED=0
for i in $(seq 1 60); do
    if [ -f "$LOG_DIR/nrMAC_stats.log" ]; then
        UE_COUNT=$(grep -c "UE RNTI" "$LOG_DIR/nrMAC_stats.log" 2>/dev/null || true)
        UE_COUNT=${UE_COUNT:-0}
        if [ "$UE_COUNT" -gt 0 ] 2>/dev/null; then
            echo "[benchmark] UE 연결 감지: ${UE_COUNT}개 UE (${i}초)"
            UE_CONNECTED=1
            break
        fi
    fi
    sleep 1
done

if [ "$UE_CONNECTED" -eq 0 ]; then
    echo "[benchmark] WARN: UE 연결 감지 실패 (60초 초과), 로그 수집 후 종료..."
    DURATION=5
fi

echo "[benchmark] 측정 시작: ${DURATION}초 동안 실행..."
sleep "$DURATION"

echo "[benchmark] 측정 완료. 로그 수집 중..."

for f in proxy.log gnb.log nrMAC_stats.log nrRRC_stats.log nrL1_stats.log; do
    if [ -f "$LOG_DIR/$f" ]; then
        cp "$LOG_DIR/$f" "$RUN_DIR/"
    fi
done

for ue_log in "$LOG_DIR"/ue*.log; do
    if [ -f "$ue_log" ]; then
        cp "$ue_log" "$RUN_DIR/"
    fi
done

for ue_l1 in "$LOG_DIR"/nrL1_UE_stats*.log; do
    if [ -f "$ue_l1" ]; then
        cp "$ue_l1" "$RUN_DIR/"
    fi
done

cat > "$RUN_DIR/metadata.json" <<METAEOF
{
    "timestamp": "$(date -Iseconds)",
    "num_ues": $NUM_UES,
    "duration_sec": $DURATION,
    "mode": "$MODE",
    "proxy_ver": "$PROXY_VER",
    "bypass_channel": $([ -n "$BYPASS" ] && echo 'true' || echo 'false'),
    "ue_connected": $([ "$UE_CONNECTED" -eq 1 ] && echo 'true' || echo 'false')
}
METAEOF

echo "[benchmark] 결과 저장 완료: ${RUN_DIR}"
echo "[benchmark] 종료 중..."

kill "$LAUNCHER_PID" 2>/dev/null || true
wait "$LAUNCHER_PID" 2>/dev/null || true

echo "[benchmark] 완료."
echo ""
echo "결과 파일:"
ls -la "$RUN_DIR/"
echo ""
echo "통계 파싱:"
echo "  python3 $PROJ_DIR/parse_stats.py $RUN_DIR"

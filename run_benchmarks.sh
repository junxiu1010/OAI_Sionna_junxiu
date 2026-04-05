#!/bin/bash
#
# 1 UE vs 4 UE 성능 비교 — 전체 자동화 스크립트
#
# 사용법:
#   sudo bash run_benchmarks.sh
#
# 이 스크립트는 다음 순서로 실행됩니다:
#   1. Phase 1: 채널 바이패스 테스트 (GPU-IPC, 1 UE, --no-custom-channel)
#      → UE가 gNB에 연결 가능한지 확인
#   2. Phase 3-1: 1 UE 벤치마크 (60초)
#   3. Phase 3-2: 4 UE 벤치마크 (60초)
#   4. 비교 리포트 생성
#

set -euo pipefail

PROJ_DIR="/home/dclcom57/oai_sionna_junxiu"
RESULTS_DIR="$PROJ_DIR/results"
BENCHMARK="$PROJ_DIR/benchmark.sh"
PARSER="$PROJ_DIR/parse_stats.py"
DURATION=60

echo "============================================================"
echo "  1 UE vs 4 UE E2E 5G 성능 비교 — 자동화 실행"
echo "============================================================"
echo ""

# ── Phase 1: 채널 바이패스 테스트 ─────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1: 채널 바이패스 테스트 (GPU-IPC, 1 UE)"
echo "  목적: Sionna 채널 없이 IPC 경로만으로 UE 연결 가능 여부 확인"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$BENCHMARK" 30 1 -b 2>&1
BYPASS_DIR=$(ls -td "$RESULTS_DIR"/1ue_gpu-ipc_bypass/*/ 2>/dev/null | head -1)

if [ -z "$BYPASS_DIR" ]; then
    echo "[ERROR] 채널 바이패스 테스트 결과 디렉토리를 찾을 수 없습니다."
    exit 1
fi

echo ""
echo "[Phase 1] 결과 분석..."
python3 "$PARSER" "$BYPASS_DIR"

BYPASS_CONNECTED=$(python3 -c "
import json, sys
try:
    with open('${BYPASS_DIR}/metadata.json') as f:
        m = json.load(f)
    print('yes' if m.get('ue_connected') else 'no')
except: print('no')
")

echo ""
if [ "$BYPASS_CONNECTED" = "yes" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  [Phase 1 성공] 채널 바이패스로 UE 연결됨!"
    echo "  → Sionna 채널 모델이 PBCH 디코딩 실패의 원인일 가능성 높음"
    echo "  → Phase 3으로 진행 (채널 바이패스 상태에서 1 UE vs 4 UE 비교)"
    echo "═══════════════════════════════════════════════════════════════"
    CHANNEL_FLAG="-b"
else
    echo "═══════════════════════════════════════════════════════════════"
    echo "  [Phase 1] 채널 바이패스에서도 UE 연결 실패"
    echo "  → GPU IPC 경로 자체에 문제가 있을 수 있음"
    echo "  → 채널 활성화 상태에서도 비교 테스트 진행 (proxy 지연 비교)"
    echo "═══════════════════════════════════════════════════════════════"
    CHANNEL_FLAG=""
fi

echo ""
sleep 5

# ── Phase 3-1: 1 UE 벤치마크 ─────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 3-1: 1 UE 벤치마크 (${DURATION}초)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$BENCHMARK" "$DURATION" 1 $CHANNEL_FLAG 2>&1
DIR_1UE=$(ls -td "$RESULTS_DIR"/1ue_gpu-ipc*/*/ 2>/dev/null | head -1)

echo ""
echo "[Phase 3-1] 1 UE 결과 분석..."
python3 "$PARSER" "$DIR_1UE"

echo ""
sleep 5

# ── Phase 3-2: 4 UE 벤치마크 ─────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 3-2: 4 UE 벤치마크 (${DURATION}초)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$BENCHMARK" "$DURATION" 4 $CHANNEL_FLAG 2>&1
DIR_4UE=$(ls -td "$RESULTS_DIR"/4ue_gpu-ipc*/*/ 2>/dev/null | head -1)

echo ""
echo "[Phase 3-2] 4 UE 결과 분석..."
python3 "$PARSER" "$DIR_4UE"

echo ""
sleep 2

# ── Phase 3-3: 비교 리포트 생성 ──────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 3-3: 1 UE vs 4 UE 비교 리포트"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

REPORT_PATH="$RESULTS_DIR/comparison_report_$(date +%Y%m%d_%H%M%S).txt"
python3 "$PARSER" --compare "$DIR_1UE" "$DIR_4UE" -o "$REPORT_PATH" 2>&1

echo ""
python3 "$PARSER" --compare "$DIR_1UE" "$DIR_4UE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  완료!"
echo ""
echo "  결과 디렉토리:"
echo "    바이패스 테스트: $BYPASS_DIR"
echo "    1 UE 벤치마크:  $DIR_1UE"
echo "    4 UE 벤치마크:  $DIR_4UE"
echo "    비교 리포트:    $REPORT_PATH"
echo ""
echo "  JSON 형식 리포트:"
echo "    python3 $PARSER --compare $DIR_1UE $DIR_4UE --json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

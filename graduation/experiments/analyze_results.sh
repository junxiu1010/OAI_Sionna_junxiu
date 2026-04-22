#!/bin/bash
# ================================================================
#  실험 결과 요약 — 9가지 실험의 핵심 KPI 추출
#
#  Usage: bash analyze_results.sh [RESULTS_DIR]
# ================================================================

RESULTS_DIR="${1:-/home/dclserver78/oai_sionna_junxiu/graduation/experiments/results}"

echo "================================================================"
echo "  Baseline 실험 결과 요약"
echo "  디렉토리: ${RESULTS_DIR}"
echo "================================================================"
echo ""

printf "%-4s %-12s %-18s %5s %5s %5s %6s %6s %5s %8s\n" \
    "#" "Channel" "Mode" "PDU" "RLF" "HARQ" "MCS↑" "BLER%" "MU%" "Tput(kb)"
printf "%s\n" "$(printf '%.0s-' {1..95})"

for exp_dir in "$RESULTS_DIR"/[0-9]*_*; do
    [ ! -d "$exp_dir" ] && continue

    dir_name=$(basename "$exp_dir")
    exp_num=$(echo "$dir_name" | cut -d'_' -f1)
    scenario=$(echo "$dir_name" | cut -d'_' -f2)
    mode=$(echo "$dir_name" | cut -d'_' -f3-)

    gnb_log="$exp_dir/gnb_cell0.log"
    [ ! -f "$gnb_log" ] && continue

    pdu_ok=0
    pdu_fail=0
    for ue_log in "$exp_dir"/ue_cell0_ue*.log; do
        [ ! -f "$ue_log" ] && continue
        if grep -q "PDU SESSION ESTABLISHMENT ACCEPT\|NAS_CONN_ESTABLI_CNF" "$ue_log" 2>/dev/null; then
            pdu_ok=$((pdu_ok + 1))
        else
            pdu_fail=$((pdu_fail + 1))
        fi
    done

    rlf_count=$(grep -c "Radio Link Failure\|RLF detected\|Detected rlf" "$gnb_log" 2>/dev/null || echo "0")

    harq_drop=$(grep -c "Could not find a HARQ process" "$gnb_log" 2>/dev/null || echo "0")

    max_mcs=$(grep -oP "MCS \d+ -> \d+" "$gnb_log" 2>/dev/null | \
        awk -F'[ >-]+' '{print $NF}' | sort -n | tail -1 || echo "0")
    [ -z "$max_mcs" ] && max_mcs=0

    bler_lines=$(grep -oP "BLER wnd [\d.]+" "$gnb_log" 2>/dev/null | awk '{print $NF}')
    if [ -n "$bler_lines" ]; then
        avg_bler=$(echo "$bler_lines" | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n*100; else print "N/A"}')
    else
        avg_bler="N/A"
    fi

    mu_total=$(grep -c "MU-MIMO pairing\|mu_mimo_pair\|Pairing UE" "$gnb_log" 2>/dev/null || echo "0")

    mac_stats="$exp_dir/nrMAC_stats.log"
    tput="N/A"
    if [ -f "$mac_stats" ]; then
        tput_val=$(grep -oP "dlsch_total_bytes\s*\d+" "$mac_stats" 2>/dev/null | awk '{sum+=$2} END {if(NR>0) printf "%.0f", sum/NR/1024; else print "N/A"}')
        [ -n "$tput_val" ] && tput="$tput_val"
    fi

    printf "%-4s %-12s %-18s %2d/%d %5d %5d %6s %6s %5d %8s\n" \
        "$exp_num" "$scenario" "$mode" "$pdu_ok" "$((pdu_ok+pdu_fail))" \
        "$rlf_count" "$harq_drop" "$max_mcs" "$avg_bler" "$mu_total" "$tput"
done

echo ""
echo "KPI 범례:"
echo "  PDU    = PDU 세션 수립 성공/전체"
echo "  RLF    = Radio Link Failure 횟수"
echo "  HARQ   = HARQ 프로세스 못찾음 횟수"
echo "  MCS↑   = 도달한 최대 MCS"
echo "  BLER%  = 평균 BLER (%)"
echo "  MU%    = MU-MIMO 페어링 이벤트 수"
echo "  Tput   = 대략적인 DL throughput (KB)"

#!/usr/bin/env python3
"""
OAI nrMAC_stats.log / proxy.log 파서

사용법:
  python3 parse_stats.py <결과_디렉토리>
  python3 parse_stats.py results/1ue_gpu-ipc/20260301_120000

  비교 모드:
  python3 parse_stats.py --compare <디렉토리1> <디렉토리2>
  python3 parse_stats.py --compare results/1ue_gpu-ipc/run1 results/4ue_gpu-ipc/run1
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_nrmac_stats(filepath: str) -> list:
    """nrMAC_stats.log에서 UE별 메트릭을 추출한다.

    OAI는 이 파일을 매초 덮어쓰므로, 파일 내용은 마지막 스냅샷이다.
    """
    if not os.path.isfile(filepath):
        return []

    with open(filepath) as f:
        content = f.read()

    if not content.strip():
        return []

    ues = []
    ue_blocks = re.split(r'(?=UE RNTI)', content)

    for block in ue_blocks:
        block = block.strip()
        if not block.startswith("UE RNTI"):
            continue

        ue = {}

        m = re.search(r'UE RNTI\s+([0-9a-fA-Fx]+)', block)
        if m:
            ue['rnti'] = m.group(1)

        m = re.search(r'CQI\s+(\d+)', block)
        if m:
            ue['cqi'] = int(m.group(1))

        m = re.search(r'RI\s+(\d+)', block)
        if m:
            ue['ri'] = int(m.group(1))

        m = re.search(r'PMI\s+\((\d+),(\d+)\)', block)
        if m:
            ue['pmi'] = (int(m.group(1)), int(m.group(2)))

        m = re.search(r'dlsch_rounds\s+([\d/]+)', block)
        if m:
            ue['dlsch_rounds'] = m.group(1)

        m = re.search(r'dlsch_errors\s+(\d+)', block)
        if m:
            ue['dlsch_errors'] = int(m.group(1))

        m = re.search(r'dlsch_mcs\s+(\d+)', block)
        if m:
            ue['dl_mcs'] = int(m.group(1))

        m = re.search(r'ulsch_rounds\s+([\d/]+)', block)
        if m:
            ue['ulsch_rounds'] = m.group(1)

        m = re.search(r'ulsch_errors\s+(\d+)', block)
        if m:
            ue['ulsch_errors'] = int(m.group(1))

        m = re.search(r'ulsch_mcs\s+(\d+)', block)
        if m:
            ue['ul_mcs'] = int(m.group(1))

        for direction in ['dl', 'ul']:
            pattern = rf'{direction.upper()}\s+BLER\s+([\d.eE+\-]+)'
            m = re.search(pattern, block, re.IGNORECASE)
            if m:
                try:
                    ue[f'{direction}_bler'] = float(m.group(1))
                except ValueError:
                    pass

        m = re.search(r'BLER\s+([\d.eE+\-]+)\s.*?BLER\s+([\d.eE+\-]+)', block)
        if m and 'dl_bler' not in ue:
            try:
                ue['dl_bler'] = float(m.group(1))
                ue['ul_bler'] = float(m.group(2))
            except ValueError:
                pass

        m = re.search(r'avg\s+RSRP\s+(-?\d+)', block)
        if m:
            ue['rsrp'] = int(m.group(1))

        m = re.search(r'SNR\s+([-\d.]+)\s*dB', block)
        if m:
            ue['snr_db'] = float(m.group(1))

        m = re.search(r'mac_DL_total_bytes\s+(\d+)', block)
        if m:
            ue['dl_total_bytes'] = int(m.group(1))

        m = re.search(r'mac_UL_total_bytes\s+(\d+)', block)
        if m:
            ue['ul_total_bytes'] = int(m.group(1))

        if ue:
            ues.append(ue)

    return ues


def parse_proxy_log(filepath: str) -> dict:
    """proxy.log에서 E2E 통계를 추출한다."""
    if not os.path.isfile(filepath):
        return {}

    stats = {
        'mode': None,
        'num_ues': None,
        'custom_channel': None,
        'total_frames': 0,
        'e2e_times_ms': [],
        'dl_times_ms': [],
        'ul_times_ms': [],
    }

    with open(filepath) as f:
        for line in f:
            if line.startswith('Mode:'):
                m = re.search(r'^Mode:\s*(\S+)', line)
                if m:
                    stats['mode'] = m.group(1)

            if line.startswith('Number of UEs:'):
                m = re.search(r'^Number of UEs:\s*(\d+)', line)
                if m:
                    stats['num_ues'] = int(m.group(1))

            if line.startswith('Custom Channel:'):
                stats['custom_channel'] = 'Enabled' in line

            m = re.search(r'\[E2E\s+frame#(\d+)', line)
            if m:
                stats['total_frames'] = max(stats['total_frames'], int(m.group(1)))

            if '[E2E frame#' in line and 'wall=' in line:
                m = re.search(r'wall=([\d.]+)ms', line)
                if m:
                    stats['e2e_times_ms'].append(float(m.group(1)))

                m = re.search(r'DL=([\d.]+)\+UL=([\d.]+)', line)
                if m:
                    stats['dl_times_ms'].append(float(m.group(1)))
                    stats['ul_times_ms'].append(float(m.group(2)))

    if stats['e2e_times_ms']:
        times = stats['e2e_times_ms']
        stats['e2e_avg_ms'] = sum(times) / len(times)
        stats['e2e_max_ms'] = max(times)
        stats['e2e_min_ms'] = min(times)
    if stats['dl_times_ms']:
        times = stats['dl_times_ms']
        stats['dl_avg_ms'] = sum(times) / len(times)
    if stats['ul_times_ms']:
        times = stats['ul_times_ms']
        stats['ul_avg_ms'] = sum(times) / len(times)

    del stats['e2e_times_ms']
    del stats['dl_times_ms']
    del stats['ul_times_ms']

    return stats


def parse_gnb_log(filepath: str) -> dict:
    """gnb.log에서 주요 이벤트 정보를 추출한다."""
    if not os.path.isfile(filepath):
        return {}

    stats = {
        'rrc_connections': 0,
        'prach_received': 0,
        'errors': [],
    }

    with open(filepath) as f:
        for line in f:
            if 'RRCSetupComplete' in line or 'RRC_CONNECTED' in line:
                stats['rrc_connections'] += 1
            if 'PRACH' in line and 'received' in line.lower():
                stats['prach_received'] += 1
            if 'Assertion' in line or 'FATAL' in line:
                stats['errors'].append(line.strip()[:200])

    return stats


def parse_result_dir(result_dir: str) -> dict:
    """결과 디렉토리의 모든 로그를 파싱한다."""
    result_dir = Path(result_dir)
    result = {
        'directory': str(result_dir),
    }

    meta_path = result_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            result['metadata'] = json.load(f)

    result['mac_stats'] = parse_nrmac_stats(str(result_dir / 'nrMAC_stats.log'))
    result['proxy_stats'] = parse_proxy_log(str(result_dir / 'proxy.log'))
    result['gnb_stats'] = parse_gnb_log(str(result_dir / 'gnb.log'))

    return result


def format_single_report(data: dict) -> str:
    """단일 실행 결과를 텍스트로 출력한다."""
    lines = []
    lines.append("=" * 60)
    lines.append("  5G E2E 벤치마크 결과")
    lines.append("=" * 60)

    meta = data.get('metadata', {})
    if meta:
        lines.append(f"  UE 수      : {meta.get('num_ues', '?')}")
        lines.append(f"  실행 시간   : {meta.get('duration_sec', '?')}초")
        lines.append(f"  모드       : {meta.get('mode', '?')}")
        lines.append(f"  Proxy 버전  : {meta.get('proxy_ver', '?')}")
        lines.append(f"  채널 바이패스: {meta.get('bypass_channel', '?')}")
        lines.append(f"  UE 연결     : {meta.get('ue_connected', '?')}")
        lines.append(f"  타임스탬프   : {meta.get('timestamp', '?')}")

    lines.append("")
    lines.append("── Proxy 통계 ──")
    ps = data.get('proxy_stats', {})
    if ps:
        lines.append(f"  모드          : {ps.get('mode', '?')}")
        lines.append(f"  UE 수         : {ps.get('num_ues', '?')}")
        lines.append(f"  커스텀 채널    : {ps.get('custom_channel', '?')}")
        lines.append(f"  처리 프레임 수 : {ps.get('total_frames', 0)}")
        if 'e2e_avg_ms' in ps:
            lines.append(f"  E2E 지연 (avg) : {ps['e2e_avg_ms']:.2f} ms")
            lines.append(f"  E2E 지연 (max) : {ps['e2e_max_ms']:.2f} ms")
            lines.append(f"  E2E 지연 (min) : {ps['e2e_min_ms']:.2f} ms")
        if 'dl_avg_ms' in ps:
            lines.append(f"  DL 평균 지연    : {ps['dl_avg_ms']:.2f} ms")
        if 'ul_avg_ms' in ps:
            lines.append(f"  UL 평균 지연    : {ps['ul_avg_ms']:.2f} ms")
    else:
        lines.append("  (데이터 없음)")

    lines.append("")
    lines.append("── gNB 통계 ──")
    gs = data.get('gnb_stats', {})
    if gs:
        lines.append(f"  RRC 연결 수   : {gs.get('rrc_connections', 0)}")
        lines.append(f"  PRACH 수신 수 : {gs.get('prach_received', 0)}")
        if gs.get('errors'):
            lines.append(f"  에러 수       : {len(gs['errors'])}")
            for e in gs['errors'][:5]:
                lines.append(f"    - {e}")
    else:
        lines.append("  (데이터 없음)")

    lines.append("")
    lines.append("── UE별 MAC 통계 ──")
    mac = data.get('mac_stats', [])
    if mac:
        for i, ue in enumerate(mac):
            lines.append(f"  UE #{i} (RNTI={ue.get('rnti', '?')}):")
            lines.append(f"    CQI       : {ue.get('cqi', '?')}")
            lines.append(f"    RI        : {ue.get('ri', '?')}")
            lines.append(f"    DL MCS    : {ue.get('dl_mcs', '?')}")
            lines.append(f"    UL MCS    : {ue.get('ul_mcs', '?')}")
            lines.append(f"    DL BLER   : {ue.get('dl_bler', '?')}")
            lines.append(f"    UL BLER   : {ue.get('ul_bler', '?')}")
            lines.append(f"    RSRP      : {ue.get('rsrp', '?')} dBm")
            lines.append(f"    SNR       : {ue.get('snr_db', '?')} dB")
            lines.append(f"    DL 바이트  : {ue.get('dl_total_bytes', '?')}")
            lines.append(f"    UL 바이트  : {ue.get('ul_total_bytes', '?')}")
            lines.append(f"    DLSCH 에러 : {ue.get('dlsch_errors', '?')}")
            lines.append(f"    ULSCH 에러 : {ue.get('ulsch_errors', '?')}")
    else:
        lines.append("  (데이터 없음 — UE가 연결되지 않았을 수 있음)")

    lines.append("=" * 60)
    return "\n".join(lines)


def _safe_avg(values: list, key: str):
    """리스트의 dict에서 key 값의 평균을 구한다."""
    vals = [d[key] for d in values if key in d and d[key] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def format_comparison_report(data1: dict, data2: dict) -> str:
    """두 실행 결과를 비교 테이블로 출력한다."""
    lines = []
    lines.append("=" * 72)
    lines.append("  1 UE vs 4 UE End-to-End 5G 성능 비교 리포트")
    lines.append("=" * 72)

    m1 = data1.get('metadata', {})
    m2 = data2.get('metadata', {})
    ps1_info = data1.get('proxy_stats', {})
    ps2_info = data2.get('proxy_stats', {})

    n1 = m1.get('num_ues') or ps1_info.get('num_ues') or '?'
    n2 = m2.get('num_ues') or ps2_info.get('num_ues') or '?'
    mode1 = m1.get('mode') or ps1_info.get('mode') or '?'
    mode2 = m2.get('mode') or ps2_info.get('mode') or '?'
    label1 = f"{n1} UE"
    label2 = f"{n2} UE"

    lines.append("")
    lines.append(f"  설정 A: {label1} ({mode1}, {m1.get('proxy_ver', '?')})")
    lines.append(f"  설정 B: {label2} ({mode2}, {m2.get('proxy_ver', '?')})")
    lines.append("")

    mac1 = data1.get('mac_stats', [])
    mac2 = data2.get('mac_stats', [])

    def avg_metric(mac_list, key):
        vals = [ue[key] for ue in mac_list if key in ue and ue[key] is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def fmt(val, decimal=1):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{decimal}f}"
        return str(val)

    def diff_str(v1, v2, higher_better=True):
        if v1 is None or v2 is None:
            return "N/A"
        d = v2 - v1
        pct = (d / abs(v1) * 100) if v1 != 0 else float('inf')
        sign = "+" if d > 0 else ""
        quality = ""
        if abs(pct) > 1:
            if (d > 0 and higher_better) or (d < 0 and not higher_better):
                quality = " (better)"
            else:
                quality = " (worse)"
        return f"{sign}{d:.1f} ({sign}{pct:.1f}%){quality}"

    metrics = [
        ("Avg CQI",          'cqi',            True),
        ("DL MCS",           'dl_mcs',         True),
        ("UL MCS",           'ul_mcs',         True),
        ("DL BLER",          'dl_bler',        False),
        ("UL BLER",          'ul_bler',        False),
        ("RSRP (dBm)",       'rsrp',           True),
        ("SNR (dB)",         'snr_db',         True),
        ("DL 바이트 (avg)",   'dl_total_bytes', True),
        ("UL 바이트 (avg)",   'ul_total_bytes', True),
        ("DLSCH 에러 (avg)",  'dlsch_errors',   False),
        ("ULSCH 에러 (avg)",  'ulsch_errors',   False),
    ]

    header = f"{'메트릭':<20} {'':>2}{label1:>12} {label2:>12} {'차이':>24}"
    lines.append("── UE MAC 메트릭 비교 (UE 평균) ──")
    lines.append(header)
    lines.append("-" * 72)

    for name, key, higher_better in metrics:
        v1 = avg_metric(mac1, key)
        v2 = avg_metric(mac2, key)
        d = diff_str(v1, v2, higher_better)
        lines.append(f"{name:<20}   {fmt(v1):>12} {fmt(v2):>12} {d:>24}")

    lines.append("")
    lines.append("── Proxy 지연 비교 ──")

    ps1 = data1.get('proxy_stats', {})
    ps2 = data2.get('proxy_stats', {})

    proxy_metrics = [
        ("E2E 평균 지연 (ms)", 'e2e_avg_ms', False),
        ("E2E 최대 지연 (ms)", 'e2e_max_ms', False),
        ("DL 평균 지연 (ms)",  'dl_avg_ms',  False),
        ("UL 평균 지연 (ms)",  'ul_avg_ms',  False),
        ("처리 프레임 수",     'total_frames', True),
    ]

    header = f"{'메트릭':<20} {'':>2}{label1:>12} {label2:>12} {'차이':>24}"
    lines.append(header)
    lines.append("-" * 72)

    for name, key, higher_better in proxy_metrics:
        v1 = ps1.get(key)
        v2 = ps2.get(key)
        d = diff_str(v1, v2, higher_better)
        lines.append(f"{name:<20}   {fmt(v1, 2):>12} {fmt(v2, 2):>12} {d:>24}")

    lines.append("")
    lines.append("── gNB 이벤트 비교 ──")
    gs1 = data1.get('gnb_stats', {})
    gs2 = data2.get('gnb_stats', {})
    lines.append(f"  {'항목':<20} {label1:>12} {label2:>12}")
    lines.append(f"  {'RRC 연결':<20} {gs1.get('rrc_connections', 'N/A'):>12} {gs2.get('rrc_connections', 'N/A'):>12}")
    lines.append(f"  {'PRACH 수신':<20} {gs1.get('prach_received', 'N/A'):>12} {gs2.get('prach_received', 'N/A'):>12}")
    lines.append(f"  {'에러':<20} {len(gs1.get('errors', [])):>12} {len(gs2.get('errors', [])):>12}")

    lines.append("")
    lines.append("=" * 72)

    result_json = {
        'label_a': label1,
        'label_b': label2,
        'mac_comparison': {},
        'proxy_comparison': {},
    }
    for name, key, _ in metrics:
        result_json['mac_comparison'][key] = {
            'a': avg_metric(mac1, key),
            'b': avg_metric(mac2, key),
        }
    for name, key, _ in proxy_metrics:
        result_json['proxy_comparison'][key] = {
            'a': ps1.get(key),
            'b': ps2.get(key),
        }

    return "\n".join(lines), result_json


def main():
    ap = argparse.ArgumentParser(description="OAI 5G 벤치마크 통계 파서")
    ap.add_argument("result_dirs", nargs="+", help="결과 디렉토리 경로(들)")
    ap.add_argument("--compare", action="store_true",
                    help="두 디렉토리를 비교 모드로 분석")
    ap.add_argument("--json", action="store_true",
                    help="JSON 형식으로 출력")
    ap.add_argument("--output", "-o", type=str, default=None,
                    help="결과를 파일로 저장")
    args = ap.parse_args()

    if args.compare:
        if len(args.result_dirs) != 2:
            print("ERROR: --compare 모드에서는 정확히 2개의 디렉토리가 필요합니다.", file=sys.stderr)
            sys.exit(1)

        data1 = parse_result_dir(args.result_dirs[0])
        data2 = parse_result_dir(args.result_dirs[1])

        report_text, report_json = format_comparison_report(data1, data2)

        if args.json:
            output = json.dumps(report_json, indent=2, ensure_ascii=False)
        else:
            output = report_text

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"리포트 저장: {args.output}")
        else:
            print(output)
    else:
        for d in args.result_dirs:
            data = parse_result_dir(d)

            if args.json:
                output = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            else:
                output = format_single_report(data)

            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"리포트 저장: {args.output}")
            else:
                print(output)


if __name__ == "__main__":
    main()

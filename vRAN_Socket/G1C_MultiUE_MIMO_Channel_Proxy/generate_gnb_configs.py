#!/usr/bin/env python3
"""
generate_gnb_configs.py — 멀티셀 gNB 설정 파일 자동 생성

기존 gnb.conf 템플릿을 읽어 셀별로 고유한 파라미터를 갖는 설정 파일을 생성.

사용법:
    python3 generate_gnb_configs.py \
        --template /path/to/gnb.sa.band78.fr1.106PRB.usrpb210.conf \
        --num-cells 3 \
        --output-dir /tmp/multicell_configs

변경되는 파라미터 (셀 c):
    - gNB_ID = 0xe00 + c
    - nr_cellid = 12345678 + c
    - physCellId = c
    - prach_RootSequenceIndex = 1 + c*10  (PRACH 충돌 방지)
"""
import argparse
import os
import re
import sys


def patch_config(template_text: str, cell_idx: int, total_cells: int) -> str:
    """Apply per-cell parameter patches to the template config text."""
    text = template_text

    text = re.sub(
        r'(gNB_ID\s*=\s*)0x[0-9a-fA-F]+',
        rf'\g<1>0x{0xe00 + cell_idx:x}',
        text, count=1)

    text = re.sub(
        r'(nr_cellid\s*=\s*)\d+L?',
        rf'\g<1>{12345678 + cell_idx}L',
        text, count=1)

    text = re.sub(
        r'(physCellId\s*=\s*)\d+',
        rf'\g<1>{cell_idx}',
        text, count=1)

    text = re.sub(
        r'(prach_RootSequenceIndex\s*=\s*)\d+',
        rf'\g<1>{1 + cell_idx * 10}',
        text, count=1)

    cell_name = f"gNB-Cell{cell_idx}"

    text = re.sub(
        r'(gNB_name\s*=\s*)"[^"]*"',
        rf'\1"{cell_name}"',
        text, count=1)

    text = re.sub(
        r'(Active_gNBs\s*=\s*\(\s*)"[^"]*"',
        rf'\1"{cell_name}"',
        text, count=1)

    gtpu_port = 2152 + cell_idx
    text = re.sub(
        r'(GNB_PORT_FOR_S1U\s*=\s*)\d+',
        rf'\g<1>{gtpu_port}',
        text, count=1)

    return text


def main():
    parser = argparse.ArgumentParser(description='멀티셀 gNB 설정 파일 생성기')
    parser.add_argument('--template', required=True,
                        help='기존 gnb.conf 템플릿 파일 경로')
    parser.add_argument('--num-cells', type=int, required=True,
                        help='생성할 셀 수')
    parser.add_argument('--output-dir', required=True,
                        help='출력 디렉토리')
    args = parser.parse_args()

    if not os.path.isfile(args.template):
        print(f"ERROR: 템플릿 파일을 찾을 수 없습니다: {args.template}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.template, 'r') as f:
        template_text = f.read()

    generated = []
    for c in range(args.num_cells):
        patched = patch_config(template_text, c, args.num_cells)
        out_name = f"gnb_cell{c}.conf"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, 'w') as f:
            f.write(patched)
        generated.append(out_path)
        print(f"[gen] Cell {c}: {out_path} "
              f"(PCI={c}, gNB_ID=0x{0xe00 + c:x}, "
              f"prach_root={1 + c * 10}, "
              f"gtpu_port={2152 + c})")

    print(f"\n[gen] {args.num_cells}개 셀 설정 파일 생성 완료 → {args.output_dir}/")
    for p in generated:
        print(f"  {p}")


if __name__ == '__main__':
    main()

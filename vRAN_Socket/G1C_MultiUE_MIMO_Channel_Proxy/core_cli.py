#!/usr/bin/env python3
"""Core Emulator CLI client.

Usage examples:
    # Show full config
    python3 core_cli.py config

    # Show current status
    python3 core_cli.py status

    # Get launch params for shell script
    python3 core_cli.py launch-params

    # Get proxy params
    python3 core_cli.py proxy-params

    # Render gnb.conf
    python3 core_cli.py gnb-conf
    python3 core_cli.py gnb-conf -o /tmp/gnb.conf   # save to file

    # Runtime: update proxy params (hot-swap, no restart)
    python3 core_cli.py update channel.snr_dB=20
    python3 core_cli.py update channel.path_loss_dB=3.0 channel.speed=10

    # Runtime: update gNB params (triggers managed restart)
    python3 core_cli.py update --restart-gnb codebook.type=type1 antenna.gnb.polarization=single

    # Show specific section
    python3 core_cli.py get antenna
    python3 core_cli.py get codebook
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any, Dict, Optional


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 7100


def send_cmd(host: str, port: int, cmd: str, **payload) -> dict:
    """Connect, send a single command, receive the response, and disconnect."""
    req = {"cmd": cmd, **payload}
    data = json.dumps(req, ensure_ascii=False, default=str) + "\n"
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(data.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks).decode("utf-8").strip()
    if not raw:
        return {"status": "error", "msg": "empty response"}
    return json.loads(raw)


def _pretty(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)


def _parse_kv(args: list[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE pairs. Automatically converts numeric types."""
    result = {}
    for arg in args:
        if "=" not in arg:
            print(f"ERROR: invalid key=value: {arg}", file=sys.stderr)
            sys.exit(1)
        k, v = arg.split("=", 1)
        if v.lower() == "null" or v.lower() == "none":
            result[k] = None
        elif v.lower() in ("true", "false"):
            result[k] = v.lower() == "true"
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


def cmd_config(args):
    resp = send_cmd(args.host, args.port, "GET_CONFIG")
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)
    print(_pretty(resp["config"]))


def cmd_status(args):
    resp = send_cmd(args.host, args.port, "STATUS")
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)
    for k, v in resp.items():
        if k != "status":
            print(f"  {k}: {v}")


def cmd_launch_params(args):
    resp = send_cmd(args.host, args.port, "GET_LAUNCH_PARAMS")
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)
    params = resp["params"]
    if args.shell:
        for k, v in params.items():
            print(f"{k}={v}")
    else:
        print(_pretty(params))


def cmd_proxy_params(args):
    resp = send_cmd(args.host, args.port, "GET_PROXY_PARAMS")
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)
    print(_pretty(resp["params"]))


def cmd_gnb_conf(args):
    resp = send_cmd(args.host, args.port, "GET_GNB_CONF")
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)
    text = resp["gnb_conf"]
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"gnb.conf written to {args.output}")
    else:
        print(text)


def cmd_update(args):
    updates = _parse_kv(args.params)
    if not updates:
        print("ERROR: no updates provided", file=sys.stderr)
        sys.exit(1)

    cmd = "UPDATE_GNB" if args.restart_gnb else "UPDATE_PROXY"
    resp = send_cmd(args.host, args.port, cmd, updates=updates)
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)

    if args.restart_gnb:
        conf_path = args.save_conf or "/tmp/generated_gnb.conf"
        gnb_conf_text = resp.get("gnb_conf", "")
        if gnb_conf_text:
            with open(conf_path, "w") as f:
                f.write(gnb_conf_text)
            print(f"New gnb.conf saved to {conf_path}")

        print("gNB config updated. Performing managed restart...")
        _managed_gnb_restart(conf_path)
    else:
        print("Proxy params updated (hot-swap).")
        if "proxy_params" in resp:
            print(_pretty(resp["proxy_params"]))


def _managed_gnb_restart(conf_path: str):
    """Stop gNB + UEs, then restart gNB with new config.

    This is a best-effort implementation. The proxy stays alive;
    gNB and UEs are terminated and relaunched.
    """
    import subprocess
    import shlex

    print("[Managed Restart] 1/3: gNB + UE 프로세스 종료...")
    subprocess.run(["pkill", "-9", "-f", "nr-uesoftmodem"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "nr-softmodem"], capture_output=True)

    import time
    time.sleep(3)

    print("[Managed Restart] 2/3: IPC 버퍼 리셋...")
    subprocess.run(["rm", "-f", "/tmp/oai_gpu_ipc/gpu_ipc_shm"], capture_output=True)
    for i in range(64):
        subprocess.run(["rm", "-f", f"/tmp/oai_gpu_ipc/gpu_ipc_shm_ue{i}"], capture_output=True)

    print(f"[Managed Restart] 3/3: 새 gnb.conf ({conf_path})로 gNB 재시작 필요.")
    print(f"  gNB를 수동으로 재시작하거나, launch_all.sh -c 로 전체 재기동하세요:")
    print(f"    sudo bash launch_all.sh -c <CORE_EMULATOR_ADDR>")
    print(f"  또는 기존 로그 디렉토리에서:")
    proj_dir = "/home/dclserver78/oai_sionna_junxiu"
    build_dir = f"{proj_dir}/openairinterface5g_whan/cmake_targets/ran_build/build"
    print(f"    RFSIM_GPU_IPC_V6=1 {build_dir}/nr-softmodem \\")
    print(f"      -O {conf_path} --gNBs.[0].min_rxtxtime 6 --rfsim")


def cmd_get(args):
    resp = send_cmd(args.host, args.port, "GET_CONFIG")
    if resp.get("status") != "ok":
        print(f"ERROR: {resp.get('msg')}", file=sys.stderr)
        sys.exit(1)
    cfg = resp["config"]
    section = cfg.get(args.section)
    if section is None:
        print(f"ERROR: section '{args.section}' not found. "
              f"Available: {list(cfg.keys())}", file=sys.stderr)
        sys.exit(1)
    print(_pretty(section))


def main():
    ap = argparse.ArgumentParser(
        description="Core Emulator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)

    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="Show full configuration")
    sub.add_parser("status", help="Show emulator status")

    lp = sub.add_parser("launch-params", help="Get launch parameters")
    lp.add_argument("--shell", action="store_true",
                    help="Output as KEY=VALUE (for eval in bash)")

    sub.add_parser("proxy-params", help="Get proxy parameters")

    gc = sub.add_parser("gnb-conf", help="Render gnb.conf from template")
    gc.add_argument("-o", "--output", help="Save to file instead of stdout")

    up = sub.add_parser("update", help="Update parameters at runtime")
    up.add_argument("params", nargs="+", metavar="KEY=VALUE",
                    help="Dotted key=value pairs (e.g. channel.snr_dB=20)")
    up.add_argument("--restart-gnb", action="store_true",
                    help="Mark as gNB update (triggers managed restart)")
    up.add_argument("--save-conf", metavar="PATH",
                    help="Save regenerated gnb.conf to file")

    gt = sub.add_parser("get", help="Get a specific config section")
    gt.add_argument("section", help="Section name (antenna, codebook, channel, ...)")

    args = ap.parse_args()

    dispatch = {
        "config": cmd_config,
        "status": cmd_status,
        "launch-params": cmd_launch_params,
        "proxy-params": cmd_proxy_params,
        "gnb-conf": cmd_gnb_conf,
        "update": cmd_update,
        "get": cmd_get,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

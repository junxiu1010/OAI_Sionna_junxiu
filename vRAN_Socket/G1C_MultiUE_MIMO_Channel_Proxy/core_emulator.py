#!/usr/bin/env python3
"""Core Emulator – TCP/JSON server that manages all PHY/channel configuration.

Reads master_config.yaml and serves configuration to launch_all.sh, proxy (v0.py),
and provides runtime parameter update capabilities.

Protocol:
    Each TCP message is a JSON object terminated by newline (\n).
    Request:  {"cmd": "<CMD>", ...optional payload...}
    Response: {"status": "ok"|"error", ...payload...}

Commands:
    GET_CONFIG         – return full config dict
    GET_GNB_CONF       – return rendered gnb.conf text
    GET_LAUNCH_PARAMS  – return shell-friendly launch parameters
    UPDATE_PROXY       – update proxy-hot-swappable params at runtime
    UPDATE_GNB         – update gNB params (triggers managed restart)
    STATUS             – return current state
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    sys.exit("ERROR: jinja2 is required.  pip install jinja2")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
DEFAULT_CONFIG = BASE_DIR / "master_config.yaml"
DEFAULT_PORT = 7100

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (mutates base)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


SCS_MAP = {15: 0, 30: 1, 60: 2, 120: 3}


def _derived(cfg: dict) -> dict:
    """Compute derived values used by templates and launch script."""
    ant = cfg["antenna"]
    gnb_nx = ant["gnb"]["nx"]
    gnb_ny = ant["gnb"]["ny"]
    pol = ant["gnb"].get("polarization", "single")
    pol_mult = 2 if pol == "dual" else 1
    gnb_spatial = gnb_nx * gnb_ny
    gnb_ant = gnb_spatial * pol_mult

    ue_nx = ant["ue"]["nx"]
    ue_ny = ant["ue"]["ny"]
    ue_ant = ue_nx * ue_ny * pol_mult

    xp = pol_mult
    n1 = gnb_spatial if xp == 2 else gnb_ant

    scs_khz = cfg["carrier"]["scs_kHz"]
    scs_idx = SCS_MAP.get(scs_khz, 1)

    return {
        "gnb_nx": gnb_nx,
        "gnb_ny": gnb_ny,
        "ue_nx": ue_nx,
        "ue_ny": ue_ny,
        "pol_mult": pol_mult,
        "gnb_spatial": gnb_spatial,
        "gnb_ant": gnb_ant,
        "ue_ant": ue_ant,
        "xp": xp,
        "n1": n1,
        "polarization": pol,
        "scs_idx": scs_idx,
    }


# ---------------------------------------------------------------------------
# gnb.conf renderer
# ---------------------------------------------------------------------------

def render_gnb_conf(cfg: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("gnb.conf.j2")
    d = _derived(cfg)
    ctx = {**cfg, **d}
    return tmpl.render(**ctx)


# ---------------------------------------------------------------------------
# Launch params (shell-friendly flat dict)
# ---------------------------------------------------------------------------

def get_launch_params(cfg: dict) -> dict:
    d = _derived(cfg)
    sys_cfg = cfg["system"]
    ch_cfg = cfg["channel"]
    return {
        "NUM_UES": sys_cfg["num_ues"],
        "CHANNEL_MODE": sys_cfg.get("channel_mode", "static"),
        "BATCH_SIZE": sys_cfg.get("batch_size", 4),
        "GNB_NX": d["gnb_nx"],
        "GNB_NY": d["gnb_ny"],
        "UE_NX": d["ue_nx"],
        "UE_NY": d["ue_ny"],
        "POLARIZATION": d["polarization"],
        "POL_MULT": d["pol_mult"],
        "GNB_ANT": d["gnb_ant"],
        "UE_ANT": d["ue_ant"],
        "XP": d["xp"],
        "N1": d["n1"],
        "GNB_SPATIAL": d["gnb_spatial"],
        "SCS_IDX": d["scs_idx"],
        "CARRIER_BAND": cfg["carrier"]["band"],
        "CARRIER_BW_PRB": cfg["carrier"]["bandwidth_prb"],
        "CARRIER_FREQ_GHZ": cfg["carrier"]["frequency_GHz"],
        "PATH_LOSS_DB": ch_cfg.get("path_loss_dB", 0.0),
        "SPEED": ch_cfg.get("speed", 3.0),
        "SCENARIO": ch_cfg.get("scenario", "UMa-NLOS"),
        "CODEBOOK_TYPE": cfg["codebook"]["type"],
    }


# ---------------------------------------------------------------------------
# Proxy params (subset relevant to Sionna proxy)
# ---------------------------------------------------------------------------

def get_proxy_params(cfg: dict) -> dict:
    d = _derived(cfg)
    ch = cfg["channel"]
    return {
        "num_ues": cfg["system"]["num_ues"],
        "channel_mode": cfg["system"].get("channel_mode", "static"),
        "gnb_nx": d["gnb_nx"],
        "gnb_ny": d["gnb_ny"],
        "ue_nx": d["ue_nx"],
        "ue_ny": d["ue_ny"],
        "polarization": d["polarization"],
        "gnb_ant": d["gnb_ant"],
        "ue_ant": d["ue_ant"],
        "path_loss_dB": ch.get("path_loss_dB", 0.0),
        "snr_dB": ch.get("snr_dB"),
        "noise_dBFS": ch.get("noise_dBFS"),
        "speed": ch.get("speed", 3.0),
        "scenario": ch.get("scenario", "UMa-NLOS"),
        "carrier_frequency_GHz": cfg["carrier"]["frequency_GHz"],
        "sector_half_deg": ch.get("sector_half_deg", 60.0),
        "jitter_std_deg": ch.get("jitter_std_deg", 10.0),
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CoreState:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.lock = threading.Lock()
        self.version = 1
        self.proxy_clients: list[socketserver.BaseRequestHandler] = []
        self.gnb_restart_requested = False
        self._start_time = time.time()

    def get_config(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.cfg)

    def update_section(self, section: str, updates: dict) -> dict:
        with self.lock:
            if section not in self.cfg:
                raise KeyError(f"unknown section: {section}")
            if isinstance(self.cfg[section], dict):
                deep_update(self.cfg[section], updates)
            else:
                self.cfg[section] = updates
            self.version += 1
            return copy.deepcopy(self.cfg[section])

    def update_flat(self, dotted_key: str, value: Any) -> None:
        """Update a config value using dotted notation, e.g. 'channel.snr_dB'."""
        parts = dotted_key.split(".")
        with self.lock:
            obj = self.cfg
            for p in parts[:-1]:
                obj = obj[p]
            obj[parts[-1]] = value
            self.version += 1

    def status(self) -> dict:
        with self.lock:
            return {
                "version": self.version,
                "uptime_s": round(time.time() - self._start_time, 1),
                "num_ues": self.cfg["system"]["num_ues"],
                "gnb_restart_requested": self.gnb_restart_requested,
            }


# ---------------------------------------------------------------------------
# TCP Handler
# ---------------------------------------------------------------------------

class JsonHandler(socketserver.StreamRequestHandler):
    """One connection per client. Reads newline-delimited JSON."""

    def handle(self):
        state: CoreState = self.server.state  # type: ignore[attr-defined]
        peer = self.client_address
        print(f"[Core] client connected: {peer}")
        try:
            for raw_line in self.rfile:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    self._reply({"status": "error", "msg": f"bad JSON: {e}"})
                    continue
                cmd = req.get("cmd", "").upper()
                try:
                    resp = self._dispatch(cmd, req, state)
                except Exception as e:
                    resp = {"status": "error", "msg": str(e)}
                self._reply(resp)
        except (ConnectionResetError, BrokenPipeError):
            pass
        print(f"[Core] client disconnected: {peer}")

    def _dispatch(self, cmd: str, req: dict, state: CoreState) -> dict:
        if cmd == "GET_CONFIG":
            return {"status": "ok", "config": state.get_config()}

        elif cmd == "GET_GNB_CONF":
            text = render_gnb_conf(state.get_config())
            return {"status": "ok", "gnb_conf": text}

        elif cmd == "GET_LAUNCH_PARAMS":
            return {"status": "ok", "params": get_launch_params(state.get_config())}

        elif cmd == "GET_PROXY_PARAMS":
            return {"status": "ok", "params": get_proxy_params(state.get_config())}

        elif cmd == "UPDATE_PROXY":
            updates = req.get("updates", {})
            if not updates:
                return {"status": "error", "msg": "no updates provided"}
            for dotted, val in updates.items():
                state.update_flat(dotted, val)
            new_proxy = get_proxy_params(state.get_config())
            self._broadcast_proxy_update(state, new_proxy)
            return {"status": "ok", "proxy_params": new_proxy}

        elif cmd == "UPDATE_GNB":
            updates = req.get("updates", {})
            if not updates:
                return {"status": "error", "msg": "no updates provided"}
            for dotted, val in updates.items():
                state.update_flat(dotted, val)
            with state.lock:
                state.gnb_restart_requested = True
            gnb_conf = render_gnb_conf(state.get_config())
            return {
                "status": "ok",
                "msg": "gNB config updated – restart required",
                "gnb_conf": gnb_conf,
            }

        elif cmd == "SUBSCRIBE_PROXY":
            state.proxy_clients.append(self)
            return {"status": "ok", "msg": "subscribed to proxy updates"}

        elif cmd == "STATUS":
            return {"status": "ok", **state.status()}

        else:
            return {"status": "error", "msg": f"unknown cmd: {cmd}"}

    def _reply(self, obj: dict):
        data = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
        try:
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def _broadcast_proxy_update(state: CoreState, params: dict):
        """Push to all registered proxy listener connections (Phase 2)."""
        msg = json.dumps({"event": "PROXY_UPDATE", "params": params},
                         ensure_ascii=False, default=str) + "\n"
        dead = []
        for client in state.proxy_clients:
            try:
                client.wfile.write(msg.encode("utf-8"))  # type: ignore[attr-defined]
                client.wfile.flush()  # type: ignore[attr-defined]
            except Exception:
                dead.append(client)
        for c in dead:
            state.proxy_clients.remove(c)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ---------------------------------------------------------------------------
# HTTP convenience endpoints (optional, simple GET handler)
# ---------------------------------------------------------------------------

def start_http_api(state: CoreState, port: int):
    """Minimal HTTP wrapper for curl-friendly access from launch_all.sh."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path.strip("/")
            if path == "config":
                body = json.dumps(state.get_config(), ensure_ascii=False, default=str)
            elif path == "gnb_conf":
                body = render_gnb_conf(state.get_config())
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
                return
            elif path == "launch_params":
                body = json.dumps(get_launch_params(state.get_config()),
                                  ensure_ascii=False, default=str)
            elif path == "proxy_params":
                body = json.dumps(get_proxy_params(state.get_config()),
                                  ensure_ascii=False, default=str)
            elif path == "status":
                body = json.dumps(state.status(), ensure_ascii=False, default=str)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"not found\n")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            self.wfile.write(b"\n")

    srv = HTTPServer(("0.0.0.0", port + 1), Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[Core] HTTP API listening on port {port + 1} (GET /config, /gnb_conf, /launch_params, /proxy_params, /status)")
    return srv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Core Emulator – central config server")
    ap.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG,
                    help="Path to master_config.yaml")
    ap.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                    help="TCP JSON port (default: %(default)s)")
    ap.add_argument("--render-gnb-conf", metavar="OUT",
                    help="Render gnb.conf to file and exit (no server)")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        sys.exit(f"ERROR: config not found: {cfg_path}")
    print(f"[Core] loading config: {cfg_path}")

    state = CoreState(cfg_path)
    d = _derived(state.cfg)
    print(f"[Core] system: UEs={state.cfg['system']['num_ues']}  "
          f"gNB={d['gnb_ant']}T{d['ue_ant']}R  pol={d['polarization']}  "
          f"codebook={state.cfg['codebook']['type']}")

    if args.render_gnb_conf:
        out = Path(args.render_gnb_conf)
        text = render_gnb_conf(state.get_config())
        out.write_text(text)
        print(f"[Core] gnb.conf rendered → {out}")
        return

    srv = ThreadedTCPServer(("0.0.0.0", args.port), JsonHandler)
    srv.state = state  # type: ignore[attr-defined]

    http_srv = start_http_api(state, args.port)

    def _shutdown(sig, frame):
        print("\n[Core] shutting down...")
        srv.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[Core] TCP JSON server listening on port {args.port}")
    print(f"[Core] ready.")
    srv.serve_forever()


if __name__ == "__main__":
    main()

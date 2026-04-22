#!/usr/bin/env python3
"""Core Emulator — API-based 5G RAN configuration server.

Provides a FastAPI REST interface + legacy TCP/JSON interface for managing
all PHY/channel/RRC parameters.  Designed so the API shape matches a real
core-network NMS, allowing future swap to a real 5GC without client changes.

REST API (default port 7101):
    POST /api/v1/intent          — natural-language intent → config
    POST /api/v1/config          — direct parameter update
    POST /api/v1/apply-preset    — apply a named preset profile
    POST /api/v1/apply           — render gnb.conf + trigger restart
    GET  /api/v1/presets          — list available presets
    GET  /api/v1/config           — current full config
    GET  /api/v1/gnb-conf         — rendered gnb.conf text
    GET  /api/v1/message-map      — 3GPP message mapping for current config
    GET  /api/v1/status           — server status
    POST /api/v1/cell/configure   — configure a cell (multi-cell)
    POST /api/v1/cell/{id}/activate
    GET  /api/v1/cell/{id}/status
    GET  /api/v1/kpi
    GET  /api/v1/bearer           — current bearer/QoS settings
    POST /api/v1/bearer           — update bearer settings (gNB side)
    POST /api/v1/qos              — update QoS profile (SMF side)
    POST /api/v1/qos/apply        — render CN5G config + restart SMF
    POST /api/v1/qos/db-update    — update MySQL subscriber QoS data

Legacy TCP/JSON (default port 7100):
    Same protocol as before for backward compatibility with v0.py proxy.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    sys.exit("ERROR: jinja2 required.  pip install jinja2")

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from param_validator import validate_config, ValidationError as VErr
from message_mapper import (
    get_message_map, format_change_summary, get_hotswap_keys,
)
from intent_parser import parse_intent
from traffic_emulator import (
    TrafficEmulator, TrafficProfile, TrafficPattern, Direction,
    load_from_config as load_traffic_from_config, profile_from_dict,
)

if HAS_FASTAPI:
    class IntentRequest(BaseModel):
        text: str

    class ConfigUpdateRequest(BaseModel):
        updates: Dict[str, Any]

    class PresetApplyRequest(BaseModel):
        preset: str
        overrides: Optional[Dict[str, Any]] = None

    class CellConfigRequest(BaseModel):
        cell_id: int
        overrides: Optional[Dict[str, Any]] = None

    class BearerUpdateRequest(BaseModel):
        enable_sdap: Optional[bool] = None
        drbs: Optional[int] = None
        um_on_default_drb: Optional[bool] = None
        drb_ciphering: Optional[bool] = None
        drb_integrity: Optional[bool] = None

    class QosUpdateRequest(BaseModel):
        default_5qi: Optional[int] = None
        session_ambr_ul: Optional[str] = None
        session_ambr_dl: Optional[str] = None
        arp_priority: Optional[int] = None
        arp_preempt_cap: Optional[str] = None
        arp_preempt_vuln: Optional[str] = None

    class DbQosUpdateRequest(BaseModel):
        imsi_list: Optional[List[str]] = None
        default_5qi: Optional[int] = None
        session_ambr_ul: Optional[str] = None
        session_ambr_dl: Optional[str] = None

    class TrafficStartRequest(BaseModel):
        ue_idx: Optional[int] = None  # None = all UEs

    class TrafficProfileRequest(BaseModel):
        pattern: Optional[str] = None
        direction: Optional[str] = None
        bitrate_mbps: Optional[float] = None
        protocol: Optional[str] = None
        duration_s: Optional[float] = None
        burst_on_ms: Optional[float] = None
        burst_off_ms: Optional[float] = None
        burst_rate_mbps: Optional[float] = None
        dscp: Optional[int] = None
        speed_kmh: Optional[float] = None

    class TrafficUeProfileRequest(BaseModel):
        ue_idx: int
        profile: Dict[str, Any]

    class CsiNetConfigRequest(BaseModel):
        enabled: Optional[bool] = None
        mode: Optional[str] = None
        compression_ratio: Optional[float] = None
        scenario: Optional[str] = None
        checkpoint_dir: Optional[str] = None
        csi_rs_period: Optional[int] = None
        csinet_path: Optional[str] = None
        diff_enabled: Optional[bool] = None
        diff_threshold: Optional[float] = None
        diff_max_stale_slots: Optional[int] = None

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
PRESETS_DIR = BASE_DIR / "presets"
DEFAULT_CONFIG = BASE_DIR / "master_config.yaml"
DEFAULT_TCP_PORT = 7100
DEFAULT_HTTP_PORT = 7101

SCS_MAP = {15: 0, 30: 1, 60: 2, 120: 3}


# ═══════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _derived(cfg: dict) -> dict:
    ant = cfg.get("antenna", {})
    gnb = ant.get("gnb", {})
    pol = gnb.get("polarization", "single")
    pol_mult = 2 if pol == "dual" else 1
    gnb_nx = gnb.get("nx", 1)
    gnb_ny = gnb.get("ny", 1)
    gnb_spatial = gnb_nx * gnb_ny
    gnb_ant = gnb_spatial * pol_mult

    ue = ant.get("ue", {})
    ue_nx = ue.get("nx", 1)
    ue_ny = ue.get("ny", 1)
    ue_ant = ue_nx * ue_ny * pol_mult

    xp = pol_mult
    n1 = gnb_spatial if xp == 2 else gnb_ant
    scs_khz = cfg.get("carrier", {}).get("scs_kHz", 30)

    return {
        "gnb_nx": gnb_nx, "gnb_ny": gnb_ny,
        "ue_nx": ue_nx, "ue_ny": ue_ny,
        "pol_mult": pol_mult, "gnb_spatial": gnb_spatial,
        "gnb_ant": gnb_ant, "ue_ant": ue_ant,
        "xp": xp, "n1": n1,
        "polarization": pol,
        "scs_idx": SCS_MAP.get(scs_khz, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# gnb.conf renderer
# ═══════════════════════════════════════════════════════════════════

def render_gnb_conf(cfg: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("gnb.conf.j2")
    d = _derived(cfg)
    ctx = {**cfg, **d}
    return tmpl.render(**ctx)


# ═══════════════════════════════════════════════════════════════════
# CN5G config renderer (SMF QoS profile)
# ═══════════════════════════════════════════════════════════════════

CN5G_CONFIG_PATH = Path("/home/dclserver78/DevChannelProxyJIN/openairinterface5g_whan/"
                        "doc/tutorial_resources/oai-cn5g/conf/config.yaml")


def render_cn5g_config(cfg: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("cn5g_config.yaml.j2")
    return tmpl.render(**cfg)


def apply_cn5g_config(cfg: dict) -> dict:
    """Render CN5G config, write to mounted volume, and restart SMF."""
    rendered = render_cn5g_config(cfg)
    try:
        CN5G_CONFIG_PATH.write_text(rendered)
    except Exception as e:
        return {"status": "error", "msg": f"failed to write CN5G config: {e}"}

    try:
        result = subprocess.run(
            ["docker", "restart", "oai-smf"],
            capture_output=True, text=True, timeout=30)
        smf_ok = result.returncode == 0
    except Exception as e:
        return {
            "status": "partial",
            "msg": f"config written but SMF restart failed: {e}",
            "config_path": str(CN5G_CONFIG_PATH),
        }

    return {
        "status": "ok" if smf_ok else "warning",
        "msg": "CN5G config applied" + ("" if smf_ok else " (SMF restart may have failed)"),
        "config_path": str(CN5G_CONFIG_PATH),
        "smf_restart": "success" if smf_ok else result.stderr[:200],
    }


def update_subscriber_qos(imsi_list: Optional[List[str]],
                           fiveqi: int, ambr_ul: str, ambr_dl: str) -> dict:
    """Update QoS profile in MySQL SessionManagementSubscriptionData."""
    db_host = "mysql"
    db_user = "test"
    db_pass = "test"
    db_name = "oai_db"

    dnn_template = (
        '{{"oai":{{"pduSessionTypes":{{"defaultSessionType":"IPV4"}},'
        '"sscModes":{{"defaultSscMode":"SSC_MODE_1"}},'
        '"5gQosProfile":{{"5qi":{fiveqi},"arp":{{"priorityLevel":15,'
        '"preemptCap":"NOT_PREEMPT","preemptVuln":"PREEMPTABLE"}},"priorityLevel":1}},'
        '"sessionAmbr":{{"uplink":"{ambr_ul}","downlink":"{ambr_dl}"}}}}'
        '}}'
    )
    dnn_json = dnn_template.format(fiveqi=fiveqi, ambr_ul=ambr_ul, ambr_dl=ambr_dl)

    if imsi_list:
        placeholders = ",".join(f"'{imsi}'" for imsi in imsi_list)
        where_clause = f"WHERE ueid IN ({placeholders})"
    else:
        where_clause = ""

    sql = f"UPDATE SessionManagementSubscriptionData SET dnnConfigurations='{dnn_json}' {where_clause};"

    try:
        result = subprocess.run(
            ["docker", "exec", "mysql", "mysql",
             f"-u{db_user}", f"-p{db_pass}", db_name,
             "-e", sql],
            capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return {"status": "error", "msg": result.stderr[:300]}
        return {
            "status": "ok",
            "msg": f"Updated QoS: 5QI={fiveqi}, AMBR UL={ambr_ul}, DL={ambr_dl}",
            "affected": "all subscribers" if not imsi_list else imsi_list,
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Launch / Proxy params
# ═══════════════════════════════════════════════════════════════════

def get_launch_params(cfg: dict) -> dict:
    d = _derived(cfg)
    sys_cfg = cfg.get("system", {})
    ch_cfg = cfg.get("channel", {})
    return {
        "NUM_UES": sys_cfg.get("num_ues", 4),
        "CHANNEL_MODE": sys_cfg.get("channel_mode", "static"),
        "BATCH_SIZE": sys_cfg.get("batch_size", 4),
        "GNB_NX": d["gnb_nx"], "GNB_NY": d["gnb_ny"],
        "UE_NX": d["ue_nx"], "UE_NY": d["ue_ny"],
        "POLARIZATION": d["polarization"],
        "POL_MULT": d["pol_mult"],
        "GNB_ANT": d["gnb_ant"], "UE_ANT": d["ue_ant"],
        "XP": d["xp"], "N1": d["n1"],
        "GNB_SPATIAL": d["gnb_spatial"],
        "SCS_IDX": d["scs_idx"],
        "CARRIER_BAND": cfg.get("carrier", {}).get("band", 78),
        "CARRIER_BW_PRB": cfg.get("carrier", {}).get("bandwidth_prb", 106),
        "CARRIER_FREQ_GHZ": cfg.get("carrier", {}).get("frequency_GHz", 3.5),
        "PATH_LOSS_DB": ch_cfg.get("path_loss_dB", 0.0),
        "SPEED": ch_cfg.get("speed", 3.0),
        "SCENARIO": ch_cfg.get("scenario", "UMa-NLOS"),
        "BS_HEIGHT_M": ch_cfg.get("bs_height_m", 25.0),
        "UE_HEIGHT_M": ch_cfg.get("ue_height_m", 1.5),
        "ISD_M": ch_cfg.get("isd_m", 500),
        "MIN_UE_DISTANCE_M": ch_cfg.get("min_ue_distance_m", 35),
        "MAX_UE_DISTANCE_M": ch_cfg.get("max_ue_distance_m", 500),
        "SHADOW_FADING_STD_DB": ch_cfg.get("shadow_fading_std_dB", 6.0),
        "K_FACTOR_MEAN_DB": ch_cfg.get("k_factor_mean_dB"),
        "K_FACTOR_STD_DB": ch_cfg.get("k_factor_std_dB"),
        "CODEBOOK_TYPE": cfg.get("codebook", {}).get("type", "type1"),
        # CsiNet parameters
        "CSINET_ENABLED": cfg.get("csinet", {}).get("enabled", False),
        "CSINET_MODE": cfg.get("csinet", {}).get("mode", "baseline"),
        "CSINET_GAMMA": cfg.get("csinet", {}).get("compression_ratio", 0.25),
        "CSINET_SCENARIO": cfg.get("csinet", {}).get("scenario", "UMi_NLOS"),
        "CSINET_CHECKPOINT_DIR": cfg.get("csinet", {}).get("checkpoint_dir", "/workspace/csinet_checkpoints"),
        "CSINET_PERIOD": cfg.get("csinet", {}).get("csi_rs_period", 20),
        "CSINET_PATH": cfg.get("csinet", {}).get("csinet_path", "/workspace/graduation/csinet"),
        # CsiNet differential encoding
        "CSINET_DIFF_ENABLED": cfg.get("csinet", {}).get("differential", {}).get("enabled", False),
        "CSINET_DIFF_THRESHOLD": cfg.get("csinet", {}).get("differential", {}).get("threshold", 0.01),
        "CSINET_DIFF_MAX_STALE": cfg.get("csinet", {}).get("differential", {}).get("max_stale_slots", 100),
    }


def get_proxy_params(cfg: dict) -> dict:
    d = _derived(cfg)
    ch = cfg.get("channel", {})
    return {
        "num_ues": cfg.get("system", {}).get("num_ues", 4),
        "channel_mode": cfg.get("system", {}).get("channel_mode", "static"),
        "gnb_nx": d["gnb_nx"], "gnb_ny": d["gnb_ny"],
        "ue_nx": d["ue_nx"], "ue_ny": d["ue_ny"],
        "polarization": d["polarization"],
        "gnb_ant": d["gnb_ant"], "ue_ant": d["ue_ant"],
        "path_loss_dB": ch.get("path_loss_dB", 0.0),
        "snr_dB": ch.get("snr_dB"),
        "noise_dBFS": ch.get("noise_dBFS"),
        "speed": ch.get("speed", 3.0),
        "scenario": ch.get("scenario", "UMa-NLOS"),
        "carrier_frequency_GHz": cfg.get("carrier", {}).get("frequency_GHz", 3.5),
        "bs_height_m": ch.get("bs_height_m", 25.0),
        "ue_height_m": ch.get("ue_height_m", 1.5),
        "isd_m": ch.get("isd_m", 500),
        "min_ue_distance_m": ch.get("min_ue_distance_m", 35),
        "max_ue_distance_m": ch.get("max_ue_distance_m", 500),
        "shadow_fading_std_dB": ch.get("shadow_fading_std_dB", 6.0),
        "k_factor_mean_dB": ch.get("k_factor_mean_dB"),
        "k_factor_std_dB": ch.get("k_factor_std_dB"),
        # CsiNet
        "csinet_enabled": cfg.get("csinet", {}).get("enabled", False),
        "csinet_mode": cfg.get("csinet", {}).get("mode", "baseline"),
        "csinet_gamma": cfg.get("csinet", {}).get("compression_ratio", 0.25),
        "csinet_scenario": cfg.get("csinet", {}).get("scenario", "UMi_NLOS"),
        "csinet_checkpoint_dir": cfg.get("csinet", {}).get("checkpoint_dir", "/workspace/csinet_checkpoints"),
        "csinet_period": cfg.get("csinet", {}).get("csi_rs_period", 20),
        "csinet_path": cfg.get("csinet", {}).get("csinet_path", "/workspace/graduation/csinet"),
        # CsiNet differential encoding
        "csinet_diff_enabled": cfg.get("csinet", {}).get("differential", {}).get("enabled", False),
        "csinet_diff_threshold": cfg.get("csinet", {}).get("differential", {}).get("threshold", 0.01),
        "csinet_diff_max_stale": cfg.get("csinet", {}).get("differential", {}).get("max_stale_slots", 100),
    }


# ═══════════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════════

def load_presets() -> Dict[str, dict]:
    presets = {}
    if not PRESETS_DIR.is_dir():
        return presets
    for p in sorted(PRESETS_DIR.glob("*.yaml")):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            key = p.stem
            presets[key] = data
        except Exception as e:
            print(f"[Core] WARNING: failed to load preset {p}: {e}")
    return presets


# ═══════════════════════════════════════════════════════════════════
# Core State
# ═══════════════════════════════════════════════════════════════════

class CoreState:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.lock = threading.Lock()
        self.version = 1
        self.proxy_clients: list = []
        self.gnb_restart_requested = False
        self._start_time = time.time()
        self.presets = load_presets()
        self._cells: Dict[int, dict] = {}
        self.traffic: Optional[TrafficEmulator] = None
        self._init_traffic()

    def get_config(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.cfg)

    def update_section(self, section: str, updates: dict) -> dict:
        with self.lock:
            if section not in self.cfg:
                self.cfg[section] = {}
            if isinstance(self.cfg[section], dict):
                deep_update(self.cfg[section], updates)
            else:
                self.cfg[section] = updates
            self.version += 1
            return copy.deepcopy(self.cfg[section])

    def update_flat(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        with self.lock:
            obj = self.cfg
            for p in parts[:-1]:
                if p not in obj:
                    obj[p] = {}
                obj = obj[p]
            obj[parts[-1]] = value
            self.version += 1

    def apply_preset(self, preset_name: str) -> dict:
        preset = self.presets.get(preset_name)
        if not preset:
            raise KeyError(f"unknown preset: {preset_name}")
        with self.lock:
            deep_update(self.cfg, preset.get("config", {}))
            self.version += 1
            return copy.deepcopy(self.cfg)

    def apply_overrides(self, overrides: Dict[str, Any]) -> List[str]:
        changed = []
        for dotted, val in overrides.items():
            parts = dotted.split(".")
            with self.lock:
                obj = self.cfg
                for p in parts[:-1]:
                    if p not in obj:
                        obj[p] = {}
                    obj = obj[p]
                obj[parts[-1]] = val
            changed.append(dotted)
        with self.lock:
            self.version += 1
        return changed

    def validate(self) -> List[dict]:
        cfg = self.get_config()
        errors = validate_config(cfg)
        return [e.to_dict() for e in errors]

    def _init_traffic(self):
        try:
            self.traffic = load_traffic_from_config(self.cfg)
        except Exception as e:
            print(f"[Core] WARNING: traffic emulator init failed: {e}")
            self.traffic = TrafficEmulator(
                num_ues=self.cfg.get("system", {}).get("num_ues", 4))

    def status(self) -> dict:
        with self.lock:
            d = _derived(self.cfg)
            traffic_active = 0
            if self.traffic:
                ts = self.traffic.get_status()
                traffic_active = ts.get("active_ues", 0)
            return {
                "version": self.version,
                "uptime_s": round(time.time() - self._start_time, 1),
                "num_ues": self.cfg.get("system", {}).get("num_ues", 0),
                "gnb_ant": d["gnb_ant"],
                "ue_ant": d["ue_ant"],
                "codebook_type": self.cfg.get("codebook", {}).get("type"),
                "mu_mimo": self.cfg.get("system", {}).get("mu_mimo", False),
                "gnb_restart_requested": self.gnb_restart_requested,
                "presets_loaded": len(self.presets),
                "cells": len(self._cells),
                "traffic_active_ues": traffic_active,
                "csinet_enabled": self.cfg.get("csinet", {}).get("enabled", False),
                "csinet_mode": self.cfg.get("csinet", {}).get("mode", "baseline"),
            }

    def configure_cell(self, cell_id: int, overrides: Optional[dict] = None) -> dict:
        cfg = self.get_config()
        d = _derived(cfg)
        cell = {
            "cell_id": cell_id,
            "status": "configured",
            "pci": cell_id,
            "gnb_id": f"0x{0xe00 + cell_id:x}",
            "nr_cellid": 12345678 + cell_id,
            "prach_root": 1 + cell_id * 10,
            "gtpu_port": 2152 + cell_id,
            "gnb_ant": d["gnb_ant"],
            "ue_ant": d["ue_ant"],
        }
        if overrides:
            cell.update(overrides)
        with self.lock:
            self._cells[cell_id] = cell
        return cell

    def get_cell(self, cell_id: int) -> Optional[dict]:
        with self.lock:
            return self._cells.get(cell_id)

    def list_cells(self) -> List[dict]:
        with self.lock:
            return list(self._cells.values())


# ═══════════════════════════════════════════════════════════════════
# Legacy TCP/JSON Handler (backward compatibility)
# ═══════════════════════════════════════════════════════════════════

class JsonHandler(socketserver.StreamRequestHandler):
    def handle(self):
        state: CoreState = self.server.state
        peer = self.client_address
        print(f"[Core] TCP client connected: {peer}")
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
        print(f"[Core] TCP client disconnected: {peer}")

    def _dispatch(self, cmd, req, state):
        if cmd == "GET_CONFIG":
            return {"status": "ok", "config": state.get_config()}
        elif cmd == "GET_GNB_CONF":
            return {"status": "ok", "gnb_conf": render_gnb_conf(state.get_config())}
        elif cmd == "GET_LAUNCH_PARAMS":
            return {"status": "ok", "params": get_launch_params(state.get_config())}
        elif cmd == "GET_PROXY_PARAMS":
            return {"status": "ok", "params": get_proxy_params(state.get_config())}
        elif cmd == "UPDATE_PROXY":
            updates = req.get("updates", {})
            if not updates:
                return {"status": "error", "msg": "no updates"}
            for k, v in updates.items():
                state.update_flat(k, v)
            new_proxy = get_proxy_params(state.get_config())
            self._broadcast_proxy_update(state, new_proxy)
            return {"status": "ok", "proxy_params": new_proxy}
        elif cmd == "UPDATE_GNB":
            updates = req.get("updates", {})
            if not updates:
                return {"status": "error", "msg": "no updates"}
            for k, v in updates.items():
                state.update_flat(k, v)
            with state.lock:
                state.gnb_restart_requested = True
            return {
                "status": "ok",
                "msg": "gNB config updated - restart required",
                "gnb_conf": render_gnb_conf(state.get_config()),
            }
        elif cmd == "SUBSCRIBE_PROXY":
            state.proxy_clients.append(self)
            return {"status": "ok", "msg": "subscribed"}
        elif cmd == "STATUS":
            return {"status": "ok", **state.status()}
        else:
            return {"status": "error", "msg": f"unknown cmd: {cmd}"}

    def _reply(self, obj):
        data = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
        try:
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def _broadcast_proxy_update(state, params):
        msg = json.dumps({"event": "PROXY_UPDATE", "params": params},
                         ensure_ascii=False, default=str) + "\n"
        dead = []
        for client in state.proxy_clients:
            try:
                client.wfile.write(msg.encode("utf-8"))
                client.wfile.flush()
            except Exception:
                dead.append(client)
        for c in dead:
            state.proxy_clients.remove(c)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


# ═══════════════════════════════════════════════════════════════════
# FastAPI REST Application
# ═══════════════════════════════════════════════════════════════════

def build_fastapi_app(state: CoreState) -> "FastAPI":
    app = FastAPI(
        title="Core Emulator API",
        description="5G RAN Configuration Server — NMS-compatible REST API",
        version="2.0.0",
    )

    # ── GET endpoints ────────────────────────────────────────────

    @app.get("/api/v1/config")
    def api_get_config():
        return state.get_config()

    @app.get("/api/v1/gnb-conf", response_class=PlainTextResponse)
    def api_get_gnb_conf():
        return render_gnb_conf(state.get_config())

    @app.get("/api/v1/status")
    def api_get_status():
        return state.status()

    @app.get("/api/v1/presets")
    def api_list_presets():
        result = {}
        for name, data in state.presets.items():
            result[name] = {
                "name": data.get("name", name),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
            }
        return {"presets": result}

    @app.get("/api/v1/presets/{name}")
    def api_get_preset(name: str):
        p = state.presets.get(name)
        if not p:
            raise HTTPException(404, f"preset not found: {name}")
        return p

    @app.get("/api/v1/message-map")
    def api_get_message_map():
        return get_message_map()

    @app.get("/api/v1/message-map/{param_key:path}")
    def api_get_param_mapping(param_key: str):
        m = get_message_map()
        info = m.get(param_key)
        if not info:
            raise HTTPException(404, f"no mapping for: {param_key}")
        return info

    @app.get("/api/v1/validate")
    def api_validate():
        errors = state.validate()
        return {
            "valid": all(e["severity"] != "error" for e in errors),
            "issues": errors,
        }

    @app.get("/api/v1/launch-params")
    def api_launch_params():
        return get_launch_params(state.get_config())

    @app.get("/api/v1/proxy-params")
    def api_proxy_params():
        return get_proxy_params(state.get_config())

    @app.get("/api/v1/derived")
    def api_derived():
        return _derived(state.get_config())

    @app.get("/api/v1/hotswap-keys")
    def api_hotswap_keys():
        return {"keys": get_hotswap_keys()}

    # ── POST endpoints ───────────────────────────────────────────

    @app.post("/api/v1/intent")
    def api_intent(req: IntentRequest):
        result = parse_intent(req.text)
        if result.matched_preset and result.matched_preset in state.presets:
            state.apply_preset(result.matched_preset)
        if result.overrides:
            changed = state.apply_overrides(result.overrides)
            impact = format_change_summary(changed)
        else:
            impact = {}
        errors = state.validate()
        return {
            "intent": result.to_dict(),
            "config": state.get_config(),
            "impact": impact,
            "validation": {
                "valid": all(e["severity"] != "error" for e in errors),
                "issues": errors,
            },
        }

    @app.post("/api/v1/config")
    def api_update_config(req: ConfigUpdateRequest):
        changed = state.apply_overrides(req.updates)
        errors = state.validate()
        impact = format_change_summary(changed)
        return {
            "status": "ok",
            "changed": changed,
            "impact": impact,
            "validation": {
                "valid": all(e["severity"] != "error" for e in errors),
                "issues": errors,
            },
            "config": state.get_config(),
        }

    @app.post("/api/v1/apply-preset")
    def api_apply_preset(req: PresetApplyRequest):
        try:
            state.apply_preset(req.preset)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if req.overrides:
            state.apply_overrides(req.overrides)
        errors = state.validate()
        return {
            "status": "ok",
            "preset": req.preset,
            "config": state.get_config(),
            "validation": {
                "valid": all(e["severity"] != "error" for e in errors),
                "issues": errors,
            },
        }

    @app.post("/api/v1/apply")
    def api_apply():
        errors = state.validate()
        has_errors = any(e["severity"] == "error" for e in errors)
        if has_errors:
            raise HTTPException(400, {
                "msg": "config has validation errors",
                "issues": errors,
            })
        gnb_conf = render_gnb_conf(state.get_config())
        with state.lock:
            state.gnb_restart_requested = True
        return {
            "status": "ok",
            "msg": "gnb.conf rendered — restart gNB to apply",
            "gnb_conf_preview": gnb_conf[:500] + "..." if len(gnb_conf) > 500 else gnb_conf,
            "gnb_conf_lines": gnb_conf.count("\n"),
            "requires_restart": True,
        }

    # ── Cell management (NMS-style) ──────────────────────────────

    @app.post("/api/v1/cell/configure")
    def api_cell_configure(req: CellConfigRequest):
        cell = state.configure_cell(req.cell_id, req.overrides)
        return {"status": "ok", "cell": cell}

    @app.post("/api/v1/cell/{cell_id}/activate")
    def api_cell_activate(cell_id: int):
        cell = state.get_cell(cell_id)
        if not cell:
            raise HTTPException(404, f"cell {cell_id} not configured")
        cell["status"] = "active"
        return {"status": "ok", "cell": cell}

    @app.get("/api/v1/cell/{cell_id}/status")
    def api_cell_status(cell_id: int):
        cell = state.get_cell(cell_id)
        if not cell:
            raise HTTPException(404, f"cell {cell_id} not configured")
        return cell

    @app.get("/api/v1/cells")
    def api_list_cells():
        return {"cells": state.list_cells()}

    @app.get("/api/v1/kpi")
    def api_kpi():
        d = _derived(state.get_config())
        mc = state.get_config().get("multicell", {})
        n_cells = mc.get("num_cells", 1) if mc.get("enabled") else 1
        n_ues = state.get_config().get("system", {}).get("num_ues", 1)
        return {
            "total_cells": n_cells,
            "total_ues": n_cells * n_ues,
            "gnb_ant": d["gnb_ant"],
            "ue_ant": d["ue_ant"],
            "max_mimo_layers": d["gnb_ant"],
            "codebook_type": state.get_config().get("codebook", {}).get("type"),
            "mu_mimo": state.get_config().get("system", {}).get("mu_mimo", False),
        }

    # ── Bearer / QoS management ─────────────────────────────────

    @app.get("/api/v1/bearer")
    def api_get_bearer():
        cfg = state.get_config()
        return {
            "bearer": cfg.get("bearer", {}),
            "qos": cfg.get("qos", {}),
        }

    @app.post("/api/v1/bearer")
    def api_update_bearer(req: BearerUpdateRequest):
        updates = {k: v for k, v in req.dict().items() if v is not None}
        if not updates:
            raise HTTPException(400, "no bearer parameters to update")
        for k, v in updates.items():
            state.update_flat(f"bearer.{k}", v)
        cfg = state.get_config()
        return {
            "status": "ok",
            "msg": "bearer config updated — restart gNB to apply",
            "bearer": cfg.get("bearer", {}),
            "requires_restart": True,
        }

    @app.post("/api/v1/qos")
    def api_update_qos(req: QosUpdateRequest):
        updates = {k: v for k, v in req.dict().items() if v is not None}
        if not updates:
            raise HTTPException(400, "no QoS parameters to update")
        for k, v in updates.items():
            state.update_flat(f"qos.{k}", v)
        cfg = state.get_config()
        return {
            "status": "ok",
            "msg": "QoS config updated — call /api/v1/qos/apply to push to SMF",
            "qos": cfg.get("qos", {}),
        }

    @app.post("/api/v1/qos/apply")
    def api_qos_apply():
        cfg = state.get_config()
        result = apply_cn5g_config(cfg)
        return result

    @app.get("/api/v1/cn5g-conf", response_class=PlainTextResponse)
    def api_get_cn5g_conf():
        return render_cn5g_config(state.get_config())

    @app.post("/api/v1/qos/db-update")
    def api_qos_db_update(req: DbQosUpdateRequest):
        cfg = state.get_config()
        qos = cfg.get("qos", {})
        fiveqi = req.default_5qi or qos.get("default_5qi", 9)
        ambr_ul = req.session_ambr_ul or qos.get("session_ambr_ul", "10Gbps")
        ambr_dl = req.session_ambr_dl or qos.get("session_ambr_dl", "10Gbps")
        result = update_subscriber_qos(req.imsi_list, fiveqi, ambr_ul, ambr_dl)
        return result

    # ── Traffic Emulator endpoints ──────────────────────────────

    @app.get("/api/v1/traffic/status")
    def api_traffic_status():
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        return state.traffic.get_status()

    @app.post("/api/v1/traffic/start")
    def api_traffic_start(req: TrafficStartRequest):
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        if req.ue_idx is not None:
            ok = state.traffic.start_ue_traffic(req.ue_idx)
            return {"status": "ok" if ok else "error", "ue_idx": req.ue_idx}
        results = state.traffic.start_all()
        return {
            "status": "ok",
            "results": {str(k): v for k, v in results.items()},
        }

    @app.post("/api/v1/traffic/stop")
    def api_traffic_stop(req: TrafficStartRequest):
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        if req.ue_idx is not None:
            ok = state.traffic.stop_ue_traffic(req.ue_idx)
            return {"status": "ok" if ok else "not_running", "ue_idx": req.ue_idx}
        state.traffic.stop_all()
        return {"status": "ok", "msg": "all traffic stopped"}

    @app.post("/api/v1/traffic/profile")
    def api_traffic_set_default_profile(req: TrafficProfileRequest):
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        d = {k: v for k, v in req.dict().items() if v is not None}
        if not d:
            raise HTTPException(400, "no profile parameters provided")
        current = state.traffic.default_profile.to_dict()
        current.update(d)
        new_profile = profile_from_dict(current)
        state.traffic.set_default_profile(new_profile)
        return {"status": "ok", "profile": new_profile.to_dict()}

    @app.post("/api/v1/traffic/profile/ue")
    def api_traffic_set_ue_profile(req: TrafficUeProfileRequest):
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        profile = profile_from_dict(req.profile)
        state.traffic.set_profile(req.ue_idx, profile)
        return {"status": "ok", "ue_idx": req.ue_idx, "profile": profile.to_dict()}

    @app.get("/api/v1/traffic/results/{ue_idx}")
    def api_traffic_results(ue_idx: int):
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        results = state.traffic.get_results(ue_idx)
        if results is None:
            raise HTTPException(404, f"no results for UE {ue_idx}")
        return results

    @app.get("/api/v1/traffic/speeds")
    def api_traffic_speeds():
        """Per-UE speeds (m/s) for Proxy channel velocity injection."""
        if not state.traffic:
            raise HTTPException(503, "traffic emulator not initialized")
        speeds_ms = state.traffic.get_ue_speeds()
        speeds_kmh = {i: state.traffic.get_profile(i).speed_kmh
                      for i in range(state.traffic.num_ues)}
        return {"speeds_ms": speeds_ms, "speeds_kmh": speeds_kmh}

    # ── CsiNet endpoints ────────────────────────────────────────

    @app.get("/api/v1/csinet/config")
    def api_csinet_config():
        """Current CsiNet configuration."""
        return state.get_config().get("csinet", {})

    @app.post("/api/v1/csinet/config")
    def api_csinet_update(req: CsiNetConfigRequest):
        """Update CsiNet configuration."""
        diff_key_map = {
            "diff_enabled": "csinet.differential.enabled",
            "diff_threshold": "csinet.differential.threshold",
            "diff_max_stale_slots": "csinet.differential.max_stale_slots",
        }
        updates = {k: v for k, v in req.dict().items() if v is not None}
        if not updates:
            raise HTTPException(400, "no CsiNet parameters provided")
        for k, v in updates.items():
            flat_key = diff_key_map.get(k, f"csinet.{k}")
            state.update_flat(flat_key, v)
        return {
            "status": "ok",
            "msg": "CsiNet config updated — restart proxy to apply",
            "csinet": state.get_config().get("csinet", {}),
            "requires_restart": True,
        }

    @app.get("/api/v1/csinet/env")
    def api_csinet_env():
        """CsiNet environment variables for proxy docker exec."""
        csinet_cfg = state.get_config().get("csinet", {})
        diff_cfg = csinet_cfg.get("differential", {})
        enabled = csinet_cfg.get("enabled", False)
        return {
            "CSINET_ENABLED": "1" if enabled else "0",
            "CSINET_MODE": csinet_cfg.get("mode", "baseline"),
            "CSINET_GAMMA": str(csinet_cfg.get("compression_ratio", 0.25)),
            "CSINET_SCENARIO": csinet_cfg.get("scenario", "UMi_NLOS"),
            "CSINET_PERIOD": str(csinet_cfg.get("csi_rs_period", 20)),
            "CSINET_PATH": csinet_cfg.get("csinet_path", "/workspace/graduation/csinet"),
            "CSINET_CHECKPOINT_DIR": csinet_cfg.get("checkpoint_dir", "/workspace/csinet_checkpoints"),
            "CSINET_DIFF_ENABLED": "1" if diff_cfg.get("enabled", False) else "0",
            "CSINET_DIFF_THRESHOLD": str(diff_cfg.get("threshold", 0.01)),
            "CSINET_DIFF_MAX_STALE": str(diff_cfg.get("max_stale_slots", 100)),
        }

    # ── Legacy GET compatibility (same paths as old HTTP server) ─

    @app.get("/config")
    def legacy_config():
        return state.get_config()

    @app.get("/gnb_conf", response_class=PlainTextResponse)
    def legacy_gnb_conf():
        return render_gnb_conf(state.get_config())

    @app.get("/launch_params")
    def legacy_launch_params():
        return get_launch_params(state.get_config())

    @app.get("/proxy_params")
    def legacy_proxy_params():
        return get_proxy_params(state.get_config())

    @app.get("/status")
    def legacy_status():
        return state.status()

    return app


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Core Emulator — 5G RAN config API server")
    ap.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG,
                    help="Path to master_config.yaml")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT,
                    help="Legacy TCP/JSON port (default: %(default)s)")
    ap.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT,
                    help="FastAPI HTTP port (default: %(default)s)")
    ap.add_argument("--render-gnb-conf", metavar="OUT",
                    help="Render gnb.conf to file and exit")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        sys.exit(f"ERROR: config not found: {cfg_path}")

    print(f"[Core] loading config: {cfg_path}")
    state = CoreState(cfg_path)
    d = _derived(state.cfg)
    print(f"[Core] system: UEs={state.cfg.get('system', {}).get('num_ues', '?')}  "
          f"gNB={d['gnb_ant']}T{d['ue_ant']}R  pol={d['polarization']}  "
          f"codebook={state.cfg.get('codebook', {}).get('type', '?')}")
    print(f"[Core] presets loaded: {list(state.presets.keys())}")

    if args.render_gnb_conf:
        out = Path(args.render_gnb_conf)
        out.write_text(render_gnb_conf(state.get_config()))
        print(f"[Core] gnb.conf rendered → {out}")
        return

    errors = state.validate()
    if errors:
        for e in errors:
            print(f"[Core] VALIDATION {e['severity'].upper()}: {e['path']}: {e['message']}")

    tcp_srv = ThreadedTCPServer(("0.0.0.0", args.tcp_port), JsonHandler)
    tcp_srv.state = state
    tcp_thread = threading.Thread(target=tcp_srv.serve_forever, daemon=True)
    tcp_thread.start()
    print(f"[Core] Legacy TCP/JSON server on port {args.tcp_port}")

    if HAS_FASTAPI:
        app = build_fastapi_app(state)
        config = uvicorn.Config(
            app, host="0.0.0.0", port=args.http_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        def _shutdown(sig, frame):
            print("\n[Core] shutting down...")
            tcp_srv.shutdown()
            server.should_exit = True
            sys.exit(0)
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        print(f"[Core] FastAPI REST server on port {args.http_port}")
        print(f"[Core]   Docs: http://localhost:{args.http_port}/docs")
        print(f"[Core] ready.")
        server.run()
    else:
        print("[Core] WARNING: FastAPI not installed, REST API disabled")
        print("[Core]   Install: pip install fastapi uvicorn pydantic")

        def _shutdown(sig, frame):
            print("\n[Core] shutting down...")
            tcp_srv.shutdown()
            sys.exit(0)
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        print(f"[Core] ready (TCP-only mode).")
        tcp_srv.serve_forever()


if __name__ == "__main__":
    main()

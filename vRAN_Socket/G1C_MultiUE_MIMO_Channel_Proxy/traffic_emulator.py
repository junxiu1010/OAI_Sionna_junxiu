#!/usr/bin/env python3
"""Traffic Emulator — DL/UL traffic generation for RAN Twin experiments.

Generates realistic traffic patterns through the 5GC data plane using iperf3.
Each UE can have an independent traffic profile with configurable patterns,
data rates, and scheduling.

Supported traffic patterns:
  - full_buffer:  Continuous maximum-rate TCP/UDP stream (saturates the link)
  - periodic:     Fixed-rate UDP at specified bitrate with constant interval
  - bursty:       ON/OFF model with Poisson-distributed burst arrivals
  - mixed:        Per-UE heterogeneous profiles (e.g., some full-buffer, some bursty)

Integration:
  - Standalone:   python3 traffic_emulator.py --config master_config.yaml
  - Core Emulator: imported and controlled via REST API (/api/v1/traffic/*)
  - CLI:          python3 traffic_emulator.py --pattern full_buffer --ues 4

Architecture:
  iperf3 server (UPF side) → GTP tunnel → gNB → Sionna proxy → UE
  Each UE runs an iperf3 client in the UE network namespace (oaitun_ue{N}).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("traffic_emulator")


class TrafficPattern(str, Enum):
    FULL_BUFFER = "full_buffer"
    PERIODIC = "periodic"
    BURSTY = "bursty"


class Direction(str, Enum):
    DL = "dl"
    UL = "ul"
    BIDIR = "bidir"


@dataclass
class TrafficProfile:
    """Traffic profile for a single UE or group of UEs."""
    pattern: TrafficPattern = TrafficPattern.FULL_BUFFER
    direction: Direction = Direction.DL
    bitrate_mbps: float = 100.0
    duration_s: float = 0       # 0 = indefinite
    protocol: str = "udp"       # tcp | udp

    # Periodic pattern
    packet_size_bytes: int = 1400
    interval_ms: float = 1.0    # inter-packet interval for periodic

    # Bursty (ON/OFF) pattern
    burst_on_ms: float = 100.0     # mean ON duration
    burst_off_ms: float = 200.0    # mean OFF duration (Poisson)
    burst_rate_mbps: float = 50.0  # rate during ON period

    # QoS tagging
    dscp: int = 0               # DSCP marking (0=best effort, 46=EF)
    tos: int = 0                # TOS byte (overrides dscp if nonzero)

    # UE mobility
    speed_kmh: float = 3.0      # UE speed in km/h (mapped to Sionna velocity)
    # Preset labels: "static"=0, "pedestrian"=3, "vehicular_urban"=30,
    #                "vehicular_highway"=120, "hst"=350 (high-speed train)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern"] = self.pattern.value
        d["direction"] = self.direction.value
        return d

    @property
    def speed_ms(self) -> float:
        """Speed in m/s (for Sionna velocity tensor)."""
        return self.speed_kmh / 3.6


@dataclass
class UETrafficState:
    """Runtime state for a single UE's traffic generation."""
    ue_idx: int
    profile: TrafficProfile
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    burst_thread: Optional[threading.Thread] = field(default=None, repr=False)
    running: bool = False
    bytes_sent: int = 0
    bytes_received: int = 0
    start_time: float = 0.0
    stop_event: threading.Event = field(default_factory=threading.Event)


class TrafficEmulator:
    """Manages traffic generation for all UEs."""

    def __init__(self, num_ues: int = 4,
                 default_profile: Optional[TrafficProfile] = None,
                 ue_profiles: Optional[Dict[int, TrafficProfile]] = None,
                 server_ip: str = "12.1.1.1",
                 iperf3_path: str = "iperf3"):
        self.num_ues = num_ues
        self.default_profile = default_profile or TrafficProfile()
        self.ue_profiles = ue_profiles or {}
        self.server_ip = server_ip
        self.iperf3_path = iperf3_path
        self._ue_states: Dict[int, UETrafficState] = {}
        self._server_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._global_stop = threading.Event()

    def get_profile(self, ue_idx: int) -> TrafficProfile:
        return self.ue_profiles.get(ue_idx, self.default_profile)

    def set_profile(self, ue_idx: int, profile: TrafficProfile) -> None:
        with self._lock:
            self.ue_profiles[ue_idx] = profile

    def set_default_profile(self, profile: TrafficProfile) -> None:
        with self._lock:
            self.default_profile = profile

    # ── iperf3 server management ──────────────────────────────────

    def start_server(self, port: int = 5201) -> bool:
        if self._server_proc and self._server_proc.poll() is None:
            log.info("iperf3 server already running (PID %d)", self._server_proc.pid)
            return True
        try:
            self._server_proc = subprocess.Popen(
                [self.iperf3_path, "-s", "-p", str(port), "-D", "--one-off"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            log.info("iperf3 server started on port %d (PID %d)", port, self._server_proc.pid)
            return True
        except FileNotFoundError:
            log.error("iperf3 not found at '%s'", self.iperf3_path)
            return False

    def stop_server(self) -> None:
        if self._server_proc:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None
            log.info("iperf3 server stopped")

    # ── Per-UE traffic control ────────────────────────────────────

    def _build_iperf_cmd(self, ue_idx: int, profile: TrafficProfile,
                          port: int = 5201) -> List[str]:
        """Build iperf3 client command for a UE."""
        cmd = [self.iperf3_path, "-c", self.server_ip, "-p", str(port)]

        if profile.protocol == "udp":
            cmd.append("-u")

        if profile.direction == Direction.DL:
            cmd.append("-R")  # reverse = server sends to client (DL)
        elif profile.direction == Direction.BIDIR:
            cmd.append("--bidir")

        if profile.duration_s > 0:
            cmd.extend(["-t", str(int(profile.duration_s))])
        else:
            cmd.extend(["-t", "0"])  # indefinite

        if profile.pattern == TrafficPattern.FULL_BUFFER:
            cmd.extend(["-b", "0"])  # unlimited for UDP, or TCP max
            if profile.protocol == "tcp":
                cmd.extend(["-P", "4"])  # parallel streams for saturation
        elif profile.pattern == TrafficPattern.PERIODIC:
            cmd.extend(["-b", f"{profile.bitrate_mbps}M"])
            cmd.extend(["-l", str(profile.packet_size_bytes)])
        elif profile.pattern == TrafficPattern.BURSTY:
            cmd.extend(["-b", f"{profile.burst_rate_mbps}M"])

        if profile.tos > 0:
            cmd.extend(["--tos", str(profile.tos)])
        elif profile.dscp > 0:
            cmd.extend(["--tos", str(profile.dscp << 2)])

        cmd.extend(["--json", "--logfile", f"/tmp/iperf3_ue{ue_idx}.json"])

        return cmd

    def _get_ue_netns(self, ue_idx: int) -> Optional[str]:
        """Find the network namespace for a UE (oaitun_ueN interface)."""
        iface = f"oaitun_ue{ue_idx + 1}"
        result = subprocess.run(
            ["ip", "link", "show", iface],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return None  # interface in default namespace
        return f"ue{ue_idx}"  # try named netns

    def start_ue_traffic(self, ue_idx: int, port: int = 5201) -> bool:
        """Start traffic for a single UE."""
        with self._lock:
            if ue_idx in self._ue_states and self._ue_states[ue_idx].running:
                log.warning("UE %d traffic already running", ue_idx)
                return True

        profile = self.get_profile(ue_idx)

        if profile.pattern == TrafficPattern.BURSTY:
            return self._start_bursty(ue_idx, profile, port)

        cmd = self._build_iperf_cmd(ue_idx, profile, port)
        log.info("UE %d: starting %s traffic (%s, %s)",
                 ue_idx, profile.pattern.value, profile.direction.value,
                 f"{profile.bitrate_mbps}Mbps" if profile.pattern != TrafficPattern.FULL_BUFFER else "max-rate")

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except Exception as e:
            log.error("UE %d: failed to start iperf3: %s", ue_idx, e)
            return False

        state = UETrafficState(
            ue_idx=ue_idx, profile=profile,
            process=proc, running=True, start_time=time.time(),
        )
        with self._lock:
            self._ue_states[ue_idx] = state

        log.info("UE %d: iperf3 PID %d", ue_idx, proc.pid)
        return True

    def _start_bursty(self, ue_idx: int, profile: TrafficProfile,
                       port: int = 5201) -> bool:
        """Bursty ON/OFF traffic using Poisson-distributed OFF intervals."""
        state = UETrafficState(
            ue_idx=ue_idx, profile=profile,
            running=True, start_time=time.time(),
        )
        state.stop_event.clear()

        def burst_loop():
            while not state.stop_event.is_set() and not self._global_stop.is_set():
                # ON period
                on_dur = max(0.01, random.expovariate(1.0 / (profile.burst_on_ms / 1000)))
                on_dur = min(on_dur, 10.0)  # cap at 10s

                burst_cmd = [
                    self.iperf3_path, "-c", self.server_ip, "-p", str(port),
                    "-u" if profile.protocol == "udp" else "",
                    "-b", f"{profile.burst_rate_mbps}M",
                    "-t", str(max(1, int(on_dur))),
                ]
                if profile.direction == Direction.DL:
                    burst_cmd.append("-R")
                burst_cmd = [c for c in burst_cmd if c]

                log.debug("UE %d: burst ON (%.1fs, %.1f Mbps)",
                          ue_idx, on_dur, profile.burst_rate_mbps)
                try:
                    proc = subprocess.Popen(
                        burst_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    state.process = proc
                    proc.wait(timeout=on_dur + 5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except Exception as e:
                    log.error("UE %d: burst error: %s", ue_idx, e)

                if state.stop_event.is_set():
                    break

                # OFF period (Poisson inter-arrival)
                off_dur = random.expovariate(1.0 / (profile.burst_off_ms / 1000))
                off_dur = min(off_dur, 30.0)
                log.debug("UE %d: burst OFF (%.1fs)", ue_idx, off_dur)
                state.stop_event.wait(timeout=off_dur)

            state.running = False
            log.info("UE %d: bursty traffic stopped", ue_idx)

        t = threading.Thread(target=burst_loop, daemon=True, name=f"burst_ue{ue_idx}")
        state.burst_thread = t
        with self._lock:
            self._ue_states[ue_idx] = state
        t.start()
        log.info("UE %d: bursty traffic started (ON=%.0fms, OFF=%.0fms, rate=%.1fMbps)",
                 ue_idx, profile.burst_on_ms, profile.burst_off_ms, profile.burst_rate_mbps)
        return True

    def stop_ue_traffic(self, ue_idx: int) -> bool:
        with self._lock:
            state = self._ue_states.get(ue_idx)
            if not state or not state.running:
                return False

        state.stop_event.set()
        if state.process and state.process.poll() is None:
            state.process.terminate()
            try:
                state.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                state.process.kill()
        if state.burst_thread and state.burst_thread.is_alive():
            state.burst_thread.join(timeout=10)

        state.running = False
        log.info("UE %d: traffic stopped", ue_idx)
        return True

    # ── Bulk operations ───────────────────────────────────────────

    def start_all(self, port: int = 5201) -> Dict[int, bool]:
        """Start traffic for all configured UEs."""
        results = {}
        base_port = port
        for ue_idx in range(self.num_ues):
            p = base_port + ue_idx
            ok = self.start_ue_traffic(ue_idx, port=p)
            results[ue_idx] = ok
        return results

    def stop_all(self) -> None:
        self._global_stop.set()
        for ue_idx in list(self._ue_states.keys()):
            self.stop_ue_traffic(ue_idx)
        self.stop_server()
        log.info("All traffic stopped")

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            ue_status = {}
            for idx, st in self._ue_states.items():
                elapsed = time.time() - st.start_time if st.running else 0
                ue_status[idx] = {
                    "running": st.running,
                    "pattern": st.profile.pattern.value,
                    "direction": st.profile.direction.value,
                    "speed_kmh": st.profile.speed_kmh,
                    "elapsed_s": round(elapsed, 1),
                    "pid": st.process.pid if st.process and st.process.poll() is None else None,
                }
            ue_speeds = {i: self.get_profile(i).speed_kmh for i in range(self.num_ues)}
            return {
                "num_ues": self.num_ues,
                "active_ues": sum(1 for s in self._ue_states.values() if s.running),
                "server_running": self._server_proc is not None and self._server_proc.poll() is None,
                "default_profile": self.default_profile.to_dict(),
                "ue_speeds_kmh": ue_speeds,
                "ue_status": ue_status,
            }

    def get_ue_speeds(self) -> Dict[int, float]:
        """Return per-UE speeds in m/s for the Proxy channel producer."""
        speeds = {}
        for ue_idx in range(self.num_ues):
            p = self.get_profile(ue_idx)
            speeds[ue_idx] = p.speed_ms
        return speeds

    def get_results(self, ue_idx: int) -> Optional[dict]:
        """Read iperf3 JSON results for a UE."""
        path = f"/tmp/iperf3_ue{ue_idx}.json"
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None


# ═══════════════════════════════════════════════════════════════════
# Configuration helpers
# ═══════════════════════════════════════════════════════════════════

def profile_from_dict(d: dict) -> TrafficProfile:
    """Create TrafficProfile from a config dictionary."""
    p = TrafficProfile()
    if "pattern" in d:
        p.pattern = TrafficPattern(d["pattern"])
    if "direction" in d:
        p.direction = Direction(d["direction"])
    for attr in ["bitrate_mbps", "duration_s", "protocol", "packet_size_bytes",
                 "interval_ms", "burst_on_ms", "burst_off_ms", "burst_rate_mbps",
                 "dscp", "tos", "speed_kmh"]:
        if attr in d:
            setattr(p, attr, type(getattr(p, attr))(d[attr]))
    return p


def load_from_config(cfg: dict) -> TrafficEmulator:
    """Create TrafficEmulator from master_config.yaml traffic section."""
    traffic = cfg.get("traffic", {})
    num_ues = cfg.get("system", {}).get("num_ues", 4)

    default_d = traffic.get("default_profile", {})
    default_profile = profile_from_dict(default_d) if default_d else TrafficProfile()

    ue_profiles = {}
    for ue_cfg in traffic.get("ue_profiles", []):
        ue_idx = ue_cfg.get("ue_idx")
        if ue_idx is not None:
            ue_profiles[ue_idx] = profile_from_dict(ue_cfg)

    server_ip = traffic.get("server_ip", "12.1.1.1")

    return TrafficEmulator(
        num_ues=num_ues,
        default_profile=default_profile,
        ue_profiles=ue_profiles,
        server_ip=server_ip,
    )


# ═══════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Traffic Emulator — DL/UL traffic generation for RAN Twin",
    )
    ap.add_argument("--config", "-c", type=Path,
                    help="master_config.yaml path (uses traffic section)")
    ap.add_argument("--pattern", choices=["full_buffer", "periodic", "bursty"],
                    default="full_buffer", help="Traffic pattern (default: full_buffer)")
    ap.add_argument("--direction", choices=["dl", "ul", "bidir"],
                    default="dl", help="Traffic direction (default: dl)")
    ap.add_argument("--bitrate", type=float, default=100.0,
                    help="Target bitrate in Mbps (default: 100)")
    ap.add_argument("--ues", type=int, default=4,
                    help="Number of UEs (default: 4)")
    ap.add_argument("--duration", type=float, default=0,
                    help="Duration in seconds (0=indefinite, default: 0)")
    ap.add_argument("--server-ip", default="12.1.1.1",
                    help="iperf3 server IP (default: 12.1.1.1)")
    ap.add_argument("--protocol", choices=["tcp", "udp"], default="udp",
                    help="Transport protocol (default: udp)")
    ap.add_argument("--burst-on-ms", type=float, default=100,
                    help="Bursty: mean ON duration in ms (default: 100)")
    ap.add_argument("--burst-off-ms", type=float, default=200,
                    help="Bursty: mean OFF duration in ms (default: 200)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        te = load_from_config(cfg)
    else:
        profile = TrafficProfile(
            pattern=TrafficPattern(args.pattern),
            direction=Direction(args.direction),
            bitrate_mbps=args.bitrate,
            duration_s=args.duration,
            protocol=args.protocol,
            burst_on_ms=args.burst_on_ms,
            burst_off_ms=args.burst_off_ms,
            burst_rate_mbps=args.bitrate,
        )
        te = TrafficEmulator(
            num_ues=args.ues,
            default_profile=profile,
            server_ip=args.server_ip,
        )

    if args.dry_run:
        print("Traffic Emulator — Dry Run")
        print(f"  UEs: {te.num_ues}")
        print(f"  Server: {te.server_ip}")
        print(f"  Default profile: {json.dumps(te.default_profile.to_dict(), indent=2)}")
        for ue_idx in range(te.num_ues):
            p = te.get_profile(ue_idx)
            cmd = te._build_iperf_cmd(ue_idx, p)
            print(f"  UE {ue_idx}: {' '.join(cmd)}")
        return

    print("Traffic Emulator starting...")
    print(f"  UEs: {te.num_ues}")
    print(f"  Pattern: {te.default_profile.pattern.value}")
    print(f"  Direction: {te.default_profile.direction.value}")
    print(f"  Server: {te.server_ip}")

    def shutdown(sig, frame):
        print("\nShutting down traffic emulator...")
        te.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    results = te.start_all()
    for ue_idx, ok in results.items():
        status = "started" if ok else "FAILED"
        print(f"  UE {ue_idx}: {status}")

    print("\nTraffic running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            status = te.get_status()
            active = status["active_ues"]
            print(f"  [{time.strftime('%H:%M:%S')}] Active: {active}/{te.num_ues} UEs")
    except KeyboardInterrupt:
        pass
    finally:
        te.stop_all()


if __name__ == "__main__":
    main()

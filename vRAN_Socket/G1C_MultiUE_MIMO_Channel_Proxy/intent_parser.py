"""intent_parser.py - Rule-based intent parser for Core Emulator.

Parses natural-language-style intent strings into preset selection and
parameter overrides.  Designed to be replaced/augmented with LLM in Phase 2.

Usage:
    result = parse_intent("4T4R MU-MIMO type2, CSI-RS 주기 20, 4 UE")
    # => IntentResult(preset="mu_mimo_4t4r_type2", overrides={...}, ...)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IntentResult:
    matched_preset: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "matched_preset": self.matched_preset,
            "overrides": self.overrides,
            "tags": self.tags,
            "confidence": round(self.confidence, 2),
            "explanation": self.explanation,
        }


# ── Keyword rules ──────────────────────────────────────────────────

_ANTENNA_PATTERNS = [
    (re.compile(r'\b(\d+)\s*[Tt]\s*(\d+)\s*[Rr]\b'), "antenna_txrx"),
    (re.compile(r'\b(\d+)\s*[xX]\s*(\d+)\b'), "antenna_grid"),
]

_MIMO_KEYWORDS = {
    "mu-mimo":  {"system.mu_mimo": True, "codebook.type": "type2"},
    "mu_mimo":  {"system.mu_mimo": True, "codebook.type": "type2"},
    "mumimo":   {"system.mu_mimo": True, "codebook.type": "type2"},
    "su-mimo":  {"system.mu_mimo": False},
    "su_mimo":  {"system.mu_mimo": False},
    "sumimo":   {"system.mu_mimo": False},
}

_CODEBOOK_KEYWORDS = {
    "type1":    {"codebook.type": "type1"},
    "type-1":   {"codebook.type": "type1"},
    "type 1":   {"codebook.type": "type1"},
    "type2":    {"codebook.type": "type2"},
    "type-2":   {"codebook.type": "type2"},
    "type 2":   {"codebook.type": "type2"},
    "typeii_portselection": {"codebook.sub_type": "typeII_PortSelection"},
    "port selection":       {"codebook.sub_type": "typeII_PortSelection"},
    "typei_singlepanel":    {"codebook.sub_type": "typeI_SinglePanel"},
    "single panel":         {"codebook.sub_type": "typeI_SinglePanel"},
}

_POLARIZATION_KEYWORDS = {
    "dual-pol":    {"antenna.gnb.polarization": "dual"},
    "dual pol":    {"antenna.gnb.polarization": "dual"},
    "dual_pol":    {"antenna.gnb.polarization": "dual"},
    "cross-pol":   {"antenna.gnb.polarization": "dual"},
    "single-pol":  {"antenna.gnb.polarization": "single"},
    "single pol":  {"antenna.gnb.polarization": "single"},
}

_SCENARIO_KEYWORDS = {
    "uma-nlos": {"channel.scenario": "UMa-NLOS"},
    "uma-los":  {"channel.scenario": "UMa-LOS"},
    "umi-nlos": {"channel.scenario": "UMi-NLOS"},
    "umi-los":  {"channel.scenario": "UMi-LOS"},
}

_NUMERIC_PATTERNS = [
    (re.compile(r'csi[-_\s]*rs\s*(?:주기|period(?:icity)?)\s*[=:]?\s*(\d+)', re.I),
     "csi_rs.periodicity", int),
    (re.compile(r'srs\s*(?:주기|period(?:icity)?)\s*[=:]?\s*(\d+)', re.I),
     "srs.periodicity", int),
    (re.compile(r'(?:ue|유이)\s*(?:수|개수|count|num)?\s*[=:]?\s*(\d+)', re.I),
     "system.num_ues", int),
    (re.compile(r'(\d+)\s*(?:ue|유이)', re.I),
     "system.num_ues", int),
    (re.compile(r'speed\s*[=:]?\s*(\d+(?:\.\d+)?)', re.I),
     "channel.speed", float),
    (re.compile(r'snr\s*[=:]?\s*(-?\d+(?:\.\d+)?)', re.I),
     "channel.snr_dB", float),
    (re.compile(r'path[-_\s]*loss\s*[=:]?\s*(-?\d+(?:\.\d+)?)', re.I),
     "channel.path_loss_dB", float),
    (re.compile(r'(\d+)\s*(?:셀|cell)', re.I),
     "multicell.num_cells", int),
    (re.compile(r'(?:셀|cell)\s*(\d+)', re.I),
     "multicell.num_cells", int),
    (re.compile(r'report\s*(?:주기|period)\s*[=:]?\s*(\d+)', re.I),
     "csi_measurement.report_periodicity", int),
]


def _infer_antenna_config(n_tx: int) -> Dict[str, Any]:
    """Infer gnb nx/ny/polarization from total TX antenna count."""
    overrides: Dict[str, Any] = {}
    if n_tx == 2:
        overrides["antenna.gnb.nx"] = 2
        overrides["antenna.gnb.ny"] = 1
        overrides["antenna.gnb.polarization"] = "single"
    elif n_tx == 4:
        overrides["antenna.gnb.nx"] = 2
        overrides["antenna.gnb.ny"] = 1
        overrides["antenna.gnb.polarization"] = "dual"
    elif n_tx == 8:
        overrides["antenna.gnb.nx"] = 4
        overrides["antenna.gnb.ny"] = 1
        overrides["antenna.gnb.polarization"] = "dual"
    elif n_tx == 16:
        overrides["antenna.gnb.nx"] = 4
        overrides["antenna.gnb.ny"] = 2
        overrides["antenna.gnb.polarization"] = "dual"
    elif n_tx == 32:
        overrides["antenna.gnb.nx"] = 8
        overrides["antenna.gnb.ny"] = 2
        overrides["antenna.gnb.polarization"] = "dual"
    return overrides


def _select_preset(tags: List[str], overrides: Dict[str, Any]) -> Optional[str]:
    """Select best-matching preset based on extracted tags and overrides."""
    is_mu = overrides.get("system.mu_mimo", False)
    cb_type = overrides.get("codebook.type", "type1")
    pol = overrides.get("antenna.gnb.polarization", "dual")
    nx = overrides.get("antenna.gnb.nx", 2)
    ny = overrides.get("antenna.gnb.ny", 1)
    pol_mult = 2 if pol == "dual" else 1
    n_ant = nx * ny * pol_mult
    n_cells = overrides.get("multicell.num_cells", 1)

    if n_cells >= 2:
        n_ues = overrides.get("system.num_ues", 4)
        return f"multicell_{n_cells}cell_{n_ues}ue"

    if is_mu and cb_type == "type2":
        if n_ant >= 8:
            return "mu_mimo_8t8r_type2"
        return "mu_mimo_4t4r_type2"

    if cb_type == "type1" or not is_mu:
        if n_ant <= 2:
            return "su_mimo_2t2r_type1"
        return "su_mimo_4t4r_type1"

    return None


def parse_intent(text: str) -> IntentResult:
    """Parse a natural-language intent string into preset + overrides."""
    result = IntentResult()
    lower = text.lower().strip()
    tags: List[str] = []
    overrides: Dict[str, Any] = {}
    matched_rules = 0

    for pattern, ptype in _ANTENNA_PATTERNS:
        m = pattern.search(lower)
        if m:
            if ptype == "antenna_txrx":
                n_tx = int(m.group(1))
                ant_overrides = _infer_antenna_config(n_tx)
                overrides.update(ant_overrides)
                tags.append(f"{n_tx}T{m.group(2)}R")
                matched_rules += 1
            elif ptype == "antenna_grid":
                overrides["antenna.gnb.nx"] = int(m.group(1))
                overrides["antenna.gnb.ny"] = int(m.group(2))
                tags.append(f"{m.group(1)}x{m.group(2)}")
                matched_rules += 1

    for kw, kw_overrides in _MIMO_KEYWORDS.items():
        if kw in lower.replace(" ", "").replace("-", "").replace("_", ""):
            overrides.update(kw_overrides)
            tags.append(kw.replace("_", "-").upper())
            matched_rules += 1
            break

    for kw, kw_overrides in _CODEBOOK_KEYWORDS.items():
        if kw in lower:
            overrides.update(kw_overrides)
            tags.append(kw)
            matched_rules += 1

    for kw, kw_overrides in _POLARIZATION_KEYWORDS.items():
        if kw in lower:
            overrides.update(kw_overrides)
            tags.append(kw)
            matched_rules += 1

    for kw, kw_overrides in _SCENARIO_KEYWORDS.items():
        if kw in lower:
            overrides.update(kw_overrides)
            tags.append(kw)
            matched_rules += 1

    for pattern, key, cast_fn in _NUMERIC_PATTERNS:
        m = pattern.search(text)
        if m:
            overrides[key] = cast_fn(m.group(1))
            matched_rules += 1

    if overrides.get("multicell.num_cells", 1) >= 2:
        overrides["multicell.enabled"] = True

    preset = _select_preset(tags, overrides)

    total_possible = 6
    confidence = min(1.0, matched_rules / total_possible)

    parts = []
    if preset:
        parts.append(f"preset={preset}")
    if overrides:
        parts.append(f"{len(overrides)} param overrides")
    if not parts:
        parts.append("no matching rules found")

    result.matched_preset = preset
    result.overrides = overrides
    result.tags = tags
    result.confidence = confidence
    result.explanation = "; ".join(parts)

    return result

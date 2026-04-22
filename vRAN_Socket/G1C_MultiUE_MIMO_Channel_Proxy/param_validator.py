"""param_validator.py - 3GPP spec-based parameter validation for Core Emulator.

Validates configuration parameters against 3GPP TS 38.331 / 38.214 constraints
to catch invalid combinations before they reach OAI nr-softmodem.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

VALID_CSI_RS_PERIODICITIES = {4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640}
VALID_SRS_PERIODICITIES = {1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, 2560}
VALID_NZP_CSI_RS_PORTS = {1, 2, 4, 8, 12, 16, 24, 32}
VALID_SRS_PORTS = {1, 2, 4}
VALID_CODEBOOK_TYPES = {"type1", "type2"}
VALID_CODEBOOK_SUB_TYPES = {
    "type1": {"typeI_SinglePanel", "typeI_MultiPanel"},
    "type2": {"typeII", "typeII_PortSelection"},
}
VALID_POLARIZATIONS = {"single", "dual"}
VALID_CDM_TYPES = {"noCDM", "fd_CDM2", "cdm4_FD2_TD2", "cdm8_FD2_TD4"}
VALID_SCS_KHZ = {15, 30, 60, 120, 240}
VALID_BANDS_FR1 = set(range(1, 100))
VALID_N1_N2_CONFIGS = {
    "one_one", "two_one", "two_two", "four_one", "three_two",
    "six_one", "four_two", "eight_one", "four_three", "six_two",
    "twelve_one", "four_four", "eight_two", "sixteen_one",
}
VALID_CSI_REPORT_PERIODICITIES = {4, 5, 8, 10, 16, 20, 40, 80, 160, 320}
VALID_PHASE_ALPHABET = {"n4", "n8"}
VALID_GROUP_HOPPING = {"neither", "groupHopping", "sequenceHopping"}
VALID_ALPHA = {"alpha0", "alpha04", "alpha05", "alpha06", "alpha07",
               "alpha08", "alpha09", "alpha1"}
VALID_5QI_NON_GBR = {5, 6, 7, 8, 9, 69, 70, 79, 80}
VALID_5QI_GBR = {1, 2, 3, 4, 65, 66, 67, 71, 72, 73, 74, 75, 76}
VALID_5QI = VALID_5QI_NON_GBR | VALID_5QI_GBR
VALID_ARP_PREEMPT_CAP = {"NOT_PREEMPT", "MAY_PREEMPT"}
VALID_ARP_PREEMPT_VULN = {"PREEMPTABLE", "NOT_PREEMPTABLE"}


class ValidationError:
    """Single validation issue."""
    def __init__(self, path: str, message: str, severity: str = "error"):
        self.path = path
        self.message = message
        self.severity = severity

    def to_dict(self) -> dict:
        return {"path": self.path, "message": self.message, "severity": self.severity}

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.path}: {self.message}"


def _get_nested(cfg: dict, dotted: str, default=None):
    parts = dotted.split(".")
    obj = cfg
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return default
    return obj


def validate_config(cfg: dict) -> List[ValidationError]:
    """Run all validation rules against the config dict. Returns list of issues."""
    errors: List[ValidationError] = []

    _validate_antenna(cfg, errors)
    _validate_codebook(cfg, errors)
    _validate_csi_rs(cfg, errors)
    _validate_srs(cfg, errors)
    _validate_csi_measurement(cfg, errors)
    _validate_carrier(cfg, errors)
    _validate_channel(cfg, errors)
    _validate_system(cfg, errors)
    _validate_bearer(cfg, errors)
    _validate_qos(cfg, errors)
    _validate_cross_section(cfg, errors)

    return errors


def _validate_antenna(cfg: dict, errs: List[ValidationError]):
    ant = cfg.get("antenna", {})
    gnb = ant.get("gnb", {})
    ue = ant.get("ue", {})

    for who, section in [("gnb", gnb), ("ue", ue)]:
        nx = section.get("nx", 1)
        ny = section.get("ny", 1)
        if nx < 1 or ny < 1:
            errs.append(ValidationError(
                f"antenna.{who}.nx/ny",
                f"nx={nx}, ny={ny} must be >= 1"))

    pol = gnb.get("polarization", "single")
    if pol not in VALID_POLARIZATIONS:
        errs.append(ValidationError(
            "antenna.gnb.polarization",
            f"'{pol}' not in {VALID_POLARIZATIONS}"))


def _validate_codebook(cfg: dict, errs: List[ValidationError]):
    cb = cfg.get("codebook", {})
    cb_type = cb.get("type", "type1")
    if cb_type not in VALID_CODEBOOK_TYPES:
        errs.append(ValidationError(
            "codebook.type", f"'{cb_type}' not in {VALID_CODEBOOK_TYPES}"))
        return

    sub = cb.get("sub_type", "")
    valid_subs = VALID_CODEBOOK_SUB_TYPES.get(cb_type, set())
    if sub and sub not in valid_subs:
        errs.append(ValidationError(
            "codebook.sub_type",
            f"'{sub}' not valid for {cb_type}; expected one of {valid_subs}"))

    n1n2 = cb.get("n1_n2_config")
    if n1n2 and n1n2 not in VALID_N1_N2_CONFIGS:
        errs.append(ValidationError(
            "codebook.n1_n2_config",
            f"'{n1n2}' not in {VALID_N1_N2_CONFIGS}"))

    if cb_type == "type2":
        phase = cb.get("phase_alphabet_size", "n4")
        if phase not in VALID_PHASE_ALPHABET:
            errs.append(ValidationError(
                "codebook.phase_alphabet_size",
                f"'{phase}' not in {VALID_PHASE_ALPHABET}"))


def _validate_csi_rs(cfg: dict, errs: List[ValidationError]):
    csi = cfg.get("csi_rs", {})

    per = csi.get("periodicity")
    if per is not None and per not in VALID_CSI_RS_PERIODICITIES:
        errs.append(ValidationError(
            "csi_rs.periodicity",
            f"{per} not in {sorted(VALID_CSI_RS_PERIODICITIES)}"))

    ports = csi.get("nrof_ports")
    if ports is not None and ports not in VALID_NZP_CSI_RS_PORTS:
        errs.append(ValidationError(
            "csi_rs.nrof_ports",
            f"{ports} not in {sorted(VALID_NZP_CSI_RS_PORTS)}"))

    cdm = csi.get("cdm_type")
    if cdm and cdm not in VALID_CDM_TYPES:
        errs.append(ValidationError(
            "csi_rs.cdm_type",
            f"'{cdm}' not in {VALID_CDM_TYPES}"))


def _validate_srs(cfg: dict, errs: List[ValidationError]):
    srs = cfg.get("srs", {})

    per = srs.get("periodicity")
    if per is not None and per not in VALID_SRS_PERIODICITIES:
        errs.append(ValidationError(
            "srs.periodicity",
            f"{per} not in {sorted(VALID_SRS_PERIODICITIES)}"))

    ports = srs.get("nrof_srs_ports")
    if ports is not None and ports not in VALID_SRS_PORTS:
        errs.append(ValidationError(
            "srs.nrof_srs_ports",
            f"{ports} not in {sorted(VALID_SRS_PORTS)}"))

    gh = srs.get("group_hopping")
    if gh and gh not in VALID_GROUP_HOPPING:
        errs.append(ValidationError(
            "srs.group_hopping",
            f"'{gh}' not in {VALID_GROUP_HOPPING}"))

    alpha = srs.get("alpha")
    if alpha and alpha not in VALID_ALPHA:
        errs.append(ValidationError(
            "srs.alpha",
            f"'{alpha}' not in {VALID_ALPHA}"))


def _validate_csi_measurement(cfg: dict, errs: List[ValidationError]):
    meas = cfg.get("csi_measurement", {})
    per = meas.get("report_periodicity")
    if per is not None and per not in VALID_CSI_REPORT_PERIODICITIES:
        errs.append(ValidationError(
            "csi_measurement.report_periodicity",
            f"{per} not in {sorted(VALID_CSI_REPORT_PERIODICITIES)}"))


def _validate_carrier(cfg: dict, errs: List[ValidationError]):
    carr = cfg.get("carrier", {})
    scs = carr.get("scs_kHz")
    if scs is not None and scs not in VALID_SCS_KHZ:
        errs.append(ValidationError(
            "carrier.scs_kHz",
            f"{scs} not in {sorted(VALID_SCS_KHZ)}"))

    bw = carr.get("bandwidth_prb")
    if bw is not None and bw < 1:
        errs.append(ValidationError(
            "carrier.bandwidth_prb", f"{bw} must be >= 1"))


def _validate_channel(cfg: dict, errs: List[ValidationError]):
    ch = cfg.get("channel", {})
    speed = ch.get("speed")
    if speed is not None and speed < 0:
        errs.append(ValidationError(
            "channel.speed", f"{speed} must be >= 0"))

    scenario = ch.get("scenario", "")
    valid_scenarios = {"UMi-LOS", "UMi-NLOS", "UMa-LOS", "UMa-NLOS"}
    if scenario and scenario not in valid_scenarios:
        errs.append(ValidationError(
            "channel.scenario",
            f"'{scenario}' not in {valid_scenarios}", severity="warning"))

    bs_h = ch.get("bs_height_m")
    if bs_h is not None:
        if bs_h <= 0 or bs_h > 200:
            errs.append(ValidationError(
                "channel.bs_height_m", f"{bs_h}m out of range (0,200]"))
        if scenario and "UMi" in scenario and bs_h != 10:
            errs.append(ValidationError(
                "channel.bs_height_m",
                f"UMi scenario: BS height should be 10m (got {bs_h}m)",
                severity="warning"))
        if scenario and "UMa" in scenario and bs_h != 25:
            errs.append(ValidationError(
                "channel.bs_height_m",
                f"UMa scenario: BS height should be 25m (got {bs_h}m)",
                severity="warning"))

    ue_h = ch.get("ue_height_m")
    if ue_h is not None and (ue_h < 0.5 or ue_h > 25):
        errs.append(ValidationError(
            "channel.ue_height_m", f"{ue_h}m out of range [0.5,25]"))

    min_d = ch.get("min_ue_distance_m")
    max_d = ch.get("max_ue_distance_m")
    if min_d is not None and min_d < 0:
        errs.append(ValidationError(
            "channel.min_ue_distance_m", f"{min_d}m must be >= 0"))
    if max_d is not None and min_d is not None and max_d < min_d:
        errs.append(ValidationError(
            "channel.max_ue_distance_m",
            f"max({max_d}) < min({min_d})"))

    sf_std = ch.get("shadow_fading_std_dB")
    if sf_std is not None and sf_std < 0:
        errs.append(ValidationError(
            "channel.shadow_fading_std_dB", f"{sf_std} must be >= 0"))

    kf_std = ch.get("k_factor_std_dB")
    if kf_std is not None and kf_std < 0:
        errs.append(ValidationError(
            "channel.k_factor_std_dB", f"{kf_std} must be >= 0"))


def _validate_system(cfg: dict, errs: List[ValidationError]):
    sys_cfg = cfg.get("system", {})
    n_ues = sys_cfg.get("num_ues")
    if n_ues is not None and n_ues < 1:
        errs.append(ValidationError(
            "system.num_ues", f"{n_ues} must be >= 1"))


def _validate_bearer(cfg: dict, errs: List[ValidationError]):
    br = cfg.get("bearer", {})
    if not br:
        return

    drbs = br.get("drbs")
    if drbs is not None:
        if not isinstance(drbs, int) or drbs < 1 or drbs > 32:
            errs.append(ValidationError(
                "bearer.drbs",
                f"drbs={drbs} must be integer 1-32 (3GPP TS 38.413 maxnoofDRBs)"))


def _validate_qos(cfg: dict, errs: List[ValidationError]):
    qos = cfg.get("qos", {})
    if not qos:
        return

    fiveqi = qos.get("default_5qi")
    if fiveqi is not None:
        if fiveqi not in VALID_5QI:
            errs.append(ValidationError(
                "qos.default_5qi",
                f"5QI={fiveqi} not a standard 5QI value; "
                f"GBR: {sorted(VALID_5QI_GBR)}, Non-GBR: {sorted(VALID_5QI_NON_GBR)}",
                severity="warning"))

    arp = qos.get("arp_priority")
    if arp is not None and (not isinstance(arp, int) or arp < 1 or arp > 15):
        errs.append(ValidationError(
            "qos.arp_priority",
            f"ARP priority={arp} must be 1-15 (3GPP TS 23.501 §5.7.2.2)"))

    cap = qos.get("arp_preempt_cap")
    if cap and cap not in VALID_ARP_PREEMPT_CAP:
        errs.append(ValidationError(
            "qos.arp_preempt_cap",
            f"'{cap}' not in {VALID_ARP_PREEMPT_CAP}"))

    vuln = qos.get("arp_preempt_vuln")
    if vuln and vuln not in VALID_ARP_PREEMPT_VULN:
        errs.append(ValidationError(
            "qos.arp_preempt_vuln",
            f"'{vuln}' not in {VALID_ARP_PREEMPT_VULN}"))


def _validate_cross_section(cfg: dict, errs: List[ValidationError]):
    """Cross-section consistency checks."""
    ant = cfg.get("antenna", {})
    gnb = ant.get("gnb", {})
    pol = gnb.get("polarization", "single")
    pol_mult = 2 if pol == "dual" else 1
    gnb_ant = gnb.get("nx", 1) * gnb.get("ny", 1) * pol_mult

    cb = cfg.get("codebook", {})
    if cb.get("type") == "type2" and gnb_ant < 4:
        errs.append(ValidationError(
            "codebook.type",
            f"Type-II codebook requires gnb_ant >= 4 (current: {gnb_ant})",
            severity="error"))

    csi_ports = cfg.get("csi_rs", {}).get("nrof_ports")
    if csi_ports and csi_ports > gnb_ant:
        errs.append(ValidationError(
            "csi_rs.nrof_ports",
            f"CSI-RS ports ({csi_ports}) > gNB antennas ({gnb_ant})",
            severity="warning"))

    sys_cfg = cfg.get("system", {})
    if sys_cfg.get("mu_mimo") and cb.get("type") != "type2":
        errs.append(ValidationError(
            "system.mu_mimo",
            "MU-MIMO generally requires Type-II codebook",
            severity="warning"))

    mc = cfg.get("multicell", {})
    if mc.get("enabled") and mc.get("num_cells", 1) < 2:
        errs.append(ValidationError(
            "multicell.num_cells",
            "multicell.enabled=true but num_cells < 2",
            severity="warning"))

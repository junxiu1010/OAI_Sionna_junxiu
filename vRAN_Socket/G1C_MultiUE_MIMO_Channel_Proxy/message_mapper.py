"""message_mapper.py - Maps config parameters to 3GPP RRC/NGAP message IEs.

Provides metadata showing which OAI config parameter ends up in which
3GPP ASN.1 IE, which RRC message carries it, and what OAI source file
generates it.  Used by the Core Emulator API to explain the impact of
configuration changes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


PARAM_MAP: Dict[str, Dict[str, Any]] = {
    # ── CSI-RS ────────────────────────────────────────────────────
    "csi_rs.periodicity": {
        "3gpp_ie": "NZP-CSI-RS-Resource.periodicityAndOffset",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.331 §6.3.2 (CSI-MeasConfig)",
        "oai_source": "nr_radio_config.c::set_csirs_periodicity()",
        "oai_config_key": "csirs_detailed_config.periodicity",
        "requires_restart": True,
        "description": "NZP CSI-RS resource transmission periodicity in slots",
    },
    "csi_rs.nrof_ports": {
        "3gpp_ie": "NZP-CSI-RS-Resource.nrofPorts",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.331 §6.3.2",
        "oai_source": "nr_radio_config.c::config_csirs()",
        "oai_config_key": "csirs_detailed_config.nrof_ports",
        "requires_restart": True,
        "description": "Number of antenna ports for NZP CSI-RS resource",
    },
    "csi_rs.cdm_type": {
        "3gpp_ie": "NZP-CSI-RS-Resource.cdm-Type",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.331 §6.3.2",
        "oai_source": "nr_radio_config.c::config_csirs()",
        "oai_config_key": "csirs_detailed_config.cdm_type",
        "requires_restart": True,
        "description": "CDM type for CSI-RS (noCDM, fd_CDM2, cdm4, cdm8)",
    },
    "csi_rs.density": {
        "3gpp_ie": "NZP-CSI-RS-Resource.density",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.331 §6.3.2",
        "oai_source": "nr_radio_config.c::config_csirs()",
        "oai_config_key": "csirs_detailed_config.density",
        "requires_restart": True,
        "description": "CSI-RS density (0.5, 1, or 3 RE per RB per port)",
    },

    # ── SRS ───────────────────────────────────────────────────────
    "srs.periodicity": {
        "3gpp_ie": "SRS-PeriodicityAndOffset",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → BWP-UplinkDedicated → SRS-Config",
        "3gpp_spec": "TS 38.331 §6.3.2 (SRS-Config)",
        "oai_source": "nr_radio_config.c::configure_periodic_srs()",
        "oai_config_key": "srs_detailed_config.periodicity",
        "requires_restart": True,
        "description": "SRS transmission periodicity in slots",
    },
    "srs.nrof_srs_ports": {
        "3gpp_ie": "SRS-Resource.nrofSRS-Ports",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → SRS-Config",
        "3gpp_spec": "TS 38.331 §6.3.2",
        "oai_source": "nr_radio_config.c::get_srs_resource()",
        "oai_config_key": "srs_detailed_config.nrof_srs_ports",
        "requires_restart": True,
        "description": "Number of SRS antenna ports (1, 2, or 4)",
    },

    # ── Codebook ──────────────────────────────────────────────────
    "codebook.type": {
        "3gpp_ie": "CodebookConfig.codebookType",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig → CSI-ReportConfig",
        "3gpp_spec": "TS 38.331 §6.3.2 (CodebookConfig), TS 38.214 §5.2.2.3",
        "oai_source": "nr_radio_config.c::config_csi_codebook()",
        "oai_config_key": "codebook_detailed_config.codebook_type",
        "requires_restart": True,
        "description": "PMI codebook type (type1: single/multi panel, type2: enhanced)",
    },
    "codebook.sub_type": {
        "3gpp_ie": "CodebookConfig.codebookType.typeX.subType",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.214 §5.2.2.3",
        "oai_source": "nr_radio_config.c::config_csi_codebook()",
        "oai_config_key": "codebook_detailed_config.sub_type",
        "requires_restart": True,
        "description": "Codebook sub-type (e.g., typeII_PortSelection)",
    },
    "codebook.ri_restriction": {
        "3gpp_ie": "CodebookConfig.ri-Restriction",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.214 §5.2.2.3",
        "oai_source": "nr_radio_config.c::config_csi_codebook()",
        "oai_config_key": "codebook_detailed_config.ri_restriction",
        "requires_restart": True,
        "description": "Bitmap restricting which RI values UE can report",
    },
    "codebook.n1_n2_config": {
        "3gpp_ie": "CodebookConfig.n1-n2",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.214 Table 5.2.2.2.1-2",
        "oai_source": "nr_radio_config.c::config_csi_codebook()",
        "oai_config_key": "codebook_detailed_config.n1_n2_config",
        "requires_restart": True,
        "description": "Antenna panel dimension for codebook (N1 x N2)",
    },

    # ── CSI Measurement ──────────────────────────────────────────
    "csi_measurement.report_periodicity": {
        "3gpp_ie": "CSI-ReportConfig.reportConfigType.periodic.reportSlotConfig",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → CSI-MeasConfig",
        "3gpp_spec": "TS 38.331 §6.3.2 (CSI-ReportConfig)",
        "oai_source": "nr_radio_config.c::config_csi_meas_report()",
        "oai_config_key": "csi_measurement_config.report_periodicity",
        "requires_restart": True,
        "description": "Periodicity of CSI measurement report in slots",
    },

    # ── Antenna ───────────────────────────────────────────────────
    "antenna.gnb.nx": {
        "3gpp_ie": "PDSCH-Config.codebookConfig (via N1)",
        "rrc_message": "RRCReconfiguration → CellGroupConfig",
        "3gpp_spec": "TS 38.214 Table 5.2.2.2.1-2",
        "oai_source": "gnb_config.c → pdsch_AntennaPorts_N1",
        "oai_config_key": "pdsch_AntennaPorts_N1",
        "requires_restart": True,
        "description": "Horizontal antenna elements (contributes to N1 dimension)",
    },
    "antenna.gnb.ny": {
        "3gpp_ie": "PDSCH-Config.codebookConfig (via N2)",
        "rrc_message": "RRCReconfiguration → CellGroupConfig",
        "3gpp_spec": "TS 38.214 Table 5.2.2.2.1-2",
        "oai_source": "gnb_config.c → pdsch_AntennaPorts_N2",
        "oai_config_key": "pdsch_AntennaPorts_N2",
        "requires_restart": True,
        "description": "Vertical antenna elements (N2 dimension)",
    },
    "antenna.gnb.polarization": {
        "3gpp_ie": "PDSCH-Config.codebookConfig (via XP)",
        "rrc_message": "RRCReconfiguration → CellGroupConfig",
        "3gpp_spec": "TS 38.214 §5.2.2.2",
        "oai_source": "gnb_config.c → pdsch_AntennaPorts_XP",
        "oai_config_key": "pdsch_AntennaPorts_XP",
        "requires_restart": True,
        "description": "Antenna polarization mode: single (XP=1) or dual (XP=2)",
    },

    # ── NGAP / Network ────────────────────────────────────────────
    "network.amf_ip": {
        "3gpp_ie": "NGAP: NGSetupRequest (peer address)",
        "rrc_message": "N/A (NGAP transport layer)",
        "3gpp_spec": "TS 38.413 §8.7",
        "oai_source": "gnb_config.c → amf_ip_address",
        "oai_config_key": "amf_ip_address.ipv4",
        "requires_restart": True,
        "description": "AMF IP address for NGAP/SCTP connection",
    },
    "network.gnb_ip": {
        "3gpp_ie": "NGAP: GNB-ID, GTP-U local endpoint",
        "rrc_message": "N/A (transport layer)",
        "3gpp_spec": "TS 38.413 §9.2.6",
        "oai_source": "gnb_config.c → GNB_IPV4_ADDRESS_FOR_NG_AMF",
        "oai_config_key": "GNB_IPV4_ADDRESS_FOR_NG_AMF",
        "requires_restart": True,
        "description": "gNB local IP for NGAP and GTP-U interfaces",
    },

    # ── System / MAC ──────────────────────────────────────────────
    "system.mu_mimo": {
        "3gpp_ie": "N/A (scheduler implementation)",
        "rrc_message": "N/A (MAC scheduler internal)",
        "3gpp_spec": "TS 38.214 §5.2 (MU-MIMO is scheduler decision)",
        "oai_source": "gNB_scheduler_dlsch.c (mu_mimo flag)",
        "oai_config_key": "MACRLCs.mu_mimo",
        "requires_restart": True,
        "description": "Enable MU-MIMO scheduling (pairs multiple UEs in same PRB)",
    },

    # ── Bearer / SDAP / DRB / RLC ─────────────────────────────────
    "bearer.enable_sdap": {
        "3gpp_ie": "SDAP-Config.sdap-HeaderDL / sdap-HeaderUL",
        "rrc_message": "RRCReconfiguration → RadioBearerConfig → DRB-ToAddMod → SDAP-Config",
        "3gpp_spec": "TS 38.331 §6.3.2 (SDAP-Config), TS 37.324",
        "oai_source": "rrc_gNB_radio_bearers.c::set_bearer_config()",
        "oai_config_key": "gNBs.enable_sdap",
        "requires_restart": True,
        "description": "Enable SDAP header (QoS flow mapping) on DL/UL",
    },
    "bearer.drbs": {
        "3gpp_ie": "DRB-ToAddModList",
        "rrc_message": "RRCReconfiguration → RadioBearerConfig → DRB-ToAddModList",
        "3gpp_spec": "TS 38.331 §6.3.2, TS 38.413 maxnoofDRBs=32",
        "oai_source": "rrc_gNB_radio_bearers.c::generateDRB()",
        "oai_config_key": "gNBs.drbs",
        "requires_restart": True,
        "description": "Number of DRBs per PDU session (each DRB creates one RLC entity)",
    },
    "bearer.um_on_default_drb": {
        "3gpp_ie": "RLC-BearerConfig.rlc-Config (UM-Bi-Directional vs AM)",
        "rrc_message": "RRCReconfiguration → CellGroupConfig → RLC-BearerConfig",
        "3gpp_spec": "TS 38.331 §6.3.2 (RLC-Config)",
        "oai_source": "mac_rrc_dl_handler.c::nr_rlc_add_drb()",
        "oai_config_key": "gNBs.um_on_default_drb",
        "requires_restart": True,
        "description": "Use RLC UM (Unacknowledged Mode) for default DRB instead of AM",
    },
    "bearer.drb_ciphering": {
        "3gpp_ie": "SecurityConfig.securityAlgorithmConfig.cipheringAlgorithm",
        "rrc_message": "RRCReconfiguration → RadioBearerConfig → SecurityConfig",
        "3gpp_spec": "TS 38.331 §6.3.2 (SecurityConfig)",
        "oai_source": "rrc_gNB_radio_bearers.c",
        "oai_config_key": "security.drb_ciphering",
        "requires_restart": True,
        "description": "Enable ciphering (NEA) for DRB PDCP",
    },
    "bearer.drb_integrity": {
        "3gpp_ie": "SecurityConfig.securityAlgorithmConfig.integrityProtAlgorithm",
        "rrc_message": "RRCReconfiguration → RadioBearerConfig → SecurityConfig",
        "3gpp_spec": "TS 38.331 §6.3.2 (SecurityConfig)",
        "oai_source": "rrc_gNB_radio_bearers.c",
        "oai_config_key": "security.drb_integrity",
        "requires_restart": True,
        "description": "Enable integrity protection (NIA) for DRB PDCP",
    },

    # ── QoS (Core Network / SMF / UDR) ─────────────────────────
    "qos.default_5qi": {
        "3gpp_ie": "QosFlowSetupRequestItem.qosFlowIdentifier + QoS characteristics",
        "rrc_message": "N/A (NGAP: PDUSessionResourceSetupRequest → QoS Flow)",
        "3gpp_spec": "TS 23.501 §5.7.2 (5QI), TS 38.413 §9.3.1.12",
        "oai_source": "ngap_gNB_handlers.c::fill_qos()",
        "oai_config_key": "smf.local_subscription_infos[].qos_profile.5qi",
        "requires_restart": False,
        "description": "Default 5G QoS Identifier (determines latency/priority class)",
    },
    "qos.session_ambr_ul": {
        "3gpp_ie": "PDUSessionAggregateMaximumBitRate.uL",
        "rrc_message": "N/A (NGAP: PDUSessionResourceSetupRequest)",
        "3gpp_spec": "TS 23.501 §5.7.2.6, TS 38.413 §9.3.1.12",
        "oai_source": "rrc_gNB_NGAP.c (Session AMBR)",
        "oai_config_key": "smf.local_subscription_infos[].qos_profile.session_ambr_ul",
        "requires_restart": False,
        "description": "Session Aggregate Maximum Bit Rate (uplink)",
    },
    "qos.session_ambr_dl": {
        "3gpp_ie": "PDUSessionAggregateMaximumBitRate.dL",
        "rrc_message": "N/A (NGAP: PDUSessionResourceSetupRequest)",
        "3gpp_spec": "TS 23.501 §5.7.2.6, TS 38.413 §9.3.1.12",
        "oai_source": "rrc_gNB_NGAP.c (Session AMBR)",
        "oai_config_key": "smf.local_subscription_infos[].qos_profile.session_ambr_dl",
        "requires_restart": False,
        "description": "Session Aggregate Maximum Bit Rate (downlink)",
    },
    "qos.arp_priority": {
        "3gpp_ie": "AllocationAndRetentionPriority.priorityLevel",
        "rrc_message": "N/A (NGAP: PDUSessionResourceSetupRequest → QoS Flow)",
        "3gpp_spec": "TS 23.501 §5.7.2.2",
        "oai_source": "ngap_gNB_handlers.c::fill_qos()",
        "oai_config_key": "UDR DB: SessionManagementSubscriptionData.arp.priorityLevel",
        "requires_restart": False,
        "description": "ARP priority level (1=highest, 15=lowest)",
    },

    # ── Channel (Sionna proxy) ────────────────────────────────────
    "channel.path_loss_dB": {
        "3gpp_ie": "N/A (simulation parameter)",
        "rrc_message": "N/A",
        "3gpp_spec": "TR 38.901 (channel model)",
        "oai_source": "v4.py pathLossLinear",
        "oai_config_key": "proxy: --path-loss-dB",
        "requires_restart": False,
        "description": "Path loss in dB applied by channel proxy (hot-swappable)",
    },
    "channel.snr_dB": {
        "3gpp_ie": "N/A (simulation parameter)",
        "rrc_message": "N/A",
        "3gpp_spec": "TR 38.901",
        "oai_source": "v4.py snr_dB",
        "oai_config_key": "proxy: --snr-dB",
        "requires_restart": False,
        "description": "Target SNR in dB for AWGN noise injection (hot-swappable)",
    },
    "channel.speed": {
        "3gpp_ie": "N/A (simulation parameter)",
        "rrc_message": "N/A",
        "3gpp_spec": "TR 38.901 (UE mobility)",
        "oai_source": "v4.py Speed",
        "oai_config_key": "proxy: --speed",
        "requires_restart": False,
        "description": "UE mobility speed in m/s for Doppler (hot-swappable)",
    },
}


def get_message_map(cfg: Optional[dict] = None) -> Dict[str, dict]:
    """Return the full parameter-to-message mapping."""
    return PARAM_MAP


def get_mapping_for_param(dotted_key: str) -> Optional[dict]:
    """Return mapping info for a single parameter."""
    return PARAM_MAP.get(dotted_key)


def get_affected_messages(changed_keys: List[str]) -> Dict[str, List[str]]:
    """Given a list of changed config keys, return which 3GPP messages are affected."""
    messages: Dict[str, List[str]] = {}
    for key in changed_keys:
        info = PARAM_MAP.get(key)
        if info:
            msg = info["rrc_message"]
            if msg not in messages:
                messages[msg] = []
            messages[msg].append(key)
    return messages


def get_restart_required(changed_keys: List[str]) -> bool:
    """Check if any changed key requires a gNB restart."""
    for key in changed_keys:
        info = PARAM_MAP.get(key)
        if info and info.get("requires_restart", True):
            return True
    return False


def get_hotswap_keys() -> List[str]:
    """Return list of parameter keys that can be changed without restart."""
    return [k for k, v in PARAM_MAP.items() if not v.get("requires_restart", True)]


def format_change_summary(changed_keys: List[str]) -> dict:
    """Format a human-readable summary of what a config change will affect."""
    affected_msgs = get_affected_messages(changed_keys)
    restart = get_restart_required(changed_keys)
    hotswap_keys = [k for k in changed_keys
                    if not PARAM_MAP.get(k, {}).get("requires_restart", True)]
    restart_keys = [k for k in changed_keys
                    if PARAM_MAP.get(k, {}).get("requires_restart", True)]

    return {
        "requires_restart": restart,
        "hotswap_params": hotswap_keys,
        "restart_params": restart_keys,
        "affected_3gpp_messages": affected_msgs,
        "details": {k: PARAM_MAP[k] for k in changed_keys if k in PARAM_MAP},
    }

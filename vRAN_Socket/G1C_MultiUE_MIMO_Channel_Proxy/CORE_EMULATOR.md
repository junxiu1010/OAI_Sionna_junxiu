# Core Emulator 설명서

## 1. 개요

Core Emulator는 OAI 5G 시뮬레이터의 모든 PHY/채널/안테나/멀티셀 설정을
하나의 YAML 파일(`master_config.yaml`)로 통합 관리하는 **중앙 설정 서버**이다.

실제 5GC(코어 네트워크) NMS와 동일한 REST API 형태로 설계되어,
추후 코어 에뮬레이터를 떼어내고 실제 코어에 연결해도 클라이언트 변경 없이 동작한다.

### 1.1 기존 문제점

설정이 3곳에 분산·하드코딩되어 있었다:

| 기존 위치 | 내용 |
|-----------|------|
| `gnb.conf` | gNB PHY/MAC/RRC (코드북, CSI-RS, SRS, 안테나 포트 등) |
| `launch_all.sh` | CLI 옵션, 안테나 계산, OAI 인자 조합 |
| `v0.py` | Proxy 글로벌 상수 + argparse 기본값 |

파라미터 변경 시 3개 파일을 수동으로 일관성 있게 수정해야 했으며,
런타임 변경은 불가능했다.

### 1.2 Core Emulator 도입 후

```
master_config.yaml  ─→  Core Emulator 서버 (FastAPI)
                              │
                              ├─→  gnb.conf (Jinja2 렌더링: PHY + Bearer 설정)
                              ├─→  CN5G config.yaml (Jinja2 렌더링: SMF QoS 프로파일)
                              ├─→  MySQL UDR DB (가입자별 QoS 업데이트)
                              ├─→  launch_multicell.sh (시작 시 파라미터)
                              ├─→  v4.py / v4_multicell.py Proxy (초기 설정 + 런타임 핫스왑)
                              ├─→  프리셋 적용 (YAML 프로필)
                              ├─→  Intent 파싱 (자연어 → 설정)
                              ├─→  3GPP 규격 검증 (파라미터 유효성)
                              └─→  셀/UE 관리 (NMS 스타일 API)
```

## 2. 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Core Emulator (Python + FastAPI)                    │
│                                                                       │
│  ┌─────────────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │ master_config   │  │ presets/*.yaml  │  │ templates/            │  │
│  │  .yaml          │  │ (6개 프리셋)    │  │  gnb.conf.j2          │  │
│  │ (bearer/qos     │  │ (bearer/qos    │  │  cn5g_config.yaml.j2  │  │
│  │  섹션 포함)      │  │  기본값 포함)   │  │                       │  │
│  └────────┬────────┘  └───────┬────────┘  └───────────┬───────────┘  │
│           │                   │                        │              │
│           ▼                   ▼                        ▼              │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │                     CoreState (설정 관리)                       │    │
│  │  - 설정 로드/저장      - 프리셋 적용      - gnb.conf 렌더링    │    │
│  │  - 파생값 계산         - 파라미터 검증     - 셀/UE 상태 관리    │    │
│  │  - bearer/qos 관리    - CN5G config 렌더링 - DB QoS 업데이트   │    │
│  └───────────────────────────┬───────────────────────────────────┘    │
│                              │                                        │
│  ┌──────────┐  ┌─────────────┴──────────┐  ┌──────────────────────┐  │
│  │ intent   │  │    FastAPI REST         │  │  Legacy TCP/JSON     │  │
│  │ _parser  │  │    (port 7101)          │  │  (port 7100)         │  │
│  ├──────────┤  ├────────────────────────┤  ├──────────────────────┤  │
│  │ param    │  │ /api/v1/config         │  │ GET_CONFIG           │  │
│  │ _valid   │  │ /api/v1/cell/*         │  │ UPDATE_PROXY         │  │
│  │ ator     │  │ /api/v1/bearer  ← NEW  │  │ SUBSCRIBE_PROXY      │  │
│  ├──────────┤  │ /api/v1/qos     ← NEW  │  │ ...                  │  │
│  │ message  │  │ /api/v1/qos/apply      │  │                      │  │
│  │ _mapper  │  │ /api/v1/qos/db-update  │  │                      │  │
│  └──────────┘  └────────────────────────┘  └──────────────────────┘  │
└───┬─────────────┬─────────────┬────────────┬────────────┬────────────┘
    │             │             │            │            │
gnb.conf 생성  CN5G config   Proxy 설정  MySQL DB    gNB/SMF
(bearer 포함)  (QoS 프로파일) (채널 핫스왑) (가입자 QoS) 재시작
    │             │             │            │            │
    ▼             ▼             ▼            ▼            ▼
nr-softmodem   oai-smf      v4_multicell  mysql       docker
               컨테이너       .py          컨테이너    restart
```

### 2.1 통신 인터페이스

| 인터페이스 | 포트 | 프레임워크 | 용도 |
|-----------|------|-----------|------|
| **FastAPI REST** | 7101 | FastAPI + Uvicorn | 메인 API (모든 새 기능) |
| **Legacy TCP/JSON** | 7100 | asyncio socket | v0.py Proxy 하위 호환 |
| **Legacy HTTP GET** | 7101 `/config` 등 | FastAPI | launch_all.sh `curl` 호환 |

## 3. FastAPI란?

**FastAPI**는 Python 기반 고성능 웹 프레임워크이다.

| 특징 | 설명 |
|------|------|
| **고성능** | 내부적으로 Starlette(ASGI) + Uvicorn(비동기 서버) 사용. Node.js/Go급 성능 |
| **자동 문서화** | API 생성 시 Swagger UI (`/docs`)와 ReDoc (`/redoc`) 자동 생성 |
| **타입 검증** | Pydantic 모델로 요청/응답 데이터를 자동 검증 |
| **async 지원** | `async/await` 네이티브 지원 |

### 3.1 왜 FastAPI를 선택했나?

| 비교 | Flask | Django | **FastAPI** |
|------|-------|--------|-------------|
| 성능 | 느림 (WSGI) | 보통 | **빠름 (ASGI)** |
| 타입 검증 | 수동 | 수동 | **Pydantic 자동** |
| API 문서 | 별도 설치 | 별도 설치 | **자동 생성** |
| async | 제한적 | 제한적 | **네이티브** |
| 학습 곡선 | 낮음 | 높음 | **낮음** |

### 3.2 Pydantic 모델 예시

```python
from pydantic import BaseModel

class IntentRequest(BaseModel):
    text: str           # "4T4R MU-MIMO로 설정해줘"

class ConfigUpdateRequest(BaseModel):
    overrides: dict     # {"channel": {"snr_dB": 20}}
    apply_gnb: bool = False

class PresetApplyRequest(BaseModel):
    name: str           # "mu_mimo_4t4r_type2"
    apply_gnb: bool = False
```

FastAPI가 요청 JSON을 자동으로 검증하고, 잘못된 형식이면 422 에러를 반환한다.

### 3.3 실제 코어 교체 시 장점

Core Emulator의 FastAPI 엔드포인트들은 실제 5GC NMS의 REST API와 동일한 구조로 설계되어 있다:

```
Core Emulator:  POST /api/v1/cell/configure    → 셀 설정
실제 5GC:       POST /api/v1/cell/configure    → 동일 인터페이스

교체 시 URL과 인증만 변경하면 클라이언트 코드 수정 불필요
```

## 4. 파일 구조

```
G1C_MultiUE_MIMO_Channel_Proxy/
├── core_emulator.py          # Core Emulator 서버 (FastAPI + Legacy TCP)
├── core_cli.py               # CLI 클라이언트
├── master_config.yaml        # 마스터 설정 (유일한 설정 원본)
├── param_validator.py        # 3GPP 규격 기반 파라미터 유효성 검증 (bearer/QoS 포함)
├── message_mapper.py         # 파라미터 → 3GPP RRC/NGAP 메시지 매핑 (SDAP/DRB/QoS 포함)
├── intent_parser.py          # 자연어 Intent → 구조화된 설정 변환
├── generate_gnb_configs.py   # 멀티셀 gnb.conf 배치 생성기
├── templates/
│   ├── gnb.conf.j2           # gnb.conf Jinja2 템플릿 (bearer 파라미터 포함)
│   └── cn5g_config.yaml.j2   # OAI CN5G config 템플릿 (SMF QoS 프로파일)
├── presets/                   # 프리셋 프로필 디렉토리 (모두 bearer/qos 기본값 포함)
│   ├── su_mimo_2t2r_type1.yaml
│   ├── su_mimo_4t4r_type1.yaml
│   ├── mu_mimo_4t4r_type2.yaml
│   ├── mu_mimo_8t8r_type2.yaml
│   ├── multicell_2cell_4ue.yaml
│   └── multicell_4cell_4ue.yaml
├── v4.py                     # 단일셀 Proxy
├── v4_multicell.py           # 멀티셀 Proxy
└── launch_multicell.sh       # 멀티셀 통합 런처
```

## 5. master_config.yaml 구조

모든 설정을 12개 섹션으로 구분한다:

### 5.1 `system` — 시스템 전역 설정

```yaml
system:
  num_ues: 4               # UE 수
  mode: gpu-ipc            # gpu-ipc | socket
  channel_mode: static     # dynamic | static
  batch_size: 4            # UE 배치 기동 크기
  mu_mimo: false           # MU-MIMO 스케줄링 활성화
  force_mu_dl_traffic: false  # MU-MIMO 테스트용 더미 DL 데이터 주입
```

### 5.2 `antenna` — 안테나 배열

```yaml
antenna:
  gnb:
    nx: 2                  # 수평 안테나 수
    ny: 1                  # 수직 안테나 수
    polarization: dual     # single (V, XP=1) | dual (cross ±45°, XP=2)
  ue:
    nx: 2
    ny: 1
```

**자동 파생값** (Core Emulator가 내부에서 계산):

| 파생값 | 계산식 | 예시 (dual, 2x1) |
|--------|--------|-------------------|
| `pol_mult` | dual→2, single→1 | 2 |
| `gnb_spatial` | nx × ny | 2 |
| `gnb_ant` | gnb_spatial × pol_mult | 4 |
| `ue_ant` | ue_nx × ue_ny × pol_mult | 4 |
| `xp` | pol_mult | 2 |
| `n1` | xp==2 → gnb_spatial, else gnb_ant | 2 |

이 파생값들이 `gnb.conf`의 `pdsch_AntennaPorts_N1`, `pdsch_AntennaPorts_XP`,
`nb_tx`, `nb_rx` 등에 자동 반영된다.

### 5.3 `codebook` — 코드북 설정

```yaml
codebook:
  type: type2                        # type1 | type2
  sub_type: typeII_PortSelection     # typeI_SinglePanel | typeII_PortSelection
  mode: 1
  pmi_restriction: 0xff
  ri_restriction: 0x03
  n1_n2_config: two_one
  phase_alphabet_size: n4            # n4 (QPSK) | n8 (8PSK) — Type-II 전용
  subband_amplitude: 0
  number_of_beams: 2
  port_selection_sampling_size: 2
```

### 5.4 `csi_rs` — CSI-RS 설정

```yaml
csi_rs:
  periodicity: 20          # 슬롯 주기
  nrof_ports: 4            # CSI-RS 포트 수
  cdm_type: fd_CDM2        # noCDM | fd_CDM2 | cdm4_FD2_TD2 | cdm8_FD2_TD4
  density: 1
  freq_start_rb: 0
  freq_nrof_rbs: 106
```

### 5.5 `srs` — SRS 설정

```yaml
srs:
  periodicity: 20          # 슬롯 주기
  nrof_srs_ports: 1        # SRS 포트 수
```

### 5.6 `csi_measurement` — CSI 측정 보고 설정

```yaml
csi_measurement:
  report_periodicity: 20   # 보고 주기 (슬롯)
  cqi_table: table1
  enable_ri: 1
  enable_pmi: 1
  enable_cqi: 1
```

### 5.7 `channel` — Sionna 채널 모델 파라미터

```yaml
channel:
  path_loss_dB: 0.0        # 경로 손실
  snr_dB: null             # 상대 SNR (null=비활성)
  noise_dBFS: null         # 절대 노이즈 플로어 (null=비활성)
  speed: 3.0               # UE 이동 속도 (m/s)
  scenario: UMa-NLOS       # 3GPP 채널 시나리오
  sector_half_deg: 90.0    # 셀 섹터 반각 (도)
  jitter_std_deg: 20.0     # AoD/AoA 지터 표준편차 (도)
  # 3GPP TR 38.901 topology parameters
  bs_height_m: 25.0        # BS 안테나 높이 (UMi=10, UMa=25)
  ue_height_m: 1.5         # UE 높이 (1.5~2.5 m)
  isd_m: 500               # Inter-Site Distance (UMi=200, UMa=500)
  min_ue_distance_m: 35    # 최소 BS-UE 수평 거리
  max_ue_distance_m: 500   # 최대 BS-UE 수평 거리
  # 3GPP TR 38.901 large-scale fading
  shadow_fading_std_dB: 6.0  # shadow fading σ
  k_factor_mean_dB: null   # K-factor 평균 (LOS only; null=시나리오 기본값)
  k_factor_std_dB: null    # K-factor 표준편차 (LOS only; null=시나리오 기본값)
```

**3GPP TR 38.901 시나리오별 기본값:**

| 파라미터 | UMi-LOS | UMi-NLOS | UMa-LOS | UMa-NLOS |
|---------|---------|----------|---------|----------|
| BS Height | 10 m | 10 m | 25 m | 25 m |
| UE Height | 1.5~2.5 m | 1.5~2.5 m | 1.5~2.5 m | 1.5~2.5 m |
| ISD | 200 m | 200 m | 500 m | 500 m |
| UE Distance | 10~150 m | 10~150 m | 35~500 m | 35~500 m |
| Shadow Fading σ | 4 dB | 7.82 dB | 4 dB | 6 dB |
| K-factor μ/σ | 9/5 dB | N/A | 9/3.5 dB | N/A |

**Proxy 내부 처리 (v4.py → Sionna):**

```
master_config.yaml channel 섹션
    → Core Emulator → launch_all.sh → v4.py --bs-height-m 25 ...
    → UnifiedChannelProducerProcess:
        1. UE 수평 거리: Uniform[min_d, max_d] → d_2d
        2. 3D 거리: d_3d = √(d_2d² + (BS_h - UE_h)²)
        3. LOS 확률: TR 38.901 Table 7.4.2-1 기반
        4. 앙각 계산: elev = arctan(Δh / d_2d)
        5. Shadow fading: N(0, σ²) → power_rays 스케일링
        6. K-factor: N(μ, σ²) (LOS만) → Topology에 반영
        7. Topology(velocities, los, distance_3d, ...) → Sionna CIR
```

### 5.8 `carrier` — 캐리어 설정

```yaml
carrier:
  frequency_GHz: 3.5
  bandwidth_prb: 106
  scs_kHz: 30
  band: 78
```

### 5.9 `multicell` — 멀티셀 설정

```yaml
multicell:
  enabled: false           # 멀티셀 활성화
  num_cells: 1             # 셀 수
  ues_per_cell: 4          # 셀당 UE 수
  ici_atten_dB: 15.0       # 셀간 간섭 감쇠 (dB)
  p1b_npz: null            # Ray-tracing 데이터 경로
  per_cell_overrides: {}   # 셀별 오버라이드 (예: {0: {carrier: {band: 77}}})
```

### 5.10 `bearer` — SDAP/DRB/RLC 베어러 설정

gNB 측의 데이터 라디오 베어러 관련 파라미터를 제어한다.
`gnb.conf`의 gNBs 블록과 security 블록에 반영된다.

```yaml
bearer:
  enable_sdap: false          # SDAP 헤더 DL/UL 삽입 (QoS flow ↔ DRB 매핑)
  drbs: 1                     # PDU 세션당 DRB 수 (1-32, OAI 기본값=1)
  um_on_default_drb: false    # 기본 DRB에 RLC UM 사용 (false=AM)
  drb_ciphering: true         # DRB PDCP 암호화 (NEA)
  drb_integrity: false        # DRB PDCP 무결성 보호 (NIA)
```

**gnb.conf 매핑:**

| YAML 키 | gnb.conf 위치 | 영향 |
|----------|-------------|------|
| `enable_sdap` | `gNBs.enable_sdap` | SDAP 헤더 활성화 → QoS Flow-DRB 매핑 |
| `drbs` | `gNBs.drbs` | PDU 세션당 생성할 DRB 수 |
| `um_on_default_drb` | `gNBs.um_on_default_drb` | RLC 모드 (AM=재전송, UM=비재전송) |
| `drb_ciphering` | `security.drb_ciphering` | DRB 데이터 암호화 |
| `drb_integrity` | `security.drb_integrity` | DRB 데이터 무결성 보호 |

**OAI 내부 처리 흐름:**

```
bearer.drbs=2 설정 시:
  gnb.conf → rrc_gNB_radio_bearers.c::generateDRB()
    → DRB1 (LCID=4) + DRB2 (LCID=5) 생성
    → 각 DRB에 대해 SDAP, PDCP, RLC 엔티티 생성
    → nr_rlc_add_drb() 호출로 RLC 채널 생성

bearer.um_on_default_drb=true 설정 시:
  RLC AM (재전송 보장) 대신 RLC UM (낮은 레이턴시, 비재전송) 사용
  → URLLC 시나리오에 유용
```

### 5.11 `qos` — QoS 프로파일 (코어 네트워크 측)

코어 네트워크(SMF)에서 gNB로 전달되는 QoS 파라미터를 제어한다.
SMF `config.yaml`의 `local_subscription_infos`와 MySQL UDR DB에 반영된다.

```yaml
qos:
  default_5qi: 9              # 기본 5QI (5G QoS Identifier)
  session_ambr_ul: "10Gbps"   # 세션 AMBR 업링크
  session_ambr_dl: "10Gbps"   # 세션 AMBR 다운링크
  arp_priority: 15            # ARP 우선순위 (1=최고, 15=최저)
  arp_preempt_cap: "NOT_PREEMPT"   # 선점 능력
  arp_preempt_vuln: "PREEMPTABLE"  # 선점 취약성
```

**5QI 값 의미 (3GPP TS 23.501 §5.7.4):**

| 5QI | 리소스 유형 | 우선순위 | 지연 한도 | 에러율 | 용도 |
|-----|-----------|---------|----------|-------|------|
| 1 | GBR | 2 | 100ms | 10⁻² | 대화형 음성 |
| 2 | GBR | 4 | 150ms | 10⁻³ | 대화형 영상 |
| 5 | Non-GBR | 1 | 100ms | 10⁻⁶ | IMS 시그널링 |
| 6 | Non-GBR | 6 | 300ms | 10⁻⁶ | 영상 스트리밍 |
| 9 | Non-GBR | 9 | 300ms | 10⁻⁶ | 기본 인터넷 (Best Effort) |

**QoS 파라미터 흐름:**

```
master_config.yaml (qos 섹션)
    │
    ├─→ /api/v1/qos/apply ─→ cn5g_config.yaml.j2 렌더링
    │                           → SMF config.yaml 덮어쓰기
    │                           → docker restart oai-smf
    │
    └─→ /api/v1/qos/db-update ─→ MySQL SessionManagementSubscriptionData
                                   → 가입자별 dnnConfigurations UPDATE
```

### 5.12 `network` — 네트워크 설정

```yaml
network:
  amf_ip: 192.168.70.132
  gnb_ip: 192.168.70.129
  gnb_ip_prefix: 24
```

## 6. REST API 엔드포인트 (FastAPI)

### 6.1 설정 관리

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/config` | 현재 전체 설정 반환 |
| `POST` | `/api/v1/config` | 설정 직접 업데이트 (JSON override) |
| `GET` | `/api/v1/gnb-conf` | 렌더링된 gnb.conf 텍스트 반환 |
| `GET` | `/api/v1/derived` | 자동 파생값 반환 |
| `GET` | `/api/v1/validate` | 현재 설정의 유효성 검증 결과 |
| `GET` | `/api/v1/launch-params` | launch 스크립트용 파라미터 |
| `GET` | `/api/v1/proxy-params` | Proxy용 파라미터 |
| `GET` | `/api/v1/hotswap-keys` | 핫스왑 가능 파라미터 목록 |

### 6.2 프리셋

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/presets` | 사용 가능한 프리셋 목록 |
| `GET` | `/api/v1/presets/{name}` | 특정 프리셋 상세 내용 |
| `POST` | `/api/v1/apply-preset` | 프리셋 적용 |

### 6.3 Intent (자연어 처리)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/v1/intent` | 자연어 → 설정 변환 및 적용 |

### 6.4 3GPP 메시지 매핑

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/message-map` | 전체 파라미터 → 3GPP 메시지 매핑 |
| `GET` | `/api/v1/message-map/{key}` | 특정 파라미터의 매핑 정보 |

### 6.5 NMS 스타일 셀/UE 관리

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/v1/cell/configure` | 셀 설정 (cell_id, pci, frequency 등) |
| `POST` | `/api/v1/cell/{id}/activate` | 셀 활성화 |
| `GET` | `/api/v1/cell/{id}/status` | 특정 셀 상태 조회 |
| `GET` | `/api/v1/cells` | 전체 셀 목록 |
| `GET` | `/api/v1/kpi` | KPI 요약 (셀 수, UE 수, 설정 요약) |

### 6.6 Bearer/QoS 관리 (SDAP/DRB/RLC)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/bearer` | 현재 bearer + qos 설정 조회 |
| `POST` | `/api/v1/bearer` | bearer 설정 업데이트 (gNB 재시작 필요) |
| `POST` | `/api/v1/qos` | QoS 프로파일 업데이트 (메모리만) |
| `POST` | `/api/v1/qos/apply` | CN5G config 렌더링 → SMF config.yaml 갱신 → SMF 컨테이너 재시작 |
| `GET` | `/api/v1/cn5g-conf` | 렌더링된 CN5G config.yaml 미리보기 |
| `POST` | `/api/v1/qos/db-update` | MySQL UDR DB의 가입자별 QoS 직접 업데이트 |

**2단계 제어 구조:**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1단계: gNB 측 (gnb.conf)                                        │
│   POST /api/v1/bearer → master_config 업데이트                   │
│   POST /api/v1/apply  → gnb.conf.j2 렌더링 → gNB 재시작         │
│                                                                  │
│   bearer.enable_sdap     → gnb.conf: gNBs.enable_sdap           │
│   bearer.drbs            → gnb.conf: gNBs.drbs                  │
│   bearer.um_on_default_drb → gnb.conf: gNBs.um_on_default_drb   │
│   bearer.drb_ciphering   → gnb.conf: security.drb_ciphering     │
│   bearer.drb_integrity   → gnb.conf: security.drb_integrity     │
├─────────────────────────────────────────────────────────────────┤
│ 2단계: 코어 측 (SMF + DB)                                        │
│   POST /api/v1/qos       → master_config 업데이트                │
│   POST /api/v1/qos/apply → cn5g_config.yaml.j2 렌더링           │
│                           → SMF config.yaml 덮어쓰기             │
│                           → docker restart oai-smf               │
│                                                                  │
│   POST /api/v1/qos/db-update → MySQL 직접 UPDATE                │
│                                                                  │
│   qos.default_5qi        → SMF 5qi + DB 5gQosProfile.5qi        │
│   qos.session_ambr_ul/dl → SMF session_ambr + DB sessionAmbr    │
│   qos.arp_priority       → DB arp.priorityLevel                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.7 서버 상태

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/status` | 서버 상태, 버전, 업타임 등 |
| `POST` | `/api/v1/apply` | gnb.conf 렌더링 + gNB 재시작 플래그 설정 |

### 6.8 Legacy HTTP (하위 호환)

| 경로 | 용도 |
|------|------|
| `GET /config` | 전체 설정 (JSON) |
| `GET /gnb_conf` | 렌더링된 gnb.conf (text/plain) |
| `GET /launch_params` | 런치 파라미터 (JSON) |
| `GET /proxy_params` | 프록시 파라미터 (JSON) |
| `GET /status` | 서버 상태 (JSON) |

## 7. API 사용 예시

### 7.1 현재 설정 조회

```bash
curl http://localhost:7101/api/v1/config | python3 -m json.tool
```

### 7.2 Intent 파싱 (자연어 → 설정)

```bash
# "4T4R MU-MIMO Type-II"로 설정
curl -X POST http://localhost:7101/api/v1/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "4T4R dual-pol MU-MIMO Type-II codebook"}'
```

응답 예시:
```json
{
  "status": "ok",
  "intent_result": {
    "preset": "mu_mimo_4t4r_type2",
    "overrides": {"antenna": {"gnb": {"nx": 2, "ny": 1, "polarization": "dual"}}},
    "confidence": 0.85
  },
  "validation": {"valid": true, "errors": []}
}
```

### 7.3 프리셋 적용

```bash
# 사용 가능한 프리셋 목록
curl http://localhost:7101/api/v1/presets

# 프리셋 적용
curl -X POST http://localhost:7101/api/v1/apply-preset \
  -H "Content-Type: application/json" \
  -d '{"name": "mu_mimo_4t4r_type2"}'
```

### 7.4 직접 설정 업데이트

```bash
curl -X POST http://localhost:7101/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{"overrides": {"channel": {"snr_dB": 20}, "system": {"num_ues": 8}}}'
```

### 7.5 gnb.conf 생성 + gNB 재시작

```bash
curl -X POST http://localhost:7101/api/v1/apply \
  -H "Content-Type: application/json" \
  -d '{"save_path": "/tmp/gnb.conf"}'
```

### 7.6 셀 관리 (NMS 스타일)

```bash
# 셀 설정
curl -X POST http://localhost:7101/api/v1/cell/configure \
  -H "Content-Type: application/json" \
  -d '{"cell_id": 0, "pci": 0, "frequency_ghz": 3.5, "bandwidth_prb": 106}'

# 셀 활성화
curl -X POST http://localhost:7101/api/v1/cell/0/activate

# 셀 상태 조회
curl http://localhost:7101/api/v1/cell/0/status

# 전체 KPI
curl http://localhost:7101/api/v1/kpi
```

### 7.7 유효성 검증

```bash
curl http://localhost:7101/api/v1/validate
```

응답 예시:
```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Type-II codebook requires >= 4 gNB antennas (current: 4) — OK"]
}
```

### 7.8 Bearer 설정 변경 (gNB 측 SDAP/DRB/RLC)

```bash
# 현재 bearer/qos 설정 조회
curl http://localhost:7101/api/v1/bearer | python3 -m json.tool
```

응답 예시:
```json
{
  "bearer": {
    "enable_sdap": false,
    "drbs": 1,
    "um_on_default_drb": false,
    "drb_ciphering": true,
    "drb_integrity": false
  },
  "qos": {
    "default_5qi": 9,
    "session_ambr_ul": "10Gbps",
    "session_ambr_dl": "10Gbps",
    "arp_priority": 15,
    "arp_preempt_cap": "NOT_PREEMPT",
    "arp_preempt_vuln": "PREEMPTABLE"
  }
}
```

```bash
# DRB 수를 2개로 늘리고 SDAP 활성화
curl -X POST http://localhost:7101/api/v1/bearer \
  -H "Content-Type: application/json" \
  -d '{"drbs": 2, "enable_sdap": true}'

# URLLC 시나리오: RLC UM 모드 + 무결성 보호 활성화
curl -X POST http://localhost:7101/api/v1/bearer \
  -H "Content-Type: application/json" \
  -d '{"um_on_default_drb": true, "drb_integrity": true}'

# gnb.conf 적용 (재시작 플래그 설정)
curl -X POST http://localhost:7101/api/v1/apply
```

### 7.9 QoS 프로파일 변경 (코어 네트워크 측)

```bash
# QoS 프로파일 변경 (메모리만, 아직 적용 안 됨)
curl -X POST http://localhost:7101/api/v1/qos \
  -H "Content-Type: application/json" \
  -d '{"default_5qi": 5, "session_ambr_dl": "1Gbps", "session_ambr_ul": "500Mbps"}'

# 렌더링된 CN5G config 미리보기
curl http://localhost:7101/api/v1/cn5g-conf

# SMF에 실제 적용 (config.yaml 쓰기 + SMF 컨테이너 재시작)
curl -X POST http://localhost:7101/api/v1/qos/apply
```

응답 예시:
```json
{
  "status": "ok",
  "msg": "CN5G config applied",
  "config_path": "/home/.../oai-cn5g/conf/config.yaml",
  "smf_restart": "success"
}
```

### 7.10 MySQL DB 가입자 QoS 직접 업데이트

```bash
# 전체 가입자의 QoS를 영상 통화 프로파일로 변경
curl -X POST http://localhost:7101/api/v1/qos/db-update \
  -H "Content-Type: application/json" \
  -d '{"default_5qi": 2, "session_ambr_dl": "5Gbps", "session_ambr_ul": "2Gbps"}'

# 특정 가입자만 업데이트
curl -X POST http://localhost:7101/api/v1/qos/db-update \
  -H "Content-Type: application/json" \
  -d '{
    "imsi_list": ["001010000000001", "001010000000002"],
    "default_5qi": 1,
    "session_ambr_dl": "100Mbps"
  }'
```

### 7.11 Bearer/QoS 전체 워크플로우 예시

```bash
# ── 시나리오: MU-MIMO + 2 DRB + 영상 스트리밍 QoS ────────────

# 1. 프리셋 적용 (MU-MIMO 기본 설정)
curl -X POST http://localhost:7101/api/v1/apply-preset \
  -d '{"preset": "mu_mimo_4t4r_type2"}'

# 2. gNB 측 베어러 설정 (DRB 2개, SDAP 활성화)
curl -X POST http://localhost:7101/api/v1/bearer \
  -d '{"drbs": 2, "enable_sdap": true}'

# 3. QoS 프로파일 변경 (5QI=6: 영상 스트리밍)
curl -X POST http://localhost:7101/api/v1/qos \
  -d '{"default_5qi": 6, "session_ambr_dl": "5Gbps"}'

# 4. 유효성 검증
curl http://localhost:7101/api/v1/validate

# 5. gNB에 적용 (gnb.conf 렌더링)
curl -X POST http://localhost:7101/api/v1/apply

# 6. SMF에 적용 (CN5G config 업데이트 + SMF 재시작)
curl -X POST http://localhost:7101/api/v1/qos/apply

# 7. DB에도 반영 (가입자 QoS 일괄 업데이트)
curl -X POST http://localhost:7101/api/v1/qos/db-update \
  -d '{"default_5qi": 6, "session_ambr_dl": "5Gbps"}'
```

### 7.12 3GPP 메시지 매핑 조회

```bash
# 전체 매핑
curl http://localhost:7101/api/v1/message-map

# 특정 파라미터
curl http://localhost:7101/api/v1/message-map/codebook.type
```

응답 예시:
```json
{
  "codebook.type": {
    "3gpp_ie": "CodebookConfig",
    "rrc_message": "RRCReconfiguration → CSI-MeasConfig → CodebookConfig",
    "3gpp_spec": "TS 38.331 §6.3.2",
    "oai_source": "openair2/RRC/NR/nr_rrc_config.c",
    "oai_config_key": "codebook_type",
    "requires_restart": true,
    "description": "Codebook type (Type-I or Type-II)"
  }
}
```

## 8. 모듈 상세 설명

### 8.1 param_validator.py — 3GPP 파라미터 유효성 검증

3GPP 규격에 기반하여 설정 파라미터의 유효성을 검증한다.

**검증 항목:**

| 카테고리 | 검증 내용 |
|----------|----------|
| 안테나 | nx, ny ≥ 1, polarization ∈ {single, dual} |
| 코드북 | type ∈ {type1, type2}, Type-II는 gNB 안테나 ≥ 4 필요 |
| CSI-RS | periodicity ∈ {4,5,8,10,16,20,40,80,160,320}, nrof_ports ∈ {1,2,4,8,12,16,24,32} |
| SRS | periodicity ∈ {1,2,4,5,8,10,16,20,32,40,64,80,160,320,640,1280,2560} |
| 캐리어 | scs_kHz ∈ {15,30,60,120,240}, bandwidth_prb ∈ {11,24,52,79,106,133,162,216,270} |
| **Bearer** | drbs 1~32 (TS 38.413 maxnoofDRBs) |
| **QoS** | 5QI ∈ 표준 GBR/Non-GBR 값, ARP 우선순위 1~15 (TS 23.501), 선점 능력/취약성 유효값 |
| 교차 검증 | Type-II + gNB 안테나 < 4 → 에러, CDM 타입과 포트 수 호환 등 |

```python
from param_validator import validate_config

errors = validate_config(config_dict)
if errors:
    for e in errors:
        print(f"[{e.section}.{e.param}] {e.message}")
```

### 8.2 message_mapper.py — 3GPP 메시지 매핑

설정 파라미터가 어떤 3GPP ASN.1 IE, RRC/NGAP 메시지에 매핑되는지,
OAI 소스코드의 어디에 구현되어 있는지를 제공한다.

**매핑 카테고리:**

| 카테고리 | 파라미터 수 | 주요 RRC/NGAP 메시지 |
|----------|-----------|---------------------|
| CSI-RS | 4개 | RRCReconfiguration → CSI-MeasConfig |
| SRS | 2개 | RRCReconfiguration → SRS-Config |
| 코드북 | 4개 | RRCReconfiguration → CodebookConfig |
| 안테나 | 3개 | RRCReconfiguration → CellGroupConfig |
| 시스템/MAC | 1개 | MAC 스케줄러 내부 |
| **Bearer (SDAP/DRB/RLC)** | **5개** | **RRCReconfiguration → RadioBearerConfig, SecurityConfig** |
| **QoS (5QI/AMBR/ARP)** | **4개** | **NGAP: PDUSessionResourceSetupRequest** |
| 채널 (Proxy) | 3개 | N/A (시뮬레이션) |
| 네트워크 | 2개 | NGAP: NGSetupRequest |

**Bearer/QoS 매핑 상세:**

```python
PARAM_MAP = {
    "bearer.enable_sdap": {
        "3gpp_ie": "SDAP-Config.sdap-HeaderDL / sdap-HeaderUL",
        "rrc_message": "RRCReconfiguration → RadioBearerConfig → DRB-ToAddMod → SDAP-Config",
        "oai_source": "rrc_gNB_radio_bearers.c::set_bearer_config()",
    },
    "bearer.drbs": {
        "3gpp_ie": "DRB-ToAddModList",
        "rrc_message": "RRCReconfiguration → RadioBearerConfig → DRB-ToAddModList",
        "oai_source": "rrc_gNB_radio_bearers.c::generateDRB()",
    },
    "qos.default_5qi": {
        "3gpp_ie": "QosFlowSetupRequestItem.qosFlowIdentifier",
        "rrc_message": "N/A (NGAP: PDUSessionResourceSetupRequest → QoS Flow)",
        "oai_source": "ngap_gNB_handlers.c::fill_qos()",
    },
    "qos.session_ambr_dl": {
        "3gpp_ie": "PDUSessionAggregateMaximumBitRate.dL",
        "rrc_message": "N/A (NGAP: PDUSessionResourceSetupRequest)",
        "oai_source": "rrc_gNB_NGAP.c (Session AMBR)",
    },
    # ... 기타 파라미터
}
```

**활용:**
- `get_affected_messages(overrides)` → 변경 시 영향받는 3GPP 메시지 목록
- `get_restart_required(overrides)` → 재시작 필요 여부 판단
- `get_hotswap_keys()` → 핫스왑 가능 파라미터 목록
- `format_change_summary(overrides)` → 변경 요약 텍스트 생성

### 8.3 intent_parser.py — Intent 해석

자연어 입력을 구조화된 설정 오버라이드로 변환한다.
현재는 **룰 기반 + 프리셋 매칭** 방식이며, 추후 LLM 연동이 가능하도록 설계되어 있다.

**처리 흐름:**

```
"4T4R dual-pol MU-MIMO Type-II"
        │
        ▼
  1. 정규식 패턴 매칭 (안테나 수, 편파)
  2. 키워드 매칭 (MIMO 타입, 코드북 타입)
  3. 안테나 설정 추론 (_infer_antenna_config)
  4. 최적 프리셋 선택 (_select_preset)
        │
        ▼
  IntentResult(
      preset="mu_mimo_4t4r_type2",
      overrides={"antenna": {"gnb": {"nx": 2, "ny": 1}}},
      confidence=0.85
  )
```

**지원 패턴:**
- 안테나: `"4T4R"`, `"8x8"`, `"2 antenna"` 등
- MIMO: `"mu-mimo"`, `"su-mimo"`, `"multi-user"` 등
- 코드북: `"type1"`, `"type-II"`, `"Type 2"` 등
- 편파: `"dual-pol"`, `"cross-pol"`, `"single"` 등
- 시나리오: `"urban macro"`, `"indoor"`, `"rural"` 등

### 8.4 presets/ — 프리셋 프로필

미리 정의된 YAML 프로필로, 복잡한 설정을 한 번에 적용할 수 있다.

| 프리셋 | 설명 |
|--------|------|
| `su_mimo_2t2r_type1` | SU-MIMO 2안테나, Type-I 코드북 |
| `su_mimo_4t4r_type1` | SU-MIMO 4안테나, Type-I 코드북 |
| `mu_mimo_4t4r_type2` | MU-MIMO 4안테나, Type-II 코드북, 4 UE |
| `mu_mimo_8t8r_type2` | MU-MIMO 8안테나, Type-II 코드북, 4 UE |
| `multicell_2cell_4ue` | 2셀 각 4UE, ICI 15dB |
| `multicell_4cell_4ue` | 4셀 각 4UE, ICI 15dB |

프리셋 YAML 구조 (모든 프리셋에 bearer/qos 기본값 포함):
```yaml
name: "MU-MIMO 4T4R Type-II"
description: "4 antenna, dual-pol, Type-II codebook, 4 UE MU-MIMO"
tags: [mu-mimo, type2, 4t4r, dual-pol]
config:
  system:
    mu_mimo: true
    num_ues: 4
  antenna:
    gnb: { nx: 2, ny: 1, polarization: dual }
    ue: { nx: 1, ny: 1 }
  codebook:
    type: type2
    sub_type: typeII_PortSelection
  bearer:                         # ← 모든 프리셋에 포함
    enable_sdap: false
    drbs: 1
    um_on_default_drb: false
    drb_ciphering: true
    drb_integrity: false
  qos:                            # ← 모든 프리셋에 포함
    default_5qi: 9
    session_ambr_ul: "10Gbps"
    session_ambr_dl: "10Gbps"
    arp_priority: 15
```

## 9. gnb.conf 템플릿 엔진

### 9.1 원리

1. Core Emulator가 `master_config.yaml`을 로드
2. 파생값 계산 (`_derived()` 함수)
3. YAML 설정 + 파생값을 Jinja2 컨텍스트에 주입
4. `gnb.conf.j2` 렌더링 → 완성된 `gnb.conf` 텍스트 출력

### 9.2 주요 템플릿 변수

| 템플릿 변수 | 설정 원본 | gnb.conf 위치 |
|-------------|----------|---------------|
| `{{ n1 }}` | 자동 계산 | `pdsch_AntennaPorts_N1` |
| `{{ xp }}` | 자동 계산 | `pdsch_AntennaPorts_XP` |
| `{{ ue_ant }}` | 자동 계산 | `pusch_AntennaPorts` |
| `{{ gnb_ant }}` | 자동 계산 | `RUs.nb_tx` |
| `{{ csi_rs.periodicity }}` | YAML | `csirs_detailed_config.periodicity` |
| `{{ codebook.type }}` | YAML | `codebook_detailed_config` 분기 |

### 9.3 코드북 분기 예시

```jinja2
{% if codebook.type == "type1" %}
    codebook_type = "type1";
    sub_type = "{{ codebook.sub_type }}";
{% elif codebook.type == "type2" %}
    codebook_type = "type2";
    phase_alphabet_size = "{{ codebook.phase_alphabet_size }}";
    number_of_beams = {{ codebook.number_of_beams }};
{% endif %}
```

### 9.4 멀티셀 지원

`generate_gnb_configs.py`를 사용하여 셀별 gnb.conf를 배치 생성한다:
- 셀마다 고유한 PCI, GTP-U 포트, 셀 인덱스 부여
- `per_cell_overrides`로 셀별 주파수/대역 등 차별화 가능

## 10. Legacy TCP/JSON 명령어

v0.py Proxy와의 하위 호환을 위해 기존 TCP/JSON 인터페이스도 유지한다.

| 명령어 | 용도 | 응답 키 |
|--------|------|---------|
| `GET_CONFIG` | 전체 설정 반환 | `config` |
| `GET_GNB_CONF` | gnb.conf 텍스트 렌더링 | `gnb_conf` |
| `GET_LAUNCH_PARAMS` | launch_all.sh용 파라미터 | `params` |
| `GET_PROXY_PARAMS` | v0.py Proxy용 파라미터 | `params` |
| `UPDATE_PROXY` | Proxy 핫스왑 (런타임) | `proxy_params` |
| `UPDATE_GNB` | gNB 재설정 (재시작 필요) | `gnb_conf` |
| `SUBSCRIBE_PROXY` | Proxy 업데이트 구독 (push) | 이벤트 스트림 |
| `STATUS` | 서버 상태 조회 | `version`, `uptime_s` 등 |

프로토콜:
```
요청: {"cmd": "GET_CONFIG"}  + 줄바꿈(\n)
응답: {"status": "ok", "config": {...}}  + 줄바꿈(\n)
```

## 11. 사용법

### 11.1 기본 실행 흐름

```bash
# 1단계: Core Emulator 서버 시작
python3 core_emulator.py --config master_config.yaml

# 2단계: (선택) 프리셋 적용
curl -X POST http://localhost:7101/api/v1/apply-preset \
  -d '{"name": "mu_mimo_4t4r_type2"}'

# 3단계: 시스템 기동
sudo bash launch_multicell.sh -nc 1 -n 4 -ga 2 1 -ua 2 1 -pol dual
```

### 11.2 gnb.conf만 생성 (서버 없이)

```bash
python3 core_emulator.py --render-gnb-conf /tmp/my_gnb.conf
```

### 11.3 CLI 클라이언트 사용

```bash
# 전체 설정 확인
python3 core_cli.py config

# 서버 상태
python3 core_cli.py status

# 특정 섹션 확인
python3 core_cli.py get antenna
python3 core_cli.py get codebook

# gnb.conf 렌더링
python3 core_cli.py gnb-conf -o /tmp/gnb.conf
```

### 11.4 런타임 파라미터 변경

#### 핫스왑 (Proxy 즉시 반영, gNB 재시작 불필요)

```bash
python3 core_cli.py update channel.path_loss_dB=5.0
python3 core_cli.py update channel.snr_dB=20
python3 core_cli.py update channel.path_loss_dB=3.0 channel.speed=10
```

핫스왑 가능 파라미터:

| 파라미터 | 설명 |
|----------|------|
| `channel.path_loss_dB` | 경로 손실 (dB) |
| `channel.snr_dB` | 상대 AWGN SNR (dB) |
| `channel.noise_dBFS` | 절대 노이즈 플로어 (dBFS) |
| `channel.speed` | UE 이동 속도 (m/s) |

#### gNB 재설정 (managed restart 필요)

```bash
python3 core_cli.py update --restart-gnb codebook.type=type1
python3 core_cli.py update --restart-gnb antenna.gnb.polarization=single
```

## 12. 핫스왑 원리

```
core_cli.py update channel.snr_dB=20
        │
        ▼
  Core Emulator (TCP JSON)
    1. master_config 내부 상태 업데이트
    2. PROXY_UPDATE 이벤트 생성
    3. 구독된 모든 Proxy 클라이언트에 push
        │
        ▼
  Proxy (백그라운드 리스너 스레드)
    1. PROXY_UPDATE 이벤트 수신
    2. _apply_proxy_hotswap() 호출
    3. 글로벌 변수 atomic 업데이트
    4. 다음 슬롯부터 새 값 적용
```

## 13. 설정 구성 예시

### 13.1 4T4R Dual-Pol Type-II MU-MIMO (기본값)

```yaml
antenna:
  gnb: { nx: 2, ny: 1, polarization: dual }
  ue:  { nx: 2, ny: 1 }
codebook:
  type: type2
  sub_type: typeII_PortSelection
system:
  mu_mimo: true
```

결과: `N1=2, N2=1, XP=2 → 4포트, nb_tx=4, nb_rx=4`

### 13.2 2T2R Single-Pol Type-I SU-MIMO

```yaml
antenna:
  gnb: { nx: 2, ny: 1, polarization: single }
  ue:  { nx: 2, ny: 1 }
codebook:
  type: type1
  sub_type: typeI_SinglePanel
```

결과: `N1=2, N2=1, XP=1 → 2포트, nb_tx=2, nb_rx=2`

### 13.3 2 DRB + SDAP 활성화 + 영상 스트리밍 QoS

```yaml
bearer:
  enable_sdap: true
  drbs: 2                    # DRB1 (LCID=4) + DRB2 (LCID=5)
  um_on_default_drb: false   # AM 모드 (재전송 보장)
  drb_ciphering: true
  drb_integrity: false
qos:
  default_5qi: 6             # 영상 스트리밍 (Non-GBR, 300ms)
  session_ambr_dl: "5Gbps"
  session_ambr_ul: "2Gbps"
  arp_priority: 6
```

결과:
- gnb.conf: `enable_sdap=1`, `drbs=2`, `um_on_default_drb=0`
- SMF config: `5qi: 6`, `session_ambr_dl: "5Gbps"`
- UE당 RLC 채널: SRB0(LCID=0) + SRB1(1) + SRB2(2) + DRB1(4) + DRB2(5) = 5개

### 13.4 URLLC 시나리오 (저지연)

```yaml
bearer:
  enable_sdap: true
  drbs: 1
  um_on_default_drb: true    # UM 모드 (비재전송, 낮은 레이턴시)
  drb_integrity: true         # 무결성 보호 활성화
qos:
  default_5qi: 5             # IMS 시그널링 (Non-GBR, 100ms)
  session_ambr_dl: "10Gbps"
  arp_priority: 1            # 최고 우선순위
  arp_preempt_cap: "MAY_PREEMPT"
```

### 13.5 2셀 멀티셀 MU-MIMO

```yaml
multicell:
  enabled: true
  num_cells: 2
  ues_per_cell: 4
  ici_atten_dB: 15.0
system:
  mu_mimo: true
  num_ues: 4
antenna:
  gnb: { nx: 2, ny: 1, polarization: dual }
```

## 14. 실제 코어 교체 로드맵

Core Emulator는 실제 5GC NMS 교체를 단계적으로 지원하도록 설계되었다:

### Stage 1 (완료): 에뮬레이션 + PHY 설정

```
Core Emulator (FastAPI)  ←→  launch_multicell.sh / Proxy
  - 모든 PHY/채널 설정이 YAML + API로 관리됨
  - gnb.conf Jinja2 렌더링
  - 프리셋 + Intent 파싱 + 3GPP 검증
```

### Stage 2 (완료): Bearer/QoS + 코어 네트워크 제어

```
Core Emulator  →  gnb.conf (bearer: SDAP/DRB/RLC)
               →  CN5G config.yaml (SMF QoS 프로파일)
               →  MySQL DB (가입자별 QoS 업데이트)
  - gNB 측: enable_sdap, drbs, um_on_default_drb, drb_ciphering/integrity
  - 코어 측: 5QI, Session AMBR, ARP → SMF config 렌더링 + 재시작
  - DB 직접 접근: 가입자별 dnnConfigurations UPDATE
```

### Stage 3 (예정): 룰 기반 Intent + LLM 연동

```
LLM/AI  →  Intent Parser  →  Core Emulator  →  시스템
  - CSI-RS/SRS/RRC 등 복잡한 세팅 풀세트를 AI가 생성
  - 기본 세팅은 프리셋, 파라미터 미세 조정은 AI가 담당
```

### Stage 4 (미래): 실제 코어 연결

```
실제 5GC NMS REST API  ←→  동일한 클라이언트 코드
  - Core Emulator를 떼어내고 실제 코어의 REST API에 연결
  - URL과 인증만 변경하면 동작
```

## 15. 하위 호환

Core Emulator 없이도 기존 방식대로 동작한다:

```bash
# 기존 방식 (변경 없음)
sudo bash launch_multicell.sh -nc 1 -n 4 -ga 2 1 -ua 2 1 -pol dual

# Core Emulator 연동 방식
python3 core_emulator.py &
sudo bash launch_all.sh -c localhost:7100
```

## 16. Swagger UI (자동 API 문서)

FastAPI는 API를 정의하면 Swagger UI를 자동 생성한다.
서버 실행 후 브라우저에서 접속:

- **Swagger UI**: `http://localhost:7101/docs`
- **ReDoc**: `http://localhost:7101/redoc`

모든 엔드포인트를 시각적으로 탐색하고, 직접 API 호출을 테스트할 수 있다.

## 17. SDAP/DRB/RLC 제어 아키텍처

### 17.1 OAI에서의 SDAP/DRB/RLC 생성 흐름

UE가 PDU 세션을 수립하면 다음 순서로 베어러가 생성된다:

```
AMF/SMF (코어)                      gNB (RAN)
──────────                          ─────────
  PDU Session Establishment         
  Accept (NGAP)                     
    │                               
    │  QoS Flow: 5QI, ARP, AMBR    
    ▼                               
  PDUSessionResourceSetupRequest ──→ rrc_gNB_NGAP.c
                                       ::rrc_gNB_process_NGAP_PDUSESSION_SETUP_REQ()
                                       │
                                       ▼
                                    rrc_gNB_radio_bearers.c::generateDRB()
                                       │  SDAP config (QoS Flow → DRB 매핑)
                                       │  PDCP config (ciphering, integrity, SN size)
                                       │  DRB status = active
                                       ▼
                                    mac_rrc_dl_handler.c::nr_rlc_add_drb()
                                       │  RLC 엔티티 생성 (AM 또는 UM)
                                       │  LCID = DRB_ID + 3
                                       ▼
                                    RRCReconfiguration → UE에 전달
```

### 17.2 Core Emulator의 제어 포인트

```
                  Core Emulator
                       │
         ┌─────────────┼────────────────┐
         ▼             ▼                ▼
    ┌─────────┐   ┌─────────┐    ┌──────────┐
    │ gnb.conf│   │ SMF     │    │ MySQL    │
    │ (bearer)│   │ config  │    │ UDR DB   │
    └────┬────┘   └────┬────┘    └────┬─────┘
         │             │              │
         ▼             ▼              ▼
    nr-softmodem   oai-smf      SessionMgmt
    (gNB 기동 시)  (재시작 시)   SubscriptionData
         │             │              │
         │   ┌─────────┘              │
         │   │                        │
         ▼   ▼                        ▼
    NGAP: PDUSessionResourceSetupRequest
    (5QI, AMBR, ARP가 SMF→gNB로 전달)
         │
         ▼
    gNB에서 DRB 개수, SDAP, RLC 모드 결정
    (gnb.conf의 drbs, enable_sdap, um_on_default_drb)
```

### 17.3 UE당 생성되는 채널 요약

| DRB 수 | SRB | DRB | 총 RLC 채널 | LCID 범위 |
|--------|-----|-----|------------|----------|
| 1 (기본) | SRB0(0), SRB1(1), SRB2(2) | DRB1(4) | 4개 | 0,1,2,4 |
| 2 | SRB0(0), SRB1(1), SRB2(2) | DRB1(4), DRB2(5) | 5개 | 0,1,2,4,5 |
| 4 | SRB0(0), SRB1(1), SRB2(2) | DRB1~4(4~7) | 7개 | 0,1,2,4,5,6,7 |

> LCID 계산: SRB는 `LCID = SRB_ID`, DRB은 `LCID = DRB_ID + 3`
> (OAI: `gNB_scheduler_primitives.c::get_lcid_from_drbid()`)

## 18. 의존성

| 패키지 | 용도 |
|--------|------|
| `fastapi` | REST API 프레임워크 |
| `uvicorn` | ASGI 서버 (FastAPI 실행) |
| `pydantic` | 요청/응답 데이터 모델 검증 |
| `pyyaml` | YAML 설정 파일 파싱 |
| `jinja2` | gnb.conf / CN5G config 템플릿 렌더링 |

설치:
```bash
pip install fastapi uvicorn pydantic pyyaml jinja2
```

외부 런타임 의존성 (2단계 QoS 제어):

| 구성 요소 | 용도 |
|----------|------|
| Docker | SMF 컨테이너 재시작 (`docker restart oai-smf`) |
| MySQL (Docker) | 가입자 QoS 업데이트 (`docker exec mysql mysql ...`) |
| SMF config 볼륨 마운트 | CN5G config.yaml 파일 쓰기 |


시나리오	mean_xpr (dB)	stddev_xpr (dB)	의미
UMi-LOS
9
3
높은 XPR → 교차편파 간 분리 우수 → 더 나은 MIMO 성능
UMi-NLOS
8
3
중간 XPR
UMa-LOS
8
4
중간 XPR, 분산 큼
UMa-NLOS
7
4
낮은 XPR → 교차편파 간 간섭 증가

## 19. CsiNet 통합 (CSI Compression Sidecar)

### 19.0.1 개요

CsiNet은 딥러닝 기반 CSI 압축 네트워크로, Sionna 프록시가 생성한 채널 행렬 H를
인코더-디코더 파이프라인을 통해 압축/복원하고, 복원된 H_hat으로부터
PMI, CQI, RI를 계산하는 sidecar 모듈이다.

```
Sionna Proxy (v4.py / v4_multicell.py)
    │  DL 슬롯 처리 중 채널 H 캡처
    ▼
ChannelHook.capture(cell_idx, ue_idx, channels)
    │  CSI-RS period마다 콜백 트리거
    ▼
CsiNetInferenceEngine.encode_decode(H_freq, R_H, pdp)
    │  angular-delay 변환 → 인코더 → 디코더 → 역변환
    ▼
CSIInjector.process_channel(cell_idx, ue_idx, H_hat)
    │  SVD → PMI, CQI, RI, precoding weights 계산
    ▼
메모리 보관 (get_report로 조회 가능)
```

### 19.0.2 지원 모드

| 모드 | 설명 | 체크포인트 패턴 |
|------|------|----------------|
| `baseline` | 표준 CsiNet 오토인코더 | `csinet_{scenario}_gamma{γ}_best.weights.h5` |
| `conditioned` | FiLM 조건부 CsiNet + Statistics AE | `cond_csinet_{scenario}_gamma{γ}_best.weights.h5` + `stat_ae_{scenario}.weights.h5` |

### 19.0.3 master_config.yaml 설정

```yaml
csinet:
  enabled: false              # CsiNet sidecar 활성화
  mode: baseline              # baseline | conditioned
  compression_ratio: 0.25     # gamma (인코더 압축률)
  scenario: UMi_NLOS          # UMi_NLOS | UMi_LOS | UMa_NLOS | UMa_LOS
  checkpoint_dir: /workspace/csinet_checkpoints
  csi_rs_period: 20           # H 캡처 주기 (슬롯)
  csinet_path: /workspace/graduation/csinet   # CsiNet 모듈 루트
```

### 19.0.4 REST API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/csinet/config` | 현재 CsiNet 설정 조회 |
| `POST` | `/api/v1/csinet/config` | CsiNet 설정 업데이트 (프록시 재시작 필요) |
| `GET` | `/api/v1/csinet/env` | 프록시 docker exec용 환경변수 반환 |

### 19.0.5 사용 예시

```bash
# CsiNet 설정 조회
curl http://localhost:7101/api/v1/csinet/config

# CsiNet 활성화 (conditioned 모드)
curl -X POST http://localhost:7101/api/v1/csinet/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "mode": "conditioned", "scenario": "UMa_LOS"}'

# 환경변수 조회 (launch 스크립트가 사용)
curl http://localhost:7101/api/v1/csinet/env
```

### 19.0.6 환경변수 → 프록시 전달 흐름

```
master_config.yaml (csinet 섹션)
    → Core Emulator /api/v1/csinet/env
    → launch_all.sh / launch_multicell.sh:
        curl → CSINET_ENABLED, CSINET_MODE, CSINET_GAMMA, ...
    → docker exec -e CSINET_ENABLED=1 -e CSINET_MODE=conditioned ...
    → v4.py / v4_multicell.py: get_csinet_hook()
        → ChannelHook + CsiNetInferenceEngine + CSIInjector 초기화
```

### 19.0.7 CLI 사용 (launch_multicell.sh)

```bash
# Core Emulator 연동 (자동 CsiNet 설정 로드)
sudo bash launch_multicell.sh -nc 2 -n 4 -ga 2 1 -pol dual -c localhost:7100

# CLI에서 직접 CsiNet 활성화
sudo bash launch_multicell.sh -nc 2 -n 4 --csinet --csinet-mode conditioned
```

### 19.0.8 통합 모듈 구조

```
graduation/csinet/integration/
├── channel_hook.py      # ChannelHook: 채널 H 캡처 + 링버퍼 + 콜백
├── csinet_engine.py     # CsiNetInferenceEngine: 인코더/디코더 추론
├── csi_injection.py     # CSIInjector: H_hat → PMI/CQI/RI 계산
└── test_integration.py  # 단위 테스트
```

## 20. Traffic Emulator

### 20.1 개요

Traffic Emulator는 5GC 데이터 플레인을 통해 실제 DL/UL 트래픽을 생성하는 모듈이다.
`iperf3`을 이용하여 UPF → gNB → Sionna Proxy → UE 경로로 트래픽을 주입한다.
각 UE마다 독립적인 트래픽 프로파일을 설정할 수 있다.

### 20.2 지원 트래픽 패턴

| 패턴 | 설명 | 용도 |
|------|------|------|
| `full_buffer` | 최대 속도 연속 스트림 (TCP/UDP) | 링크 포화 테스트, 최대 처리량 측정 |
| `periodic` | 고정 비트레이트 UDP 스트림 | 일정 부하, CBR 트래픽 모델링 |
| `bursty` | ON/OFF 모델 (Poisson 분포 inter-arrival) | 웹 브라우징, 영상 스트리밍 등 실제 패턴 |

### 20.3 master_config.yaml 설정

```yaml
traffic:
  enabled: false                # true: UE attach 후 자동 트래픽 시작
  server_ip: "12.1.1.1"        # iperf3 서버 IP (UPF DN 인터페이스)
  default_profile:
    pattern: full_buffer        # full_buffer | periodic | bursty
    direction: dl               # dl | ul | bidir
    bitrate_mbps: 100           # 목표 비트레이트 (full_buffer TCP에서는 무시)
    protocol: udp               # tcp | udp
    duration_s: 0               # 0 = 무제한
    # Periodic 설정
    packet_size_bytes: 1400
    interval_ms: 1.0
    # Bursty (ON/OFF) 설정
    burst_on_ms: 100            # 평균 ON 기간 (지수 분포)
    burst_off_ms: 200           # 평균 OFF 간격 (Poisson)
    burst_rate_mbps: 50
    # QoS 마킹
    dscp: 0                     # 0=BE, 46=EF, 34=AF41
  # UE별 오버라이드:
  # ue_profiles:
  #   - ue_idx: 0
  #     pattern: bursty
  #   - ue_idx: 1
  #     pattern: periodic
  #     bitrate_mbps: 10
```

### 20.4 REST API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/v1/traffic/status` | 트래픽 상태 조회 (활성 UE 수, 프로파일) |
| `POST` | `/api/v1/traffic/start` | 트래픽 시작 (ue_idx 지정 또는 전체) |
| `POST` | `/api/v1/traffic/stop` | 트래픽 정지 (ue_idx 지정 또는 전체) |
| `POST` | `/api/v1/traffic/profile` | 기본 프로파일 업데이트 |
| `POST` | `/api/v1/traffic/profile/ue` | UE별 프로파일 설정 (속도 포함) |
| `GET` | `/api/v1/traffic/speeds` | UE별 이동 속도 조회 (km/h, m/s) |
| `GET` | `/api/v1/traffic/results/{ue_idx}` | UE별 iperf3 결과 조회 |

### 20.5 사용 예시

```bash
# 트래픽 상태 확인
curl http://localhost:7101/api/v1/traffic/status

# 전체 UE에 DL full-buffer 트래픽 시작
curl -X POST http://localhost:7101/api/v1/traffic/start

# 특정 UE만 시작
curl -X POST http://localhost:7101/api/v1/traffic/start \
  -H "Content-Type: application/json" \
  -d '{"ue_idx": 0}'

# 기본 프로파일을 bursty로 변경
curl -X POST http://localhost:7101/api/v1/traffic/profile \
  -H "Content-Type: application/json" \
  -d '{"pattern": "bursty", "burst_on_ms": 50, "burst_off_ms": 100}'

# UE 2에 periodic 10Mbps + 차량 속도 60km/h 설정
curl -X POST http://localhost:7101/api/v1/traffic/profile/ue \
  -H "Content-Type: application/json" \
  -d '{"ue_idx": 2, "profile": {"pattern": "periodic", "bitrate_mbps": 10, "speed_kmh": 60}}'

# UE별 이동 속도 조회
curl http://localhost:7101/api/v1/traffic/speeds

# 전체 트래픽 정지
curl -X POST http://localhost:7101/api/v1/traffic/stop
```

### 20.6 독립 실행 (CLI)

Core Emulator 없이 단독으로 트래픽을 생성할 수도 있다:

```bash
# Full-buffer DL (4 UE)
python3 traffic_emulator.py --pattern full_buffer --direction dl --ues 4

# Bursty 트래픽 (ON=50ms, OFF=100ms)
python3 traffic_emulator.py --pattern bursty --burst-on-ms 50 --burst-off-ms 100

# master_config.yaml에서 설정 로드
python3 traffic_emulator.py --config master_config.yaml

# Dry-run (명령어만 출력)
python3 traffic_emulator.py --pattern periodic --bitrate 10 --dry-run
```

### 20.7 UE 이동 속도 제어

각 UE의 이동 속도(`speed_kmh`)는 Sionna 채널 모델의 Doppler 효과에 직접 반영된다.
속도가 높을수록 채널 변화가 빨라져 CSI 피드백의 aging 효과가 커진다.

| 프리셋 레이블 | 속도 (km/h) | 속도 (m/s) | 시나리오 |
|-------------|-----------|----------|---------|
| static | 0 | 0 | 고정 UE (실내, IoT) |
| pedestrian | 3 | 0.83 | 보행자 |
| vehicular_urban | 30 | 8.33 | 도심 차량 |
| vehicular_highway | 120 | 33.33 | 고속도로 차량 |
| hst | 350 | 97.22 | 고속열차 (HST) |

**Proxy 연동 (`v4.py`):**

```bash
# CLI에서 직접 UE별 속도 지정
python3 v4.py --num-ues 4 --ue-speeds 3,30,120,3
```

Core Emulator 연동 시 `launch_all.sh`가 자동으로 `/api/v1/traffic/speeds`에서
UE별 속도를 가져와 `--ue-speeds` 인자로 Proxy에 전달한다.

**Sionna 내부 처리:**

```
TrafficProfile.speed_kmh = 60        (km/h)
    → speed_ms = 60 / 3.6 = 16.67   (m/s)
    → velocities tensor [batch, N_UE, 3]  (UE별 다른 속도)
    → Sionna CIR: Doppler shift ∝ speed × cos(angle) × f_c / c
```

### 20.8 트래픽 흐름 아키텍처

```
┌──────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────┐
│ iperf3 서버   │ GTP │   gNB    │     │ Sionna Proxy│     │  UE  │
│ (UPF DN)     │────→│ (DL 스케줄)│────→│ (채널 적용)  │────→│      │
│ 12.1.1.1     │     │          │     │             │     │      │
└──────────────┘     └──────────┘     └─────────────┘     └──────┘
       ▲                                                       │
       │                    Core Emulator                      │
       │              /api/v1/traffic/start                    │
       │              /api/v1/traffic/profile                  │
       └───── 제어 ─── traffic_emulator.py ── iperf3 클라이언트 ┘
```
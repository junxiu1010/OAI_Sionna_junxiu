# OAI + Sionna Channel Simulation Twin — 개발 현황 기술 문서

> **작성 일자**: 2026-03-18
> **프로젝트**: OAI 기반 RAN 디지털 트윈 구축 및 채널 통계 조건부 AI/ML CSI 압축

---

## 1. 전체 아키텍처 개요

### 1.1 시스템 구성도

```
┌────────────────────────────────────────────────────────────────────┐
│                     Host Machine (GPU Server)                      │
│                                                                    │
│  ┌──────────────┐     GPU IPC (SHM + CUDA)     ┌────────────────┐ │
│  │  OAI gNB     │◄══════════════════════════════►│                │ │
│  │  (C/C++)     │  dl_tx / ul_rx                │  Sionna Channel│ │
│  │  - PHY/MAC   │                               │  Proxy (Python)│ │
│  │  - RRC       │  ┌──────────────┐             │  - v4_multi    │ │
│  │  - scheduler │  │  OAI UE #0~3 │  dl_rx/ul_tx│    cell.py     │ │
│  └──────────────┘  │  (C/C++)     │◄════════════►│  - GPUSlot     │ │
│                    │  - PHY/MAC   │             │    Pipeline    │ │
│                    │  - CSI report│             │  - Sionna 3GPP │ │
│                    └──────────────┘             │    ch. model   │ │
│                                                 │  - CsiNet Hook │ │
│  ┌──────────────────────────────────────┐       └────────┬───────┘ │
│  │  Core Emulator (FastAPI, port 7101)  │                │         │
│  │  - master_config.yaml                │  REST/TCP ─────┘         │
│  │  - gnb.conf Jinja2 render            │                          │
│  │  - param_validator (3GPP 준수)       │                          │
│  │  - preset system                     │                          │
│  │  - traffic emulator                  │                          │
│  └──────────────────────────────────────┘                          │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  CsiNet Engine (TensorFlow/GPU)                              │  │
│  │  - Statistics AE → Conditioning Vector (48-dim)              │  │
│  │  - CsiNet Baseline / Conditioned CsiNet                     │  │
│  │  - Channel Hook: H 캡처 → 압축 → CSI Report 생성            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 사용 오픈소스 및 특징

| 오픈소스 | 버전/특징 | 활용 방식 |
|---------|----------|----------|
| **OpenAirInterface (OAI)** | 5G NR gNB/UE, C/C++, 실제 3GPP 프로토콜 스택 (RRC/MAC/PHY) 구현 | rfsimulator 확장(GPU IPC), PRACH/HARQ 타이밍 보정, 4포트 RI/PMI 구현 |
| **NVIDIA Sionna** | v1.0.2, TensorFlow 기반, 3GPP TR 38.901 채널 모델 GPU 가속 | 채널 계수 생성, OFDM 처리, 학습용 데이터셋 생성, 채널 통계 추출 |
| **CuPy** | CUDA Python, GPU 배열 연산 | I/Q 샘플 실시간 처리, 주파수 도메인 채널 적용 |
| **TensorFlow** | 2.17.0-gpu | CsiNet/Conditioned CsiNet/Statistics AE 학습 및 추론 |
| **FastAPI** | Python 비동기 웹 프레임워크 | Core Emulator REST API 서버 |

### 1.3 컨테이너 환경

```yaml
# docker-compose.yml (루트)
services:
  oai_sionna_proxy:
    build: ./vRAN_Socket
    network_mode: host        # OAI와 동일 네트워크
    ipc: host                 # GPU IPC 공유메모리 접근
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy:/workspace/proxy
      - ./openairinterface5g_whan:/workspace/oai
      - ./graduation:/workspace/graduation
      - csinet_checkpoints:/workspace/csinet_checkpoints
      - /tmp/oai_gpu_ipc:/tmp/oai_gpu_ipc   # GPU IPC SHM
```

**Dockerfile 기반**: `tensorflow:2.17.0-gpu-jupyter` + `sionna==1.0.2` + `cupy-cuda12x` + `numpy==1.26.4`

---

## 2. 핵심 모듈별 상세 구현

### 2.1 OAI 수정 사항 (`openairinterface5g_whan/`)

#### 2.1.1 rfsimulator → GPU IPC 확장

OAI의 기존 rfsimulator는 TCP 소켓으로 gNB-UE 간 I/Q 샘플을 교환합니다. 이를 **GPU 공유 메모리 기반 순환 버퍼(GPU IPC V7)**로 확장하여 지연 최소화 및 대역폭 확보:

**핵심 구조 (`radio/rfsimulator/gpu_ipc_v7.h`)**:
- 4개 GPU 순환 버퍼: `dl_tx`(gNB→Proxy), `dl_rx`(Proxy→UE), `ul_tx`(UE→Proxy), `ul_rx`(Proxy→gNB)
- 버퍼 크기: `CIR_TIME = 4,608,000` 샘플 (75ms, 150 슬롯)
- 샘플 형식: `c16_t` (16-bit I + 16-bit Q = 4바이트)
- 동기화: `/tmp/oai_gpu_ipc/` SHM + futex 시퀀스 카운터
- 타임스탬프 인덱싱: `offset = (ts × nbAnt) % buffer_cir_size`

**I/Q 프로토콜 (TCP 경로, `simulator.c`)**:
```c
// 헤더 구조 (common_lib.h)
typedef struct {
    uint32_t size;       // 안테나당 샘플 수
    uint32_t nbAnt;      // 안테나 수
    uint64_t timestamp;  // 첫 샘플 타임스탬프
    uint32_t option_value;
    uint32_t option_flag;
} samplesBlockHeader_t;

// 전송: 헤더 → [시간×안테나] interleaved 샘플
// tmpSamples[sample_idx][ant_idx] 형태로 재배열 후 write
```

#### 2.1.2 타이밍/동기 보정

GPU IPC는 FIFO 없이 타임스탬프 인덱싱하므로, OAI의 슬롯 처리 타이밍과 정합이 필요:

- **UL 슬롯 제로 패딩** (`executables/nr-gnb.c`): UL-only 슬롯에서 gNB가 DL TX를 기록하지 않으면 UE가 다음 DL 데이터를 잘못 읽음 → 제로 I/Q를 기록하여 슬롯 단위 정합 유지
- **PRACH 스큐 완화** (`openair1/PHY/NR_TRANSPORT/nr_prach.c`): 비실시간 프록시에서 PRACH 타이밍이 어긋날 수 있어, Assert 대신 skip + stale 엔트리 GC
- **HARQ 조기 수신** (`openair2/.../gNB_scheduler_uci.c`): 프록시 지연으로 인해 HARQ ACK가 8~11슬롯 일찍 도착하는 현상 수용

#### 2.1.3 4×4 MIMO RI/PMI 구현

OAI 원본은 2포트까지만 지원하므로 직접 구현:

- `nr_csi_rs_ri_estimation_4x4()`: Rank 1~4에서의 RI 추정
- `nr_csi_rs_pmi_estimation_4port()`: 4포트 PMI 후보 탐색
- 1포트 SINR 46dB 하드코딩 버그 수정 → 실제 채널 품질 반영

---

### 2.2 Sionna 채널 프록시 (`vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/`)

#### 2.2.1 엔진 코드 구조

```
v4_multicell.py    ← 멀티셀 메인 (GPU IPC + Sionna 채널)
v4.py              ← 단일셀/OFDM 파이프라인, GPUSlotPipeline
core_emulator.py   ← FastAPI 설정 서버
param_validator.py ← 3GPP 규격 검증
traffic_emulator/  ← 트래픽 생성기
templates/         ← gnb.conf Jinja2 템플릿
presets/           ← 시나리오 프리셋 YAML
```

#### 2.2.2 채널 적용 파이프라인

**DL 경로** (gNB → 채널 → UE):

```python
# v4_multicell.py :: _dl_apply_channel_slot()
def _dl_apply_channel_slot(self, cell, ue_idx, ts, nsamps):
    # 1. gNB DL TX GPU 버퍼에서 I/Q 읽기
    arr_in = gnb.circ_read(gnb.gpu_dl_tx_ptr, ts, nsamps,
                           gnb.dl_tx_nbAnt, gnb.dl_tx_cir_size, cp.int16)

    # 2. Sionna 채널 계수 가져오기 (주파수 도메인)
    #    channels shape: (N_SYM, n_rx, n_tx, FFT_SIZE) complex
    channels, n_held = cell.channel_buffers[ue_idx].get_batch_view(N_SYM)

    # 3. CsiNet Hook: H 캡처 (CSINET_ENABLED=1일 때)
    csinet_hook = get_csinet_hook()
    if csinet_hook is not None and csinet_hook.enabled:
        csinet_hook.capture(cell.cell_idx, ue_idx, channels)

    # 4. GPU 슬롯 파이프라인으로 채널 적용
    #    I/Q(시간) → FFT → H 곱셈 → IFFT → I/Q(시간)
    cell.pipelines_dl[ue_idx].process_slot_ipc(
        arr_in, channels, dl_gain, None, False, arr_out, None)

    # 5. UE DL RX GPU 버퍼에 결과 기록
    ue.circ_write(ue.gpu_dl_rx_ptr, ts, nsamps, ...)
```

**UL 경로** (UE → 채널 → gNB):
- 동일 구조이나 채널 행렬 `transpose(0, 2, 1, 3)` (TX/RX 역할 교환)
- 멀티UE: 각 UE의 UL 신호를 채널 적용 후 **합산(superposition)**하여 gNB에 기록

**OFDM 수치 (NR numerology=1, 30 kHz SCS)**:
```python
FFT_SIZE = 2048
CP0 = 176          # 첫 심볼 CP
CP1 = 144          # 나머지 심볼 CP
N_SYM = 14         # 슬롯당 OFDM 심볼
SYMBOL_SIZES = [CP0+FFT_SIZE] + [CP1+FFT_SIZE]*13
# total_cpx = 30720 (= 1 슬롯 시간 도메인 샘플)
carrier_frequency = 3.5e9  # FR1 Band 78
```

**채널 행렬 형식**:
- DL: `(N_SYM=14, n_rx, n_tx, FFT_SIZE=2048)` complex64
- UL: `transpose(0, 2, 1, 3)` → `(N_SYM, n_tx_ue, n_rx_gnb, FFT_SIZE)`

#### 2.2.3 멀티셀 ICI (Inter-Cell Interference)

```python
# _add_dl_ici(): 타 셀 gNB DL을 감쇠하여 UE RX에 가산
for other_cell in cells:
    if other_cell.cell_idx != target_cell.cell_idx:
        ici_signal = apply_channel(other_gnb_tx, ici_channel, ici_gain)
        ue_rx += ici_signal  # 간섭 중첩
```

---

### 2.3 Core Emulator (`core_emulator.py`)

#### 2.3.1 설계 철학

기존 OAI 시뮬레이션의 설정이 gnb.conf, CLI, 프록시 상수의 3곳에 분산되어 실험 재현성이 낮은 문제를 해결하기 위해 **단일 진실 소스(Single Source of Truth)** 패턴 적용.

#### 2.3.2 구현 상세

```python
# core_emulator.py
# REST API (port 7101) + Legacy TCP JSON (port 7100)

# 12개 설정 섹션
master_config.yaml:
  system:     # 대역폭, SCS, TDD 패턴
  antenna:    # N1, N2, XP, 안테나 파생값 자동계산
  codebook:   # Type 1/2, L빔, 위상 비트
  csi_rs:     # 포트, 주기, CDM, 밀도
  srs:        # SRS 설정
  channel:    # 시나리오(UMa/UMi), 경로손실, SNR
  carrier:    # 캐리어 주파수, ARFCN
  multicell:  # 셀 수, ICI 설정
  bearer:     # DRB, QoS
  qos:        # 5QI, AMBR
  network:    # IP 설정
  traffic:    # 트래픽 패턴, UE 속도

# API 엔드포인트
POST /api/v1/config          # 파라미터 직접 변경
POST /api/v1/apply-preset    # 프리셋 원클릭 적용
POST /api/v1/apply           # gnb.conf 렌더 + 재시작
POST /api/v1/intent          # 자연어 → 설정 변환
GET  /api/v1/status          # 현재 설정 조회
POST /api/v1/csinet/config   # CsiNet 모드 설정
```

**프리셋 시스템**: `presets/` 디렉터리에 YAML로 정의
- `type1_su_mimo.yaml`: Type 1 codebook + SU 스케줄링
- `type2_su_mimo.yaml`: Type 2 codebook + SU 스케줄링
- `type2_mu_mimo.yaml`: Type 2 codebook + MU 스케줄링

**파라미터 검증 (`param_validator.py`)**: TS 38.331/38.214 기반으로 CSI-RS/SRS/안테나/캐리어 파라미터 상호 의존성 검증

**gnb.conf 자동 렌더링**: Jinja2 템플릿으로 `master_config.yaml` → OAI `gnb.conf` 자동 변환

#### 2.3.3 Traffic Emulator

UE 속도, QoS 프로파일, 트래픽 패턴을 Core Emulator에서 통합 설정하여 프록시에 전달.

---

### 2.4 CsiNet 엔진 (`graduation/csinet/`)

#### 2.4.1 모델 아키텍처

**CsiNet Baseline** (`models/csinet.py`):
```
입력: (B, 2, Nt=4, Nc'=32) — 실수/허수 2채널

Encoder (UE 측):
  Conv2D(16, 3×3) → BN → LeakyReLU(0.3)
  Conv2D(8, 3×3)  → BN → LeakyReLU(0.3)
  Conv2D(4, 3×3)  → BN → LeakyReLU(0.3)
  Flatten → Dense(M)        ← M = 2×Nt×Nc'×γ

Decoder (gNB 측):
  Dense(2×Nt×Nc') → Reshape(Nt, Nc', 2)
  RefineNetBlock ×2: Conv(8,3×3)→BN→LeakyReLU→Conv(8,3×3)→BN + 잔차
  Conv2D(2, 3×3)  ← 출력

압축률 γ ∈ {1/4, 1/8, 1/16, 1/32}
  γ=1/4 → M=64, γ=1/8 → M=32, γ=1/16 → M=16, γ=1/32 → M=8
```

**Statistics Autoencoder** (`models/stat_autoencoder.py`):
```
Covariance 경로:
  입력: vec(R_H) 상삼각 실수화 → dim=72
  Dense(128) → Dense(64) → Dense(cov_latent=32)  ← 인코더
  Dense(64) → Dense(128) → Dense(72)             ← 디코더

PDP 경로:
  입력: PDP vector → dim=72
  Dense(64) → Dense(pdp_latent=16)  ← 인코더
  Dense(64) → Dense(72, softplus)   ← 디코더

Conditioning vector = [z_cov(32) ∥ z_pdp(16)] = 48차원
```

**Conditioned CsiNet** (`models/conditioned_csinet.py`):
```
CsiNet + FiLM (Feature-wise Linear Modulation)

FiLM Layer:
  gamma_net: Dense(cond_dim → feature_ch), init: kernel=0, bias=1
  beta_net:  Dense(cond_dim → feature_ch), init: kernel=0, bias=0
  output = gamma * feature + beta
  → 초기값이 identity transform이므로 baseline 가중치와 호환

Encoder: 각 Conv-BN 뒤에 FiLM 삽입
Decoder: ConditionedRefineBlock (conv-BN-FiLM-ReLU-conv-BN-FiLM + 잔차)

호출: model([x, cond_vector], training=...)
```

#### 2.4.2 학습 파이프라인

**데이터 생성** (`data/generate_dataset.py`):
```python
# Sionna 3GPP 채널 생성
안테나: gNB PanelArray 2×1 dual-pol (Nt=4), UE 1×1 dual-pol (Nr=2)
ResourceGrid: fft_size=128, SCS=30kHz, 활성 SC=72
위치: 1000개 (셀 반경 200m 내 균일 분포)
위치당 샘플: stat=50, train=60, val=5, test=5

# 출력 (dataset_{scenario}.h5)
H_*:      (N, Nr=2, Nt=4, Nsc=72) complex64
R_H:      (N_loc, 8, 8) complex64    # 위치별 공분산
PDP:      (N_loc, 72) float32         # 위치별 PDP
loc_idx_*: 샘플→위치 매핑 인덱스
```

**전처리** (`data/preprocess.py`):
```
H(Nr,Nt,Nsc) → Rx 평균 → 공간 DFT(각도) + IFFT(지연)
  → 앞 Nc'=32 탭 절단 → 샘플별 전력 정규화
  → 실수/허수 분리 → X: (N, 2, Nt=4, 32) float32

출력: preprocessed_{scenario}.h5
```

**3단계 학습**:

| 단계 | 대상 | Epochs | 학습률 | 설명 |
|------|------|--------|--------|------|
| Stage 1 | Statistics AE | 200 | 1e-3 (Adam) | R_H+PDP 재구성 (MSE) |
| Stage 2 Phase 1 | Cond CsiNet FiLM만 | 150 | 1e-3, grad clip 1.0 | FiLM 레이어 워밍업 |
| Stage 2 Phase 2 | Cond CsiNet 전체 | 600 (early stop 80) | 베이스 0.1×LR, FiLM 0.5×LR (Cosine) | 차등 학습률 미세조정 |

**Baseline 가중치 이전**: CsiNet → ConditionedCsiNet으로 encoder conv/BN/dense, decoder dense/conv_out/refine_blocks의 가중치 복사 (FiLM 제외)

#### 2.4.3 프록시 연동 (Channel Hook)

```python
# v4_multicell.py 상단
if os.environ.get("CSINET_ENABLED") == "1":
    csinet_hook = ChannelHook(...)      # H 캡처
    csinet_engine = CsiNetInferenceEngine(...)  # 추론
    csi_injector = CSIInjector(...)     # CSI 보고 생성

# 채널 적용 시점에서 Hook
csinet_hook.capture(cell_idx, ue_idx, channels)
# channels: (14, n_rx, n_tx, 2048) → csi_rs_period마다 캡처

# 추론: H → 압축(encoder) → 복원(decoder) → CSI Report
# CSI Report: RI, PMI, CQI + precoding weights (ZF)
```

#### 2.4.4 Differential Encoding (`integration/differential_cond.py`)

```python
# Two-timescale 피드백:
#   빠른 채널: 매 슬롯 codeword (M×32 bits)
#   느린 통계: 변화가 threshold 초과 시에만 conditioning vector 전송

class DifferentialConditioner:
    threshold = 0.01  # ||Δc||²/||c_prev||² 임계값
    max_stale_slots = 100  # 최대 미전송 슬롯 수

    def should_update(self, c_new, c_prev):
        delta = c_new - c_prev
        relative_change = ||delta||² / ||c_prev||²
        return relative_change > threshold or stale > max_stale_slots

# 오버헤드 모델:
#   업데이트 시: cond_dim × bits_per_float (= 48×32 = 1536 bits)
#   미업데이트 시: 0 bits (이전 값 재사용)
```

---

## 3. 데이터 흐름 전체 경로

### 3.1 실시간 시뮬레이션 경로

```
OAI gNB PHY (DL TX)
  │ c16_t I/Q, 타임스탬프 포함
  │ GPU 순환버퍼 dl_tx 기록
  ▼
Sionna Channel Proxy
  │ ① circ_read: gNB dl_tx에서 슬롯(30720 cpx) 읽기
  │ ② Sionna 채널: H(14, Nr, Nt, 2048) complex
  │ ③ process_slot_ipc:
  │    I/Q → OFDM 심볼별 FFT → H 곱셈(주파수 도메인) → IFFT → I/Q
  │ ④ [옵션] CsiNet Hook: H 캡처 → 압축/복원 → CSI Report
  │ ⑤ circ_write: UE dl_rx에 결과 기록
  ▼
OAI UE PHY (DL RX)
  │ CSI-RS 수신 → 채널 추정 → RI/PMI/CQI 보고
  ▼
OAI gNB MAC
  │ CSI Report 수신 → MCS 결정 → 스케줄링 → 프리코딩
  ▼
[반복]
```

### 3.2 AI/ML 학습 데이터 경로

```
Sionna 채널 모델 (UMa/UMi)
  │ 1000 위치 × 60 채널 실현/위치
  ▼
generate_dataset.py
  │ H: (N, Nr, Nt, Nsc) complex64
  │ R_H: 위치별 공분산, PDP: 위치별 전력 지연 프로파일
  ▼
preprocess.py
  │ 각도-지연 도메인 변환 + 절단(32탭) + 정규화
  │ X: (N, 2, 4, 32) float32
  ▼
train_csinet.py          → CsiNet Baseline 체크포인트
train_conditioned.py     → Statistics AE + Conditioned CsiNet 체크포인트
  ▼
evaluate_csinet.py       → csinet_evaluation.json (NMSE, cos_sim)
eval_offline_abc.py      → 오프라인 성능 그래프 (Part A/B1/B2)
run_e2e_with_measured_nmse.py → E2E 시스템 성능 그래프
```

---

## 4. 실험 및 평가 체계

### 4.1 오프라인 평가

| 실험 | 목적 | 스크립트 | 산출물 |
|------|------|---------|--------|
| NMSE vs 압축률 | γ별 Baseline/Conditioned/Type2 비교 | `evaluate_csinet.py` + `run_e2e_with_measured_nmse.py` | `csinet_nmse_comparison.png` |
| Part A: H Freshness | 동일 오버헤드에서 H 갱신 빈도 증가의 NMSE 이득 | `eval_offline_abc.py` | `part_a_freshness_*.png` |
| Part B1: 통계 추정 정확도 | 샘플 수 N에 따른 R_H/PDP 추정 오차 | `eval_offline_abc.py` | `part_b1_estimability.png` |
| Part B2: 시간 안정성 | Δt에 따른 R_H 변화량 측정 | `eval_offline_abc.py` | `part_b2_temporal_stability.png` |
| Differential 평가 | 임계값별 오버헤드 절감 vs NMSE 열화 | `eval_differential.py` | `diff_eval_*.png` |

### 4.2 E2E 시스템 평가

```
5가지 모드 비교 (Monte Carlo 시뮬레이션):
  1. Type 1 SU-MIMO    — 단일 DFT 빔 + SU
  2. Type 2 SU-MIMO    — L빔 결합 + SU
  3. Type 2 MU-MIMO    — L빔 결합 + MU
  4. CsiNet MU-MIMO    — AI/ML baseline + MU
  5. Cond-CsiNet MU-MIMO — 통계조건부 + MU

시나리오: UMi-LOS, UMi-NLOS, UMa-LOS, UMa-NLOS
UE 수: 1, 2, 4, 8, 16
지표: BLER, MCS, Cell Throughput (Mbps)
```

**NMSE → SNR 패널티 변환**:
```python
CSI_loss_dB = -10 * log10(1 - 10^(NMSE_dB / 10))
# NMSE=-18 dB → loss=0.07 dB (우수한 CSI)
# NMSE=-5 dB  → loss=1.55 dB (열악한 CSI)
```

### 4.3 최신 실험 결과 (2026-03-18 기준)

**NMSE 비교 (γ=1/4)**:

| 시나리오 | Type 2 | CsiNet BL | Cond CsiNet | BL→Cond 개선 |
|---------|--------|-----------|-------------|-------------|
| UMi-LOS | -3.7 dB | -14.88 dB | **-18.01 dB** | +3.1 dB |
| UMi-NLOS | -4.4 dB | -12.63 dB | **-13.97 dB** | +1.3 dB |
| UMa-NLOS | -4.2 dB | -5.59 dB | **-9.29 dB** | +3.7 dB |

모든 시나리오/압축률에서 **Cond CsiNet > Baseline > Type 2** 순서 일관.
NLOS 시나리오에서 conditioning 효과가 가장 두드러짐 (채널 통계 구조가 복잡할수록 side information의 가치 증가).

---

## 5. 에이전트화 및 확장 방향

### 5.1 현재 엔진 형태

| 엔진 | 형태 | 인터페이스 |
|------|------|-----------|
| Sionna 채널 프록시 | Python 프로세스 (GPU) | GPU IPC SHM + 환경변수 |
| Core Emulator | FastAPI 서버 | REST API (port 7101) + TCP JSON (7100) |
| CsiNet 추론 | TF SavedModel / weights.h5 | Python 함수 호출 (Channel Hook) |
| OAI gNB/UE | C 바이너리 프로세스 | gnb.conf + CLI 옵션 |

### 5.2 에이전트화를 위한 현재 작업

1. **Core Emulator의 NMS-스타일 API**: 실제 5GC NMS와 동일한 REST 구조로 설계되어, 클라이언트 변경 없이 실 코어로 교체 가능
2. **Intent API** (`POST /api/v1/intent`): 자연어 의도 → 파라미터 설정 자동 변환 (LLM 연동 준비)
3. **프리셋 시스템**: 복잡한 설정 조합을 단일 프리셋으로 추상화하여 자동화 파이프라인에서 호출 가능
4. **CsiNet Hook의 모듈화**: `CSINET_ENABLED` 환경변수로 on/off, 모델 체크포인트 경로 외부 주입, 추론 결과 JSON 출력

### 5.3 향후 에이전트화 방향

- **자동 실험 오케스트레이션**: Core Emulator API로 시나리오 자동 전환 → 프록시 자동 재설정 → 결과 수집 → 분석 파이프라인
- **온라인 학습 루프**: 프록시에서 실시간 캡처한 채널로 CsiNet 모델 점진적 업데이트
- **멀티 에이전트**: gNB/UE/프록시를 독립 에이전트로 분리하여 분산 실행 및 스케일아웃

---

## 6. 코드 디렉터리 구조

```
oai_sionna_junxiu/
├── docker-compose.yml                    # 컨테이너 오케스트레이션
├── openairinterface5g_whan/              # OAI 수정 소스
│   ├── radio/rfsimulator/
│   │   ├── simulator.c                   # TCP I/Q + GPU IPC 분기
│   │   ├── gpu_ipc_v7.h/c               # GPU 순환버퍼 V7
│   │   └── gpu_ipc_v2~v6.*              # 이전 버전들
│   ├── executables/nr-gnb.c              # 타이밍 보정
│   ├── openair1/PHY/                     # PHY 수정 (PRACH, RI/PMI)
│   └── openair2/                         # MAC 수정 (HARQ, 스케줄러)
│
├── vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/
│   ├── v4_multicell.py                   # 멀티셀 프록시 메인
│   ├── v4.py                             # OFDM 파이프라인, GPUSlotPipeline
│   ├── core_emulator.py                  # FastAPI 설정 서버
│   ├── param_validator.py                # 3GPP 규격 검증
│   ├── master_config.yaml                # 통합 설정 파일
│   ├── templates/                        # gnb.conf Jinja2 템플릿
│   ├── presets/                          # 시나리오 프리셋
│   └── traffic_emulator/                 # 트래픽 생성기
│
├── graduation/
│   ├── csinet/
│   │   ├── models/
│   │   │   ├── csinet.py                 # CsiNet Baseline
│   │   │   ├── conditioned_csinet.py     # FiLM 조건부 CsiNet
│   │   │   └── stat_autoencoder.py       # Statistics AE
│   │   ├── data/
│   │   │   ├── generate_dataset.py       # Sionna → H5 데이터셋
│   │   │   └── preprocess.py             # 각도-지연 도메인 변환
│   │   ├── integration/
│   │   │   ├── csinet_engine.py          # TF 추론 엔진
│   │   │   ├── channel_hook.py           # 프록시 H 캡처
│   │   │   ├── csi_injection.py          # CSI Report 생성
│   │   │   └── differential_cond.py      # Differential encoding
│   │   ├── e2e_evaluation/
│   │   │   ├── eval_offline_abc.py       # 오프라인 A/B1/B2
│   │   │   ├── run_e2e_with_measured_nmse.py  # E2E 실험
│   │   │   └── plot_e2e_results.py       # 그래프 생성
│   │   ├── train_csinet.py               # Baseline 학습
│   │   ├── train_conditioned.py          # 조건부 모델 학습
│   │   └── evaluate_csinet.py            # NMSE 평가
│   │
│   ├── experiments/
│   │   ├── simulate_calibrated_performance.py  # 보정 MC 시뮬
│   │   └── plot_*.py                     # 다양한 그래프 스크립트
│   │
│   └── thesis_structure.md               # 논문 구조
│
└── csinet_checkpoints/                   # 학습된 모델 체크포인트
    ├── stat_ae_{scenario}.weights.h5
    ├── csinet_{scenario}_gamma*.weights.h5
    └── cond_csinet_{scenario}_gamma*.weights.h5
```

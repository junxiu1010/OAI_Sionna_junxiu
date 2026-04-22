# CsiNet MU-MIMO Simulator 사용 가이드

> **목표**: Proposed Conditioned CsiNet을 사용하여 1-cell 4UE 4T4R MU-MIMO 환경에서 OAI-Sionna Simulator를 실행하는 방법

---

## 목차

1. [사전 조건](#1-사전-조건)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [방법 A: Core Emulator 사용](#3-방법-a-core-emulator-사용)
4. [방법 B: Core Emulator 없이 직접 실행](#4-방법-b-core-emulator-없이-직접-실행)
5. [CsiNet 설정 상세](#5-csinet-설정-상세)
6. [실행 확인 및 모니터링](#6-실행-확인-및-모니터링)
7. [FAQ / 문제 해결](#7-faq--문제-해결)

---

## 1. 사전 조건

### 1.1 Docker 컨테이너

`docker-compose.yml`에 다음 볼륨이 마운트되어 있어야 함:

```yaml
volumes:
  - ./vRAN_Socket:/workspace/vRAN_Socket
  - ./openairinterface5g_whan:/workspace/openairinterface5g_whan
  - ./graduation:/workspace/graduation           # CsiNet 모델 코드
  - ./csinet_checkpoints:/workspace/csinet_checkpoints  # 학습된 가중치
  - /tmp/oai_gpu_ipc:/tmp/oai_gpu_ipc
```

컨테이너 상태 확인:

```bash
docker ps | grep oai_sionna_proxy
# 만약 중지 상태라면:
cd ~/oai_sionna_junxiu && docker compose up -d
```

### 1.2 체크포인트 파일

`csinet_checkpoints/` 디렉토리에 다음 파일이 존재해야 함:

| 파일 패턴 | 설명 |
|-----------|------|
| `csinet_{scenario}_gamma{ratio}_best.weights.h5` | Baseline CsiNet |
| `cond_csinet_{scenario}_gamma{ratio}_best.weights.h5` | Conditioned CsiNet |
| `stat_ae_{scenario}.weights.h5` | Statistics Autoencoder (conditioned 모드 전용) |

scenario = `UMi_LOS`, `UMi_NLOS`, `UMa_NLOS`
ratio = `0.2500`, `0.1250`, `0.0625`, `0.0312`

확인 명령:

```bash
ls ~/oai_sionna_junxiu/csinet_checkpoints/*.h5 | wc -l
# 27개 이상이면 정상
```

### 1.3 4T4R 안테나 구성

4T4R = gNB 4 안테나 + UE 4 안테나. 다음 설정으로 구현:

| 파라미터 | 값 | 의미 |
|----------|---|------|
| gNB Nx=2, Ny=1, pol=dual | 2×1×2=**4 포트** | gNB 4T |
| UE Nx=2, Ny=1 | 2×1×2=**4 포트** | UE 4R (dual-pol 자동) |

### 1.4 5G Core Network

OAI 5GC (AMF, SMF, UPF 등)가 실행 중이어야 함. 네트워크 IP:

| 컴포넌트 | 기본 IP |
|----------|---------|
| AMF | 192.168.70.132 |
| gNB | 192.168.70.129 |
| UPF DN | 12.1.1.1 |

---

## 2. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│  Host Machine                                               │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │Core Emulator │    │ Docker: oai_sionna_proxy          │   │
│  │ (optional)   │    │                                    │   │
│  │              │    │  Sionna Proxy (v4.py)              │   │
│  │ master_      │───>│    ↓                               │   │
│  │ config.yaml  │    │  Channel Hook (channel_hook.py)    │   │
│  │              │    │    ↓                               │   │
│  │ REST API     │    │  CsiNet Engine (csinet_engine.py)  │   │
│  │ :7101        │    │    ↓                               │   │
│  └──────────────┘    │  CSI Injection (csi_injection.py)  │   │
│                      │    ↓                               │   │
│  launch_all.sh ─────>│  gNB ←→ UE (GPU IPC)              │   │
│                      └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**CsiNet 데이터 흐름**:
1. Sionna가 채널 H를 생성
2. `channel_hook.py`가 DL 채널 H를 가로챔
3. `csinet_engine.py`가 H → 인코딩 → 디코딩 → H_hat 복원
4. `csi_injection.py`가 H_hat으로 PMI/CQI/RI 계산 (메모리 저장)
5. 원본 H는 그대로 gNB에 전달됨 (H_hat은 분석/로깅용)

---

## 3. 방법 A: Core Emulator 사용

Core Emulator는 `master_config.yaml`을 중앙 관리하고, REST API로 런타임 설정 변경을 지원함.

### 3.1 Step 1: master_config.yaml 수정

```bash
cd ~/oai_sionna_junxiu/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy
vi master_config.yaml
```

**필수 수정 항목**:

```yaml
# ── 시스템 ──
system:
  num_ues: 4
  mode: gpu-ipc
  channel_mode: dynamic       # 실시간 Sionna 채널
  mu_mimo: true

# ── 안테나 (4T4R) ──
antenna:
  gnb:
    nx: 2
    ny: 1
    polarization: dual        # 2×1×dual = 4 포트
  ue:
    nx: 2
    ny: 1

# ── 채널 ──
channel:
  scenario: UMa-NLOS          # 원하는 시나리오 (하이픈 형식)
  speed: 3.0                  # UE 이동 속도 (km/h)
  bs_height_m: 25.0
  ue_height_m: 1.5
  isd_m: 500

# ── CsiNet (핵심) ──
csinet:
  enabled: true               # ← 반드시 true
  mode: conditioned           # ← Proposed CsiNet (conditioned)
  compression_ratio: 0.25     # gamma = 1/4
  scenario: UMa_NLOS          # ← 밑줄 형식, channel.scenario와 동일 시나리오
  checkpoint_dir: /workspace/csinet_checkpoints
  csi_rs_period: 20
  csinet_path: /workspace/graduation/csinet
```

> **주의**: `channel.scenario`는 하이픈(`UMa-NLOS`), `csinet.scenario`는 밑줄(`UMa_NLOS`) 형식.

### 3.2 Step 2: Core Emulator 시작

```bash
cd ~/oai_sionna_junxiu/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy

python3 core_emulator.py --config master_config.yaml
```

정상 시작 시 출력:

```
[Core Emulator] TCP: 7100 | HTTP: 7101
[Core Emulator] Loaded config: master_config.yaml
```

설정 확인 (별도 터미널):

```bash
# CsiNet 설정 확인
curl -s http://localhost:7101/api/v1/csinet/config | python3 -m json.tool

# CsiNet 환경변수 확인
curl -s http://localhost:7101/api/v1/csinet/env | python3 -m json.tool

# 전체 launch params 확인
curl -s http://localhost:7101/launch_params | python3 -m json.tool
```

### 3.3 Step 3: Simulator 실행

```bash
sudo bash launch_all.sh -c localhost:7100
```

`-c localhost:7100` 옵션으로 Core Emulator에 연결하면:
- 안테나, UE 수, 채널 파라미터를 자동으로 가져옴
- CsiNet 환경변수를 `/api/v1/csinet/env`에서 가져와 컨테이너에 전달
- gNB 설정 파일을 자동 생성

따라서 `-c` 사용 시 다른 옵션(`-n`, `-ga`, `-ua`, `-pol` 등)을 별도로 줄 **필요 없음** (YAML에서 모두 읽어옴).

### 3.4 런타임 CsiNet 설정 변경 (프록시 재시작 필요)

```bash
# CsiNet 모드를 baseline으로 변경
curl -X POST http://localhost:7101/api/v1/csinet/config \
  -H "Content-Type: application/json" \
  -d '{"mode": "baseline"}'

# compression ratio 변경
curl -X POST http://localhost:7101/api/v1/csinet/config \
  -H "Content-Type: application/json" \
  -d '{"compression_ratio": 0.125}'

# CsiNet 비활성화
curl -X POST http://localhost:7101/api/v1/csinet/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

> 변경 후 프록시를 재시작해야 적용됨 (`Ctrl+C` 후 `sudo bash launch_all.sh -c localhost:7100` 재실행).

---

## 4. 방법 B: Core Emulator 없이 직접 실행

Core Emulator 없이 `launch_all.sh` CLI 옵션과 환경변수로 직접 실행.

### 4.1 Step 1: 환경변수 설정

CsiNet 관련 환경변수를 `export`로 설정:

```bash
# CsiNet 환경변수 설정
export CSINET_ENABLED=1
export CSINET_MODE=conditioned          # baseline | conditioned
export CSINET_GAMMA=0.25                # compression ratio
export CSINET_SCENARIO=UMa_NLOS         # 밑줄 형식
export CSINET_PERIOD=20                 # CSI-RS period (슬롯)
export CSINET_PATH=/workspace/graduation/csinet
export CSINET_CHECKPOINT_DIR=/workspace/csinet_checkpoints
```

### 4.2 Step 2: launch_all.sh 실행

```bash
sudo -E bash launch_all.sh \
  -n 4 \
  -ga 2 1 \
  -ua 2 1 \
  -pol dual \
  -cm dynamic
```

**옵션 설명**:

| 옵션 | 값 | 의미 |
|------|---|------|
| `-n 4` | 4 | UE 4대 |
| `-ga 2 1` | Nx=2, Ny=1 | gNB 안테나 2×1 |
| `-ua 2 1` | Nx=2, Ny=1 | UE 안테나 2×1 |
| `-pol dual` | dual | Cross-polarization (×2) → 총 4T4R |
| `-cm dynamic` | dynamic | 실시간 Sionna 채널 |

### 4.3 직접 docker exec로 실행하는 방법 (고급)

`launch_all.sh`를 사용하지 않고 직접 컨테이너 내에서 실행:

```bash
docker exec -it \
  -e CSINET_ENABLED=1 \
  -e CSINET_MODE=conditioned \
  -e CSINET_GAMMA=0.25 \
  -e CSINET_SCENARIO=UMa_NLOS \
  -e CSINET_PERIOD=20 \
  -e CSINET_PATH=/workspace/graduation/csinet \
  -e CSINET_CHECKPOINT_DIR=/workspace/csinet_checkpoints \
  -e GPU_IPC_V5_GNB_ANT=4 \
  -e GPU_IPC_V5_GNB_NX=2 \
  -e GPU_IPC_V5_GNB_NY=1 \
  -e GPU_IPC_V5_UE_ANT=4 \
  -e GPU_IPC_V5_UE_NX=2 \
  -e GPU_IPC_V5_UE_NY=1 \
  oai_sionna_proxy python3 -u \
    /workspace/vRAN_Socket/G1C_MultiUE_MIMO_Channel_Proxy/v4.py \
    --mode=gpu-ipc \
    --gnb-ant=4 --ue-ant=4 \
    --gnb-nx=2 --gnb-ny=1 \
    --ue-nx=2 --ue-ny=1 \
    --num-ues=4 \
    --channel-mode=dynamic \
    --polarization=dual
```

> 이 방법은 gNB/UE 프로세스를 별도로 관리해야 하므로, 일반적으로 `launch_all.sh` 사용을 권장.

---

## 5. CsiNet 설정 상세

### 5.1 CsiNet 모드 비교

| 모드 | 설명 | 필요 체크포인트 |
|------|------|----------------|
| `baseline` | 기본 CsiNet (statistics 미사용) | `csinet_{scenario}_gamma{ratio}_best.weights.h5` |
| `conditioned` | **Proposed** — 채널 통계 기반 FiLM 컨디셔닝 | `cond_csinet_...h5` + `stat_ae_...h5` |

### 5.2 Compression Ratio (gamma)

| gamma | Codeword 크기 (M) | 압축률 | 체크포인트 존재 |
|-------|-------------------|--------|----------------|
| 0.25 (1/4) | 64 | 4:1 | O |
| 0.125 (1/8) | 32 | 8:1 | O |
| 0.0625 (1/16) | 16 | 16:1 | O |
| 0.03125 (1/32) | 8 | 32:1 | O |

> M = 2 × Nt × Nc' × gamma = 2 × 4 × 32 × gamma

### 5.3 시나리오 매핑

| channel.scenario (하이픈) | csinet.scenario (밑줄) | 환경 |
|---------------------------|------------------------|------|
| `UMi-LOS` | `UMi_LOS` | Urban Micro, 가시선 |
| `UMi-NLOS` | `UMi_NLOS` | Urban Micro, 비가시선 |
| `UMa-NLOS` | `UMa_NLOS` | Urban Macro, 비가시선 |
| `UMa-LOS` | `UMa_LOS` | Urban Macro, 가시선 |

### 5.4 환경변수 참조표

| 환경변수 | YAML 키 | 기본값 | 설명 |
|----------|---------|--------|------|
| `CSINET_ENABLED` | `csinet.enabled` | `0` | 1=활성, 0=비활성 |
| `CSINET_MODE` | `csinet.mode` | `baseline` | baseline / conditioned |
| `CSINET_GAMMA` | `csinet.compression_ratio` | `0.25` | 압축률 |
| `CSINET_SCENARIO` | `csinet.scenario` | `UMi_NLOS` | 시나리오 (밑줄 형식) |
| `CSINET_PERIOD` | `csinet.csi_rs_period` | `20` | H 캡처 주기 (슬롯) |
| `CSINET_PATH` | `csinet.csinet_path` | `/workspace/graduation/csinet` | CsiNet 모듈 경로 |
| `CSINET_CHECKPOINT_DIR` | `csinet.checkpoint_dir` | `/workspace/csinet_checkpoints` | 가중치 경로 |

---

## 6. 실행 확인 및 모니터링

### 6.1 CsiNet 초기화 확인

프록시 시작 시 로그에서 다음을 확인:

```
[CsiNet Hook] Initializing CsiNet sidecar...
[CsiNet Engine] Loaded conditioned: /workspace/csinet_checkpoints/cond_csinet_UMa_NLOS_gamma0.2500_best.weights.h5
[CsiNet Hook] CsiNet hook active — mode=conditioned, gamma=0.25, scenario=UMa_NLOS
```

### 6.2 MU-MIMO 동작 확인

gNB 로그에서 MU-MIMO 스케줄링이 활성화되었는지 확인:

```
[MAC] MU-MIMO: scheduling 4 UEs simultaneously
```

### 6.3 CsiNet 비활성 상태 확인

CsiNet이 비활성 상태(`CSINET_ENABLED=0`)이면:

```
[CsiNet Hook] CsiNet disabled — skipping initialization
```

이 경우 채널은 Sionna 원본 H가 그대로 사용됨 (기존 동작과 동일).

---

## 7. FAQ / 문제 해결

### Q1: "No such file or directory: csinet_checkpoints" 오류

체크포인트가 컨테이너에 마운트되지 않음.

```bash
# 호스트에서 확인
ls ~/oai_sionna_junxiu/csinet_checkpoints/

# 컨테이너 안에서 확인
docker exec oai_sionna_proxy ls /workspace/csinet_checkpoints/

# 마운트 안 되어 있으면 docker-compose.yml 확인 후 재시작
cd ~/oai_sionna_junxiu && docker compose down && docker compose up -d
```

### Q2: CsiNet 가중치 로딩 에러 (ValueError)

모델 아키텍처와 체크포인트 불일치. 체크포인트는 아래 아키텍처로 학습됨:
- `COV_LATENT=16`, `PDP_LATENT=8`, `COND_DIM=24`
- FiLM: single Dense layer
- Decoder: 2 refine blocks, 8 filters

`graduation/csinet/models/` 내 모델 파일이 체크포인트와 호환되는 버전인지 확인.

### Q3: GPU 메모리 부족

CsiNet은 추가 GPU 메모리를 사용함. 4UE 시뮬레이션 + CsiNet은 약 4-6GB 추가 필요.

```bash
# GPU 사용량 확인
nvidia-smi
```

필요 시 `CUDA_VISIBLE_DEVICES`로 GPU 지정:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 ...
```

### Q4: Core Emulator 연결 실패

```bash
# Core Emulator가 실행 중인지 확인
curl -s http://localhost:7101/api/v1/csinet/config

# 포트 확인: TCP=7100, HTTP=7101 (기본)
ss -tlnp | grep 7101
```

### Q5: channel.scenario와 csinet.scenario가 다르면?

다를 수 있지만 권장하지 않음. CsiNet은 학습 시나리오에 특화되어 있으므로, 채널 시나리오와 CsiNet 시나리오를 일치시키는 것이 최적 성능을 보장함.

### Q6: Baseline vs Conditioned 성능 차이는?

Conditioned CsiNet이 채널 통계(공분산, PDP)를 활용하여 일반적으로 2-5dB NMSE 개선. 특히 저압축률(gamma=1/32)에서 차이가 큼.

---

## 부록: 빠른 시작 명령어 모음

### A. Core Emulator 사용 (권장)

```bash
# 1. master_config.yaml에서 csinet.enabled=true, csinet.mode=conditioned 설정
# 2. Core Emulator 시작
python3 core_emulator.py --config master_config.yaml &

# 3. Simulator 실행
sudo bash launch_all.sh -c localhost:7100
```

### B. Core Emulator 없이

```bash
# 1. 환경변수 설정 + 실행 (한 줄)
CSINET_ENABLED=1 CSINET_MODE=conditioned CSINET_GAMMA=0.25 \
CSINET_SCENARIO=UMa_NLOS CSINET_PERIOD=20 \
CSINET_PATH=/workspace/graduation/csinet \
CSINET_CHECKPOINT_DIR=/workspace/csinet_checkpoints \
sudo -E bash launch_all.sh -n 4 -ga 2 1 -ua 2 1 -pol dual -cm dynamic
```

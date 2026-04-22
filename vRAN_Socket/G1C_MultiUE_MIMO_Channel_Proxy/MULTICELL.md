# Multi-Cell MU-MIMO 시뮬레이션 아키텍처

## 1. 개요

본 시스템은 **단일 서버**에서 여러 5G NR 기지국(gNB)과 UE를 동시에 실행하여 멀티셀 MU-MIMO 환경을 시뮬레이션한다.
핵심 구성요소는 다음 세 가지이다:

1. **Multi-Cell Channel Proxy** (`v4_multicell.py`): GPU 공유 메모리(IPC V7)를 통해 셀 간 IQ 데이터를 중계하고 채널 모델·ICI를 적용
2. **OAI gNB/UE**: 셀별 독립 프로세스로 실행, RFsimulator + GPU IPC로 Proxy와 연결
3. **통합 런처** (`launch_multicell.sh`): 설정 생성 → Proxy → gNB → UE 순서로 전체 스택 기동

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Host (Linux)                                 │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │            Docker: oai_sionna_proxy (GPU)                    │   │
│  │                                                              │   │
│  │   v4_multicell.py (단일 프로세스)                              │   │
│  │   ┌─────────────────────────────────────────────────────┐    │   │
│  │   │  CellContext[0]    CellContext[1]   ... CellContext[N-1]│ │   │
│  │   │   ├ ipc_gnb         ├ ipc_gnb           ├ ipc_gnb      │ │   │
│  │   │   ├ ipc_ues[0..K]   ├ ipc_ues[0..K]     ├ ipc_ues[0..K]│ │   │
│  │   │   ├ pipelines_dl    ├ pipelines_dl       ├ pipelines_dl │ │   │
│  │   │   ├ pipelines_ul    ├ pipelines_ul       ├ pipelines_ul │ │   │
│  │   │   └ channel_bufs    └ channel_bufs       └ channel_bufs │ │   │
│  │   └─────────────────────────────────────────────────────────┘ │   │
│  │            ↕ GPU SHM (IPC V7)                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌── Cell 0 ──────────┐  ┌── Cell 1 ──────────┐  ┌── Cell N-1 ──┐ │
│  │ gNB (nr-softmodem)  │  │ gNB (nr-softmodem)  │  │ gNB          │ │
│  │   CELL_IDX=0        │  │   CELL_IDX=1        │  │   CELL_IDX=  │ │
│  │                     │  │                     │  │     N-1      │ │
│  │ UE0 (nr-uesoftmodem)│  │ UE0 (nr-uesoftmodem)│  │ UE0          │ │
│  │ UE1                 │  │ UE1                 │  │ UE1          │ │
│  │ ...                 │  │ ...                 │  │ ...          │ │
│  │ UE(K-1)             │  │ UE(K-1)             │  │ UE(K-1)      │ │
│  └─────────────────────┘  └─────────────────────┘  └──────────────┘ │
│                                                                     │
│  ┌── OAI CN5G (Docker) ────────────────────────────────────────┐    │
│  │  AMF ─── SMF ─── UPF ─── NRF/UDR/UDM/AUSF                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. GPU IPC V7 공유 메모리 구조

### 2.1 SHM 파일 명명 규칙

멀티셀 환경에서는 셀 인덱스와 UE 인덱스를 포함한 SHM 파일을 생성한다.

| 경로 | 연결 | 용도 |
|------|------|------|
| `/tmp/oai_gpu_ipc/gpu_ipc_shm_cell{C}` | Proxy ↔ gNB Cell C | gNB DL TX / UL RX |
| `/tmp/oai_gpu_ipc/gpu_ipc_shm_cell{C}_ue{K}` | Proxy ↔ UE K of Cell C | UE DL RX / UL TX |

환경변수로 셀/UE 인덱스를 지정한다:

```bash
# gNB (Cell 0)
RFSIM_GPU_IPC_V7=1 RFSIM_GPU_IPC_CELL_IDX=0 nr-softmodem -O gnb_cell0.conf ...

# UE (Cell 1, UE 2)
RFSIM_GPU_IPC_V7=1 RFSIM_GPU_IPC_CELL_IDX=1 RFSIM_GPU_IPC_UE_IDX=2 nr-uesoftmodem ...
```

### 2.2 SHM 레이아웃 (4096 Bytes)

```
Offset   Size    Field
──────   ─────   ────────────────────────────────────
0-63     64B     dl_tx CUDA IPC handle (gNB → Proxy)
64-127   64B     dl_rx CUDA IPC handle (Proxy → UE)
128-191  64B     ul_tx CUDA IPC handle (UE → Proxy)
192-255  64B     ul_rx CUDA IPC handle (Proxy → gNB)

256      4B      magic (0x47505538 = "GPU8")
260      4B      version (1)
264      4B      cir_time (4608000 = 150 slots = 75ms)
268      4B      num_ues

272-303  32B     Per-buffer antenna config (nbAnt, cir_size × 4 buffers)
304-351  48B     Producer timestamps (dl_tx, dl_rx, ul_tx, ul_rx)
352-367  16B     Consumer timestamps (dl_consumer, ul_consumer)
368-383  16B     Futex sequence counters (dl_tx, dl_rx, ul_tx, ul_rx)
384-391  8B      UL sync timestamp
392-4095         Reserved
```

### 2.3 순환 버퍼

각 버퍼의 크기는 `cir_time × nbAnt × 4 bytes`이다.

- `cir_time` = 4,608,000 (150 슬롯, 약 75ms)
- 예: 4안테나 DL TX → 4,608,000 × 4 × 4 = **약 70 MB**
- 셀당 4개 버퍼 + UE별 4개 버퍼 → 셀당 메모리 = `(1 + K_ue) × 4 × cir_time × avg_ant × 4`

### 2.4 IQ 데이터 흐름

```
DL (Downlink):
  gNB → [dl_tx GPU buf] → Proxy reads → Sionna Channel → [dl_rx GPU buf] → UE reads

UL (Uplink):
  UE → [ul_tx GPU buf] → Proxy reads → Sionna Channel → superposition → [ul_rx GPU buf] → gNB reads
```

---

## 3. 셀별 설정 생성 (`generate_gnb_configs.py`)

기존 `gnb.conf` 템플릿에서 셀별로 고유한 파라미터를 갖는 설정 파일을 자동 생성한다.

### 셀별 변경 파라미터

| 파라미터 | Cell 0 | Cell 1 | Cell 2 | 규칙 |
|---------|--------|--------|--------|------|
| `gNB_ID` | `0xe00` | `0xe01` | `0xe02` | `0xe00 + c` |
| `gNB_name` | `gNB-Cell0` | `gNB-Cell1` | `gNB-Cell2` | 문자열 |
| `nr_cellid` | `12345678L` | `12345679L` | `12345680L` | `12345678 + c` |
| `physCellId` | `0` | `1` | `2` | `c` |
| `prach_RootSequenceIndex` | `1` | `11` | `21` | `1 + c*10` (충돌 방지) |
| `GNB_PORT_FOR_S1U` | `2152` | `2153` | `2154` | `2152 + c` |

### 사용법

```bash
python3 generate_gnb_configs.py \
    --template /path/to/gnb.sa.band78.fr1.106PRB.usrpb210.conf \
    --num-cells 3 \
    --output-dir /tmp/multicell_configs
```

출력: `/tmp/multicell_configs/gnb_cell0.conf`, `gnb_cell1.conf`, `gnb_cell2.conf`

**불변 파라미터**: AMF IP, 주파수, TDD 패턴, 안테나 설정, 코드북 등은 모든 셀에서 동일하다.
(셀마다 다른 주파수나 안테나를 쓰려면 `per_cell_overrides`를 확장해야 한다.)

---

## 4. Multi-Cell Channel Proxy (`v4_multicell.py`)

### 4.1 클래스 구조

```
MultiCellProxy
├── cells: List[CellContext]      # N개 셀 컨텍스트
├── ici_atten_dB: float           # ICI 감쇠 (dB)
├── ici_linear: float             # 10^(-dB/20) 선형 게인
├── enable_ici: bool              # ICI 적용 여부 (< 100dB이면 ON)
├── time_dilation: float          # 시간 팽창 배수
│
├── run()                         # 메인 루프
├── _dl_broadcast_cell()          # DL: gNB → 채널 → UE[K]
├── _ul_combine_cell()            # UL: UE[K] → 채널 → 중첩 → gNB
├── _add_dl_ici()                 # DL ICI: 타 셀 gNB → victim UE
└── _add_ul_ici()                 # UL ICI: 타 셀 UE → victim gNB

CellContext
├── cell_idx: int
├── ipc_gnb: GPUIpcV7Interface    # gNB SHM 인터페이스
├── ipc_ues: List[GPUIpcV7Interface]  # UE SHM 인터페이스
├── pipelines_dl: List[GPUSlotPipeline]
├── pipelines_ul: List[GPUSlotPipeline]
├── channel_buffers: List[...]    # Sionna 채널 버퍼 (per-UE)
└── init_ipc()                    # SHM 파일 생성/초기화
```

### 4.2 메인 루프 (`run()`)

```
while True:
    ┌─ DL Phase ──────────────────────────────────────────────┐
    │ for each cell:                                          │
    │   1. gNB의 dl_tx 타임스탬프 확인 (새 데이터 존재?)        │
    │   2. _dl_broadcast_cell():                              │
    │      - gNB DL TX 읽기                                   │
    │      - Sionna 채널 적용 (per-UE)                        │
    │      - UE DL RX에 쓰기                                  │
    │   3. _add_dl_ici():                                     │
    │      - 타 셀 gNB 신호 읽기 → 감쇠 → victim UE에 가산    │
    └─────────────────────────────────────────────────────────┘

    ┌─ Keepalive ─────────────────────────────────────────────┐
    │ for each cell:                                          │
    │   ka_target = max(proxy_dl_head, proxy_ul_head)         │
    │   gNB.set_last_ul_rx_ts(ka_target)                      │
    │   → gNB가 Proxy 속도에 맞추어 DL 전송 (오버런 방지)      │
    └─────────────────────────────────────────────────────────┘

    ┌─ UL Phase ──────────────────────────────────────────────┐
    │ for each cell:                                          │
    │   1. UE의 ul_tx 타임스탬프 확인                          │
    │   2. _ul_combine_cell():                                │
    │      - UE UL TX 읽기                                    │
    │      - Sionna 채널 적용 (per-UE)                        │
    │      - 모든 UE 신호 중첩 → gNB UL RX에 쓰기             │
    │   3. _add_ul_ici():                                     │
    │      - 타 셀 UE 신호 읽기 → 감쇠 → victim gNB에 가산    │
    └─────────────────────────────────────────────────────────┘

    ┌─ Time Dilation ─────────────────────────────────────────┐
    │ if dilation > 1.0:                                      │
    │   sleep(slot_duration × (dilation - 1))                 │
    └─────────────────────────────────────────────────────────┘
```

### 4.3 ICI (Inter-Cell Interference) 모델

현재 구현은 **고정 감쇠 + 직접 가산** 모델이다:

```
ICI 신호 = 타 셀 gNB 원시 DL 신호 × ici_linear
victim UE 수신 = 서빙 셀 채널 출력 + Σ(ICI 신호)
```

- `ici_linear = 10^(-ici_atten_dB / 20)` (전압 도메인)
- 예: `-15dB` → `ici_linear ≈ 0.178`
- ICI 경로에 별도 채널 모델은 적용하지 않는다 (추후 확장 가능)
- `ici_atten_dB >= 100`이면 ICI 비활성화

### 4.4 Keepalive 및 동기화

gNB는 Proxy보다 훨씬 빠르게 DL 데이터를 쓴다 (~150 slots/s vs Proxy ~50 slots/s).
이로 인해 순환 버퍼 오버런이 발생할 수 있다.

**해결 방법**: Proxy가 처리한 위치(`proxy_dl_head`)까지만 gNB의 UL RX 타임스탬프를 전진시킨다.
gNB는 UL RX가 전진해야 다음 DL 슬롯을 처리하므로, 결과적으로 **Proxy 속도에 맞추어 gNB가 조절된다**.

```python
ka_target = max(proxy_dl_heads[ci], proxy_ul_heads[ci])
cell.ipc_gnb.set_last_ul_rx_ts(ka_target)
```

**모니터링**: `[MC STATS]` 로그의 `gap` 필드로 확인:
- `gap = (gNB DL head - Proxy DL head) / slot_samples`
- 정상: 3~6 슬롯
- 비정상: 수백~수천 슬롯 (오버런 징후)

### 4.5 시간 팽창 (Time Dilation)

멀티셀 스케일링 테스트에서 안정성 확인을 위해, 슬롯 처리를 인위적으로 느리게 한다.

```
slot_duration = total_cpx / 30720000   ≈ 1.008ms (30kHz SCS)
dilation_sleep = slot_duration × (dilation - 1.0)
```

예:
- `--time-dilation 1.0` → 실시간 (sleep 없음)
- `--time-dilation 10.0` → 슬롯당 약 9ms 추가 sleep → 모든 것이 10배 느리게

---

## 5. 통합 런처 (`launch_multicell.sh`)

### 5.1 실행 순서

```
1. 이전 프로세스 정리 (pkill, Docker 내 kill, SHM 삭제)
2. generate_gnb_configs.py로 셀별 gnb.conf 생성
3. Docker oai_sionna_proxy에서 v4_multicell.py 실행
4. SHM 파일 권한 수정 (chmod 666)
5. Proxy 준비 완료 대기 (최대 120초)
6. 셀별 gNB 프로세스 시작 (5초 간격)
7. 셀별 UE 프로세스 시작 (1초 간격)
8. UE RRC_CONNECTED 대기 (셀당 최대 60초)
9. 모니터링 (tail -f proxy.log)
```

### 5.2 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-nc NUM` | 2 | 셀 수 |
| `-n NUM` | 2 | 셀당 UE 수 |
| `-ga Nx Ny` | 2 1 | gNB 안테나 격자 |
| `-ua Nx Ny` | 1 1 | UE 안테나 격자 |
| `-pol MODE` | single | 편파 (single / dual) |
| `-ici dB` | 15 | 셀 간 간섭 감쇠 (>100=OFF) |
| `-pl dB` | 0 | 경로 손실 |
| `-snr dB` | OFF | AWGN SNR |
| `-nf dBFS` | OFF | AWGN noise floor |
| `-p1b PATH` | P1B npz | Ray-tracing 채널 데이터 (컨테이너 내 경로) |
| `-rx INDICES` | random | UE RX 인덱스 (쉼표 구분) |
| `-td FACTOR` | 1.0 | 시간 팽창 배수 |
| `-d SEC` | ∞ | 실행 시간 (초) |
| `-b` | OFF | 채널 바이패스 (IQ 패스스루) |

### 5.3 실행 예시

```bash
# 2셀, 셀당 4UE, dual-pol, 120초 실행
sudo bash launch_multicell.sh -nc 2 -n 4 -ga 2 1 -pol dual -d 120

# 4셀, 셀당 2UE, ray-tracing 채널, 시간 10배 느리게
sudo bash launch_multicell.sh -nc 4 -n 2 -ga 2 1 \
    -p1b /workspace/vRAN_Socket/P1B_Valid_Results/Area1_7.5GHz_Rays_Valid_RXs.npz \
    -td 10.0

# ICI 없이 단순 멀티셀 테스트
sudo bash launch_multicell.sh -nc 2 -n 2 -ici 200
```

### 5.4 로그 구조

```
~/oai_sionna_junxiu/logs/20260318_143025_MC_2cell_4ue_ga2x1/
├── proxy.log              # v4_multicell.py 로그 ([MC STATS] 포함)
├── gnb_cell0.log          # Cell 0 gNB 로그
├── gnb_cell1.log          # Cell 1 gNB 로그
├── ue_cell0_ue0.log       # Cell 0, UE 0 로그
├── ue_cell0_ue1.log       # Cell 0, UE 1 로그
├── ue_cell1_ue0.log       # Cell 1, UE 0 로그
└── ue_cell1_ue1.log       # Cell 1, UE 1 로그
```

심볼릭 링크: `~/oai_sionna_junxiu/logs/latest/` → 최신 로그 디렉토리

---

## 6. UE IMSI 할당 규칙

UE의 IMSI는 글로벌 인덱스 기반으로 자동 할당된다:

```
global_ue_idx = cell_idx × ues_per_cell + ue_idx
imsi = 00101000000{global_ue_idx + 1:04d}
```

| Cell | UE | global_idx | IMSI |
|------|-----|-----------|------|
| 0 | 0 | 0 | 001010000000001 |
| 0 | 1 | 1 | 001010000000002 |
| 0 | 2 | 2 | 001010000000003 |
| 0 | 3 | 3 | 001010000000004 |
| 1 | 0 | 4 | 001010000000005 |
| 1 | 1 | 5 | 001010000000006 |
| ... | ... | ... | ... |

이 IMSI들은 OAI CN5G MySQL DB의 `AuthenticationSubscription` 테이블에 사전 등록되어 있어야 한다.
(현재 `oai_db.sql`에 64개 UE 분량이 등록됨)

---

## 7. Core Emulator 멀티셀 API

`core_emulator.py`의 FastAPI 엔드포인트로 멀티셀 설정을 관리할 수 있다.

### 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/v1/cell/configure` | 셀 등록/구성 (PCI, gNB_ID, GTP-U 포트 등 자동 생성) |
| POST | `/api/v1/cell/{id}/activate` | 셀 활성화 |
| GET | `/api/v1/cell/{id}/status` | 셀 상태 조회 |
| GET | `/api/v1/cells` | 전체 셀 목록 |
| GET | `/api/v1/kpi` | 멀티셀 통합 KPI (총 셀 수, 총 UE 수 등) |

### 셀 구성 예시

```bash
# Cell 0 구성
curl -X POST http://localhost:7101/api/v1/cell/configure \
  -H "Content-Type: application/json" \
  -d '{"cell_id": 0}'

# 응답:
{
  "cell_id": 0,
  "pci": 0,
  "gnb_id": "0xe00",
  "nr_cellid": 12345678,
  "prach_root": 1,
  "gtpu_port": 2152,
  "status": "configured"
}

# Cell 0 활성화
curl -X POST http://localhost:7101/api/v1/cell/0/activate
```

### master_config.yaml 멀티셀 섹션

```yaml
multicell:
  enabled: true       # 멀티셀 모드 활성화
  num_cells: 2        # 셀 수
  ues_per_cell: 4     # 셀당 UE 수
  ici_atten_dB: 15.0  # ICI 감쇠 (dB)
  p1b_npz: null       # Ray-tracing 채널 데이터 경로
  per_cell_overrides: {}  # 셀별 파라미터 오버라이드
```

---

## 8. 프리셋 프로파일

`presets/` 디렉토리에 멀티셀 전용 프리셋이 있다:

| 프리셋 파일 | 셀 수 | UE/셀 | 설명 |
|------------|-------|-------|------|
| `multicell_2cell_4ue.yaml` | 2 | 4 | 2셀 MU-MIMO, Type-II, ICI 15dB |
| `multicell_4cell_4ue.yaml` | 4 | 4 | 4셀 MU-MIMO, Type-II, ICI 15dB |

프리셋 적용:

```bash
curl -X POST http://localhost:7101/api/v1/apply-preset \
  -H "Content-Type: application/json" \
  -d '{"preset": "multicell_2cell_4ue"}'
```

---

## 9. 스케일링 테스트 가이드

### 9.1 목적

셀 수를 늘렸을 때의 성능 변화(슬롯 처리 시간, 안정성)를 검증한다.

### 9.2 방법

1. **기준선 측정** (1셀):

```bash
sudo bash launch_multicell.sh -nc 1 -n 4 -d 120
# proxy.log에서 [MC STATS] avg slot_time 확인
```

2. **스케일 업** (2셀, 4셀):

```bash
sudo bash launch_multicell.sh -nc 2 -n 4 -d 120
sudo bash launch_multicell.sh -nc 4 -n 4 -d 120
```

3. **시간 팽창으로 안정성 검증**:

```bash
# 10배 느리게 실행하여 동기화/메모리 문제 확인
sudo bash launch_multicell.sh -nc 4 -n 4 -td 10.0 -d 300
```

### 9.3 핵심 지표 (`[MC STATS]` 로그)

```
[MC STATS] 60s | DL=2847 UL=2842 | slot_time: avg=39050us min=28000us max=85000us | rate=47 slots/s | gap=[c0:4slots,c1:5slots] | dilation=1.0x
```

| 지표 | 의미 | 정상 범위 |
|------|------|----------|
| `slot_time avg` | 슬롯 평균 처리 시간 | 1셀: ~21ms, 2셀: ~40ms |
| `rate` | 초당 처리 슬롯 수 | ≥ 25 (실시간 기준 ~990) |
| `gap` | gNB DL head - Proxy DL head (슬롯) | 3~10 (안정), >100 (오버런 위험) |
| `DL/UL` | 처리된 DL/UL 이벤트 누적 | 지속 증가해야 정상 |
| `dilation` | 시간 팽창 배수 | 설정값과 일치 |

### 9.4 이상적 스케일링

| 셀 수 | UE 수 | 예상 slot_time | 스케일링 팩터 |
|-------|-------|---------------|-------------|
| 1 | 4 | ~21ms | 1.0× |
| 2 | 8 | ~40-55ms | ~1.9-2.6× |
| 4 | 16 | ~80-120ms | ~3.8-5.7× |

선형 스케일링에 가까울수록 좋다. 급격한 증가나 크래시는 동기화/메모리 문제를 시사한다.

---

## 10. 메모리 예산

### 10.1 GPU 메모리 (per cell)

```
cir_time = 4,608,000 samples
sample_size = 4 bytes (c16_t)

gNB SHM (4 buffers):
  dl_tx: cir_time × gnb_ant × 4  (gNB write)
  dl_rx: (없음, gNB SHM에 UE용 DL RX 없음)
  ul_tx: (없음)
  ul_rx: cir_time × gnb_ant × 4  (Proxy write)

UE SHM (4 buffers × K_ue):
  dl_rx: cir_time × ue_ant × 4   (Proxy write)
  ul_tx: cir_time × ue_ant × 4   (UE write)
```

4T4R, 4 UE/셀 기준 per-cell GPU:
- gNB: `2 × 4,608,000 × 4 × 4` ≈ **140 MB**
- UE×4: `4 × 2 × 4,608,000 × 1 × 4` ≈ **140 MB**
- **셀당 총: ~280 MB**

### 10.2 시스템 메모리

| 프로세스 | 메모리 (approx) |
|---------|---------------|
| gNB (nr-softmodem) | ~2-3 GB |
| UE (nr-uesoftmodem) | ~0.5-1 GB |
| v4_multicell.py | ~2-4 GB (Sionna + CuPy) |
| OAI CN5G (AMF+SMF+UPF+...) | ~3 GB |

### 10.3 최대 셀 수 추정 (128 GB RAM, 48 GB GPU 기준)

- GPU: 48 GB / 280 MB ≈ **~170셀** (이론상)
- RAM: (128 - 10) GB / (3 + 4×1) GB ≈ **~17셀** (실제 병목)
- **실질적 최대: ~8-12셀** (여유 확보 포함)

---

## 11. 알려진 제약 및 향후 과제

### 현재 제약

1. **단일 AMF**: 모든 셀이 하나의 AMF에 연결 (멀티-AMF 미지원)
2. **동일 주파수**: 모든 셀이 동일 주파수/대역폭 사용 (co-channel)
3. **ICI 단순 모델**: 고정 감쇠만 적용, 간섭 경로에 독립 채널 모델 미적용
4. **핸드오버 미지원**: UE가 셀 간 이동 불가
5. **순차 처리**: 셀 간 DL/UL이 직렬 처리 (병렬화 가능)

### 향후 확장 방향

1. **ICI 고도화**: 간섭 경로에 독립적 Sionna 채널 모델 적용
2. **핸드오버**: X2/Xn 인터페이스 에뮬레이션
3. **이종 셀**: 셀별 다른 주파수/대역폭/안테나 구성
4. **분산 처리**: 셀별 GPU 분산 (multi-GPU)
5. **SDAP/DRB/RLC 제어**: Core Emulator에서 QoS 프로파일(5QI, AMBR) 및 Bearer 설정 통합 관리

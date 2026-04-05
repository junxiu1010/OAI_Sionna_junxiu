# G1A: Multi-UE Sionna Channel Proxy

G0 v12 기반 1 gNB - N UE (SISO) 멀티 UE 채널 프록시.

## 버전 이력

| 버전 | 파일 | 핵심 변경 |
|------|------|----------|
| v1 | `v1_multi_ue.py` | 기본 멀티 UE (순차 DL/UL 처리) |
| v2 | `v2_multi_ue.py` | DL 배치 처리 + per-UE 로깅 |
| v3 | `v3_multi_ue.py` | DL+UL 모두 배치 처리 |
| **v4** | **`v4_multi_ue.py`** | **통합 채널 생성 (1 Producer, 1 RingBuffer)** |

## 주요 변경 (G0 → G1A)

| 항목 | G0 v12 | G1A v1 |
|------|--------|--------|
| UE 수 | 1 | N (최대 8) |
| DL 채널 | 단일 | per-UE 독립 |
| UL 처리 | passthrough | per-UE 채널 + 합산 |
| GPU IPC 버퍼 | 4개 고정 | 2 + 2N (per-UE) |
| SHM 크기 | 512B | 4096B |
| FFT 배치 | 14 sym | N×14 sym (배치 모드) |

## 신호 흐름 (Multi-UE SISO, TDD Reciprocity)

### 채널 생성

#### v1~v3: per-UE Producer

Sionna가 UE별 독립 채널 `H_k` 생성 (서로 다른 속도, 거리, Topology):

```
ChannelProducer[0] (N_UE=1) → RingBuffer[0] → H_0 ∈ ℂ^(N_sym × N_FFT)
ChannelProducer[1] (N_UE=1) → RingBuffer[1] → H_1 ∈ ℂ^(N_sym × N_FFT)
  ...
ChannelProducer[k] (N_UE=1) → RingBuffer[k] → H_k ∈ ℂ^(N_sym × N_FFT)
```

#### v4: 통합 채널 생성

단일 Producer가 모든 UE 채널을 한 번에 생성:

```
UnifiedProducer (N_UE=num_ues) → SingleRingBuffer → H ∈ ℂ^(N_sym × N_UE × N_FFT)

GPU 커널 런치: N회 → 1회
TF graph trace: N회 → 1회
Python 스레드: N개 → 1개
```

Sionna의 `_H_PDP_FIX`와 `_H_TTI` 함수는 내부적으로 `N_UE` 차원을 벡터화하여 처리하므로,
`N_UE_sionna=num_ues`로 설정하면 단일 호출에서 모든 UE의 채널 계수가 동시 생성됨.

Rays 데이터는 `N_UE` 축으로 `tf.tile`하고, Topology (velocity, distance, LOS 등)는
UE별로 다른 값을 N_UE 차원에 배치:

```python
tau_rays_multi = tf.tile(tau_rays, [1, 1, num_ues, 1, 1])  # (1,1,N_UE,1,400)
velocities = tf.concat([...per-UE vel...], axis=1)          # (1,N_UE,3)
distance_3d = tf.constant([[[1.0, 1.3, 1.6, ...]]])         # (1,1,N_UE)
```

SISO에서 각 서브캐리어의 채널은 복소 스칼라이므로:

```
H_k^T = H_k     (스칼라 전치 = 항등)
```

TDD reciprocity에 의해 **DL과 UL은 동일 채널 계수 사용**.
v4에서는 `channel_buffer` 1개에서 `[:, k, :]` 슬라이싱으로 per-UE 채널 추출.
GPU view slice이므로 메모리 복사 없음.

### DL 흐름 (gNB → UE_k)

gNB의 단일 신호 `x_dl`을 N개 UE에 각각 다른 채널 적용:

```
                                    ┌─ H_0: Y_0[f] = IFFT(FFT(X_dl[f]) · H_0[f]) ──→ UE_0
                                    │
gNB ──→ x_dl (동일 신호) ──→ Proxy ─┼─ H_1: Y_1[f] = IFFT(FFT(X_dl[f]) · H_1[f]) ──→ UE_1
                                    │
                                    └─ H_k: Y_k[f] = IFFT(FFT(X_dl[f]) · H_k[f]) ──→ UE_k
```

수식 (서브캐리어 f, OFDM 심볼 s):

```
Y_k[s,f] = IFFT( FFT(X_dl[s,f]) · H_k[s,f] )     k = 0, 1, ..., N-1
```

배치 처리: `(N_UE × N_sym, N_FFT)` 텐서로 단일 GPU FFT 커널 실행.

### UL 흐름 (UE_k → gNB)

각 UE의 신호에 해당 UE의 채널 적용 후 합산:

```
UE_0 ──→ x_0 ──→ H_0^T: Z_0 = IFFT(FFT(X_0) · H_0^T) ─┐
                                                          │
UE_1 ──→ x_1 ──→ H_1^T: Z_1 = IFFT(FFT(X_1) · H_1^T) ─┼──→ Σ Z_k ──→ gNB
                                                          │
UE_k ──→ x_k ──→ H_k^T: Z_k = IFFT(FFT(X_k) · H_k^T) ─┘
```

수식:

```
Z_k[s,f] = IFFT( FFT(X_k[s,f]) · H_k^T[s,f] )    k = 0, 1, ..., N-1
Z[s,f]   = Σ_k Z_k[s,f]                            (복소 합산 후 int16 변환)
```

SISO에서 `H_k^T = H_k` 이므로, DL/UL 동일 연산.
코드: `_get_channels_for_ue(ue_idx)` → v4에서는 통합 `channel_buffer[:, k, :]`에서 슬라이스.

### 물리적 의미

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Proxy 내부 처리 순서                                │
│                                                                      │
│  DL/UL 공통:  X → FFT → ×H_k → IFFT → ×PL → +AWGN → clip → int16  │
│                                                                      │
│  DL: 입력 1개(gNB) → 출력 N개(UE별)        per-UE 독립 페이딩         │
│  UL: 입력 N개(UE별) → 출력 1개(gNB)        per-UE 채널 후 합산        │
│                                                                      │
│  채널 재사용: H_k 버퍼를 DL/UL 공유 (TDD reciprocity, SISO H^T=H)    │
└──────────────────────────────────────────────────────────────────────┘
```

### MIMO 확장 시 변화 (G1B 예정)

현재 SISO에서는 `H_k^T = H_k`로 동일하지만, MIMO로 확장하면:

```
SISO:  H_k ∈ ℂ^(1×1)  →  H_k^T = H_k           (스칼라, 전치 불필요)
MIMO:  H_k ∈ ℂ^(N_r×N_t)  →  H_k^T ∈ ℂ^(N_t×N_r)  (행렬 전치 필요)

DL:  y = H_k · x          (N_r×1) = (N_r×N_t) · (N_t×1)
UL:  z = H_k^T · x        (N_t×1) = (N_t×N_r) · (N_r×1)
```

MIMO에서는 UL 처리 시 명시적 전치 연산이 추가되며,
element-wise 곱 → 행렬-벡터 곱 (`einsum` 또는 `cp.matmul`)으로 변경.

## 개발 이력: 문제점 및 해결

### 문제 1: TDD 슬롯 비대칭 데드락

**증상**: Proxy가 DL 9개 처리 후 전체 시스템 정지 (freeze).

**원인**: TDD 패턴(7DL + 1flex + 2UL = 10 slots/period)에서 gNB는 DL/flex 슬롯에만
`trx_write`를 호출하므로 ring에 ~8개만 씀. UE는 10개 슬롯 모두에서 `trx_read`를
호출하므로 9번째 이후 무한 블로킹 → 교착 상태 발생.

```
데드락 사이클:
UE dl_read 블로킹 (데이터 없음)
→ UE ul_tx 쓸 수 없음
→ Proxy UL 못 받음 → ul_rx 못 씀
→ gNB dl_write 블로킹 (ring full)
→ gNB ul_read 못 함 → 완전한 교착
```

**해결 (3가지 수정)**:

| 수정 | 파일 | 내용 |
|------|------|------|
| UE DL read 타임아웃 | `gpu_ipc_v2.c` | 무한 블로킹 → 10ms 타임아웃, 0 리턴 |
| UE DL read 0 처리 | `simulator.c` | 0 리턴 시 zeros 패딩 + 타임스탬프 진행 |
| UL enqueue 논블로킹 | `v1_multi_ue.py` | ring full이면 skip (drop) |
| SHM 가시성 | `v1_multi_ue.py` | ring head/tail 갱신 후 `mmap.flush()` 추가 |

### 문제 2: PRACH `prach_id -1` Assertion 크래시

**증상**: 수백 프레임 정상 동작 후 gNB가 `AssertFatal((prach_id >= 0) && (prach_id < 8))`로 크래시.

**원인 분석**:

GPU IPC에서는 소켓과 달리 자연스러운 블로킹이 없어 RU 스레드가 L1-TX보다 훨씬
빠르게 진행됨. 이 속도 차이로 인해:

1. **PRACH 리스트 오버플로**: L1-TX가 MAC 스케줄러를 통해 `nr_fill_prach_ru()`를
   호출하여 PRACH 항목을 등록하지만, RU가 이미 해당 슬롯을 지나쳐 `free_nr_ru_prach_entry()`로
   해제하지 못함. 미해제 항목이 SFN 랩(10.24초)까지 누적되어 리스트(용량 8) 오버플로.

2. **Assertion 상수 버그**: `nr_fill_prach_ru`의 assertion이 `NUMBER_OF_NR_PRACH_MAX`(gNB용, 8)를
   사용했으나, 실제 배열은 `NUMBER_OF_NR_RU_PRACH_MAX`(RU용, 8)로 인덱싱 — 의미상 다른 상수.

3. **L1_tx_out 큐 무제한**: RU → L1-TX 사이의 `notifiedFIFO`에 백프레셔가 없어
   RU가 무제한으로 앞서감.

**해결 (A+B-light+C, 3단계)**:

| 단계 | 수정 | 파일 | 내용 |
|------|------|------|------|
| **A** | 버퍼 확장 + assertion 수정 | `defs_RU.h`, `nr_prach.c` | `NUMBER_OF_NR_RU_PRACH_MAX` 8→64, assertion 상수 교정 |
| **B-light** | RU-L1TX 갭 제한 | `defs_gNB.h`, `nr-ru.c`, `nr-gnb.c` | 원자 카운터 `ru_push_count`/`l1tx_done_count`로 갭 ≥ `sl_ahead-2`(=4)이면 RU `usleep(200)` |
| **C** | 과거 슬롯 PRACH skip | `nr_prach.c` | `nr_fill_prach_ru` 진입 시 RU의 `(frame_rx, tti_rx)` 원자 로드 → 대상 슬롯이 과거이면 등록 skip |

모든 OAI 수정은 `#ifdef USE_GPU_IPC_V2` 조건부로, 소켓 모드에 영향 없음.

### 문제 해결의 소켓 모드 대비 설계 근거

소켓 모드에서는 `epoll_wait` 블로킹이 자연스러운 동기화를 제공하여 RU가 L1-TX를
앞서가지 못함. GPU IPC에서는 이 블로킹이 없으므로 B-light 페이싱이 소켓의
암묵적 백프레셔를 명시적으로 재현. 단, 소켓의 "완전 동기"가 아닌 "제한적 비동기"
(최대 4슬롯 앞서기 허용)로 구현하여 파이프라인 병렬성 유지.

## 실험 결과

### 테스트 환경

- 서버: 128 코어 CPU, NVIDIA GPU
- OAI: gNB 1대 + UE 2대 (SISO)
- 채널: Sionna 통계 채널 (TDL), TDD FR1 Band 78, 106 PRB, SCS 30kHz
- 통신: GPU IPC V2 (SPSC 링버퍼, depth=4)

### G0 v12 (1 UE) vs G1A v1 (2 UE) 성능 비교

| 항목 | G0 v12 (1 UE) | G1A v1 (2 UE) |
|------|---------------|---------------|
| DL+UL 슬롯 비율 | 10D+0U (UL 거의 미처리) | 5D+5U (정상 밸런스) |
| wall/slot | 0.48~0.65 ms | 2.43~3.33 ms |
| proxy/slot | 0.45~0.62 ms | 2.39~3.29 ms |
| ipc+oai/slot | 0.02~0.05 ms | 0.04~0.06 ms |
| wall/frame (10 slots) | ~5~6.5 ms | ~24~33 ms |
| DL throughput | ~167 DL/sec | ~20~24 DL/sec |
| 연속 실행 | ~560 프레임 후 PRACH 크래시 | **1250+ 프레임 안정** |

### 분석

**속도 차이의 원인**:
- G0 v12는 B-light 페이싱 없이 RU가 자유 질주 → 빠르지만 PRACH 크래시
- G1A v1은 B-light 페이싱(RU ≤ L1-TX + 4 slots)으로 안정성 확보
- Proxy 처리시간: 1 UE ~0.5ms → 2 UE ~2.5ms (채널 2배 + UL 합산)

**IPC+OAI 오버헤드는 UE 수에 거의 무관**:
- OAI 자체 처리: ~0.04ms/slot (전체의 ~1.6%)
- Proxy 처리: ~2.5ms/slot (전체의 ~98.4%)
- **병목은 100% Proxy** — OAI 측은 수십 UE까지도 여유

**DL/UL 밸런스 정상화**:
- G0 v12: `10D+0U` (UL 미처리 구간 빈번) → G1A: `5D+5U` (TDD 패턴 정상 반영)
- 링버퍼 + 페이싱으로 DL/UL 양방향 데이터 흐름이 균등하게 유지

**UE 수별 예상 스케일링** (Proxy가 병목이므로 선형에 가까움):

| UE 수 | proxy/slot (예상) | wall/frame (예상) | DL rate (예상) |
|--------|-------------------|-------------------|----------------|
| 1 UE | ~1.0~1.3 ms | ~12~15 ms | ~60~80/sec |
| 2 UE | ~2.5 ms (실측) | ~28 ms (실측) | ~23/sec (실측) |
| 4 UE | ~4~5 ms | ~45~55 ms | ~10~12/sec |
| 8 UE | ~8~10 ms | ~90~110 ms | ~5~6/sec |

### v4 통합 채널 생성 최적화 (예상)

v3 대비 v4의 구조적 이점:

| 항목 | v3 (per-UE) | v4 (통합) | 효과 |
|------|-------------|-----------|------|
| GPU 커널 런치 | N회/batch | 1회/batch | GPU launch overhead 제거 |
| TF graph tracing | N회 | 1회 | XLA 컴파일 시간 1/N |
| Python 스레드 | N개 | 1개 | GIL 경합 감소 |
| RingBuffer | N개 (FFT_SIZE) | 1개 (N_UE×FFT_SIZE) | 메모리 관리 단순화 |
| 채널 접근 | per-UE get_batch() | 슬롯당 1회 + GPU view slice | get_batch 호출 1/N |

### 향후 최적화 방향

1. **B-light threshold 조정**: Proxy가 빨라지면 threshold를 높여 파이프라인 병렬성 확대
2. **CUDA Graph 확장**: 멀티 UE 배치 처리를 단일 CUDA Graph로 캡처
3. **채널 계수 프리페치**: ChannelProducer 링버퍼에서 다음 슬롯 계수를 미리 GPU에 로드

## 실행

### 소켓 모드

```bash
# Proxy (2 UE)
python v4_multi_ue.py --mode socket --num-ues 2 --custom-channel

# gNB
sudo ./nr-softmodem -O gnb.conf --gNBs.[0].min_rxtxtime 6 --rfsim --rfsimulator.serverport 6017

# UE 0
sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 \
  --uicc0.imsi 001010000000001 --rfsim --rfsimulator.serverport 6018

# UE 1
sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 \
  --uicc0.imsi 001010000000002 --rfsim --rfsimulator.serverport 6018
```

### GPU IPC V2 모드

```bash
# Proxy
python v4_multi_ue.py --mode gpu-ipc --num-ues 2 --custom-channel

# gNB
sudo RFSIM_GPU_IPC_V2=1 ./nr-softmodem \
  -O gnb.conf --gNBs.[0].min_rxtxtime 6 --rfsim

# UE 0
sudo RFSIM_GPU_IPC_V2=1 RFSIM_GPU_IPC_UE_IDX=0 ./nr-uesoftmodem \
  -r 106 --numerology 1 --band 78 -C 3619200000 \
  --uicc0.imsi 001010000000001 --rfsim

# UE 1
sudo RFSIM_GPU_IPC_V2=1 RFSIM_GPU_IPC_UE_IDX=1 ./nr-uesoftmodem \
  -r 106 --numerology 1 --band 78 -C 3619200000 \
  --uicc0.imsi 001010000000002 --rfsim
```

순서: Proxy → gNB → UE 0 → UE 1 (반드시 순서대로)

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--num-ues N` | 2 | UE 수 (1~8) |
| `--mode` | socket | socket / gpu-ipc |
| `--batch-mode` | on | 배치 FFT (N UE 동시 처리) |
| `--custom-channel` | on | Sionna 채널 활성화 |
| `--path-loss-dB` | 0 | 경로 손실 |
| `--snr-dB` | None | AWGN 노이즈 (None=off) |

## 파일 구조

```
G1A_MultiUE_Channel_Proxy/
  v4_multi_ue.py              # 최신: 통합 채널 생성 (1 Producer, 1 RingBuffer)
  v3_multi_ue.py              # DL+UL 배치 (per-UE producer)
  v2_multi_ue.py              # DL 배치 + per-UE 로깅
  v1_multi_ue.py              # 기본 다중 UE (보존)
  channel_coefficients_JIN.py # → 상위 루트 심링크
  launch_all.sh               # 통합 런처 (-v v1/v2/v3/v4 -m socket/gpu-ipc -n NUM)
  README.md
  실행_매뉴얼.txt

openairinterface5g_whan/radio/rfsimulator/
  gpu_ipc_v2.h                # Multi-UE SHM 레이아웃 (SPSC 링버퍼)
  gpu_ipc_v2.c                # Multi-UE IPC C 구현 (ring enqueue/dequeue)
  simulator.c                 # V2 IPC 통합 (#ifdef USE_GPU_IPC_V2)

openairinterface5g_whan/ (PRACH 크래시 수정, #ifdef USE_GPU_IPC_V2):
  openair1/PHY/defs_RU.h              # NUMBER_OF_NR_RU_PRACH_MAX 8→64
  openair1/PHY/defs_gNB.h             # ru_push_count, l1tx_done_count 추가
  openair1/PHY/NR_TRANSPORT/nr_prach.c # assertion 수정 + 과거슬롯 skip
  executables/nr-ru.c                  # RU-L1TX 갭 제한 (B-light)
  executables/nr-gnb.c                 # l1tx_done_count 증분
```

## OAI 빌드

```bash
cd openairinterface5g_whan/cmake_targets

# 초기 빌드 (gNB + UE 동시, GPU IPC V1/V2 플래그는 CMakeLists.txt에 포함)
./build_oai -w SIMU --ninja --gNB --nrUE

# 증분 빌드
cd ran_build/build
ninja nr-softmodem nr-uesoftmodem
```

`-c` (clean) 사용 시 build 폴더 전체 삭제되므로 gNB/UE 모두 재빌드 필요.

# G0: Sionna Channel Proxy for OAI 5G NR

## 프로젝트 개요

이 프로젝트는 **OpenAirInterface (OAI) 5G NR** 시뮬레이션에서 **Sionna 채널 모델**을 적용하기 위한 **Channel Proxy** 구현입니다.

### 동작 방식
- **컨테이너 외부**: OAI gNB 및 UE 실행 (`openairinterface5g_whan` 폴더)
- **컨테이너 내부**: Sionna Channel Proxy 실행 (이 폴더의 Python 스크립트)
- Proxy가 gNB ↔ UE 사이에서 채널 효과를 적용하여 신호 전달

```
┌─────────────────────────────────────────────────────────────────┐
│                    컨테이너 외부 (Host)                          │
│  ┌─────────┐                              ┌─────────┐           │
│  │   gNB   │ ←──── port 6017 ────────────→│   UE    │           │
│  │(터미널2)│                              │(터미널3)│           │
│  └─────────┘                              └─────────┘           │
│       ↑                                        ↑                │
└───────│────────────────────────────────────────│────────────────┘
        │              Socket 통신               │
┌───────│────────────────────────────────────────│────────────────┐
│       ↓                                        ↓                │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Sionna Channel Proxy                │           │
│  │                 (컨테이너 내부)                   │           │
│  │   gNB(6017) ←→ [Channel Model] ←→ UE(6018)      │           │
│  └──────────────────────────────────────────────────┘           │
│                    컨테이너 내부 (Docker)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## OAI 사용법

> **참고**: 빌드는 한 번만 실행하면 됩니다. 이후에는 gNB/UE 실행만 하면 됩니다.

### 터미널 1: OAI 빌드 (최초 1회)

```bash
cd ~/openairinterface5g_whan/cmake_targets
./build_oai -w SIMU --ninja --nrUE --gNB --build-lib "nrscope" -C
```

### 터미널 2: gNB 실행

```bash
cd && cd ~/openairinterface5g_whan/cmake_targets/ran_build/build &&
sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --gNBs.[0].min_rxtxtime 6 --rfsim --rfsimulator.serverport 6017
```

### 터미널 3: UE 실행

```bash
cd && cd ~/openairinterface5g_whan/cmake_targets/ran_build/build &&
sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --uicc0.imsi 001010000000001 --rfsim --rfsimulator.serverport 6018
```

### 터미널 4 (컨테이너 내부): Sionna Proxy 실행

```bash
python3 v8_cupy_slot_batch_pinned_fastest.py
```

---

## 🚀 속도 핵심 원인 분석

### 빠르게 만드는 요소

| 최적화 기법 | 효과 | 사용 버전 |
|------------|------|----------|
| **Full GPU Pipeline** | CPU 데이터 변환 제거 (~5.7ms→0) | v9 |
| **TF→CuPy DLPack** | 채널 H GPU-to-GPU (numpy 경유 제거) | v9.1 |
| **GPU RingBuffer** | 채널 H를 GPU 메모리에 저장 | v9.1 |
| **GPU AWGN** | CPU noise 생성 제거 (4.5ms→~0.01ms) | v9 |
| **int16 H2D/D2H** | IQ만 전송, 채널 H는 GPU 직접 | v9.1 |
| **GPU 인덱스 사전계산** | symbol extract/reconstruct CuPy 자동화 | v9 |
| **Pinned Memory** | PCIe 전송 2배 가속 | v6, v7, v8, v9 |
| **Batch FFT** | 14심볼 동시 처리 → 오버헤드 14배 감소 | v7, v8, v9 |
| **사전 계산된 채널** | 실시간 계산 제거 | v3~v9 |

### 느리게 만드는 요소

| 원인 | 영향 | 해당 버전 |
|------|------|----------|
| **CPU 데이터 변환** | IQ deinterleave + int16 변환 = 5.7ms/slot | v8.1 이하 |
| **numpy 경유 채널 H** | TF→numpy→CuPy (GPU→CPU→GPU) 왕복 | v9.0 이하 |
| **CPU AWGN (np.random)** | 4.5ms/slot | v8.1 이하 |
| **실시간 Sionna 채널 생성** | 100배 이상 느림 | v2, v4_realtime |
| **심볼 단위 파이프라인** | 오버헤드 누적 → 3배 느림 | v6 |
| **TensorFlow FFT** | CuPy 대비 느림 | v2~v5 |
| **Pinned Memory 미사용** | PCIe 전송 2배 느림 | v2~v5 |

---

## 📁 파일 목록

| 파일명 | 성능 | GPU | 주요 특징 |
|--------|:----:|:---:|----------|
| `v2_tf_realtime_slow.py` | ❌ 에러 | TF | 실시간 Sionna (GPU 충돌) |
| `v2_tf_realtime_slow_fix.py` | ❌ 에러 | TF | v2 버그 수정 |
| `v3_tf_precomputed.py` | ~73ms | TF | 사전 계산 채널 |
| `v4_tf_ringbuffer.py` | ~85ms | TF | RingBuffer 비동기 |
| `v4_tf_ringbuffer_copy.py` | ~92ms | TF | v4 복사본 |
| `v4_tf_realtime_very_slow.py` | ⚠️ ~1.3초 | TF | 실시간 Sionna (매우 느림) |
| `v4_tf_realtime_very_slow_copy.py` | ❌ 에러 | TF | v4_realtime 복사본 |
| `v5_tf_precomputed_moving.py` | ~73ms | TF | 사전 계산 + 이동 RX |
| `v6_cupy_stream_per_symbol_slow.py` | ⚠️ ~130ms | CuPy | 심볼 단위 파이프라인 (오버헤드) |
| `v7_cupy_batch_pinned_fast.py` | 🥈 ~47ms | CuPy | Batch FFT + Pinned Memory |
| `v8_cupy_slot_batch_pinned_fastest.py` | 🥈 ~41ms | CuPy | Slot Pipeline + Batch + Pinned |
| `v9_cupy_full_gpu_pipeline.py` | 🥇 **3.2ms** | TF+CuPy | Full GPU Pipeline (H100 실측, PL+AWGN 포함) |
| `v10_cuda_graph.py` | 🏆 **~1.1ms** | TF+CuPy | CUDA Graph + complex128 (H100 실측, v9 + 커널 런치 제거) |
| `v11_profile_instrumented.py` | 🏆 **~1.9ms** | TF+CuPy | v10 + E2E TDD 프레임 계측 (H100 실측, Proxy/Socket+OAI 분리, 10-slot 단위) |
| `v12_gpu_ipc.py` | 🏆🏆 **TBD** | TF+CuPy | CUDA IPC + CUDA Graph + v11 계측 이식 (듀얼 타이머, WindowProfiler, E2E TDD) |
| `v12_gpu_ipc_old.py` | - | TF+CuPy | v12 측정 업그레이드 전 원본 백업 (v10 수준 inline print) |

---

## 📊 성능 순위

### 테스트 환경
- **GPU**: NVIDIA TITAN RTX (22GB VRAM)
- **측정 기준**: Δwall (10 DL 처리 시간)
- **실시간 요구사항**: ~12ms (10슬롯 × 1ms)

| 순위 | 파일명 | 평균 Δwall | 핵심 요인 |
|:----:|--------|:----------:|----------|
| 🏆🏆 | `v12_gpu_ipc.py` | **~0.56ms** | CUDA IPC + CUDA Graph + 듀얼 타이머 계측 (소켓 제거, GPU-to-GPU, H2D/D2H 제거, H100 실측) |
| 🏆 | `v10_cuda_graph.py` | **~1.1ms** | CUDA Graph + complex128 (H100 실측, v9 + 커널 런치 제거) |
| 🥇 | `v9_cupy_full_gpu_pipeline.py` | **3.2ms** | Full GPU Pipeline + 배치 최적화 (H100 실측) |
| 🥈 | `v8_cupy_slot_batch_pinned_fastest.py` | **~41ms** | Synchronous Batch + Pinned |
| 3 | `v7_cupy_batch_pinned_fast.py` | **~47ms** | Batch FFT + Pinned Memory |
| 3 | `v3_tf_precomputed.py` | ~73ms | 사전 계산 채널 |
| 3 | `v5_tf_precomputed_moving.py` | ~73ms | 사전 계산 + 이동 RX |
| 5 | `v4_tf_ringbuffer.py` | ~85ms | RingBuffer |
| 6 | `v4_tf_ringbuffer_copy.py` | ~92ms | RingBuffer 복사본 |
| ⚠️ 7 | `v6_cupy_stream_per_symbol_slow.py` | ~130ms | 심볼 단위 오버헤드 |
| ⚠️ 8 | `v4_tf_realtime_very_slow.py` | ~1.3초 | 실시간 Sionna 생성 |

### H100 NVL 실측 성능 (v9.2)

- **GPU**: NVIDIA H100 NVL (94GB HBM3)
- **측정 조건**: `--path-loss-dB -3 --snr-dB 15` (PL 3dB 감쇠 + AWGN SNR 15dB)
- **측정 구간**: 슬롯 #5600 ~ #10500 (약 50+ 샘플)

**GPU 처리 단계별 소요 시간 (슬롯 평균)**:

| 단계 | 평균 | 최소 | 최대 | 설명 |
|------|:----:|:----:|:----:|------|
| H2D | 0.26ms | 0.12ms | 0.67ms | int16 IQ → pinned → GPU (120KB) |
| CH_COPY | 0.35ms | 0.23ms | 0.55ms | 채널 H 2D slice assign |
| DEINTLV | 0.53ms | 0.34ms | 0.64ms | int16 → float64 → complex128 |
| FFT | 0.54ms | 0.37ms | 0.68ms | Batch FFT convolution (14 symbols) |
| RECON | 0.36ms | 0.29ms | 0.59ms | symbol reconstruct + CP insert |
| PL+AWGN | 0.63ms | 0.53ms | 1.44ms | Path Loss 스칼라곱 + GPU noise |
| INT16+D2H | 0.57ms | 0.53ms | 0.68ms | clip → int16 → D2H (120KB) |
| **TOTAL** | **3.23ms** | **2.88ms** | **4.00ms** | **전체 GPU 파이프라인** |

**OFDM 슬롯 레벨** (채널 읽기 + GPU 처리):

| 단계 | 평균 | 설명 |
|------|:----:|------|
| CH_GET | 0.17ms | RingBuffer `get_batch(14)` |
| GPU_PROC | 3.12ms | `process_slot` 전체 |
| **TOTAL** | **3.30ms** | **슬롯당 총 처리 시간** |

**실시간성 분석**:

| 항목 | 값 |
|------|:----:|
| 실시간 예산 | 1.0ms/slot (30kHz SCS) |
| 실측 평균 | 3.23ms/slot |
| 실시간 대비 | **3.2x 초과** |
| 주요 병목 | CuPy 커널 런치 오버헤드 (~15 커널 × ~0.1ms) |

### H100 NVL 실측 성능 (v10 CUDA Graph)

- **GPU**: NVIDIA H100 NVL (94GB HBM3)
- **측정 조건**: `--path-loss-dB -3 --snr-dB 15` (PL 3dB 감쇠 + AWGN SNR 15dB)
- **측정 구간**: 슬롯 #100 ~ #10600 (약 100+ 샘플)
- **CUDA Graph**: 워밍업 3슬롯 후 캡처, 이후 `graph.launch()` 실행

**GPU 처리 단계별 소요 시간 (슬롯 평균, CUDA Graph 모드)**:

| 단계 | 평균 | 최소 | 최대 | 설명 |
|------|:----:|:----:|:----:|------|
| H2D | 0.19ms | 0.11ms | 0.34ms | int16 IQ → pinned → GPU (120KB) |
| CH_COPY | 0.31ms | 0.19ms | 0.45ms | 채널 H 2D slice assign |
| GPU_COMPUTE | 0.47ms | 0.40ms | 0.58ms | **CUDA Graph replay** (deintlv+FFT+PL+AWGN+int16) |
| D2H | 0.14ms | 0.08ms | 0.20ms | GPU → pinned → int16 (120KB) |
| **TOTAL** | **1.11ms** | **0.99ms** | **1.28ms** | **전체 GPU 파이프라인** |

**OFDM 슬롯 레벨** (채널 읽기 + GPU 처리):

| 단계 | 평균 | 설명 |
|------|:----:|------|
| CH_GET | 0.14ms | RingBuffer `get_batch(14)` |
| GPU_PROC | 1.04ms | `process_slot` 전체 (CUDA Graph) |
| **TOTAL** | **1.20ms** | **슬롯당 총 처리 시간** |

**v9.2 → v10 성능 비교**:

| 항목 | v9.2 (일반 실행) | v10 (CUDA Graph) | 개선 |
|------|:---:|:---:|:---:|
| H2D | 0.26ms | 0.19ms | 1.4x |
| CH_COPY | 0.35ms | 0.31ms | 1.1x |
| GPU 연산 (DEINTLV~INT16) | 2.63ms | 0.47ms | **5.6x** |
| D2H | - (INT16+D2H 합산) | 0.14ms | - |
| **슬롯 TOTAL** | **3.23ms** | **1.11ms** | **2.9x** |
| **OFDM TOTAL** | **3.30ms** | **1.20ms** | **2.75x** |

> **핵심**: CUDA Graph가 6개 GPU 커널(DEINTLV, FFT, RECON, PL, AWGN, INT16)을 단일 graph.launch()로 통합하여
> 커널 런치 오버헤드 ~1.5ms를 제거. 실제 GPU 연산 시간만 남음.

**실시간성 분석 (v10)**:

| 항목 | 값 |
|------|:----:|
| 실시간 예산 | 1.0ms/slot (30kHz SCS) |
| 실측 평균 | 1.20ms/slot |
| 실시간 대비 | **1.2x 초과** (거의 실시간 근접) |
| v9.2 대비 개선 | **2.75x** (3.30ms → 1.20ms) |
| 추가 개선 여지 | RawKernel 퓨전 시 실시간 달성 가능 |

### H100 NVL 실측 성능 (v11 E2E TDD 프레임 계측)

- **GPU**: NVIDIA H100 NVL (94GB HBM3)
- **측정 조건**: `--path-loss-dB -3 --snr-dB 3` (PL 3dB 감쇠 + AWGN SNR 3dB)
- **측정 구간**: TDD frame #600 ~ #3430 (284 samples, 10-slot TDD 프레임 단위)
- **CUDA Graph**: 워밍업 3슬롯 후 캡처, 이후 `graph.launch()` 실행
- **v11 계측**: TDD 프레임(10 total slots) 단위 Δwall vs Proxy(DL+UL) → Socket+OAI 시간 추정

**v11 계측 구조**:

```
TDD 프레임 (10 total slots, DL+UL 혼합)
├── Proxy DL 처리 (gNB→UE 채널처리 + send)  ← PROXY_E2E(dir=gNB) 누적
├── Proxy UL 처리 (UE→gNB 채널처리 + send)  ← PROXY_E2E(dir=UE) 누적
└── Socket + OAI (추정)                     ← Δwall - Proxy(DL+UL)
    ├── gNB 내부 슬롯 처리
    ├── UE 내부 슬롯 처리
    ├── 소켓 수신 대기 (select)
    └── recv + 파싱 (read_blocks)

OAI TDD 설정: 5ms 주기 (DL 7슬롯 + Special 1슬롯 + UL 2슬롯)
10-slot 윈도우가 TDD 주기와 비동기 → 프레임마다 DL/UL 비율 변동
```

**E2E 시간 분해 (TDD 프레임 단위)**:

| 항목 | 10-slot 합계 | per slot 평균 | 시간 비율 |
|------|:---:|:---:|:---:|
| **Δwall** | 22.15ms | **2.21ms** | 100% |
| **Proxy (DL+UL)** | 20.07ms | **2.01ms** | **90.6%** |
| **Socket+OAI** | 2.08ms | **0.21ms** | **9.4%** |

> Socket+OAI가 0.21ms/slot로 매우 작음 → 전체 시간의 90%가 Proxy GPU 처리

**DL/UL Proxy 처리시간 비교**:

| 방향 | per slot 평균 | 비고 |
|------|:---:|------|
| DL (gNB→UE) | 2.63ms | 채널 적용 + send (데이터 크기 더 큼) |
| UL (UE→gNB) | 1.59ms | 채널 적용 + send |

**전체 통계 (N=284)**:

| 항목 | 평균 | 최소 | p50 | p95 | p99 | 최대 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Δwall (10-slot) | 22.15ms | 14.81ms | 20.64ms | 29.99ms | 48.85ms | 54.01ms |
| Proxy(DL+UL) | 20.07ms | 13.14ms | 18.68ms | 28.07ms | 46.54ms | 49.76ms |
| Socket+OAI | 2.08ms | 1.53ms | 1.87ms | 3.98ms | 4.25ms | 4.59ms |
| wall/slot | 2.21ms | 1.48ms | 2.06ms | 3.00ms | 4.89ms | 5.40ms |
| proxy/slot | 2.01ms | 1.31ms | 1.87ms | 2.81ms | 4.65ms | 4.98ms |
| sock+oai/slot | 0.21ms | 0.15ms | 0.19ms | 0.40ms | 0.43ms | 0.46ms |

**Outlier 분석 (Δwall > 30ms)**:

| 구분 | 수 | 비율 | 비고 |
|------|:---:|:---:|------|
| Outlier (spike) | 14 | 4.9% | GPU spike (CUDA 스케줄링/메모리) |
| Normal | 270 | 95.1% | 안정 구간 |

> Outlier 특징: Proxy 처리 spike (max 54.0ms)가 원인, Socket+OAI는 정상 (1.9~4.6ms)

**Normal 프레임 통계 (Δwall ≤ 30ms, N=270)**:

| 항목 | 평균 | p50 | p95 | p99 | 최대 |
|------|:---:|:---:|:---:|:---:|:---:|
| Δwall (10-slot) | 21.09ms | 20.55ms | 27.65ms | 29.80ms | 29.99ms |
| wall/slot | 2.11ms | 2.06ms | 2.77ms | 2.98ms | 3.00ms |
| proxy/slot | 1.90ms | 1.86ms | 2.56ms | 2.79ms | 2.82ms |
| sock+oai/slot | 0.21ms | 0.19ms | 0.40ms | 0.42ms | 0.43ms |

**PROXY_E2E 프로파일 (최근 안정 구간, 500-sample 롤링 윈도우)**:

| 단계 | 평균 | p95 | p99 | 최대 | 설명 |
|------|:----:|:---:|:---:|:----:|------|
| PROC | 1.69ms | 5.36ms | 9.81ms | 34.45ms | 채널 읽기 + GPU Pipeline 전체 |
| SEND | 0.18ms | 0.21ms | 0.25ms | 0.33ms | sock.sendall() |
| **TOTAL** | **1.88ms** | **5.57ms** | **10.03ms** | **34.69ms** | **Proxy 슬롯 처리 전체** |

**TDD 패턴별 성능 (10-slot 프레임)**:

| 패턴 | 비율 | Δwall 평균 | Proxy 평균 | Socket+OAI |
|------|:---:|:---:|:---:|:---:|
| 3D+7U | 12.3% | 23.2ms | 21.1ms | 2.1ms |
| 4D+6U | 42.3% | 24.1ms | 22.0ms | 2.1ms |
| 5D+5U | 34.2% | 20.2ms | 18.2ms | 2.1ms |
| 6D+4U | 11.3% | 19.4ms | 17.4ms | 2.0ms |

> DL/UL 비율이 프레임마다 다른 이유: Proxy는 size>1인 슬롯만 채널 처리하며,
> 10-slot 측정 윈도우가 TDD 5ms 주기(7DL+1S+2UL)와 정렬되지 않아 슬라이딩 위치에 따라 변동.
> Socket+OAI는 패턴과 무관하게 ~2.0ms로 일정 → OAI 처리시간은 안정적.

**Throughput 분석**:

| 기준 | Δwall/10slots | Throughput |
|------|:---:|:---:|
| Normal (spike 제외) | 21.09ms | **474 slots/sec** |
| 전체 (spike 포함) | 22.15ms | **452 slots/sec** |

### H100 NVL 실측 성능 (v12 GPU IPC + E2E TDD 프레임 계측)

- **GPU**: NVIDIA H100 NVL (94GB HBM3)
- **통신 방식**: CUDA IPC (Socket 완전 제거, GPU-to-GPU 직접 전송)
- **측정 조건**: `--path-loss-dB -3 --snr-dB 15 --mode=gpu-ipc`
- **측정 구간**: E2E frame #1860 ~ #5230 (337 frames, Ctrl-C 종료)
- **총 슬롯**: DL=4090, UL=1144 (TDD 비율 ~3.6:1)
- **CUDA Graph**: 워밍업 후 캡처, `graph.launch()` 실행
- **v12 계측**: E2E TDD 프레임(10 total slots) 단위, `_check_e2e_frame()` (idle 중복 출력 버그 수정)

**v12 계측 구조**:

```
TDD 프레임 (10 total slots, DL+UL 혼합)
├── Proxy DL 처리 (GPU Copy-In → Channel → GPU Copy-Out)  ← _ipc_process_dl() 누적
├── Proxy UL 처리 (GPU Copy passthrough)                   ← _ipc_process_ul() 누적
└── IPC + OAI (추정)                                       ← Δwall - Proxy(DL+UL)
    ├── gNB 내부 슬롯 처리 (L1/L2)
    ├── UE 내부 슬롯 처리
    ├── IPC spin-wait (dl_tx_ready/ul_tx_ready 폴링)
    └── mmap 동기화 오버헤드

※ Socket 제거 → H2D/D2H 불필요, recv/send/select 제거
```

**Phase 1: DL Only — UE 접속 전 (10D+0U)**:

| 항목 | 10-slot 합계 | per slot 평균 | 비고 |
|------|:---:|:---:|------|
| **Δwall** | ~13.3ms | **~1.33ms** | gNB만 DL 전송 |
| **Proxy** | ~6.3ms | **~0.63ms** | DL 채널 처리만 |
| **IPC+OAI** | ~7.0ms | **~0.70ms** | gNB 내부 + IPC 대기 |

> DL only 상태에서 IPC+OAI가 ~53%로 높은 이유: gNB가 UE 응답 없이 내부 L1/L2 처리를 완료해야 다음 슬롯을 전송하므로 대기 시간이 길어짐.

**Phase 2: DL+UL 정상 동작 — UE 접속 후 (steady state)**:

| 항목 | (6D+4U) per slot | (7D+3U) per slot | 비고 |
|------|:---:|:---:|------|
| **Δwall** | **~0.52ms** | **~0.60ms** | 전체 E2E |
| **Proxy** | **~0.48ms** | **~0.55ms** | GPU 처리 |
| **IPC+OAI** | **~0.04ms** | **~0.05ms** | IPC + gNB/UE 내부 |
| **IPC+OAI 비율** | ~7% | ~8% | **92~93%가 Proxy** |

> UE 접속 후 IPC+OAI가 0.02~0.05ms로 극적으로 감소. Socket 제거 + GPU 직접 전송 효과.

**주기적 Spike (채널 계수 재생성)**:

| Frame | wall | Proxy | 원인 |
|------|:---:|:---:|------|
| #2030 | 26.0ms | 25.8ms | DL=24.5ms spike |
| #2660 | 27.4ms | 27.1ms | DL=26.3ms spike |
| #3280 | 25.8ms | 25.6ms | UL=19.0ms spike |
| #3890 | 26.4ms | 25.9ms | UL=19.4ms spike |
| #4500 | 25.6ms | 25.3ms | UL=19.2ms spike |
| #5110 | 26.1ms | 25.9ms | DL=24.9ms spike |

> ~600 프레임(~6000슬롯) 간격으로 채널 버퍼 재생성 spike 발생. Proxy GPU 처리가 원인 (IPC+OAI 정상).

**Throughput 추이**:

| DL 슬롯 수 | 누적 Rate | 비고 |
|:---:|:---:|------|
| 1900 | 87.1 DL/s | DL only |
| 2000 | 91.2 DL/s | UE 접속 시작 |
| 2500 | 111.4 DL/s | DL+UL 안정화 |
| 3000 | 130.9 DL/s | |
| 3500 | 149.5 DL/s | |
| 4000 | 167.4 DL/s | 최종 측정 |

> 누적 rate이므로 초기 DL-only 구간이 포함되어 점진적으로 증가. 정상 구간 instant rate는 더 높음.

**v10 vs v11 vs v12 비교** (동일 GPU, 측정 방식 차이):

| 항목 | v10 (Socket) | v11 (Socket) | **v12 (GPU IPC)** | 비고 |
|------|:---:|:---:|:---:|------|
| 통신 방식 | TCP Socket | TCP Socket | **CUDA IPC** | v12에서 Socket 완전 제거 |
| GPU Pipeline | 1.11ms | - | - | v10 기준 GPU 연산만 |
| PROXY_E2E TOTAL | - | 1.88ms | - | OFDM + SEND + 오버헤드 |
| **Proxy/slot** | - | ~2.01ms | **~0.52ms** | **3.9x 개선** (H2D/D2H 제거) |
| Socket+OAI/slot | - | 0.21ms | - | v11 Socket 기준 |
| **IPC+OAI/slot** | - | - | **~0.04ms** | **5.3x 개선** (Socket 제거) |
| **wall/slot** | - | 2.21ms | **~0.56ms** | **3.9x 개선** |
| **Throughput** | - | 452 slots/sec | **167+ DL/s** (누적) | v12 instant rate 더 높음 |

> **v11→v12 핵심 개선**: Socket 제거로 H2D/D2H + send/recv/select 오버헤드 완전 제거.
> v11 Proxy=2.01ms → v12 Proxy=0.52ms (**3.9x**), v11 Socket+OAI=0.21ms → v12 IPC+OAI=0.04ms (**5.3x**).
> v11은 SNR=3dB, v12는 SNR=15dB로 noise 강도 차이 있음 (noise 강할수록 GPU AWGN 연산 부하 증가).

### TITAN RTX vs H100 NVL 비교

| 항목 | TITAN RTX (24GB) | H100 NVL (94GB) |
|------|:----------------:|:----------------:|
| v8 (Δwall 10 DL) | ~41ms | ~28ms |
| v8.1 per slot (AWGN 없음) | 미측정 | ~2.8ms |
| v8.1 per slot (CPU AWGN) | 미측정 | ~7.0ms |
| **v9.2 per slot (GPU PL+AWGN)** | 미측정 | **~3.2ms** |
| **v10 per slot (CUDA Graph)** | 미측정 | **~1.1ms** |
| **v11 PROXY_E2E per slot** | 미측정 | **~1.88ms** |
| **v11 wall/slot (TDD 프레임)** | 미측정 | **~2.21ms** |
| **v11 Socket+OAI per slot** | 미측정 | **~0.21ms** |
| **v11 Throughput** | 미측정 | **452 slots/sec** |
| **v12 Proxy/slot (GPU IPC)** | 미측정 | **~0.52ms** |
| **v12 wall/slot (GPU IPC)** | 미측정 | **~0.56ms** |
| **v12 IPC+OAI per slot** | 미측정 | **~0.04ms** |
| **v12 Throughput** | 미측정 | **167+ DL/s** (누적) |
| v8.1→v9.2 개선 (AWGN 포함) | - | **7.0→3.2ms (2.2x)** |
| v9.2→v10 개선 (CUDA Graph) | - | **3.2→1.1ms (2.9x)** |
| **v11→v12 개선 (GPU IPC)** | - | **2.21→0.56ms (3.9x)** |
| **v8.1→v12 총 개선** | - | **7.0→0.56ms (12.5x)** |

> **참고**: TITAN RTX 측정은 Δwall (10 DL 배치 시간), H100 측정은 per-slot 프로파일링.
> TITAN RTX에서 v9.2/v10/v11/v12는 아직 미측정.
> v12의 wall/slot은 TDD 프레임(10 total slots) E2E 계측 기준 (Proxy + IPC + OAI 전체 포함, UE 접속 후 정상 구간).

---

## 🔬 상세 분석

### v9 vs v8.1 vs v7 vs v6 비교 (모두 CuPy)

```
v6: 심볼 단위 파이프라인 (Pinned Memory만)
    → 매 심볼마다 H2D→Compute→D2H 사이클
    → 오버헤드 누적 → ~130ms (최악)

v7: Batch FFT + Pinned Memory (파이프라인 없음)
    → 14심볼을 한 번에 처리
    → 오버헤드 14배 감소 → ~47ms

v8.1: Synchronous Batch + Pinned Memory
    → 동기 처리 (delay=0), 하지만 CPU에서 데이터 변환
    → ~2.8ms/slot (CPU 병목: IQ변환 5.7ms + AWGN 4.5ms)

v9.0: Full GPU Pipeline (int16-to-int16)
    → 모든 데이터 변환/AWGN을 GPU에서 처리
    → H2D/D2H int16 (120KB, complex128 대비 4x 축소)
    → GPU 인덱스 사전계산 (symbol extract/reconstruct)
    → 채널 H는 여전히 TF→numpy→CuPy (GPU→CPU→GPU 왕복)

v9.1: numpy-free Pipeline (TF+CuPy+DLPack)
    → 채널 H: TF→DLPack→CuPy (GPU-to-GPU zero-copy)
    → RingBuffer: CuPy GPU 메모리 직접 저장
    → IQ bytes→pinned memory 직접 복사 (ctypes.memmove)
    → 채널 H2D 전송 완전 제거 (이미 GPU에 있음)
    → numpy 사용: pinned memory view + .npy 로딩만 (hot path 제거)

v9.2: 커널 배치 최적화 ← 현재 버전
    → RingBuffer.get_batch(14): 1회 lock + 1회 GPU slice copy (14x)
    → ChannelProducer: 배치 TF 정규화 + 1회 DLPack (~250→5 커널)
    → process_slot: 2D slice assign (~42→2 커널)
    → 총 GPU 커널/슬롯: ~65 → ~15 (4x 감소)
```

### Pinned Memory 효과

```
┌─────────────────────────────────────────────────────────────┐
│ 일반 메모리 (Pageable)                                       │
│   CPU Memory → [복사] → Staging Buffer → [DMA] → GPU        │
│   ※ 추가 복사 필요 → 느림                                    │
├─────────────────────────────────────────────────────────────┤
│ Pinned Memory (Page-locked)                                 │
│   CPU Memory ────────────────── [DMA 직접] ──────→ GPU      │
│   ※ 직접 전송 가능 → 2배 빠름                                │
└─────────────────────────────────────────────────────────────┘
```

### Batch FFT 효과

```
개별 처리 (14회 커널 런치):
  for i in range(14):
      gpu_fft(symbol[i])      ← 매번 커널 런치 오버헤드

배치 처리 (1회 커널 런치):
  gpu_fft(symbols[0:14])      ← 1번만 커널 런치, 14개 동시 처리
                               → 14배 효율적
```

---

## 🎯 핵심 교훈

### ✅ 해야 할 것

1. **Full GPU Pipeline**: 데이터 변환, noise 등 모든 처리를 GPU에서 수행
2. **DLPack GPU-to-GPU**: TF에서 생성한 채널 H를 CuPy로 GPU 메모리 직접 전달
3. **GPU RingBuffer**: 채널 H를 GPU 메모리에 저장하여 H2D 전송 제거
4. **int16 H2D/D2H**: IQ 데이터만 pinned memory로 전송 (채널 H는 GPU 직접)
5. **GPU 인덱스 사전계산**: symbol extract/reconstruct를 CuPy gather/scatter로
6. **Pinned Memory 사용**: PCIe 전송 2배 가속
7. **Batch 처리**: 여러 심볼을 모아서 한 번에 처리
8. **채널 사전 계산**: 실시간 생성 절대 금지

### ❌ 피해야 할 것

1. **numpy 경유 데이터 전달**: TF→numpy→CuPy는 GPU→CPU→GPU 왕복 (DLPack 사용)
2. **CPU 데이터 변환**: IQ deinterleave, int16 변환을 CPU에서 하면 ~5.7ms 낭비
3. **CPU AWGN (np.random)**: 4.5ms/slot, GPU cp.random은 ~0.01ms
4. **실시간 Sionna 채널 생성**: 100배 이상 느림
5. **심볼 단위 파이프라인**: 오버헤드가 이점을 상쇄
6. **TensorFlow FFT**: CuPy가 더 빠름

---

## 📡 채널 모델 신호처리 아키텍처

### 신호 체인

Proxy 내부에서 채널(multipath) + Path Loss + AWGN noise를 모두 적용한 뒤 int16으로 변환합니다.
이는 실제 RF 수신 체인 (signal + noise → ADC) 과 동일한 순서입니다.

**Version 9.2: Full GPU Pipeline + 커널 배치 최적화** — numpy-free

```
[TF/Sionna 스레드]
    ChannelProducer: 배치 TF 정규화 (벡터 연산, ~5 커널)
    │
    ├─ 1회 DLPack: TF batch → CuPy batch (GPU-to-GPU)
    │
    ├─ put_batch(): 1회 lock으로 N개 삽입
    │
    ▼ GPU RingBuffer (CuPy 배열, GPU 메모리)

[메인 스레드]
gNB Tx (int16 bytes)
    │
    ├─ ctypes.memmove → pinned memory → H2D (120KB)
    │
    ▼ ═══════════════════ GPU ═══════════════════
    │
    ├─ int16 → float64 → complex128 (IQ deinterleave, CuPy)
    │
    ├─ 채널 H ← get_batch(14): 1회 lock + 1회 GPU slice copy
    │
    ├─ 채널 복사: 2D slice assign (for-loop 제거, ~42→2 커널)
    │
    ├─ Symbol extraction (CuPy 사전계산 인덱스 gather)
    │
    ├─ Batch FFT Convolution (14 symbols 동시, CuPy)
    │  h_normalized = h / sqrt(sum|h|²)    ← TF에서 이미 정규화됨
    │  y = IFFT( FFT(x) * FFT(h) )
    │
    ├─ Reconstruct (CuPy 사전계산 인덱스 scatter, CP 삽입)
    │
    ├─ Path Loss: out *= pathLossLinear     ← 10^(path_loss_dB / 20.0)
    │
    ├─ AWGN noise (cp.random.randn, 옵션)
    │
    ├─ clip(-32768, 32767) → int16 (CuPy)
    │
    ▼ ═══════════════════════════════════════════
    │
    ▼ ─── D2H (120KB, pinned memory) ───
    │
    ▼
OAI UE (rfsimulator channel_mode=2, bypass)
```

**비교: v9.0 vs v9.1 (numpy 제거)**

| | v9.0 (numpy 경유) | v9.1 (numpy-free) |
|---|---|---|
| 채널 H 전달 | TF→numpy→CuPy (GPU→CPU→GPU) | TF→DLPack→CuPy (GPU→GPU) |
| RingBuffer | numpy CPU 배열 | **CuPy GPU 배열** |
| 채널 H2D | pinned memory → GPU (458KB) | **불필요 (이미 GPU)** |
| IQ H2D | pinned memory → GPU (120KB) | pinned memory → GPU (120KB) |
| IQ bytes 처리 | np.frombuffer → pinned | **ctypes.memmove → pinned** |
| 에너지 정규화 | numpy CPU | **TF GPU** |
| 인덱스 사전계산 | numpy → cp.asarray | **CuPy 직접** |
| numpy 의존성 | 전체 파이프라인 | **pinned view + .npy만** |

**비교: v9.1 vs v9.2 (커널 배치 최적화)**

| | v9.1 (개별 커널) | v9.2 (배치) |
|---|---|---|
| RingBuffer 읽기 | 14 lock + 14 GPU copy | **1 lock + 1 GPU slice copy** |
| 채널 복사 (process_slot) | for-loop ~42 커널 | **2D slice ~2 커널** |
| ChannelProducer (per batch) | 원소별 ~250 커널 | **배치 ~5 커널** |
| **총 GPU 커널/슬롯** | **~65** | **~15 (4x 감소)** |

**비교: v8.1 (CPU 변환) vs v9.2 (Full GPU + 배치)**

| | v8.1 (CPU 변환) | v9.2 (Full GPU) |
|---|---|---|
| IQ deinterleave | CPU numpy (0.36ms) | GPU CuPy (~0ms) |
| symbol extract | CPU for-loop (0.26ms) | GPU 인덱스 gather (~0ms) |
| FFT convolution | GPU CuPy (0.05ms) | GPU CuPy (0.05ms) |
| reconstruct | CPU for-loop (0.33ms) | GPU 인덱스 scatter (~0ms) |
| AWGN noise | CPU numpy (4.5ms) | GPU cp.random (~0.01ms) |
| int16 변환 | CPU numpy (0.6ms) | GPU CuPy (~0ms) |
| 채널 H 전달 | GPU→CPU→GPU (458KB×2) | **GPU→GPU (0 bytes)** |
| 채널 읽기 | 14 lock + 14 copy | **1 lock + 1 copy** |
| H2D 전송량 | 480KB+458KB | **120KB (IQ만)** |
| D2H 전송량 | 480KB (complex128) | **120KB (int16)** |
| **합계/slot** | **~2.8ms** (AWGN 시 ~7ms) | **~3.2ms** (PL+AWGN 포함, H100 실측) |

**비교: OAI rfsimulator 내장 채널 vs Sionna Proxy**

| | OAI rfsimulator (chanmod) | Sionna Proxy (v9) |
|---|---|---|
| 채널 모델 | 3GPP TDL (간단) | Sionna 3GPP TR 38.901 (상세) |
| noise 추가 시점 | int16 → double → noise → int16 (양자화 2번) | GPU complex128 → noise → int16 (양자화 1번) |
| PL 적용 | convolution 후 스칼라곱 | 동일 |
| 데이터 변환 | CPU | **전체 GPU** |
| 런타임 SNR 변경 | telnet 가능 | CLI 인자 (`--snr-dB`) |

### Path Loss (PL) 적용 방식

**OAI 규약**을 따릅니다 (전압 도메인, `/20`):

| path_loss_dB | pathLossLinear | 효과 |
|:---:|:---:|---|
| -10 | 0.316 | 10 dB 감쇠 |
| -3 | 0.708 | 3 dB 감쇠 |
| **0** (기본) | **1.0** | **무손실** (채널 효과만) |
| +3 | 1.413 | 3 dB 증폭 |

**수정 전 문제점**:

기존 코드는 PL을 채널 계수 안에 포함시키고, 부호 규약도 OAI와 반대였습니다.

```python
# 수정 전 (문제)
PathLoss_dB = -10                                         # 파워 도메인 /10, 부호 반대
PathLoss = 10**(PathLoss_dB/10)                           # = 0.1
h_normalized = h / np.sqrt(np.sum(np.abs(h)**2)) / np.sqrt(PathLoss)  # 3.16x 증폭!
# → int16 clipping → Zadoff-Chu preamble 파괴 → PRACH 검출 실패

# 수정 후 (정상)
path_loss_dB = 0                                          # OAI 규약, 전압 도메인 /20
pathLossLinear = 10**(path_loss_dB / 20.0)                # = 1.0
h_normalized = h / np.sqrt(np.sum(np.abs(h)**2))          # 에너지 정규화만
out *= pathLossLinear                                      # convolution 후 별도 적용
```

### AWGN Noise 적용 방식

Noise는 convolution + PL 적용 후, int16 변환 전에 추가됩니다.
Proxy에서 채널과 noise를 함께 관리하면:

1. **양자화 1번**: complex128에서 noise 추가 → int16 변환 (실제 ADC와 동일)
2. **정밀 SNR 제어**: convolution 직후 signal power를 정확히 알 수 있음
3. **OAI 수정 불필요**: rfsimulator는 channel_mode=2 (bypass) 그대로 사용

```python
# SNR 기반 AWGN 수식 (GPU, CuPy)
sig_pwr = cp.mean(cp.abs(gpu_out)**2)
n_pwr = sig_pwr / 10**(snr_dB / 10)
n_std = cp.sqrt(n_pwr / 2)
gpu_out += n_std * (cp.random.randn(N) + 1j * cp.random.randn(N))  # GPU 난수 생성
```

**UE 측정값에 대한 영향**:

UE는 noise 출처와 무관하게 수신 신호에서 직접 측정합니다.

| 측정값 | noise 없을 때 | noise 있을 때 |
|--------|:---:|:---:|
| RSRP | 채널 gain만 반영 | 동일 (약간의 noise bias) |
| SINR | 매우 높음 (60~90 dB) | 현실적 (SNR에 따라 변동) |
| CQI | 항상 15 (최대) | SNR에 따라 가변 |
| PMI | multipath에 의해 결정 | 동일 |
| RI | 최대 rank | SNR에 따라 가변 |

### 발견된 문제와 해결

**문제**: 채널 적용 시 PRACH 검출 실패 (UE가 RAR 수신 반복 실패)

| 단계 | 내용 |
|------|------|
| 증상 | UE: SSB 동기화 성공, PRACH 전송 확인 → gNB: PRACH detection 로그 없음 |
| 확인 | `--no-ch-en` (채널 비활성화) → RACH 정상 성공 |
| 원인 1 | `PathLoss_dB = -10` → proxy가 `1/sqrt(0.1) = 3.16x` 증폭으로 해석 |
| 원인 2 | PL이 채널 계수 안에 포함 → convolution 자체가 신호를 3.16배 증폭 |
| 원인 3 | 증폭된 신호가 int16 범위 초과 → clipping → ZC preamble constant envelope 파괴 |
| 해결 | PL을 채널 밖으로 분리, OAI 규약(`/20`, 음수=감쇠)으로 통일, 기본값 0 dB |

### CLI 옵션 레퍼런스 (v8)

| 옵션 | 기본값 | 설명 |
|------|:------:|------|
| `--ue-port` | 6018 | UE 수신 포트 |
| `--gnb-host` | 127.0.0.1 | gNB 주소 |
| `--gnb-port` | 6017 | gNB 포트 |
| `--ch-en` / `--no-ch-en` | 활성화 | 채널 convolution on/off |
| `--custom-channel` / `--no-custom-channel` | 활성화 | Sionna 채널 모델 on/off |
| `--path-loss-dB` | 0.0 | Path Loss (OAI 규약: 음수=감쇠, 0=무손실) |
| `--snr-dB` | None | AWGN noise SNR (None=noise 없음) |
| `--enable-gpu` / `--disable-gpu` | 활성화 | GPU 가속 on/off |
| `--use-pinned-memory` / `--no-pinned-memory` | 활성화 | Pinned Memory on/off |
| `--buffer-len` | 42000 | 채널 버퍼 크기 |
| `--buffer-symbol-size` | 4200 | 배치당 생성 심볼 수 |

---

## 💡 권장 사용법

```bash
# v12 GPU IPC 모드 (소켓 제거, 듀얼 타이머 + WindowProfiler + E2E TDD 통계)
python3 v12_gpu_ipc.py --mode=gpu-ipc

# v12 GPU IPC + PL 감쇠 + noise
python3 v12_gpu_ipc.py --mode=gpu-ipc --path-loss-dB -3 --snr-dB 3

# v12 GPU IPC + 프로파일링 간격/윈도우 조정
python3 v12_gpu_ipc.py --mode=gpu-ipc --profile-interval 50 --profile-window 200

# v12 GPU IPC + CUDA Event 비교 비활성화 (CPU+sync만)
python3 v12_gpu_ipc.py --mode=gpu-ipc --no-dual-timer-compare

# v12 소켓 모드 (v10 호환, 듀얼 타이머 계측 포함)
python3 v12_gpu_ipc.py --mode=socket

# 최고 성능 + E2E 계측 (v11) - CUDA Graph + Proxy/Socket+OAI 분리 측정
python3 v11_profile_instrumented.py

# v11 + PL 감쇠 + noise
python3 v11_profile_instrumented.py --path-loss-dB -3 --snr-dB 3

# v11 프로파일링 간격/윈도우 조정
python3 v11_profile_instrumented.py --profile-interval 50 --profile-window 200

# v10 (v11 이전 버전) - CUDA Graph + complex128 Pipeline
python3 v10_cuda_graph.py

# v10 + noise (SNR 20 dB)
python3 v10_cuda_graph.py --snr-dB 20

# v10 + PL 감쇠 + noise
python3 v10_cuda_graph.py --path-loss-dB -3 --snr-dB 15

# v10 채널 없이 bypass (디버깅용)
python3 v10_cuda_graph.py --no-ch-en

# 이전 버전 (v9) - Full GPU Pipeline (CUDA Graph 없음)
python3 v9_cupy_full_gpu_pipeline.py

# v8 (이전 버전) - CPU 데이터 변환
python3 v8_cupy_slot_batch_pinned_fastest.py

# 단순하고 빠름 (v7) - 디버깅 용이
python3 v7_cupy_batch_pinned_fast.py

# TensorFlow 필요시 (v3 또는 v5)
python3 v3_tf_precomputed.py
python3 v5_tf_precomputed_moving.py
```

---

## 🔮 Future Work

### 1. ✅ CUDA Graph (커널 런치 오버헤드 제거) — v10 구현 완료

**v9.2 병목**: 슬롯당 ~15개 CuPy 커널을 개별 런치 → 커널당 ~0.1ms 오버헤드 누적 → ~1.5ms 낭비

**CUDA Graph 개념**: GPU 연산 시퀀스를 한 번 캡처한 후, 단일 CPU 호출로 전체를 재생

```
[일반 실행]  CPU→커널1(0.1ms) → CPU→커널2(0.1ms) → ... → CPU→커널15(0.1ms)
             = GPU 연산 + 커널 런치 15회 = 총 ~3.2ms

[CUDA Graph] CPU→graph.launch(0.015ms) → GPU가 15커널 자체 실행
             = GPU 연산만 = 총 ~1.7ms (예상)
```

**CuPy 적용 방법**:

```python
# 캡처 (최초 1회)
stream.begin_capture()
# int16→cpx→FFT conv→PL→AWGN→clip→int16 전체 기록
graph = stream.end_capture()

# 실행 (매 슬롯) — 입출력 버퍼 주소 고정, 데이터만 memcpy 후 replay
graph.launch(stream)
```

**적용 가능성**: `process_slot`은 매 슬롯 동일 shape (14×4096), 동일 연산 → **이상적**

| 제약 | v9.2 상황 | 충족 여부 |
|------|----------|:--------:|
| 고정 텐서 shape | 14×4096 항상 동일 | ✅ |
| 고정 연산 흐름 | if 분기 없음 (표준 슬롯) | ✅ |
| 고정 메모리 주소 | gpu_iq_in/out, gpu_H 재사용 | ✅ |
| CPU 개입 불가 | Python 코드 없음 (GPU 전용) | ✅ |

**예상 효과**: 3.2ms → **~1.7ms** (커널 런치 ~1.5ms 제거)

**실측 결과 (v10)**: 3.23ms → **1.11ms** (예상보다 더 좋음, 2.9x 개선)

### 2. tf.function + XLA vs CUDA Graph 비교

| 비교 항목 | CUDA Graph | tf.function + XLA |
|-----------|-----------|-------------------|
| 최적화 수준 | 런타임 (런치 순서 재생) | 컴파일러 (커널 퓨전/최적화) |
| 커널 수 변화 | 불변 (15→15) | **감소** (15→3~5 퓨전) |
| 메모리 트래픽 | 불변 | **감소** (중간 텐서를 레지스터에 유지) |
| 런치 오버헤드 | **극적 감소** (15회→1회) | 감소 (퓨전으로 커널 수 자체 감소) |
| 프레임워크 | CuPy/CUDA 전체 | TensorFlow/JAX 전용 |
| 적용 난이도 | **중간** | 높음 (CuPy→TF 전환 필요) |

```
비유:
  CUDA Graph  = "녹음된 지휘 동작을 그대로 재생" → 지휘(런치) 횟수는 같지만 지휘자(CPU)가 쉼
  XLA         = "악보를 다시 편곡" → 연주 파트(커널) 수 자체가 줄어듦
```

**결론**: 현재 CuPy 파이프라인 유지 시 **CUDA Graph가 가장 현실적**. TF로 전환 시 XLA가 추가 이점 제공.

### 3. ~~H 채널 정밀도 최적화 (complex128 → complex64)~~ — ❌ 검증 실패

**v10에서 실제 테스트한 결과, complex64로는 UE PSS/SSS 동기화가 실패합니다.**

| 항목 | complex128 | complex64 | 비고 |
|------|:---:|:---:|:---:|
| FFT 처리 속도 | 1x | ~2x | 메모리 대역폭 절반 |
| GPU 메모리 사용 | 14×4096×16 = 896KB | 14×4096×8 = 448KB | 2x 절약 |
| mantissa 비트 | 52비트 (float64) | 23비트 (float32) | |
| **UE sync** | **✅ 정상** | **❌ 실패** | **PSS 상관 피크 열화** |

**실패 원인**: 2048-pt FFT convolution에서 int16(±32767) × 2048 누적 합산 시 ~26비트 필요.
float32 mantissa(23비트)로는 하위 3비트 소실 → FFT→곱셈→IFFT 이중 통과에서 오차 누적 →
PSS ZC 시퀀스 상관 피크가 threshold 이하로 저하 → UE 동기화 불가.

**결론**: **complex128(float64)이 필수**. complex64는 단순 연산에는 충분하나 FFT convolution 체인에서 정밀도 부족.

### 4. Custom CUDA Kernel (cupy.RawKernel)

여러 CuPy 연산을 하나의 CUDA C 커널로 수동 퓨전:

```
가능한 퓨전:
  1. int16 → complex128 + symbol extract  → 1 커널
  2. PL + AWGN + clip + int16             → 1 커널
  3. reconstruct + CP insert              → 1 커널
  → 15 커널 → 5~6 커널, CUDA Graph 조합 시 극대화
```

### 5. H 채널 int16 표현 (연구 단계)

채널 H를 int16으로 양자화하여 전송/저장하면 메모리 대역폭을 극적으로 줄일 수 있으나, 채널 계수는 복소수이며 값 범위가 가변적이므로 스케일링 팩터 관리가 필요.

| 항목 | complex64 H | int16 H (스케일링) |
|------|:---:|:---:|
| 메모리/채널 | 8 bytes/sample | 4 bytes/sample (2x 절약) |
| 정밀도 | 7자리 | ~4자리 (16비트) |
| 구현 복잡도 | 낮음 | 높음 (min/max 스케일링 필요) |
| 적용 효과 | - | CH_COPY 시간 ~50% 감소 가능 |

**현실적 우선순위**: ~~CUDA Graph~~ (✅ v10 구현 완료, 3.2→1.1ms) → ~~Socket 제거~~ (✅ v12 구현 완료, CUDA IPC) → RawKernel (1순위, 실시간 달성 목표) → int16 H (연구)
> ※ complex64는 v10에서 테스트 결과 PSS 검출 실패로 **제외**. complex128 필수.

### 6. ✅ Socket 제거 / CUDA IPC (GPU 직접 접근) — v12 구현 완료

**v10 병목**: Proxy의 H2D/D2H + TCP 소켓 오버헤드가 매 슬롯 발생

**CUDA IPC 아키텍처** (v12, 2026-02-24 업데이트):

**Proxy가 SERVER** — GPU 메모리 할당 + IPC 핸들 생성 + shm 파일 생성.
gNB/UE는 모두 CLIENT — IPC 핸들을 열어 동일 GPU 메모리 접근.

> Proxy가 메모리를 소유하므로, RAN 측(OAI)을 Aerial 등으로 교체할 때 메모리 관리 코드를 변경할 필요 없음.

```
[Socket 모드 (v10)]
  gNB → socket TX → Proxy H2D → GPU처리 → D2H → socket TX → UE
  CPU 복사 4회 + 소켓 2회

[CUDA IPC 모드 (v12)]
  gNB H2D → [gpu_dl_tx] → Proxy GPU처리 → [gpu_dl_rx] → UE D2H
  CPU 복사 2회 (gNB/UE의 H2D/D2H만), Proxy측 H2D/D2H 제거, 소켓 제거

  메모리 소유:  Proxy (SERVER) — cp.cuda.alloc() × 4
  핸들 오픈:    gNB (CLIENT), UE (CLIENT) — cudaIpcOpenMemHandle()
```

**동기화 메커니즘**:
- DL write (gNB → Proxy): 블로킹 spin-wait (`dl_tx_ready == 0` 대기). 모든 슬롯이 순서대로 Proxy에 전달.
- UL write (UE → Proxy): 비블로킹 skip. Proxy가 미소비면 건너뛰고 즉시 리턴.
- RU pacing: UL 데이터 미도착 시 `usleep(1ms)` (소켓 모드의 `epoll_wait` 타임아웃과 동일 역할).
- Proxy 사전 워밍업: `_warmup_pipeline()`에서 TF XLA 컴파일 + CUDA Graph 캡처를 메인 루프 전에 완료.

> **참고**: 초기에는 `dl_write_count` atomic 카운터로 RU-L1TX pacing을 시도했으나,
> TDD UL 슬롯에서 DL write가 발생하지 않아 영구 데드락. `usleep(1ms)` 방식으로 교체.

**구현 파일**:
- OAI C: `gpu_ipc.h/c` (dlopen CUDA, mmap 동기화, CLIENT only), `simulator.c` (분기 추가)
- Proxy: `v12_gpu_ipc.py` (`GPUIpcInterface` SERVER, `process_slot_ipc`, `_warmup_pipeline`, WindowProfiler + 듀얼 타이머 + E2E TDD 통계)
- Proxy 백업: `v12_gpu_ipc_old.py` (측정 업그레이드 전 원본)
- Docker: `docker-compose.yml` (`ipc: host`, `/tmp/oai_gpu_ipc` 공유 볼륨)

**v12 측정 시스템** (v11에서 이식, 2026-02-24):
- WindowProfiler: 롤링 윈도우 avg/p95/p99/max 통계 (IPC/Socket 양쪽)
- CUDA Event 듀얼 타이머: CPU+sync와 GPU Event 동시 측정 → CPU 오버헤드 분리
- E2E TDD 프레임 통계: 10슬롯 단위 wall vs Proxy(DL+UL) vs IPC+OAI
- CLI: `--profile-interval`, `--profile-window`, `--dual-timer-compare`

**시작 순서** (Proxy 먼저):
```bash
# 1. Proxy (GPU 할당 + 워밍업)
docker exec -it oai_sionna_proxy python3 \
    /workspace/vRAN_Socket/G0_Sionna_Channel_Proxy/v12_gpu_ipc.py --mode=gpu-ipc
# → "Entering main loop..." 출력 확인 후:

# 2. gNB
sudo RFSIM_GPU_IPC=1 ./nr-softmodem -O <conf> --gNBs.[0].min_rxtxtime 6 --rfsim

# 3. UE
sudo RFSIM_GPU_IPC=1 ./nr-uesoftmodem -r 106 --numerology 1 --band 78 \
    -C 3619200000 --uicc0.imsi 001010000000001 --rfsim
```

### 7. Multi-stream E2E Pipeline (추후)

v12 CUDA IPC 기반으로 DL/UL을 별도 CUDA 스트림에서 병렬 처리하여 추가 처리량 확보.

---

## 📅 테스트 정보
- **측정일**: 2026-01-20 (v3~v8 성능 측정, TITAN RTX)
- **테스트 환경**: NVIDIA TITAN RTX / H100 NVL, Docker Container (TensorFlow + CuPy)
- **채널 모델 수정일**: 2026-02-12 (PL 분리, noise 추가)
- **Full GPU Pipeline (v9.0) 수정일**: 2026-02-12 (int16 I/O, GPU 데이터 변환, GPU AWGN)
- **numpy-free Pipeline (v9.1) 수정일**: 2026-02-12 (TF→DLPack→CuPy, GPU RingBuffer, numpy 제거)
- **커널 배치 최적화 (v9.2) 수정일**: 2026-02-12 (get_batch, put_batch, 배치 정규화, 2D slice)
- **H100 NVL 실측일**: 2026-02-12 (v9.2, `--path-loss-dB -3 --snr-dB 15`, 평균 3.23ms/slot)
- **CUDA Graph v10 실측일**: 2026-02-12 (v10, `--path-loss-dB -3 --snr-dB 15`, 평균 1.11ms/slot, CUDA Graph)
- **E2E 계측 v11 실측일**: 2026-02-13 (v11, `--path-loss-dB -3 --snr-dB 3`, TDD 10-slot 프레임 단위, Proxy 2.01ms + Socket+OAI 0.21ms = wall 2.21ms/slot, 452 slots/sec)
- **CUDA IPC v12 구현일**: 2026-02-24 (v12, Socket 제거, GPU IPC, Proxy H2D/D2H 제거)
- **v12 메모리 소유권 이전**: 2026-02-24 (Proxy가 SERVER로 GPU 할당, gNB/UE 모두 CLIENT, Aerial 교체 대비)
- **v12 TDD 데드락 수정**: 2026-02-24 (dl_write_count pacing 제거, usleep(1ms) 복귀, librfsimulator.so 빌드 이슈 해결)
- **v12 측정 시스템 업그레이드**: 2026-02-24 (v11 이식: WindowProfiler, CUDA Event 듀얼 타이머, E2E TDD 통계, 원본 v12_gpu_ipc_old.py 보존)
- **v12 E2E 프레임 버그 수정**: 2026-02-24 (idle 루프 중복 출력 제거, `_check_e2e_frame()` 메서드 분리)
- **v12 GPU IPC 실측일**: 2026-02-24 (v12, `--path-loss-dB -3 --snr-dB 15 --mode=gpu-ipc`, DL=4090/UL=1144, wall=0.56ms/slot, Proxy=0.52ms, IPC+OAI=0.04ms, v11 대비 3.9x 개선)

# G1C Changelog

> [README.md](../README.md)로 돌아가기

---

## v4 — Unified ChannelProducer (2026-04-02)

### 해결한 과제: 통합 채널 생산 + 리소스 모니터링

v3의 N개 독립 TF 프로세스를 1개 `UnifiedChannelProducerProcess`로 통합.

| 항목 | v3 (N 프로세스) | v4 (1 프로세스) |
|------|:-:|:-:|
| TF 프로세스 수 | N개 (UE당 1개) | **1개** |
| VRAM (2UE) | ~31GB | **~20GB (-35%)** |
| GPU 경합 | TF ↔ TF ↔ Main (3자) | **TF ↔ Main (2자)** |
| 채널 생성 | N회/배치 (UE별 독립) | **1회/배치 (N_UE 차원 통합)** |
| 버퍼 분배 | `put_batch` (blocking) | **`try_put_batch` (non-blocking)** |
| UE 간 공정성 | GPU 스케줄러 의존 | **보장 (동일 배치)** |

### 통신 결과 (v3 2UE → v4 2UE)

| 지표 | v3 2UE | **v4 2UE** |
|------|:------:|:----------:|
| 양쪽 UE 접속 | **❌ (UE0 실패)** | **✅** |
| UE0 stall | 260 프레임 | **13 프레임 (-95%)** |
| DL BLER | 28.0% | **13.6~16.1%** |
| Per-slot avg | 1.50ms | 2.15ms (+43%) |

### 시스템 리소스 모니터링 (sysmon)

- `launch_all.sh` 실행 시 `sysmon.csv` 자동 기록 (CPU/RAM/GPU Core/HBM, 1초 간격)
- `proxy.log`의 E2E frame에 `ts=` epoch 타임스탬프 추가 → sysmon과 시간 매칭 가능

### 성능 요약 (2UE 2×2 MIMO bs280)

| 지표 | 값 |
|------|:--:|
| Slot rate | **483/s** (실시간의 **24.2%**) |
| Per-slot avg | **2.07ms** |
| Proxy 비율 | **99.3%** (Proxy-bound) |
| CQI / RI | 15 / **2** (양쪽 UE) |
| VRAM | ~20GB |

### CLI 옵션 (v4 전용)

- `--xla` / `--no-xla`: TF XLA JIT (기본 off). GPU 독점 이슈로 비활성 권장

```bash
sudo bash launch_all.sh -v v4 -m gpu-ipc -ga 2 1 -ua 2 1 -n 2 -bs 280 \
  -p1b ../P1B_Valid_Results/Area1_7.5GHz_Rays_Valid_RXs.npz -rx 98,498
```

### 제한사항

- Per-slot 시간 증가: v3 2UE 1.50ms → v4 2UE 2.15ms (+43%). 통합 TF의 큰 커널이 Main Process CUDA Graph를 더 오래 blocking
- DL MCS 0 고착: v3와 동일 (OAI MAC 스케줄러 이슈)
- XLA 비호환: `--xla` 옵션 구현됨이지만, 커널 퓨전이 GPU 독점을 심화시켜 UE 접속 실패 유발. 비활성 상태로 유지 권장

상세: [EXPERIMENTS.md — v3 vs v4 비교](EXPERIMENTS.md#6-v3-vs-v4-비교-2026-04-0102)

---

## OAI 4T4R RI/PMI 구현 + 운용 변경 (2026-04-03~06)

v4 proxy 환경에서 4×4 MIMO를 정상 동작시키기 위한 OAI 측 수정. Proxy 코드 자체는 변경 없음.

### 4T4R RI/PMI (2026-04-03/05)

`csi_rx.c`에 2개 신규 함수 추가로 4×4 시스템에서 RI 1~4 및 PMI Rank 1~4 전체 지원.

| 함수 | 역할 |
|------|------|
| `nr_csi_rs_ri_estimation_4x4()` | Progressive sub-matrix condition number 기반 RI 추정 |
| `nr_csi_rs_pmi_estimation_4port()` | 4-port Type I SinglePanel codebook 탐색 (Rank 1~4) |

검증: Bypass RI=4/CQI=15, P1B 채널 RI=2↔4 동적 전환 확인.

### IPC Read Timeout 확장 (2026-04-06)

`simulator.c`의 `gpu_ipc_v7_dl_read` / `gpu_ipc_v7_ul_read` timeout을 **30ms → 5000ms**로 변경. 8 UE 4T4R에서 proxy 처리 시간이 30ms를 초과하여 접속 실패하던 문제 해결 (0/8 → 6/8 접속).

### GPU 1 전환 (2026-04-06)

GPU 0의 좀비 프로세스 회피를 위해 `gpu_ipc_v7.c`의 `cudaSetDevice(0)` → `cudaSetDevice(1)`, `v4.py`의 `gpu_num` → `1`로 변경.

상세: [OAI_CHANGES.md](OAI_CHANGES.md) · [EXPERIMENTS.md §7.5](EXPERIMENTS.md#75-rank-34-pmi-구현-검증-2026-04-05) · [MODIFICATION_LOG.md](../../../openairinterface5g_whan/MODIFICATION_LOG.md)

---

## v3 — P1B Ray + Stall Detection (2026-03-27)

### 해결한 과제: Multi-UE 안정화 + 물리 정확성

v2에서 실패하던 Multi-UE superposition의 근본 원인 두 가지를 해결:

| 문제 | v2 | v3 |
|------|:--:|:--:|
| UE별 채널 데이터 | 전 UE 동일 ray → RA 간섭 | **UE별 독립 P1B ray** |
| UE stall 처리 | `min_ue_head` (전체 멈춤) | **`active_set` (자동 제외)** |
| 2UE 380초 안정 | ❌ | **✅** |
| RA 성공률 | 빈번 실패 | **안정적** |
| OAI segfault | 발생 | **0건** |

### CLI 옵션 (v3 전용)

- `--p1b-npz PATH`: P1B npz 파일 경로
- `--ue-rx-indices`: `"100,500"` (수동), `"random"`, 미지정 시 auto random
- 무효 인덱스 시 가장 가까운 유효 인덱스 안내 후 종료

### 성능 요약 (2×2 MIMO bs280)

| 지표 | v3 1UE | v3 2UE |
|------|:------:|:------:|
| Slot rate | **1,233/s** (61.7%) | **671/s** (33.6%) |
| Per-slot avg | **0.81ms** | **1.49ms** |
| CQI / RI | 15 / **2** | 15 / **2** |
| DL BLER | 16.2% | 28.0% |
| UL BLER | 0.0% | 0.0% |

### 제한사항 (v4에서 해결)

- 2UE 시 UE0 PDU Session 미수립: 2개 독립 TF 프로세스의 비대칭 GPU 경합
- VRAM 높음: TF 프로세스 N개 × ~13GB

```bash
sudo bash launch_all.sh -v v3 -m gpu-ipc -ga 2 1 -ua 2 1 -n 2 -bs 280 \
  -p1b ../P1B_Valid_Results/Area1_7.5GHz_Rays_Valid_RXs.npz -rx 98,498
```

상세: [EXPERIMENTS.md — v3 2UE 독립 P1B](EXPERIMENTS.md#5-v3-2ue-독립-p1b-채널-2026-03-27)

---

## v2 — multiprocessing ChannelProducer (2026-03-26)

### 해결한 과제: 2UE+ TF 크래시

v1에서 2UE 이상 실행 시 `ChannelProducer` 쓰레드들이 동일 TF GPU 컨텍스트를 공유하여 텐서 shape 오염 → 크래시.

| 항목 | v1 (Thread) | v2 (Process) |
|------|:---:|:---:|
| ChannelProducer | `threading.Thread` | **`multiprocessing.Process`** (N개) |
| TF 컨텍스트 | 공유 (2UE+ 크래시) | **프로세스별 독립** |
| RingBuffer | `threading.Condition` | **`mp.Condition` + CUDA IPC** |
| TF 메모리 | 공유 (OOM) | **`set_memory_growth(True)`** |
| 링버퍼 길이 | 42000 | **10500** (1/4 축소, VRAM 절약) |

### 추가 특징

- Graceful degradation: ChannelProducerProcess 크래시 시 bypass copy 전환 (timeout 기반)
- E2E 로그 개선: per-UE DL/UL 비율 표시, `>= next_boundary` 방식

### 제한사항 (v3에서 해결)

- 모든 UE가 동일 ray 데이터 사용 → 채널 상관으로 RA 실패 빈발
- 1 UE 크래시 시 `min_ue_head` 고정 → 전체 UL 멈춤
- OAI UE segfault: `nr_rrc_prepare_msg3_payload()` assert(0)

상세: [EXPERIMENTS.md — v2 Multi-UE 분석](EXPERIMENTS.md#4-v2-multi-ue-분석-2026-03-26)

---

## v1 — IPC V7 Futex + 스파이크 완화 (2026-03-26~27)

### IPC: V6 → V7 (Futex 기반 즉시 알림)

V6의 `usleep(1000)` 폴링을 **futex 기반 즉시 알림**으로 교체.

| 항목 | V6 | V7 |
|------|:--:|:--:|
| 동기화 | `usleep(1000)` × 30 = ~30ms max | `futex_wait` (즉시 알림, timeout 30ms) |
| SHM MAGIC | `0x47505537` | `0x47505538` |
| Read API | `v6_read()` + 외부 loop | `v7_read(..., timeout_ms)` — 단일 호출 |
| Seq counter | 없음 | **4개** (dl_tx, dl_rx, ul_tx, ul_rx) |
| Write 후 알림 | 없음 | **seq++ → futex_wake** |
| 환경변수 | `RFSIM_GPU_IPC_V6=1` | `RFSIM_GPU_IPC_V7=1` |
| C 파일 | `gpu_ipc_v6.c/.h` | `gpu_ipc_v7.c/.h` 추가 |

Write ordering: GPU write 완료 → timestamp 갱신 → seq counter++ → futex_wake (C/Python 양쪽 고정)

### 해결한 과제: 스파이크 완화 (배치 크기 튜닝)

`-bs` (buffer-symbol-size) 배치 크기 튜닝으로 스파이크를 사실상 제거.

| 지표 | baseline (bs4200) | **bs280 (최적)** | 변화 |
|------|:-:|:-:|:-:|
| 순간 처리율 | 420~650 DL/s | **836~900 DL/s** | ×1.6~2.0 |
| wall p50 / p95 | 11.78 / 88.44ms | **9.86 / 27.20ms** | -16% / -69% |
| wall max | 226.78ms | **79.56ms** | -65% |
| >80ms 스파이크 | 6.1% | **0.0%** | **제거** |

역 U자형 성능 곡선: bs140(너무 작음) → **bs280(최적)** → bs560(양호) → bs840(너무 큼)

### 추가 특징

- Fused clip+cast RawKernel: UL superposition의 `round+clip+cast` 4-5개 커널 → 1개 CUDA RawKernel
- IPC 대기 시간: 0.02~1.29ms → ≈0ms (futex 효과)

### 성능 요약 (1UE 2×2 MIMO bs280)

| 지표 | 값 |
|------|:--:|
| Slot rate | **1,233 slots/s** (실시간의 **61.7%**) |
| Per-slot | **0.81ms** avg |
| >80ms 스파이크 | **0.0%** |
| CQI / RI | 15 / **2** |

### 제한사항

- ChannelProducer가 여전히 `threading.Thread` → 2UE+ TF 크래시 미해결
- DL MCS 0 고착 (DL BLER >10%)

상세: [EXPERIMENTS.md — v0 vs v1 비교](EXPERIMENTS.md#2-v0-vs-v1-비교-2026-03-26), [EXPERIMENTS.md — bs 배치 크기 튜닝](EXPERIMENTS.md#3-v1-bs-배치-크기-튜닝-2026-03-27)

---

## v0 — G1B v8 기반 Multi-UE 프레임워크 (2026-03-12)

### IPC: V5 → V6 (GPU Circular Buffer)

G1B v1에서 도입된 IPC V5 (timestamp-indexed GPU circular buffer)를 Multi-UE용으로 확장한 **IPC V6**.

| 항목 | IPC V5 (G1B) | IPC V6 (G1C v0) |
|------|:---:|:---:|
| SHM 인스턴스 | 1개 (gNB) | **1 + N개** (gNB + UE별) |
| SHM 경로 | `/tmp/oai_gpu_ipc/gpu_ipc_shm` | gNB: 동일 / UE[k]: `gpu_ipc_shm_ue{k}` |
| 동기화 | `usleep(1000)` × 30 폴링 | 동일 |
| MAGIC | `0x47505537` | 동일 |
| C 코드 | `gpu_ipc_v5.c` | `gpu_ipc_v6.c` + `RFSIM_GPU_IPC_UE_IDX` |

### 특징

- G1B v8의 CUDA Graph, CH_COPY view+release, NoiseProducer batch 사전 생성을 계승
- N개 UE에 대한 독립 IPC 인스턴스 + DL Broadcast + UL Superposition 구현
- `threading.Thread` 기반 ChannelProducer (1UE에서만 안정)

### 성능 (1UE 2×2 MIMO Channel, H100 NVL)

| 지표 | 값 |
|------|:--:|
| DL rate | 416 DL/s |
| Proxy per-slot | 1.71ms avg |
| IPC+OAI 대기 | 0.02~1.29ms (usleep 폴링) |
| >80ms 스파이크 | **5.5%** |
| 2UE 2×2 channel | ✅ 동작, per-slot ~1.8ms |

### 제한사항

- 스파이크 빈발 (>80ms 5.5%, usleep 폴링 + 채널 버퍼 고갈)
- 2UE+ 채널 모드에서 ChannelProducer(`threading.Thread`)의 TF 컨텍스트 공유 → 크래시

상세: [EXPERIMENTS.md — v0 초기 실험](EXPERIMENTS.md#1-v0-초기-실험-2026-03-12)

---

## IPC 버전 히스토리

```
V1/V2 (ring buffer, ready-flag)
  → V5 (circular buffer, timestamp-indexed)         ← G1B v1
    → V6 (V5 + Multi-Instance per-UE SHM)           ← G1C v0
      → V7 (V6 + futex seq counter, 즉시 알림)      ← G1C v1~v4
```

| IPC 버전 | 도입 시점 | 동기화 | SHM MAGIC | 핵심 변경 |
|:--------:|:--------:|--------|:---------:|----------|
| V5 | G1B v1 | `usleep(1ms)` × 30 | `0x47505537` | timestamp-indexed circular buffer |
| V6 | G1C v0 | 동일 | 동일 | per-UE SHM 인스턴스 (`gpu_ipc_shm_ue{k}`) |
| V7 | G1C v1 | **`futex_wait`/`futex_wake`** | `0x47505538` | seq counter 4개, 즉시 알림, 단일 read API |

상세: [OAI_CHANGES.md](OAI_CHANGES.md) (요약) · [MODIFICATION_LOG.md](../../../openairinterface5g_whan/MODIFICATION_LOG.md) (원본)

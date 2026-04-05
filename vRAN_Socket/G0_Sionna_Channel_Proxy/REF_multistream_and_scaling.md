# G0 참고: 멀티스트림 파이프라인 & 안테나 스케일링

> G1B (100T Scalable Mock) 실험에서 검증된 설계 패턴과 수치를 G0 환경에 맞게 정리.
> G1B 조건: TITAN RTX, complex64, Socket/Pinned Memory, Mock 데이터
> G0 조건: H100 NVL, complex128, CUDA IPC, Sionna 채널 모델

---

## 1. 멀티스트림 파이프라인

### 1.1 개념: 2-슬롯 오버랩

두 개의 CUDA 스트림을 교대로 사용하여, 이전 슬롯의 D2H와 다음 슬롯의 H2D를 PCIe 양방향으로 동시 실행.

```
Stream 0: H2D[0] → GPU[0] → D2H[0]    H2D[2] → GPU[2] → D2H[2]    ...
Stream 1:     H2D[1] → GPU[1] → D2H[1]    H2D[3] → GPU[3] → D2H[3] ...
                        ↑
              D2H[0]과 H2D[1]이 동시 실행 (PCIe 양방향)
```

스트림별로 독립 Pinned Memory + GPU 버퍼가 필요:

| 리소스 | Stream 0 | Stream 1 |
|--------|----------|----------|
| Pinned In | `pinned_in[0]` | `pinned_in[1]` |
| Pinned Out | `pinned_out[0]` | `pinned_out[1]` |
| GPU In | `gpu_in[0]` | `gpu_in[1]` |
| GPU Out | `gpu_out[0]` | `gpu_out[1]` |
| 채널 Hf | 공유 (읽기 전용) | 공유 (읽기 전용) |

### 1.2 G1B 실측 결과 (TITAN RTX, complex64)

| 안테나 | 순차 Throughput | 멀티스트림 Throughput | 향상 | PCIe 양방향 speedup |
|--------|:-:|:-:|:-:|:-:|
| 8T | 2455 slots/sec | 2769 slots/sec | +12.8% | 1.74x |
| 16T | 1236 slots/sec | 1448 slots/sec | +17.2% | 1.74x |
| 32T | 598 slots/sec | 738 slots/sec | +23.4% | 1.74x |
| 64T | 276 slots/sec | 352 slots/sec | +27.5% | 1.74x |

- 개별 슬롯 latency는 변하지 않음 (throughput만 향상)
- PCIe 양방향 동시 전송 효율: 87.2% (이론 최대 2.0x 대비)
- 안테나 수가 클수록 전송량이 많아져 오버랩 이득이 커짐

### 1.3 G0에 적용 시 고려사항

**G0 v12(CUDA IPC)에서는 Proxy 측 H2D/D2H가 없다.**

```
[Socket 모드] gNB → TCP → Proxy H2D → GPU → D2H → TCP → UE
                                ↑ 여기서 멀티스트림 오버랩 가능

[IPC 모드]    gNB H2D → [gpu_dl_tx] → Proxy GPU Copy → [gpu_dl_rx] → UE D2H
                                       ↑ GPU-to-GPU 복사만, PCIe 전송 없음
```

따라서 G1B 방식(H2D/D2H 오버랩)은 **IPC 모드에서 직접 적용 불가**.

**G0에서 멀티스트림이 유효한 시나리오:**

| 시나리오 | 적용 가능성 | 설명 |
|----------|:-:|------|
| DL/UL 병렬 처리 | ○ | DL 채널 처리와 UL passthrough를 별도 스트림 |
| 채널 생성 + 채널 적용 분리 | ○ | ChannelProducer와 process_slot을 별도 스트림 |
| Socket 모드 H2D/D2H 오버랩 | ○ | v12 `--mode=socket` 사용 시 G1B 패턴 적용 |
| IPC 모드 H2D/D2H 오버랩 | ✗ | Proxy 측 PCIe 전송 없음 |

**G0 README의 Future Work #7 (Multi-stream E2E Pipeline)에 해당:**
DL 슬롯 처리 중 UL 슬롯이 도착하면, 별도 스트림에서 UL을 passthrough하여 DL 처리와 병렬화.

```
Stream DL: [CH_COPY] → [CUDA Graph Launch] → [Copy-Out]
Stream UL:         [UL Copy-In → Copy-Out]
                   ↑ DL GPU 처리 중에 UL 동시 처리
```

### 1.4 G1B 핵심 코드 패턴 (참고)

```python
# 비동기 H2D (스트림별 독립)
def _h2d_async(self, stream_id):
    with self.streams[stream_id]:
        self.gpu_in[stream_id].set(self.pinned_in[stream_id])
        self.events_h2d[stream_id].record(self.streams[stream_id])

# 비동기 GPU 연산 (이벤트 기반 의존성)
def _gpu_compute_async(self, stream_id):
    with self.streams[stream_id]:
        self.streams[stream_id].wait_event(self.events_h2d[stream_id])
        Xf = cp.fft.fft(self.gpu_in[stream_id], axis=2)
        Yf = Xf * self.Hf_gpu
        self.gpu_out[stream_id][:] = cp.fft.ifft(Yf, axis=2)
        self.events_gpu[stream_id].record(self.streams[stream_id])

# 비동기 D2H (GPU 완료 대기)
def _d2h_async(self, stream_id):
    with self.streams[stream_id]:
        self.streams[stream_id].wait_event(self.events_gpu[stream_id])
        self.gpu_out[stream_id].get(out=self.pinned_out[stream_id])
        self.events_d2h[stream_id].record(self.streams[stream_id])
```

동기화 포인트: `events_d2h[prev_stream].synchronize()` — 이전 슬롯의 D2H가 끝나야 결과 사용 가능.

---

## 2. 안테나 수별 스케일링

### 2.1 G1B 방법론

데이터 형상을 `(n_ant, 14, 2048)`으로 잡고, 첫 번째 차원(안테나)만 1→100으로 변경하며 측정.

```python
# 안테나 수별 데이터 크기
data_size = n_ant * N_SYM * FFT_SIZE * bytes_per_sample
# complex64:  n_ant × 14 × 2048 × 8  bytes
# complex128: n_ant × 14 × 2048 × 16 bytes  ← G0 환경
```

파이프라인: H2D → FFT(axis=2) → Hf 곱셈 → IFFT(axis=2) → D2H

### 2.2 G1B 실측: 안테나별 데이터 크기 & 처리 시간

#### Baseline (Pageable Memory + asarray, TITAN RTX, complex64)

| 안테나 | 데이터 크기 | H2D | FFT | 곱셈 | IFFT | D2H | 총합 | 500μs |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1T | 224 KB | 73μs | 21μs | 25μs | 47μs | 61μs | **227μs** | ✅ |
| 4T | 896 KB | 190μs | 9μs | 5μs | 40μs | 177μs | **421μs** | ✅ |
| 16T | 3.5 MB | 665μs | 18μs | 19μs | 20μs | 430μs | **1152μs** | ✗ |
| 64T | 14 MB | 2482μs | 54μs | 76μs | 104μs | 1393μs | **4109μs** | ✗ |
| 100T | 21.9 MB | 3769μs | 81μs | 117μs | 159μs | 2128μs | **6253μs** | ✗ |

#### Pinned + set()/get() 적용 후 (TITAN RTX, complex64)

| 안테나 | 데이터 크기 | H2D | GPU | D2H | 총합 | H2D 대역폭 | 500μs |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1T | 0.22 MB | 31μs | 126μs | 37μs | **195μs** | 6.9 GB/s | ✅ |
| 4T | 0.88 MB | 87μs | 128μs | 90μs | **305μs** | 9.8 GB/s | ✅ |
| **8T** | 1.75 MB | 163μs | 139μs | 161μs | **463μs** | 10.5 GB/s | ✅ |
| 16T | 3.50 MB | 315μs | 165μs | 302μs | **782μs** | 10.9 GB/s | ✗ |
| 32T | 7.00 MB | 608μs | 170μs | 575μs | **1353μs** | 11.2 GB/s | ✗ |

핵심 발견:
- Pinned + set()/get()로 대역폭 5.5→11 GB/s (2x)
- **TITAN RTX + complex64 기준 8T가 500μs 상한**
- H2D+D2H가 총 시간의 70% 이상 → PCIe 대역폭이 병목

### 2.3 안테나별 데이터 크기 비교 (complex64 vs complex128)

| 안테나 | complex64 | complex128 (G0) | 비고 |
|--------|:-:|:-:|------|
| 1T | 224 KB | **448 KB** | 현재 G0 운용 크기 |
| 4T | 896 KB | **1.75 MB** | |
| 8T | 1.75 MB | **3.50 MB** | G1B 상한(complex64) = G0 4T 크기 |
| 16T | 3.50 MB | **7.00 MB** | |
| 64T | 14.3 MB | **28.7 MB** | |
| 100T | 22.4 MB | **44.8 MB** | |

### 2.4 PCIe 대역폭 한계 분석

#### TITAN RTX (PCIe Gen3 x16)

| 항목 | 값 |
|------|-----|
| 이론 대역폭 | 15.75 GB/s |
| 실측 H2D | 11.4 GB/s (이론의 72%) |
| 실측 D2H | 12.2 GB/s (이론의 77%) |
| 500μs 내 전송 가능 (왕복) | ~3 MB → **~13T (complex64)** |

#### H100 NVL (PCIe Gen5 x16) — G0 환경

| 항목 | 값 (추정) |
|------|-----|
| 이론 대역폭 | 63 GB/s |
| 예상 실측 | ~50 GB/s (이론의 ~80%) |
| 500μs 내 전송 가능 (왕복) | ~12.5 MB → **~27T (complex128)** |
| 1ms 내 전송 가능 (왕복) | ~25 MB → **~55T (complex128)** |

> 단, G0 v12 IPC 모드에서는 Proxy 측 PCIe 전송이 없으므로 (GPU-to-GPU 복사),
> PCIe 한계는 gNB/UE 측 H2D/D2H에만 적용됨.

### 2.5 G1B의 핵심 발견 (G0에 이전 가능)

1. **Pinned Memory만으로는 효과 없음** — `asarray()`가 매번 GPU 메모리 재할당
2. **Pinned + GPU 버퍼 사전할당 + set()/get()이 세트** — 이 네 가지가 조합되어야 효과
3. **H2D/D2H가 병목** — 안테나 수가 커지면 GPU 연산보다 전송 시간이 지배적
4. **8T에서 1000슬롯 미스율 0%** — 순차 처리로 충분, 파이프라이닝 불필요 (TITAN RTX 기준)
5. **TCP 소켓이 진짜 병목** — GPU 358μs OK, 소켓 ~2400μs → G0 v12에서 CUDA IPC로 해결 완료

---

## 3. G0에서의 실험 계획 (미수행)

### 3.1 멀티스트림: DL/UL 병렬 처리 실험

**목표**: DL 채널 처리 중 UL passthrough를 별도 스트림에서 동시 실행

**방법**:
1. v12에 CUDA 스트림 2개 추가 (stream_dl, stream_ul)
2. DL: stream_dl에서 CH_COPY → CUDA Graph Launch → Copy-Out
3. UL: stream_ul에서 Copy-In → Copy-Out (passthrough)
4. 이벤트 기반 의존성으로 동기화

**측정 지표**: wall/slot 개선율, DL/UL 동시 도착 시 처리 시간

**예상 효과**: UL passthrough가 ~0.04ms로 매우 짧아 이득 제한적. DL이 0.52ms를 차지하므로, DL 처리 중 UL을 숨길 수 있지만 현재 TDD 구조상 DL/UL이 동시에 오지 않아 실질 이득은 미미할 수 있음.

### 3.2 안테나 스케일링 실험

**목표**: H100 NVL + complex128 + CUDA IPC 환경에서 안테나 수별 처리 시간 측정

**방법**:
1. v12의 GPU 파이프라인을 기반으로 `(n_ant, 14, 2048)` complex128 데이터 생성
2. CUDA Graph 포함한 전체 파이프라인으로 측정
3. 안테나 수: 1T, 2T, 4T, 8T, 16T, 32T
4. 1000슬롯 연속 처리로 안정성 검증

**측정 지표**:
- GPU 처리 시간 (CUDA Event)
- GPU-to-GPU 복사 시간 (IPC 모드)
- 1ms/slot 충족 가능 최대 안테나 수
- 미스율 (1000슬롯 기준)

**G1B와의 차이점 (실험 설계 시 주의)**:

| 항목 | G1B (Mock) | G0 (실제) |
|------|-----------|-----------|
| 정밀도 | complex64 | complex128 (2x 데이터) |
| 채널 | 랜덤 Hf | Sionna 3GPP TR 38.901 |
| 통신 | Socket / Pinned Memory | CUDA IPC (GPU-to-GPU) |
| GPU | TITAN RTX (PCIe Gen3) | H100 NVL (PCIe Gen5) |
| 파이프라인 | FFT→곱셈→IFFT | int16→deintlv→FFT conv→reconstruct→PL→AWGN→int16 |
| CUDA Graph | 미사용 | 사용 (커널 런치 오버헤드 제거) |
| 슬롯 주기 | 500μs (30kHz SCS) | 1ms (30kHz SCS, OAI 기준) |
| RingBuffer | 없음 (직접 생성) | GPU RingBuffer + ChannelProducer |

---

## 4. 참고: G1B 소켓 병목 분석 (해결 완료)

G0 v12에서 이미 해결했지만, 향후 Socket 모드 사용 시 참고.

| 구간 | 시간 | 병목 |
|------|:-:|:-:|
| GPU 처리 (8T) | 358μs | ✗ |
| TCP 소켓 (loopback) | ~2400μs | ✅ |
| E2E 합계 | 2752μs | - |

원인: Proxy가 GPU 처리하는 동안 TCP 수신버퍼 정체 → 흐름제어(window=0) → gNB 블로킹.
100MB 소켓 버퍼로도 해결 안 됨 (TCP 흐름제어가 근본 원인).

해결: G0 v12 CUDA IPC로 Socket 완전 제거. IPC+OAI=0.04ms/slot.

---

*출처: G1B_100T_Scalable_Mock (2026-01-21), G0 v12 실측 (2026-02-24)*
*작성일: 2026-02-26*

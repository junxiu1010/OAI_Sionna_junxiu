# 프로젝트 수정 로그

> 이 문서는 OAI 소스코드 및 Sionna Channel Proxy에 대한 수정 사항을 기록합니다.
> OAI 기본 브랜치: `develop` (Abrev. Hash: `94da422712`)
> Sionna Proxy: `vRAN_Socket/G0_Sionna_Channel_Proxy/v9_cupy_full_gpu_pipeline.py` (v8 원본 유지)

---

## 수정 1: CSI-RS PMI 단일 포트 SINR 실측 계산

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-12 |
| **파일** | `openair1/PHY/NR_UE_TRANSPORT/csi_rx.c` |
| **함수** | `nr_csi_rs_pmi_estimation()` |
| **라인** | 약 597~635 (수정 후) |

### 문제

OAI 원본 코드에서 CSI-RS 포트가 1개(`N_ports == 1`)인 경우, SINR을 **46 dB로 hardcoding** 하고 즉시 반환:

```c
// 수정 전 (원본)
if(N_ports == 1) {
    i2[0] = 0;
    *precoded_sinr_dB = 46; // 기본 SINR 값 (46 dB)
    return 0;
}
```

이로 인해:
- `nr_csi_rs_cqi_estimation()`에서 SINR 46 dB → **CQI = 15 (최대)** 항상 고정
- Sionna proxy에서 `--snr-dB`로 AWGN noise를 추가해도 **CQI가 변하지 않음**
- UE가 측정한 `noise_power`는 정상 변동 (32 → 56~77)하지만, SINR/CQI에 반영되지 않음
- 채널 품질 제어 (SNR 변화에 따른 MCS 적응 등)가 불가능

### 원인 분석

- OAI에서 2포트 이상일 때는 precoding 후보 4가지에 대해 SINR을 `Σ|precoded_H|² / noise_power`로 계산
- 1포트일 때는 precoding이 identity뿐이라 PMI 선택이 불필요하므로, OAI가 SINR 계산 자체를 생략
- **5G 표준 (TS 38.214)에서는 1포트에서도 SINR을 계산하여 CQI에 반영해야 함**

### 수정 내용

`N_ports == 1`일 때 실제 SINR을 계산하도록 변경:

```c
// 수정 후 (v2 - per-RB 정규화 포함)
if(N_ports == 1) {
    i2[0] = 0;

    uint32_t effective_noise_power = interference_plus_noise_power;
    if (effective_noise_power == 0) effective_noise_power = 1;

    int64_t signal_power = 0;
    int rb_count = 0;
    for (각 RB) {
        rb_count++;
        for (각 수신 안테나) {
            signal_power += |H_estimated(rb)|²;
        }
    }
    signal_power /= nb_antennas_rx;  // 안테나 평균
    signal_power /= rb_count;        // per-RB 평균 (noise_power 스케일 일치)

    sinr_linear = signal_power / effective_noise_power;
    *precoded_sinr_dB = dB_fixed(sinr_linear);
}
```

#### v1 → v2 수정 (per-RB 정규화 추가)

| 버전 | signal_power | noise_power | SINR 결과 |
|:---:|:---:|:---:|:---:|
| v1 (합산) | `Σ_rb \|H\|² / nb_ant` ≈ 2,400,000 | per-RB variance ≈ 36 | 48 dB (과대) |
| v2 (per-RB) | `Σ_rb \|H\|² / nb_ant / rb_count` ≈ 45,000 | per-RB variance ≈ 36 | ~31 dB |

- `noise_power`는 `calc_power_csirs()`가 per-RB 분산으로 계산 (`E[x²]-E[x]²`)
- v1에서는 signal만 전체 RB 합산하여 스케일 불일치 → 약 `10*log10(53) ≈ 17 dB` 과대 추정
- v2에서 `signal_power /= rb_count`로 동일 스케일 정규화

### 기대 효과

| SNR 설정 (Sionna proxy) | 수정 전 SINR | 수정 후 SINR (예상) | 수정 후 CQI (예상) |
|:---:|:---:|:---:|:---:|
| noise 없음 | 46 dB | 40~46 dB | 15 |
| `--snr-dB 20` | 46 dB | ~20 dB | 15 |
| `--snr-dB 15` | 46 dB | ~15 dB | 11 |
| `--snr-dB 10` | 46 dB | ~10 dB | 9 |
| `--snr-dB 5` | 46 dB | ~5 dB | 6 |

※ 실제 값은 채널 추정 정확도 및 noise estimation 방식에 따라 다를 수 있음

### 빌드

이 파일은 UE 빌드에 포함됩니다:

```bash
cd cmake_targets
./build_oai --nrUE -c
```

### 참고

- `noise_power`는 CSI-RS 채널 추정 잔차 (`|h_LS - h_interpolated|²`)로 추정됨 (동일 파일 내 `nr_csi_rs_channel_estimation()`)
- `dB_fixed(uint32_t x)`: 정수 기반 `10*log10(x)` 계산 (`openair1/PHY/TOOLS/dB_routines.c`)
- CQI 테이블: `nr_csi_rs_cqi_estimation()` (동일 파일, TS 38.214 Table 5.2.2.1-2 기반)
- `interference_plus_noise_power`: CSI-IM이 설정되었으면 CSI-IM 기반 간섭+잡음, 아니면 CSI-RS 기반 잡음 추정치 사용

### 추가 이력 (2026-02-12): 정수 기반 유지로 롤백

| 항목 | 내용 |
|------|------|
| **파일** | `openair1/PHY/NR_UE_TRANSPORT/csi_rx.c` |
| **대상 함수** | `nr_csi_rs_pmi_estimation()` (`N_ports == 1` 경로) |
| **변경 시도** | `sinr_linear = signal/noise` 계산에서 반올림 및 소수점 디버그(`x100`) 추가 |
| **관찰 결과** | 저 SNR 구간에서 단기적으로 완화 효과는 있으나, OAI의 기존 정수 dB/CQI 매핑 체계와 해석 일관성이 떨어짐 |
| **최종 조치** | 정수 기반 계산(`sinr_linear = signal_power / effective_noise_power`)으로 복원 |
| **복원 이유** | 본 경로의 목적이 정밀 계측이 아니라 CQI/PMI/RI 의사결정용 지표이므로, OAI 기존 정수 기반 동작과 일관성 유지가 우선 |

복원 후에도 `N_ports == 1`에서의 핵심 수정(46 dB 하드코딩 제거, per-RB 정규화, `log2_re` 반영)은 유지됨.

---

## 수정 2: Sionna Proxy Full GPU Pipeline (v9)

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-12 |
| **파일** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v9_cupy_full_gpu_pipeline.py` |
| **클래스** | `GPUSlotPipeline` |
| **함수** | `process_slot()`, `process_ofdm_slot_with_slot_pipeline()` |

### 문제

v8.1 (Synchronous Batch) 에서 H100 서버로 이전 후 **실시간성 미달** (~2.8ms/slot, 목표 1ms/slot):

| 병목 | 시간 | 원인 |
|------|:----:|------|
| IQ deinterleave | 0.36ms | CPU `np.frombuffer` + 복소수 변환 |
| Pinned memory prep | 0.26ms | CPU for-loop + `.astype()` |
| Reconstruct | 0.33ms | CPU for-loop + slice 할당 |
| AWGN noise | 4.5ms | CPU `np.random.randn` (가장 큰 병목) |
| int16 변환 | 0.6ms | CPU `np.clip` + `np.round` + `.astype` |
| **합계** | **~6ms** | **GPU 처리 (0.05ms)의 120배** |

### 수정 내용

**모든 데이터 변환과 noise 생성을 GPU로 이동** (Full GPU Pipeline):

#### 1. AWGN noise → GPU (`cp.random.randn`)

```python
# 수정 전 (CPU numpy): 4.5ms/slot
noise = noise_std * (np.random.randn(N) + 1j * np.random.randn(N))

# 수정 후 (GPU CuPy): ~0.01ms/slot
self.gpu_out += n_std * (cp.random.randn(N) + 1j * cp.random.randn(N))
```

#### 2. 데이터 변환 → GPU

```python
# 수정 전: CPU에서 int16 → complex → ... → complex → int16
x = np.frombuffer(iq_bytes, dtype='<i2')           # CPU
x_cpx = x[::2] + 1j * x[1::2]                     # CPU: 0.36ms
# ... GPU FFT convolution ...                       # GPU: 0.05ms
y_int16[::2] = np.clip(np.round(out.real), ...).astype('<i2')  # CPU: 0.6ms

# 수정 후: GPU에서 전부 처리
gpu_f64 = self.gpu_iq_in.astype(cp.float64)        # GPU
gpu_cpx = gpu_f64[::2] + 1j * gpu_f64[1::2]       # GPU
# ... GPU FFT convolution ...                       # GPU
self.gpu_iq_out[::2] = cp.clip(cp.around(gpu_out.real), ...).astype(cp.int16)  # GPU
```

#### 3. H2D/D2H → int16 (전송량 4x 감소)

```
수정 전: H2D complex128 (480KB) → GPU → D2H complex128 (480KB)
수정 후: H2D int16     (120KB) → GPU → D2H int16     (120KB)
```

#### 4. Symbol extract/reconstruct → GPU 인덱스 사전계산

```python
# 초기화 시 인덱스 배열 사전계산 (한 번만)
self.gpu_ext_idx = cp.asarray(extract_indices)   # (14, 2048) 심볼 추출용
self.gpu_data_dst = cp.asarray(data_dst_flat)    # (28672,) 데이터 scatter용
self.gpu_cp_dst = cp.asarray(cp_dst_flat)        # (2048,) CP scatter용
self.gpu_cp_src = cp.asarray(cp_src_flat)        # (2048,) CP gather용

# 처리 시 GPU fancy indexing (for-loop 제거)
self.gpu_x[:] = gpu_cpx[self.gpu_ext_idx]            # symbol extract
self.gpu_out[self.gpu_data_dst] = gpu_y_flat          # data reconstruct
self.gpu_out[self.gpu_cp_dst] = gpu_y_flat[self.gpu_cp_src]  # CP reconstruct
```

### 기대 효과

| 단계 | v8.1 (CPU) | v9 (GPU) | 개선 |
|------|:---:|:---:|:---:|
| IQ deinterleave | 0.36ms | ~0ms | GPU |
| Symbol extract | 0.26ms | ~0ms | GPU index |
| FFT convolution | 0.05ms | 0.05ms | 동일 |
| Reconstruct | 0.33ms | ~0ms | GPU index |
| Path Loss | ~0ms | ~0ms | 동일 |
| AWGN noise | 4.5ms | ~0.01ms | GPU random |
| int16 변환 | 0.6ms | ~0ms | GPU |
| H2D | 0.09ms | ~0.02ms | 4x 축소 |
| D2H | 0.09ms | ~0.02ms | 4x 축소 |
| **합계** | **~6.3ms** | **~0.2ms** | **~30x** |

### 변경 사항 요약

| 항목 | v8.1 | v9 |
|------|------|------|
| Pipeline | Synchronous Batch | Full GPU int16-to-int16 |
| H2D 데이터 | complex128 (480KB) | int16 (120KB) |
| D2H 데이터 | complex128 (480KB) | int16 (120KB) |
| CPU 작업 | IQ변환, 심볼추출, 재구성, AWGN, int16변환 | 채널 버퍼 읽기만 |
| AWGN | `np.random.randn` (CPU) | `cp.random.randn` (GPU) |
| Symbol 처리 | CPU for-loop | GPU index gather/scatter |
| CPU fallback | 없음 | 비표준 슬롯 자동 전환 |

---

## 수정 3: Sionna Proxy numpy-free Pipeline (v9.1)

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-12 |
| **파일** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v9_cupy_full_gpu_pipeline.py` |
| **클래스** | `ChannelProducer`, `RingBuffer`, `GPUSlotPipeline`, `Proxy` |

### 문제

v9.0에서 채널 H가 **TF → numpy → CuPy** (GPU → CPU → GPU) 경유하여 불필요한 왕복 발생:

```
TF GPU tensor → .numpy() (GPU→CPU) → cp.asarray() (CPU→GPU)
                    ↑                         ↑
              불필요한 CPU 경유           다시 GPU로 올림
```

### 수정 내용

#### 1. ChannelProducer: TF → DLPack → CuPy (GPU-to-GPU)

```python
# 수정 전: TF → numpy → buffer (CPU 경유)
h_delay = tf.squeeze(h_delay).numpy()           # GPU → CPU
h_normalized = h / np.sqrt(np.sum(np.abs(h)**2))  # CPU numpy
self.buffer.put(h_normalized)                    # CPU numpy

# 수정 후: TF → DLPack → CuPy (GPU 직접)
h_delay = tf.squeeze(h_delay)                    # TF tensor, GPU 유지
energy = tf.reduce_sum(tf.abs(h_tf)**2)          # TF GPU 정규화
h_normalized = h_tf / tf.cast(tf.sqrt(energy), h_tf.dtype)
h_cp = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c64)).copy()
self.buffer.put(h_cp)                            # CuPy GPU array
```

#### 2. RingBuffer: numpy CPU → CuPy GPU

```python
# 수정 전: CPU 메모리
self.buffer = np.zeros((maxlen,) + shape, dtype=np.complex64)

# 수정 후: GPU 메모리
self.buffer = cp.zeros((maxlen,) + shape, dtype=cp.complex64)
```

#### 3. process_slot: 채널 H2D 제거 + IQ bytes 직접

```python
# 수정 전: numpy frombuffer + pinned_H + set()
iq_int16 = np.frombuffer(iq_bytes, dtype='<i2')
self.pinned_iq_in[:] = iq_int16
self.pinned_H[:] = 0
# ... numpy channel copy ...
self.gpu_H.set(self.pinned_H, stream=self.stream)

# 수정 후: ctypes memmove + GPU 직접
ctypes.memmove(self.pinned_iq_in.ctypes.data, iq_bytes, len(iq_bytes))
# 채널 H: 이미 GPU에 있으므로 gpu_H에 직접 대입
self.gpu_H[i, :n] = channels_gpu[i][:n].astype(cp.complex128)
```

#### 4. 인덱스 사전계산: CuPy 직접 생성

```python
# 수정 전: numpy → cp.asarray
ext_idx = np.zeros(...); self.gpu_ext_idx = cp.asarray(ext_idx)

# 수정 후: CuPy 직접
ext_idx = cp.zeros(...); self.gpu_ext_idx = ext_idx
```

### 변경 사항 요약

| 항목 | v9.0 | v9.1 |
|------|------|------|
| 채널 H 전달 | TF→numpy→CuPy (GPU→CPU→GPU) | TF→DLPack→CuPy (GPU→GPU) |
| RingBuffer | numpy CPU | **CuPy GPU** |
| 채널 H2D | pinned_H → GPU (458KB) | **불필요** (이미 GPU) |
| IQ bytes 처리 | np.frombuffer → pinned | ctypes.memmove → pinned |
| 에너지 정규화 | numpy CPU | **TF GPU** |
| pinned_H 버퍼 | 할당됨 (458KB) | **제거** |
| numpy 의존성 | 전체 파이프라인 | pinned memory view + .npy만 |

---

## 수정 4: Sionna Proxy 커널 배치 최적화 (v9.2)

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-12 |
| **파일** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v9_cupy_full_gpu_pipeline.py` |
| **클래스** | `RingBuffer`, `ChannelProducer`, `GPUSlotPipeline`, `Proxy` |

### 문제

v9.1에서 **Python for-loop 내 개별 GPU 커널 실행**이 여전히 남아 있어 커널 런치 오버헤드가 큼:

| 병목 위치 | 커널 횟수 | 원인 |
|-----------|:---------:|------|
| `process_slot` 채널 복사 | ~42 | `for i in range(14)` + dtype 체크 + slice 3회 |
| `process_ofdm` 채널 읽기 | 14 lock + 14 GPU copy | `for _ in range(14): buffer.get()` |
| `ChannelProducer` 정규화 | ~250/batch | 원소별 `tf.reduce_sum` + `tf.sqrt` + DLPack × N |

### 수정 내용

#### 1. RingBuffer: `get_batch(n)` + `put_batch(batch)` 추가

```python
# 수정 전: 14회 lock + 14회 GPU copy
channels = []
for _ in range(14):
    channels.append(self.channel_buffer.get())  # 각각 lock + copy

# 수정 후: 1회 lock + 1회 GPU slice copy
channels = self.channel_buffer.get_batch(14)  # (14, 2048) GPU array, 1회
```

- `get_batch(n)`: 연속 슬라이스이면 단일 GPU copy (14배 빠름)
- `put_batch(batch)`: 1회 lock으로 N개 삽입 (ChannelProducer용)

#### 2. ChannelProducer: 배치 TF 정규화 + 1회 DLPack

```python
# 수정 전: 원소별 loop (N ≈ 4200, ~250 GPU kernels/batch)
for i in range(h_delay.shape[0]):
    h_tf = h_delay[i]
    energy = tf.reduce_sum(tf.abs(h_tf)**2)
    h_normalized = h_tf / tf.sqrt(energy)
    h_cp = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c64)).copy()
    self.buffer.put(h_cp)

# 수정 후: 배치 벡터 연산 (~5 GPU kernels/batch)
h_c128 = tf.cast(h_delay, tf.complex128)
energy = tf.reduce_sum(tf.abs(h_c128)**2, axis=-1, keepdims=True)  # (N, 1)
h_norm = h_delay / tf.cast(tf.sqrt(energy), h_delay.dtype)         # 배치
h_cp_batch = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(h_c64)).copy()  # 1회
self.buffer.put_batch(h_cp_batch)                                  # 1회 lock
```

#### 3. process_slot 채널 복사: 2D slice assign

```python
# 수정 전: for-loop (~42 kernels)
self.gpu_H[:] = 0
for i in range(n_ch):
    ch = channels_gpu[i]
    n = min(len(ch), self.fft_size)
    self.gpu_H[i, :n] = ch[:n].astype(cp.complex128)

# 수정 후: 단일 2D slice (~2 kernels)
self.gpu_H[:] = 0
self.gpu_H[:n_ch, :n_w] = channels_gpu[:n_ch, :n_w].astype(cp.complex128)
```

#### 4. process_ofdm: 2D array 전달

```python
# channels가 이제 (14, 2048) 2D CuPy array → process_slot에 그대로 전달
# 리스트 → 개별 인덱싱 오버헤드 제거
```

### 효과 요약

| 항목 | v9.1 (기존) | v9.2 (배치) | 개선 |
|------|:-----------:|:-----------:|:----:|
| RingBuffer 읽기 | 14 lock + 14 copy | 1 lock + 1 copy | 14x |
| 채널 복사 (process_slot) | ~42 커널 | ~2 커널 | 21x |
| ChannelProducer (per batch) | ~250 커널 | ~5 커널 | 50x |
| **총 GPU 커널/슬롯** | **~65** | **~15** | **~4x** |

---

## 기존 수정 사항 (이전 세션)

### CSI 측정 디버그 로그 추가

| 항목 | 내용 |
|------|------|
| **파일** | `openair1/PHY/NR_UE_TRANSPORT/csi_rx.c` |
| **내용** | CSI 측정 결과 (RSRP, RI, PMI, SINR, CQI) 상세 로그 출력 추가 |
| **목적** | Sionna proxy 채널의 UE 측 영향 분석을 위한 디버그 |

추가된 로그 예시:
```
[NR_PHY]   [UE 0] CSI-RS Measurement Results:
[NR_PHY]     - Measurement Bitmap: 0x1e
[NR_PHY]     - RSRP: -46 dBm
[NR_PHY]     - Rank Indicator (RI): 1
[NR_PHY]     - PMI i1: [0, 0, 0]
[NR_PHY]     - PMI i2: 0
[NR_PHY]     - SINR: 15 dB       ← 수정 1 이후, 실측 SINR 반영
[NR_PHY]     - CQI: 11           ← SINR에 따라 가변
```

---

## 수정 5: OAI rfsimulator GPU IPC 모드 추가

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-24 |
| **파일** | `radio/rfsimulator/gpu_ipc.h`, `radio/rfsimulator/gpu_ipc.c`, `radio/rfsimulator/simulator.c`, `radio/rfsimulator/CMakeLists.txt` |
| **함수** | `gpu_ipc_init()`, `gpu_ipc_cleanup()`, `gpu_ipc_dl_write()`, `gpu_ipc_dl_read()`, `gpu_ipc_ul_write()`, `gpu_ipc_ul_read()` |
| **문제** | Socket 기반 통신(TCP)은 DL/UL 경로에서 불필요한 CPU 복사 + 커널 전환 발생. H100 NVL에서 소켓 오버헤드가 병목. |
| **해결** | CUDA IPC로 gNB/UE ↔ Proxy 간 GPU 메모리 직접 공유. `dlopen("libcudart.so")`으로 런타임 로드, 빌드 시 CUDA 의존성 제거. mmap 기반 공유 파일로 IPC 핸들 교환 및 동기화. |
| **빌드** | `USE_GPU_IPC` 컴파일 정의 + `-ldl` 링크 (`CMakeLists.txt`에 추가) |
| **실행** | `sudo RFSIM_GPU_IPC=1 ./nr-softmodem --rfsim ...` (포트 불필요) |
| **기대 효과** | DL/UL 소켓 제거로 ~1ms 지연 감소, CPU 복사 제거, NVIDIA Aerial 전환 대비 |

### 주요 구현 사항

- **gpu_ipc.h**: 공유 메모리 레이아웃 (512 bytes, 4개 IPC handle + sync flags + metadata)
- **gpu_ipc.c**: `dlopen` CUDA 로드, SERVER(gNB: cudaMalloc + IPC handle 생성), CLIENT(UE: handle 오픈)
- **simulator.c**: `device_init()`에 `RFSIM_GPU_IPC` 환경변수 체크, `write_internal()/read()`에 GPU IPC 분기
- **동기화**: `__atomic_store_n` / `__atomic_load_n` + `sched_yield()` spin-lock

### 데이터 흐름

```
DL: gNB H2D → [gpu_dl_tx] → Proxy GPU 채널처리 → [gpu_dl_rx] → UE D2H
UL: UE  H2D → [gpu_ul_tx] → Proxy GPU 복사     → [gpu_ul_rx] → gNB D2H
```

---

## 수정 6: Sionna Proxy v12 GPU IPC 모드 (v10 기반)

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-24 |
| **파일** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v12_gpu_ipc.py` |
| **클래스** | `GPUIpcInterface`, `GPUSlotPipeline.process_slot_ipc()` |
| **문제** | v10 소켓 모드에서 Proxy의 H2D/D2H가 매 슬롯 발생 (~120KB x 2). DL 채널 처리 경로에서 불필요한 호스트 경유. |
| **해결** | `GPUIpcInterface`: mmap + `cupy.cuda.runtime.ipcOpenMemHandle()`로 gNB 할당 GPU 버퍼에 직접 접근. `process_slot_ipc()`: GPU-to-GPU 채널 처리 (H2D/D2H 제거). Dual-mode (`--mode=socket`/`--mode=gpu-ipc`). |
| **기대 효과** | Proxy측 H2D/D2H 완전 제거 (~0.3ms 절감). GPU-to-GPU 데이터 경로로 소켓 오버헤드 제거. |

### 주요 구현 사항

- **GPUIpcInterface**: mmap으로 공유 메모리 오픈, 4개 GPU 버퍼 IPC handle을 CuPy 어레이로 매핑
- **process_slot_ipc()**: IPC GPU 버퍼에서 직접 int16 읽기 → GPU 채널 처리 → IPC GPU 버퍼에 결과 쓰기
- **UL 패스스루**: `gpu_ul_tx → gpu_ul_rx` GPU-to-GPU 복사 (채널 처리 없음)
- **Dual-mode**: `--mode=socket` (v10 호환), `--mode=gpu-ipc` (CUDA IPC)
- **docker-compose.yml**: `ipc: host` 추가, `/tmp/oai_gpu_ipc` 공유 볼륨 마운트

---

## 수정 7: GPU IPC 메모리 소유권 Proxy로 이전 + 동기화 안정화

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-24 |
| **파일 (OAI)** | `radio/rfsimulator/gpu_ipc.h`, `gpu_ipc.c`, `simulator.c` |
| **파일 (Proxy)** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v12_gpu_ipc.py` |

### 변경 1: GPU 메모리 소유권을 Proxy(SERVER)로 이전

| 변경 전 | 변경 후 |
|---------|---------|
| gNB(SERVER): `cudaMalloc` × 4 + IPC 핸들 생성 | **Proxy(SERVER)**: `cp.cuda.alloc()` × 4 + IPC 핸들 생성 |
| Proxy: `ipcOpenMemHandle` (CLIENT) | gNB: `cudaIpcOpenMemHandle` (CLIENT) |
| UE: `cudaIpcOpenMemHandle` (CLIENT) | UE: `cudaIpcOpenMemHandle` (CLIENT) |

**이유**: RAN 측(OAI)을 향후 Aerial 등으로 교체할 때, Proxy가 메모리를 소유하면 RAN 교체 시 메모리 관리 코드를 변경할 필요 없음. Proxy가 중앙 조정자 역할.

**구현**:
- `v12_gpu_ipc.py`: `GPUIpcInterface.init()` — CuPy `cp.cuda.alloc()`로 4개 GPU 버퍼 할당, `ipcGetMemHandle`로 핸들 export, shm 파일 생성 + magic/version/buf_size 기록
- `gpu_ipc.c`: `server_init()` 제거. `gpu_ipc_init()`에서 항상 CLIENT 경로(`client_init()`)만 실행. `gpu_ipc_cleanup()`에서 `cudaIpcCloseMemHandle`만 호출 (cudaFree 제거)

### 변경 2: DL write 블로킹 + UL write 비블로킹

**문제**: GPU IPC의 single-buffer 구조에서 `gpu_ipc_dl_write()`가 Proxy 소비 완료까지 블로킹 → L1 TX가 RU보다 느려짐 → PRACH `prach_list` overflow (assertion 실패)

**원인 분석**: 소켓 모드에서는 OS TCP 버퍼(~수 MB)가 비동기 전달하여 write가 즉시 리턴. GPU IPC는 single buffer + 동기 handshake라서 write가 Proxy 처리 완료까지 블로킹.

**해결**:
- `gpu_ipc_dl_write()`: 블로킹 유지 (모든 DL 데이터 순서대로 전달 보장)
- `gpu_ipc_ul_write()`: 비블로킹 skip (UE가 Proxy 대기로 블로킹되지 않도록)
- `rfsimulator_read()` gNB UL RX: UL 데이터 미도착 시 `usleep(1ms)`로 대기 (소켓 모드의 `epoll_wait` 타임아웃과 동일 역할)

> **참고**: 초기에는 `dl_write_count` atomic 카운터로 RU를 L1 TX에 pacing하는 방식을 시도했으나, TDD UL 슬롯에서 DL write가 발생하지 않아 영구 데드락 발생. 수정 8에서 제거됨.

### 변경 3: Proxy 파이프라인 사전 워밍업

**문제**: TensorFlow XLA 컴파일(첫 DL 슬롯에서 ~10초 소요)이 gNB DL write를 블로킹하여 전체 시스템 정지

**해결**: `v12_gpu_ipc.py`에 `_warmup_pipeline()` 추가. IPC 메인 루프 진입 전에 더미 데이터로 XLA 컴파일 + CUDA Graph 캡처를 완료.

### 시작 순서

```
1. Proxy 시작 → GPU 할당 + shm 생성 + 워밍업 → "Entering main loop..."
2. gNB 시작  → shm 대기 → IPC 핸들 open → "RU 0 RF started"
3. UE 시작   → shm 대기 → IPC 핸들 open → sync 시작
```

---

## 수정 8: TDD 데드락 수정 + 빌드 시스템 이슈

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-24 |
| **파일** | `radio/rfsimulator/simulator.c` |

### 문제 1: TDD 데드락 (`dl_write_count` pacing)

수정 7에서 `rfsimulator_read()`의 gNB UL RX 경로에 `dl_write_count` 카운터 기반 pacing을 도입했으나, TDD 구조에서 데드락 발생:

- TDD 10슬롯 중 L1 TX는 DL/Mixed 슬롯(8개)에서만 `gpu_ipc_dl_write()` 호출
- UL-only 슬롯(2개)에서는 `dl_write_count`가 증가하지 않음
- RU 스레드가 `dl_write_count` 증가를 무한 대기 → 전체 파이프라인 정지

**해결**: `dl_write_count` 기반 pacing 완전 제거. `rfsimulator_read()` gNB UL RX에서 UL 데이터 미도착 시 항상 `usleep(t->wait_timeout * 1000)` (1ms) 사용.

```c
// 수정 전 (TDD 데드락)
if (ul_rx_ready == 0) {
    while (dl_write_count <= last_dl_seq) sched_yield(); // UL 슬롯에서 영구 대기
}

// 수정 후 (안정)
if (ul_rx_ready == 0) {
    usleep(t->wait_timeout * 1000); // 항상 1ms sleep
    memset(samplesVoid, 0, ...);
    return nsamps;
}
```

### 문제 2: 빌드 시스템 (`librfsimulator.so` 별도 빌드)

`simulator.c` 수정 후 `ninja nr-softmodem`으로만 빌드 → 변경사항 미반영:

- `rfsimulator`는 별도 shared library (`librfsimulator.so`)로 빌드되며, `nr-softmodem` 타겟에 포함되지 않음
- `dlopen()`으로 런타임 로드되므로 빌드 의존성 없음

**해결**: `simulator.c`, `gpu_ipc.c`, `gpu_ipc.h` 수정 시 반드시 명시적으로 빌드:

```bash
cd ~/openairinterface5g_whan/cmake_targets/ran_build/build
ninja rfsimulator    # librfsimulator.so 재빌드 (필수!)
ninja nr-softmodem   # gNB 빌드 (simulator.c 변경과 무관)
ninja nr-uesoftmodem # UE 빌드
```

> **참고**: `.cursor/rules/oai-build.mdc` 룰로 기록하여 이후 자동 참조되도록 설정.

---

## 수정 9: v12 측정 시스템 업그레이드 (v11 이식)

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-02-24 |
| **파일** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v12_gpu_ipc.py` |
| **백업** | `vRAN_Socket/G0_Sionna_Channel_Proxy/v12_gpu_ipc_old.py` (원본 보존) |

### 문제

v12는 v10 기반으로 작성되어 측정 시스템이 단순 inline print 수준. v11에 구현된 정밀 계측(WindowProfiler, CUDA Event, E2E TDD 통계)이 누락됨.

| 항목 | v12_old (v10 기반) | v12_new (v11 이식) |
|------|:---:|:---:|
| 타이머 | `time.perf_counter()` 단일 | CPU + CUDA Event 듀얼 |
| 통계 | 순간값 1줄 print (100슬롯마다) | WindowProfiler 롤링 윈도우 (avg/p95/p99/max) |
| 구간 분리 | H2D/CH/GPU/D2H (4구간) | +NOISE_PREP 분리 (5구간) |
| CPU-GPU 차이 | 없음 | CPU-EVT 차이 출력 (CPU 오버헤드 정량화) |
| E2E 통계 | 없음 | TDD 프레임 단위 wall vs Proxy vs IPC+OAI |
| Proxy 레벨 | 없음 | OFDM_SLOT (CH_GET/CH_PAD/GPU_PROC) + PROXY_E2E (PROC/SEND/TOTAL) |

### 추가된 구성요소

**1. WindowProfiler 클래스** (v11에서 이식)
- 고정 길이 롤링 윈도우 (`deque(maxlen=window)`)
- `add()` 호출 시 자동으로 `report_interval`마다 통계 출력
- avg / p95 / p99 / max 계산

**2. GPUSlotPipeline 프로파일러 인스턴스** (6개)

| 인스턴스 | 모드 | 메트릭 |
|----------|------|--------|
| `profile_gpu` | Socket CPU+sync | H2D, CH_COPY, NOISE_PREP, GPU_COMPUTE, D2H, TOTAL |
| `profile_gpu_evt` | Socket CUDA Event | 동일 |
| `profile_gpu_diff` | Socket CPU-EVT 차이 | 동일 |
| `profile_ipc` | IPC CPU+sync | GPU_COPY_IN, CH_COPY, NOISE_PREP, GPU_COMPUTE, GPU_COPY_OUT, TOTAL |
| `profile_ipc_evt` | IPC CUDA Event | 동일 |
| `profile_ipc_diff` | IPC CPU-EVT 차이 | 동일 |

**3. CUDA Event 듀얼 타이머**
- `cp.cuda.Event()` 쌍으로 각 구간의 GPU 실제 실행시간 측정
- `cp.cuda.get_elapsed_time(start, end)`로 ms 단위 반환
- CPU 타이머와 동시 측정하여 CPU 측 지터(GIL, 스케줄러) 분리

**4. E2E TDD 프레임 통계** (IPC + Socket 양쪽)
- 10슬롯(TDD 프레임) 단위로 wall-clock vs Proxy 처리시간 비교
- IPC 모드: `IPC+OAI = wall - Proxy(DL+UL)` (소켓 모드의 `Socket+OAI`에 대응)
- `_ipc_process_dl()`, `_ipc_process_ul()`이 처리 시간(ms) 반환

**5. Proxy 레벨 프로파일러** (Socket 모드)
- `profile_ofdm`: CH_GET, CH_PAD, GPU_PROC, TOTAL
- `profile_proxy`: PROC, SEND, TOTAL

### 새 CLI 인수

| 인수 | 기본값 | 설명 |
|------|:------:|------|
| `--profile-interval` | 100 | 프로파일 리포트 주기 (슬롯 단위) |
| `--profile-window` | 500 | 롤링 윈도우 크기 |
| `--dual-timer-compare` / `--no-dual-timer-compare` | 활성화 | CUDA Event 듀얼 타이머 ON/OFF |


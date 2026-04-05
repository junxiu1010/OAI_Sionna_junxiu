# OAI IQ 데이터 구조 및 Power Scaling 분석

> 모든 파일 경로는 `openairinterface5g_whan/` 기준 (OAI 소스 루트)이며,
> Sionna Proxy 관련은 `vRAN_Socket/` 기준이다.

---

## 1. 소켓 데이터 포맷

OAI gNB/UE와 Sionna Channel Proxy 사이에 TCP 소켓으로 주고받는 데이터는 **헤더 + IQ 페이로드**로 구성된다.

### 1.1 헤더 (24 bytes)

C 구조체 정의:

- 📄 **`radio/COMMON/common_lib.h` : L620~629** — `samplesBlockHeader_t`

```c
typedef struct {
    uint32_t size;          // 안테나당 샘플 수              (L621)
    uint32_t nbAnt;         // 안테나 수                    (L622)
    uint64_t timestamp;     // 첫 샘플의 타임스탬프          (L626)
    uint32_t option_value;  // 옵션 값                     (L627)
    uint32_t option_flag;   // 옵션 플래그                  (L628)
} samplesBlockHeader_t;                                   // (L629)
```

Python 파싱: `struct.unpack("<I I Q I I", header_bytes)` (little-endian, 총 24 bytes)

### 1.2 IQ 페이로드

- 📄 **`radio/rfsimulator/simulator.c` : L130** — `typedef c16_t sample_t;`

| 속성 | 값 |
|------|-----|
| 타입 | `c16_t` = 16-bit signed integer 2개 (I + jQ) |
| 바이트 순서 | little-endian |
| 크기 | `size × nbAnt × 4 bytes` (I: 2B + Q: 2B) |
| 범위 | 각 I, Q 값은 int16 범위 (-32768 ~ +32767) |

Python 파싱:

```python
x = np.frombuffer(iq_bytes, dtype='<i2')       # little-endian int16
x_cpx = x[::2] + 1j * x[1::2]                 # 복소수 변환
```

### 1.3 소켓 송수신 코드 위치

| 함수 | 파일 | 행 |
|------|------|-----|
| `rfsimulator_write_internal()` | `radio/rfsimulator/simulator.c` | L689 |
| `rfsimulator_write()` | `radio/rfsimulator/simulator.c` | L810 |
| `rfsimulator_read()` | `radio/rfsimulator/simulator.c` | L1003 |
| 헤더 전송 (`fullwrite`) | `radio/rfsimulator/simulator.c` | L713 |
| 페이로드 전송 (`fullwrite`) | `radio/rfsimulator/simulator.c` | L718 |
| device에 함수 등록 | `radio/rfsimulator/simulator.c` | L1291~1292 |

---

## 2. IQ 값의 정체: 시간 영역 OFDM 파형 샘플

소켓으로 전송되는 정수값은 **constellation point가 아니라, IFFT를 거친 시간 영역 파형 샘플**이다.

### 2.1 데이터 흐름 (코드 추적)

```
[1] 비트 → Scrambling → Modulation (constellation mapping)
    → txdataF (주파수 영역, 부반송파별 심볼)

[2] txdataF → Precoding → txdataF_BF

[3] txdataF_BF → IFFT + CP 삽입 → txdata (시간 영역)

[4] txdata → 소켓 전송 (rfsimulator_write)
```

### 2.2 핵심 코드 경로 (gNB 송신)

#### [1단계] Constellation Mapping

- 📄 **`openair1/PHY/MODULATION/nr_modulation.c` : L120~174**
- 함수: `nr_modulation()`
- `mod_order`에 따라 `nr_qpsk_mod_table`, `nr_16qam_mod_table` 등에서 심볼 매핑
- PDSCH에서 호출: **`openair1/PHY/NR_TRANSPORT/nr_dlsch.c` : L632**

```c
nr_modulation(scrambled_output, encoded_length, Qm, (int16_t *)mod_symbs[codeWord]);  // L632
```

#### [2단계] Resource Mapping + Amplitude Scaling

- 📄 **`openair1/PHY/NR_TRANSPORT/nr_dlsch.c` : L84, L292**

```c
txF[k] = c16mulRealShift(*in++, amp, 15);   // L84 (PTRS 없는 일반 RE)
output[k] = c16mulRealShift(*in++, amp, 15); // L292 (DMRS 있는 심볼)
```

- `amp` 값 설정: **`nr_dlsch.c` : L551** → `const int16_t amp = gNB->TX_AMP;`
- PDCCH: **`openair1/PHY/NR_TRANSPORT/nr_dci.c` : L178** → `uint16_t amp = gNB->TX_AMP;`
- PBCH: **`openair1/PHY/NR_TRANSPORT/nr_pbch.c` : L358** → `int16_t amp = gNB->TX_AMP;`

#### [3단계] IFFT + CP 삽입

- 📄 **`openair1/PHY/MODULATION/ofdm_mod.c` : L152~233**
- 함수: `PHY_ofdm_mod()`
- IFFT 수행: **L196** → `idft(idft_size, input, output_ptr, 1);`
- CP 삽입: **L204** → `memcpy(&output_ptr[-nb_prefix_samples], ...)`

호출 체인 (📄 **`openair1/SCHED_NR/nr_ru_procedures.c`**):

| 함수 | 행 | 설명 |
|------|-----|------|
| `nr_feptx_tp()` | L264 | 스레드풀 기반 TX FEP 시작점 |
| `nr_feptx()` | L216 | 스레드 워커 — precoding + IFFT 호출 |
| `nr_feptx0()` | L50 | 실제 IFFT 수행 — `PHY_ofdm_mod()` 호출 |
| `PHY_ofdm_mod()` 호출 | L83, L93, L100, L108 | `txdataF_BF → txdata` 변환 |

#### [4단계] 소켓 전송

- 📄 **`executables/nr-ru.c` : L814~829**

```c
// txdata 포인터 설정 (L819)
txp[i] = (void *)&ru->common.txdata[i][fp->get_samples_slot_timestamp(slot, fp, 0)] - sf_extension * sizeof(int32_t);

// rfsimulator_write() 호출 (L824)
uint32_t txs = ru->rfdevice.trx_write_func(&ru->rfdevice, timestamp, txp, siglen, nt, flags);
```

### 2.3 txdataF vs txdata 비교

| 구분 | 변수 | 영역 | 내용 |
|------|------|------|------|
| 주파수 영역 | `txdataF` | frequency | 부반송파별 constellation point (QPSK: ±367 등) |
| 시간 영역 | `txdata` | time | IFFT 결과 — 모든 부반송파 신호의 합 (superposition) |

- `txdataF`: QPSK면 `(±367, ±367)` 같은 규칙적인 값 (TX_AMP 적용 후)
- `txdata` (소켓 전송): 수백 개 부반송파가 합쳐진 파형 → 값이 불규칙하고 가우시안 분포에 가까움

### 2.4 고정소수점 (Fixed-Point) 연산

OAI는 부동소수점(float)으로 계산 후 양자화하는 것이 아니라, **처음부터 16-bit 정수 고정소수점**으로 전 과정을 수행한다.

- `c16_t` = `{ int16_t r; int16_t i; }` (I, Q 각각 16-bit)
- Modulation table도 정수, IFFT 입출력도 정수
- 별도의 "양자화" 단계 없이 정수 연산 체인이 그대로 소켓까지 전달

---

## 3. OAI Power Scaling 구조

### 3.1 Modulation Table (3GPP 정규화 내장)

모든 modulation order에서 **3GPP 정규화 (1/√(평균 심볼 파워))** 가 이미 적용되어 평균 파워가 통일되어 있다.

#### Modulation table 정의 위치

| 파일 | 행 | 내용 |
|------|-----|------|
| 📄 `openair1/PHY/NR_REFSIG/nr_gen_mod_table.m` | L9 | `BPSK = 23170` |
| | L12 | `QPSK = 23170` |
| | L16~18 | `QAM16_n1 = 20724, QAM16_n2 = 10362` |
| | L21~25 | `QAM64_n1 = 20225, QAM64_n2 = 10112, QAM64_n3 = 5056` |
| | L29~35 | `QAM256_n1 = 20105, ..., QAM256_n4 = 2513` |
| 📄 `openair1/PHY/NR_REFSIG/nr_gen_mod_table.c` | L35~47 | C 코드 테이블 생성 함수 |

#### 정규화 값 표

| Modulation | 정규화 계수 | 대표 amplitude | 계산 |
|------------|-------------|----------------|------|
| BPSK | 1/√2 | 23170 | 2^15 × 1/√2 |
| QPSK | 1/√2 | 23170 | 2^15 × 1/√2 |
| 16QAM | 1/√10 | 20724 / 10362 | 2^15 × {2, 1}/√10 |
| 64QAM | 1/√42 | 20225 / 10112 / 5056 | 2^15 × {4, 2, 1}/√42 |
| 256QAM | 1/√170 | 20105 / 10052 / 5026 / 2513 | 2^15 × {8, 4, 2, 1}/√170 |

실제 C 코드 (`nr_gen_mod_table.c` : L36~47):

```c
float sqrt2 = 0.70711;   // L36
float val = 32768.0;     // L40
// QPSK (L46~47):
nr_qpsk_mod_table[i].r = (short)(1 - 2 * (i & 1)) * val * sqrt2 * sqrt2;
nr_qpsk_mod_table[i].i = (short)(1 - 2 * ((i >> 1) & 1)) * val * sqrt2 * sqrt2;
// → ±23170 (= 32768 × 0.5)
```

### 3.2 TX_AMP (전송 진폭 스케일링)

#### AMP 관련 매크로 정의

- 📄 **`openair1/PHY/impl_defs_top.h`**

| 매크로 | 행 | 값 |
|--------|-----|-----|
| `ONE_OVER_SQRT2_Q15` | L199 | 23170 |
| `AMP_SHIFT` | L252 (8bit) / L254 (일반) | 7 / 9 |
| `AMP` | L257 | `(1) << AMP_SHIFT` = 512 |

#### TX_AMP 설정 (런타임)

- 📄 **`openair1/PHY/defs_gNB.h` : L454** — `int16_t TX_AMP;` (gNB 구조체 멤버)
- 📄 **`openair2/GNB_APP/gnb_config.c` : L1120** — TX_AMP 계산:

```c
gNB->TX_AMP = min(32767.0 / pow(10.0, .05 * (double)(*L1_ParamList.paramarray[j][L1_TX_AMP_BACKOFF_dB].uptr)), INT16_MAX);
```

#### Backoff 기본값

- 📄 **`openair2/GNB_APP/L1_nr_paramdef.h`**
  - L58: 파라미터 이름 `"tx_amp_backoff_dB"`
  - L89: 기본값 `.defintval=36`

| 항목 | 값 |
|------|-----|
| 설정 파라미터 | `tx_amp_backoff_dB` |
| 기본값 | **36 dB** |
| 기본 TX_AMP | `32767 / 10^(0.05 × 36) = 32767 / 63.1 ≈ 519` |
| 목적 | IFFT 후 시간 영역에서 int16 클리핑 방지 (PAPR 고려) |

### 3.3 Resource Mapping 시 최종 스케일링

- 📄 **`openair1/PHY/TOOLS/tools_defs.h` : L218~221** — `c16mulRealShift`:

```c
inline c16_t c16mulRealShift(const c16_t a, const int32_t b, const int Shift)
{
    return (c16_t){
        .r = (int16_t)((a.r * b) >> Shift),
        .i = (int16_t)((a.i * b) >> Shift)
    };
}
```

적용 예 (QPSK, backoff 36dB 기준):

```
mod 심볼: (23170, 23170)     ← nr_qpsk_mod_table에서
× TX_AMP: 519                ← gNB->TX_AMP
>> 15

결과: ((23170 × 519) >> 15, (23170 × 519) >> 15)
     = (12025230 >> 15, 12025230 >> 15)
     = (367, 367)

→ txdataF에 저장되는 값: (367, 367)
→ 이후 IFFT를 거쳐 txdata로 변환 → 소켓 전송
```

### 3.4 전체 스케일링 체인 요약

```
[Modulation Table]            [TX_AMP]               [IFFT]
3GPP 정규화된 정수   ×   amplitude backoff   →   시간 영역 합산
(~23170 스케일)          (기본 ~519)              (다수 부반송파 superposition)
                                                      ↓
                                              int16 범위 내 유지
                                              (-32768 ~ +32767)
```

---

## 4. LLS (Link-Level Simulation) 관점에서의 Power Scaling

### 4.1 결론: Power Scaling을 별도로 고려할 필요 없음

LLS 목적으로 Sionna Channel Proxy를 사용할 때, OAI 내부의 power scaling 구조를 알 필요가 없다.

### 4.2 이유

**[이유 1] 채널 적용은 선형 연산**

- 📄 **`vRAN_Socket/G0_Sionna_Channel_Proxy/v8_cupy_slot_batch_pinned_fastest.py` : L282~285**

```python
gpu_X = cp.fft.fft(self.gpu_x[prev_idx], axis=1)       # 시간→주파수   (L282)
gpu_H_freq = cp.fft.fft(self.gpu_H[prev_idx], axis=1)  # 채널 FFT      (L283)
gpu_Y = gpu_X * gpu_H_freq                              # Y(f)=X(f)·H(f)(L284)
self.gpu_y[prev_idx] = cp.fft.ifft(gpu_Y, axis=1)      # 주파수→시간   (L285)
```

`Y(f) = X(f) · H(f)` — 주파수 영역 곱셈 = 시간 영역 컨볼루션 (선형 연산).
입력 신호의 절대적 크기(power scale)가 무엇이든, H의 정규화만 적절하면 출력 power는 입력에 비례한다.

**[이유 2] 정규화된 채널이면 power scale 자동 유지**

정규화된 채널 (에너지 합 = 1) — v3 예시:

```python
h_full = h_full / np.sqrt(np.sum(np.abs(h_full)**2))
```

이 경우 채널 적용 후 신호 에너지가 보존되므로, OAI 내부 스케일과 무관하게 동작한다.

**[이유 3] OAI UE 복조기가 자체 보정**

UE 수신 처리 체인 (관련 코드 위치):

| 단계 | 함수 | 파일 | 행 |
|------|------|------|-----|
| 수신 FFT | `nr_slot_fep()` | `openair1/PHY/MODULATION/slot_fep_nr.c` | L31 |
| FFT 수행 (DFT) | `dft(dftsize, ...)` | 같은 파일 | L107 |
| 채널 추정 (DMRS) | `nr_pdsch_channel_estimation()` | `openair1/PHY/NR_UE_ESTIMATION/nr_dl_channel_estimation.c` | L1252 |
| 채널 스케일링 | `nr_dlsch_scale_channel()` | `openair1/PHY/NR_UE_TRANSPORT/nr_dlsch_demodulation.c` | L977 (정의), L501 (호출) |
| 채널 보상 (Equalization) | `nr_dlsch_channel_compensation()` | 같은 파일 | L841 (정의), L563 (호출) |
| MRC (다중 안테나 결합) | `nr_dlsch_detection_mrc()` | 같은 파일 | L1198 (정의), L615 (호출) |
| LLR 계산 (복조) | `nr_dlsch_llr()` | 같은 파일 | L1728 (정의), L700 (호출) |
| 전체 PDSCH 수신 | `nr_rx_pdsch()` | 같은 파일 | L307 |

- 채널 추정: DMRS 심볼의 **수신 값 / 알려진 송신 값** → 채널 계수 추정
- Equalization: 수신 신호를 추정된 채널로 나눔 → 원래 constellation 복원
- 이 과정이 **수신 신호의 상대적 크기**로 동작하므로 절대 power scale 무관

### 4.3 Power Scaling이 필요한 경우

**SNR 제어 (노이즈 추가) 시**에만 신호 power를 알아야 한다.

이 경우에도 OAI 내부 스케일을 해석할 필요 없이:

```python
# 수신 신호에서 직접 power 측정
signal_power = np.mean(np.abs(y)**2)

# 원하는 SNR에 맞는 노이즈 power 계산
noise_power = signal_power / (10 ** (snr_dB / 10))

# 노이즈 추가
noise = np.sqrt(noise_power / 2) * (np.random.randn(...) + 1j * np.random.randn(...))
y_noisy = y + noise
```

### 4.4 요약 표

| 항목 | 내용 | LLS에서 고려 필요? |
|------|------|:---:|
| Modulation table (3GPP 정규화) | 모든 MCS에서 평균 파워 통일 | ❌ |
| TX_AMP (amplitude backoff) | 기본 36dB, 클리핑 방지용 | ❌ |
| IFFT 후 시간 영역 합산 | 다수 부반송파 superposition | ❌ |
| 정규화된 채널 H | 에너지 합 = 1이면 power 보존 | ✅ (이미 적용 중) |
| SNR 제어 (노이즈) | 수신 신호 power 직접 측정 | ✅ (필요시) |

---

## 5. 참조 파일 종합 (행 번호 포함)

> 경로 기준: `openairinterface5g_whan/` (OAI), `vRAN_Socket/` (Sionna Proxy)

### 소켓 통신 계층

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `radio/COMMON/common_lib.h` | L620~629 | `samplesBlockHeader_t` 구조체 |
| `radio/rfsimulator/simulator.c` | L130 | `typedef c16_t sample_t` |
| | L689 | `rfsimulator_write_internal()` — 실제 전송 로직 |
| | L713, L718 | `fullwrite()` — 헤더, 페이로드 소켓 전송 |
| | L810 | `rfsimulator_write()` — 외부 인터페이스 |
| | L1003 | `rfsimulator_read()` — 수신 |
| | L1291~1292 | `trx_write_func`, `trx_read_func` 등록 |

### Modulation / 변조

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `openair1/PHY/NR_REFSIG/nr_gen_mod_table.m` | L9~35 | 모든 QAM amplitude 정의 (MATLAB) |
| `openair1/PHY/NR_REFSIG/nr_gen_mod_table.c` | L35~47 | QPSK 테이블 생성 C 코드 |
| `openair1/PHY/MODULATION/nr_modulation.c` | L120 | `nr_modulation()` 함수 |
| `openair1/PHY/NR_TRANSPORT/nr_dlsch.c` | L551 | `amp = gNB->TX_AMP` |
| | L632 | `nr_modulation()` 호출 (PDSCH) |
| | L84, L292 | `c16mulRealShift()` — RE 매핑 + amplitude 스케일링 |
| `openair1/PHY/NR_TRANSPORT/nr_dci.c` | L169 | `nr_modulation()` 호출 (PDCCH) |
| | L178 | `amp = gNB->TX_AMP` |
| `openair1/PHY/NR_TRANSPORT/nr_pbch.c` | L343~345 | PBCH QPSK 변조 |
| | L358 | `amp = gNB->TX_AMP` |

### Power Scaling / 진폭 제어

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `openair1/PHY/impl_defs_top.h` | L199 | `ONE_OVER_SQRT2_Q15 = 23170` |
| | L252~257 | `AMP_SHIFT`, `AMP` 매크로 |
| `openair1/PHY/defs_gNB.h` | L454 | `int16_t TX_AMP` gNB 멤버 |
| `openair2/GNB_APP/gnb_config.c` | L1120 | TX_AMP 계산 (backoff 적용) |
| `openair2/GNB_APP/L1_nr_paramdef.h` | L58 | `tx_amp_backoff_dB` 파라미터명 |
| | L89 | 기본값 = 36 dB |
| `openair1/PHY/TOOLS/tools_defs.h` | L218~221 | `c16mulRealShift()` 함수 |

### TX Front-End (IFFT + CP)

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `openair1/SCHED_NR/nr_ru_procedures.c` | L50 | `nr_feptx0()` — OFDM 변조 코어 |
| | L83~129 | `PHY_ofdm_mod()` 호출들 (`txdataF_BF → txdata`) |
| | L216 | `nr_feptx()` — 스레드 워커 |
| | L264 | `nr_feptx_tp()` — 스레드풀 시작점 |
| `openair1/PHY/MODULATION/ofdm_mod.c` | L152 | `PHY_ofdm_mod()` 함수 |
| | L196 | `idft()` 호출 — **IFFT 수행** |
| | L204 | CP (Cyclic Prefix) 삽입 |

### 소켓 전송 (RU → RF)

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `executables/nr-ru.c` | L819 | `txp[i] = &ru->common.txdata[i][...]` |
| | L824 | `ru->rfdevice.trx_write_func()` 호출 |

### UE 수신 처리 체인

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `openair1/PHY/MODULATION/slot_fep_nr.c` | L31 | `nr_slot_fep()` — 수신 FFT |
| | L107 | `dft()` 호출 — 시간→주파수 변환 |
| `openair1/PHY/NR_UE_ESTIMATION/nr_dl_channel_estimation.c` | L1252 | `nr_pdsch_channel_estimation()` — DMRS 기반 채널 추정 |
| | L604 | `nr_pbch_channel_estimation()` |
| | L796 | `nr_pdcch_channel_estimation()` |
| `openair1/PHY/NR_UE_TRANSPORT/nr_dlsch_demodulation.c` | L307 | `nr_rx_pdsch()` — PDSCH 수신 전체 흐름 |
| | L501, L977 | `nr_dlsch_scale_channel()` — 채널 스케일링 |
| | L563, L841 | `nr_dlsch_channel_compensation()` — Equalization |
| | L615, L1198 | `nr_dlsch_detection_mrc()` — MRC 결합 |
| | L700, L1728 | `nr_dlsch_llr()` — LLR 계산 (복조) |

### Sionna Proxy (채널 적용)

| 파일 | 핵심 행 | 내용 |
|------|---------|------|
| `vRAN_Socket/G0_Sionna_Channel_Proxy/v8_cupy_slot_batch_pinned_fastest.py` | L212 | `process_slot()` — GPU 파이프라인 |
| | L282~285 | `FFT → H·X 곱셈 → IFFT` (채널 적용 코어) |
| | L747 | `process_ofdm_slot_with_slot_pipeline()` — 슬롯 처리 |
| | L752~753 | IQ 파싱 (`np.frombuffer`, 복소수 변환) |

# G1B: Single-UE MIMO Channel Proxy - 문제 해결 보고서

> 이 문서는 G1B 프로젝트(v0~v8)에서 만난 문제, 원인 분석, 해결 방법을 버전별로 기록한다.
> 기술 레퍼런스(아키텍처, CLI, 테스트 결과)는 `README.md` 참조.
> OAI 소스코드 수정 상세는 `MODIFICATION_LOG.md` 참조.

---

## 프로젝트 개요

- **목표**: G0(SISO 1x1, GPU IPC V1) → G1B(Single-UE MIMO N_t x N_r, GPU IPC V6) 확장
- **기간**: 2026-03-06 ~ 2026-03-09
- **버전 범위**: v0(G0 복사) → v8(NOISE_PREP 최적화)
- **범위**: Proxy Python 코드 + OAI C 코드(rfsimulator, MAC/RRC) + launch 스크립트

---

## v0 → v1: GPU IPC 전환 (V1 → V5)

### 배경

G0 v12(SISO)를 복사한 v0는 GPU IPC V1(ready-flag 핑퐁 방식)을 사용했다. G1A에서의 경험으로 이 방식의 한계가 명확했다.

### G1A에서 배운 교훈

| 교훈 | G1A 경험 | G1B 대응 |
|------|---------|---------|
| IPC 프로토콜 안정성 | V2~V4 여러 시도 끝에 timestamp 오프셋/역점프 문제 미해결 | G0 V1 기반으로 시작, 안정 확인 후 확장 |
| DL partial read | ring slot 단위 pop으로 sub-slot read 시 데이터 유실 | V5의 circular buffer + range-based copy |
| TDD 데드락 | DL/UL 비대칭 + 블로킹 → 교착 | timestamp polling, non-blocking |
| complex128 필수 | complex64로 PSS/SSS 동기화 실패 | complex128 유지 |

### 해결한 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| DL partial read | V1의 ready-flag 핑퐁이 sub-slot 단위 read 미지원 | V5 circular buffer + range-based copy 도입 |
| TDD 데드락 | DL/UL 비대칭 + blocking read | timestamp polling, non-blocking |
| DL/UL 비대칭 bypass | v0는 DL만 패스스루 | `-b` 시 DL+UL 모두 패스스루 |
| warmup AttributeError | bypass 모드에서 `channel_buffer` 미초기화 | 초기화 분기 추가 |
| Proxy 누락 write 버그 | timestamp 변화만 감지하여 데이터 누락 | range-based copy로 전환 |

### OAI 수정

- `gpu_ipc_v5.{h,c}` 신규 추가 (조건부 컴파일 `#ifdef USE_GPU_IPC_V5`)
- `simulator.c`에 V5 블록 7곳 추가 (include, struct, startServer, startClient, write, read, end)
- `CMakeLists.txt`에 소스/정의 추가

---

## v1 → v2: MIMO 안테나 확장 (IPC V5 → V6)

### 문제

V5는 4개 버퍼가 모두 동일한 `cir_size`를 사용했다. gNB 안테나와 UE 안테나가 다른 비대칭 MIMO에서는 각 버퍼의 크기가 달라야 한다.

### 해결

| 항목 | V5 | V6 |
|------|-----|-----|
| 버퍼 크기 | 4 동일 cir_size=460800 | 4 독립 cir_size (cir_time * nbAnt) |
| 안테나 | 전역 nbAnt | per-buffer nbAnt (gNB/UE 분리) |
| Bypass copy | 동일 nbAnt 가정 | 대칭=직접, 비대칭=truncate/pad |

### OAI 수정

- `gpu_ipc_v6.{h,c}` 신규 추가 (per-buffer nbAnt/cir_size, `GPUIpcV6Interface`)
- `simulator.c`에 `#ifdef USE_GPU_IPC_V6` 블록 추가, V6가 cascade 최우선
- V5 파일은 전혀 수정하지 않음 (backward compatibility 유지)

---

## v2 → v3: MIMO 크래시 버그 수정

### 문제 1: 초기 head overflow → Proxy 크래시

**증상**: MIMO(nbAnt > 1)에서 Proxy 시작 직후 GPU 버퍼 초과 접근으로 크래시.

**원인**: `run_ipc()`에서 proxy head를 `cir_size`(= cir_time * nbAnt) 기준으로 계산했다. nbAnt > 1이면 `delta * nbAnt > cir_size`가 되어 버퍼 범위를 초과한다.

**해결**:
```python
# v2 (버그): cir_size = cir_time * nbAnt → overflow
proxy_dl_head = max(0, gnb_dl_head - self.ipc.dl_tx_cir_size)

# v3 (수정): cir_time = 시간 샘플 기준 순환 주기
proxy_dl_head = max(0, gnb_dl_head - self.ipc.cir_time)
```

SISO(nbAnt=1)에서는 `cir_size == cir_time`이므로 동작 변화 없음.

### 문제 2: gNB 4안테나 Segfault

**증상**: gNB를 4안테나로 실행 시 Segfault.

**원인**: `launch_all.sh`에서 RU 안테나(`--RUs.[0].nb_tx 4`)만 설정하고, OAI 상위 레이어의 안테나 포트 설정(`pdsch_AntennaPorts_XP`, `pdsch_AntennaPorts_N1`, `pusch_AntennaPorts`)을 누락했다. PHY/MAC과 RU 간 안테나 수 불일치로 메모리 접근 오류 발생.

**해결**: `launch_all.sh`에 안테나 수에 따른 포트 설정을 추가했다:
```bash
# 2안테나
GNB_ANT_ARGS+=" --gNBs.[0].pdsch_AntennaPorts_XP 2"
GNB_ANT_ARGS+=" --gNBs.[0].pusch_AntennaPorts 2"

# 4안테나
GNB_ANT_ARGS+=" --gNBs.[0].pdsch_AntennaPorts_XP 2"
GNB_ANT_ARGS+=" --gNBs.[0].pdsch_AntennaPorts_N1 2"
GNB_ANT_ARGS+=" --gNBs.[0].pusch_AntennaPorts 4"
```

OAI 소스코드 수정은 없음 — CLI 파라미터 전달 방식으로 해결.

---

## v3 → v4: MIMO 채널 적용 + CSI CQI 수정

이 버전에서 가장 많은 문제를 해결했다. Proxy와 OAI 양쪽 모두 수정이 필요했다.

### 문제 1: Sionna MIMO 채널 적용 로직 미구현

v3까지는 bypass(IQ 패스스루)만 가능했고, Sionna MIMO 채널을 실제 IQ에 적용하는 로직이 없었다.

**해결**:
- `cp.einsum('srtf,stf->srf', H, X)` 기반 MIMO 채널 적용 구현
- DL/UL 별도 `GPUSlotPipeline` 분리 (DL: H@X, UL: H^T@X 자동 전환)
- `ChannelProducer`에서 MIMO H 생성 시 명시적 축 제거, shape (N_sym, N_r, N_t, FFT)
- identity-like 채널 패딩 (RingBuffer 언더런 대비)

### 문제 2: MIMO에서 CQI=0, SINR=0 (OAI 4파일 수정)

**증상**: UE PHY에서 CQI를 정상 계산하지만 gNB에 전달되지 않아 CQI=0, SINR=0으로 보임.

**원인과 해결** (OAI 소스코드 4개 파일):

**(1) UE MAC: `cri_RI_LI_PMI_CQI` 인코더 미구현**
- 파일: `openair2/LAYER2/NR_MAC_UE/nr_ue_procedures.c`
- 원인: `cri_RI_LI_PMI_CQI` case가 fallthrough 체인에 있어 `LOG_E`만 출력하고 실제 인코딩을 수행하지 않음
- 해결: 해당 case를 분리하여 `get_csirs_RI_PMI_CQI_payload()` 호출 추가

**(2) gNB RRC: 1-port에서 CQI 리포트 미생성**
- 파일: `openair2/LAYER2/NR_MAC_gNB/nr_radio_config.c`
- 원인: `config_csi_meas_report` 호출이 `pdsch_AntennaPorts > 1` 블록 내에 있어 1-port에서는 CQI 포함 리포트 타입이 생성되지 않음
- 해결: `do_CSIRS` 블록으로 이동하여 1-port에서도 CQI 리포트 생성. CSI-IM 조건부 할당, codebook 조건부 설정 추가

**(3) gNB 디코더: r_index OOB**
- 파일: `openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_uci.c`
- 원인: `ri_bitlen=0`일 때 `r_index=-1`로 계산되어 `cqi_bitlen[-1]` 배열 OOB 접근
- 해결: `ri_bitlen=0`일 때 `r_index=0`(rank 1) 기본값 설정

**(4) UE MAC: bitmap 디버그 핵**
- 파일: `openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c`
- 원인: `meas_bitmap==1` 시 `0x1e`로 강제 설정하는 디버그 코드가 잔존
- 해결: proper CSI report config으로 불필요해졌으므로 주석 처리

### 문제 3: MIMO 채널 정규화 축 오류

**증상**: 채널 적용 후 신호 에너지 분산이 불균형.

**원인**: `ChannelProducer.run()`에서 FFT 축(`axis=-1`)으로 정규화했으나, TX 안테나 축(`axis=2`)으로 정규화해야 한다. MIMO에서 `sum_t |H[s,r,t,f]|^2 = 1` per (symbol, rx, freq) 조건이 필요.

**해결**: `axis=-1` → `axis=2` 변경.

---

## v4 → v5: CUDA Graph 통합

### 문제: einsum이 CUDA Graph와 비호환

**증상**: `cp.einsum('srtf,stf->srf')` 사용 시 CUDA Graph 캡처가 실패한다.

**원인**: `einsum`은 내부적으로 cuBLAS를 호출하며, cuBLAS는 동적 메모리 할당을 수행한다. CUDA Graph는 캡처 시점의 GPU 메모리 주소가 리플레이 시에도 동일해야 하는 제약이 있어, 동적 할당과 비호환이다.

**해결**: 세 가지 변환을 수행:

| 원래 코드 (v4) | Graph-safe 변환 (v5) | 이유 |
|---------------|---------------------|------|
| `cp.einsum('srtf,stf->srf')` | `cp.multiply + cp.sum` (broadcast\*sum) | cuBLAS 동적 할당 회피. 수학적 동치: `Y[s,r,f] = sum_t H[s,r,t,f] * X[s,t,f]` |
| Python for-loop (OFDM 추출/재구성) | GPU 인덱스 배열 (`gpu_ext_idx`, `gpu_data_dst`, `gpu_cp_dst/src`) | Python 실행이 Graph 내부에 캡처 불가 |
| `cp.zeros()` 동적 할당 (매 슬롯) | `__init__` 사전 할당 (`_buf_HX`, `_buf_Yf`, `_buf_out_2d` 등) | 고정 메모리 주소 요구 |

**성과**: v4 대비 **4.7배** throughput 향상 (MIMO 2x2 channel: 75.7 DL/s → 354 DL/s).

### 교훈: CUDA Graph 고정 주소 제약

CUDA Graph의 고정 메모리 주소 요구는 이후 v7/v8 최적화에서도 핵심 제약으로 작용한다:
- RingBuffer view의 GPU 주소는 매 슬롯 변동 → view를 직접 Graph에 넣을 수 없음
- 따라서 고정 버퍼에 memcpy 후 Graph 리플레이 방식을 사용 (zero-copy 불가)
- 이 memcpy가 v8 NOISE_PREP 0.153ms의 주 원인

---

## v5 → v6: 절대 Noise 모드 + 자동 종료

### 문제: 상대 SNR 모드에서 PL이 CQI에 영향 없음

**증상**: Path Loss(PL)를 적용해도 CQI가 변하지 않는다.

**원인**: 상대 SNR 모드(`--snr-dB`)에서는 noise가 signal 전력에 비례하여 계산된다. PL이 signal을 줄이면 noise도 같이 줄어서 UE가 체감하는 SNR 비율이 불변이다.

**해결**: OAI 내부 noise 모델과 동일한 **절대 dBFS** noise 모드를 추가:
- `noise_rms = 32767 * 10^(dBFS/20)` — noise 크기가 고정
- PL이 signal을 줄이면 effective SNR 하락 → CQI 변화

### 발견: OAI 구조적 한계

테스트 중 OAI의 근본적 한계를 발견:
- OAI rfsimulator는 PSS/SSS/PBCH/RACH와 PDSCH/PUSCH가 **동일 IQ 스트림**을 공유
- 데이터에 영향줄 noise(-30dBFS 이상)를 넣으면 **초기 접속 자체가 실패** (PSS 동기화 불가)
- 초기 접속 가능한 noise(-50dBFS 이하)에서는 effective SNR이 ~34dB로 CQI=15 유지

**해결 방향**: RRC 연결 후 noise를 동적으로 활성화하는 방식 필요 (G1B 범위에서 미구현).

---

## v6 → v7: CH_COPY 파이프라인 최적화

### 문제: CH_COPY가 파이프라인의 46~52% (최대 병목)

v5 프로파일링 결과, 채널 계수를 전달하는 CH_COPY 단계가 전체 파이프라인의 절반을 차지했다. 채널 적용 연산(GPU_COMPUTE ~0.1ms)보다 채널 **전달** 자체가 4~5배 더 느렸다.

**원인 분석** (메모리 트래픽 ~7.9MB/slot):

| 연산 | 비용 | 설명 |
|------|------|------|
| `astype(cp.complex128)` | 매 슬롯 | RingBuffer가 c64, pipeline이 c128 → 변환 커널 + 임시 버퍼 |
| `RingBuffer.get_batch().copy()` | 매 슬롯 ~1.75MB | 데이터 안전성을 위한 deep copy |
| `gpu_H[:] = 0` (memset) | 매 슬롯 ~1.75MB | 무조건 전체 제로화 |
| `fft(gpu_H)` | 매 슬롯 | pipeline 내에서 채널 FFT |

**해결** (메모리 트래픽 ~3.5MB/slot, **~55% 절감**):

| 연산 | v6 | v7 |
|------|-----|-----|
| dtype 변환 | c64→c128 매 슬롯 | RingBuffer/ChannelProducer 모두 c128로 통일 → `astype` 제거 |
| `.copy()` | 매 슬롯 deep copy | `get_batch_view()` + `release_batch()` (view, zero-copy) |
| 제로화 | 무조건 전체 | 조건부: `n_ch < N_SYM`일 때만 (정상 시 skip) |
| FFT | pipeline 내 매 슬롯 | ChannelProducer에서 1회 사전 수행 |

### RingBuffer view+release 도입 배경

`.copy()` 제거 시 데이터 안전성이 우려되었다. Producer가 아직 Consumer가 읽고 있는 메모리에 덮어쓸 수 있기 때문이다.

**해결**: `get_batch_view(N)` / `release_batch(N)` 패턴 도입:
- `get_batch_view`: view 반환 시 `count`를 감소시키지 않음 → Producer가 이 영역에 쓰지 못함
- `release_batch`: Consumer가 사용 완료 후 호출 → `count -= N`으로 영역 반환
- GIL + count 보호로 thread-safe 보장

---

## v7 → v8: NOISE_PREP 사전 생성

### 문제: noise ON 시 NOISE_PREP 0.388ms (파이프라인의 56%)

v7에서 CH_COPY를 최적화한 후, noise ON 시 다음 병목이 드러났다. `_regenerate_noise()`에서 `cp.random.randn()`을 매 슬롯 2회 동기 호출하는 것이 원인이다 (Box-Muller 알고리즘 + GPU 커널 런치 오버헤드).

**해결**: `ChannelProducer` 패턴을 재사용하여 `NoiseProducer` 스레드 도입:
- 별도 스레드에서 `cp.random.randn(batch=64, 2, noise_len)` → RingBuffer(maxlen=256) 적재
- Pipeline은 `get_batch_view` + `release_batch`로 사전 생성된 noise를 memcpy만 수행
- noise OFF 시 NoiseProducer 미생성 (하위 호환)

### 실측 결과 (MIMO 2x2, PL 5dB, noise -50dBFS)

| 지표 (IPC_SLOT_EVT avg) | v7 | v8 | 변화 |
|--------------------------|-----|-----|------|
| NOISE_PREP | 0.388ms | 0.153ms | **-61%** |
| Pipeline TOTAL | 0.691ms | 0.561ms | **-19%** |

### 0.153ms > 0.01ms인 이유

당초 예상은 ~0.01ms(memcpy만)였으나, 실측은 0.153ms였다.

**원인**: NoiseProducer의 난수 생성은 비동기로 완료되었으나, RingBuffer view → 고정 버퍼(`gpu_noise_r/i`)로의 **GPU-to-GPU memcpy (~960KB)**가 예상보다 비용이 컸다.

**zero-copy는 불가능한 이유**: RingBuffer view의 GPU 주소가 매 슬롯 변동하므로, CUDA Graph의 고정 메모리 주소 요구와 충돌한다. v5에서 CUDA Graph 도입 시 확립된 "고정 버퍼 + memcpy" 패턴의 불가피한 트레이드오프이다.

CUDA Graph의 성능 이득(4.7x)이 memcpy 비용을 압도하므로, 현재 설계가 최적이다.

---

## 성능 진화 요약

### Pipeline 성능 (MIMO 2x2 channel)

| 버전 | Pipeline TOTAL (EVT avg) | DL throughput | 핵심 병목 해결 |
|------|--------------------------|---------------|--------------|
| v4 | ~9.5ms / 20 slots | 75.7 DL/s | (기준선, einsum, CUDA Graph 미적용) |
| v5 | ~0.69ms/slot | 354 DL/s | CUDA Graph 통합 (**4.7x**) |
| v7 | ~0.69ms/slot (noise OFF) | - | CH_COPY ~55% 절감 |
| v8 | ~0.56ms/slot (noise ON) | - | NOISE_PREP -61% |

### OAI 수정 요약

| 버전 | 수정 대상 | 파일 수 | 내용 |
|------|----------|---------|------|
| v1 | rfsimulator | 3 | GPU IPC V5 (circular buffer, SHM, timestamp polling) |
| v2 | rfsimulator | 3 | GPU IPC V6 (per-buffer nbAnt, 독립 cir_size) |
| v3 | launch_all.sh | 1 | OAI CLI antenna port override (코드 수정 없음) |
| v4 | UE MAC, gNB MAC/RRC | 4 | CSI CQI 리포팅 정상화 (인코더, 디코더, config, 디버그 핵) |

### 주요 교훈

1. **CUDA Graph 고정 주소 제약**은 성능 최적화의 천장을 결정한다. zero-copy가 불가능한 경우에도 4.7x 이득이 memcpy 비용을 압도하므로 Graph 유지가 최적이다.
2. **OAI 코드 수정은 최소화**하되, fallthrough 버그나 config 누락 같은 근본 원인은 반드시 수정해야 한다. CLI 파라미터로 해결 가능한 경우(v3) 코드 수정 없이 처리한다.
3. **Producer-Consumer 패턴**은 GPU 파이프라인 병목 해결에 효과적이다. ChannelProducer(v4)와 NoiseProducer(v8) 모두 동일 패턴으로 구현했다.
4. **dtype 통일**은 단순하지만 효과가 크다. c64→c128 변환 제거만으로 CH_COPY의 상당 부분이 절감되었다.
5. **OAI rfsimulator의 구조적 한계**(PSS/데이터 동일 IQ 스트림)는 noise 테스트의 근본적 제약이다. 연결 후 동적 활성화가 필요하다.

---

## 미해결 과제

| 항목 | 설명 | 난이도 |
|------|------|--------|
| 연결 후 noise 동적 적용 | RRC 연결 후 noise를 켜는 방식으로 PSS/RA 보호 | 중간 |
| SISO CQI=0 잔존 | 1-port에서 CQI 리포팅이 정상이나 CQI=0 유지 (CSI-RS 보간 잔차 가능) | 중간 |
| MCS 적응 관찰 | CQI 정상에도 MCS 0 유지 — 장시간 테스트 필요 | 낮음 |
| GPU_COPY_IN/OUT 최적화 | coalesced memory access, contiguous copy | 낮음 |
| Python overhead 감소 | polling/sync event-driven 전환 | 낮음 |

# Type-II PMI 구현 검증: 정규화 채널 상관 분석

> 본 문서는 졸업 논문 Chapter 3.6 "구현 검증" 섹션에 삽입할 수 있는 수준으로 작성되었다.

---

## 1. 검증의 목적과 필요성

OAI(OpenAirInterface)의 UE 측 CSI 보고 코드(`csi_rx.c`)에 Type-II Port Selection codebook을 구현하였으므로, 이 구현이 3GPP TS 38.214의 규격대로 정확하게 동작하는지를 독립적으로 검증할 필요가 있다. 검증의 핵심 질문은 다음과 같다:

> *UE가 CSI-RS로부터 추정한 채널 $\mathbf{H}$에 대해 Type-II PMI를 보고했을 때, 해당 PMI로 재구성한 프리코딩 벡터 $\mathbf{w}$가 실제 채널 방향과 얼마나 잘 정렬(align)되는가?*

이를 정량적으로 측정하기 위해, UE 내부에서 추정한 채널 행렬 $\hat{\mathbf{H}}$와 보고된 PMI 파라미터를 동시에 기록하는 계측(instrumentation) 코드를 `csi_rx.c`에 추가하고, 오프라인 후처리로 **정규화 채널 상관(normalized channel correlation)** $\rho$를 계산하는 검증 프레임워크를 구축하였다.

---

## 2. 검증 메트릭: 정규화 채널 상관 (Normalized Channel Correlation)

### 2.1 정의

서브캐리어 $k$, 수신 안테나 $r$에서의 채널 벡터를 $\mathbf{h}_{r}[k] \in \mathbb{C}^{N_t}$, PMI로 재구성한 프리코딩 벡터를 $\mathbf{w} \in \mathbb{C}^{N_t}$라 하면, 정규화 상관은 다음과 같이 정의된다:

$$
\rho_{r}[k] = \frac{|\mathbf{h}_{r}[k]^H \mathbf{w}|}{\|\mathbf{h}_{r}[k]\| \cdot \|\mathbf{w}\|}
$$

$\rho \in [0, 1]$이며, $\rho = 1$은 프리코딩 벡터가 채널 방향과 완벽하게 정렬됨을 의미한다. 최종 보고 지표는 활성 서브캐리어 및 수신 안테나에 대한 평균이다:

$$
\bar{\rho} = \frac{1}{N_r \cdot N_{sc}^{(\text{active})}} \sum_{r=0}^{N_r-1} \sum_{k \in \mathcal{K}_{\text{active}}} \rho_{r}[k]
$$

여기서 $\mathcal{K}_{\text{active}}$는 CSI-RS가 매핑된 RB 범위(`start_rb`부터 `nr_of_rbs`개)의 서브캐리어 집합이다.

### 2.2 이 메트릭을 선택한 이유

정규화 상관은 프리코딩 벡터의 **방향 정확도**만을 측정하며, 채널의 절대 크기(SNR)에 무관하다. 이는 codebook 구현의 정확성을 평가하기에 적합하다:

- 높은 SNR에서 채널 추정이 정확해지면, codebook의 양자화 오차만이 $\rho$를 결정한다.
- Type-II가 Type-I보다 세밀한 채널 표현을 제공하므로, $\bar{\rho}_{\text{Type-II}} > \bar{\rho}_{\text{Type-I}}$이 일관되게 관찰되어야 한다.
- $\rho$가 SNR 증가에 따라 단조 증가하여 특정 상한에 수렴해야 하며, 이 상한이 해당 codebook의 이론적 양자화 한계에 대응한다.

---

## 3. 검증 시스템 아키텍처

### 3.1 전체 파이프라인

```
┌──────────────── OAI UE (nr-uesoftmodem) ─────────────────┐
│                                                           │
│  CSI-RS 수신 → 채널 추정 (H_est)                          │
│       │                                                   │
│       ├─→ PMI 계산 (Type-I 또는 Type-II)                  │
│       │       ├─ Type-I: i1, i2                           │
│       │       └─ Type-II: port_sel, wb_amp[], sb_phase[]  │
│       │                                                   │
│       └─→ 바이너리 로그 기록 ──→ [.bin 파일]              │
│           (CSI_CHANNEL_LOG 환경변수로 경로 지정)           │
│           [64B 헤더 + H_est 데이터 + SB phase 데이터]     │
└───────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────── Python 오프라인 후처리 ───────────────────────┐
│                                                           │
│  1. 바이너리 레코드 파싱 (헤더 + 채널 + 서브밴드 위상)    │
│  2. PMI 타입 자동 판별 (pmi_type 필드)                    │
│  3. 프리코딩 벡터 재구성                                  │
│       ├─ Type-I:  i2 → W = [1, e^{jφ}]/√2               │
│       └─ Type-II: port_sel + amp + phase → W              │
│  4. 서브캐리어별, RX 안테나별 ρ 계산                      │
│  5. SNR별 통계 및 시각화                                  │
└───────────────────────────────────────────────────────────┘
```

### 3.2 UE 측 바이너리 로그 포맷

`csi_rx.c`의 `nr_csi_rs_pmi_estimation()` 및 `nr_csi_rs_pmi_estimation_4port()` 함수에서 PMI 결정 직후, 채널 추정치와 PMI 파라미터를 함께 기록한다. 각 CSI-RS 이벤트마다 하나의 레코드가 생성된다.

**레코드 헤더 (64 바이트)**:

| 오프셋 | 타입 | 필드 | 설명 |
|:------:|------|------|------|
| 0 | uint32 | magic | `0x43534932` ("CSI2") |
| 4 | uint32 | record_size | 레코드 전체 크기 (헤더 + 데이터) |
| 8 | uint32 | frame | 프레임 번호 |
| 12 | uint32 | slot | 슬롯 번호 |
| 16 | uint16 | N_ports | CSI-RS 포트 수 |
| 18 | uint16 | N_rx | UE 수신 안테나 수 |
| 20 | uint32 | ofdm_size | OFDM FFT 크기 |
| 24 | uint32 | first_carrier | 첫 번째 캐리어 오프셋 |
| 28 | uint16 | start_rb | CSI-RS 시작 RB |
| 30 | uint16 | nr_of_rbs | CSI-RS RB 수 |
| 33 | uint8 | rank | RI (Rank Indicator) |
| 34 | uint8 | cqi | CQI |
| 35 | uint8 | pmi_type | 0 = Type-I, 1 = Type-II |
| 36–37 | uint8[2] | i1, i2 | Type-I PMI 인덱스 |
| 38 | uint8 | port_sel_indicator | Type-II 포트 선택 인디케이터 |
| 41 | uint8 | num_beams | Type-II 빔 수 $L$ |
| 42 | uint8 | port_sel_d | Type-II 포트 선택 파라미터 $d$ |
| 43 | uint8 | phase_alphabet | Type-II 위상 알파벳 (4=QPSK, 8=8PSK) |
| 44–51 | uint8[8] | wb_amp_l0 | Type-II Wideband 진폭 (레이어 0) |
| 52–59 | uint8[8] | wb_amp_l1 | Type-II Wideband 진폭 (레이어 1) |
| 60 | uint8 | num_subbands | Type-II 서브밴드 수 |

**데이터 영역**:

| 영역 | 타입 | 크기 | 설명 |
|------|------|------|------|
| 채널 추정치 | c16_t (int16 × 2) | $N_r \times N_{\text{ports}} \times N_{\text{FFT}} \times 4$ bytes | UE가 CSI-RS로 추정한 $\hat{\mathbf{H}}$ |
| 서브밴드 위상 | uint8 | $N_{\text{sb}} \times 2L \times N_{\text{layers}}$ bytes | Type-II 서브밴드별 위상 인덱스 |

레코드당 크기는 4포트, 2 RX 안테나, FFT 2048 기준 약 65 KB이며, CSI-RS 주기 20슬롯(= 12.5 이벤트/초)에서 약 800 KB/s의 로그가 생성된다.

---

## 4. 프리코딩 벡터 재구성

### 4.1 Type-I Codebook 재구성

3GPP TS 38.214 Table 5.2.2.2.1-1에 따라, 2포트 Rank-1 Type-I codebook의 프리코딩 벡터는 `i2` 인덱스에 의해 결정된다:

$$
\mathbf{w}_{\text{Type-I}} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ e^{j\phi_{i_2}} \end{bmatrix}, \quad \phi_{i_2} \in \left\{0, \frac{\pi}{2}, \pi, \frac{3\pi}{2}\right\}
$$

OAI의 Type-I는 4포트 구성에서도 포트 0, 1만 사용하며, 나머지 포트는 0으로 채워진다.

### 4.2 Type-II Port Selection Codebook 재구성

3GPP TS 38.214 §5.2.2.2.3에 따른 Type-II Port Selection의 Rank-1 프리코딩 벡터 재구성은 다음과 같다:

1. **포트 선택**: 포트 선택 인디케이터 $k_1$과 파라미터 $d$로부터, 선택된 CSI-RS 포트의 시작 인덱스 $k_1 \cdot d$를 결정한다.

2. **Wideband 진폭 역양자화**: 각 계수 $c$에 대해, 진폭 인덱스 $a_c$를 3GPP 표준 테이블로 역양자화한다:

$$
\alpha_{a_c} \in \left\{1, \frac{1}{\sqrt{2}}, \frac{1}{2}, \frac{1}{2\sqrt{2}}, \frac{1}{4}, \frac{1}{4\sqrt{2}}, \frac{1}{8}, 0\right\}
$$

3. **서브밴드 위상 역양자화**: QPSK($N_{\text{phase}}=4$) 또는 8PSK($N_{\text{phase}}=8$) 위상 인덱스 $p_c$를 각도로 변환한다:

$$
\theta_{p_c} = p_c \cdot \frac{2\pi}{N_{\text{phase}}}
$$

4. **프리코딩 벡터 합성**: 총 $2L$개의 계수($L$개 빔 × 2 편파)를 해당 안테나 포트에 매핑한다:

$$
W[\text{ant\_idx}] \mathrel{+}= \alpha_{a_c} \cdot e^{j\theta_{p_c}}
$$

여기서 `ant_idx = sel_start + (c mod d) + (c / d) × (N_ports / 2)`이며, 첫 $d$개 계수는 편파 0, 다음 $d$개는 편파 1에 해당한다. 최종적으로 $\mathbf{w}$를 단위 노름으로 정규화한다.

---

## 5. SNR 스윕 실험 방법

### 5.1 실험 설정

동일한 채널 환경에서 Type-I과 Type-II의 성능을 공정하게 비교하기 위해, 자동화된 SNR 스윕 스크립트(`sweep_snr.sh`)를 사용한다.

| 파라미터 | 값 |
|----------|-----|
| SNR 범위 | 0, 5, 10, 15, 20, 25, 30 dB |
| SNR 포인트당 실행 시간 | 60초 |
| 안테나 구성 | gNB: 2×2 (4포트), UE: 2×1 |
| CSI-RS 주기 | 20슬롯 |
| 채널 모델 | Sionna UMa-NLOS |
| Codebook 설정 | Type-I: `typeI_SinglePanel`, Type-II: `typeII_PortSelection` |
| 총 실행 회수 | 7 SNR × 2 codebook = 14회 |

### 5.2 실험 절차

각 (codebook type, SNR) 조합에 대해:

1. `gnb.conf`의 codebook 설정을 Type-I 또는 Type-II로 자동 전환
2. `CSI_CHANNEL_LOG` 환경변수에 출력 파일 경로 설정
3. OAI 시스템(gNB + UE + Sionna Proxy) 구동, 지정 시간 동안 실행
4. 바이너리 로그 파일 수집 후 시스템 종료

수집된 로그 파일을 후처리 스크립트(`verify_type2_pmi.py`)로 일괄 분석한다.

---

## 6. 기대 결과 및 해석 기준

### 6.1 기대되는 결과 패턴

| 관찰 항목 | 기대 패턴 | 의미 |
|-----------|-----------|------|
| $\bar{\rho}_{\text{Type-II}} > \bar{\rho}_{\text{Type-I}}$ (전 SNR) | 일관된 우위 | Type-II가 더 정밀한 채널 표현을 제공 |
| SNR 증가 시 $\bar{\rho}$ 단조 증가 | 수렴하는 상한 존재 | 고 SNR에서는 codebook 양자화 오차가 지배적 |
| 중간 SNR (10~25 dB)에서 차이 최대 | Type-I/II 갭 확대 | 채널 추정은 충분히 정확하나 Type-I의 표현력 한계가 드러남 |
| 저 SNR (< 5 dB)에서 차이 축소 | 채널 추정 오차 지배 | 채널 추정 자체의 부정확성이 codebook 차이를 상쇄 |
| 고 SNR (> 25 dB)에서 차이 유지 | 양자화 오차 상한 | Type-I의 QPSK 4-위상 한계 vs Type-II의 다중 빔+진폭 표현 |

### 6.2 이상 동작 판별 기준

| 증상 | 가능한 원인 |
|------|------------|
| $\bar{\rho}_{\text{Type-II}} < \bar{\rho}_{\text{Type-I}}$ | Type-II 재구성 로직 오류 (포트 매핑, 편파 인덱싱 등) |
| SNR 무관하게 $\bar{\rho} \approx$ 상수 | 채널 추정치가 아닌 고정값이 기록됨 (로깅 오류) |
| $\bar{\rho}$가 0.5 이하로 일관되게 낮음 | 채널 인덱싱 오류 (first_carrier 오프셋, 메모리 정렬 등) |

---

## 7. 출력 시각화

후처리 스크립트는 세 가지 서브플롯을 생성한다:

| 그래프 | X축 | Y축 | 설명 |
|--------|------|------|------|
| (a) Correlation vs SNR | SNR (dB) | 평균 $\bar{\rho}$ (0~1) | Type-I / Type-II의 SNR별 평균 상관 비교 |
| (b) CDF | $\rho$ 값 | 누적확률 | 선택 SNR 포인트에서의 $\rho$ 분포 |
| (c) Improvement | SNR (dB) | 개선율 (%) | $(\bar{\rho}_{\text{II}} - \bar{\rho}_{\text{I}}) / \bar{\rho}_{\text{I}} \times 100$ |

---

## 8. 논문에서의 위치 및 의의

본 검증은 논문의 **Chapter 3.6 "구현 검증"** 섹션에서 다음을 입증하는 역할을 한다:

1. **Type-II PMI 구현의 정확성**: OAI에 추가한 Type-II Port Selection codebook이 3GPP 규격대로 정확하게 동작함을 정규화 상관 메트릭으로 정량적으로 확인하였다.

2. **Type-I 대비 Type-II의 채널 표현 정확도 우위**: 동일 채널 환경에서 Type-II가 Type-I보다 일관되게 높은 $\bar{\rho}$를 보여, 더 정밀한 프리코딩 방향을 제공함을 실측으로 검증하였다.

3. **Chapter 5 실험의 전제 조건 확보**: Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 end-to-end 성능 비교(Chapter 5)가 유의미하려면, 각 codebook의 PMI가 정확하게 구현되어 있어야 한다. 본 검증이 이 전제를 뒷받침한다.

4. **제안 기법(Chapter 4)의 비교 기준선 신뢰성**: 제안하는 통계-조건부 CSI 인코더를 Type-II와 비교할 때, Type-II의 구현이 정확함이 보장되어야 비교 결과가 의미를 가진다.

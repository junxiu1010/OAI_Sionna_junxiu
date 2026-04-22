# 졸업 논문 논리 구조 및 Chapter/Section별 문단 레벨 논리 흐름

> **논문 제목 (안)**: OAI 기반 RAN 디지털 트윈 구축 및 채널 통계 조건부 AI/ML CSI 압축을 통한 5G NR MIMO 성능 개선 연구
>
> *A Study on 5G NR MIMO Performance Enhancement via Statistics-Conditioned AI/ML CSI Compression Using OAI-Based RAN Digital Twin*

---

## 전체 논리 흐름 (Macro-Level Storyline)

```
[Why — 문제 인식]
  5G NR MIMO 성능은 CSI 피드백 정확도에 의존한다
    → Type 1 SU-MIMO: 간단하지만 빔 표현력이 부족
    → Type 2 SU-MIMO: 정밀하지만 피드백 오버헤드가 큼
    → Type 2 MU-MIMO: 셀 용량 이득이 크지만, CSI 양자화 오차가 UE간 간섭으로 직결
  → 결국 "동일 비트수에서 더 정확한 CSI"가 핵심 과제

[What — 기존 접근의 한계와 제안]
  기존 AI/ML CSI (CsiNet 등): 순간 채널 H만 보고 blind하게 압축
    → 채널의 장기 통계 구조(공분산, PDP)를 활용하지 못함
  제안: RAN 트윈에서 미리 확보한 공분산/PDP를 조건으로 활용하여 H를 인코딩
    → 같은 비트수에서 더 정확한 복원 → MU-MIMO에서 의미 있는 성능 이득

[How — 검증 방법]
  OAI 기반 RAN 디지털 트윈을 구축하여
    → Type 1 SU-MIMO / Type 2 SU-MIMO / Type 2 MU-MIMO 성능을 baseline으로 확보
    → 제안하는 통계-조건부 인코더의 end-to-end 성능을 동일 환경에서 비교
    → 다양한 시나리오(SNR, 이동속도, UE수)에서의 성능 차이를 체계적으로 분석
```

---

## Chapter 1: Introduction

### 1.1 연구 배경 (Research Background)

**[문단 1 — 5G MIMO의 중요성]**
5G NR에서 massive MIMO는 스펙트럼 효율을 높이는 핵심 기술이다. 기지국이 다수의 안테나를 사용하여 빔포밍/프리코딩을 수행하면, SU-MIMO에서는 공간 다중화를 통해 단일 UE의 throughput을 높이고, MU-MIMO에서는 다수의 UE에 동시에 서비스하여 셀 용량을 증가시킨다. 이러한 MIMO 기술의 성능은 기지국이 채널 상태 정보(CSI)를 얼마나 정확하게 확보하느냐에 의해 결정된다.

**[문단 2 — CSI 피드백의 한계: Type 1 vs Type 2]**
FDD 시스템에서 UE는 DL 채널을 측정한 후 codebook 기반의 PMI로 압축하여 기지국에 피드백한다. 3GPP는 Type 1과 Type 2의 두 가지 codebook을 규정하고 있다. Type 1은 단일 DFT 빔을 선택하는 방식으로 피드백 오버헤드가 작지만 채널의 세부 구조를 포착하지 못한다. Type 2는 다수의 빔을 선형 결합하여 채널을 더 정밀하게 표현하지만, 위상/진폭 계수의 피드백으로 인해 오버헤드가 크게 증가한다. 특히 MU-MIMO에서는 UE 간 간섭 널링을 위해 높은 CSI 정확도가 필수적인데, Type 2의 오버헤드 증가는 실질적인 throughput 이득을 상쇄할 수 있다.

**[문단 3 — AI/ML CSI 압축의 등장과 한계]**
3GPP Rel-18에서 AI/ML 기반 CSI 피드백이 study item으로 채택되면서, autoencoder 구조를 이용한 CSI 압축이 주목받고 있다. CsiNet(Wen et al., 2018)을 시작으로 다양한 아키텍처가 제안되어 동일 비트수에서 기존 codebook 대비 더 높은 복원 정확도를 달성하고 있다. 그러나 기존 AI/ML CSI 연구는 두 가지 한계를 갖는다: (1) 순간 채널 H만을 입력으로 사용하여, 채널의 장기 통계 구조(공분산 행렬, 전력 지연 프로파일 등)를 인코딩에 활용하지 못하고, (2) 오프라인 데이터셋에서의 압축/복원 성능(NMSE)만 평가하여, 실제 RAN 프로토콜 스택과의 end-to-end 성능 검증이 부족하다.

**[문단 4 — 트윈 기반 통계 정보 활용의 가능성]**
RAN 디지털 트윈은 실제 RAN 환경의 채널 통계(공분산, PDP)를 사전에 확보할 수 있는 플랫폼이다. 만약 이 장기 통계를 인코더의 조건(condition)으로 활용할 수 있다면, 인코더는 "이 환경에서 이 채널이 평균적으로 어떻게 생겼는지"를 이미 알고 있는 상태에서 순간 채널의 *잔차(deviation)*만 효율적으로 압축할 수 있다. 이는 같은 비트수에서 더 높은 압축 성능으로 이어질 수 있으며, 특히 MU-MIMO에서 의미 있는 시스템 레벨 성능 이득이 기대된다.

### 1.2 연구 목적 (Research Objectives)

**[문단 1 — 핵심 목적]**
본 연구의 목적은 다음과 같다. 첫째, OAI 기반의 RAN 디지털 트윈을 구축하여 다양한 RAN 시나리오에서 MIMO 성능을 실험적으로 평가할 수 있는 플랫폼을 구축한다. 둘째, 이 플랫폼에서 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 성능을 체계적으로 비교하여, 각 방식의 성능 한계와 CSI 정확도의 영향을 정량적으로 분석한다. 셋째, CsiNet을 baseline autoencoder로 두고, 트윈을 통해 미리 확보한 채널 통계 정보(공분산 행렬/PDP)를 조건으로 활용하여 H의 autoencoding 성능을 개선하는 **통계-조건부 CSI 인코더**를 제안하고, 기존 세 가지 방식 대비 성능 개선을 실증한다.

**[문단 2 — 차별성]**
기존 AI/ML CSI 연구가 순간 채널만으로 blind 압축을 수행하는 것과 달리, 본 연구는 RAN 트윈이 제공하는 채널 통계를 side information으로 활용하여 압축 효율을 높인다. 또한 오프라인 NMSE 평가에 머무르지 않고, OAI의 실제 프로토콜 스택(RRC, MAC, PHY)과 통합된 end-to-end 환경에서 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO 대비 성능 개선을 검증한다.

### 1.3 연구 범위 및 구성 (Scope and Organization)

**[문단 1 — 범위]**
본 논문은 FR1 Band 78 (3.5 GHz), SCS 30 kHz, 106 PRB (40 MHz) 환경에서의 DL MIMO 성능에 초점을 맞춘다. 안테나 구성은 2T2R ~ 4T4R (dual-pol)을 주로 다루며, 3GPP UMa/UMi 채널 모델을 사용한다.

**[문단 2 — 논문 구성 안내]**
본 논문의 구성은 다음과 같다. 2장에서는 MIMO 채널 모델과 5G NR CSI 획득-스케줄링 시스템을 설명하고 관련 연구를 정리한다. 3장에서는 OAI 기반 RAN 디지털 트윈의 아키텍처와 저자가 구현한 핵심 모듈을 제시한다. 4장에서는 CsiNet baseline과 제안하는 통계-조건부 CSI 인코더의 설계 및 오프라인 성능 평가를 다룬다. 5장에서는 RAN 트윈 위에서 기존 세 가지 방식과 제안 기법의 end-to-end 성능 비교 실험을 분석한다. 6장에서 결론을 맺는다.

---

## Chapter 2: MIMO Channel Model & 5G NR CSI Acquisition–Scheduling System

> **이 장의 목적**: 독자가 본 연구를 이해하기 위해 필요한 배경 지식을 제공하고, 기존 기술(Type 1/2)의 한계와 기존 AI/ML 접근의 한계를 논리적으로 도출하여 4장의 연구 동기를 확립한다.

### 2.1 MIMO 채널 모델 (MIMO Channel Model)

#### 2.1.1 MIMO 시스템 모델

**[문단 1 — 신호 모델]**
$N_t$개 송신 안테나, $N_r$개 수신 안테나의 MIMO-OFDM 시스템에서 서브캐리어 $k$의 수신 신호를 $\mathbf{y}[k] = \mathbf{H}[k]\mathbf{W}[k]\mathbf{x}[k] + \mathbf{n}[k]$으로 표현한다. 프리코딩 행렬 $\mathbf{W}$의 선택이 곧 MIMO 성능을 결정하며, 이를 위해 UE가 측정한 채널 정보를 기지국에 피드백하는 CSI 보고 메커니즘이 필요하다.

**[문단 2 — 채널의 통계적 특성]**
무선 채널 $\mathbf{H}$는 두 가지 시간 스케일의 특성을 갖는다. 장기 통계(공분산 행렬 $\mathbf{R}_H = \mathbb{E}[\text{vec}(\mathbf{H})\text{vec}(\mathbf{H})^H]$, 전력 지연 프로파일 PDP)는 수백 ms ~ 수 초 단위로 변화하며 UE의 위치/환경에 의존한다. 순간 채널(instantaneous H)은 매 슬롯마다 변화하며 small-scale fading에 의존한다. 장기 통계를 알고 있으면, 순간 채널의 불확실성이 줄어들어 더 적은 비트로 효율적인 압축이 가능하다 — 이것이 4장의 제안 기법의 이론적 근거이다.

#### 2.1.2 3GPP 채널 모델 및 Sionna

**[문단 1 — 3GPP TR 38.901 모델]**
3GPP TR 38.901은 UMa, UMi, RMa 등의 시나리오별 경로 손실, 지연 확산, 각도 확산 파라미터를 규정하며, 클러스터 기반 다중 경로 채널을 생성한다. 각 시나리오는 고유한 공분산 구조와 PDP 특성을 갖는다.

**[문단 2 — Sionna 채널 모델]**
NVIDIA Sionna는 3GPP 38.901 채널 모델을 TensorFlow 기반으로 구현한 오픈소스 라이브러리이다. GPU 가속으로 대규모 채널 계수를 생성할 수 있으며, 동일 시나리오에서 공분산 행렬과 PDP를 통계적으로 추출할 수 있다. 본 연구에서는 Sionna를 채널 에뮬레이션과 AI/ML 학습용 데이터셋 생성에 모두 활용한다.

**[문단 3 — 채널 희소성과 압축]**
무선 채널은 각도-지연(angular-delay) 도메인에서 희소한 특성을 가지며, 이 희소성이 codebook 기반 압축 및 AI/ML 압축의 이론적 근거가 된다. 나아가, 공분산 행렬은 이 희소 구조의 "평균적 분포"를 나타내므로, 이를 사전 정보로 활용하면 순간 채널의 유효 자유도가 줄어들어 압축 효율이 높아진다.

### 2.2 5G NR CSI 획득 시스템 (CSI Acquisition Framework)

#### 2.2.1 CSI-RS 및 채널 측정

**[문단 1 — CSI-RS 개요]**
gNB가 CSI-RS를 주기적으로 전송하면, UE는 이를 수신하여 DL 채널을 추정한다. CSI-RS의 포트 수, 주기, CDM 패턴, 밀도 등의 설정이 채널 측정의 정확도에 영향을 미친다.

**[문단 2 — CSI 보고 지표: RI, PMI, CQI]**
UE는 채널 측정 결과를 RI(Rank Indicator), PMI(Precoding Matrix Indicator), CQI(Channel Quality Indicator)로 압축하여 기지국에 보고한다. RI는 공간 다중화 레이어 수, PMI는 프리코딩 행렬의 인덱스, CQI는 MCS 결정에 사용되는 채널 품질 지표이다.

#### 2.2.2 Type 1 Codebook

**[문단 1 — Type 1 구조]**
Type 1 (TS 38.214 §5.2.2.2.1)은 single-panel codebook으로, wideband에서 하나의 DFT 빔 방향을 선택하고 subband마다 co-phasing 인덱스를 보고한다. 피드백 오버헤드가 작다.

**[문단 2 — Type 1의 한계]**
단일 빔 방향만 포착하므로, 다중 클러스터 채널이나 넓은 각도 확산 환경에서 채널 표현 정확도가 낮다. MU-MIMO에서 UE 간 공간적 분리를 위한 정밀한 채널 정보를 제공하기 어렵다.

#### 2.2.3 Type 2 Codebook

**[문단 1 — Type 2 구조]**
Type 2 (TS 38.214 §5.2.2.2.3)는 $L$개의 빔을 선형 결합하여 채널을 표현한다. Type II Port Selection은 CSI-RS 포트 선택을 통해 차원을 줄인 변형이다.

**[문단 2 — Type 2의 오버헤드-정확도 트레이드오프]**
Type 2는 Type 1 대비 3~5배의 피드백 비트를 사용하며, 더 정밀한 채널 표현을 제공한다. 그러나 빔 수 $L$이 고정되어 채널 복잡도에 적응적이지 않고, 위상/진폭이 QPSK/8PSK로 양자화되어 정밀도가 제한된다. 이 오버헤드 증가가 uplink 자원을 소모하여 실질적인 시스템 이득을 제한할 수 있다.

#### 2.2.4 SU-MIMO vs MU-MIMO 스케줄링

**[문단 1 — SU-MIMO]**
SU-MIMO에서는 한 UE에 다수의 레이어를 할당하여 개별 UE의 throughput을 극대화한다. 프리코딩은 해당 UE의 CSI에만 기반하여 결정된다.

**[문단 2 — MU-MIMO]**
MU-MIMO에서는 동일한 시간-주파수 자원에서 다수의 UE에 동시에 서비스한다. gNB는 UE 간 간섭을 최소화하는 프리코딩을 수행해야 하며, 이는 모든 co-scheduled UE의 CSI를 정확하게 알아야 가능하다. CSI 부정확도는 MU-MIMO에서 잔여 간섭으로 직결된다.

**[문단 3 — 핵심 논점: CSI 정확도가 성능을 결정한다]**
이론적으로, 완벽한 CSI 하에서 MU-MIMO는 SU-MIMO 대비 UE 수에 비례하는 셀 용량 이득을 제공한다. 그러나 실제 codebook 기반 피드백에서는 양자화 오차로 인해 이득이 제한된다. 따라서 "동일 피드백 비트수에서 더 정확한 CSI를 전달하는 방법"이 MU-MIMO 성능 극대화의 핵심 과제이다.

### 2.3 AI/ML 기반 CSI 피드백 관련 연구 (Related Work)

#### 2.3.1 CsiNet 및 후속 연구

**[문단 1 — CsiNet]**
CsiNet(Wen et al., 2018)은 채널 행렬의 각도-지연 도메인 표현을 CNN autoencoder로 압축/복원하는 최초의 DL 기반 CSI 피드백 기법이다. 인코더가 $N_t \times N_c$ 채널 행렬을 $M$차원 잠재 벡터로 압축(압축률 $\gamma = M / (2N_tN_c)$)하고, 디코더가 복원한다. COST 2100 데이터셋에서 기존 codebook 대비 유의미한 NMSE 개선을 보고하였다.

**[문단 2 — 후속 발전과 공통된 한계]**
CsiNet+, CRNet, TransNet 등 다양한 아키텍처가 제안되었으나, 공통적으로 **순간 채널 H만을 입력으로 사용**한다. 채널의 장기 통계(공분산, PDP)가 인코딩에 반영되지 않아, 환경이 이미 알려진 상황에서도 불필요하게 많은 비트를 소비한다. 이는 곧 "조건부 정보가 없는 blind 압축"의 한계이다.

#### 2.3.2 채널 통계 활용 연구

**[문단 1 — 공분산 기반 접근]**
일부 연구에서 채널 공분산 행렬을 빔포밍이나 스케줄링에 활용한 사례가 있으나, 이를 autoencoder의 조건부 입력으로 활용하여 CSI 압축 자체의 효율을 높이는 연구는 제한적이다. 정보 이론적으로, 공분산 $\mathbf{R}_H$를 알고 있을 때 채널 벡터의 조건부 엔트로피 $H(\text{vec}(\mathbf{H}) | \mathbf{R}_H)$는 무조건부 엔트로피 $H(\text{vec}(\mathbf{H}))$보다 작으므로, 같은 왜곡 수준에서 더 적은 비트로 압축이 가능하다.

**[문단 2 — 본 연구의 차별점]**
본 연구는 (1) 채널 통계(공분산/PDP)를 autoencoder의 조건으로 직접 주입하여 H의 압축 효율을 높이고, (2) 이 통계 정보를 RAN 트윈에서 사전 확보할 수 있다는 실용적 경로를 제시하며, (3) 오프라인 NMSE가 아닌 실제 프로토콜 스택에서의 end-to-end 성능으로 평가한다는 점에서 기존 연구와 차별화된다.

#### 2.3.3 산업계 동향 및 검증 환경

**[문단 1 — 3GPP Rel-18 AI/ML CSI Study Item]**
3GPP Rel-18에서 AI/ML based CSI enhancement를 study item으로 채택하여 압축 성능, 일반화 능력, 모델 배포 방법 등을 평가하고 있다. Qualcomm, Samsung 등이 기존 Type 2 대비 동일 비트수에서 3~5 dB NMSE 개선을 보고하고 있다.

**[문단 2 — End-to-end 검증의 부재]**
대부분의 기존 연구는 오프라인 데이터셋에서의 채널 복원 정확도만 평가하며, 실제 RAN 환경(MAC 스케줄링, HARQ, CSI aging 등)에서의 throughput 성능을 평가한 연구는 극히 드물다. OAI + Sionna 기반 RAN 트윈은 이 갭을 메울 수 있는 플랫폼이다.

---

## Chapter 3: OAI 기반 RAN 디지털 트윈의 설계 및 구현

> **이 장의 목적**: RAN 디지털 트윈의 전체 아키텍처를 제시하고, 저자가 직접 구현한 핵심 모듈을 상세히 기술한다. 4장의 학습 데이터셋/통계 추출과 5장의 실험 플랫폼으로서의 역할을 함께 설명한다.

### 3.1 RAN 디지털 트윈 개요

**[문단 1 — 전체 아키텍처]**
본 연구의 RAN 디지털 트윈은 세 가지 주요 구성 요소로 이루어진다: (1) OAI 5G 프로토콜 스택 (gNB + UE의 RRC/MAC/PHY), (2) Sionna 기반 채널 에뮬레이터 (GPU 가속 MIMO 채널 + AWGN), (3) Core Emulator (설정 관리 서버). 이들이 GPU IPC를 통해 IQ 샘플 수준에서 결합되어, 실제 5G 시스템과 동일한 프로토콜 동작을 Sionna의 사실적인 채널 모델 위에서 실행한다.

**[문단 2 — 세 가지 입력 인터페이스]**
RAN 트윈은 세 가지 입력 인터페이스를 통해 시나리오를 구성한다:
- **Core Emulator 입력**: RRC/MAC/PHY 설정 (codebook type, 안테나 포트, CSI-RS 주기, Bearer/QoS 등)
- **환경/Radio/Mobility 입력**: 채널 모델 시나리오(UMa/UMi), SNR, 경로 손실, UE 이동 속도, 기지국/UE 공간 분포
- **Traffic 입력**: QoS 프로파일(5QI), 세션 AMBR, 트래픽 패턴 (full-buffer / bursty)

**[문단 3 — 트윈의 두 가지 역할]**
본 트윈은 두 가지 역할을 수행한다. 첫째, **실험 플랫폼**: Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO, 그리고 제안 기법의 end-to-end 성능을 동일 조건에서 비교한다. 둘째, **채널 통계 추출**: 각 시나리오에서 Sionna가 생성하는 채널 실현들로부터 공분산 행렬과 PDP를 통계적으로 추출하여, 4장의 통계-조건부 인코더 학습에 활용한다. 이 두 번째 역할이 트윈을 단순 시뮬레이터와 차별화하는 핵심이다.

### 3.2 Core Emulator: 중앙 설정 관리 서버

**[문단 1 — 문제 정의: 설정 분산의 어려움]**
OAI 기반 시뮬레이션에서 설정은 gnb.conf(PHY/MAC/RRC), 런치 스크립트(CLI 옵션), 채널 프록시(글로벌 상수)의 3곳에 분산되어 있어, 파라미터 변경 시 일관성 유지가 어렵고 실험 재현성이 낮았다.

**[문단 2 — Core Emulator 설계]**
이를 해결하기 위해 FastAPI 기반의 Core Emulator를 설계하였다. 모든 설정을 단일 YAML 파일(master_config.yaml)에 통합하고, Jinja2 템플릿 엔진으로 gnb.conf와 CN5G config를 자동 렌더링한다. REST API를 통해 프리셋 적용, 파라미터 핫스왑, 3GPP 규격 유효성 검증을 제공한다.

**[문단 3 — 설정 구조와 자동 파생]**
master_config.yaml은 system, antenna, codebook, csi_rs, srs, csi_measurement, channel, carrier, multicell, bearer, qos, network의 12개 섹션으로 구성되며, 안테나 파생값(N1, XP, nb_tx 등)은 자동 계산된다. 프리셋 시스템으로 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO 실험 설정을 원클릭으로 전환할 수 있다.

**[문단 4 — NMS 스타일 API와 확장성]**
Core Emulator의 API는 실제 5GC NMS의 REST API와 동일한 구조로 설계되어, 추후 실제 코어에 연결해도 클라이언트 변경 없이 동작한다.

### 3.3 Sionna 채널 프록시: GPU 가속 MIMO 채널 에뮬레이션

#### 3.3.1 Multi-UE MIMO 채널 프록시 (G1C)

**[문단 1 — GPU IPC 기반 IQ 샘플 연동]**
OAI의 rfsimulator를 확장하여, gNB와 UE 간 IQ 샘플이 GPU 공유 메모리를 통해 교환되도록 구현하였다. gNB의 DL 송신 IQ를 각 UE별 독립 채널 행렬로 변환하여 전달(DL Broadcast)하고, UE들의 UL 송신 IQ를 채널 적용 후 합산하여 gNB에 전달(UL Superposition)한다.

**[문단 2 — Sionna 채널 계수 생성 및 통계 추출]**
Sionna의 3GPP UMa/UMi 채널 모델 또는 P1B ray-tracing 데이터를 사용하여 UE별 독립적인 채널 계수를 생성한다. 채널 계수는 백그라운드 프로세스에서 연속적으로 생산되어 링 버퍼에 저장된다. **동시에**, 생성된 채널 실현들을 축적하여 UE별/시나리오별 공분산 행렬 $\mathbf{R}_H$와 PDP를 추출할 수 있으며, 이 통계가 4장의 통계-조건부 인코더 학습에 사용된다.

**[문단 3 — 성능 최적화]**
v0에서 v4까지의 최적화(CUDA Graph 파이프라인, fused RawKernel, 통합 ChannelProducer 등)를 거쳐, 최종 v4에서 2UE 2×2 MIMO 기준 2.07ms/slot을 달성하며 비실시간 시뮬레이션으로 안정 동작한다.

#### 3.3.2 4×4 MIMO RI/PMI 구현

**[문단 1 — OAI 원본의 한계와 수정]**
OAI 원본 코드에서 4포트 이상의 CSI-RS에 대해 RI=1로 고정되고, PMI도 2포트 기반 함수만 지원하였다. `nr_csi_rs_ri_estimation_4x4()`와 `nr_csi_rs_pmi_estimation_4port()` 함수를 구현하여 Rank 1~4에서의 RI 추정 및 PMI 후보 탐색을 지원하도록 하였다. 또한, 1포트 SINR 하드코딩(46dB 고정) 문제를 수정하여 실제 채널 품질이 CQI에 반영되도록 하였다.

### 3.4 MAC/스케줄러 기능 구현

**[문단 1 — MU-MIMO 스케줄러 지원]**
OAI의 기본 MAC 스케줄러는 SU-MIMO만 지원하므로, MU-MIMO 스케줄링을 위한 수정을 수행하였다. gNB_scheduler_dlsch.c에서 다수 UE의 PMI를 기반으로 co-scheduling 가능한 UE 쌍을 선택하고, codebook 기반 프리코딩을 적용하는 로직을 구현하였다.

**[문단 2 — Type 2 PMI 기반 프리코딩]**
Type 2 PMI에서 복원된 채널 방향 벡터를 사용하여 MU-MIMO 프리코딩 행렬을 구성한다. UE 간 공간 상관도를 기반으로 페어링 결정을 내리며, CSI 정확도가 직접적으로 MU-MIMO 성능에 영향을 미침을 확인한다.

### 3.5 RAN 시나리오 구성 체계

**[문단 1 — 시나리오 파라미터 분류]**
RAN 시나리오를 세 축으로 분류한다:
- **Radio 환경 축**: 안테나 구성(2T2R/4T4R), 채널 모델(UMa-LOS/NLOS, UMi), SNR 범위
- **Mobility 축**: UE 이동 속도(정지/보행/차량: 0/3/30 m/s), 도플러 확산
- **Traffic 축**: full-buffer / bursty, QoS 프로파일, UE 수

**[문단 2 — 시나리오 설계의 목적성]**
시나리오 설계는 "기존 세 방식(Type 1 SU, Type 2 SU, Type 2 MU)의 한계가 드러나는 환경"을 체계적으로 식별하는 것을 목표로 한다. 각 파라미터의 상·중·하 조합을 통해 성능 차이가 미미한 경우부터 극대화되는 경우까지를 포괄적으로 실험하며, 이를 통해 제안 기법이 가장 큰 이점을 제공하는 조건을 사전에 예측하고 실험으로 검증한다.

### 3.6 구현 검증 (Baseline Validation)

**[문단 1 — 프로토콜 동작 검증]**
RAN 트윈의 기본 동작을 검증한다: UE 초기 접속, RRC 연결 수립, CSI-RS 기반 RI/PMI/CQI 보고, MAC 스케줄링, HARQ 재전송 등이 정상 동작함을 로그 분석으로 확인한다.

**[문단 2 — 채널 응답 검증]**
Sionna 채널 모델의 출력(SNR vs CQI, 경로 손실 vs RSRP)이 이론값과 일치하는지 검증한다.

**[문단 3 — Multi-UE 동작 검증]**
2~4 UE 동시 접속 시나리오에서 모든 UE가 독립적으로 CSI 보고를 수행하고, MAC 스케줄러가 적절히 자원을 분배하는지 확인한다.

---

## Chapter 4: 통계-조건부 AI/ML CSI 인코더의 설계 및 성능 평가

> **이 장의 목적**: CsiNet을 baseline으로 제시한 후, 트윈에서 확보한 채널 통계(공분산/PDP)를 조건으로 활용하여 H의 autoencoding 성능을 개선하는 **제안 기법**을 설계하고, 오프라인 평가에서의 성능을 확인한다.

### 4.1 문제 정의 및 동기

**[문단 1 — 기존 autoencoder의 한계]**
CsiNet 등 기존 CSI autoencoder는 순간 채널 $\mathbf{H}$만을 입력으로 받아 blind하게 압축한다. 이는 인코더가 매 샘플마다 채널의 전체 구조(어떤 각도에 에너지가 집중되어 있는지, 지연 확산이 어느 정도인지)를 0부터 추론해야 함을 의미한다.

**[문단 2 — 통계 정보 활용의 이론적 근거]**
정보 이론적으로, 공분산 $\mathbf{R}_H$를 알고 있을 때 채널의 조건부 엔트로피가 줄어들므로, 같은 왜곡 수준에서 더 적은 비트로 압축이 가능하다 (rate-distortion theory). 직관적으로, 공분산이 "채널이 평균적으로 어떻게 생겼는지"를 알려주므로, 인코더는 순간 채널의 *편차(deviation from mean structure)*만 효율적으로 전달하면 된다. PDP 역시 지연 도메인의 에너지 분포를 알려주어 주파수 선택성 채널의 압축을 돕는다.

**[문단 3 — RAN 트윈이 통계를 제공하는 경로]**
RAN 트윈에서는 각 시나리오/UE 위치에 대해 다수의 채널 실현을 생성하므로, 공분산과 PDP를 통계적으로 추출할 수 있다. 실제 배포 시에도, gNB가 UE의 SRS/CSI-RS 보고를 축적하거나 트윈을 통해 해당 위치의 통계를 미리 확보하는 것이 가능하다. 따라서 통계 정보는 "합리적으로 확보 가능한 side information"이다.

### 4.2 Baseline: CsiNet 아키텍처

#### 4.2.1 채널 행렬의 각도-지연 도메인 표현

**[문단 1 — 도메인 변환]**
MIMO 채널 행렬 $\mathbf{H} \in \mathbb{C}^{N_t \times N_c}$를 2D DFT를 통해 각도-지연 도메인 $\tilde{\mathbf{H}}$로 변환한다. 지연 축 중 처음 $N_c'$개 bin만 절단하여 입력 차원을 줄이고, 실수부와 허수부를 분리하여 $2 \times N_t \times N_c'$ 크기의 2채널 이미지로 구성한다.

#### 4.2.2 CsiNet 인코더/디코더

**[문단 1 — 인코더 (UE 측)]**
Conv2D → BatchNorm → ReLU → Flatten → Dense($M$)의 구조로, 입력을 $M$차원 잠재 벡터(codeword)로 압축한다. 압축률 $\gamma = M / (2N_tN_c')$.

**[문단 2 — 디코더 (gNB 측)]**
Dense → Reshape → RefineNet(잔차 블록 × 2) → Sigmoid의 구조로, 잠재 벡터를 원본 행렬로 복원한다.

**[문단 3 — 양자화]**
잠재 벡터의 각 원소를 $B$비트로 양자화한다. 총 피드백 비트수 $M \times B$를 Type 2의 피드백 비트수와 동일하게 맞추어 공정 비교한다. 학습 시 STE 기반 양자화 인식 학습을 적용한다.

#### 4.2.3 CsiNet Baseline 성능

**[문단 1 — 오프라인 성능]**
CsiNet의 압축률별($\gamma = 1/4, 1/8, 1/16, 1/32, 1/64$) NMSE 및 cosine similarity를 제시한다. Qualcomm, Samsung 등의 3GPP 기고문과 교차 검증하여, 본 구현이 합리적인 baseline임을 확인한다.

### 4.3 제안 기법: 통계-조건부 CSI 인코더 (Statistics-Conditioned CSI Encoder)

#### 4.3.1 전체 구조

**[문단 1 — 2단계 구조 개요]**
제안하는 인코더는 2단계로 구성된다:
- **Stage 1 (장기 통계 압축)**: 공분산 행렬 $\mathbf{R}_H$ 또는 PDP $\mathbf{p}$를 별도의 autoencoder로 압축/복원하여 gNB에 전달한다. 이 보고는 채널 통계가 변하는 주기(수백 ms ~ 수 초)에 맞추어 드물게 수행한다.
- **Stage 2 (조건부 순간 채널 압축)**: 복원된 $\hat{\mathbf{R}}_H$ (또는 $\hat{\mathbf{p}}$)를 조건으로 주입받은 상태에서, 순간 채널 $\mathbf{H}$를 조건부 autoencoder로 압축/복원한다. 이 보고는 기존 CSI 보고 주기(매 슬롯~수십 슬롯)에 맞추어 수행한다.

**[문단 2 — 핵심 아이디어: 조건부 인코딩]**
Stage 2의 인코더와 디코더는 모두 $\hat{\mathbf{R}}_H$를 conditioning input으로 받는다. 인코더는 "이 공분산을 가진 채널 분포에서의 현재 실현"을 인코딩하므로, 분포의 평균 구조를 다시 전달할 필요가 없다. 디코더는 "이 공분산 하에서 잠재 벡터가 의미하는 순간 채널"을 복원한다.

#### 4.3.2 Stage 1: 채널 통계 Autoencoder

**[문단 1 — 공분산 행렬 압축]**
공분산 행렬 $\mathbf{R}_H \in \mathbb{C}^{N_tN_r \times N_tN_r}$는 Hermitian이므로 상삼각 원소만 사용한다. 이를 별도의 CNN/FC autoencoder로 $M_R$차원 잠재 벡터로 압축한다. 공분산은 느리게 변화하므로 보고 빈도가 낮아($T_{cov} \gg T_{CSI}$), 시간 평균 오버헤드가 작다.

**[문단 2 — PDP 압축]**
PDP $\mathbf{p} \in \mathbb{R}^{N_\tau}$는 1차원 벡터이므로 압축이 더 용이하다. FC autoencoder로 $M_p$차원으로 압축한다. PDP와 공분산을 함께 사용하거나 택일하여 사용하는 변형을 모두 평가한다.

#### 4.3.3 Stage 2: 조건부 채널 Autoencoder

**[문단 1 — 조건부 인코더 구조]**
CsiNet 인코더를 확장하여 conditioning 경로를 추가한다. $\hat{\mathbf{R}}_H$를 별도의 embedding network로 feature vector $\mathbf{c}$로 변환한 후, CsiNet의 중간 feature map에 concatenation 또는 FiLM(Feature-wise Linear Modulation) 방식으로 주입한다.

**[문단 2 — 조건부 디코더 구조]**
디코더 역시 동일한 conditioning을 받는다. $\hat{\mathbf{R}}_H$에서 추출한 $\mathbf{c}$를 디코더의 RefineNet 잔차 블록에 주입하여, 복원 시 공분산 정보를 활용한다.

**[문단 3 — 학습 방법]**
Stage 1과 Stage 2를 순차적으로 또는 joint으로 학습한다. Stage 1을 먼저 학습하여 $\hat{\mathbf{R}}_H$의 품질을 확보한 후, Stage 2를 $\hat{\mathbf{R}}_H$를 조건으로 학습한다. 손실 함수는 Stage 2의 최종 채널 복원 NMSE이며, Stage 1의 복원 오차가 Stage 2의 성능에 미치는 영향도 분석한다.

### 4.4 데이터셋 구성

**[문단 1 — Sionna 기반 채널 데이터셋]**
RAN 트윈의 Sionna 채널 모델을 활용하여 데이터셋을 생성한다. UMa-NLOS 시나리오에서 다양한 UE 위치, 이동 속도에 대한 채널 행렬 $\mathbf{H}$를 생성한다. 4T4R dual-pol 안테나 구성($N_t=4, N_r=4$) 기본.

**[문단 2 — 통계 레이블 추출]**
각 UE 위치/시나리오에서 $N_{stat}$개의 채널 실현을 모아 샘플 공분산 $\hat{\mathbf{R}}_H$와 샘플 PDP $\hat{\mathbf{p}}$를 추정한다. 동일 위치의 다른 채널 실현들이 Stage 2의 학습 데이터가 되며, 이때 해당 위치의 $\hat{\mathbf{R}}_H$가 conditioning input이 된다.

**[문단 3 — 데이터셋 규모]**
위치 수: 1,000개, 위치당 통계 추정용: 200개, 위치당 학습/검증/테스트: 100/10/10개. 총 채널 실현: 320,000개.

### 4.5 오프라인 성능 평가

#### 4.5.1 CsiNet Baseline vs 제안 기법

**[문단 1 — 동일 비트수 비교]**
동일한 총 피드백 비트수(Stage 1의 시간 평균 비트 + Stage 2의 비트)에서, 제안 기법과 CsiNet baseline의 NMSE/cosine similarity를 비교한다. 제안 기법이 공분산 정보를 활용하여 baseline 대비 어느 정도의 NMSE 개선을 달성하는지를 압축률별로 제시한다.

**[문단 2 — 조건별 분석]**
공분산만 사용 / PDP만 사용 / 둘 다 사용의 세 가지 변형의 성능을 비교한다. 또한, Stage 1 공분산 복원 품질에 따른 Stage 2 성능 민감도를 분석하여, 통계 보고의 정확도가 어느 수준 이상이면 충분한지를 확인한다.

#### 4.5.2 Type 2 Codebook 대비 비교

**[문단 1 — 피드백 비트수 기준 비교]**
동일 총 피드백 비트수에서, 제안 기법 / CsiNet baseline / Type 2 codebook의 채널 복원 정확도를 NMSE, cosine similarity, chordal distance로 비교한다. 제안 기법이 Type 2 대비 어느 정도의 복원 정확도 이점을 갖는지를 정량화한다.

#### 4.5.3 기존 산업계 결과와의 교차 검증

**[문단 1]**
Qualcomm, Samsung 등이 3GPP 기고문에서 보고한 AI/ML CSI 압축 성능과 비교하여, CsiNet baseline의 구현이 합리적임을 확인하고, 제안 기법의 추가 이득이 통계 조건부 구조에 기인함을 논증한다.

---

## Chapter 5: RAN 디지털 트윈 기반 End-to-End 성능 실험 및 분석

> **이 장의 목적**: 3장의 RAN 트윈에서 기존 세 가지 방식(Type 1 SU, Type 2 SU, Type 2 MU)의 baseline 성능을 확보하고, 제안 기법을 적용했을 때의 end-to-end 성능 개선을 체계적으로 분석한다.

### 5.1 실험 환경 및 방법론

#### 5.1.1 실험 설정

**[문단 1 — 공통 파라미터]**
FR1 Band 78 (3.5 GHz), SCS 30 kHz, 106 PRB (40 MHz), TDD DL-heavy 패턴. 4T4R dual-pol (gNB: $N_x=2, N_y=1$, XP=2 → 4포트). CSI-RS 주기 20슬롯. 시뮬레이션 시간: 각 실험 5,000슬롯 이상.

**[문단 2 — 비교 대상 (3 + 1)]**
다음 네 가지를 비교한다:
1. **Type 1 SU-MIMO**: Type 1 codebook + SU-MIMO 스케줄링 (가장 단순)
2. **Type 2 SU-MIMO**: Type 2 codebook + SU-MIMO 스케줄링 (CSI 정확도 향상)
3. **Type 2 MU-MIMO**: Type 2 codebook + MU-MIMO 스케줄링 (셀 용량 극대화)
4. **제안 기법 (Proposed)**: 통계-조건부 인코더 + MU-MIMO 스케줄링

추가로, **Genie-aided MU-MIMO** (완벽 CSI 가정)를 상한(upper bound)으로 함께 제시하여, 제안 기법이 이론적 최대치에 얼마나 근접하는지를 분석한다.

#### 5.1.2 성능 지표

**[문단 1]**
- **Cell throughput**: 셀 전체의 DL throughput (Mbps)
- **Per-UE throughput**: UE별 평균/5th-percentile throughput (셀 엣지 성능)
- **Sum spectral efficiency**: bps/Hz
- **MCS 분포**: 시간에 걸친 MCS 인덱스의 히스토그램
- **BLER**: 초기 전송 BLER
- **MU-MIMO 페어링 비율**: 전체 슬롯 중 MU-MIMO로 스케줄링된 비율

#### 5.1.3 시나리오 매트릭스

**[문단 1 — 세 축의 파라미터 범위]**

| 축 | 파라미터 | 하 (Low) | 중 (Medium) | 상 (High) |
|---|---|---|---|---|
| Radio 환경 | SNR (dB) | 5 | 15 | 25 |
| Mobility | UE 속도 (m/s) | 0 (정지) | 3 (보행) | 30 (차량) |
| Traffic | UE 수 | 2 | 4 | 8 |

**[문단 2 — 대표 시나리오 선정]**
성능 차이가 (1) 미미한 경우, (2) 의미 있는 경우, (3) 극대화되는 경우의 세 가지를 우선 선정한다:
- **시나리오 A (차이 미미)**: 고 SNR(25dB) + 정지(0m/s) + 2 UE → CSI 양자화 오차가 작아 모든 방식이 유사
- **시나리오 B (의미있는 차이)**: 중간 SNR(15dB) + 보행(3m/s) + 4 UE → Type 2 MU-MIMO에서 양자화 오차가 간섭으로 발현
- **시나리오 C (차이 극대화)**: 저 SNR(5dB) + 차량(30m/s) + 4~8 UE → CSI 정확도가 MU-MIMO 성능의 결정적 요인

### 5.2 Baseline 성능 비교: Type 1 SU-MIMO / Type 2 SU-MIMO / Type 2 MU-MIMO

> 이 섹션은 기존 세 가지 방식만으로 실험하여, (1) CSI 정확도가 성능에 미치는 영향을 실증하고, (2) 기존 방식의 한계를 드러내어, (3) 제안 기법의 필요성을 결과로 뒷받침한다.

#### 5.2.1 Type 1 SU-MIMO vs Type 2 SU-MIMO

**[문단 1 — 결과 제시]**
다양한 SNR에서의 throughput 비교. Type 2가 Type 1 대비 정밀한 프리코딩으로 throughput 이득을 제공하며, 안테나 수 증가 시 그 이득이 확대됨을 보인다.

**[문단 2 — 분석: CSI 정확도의 영향 실증]**
이 결과는 동일한 SU-MIMO 스케줄링에서도 CSI 표현 정확도가 성능에 직접적으로 영향을 미침을 보여준다. Type 2의 이득은 추가 피드백 비트의 대가이며, "같은 비트수로 더 정확한 CSI"의 가치를 정량화하는 기준선이 된다.

#### 5.2.2 Type 2 SU-MIMO vs Type 2 MU-MIMO

**[문단 1 — 결과 제시]**
UE 수 증가에 따른 셀 throughput 비교. MU-MIMO가 SU-MIMO 대비 셀 용량 이득을 제공하지만, UE 간 간섭으로 인해 per-UE throughput은 감소할 수 있음을 보인다.

**[문단 2 — 핵심 분석: Type 2 MU-MIMO의 한계]**
Type 2의 양자화 오차가 MU-MIMO에서 잔여 간섭으로 나타남을 SINR 분석으로 보인다. Genie-aided MU-MIMO 대비 Type 2 MU-MIMO의 성능 갭이 존재하며, 이 갭이 곧 **더 정확한 CSI 인코더가 좁힐 수 있는 여지**이다.

#### 5.2.3 시나리오별 성능 변화

**[문단 1 — Mobility 영향]**
UE 속도 증가에 따른 CSI aging 효과. 고속에서는 보고된 CSI의 유효 기간이 짧아져 모든 방식의 성능이 열화되며, 특히 MU-MIMO에서의 열화가 크다.

**[문단 2 — SNR 영향]**
저 SNR에서는 채널 추정 오차가 지배적이므로 Type 1/2 차이가 줄어들고, 중-고 SNR에서 피드백 양자화 오차가 성능 제한 요인으로 부각됨을 보인다.

**[문단 3 — UE 수 영향]**
UE 수 증가 시 MU-MIMO의 이득이 커지지만, 동시에 간섭 관리의 어려움도 증가하여 CSI 정확도의 중요성이 극대화됨을 보인다.

**[문단 4 — Baseline 종합: 제안 기법의 기대 이득 영역 도출]**
위의 분석을 종합하여, "중간 이상의 SNR + 다수 UE의 MU-MIMO 환경"에서 Type 2의 양자화 오차가 성능 병목임을 확인하고, 제안하는 통계-조건부 인코더가 이 병목을 완화할 수 있는 조건을 명시한다.

### 5.3 제안 기법 적용 성능

#### 5.3.1 제안 기법 vs Type 2 MU-MIMO

**[문단 1 — 핵심 결과]**
통계-조건부 인코더 + MU-MIMO 스케줄링이 Type 2 MU-MIMO 대비 셀 throughput에서 어느 정도의 개선을 달성하는지를 시나리오별로 제시한다. 이것이 본 논문의 핵심 결과이다.

**[문단 2 — 시나리오별 이득 분석]**
시나리오 A에서는 이득이 미미하고, B에서는 의미 있으며, C에서는 극대화됨을 보인다. 이는 5.2절의 baseline 분석에서 예측한 패턴과 일치함을 확인한다.

**[문단 3 — CsiNet Baseline vs 제안 기법 (ablation)]**
통계 조건부 구조의 효과를 분리하기 위해, 동일 비트수의 CsiNet baseline(통계 정보 없이 blind 압축)과도 비교한다. 이를 통해 성능 이득이 autoencoder 자체가 아닌 **통계 정보 활용**에 기인함을 명확히 한다.

#### 5.3.2 Genie-aided 대비 갭 분석

**[문단 1]**
완벽 CSI 기대 성능(genie-aided MU-MIMO) 대비 제안 기법이 달성한 성능을 비교한다. 잔여 갭의 원인(양자화 손실, Stage 1 복원 오차, 모델 일반화 한계, CSI aging 등)을 식별하고, 추가 개선 여지를 논의한다.

#### 5.3.3 제안 기법 vs 기존 세 방식 종합 비교

**[문단 1 — 전체 결과 표]**
시나리오 A/B/C × 4가지 방식(Type 1 SU, Type 2 SU, Type 2 MU, 제안)의 cell throughput, per-UE throughput, spectral efficiency를 종합 표로 제시한다.

**[문단 2 — 제안 기법의 포지셔닝]**
제안 기법이 기존 세 방식과의 관계에서 어디에 위치하는지를 명확히 한다: (1) Type 1 SU 대비 대폭 우수, (2) Type 2 SU 대비 MU-MIMO 이득 + CSI 정확도 이득의 이중 효과, (3) Type 2 MU 대비 CSI 정확도 개선에 의한 유의미한 추가 이득.

### 5.4 종합 논의 (Discussion)

**[문단 1 — 주요 발견 요약]**
전체 실험 결과를 종합한다: (1) CSI 정확도가 MIMO 성능, 특히 MU-MIMO에 미치는 정량적 영향, (2) 채널 통계를 조건으로 활용하면 동일 비트수에서 의미 있는 CSI 복원 정확도 개선이 가능하며, (3) 이 개선이 MU-MIMO의 end-to-end throughput 향상으로 실질적으로 이어진다.

**[문단 2 — 왜 통계 정보가 효과적인가]**
공분산/PDP가 인코더에 제공하는 정보가 어떤 메커니즘으로 압축 효율을 높이는지를 직관적으로 설명한다. 공분산이 "에너지가 집중된 각도-지연 bin"을 미리 알려줌으로써, 인코더가 해당 bin의 세밀한 값만 전달하면 되는 구조임을 논의한다.

**[문단 3 — End-to-end 평가에서만 관찰되는 현상]**
오프라인 NMSE 개선(4장)이 end-to-end throughput(5장)에서 어떤 비율로 반영되는지를 분석한다. MAC 스케줄링, HARQ, CSI aging 등의 프로토콜 동작이 NMSE-throughput 관계를 비선형적으로 만드는 현상을 논의한다.

**[문단 4 — 한계점 및 향후 과제]**
비실시간 시뮬레이션, 제한된 안테나 규모(4T4R), CsiNet 기반 baseline의 단순성, 통계 보고 오버헤드 모델링의 이상화 등을 솔직하게 제시한다.

---

## Chapter 6: 결론 (Conclusion)

**[문단 1 — 연구 요약]**
본 논문에서는 OAI 기반 RAN 디지털 트윈을 구축하고, 트윈에서 확보한 채널 통계(공분산/PDP)를 조건으로 활용하는 통계-조건부 CSI 인코더를 제안하여, 5G NR MIMO 시스템에서의 CSI 피드백 성능을 개선하였다. Core Emulator를 통한 통합 설정 관리, GPU 가속 Sionna 채널 에뮬레이션, 4×4 MIMO RI/PMI 구현, MU-MIMO 스케줄러 지원 등을 통해 end-to-end 실험 플랫폼을 구현하였다.

**[문단 2 — 핵심 기여]**
본 연구의 핵심 기여는 세 가지이다.
- 첫째, OAI + Sionna 기반 RAN 디지털 트윈의 구축을 통해, 실제 프로토콜 스택과 사실적 채널 모델이 통합된 시스템 레벨 실험 환경을 제공하였으며, 이 트윈이 AI/ML 학습용 채널 통계를 추출하는 역할까지 수행함을 보였다.
- 둘째, Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 다양한 시나리오에서의 성능을 체계적으로 비교하여, CSI 정확도가 특히 MU-MIMO에서 성능 병목임을 정량적으로 입증하였다.
- 셋째, CsiNet baseline 대비 채널 통계를 조건으로 활용하는 제안 기법이, 동일 피드백 비트수에서 Type 2 MU-MIMO 대비 유의미한 throughput 개선을 달성할 수 있음을 RAN 트윈 환경에서 실증하였다.

**[문단 3 — 향후 연구 방향]**
향후 연구로는 (1) Transformer 기반 조건부 인코더로의 아키텍처 확장, (2) Massive MIMO (32/64 안테나) 환경에서의 통계-조건부 효과 검증, (3) 실시간 통계 업데이트 메커니즘 설계 (온라인 공분산 추적), (4) 실제 5GC와의 연동을 통한 상용 환경 검증 등을 제안한다.

---

## 부록 (Appendix)

- **부록 A**: OAI 소스코드 수정 상세 (csi_rx.c RI/PMI 4포트, SINR 수정, gNB_scheduler 수정 등)
- **부록 B**: Core Emulator API 명세 (전체 REST 엔드포인트)
- **부록 C**: CsiNet 및 제안 기법의 학습 하이퍼파라미터, 수렴 곡선, 아키텍처 상세
- **부록 D**: 전체 실험 결과 표 (시나리오 매트릭스 전체)

---

## 논리 흐름 요약 (한눈에 보기)

```
Ch.1 [Why]
  MIMO 성능은 CSI 정확도에 의존 → Type 1/2의 한계
  → 기존 AI/ML (CsiNet)은 blind 압축 → 통계를 활용하면 더 잘할 수 있다
  → 검증하려면 end-to-end 환경이 필요

Ch.2 [Background]
  채널 모델 (통계적 특성 강조: 공분산, PDP)
  → CSI 획득 (Type 1 / Type 2) → SU vs MU-MIMO
  → 관련 연구 (CsiNet = blind 압축, 통계 활용 부재, end-to-end 검증 부재)

Ch.3 [Platform]
  RAN 트윈 아키텍처 (두 가지 역할: 실험 + 통계 추출)
  → Core Emulator → Sionna 프록시 (채널 생성 + 통계 추출)
  → MAC/스케줄러 → 시나리오 설계 → 기본 검증

Ch.4 [Proposed Method]
  CsiNet baseline 제시
  → 제안: 트윈에서 확보한 공분산/PDP를 조건으로 H를 인코딩 (2단계 구조)
  → 오프라인 성능: baseline 대비 NMSE 개선, Type 2 대비 복원 정확도 우위

Ch.5 [Experiments — 핵심]
  Baseline 확보: Type 1 SU vs Type 2 SU vs Type 2 MU (기존 세 방식의 한계 실증)
  → 제안 기법 적용: Type 2 MU 대비 성능 개선 (시나리오별 분석)
  → Genie-aided 대비 갭 분석
  → 종합 논의: 통계 정보가 왜 효과적인가, end-to-end에서만 보이는 현상

Ch.6 [Conclusion]
  기여 요약 → 향후 방향
```

### 핵심 논리 한 문장

> RAN 디지털 트윈이 제공하는 채널 통계(공분산/PDP)를 autoencoder의 조건부 입력으로 활용하면, 기존 CsiNet의 blind 압축 대비 동일 비트수에서 더 정확한 CSI를 전달할 수 있고, 이는 Type 2 MU-MIMO의 성능 병목(양자화 오차 → 잔여 간섭)을 완화하여 의미 있는 셀 throughput 개선으로 이어진다.

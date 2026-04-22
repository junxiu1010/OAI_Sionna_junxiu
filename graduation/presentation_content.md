# 발표 자료 콘텐츠 정리

> **논문 제목**: Statistics-Conditioned AI/ML CSI Feedback for Practical 5G MU-MIMO Transmission
>
> 발표 구성: Part 1 (배경/기존연구/Contribution) 3~4p → Part 2 (Contribution 상세) 3~4p → Part 3 (실험환경/결과) 5~6p

---

## Part 1: 연구 배경 / 기존 연구 / Contribution (3~4페이지)

---

### 슬라이드 1-1: 연구 배경 — 5G MU-MIMO와 CSI 병목

**핵심 메시지**: MU-MIMO는 셀 용량 극대화의 핵심이지만, CSI 정확도에 의해 성능이 제한된다

**내용:**

- 5G NR에서 MIMO 기술의 역할
  - SU-MIMO: 공간 다중화로 단일 UE의 throughput 향상
  - MU-MIMO: 동일 시간-주파수 자원에서 다수 UE 동시 서비스 → 셀 용량 증가
- FDD 환경에서 UE는 CSI-RS 측정 후 RI/PMI/CQI로 압축하여 기지국에 피드백
- CSI 오차의 영향 차이
  - SU-MIMO: 빔포밍 이득 감소 수준
  - **MU-MIMO: UE간 간섭 억제 실패 → 잔여 간섭(residual inter-user interference)으로 직결**
    - 스케줄러: 실제로 간섭이 큰 UE 쌍을 orthogonal로 오판하여 선택
    - 프리코더: 왜곡된 채널 행렬로 RZF 계산 → 간섭 억제 불완전
- **결론**: "동일 피드백 비트수에서 더 정확한 CSI 전달"이 MU-MIMO 성능의 핵심 과제

---

### 슬라이드 1-2: 기존 접근법 분류 및 한계

**핵심 메시지**: Type 1 → Type 2 → AI/ML CSI로 발전했지만, 각각 한계가 존재

**내용:**

| 방식 | 장점 | 한계 |
|------|------|------|
| **Type 1 Codebook** (TS 38.214) | 오버헤드 작음 (단일 DFT 빔 선택) | 다중 클러스터 채널 표현 불가, MU-MIMO에 부적합 |
| **Type 2 Codebook** (TS 38.214) | L개 빔 선형결합, 정밀한 채널 표현 | Type 1 대비 3~5배 오버헤드, 위상/진폭 양자화 정밀도 제한 |
| **AI/ML CSI (CsiNet 등)** | 동일 비트에서 codebook 대비 NMSE 개선 | (1) 순간 채널 H만 blind 압축 — 장기 통계 미활용 (2) 오프라인 NMSE만 평가 — E2E 검증 부재 |

- 핵심 갭: 기존 AI/ML CSI 연구는 채널의 장기 구조(공분산 R_H, PDP)를 활용하지 않고, 실제 프로토콜 스택에서의 성능 검증도 없음

**그림**: Figure 2.1 (Type I vs Type II CSI 비교 다이어그램 — 논문 PDF에서 캡처)

---

### 슬라이드 1-3: 제안 동기 및 핵심 Contribution

**핵심 메시지**: RAN Digital Twin이 제공하는 장기 채널 통계를 조건으로 활용하면 CSI 압축 효율을 높일 수 있다

**동기:**

- 무선 채널은 두 가지 시간 스케일의 특성을 가짐
  - 장기 통계 (수백 ms ~ 수 초): 공분산 행렬 R_H, 전력 지연 프로파일 PDP — 환경/위치에 의존
  - 순간 채널 (매 슬롯): small-scale fading에 의존
- 정보이론적 근거: 장기 통계를 알면 조건부 엔트로피 H(vec(H)|R_H) < H(vec(H)) → 같은 비트로 더 정확한 압축 가능
- RAN Digital Twin에서 장기 통계를 사전 확보 가능 → "합리적으로 확보 가능한 side information"

**3가지 Contribution:**

1. **OAI-Sionna 기반 RAN Digital Twin 구축**
   - 실제 프로토콜 스택 + 사실적 채널 모델 통합
   - AI/ML 학습용 채널 통계 추출 겸용 (이중 역할)

2. **실용적 MU-MIMO 전송 체인 구현**
   - Type 2 PMI 처리 + SUS+PF 스케줄링 + RZF 프리코딩
   - CSI 품질이 직접 성능에 영향을 미치는 경로 확보

3. **통계-조건부 CSI 인코더 제안 및 E2E 검증**
   - CsiNet 확장, 공분산/PDP를 조건으로 주입
   - 기존 5개 방식 대비 end-to-end 성능 개선 실증

---

## Part 2: Contribution 상세 (3~4페이지)

---

### 슬라이드 2-1: Contribution 1 — RAN Digital Twin 아키텍처

**핵심 메시지**: OAI + Sionna + Core Emulator로 구성된 이중 역할(실행 + 통계추출) 플랫폼

**3대 구성요소:**

- **OAI 5G 프로토콜 스택**: gNB/UE의 RRC, MAC, PHY 전체 동작
- **Sionna 채널 프록시**: GPU IPC 기반 IQ 샘플 교환
  - DL: gNB 송신 IQ → UE별 독립 채널 행렬 적용 → 각 UE에 전달 (Broadcast)
  - UL: 각 UE 송신 IQ → 채널 적용 후 합산 → gNB에 전달 (Superposition)
  - 환경 인식 다중 UE 채널 생성 + 장기 통계(R_H, PDP) 추출
- **Core Emulator**: FastAPI 기반 중앙 설정 관리 서버
  - 12개 섹션 YAML 통합 설정 (system, antenna, codebook, csi_rs 등)
  - Jinja2 템플릿으로 gnb.conf, CN5G config 자동 렌더링
  - 프리셋 시스템: Type 1 SU / Type 2 SU / Type 2 MU 원클릭 전환
  - 파라미터 유효성 검증 + 런타임 핫스왑

**이중 역할:**

- 역할 1 — **실험 플랫폼**: 5개 전송 모드의 E2E 성능을 동일 조건에서 비교
- 역할 2 — **통계 추출 플랫폼**: 동일 환경에서 채널 실현을 축적하여 공분산/PDP 추출 → AI/ML 학습에 활용
- 학습 환경과 평가 환경의 일치 보장

**그림**: Figure 3.1 (4T4R OAI-Sionna RAN Twin 구조도), `core_emulator_control_flow.png`

---

### 슬라이드 2-2: Contribution 2 — MU-MIMO 스케줄링/프리코딩 구현

**핵심 메시지**: OAI에 Type 2 PMI 처리 + SUS+PF 스케줄러 + RZF 프리코더를 구현하여 실용적 MU-MIMO 체인 완성

**Type 2 PMI 기반 채널 재생성:**

- UE 보고 PMI → gNB에서 채널 방향 벡터 복원
- Type 1 vs Type 2 정규화 상관도 비교: SNR 증가에 따라 Type 2 우위 확대
  - 저 SNR: 두 방식 유사
  - 중~고 SNR: Type 2가 유의미하게 높은 상관도 달성

**SUS+PF 스케줄러:**

- Semi-orthogonality 기준: ρ_ij = |h_i^H h_j| / (||h_i|| ||h_j||)
  - ρ_ij가 낮으면 → 동시 전송에 유리 (orthogonal에 가까움)
  - ρ_ij가 높으면 → 유사한 공간 방향 → MU-MIMO 파트너로 부적합
- PF utility: M_u = R_u^inst / R_u_bar (기회적 스케줄링 + 공정성 유지)
- 런타임 검증: 8 UE 중 4 UE 선택, 높은 상관도 UE는 reject

**RZF 프리코더:**

- W = H_S^H (H_S H_S^H + αI)^{-1}
- CSI 오차에 강건 (순수 ZF 대비), 실용적 limited-feedback 환경에 적합

**핵심**: 스케줄러와 프리코더 모두 **보고된 실용 CSI에 기반** → CSI 품질 향상이 시스템 성능으로 직결되는 두 가지 경로 (user grouping + interference suppression) 확보

**그림**: Figure 3.3 (PMI correlation vs SNR), Figure 3.4 (MU-MIMO scheduling log)

---

### 슬라이드 2-3: Contribution 3 — 통계-조건부 CSI 인코더

**핵심 메시지**: CsiNet의 blind 압축을 2단계 조건부 압축으로 확장

**기존 CsiNet (blind 압축):**

```
z = f_enc(H)         ← 순간 채널만 입력
H_hat = f_dec(z)
```

**제안 기법 (조건부 압축):**

```
z = f_enc(H, c)      ← 장기 통계 embedding c를 조건으로 주입
H_hat = f_dec(z, c)
```

**2단계 구조:**

- **Stage 1 (장기 통계 압축)**
  - 공분산 R_H (Hermitian → 상삼각 원소만) → 통계 autoencoder → z_R
  - PDP p (1D 실수 벡터) → FC autoencoder → z_p
  - 갱신 주기 T_stat >> T_CSI → 시간 평균 오버헤드 작음

- **Stage 2 (조건부 순간 채널 압축)**
  - 복원된 R_hat → embedding network → feature vector c
  - c를 CsiNet 중간 feature map에 FiLM 또는 Concatenation으로 주입
  - **인코더**: "이 환경에서의 순간 채널 편차"만 효율적으로 압축
  - **디코더**: "이 환경 하에서 잠재 벡터가 의미하는 채널"을 복원

**오프라인 결과:**

- 모든 시나리오(UMi-LOS, UMa-LOS, UMa-NLOS)에서 CsiNet 대비 NMSE 개선
- 모든 압축률(1/4 ~ 1/64)에서 일관된 이득
- 특히 UMa-NLOS에서 가장 큰 이득 (분산된 채널 구조에서 통계 side info 가치 극대화)

**그림**: Figure 4.1 (CsiNet 구조), Figure 4.2 / `csinet_nmse_comparison.png` (NMSE 비교)

---

## Part 3: 실험 환경 및 결과 (5~6페이지)

---

### 슬라이드 3-1: 실험 환경 설정

**핵심 메시지**: 단일 프레임워크에서 5개 모드를 공정하게 비교

**시스템 파라미터:**

| 항목 | 설정 |
|------|------|
| 대역 | FR1 Band 78 (3.5 GHz) |
| SCS / BW | 30 kHz / 106 PRB (~40 MHz) |
| 듀플렉스 | TDD (DL-heavy 패턴) |
| 안테나 | 4T4R dual-pol (N_x=2, N_y=1, XP=2) |
| CSI-RS 주기 | 20 슬롯 |

**5가지 비교 대상:**

| # | 모드 | CSI 방식 | 전송 방식 |
|---|------|----------|-----------|
| 1 | Type 1 SU-MIMO | Type 1 Codebook | SU-MIMO |
| 2 | Type 2 SU-MIMO | Type 2 Codebook | SU-MIMO |
| 3 | Type 2 MU-MIMO | Type 2 Codebook | MU-MIMO (SUS+PF, RZF) |
| 4 | Baseline Encoder | blind CsiNet | MU-MIMO (SUS+PF, RZF) |
| 5 | **Proposed** | **통계-조건부 CsiNet** | MU-MIMO (SUS+PF, RZF) |

**3개 전파 시나리오:**

| 시나리오 | 특징 |
|----------|------|
| UMi-LOS | 마이크로셀, 강한 LOS 성분, 방향성 집중 |
| UMa-NLOS | 매크로셀, 반사/산란 지배, 넓은 각도 확산, 풍부한 다중경로 |
| UMa-LOS | 매크로셀 + LOS, UMi-LOS보다 넓은 공간 가변성 |

**성능 지표**: BLER, 평균 MCS Index, Cell DL Throughput (Mbps)

**그림**: `graduation/figures/3gpp_channel_parameters_table.png`

---

### 슬라이드 3-2: Calibrated Baseline 성능 (3모드 × 3시나리오)

**핵심 메시지**: Type 1 → Type 2 SU → Type 2 MU 순으로 성능 향상, 다중 UE에서 MU-MIMO 이점 확인, 동시에 CSI 한계도 확인

**공통 패턴:**

- Type 2 SU > Type 1 SU: CSI 정확도 향상 효과 확인
- Type 2 MU가 다중 UE(2~8)에서 최고 cell throughput: 공간 다중화 이득
- 단, 고 UE에서 CSI 양자화 오차 → 잔여 간섭 → MU-MIMO 이득 제한

**시나리오별 특징:**

- **UMi-LOS**:
  - MU-MIMO가 2~4 UE에서 최고 throughput
  - 고 UE에서 LOS 방향 유사성으로 BLER 소폭 증가
  - Type 1 ~33 Mbps (1 UE), Type 2 SU ~70 Mbps, Type 2 MU ~82 Mbps (4 UE)

- **UMa-NLOS**:
  - 세 모드 간 차이 가장 큼
  - Type 1 BLER 26~29%, Type 2 SU ~15~18%, Type 2 MU ~9~15%
  - MU-MIMO가 2 UE에서 throughput 최대 (~28 Mbps), 이후 점진적 감소

- **UMa-LOS**:
  - MU-MIMO가 2 UE부터 throughput 우위, 4~8 UE에서 최대 (~49 Mbps)
  - Type 2 SU 대비 MU-MIMO의 BLER 차이가 고 UE에서 축소

→ **"더 정확한 CSI가 Type 2 MU-MIMO의 성능 갭을 줄일 수 있다"는 실험적 동기 확보**

**그림**: `calibrated_UMi_LOS_bar.png`, `calibrated_UMa_NLOS_bar.png`, `calibrated_UMa_LOS_bar.png`

---

### 슬라이드 3-3: E2E 5-Mode 결과 — UMi-LOS & UMa-LOS

**핵심 메시지**: LOS 환경에서도 제안 기법이 최고 성능, blind CsiNet 대비 일관된 추가 이득

**UMi-LOS:**

- 두 AI 기반 방식(Baseline Encoder, Proposed) 모두 Type 2 MU 대비 throughput 개선
- **제안 기법이 전 UE 범위에서 최고 throughput**
- blind CsiNet과의 차이는 moderate → LOS 지배적 환경에서는 순간 채널만으로도 상당 부분 포착 가능
- 그러나 통계 조건부가 여전히 추가 이득 제공

**UMa-LOS:**

- 제안 기법이 전 UE 범위에서 최고 성능
- blind CsiNet 대비 gain 존재하나 UMa-NLOS보다는 작음
- macro-cell의 넓은 공간적 가변성이 통계 조건부의 가치를 만듦

**그림**: `e2e_5mode_UMi_LOS.png`, `e2e_5mode_UMa_LOS.png`

---

### 슬라이드 3-4: E2E 5-Mode 결과 — UMa-NLOS (핵심 시나리오)

**핵심 메시지**: UMa-NLOS에서 통계-조건부 인코더의 이득이 가장 크다 — 논문의 핵심 결과

**결과:**

- 제안 기법이 **BLER, 평균 MCS, Cell Throughput 모든 지표에서 5개 방식 중 최고**
- blind CsiNet 대비 **가장 큰 추가 이득** — 특히 저/중 UE 영역에서 두드러짐
- Type 2 MU-MIMO 대비 substantial한 개선

**이유 분석:**

- NLOS 환경: 더 분산된 채널 구조(넓은 각도 확산, 풍부한 다중경로)
- blind 인코더: 매 샘플마다 환경 구조를 처음부터 추론해야 → 제한된 latent 용량을 구조 표현에 소모
- **통계 조건부**: 환경의 평균 구조(공분산, PDP)를 미리 알려줌 → 인코더가 순간 변동에만 집중 가능
- → 장기 통계가 가장 필요한 환경에서 가장 큰 이득이 발생

**그림**: `e2e_5mode_UMa_NLOS.png`

---

### 슬라이드 3-5: Cross-Scenario 분석, 핵심 발견, 결론

**핵심 메시지**: 제안 기법은 모든 시나리오에서 최고 성능이며, 이득의 크기는 채널 복잡도에 비례

**시나리오별 이득 크기:**

```
UMa-NLOS  >>>  UMa-LOS  >  UMi-LOS
(이득 최대)              (이득 moderate)
```

- LOS 환경: 채널이 directional → blind 인코더도 상당 부분 포착 → 통계 side info 추가 이득 작음
- NLOS 환경: 채널이 dispersed → blind 인코더 한계 → 통계 side info 가치 극대화

**오프라인 NMSE vs E2E Throughput 관계:**

- NMSE 개선이 항상 비례적 throughput 개선은 아님 (비선형 관계)
- CSI 개선 → SUS 스케줄링 결정 변경 (더 좋은 UE 그룹 선택) → 간섭 억제 개선 → throughput 비선형 증가
- MAC 스케줄링, HARQ, CSI aging 등이 NMSE-throughput 관계를 비선형적으로 만듦
- → **E2E 평가의 필수성 확인**: 오프라인 지표만으로는 실제 시스템 이득을 예측할 수 없음

**결론:**

> 실용적 CSI 피드백은 blind 순간채널 압축이 아닌, **환경-조건부 채널 표현**으로 공식화될 때 유의미하게 개선된다. RAN Digital Twin에서 추출한 장기 채널 통계를 활용하여, 동일 피드백 비트수에서 더 정확한 CSI를 전달하고, 이것이 MU-MIMO의 end-to-end throughput 향상으로 이어짐을 실증하였다.

**한계점:**

- 4T4R 스케일 (Massive MIMO 32/64T는 미검증)
- 비실시간 시뮬레이션
- CsiNet 기반의 비교적 단순한 아키텍처
- 통계 보고 오버헤드 모델링의 이상화

**향후 연구 방향:**

1. Transformer 기반 조건부 인코더로 아키텍처 확장
2. Massive MIMO (32/64 안테나) 환경에서 통계-조건부 효과 검증
3. 온라인 통계 업데이트 메커니즘 설계 (실시간 공분산 추적)
4. 실제 5GC 연동을 통한 상용 환경 검증

**그림**: `e2e_5mode_combined.png` (4시나리오 종합)

---

## 사용 가능한 그림 파일 목록

### 아키텍처 / 테이블

| 파일 | 용도 |
|------|------|
| `graduation/figures/core_emulator_control_flow.png` | Core Emulator 제어 흐름도 |
| `graduation/figures/core_emulator_architecture.png` | Core Emulator 내부 아키텍처 |
| `graduation/figures/3gpp_channel_parameters_table.png` | 3GPP 채널 파라미터 비교표 |

### Calibrated Baseline 결과 (막대 도표)

| 파일 | 용도 |
|------|------|
| `graduation/experiments/figures/calibrated_UMi_LOS_bar.png` | UMi-LOS 3모드 비교 |
| `graduation/experiments/figures/calibrated_UMi_NLOS_bar.png` | UMi-NLOS 3모드 비교 |
| `graduation/experiments/figures/calibrated_UMa_LOS_bar.png` | UMa-LOS 3모드 비교 |
| `graduation/experiments/figures/calibrated_UMa_NLOS_bar.png` | UMa-NLOS 3모드 비교 |

### CSINet 오프라인 결과

| 파일 | 용도 |
|------|------|
| `graduation/csinet/figures/csinet_nmse_comparison.png` | CsiNet vs Conditioned CsiNet NMSE |

### E2E 5-Mode 결과

| 파일 | 용도 |
|------|------|
| `graduation/csinet/figures/e2e_5mode_UMi_LOS.png` | UMi-LOS 5모드 E2E |
| `graduation/csinet/figures/e2e_5mode_UMi_NLOS.png` | UMi-NLOS 5모드 E2E |
| `graduation/csinet/figures/e2e_5mode_UMa_LOS.png` | UMa-LOS 5모드 E2E |
| `graduation/csinet/figures/e2e_5mode_UMa_NLOS.png` | UMa-NLOS 5모드 E2E |
| `graduation/csinet/figures/e2e_5mode_combined.png` | 4시나리오 종합 |

### 논문 PDF에서 캡처 필요

| Figure | 용도 |
|--------|------|
| Figure 2.1 | Type I vs Type II CSI 비교 다이어그램 |
| Figure 3.1 | 4T4R OAI-Sionna RAN Twin 구조도 |
| Figure 3.2 | Core Emulator 기능 아키텍처 |
| Figure 3.3 | Type I/II PMI 정규화 상관도 vs SNR |
| Figure 3.4 | MU-MIMO 스케줄링 런타임 로그 |
| Figure 4.1 | CsiNet encoder-decoder 구조 |
| Figure 4.2 | CsiNet vs Conditioned CsiNet NMSE 비교 |

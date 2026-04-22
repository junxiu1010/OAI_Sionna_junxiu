# AWS 연구 환경 플랫폼 PoC 요청사항

---

## 1. 연구 주제

**OAI 기반 RAN 디지털 트윈 구축 및 채널 통계 조건부 AI/ML CSI 압축을 통한 5G NR MIMO 성능 개선**

> Statistics-Conditioned AI/ML CSI Compression for 5G NR MIMO Performance Enhancement  
> Using OAI-Based RAN Digital Twin

### 핵심 키워드

5G NR MIMO, CSI feedback, Type 1 / Type 2 codebook, SU-MIMO, MU-MIMO,  
AI/ML CSI compression, CsiNet (autoencoder baseline), channel covariance matrix,  
power delay profile (PDP), statistics-conditioned encoder, RAN digital twin,  
OpenAirInterface (OAI), Sionna channel model, 3GPP 38.901, GPU IPC

### 연구 범위

- **Baseline 성능 확보**: 기존 구축한 RAN twin에서 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 다양한 RAN 시나리오(SNR, UE 이동속도, UE 수)에서의 throughput/BLER/MCS 성능을 체계적으로 비교
- **AI/ML CSI 인코더 설계**: CsiNet을 baseline autoencoder로 두고, 트윈에서 미리 확보한 채널 공분산 행렬($\mathbf{R}_H$) 또는 전력 지연 프로파일(PDP)을 조건(condition)으로 활용하여 순간 채널 $\mathbf{H}$의 autoencoding 성능을 개선하는 **통계-조건부 CSI 인코더** 제안
- **End-to-End 성능 검증**: 제안 기법을 RAN 트윈에 통합하여, 기존 Type 1 SU-MIMO / Type 2 SU-MIMO / Type 2 MU-MIMO 대비 동일 피드백 비트수에서의 throughput 개선을 실증

### 논문 수집 참고

- 위 키워드 기반 SCI/KCI 논문 40~60편 수집 요청 (학교 계정 별도 전달)

---

## 2. 필요 컴퓨팅 자원 (H100 기준)

### 하드웨어

| 항목 | 사양 | 근거 |
|------|------|------|
| **GPU** | H100 SXM 80GB × 2장 | Sionna 채널 생성 + CUDA Graph IPC 프록시(GPU당 ~20GB), ML 학습(CsiNet/조건부 인코더) 병행 |
| **CPU** | 32코어 이상 (AMD EPYC / Intel Xeon) | OAI gNB + 다수 UE 프로세스 동시 구동, 데이터 전처리 병렬화 |
| **메모리** | 256GB DDR5 | OAI 프로세스(gNB+UE×4~8) + Sionna TF 컨텍스트 + 학습 배치 버퍼 동시 로딩 |
| **스토리지** | NVMe SSD 1TB + 네트워크 스토리지 2TB | 채널 H 데이터셋(수십~수백 GB), OAI 빌드/로그, 학습 체크포인트 |
| **네트워크** | 고대역폭 인터커넥트 (EFA 등) | 멀티 GPU gradient 동기화 (학습 시) |
---

## 3. 학습 데이터

Autoencoder 학습에 필요한 데이터를 Sionna 채널 모델로 자체 생성하며, 세 가지 데이터셋을 구성합니다.

### 3-1. Autoencoder 입력 — 각도-지연 도메인 채널 행렬 $\tilde{H}$

CsiNet 및 제안 기법의 autoencoder가 압축/복원하는 대상입니다.

**생성 절차**: Sionna 3GPP 채널 모델 → 주파수 도메인 $H$ 생성 → 2D DFT → 각도-지연 도메인 $\tilde{H}$ → 지연축 절단

| 항목 | 값 |
|------|-----|
| **원본 채널** | $H \in \mathbb{C}^{N_t \times N_{sc}}$ (TX ant × 서브캐리어) |
| **Autoencoder 입력** | $\tilde{H}_{\text{trunc}} \in \mathbb{R}^{2 \times N_t \times N_c'}$ (실수부/허수부 분리, 지연축 절단) |
| **안테나 구성** | TX(gNB): 4 ant ($2{\times}1$ dual-pol), RX(UE): 4 ant |
| **서브캐리어** | 106 PRB × 12 = 1272 유효 서브캐리어, SCS 30 kHz |
| **지연축 절단** | $N_c' = 32$ (전체 에너지의 95% 이상 보존) |
| **입력 차원** | $2 \times 4 \times 32 = 256$ (실수 원소) |
| **채널 시나리오** | 3GPP UMa-NLOS, UMi-NLOS (Sionna 38.901) |
| **SNR 범위** | 5 / 15 / 25 dB |
| **UE 속도** | 0 (정지) / 3 (보행) / 30 (차량) m/s |

### 3-2. Conditioning 입력 — 채널 통계 (공분산 / PDP)

제안하는 통계-조건부 인코더의 **조건부 입력(conditioning input)**입니다.  
동일 위치/시나리오에서 다수의 채널 실현을 축적하여 추정한 장기 통계이며, Stage 1 autoencoder가 별도로 압축/복원합니다.

| 데이터 | 크기 | 설명 | 인코더 내 역할 |
|--------|------|------|----------------|
| 공분산 $\mathbf{R}_H$ | $(N_t N_r)^2 = 256$ complex elements | 채널 벡터의 샘플 공분산 (Hermitian) | embedding → feature $\mathbf{c}$ → FiLM 조건 주입 |
| PDP $\mathbf{p}$ | $N_\tau$ float elements | 지연 bin별 평균 전력 분포 | embedding → feature $\mathbf{c}$ → FiLM 조건 주입 |

- 위치당 채널 실현 200회를 축적하여 샘플 공분산/PDP 추정
- 장기 통계이므로 보고 주기가 길어($T_{\text{cov}} \gg T_{\text{CSI}}$) 시간 평균 오버헤드가 작음

### 3-3. 데이터셋 분할 및 규모

| 구분 | 위치 수 | 위치당 실현 | 용도 | 비고 |
|------|:-------:|:----------:|------|------|
| 통계 추출용 | 1,000 | 200 | $\mathbf{R}_H$, PDP 추정 | Stage 1 학습 + Stage 2 conditioning |
| 학습 (train) | 1,000 | 100 | Stage 1 + Stage 2 autoencoder 학습 | 통계와 다른 독립 실현 |
| 검증 (val) | 1,000 | 10 | 조기 종료, 하이퍼파라미터 선택 | |
| 테스트 (test) | 1,000 | 10 | 최종 성능 평가 (NMSE, cosine sim) | |
| **합계** | **1,000** | **320** | | **~128 GB** (전체) |


### 데이터 생성 역량

- Sionna GPU 가속 채널 생성 → 2D DFT 변환 → 지연축 절단 → NPZ 저장 **자동화 파이프라인** 완성
- 동일 파이프라인에서 위치별 공분산/PDP 통계 자동 추출
- 파라미터 변경만으로 시나리오(채널 모델, SNR, 속도, 안테나 구성) 확장 가능
- AWS 플랫폼 위에서도 동일 파이프라인 구동 → **생성 → 통계 추출 → 학습 → 평가** 사이클 일괄 운용 가능

---
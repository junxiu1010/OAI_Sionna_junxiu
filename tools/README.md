# Type-I vs Type-II PMI 검증 도구

OAI UE에서 추정한 채널(H)과 PMI로 재구성한 프리코딩 벡터(W) 간의
정규화 상관관계(normalized correlation)를 측정하여 Type-II PMI 구현의
정확성을 검증합니다.

## 구성 파일

```
tools/
├── README.md                  # 이 문서
├── verify_type2_pmi.py        # 바이너리 로그 분석 및 시각화
├── sweep_snr.sh               # SNR 스윕 자동화 스크립트
└── results/                   # 결과 바이너리 로그 및 그래프 저장
```

## 동작 원리

```
[OAI UE - csi_rx.c]                    [Python 후처리]
                                       
 CSI-RS 수신                            바이너리 로그 읽기
     │                                       │
 채널 추정 (H_est)──────┐              Type-I: i2 → W=[1, e^jφ, 0, 0]/√2
     │                  │              Type-II: port_sel + amp + phase → W
 PMI 계산 (Type-I/II)   │                    │
     │                  │              ρ = |h^H w| / (‖h‖ ‖w‖)
     ▼                  ▼              서브캐리어 × RX 안테나 평균
 바이너리 파일에 기록                         │
 (CSI_CHANNEL_LOG 환경변수)             SNR별 그래프 생성
```

## 사전 준비

1. **빌드**: `csi_rx.c`에 로깅 코드가 포함된 상태로 UE를 빌드합니다.

```bash
cd openairinterface5g_whan/cmake_targets/ran_build/build
ninja nr-uesoftmodem nr-softmodem
```

2. **Python 의존성**: `numpy`, `matplotlib` 필요 (시스템에 이미 설치됨).

## 사용법

### 1. 단일 실행으로 로그 수집

환경변수 `CSI_CHANNEL_LOG`에 파일 경로를 설정하면 UE가 CSI-RS 처리 시마다
채널 추정치와 PMI를 바이너리로 기록합니다.

```bash
# Type-II 설정 (gnb.conf에 codebook_type = "type2" 확인)
export CSI_CHANNEL_LOG=/home/dclserver78/oai_sionna_junxiu/tools/results/type2_snr20.bin

# 시스템 실행 (SNR=20dB, 60초)
sudo -E bash vRAN_Socket/G1B_SingleUE_MIMO_Channel_Proxy/launch_all.sh \
    -v v8 -ga 2 2 -ua 2 1 -snr 20 -d 60
```

> **주의**: `sudo -E`로 실행해야 환경변수가 자식 프로세스에 전달됩니다.

### 2. 결과 분석

```bash
python3 tools/verify_type2_pmi.py tools/results/type2_snr20.bin
```

출력:
- 터미널에 통계 테이블 (평균 ρ, 표준편차, 레코드 수)
- `tools/results/pmi_correlation.png`에 그래프 저장

출력 경로 변경:
```bash
python3 tools/verify_type2_pmi.py tools/results/type2_snr20.bin -o tools/results/my_plot.png
```

### 3. SNR 스윕 (Type-I vs Type-II 비교)

여러 SNR에서 Type-I과 Type-II를 자동으로 실행하고 로그를 수집합니다.

```bash
sudo -E bash tools/sweep_snr.sh
```

기본값:
- SNR: 0, 5, 10, 15, 20, 25, 30 dB
- 코드북: type1, type2 (둘 다)
- 실행 시간: SNR당 60초
- 출력: `tools/results/type1_snr*.bin`, `tools/results/type2_snr*.bin`

옵션:
```bash
# SNR 범위와 실행 시간 지정
sudo -E bash tools/sweep_snr.sh -s "10 20 30" -d 30

# Type-II만 실행
sudo -E bash tools/sweep_snr.sh --skip-type1

# 출력 디렉토리 지정
sudo -E bash tools/sweep_snr.sh -o /tmp/my_results
```

스윕 완료 후 분석:
```bash
python3 tools/verify_type2_pmi.py tools/results/type1_snr*.bin tools/results/type2_snr*.bin
```

## 출력 그래프 설명

`pmi_correlation.png`에 3개의 서브플롯이 생성됩니다:

| 그래프 | x축 | y축 | 설명 |
|--------|------|------|------|
| Correlation vs SNR | SNR (dB) | 평균 ρ (0~1) | Type-I / Type-II 곡선 비교 |
| CDF | ρ 값 | 누적확률 | 선택된 SNR에서의 분포 |
| Improvement | SNR (dB) | 개선율 (%) | (ρ_type2 - ρ_type1) / ρ_type1 × 100 |

**기대 결과**: Type-II의 ρ가 Type-I보다 높아야 하며, 특히 중간~높은 SNR
(10~25 dB)에서 차이가 뚜렷해야 합니다.

## 바이너리 로그 포맷

각 CSI-RS 이벤트마다 하나의 레코드가 기록됩니다:

```
[64바이트 헤더]
  오프셋  타입       필드
  0       uint32     매직넘버 (0x43534932 = "CSI2")
  4       uint32     레코드 총 크기
  8       uint32     프레임 번호
  12      uint32     슬롯 번호
  16      uint16     CSI-RS 포트 수 (N_ports)
  18      uint16     수신 안테나 수 (N_rx)
  20      uint32     OFDM 심볼 크기
  24      uint32     첫 번째 캐리어 오프셋
  28      uint16     시작 RB
  30      uint16     RB 수
  32      uint8      메모리 오프셋
  33      uint8      랭크 인디케이터
  34      uint8      CQI
  35      uint8      PMI 타입 (0=Type-I, 1=Type-II)
  36      uint8      i1 (Type-I)
  37      uint8      i2 (Type-I)
  38      uint8      포트 선택 인디케이터 (Type-II)
  39-40   uint8[2]   최강 계수 인디케이터 (Type-II)
  41      uint8      빔 수 L (Type-II)
  42      uint8      포트 선택 d (Type-II)
  43      uint8      위상 알파벳 (Type-II)
  44-51   uint8[8]   WB 진폭 레이어 0 (Type-II)
  52-59   uint8[8]   WB 진폭 레이어 1 (Type-II)
  60      uint8      서브밴드 수 (Type-II)
  61-63   uint8[3]   패딩

[채널 데이터]
  c16_t H_est[N_rx][N_ports][ofdm_size]
  크기: N_rx × N_ports × ofdm_size × 4 바이트

[서브밴드 위상] (Type-II, num_subbands > 0인 경우)
  uint8 phases[num_subbands][2*L][num_layers]
```

## 유의사항

- OAI의 Type-I PMI는 2포트 코드북만 지원합니다 (i2 ∈ {0,1,2,3}).
  4포트 구성에서도 포트 0, 1만 사용합니다.
- Type-II는 4포트 전체를 사용하므로, 비교 결과에는 코드북 해상도와
  포트 수 차이가 모두 반영됩니다.
- 순수 코드북 해상도만 비교하려면 `gnb.conf`에서 `nrof_ports = 2`로
  설정하여 둘 다 2포트로 실행하면 됩니다.
- 데이터 크기: 약 65KB/레코드 × 12.5 events/sec ≈ 800KB/s

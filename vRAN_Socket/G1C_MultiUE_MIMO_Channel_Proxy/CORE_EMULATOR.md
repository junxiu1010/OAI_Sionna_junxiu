# Core Emulator 설명서

## 1. 개요

Core Emulator는 OAI 5G 시뮬레이터의 모든 PHY/채널/안테나 설정을
하나의 YAML 파일(`master_config.yaml`)로 통합 관리하는 중앙 설정 서버이다.

기존에는 설정이 3곳에 분산·하드코딩되어 있었다:

| 기존 위치 | 내용 |
|-----------|------|
| `gnb.conf` | gNB PHY/MAC/RRC (코드북, CSI-RS, SRS, 안테나 포트 등) |
| `launch_all.sh` | CLI 옵션, 안테나 계산, OAI 인자 조합 |
| `v0.py` | Proxy 글로벌 상수 + argparse 기본값 |

파라미터 변경 시 3개 파일을 수동으로 일관성 있게 수정해야 했으며,
런타임 변경은 불가능했다.

Core Emulator 도입 후:

```
master_config.yaml  ─→  Core Emulator 서버  ─→  launch_all.sh (시작 시)
                                              ─→  gnb.conf (템플릿 렌더링)
                                              ─→  v0.py Proxy (초기 설정 + 런타임 핫스왑)
```

## 2. 아키텍처

```
┌─────────────────────────────────────────────────┐
│              Core Emulator (Python)              │
│                                                  │
│  master_config.yaml ──→ 설정 관리 (CoreState)    │
│                         │                        │
│  templates/gnb.conf.j2 ─┤  Jinja2 렌더링 엔진    │
│                         │                        │
│  TCP/JSON (port 7100) ──┤  메시지 라우터          │
│  HTTP GET (port 7101) ──┘  (curl 호환)           │
└────────┬──────────┬──────────┬───────────────────┘
         │          │          │
    시작 시 1회    시작 시 1회   런타임
         │          │          │
         ▼          ▼          ▼
   launch_all.sh  gnb.conf   v0.py Proxy
   (파라미터 수신) (생성)     (핫스왑 수신)
```

### 2.1 두 가지 통신 인터페이스

| 인터페이스 | 포트 | 용도 |
|-----------|------|------|
| **TCP/JSON** | 7100 | CLI 클라이언트, Proxy 리스너, 양방향 명령/응답 |
| **HTTP GET** | 7101 | `launch_all.sh`에서 `curl`로 설정 가져오기 |

TCP/JSON 프로토콜:
```
요청: {"cmd": "GET_CONFIG"}  + 줄바꿈(\n)
응답: {"status": "ok", "config": {...}}  + 줄바꿈(\n)
```

## 3. 파일 구조

```
G1C_MultiUE_MIMO_Channel_Proxy/
├── core_emulator.py       # Core Emulator 서버
├── core_cli.py            # CLI 클라이언트
├── master_config.yaml     # 마스터 설정 파일 (유일한 설정 원본)
├── templates/
│   └── gnb.conf.j2        # gnb.conf Jinja2 템플릿
├── v0.py                  # Proxy (--core-emulator 옵션 추가)
└── launch_all.sh          # 통합 런처 (-c 옵션 추가)
```

## 4. master_config.yaml 구조

모든 설정을 8개 섹션으로 구분한다:

### 4.1 `system` — 시스템 전역 설정

```yaml
system:
  num_ues: 4               # UE 수
  mode: gpu-ipc            # gpu-ipc | socket
  channel_mode: static     # dynamic | static
  batch_size: 4            # UE 배치 기동 크기
```

### 4.2 `antenna` — 안테나 배열

```yaml
antenna:
  gnb:
    nx: 2                  # 수평 안테나 수
    ny: 1                  # 수직 안테나 수
    polarization: dual     # single (V, XP=1) | dual (cross ±45°, XP=2)
  ue:
    nx: 2
    ny: 1
```

**자동 파생값** (Core Emulator가 내부에서 계산):

| 파생값 | 계산식 | 예시 (dual, 2x1) |
|--------|--------|-------------------|
| `pol_mult` | dual→2, single→1 | 2 |
| `gnb_spatial` | nx × ny | 2 |
| `gnb_ant` | gnb_spatial × pol_mult | 4 |
| `ue_ant` | ue_nx × ue_ny × pol_mult | 4 |
| `xp` | pol_mult | 2 |
| `n1` | xp==2 → gnb_spatial, else gnb_ant | 2 |

이 파생값들이 `gnb.conf`의 `pdsch_AntennaPorts_N1`, `pdsch_AntennaPorts_XP`,
`nb_tx`, `nb_rx` 등에 자동 반영된다.

### 4.3 `codebook` — 코드북 설정

```yaml
codebook:
  type: type2                        # type1 | type2
  sub_type: typeII_PortSelection     # typeI_SinglePanel | typeII_PortSelection
  mode: 1
  pmi_restriction: 0xff
  ri_restriction: 0x03
  n1_n2_config: two_one
  # Type-II 전용
  phase_alphabet_size: n4            # n4 (QPSK) | n8 (8PSK)
  subband_amplitude: 0
  number_of_beams: 2
  port_selection_sampling_size: 2
```

`type` 값에 따라 `gnb.conf` 템플릿에서 `codebook_detailed_config` 블록이
Type-I 또는 Type-II로 분기 렌더링된다.

### 4.4 `csi_rs` — CSI-RS 설정

```yaml
csi_rs:
  periodicity: 80          # 슬롯 주기
  nrof_ports: 4            # CSI-RS 포트 수
  # ... (기타 세부 파라미터)
```

### 4.5 `srs` — SRS 설정

```yaml
srs:
  periodicity: 20          # 슬롯 주기
  nrof_srs_ports: 1        # SRS 포트 수
  # ...
```

### 4.6 `csi_measurement` — CSI 측정 보고 설정

```yaml
csi_measurement:
  report_periodicity: 80   # 보고 주기 (슬롯)
  cqi_table: table1
  enable_ri: 1
  enable_pmi: 1
  # ...
```

### 4.7 `channel` — Sionna 채널 모델 파라미터

```yaml
channel:
  path_loss_dB: 0.0        # 경로 손실
  snr_dB: null             # 상대 SNR (null=비활성)
  noise_dBFS: null         # 절대 노이즈 플로어 (null=비활성)
  speed: 3.0               # UE 이동 속도 (m/s)
  scenario: UMa-NLOS       # 3GPP 채널 시나리오
```

### 4.8 `carrier` — 캐리어 설정

```yaml
carrier:
  frequency_GHz: 3.5
  bandwidth_prb: 106
  scs_kHz: 30
  band: 78
```

## 5. gnb.conf 템플릿 엔진

`templates/gnb.conf.j2`는 현재 `gnb.conf` 파일을 Jinja2 템플릿으로 변환한 것이다.

### 5.1 원리

1. Core Emulator가 `master_config.yaml`을 로드
2. 파생값 계산 (`_derived()` 함수)
3. YAML 설정 + 파생값을 Jinja2 컨텍스트에 주입
4. `gnb.conf.j2` 렌더링 → 완성된 `gnb.conf` 텍스트 출력

### 5.2 주요 템플릿 변수

| 템플릿 변수 | 설정 원본 | gnb.conf 위치 |
|-------------|----------|---------------|
| `{{ n1 }}` | 자동 계산 | `pdsch_AntennaPorts_N1` |
| `{{ xp }}` | 자동 계산 | `pdsch_AntennaPorts_XP` |
| `{{ ue_ant }}` | 자동 계산 | `pusch_AntennaPorts` |
| `{{ gnb_ant }}` | 자동 계산 | `RUs.nb_tx` |
| `{{ csi_rs.periodicity }}` | YAML | `csirs_detailed_config.periodicity` |
| `{{ codebook.type }}` | YAML | `codebook_detailed_config` 분기 |
| `{{ scs_idx }}` | 자동 (SCS→인덱스) | `dl_subcarrierSpacing` 등 |

### 5.3 코드북 분기 예시

```jinja2
{% if codebook.type == "type1" %}
    codebook_type = "type1";
    sub_type = "{{ codebook.sub_type }}";
    ...
{% elif codebook.type == "type2" %}
    codebook_type = "type2";
    ...
    phase_alphabet_size = "{{ codebook.phase_alphabet_size }}";
    number_of_beams = {{ codebook.number_of_beams }};
{% endif %}
```

## 6. TCP/JSON 명령어

| 명령어 | 용도 | 응답 키 |
|--------|------|---------|
| `GET_CONFIG` | 전체 설정 반환 | `config` |
| `GET_GNB_CONF` | gnb.conf 텍스트 렌더링 | `gnb_conf` |
| `GET_LAUNCH_PARAMS` | launch_all.sh용 파라미터 | `params` |
| `GET_PROXY_PARAMS` | v0.py Proxy용 파라미터 | `params` |
| `UPDATE_PROXY` | Proxy 핫스왑 (런타임) | `proxy_params` |
| `UPDATE_GNB` | gNB 재설정 (재시작 필요) | `gnb_conf` |
| `SUBSCRIBE_PROXY` | Proxy 업데이트 구독 (push) | 이벤트 스트림 |
| `STATUS` | 서버 상태 조회 | `version`, `uptime_s` 등 |

## 7. HTTP 엔드포인트

| 경로 | 용도 |
|------|------|
| `GET /config` | 전체 설정 (JSON) |
| `GET /gnb_conf` | 렌더링된 gnb.conf (text/plain) |
| `GET /launch_params` | 런치 파라미터 (JSON) |
| `GET /proxy_params` | 프록시 파라미터 (JSON) |
| `GET /status` | 서버 상태 (JSON) |

## 8. 사용법

### 8.1 기본 실행 흐름

```bash
# 1단계: 설정 편집
vim master_config.yaml

# 2단계: Core Emulator 서버 시작
python3 core_emulator.py --config master_config.yaml

# 3단계: 시스템 기동 (Core Emulator 연동)
sudo bash launch_all.sh -c localhost:7100
```

`launch_all.sh -c` 실행 시 내부 동작:
1. `curl http://localhost:7101/launch_params` → 전체 파라미터 수신
2. `curl http://localhost:7101/gnb_conf` → gnb.conf 생성 (`/tmp/generated_gnb.conf`)
3. 수신된 파라미터로 gNB/UE/Proxy 기동
4. Proxy에 `--core-emulator=localhost:7100` 전달

### 8.2 gnb.conf만 생성 (서버 없이)

```bash
python3 core_emulator.py --render-gnb-conf /tmp/my_gnb.conf
```

### 8.3 CLI 클라이언트 사용

```bash
# 전체 설정 확인
python3 core_cli.py config

# 서버 상태
python3 core_cli.py status

# 특정 섹션 확인
python3 core_cli.py get antenna
python3 core_cli.py get codebook
python3 core_cli.py get channel

# 런치 파라미터 (bash eval 형식)
python3 core_cli.py launch-params --shell

# gnb.conf 렌더링 및 파일 저장
python3 core_cli.py gnb-conf -o /tmp/gnb.conf
```

### 8.4 런타임 파라미터 변경

#### 핫스왑 (Proxy 즉시 반영, gNB 재시작 불필요)

```bash
# 경로 손실 변경
python3 core_cli.py update channel.path_loss_dB=5.0

# SNR 변경
python3 core_cli.py update channel.snr_dB=20

# 노이즈 플로어 변경
python3 core_cli.py update channel.noise_dBFS=-40

# 여러 파라미터 동시 변경
python3 core_cli.py update channel.path_loss_dB=3.0 channel.speed=10
```

핫스왑 가능 파라미터 목록:

| 파라미터 | 설명 |
|----------|------|
| `channel.path_loss_dB` | 경로 손실 (dB) |
| `channel.snr_dB` | 상대 AWGN SNR (dB) |
| `channel.noise_dBFS` | 절대 노이즈 플로어 (dBFS) |
| `channel.speed` | UE 이동 속도 (m/s) |

#### gNB 재설정 (managed restart 필요)

```bash
# 코드북 변경: Type-II → Type-I
python3 core_cli.py update --restart-gnb \
    codebook.type=type1 \
    codebook.sub_type=typeI_SinglePanel

# 편파 변경: dual → single
python3 core_cli.py update --restart-gnb \
    antenna.gnb.polarization=single

# gnb.conf 저장 + 재시작 안내
python3 core_cli.py update --restart-gnb \
    codebook.type=type2 \
    --save-conf /tmp/new_gnb.conf
```

gNB 재설정이 필요한 파라미터:

| 파라미터 | 설명 |
|----------|------|
| `antenna.gnb.nx/ny` | gNB 안테나 배열 |
| `antenna.gnb.polarization` | 편파 모드 |
| `codebook.type` | 코드북 타입 (Type-I/II) |
| `codebook.*` | 코드북 세부 설정 |
| `csi_rs.periodicity` | CSI-RS 주기 |
| `srs.periodicity` | SRS 주기 |
| `csi_measurement.report_periodicity` | CSI 보고 주기 |

## 9. 핫스왑 원리

```
core_cli.py update channel.snr_dB=20
        │
        ▼
  Core Emulator (TCP JSON)
    1. master_config 내부 상태 업데이트
    2. PROXY_UPDATE 이벤트 생성
    3. 구독된 모든 Proxy 클라이언트에 push
        │
        ▼
  v0.py Proxy (백그라운드 리스너 스레드)
    1. PROXY_UPDATE 이벤트 수신
    2. _apply_proxy_hotswap() 호출
    3. 글로벌 변수 atomic 업데이트:
       path_loss_dB, pathLossLinear,
       snr_dB, noise_mode, noise_dBFS 등
    4. 다음 슬롯부터 새 값 적용
```

Proxy는 시작 시 `--core-emulator` 옵션이 있으면:
1. `GET_PROXY_PARAMS` 명령으로 초기 설정 수신 (CLI 인자 오버라이드)
2. 백그라운드 `_core_emulator_listener` 스레드 시작
3. `SUBSCRIBE_PROXY` 명령으로 이벤트 구독
4. 연결 끊김 시 5초 후 자동 재접속

## 10. gNB Managed Restart 원리

```
core_cli.py update --restart-gnb codebook.type=type1
        │
        ▼
  Core Emulator
    1. master_config 업데이트
    2. 새 gnb.conf 렌더링
    3. gnb_restart_requested = True
        │
        ▼
  core_cli.py (클라이언트 측)
    1. 새 gnb.conf 파일 저장
    2. gNB + UE 프로세스 SIGKILL
    3. IPC 버퍼 (/tmp/oai_gpu_ipc/*) 삭제
    4. 재시작 명령 안내 출력
```

Proxy 프로세스는 종료하지 않고 유지된다.
gNB/UE만 재기동하면 Proxy가 다시 연결을 수락한다.

## 11. 설정 구성 예시

### 11.1 4T4R Dual-Pol Type-II (기본값)

```yaml
antenna:
  gnb: { nx: 2, ny: 1, polarization: dual }
  ue:  { nx: 2, ny: 1 }
codebook:
  type: type2
  sub_type: typeII_PortSelection
```

결과: `N1=2, N2=1, XP=2 → 4포트, nb_tx=4, nb_rx=4`

### 11.2 2T2R Single-Pol Type-I

```yaml
antenna:
  gnb: { nx: 2, ny: 1, polarization: single }
  ue:  { nx: 2, ny: 1 }
codebook:
  type: type1
  sub_type: typeI_SinglePanel
```

결과: `N1=2, N2=1, XP=1 → 2포트, nb_tx=2, nb_rx=2`

### 11.3 8T8R Dual-Pol (4x1 배열)

```yaml
antenna:
  gnb: { nx: 4, ny: 1, polarization: dual }
  ue:  { nx: 4, ny: 1 }
codebook:
  type: type2
  n1_n2_config: four_one
```

결과: `N1=4, N2=1, XP=2 → 8포트, nb_tx=8, nb_rx=8`

### 11.4 32 UE Static Channel

```yaml
system:
  num_ues: 32
  channel_mode: static
  batch_size: 2
csi_rs:
  periodicity: 80
srs:
  periodicity: 20
csi_measurement:
  report_periodicity: 80
```

## 12. 하위 호환

Core Emulator 없이도 기존 방식대로 동작한다:

```bash
# 기존 방식 (변경 없음)
sudo bash launch_all.sh -n 4 -ga 2 1 -ua 2 1 -pol dual -cm static

# Core Emulator 연동 방식 (신규)
sudo bash launch_all.sh -c localhost:7100
```

`-c` 옵션이 없으면 CLI 인자가 직접 사용되며,
`v0.py`에서 `--core-emulator`가 없으면 기존 동작과 동일하다.

## 13. 의존성

| 패키지 | 용도 |
|--------|------|
| `pyyaml` | YAML 설정 파일 파싱 |
| `jinja2` | gnb.conf 템플릿 렌더링 |

설치:
```bash
pip install pyyaml jinja2
```

# OAI 수정 요약 (Proxy 관점)

> 이 문서는 G1C Proxy와 연동하기 위해 OAI 소스코드(`openairinterface5g_whan/`)에 적용한 수정 사항의 **요약**입니다.
> 알고리즘 상세, 코드 diff, 빌드 이슈 등은 원본을 참조하세요.
>
> **원본**: [`openairinterface5g_whan/MODIFICATION_LOG.md`](../../../openairinterface5g_whan/MODIFICATION_LOG.md)
>
> [README.md](../README.md)로 돌아가기 · [CHANGELOG.md](CHANGELOG.md) · [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 수정 이력 (최신순)

### 1. IPC Read Timeout 확장 (2026-04-06)

| 항목 | 내용 |
|------|------|
| **파일** | `radio/rfsimulator/simulator.c` |
| **변경** | `gpu_ipc_v7_dl_read` / `gpu_ipc_v7_ul_read`의 timeout: **30ms → 5000ms** |
| **이유** | 8 UE 4T4R에서 per-slot proxy 처리 시간(15~20ms)이 30ms timeout을 초과 → IQ 버퍼 0 채움 → PBCH 디코딩 실패. 시뮬레이션 환경에서 proxy가 느려도 OAI가 충분히 기다려야 함 |
| **빌드** | `ninja rfsimulator` (`librfsimulator.so` 재빌드) |
| **검증** | 8 UE 접속 0/8 → **6/8** 성공 ([EXPERIMENTS.md 우선2-d](EXPERIMENTS.md)) |

### 2. GPU 1 전환 (2026-04-06)

| 항목 | 내용 |
|------|------|
| **파일** | `radio/rfsimulator/gpu_ipc_v7.c` |
| **변경** | `cudaSetDevice(0)` → `cudaSetDevice(1)` |
| **이유** | GPU 0에 좀비 프로세스(~15 GB) 상주, VRAM 실측 정확도 확보를 위해 GPU 1 사용 |
| **빌드** | `ninja rfsimulator` |
| **참고** | Proxy 측도 동시 변경: `v4.py` `gpu_num = 0` → `1` |

### 3. 4T4R Rank 4 RI/PMI 구현 (2026-04-03/05)

| 항목 | 내용 |
|------|------|
| **파일** | `openair1/PHY/NR_UE_TRANSPORT/csi_rx.c` |
| **신규 함수** | `nr_csi_rs_ri_estimation_4x4()` — progressive sub-matrix condition number 기반 RI 추정 (2x2→3x3→4x4) |
| | `nr_csi_rs_pmi_estimation_4port()` — 4-port Type I SinglePanel codebook 탐색, Rank 1~4 |
| | `cd_t` 복소수 double 헬퍼 (`cd_mul`, `cd_sub`, `cd_add`, `cd_mag2`, `cd_from_c16`) |
| **분기 수정** | `nr_csi_rs_ri_estimation()`: `nb_antennas_rx >= 4 && N_ports >= 4` → `_4x4()` dispatch |
| | `nr_csi_rs_pmi_estimation()`: `N_ports >= 4` → `_4port()` dispatch |
| **gNB 설정** | `nrof_ports` 2→4, `freq_allocation` row4, `ri_restriction` 0xF, `nrof_srs_ports` 2 |
| **빌드** | `./build_oai --ninja -w USRP --nrUE --gNB -c` (전체) 또는 `ninja nr-uesoftmodem` (UE만) |
| **검증** | Bypass: RI=4, PMI=[0,0,0], CQI=15. P1B 채널: RI=2↔4 동적, PMI 채널 적응 확인 ([EXPERIMENTS.md §7.5](EXPERIMENTS.md)) |

### 4. IPC V7 Futex (2026-03-26)

| 항목 | 내용 |
|------|------|
| **파일** | `radio/rfsimulator/gpu_ipc_v7.{h,c}` (신규), `simulator.c`, `CMakeLists.txt` |
| **변경** | V6의 `usleep(1000)` 폴링 → futex 기반 즉시 알림. seq counter 4개, SHM MAGIC `0x47505538` |
| **효과** | IPC 대기 시간 0.02~1.29ms → ≈0ms |
| **빌드** | `ninja rfsimulator nr-softmodem nr-uesoftmodem` |

### 5. IPC V6 Multi-Instance (2026-03-12)

| 항목 | 내용 |
|------|------|
| **파일** | `radio/rfsimulator/gpu_ipc_v6.{h,c}` (신규), `simulator.c`, `CMakeLists.txt` |
| **변경** | V5의 단일 SHM → per-UE SHM 인스턴스 (`gpu_ipc_shm_ue{k}`). Multi-UE 지원 |
| **빌드** | `ninja rfsimulator nr-softmodem nr-uesoftmodem` |

### 6. 이전 수정 (G1B/G1A 시기)

원본 `MODIFICATION_LOG.md`에 상세 기록. 주요 항목:

| # | 내용 | 시기 |
|:-:|------|------|
| 수정 1 | CSI-RS PMI 단일 포트 SINR 실측 계산 | 2026-02 |
| 수정 5 | OAI rfsimulator GPU IPC 모드 최초 추가 (V1) | 2026-02-24 |
| 수정 8 | TDD 데드락 수정 + 빌드 시스템 이슈 | 2026-02-24 |
| 수정 10 | GPU IPC V2 Multi-UE (G1A) | 2026-02-27 |
| 수정 11 | PRACH 크래시 수정 | 2026-02-27 |
| 수정 12 | RU-L1TX 갭 제한 (B-light 페이싱) | 2026-02-28 |
| V3 | Timestamp-Indexed Ring Buffer | 2026-03-04 |
| V4→V5 | Circular Buffer → Per-UE 분리 | 2026-03-06 |

---

## 빌드 가이드 (빠른 참조)

```bash
cd openairinterface5g_whan/cmake_targets/ran_build/build

# simulator.c / gpu_ipc_v7.c 수정 시 (IPC 관련)
ninja rfsimulator        # librfsimulator.so 재빌드 (필수!)

# csi_rx.c 수정 시 (UE PHY)
ninja nr-uesoftmodem

# 전체 재빌드
cd openairinterface5g_whan/cmake_targets
./build_oai --ninja -w USRP --nrUE --gNB -c
```

> **주의**: `simulator.c` 수정 후 `ninja nr-softmodem`만 하면 반영되지 않음. 반드시 `ninja rfsimulator`로 `librfsimulator.so`를 재빌드해야 함.

---

## IPC 버전 히스토리

```
V1 (ring buffer, ready-flag)
  → V2 (Multi-UE, per-UE rings)               ← G1A
    → V3 (timestamp-indexed ring buffer)       ← G1A
      → V4→V5 (circular buffer)               ← G1B
        → V6 (V5 + Multi-Instance per-UE SHM) ← G1C v0
          → V7 (V6 + futex seq counter)        ← G1C v1~v4 (현재)
```

---

## 현재 활성 수정 파일 목록

| 파일 | 수정 내용 | 최종 수정 |
|------|----------|-----------|
| `openair1/PHY/NR_UE_TRANSPORT/csi_rx.c` | 4x4 RI + PMI Rank 1~4 | 2026-04-05 |
| `radio/rfsimulator/gpu_ipc_v7.{h,c}` | IPC V7 futex + GPU 1 전환 | 2026-04-06 |
| `radio/rfsimulator/gpu_ipc_v6.{h,c}` | IPC V6 Multi-Instance | 2026-03-12 |
| `radio/rfsimulator/simulator.c` | V7/V6/V5 dispatch + timeout 5000ms | 2026-04-06 |
| `radio/rfsimulator/CMakeLists.txt` | V7/V6/V5 빌드 정의 | 2026-03-26 |
| `targets/.../gnb.sa.band78.fr1.106PRB.usrpb210.conf` | 4-port CSI-RS, RI 1~4 허용 | 2026-04-03 |

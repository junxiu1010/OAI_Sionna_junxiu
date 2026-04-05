# G2: Socket 제거 + GPU Throughput 극대화 계획

> **Socket → GPU 직접 접근 (CUDA IPC)** + **멀티스트림 파이프라인으로 Throughput 극대화**

## 1. 배경 및 필요성

### 현재 구조의 문제

현재 Sionna Channel Proxy는 OAI gNB/UE와 **TCP Socket**을 통해 IQ 데이터를 주고받습니다.

```
OAI gNB (CPU, 컨테이너 외부)
  ↓ IQ 생성 (CPU 메모리, int16)
  ↓ socket send (TCP, port 6017)
  ↓ --- 네트워크 스택 + syscall ---
Sionna Proxy (컨테이너 내부)
  ↓ socket recv (CPU 메모리)
  ↓ memcpy → pinned memory
  ↓ H2D (cudaMemcpy, 120KB)
  ↓ GPU 채널 처리 (FFT conv, PL, AWGN 등)
  ↓ D2H (cudaMemcpy, 120KB)
  ↓ socket send (TCP, port 6018)
  ↓ --- 네트워크 스택 + syscall ---
OAI UE (CPU, 컨테이너 외부)
  ↓ socket recv
  ↓ IQ 소비 (CPU 메모리)
```

### 왜 Socket을 제거해야 하는가

1. **대용량 데이터 확장 시 Socket이 병목**
   - 현재: 슬롯당 120KB (1x1 SISO, 30.72MHz 샘플링)
   - MIMO 4x4: 480KB/슬롯
   - MIMO 8x8: 960KB/슬롯
   - 200MHz 대역폭: 2MB+/슬롯
   - TCP Socket은 커널 네트워크 스택(syscall, 버퍼 복사, TCP 프로토콜 처리)을 거치므로 **PCIe 대역폭을 활용하지 못함**

2. **불필요한 데이터 경로**
   - 현재: CPU → socket → CPU → pinned → GPU (3~4회 복사)
   - 이상적: CPU → GPU 직접 (1회 복사, H2D만)
   - Socket은 같은 머신 내 통신인데 TCP 프로토콜 오버헤드를 부담함

3. **향후 Aerial 전환 대비**
   - gNB는 NVIDIA Aerial(GPU 기반 PHY)로 대체 예정
   - UE는 OAI 유지하며 Sionna Proxy와 연결
   - Socket 제거 구조가 Aerial 연동의 기반이 됨

### v11 프로파일 측정 결과 (H100 NVL 기준)

현재 Socket 비중 추정 (안정 구간):

| 구간 | 소요 시간 | 비중 |
|------|:---------:|:----:|
| GPU 처리 (GPU_PROC) | ~1.374 ms | ~70% |
| Socket RX + 기타 오버헤드 | ~0.40 ms | ~20% |
| Socket SEND | ~0.17 ms | ~9% |
| **전체 (PROXY_E2E)** | **~1.95 ms** | **100%** |

현재 120KB 기준으로는 ~20% 수준이지만, **데이터 크기가 4~8배 증가하면 Socket 비중이 급격히 증가**할 것으로 예상됨.

---

## 2. 현재 시스템 구성 (참고 정보)

### 폴더 구조

```
DevChannelProxyJIN/
├── openairinterface5g_whan/          # OAI 소스코드 (수정된 버전)
│   ├── radio/rfsimulator/            # ★ 수정 대상: rfsimulator
│   │   ├── simulator.c               # 핵심 파일 (Socket TX/RX 구현)
│   │   ├── rfsimulator.h
│   │   ├── apply_channelmod.c
│   │   ├── mock_channel.c
│   │   └── CMakeLists.txt
│   └── cmake_targets/ran_build/build/ # 빌드 출력
├── vRAN_Socket/
│   └── G0_Sionna_Channel_Proxy/      # Sionna Proxy 구현
│       ├── v10_cuda_graph.py          # 현재 최적 버전 (CUDA Graph)
│       ├── v11_profile_instrumented.py # 계측 고도화 버전
│       └── README.md                  # 상세 성능 분석
└── 이 문서 (G2_Socket제거_GPU직접접근_계획.md)
```

### OAI rfsimulator 핵심 함수 (simulator.c)

Socket을 제거하려면 아래 함수들을 수정해야 합니다:

| 함수 | 역할 | 수정 필요 |
|------|------|:---------:|
| `rfsimulator_write_internal()` | IQ 데이터를 Socket으로 전송 (TX) | ★ 핵심 |
| `flushInput()` | Socket에서 IQ 데이터 수신 (RX) | ★ 핵심 |
| `rfsimulator_read()` | 수신 데이터를 OAI에 전달 | ★ 핵심 |
| `allocCirBuf()` | Circular buffer + Socket 초기화 | 수정 |
| `startServer()` / `startClient()` | TCP 서버/클라이언트 설정 | 대체 |
| `device_init()` | rfsimulator 초기화 | 수정 |

#### rfsimulator_write_internal() — TX 경로 (현재)

```c
// simulator.c:689-808
// OAI가 IQ 데이터를 전송하는 핵심 함수
static int rfsimulator_write_internal(...) {
    for (int i = 0; i < MAX_FD_RFSIMU; i++) {
        buffer_t *b = &t->buf[i];
        if (b->conn_sock >= 0) {
            // 1. 헤더 전송 (타임스탬프, 샘플 수, 안테나 수)
            samplesBlockHeader_t header = {nsamps, nbAnt, timestamp};
            fullwrite(b->conn_sock, &header, sizeof(header), t);
            
            // 2. IQ 데이터 전송 (int16, socket write)
            fullwrite(b->conn_sock, samplesVoid[0], sampleToByte(nsamps, nbAnt), t);
        }
    }
}
```

#### flushInput() — RX 경로 (현재)

```c
// simulator.c:816-1001
// Socket에서 IQ 데이터를 수신하여 circular buffer에 저장
static bool flushInput(...) {
    // epoll_wait로 소켓 이벤트 감시
    int nfds = epoll_wait(t->epollfd, events, MAX_FD_RFSIMU, timeout);
    
    for (int nbEv = 0; nbEv < nfds; ++nbEv) {
        // recv()로 데이터 수신
        ssize_t sz = recv(b->conn_sock, b->transferPtr, blockSz, MSG_DONTWAIT);
        // circular buffer에 저장
    }
}
```

### Sionna Proxy 핵심 경로 (v10_cuda_graph.py)

```python
# Proxy가 Socket에서 IQ를 받아 GPU로 보내는 경로
def process_slot(self, iq_bytes, channels_gpu, ...):
    # 1. iq_bytes = Socket에서 받은 raw bytes
    # 2. pinned memory에 복사
    ctypes.memmove(self.pinned_iq_in.ctypes.data, iq_bytes, len(iq_bytes))
    # 3. H2D 전송
    self.gpu_iq_in.set(self.pinned_iq_in, stream=self.stream)
    # 4. GPU 채널 처리 (CUDA Graph)
    # 5. D2H
    self.gpu_iq_out.get(out=self.pinned_iq_out, stream=self.stream)
    # 6. 결과를 bytes로 변환 → Socket으로 전송
```

---

## 3. 목표 구조

### 핵심 아이디어

OAI가 IQ 데이터를 Socket으로 보내는 대신, **GPU 메모리에 직접 올림 (H2D)**.
Proxy가 그 GPU 메모리를 **CUDA IPC로 직접 읽어서** 채널 처리.
결과도 GPU 메모리에 저장하고, OAI가 **D2H로 직접 가져감**.

```
OAI gNB (CPU)
  ↓ IQ 생성 (CPU 메모리, int16)
  ↓ cudaMemcpy H2D → 공유 GPU 메모리 (gpu_buf_tx)
  ↓ 동기화 신호 (eventfd / semaphore)
  ↓
Sionna Proxy (같은 GPU)
  ↓ CUDA IPC로 gpu_buf_tx 직접 접근 (복사 없음)
  ↓ GPU 채널 처리 (FFT conv, PL, AWGN 등)
  ↓ 결과를 gpu_buf_rx에 저장
  ↓ 동기화 신호
  ↓
OAI UE (CPU)
  ↓ CUDA IPC로 gpu_buf_rx 접근
  ↓ cudaMemcpy D2H → CPU 메모리
  ↓ IQ 소비
```

### 성능 비교 (예상)

| 항목 | Socket 방식 (현재) | GPU 직접 접근 (목표) |
|------|:------------------:|:-------------------:|
| 데이터 복사 횟수 | 3~4회 | 1회 (H2D만) |
| 커널 syscall | send/recv 매번 | 없음 |
| PCIe 대역폭 활용 | 제한적 | DMA 직접 |
| 120KB 지연 | ~0.4 ms | ~0.1 ms |
| 1MB 지연 (예상) | ~2+ ms | ~0.3 ms |
| 대용량 확장성 | 병목 | PCIe 한계까지 OK |

---

## 4. 기술 요소

### CUDA IPC (Inter-Process Communication)

두 프로세스가 같은 GPU의 메모리를 공유하는 CUDA 기능.

```c
// 프로세스 A (OAI): GPU 메모리 할당 + handle 생성
cudaMalloc(&gpu_buf, size);
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, gpu_buf);
// handle을 프로세스 B에 전달 (파일, pipe 등)

// 프로세스 B (Proxy): handle로 같은 GPU 메모리에 접근
void *gpu_buf;
cudaIpcOpenMemHandle(&gpu_buf, handle, cudaIpcMemLazyEnablePeerAccess);
// gpu_buf를 직접 사용 (별도 복사 없음)
```

CuPy(Python)에서도 CUDA IPC 지원:
```python
import cupy as cp
# handle (64바이트)을 받아서 GPU 메모리 접근
ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)
gpu_array = cp.ndarray(shape, dtype, cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size, None), 0))
```

### 동기화 메커니즘

OAI(writer)와 Proxy(reader)가 데이터 준비 완료를 알리는 방법:

| 방법 | 장점 | 단점 |
|------|------|------|
| eventfd | 가볍고 빠름, Linux 표준 | 컨테이너 공유 필요 |
| POSIX semaphore | 프로세스 간 공유 용이 | 약간의 syscall 오버헤드 |
| CUDA Event + polling | GPU 동기화와 통합 | CPU busy-wait |
| shared memory flag | 매우 빠름 (메모리 접근만) | spin-lock 필요 |

**권장**: shared memory flag (spin-lock) + CUDA Event 조합

### Docker 컨테이너 고려사항

현재 Proxy는 Docker 컨테이너 내부에서 실행됨:

| 요구사항 | 방법 |
|----------|------|
| GPU 접근 | `--gpus all` (이미 사용 중) |
| CUDA IPC | 같은 GPU 접근 시 자동 동작 |
| IPC handle 교환 | volume mount된 파일 사용 |
| 동기화 | shared memory (`--ipc=host` 또는 명시적 공유) |

Docker 실행 시 추가 옵션:
```bash
docker run --gpus all --ipc=host ...
```

---

## 5. 구현 계획

### Phase 1: OAI rfsimulator 수정 (2~3주)

#### 1-1. CUDA 빌드 환경 추가

**파일**: `openairinterface5g_whan/radio/rfsimulator/CMakeLists.txt`

- CUDA 라이브러리 링크 추가 (`cudart`)
- CUDA 헤더 include 경로 추가
- OAI 빌드 시스템에 CUDA 의존성 등록

#### 1-2. GPU 공유 메모리 구조체 정의

**파일**: `openairinterface5g_whan/radio/rfsimulator/gpu_ipc.h` (신규)

```c
// GPU IPC 공유 구조체
typedef struct {
    cudaIpcMemHandle_t tx_handle;  // gNB TX → Proxy 입력
    cudaIpcMemHandle_t rx_handle;  // Proxy 출력 → UE RX
    volatile int tx_ready;         // 1 = 데이터 준비 완료
    volatile int rx_ready;         // 1 = 결과 준비 완료
    int nsamps;                    // 샘플 수
    int nbAnt;                     // 안테나 수
    uint64_t timestamp;            // 타임스탬프
    size_t data_size;              // 데이터 크기 (bytes)
} gpu_ipc_shared_t;
```

#### 1-3. rfsimulator_write_internal() 수정

**파일**: `openairinterface5g_whan/radio/rfsimulator/simulator.c`

```c
// 현재: Socket 전송
fullwrite(b->conn_sock, samplesVoid[0], sampleToByte(nsamps, nbAnt), t);

// 수정 후: GPU 메모리에 H2D
cudaMemcpy(gpu_buf_tx, samplesVoid[0], sampleToByte(nsamps, nbAnt), cudaMemcpyHostToDevice);
__sync_synchronize();  // memory fence
shared->tx_ready = 1;  // Proxy에 알림
```

#### 1-4. flushInput() / rfsimulator_read() 수정

```c
// 현재: Socket recv
ssize_t sz = recv(b->conn_sock, b->transferPtr, blockSz, MSG_DONTWAIT);

// 수정 후: GPU 메모리에서 D2H
while (!shared->rx_ready) { /* spin or yield */ }
cudaMemcpy(samplesVoid[0], gpu_buf_rx, sampleToByte(nsamps, nbAnt), cudaMemcpyDeviceToHost);
shared->rx_ready = 0;  // 플래그 리셋
```

#### 1-5. device_init() 수정

```c
// GPU IPC 초기화
cudaMalloc(&gpu_buf_tx, max_data_size);
cudaMalloc(&gpu_buf_rx, max_data_size);
cudaIpcGetMemHandle(&shared->tx_handle, gpu_buf_tx);
cudaIpcGetMemHandle(&shared->rx_handle, gpu_buf_rx);
// shared memory 파일에 handle 저장
```

### Phase 2: Sionna Proxy 수정 (1~2주)

#### 2-1. GPU IPC 수신 모드 추가

**파일**: `vRAN_Socket/G0_Sionna_Channel_Proxy/v12_gpu_ipc.py` (신규, v10 기반)

```python
class GPUIpcInterface:
    """CUDA IPC로 OAI와 GPU 메모리 직접 공유"""
    
    def __init__(self, handle_file):
        # IPC handle 파일에서 읽기
        handles = load_handles(handle_file)
        
        # GPU 메모리 직접 접근 (복사 없음)
        self.gpu_buf_tx = open_ipc_handle(handles['tx'])
        self.gpu_buf_rx = open_ipc_handle(handles['rx'])
    
    def read_iq(self):
        """OAI가 올린 IQ를 GPU에서 직접 읽기 (H2D 불필요)"""
        while not shared.tx_ready:
            pass  # spin
        # self.gpu_buf_tx가 이미 GPU에 있으므로 바로 사용
        return self.gpu_buf_tx
    
    def write_result(self, gpu_result):
        """처리 결과를 GPU 메모리에 저장 (D2H 불필요)"""
        cp.copyto(self.gpu_buf_rx, gpu_result)
        shared.rx_ready = 1
```

#### 2-2. GPUSlotPipeline 수정

```python
def process_slot_ipc(self, gpu_iq_in, channels_gpu, ...):
    """GPU IPC 모드: H2D/D2H 제거, GPU 메모리 직접 사용"""
    # gpu_iq_in은 이미 GPU에 있음 (OAI가 올림)
    # → H2D 단계 생략
    # → GPU 채널 처리만 수행
    # → 결과를 공유 GPU 메모리에 저장 (D2H 생략)
```

#### 2-3. Dual-mode 지원

```python
# CLI 옵션
--mode=socket    # 기존 Socket 모드 (호환성 유지)
--mode=gpu-ipc   # GPU IPC 모드 (신규)
```

### Phase 3: 통합 테스트 (1주)

1. OAI gNB + Proxy + OAI UE 전체 연결 테스트
2. PSS/SSS 동기화 정상 확인
3. RACH 성공 확인
4. 성능 프로파일 비교 (Socket vs GPU IPC)

### Phase 4: 대용량 데이터 검증 (1주)

1. MIMO 2x2 / 4x4 테스트
2. 대역폭 확장 시 Socket vs GPU IPC 비교
3. 확장성 한계 측정

---

## 6. 수정 파일 요약

### OAI 쪽 (C 코드)

| 파일 | 작업 | 난이도 |
|------|------|:------:|
| `radio/rfsimulator/simulator.c` | write/read를 GPU IPC로 변경 | ★★★ |
| `radio/rfsimulator/gpu_ipc.h` | 공유 구조체 정의 (신규) | ★ |
| `radio/rfsimulator/gpu_ipc.c` | IPC 초기화/정리 (신규) | ★★ |
| `radio/rfsimulator/CMakeLists.txt` | CUDA 빌드 추가 | ★ |

### Proxy 쪽 (Python 코드)

| 파일 | 작업 | 난이도 |
|------|------|:------:|
| `G0_Sionna_Channel_Proxy/v12_gpu_ipc.py` | IPC 모드 Proxy (신규, v10 기반) | ★★ |

### Docker 설정

| 파일 | 작업 | 난이도 |
|------|------|:------:|
| `docker-compose.yml` | `--ipc=host`, volume mount 추가 | ★ |

---

## 7. 리스크 및 대안

### 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| OAI CUDA 빌드 호환성 | 빌드 실패 | CUDA를 런타임 로드 (dlopen) |
| Docker CUDA IPC 미동작 | 기능 불가 | `--ipc=host` + `--pid=host` |
| 동기화 지연 | 성능 저하 | spin-lock 최적화, CPU affinity |
| OAI 업데이트 시 충돌 | 유지보수 | rfsimulator 분리 모듈화 |

### 대안 (GPU IPC 실패 시)

**Plan B: POSIX Shared Memory**

```
OAI → shm_open() → shared memory (pinned) → Proxy → H2D → GPU
```

- CUDA 의존성 없이 구현 가능
- Socket 대비 syscall 제거, 복사 1회 감소
- GPU IPC보다는 느리지만 Socket보다 빠름

---

## 8. 멀티스트림 파이프라인 — Throughput 극대화

### 8.1 핵심 철학

```
1. 1 TTI(500μs) 최대한 맞추기 ← 목표
2. 못 맞추면 어쩔 수 없음 ← 현실
3. Throughput은 최대한 높여야 함 ← 핵심 가치

목표 Throughput = 2000 slots/sec (= 1 TTI 기준)
```

**왜 Throughput인가?**

1 TTI를 초과하더라도, throughput이 높으면 전체 시스템 처리량이 늘어남.
예: 16T에서 1 TTI 불가하지만, throughput 1500 slots/sec은 전체 처리량의 75%를 커버.

### 8.2 이론적 배경 — 왜 멀티스트림이 Throughput을 높이는가

#### CUDA 스트림 오버랩

GPU 파이프라인은 3단계로 구성됨: **H2D → GPU Compute → D2H**

이 3단계는 서로 다른 하드웨어 엔진을 사용:

| 단계 | 하드웨어 | PCIe 방향 |
|------|----------|:---------:|
| H2D (Host→Device) | DMA Copy Engine | ↑ (업로드) |
| GPU Compute | SM (Streaming Multiprocessors) | - |
| D2H (Device→Host) | DMA Copy Engine | ↓ (다운로드) |

**핵심**: PCIe는 **양방향 동시 전송** 가능. H2D와 D2H가 물리적으로 다른 레인을 사용하므로, 서로 다른 스트림에서 동시에 실행하면 **대역폭이 2배로 활용**됨.

```
순차 처리 (단일 스트림):
  Slot 1: [H2D]────[GPU]────[D2H]
                                  (idle)
  Slot 2:                   [H2D]────[GPU]────[D2H]

  Wall-clock: T_slot × N_slots
  → H2D/D2H 할 때 반대 방향 레인은 놀고 있음

멀티스트림 (2-스트림 오버랩):
  Stream 0: [H2D₀]────[GPU₀]────[D2H₀]         [H2D₂]────[GPU₂]────[D2H₂]
  Stream 1:       [H2D₁]────[GPU₁]────[D2H₁]         [H2D₃]────[GPU₃]────...
                        ↑                    ↑
                        오버랩!              오버랩!

  D2H(N)과 H2D(N+1)이 동시 실행 → PCIe 양방향 활용
  Wall-clock: T_slot + (N-1) × T_effective  (T_effective < T_slot)
```

#### PCIe 양방향 동시 전송 이론

| 항목 | PCIe 3.0 x16 | PCIe 4.0 x16 | PCIe 5.0 x16 |
|------|:---:|:---:|:---:|
| 단방향 대역폭 | ~15.75 GB/s | ~31.5 GB/s | ~63 GB/s |
| 양방향 합산 | ~31.5 GB/s | ~63 GB/s | ~126 GB/s |
| 이론적 Speedup | 2.0x | 2.0x | 2.0x |

**양방향 동시 전송 활성화 조건**:
1. **CUDA Streams**: 서로 다른 스트림에서 H2D/D2H 동시 실행
2. **독립 버퍼**: H2D와 D2H가 서로 다른 메모리 영역 사용
3. **비동기 API**: `cudaMemcpyAsync()` 사용 (CuPy: `gpu_array.set()` / `.get()`)
4. **Pinned Memory**: DMA 전송을 위해 page-locked 메모리 필수

#### 3-스트림 파이프라인 구조

```
Slot N:   [H2D: 1.75MB] → Event(h2d_done)
Slot N-1:                 [wait h2d_done] → [Compute: Batch FFT] → Event(compute_done)
Slot N-2:                                    [wait compute_done] → [D2H: 1.75MB]

Ring Buffer: 3개 버퍼가 순환 (Write → Compute → Read)
Effective Time/Slot = max(T_H2D, T_GPU, T_D2H) + 동기화 오버헤드
```

#### 멀티스트림이 유리한 조건 vs 불리한 조건

| 조건 | 순차 처리 | 멀티스트림 | 비고 |
|------|:---:|:---:|------|
| 단일 슬롯 Latency 우선 | ✓ | × | 멀티스트림은 Latency 증가 가능 |
| **Throughput 우선** | × | **✓** | **권장** |
| 구현 복잡도 | 간단 | 중간 | 스트림 관리 + 더블 버퍼링 |
| 메모리 사용량 | ×1 | ×2~×3 | 버퍼 다중화 |
| H2D/D2H 비중 높을 때 | × | **✓** | PCIe 양방향 효과 극대 |
| GPU 연산 비중 높을 때 | ○ | △ | 오버랩 효과 제한 |

### 8.3 G1B 멀티스트림 실험 결과 (TITAN RTX, 2026-01-21)

#### 테스트 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA TITAN RTX |
| PCIe | 3.0 x16 (단방향 ~12 GB/s) |
| 데이터 | n_ant × 14 × 2048 × complex64 |
| 멀티스트림 | 2-스트림 (Stream 0, Stream 1 교대) |
| 측정 방법 | 100 slots wall-clock / N_slots |

#### 순차 vs 멀티스트림 Throughput 비교

| 안테나 | 데이터 크기 | 순차 (slots/sec) | 멀티스트림 (slots/sec) | **TP 향상** | 1 TTI (500μs) |
|:------:|:---:|:---:|:---:|:---:|:---:|
| **8T** | 1.75 MB | 2455 | **2769** | **+12.8%** | ✅ |
| 16T | 3.50 MB | 1431 | 1500 | +4.8% | ❌ |
| 32T | 7.00 MB | 747 | 762 | +2.0% | ❌ |
| 64T | 14.68 MB | 380 | 383 | +0.9% | ❌ |

**핵심 관찰**:
- **8T**: 멀티스트림으로 **+314 slots/sec** 추가 확보 (2455 → 2769)
- **16T 이상**: 데이터 크기 증가로 PCIe 전송 시간이 지배적 → 오버랩 효과 감소
- 안테나 수 증가 시 Throughput 향상 비율 감소 (12.8% → 4.8% → 2.0% → 0.9%)

#### PCIe 양방향 동시 전송 실측 (H2D || D2H)

| 안테나 | 순차 (H2D→D2H) | 동시 (H2D ∥ D2H) | Speedup | 효율 |
|:------:|:---:|:---:|:---:|:---:|
| 8T | 307 μs | 178 μs | 1.72x | 86% |
| **16T** | 596 μs | 341 μs | **1.75x** | **87.5%** |
| 32T | 1174 μs | 665 μs | 1.76x | 88% |

- 이론적 최대 Speedup: 2.0x
- 실측 효율: **86~88%** (PCIe 컨트롤러 오버헤드 + CUDA 동기화 지연)

#### 단일 슬롯 시간 분해 (8T 기준)

| 단계 | 시간 | 비중 |
|------|:---:|:---:|
| H2D (1.75 MB) | 163 μs | 35% |
| GPU Compute (Batch FFT) | 139 μs | 30% |
| D2H (1.75 MB) | 161 μs | 35% |
| **합계** | **463 μs** | **100%** |

```
H2D/D2H 합산: 324 μs (70%) ← 멀티스트림 오버랩 대상
GPU Compute:  139 μs (30%) ← 독립 실행

멀티스트림 실효 시간 (Steady State):
  max(H2D, GPU, D2H) + 동기화 = max(163, 139, 161) + ~35μs ≈ 198μs
  → 단일 슬롯 대비 463/198 = 2.3x 향상 (이론값)
  → 실측: 361μs/slot (1.28x 향상, 2769 slots/sec)
```

#### 1 TTI 달성 상황 요약

```
8T:  2769 slots/sec (멀티스트림) → 목표 2000 초과 ✅ (+38% 여유)
16T: 1500 slots/sec → 목표 대비 75% (어쩔 수 없음)
32T:  762 slots/sec → 목표 대비 38%
64T:  383 slots/sec → 목표 대비 19%
```

#### 권장 구성

| 상황 | 권장 | Throughput | Latency |
|------|------|:----------:|:-------:|
| **1 TTI 필수** | 8T 멀티스트림 | 2769 slots/sec | 361μs ✅ |
| 1 TTI 초과 감수 | 16T 멀티스트림 | 1500 slots/sec | 667μs |
| 대규모 안테나 | 32T+ 멀티스트림 | Throughput 최대화 | TTI 초과 |

### 8.4 추가 Throughput 최적화 기법

#### 1. Custom CUDA Kernel 퓨전 (cupy.RawKernel)

현재 CuPy 파이프라인은 ~15개 커널을 CUDA Graph로 묶어 런치 오버헤드를 제거함.
추가로 커널 자체를 **수동 퓨전**하면 중간 텐서 메모리 접근이 줄어듦:

```
현재 (15 커널, CUDA Graph):
  K1: int16→float64 | K2: complex128 변환 | K3: symbol extract
  K4: FFT | K5: 채널곱 | K6: IFFT | K7: reconstruct
  K8: PL | K9: AWGN | K10: clip | K11: int16 변환 | ...

퓨전 후 (5~6 커널):
  F1: int16 → complex128 + symbol extract  → 1 커널
  F2: FFT (cuFFT, 퓨전 불가)               → 1 커널
  F3: 채널곱 + IFFT                         → 1 커널  
  F4: reconstruct + CP insert               → 1 커널
  F5: PL + AWGN + clip + int16              → 1 커널
```

| 항목 | CUDA Graph (현재) | + RawKernel 퓨전 |
|------|:---:|:---:|
| 커널 수 | 15 | 5~6 |
| 중간 텐서 메모리 접근 | 15회 R/W | 5~6회 R/W |
| 예상 GPU Compute 시간 | ~0.47 ms | ~0.35 ms |
| 적용 난이도 | 완료 | 중간 |

#### 2. CH_COPY in-place 참조

현재 채널 H 복사에 ~0.31ms 소요. GPU 메모리 내 직접 참조로 전환:

| 항목 | 현재 (복사) | 개선 (in-place) |
|------|:---:|:---:|
| CH_COPY 시간 | 0.31 ms | ~0.05 ms |
| 절약량 | - | ~0.25 ms |
| 구현 방법 | RingBuffer에서 GPU 슬라이스 직접 참조 |

#### 3. Python 오버헤드 감소

| 방법 | 절약량 | 난이도 |
|------|:---:|:---:|
| C 확장 모듈 (Socket I/O) | ~0.2 ms | 높음 |
| Cython 래퍼 | ~0.15 ms | 중간 |
| ctypes 최적화 | ~0.05 ms | 낮음 |

#### 4. 전체 최적화 조합 예상 효과

| 최적화 | 절약량 | 누적 시간 |
|--------|:---:|:---:|
| 현재 (v11, CUDA Graph) | - | ~1.88 ms/slot |
| + CH_COPY in-place | -0.25 ms | ~1.63 ms |
| + RawKernel 퓨전 | -0.12 ms | ~1.51 ms |
| + Socket 제거 (CUDA IPC) | -0.50 ms | ~1.01 ms |
| + Python 오버헤드 감소 | -0.15 ms | ~0.86 ms |
| **이론적 최소** | | **~0.86 ms/slot** |

```
현재:          ████████████████████ 1.88 ms
최적화 후:     ████████████████     1.01 ms (Socket 제거 시)
이론적 최소:   ██████████████       0.86 ms
1 TTI 목표:    █████                0.50 ms ← 여전히 도달 어려움

결론: Socket 제거 + 커널 퓨전으로 ~1.0ms 수준 가능
      1 TTI(0.5ms)는 GPU 연산 자체 한계로 도달 어려움
      → Throughput 극대화가 현실적 목표
```

---

## 9. Socket 제거 + 멀티스트림 통합 전략

### 9.1 두 방향의 시너지

| 방향 | 효과 | 작용 대상 |
|------|------|-----------|
| Socket 제거 (CUDA IPC) | ~0.5ms 절약 | 데이터 전송 경로 |
| 멀티스트림 파이프라인 | Throughput +12.8% | GPU 파이프라인 오버랩 |

**통합 시 예상 구조**:

```
OAI gNB (CPU)
  ↓ cudaMemcpyAsync H2D (Stream 0) → gpu_buf_tx_0
  ↓
Sionna Proxy (같은 GPU)
  ↓ Stream 0: CUDA IPC → GPU 채널처리 → D2H async
  ↓ Stream 1: 다음 슬롯 H2D async (동시 진행)
  ↓ PCIe 양방향: D2H(현재) || H2D(다음) 동시
  ↓
OAI UE (CPU)
  ↓ cudaMemcpyAsync D2H (완료 대기)
```

### 9.2 예상 성능 (Socket 제거 + 멀티스트림)

| 항목 | 현재 (Socket + 순차) | Socket 제거만 | Socket 제거 + 멀티스트림 |
|------|:---:|:---:|:---:|
| 데이터 전송 | ~0.57 ms | ~0.1 ms | ~0.1 ms (오버랩) |
| GPU 연산 | ~0.47 ms | ~0.47 ms | ~0.47 ms |
| 총 시간/slot | ~1.88 ms | ~1.01 ms | ~0.8 ms (예상) |
| Throughput | 452 slots/sec | ~990 slots/sec | ~1250 slots/sec (예상) |

### 9.3 적용 우선순위

```
Phase 1 (완료): CUDA Graph → 3.2ms → 1.88ms ✅
Phase 2 (완료): 멀티스트림 검증 (G1B 목업) → +12.8% Throughput ✅
Phase 3 (진행): E2E 계측 고도화 (v11) → 병목 분석 ✅
Phase 4 (예정): Socket 제거 (CUDA IPC) → ~1.0ms 목표
Phase 5 (예정): 멀티스트림 E2E 적용 → Throughput 극대화
Phase 6 (옵션): RawKernel 퓨전 → ~0.86ms 목표
```

---

## 10. 성공 기준

### Socket 제거 성공 기준

| 항목 | 기준 |
|------|------|
| Socket 완전 제거 | gNB ↔ Proxy ↔ UE 간 TCP 연결 없음 |
| 기능 정상 | PSS/SSS 동기화 + RACH 성공 |
| 성능 개선 | Socket 구간 지연 50% 이상 감소 |
| 대용량 확장 | 1MB+ 데이터에서 선형 확장 |
| 호환성 유지 | `--mode=socket` 으로 기존 동작 가능 |

### Throughput 극대화 성공 기준

| 항목 | 기준 |
|------|------|
| 멀티스트림 E2E 적용 | OAI E2E 환경에서 멀티스트림 동작 |
| Throughput 향상 | 순차 대비 +10% 이상 (8T 기준) |
| Latency 허용 범위 | 단일 슬롯 latency 증가 < 50% |
| 안정성 | 연속 10000+ 슬롯 미스율 0% |

---

## 11. 일정 요약

| 주차 | 작업 | 분류 |
|:----:|------|:----:|
| 1주 | OAI rfsimulator 코드 분석 + CUDA 빌드 환경 구성 | Socket 제거 |
| 2~3주 | OAI rfsimulator GPU IPC 구현 (write/read 수정) | Socket 제거 |
| 3~4주 | Proxy GPU IPC 모드 구현 (v12) | Socket 제거 |
| 4주 | 통합 테스트 + 성능 비교 | Socket 제거 |
| 5주 | 대용량 데이터 검증 | Socket 제거 |
| 5~6주 | 멀티스트림 E2E 적용 (v12에 통합) | Throughput |
| 6~7주 | RawKernel 퓨전 + CH_COPY 최적화 | Throughput |
| 7주 | 최종 성능 비교 + 문서 정리 | 공통 |

---

## 12. 관련 문서

- `vRAN_Socket/G0_Sionna_Channel_Proxy/README.md` — 현재 Proxy 성능 분석 및 아키텍처
- `vRAN_Socket/G1B_100T_Scalable_Mock/README.md` — 멀티스트림 실험 결과 (Throughput 분석)
- `openairinterface5g_whan/radio/rfsimulator/simulator.c` — OAI rfsimulator 핵심 코드
- `openairinterface5g_whan/radio/rfsimulator/README.md` — rfsimulator 사용법
- `vRAN_Socket/G0_Sionna_Channel_Proxy/v10_cuda_graph.py` — 현재 최적 Proxy (CUDA Graph)
- `vRAN_Socket/G0_Sionna_Channel_Proxy/v11_profile_instrumented.py` — 계측 고도화 버전
- `vRAN_Socket/G1B_100T_Scalable_Mock/src/test_multistream_pipeline.py` — 멀티스트림 실험 코드



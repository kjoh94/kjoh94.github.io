---
layout: post
title: "GPU 없어도 된다: RunPod에서 10분 만에 CUDA 맛보기"
date: 2026-10-19 22:30:00 +0900
categories: [gpu, cuda, tutorial]
tags: [runpod, cuda, nvcc, pytorch, gpu, beginner, cloud-gpu]
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/U2VZT2jFJgo" title="RunPod CUDA 맛보기" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

내 컴퓨터에 GPU가 없어도 CUDA는 써 볼 수 있다. 클라우드 GPU를 몇 분만 빌리면 된다. 이번에는 RunPod에서 GPU 하나를 켜고, 첫 CUDA 프로그램을 컴파일해서 CPU와 속도를 비교하고, 초보자가 자주 하는 실수 두 가지까지 직접 겪어 봤다. 실제로 돌린 시간은 2분 남짓, 비용은 약 2센트였다.

- 코드: <https://github.com/kjoh94/runpod-cuda-taster>
- 영상 (한국어 본편): <https://youtu.be/U2VZT2jFJgo>
- 영상 (한국어 Shorts): <https://youtube.com/shorts/LihhVYCMQJM>
- Video (English full tutorial): <https://youtu.be/3bGNNVeXUiQ>
- Video (English Shorts): <https://youtube.com/shorts/VeQC7OlXWLc>

## 결과부터

GPU는 RTX PRO 6000 Blackwell을 MIG로 나눈 **24GB 조각(1g.24gb)** 이다. GPU 한 장을 여러 사람이 나눠 쓰는 형태라 시간당 $0.69로 저렴하다.

| 실험 | CPU | GPU | 배속 |
|---|---|---|---|
| 벡터 덧셈 (float 5천만 개) | 71.9 ms | 1.61 ms | **약 45배** |
| 행렬곱 4096×4096 (PyTorch) | 896.5 ms | 7.44 ms | **약 120배** |

두 번째 실행에서는 벡터 덧셈이 62.8ms / 1.62ms로 약 39배였다. 숫자는 GPU, CPU, 실행할 때마다 조금씩 달라진다.

## 1. Pod 만들기

1. RunPod에 가입하고 **Billing**에서 크레딧을 조금 충전한다. 실습은 몇백 원이면 충분하다.
2. 왼쪽 메뉴 **Pods → Deploy**.
3. GPU는 싼 것 하나면 된다. RTX A4000, RTX 4000 Ada, RTX 3090, MIG 조각 등 시간당 $0.2~0.7 정도.
4. 템플릿은 공식 **RunPod PyTorch**. CUDA 컴파일러 `nvcc`가 들어 있다.
5. 디스크는 Container 20GB, Volume 20GB(`/workspace`)면 넉넉하다.
6. **Deploy On-Demand**를 누르고 1~2분 기다린다.

## 2. 접속하기

Pod 카드에서 **Connect → Web Terminal**을 누르면 브라우저에 터미널이 뜬다. Jupyter Lab의 Terminal을 써도 된다.

## 3. GPU 확인

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi`에 GPU 이름과 메모리가 보이면 준비 끝이다. MIG 조각이면 아래쪽에 `MIG devices` 표가 따로 나오고 메모리가 `24192MiB`로 보인다. `nvcc`는 12.8이었다.

## 4. 코드 받기

```bash
cd /workspace
git clone https://github.com/kjoh94/runpod-cuda-taster.git cuda_taster
cd cuda_taster
```

## 5. 첫 CUDA 프로그램: 벡터 덧셈

핵심은 이 커널 하나다. 스레드 하나가 원소 하나를 맡는다.

```cpp
__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 내 스레드 번호
    if (i < n) c[i] = a[i] + b[i];
}
```

순서는 이렇다. CPU 메모리에 데이터를 만들고 → `cudaMalloc`으로 GPU 메모리를 잡고 → `cudaMemcpy`로 복사하고 → 블록당 256 스레드로 커널을 실행하고 → `cudaEvent`로 커널 시간만 재고 → 결과를 다시 가져와 CPU 결과와 비교한다.

```bash
mkdir -p ~/build
nvcc -O2 -arch=native vector_add.cu -o ~/build/vector_add
~/build/vector_add
```

```
N = 50000000 floats (200 MB per array)
CPU loop   :    71.91 ms
GPU kernel :     1.61 ms
Speedup    :     44.6x
Check      : PASS (0 wrong)
```

**삽질 하나:** 처음엔 실행 파일을 `/workspace`에 만들었더니 `Permission denied`가 났다. 네트워크 볼륨은 실행 파일을 직접 실행하지 못하게 마운트될 수 있다. 소스는 `/workspace`에 두고, 빌드 결과는 `~/build` 같은 컨테이너 로컬 디스크에 만들면 해결된다. `-arch=native`를 붙이면 지금 GPU에 맞춰 컴파일되고 경고도 사라진다.

## 6. 초보자 실수 두 가지

```bash
nvcc -O2 -arch=native beginner_error.cu -o ~/build/beginner_error
~/build/beginner_error
```

```
[BAD ] fill<<<..., 2048>>>  (block size too big)
       h[0] = 0  (expected 7)  <- 아무 에러 메시지 없이 틀린 값
[FIX ] cudaGetLastError() -> invalid configuration argument
[GOOD] fill<<<..., 256>>>
       launch -> no error
       h[0] = 7  (expected 7)
[BAD ] timing without sync : 0.12 ms (너무 빠름 = 거짓말)
[FIX ] timing with cudaDeviceSynchronize : 35.18 ms
```

- **블록당 스레드는 최대 1024개.** 2048개로 실행하면 커널이 아예 안 도는데, 에러 메시지도 안 나온다. 실행 직후 `cudaGetLastError()`를 꼭 확인하자.
- **커널 실행은 비동기다.** CPU는 GPU가 끝나기 전에 다음 줄로 넘어간다. 동기화 없이 시간을 재면 0.12ms 같은 거짓 숫자가 나온다. `cudaDeviceSynchronize()`로 기다린 뒤에 재야 진짜 35ms가 나온다.

## 7. 보너스: PyTorch는 `.to("cuda")` 한 줄

```bash
python torch_matmul.py
```

```
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
matmul 4096x4096  CPU:    896.5 ms
matmul 4096x4096  GPU:     7.44 ms
Speedup: 120x
```

여기서도 시간을 재기 전에 `torch.cuda.synchronize()`를 부른다. 6번의 교훈과 같다.

## 8. ⚠️ 다 쓰면 꼭 Stop

GPU Pod는 켜 둔 시간만큼 돈이 나간다. 실습이 끝나면 바로 **Stop**을 누르자.

- **Stop**: GPU 요금이 멈춘다. 볼륨 데이터는 남는다(보관 요금은 소액 발생).
- **Terminate**: Pod 자체를 지운다. 네트워크 볼륨이 아니면 데이터도 사라진다.

이번 실습은 Pod를 켜고 끄기까지 약 2분, 비용은 $0.69 × 2/60 ≈ **$0.02** 였다.

## 정리

- GPU가 없어도 클라우드 GPU로 몇 분 만에 CUDA를 맛볼 수 있다.
- 단순한 벡터 덧셈도 약 45배, 행렬곱은 약 120배 빨랐다.
- 커널 실행 후에는 `cudaGetLastError()`, 시간을 잴 때는 동기화.
- 끝나면 Pod는 꼭 Stop.

다음에는 같은 Pod에서 PyTorch 모델을 TensorRT로 바꿔 속도를 비교해 볼 생각이다.

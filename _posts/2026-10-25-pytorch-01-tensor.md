---
layout: post
title: "PyTorch 시리즈 1 — 텐서 속을 열어 보기: storage·stride·view·contiguous·dtype·device"
date: 2026-10-25 21:01:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, tensor, stride, view, contiguous, dtype, device, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [기초편 C](/blog/2026/10/31/pytorch-00c-loss-train-loop/)  
> → 다음: [2. 모델은 부품 상자](/blog/2026/10/25/pytorch-02-module-state-dict/)

> **처음이신가요?** 이 장은 기초편 A에서 본 "숫자 상자"의 속을 열어 봐요. 텐서·신경망이 처음이라면 [기초편 A. 텐서를 숫자 상자로](/blog/2026/10/31/pytorch-00a-tensor-numbers/) → [B. Linear와 순전파](/blog/2026/10/31/pytorch-00b-linear-forward/) → [C. loss와 학습 루프](/blog/2026/10/31/pytorch-00c-loss-train-loop/)를 먼저 읽고 오면 훨씬 쉬워요.

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/01-tensor](https://github.com/kjoh94/pytorch-series-examples/tree/main/01-tensor) (`python example.py`)

기초편 A에서 텐서를 "숫자를 담은 상자"로 봤죠. 이번 장은 그 상자를 **한 겹 열어서 안을 들여다보는** 장이에요. 어렵게 들리지만 걱정 마세요. 결론부터 말하면 딱 한 문장이에요.

> **텐서 = (숫자가 한 줄로 놓인 메모리) + (그걸 어떻게 읽을지 적은 메모)**

이 한 문장만 이해하면, 언젠가 꼭 만나게 될 이런 에러들이 왜 나는지 스스로 풀 수 있어요.

- `RuntimeError: view size is not compatible with input tensor's size and stride`
- `Expected all tensors to be on the same device`
- `mat1 and mat2 must have the same dtype, but got Half and Float`

그리고 시리즈 뒤쪽(ATen, `torch.compile`, `torch.export`, 양자화)도 전부 이 장의 단어 여섯 개 — **storage, shape, stride, offset, dtype, device** — 로 이야기해요. 그래서 이 장은 조금 길어도 천천히 가요.

모든 예제는 CPU 하나로 돌아가요. `pip install torch` 하고 그대로 복사해서 실행해 보세요.

### 이 장에서 배우는 것 (미리보기)

| 단어 | 한 줄 뜻 | 비유 |
|---|---|---|
| storage | 숫자가 실제로 놓인 한 줄짜리 메모리 | 번호 붙은 사물함 한 줄 |
| shape | 우리 눈에 보이는 모양 | "2줄 3칸으로 보세요" |
| stride | 다음 칸으로 가려면 사물함 몇 개 건너뛰나 | 걸음 폭 |
| offset | 몇 번 사물함부터 읽기 시작하나 | 출발점 |
| dtype | 숫자 한 개의 종류·크기 | 사물함 하나의 크기 |
| device | 메모리가 어디(CPU/GPU) 있나 | 사물함이 있는 건물 |

## 0. 먼저 — 컴퓨터 메모리는 "한 줄"이다

![메모리는 한 줄](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig10.jpg)

**그림 0.** 우리는 행렬을 표처럼 생각하지만, 컴퓨터 메모리에는 "행"이나 "열"이 없어요. 그냥 **0번, 1번, 2번… 번호가 붙은 칸이 한 줄로** 있을 뿐이에요. 그래서 2×3 표를 저장하려면 첫 줄 `0 1 2` 뒤에 둘째 줄 `3 4 5`를 이어 붙여요. 이걸 **행 우선(row-major)** 저장이라고 불러요.

그럼 자연스럽게 질문이 생겨요. "한 줄짜리 메모리를 보고 어떻게 다시 2×3 표로 읽지?" — 그 답을 적어 둔 게 바로 다음에 나올 **메모(메타데이터)** 예요.

```python
import torch
x = torch.tensor([[0., 1., 2.],
                  [3., 4., 5.]])
print(x.flatten())   # tensor([0., 1., 2., 3., 4., 5.])  ← 메모리에 놓인 순서 그대로
```

## 1. 텐서는 두 부분으로 되어 있다

![텐서 = 메타데이터 + 저장소](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig01.jpg)

**그림 1.** 아래 파란 칸이 실제 숫자가 놓인 **Storage**다. 몇 차원 텐서든 Storage는 항상 **1차원으로 이어진 메모리 덩어리**다. 왼쪽 위 상자는 텐서의 "머리표(메타데이터)"로, 이 1차원 메모리를 **어떤 모양(shape)으로, 몇 칸씩 건너뛰며(stride), 어디서부터(offset), 어떤 타입(dtype)으로, 어떤 장치(device)에서** 읽을지 적어 둔다. 오른쪽의 2×3 행렬은 우리가 `print`로 보는 모습일 뿐, 메모리에는 `0 1 2 3 4 5`가 한 줄로 있다.

```python
import torch

x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(x)
print("shape :", x.shape)            # torch.Size([2, 3])
print("stride:", x.stride())         # (3, 1)
print("offset:", x.storage_offset()) # 0
print("dtype :", x.dtype)            # torch.float32
print("device:", x.device)           # cpu
print("storage:", x.untyped_storage().nbytes(), "bytes")  # 6 * 4 = 24
```

핵심 문장 하나만 기억하자. **"텐서 연산의 절반은 숫자를 바꾸지 않고 머리표만 바꾼다."** 이것이 PyTorch가 빠른 이유이자, 헷갈리는 이유다.

> **쉽게 말하면:** 사물함 한 줄 앞에서 "아래 행으로 한 칸" = 3개 건너뛰기, "옆 열로 한 칸" = 1개 건너뛰기. 이 걸음 폭 목록이 stride예요.

## 2. stride — 다음 칸까지 몇 칸 건너뛰나

![stride](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig02.jpg)

**그림 2.** `x[1, 2]`는 메모리 몇 번째 칸일까? 공식은 `offset + i*stride[0] + j*stride[1]`. 행을 하나 내려가면 3칸(빨간 화살표, `stride[0]=3`), 열을 하나 옆으로 가면 1칸(초록 화살표, `stride[1]=1`)을 건너뛴다. 그래서 `0 + 1*3 + 2*1 = 5`번 칸, 즉 값 5가 나온다.

```python
x = torch.arange(6.).reshape(2, 3)
i, j = 1, 2
flat = x.storage_offset() + i * x.stride(0) + j * x.stride(1)
print(flat, x[i, j].item())              # 5 5.0
print(x.flatten()[flat].item())          # 5.0  (연속 텐서라서 가능)
```

3차원 이상도 똑같다. shape `(2, 3, 4)`의 연속 텐서는 stride가 `(12, 4, 1)` — 뒤 차원 크기를 곱해 온 값이다. 이렇게 **마지막 차원 stride가 1이고 앞으로 갈수록 뒤 크기의 곱**인 상태를 **C-contiguous(행 우선 연속)** 라고 부른다.

```python
print(torch.zeros(2, 3, 4).stride())  # (12, 4, 1)
```

> **손으로 한 번 계산해 보기:** shape `(2, 3, 4)`에서 `x[1, 2, 3]`은 몇 번 칸일까요? stride가 `(12, 4, 1)`이니 `1*12 + 2*4 + 3*1 = 23`. 실제로 `torch.arange(24.).reshape(2,3,4)[1,2,3]`을 찍어 보면 `23.`이 나와요. 이 계산을 한 번 손으로 해 보면 stride가 확 와닿아요.

## 3. view — 복사 없이 다른 모양으로 읽기

![view는 복사가 아니다](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig03.jpg)

**그림 3.** 세 텐서(청록 `x`, 노랑 `view(2,3)`, 초록 `view(3,2)`)는 모두 **같은 Storage 하나**를 가리킨다. 머리표의 shape/stride만 다르다. 그래서 하나를 고치면 나머지도 바뀐다(위쪽 빨간 문장).

```python
x = torch.arange(6.)
y = x.view(2, 3)
z = x.view(3, 2)
print(x.data_ptr() == y.data_ptr() == z.data_ptr())  # True — 같은 메모리
y[0, 0] = 100
print(x[0].item(), z[0, 0].item())                    # 100.0 100.0
```

view 계열 연산(복사 없음): `view`, 대부분의 슬라이싱 `x[:, 1:]`, `transpose`/`t`/`permute`, `unsqueeze`/`squeeze`, `expand`, `narrow`, `select`. 슬라이싱은 **offset**을 바꾸는 것으로 구현된다.

```python
x = torch.arange(12.).reshape(3, 4)
s = x[1:, 2:]                     # 오른쪽 아래 2×2
print(s.storage_offset(), s.stride())   # 6 (4, 1)  — 1행*4 + 2열 = 6에서 시작
print(s.data_ptr() == x.data_ptr() + 6 * x.element_size())  # True
```

> **주의 — view는 "같은 사물함을 공유"해요.** 그래서 `y = x.view(...)`를 고치면 `x`도 바뀌어요. 원본을 지키고 싶으면 `x.clone()`으로 진짜 복사본을 만드세요.

## 4. transpose — stride만 바꾸면 "비연속"이 된다

![transpose](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig04.jpg)

**그림 4.** `x.t()`는 숫자를 하나도 옮기지 않는다. stride `(3,1)`을 `(1,3)`으로 **맞바꿀 뿐**이다. 그런데 이 결과를 행 순서대로 읽으면 메모리 주소가 `0→3→1→4→2→5`로 뛰어다닌다(오른쪽 아래). 이것이 **non-contiguous** 상태다.

```python
x = torch.arange(6.).reshape(2, 3)
xt = x.t()
print(xt.shape, xt.stride(), xt.is_contiguous())  # (3, 2) (1, 3) False
try:
    xt.view(6)
except RuntimeError as e:
    print("view 실패:", str(e)[:60], "...")
```

왜 `view(6)`이 실패할까? 1차원으로 펴려면 "한 칸씩 연속으로 읽기(stride 1)"가 되어야 하는데, 이 메모리 배치로는 그런 stride 하나를 만들 수 없기 때문이다.

## 5. contiguous()와 reshape() — 언제 복사가 일어나나

![contiguous](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig05.jpg)

**그림 5.** `contiguous()`는 **이미 연속이면 자기 자신을 그대로** 돌려주고, 아니면 **새 Storage를 만들어 행 순서로 복사**한다. 아래 두 상자는 자주 헷갈리는 `view`와 `reshape`의 차이: `view`는 절대 복사하지 않는 대신 실패할 수 있고, `reshape`는 되면 view, 안 되면 조용히 복사한다.

```python
x = torch.arange(6.).reshape(2, 3)
xt = x.t()
c = xt.contiguous()
print(c.stride(), c.data_ptr() == x.data_ptr())   # (2, 1) False — 복사됨
print(x.contiguous() is x)                         # True — 이미 연속이면 그대로

r1 = x.reshape(3, 2);  print(r1.data_ptr() == x.data_ptr())   # True  (view로 처리)
r2 = xt.reshape(6);    print(r2.data_ptr() == x.data_ptr())   # False (복사)
```

실전 감각:

- **성능**: 비연속 텐서를 받으면 커널이 내부에서 `contiguous()` 복사를 하거나 느린 strided 경로를 탄다. 큰 텐서면 메모리 대역폭을 그대로 잡아먹는다.
- **내보내기/컴파일**: 많은 백엔드(ONNX 런타임, 가속기 커널, 커스텀 C++ 확장)는 "연속 메모리"를 가정한다. 커스텀 커널에 넘기기 전 `x.contiguous()`는 거의 관례다(12장).
- **channels_last**: 이미지 모델에서는 `NCHW` 텐서를 메모리상 `NHWC` 순서로 두는 `torch.channels_last` 포맷이 있다. shape은 그대로, stride만 바뀐다.

```python
img = torch.randn(1, 3, 4, 4)
cl = img.contiguous(memory_format=torch.channels_last)
print(img.stride(), cl.stride())   # (48, 16, 4, 1) (48, 1, 12, 3)
```

### 복사가 생길지 헷갈릴 때

![복사 판단 순서](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig11.jpg)

**그림 5-2.** 대충 이렇게 판단하면 돼요. **값을 새로 계산하는 연산**(`x + 1`, `x * 2`, `torch.relu(x)`)은 결과를 담을 새 메모리가 필요하니 항상 새로 만들어요. **모양이나 읽는 순서만 바꾸는 연산**(`view`, `t`, `permute`, 슬라이싱)은 메모만 바꾸는 공짜 연산이에요. 다만 그 결과가 비연속인데 연속이 꼭 필요한 곳(`view`, 일부 커널)에 넘기면, 그때 `contiguous()`/`reshape`가 복사를 해요.

```python
x = torch.arange(6.).reshape(2, 3)
print((x + 1).data_ptr() == x.data_ptr())   # False — 새 메모리
print(x.t().data_ptr() == x.data_ptr())     # True  — 메모만 바뀜
```

### 실전 예: 이미지 배치 순서 바꾸기

![NHWC to NCHW](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig13.jpg)

**그림 5-3.** 이미지 라이브러리는 보통 `(높이, 너비, 채널)` 순서로 주고, PyTorch의 `Conv2d`는 `(채널, 높이, 너비)` 순서를 기대해요. 이때 `permute`를 쓰는데, 이건 숫자를 옮기지 않고 stride만 바꿔요. 그래서 결과는 비연속이에요.

```python
imgs = torch.rand(8, 32, 32, 3)          # N, H, W, C
chw = imgs.permute(0, 3, 1, 2)           # N, C, H, W
print(chw.shape, chw.is_contiguous())    # torch.Size([8, 3, 32, 32]) False
conv = torch.nn.Conv2d(3, 16, 3)
print(conv(chw).shape)                   # 그대로 넣어도 동작 (내부에서 처리)
flat = chw.reshape(8, -1)                # view 대신 reshape — 필요하면 알아서 복사
print(flat.shape)                        # torch.Size([8, 3072])
```

기초편 B에서 `Linear`에 넣기 전 이미지를 "펴는" 장면이 바로 이 마지막 줄이에요. 여기서 `view`를 쓰면 에러가 나고 `reshape`를 쓰면 되는 이유, 이제 설명할 수 있죠?

## 6. broadcasting과 stride 0

![broadcasting](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig06.jpg)

**그림 6.** 크기 3짜리 벡터 `b`를 `(4,3)` 행렬에 더할 때, PyTorch는 `b`를 4번 복사하지 않는다. `expand(4,3)`은 **행 방향 stride를 0**으로 둔 view를 만든다. "다음 행으로 가도 0칸 이동"이니 같은 3칸을 계속 다시 읽는다. 흐릿한 아래 세 줄은 실제로 메모리에 존재하지 않는 "보이기만 하는" 행이다.

```python
b = torch.tensor([10., 20., 30.])
e = b.expand(4, 3)
print(e.stride(), e.data_ptr() == b.data_ptr())   # (0, 1) True
A = torch.ones(4, 3)
print(A + b)   # 내부적으로 같은 broadcasting 규칙

# 주의: stride 0 텐서에 in-place 쓰기는 금지
try:
    e.add_(1)
except RuntimeError as e_:
    print("in-place 실패:", str(e_)[:50], "...")
```

broadcasting 규칙은 "**뒤 차원부터 맞춰 보고, 크기가 같거나 한쪽이 1이면 OK**"다. `(4,3) + (3,)` → OK, `(4,3) + (4,)` → 실패, `(4,3) + (4,1)` → OK.

## 7. dtype — 같은 숫자, 다른 비트 폭

![dtype](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig07.jpg)

**그림 7.** 막대 길이가 원소 하나의 비트 수다. 메모리 = `원소 수 × element_size()`. float16과 bfloat16은 둘 다 16비트지만 성격이 다르다. fp16은 가수가 길어 **정밀**하지만 **범위가 좁고**(65504 넘으면 inf), bf16은 지수가 fp32와 같아 **범위가 넓은 대신 거칠다**. 그래서 학습엔 bf16, 추론·양자화엔 fp16/int8이 자주 쓰인다(10장).

```python
for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.bool]:
    t = torch.zeros(1000, dtype=dt)
    print(f"{str(dt):15s} element_size={t.element_size()}  total={t.nbytes} bytes")

print(torch.tensor(70000., dtype=torch.float16))   # inf  — 범위 초과
print(torch.tensor(70000., dtype=torch.bfloat16))  # 70144 — 범위는 OK, 정밀도 손실
print(torch.tensor(0.1, dtype=torch.float16).item(), torch.tensor(0.1, dtype=torch.bfloat16).item())
```

타입 승격(type promotion)도 알아 두자. `int + float → float`, `float16 + float32 → float32`. 의도치 않게 fp32로 올라가 메모리가 두 배가 되는 일이 흔하다.

```python
a = torch.ones(3, dtype=torch.float16)
print((a + torch.ones(3)).dtype)        # float32 — 승격!
print((a + 1.0).dtype)                  # float16 — 파이썬 스칼라는 승격 안 시킴
```

## 8. device — 텐서가 사는 곳

![device](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig08.jpg)

**그림 8.** CPU 메모리와 GPU 메모리는 물리적으로 다른 곳이다. `.to('cuda')`는 **복사**이고, `.cpu()`나 `.item()`은 GPU 작업이 끝날 때까지 **기다리는(동기화)** 비용까지 든다. 다른 device 텐서끼리 연산하면 아래 빨간 에러가 난다. device는 문자열 `"cpu"`, `"cuda:0"`, `"mps"`, `"meta"`처럼 생겼고, 5장에서는 여기에 **내가 만든 장치 이름**을 등록하는 법을 본다.

```python
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, device=dev)
y = x.to("cpu")                     # GPU였다면 복사 + 동기화
print(x.device, y.device)

# meta 디바이스: 메모리 없이 shape/dtype만 가진 텐서 — 거대 모델 구조 확인에 유용
m = torch.empty(10_000, 10_000, device="meta")
print(m.shape, m.device)             # 실제 400MB를 할당하지 않는다
```

`meta` 텐서는 4장(ATen), 7장(export)에서 "숫자 없이 shape만 계산하는" 핵심 도구로 다시 나온다.

## 8-2. 자주 보는 에러 3개와 처방

![에러 처방](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig12.jpg)

**그림 8-2.** 처음 PyTorch를 쓰면 거의 누구나 만나는 에러 셋이에요. 각각 이 장의 단어 하나씩에 대응해요.

```python
# 1) stride 문제
x = torch.arange(6.).reshape(2, 3).t()
# x.view(6)              # 에러!
print(x.reshape(6))      # OK

# 2) device 문제
w = torch.randn(3)               # cpu
# h = w + torch.randn(3, device="cuda")   # GPU가 있으면 에러
# 처방: 모델과 입력을 같은 곳으로 → model.to(dev); x = x.to(dev)

# 3) dtype 문제
lin = torch.nn.Linear(3, 1)      # 가중치 float32
inp = torch.randn(2, 3, dtype=torch.float16)
try:
    lin(inp)
except RuntimeError as e:
    print("dtype 에러:", str(e)[:60])
print(lin(inp.float()).dtype)    # 처방: 같은 dtype으로 맞추기 → float32
```

에러 메시지를 보면 먼저 **"이건 stride·device·dtype 중 뭐지?"** 하고 분류해 보세요. 그것만으로 절반은 풀려요.

## 9. 정리 — 텐서 해부도

![해부도](/assets/img/posts/2026-10-25-pytorch-01-tensor/fig09.jpg)

**그림 9.** 앞으로 모든 장에서 이 그림을 머릿속에 띄우자.

| 필드 | 뜻 | 바꾸는 연산(복사 없음) | 복사가 생기는 연산 |
|---|---|---|---|
| storage / data_ptr | 숫자가 놓인 1차원 메모리 | — | `clone`, `contiguous`(비연속일 때) |
| shape | 보이는 모양 | `view`, `reshape`, `unsqueeze` | — |
| stride | 각 차원 한 칸 이동 시 건너뛸 원소 수 | `t`, `permute`, `expand` | — |
| storage_offset | 시작 위치 | 슬라이싱 `x[1:]` | — |
| dtype | 원소 타입 | (`view(dtype)`은 예외적으로 재해석) | `.to(dtype)`, `.half()` |
| device | 사는 장치 | — | `.to(device)`, `.cpu()` |

## 10. 직접 해 보기

```python
import torch
def describe(name, t):
    print(f"{name:10s} shape={tuple(t.shape)} stride={t.stride()} off={t.storage_offset()} "
          f"contig={t.is_contiguous()} storage={t.untyped_storage().data_ptr() % 10000}")

x = torch.arange(24.).reshape(2, 3, 4)
describe("x", x)
describe("x[:,1]", x[:, 1])
describe("permute", x.permute(2, 0, 1))
describe("unsqueeze", x.unsqueeze(0))
describe("expand", torch.ones(1, 4).expand(3, 4))
describe("contig", x.permute(2, 0, 1).contiguous())
```

출력에서 `storage`가 같은 줄은 같은 메모리다. **expand 줄(애초에 다른 텐서)과 마지막 contig 줄만 새 메모리**, 즉 실제로 복사가 일어났다**는 것을 확인하자.

### 자주 묻는 질문

- **Q. 그냥 항상 `reshape`만 쓰면 안 되나요?** 대부분 괜찮아요. 다만 `reshape`는 몰래 복사할 수 있어서, "복사 없이 원본을 공유하고 싶다"는 의도가 있을 땐 `view`가 더 정직해요(안 되면 에러로 알려 주니까요).
- **Q. `contiguous()`는 언제 불러야 하나요?** 평소엔 거의 안 불러도 돼요. 커스텀 C++/CUDA 코드에 넘길 때, `view`가 실패할 때 정도예요.
- **Q. dtype은 뭘 쓰면 되나요?** 처음엔 기본값 `float32`로 충분해요. 메모리가 부족해지면 그때 bf16/fp16을 고려하세요(10장).
- **Q. GPU가 없는데 괜찮나요?** 이 시리즈 예제는 전부 CPU로 돌아가요.

### 퀴즈
1. `torch.zeros(5, 6)[:, ::2]`의 stride는? (답: `(6, 2)`)
2. `x.t().reshape(-1)`과 `x.t().contiguous().view(-1)`의 차이는? (답: 결과는 같다. 둘 다 복사. 후자는 복사를 명시적으로 드러낸다.)
3. `torch.ones(3).expand(2,3).clone().stride()`는? (답: `(3, 1)` — clone은 연속 메모리로 실체화한다)

다음 장에서는 이 텐서들이 **모델의 파라미터**로 어떻게 묶이고, `state_dict`라는 이름표 사전으로 저장·로드되는지 본다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> → 다음: [2. 모델은 부품 상자](/blog/2026/10/25/pytorch-02-module-state-dict/)

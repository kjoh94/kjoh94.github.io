---
layout: post
title: "PyTorch 시리즈 4 — torch.mm 한 줄은 어디로 갈까: ATen과 Dispatcher 쉽게 보기"
date: 2026-10-25 21:04:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, aten, dispatcher, dispatch-key, meta, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [3. 미분은 autograd가 알아서](/blog/2026/10/25/pytorch-03-autograd/)  
> → 다음: [5. 내 장치를 PyTorch에 꽂기](/blog/2026/10/25/pytorch-05-privateuse1/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/04-aten-dispatcher](https://github.com/kjoh94/pytorch-series-examples/tree/main/04-aten-dispatcher) (`python example.py`)

> **처음이신가요?** 기초편 A–C와 1–3장을 읽었다면 준비 끝이에요. 이 장은 "파이썬 한 줄 뒤에서 실제로 계산하는 건 누구냐"를 봐요. 처음 나오는 단어는 쓰기 전에 먼저 풀어 줄게요.

지금까지 우리는 `a @ b`, `torch.mm(a, b)`, `nn.Linear` 같은 걸 그냥 썼어요. 그런데 신기하지 않나요? **똑같은 한 줄**인데 텐서가 CPU에 있으면 CPU가 계산하고, GPU에 있으면 GPU가 계산해요(1장의 device). requires_grad가 켜져 있으면 그래프도 그려 줘요(3장). 이걸 누가 골라 주는 걸까요?

답은 두 단어예요.

- **ATen** — PyTorch 안에 있는 **C++ 연산 모음집**. `mm`, `add`, `relu` 같은 연산(op) 수천 개가 여기 있어요. 파이썬 `torch.*`는 대부분 이걸 부르는 얇은 껍데기예요.
- **Dispatcher** — ATen 안의 **안내 데스크**. 주문(op 이름)을 받으면 텐서에 붙은 꼬리표를 보고 알맞은 주방(커널)으로 보내요.

> **커널(kernel)이 뭐예요?** 특정 장치에서 특정 연산을 실제로 계산하는 함수예요. "CPU용 행렬곱 함수", "GPU용 행렬곱 함수"가 각각 하나의 커널이에요.

5장(새 장치 꽂기), 6–7장(그래프 캡처·export), 12장(커스텀 op)이 모두 이 구조 위에 있어서, 시리즈에서 가장 중요한 장이라고 해도 좋아요.

## 0. 비유로 먼저 — 같은 주문, 다른 주방

![식당 비유](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig08.jpg)

**그림 0.** 식당을 떠올려 보세요. 손님은 메뉴판의 이름("mm 하나요")만 말해요. 안내 데스크(Dispatcher)는 손님의 꼬리표를 보고 "CPU 주방", "GPU 주방", 혹은 5장에서 우리가 직접 만들 "내 장치 주방"으로 주문을 넘겨요. **메뉴 이름은 하나, 요리법은 주방마다 따로** — 이게 이 장의 전부예요.

## 0-2. 한 줄이 내려가는 층

![층](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig09.jpg)

**그림 0-2.** 조금 더 정확히 보면, `torch.mm(a, b)`는 파이썬 → C++ 연결(바인딩) → Dispatcher → (필요하면) Autograd 정거장 → 실제 커널 순서로 내려가요. 우리는 맨 위 한 줄만 쓰고, 나머지는 PyTorch가 해요. 3장에서 본 그래프 기록도 사실 이 길 중간의 한 정거장이에요.

### 직접 보기 — profiler로 op 이름 찍어 보기

![profiler](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig10.jpg)

**그림 0-3.** 말로만 들으면 와닿지 않으니 직접 확인해 봐요. `torch.profiler`로 감싸면 실제로 불린 ATen op 이름이 나와요. `nn.Linear` 하나가 `aten::linear` → `aten::addmm`(곱하고 더하기) 같은 op로 바뀌는 게 보여요.

```python
import torch, torch.nn as nn
x = torch.randn(2, 4)
lin = nn.Linear(4, 3)
with torch.profiler.profile() as prof:
    lin(x)
print(prof.key_averages().table(row_limit=6))   # aten::linear, aten::addmm, aten::t ...
```

> **1장과 이어지는 장면:** 표에 `aten::t`, `aten::transpose`, `aten::as_strided`도 보여요. weight를 뒤집는(transpose) 데 숫자를 옮기지 않고 stride만 바꾸는 view 연산 — 1장에서 본 바로 그거예요.

> **쉽게 말하면:** `aten::`이 붙은 이름이 안내 데스크가 아는 "메뉴 이름"이에요. 6·7장의 그래프에도 이 이름들이 그대로 찍혀요.

## 1. 큰 그림

![큰 그림](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig01.jpg)

**그림 1.** 위에서 아래로 읽자. 파이썬의 `torch.mm`은 자동 생성된 C++ 바인딩을 거쳐 **Dispatcher**에 도착한다. Dispatcher는 입력 텐서들이 가진 **dispatch key**(꼬리표)를 모아 가장 우선순위가 높은 키의 커널을 부른다. 텐서가 `requires_grad`면 먼저 **Autograd 커널**이 불려 그래프를 기록하고, 키를 하나 내려 다시 디스패치하면 마지막에 **백엔드 커널**(CPU면 BLAS 행렬곱)이 실제 계산을 한다.

```python
import torch
a, b = torch.randn(3, 4), torch.randn(4, 5)
print(torch.ops.aten.mm(a, b).shape)          # torch.mm 과 같은 op를 '이름'으로 호출
print(torch.ops.aten.mm.default._schema)     # aten::mm(Tensor self, Tensor mat2) -> Tensor
```

`torch.ops.aten.*`은 ATen op를 직접 부르는 창구다. 6–7장의 그래프에 찍히는 노드 이름(`aten.mm.default`)이 바로 이것이다.

> **스키마(schema)가 뭐예요?** 메뉴판의 한 줄이에요. "이름은 mm, 텐서 두 개 받고, 텐서 하나 돌려줌." 어느 주방에서 만들든 이 약속은 같아요.

## 2. 스키마와 커널은 따로다

![스키마와 커널](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig02.jpg)

**그림 2.** PyTorch 소스의 `aten/src/ATen/native/native_functions.yaml`에는 op마다 **스키마**(이름, 인자 타입, 반환 타입) 한 줄과, 백엔드별로 어떤 C++ 함수가 구현하는지 적힌 `dispatch:` 표가 있다. 스키마는 하나, 커널은 여러 개. 새 장치를 지원한다는 것은 **스키마를 바꾸지 않고 내 장치용 커널을 그 표에 추가 등록**하는 일이다(5장).

```python
op = torch.ops.aten.add.Tensor
print(op._schema)       # aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# op 이름의 점 뒤는 overload 이름: add.Tensor, add.Scalar, add.out ...
print([o for o in torch.ops.aten.add.overloads()][:5])

# 이 op에 어떤 키로 커널이 등록되어 있나
for key in ["CPU", "CUDA", "Meta", "AutogradCPU", "PrivateUse1"]:
    print(key, torch._C._dispatch_has_kernel_for_dispatch_key("aten::add.Tensor", key))
```

## 3. dispatch key — 텐서의 꼬리표

![dispatch key](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig03.jpg)

**그림 3.** 모든 텐서는 **DispatchKeySet**(키 묶음)을 들고 다닌다. CPU 텐서는 `CPU`, `AutogradCPU`, `ADInplaceOrView`, `AutocastCPU` 같은 키를 가진다. Dispatcher는 입력 전체의 키를 합친 뒤 **우선순위가 높은 것부터**(위→아래) 처리한다. 각 층은 자기 일만 하고, "내 키를 빼고" 다시 디스패치한다. 이 계단 구조 덕분에 autograd·autocast·추적(tracing) 기능이 백엔드와 독립적으로 붙는다.

```python
x = torch.randn(2)
print(torch._C._dispatch_keys(x))
y = torch.randn(2, requires_grad=True)
print(torch._C._dispatch_keys(y))
m = torch.empty(2, device="meta")
print(torch._C._dispatch_keys(m))   # Meta 키
```

## 4. 한 번의 호출, 여러 정거장

![정거장](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig04.jpg)

**그림 4.** `requires_grad` CPU 텐서로 `mm`을 부르면: Autocast 정거장(켜져 있으면 dtype을 낮춰 재호출) → **Autograd 정거장**(`MmBackward0` 노드를 붙이고 키를 내려 재호출) → **CPU 커널**(실제 GEMM). 3장의 `grad_fn`은 바로 이 Autograd 정거장에서 생긴다.

```python
a = torch.randn(3, 4, requires_grad=True)
z = torch.mm(a, torch.randn(4, 5))
print(z.grad_fn)                     # <MmBackward0> — Autograd 정거장의 흔적

# 정거장을 직접 확인: 디스패처 전체 등록 표 일부 출력
dump = torch._C._dispatch_dump("aten::mm")
for line in dump.splitlines():
    if line.split(":")[0] in ("CPU", "Meta", "AutogradCPU", "AutocastCPU", "CompositeExplicitAutograd"):
        print(line[:90])
```

## 5. 할당 op와 계산 op

![할당 vs 계산](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig05.jpg)

**그림 5.** ATen op는 크게 두 부류로 생각하면 편하다. **메모리 계열**(`empty`, `empty_strided`, `copy_`, `resize_`)은 텐서를 만들고 옮기는 기초라 새 장치라면 반드시 있어야 한다. `view`, `transpose` 같은 view 계열은 1장에서 봤듯 메타데이터만 바꾸므로 대부분 공용 구현이 그대로 쓰인다. **계산 계열**(`add`, `mm`, `softmax` …)은 하나씩 채워 나간다.

```python
# 모든 텐서 생성의 바닥에는 empty 계열 op가 있다
from torch.utils._python_dispatch import TorchDispatchMode
class LogOps(TorchDispatchMode):
    def __init__(self): super().__init__(); self.ops = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))

with LogOps() as log:
    t = torch.zeros(3, 4)
    u = t.t()                # view
    v = u.contiguous()       # 복사 필요
print(log.ops)               # zeros, t, clone 등 — 실제로 내려간 aten op
```

## 6. Composite op — 분해되는 op

![composite](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig06.jpg)

**그림 6.** 모든 op가 자기 커널을 갖지는 않는다. `F.linear`는 **CompositeImplicitAutograd** op라서 디스패치 전에 `t` + `addmm` 같은 더 작은 op로 **분해**된다. 덕분에 새 백엔드는 "기본 op"만 구현하면 상위 op가 공짜로 따라온다. 반대로 **CompositeExplicitAutograd**는 "모든 백엔드 공용 커널이 있지만 autograd 공식은 따로 있는" op다.

```python
import torch.nn.functional as F
with LogOps() as log:
    F.linear(torch.randn(2, 3), torch.randn(4, 3), torch.randn(4))
print(log.ops[-2:])          # ['aten.t.default', 'aten.addmm.default']

with LogOps() as log:
    F.gelu(torch.randn(3)); torch.nn.functional.silu(torch.randn(3)); F.layer_norm(torch.randn(2, 4), (4,))
print(log.ops)
```

어떤 op가 분해되고 어떤 op가 커널로 가는지가 바로 **"백엔드가 구현해야 할 op 목록"** 을 결정한다. 7장의 **Core ATen IR**은 이 분해를 끝까지 해서 남는 약 180여 개의 기본 op 집합이다.

## 7. meta 커널 — 숫자 없이 shape 계산

![meta](/assets/img/posts/2026-10-25-pytorch-04-aten-dispatcher/fig07.jpg)

**그림 7.** 많은 op에는 **Meta 커널**이 있다. 숫자는 계산하지 않고 출력의 shape·dtype·stride만 계산한다. 메모리를 안 쓰니 70B 모델 구조도 노트북에서 "가상 실행"할 수 있다. `torch.compile`의 FakeTensor(6장)와 `torch.export`(7장)가 그래프를 추적할 때 이 커널을 쓴다.

```python
a = torch.empty(3, 4, device="meta")
b = torch.empty(4, 5, device="meta")
c = a @ b
print(c.shape, c.device)                                   # (3, 5) meta

big = torch.nn.Linear(50_000, 50_000, device="meta")       # 10GB 짜리지만 메모리 0
print(big(torch.empty(8, 50_000, device="meta")).shape)
```

## 8. 내 커널을 디스패처에 등록해 보기 (맛보기)

12장에서 자세히 다루지만, 파이썬에서도 디스패처 표에 커널을 넣을 수 있다.

```python
lib = torch.library.Library("demo", "DEF")
lib.define("scale2(Tensor x) -> Tensor")
def scale2_cpu(x): return x * 2
def scale2_meta(x): return torch.empty_like(x)
lib.impl("scale2", scale2_cpu, "CPU")
lib.impl("scale2", scale2_meta, "Meta")

print(torch.ops.demo.scale2(torch.ones(3)))
print(torch.ops.demo.scale2(torch.empty(3, device="meta")).device)
print(torch._C._dispatch_has_kernel_for_dispatch_key("demo::scale2", "CPU"))
```

스키마 하나(`define`)와 키별 커널(`impl`) — 그림 2의 구조 그대로다.

## 자주 묻는 질문

- **Q. 이걸 몰라도 PyTorch를 쓸 수 있나요?** 네, 평소엔 몰라도 돼요. 다만 "이 장치에서 이 op가 지원 안 됨"(`NotImplementedError: Could not run 'aten::...' with arguments from the '...' backend`) 같은 에러를 읽거나, 새 장치·커스텀 op를 붙일 때(5·12장) 꼭 필요해요.
- **Q. 그 에러는 무슨 뜻이에요?** "그 주방에는 그 메뉴 요리법이 없어요"라는 뜻이에요. 그 키로 커널이 등록되지 않은 거예요.
- **Q. `torch.ops.aten.mm`과 `torch.mm`은 다른가요?** 같은 op예요. 앞의 것은 메뉴 이름으로 직접 부르는 창구예요.
- **Q. meta 장치는 왜 있어요?** 숫자 계산 없이 "결과 모양만" 알고 싶을 때 쓰는 가짜 주방이에요(7절). 큰 모델 구조 확인, 컴파일러의 모양 추론에 써요.

### 퀴즈
1. 같은 op가 CPU/GPU에서 다르게 계산되는 건 무엇 덕분인가? (답: 텐서의 dispatch key를 보고 Dispatcher가 커널을 고르기 때문)
2. `requires_grad=True` 텐서로 `mm`을 부르면 백엔드 커널보다 먼저 들르는 정거장은? (답: Autograd)
3. 새 장치를 지원하려면 스키마를 바꿔야 하나? (답: 아니요, 그 장치 키로 커널만 등록)

## 정리

- **ATen** = PyTorch의 C++ op 라이브러리. 각 op는 **스키마 하나 + 백엔드별 커널 여러 개**.
- **Dispatcher**는 텐서의 **dispatch key** 묶음을 보고 우선순위대로 정거장(Autocast → Autograd → 백엔드)을 거친다.
- Composite op는 기본 op로 분해되고, Meta 커널은 shape만 계산한다.
- 다음 장: 이 표의 빈 칸 **PrivateUse1**에 **내 장치**의 커널을 채우는 방법.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [3. 미분은 autograd가 알아서](/blog/2026/10/25/pytorch-03-autograd/)  
> → 다음: [5. 내 장치를 PyTorch에 꽂기](/blog/2026/10/25/pytorch-05-privateuse1/)

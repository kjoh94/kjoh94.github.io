---
layout: post
title: "PyTorch 시리즈 4 — ATen과 Dispatcher: torch.mm 한 줄이 커널까지 가는 길"
date: 2026-10-25 21:04:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, aten, dispatcher, dispatch-key, meta, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [3. Autograd와 추론 모드](/blog/2026/10/25/pytorch-03-autograd/)  
> → 다음: [5. 커스텀 디바이스 백엔드 (PrivateUse1)](/blog/2026/10/25/pytorch-05-privateuse1/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/04-aten-dispatcher](https://github.com/kjoh94/pytorch-series-examples/tree/main/04-aten-dispatcher) (`python example.py`)

같은 `torch.mm(a, b)`가 CPU 텐서면 CPU 행렬곱을, CUDA 텐서면 GPU 행렬곱을 부른다. 누가 고르는 걸까? 답은 **ATen**(PyTorch의 C++ 텐서 연산 라이브러리)과 그 안의 **Dispatcher**(op 이름 → 커널 라우터)다. 이 장은 [초기 정리본](/blog/2026/10/25/pytorch-aten-dispatcher/)을 시리즈 형식으로 다시 풀어 쓴 것이다. 5장(새 장치 꽂기), 6–7장(그래프 캡처·export), 12장(커스텀 op)이 모두 이 구조 위에 있으니 시리즈에서 가장 중요한 장이라고 해도 좋다.

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

## 정리

- **ATen** = PyTorch의 C++ op 라이브러리. 각 op는 **스키마 하나 + 백엔드별 커널 여러 개**.
- **Dispatcher**는 텐서의 **dispatch key** 묶음을 보고 우선순위대로 정거장(Autocast → Autograd → 백엔드)을 거친다.
- Composite op는 기본 op로 분해되고, Meta 커널은 shape만 계산한다.
- 다음 장: 이 표의 빈 칸 **PrivateUse1**에 **내 장치**의 커널을 채우는 방법.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [3. Autograd와 추론 모드](/blog/2026/10/25/pytorch-03-autograd/)  
> → 다음: [5. 커스텀 디바이스 백엔드 (PrivateUse1)](/blog/2026/10/25/pytorch-05-privateuse1/)

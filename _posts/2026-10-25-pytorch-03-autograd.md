---
layout: post
title: "PyTorch 시리즈 3 — Autograd와 추론 모드: eval·no_grad·inference_mode"
date: 2026-10-25 21:03:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, autograd, inference_mode, no_grad, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [2. nn.Module과 state_dict](/blog/2026/10/25/pytorch-02-module-state-dict/)  
> → 다음: [4. ATen과 Dispatcher](/blog/2026/10/25/pytorch-04-aten-dispatcher/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/03-autograd](https://github.com/kjoh94/pytorch-series-examples/tree/main/03-autograd) (`python example.py`)

학습할 땐 gradient가 필요하고, 추론할 땐 필요 없다. 말은 쉽지만 `model.eval()`, `torch.no_grad()`, `torch.inference_mode()`가 각각 **무엇을** 끄는지 헷갈리는 경우가 많다. 이 장에서는 autograd가 forward 중에 몰래 그리는 그래프를 직접 들여다보고, 추론 모드가 그 그래프와 메모리를 어떻게 아끼는지 본다. 6·7장의 컴파일러가 "AOTAutograd"라는 단계를 두는 이유도 여기서 나온다.

## 1. forward는 그래프를 그린다

![계산 그래프](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig01.jpg)

**그림 1.** `requires_grad=True`인 텐서(노랑 `w`, `b`)가 연산에 들어가면, 결과 텐서에 **`grad_fn`** 이라는 노드가 붙는다. `MulBackward0`, `AddBackward0`이 그것이다. 이 노드들이 화살표로 이어진 것이 계산 그래프(DAG)다. forward는 왼쪽→오른쪽으로 값을 계산하면서 그래프를 기록하고, `backward()`는 오른쪽→왼쪽으로 거슬러 가며 gradient를 전달한다.

```python
import torch
x = torch.tensor(3.0)                       # 입력: grad 필요 없음
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = w * x + b
L = y ** 2
print(y.grad_fn, L.grad_fn)                 # <AddBackward0> <PowBackward0>
print(L.grad_fn.next_functions)             # 거꾸로 이어진 노드들
print(x.is_leaf, w.is_leaf, y.is_leaf)      # True True False
```

## 2. backward는 연쇄법칙을 거꾸로 돈다

![연쇄법칙](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig02.jpg)

**그림 2.** 손으로 계산해 보자. `y = 2·3 + 1 = 7`, `L = 49`. `dL/dy = 2y = 14`, `dL/dw = 14·x = 42`, `dL/db = 14`. 각 `grad_fn` 노드는 "출력 쪽 gradient를 받아 입력 쪽 gradient를 계산하는 법"만 안다. 엔진은 이 노드들을 역순으로 호출할 뿐이다. 결과는 leaf 텐서의 `.grad`에 **더해진다(누적)** — 그래서 학습 루프에 `zero_grad()`가 있다.

```python
L.backward()
print(w.grad, b.grad)        # tensor(42.) tensor(14.)

# 누적 확인
y = w * x + b; (y ** 2).backward()
print(w.grad)                # tensor(84.) — 42가 또 더해짐
w.grad = None                # 또는 optimizer.zero_grad()
```

## 3. 그래프는 메모리를 먹는다

![활성값 저장](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig03.jpg)

**그림 3.** 곱셈의 미분에는 상대 입력값이 필요하다(`d(w·x)/dw = x`). 그래서 autograd는 forward 중 각 층의 **중간 활성값을 저장**해 둔다(노란 상자). 큰 모델에서 학습 메모리의 상당 부분이 이것이다. 추론만 할 거라면 전부 낭비다.

```python
import torch.nn as nn
net = nn.Sequential(*[nn.Linear(256, 256) for _ in range(8)])
inp = torch.randn(64, 256)

out = net(inp)
n_saved = 0
fn = out.grad_fn
seen, stack = set(), [fn]
while stack:                         # 그래프를 따라가며 저장된 텐서 수 세기
    f = stack.pop()
    if f is None or f in seen: continue
    seen.add(f)
    n_saved += sum(1 for a in dir(f) if a.startswith("_saved_"))
    stack += [g for g, _ in f.next_functions]
print("grad_fn 노드 수:", len(seen), " 저장 슬롯 수:", n_saved)

with torch.no_grad():
    out2 = net(inp)
print(out2.grad_fn)                  # None — 그래프 없음
```

## 4. 세 가지 스위치

![세 스위치](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig04.jpg)

**그림 4.** 표를 가로로 읽자.

- **`model.eval()`**: Dropout을 끄고 BatchNorm이 저장된 통계를 쓰게 한다. **그래프 기록은 계속된다.**
- **`torch.no_grad()`**: 그래프 기록을 끈다. 결과 텐서는 `requires_grad=False`.
- **`torch.inference_mode()`**: no_grad에 더해 텐서의 버전 카운터·view 추적 같은 autograd 부가 장치까지 끈다. 가장 가볍다.

추론 코드의 표준은 **`model.eval()` + `with torch.inference_mode():`** 둘 다다. 하나는 모듈 동작, 하나는 autograd다.

```python
import time
net = nn.Sequential(nn.Linear(512, 512), nn.Dropout(0.5), nn.Linear(512, 512))
x = torch.randn(256, 512)

net.train()
print("train 모드 두 번 결과 같나?", torch.equal(net(x), net(x)))     # False (dropout)
net.eval()
print("eval 모드 두 번 결과 같나?", torch.equal(net(x), net(x)))      # True
print("eval 모드에서도 grad_fn:", net(x).grad_fn is not None)          # True!

def bench(ctx):
    t = time.perf_counter()
    for _ in range(200):
        with ctx():
            net(x)
    return time.perf_counter() - t
import contextlib
print(f"기본       {bench(contextlib.nullcontext):.3f}s")
print(f"no_grad    {bench(torch.no_grad):.3f}s")
print(f"inference  {bench(torch.inference_mode):.3f}s")
```

## 5. inference_mode의 대가

![inference_mode 제약](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig05.jpg)

**그림 5.** `inference_mode` 안에서 만든 텐서는 "inference 텐서"로 표시되어, 나중에 autograd가 필요한 연산에 **다시 넣을 수 없다.** 캐시를 만들어 두고 학습에 재사용하려 할 때 이 에러를 만난다. `clone()`으로 일반 텐서를 만들면 해결된다.

```python
lin = nn.Linear(3, 3)
with torch.inference_mode():
    feat = lin(torch.randn(2, 3))
print(feat.is_inference())                      # True
w2 = torch.randn(3, requires_grad=True)
try:
    (feat * w2).sum().backward()
except RuntimeError as e:
    print("에러:", str(e)[:70], "...")
(feat.clone() * w2).sum().backward()            # OK
print(w2.grad)
```

## 6. 컴파일러와 autograd

![AOTAutograd](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig06.jpg)

**그림 6.** `torch.compile`로 **학습**을 가속할 때는 forward만 잡아서는 안 된다. backward도 최적화해야 한다. 그래서 그래프 캡처(6장) 뒤에 **AOTAutograd**가 forward 그래프를 미리 미분해 backward 그래프를 만들고, 어떤 활성값을 저장할지(또는 다시 계산할지)까지 계획한다. 반대로 **추론용 export(7장)** 는 grad 노드가 없는 순수 forward 그래프만 다룬다 — 바로 inference_mode가 만들어 내는 세계다.

```python
# 직접 미분 규칙 정의하기: autograd.Function
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)     # 그림 3의 '저장'을 직접 하는 셈
        return x * x
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * 2 * x
t = torch.tensor([1., 2., 3.], requires_grad=True)
Square.apply(t).sum().backward()
print(t.grad)                       # tensor([2., 4., 6.])
print(torch.autograd.gradcheck(Square.apply, torch.randn(3, dtype=torch.double, requires_grad=True)))
```

`autograd.Function`은 12장에서 커스텀 커널에 미분 규칙을 붙일 때 다시 등장한다.

## 정리

- `requires_grad` 텐서에서 시작한 연산은 `grad_fn` 노드를 남겨 그래프를 만든다.
- backward는 그 그래프를 역순으로 돌며 연쇄법칙을 적용하고, `.grad`에 **누적**한다.
- 그래프는 중간 활성값을 저장하므로 메모리를 먹는다.
- `eval()`은 모듈 동작, `no_grad`/`inference_mode`는 autograd. **추론엔 둘 다.**

다음 장에서는 `w * x` 같은 한 줄이 실제로 어떤 C++ 커널까지 내려가는지 — **ATen과 Dispatcher** — 를 본다. autograd도 사실 그 길 위의 한 정거장이다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [2. nn.Module과 state_dict](/blog/2026/10/25/pytorch-02-module-state-dict/)  
> → 다음: [4. ATen과 Dispatcher](/blog/2026/10/25/pytorch-04-aten-dispatcher/)

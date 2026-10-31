---
layout: post
title: "PyTorch 시리즈 3 — 미분은 autograd가 알아서: 학습과 추론 모드(eval·no_grad·inference_mode)"
date: 2026-10-25 21:03:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, autograd, inference_mode, no_grad, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [2. 모델은 부품 상자](/blog/2026/10/25/pytorch-02-module-state-dict/)  
> → 다음: [4. ATen과 Dispatcher](/blog/2026/10/25/pytorch-04-aten-dispatcher/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/03-autograd](https://github.com/kjoh94/pytorch-series-examples/tree/main/03-autograd) (`python example.py`)

> **처음이신가요?** 기초편 [C. loss와 학습 루프](/blog/2026/10/31/pytorch-00c-loss-train-loop/)에서 `loss.backward()`를 한 줄 썼죠. 이 장은 그 한 줄 뒤에서 무슨 일이 일어나는지 봐요. 텐서(1장)와 Parameter(2장)를 알면 충분해요.

기초편 C의 학습 루프를 떠올려 볼게요. 우리는 `loss.backward()` 한 줄만 썼는데, 갑자기 모든 Parameter에 `.grad`가 채워졌어요. 미분 공식을 하나도 안 썼는데 말이에요. 이걸 해 주는 게 **autograd(자동 미분)** 예요.

이 장에서는 세 가지를 차근차근 볼게요.

1. **gradient가 뭔지** — 수학 없이 "기울기" 감각으로
2. **autograd가 어떻게 알아서 미분하는지** — forward 중에 몰래 그리는 "계산 그래프"
3. **추론할 땐 그걸 어떻게 끄는지** — `model.eval()`, `torch.no_grad()`, `torch.inference_mode()`가 각각 **무엇을** 끄는지

세 번째가 실전에서 제일 헷갈리는 부분이에요. 그리고 6·7장의 컴파일러가 "AOTAutograd"라는 단계를 두는 이유도 여기서 나와요.

## 0. gradient = 지금 위치의 기울기

![기울기](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig07.jpg)

**그림 0.** loss를 언덕이라고 생각해 보세요. 우리는 가장 낮은 곳(loss가 작은 곳)에 가고 싶어요. 지금 서 있는 곳의 **기울기**가 +4라면 "오른쪽으로 가면 올라간다"는 뜻이니 왼쪽으로 가면 돼요. 그래서 기초편 C의 갱신 규칙이 `w ← w − lr × gradient`예요. **gradient는 "이 숫자를 조금 늘리면 loss가 얼마나 변하나"** 를 알려 주는 값일 뿐이에요.

```python
import torch
w = torch.tensor(2.0, requires_grad=True)
loss = w ** 2          # 언덕: loss = w²
loss.backward()
print(w.grad)          # tensor(4.) — w=2에서 기울기 4
with torch.no_grad():
    w -= 0.1 * w.grad  # 한 걸음 내려가기
print(w)               # 1.6 — 바닥(0) 쪽으로 이동
```

> **쉽게 말하면:** 숫자가 수백만 개여도 똑같아요. 숫자마다 "늘리면 loss가 오르나 내리나, 얼마나?"를 하나씩 구한 게 `.grad`예요.

## 0-2. 학습 루프에서 autograd의 자리

![학습 루프](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig08.jpg)

**그림 0-2.** 학습 루프 네 줄 중 **우리가 미분에 관여하는 건 `backward()` 한 줄**뿐이에요. 대신 autograd는 forward(`model(x)`)를 하는 동안 "무슨 계산을 했는지" 기록해 둬요. 그 기록이 있어야 거꾸로 돌아가며 미분할 수 있거든요. 다음 절에서 그 기록 — 계산 그래프 — 을 직접 들여다봐요.

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

> **용어 정리:** `requires_grad=True`인 텐서 = "미분을 구해 줘" 표시. Parameter는 기본으로 켜져 있어요(2장). **leaf** = 사용자가 직접 만든 텐서(그래프의 시작점), `.grad`는 leaf에만 채워져요.

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

> **가장 흔한 실수 — `zero_grad()` 빼먹기:** `.grad`는 덮어쓰지 않고 **더해져요.** 루프마다 비우지 않으면 gradient가 계속 쌓여서 학습이 엉뚱하게 튀어요. 기초편 C의 루프에 `opt.zero_grad()`가 있던 이유예요.

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

### 그래서 뭘 쓰면 되나요?

![스위치 고르기](/assets/img/posts/2026-10-25-pytorch-03-autograd/fig09.jpg)

**그림 4-2.** 상황별로 외워 두면 편해요.

```python
# 학습 중간에 검증
model.eval()
with torch.no_grad():
    val_loss = loss_fn(model(x_val), y_val)
model.train()          # ← 다시 학습 모드로! 빼먹기 쉬움

# 배포·순수 추론
model.eval()
with torch.inference_mode():
    pred = model(x)
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

## 자주 묻는 질문

- **Q. `.grad`가 `None`이에요.** ① `backward()`를 아직 안 불렀거나, ② 그 텐서가 `requires_grad=False`거나, ③ leaf가 아닌 중간 텐서예요(중간값이 필요하면 `y.retain_grad()`).
- **Q. `backward()`를 두 번 부르면 에러가 나요.** 첫 backward가 끝나면 저장된 중간값을 지워요(메모리 절약). 정말 필요하면 `backward(retain_graph=True)`.
- **Q. Parameter를 직접 고치려니 에러가 나요.** `w -= ...` 같은 직접 수정은 `with torch.no_grad():` 안에서 하세요. 그 수정까지 그래프에 기록되면 안 되니까요.
- **Q. `.detach()`는 뭐예요?** "같은 숫자(같은 메모리, 1장!)인데 그래프에서 떼어 낸 텐서"예요. 로그 찍기, numpy 변환 전에 써요.

### 퀴즈
1. `w=3`, `L = w**2`일 때 `w.grad`는? (답: `6`)
2. `model.eval()`만 하고 `no_grad`를 안 쓰면 그래프가 생길까? (답: 생긴다)
3. 루프에서 `zero_grad()`를 빼면 두 번째 step의 grad는? (답: 1·2번째 gradient의 합)

## 정리

- `requires_grad` 텐서에서 시작한 연산은 `grad_fn` 노드를 남겨 그래프를 만든다.
- backward는 그 그래프를 역순으로 돌며 연쇄법칙을 적용하고, `.grad`에 **누적**한다.
- 그래프는 중간 활성값을 저장하므로 메모리를 먹는다.
- `eval()`은 모듈 동작, `no_grad`/`inference_mode`는 autograd. **추론엔 둘 다.**

다음 장에서는 `w * x` 같은 한 줄이 실제로 어떤 C++ 커널까지 내려가는지 — **ATen과 Dispatcher** — 를 본다. autograd도 사실 그 길 위의 한 정거장이다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [2. 모델은 부품 상자](/blog/2026/10/25/pytorch-02-module-state-dict/)  
> → 다음: [4. ATen과 Dispatcher](/blog/2026/10/25/pytorch-04-aten-dispatcher/)

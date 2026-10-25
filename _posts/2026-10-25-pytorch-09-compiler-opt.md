---
layout: post
title: "PyTorch 시리즈 9장 — 컴파일러 최적화 감각: folding·fusion·메모리"
date: 2026-10-25 21:09:00 +0900
categories: [pytorch]
tags: [pytorch, compiler, fusion]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [8. PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/)  
> → 다음: [10. 양자화, PyTorch 쪽에서](/blog/2026/10/25/pytorch-10-quantization/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/09-compiler-opt](https://github.com/kjoh94/pytorch-series-examples/tree/main/09-compiler-opt) (`python example.py`)

그래프를 손에 넣었다(6–8장). 이제 컴파일러가 그 그래프로 **무엇을** 하는지 감을 잡을 차례다. 이 장은 세 가지 대표 최적화 — constant folding, operator fusion, memory planning — 를 작은 FX 패스로 직접 구현해 본다.

## 1. 컴파일러가 하는 일 세 가지

![three](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig01.jpg)

**그림 1.** ① 입력과 무관한 계산은 미리 해 둔다(folding). ② 여러 op를 한 커널로 합쳐 메모리 왕복을 줄인다(fusion). ③ 수명이 겹치지 않는 텐서끼리 메모리를 나눠 쓴다(planning). 셋 다 "계산보다 메모리가 비싸다"는 현실에서 나온다.

## 2. constant folding

![folding](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig02.jpg)

**그림 2.** `x @ (W*scale).T`에서 `W`와 `scale`은 상수다. 매번 곱하고 전치할 필요 없이 `W'`을 한 번 만들어 두면 실행 시엔 matmul 하나뿐이다. BatchNorm을 앞 Linear/Conv 가중치에 접어 넣는 **BN folding**도 같은 원리다 — 직접 해 보자.

```python
import torch, time
torch.manual_seed(0)

lin = torch.nn.Linear(8, 4); bn = torch.nn.BatchNorm1d(4)
bn.running_mean.uniform_(-1, 1); bn.running_var.uniform_(0.5, 2)
seq = torch.nn.Sequential(lin, bn).eval()

# BN(y) = (y - mean)/sqrt(var+eps)*gamma + beta  →  y = W x + b 에 접어 넣기
s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
fused = torch.nn.Linear(8, 4)
with torch.no_grad():
    fused.weight.copy_(lin.weight * s[:, None])
    fused.bias.copy_((lin.bias - bn.running_mean) * s + bn.bias)

x = torch.randn(5, 8)
with torch.no_grad():
    print("BN folding 오차:", (seq(x) - fused(x)).abs().max().item())
```

## 3. 왜 fusion인가: 메모리가 병목

![fusion why](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig03.jpg)

**그림 3.** `relu → +1 → *2`를 따로 돌리면 op마다 전체 텐서를 읽고 쓴다: 메모리 왕복 6번. 합치면 한 번 읽고 한 번 쓴다: 2번. elementwise op는 바이트당 연산이 적어(**arithmetic intensity**가 낮아) 메모리 대역폭에 묶인다. 그래서 fusion이 효과가 크다.

```python
def chain(x):
    return (x.relu() + 1) * 2

x = torch.randn(4_000_000)
fc = torch.compile(chain)          # Inductor 가 한 커널로 합친다
fc(x)                              # 첫 호출 = 컴파일
for name, fn in [("eager", chain), ("compiled", fc)]:
    t = time.perf_counter()
    for _ in range(20): fn(x)
    print(f"{name:9s} {(time.perf_counter() - t) / 20 * 1e3:.2f} ms")
print("same:", torch.allclose(chain(x), fc(x)))
```

(숫자는 기계마다 다르다. CPU에서도 대개 compiled가 빠르다.)

## 4. fusion 패턴

![patterns](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig04.jpg)

**그림 4.** 합칠 수 있는 대표 패턴: elementwise 연쇄, matmul/conv 뒤에 bias·활성화를 붙이는 **epilogue fusion**, reduction과 elementwise(softmax, layernorm), 그리고 attention 전체(FlashAttention 류). 아래는 FX로 "연속된 elementwise op"를 찾아내는 작은 패스다 — 컴파일러의 fusion 그룹핑이 하는 일의 축소판이다.

```python
from torch.fx import symbolic_trace

ELEMENTWISE = {torch.relu, torch.sigmoid, torch.tanh, torch.add, torch.mul,
               "relu", "add", "mul", "sigmoid"}
import operator
ELEMENTWISE |= {operator.add, operator.mul}

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fc = torch.nn.Linear(4, 4)
    def forward(self, x):
        y = self.fc(x)
        return torch.sigmoid(torch.relu(y) + 1) * 2

gm = symbolic_trace(Net())
groups, cur = [], []
for n in gm.graph.nodes:
    if n.op in ("call_function", "call_method") and n.target in ELEMENTWISE:
        cur.append(n.name)
    else:
        if cur: groups.append(cur); cur = []
if cur: groups.append(cur)
print("fusion 후보 그룹:", groups)    # [['relu', 'add', 'sigmoid', 'mul']]
```

## 5. memory planning

![memory](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig05.jpg)

**그림 5.** 그래프의 각 중간 텐서는 만들어지는 순간부터 마지막으로 쓰이는 순간까지 산다. 수명이 겹치지 않으면 같은 버퍼를 다시 쓸 수 있다. 그림에서 a와 c, b와 d가 겹치지 않으니 버퍼 2개로 충분하다. 아래 코드는 FX 그래프에서 수명을 계산하고 탐욕적으로 버퍼를 배정한다.

```python
class Deep(torch.nn.Module):
    def forward(self, x):
        a = x.relu(); b = a + 1; c = b * 2; d = c.sigmoid(); return d - 1

gm = symbolic_trace(Deep())
nodes = list(gm.graph.nodes)
order = {n: i for i, n in enumerate(nodes)}
life = {n.name: (order[n], max([order[u] for u in n.users] or [order[n]]))
        for n in nodes if n.op in ("call_function", "call_method")}
free_at, assign = [], {}            # 버퍼별 '비는 시점'
for name, (s, e) in sorted(life.items(), key=lambda kv: kv[1][0]):
    for i, f in enumerate(free_at):
        if f < s:                   # 이미 비었으면 재사용
            assign[name] = i; free_at[i] = e; break
    else:
        assign[name] = len(free_at); free_at.append(e)
print("수명:", life)
print("버퍼 배정:", assign, "→ 버퍼", len(free_at), "개 (텐서", len(life), "개)")
```

## 6. 직접 보기

![inductor](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig06.jpg)

**그림 6.** Inductor가 실제로 만든 코드는 `TORCH_LOGS=output_code python example.py`로 볼 수 있다. 위 `chain`이 커널 하나가 되는 걸 확인해 보자. 내 컴파일러 백엔드(6장 그림 7)를 만든다면, 위의 그룹핑·수명 분석이 바로 출발점이 된다.

## 정리

- folding: 상수 부분은 미리. BN folding이 대표 예.
- fusion: 메모리 왕복을 줄인다. elementwise는 memory-bound.
- planning: 수명이 안 겹치면 버퍼 공유.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [8. PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/)  
> → 다음: [10. 양자화, PyTorch 쪽에서](/blog/2026/10/25/pytorch-10-quantization/)

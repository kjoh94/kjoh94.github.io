---
layout: post
title: "PyTorch 시리즈 9장 — 컴파일러는 무엇을 빠르게 하나: 미리 계산·합치기·메모리 아끼기"
date: 2026-10-25 21:09:00 +0900
categories: [pytorch]
tags: [pytorch, compiler, fusion]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [8. 어디서나 열리는 모델 파일 만들기](/blog/2026/10/25/pytorch-08-onnx/)  
> → 다음: [10. 숫자를 작게 만들어 모델 줄이기](/blog/2026/10/25/pytorch-10-quantization/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/09-compiler-opt](https://github.com/kjoh94/pytorch-series-examples/tree/main/09-compiler-opt) (`python example.py`)

> **처음이라면 이것만 기억하세요.** 요즘 컴퓨터에서 느린 건 "계산"보다 "**메모리에서 데이터를 꺼내고 다시 넣는 일**"이에요. 컴파일러 최적화의 대부분은 이 왕복을 줄이는 기술이에요. 이번 장은 그 감각을 실제 측정으로 먼저 잡고, 세 가지 대표 기법을 차례로 봐요.

[6장](/blog/2026/10/25/pytorch-06-graph-capture/)~[8장](/blog/2026/10/25/pytorch-08-onnx/)에서 우리는 모델을 **그래프**로 손에 넣었어요. 그래프가 있으면 컴파일러가 "전체를 미리 보고" 손을 볼 수 있죠(6장의 번역가 비유). 그럼 대체 무엇을 손보길래 빨라질까요?

## 0. 비유로 먼저: 요리보다 냉장고 왕복이 오래 걸린다

![냉장고 비유](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig07.jpg)

**그림 7.** 요리사(계산 장치)는 손이 엄청 빨라요. 그런데 재료가 냉장고(메모리)에 있어서, 칼질 한 번 할 때마다 냉장고에 다녀오면 대부분의 시간이 **왕복**에 쓰여요. `y = relu(x*2 + 1)`을 그냥 실행하면:

1. `x`를 전부 꺼내서 ×2 → 결과를 메모리에 다시 저장
2. 그걸 다시 꺼내서 +1 → 다시 저장
3. 또 꺼내서 relu → 저장

**왕복 3번**이에요. 세 연산을 하나로 합치면(fusion) 한 번 꺼내서 ×2, +1, relu를 몽땅 하고 한 번 저장 — **왕복 1번**이 됩니다.

### 0-1. 진짜 그런지 재 보기

```python
import torch, time
torch.set_num_threads(1)
x = torch.randn(16_000_000)          # float 1600만 개 = 64MB

def t(f, n=5):
    f(); s = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - s) / n * 1000

print(t(lambda: x.add(1)))                 # 메모리 1번 훑기
print(t(lambda: torch.sin(x)))             # 1번 훑기, 계산은 훨씬 무거움
print(t(lambda: torch.relu(x * 2 + 1)))    # 가벼운 연산 3개 = 3번 훑기
```

![실측](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig08.jpg)

**그림 8.** 이 블로그를 쓰는 환경(CPU 1스레드)에서 실제로 잰 값이에요: `add` **19.4ms**, `sin` **25.7ms**, `×2, +1, relu` **68.4ms**.

여기서 놀라운 점 두 가지:

- `sin`은 덧셈보다 계산이 **수십 배** 복잡한데, 시간은 고작 30% 정도 더 걸렸어요. 계산은 별로 안 중요했다는 뜻이죠.
- 곱하기·더하기·relu는 셋 다 아주 가벼운 계산인데, 합치면 **약 3.5배** 걸렸어요. 메모리를 3번 훑었기 때문이에요.

그래서 **"연산 개수"가 아니라 "메모리를 몇 번 훑나"가 시간을 정한다**는 감각이 이 장의 핵심이에요. 숫자는 컴퓨터마다 다르지만, 이 경향은 대부분의 CPU·GPU에서 비슷하게 나타나요.

> 원래는 `torch.compile`로 합친 버전도 재 보고 싶었지만, 6장에서 말했듯 이 환경엔 C++ 컴파일러(g++)가 없어서 기본 Inductor를 못 돌렸어요. 대신 아래 4절에서 "합치는 원리"를 FX로 직접 구현해 봐요. g++이 있는 환경이라면 `torch.compile(lambda x: torch.relu(x*2+1))`을 재 보면 `add` 한 번과 비슷한 시간이 나오는 걸 볼 수 있을 거예요.

### 0-2. 미리 계산해 두기 (constant folding)

![folding](/assets/img/posts/2026-10-25-pytorch-09-compiler-opt/fig09.jpg)

**그림 9.** 입력과 상관없는 계산은 실행할 때마다 할 필요가 없어요. `x * (2 * 3.14)`라면 컴파일할 때 `6.28`을 미리 적어 두면 되죠. 아래 2절의 "BatchNorm을 Conv/Linear에 접어 넣기"도 똑같은 원리예요. 연산 하나와 **메모리 왕복 하나**가 통째로 사라져요.

### 0-3. 세 가지 기법 한 줄 요약

| 기법 | 비유 | 줄이는 것 |
|---|---|---|
| constant folding | 답을 미리 적어 두기 | 불필요한 계산과 왕복 |
| operator fusion | 냉장고에서 한 번에 꺼내 한 번에 요리 | 메모리 왕복 |
| memory planning | 다 쓴 그릇 씻어서 재사용 | 메모리 사용량 |

이제 정식 용어로 하나씩 자세히 볼게요.

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

## 자주 묻는 질문

**Q. 그럼 연산 수(FLOPs)는 아예 상관없나요?**
아니요. 행렬곱처럼 데이터 하나로 계산을 많이 하는 연산은 계산이 병목이에요. 하지만 덧셈·relu 같은 원소별 연산은 거의 항상 메모리가 병목이라, 이런 걸 합치는 게 효과가 커요.

**Q. 제가 직접 fusion을 해야 하나요?**
보통은 아니에요. `torch.compile`(6장)이나 ONNX를 받는 런타임·컴파일러(8장)가 자동으로 해 줘요. 이 장은 "왜 빨라지는지" 감을 잡는 용도예요.

**Q. BN 접기는 학습 중에도 하나요?**
아니요. 학습 중엔 BN 통계가 바뀌니까, 학습이 끝나고 `eval()` 상태에서만 접어요.

**Q. memory planning을 하면 결과가 달라지나요?**
아니요. 같은 숫자를 계산하되, 다 쓴 버퍼를 다음 텐서가 이어서 쓰게 해서 메모리만 아껴요.

## 퀴즈

1. `relu(x*2+1)`은 fusion 전 메모리를 몇 번 훑나? → 3번(fusion 후 1번).
2. `sin`이 `add`보다 계산은 훨씬 무거운데 시간이 비슷한 이유는? → 둘 다 메모리를 1번 훑고, 시간은 주로 메모리 왕복이 정하니까.
3. `x * (2*3.14)`에서 컴파일러가 미리 하는 일은? → constant folding으로 `6.28`을 미리 계산.

## 정리

- 핵심 감각: 시간은 "연산 개수"보다 "메모리를 몇 번 훑나"가 정한다.

- folding: 상수 부분은 미리. BN folding이 대표 예.
- fusion: 메모리 왕복을 줄인다. elementwise는 memory-bound.
- planning: 수명이 안 겹치면 버퍼 공유.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [8. 어디서나 열리는 모델 파일 만들기](/blog/2026/10/25/pytorch-08-onnx/)  
> → 다음: [10. 숫자를 작게 만들어 모델 줄이기](/blog/2026/10/25/pytorch-10-quantization/)

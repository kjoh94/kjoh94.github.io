---
layout: post
title: "PyTorch 시리즈 6장 — 그래프 캡처: FX, Dynamo, torch.compile"
date: 2026-10-25 21:06:00 +0900
categories: [pytorch]
tags: [pytorch, torch.compile, dynamo, fx]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [5. 커스텀 디바이스 백엔드 (PrivateUse1)](/blog/2026/10/25/pytorch-05-privateuse1/)  
> → 다음: [7. torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/06-graph-capture](https://github.com/kjoh94/pytorch-series-examples/tree/main/06-graph-capture) (`python example.py`)

지금까지의 PyTorch는 **eager 모드**였다. 파이썬이 한 줄을 읽으면 그 즉시 커널 하나가 돈다. 쉽고 디버깅하기 좋지만, 컴파일러 입장에서는 "다음에 무엇이 올지" 모른 채 한 걸음씩 걷는 셈이다. 이 장은 모델을 **그래프로 붙잡는(capture)** 방법을 다룬다. FX → Dynamo → `torch.compile` → 내 백엔드 순서로 간다.

## 1. eager vs 그래프

![eager vs graph](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig01.jpg)

**그림 1.** 왼쪽 eager는 op 하나 = 커널 하나. 파이썬 분기, print, 디버거가 전부 자연스럽게 동작한다. 오른쪽 그래프는 **op 목록을 미리 받는다.** 그러면 컴파일러는 `relu(x)+1`을 커널 하나로 합치거나(fusion), 메모리를 미리 계획하거나, 다른 장치용 코드를 만들 수 있다. 대가는 "프로그램을 그래프로 표현할 수 있어야 한다"는 제약이다.

## 2. FX Graph: 파이썬으로 된 IR

![FX graph](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig02.jpg)

**그림 2.** FX 그래프의 노드는 딱 6종이다: `placeholder`(입력), `get_attr`(파라미터), `call_function`, `call_method`, `call_module`, `output`. 그래프와 함께 **그래프를 그대로 실행하는 파이썬 코드**가 생성된다. 그래서 그래프를 고치고 `recompile()`하면 바로 실행 가능한 모듈이 된다 — 그래프 변환 패스의 기본 단위다.

```python
import torch
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1

gm = symbolic_trace(M())
print(gm.graph)        # placeholder → call_function relu → add → output
print(gm.code)         # 그래프에서 다시 생성된 파이썬 코드

# 그래프 고치기: relu 를 sigmoid 로 바꾸는 작은 패스
for node in gm.graph.nodes:
    if node.op == "call_function" and node.target is torch.relu:
        node.target = torch.sigmoid
gm.recompile()
print(gm(torch.zeros(2)))   # sigmoid(0)+1 = 1.5
```

## 3. symbolic_trace의 한계

![symbolic_trace limits](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig03.jpg)

**그림 3.** `symbolic_trace`는 진짜 텐서 대신 **Proxy**(값이 없는 가짜)를 넣고 한 번 실행한다. 그런데 `if x.sum() > 0`처럼 *값*에 따라 갈리는 분기를 만나면 Proxy는 True/False를 대답할 수 없다 → `TraceError`. 그래서 PyTorch 2는 한 단계 더 깊이, **파이썬 바이트코드 수준**에서 그래프를 잡는 Dynamo를 만들었다.

```python
def f(x):
    if x.sum() > 0:
        return x * 2
    return x

try:
    symbolic_trace(f)
except Exception as e:
    print("FX 실패:", type(e).__name__)
```

## 4. torch.compile 파이프라인

![compile pipeline](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig04.jpg)

**그림 4.** `torch.compile(model)`은 세 단계를 거친다. ① **Dynamo**: 함수가 실행되려는 순간(PEP 523 프레임 평가 훅) 바이트코드를 읽어 텐서 연산만 FX 그래프로 기록하고, 이 그래프가 유효한 조건(**guard**)을 함께 만든다. ② **AOTAutograd**: 미분 그래프까지 미리 만들고 ATen 수준으로 분해한다. ③ **백엔드**(기본은 Inductor): 실제 코드를 만든다. 형태 계산은 4장의 meta/FakeTensor로, 실제 계산 없이 한다.

## 5. graph break: 그래프가 끊기는 곳

![graph break](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig05.jpg)

**그림 5.** Dynamo는 모르는 것을 만나면 포기하지 않고 **그래프를 끊는다.** 앞부분을 그래프 1로 만들고, 문제 구간은 그냥 파이썬으로 돌리고, 다시 그래프 2를 시작한다. 결과는 늘 맞지만, 조각이 많을수록 최적화 여지가 사라진다. `dynamo.explain`으로 확인하자 — 아래 함수는 **그래프 3개, break 2개**가 나온다.

```python
import torch._dynamo as dynamo

def g(x):
    y = torch.sin(x)
    if y.sum() > 0:      # 텐서 값에 의존하는 분기 → break
        y = y * 2
    print("hi")          # 파이썬 부수효과 → break
    return torch.cos(y)

e = dynamo.explain(g)(torch.randn(4))
print("graphs:", e.graph_count, "breaks:", e.graph_break_count)
for r in e.break_reasons:
    print(" -", str(r.reason)[:70])
```

`torch.compile(model, fullgraph=True)`로 하면 break가 생길 때 에러를 내 준다. 배포용 모델이라면 이게 좋은 습관이다.

## 6. guard와 재컴파일

![guard](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig06.jpg)

**그림 6.** 컴파일된 그래프에는 "입력 shape가 (3,), dtype float32, 정수 인자 n=1…" 같은 **guard**가 붙는다. 다음 호출에서 guard가 통과하면 캐시된 그래프를 쓰고, 실패하면 다시 컴파일한다. 영리한 점: 두 번째로 달라지면 Dynamo가 "이 값은 변하는구나" 하고 **dynamic**(심볼)으로 일반화한다. 그래서 아래 두 실험 모두 4번 호출에 컴파일은 **2번**뿐이다.

```python
from torch._dynamo.testing import CompileCounter

dynamo.reset(); c = CompileCounter()
h = torch.compile(lambda x, n: x * n, backend=c)
for n in [1, 2, 3, 4]:
    h(torch.randn(3), n)
print("int 인자 바뀜 → frames:", c.frame_count)    # 2

dynamo.reset(); c = CompileCounter()
h = torch.compile(lambda x: x * 2, backend=c)
for s in [2, 3, 4, 5]:
    h(torch.randn(s))
print("shape 바뀜 → frames:", c.frame_count)       # 2
```

## 7. 내 컴파일러 백엔드 붙이기

![custom backend](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig07.jpg)

**그림 7.** 백엔드는 생각보다 단순하다: **`(GraphModule, 예제 입력) → 호출 가능한 것`** 을 돌려주는 함수다. 받은 그래프를 출력해 보고, 패턴을 바꾸고, 자기 장치 코드를 만들어 돌려주면 된다. 아무것도 안 하고 `gm.forward`를 돌려주면 "그래프를 보기만 하는" 백엔드가 된다. 5장의 eager 장치 지원과 달리, 이 방식은 장치 메모리/텐서 없이도 컴파일러만으로 하드웨어를 붙일 수 있다.

```python
def my_backend(gm, example_inputs):
    print("=== 받은 그래프 ===")
    print(gm.graph)
    return gm.forward          # 여기서 패턴 치환·융합·코드 생성을 하면 된다

dynamo.reset()
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
cm = torch.compile(model, backend=my_backend)
print(cm(torch.randn(3, 4)).shape)
```

## 정리

- eager는 쉽고, 그래프는 빠르다. 그래프 캡처는 둘 사이의 다리다.
- FX는 그래프 + 코드, 노드 6종. 값 의존 분기는 못 잡는다.
- Dynamo는 바이트코드에서 잡고, 못 하면 graph break, 조건은 guard.
- 백엔드 = GraphModule을 받아 함수를 돌려주는 함수.

다음 장에서는 이 그래프를 **실행 시점이 아니라 미리, 통째로** 뽑아내는 `torch.export`를 본다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [5. 커스텀 디바이스 백엔드 (PrivateUse1)](/blog/2026/10/25/pytorch-05-privateuse1/)  
> → 다음: [7. torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/)

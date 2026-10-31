---
layout: post
title: "PyTorch 시리즈 6장 — 파이썬 코드를 그래프로 붙잡기: FX, Dynamo, torch.compile"
date: 2026-10-25 21:06:00 +0900
categories: [pytorch]
tags: [pytorch, torch.compile, dynamo, fx]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [5. 내 장치를 PyTorch에 꽂기](/blog/2026/10/25/pytorch-05-privateuse1/)  
> → 다음: [7. torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/06-graph-capture](https://github.com/kjoh94/pytorch-series-examples/tree/main/06-graph-capture) (`python example.py`)

> **처음이신가요?** 기초편 A–C와 1–4장을 읽었다면 충분해요. 특히 4장의 `aten::` op 이름이 이 장의 그래프에 그대로 나와요. `torch.compile`을 "모델을 빠르게 하는 한 줄"로만 알고 있었다면, 이 장에서 그 한 줄 뒤를 열어 봐요.

지금까지 우리가 쓴 PyTorch는 전부 **eager(즉시 실행) 모드**였어요. 파이썬이 한 줄을 읽으면 그 자리에서 바로 계산해요. `y = x * 2` 하면 곱셈이 바로 돌고, 다음 줄로 넘어가요. 그래서 `print`로 중간값을 찍기도 쉽고, 디버깅도 편해요.

그런데 단점이 있어요. PyTorch 입장에서는 **"다음에 무슨 계산이 올지"를 모른 채** 한 걸음씩 걷는 셈이에요. 다음 줄을 미리 알면 "곱하기와 더하기를 한 번에 해 버리자" 같은 최적화를 할 수 있는데, 그걸 못 하는 거죠.

그래서 이 장의 주제는 **"파이썬 코드를 실행하기 전에(또는 처음 실행할 때) 계산 순서를 통째로 적어 두기"** 예요. 이렇게 적어 둔 계산 순서를 **그래프**라고 불러요. 3장의 계산 그래프와 비슷한 개념이에요.

> **그래프(graph)가 뭐예요?** "어떤 연산이 어떤 순서로, 무엇을 입력으로 쓰는지" 적은 목록이에요. 노드 = 연산(`mul`, `add`, `relu`), 화살표 = 데이터가 흘러가는 길.

## 0. 비유 — 한 줄씩 통역 vs 원고 통째로 번역

![통역 vs 번역](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig08.jpg)

**그림 0.** eager는 **동시통역사**예요. 한 문장 들으면 바로 옮겨요. 빠르게 반응하지만 문장들을 합쳐서 더 매끄럽게 다듬을 수는 없어요. 그래프 방식은 **원고를 통째로 받은 번역가**예요. 전체를 먼저 읽고, 겹치는 부분은 합치고, 순서를 정리한 다음 한 번에 내놓아요.

이걸 가장 쉽게 보는 방법은 FX로 함수를 "읽혀" 보는 거예요.

```python
import torch
def f(x):
    return torch.relu(x * 2 + 1)

gm = torch.fx.symbolic_trace(f)
print(gm.code)
# def forward(self, x):
#     mul = x * 2
#     add = mul + 1
#     relu = torch.relu(add)
#     return relu
```

파이썬 한 줄이 **연산 세 개(mul → add → relu)의 목록**으로 바뀌었어요. 이게 그래프예요. 이제 컴파일러는 "세 개를 한 번에 계산하는 커널 하나로 합치자(fusion)"를 할 수 있어요.

## 0-2. torch.compile 한 줄이 하는 일

![compile 흐름](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig09.jpg)

**그림 0-2.** 실전에서는 `symbolic_trace`를 직접 부르기보다 `torch.compile`을 써요. 코드는 한 줄이지만 뒤에서는 이렇게 흘러가요.

1. **Dynamo**가 내 함수를 처음 실행할 때 옆에서 지켜보며, 텐서 연산만 골라 기록해요.
2. 기록한 걸 **FX 그래프**로 만들어요(위에서 본 그 목록).
3. **백엔드**(기본은 Inductor)가 그래프를 받아 합치고 정리해서 빠른 코드를 만들어요.

```python
cf = torch.compile(f)            # 이 줄은 아직 아무것도 안 함
x = torch.randn(1000)
print(torch.allclose(cf(x), f(x)))   # 첫 호출 때 캡처+컴파일, 결과는 같다
```

> **참고:** 기본 백엔드(Inductor)는 CPU에서 C++ 컴파일러(g++ 등)가 필요해요. 없으면 `InvalidCxxCompiler` 에러가 나요. 그럴 땐 공부용으로 `torch.compile(f, backend="aot_eager")`를 쓰면 캡처 과정은 똑같이 보고 실행만 eager로 해요. 예제 저장소도 이 설정으로 돌아가요.

### 왜 첫 호출만 느릴까?

![첫 호출](/assets/img/posts/2026-10-25-pytorch-06-graph-capture/fig10.jpg)

**그림 0-3.** 처음 부를 때는 캡처하고 컴파일하는 시간이 들어서 느려요. 두 번째부터는 만들어 둔 코드를 바로 써서 빨라요. 그래서 벤치마크할 땐 **첫 호출(워밍업)은 빼고** 재야 해요. 그리고 입력 모양이 바뀌면 "이 코드가 아직 맞나?" 확인(guard, 6절)에 걸려 다시 컴파일될 수 있어요.

```python
import time
cf = torch.compile(f, backend="aot_eager")
for i in range(3):
    t = time.perf_counter(); cf(x)
    print(f"{i+1}번째 호출: {time.perf_counter() - t:.3f}s")   # 첫 번째만 크다
```

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

## 자주 묻는 질문

- **Q. 항상 `torch.compile`을 쓰면 좋나요?** 대부분 학습·추론이 빨라지지만, 첫 호출이 느리고 디버깅이 어려워져요. 먼저 eager로 맞게 만든 다음, 마지막에 붙이세요.
- **Q. 모델 안에 `print`나 `if`가 있으면요?** 그래프가 거기서 끊길 수 있어요(graph break, 5절). 결과는 맞지만 빨라지는 폭이 줄어요.
- **Q. 결과가 eager랑 조금 달라요.** 연산을 합치면 소수점 계산 순서가 바뀌어 아주 작은 차이(1e-6 수준)가 날 수 있어요. 정상이에요. 비교할 땐 `torch.allclose`를 쓰세요.
- **Q. FX, Dynamo, Inductor… 다 알아야 하나요?** 사용자는 `torch.compile`만 알면 돼요. FX는 "그래프 모양", Dynamo는 "그래프 잡는 사람", Inductor는 "그래프로 코드 만드는 사람" 정도로 기억하세요.

### 퀴즈
1. eager 모드의 장점과 단점 하나씩은? (답: 디버깅 쉬움 / 다음 연산을 몰라 최적화가 어려움)
2. `torch.compile(model)`을 부른 순간 컴파일될까? (답: 아니요, 첫 호출 때)
3. 그래프가 끊기는 대표적 원인은? (답: 텐서 값에 따른 파이썬 분기, print 같은 부수효과 등)

## 정리

- eager는 쉽고, 그래프는 빠르다. 그래프 캡처는 둘 사이의 다리다.
- FX는 그래프 + 코드, 노드 6종. 값 의존 분기는 못 잡는다.
- Dynamo는 바이트코드에서 잡고, 못 하면 graph break, 조건은 guard.
- 백엔드 = GraphModule을 받아 함수를 돌려주는 함수.

다음 장에서는 이 그래프를 **실행 시점이 아니라 미리, 통째로** 뽑아내는 `torch.export`를 본다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [5. 내 장치를 PyTorch에 꽂기](/blog/2026/10/25/pytorch-05-privateuse1/)  
> → 다음: [7. torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/)

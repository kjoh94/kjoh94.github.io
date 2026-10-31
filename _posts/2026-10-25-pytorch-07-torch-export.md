---
layout: post
title: "PyTorch 시리즈 7장 — 모델을 미리 통째로 뽑아 두기: torch.export와 ExportedProgram"
date: 2026-10-25 21:07:00 +0900
categories: [pytorch]
tags: [pytorch, torch.export, aten]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [6. 파이썬 코드를 그래프로 붙잡기](/blog/2026/10/25/pytorch-06-graph-capture/)  
> → 다음: [8. PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/07-torch-export](https://github.com/kjoh94/pytorch-series-examples/tree/main/07-torch-export) (`python example.py`)

> **처음이라면 이것만 기억하세요.** 6장에서는 모델을 *돌리면서* 그래프를 잡았어요. 이번 7장은 모델을 **미리, 통째로** 그래프로 뽑아서 **파일 하나**로 만들어 두는 이야기예요. 어려운 용어(ExportedProgram, ATen IR)는 비유를 먼저 보고 나서 천천히 만나요.

[6장](/blog/2026/10/25/pytorch-06-graph-capture/)에서 `torch.compile`을 "통역사"에 비유했죠. 옆에서 실시간으로 번역해 주니 편하지만, **통역사(파이썬)가 늘 곁에 있어야** 해요. 그런데 모델을 휴대폰이나 작은 기기, 혹은 다른 회사의 컴파일러에 보내야 할 때는 어떨까요? 거기엔 파이썬이 없을 수도 있어요. 그럴 땐 **번역서를 미리 인쇄해서** 들려 보내야 합니다. 그게 `torch.export`예요.

## 0. 비유로 먼저: 동시통역 vs 인쇄된 번역서

![동시통역 vs 번역서](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig08.jpg)

**그림 8.** `torch.compile`(6장)은 실행하면서 그래프를 잡고, 막히는 곳(graph break)이 있으면 잠깐 파이썬으로 돌아가서라도 어떻게든 돌아가요. `torch.export`는 반대로 **처음부터 끝까지 한 장짜리 그래프**를 요구해요. 한 군데라도 그래프로 못 바꾸면 "인쇄 실패" — 즉 에러를 냅니다. 엄격한 대신, 결과물은 파이썬 없이도 뜻이 완전히 통하는 그래프예요.

### 0-1. 직접 해 보기: 가장 작은 export

```python
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)
    def forward(self, x):
        return torch.relu(self.fc(x))

ep = torch.export.export(M(), (torch.randn(4, 3),))   # 예시 입력을 같이 준다
print([n.target for n in ep.graph.nodes if n.op == "call_function"])
# [aten.linear.default, aten.relu.default]
```

포인트 두 가지:

1. **예시 입력**을 같이 넘겨요. PyTorch가 이 입력으로 모델을 한 번 "따라가 보며" 연산을 기록하거든요(6장의 Dynamo와 같은 원리).
2. 결과로 나온 연산 이름이 `aten.linear`, `aten.relu`예요. **ATen**은 PyTorch 내부의 "기본 연산 사전"이라고 생각하면 돼요([4장](/blog/2026/10/25/pytorch-04-aten-dispatcher/)에서 본 dispatcher가 부르는 그 연산들).

이 환경은 C++ 컴파일러(g++)가 없어서 6장의 기본 `torch.compile`은 못 돌렸지만, **`torch.export`는 코드 생성을 하지 않으니 g++ 없이도 잘 돌아갑니다.** 아래 예제는 전부 실제로 실행해서 확인한 결과예요.

### 0-2. ExportedProgram = 도시락 한 통

![도시락 비유](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig09.jpg)

**그림 9.** `export`가 돌려주는 `ExportedProgram`(위의 `ep`)은 도시락 한 통처럼 필요한 걸 다 담고 있어요. **레시피**(연산 그래프), **이름표**(어떤 입력이 가중치이고 어떤 게 사용자 입력인지), **재료**(실제 가중치 값 — [2장](/blog/2026/10/25/pytorch-02-module-state-dict/)의 state_dict), **주의사항**(입력 크기 규칙). 아래 2절에서 이걸 정식 이름으로 다시 만나요.

```python
print([(s.kind.name, s.arg.name) for s in ep.graph_signature.input_specs])
# [('PARAMETER', 'p_fc_weight'), ('PARAMETER', 'p_fc_bias'), ('USER_INPUT', 'x')]
```

가중치도 "입력"으로 적혀 있는 게 신기하죠? 3절에서 이유를 설명해요.

### 0-3. 파일로 저장하고 다시 불러오기

```python
torch.export.save(ep, "m.pt2")
ep2 = torch.export.load("m.pt2")
x = torch.randn(4, 3)
print(torch.allclose(ep2.module()(x), ep.module()(x)))   # True
```

`.pt2` 파일 하나에 도시락이 통째로 들어가요. 불러올 때 `class M` 코드가 없어도 됩니다.

### 0-4. 함정: 입력 크기는 기본이 "고정"

![고정 vs Dim](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig10.jpg)

**그림 10.** 예시 입력이 `(4, 3)`이었으니, 그냥 export하면 배치 크기 4가 **고정**돼요. `(5, 3)`을 넣으면 이렇게 거절당합니다:

```
AssertionError: Guard failed: x.size()[0] == 4
```

6장의 guard와 같은 개념이에요. 배치 크기를 바꾸고 싶다면 5절처럼 `Dim`으로 "이 차원은 변해도 돼요"라고 미리 선언하면 됩니다.

이제 비유를 정식 용어로 바꿔서 하나씩 자세히 볼게요.

## 1. JIT vs AOT

![JIT vs AOT](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig01.jpg)

**그림 1.** `torch.compile`은 파이썬 런타임이 필요하고, graph break가 나도 파이썬으로 fallback해서 어떻게든 돈다. `torch.export`는 반대다. **graph break 없이 전체가 한 그래프여야** 하고, 안 되면 에러로 알려 준다. 그 대신 결과물은 파이썬 없이도 의미가 완결된 그래프 — ONNX, 모바일, AOTInductor의 입력이 된다.

## 2. ExportedProgram 해부

![ExportedProgram](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig02.jpg)

**그림 2.** `export`의 결과 `ExportedProgram`은 네 부분이다: ATen op로 된 `graph_module`, 각 입력이 파라미터·버퍼·사용자 입력 중 무엇인지 적은 `graph_signature`, 실제 가중치 `state_dict`(2장), 그리고 동적 차원의 범위 `range_constraints`.

```python
import torch
from torch.export import export, Dim

lin = torch.nn.Linear(3, 2)
ep = export(lin, (torch.randn(4, 3),))
print(ep.graph_module.code)    # forward(self, p_weight, p_bias, input): aten.linear ...
print(ep.graph_signature)      # 어떤 입력이 PARAMETER 이고 어떤 게 USER_INPUT 인지
print(list(ep.state_dict.keys()))
print(ep.module()(torch.randn(4, 3)).shape)   # 다시 실행 가능한 모듈로
```

## 3. 파라미터도 그래프 입력이 된다

![functional](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig03.jpg)

**그림 3.** 위 코드에서 생성된 `forward`를 보면 `self.weight`가 사라지고 `p_weight`, `p_bias`가 **인자로** 들어온다. 이런 그래프를 **함수형(functional)** 이라 한다 — 숨은 상태가 없고 모든 것이 입력이다. 그래서 가중치만 바꿔 끼우거나, 양자화해서 넣거나, 장치 메모리에 미리 올려두기가 쉽다. 그래프는 "자리"만 알고 값은 밖에 있다.

## 4. IR 사다리

![IR ladder](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig04.jpg)

**그림 4.** export 직후의 그래프는 `aten.linear`처럼 큰 op를 그대로 둔다(Torch IR). `run_decompositions()`를 부르면 4장에서 본 분해 테이블로 **Core ATen IR**(~180 op: `addmm`, `permute` …)로 내려간다. 더 내리면 Prims. 백엔드 개발자에게 중요한 사실: **Core ATen만 구현하면 대부분의 모델이 돈다.**

```python
ep2 = ep.run_decompositions()
print([str(n.target) for n in ep2.graph.nodes if n.op == "call_function"])
# linear → permute + addmm 으로 분해됨
```

## 5. 동적 shape 선언

![Dim](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig05.jpg)

**그림 5.** 아무 말 안 하면 export는 예제 입력의 shape를 **상수로 고정**한다. 배치 크기 4로 export하면 8을 넣었을 때 에러다. `Dim`으로 "이 차원은 변한다"고 약속하면 그래프 안에 `s0` 같은 심볼이 들어간다. 만약 코드가 실제로는 특정 크기를 가정하고 있다면(예: `if x.shape[0]==8`), export가 약속 위반을 에러로 알려준다.

```python
model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
batch = Dim("batch", min=1, max=64)
ep_dyn = export(model, (torch.randn(4, 3),), dynamic_shapes={"input": {0: batch}})
print(ep_dyn.range_constraints)
m = ep_dyn.module()
for b in [1, 7, 33]:
    print(b, m(torch.randn(b, 3)).shape)

ep_static = export(model, (torch.randn(4, 3),))
try:
    ep_static.module()(torch.randn(8, 3))
except Exception as e:
    print("정적 export 에 배치 8 → 에러:", type(e).__name__)
```

## 6. ExportedProgram이 가는 곳

![targets](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig06.jpg)

**그림 6.** 한 번 뽑은 그래프는 여러 경로로 간다: `torch.onnx.export(..., dynamo=True)`(8장), AOTInductor로 `.so` 패키지, 모바일·엣지 런타임, 그리고 **내 컴파일러 백엔드**. 저장/로드도 된다.

```python
torch.export.save(ep_dyn, "mlp.pt2")
loaded = torch.export.load("mlp.pt2")
print(loaded.module()(torch.randn(5, 3)).shape)
```

## 7. TorchScript는 왜 저물었나

![TorchScript](/assets/img/posts/2026-10-25-pytorch-07-torch-export/fig07.jpg)

**그림 7.** 예전 방법인 TorchScript(`jit.trace`/`jit.script`)는 파이썬 부분집합 언어를 따로 정의해야 했고, trace는 분기를 조용히 고정해 버렸으며, 자체 IR이 ATen과 따로 놀았다. `torch.export`는 Dynamo로 진짜 파이썬을 읽고, 실패하면 이유를 말해 주고, 표준 ATen FX 그래프를 낸다. 새 코드는 export를 쓰자.

## 자주 묻는 질문

**Q. torch.compile이랑 torch.export 중 뭘 써야 하나요?**
파이썬 안에서 그냥 빠르게 돌리고 싶다 → `torch.compile`. 모델을 파일로 뽑아서 다른 곳(모바일, ONNX, 다른 컴파일러)에 넘기고 싶다 → `torch.export`.

**Q. export가 에러를 내면 어떻게 하죠?**
대부분 `if x.sum() > 0:`처럼 **값에 따라 갈라지는 파이썬 코드** 때문이에요. 6장의 graph break 원인과 같아요. `torch.cond`로 바꾸거나 그 분기를 모델 밖으로 빼면 됩니다.

**Q. Core ATen이 뭐예요?**
ATen 연산은 수천 개라 받는 쪽이 다 구현하기 힘들어요. 그래서 "이 작은 기본 세트(약 200개)로만 표현해 줘"라고 약속한 게 Core ATen이에요. `run_decompositions()`가 그 번역을 해 줍니다(4절).

**Q. g++이 없어도 되나요?**
네. export 자체는 그래프만 뽑을 뿐 기계어를 만들지 않아요. 기계어까지 만드는 AOTInductor 단계(6절)에서는 컴파일러가 필요합니다.

## 퀴즈

1. 예시 입력 `(4, 3)`으로 export한 모델에 `(8, 3)`을 넣으면? → 에러(guard 실패). `Dim`으로 선언해야 함.
2. `graph_signature`에서 `PARAMETER`로 표시되는 건? → `fc.weight`, `fc.bias` 같은 가중치.
3. `aten.linear`를 Core ATen으로 내리면? → `permute` + `addmm`.

## 정리

- export = 미리 통째로(AOT) 뽑은 한 장짜리 그래프. 못 뽑으면 에러.
- 예시 입력으로 따라가 보며 기록하고, 크기는 기본 고정.
- ExportedProgram = 함수형 ATen 그래프 + signature + state_dict + 범위.
- `run_decompositions()`로 Core ATen까지 내린다.
- 동적 차원은 `Dim`으로 명시한다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [6. 파이썬 코드를 그래프로 붙잡기](/blog/2026/10/25/pytorch-06-graph-capture/)  
> → 다음: [8. PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/)

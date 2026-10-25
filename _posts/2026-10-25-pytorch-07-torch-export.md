---
layout: post
title: "PyTorch 시리즈 7장 — torch.export와 ExportedProgram"
date: 2026-10-25 21:07:00 +0900
categories: [pytorch]
tags: [pytorch, torch.export, aten]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [6. 그래프 캡처 — FX·Dynamo·graph break](/blog/2026/10/25/pytorch-06-graph-capture/)  
> → 다음: [8. PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/07-torch-export](https://github.com/kjoh94/pytorch-series-examples/tree/main/07-torch-export) (`python example.py`)

6장의 `torch.compile`은 **실행하면서(JIT)** 그래프를 잡는다. 서버 학습에는 완벽하지만, 파이썬이 없는 기기에 모델을 보내거나 다른 컴파일러에 넘기려면 "**미리, 통째로, 한 번에**" 뽑힌 그래프가 필요하다. 그게 `torch.export`다.

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

## 정리

- export = AOT, 전체 한 그래프, 실패는 에러.
- ExportedProgram = 함수형 ATen 그래프 + signature + state_dict + 범위.
- `run_decompositions()`로 Core ATen까지 내린다.
- 동적 차원은 `Dim`으로 명시한다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [6. 그래프 캡처 — FX·Dynamo·graph break](/blog/2026/10/25/pytorch-06-graph-capture/)  
> → 다음: [8. PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/)

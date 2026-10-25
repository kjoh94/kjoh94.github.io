---
layout: post
title: "PyTorch 시리즈 12장 — 커스텀 op: torch.library"
date: 2026-10-25 21:12:00 +0900
categories: [pytorch]
tags: [pytorch, torch.library, custom-op]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [11. LLM 추론을 코드로 — prefill·decode·KV cache](/blog/2026/10/25/pytorch-11-llm-inference/)  

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/12-custom-ops](https://github.com/kjoh94/pytorch-series-examples/tree/main/12-custom-ops) (`python example.py`)

시리즈의 마지막 장. ATen에 없는 연산이나, 내가 만든 더 빠른 커널을 PyTorch에 **정식으로** 넣는 방법이다. `torch.library`로 op를 등록하면 그 op는 dispatcher(4장), 장치(5장), compile/export(6·7장), ONNX(8장)를 모두 통과하는 "1급 시민"이 된다.

## 1. 왜 커스텀 op인가

![why](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig01.jpg)

**그림 1.** 그냥 파이썬 함수로 두면 `torch.compile`이 안을 들여다보고 잘게 쪼갠다. 내 특수 커널로 통째로 바꾸고 싶어도 그래프에 흔적이 없다. 커스텀 op로 등록하면 그래프에 **한 덩어리 노드**로 남고, 장치별 커널·shape 규칙·역전파 규칙을 내가 정한다.

## 2. custom_op의 구성

![parts](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig02.jpg)

**그림 2.** 커스텀 op = 이름(`네임스페이스::op`) + 스키마(타입 힌트에서 자동 생성) + 커널들. 커널은 장치별 구현, fake(shape만 계산), autograd 규칙으로 나뉜다. 4장의 디스패처 표에 행 하나를 추가하는 것과 같다.

```python
import torch
from torch import Tensor

@torch.library.custom_op("mylib::scaled_add", mutates_args=())
def scaled_add(a: Tensor, b: Tensor, alpha: float) -> Tensor:
    # 실제 커널 (여기선 CPU 파이썬; 실전에선 C++/CUDA/내 장치 코드)
    return a + alpha * b

x, y = torch.randn(3), torch.randn(3)
print(scaled_add(x, y, 2.0))
print(torch.ops.mylib.scaled_add(x, y, 2.0))     # dispatcher 경유 호출
print(torch.ops.mylib.scaled_add.default._schema)
```

## 3. fake 커널이 필요한 이유

![fake](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig03.jpg)

**그림 3.** compile과 export는 진짜 데이터 없이 **FakeTensor**(4장의 meta와 같은 발상)로 shape만 따라간다. 내 op가 출력 shape를 어떻게 정하는지 알려 주지 않으면 진행할 수 없다. `register_fake`는 계산 없이 빈 텐서만 돌려주면 된다.

```python
@scaled_add.register_fake
def _(a, b, alpha):
    torch._check(a.shape == b.shape)
    return torch.empty_like(a)

def f(a, b):
    return scaled_add(a, b, 0.5).relu()

cf = torch.compile(f, fullgraph=True)
print("compile:", torch.allclose(cf(x, y), f(x, y)))

class M(torch.nn.Module):
    def forward(self, a, b): return scaled_add(a, b, 0.5)
ep = torch.export.export(M(), (x, y))
print([str(n.target) for n in ep.graph.nodes if n.op == "call_function"])  # mylib.scaled_add 한 덩어리
```

## 4. autograd 등록

![autograd](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig04.jpg)

**그림 4.** 커스텀 op는 기본적으로 미분 규칙이 없다. `setup_context`에서 역전파에 필요한 것을 저장하고, `backward`에서 입력별 기울기를 돌려준다(3장). `out = a + α·b`이니 ∂a = g, ∂b = g·α, α는 텐서가 아니니 None.

```python
def setup_context(ctx, inputs, output):
    a, b, alpha = inputs
    ctx.alpha = alpha

def backward(ctx, g):
    return g, g * ctx.alpha, None

scaled_add.register_autograd(backward, setup_context=setup_context)

a = torch.randn(3, requires_grad=True); b = torch.randn(3, requires_grad=True)
scaled_add(a, b, 3.0).sum().backward()
print("grad a:", a.grad, "grad b:", b.grad)   # 1, 3
```

## 5. opcheck

![opcheck](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig05.jpg)

**그림 5.** 등록이 서로 맞는지 사람이 다 확인하기는 어렵다. `torch.library.opcheck`가 스키마, autograd 등록, fake 커널, 동적 shape 컴파일까지 자동으로 검사한다. 커스텀 op를 만들면 테스트에 꼭 넣자.

```python
res = torch.library.opcheck(scaled_add, (torch.randn(4), torch.randn(4), 2.0))
print(res)
```

장치별 커널은 같은 op에 따로 등록한다. 5장의 커스텀 장치라면 `@scaled_add.register_kernel("mydev")` 식이다. C++ 확장에서는 `TORCH_LIBRARY(mylib, m)`로 스키마를, `TORCH_LIBRARY_IMPL(mylib, CPU, m)`으로 커널을 등록한다 — 파이썬 API는 이것의 얇은 포장이다.

```python
@scaled_add.register_kernel("cpu")
def _(a, b, alpha):
    print("  (CPU 전용 커널 호출)")
    return torch.add(a, b, alpha=alpha)
print(scaled_add(torch.ones(2), torch.ones(2), 4.0))
```

## 6. 시리즈 전체 지도 위에서

![map](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig06.jpg)

**그림 6.** 돌아보면: 텐서는 storage+stride(1장), 모듈은 state_dict(2장), 미분은 그래프(3장), 모든 op는 dispatcher를 지나고(4장), 새 장치는 PrivateUse1로(5장), 그래프는 Dynamo/export로 잡아(6·7장), ONNX·최적화·양자화·LLM 추론으로 배포한다(8–11장). `torch.library`는 이 모든 층에 내 연산을 정식으로 등록하는 통로다. 시리즈를 끝까지 따라와 주셔서 감사합니다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [11. LLM 추론을 코드로 — prefill·decode·KV cache](/blog/2026/10/25/pytorch-11-llm-inference/)  

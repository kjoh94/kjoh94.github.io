---
layout: post
title: "PyTorch 시리즈 12장 — 내가 만든 연산을 PyTorch에 정식 등록하기: torch.library 입문"
date: 2026-10-25 21:12:00 +0900
categories: [pytorch]
tags: [pytorch, torch.library, custom-op]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [11. LLM이 답을 한 단어씩 쓰는 법](/blog/2026/10/25/pytorch-11-llm-inference/)  

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/12-custom-ops](https://github.com/kjoh94/pytorch-series-examples/tree/main/12-custom-ops) (`python example.py`)

> **처음이라면 이것만 기억하세요.** PyTorch에 없는 연산을 만들었을 때, 그냥 파이썬 함수로 두지 말고 `torch.library`로 **"정식 메뉴"로 등록**하면 compile·export·ONNX 같은 도구들이 그 연산을 제대로 알아봐요. 등록할 땐 네 가지만 챙기면 돼요: 진짜 계산, 출력 모양 규칙(fake), 미분 규칙(autograd), 검사(opcheck).

드디어 시리즈의 마지막 장이에요. 지금까지 [4장](/blog/2026/10/25/pytorch-04-aten-dispatcher/)에서 `torch.mm` 한 줄이 dispatcher를 거쳐 커널로 가는 길을 봤고, [6장](/blog/2026/10/25/pytorch-06-graph-capture/)·[7장](/blog/2026/10/25/pytorch-07-torch-export/)에서 그래프를 뽑고, [8장](/blog/2026/10/25/pytorch-08-onnx/)에서 다른 런타임으로 옮겼어요. 그런데 **PyTorch에 아예 없는 연산**이라면요? 혹은 내가 직접 짠 더 빠른 커널을 쓰고 싶다면요? 그 답이 이번 장이에요.

## 0. 비유로 먼저

### 0-1. 식당 메뉴판에 내 요리를 정식 등록하기

![메뉴판 비유](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig07.jpg)

**그림 7.** PyTorch를 큰 식당 주방이라고 생각해 볼게요. 내가 새 요리를 개발했는데, 메뉴판에 안 올리고 내 자리에서만 몰래 만들면 주방 직원들(dispatcher, compile, export)은 그 요리를 몰라요. 정식 등록하려면 이런 걸 내야 해요:

1. **레시피** — 실제로 어떻게 계산하는지 (진짜 커널)
2. **그릇 크기표** — 재료 없이도 "이 입력이면 이만한 그릇에 담겨 나와요"를 알려 주는 규칙 (**fake 커널**). compile·export는 진짜 데이터 없이 모양만 따라가며 그래프를 짜거든요(6·7장).
3. **되돌리는 법** — 학습할 때 미분을 어떻게 하는지 (**autograd**, [3장](/blog/2026/10/25/pytorch-03-autograd/))
4. **위생 검사** — 위 세 가지가 서로 맞는지 자동 점검 (**opcheck**)

`torch.library`가 바로 그 **메뉴판**이에요.

### 0-2. 직접 해 보기: 가장 작은 커스텀 op

"절댓값이 t보다 작으면 0, 크면 t만큼 깎기"라는 작은 연산을 등록해 볼게요.

```python
import torch

@torch.library.custom_op("easy::shrink", mutates_args=())
def shrink(x: torch.Tensor, t: float) -> torch.Tensor:          # 1. 레시피
    return torch.where(x.abs() > t, x - t * x.sign(), torch.zeros_like(x))

@shrink.register_fake
def _(x, t):                                                    # 2. 그릇 크기표
    return torch.empty_like(x)        # 계산 없이 "입력과 같은 모양"만 알려 줌

x = torch.tensor([-2.0, -0.3, 0.1, 1.5])
print(torch.ops.easy.shrink(x, 0.5))   # tensor([-1.5, 0.0, 0.0, 1.0])

class M(torch.nn.Module):
    def forward(self, x):
        return torch.ops.easy.shrink(x * 2, 0.5)

ep = torch.export.export(M(), (x,))
print([str(n.target) for n in ep.graph.nodes if n.op == "call_function"])
# ['aten.mul.Tensor', 'easy.shrink.default']
```

실제로 돌린 결과예요. 볼 포인트: export한 그래프에 `easy.shrink`가 **한 덩어리**로 들어 있어요. 내 연산이 ATen 연산(`aten.mul`)과 나란히 "정식 시민"이 된 거예요.

### 0-3. 등록 전 vs 등록 후

![등록 전후](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig08.jpg)

**그림 8.** 등록하지 않은 그냥 파이썬 함수라면, compile/export는 함수 안을 들여다보고 `abs`, `sign`, `where` 같은 작은 연산들로 잘게 쪼개 기록해요(그림의 목록은 예시). 그러면 나중에 "이 부분을 내 빠른 커널로 통째로 바꾸고 싶다"고 해도 그래프에 흔적이 없어요. 등록하면 **한 덩어리 노드**로 남아서, 장치별 커널을 갈아 끼우거나(5장) ONNX에서 대응 연산으로 매핑하기(8장) 쉬워져요.

> 참고: 이 환경엔 C++ 컴파일러(g++)가 없어서, 아래 `torch.compile` 예제는 `backend="aot_eager"`로 돌렸어요. 이 백엔드도 fake 커널로 모양을 따라가는 과정은 똑같이 거치기 때문에, 등록이 제대로 됐는지 확인하기엔 충분해요. C++/CUDA로 커널을 직접 짜는 extension 빌드도 같은 이유로 여기서는 개념만 다뤄요.

이제 정식 용어로 하나씩 자세히 볼게요.

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

cf = torch.compile(f, fullgraph=True, backend="aot_eager")  # g++ 없는 환경에서도 동작
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



## 자주 묻는 질문

**Q. 그냥 파이썬 함수로 쓰면 안 되나요?**
eager로만 돌린다면 괜찮아요. 하지만 compile/export/ONNX로 가거나 내 커널로 바꾸고 싶다면 등록하는 게 좋아요.

**Q. fake 커널은 왜 계산을 안 해요?**
compile·export는 진짜 데이터 없이 "모양"만 따라가며 그래프를 짜요(6·7장). 그래서 출력의 모양과 dtype만 알려 주면 충분해요.

**Q. C++이나 CUDA 커널도 이렇게 등록하나요?**
네. 레시피 부분만 C++/CUDA로 짜고, 같은 `torch.library` 방식으로 연결해요(1·2절). 빌드엔 컴파일러가 필요해요.

**Q. opcheck는 꼭 돌려야 하나요?**
강력 추천해요. fake와 진짜 커널의 모양이 안 맞는다든지 하는 실수를 자동으로 잡아 줘요(5절).

## 퀴즈

1. 커스텀 op 등록 때 챙길 네 가지는? → 진짜 커널, fake 커널, autograd, opcheck.
2. 등록한 op는 export 그래프에 어떻게 보이나? → `easy.shrink.default`처럼 한 덩어리 노드.
3. fake 커널이 돌려주는 것은? → 계산 없이 올바른 모양의 빈 텐서.

## 정리

- `torch.library.custom_op` = 내 연산을 PyTorch 메뉴판에 정식 등록.
- `register_fake`로 모양 규칙을, autograd 등록으로 미분 규칙을, `opcheck`로 검사를.
- 등록된 op는 dispatcher·compile·export·ONNX 전부를 한 덩어리로 통과한다.

![시리즈 지도](/assets/img/posts/2026-10-25-pytorch-12-custom-ops/fig09.jpg)

**그림 9.** 시리즈 전체를 한 장으로. 텐서에서 시작해 LLM 추론까지 왔고, 12장은 이 모든 길 위에 내 연산을 태우는 방법이었어요. 끝까지 읽어 주셔서 고마워요! [전체 목차](/blog/2026/10/25/pytorch-series/)에서 다시 골라 읽을 수 있어요.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [11. LLM이 답을 한 단어씩 쓰는 법](/blog/2026/10/25/pytorch-11-llm-inference/)  

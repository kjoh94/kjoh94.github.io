---
layout: post
title: "PyTorch 시리즈 5 — 내 장치를 PyTorch에 꽂기: PrivateUse1 쉽게 보기"
date: 2026-10-25 21:05:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, privateuse1, backend, dispatcher, fallback, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [4. torch.mm 한 줄은 어디로 갈까](/blog/2026/10/25/pytorch-04-aten-dispatcher/)  
> → 다음: [6. 그래프 캡처 — FX·Dynamo·graph break](/blog/2026/10/25/pytorch-06-graph-capture/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/05-privateuse1](https://github.com/kjoh94/pytorch-series-examples/tree/main/05-privateuse1) (`python example.py`)

> **처음이신가요?** 이 장은 4장의 "안내 데스크(Dispatcher)와 주방(커널)" 비유를 그대로 이어 가요. 1장의 `device`와 4장을 읽고 오면 충분해요. 평소 PyTorch를 *쓰기만* 한다면 이 장은 "아, 새 장치는 이렇게 붙는구나" 정도로 가볍게 읽어도 괜찮아요.

1장에서 텐서에는 `device` 이름표가 붙는다고 했죠. `"cpu"`, `"cuda"`, `"mps"`… 그럼 PyTorch에 없는 **새 가속기**(새 하드웨어, 시뮬레이터, 원격 장치 등)를 만들었다면 어떻게 `x.to("mydev")`이 되게 할까요?

PyTorch 본체 코드를 고쳐서 다시 빌드해야 할 것 같지만, 아니에요. PyTorch는 처음부터 **"바깥 사람이 쓸 빈 자리"** 를 하나 비워 뒀어요. 그 자리 이름이 **PrivateUse1**이에요. 여기에 내 장치를 채우면 **pip 패키지 하나로** 새 장치가 붙어요. 이 장은 특정 제품이 아니라, 공개 PyTorch API로 누구나 쓰는 일반 패턴을 다뤄요.

## 0. 비유 — 식당에 새 주방 하나 더 열기

![새 주방](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig07.jpg)

**그림 0.** 4장의 식당에는 CPU 주방, GPU 주방이 있었어요. 그런데 안내 데스크 표에는 처음부터 **빈 칸 하나(PrivateUse1)** 가 있어요. 우리가 할 일은 그 칸에 "내 가속기 주방"을 차리고 요리법(커널)을 하나씩 채우는 거예요. 메뉴판(op 스키마)은 그대로라서, 사용자는 평소처럼 `torch.mm`, `nn.Linear`를 쓰면 돼요.

## 0-2. 붙이는 순서 — 딱 3단계

![3단계](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig08.jpg)

**그림 0-2.** 전체 흐름을 먼저 머리에 넣고 본문을 보면 쉬워요.

1. **이름 붙이기** — 빈 자리 `privateuse1`에 `"mydev"` 같은 이름을 붙여요.
2. **메모리 다루기** — 그 장치에 텐서를 *만들고*(`empty`), CPU와 *오가게*(`copy_`) 해요. 1장의 storage가 이 장치 메모리에 생기는 거예요.
3. **계산 채우기** — `add`, `mm`, `relu`… op별 커널을 하나씩 등록해요. 아직 없는 op는 CPU로 대신 계산(fallback)하게 둘 수 있어요.

1단계는 지금 바로 해 볼 수 있어요.

```python
import torch
torch.utils.rename_privateuse1_backend("mydev")
print(torch._C._get_privateuse1_backend_name())   # mydev
print(torch.device("mydev:0"))                  # 이제 이 이름을 장치로 알아본다

try:
    torch.empty(2, device="mydev")              # 2단계(메모리)를 아직 안 채웠으니
except Exception as e:
    print(type(e).__name__, str(e)[:60])          # → 에러. 주방 간판만 단 상태
```

> **쉽게 말하면:** 이름만 붙이면 "간판만 단 주방"이에요. 메모리와 커널을 채워야 손님(텐서)을 받을 수 있어요. 아래 본문이 그 채우는 과정이에요.

## 1. 새 장치를 꽂는 길

![두 가지 길](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig01.jpg)

**그림 1.** CUDA·XPU·MPS처럼 PyTorch 본체에 키가 박힌 1급 장치도 있지만, 그러려면 PyTorch 저장소에 코드를 넣어야 한다. 그 대신 PyTorch는 **`PrivateUse1`이라는 예약 키**를 비워 두었다. 외부 패키지가 이 키를 빌려 커널을 등록하고, 이름도 `"mydev"`처럼 바꿀 수 있다. 아래 회색 문장: 장치를 eager 텐서로 지원하지 않고 `torch.compile` 백엔드로만 붙이는 방법도 있다(6장).

## 2. 백엔드 패키지의 구조

![패키지 구조](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig02.jpg)

**그림 2.** 위에서부터: ① 파이썬에서 키 이름 바꾸기, ② C++로 **메모리 할당자와 기초 op**(`empty`, `copy_`) 등록, ③ 계산 커널 등록, ④ `torch.mydev` 장치 모듈(장치 수, 현재 장치, 동기화 등), ⑤ `x.mydev()`, `x.is_mydev` 같은 편의 메서드 자동 생성. 오른쪽이 최종 사용자 경험이다.

```python
import torch
torch.utils.rename_privateuse1_backend("mydev")

# ④ 장치 모듈: torch.mydev — 실제 백엔드는 C++ 확장과 연결된 함수들을 넣는다
import types
mydev_mod = types.ModuleType("mydev")
mydev_mod.is_available = lambda: False      # 장난감이라 실제 장치는 없음
mydev_mod.device_count = lambda: 0
mydev_mod.current_device = lambda: 0
mydev_mod._is_in_bad_fork = lambda: False
mydev_mod.manual_seed_all = lambda seed: None
torch._register_device_module("mydev", mydev_mod)
print(torch._C._get_privateuse1_backend_name())     # mydev

torch.utils.generate_methods_for_privateuse1_backend()
print(hasattr(torch.Tensor, "mydev"), hasattr(torch.Tensor, "is_mydev"))  # True True
print(torch.device("mydev:0"))                      # 이제 장치 문자열로 인식된다
```

C++ 쪽 등록은 이렇게 생겼다(구조 이해용, 빌드 필요):

```cpp
// allocator: 장치 메모리를 잡고 푸는 법
struct MyAllocator final : c10::Allocator { /* allocate()/raw_deleter() */ };
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_my_alloc);

// 4장의 '스키마는 그대로, 키만 PrivateUse1'
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &my_empty);
  m.impl("empty_strided",       &my_empty_strided);
  m.impl("_copy_from",          &my_copy_from);   // CPU <-> mydev
  m.impl("add.Tensor",          &my_add);
  m.impl("mm",                  &my_mm);
}
```

## 3. 텐서가 이사할 때

![이사](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig03.jpg)

**그림 3.** `x.to("mydev")`는 내부적으로 ① 장치 할당자로 **빈 텐서(empty)** 를 만들고 ② CPU → 장치 **copy_** 를 한다. 이 두 커널만 있으면 텐서를 올리고 내리는 것은 된다. 계산은 그 다음이다.

C++ 빌드 없이 개념을 체험하려면, 파이썬 커널을 `PrivateUse1` 키에 직접 등록해 볼 수 있다(키 등록이 실제로 표에 들어가는지 확인):

```python
lib = torch.library.Library("aten", "IMPL")
calls = []
def my_add(a, b, alpha=1):
    calls.append("mydev add")
    return a
lib.impl("add.Tensor", my_add, "PrivateUse1")
print(torch._C._dispatch_has_kernel_for_dispatch_key("aten::add.Tensor", "PrivateUse1"))  # True
```

실제 텐서 메모리 없이 "가짜 장치"로 디스패치 흐름 전체를 시뮬레이션하려면 **텐서 서브클래스 + `__torch_dispatch__`** 가 편하다. 아래는 CPU 메모리를 쓰지만 스스로를 다른 장치처럼 다루고, 어떤 op가 "장치 커널"로 처리되고 어떤 op가 fallback으로 가는지 세는 장난감 백엔드다.

```python
from torch.utils._pytree import tree_map

SUPPORTED = {"aten.add.Tensor", "aten.gelu.default", "aten.native_layer_norm.default", "aten.mul.Tensor", "aten.mm.default", "aten.t.default",
             "aten.addmm.default", "aten.relu.default", "aten.detach.default"}
stats = {"device": 0, "fallback": 0}

class ToyDev(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_wrapper_subclass(cls, data.shape, dtype=data.dtype,
                                                   strides=data.stride(), device="meta")
    def __init__(self, data): self.data_ = data
    def __repr__(self): return f"ToyDev({self.data_})"
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        name = str(func)
        stats["device" if name in SUPPORTED else "fallback"] += 1
        unwrap = lambda t: t.data_ if isinstance(t, ToyDev) else t
        wrap = lambda t: ToyDev(t) if isinstance(t, torch.Tensor) else t
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {}))
        return tree_map(wrap, out)

x = ToyDev(torch.randn(4, 8))
lin = torch.nn.Linear(8, 3)
w, b = ToyDev(lin.weight.detach()), ToyDev(lin.bias.detach())
y = torch.relu(torch.nn.functional.linear(x, w, b))
y2 = torch.softmax(y, dim=-1)        # softmax 는 '미지원' → fallback 카운트
print(stats)
print(torch.allclose(y2.data_, torch.softmax(torch.relu(lin(x.data_)), -1)))
```

## 4. 커널이 없으면: CPU fallback

![fallback](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig04.jpg)

**그림 4.** 모든 op를 처음부터 구현할 수는 없다. 그래서 백엔드는 보통 **boxed CPU fallback**을 등록한다: 장치 텐서를 CPU로 복사 → CPU 커널 실행 → 결과를 다시 장치로 복사. 모델이 "일단 돌게" 만드는 데는 최고지만, 복사가 두 번 붙으니 느리다. 성능을 잴 때는 **fallback으로 통과한 op를 성공으로 세면 안 된다.**

```cpp
// C++: 커널이 없는 모든 aten op를 CPU로 우회시키는 fallback
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&at::native::cpu_fallback>());
}
```

## 5. 커버리지는 op 단위로 쌓인다

![커버리지](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig05.jpg)

**그림 5.** 새 백엔드의 진척도는 "모델이 쓰는 aten op 중 몇 개에 장치 커널이 있나"다. 그러려면 모델이 실제로 쓰는 op 목록부터 뽑아야 한다. 4장의 `TorchDispatchMode`로 간단히 뽑을 수 있다.

```python
from collections import Counter
from torch.utils._python_dispatch import TorchDispatchMode
class OpCount(TorchDispatchMode):
    def __init__(self): super().__init__(); self.c = Counter()
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))

model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.LayerNorm(64),
                            torch.nn.Linear(64, 10), torch.nn.Softmax(-1)).eval()
inp = torch.randn(2, 32)
with torch.no_grad(), OpCount() as oc:
    model(inp)
# 참고: nn.TransformerEncoderLayer 는 추론 시 '_transformer_encoder_layer_fwd' 같은
# 융합 op 하나로 내려가기도 한다 — 백엔드 입장에선 그 op도 커버하거나 분해시켜야 한다
for op, n in oc.c.most_common():
    mark = "O" if op in SUPPORTED else "-"
    print(f"{mark} {op:45s} x{n}")
print("커버리지:", sum(op in SUPPORTED for op in oc.c), "/", len(oc.c))
```

## 6. eager 백엔드와 compile 백엔드

![eager vs compile](/assets/img/posts/2026-10-25-pytorch-05-privateuse1/fig06.jpg)

**그림 6.** PrivateUse1은 **eager**(op 하나씩 실행) 수준의 지원이다. 디버깅은 쉽지만 op마다 호출 오버헤드가 있고, 여러 op를 합치는 최적화(fusion, 9장)는 못 한다. 그래서 많은 백엔드가 **두 층을 함께** 제공한다: eager는 PrivateUse1으로, 성능은 `torch.compile(backend="mydev")` 같은 그래프 컴파일러로. 다음 장이 그 그래프를 어떻게 잡는지에 대한 이야기다.

```python
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    print("내 컴파일러가 받은 그래프 op 수:", len(gm.graph.nodes))
    return gm.forward                  # 여기서 내 장치용 코드로 바꾸면 된다
f = torch.compile(lambda x: torch.relu(x @ x.T) + 1, backend=my_backend)
print(f(torch.randn(4, 4)).shape)
```

## 자주 묻는 질문

- **Q. 이걸 직접 할 일이 있을까요?** 가속기를 만드는 팀이 아니라면 드물어요. 하지만 새 장치용 PyTorch 패키지를 설치해서 쓸 때 "왜 이 op는 느리지?(fallback)", "왜 이 op는 지원 안 된대?(커널 없음)"를 이해하는 데 도움이 돼요.
- **Q. fallback이면 결과는 맞나요?** 맞아요. CPU로 옮겨 계산하고 다시 가져오니까요. 대신 이사 비용(1장의 `.to()` 복사) 때문에 느려요.
- **Q. 모든 op를 다 구현해야 하나요?** 아니요. 실제 모델이 쓰는 op부터 세어 보고(5절) 많이 쓰는 것부터 채워요. 수천 개 중 수십 개면 대부분의 모델이 돌아가요.
- **Q. 왜 이름이 PrivateUse1이에요?** "PyTorch 본체가 쓰지 않는, 외부 전용 칸 1번"이라는 뜻이에요.

### 퀴즈
1. 새 장치를 붙일 때 PyTorch 본체를 다시 빌드해야 할까? (답: 아니요, PrivateUse1 + 패키지)
2. 텐서를 그 장치에 만들려면 최소한 무엇이 필요할까? (답: 메모리 할당 + `empty` + `copy_`)
3. 커널이 없는 op를 일단 돌게 하는 방법은? (답: CPU fallback)

## 정리

- **PrivateUse1**은 out-of-tree 장치용 예약 dispatch key. `rename_privateuse1_backend`로 이름을 붙인다.
- 최소 구성: **allocator + empty + copy_**. 그다음 계산 커널을 op 단위로 채운다.
- 없는 op는 **CPU fallback**으로 돌게 할 수 있지만 성능 지표와는 분리하자.
- 성능은 그래프 단위 컴파일 백엔드(6장)로 보완한다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [4. torch.mm 한 줄은 어디로 갈까](/blog/2026/10/25/pytorch-04-aten-dispatcher/)  
> → 다음: [6. 그래프 캡처 — FX·Dynamo·graph break](/blog/2026/10/25/pytorch-06-graph-capture/)

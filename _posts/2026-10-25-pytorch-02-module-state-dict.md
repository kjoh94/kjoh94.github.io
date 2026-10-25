---
layout: post
title: "PyTorch 시리즈 2 — nn.Module과 state_dict: 모델은 이름표 붙은 텐서 트리다"
date: 2026-10-25 21:02:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, nn-module, state_dict, parameter, buffer, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [1. Tensor의 정체 — storage·stride·view](/blog/2026/10/25/pytorch-01-tensor/)  
> → 다음: [3. Autograd와 추론 모드](/blog/2026/10/25/pytorch-03-autograd/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/02-module-state-dict](https://github.com/kjoh94/pytorch-series-examples/tree/main/02-module-state-dict) (`python example.py`)

1장에서 텐서가 "메모리를 읽는 규칙"이라는 것을 봤다. 모델은 이런 텐서 수백, 수천 개의 묶음이다. 이 묶음을 정리하는 틀이 `nn.Module`이고, 그 묶음에서 **숫자만** 이름표를 붙여 꺼낸 것이 `state_dict`다. 모델 저장·로드, 허깅페이스 업로드, 양자화(Linear를 양자화 Linear로 교체), LoRA, 커스텀 커널 꽂기가 전부 이 두 개념 위에서 돌아간다.

## 1. Module은 트리다

![모듈 트리](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig01.jpg)

**그림 1.** `Net`이라는 뿌리 아래 `fc1`, `act`, `block`이 가지로 붙고, `block`(Sequential) 아래 다시 `0`, `1`이 붙는다. 잎에는 `weight`, `bias` 같은 **Parameter**가 달린다. `ReLU`처럼 파라미터가 없는 모듈도 있다. 핵심: `self.fc1 = nn.Linear(...)`라고 **속성에 대입하는 순간** `Module.__setattr__`이 "이건 서브모듈이네" 하고 내부 사전(`_modules`)에 등록한다.

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.block = nn.Sequential(nn.Linear(8, 2), nn.BatchNorm1d(2))
    def forward(self, x):
        return self.block(self.act(self.fc1(x)))

model = Net()
print(model)
for name, m in model.named_modules():
    print(f"{name or '(root)':10s} {type(m).__name__}")
for name, p in model.named_parameters():
    print(f"{name:16s} {tuple(p.shape)}")
print("총 파라미터 수:", sum(p.numel() for p in model.parameters()))
```

주의할 함정: 파이썬 `list`에 모듈을 담으면 등록되지 않는다. `nn.ModuleList` / `nn.ModuleDict`를 써야 한다.

```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2) for _ in range(3)]      # 등록 안 됨!
class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(3))
print(len(list(Bad().parameters())), len(list(Good().parameters())))   # 0 6
```

## 2. Parameter, Buffer, 그냥 속성

![파라미터 vs 버퍼](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig02.jpg)

**그림 2.** 모듈 안의 텐서는 세 종류다. **Parameter**는 학습되고, 저장되고, `.to(device)`를 따라간다. **Buffer**(`register_buffer`)는 학습되지 않지만 저장되고 장치도 따라간다 — BatchNorm의 `running_mean`, 위치 인코딩 테이블, 양자화 scale이 대표적이다. `persistent=False` 버퍼는 저장에서 빠진다. 그냥 `self.x = torch.tensor(...)`는 어디에도 안 잡혀서 `model.cuda()` 후에도 CPU에 남는 버그를 만든다.

```python
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))                       # 학습 O
        self.register_buffer("scale", torch.tensor(2.0))           # 저장 O, 학습 X
        self.register_buffer("cache", torch.zeros(3), persistent=False)  # 저장 X
        self.plain = torch.zeros(3)                                # 아무것도 아님
    def forward(self, x):
        return x * self.w * self.scale

m = Scale()
print(list(m.state_dict().keys()))           # ['w', 'scale']
print([n for n, _ in m.named_buffers()])     # ['scale', 'cache']
m = m.to(torch.float16)
print(m.w.dtype, m.scale.dtype, m.cache.dtype, m.plain.dtype)  # plain만 float32로 남는다
```

## 3. state_dict — 이름표 사전

![state_dict](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig03.jpg)

**그림 3.** `state_dict()`는 `"경로.이름" → 텐서` 사전이다. 키는 그림 1 트리의 경로를 점으로 이은 것이다(`block.0.weight`). **클래스 코드는 들어 있지 않다.** 그래서 로드하려면 같은 구조의 모델 객체를 먼저 코드로 만들어야 한다. 허깅페이스가 `config.json`(구조 설명)과 `model.safetensors`(숫자)를 따로 두는 이유가 이것이다([번외편](/blog/2026/10/25/pytorch-huggingface-landing/)).

```python
sd = model.state_dict()
for k, v in sd.items():
    print(f"{k:30s} {str(tuple(v.shape)):10s} {v.dtype}")
# 값은 모델 파라미터와 같은 메모리를 공유하는 detach된 텐서다
print(sd["fc1.weight"].data_ptr() == model.fc1.weight.data_ptr())   # True
```

## 4. 저장과 로드

![왕복](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig04.jpg)

**그림 4.** 왕복 과정. 모델에서 `state_dict()`를 뽑아 파일로 저장하고, 로드할 때는 **코드로 같은 구조를 새로 만든 뒤** `load_state_dict()`로 키를 맞춰 숫자를 복사한다. 아래 빨간 상자는 실전에서 자주 만나는 키 불일치 원인이다.

```python
import os, tempfile
path = os.path.join(tempfile.mkdtemp(), "net.pt")
torch.save(model.state_dict(), path)

new = Net()
missing_unexpected = new.load_state_dict(torch.load(path, weights_only=True))
print(missing_unexpected)                   # <All keys matched successfully>
print(torch.equal(new.fc1.weight, model.fc1.weight))  # True

# 키가 다를 때: 'module.' 접두사 제거
wrapped = {"module." + k: v for k, v in model.state_dict().items()}
fixed = {k.removeprefix("module."): v for k, v in wrapped.items()}
print(new.load_state_dict(fixed))

# strict=False 는 '일부만' 로드할 때 (예: 분류 헤드만 다른 경우)
partial = {k: v for k, v in model.state_dict().items() if k.startswith("fc1")}
print(new.load_state_dict(partial, strict=False).missing_keys[:2])
```

`torch.save(model)`(모델 객체 통째 저장)은 pickle에 클래스 경로가 박혀서 코드가 바뀌면 깨지고, 임의 코드 실행 위험도 있다. **state_dict만 저장하고 `weights_only=True`로 로드**하는 것이 기본이다. 배포용으로는 `safetensors` 형식이 더 안전하다.

## 5. 모듈 교체 — 가지 갈아 끼우기

![모듈 교체](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig05.jpg)

**그림 5.** 트리 구조 덕분에 "모든 `nn.Linear`를 내가 만든 `MyLinear`로 바꾸기"가 몇 줄이면 된다. `named_modules()`로 찾고, 부모 모듈을 구해서 `setattr`로 바꿔 끼운다. 양자화(10장), LoRA, 커스텀 커널(12장)이 모두 이 패턴이다.

```python
class MyLinear(nn.Module):
    """예: 가중치를 fp16으로 들고 있다가 계산 때만 올리는 Linear"""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.register_buffer("w16", lin.weight.detach().half())
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
    def forward(self, x):
        return nn.functional.linear(x, self.w16.float(), self.bias)

def replace_linear(root: nn.Module):
    for name, child in root.named_children():
        if isinstance(child, nn.Linear):
            setattr(root, name, MyLinear(child))
        else:
            replace_linear(child)          # 재귀로 내려가기

x = torch.randn(5, 4)
model.eval()
ref = model(x)
replace_linear(model)
print(model)
print("최대 오차:", (model(x) - ref).abs().max().item())   # fp16 반올림 정도의 작은 값
print(list(model.state_dict().keys())[:3])   # 키 이름도 바뀌었다: fc1.w16 ...
```

마지막 줄이 중요하다. **모듈을 바꾸면 state_dict 키도 바뀐다.** 원래 체크포인트를 그대로 로드할 수 없으니, 교체 클래스에 `from_linear`나 `_load_from_state_dict` 훅을 두어 변환하는 것이 일반적이다.

## 6. model(x) 한 번에 일어나는 일

![forward 호출](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig06.jpg)

**그림 6.** `model(x)`는 `__call__`을 거쳐 pre-hook → `forward` → hook 순서로 실행된다. hook은 중간 활성값 관찰, 디버깅, 양자화 캘리브레이션(10장)에 쓰인다. `train()`/`eval()`은 Dropout과 BatchNorm 같은 모듈의 **동작 모드만** 바꾸는 플래그다. gradient 계산을 끄는 것은 별개다(3장).

```python
acts = {}
def save_act(name):
    def hook(mod, inp, out):
        acts[name] = (out.shape, out.abs().max().item())
    return hook
model2 = Net().eval()
hs = [m.register_forward_hook(save_act(n)) for n, m in model2.named_modules() if isinstance(m, nn.Linear)]
model2(torch.randn(5, 4))
print(acts)
for h in hs: h.remove()

print(model2.training, model2.block[1].training)   # False False — eval()은 재귀로 전파
```

## 정리

- `nn.Module`은 **서브모듈·파라미터·버퍼를 자동 등록하는 트리**다. list 말고 ModuleList.
- 저장되는 건 **state_dict(이름→텐서)** 뿐. 구조는 코드(또는 config)가 책임진다.
- Parameter(학습), Buffer(저장만), persistent=False(둘 다 아님)를 구분하자.
- `named_modules` + `setattr`로 모듈을 교체하는 것이 양자화·LoRA·커스텀 커널의 출발점이다.

다음 장에서는 Parameter의 `requires_grad`가 만들어 내는 **계산 그래프**와, 추론할 때 그것을 끄는 세 가지 스위치를 본다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [1. Tensor의 정체 — storage·stride·view](/blog/2026/10/25/pytorch-01-tensor/)  
> → 다음: [3. Autograd와 추론 모드](/blog/2026/10/25/pytorch-03-autograd/)

---
layout: post
title: "PyTorch 시리즈 2 — 모델은 부품 상자: nn.Module과 state_dict"
date: 2026-10-25 21:02:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, nn-module, state_dict, parameter, buffer, pytorch-series]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [1. 텐서 속을 열어 보기](/blog/2026/10/25/pytorch-01-tensor/)  
> → 다음: [3. 미분은 autograd가 알아서](/blog/2026/10/25/pytorch-03-autograd/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/02-module-state-dict](https://github.com/kjoh94/pytorch-series-examples/tree/main/02-module-state-dict) (`python example.py`)

> **처음이신가요?** 기초편 [B. Linear와 순전파](/blog/2026/10/31/pytorch-00b-linear-forward/)와 [C. loss와 학습 루프](/blog/2026/10/31/pytorch-00c-loss-train-loop/)에서 `nn.Linear`와 `model.parameters()`를 이미 써 봤어요. 이 장은 "그게 내부에서 어떻게 정리되어 있나"를 봐요.

1장에서 텐서는 "한 줄 메모리 + 읽는 법 메모"라고 했죠. 그런데 진짜 모델에는 이런 텐서가 수백, 수천 개 있어요. 이걸 아무렇게나 두면 "어떤 게 학습할 숫자고, 뭘 저장해야 하고, GPU로 옮길 땐 뭘 옮겨야 하지?"가 엉망이 돼요. 그래서 PyTorch는 두 가지 도구를 줘요.

- **`nn.Module`** — 부품(층)과 숫자(파라미터)를 담는 **상자**. 상자 안에 상자를 넣을 수 있어요.
- **`state_dict`** — 그 상자에서 **숫자만** 이름표를 붙여 꺼낸 사전. 저장·불러오기는 이걸로 해요.

모델 저장, 허깅페이스 업로드, 양자화, LoRA, 커스텀 커널 꽂기가 전부 이 두 개념 위에서 돌아가요. 천천히 가 볼게요.

## 0. 상자 하나부터 — Linear 안에는 뭐가 들었나

![Linear 안의 숫자](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig08.jpg)

**그림 0.** 기초편 B의 `nn.Linear(4, 8)`을 열어 보면 텐서 두 개가 있어요. `weight`는 모양이 **(출력 8, 입력 4)** — 순서가 "입력, 출력"이 아니라는 점이 처음엔 헷갈려요. `bias`는 출력마다 하나씩 8개. 합치면 `4×8 + 8 = 40`개의 숫자가 이 층이 "배우는" 전부예요.

```python
import torch
import torch.nn as nn

lin = nn.Linear(4, 8)
print(lin.weight.shape, lin.bias.shape)           # torch.Size([8, 4]) torch.Size([8])
print(sum(p.numel() for p in lin.parameters()))   # 40
print(type(lin.weight))                           # Parameter — '학습할 텐서' 표시가 붙은 텐서
print(lin.weight.requires_grad)                   # True
```

`nn.Parameter`는 특별한 게 아니라 **"이 텐서는 학습 대상이에요"라는 표시가 붙은 텐서**예요. 1장에서 본 shape·stride·dtype·device가 전부 그대로 있어요.

> **손으로 세어 보기:** `nn.Linear(784, 10)`의 파라미터 수는? `784×10 + 10 = 7850`. 큰 모델의 "파라미터 70억 개"도 결국 이런 덧셈을 층마다 한 거예요.

## 0-2. 상자 안에 상자 — 내 모델 만들기

![부품 상자](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig07.jpg)

**그림 0-2.** 내 모델 `Net`은 큰 상자예요. 안에 `fc1`(Linear), `act`(ReLU), 그리고 작은 상자 `block`(Sequential)이 들어 있고, `block` 안에 다시 Linear와 BatchNorm이 있어요. 마법은 한 줄이에요. `__init__`에서 **`self.이름 = 부품`이라고 쓰기만 하면** PyTorch가 알아서 "이 상자에 이 부품이 있다"고 등록해요. 그래서 `model.parameters()` 한 번이면 안쪽 깊숙한 숫자까지 다 나와요 — 기초편 C에서 optimizer에 넘긴 바로 그것이에요.

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

> **쉽게 말하면:** `named_modules()`는 "상자 목록", `named_parameters()`는 "숫자 목록"이에요. 이름의 점(`.`)은 상자 안으로 한 칸 들어간다는 뜻이에요 — `block.0.weight` = block 상자 → 0번 부품 → weight.

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

> **언제 Buffer를 쓰나요?** "학습으로 바꾸진 않지만, 모델과 같이 저장되고 같이 GPU로 가야 하는 숫자"면 Buffer예요. 헷갈리면 이렇게 물어보세요: *optimizer가 이 숫자를 바꿔야 하나?* → 예: Parameter / 아니오지만 저장은 필요: Buffer.

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

### 실전: 학습을 이어서 하려면

![체크포인트](/assets/img/posts/2026-10-25-pytorch-02-module-state-dict/fig09.jpg)

**그림 4-2.** 추론만 할 거면 `model.state_dict()`로 충분해요. 그런데 학습을 멈췄다가 내일 **이어서** 하려면 optimizer도 저장해야 해요. Adam 같은 optimizer는 내부에 "지금까지의 이동 방향(모멘텀)" 같은 숫자를 들고 있고, 이것도 `state_dict()`로 꺼낼 수 있어요.

```python
model = Net()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = model(torch.randn(8, 4)).pow(2).mean(); loss.backward(); opt.step()

ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": 3}
torch.save(ckpt, path)

model2 = Net(); opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
c = torch.load(path, weights_only=True)
model2.load_state_dict(c["model"]); opt2.load_state_dict(c["opt"])
print("이어서 시작할 epoch:", c["epoch"] + 1)
```

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

## 자주 묻는 질문

- **Q. `forward`를 직접 부르면 안 되나요?** `model.forward(x)` 대신 `model(x)`를 쓰세요. 그래야 6절의 hook이 실행돼요.
- **Q. `super().__init__()`을 빼먹으면?** `self.fc1 = ...`에서 바로 에러가 나요. 등록할 사전이 아직 없거든요. 항상 첫 줄에 쓰세요.
- **Q. 로드했는데 결과가 이상해요.** `model.eval()`을 빼먹은 경우가 많아요. Dropout·BatchNorm이 학습 모드로 돌고 있어요.
- **Q. 키가 안 맞는다고 나와요.** 에러에 나온 `missing_keys`/`unexpected_keys`를 보고, 4절처럼 접두사를 고치거나 `strict=False`로 일부만 로드하세요.

### 퀴즈
1. `nn.Linear(3, 5)`의 weight 모양은? (답: `(5, 3)`)
2. `nn.Sequential(nn.Linear(2,4), nn.ReLU(), nn.Linear(4,1))`의 파라미터 수는? (답: `2*4+4 + 4*1+1 = 17`)
3. `self.layers = [nn.Linear(2,2)]`로 쓰면 `state_dict()`에 무엇이 나올까? (답: 아무것도 — 등록 안 됨)

## 정리

- `nn.Module`은 **서브모듈·파라미터·버퍼를 자동 등록하는 트리**다. list 말고 ModuleList.
- 저장되는 건 **state_dict(이름→텐서)** 뿐. 구조는 코드(또는 config)가 책임진다.
- Parameter(학습), Buffer(저장만), persistent=False(둘 다 아님)를 구분하자.
- `named_modules` + `setattr`로 모듈을 교체하는 것이 양자화·LoRA·커스텀 커널의 출발점이다.

다음 장에서는 Parameter의 `requires_grad`가 만들어 내는 **계산 그래프**와, 추론할 때 그것을 끄는 세 가지 스위치를 본다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [1. 텐서 속을 열어 보기](/blog/2026/10/25/pytorch-01-tensor/)  
> → 다음: [3. 미분은 autograd가 알아서](/blog/2026/10/25/pytorch-03-autograd/)

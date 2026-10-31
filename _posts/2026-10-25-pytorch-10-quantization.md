---
layout: post
title: "PyTorch 시리즈 10장 — 숫자를 작게 만들어 모델 줄이기: 양자화 입문"
date: 2026-10-25 21:10:00 +0900
categories: [pytorch]
tags: [pytorch, quantization, int8]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [9. 컴파일러는 무엇을 빠르게 하나](/blog/2026/10/25/pytorch-09-compiler-opt/)  
> → 다음: [11. LLM이 답을 한 단어씩 쓰는 법](/blog/2026/10/25/pytorch-11-llm-inference/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/10-quantization](https://github.com/kjoh94/pytorch-series-examples/tree/main/10-quantization) (`python example.py`)

> **처음이라면 이것만 기억하세요.** 양자화는 모델 안의 숫자를 **더 거친 자(눈금이 256칸뿐인 자)**로 재는 거예요. 조금 부정확해지는 대신 크기는 1/4이 되고, 메모리 왕복이 줄어서 빨라져요. scale, zero_point, weight-only, PTQ 같은 말은 비유부터 보고 갈게요.

[9장](/blog/2026/10/25/pytorch-09-compiler-opt/)에서 "시간은 계산보다 **메모리를 몇 번, 얼마나 훑나**가 정한다"를 실측으로 봤죠. 그렇다면 데이터 자체를 작게 만들면 어떨까요? float32 하나는 4바이트, int8 하나는 1바이트예요. 같은 모델을 int8로 저장하면 메모리에서 꺼내 올 양이 **1/4**로 줄어요. 이게 양자화(quantization)의 출발점이에요. [1장](/blog/2026/10/25/pytorch-01-tensor/)에서 본 dtype 이야기의 연장선이기도 해요.

## 0. 비유로 먼저

### 0-1. 256칸짜리 자에 올리기 = scale과 zero_point

![256칸 자](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig07.jpg)

**그림 7.** int8은 -128부터 127까지, **딱 256개 정수**만 표현할 수 있어요. 그래서 실수들을 256칸짜리 자에 올려야 해요. 이때 정할 건 두 개뿐이에요.

- **scale (눈금 간격):** 한 칸이 실수로 얼마인지. 값이 -1.0~2.5 사이라면 `3.5 / 255 ≈ 0.0137`.
- **zero_point (0의 위치):** 실수 `0.0`이 자의 몇 번째 눈금에 오는지. 위 예에선 `-55`.

그리고 변환 공식은 이것 하나예요: **실수 = (정수 − zero_point) × scale**.

```python
import torch
x = torch.tensor([-1.0, 0.0, 0.3, 2.5])
scale = (x.max() - x.min()) / 255
zp = int(torch.round(-x.min() / scale)) - 128
q = torch.clamp(torch.round(x / scale) + zp, -128, 127).to(torch.int8)
print(scale.item(), zp, q.tolist())          # 0.01373  -55  [-128, -55, -33, 127]
print(((q.float() - zp) * scale).tolist())   # [-1.002, 0.0, 0.302, 2.498]
```

실제로 돌린 결과예요. 되돌린 값이 원래 값과 **0.002 정도**밖에 안 달라요. 그리고 `0.0`은 정확히 `0.0`으로 돌아와요 — zero_point가 있는 이유가 이거예요(0은 패딩, relu 등에서 아주 많이 나오는 값이라 정확해야 해요).

### 0-2. weight-only = 가중치만 압축해서 보관

![weight-only](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig08.jpg)

**그림 8.** 제일 쉬운 양자화는 **가중치만** int8로 저장하고, 계산하기 직전에 float로 펼치는 방법이에요. 옷을 압축팩에 넣어 보관하다가 입을 때 꺼내는 것과 비슷해요. 입력(activation)은 건드리지 않으니 안전해요.

```python
lin = torch.nn.Linear(256, 256)
W = lin.weight.data
s = W.abs().amax(dim=1, keepdim=True) / 127          # 줄(출력 채널)마다 눈금 따로
Wq = torch.round(W / s).clamp(-127, 127).to(torch.int8)
print(W.numel() * 4, Wq.numel() + s.numel() * 4)    # 262144 → 66560 바이트
x = torch.randn(8, 256)
y_ref = lin(x)
y_q = x @ (Wq.float() * s).T + lin.bias
print(((y_ref - y_q).norm() / y_ref.norm()).item()) # 약 0.004 (0.4%)
```

크기는 **약 1/4**, 출력 오차는 **0.4%** 정도예요. "줄마다 눈금을 따로" 잡는 게 3절의 per-channel이에요. LLM(11장)에서 많이 쓰는 방식이 바로 이 weight-only예요. 메모리에서 꺼내 오는 양이 줄어드니 9장의 감각대로 빨라지죠.

### 0-3. PTQ = 학습 끝난 모델에 샘플 몇 개 흘려 눈금 맞추기

![PTQ](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig09.jpg)

**그림 9.** 가중치는 이미 값이 정해져 있어서 바로 눈금을 잡을 수 있어요. 그런데 **입력(activation)**까지 int8로 하려면, 실행 중에 값이 어느 범위로 나오는지 알아야 하죠. 그래서 학습이 끝난 모델에 **샘플 데이터를 몇 묶음 흘려 보내** 최소·최대를 기록하고, 그걸로 scale과 zero_point를 정해요. 다시 학습하지 않으니 **Post-Training Quantization(PTQ)**이라고 불러요. 이 과정을 보정(calibration)이라고도 해요(2절).

### 0-4. 세 가지 방식 한눈에

| 방식 | 비유 | 무엇을 int8로? | 난이도 |
|---|---|---|---|
| weight-only | 압축팩 보관 | 가중치만 | 제일 쉬움 |
| PTQ (정적) | 샘플로 눈금 맞추기 | 가중치 + activation | 보통 |
| QAT | 거친 자로 연습하며 재학습 | 가중치 + activation | 어려움, 정확도 최고 |

이제 정식 용어로 하나씩 자세히 볼게요.

## 1. 양자화 = 실수를 정수 격자에 올리기

![grid](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig01.jpg)

**그림 1.** 실수 축 위에 간격이 `scale`인 점(격자)을 찍고, 각 값을 가장 가까운 점으로 옮긴다. 점의 번호가 정수 `q`다. 공식은 `q = round(x/scale) + zero_point`, 복원은 `x ≈ (q - zero_point) × scale`. 반올림 때문에 생기는 차이가 **양자화 오차**다(최대 scale/2).

```python
import torch
torch.manual_seed(0)

def quant(x, scale, zp, qmin=-128, qmax=127):
    return torch.clamp(torch.round(x / scale) + zp, qmin, qmax).to(torch.int8)

def dequant(q, scale, zp):
    return (q.float() - zp) * scale

x = torch.randn(8)
scale, zp = x.abs().max().item() / 127, 0
q = quant(x, scale, zp)
print("x  :", x)
print("q  :", q)
print("err:", (x - dequant(q, scale, zp)).abs().max().item(), "<= scale/2 =", scale / 2)
```

## 2. scale과 zero_point 정하기

![scale](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig02.jpg)

**그림 2.** 관측한 min/max를 정수 범위에 맞추면 scale과 zero_point가 나온다. **대칭(symmetric)** 은 zero_point=0으로 두고 `max|x|/127`을 쓴다 — 가중치에 흔하다. **비대칭**은 ReLU 뒤처럼 한쪽으로 치우친 활성값에 좋다. 함정: 이상치 하나가 범위를 늘리면 나머지 값의 해상도가 떨어진다.

```python
def affine_params(x, qmin=-128, qmax=127):
    lo, hi = min(x.min().item(), 0.0), max(x.max().item(), 0.0)
    scale = (hi - lo) / (qmax - qmin)
    zp = int(qmin - round(lo / scale))
    return scale, zp

act = torch.relu(torch.randn(1000))            # 0 이상으로 치우친 분포
for name, (s, z) in {"symmetric": (act.abs().max().item() / 127, 0),
                     "affine": affine_params(act)}.items():
    e = (act - dequant(quant(act, s, z), s, z)).abs().mean().item()
    print(f"{name:9s} scale={s:.4f} zp={z:4d} mean err={e:.5f}")

out = torch.cat([act, torch.tensor([50.0])])   # 이상치 하나
s, z = affine_params(out)
print("이상치 포함 scale:", round(s, 4), "→ 해상도 저하")
```

## 3. per-tensor vs per-channel

![channel](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig03.jpg)

**그림 3.** Linear 가중치의 출력 채널(행)마다 크기가 크게 다를 수 있다. scale 하나로 전부 덮으면 작은 채널은 격자가 너무 성기다. 채널마다 scale을 두면(per-channel) 오차가 확 준다. PyTorch는 둘 다 내장 연산으로 제공한다.

```python
W = torch.randn(4, 16) * torch.tensor([[0.01], [0.1], [1.0], [5.0]])   # 채널별 크기 차이

qt = torch.quantize_per_tensor(W, scale=W.abs().max().item() / 127, zero_point=0, dtype=torch.qint8)
sc = W.abs().amax(dim=1) / 127
qc = torch.quantize_per_channel(W, sc, torch.zeros(4, dtype=torch.long), axis=0, dtype=torch.qint8)
for name, qq in [("per-tensor", qt), ("per-channel", qc)]:
    rel = ((W - qq.dequantize()).abs().amax(dim=1) / W.abs().amax(dim=1))
    print(f"{name:11s} 채널별 상대오차:", [f"{v:.3f}" for v in rel.tolist()])
```

## 4. 양자화 방식 세 가지

![modes](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig04.jpg)

**그림 4.** **dynamic**: 가중치만 미리 int8, 활성값 범위는 실행 중 계산 — 준비가 제일 쉽다. **static(PTQ)**: 보정(calibration) 데이터를 흘려 활성값 범위를 미리 재 둔다 — 가장 빠르다. **QAT**: 학습 중에 양자화 오차를 흉내 내 모델이 적응하게 한다 — 정확도가 가장 좋다. 아래는 static 보정의 핵심인 **observer**를 손으로 만든 것이다.

```python
class MinMaxObserver:
    def __init__(self): self.lo, self.hi = float("inf"), float("-inf")
    def __call__(self, t):
        self.lo = min(self.lo, t.min().item()); self.hi = max(self.hi, t.max().item()); return t
    def params(self): return affine_params(torch.tensor([self.lo, self.hi]))

net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).eval()
obs = MinMaxObserver()
with torch.no_grad():
    for _ in range(50):                      # 보정 데이터
        obs(net(torch.randn(16, 8)))
print("보정된 activation scale, zp:", obs.params())
```

## 5. fake quantization

![fake](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig05.jpg)

**그림 5.** quantize → dequantize를 바로 이어 붙이면 값은 float 그대로지만 **격자에 스냅된** 값이 된다. 이것이 fake quant. 정수 하드웨어 없이도 양자화 후 정확도를 미리 볼 수 있다. QAT에서는 round의 기울기가 0이라 학습이 안 되므로, 역전파 때는 그냥 통과시키는 **STE(straight-through estimator)** 를 쓴다.

```python
x = torch.randn(6, requires_grad=True)
s = 0.05
y = torch.fake_quantize_per_tensor_affine(x, s, 0, -128, 127)
y.sum().backward()
print("snap:", (y / s).detach())          # 모두 정수
print("grad (STE):", x.grad)              # 범위 안은 1
```

## 6. PyTorch 쪽에서 쥘 것

![summary](/assets/img/posts/2026-10-25-pytorch-10-quantization/fig06.jpg)

**그림 6.** 양자화 도구(구 `torch.ao.quantization`, 새 `torchao`, PT2E 흐름)는 계속 바뀌지만 개념은 같다: observer로 범위를 재고, scale/zp를 정하고, export 그래프에 **Q/DQ 노드**를 넣으면 백엔드가 그 패턴을 정수 커널로 바꾼다(9장 fusion). 그리고 항상 float 결과와 비교한다.

```python
lin = torch.nn.Linear(16, 4).eval()
xin = torch.randn(32, 16)
Wq = torch.quantize_per_channel(lin.weight.detach(), lin.weight.abs().amax(1).detach() / 127,
                                torch.zeros(4, dtype=torch.long), 0, torch.qint8)
with torch.no_grad():
    ref = lin(xin)
    approx = torch.nn.functional.linear(xin, Wq.dequantize(), lin.bias)
print("int8 가중치 출력 상대오차:", ((ref - approx).norm() / ref.norm()).item())
```



## 자주 묻는 질문

**Q. 정확도가 얼마나 떨어지나요?**
weight-only int8은 보통 거의 티가 안 나요(위 예제 0.4%). activation까지 int8로 하면 모델에 따라 차이가 커서, 꼭 검증 데이터로 확인해야 해요.

**Q. zero_point는 꼭 필요한가요?**
가중치처럼 0을 중심으로 대칭인 값은 zero_point를 0으로 두는 "대칭 양자화"를 많이 써요(0-2 예제). relu 뒤처럼 한쪽으로 쏠린 값은 zero_point가 있으면 256칸을 더 알뜰하게 써요.

**Q. int4는요?**
16칸짜리 자예요. 더 작아지지만 오차가 커져서, 보통 작은 묶음(group)마다 scale을 따로 두는 방식과 함께 써요.

**Q. PTQ에 샘플은 몇 개나 필요해요?**
실제 입력과 비슷한 데이터 수십~수백 개면 보통 충분해요. 너무 다른 데이터로 보정하면 눈금이 어긋나요.

## 퀴즈

1. 실수 → 정수 공식을 거꾸로 쓰면? → 실수 = (정수 − zero_point) × scale.
2. weight-only 양자화에서 int8이 되는 것은? → 가중치만. activation은 float 그대로.
3. PTQ에서 샘플 데이터를 흘려 보내는 이유는? → activation의 범위를 재서 scale·zero_point를 정하려고.



## 정리

- 양자화 = 실수를 256칸 자(int8)에 올리기. 실수 = (정수 − zero_point) × scale.
- weight-only: 가중치만 int8 → 크기 1/4, 오차 작음.
- PTQ: 샘플로 activation 범위를 재서 눈금을 정한다. 재학습까지 하면 QAT.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [9. 컴파일러는 무엇을 빠르게 하나](/blog/2026/10/25/pytorch-09-compiler-opt/)  
> → 다음: [11. LLM이 답을 한 단어씩 쓰는 법](/blog/2026/10/25/pytorch-11-llm-inference/)

---
layout: post
title: "PyTorch 시리즈 10장 — 양자화, PyTorch 쪽에서"
date: 2026-10-25 21:10:00 +0900
categories: [pytorch]
tags: [pytorch, quantization, int8]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [9. 컴파일러 최적화 감각 — folding·fusion·메모리](/blog/2026/10/25/pytorch-09-compiler-opt/)  
> → 다음: [11. LLM 추론을 코드로 — prefill·decode·KV cache](/blog/2026/10/25/pytorch-11-llm-inference/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/10-quantization](https://github.com/kjoh94/pytorch-series-examples/tree/main/10-quantization) (`python example.py`)

모델을 더 작고 빠르게 만드는 가장 강력한 방법은 **숫자를 작게** 만드는 것이다. float32(4바이트)를 int8(1바이트)로 바꾸면 메모리는 1/4, 정수 연산기는 훨씬 싸다. 이 장은 양자화의 수학을 PyTorch 코드로 하나씩 확인한다.

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


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [9. 컴파일러 최적화 감각 — folding·fusion·메모리](/blog/2026/10/25/pytorch-09-compiler-opt/)  
> → 다음: [11. LLM 추론을 코드로 — prefill·decode·KV cache](/blog/2026/10/25/pytorch-11-llm-inference/)

---
layout: post
title: "PyTorch 기초편 B — Linear와 순전파: 숫자를 섞는 층 하나"
date: 2026-10-31 06:10:00 +0900
permalink: /blog/2026/10/31/pytorch-00b-linear-forward/
categories: [pytorch, tutorial]
tags: [pytorch, nn, linear, activation, beginner, pytorch-series]
---

> **PyTorch 시리즈 · 기초편 B/C** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [기초편 A. 텐서를 숫자 상자로](/blog/2026/10/31/pytorch-00a-tensor-numbers/) · → 다음: [기초편 C. loss와 학습 루프](/blog/2026/10/31/pytorch-00c-loss-train-loop/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/00b-linear-forward](https://github.com/kjoh94/pytorch-series-examples/tree/main/00b-linear-forward) (`python example.py`)

신경망은 복잡해 보이지만, 벽돌 하나는 아주 단순해요. **입력 숫자들에 가중치를 곱해서 더하고, 기본 점수를 하나 더하는 것.** 이 벽돌을 PyTorch에서는 `nn.Linear`라고 불러요. 이 글에서는

1. 뉴런 하나를 손으로 계산하고,
2. 그걸 여러 개 모아 행렬 곱으로 바꾸고,
3. `nn.Linear`가 정확히 같은 일을 한다는 걸 확인하고,
4. 활성화 함수로 "꺾임"을 넣고,
5. 층을 쌓아 **순전파(forward pass)** 를 따라가 볼게요.

[기초편 A](/blog/2026/10/31/pytorch-00a-tensor-numbers/)에서 본 shape, `*`와 `@`의 차이, 브로드캐스팅이 전부 다시 나와요.

## 1. 뉴런 하나 = 가중 합 + 편향

![입력 x1, x2, x3가 가중치 w1, w2, w3 화살표를 타고 Σ+b 원으로 모여 출력 y가 되는 그림](/assets/img/posts/2026-10-31-pytorch-00b/fig01.jpg)

*그림 1. 입력마다 가중치(w)를 곱해 모두 더하고, 편향(b)을 더하면 출력 하나가 나와요. w는 "이 입력이 얼마나 중요한가", b는 "기본 점수"예요.*

예를 들어 "오늘 산책을 갈까?" 점수를 매긴다고 해 볼게요.

- x1 = 날씨 좋음 정도 (2)
- x2 = 피곤함 (1)
- x3 = 시간 여유 (3)

날씨는 좋을수록 가고 싶고(w1 = +0.5), 피곤하면 가기 싫고(w2 = −1.0), 시간 여유는 조금 도움이 돼요(w3 = +0.2). 그리고 원래 산책을 좋아해서 기본 점수 b = 0.1.

```python
import torch

x = torch.tensor([2., 1., 3.])
w = torch.tensor([0.5, -1.0, 0.2])
b = torch.tensor(0.1)

y = (w * x).sum() + b
print(y)   # 0.5*2 + (-1)*1 + 0.2*3 + 0.1 = 0.7
```

`w * x`는 칸끼리 곱하기, `.sum()`은 다 더하기였죠(기초편 A 4절). 이게 뉴런 하나의 전부예요.

여기서 중요한 생각 하나: **학습이란 이 w와 b를 좋은 값으로 찾아가는 과정**이에요. 지금은 우리가 숫자를 정했지만, 실제로는 데이터를 보고 컴퓨터가 스스로 정해요. 그 방법이 기초편 C의 주제예요.

### 같은 계산을 내적으로

`(w * x).sum()`은 수학에서 **내적(dot product)** 이라고 불러요. PyTorch에서는 `@`로도 쓸 수 있어요.

```python
print(w @ x + b)   # 같은 0.7
```

1차원 벡터끼리 `@`를 하면 "칸끼리 곱해서 다 더하기"가 돼요. 이걸 여러 개 동시에 하면 행렬 곱이 돼요.

## 2. 뉴런 여러 개 = 행렬 곱

![입력 [2,1,3]에 2×3 행렬 W를 곱해 출력 [5,2]가 되는 그림. 아래에 각 출력의 계산식](/assets/img/posts/2026-10-31-pytorch-00b/fig02.jpg)

*그림 2. W의 각 **행**이 뉴런 하나의 가중치예요. 행이 2개면 출력이 2개. 출력 하나하나는 "입력 전부를 그 행의 비율로 섞은 값"이에요.*

출력이 하나가 아니라 여러 개 필요할 때가 많아요. 예를 들어 같은 입력으로 "산책 점수"와 "영화 점수"를 동시에 내고 싶다면, 뉴런 두 개를 나란히 두면 돼요. 각 뉴런의 가중치를 한 줄씩 쌓으면 행렬이 되죠.

```python
W = torch.tensor([[1., 0., 1.],    # 뉴런 1의 가중치
                  [0., 2., 0.]])   # 뉴런 2의 가중치
x = torch.tensor([2., 1., 3.])
print(W @ x)   # tensor([5., 2.])
```

- 뉴런 1: 1·2 + 0·1 + 1·3 = **5** → x1과 x3만 섞음
- 뉴런 2: 0·2 + 2·1 + 0·3 = **2** → x2만 두 배로

그래서 행렬 곱을 **"특징을 섞는 레시피"** 라고 생각하면 편해요. W의 한 행은 "재료(입력 특징)를 어떤 비율로 섞을지" 적은 레시피 한 장이고, 행이 여러 개면 요리(출력)도 여러 개 나와요.

### shape 규칙

`W`가 `(출력 수, 입력 수)` = `(2, 3)`, `x`가 `(3,)`이면 결과는 `(2,)`예요. **안쪽 숫자(3과 3)가 같아야** 곱할 수 있고, 그 숫자는 사라져요.

```
(2, 3) @ (3,)  →  (2,)
   ↑       ↑
   └ 같아야 함 ┘
```

## 3. 배치로 한꺼번에: x @ W.T + b

실제로는 샘플 하나가 아니라 여러 개를 한 번에 넣어요. 기초편 A에서 본 대로 **맨 앞 차원이 배치**예요.

```python
X = torch.tensor([[2., 1., 3.],    # 샘플 1
                  [1., 1., 1.]])   # 샘플 2      shape (2, 3)
B = torch.tensor([0.0, 1.0])       # 뉴런별 편향 shape (2,)

Y = X @ W.T + B
print(Y)
# [[5., 3.],
#  [2., 3.]]
```

여기서 두 가지가 새로 나왔어요.

**`W.T` (전치)**: 행과 열을 뒤집어요. `W`는 `(2, 3)`이라 `W.T`는 `(3, 2)`. 그래야 `(2, 3) @ (3, 2)`로 안쪽 3이 맞아요. 결과는 `(배치 2, 출력 2)`.

**`+ B` (브로드캐스팅)**: `B`는 `(2,)`인데 `(2, 2)`에 더해져요. 모든 샘플에 같은 편향이 더해진 거죠. 기초편 A 4절의 규칙 그대로예요.

이 한 줄, **`y = x @ W.T + b`** 가 신경망에서 가장 많이 실행되는 식이에요. 언어 모델도, 이미지 모델도 대부분의 계산 시간이 이 식에 들어가요.

## 4. nn.Linear: 같은 일을 하는 PyTorch 부품

![nn.Linear(3, 2) 코드와 weight shape (2,3), bias shape (2,)를 보여 주는 그림](/assets/img/posts/2026-10-31-pytorch-00b/fig03.jpg)

*그림 3. `nn.Linear(3, 2)`는 입력 3개를 받아 출력 2개를 내는 층이에요. 안에 `weight (2, 3)`과 `bias (2,)`가 들어 있고, 호출하면 `x @ W.T + b`를 계산해요.*

손으로 W와 b를 만드는 대신 PyTorch 부품을 쓰면 이렇게 돼요.

```python
import torch.nn as nn

layer = nn.Linear(3, 2)           # in_features=3, out_features=2
print(layer.weight.shape)         # torch.Size([2, 3])
print(layer.bias.shape)           # torch.Size([2])
print(layer(X))                   # 무작위 가중치라 숫자는 매번 다름
```

정말 같은 계산인지 확인해 볼까요? 가중치를 우리가 정한 값으로 덮어써 보면 돼요.

```python
with torch.no_grad():             # "지금은 학습용 기록 안 해도 돼"
    layer.weight.copy_(W)
    layer.bias.copy_(B)

print(layer(X))
print(torch.allclose(layer(X), X @ W.T + B))   # True
```

`True`가 나오면 성공이에요. **`nn.Linear`는 마법이 아니라 그냥 `x @ W.T + b`** 예요. 다른 점은 딱 두 가지예요.

1. W와 b를 무작위로 **적당히** 초기화해 줘요(너무 크거나 작지 않게).
2. W와 b를 **"학습할 숫자(파라미터)"로 등록**해 줘요. 그래서 나중에 optimizer가 찾아서 고칠 수 있어요.

`torch.no_grad()`는 "지금 하는 계산은 학습 기록에 넣지 마"라는 뜻이에요. 기초편 C에서 이유가 나와요.

## 5. 활성화 함수: 직선을 꺾는 관절

![왼쪽에 ReLU 그래프(음수는 0, 양수는 그대로), 오른쪽에 Sigmoid 그래프(0~1 S자 곡선)](/assets/img/posts/2026-10-31-pytorch-00b/fig04.jpg)

*그림 4. ReLU는 음수를 0으로 잘라요. Sigmoid는 어떤 숫자든 0과 1 사이로 눌러 줘서 확률처럼 읽을 수 있어요. 둘 다 "직선이 아닌" 함수예요.*

Linear를 두 개 이어 붙이면 더 똑똑해질까요? 놀랍게도 **아니에요.** 직선 변환을 두 번 해도 결국 직선 변환 하나와 같아요. `(x @ A) @ B = x @ (A @ B)`니까요. 100층을 쌓아도 1층과 같은 표현력이에요.

그래서 층 사이에 **꺾임**을 넣어요. 이게 활성화 함수예요.

```python
t = torch.linspace(-3, 3, 7)    # [-3, -2, -1, 0, 1, 2, 3]
print(torch.relu(t))            # [0, 0, 0, 0, 1, 2, 3]
print(torch.sigmoid(t))         # [0.05, 0.12, 0.27, 0.5, 0.73, 0.88, 0.95]
```

### ReLU: 제일 많이 쓰는 관절

`ReLU(x) = max(0, x)`. 음수면 0, 양수면 그대로. 너무 단순해서 의아할 정도지만, 이 꺾임 하나 덕분에 층을 쌓을수록 복잡한 모양을 그릴 수 있어요. 직선 조각 여러 개를 이어 붙이면 곡선을 흉내 낼 수 있는 것과 같아요.

### Sigmoid: 0~1로 누르기

`sigmoid(x) = 1 / (1 + e^(-x))`. 아주 큰 수는 1에 가깝게, 아주 작은 수는 0에 가깝게, 0은 정확히 0.5. 그래서 "예/아니오" 문제의 마지막 출력을 **확률처럼** 읽고 싶을 때 써요. 기초편 C의 분류 예제에서 다시 만나요.

PyTorch에는 함수 버전(`torch.relu`)과 부품 버전(`nn.ReLU()`)이 둘 다 있어요. 하는 일은 같고, 부품 버전은 `nn.Sequential`에 끼우기 편해요.

## 6. 순전파: 층을 쌓고 shape을 따라가기

![입력 x → Linear → ReLU → Linear → 출력 y 블록이 화살표로 이어지고, 아래에 (4,3)→(4,8)→(4,8)→(4,1) shape이 적힌 그림](/assets/img/posts/2026-10-31-pytorch-00b/fig05.jpg)

*그림 5. 입력이 층을 왼쪽에서 오른쪽으로 한 번 통과하는 것을 순전파라고 해요. 각 층을 지날 때 shape이 어떻게 바뀌는지 따라가면 모델 구조가 한눈에 보여요. 앞의 4는 배치 크기라 끝까지 그대로예요.*

```python
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(3, 8),   # 특징 3개 → 8개로 넓히기
    nn.ReLU(),         # 꺾기
    nn.Linear(8, 1),   # 8개 → 점수 1개로 모으기
)

inp = torch.randn(4, 3)        # 샘플 4개, 특징 3개
h = inp
for i, mod in enumerate(model):
    h = mod(h)
    print(i, mod.__class__.__name__, tuple(h.shape))
# 0 Linear (4, 8)
# 1 ReLU   (4, 8)
# 2 Linear (4, 1)
```

`model(inp)` 한 줄이면 위 루프와 똑같이 계산돼요. 이렇게 입력 → 출력으로 한 번 흘러가는 걸 **순전파**라고 불러요.

### 중간에 8개로 넓히는 이유

가운데 8개짜리 층을 **은닉층(hidden layer)** 이라고 불러요. 입력 특징 3개를 8가지 방식으로 섞어 보고(Linear), 쓸모없는 건 꺼 버리고(ReLU), 남은 걸 다시 모아서 답을 내요(Linear). 8은 우리가 정하는 숫자예요. 크게 하면 표현력이 늘지만 학습할 숫자도 늘어요.

### 파라미터 세기

```python
print(sum(p.numel() for p in model.parameters()))   # 41
```

- 첫 Linear: weight 3×8 = 24, bias 8 → 32
- ReLU: 파라미터 없음
- 둘째 Linear: weight 8×1 = 8, bias 1 → 9
- 합계 **41**

뉴스에서 듣는 "70억 파라미터 모델"도 원리는 이 숫자 세기와 같아요. Linear가 엄청 크고 많을 뿐이에요.

## 7. 내 모델을 클래스로 만들기 (미리 보기)

`nn.Sequential`은 일렬로 쌓을 때 편하지만, 보통은 `nn.Module`을 상속해서 직접 써요. 기초편 C와 [2장](/blog/2026/10/25/pytorch-02-module-state-dict/)에서 계속 쓸 모양이라 미리 보여 드릴게요.

```python
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):          # 순전파를 여기에 적는다
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = TinyMLP()
print(model(torch.randn(4, 3)).shape)   # (4, 1)
```

규칙은 두 개예요. `__init__`에서 **부품(층)을 만들고**, `forward`에서 **부품을 어떤 순서로 쓸지** 적어요. `model(x)`를 부르면 PyTorch가 `forward(x)`를 대신 불러 줘요.

## 8. 자주 만나는 에러

| 에러 메시지 일부 | 원인 | 해결 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (4x5 and 3x8)` | 입력 특징 수(5)와 Linear의 in_features(3)가 다름 | `nn.Linear(5, 8)`로 맞추거나 입력 확인 |
| `expected scalar type Float but found Long` | 정수 텐서를 넣음 | `x.float()` |
| 출력 shape이 `(8,)`로 나옴 | 배치 차원을 빼먹음 | `x.unsqueeze(0)`으로 `(1, 8)` 만들기 |

## 정리

- 뉴런 하나 = **가중 합 + 편향**. w는 중요도, b는 기본 점수.
- 뉴런 여러 개 = **행렬 곱**. W의 각 행이 "특징 섞는 레시피".
- 배치까지 합치면 **`y = x @ W.T + b`**, 그게 `nn.Linear`.
- Linear만 쌓으면 직선 하나와 같아요. **활성화(ReLU, Sigmoid)** 로 꺾어야 쌓는 의미가 생겨요.
- 입력이 층을 한 번 통과하는 게 **순전파**. shape을 따라가면 모델이 보여요.

그런데 지금까지 W와 b는 무작위였어요. 이 숫자를 **데이터에 맞게 고치는 방법**, 즉 학습이 다음 글의 주제예요.

> ← 이전: [기초편 A](/blog/2026/10/31/pytorch-00a-tensor-numbers/) · → 다음: [기초편 C. loss와 학습 루프](/blog/2026/10/31/pytorch-00c-loss-train-loop/) · [전체 목차](/blog/2026/10/25/pytorch-series/)

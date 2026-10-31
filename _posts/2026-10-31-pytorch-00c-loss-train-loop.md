---
layout: post
title: "PyTorch 기초편 C — loss와 학습 루프: 숫자를 조금씩 고쳐 가기"
date: 2026-10-31 06:20:00 +0900
permalink: /blog/2026/10/31/pytorch-00c-loss-train-loop/
categories: [pytorch, tutorial]
tags: [pytorch, loss, gradient, optimizer, training, beginner, pytorch-series]
---

> **PyTorch 시리즈 · 기초편 C/C** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [기초편 B. Linear와 순전파](/blog/2026/10/31/pytorch-00b-linear-forward/) · → 다음: [1장. 텐서 속을 열어 보기](/blog/2026/10/25/pytorch-01-tensor/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/00c-loss-train-loop](https://github.com/kjoh94/pytorch-series-examples/tree/main/00c-loss-train-loop) (`python example.py`)

[기초편 B](/blog/2026/10/31/pytorch-00b-linear-forward/)에서 모델은 `x @ W.T + b`를 계산하는 기계라는 걸 봤어요. 그런데 W와 b가 무작위라서, 지금 모델은 아무 말이나 하는 상태예요. 이 글에서는 **데이터를 보고 W와 b를 스스로 고치게** 만들 거예요. 그게 학습이에요.

학습은 딱 세 가지 질문으로 정리돼요.

1. **얼마나 틀렸나?** → loss
2. **어느 쪽으로 고쳐야 덜 틀리나?** → gradient
3. **실제로 고치기** → optimizer

이 세 개를 반복하는 게 **학습 루프**예요. 마지막엔 직선 맞추기와 2-클래스 분류를 처음부터 끝까지 직접 돌려 볼게요.

## 1. loss: 정답과 얼마나 먼가

![점들과 파란 예측 직선 사이를 빨간 세로선(오차)으로 이은 그림, 옆에 MSE = 평균(오차²)](/assets/img/posts/2026-10-31-pytorch-00c/fig01.jpg)

*그림 1. 금색 점이 정답, 파란 선이 모델의 예측이에요. 빨간 선 길이가 각 점의 오차예요. 오차를 제곱해서 평균 낸 것이 MSE loss예요. 작을수록 잘 맞는 거예요.*

**loss(손실)** 는 "모델이 얼마나 틀렸는지"를 숫자 하나로 요약한 점수예요. 시험 점수의 반대라고 생각하면 돼요. 높을수록 나쁘고, 0이면 완벽해요.

숫자를 맞히는 문제(회귀)에서 가장 기본적인 loss는 **MSE(평균 제곱 오차)** 예요.

```python
import torch
import torch.nn as nn

pred = torch.tensor([2.5, 0.0, 2.0])   # 모델 예측
true = torch.tensor([3.0, -0.5, 2.0])  # 정답

mse = ((pred - true) ** 2).mean()
print(mse)                      # 0.1667
print(nn.MSELoss()(pred, true)) # 같은 값
```

차례대로 보면

1. `pred - true` → 오차 `[-0.5, 0.5, 0.0]`
2. `** 2` → 제곱 `[0.25, 0.25, 0.0]` (부호를 없애고, 큰 오차를 더 크게 벌줌)
3. `.mean()` → 평균 `0.1667`

기초편 A에서 배운 칸끼리 연산과 `mean()`만으로 만들어져요.

### 왜 제곱하나?

그냥 오차를 더하면 +0.5와 −0.5가 상쇄돼서 0이 돼 버려요. 제곱하면 전부 양수가 되고, 크게 틀린 것을 특히 크게 벌줘요. 절댓값(`abs`)을 쓰는 L1 loss도 있지만, 처음엔 MSE면 충분해요.

### 분류 문제의 loss는?

"고양이냐 개냐" 같은 문제에서는 **cross entropy** 계열을 써요. 직관은 "정답에 준 확률이 낮을수록 크게 벌준다"예요. 정답에 확률 0.9를 줬으면 loss가 작고, 0.1을 줬으면 loss가 커요. 6절의 분류 예제에서 `BCEWithLogitsLoss`로 써 볼게요.

## 2. gradient: 어느 쪽으로 밀어야 내려가나

![포물선 f(w)=w² 위의 w=2 지점에 접선(기울기 +4)과 왼쪽 아래로 향하는 빨간 화살표, 위에 w ← w − lr × grad](/assets/img/posts/2026-10-31-pytorch-00c/fig02.jpg)

*그림 2. 산비탈에 서 있다고 상상해 보세요. 기울기(gradient)가 +4라는 건 "오른쪽으로 가면 오르막"이라는 뜻이에요. 내려가려면 반대쪽, 왼쪽으로 한 걸음. 이게 경사하강법이에요.*

loss를 줄이고 싶은데, W를 키워야 할까요 줄여야 할까요? 이 질문에 답하는 게 **gradient(기울기)** 예요.

가장 간단한 예로 `f(w) = w²`를 생각해 볼게요. 최솟값은 w = 0이에요. 지금 w = 2에 서 있다면

- 기울기 = `2w` = **4** (양수 → 오른쪽이 오르막)
- 그러니 **왼쪽(작아지는 쪽)** 으로 가야 내려가요.

PyTorch는 이 기울기를 **자동으로** 계산해 줘요.

```python
w = torch.tensor(2.0, requires_grad=True)   # "이 숫자의 기울기가 궁금해"
f = w ** 2
f.backward()                                # 기울기 계산!
print(w.grad)                               # tensor(4.)
```

- `requires_grad=True`: 이 텐서를 학습할 숫자로 표시해요. `nn.Linear`의 weight와 bias는 처음부터 이 표시가 붙어 있어요.
- `f.backward()`: f를 기준으로 거꾸로 거슬러 올라가며 기울기를 계산해요. 이걸 **역전파(backpropagation)** 라고 불러요.
- `w.grad`: 계산된 기울기가 여기에 저장돼요.

이 자동 계산이 어떻게 돌아가는지는 [3장 Autograd](/blog/2026/10/25/pytorch-03-autograd/)에서 자세히 볼 거예요. 지금은 "`backward()`를 부르면 `.grad`에 방향이 적힌다"만 알면 돼요.

### 손으로 몇 걸음 내려가 보기

규칙은 하나예요. **`w ← w − lr × grad`**. lr(learning rate, 학습률)은 한 걸음의 크기예요.

```python
w = torch.tensor(2.0, requires_grad=True)
for step in range(5):
    f = w ** 2
    f.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad     # 한 걸음
    w.grad.zero_()            # 기울기 지우기 (중요!)
    print(step, round(w.item(), 4), round(f.item(), 4))
# 0 1.6    4.0
# 1 1.28   2.56
# 2 1.024  1.6384
# 3 0.8192 1.0486
# 4 0.6554 0.6711
```

w가 2 → 1.6 → 1.28 → … 로 0을 향해 가고, f도 줄어들어요. 학습이 바로 이거예요.

여기 나온 두 가지 디테일:

- **`torch.no_grad()`**: w를 고치는 계산 자체는 기울기 기록에 넣으면 안 돼요. 기초편 B에서 본 그것이에요.
- **`w.grad.zero_()`**: PyTorch는 `backward()`를 부를 때마다 기울기를 **더해서 쌓아요**. 지우지 않으면 지난 걸음의 기울기가 계속 섞여요. 초보자가 가장 많이 빼먹는 줄이에요.

### 학습률(lr) 고르기

- 너무 작으면(0.0001): 한참 걸려요.
- 너무 크면(1.5): 골짜기를 넘어 반대편으로 튀어 올라가고, loss가 오히려 커져요. 직접 `0.1`을 `1.5`로 바꿔 돌려 보세요.
- 처음엔 0.01~0.1 사이에서 시작해 보는 게 보통이에요.

## 3. 학습 루프: 다섯 박자

![① 예측 → ② loss → ③ zero_grad → ④ backward → ⑤ step 이 원형으로 이어진 그림](/assets/img/posts/2026-10-31-pytorch-00c/fig03.jpg)

*그림 3. PyTorch 학습 코드는 거의 전부 이 다섯 줄의 반복이에요. 한 바퀴를 한 step이라고 불러요.*

손으로 하던 걸 PyTorch 부품으로 바꾸면 이 다섯 줄이 돼요.

```python
pred = model(x)              # ① 예측 (순전파)
loss = loss_fn(pred, y)      # ② 채점
optimizer.zero_grad()        # ③ 지난 기울기 지우기
loss.backward()              # ④ 기울기 계산 (역전파)
optimizer.step()             # ⑤ 모든 파라미터를 한 걸음씩
```

**optimizer**는 2절의 `w -= lr * w.grad`와 `w.grad.zero_()`를 **모델의 모든 파라미터에 대해** 대신 해 주는 도우미예요. 파라미터가 41개든 70억 개든 코드는 똑같아요.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
```

`model.parameters()`는 기초편 B에서 셌던 그 숫자들(weight, bias)이에요. SGD 말고 `Adam`도 많이 쓰는데, "걸음 크기를 똑똑하게 조절하는 SGD"라고 생각하면 돼요. 사용법은 똑같아요.

## 4. 처음부터 끝까지: 직선 맞추기

![금색 데이터 점들 위에 step 0(평평), step 20, step 100, step 500 직선이 점점 점들에 맞춰지는 그림](/assets/img/posts/2026-10-31-pytorch-00c/fig04.jpg)

*그림 4. 처음엔 엉뚱한 직선(회색)이었다가, step이 쌓일수록 정답 직선 y = 2x + 1에 가까워져요. 모델이 "w는 2쯤, b는 1쯤"이라는 걸 데이터에서 스스로 찾아낸 거예요.*

정답이 `y = 2x + 1`인 데이터를 만들고(잡음을 살짝 섞어서), 모델이 w = 2, b = 1을 찾아내는지 볼게요.

### 4-1. 데이터 만들기

```python
torch.manual_seed(0)
X = torch.linspace(0, 5, 50).unsqueeze(1)    # (50, 1)
Y = 2 * X + 1 + 0.3 * torch.randn_like(X)    # (50, 1)
```

`unsqueeze(1)`은 `(50,)`을 `(50, 1)`로 바꿔요. 샘플 50개, 특징 1개. 기초편 B에서 본 "맨 앞은 배치" 규칙에 맞추는 거예요.

### 4-2. 모델 만들기

```python
class LineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)    # 입력 1개 → 출력 1개: y = w·x + b
    def forward(self, x):
        return self.linear(x)

model = LineModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
```

`nn.Linear(1, 1)`은 파라미터가 딱 두 개(w, b)인 모델이에요. 중학교 때 배운 일차함수 그대로예요.

### 4-3. 학습 루프 돌리기

```python
for step in range(501):
    pred = model(X)
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, round(loss.item(), 4),
              round(model.linear.weight.item(), 3),
              round(model.linear.bias.item(), 3))
```

실제로 돌려 보면 이렇게 나와요.

```
step   0  loss=41.0900  w=0.780  b=0.459
step 100  loss=0.1019  w=2.044  b=0.898
step 200  loss=0.1017  w=2.038  b=0.917
step 300  loss=0.1017  w=2.036  b=0.924
step 400  loss=0.1017  w=2.035  b=0.927
step 500  loss=0.1017  w=2.035  b=0.928
```

loss가 41에서 0.1로 떨어지고, w는 2 근처, b는 1 근처로 왔어요. 우리가 정답을 알려 준 적이 없는데, **데이터만 보고 찾아낸 거예요.** loss가 0이 안 되는 건 일부러 섞은 잡음 때문이에요. 잡음까지 맞출 수는 없으니까요.

### 4-4. 학습한 모델 써 보기

```python
model.eval()
with torch.no_grad():
    print(model(torch.tensor([[10.0]])))   # 약 21.27 (정답 2·10+1 = 21)
```

`model.eval()`은 "이제 학습 끝, 시험 모드"라는 표시예요. 이 모델엔 차이가 없지만, 나중에 Dropout 같은 층이 있으면 중요해요. `no_grad()`는 예측만 할 땐 기울기 기록이 필요 없어서 메모리와 시간을 아껴 줘요.

## 5. loss 곡선 읽기

![가로축 step, 세로축 loss. 처음에 급하게 떨어지다가 평평해지는 빨간 곡선](/assets/img/posts/2026-10-31-pytorch-00c/fig05.jpg)

*그림 5. 건강한 학습의 loss 곡선은 보통 이렇게 생겼어요. 처음엔 빠르게 떨어지고, 나중엔 천천히 평평해져요. 평평해지면 대체로 배울 만큼 배운 거예요.*

학습할 때는 loss를 꼭 찍어 보세요. 모양으로 상태를 진단할 수 있어요.

| loss 모양 | 뜻 | 해볼 것 |
|---|---|---|
| 빠르게 내려가다 평평 | 정상 | 그대로 |
| 거의 안 내려감 | lr이 너무 작거나 모델이 너무 단순 | lr 키우기, 층 늘리기 |
| 출렁이거나 점점 커짐 | lr이 너무 큼 | lr 줄이기 |
| `nan`이 나옴 | 숫자가 폭발함 | lr 확 줄이기, 입력 크기 확인 |
| 학습 loss만 계속 줄고 새 데이터에선 나쁨 | 외워 버림(과적합) | 데이터 늘리기, 모델 줄이기 |

## 6. 보너스: 2-클래스 분류

이번엔 숫자 대신 "A냐 B냐"를 맞혀 볼게요. 점 구름 두 개를 만들고, 경계선을 학습시켜요.

```python
A = torch.randn(100, 2) + torch.tensor([2., 2.])     # 오른쪽 위 구름 → 정답 1
B = torch.randn(100, 2) + torch.tensor([-2., -2.])   # 왼쪽 아래 구름 → 정답 0
Xc = torch.cat([A, B])                                # (200, 2)
Yc = torch.cat([torch.ones(100), torch.zeros(100)]).unsqueeze(1)   # (200, 1)

clf = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.SGD(clf.parameters(), lr=0.1)
bce = nn.BCEWithLogitsLoss()

for step in range(200):
    logit = clf(Xc)
    loss = bce(logit, Yc)
    opt.zero_grad(); loss.backward(); opt.step()

acc = ((clf(Xc) > 0).float() == Yc).float().mean()
print(loss.item(), acc.item())   # loss 약 0.035, 정확도 약 98.5%
```

달라진 건 세 군데뿐이에요.

- **모델**: 기초편 B의 작은 MLP(Linear → ReLU → Linear) 그대로.
- **loss**: `BCEWithLogitsLoss`. 내부에서 sigmoid로 0~1 확률을 만든 뒤 "정답에 준 확률이 낮으면 크게 벌주는" 계산을 해요. 마지막 층 출력(logit)을 sigmoid 없이 그대로 넣는 게 규칙이에요(그게 숫자상 더 안정적이에요).
- **판정**: logit > 0 이면 sigmoid > 0.5, 즉 "1번 클래스"로 판정.

**학습 루프 다섯 줄은 똑같아요.** 이게 핵심이에요. 문제가 바뀌어도 모델과 loss만 바꾸면 돼요. 이미지 분류든 언어 모델이든 이 뼈대 그대로예요.

## 7. 초보자가 자주 하는 실수

1. **`zero_grad()` 빼먹기** → 기울기가 쌓여서 학습이 이상하게 튀어요.
2. **`loss.backward()` 전에 `step()`** → 순서가 중요해요. 기울기를 계산한 다음 걸어야죠.
3. **pred와 정답 shape이 다름** → `(50,)` vs `(50, 1)`이면 브로드캐스팅 때문에 `(50, 50)`짜리 엉뚱한 loss가 나와요. 경고 메시지를 꼭 읽고 shape을 맞추세요.
4. **`loss`를 리스트에 그대로 저장** → 기울기 기록까지 붙어 메모리가 새요. `loss.item()`으로 숫자만 저장하세요.
5. **lr 너무 큼** → loss가 `nan`. 10배씩 줄여 보세요.

## 정리

- **loss**: 얼마나 틀렸는지 숫자 하나. 회귀는 MSE, 분류는 cross entropy 계열.
- **gradient**: 파라미터를 어느 쪽으로 밀어야 loss가 오르는지. 반대로 가면 내리막. `backward()`가 자동 계산.
- **optimizer**: `w ← w − lr × grad`를 모든 파라미터에 적용.
- **학습 루프 다섯 박자**: 예측 → 채점 → 지우기 → 역전파 → 한 걸음.
- 직선 맞추기에서 모델은 정답 w=2, b=1을 데이터만 보고 찾아냈어요.

여기까지가 기초편이에요. 텐서를 만들고(A), 층으로 계산하고(B), 학습시키는(C) 흐름이 손에 잡혔다면 이제 본편으로 넘어갈 준비가 됐어요. [1장](/blog/2026/10/25/pytorch-01-tensor/)에서는 오늘 "숫자 상자"라고 부른 텐서가 메모리에서 실제로 어떻게 생겼는지 열어 볼게요.

> ← 이전: [기초편 B](/blog/2026/10/31/pytorch-00b-linear-forward/) · → 다음: [1장. 텐서 속을 열어 보기](/blog/2026/10/25/pytorch-01-tensor/) · [전체 목차](/blog/2026/10/25/pytorch-series/)

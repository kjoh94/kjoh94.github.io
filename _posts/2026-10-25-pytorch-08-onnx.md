---
layout: post
title: "PyTorch 시리즈 8장 — 어디서나 열리는 모델 파일 만들기: PyTorch에서 ONNX로"
date: 2026-10-25 21:08:00 +0900
categories: [pytorch]
tags: [pytorch, onnx, onnxruntime]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [7. 모델을 미리 통째로 뽑아 두기](/blog/2026/10/25/pytorch-07-torch-export/)  
> → 다음: [9. 컴파일러는 무엇을 빠르게 하나](/blog/2026/10/25/pytorch-09-compiler-opt/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/08-onnx](https://github.com/kjoh94/pytorch-series-examples/tree/main/08-onnx) (`python example.py`)

> **처음이라면 이것만 기억하세요.** ONNX는 "어떤 프로그램에서도 열리는 모델 파일"이에요. 이번 장에서는 7장에서 뽑은 그래프를 `.onnx` 파일로 저장하고, **PyTorch 없이** 실행해서 숫자가 똑같이 나오는지 확인해요. opset, dynamic shape 같은 말은 비유부터 보고 갈게요.

[7장](/blog/2026/10/25/pytorch-07-torch-export/)에서 모델을 `torch.export`로 "인쇄된 번역서"처럼 뽑았죠. 그런데 그 `.pt2` 파일은 여전히 **PyTorch가 있어야** 읽을 수 있어요. 휴대폰 앱이나 다른 회사의 런타임은 PyTorch를 모를 수도 있어요. 그래서 모두가 읽을 수 있는 **공용 형식**으로 한 번 더 바꿔 줍니다. 그게 ONNX예요.

## 0. 비유로 먼저

### 0-1. ONNX = 모델계의 PDF

![ONNX는 PDF](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig07.jpg)

**그림 7.** 워드로 쓴 문서를 PDF로 저장하면 워드가 없는 컴퓨터에서도 똑같이 열리죠. ONNX도 같아요. PyTorch로 만든 모델을 `.onnx`로 저장하면, PyTorch가 없는 런타임이나 컴파일러에서도 같은 계산을 할 수 있어요.

### 0-2. opset = 사전의 판 번호

![opset](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig08.jpg)

**그림 8.** ONNX 파일 안의 연산(`Gemm`, `Relu` 같은 단어)은 "ONNX 사전"에 뜻이 적혀 있어요. 이 사전은 해마다 개정판이 나오고, 그 판 번호가 **opset**이에요. 파일이 opset 20으로 쓰였다면, 받는 쪽도 opset 20을 읽을 수 있어야 해요. 오래된 런타임에 보낼 땐 opset을 낮춰서 저장하기도 해요.

### 0-3. dynamic shape = 서류의 빈칸

![dynamic shape](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig09.jpg)

**그림 9.** 7장에서 "입력 크기는 기본이 고정"이라는 함정을 봤죠. ONNX도 같아요. 배치 크기를 숫자 `4`로 박아 두면 4개짜리만 받아요. 7장처럼 `Dim`으로 선언하면, ONNX 파일 안에 `batch`라는 **이름 붙은 빈칸**으로 남아서 몇 개를 넣어도 돼요.

### 0-4. 직접 해 보기: 저장 → PyTorch 없이 실행 → 비교

필요한 패키지: `pip install onnx onnxruntime onnxscript`

```python
import torch, onnx, onnxruntime as ort, numpy as np

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)
    def forward(self, x):
        return torch.relu(self.fc(x))

m = M().eval()
batch = torch.export.Dim("batch", min=1, max=64)          # 빈칸 선언 (7장)
prog = torch.onnx.export(m, (torch.randn(4, 3),), dynamo=True,
                         dynamic_shapes={"x": {0: batch}})
prog.save("m.onnx")

g = onnx.load("m.onnx")
print([n.op_type for n in g.graph.node])   # ['Gemm', 'Relu']
print([o.version for o in g.opset_import if o.domain == ""])   # [20]

sess = ort.InferenceSession("m.onnx")      # 여기부터 PyTorch 안 씀
name = sess.get_inputs()[0].name
for bs in (4, 7):
    x = torch.randn(bs, 3)
    out = sess.run(None, {name: x.numpy()})[0]
    print(bs, out.shape, np.abs(out - m(x).detach().numpy()).max())
# 4 (4, 2) 0.0
# 7 (7, 2) 5.96e-08
```

실제로 돌려 본 결과예요. 볼 포인트:

1. PyTorch의 `linear`가 ONNX에선 `Gemm` 한 개로 바뀌었어요. 번역하면서 "단어"가 바뀌는 거죠.
2. 이 파일은 opset 20으로 쓰였어요(PyTorch 버전에 따라 다를 수 있어요).
3. 예시 입력은 4개였는데 **7개를 넣어도** 잘 돌아요. 빈칸 덕분이에요.
4. 차이가 `0.0`이나 `6e-08`처럼 아주 작아요. 이 정도는 소수점 계산 순서 차이라 "같다"고 봐요. 이렇게 **항상 숫자로 비교하는 습관**이 이 장의 핵심이에요(5절).

이제 정식 용어로 하나씩 자세히 볼게요.

## 1. ONNX란?

![ONNX](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig01.jpg)

**그림 1.** ONNX는 프레임워크 중립적인 **모델 파일 형식**이다. 파일 안에는 그래프(노드와 입출력), 가중치(initializer), 그리고 각 op의 의미를 정하는 **opset 버전**이 들어 있다. PyTorch로 만들고, ONNX Runtime이나 다른 컴파일러가 읽는다. "PyTorch를 몰라도 이 파일만 보면 계산을 재현할 수 있다"가 핵심 약속이다.

## 2. 두 개의 exporter

![exporters](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig02.jpg)

**그림 2.** 예전 exporter(`dynamo=False`)는 TorchScript trace로 실행을 기록했다. 분기를 조용히 고정하는 문제(7장 그림 7)를 그대로 물려받았다. 새 exporter(`dynamo=True`)는 **`torch.export`로 ExportedProgram을 만든 뒤** ATen → ONNX 변환 테이블로 옮긴다. 새 코드는 `dynamo=True`를 쓰자.

```python
import torch, numpy as np
import onnx, onnxruntime as ort
from torch.export import Dim

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 16)
        self.fc2 = torch.nn.Linear(16, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP().eval()
x = torch.randn(3, 4)
onnx_prog = torch.onnx.export(model, (x,), dynamo=True,
                              input_names=["x"], output_names=["y"],
                              dynamic_shapes={"x": {0: Dim("batch")}})
onnx_prog.save("mlp.onnx")
```

## 3. ONNX 그래프의 모양

![graph](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig03.jpg)

**그림 3.** 노드 하나는 `op_type`(Gemm, Relu…), 입력 이름들, 출력 이름들, 속성으로 이뤄진다. 가중치는 그래프 밖의 **이름 붙은 상수(initializer)** 로 들어간다 — 7장의 함수형 그래프와 같은 생각이다. 파일을 열어 직접 보자.

```python
m = onnx.load("mlp.onnx")
onnx.checker.check_model(m)
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
for n in m.graph.node:
    print(f"{n.op_type:10s} {list(n.input)} -> {list(n.output)}")
print("initializers:", [i.name for i in m.graph.initializer])
```

## 4. dynamic axes

![dynamic](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig04.jpg)

**그림 4.** 입력 shape 차원이 숫자로 적혀 있으면 고정, 문자열(`"batch"`)이면 동적이다. 7장의 `Dim`이 그대로 ONNX의 심볼 차원이 된다.

```python
def dims(vi):
    return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]
print("input :", dims(m.graph.input[0]))     # ['batch', 4]
print("output:", dims(m.graph.output[0]))
```

## 5. 검증: 숫자가 같은가

![verify](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig05.jpg)

**그림 5.** export가 에러 없이 끝났다고 정답인 건 아니다. 같은 입력을 PyTorch와 ONNX Runtime에 넣고 **최대 오차**를 본다. 동적 차원은 여러 크기로 시험한다.

```python
sess = ort.InferenceSession("mlp.onnx", providers=["CPUExecutionProvider"])
for b in [1, 3, 17]:
    xb = torch.randn(b, 4)
    with torch.no_grad():
        ref = model(xb).numpy()
    out = sess.run(None, {"x": xb.numpy()})[0]
    print(b, out.shape, "max err:", float(np.abs(ref - out).max()))
    assert np.allclose(ref, out, atol=1e-5)
```

## 6. 자주 만나는 함정

![pitfalls](/assets/img/posts/2026-10-25-pytorch-08-onnx/fig06.jpg)

**그림 6.** 에러는 대개 넷 중 하나다. ① 변환 테이블에 없는 op → 분해하거나 커스텀 함수(12장). ② 텐서 값에 의존하는 분기 → `torch.cond`로 바꾸거나 모델 밖으로. ③ opset 버전 차이 → `opset_version` 지정. ④ 동적 차원을 선언 안 해서 고정됨 → `dynamic_shapes`. 아래는 ②의 해법인 `torch.cond` 예다.

```python
class Branchy(torch.nn.Module):
    def forward(self, x):
        return torch.cond(x.sum() > 0, lambda t: t * 2, lambda t: t - 1, (x,))

ep = torch.export.export(Branchy(), (torch.randn(3),))
print([str(n.target) for n in ep.graph.nodes if n.op == "call_function"])  # cond 가 그래프에 남는다
```

## 자주 묻는 질문

**Q. .pt2(7장)와 .onnx 중 뭘 저장해야 하나요?**
PyTorch 안에서만 쓰면 `.pt2`로 충분해요. PyTorch가 없는 곳으로 보낸다면 `.onnx`예요.

**Q. opset은 몇으로 해야 하나요?**
받는 쪽 런타임이 지원하는 가장 높은 번호가 무난해요. 모르면 기본값으로 먼저 저장하고, 받는 쪽에서 에러가 나면 낮춰 보세요.

**Q. 결과가 조금 다르게 나와요. 버그인가요?**
`1e-5` 정도 이하면 보통 정상이에요. 그보다 크면 지원 안 되는 연산이 다른 방식으로 번역됐을 수 있으니 6절 함정을 확인하세요.

**Q. dynamo=True는 왜 붙이나요?**
7장의 `torch.export`를 거쳐 ONNX로 바꾸는 최신 방식이라서예요. 예전 방식(TorchScript 기반)은 점점 저물고 있어요(2절).

## 퀴즈

1. ONNX를 한마디로 비유하면? → 모델계의 PDF.
2. 배치 크기를 고정하지 않으려면? → `Dim`으로 선언 → ONNX에서 이름 붙은 빈칸(`batch`)이 된다.
3. 내보낸 뒤 꼭 해야 하는 일은? → ONNX Runtime 결과와 PyTorch 결과를 숫자로 비교.

## 정리

- ONNX = 어디서나 열리는 모델 파일(모델계의 PDF). 안에는 그래프 + 가중치(initializer) + opset(사전 판 번호).
- `torch.onnx.export(..., dynamo=True)` = export → ONNX 변환.
- 동적 차원은 문자열 dim으로 남는다.
- 항상 ONNX Runtime 결과와 숫자로 비교한다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [7. 모델을 미리 통째로 뽑아 두기](/blog/2026/10/25/pytorch-07-torch-export/)  
> → 다음: [9. 컴파일러는 무엇을 빠르게 하나](/blog/2026/10/25/pytorch-09-compiler-opt/)

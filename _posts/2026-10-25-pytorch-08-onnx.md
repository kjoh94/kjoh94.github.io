---
layout: post
title: "PyTorch 시리즈 8장 — PyTorch에서 ONNX로"
date: 2026-10-25 21:08:00 +0900
categories: [pytorch]
tags: [pytorch, onnx, onnxruntime]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [7. torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/)  
> → 다음: [9. 컴파일러 최적화 감각 — folding·fusion·메모리](/blog/2026/10/25/pytorch-09-compiler-opt/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/08-onnx](https://github.com/kjoh94/pytorch-series-examples/tree/main/08-onnx) (`python example.py`)

7장에서 모델을 `ExportedProgram`으로 뽑았다. 이 장에서는 그 그래프를 **ONNX 파일**로 바꾸고, ONNX Runtime으로 실행해서 **숫자가 같은지** 확인하는 전 과정을 따라간다.

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

## 정리

- ONNX = 그래프 + initializer + opset.
- `torch.onnx.export(..., dynamo=True)` = export → ONNX 변환.
- 동적 차원은 문자열 dim으로 남는다.
- 항상 ONNX Runtime 결과와 숫자로 비교한다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [7. torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/)  
> → 다음: [9. 컴파일러 최적화 감각 — folding·fusion·메모리](/blog/2026/10/25/pytorch-09-compiler-opt/)

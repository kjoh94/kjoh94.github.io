---
layout: post
title: "PyTorch 시리즈 — 목차: 텐서부터 커스텀 op까지 12장"
date: 2026-10-25 21:00:00 +0900
categories: [pytorch, tutorial]
tags: [pytorch, pytorch-series]
---

PyTorch를 "쓸 줄 아는" 단계에서 "**안에서 무슨 일이 일어나는지 아는**" 단계로 가기 위한 시리즈다. 모델을 학습시키는 법보다는, 학습된 모델을 **이해하고, 옮기고, 컴파일하고, 가속기에 올리는** 데 필요한 개념만 골랐다. 각 장은 한 편의 긴 글이고, 3Blue1Brown 스타일의 다이어그램과 바로 돌려 볼 수 있는 작은 예제 코드로 구성된다.

![시리즈 지도](/assets/img/posts/2026-10-25-pytorch-series/fig01.jpg)

**그림 1.** 시리즈 지도. 왼쪽 **기초(1–3장)**: 텐서·모듈·autograd. 가운데 **내부 구조(4–5장)**: 연산이 커널로 가는 길(ATen·Dispatcher)과, 거기에 새 장치를 꽂는 법. 오른쪽 **그래프와 배포(6–12장)**: 파이썬 코드를 그래프로 잡고(FX·Dynamo·export), 다른 런타임으로 옮기고(ONNX), 최적화·양자화하고, LLM 추론과 커스텀 op까지.

## 목차

| 장 | 제목 | 이것만은 꼭 |
|---|---|---|
| 1 | [Tensor의 정체 — storage·stride·view·contiguous·dtype·device](/blog/2026/10/25/pytorch-01-tensor/) | 텐서는 "메모리를 읽는 규칙". 복사가 언제 생기나 |
| 2 | [nn.Module과 state_dict](/blog/2026/10/25/pytorch-02-module-state-dict/) | 파라미터/버퍼, 이름표 사전, 모듈 교체 |
| 3 | [Autograd와 추론 모드](/blog/2026/10/25/pytorch-03-autograd/) | 계산 그래프, `no_grad`, `inference_mode` |
| 4 | [ATen과 Dispatcher](/blog/2026/10/25/pytorch-04-aten-dispatcher/) | `torch.mm` 한 줄이 커널까지 가는 길, dispatch key |
| 5 | [커스텀 디바이스 백엔드 (PrivateUse1)](/blog/2026/10/25/pytorch-05-privateuse1/) | 새 장치를 PyTorch에 등록하는 패턴, CPU fallback |
| 6 | [그래프 캡처 — FX·Dynamo·graph break·재컴파일](/blog/2026/10/25/pytorch-06-graph-capture/) | `torch.compile`이 파이썬을 잡는 법, guard |
| 7 | [torch.export와 Core ATen IR](/blog/2026/10/25/pytorch-07-torch-export/) | AOT 그래프, ExportedProgram, TorchScript가 저문 이유 |
| 8 | [PyTorch → ONNX](/blog/2026/10/25/pytorch-08-onnx/) | opset, dynamic shape, 커스텀 op 매핑 |
| 9 | [컴파일러 최적화 감각 — folding·fusion·메모리 IO](/blog/2026/10/25/pytorch-09-compiler-opt/) | 왜 "연산 수"보다 "메모리 왕복"이 중요한가 |
| 10 | [양자화, PyTorch 쪽에서](/blog/2026/10/25/pytorch-10-quantization/) | scale/zero-point, weight-only, PTQ |
| 11 | [LLM 추론을 코드로 — prefill·decode·KV cache](/blog/2026/10/25/pytorch-11-llm-inference/) | 동적 shape, 정적 캐시 |
| 12 | [커스텀 op — torch.library와 C++/CUDA extension](/blog/2026/10/25/pytorch-12-custom-ops/) | 내 커널을 컴파일러가 이해하게 등록하기 |

**추천 순서**: 1 → 2 → 4 → 5 → 6 → 7 → 8을 먼저. 3·9·10·11·12는 필요할 때.

## 함께 읽으면 좋은 글

- [PyTorch로 학습한 모델은 어떻게 허깅페이스에 '착륙'하는가](/blog/2026/10/25/pytorch-huggingface-landing/) — 2장(state_dict) 다음에 읽으면 좋은 **번외편**. safetensors·config·GGUF까지.
- [ATen이란? (초기 정리본)](/blog/2026/10/25/pytorch-aten-dispatcher/) — 4장의 바탕이 된 첫 정리.
- [TensorRT 추론 기초](/blog/2026/10/25/tensorrt-inference-basics/) — 8장(ONNX) 이후 실전 배포 예.
- [On-Device LLM: Prefill·Decode·KV 캐시](/blog/2026/10/25/ondevice-llm-prefill-decode-kv-cache/) — 11장의 개념편.

모든 예제는 공개된 PyTorch 2.x(작성 시점 2.14 CPU 빌드에서 확인) 기준이다.

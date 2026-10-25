---
layout: post
title: "TensorRT 추론 기초 — ONNX부터 엔진 빌드, 실행까지"
date: 2026-10-25 17:00:00 +0900
categories: [gpu, tensorrt, tutorial]
tags: [tensorrt, onnx, cuda, inference, fp16, int8, resnet18, pytorch]
---

[지난 글](/blog/2026/10/19/runpod-cuda-taster/)에서는 RunPod GPU 위에서 CUDA 커널을 직접 짜서 CPU와 속도를 비교했다. 이번에는 한 단계 위로 올라간다. **이미 학습된 PyTorch 모델을 NVIDIA TensorRT로 "추론 전용 실행 계획"으로 바꾸고 돌리는 과정**을, 파일 하나·API 호출 하나 단위까지 따라간다.

- 코드: <https://github.com/kjoh94/tensorrt-resnet-example> (PyTorch ResNet18 → ONNX → TensorRT 엔진 → 추론)
- 이 글은 NVIDIA 공개 문서와 공개 예제 수준의 일반 내용이다. 그림 속 숫자 중 "예시"라고 적힌 것은 이해를 돕기 위한 값이다.

## 0. 전체 큰 루프 — 한 장으로 먼저

![TensorRT 전체 파이프라인](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig01.jpg)
*그림 1. TensorRT의 전체 흐름. 위쪽 점선 상자가 **① 빌드 단계**다: PyTorch 모델을 ONNX 파일로 내보내고(export), OnnxParser가 그 그래프를 TensorRT 네트워크로 옮기고, Builder가 레이어 퓨전·정밀도 선택·커널 측정을 거쳐 `build_serialized_network`로 엔진 bytes를 만든다. 이건 **한 번만**(수 초~수 분) 한다. 아래쪽이 **② 추론 루프**다: Runtime이 엔진 파일을 복원(deserialize)하고, ExecutionContext가 스트림에 실행을 예약(enqueue)하면 출력(logits → top-5)이 나온다. 이미지가 들어올 때마다 Context → 출력 부분만 **반복**된다(ms 단위).*

이 글 전체는 이 그림의 상자를 왼쪽 위부터 하나씩 열어 보는 순서로 간다. 핵심 한 줄은 이것이다.

> **무거운 고민(무엇을 합치고, 어떤 커널을 쓸지)은 빌드 때 한 번에 끝내고, 실행 때는 정해진 계획만 빠르게 반복한다.**

![큰 루프 안의 작은 루프](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig19.jpg)
*그림 2. 두 단계 각각 안쪽에도 루프가 있다. **빌드**: 퓨전이 끝난 레이어마다, 허용된 정밀도의 후보 커널(tactic)들을 실제 GPU에서 돌려 보고 가장 빠른 것을 계획표에 적는다. 즉 "레이어 수 × 후보 수"만큼 측정한다 — 빌드가 느린 이유다. **추론**: 이미지마다 CPU 전처리 → GPU로 복사 → `execute_async_v3` 한 번(안에서는 퓨전된 커널 k개가 순서대로 실행) → 결과 복사 → 동기화 → softmax/top-5. 반복되는 건 이 바깥 루프뿐이고, 커널 선택은 다시 하지 않는다.*

## 1. 왜 CUDA 다음에 TensorRT인가

![CUDA vs PyTorch vs TensorRT](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig02.jpg)
*그림 3. 같은 GPU를 쓰는 세 가지 방식. **CUDA C++**는 커널을 내가 직접 짜고 스레드·블록·메모리를 직접 관리한다 — 자유도가 가장 높고 수고도 가장 크다. **PyTorch eager**는 연산 하나마다 미리 만들어 둔 커널 하나를 파이썬이 한 줄씩 지시한다 — 편하지만 연산 사이 빈틈이 많다. **TensorRT**는 그래프 전체를 한 번에 보고, 합치고, 줄이고, 이 GPU에 가장 빠른 커널을 골라 추론 전용 계획을 만든다.*

지난 글의 벡터 덧셈 커널을 떠올려 보자. ResNet18 하나만 해도 Conv·BatchNorm·ReLU·Add·Pool·FC가 수십 개 이어진다. 이걸 다 손으로 짜고 튜닝하는 건 현실적이지 않다. 그렇다고 PyTorch를 그대로 쓰면 학습용 유연성 비용(파이썬 디스패치, 연산마다 따로 실행되는 커널, 중간 결과를 매번 메모리에 썼다 읽기)을 추론 때도 그대로 낸다.

![커널 실행 타임라인](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig03.jpg)
*그림 4. 시간 축 위의 커널 실행. **Eager**는 Conv, BN, ReLU, Conv, BN, Add, ReLU가 각각 따로 실행되고, 사이사이에 빈틈(실행 지시 오버헤드 + 중간 텐서를 메모리에 쓰고 다시 읽는 시간)이 있다. **TensorRT**는 이를 "Conv+BN+ReLU", "Conv+BN+Add+ReLU" 두 개의 커널로 합쳐서 빈틈과 메모리 왕복이 줄어든다.*

추론에서 GPU가 느려지는 이유는 계산 자체보다 **메모리 왕복**과 **실행 오버헤드**인 경우가 많다. TensorRT의 최적화 대부분이 이 둘을 줄이는 쪽이다.

## 2. 1단계 — PyTorch → ONNX

![ONNX export](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig04.jpg)
*그림 5. `torch.onnx.export`는 더미 입력(1×3×224×224)을 모델에 한 번 흘려 보내면서 지나간 연산을 기록해 그래프로 저장한다(전통적 방식은 tracing). 결과는 Conv → BatchNorm → Relu → MaxPool → … → Gemm으로 이어지는 그래프와 1×1000 출력이다. **`eval()`이 필수**인 이유: Dropout과 BatchNorm이 학습 모드면 다른 그래프가 기록된다. **opset**은 "ONNX 연산 사전의 버전"으로, 어떤 연산을 어떤 의미로 쓸지 정한다.*

저장소의 `export_onnx.py`가 하는 일은 정확히 이것이다.

```python
model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "resnet18.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=17, do_constant_folding=True)
```

- `input_names` / `output_names`: 나중에 TensorRT에서 버퍼를 연결할 때 **이 이름**을 그대로 쓴다.
- `do_constant_folding=True`: 입력과 무관하게 계산되는 부분을 export 시점에 미리 계산해 넣는다.
- tracing은 **실제로 지나간 경로만** 기록한다. 모델 안에 입력값에 따라 갈리는 파이썬 `if`가 있으면 더미 입력이 지나간 쪽만 남는다. ResNet처럼 고정된 구조는 문제없다.

![ONNX 파일 구조](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig05.jpg)
*그림 6. ONNX 파일(protobuf) 안의 다섯 부분. **graph.input**: 이름·dtype·모양(`input`, float32, [1,3,224,224]). **graph.node**: 연산 목록과 연결. **graph.initializer**: 가중치 텐서 — ResNet18은 약 1,170만 개 파라미터, FP32로 약 45MB. **graph.output**: `output`, [1,1000]. **opset_import**: 연산 정의 버전. 배치 크기를 바꾸고 싶다면 export 때 `dynamic_axes`로 그 차원을 가변(-1)으로 만든다.*

확인 팁: `onnx.checker.check_model`로 형식 검사를 하고, Netron 같은 뷰어로 열어 보면 그래프가 그대로 보인다.

## 3. 2단계 — Parser: ONNX를 TensorRT 네트워크로

![OnnxParser](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig06.jpg)
*그림 7. `trt.OnnxParser`가 ONNX 노드를 하나씩 TensorRT의 `INetworkDefinition` 레이어로 번역한다. Conv → IConvolutionLayer, BatchNorm → IScaleLayer, Relu → IActivationLayer, Add → IElementWiseLayer, GlobalAveragePool → Reduce/Pooling, Gemm → MatrixMultiply 계열. 지원하지 않는 연산이나 opset이 나오면 `parse()`가 False를 돌려주므로 `parser.get_error(i)`로 원인을 반드시 출력한다.*

```python
logger  = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)          # TensorRT 10: explicit batch가 기본
parser  = trt.OnnxParser(network, logger)
with open("resnet18.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")
```

이 시점의 `network`는 아직 "무엇을 계산할지"만 담긴 그래프다. **어떻게** 계산할지는 다음 단계에서 정해진다.

## 4. 3단계 — Builder와 BuilderConfig

![Builder config](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig07.jpg)
*그림 8. Network(무엇을)와 BuilderConfig(무엇을 허용할지)를 Builder에 주면, 빌더가 실행 계획을 찾아 직렬화된 엔진 bytes를 내놓는다. Config의 주요 항목: **WORKSPACE 메모리 풀**(빌드·실행 중 임시 버퍼 한도, 예: 1GiB — 너무 작으면 빠른 커널을 못 쓴다), **FP16 플래그**(FP16 커널 허용), **INT8 플래그**(INT8 허용, 스케일 정보 필요), **Optimization Profile**(가변 입력의 min/opt/max), **Timing Cache**(커널 측정 결과를 저장해 재빌드를 빠르게).*

```python
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
config.set_flag(trt.BuilderFlag.FP16)        # '허용'이지 '강제'가 아니다
serialized = builder.build_serialized_network(network, config)
open("resnet18.engine", "wb").write(serialized)
```

같은 일을 커맨드라인 한 줄로도 할 수 있다: `trtexec --onnx=resnet18.onnx --saveEngine=resnet18.engine --fp16`. 저장소의 `build_engine_python.py`는 trtexec이 내부에서 하는 일을 Python API로 풀어 쓴 것이다.

빌더 안에서 일어나는 일은 크게 두 가지다: **그래프 최적화**와 **커널 선택**.

### 4-1. 그래프 최적화 — 레이어 퓨전

![레이어 퓨전](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig08.jpg)
*그림 9. ResNet의 BasicBlock 하나. 원래는 Conv 3×3 → BN → ReLU → Conv 3×3 → BN → Add(+x, 잔차) → ReLU, 레이어 7개다. 빌더는 BN을 Conv에 접어 넣고(아래 그림), 뒤따르는 ReLU·Add를 같은 커널 안에서 처리하도록 합쳐 **커널 2개**로 만든다. 그 밖에 상수 폴딩, 결과가 쓰이지 않는 레이어 제거, 불필요한 Reshape/Transpose 정리, 텐서 메모리 재사용도 한다.*

퓨전이 빠른 이유: ReLU는 계산이 거의 없다. 따로 실행하면 "Conv 결과를 메모리에 쓰고 → ReLU 커널이 다시 읽고 → 다시 쓴다". 합치면 Conv가 결과를 레지스터에 들고 있는 동안 바로 max(0, x)를 적용하고 한 번만 쓴다. CUDA 글에서 본 "메모리 대역폭이 병목"이라는 얘기가 여기서 그대로 효과를 낸다.

![BN 접어 넣기](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig09.jpg)
*그림 10. 추론 때 BatchNorm의 μ, σ², γ, β는 모두 고정값이다. 그래서 BN(Conv(x))를 풀어 쓰면 새 가중치 W′ = γ/√(σ²+ε) · W 와 새 편향 b′ = γ(b−μ)/√(σ²+ε) + β 를 가진 **Conv 하나**와 똑같다. 빌드 때 한 번 계산해 두면 실행 중 BN 연산은 0번이다.*

### 4-2. 커널(tactic) 선택

![tactic 선택](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig10.jpg)
*그림 11. 퓨전된 레이어 하나(예: Conv+BN+ReLU)에도 구현 방법이 여러 개 있다 — 타일 크기, 데이터 배치(layout), Tensor Core 사용 여부, 정밀도 등. 빌더는 후보들을 **실제 GPU에서 돌려 시간을 재고** 가장 빠른 것(그림에선 구현 B)을 고른다. 막대 길이는 예시. 이 측정 결과가 엔진에 박히기 때문에 **엔진은 빌드한 그 GPU와 그 TensorRT 버전 전용**이다.*

## 5. 정밀도 — FP32, FP16, INT8

![정밀도 비트](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig11.jpg)
*그림 12. FP32는 부호 1 + 지수 8 + 가수 23비트, FP16은 1 + 5 + 10비트, INT8은 −128…127 정수 8비트다. 비트가 줄면 가중치·활성값의 메모리와 대역폭이 줄고, Tensor Core 같은 전용 연산 유닛을 쓸 수 있다. **FP16**은 플래그만 켜면 대부분 그대로 동작하지만 표현 범위가 ±65504라 그보다 큰 값은 inf가 된다. **INT8**은 실수 ↔ 정수 변환용 스케일이 있어야 한다.*

중요한 디테일: `BuilderFlag.FP16`은 "FP16을 써도 된다"는 **허용**이다. 빌더는 레이어마다 FP32와 FP16 중 더 빠른 쪽을 고르고, 레이어 사이에 필요한 형 변환도 알아서 넣는다. 정확도가 중요한 특정 레이어만 FP32로 묶어 두는 설정도 있다.

![INT8 양자화](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig12.jpg)
*그림 13. INT8은 값의 범위 [−T, +T]를 정해 256칸으로 나눈다. 스케일 s = T/127, 양자화 q = round(x/s), 복원 x̂ = q·s. T를 정하는 방법이 핵심이다. **캘리브레이션**: 대표 입력 수백 장을 흘려 층마다 활성값 분포를 모으고, 드물게 튀는 값까지 다 담느라 해상도를 버리지 않도록 적당한 지점(점선)에서 자른다. 다른 방법은 양자화 도구(PTQ/QAT)로 ONNX에 Q/DQ 노드를 넣어 스케일을 명시하는 **explicit quantization**이다 — 최근 TensorRT는 이 방식을 권장하는 흐름이다.*

INT8은 속도 이득이 크지만 정확도 검증이 필수다. 캘리브레이션 데이터가 실제 입력 분포와 다르면(예: 낮 사진으로만 캘리브레이션하고 밤 사진을 추론) 정확도가 크게 떨어질 수 있다. 처음엔 FP16으로 시작하고, 필요할 때 INT8을 검증하며 도입하는 순서를 권한다.

## 6. 엔진 파일

![엔진 파일](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig13.jpg)
*그림 14. 엔진(.engine / .plan)은 ONNX 같은 "그래프"가 아니라 "이 GPU에서 이렇게 실행하라"는 **계획서**다. 안에는 선택된 커널 목록, 퓨전된 레이어 구성, (FP16 등으로 변환된) 가중치, 메모리 배치 계획, 입출력 이름·모양·dtype이 들어 있다. 그리고 **GPU 아키텍처(SM)**와 **TensorRT 버전**에 묶인다. 노트북에서 빌드한 엔진을 다른 세대 GPU 서버로 그대로 복사하면 로드가 실패하거나 최적이 아니다 — 배포처에서 다시 빌드하거나 같은 조합을 맞춘다.*

## 7. 4단계 — 런타임 추론 API

![런타임 흐름](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig14.jpg)
*그림 15. TensorRT 10 Python 기준 7단계. ① `trt.Runtime(logger)` ② `deserialize_cuda_engine(bytes)`로 엔진 복원 ③ `create_execution_context()` — 여기까지는 프로그램 시작 시 **한 번**. 이후 **입력마다**: ④ 가변 입력이면 `set_input_shape` ⑤ `set_tensor_address(name, ptr)`로 입·출력 GPU 버퍼 주소 등록 ⑥ `execute_async_v3(stream)`로 실행 예약 ⑦ `synchronize()`로 완료 대기 후 결과 사용.*

저장소의 `infer_engine.py` 핵심부:

```python
context = engine.create_execution_context()
for i in range(engine.num_io_tensors):
    name  = engine.get_tensor_name(i)
    shape = tuple(context.get_tensor_shape(name))
    dtype = NP_TO_TORCH[np.dtype(trt.nptype(engine.get_tensor_dtype(name)))]
    t = torch.empty(shape, dtype=dtype, device="cuda")   # GPU 버퍼
    context.set_tensor_address(name, int(t.data_ptr()))
    ...
stream = torch.cuda.current_stream()
context.execute_async_v3(int(stream.cuda_stream))
stream.synchronize()
```

GPU 메모리 할당은 PyTorch 텐서를 빌려 쓴다(`data_ptr()`가 GPU 주소). pycuda나 cuda-python으로 직접 `cudaMalloc`해도 원리는 같다. 텐서 **이름**으로 연결하므로 ONNX export 때 붙인 `input`/`output` 이름이 여기서 다시 등장한다. 옛 API(`execute_v2(bindings)`, 바인딩 인덱스)는 TensorRT 10에서 이름 기반 API로 바뀌었다.

![데이터 이동](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig15.jpg)
*그림 16. 데이터가 움직이는 길. CPU 메모리의 전처리된 이미지 → **H→D 복사** → GPU input 버퍼 → 엔진 커널들 → output 버퍼 → **D→H 복사** → CPU에서 top-5. 이 모든 작업을 같은 CUDA 스트림에 순서대로 예약하고, 마지막에 synchronize를 한 번 한다. 작은 모델은 커널 시간보다 복사·전처리 시간이 더 클 수도 있다.*

## 8. 가변 입력과 Optimization Profile

![dynamic shape](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig16.jpg)
*그림 17. 배치 크기가 요청마다 달라지는 서비스라면, export 때 배치 차원을 가변으로 만들고 빌드 때 **Optimization Profile**로 min=1, opt=8, max=32처럼 범위를 알려 준다. 빌더는 **opt 모양**에 맞춰 커널을 가장 잘 고르고, 범위 안의 다른 모양도 동작하게 만든다. 실행 때는 `set_input_shape`로 이번 모양을 지정한다. 범위 밖이면 오류.*

```python
profile = builder.create_optimization_profile()
profile.set_shape("input", (1,3,224,224), (8,3,224,224), (32,3,224,224))
config.add_optimization_profile(profile)
```

## 9. 자주 막히는 곳

![체크리스트](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig17.jpg)
*그림 18. 증상 → 원인. **ONNX 파싱 실패**: opset·지원 연산 확인, onnx-simplifier 등으로 그래프 정리. **모양 오류**: 가변 입력인데 profile이나 set_input_shape를 빠뜨림. **엔진 로드 실패**: 빌드한 GPU·TensorRT 버전과 실행 환경이 다름. **정확도 하락**: FP16 오버플로, INT8 캘리브레이션 데이터 부족. **결과가 엉뚱함**: 전처리 불일치 — 크기, 정규화 mean/std, RGB vs BGR, NCHW vs NHWC. **속도가 이상함**: 워밍업·synchronize 없이 측정했거나, 복사 시간 포함 여부가 다름.*

"결과가 엉뚱함"이 실제로 가장 흔하다. 엔진은 맞는데 전처리가 학습 때와 다른 경우다. PyTorch 원본과 TensorRT 출력을 같은 입력으로 비교해서(예: top-5 일치, 최대 절대오차) 먼저 확인하자.

## 10. 속도는 정직하게 재기

![벤치마크](/assets/img/posts/2026-10-25-tensorrt-inference-basics/fig18.jpg)
*그림 19. GPU 실행은 비동기다 — `execute_async_v3`는 예약만 하고 바로 돌아온다. 그래서 워밍업 20회 → synchronize → 타이머 시작 → 200회 → **다시 synchronize** → 평균을 낸다. 마지막 synchronize를 빼면 "예약하는 시간"만 재서 터무니없이 빠른 숫자가 나온다. trtexec도 같은 원리로 평균·백분위 지연을 출력한다. 비교는 같은 GPU, 같은 배치, 같은 입력 크기, 같은 측정 범위에서.*

저장소에 이 방식 그대로의 `benchmark.py`를 추가해 두었다.

```bash
python export_onnx.py
python build_engine_python.py --onnx resnet18.onnx --engine resnet18.engine --fp16
python infer_engine.py --engine resnet18.engine --image cat.jpg
python benchmark.py --engine resnet18.engine
```

## 정리

1. **왜**: 추론에선 계산보다 메모리 왕복·실행 오버헤드가 문제인 경우가 많고, TensorRT는 그래프 전체를 보고 이를 줄인다.
2. **ONNX**: `eval()` + 더미 입력으로 그래프를 기록. 입출력 이름을 기억해 둔다.
3. **Parser → Builder**: 네트워크로 번역 → 퓨전·BN 접기·정밀도·커널 측정 → 엔진.
4. **엔진**: GPU 아키텍처·TensorRT 버전 전용 계획서. 배포처에서 빌드.
5. **런타임**: Runtime → Engine → Context(한 번) / 모양·주소 지정 → execute_async_v3 → synchronize(입력마다).
6. **정밀도**: FP16부터, INT8은 스케일·검증과 함께.

다음엔 같은 흐름을 트랜스포머 계열 모델에 적용할 때 무엇이 달라지는지(가변 길이, KV 캐시 등)를 다뤄 볼 생각이다.

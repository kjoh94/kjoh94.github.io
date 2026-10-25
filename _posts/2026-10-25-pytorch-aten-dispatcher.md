---
layout: post
title: "ATen이란? PyTorch 연산이 CPU·CUDA·NPU 커널로 갈라지는 방식"
date: 2026-10-25 20:05:00 +0900
categories: [pytorch, aten]
tags: [aten, c10, dispatcher, native_functions, PrivateUse1, NPU, torch-rbln, pytorch]
---

### — `torch.mm` 한 줄이 CPU, GPU, 그리고 NPU 커널로 갈라지는 순간

---

## 0. 질문 하나로 시작해 보자

같은 코드를 두 번 돌린다.

```python
y = torch.mm(a, b)                        # CPU 텐서 → CPU GEMM (MKL/OpenBLAS 등)
y = torch.mm(a.cuda(), b.cuda())          # CUDA 텐서 → cuBLAS GEMM
y = torch.mm(a.to("npu"), b.to("npu"))    # 외부 백엔드(NPU) → ? (9장에서)
```

Python 코드는 한 글자도 안 바뀌었다. 그런데 실제로 계산하는 칩은 완전히 다르다. 누가, 언제, 어떻게 "이번엔 CUDA 커널로 가"라고 정하는 걸까? 이 글은 PyTorch 본가의 CPU/CUDA 이야기를 기본 줄거리로 삼고, NPU 같은 외부 가속기는 마지막 확장 장(9~10장)에서 다룬다.

답의 이름이 **ATen**과 **Dispatcher**다. 이 글은 이 둘을 끝까지 따라가 본다.

---

## 1. 비유: 메뉴판과 주방

식당을 떠올려 보자.

- **메뉴판**: "김치찌개 — 재료: 김치, 돼지고기 — 나오는 것: 찌개 한 그릇". 무엇을 주문할 수 있는지에 대한 **약속**이다.
- **주방**: 그 약속을 실제로 만드는 곳. 본점 주방(CPU), 대형 분점 주방(CUDA), 그리고 프랜차이즈로 새로 들어온 외부 주방(PrivateUse1: NPU 등)이 있다.
- **홀 매니저**: 주문서를 보고 "이 손님은 CUDA 분점 손님이네" 하고 주방을 고르는 사람.

ATen에서는 메뉴판이 **스키마(`aten::mm`)**, 주방이 **커널**, 홀 매니저가 **Dispatcher**다.

> 손님(Python 코드)은 메뉴판만 본다. 어느 주방에서 요리하는지는 몰라도 된다.

---

## 2. 큰 그림 먼저

![torch.mm 한 줄의 여정](/assets/img/posts/2026-10-25-pytorch-aten-dispatcher/01_big_loop.jpg)

1. Python에서 `torch.mm(a, b)`를 부른다.
2. 자동 생성된 C++ 바인딩이 이를 **`aten::mm`** 호출로 바꾼다.
3. **Dispatcher**가 입력 텐서들의 **DispatchKey**(대표적으로 device: CPU / CUDA / PrivateUse1 …)를 본다.
4. 그 키에 등록된 **커널 함수**를 부른다.
5. 결과 텐서를 담을 메모리는 옆길로 **Allocator**에서 받아 온다.

이 그림 하나가 글 전체의 지도다.

---

## 3. ATen이란? (그리고 아닌 것)

**ATen**("A Tensor library")은 PyTorch의 C++ 텐서·연산 라이브러리다. `torch.add`, `torch.mm`, `torch.softmax` 같은 거의 모든 텐서 연산은 결국 `aten::` 이름을 가진 **op**로 내려간다.

아닌 것도 분명히 해 두자.

- `pip install aten` 같은 **별도 Python 라이브러리가 아니다.** `torch` 안의 C++ 코어다.
- Python에서 직접 만질 때는 `torch.ops.aten.mm(a, b)` 처럼 보이지만, 그건 같은 op를 부르는 다른 문일 뿐이다.
- 신경망 레이어(`nn.Linear`)나 autograd 엔진 자체도 아니다. 그들은 ATen op를 **조합해서 쓰는 위층**이다.

> ATen = PyTorch가 "할 수 있는 연산"의 공식 목록 + 그 구현들.

### 3-1. ATen의 경계: 바닥은 어디고 천장은 어디인가

PyTorch를 층으로 잘라 보면 이렇다 (위가 사용자 쪽).

```
 ┌──────────────────────────────────────────────────────────────┐
 │ Python  torch.* / torch.nn / torch.optim / torch.compile     │  ← ATen 아님
 │ torch/csrc: Python 바인딩, autograd 엔진, JIT/FX/Dynamo       │  ← ATen 아님 (ATen op를 "호출·기록"함)
 ├──────────────────────────────────────────────────────────────┤
 │ Dispatcher (c10/core/dispatch, 실제 소스는 aten/src/ATen/core)│  ← 경계: 키를 보고 커널을 고름
 ├──────────────────────────────────────────────────────────────┤
 │ ATen: at::Tensor API, aten:: op 스키마(native_functions.yaml)│
 │       native 커널(CPU/CUDA/MPS…), TensorIterator, 분해 규칙   │  ← 여기가 ATen
 ├──────────────────────────────────────────────────────────────┤
 │ c10: TensorImpl, Storage, Allocator, Device, DispatchKey,    │
 │      DeviceGuard, Stream, ScalarType                          │  ← ATen의 바닥 (ATen 아님)
 ├──────────────────────────────────────────────────────────────┤
 │ 벤더 라이브러리/드라이버: MKL, oneDNN, cuBLAS, cuDNN, NPU RT   │
 └──────────────────────────────────────────────────────────────┘
```

- **바닥(c10)**: "텐서란 무엇인가"(TensorImpl = 메타데이터 + Storage), "메모리는 누가 주나"(Allocator), "어느 디바이스/키인가"(Device, DispatchKey). c10은 Caffe2와 PyTorch가 공유하려고 떼어 낸 핵심 타입 계층이다.
- **ATen**: 그 텐서 위에서 정의되는 **연산들**(스키마 + 커널). C++에서 `at::mm(a, b)`로 바로 쓸 수 있다.
- **천장**: autograd는 ATen 바깥이다. 다만 autograd도 Dispatcher의 한 **키**(`Autograd*`)로 끼어들어, 생성된(VariableType) 래퍼가 그래프를 기록한 뒤 아래 백엔드 키로 다시 디스패치한다. `nn.Linear`는 Python 모듈이고, 결국 `aten::linear` → (분해) `aten::addmm`/`mm`을 부른다.
- **torchgen**: `native_functions.yaml`(+ autograd용 `derivatives.yaml`)을 읽어 C++ API, Python 바인딩, 디스패처 등록 코드를 **빌드 타임에 생성**하는 코드 생성기. 사람이 직접 바인딩을 쓰지 않는다.

### 3-2. "일반 ATen" 입문에서 꼭 알아 둘 것

1. **모든 op가 커널을 갖는 건 아니다 — Composite.** `native_functions.yaml`의 `dispatch:` 항목에 따라
   - `CompositeImplicitAutograd`(dispatch 항목이 없으면 기본): 다른 aten op로 **분해**되어 실행. 백엔드는 따로 구현할 필요 없고 autograd도 자동. (예: 많은 편의 op)
   - `CompositeExplicitAutograd`: 모든 백엔드에 공통인 구현 하나, 단 backward 공식은 명시.
   - `CPU`, `CUDA`, `MPS`, … : 백엔드별 진짜 커널. 실제 yaml의 `mm`은 대략 이렇게 생겼다:
     ```yaml
     - func: mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
       structured: True
       dispatch:
         CPU: mm_out_cpu
         CUDA: mm_out_cuda
         ...
     ```
2. **functional / in-place / out= 3형제.** `add`, `add_`, `add.out`. 스키마의 `Tensor(a!)`는 "이 인자를 변경/별칭함"이라는 표시다.
3. **Structured kernel.** `mm.out` 하나만 구현하면(meta: shape 계산 + impl: 실제 계산) functional/in-place 버전이 자동 생성된다. 그래서 4장 예처럼 functional `mm`에 직접 붙이기보다 **`mm.out`에 커널을 붙이는 게 PyTorch 본가의 관용**이다.
4. **View op vs 계산 op.** `view`, `as_strided`, `transpose`, `slice`는 메모리를 새로 잡지도 계산하지도 않고 **같은 Storage를 다른 stride로 보는** 메타데이터 연산이다. (6장의 "메모리 계열"과 구분하자.)
5. **TensorIterator.** CPU/CUDA의 elementwise·reduction 커널 대부분이 쓰는 공용 엔진: broadcasting, dtype 승격, stride 처리, 벡터화/병렬화를 대신 해 준다. "커널 = GEMM 루프"만은 아니라는 뜻.
6. **op 수.** aten에는 오버로드 포함 **2,000개+** op가 있다. 그래서 컴파일러/외부 백엔드용으로 **Core ATen opset**(약 180여 개, 분해 후 남는 기본 집합)과 더 작은 **Prims**(PrimTorch)가 정의되어 있다.
7. **그래프 IR도 aten 말을 쓴다.** TorchScript(구식), FX/`torch.export`(ATen dialect, `aten.mm.default`), Dynamo+AOTAutograd(분해 후 aten/core aten 그래프). 그래서 "aten"은 eager 런타임이자 **그래프 공통어**다.

---

## 4. 스키마 vs 커널: 약속과 구현을 떼어 놓기

![스키마 하나, 커널 여러 개](/assets/img/posts/2026-10-25-pytorch-aten-dispatcher/02_schema_vs_kernel.jpg)

PyTorch 소스에는 `native_functions.yaml`이라는 파일이 있다. 여기 op의 **시그니처(스키마)**가 적혀 있다.

```yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
```

이게 메뉴판이다. 이름, 입력, 출력만 있다. 계산 방법은 없다.

구현(커널)은 백엔드별로 따로 **등록**한다. 외부(out-of-tree) 백엔드는 보통 이렇게 쓴다.

```cpp
at::Tensor npu_mm(const at::Tensor& self, const at::Tensor& mat2);

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &npu_mm);
}
```

- `TORCH_LIBRARY`: 새 op **스키마를 정의**할 때
- `TORCH_LIBRARY_IMPL(aten, <Key>, m)`: **이미 있는 aten op에 특정 키의 커널만 붙일** 때
- 참고: CPU/CUDA 같은 in-tree 백엔드는 이 매크로 대신 `native_functions.yaml`의 `dispatch:` 항목으로 등록되고, torchgen이 등록 코드를 생성한다. 위 매크로는 주로 **out-of-tree** 백엔드용 문이다.

> 새 칩은 메뉴판을 새로 쓰지 않는다. 기존 메뉴판에 "우리 주방도 이 요리 합니다"라고 도장을 찍을 뿐이다.

---

## 5. 하나를 끝까지 따라가 보자: `aten::mm` (CUDA를 기본으로, NPU도 똑같다)

![aten::mm 따라가기](/assets/img/posts/2026-10-25-pytorch-aten-dispatcher/03_mm_walkthrough.jpg)

`A: [M, K]`, `B: [K, N]` 두 텐서가 CUDA에 있다고 하자. (그림은 NPU로 그렸지만 키 이름만 CUDA → PrivateUse1로 바뀔 뿐 경로는 같다.)

1. **`torch.mm(A, B)`** — Python 호출.
2. **키 계산** — Dispatcher가 A, B의 DispatchKeySet을 합친다. device가 CUDA이므로 실행 백엔드 키는 CUDA (NPU라면 PrivateUse1). (학습이라면 그 위에 Autograd 키가 먼저 거쳐 가며 backward용 기록을 남긴 뒤 다시 내려온다.)
3. **CUDA `mm` 커널 진입** (structured: meta 함수가 출력 shape `[M, N]` 계산) — shape이 맞는지, dtype이 지원되는지 확인.
4. **출력 메모리 확보** — `at::empty({M, N}, A.options())`. 이 `empty`도 또 하나의 aten op이고, 다시 디스패처를 거쳐 **해당 백엔드의 empty 커널 → 그 디바이스 Allocator**(CUDA라면 caching allocator)로 간다.
5. **계산** — CUDA라면 cuBLAS GEMM, NPU라면 그 런타임/컴파일러. 타일링, 온칩 SRAM, MAC 배열… 하드웨어 이야기는 전부 이 안에 숨는다.
6. **반환** — 결과는 입력과 같은 디바이스 텐서. 다음 op도 같은 길을 탄다.

외부 백엔드 입장에서 간단히 쓰면 커널은 이런 모양이다 (교육용 단순화).

```cpp
at::Tensor npu_mm(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0));
  auto out = at::empty({a.size(0), b.size(1)}, a.options());  // 4. 메모리
  my_npu_gemm(a.data_ptr(), b.data_ptr(), out.data_ptr(),      // 5. 계산
              a.size(0), a.size(1), b.size(1));
  return out;
}
```

> `mm` 하나 안에도 "메모리 확보"와 "계산"이 둘 다 들어 있다. 이게 다음 질문의 답이다.

---

## 6. ATen에 alloc도 포함인가? 아니면 matmul 같은 커널만?

![메모리 vs 계산](/assets/img/posts/2026-10-25-pytorch-aten-dispatcher/04_alloc_vs_compute.jpg)

정답은 **층이 나뉘어 있다**.

| 층 | 무엇 | 예 |
|---|---|---|
| **c10** (ATen 아래) | 실제 메모리 할당기, 저장소 | `c10::Allocator`, `Storage`, `DeviceGuard`, Stream |
| **ATen op – 생성/복사 계열** | 텐서를 만들고 옮기는 op | `aten::empty`, `empty_strided`, `copy_`/`_copy_from`, `_to_copy`, `resize_` |
| **ATen op – view 계열** | 메모리 공유, 메타데이터만 | `view`, `as_strided`, `_reshape_alias` |
| **ATen op – 계산 계열** | 숫자를 계산하는 op | `aten::mm`, `addmm`, `add.Tensor`, `softmax`, `convolution` |

- **Allocator 자체**는 엄밀히 ATen이 아니라 그 아래 **c10**에 있다. "NPU 메모리를 어떻게 잡고 돌려주는가"를 정한다.
- 하지만 그걸 **부르는 입구**는 `aten::empty` 같은 ATen op다. 그래서 백엔드는 `empty`, `copy_` 같은 op도 커널로 등록해야 한다.
- `mm` 같은 **계산 커널**은 그 위에서 돈다.

> 새 디바이스 백엔드에는 두 개가 다 필요하다. 메모리가 없으면 텐서를 못 만들고, 계산 커널이 없으면 아무것도 못 한다.

실제로 가장 먼저 만드는 순서도 보통 이렇다: **Allocator → `empty` / `copy_` (CPU↔NPU) → 계산 op 하나씩**.

---

## 7. 왜 NPU 커널을 굳이 "aten 모양"으로 맞추나?

이유는 세 가지다.

1. **사용자 코드가 그대로 돈다.** 모델 코드는 `device="npu"`만 바꾸면 된다. `transformers` 같은 수많은 라이브러리도 결국 aten op를 부르니까 그대로 따라온다.
2. **그래프 도구도 aten 언어를 쓴다.** `torch.export`가 뽑는 그래프의 노드는 `aten.mm.default` 같은 aten op들이다. `torch.compile`의 백엔드, ONNX exporter도 이 aten 그래프에서 출발한다. aten으로 맞추면 eager와 컴파일 양쪽 길에 다 붙는다.
3. **부족한 건 빌려 쓴다.** CompositeImplicitAutograd op는 자동으로 분해되고, 그래도 남은 미등록 op는 백엔드가 **boxed fallback**(`TORCH_LIBRARY_IMPL(_, PrivateUse1, m) { m.fallback(...) }`)을 등록해 CPU로 넘길 수 있다. 자동은 아니고 백엔드가 직접 등록하는 것이다.

---

## 8. "aten이 NPU 커널로 바뀐다"는 말의 정체

정확히 말하면 **아무것도 바뀌지 않는다.**

- `aten::mm`이라는 **스키마는 하나 그대로**다.
- Dispatcher 테이블에는 `(mm, CPU) → cpu_mm`, `(mm, CUDA) → cuda_mm`, `(mm, PrivateUse1) → npu_mm`이 **나란히** 있다.
- 입력 텐서가 NPU에 있으면, 홀 매니저가 **NPU 줄을 고를 뿐**이다.

> 바뀌는 건 op가 아니라, 같은 op에 대해 "어느 주방이 요리하느냐"다.

---

## 9. 확장 장: PrivateUse1 — ATen 전체를 포크하지 않는다

PyTorch는 외부 하드웨어를 위해 **PrivateUse1**이라는 비어 있는 디바이스 키를 준비해 두었다. 백엔드는 이 키에 런타임(Allocator, DeviceGuard, Stream/Event, hooks)을 붙이고, **필요한 aten op만** `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`로 등록한다. `torch.utils.rename_privateuse1_backend("npu")` 같은 호출로 이름도 붙일 수 있다. 등록 안 된 op는 CPU fallback으로 돌린다. 즉 **PyTorch를 포크하거나 ATen을 통째로 포팅하는 게 아니라, 플러그인처럼 끼워 넣는** 방식이다.

공개 참고 자료:

- PyTorch 공식 튜토리얼 — [Facilitating New Backend Integration by PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) (레퍼런스 구현 OpenReg 포함)
- PyTorch [Accelerator 통합 문서](https://docs.pytorch.org/docs/main/accelerator/index.html)
- 오픈소스 NPU 예: [RBLN-SW/torch-rbln](https://github.com/RBLN-SW/torch-rbln) — out-of-tree로 `rbln` 디바이스를 붙이고 aten 커널을 PrivateUse1에 등록하는 구조

---

## 10. 실전 예: torch-rbln (Rebellions NPU, 공개 소스 기준)

이론을 실제 공개 코드로 확인해 보자. [RBLN-SW/torch-rbln](https://github.com/RBLN-SW/torch-rbln)(Apache-2.0, 현재 beta)은 PyTorch를 포크하지 않는 **out-of-tree 확장**이다. 2026-10 기준 저장소에서 확인한 구조:

| 레포 경로 | 층 | 하는 일 |
|---|---|---|
| `c10/rbln/` | **c10** | `RBLNAllocator`, `RBLNPinnedAllocator`, `impl/RBLNGuardImpl`(DeviceGuard), `RBLNHooksInterface`, `RBLNGenerator`, fallback 설정·로깅 |
| `aten/src/ATen/native/native_functions.yaml` | **ATen 스키마 쪽** | 본가 형식 그대로의 yaml로 **약 50개 계산 op만** `dispatch: PrivateUse1: …_rbln` 지정 (`add/sub/mul/div.out`, `mm.out`, `bmm.out`, `addmm.out`, `linear`, `_softmax.out`, `silu.out`, `rsqrt.out`, `mean.out`, 비교 op, SDPA overrideable 등). `mm.out`은 `structured: True` |
| `aten/src/ATen/native/RBLNRegisterOps.cpp` | **ATen 등록** | `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`로 `empty.memory_format`, `empty_strided`, `_copy_from`, `resize_`, `cat.out`, `index_select` 등 + **view op는 본가 `at::native::view`/`as_strided` 구현을 그대로 재사용** |
| 같은 파일 | **fallback** | `TORCH_LIBRARY_IMPL(_, PrivateUse1, m) { m.fallback(fallback_rbln) }` — 미지원 op는 CPU로. 환경변수 `TORCH_RBLN_DISABLE_FALLBACK`로 끄면 에러. `AutogradPrivateUse1`에도 fallback 등록 |
| `torch_rbln/__init__.py` | **Python** | `torch.utils.rename_privateuse1_backend("rbln")`, `torch._register_device_module("rbln", …)` → `device="rbln"`, `torch.rbln.*` |
| `torch_rbln/_internal/register_custom_ops.py` | **컴파일러 연결** | `torch.library`로 커스텀 op(paged attention 등)를 PrivateUse1에 구현. 계산 op는 내부적으로 **`rebel-compiler`**(별도 배포, RBLN Portal 계정 필요)로 컴파일·실행 |

읽을 포인트:

- **3층 구조가 그대로 보인다**: c10(메모리·가드) / ATen(생성·복사·view + 소수의 계산 op) / Python(이름·모듈). 9장에서 말한 "Allocator → `empty`/`copy_` → 계산 op 하나씩" 순서와 일치한다.
- **op 부분집합 + CPU fallback**: 2,000개 aten op 중 수십 개만 직접 구현하고, Composite 분해와 boxed fallback에 기댄다. LLM에 필요한 GEMM·softmax·SDPA·elementwise에 집중한 선택이다.
- **커널 = 손으로 쓴 GEMM이 아닐 수도 있다**: 여기서 "NPU 커널"의 실체는 상당 부분 컴파일러(rebel-compiler)를 부르는 래퍼다. eager 모드라도 뒤에서 작은 그래프를 컴파일·캐시해 실행하는 방식 — CUDA의 "미리 빌드된 커널 호출"과 다른 점이다.
- 공식 문서: [PyTorch RBLN Overview](https://docs.rbln.ai/latest/software/rbln_pytorch/overview.html) (아키텍처·지원 op 목록), [Installation](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html).

> 위 내용은 공개 저장소의 파일 구조와 코드에서 직접 확인한 것만 적었다. 컴파일러 내부는 비공개다.

---

## 11. 이것만 기억하자

- **ATen** = PyTorch의 C++ 텐서 연산 라이브러리. `torch.*`는 결국 `aten::` op로 내려간다.
- **경계**: 아래는 c10(TensorImpl/Storage/Allocator/DispatchKey), 위는 autograd·nn·Python·컴파일러. ATen은 그 사이의 "연산 목록 + 구현".
- **Composite / structured / out= / view**를 알면 "백엔드가 실제로 구현해야 하는 op"가 생각보다 적은 이유가 보인다.
- **스키마(약속)와 커널(구현)은 분리**되어 있다. 하나의 `aten::mm`에 CPU/CUDA(본가) 커널이, 외부 백엔드는 PrivateUse1 커널이 붙는다.
- **Dispatcher**가 텐서의 DispatchKey(device 등)를 보고 커널을 고른다. "aten이 NPU 커널로 바뀌는" 게 아니라 "NPU 줄이 선택되는" 것이다.
- **메모리**: Allocator는 c10 층, 그걸 부르는 `empty`/`copy_`는 aten op. 백엔드는 **메모리 op + 계산 op** 둘 다 필요하다.
- **PrivateUse1**로 필요한 op만 등록하면 된다. 포크 없이, 같은 Python 코드와 export/compile 그래프까지 그대로 따라온다.

---

### 더 읽을거리 (공식)

- [PyTorch internals (ezyang)](http://blog.ezyang.com/2019/05/pytorch-internals/) — Tensor/Storage/dispatch 큰 그림
- [Let's talk about the PyTorch dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [native_functions.yaml README](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md) — dispatch 키, Composite, structured
- [IRs: Core ATen & Prims](https://docs.pytorch.org/docs/stable/torch.compiler_ir.html)
- [Extending dispatcher for a new backend in C++](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)

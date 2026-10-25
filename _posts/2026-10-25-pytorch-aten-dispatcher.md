---
layout: post
title: "ATen이란? PyTorch가 NPU 커널을 고르는 방식"
date: 2026-10-25 20:05:00 +0900
categories: [pytorch, aten]
tags: [aten, dispatcher, PrivateUse1, NPU, pytorch]
---

### — `torch.mm` 한 줄이 CPU, GPU, 그리고 NPU 커널로 갈라지는 순간

---

## 0. 질문 하나로 시작해 보자

같은 코드를 두 번 돌린다.

```python
y = torch.mm(a, b)            # a, b가 CPU에 있으면 → CPU GEMM
y = torch.mm(a.to("npu"), b.to("npu"))   # → NPU GEMM ?
```

Python 코드는 한 글자도 안 바뀌었다. 그런데 실제로 계산하는 칩은 완전히 다르다. 누가, 언제, 어떻게 "이번엔 NPU 커널로 가"라고 정하는 걸까?

답의 이름이 **ATen**과 **Dispatcher**다. 이 글은 이 둘을 끝까지 따라가 본다.

---

## 1. 비유: 메뉴판과 주방

식당을 떠올려 보자.

- **메뉴판**: "김치찌개 — 재료: 김치, 돼지고기 — 나오는 것: 찌개 한 그릇". 무엇을 주문할 수 있는지에 대한 **약속**이다.
- **주방**: 그 약속을 실제로 만드는 곳. 본점 주방(CPU), 대형 분점 주방(CUDA), 새로 연 우리 분점 주방(NPU)이 있다.
- **홀 매니저**: 주문서를 보고 "이 손님은 NPU 분점 손님이네" 하고 주방을 고르는 사람.

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

> 새 칩은 메뉴판을 새로 쓰지 않는다. 기존 메뉴판에 "우리 주방도 이 요리 합니다"라고 도장을 찍을 뿐이다.

---

## 5. 하나를 끝까지 따라가 보자: `aten::mm` on NPU

![aten::mm 따라가기](/assets/img/posts/2026-10-25-pytorch-aten-dispatcher/03_mm_walkthrough.jpg)

`A: [M, K]`, `B: [K, N]` 두 텐서가 NPU에 있다고 하자.

1. **`torch.mm(A, B)`** — Python 호출.
2. **키 계산** — Dispatcher가 A, B의 DispatchKeySet을 합친다. device가 NPU(PrivateUse1)이므로 가장 우선순위 높은 실행 키는 PrivateUse1. (학습이라면 그 위에 Autograd 키가 먼저 거쳐 가며 backward용 기록을 남긴 뒤 다시 내려온다.)
3. **`npu_mm` 진입** — shape이 맞는지, dtype이 지원되는지 확인.
4. **출력 메모리 확보** — `at::empty({M, N}, A.options())`. 이 `empty`도 또 하나의 aten op이고, 다시 디스패처를 거쳐 **NPU의 empty 커널 → NPU Allocator**로 간다.
5. **계산** — NPU 런타임/드라이버를 통해 GEMM 실행. 타일링, 온칩 SRAM, MAC 배열… 하드웨어 이야기는 전부 이 안에 숨는다.
6. **반환** — 결과는 여전히 `device="npu"` 텐서. 다음 op도 같은 길을 탄다.

간단히 쓰면 커널은 이런 모양이다.

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
| **ATen op – 메모리 계열** | 텐서를 만들고 옮기는 op | `aten::empty`, `empty_strided`, `copy_`, `_to_copy`, view류 |
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
3. **부족한 건 빌려 쓴다.** 등록하지 않은 op는 CPU로 넘겨 계산하는 fallback을 둘 수 있다. 처음부터 수천 개를 다 만들 필요가 없다.

---

## 8. "aten이 NPU 커널로 바뀐다"는 말의 정체

정확히 말하면 **아무것도 바뀌지 않는다.**

- `aten::mm`이라는 **스키마는 하나 그대로**다.
- Dispatcher 테이블에는 `(mm, CPU) → cpu_mm`, `(mm, CUDA) → cuda_mm`, `(mm, PrivateUse1) → npu_mm`이 **나란히** 있다.
- 입력 텐서가 NPU에 있으면, 홀 매니저가 **NPU 줄을 고를 뿐**이다.

> 바뀌는 건 op가 아니라, 같은 op에 대해 "어느 주방이 요리하느냐"다.

---

## 9. PrivateUse1: ATen 전체를 포크하지 않는다

PyTorch는 외부 하드웨어를 위해 **PrivateUse1**이라는 비어 있는 디바이스 키를 준비해 두었다. 백엔드는 이 키에 런타임(Allocator, DeviceGuard, Stream/Event, hooks)을 붙이고, **필요한 aten op만** `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`로 등록한다. `torch.utils.rename_privateuse1_backend("npu")` 같은 호출로 이름도 붙일 수 있다. 등록 안 된 op는 CPU fallback으로 돌린다. 즉 **PyTorch를 포크하거나 ATen을 통째로 포팅하는 게 아니라, 플러그인처럼 끼워 넣는** 방식이다.

공개 참고 자료:

- PyTorch 공식 튜토리얼 — [Facilitating New Backend Integration by PrivateUse1](https://docs.pytorch.org/tutorials/advanced/privateuseone.html) (레퍼런스 구현 OpenReg 포함)
- PyTorch [Accelerator 통합 문서](https://docs.pytorch.org/docs/main/accelerator/index.html)
- 오픈소스 NPU 예: [RBLN-SW/torch-rbln](https://github.com/RBLN-SW/torch-rbln) — out-of-tree로 `rbln` 디바이스를 붙이고 aten 커널을 PrivateUse1에 등록하는 구조

---

## 10. 이것만 기억하자

- **ATen** = PyTorch의 C++ 텐서 연산 라이브러리. `torch.*`는 결국 `aten::` op로 내려간다.
- **스키마(약속)와 커널(구현)은 분리**되어 있다. 하나의 `aten::mm`에 CPU/CUDA/NPU 커널이 각각 붙는다.
- **Dispatcher**가 텐서의 DispatchKey(device 등)를 보고 커널을 고른다. "aten이 NPU 커널로 바뀌는" 게 아니라 "NPU 줄이 선택되는" 것이다.
- **메모리**: Allocator는 c10 층, 그걸 부르는 `empty`/`copy_`는 aten op. 백엔드는 **메모리 op + 계산 op** 둘 다 필요하다.
- **PrivateUse1**로 필요한 op만 등록하면 된다. 포크 없이, 같은 Python 코드와 export/compile 그래프까지 그대로 따라온다.

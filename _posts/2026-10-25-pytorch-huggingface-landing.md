---
layout: post
title: "PyTorch로 학습한 모델은 어떻게 허깅페이스에 '착륙'하는가"
date: 2026-10-25 19:40:00 +0900
categories: [ml, huggingface, tutorial]
tags: [pytorch, huggingface, safetensors, gguf, unsloth, qwen, transformers, onnx]
---

> **PyTorch 시리즈 번외편** · [전체 목차](/blog/2026/10/25/pytorch-series/) · 함께 보기: [2. 모델은 부품 상자](/blog/2026/10/25/pytorch-02-module-state-dict/)

### — safetensors, GGUF, Unsloth, 그리고 Qwen 같은 모델이 공개되는 날 벌어지는 일

---

## 0. 질문 하나로 시작해 보자

PyTorch에서 우리는 이런 구조를 봤다.

- **모델 구조** = Python 코드 (`class SimpleMLP(nn.Module)`)
- **가중치** = `.pt` 파일 (`state_dict`)

ONNX는 달랐다. **그래프와 가중치가 `.onnx` 한 파일**에 같이 들어간다.

그럼 이런 궁금증이 생긴다. 허깅페이스(HF)에 있는 수천 개의 모델은 어느 쪽일까? 누군가 Python 클래스를 같이 올리는 걸까, 아니면 ONNX처럼 한 파일로 묶어 올리는 걸까?

답은 둘 다 아니다. 또 둘 다이기도 하다. 이걸 풀어 보자.

---

## 1. 비유: 레고 설명서와 레고 블록

모델을 레고 작품이라고 생각해 보자.

- **설명서**: 어떤 모양의 블록을 어떤 순서로 끼우는지 적혀 있다. 이게 아키텍처다.
- **블록 상자**: 실제 블록이 들어 있다. 이게 가중치, 즉 수십억 개의 숫자다.

PyTorch 방식은 설명서를 **Python 코드**로 들고 다니고, 블록 상자는 `.pt`로 따로 들고 다닌다.
ONNX 방식은 설명서와 블록을 **한 상자에 넣고 밀봉**한다.

HF 방식은 그 중간이다. 설명서를 **아주 짧은 메모**(`config.json`)로 줄이고, 진짜 긴 설명서(Python 클래스)는 **모두가 이미 가진 공용 책**, 곧 `transformers` 라이브러리에 맡긴다.

> "Qwen2 설계도 기억하지? 층은 28개, 히든 크기는 3584야."
> 메모 한 장이면 충분하다. 설계도 본체는 `transformers` 안에 이미 있으니까.

이 감각이 이 글 전체의 열쇠다.

![세 가지 포장법](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig01.jpg)
*그림 1. 같은 '레고 작품'을 포장하는 세 가지 방법. **PyTorch**는 설계도(Python 클래스)와 블록 상자(`.pt` state_dict)를 따로 들고 다닌다. **ONNX**는 그래프와 가중치를 `model.onnx` 한 상자에 밀봉한다. **Hugging Face**는 설계도를 `config.json` 메모 한 장으로 줄이고, 설계도 본체는 모두가 이미 가진 공용 책 `transformers`(예: `modeling_qwen2.py`)에 맡긴 채 블록만 safetensors로 보낸다.*

---

## 2. 학습 → 로컬 저장: 무엇이 디스크에 떨어지는가

### 2.1 순수 PyTorch: `state_dict` vs 전체 체크포인트

```python
torch.save(model.state_dict(), "model.pt")          # (A) 가중치만
torch.save({"model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch}, "ckpt.pt")              # (B) 학습 재개용 체크포인트
torch.save(model, "whole.pt")                        # (C) 객체 통째로 (pickle)
```

- **(A)** `state_dict`는 `{"fc1.weight": 텐서, "fc1.bias": 텐서, ...}` 형태의 이름→텐서 사전이다. 구조는 들어 있지 않다.
- **(B)** 학습을 이어서 하려면 옵티마이저 상태도 필요하다. `ort` 레포의 `05`가 저장하는 `models/torch_classifier.pt`가 이런 딕셔너리이고, 그래서 `06`이 `checkpoint["model_state_dict"]`로 꺼낸다.
- **(C)** 클래스까지 pickle로 저장한다. 하지만 불러오려면 **원래 클래스 코드가 같은 경로에 있어야** 해서 깨지기 쉽다. 그래서 공유용으로는 권장하지 않는다.

여기서 중요한 사실이 하나 있다. `.pt`/`.bin`은 내부적으로 **pickle**을 쓴다. pickle은 로드할 때 임의의 Python 코드를 실행할 수 있다. 남이 올린 파일을 열면 남의 코드가 내 컴퓨터에서 돈다는 뜻이다. 이 사실은 3장에서 다시 등장한다.

![torch.save 세 가지](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig02.jpg)
*그림 2. `torch.save`로 떨어뜨릴 수 있는 세 가지. **(A)** 이름→텐서 사전인 state_dict(구조 없음), **(B)** 옵티마이저 상태·epoch까지 담은 학습 재개용 체크포인트, **(C)** 클래스 객체를 통째로 pickle한 것(원래 코드가 같은 경로에 없으면 안 열림). 아래 빨간 줄이 핵심이다: `.pt`/`.bin`은 pickle이라 로드하는 순간 임의 코드가 실행될 수 있다.*

### 2.2 transformers 방식: `save_pretrained`

HF 생태계에서는 이렇게 저장한다.

```python
model.save_pretrained("my-model/")
tokenizer.save_pretrained("my-model/")
```

폴더를 열어 보면 대략 이런 파일이 있다.

| 파일 | 역할 | 레고 비유 |
|---|---|---|
| `config.json` | `architectures`, `model_type`, 층 수, 히든 크기 등 | 메모 한 장 |
| `generation_config.json` | 기본 생성 설정(eos 토큰, 샘플링 기본값 등) | 사용 팁 |
| `model.safetensors` (또는 샤드) | 가중치 | 블록 상자 |
| `tokenizer.json`, `tokenizer_config.json` | 토크나이저와 채팅 템플릿 | 글자를 블록 번호로 바꾸는 사전 |
| `vocab.json`, `merges.txt` | BPE 토크나이저 원재료(모델에 따라 다름) | 사전의 원본 |

![save_pretrained 폴더](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig03.jpg)
*그림 3. `save_pretrained` 결과 폴더의 트리. 파란 `config.json`이 아키텍처 메모, 금색 `model.safetensors`가 실제 가중치, 초록 두 파일이 토크나이저와 채팅 템플릿이다. 가중치 말고는 전부 작은 텍스트 파일이라는 점을 눈여겨보자.*

`config.json` 안의 `"model_type": "qwen2"`와 `"architectures": ["Qwen2ForCausalLM"]`를 보고 `AutoModelForCausalLM.from_pretrained(...)`가 **어느 Python 클래스를 쓸지** 고른다. 클래스 본체는 `transformers` 패키지 안의 `modeling_qwen2.py`에 있다.

**새 아키텍처라서 아직 `transformers`에 없다면?** 레포에 `modeling_xxx.py` 같은 코드를 직접 넣고, 사용자가 `trust_remote_code=True`로 불러오게 한다. 공용 책에 아직 실리지 않은 설계도를 상자에 같이 넣어 보내는 셈이다. 그래서 이 옵션은 "이 레포의 코드를 실행해도 된다"는 동의다.

![from_pretrained 클래스 선택](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig04.jpg)
*그림 4. `AutoModelForCausalLM.from_pretrained`의 디스패치. `config.json`의 `architectures`/`model_type`을 읽고 `transformers` 안에서 `modeling_qwen2.py`의 `Qwen2ForCausalLM`을 골라 그 위에 가중치를 붓는다. 공용 책에 아직 없는 구조면 레포 안의 `modeling_xxx.py`를 실행해야 하고, `trust_remote_code=True`가 그 동의다.*

**토크나이저가 왜 꼭 필요할까?** 모델은 숫자만 먹는다. "안녕"을 `[12345, 678]`로 바꾸는 규칙이 다르면 가중치가 같아도 엉뚱한 말을 한다. 그리고 `tokenizer_config.json` 안에 **chat_template**(Jinja 템플릿)이 들어 있다. Qwen2.5 Instruct라면 `<|im_start|>user ... <|im_end|>` 같은 대화 포맷이 여기에 정의돼 있다. 이 템플릿이 맞지 않으면 Instruct 모델이 바보가 된다.

![토크나이저와 채팅 템플릿](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig05.jpg)
*그림 5. 위: 문자열 "안녕"이 `tokenizer.json` 규칙으로 토큰 id `[12345, 678]`가 되어야 모델이 먹을 수 있다. 아래: `tokenizer_config.json`의 chat_template이 만드는 Qwen식 대화 포맷. Instruct 모델은 학습 때 이 포맷만 봤기 때문에 템플릿이 어긋나면 품질이 무너진다.*

---

## 3. 허깅페이스 허브의 가중치 포맷: "먼저 보이는 것" 순위

HF 레포의 Files 탭을 열었을 때 보이는 빈도 순으로 정리했다.

### 🥇 1위: `*.safetensors`, 지금의 표준

safetensors는 HF가 만든 포맷이다. 구조가 아주 단순하다.

```
[8바이트: 헤더 길이 N][N바이트 JSON 헤더][텐서 원시 바이트들...]
```

JSON 헤더에는 `"model.layers.0.self_attn.q_proj.weight": {"dtype":"BF16","shape":[3584,3584],"data_offsets":[a,b]}` 같은 정보가 들어 있다. 이름, 타입, 모양, 위치가 전부다.

왜 표준이 됐을까?
1. **안전**: pickle이 아니다. 로드는 바이트를 읽는 것뿐이고 코드는 실행되지 않는다.
2. **빠름**: 헤더만 읽으면 각 텐서가 어디 있는지 안다. 그래서 mmap으로 필요한 부분만 바로 매핑할 수 있다(지연 로딩, zero-copy에 가깝다).
3. **프레임워크 중립**: PyTorch, JAX, NumPy 등에서 읽을 수 있다.

최근 버전의 `save_pretrained`는 기본으로 safetensors를 쓴다.

![safetensors 바이트 구조](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig06.jpg)
*그림 6. safetensors 파일을 바이트 순서대로 펼친 그림. 맨 앞 8바이트(빨강)가 헤더 길이 N, 그다음 N바이트(금색)가 JSON 헤더, 나머지가 텐서 원시 바이트다. 헤더의 `data_offsets: [a, b]`가 텐서 데이터 구간을 직접 가리키므로, 헤더만 읽으면 mmap으로 원하는 텐서만 바로 매핑할 수 있다. 코드 실행은 없다.*

> 주의: safetensors는 **가중치만** 담는다. ONNX처럼 그래프가 들어 있지 않다. 구조는 여전히 `config.json`과 `transformers` 코드가 맡는다.

### 🥈 2위: 샤딩(sharding)과 index json

7B 모델은 BF16 기준으로 약 15GB다. 파일 하나로 두기엔 크다. 그래서 여러 조각으로 나눈다.

실제 `Qwen/Qwen2.5-7B-Instruct` 레포의 파일 목록(HF API로 확인)은 이렇다.

```
.gitattributes
LICENSE
README.md
config.json
generation_config.json
merges.txt
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
vocab.json
```

`model.safetensors.index.json`은 **목차**다.

```json
{
  "metadata": {"total_size": ...},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.layers.27.mlp.down_proj.weight": "model-00004-of-00004.safetensors",
    ...
  }
}
```

"이 텐서는 몇 번 상자에 있다"는 지도다. 로더는 목차를 먼저 읽고 필요한 상자만 연다. 이 레포의 파라미터는 약 76억 개(BF16)이고, 저장 용량은 약 15GB다.

![샤딩과 index.json](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig07.jpg)
*그림 7. `model.safetensors.index.json`의 `weight_map`이 텐서 이름마다 "몇 번 샤드에 있는지"를 적어 둔 목차다. 로더는 목차를 먼저 읽고 필요한 샤드만 연다. Qwen2.5-7B-Instruct는 약 15GB를 4조각으로 나눴다.*

### 🥉 3위: `pytorch_model.bin`, 옛 표준

safetensors 이전의 기본값이다. 사실상 `torch.save(state_dict)` 결과물이고 pickle 기반이다. 샤딩되면 `pytorch_model-00001-of-0000N.bin`과 `pytorch_model.bin.index.json` 형태가 된다. 오래된 레포에서 자주 보인다. HF는 이런 레포를 safetensors로 변환하라고 권장하고, 변환 PR 봇도 돌렸다.

`.pt`/`.pth`라는 확장자는 HF 표준 파일명은 아니다. 연구 코드 레포(예: 논문 원본 체크포인트)에서 가끔 보인다. 이런 레포는 **그 연구자의 Python 코드**가 있어야 열린다.

### 니치: 배포 런타임용 포맷 (별도 폴더나 별도 레포)

- **ONNX** (`*.onnx`, 큰 모델은 `model.onnx_data` 외부 데이터 파일과 함께): 레포 안의 `onnx/` 폴더에 두거나 `xxx-onnx` 같은 별도 레포로 올린다. Optimum이나 Transformers.js 생태계에서 자주 보인다. 경주 님이 공부하는 ORT로 바로 돌릴 수 있는 포맷이 이것이다.
- **OpenVINO** (`openvino_model.xml` + `.bin`): Intel 런타임용이다. Optimum-Intel로 변환한다.
- **TensorRT**: 엔진 파일은 **GPU 종류와 TRT 버전에 묶여** 있어서 허브에 범용으로 올리기 어렵다. 보통은 사용자가 safetensors나 ONNX에서 직접 빌드한다. TensorRT-LLM용 체크포인트 레포가 따로 있는 경우는 있다.
- **MLX** (Apple Silicon용, `mlx-community` 같은 조직): 이것도 safetensors를 쓰지만 MLX 레이아웃이다.

한 줄 요약: **HF에서 "모델"이라고 하면 90%는 `config.json` + 토크나이저 + safetensors(샤드) 조합**이다. 나머지는 특정 런타임을 위한 변환본이다.

![HF 포맷 순위](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig08.jpg)
*그림 8. Files 탭에서 보이는 빈도를 막대 길이로 그린 개념도(정확한 통계 아님). safetensors와 샤드가 대부분이고, `pytorch_model.bin`은 옛 레포, ONNX·OpenVINO·MLX는 런타임별 변환본, TensorRT 엔진은 GPU·버전에 묶여 보통 직접 빌드한다.*

---

## 4. GGUF: "hggl 같은 이상한 포맷"의 정체

### 4.1 이름부터 정리

경주 님이 기억한 "hggl"은 아마 **GGUF**다. 헷갈리는 게 당연하다. 이름이 비슷한 것들이 많다.

- **GGML**: Georgi Gerganov가 만든 C 텐서 라이브러리 이름이자, 예전 모델 파일 포맷 이름이다.
- **GGUF**: 그 뒤를 잇는 현재 포맷이다(2023년 8월 llama.cpp에서 도입). 이전 GGML/GGMF/GGJT 포맷을 대체했다.
- **llama.cpp**: GGUF를 읽어 CPU/GPU에서 LLM을 돌리는 C/C++ 추론 엔진이다. Ollama, LM Studio 같은 앱이 내부에서 이 계열 엔진을 쓴다.

### 4.2 GGUF는 무엇이 다른가

safetensors가 **블록 상자**라면 GGUF는 **ONNX처럼 한 파일에 거의 다 들어 있는 상자**다. 다만 그래프를 담는 건 아니다.

- 헤더에 **메타데이터 key-value**: 아키텍처 이름(`llama`, `qwen2` 등), 층 수, 컨텍스트 길이, RoPE 설정 등
- **토크나이저**(어휘, 병합 규칙, 특수 토큰)와 채팅 템플릿까지 메타데이터에 포함
- 그리고 **양자화된 텐서들**

그래서 `.gguf` 파일 하나만 있으면 llama.cpp가 바로 돌린다. `config.json`도 `tokenizer.json`도 필요 없다. 계산 그래프 자체는 파일에 없다. llama.cpp 코드가 아키텍처 이름을 보고 그래프를 직접 만든다. 이 점에서 HF 방식(메모 + 공용 책)과 철학이 같다. 공용 책이 `transformers` 대신 `llama.cpp`일 뿐이다.

![HF 폴더 vs GGUF](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig09.jpg)
*그림 9. 왼쪽 HF 폴더는 메모·사전·가중치가 여러 파일로 흩어져 있고 `transformers` 코드가 따로 필요하다. 오른쪽 `.gguf`는 한 파일 안에 magic/버전 → 메타데이터 KV(arch, 층 수, 컨텍스트, RoPE) → 토크나이저·채팅 템플릿 → 양자화 텐서 순으로 들어 있다. 계산 그래프는 없고 llama.cpp가 arch 이름을 보고 직접 만든다.*

### 4.3 파일 이름에 붙는 암호: Q4_K_M 같은 것

GGUF 레포에는 보통 같은 모델이 **양자화 수준별로 여러 파일** 들어 있다.

```
Qwen2.5-7B-Instruct-Q4_K_M.gguf
Qwen2.5-7B-Instruct-Q5_K_M.gguf
Qwen2.5-7B-Instruct-Q8_0.gguf
...
```

- `Q4`, `Q5`, `Q8`: 가중치당 대략의 비트 수
- `K`: llama.cpp의 "k-quant" 방식(블록 단위 스케일을 계층적으로 둔다)
- `_S/_M/_L`: 같은 비트 안에서 small/medium/large 변형(중요한 텐서에 비트를 더 준다)
- `F16`/`BF16`: 양자화하지 않은 원본 정밀도

직관적으로는 해상도가 다른 JPEG 여러 장과 비슷하다. 원본 사진(BF16)은 하나인데, 용량과 화질을 맞바꾼 버전을 골라 받는다.

![GGUF 양자화 수준](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig10.jpg)
*그림 10. 같은 7B 모델의 양자화 수준별 파일 크기 감각(대략치). BF16 약 15GB에서 Q4_K_M은 약 4.7GB로 줄어든다. 오른쪽은 이름 읽는 법: 숫자 = 대략 비트 수, K = k-quant, `_S/_M/_L` = 중요 텐서에 비트를 더 주는 변형.*

### 4.4 누가 만드는가

변환은 llama.cpp 레포의 `convert_hf_to_gguf.py`로 safetensors를 GGUF(F16 등)로 바꾼 뒤, `llama-quantize`로 Q4_K_M 등으로 줄이는 흐름이다.

만드는 주체는 두 부류다.
- **공식**: 원 제작사가 직접 올리기도 한다. Qwen은 `Qwen/Qwen2.5-7B-Instruct-GGUF`처럼 공식 GGUF 레포를 냈다.
- **커뮤니티와 양자화 전문 조직**: `bartowski`, `unsloth`, `lmstudio-community` 같은 계정이 새 모델이 나오면 빠르게 GGUF를 만들어 올린다. 예전에는 `TheBloke`가 이 역할의 상징이었다.

---

## 5. Unsloth: "다시 올리는" 사람들의 정체

### 5.1 Unsloth는 무엇인가

Unsloth는 LLM **파인튜닝을 빠르고 메모리 적게** 하도록 만든 오픈소스 라이브러리다(커스텀 커널, LoRA/QLoRA 중심). 그리고 HF에 `unsloth`라는 **조직 계정**이 있다. 경주 님이 본 "언슬로쓰가 다시 올린 것"은 이 계정의 레포들이다.

### 5.2 unsloth 계정에 올라오는 것들

새 모델이 나오면 이 계정에는 보통 이런 레포들이 생긴다(모델에 따라 다르다).

1. **원본 미러/재업로드**: 원본과 같은 safetensors. 원본이 gated(동의 필요)이거나, 채팅 템플릿이나 토크나이저 버그를 고친 버전인 경우가 있다.
2. **`...-bnb-4bit`**: bitsandbytes 4bit로 **미리 양자화한** 가중치. QLoRA 파인튜닝을 시작할 때 다운로드 용량과 VRAM을 줄여 준다.
3. **`...-unsloth-bnb-4bit`**: Unsloth의 "dynamic 4bit". 민감한 층은 양자화하지 않고 남겨 정확도를 지키는 변형이다.
4. **`...-GGUF`**: llama.cpp/Ollama용 GGUF 파일 모음.

### 5.3 사용자가 Unsloth로 파인튜닝해서 올리면

흐름은 이렇다.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, ...)   # LoRA 부착
# ... 학습 ...
```

학습 뒤 올리는 방식은 세 갈래다.

| 방식 | 대표 메서드 | 레포에 남는 파일 | 비유 |
|---|---|---|---|
| **LoRA 어댑터만** | `model.save_pretrained(...)` / `model.push_to_hub(...)` (PEFT) | `adapter_config.json`, `adapter_model.safetensors` | 원본 레고는 두고 "덧붙이는 작은 블록 세트"만 보냄 |
| **병합된 전체 가중치** | `save_pretrained_merged` / `push_to_hub_merged` (16bit 또는 4bit 선택) | `config.json` + safetensors 샤드 + 토크나이저 | 덧블록을 붙여 새 작품 전체를 다시 포장 |
| **GGUF로 바로** | `save_pretrained_gguf` / `push_to_hub_gguf` (quantization_method 지정, 예: q4_k_m) | `*.gguf` | 바로 llama.cpp/Ollama용 완제품 |

**LoRA 직관**: 거대한 가중치 행렬 W(예: 3584×3584)는 그대로 얼린다. 옆에 아주 얇은 두 행렬 A(3584×16)와 B(16×3584)를 붙여 `W + BA`로 쓴다. 학습되는 건 얇은 A, B뿐이다. 그래서 어댑터 파일은 수십~수백 MB로 작다. 대신 쓸 때는 **베이스 모델이 꼭 필요**하다. `adapter_config.json` 안의 `base_model_name_or_path`가 "어떤 원본 위에 붙이는지"를 가리킨다.

![LoRA 행렬](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig11.jpg)
*그림 11. LoRA의 모양. 3584×3584 W(파랑)는 얼려 두고, 3584×16 B와 16×3584 A(금색)만 학습한다. 그래서 어댑터 파일은 작지만, 쓸 때는 `base_model_name_or_path`가 가리키는 베이스 모델이 꼭 있어야 한다.*

**병합(merge)**: `W' = W + BA`를 실제로 계산해서 새 W'를 저장한다. 이제 혼자서도 돌아가는 일반 모델 레포가 된다. 그래서 GGUF, ONNX 변환도 이 병합본에서 출발한다.

![Unsloth 세 갈래](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig12.jpg)
*그림 12. Unsloth 학습 뒤의 세 출구. `push_to_hub`는 어댑터 두 파일만, `push_to_hub_merged`는 W'=W+BA를 계산한 독립 모델 레포를, `push_to_hub_gguf`는 llama.cpp/Ollama용 완제품을 올린다.*

---

## 6. 대형 오픈 모델이 공개되는 날: Qwen 사례로 따라가 보기

"Qwen3.5 같은 게 나오면 어떻게 올라오나?" 구체적인 버전마다 세부는 다르니, 실제 파일을 확인한 **Qwen2.5 패턴**으로 설명한다. 새 세대도 대체로 이 패턴을 따른다.

### 1막: 제작사 내부
거대한 클러스터에서 PyTorch(분산 학습 프레임워크)로 사전학습(**Base**)을 한다. 이어 SFT와 RLHF 같은 후처리로 **Instruct** 모델을 만든다. 내부 체크포인트(분산 샤드, 옵티마이저 상태 포함)는 공개용이 아니다.

### 2막: HF 포맷으로 정리
- 아키텍처를 `transformers`에 맞춘다. 새 구조면 `transformers`에 PR을 넣어 `modeling_qwen2.py` 같은 클래스를 추가한다. 머지 전에는 `trust_remote_code`로 공개한다.
- 가중치를 BF16 safetensors 샤드로 저장하고, `config.json`, `generation_config.json`, 토크나이저, 채팅 템플릿을 정리한다.
- README(모델 카드)에 라이선스, 벤치마크, 사용 예제를 쓴다. 메타데이터에 `base_model: Qwen/Qwen2.5-7B`를 넣으면 HF가 "이 모델은 저 모델의 파인튜닝"이라는 **가계도**를 그린다.

### 3막: 공식 레포 공개 (보통 Collection으로 묶는다)
Qwen2.5 시절의 공식 레포 패턴은 이랬다.
- `Qwen/Qwen2.5-7B` (Base)
- `Qwen/Qwen2.5-7B-Instruct`
- 공식 양자화본: `Qwen/Qwen2.5-7B-Instruct-GGUF`, `Qwen/Qwen2.5-7B-Instruct-AWQ`, `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` / `-GPTQ-Int8`
- 그리고 사이즈별(0.5B~72B) 반복

업로드 자체는 git(대용량은 LFS/Xet 스토리지) 또는 `huggingface_hub`의 `upload_folder`나 `push_to_hub`로 한다. 공개 직전까지 private으로 두었다가 발표 시각에 public으로 전환하는 경우가 많다.

### 4막: 몇 시간~며칠 안에 커뮤니티 파도
- **GGUF**: bartowski, unsloth, lmstudio-community 등
- **AWQ / GPTQ**: vLLM, TGI 같은 GPU 서빙용 4bit 양자화. GGUF가 llama.cpp 계열이라면 이쪽은 GPU 서버 계열이다.
- **MLX**: `mlx-community`
- **ONNX**: onnx-community 같은 계정(Transformers.js와 ORT용)
- **파인튜닝 파생**: 수많은 개인과 조직의 LoRA, 병합 모델, 그리고 그걸 또 GGUF로 만든 것들

모델 페이지 오른쪽의 "Quantizations", "Finetunes", "Adapters", "Merges" 숫자가 바로 이 가계도다.

![공개일 4막](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig13.jpg)
*그림 13. 대형 모델 공개를 4막으로 그린 타임라인. 내부 학습(비공개) → HF 포맷 정리(transformers PR, 샤드, 모델카드) → 발표 시각 공식 레포 public 전환(Base·Instruct·공식 양자화본) → 몇 시간~며칠 안에 커뮤니티의 GGUF·MLX·ONNX·LoRA 파도.*

### 양자화 포맷 한눈에

| 포맷 | 주 런타임 | 파일 모양 | 주로 누가 |
|---|---|---|---|
| BF16 safetensors | transformers, vLLM, SGLang 등 | 샤드 + index json | 원 제작사 |
| GGUF | llama.cpp, Ollama, LM Studio | 단일(또는 분할) `.gguf` | 제작사 일부 + 커뮤니티 |
| AWQ / GPTQ | vLLM, TGI, transformers | safetensors + `quantization_config`가 들어간 config | 제작사 일부 + 커뮤니티 |
| bnb-4bit | transformers + bitsandbytes (QLoRA) | safetensors | unsloth 등 |
| ONNX | ONNX Runtime, Transformers.js | `.onnx` (+ `_data`) | Optimum, 커뮤니티 |
| MLX | Apple MLX | safetensors (MLX 레이아웃) | mlx-community |

![포맷 지도](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig14.jpg)
*그림 14. 표를 한 장으로: 포맷마다 어느 런타임이 먹고, 주로 누가 올리는지.*

---

## 7. 전체 그림 한 장

![전체 그림](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig15.jpg)
*그림 15. PyTorch 학습 → 내부 체크포인트 → `save_pretrained`(config + tokenizer + safetensors + index) → `push_to_hub`로 허브의 공식 레포 → 오른쪽으로 GGUF·AWQ/GPTQ·ONNX/OV/MLX 변환본, 아래로 Unsloth/PEFT 파인튜닝과 그 세 출구(어댑터·병합·GGUF).*

---

## 8. 경주 님의 Plan B와 연결하면

- `ort` 레포의 `05` → `.pt`(state_dict 체크포인트)는 **2.1절의 (B)**다.
- `06` → `.onnx`는 **3장의 "니치: 배포 런타임용 포맷"**이다.
- HF 세계로 옮기면 `05`의 결과를 `save_pretrained`로 safetensors + config로 바꾸고, `06`의 ONNX는 `onnx/` 폴더나 별도 레포로 붙이는 셈이다.
- 앞으로 할 **llama.cpp 주차**에서 GGUF를 직접 만들어 보면 4장이 손에 잡힐 것이다.

결국 질문은 처음과 같다. **설계도는 어디에 두고, 블록은 어디에 담을까?**
- PyTorch는 설계도가 코드, 블록은 `.pt`다.
- HF는 설계도가 짧은 메모 + 공용 책(`transformers`), 블록은 safetensors다.
- ONNX는 설계도와 블록을 한 상자에 담는다.
- GGUF는 메모와 사전과 압축 블록을 한 상자에 담고, 설계도 본체는 llama.cpp가 갖는다.

![네 줄 요약](/assets/img/posts/2026-10-25-pytorch-huggingface-landing/fig16.jpg)
*그림 16. 이 글 전체를 네 줄로: 설계도는 어디에, 블록은 어디에.*

이 네 줄이 머리에 남으면, 다음에 새 모델이 공개되는 날 Files 탭을 열었을 때 파일 하나하나가 왜 거기 있는지 읽힐 것이다.

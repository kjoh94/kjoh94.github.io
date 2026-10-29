---
layout: post
title: "TensorRT-LLM으로 Hugging Face 모델 돌리기 — 원리부터 실행까지"
date: 2026-10-29 19:00:00 +0900
categories: [gpu, llm, tensorrt-llm]
tags: [tensorrt-llm, huggingface, llm, kv-cache, autodeploy, trtllm-build, inference]
---

[지난 글](/blog/2026/10/25/tensorrt-inference-basics/)에서는 ResNet18을 ONNX → TensorRT 엔진으로 바꿔 돌렸다. 이번엔 같은 질문을 LLM에 던진다. **Hugging Face에 있는 언어 모델을 NVIDIA TensorRT-LLM으로 돌리면, 안에서 무슨 일이 일어나는가?**

- 코드: <https://github.com/kjoh94/tensorrt-llm-hf-example> (TinyLlama-1.1B를 LLM API / AutoDeploy / 고전 엔진 경로로 실행)
- 이 글은 NVIDIA·Hugging Face 공개 문서 수준의 일반 내용이다. "예시"라고 적힌 숫자는 이해를 돕기 위한 값이다. TensorRT-LLM은 버전마다 API가 자주 바뀌므로(특히 AutoDeploy는 beta) 설치한 버전의 문서를 같이 보자.

## 0. 전체 큰 루프 — 한 장으로 먼저

![TensorRT-LLM 큰 루프](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig01.jpg)
*그림 1. 위쪽 점선 상자가 **① 준비 단계**다. HF Hub의 모델 id에서 config와 가중치를 받고, 그것을 TensorRT-LLM이 실행할 수 있는 형태로 바꾼다(AutoDeploy의 그래프 변환, 또는 체크포인트 변환 + `trtllm-build`). 이건 모델당 **한 번**이다. 아래쪽이 **② 요청마다 반복되는 루프**다. 프롬프트 전체를 한 번에 처리하는 **Prefill 1회**가 KV cache를 채우고, 이어서 토큰을 하나씩 만드는 **Decode가 N회** 돈다. 두 단계 모두 Paged KV cache를 읽고 쓰며, 스케줄러는 여러 요청을 섞어(in-flight batching) GPU를 쉬지 않게 한다. 예시: 프롬프트 500토큰이면 Prefill 1번, 답변 200토큰이면 Decode 200번.*

TensorRT 글의 핵심 한 줄이 "무거운 고민은 빌드 때, 실행은 빠르게 반복"이었다면, LLM에서는 한 줄이 더 붙는다.

> **LLM 추론은 "같은 모델을 토큰마다 다시 돌리는 루프"이고, 그 루프를 빠르게 만드는 핵심은 커널보다 KV cache와 스케줄링이다.**

## 1. TensorRT와 무엇이 다른가

![TensorRT vs TensorRT-LLM](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig02.jpg)
*그림 2. 왼쪽 TensorRT는 ONNX 그래프를 받아 엔진을 만들고, 입력 한 장 → 출력 하나로 끝난다. 모양이 고정이고 요청 사이에 상태가 없다. 오른쪽 TensorRT-LLM은 Transformers 모델을 받아 LLM 전용 커널(attention, GEMM, 양자화)과 **런타임**을 함께 제공한다. 출력이 토큰을 반복 생성하는 과정이라, 요청마다 길이가 다르고 KV cache라는 상태가 생기며, 여러 요청을 묶는 배칭이 성능을 좌우한다.*

ResNet은 "한 번 통과"면 끝나지만 LLM은 토큰 하나마다 전체 레이어를 다시 통과한다. 그래서 TensorRT-LLM은 단순한 그래프 컴파일러가 아니라 **컴파일 + 런타임(스케줄러·메모리 관리자)** 묶음이다.

## 2. HF 모델이 실행 가능한 형태가 되는 두 갈래

![두 갈래](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig03.jpg)
*그림 3. **A. LLM API / AutoDeploy**: Hub id를 주면 Transformers로 config·가중치를 읽고, AutoDeploy는 `torch.export`로 그래프를 뽑은 뒤 자동 변환(KV cache 삽입, attention 커널 교체, GEMM 퓨전, 멀티 GPU 샤딩)을 거쳐 TensorRT-LLM의 PyTorch 런타임에서 바로 실행한다. 별도의 엔진 빌드 단계가 없다. **B. 고전 엔진 경로**: `convert_checkpoint.py`로 HF 가중치를 TensorRT-LLM 체크포인트(config.json + 랭크별 safetensors)로 바꾸고, `trtllm-build`가 GPU·최대 모양을 고정한 엔진을 만든 뒤 runner로 실행한다.*

### A. 지금 권장되는 쉬운 길 — LLM API

최근 TensorRT-LLM의 기본 `LLM` API는 PyTorch 백엔드를 쓰고, HF 모델 id를 바로 받는다.

```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
out = llm.generate(["What is a KV cache?"], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
```

- TensorRT-LLM이 그 아키텍처(Llama 등)를 직접 구현해 두었으면 그 최적화 구현으로 가중치를 옮겨 싣는다.
- **AutoDeploy**(beta)는 그런 구현이 없는 HF 모델도 *원본 코드를 고치지 않고* 그래프를 뽑아 변환한다. 저장소의 `scripts/02_autodeploy.py`가 그 예다(`from tensorrt_llm._torch.auto_deploy import LLM`).
- `trtllm-serve <hf_id>`로 OpenAI 호환 서버도 바로 띄울 수 있다(`scripts/04_serve.sh`).

AutoDeploy의 변환은 TensorRT 빌더가 한 일과 닮았다. 다만 대상이 ONNX가 아니라 `torch.export`가 만든 ATen 그래프이고, "레이어 퓨전" 외에 **"KV cache를 그래프에 끼워 넣기"**라는 LLM 전용 작업이 들어간다.

### B. 고전 경로 — 체크포인트 → trtllm-build → 엔진

```bash
python examples/models/core/llama/convert_checkpoint.py \
  --model_dir hf_models/tinyllama --output_dir ckpt/tinyllama_fp16 --dtype float16
trtllm-build --checkpoint_dir ckpt/tinyllama_fp16 --output_dir engines/tinyllama_fp16 \
  --gemm_plugin auto --max_batch_size 8 --max_input_len 512 --max_seq_len 1024
python examples/run.py --engine_dir engines/tinyllama_fp16 \
  --tokenizer_dir hf_models/tinyllama --max_output_len 64 --input_text "What is a KV cache?"
```

체크포인트는 "TensorRT-LLM 레이어 이름 규칙에 맞게 다시 쓴 가중치 + 모델 설정"이다. 양자화(예: INT8/FP8/INT4 weight-only)를 고르면 이 변환 단계에서 가중치가 미리 바뀐다.

## 3. 엔진은 왜 모양과 GPU에 묶이나

![빌드는 무겁게](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig07.jpg)
*그림 4. `trtllm-build`는 GPU 아키텍처, 최대 배치·입력·전체 길이, 빔 수, 정밀도, 플러그인을 입력으로 받는다. 이 한도 안에서 커널을 고르고 메모리(특히 활성값·KV 영역)를 계획하기 때문에 빌드는 무겁고, 실행은 정해진 계획을 반복하니 가볍다. 한도를 넘는 길이를 쓰거나 다른 세대의 GPU로 옮기면 다시 빌드해야 한다 — ResNet 엔진과 같은 원리다.*

## 4. Prefill과 Decode — 한 요청 안의 두 얼굴

![Prefill vs Decode](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig04.jpg)
*그림 5. **Prefill**은 프롬프트 6토큰을 한 번에 L개 레이어에 통과시킨다. 행렬×행렬 연산이라 GPU 계산 유닛이 바쁘다(계산 위주). 이때 각 레이어의 K, V가 KV cache에 저장된다. **Decode**는 직전 토큰 하나만 넣어 다음 토큰 하나를 만든다. 새로 계산하는 건 그 토큰의 Q·K·V뿐이고, 과거 토큰은 KV cache에서 읽는다. 행렬×벡터라 계산보다 **가중치와 KV를 메모리에서 읽는 속도**가 병목이 된다. 오른쪽: 매 스텝 레이어 1…L을 모두 지나며 attention(KV 읽기) → MLP를 한다.*

그래서 지표도 둘로 나뉜다. 첫 토큰까지 시간(TTFT)은 주로 Prefill, 토큰당 시간(TPOT)은 Decode가 정한다. Decode가 메모리 위주이기 때문에, 요청 여러 개를 한 스텝에 묶어 가중치를 한 번 읽고 여러 토큰을 만드는 **배칭**이 처리량을 크게 올린다. TensorRT-LLM의 in-flight batching은 끝난 요청을 빼고 새 요청을 스텝 단위로 끼워 넣어 배치를 계속 꽉 채운다.

## 5. Paged KV cache

![Paged KV cache](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig05.jpg)
*그림 6. KV cache를 요청마다 "최대 길이만큼 연속 공간"으로 잡으면 대부분이 빈 채로 낭비된다. TensorRT-LLM은 KV를 고정 크기 **블록**(예시: 블록당 32토큰)으로 쪼갠 풀에서 필요할 때마다 블록을 하나씩 준다. 요청 A는 블록 0·7·13, B는 3·10, C는 20·21·16·5를 쓰는 식이고, 각 요청의 **블록 테이블**이 논리 순서 → 물리 위치를 이어 준다. 운영체제의 가상 메모리 페이지와 같은 생각이다. 덕분에 조각 낭비가 줄고 같은 메모리로 더 많은 요청을 동시에 돌린다.*

## 6. 프리픽스 재사용

![프리픽스 재사용](/assets/img/posts/2026-10-29-tensorrt-llm-huggingface/fig06.jpg)
*그림 7. 많은 요청이 같은 시스템 프롬프트로 시작한다. 블록 단위로 KV를 관리하면 "앞부분이 같은 블록"을 트리(radix tree)로 찾아 그대로 재사용할 수 있다. 질문 A와 B는 system 블록을 공유하고, A의 후속 질문은 A 블록까지 공유한다. 재사용된 부분은 Prefill을 건너뛰므로 TTFT가 줄어든다. TensorRT-LLM 문서에서는 KV cache reuse로 설명하며, 블록이 꽉 찬 단위로 재사용된다.*

## 7. 정리 — 루프를 다시 보며

1. **준비(한 번)**: HF id → (A) LLM API/AutoDeploy가 그래프를 변환해 PyTorch 런타임에 올리거나, (B) 체크포인트 → `trtllm-build` 엔진.
2. **요청마다**: Prefill 1회로 KV 채우기 → Decode N회로 토큰 생성.
3. **빠른 이유**: LLM 전용 커널(퓨전된 attention·GEMM, FP16/FP8/양자화) + Paged KV cache + 프리픽스 재사용 + in-flight batching.

실행 환경은 NVIDIA GPU와 TensorRT-LLM 설치가 필요하다. 가장 쉬운 건 NGC의 TensorRT-LLM 컨테이너(클라우드 GPU 포함)에서 저장소 스크립트를 그대로 돌리는 것이다.

## 참고 문서
- [Transformers × TensorRT-LLM](https://huggingface.co/docs/transformers/main/en/community_integrations/tensorrt-llm)
- [AutoDeploy](https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/auto-deploy.html)
- [Checkpoint](https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html)
- [KV cache](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
- [TGI TensorRT-LLM backend](https://huggingface.co/docs/text-generation-inference/en/backends/trtllm)

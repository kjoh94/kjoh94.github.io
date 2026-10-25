---
layout: post
title: "AI는 왜 읽기는 빠르고 쓰기는 느릴까? — On-Device LLM 추론 기초 (Prefill · Decode · KV Cache)"
date: 2026-10-25 12:40:00 +0900
categories: [ai, on-device, llm]
tags: [llm, prefill, decode, kv-cache, gqa, rope, rmsnorm, inference, on-device]
---

이 글은 특정 회사·제품과 무관한 일반 LLM(트랜스포머 디코더) 추론 원리 정리입니다. 숫자는 이해를 돕는 **예시**입니다.


![읽기는 한 번에](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig01_read_fast_3b1b.jpg)
*그림 1. Prefill: 질문의 토큰 8개가 동시에 모델로 들어갑니다. 아무리 길어도 "읽기"는 큰 행렬 곱 한 번으로 병렬 처리됩니다.*

![쓰기는 한 글자씩](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig02_write_slow_3b1b.jpg)
*그림 2. Decode: 토큰을 하나 만들 때마다(t=1, 2, 3…) 모델 전체를 한 번씩 통과하고, 방금 나온 토큰이 다시 입력이 됩니다(자기회귀).*

![많아도 빠름, 적어도 느림](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig03_contrast_3b1b.jpg)
*그림 3. 읽을 양(1000토큰)은 많아도 빠르고, 쓸 양(100토큰)은 적어도 느립니다. 숫자는 예시예요. 이 글은 이 역설의 이유를 따라갑니다.*

## 한 줄 요약

LLM은 질문을 **한꺼번에 읽고(Prefill)**, 답을 **한 토큰씩 쓴다(Decode)**. 쓰는 단계는 계산보다 **메모리에서 가중치를 꺼내 오는 시간**이 병목이고, 이미 계산한 정보를 다시 계산하지 않으려고 **KV 캐시**를 쓴다. 캐시가 커지는 걸 줄이려고 **GQA** 같은 기법을 쓴다.

## 1. 토큰과 임베딩

- 문장은 **토큰**(단어 조각)으로 잘린다. 예: "안녕하세요" → 2~3개 토큰(토크나이저마다 다름).
- 각 토큰 번호는 **임베딩 표**에서 숫자 벡터(예: 길이 2048)로 바뀐다. 이 벡터를 "hidden state"라 부르며, 레이어를 지날 때마다 뜻이 풍부해진다.


![토큰과 임베딩](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig04_token_embed_3b1b.jpg)
*그림 4. 각 토큰은 id 번호로 바뀌고, 임베딩 표에서 그 번호의 한 줄(노란 테두리)을 꺼내 벡터 x₁, x₂, x₃가 됩니다. 실제로는 한 벡터에 숫자가 수천 개예요.*


## 2. 레이어 한 층의 순서 (Pre-Norm 디코더 블록)

요즘 공개 LLM(Llama 계열 등)의 일반적인 한 층:

```
x ─┬─> RMSNorm ─> Attention(RoPE 포함) ─┐
   └──────────────── + <────────────────┘   (Residual 1)
h ─┬─> RMSNorm ─> FFN(MLP, 예: SwiGLU) ──┐
   └──────────────── + <────────────────┘   (Residual 2)
```


![모델 안의 층](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig05_brain_layers_3b1b.jpg)
*그림 5. 디코더 블록 한 층: RMSNorm → Attention → +Residual → RMSNorm → FFN → +Residual. 위 코드 블록이 바로 이 층이고, 이걸 N번 반복합니다.*

![문장의 길](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig10_sentence_path_3b1b.jpg)
*그림 6. 문장이 지나가는 길: 토큰 → 임베딩 → 레이어 1…N → Norm·출력층 → 다음 토큰 확률(막대그래프).*

이 블록이 수십 층(예시: 24~32층) 반복되고, 마지막에 RMSNorm → 출력층(LM head) → 다음 토큰 확률.

## 3. RMSNorm — 크기 맞추기

- 벡터 값들이 층을 지나며 너무 커지거나 작아지면 학습·추론이 불안정해진다.
- RMSNorm은 벡터를 **제곱평균제곱근(RMS)**으로 나눠 크기를 일정하게 맞추고, 학습된 가중치(γ)를 곱한다.
- 식: `y = x / sqrt(mean(x²) + ε) × γ`
- LayerNorm과 달리 평균을 빼지 않아 계산이 조금 더 가볍다.


![RMSNorm](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig06_rmsnorm_3b1b.jpg)
*그림 7. 흐린 화살표(원래 벡터)는 길이가 제각각이지만, RMSNorm을 지나면 진한 화살표처럼 같은 원 위로 길이가 맞춰집니다. 방향(뜻)은 그대로예요.*


## 4. Attention — "누구를 참고할까?"

각 토큰은 세 벡터를 만든다.

- **Q(Query)**: 내가 찾는 정보
- **K(Key)**: 나는 이런 정보야(이름표)
- **V(Value)**: 실제 내용(상자)

계산: `Attention = softmax(Q·Kᵀ / √d) · V`

1. 현재 토큰의 Q를 앞 토큰들의 K와 내적 → 관련도 점수
2. √d로 나눠 값 크기 조절, softmax로 합이 1인 비율로
3. 그 비율로 V들을 섞어 새 정보 생성

- **Causal mask**: 미래 토큰은 못 본다(답을 미리 엿보지 않게).
- **Multi-Head**: 이 과정을 여러 "헤드"(예시: 16개)가 다른 관점으로 동시에 한다.


![Attention Q K V](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig07_attention_qkv_3b1b.jpg)
*그림 8. 행(Q)은 지금 토큰, 열(K)은 참고할 토큰입니다. 밝을수록 점수가 높고, 그 점수로 V를 가중합합니다. 오른쪽 위(미래 토큰)는 비어 있어요 = causal mask.*


## 5. RoPE — 위치 정보 넣기

- 어텐션 자체는 순서를 모른다. "개가 사람을 물었다"와 "사람이 개를 물었다"를 구분하려면 위치가 필요하다.
- **RoPE(Rotary Position Embedding)**는 Q와 K 벡터를 위치에 비례한 각도만큼 **회전**시킨다.
- 두 토큰의 Q·K 내적이 **상대 거리**에 따라 달라지게 되어, 가까운/먼 관계를 자연스럽게 반영.
- V에는 적용하지 않는다. K는 회전된 상태로 캐시에 저장된다.


![RoPE](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig08_rope_3b1b.jpg)
*그림 9. 위치 m이 커질수록 벡터를 m·θ만큼 더 돌립니다. 두 벡터의 내적은 각도 차이, 즉 상대 거리 (m − n)에만 달려 있어요.*


## 6. Residual — 원본 지키기

- 각 서브블록 출력을 입력에 **더한다**(x + f(x)).
- 깊은 층에서도 원래 정보가 사라지지 않고, 학습 시 기울기가 잘 흐른다.


![Residual과 FFN](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig09_residual_ffn_3b1b.jpg)
*그림 10. 노란 지름길(Residual)이 원본 x를 그대로 실어 나르고, FFN(d → 4d → d)을 거친 변화량이 + 에서 더해집니다. 그래서 원본이 지워지지 않아요.*


## 7. FFN(MLP)

- 토큰마다 독립적으로 벡터를 넓혔다(예: 4배) 다시 줄이며 지식을 꺼내는 단계. 파라미터의 상당 부분이 여기 있다.


![Prefill vs Decode](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig11_prefill_vs_decode_3b1b.jpg)
*그림 11. Prefill은 행렬 × 행렬이라 계산이 병목(compute-bound), Decode는 벡터 × 행렬이라 가중치를 읽는 속도가 병목(memory-bound)입니다.*

## 8. Prefill — 읽기 단계

- 프롬프트 토큰 N개가 **이미 다 주어져 있음** → 모든 토큰을 행렬로 묶어 **한꺼번에** 계산.
- 큰 행렬곱 위주 → 연산 장치를 꽉 채워 씀(**compute-bound** 경향).
- 이 과정에서 각 층·각 토큰의 K, V를 계산해 **KV 캐시에 저장**하고, 마지막 토큰으로 첫 답 토큰을 뽑는다.
- 사용자가 느끼는 지표: **TTFT(Time To First Token, 첫 글자까지 시간)**.

## 9. Decode — 쓰기 단계

- 답 토큰은 앞 토큰이 정해져야 다음을 만들 수 있다(**autoregressive**) → 한 번에 1개.
- 토큰 하나를 만들 때마다 **모든 층의 가중치 전체**를 메모리에서 읽어 와야 하는데, 계산량은 토큰 1개분이라 적다 → **memory-bandwidth-bound**.
- 대략적 감: 가중치 크기 ÷ 메모리 대역폭 ≈ 토큰당 최소 시간. (예시: 2GB 모델, 대역폭 50GB/s → 이론상 최대 약 25 토큰/초. 실제는 더 낮음)
- 지표: **TPOT / tokens per second**.


![Decode 병목](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig12_decode_bottleneck_3b1b.jpg)
*그림 12. 토큰 하나를 쓸 때마다 메모리의 가중치 전체가 좁은 통로(메모리 대역폭)를 지나야 해서, 연산 장치는 대부분 기다립니다. 대략 토큰/초 ≈ 대역폭 ÷ 모델 크기.*

![Prefill Decode 흐름도](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig16_prefill_decode_flow_3b1b.jpg)
*그림 13. 흐름 정리: Prefill이 KV 캐시를 채우고 첫 토큰을 내면, Decode가 캐시를 읽고 한 칸씩 추가하며 EOS까지 반복합니다.*


## 10. KV 캐시 — 다시 계산하지 않기

- 디코드에서 새 토큰은 앞의 **모든** 토큰의 K, V가 필요하다.
- 캐시가 없다면: 매 스텝 전체 시퀀스를 처음부터 다시 계산 → 길어질수록 누적 작업이 제곱으로 증가.
- 캐시가 있으면: 새 토큰은 **자기 Q, K, V만** 계산하고, K·V를 캐시에 한 칸 추가, 저장된 K·V들과 어텐션.
- 대가: 메모리. 크기 공식

  `KV 바이트 = 2(K,V) × 레이어 수 × KV 헤드 수 × head_dim × 토큰 길이 × 바이트/값`

- 예시 계산: 레이어 24, KV 헤드 16, head_dim 128, 길이 4096, FP16(2바이트) → 2×24×16×128×4096×2 ≈ **805MB**. 폰에서는 큰 부담.


![KV 캐시 선반](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig13_kv_cache_3b1b.jpg)
*그림 14. 과거 토큰의 K(초록)·V(빨강)는 캐시에 있으니 새 토큰은 자기 q로 꺼내 쓰기만 하고, 자기 K·V 한 칸(진한 칸)만 추가합니다. 답이 길어질수록 캐시도 커져요.*

![KV 캐시 크기](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig17_kv_size_3b1b.jpg)
*그림 15. 문맥 길이에 따른 KV 캐시 크기(예시). KV 헤드를 32 → 8로 줄이면(GQA) 같은 길이에서 캐시가 1/4이 됩니다.*


## 11. GQA — 캐시 다이어트

- **MHA**: Q헤드마다 자기 K/V 헤드 (16 Q → 16 KV)
- **MQA**: 모든 Q헤드가 K/V 1개 공유 (메모리 최소, 품질 손해 가능)
- **GQA**: 중간. Q헤드를 그룹으로 묶어 K/V 공유 (예시: 16 Q → 8 KV, 두 개씩 공유)
- 위 예시에서 KV 헤드 16→8이면 캐시 **805MB → 약 403MB**. 디코드 때 읽을 캐시 양도 줄어 속도에도 도움.


![GQA 사물함](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig14_gqa_3b1b.jpg)
*그림 16. ○ = Q 헤드, □ = K·V 헤드. MHA는 하나씩, GQA는 여러 Q가 한 K·V를 같이 쓰고, MQA는 모두가 하나를 씁니다. □ 수만큼 캐시가 줄어요.*


## 12. 온디바이스에서 왜 더 중요할까

- 폰·노트북은 메모리 용량과 대역폭이 서버보다 작다 → Decode 병목과 KV 캐시 크기가 체감 성능을 좌우.
- 흔한 대응: 가중치 양자화(예: 4비트), KV 캐시 양자화, GQA 모델 선택, 컨텍스트 길이 제한, 슬라이딩 윈도우 등.

![작은 기기 위의 모델](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig15_ondevice_phone_3b1b.jpg)
*그림 17. 작은 기기에서는 OS·앱, 모델 가중치, KV 캐시가 좁은 메모리를 나눠 씁니다. 그래서 KV 캐시·GQA·양자화가 체감 속도를 좌우합니다.*

## 13. 전체 흐름 한 장 정리

```
[질문] → 토큰화 → 임베딩
  → (Prefill) N토큰 한꺼번에 × 모든 층 → KV 캐시 채움 → 첫 토큰
  → (Decode 반복) 새 토큰 1개 × 모든 층(가중치 전부 읽기) + 캐시 참조/추가 → 다음 토큰
  → 종료 토큰 나오면 끝
```

## 용어 미니사전

토큰, 임베딩, RMSNorm, RoPE, Attention(Q/K/V), Residual, FFN, Prefill, Decode, TTFT, KV 캐시, MHA/MQA/GQA, 양자화.

## 참고 (공개 자료)

- Vaswani et al., "Attention Is All You Need" (2017)
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
- Su et al., "RoFormer: Rotary Position Embedding" (2021)
- Ainslie et al., "GQA" (2023)

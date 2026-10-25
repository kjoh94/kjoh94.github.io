---
layout: post
title: "AI는 왜 읽기는 빠르고 쓰기는 느릴까? — On-Device LLM 추론 기초 (Prefill · Decode · KV Cache)"
date: 2026-10-25 12:40:00 +0900
categories: [ai, on-device, llm]
tags: [llm, prefill, decode, kv-cache, gqa, rope, rmsnorm, inference, on-device]
---

이 글은 특정 회사·제품과 무관한 일반 LLM(트랜스포머 디코더) 추론 원리 정리입니다. 숫자는 이해를 돕는 **예시**입니다.


![읽기는 한 번에](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig01_read_fast.jpg)
*그림 1. 종이 더미(긴 질문)가 리본 하나로 뇌(모델)에 한꺼번에 흘러 들어갑니다. 아무리 길어도 "읽기"는 한 번에 처리됩니다.*

![쓰기는 한 글자씩](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig02_write_slow.jpg)
*그림 2. 답을 쓸 때는 타자기가 블록(토큰)을 하나씩 찍습니다. 빨간 블록이 방금 나온 토큰 — 이게 나와야 다음 블록을 찍을 수 있어요.*

![많아도 빠름, 적어도 느림](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig03_contrast.jpg)
*그림 3. 왼쪽 높은 종이 더미(읽을 양 많음)는 빠르고, 오른쪽 짧은 종이(쓸 양 적음)는 느립니다. 이 글은 이 역설의 이유를 따라갑니다.*

## 한 줄 요약

LLM은 질문을 **한꺼번에 읽고(Prefill)**, 답을 **한 토큰씩 쓴다(Decode)**. 쓰는 단계는 계산보다 **메모리에서 가중치를 꺼내 오는 시간**이 병목이고, 이미 계산한 정보를 다시 계산하지 않으려고 **KV 캐시**를 쓴다. 캐시가 커지는 걸 줄이려고 **GQA** 같은 기법을 쓴다.

## 1. 토큰과 임베딩

- 문장은 **토큰**(단어 조각)으로 잘린다. 예: "안녕하세요" → 2~3개 토큰(토크나이저마다 다름).
- 각 토큰 번호는 **임베딩 표**에서 숫자 벡터(예: 길이 2048)로 바뀐다. 이 벡터를 "hidden state"라 부르며, 레이어를 지날 때마다 뜻이 풍부해진다.


![토큰과 임베딩](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig04_token_embed.jpg)
*그림 4. 가위가 문장을 하얀 블록(토큰)으로 자르고, 각 블록은 작은 큐브가 쌓인 막대(임베딩 벡터)로 바뀝니다. 큐브 하나가 숫자 하나 — 실제로는 수천 개가 쌓여 있다고 생각하면 됩니다. 빨간 막대는 "지금 보는 토큰 하나의 벡터"예요.*


## 2. 레이어 한 층의 순서 (Pre-Norm 디코더 블록)

요즘 공개 LLM(Llama 계열 등)의 일반적인 한 층:

```
x ─┬─> RMSNorm ─> Attention(RoPE 포함) ─┐
   └──────────────── + <────────────────┘   (Residual 1)
h ─┬─> RMSNorm ─> FFN(MLP, 예: SwiGLU) ──┐
   └──────────────── + <────────────────┘   (Residual 2)
```


![모델 안의 층](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig05_brain_layers.jpg)
*그림 5. 뇌(모델)를 잘라 보면 같은 모양의 층이 차곡차곡 쌓여 있습니다. 위 코드 블록이 바로 이 층 하나이고, 토큰 벡터는 아래에서 위로 모든 층을 차례로 통과합니다.*

![문장의 길](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig10_sentence_path.jpg)
*그림 6. 토큰 블록들이 층을 타고 올라가는 길. 층마다 RMSNorm → Attention → 더하기 → RMSNorm → FFN → 더하기를 거칩니다.*

이 블록이 수십 층(예시: 24~32층) 반복되고, 마지막에 RMSNorm → 출력층(LM head) → 다음 토큰 확률.

## 3. RMSNorm — 크기 맞추기

- 벡터 값들이 층을 지나며 너무 커지거나 작아지면 학습·추론이 불안정해진다.
- RMSNorm은 벡터를 **제곱평균제곱근(RMS)**으로 나눠 크기를 일정하게 맞추고, 학습된 가중치(γ)를 곱한다.
- 식: `y = x / sqrt(mean(x²) + ε) × γ`
- LayerNorm과 달리 평균을 빼지 않아 계산이 조금 더 가볍다.


![RMSNorm](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig06_rmsnorm.jpg)
*그림 7. 왼쪽의 들쭉날쭉한 막대(값이 너무 크거나 작은 벡터)가 빨간 프레스(RMSNorm)를 지나면 오른쪽처럼 비슷한 높이로 정리됩니다. 막대 사이의 "상대적 차이"(뜻)는 남기고 전체 크기만 맞추는 것이 핵심.*


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


![Attention Q K V](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig07_attention_qkv.jpg)
*그림 8. 빨간 블록(지금 토큰)이 손전등(Q)으로 앞 블록들을 비춥니다. 각 블록에는 열쇠 모양 이름표(K)와 상자(V)가 있어요. 이름표가 손전등과 잘 맞을수록 밝게 빛나고(점수↑), 밝은 블록의 상자 내용을 더 많이 가져옵니다(화살표). 뒤쪽(미래) 블록은 없으니 못 봅니다 = causal mask.*


## 5. RoPE — 위치 정보 넣기

- 어텐션 자체는 순서를 모른다. "개가 사람을 물었다"와 "사람이 개를 물었다"를 구분하려면 위치가 필요하다.
- **RoPE(Rotary Position Embedding)**는 Q와 K 벡터를 위치에 비례한 각도만큼 **회전**시킨다.
- 두 토큰의 Q·K 내적이 **상대 거리**에 따라 달라지게 되어, 가까운/먼 관계를 자연스럽게 반영.
- V에는 적용하지 않는다. K는 회전된 상태로 캐시에 저장된다.


![RoPE](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig08_rope.jpg)
*그림 9. 받침대 숫자가 토큰 위치, 시계 바늘이 Q·K 벡터 방향입니다. 위치가 바뀔수록 바늘이 일정 규칙으로 더 돌아가요. 두 바늘의 각도 차이만 보면 "얼마나 떨어져 있는지"(상대 거리)를 알 수 있습니다.*


## 6. Residual — 원본 지키기

- 각 서브블록 출력을 입력에 **더한다**(x + f(x)).
- 깊은 층에서도 원래 정보가 사라지지 않고, 학습 시 기울기가 잘 흐른다.


![Residual과 FFN](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig09_residual_ffn.jpg)
*그림 10. 회색 본선 도로가 원본 정보(Residual 경로), 빨간 곁길이 FFN(또는 Attention) 작업장을 들렀다 오는 길입니다. 곁길은 + 교차로에서 본선에 "더해져" 합류하므로 원본이 지워지지 않아요.*


## 7. FFN(MLP)

- 토큰마다 독립적으로 벡터를 넓혔다(예: 4배) 다시 줄이며 지식을 꺼내는 단계. 파라미터의 상당 부분이 여기 있다.


![Prefill vs Decode](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig11_prefill_vs_decode.jpg)
*그림 11. 낮은 벽 왼쪽은 Prefill(종이 더미를 한 번에 읽기), 오른쪽은 Decode(타자기로 한 글자씩 쓰기). 같은 모델이지만 일하는 방식이 완전히 다릅니다.*

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


![Decode 병목](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig12_decode_bottleneck.jpg)
*그림 12. 왼쪽 거대한 창고 = 모델 가중치, 오른쪽 작은 톱니 기계 = 연산 장치, 그 사이 빨간 가는 파이프 = 메모리 대역폭. 토큰 하나를 쓸 때마다 창고의 상자를 전부 파이프로 날라야 해서, 기계는 금방 끝내고 기다립니다. 이게 Decode가 느린 이유.*

![Prefill Decode 흐름도](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig16_prefill_decode_flow.svg)
*그림 13. 흐름 정리: Prefill이 KV 캐시를 채우고 첫 토큰을 내면, Decode가 캐시를 참조·추가하며 한 토큰씩 반복합니다.*


## 10. KV 캐시 — 다시 계산하지 않기

- 디코드에서 새 토큰은 앞의 **모든** 토큰의 K, V가 필요하다.
- 캐시가 없다면: 매 스텝 전체 시퀀스를 처음부터 다시 계산 → 길어질수록 누적 작업이 제곱으로 증가.
- 캐시가 있으면: 새 토큰은 **자기 Q, K, V만** 계산하고, K·V를 캐시에 한 칸 추가, 저장된 K·V들과 어텐션.
- 대가: 메모리. 크기 공식

  `KV 바이트 = 2(K,V) × 레이어 수 × KV 헤드 수 × head_dim × 토큰 길이 × 바이트/값`

- 예시 계산: 레이어 24, KV 헤드 16, head_dim 128, 길이 4096, FP16(2바이트) → 2×24×16×128×4096×2 ≈ **805MB**. 폰에서는 큰 부담.


![KV 캐시 선반](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig13_kv_cache.jpg)
*그림 14. 선반 한 칸 = 토큰 하나의 K(이름표)와 V(상자). 앞 토큰 것은 이미 선반에 있으니 다시 만들지 않고 꺼내 쓰고, 새 토큰 것(빨간 상자)만 한 칸 추가합니다. 대신 답이 길어질수록 선반이 계속 커져요.*

![KV 캐시 크기](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig17_kv_size.svg)
*그림 15. 같은 조건에서 KV 헤드 수만 바꿨을 때 캐시 크기. 헤드를 줄이면 막대(메모리)가 그대로 줄어듭니다.*


## 11. GQA — 캐시 다이어트

- **MHA**: Q헤드마다 자기 K/V 헤드 (16 Q → 16 KV)
- **MQA**: 모든 Q헤드가 K/V 1개 공유 (메모리 최소, 품질 손해 가능)
- **GQA**: 중간. Q헤드를 그룹으로 묶어 K/V 공유 (예시: 16 Q → 8 KV, 두 개씩 공유)
- 위 예시에서 KV 헤드 16→8이면 캐시 **805MB → 약 403MB**. 디코드 때 읽을 캐시 양도 줄어 속도에도 도움.


![GQA 사물함](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig14_gqa.jpg)
*그림 16. 사람 = Q 헤드, 사물함 = K/V 헤드. 두 명이 사물함 하나를 같이 쓰면 사물함(캐시) 수가 절반이 됩니다. 모두가 하나를 쓰면 MQA, 한 명당 하나면 MHA.*


## 12. 온디바이스에서 왜 더 중요할까

- 폰·노트북은 메모리 용량과 대역폭이 서버보다 작다 → Decode 병목과 KV 캐시 크기가 체감 성능을 좌우.
- 흔한 대응: 가중치 양자화(예: 4비트), KV 캐시 양자화, GQA 모델 선택, 컨텍스트 길이 제한, 슬라이딩 윈도우 등.

![작은 기기 위의 모델](/assets/img/posts/2026-10-25-ondevice-llm-prefill-decode-kv-cache/fig15_ondevice_phone.jpg)
*그림 17. 같은 뇌(모델)를 작은 폰 위에 올리면, 창고는 작아지고 파이프는 더 가늘어집니다. 그래서 온디바이스에서는 양자화로 상자를 줄이고, GQA로 선반을 줄이는 것이 체감 속도를 좌우합니다.*

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

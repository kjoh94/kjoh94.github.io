---
permalink: /blog/2026/10/31/tokens-09-streaming/
layout: post
title: "스트리밍: 토큰이 나오는 대로 바로 보내기"
date: 2026-10-31 05:40:00 +0900
categories: [llm, inference]
tags: [llm, inference, 3brown, streaming, sse, ttft]
---

> 시리즈: 토큰이 만들어지는 10가지 장치 · 9/11 · [← 8. OpenAI 호환 API](/blog/2026/10/31/tokens-08-openai-compatible-api/) · [10. CPU와 가속기 분담 →](/blog/2026/10/31/tokens-10-heterogeneous-execution/)

> **한 줄 핵심** — 답 전체를 다 만들 때까지 기다리지 않고, **Decode가 한 바퀴 돌 때마다 나온 토큰을 바로 보내요.** 전체 생성 시간은 그대로인데, 사용자는 0.3초 만에 첫 글자를 보고 "빠르다"고 느껴요.

## 0. 큰 루프에서 어디에 꽂히나

![스트리밍이 꽂히는 자리](/assets/img/posts/tokens-09-streaming/fig01.jpg)

*그림 1. 스트리밍은 Decode 원의 출구에 꽂혀요. 루프 자체는 바뀌지 않고, 바퀴마다 나온 토큰을 모았다가 한꺼번에 보내느냐, 바로 보내느냐만 달라져요.*

---

## 1. 같은 10초, 다른 체감

![스트리밍 없음과 스트리밍 비교](/assets/img/posts/tokens-09-streaming/fig02.jpg)

*그림 2. 스트리밍이 없으면 10초 동안 빈 화면이다가 답이 한꺼번에 나오고, 스트리밍이면 0.3초에 첫 토큰이 나오고 이후 일정한 간격으로 이어져요. 끝나는 시각은 같아요.*

사람이 읽는 속도는 초당 대략 5–10토큰이에요. 엔진이 그보다만 빨리 토큰을 내면, 사용자는 **기다린다는 느낌 없이** 읽기 시작해요. 그래서 스트리밍 환경에서는 지표의 의미가 바뀌어요.

| 지표 | 스트리밍 없음 | 스트리밍 |
|---|---|---|
| 체감을 정하는 것 | 전체 시간 | **TTFT** (첫 토큰) |
| 그다음 중요한 것 | — | **TPOT 안정성** (끊김 없이) |
| 5편이 줄이는 것 | 전체 시간 일부 | 체감 그 자체 |
| 7편이 줄이는 것 | 거의 안 보임 | 중간 멈춤 |

5편(Prefix 재사용)과 7편(청크 Prefill)의 가치가 스트리밍 위에서 비로소 눈에 보여요.

---

## 2. 선 위에서는: SSE

대부분의 LLM 서버는 **SSE(Server-Sent Events)** 로 스트리밍해요. HTTP 응답 하나를 닫지 않고 열어 둔 채 `data:` 줄을 계속 흘려보내는 방식이에요.

![SSE 와이어 형식](/assets/img/posts/tokens-09-streaming/fig03.jpg)

*그림 3. Content-Type이 text/event-stream인 응답에 data 줄이 이어져요. delta에는 이번에 새로 생긴 조각만 들어 있고, 빈 줄이 이벤트 하나의 끝, data: [DONE]이 스트림의 끝이에요.*

```text
data: {"choices":[{"delta":{"content":"서울"}}]}

data: {"choices":[{"delta":{"content":"의 맛집은"}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

왜 WebSocket이 아니라 SSE일까요? 서버 → 클라이언트 **한 방향**이면 충분하고, 그냥 HTTP라서 프록시·로드밸런서·브라우저(`EventSource`, `fetch` 스트림)와 잘 맞기 때문이에요.

클라이언트 코드는 짧아요.

```python
stream = client.chat.completions.create(model="qwen3-8b", stream=True,
            messages=[{"role": "user", "content": "서울 맛집 추천"}])
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 3. 함정 1: 토큰 ≠ 글자

![UTF-8 바이트와 토큰 경계](/assets/img/posts/tokens-09-streaming/fig04.jpg)

*그림 4. 한글 '서'(EC 84 9C)와 '울'(EC 9A B8)의 바이트가 토큰 경계와 어긋나 한 글자가 두 토큰에 걸칠 수 있어요. 토큰마다 바로 문자열로 바꾸면 깨진 글자(�)가 보이니, 완성된 글자까지만 내보내고 남은 바이트는 들고 있어야 해요.*

바이트 단위 토크나이저(BPE)에서는 한글·이모지 한 글자가 여러 토큰으로 쪼개질 수 있어요. 그래서 서버에는 **증분 디토크나이저**가 필요해요.

- 지금까지의 토큰을 이어 붙여 디코딩해 보고,
- 끝이 완성되지 않은 UTF-8 바이트면 **내보내지 않고 기다려요.**
- 다음 토큰이 와서 글자가 완성되면 그때 보내요.

`stop` 문자열도 비슷해요. `"\n\n사용자:"` 같은 stop이 여러 토큰에 걸쳐 나올 수 있어서, 서버는 stop의 앞부분과 겹치는 몇 글자를 **잠시 보류**했다가 stop이 아니면 보내요.

---

## 4. 함정 2: 사용자가 창을 닫으면

![연결 끊김 처리](/assets/img/posts/tokens-09-streaming/fig05.jpg)

*그림 5. 클라이언트 연결이 끊기면 HTTP 계층이 감지해 스케줄러에 abort를 보내고, 스케줄러는 그 요청을 배치에서 빼고 KV 블록을 반납해야 해요.*

| 문제 | 안 챙기면 | 챙기는 법 |
|---|---|---|
| 클라이언트 끊김 | 아무도 안 읽는 토큰을 max_tokens까지 생성, KV·배치 자리 낭비 | 끊김 감지 → `abort(request_id)` |
| 느린 클라이언트 | 네트워크 쓰기가 엔진 루프를 막음 | 요청별 출력 큐로 분리 (backpressure) |
| 프록시 버퍼링 | 토큰이 모였다가 한꺼번에 도착 | `proxy_buffering off`, `X-Accel-Buffering: no` |
| 타임아웃 | 긴 답 중간에 연결 끊김 | 프록시 read timeout 늘리기, keep-alive 주석 이벤트 |
| usage 누락 | 스트리밍 과금 집계 불가 | `stream_options: {"include_usage": true}` |

특히 첫 번째가 중요해요. 연속 배치(6편)에서 자리 하나, KV 블록 몇 개가 곧 다른 사용자의 TTFT예요.

---

## 5. 스트리밍과 다른 장치의 관계

- **샘플링(3편)**: 스트리밍해도 결과는 같아요. 같은 seed면 같은 토큰이 같은 순서로 나와요.
- **연속 배치(6편)**: 엔진은 바퀴마다 배치 전체의 토큰을 한꺼번에 내고, 서버가 요청별로 나눠 각 SSE 연결에 흘려요.
- **도구 호출·JSON**: 스트리밍 중에는 JSON이 반쯤 완성된 상태로 와요. 클라이언트가 조각을 모아 파싱해야 해요.
- **엣지 장치**: 음성 비서라면 SSE 대신 토큰을 바로 TTS로 넘기는 것도 같은 아이디어예요. 문장 단위로 끊어 말하기 시작하면 체감 지연이 줄어요.

---

## 6. 얻는 것과 내주는 것

| 얻는 것 | 내주는 것 |
|---|---|
| 체감 지연이 TTFT 수준으로 | 연결을 오래 열어 둠 (서버 연결 수) |
| 사용자가 중간에 끊을 수 있음 | 끊김·버퍼링·디토크나이즈 처리 복잡도 |
| 긴 답에서도 타임아웃 회피 | 클라이언트 파싱이 조금 더 어려움 |

## 직접 만져보기

```bash
curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","stream":true,"messages":[{"role":"user","content":"한국 사계절을 설명해 줘"}]}'
```

`-N`(버퍼링 끔)을 빼고 다시 해 보세요. 클라이언트 쪽 버퍼링만으로도 스트리밍이 "한꺼번에"로 바뀌는 걸 볼 수 있어요.

마지막 편에서는 이 모든 일이 **어떤 하드웨어 위에서** 나뉘어 돌아가는지 봐요.

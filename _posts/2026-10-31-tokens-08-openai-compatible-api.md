---
layout: post
title: "OpenAI 호환 API: 콘센트 규격 맞추기"
date: 2026-10-31 05:32:00 +0900
categories: [llm, inference]
tags: [llm, inference, 3brown, openai-api, serving, http]
---

> 시리즈: 토큰이 만들어지는 10가지 장치 · 8/11 · [← 7. 청크 Prefill](/blog/2026/10/31/tokens-07-chunked-prefill/) · [9. 스트리밍 →](/blog/2026/10/31/tokens-09-streaming/)

> **한 줄 핵심** — 엔진이 아무리 빨라도 **기존 앱이 못 부르면** 쓸 수 없어요. `POST /v1/chat/completions`라는 사실상의 표준 규격에 맞추면, 수천 개의 도구와 앱이 **주소(base_url) 한 줄만 바꿔서** 새 엔진·새 하드웨어를 쓸 수 있어요.

## 0. 큰 루프에서 어디에 꽂히나

![OpenAI 호환 API가 꽂히는 자리](/assets/img/posts/tokens-08-openai-compatible-api/fig01.jpg)

*그림 1. API는 루프 바깥의 현관문이에요. HTTP 요청이 들어와 채팅 템플릿과 토크나이저를 거쳐 Prefill/Decode 루프로 들어가고, 나온 토큰이 다시 글자와 JSON이 되어 나가요.*

지금까지 1–7편은 루프 **안쪽**을 빠르게 만드는 장치였어요. 8–9편은 루프와 **바깥 세상을 잇는** 장치예요. 성능 숫자에는 안 잡히지만, 채택에는 가장 크게 작용해요.

---

## 1. 비유: 전기 콘센트

가전 회사가 콘센트 규격을 자기 맘대로 만들면, 아무리 좋은 청소기라도 집에 꽂을 수가 없어요. LLM 생태계의 콘센트가 OpenAI API 형식이에요. 원래 한 회사의 API였지만, 워낙 많은 도구가 이 형식으로 짜이면서 **사실상의 표준**이 됐어요.

![하나의 규격에 꽂히는 앱과 엔진](/assets/img/posts/tokens-08-openai-compatible-api/fig02.jpg)

*그림 2. 왼쪽의 프레임워크·UI·사내 챗봇·코딩 도우미가 모두 같은 형식으로 요청을 보내고, 오른쪽의 클라우드·서버 엔진·로컬 엔진·엣지 NPU 엔진 중 어디로든 갈 수 있어요.*

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # 이 줄만 바꾸면 끝
r = client.chat.completions.create(model="qwen3-8b",
        messages=[{"role": "user", "content": "안녕"}])
print(r.choices[0].message.content)
```

---

## 2. 요청 한 장에 시리즈 전체가 들어 있다

![요청 JSON 필드와 장치의 대응](/assets/img/posts/tokens-08-openai-compatible-api/fig03.jpg)

*그림 3. 요청 JSON의 각 필드가 루프의 어느 장치를 움직이는지 보여 줘요. model은 가중치와 컴파일본(1, 2편), messages는 Prefill 입력(5편의 재사용 대상), temperature·top_p는 샘플링(3편), stream은 스트리밍(9편)이에요.*

| 필드 | 루프에서 하는 일 | 관련 편 |
|---|---|---|
| `model` | 어떤 가중치·양자화·컴파일본을 쓸지 | 1, 2 |
| `messages` | 채팅 템플릿으로 한 줄이 되어 Prefill | 0, 5 |
| `temperature`, `top_p`, `seed` | 출구의 샘플링 규칙 | 3 |
| `max_tokens`, `stop` | Decode 바퀴의 종료 조건 | 3, 6 |
| `stream` | 토큰마다 보낼지, 다 모아 보낼지 | 9 |
| `tools`, `response_format` | 도구 호출, JSON 형식 강제 | (제약 디코딩) |

응답에는 `choices[0].message.content`, 왜 끝났는지(`finish_reason`: `stop` 또는 `length`), 그리고 **`usage`**(입력·출력 토큰 수)가 들어 있어요. 과금과 모니터링은 거의 이 `usage`로 해요.

---

## 3. 서버 안에서 요청이 지나가는 길

![서버 내부 경로](/assets/img/posts/tokens-08-openai-compatible-api/fig04.jpg)

*그림 4. HTTP 서버 → 채팅 템플릿 → 토크나이저 → 스케줄러 → 엔진 루프 → 디토크나이저 → JSON/SSE. 스케줄러가 6, 7편, 엔진 루프가 4편의 장치를 품고 있어요.*

여기서 가장 자주 사고가 나는 곳은 엔진이 아니라 **채팅 템플릿**이에요. `messages` 배열을 모델이 학습한 그대로의 특수 토큰 문자열로 바꾸는 단계예요.

```text
<|im_start|>system
너는 친절한 비서야.<|im_end|>
<|im_start|>user
안녕<|im_end|>
<|im_start|>assistant
```

템플릿이 모델과 조금만 어긋나도(특수 토큰 하나 누락, 줄바꿈 위치 차이) 모델은 멀쩡한데 **답이 이상해지거나 끝나지 않아요.** 엔진 벤치마크는 좋은데 실제 대화 품질이 나쁘다면 제일 먼저 볼 곳이에요.

---

## 4. "호환"의 깊이

![기능별 호환 수준](/assets/img/posts/tokens-08-openai-compatible-api/fig05.jpg)

*그림 5. 기본 채팅·스트리밍·샘플링 인자는 거의 모든 서버가 지원하지만, 도구 호출·JSON 형식·logprobs·임베딩은 엔진마다 달라요. 에이전트를 붙이기 전에 꼭 확인해야 하는 부분이에요.*

"OpenAI 호환"이라고 적혀 있어도 수준이 달라요.

| 수준 | 되는 것 | 이걸로 돌아가는 것 |
|---|---|---|
| 기본 | chat/completions, stream, 샘플링 인자 | 채팅 UI, 단순 챗봇 |
| 중간 | + `/v1/models`, `usage`, `seed`, `stop` 여러 개 | 대부분의 프레임워크 |
| 깊음 | + `tools`, `response_format`, `logprobs`, 임베딩 | 에이전트, 코딩 도구, RAG |

---

## 5. 실패 모드

| 증상 | 원인 |
|---|---|
| 답이 끝나지 않거나 특수 토큰이 그대로 보임 | 채팅 템플릿·EOS 설정 불일치 |
| 프레임워크에서 "model not found" | `/v1/models` 응답과 model 이름 불일치 |
| 에이전트가 도구를 안 부름 | `tools` 미지원 또는 파서 형식 차이 |
| 토큰 수 과금이 안 맞음 | `usage` 누락, 스트리밍 시 마지막 청크에 usage 없음 |
| 브라우저 앱에서 호출 실패 | CORS 헤더 |
| 외부에 열어 뒀다가 남이 씀 | 인증(API 키) 없이 0.0.0.0 바인딩 |

---

## 6. 엣지·새 하드웨어에서 의미

새 가속기용 엔진을 만들면 보통 자체 C/Python API부터 생겨요. 그런데 그 API로 바로 쓰려는 사람은 거의 없어요. 같은 엔진 앞에 얇은 OpenAI 호환 서버를 하나 세우면,

- Open WebUI 같은 채팅 화면이 **그대로** 붙고,
- LangChain·LlamaIndex 기반 RAG가 **그대로** 붙고,
- 기존 클라우드 앱을 로컬 장치로 옮기는 게 **설정 변경**이 돼요.

엔진 성능 1.2배보다 이게 더 큰 차이를 만드는 경우가 많아요.

---

## 7. 얻는 것과 내주는 것

| 얻는 것 | 내주는 것 |
|---|---|
| 기존 생태계 즉시 연결 | 남의 규격 변경을 계속 따라가야 함 |
| 엔진 교체가 설정 변경으로 | 엔진 고유 기능을 규격에 끼워 넣기 어려움 |
| 테스트·벤치 도구 재사용 | HTTP·JSON 직렬화 오버헤드(작음) |

## 직접 만져보기

```bash
ollama serve &            # 또는 vllm serve, llama-server
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"안녕"}]}'
```

같은 curl을 주소만 바꿔 다른 엔진에 보내 보세요. 응답 모양이 거의 같은 것, 그게 규격의 힘이에요.

다음 편은 이 현관문으로 **토큰을 하나씩 흘려보내는** 스트리밍이에요.

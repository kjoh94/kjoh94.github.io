---
layout: post
title: "Obsidian 자연어 검색 텔레그램 봇 빠르게 붙이기"
date: 2026-03-28 23:45:00 +0900
categories: [ai, obsidian, telegram, openai]
tags: [obsidian, telegram-bot, embeddings, search, openai, personal-knowledge]
---

오늘은 Obsidian vault 안의 문서를 자연어로 검색하고, 결과를 텔레그램에서 바로 받아보는 작은 검색봇을 빠르게 붙였다.

핵심 목표는 단순했다.

- Obsidian md 파일 전체를 한 번 인덱싱하고
- 텔레그램에서 한국어/영어/혼합 질의로 검색하고
- 검색 결과를 누르면 Obsidian 앱에서 해당 노트가 바로 열리게 만들기

처음에는 Pi 5 + 로컬 모델 기반으로 더 크게 가려 했지만, 일단은 **빨리 쓸 수 있는 MVP**를 먼저 만드는 쪽이 맞다고 판단했다. 그래서 임베딩은 OpenAI API를 쓰고, 벡터 저장은 가볍게 JSONL 기반으로 시작했다.

## 전체 구조

구조는 아주 단순하다.

```text
[Obsidian vault md 파일들]
        ↓
[indexer.py]
        ↓
[data/index.jsonl]
        ↓
[search.py / bot.py]
        ↓
[Telegram bot]
        ↓
[GitHub Pages opener]
        ↓
[Obsidian Advanced URI]
```

즉, 초기엔 무거운 DB나 복잡한 서버 없이도 꽤 쓸만한 검색 경험을 만들 수 있다.

## 구현한 것

이번에 붙인 구성은 대략 이렇다.

- `indexer.py`
  - vault의 markdown 파일을 순회
  - frontmatter와 body를 나누고
  - chunking 후 OpenAI embedding 생성
  - 결과를 JSONL로 저장
- `search.py`
  - 질의를 embedding으로 바꾼 뒤 cosine similarity로 top-k 검색
- `bot.py`
  - Telegram에서 일반 텍스트를 검색 질의로 받아 결과 반환
- `obsidian-open.html`
  - Telegram 안에서 직접 `obsidian://` 링크가 잘 먹지 않아 GitHub Pages opener 페이지를 둠

## 검색 품질 쪽에서 한 일

처음 버전은 그냥 본문 chunk만 넣었는데, 그렇게 하면 한국어 질의에서 미묘하게 아쉬운 경우가 있었다. 그래서 아래를 추가했다.

- 제목 / 경로 / 태그를 함께 넣는 metadata-aware chunking
- heading 기준 분할
- 본문 중심 snippet 정리
- 일부 한영 alias 보강
  - 예: 한국어/영어 혼용 키워드 쌍
- incremental indexing을 위한 `file_hash`
- 이후 질의를 더 정교하게 처리하기 위한 metadata 필드 추가
  - folder
  - mtime
  - frontmatter date/type/tags

즉 단순 semantic search에서 끝내지 않고, 나중에

- "오늘 수정한 문서"
- "시작 안 한 TODO"

같은 질의도 더 잘 다룰 수 있도록 바닥을 깔아뒀다.

## 텔레그램 봇 쪽에서 손본 부분

텔레그램에서 쓸 때 중요한 건 검색 정확도만이 아니었다. UX가 더 중요했다.

그래서 이런 것들을 붙였다.

- `/start`, `/help`, `/health`, `/topk`
- 기본 결과 개수 3개
- 질의 직후 typing 표시
- preload로 첫 질의 체감 속도 개선
- 결과 없을 때도 `Top 0` 응답
- 에러 나도 침묵하지 않고 fallback 응답
- 경로 중심의 짧은 결과 포맷

모바일에서 길게 주르륵 나열되는 응답은 보기 힘들어서, 제목보다 **경로 중심**으로 짧게 보여주도록 정리했다.

## 제일 까다로웠던 부분: Obsidian 열기

사실 제일 시간을 잡아먹은 건 검색이 아니라 **검색 결과에서 Obsidian을 정확히 여는 것**이었다.

처음엔 Telegram 메시지 안에 `obsidian://open?...` 링크를 직접 넣으려고 했다. 그런데 Telegram이 커스텀 스킴 링크를 기대만큼 잘 처리하지 않았다. 그래서 우회가 필요했다.

최종적으로는 아래 조합으로 정착했다.

- Telegram에서는 HTTPS 링크만 보여줌
- GitHub Pages의 opener 페이지가 중간에서 받음
- 그 페이지에서 Obsidian Advanced URI로 넘김
- 결과 경로 텍스트 자체를 클릭 링크로 사용

그리고 여기서 또 하나 중요했던 게 vault 이름과 opener 파라미터를 실제 환경과 정확히 맞추는 것이었다. 이 값이 어긋나면 원하는 노트가 정확히 열리지 않는다.

## 실제 동작 느낌

말로만 쓰면 감이 덜 오는데, 실제 사용 흐름은 아래와 같다.

### 1) 텔레그램에서 자연어로 검색

![텔레그램 검색 결과](/assets/img/posts/2026-03-28-obsidian-natural-search-telegram-bot/telegram-search-result.jpg)

검색 결과는 경로 중심으로 짧게 보여주고, 각 결과를 누르면 GitHub Pages opener 링크로 넘어간다.

### 2) opener 링크 확인 후 열기

![링크 열기 확인](/assets/img/posts/2026-03-28-obsidian-natural-search-telegram-bot/open-link-confirm.jpg)

Telegram 안에서는 `obsidian://` 커스텀 스킴을 바로 다루기 까다로워서, 중간에 HTTPS opener 페이지를 두는 방식이 훨씬 안정적이었다.

### 3) 최종적으로 Obsidian 노트가 정확히 열림

![Obsidian 노트 열린 화면](/assets/img/posts/2026-03-28-obsidian-natural-search-telegram-bot/obsidian-note-opened.jpg)

결과적으로는 **텔레그램 검색 → 링크 클릭 → Obsidian 특정 노트 열기** 흐름이 실제로 매끄럽게 이어진다. 이 부분이 되면서 비로소 “검색 데모”가 아니라 “매일 쓸 수 있는 개인 검색도구” 느낌이 살아났다.

## 지금 상태

지금은 다음이 된다.

- vault 전체 인덱싱
- 한국어 / 영어 / 혼합 질의 검색
- Telegram에서 검색
- 결과 경로 클릭
- Obsidian에서 해당 노트 열기

즉 "작동하는 개인 검색도구" 수준까지는 왔다.

## 다음 할 일

MVP는 완성됐고, 이제 남은 건 검색 정교화다.

예를 들면 이런 것들.

- snippet 품질 개선
- metadata-aware filtering
- `오늘 수정한 문서`, `시작 안 한 todo` 같은 질의 정교화
- rerank / threshold 검토
- 로컬 임베딩 / Pi 5 마이그레이션

지금 느끼는 건 하나다.

**검색 MVP 자체를 만드는 것보다, 실제로 매일 쓰고 싶게 만드는 UX와 정밀도를 다듬는 일이 더 어렵고 더 중요하다.**

그래도 이 단계까지 오면 그다음부터는 훨씬 재미있다. 이제부터는 "될까?"가 아니라 "얼마나 잘 되게 만들까?"의 문제니까.

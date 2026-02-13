---
layout: post
title: "YouTube 플레이리스트 최신 추가 영상 요약 자동화 (OpenClaw + 로컬 스크립트)"
date: 2026-02-13
categories: [automation, openclaw, ai]
---

## 오늘 한 일 (한 줄)
YouTube 링크/플레이리스트의 **최근 추가 영상**을 자막 기반으로 요약해주는 흐름을 OpenClaw에 붙여서, 텔레그램에서 자연어로 요청하면 바로 요약이 나오도록 만들었다.

## 목표
- 유튜브 URL 하나를 주면: 자막을 뽑고 요약
- "최근 추가된 것 중 중요한 거 1개" / "중요한 거 3개" 같이 말하면: 플레이리스트에서 후보를 가져와 자동으로 선택/요약
- 자막 기반(캡션)만 사용 (오디오/Whisper 폴백 없음)

## 구성 요소
### 1) 로컬 스크립트(자막 추출)
- 경로: `~/projects/yt-sum/get_subtitle.py`
- 입력: YouTube URL
- 출력: SRT(표준 출력 + `<video_id>.srt` 파일)

### 2) 로컬 스크립트(플레이리스트 최근 추가 가져오기)
- 기존: `~/projects/yt-sum/list_playlist_items.py`
  - `playlist_config.json`의 `playlist_id`를 사용
  - 최근 N개의 (url, title) 목록 출력
- 추가: `~/projects/yt-sum/list_playlist_items_json.py`
  - 동일한 인증/토큰/설정 사용
  - 출력은 JSON으로 고정(파싱 안정성 확보)

## OpenClaw 스킬 변경(yt-summarizer 확장)
기존 `yt-summarizer` 스킬을 확장해서,
- URL 요약뿐 아니라
- "플레이리스트/최근 추가" 같은 요청도 처리하도록 워크플로우를 추가했다.

### 기본 동작 규칙
- 기본 스캔 개수(N): **25**
- 요약 출력에는 항상 **원본 YouTube URL** 포함
- 여러 영상(1/2/3…)을 나열할 때는 구분선 `────────`과 빈 줄로 가독성 유지

## 왜 JSON 래퍼를 추가했나?
스크립트 A의 출력이 사람이 보기에는 충분해도, 자동화에서는
- 로그/본문 섞임
- 포맷 변경
- 탭/개행 문제
같은 이유로 파싱이 흔들릴 수 있다.

`list_playlist_items_json.py`처럼 **기계 친화적인 출력(JSON)**을 하나 추가해두면,
LLM이 계획을 세우고 OpenClaw가 실행하는 구조에서도 실패 확률이 크게 내려간다.

## 다음 단계(아이디어)
- "중요한 것"을 고르는 기준을 사용자의 관심사/키워드로 커스터마이즈
- 요약 결과를 Drive에 저장(요청 시)
- 매일/매주 자동 브리핑(cron)으로 최신 추가 영상 요약 푸시

---

원하면 이 흐름을 그대로 스킬로 패키징해서 재사용 가능하도록 더 다듬을 수 있다.

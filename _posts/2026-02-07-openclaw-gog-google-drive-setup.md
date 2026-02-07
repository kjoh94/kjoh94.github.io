---
layout: post
title: "OpenClaw에서 Google Drive 연동하기 (gogcli)"
date: 2026-02-07 23:37:00 +0900
categories: [openclaw, gog, google]
---

# OpenClaw에서 Google Drive 연동하기 (gogcli)

작성일: 2026-02-07 (KST)

## 결론(현재 상태)
- ✅ OpenClaw에서 `gog`(gogcli)로 **Google Drive 검색/업로드**까지 동작 확인 완료
- ✅ 문서(Docs), 스프레드시트(Sheets)도 **권한 범위 포함**(drive, docs, sheets)
- ❌ 캘린더(Calendar)는 **아직 권한/토큰에 포함되지 않음** → 원하면 추가로 붙이면 됨

## 1) 준비물
- Google Cloud에서 발급한 OAuth 클라이언트(Desktop/Installed 형태 권장)
  - `client_id`, `client_secret`, `redirect_uris` 포함된 JSON
- 연동할 구글 계정 이메일(예: `kjoh94@gmail.com`)

## 2) 사용한 도구
- `gog` v0.9.0

## 3) 실제로 바뀐 파일/저장 위치
### (A) OAuth 클라이언트 크리덴셜
- 입력 받은 JSON 원본: OpenClaw inbound media
- 안전 보관(복사본): `~/.openclaw/secrets/gog_client_secret.json` (권한 600)
- gog가 사용하는 credentials 저장 위치: `~/.config/gogcli/credentials.json`

### (B) gog 토큰 저장(키링)
- gog 설정 파일: `~/.config/gogcli/config.json`
  - keyring backend를 `file`로 설정
- OpenClaw 환경변수(재시작에도 유지되도록): `~/.openclaw/openclaw.json`의 `env.vars`에 `GOG_KEYRING_PASSWORD` 설정
  - *문서에는 비밀번호 값을 적지 않음(보안)*

## 4) 셋업 과정 요약
1. `gog auth credentials <client_secret.json>` 로 OAuth 클라이언트 등록
2. `gog auth add <email> --services drive,docs,sheets` 시도
   - 모바일/리다이렉트/토큰 교환 단계에서 타임아웃 이슈 발생
3. 우회: `--manual` 플로우로 authorization redirect URL 확보
4. (핵심) authorization code를 **직접 토큰 엔드포인트에 교환**(네트워크 타임아웃 우회)
5. 나온 **refresh token**을 `gog auth tokens import`로 keyring에 저장
6. `gog auth list`로 계정/서비스 확인

## 5) 동작 테스트
### Drive 검색
- 예: `gog drive search "pdf" --max 3 --account kjoh94@gmail.com`

### Drive 업로드
- 예: `gog drive upload ./파일.txt --account kjoh94@gmail.com`

## 6) 캘린더/지메일/연락처 등 추가 연동은?
현재 토큰 서비스는 `drive,docs,sheets`만 포함.

### 캘린더도 붙이려면
- (권장) 같은 계정으로 다시 권한 추가:
  - `gog auth add kjoh94@gmail.com --services drive,docs,sheets,calendar --force-consent`
  - 또는 `--services all` (너무 광범위하면 비추)

### 지메일도 붙이려면
- `gog auth add kjoh94@gmail.com --services gmail,drive,docs,sheets --force-consent`

> 주의: 새로 권한 추가할 때는 refresh token이 갱신/변경될 수 있으니, 끝나면 `gog auth list`로 확인.

## 7) 자주 겪는 문제(오늘 겪은 것)
- Google OAuth에서 "앱이 검증되지 않음/테스트 사용자 아님"으로 403 차단
  - 해결: OAuth consent screen에서 Test user 등록 + 필요한 API Enable
- `exchange code: ... context deadline exceeded`
  - 원인: 토큰 교환(POST https://oauth2.googleapis.com/token)에서 gog 내부 타임아웃
  - 해결: manual code 확보 → 별도 교환 후 refresh token import(또는 PC 브라우저에서 재시도)

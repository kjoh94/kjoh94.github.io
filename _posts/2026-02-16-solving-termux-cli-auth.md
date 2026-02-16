---
layout: post
title: "Gemini/Codex CLI의 Termux 인증, 설정 파일 복사로 해결하기"
date: 2026-02-16 09:25:00 +0900
categories: dev-environment
---

### **1. 요약**
`gemini cli` 및 `codex cli`를 로컬(데스크탑, Termux)에 설치하고, `gemini cli oauth`를 통해 OpenClaw 모델 연동까지 성공적으로 완료함. 특히 Termux에서 발생한 인증 문제를 데스크탑의 설정 파일을 이식하여 해결하는 핵심적인 워크플로우를 확인함.

### **2. 문제 상황**
- 초기 목표: Android(Termux) 환경에서 `gemini cli oauth`를 통해 OpenClaw 모델 인증 및 연동.
- 발생 문제: Termux 환경에서 직접 인증 시도 시, 정상적으로 처리되지 않고 실패함.

### **3. 해결 과정**
- **가설:** 인증 프로세스 자체는 환경 종속적이지만, 생성된 인증 토큰/설정 파일은 환경 독립적일 것이라 가정.
- **검증:**
    1.  먼저 **데스크탑 환경**에서 `gemini cli oauth` 인증 및 OpenClaw 연동을 성공시킴.
    2.  데스크탑에 생성된 **인증/설정 파일들**을 특정함.
        - OpenClaw 에이전트 `auth` 정보가 담긴 JSON 파일
        - `gemini cli` 관련 `credentials.json` 파일
    3.  해당 파일들을 **Termux 환경의 동일한 경로**로 그대로 복사함.
- **결과:** 별도의 인증 절차 없이, Termux 환경에서 `gemini cli`와 `codex cli`가 즉시 정상 동작함을 확인함.

### **4. 결론 및 인사이트**
- OpenClaw 및 관련 CLI의 인증 정보는 **환경 독립적(environment-agnostic)**이다.
- 데스크탑처럼 인증이 용이한 환경에서 생성된 유효한 인증/설정 파일을 모바일(Termux) 등 다른 환경에 **이식(porting)**하여 복잡한 인증 절차를 건너뛸 수 있다.

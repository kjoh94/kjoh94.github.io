---
layout: default
title: Home
---

## 글 목록

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%Y-%m-%d" }})
{% endfor %}

## 프로젝트

- [YouthShield](https://kjoh94.github.io/youthshield/) - 자녀 유튜브 차단 앱

## PDF 요약 워크플로 (스킬 규칙)

PDF를 요약할 때는 아래 규칙을 기본으로 사용한다.

1) PDF에서 **텍스트 추출로 1차 요약**을 시도한다.
2) 텍스트가 안 뽑히거나(에러/너무 짧음/품질 낮음) 요약이 불가능하면,
   `ocrmypdf`로 **같은 위치에** `*_ocr.pdf`를 만든다.
3) `*_ocr.pdf`로 다시 **텍스트 추출 → 요약**을 수행한다.
4) 결과는 **같은 위치/같은 베이스명**으로 `.txt`에 저장하고, 요약 내용도 채팅에 출력한다.

예시:

```text
paper.pdf  -> paper.txt
paper.pdf  -> paper_ocr.pdf (필요 시)
```

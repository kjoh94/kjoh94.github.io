---
layout: default
title: Home
---

# KJ's Blog

개발 일지와 이것저것 기록하는 공간입니다.

## 글 목록

{% for post in site.posts %}
- [{{ post.title }}]({{ post.url }}) ({{ post.date | date: "%Y-%m-%d" }})
{% endfor %}

## 프로젝트

- [YouthShield](https://kjoh94.github.io/youthshield/) - 자녀 유튜브 차단 앱

---
layout: default
title: 일기장
permalink: /d/deb672b2/
---

# 일기장

{% assign diary_pages = site.pages
  | where_exp: "p", "p.dir contains '/d/deb672b2/'"
  | where: "name", "index.md"
  | where_exp: "p", "p.url != '/d/deb672b2/'"
  | sort: "url"
  | reverse %}

{% for p in diary_pages %}
- [{{ p.title }}]({{ p.url | relative_url }})
{% endfor %}

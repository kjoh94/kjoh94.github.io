---
layout: post
title: "PyTorch 시리즈 11장 — LLM 추론을 코드로: prefill·decode·KV cache"
date: 2026-10-25 21:11:00 +0900
categories: [pytorch]
tags: [pytorch, llm, kv-cache]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [10. 양자화, PyTorch 쪽에서](/blog/2026/10/25/pytorch-10-quantization/)  
> → 다음: [12. 커스텀 op — torch.library와 extension](/blog/2026/10/25/pytorch-12-custom-ops/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/11-llm-inference](https://github.com/kjoh94/pytorch-series-examples/tree/main/11-llm-inference) (`python example.py`)

마지막 응용 장이다. ChatGPT 같은 LLM이 답을 한 글자씩 써 내려가는 과정을 **순수 PyTorch 코드로** 처음부터 만들어 본다. 작은 디코더 하나로 prefill, decode, KV cache, causal mask를 모두 확인한다.

## 1. 다음 토큰 맞히기를 반복

![loop](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig01.jpg)

**그림 1.** LLM은 "지금까지의 토큰들 → 다음 토큰의 점수(logits)"를 내는 함수다. 가장 높은 점수(또는 샘플링)를 골라 뒤에 붙이고 다시 넣는다. 이것이 **autoregressive** 생성이다. 먼저 아주 작은 1층 디코더를 만들자.

```python
import torch, torch.nn.functional as F, time
torch.manual_seed(0)
V, D, H = 50, 32, 4          # 어휘, 임베딩, 헤드 수
hd = D // H

class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D); self.pos = torch.nn.Embedding(128, D)
        self.qkv = torch.nn.Linear(D, 3 * D); self.o = torch.nn.Linear(D, D)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 4 * D), torch.nn.GELU(), torch.nn.Linear(4 * D, D))
        self.head = torch.nn.Linear(D, V)

    def forward(self, ids, start=0, cache=None):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(start, start + T))
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = [t.view(1, T, H, hd).transpose(1, 2) for t in (q, k, v)]
        if cache is not None:                       # KV cache: 과거 K,V 를 앞에 붙인다
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        att = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        x = x + self.o(att.transpose(1, 2).reshape(1, T, D))
        x = x + self.mlp(x)
        return self.head(x)

model = TinyDecoder().eval()
prompt = torch.tensor([[1, 7, 3, 9, 4]])
```

## 2. prefill vs decode

![phases](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig02.jpg)

**그림 2.** 생성은 두 단계다. **prefill**: 프롬프트 N 토큰을 한 번에 넣는다 — 큰 행렬곱이라 계산이 병목(compute-bound), 첫 토큰이 나오기까지의 시간(TTFT)을 정한다. **decode**: 매 스텝 토큰 1개 — 계산은 작은데 가중치 전체를 매번 읽으니 메모리가 병목(memory-bound), 토큰당 시간(TPOT)을 정한다.

## 3. 캐시 없이 하면

![no cache](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig03.jpg)

**그림 3.** 가장 순진한 방법은 매 스텝 **전체 시퀀스를 처음부터 다시** 넣는 것이다. step t에서 t개 토큰의 K, V를 다시 계산하니 총 계산량이 1+2+…+T = O(T²)이다.

```python
@torch.no_grad()
def generate_naive(ids, n):
    for _ in range(n):
        logits = model(ids)                                # 매번 전체
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
    return ids
```

## 4. KV cache

![kv](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig04.jpg)

**그림 4.** 과거 토큰의 K, V는 변하지 않는다(causal이니까). 그러니 저장해 두고, 새 토큰의 K, V만 계산해 **뒤에 붙이면** 된다. 새 쿼리 q는 캐시 전체와 attention한다. 대가는 메모리: `2 × 층수 × 헤드 × head_dim × 길이 × 바이트`. 긴 문맥에서 KV cache가 가중치보다 커지는 이유다.

```python
@torch.no_grad()
def generate_cached(ids, n):
    cache = {"k": None, "v": None}
    logits = model(ids, 0, cache)                          # prefill
    out = ids
    for _ in range(n):
        nxt = logits[:, -1:].argmax(-1)
        out = torch.cat([out, nxt], 1)
        logits = model(nxt, out.shape[1] - 1, cache)       # decode: 토큰 1개
    return out

a = generate_naive(prompt, 20); b = generate_cached(prompt, 20)
print("같은 결과:", torch.equal(a, b))
print(a.tolist())
for name, fn in [("naive", generate_naive), ("kv cache", generate_cached)]:
    t = time.perf_counter(); fn(prompt, 60)
    print(f"{name:8s} {(time.perf_counter() - t) * 1e3:.1f} ms")
```

## 5. causal mask

![mask](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig05.jpg)

**그림 5.** 학습·prefill 때는 모든 위치를 한 번에 계산하되 **미래를 못 보게** 위쪽 삼각형을 -inf로 막는다(softmax 후 0). decode 한 스텝은 마지막 행 하나뿐이고, 그 행은 전부 과거라 마스크가 필요 없다 — 그래서 위 코드는 `is_causal=(T > 1)`이다.

```python
T = 4
mask = torch.full((T, T), float("-inf")).triu(1)
print(mask)
print(torch.softmax(torch.zeros(T, T) + mask, -1))   # 각 행: 과거에 균등 분배
```

## 6. 정적 shape로 만들기

![static](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig06.jpg)

**그림 6.** 위 캐시는 매 스텝 `torch.cat`으로 길어진다 → shape가 계속 변해 컴파일러에 불리하다(6장 guard). 실전 방식은 **최대 길이로 미리 할당**해 두고, 현재 위치에 써넣고, 빈칸은 마스크로 가리는 것이다. 그러면 매 스텝 그래프 모양이 같아 한 번 컴파일로 계속 재사용한다.

```python
MAXLEN = 16
k_cache = torch.zeros(1, H, MAXLEN, hd)
def write_step(k_new, pos):
    k_cache[:, :, pos:pos + 1] = k_new                     # 위치에 써넣기 (in-place)
    valid = torch.arange(MAXLEN) <= pos
    return torch.where(valid, 0.0, float("-inf"))          # 빈칸 마스크 (shape 고정)
for p in range(3):
    m = write_step(torch.randn(1, H, 1, hd), p)
print("항상 같은 shape:", k_cache.shape, m.tolist())
```


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [10. 양자화, PyTorch 쪽에서](/blog/2026/10/25/pytorch-10-quantization/)  
> → 다음: [12. 커스텀 op — torch.library와 extension](/blog/2026/10/25/pytorch-12-custom-ops/)

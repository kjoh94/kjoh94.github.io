---
layout: post
title: "PyTorch 시리즈 11장 — LLM이 답을 한 단어씩 쓰는 법: prefill·decode·KV cache"
date: 2026-10-25 21:11:00 +0900
categories: [pytorch]
tags: [pytorch, llm, kv-cache]
---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [10. 숫자를 작게 만들어 모델 줄이기](/blog/2026/10/25/pytorch-10-quantization/)  
> → 다음: [12. 커스텀 op — torch.library와 extension](/blog/2026/10/25/pytorch-12-custom-ops/)

> 이 글의 예제 코드 전체: [kjoh94/pytorch-series-examples/tree/main/11-llm-inference](https://github.com/kjoh94/pytorch-series-examples/tree/main/11-llm-inference) (`python example.py`)

> **처음이라면 이것만 기억하세요.** LLM은 질문을 **한 번에 읽고(prefill)**, 답은 **한 단어씩 써요(decode)**. 한 단어 쓸 때마다 처음부터 다시 읽으면 너무 느리니까, 읽은 내용을 **메모장(KV cache)**에 적어 둬요. 이 세 가지만 알면 이 장의 나머지는 코드로 확인하는 과정이에요.

ChatGPT 같은 LLM이 답을 한 글자씩 "타이핑"하듯 보여 주는 걸 본 적 있죠? 연출이 아니라, 실제로 **한 번에 한 토큰씩만** 만들 수 있어서 그래요. 이 장에서는 그 과정을 순수 PyTorch로 직접 만들어 보면서, 앞 장들의 내용이 어떻게 쓰이는지 연결해 볼게요. 9장의 "메모리 왕복"과 10장의 "양자화"가 왜 LLM에서 특히 중요한지도 여기서 보여요.

## 0. 비유로 먼저

### 0-1. 질문은 한눈에 읽고, 답은 한 단어씩 쓴다

![prefill vs decode](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig07.jpg)

**그림 7.** 시험지를 받으면 질문은 한 번에 쭉 읽죠. 이게 **prefill**이에요. 질문의 모든 토큰을 행렬 하나로 묶어 한 번에 처리하니까, 계산 장치가 아주 바빠요(계산이 병목).

답은 달라요. "오늘" 다음에 뭘 쓸지는 "오늘"을 써 봐야 알고, "날씨는" 다음은 "날씨는"을 써 봐야 알아요. 그래서 **한 단어(토큰)씩** 만들어요. 이게 **decode**예요. 한 번에 처리하는 양이 토큰 하나뿐이라 계산은 금방 끝나지만, 매번 모델 가중치 전체를 메모리에서 꺼내 와야 해서 **메모리가 병목**이 돼요. [9장](/blog/2026/10/25/pytorch-09-compiler-opt/)에서 본 바로 그 상황이고, [10장](/blog/2026/10/25/pytorch-10-quantization/)의 weight-only 양자화가 LLM에서 잘 먹히는 이유예요(꺼내 올 양 자체가 1/4).

### 0-2. KV cache = 읽은 내용을 적어 둔 메모장

![KV cache 메모장](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig08.jpg)

**그림 8.** 단어 하나를 쓸 때마다 모델은 "앞에 나온 모든 단어"를 참고해야 해요(attention). 아무 준비 없이 하면, 100번째 단어를 쓸 때 앞의 99개를 **처음부터 다시** 계산해야 하죠. 그런데 앞 단어들에서 뽑은 정보(**K**ey, **V**alue)는 새 단어가 생겨도 변하지 않아요. 그러니 한 번 계산한 K, V를 **메모장에 적어 두고**, 새 단어의 것만 추가하면 돼요. 이 메모장이 KV cache예요.

### 0-3. 직접 재 보기: 캐시 있을 때 vs 없을 때

작은 4층짜리 attention 모델로, 질문 32토큰 + 답 64토큰을 만들어 봤어요(examples 레포에 전체 코드).

```python
def step(x, cache=None):              # x: 새로 들어온 토큰들 (T, D)
    new = []
    for i, layer in enumerate(layers):
        q, k, v = layer(x).chunk(3, -1)
        if cache:                     # 메모장에 적힌 앞 토큰들의 K, V를 이어 붙임
            k = torch.cat([cache[i][0], k]); v = torch.cat([cache[i][1], v])
        new.append((k, v))
        T, S = q.shape[0], k.shape[0]
        mask = torch.ones(T, S).tril(S - T).bool()      # 미래는 못 보게 (5절)
        a = (q @ k.T / D**0.5).masked_fill(~mask, float("-inf")).softmax(-1)
        x = F.layer_norm(x + a @ v, (D,))
    return x, new

# 캐시 없음: 매번 전체 시퀀스를 다시 넣음
# KV cache : prefill = step(prompt) 한 번, 그 뒤엔 step(새 토큰 1개, cache)
```

![실측](/assets/img/posts/2026-10-25-pytorch-11-llm-inference/fig09.jpg)

**그림 9.** 이 환경(CPU 1스레드)에서 실제로 잰 결과예요.

| | 처리한 토큰 줄 수 | 시간 |
|---|---|---|
| 캐시 없음 | 4064 | 약 72 ms |
| KV cache | 95 | 약 18 ms |

두 방식으로 만든 답의 차이는 **1e-6 이하** — 사실상 똑같아요. 처리량은 43배 차이인데 시간은 4배 정도인 이유는, 이 모델이 너무 작아서 파이썬 호출 같은 고정 비용이 커서예요. 답이 길어질수록(1000토큰, 10000토큰) 캐시 없는 쪽은 **제곱으로** 느려지고 캐시 쪽은 거의 일정하게 늘어서, 실제 LLM에선 차이가 훨씬 커져요.

대신 공짜는 아니에요. 메모장은 **메모리를 차지**해요. 층 수 × 토큰 수 × 차원만큼 늘어나서, 긴 대화에선 모델 가중치만큼 커지기도 해요. 이 이야기는 tokens 시리즈의 [KV cache 편](/blog/2026/10/31/tokens-04-kv-cache/)에서 더 깊게 다뤄요.

> 더 읽을거리: prefill/decode를 서비스 관점에서 본 [tokens 시리즈 프롤로그](/blog/2026/10/30/tokens-00-prologue-prefill-decode/), 작은 기기에서 돌리는 관점의 [온디바이스 LLM: prefill·decode·KV cache](/blog/2026/10/25/ondevice-llm-prefill-decode-kv-cache/).

이제 정식 용어로 하나씩 자세히 볼게요.

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



## 자주 묻는 질문

**Q. 왜 답을 여러 단어씩 한 번에 못 만드나요?**
다음 단어가 바로 앞 단어에 달려 있어서예요. (여러 개를 미리 추측하고 검증하는 speculative decoding 같은 기법도 있지만, 기본은 한 개씩이에요.)

**Q. KV cache를 쓰면 답이 달라지나요?**
아니요. 위 실험처럼 결과는 같고(차이 1e-6 이하), 다시 계산하는 일만 없어져요.

**Q. "첫 글자가 나오기까지"가 느린 건 왜죠?**
그게 prefill 시간이에요. 질문(프롬프트)이 길수록 길어져요. 그 뒤 글자가 나오는 속도는 decode 속도예요.

**Q. 6장의 compile이나 7장의 export는 여기서 어떻게 쓰이나요?**
decode는 같은 모양의 작은 계산을 수천 번 반복하니까, 그래프로 묶어 최적화하면 효과가 커요. 다만 KV cache가 매번 길어지면 shape이 바뀌니까, 미리 큰 칸을 잡아 두는 "정적 shape" 기법을 써요(6절).

## 퀴즈

1. 질문 전체를 한 번에 처리하는 단계는? → prefill. 한 토큰씩 만드는 단계는? → decode.
2. KV cache에 저장하는 것은? → 앞 토큰들의 Key와 Value.
3. decode가 메모리 병목인 이유는? → 토큰 하나 계산하려고 매번 가중치 전체를 메모리에서 꺼내 와야 해서.

## 정리

- prefill = 질문을 한 번에 읽기(계산 병목), decode = 한 토큰씩 쓰기(메모리 병목).
- KV cache = 앞 토큰의 K, V를 적어 둔 메모장. 결과는 같고 재계산이 사라진다. 대가는 메모리.
- causal mask로 미래를 가리고, 정적 shape로 캐시 칸을 미리 잡는다.


---

> **PyTorch 시리즈** · [전체 목차](/blog/2026/10/25/pytorch-series/)  
> ← 이전: [10. 숫자를 작게 만들어 모델 줄이기](/blog/2026/10/25/pytorch-10-quantization/)  
> → 다음: [12. 커스텀 op — torch.library와 extension](/blog/2026/10/25/pytorch-12-custom-ops/)

---
layout: post
title: "복잡한 에이전트 메모리, 꼭 필요할까? — RAW-RAG가 보여 준 단순 RAG의 힘"
date: 2026-10-30 09:00:00 +0900
categories: [llm, rag, memory]
tags: [rag, agentic-memory, raw-rag, locomo, longmemeval, meme, memoryagentbench]
---

LLM에게 "지난달에 내가 뭐라고 했더라?"를 기억시키는 **장기 대화 메모리(long-term conversational memory)**는 요즘 에이전트 스택에서 빠지지 않는 부품이다. Mem0, MemOS, EverMemOS, MemMachine, Memora… 이름도 많고, 하는 일도 화려하다. 대화에서 사실을 **추출(extraction)**하고, **요약(summarization)**하고, 옛 사실을 **수정·삭제(mutation)**하고, **그래프**를 만들고, 커뮤니티를 찾는다.

그런데 이 논문은 아주 단순한 질문을 던진다. **"이 복잡함이 정말 다 필요한가?"** 그리고 대답 대신 실험을 내놓는다. 원문 대화를 잘라 두고, 질문이 오면 찾아서 다시 줄 세운 뒤 LLM에게 건네는 — 교과서적인 **naive RAG** 하나로 4개 벤치마크, 7개 설정 중 6개에서 비교한 메모리 시스템들 중 1위를 했다.

- 논문: Zhengye Bao, Shengtao Zhang, Haokai Zhang, Shiqi Lou, Muning Wen, Minnan Luo, *Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought*, arXiv 2610.30755 (2026-10) — <https://arxiv.org/abs/2610.30755> · PDF <https://arxiv.org/pdf/2610.30755>
- 코드(저자 공개): <https://github.com/Litmeb/RAW-RAG>
- 논문은 **CC BY 4.0** 라이선스다. 이 글의 그림은 논문 그림을 캡처한 것이 아니라, **논문 본문·표의 숫자로 새로 그린 것**이다(Manim). 숫자는 모두 논문에서 그대로 옮겼고, 내가 계산한 값은 "(계산)"이라고 표시했다.

## 0. 한 장 요약 — 두 갈래의 메모리

![두 갈래의 장기 대화 메모리](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig01.jpg)
*그림 1. 같은 대화 기록에서 출발하는 두 갈래. 위쪽(빨강)은 **agentic memory**: 쓰기(ingestion) 시점에 LLM이 사실을 추출·요약하고, UPDATE/DELETE로 상태를 고치고, 그래프·클러스터로 구조화한 **변환된 스토어**를 만든다. 아래쪽(초록)은 **RAW-RAG**: 원문을 256토큰 창으로 잘라 그대로 저장하고, 질문이 오면 dense 검색 top-128 → cross-encoder 재정렬 top-15 → Reader LLM 순서로 흐른다. 핵심 차이는 "상태 변화를 누가 푸느냐"다. 위는 메모리 시스템이 쓸 때 미리 풀고, 아래는 Reader가 읽을 때 원문을 보고 푼다.*

한 줄로 요약하면 이렇다.

> **원문을 버리지 말고, 잘 찾아서, 좋은 Reader에게 주자. 평가된 장기 대화 과제에서는 그게 대부분 복잡한 메모리보다 낫거나 같다.**

## 1. 배경 — 왜 메모리가 필요하고, 왜 복잡해졌나

LLM이 여러 세션, 여러 작업에 걸쳐 사용자와 대화하면 기록이 쌓인다. 논문 표현으로 "결과적으로 쌓인 기록은 모델의 context window를 훨씬 넘을 수 있다(far exceed a model's context window)." 그렇다고 과거 정보를 버릴 수는 없으니 context 밖에 저장해 두고 필요할 때 꺼내는 **외부 메모리**가 등장했다.

여기서 대부분의 연구는 "그냥 저장"으로 끝내지 않고 **메모리 관리(memory management)**를 붙였다. 논문이 나열한 연산만 해도 다음과 같다.

- extraction(사실 추출), summarization(요약), mutation(수정·삭제), consolidation(통합), clustering(군집), graph construction(그래프 구축), community discovery(커뮤니티 탐지)
- 읽기 쪽에서는 iterative search(반복 검색)나 학습 가능한 retrieval 정책까지

논문은 이런 것들을 통틀어 **agentic memory system**이라는 우산 용어로 부른다 — "고정된 raw-history 검색 파이프라인을 넘어서는 에이전트적 내용 처리, 구조화, 검색 제어"를 하는 메모리 시스템.

## 2. "RAG로는 부족하다"는 두 가지 반론

RAG는 장기 메모리에 부족하다는 말을 자주 듣는다. 논문은 이 주장을 두 갈래로 나눠 정면으로 다룬다.

### 반론 1 — 개념적(conceptual): "RAG는 정적인 문서용이다"

RAG는 보통 **정적 코퍼스**(위키, 매뉴얼)를 떠올리게 한다. 그런데 장기 대화는 정적이지 않다.

- 덮어쓴 사실: "나 서울 살아" → 3주 뒤 "부산으로 이사했어"
- 삭제: "아까 말한 거 잊어 줘"
- 연쇄적인 상태 변화(cascade): A가 바뀌면 그에 의존하던 B도 바뀜

그러니 "현재 상태를 명시적으로 유지하지 않고, 기록을 검색만 해서 충분하겠냐"는 의문이 자연스럽다.

### 반론 2 — 경험적(empirical): "이전 논문들에서 RAG가 졌다"

여러 메모리 논문이 RAG baseline이 자기 시스템보다 낮다고 보고했다. 숫자로 졌으니 할 말이 없어 보인다.

논문은 두 반론을 각각 다른 실험으로 검증한다. 반론 1은 **mutation 감사 + 오라클 실험**으로, 반론 2는 **baseline 재점검 + RAW-RAG 본 실험**으로.

## 3. 반론 1 검증 ① — Mem0는 실제로 얼마나 "수정"하나?

![Mem0 쓰기 결정 감사](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig03.jpg)
*그림 2. Mem0의 모든 쓰기 결정을 센 결과. LoCoMo 전체 대화(n=33,902 결정)와 LongMemEval 500개 예제 전체(n=428,858 결정)에서 대부분은 NONE(아무것도 안 함, 78.48% / 73.04%)이나 ADD(추가, 17.13% / 24.51%)였다. UPDATE+DELETE는 4.39% / 2.46%에 불과하고, 실제로 텍스트가 바뀌어 저장된 "effective mutation"은 1.90% / 1.34%뿐이다. 아래는 UPDATE를 ADD로, DELETE를 NONE으로 바꾼 정책의 LoCoMo 정확도: 67.36% → 69.42%.*

Mem0는 대화가 들어올 때마다 LLM이 ADD / UPDATE / DELETE / NONE 중 하나를 고르는 대표적인 "mutation형" 메모리다. 저자들은 LoCoMo의 모든 대화와 LongMemEval의 500개 예제 전부에서 Mem0의 쓰기 결정을 감사(audit)했다.

| | 전체 결정 수 | ADD | NONE | UPDATE | DELETE | UPDATE+DELETE |
|---|---:|---:|---:|---:|---:|---:|
| LoCoMo | 33,902 | 17.13% (5,806) | 78.48% (26,607) | 3.39% (1,148) | 1.01% (341) | **4.39%** |
| LongMemEval | 428,858 | 24.51% (105,093) | 73.04% (313,236) | 2.30% (9,864) | 0.16% (665) | **2.46%** |

게다가 제안된 update 중 상당수는 **중복이거나 형식이 깨진(malformed)** 것이었다. 실제로 반영된, 텍스트를 바꾸는 수정·삭제(effective mutation)는 전체 결정의 **1.90%(LoCoMo), 1.34%(LongMemEval)**에 그쳤다.

그럼 mutation을 아예 끄면? UPDATE → append-only ADD, DELETE → NONE으로 바꿨더니 LoCoMo 정확도가 오히려 올랐다(카테고리 5 adversarial 제외).

| 카테고리 | Full policy | ADD/NONE only | Δ |
|---|---:|---:|---:|
| Multi-hop | 63.24 | 64.66 | +1.42 |
| Temporal | 49.74 | 53.06 | +3.32 |
| Open-domain | 53.47 | 60.42 | +6.94 |
| Single-hop | 77.05 | 78.28 | +1.23 |
| **Overall** | **67.36** | **69.42** | **+2.06** |

어느 카테고리도 떨어지지 않았다. 논문의 결론은 조심스럽다: "**이 설정에서는** 명시적 mutation이 정확도 이득을 주지 않는다." Mem0 한 시스템, 이 구성에서의 결과라는 한정을 기억하자.

## 4. 반론 1 검증 ② — 원문만 주면 Reader가 상태 변화를 풀 수 있나?

mutation이 드물다는 건 알았다. 그럼 진짜 어려운 상태 변화(삭제, 연쇄, 무효화)가 나오면? 저자들은 **MEME** 벤치마크(exact recall, aggregation, tracking, deletion, cascade, invalidation 6개 과제)에서 **phase-visible oracle** 실험을 했다.

- Reader에게 질문 시점(phase)까지 볼 수 있는 증거를 **시간순으로 전부** 준다.
- 검색 오류는 0이 된다. 하지만 "무엇이 지금 유효한가"는 Reader가 직접 풀어야 한다.

| 과제 | GPT-4o-mini | DeepSeek-V4-Flash | Qwen3.5-35B-A3B |
|---|---:|---:|---:|
| ER (exact recall) | 98.00 | 99.00 | 97.00 |
| Agg (aggregation) | 47.00 | 63.00 | 56.00 |
| Tr (tracking) | 81.00 | 92.00 | 89.00 |
| Del (deletion) | 39.00 | 86.00 | 46.00 |
| Cas (cascade) | 1.83 | **82.32** | 39.63 |
| Abs (invalidation) | 36.15 | **91.54** | 93.85 |
| Overall (6과제 macro 평균) | 50.50 | 85.48 | 70.25 |

DeepSeek-V4-Flash는 cascade 82.32%, invalidation 91.54%. 반면 GPT-4o-mini는 cascade 1.83%다. 논문도 "Reader에 따라 성능 차이가 크다"고 분명히 적는다. 그래도 메시지는 이것이다: **충분한 원문 증거가 있으면, 유능한 Reader는 어려운 상태 전이를 직접 풀 수 있다.** 즉 "상태 관리는 쓰기 때 해야 한다"는 전제가 필수는 아니다. 단, Reader가 약하면 이야기가 달라진다(§10 한계).

## 5. 반론 2 검증 — 이전 논문의 RAG baseline은 공정했나?

저자들은 이전 연구 **14편**을 리뷰하고, 여러 논문의 RAG baseline에서 다음 문제를 찾았다(부록 E).

- 너무 거친 검색 단위(coarse retrieval units)
- 제한된 검색 깊이(restricted retrieval depth)
- 재정렬(reranking) 부재

즉 "RAG가 졌다"는 결과가 **agentic memory가 필요해서**인지, **허술한 RAG baseline 때문**인지 구분되지 않는다는 것. 그래서 제대로 설정한 naive RAG를 직접 만들어 보기로 한다. 그게 RAW-RAG다.

## 6. RAW-RAG — 방법은 정말 이게 전부다

![RAW-RAG 파이프라인](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig02.jpg)
*그림 3. RAW-RAG 파이프라인. 대화를 메시지 타임스탬프·이미지 캡션과 함께 직렬화하고, 256토큰 창을 32토큰씩 겹쳐 잘라 flat index에 임베딩한다(위의 엇갈린 초록 상자). 질문이 오면 질문을 임베딩해 코사인 유사도로 가장 가까운 128개 창을 가져오고, cross-encoder가 질문–창 쌍을 함께 읽어 관련도를 매긴 뒤 상위 15개만 Reader에게 준다. 아래 막대는 128 → 15로 줄어드는 비율. 정렬된 비교 설정에서는 Qwen3-Embedding-4B와 Qwen3-Reranker-4B를 쓴다.*

RAW-RAG는 **LLM 기반 추출, 요약, 그래프 구축, mutation을 하나도 하지 않고** 원문 기록을 인덱싱한다.

**쓰기(ingestion)**
1. 각 대화를 메시지 **타임스탬프**와 **이미지 캡션**을 포함해 텍스트로 직렬화
2. **256토큰 창(window), 32토큰 겹침(overlap)**으로 분할
3. 각 창을 임베딩해 **flat index**(구조 없는 벡터 목록)에 저장. 임베딩은 검색용이고, Reader에게는 **원래 창 텍스트**가 그대로 간다.

**읽기(query)**
1. 질문을 임베딩
2. **코사인 유사도로 가장 가까운 128개 창** 검색 (dense retrieval)
3. **cross-encoder**가 질문–창 쌍을 함께 넣고 관련도를 채점 → **top-15** 선택 (reranking)
4. 질문 + 증거 15개를 Reader LLM에 전달. **시간 순서와 상태 변화를 해석하는 건 Reader의 몫**

용어 정리:
- **Dense retrieval**: 질문과 문서를 각각 벡터로 바꿔 거리로 찾는 방식. 빠르지만 질문과 문서를 따로 보기 때문에 대략적이다(bi-encoder).
- **Cross-encoder reranker**: 질문과 후보 문서를 **한 입력으로 같이** 넣어 점수를 낸다. 느리지만 정확하다. 그래서 128개만 먼저 추린 뒤 그 안에서 재정렬한다.

의사코드로 쓰면 이 정도다(설명용, 실행 코드는 저자 저장소 참고).

```python
# RAW-RAG 개념 의사코드
def ingest(conversation, tokenizer, embed):
    text = serialize(conversation, with_timestamps=True, with_image_captions=True)
    toks = tokenizer(text)
    windows = [toks[i:i+256] for i in range(0, len(toks), 256 - 32)]  # 32토큰 겹침
    return [(w, embed(detok(w))) for w in windows]                       # flat index

def answer(question, index, embed, rerank, reader):
    q = embed(question)
    cands = sorted(index, key=lambda x: -cosine(q, x[1]))[:128]          # dense top-128
    top = sorted(cands, key=lambda x: -rerank(question, detok(x[0])))[:15]  # cross-encoder top-15
    return reader(question, evidence=[detok(w) for w, _ in top])         # 상태 해석은 Reader가
```

쓰기 시점에 LLM 호출이 **0번**이라는 점이 뒤에서 비용 차이로 이어진다.

## 7. 실험 설정 — 4개 벤치마크, 12개 메모리 시스템

| 벤치마크 | 규모 | 무엇을 보나 |
|---|---|---|
| **LoCoMo** | 대화 10개, non-adversarial 질문 1,540개 | single-hop, multi-hop, temporal, open-domain |
| **LongMemEval** | 예제 500개 | recall, temporal reasoning, preference, knowledge update |
| **MEME** | 에피소드 100개, phase별 질문 1,188개 | 동적 메모리 업데이트, 복잡한 의존성 |
| **MemoryAgentBench** | 14개 소스, 질문 2,071개 | accurate retrieval(AR), test-time learning(TTL), long-range understanding(LRU), conflict resolution(CR) |

Baseline: EverMemOS, MemOS, EMem, SimpleMem, MemU, MemMachine, Mem0, Memora, Nemori, MemoryOS, A-MEM, RecMem — 이름을 세면 12개(계산) — 그리고 메모리 없이 전체 기록을 다 넣는 **Full-context**.

공정성 장치:
- LoCoMo·LongMemEval은 각 비교 대상 논문의 평가 설정에 **완전히 맞춰서** RAW-RAG를 돌렸다(부록 A).
- MEME·MemoryAgentBench는 모든 시스템이 **같은 평가 절차**를 따른다.
- 증거 길이 차이를 의심할 수 있으니 **5,000토큰 제한 대조 실험**도 했고, 거기서도 RAW-RAG는 경쟁력을 유지했다(부록 A.4).

"7개 설정"은 벤치마크 × Reader 모델 조합이다: LoCoMo(GPT-4o-mini, GPT-4.1-mini), LongMemEval(GPT-4.1-mini), MEME(GPT-4o-mini, DeepSeek-V4-Flash), MemoryAgentBench(GPT-4o-mini, DeepSeek-V4-Flash).

## 8. 메인 결과 — 7개 중 6개 1위, 그리고 예외

![메인 결과](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig04.jpg)
*그림 4. 왼쪽은 LoCoMo(GPT-4o-mini) Overall, 오른쪽은 MemoryAgentBench(GPT-4o-mini) Overall에서 상위 시스템들. 금색이 RAW-RAG, 회색이 Full-context다. LoCoMo에서 RAW-RAG 89.07 vs MemMachine 88.12, MemoryAgentBench에서 48.32 vs Memora 43.84. 아래 빨간 상자는 유일한 예외: MEME·DeepSeek-V4-Flash에서는 MemMachine(44.71)이 RAW-RAG(43.19)를 1.52점 앞서고, Full-context(63.17)는 그보다 훨씬 높다.*

논문의 핵심 문장: RAW-RAG는 **Full-context를 제외하고** 비교한 메모리 시스템 가운데 **7개 벤치마크–모델 설정 중 6개에서 최고점**이고, 가장 강한 agentic memory 시스템을 **0.13–5.53점** 앞선다.

설정별 격차(논문 수치, 괄호 안은 표에서 확인한 값):

| 설정 | RAW-RAG | 2위 메모리 시스템 | 격차 |
|---|---:|---:|---:|
| LoCoMo · GPT-4o-mini | 89.07 | MemMachine 88.12 | +0.95 |
| LoCoMo · GPT-4.1-mini | 93.70 | EverMemOS 93.05 | +0.65 |
| LongMemEval · GPT-4.1-mini | 87.53 | Memora 87.40 | +0.13 |
| MEME · GPT-4o-mini | 39.30 | MemMachine 37.06 | +2.24 |
| MemoryAgentBench · GPT-4o-mini | 48.32 | Memora 43.84 | +4.48 |
| MemoryAgentBench · DeepSeek-V4-Flash | 53.41 | Memora 47.88 | +5.53 |
| **MEME · DeepSeek-V4-Flash** | 43.19 | **MemMachine 44.71** | **−1.52** |

논문은 LoCoMo 0.65–0.95점, LongMemEval 0.13점, GPT-4o-mini MEME 2.24점, MemoryAgentBench 4.48–5.53점 차이로 앞선다고 적는다.

읽을 때 주의할 점:
- **LongMemEval 0.13점, LoCoMo 1점 미만**은 "압승"이 아니라 "동급 이상"으로 읽는 게 맞다. 논문 기여 문장도 "대부분 설정에서 **matches or exceeds**"라고 쓴다.
- **예외 1개**: MEME · DeepSeek-V4-Flash에서는 MemMachine이 1.52점 앞선다. 흥미롭게도 MemMachine도 저장소에 **원문 대화 내용을 보존**하는 시스템이다(§9.3).
- **Full-context는 별개**: LoCoMo에서는 Full-context(GPT-4o-mini 72.30)가 RAW-RAG보다 낮지만, MEME · DeepSeek-V4-Flash에서는 Full-context가 RAW-RAG보다 **19.98점** 높다(63.17). 기록이 context에 다 들어가고 Reader가 강하면 "다 넣기"가 여전히 강력하다는 뜻이다. MemoryAgentBench에서는 context 길이 한계로 Full-context 결과가 없다.
- MEME 점수 자체가 전반적으로 낮다(최고 40점대). 동적 상태 과제가 아직 어렵다는 신호다.

LongMemEval 세부를 보면 RAW-RAG는 single-session user 100.00, preference 100.00, knowledge update 91.03이지만 multi-session(77.69)과 temporal reasoning(81.45)은 일부 시스템보다 낮다(예: Memora TR 89.50). 모든 칸에서 이기는 건 아니다.

## 9. 왜 단순한 게 이기나 — Ablation과 진단

논문은 세 가지 연구 질문(RQ)으로 정리한다. 모두 MemoryAgentBench, 20,000번의 paired source-stratified bootstrap과 Holm 보정을 썼다.

### 9.1 RQ1 — RAW-RAG는 왜 이렇게 설계됐나

![Ablation](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig05.jpg)
*그림 5. MemoryAgentBench Total(기본 48.32%) 대비 변화량. 빨간 막대는 95% 신뢰구간이 0을 포함하지 않는 유의한 하락이다. reranker를 빼면 −3.92, 메모리 관리 기법을 "더하면" atomic facts −19.62, raw summary −5.81, graph expansion −5.78. 회색인 chunk 크기 변화(128/16, 512/64, 1024/128)는 −0.28~+0.04로 유의차가 없다.*

| 설정 | AR | TTL | LRU | CR | Total | Δ | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| RAW-RAG (기본) | 67.85 | 47.68 | 31.74 | 46.00 | 48.32 | – | – |
| − reranker | 59.13 | 44.29 | 29.66 | 44.50 | 44.40 | −3.92 | [−5.80, −2.06] |
| 128/16 | 70.50 | 46.89 | 28.93 | 46.00 | 48.08 | −0.24 | [−1.54, +1.05] |
| 512/64 | 65.68 | 49.27 | 32.49 | 46.00 | 48.36 | +0.04 | [−1.15, +1.25] |
| 1024/128 | 66.28 | 48.08 | 31.79 | 46.00 | 48.04 | −0.28 | [−1.76, +1.19] |

- **Reranker 제거: −3.92점** (p_Holm = 0.0012). 재정렬은 필수 부품이다.
- **최종 k**: 1개만 주면 −11.73점. k = 5, 10, 30, 60은 기본(15) 대비 1.3점 이내. 반대로 **120개를 주면 −14.57점** — 주로 long-range understanding과 conflict resolution에서 떨어진다. 많이 준다고 좋은 게 아니다.
- **Chunk 크기 128~1,024**: Total에 유의한 변화 없음.

즉 "256/32, k=15"라는 숫자가 마법은 아니다. 중요한 건 **reranking이 있고, 검색 깊이가 적당하다**는 것이다.

### 9.2 RQ2 — 메모리 관리를 더하면 좋아지나?

같은 chunking·임베딩·reranker를 그대로 두고, 인덱스 표현이나 후보 집합만 바꾼 세 변형을 시험했다.

| 추가 기법 | AR | TTL | LRU | CR | Total | Δ |
|---|---:|---:|---:|---:|---:|---:|
| + Atomic facts (사실 추출) | 57.25 | **15.37** | 21.17 | 21.00 | 28.70 | **−19.62** |
| + Raw summary (요약 추가) | 67.53 | 45.89 | 29.62 | **27.00** | 42.51 | −5.81 |
| + Graph-expanded (그래프 확장) | 65.42 | 46.49 | 31.74 | 26.50 | 42.54 | −5.78 |

**하나도 48.32%를 넘지 못했다.**
- **Atomic facts**: test-time learning이 47.68% → 15.37%로 무너진다. 대화 속 "예시(demonstration)"가 사실 단위로 쪼개지며 쓸 수 없게 된 것으로 해석한다.
- **Summary**: accurate retrieval은 거의 유지하지만 주로 conflict resolution에서 잃는다.
- **Graph**: long-range understanding은 RAW-RAG와 같지만 전체로는 5.78점 낮다.
- 참고로 표시 턴 중복 제거(display-turn dedup)도 −2.39점이었다.

논문 문장 그대로: "이 세 메모리 관리 추가는 **이 설정에서** 전체 정확도를 낮춘다."

### 9.3 Coverage vs Utilization — 진짜 병목은 "증거 접근"

![Coverage vs Utilization](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig06.jpg)
*그림 6. 왼쪽 산점도: MemoryAgentBench에서 x축 coverage(충분한 증거를 가져온 질문 비율, 19~46% 범위), y축 utilization(증거가 충분할 때의 정답률, 80~90% 범위). RAW-RAG(금색)는 coverage가 가장 오른쪽(45.29%)이지만 utilization은 중간(84.43%)이다. 가로로 퍼진 폭이 세로보다 훨씬 크다 → 시스템 간 차이는 "가져오느냐"에서 주로 난다. 오른쪽: MEME에서도 coverage 76.35%로 최고이고, 오라클 복구 실험에서 원문 증거(61.60%)가 변환된 스토어보다 높으며, 정답 값을 가져왔을 때 근거 경로까지 같이 가져오는 비율 P(path|result)도 RAW-RAG가 62.34%로 가장 높다.*

RAW-RAG가 잘하는 이유가 "증거를 잘 가져와서"인지 "Reader가 잘 써서"인지 나눠 본다.
- **Coverage**: 질문에 답하기에 충분한 증거를 가져온 비율 (MemoryAgentBench는 LLM judge, MEME는 과제별 결정적 검사)
- **Utilization**: 충분한 증거가 있을 때의 정답률(조건부)

| MemoryAgentBench | Coverage | Utilization |
|---|---:|---:|
| MemU | 19.07 | 89.37 |
| MemMachine | 21.54 | 84.30 |
| EverMemOS | 25.11 | 80.00 |
| MemOS | 34.48 | 85.85 |
| EMem | 34.72 | 89.71 |
| Mem0 | 40.03 | 82.39 |
| **RAW-RAG** | **45.29** | 84.43 |

| MEME | Coverage | Utilization |
|---|---:|---:|
| MemU | 35.69 | 84.20 |
| EverMemOS | 47.64 | 76.86 |
| EMem | 50.08 | 86.55 |
| MemOS | 62.12 | 66.67 |
| Mem0 | 63.22 | 65.51 |
| MemMachine | 70.37 | 82.66 |
| **RAW-RAG** | **76.35** | 73.76 |

- MemoryAgentBench: coverage는 19.07–45.29%로 크게 갈리는데, utilization은 80.00–89.71%로 좁다.
- MEME: coverage 폭 40.66점 vs utilization 폭 21.04점.
- RAW-RAG는 utilization이 최고는 아니지만 **coverage가 가장 높아서 그걸 상쇄**한다.

> **Finding 1.** 증거 접근(evidence access)이 주요 병목이다. RAW-RAG는 충분한 증거를 더 자주 공급해 낮은 조건부 활용도를 상쇄한다.

주의: 시스템마다 "증거가 충분한" 질문 집합이 달라서, Reader 능력 자체를 통제 비교한 건 아니라고 논문이 명시한다.

### 9.4 원문은 "답"뿐 아니라 "답에 이르는 길"도 보존한다

"지금 사는 곳은 부산"이라는 값만 가져와선 부족할 때가 있다. **왜 부산이 현재 값인지**(언제 이사했는지, 무엇이 무엇에 의존하는지) 순서·의존 관계까지 있어야 Reader가 확신하고 답할 수 있다.

MEME의 tracking·cascade after-state 질문에서, 정답 값을 가져온 질문 중 **완전한 근거 경로(시간 순서 또는 의존 규칙·상위 트리거)까지 함께 가져온 비율** P(path | result)를 쟀다.
- **RAW-RAG 62.34%** (최고), MemMachine 56.70%, MemU 9.62%

값을 가져와도 관계를 보존하는 정도는 시스템마다 크게 다르다.

### 9.5 저장소를 통째로 줘도 원문을 못 따라간다

"그럼 변환된 스토어를 검색이 못 찾아서 그런 것 아니냐?"를 확인하는 실험. MEME에서 세 조건을 비교했다.
- **Native**: 각 시스템의 원래 검색
- **Complete-store oracle**: 정답 정보를 이용해, 내보낸 phase-visible 메모리 **전체** 중에서 고름
- **Raw-evidence control**: 같은 정답 정보로 원래 근거 세션(원문)을 고름
- 둘 다 같은 정책, **5,000토큰 패킹 예산**

결과: 오라클은 7개 시스템 모두를 개선했지만, 회복 정도는 저장 표현마다 달랐다. 원문 대화를 저장소에 유지하는 **RAW-RAG 57.54%, MemMachine 57.46%** vs **원문 증거 61.60%**. 변환된 스토어를 가진 시스템들은 더 큰 격차가 남았다. 패킹 한도를 없애도 7개 스토어 모두 원문 증거보다 **7.58–29.38점** 낮았고, 모든 paired 95% 구간이 0을 제외했다.

논문은 공정하게 덧붙인다: 이 잔여 격차가 순수한 "정보 손실"만은 아니다. RAW-RAG조차 못 미치고, 증거 조립 방식이나 lost-in-the-middle 같은 위치 효과도 섞여 있을 수 있다.

> **Finding 3.** 수집(ingestion) 때 정보를 보존하는 게 이후 복구에 중요하다. 수집 때 버려지거나 왜곡된 정보는 검색만으로 되살리기 어렵다.

## 10. 비용과 지연 — 쓰기 때 LLM을 안 부르면

![비용과 지연](/assets/img/posts/2026-10-30-raw-rag-naive-stronger/fig07.jpg)
*그림 7. 왼쪽: MEME 10개 에피소드·110개 질문에서 Add→ready(수집 후 검색 가능해질 때까지) + Search 합계 시간(초). EverMemOS 9,301.48초부터 RAW-RAG 305.91초까지. 오른쪽: RAW-RAG는 다음으로 빠른 시스템보다 5.1배, EverMemOS보다 30.4배 빠르다. 약 100만 토큰짜리 BEAM-1M 대화를 순차로 넣는 실험에서 누적 Add 시간은 RAW-RAG 0.33시간, 다음으로 빠른 완주인 MemOS 3.25시간, Memora 26.95시간. 두 실험 모두 답변 생성·평가 시간은 제외다.*

| 시스템 | Add→ready (s) | Search (s) | Total (s) |
|---|---:|---:|---:|
| EverMemOS | 8217.69 | 1083.79 | 9301.48 |
| Mem0 | 4025.23 | 117.72 | 4142.95 |
| MemMachine | 3104.35 | 411.39 | 3515.74 |
| MemOS | 2591.59 | 177.97 | 2769.56 |
| EMem | 1082.37 | 479.59 | 1561.97 |
| **RAW-RAG** | **50.32** | 255.59 | **305.91** |

(외부 conversation/search worker 1개, 각 시스템의 내부 병렬성은 그대로)

눈여겨볼 점: RAW-RAG의 **Search(255.59초)는 Mem0(117.72초)나 MemOS(177.97초)보다 느리다.** cross-encoder 재정렬 비용이다. 대신 **Add가 50.32초**로 압도적으로 짧아서 합계에서 이긴다. 비용 구조가 "쓰기 무겁고 읽기 가벼움"에서 "쓰기 가볍고 읽기 약간 무거움"으로 옮겨 간 셈이다.

추가로, MemoryAgentBench의 공통 측정 11개 non-conflict 소스에서 §9.2의 세 추가 기법은 **2,160만~1억 930만(21.6–109.3M) LLM 토큰, Add+Search 2.9–8.8시간**이 들었고, RAW-RAG는 **12분**이었다. 정확도도 떨어지는데 비용은 훨씬 더 든다.

**스케일링(BEAM-1M)**: 1,710개 메시지, 거의 100만 토큰에 이르는 한 대화를 각 시스템에 스트리밍. 끝점에서 누적 Add 시간은 **RAW-RAG 0.33h ~ Memora 26.95h**, 다음으로 빠른 완주는 **MemOS 3.25h**. 완주한 순차 실행은 6개다.

## 11. 한계 — 논문이 스스로 말하는 것

- **오프라인·공개 벤치마크 한정**: 인터랙티브, 멀티모달, 더 넓은 운영 환경에서의 성능은 보여 주지 않는다.
- **저장·프라이버시**: 원문을 다 저장하면 저장 공간과 프라이버시(특히 "잊어 줘" 같은 삭제 요구) 문제가 생긴다. 향후 명시적 프라이버시·삭제 제약을 고려한 평가가 필요하다고 적는다.
- **Reader 의존성**: 상태 재구성은 Reader 능력에 달려 있다(§4에서 GPT-4o-mini의 cascade 1.83%를 떠올리자). 약한 Reader에게는 명시적 상태 유지가 이득일 수 있고, 같은 추론을 질의마다 반복하는 비용을 줄일 수도 있다 — 이건 앞으로 검증할 질문이다.
- **구조화·압축 메모리가 이기는 조건**은 아직 열려 있다. 정확도·지연·계산 비용의 trade-off를 따져 찾아야 한다.
- 내 덧붙임: 대부분의 격차는 1점 안팎~5점대이고, 7개 설정 중 1개는 졌다. "agentic memory는 쓸모없다"가 아니라 "**평가된 대화 과제에서** 복잡함의 필요성이 입증되지 않았다"가 정확한 요약이다.

## 12. 실무 takeaway — 메모리 스택을 고를 때

1. **먼저 baseline부터.** 장기 메모리를 만든다면 첫 버전은 "원문 chunk + dense top-k(~128) + cross-encoder rerank(~15) + 좋은 Reader"로 시작하자. 논문 결론도 이것이다: 개선된 agentic memory는 **잘 설정된 naive RAG baseline보다 나은 점을 보이고, 추가된 복잡함이 무엇에 기여하는지 입증해야 한다.**
2. **Reranker는 빼지 말자.** 빼면 −3.92점. 그리고 k를 너무 키우지 말자(120개 → −14.57점).
3. **Chunk 크기에 너무 집착하지 말자.** 128~1,024 범위에서 유의차가 없었다.
4. **원문을 버리는 변환은 신중하게.** 추출·요약은 되돌릴 수 없는 손실을 만들 수 있다(atomic facts −19.62점). 구조를 얹더라도 **원문을 함께 보존**하는 쪽(MemMachine, RAW-RAG)이 복구력이 좋았다.
5. **Mutation 로직은 측정 후에.** Mem0에서 실제 반영된 수정·삭제는 1~2%였고, 꺼도 정확도가 올랐다. 내 워크로드에서 UPDATE/DELETE가 얼마나 일어나는지부터 세 보자.
6. **비용 프로파일을 보자.** 쓰기 때 LLM을 부르지 않으면 ingestion이 수십 배 싸진다. 대신 검색 시 rerank 비용이 생긴다.
7. **Reader가 약하면 다시 생각하자.** 작은 모델을 Reader로 쓴다면 상태 유지형 메모리가 의미 있을 수 있다 — 직접 A/B 하자.
8. **Context에 다 들어가면 Full-context도 후보.** MEME · DeepSeek-V4-Flash에서 Full-context가 RAW-RAG보다 19.98점 높았다.

## 마무리

이 논문의 가치는 새 기법이 아니라 **기준선을 다시 세운 것**이다. "RAG는 장기 메모리에 부족하다"는 통념에 대해 (1) 상태 변화는 생각보다 드물고 유능한 Reader는 원문에서 직접 풀 수 있으며, (2) 이전의 RAG baseline은 허술했고, (3) 제대로 만든 naive RAG는 평가된 과제에서 대부분 이기거나 비기면서 훨씬 싸다는 것을 보여 줬다. 복잡한 메모리를 만들고 싶다면, 이제 그 복잡함이 RAW-RAG보다 무엇을 더 해 주는지 숫자로 보여 줘야 한다.

---

*출처: Bao et al., "Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought", arXiv:2610.30755 (CC BY 4.0). 본문의 표와 그림 1–7은 논문 수치를 바탕으로 새로 정리·재작성한 것이며 원본 그림을 그대로 사용하지 않았다. 표의 "격차" 열과 baseline 개수는 논문 표에서 계산한 값이다.*

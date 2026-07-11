# Agent-friendly related-work DB — 통합 설계

토큰 효율적인 agent용 related-work DB + connected-style 시각화 도구의 전체 설계.
`schema.sql`(SQLite 파생 인덱스)과 짝을 이룬다.

---

## 1. 3층 스택

```
OKF 번들 (정본, source of truth)
  · concept 1개 = markdown+YAML 파일 1개, 크로스링크 = 엣지, index.md = progressive disclosure
        │  (이 번들에서 빌드하는 파생 인덱스 ↓)
파생 검색 인덱스 = SQLite + sqlite-vec + FTS5
  · 카드/구조화필드/엣지/클러스터 + 벡터(SPECTER2) + BM25.  전부 결정론적, 0 LLM 토큰.
        │
MCP 서버 (agent 경계)
  · search / neighbors / expand / subgraph / get_cluster (0토큰) + ingest (LLM 1회)
  · 기본 반환 = 압축 카드 투영(TOON/TSV). 무거운 계층은 명시적 drill-down에서만.
        │
그래프 뷰 (connected 스타일)
  · subgraph → force-directed. 노드 크기=인용, 색=연도. OKF 레퍼런스 비주얼라이저 재활용 가능.
```

핵심 원칙: **결정론적 검색(0토큰)과 LLM 이해(토큰)를 물리적으로 분리하고, LLM 비용은
ingest 때 논문당 1회로 상각해 DB를 "agent가 이미 이해해 둔 것의 캐시"로 만든다.**
query time에 논문을 다시 읽는 순간 토큰 설계는 실패한다.

> 벡터 백엔드는 sqlite-vec 단일. (TurboQuant/turbovec은 검토했으나, LLM 토큰이 아니라
> 벡터 RAM·검색속도·코퍼스 규모 축을 최적화하는 도구라 이번 계획에선 제외.
> 코퍼스가 수백만 편으로 커지거나 RAM 압박이 실제로 생기면 그때 파생 인덱스 층만 교체 가능.)

---

## 2. 데이터 계층 (schema.sql 요약)

| 계층 | 저장 | 용도 | 언제 로드 |
|---|---|---|---|
| L0 card | `paper` (flat 컬럼) | agent 첫 스캔. title·tldr·tags·year·cit | 항상 (기본 반환) |
| L1 fields | `paper_fields` | problem/method/dataset/metric/result/limitation. abstract 재독 대체 | `expand(level=fields)` |
| L2 abstract | `paper_abstract` | 원문 초록 | `expand(level=abstract)` |
| L3 text+emb | `paper_chunk` + `vec_chunk` | 전문 청크 RAG | `expand(level=chunks)` |
| edges | `edge (src,dst,kind,weight,intent)` | cites/similar/shared_* 사전계산 | `neighbors` / `subgraph` |
| clusters | `cluster` | 라벨·요약 사전계산 | `get_cluster` |
| cache | `ingest_cache (content_hash,task,prompt_version)` | LLM 산출물 상각 | `ingest` 내부 |

---

## 3. MCP 도구 계약

공통 규약:
- `format`: `"cards"`(기본, 압축 TOON/TSV) · `"json"`(전체 객체) · `"ids"`(ID만).
- 검색어·ID는 전부 바인드 파라미터. SQL/FTS MATCH 문자열 보간 금지(injection 표면).
- `search`/`neighbors`/`subgraph`/`get_cluster`/`expand` = 0 LLM 토큰. `ingest`만 LLM 1회(캐시 미스 시).
- semantic/hybrid 검색의 쿼리 임베딩은 **로컬 임베딩 모델**(코퍼스와 동일 계열)이 만든다 → agent 토큰 0.

### 3.1 `search` — 하이브리드 검색 (BM25 + 벡터 RRF)
```jsonc
// input
{
  "query": "backdoor attacks on DNNs",
  "k": 20,
  "mode": "hybrid",            // "hybrid" | "keyword" | "semantic"
  "filters": {                  // 모두 optional, 서브쿼리 WHERE로 선적용
    "year_min": 2018, "year_max": 2026,
    "tags": ["backdoor"], "cluster_id": 3, "venue": null
  },
  "format": "cards"
}
// output (format=cards) — 헤더에서 컬럼 1회 선언 후 값 행만 (§4)
// output (format=json)
{ "results": [
  { "paper_id":"S2:14", "title":"BadNets", "tldr":"...", "tags":["backdoor","dnn"],
    "year":2019, "venue":"IEEE Access", "n_citations":980,
    "cluster_id":3, "fields_status":"extracted", "score":0.0421 }
]}
```
동작: keyword=BM25(FTS5), semantic=vec_paper 최근접, hybrid=둘의 RRF(`schema.sql` 쿼리 주석 참조).

### 3.2 `neighbors` — 그래프 이웃 (엣지 종류별)
```jsonc
// input
{ "paper_id":"S2:14",
  "kind":"similar",           // "cites"|"cited_by"|"similar"|"shared_method"|"shared_dataset"
  "k":15, "min_weight":0.5, "format":"cards" }
// output (format=ids)
{ "paper_id":"S2:14", "kind":"similar",
  "neighbors":[ {"id":"S2:22","weight":0.83,"intent":null}, ... ] }
```
`cites`는 `edge.src=:id`, `cited_by`는 `edge.dst=:id` 조회. 인덱스 range scan → 사실상 0 비용.

### 3.3 `expand` — 계층 drill-down (명시적일 때만)
```jsonc
// input
{ "paper_ids":["S2:14","S2:22"],
  "level":"fields",           // "fields" | "abstract" | "chunks"
  "query":"trigger design" }  // chunks일 때만: 관련 청크 선별용
// output (level=fields)
{ "S2:14": { "problem":"...", "method":"...", "dataset":"...",
             "metric":"...", "result":"...", "limitation":"..." } }
```
기본 경로(search/neighbors)를 싸게 유지하는 대신, 무거운 계층은 여기서만 당겨온다.

### 3.4 `subgraph` — 시각화용 서브그래프
```jsonc
// input
{ "seeds":["S2:14"], "hops":1,
  "kinds":["cites","similar"], "max_nodes":60, "min_weight":0.5 }
// output — 그래프 뷰가 그대로 렌더 (노드 크기=n_citations, 색=year)
{ "nodes":[ {"id":"S2:14","title":"BadNets","year":2019,"n_citations":980,"cluster_id":3} ],
  "edges":[ {"src":"S2:14","dst":"S2:22","kind":"similar","weight":0.83} ] }
```

### 3.5 `get_cluster` — 클러스터 lookup (사전계산)
```jsonc
// input
{ "cluster_id":3 }            // 또는 {"paper_id":"S2:14"}
// output
{ "cluster_id":3, "label":"Data-poisoning backdoors", "summary":"...", "size":42,
  "top_papers":[ /* cards */ ] }
```
"이 클러스터 뭐야?"가 LLM 호출이 아니라 조회가 되는 지점.

### 3.6 `ingest` — 논문 추가/갱신 (유일한 LLM 접점, 1회)
```jsonc
// input
{ "source": { "s2_id":"...", "arxiv_id":null, "doi":null, "raw":null },
  "prompt_version":"v1", "force":false }
// output
{ "paper_id":"S2:14", "okf_path":"papers/badnets.md",
  "fields_status":"extracted", "cached":true, "llm_calls":0 }
```
파이프라인:
1. S2/OpenAlex에서 메타(title, abstract, TLDR, SPECTER2 emb, citations) fetch.
2. `content_hash = sha256(normalize(title + "\n" + abstract))`.
3. `ingest_cache(content_hash,'fields',prompt_version)` 조회 → **hit이면 LLM skip**(`cached:true`).
4. miss면 작은 모델(예: Haiku)로 L1 필드+contribution+tags 추출 → `ingest_cache` + `paper_fields`.
5. **OKF concept 파일 write**(정본): frontmatter + `## Structured fields` + `## Cites`/`## Similar` 링크.
6. SQLite upsert: `paper`, `paper_abstract`, `vec_paper`, `edge`(citation=S2, similar=벡터 top-k).
7. (배치) 클러스터 재할당·`cluster.summary` 재생성.

`cached`/`llm_calls`가 상각이 실제로 작동하는지 알려준다. `prompt_version`을 올리면 해당 task만 전역 재추출.

---

## 4. 압축 카드 투영 포맷 (`format:"cards"`)

`v_card` 뷰(§schema)를 헤더 1회 + 값 행으로 직렬화. JSON array-of-objects 대비 균일 레코드에서
30~50% 입력 토큰 절감. TOON식(배열 길이+필드 선언) 또는 순수 TSV 중 택1.

```
cards[3]{paper_id,year,cit,f,title,tldr}:
  S2:14  2019  980  Y  BadNets         Backdoor via poisoned training data
  S2:22  2021  410  N  Trojaning DNNs  Trigger-implanting attack on trained nets
  S2:31  2022  260  Y  Fine-Pruning    Defense combining pruning and fine-tuning
```
- `f` = fields_status(Y/N): agent가 이미 구조화 이해가 있는지 = 굳이 `expand` 안 해도 되는지 신호.
- 파싱이 필요한 응답만 `format:"json"`. 스캔 목적이면 항상 cards.

---

## 5. 토큰 회계

| 도구 | LLM 토큰 | 비고 |
|---|---|---|
| search / neighbors / subgraph / get_cluster | 0 | sqlite 내부 결정론 연산 |
| expand | 0 | 계층 조회만 (당겨온 텍스트는 이후 agent가 읽을 때 비용 발생) |
| ingest | 논문당 1회 (캐시 미스 시) | 작은 모델·배치. 재수집·동일 prompt_version이면 0 |

에이전트가 자연스럽게 토큰을 덜 쓰는 이유: **싼 경로(cards)를 기본 경로로** 두어, 답이 카드에서
나오면 agent가 더 깊이 안 들어간다. 무거운 계층은 전부 명시적 `expand`로 옵트인.

---

## 6. 예시 호출 흐름 (related-work 서베이)

```
1. search("backdoor attacks on DNNs", k=20)      → 카드 20장 (싸다)
2. neighbors("S2:14", "similar", k=10)           → 유사 논문 ID
3. subgraph(seeds=[골라낸 3편], hops=1)          → 그래프 뷰 렌더
4. expand([정말 필요한 5편], "fields")           → 그 5편만 구조화 필드
5. (심층 필요 시) expand(..., "chunks")          → 전문 청크
```
1~3에서 대부분 끝난다. 4~5는 실제로 깊게 볼 소수에만. ingest는 이미 이해를 캐시해 둠 → 재독 없음.

---

## 7. MVP 빌드 순서

1. **시드**: S2/OpenAlex API로 노드·엣지·TLDR·SPECTER2 확보(크롤러 자작 금지).
2. **정본**: 각 논문을 OKF concept 파일로 write. `papers/index.md`로 목록/계층.
3. **파생 인덱스**: `schema.sql` 적용 후 OKF 번들에서 SQLite(+vec+FTS) 빌드. 언제든 재빌드 가능하게.
4. **ingest 캐시**: 작은 모델로 L1 추출, `(content_hash,task,prompt_version)` 캐시.
5. **MCP 서버**: 기존 논문 MCP 서버(arxiv-scholar-mcp 등)를 fork해 위 6개 도구 + cards 투영 얹기.
6. **그래프 뷰**: `subgraph` → d3-force/cytoscape/sigma, 또는 OKF 레퍼런스 비주얼라이저 재활용.

---

## 8. 선행연구 (겹침 확인)

- 시각화(connected): Connected Papers, ResearchRabbit, Inciteful, Litmaps.
- 데이터/그래프: Semantic Scholar S2AG(TLDR·SPECTER2·citation intent·recommendations), OpenAlex.
- 통합 학술 KG+LLM: NLP-KG(BM25+SPECTER2 RRF, 계층 필드, LLM Q&A) — 전체 아이디어와 최근접.
- agent/MCP: arxiv-scholar-mcp, paper-search-mcp 등(search·metadata·TLDR·citation_graph 이미 노출);
  agentic 논문검색 PaSa·PaperScout·PaperQA2.
- 정본 포맷: Google OKF v0.1(2026-06, markdown+YAML, 크로스링크 그래프, index.md).

→ 비어 있는 차별점: **agent 토큰 예산에 최적화된 계층 스키마 + 압축 투영 MCP + OKF 정본 연동.**

---

## 9. 보안 노트 (시스템 보안 연구자 관점)

- retrieved content(abstract/full text)는 prompt-injection 표면. ingest 시 정제, agent context에선
  데이터로 격리(지시로 해석 금지).
- MCP tool description 오염(tool-poisoning) 보고 존재 → 도구 메타데이터 무결성 관리.
- 모든 agent 입력은 바인드 파라미터. FTS MATCH 표현식까지 injection 대상.
- 로컬·에어갭 운영으로 데이터 유출 표면 최소화(파생 인덱스·정본 모두 로컬).

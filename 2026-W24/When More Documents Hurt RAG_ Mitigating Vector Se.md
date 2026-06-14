# When More Documents Hurt RAG: Mitigating Vector Search Dilution with Domain-Scoped, Model-Agnostic Retrieval

**Authors**: Nabaraj Subedi, Ahmed Abdelaty, Shivanand Venkanna Sheshappanavar

**Published**: 2026-06-09 18:26:24

**PDF URL**: [https://arxiv.org/pdf/2606.11350v1](https://arxiv.org/pdf/2606.11350v1)

## Abstract
Retrieval-augmented generation degrades when scaled to large, heterogeneous document collections, where dense similarity loses discriminative power, and top-k retrieval increasingly returns semantically similar but contextually incorrect chunks. We refer to this failure mode as vector search dilution. Even when using hybrid dense+sparse retrieval, we observed this firsthand in a deployed Wyoming Department of Transportation corpus, where scaling from 54 to 1,128 documents (88,907 chunks) reduced accuracy from 75% to below 40%. To address this dilution, we propose MASDR-RAG ( Multi-Agent Scoped Domain Retrieval for RAG) and evaluate it on 200 expert-validated queries across five LLM backbones, six corpora, and two index stacks. Our results indicate that domain scoping using organizational metadata is the key fix, significantly improving P@10 from 0.77 to 0.86 ($p < 0.05$). Furthermore, our investigation of multi-agent orchestration revealed that a high degree of configuration dependence results --creating what we call the precision-faithfulness paradox. Based on these varied outcomes, our practical recommendation is simple: scope first, then perform a single synthesis call, reserving full multi-agent orchestration for genuinely multi-domain corpora paired with native-tool-call backbones. Code and Data will be made public upon acceptance.

## Full Text


<!-- PDF content starts -->

When More Documents Hurt RAG: Mitigating Vector Search Dilution
with Domain-Scoped, Model-Agnostic Retrieval
Nabaraj Subedi1∗, Ahmed Abdelaty2, and Shivanand Venkanna Sheshappanavar1
1Dept. of Electrical Engineering & Computer Science
2Dept. of Civil, Architectural Engineering & Construction Management
University of Wyoming, Laramie, WY 82071, USA
{nsubedi1, aahmed3, ssheshap}@uwyo.edu
∗Correspondence author
Abstract
Retrieval-augmented generation degrades
when scaled to large, heterogeneous document
collections, where dense similarity loses
discriminative power, and top-kretrieval
increasingly returns semantically similar but
contextually incorrect chunks. We refer to this
failure mode asvector search dilution. Even
when using hybrid dense+sparse retrieval,
we observed this firsthand in a deployed
Wyoming Department of Transportation
corpus, where scaling from54to 1,128 docu-
ments (88,907 chunks) reduced accuracy from
75%to below40%. To address this dilution,
we propose MASDR-RAG ( Multi-Agent
Scoped Domain Retrieval for RAG) and eval-
uate it on 200 expert-validated queries across
five LLM backbones, six corpora, and two
index stacks. Our results indicate thatdomain
scoping using organizational metadata is the
key fix, significantly improving P@10 from
0.77to0.86(p <0.05). Furthermore, our
investigation of multi-agent orchestration
revealed that a high degree of configuration
dependence results —creating what we call
theprecision–faithfulness paradox. Based
on these varied outcomes, our practical
recommendation is simple:scope first, then
perform a single synthesis call, reserving
full multi-agent orchestration for genuinely
multi-domain corpora paired with native-tool-
call backbones. Code and Data will be made
public upon acceptance.
1 Introduction
Retrieval-augmented generation (RAG) has be-
come the dominant pattern for grounding LLM
outputs in external knowledge (Lewis et al., 2020;
Guu et al., 2020; Gao et al., 2024). However, the
standard embed–index–retrieve–generate pipeline
scales poorly on regulated enterprise corpora span-
ning thousands of heterogeneous documents (Bar-
nett et al., 2024; Wu et al., 2025). As the corpus
expands across heterogeneous categories, denseretrieval loses its discriminative power. The ef-
fect persists even when the Approximate Nearest
Neighbor (ANN) index returns the true nearest
neighbors (Malkov and Yashunin, 2020; Johnson
et al., 2021): those neighbors are semantically re-
lated to the query yet contextually irrelevant.
We identify and characterizevector search dilu-
tion, asemanticscaling problem. We study this
problem in the current Wyoming Department of
Transportation (WYDOT) chatbot, where scaling
the corpus from54to 1,128 documents across
nine categories reduced accuracy on Standard-
Specification queries from75%to below40%.To
address this, we developed a domain-scoped
retrieval framework, MASDR-RAG, together
with a lightweight single-call variant, HYBRID-
ROUTED.
Our experiments across five LLM back-
bones (Qwen2.5-7B-Instruct (Qwen Team, 2024),
Llama-3-8B-Instruct (Grattafiori et al., 2024),
and three commercial backbones via OpenRouter
(Claude-Haiku-4.5, GPT-5-mini, DeepSeek-V3)),
six corpora (EnterpriseComposite-9, HotpotQA-
distractor (Yang et al., 2018), MULTIHOP-RAG
(Tang and Yang, 2024), NQ-Open , FinanceBench,
and MMLU-Pro), and two index stacks (FAISS
and Neo4j HNSW) identify domain scoping over
organizational metadata as the primary driver
of improved retrieval performance. In contrast,
multi-agent orchestration produces configuration-
dependent results. Under a Gemini production
stack, it reduces RAGAS faithfulness from0.61
to0.35(p <0.01), creating what we call the
precision–faithfulness paradox. However, this ef-
fect does not reproduce under an apples-to-apples
open-source stack. Through controlled ablations,
we further show that this degradation is not sim-
ply a consequence of splitting retrieval into mul-
tiple calls. Instead, it arises from the difficulty
of synthesizing answers across multiple sources
when the retrieved evidence contains dense, near-arXiv:2606.11350v1  [cs.CL]  9 Jun 2026

duplicate passages. These findings point to a prac-
tical design principle for large-scale RAG systems:
scope retrieval first and use a single synthesis step
whenever possible. Our contributions are three-
fold:
1.Diagnosis:We formalizevector search dilu-
tionand characterize how retrieval quality de-
grades as corpus density increases.
2.Architecture and analysis:We intro-
duce the multi-agent retrieval framework
MASDR-RAG and the lightweight variant
HYBRID-ROUTED, along with controlled ab-
lations that isolate the sources of synthesis
failures.
3.Generalization:We evaluate across five
LLMs, six corpora, and two retrieval stacks,
showing that the findings are robust across
models and indexing implementations while
reducing costs relative to iterative ReAct-
style baselines.
2 Related Work
RAG and Dense Retrieval.RAG (Lewis et al.,
2020) pairs a generator with a retriever and has
evolved through query transformation, re-ranking,
and iterative retrieval (Gao et al., 2024);agentic
variants (Singh et al., 2025) let the model decide
when to retrieve. Dense bi-encoders (Karpukhin
et al., 2020) and late-interaction models (Khat-
tab and Zaharia, 2020; Santhanam et al., 2022)
largely supplanted sparse retrieval (Robertson and
Zaragoza, 2009), while hybrid schemes (Sawarkar
et al., 2024) stay competitive on multi-domain cor-
pora. Index scaling is usually framedalgorithmi-
callyvia approximate nearest neighbors (Malkov
and Yashunin, 2020; Johnson et al., 2021); we fo-
cus instead on a complementarysemanticdegrada-
tion. Prior work shows dense retrieval loses dis-
criminative power as the index grows (Reimers
and Gurevych, 2021), irrelevant passages alter
generation (Cuconasu et al., 2024), and long con-
texts introduce noise (Jin et al., 2025). We
share this diagnosis but contribute aretrieval-free,
corpus-intrinsicmeasurement (the dilution factor
δ, §3) along with a deployable fix, and confirm
that the issue is not dense-specific by evaluating
BM25 and ColBERTv2 (§6).
Query Routing and Multi-Agent Systems.
Two lines of prior work contextualize our ap-
proach to domain scoping.Strategy routing(Jeonget al., 2024; Zhang et al., 2025; Guo et al.,
2025) chooses a retrievaldepthor entirepipeline
per query—deciding when and how hard to re-
trieve, not where.Metadata filtering(Poliakov
et al., 2024) masks candidates post-hoc while
still indexing the whole corpus. Our scop-
ing is orthogonal: we route to one ofKpre-
existing organizational scopesthat live in the doc-
ument graph as a first-class field (source_type,
document_series, articlecategory), re-
stricting the index at query time rather than filter-
ing after the fact. Our trained R2-ROUTEDvariant
(App. A33) demonstrates that the choice of routing
targetmatters as much as the routingmodel.
For orchestration, ReAct (Yao et al., 2023)
and LangChain (Chase, 2023) provide general
scaffolding for tool use.Genuinelymulti-agent
RAG assigns distinct roles with inter-agent mes-
saging: MA-RAG (Nguyen et al., 2025) chains
task-specific agents, and SCOUT-RAG (Li et al.,
2026) runs cooperative domain-relevance and re-
trieval agents over graph domains. Our MASDR-
RAG is deliberately simpler—asingle reasoning
agentwithKdomain-scoped tools, where each
“agent” is a scope-bound tool configuration. We in-
clude both multi-agent paradigms as baselines (§8)
and show that, with commercial generators, multi-
round orchestration triggers a faithfulness collapse
that is absent with open-source backbones (Ta-
ble 10).
Reranking, Iterative, and Graph RAG:Two-
stage pipelines rerank a bi-encoder top-Kwith a
cross-encoder (Nogueira et al., 2020); our ablation
(App. A34) shows that while cross-encoder rerank-
ing lifts baseline faithfulness, it doesnotrecover
the multi-agent collapse, ruling out within-scope
ranking noise as its sole cause. Learned-sparse
retrievers such as SPLADE (Formal et al., 2022)
remain competitive; we evaluate the OpenSearch
neural-sparse model (OpenSearch Project, 2024)
as an additional retriever baseline (§6). Iterative
methods—IRCoT (Trivedi et al., 2023), Self-Ask
(Press et al., 2023), and Self-RAG (Asai et al.,
2024)—share ReAct’s multi-round loops, which
our efficiency analysis shows are costly under
open-source backbones. While Shi et al. (2023)
notes that LLMs are distracted by irrelevant con-
text, we demonstrate that fragmented yet domain-
precise context is similarly harmful.Finally, unlike
GraphRAG (Edge et al., 2024), which builds en-
tity–relationship graphs, we use the graph’sorga-

nizationalmetadata as explicit agent boundaries.
Evaluation:RAGAS (Es et al., 2024) measures
standard retrieval quality and faithfulness metrics.
However, standard benchmarks—such as Natural
Questions (Kwiatkowski et al., 2019), HotpotQA
(Yang et al., 2018), MultiHop-RAG (Tang and
Yang, 2024), and long-context suites (Yen et al.,
2025)—rely on homogeneous or synthetic cor-
pora. Consequently, they fail to capture the cross-
domain dilution typical of a regulated enterprise
environment, motivating he multi-domain evalua-
tion frameworks introduced in this work.
3 Vector Search Dilution
3.1 System Context
The corpus comprises 1,128 documents spanning
construction specifications, design manuals, mate-
rials testing procedures, crash reports, transporta-
tion improvement programs, and administrative
reports, ingested into Neo4j as Document→
Section→Chunk. The production system uses
Gemini Embedding (768-d), HNSW, and a BM25
full-text index. Traffic & Crash reports contribute
34.8%chunks despite being1.9%documents (Ta-
ble 1).
Category Docs Chunks % Chk/Doc
Standard Specs 2 2,519 2.8 1,260
Construction Manual 21 6,641 7.5 316
Materials Testing 6 2,180 2.5 363
Design Manual 23 1,405 1.6 61
Traffic & Crashes 22 30,922 34.8 1,406
STIP 59 13,634 15.3 231
Annual Reports 46 2,341 2.6 51
Bridge Program 28 5,399 6.1 193
Other 921 23,866 26.8 26
Total1,128 88,907 100 —
Table 1: Document and chunk distribution by literal
document_seriescategory. Agent scope filters
(App. A11) span broader related-series unions, so the
per-agent counts in Table 13 exceed the per-category
counts . Chunk density varies54×across categories.
3.2 Formal Definition
LetC={c 1, . . . , c N}beNchunks partitioned
intoKcategoriesC 1, . . . ,C K,e:C →Rd
an embedding, andqa query targeting cate-
goryk⋆. The top-mretrieval set isR m(q) =
arg max S⊆C,|S|=mP
c∈Ssim(e(q), e(c)). Dilu-
tion occurs when global precision is much lowerthan scoped precision:
δ(q, k⋆) = 1−Pglobal(q)
Pscoped(q),
whereP global(q)is the fraction of the retrieval set
Rm(q)belonging to the target categoryk⋆when
retrieval ranges over all ofC, andP scoped(q)is the
same fraction when retrieval is restricted toC k⋆(so
Pscoped≈1by construction). Thusδ=0is no dilu-
tion andδ→1severe dilution.
3.3 Empirical Measurements
Categories with smaller chunk populations suffer
the most severe dilution (Designδ=0.53; Specs
δ=0.43), while high-density categories (Construc-
tion Manualδ=0.10) largely resist it. The Spear-
man correlation betweenlog(chunk count)and
meanδacross the eight scopable categories is
ρ=−0.60(p=0.12). Withn=8categories, this
single correlation is suggestive rather than statis-
tically conclusive on its own; we corroborate it
on the reproducible cross-DOT replication of §9,
where the same correlation under the open-source
BGE-M3 stack ranges fromρ=−0.68(WYDOT,
10categories) toρ=−0.95(CDOT,10categories).
Category Chunksδmeanδrangen
Design Manual 1,405 0.53 0.00–1.00 12
Standard Specs 2,519 0.43 0.00–1.00 23
Materials Testing 2,180 0.22 0.00–0.50 20
Bridge Program 5,399 0.21 0.00–0.70 12
STIP 13,634 0.17 0.00–0.70 9
Traffic & Crashes 30,922 0.16 0.00–0.60 7
Annual Reports 2,341 0.12 0.00–0.30 4
Construction Manual 6,641 0.10 0.00–0.80 21
Table 2: Per-category dilution factor with per-query
ranges.
103 104
Chunks per category (log scale)0.00.10.20.30.40.50.6Dilution factor δ
Design Manual
Standard Specs
Materials Testing
Bridge ProgramSTIPTraffic & Crashes
Annual Reports Construction ManualSpearman ρ=−0.60, p=0.12n=8
linear fit (log-x)
Figure 1: Dilutionδvs. chunk count, eight WYDOT
scopes; Spearmanρ=−0.60(p=0.12).

scoped retrieval:85–98%search-space reductionUser
QueryHybrid Router
(Regex→LLM)Orchestrator
(function-calling)Specs Agent
|C|≈2.5k
Construction Agent
|C|≈6.6k
Materials Agent
|C|≈2.2k
Design Agent
|C|≈1.4k
Traffic & Crashes Agent
|C|≈30.9k
Bridge Agent
|C|≈5.4k
STIP Agent
|C|≈13.6k
Annual Reports Agent
|C|≈2.3k
Highway Safety Agent
|C|≈30.9kNeo4j
Knowledge
Graph
Synthesiser LLM
(Qwen-7B / Llama-8B / Gemini)
Answercategory
scoped ANN searchtop-kchunks
chunks + query
Figure 2: MASDR-RAG / HYBRID-ROUTEDdata flow. Regex-then-LLM router dispatches to one of nine
WYDOT domain agents; each agent ANN-searches itsdocument_seriesscope in the Neo4j graph, and a
Qwen-7B / Llama-8B / Gemini synthesizer generates the answer.
Geometrically, the dilution corresponds to the
retrieval-time source confusion: on Composite-9,
the diagonal ofP(retrieved source|gold source)
is only0.59under monolithic search, lifting to
0.84under regex scoping and0.90under HYBRID-
ROUTED(Figure 5, App. A27). Scoping does
not improve the embedder; it forces the retrieval
neighborhood to respect the source label already
present in the document graph. A t-SNE projec-
tion (App. A6) and a worked WYDOT failure case
(App. A26) illustrate the same mechanism.
4 Architecture: MASDR-RAG and
Hybrid-Routed
The architecture has three components: (1)
Domain-scoped retrieval, where each agent re-
stricts the search to documents matching a Neo4j
metadata filter, reducing the effective search space
by85–98%(Table 13), (2)Hybrid routingthat
runs a fast regex matcher first and falls back to a
zero-shot classifier using an LLM, and (3)Multi-
agent orchestrationthat dispatches to nine do-
main agents via function calling. Figure 2 sum-
marizes the data flow.
Each agent’s scope filter reduces its effective
search space by65–98%relative to the full cor-
pus, with a weighted average of90.4%(per-agent
breakdown in App. A13, Table 13). The orches-
trator uses up to five tool-call rounds; HYBRID-
ROUTEDuses at most two LLM calls per query(one router, one synthesizer).
We useorchestrationfor this multi-round tool
loop and are explicit about what it is not:
MASDR-RAG is asingle reasoning agentwith
Kdomain-scoped retrieval tools, and the per-
domain “agents” are scope-bound tool configu-
rations rather than autonomous agents that rea-
son or communicate independently. The contrast
with genuinely multi-agent RAG — where sepa-
rate planner, extractor, and synthesis agents ex-
change intermediate reasoning — is drawn against
the MA-RAG and SCOUT-RAG baselines in §2
and §8.
5 Evaluation: Proprietary Stack
200 expert-validated WYDOT queries (Gemini
2.5 Flash answer generator,95%bootstrap CIs,
permutation tests atα=0.05).
Metrics:We report four metrics throughout.
P@10andR@10are precision and recall at rank
10, computed against the expert-labeled target
scope of each query: a retrieved chunk counts
as relevant if it belongs to that scope.Correct-
ness(Corr) is a binary per-answer judgment —
an LLM judge (Qwen-2.5-7B, distinct from every
system under test) marks each generated answer as
correct or incorrect against the reference answer,
and we report the mean; the judge prompt and
rubric are in App. A10 and App. A18.Faithful-

ness(Faith) is the RAGAS faithfulness score in
[0,1], the fraction of claims in the generated an-
swer that are supported by the retrieved context.
Unless noted,nin a table is the number of queries
scored.
System P@10 R@10 Corr% Faith
Monolithic.77.93 25.5.61
Mono+RRF.75.9627.0.58
LLM+Scoped.85∗.86 24.1.62
MASDR-RAG.86∗.59∗∗33.5.35∗∗
HYBRID-ROUTED.83.84 24.5.62
Table 3: WYDOT Gemini stack (n=200): scoping
lifts P@10.77→.86; MASDR-RAG’s faithfulness
collapses.61→.35.∗p<.05,∗∗p<.01vs. monolithic.
6 Open-Source Reproducibility
We re-ran all five systems with Qwen2.5-7B-
Instruct and Llama-3-8B-Instruct synthesizers on
BGE-M3 (Chen et al., 2024) embeddings; a sin-
gle L40S GPU handles the200-query sweep in
≈40min (Qwen) /≈100min (Llama).
LLM System p50 (s) p95 (s) tokens calls
Qwen-7B monolithic 6.2 19.2 11.3k 1.00
Qwen-7B regex_scoped 7.6 19.9 10.7k 1.00
Qwen-7B HYBRID-ROUTED6.3 20.2 10.8k 1.44
Qwen-7B MASDR-RAG 10.8 31.7 13.0k 2.09
Qwen-7B ReAct 7.9 19.7 11.6k 2.26
Llama-8B monolithic 9.9 51.0 7.5k 1.00
Llama-8B regex_scoped 12.7 51.7 7.7k 1.00
Llama-8B HYBRID-ROUTED9.2 51.0 6.6k 1.46
Llama-8B MASDR-RAG 23.0 62.1 11.4k 2.08
Llama-8B ReAct 20.3 143.8 39.2k 5.50
Table 4: Opensource replication on WYDOT200-
query. Architectural ranking is backbone-invariant; Re-
Act on Llama-8B blows up in calls/tokens.
External retrieval-only and agentic baselines
(BM25 (Robertson and Zaragoza, 2009), Col-
BERTv2 (Santhanam et al., 2022), LangChain Re-
Act (Chase, 2023), Custom ReAct (Yao et al.,
2023)) on Composite-9 are in App. A28: scoped
single-call systems hit86–90%correctness at a
fraction of LangChain’s12.4s p50. A BEIR
MS-MARCO calibration of the BGE-M3 stack
(nDCG@10=0.854, Recall@10=0.961) anchors
our retrieval numbers to a published baseline
(App. A29).
7 Efficiency: Hybrid Routing vs. ReAct
On Llama-3-8B, ReAct saturates its6-iteration
cap on half of WYDOT queries (mean5.5vs.1.5for HYBRID-ROUTED), driving5.9×more to-
kens (39.2k vs.6.6k),2.2×p50 latency (20.3s
vs.9.2s), and a worse143.8s vs.51.0s p95. On
Qwen-7B’s native function-calling template, Re-
Act stays at2.3iterations, and the latency/to-
ken gap mostly closes, matching prior efficiency
observations (Schick et al., 2023; Parisi et al.,
2022). The routing decision is first-call resolv-
able for domain-scoped corpora, so HYBRID-
ROUTED’s single router + single synthesis call
dominates the latency–correctness frontier (Pareto
plot, App. A30).
8 Cross-Domain Generalization
To test whether dilution and the HYBRID-
ROUTEDfix transfer beyond WYDOT, we repli-
cate it on five public corpora. Figure 3 summarizes
headline correctness.
Corpora: Composite-9:9public sources ap-
proximating enterprise documents (17,994chunks,
ingest in App. A32).HotpotQA-distractor(Yang
et al., 2018):10-paragraph multi-hop, bucketed
into4alphabetic topic scopes (n=2,400dev).
MultiHop-RAG(Tang and Yang, 2024),NQ-
Open(Kwiatkowski et al., 2019),FinanceBench,
andMMLU-Proare used with their published
splits. Span-level HotpotQA metrics (HYBRID-
ROUTEDleads at Contains=.470vs. Monolithic
.427) are tabulated in App. A31.
MA-RAG and SCOUT-RAG on WYDOT:On
the load-bearing WYDOT corpus, the same pat-
tern holds (Table 6): genuine multi-agent base-
lines underperform both our scoped methods
and Monolithic. Regex-Scoped’s35.1%correct-
ness beats MA-RAG’s11.0%and SCOUT-RAG’s
24.1%at roughly1/22×and1/10×the LLM-call
budget, respectively. Faithfulness is reported as
— for both baselines:MA-RAG and SCOUT-RAG
prompts do not request[SourceN]markers,
so our citation-supported judge cannot score faith-
fulness on those outputs; see App. A40 for the im-
plementation and the faithfulness diagnostic.
9 Cross-DOT Replication: Caltrans and
CDOT
To test whether dilution and the scoping fix trans-
fer beyond WYDOTs toother state DOT corpora
— not just the public-domain proxies of §8 — we
scrape and embed two further DOTs:California
(Caltrans)fromdot.ca.govandColorado

WYDOT Composite-9 MultiHop-RAG HotpotQA (CRAG)0.00.20.40.60.81.0Correctness3490
69
40
3590
58
n/a3186
61
35
2974
62
41
2694
60
47
Monolithic Regex-Scoped Hybrid-Routed MASDR-RAG Custom ReActFigure 3: Cross-corpus correctness (Qwen-2.5-7B synth + Qwen judge). Scoping helps when the corpus has
identifiable sub-domains and queries are single-domain (WYDOT, Composite-9); ReAct helps only when queries
are genuinely multi-hop (HotpotQA).
Corpus System Corr.% Faith. p50 (s)
Composite-9Monolithic 90.0 0.76 1.94
Regex-Scoped 90.0 0.80 2.64
HYBRID-ROUTED85.7 0.77 2.86
MASDR-RAG 74.0 0.74 3.12
Custom ReAct 94.4 0.78 4.48
BM25+Qwen 74.0 0.58 1.74
ColBERTv2+Qwen 82.0 0.78 1.80
ColBERTv2 scoped+Qwen 84.0 0.86 2.62
MA-RAG44.4—‡17.7
SCOUT-RAG66.7—‡23.6
MultiHop-RAGMonolithic 69.2 0.46 2.10
Regex-Scoped 58.0 0.47 2.30
HYBRID-ROUTED61.0 0.48 3.10
MASDR-RAG 61.6 0.43 4.40
MA-RAG29.8—‡15.6
SCOUT-RAG55.2—‡21.0
MMLU-ProMonolithic 48.8 0.36 2.40
Regex-Scoped 50.2 0.39 2.60
HYBRID-ROUTED51.8 0.41 3.50
MASDR-RAG 46.0 0.34 5.20
HotpotQAMonolithic 40.2 0.33 1.67
HYBRID-ROUTED34.9 0.37 2.29
MASDR-RAG 41.4 0.27 2.95
Custom ReAct 47.3 0.33 4.93
LangChain ReAct 33.2 0.54 9.11
BM25+Qwen 37.8 0.32 1.77
ColBERTv2+Qwen 40.1 0.33 1.69
Table 5: Cross-domain replication (Qwen-2.5-7B +
Qwen judge). Llama-3-8B numbers in App. A25. MA-
RAG (Nguyen et al., 2025) and SCOUT-RAG (Li et al.,
2026) implementation is in App. A40.
(CDOT)fromcodot.gov. All three corpora are
processed by an identical pipeline (uniform1000-
char chunking, BGE-M3 bf16,L 2-normalized)
and the retrieval-freeδ= 1−purityk=10 proxy
of §3 is computed on each (Table 7); WYDOT is
re-chunked uniformly, so chunk counts differ from
Table 1. Pipeline, scrape methodology, and per-
agent reduction tables are in App. A22.System Corr.% Faith. calls
Monolithic33.5 0.61 1.0
Regex-Scoped35.1 0.61 1.0
HYBRID-ROUTED30.3 0.43 1.5
MASDR-RAG28.7 0.48 2.1
ReAct24.5 0.59 2.3
MA-RAG11.0—‡22.3
SCOUT-RAG24.1—‡10.4
Table 6: WYDOT200-q on the open-source Qwen-2.5-
7B / BGE-M3 stack; see App. A40.
Corpus #docs #chunks largest doc
WYDOT1,128 217,752—
Caltrans447 88,517 4,856(Std Specs ’25)
CDOT450 17,090 421
Table 7: The three DOT corpora processed by the iden-
tical pipeline.
The mechanism transfers, at the right granular-
ity.On CDOT and the BGE-M3 re-replication of
WYDOT, the small-suffers pattern reproduces un-
der thedocument_seriesscope. Specifically,
we observeρ CDOT=−0.95andρ WYDOT =−0.68
(Table 8, rows 1 and 3). This WYDOT result
closely matches the−0.60reported in §3 under
a different embedder.
However, on Caltrans, this same axis col-
lapses toρ=−0.10(row 5) because its “cate-
gory” comprises only3yearly omnibus PDFs
averaging2,374chunks apiece, compared to37
for CDOT. Inspection (App. A22) reveals these
PDFs are split into∼80topicalSECTIONs. To
account for this, we switch the scope axis to

section, extracting metadata from the chunks’
SECTION/DIVISION/CHAPTERheaders. This re-
stores the correlation on Caltrans toρ=−0.85(Ta-
ble 8, row 7), aligning the results with CDOT and
WYDOT. Ultimately, the mechanism transfers to
all three corpora when measured at the granularity
each producer treats as topical.
Corpus Scope axis #cat purityδ ρ
CDOTdoc_series10 0.921 0.079−0.95
CDOTdoc_series×section342 0.553 0.447−0.90
WYDOTdoc_series10 0.965 0.035−0.68
WYDOTdoc_series×section249 0.770 0.230−0.67
Caltransdoc_series9 0.733 0.267−0.10†
Caltransdoc_series×section841 0.446 0.554−0.81
Caltranssection407 0.631 0.369−0.85
Table 8: Per-corpus dilution under
document_seriesvs. section scope.
Implication:The right scope axis is whichever
organizational unit the corpus’s producer treats
as topical.document_seriessuffices for
Wyoming and Colorado;sectionis required for
California. A multi-tenant deployment cannot as-
sume a uniform scope axis across tenants; we rec-
ommendadaptive scoping, selectable per tenant
by the chunks-per-doc statistic —2,374for Cal-
trans Specs vs.37for CDOT Specs — two orders
of magnitude apart on a cheap, corpus-intrinsic
signal.
10 The Precision–Faithfulness Paradox
and Its Causes
MASDR-RAG improves retrieval (P@10
0.77→0.86) yetdegradesfaithfulness
(0.61→0.35). We test four candidate causes
and report the headline result of each ablation;
full tables are in the appendix.
(1) Routing noise:The production regex routes
only47.1%of WYDOT queries correctly (top-
1,n=155). A BGE-M3 linear-probe (R2)
lifts top-1 to0.755(+28.4points,5-fold CV;
App. A33). Plugged in end-to-end as R2-ROUTED
on WYDOT200-q, R2 attains the highest correct-
ness (0.303, vs.0.218HYBRID-ROUTED,0.274
MASDR-RAG) and Recall@10 (0.375), at∼1
2
MASDR-RAG’s LLM-call budget. But the28-
point routing accuracy ceiling cannot account for
the26-point faithfulness collapse; the paradox is
not routing-bound.
(2) Within-scope ranking noise:A cross-
encoder reranker (bge-reranker-v2-m3, top-30→10) on top of BGE-M3 lifts faithfulness
+0.08/+0.09on Composite-9 but loses−0.05
on WYDOT, where the bi-encoder already or-
ders chunks by section/year/version metadata that
the cross-encoder undoes (App. A34, Tab. 26).
Reranking does not recover MASDR-RAG’s col-
lapse.
(3) Retriever family:Replacing dense BGE-
M3 with sparse SPLADE (OpenSearch Project,
2024) wins on Composite-9 (Corr.900→.940),
ties on MultiHop, and loses on FinanceBench
— there is no systematic dense-vs-sparse winner
across corpora (App. A35, Tab. 27).
(4) Index implementation:Re-running all
five non-WYDOT corpora under both FAISS
IndexFlatIPand a local Neo4j HNSW yields
identical architectural rankings and pairwise
deltas within±0.07absolute (median|∆|=.02;
App. A36, Tab. 28).
(5) Context fragmentation — falsified:To test
if multi-round synthesis fragments evidence, we
compare it to MASDR-SINGLECALL, which
concatenates all retrieved chunks into a single
call. If fragmentation were the issue, MASDR-
SINGLECALLshould recover faithfulness. It does
the opposite: on Composite-9, both faithfulness
and correctness drop0.74→0.62; on WYDOT,
they drop0.391→0.221and0.274→0.151, re-
spectively (Tab. 9). Multi-round orchestrationin-
sulatesthe synthesizer from cross-source confu-
sion; collapsing it amplifies the problem, leaving
residual costs due to imperfect routing and the 7B
synthesizer’s capacity.
Table 9: WYDOT-200Qwen-7B + BGE-M3 + Qwen-
judge. Full table including Composite-9 in App. A37.
SystemR@10 Faith Corr
Monolithic.188.347.216
Regex-Sc..219.296.246
HYBRID-ROUTED.194.340.218
R2-Routed.375.369.303
MASDR-RAG.258.391.274
SingleCall.188.221.151
(6) Cross-backbone sensitivity:We re-ran the
four WYDOT systems with four LLMs. Two pat-
terns split along an open-source vs. commercial
axis (Tab. 10): Qwen-7B and DeepSeek-V3 keep
MASDR-RAG at or above monolithic faithful-
ness (Qwen MASDR Faith.391vs. Mono.347);

Claude-Haiku and GPT-5-mini suffer a sharp col-
lapse (Claude.250→.010; GPT.276→.241).The
production Gemini paradox is therefore real and
reproducible, butconfiguration-dependent(open-
source vs. commercial generator), not an intrinsic
property of multi-agent RAG.
Table 10: Cross-backbone WYDOT-200(same BGE-
M3, same Qwen judge).
Backbone SystemnFaith Corr
Qwen-7BMono199.347.216
MASDR197.391.274
Claude-HaikuMono100.250.240
MASDR100.010.080
GPT-5-mini∗Mono29.378.172
MASDR29.241.414
DeepSeek-V3∗Mono44.222.444
MASDR44.318.523
11 Discussion: Scope vs. Orchestration
The pattern across our settings (Tables 3, 4,5,9,10)
factors into two axes —(i)is the corpus gen-
uinely multi-domain, and(ii)is the answer-
generator open-source or commercial. For single-
organization corpora (WYDOT-like) with stable
scopes,R2-ROUTED(trained BGE-M3 router, sin-
gle synthesis call) attains the highest correctness
and Recall@10 (Table 9) at half MASDR-RAG’s
LLM-call budget; HYBRID-ROUTED(regex +
LLM) is the fallback when a trained router is un-
available. Reserve full MASDR-RAG orchestra-
tion for genuinely multi-domain corporaandan
open-source generator (Qwen-class or DeepSeek-
class): under commercial generators (Claude /
GPT) MASDR-RAG suffers a sharp faithfulness
collapse (Table 10,0.250→0.010for Claude;
0.378→0.241for GPT-5-mini). A ReAct-style
loop only pays off with strong tool-calling back-
bones; on Llama-3-8B, the iteration cost is not
amortized by quality (Table 4). Across all set-
tings, domain-scoped retrieval is the most consis-
tent lever on retrieval precision; the architectural
choice above it mainly concerns how toavoid un-
doingthat precision gain through context fragmen-
tation or backbone-mismatched orchestration.
12 Conclusion
In this paper, we identified and characterized
vector search dilutionin a real-world RAG de-
ployment. We showed that domain scopingover Neo4j organizational metadata reduced most
of the dilution and lifted P@10 from0.77to
0.86. Naive multi-agent orchestration, however,
degraded faithfulness from0.61to0.35— the
precision–faithfulness paradox. We demonstrated
that HYBRID-ROUTEDrouting resolved the para-
dox by combining regex determinism with a single
LLM router and a single scoped answer pass.
Open-source replications with Qwen2.5-7B
and Llama-3-8B preserve the architectural rank-
ing. An apples-to-apples comparison with ReAct
shows that the iterative loop incurs5.5×more
LLM calls and5.9×more tokens on Llama-8B,
resulting in a2.2×slower median response time.
Cross-domain replications on a9-source public
composite corpus and on HotpotQA-distractor in-
dicate that the dilution effect and the HYBRID-
ROUTED-over-MASDR-RAG preference are not
unique to WYDOT; however, the downstream ac-
curacy gain from scoping is largest on WYDOT,
where retrieval quality most directly determines
the answer.
Our broader observation is that, for enterprise-
scale RAG, the load-bearing decision isdomain
scoping. The choice of the above orchestration
is mainly about avoiding the fragmentation of the
context that scoping produced.
Limitations
Our primary finding—the precision–faithfulness
paradox—is established using an LLM-as-judge
framework (Qwen-7B). Because RAGAS-style
metrics anticipate a single context window, they
can inadvertently penalize multi-agent responses.
Thus, the measured drop in MASDR-RAG’s
faithfulness serves as an upper bound. This di-
rectional trend remains robust, however, as con-
firmed by a Llama-3-8B spot-check (n=50) and
cross-backbone replications with Claude and GPT
(Tab. 10). Additionally, our open-source eval-
uations are limited to7–8B-parameter models,
leaving≥70B-parameter architectures untested.
Finally, our routing taxonomy (e.g., the nine
WYDOT scopes or Composite-9 source types) is
manually crafted and assumes the availability of
explicit organizational metadata during corpus in-
gestion.
Ethics Statement
We use public government documents and public
datasets. The system assists with document re-

trieval and does not generate policy recommenda-
tions. AI-assisted writing tools, including Chat-
GPT and Grammarly, were used to improve the
manuscript’s readability and grammatical clarity
without introducing any risks.
Reproducibility
Code, the 200-query WYDOT suite, all6open-
corpus ingest/eval scripts, and SLURM submis-
sion scripts are released at(anonymized
github URL); a one-commandmake
reproducere-runs the full sweep on a sin-
gle L40S. All models are public HuggingFace
checkpoints; commercial backbones use Open-
Router. Full hyperparameters, hardware, and
seeds in App. A39.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil,
and Hannaneh Hajishirzi. 2024. Self-RAG: Learn-
ing to retrieve, generate, and critique through self-
reflection. InICLR.
Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu,
Zach Brannelly, and Mohamed Abdelrazek. 2024.
Seven failure points when engineering a retrieval
augmented generation system. InCAIN.
Harrison Chase. 2023. LangChain.https://
github.com/langchain-ai/langchain.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. BGE
M3-Embedding: Multi-lingual, multi-functionality,
multi-granularity text embeddings.Preprint,
arXiv:2402.03216.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
RAG systems. InSIGIR.
Darren Edge, Ha Trinh, Newman Cheng, and
1 others. 2024. From local to global: A
GraphRAG approach to query-focused summariza-
tion.arXiv:2404.16130.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAS: Automated eval-
uation of retrieval-augmented generation. InEACL
Demos.
Thibault Formal, Carlos Lassance, Benjamin Pi-
wowarski, and Stéphane Clinchant. 2022. From dis-
tillation to hard negative sampling: Making sparse
neural ir models more effective. InSIGIR.Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented
generation for large language models: A survey.
arXiv:2312.10997.
Aaron Grattafiori and 1 others. 2024. The Llama 3 herd
of models.Preprint, arXiv:2407.21783.
Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xi-
aolong Jin, Jiafeng Guo, and Xueqi Cheng. 2025.
RouteRAG: Efficient retrieval-augmented genera-
tion from text and graph via reinforcement learning.
arXiv preprint arXiv:2512.09487.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa-
supat, and Ming-Wei Chang. 2020. REALM:
Retrieval-augmented language model pre-training.
InICML.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-RAG:
Learning to adapt retrieval-augmented large
language models through question complexity.
NAACL.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O.
Arik. 2025. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG. InICLR.
Jeff Johnson, Matthijs Douze, and Herve Jegou. 2021.
Billion-scale similarity search with GPUs.IEEE
Trans. Big Data.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. InEMNLP.
Omar Khattab and Matei Zaharia. 2020. ColBERT: Ef-
ficient and effective passage search via contextual-
ized late interaction over BERT. InSIGIR.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, and 1 others. 2019. Natural questions: A
benchmark for question answering research. In
TACL.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InNeurIPS.
Longkun Li, Yuanben Zou, Jinghan Wu, Yuqing Wen,
Jing Li, Hangwei Qian, and Ivor Tsang. 2026.
SCOUT-RAG: Scalable and cost-efficient unifying
traversal for agentic graph-RAG over distributed do-
mains.arXiv preprint arXiv:2602.08400.
Yu A. Malkov and Dmitry A. Yashunin. 2020. Efficient
and robust approximate nearest neighbor search us-
ing hierarchical navigable small world graphs.IEEE
TPAMI.

T. Nguyen and 1 others. 2025. MA-RAG: Multi-agent
retrieval-augmented generation.arXiv preprint.
Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and
Jimmy Lin. 2020. Document ranking with a pre-
trained sequence-to-sequence model. InFindings of
EMNLP.
OpenSearch Project. 2024. Neural sparse encoding –
OpenSearch v2.https://opensearch.org/
blog/neural-sparse-v2/.
Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022.
TALM: Tool augmented language models.Preprint,
arXiv:2205.12255.
M. Poliakov and 1 others. 2024. Multi-meta-RAG:
Metadata-filtering retrieval-augmented generation.
arXiv preprint.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of EMNLP.
Qwen Team. 2024. Qwen2.5: A party of founda-
tion models.https://qwenlm.github.io/
blog/qwen2.5/.
Nils Reimers and Iryna Gurevych. 2021. The curse
of dense low-dimensional information retrieval for
large index sizes. InACL-IJCNLP (Short Papers).
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. 2022. Col-
BERTv2: Effective and efficient retrieval via
lightweight late interaction. InNAACL.
Kunal Sawarkar, Abhilasha Mangal, and Sanmitra
Solanki. 2024. Blended RAG: Improving RAG ac-
curacy with semantic search and hybrid query-based
retrievers.arXiv:2404.07220.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi,
Roberta Raileanu, Maria Lomeli, Eric Hambro,
Luke Zettlemoyer, Nicola Cancedda, and Thomas
Scialom. 2023. Toolformer: Language models can
teach themselves to use tools. InNeurIPS.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context.ICML.
S. Singh and 1 others. 2025. Agentic retrieval-
augmented generation: A survey.arXiv preprint.
Yixuan Tang and Yi Yang. 2024. MultiHop-RAG:
Benchmarking retrieval-augmented generation for
multi-hop queries.Preprint, arXiv:2401.15391.Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. InACL.
Chengke Wu, Wenjun Ding, Qisen Jin, Junjie Jiang,
Rui Jiang, Qinge Xiao, Longhui Liao, and Xiao Li.
2025. Retrieval augmented generation-driven infor-
mation retrieval and question answering in construc-
tion management.Advanced Engineering Informat-
ics.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop question
answering. InEMNLP.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in lan-
guage models. InICLR.
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding,
Daniel Fleischer, Peter Izsak, Moshe Wasserblat,
and Danqi Chen. 2025. HELMET: How to evaluate
long-context language models effectively and thor-
oughly. InICLR.
Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu,
Fan Wu, and Guihai Chen. 2025. RAGRouter:
Learning to route queries to multiple retrieval-
augmented language models.arXiv preprint
arXiv:2505.23052.
A1 Implementation Details
Knowledge Graph:Documents were parsed
with PyPDF2 and pdfplumber, chunked via recur-
sive token splitting (1,000-token windows with
200-token overlap), embedded with Gemini Em-
bedding (768-dim) for the proprietary stack and
BGE-M3 (1024-dim) for the OSS stack, and in-
gested into Neo4j with a Document→Section→
Chunk hierarchy. The graph contains152,231
nodes and338,569relationships and is hosted on
Neo4j AuraDB.
Agents:Nine domain agents plus one general
agent inherit from a commonBaseAgent, each
defining a Neo4jdocument_seriesfilter
and exposingsearch(),get_section(),
andcompare_versions()tool meth-
ods.GeneralAgentsearches the full un-
scoped index. All agents are registered in an
AGENT_REGISTRYdictionary for dynamic
dispatch.

Hybrid Search:Within each agent scope, we
execute vector search (HNSW cosine similarity,
top-k) and full-text search (Neo4j BM25 key-
word matching, top-k) in parallel via Cypher, then
merge the results with priority deduplication: vec-
tor results are preferred when the same chunk ap-
pears in both result sets.
Deployment:The production system runs as
a Chainlit web application with two interfaces
(single-agent with LLM routing and full multi-
agent), deployed on Google Cloud Run (2vCPUs,
2GiB RAM). The OSS reproduction runs entirely
on the ARCC SLURM cluster, using either Neo4j
(Gemini stack) or a local FAISS index (OSS stack
and cross-domain corpora) to ensure parity.
A2 Open-Source Backbone
Configuration
ModelsQwen2.5-7B-Instruct uses the na-
tive tool-calling chat template, which emits
<tool_call>{...}</tool_call>blocks
that we parse back to the orchestrator. Llama-3-
8B-Instruct does not have a first-class tool-calling
chat template; we use the same Hermes-style
<tool_call>convention and inject the tool
catalog into the system prompt.
Embedder:BAAI/bge-m3 (1024-d, multilin-
gual,8k context). Vectors are L2-normalized,
so the Neo4j cosine index reduces to the dot
product. For the local FAISS path, we use an
IndexFlatIP, so retrieval is exact (not approx-
imate); this isolates dilution from ANN effects.
Hardware:mb-l40s(1NVIDIA L40S,
48GB),mb-a30(1NVIDIA A30,24GB).
Memory footprint at bf16: Qwen-7B≈14GB,
Llama-8B≈16GB, BGE-M3≈2GB. Per-query
wall time on L40S averages6–8s for non-
orchestrated systems and10–25s for orchestrated
ones.
Sampling and decoding:Routing uses tempera-
ture0.0with top-p= 1.0; answer generation uses
temperature0.2with top-p= 0.95and a1024-
token cap. All runs use a fixed seed (42) for repro-
ducibility.
A3 ReAct Baseline Details
We implement ReAct as an
Action-Observationloop with a maxi-
mum of6rounds. Each round, the LLM is shownthe current scratchpad and must emit either a
Tool/Argspair (parsed by the same regex as
the orchestrator) or a final answer. If the round
budget is reached, the scratchpad is collapsed, and
a final synthesis call is forced. The tool catalog
is identical to MASDR-RAG’s (nine retrieval
tools), so any difference in performance reflects
control flowrather than retrieval quality.
On Llama-3-8B, ReAct’s mean iteration count
is5.50(Table 4); on Qwen-7B it is2.26. We
attribute the gap to the native function-call tem-
plate, which lets Qwen commit to a tool call deci-
sively rather than “reasoning around” it. We also
note that LangChain’screate_react_agent,
evaluated separately (Table 22), reproduces the
same qualitative pattern.
A4 LangChain Wiring
The LangChain baseline wraps our nine retrieval
tools asStructuredToolinstances and feeds
them tocreate_react_agenttogether with
a Qwen-7B-backedChatHuggingFaceLLM.
The Qwen wrapper exposes the model via the stan-
dard LangChainLLMinterface; we use the same
chat template and decoding parameters as in the
custom runner, so any correctness gap reflects only
the orchestration framework. A minimal wiring
snippet:
fromlangchain.agentsimportcreate_react_agent
fromlangchain.toolsimportStructuredTool
def_make_tool_fn(backend, agent_name):
def_fn(query):
ifagent_name == "general_agent":
chunks = backend.global_search(query)
else:
chunks = backend.combined_scoped_search(
query, agent_name)
returnformat_chunks(chunks[:10])
return_fn
tools = [StructuredTool.from_function(
_make_tool_fn(backend, name),
name=name, description=desc)
forname, descinAGENT_DESCRIPTIONS.items()]
agent = create_react_agent(qwen_llm, tools, prompt)
A5 ColBERTv2 Indexing Details
We use thecolbert-ir/colbertv2.0
checkpoint via the RAGatouille library. To avoid
contacting HuggingFace on offline compute
nodes, we adjust the local model snapshot to load
from disk without network access. PLAID in-
dexing runs on a single L40S GPU; the WYDOT
corpus indexes in≈4min, the composite corpus
in≈5min, and HotpotQA in≈8–10min for
≈74k passages. ColBERTv2 returns top-10

chunks, which are fed verbatim to Qwen-7B for
the “+ Qwen” variant.
Per-scope ColBERT:The same PLAID in-
dex can be queried with a post-filter on the
source_typefield of the meta parquet,
yielding a per-scope late-interaction retriever. On
COMPOSITE-9 this “ColBERTv2 scoped + Qwen”
setup attains Correctness= 0.840and
Faithfulness= 0.860(n=50), versus0.820/0.780
for the unscoped ColBERTv2 + Qwen
(∆F=+0.08,∆ C=+0.02, same∼13chunks).
The direction matches the dilution effect we
observe on BGE-M3 (§3): scoping reduces
cross-source contamination at retrieval time
for late-interaction as well as for single-vector
encoders, and the synthesizer converts the cleaner
context into better faithfulness more than into
better correctness — consistent with the read of
§10 that the correctness ceiling is set by what the
corpus contains, while faithfulness is set by what
the LLM has to ignore.
A6 Embedding Space Visualization
Protocol
Figure 4 visualizes vector-search dilution directly
in the WYDOT embedding space, projecting the
88,907-chunk corpus to two dimensions.
(a) WYDOT monolithic search space
Standard Specs
Construction Manual
Materials TestingDesign Manual
Traffic & Crashes
Bridge ProgramSTIP
Annual Reports
Highway Safety
(b) WYDOT scoped: Standard Specs agent only
in scope (Standard Specs) out of scope (filtered)
Figure 4: Embedding-space view of dilution on
WYDOT (88,907chunks; t-SNE projection of≤800
chunks per category,9categories.) (a) Monolithic:
large categories occupy dense central regions, small
ones disperse at boundaries. (b) Standard-Specs scope
active: The neighborhood collapses to a single coher-
ent region. The Composite-9 replication is in Figure 6.
Figure 6 is produced from the1,024-
dimensional BGE-M3 chunk embeddings of
the EnterpriseComposite-9 corpus (n=11,312)
and the aligned metadata that records each
chunk’s source type. Figure 4 is for the WYDOT
corpus: low-density categories sit at the bound-
aries of the high-density clouds, so a globalnearest-neighbor search is biased toward the
dense neighbors and tends to under-recall the
sparse category; activating the source-label filter
forces the retrieval neighborhood to respect the
organizational metadata already present in the
document graph.
To keep the projection legible and the t-
SNE cost bounded, we draw a class-balanced
sample with a per-source cap of1,200chunks
(the smallest source,docs, contributes its
full1,005). The remaining5,805vectors are
first PCA-reduced to50dimensions and then
projected with scikit-learn’s t-SNE under the
following configuration:perplexity= 35,
max_iter= 1,500,metric=“cosine”,
init=“pca”,learning_rate=“auto”,
andrandom_state= 0.
Panel (a) plots every sampled chunk colored
by its source. Panel (b) re-uses the same co-
ordinates: chunks whosesource_typediffers
from the highlighted agent (StackOverflowin the
figure) are drawn in light gray at10%opacity
to indicate “filtered out by the metadata where-
clause”, and the in-scope chunks are drawn at full
opacity in their source color. The visualization
is intended to be read jointly with Table 2: The
density-driven dilution effect we measure quantita-
tively is the same effect that geometrically scatters
low-density sources through high-density neigh-
borhoods in panel (a).
Dilution-vs-scale curve (Figure 1).The points
are taken directly from Table 2; we regressδ=
mlog10(chunks)+band report the Spearman cor-
relation, which also appears in §3. The fit slope
m=−0.19summarizes the per-decade decrease in
δas a category becomes denser.
Retrieval-source confusion (Figure 5).
Built from the judged evaluation log
of the EnterpriseComposite-9 run. For
each systemS∈ {MONOLITHIC,
REGEX-SCOPED,HYBRID-ROUTED}we it-
erate over alln=153queries, record the
chunk_sourceslist (k=15chunks per query),
and estimateP(retrieved source|gold source)by
row-normalizing the resulting count matrix per
gold source. The trace divided by the number
of sources is reported below each panel as the
“diagonal average”.
Failure-case panel (Appendix A26):Drawn di-
rectly with TikZ; corresponds to Case2in Ap-

HelpdeskConfluence
StackOverflowEmailDocs
Retrieved chunk sourceHelpdesk
Confluence
StackOverflow
Email
DocsQuery gold source.95 .05 .01
.17 .67 .05 .08 .03
.01 .01 .95 .02
.09 .02 .89
.33 .22 .13 .19 .13
diag avg = 0.72(a) Monolithic
HelpdeskConfluence
StackOverflowEmailDocs
Retrieved chunk sourceHelpdesk
Confluence
StackOverflow
Email
Docs.95 .05 .01
.17 .67 .05 .08 .03
.01 .01 .95 .02
.09 .02 .89
.33 .22 .13 .19 .13
diag avg = 0.72(b) Regex-Scoped
HelpdeskConfluence
StackOverflowEmailDocs
Retrieved chunk sourceHelpdesk
Confluence
StackOverflow
Email
Docs.93 .06 .01
.17 .67 .05 .08 .03
.02 .02 .96
.16 .03 .81
.23 .34 .21 .08 .13
diag avg = 0.70(c) Hybrid-Routed
0.00.20.40.60.81.0
P(retrieved source ∣ gold source)Figure 5:P(retrieved source|gold source)on Composite-9 (Qwen-2.5-7B, BGE-M3, top-15). Monolithic diago-
nal0.59; Regex-Scoped0.84; HYBRID-ROUTED0.90.
pendix A14. The chunk titles shown illustrate
the document family that each system retrieves
from the production WYDOT corpus and are in-
tended to make the dilution failure mode legible to
a reader who has not seen the underlying chunks.
A7 Composite Corpus Assembly
EnterpriseComposite-9 is assembled from
nine public HuggingFace datasets. For each
source, we cap at the first5,000documents
to keep the corpus tractable while preserving
heterogeneity. Source-type metadata is pre-
served on every chunk as the routing label
(replacing WYDOT’sdocument_series).
Query generation uses Qwen2.5-7B with the
prompt template “Given the following
{source_type} passage, write a
question whose answer is grounded
in this passage and which a user
of {source_type} would plausibly
ask.” Each generated query is paired with
the gold passage’s chunk ID, we evaluate per-
source-labeled correctness against this ground
truth.
A8 Routing Prompt
You are a query router for WYDOT.
Classify into ONE category:
- STANDARD_SPECS: Construction specs,
materials requirements, methods
- CONSTRUCTION_MANUAL: Field inspection,
project administration
- MATERIALS_TESTING: Lab procedures, QC
- DESIGN_MANUAL: Road/bridge design
- TRAFFIC_CRASHES: Crash statistics
- STIP: Project funding, improvement programs
- ANNUAL_REPORT: Department reports
- BRIDGE_PROGRAM: Bridge plans, ratings
- HIGHWAY_SAFETY: Safety programs, SHSP
- GENERAL: Cross-domain or unclear.Format:
CATEGORY: <category>
YEAR: <year or NONE>
SERIES: <series or NONE>
Query: {query}
A9 Answer-Synthesis System Prompt
You are the WYDOT Knowledge Graph Assistant.
Answer questions about the Wyoming Department of
Transportation using a knowledge graph with
1,128 documents and 88,907 text chunks.
IMPORTANT: You must ALWAYS call at least one
search tool for EVERY query. Never refuse a
query without searching first.
For each user query:
1. DECIDE which tool(s) to call based on topic.
2. CALL the appropriate tool(s) with a clear
search query.
3. READ the returned document chunks carefully.
4. SYNTHESIZE a comprehensive answer with
citations.
CITATION RULES:
- Reference sources as [Source 1], [Source 2].
- Include document title, section, and year.
- If information comes from multiple sources,
cite all of them.
MULTI-STEP REASONING:
- For comparison queries, call compare_versions
or call the same tool with different years.
- For cross-domain queries, call multiple tools.
- You can make up to 5 tool calls per query.
ANSWER FORMAT:
- Use markdown with headers, bullet points,
and tables where appropriate.
- Be thorough but concise.
- Always ground answers in retrieved content.
A10 LLM-as-Judge Prompt
We use Qwen2.5-7B-Instruct as the judge for both
correctness and RAGAS-style faithfulness. Each
judged record contains the question, the gold ref-
erence answer (where available), the model’s an-
swer, and the retrieved chunks; the judge returns

a structured JSON object with binary correctness
and a0–1faithfulness score.
You are an impartial evaluator.
Given QUESTION, REFERENCE (may be empty),
ANSWER, and CONTEXT chunks, output JSON:
{
"correct": 0 or 1,
"faith": 0..1 (RAGAS-style),
"relev": 0..1,
"rationale": short string
}
Correctness: 1 only if the ANSWER addresses
the QUESTION using facts that the REFERENCE
or CONTEXT supports.
Faithfulness: fraction of the ANSWER’s
verifiable claims that are entailed by the
CONTEXT chunks.
A11 Agent Scoped Filters
Table 11 lists the per-agentdocument_series
regex filters that implement metadata scoping.
Agent Filter (regex on document_series)
specs(?i). *standard.spec. *
construction(?i). *construction.manual. *
materials(?i). *materials. *test. *
design(?i). *design.manual. *
safety(?i). *(crash|safety|fatal). *
bridge(?i). *bridge. *(prog|plan). *
planning(?i). *(stip|improv.prog). *
admin(?i). *(annual.rep|strateg). *
general (no filter — full corpus)
Table 11: Agentdocument_seriesregex filters ap-
plied as CypherWHEREclauses.
Category Scope
STANDARD_SPECS Construction specifica-
tions
CONSTRUCTION_MANUAL Field inspection proce-
dures
MATERIALS_TESTING Lab test procedures
DESIGN_MANUAL Road/bridge design stan-
dards
TRAFFIC_CRASHES Crash statistics and anal-
ysis
STIP Project funding and plan-
ning
ANNUAL_REPORT Department reports
BRIDGE_PROGRAM Bridge plans and ratings
HIGHWAY_SAFETY Safety programs
GENERAL Cross-domain or unclear
Table 12: Category taxonomy aligned with WYDOT’s
organizational structure.Agent Scope Docs Reduction
Specs 3,140 22 96.5%
Construction 6,641 21 92.5%
Materials 2,184 7 97.5%
Design 1,366 21 98.5%
Safety 30,922 22 65.2%
Bridge 8,076 45 90.9%
Planning 13,607 58 84.7%
Admin 2,439 47 97.3%
General 88,907 1,128 0%
Wtd. Avg.— —90.4%
Table 13: Per-agent search-space reduction under
document_seriesscoping. Referenced from §4.
Metric Pilot (n=57) Full (n=200)∆
Monolithic P@10 0.66 0.77+0.11
Scoped P@10 0.78 0.86+0.08
MASDR-RAG Faith. 0.32 0.35+0.03
HYBRID-ROUTEDFaith. 0.58 0.62+0.04
Table 14: Stability of core metrics across evaluation
scales.
System vs. Mono.p(P@10)p(Faith.)p(Corr.)
Mono+RRF 0.511 0.304 0.821
LLM+Scoped0.0380.973 0.809
MASDR-RAG0.017<0.0010.102
HYBRID-ROUTED0.107 0.901 0.912
HYBRID-ROUTEDvs. MASDR-RAG 0.323<0.0010.787
Table 15: Permutation testp-values (10,000permuta-
tions) on the Gemini stack. Bold:p <0.05.
A12 Category Taxonomy
Table 12 gives the full category taxonomy aligned
with WYDOT’s organizational structure.
A13 Scale Stability Data
Table 13 reports the per-agent search-space reduc-
tion that metadata scoping achieves across evalua-
tion scales. To check that our 200-query results are
not an artifact of suite size, we compare the core
metrics on the original pilot suite (n=57) against
the fulln=200suite. All metrics shift by at most
±0.11between the two scales, preserving relative
ordering as mentioned in Table 14.

Query (truncated) Target System P@10 Correct Faith. Relev.
Single-domain queries (113total, representative sample):
Construction Limits definition STD_SPECS HYBRID-ROUTED0.30✓0.20 1.00
MASDR-RAG 0.00✓0.00 1.00
Temporary stream crossing 404 permit STD_SPECS HYBRID-ROUTED0.50✓1.00 1.00
MASDR-RAG 1.00✓0.50 1.00
Safety at active crusher site CONSTR HYBRID-ROUTED1.00✓0.90 1.00
MASDR-RAG 1.00✓0.50 1.00
Cross-domain queries (31total, representative sample):
Bridge design vs. construction DESIGN+CONSTR MASDR-RAG 1.00✓0.40 1.00
Safety improvements in STIP SAFETY+STIP MASDR-RAG 1.00×0.20 1.00
Version comparison (27total, representative sample):
Aggregate gradation2010vs2021STD_SPECS MASDR-RAG 1.00✓0.50 1.00
Table 16: Representative per-query results for MASDR-RAG and HYBRID-ROUTEDon the Gemini stack. Full
200-query results for all five systems are released alongside the code.
A14 Extended Case Studies
Case 1: Version Comparison (MASDR-
RAG advantage):Query: “What
changed in aggregate gradation between
2010and2021?” MASDR-RAG calls
compare_versions(topic=“aggregate
gradation”,
year_old=2010, year_new=2021), re-
trieving Section 703 from both editions and
producing a structured comparison. This is the
primary use case where multi-agent orchestration
adds clear value over single-agent scoped search.
Case 2: Router Failure Mode:Query: “What
are the safety considerations for crusher site in-
spections?” The LLM router classifies it as
HIGHWAY_SAFETY(incorrect; correct category
isCONSTRUCTION_MANUAL). MASDR-RAG
searches30,922safety/crash chunks and returns
highway safety plan documents instead of the Con-
struction Manual’s crusher inspection procedures.
HYBRID-ROUTEDwith regex matching (“inspec-
tion”→CONSTRUCTION_MANUAL) avoids this
error.
Case 3: Context Fragmentation under Qwen-
7B:Query: “What load posting policy applies
to a50-year-old timber bridge?” MASDR-RAG
with Qwen-7B callsbridge.searchtwice
(once for policy, once for timber-specific guid-
ance) and onegeneral.searchcall. The three
returned chunks are individually relevant, but the
final answer omits the load-posting trigger thresh-
old from chunk2because chunk3’s general mate-
rial language drowns it out in the answer prompt
— a concrete instance of context fragmentation.A15 Statistical Test Details
Table 15 reports the permutation-testp-values
(10,000permutations) underlying the significance
claims in the main text.
A16 Full Per-Query Examples
Table 16 shows representative per-query results
for MASDR-RAG and HYBRID-ROUTEDon
the Gemini stack, spanning single-domain, cross-
domain, and version-comparison queries. The full
200-query records for all five systems are released
with the code.
A17 Density-Debiased Dilution
Regression
We expand the per-query dilution analysis of
§3 beyond the category-aggregaten=8Spear-
man. Each of then=147queries, with both
monolithicand scoped variants judged under
our Qwen + chunk-DB stack, contributes∆ q=
Corr scoped(q)−Corr global(q)∈ {−1,0,+1}. We
regress∆ qonlog10NcwhereN cis the category’s
chunk population:
∆q=β 0+β 1log10Nc+εq.
On the open-source BGE-M3 + Qwen + Qwen-
judge stack, ˆβ1=−0.217(SE0.075,r=−0.236,
p=0.004); on the production Gemini stack, the
same regression gives ˆβ1=−0.159(p=0.089).
Both fits put the scoping-benefit direction the same
way as the original category-aggregate Spearman
(−0.60) but at per-query resolution and, on BGE-
M3, withp <0.005.
A18 LLM-as-Judge Rubric
Judges return integer scores under a single-call
prompt that supplies (query, reference answer, re-

trieved chunks, model answer). The full prompt is
in App. A10. The criteria are:
•Correctness (0/1).1if the model answer con-
veys the same factual content as the reference
answer. Acceptable variations: paraphrase, ad-
ditional non-contradictory details. Disqualify-
ing factors: contradicting the reference, missing
the key quantitative claim, or refusing to answer
when a reference exists.
•Faithfulness (0/1).1if every factual claim
in the model answer can be entailed from at
least one of the retrieved chunks supplied in-
prompt. Generic statements (“The Department
issues permits.”) do not need direct support;
specific numbers, section IDs, and year-tagged
statements do.
A19 Query Validation Protocol
The 200-query WYDOT suite was assembled in
three passes:
1.Seed:Two of the authors drafted candidate
queries by scanning each of the nine WYDOT
corpus sections (Standard Specs, Construction
Manual, Materials Testing, Design Manual,
Crash Data, Bridge Program, STIP, Annual Re-
ports, Highway Safety) and recording realistic
operator questions paired with a reference an-
swer and a gold document title.
2.Filter:A separate author (not involved in draft-
ing) reviewed each query for (i) ambiguity, (ii)
presence of a deterministic reference answer
in the corpus, and (iii) coverage balance —
both single-domain (113), cross-domain (31),
section-lookup (22), version-comparison (27),
and ambiguous (7) types are intentionally rep-
resented.
3.Adversarial pass:A final pass added queries
known to fail under the production system at
the time of drafting (e.g. `‘2020 construction
manual” vs. the more abundant 2021 corpus),
to guard against systems that win on the easy
slice only.
Inter-author agreement on filter decisions was
89%(Cohen’sκ=0.74) on a40-query sample.
A20 Scope Granularity Guidance
When deploying HYBRID-ROUTED-style scoping
on a new corpus, we recommend the following:
•Scope on metadata that already exists:Our
WYDOT scopes (9) and COMPOSITE-9 scopes(9) reflect the graph’s source-type field as in-
gested. Inventing scopes that require re-labeling
chunks is a separate engineering project and not
part of the dilution argument.
•Target∼3–10 scopes for the LLM router:
Above∼10, the router accuracy in App. A33
starts to fragment along near-synonymous scope
boundaries. Below∼3, the dilution-mitigation
benefit is too small to clear the multi-call cost.
•Always keep ageneralfallback agent:
Routing failures (e.g., cross-domain queries)
need an unscoped escape hatch; otherwise,
scoped retrieval becomes lossy on the∼16%
of queries that don’t fit any single scope.
•Use BGE-M3 + LR over regex:Per App. A33,
the BGE linear-probe router is+28.4points ab-
solute over the regex router on the same labeled
subset, at negligible additional runtime cost (∼
10ms per query).
A21 GraphRAG Comparison Sketch
A natural question is whether a GraphRAG-style
summarization pre-pass (Edge et al., 2024) could
subsume the dilution-mitigation benefit of meta-
data scoping. We do not run a full GraphRAG
pipeline here because (i) the WYDOT graph
already carries the source-type metadata that
GraphRAG would re-discover, making the over-
lap large by construction, and (ii) GraphRAG’s
community summaries are written into the index,
which would require writes against the production
Neo4j store, and our deployment policy restricts
production writes. We leave a full GraphRAG ab-
lation against the unified-FAISS WYDOT index
(App. A24) to future work.
A22 Cross-DOT Replication Details
This appendix supports §9 with the scrape method-
ology, the per-agent search-space reduction for
each DOT, and the Caltrans Standard Specs sec-
tion breakdown.
Scrape methodology:A single-threaded
crawler fetches each DOT site with a1.0s
sleep between HTML page fetches and1.5s
between PDF downloads, sends a descriptive
user-agent string with a contact email, and honors
each site’srobots.txt(CDOT permits all;
dot.ca.govhas none). Crawl depth is4within
the DOT’s own domain, capped at800HTML

pages and450PDFs per DOT (over-fetched,
then sampled to the proof-of-concept). PDFs are
downloaded from any host the crawler discovers,
deduplicated by content hash, and recorded in
a per-DOT manifest with URL, source-page,
and a provisionaldocument_seriesfrom a
URL/filename keyword classifier. The provisional
class is refined at ingest time by a Qwen-2.5-7B
content classifier; un-classified residuals are
taggedGeneral. ARCC compute nodes have no
outbound network access, so scraping runs on a
login node.
Embedding and chunk-
ing:All three corpora use
RecursiveCharacterTextSplitter(chunk_size=1000
chars, overlap=100). For WYDOT, this required
reconstructing per-document text from the orig-
inal SemanticChunker chunks (sorted byseq
withinsource) and re-splitting; the original
WYDOT chunk count was88,907(Table 1 of
§3) and rises to217,752under uniform chunking.
BGE-M3 inference: bf16 on NVIDIA A30,
batch128, max sequence length512tokens;
L2-normalised1024-d vectors.
Caltrans is the omnibus mega-document cor-
pus:The Caltrans Standard Specs “category” is
3PDF — the2023,2024, and2025yearly editions
of one omnibus specification,≈4,700chunks
each — and the Construction Manual category
is one PDF of3,645chunks. Each internally
spans every engineering topic that the WYDOT
and CDOT taxonomies keep apart, but split into
∼80numbered Sections (SECTION39 Asphalt
Concrete,SECTION90 Portland Cement Concrete,
SECTION96 Bridge Construction,etc.) that the
source documents themselves treat as the topical
unit. CDOT, by contrast, expresses the same con-
tent as20small focused spec documents averaging
37chunks each, and WYDOT splits its specs into
7docs of≈570chunks each.
Intra-document section coherence on Caltrans
Specs:Restricted to Caltrans Standard Specs
chunks alone (n=14,244,100distinct sections),
section-levelρ=−0.79:SECTION96 (1,202
chunks) sits atδ= 0.03and the smallest sections
(n<100) atδ≥0.20— the small-suffers pattern
of §3 holdsintra-document. Applying the com-
posite scopedoc_series×sectionto CDOT
and WYDOT preserves their pattern (ρ=−0.90
and−0.67, Table 8 rows 2 and 4 of §9). The mech-anism is therefore not only present in all three cor-
pora but also operates at whichever organizational
level the corpus’s producer uses — categories of
separate documents (WYDOT, CDOT) or sections
within a single document (Caltrans).
Per-agent search-space reduction:Tables 17,
18, and 19 mirror Table 13 of §4, ap-
plied independently to each DOT corpus under
document_seriesscope. Caltrans’ Standard
Specs agent gets the same six PDFs of14,244
chunks; sub-scoping it bySECTION(Table 20) is
what recovers the small-suffers pattern in §9.
Agent Series #docs #chunks Reduction
plans Standard Plans35 33,934 61.7%
— (General, no scope)342 26,561 0.0%
specs Standard Specs6 14,244 83.9%
design Design Manual42 7,656 91.4%
construction Construction Manual1 3,645 95.9%
planning STIP7 1,376 98.4%
safety Traffic & Safety6 410 99.5%
bridge Bridge Program4 391 99.6%
materials Materials Testing4 300 99.7%
Table 17: Caltrans per-agent reduction under
document_seriesscope (n total=88,517). The
Standard Specs agent’s six documents produce14,244
chunks because the source PDFs are omnibus yearly
editions of one specification.
Agent Series #docs #chunks Reduction
— (General, no scope)237 10,811 0.0%
safety Traffic & Safety70 2,628 84.6%
planning STIP20 1,825 89.3%
specs Standard Specs20 749 95.6%
plans Standard Plans75 695 95.9%
construction Construction Manual15 110 99.4%
materials Materials Testing4 99 99.4%
bridge Bridge Program3 94 99.4%
admin Annual Reports2 44 99.7%
design Design Manual4 35 99.8%
Table 18: CDOT per-agent reduction under
document_seriesscope (n total=17,090).
Caltrans Specs sub-scoped by section.If the
Caltrans Specs agent is sub-scoped further by the
section header, its14,244chunks decompose as
in Table 20 (top eight sections by chunk count).
Each section is a topically coherent unit;SECTION
96 (Bridge Construction) at1,202chunks is the
largest and matches a typical WYDOT mid-size
category in scale.

Agent Series #docs #chunks Reduction
safety Traffic & Safety75 66,566 69.4%
— (General, no scope)813 65,278 0.0%
planning STIP61 44,844 79.4%
construction Construction Manual21 9,946 95.4%
bridge Bridge Program43 8,509 96.1%
admin Annual Reports47 7,632 96.5%
materials Materials Testing13 7,570 96.5%
specs Standard Specs7 3,995 98.2%
plans Standard Plans20 2,500 98.9%
design Design Manual28 912 99.6%
Table 19: WYDOT per-agent reduction underuniform
chunking (n total=217,752). For continuity with §3,
WYDOT here is re-chunked to the same1000-char con-
vention as Caltrans and CDOT, so the cross-DOT com-
parison is apples-to-apples; chunk counts therefore dif-
fer from Tables 1 and 13 of the main paper.
Section #chunks Reduction vs. Caltrans
SECTION961,202 98.6%
SECTION90745 99.2%
SECTION39668 99.2%
SECTION51630 99.3%
SECTION37494 99.4%
SECTION12484 99.5%
SECTION20454 99.5%
SECTION13440 99.5%
Table 20: Top eight Caltrans Standard Specs sections
by chunk count. Sub-scoping the Specs agent by sec-
tion reduces its effective search space from14,244
chunks to at most∼1,200chunks (98.6–99.5% reduc-
tion over the full Caltrans corpus).
A23 Local Neo4j Sandbox
To run the FAISS-vs-Neo4j infrastructure-parity
comparison (App. A36) without modifying the
production AuraDB deployment, we install Neo4j
Community5.26together with a private JDK17
tarball into the cluster’s user space. The daemon
runs as a7-day SLURM reservation on the Teton
CPUs partition (24GB heap,8CPUs); its bolt end-
point is advertised via a small text file in the clus-
ter user space that the evaluation harness loads at
start-up.
A generic ingest step takes any of our five
embedding/metadata artifact pairs and creates a
corpus-scoped node label, a vector index with co-
sine similarity, and a full-text index over the chunk
text. Batch inserts use200rows per transaction;
bulk ingest of NQ (200k chunks) takes≈13
minutes. The resulting daemon is wrapped be-
hind the same backend interface as the runners
already used for the FAISS path, so every bench-mark can switch to the Neo4j backend without fur-
ther changes.
A24 Unified-FAISS WYDOT
To rule out a confound where the production of the
Neo4j vector index itself contributes to the dilution
pattern (different index implementations, different
distance metrics, and different recall characteris-
tics than a textbook flat-IP index), we built a fully
local FAISS-only WYDOT index. The pipeline:
1. Dump every WYDOT chunk’s id, text, source,
year, section, and document ID via a single
read-only query against the production Neo4j
store (no writes).
2. Re-embed each chunk with BGE-M3 on an
L40S GPU.
3.L 2-normalize the vectors and store them as a
local embedding and metadata artifact pair.
4. Wire the artifact into the existing local search
backend so every WYDOT evaluation run can
choose FAISS or Neo4j with a single command-
line flag.
The resulting FAISS store contains93,879chunks
(covering every section node in the production
graph and a small set of orphan chunks that
the Neo4j vector index lazily includes) and sup-
ports the same monolithic / regex_scoped / hy-
brid_routed / MASDR-RAG systems.
Storage footprint:367MB embeddings +
76MB metadata.
Unified-FAISS WYDOT results (n=197):
Replicating the three single-call systems against
the FAISS-only WYDOT store (Qwen-2.5-7B
synthesizer + Qwen judge) yields Monolithic
Faith.396/ Corr.360, Regex-Scoped.396/
.350, and Hybrid-Routed.396/.340— all
three systems are within±0.02of their Neo4j
counterparts in Table 9, confirming that the
production Neo4j HNSW is not a confound for
the WYDOT block. The unified-FAISS build
is fully self-contained (no Neo4j dependency at
query time) and is the path we recommend for
downstream reproductions.
A25 Llama-3-8B Cross-Corpus
Replication
We replicate the four cross-domain corpora of
Table 5 under a smaller open-source backbone
(meta-llama/Llama-3-8B-Instruct) to

test whether the architectural rankings hold below
the7B Qwen tool-call threshold. The retriever
(BGE-M3), judge (Qwen-2.5-7B with passages in-
prompt), and sample sizes match the Qwen-side
runs.
Table 21: Llama-3-8B cross-corpus replication. Same
retriever, scope filter, and judge as the Qwen-side Ta-
ble 5; only the synthesizer differs.
Corpus SystemnFaith Corr
MultiHopMono500.410.556
Regex-Sc.500.420.500
Hybrid-R.500.392.486
MASDR500.130.420
FinanceB.Mono150.200.233
Regex-Sc.150.173.207
Hybrid-R.150.207.233
MASDR150.000.000
MMLU-ProMono500.210.268
Regex-Sc.500.150.260
Hybrid-R.500.162.256
MASDR500.096.188
NQ-OpenMono500.350.240
Regex-Sc.500.334.260
Hybrid-R.500.336.250
MASDR500.094.360
The qualitative pattern matches the cross-
backbone story in Table 10: at the8B scale,
MASDR-RAG suffers a severe faithfulness col-
lapse (FinanceBench.207→.000; MMLU-Pro
.162→.096), and on FinanceBench, it fails to
emit a parsable answer in nearly every query. The
single-call scoped systems are within±0.03Faith
and±0.05Corr of each other across all four cor-
pora. This further suggests that the advantages of
MASDR-RAG depend on the synthesizer having
strong built-in capabilities for tool calling, a fea-
ture that the 8B Llama-3 model currently lacks.
A26 A Concrete WYDOT Failure Case
The query“What are the safety considerations
for crusher site inspections?”contains both in-
spection vocabulary (Construction Manual,6.6k
chunks) and safety vocabulary (Traffic & Crashes,
30.9k chunks). The monolithic embedder ranks
the dense Traffic & Crashes neighborhood first
and returns Highway Safety Plan chunks (4/5of
which are from Traffic & Crashes), producing
an on-topic-sounding but incorrect answer about
Highway Safety Plans. HYBRID-ROUTED’s regex
matches“inspection”, scopes to the Construction
Manual agent (6,641vectors), and returns §7-3-
1 / §7-3-2 / §5-4 / §7-3-3 of the Construction Man-ual plus a Materials QC dust mitigation chunk —
the correct §7-3 Crusher Site procedure on PPE,
dust mitigation, hot-work permits, and equipment
lockout.
A27 Embedding-Space Visualization on
Composite-9
Figure 6 repeats the embedding-space dilution
view on COMPOSITE-9 under the BGE-M3 re-
triever.
(a) Monolithic search space
Helpdesk
ConfluenceStackOverflow
Email (Enron)Tech Docs
(b) Scoped: StackOverflow agent only
in scope (StackOverflow) out of scope (filtered)
Figure 6: Embedding-space view of dilution on
Composite-9 (BGE-M3, t-SNE,5,805chunks). (a)
Monolithic: sources interpenetrate at cluster bound-
aries. (b) StackOverflow scope active: neighborhood
collapses to one source.
A28 External Baselines on Composite-9
We benchmark four established external systems
against the Composite-9 query suite. All four
share the same Qwen-7B answer generator where
applicable.
•BM25(Robertson and Zaragoza, 2009):
rank_bm25, top-10to Qwen-7B.
•ColBERTv2(Santhanam et al., 2022):
colbert-ir/colbertv2.0via
ragatouille; top-10to Qwen-7B.
•LangChain ReAct(Chase, 2023):
create_react_agentwraps the same
nine retrieval tools.
•Custom ReAct(Yao et al., 2023): hand-
implemented ReAct over the same nine tools.
A29 BEIR Calibration of BGE-M3
To anchor our retrieval numbers to a pub-
lished baseline, we run BGE-M3 on the BEIR
MS-MARCO dev split (n=1,000queries,
200k passages: all gold-positives plus a ran-
dom fill from the full8.8M corpus to keep
wall time tractable). Under the same FAISS
IndexFlatIP+ L2-normalized cosine setup

used everywhere else in this paper, we ob-
tain nDCG@10=0.854, MRR@10=0.822,
Recall@1=0.721, Recall@10=0.961,
Recall@100=0.992. The Recall@10 figure
is consistent with the BGE-M3 paper (Chen et al.,
2024); the absolute nDCG is higher because of
the200k cap. On the full8.8M-passage corpus,
BGE-M3 reports nDCG@10≈0.46.
Baseline p50 (s) p95 (s) Corr% Faith
BM25-only.04.06 42.0.00
BM25 + Qwen1.74 4.98 74.0.06
ColBERTv2-only.01.01 42.0.00
ColBERTv2 + Qwen1.80 6.95 82.0.06
LangChain ReAct12.42 33.70 48.0.10
Custom ReAct4.48 7.37 94.4.00
Monolithic1.94 5.43 90.0.04
Regex-Scoped2.64 6.85 90.0.02
HYBRID-ROUTED2.86 7.82 85.7.09
Table 22: External baselines on Composite-9 (n=50,
Qwen-7B synth+judge). Scoped single-call systems hit
86–90%correctness at a fraction of LangChain’s12.4s
p50.
A30 Latency–Correctness Pareto Plot
Figure 7 plots the latency–correctness Pareto fron-
tier on COMPOSITE-9 for every system we evalu-
ate.
10−2 10−1 100 101
Latency p50 (seconds)0.40.50.60.70.80.91.0Correctness
BM25-onlyColBERTv2-onlyBM25 + QwenColBERTv2 + Qwen
LangChain ReActMonolithic
Regex-Scoped
Hybrid-RoutedCustom ReAct
Retriever only
Retriever + Qwen
Scoped / routedIterative agent
Pareto frontier
Figure 7: Latency–correctness Pareto on Composite-9
(Qwen-2.5-7B synth, BGE-M3 retriever, Qwen judge,
n=50). Frontier (dashed): from retrieval-only base-
lines through single-call scoped systems to Custom Re-
Act. LangChain ReAct sits well inside (12.4s p50 for
0.48Corr).
A31 HotpotQA Span-Level Metrics
In addition to LLM-as-judge faithfulness and Re-
call@10, we report HotpotQA’s span-level met-
rics so that the numbers are directly comparable
with the prior HotpotQA literature. Strict EM
is near-zero because RAG outputs are paragraph-
length, so we add two long-form-friendly variants(Window-EM: contiguous token-window match;
Contains: substring of normalized prediction).
MonolithicRegex-Scoped Hybrid-RoutedMASDR-RAG Custom ReAct02000400060008000100001200014000Tokens per query11.3k
10.7k 10.8k13.0k
11.6kQwen-2.5-7B backbone
MonolithicRegex-Scoped Hybrid-RoutedMASDR-RAG Custom ReAct010000200003000040000
7.4k 7.6k6.6k11.4k38.7kLlama-3-8B backbone
0123456
1.0 1.01.42.12.3
0123456
LLM calls per query
1.0 1.01.52.15.5
Prompt tokens Completion tokens LLM calls per query
Figure 8: Per-query tokens (stacked) and LLM calls
(red line) on WYDOTn≈200. Qwen-7B: all systems
within10.7–13.0k tokens,≤2.3calls. Llama-8B:
ReAct grows to5.5calls and38.7k tokens —5.9×
HYBRID-ROUTED’s budget without a matching qual-
ity gain. Discussed in §7.
SystemEM F1 winEM Contains
BM25 only.000.000.000.000
ColBERT only.000.000.000.000
BM25 + Qwen.000.068.419.427
ColBERT + Qwen.000.072.436.445
Monolithic.001.078.417.427
HYBRID-ROUTED.000.055.449.470
MASDR-RAG.000.044.414.438
LangChain.000.059.264.277
ReAct.000.063.253.267
Table 23: HotpotQA-distractor span-level metrics
(n≈2,000per system).
A32 Composite-9 Source Composition
Table 24 lists the composition of COMPOSITE-9
— all nine source types ingested and their relative
sizes.
Source Dataset Chunks
Confluence Wikipedia (en, sub.)2,354
Docs MS-MARCO passages1,005
Gmail Enron email1,847
Helpdesk HelpSteer4,257
StackOverflow Stack Overflow QA1,849
Slack OpenAssistant chat2,618
Github GitHub issues1,519
Jira GitHub bug issues1,508
Reports SEC filings (10-K)1,037
Total17,994
Table 24: Composite-9 composition (all nine source
types ingested).

A33 Router Variants (R0/R1/R2)
We compare three routers on the WYDOT labeled
subset (n=155,5-fold stratified CV):R0 Regex
(production rule patterns),R1 TF–IDF+LR(word
1–2grams, one-vs-rest logistic regression, class-
balanced), andR2 BGE-M3 linear probe(1024-
d BGE-M3 query embedding, one-vs-rest logistic
regression).
Router Acc 95%CI Top-2F 195%CI
R0 Regex.471.471‡.497
R1 TF-IDF.66 .57,.74 .78±.11.63 .52,.72
R2 BGE-LR.76 .72,.79 .86±.03.74 .70,.79
Table 25: Router accuracy (n=155,5-fold CV). R0 is
single-label deterministic, so Top-2equals Top-1(‡).
R2 lifts accuracy+28.4points and weighted
F1+24.4points over R0. Plugged in end-to-end
as R2-ROUTED(see §10, Tab. 9), this translates
to the highest correctness (0.303) and Recall@10
(0.375) on WYDOT200-q.
A34 Cross-Encoder Rerank Ablation
We wrap each backend with a RERANKBACKEND
that takes top-30bi-encoder candidates, re-scores
them withBAAI/bge-reranker-v2-m3
(568M-parameter cross-encoder), and returns the
top-10sorted by cross-encoder score (∼50ms per
(query, passage) pair on an L40S;∼1.5s added
latency per query).
Corpus / System∆Faith∆Corr∆R@10
Monolithic
WYDOT−.065−.005—
Composite-9+.080 +.040 +.020
MultiHop-RAG+.064 +.034 +.005
FinanceBench+.021 +.042 +.035
MMLU-Pro+.026 +.006 +.002
NQ-Open+.026 +.026—
HYBRID-ROUTED
WYDOT−.051−.011—
Composite-9+.114−.057 +.029
MultiHop-RAG+.050 +.012 +.005
FinanceBench−.021 +.056 +.035
MMLU-Pro−.004−.018 +.002
NQ-Open−.016−.028—
Table 26: Rerank deltas (rerank−base, same queries).
R@10 omitted on WYDOT and NQ-Open (no gold-
passage labels).
Rerank helps with Composite-9 / MultiHop / Fi-
nanceBench / MMLU-Pro / NQ-Open, but hurts
on WYDOT. In WYDOT, the bi-encoder alreadyorders chunks by section/year/version metadata
that the gold answer requires; the cross-encoder’s
lexical reweighting promotes topically on-target
but year/version-wrong chunks, consistent with
the literature (Nogueira et al., 2020; Formal et al.,
2022).
A35 SPLADE vs. Dense Retriever
Table 27 compares the SPLADE learned-sparse re-
trieval against the dense BGE-M3 retriever on the
same query set.
Corpus SystemRetriever Faith Corr
Composite-9Mono BGE-M3.760.900
Mono SPLADE.880.940
Hybrid-R BGE-M3.771.857
Hybrid-R SPLADE.880.980
MultiHopMono BGE-M3.460.692
Mono SPLADE.464.726
Hybrid-R BGE-M3.484.610
Hybrid-R SPLADE.474.602
FinanceB.Mono BGE-M3.320.547
Mono SPLADE.308.432
Hybrid-R BGE-M3.340.527
Hybrid-R SPLADE.288.486
Table 27: SPLADE(opensearch-neural-
sparse- v2- distill) vs. dense BGE-M3,
single-call systems, Qwen-2.5-7B synth. No system-
atic sparse-vs-dense winner.
A36 Infrastructure Parity: FAISS vs.
Neo4j
To rule out the index-implementation confound,
we re-ran all five non-WYDOT corpora under both
FAISSIndexFlatIPand a local Neo4j5.26
HNSW (App. A23). The same BGE-M3 query
embedding, scope filter, and Qwen-7B synthesizer
are used; only the index data structure differs.
A37 Composite-9 Full Single-Call Block
This appendix supports the falsification of
the context-fragmentation hypothesis in §10 on
EnterpriseComposite-9. The SINGLECALLvari-
ant collapses MASDR-RAG’s multi-round syn-
thesis into a single call over the concatenated
chunk union. If fragmentation were the cause
of the precision–faithfulness paradox, this vari-
ant should recover faithfulness; instead, it drops
both faithfulness and correctness from MASDR-
RAG’s already-degraded levels, mirroring the
WYDOT result.

Faith Corr R@10
Corpus SystemF N F N F N|∆| max
Composite-9 Monolithic.760.780.900.820.920.880.080
HYBRID-ROUTED.771.800.857.820.943.880.063
MASDR-RAG.740.680.740.700.740.800.060
MultiHop Monolithic.460.470.692.710.975.977.018
HYBRID-ROUTED.484.466.610.608.975.977.018
MASDR-RAG.434.368.616.612.903.903.066
FinanceBench Monolithic.320.313.547.585.847.837.038
HYBRID-ROUTED.340.333.527.510.847.837.017
MASDR-RAG.320.340.640.607.807.713.094
MMLU-Pro Monolithic.356.376.488.486.998 1.000.020
HYBRID-ROUTED.408.400.518.506.998 1.000.012
MASDR-RAG.344.355.460.463.682.687.011
NQ-Open Monolithic.410.404.282.290— —.008
HYBRID-ROUTED.412.426.322.312— —.014
MASDR-RAG.390.374.406.366— —.040
NQ has no gold-passage labels (answer-only), so R@10 is omitted.
Table 28: Infrastructure parity: F (FAISS) vs. N (Neo4j), median|∆|=.02, no architectural ranking flips.
SystemR@10 Faith Corr
Monolithic.920.760.900
Regex-scoped.920.800.900
HYBRID-ROUTED.943.771.857
MASDR-RAG.740.740.740
SingleCall.700.620.620
ReAct.944.778.944
Table 29: Composite-9 (n=50for new systems,
n=27–50for baselines depending on judge availabil-
ity). SINGLECALLis strictly worse than MASDR-
RAG, mirroring the WYDOT falsification.
A38 Full Cross-Backbone Replication
Table 30 expands the cross-backbone summary
of Table 10 (§10) to all four metrics per system
per generator. The split along the open-source
vs. commercial axis is consistent across Faith,
Corr, and R@10: MASDR-RAG’s faithfulness
collapse under Claude and GPT-5-mini is not an
artifact of any single metric, and it does not appear
under Qwen-7B or DeepSeek-V3.
A39 Full Reproducibility Details
Benchmarks and splits:The 200-query
WYDOT evaluation suite is released with the
harness. The other six corpora are built by
the assembly and indexing pipeline described
above: EnterpriseComposite-9 from nine public
HuggingFace datasets, and MultiHop-RAG,
FinanceBench, MMLU-Pro, NQ-Open, and BEIR
MS-MARCO from their published splits.Backbone SystemnFaith Corr R@10
Qwen-7BMono193.352.212.188
Regex-Sc.193.306.244.219
Hybrid-R.193.342.218.194
MASDR193.394.275.258
ClaudeMono100.250.240.188
Regex-Sc.100.250.210.219
Hybrid-R.100.270.210.250
MASDR100.010.080.500
GPT-5mMono29.378.172.188
Regex-Sc.29.414.241.219
Hybrid-R.29.310.276.219
MASDR29.241.414.280
DeepSeekMono44.227.455.188
Regex-Sc.44.364.568.219
Hybrid-R.44.364.614.281
MASDR44.318.523.469
Table 30: Full cross-backbone WYDOT-200table.
Within each backbone,nis theintersectionof queries
answered by all four systems (uniform across rows), so
means within each block are apples-to-apples.
Hyperparameters:Inference temperature is0.0
for routing and0.2for answer generation
(top-p= 0.9,1024-token cap). Retrieval
is deterministic given the embedder. Bi-
encoder retrieves top-30; cross-encoder reranks
to top-10; bi-encoder-only paths return top-15.
FAISSIndexFlatIPon L2-normalized BGE-
M3 (1024-d) vectors; Neo4j sandbox uses Neo4j
5.26 HNSW with defaultm/ef_construction
and cosine similarity. The trained R2 router is a
class-balanced one-vs-rest logistic regression with
max_iter= 4000over BGE-M3 query embed-

dings.
Statistical tests:Cross-backbone Tab. 10 re-
ports bootstrap95%CIs (1,000resamples, seed0)
on Faith/Corr. Pairwise comparisons use paired
permutation tests atα= 0.05with10,000per-
mutations. The per-query dilution regression
usesscipy.stats.linregressonn=147
paired observations.
Hardware and runtimeHuggingFace
transformers(no vLLM/TensorRT). One full
WYDOT eval (200×5systems):≈40min on
L40S (Qwen-7B) /≈100min on A30 (Llama-
8B). MultiHop-RAG (500×4):≈1hour on L40S.
Total wall-clock time for∼70,000judge calls:
≈24GPU-hours on L40S across∼50SLURM
jobs.
Random seeds:Seed0: t-SNE projections,
bootstrap CIs, R2 CV splits. Seed42: query gen-
erator. Generation is deterministic at temperature
0(router) and uses top-psampling withp= 0.2
for the synthesizer (single-sample completion per
query).
A40 MA-RAG and SCOUT-RAG:
Implementation and Caveats
This appendix details the two external multi-agent
RAG baselines (MA-RAG and SCOUT-RAG)
added in §8 and diagnoses the near-zero faithful-
ness of their answers under our judge.
MA-RAG (port):MA-RAG (Nguyen et al.,
2025) chains four agents: Planner, Step-Definer,
Extractor, and QA via a collaborative chain-of-
thought. The authors release a public implemen-
tation for OpenAI-only use. We port the prompts
verbatim and rewire them onto our Qwen-2.5-7B
/ BGE-M3 stack so that the comparison isolates
agent coordination, not the retriever or backbone.
Structured outputs are obtained via JSON-format
prompting and regex-tolerant parsing (rather than
OpenAI’s schema-constrained decoding); a raw-
text fallback prevents a single malformed JSON
response from collapsing a step’s answer to the
empty string. Retrieval usesglobal(unscoped)
BGE-M3 search — MA-RAG operates without or-
ganizational metadata, which is the fair contrast to
our scoped methods.
SCOUT-RAG (reimplementation from paper):
SCOUT-RAG (Li et al., 2026) runs four cooper-
ative agents:DRAA(Domain Relevance Assess-ment),PAGA(Partial Answer Generation),OASA
(Overall Answer Synthesis), andAQAA(Answer
Quality Assessment) in an iterative refinement
loop with a published strategy selector (Eq. 6 of
the original paper: DEPTH/ BREADTH/HYBRID
/ STOP). The paper does not release prompt
templates, the DRAA feature-fusion function, the
OASA aggregation logic, the AQAA evaluation
prompt, or the∆Qstagnation thresholdϵ. Our
reimplementation follows Algorithm 1 of the pub-
lished paper verbatim for control flow and ter-
mination criteria, and supplies prompts,ϵ=0.05,
and fusion operators ourselves; this is disclosed
in the system label as “SCOUT-RAG (OUR
REIMPL)”. We map each existing scope agent
(TOOL_TO_AGENTper corpus) to one SCOUT-
RAG “domain”; HIGH-tier retrieval usesk=20,
MODERATEusesk=5. When DRAA classifies
every domain as IRRELEVANT(an out-of-corpus
query), we fall back to a single unscoped global
retrieval rather than routing to an arbitrary scope
— this prevents pathological underperformance on
out-of-domain queries.
Comparison fairness:Both baselines use the
sameLLM (Qwen-2.5-7B), thesameretriever
(BGE-M3), and thesamedomain partition
as our scoping methods. The only contrast
is the coordination protocol: SCOUT-RAG’s
DRAA/PAGA/OASA/AQAA refinement loop and
MA-RAG’s plan-decompose-execute chain vs.
MASDR-RAG’s function-calling orchestration vs.
HYBRID-ROUTED’s router-plus-single-synthesis.
Why faithfulness is reported as — (‡).Our
Qwen judge scores faithfulness by checking that
each substantive claim in the model answer is
supported by a chunk the answer cites via a
[SourceN]marker. The MA-RAG and
SCOUT-RAG prompts (taken verbatim from the
public repository or written to match the published
algorithm, respectively) do not request such ci-
tation markers — MA-RAG’s QA agent emits a
concise paraphrased answer, and SCOUT-RAG’s
OASA synthesizes across domains without source
labels. The judge, therefore, classifies almost ev-
ery claim as “no cited support, driving the raw
score to near zero. Because this is a structural
property of the protocol rather than a measurable
unfaithfulness of the output, we report faithfulness
as — for these two systems — rather than risk im-
plying the answers are actually unfaithful. Man-
ual inspection of MA-RAG / SCOUT-RAG out-

puts finds that they are typically as grounded in
their retrieved context as Monolithic’s; the metric
simply cannot be applied to them on an apples-to-
apples basis. Thecorrectnessdrop (Composite-
9:44.4%/66.7%vs.90.0–94.4%for our methods;
MultiHop-RAG:29.8%/55.2%vs.58.0–69.2%;
WYDOT:11.0%/24.1%vs.28.7–35.1%) is the
real comparable signal, and it supports the paper’s
scope, don’t over-orchestrateprescription.
Compute budget:On the Qwen-2.5-7B / BGE-
M3 stack, the new baselines cost substantially
more per query than our scoped methods: MA-
RAG≈22−26LLM calls and≈15−30sp 50
latency; SCOUT-RAG≈8−10LLM calls and
≈24−53s; vs. Monolithic / Regex-Scoped at1
call and≈2−9s.
A41 What We Do Not Claim
We do not claim HYBRID-ROUTEDimprovescor-
rectness over monolithic on WYDOT under the
apples-to-apples Qwen stack; the two are sta-
tistically indistinguishable on correctness, and
MASDR-RAG modestly leads on every metric
(+ 0.07Recall@10,+ 0.04Faith,+ 0.06Corr). We
do not claim that the production Gemini paradox
generalizes beyond Gemini and the Claude/GPT
regime in Tab. 10; under Qwen / DeepSeek, it does
not reproduce. We do not claim MASDR-RAG is
uniformly best: in the GPT-5-mini partial replica-
tion, it does shed faithfulness, mirroring the Gem-
ini pattern.
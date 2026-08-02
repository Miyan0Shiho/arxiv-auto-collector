# A Structured Knowledge Infrastructure for Domain-Specific Data Asset Discovery

**Authors**: Mengdi Chen, Yuanxin Huang, Yulin Jiang, Wei Sun

**Published**: 2026-07-30 06:40:05

**PDF URL**: [https://arxiv.org/pdf/2607.27748v1](https://arxiv.org/pdf/2607.27748v1)

## Abstract
Enterprise data analytics agents face two structural failures: generic RAG retrieves the wrong asset (Hit@10=19.1%) and delivers no usage knowledge to prevent metric misinterpretation---stemming from four root causes (C1--C4) ranging from semantic gap and entity ambiguity to schema drift and asset-usage gap. We present a two-layer solution deployed in the commercial advertising data warehouse at Xiaohongshu (5,300+ Hive tables, 14 domains). A three-tier dual-purpose knowledge base (179 documents, eight-section annotation template) serves both retrieval and generation, with a closed-loop refresh pipeline maintaining day-level freshness (one yes/no approval, 30s hot-reload). The Graph-Guided Retriever (GGR) uses a 2,859-node knowledge graph as a candidate gate with intent routing to deliver 71.6x token reduction. The Scene-Aware Ranker (SAR) applies 19-class entity recognition and explicit scenario annotations; negative knowledge alone contributes 25 percentage points of Hit@10 gain. On two 100-question benchmarks, Hit@10 rises from 19.1% to 96.6% (+77.5pp) and knowledge coverage from 56% to 77%, at 4.84--5.33s end-to-end latency.

## Full Text


<!-- PDF content starts -->

A Structured Knowledge Infrastructure for Domain-Specific Data
Asset Discovery
Mengdi Chen
Xiaohongshu
Shanghai, China
chenmengdi@xiaohongshu.comYuanxin Huang
Xiaohongshu
Shanghai, China
huangyuanxin@xiaohongshu.com
Yulin Jiang
Xiaohongshu
Shanghai, China
jiangyulin@xiaohongshu.comWei Sun
Xiaohongshu
Shanghai, China
sunwei@xiaohongshu.com
Abstract
Enterprise data analytics agents face two structural failures: generic
RAG retrieves the wrong asset (Hit@10 = 19.1%) and delivers no
usage knowledge to prevent metric misinterpretation—stemming
from four root causes (C1–C4) ranging from semantic gap and en-
tity ambiguity to schema drift and asset-usage gap. We present a
two-layer solution deployed in the commercial advertising data
warehouse at Xiaohongshu (5,300+ Hive tables, 14 domains). A
three-tier dual-purpose knowledge base(179 documents, eight-
section annotation template) serves both retrieval and generation,
with a closed-loop refresh pipeline maintaining day-level fresh-
ness (one yes/no approval, 30 s hot-reload). TheGraph-Guided
Retriever(GGR) uses a 2,859-node knowledge graph as a candi-
date gate with intent routing to deliver 71.6 ×token reduction. The
Scene-Aware Ranker(SAR) applies 19-class entity recognition and
explicit scenario annotations; negative knowledge alone contributes
25 percentage points of Hit@10 gain. On two 100-question bench-
marks, Hit@10 rises from 19.1% to96.6%(+77.5 pp) and knowledge
coverage from 56% to77%, at 4.84–5.33 s end-to-end latency.
Keywords
Agentic AI, Knowledge Management, Data Asset Retrieval, Knowl-
edge Graph, Scenario Annotation, Enterprise RAG
1 Introduction
Enterprise data analytics agents face two compounding failures:
finding the right asset among thousands of overlapping candidates,
and using it correctly despite heterogeneous schemas and polyse-
mous metrics. Production analysis reveals four root causes:(C1)
high-frequency terms are non-discriminative noise for embeddings;
(C2)entity references require structured grounding;(C3)static an-
notations drift as schemas evolve;(C4)correct assets still produce
wrong results without usage knowledge.
Our approach.We build a dual-purpose knowledge infrastruc-
ture where GGR targets C1/C2 via graph-guided narrowing and
intent routing, and SAR targets C2/C4 via entity-matched scoring
and structured knowledge delivery. For C3, a closed-loop pipeline
hot-reloads LLM-drafted patches into memory within 30 s after
expert yes/no approval (median 18 s). Unlike GraphRAG [ 4] (global
synthesis) or standard RAG [ 7], we use graph community clustering
as a precision gate and treat explicit domain knowledge as first-classretrieval signals. Downstream agents (SQL-writing agents, BI report
assemblers) invoke GGR and SAR as tool-style structured APIs—not
as a text-chunk retriever—receiving ranked asset IDs paired with
intent-filtered knowledge slices, and may re-query with narrowed
scope during multi-step reasoning.
Contributions.GGR: a 2,859-node knowledge graph used as a
candidate gate with 7-class intent routing and 459 synonym groups;
GGR alone (asset level) reaches Hit@10 = 0.73, with 71.6 ×token re-
duction vs. naive full-corpus RAG.SAR: 19-class entity recognition
and explicit scenario annotations; applicable annotations alone
contribute 48 pts, not-applicable suppression 25 pts. The full
pipeline (GGR+SAR) raises Hit@10 to 96.6% (+77.5 pp over legacy).
2 Problem Definition
We target the Xiaohongshu commercial advertising data ware-
house (5,300+ Hive tables, 1,200+ BI datasets, 14 domains) with
two tasks. Prior data-discovery systems focus on table linkage and
join finding [ 5,10], whereas our target is semantic asset routing
for downstream agent execution.Task 1 (Asset discovery):given
query𝑞, rank assets so ground-truth 𝑎∗appears in top- 𝐾(metric:
Hit@10; legacy baseline: 19.1%).Task 2 (Knowledge coverage):re-
trieve the required usage knowledge (table form, metric semantics,
disambiguation rules) for 𝑎∗; we evaluate file-level recall, leaving
downstream execution (e.g., text-to-SQL benchmarks [ 9]) to future
work. Low candidate-set purity in the legacy system motivates our
GGR→SAR pipeline.
3 Knowledge Infrastructure
Our knowledge infrastructure is structured into three tiers (Fig. 1)
to balance routing speed with semantic richness.
Tier 1(routing table + hotlist): 𝑂(1)keyword routing to the
50 most-queried assets or tool/SOP documents.Tier 2: 179 Mark-
down documents (Common / Domain / Production) each with an
eight-section annotation—§ 1–2 (aliases, entity labels) retrieval-
oriented; § 3 (applicable/not-applicable scenarios) for retrieval and
generation; § 4–8 (SQL, metrics, BI, schema, disambiguation) as
480-token intent-filtered slices (vs. 2,400-token full documents).
Tier 3: knowledge graph [ 1] with 474 Louvain communities [ 2],
each mapping to 3–15 candidates per query. For C3, LLM-drafted
arXiv:2607.27748v1  [cs.IR]  30 Jul 2026

M. Chen et al.
Figure 1: System overview and three-tier dual-purpose knowledge base. Top: GGR (coarse recall) and SAR (fine ranking) invoke
the knowledge base in series. Tier 1: 𝑂(1)keyword routing + high-frequency asset hotlist. Tier 2: 179 structured documents
with eight-section annotation. Tier 3: 2,859-node knowledge graph as candidate gate (474 communities).
patches hot-reload within 30 s (P95; median 18 s, ∼45/month) af-
ter expert yes/no approval, mitigating annotation drift through
continual maintenance [3].
4 Graph-Guided Retriever (GGR)
GGR (Fig. 2, left) treats the knowledge graph as acandidate gate:
graph traversal narrows the search space before any vector compar-
ison, constraining retrieval to semantically relevant communities.
The GGR index covers 211 documents (179 Tier-2 content docu-
ments plus 32 directory-style index documents—cross-tier aggre-
gate files that each link to multiple content documents).
4.1 Knowledge Graph Construction
We build the graph via deterministic AST parsing (confidence 1.0)
and LLM relation extraction (confidence 0.6–0.9); the confidence
scoring helps mitigate LLM hallucination in graph edges [ 6]. The
graph has 2,859 nodes (1,874 concept, 526 code, 440 document, 19 ra-
tionale) and 3,325 edges across three relation types (Doc ↔Concept
2,712; Concept↔Concept 562; Code/Rationale 51).
4.2 Dual-Path Architecture
The two paths run in parallel and complement each other:Path 1
(Graph traversal)provides broad semantic coverage: it expandsquery tokens via 459 synonym groups, matches at three preci-
sion levels (exact/substring/path-segment), and propagates scores
through a 6,202-term inverted index (built from graph node labels
and synonym expansions) with IDF weighting and hub-node fanout
decay. Caller-specified domains receive a path-prefix boost; high-
frequency core datasets are pinned as fallbacks to prevent them from
being ranked out by rare-term bias.Path 2 (Intent routing)pro-
vides high-precision directional recall: it classifies the query into 7
intent classes ( discovery ,definition ,data_fetch_bi ,write_sql ,
diagnose ,attribute ,interpret ) and maps each intent directly
to its specialist documents. The two paths merge with graph pri-
ority. On the same 100-question Task 1 benchmark atdocument
level: graph-only Hit@10 = 0.79, intent-only = 0.63, combined = 0.85
(asset-level in Table 2). By first narrowing thousands of documents
to 3–15 community candidates, GGR lets downstream RAG operate
in a low-noise space without depending on a large LLM. GGR alone
reaches asset-level Hit@10 = 0.73 (est.), lower than the document-
level 0.85 due to the stricter one-to-many doc →asset mapping; SAR
then lifts asset-level to 0.966.
Implementation.The intent classifier is a keyword-rule matcher
over 183 domain-specific trigger terms (7 classes) with LLM fallback
for unmatched queries—avoiding fine-tuning cost while maintain-
ing single-digit-ms latency. Entity recognition (19 classes, Section 5)

A Structured Knowledge Infrastructure for Domain-Specific Data Asset Discovery
Figure 2: Overview of retrieval and ranking pipeline. Left (GGR): dual-path retrieval merging graph traversal (Path 1) and
intent routing (Path 2). Right (SAR): five-stage ranking pipeline producing ranked assets and knowledge slices.
uses rule-based matching against curated vocabularies (2–40 aliases
per entity), tuned on production query logs. The 459 synonym
groups were semi-automatically expanded from 120 seed terms
via embedding-based clustering plus expert review; the 6,202-term
inverted index is auto-rebuilt on every KB refresh.
5 Scene-Aware Ranker (SAR)
SAR (Fig. 2, right) elevates ranking from keyword matching to
semantic-scene alignment—the joint optimization of (i) entity-level
alignment between query business terms and dataset entity tags,
and (ii) scenario-level alignment between query intent and dataset
applicable/not-applicabledescriptions.
5.1 Five-Stage Pipeline
SAR processes candidates in five stages: (1) expand document hits
to individual assets via a precomputed doc→asset mapping with
high-frequency fallback indexes; (2) hard-recall entity-labeled as-
sets to counter graph sparsity; (3) identify 19 entity classes via rule-
based matching (zero LLM latency)—spanning advertiser-centric
(advertiser, agency, brand, targeting), content (creative, note, note
keyword, commercial note type, IP project), data (SPU, third-party
platform, reputation pass, strategy) and query dimensions (search
term, upstream/downstream term, back-search term, region, user,
enterprise account)—aligning query business terms with dataset en-
tity tags; (4) match applicable /not-applicable scenario phrases;
(5) compute tiered scores.
Stage 5 (Tiered scoring).A resource-type tier 𝑇(𝑎)∈{ 1000,600,500}
(BI dataset / dashboard / Hive table) enforces a hard hierarchy. Posi-
tive signals include IDF-weighted entity alignment (+8.0, dominant),
asset-name match ( +6.0), and applicable annotation match (+6.0,
with an extra+4.0for full-phrase matches); the single negative sig-
nalnot-applicable (−8.0) suppresses false positives regardless of
other signals. IDF weighting 𝜙(𝑒)=max( 0.375,1−0.8𝐶(𝑒)) , where
𝐶(𝑒) is the corpus-level document frequency of entity 𝑒across all
179 files, discounts ubiquitous terms (e.g., “advertising”, 𝐶=86%→0.375) and preserves rare entities (brand,𝐶=0.3%→1.0). Queries
involving organizational hierarchy (sales / operations / industry
/ track) trigger a targeted boost on wide-table assets—addressing
high-frequency structural queries that keyword matching alone
struggles with.
After ranking, SAR selects intent-driven slices from § 4–8 ( write_sql→ § 4+§ 7;
definition→ § 5;data_fetch_bi→ § 3+§ 6) and delivers them to
the LLM as in-context knowledge [8].
6 Evaluation
6.1 Setup
Benchmark.Two separate 100-question production benchmarks:
Benchmark-A for Task 1 (asset discovery) and Benchmark-B for
Task 2 (knowledge coverage), each sampled independently from
production query logs, evaluated against the live system (v9) vs.
the legacy production system. The legacy system performs BM25-
style full-corpus retrieval over 179 documents without graph/entity
signals, serving as the primary baseline; embedding-based com-
parisons require domain-specific fine-tuning in this 14-domain
warehouse and are left for future work.Task 1(Asset Discovery):
ground truth is the canonical asset over 5,300+ Hive tables and
1,200+ BI datasets across 14 domains.Task 2(Knowledge Cover-
age): each question is annotated with 1–2 must-read files; coverage
= recall of ground-truth files (before slicing).
6.2 Main Results
As shown in Table 1, our system achieves Hit@10 = 96.6% vs. 19.1%
(legacy),+77.5 pp, Hit@1 from 12.4% to 49.4%, and knowledge cov-
erage from 56% to77%(+21 pts). On a standalone SAR benchmark
(100 questions, independently constructed from asset-ranking logs,
separate from Benchmark-A/B), SAR alone achieves Hit@5 = 98%,
Recall = 100%, MRR = 0.803. Despite retrieving more files per query
(6.5 vs. 4.5), token consumption is 71.6 ×lower vs. naive full-corpus
RAG, because our system delivers 480-token intent-filtered slices
rather than full 2,400-token documents ( (179docs× 2,400)÷(12.5×

M. Chen et al.
Table 1: Main results on two 100-question production benchmarks. Top: Task 1 asset discovery (Hit@K). Legacy Hit@5 = Hit@10
because the system returns exactly 10 candidates. Bottom: Task 2 knowledge coverage (file-level recall before slicing).
Task 1: Asset Discovery (Hit@K)
Method Hit@1 Hit@3 Hit@5 Hit@10 Latency
Legacy system 0.124 0.169 0.191 0.191 2.97 s
Ours (v9, online) 0.494 0.888 0.933 0.966 4.84 s
Task 2: Knowledge Coverage
Method Coverage Files/Query Latency
Legacy system 56.0% 4.5 4.03 s
Ours (v9, online) 77.0% 6.5 5.33 s
480), where 12.5 is the measured average number of GGR candidate
documents per query). Latency overhead (+1.87 s Task 1, +1.30 s
Task 2) stems from graph traversal and entity recognition; both
remain within SLA and are negligible relative to analysts’ prior
manual search of several minutes per query.
6.3 Ablation Study
Table 2: Ablation (Hit@10). Each Δis marginal contribution
of the removed component.
Configuration Hit@10ΔHit@10
Ours (v9, online) 0.966—
w/o applicable annotation 0.486−0.480
w/o graph path (intent only) 0.606−0.360
w/o not-applicable annotation 0.716−0.250
w/o entity recognition 0.786−0.180
w/o IDF entity weighting 0.846−0.120
Table 2 shows applicable annotations contribute the most (–
48 pp); graph path –36 pp; not-applicable suppression –25 pp
(negative knowledge matters as much as positive); entity recogni-
tion –18 pp and IDF weighting –12 pp confirm the value of fine-
grained entity modeling. All ablations (0.486–0.846) far exceed
legacy (0.191); Δvalues are single-factor removals and not additive
due to feature interactions.
7 Deployment Notes
Two engineering decisions shaped the deployed system. First, C3
patches use expert yes/no approval (median 18 s) rather than fully
automated LLM merges: in early pilots, unsupervised auto-merges
surfaced schema-drift-induced regressions that were expensive to
roll back; a lightweight human gate at 45 patches/month is afford-
able and eliminates this failure mode. Second, GGR and SAR are
deliberately staged (coarse →fine) rather than fused in parallel: in
our warehouse, parallel fusion left too much noise in the candi-
date set for the downstream ranker to filter effectively, whereas
the staged pipeline lets SAR operate on an already high-precision
candidate pool.8 Conclusion
We present a knowledge infrastructure that serves as the retrieval
and knowledge layer of a production agentic analytics pipeline:
agents invoke GGR and SAR to ground every query in structured
domain knowledge before SQL generation or BI report assembly—
directly realising the “Agentic AI Goes Live” agenda. Deployed in
the Xiaohongshu commercial advertising data warehouse, GGR
and SAR achieve 96.6% Hit@10 (+77.5 pp) and 77% knowledge
coverage (up from 56%, +21 pp), while reducing token footprint
by 71.6×vs. naive full-corpus RAG, at 4.84–5.33 s end-to-end la-
tency. Our results show that in specialized domains, explicit do-
main knowledge—graph structure, intent routing, and scenario
annotations—dominates generic embeddings. The eight-section
template is domain-agnostic; adapting to a new domain requires
document authoring ( ∼2 h/doc) with automated graph construc-
tion. Future work: cross-domain transfer and closed-loop agentic
execution.
References
[1]aivi fyi. 2024. Graphify: Compile any codebase or document folder into a
queryable knowledge graph. https://github.com/aivi-fyi/graphify. MIT License.
[2] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefeb-
vre. 2008. Fast unfolding of communities in large networks.Journal of Statistical
Mechanics: Theory and Experiment2008, 10 (2008), P10008.
[3]Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales
Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. 2021. A continual learning
survey: Defying forgetting in classification tasks.IEEE Transactions on Pattern
Analysis and Machine Intelligence44, 7 (2021), 3366–3385.
[4]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
RAG approach to query-focused summarization.arXiv preprint arXiv:2404.16130
(2024).
[5]Raul Castro Fernandez, Ziawasch Abedjan, Famien Koko, Gina Yuan, Samuel
Madden, and Michael Stonebraker. 2018. Aurum: A data discovery system. In
Proceedings of the IEEE International Conference on Data Engineering (ICDE). IEEE,
1001–1012.
[6] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S Yu. 2022.
A survey of knowledge graph embedding approaches: Problems, methods, and
applications.IEEE Transactions on Knowledge and Data Engineering34, 3 (2022),
1571–1590.
[7]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive NLP
tasks.Advances in Neural Information Processing Systems33 (2020), 9459–9474.
[8] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi,
Nathanael Schärf, and Denny Zhou. 2022. Rethinking the role of demonstrations:
What makes in-context learning work?arXiv preprint arXiv:2202.12837(2022).
[9] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James
Ma, Irene Li, Qingning Yao, Shanelle Roman, et al .2018. Spider: A large-scale
human-labeled dataset for complex and cross-domain semantic parsing and
text-to-SQL task. InProceedings of the 2018 Conference on Empirical Methods in

A Structured Knowledge Infrastructure for Domain-Specific Data Asset Discovery
Natural Language Processing. 3911–3921.
[10] Erkang Zhu, Dong Deng, Fatemeh Nargesian, and Renée J Miller. 2019. JOSIE:
Overlap set similarity search for finding joinable tables in data lakes. InPro-
ceedings of the ACM SIGMOD International Conference on Management of Data.847–864.
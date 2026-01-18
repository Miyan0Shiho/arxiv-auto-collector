# TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval

**Authors**: Abdelrahman Abdallah, Mohammed Ali, Muhammad Abdul-Mageed, Adam Jatowt

**Published**: 2026-01-14 14:45:20

**PDF URL**: [https://arxiv.org/pdf/2601.09523v1](https://arxiv.org/pdf/2601.09523v1)

## Abstract
Existing temporal QA benchmarks focus on simple fact-seeking queries from news corpora, while reasoning-intensive retrieval benchmarks lack temporal grounding. However, real-world information needs often require reasoning about temporal evolution and synthesizing evidence across time periods. We introduce TEMPO, the first benchmark combining temporal reasoning with reasoning-intensive retrieval across 13 domains. TEMPO features: (1) 1,730 complex queries requiring deep temporal reasoning such as tracking changes, identifying trends, or comparing cross-period evidence; (2) step-wise retrieval planning with 3,976 decomposed steps and gold documents mapped to each step for multi-hop evaluation; and (3) novel temporal metrics including Temporal Coverage@k and Temporal Precision@k measuring whether results span required time periods. Evaluation of 12 retrieval systems reveals substantial challenges: the best model (DiVeR) achieves only 32.0 NDCG@10 and 71.4\% Temporal Coverage@10, demonstrating difficulty in retrieving temporally complete evidence. We believe TEMPO provides a challenging benchmark for improving temporal reasoning in retrieval and RAG systems. Our code and data are available at https://github.com/tempo-bench/Tempo. See also our official website: https://tempo-bench.github.io/.

## Full Text


<!-- PDF content starts -->

TEMPO: A Realistic Multi-Domain Benchmark for Temporal
Reasoning-Intensive Retrieval
Abdelrahman Abdallah1, Mohammed Ali1, Muhammad Abdul-Mageed2, Adam Jatowt1
1University of Innsbruck2University of British Columbia
{abdelrahman.abdallah,mohammed.ali,adam.jatowt}@uibk.ac.at
muhammad.mageed@ubc.ca
Abstract
Existing temporal QA benchmarks focus on
simple fact-seeking queries from news cor-
pora, while reasoning-intensive retrieval bench-
marks lack temporal grounding. However, real-
world information needs often require reason-
ing about temporal evolution and synthesiz-
ing evidence across time periods. We intro-
duceTEMPO, the first benchmark combining
temporal reasoning with reasoning-intensive re-
trieval across 13 domains. TEMPO features:
(1) 1,730 complex queries requiring deep tem-
poral reasoning such as tracking changes, iden-
tifying trends, or comparing cross-period ev-
idence; (2) step-wise retrieval planning with
3,976 decomposed steps and gold documents
mapped to each step for multi-hop evaluation;
and (3) novel temporal metrics including Tem-
poral Coverage@k and Temporal Precision@k
measuring whether results span required time
periods. Evaluation of 12 retrieval systems re-
veals substantial challenges: the best model
(DiVeR) achieves only 32.0 NDCG@10 and
71.4% Temporal Coverage@10, demonstrat-
ing difficulty in retrieving temporally complete
evidence. We believe TEMPO provides a chal-
lenging benchmark for improving temporal rea-
soning in retrieval and RAG systems1.
1 Introduction
Information retrieval is a fundamental technology
that assists users in locating relevant information
from extensive corpora (Ali et al., 2025; Thakur
et al., 2021; Abdallah et al., 2025a,b; Nguyen et al.,
2016). In real-world applications, many informa-
tion needs inherently involve temporal dimensions,
including understanding how phenomena evolve
over time, comparing historical baselines with cur-
rent states, or tracking changes across multiple time
periods (Campos et al., 2014; Joho et al., 2014).
1Our code and data are available at https://github.
com/tempo-bench/Tempo . See also our official website:
https://tempo-bench.github.io/.
Figure 1:TEMPO combines temporal reasoning with
reasoning-intensive retrieval.TEMPO introduce com-
plex temporal reasoning with cross-period analysis, step-
wise retrieval planning, and specialized temporal met-
rics.
For instance, a cryptocurrency developer might ask
"How have block reorganizations changed since
2017?", a legal researcher might need to understand
"How has privacy law evolved after GDPR?", or an
economist might investigate "What were the trends
in quantitative easing before and after the 2008 fi-
nancial crisis?". In these scenarios, the temporal
aspects are not merely supplementary; they carry
essential information that fundamentally changes
what constitutes a relevant document.
Despite the prevalence of such complex tempo-
ral information needs, existing benchmarks remain
inadequate. Temporal QA benchmarks (Chen et al.,
2025; Wei et al., 2025; Mandal et al., 2025) largely
emphasize fact-seeking questions where identi-
fying a date or timestamp is sufficient, whereas
reasoning-intensive retrieval benchmarks such as
BRIGHT (Su et al., 2024) and RAR-b (Xiao et al.,
2024) do not explicitly model or evaluatetempo-
ral grounding—i.e., whether retrieved evidence is
aligned with the required time periods and supports
cross-period comparison (Figure 1). This leaves a
critical gap: how do retrieval systems perform on
queries that aresimultaneouslyreasoning-intensive
and temporally grounded, requiring retrieval of top-
ically relevant evidenceandtemporally appropriatearXiv:2601.09523v1  [cs.IR]  14 Jan 2026

Benchmark #Q #D Src. Temp. Reason. Expert Step Cross
Reasoning-Intensive Retrieval Benchmarks
BRIGHT 1,384 12 Mixed✗ ✓ ✓ ✗ ✗
RAR-b 45,745 17 Mixed✗ ✓ ✓ ✗ ✗
Temporal IR Benchmarks
NTCIR Temporalia 100 Open News/Blogs✓ ✗ ✗ ✗ ✗
Temporal QA Benchmarks
TempQuestions 1,271 Open Freebase✓ ✗ ✗ ✓ ✗
ChronoQA 5,176 Open News (CN)✓ ✗ ✗ ✗ ✗
TIME 38,522 3 Wiki/News/D✓ ✗ ✗ ✗ ✗
HistoryBankQA 535K 10 Wikipedia✓ ✗ ✗ ✗ ✗
ComplexTempQA 100M+ Open Wikipedia✓ ✗ ✗ ✓ ✗
TEMPO (Ours) 1,730 13 Stack Exch.✓ ✓ ✓ ✓ ✓
Table 1: Comparison of TEMPO with existing
temporal reasoning and retrieval benchmarks. TEMPO
combines temporal reasoning, complex retrieval,
and step-wise evaluation in different domains.Col-
umn legend:Src.=Source Data; Temp.=Temporal
Reasoning; Reason.=Reasoning-Intensive; Ex-
pert=Technical/Expert; Step=Multi-Hop/Step-Wise;
IR=Retrieval Task; Cross=Cross-Period Analysis.
coverage across periods?
In this work, we address this gap by introducing
TEMPO, a benchmark forreasoning-intensive re-
trieval with explicit temporal requirementsacross
13 domains. Prior temporal QA benchmarks pri-
marily evaluate answer generation and often re-
duce to locating a date, while existing reasoning-
intensive retrieval benchmarks (e.g., BRIGHT,
RAR-b) do not require temporal alignment or cross-
period evidence. TEMPO targets queries that are
simultaneouslyreasoning-intensive and temporally
grounded, emphasizing retrieval of topically rel-
evant evidence that is also appropriate across the
required time periods. It consists of 1,730 natu-
rally occurring Stack Exchange queries spanning
blockchain (Bitcoin, Cardano, IOTA, Monero), so-
cial sciences (Economics, Law, Politics, History),
applied domains (Quantitative Finance, Travel,
Workplace, Genealogy), and STEM (History of
Science and Mathematics).
To evaluate systems, we define two retrieval
tasks:(1) Query →Documents: Traditional
temporal retrieval with 1,730 queries, evaluat-
ing whether systems can retrieve temporally rel-
evant documents that address complex temporal
information needs;(2) Query →Step→Docu-
ments: Multi-step temporal reasoning with 1,605
queries decomposed into 3,976 retrieval steps, test-
ing whether systems can follow step-wise retrieval
plans where each step targets specific time periods
or aspects of the query.
Our key contributions include three novel com-
ponents that distinguish TEMPO from prior work.First, we providecomprehensive temporal an-
notationsat three levels: (i)query-level with 10
fine-grained temporal reasoning classes(e.g., trend
changes and cross-period, event analysis and local-
ization, causation analysis), temporal intent clas-
sification, temporal signals, and key time anchors;
(ii)step-wise retrieval planningthat decomposes
queries into sequential steps mapped to gold doc-
uments, enabling multi-hop temporal evaluation;
and (iii)passage-level annotationswith temporal
signals, events, and ISO-formatted time scopes.
Second, we introduce new temporal evaluation met-
rics designed to capture temporal reasoning aspects
missed by traditional IR metrics: Temporal Preci-
sion@k uses LLM-as-judge evaluation to measure
temporal relevance quality; Temporal Coverage@k
assesses whether top-k results span required base-
line and comparison time periods for cross-period
queries. Third, ourreal-world complex queries
average approximately 300 words, featuring rel-
evant documents and challenging hard negatives
mined through multi-LLM query reformulation and
web search.
2 Related Work
Early work on temporal query classification (Joho
et al., 2014, 2016; Campos et al., 2014) produced
benchmarks like NTCIR Temporalia. They rely
on news/blog corpora where timestamps and ba-
sic temporal expressions suffice. As illustrated
in Table 1, existing temporal QA benchmarks
have made progress but remain limited for evaluat-
ing temporally grounded retrieval. Recent efforts
include TempQuestions (Jia et al., 2018), Com-
plexTempQA (Gruber et al., 2025), TIME (Wei
et al., 2025), HistoryBankQA (Mandal et al., 2025),
and ChronoQA (Chen et al., 2025). While these
datasets go beyond simple timestamp lookup (e.g.,
temporal constraints, ordering, and cross-time com-
parisons), they are primarilyanswer-generation
benchmarks (Piryani et al., 2025; Wallat et al.,
2025; Abdallah et al., 2025d; Qian et al., 2024; Xu
et al., 2024; Brown et al., 2025) and do not explic-
itly evaluate whether retrieval results providetem-
porally alignedevidence that covers all required
time periods for cross-period analysis. We provide
more description of related work in Appendix A
and H.

Figure 2: Overview of TEMPO construction. We collect temporally grounded Stack Exchange queries, curate and
verify positive evidence (from answer links and Gemini-assisted web search), and mine hard negatives via GPT-4o
queries targeting topically similar but temporally mismatched documents.
3 TEMPO Dataset
In this section, we first formulate the task (§3.1),
then detail the data collection and annotation pro-
cess from Stack Exchange (§3.2). Data statistics
are presented in Table 2.
3.1 Task Formulation
Given a temporal query Q=Q textand a retrieval
corpus D={D 1, . . . , D n}, retrievers are tasked
to find temporally relevant documents D+
Q=
{D+
Q,1, . . . , D+
Q,m} ⊂ D where m≪n . Nega-
tive documents are defined as D−
Q=D \ D+
Q. In
temporal reasoning-intensive retrieval, the relevant
document set D+
Qis connected to query Qthrough
temporal reasoning traces involving temporal evo-
lution understanding, cross-period analysis, and
temporal dependency resolution, rather than simple
keyword matching or date filtering.
Query-Level Annotations.Each query Qis
annotated with a tuple AQ= (τ,S Q,EQ, ρ,P,T)
where: τ∈ I denotes temporal intent from the
setI={when, duration, order, before_after,
ongoing_status, period_definition, timeline} ;
SQ={s 1, . . . , s k}is the set of temporal sig-
nals (e.g., “since 2017”, “before the war”);
EQ={e 1, . . . , e l}is the set of temporal events;
ρ∈ R is the primary temporal reasoning
class from 10 categories (see Appendix F.2);
P={(p 1,D+
p1), . . . ,(p j,D+
pj)}is the step-wise
retrieval plan mapping each step pito its gold
documents D+
pi; andT={t 1, . . . , t h}is the set of
key time anchors.
Passage-Level Annotations.Each docu-
ment D∈ D is annotated with AD=(SD,ED,[ts, te], ϕ) where: SDandEDde-
note temporal signals and events in the pas-
sage; [ts, te]represents the temporal scope
as ISO-formatted start and end dates; ϕ∈
{past,present,future,mixed} indicates the
dominant tense.
3.2 Data Collection from Stack Exchange
StackExchange2is a community-driven platform
where domain experts ask and answer complex
technical questions. We select 13 diverse domains
spanning blockchain (Bitcoin, Cardano, IOTA,
Monero), social sciences (Economics, Law, Pol-
itics, History), applied fields (Quantitative Finance,
Travel, Workplace, Genealogy), and STEM (His-
tory of Science and Mathematics). StackExchange
posts we collect contain detailed temporal descrip-
tions requiring reasoning about how phenomena
evolved, changed over time, or differ across peri-
ods. We construct query-document pairs based on
user posts and documents referenced in answers
(Figure 2).
Selecting posts.Human annotators3browse
posts from newest to oldest and select posts that:
(1) have at least one answer that is either accepted
by the user or receives >10 votes, and (2) re-
quiretemporal reasoningas defined by our query-
level taxonomy ρ(Appendix F.2), e.g., event local-
ization, time-period contextualization, and cross-
period comparison/trend analysis. The distribution
of selected queries across these temporal reason-
ing categories is shown in Figure 4, and we later
analyze retrieval difficulty by category (Figure 7),
2https://stackexchange.com/
3Five PhD and two master students.

Total Number Avg. Length Avg.
Dataset QD D+QD Steps
Blockchain
Bitcoin 100 153,291 3.3 222.0 596.9 2.93
Cardano 51 87,201 2.5 161.1 647.2 2.84
Iota 10 10,372 3.8 148.6 1,036.5 3.20
Monero 65 85,093 2.6 171.8 703.3 2.72
Social Sciences
Economics 83 93,756 3.6 290.2 495.9 3.08
Law 35 43,288 3.0 258.5 500.7 3.23
Politics 150 183,394 2.7 343.2 476.3 3.35
History 801 356,493 4.5 374.2 682.7 3.42
Applied
Quant 34 28,785 2.4 422.5 477.1 2.68
Travel 100 177,677 2.6 264.5 374.8 3.11
Workplace 36 64,659 2.8 291.8 368.7 2.42
Genealogy 115 156,228 2.8 359.6 629.2 3.78
STEM
HSM 150 213,818 2.5 303.5 563.6 3.25
Total 1,730 1,654,055– – – –
Table 2: TEMPO dataset statistics across 13 domains.
Q: number of queries; D: corpus size (total docu-
ments);D+: average number of positive (relevant) doc-
uments per query; Avg. Length: average token count for
queries and documents; Steps: average number of steps.
Domains are grouped into four categories: Blockchain,
Social Sciences, Applied, and STEM.
where cross-period comparison queries are consis-
tently more challenging than single-period tempo-
ral localization. Detailed examples in Appendix I
Constructing query and positive documents.
For each selected post, annotators use the title and
body text to form the query Q. Annotators visit
web pages linked in the answers and use Gemini
(Google’s AI assistant) toreturn relevant web doc-
uments (which are not AI-generated)by prompt-
ing:"Give me articles from the internet to answer
this query: [post content]". For each discovered
web page, annotators extract passages that pro-
vide critical temporal information for answering
the query. Sources include Wikipedia, technical
blogs, research articles, official documentation, and
news sites. If no temporally relevant documents
are found, the post is discarded.
Constructing hard negative documents.To
prevent models from relying on simple semantic
matching, we ensure negative documents are top-
ically related but temporally incomplete or irrel-
evant. We use GPT-4o to analyze each post and
generate a search query designed to find hard neg-
atives, along with entities and events mentioned
in the post (prompt details in Appendix B.1). An-
notators use the generated query to search Google
and collect hard negative passages per query. They
extract passages that are topically related but do not
provide the temporal reasoning steps or time-period
coverage needed to answer the query.
80 82 84 86 88 90 92
Quality ScoreHSM
History
Cardano
Quant
Law
Economics
Genealogy
Travel
Politics
Iota
Workplace
Monero
Bitcoin89.1
88.7
87.3
87.3
87.1
86.8
86.6
86.3
86.1
85.7
85.7
85.5
84.3
Avg: 86.7Figure 3: Dataset quality validation using Qwen-72B as
LLM judge.
3.3 Temporal Annotation and Quality Control
Temporal annotations.For comprehensive tem-
poral evaluation, we annotate queries and passages
at multiple levels using GPT-4o, with human an-
notators reviewing a sample to ensure annotation
quality. Specifically, two expert annotators inde-
pendently verified a random sample of 200 queries
(11.6%) and their associated annotations, mea-
suring inter-annotator agreement using Cohen’s
Kappa. We achieved κ= 0.82 for temporal rea-
soning class assignment, κ= 0.78 for temporal
intent classification, and κ= 0.85 for gold docu-
ment relevance judgments. Additionally, we em-
ploy Qwen-72B as an independent LLM judge to
evaluate alignment between queries, retrieval steps,
and gold documents on a 0–100 scale. As shown
in Figure 3, TEMPO achieves an average quality
score of 86.7 across all domains (range: 84.3–89.1),
further validating annotation quality.
0100 200 300 400 500 600
Number of QueriesEAL
TPC
OEC
TCP
EVA
MAP
OTH
SMD
CAU
HAC624 (36.1%)
365 (21.1%)
256 (14.8%)
154 (8.9%)
115 (6.6%)
89 (5.1%)
58 (3.4%)
25 (1.4%)
23 (1.3%)
21 (1.2%)
Figure 4: Distribution of temporal reasoning classes.
See Appendix F.2 for class definitions.
At thequery level, we extract temporal intent
(when/duration/order/before_after/ongoing_status/
period_definition/timeline/none), temporal signals

(phrases like "since 2017", "nowadays"), temporal
events, temporal reasoning class (10 fine-grained
categories), retrieval plan with sequential steps,
key time anchors, expected granularity, and
quality checks (prompt in Appendix B.2). At
thepassage level, we extract temporal signals,
temporal events, time mentions, time scope
(start/end ISO dates with granularity), tense,
and confidence score (prompt in Appendix B.3).
This multi-level annotation enables fine-grained
temporal evaluation beyond traditional IR metrics.
Step-wise retrieval planning.ForTask 2, we
use the step-wise retrieval plans described above,
mapping each step to step-specific gold documents.
Each step describes a specific retrieval action
(e.g., "Retrieve historical baseline statistics from
2013-2017", "Retrieve current statistics from
2020-present") and is mapped to gold documents
that satisfy that step. This enables evaluation of
multi-hop temporal reasoning where systems must
retrieve evidence from multiple time periods in
sequence.
Temporal reasoning classification.We cate-
gorize queries into 10 fine-grained temporal rea-
soning classes based on the type of temporal in-
ference required (Figure 4; see Appendix F.2 for
full definitions). The dataset is dominated by
Event Analysis & Localization(EAL; 624 queries,
36.1%), which requires pinpointing when events
occurred and understanding their temporal context,
followed byTime Period Contextualization(TPC;
365 queries, 21.1%), which situates phenomena
within specific historical periods.Origins & Evo-
lution Comparative(OEC; 256 queries, 14.8%)
andTrends & Cross-Period Comparison(TCP; 154
queries, 8.9%) queries require tracking how con-
cepts evolved over time or comparing states across
periods. The remaining categories, including Event
Verification (EV A), Materials & Artifacts Prove-
nance (MAP), Sources & Methods Documentation
(SMD), Causation Analysis (CAU), and Histori-
cal Attribution & Context (HAC) represent more
specialized temporal reasoning patterns. We ana-
lyze retrieval performance across these reasoning
classes in §6.
Temporal Distribution.TEMPO spans Pre-
1900 to 2020+ (Figure 5), emphasizing cross-
period reasoning over simple news retrieval. The
distribution features historical queries: 806 queries
from Pre-1900 and 327 from 1900–49. However, it
maintains strong modern representation, including
143 queries for 2010–19 and 162 for 2020+. This
Pre-19001900-49 1950-79 1980-99 2000-09 2010-192020+
Time Period0250500750Number of Queries806
327
145
79 68143 162Figure 5: Overall distribution of query temporal anchors
across time periods in TEMPO.
breadth facilitates evaluating both long-term evolu-
tionary patterns and contemporary dynamics. See
Appendix G.2 for domain-specific breakdowns.
4 Temporal Evaluation Metrics
Traditional IR metrics (e.g., NDCG@k) do not mea-
sure whether retrieved evidence is temporally ap-
propriate or spans the required time periods. We
therefore report four temporal metrics computed
using an LLM-as-judge that labels (i) whether a
retrieved document is temporally relevant to the
query and (ii) which required period(s) it supports
(Appendix C).TP@k (Temporal Precision@k).
A position-weighted metric that rewards ranking
temporally relevant documents earlier in the top- k.
TR@k (Temporal Relevance@k).The fraction
of the top- kdocuments judged temporally relevant.
TC@k (Temporal Coverage@k).The fraction
of required time periods covered by at least one
document in the top- k(we use two periods in our
main experiments).NDCG|FC@k.NDCG@k
computed only on queries where full temporal cov-
erage is achieved (i.e., TC@k= 1).
5 Experimental
5.1 Experimental Setup
We evaluate 12 representative retrieval models
spanning sparse, dense, and reasoning-enhanced ar-
chitectures. All experiments use NDCG@10 as the
primary metric following prior IR benchmarks (Ab-
dallah et al., 2025c; Thakur et al., 2021; Nguyen
et al., 2016; Su et al., 2024). While we focus
on NDCG@10 for ranking evaluation, a compre-
hensive set of additional metrics—including Pre-
cision, Recall, Mean Average Precision (MAP),
and Mean Reciprocal Rank (MRR)—is provided
in Appendix F.4.

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Cardano13.4 13.1 12.1 29.3 35.721.7 14.6 20.6 18.6 22.9 21.4 28.1
Iota9.7 36.1 38.3 38.241.736.6 34.3 28.6 19.241.733.2 37.1
Monero2.8 14.5 9.9 20.3 20.0 14.7 16.9 11.0 21.0 19.6 15.123.7
Bitcoin6.2 14.4 13.3 17.4 16.319.115.7 11.4 14.9 16.3 14.3 17.6
Social Sci.
Economics5.8 12.6 16.327.825.0 17.2 17.5 17.1 22.7 20.0 15.3 21.9
Law12.7 31.9 28.1 40.4 34.0 38.3 37.3 32.0 33.5 37.9 33.840.8
Politics32.7 28.2 31.6 45.5 47.941.4 32.6 38.1 32.4 35.4 34.6 44.9
History9.2 27.4 26.534.528.7 27.3 28.5 25.6 25.8 34.3 28.7 32.4
Applied
Quant2.5 11.7 11.1 27.2 13.8 21.6 14.6 12.727.819.5 15.7 16.8
Travel4.6 23.8 23.7 26.8 28.3 25.0 25.0 22.0 26.1 21.4 27.329.7
Workplace6.2 27.2 23.942.632.9 30.8 36.2 30.3 36.6 30.0 34.6 31.6
Genealogy13.3 22.0 24.935.633.5 26.9 24.6 25.3 18.7 30.3 23.5 31.7
STEM
HSM21.2 23.2 18.9 31.037.733.4 24.4 21.3 16.9 24.7 26.1 33.5
Avg.10.8 22.0 21.432.030.4 27.2 24.8 22.8 24.2 27.2 24.9 30.0
Table 3: NDCG@10 performance of retrieval models on TEMPO across 13 domains: Blockchain (Bitcoin, Cardano,
Iota, Monero), Social Sciences (Economics, Law, Politics, History), Applied (Quant, Travel, Workplace, Genealogy),
and STEM (HSM: History of Science and Mathematics). Avg. denotes the average score across all domains. The
best score on each domain is shown inboldand the second best is underlined .
Model TP@10 TR@10 TC@10 NDCG|FC@10
Sparse model
BM25 24.0 11.1 32.8 22.5
Open-sourced models (<1B)
BGE 53.7 34.0 66.1 25.3
Contriever 49.6 30.3 64.1 26.2
Inst-L 53.1 33.8 66.9 25.8
SBERT 54.1 35.0 67.2 26.9
Open-sourced models (>1B)
E5 53.5 31.1 63.233.4
GritLM 53.8 35.0 69.1 27.2
Qwen 46.2 28.8 61.0 29.9
Reasoning models (>1B)
DiVeR62.0 41.371.4 32.5
Rader 50.6 32.3 66.1 26.1
ReasonIR 57.4 38.2 72.428.8
Table 4: Temporal evaluation metrics at rank 10 aver-
aged across all domains. TP@10: Temporal Precision
(position-weighted precision of temporally relevant doc-
uments); TR@10: Temporal Relevance; TC@10: Tem-
poral Coverage; NDCG|FC@10: NDCG conditioned
on full temporal coverage.
Sparse retrieval.We use BM25 (Robertson
et al., 2009) as our lexical baseline, which re-
mains competitive on traditional retrieval bench-
marks despite its simplicity.Dense retrieval.
We evaluate BGE (Chen et al., 2024), Con-
triever (Izacard et al., 2021), E5-Mistral (Wang
et al., 2022), GritLM (Muennighoff et al., 2024),
Inst-L (Su et al., 2024), Qwen (Li et al., 2023),
SBERT (Reimers and Gurevych, 2019), and
SFR (Meng et al., 2024).Reasoning-enhanced
retrievers.Given TEMPO’s emphasis on temporal
reasoning, we evaluate three specialized modelsModel Baseline Strip Temporal-Only Normalized
Sparse
BM25 10.8 10.4 11.0 10.9
Dense (<1B)
BGE 22.0 19.5 9.6 22.1
Contriever 21.4 18.9 10.2 22.4
Inst-L 24.8 23.2 13.5 24.7
SBERT 24.9 23.3 8.7 23.5
Dense (>1B)
E5 30.4 27.0 14.2 29.1
GritLM 27.2 24.9 15.2 27.1
Qwen 22.8 21.2 17.2 23.3
SFR 30.0 27.8 16.3 29.5
Reasoning
DiVeR32.0 29.9 17.730.2
Rader 24.2 21.8 9.4 21.6
ReasonIR 27.3 25.1 13.835.3
Table 5: Study on temporal query variants (NDCG@10
averaged across all domains). Baseline: original queries;
Strip: temporal signals removed; Temporal-Only: only
temporal information retained; Normalized: explicit
temporal intent tags added.
designed to incorporate logical inference during
retrieval: DiVeR (Long et al., 2025), Rader (Das
et al., 2025), and ReasonIR (Shao et al., 2025).
Temporal metrics.We report TP@10, TR@10,
TC@10, and NDCG|FC@10; formal definitions
and the LLM judging prompts are provided in Ap-
pendix C. In our evaluation we instantiate TC@k
withM=2 periods (baseline vs. comparison), but
the definition generalizes toM>2.
Step-wise EvaluationEach Task 2 query qhas
Sqsteps{pq,i}Sq
i=1with step-specific gold docu-
ments D+
q,i. We evaluate at the step level by re-

trieving for each step and computing NDCG@10
against D+
q,i, then macro-averaging over steps and
queries. Step-Only uses only pq,i, Query+Step
usesq⊕p q,i, and Query+All retrieves once with
q⊕p q,1⊕···⊕p q,Sqand evaluates the same ranking
against every step.
BM25BGE
ContrieverDiVeRE5
GritLMInst-L Qwen Rader
ReasonIRSBERTSFR
Retrieval Model010203040NDCG@10 (%)
11.6
8.3
10.5
20.1
18.0
20.0
13.7
19.4
8.8
17.2
11.2
20.710.8
21.5
20.7
33.3
32.1
28.2
26.9
28.2
25.5
35.0
25.9
33.310.3
23.2
23.8
32.0
31.8
27.9
26.3
26.5
23.9
35.3
25.1
32.0Step-Only Query+Step Query+All
Figure 6: Comparison of step-wise retrieval strategies
across 12 retrieval models.
5.2 Main Results
Existing retrieval systems perform poorly on
TEMPO:Table 3 presents NDCG@10 perfor-
mance across all 13 domains. Reasoning-enhanced
models achieve the best results, with DiVeR lead-
ing at 32.0 NDCG@10, followed by E5 (30.4) and
SFR (30.0). The sparse baseline BM25 achieves
only 10.8, demonstrating the inadequacy of key-
word matching for retrieval that requires tempo-
ral reasoning. Dense retrievers show substantial
improvements (21.4–30.4 range), while reasoning-
enhanced models reach 27.3–32.0, highlighting the
value of explicit reasoning mechanisms. Domain
difficulty varies substantially. History achieves
only 34.5 NDCG@10 with DiVeR. Politics and
Law reach higher scores (47.9 and, 40.8 respec-
tively), suggesting differences in temporal reason-
ing complexity. Blockchain domains show mixed
results: Iota achieves 41.7 (ReasonIR), while Mon-
ero proves considerably harder at 23.7 (SFR).
Temporal Precision and RelevanceTable 4 re-
veals temporal reasoning capabilities through spe-
cialized metrics. DiVeR achieves the highest tem-
poral precision at 62.0%, indicating superior rank-
ing of temporally relevant documents. ReasonIR
follows at 57.4%, while standard dense retriev-
ers cluster around 53–54%. BM25 achieves only
24.0%, confirming keyword matching cannot effec-
tively identify temporal relevance. For TR@10,
DiVeR leads at 41.3%, followed by ReasonIR
(38.2%). BM25 manages only 11.1%, highlighting
the gap between sparse and neural approaches. Per-
domain analysis (Appendix C.2) shows the Quant
BM25BGE
ContrieverSBERTE5
Qwen Rader
ReasonIR
Retrieval Model02040NDCG@10 (%)Reasoning Class
EAL
TPCOEC
TCPEVA
MAPOTH
SMDCAU
HACFigure 7: NDCG@10 performance across temporal rea-
soning classes.
domain is particularly challenging, with DiVeR
achieving only 10.8% temporal relevance, suggest-
ing financial queries may require specialized tem-
poral reasoning patterns.
Step-wise Retrieval PlanningWe evaluate next
whether decomposing temporal queries into ex-
plicit reasoning steps improves retrieval (Task 2).
Figure 6 compares three strategies:Step-Only(re-
trieving with individual steps),Query+Step(query
concatenated with each step sequentially), and
Query+All(query with all steps combined). Step-
Only yields substantially lower performance (avg.
14.6 NDCG@10), confirming that isolated reason-
ing steps lack sufficient context for effective re-
trieval. Both query-augmented strategies achieve
comparable results, with Query+Step marginally
outperforming Query+All (avg. 26.4 vs. 25.9).
ReasonIR benefits most from step-wise planning,
improving from 17.2 (Step-Only) to 35.0–35.3, a
gain of over 18 points. Dense retrievers show simi-
lar patterns, BGE improves from 8.3 to 21.5–23.2
while BM25 remains largely unaffected (10.3–11.6
across strategies). Full per-domain results are pro-
vided in Appendix F.1.
6 Additional Analyses
Temporal signals show modest impact on re-
trieval performanceWe compare four query
variants:Baseline(original),Strip(temporal sig-
nals removed),Temporal-Only(only temporal
information retained), andNormalized(explicit
temporal intent tags added). Table 5 shows that
removing temporal signals causes modest degra-
dation (avg. 2.2 points), with BM25 virtually un-
changed and reasoning models being robust. The
Temporal-Onlyvariant reveals temporal informa-

tion alone is insufficient—DiVeR drops from 32.0
to 17.7 NDCG@10, demonstrating that temporal
reasoning requires topical grounding. Notably, the
Normalizedvariant produces dramatic improve-
ment for ReasonIR (35.3 NDCG@10, +8.0 points),
establishing it as best-performing when provided
structured temporal metadata.
OriginalGPT-4o
Llama-70B Qwen-72B Qwen-32B
DeepSeek-32B
LLM Models010203040NDCG@10
10.8
6.2
4.7
4.5
4.3
5.322.0
25.8
21.1
24.3
21.8
26.332.0
30.7
30.9
30.9
31.2
32.224.2
21.5
20.5
19.2
20.1
24.827.3
41.0
37.8
39.2
34.6
39.6BM25
BGEDiVeR
RaderReasonIR
Figure 8: Impact of LLM-based query reformulation on
retrieval performance.
Performance varies substantially across reason-
ing classesTo understand which types of tempo-
ral reasoning pose the greatest challenges, we eval-
uate all retrieval models across 10 reasoning classes
derived from our taxonomy. Figure 7 reveals
substantial variation in difficulty across reason-
ing types.Trends & Cross-Period Comparison
(TCP)emerges as the most challenging class, with
even DiVeR achieving only 23.9 NDCG@10, as
this task requires synthesizing information across
multiple time periods. In contrast,Historical At-
tribution & Context (HAC)proves comparatively
easier (SFR: 51.8), likely due to more explicit an-
swers in the corpus. Model performance profiles
differ markedly across reasoning types. DiVeR
demonstrates the most balanced performance, lead-
ing in 6 of 10 classes. SFR excels at fact-oriented
classes (HAC: 51.8) but underperforms on tempo-
ral synthesis tasks, while ReasonIR shows strength
inCausation Analysis (CAU)at 28.1. BM25’s
performance varies dramatically (4.3–23.3 range),
confirming that lexical matching depends heavily
on term overlap rather than temporal understanding.
Full results are given in Appendix F.2.
LLM-generated reasoning enhances reasoning-
aware retrievers.Inspired by reasoning-
augmented retrieval (Su et al., 2024), we prompt
six LLMs to generate step-by-step reasoning,Retriever Avg. Score
None (No Retrieval) 77.3
Oracle (Gold Docs)80.5
DiVeR 75.6
BGE 74.8
BM25 73.8
Table 6: RAG performance (answer correctness, 0–100)
averaged across 13 domains.
then concatenate with original queries. Figure 8
reveals striking asymmetry: ReasonIR benefits
dramatically (+13.7 NDCG@10 with GPT-4o),
while BM25 consistently degrades (10.8 →4.3–6.2)
as reasoning text dilutes lexical signals. DiVeR
remains stable (30.7–32.2), indicating internal-
ized reasoning that neither benefits from nor
requires external augmentation. Full results in
Appendix F.3.
Retrieval-Augmented Generation Evaluation
We finally evaluate whether improved retrieval
translates to better downstream QA performance.
We use Llama-3-70B-Instruct as the generator and
GPT-4o to score answers (0–100) based on refer-
ence coverage (details in Appendix G.3). Table 6
reveals that current retrievers fail to improve over
parametric knowledge: the no-retrieval baseline
(77.3) outperforms all retrieval-augmented config-
urations, with BM25 causing the largest degrada-
tion (−3.5 points). Oracle retrieval achieves 80.5
(+3.2), demonstrating substantial headroom. We
hypothesise that temporally incomplete retrieved
documents actively mislead the generator. Simi-
lar to observations in BRIGHT (Su et al., 2024),
this suggests that QA results may not always per-
fectly capture retrieval performance. This gap often
occurs because generator models may struggle to
integrate retrieved evidence effectively, or evalua-
tors may fail to recognize nuances in open-ended
answers. We further hypothesize that temporally
incomplete documents in TEMPO actively mislead
the generator; for complex temporal queries, re-
trieving wrong temporal evidence is worse than
retrieving nothing.
7 Conclusion
We introduced TEMPO, the first benchmark com-
bining temporal reasoning with reasoning-intensive
retrieval across 13 domains. TEMPO features
1,730 complex queries with step-wise retrieval plan-
ning and novel temporal metrics measuring cross-

period coverage. Evaluation of 12 retrieval sys-
tems reveals substantial challenges: the best model
achieves only 32.0 NDCG@10 and 71.4% tempo-
ral coverage, demonstrating that current systems
struggle to retrieve temporally complete evidence.
We believe TEMPO provides a testbed for advanc-
ing temporal reasoning in retrieval and RAG.
Limitations
Domain and Language Coverage.TEMPO fo-
cuses on 13 English-language domains from Stack
Exchange. While these domains span blockchain,
social sciences, applied fields, and STEM, they may
not fully represent all temporal reasoning scenar-
ios. Future work could extend to other languages
and domains such as medicine, legal case law, or
scientific literature.
Temporal Scope.Our queries reflect the tempo-
ral distribution naturally occurring in Stack Ex-
change posts, which may over-represent recent
time periods. Historical queries from decades or
centuries ago are less frequent, potentially limiting
evaluation of long-range temporal reasoning.
Annotation Methodology.While we employ
LLM-assisted annotation with human verification,
the temporal annotations (reasoning classes, re-
trieval steps) may contain errors. The LLM-as-
judge temporal metrics, though validated on sam-
ples, inherit limitations of current language models
in temporal understanding.
References
Abdelrahman Abdallah, Jamshid Mozafari, Bhawna
Piryani, and Adam Jatowt. 2025a. Dear: Dual-stage
document reranking with reasoning agents via llm
distillation.arXiv preprint arXiv:2508.16998.
Abdelrahman Abdallah, Bhawna Piryani, Jamshid
Mozafari, Mohammed Ali, and Adam Jatowt. 2025b.
How good are LLM-based rerankers? an empiri-
cal analysis of state-of-the-art reranking models. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2025, pages 5693–5709, Suzhou,
China. Association for Computational Linguistics.
Abdelrahman Abdallah, Bhawna Piryani, Jamshid
Mozafari, Mohammed Ali, and Adam Jatowt. 2025c.
Rankify: A comprehensive python toolkit for re-
trieval, re-ranking, and retrieval-augmented gener-
ation.arXiv preprint arXiv:2502.02464.
Abdelrahman Abdallah, Bhawna Piryani, Jonas Wal-
lat, Avishek Anand, and Adam Jatowt. 2025d. Tem-
pretriever: Fusion-based temporal dense passage re-trieval for time-sensitive questions.arXiv preprint
arXiv:2502.21024.
Mohammed Ali, Abdelrahman Abdallah, and Adam
Jatowt. 2025. Sustainableqa: A comprehensive
question answering dataset for corporate sustain-
ability and eu taxonomy reporting.arXiv preprint
arXiv:2508.03000.
Andrew Brown, Muhammad Roman, and Barry De-
vereux. 2025. A systematic literature review of
retrieval-augmented generation: Techniques, metrics,
and challenges.arXiv preprint arXiv:2508.06401.
Ricardo Campos, Gaël Dias, Alípio M Jorge, and Adam
Jatowt. 2014. Survey of temporal information re-
trieval and related applications.ACM Computing
Surveys (CSUR), 47(2):1–41.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Wenhu Chen, Xinyi Wang, and William Yang Wang.
2021. A dataset for answering time-sensitive ques-
tions.arXiv preprint arXiv:2108.06314.
Ziyang Chen, Erxue Min, Xiang Zhao, Yunxin Li, Xin
Jia, Jinzhi Liao, Jichao Li, Shuaiqiang Wang, Baotian
Hu, and Dawei Yin. 2025. a question answering
dataset for temporal-sensitive retrieval-augmented
generation.Scientific Data, 12(1):1855.
Debrup Das, Sam O’Nuallain, and Razieh Rahimi. 2025.
Rader: Reasoning-aware dense retrieval models. In
Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, pages
19981–20008.
Raphael Gruber, Abdelrahman Abdallah, Michael Fär-
ber, and Adam Jatowt. 2025. ComplexTempQA: A
100m dataset for complex temporal question answer-
ing. InProceedings of the 2025 Conference on Em-
pirical Methods in Natural Language Processing,
pages 9111–9123, Suzhou, China. Association for
Computational Linguistics.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jan-
nik Strötgen, and Gerhard Weikum. 2018. Tempques-
tions: A benchmark for temporal question answering.
InCompanion Proceedings of the The Web Confer-
ence 2018, WWW ’18, page 1057–1062, Republic
and Canton of Geneva, CHE. International World
Wide Web Conferences Steering Committee.
Hideo Joho, Adam Jatowt, and Roi Blanco. 2014. Ntcir
temporalia: a test collection for temporal information

access research. InProceedings of the 23rd Interna-
tional Conference on World Wide Web, WWW ’14
Companion, page 845–850, New York, NY , USA.
Association for Computing Machinery.
Hideo Joho, Adam Jatowt, Roi Blanco, Haitao Yu, and
Shuhei Yamamoto. 2016. Building test collections
for evaluating temporal ir. InProceedings of the 39th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR
’16, page 677–680, New York, NY , USA. Association
for Computing Machinery.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning.arXiv preprint arXiv:2308.03281.
Meixiu Long, Duolin Sun, Dan Yang, Junjie Wang,
Yue Shen, Jian Wang, Peng Wei, Jinjie Gu, and Ji-
ahai Wang. 2025. Diver: A multi-stage approach
for reasoning-intensive information retrieval.arXiv
preprint arXiv:2508.07995.
Biswadip Mandal, Anant Khandelwal, and Manish
Gupta. 2025. Historybankqa: Multilingual tempo-
ral question answering on historical events.arXiv
preprint arXiv:2509.12720.
Costas Mavromatis, Prasanna Lakkur Subramanyam,
Vassilis N Ioannidis, Adesoji Adeshina, Phillip R
Howard, Tetiana Grinberg, Nagib Hakim, and George
Karypis. 2022. Tempoqr: temporal question reason-
ing over knowledge graphs. InProceedings of the
AAAI conference on artificial intelligence, volume 36,
pages 5825–5833.
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming
Xiong, Yingbo Zhou, and Semih Yavuz. 2024.
Sfrembedding-mistral: enhance text retrieval with
transfer learning.Salesforce AI Research Blog, 3:6.
Niklas Muennighoff, SU Hongjin, Liang Wang, Nan
Yang, Furu Wei, Tao Yu, Amanpreet Singh, and
Douwe Kiela. 2024. Generative representational in-
struction tuning. InThe Thirteenth International
Conference on Learning Representations.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
2016. Ms marco: A human-generated machine read-
ing comprehension dataset.
Bhawna Piryani, Abdelrahman Abdullah, Jamshid
Mozafari, Avishek Anand, and Adam Jatowt. 2025.
It’s high time: A survey of temporal information
retrieval and question answering.arXiv preprint
arXiv:2505.20243.
Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou,
Xuhui Sui, Li Zhang, and Kehui Song. 2024. Timer4:
Time-aware retrieval-augmented large language mod-
els for temporal knowledge graph question answering.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
6942–6952.Xubo Qin, Jun Bai, Jiaqi Li, Zixia Jia, and Zilong Zheng.
2025. Reinforced query reasoners for reasoning-
intensive retrieval tasks. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 21261–21274.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
Preprint, arXiv:1908.10084.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.
Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muen-
nighoff, Xi Victoria Lin, Daniela Rus, Bryan
Kian Hsiang Low, Sewon Min, Wen-tau Yih,
Pang Wei Koh, and 1 others. 2025. Reasonir: Train-
ing retrievers for reasoning tasks.arXiv preprint
arXiv:2504.20595.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S Siegel, Michael Tang, and 1 others.
2024. Bright: A realistic and challenging bench-
mark for reasoning-intensive retrieval.arXiv preprint
arXiv:2407.12883.
Wang-Chiew Tan, Jane Dwivedi-Yu, Yuliang Li, Lam-
bert Mathias, Marzieh Saeidi, Jing Nathan Yan, and
Alon Halevy. 2023. Timelineqa: A benchmark for
question answering over timelines. InFindings of
the Association for Computational Linguistics: ACL
2023, pages 77–91.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:
A heterogenous benchmark for zero-shot evalua-
tion of information retrieval models.arXiv preprint
arXiv:2104.08663.
Jonas Wallat, Abdelrahman Abdallah, Adam Jatowt,
and Avishek Anand. 2025. A study into investi-
gating temporal robustness of llms.arXiv preprint
arXiv:2503.17073.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi
Zhuang, Haochen Tan, Zhijiang Guo, and Houfeng
Wang. 2025. Time: A multi-level benchmark for
temporal reasoning of llms in real-world scenarios.
arXiv preprint arXiv:2505.12891.
Chenghao Xiao, G Thomas Hudson, and Noura Al
Moubayed. 2024. Rar-b: Reasoning as retrieval
benchmark.arXiv preprint arXiv:2404.06347.
Kehan Xu, Kun Zhang, Jingyuan Li, Wei Huang,
and Yuanzhuo Wang. 2024. Crp-rag: A retrieval-
augmented generation framework for supporting

complex logical reasoning and knowledge planning.
Electronics, 14(1):47.

Appendix Contents
Appendix A Benchmark Comparison. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
Appendix B Dataset Construction Prompts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
B.1Hard Negative Mining Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
B.2Query-Level Annotation Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
B.3Passage-Level Annotation Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
Appendix C Temporal Metrics & Evaluation Prompts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
C.1Metric Definitions. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
C.2Temporal Intent Detection Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
C.3Temporal Relevance Judgment Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
C.4Temporal Evidence Judgment Prompt. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
C.5Detailed Temporal Metrics by Domain. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
C.6Metric Validation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
Appendix D Detailed Analysis & Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
D.1Step-wise Retrieval Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
D.2Reasoning Class Definitions & Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
D.3Query Reformulation Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
D.4Additional Retrieval Metrics. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
Appendix E Quality Assessment & RAG. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
E.1Dataset Quality Validation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
E.2Domain Temporal Distribution. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
E.3RAG Evaluation Details & Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
Appendix H Extended Related Work. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
Appendix I Dataset Examples. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
Appendix J Annotation Guidelines. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55

A Benchmark Comparison
Table 1 provides a comprehensive comparison of
TEMPO with existing temporal reasoning and re-
trieval benchmarks. We describe each benchmark
category and highlight TEMPO’s distinguishing
characteristics.
Reasoning-Intensive Retrieval Benchmarks.
BRIGHT (Su et al., 2024) introduced the first
benchmark requiring intensive reasoning for docu-
ment retrieval, featuring 1,384 queries across 12 do-
mains including economics, psychology, and cod-
ing. While state-of-the-art models achieve 59.0
nDCG@10 on standard benchmarks, they score
only 18.3 on BRIGHT, revealing significant gaps
in reasoning capabilities. RAR-b (Xiao et al., 2024)
extends this with 45,745 queries across 17 domains,
framing reasoning tasks as retrieval problems to
evaluate whether retrievers can solve reasoning
problems directly. However, both benchmarks lack
temporal grounding—queries do not require rea-
soning about time periods, temporal evolution, or
cross-period analysis.
Temporal IR Benchmarks.NTCIR Tempo-
ralia (Joho et al., 2014) established founda-
tional benchmarks for temporal information re-
trieval through two subtasks: Temporal Query
Intent Classification (classifying queries as
past/recency/future/atemporal) and Temporal In-
formation Retrieval (ranking documents by tem-
poral relevance). Built on the LivingKnowledge
News/Blog Corpus containing 3.8 million times-
tamped documents, Temporalia pioneered system-
atic temporal IR evaluation. However, it relies on
news/blog corpora with simple temporal queries
where document timestamps and basic temporal ex-
pressions suffice, lacking the reasoning complexity
required for technical domain queries.
Temporal QA Benchmarks.TempQues-
tions (Jia et al., 2018) provides 1,271 temporal
questions over Freebase, introducing a formal
definition of temporal questions covering explicit
(“in 2008”) and implicit (“during the Cold War”)
temporal expressions. Questions require decompo-
sition into sub-questions, but focus on knowledge
base QA rather than document retrieval.
ChronoQA (Chen et al., 2025) is a Chinese
benchmark with 5,176 questions constructed from
300,000+ news articles (2019–2024), designed for
evaluating temporal reasoning in RAG systems. Itcovers absolute, aggregate, and relative temporal
types but is limited to Chinese and news domain.
TIME (Wei et al., 2025) offers 38,522 QA pairs
across three sub-datasets: TIME-Wiki (Wikipedia),
TIME-News (news articles), and TIME-Dial (multi-
session dialogues), covering 11 fine-grained tem-
poral reasoning subtasks. While comprehensive in
task coverage, it focuses on LLM evaluation rather
than retrieval system assessment.
HistoryBankQA (Mandal et al., 2025) presents
a multilingual database of 10M+ historical events
from Wikipedia, generating 535K questions across
10 languages. It evaluates temporal reasoning
over historical events but focuses on factual recall
from encyclopedic content rather than reasoning-
intensive retrieval.
ComplexTempQA (Gruber et al., 2025) is the
largest temporal QA dataset with 100M+ synthetic
question-answer pairs from Wikipedia and Wiki-
data, spanning 1987–2023. It introduces a taxon-
omy of attribute, comparison, and counting ques-
tions requiring multi-hop reasoning. Despite its
scale, questions are synthetically generated and
focus on answer generation rather than document
retrieval.
TEMPO’s Distinguishing Characteristics.Un-
like existing benchmarks, TEMPO uniquely com-
bines: (1)Temporal + Reasoning-Intensive:
Queries require both temporal reasoning (tracking
changes, cross-period comparison) and deep do-
main understanding, addressing the gap between
temporal QA and reasoning-intensive retrieval; (2)
Domains: 13 specialized domains from Stack Ex-
change where domain experts pose naturally com-
plex questions, rather than news or encyclopedia
sources; (3)Step-Wise Retrieval: 3,976 decom-
posed retrieval steps with gold documents mapped
to each step, enabling multi-hop temporal evalu-
ation; (4)Cross-Period Analysis: Novel metrics
(Temporal Coverage@k) measuring whether sys-
tems retrieve evidence spanning both baseline and
comparison time periods; (5)Retrieval Focus: Em-
phasizes document retrieval over answer genera-
tion, aligning with real-world RAG system require-
ments.
B Dataset Construction Prompts
B.1 Hard Negative Mining Prompt
We use GPT-4o to analyze Stack Exchange posts
and generate search queries designed to find chal-
lenging negative documents that are topically re-

lated but temporally incomplete or irrelevant. The
complete prompt is shown in Figure 9.
The LLM generates a search query designed to
retrieve topically similar but temporally incomplete
content, along with entities and events that help
construct effective negative search queries. An-
notators use the generated query to collect hard
negative passages from Google search.
B.2 Query-Level Temporal Annotation
Prompt
We use GPT-4o to extract temporal characteristics
from queries. The complete prompt is shown in
10.
GPT-4o processes each query and outputs a JSON
object containing temporal intent classification,
temporal signals, reasoning class, and a step-wise
retrieval plan. This annotation enables fine-grained
evaluation of temporal reasoning capabilities.
B.3 Passage-Level Temporal Annotation
Prompt
We use GPT-4o to extract temporal information
from passages. The complete prompt is shown in
Figure 11.
GPT-4o processes each query-passage pair and
outputs a JSON object with temporal signals,
events, time scope with ISO-formatted dates, and
temporal granularity. This passage-level annotation
enables evaluation of whether retrieved documents
contain appropriate temporal evidence.
C Temporal Evaluation Metrics:
Definitions and Prompts
C.1 Temporal Evaluation Metrics
Traditional IR metrics like NDCG and Recall mea-
sure whether relevant documents are retrieved, but
they do not capture temporal reasoning quality,
specifically, whether retrieved documents contain
appropriate temporal information and cover re-
quired time periods. We introduce novel temporal
metrics that address these limitations, implemented
using LLM-as-judge evaluation.
Temporal Intent Detection.Before evaluating
temporal relevance, we use LLM-as-judge to deter-
mine whether a query requires temporal reasoning.
LLM classifies queries into temporal intents
(when/duration/order/before_after/ongoing_status/
period_definition/timeline/none) and identifies
temporal keywords. Only queries with detectedtemporal intent proceed to temporal evaluation
(prompt in Appendix C.2.1).
Temporal Precision@k (TP@k).Position-
weighted precision measuring the quality of tempo-
rally relevant documents using LLM judgment. For
each document diat rank i∈ {1, . . . , k} , we obtain
a binary verdict vi∈ {0,1} from the LLM indicat-
ing temporal relevance (prompt in Appendix C.2.2).
LetR={r 1, r2, . . . , r m} ⊆ {1, . . . , k} denote
the set of rank positions where documents are tem-
porally relevant (i.e.,v ri= 1). Then:
TP@k=1
|R|X
r∈R|{j∈R:j≤r}|
r(1)
where |{j∈R:j≤r}| counts the number of
relevant documents at or before rankr.
This rewards documents that are both tempo-
rally relevant and highly ranked, with higher weight
given to relevant documents appearing earlier in
the ranking.
Temporal Relevance@k (TR@k).Simple pro-
portion of temporally relevant documents in top- k:
TR@k=1
kkX
i=1vi (2)
where vi∈ {0,1} is the LLM’s verdict for doc-
ument at ranki.
Temporal Coverage@k (TC@k).For queries
that require evidence spanning multiple time pe-
riods, we define a set of required periods PQ=
{P1, . . . , P M}, where each Pmis a time interval
(e.g., a baseline period, a comparison period, or ad-
ditional historical intervals). For each retrieved doc-
ument diat rank i, an LLM judge predicts a binary
coverage vector ci∈ {0,1}M, where ci,m= 1if
dicontains evidence relevant to period Pm. We
compute cumulative period coverage up to rankk:
Cm(k) = max
i≤kci,m.(3)
Temporal Coverage@k is the fraction of required
periods covered in the top-kresults:
TC@k=1
MMX
m=1Cm(k).(4)
This metric ranges from 0 (no period covered) to
1 (all required periods covered). When M= 2 ,
TC@k reduces to the baseline/comparison formu-
lation used in prior work and in our main experi-
ments.

Prompt for Hard Negative Mining
You are a document annotator specializing in hard negative mining for information retrieval.
Your task: Analyze the following Stack Exchange question and extract key information for finding
challenging negative passages from web search.
Question Title:{title}
Question Body:{clean_body}...
Tags:{’, ’.join(tags)}
Generate a structured analysis with:
1. give me a query for hard negative to use it to search on google
2. All entities (people, places, organizations, concepts, technologies, etc.)
3. All events, actions, or processes mentioned
Output ONLY a valid JSON object with this exact structure:
{
"llm_summary": "give me a query for hard negative to use it to search on google...",
"entities_events": ["entity1", "entity2", "event1", "event2", ...]
}
Rules:
•Summary must be EXACTLY 32 words (count carefully)
•List ALL relevant entities and events
•Include technical terms, concepts, and domain-specific vocabulary
•Extract named entities (people, places, companies, technologies)
•Include temporal events and processes
•Output ONLY valid JSON, no explanations
Figure 9: Prompt used for mining hard negative documents. GPT-4o generates a search query and extracts entities
to find topically similar but temporally distinct content.
NDCG | Full Coverage.We compute
NDCG@k conditioned on full temporal coverage
across all required periods:
NDCG|FC@k=(
NDCG@k if TC@k= 1.0
NaN otherwise.
(5)
To ensure the reliability of our proposed tem-
poral metrics, we conducted a meta-evaluation us-
ing a set of control queries and documents. Table
11 demonstrates that our intent classifier correctly
identifies temporal grounding without false posi-
tives on atemporal queries. Furthermore, Table 12
and 13 confirm that our metrics correctly enforce
cross-period synthesis requirements and maintain
rank-sensitivity, respectively.
C.2 Temporal Evaluation Prompts
C.2.1 Temporal Intent Detection Prompt
We use GPT-4o to classify whether queries require
temporal reasoning and identify temporal character-
istics. The complete prompt is shown in Figure 12:GPT-4o classifies the query’s temporal intent and
extracts temporal keywords. Queries without tem-
poral intent ( has_temporal_intent=false ) receive
NaN for all LLM-based temporal metrics.
C.2.2 Temporal Relevance Judgment Prompt
For queries with detected temporal intent, we use
GPT-4o to judge whether each retrieved docu-
ment provides temporal information. The complete
prompt is shown in Figure 13:
GPT-4o outputs a binary verdict (1 = temporally
relevant, 0 = not temporally relevant) with explana-
tion. These verdicts are used to compute Temporal
Precision@k and Temporal Relevance@k.
C.2.3 Temporal Evidence Judgment Prompt
For cross-period queries requiring comparison
across time periods, we use a specialized prompt
that evaluates baseline and comparison period cov-
erage. The complete prompt is shown in Figure 14:
GPT-4o judges temporal relevance and sepa-
rately tracks baseline/comparison period coverage.

Prompt for Query-Level Temporal Annotation
You are an expert intemporalinformation retrieval. Analyze ONLY the QUERY below and produce
retrieval guidance and categories.
Goals:
1. Determine whether the query is temporal and classify its intent. (1) temporal_intent: one
of ["when","duration","order","before_after","ongoing_status",
"period_definition","timeline","none"] (2) query_temporal_signals: phrases indicating time
(e.g., "in 1914", "during", "after",
"first", "since", "today", "in the 18th century") (3) query_temporal_events: ONLY time-bound
events (e.g., "Battle of Hastings",
"signing of Treaty X", "election of Y"). Exclude generic actions unless anchored in time.
2. Provide a compact, specific plan to retrievetemporalevidence.
3. Identify time anchors, expected granularity, and sanity checks.
Allowed temporal reasoning classes(choose one primary, optional secondaries):
•"event_analysis_and_localization"
•"time_period_contextualization"
•"event_verification_and_authenticity"
•"sources_methods_and_documentation"
•"materials_artifacts_and_provenance"
•"trends_changes_and_cross_period"
•"origins_evolution_comparative_analysis"
•"historical_misinterpretation_or_reenactment"
•"causation_analysis"
•"artifact_verification"
•"historical_attribution_and_context"
CRUCIAL RULES:(1) Use only the QUERY content (do NOT assume any passage). (2) All arrays must
be present even if empty. Use "" for missing strings. (3) Return ONLY one JSON object with EXACT
keys and value types below.
JSON schema to output:
{
"is_temporal_query": true,
"temporal_intent": "when",
"query_temporal_signals": ["..."],
"query_temporal_events": ["..."],
"query_summary": "summary of the query <=50 words",
"temporal_reasoning_class_primary": "time_period_contextualization",
"temporal_reasoning_class_secondary": ["materials_artifacts_and_provenance"],
"retrieval_reasoning": "explanation of how to retrieve temporal evidence",
"retrieval_plan": [
{"step": 1, "action": ".."},
{"step": 2, "action": ".."}
],
"key_time_anchors": ["..."],
"expected_granularity": "date",
"quality_checks": ["cross-check dates from multiple sources", "prefer primary/authoritative
sources"]
}
QUERY:
{query}
Figure 10: Query-level temporal annotation prompt. The model identifies temporal intent, signals, and reasoning
classes, and generates a step-wise retrieval plan.

Prompt for Passage-Level Temporal Annotation
You are an expert annotator fortemporalinformation retrieval.
Given a QUERY and a PASSAGE, do BOTH of the following:
1) Extract TEMPORAL info from the PASSAGE only:
•passage_temporal_signals: time cues (e.g., "in 1914", "during the 18th century",
"after the treaty")
•passage_temporal_events: ONLY time-bound events (battle/treaty/reign/election/founding).
Exclude non-temporal events.
•time_mentions: explicit or relative expressions (years, dates, centuries, eras, "after X",
"during Y")
•time_scope_guess:
–start_iso: ISO-like if visible (YYYY or YYYY-MM or YYYY-MM-DD), else ""
–end_iso: same format; "" if none
–granularity: one of ["date","month","year","decade","century","multicentury","unknown"]
•tense_guess: one of ["past","present","future","mixed","unknown"]
•confidence: 0.0–1.0
CRUCIAL RULES:
•Do NOT output any query-level fields here (no is_temporal_query, temporal_intent, etc.).
•Return empty lists (not null) when nothing is found.
•Return ONLY one JSON object with EXACT keys and value types below.
JSON schema to output:
{
"passage_temporal_signals": ["..."],
"passage_temporal_events": ["..."],
"time_mentions": ["..."],
"time_scope_guess": {"start_iso": "", "end_iso": "", "granularity": "unknown"},
"tense_guess": "past",
"confidence": 0.0
}
QUERY:
{query}
PASSAGE:
{passage}
Figure 11: Passage-level temporal annotation prompt. The model extracts temporal signals, events, and estimates
the ISO time scope of the retrieved passage.
These judgments enable computation of Temporal
Coverage@k and NDCG | Full Coverage metrics,
which are critical for evaluating cross-period tem-
poral reasoning.
C.3 Detailed Temporal Metrics
This section presents detailed per-domain results
for the four temporal metrics introduced in Sec-
tion C.1. All metrics are computed at rank 10
(k=10) and averaged across temporal queries within
each domain. Values are reported as percent-
ages, with best results inboldand second-best
underlined .C.3.1 Temporal Precision@10
Temporal Precision@10 (TP@10) measures the
position-weighted precision of temporally relevant
documents, rewarding systems that rank temporal
evidence higher. Table 7 shows per-domain per-
formance. Reasoning models (DiVeR, ReasonIR)
consistently achieve the highest scores across most
domains, with particularly strong performance in
History (55.8%, 57.9%), Politics (57.6%, 58.2%),
and Law (56.4%, 65.0%). BM25 struggles across
all domains, with particularly poor performance
in Monero (40.0%) and Quant (29.1%), domains
characterized by technical jargon and evolving ter-

Prompt for Temporal Intent Detection
Analyze if this query requires temporal reasoning to answer correctly.
Query:"{query}"
Temporal queries ask about:
•WHEN something happened (specific time/date)
•HOW LONG something takes/lasts (duration)
•RECENT events or changes (recency)
•Changes OVER TIME (temporal evolution)
•BEFORE/AFTER relationships (temporal ordering)
Respond ONLY with valid JSON in this exact format:
{
"has_temporal_intent": true/false,
"temporal_keywords": ["keyword1", "keyword2"],
"temporal_focus": "duration" or "specific_time" or "recency" or "change_over_time" or "none"
}
Examples:
Query: "When did Bitcoin Core introduce pruning?"
Output: {"has_temporal_intent": true, "temporal_keywords": ["when", "introduce"],
"temporal_focus": "specific_time"}
Query: "How long does Bitcoin Core store forked chains?"
Output: {"has_temporal_intent": true, "temporal_keywords": ["how long", "store"],
"temporal_focus": "duration"}
Query: "What are recent developments in Bitcoin storage?"
Output: {"has_temporal_intent": true, "temporal_keywords": ["recent", "developments"],
"temporal_focus": "recency"}
Query: "What is Bitcoin Core?"
Output: {"has_temporal_intent": false, "temporal_keywords": [], "temporal_focus": "none"}
Now analyze: "{query}"
Figure 12: Temporal intent detection prompt. The LLM judge classifies whether a query requires temporal reasoning
and identifies specific temporal keywords.
minology that keyword matching cannot capture
effectively.
C.3.2 Temporal Relevance@10
Temporal Relevance@10 (TR@10) measures the
proportion of retrieved documents that contain tem-
poral information relevant to the query. Table 8 re-
veals substantial variation across domains. DiVeR
achieves the highest average at 41.3%, with par-
ticularly strong performance in HSM (39.3%) and
Law (38.5%). Notably, the Quant domain proves
extremely challenging, with even DiVeR achiev-
ing only 10.8% temporal relevance, suggesting that
financial queries may require specialized tempo-
ral reasoning patterns. The large gap between
reasoning models (38–41%) and sparse retrieval
(11.1%) underscores the importance of understand-
ing query semantics for identifying temporal infor-
mation needs.C.3.3 Temporal Coverage@10
Temporal Coverage@10 (TC@10) evaluates
whether systems retrieve evidence from both the
baseline and comparison time periods for cross-
period queries. Table 9 shows that ReasonIR
and DiVeR achieve the highest average coverage
(72.4%, 71.4%), yet still fail to provide complete
temporal evidence for approximately 30% of cross-
period queries. Domain variation is substantial:
Law achieves 84.6% coverage (ReasonIR), while
Quant reaches only 38.2% (ReasonIR), a 46-point
gap. This suggests that legal and policy domains
may have more structured temporal discourse pat-
terns that facilitate cross-period retrieval, while
financial and quantitative analysis requires more
sophisticated temporal reasoning.
C.3.4 NDCG | Full Coverage@10
NDCG|Full Coverage@10 evaluates ranking qual-
ity specifically for queries where complete tempo-
ral evidence (both baseline and comparison peri-

Prompt for Temporal Relevance Judgment
Judge if a retrieved document helps answer the TEMPORAL aspects of a query.
Query:"{query}"
Temporal Focus:{temporal_focus}
Document:
{document}
Question:Does this document provide information that DIRECTLY helps answer the temporal aspects
of the query?
Guidelines:
•Verdict = 1 if document contains temporal information (dates, durations, time periods,
temporal sequences)
•Verdict = 0 if document lacks temporal information even if generally relevant
•For "when" queries: document must mention specific times/dates
•For "how long" queries: document must mention durations/time periods
•For "recent" queries: document must mention recency or recent dates
•Be STRICT: generic facts without temporal markers are NOT temporally relevant
Respond ONLY with valid JSON:
{
"verdict": 1 or 0,
"reason": "brief explanation",
"temporal_contribution": "what temporal information provided, or ’none’"
}
Figure 13: Temporal relevance judgment prompt. Used to compute Temporal Precision and Relevance by determin-
ing if a document addresses specific temporal aspects of the query.
ods) is retrieved in the top-10 results. Table 10
reveals that E5 achieves the highest average score
(33.4%), closely followed by DiVeR (32.5%). In-
terestingly, while reasoning models excel at en-
suring temporal coverage (previous subsection),
E5 produces superior rankings when all necessary
evidence happens to be retrieved. This suggests
complementary strengths: reasoning models better
identify what temporal evidence is needed across
time periods, while dense retrievers may better dis-
tinguish relevance gradations among temporally
appropriate documents. Domain-specific patterns
mirror those in standard NDCG@10, with Law
(40.5%, E5) and Workplace (34.4%, E5) showing
strong performance, while technically complex do-
mains like Monero (7.0%, DiVeR) remain challeng-
ing even with complete temporal coverage.
C.3.5 Discussion of Metric Utility
Standard metrics like NDCG measure topical rele-
vance but are often "temporally blind." For exam-
ple, in a query regarding Bitcoin’s evolution from
2017 to 2024, a BM25 retriever might achieve a
high NDCG by returning several documents about
Bitcoin from 2015. However, our TC metric wouldassign a low score (0.0 or 0.5) because the retrieved
evidence fails to span the required comparison pe-
riods. This meta-evaluation proves that our metrics
capture a unique dimension of retrieval quality es-
sential for RAG systems in technical and evolving
domains.
C.4 Metric Validation and Meta-Evaluation
To validate the scientific rigor of our proposed met-
rics—Temporal Precision (TP) and Temporal Cov-
erage (TC)—we performed a meta-evaluation us-
ing a suite of "golden cases." These cases test the
intent detection, temporal synthesis requirements,
and mathematical stability of the LLM-as-judge
framework.
C.4.1 Intent and Cross-Period Gating
Table 11 illustrates the system’s ability to distin-
guish between different temporal needs. Notably,
the system correctly identifies atemporal technical
definitions (e.g., "What is a Merkle tree?") as hav-
ing no temporal intent, preventing the inflation of
metrics on non-temporal data. It also successfully
identifies queries requiring cross-period analysis
(e.g., "since 2017").

Prompt for Temporal Evidence Judgment (Cross-Period Queries)
You are grading retrieved documents for a temporal trend/change query that needs cross-period
evidence.
Query:"{query}"
Temporal Focus:{temporal_focus}
Baseline anchor period:{baseline_anchor}
Comparison/current anchor period:{comparison_anchor}
Document:
{document}
Decide:
1. verdict: 1 if the document DIRECTLY helps answer the temporal aspects of the query
(contains relevant temporal info), else 0.
2. covers_baseline: true if the document contains evidence about the BASELINE anchor period.
3. covers_comparison: true if the document contains evidence about the COMPARISON/current
anchor period.
Strictness rules:
•A random date not related to the anchors does NOT count.
•"Currently/as of 2023/recent years" can count for comparison coverage if relevant.
•Baseline coverage should connect to the baseline anchor period (e.g., around 2017).
Return ONLY valid JSON:
{
"verdict": 1 or 0,
"reason": "brief explanation",
"temporal_contribution": "what temporal information provided, or ’none’",
"covers_baseline": true/false,
"covers_comparison": true/false
}
Figure 14: Cross-period temporal evidence judgment prompt. Used to compute Temporal Coverage by verifying if
documents cover both baseline and comparison periods.
C.4.2 Temporal Coverage Synthesis
Table 12 demonstrates how the TC metric enforces
evidence synthesis. A retriever cannot achieve a
perfect score by simply finding historical or current
data in isolation; it must provide a set of docu-
ments that cover both the baseline and comparison
periods.
C.4.3 Ranking and Mathematical Stability
Table 13 confirms the rank-sensitivity of our Tem-
poral Precision metric. By applying a position-
weighted calculation, the metric appropriately re-
wards systems that surface temporally relevant evi-
dence at higher ranks.
D Temporal signals Ablation Analysis
D.1 Performance Degradation by Domain
Table 14 shows the performance change (in
NDCG@10 points) when temporal signals are re-
moved from queries. Positive values indicate degra-dation; negative values (in blue) indicate improve-
ment. Models with smaller positive values demon-
strate better robustness to missing temporal infor-
mation. Notably, several model-domain combina-
tions show improved performance when temporal
signals are removed, suggesting that explicit tempo-
ral phrases may sometimes introduce noise. BM25
shows remarkably stable performance across do-
mains (average +0.4), while E5 shows the highest
average degradation (+3.4), with particularly large
drops on Cardano (+10.3) and Genealogy (+6.3).
D.2 Per-Domain Results by Query Variant
This section presents complete per-domain
NDCG@10 results for each query variant, enabling
analysis of domain-specific sensitivity to temporal
information.
D.2.1 Baseline (Original Queries)
Results using original TEMPO queries containing
natural temporal expressions, topical content, and

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT
Blockchain
Bitcoin26.2 52.7 42.361.255.9 55.6 52.8 41.9 50.5 56.4 52.1
Cardano27.2 49.6 37.156.549.9 50.6 44.1 44.4 46.4 50.1 52.2
Iota10.0 21.2 21.6 19.7 29.8 19.5 23.7 9.5 12.0 27.332.9
Monero12.7 40.0 29.947.740.3 45.5 36.2 30.2 34.8 42.3 32.5
Social Sci.
Economics17.5 50.3 44.763.856.3 48.8 50.7 44.5 45.4 50.4 44.6
Law41.9 56.4 56.565.860.4 58.2 54.9 57.3 50.4 61.2 60.6
Politics39.0 57.6 55.664.461.7 62.7 55.5 56.5 55.8 57.6 59.8
History19.8 55.8 53.364.052.1 52.9 55.0 45.3 53.1 61.8 56.5
Applied
Quant16.9 29.1 29.3 33.841.831.3 29.0 33.2 33.4 39.3 32.3
Travel29.3 56.6 53.764.755.8 56.0 62.9 45.8 51.4 57.8 54.1
Workplace14.3 49.6 47.2 51.653.246.7 49.5 44.4 46.3 44.0 51.8
Genealogy19.6 43.3 43.458.347.9 50.2 44.7 47.0 48.7 51.8 46.5
STEM
HSM31.6 62.7 53.167.759.1 61.6 60.1 51.9 52.6 61.5 63.8
Avg.24.0 53.7 49.662.053.5 53.8 53.1 46.2 50.6 57.4 54.1
Table 7: TP@10 across all domains (grouped by category). Best inbold, second best underlined .
reasoning requirements. These serve as the refer-
ence point for comparing other query variants.
D.2.2 Strip (Temporal Signals Removed)
Results after removing explicit temporal signals
(dates, temporal phrases) from queries while pre-
serving topical content. Most models show modest
degradation (1.6-3.4 points on average), with some
domain-specific improvements indicating temporal
signals may introduce noise in certain contexts.
D.2.3 Temporal-Only (Only Temporal
Information Retained)
Results using only temporal information (time an-
chors, temporal events) with topical content re-
moved. All models experience substantial perfor-
mance drops (DiVeR: 32.0 →17.7), demonstrating
that temporal reasoning requires grounding in topi-
cal context.
D.2.4 Normalized (Explicit Temporal Intent
Tags)
Results with explicit temporal intent tags
(e.g., "TEMPORAL_INTENT=when; AN-
CHORS=[2015, 2023]") appended to original
queries. ReasonIR shows dramatic improvement to
35.3 NDCG@10 (+8.0 points), while other models
show minimal change, indicating that structured
temporal metadata benefits instruction-tuned
reasoning models specifically.E Temporal Reasoning Class Definitions
TEMPO queries are categorized into 10 temporal
reasoning classes based on the type of temporal
inference required. Each class represents distinct
patterns of temporal information needs commonly
encountered in technical and academic domains.
E.1 Event Analysis and Localization (624
queries, 36.1%)
Queries requiring identification of when specific
events occurred and understanding their temporal
context. These queries ask about the timing of
technical developments, policy implementations,
historical incidents, or procedural changes. Exam-
ples: "When did Bitcoin Core introduce transac-
tion pruning?", "When was the first recorded use
of proof-of-work in cryptocurrency?"
E.2 Time Period Contextualization (365
queries, 21.1%)
Queries requiring situating phenomena, practices,
or concepts within specific historical or contempo-
rary time periods. These queries seek to understand
what existed, was practiced, or was true during a
particular era. Examples: "What consensus mech-
anisms were available before 2015?", "How were
international borders managed during the Cold War
era?"

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT
Blockchain
Bitcoin14.9 38.3 30.747.339.5 39.8 38.0 29.8 33.7 40.1 37.6
Cardano13.0 31.1 20.239.835.2 35.9 29.8 29.1 32.8 33.0 31.3
Iota1.0 6.0 5.0 7.0 7.0 6.0 5.0 5.0 2.0 7.08.0
Monero6.4 23.7 18.030.525.0 28.4 22.7 18.2 23.9 26.4 19.6
Social Sci.
Economics9.2 31.1 26.439.532.9 33.8 30.7 27.9 29.4 35.5 32.6
Law17.9 38.5 30.943.036.7 38.2 36.1 38.2 33.3 42.1 36.1
Politics16.2 30.9 28.7 36.1 31.336.731.1 29.5 30.3 33.8 31.2
History8.9 36.9 34.544.729.4 35.8 36.9 29.7 34.7 43.5 38.3
Applied
Quant3.5 10.8 10.8 12.714.212.0 10.0 9.6 11.9 13.1 11.5
Travel14.7 37.4 37.645.440.5 39.2 40.7 28.6 35.5 37.5 37.8
Workplace3.7 27.8 18.528.921.9 25.6 26.7 19.3 26.7 26.3 24.4
Genealogy10.5 25.8 24.635.428.4 26.8 25.7 29.3 29.5 30.2 27.8
STEM
HSM15.1 39.3 31.746.033.8 40.8 37.3 32.7 35.0 41.8 43.7
Avg.11.1 34.0 30.341.331.1 35.0 33.8 28.8 32.3 38.2 35.0
Table 8: TR@10 across all domains (grouped by category). Best inbold, second best underlined .
E.3 Origins, Evolution, and Comparative
Analysis (256 queries, 14.8%)
Queries requiring tracking how concepts, technolo-
gies, policies, or practices emerged and evolved
over time, often comparing earlier and later forms.
These queries examine historical development, in-
cremental changes, and evolutionary patterns. Ex-
amples: "How has Bitcoin’s block size debate
evolved since 2015?", "How did voting systems
change from the 19th to 21st century?"
E.4 Trends, Changes, and Cross-Period
Comparison (154 queries, 8.9%)
Queries requiring comparison of states, statistics,
or conditions across distinct time periods, often
analyzing trends or identifying changes. These
queries explicitly contrast baseline and comparison
periods to assess temporal shifts. Examples: "How
has cryptocurrency adoption changed since 2017?",
"How do current immigration policies differ from
those in the 1990s?"
E.5 Event Verification and Authenticity (115
queries, 6.6%)
Queries requiring verification of whether events
occurred, validation of temporal claims, or assess-
ment of historical accuracy. These queries seek
authoritative evidence to confirm or refute tempo-
ral assertions. Examples: "Did Satoshi Nakamotoreally propose Bitcoin in 2008?", "Was there actu-
ally a treaty signed in 1648 ending the Thirty Years’
War?"
E.6 Materials, Artifacts, and Provenance (89
queries, 5.1%)
Queries requiring temporal information about phys-
ical or digital artifacts, their creation dates, prove-
nance, or historical authenticity. These queries
focus on when objects or documents were pro-
duced and their historical chain of custody. Exam-
ples: "When was this historical document dated?",
"What is the earliest known manuscript of this
text?"
E.7 Other (58 queries, 3.4%)
Queries with temporal elements that do not fit the
primary categories or combine multiple temporal
reasoning patterns in unique ways. These represent
edge cases or highly specialized temporal informa-
tion needs.
E.8 Sources, Methods, and Documentation
(25 queries, 1.4%)
Queries requiring temporal information about re-
search methods, documentation practices, or source
materials used in different time periods. These
queries ask about how information was recorded,
preserved, or analyzed historically. Examples:
"How were genealogical records maintained in

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT
Blockchain
Bitcoin37.9 56.2 53.367.754.7 61.8 53.1 40.9 57.6 67.2 60.3
Cardano25.0 54.5 41.7 62.5 50.0 54.5 54.2 58.3 50.0 50.063.6
Iota20.0 40.0 20.0 20.050.020.0 20.0 20.0 20.0 20.0 20.0
Monero28.0 42.0 32.0 44.0 34.056.044.0 30.0 40.0 52.0 34.0
Social Sci.
Economics36.4 68.5 64.580.971.8 70.5 72.7 72.7 68.5 70.0 65.2
Law57.7 84.6 73.1 79.288.580.8 88.5 80.8 79.2 83.3 83.3
Politics49.5 71.3 75.0 81.2 74.082.871.9 71.4 71.8 77.3 74.5
History26.9 72.5 71.8 75.6 64.0 71.3 72.0 61.7 70.179.673.4
Applied
Quant21.9 38.2 32.4 44.1 46.7 46.4 34.4 41.253.153.1 38.2
Travel34.8 40.9 39.6 43.8 43.5 45.7 45.7 44.0 37.5 47.6 50.0
Workplace0.0 36.4 31.8 36.4 36.4 27.3 25.0 40.0 40.927.8 31.8
Genealogy17.6 45.5 50.0 55.6 51.5 50.0 54.5 48.561.857.4 58.8
STEM
HSM41.8 82.4 75.0 83.3 74.185.584.3 78.2 81.2 84.5 78.3
Avg.32.8 66.1 64.1 71.4 63.2 69.1 66.9 61.0 66.172.467.2
Table 9: TC@10 across all domains (grouped by category). Best inbold, second best underlined .
the 18th century?", "What statistical methods were
available to economists in the 1960s?"
E.9 Causation Analysis (23 queries, 1.3%)
Queries requiring identification of temporal cause-
effect relationships, understanding what events or
conditions led to specific outcomes. These queries
explicitly probe causal chains with temporal di-
mensions. Examples: "What caused the Bitcoin
price surge in 2017?", "What events led to the 2008
financial crisis?"
E.10 Historical Attribution and Context (21
queries, 1.2%)
Queries requiring attribution of ideas, inventions,
or discoveries to specific individuals or groups
within temporal context. These queries ask about
who did what when and under what historical cir-
cumstances. Examples: "Who first proposed the
concept of blockchain and when?", "Which mathe-
matician first proved this theorem in the 19th cen-
tury?"

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT
Blockchain
Bitcoin4.3 6.4 13.0 13.716.015.6 10.1 4.3 12.7 12.3 11.8
Cardano0.0 22.8 4.8 27.134.00.0 13.8 27.1 16.3 14.2 17.1
Iota39.0 38.4 58.6 63.7 41.4 58.6 63.7 13.0 0.083.263.7
Monero3.6 7.0 10.7 13.9 16.8 15.9 15.9 12.3 10.417.214.5
Social Sci.
Economics11.2 11.4 16.8 22.5 28.018.6 14.1 16.9 17.6 18.0 11.4
Law21.1 40.5 30.148.631.8 35.3 32.5 35.4 40.6 35.0 36.2
Politics43.4 30.3 36.5 43.9 53.739.8 33.7 43.3 36.3 36.4 36.8
History19.3 28.5 28.433.829.2 26.5 27.7 32.7 27.5 33.4 28.9
Applied
Quant0.0 19.0 24.9 30.1 16.8 14.9 19.0 18.734.418.3 22.1
Travel0.0 21.3 17.9 24.8 20.4 21.2 21.8 18.436.79.5 22.7
Workplace– 34.4 34.0 30.0 47.8 40.3 27.756.032.5 19.1 38.8
Genealogy27.8 33.3 28.634.133.0 20.0 30.5 27.0 21.2 32.5 26.4
STEM
HSM26.9 22.2 15.9 27.733.128.4 24.1 24.1 19.6 20.0 24.6
Avg.22.5 25.3 26.2 32.5 33.427.2 25.8 29.9 26.1 28.8 26.9
Table 10: NDCG|FC@10 across all domains (grouped by category). Best inbold, second best underlined .
Table 11: Validation of Temporal Intent and Cross-Period Classification Logic.
Query Example Detected Intent Cross-Period
How have Bitcoin fees changed since 2017?change_over_timeTrue
What is a Merkle tree?noneFalse
Who was the president in 1920?specific_timeFalse
Recent evolution of the Lightning Network.change_over_timeTrue
Table 12: Temporal Coverage (TC) Decision Matrix for Trend Analysis Queries.
Document Type Covers Baseline Covers Comparison TC Score
Baseline Only True False 0.500
Comparison Only False True 0.500
Both (Full Coverage) True True1.000
Table 13: Mathematical Weighting of Temporal Precision (TP) atk= 5.
Scenario Verdict Array Temporal Precision
Rank 1 Hit[1,0,0,0,0]1.000
Rank 5 Hit[0,0,0,0,1]0.200
Double Hit (Top)[1,1,0,0,0]1.000

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin+0.8 +2.3 +1.5 +0.3 +0.9 +2.2 -0.1-0.8+0.6 +2.7 +2.2 +0.7
Cardano+3.8 -1.2 +2.8 +0.7 +10.3 +4.7 +0.2 +3.5-2.6-0.7 +0.0 +4.8
Iota -7.5+4.2 +8.7 +6.8 +5.8 +7.1 +3.8 +5.9 -0.8 +8.9 +3.3 +5.7
Monero +0.1+1.4 +0.5 +0.5 +2.7 +0.8 +2.9 +0.8 +5.1 +2.7 +1.4 +2.0
Social Sci.
Economics+1.4 +1.0 +2.2 +1.6 +2.1 +3.5 +1.9 +0.9 +3.1 +0.7+0.3+1.5
Law+3.8 +8.5 +6.0-0.2+4.4 +0.4 +3.1 +3.6 +4.7 +0.3 +3.7 +3.0
Politics+1.9 +1.7 +1.0 +0.4 +2.3 +0.4-0.3+0.6 +1.8 +2.4 +1.0 +1.1
History +0.4+2.1 +1.1 +2.4 +1.7 +1.9 +0.9 +1.8 +1.9 +1.9 +1.8 +2.0
Applied
Quant -0.1+1.8 +0.4 +4.2 +1.3 +2.3 +1.9 +1.3 +6.9 +3.4 +0.7 +2.3
Travel+0.1 +2.9 +2.1 +1.1 +2.5 +0.8 +1.6-0.0+4.0 +0.2 +0.7 +1.6
Workplace+0.2 +2.2 +1.0 +1.8 +1.9 +2.5 +0.6 -0.0 +3.7-1.9+3.8 -0.5
Genealogy -0.3+3.0 +3.4 +5.4 +6.3 +2.0 +2.8 +1.2 +2.3 +4.4 +0.7 +2.3
STEM
HSM +0.5+2.9 +2.0 +2.0 +2.7 +2.2 +1.3 +2.0 +0.5 +2.9 +1.6 +1.9
Avg. +0.4+2.5 +2.5 +2.1 +3.4 +2.4 +1.6 +1.6 +2.4 +2.1 +1.6 +2.2
Table 14: Performance degradation (Baseline - Strip) in NDCG@10 points across domains. Positive values indicate
performance drop when temporal signals are removed. Best retention (smallest drop) inbold.
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin6.2 14.4 13.3 17.4 16.319.115.7 11.4 14.9 16.4 14.3 17.6
Cardano13.4 13.1 12.1 29.3 35.721.7 14.6 20.6 18.6 22.9 21.4 28.1
Iota9.7 36.1 38.3 38.2 41.7 36.6 34.3 28.6 19.241.833.2 37.1
Monero2.8 14.5 9.9 20.3 20.0 14.7 16.9 11.0 21.0 19.7 15.123.7
Social Sci.
Economics5.8 12.6 16.327.825.0 17.2 17.5 17.1 22.7 20.0 15.3 21.9
Law12.7 31.9 28.1 40.4 34.0 38.3 37.3 32.0 33.5 38.0 33.840.8
Politics32.7 28.2 31.6 45.5 47.941.4 32.6 38.1 32.4 35.4 34.6 44.9
History9.2 27.4 26.534.528.7 27.3 28.5 25.6 25.8 34.4 28.7 32.4
Applied
Quant2.5 11.7 11.1 27.2 13.8 21.6 14.6 12.727.819.5 15.7 16.8
Travel4.6 23.8 23.7 26.8 28.3 25.0 25.0 22.0 26.1 21.5 27.329.7
Workplace6.2 27.2 23.942.632.9 30.8 36.2 30.3 36.6 30.0 34.6 31.6
Genealogy13.3 22.0 24.935.633.5 26.9 24.6 25.3 18.7 30.3 23.5 31.7
STEM
HSM21.2 23.2 18.9 31.037.733.4 24.4 21.3 16.9 24.7 26.1 33.5
Avg.10.8 22.0 21.432.030.4 27.2 24.8 22.8 24.2 27.3 24.9 30.0
Table 15: NDCG@10 for Baseline query variant across all domains. Best inbold, second best underlined .

Table 16: NDCG@10 for Strip query variant across all domains. Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin5.4 12.1 11.817.115.5 16.9 15.7 12.2 14.3 13.6 12.0 16.9
Cardano9.6 14.3 9.328.525.4 17.1 14.4 17.1 21.3 23.6 21.4 23.3
Iota17.2 31.8 29.6 31.435.929.5 30.5 22.7 20.0 32.9 29.9 31.4
Monero2.7 13.1 9.4 19.8 17.3 13.9 14.0 10.2 16.0 17.0 13.621.7
Social Sci.
Economics4.4 11.6 14.126.122.9 13.7 15.5 16.2 19.6 19.3 15.1 20.4
Law8.9 23.4 22.140.629.7 38.0 34.3 28.5 28.8 37.6 30.1 37.8
Politics30.8 26.5 30.7 45.1 45.540.9 32.9 37.5 30.6 33.1 33.6 43.9
History8.8 25.3 25.4 32.1 27.0 25.4 27.6 23.8 24.032.526.9 30.3
Applied
Quant2.6 9.9 10.722.912.5 19.3 12.8 11.4 20.9 16.1 14.9 14.6
Travel4.5 20.9 21.5 25.7 25.7 24.2 23.4 22.0 22.1 21.2 26.6 28.1
Workplace6.0 24.9 23.040.731.0 28.3 35.6 30.3 32.9 31.9 30.9 32.1
Genealogy13.6 19.1 21.530.127.2 25.0 21.8 24.1 16.4 26.0 22.8 29.4
STEM
HSM20.7 20.2 16.9 29.035.131.3 23.0 19.3 16.3 21.8 24.5 31.5
Avg.10.4 19.5 18.929.927.0 24.9 23.2 21.2 21.8 25.1 23.3 27.8
Table 17: NDCG@10 for Temporal-Only query variant across all domains. Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin4.3 5.9 6.111.89.0 8.5 7.9 6.7 6.6 8.5 6.0 10.3
Cardano14.4 7.7 7.0 15.3 17.810.4 8.3 10.6 8.9 9.5 10.1 13.9
Iota5.3 17.6 17.823.714.8 17.8 22.4 22.3 12.0 14.2 14.7 16.3
Monero4.4 3.7 7.9 7.9 5.612.98.3 8.1 4.1 10.0 6.3 10.7
Social Sci.
Economics9.8 6.5 5.8 16.7 14.4 11.9 10.317.311.4 12.1 5.7 15.6
Law14.2 15.7 15.6 22.7 15.2 23.2 18.427.913.5 27.9 13.1 20.4
Politics24.4 9.5 9.0 23.6 22.5 22.9 16.1 22.6 9.4 13.6 9.225.3
History12.4 9.8 9.6 14.1 8.8 13.9 13.016.16.1 12.3 9.0 12.9
Applied
Quant5.3 6.2 11.4 14.1 7.7 9.0 10.815.910.9 14.7 5.6 9.5
Travel9.8 13.6 17.3 17.5 12.0 15.0 14.4 17.5 9.418.110.2 16.5
Workplace8.3 6.9 7.029.319.1 15.6 15.5 22.5 12.6 17.0 3.8 21.4
Genealogy12.9 11.7 10.0 15.417.615.2 14.3 16.9 9.5 9.6 8.9 17.5
STEM
HSM17.6 9.8 8.3 17.5 20.221.815.6 19.7 7.4 11.8 10.6 21.4
Avg.11.0 9.6 10.217.714.2 15.2 13.5 17.2 9.4 13.8 8.7 16.3

Table 18: NDCG@10 for Normalized query variant across all domains. Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin6.6 13.0 12.3 13.7 14.419.014.3 11.2 13.4 18.6 13.5 16.6
Cardano13.5 15.2 15.3 25.7 36.4 19.4 12.6 19.0 20.541.619.9 27.4
Iota8.3 35.9 37.3 39.0 41.7 38.5 34.6 32.8 21.144.731.8 37.0
Monero2.6 14.5 11.6 17.3 15.0 15.7 18.9 11.3 15.025.215.1 22.3
Social Sci.
Economics6.2 12.4 17.9 24.3 25.0 16.2 16.9 18.6 19.532.415.0 22.9
Law12.4 31.1 28.9 36.0 29.9 39.5 36.9 30.5 31.645.029.9 38.2
Politics32.8 29.0 33.3 46.6 49.141.5 34.0 40.1 30.9 42.6 35.2 45.9
History9.4 27.1 26.3 32.6 25.9 27.0 28.1 25.3 19.939.528.1 30.7
Applied
Quant2.5 13.2 12.4 24.9 13.0 20.7 14.3 13.4 23.326.014.1 15.6
Travel4.7 23.2 23.2 24.5 25.5 24.0 23.5 23.5 20.329.025.8 28.8
Workplace7.5 26.9 26.1 45.2 32.3 30.1 37.7 29.8 34.847.030.0 31.3
Genealogy13.2 21.8 26.5 32.2 33.9 27.5 24.6 27.3 17.735.522.5 32.9
STEM
HSM22.2 23.8 19.5 30.936.733.0 24.6 19.7 13.2 31.5 24.7 34.1
Avg.10.9 22.1 22.4 30.2 29.1 27.1 24.7 23.3 21.635.323.5 29.5

F Detailed Analysis & Results
F.1 Step-wise Retrieval Results
This section presents detailed results for Task 2:
Step-wise Retrieval Planning, which evaluates how
effectively retrieval systems can leverage decom-
posed reasoning steps generated by a large lan-
guage model. Given the multi-step nature of tem-
poral reasoning queries, we investigate whether
explicitly decomposing queries into intermediate
reasoning steps can improve retrieval performance
compared to using the original query alone.
F.1.1 Experimental Setup
We generate reasoning steps for each query us-
ing GPT-4o, prompting it to decompose temporal
queries into logical retrieval steps. For example, a
query asking “How did Bitcoin’s consensus mech-
anism change after the 2017 fork?” might be de-
composed into steps such as: (1) identify Bitcoin’s
original consensus mechanism, (2) find information
about the 2017 fork event, and (3) locate documen-
tation of post-fork consensus changes.
We evaluate three retrieval strategies that vary in
how these reasoning steps are incorporated:
•Step-Only: Each reasoning step is used inde-
pendently as the retrieval query, completely
replacing the original question. This tests
whether the decomposed steps contain suf-
ficient information for retrieval without the
original query context.
•Query+Step: The original query is concate-
nated with each reasoning step sequentially.
Retrieval is performed for each step separately,
and the final score is computed by averaging
NDCG@10 across all steps. This approach
preserves query context while focusing on in-
dividual reasoning components.
•Query+All: The original query is concate-
nated with all reasoning steps combined into
a single augmented query. This creates a com-
prehensive query that includes both the origi-
nal information need and all decomposed rea-
soning components.
F.1.2 Overall Performance Comparison
Figure 6 presents the average NDCG@10 across
all 13 domains for each retrieval strategy. The re-
sults reveal substantial differences in how retrieval
models leverage reasoning steps.The Step-Only strategy yields the lowest perfor-
mance across nearly all models, with an average
NDCG@10 of 14.6 across all retrievers. This con-
firms that isolated reasoning steps, while logically
sound, lack sufficient context for effective docu-
ment retrieval. The decomposed steps often contain
generic sub-queries that match many irrelevant doc-
uments, leading to poor precision. SFR achieves
the best Step-Only performance at 20.7, followed
closely by DiVeR (20.1) and GritLM (20.0), sug-
gesting that larger embedding models can partially
compensate for missing query context.
Both query-augmented strategies achieve
substantially higher performance. Query+Step
achieves an average of 26.4 NDCG@10,
marginally outperforming Query+All at 25.9. This
slight advantage suggests that averaging across
individual step-augmented retrievals provides
more robust results than combining all steps into a
single dense query, potentially because the latter
can introduce noise from less relevant steps.
ReasonIR demonstrates the most dramatic
improvement from step-wise planning, increas-
ing from 17.2 NDCG@10 (Step-Only) to 35.0
(Query+Step) and 35.3 (Query+All)—a gain ex-
ceeding 18 absolute points. This indicates that
reasoning-enhanced retrievers are specifically de-
signed to exploit structured reasoning information.
DiVeR shows similar patterns, improving from 20.1
to 33.3 and 32.0 respectively.
In contrast, BM25 remains largely unaffected by
the retrieval strategy, achieving 11.6 (Step-Only),
10.8 (Query+Step), and 10.3 (Query+All). This
consistent performance across strategies confirms
that lexical matching cannot effectively leverage
the semantic information encoded in reasoning
steps. The slight performance decrease with query
augmentation may result from keyword dilution,
where additional terms reduce the relative weight
of critical temporal keywords.
F.1.3 Per-Domain Analysis
Tables 19, 20, and 21 present comprehensive per-
domain results for each strategy. Several domain-
specific patterns emerge from this analysis.
Workplace DomainThis domain consistently
yields the highest scores across all strategies and
models. DiVeR achieves 41.0 NDCG@10 with
Step-Only, increasing to 55.4 with Query+Step—
the highest single-domain score in our evaluation.
The structured nature of workplace-related tempo-

ral queries (e.g., policy changes, employment regu-
lations over time) appears particularly amenable
to step-wise decomposition. ReasonIR follows
closely with 35.8 to 54.6 across strategies.
Blockchain DomainsThe four blockchain do-
mains exhibit high variance in step-wise re-
trieval effectiveness. Iota achieves strong results
across all strategies, with ReasonIR reaching 45.0
NDCG@10 on Query+All. This may reflect the
relatively focused nature of Iota’s temporal evolu-
tion as a newer cryptocurrency. In contrast, Mon-
ero proves consistently challenging, with the best
Query+Step result at only 30.1 (ReasonIR). Bitcoin
shows moderate improvement from step augmenta-
tion (GritLM: 13.3 Step-Only →20.2 Query+Step),
while Cardano demonstrates substantial gains (Rea-
sonIR: 10.7→32.3→41.3 across strategies).
Social Science DomainsPolitics and Law ben-
efit substantially from query-augmented strate-
gies. Politics reaches 51.2 NDCG@10 with E5
on Query+All, while Law achieves 45.0 with Rea-
sonIR on the same strategy. These domains in-
volve complex temporal reasoning about legisla-
tive changes, policy evolution, and historical po-
litical events, where explicit reasoning steps help
disambiguate the information need. History shows
more modest gains, with ReasonIR improving from
15.9 (Step-Only) to 38.7 (Query+Step) to 40.1
(Query+All).
Applied DomainsQuant (quantitative finance)
presents an interesting case where Rader achieves
its best relative performance, reaching 36.8
NDCG@10 on Query+Step compared to 15.2 on
Step-Only. This 21.6-point improvement suggests
that financial temporal queries benefit significantly
from structured decomposition. Travel shows con-
sistent improvement across strategies, with SFR
achieving 31.2 on both Query+Step and Query+All.
Genealogy demonstrates moderate gains, with E5
reaching 38.6 on Query+Step.
STEM Domain (HSM)History of Science and
Mathematics shows substantial improvement with
query augmentation. E5 achieves 37.9 NDCG@10
on Query+All, a 22.8-point improvement over its
Step-Only performance of 15.1. This domain in-
volves tracing the evolution of scientific concepts
and mathematical discoveries, where temporal de-
composition helps identify relevant historical docu-
ments.F.1.4 Model Architecture Insights
The results reveal systematic differences based on
model architecture:
Sparse Retrieval (BM25)Lexical matching
shows minimal sensitivity to retrieval strategy, with
performance hovering around 10–12 NDCG@10
regardless of how reasoning steps are incorporated.
This confirms that BM25 cannot semantically inter-
pret reasoning steps and may suffer from keyword
dilution when queries are augmented.
Standard Dense RetrieversModels like BGE,
Contriever, and SBERT show substantial improve-
ments with query augmentation. BGE improves
from 8.3 (Step-Only) to 21.5 (Query+Step) and
23.2 (Query+All), demonstrating that even smaller
dense models can leverage structured reasoning
when combined with the original query context.
Large Dense RetrieversE5, GritLM, Qwen, and
SFR achieve strong performance across strategies.
E5 demonstrates consistent improvement from 18.0
to 32.1 to 31.8, while SFR shows a similar pattern
from 20.7 to 33.3 to 32.0. These models have
sufficient capacity to encode both the original query
semantics and the additional reasoning context.
Reasoning-Enhanced RetrieversReasonIR and
DiVeR, specifically designed for reasoning-
intensive retrieval, show the largest absolute gains
from step-wise planning. ReasonIR’s improvement
of 18+ points confirms that these architectures are
optimized to exploit structured reasoning informa-
tion. Interestingly, ReasonIR performs better on
Query+All (35.3) than Query+Step (35.0), unlike
most other models, suggesting it can effectively
synthesize all reasoning steps simultaneously.
F.1.5 Implications
These results have several implications for tempo-
ral reasoning-intensive retrieval:
First, query context is essential for step-wise
retrieval. Isolated reasoning steps, while logically
coherent, lack the specificity needed for accurate
document retrieval. Systems implementing step-
wise retrieval should always maintain the original
query alongside decomposed steps.
Second, the choice between Query+Step and
Query+All strategies depends on the retrieval
model. For most dense retrievers, Query+Step
provides marginally better results, possibly be-
cause averaging across steps reduces noise from

less relevant decompositions. However, reasoning-
enhanced models like ReasonIR can effectively
leverage all steps simultaneously.
Third, the substantial performance gap between
sparse and neural retrieval in step-wise settings
(BM25: 11 vs. ReasonIR: 35) suggests that se-
mantic understanding is crucial for interpreting rea-
soning steps. This gap is larger than observed in
standard retrieval (Table 3), indicating that step-
wise retrieval amplifies the advantage of neural
approaches.

Table 19: Task 2 Step-wise Retrieval: NDCG@10 usingStep-Onlystrategy (individual retrieval steps without the
original query). Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin4.1 5.4 6.1 11.5 8.913.36.3 10.0 5.3 10.0 7.7 12.1
Cardano9.1 4.1 1.5 11.114.98.4 3.3 8.6 3.6 10.7 6.6 14.2
Iota15.1 9.7 9.4 18.525.816.5 14.8 23.4 6.2 6.8 9.5 25.5
Monero7.4 3.1 11.2 12.0 5.721.19.8 10.7 3.7 13.3 9.0 13.8
Social Sci.
Economics8.6 5.5 7.2 19.0 14.4 14.5 9.119.28.2 13.2 7.5 18.5
Law17.3 13.2 15.8 26.7 19.8 26.4 20.9 25.8 15.229.412.4 26.2
Politics18.6 7.4 9.6 22.9 24.125.517.8 22.2 8.9 17.1 13.5 25.0
History10.9 7.2 10.219.310.6 17.4 14.6 17.7 6.4 15.9 10.7 15.9
Applied
Quant6.0 5.7 18.5 20.9 10.7 22.6 9.6 21.2 15.226.19.9 14.4
Travel6.7 9.3 15.7 19.5 17.624.416.7 18.2 9.3 15.3 16.1 23.4
Workplace19.4 16.4 11.441.038.6 28.7 22.6 38.6 15.9 35.8 18.1 38.1
Genealogy15.1 15.5 12.8 23.927.123.0 20.2 23.2 12.7 19.3 14.8 26.1
STEM
HSM12.5 5.1 6.9 14.5 15.118.611.9 13.5 3.5 11.3 10.3 16.1
Avg.11.6 8.3 10.5 20.1 18.0 20.0 13.7 19.4 8.8 17.2 11.220.7
Table 20: Task 2 Step-wise Retrieval: NDCG@10 usingQuery+Stepstrategy (query concatenated with each step
sequentially, then averaged across steps). Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin6.2 13.9 12.0 19.8 19.2 20.2 17.1 14.0 18.7 20.7 16.920.8
Cardano9.7 12.6 11.0 27.1 31.5 18.0 13.9 20.5 21.532.317.7 25.5
Iota13.8 31.4 29.8 31.1 41.0 33.2 35.142.116.9 38.7 28.142.1
Monero4.2 22.4 16.0 22.9 21.1 23.7 25.2 14.2 18.530.119.3 30.0
Social Sci.
Economics5.8 11.1 15.6 27.7 26.1 17.9 16.5 24.8 22.928.516.6 26.1
Law13.8 25.4 22.3 38.1 30.8 35.6 34.9 26.1 36.339.028.6 38.2
Politics28.1 24.6 30.648.247.8 40.2 35.3 41.5 32.9 40.8 35.5 45.8
History9.3 23.1 23.5 34.5 25.7 26.5 28.8 30.8 25.238.728.8 32.0
Applied
Quant1.9 16.0 13.3 33.0 19.1 23.2 20.8 23.136.834.0 17.9 24.4
Travel4.5 19.8 21.3 25.0 29.8 27.1 26.7 24.8 23.3 29.8 28.231.2
Workplace9.4 29.4 24.355.452.1 40.1 44.2 52.4 39.8 54.6 45.3 47.9
Genealogy14.1 27.2 31.140.038.6 28.9 29.3 29.5 24.7 37.6 27.7 37.5
STEM
HSM20.2 22.0 17.6 30.135.031.6 22.3 22.6 14.2 29.8 26.2 31.8
Avg.10.8 21.5 20.7 33.3 32.1 28.2 26.9 28.2 25.535.025.9 33.3

Table 21: Task 2 Step-wise Retrieval: NDCG@10 usingQuery+Allstrategy (query concatenated with all steps
combined). Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin5.9 15.0 13.8 17.2 18.3 18.1 15.1 11.3 17.5 17.9 14.118.6
Cardano12.1 14.8 14.7 25.5 38.0 24.1 16.3 19.0 19.941.322.1 30.7
Iota8.6 33.0 39.5 37.3 41.3 37.1 31.8 35.0 19.245.032.1 39.0
Monero1.7 17.7 12.4 17.6 16.2 15.6 17.6 11.1 15.5 21.4 14.423.4
Social Sci.
Economics5.5 13.8 18.9 28.9 27.3 18.0 19.4 24.6 24.632.416.8 25.0
Law10.5 31.8 31.0 41.8 34.3 41.3 39.8 31.5 36.245.033.4 41.2
Politics32.0 29.4 35.0 47.5 51.242.0 34.8 42.9 33.7 43.2 35.2 47.3
History8.2 27.8 28.0 33.8 25.7 27.4 29.7 31.1 23.940.128.5 31.5
Applied
Quant0.6 14.6 11.5 25.0 16.0 21.2 16.3 15.825.623.9 14.5 19.3
Travel5.1 25.3 25.9 25.0 29.4 27.7 26.9 27.8 22.3 30.3 27.731.2
Workplace9.3 31.3 30.6 50.3 40.1 31.6 42.5 44.7 38.452.540.0 36.4
Genealogy12.7 23.2 27.6 35.337.426.2 27.7 27.6 21.3 33.9 22.7 35.6
STEM
HSM22.1 24.4 20.9 30.937.932.9 24.6 22.0 12.2 32.2 24.8 36.6
Avg.10.3 23.2 23.8 32.0 31.8 27.9 26.3 26.5 23.935.325.1 32.0

F.2 Reasoning Class Analysis
This section provides detailed results for the rea-
soning class analysis presented in Section 6. We
evaluate retrieval performance across 10 temporal
reasoning classes derived from our taxonomy.
F.2.1 Reasoning Class Definitions
Table 22 provides definitions for each reasoning
class abbreviation used throughout this analysis.
F.2.2 Performance Heatmap
Figure 15 presents a heatmap visualization of
NDCG@10 performance across all model-class
combinations. Darker colors indicate higher perfor-
mance. The visualization reveals clear patterns: (1)
the HAC column shows consistently higher scores
across most models, (2) the TCP column shows
uniformly lower performance, and (3) reasoning-
enhanced models (DiVeR, ReasonIR) show more
uniform performance across classes compared to
the high variance observed in BM25.
F.2.3 Detailed Results
Table 23 presents complete NDCG@10 scores for
all 11 retrieval models across 10 reasoning classes.
Key findings include:
•Best overall: DiVeR (31.5 avg) achieves the
highest average performance, followed by
SFR (31.1) and ReasonIR/E5 (29.2).
•Hardest class: TCP (Trends & Cross-Period)
with an average of 17.9 across all models.
•Easiest class: HAC (Historical Attribution &
Context) with an average of 35.6 across all
models.
•Largest performance gap: BM25 shows a
19.0-point gap between its best (HAC: 23.3)
and worst (OTH: 4.3) classes.
•Most consistent: DiVeR shows the smallest
coefficient of variation across classes among
neural models.

Table 22: Temporal reasoning class definitions and query counts.
Abbr. Full Name Description Queries
EAL Event Analysis & Localization Identifying when specific events occurred or
analyzing their temporal properties624
TPC Time Period Contextualization Understanding practices, norms, or condi-
tions within a specific historical period365
OEC Origins & Evolution Comparative Tracing how concepts, technologies, or prac-
tices developed over time256
TCP Trends & Cross-Period Comparison Comparing information across multiple time
periods to identify changes or patterns154
EV A Event Verification & Authenticity Confirming whether events occurred and ver-
ifying their temporal accuracy115
MAP Materials & Artifacts Provenance Determining the origin, age, or temporal his-
tory of physical or digital artifacts89
OTH Other Queries with missing, ambiguous, or rare
temporal classifications58
SMD Sources & Methods Documentation Locating historical documentation or
methodological records from specific
periods25
CAU Causation Analysis Understanding temporal cause-effect rela-
tionships between events23
HAC Historical Attribution & Context Attributing ideas, inventions, or actions to
specific individuals and time periods21
EAL TPC OEC TCP EVA MAP OTH SMD CAU HAC
Reasoning ClassBM25
BGE
Contriever
SBERT
E5
Qwen
Rader
ReasonIRRetrieval Model15.3 8.4 10.4 7.8 19.3 8.9 4.3 12.0 9.6 23.3
23.2 21.5 21.6 14.4 24.6 28.3 16.5 22.5 11.3 35.3
23.2 21.3 19.6 15.1 26.5 27.9 16.5 20.3 19.0 30.3
24.5 23.8 24.9 15.9 27.4 30.6 19.8 29.2 21.1 38.9
31.8 27.6 26.3 23.7 30.0 31.1 20.0 31.7 26.3 43.5
21.6 22.7 21.6 16.5 24.6 29.0 17.5 28.5 19.0 22.1
22.2 23.1 23.4 16.1 22.3 24.2 22.5 27.7 24.2 27.5
28.7 27.1 27.0 20.4 31.7 36.3 21.0 33.1 28.1 38.2
510152025303540
NDCG@10 (%)
Figure 15: Heatmap of NDCG@10 performance across retrieval models and temporal reasoning classes. Darker
colors indicate higher performance. DiVeR and ReasonIR show the most consistent performance across reasoning
types, while BM25 exhibits high variance (4.3–23.3 range).

Model EAL TPC OEC TCP EV A MAP OTH SMD CAU HAC Avg.
Sparse
BM2515.3 8.4 10.4 7.8 19.3 8.9 4.3 12.0 9.6 23.3 11.9
Dense (<1B)
BGE23.2 21.5 21.6 14.4 24.6 28.3 16.5 22.5 11.3 35.3 21.9
Contriever23.2 21.3 19.6 15.1 26.5 27.9 16.5 20.3 19.0 30.3 22.0
SBERT24.5 23.8 24.9 15.9 27.4 30.6 19.8 29.2 21.1 38.9 25.6
Dense (>1B)
E531.8 27.6 26.3 23.7 30.0 31.1 20.0 31.7 26.3 43.5 29.2
GritLM28.2 24.5 24.1 19.7 32.5 27.4 16.9 28.8 19.1 40.5 26.2
Qwen21.6 22.7 21.6 16.5 24.6 29.0 17.5 28.5 19.0 22.1 22.3
SFR 32.328.4 26.9 23.4 32.4 34.8 21.636.023.751.831.1
Reasoning
Rader22.2 23.1 23.4 16.1 22.3 24.2 22.5 27.7 24.2 27.5 23.3
DiVeR31.231.8 28.0 23.9 35.8 38.1 24.333.7 27.4 40.731.5
ReasonIR28.7 27.1 27.0 20.4 31.7 36.3 21.0 33.128.138.2 29.2
Table 23: NDCG@10 performance across temporal reasoning classes. Models are evaluated on 10 reasoning classes:
EAL=Event Analysis & Localization (624), TPC=Time Period Contextualization (365), OEC=Origins & Evolution
Comparative (256), TCP=Trends & Cross-Period Comparison (154), EV A=Event Verification & Authenticity (115),
MAP=Materials & Artifacts Provenance (89), OTH=Other (58), SMD=Sources & Methods Documentation (25),
CAU=Causation Analysis (23), HAC=Historical Attribution & Context (21). The “Other” category includes queries
with missing or rare classifications. Best score per class inbold, second best underlined .

F.3 LLM Query Reformulation Analysis
This section provides detailed methodology and
results for the query reformulation experiments pre-
sented in Section 6.
F.3.1 Motivation
Recent work on reasoning-intensive retrieval (Su
et al., 2024) has shown that augmenting queries
with LLM-generated reasoning can improve re-
trieval for complex information needs. We investi-
gate whether this approach benefits temporal rea-
soning queries, which often require understanding
implicit temporal constraints, event relationships,
and cross-period comparisons.
F.3.2 Methodology
We reformulate each of the 1,730 TEMPO queries
using six LLMs of varying architectures and
scales: (1)GPT-4o: OpenAI’s flagship multimodal
model (API-based). (2)Llama-3.3-70B-Instruct:
Meta’s instruction-tuned model. (3)Qwen2.5-72B-
Instruct: Alibaba’s large instruction model. (4)
Qwen2.5-32B-Instruct: Mid-scale Qwen variant.
(5)DeepSeek-R1-Distill-Qwen-32B: Reasoning-
distilled model. Each LLM receives the following
prompt to generate reasoning about the query’s
information needs: The generated reasoning is con-
catenated with the original query to form the re-
formulated query, which is then used for retrieval
with each of the 12 retrieval models evaluated in
the main experiments.
Query Reformulation Prompt
Analyze the following question and generate
detailed reasoning about what kind of
information would help answer it.
Question:{query}
Instructions:
1. Identify the core problem or question
being asked.
2. Reason step-by-step about what
concepts, knowledge, or solutions
would be relevant.
3. Think about what a helpful document
should contain.
Provide your analysis:
Figure 16: Query reformulation prompt. The model
generates step-by-step reasoning about the information
needs to expand the query before retrieval.Reasoning-enhanced retrievers benefit most.
ReasonIR shows the largest improvement, gain-
ing +13.7 NDCG@10 with GPT-4o reformulation
(27.2→41.0). This dramatic improvement sug-
gests that ReasonIR’s architecture—specifically
designed to process reasoning chains—can effec-
tively leverage the step-by-step analysis generated
by LLMs. The model appears to extract relevant
semantic signals from the expanded context that
align with its internal reasoning mechanisms.
Sparse retrieval suffers from context dilution.
BM25 consistently degrades across all reformu-
lation strategies, dropping from 10.8 to 4.3–6.2
NDCG@10. The LLM-generated reasoning intro-
duces many terms that, while semantically relevant,
dilute the lexical overlap between query and rel-
evant documents. This confirms that reasoning
augmentation is fundamentally incompatible with
term-matching approaches.
Some models have internalized reasoning.
DiVeR maintains remarkably stable performance
(30.7–32.2) regardless of reformulation strategy,
suggesting its reasoning capabilities are already in-
ternalized during training and do not benefit from—
nor are harmed by—external reasoning augmen-
tation. This architectural difference from Rea-
sonIR highlights distinct approaches to reasoning-
enhanced retrieval.
LLM choice matters, but not always scale.
GPT-4o and DeepSeek-32B produce the most ef-
fective reformulations, while larger models do not
consistently outperform smaller ones. Qwen-72B
underperforms Qwen-32B for several retrievers,
suggesting that reformulation quality depends more
on reasoning style than raw model capacity.
Domain-specific patterns.Per-domain analysis
(Tables 24–28) reveals that reformulation benefits
vary by domain. Technical domains (Blockchain,
Quant) show more consistent improvements with
reasoning augmentation, while social science do-
mains exhibit higher variance. This may reflect
differences in how temporal reasoning manifests
across subject areas.
F.3.3 Implications
These findings have practical implications for tem-
poral retrieval system design:
1.Architecture selection: When LLM-based
query expansion is available, reasoning-

enhanced retrievers like ReasonIR should be
preferred over sparse methods.
2.Hybrid approaches: Systems could route
queries to different retrievers based on
whether reasoning augmentation is applied—
using BM25 for original queries and Rea-
sonIR for reformulated ones.
3.Cost-benefit tradeoff: The computational
cost of LLM reformulation may be justified
for reasoning-intensive queries but unneces-
sary for simple temporal lookups.
F.3.4 Detailed Results
Tables 24–28 present per-domain NDCG@10
scores for each reformulation strategy across all
12 retrieval models and 13 domains.
F.4 Additional Retrieval Metrics
In this section, we provide a multidimensional view
of model performance on the TEMPO benchmark
using standard IR metrics. Table 29 presents the
global averages across all 13 domains, while Ta-
bles 30 through 41 offer fine-grained per-domain
breakdowns for key representative models.
The supplemental results reveal several critical
patterns regarding temporal retrieval capability:
•High-Speed Convergence:DiVeR achieves
the highest overall Mean Reciprocal Rank
(MRR) of 39.8, indicating that its reasoning-
enhanced architecture is highly effective at
surfacing a relevant document at the first po-
sition. This is further supported by its lead-
ing Recall@10 (40.1%) and Precision@10
(11.1%).
•The Parameter Tier:A clear performance
gap exists between standard dense retrievers
and large models ( >1B). E5 and SFR outper-
form smaller models by significant margins
in MAP and Recall. This suggests that repre-
senting the nuanced temporal dependencies in
our dataset requires the expanded capacity of
larger encoders.
•Domain Robustness:Domain-specific analy-
sis (Tables 30–41) shows that models gener-
ally struggle more withQuantitative Finance
andBlockchaindomains compared toHistory
orPolitics. For instance, SFR’s MRR drops
to 20.4 in the Quant domain, suggesting thatnumeric temporal reasoning remains a signifi-
cant bottleneck.
•Lexical Vulnerability:BM25 exhibits a mas-
sive performance drop-off, with a MAP@10
of only 7.7. This demonstrates that temporal
relevance is fundamentally a semantic task
that cannot be solved by keyword overlap
alone.
G Quality Assessment & RAG
G.1 Dataset Quality Validation
To ensure the quality of TEMPO annotations, we
employ Qwen-72B as an independent LLM judge
to evaluate the alignment between queries, retrieval
plans, and gold documents. This validation is
performed separately from our main evaluation
pipeline (which uses GPT-4o) to provide an un-
biased assessment.
G.1.1 Evaluation Protocol
For each query in the dataset, we construct an evalu-
ation instance containing: (1) the original temporal
query, (2) the annotated retrieval plan with sequen-
tial steps, and (3) the gold documents mapped to
the query. The LLM judge evaluates each instance
on a 0–100 scale based on five criteria:
•Temporal Relevance: Whether gold doc-
uments contain the temporal information
needed to answer the query.
•Plan-Document Alignment: Whether gold
documents align with the retrieval steps speci-
fied in the plan.
•Temporal Coverage: Whether documents
cover the required time periods (baseline and
comparison periods if applicable).
•Completeness: Whether documents provide
sufficient evidence to comprehensively an-
swer the query.
•Authority: Whether documents are from reli-
able sources appropriate for the domain.
G.1.2 Results
Table 42 presents the quality validation results
across all 13 domains. TEMPO achieves an overall
average quality score of 86.7, with domain scores
ranging from 84.3 (Bitcoin) to 89.1 (HSM). The
consistently high scores across diverse domains

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin1.4 17.0 10.8 15.8 16.0 20.2 10.6 13.3 12.022.514.7 20.9
Cardano3.2 26.2 16.6 31.9 34.5 25.4 17.6 26.6 23.540.917.4 33.8
Iota0.0 30.2 32.7 31.2 25.4 33.7 20.3 23.8 20.538.523.2 29.1
Monero0.0 17.6 7.4 14.0 9.3 16.9 10.3 11.9 11.221.912.8 16.8
Social Sci.
Economics3.7 19.3 22.8 28.9 25.5 19.8 24.4 26.9 21.145.916.5 26.3
Law3.8 34.2 30.3 37.2 35.0 43.1 32.5 34.2 28.943.733.0 40.0
Politics29.2 32.9 35.2 50.3 48.1 44.2 47.3 41.1 31.857.135.9 46.9
History1.6 28.3 28.3 30.3 21.8 27.4 22.8 28.0 18.443.327.0 28.6
Applied
Quant0.0 12.0 15.6 24.9 16.2 24.2 7.2 18.9 20.238.318.1 22.7
Travel1.1 23.9 20.1 23.1 20.6 26.9 17.7 27.0 19.032.122.5 28.3
Workplace10.1 42.6 43.4 47.6 51.7 45.6 52.9 52.3 44.770.341.0 48.2
Genealogy4.4 25.4 31.2 32.9 37.2 24.8 31.3 28.6 16.039.320.6 36.1
STEM
HSM22.1 25.6 26.2 31.2 35.3 34.6 37.8 27.1 12.339.027.2 35.2
Avg.8.0 25.8 24.7 30.7 29.0 29.7 25.6 27.7 21.541.023.8 31.8
Table 24: NDCG@10 performance using queries reformulated byGPT-4o. The LLM generates reasoning about
information needs before retrieval. Best inbold, second best underlined .
validate that our annotation process produces high-
quality query-document alignments suitable for
evaluating temporal reasoning in retrieval systems.
G.1.3 Evaluation Prompt
The following prompt is used for quality validation
with Qwen-72B:

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin1.0 15.4 11.1 18.4 18.9 18.5 10.8 16.1 18.422.715.1 19.4
Cardano2.3 19.7 22.9 28.3 29.6 22.6 13.8 18.5 20.934.719.2 27.9
Iota0.0 37.5 35.4 27.3 28.4 29.0 23.1 27.9 29.744.827.6 28.1
Monero0.0 17.7 9.7 18.3 10.8 17.3 8.6 15.3 13.323.614.9 18.1
Social Sci.
Economics2.1 16.5 22.4 32.1 24.3 20.8 18.8 26.6 24.943.517.3 24.9
Law4.4 39.4 29.8 41.5 36.4 41.5 32.2 35.1 35.743.533.6 39.1
Politics27.5 33.6 39.5 49.2 48.9 42.9 45.2 41.2 34.655.535.2 46.9
History1.7 30.5 31.5 33.9 24.4 27.6 20.7 30.9 23.643.527.9 29.5
Applied
Quant0.0 14.0 18.3 27.4 19.9 27.1 7.7 20.0 26.934.421.9 22.7
Travel0.5 25.2 21.9 23.9 19.3 26.1 14.828.219.3 27.1 26.2 24.1
Workplace4.8 38.6 43.1 50.2 45.1 36.3 46.0 47.7 40.165.041.2 43.3
Genealogy2.5 23.5 28.3 33.439.025.9 23.4 29.2 17.3 36.1 18.5 36.3
STEM
HSM22.0 30.7 30.4 34.2 37.7 34.2 39.1 29.8 18.041.027.9 36.6
Avg.6.9 26.3 26.5 32.2 29.4 28.5 23.4 28.2 24.839.625.1 30.5
Table 25: NDCG@10 performance using queries reformulated byDeepSeek-32B. The LLM generates reasoning
about information needs before retrieval. Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin1.2 12.9 6.5 15.2 15.8 19.8 12.5 13.9 11.920.710.2 20.0
Cardano1.9 13.0 10.6 26.1 26.6 23.9 17.2 21.4 11.8 28.2 19.529.0
Iota2.3 25.2 31.1 33.3 22.3 32.3 23.4 37.5 23.737.915.6 32.4
Monero0.0 11.4 4.6 15.3 6.2 17.4 13.6 12.4 9.620.210.5 13.9
Social Sci.
Economics3.2 14.5 18.4 28.8 20.9 20.9 22.6 25.3 21.643.614.6 23.5
Law1.8 27.7 25.8 41.1 31.6 42.7 35.3 37.5 32.446.137.8 37.6
Politics20.7 26.8 28.2 47.6 43.7 42.1 41.3 39.5 30.354.932.5 43.5
History0.9 24.1 22.7 30.8 18.3 27.9 26.0 27.3 18.840.624.6 25.2
Applied
Quant0.0 12.9 19.1 27.1 16.4 25.6 9.2 18.6 22.536.814.7 21.4
Travel1.7 21.2 17.9 24.0 16.928.220.0 24.8 16.6 26.7 21.3 22.5
Workplace9.3 40.5 40.6 49.4 50.2 47.4 52.8 50.3 41.966.140.5 46.1
Genealogy2.2 22.4 25.5 31.2 33.5 26.2 29.0 29.3 17.037.421.2 32.3
STEM
HSM15.4 21.9 18.4 31.7 30.534.332.1 25.8 8.3 32.8 25.6 30.3
Avg.5.5 21.1 20.7 30.9 25.6 29.9 25.8 28.0 20.537.822.2 29.0
Table 26: NDCG@10 performance using queries reformulated byLlama-70B. The LLM generates reasoning about
information needs before retrieval. Best inbold, second best underlined .

Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin1.1 16.9 8.9 17.0 16.6 18.4 12.3 14.1 14.122.910.9 21.2
Cardano0.5 19.1 14.8 27.5 26.3 23.2 18.7 18.1 14.832.816.4 28.2
Iota0.0 27.0 34.5 29.0 21.7 32.0 21.4 33.8 22.445.814.3 27.1
Monero0.0 14.9 8.1 17.3 5.1 16.8 8.9 9.3 10.319.59.5 11.8
Social Sci.
Economics2.2 19.3 21.4 29.1 18.9 19.3 21.8 25.1 18.843.814.0 23.2
Law2.5 36.8 27.1 38.9 32.5 37.5 33.8 34.6 32.646.833.5 39.6
Politics20.4 27.8 30.4 50.5 42.2 43.0 41.1 36.3 25.953.532.3 42.8
History1.2 24.8 24.3 30.2 17.5 26.2 25.6 26.6 15.141.523.4 24.4
Applied
Quant0.0 12.6 19.5 24.3 15.3 23.2 8.5 17.2 21.132.717.7 19.7
Travel1.1 23.6 17.6 25.5 17.1 26.1 20.4 25.4 14.031.619.8 23.5
Workplace9.8 41.2 35.1 48.9 42.8 44.1 49.1 50.1 37.866.340.5 42.5
Genealogy2.0 26.4 29.4 31.2 32.5 26.3 30.9 25.9 14.737.319.2 32.6
STEM
HSM17.8 25.0 21.7 31.7 30.6 34.6 35.5 26.7 8.435.625.7 32.4
Avg.5.9 24.3 22.5 30.9 24.6 28.5 25.2 26.4 19.239.221.3 28.4
Table 27: NDCG@10 performance using queries reformulated byQwen-72B. The LLM generates reasoning about
information needs before retrieval. Best inbold, second best underlined .
Domain BM25 BGE Contriever DiVeR E5 GritLM Inst-L Qwen Rader ReasonIR SBERT SFR
Blockchain
Bitcoin0.0 14.5 9.6 15.7 15.819.912.8 14.1 13.5 19.8 9.3 18.8
Cardano0.6 15.3 11.6 27.9 31.5 22.0 17.2 18.9 14.735.121.3 31.3
Iota0.0 23.0 29.7 31.6 24.5 31.3 21.5 23.9 15.237.626.7 27.3
Monero0.0 14.4 6.5 19.0 9.3 16.8 15.0 11.7 13.319.110.4 14.5
Social Sci.
Economics3.6 15.6 16.7 31.3 18.7 20.7 23.8 22.9 18.839.115.3 21.5
Law3.2 32.6 23.2 38.8 36.6 42.4 38.1 36.7 29.0 36.6 32.443.4
Politics20.6 27.0 27.8 46.7 44.6 41.2 38.4 37.2 29.549.432.0 44.5
History2.0 23.1 21.9 30.5 19.9 25.6 26.9 27.1 18.737.623.2 26.1
Applied
Quant0.0 15.3 14.1 28.9 19.3 26.0 11.7 19.3 25.534.418.6 20.6
Travel2.8 20.0 16.8 23.7 21.726.019.6 23.4 17.7 25.1 20.7 24.3
Workplace6.5 36.5 37.7 47.7 41.1 37.748.041.1 37.4 47.6 39.4 40.5
Genealogy3.4 22.6 23.6 32.8 33.6 25.7 28.3 26.3 17.534.821.2 32.4
STEM
HSM13.0 23.5 19.1 31.0 31.0 32.3 31.3 24.6 11.333.122.1 31.0
Avg.6.2 21.8 19.9 31.2 26.7 28.3 25.6 25.2 20.134.622.5 28.9
Table 28: NDCG@10 performance using queries reformulated byQwen-32B. The LLM generates reasoning about
information needs before retrieval. Best inbold, second best underlined .

Prompt for Dataset Quality Validation
You are an expert evaluator assessing the quality of dataset annotations for a temporal
retrieval benchmark.
=== QUERY ===
{query}
=== RETRIEVAL PLAN ===
{retrieval_plan}
=== GOLD DOCUMENTS ===
{gold_documents}
=== TASK ===
Evaluate how well the retrieval plan and gold documents serve to answer the temporal query.
Consider:
1.Temporal Relevance: Do the gold documents contain the temporal information needed?
2.Plan-Document Alignment: Do the gold documents align with the retrieval steps?
3.Temporal Coverage: Do the documents cover the required time periods?
4.Completeness: Would these documents provide sufficient evidence to answer the query?
5.Authority: Are the documents from reliable sources appropriate for the domain?
=== SCORING CRITERIA (0-100) ===
•90-100: Excellent - Documents perfectly match plan, comprehensive temporal coverage
•80-89: Strong - Documents highly relevant, minor gaps in temporal coverage
•70-79: Good - Documents mostly aligned, some temporal aspects not fully covered
•60-69: Adequate - Documents partially relevant, noticeable gaps
•50-59: Weak - Documents loosely related, significant misalignment
•0-49: Poor - Documents minimally helpful or irrelevant
=== OUTPUT FORMAT ===
REASONING: [Provide detailed assessment]
SCORE: [A single integer between 0 and 100]
Do not include anything after the score.
Figure 17: Dataset quality validation prompt. An independent judge (Qwen-72B) evaluates the alignment between
the query, the retrieval plan, and the gold documents.

Table 29: Comparison of retrieval models on TEMPO
(averaged across 13 domains). Best inbold, second best
underlined .
MAP Recall Precision
Model @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Sparse
BM25 4.0 7.7 8.3 4.0 13.7 19.0 9.4 3.5 2.0 14.5
Dense (<1B)
BGE 8.0 15.4 16.7 8.0 27.5 39.3 20.7 7.4 4.3 30.3
Contriever 7.6 15.2 16.5 7.6 26.9 39.4 19.2 7.2 4.3 28.8
Inst-L 8.6 17.3 18.7 8.6 31.7 44.3 21.2 8.5 4.9 32.6
SBERT 8.9 17.4 18.8 8.9 31.6 43.4 22.9 8.4 4.8 33.5
Dense (>1B)
E511.522.4 23.8 11.537.0 48.727.410.1 5.4 38.9
GritLM 10.2 19.7 21.4 10.2 34.8 49.6 23.4 9.1 5.4 34.5
Qwen 8.0 16.1 17.4 8.0 28.1 40.1 20.4 8.1 4.7 30.5
SFR 11.0 21.7 23.3 11.0 37.9 50.9 26.2 10.2 5.6 37.8
Reasoning
DiVeR 11.2 23.5 25.111.2 40.1 52.627.4 11.1 5.9 39.8
Rader 8.4 17.0 18.3 8.4 31.8 43.6 20.7 8.3 4.7 31.0
ReasonIR 9.2 19.6 21.2 9.2 35.1 48.7 22.7 9.5 5.4 34.2
Table 30: Detailed retrieval metrics forSFRon TEMPO
across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin7.7 12.5 12.9 7.7 20.6 24.8 21.0 5.4 2.9 27.2
Cardano8.6 19.2 20.7 8.6 42.4 57.8 19.6 9.0 4.9 31.7
Iota11.7 26.8 28.3 11.7 39.5 50.3 40.0 13.0 6.4 52.6
Monero9.7 16.8 17.7 9.7 28.8 40.2 23.1 7.7 4.1 32.8
Social Sci.
Economics8.3 14.4 16.2 8.3 26.2 41.8 24.1 9.2 6.0 33.8
Law16.4 29.5 32.2 16.4 55.5 72.0 28.6 14.3 8.1 45.7
Politics17.2 35.6 37.4 17.2 54.0 66.3 42.0 13.7 7.0 53.2
History11.1 22.8 24.8 11.1 39.0 53.1 32.0 12.6 7.1 43.9
Applied
Quant5.4 11.7 12.0 5.4 22.0 26.7 8.8 5.6 2.6 20.4
Travel8.9 21.1 22.8 8.9 41.1 54.4 18.0 10.1 5.6 33.0
Workplace12.2 22.4 24.8 12.2 43.9 63.9 25.0 11.9 6.9 35.2
Genealogy13.2 24.0 26.3 13.2 36.2 52.5 32.2 10.6 6.3 41.9
STEM
HSM12.9 24.9 26.4 12.9 43.7 58.2 26.7 9.9 5.4 39.5
Average 11.0 21.7 23.3 11.0 37.9 50.9 26.2 10.2 5.6 37.8
Table 31: Detailed retrieval metrics forSBERTon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin5.2 9.3 10.0 5.2 16.9 24.1 16.0 5.1 3.0 23.1
Cardano6.0 14.0 15.1 6.0 32.7 45.5 15.7 7.3 4.1 25.8
Iota15.0 23.6 25.7 15.0 31.2 44.5 50.0 10.0 5.6 57.2
Monero3.7 9.0 9.7 3.7 22.2 30.6 10.8 5.5 3.1 21.0
Social Sci.
Economics2.4 9.1 10.6 2.4 21.3 32.3 12.0 7.3 4.8 22.4
Law18.8 24.7 26.4 18.8 42.5 55.4 34.3 10.3 6.1 44.4
Politics12.3 26.3 27.4 12.3 42.7 52.6 30.0 11.1 5.5 41.6
History10.4 19.9 21.9 10.4 33.9 48.6 29.8 10.8 6.5 40.8
Applied
Quant6.1 12.0 12.8 6.1 19.0 30.1 11.8 5.0 3.1 20.7
Travel7.5 18.5 20.4 7.5 38.9 53.4 18.0 9.1 5.4 32.4
Workplace10.7 25.0 27.5 10.7 44.9 67.6 27.8 11.1 6.6 43.0
Genealogy8.8 16.8 18.0 8.8 27.7 35.7 23.5 8.5 4.4 31.7
STEM
HSM8.5 18.1 19.0 8.5 36.4 44.4 18.7 8.1 4.0 31.0
Average 8.9 17.4 18.8 8.9 31.6 43.4 22.9 8.4 4.8 33.5Table 32: Detailed retrieval metrics forReasonIRon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin5.9 11.5 12.1 5.9 21.6 29.6 13.0 5.6 3.3 21.6
Cardano6.3 15.4 17.1 6.3 35.5 52.1 13.7 7.5 4.4 25.3
Iota13.3 31.3 31.9 13.3 47.3 51.0 40.0 15.0 6.8 54.3
Monero7.7 13.7 15.0 7.7 23.2 36.5 20.0 6.6 3.9 28.9
Social Sci.
Economics5.8 13.0 15.2 5.8 26.1 43.6 18.1 8.3 6.0 28.5
Law14.1 27.9 30.5 14.1 47.3 67.9 31.4 13.1 7.4 47.7
Politics12.4 26.9 28.1 12.4 46.1 55.9 29.3 11.5 5.8 40.1
History12.0 24.3 26.6 12.0 41.9 57.4 33.3 13.3 7.7 45.6
Applied
Quant5.2 14.1 15.3 5.2 26.2 37.2 11.8 6.2 3.8 23.7
Travel7.0 15.7 17.1 7.0 29.9 44.2 11.0 7.1 4.4 23.3
Workplace8.4 20.8 22.8 8.4 42.9 62.1 19.4 11.7 6.4 33.6
Genealogy12.1 22.8 24.8 12.1 34.6 49.3 33.0 10.7 6.1 40.9
STEM
HSM9.1 17.6 19.0 9.1 33.0 47.0 20.7 7.6 4.5 30.9
Average 9.2 19.6 21.2 9.2 35.1 48.7 22.7 9.5 5.4 34.2
Table 33: Detailed retrieval metrics forRaderon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin6.8 10.7 11.7 6.8 17.2 27.3 16.0 4.8 3.1 22.8
Cardano3.5 11.4 12.3 3.5 31.2 41.0 9.8 6.1 3.5 21.4
Iota7.5 11.9 13.0 7.5 25.8 36.7 20.0 8.0 4.4 27.9
Monero7.4 15.3 16.0 7.4 26.3 34.5 16.9 6.6 3.5 27.1
Social Sci.
Economics6.1 15.0 16.6 6.1 30.0 42.9 20.5 9.5 5.6 31.0
Law10.6 22.8 24.9 10.6 47.4 62.3 25.7 11.1 6.4 41.1
Politics10.9 24.0 25.4 10.9 40.2 51.4 28.7 10.7 5.5 39.4
History9.0 17.4 19.4 9.0 31.1 45.7 26.3 9.9 6.2 37.7
Applied
Quant9.7 20.1 20.9 9.7 36.3 45.9 23.5 8.2 4.4 35.5
Travel11.8 19.8 20.9 11.8 32.2 42.9 23.0 7.9 4.4 32.4
Workplace15.1 27.3 29.5 15.1 48.6 66.7 30.6 11.9 6.9 41.6
Genealogy6.1 13.4 14.7 6.1 23.4 36.2 16.5 7.0 4.2 24.6
STEM
HSM5.1 11.4 12.3 5.1 24.0 33.6 11.3 5.9 3.3 20.2
Average 8.4 17.0 18.3 8.4 31.8 43.6 20.7 8.3 4.7 31.0
Table 34: Detailed retrieval metrics forQwenon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin4.6 7.5 7.9 4.6 12.7 17.5 14.0 3.9 2.2 19.3
Cardano5.7 13.9 15.0 5.7 29.8 42.5 13.7 6.9 3.8 24.3
Iota10.8 18.2 20.6 10.8 32.3 52.2 30.0 11.0 7.2 44.9
Monero2.9 6.7 7.4 2.9 14.6 23.2 7.7 4.3 2.6 15.8
Social Sci.
Economics4.6 11.3 12.4 4.6 21.8 30.7 12.0 7.6 4.6 23.3
Law14.2 25.2 27.3 14.2 37.1 56.9 28.6 10.6 6.6 39.7
Politics14.4 28.8 30.2 14.4 46.5 57.3 36.0 12.1 6.1 47.0
History8.9 17.5 19.2 8.9 30.5 43.5 26.0 10.0 5.9 36.5
Applied
Quant2.8 8.3 9.1 2.8 15.8 22.5 8.8 4.1 2.5 19.4
Travel9.4 16.0 17.4 9.4 28.8 41.5 18.0 6.9 4.2 26.8
Workplace10.9 21.7 24.0 10.9 36.9 54.1 30.6 11.4 6.3 39.5
Genealogy7.5 18.3 19.2 7.5 30.6 38.3 22.6 9.7 4.8 33.2
STEM
HSM7.7 15.1 16.3 7.7 28.4 40.9 17.3 6.9 4.0 26.4
Average 8.0 16.1 17.4 8.0 28.1 40.1 20.4 8.1 4.7 30.5

Table 35: Detailed retrieval metrics forInst-Lon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin5.6 10.3 10.9 5.6 20.5 26.6 15.0 5.3 3.2 23.3
Cardano1.3 8.4 9.6 1.3 25.0 39.8 3.9 5.9 3.7 16.6
Iota16.7 23.9 25.5 16.7 32.8 44.0 50.0 11.0 6.0 58.3
Monero3.7 10.2 11.5 3.7 23.9 38.4 10.8 6.5 4.0 23.3
Social Sci.
Economics4.3 11.1 12.5 4.3 22.8 34.3 13.3 7.8 4.9 24.5
Law16.6 26.6 28.7 16.6 47.2 61.4 31.4 12.0 6.9 47.6
Politics11.0 24.4 25.6 11.0 41.2 53.1 25.3 10.5 5.5 38.6
History9.8 20.0 22.0 9.8 34.4 48.2 27.7 11.1 6.6 38.7
Applied
Quant5.6 10.2 11.3 5.6 17.6 30.8 14.7 4.1 2.8 22.9
Travel6.8 17.2 18.5 6.8 37.1 47.7 13.0 9.2 4.9 25.6
Workplace10.8 27.4 29.4 10.8 46.1 64.8 27.8 11.7 6.3 42.8
Genealogy10.6 17.7 19.5 10.6 30.4 41.6 24.3 8.6 5.0 32.0
STEM
HSM8.4 17.6 18.8 8.4 32.9 45.2 18.0 7.4 4.2 29.2
Average 8.6 17.3 18.7 8.6 31.7 44.3 21.2 8.5 4.9 32.6
Table 36: Detailed retrieval metrics forGritLMon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin10.3 14.1 15.2 10.3 20.8 30.5 24.0 5.5 3.4 29.2
Cardano5.1 14.0 15.5 5.1 35.9 53.0 9.8 6.7 4.4 23.6
Iota11.7 27.8 29.1 11.7 40.8 52.0 40.0 12.0 6.8 50.6
Monero5.6 10.3 12.2 5.6 16.8 40.1 13.8 4.9 4.2 22.7
Social Sci.
Economics4.6 10.3 11.5 4.6 23.7 34.1 13.3 8.2 4.9 24.0
Law19.2 28.6 31.7 19.2 48.6 70.3 34.3 12.3 7.8 47.0
Politics15.0 32.4 34.0 15.0 50.6 65.0 35.3 13.0 6.8 48.2
History9.6 18.7 20.7 9.6 33.9 49.2 25.3 10.6 6.5 36.9
Applied
Quant9.8 16.4 17.2 9.8 26.4 37.1 14.7 6.5 3.4 25.7
Travel6.3 16.7 18.0 6.3 37.0 49.9 16.0 9.0 4.9 28.9
Workplace12.0 22.6 24.9 12.0 42.5 64.4 25.0 10.6 6.4 35.7
Genealogy11.5 19.9 21.5 11.5 32.0 44.7 26.1 8.9 5.3 35.5
STEM
HSM11.8 24.9 26.3 11.8 42.8 55.1 26.7 9.9 5.3 40.6
Average 10.2 19.7 21.4 10.2 34.8 49.6 23.4 9.1 5.4 34.5
Table 37: Detailed retrieval metrics forE5on TEMPO
across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin6.9 11.5 12.3 6.9 18.4 27.0 18.0 5.2 3.1 25.6
Cardano12.5 25.5 26.2 12.5 51.3 59.1 23.5 10.8 4.9 38.5
Iota16.7 31.5 32.9 16.7 41.2 47.0 50.0 13.0 6.0 61.8
Monero6.5 13.4 14.5 6.5 26.7 37.7 15.4 7.1 4.1 25.9
Social Sci.
Economics7.7 16.5 18.2 7.7 30.2 43.5 22.9 10.7 6.4 36.1
Law13.4 26.0 29.2 13.4 41.0 67.7 25.7 11.7 7.8 41.3
Politics18.8 38.1 39.8 18.8 56.7 68.6 45.3 14.6 7.2 56.8
History10.2 20.2 21.6 10.2 32.8 42.5 31.2 11.1 6.0 41.0
Applied
Quant4.7 9.3 9.8 4.7 17.6 24.4 8.8 4.4 2.4 18.5
Travel9.2 20.2 21.5 9.2 38.2 47.9 19.0 9.2 4.8 32.4
Workplace12.7 24.1 26.3 12.7 41.8 59.5 27.8 11.9 6.7 38.9
Genealogy14.7 25.5 27.3 14.7 38.1 51.1 34.8 11.2 6.1 44.0
STEM
HSM15.8 29.2 30.3 15.8 46.7 57.5 33.3 10.9 5.4 44.9
Average 11.5 22.4 23.8 11.5 37.0 48.7 27.4 10.1 5.4 38.9Table 38: Detailed retrieval metrics forDiVeRon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin5.9 11.7 12.5 5.9 22.2 31.3 17.0 5.9 3.6 25.5
Cardano12.3 20.8 22.5 12.3 38.3 52.7 25.5 8.4 4.7 37.9
Iota8.3 26.5 27.0 8.3 46.5 49.0 30.0 15.0 6.4 49.7
Monero8.2 14.5 15.5 8.2 26.2 35.1 18.5 6.9 3.9 25.5
Social Sci.
Economics6.0 19.0 21.4 6.0 35.7 51.3 21.7 12.2 7.3 35.9
Law11.3 28.5 31.1 11.3 57.1 74.9 25.7 15.7 8.6 43.1
Politics18.8 36.6 38.5 18.8 51.4 66.8 44.0 13.3 7.0 56.2
History11.9 24.7 26.8 11.9 41.1 55.1 34.6 13.3 7.6 46.1
Applied
Quant11.6 19.8 20.8 11.6 34.2 45.2 20.6 8.5 4.6 33.4
Travel10.4 19.5 21.3 10.4 36.2 50.9 20.0 9.0 5.1 30.5
Workplace16.9 34.0 35.7 16.9 49.8 65.4 41.7 13.9 7.1 51.1
Genealogy13.2 26.8 28.1 13.2 41.9 50.9 33.9 12.0 6.0 45.7
STEM
HSM10.3 22.9 24.5 10.3 40.0 55.7 22.7 9.8 5.4 36.8
Average 11.2 23.5 25.1 11.2 40.1 52.6 27.4 11.1 5.9 39.8
Table 39: Detailed retrieval metrics forContrieveron
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin3.1 8.7 9.4 3.1 18.0 25.3 9.0 4.7 2.8 18.6
Cardano4.6 8.3 9.9 4.6 18.3 35.9 7.8 3.3 2.7 15.2
Iota16.7 28.7 29.5 16.7 39.5 46.5 50.0 12.0 6.0 55.9
Monero2.1 5.6 6.7 2.1 13.8 26.2 6.2 3.7 2.8 16.1
Social Sci.
Economics5.7 10.7 12.0 5.7 18.6 30.6 19.3 6.1 4.1 26.8
Law9.0 19.9 21.6 9.0 40.8 57.9 14.3 10.6 6.5 29.6
Politics11.6 24.3 25.3 11.6 38.6 48.6 29.3 9.7 5.0 39.1
History9.4 18.2 20.1 9.4 31.0 45.4 28.6 10.2 6.2 38.6
Applied
Quant5.0 8.3 9.0 5.0 13.3 21.2 8.8 3.2 2.1 15.6
Travel7.1 16.3 18.0 7.1 32.8 46.9 17.0 8.0 4.7 28.5
Workplace5.8 17.1 19.0 5.8 29.5 49.5 16.7 8.6 5.1 31.9
Genealogy12.3 18.5 19.9 12.3 29.1 40.5 28.7 8.1 4.7 34.5
STEM
HSM6.1 13.1 14.1 6.1 26.3 37.1 14.0 5.6 3.3 23.5
Average 7.6 15.2 16.5 7.6 26.9 39.4 19.2 7.2 4.3 28.8
Table 40: Detailed retrieval metrics forBM25on
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin2.5 4.3 4.4 2.5 8.5 10.1 5.0 1.7 0.8 8.2
Cardano4.9 9.6 10.0 4.9 20.6 25.5 5.9 3.5 1.9 13.8
Iota2.5 5.4 5.4 2.5 11.7 11.7 10.0 4.0 1.6 17.3
Monero0.8 1.5 1.7 0.8 4.7 7.8 1.5 0.9 0.7 3.9
Social Sci.
Economics2.2 4.4 4.8 2.2 6.2 10.3 6.0 2.7 1.7 8.8
Law4.0 8.8 9.9 4.0 16.5 25.0 8.6 4.9 2.9 16.2
Politics12.8 25.6 26.5 12.8 38.3 47.2 32.7 9.5 4.7 41.5
History4.0 6.7 7.2 4.0 10.4 14.1 10.6 3.3 1.9 13.6
Applied
Quant1.5 1.8 2.1 1.5 2.9 5.9 2.9 0.6 0.6 4.2
Travel1.3 3.2 3.8 1.3 7.1 15.0 2.0 1.6 1.2 5.6
Workplace3.1 4.2 4.8 3.1 9.3 16.2 5.6 1.9 1.4 9.8
Genealogy4.8 9.7 10.6 4.8 15.2 21.8 12.2 4.9 2.8 18.1
STEM
HSM7.7 15.4 16.3 7.7 26.9 36.8 18.7 6.5 3.4 27.2
Average 4.0 7.7 8.3 4.0 13.7 19.0 9.4 3.5 2.0 14.5

Table 41: Detailed retrieval metrics forBGEon
TEMPO across all domains.
MAP Recall Precision
Domain @1 @10 @25 @1 @10 @25 @1 @10 @25 MRR
Blockchain
Bitcoin6.0 9.6 10.6 6.0 17.5 27.3 16.0 4.6 3.1 23.0
Cardano1.3 7.4 9.4 1.3 22.1 42.3 3.9 4.9 3.7 16.2
Iota15.0 26.1 27.4 15.0 34.5 42.8 50.0 11.0 5.6 59.7
Monero5.1 9.3 10.2 5.1 16.8 26.6 15.4 4.9 3.0 23.6
Social Sci.
Economics3.0 7.6 9.1 3.0 15.5 31.2 13.3 6.1 4.2 20.5
Law16.1 23.3 25.3 16.1 40.2 55.5 31.4 10.0 6.3 41.4
Politics10.2 21.7 22.7 10.2 33.8 44.1 24.7 8.5 4.5 35.8
History10.2 19.1 20.8 10.2 32.8 45.5 28.2 10.4 6.1 38.7
Applied
Quant1.5 7.1 7.8 1.5 18.9 26.9 2.9 4.4 2.8 13.8
Travel7.7 17.0 18.2 7.7 32.0 42.6 18.0 7.8 4.3 28.8
Workplace9.9 18.8 20.2 9.9 36.8 48.8 25.0 8.6 4.8 35.7
Genealogy9.3 16.2 17.2 9.3 26.0 34.8 21.7 7.6 4.1 29.2
STEM
HSM9.1 16.9 18.1 9.1 30.8 42.6 18.7 7.1 4.0 27.9
Average 8.0 15.4 16.7 8.0 27.5 39.3 20.7 7.4 4.3 30.3
Category Domain Score
BlockchainBitcoin 84.3
Cardano 87.3
Iota 85.7
Monero 85.5
Social SciencesEconomics 86.8
Law 87.1
Politics 86.1
History 88.7
AppliedQuant 87.3
Travel 86.3
Workplace 85.7
Genealogy 86.6
STEM HSM 89.1
Overall 87.0
Table 42: Dataset quality validation results using Qwen-
72B as LLM judge across all domains.

G.2 Temporal Distribution by Domain
We observe distinct temporal patterns when decom-
posing the query distribution across our primary
domain categories: Blockchain, Social Sciences,
Applied, and STEM. As illustrated in Figure 18,
each domain exhibits a unique temporal signature
that reflects its inherent characteristics:
•Blockchain: Exclusively focused on modern
history, with the vast majority of queries oc-
curring after 2010. This aligns with the tech-
nological birth and rapid evolution of decen-
tralized protocols.
•Social Sciences: Provides the strongest histor-
ical grounding, peaking at nearly 300 queries
in the Pre-1900 category. This enables rigor-
ous testing of reasoning over deep historical
records and long-term societal shifts.
•Applied: Displays a bimodal distribution;
while it includes significant historical data
(Pre-1900, 75 queries) related to fields like
Genealogy, it also shows a strong recent resur-
gence in the 2020+ period ( 50 queries) driven
by modern workplace and travel dynamics.
•STEM: Almost exclusively focuses on the
foundations of scientific and mathematical
thought, with over 80 queries anchored Pre-
1900 and a sharp decline as questions reach
the 21st century.

Pre-19001900-49 1950-79 1980-99 2000-09 2010-192020+010203040QueriesBlockchain
Pre-19001900-49 1950-79 1980-99 2000-09 2010-192020+0100200QueriesSocial Sciences
Pre-19001900-49 1950-79 1980-99 2000-09 2010-192020+0204060QueriesApplied
Pre-19001900-49 1950-79 1980-99 2010-19020406080QueriesSTEMFigure 18: Temporal anchor distribution decomposed by major domain categories. Note the varying Y-axis scales
and distinct "temporal signatures" for each field.

G.3 RAG Evaluation Details
G.3.1 Full Per-Domain Results
Table 43 presents the complete RAG evaluation
results across all 13 domains.
G.3.2 Answer Generation Prompt
For answer generation, we prompt Llama-3-70B-
Instruct with retrieved documents as context:
G.3.3 Answer Evaluation Prompt
Following BRIGHT (Su et al., 2024), we use GPT-
4o as an evaluator to score answer correctness on
a 0–100 scale based on coverage of the reference
answer.
Domain None Oracle BM25 DiVeR BGE
Blockchain
Bitcoin 79.482.779.3 79.3 78.1
Cardano 77.580.169.8 73.9 69.2
Iota 76.078.067.0 70.0 72.0
Monero 76.380.270.3 70.5 72.0
Social Sciences
Economics 80.882.679.0 80.7 80.0
Law 80.983.176.3 78.0 74.0
Politics 71.876.468.8 69.7 72.1
History 73.376.970.5 72.1 72.2
Applied
Quant 79.183.977.9 79.1 76.5
Travel 79.582.876.2 78.5 77.9
Workplace 81.484.078.9 80.8 80.0
Genealogy 80.182.078.9 79.8 79.0
STEM
HSM 69.574.266.6 70.9 69.2
Average77.380.573.8 75.6 74.8
Table 43: RAG performance (answer correctness score,
0–100) across all domains and retrievers.None: no
retrieval (parametric knowledge only);Oracle: gold
documents provided. Best scores per domain inbold.

Prompt for Answer Generation
You are a helpful assistant. Answer the following question based on the provided context
documents.
Context Documents:
{retrieved_documents}
Question:
{query}
Answer:
Figure 19: RAG answer generation prompt. The generator is instructed to answer the user query based strictly on
the provided retrieved context documents.
Prompt for Answer Evaluation (GPT-4o)
You are a teacher to judge student’s answer.
———- PROBLEM START ———-
{query}
———- PROBLEM END ———-
———- STUDENT ANSWER START ———-
{predicted_answer}
———- STUDENT ANSWER END ———-
———- REFERENCE ANSWER START ———-
{gold_answer}
———- REFERENCE ANSWER END ———-
Criteria:
•0 The student’s answer is completely irrelevant or blank.
•10 The student’s answer addresses about 10% of the reference content.
•20 The student’s answer addresses about 20% of the reference content.
•30 The student’s answer addresses about 30% of the reference content.
•40 The student’s answer addresses about 40% of the reference content.
•50 The student’s answer addresses about 50% of the reference content.
•60 The student’s answer addresses about 60% of the reference content.
•70 The student’s answer addresses about 70% of the reference content.
•80 The student’s answer addresses about 80% of the reference content.
•90 The student’s answer addresses about 90% of the reference content.
•100 The student’s answer addresses about 100% of the reference content.
Use the following format to give a score:
REASON:
Describe why you give a specific score
SCORE:
The score you give, e.g., 60
Do not say anything after the score.
Figure 20: RAG answer evaluation prompt. The judge scores the generated answer (0–100) based on semantic
coverage of the reference gold answer.

H Extended Related Work
This appendix provides a comprehensive review
of the literature surrounding temporal informa-
tion retrieval, temporal question answering (QA),
and reasoning-intensive retrieval. We contextu-
alize TEMPO within the recent surge of interest
in temporal reasoning for Large Language Mod-
els (LLMs) and Retrieval-Augmented Generation
(RAG).
H.1 Temporal Information Retrieval
Temporal Information Retrieval (TIR) has long
been recognized as a critical subfield of IR. Early
surveys (Campos et al., 2014) and tasks like NT-
CIR Temporalia (Joho et al., 2014) established
the importance of classifying queries by tempo-
ral intent (e.g., past, future, atemporal). However,
these benchmarks primarily relied on news and
blog corpora where temporal reasoning was lim-
ited to identifying explicit timestamps or "recency"
signals (Joho et al., 2016).
A recent survey by Piryani et al. (2025) high-
lights that while traditional TIR focused on docu-
ment re-ranking based on publication dates, mod-
ern challenges require "content-aware" temporal
reasoning that goes beyond metadata. TEMPO
addresses this by requiring systems to retrieve doc-
uments based on semantic temporal alignment (e.g.,
"post-GDPR era" vs. "pre-2018") rather than sim-
ple timestamp filtering.
H.2 Temporal Question Answering
The field of Temporal QA has seen rapid devel-
opment, shifting from simple factoid questions to
complex reasoning tasks.
•Fact-Centric Benchmarks:Early datasets
like TempQuestions (Jia et al., 2018) and
TimeQA (Chen et al., 2021) focused on knowl-
edge base (KB) lookup. More recent large-
scale efforts include ComplexTempQA (Gru-
ber et al., 2025), which offers 100 million syn-
thetic questions over Wikipedia, and History-
BankQA (Mandal et al., 2025), which targets
historical events across multiple languages.
•LLM-Oriented Benchmarks:The TIME
benchmark (Wei et al., 2025) evaluates LLMs
on explicit temporal reasoning tasks across
text, news, and dialogue. Similarly, Time-
lineQA (Tan et al., 2023) tests an agent’sability to construct timelines from single-
document contexts.
Limitation:As noted in recent systematic re-
views (Brown et al., 2025), these benchmarks pri-
marily evaluate the final generated answer. They
do not penalize systems that hallucinate the correct
date or retrieve temporally irrelevant documents
(e.g., retrieving 2020 data to answer a 2010 ques-
tion) as long as the final output is correct. TEMPO
uniquely evaluates the intermediate retrieval step,
ensuring that the evidence itself is temporally valid.
H.3 Reasoning-Intensive Retrieval
A parallel trend is the emergence of "Reasoning-
Intensive Retrieval," where the difficulty lies in the
logic rather than the keyword overlap.
•Atemporal Reasoning:BRIGHT (Su et al.,
2024) and RAR-b (Xiao et al., 2024) demon-
strated that state-of-the-art retrievers fail on
queries requiring logical deduction, coding, or
economic reasoning.
•Reasoning Models:New architectures like
ReasonIR (Shao et al., 2025), RaDeR (Das
et al., 2025), and query-reasoning rein-
forcers (Qin et al., 2025) have been proposed
to bridge this gap.
However, these benchmarks lack a temporal dimen-
sion. A query in BRIGHT might ask about a static
code function, whereas a query in TEMPO asks
how that function’s security vulnerabilities evolved
over five years. TEMPO is the first to combine the
logical complexity of BRIGHT with the temporal
constraints of TimeQA.
H.4 Temporal RAG and Knowledge Graphs
The intersection of Temporal Reasoning and RAG
is a nascent field. ChronoQA (Chen et al., 2025)
specifically targets "temporal-sensitive RAG," but
it is restricted to Chinese news articles and focuses
on recency bias updates. In the Knowledge Graph
(KG) domain, architectures like TimeR4 (Qian
et al., 2024) and TempoQR (Mavromatis et al.,
2022) attempt to embed temporal scopes into graph
nodes. Recent frameworks like CRP-RAG (Xu
et al., 2024) attempt to plan "knowledge actions"
to answer complex logical queries.
TEMPO complements these works by providing
a domain-diverse, English-language testbed that

requires cross-period analysis—synthesizing evi-
dence from a baseline period (e.g., "pre-2008 cri-
sis") and a comparison period (e.g., "post-2010
recovery")—a capability not explicitly measured
by existing temporal RAG benchmarks.
I Dataset Examples

Table 44:History example.A randomly sampled query with one positive and one negative document.
Query
Was the Tsar’s property separated from state property in Russian Empire in early 19th century
(regarding land)?
Was the Tsar’s property on land separated from state property in Russian Empire in early 19th century?
I mean, were there Tsar’s serfs who were not state serfs?
Example positive document
The appanage peasants lived on the personal properties of the Romanov family; Alexander II granted
them personal freedom in 1863. They received land allotments in 1863 and were placed on forty-nine-
year redemption payments in 1865. The state peasants lived on state lands under state administrators;
they received freedom in 1866.
The core "freedom" the peasants received was the elimination of the personal, arbitrary, and capricious
power of their noble and state masters. Members of the noble landowning estate and the tsar’s agents
could no longer buy and sell peasants, mortgage them for cash, order their daily labors, determine
whom and when they married, move them from one estate to another, break up families, beat them,
claim sexual rights over them, exile them to Siberia, impose both police and judicial authority over
them, demand that they gather forest products such as berries for their masters’ larders, or decide who
would enter military service for virtually their entire adult lives.
The emancipation legislation involved a land reform that transferred as much as half of the nobility’s
land to the peasants. The reformers tried to design this transfer so that it would not cause dangerous
instability in the countryside. They also tried to soften the economic blows to the nobility and to
guarantee that peasants would continue to produce crops and pay their taxes. These aims
...
Example negative document
By an odd twist of fate, defeat in the war proved of value to the new Tsar. Although he had been
trained for government from an early age, foreign observers had remarked on how diffident and unsure
he appeared. The war changed all that. Coming to the throne in 1855 in the middle of the conflict,
Alexander II was unable to save Russia from military failure, but the humiliation convinced him that,
if his nation was to have stability and peace at home and be honoured abroad, military and domestic
reforms were vitally necessary. The first step on that path would be the removal of serfdom, whose
manifest inefficiency benefited neither lord, peasant, nor nation. Alexander declared that, despite
Russia’s defeat, the end of the war marked a golden moment in the nation’s history. Now was the hour
when every Russian, under the protection of the law, could begin to enjoy ‘the fruits of his own labours’.
Alexander was right in thinking the time was propitious. It had long been appreciated that some land
reform was necessary. To the social and economic arguments were now added powerful military ones.
The army was the great symbol of Russia’s worth. As long as its army remained strong Russia could
afford to ignore its backwardness as a nation. But the Crimean defeat had undermined this notion of
Russia’s invincibility. Few now had reasoned objections to reform. Serfdom was manifestly not wor
...

Table 45:Hsm example.A randomly sampled query with one positive and one negative document.
Query
What is the most ancient milestone of mathematical reasoning or mathematical knowledge?
I know about the Plimpton 322 tablet and Pythagorean triples. But can we be sure that this is the
first instance of mathematical reasoning? I am talking about notions, propositions, or questions about
mathematics.
I ask you if you can say what the milestone of mathematical reasoning or mathematical knowledge is,
as accepted by the scientific community.
I have two ideas as to what ancient notions of mathematics could be like. Let’s say that humans were a
band of Paleolithic hunters or gatherers. If the group gets $N$ items of food, then, leaving out social
synergies, the appropriate distribution is the Euclidean division between the participants. If there are
paintings of two animals in a cave, does this mean that Paleolithic men/women had the notion of the
integer $2$? Are there scientific thoughts
...
Example positive document
It consists of 29 distinct notches that were deliberately cut into a baboon’s fibula.
The bone is between 44,200 and 43,000 years old, according to 24 radiocarbon datings. This is far
older than the Ishango bone with which it is sometimes confused. Other notched bones are 80,000
years old but it is unclear if the notches are merely decorative or if they bear a functional meaning.
According to The Universal Book of Mathematics, the Lebombo bone’s 29 notches “may have been
used as a lunar phase counter, in which case African women may have been the first mathematicians,
because keeping track of menstrual cycles requires a lunar calendar.” However, the bone is clearly
broken at one end, so the 29 notches may or may not be a minimum number. In the cases of other
notched bones since found globally, there has been no consistent notch tally, many being in the 1–10
range. The Lebombo bone resembles a calendar used by the early men of the area, coming from the
San clans of Namibia; this way of making tallies is still used by the San people today.
Lebombo Ishango bones
Top image: Lebombo bone. Bottom: Ishango bone with prime numbers engraving (J.D. Loreto and
D.H. Hurlbert Smithsonian)
According to The Universal Book of Mathematics, the Lebombo bone’s 29 notches “may have been
used as a lunar phase counter, in which case African women may have been the first mathematicians,
because ke
...
Example negative document
The discovery made by Otto Neugebauer and his assistant in the 1940s was an important one. The
numbers in Plimpton 322 are what are now called Pythagorean triples. It gives the short side and
the diagonal (hypotenuse) of 15 right triangles. The long sides of the right triangles are not shown.
As we will see below, the 15 right triangles have steadily decreasing slopes. The Sumerians in the
Old Babylonian period knew about the Pythagorean theorem over 1,000 years before the time of
Pythagoras!
Since the discovery made by Otto Neugebauer, Plimpton 322 was a subject of extensive research by
mathematicians. Obviously mathematicians are intrigued by the connection of a 4000-year tablet with
modern mathematics. Because of the intricate mathematical interpretations they made of the tablet,
many mathematicians thought highly of the tablet. For example, the author of the tablet must be a
mathematical prodigy or a professional mathematician, doing high level research in the Old Babylonian
Period.

Table 46:Politics example.A randomly sampled query with one positive and one negative document.
Query
What was the most succesful military dictatorship in the last 200 years in terms of economic growth?
Military dictatorships are controlled by military officers. I am wondering which military dictatorship
was the most economically successful, you can use any figure like GDP growth, GDP per capita,
GDP (PPP), etc. to base your argument. I don’t think the current Chinese government was a military
dictatorship. Maybe Mao’s government, but not the government that followed, so I am guessing China
wouldn’t count.
Example positive document
Park began a series of economic reforms that eventually led to rapid and unprecedented economic
growth and industrialization, a phenomenon that is now known as the Miracle on the Han River. This
made South Korea one of the fastest growing economies of the 1960s and 1970s, albeit with costs to
labor rights. This era also saw the formation of chaebols: family companies supported by the state
similar to the Japanese zaibatsu. Examples of significant chaebols include Hyundai, LG, and Samsung.
Although popular during the 1960s, Park’s popularity started to plateau by the 1970s, with closer
than expected victories during the 1971 presidential election and the subsequent legislative elections.
In 1972, Park declared martial law after carrying out a self-coup. He then introduced the highly
authoritarian Yushin Constitution, ushering in the Fourth Republic. Now ruling as a dictator, he
constantly repressed political opposition and dissent and completely controlled the military. He also
had much control over the media and expressions of art. In 1979, Park was assassinated by his close
friend Kim Jae-gyu, director of the KCIA, following the Busan–Masan Uprising.[2] Whether the
assassination was spontaneous or premeditated remains unclear to this day. Economic growth continued
in spite of the 1979 coup d’état and considerable political turmoil in the wake of his assassination. He
was soon
...
Example negative document
Determining adequate levels of military spending and sustaining the burden of conflicts have been
among key fiscal problems in history. Ancient societies were usually less complicated in terms of the
administrative, fiscal, technological, and material demands of warfare. The most pressing problem was
frequently the adequate maintenance of supply routes for the armed forces. On the other hand, these
societies were by and large subsistence societies, so they could not extract massive resources for such
ventures, at least until the arrival of the Roman and Byzantine Empires. The emerging nation states
of the early modern period were much better equipped to fight wars. On the one hand, the frequent
wars, new gunpowder technologies, and the commercialization of warfare forced them to consolidate
resources for the needs of warfare. On the other hand, the rulers had to – slowly but surely – give up
some of their sovereignty to be able to secure required credit both domestically and abroad. The Dutch
and the British were masters at this, with the latter amassing an empire that spanned the globe at the
eve of the First World War.

Table 47:Hsm example.A randomly sampled query with one positive and one negative document.
Query
How accurate was the measurement of the period of Earth’s orbit in the 19th Century?
There was a section on my textbook on history of theories of sun’s energy source.
It talks about how the Meteorite Theory was dismissed, as it would decrease the period of Earth’s orbit
by 2 seconds per year due to increased mass of the Sun.
This theory was dismissed due to disagreeing with observation. And the textbook says the change in
period is "easily measurable"
My question is how is 2 seconds difference easily measurable in Nineteenth Century?
Example positive document
In the nineteenth century, scientists thought that the source of the Sun’s heat might be the mechanical
motion of meteorites falling into it. Their calculations showed, however, that in order to produce the
total amount of energy emitted by the Sun, the mass in meteorites that would have to fall into the
Sun every 100 years would equal the mass of Earth. The resulting increase in the Sun’s mass would,
according to Kepler’s third law, change the period of Earth’s orbit by 2 seconds per year. Such a change
would be easily measurable and was not, in fact, occurring. Scientists could then disprove this as the
source of the Sun’s energy.
Gravitational Contraction as a Source of Energy
Proposing an alternative explanation, British physicist Lord Kelvin and German scientist Hermann
von Helmholtz (Figure 16.2), in about the middle of the nineteenth century, proposed that the Sun
might produce energy by the conversion of gravitational energy into heat. They suggested that the
outer layers of the Sun might be “falling” inward because of the force of gravity. In other words, they
proposed that the Sun could be shrinking in size, staying hot and bright as a result.
Kelvin (1824–1907) and Helmholtz (1821–1894).
Left: photograph of William Thomson (Lord Kelvin). Right: photograph of Hermann von Helmholtz.
Figure 16.2. (a) British physicist William Thomson (Lord Kelvin) and (b) German scien
...
Example negative document
The Principia is the founding document of physics and astronomy as we know them, and it played a
key role in the scientific revolution four centuries ago. For two centuries afterward, mathematicians
worked out the details of Newtonian mechanics, which led to determinism. The rise of modern
physics in the early twentieth century undermined determinism, leading to indeterminism. In the
seventeenth century, the so-called Enlightenment hijacked science, robbing it of its foundation in a
worldview that had biblical elements, replacing it with a foundation of humanism. This has led to
a growing hostility toward any concern of theism in scientific endeavors. The trends undermine the
worldview that created science in the first place. Consequently, the future of science may be in question.
Keywords: philosophy of science, determinism, deism, positivism, conflict thesis, special relativity,
general relativity, quantum mechanics, Copenhagen interpretation

J Annotation Guidelines
This section provides detailed guidelines for anno-
tators constructing the TEMPO dataset. The an-
notation process involves selecting StackExchange
posts with temporal reasoning requirements, identi-
fying temporally relevant documents, mining hard
negatives, and annotating temporal metadata at mul-
tiple levels.
J.1 Query Selection and Filtering
Annotators browse Stack Exchange posts from
newest to oldest within their assigned domain and
select posts meeting ALL of the following criteria:
Required Criteria:
1.High-quality answer: The post must have at
least one answer that is either:
•Accepted by the question author (marked
with green checkmark), OR
• Has received more than 10 upvotes
2.Temporal reasoning requirement: The post
must require temporal reasoning to answer.
This includes questions that:
•Track changes or evolution over time
(e.g., “How has X changed since Y?”)
•Compare historical baselines with cur-
rent states (e.g., “What was different be-
fore/after Z?”)
•Require understanding temporal depen-
dencies or causation
•Ask about trends, patterns, or develop-
ments across time periods
•Need cross-period evidence synthesis to
answer
3.Technical complexity: The question requires
reasoning beyond simple date lookup or key-
word matching. Avoid questions that can be
answered by retrieving a single date or simple
fact.
4.Temporal signals: The post should contain
explicit or implicit temporal signals such as:
•Specific dates, years, or time periods
(e.g., “in 1914”, “during the 1990s”)
•Relative temporal references (e.g., “be-
fore the war”, “after GDPR”, “since
2017”)
•Temporal keywords (e.g., “evolution”,
“history”, “change”, “trend”, “origin”)Exclusion Criteria:
•Posts with only simple fact-seeking queries
(e.g., “When did X happen?”)
•Questions where temporal aspects are merely
supplementary, not central to the answer
•Opinion-based or subjective questions without
temporal grounding
•Posts where answers rely purely on specula-
tion without temporal evidence
• Duplicate or near-duplicate questions
•Questions that can be answered by retrieving
a single Wikipedia date entry
J.2 Constructing Queries
For each selected post, construct the temporal query
as follows:
Step 1: Extract text content
• Combine the post title and body text
•Preserve HTML formatting where it aids read-
ability (lists, emphasis)
•Retain technical terminology, historical refer-
ences, and temporal expressions
•Remove broken links and irrelevant HTML
artifacts
Step 2: Verify temporal complexity
•Confirm the query requires multi-step tempo-
ral reasoning
•Identify the temporal scope (specific years,
decades, centuries, or relative periods)
•Determine if cross-period analysis is needed
(comparing multiple time periods)
•Note key temporal anchors mentioned in the
query
Step 3: Classify temporal characteristics
•Identify the primary temporal intent
(when/duration/order/before_after/ongoing
_status/period_definition/timeline)
•Assign the temporal reasoning class (see §J.6)
•Extract explicit temporal signals and events
from the query text

J.3 Positive Document Construction
Positive documents must providetemporal evi-
dencethat helps reason through the query’s tempo-
ral aspects. Follow these steps:
Step 1: Discover candidate documents
Use TWO methods to find candidate documents:
Method A - Answer links:
•Visit all external URLs linked in accepted or
highly-voted answers
•Check if the linked page contains temporal
information relevant to the query
•Prioritize sources that discuss the time periods
mentioned in the query
Method B - AI-assisted discovery:
•Use Gemini (Google AI) with the following
prompt template:
“Give me articles from the internet to answer
this temporal query: [paste full question text].
Focus on sources that discuss the time periods
and temporal evolution mentioned. ”
•Visit suggested web pages and evaluate their
temporal relevance
Step 2: Evaluate temporal relevance
For each candidate web page, extract passages
that meet the temporal relevance criteria:
A document/passage is POSITIVE if it:
•Provides baseline temporal evidence: Con-
tains information about the historical state or
starting point referenced in the query
•Provides comparison temporal evidence:
Contains information about the later state or
endpoint for cross-period queries
•Explains temporal evolution: Describes how
phenomena changed, developed, or evolved
over the relevant time periods
•Contains temporal context: Provides his-
torical background necessary to understand
temporal relationships
•Includes temporal synthesis: Helps connect
evidence across multiple time periods to form
a complete answer
A document/passage is NOT positive if it:•Discusses the topic but lacks temporal infor-
mation or covers wrong time periods
•Only mentions dates without explaining tem-
poral relationships or changes
•Provides general background without address-
ing the specific temporal scope of the query
•Covers only one time period when the query
requires cross-period comparison
•Contains anachronistic information that
doesn’t match the query’s temporal focus
Step 3: Extract passages with temporal anno-
tations
•For each positive web page, identify and ex-
tract temporally relevant passages
•Each passage should be self-contained (typi-
cally 1–5 paragraphs)
• Annotate each passage with:
–Temporal signals present in the passage
–Time scope (start and end dates in ISO
format when determinable)
–Temporal events mentioned
–Dominant tense
(past/present/future/mixed)
•Record confidence score (0–1) for temporal
annotations
Step 4: Ensure temporal coverage
•For cross-period queries, ensure positive doc-
uments cover BOTH baseline and comparison
periods
•Verify that the combined positive documents
provide sufficient temporal evidence to an-
swer the query
•If temporal coverage is incomplete, search for
additional documents
Step 5: Record metadata
• Source URL of the document
•Type of source (Wikipedia, academic article,
news archive, official documentation, blog,
etc.)
• Date accessed
• Temporal scope covered by the document
• Brief justification for temporal relevance

J.4 Hard Negative Mining
Hard negatives for TEMPO are documents that are
topically related but temporally incomplete or ir-
relevant. These prevent models from relying on
simple semantic matching without temporal under-
standing.
Types of temporal hard negatives:
1.Wrong time period: Documents discussing
the same topic but in a different time period
than the query requires
2.Missing temporal coverage: Documents cov-
ering only one time period when cross-period
analysis is needed
3.Temporally vague: Documents discussing
the topic without specific temporal grounding
4.Anachronistic: Documents with temporal in-
formation that doesn’t align with the query’s
temporal scope
Step 1: Generate hard negative search query
•Use the GPT-4o prompt (Appendix B.1) to
generate:
1.A search query designed to find semanti-
cally similar but temporally incomplete
content
2.List of entities, events, and temporal an-
chors from the post
•The LLM outputs JSON with llm_summary
andentities_events
Step 2: Collect hard negative URLs
•Use the generated llm_summary as your
Google search query
•Additionally search using combinations of
entities_events WITHOUT temporal qual-
ifiers
• Collect URLs that are:
–Topically related to the query domain
–Semantically similar to the query content
–BUT missing the specific temporal infor-
mation or time periods needed
Step 3: Extract hard negative passages
For each hard negative URL:•Extract passages that are topically related but
temporally inadequate
•Hard negatives should be challenging—they
might discuss the same general topic but:
–Cover a different time period than re-
quired
–Lack temporal specificity needed to an-
swer the query
–Miss one of the required time periods for
cross-period queries
–Discuss temporal aspects tangentially
without depth
• Avoid completely unrelated content
Step 4: Verify hard negative quality
•Confirm hard negatives are semantically simi-
lar (would rank highly with keyword match-
ing)
•Verify they fail temporal requirements (wrong
period, incomplete coverage, or temporally
vague)
•Ensure a mix of hard negative types for diver-
sity
J.5 Step-wise Retrieval Planning
For Task 2 evaluation, decompose each query into
sequential retrieval steps:
Step 1: Analyze temporal structure
•Identify distinct temporal aspects or time peri-
ods in the query
•Determine the logical order of retrieval (e.g.,
baseline first, then comparison)
• Note dependencies between retrieval steps
Step 2: Create retrieval steps
•Each step should target a specific temporal
aspect or time period
•Steps should be concrete and actionable
(e.g., “Retrieve historical statistics from 2013–
2017”)
• Typically 2–4 steps per query
• Example step structure:
–Step 1: Retrieve baseline evidence from
[time period 1]

–Step 2: Retrieve comparison evidence
from [time period 2]
–Step 3: Retrieve documents explaining
the change/evolution
Step 3: Map gold documents to steps
•Assign each positive document to the retrieval
step(s) it satisfies
•A document may satisfy multiple steps if it
covers multiple time periods
•Ensure each step has at least one mapped gold
document
J.6 Temporal Reasoning Classes
Assign each query to one primary temporal reason-
ing class:
1.Event Analysis & Localization (EAL): Pin-
pointing when events occurred and under-
standing their temporal context
2.Time Period Contextualization (TPC): Sit-
uating phenomena within specific historical
periods
3.Origins & Evolution Comparative (OEC):
Tracking how concepts evolved over time
4.Trends & Cross-Period Comparison (TCP):
Comparing states across multiple time periods
5.Event Verification & Authenticity (EV A):
Verifying temporal claims or dating artifacts
6.Materials & Artifacts Provenance (MAP):
Dating and tracing origins of physical items
7.Sources & Methods Documentation (SMD):
Understanding historical methodology and
sources
8.Causation Analysis (CAU): Analyzing tem-
poral cause-effect relationships
9.Historical Attribution & Context (HAC):
Attributing ideas or events to correct time pe-
riods
J.7 Quality Control and Review
Self-check before submission:
1.Does the selected post genuinely require tem-
poral reasoning (not just date lookup)?2.Did you verify that answers have >10 votes
or are accepted?
3.Do positive documents provide temporal evi-
dence for the required time periods?
4.For cross-period queries, do positive docu-
ments cover BOTH baseline and comparison
periods?
5.Are hard negatives temporally inadequate but
semantically similar?
6.Is the retrieval plan logical and are gold docu-
ments correctly mapped to steps?
7.Are all temporal annotations (signals, events,
time scope) accurate?
Annotation review process:
•Initial annotations are reviewed by two PhD
students with domain expertise
•Reviewers check: (1) temporal relevance of
positive documents, (2) temporal inadequacy
of hard negatives, (3) accuracy of temporal
annotations, (4) validity of retrieval plans
•Inter-annotator agreement measured using Co-
hen’s Kappa
•Only annotations withunanimous approval
from all reviewers are retained
•Disagreements are resolved through discus-
sion or the example is discarded
Common Mistakes to Avoid:
1.Selecting simple date-lookup queries: The
query must require temporal reasoning, not
just retrieving when something happened
2.Incomplete temporal coverage: For cross-
period queries, ensure positive documents
cover all required time periods
3.Too-easy temporal negatives: Hard nega-
tives should be semantically similar but fail on
temporal grounds—don’t include completely
off-topic documents
4.Ignoring temporal scope: Ensure positive
documents cover the specific time periods in
the query, not just any temporal information

5.Incorrect temporal annotations: Double-
check time scope annotations, especially ISO
date formats
6.Illogical retrieval plans: Steps should follow
a natural temporal progression and be inde-
pendently actionable
7.Missing step-document mappings: Every
retrieval step must have at least one gold doc-
ument mapped to it
J.8 Domain-Specific Annotation Notes
For Blockchain domains (Bitcoin, Cardano,
Iota, Monero):
•Temporal focus typically on protocol evolu-
tion, market changes, and technology updates
•Time periods often span 2009–present with
rapid changes
•Hard negatives can discuss similar blockchain
concepts but from different protocol versions
or time periods
•Sources include whitepapers, technical docu-
mentation, and cryptocurrency news archives
For Social Sciences (Economics, Law, Politics,
History):
•Temporal scope may span centuries; ensure
positive documents match the query’s era
•Cross-period queries common (e.g., compar-
ing policies before/after major events)
•Prefer academic sources, government docu-
ments, and authoritative analyses
•Hard negatives often discuss same topic but
in wrong historical period
For Applied domains (Quant, Travel, Work-
place, Genealogy):
•Temporal aspects often relate to policy
changes, regulation updates, or practice evo-
lution
•For Genealogy, temporal accuracy is critical—
verify historical dates carefully
•Hard negatives may discuss similar practices
but from different eras•Balance between academic sources and prac-
tical documentation
For STEM (History of Science and Mathemat-
ics):
•Focus on evolution of scientific ideas and
mathematical concepts
• Attribution to correct time periods is crucial
•Positive documents should explain how ideas
developed temporally
•Hard negatives might discuss same concepts
but misattribute time periods
•Prefer peer-reviewed history of science litera-
ture
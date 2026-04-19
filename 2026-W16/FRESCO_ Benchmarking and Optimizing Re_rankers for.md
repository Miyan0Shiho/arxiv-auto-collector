# FRESCO: Benchmarking and Optimizing Re-rankers for Evolving Semantic Conflict in Retrieval-Augmented Generation

**Authors**: Sohyun An, Hayeon Lee, Shuibenyang Yuan, Chun-cheng Jason Chen, Cho-Jui Hsieh, Vijai Mohan, Alexander Min

**Published**: 2026-04-14 17:04:25

**PDF URL**: [https://arxiv.org/pdf/2604.14227v1](https://arxiv.org/pdf/2604.14227v1)

## Abstract
Retrieval-Augmented Generation (RAG) is a key approach to mitigating the temporal staleness of large language models (LLMs) by grounding responses in up-to-date evidence. Within the RAG pipeline, re-rankers play a pivotal role in selecting the most useful documents from retrieved candidates. However, existing benchmarks predominantly evaluate re-rankers in static settings and do not adequately assess performance under evolving information -- a critical gap, as real-world systems often must choose among temporally different pieces of evidence. To address this limitation, we introduce FRESCO (Factual Recency and Evolving Semantic COnflict), a benchmark for evaluating re-rankers in temporally dynamic contexts. By pairing recency-seeking queries with historical Wikipedia revisions, FRESCO tests whether re-rankers can prioritize factually recent evidence while maintaining semantic relevance. Our evaluation reveals a consistent failure mode across existing re-rankers: a strong bias toward older, semantically rich documents, even when they are factually obsolete. We further investigate an instruction optimization framework to mitigate this issue. By identifying Pareto-optimal instructions that balance Evolving and Non-Evolving Knowledge tasks, we obtain gains of up to 27% on Evolving Knowledge tasks while maintaining competitive performance on Non-Evolving Knowledge tasks.

## Full Text


<!-- PDF content starts -->

FRESCO: Benchmarking and Optimizing Re-rankers
for Evolving Semantic Conflict in
Retrieval-Augmented Generation
Sohyun An1,2,∗,Hayeon Lee1,Shuibenyang Yuan1,Chun-cheng Jason Chen1,Cho-Jui Hsieh2,Vijai
Mohan1,Alexander Min1
1Meta Superintelligence Labs,2UCLA
∗Work done at Meta,†Joint last author
Retrieval-Augmented Generation (RAG) is a key approach to mitigating the temporal staleness of large
language models (LLMs) by grounding responses in up-to-date evidence. Within the RAG pipeline, re-
rankers play a pivotal role in selecting the most useful documents from retrieved candidates. However,
existing benchmarks predominantly evaluate re-rankers in static settings and do not adequately
assess performance under evolving information—a critical gap, as real-world systems often must
choose among temporally different pieces of evidence. To address this limitation, we introduce FRESCO
(FactualRecency andEvolvingSemanticCOnflict), a benchmark for evaluating re-rankers in temporally
dynamic contexts. By pairing recency-seeking queries with historical Wikipedia revisions, FRESCO
tests whether re-rankers can prioritize factually recent evidence while maintaining semantic relevance.
Our evaluation reveals a consistent failure mode across existing re-rankers: a strong bias toward
older, semantically rich documents, even when they are factually obsolete. We further investigate an
instruction optimization framework to mitigate this issue. By identifying Pareto-optimal instructions
that balanceEvolvingandNon-Evolving Knowledgetasks, we obtain gains of up to 27% on Evolving
Knowledge tasks while maintaining competitive performance on Non-Evolving Knowledge tasks.
Date:April 17, 2026
Correspondence:Sohyun An atsohyun0423@cs.ucla.edu
Code:https://github.com/facebookresearch/fresco
1 Introduction
Large Language Models (LLMs) have achieved strong performance across a wide range of NLP tasks (Brown
et al., 2020; Wei et al., 2021; Bommasani et al., 2022; Chowdhery et al., 2023; Zhao et al., 2025), yet their
practical reliability is often constrained bytemporal staleness—the mismatch between static pretraining
corpora and a rapidly changing world. Retrieval-Augmented Generation (RAG) has therefore become a
standard remedy, grounding LLM outputs in external and up-to-date evidence (Lewis et al., 2020; Karpukhin
et al., 2020; Gao et al., 2023; Izacard et al., 2023; Asai et al., 2024). Within the RAG pipeline, re-rankers act
as a key gatekeeper: given a pool of retrieved candidates, they must prioritize the most useful evidence so
that downstream generation remains factually correct (Nogueira and Cho, 2019; Li et al., 2025; Xiao et al.,
2023; Wang et al., 2025; Sun et al., 2023; Pradeep et al., 2023a,b).
Despite this central role, the prevailing evaluation paradigm for re-ranking largely assumes a non-evolving
information environment. Widely used benchmarks (Bajaj et al., 2018; Craswell et al., 2021; Kwiatkowski
et al., 2019; Thakur et al.) typically operationalize relevance through semantic overlap or topical alignment,
implicitly treating the target information asfixed over time. This assumption is frequently violated in
real-world RAG deployments: when facts change, retrieval systems can surface multiple snapshots of ostensibly
relevant evidence, some outdated and others reflecting the current state. For example, a query asking for the
latest stable version of vLLMmay retrieve two release notes or changelogs—one reporting version 0.5.0 from
several weeks ago and another reporting version 0.6.0 from two days ago. Both documents can appear highly
relevant under conventional semantic matching, yet only the latter is chronologically valid for answering the
query. We refer to this phenomenon asEvolving Semantic Conflict: the re-ranker must choose among multiple
1arXiv:2604.14227v1  [cs.IR]  14 Apr 2026

semantically relevant candidates whose content conflicts due to temporal evolution.
Handling temporal discrimination—incorporating factual recency in addition to semantic relevance—is a
practical requirement for robust RAG, particularly when multiple candidates are similarly relevant to the
query. Without this capability, a re-ranker may systematically promote semantically informative but obsolete
documents, causing the generator to produce confident yet incorrect answers. However, current evaluation
regimes provide limited visibility into this failure mode, leaving it unclear whether existing re-rankers can
reliably resolve conflicts induced by evolving knowledge without sacrificing relevance.
To bridge this gap, we introduce FRESCO(FactualRecency andEvolvingSemanticCOnflict), a benchmark
for evaluating re-rankers in settings wheresemantic relevance is necessary but not sufficient. FRESCOpairs
recency-seeking queries with historicalWikipediarevisions to construct candidate sets in which passages
remain topically aligned yet differ in factual recency. This design isolates the key challenge faced by real-world
RAG systems: selecting evidence that is both relevantandchronologically valid when semantically similar
candidates disagree. Our evaluation identifies a consistent failure mode across existing re-rankers: they exhibit
a strongsemantic bias, frequently preferring older, contextually dense documents over newer ones even when
the older evidence is factually obsolete.
We further study a Pareto-based instruction optimization framework for LLM-based re-rankers, which take
instructions alongside the query and candidate documents. The framework explicitly captures the trade-off
betweenEvolving Knowledgetasks (where recency is essential) andNon-Evolving Knowledgetasks (where
semantic relevance is typically sufficient), yielding a spectrum of Pareto-optimal instructions that practitioners
can select based on deployment needs. Empirically, a Pareto-optimal instruction that emphasizes Evolving
Knowledge improves performance on such tasks by up to 27% while maintaining competitive results on
Non-Evolving Knowledge tasks.
In summary, our contributions are as follows:
•We formalizeEvolving Semantic Conflictand introduce FRESCO, a benchmark that systematically
evaluates re-rankers under temporally evolving information.
•We identify a consistent failure mode in existing re-rankers, showing a strong bias toward semantically
rich but factually obsolete documents.
•We investigate a Pareto-based instruction optimization framework for LLM-based re-rankers, enabling
controllable trade-offs betweenEvolving KnowledgeandNon-Evolving Knowledgetasks.
2 Related Work
Re-ranking and Its Evaluation.Two-stage retrieval pipelines—retrieve followed by re-rank—are a standard
approach in information retrieval and are widely used to support downstream applications such as RAG
(Gao et al., 2023). Re-rankers have progressed from BERT-style cross-encoders (Nogueira and Cho, 2019) to
more recent LLM-based re-rankers (Pradeep et al., 2023a; Zhang et al., 2025), with performance commonly
reported on benchmarks such as MS MARCO (Bajaj et al., 2018), TREC DL (Craswell et al., 2021), Natural
Questions (Kwiatkowski et al., 2019), and BEIR (Thakur et al.). However, these benchmarks largely encode
a temporally static notion of relevance: once labels are created, they are treated as invariant. As a result,
they provide limited insight into whether a re-ranker can select chronologically valid evidence when multiple
semantically relevant candidates reflect different points in time.
QA Benchmarks under Evolving Knowledge.Evolving, time-sensitive knowledge has motivated benchmarks
such as RealtimeQA (Kasai et al., 2023) and FreshQA (Vu et al., 2024). These resources are valuable for
measuring end-to-end QA performance under recency demands, but they are not designed to isolate the
re-ranking decision itself. In particular, they typically evaluate final answer accuracy and do not provide
controlled candidate pools containing competing, semantically relevant evidence with explicit temporal conflicts.
This makes it difficult to perform a granular analysis of a re-ranker’s ability to resolveEvolving Semantic
Conflict. In contrast, FRESCOleverages Wikipedia revision histories to construct timestamp-conditioned
2

Positive PassagesQuery: Who is the most recent person to lead Casa Pia A.C. as its head coach?
Timestamp: 2025 -08-31 (Explicitly Recency -Seeking)
Title: Casa Pia A.C.
Casa Pia A.C. is …
Their current head 
coach is Filipe Martins .
  …
 ## Managerial history
Casa Pia A.C. have had 
many managers …
Timestamp: 2022 -04-05
 
Time
Query: Who is handling the head coaching duties for Casa Pia A.C.? 
Timestamp: 2025 -08-31 (Implicitly Recency -Seeking)subject : Casa Pia A.C.
object : Filipe Martins
start time : 2020 -09-01 
end time : 2023 -11-13
object : Gonçalo Santos
start time : 2024 -02-16 
end time : 2024 -06-30Title: Casa Pia A.C.
Casa Pia A.C. is …
Their current head 
coach is Gonçalo 
Santos .  …
## Managerial history
Casa Pia A.C. have had 
many managers …
Timestamp: 2024 -05-23
 object : João Pereira
start time : 2024 -07-01
end time : None (i.e., Present)Title: Casa Pia A.C.
Casa Pia A.C. is …
Their current head 
coach is João Pereira .  
…
  ## Managerial history
Casa Pia A.C. have had 
many managers …
Timestamp: 2025 -08-27
 relation : head coach
…Negative PassagesCasa Pia A.C. is …
Their current head coach 
is João Pereira.
Timestamp: 2025 -08-27
Casa Pia A.C. is …
Their current head coach 
is Gonçalo Santos.
Timestamp: 2024 -05-23
## Managerial history
Casa Pia A.C. have had 
many managers …
Timestamp: 2025 -08-27
…WikiData
WikiPedia
... ...Figure 1 Illustration of the construction pipeline of FRESCO .Time-annotated facts from Wikidata (left) are aligned with
Wikipedia revisions (middle) to create positive and negative (outdated or irrelevant) passages (right) for explicit and
implicit recency-seeking queries (top).
candidate pools, enabling scalable benchmarking with fine-grained control over temporal conflicts while
keeping semantic relevance comparable across candidates.
Incorporating Temporal Signals in Ranking.A growing line of work incorporates temporal signals into ranking
and retrieval by combining semantic relevance with recency-aware scoring, explicit time features, or modular
symbolic components (Gade et al., 2025; Vu et al., 2024; Siyue et al., 2024). While these approaches can
improve performance onEvolving Knowledgetasks, they often introduce an explicit recency preference that
may be suboptimal when recency is irrelevant or misleading (e.g.,Non-Evolving Knowledgetasks). Our
Pareto-based instruction optimization framework complements this direction by identifying a spectrum of
Pareto-optimal instructions, enabling practitioners to choose operating points that balance Evolving and
Non-Evolving Knowledge performance according to application needs.
3FRESCOBenchmark
FRESCOis designed to evaluate re-rankers in controlled settings where semantic relevance alone is insufficient
and temporal validity becomes necessary for selecting correct evidence. Each instance pairs a recency-seeking
query with a candidate pool of passages that vary in semantic relevance and factual recency. To enable scalable
and reproducible construction, we leverage two complementary resources.Wikidataprovides structured facts
as tuples with explicit temporal qualifiers (e.g., start and end dates), capturing how facts change over time.
Wikipediarevision histories then provide textual evidence corresponding to those facts at different points in
time. By aligning these sources, we automatically generate benchmark instances without manual annotation.
Figure 1 provides an overview of howFRESCOwas constructed.
3.1 Problem Formulation
We represent a query as q= (xq, τq), where xqis the query text and τqis the timestamp at which the
query is posed. Each candidate passage is c= (xc, τc), where xcis the passage text and τcis the document
timestamp. To make temporal information available to the re-ranker, we append the timestamp to the textual
input (e.g., {x}\nTimestamp: {τ} ), where τis formatted as YYYY-MM-DDThh:mm:ssZ . Given a candidate set
C={c1, c2, . . . , c n}, the goal is to rank highest the candidate that is (i) semantically relevant to xqand
(ii) temporally valid at τq. Importantly, FRESCOincludes candidates that are temporally valid but provide
insufficient evidence to answer the query, as well as candidates that are semantically informative but factually
outdated. Therefore, neither semantic relevance nor temporal validity alone is sufficient to solve the task.
3

3.2 Construction Pipeline
The construction proceeds in three phases: (1) fact identification and evidence alignment, (2) query generation
and candidate pool construction, and (3) instance assembly via hard negative mining.
Phase 1: Fact Identification and Evidence Alignment.This phase identifies entities with verifiable temporal
changes in Wikidata and aligns each fact with supporting evidence in Wikipedia. We first query Wikidata for
relations with explicit temporal qualifiers (e.g., start time or end time). For each relation, we select entities
with a sufficiently informative history, defined as having at least two distinct time-annotated facts between
January 1, 2020, and a reference time τref(i.e., the query time assumed in our benchmark). Each fact is
represented as a tuple(s, r, o, t s, te), wheresandrdenote the subject and relation, ando,t s, andt edenote
the object value and its validity period (start and end times). For a fixed subject–relation pair, the object
value evolves over time; for example, for (Cristiano Ronaldo, member of sports team), ochanges from Juventus
to Al Nassr with corresponding validity periods. This filtering ensures that selected entities exhibit meaningful
temporal evolution rather than static attributes. For each fact tuple, we consider its validity interval[ ts, te)
and retrieve Wikipedia page revisions for entity screated during that interval. Because updates on Wikidata
and Wikipedia are not always synchronized (Jang et al., 2022), we apply a two-step verification procedure.
First, we verify that themost recentfact is supported by Wikipedia by parsing infoboxes from revisions in
the corresponding validity interval and checking whether the target object value ois explicitly stated; entities
lacking such evidence are discarded. Second, we verify thatat least oneearlier fact is also supported by
Wikipedia by identifying an older revision whose infobox contains a different object value o′̸=ocorresponding
to a previous state. This dual verification ensures that each retained entity is backed by textual evidence for
both the latest fact and a prior conflicting fact, providing the necessary ingredients forEvolving Semantic
Conflict.
Phase 2: Query Generation and Candidate Pool Construction.After aligning verified temporal facts with
Wikipedia revisions, we generate natural-language queries and construct corresponding candidate pools. For
each subject ( s)–relation ( r) pair, we create a query targeting its most recent fact using manually crafted,
relation-specific templates. To reflect real-world usage, we include both explicit and implicit recency-seeking
forms. Explicit forms contain temporal keywords (e.g., “Who is themost recentperson to lead { s} as its head
coach?”) while implicit forms convey the same intent through tense (e.g., “Who is handling the head coaching
duties for { s}?”). The final query is obtained by filling { s} with the subject name (e.g., Tottenham Hotspur).
For each query, we construct a candidate pool of section-level passages extracted from the verified Wikipedia
revisions. The positive passage is taken from the revision whose infobox matches the latest object value. We
construct challenging negatives from two complementary sources. The first type consists of passages from
the same (temporally valid) revision as the positive passage but drawn from sections that do not contain
sufficient evidence to answer the query. The second type consists of outdated passages describing earlier facts,
drawn from older revisions of the same entity and augmented with topically similar passages retrieved from a
static Wikipedia snapshot (dated 2018-12-20) using a BGE retriever (Xiao et al., 2023). This design yields
candidate pools containing passages that are semantically close to the query while introducing temporally
conflicting evidence.
Phase 3: Instance Assembly via Hard Negative Mining.In the final phase, we assemble benchmark instances
via hard negative mining. We consolidate all candidate negatives for a query into a single pool and score
each passage using Qwen3-Embedding-0.6B (Zhang et al., 2025). Each instance is represented as a triplet
(q,Cq,Rq), where qis the query, Cqcontains the positive passage and the top- khighest-scoring negatives
(k= 50), and Rqprovides binary labels (1 for the positive, 0 for negatives). This setup evaluates whether a
re-ranker can prioritize evidence that is both relevant and temporally valid when faced withEvolving Semantic
Conflict. In total, FRESCOcontains 3,658 queries per query type (7,316 queries overall), each paired with one
positive passage and fifty negatives.
3.3 Quality Validation
To validate FRESCO, we conduct a human evaluation on 200 randomly sampled instances. Three trained
annotators are shown four candidates: the pipeline-labeled positive passage and its three highest-scoring hard
4

Table 1 FRESCO benchmark results for existing re-rankers.Re-rankers are ordered by their release dates. Detailed model
specifications are provided in Section C. Cells are highlighted based on the numerical value normalized across each
column, with darker blue indicating a higher performance.
Model MAP MRR@5 MRR@10 nDCG@5 nDCG@10 Hit Rate@5 Hit Rate@10 Obsolete%Explicitly Recency-Seeking QueriesMonoT5 26.56 21.72 24.59 26.28 33.29 41.03 62.74 89.75
RankT5 38.85 35.14 37.37 40.21 45.60 55.52 72.17 90.24
UPR 35.69 32.86 34.34 36.69 40.47 52.93 64.00 86.18
RankGPT(gpt-3.5) 33.04 30.98 32.16 41.16 43.92 71.16 79.50 87.99
RankGPT(gpt-4o) 49.51 49.38 49.49 62.30 62.55 98.93 99.67 90.44
RankVicuna 38.14 34.32 36.48 38.33 43.62 50.55 67.09 89.29
RankZephyr 41.68 38.22 40.38 43.01 48.33 57.57 74.22 89.59
bce-reranker-base-v1 32.53 28.04 30.79 32.92 39.56 47.81 68.32 87.15
InRanker 53.18 50.25 52.04 36.56 43.02 65.12 78.49 91.20
mxbai-rerank-base-v1 27.22 22.59 25.17 27.55 33.80 42.70 62.06 89.72
Twolar 47.33 44.17 46.08 48.63 53.24 62.11 76.30 88.88
jina-reranker-v1-tiny-en 40.88 37.45 39.40 41.96 46.65 55.63 70.09 85.98
jina-reranker-v1-turbo-en 44.90 41.85 43.54 46.23 50.34 59.46 72.23 86.18
jina-reranker-v2 61.05 59.17 60.66 65.20 68.75 83.38 94.20 91.78
gte-multilingual-reranker-base 55.07 52.68 54.47 58.77 63.03 77.17 90.19 84.50
LdIR-Qwen2-reranker-1.5B 42.11 38.65 41.26 45.80 52.02 67.50 86.50 93.99
IncontextReranker 24.01 19.60 22.06 26.86 32.79 48.66 66.92 90.92
Qwen3-Reranker-0.6B 57.47 55.28 57.01 60.80 64.95 80.48 93.00 95.13
Qwen3-Reranker-8B 66.29 64.87 66.10 70.69 73.60 88.35 97.10 95.73Implicitly Recency-Seeking QueriesMonoT5 28.16 23.68 26.23 28.36 34.59 43.28 62.52 91.00
RankT5 37.24 33.54 35.68 38.57 43.78 53.83 69.98 90.11
UPR 39.44 36.76 38.16 40.09 43.62 54.81 65.31 87.08
RankGPT(gpt-3.5) 28.23 25.66 27.08 34.81 38.17 62.08 72.28 90.56
RankGPT(gpt-4o) 48.15 47.79 48.11 60.60 61.32 97.10 99.23 91.26
RankVicuna 38.85 35.26 37.18 39.60 44.36 52.76 67.74 89.95
RankZephyr 40.65 37.08 39.35 42.11 47.70 57.41 74.90 90.73
bce-reranker-base-v1 34.54 30.52 32.91 35.06 40.88 48.93 66.98 87.96
InRanker 56.82 54.00 55.67 36.63 43.00 65.61 78.08 90.21
mxbai-rerank-base-v1 33.18 29.02 31.47 34.33 40.31 50.46 69.05 90.60
Twolar 49.09 46.11 47.91 50.81 55.19 65.04 78.62 89.76
jina-reranker-v1-tiny-en 45.09 41.94 43.67 45.75 49.93 57.24 70.17 86.85
jina-reranker-v1-turbo-en 46.19 43.22 44.81 47.14 51.02 58.94 71.00 87.13
jina-reranker-v2 62.85 61.21 62.51 67.20 70.31 85.27 94.78 91.34
gte-multilingual-reranker-base 61.72 59.88 61.36 65.73 69.25 83.30 94.01 85.75
LdIR-Qwen2-reranker-1.5B 43.75 40.56 42.96 47.76 53.51 69.63 87.23 94.43
IncontextReranker 24.29 20.00 22.38 27.62 33.36 50.66 68.40 91.20
Qwen3-Reranker-0.6B 49.43 46.66 48.83 52.09 57.33 75.07 90.84 96.80
Qwen3-Reranker-8B 58.54 56.67 58.25 63.52 67.31 84.31 95.82 98.26
negatives retrieved by Qwen3-Embedding-0.6B. Annotators select the single passage that best supports the
query at the given query timestamp, and may choose Noneif no passage provides sufficient evidence. We
observe near-perfect inter-annotator agreement (Fleiss’ κ= 0.9689), and the human majority vote matches
our pipeline labels in 98.5% of cases. These results indicate that FRESCOprovides reliable supervision for
evaluating re-rankers under temporally evolving information. Further details are provided in Section E.
4 Benchmarking Re-rankers
Setting.We evaluate 19 existing re-rankers on FRESCOto assess their performance underEvolving Semantic
Conflict. Re-rankers are ordered by release date in Table 1. We report standard ranking metrics—MAP,
MRR@ k, nDCG@ k, and Recall@ k—for k∈ { 5,10}. We additionally report theObsolete Ratio (%): among
the negative passages ranked above the positive, the proportion that arefactually outdated(i.e., describing an
earlier state) rather thantemporally valid but contextually insufficient. A high Obsolete Ratio indicates that
errors are dominated by selecting obsolete evidence over up-to-date evidence.
Analysis.As shown in Table 1, FRESCOclearly differentiates existing re-rankers underEvolving Semantic
Conflict. Performance varies widely across models, with MAP ranging from 24.01 to 66.29 on explicitly
recency-seeking queries and from 24.29 to 62.85 on implicitly recency-seeking queries. More recently released
5

Table 2 Performance comparison on Evolving Knowledge ( DEK) and Non-Evolving Knowledge ( DNEK) task.Our instruction
optimization method (Pareto Solution 1-4) discovers a Pareto front offering superior trade-offs compared to baselines.
The trade-off plot is provided in Section B.5.
MethodDEK↑ DNEK↑
MAP MRR@10 nDCG@10 MAP MRR@10 nDCG@10
Base Model (Qwen3-Reranker-8B) 62.41 62.17 70.45 60.93 77.73 79.68
Temporal-Aware Models
TempRALM (Gade et al., 2025) 77.79 77.69 82.73 40.32 40.11 58.29
FreshPrompt (Vu et al., 2024) 57.64 57.68 66.26 61.3679.46 80.80
MRAG (Siyue et al., 2024) 78.41 77.88 80.85 59.18 76.65 78.63
Fine Tuning
Point-wise Finetuning 76.31 76.18 81.52 61.54 78.28 80.03
List-wise Finetuning 67.91 67.73 74.95 62.02 77.58 79.75
Ours
Pareto Solution 1 79.20 79.12 83.93 59.41 76.41 78.45
Pareto Solution 2 77.67 77.58 82.77 61.02 77.93 79.71
Pareto Solution 3 72.51 72.38 78.68 61.51 78.20 79.97
Pareto Solution 4 68.88 68.71 75.76 62.2779.00 80.55
re-rankers generally perform better, although the trend is not strictly monotonic. Among the strongest models
are RankGPT (gpt-4o), InRanker, jina-reranker-v2, gte-multilingual-reranker-base, and Qwen3-Reranker-8B,
whereas earlier or smaller models such as MonoT5, RankT5, UPR, IncontextReranker, and mxbai-rerank-
base-v1 tend to perform worse. Taken together, these results show that FRESCOis sufficiently discriminative
to separate re-rankers with stronger temporal discrimination from those that rely more heavily on semantic
similarity.
At the same time, the benchmark reveals a strikingly consistent failure mode across the model spectrum: the
Obsolete Ratio remains high for nearly all re-rankers (84%–98%). This indicates that errors are rarely driven
by topical irrelevance; instead, models often prefer semantically rich but outdated passages over temporally
valid evidence. We attribute this pattern to a mismatch between the requirements of temporally dynamic
RAG and prevailing training and evaluation paradigms, which mainly reward semantic overlap and relevance
matching while providing limited supervision for temporal validity under evolving knowledge (Bajaj et al.,
2018; Craswell et al., 2021; Kwiatkowski et al., 2019; Thakur et al.).
Qwen3-Reranker-8B, the strongest model on average, illustrates this tension especially clearly. Although it
achieves the best overall effectiveness, it also exhibits one of the highest Obsolete Ratios in the benchmark,
suggesting that its remaining errors stem less from semantic mismatch than from insufficient temporal
discrimination among multiple semantically plausible candidates. We also find that this model is sensitive to
how recency requirements are expressed: its MAP drops from 66.29 to 58.54 when the recency need is implicit
rather than explicit, even though the underlying information need is unchanged. This sensitivity points to
the potential of input optimization to better elicit temporal discrimination. In the next section, we therefore
explore an input optimization framework that steers the re-ranker toward resolving temporal contradictions
without degrading overall ranking quality.
5 Pareto-Based Instruction Optimization for Re-rankers
In this section, we first formalize the problem setting and then study input-level optimization for re-rankers.
Because the instruction is the only input component fully under practitioner control, we propose an instruction
optimization framework to steer re-ranker behavior.
6

5.1 Problem Formulation
An LLM-based re-ranker such as Qwen3-Reranker (Zhang et al., 2025) takes three inputs: an instruction
prompt p, a query q, and a candidate passage c∈ C q. Given model parameters θ, the re-ranker fθproduces a
ranking permutation πp,q=fθ(p, q,C q). We consider two task types:Evolving Knowledge(EK) tasks, where
temporal validity is critical, andNon-Evolving Knowledge(NEK) tasks, where semantic relevance typically
suffices. Let DEKandDNEKdenote their respective distributions, where each data point( q,Cq,Rq)consists of
a query, candidate passages, and ground-truth labels. An instruction optimized for DEKmay overemphasize
temporal signals, potentially degrading performance on DNEKwhere such signals are irrelevant. We therefore
formulate instruction optimization as a bi-objective problem, seeking Pareto-optimal instructions that balance
both task types. For an instructionpand a task distributionD, we define the expected utility as:
JD(p) =E (q,Cq,Rq)∼D[U(f θ(p, q,C q),Rq)],(1)
where Udenotes a ranking quality metric such as MAP. We seek a set of Pareto-optimal instructions P∗that
maximize the following bi-objective function:
F(p) = (J DEK(p),J DNEK(p)).(2)
An instruction p∗∈P∗is Pareto-optimal if there exists no other instruction that improves performance on
one objective without degrading performance on the other. The resulting Pareto front P∗characterizes the
trade-offs between EK and NEK tasks, enabling practitioners to select instructions that best match their
application requirements.
5.2 Optimization via Evolutionary Search
Given that the search space P(i.e., the space of natural language instructions) is discrete and non-differentiable,
we develop an evolutionary algorithm to navigate it. The algorithm iteratively refines a population of
instructions using two text-based genetic operators.
Mutation.Inspired by Pryzant et al. (2023), our mutation operator refines an instruction by diagnosing
ranking errors and correcting them. This process mimics gradient-based optimization but operates entirely
in the textual domain. Specifically, for a given instruction pand query q, an error occurs when a negative
passage c′is ranked above the positive passage c∗. Let πp,qbe the ranking permutation induced by pfor
(q,C q). The error setE(p)is defined as the set of tuples(q, c∗,Eq(p)), where
Eq(p) ={c′∈ Cq\ {c∗}:π p,q(c′)< π p,q(c∗)}.(3)
Given an instruction p, we derive textual gradients by analyzing its ranking errors on a sampled training
batch. Specifically, the gradient estimation operator Gestimate(Section A.1) takes as input the instruction p
and its associated error instances E(p), and produces a set of textual gradientsg, where each gis a natural
language critique describing a concrete failure pattern:
g=Gestimate (p,E(p)).(4)
Each textual gradient is then applied by the gradient application operator Gapply(Section A.2) to generate a
mutated instruction that aims to correct the identified error:
pmut=Gapply(p,E(p), g).(5)
This process yields multiple candidate instructions per parent p, enabling local exploration of the instruction
space guided by observed ranking failures.
Crossover.While mutation refines individual instructions based on their local errors, it does not leverage
complementary strengths across different instructions within the population. To address this, we introduce a
crossover operator that combines instructions that excel on different objectives. Given two parent instructions
pAandpBsampled from the current population, we compare their performance on a sampled training batch
7

across the two objectives (EK and NEK), and select pairs with complementary strengths (e.g., pAstronger
on EK and pBstronger on NEK). We then construct contrastive example sets by identifying instances on
which the stronger instruction succeeds while the weaker instruction fails for each objective. For example,
EA≻Bconsists of EK instances where pAsucceeds but pBfails, and EB≻Ais defined analogously on NEK.
Conditioned on both parent instructions and their respective contrastive example sets, the crossover operator
X(Section A.3) synthesizes a child instruction that integrates the objective-specific strengths of each parent:
pcross=X(p A, pB,EA≻B,EB≻A).(6)
Overall Algorithm.Our instruction optimization algorithm proceeds iteratively, starting from an initial
instructionpopulation P0. Ateachevolutionaryround t, thecurrentpopulation PtfirstundergoesanExpansion
phase, duringwhichnewcandidateinstructionsaregeneratedviamutationandcrossover. Specifically, mutation
applies the composition Gapply◦ Gestimateto refine individual instructions based on their training-batch errors,
while crossover ( X) combines pairs of instructions with complementary strengths across objectives. Together,
these operators produce a set of newly generated instructions, denoted Pnew. In the subsequentEvaluation
phase, each instruction pin the combined pool Pt∪Pnewis evaluated on a held-out validation minibatch
to estimate its objective vector ˆF(p). TheSelectionphase then constructs the next generation Pt+1by
identifying the empirical Pareto front of non-dominated instructions. If the Pareto front exceeds the budgeted
population size, it is pruned using a diversity-preserving criterion such as crowding distance. After a fixed
number of evolutionary rounds, the algorithm terminates and returns the final Pareto front, providing a set of
optimized instructions that expose different trade-offs between EK and NEK tasks. The complete procedure
is summarized in Section F.3.
5.3 Experiments
Inst.τ q τd
Base 5.56 31.70
Opt.5.83 33.40
Table 3 Temporal At-
tention Ratio for Base
vs. Optimized Instruc-
tions.Setting.We instantiate the two task distributions as follows: DEKcorresponds
to our proposed FRESCObenchmark, while DNEKis derived from a reformulated
version of the NQ (Kwiatkowski et al., 2019) dataset in which each query and
passage is assigned a randomly generated timestamp. We maintain a fixed pop-
ulation size of 4 throughout all experiments. Full implementation details are pro-
vided in Section F. We compare our method against three categories of baselines.
(i) Base Model.We evaluate the off-the-shelf Qwen3-Reranker-8B (Zhang et al.,
2025) using its default general-purpose instruction, representing standard re-ranking
performance without temporal adaptation.(ii) Temporal-Aware Methods.We include
recent methods designed for handling temporal signals:TempRALM(Gade et al., 2025), which augments
semantic scores with temporal scores;FreshPrompt(Vu et al., 2024), which leverages few-shot in-context
learning for adaptation; andMRAG(Siyue et al., 2024), which employs a hybrid semantic-temporal module-
based approach.(iii) Fine-Tuning Baselines.As supervised comparisons, we fine-tune the base model on a
mixture of DEKandDNEKtraining data using standard point-wise and list-wise ranking losses. Additional
details are provided in Section F.2.
Figure 2 Evolution of the Pareto
front.Starting from the initial in-
struction (star), optimization pro-
gressively improves the trade-off
betweenD EKandD NEK.Results.As shown in Table 2, our instruction optimization identifies a
Pareto front that offers improved trade-offs over single-instruction baselines.
At one extreme, Pareto Solution 1 prioritizes DEK, achieving the highest
MAP of 79.20 (a 27% relative gain over the base model). This outperforms
strong baselines such as point-wise fine-tuning and MRAG on DEK, while
avoiding the severe degradation on DNEKobserved with TempRALM.
At the other extreme, Pareto Solution 4 achieves the highest MAP on
DNEK(62.27) and remains competitive with the strongest baselines under
MRR and nDCG. Notably, it still achieves a MAP of 68.88 on DEK,
improving over the base model despite prioritizing DNEK. The intermediate
solutions (Pareto Solutions 2 and 3) demonstrate a smooth and controllable
trade-off between the two objectives. Overall, our framework yields a
set of specialized instructions that jointly dominate the trade-off frontier,
enabling practitioners to select re-ranking behavior aligned with deployment
priorities. We report results on additional re-rankers in Section B.4.
8

Table 4 Ablation study.Pareto Solutions 1 and 4 are the two
extremes of the Pareto front, capturing opposite trade-offs
between DEKandDNEK. We report MAP on DEKandDNEK.
Method DEK DNEK
Pareto Solution 1 79.20 59.41
Pareto Solution 4 68.88 62.27
Pareto Solution 1 (w/o Mut.) 76.59 60.53
Pareto Solution 4 (w/o Mut.) 65.26 62.44
Pareto Solution 1 (w/o Cross.) 77.09 59.43
Pareto Solution 4 (w/o Cross.) 62.60 62.23Analysis.To better understand the effects of
our instruction optimization framework, we con-
duct additional analyses. An attention analysis,
detailed in Section B.1 and summarized in Ta-
ble 3, shows that the instruction corresponding
to the DEK-optimal point on the Pareto front
(Pareto Solution 1) increases the re-ranker’s at-
tention to both query ( τq) and document times-
tamps ( τd), with a larger gain on τd. This shift
offers a plausible explanation for its improved
temporal discrimination. Moreover, we visualize
the optimization dynamics in Figure 2, which
shows that the evolutionary process progressively
expands the Pareto front across rounds, yielding instructions with improved trade-offs between DEKand
DNEK; additional details are provided in Section B.2.
Ablation Study.We evaluate the impact of mutation and crossover by removing each operator while keeping
the total instruction budget fixed. Pareto Solutions 1 and 4 correspond to the DEK- andDNEK-optimal points
on the Pareto front, respectively. As shown in Table 4, removing mutation consistently reduces MAP on DEK
across solutions. Removing crossover also degrades performance, with a larger drop in DEKMAP for Pareto
Solution 4, while MAP on DNEKis comparatively less affected. Overall, both operators contribute to the final
Pareto front, and ablating either one yields inferior trade-offs betweenD EKandDNEK.
6 Conclusion
We introduce FRESCOto evaluate re-rankers underEvolving Semantic Conflict, where they must prioritize
up-to-date evidence while preserving semantic relevance. Our analysis reveals a common failure mode: existing
re-rankers often prefer semantically rich but obsolete evidence. To address this, we propose a bi-objective
instruction optimization framework that yields a Pareto front over EK and NEK.
Limitations
This work focuses on a practically important aspect of re-ranking under temporally evolving information, and
several considerations define the scope of the current study. First, FRESCOis constructed using Wikipedia and
Wikidata, which offer well-structured and verifiable records of factual evolution. This choice enables controlled
and scalable benchmark construction, while other information sources, such as news or rapidly changing
web content, may exhibit different temporal characteristics that are not explicitly captured here. Second,
the evaluation setup makes document timestamps explicitly available to the re-ranker. This design choice
allows us to directly assess temporal discrimination in a controlled setting, but does not address scenarios
where temporal signals are implicit, noisy, or unavailable and must instead be inferred from content. Finally,
instruction optimization operates solely at the interface level without modifying model parameters. This
design enables lightweight applications even in black-box settings and complements model-level approaches
that address temporal reasoning. Overall, these considerations reflect the scope of the present study and
highlight directions for extending temporally robust re-ranking evaluation and optimization in future work.
Ethical Considerations
This work includes a limited amount of human annotation conducted solely for evaluation purposes. The
annotation task involves verifying the semantic relevance and temporal validity of publicly available textual
passages and does not require subjective judgment or involve sensitive topics. The study was conducted in
accordance with the authors’ institutional guidelines for research ethics and was determined to be exempt
from formal ethics review. The benchmark is constructed from content publicly available on Wikipedia and
Wikidata. These resources are used in a manner consistent with their role as open encyclopedic knowledge
9

bases. When using existing datasets and models, we adhere to their respective licenses and terms of use. The
artifact introduced in this work, FRESCO, is intended for academic research and benchmarking of re-rankers
under temporally evolving information. Its intended use is compatible with the access conditions of the
original data sources from which it is derived. The dataset is accompanied by usage guidelines that prohibit
harmful or unethical applications, including deception, discrimination, or other harmful behaviors. The
dataset does not intentionally collect or include information about private individuals. The source materials
consist of curated encyclopedic content describing public entities and factual information. During dataset
construction, we checked the collected data and did not identify explicit offensive or harmful content. The
dataset primarily covers English-language encyclopedic content and focuses on entities with temporally
evolving factual attributes.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate,
and critique through self-reflection. 2024.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew
McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang.
Ms marco: A human generated machine reading comprehension dataset, 2018. https://arxiv.org/abs/1611.09268 .
Davide Baldelli, Junfeng Jiang, Akiko Aizawa, and Paolo Torroni. Twolar: a two-step llm-augmented distillation
method for passage reranking. InEuropean Conference on Information Retrieval, pages 470–485. Springer, 2024.
Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein,
Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo
Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue,
Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn,
Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto,
Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan
Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh,
Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec,
Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric
Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos
Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris
Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani,
Camilo Ruiz, Jack Ryan, Christopher Ré, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan
Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan
Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei Zaharia, Michael Zhang,
Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. On the opportunities and
risks of foundation models, 2022.https://arxiv.org/abs/2108.07258.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.Advances in neural
information processing systems, 33:1877–1901, 2020.
Shijie Chen, Bernal Jiménez Gutiérrez, and Yu Su. Attention in large language models yields efficient zero-shot
re-rankers.arXiv preprint arXiv:2410.02642, 2024.
Ziyang Chen, Xiang Zhao, Jinzhi Liao, Xinyi Li, and Evangelos Kanoulas. Temporal knowledge graph question
answering via subgraph reasoning.Knowledge-Based Systems, 251:109134, 2022. ISSN 0950-7051. doi: https://doi.
org/10.1016/j.knosys.2022.109134.https://www.sciencedirect.com/science/article/pii/S0950705122005603.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham,
Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways.
Journal of Machine Learning Research, 24(240):1–113, 2023.
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Ellen M. Voorhees, and Ian Soboroff. Trec deep learning
track: Reusable test collections in the large data regime, 2021.https://arxiv.org/abs/2104.09399.
Anoushka Gade, Jorjeta G Jetcheva, and Hardi Trivedi. It’s about time: Incorporating temporality in retrieval
augmented language models. In2025 IEEE Conference on Artificial Intelligence (CAI), pages 75–82. IEEE, 2025.
10

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and
Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv preprint arXiv:2312.10997,
2(1), 2023.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783, 2024.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand
Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models.
Journal of Machine Learning Research, 24(251):1–43, 2023.
Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, and Minjoon
Seo. Temporalwiki: A lifelong benchmark for training and evaluating ever-evolving language models. InProceedings
of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6237–6250, 2022.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. InEMNLP (1), pages 6769–6781, 2020.
Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A Smith, Yejin Choi,
Kentaro Inui, et al. Realtime qa: What’s the answer right now?Advances in neural information processing systems,
36:49025–49043, 2023.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-
Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for
question answering research.Transactions of the Association for Computational Linguistics, 7:453–466, 2019. doi:
10.1162/tacl\_a\_00276.https://doi.org/10.1162/tacl_a_00276.
Thiago Soares Laitz, Konstantinos Papakostas, Roberto Lotufo, and Rodrigo Nogueira. Inranker: Distilled rankers for
zero-shot information retrieval. InBrazilian Conference on Intelligent Systems, pages 140–154. Springer, 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems, 33:9459–9474, 2020.
Xianming Li, Aamir Shakir, Rui Huang, Julius Lipp, and Jing Li. Prorank: Prompt warmup via reinforcement learning
for small language models reranking.arXiv preprint arXiv:2506.03487, 2025.
Inc. NetEase Youdao. Bcembedding: Bilingual and crosslingual embedding for rag. https://github.com/
netease-youdao/BCEmbedding, 2023.
Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with bert.arXiv preprint arXiv:1901.04085, 2019.
Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. Document ranking with a pretrained sequence-to-
sequence model. In Trevor Cohn, Yulan He, and Yang Liu, editors,Findings of the Association for Computational
Linguistics: EMNLP 2020, pages 708–718, Online, November 2020. Association for Computational Linguistics. doi:
10.18653/v1/2020.findings-emnlp.63.https://aclanthology.org/2020.findings-emnlp.63/.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. Rankvicuna: Zero-shot listwise document reranking with
open-source large language models.arXiv preprint arXiv:2309.15088, 2023a.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. Rankzephyr: Effective and robust zero-shot listwise
reranking is a breeze!arXiv preprint arXiv:2312.02724, 2023b.
Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization
with “gradient descent” and beam search. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, pages 7957–7968, 2023.
Devendra Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer.
Improving passage retrieval with zero-shot question generation. In Yoav Goldberg, Zornitsa Kozareva, and Yue
Zhang, editors,Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages
3781–3797, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi:
10.18653/v1/2022.emnlp-main.249.https://aclanthology.org/2022.emnlp-main.249/.
Aamir Shakir, Darius Koenig, Julius Lipp, and Sean Lee. Boost your search with the crispy mixedbread rerank models,
2024.https://www.mixedbread.ai/blog/mxbai-rerank-v1.
11

Zhang Siyue, Xue Yuxiang, Zhang Yiming, Wu Xiaobao, Luu Anh Tuan, and Zhao Chen. Mrag: A modular retrieval
framework for time-sensitive question answering.arXiv preprint arXiv:2412.15540, 2024.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. Is chatgpt good at search?
investigating large language models as re-ranking agent.ArXiv, abs/2304.09542, 2023.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogeneous
benchmark for zero-shot evaluation of information retrieval models.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou,
Quoc Le, et al. Freshllms: Refreshing large language models with search engine augmentation. InFindings of the
Association for Computational Linguistics ACL 2024, pages 13697–13720, 2024.
Feng Wang, Yuqing Li, and Han Xiao. jina-reranker-v3: Last but not late interaction for document reranking.arXiv
preprint arXiv:2509.25085, 2025.
Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and
Quoc V Le. Finetuned language models are zero-shot learners.arXiv preprint arXiv:2109.01652, 2021.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to advance general
chinese embedding, 2023.
Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie,
Fei Huang, et al. mgte: Generalized long-context text representation and reranking models for multilingual text
retrieval. InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry
Track, pages 1393–1412, 2024.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng
Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176, 2025.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie
Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li,
Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models, 2025.
https://arxiv.org/abs/2303.18223.
Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni, Xuanhui Wang, and Michael
Bendersky. Rankt5: Fine-tuning t5 for text ranking with ranking losses. InProceedings of the 46th International
ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, page 2308–2313, New
York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394086. doi: 10.1145/3539618.3592047.
https://doi.org/10.1145/3539618.3592047.
12

Appendix
A Details of Operators
A.1 Gradient Estimation Operator
# Role and Goal
You are an expert Prompt Engineer specializing in optimizing inputs for ranking and information
retrieval models. Your objective is to analyze the provided examples of failure from my current
reranker model, diagnose the weaknesses in its current prompt, and craft superior, revised prompts
that will improve its performance.
# Model Context
I am using a reranker model that takes three inputs: a **`Prompt`**, a **`Query`**, and a **`
Document`**.
The model's function is to evaluate how relevant the`Document`is to the`Query`based on the
guidance provided in the`Prompt`. It returns a relevance score, where a higher score indicates
higher relevance.
# Current Prompt and Problem Statement
The current prompt I am using is:
`"{current_prompt}"`
For a given`Query`, the model often assigns a higher score to an irrelevant document (**Negative
Document**) than to the ideal, relevant document (**Positive Document**). My goal is to fix this
by improving the prompt.
# Error Examples for Analysis
Below are concrete examples where the current model fails. In each case, a`Negative Document`
incorrectly receives a higher score than the`Positive Document`. Please analyze these patterns to
identify the flaw in the current prompt.
{error_string}
# Core Task & Required Output
Give {num_gradients} reasons why the current prompt could have gotten these examples wrong. Wrap
each reason with <START> and <END>
A.2 Gradient Application Operator
# Role and Goal
You are an expert Prompt Engineer tasked with improving a reranker model's performance. The
reranker model's job is to score how relevant a **`Document`** is to a **`Query`**, **based on the
prompt I provide**. Your objective is to generate new, improved prompts for the reranker model
based on a provided analysis of a failing prompt and its error examples.
# Background Information
13

You will be given three pieces of information: the current prompt that performed poorly, the
specific examples it failed on, and an analysis of why it failed.
### 1. The Current Prompt
This is the prompt that needs improvement.
`"{current_prompt}"`
### 2. Error Examples
The current prompt failed on the following examples:
{error_str}
### 3. Analysis of the Problem
Based on the errors, the key weaknesses of the current prompt were identified as follows:
{gradient_str}
# Core Task & Required Output
Based on all the information provided above (the original prompt, its errors, and the analysis),
please perform the following tasks:
1. **Generate Prompts:** Write **{steps_per_gradient}** different and improved prompts that aim to
overcome the identified weaknesses.
2. **Encourage Diversity:** Each prompt should be distinct from the others.
3. **Formatting:** Wrap each new prompt individually with`<START>`and`<END>`.
A.3 Crossover Operator
# Role and Goal
You are an expert Prompt Engineer specializing in synergistic prompt design. Your objective is to
analyze two distinct prompts, identify the core reasons for their unique successes, and then
synthesize these insights into a superior, hybrid prompt that inherits the strengths of both.
# Contrastive Analysis of Two Prompts
We have analyzed two prompts, A and B, and have found specific examples where one succeeded while
the other failed. This contrastive analysis reveals the unique strengths of each.
### 1. Prompt A's Strengths (Where A Succeeded and B Failed)
**Prompt A:**`"{prompt_a}"`
In the following examples, **Prompt A correctly identified the Positive Document, whereas Prompt B
failed to do so.** These examples highlight the winning strategy of Prompt A.
{examples_a_wins}
### 2. Prompt B's Strengths (Where B Succeeded and A Failed)
**Prompt B:**`"{prompt_b}"`
In the following examples, **Prompt B correctly identified the Positive Document, whereas Prompt A
failed to do so.** These examples highlight the winning strategy of Prompt B.
14

{examples_b_wins}
# Core Task & Required Output
Your task is to create a new, more powerful prompt by combining the winning strategies of both A
and B.
1. **Analyze Prompt A's Winning Strategy:** Based on the first set of examples, what specific
phrasing, instruction, or principle in Prompt A allows it to succeed where B fails?
2. **Analyze Prompt B's Winning Strategy:** Similarly, based on the second set of examples, what
is the core strength of Prompt B that allows it to handle cases that A could not?
3. **Generate Hybrid Prompts:** Synthesize these two winning strategies into **{num_crossovers}**
distinct, new prompts. Each new prompt must be a cohesive instruction set that aims to solve all
provided examples by intelligently combining the best of A and B.
4. **Formatting:** Wrap each new prompt individually with`<START>`and`<END>`.
B Additional Experiments
B.1 How Instructions Steer Temporal Awareness of Re-rankers
Our primary results show that our instruction optimization method identifies a Pareto front capturing
trade-offs between DEKandDNEK. In particular, the DEK-optimal point (Pareto Solution 1) yields substantial
performance gains on DEK. To better understand the mechanism underlying this improvement, we analyze
how the optimized instruction alters model behavior. We hypothesize that it encourages the model to attend
more strongly to explicit timestamp signals in both the query and the document.
Experimental SetupTo test this hypothesis, we measure theTemporal Attention Ratio, defined as the
proportion of attention that the re-ranker’s final token directs towards the timestamp tokens within the
input. The final token’s attention distribution is a strong proxy for the model’s decision process, as it directly
precedes the final relevance judgment (e.g., yesornotokens) (Zhang et al., 2025). Formally, letA(L)be the
attention matrix of the model’s final layer L, and let Tlastbe the index of the final token before producing
the final relevance judgment. For a given passage, let Itsbe the set of token indices corresponding to its
timestamp (e.g., Timestamp: 2025-08-31T00:00:00Z), and let Icontextbe the set of indices for all tokens from
the start of the query to the end of the document. The Temporal Attention Ratio is calculated as:
Temporal Attention Ratio=P
i∈ItsA(L)
Tlast,iP
j∈IcontextA(L)
Tlast,j(7)
We compute this ratio for both the query’s timestamp ( τq) and the document’s timestamp ( τd). We compare
two instructions: a genericBase Instruction(“Given a web search query, retrieve relevant passages that answer
the query”) and our best-performingOptimized Instruction(Pareto Solution 1 from our Pareto front: “Given a
specific query, directly and accurately answer the question by retrieving the most recent, precise, and relevant
document that provides the latest available data as of the query’s timestamp. Ensure the document not only
offers the most current information but also presents the answer in a clear and concise manner, reflecting
the latest developments. Prioritize documents that are both reliable and up-to-date, filtering out outdated
information to provide the most accurate response possible”). The analysis is performed on 1000 randomly
sampled instances from ourFRESCOtest set using the Qwen3-Reranker-8B model.
Results and AnalysisThe results, summarized in Table 3, reveal a clear shift in the re-ranker’s attention
patterns induced by our optimized instruction. We observe an increase in the attention paid to the timestamps
in both the query and the document, with the most pronounced change occurring for the document’s
timestamp.
Specifically, with the Optimized Instruction, the average attention directed at the document’s timestamp
(τd) increases from 31.70% to 33.40%. This notable shift suggests that the re-ranker, guided by the explicit
15

instruction to prioritize recency, learns to more actively verify the temporal validity of the candidate passage
when making its relevance judgment. Furthermore, we observe a smaller but consistent increase in attention
towards the query’s timestamp ( τq), which rises from 5.56% to 5.83%. This suggests the optimized instruction
enhances the re-ranker’s overall sensitivity to the temporal context of the task.
This analysis provides a mechanistic explanation for the performance gains reported in our main results
(Table 2). The instruction optimization is not a black box; it works by tangibly re-directing the re-ranker’s
internal attention mechanisms to focus on the crucial temporal cues required for the task. This confirms that
our method effectively instills a temporal sensitivity into the re-ranker by reshaping its information processing
strategy.
B.2 Dynamics of Instruction Optimization Process
To gain insight into how our evolutionary search navigates the instruction space, we visualize the progression of
the Pareto front over optimization rounds. Figure 2 shows the MAP of non-dominated instructions discovered
over successive optimization rounds on the validation sets of DEKandDNEK. Each point on a curve represents
the averaged coordinates of a specific Pareto solution across 3 independent runs, and the lines connect these
average points to illustrate the shape of the averaged front at that round.
The optimization begins at Round 0 with a single, general-purpose instruction (grey star), establishing
the baseline performance. As the search progresses (represented by progressively darker blue points and
dashed lines), we observe a clear upward and rightward shift of the Pareto front. This indicates that our
algorithm, leveraging mutation via textual gradients and instruction crossover, successfully discovers new
instructions that significantly improve upon the initial instruction, achieving better performance on one or
both objectives simultaneously. By Round 10, the final averaged Pareto front (dark navy line and circles)
represents the culmination of the optimization process. It clearly dominates the fronts from earlier rounds,
signifying performance gains. The final front spans a considerable range, demonstrating that our method
identifies multiple high-performing instructions: some specialized for DEK(top-left region), others for DNEK
(bottom-right region), and several offering balanced performance in between.
These dynamics confirm the effectiveness of our evolutionary approach. The algorithm progressively explores
the instruction landscape, escaping the limitations of the initial instruction and converging towards a set of
diverse, Pareto-optimal instructions that effectively balance the competing demands of temporal awareness
and general relevance.
B.3 Case Study: Evolutionary Instruction Optimization Dynamics
To provide a concrete illustration of our evolutionary instruction optimization framework, Table 5 presents an
example optimization trajectory. The table traces how a re-ranker instruction evolves across multiple rounds
through mutation and crossover guided by feedback signals.
The initial instruction in Round 0 is adopted from the technical report of Qwen3-Reranker (Zhang et al., 2025)
and represents a commonly used, high-level formulation for passage re-ranking. In Round 1, the gradient
exposes limitations of this generic instruction, and mutation is applied to generate a modified variant. Since
the initial population contains only a single instruction (|P 0|= 1), crossover is not applicable at this stage.
As optimization proceeds, further mutation steps produce additional instruction variants. Once multiple
candidates are available, crossover is applied to recombine elements from different parent instructions. As
shown in Round 4, this process expands the set of candidate instructions explored during optimization.
Overall, this case study illustrates how mutation and crossover are used to explore a Pareto set of instructions
across both Evolving Knowledge (EK) and Non-Evolving Knowledge (NEK) settings, without manual
instruction engineering.
16

Table 5 Case Study: Pareto Front Based Instruction Optimization.An example optimization trajectory from Section 5.3,
illustrating how an initial generic instruction is progressively refined through mutation and crossover guided by gradient
signals. Since the initial population contains only a single instruction ( |P0|= 1), the crossover operator is not applicable
in Round 1.
Round Operator Instruction / Gradient
Round 0 Initial Instruction Given a web search query, retrieve relevant passages that answer the query.
Round 1 Mutation ...
Instruction ( p∈P 0):Given a web search query, retrieve relevant passages that answer
the query.
Gradient ( gin Equation (4)):The current prompt “Given a web search query, retrieve
relevant passages that answer the query” may not be specific enough to guide the
model to prioritize the most up-to-date and accurate information, leading to irrelevant
or outdated documents being ranked higher than the positive documents. This lack of
specificity can cause the model to focus on general relevance rather than the exact
information being queried, resulting in incorrect rankings.
Derived Instruction ( pmutin Equation (5)):Given a specific query, retrieve the most relevant
and up-to-date passage that directly answers the question, prioritizing information
that is both accurate and current as of the query timestamp.
...
...(Round optimization steps) ...
Round 4 Mutation ...
Instruction ( p∈P 3):Given a specific query, retrieve the most relevant and up-to-
date passage that directly answers the question, prioritizing information that is both
accurate and current as of the query timestamp.
Gradient ( gin Equation (4)):The current prompt lacks specificity regarding the handling
of out-of-date or less relevant information, which might lead the model to prioritize
documents based on their general relevance to the query rather than their accuracy or
up-to-dateness, as indicated by the provided timestamp or other temporal cues. This
could cause the model to assign higher scores to negative documents that, although
less current or less accurate, contain closely related keywords or topics, thus overriding
the importance of recency and factual precision in the scoring process.
Derived Instruction ( pmutin Equation (5)):Given a specific query, retrieve the most relevant
and up-to-date passage that directly answers the question, prioritizing information
that is both accurate and current as of the query timestamp, while considering the
temporal relevance of the documents and favoring those with more recent timestamps
when the query implies a need for the latest information.
...
Crossover ...
Instruction A ( pAin Equation (6)):Given a specific query, retrieve the most relevant and
up-to-date passage that directly answers the question, prioritizing information that is
both accurate and current as of the query timestamp.
Instruction B ( pBin Equation (6)):Given a web search query that may require the most
current, specific, or nuanced information, retrieve the most relevant and up-to-date
passage that directly answers the query. Ensure the information is not only accurate
and reflects the latest developments or facts related to the query but also consider
the context and intent behind the query to provide the most appropriate response,
prioritizing understanding over mere keyword matching.
Derived Instruction ( pcrossin Equation (6)):Given a web search query that may require the
most current, specific, or nuanced information, retrieve the most relevant and up-to-
date passage that directly answers the question. Prioritize information that is both
accurate and current as of the query timestamp. Additionally, consider the context
and intent behind the query to provide the most appropriate response, ensuring the
answer is not just a keyword match but a thoughtful and informed reply that reflects
the latest developments or facts related to the query.
...
...(Round optimization steps) ...
17

Table 6 Performance comparison on Evolving Knowledge ( DEK) and Non-Evolving Knowledge ( DNEK) task.We use RankGPT
(Sun et al., 2023) based on LLaMA 3.2-3B-Instruct (Grattafiori et al., 2024) as our base model. Our instruction
optimization method (Pareto Solution 1-4) discovers a Pareto front offering superior trade-offs compared to baselines.
The trade-off plot is provided in Section B.5.
MethodDEK↑ DNEK↑
MAP MRR@10 nDCG@10 MAP MRR@10 nDCG@10
Base Model 15.51 13.25 19.26 28.92 38.46 48.78
Temporal-Aware Models
TempRALM (Gade et al., 2025) 19.97 17.78 23.48 26.08 31.19 43.31
FreshPrompt (Vu et al., 2024) 21.42 19.05 24.50 28.5940.0548.31
MRAG (Siyue et al., 2024) 19.40 17.31 25.16 28.60 38.12 48.11
Fine Tuning
Point-wise Finetuning 18.34 16.18 23.73 28.97 38.99 48.91
List-wise Finetuning N/A N/A N/A N/A N/A N/A
Ours
Pareto Solution 1 22.28 20.52 29.67 28.82 37.91 48.23
Pareto Solution 2 21.87 20.05 29.05 29.31 38.59 49.19
Pareto Solution 3 21.80 19.93 28.65 29.68 39.05 49.65
Pareto Solution 4 21.20 19.25 27.83 29.7139.37 50.14
B.4 Pareto-Based Instruction Optimization for RankGPT (LLaMA 3.2-3B-Instruct)
Beyond Qwen3-Reranker-8B, we evaluate our Pareto-based instruction optimization framework on RankGPT
(Sun et al., 2023), built on LLaMA 3.2-3B-Instruct (Grattafiori et al., 2024). As shown in Table 6, the results
exhibit a pattern consistent with Table 2: our framework identifies a Pareto front of non-dominated solutions
that improves the trade-off betweenD EKandDNEKrelative to the baselines.
At one extreme, Pareto Solution 1 prioritizes DEK, achieving the highest MAP of 22.28, corresponding to a
44% relative improvement over the base model. It outperforms strong baselines such as TempRALM and
FreshPrompt on DEK, while avoiding the substantial degradation on DNEKobserved with TempRALM. At
the other extreme, Pareto Solution 4 achieves the best MAP on DNEKat 29.71. Notably, it still attains a
MAP of 21.20 on DEK, remaining above the base model despite prioritizing DNEK. The intermediate solutions
further exhibit a smooth trade-off between the two objectives.
(a)Qwen3-Reranker-8B
 (b)RankGPT (LLaMA)
Figure 3 Comparison of nDCG@10 scores for Qwen3-
Reranker-8B and RankGPT (LLaMA 3.2-3B-Instruct).For the fine-tuning baseline, we note that RankGPT is a
listwise re-ranker, whereas our setting does not provide
full permutation supervision over the candidate passages.
We therefore adopt a point-wise fine-tuning setup, training
the model to identify the most relevant passage among the
candidates and thereby predict the positive passage.
B.5 Trade-off Plots
We plotted the nDCG@10 scores of our method and the
baseline methods on DEKandDNEKin Figure 3. The
results show that, for both Qwen3-Reranker-8B (Table 2)
and RankGPT (LLaMA 3.2-3B-Instruct) (Table 6), our
method identifies a Pareto front of non-dominated solutions
that achieves a superior trade-off between DEKandDNEK
compared to the baselines.
18

C Model Details
C.1 Re-ranker Details
Table 7 Detailed Model Specificationsfor all re-rankers benchmarked in this work. For reference and reproducibility, we
include model name, creator organization, initial release date, hosting platform, availability type, and model size.
Model Created by Release Date Hosted by Availability Type Model Size
MonoT5 (Nogueira et al., 2020) Nogueira et al. (2020) 03/14/2020 Huggingface Open Weight 220M (T5-base)
RankT5 (Zhuang et al., 2023) Zhuang et al. (2023) 10/12/2022 Huggingface Open Weight 220M (T5-base)
UPR (Sachan et al., 2022) Sachan et al. (2022) 04/03/2023 Huggingface Open Weight 220M (T5-base)
RankGPT (Sun et al., 2023) Sun et al. (2023) 04/19/2023 OpenAI Proprietary N/A
RankVicuna (Pradeep et al., 2023a) Pradeep et al. (2023a) 09/26/2023 Huggingface Open Weight 7B
RankZephyr (Pradeep et al., 2023b) Pradeep et al. (2023b) 12/05/2023 Huggingface Open Weight 7B
bce-reranker-base-v1 (NetEase Youdao, 2023) NetEase Youdao 01/03/2024 Huggingface Open Weight 279M
InRanker (Laitz et al., 2024) Laitz et al. (2024) 01/12/2024 Huggingface Open Weight 220M
mxbai-rerank-base-v1 (Shakir et al., 2024) Mixedbread 02/29/2024 Huggingface Open Weight 184M
Twolar (Baldelli et al., 2024) Baldelli et al. (2024) 03/26/2024 Huggingface Open Weight 0.7B (twolar-large)
jina-reranker-v1-tiny-enaJina AI 04/15/2024 Huggingface Open Weight 33M
jina-reranker-v1-turbo-enbJina AI 04/15/2024 Huggingface Open Weight 37.8M
jina-reranker-v2-base-multilingualcJina AI 06/19/2024 Huggingface Open Weight 0.3B
gte-multilingual-reranker-base (Zhang et al., 2024) Alibaba 07/20/2024 Huggingface Open Weight 306M
LdlR-Qwen2-reranker-1.5B neofungd08/12/2024 Huggingface Open Weight 1.5B
IncontextReranker (Chen et al., 2024) Chen et al. (2024) 10/03/2024 Huggingface Open Weight 8B (Llama-3.1-8B)
Qwen3-Reranker-0.6B (Zhang et al., 2025) Alibaba 05/29/2025 Huggingface Open Weight 0.6B
Qwen3-Reranker-8B (Zhang et al., 2025) Alibaba 05/29/2025 Huggingface Open Weight 8B
ahttps://jina.ai/models/jina-reranker-v1-tiny-en/
bhttps://jina.ai/models/jina-reranker-v1-turbo-en/
chttps://jina.ai/models/jina-reranker-v2-base-multilingual/
dthe user account on Hugging Face
Table 7 summarizes the detailed specifications of all re-rankers evaluated in our benchmark. For each model,
we report the creator organization, initial release date, hosting platform, availability type (e.g., open-weight
or proprietary), and model size.
C.2 Listwise Re-ranking Prompts
To ensure reproducibility, we provide the exact prompt templates used for listwise re-rankers.
Prompt Template for RankGPT
system: You are RankGPT, an intelligent assistant that can rank passages based on their relevancy
to the query.
user: I will provide you with 20 passages, each indicated by number identifier [].
Rank the passages based on their relevance to query: {query}.
assistant: Okay, please provide the passages.
user: [1] {candidate document 1}
assistant: f"Received passage [1].
user: [2] {candidate document 2}
assistant: f"Received passage [2].
...
assistant: f"Received passage [20].
user: Search Query: {query}.
Rank the 20 passages above based on their relevance to search query. The passages should be listed
in descending order using identifiers. The most relevant passages should be listed first. The
output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say
any word or explain.
Prompt Template for RankVicuna and RankZephyr
19

A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER:
I will provide you with 20 passages, each indicated by a numerical identifier [].
Rank the passages based on their relevance to the search query: {query}
[1] {candidate document 1}
[2] {candidate document 2}
...
[20] {candidate document 20}
Search Query: {query}
Rank the 20 passages above based on their relevance to the search query.
All the passages should be included and listed using identifiers,
in descending order of relevance.
The output format should be [] > [], e.g., [4] > [2].
Only respond with the ranking results, do not say any word or explain.
ASSISTANT:
D Dataset Details
The final FRESCObenchmark comprises 3,658 queries for each query type (explicitly and implicitly recency-
seeking), totaling 7,316 unique queries. Each query is associated with a candidate set containing 51 passages.
For our benchmarking experiments evaluating existing re-rankers, we utilize the entire dataset. For the
experiments detailed in Section 5, this dataset was partitioned into three subsets: training, validation, and
testing. Specifically, we employed 200 instances for training, 800 instances for validation, and reserved the
remaining portion for the test set. All benchmarking experiments were run on four NVIDIA A100 80GB
GPUs, with each re-ranker evaluated in less than two hours.
E Human Annotation Details
To rigorously assess the quality of the automatically constructed FRESCObenchmark, we performed a human
evaluation study.
SetupWe randomly sampled 200 queries from the benchmark. For each query, we presented three trained
human annotators with a curated set of four passages. This set consisted of:
•The single passage designated aspositiveby our dataset construction pipeline.
•The threehighest-scoring hard negative passagesas ranked by the Qwen3-Embedding-0.6B model during
the final assembly phase (see Section 3).
Presenting only the most challenging negatives focused the annotators’ efforts on the most ambiguous cases,
ensuring an efficient and targeted validation process.
Annotation TaskThe annotators were instructed to carefully read the query (including its associated
timestamp) and all four candidate passages. Their task was to select thesingle passagethat wasmost
temporally alignedwith the query timestamp and providedsufficient and correct evidenceto answer the query.
An example of the annotation interface and task is shown in Figure 4. To handle potential imperfections
in our automated pipeline or inherent ambiguities in the source material, annotators were given the option
to select None of the above if they determined that none of the provided passages met both criteria. The
20

Instruction:
From the candidate passages below, please select the single best passage that serves as the most accurate and 
up-to-date evidence, considering the query's timestamp.
Decision
Select the single best passage:
☐ Passage 1
☐ Passage 2
☐ Passage 3
☐ Passage 4
☐ None of the above
Comment (optional):
Query : Provide the most recent sports club listed for Jeredy Hilterman. Timestamp: 2025 -08-31T23:59:59Z
Passage [1]
Title: Jeredy  Hilterman
Jeredy  Hilterman  (born 20 June 1998) is a professional footballer who plays as a forward for German club Arminia 
Bielefeld. Born in the Netherlands, he plays for the Suriname national team.
Timestamp: 2025 -07-03T18:10:09Z
Passage [2]
Title: Jeredy  Hilterman
##Club career
He made his Eerste  Divisie  debut for Jong FC Utrecht on 21 August 2017 in a game against FC Oss.
On 8 July 2021, he signed a two -year contract with Emmen.
Timestamp: 2022 -02-05T08:23:33Z
Passage [3]
Title: Jeredy  Hilterman
##Club career
He made his Eerste  Divisie  debut for Jong FC Utrecht on 21 August 2017 in a game against FC Oss.
On 8 July 2021, he signed a two -year contract with Emmen.
On 31 January 2022, Hilterman  joined NAC Breda until the summer of 2024.
Timestamp: 2022 -03-16T11:14:06Z
Passage [4]
Title: Jeredy  Hilterman
Jeredy  Hilterman  (born 20 June 1998) is a professional footballer who plays as a forward for club Willem II on loan 
from Almere City. Born in the Netherlands, he plays for the Suriname national team.
Timestamp: 2023 -08-25T14:55:58Z
Figure 4 Example of the human annotation interface.Annotators were presented with a query (including timestamp) and
four passages (one positive, three hard negatives) labeled by our pipeline. They were asked to select the single passage
that provided sufficient evidence for the query and was temporally valid given the query timestamp, or select None of
the aboveif no passage met both criteria.
annotators were informed that their judgments would be used solely for research and evaluation purposes,
and they participated on a voluntary basis.
Evaluation MetricsWe employed two standard metrics to quantify the quality of our dataset based on the
human annotations:
21

1.Inter-Annotator Agreement (IAA):We measured the consistency among the three annotators using Fleiss’
Kappa ( κ). This metric assesses the reliability of agreement beyond chance. A high κvalue indicates
that the annotation task was clear and the judgments were consistent.
2.Agreement with Pipeline Labels:This metric directly validates our automatic labeling process. We
compared the majority vote label from the three annotators (i.e., the label agreed upon by at least two
annotators) against the original positive label assigned by our pipeline. A high agreement rate signifies
that our pipeline accurately identifies the correct passage.
Results and Disagreement AnalysisOur analysis yielded aFleiss’ Kappa score of κ= 0.9689. According to
standard interpretations, this value representsalmost perfect agreement, confirming the clarity of the task and
the reliability of our annotators’ judgments. The human majority vote aligned with our pipeline’s designated
positive label on197 out of the 200sampled queries. This corresponds to a highagreement rate of 98.5%.
We manually reviewed the 3 instances where the human majority vote disagreed with the pipeline label. In
all such cases, the discrepancy arose because none of the presented passages, including the one designated
as positive by our pipeline, offered sufficiently clear or direct evidence to definitively answer the query. For
instance, a query asking for the most recent team an entity wascoachingmight be paired with passages only
mentioning the team the entity lastplayedfor. Likewise, a query about a team’s latest head coach might
yield passages discussing the team’s roster or recent performance without explicitly naming the coach. These
few disagreements primarily highlight the inherent challenge of finding perfectly explicit textual evidence for
every fact.
ConclusionThe high inter-annotator agreement and the strong agreement rate between human judgments
and our pipeline labels provide robust validation for the quality of FRESCO. The analysis confirms its suitability
as a benchmark for evaluating re-rankers under temporally evolving information.
F Implementation Details
F.1 Evaluation Metrics
We evaluate the performance of re-ranking models using standard Information Retrieval (IR) metrics to assess
the quality of the generated ranking π=fθ(q,Cq;p)relative to ground-truth relevance labels Rq. Here, p
denotes an optional instruction that can be provided to the model. Let π(i)denote the passage ranked at
position i(i= 1being the top rank) within the set of m=|Cq|candidates. We define C+
q⊆ Cqas the subset
of positive passages for queryqbased onR q.
Mean Average Precision (MAP).Average Precision (AP) measures the precision of a model integrated over
the recall curve, effectively rewarding models that rank positive documents at higher positions. Precision at
rankk(P@k) is defined as:
P@k=|{i|1≤i≤k, π(i)∈ C+
q}|
k.
The AP for a single queryqis calculated as:
AP(q) =1
|C+q|mX
k=1 
P@k·1(π(k)∈ C+
q)
,
where 1(·)is the indicator function. If |C+
q|= 0, we set AP(q) = 0. MAP is the arithmetic mean of AP scores
across the set of queriesQ:
MAP=1
|Q|X
q∈QAP(q).
22

Mean Reciprocal Rank (MRR).Reciprocal Rank (RR) focuses on the rank of the first positive document.
Letr 1be the rank of the first positive passage inπ:
r1= min{i|1≤i≤m, π(i)∈ C+
q}.
If no positive passage exists in π, we define RR(q) = 0; otherwise, RR(q) = 1 /r1. MRR is the mean of RR
scores across all queriesQ. When reported at a specific cutoffk(MRR@k),RR(q)is set to 0 ifr 1> k.
Normalized Discounted Cumulative Gain (nDCG).nDCG evaluates the gain of a document based on its
position, applying a logarithmic discount at lower ranks. Under our binary relevance setting (gain G(i) = 1if
π(i)∈ C+
qand0otherwise), the DCG at rankkis:
DCG@k=kX
i=1G(i)
log2(i+ 1).
To allow for cross-query comparison, DCG is normalized by the Ideal DCG (IDCG@ k), which represents the
maximum possible DCG achievable by ranking all positive documents at the top:
nDCG@k=DCG@k
IDCG@k.
IfIDCG@k= 0, we definenDCG@k= 0.
Hit Rate.Hit Rate at rank kis a binary metric that indicates whether at least one positive passage is
present within the top kpositions of the ranked list. It is particularly relevant in scenarios where the user’s
information need is satisfied by finding any single positive item. Using the previously defined rank of the first
relevant passager 1, the Hit Rate for a queryqis defined as:
HitRate@k(q) =1(r 1≤k),
where 1(·)is the indicator function. If no positive passage exists in π,HitRate @k(q) = 0. The Mean Hit
Rate@kis the average of these scores across all queries inQ.
F.2 Baselines
We provide further details on the baseline methods compared in our experiments (Section 5).
F.2.1 Base Model
We use the re-ranker’s default instruction as provided in the original work (Zhang et al., 2025), which is
typically a general instruction like “Given a web search query, retrieve relevant passages that answer the
query”. This baseline represents the standard, non-adapted performance of a powerful contemporary re-ranker
on our tasks.
F.2.2 Temporal-Aware Models
Wecompareagainstthreerecentmethodsspecificallydesignedtoimprovethetemporalawarenessofinformation
retrieval or RAG systems.
TempRALMTempRALM (Gade et al., 2025) augments the standard semantic relevance score s(q, d)between
a query qand a document dwith a temporal relevance score r(qt, dt). Here, qtis the query timestamp and
dtis the document timestamp. The temporal score r(qt, dt)is inversely proportional to the time difference
|qt−dt|, encouraging the re-ranking of documents temporally closer to the query timestamp. The final
retrieval score is calculated as: s(q, d) +r(qt, dt). We adapt this concept by applying the scoring mechanism
to the re-ranking candidates provided to the base re-ranker model to assess its impact in a re-ranking context.
23

FreshPromptFreshPrompt (Vu et al., 2024) prepends the main prompt with few-shot demonstrations. Each
demonstration typically includes an example query, a set of evidence snippets. Although the original format
focuses on generation, we adapt the principle for re-ranking: we provide the re-ranker with demonstrations
showcasing how to prioritize temporally relevant passages from a candidate list containing both positive and
negative examples relative to a query timestamp.
MRAGMRAG (Siyue et al., 2024) employs a modular framework involving three key stages:
1.Question and Passage Processing:The input query is segmented into its main content (MC) and temporal
constraints (TC). In addition, each passage is summarized with a LLM (LLaMA-3.3-70B-Inst here).
2.Semantic-Temporal Hybrid Ranking:A final ranking module multiplicatively combines semantic scores
with symbolic temporal scores derived using temporal score functions similar to the temporal activation
functions in Chen et al. (2022).
We apply the principles of MRAG’s hybrid ranking logic within our re-ranking evaluation framework.
F.2.3 Fine-Tuning
To establish strong supervised baselines, we fine-tune the Qwen3-Reranker-8B base model on our task data.
The fine-tuning process utilizes a combined training set containing examples from both DEKandDNEK. We
employ two standard learning-to-rank approaches:
•Point-wise Fine-tuning:Trains the model to predict the binary relevance label (relevant/irrelevant) for
each query-passage pair independently, typically using a cross-entropy loss.
•List-wise Fine-tuning:Trains the model to optimize the order of the entire list of passages for a given
query, often using losses that directly approximate ranking metrics like ListNet or LambdaRank.
We follow standard procedures outlined in the official repository1for fine-tuning the re-ranker. These baselines
represent a resource-intensive but potentially powerful approach for adapting the model to our scenario,
serving as a contrast to our parameter-free instruction optimization technique.
F.3 Algorithm
We provide a detailed description of our instruction optimization procedure in Algorithm 1. The algorithm
outlines the iterative process of expanding the instruction population through mutation and crossover,
evaluating candidate instructions against both EK and NEK objectives, and selecting the non-dominated
solutions to form the Pareto front for the subsequent round. This evolutionary search effectively navigates the
discrete instruction space to discover instructions that offer superior trade-offs.
In Section 5.3, our evolutionary search algorithm is configured to run for 10 rounds. In each round, the total
number of expanded instructions is set to8 · |Pt|, where |Pt|is the size of the current instruction population.
This expansion budget is allocated equally between the Mutation and Crossover operators. Following the
evaluation and selection phase, the Pareto front is pruned to a maximum size of 4 to form the population
for the next round. The model used to implement the textual gradient operators ( Gestimate,Gapply) and the
crossover operator ( X) is LLaMA-3.3-70B-Instruct (Grattafiori et al., 2024). All results are reported as the
average of three independent experimental runs.
1https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
24

Algorithm 1:Pareto-Based Instruction Optimization via Evolutionary Search
Input:Initial populationP(0), DatasetsD EK,DNEK, Max Pareto sizeB, RoundsT, Expansion factorE,
Mutation(G est,Gapply), CrossoverX, UtilityU
Output:Optimized Pareto-front setP(T)
fort= 0toT−1do
Pmut← ∅;P cross← ∅;
// Expansion Phase: Generate candidates via Mutation and Crossover
Sample training minibatchB tr=Btr
EK∪ Btr
NEK;
// 1. Mutation
foreachp∈P(t)do
E(p)← {(q, c∗,Eq(p))|(q,C q,Rq)∈ Btr,Eq(p)̸=∅};
whereE q(p) ={c′∈ Cq\ {c∗}:π p,q(c′)< π p,q(c∗)};
g← Gestimate (p,E(p));// Estimate textual gradients
foreachg∈gdo
pmut← Gapply(p,E(p), g);// Apply textual gradients
Pmut←Pmut∪ {pmut};
// 2. Crossover
Select parent pairs{(p A, pB)}fromP(t)to satisfy expansion factorE;
foreachpair(p A, pB)do
EA≻B← {examples inB trs.t.π pA,q(c∗)< π pB,q(c∗)};
EB≻A← {examples inB trs.t.π pB,q(c∗)< π pA,q(c∗)};
Pnew← X(p A, pB,EA≻B,EB≻A);// Synthesize hybrid prompts
Pcross←Pcross∪Pnew;
Pcand←P(t)∪Pmut∪Pcross;
// Evaluation Phase: Multi-objective Scoring
Sample validation minibatchB val=Bval
EK∪ Bval
NEK;
foreachp∈P cand do
ˆJEK(p)←1
|Bval
EK|PU(fθ(q,C q;p),R q);
ˆJNEK(p)←1
|Bval
NEK|PU(fθ(q,C q;p),R q);
ˆF(p)←( ˆJEK(p), ˆJNEK(p));
// Selection Phase: Pareto Pruning
Definep≺p′⇐⇒( ˆF(p′)≥ˆF(p))∧( ˆF(p′)̸=ˆF(p));
Pfront← {p∈P cand|∄p′∈Pcands.t.p≺p′};
if|P front|> BthenP(t+1)←SelectTopByCrowding(P front, B);
elseP(t+1)←Pfront;
returnP(T)
25
# BM25 Wins at Scale: A Scaling Study of Retrieval-Augmented Generation Paradigms

**Authors**: Pengyu Wang, Benfeng Xu, Shaohan Wang, Xin Zeng, Huarui Wu, Lei Zhang, Licheng Zhang

**Published**: 2026-07-29 05:46:11

**PDF URL**: [https://arxiv.org/pdf/2607.26497v2](https://arxiv.org/pdf/2607.26497v2)

## Abstract
Retrieval-augmented generation (RAG) spans lexical and dense retrieval, graph-based indexing, and agentic search, but these paradigms are usually evaluated on different benchmarks at one corpus size, leaving their accuracy-cost scaling unclear. To bridge this gap, we present a controlled study that varies corpus size along 28 strictly nested tiers spanning roughly 450-fold, while holding questions and a fixed bedrock of relevant and adversarial documents unchanged. Under one reader model and one judging protocol, we measure official accuracy, construction and query tokens, and latency. The results reveal a scale-dependent crossover rather than an unconditional winner. File-System Agent leads at the smallest shared tiers, but its sequential exploration costs 39 times more query tokens at the bedrock and becomes less effective as the search space grows. Around 10 million corpus tokens, BM25 overtakes it and leads at every larger shared tier, with a margin approaching 20 points at full scale. BM25 also anchors the low-cost end of the Pareto frontier without LLM-based construction. Dense retrieval remains efficient but less accurate, whereas graph-based RAG encounters construction walls before deployment scale and its scalable variants remain below BM25 at shared tiers. Overall, corpus growth increasingly favors global candidate ranking: lexical retrieval is the strongest scalable default, while agentic reasoning works best after ranked discovery rather than in place of it.

## Full Text


<!-- PDF content starts -->

BM25 Wins at Scale: A Scaling Study of
Retrieval-Augmented Generation Paradigms
Pengyu Wang1,∗,Benfeng Xu1,2,†,Shaohan Wang1,Xin Zeng3,Huarui Wu3,Lei Zhang1,Licheng
Zhang1
1University of Science and Technology of China, Hefei, China,2Metastone Technology, Beijing,
China,3Information Technology Research Center, Beijing Academy of Agriculture and Forestry
Sciences, Beijing, China
†Project Lead
Retrieval-augmented generation (RAG) spans lexical and dense retrieval, graph-based indexing, and
agentic search, but these paradigms are usually evaluated on different benchmarks at one corpus size,
leaving their accuracy–cost scaling unclear. To bridge this gap, we present a controlled study that
varies corpus size along 28 strictly nested tiers spanning roughly 450-fold, while holding questions
and a fixed bedrock of relevant and adversarial documents unchanged. Under one reader model and
one judging protocol, we measure official accuracy, construction and query tokens, and latency. The
results reveal a scale-dependent crossover rather than an unconditional winner. File-System Agent
leads at the smallest shared tiers, but its sequential exploration costs 39 times more query tokens at
the bedrock and becomes less effective as the search space grows. Around 10 million corpus tokens,
BM25 overtakes it and leads at every larger shared tier, with a margin approaching 20 points at full
scale. BM25 also anchors the low-cost end of the Pareto frontier without LLM-based construction.
Dense retrieval remains efficient but less accurate, whereas graph-based RAG encounters construction
walls before deployment scale and its scalable variants remain below BM25 at shared tiers. Overall,
corpus growth increasingly favors global candidate ranking: lexical retrieval is the strongest scalable
default, while agentic reasoning works best after ranked discovery rather than in place of it.
Date:July 30, 2026
Correspondence:Licheng Zhang atzlczlc@mail.ustc.edu.cn
1 Introduction
Retrieval-augmented generation grounds the outputs of large language models in external corpora, miti-
gating hallucination (Lewis et al., 2021). Its methods have diverged into paradigms whose costs arise at
different stages and in different forms. Lexical retrieval (Robertson and Zaragoza, 2009) and dense re-
trieval (Karpukhin et al., 2020) require little preparation: indexing takes at most an embedding pass over
the corpus. Graph-based RAG, including MS-GraphRAG (Edge et al., 2025), LightRAG (Guo et al., 2025),
and HippoRAG 2 (Gutiérrez et al., 2025a,b), invests heavily at indexing time: it runs an LLM over every
chunk to extract entities and relations, so that the resulting structure can be exploited at query time; Lin-
earRAG (Zhuang et al., 2025) builds the same kind of graph with a lightweight named-entity recognizer and
embeddings instead.
∗Work done during the internship at Metastone.
1
arXiv:2607.26497v2  [cs.CL]  30 Jul 2026

File-System Agent spends its cost at query time,
using an LLM to search the corpus through it-
erative file-system tool calls (Yao et al., 2023).
Each call is conditioned on earlier results, making
the tool loop a sequential retrieval policy. The
same file-and-command interface is established in
coding-agent systems (Jimenez et al., 2024; Yang
et al., 2024a; Wang et al., 2025) and is used by
leadingindustrialagentssuchasClaudeCodeand
Codex. We instantiate this interface over the cor-
pus’ raw per-source file tree, requiring no retrieval
index. Agents operating over graph indexes are
covered by our access-layer experiments.
However, far less is known about how these
paradigms scale. Each is typically evaluated on
itsownbenchmarkatasinglecorpussize, whereas
deployed corpora such as enterprise knowledge
bases hold hundreds of thousands of documents
and keep growing. By scaling we mean how a
paradigm behaves as the corpus grows while the
workload stays fixed.
Figure 1Schematic of the observed scaling crossover. The
point estimates cross at roughly 10M corpus tokens, after
which BM25 leads at every larger shared tier. Figure 3 re-
ports the measured curves and confidence bands.
Question-relevant and adversarial evidence is also fixed from the smallest tier onward. We measure it on
threeaxes: answeraccuracy, offlinecostinconstructiontokens(generativeandembedding), andonlinecostin
query tokens and latency. Measuring it accurately is hard, because accuracy differences are easily confounded
by the reader model, the judge, the question set, and corpus difficulty. Existing comparisons typically fix the
corpus at a single size and vary these factors freely, so the scaling question has remained open.
To bridge this gap, we conduct a controlled corpus-scaling study on EnterpriseRAG-Bench (Sun et al., 2026),
an enterprise corpus of 511,959 documents with 500 questions and per-question adversarial distractors. Scale
is varied along a ladder of 28 strictly nested tiers, growing 1.25 times per rung from 1,144 to 511,959 docu-
ments (1.7M to 601M tokens), while question-relevant documents and distractors remain fixed. Additional
documents follow one fixed, source-and-noise-stratified order. Every paradigm answers the same questions
using the same underlying reader model, with token-level cost metering, and all predictions are scored under
the benchmark’s official protocol, with robustness checked using an independent judge and a binary protocol.
The central finding, summarized in Figure 1, is a scale-dependent crossover rather than an unconditional win
at every corpus size. The File-System Agent has the higher official point estimates at the smallest measured
shared tiers. Around 10M tokens, BM25 catches it; BM25 then leads at every larger shared tier, while its
query cost remains nearly scale-invariant and the File-System Agent pays increasingly for sequential explo-
ration. At the 601M-token full corpus, the gap reaches nearly 20 points. Dense retrieval remains below both,
while graph-based RAG either reaches an early construction ceiling or remains below BM25 at the shared
tiers it completes.
Our contributions are as follows.
•A reusable 28-tier, 450-fold corpus ladder from 1,144 to 511,959 documents that holds questions, evi-
dence, and distractors fixed.
•A unified evaluation of seven pipelines across four RAG paradigms, with token-level cost metering and
matched controls.
•A scale-dependent crossover: File-System Agent leads early, while BM25 overtakes it around 10M
corpus tokens.
2

2 Related Work
2.1 Retrieval-Augmented Generation
RAG grounds generation in retrieved text through a learned or frozen reader (Lewis et al., 2021; Guu et al.,
2020; Ram et al., 2023); adaptive variants further decide when to retrieve or critique evidence (Asai et al.,
2023; Jiang et al., 2023). Retrieval ranges from lexical BM25 (Robertson and Zaragoza, 2009; Lin et al.,
2021) to learned sparse, dense, and late-interaction models (Formal et al., 2021; Karpukhin et al., 2020;
Khattab and Zaharia, 2020). We use BM25 and compact chunk embeddings as representative lexical and
dense interfaces, while holding the reader fixed.
2.2 Graph-Based RAG
These methods construct explicit structure before answering. MS-GraphRAG builds hierarchical entity com-
munities and reports (Edge et al., 2025); LightRAG indexes entities and relations (Guo et al., 2025); and
HippoRAG 2 retrieves through query-linked facts and Personalized PageRank (Gutiérrez et al., 2025a,b).
LinearRAG instead builds an entity co-occurrence graph with lightweight NER and embeddings, without
generative build calls (Zhuang et al., 2025). This family also includes tree-, curated-KG-, and GNN-based in-
dexes (Sarthi et al., 2024; Chen et al., 2024; Mavromatis and Karypis, 2024; Peng et al., 2024), but published
evaluations generally remain at a fixed, relatively small corpus size.
2.3 Agentic Retrieval
A third direction replaces the one-shot pipeline with an LLM agent that interleaves reasoning and tool
calls (Yao et al., 2023; Schick et al., 2023; Qin et al., 2023), searches the open web (Nakano et al., 2022),
retrieves adaptively during multi-step reasoning (Trivedi et al., 2023; Press et al., 2023), reflects on fail-
ures (Shinn et al., 2023), or learns a search policy with reinforcement learning (Jin et al., 2025). These
systems vary both the controller and the retrieval substrate, whereas our scaling question requires the con-
troller to remain fixed. We therefore instantiate the paradigm in the form established by coding agents: an
identical tool loop navigating a corpus through plain file-system operations (Jimenez et al., 2024; Yang et al.,
2024a; Wang et al., 2025). Our File-System Agent applies this interface to the raw enterprise corpus; the
substrate experiment then holds its policy model, prompt, tool loop, and budget fixed while replacing raw
files with graph indexes.
2.4 Benchmarks and Judging
Existing RAG benchmarks evaluate answer quality at a fixed corpus size and rarely meter offline cost (Yang
et al., 2024b). We extend EnterpriseRAG-Bench’s multi-source corpus and official evaluation (Sun et al.,
2026) with nested scaling, unified cost accounting, and cross-paradigm comparison. We bound judge depen-
dence (Zheng et al., 2023) through dual protocols and measured cross-judge agreement.
3 Method
This section describes the corpus, the nested scaling ladder, and the unified answering and metering harness.
Figure 2 summarizes the full pipeline.
3.1 Corpus and Questions
EnterpriseRAG-Bench models a fictional company that serves LLM inference. The corpus contains 511,957
documents totaling 600.8M tokens, drawn from nine sources that include wiki pages, chat threads, tickets,
e-mail, meeting transcripts, CRM records, and code reviews. Together with the benchmark’s two organiza-
tional overview pages, which we include as scaffolds, the full evaluation tier holds 511,959 documents. The
benchmark provides 500 questions in ten types, ranging from basic lookups to completeness, conflicting-
information, and high-level questions. Each question is annotated with gold documents, a gold answer, and a
list of atomic answer facts, with 722 gold documents in total. The corpus natively marks 7.7% of documents
3

Figure 2Evaluation methodology. Four paradigms access the same nested corpus ladder (T 0⊂ ··· ⊂T 27), which grows
by 1.25 per rung while retaining a fixed 1,144-document bedrock. The 500 questions split into 470 source-grounded,
10 scaffold-supported high-level, and 20 not-found questions. Each paradigm uses the same reader and is metered on
build/query tokens, latency, and official accuracy.
as noise, misfiled under a wrong source or path or near-duplicated with outdated facts, which serves as realis-
tic filing noise. Depending on type, questions are generated from source documents or by an agent exploring
the corpus. The released gold set was subsequently refined by pooling candidate evidence from BM25, dense
retrieval, and file-agent search; label construction is therefore not tied to BM25 alone.
3.2 The Bedrock
The smallest tier is constructed deterministically as the union of the 722 documents the benchmark annotates
as gold for any of the 500 questions, the 326 mined traps, the 99 lures, and the corpus’ own two organizational
pages, a company overview and an initiative index, which serve as scaffolds. Removing five cross-category
duplicates yields 1,144. Traps are mined through method-blind filtering: for each target question, BM25
contributes its full-corpus top-10, while dense retrieval reranks a BM25 top-200 pool (top-1,000 for not-found
questions) and contributes ten candidates. An LLM filter retains every candidate that concerns the same
entity or topic as a gold document while reporting the wrong version, date, or decision. This procedure yields
326 traps under a single topicality-and-wrong-fact criterion. Lures serve the info-not-found questions: the five
most similar candidates per question that the filter verifies cannot answer it, so that unanswerable questions
cannot be solved by the mere absence of retrieved text. BM25 top-10 and dense reranking of a BM25 top-200
pool contribute ten candidates each; no evaluated-system score or answer judgment enters selection. A direct
full-corpus DenseRAG audit independently tests sensitivity to this proposal pool. Because every adversarial
document resides in the bedrock rather than being introduced gradually, question-specific evidence and the
adversarial set remain fixed as the corpus grows through added non-bedrock documents.
3.3 Nested Tiers
Letπdenote a single seeded, source-and-noise-stratified order of the non-bedrock corpus. Tiertconsists
of the bedrock plus the firstn t−1,144documents ofπ. The source and noise distribution of each added
prefix approximates the global background corpus. Sizes follown t+1≈1.25n t, which yields 28 tiers. Prefix
construction guaranteesT 1⊂T 2⊂ ···exactly, verified through manifest checksums, and permits incremental
builders to extend an index from one tier to the next, which is also how marginal build cost is metered. We
report scale in corpus tokens rather than document counts because document lengths differ across sources.
4

4 Experiment
4.1 Experimental Setup
4.1.1 Paradigms
Our main scaling ladder evaluates seven native pipelines. BM25 uses an inverted index built without LLM
involvement. DenseRAG performs dense retrieval over chunk embeddings. HippoRAG 2 builds an open-
vocabulary triple graph and answers queries by linking them to facts and running Personalized PageRank.
MS-GraphRAG builds an entity graph with hierarchical community reports and answers through its local
search mode. LightRAG maintains dual-level entity and relation indexes and answers through its hybrid
mode. LinearRAG constructs an entity co-occurrence graph using a lightweight named-entity recognizer
and the shared embedding model, without generative-LLM calls. The File-System Agent operates without
any index: an agent explores the raw per-source file tree using read-only listing, search, and reading tools
under a budget of 80 LLM calls per question. Every method that exposes a retrieval depth uses the top-5
chunks. BM25, DenseRAG, and HippoRAG 2 operate on the same chunk segmentation, so their retrieval
quality is directly comparable. MS-GraphRAG and LightRAG chunk internally, and their retrieved evidence
is matched back to the shared segmentation for recall computation.
4.1.2 Reader and Metering
All paradigms use the same reader, Qwen3.6-27B at temperature zero, served by vLLM (Kwon et al., 2023).
The File-System Agent uses the same model as its policy, and all main-pipeline embedding calls go through
a shared Qwen3-Embedding-0.6B model. A shared metering layer intercepts every LLM call and attributes
prompt and completion tokens to the build phase or the query phase, per paradigm and per question. Em-
bedding calls are accounted separately from generative calls: because embedding requests carry no prompt
template, their token cost is counted from tokenizer inputs. DenseRAG counts are exact; LinearRAG includes
a documented 3% estimate for its unavailable entity-name stream. Latency is measured end to end under
single-stream conditions on an idle server.
4.1.3 Judging
Every prediction of every family is scored with the benchmark’s official protocol: holistic alignment of the
candidate against the gold answer, the fraction of atomic answer facts entailed by the candidate, a combined
score that gates completeness by correctness, and document-level recall of the retrieved evidence. All main-
ladder cells were produced in one scoring session with one judge model. Each matched control is likewise
scored in a single shared session with identical prompts and judge model; control estimates are not mixed
into the main ladder. Each system is run once per question; uncertainty resamples questions rather than
stochastic reruns. To establish that the results do not depend on this choice, we re-scored the predictions
with an independent judge and with a simpler binary protocol. The binary protocol preserves rankings at all
nine shared scales, while the independent judge agrees on 96.2% of pooled alignment verdicts; details follow
below.
4.2 Main Results
Table 1 and Figure 3 show a scale-dependent crossover. At the bedrock, File-System Agent and BM25 lead
with 77.4 and 74.7, and their 95% intervals (73.9–80.8 and 71.4–77.9) overlap. File-System Agent retains
the higher point estimate at the smallest measured shared tiers; by roughly 10M corpus tokens the curves
have crossed, and BM25 leads at every larger shared tier. At full scale, BM25 retains 50.5, versus 30.7
for File-System Agent and 29.9 for DenseRAG. Under the fixed 80-call budget used here, iterative raw-file
search resists corpus growth less well than global lexical ranking despite many more model calls, while dense
retrieval remains below both.
5

Official combined score by corpus tier (%) @ bedrock (%)
Method build tok/qN=1144 2254 6980 21614 42587 131876 511959 completeness document recall
BM25 0 5.8K 74.7 71.470.1 64.9 61.2 55.2 50.580.490.3
File-System Agent 0 226K77.4 75.469.9 62.6 58.9 50.9 30.781.686.4
DenseRAG emb.‡4.9K 58.1 55.7 51.0 44.2 40.7 36.0 29.9 65.4 73.4
Graph-based Retrieval
HippoRAG 2 7.5M 6.5K 66.2 63.1 58.6 53.8 50.5 41.0 — 72.0 81.3
LinearRAG emb.‡5.8K 46.2 44.1 38.8 34.3 31.3 29.8 — 52.0 62.6
MS-GraphRAG 35.1M 10.4K 45.9 44.0 38.4 — — — — 54.1 57.9
LightRAG 34.6M 8.5K 48.0 42.5 — — — — — 54.7 n/a§
Table 1Official main results. Build/query tokens, completeness, and document recall are bedrock values. Combined
score gates completeness by correctness. Dashes denote unbuilt or unevaluated tiers; Figure 3 shows every measured
tier.‡denotes non-generative construction: embeddings for DenseRAG, and local NER plus embeddings for Linear-
RAG.§marks LightRAG runs that did not log retrieved-evidence identifiers. LinearRAG tok/q averages its fixed,
stratified 100-question metered run.
HippoRAG 2 and LinearRAG stop at 131,876
documents, MS-GraphRAG at 8,750, and Ligh-
tRAG at 2,254. Thus large-scale graph results
are about construction feasibility, not extrapo-
lated accuracy: unsupported tiers remain missing
in the main curves and count as zero only in the
coverage-adjusted cross-scale summary.
4.3 Build Cost and Scaling Walls
Figure 4 (left) and Table 2 identify construction
asthegraphfamilies’scalingwall. HippoRAG2is
approximately linear (b= 1.01), extrapolating to
2.9B generative tokens and roughly three single-
instance days at full scale. Yet at 155M corpus
tokens its 724M-token build scores 41.0, about fif-
teen points below BM25.
 Figure 3Official combined score over the nested ladder with
95% confidence bands. Curves cross around 10M tokens;
graph curves end at their largest feasible tier.
bmeasured to fit @full tok/s est. inst.
HippoRAG 2 1.01 724M @ 154.7M 2.9B 11.9k≈3 d
MS-GraphRAG 0.92 190M @ 10.6M 7.9B 1.8k≈50 d
LightRAG 1.36 73M @ 3.0M≈102B 0.8k≈4 y
LinearRAG — embed only 0 gen. —≈1 d
Table 2Build-cost fitsC(x) =axbagainst corpus tokensx, extrapolated to the 600.8M-token full corpus. “Measured
to” reports build tokens at the largest completed corpus, and tok/s is measured single-instance throughput; the final
two columns are fitted full-scale estimates.
LightRAG is super-linear (b= 1.36), does not complete at 2,826 documents within our resource envelope,
and extrapolates to 102B tokens or four instance-years. Even an optimisticb= 1.2leaves its estimate near
42B tokens.
Token-based fits isolate construction work from hardware throughput. LightRAG’s repeated entity-merge
rewrites yield super-linear growth; even at three times the measured throughput, its projected full-scale build
6

Figure 4Cost scaling across construction and querying. Left: construction tokens and fitted power laws; hollow
markers report embedding tokens for the two non-generative builders, while BM25 and the File-System Agent require
no index. Right: single-stream query latency; lines report medians and the File-System band spans P10–P90.
exceeds one instance-year. Embedding-only construction is far cheaper: DenseRAG completes the full corpus
with 659.4M embedding tokens. LinearRAG reports zero generative construction tokens, but still incurs local
NER and embedding work.
MS-GraphRAG also fits near-linear growth and completes through 8,750 documents; its full-scale estimate
is 7.9B tokens and about 50 instance-days. Parallelism can reduce calendar time but not total construction
work or the resulting accuracy.
4.4 Query Cost and Latency
Figure 4 (right) reports end-to-end latency. Query cost is nearly scale-invariant for the one-shot pipelines:
BM25, DenseRAG, and HippoRAG 2 use 5.8K, 4.9K, and 6.5K tokens per question, dominated by the
shared reader prompt. File-System Agent instead grows from 226K at the bedrock to 343K atN= 21,614,
respectively 39 and 60 times BM25 as exploration deepens. Its median LLM calls rise from 5 to 8 by
N= 42,587; budget exhaustion is below 7% through that tier, but reaches 15% atN= 131,876and 31%
at full scale. Accuracy also falls among within-budget questions, so truncation alone does not explain the
collapse.
Amortized over 500 questions, BM25 obtains 74.7 at 2.9M tokens, versus 77.4 at 112.8M for File-System
Agent. HippoRAG 2 reaches 66.2 at 10.7M, while MS-GraphRAG and LightRAG remain dominated near
40M tokens. The BM25 point remains on the frontier for amortization horizons from 10 to 10,000.
These findings span the full 28-tier, roughly 450-fold ladder rather than a single corpus size. Fixing questions,
gold evidence, and distractors isolates the effect of added background documents. Unified metering reveals
whether cost is paid offline in graph construction or online through per-question file exploration. Across the
seven native pipelines, cross-checked judging, matched retrieval and harness controls, and the typed graph
access layer separate candidate discovery, harness, and substrate effects. This makes the crossover a measured
scaling effect rather than a comparison of independently tuned systems at isolated sizes.
Figure 5 (left) uses six tiers and a fixed 150-question sample. Unsupported graph tiers count as zero only for
coverage adjustment; cost averages construction and normalized 500-query workloads over completed tiers.
7

Figure 5Complementary scaling analyses. Left: cross-scale accuracy–cost summary over six tiers on the fixed 150-
question sample. Right: official combined score by question type atN= 42,587; labels give question counts, and
MS-GraphRAG and LightRAG cannot build at this tier.
Tier Files HippoRAG 2 LightRAG MS-GraphRAG
N=114476.181 75.2 (66.7) 70.8 (48.6) 40.9 (47.4)
1434 83.094 75.2 (65.9) 75.5 (50.7) 44.7 (44.3)
1798 76.315 81.2 (66.7) 72.8 (45.5) 46.6 (46.3)
2254 81.927 75.0 (65.2) 75.6 (45.7) 38.7 (45.4)
6980 75.111 70.6 (62.2) — 36.9 (38.2)
21614 71.242 59.6 (56.0) — —
42587 64.755 54.2 (54.8) — —
Table 3Official score of the same agent harness over each retrieval substrate, with native-pipeline scores on identical
questions in parentheses. Bedrock uses all 500 questions and later tiers a fixed 150-question subset for both members
of each pair. Files has no native pipeline; dashes denote unavailable matched evaluations. These scores come from a
separate matched scoring session and are not mixed with Table 1.
4.5 Behavior by Question Type
Figure 5 (right) shows that atN= 42,587, File-System Agent leads BM25 on intra-document, project-related,
completeness, and conflicting-information questions, including 56 versus 27 on completeness.
BM25 is best or tied on the other five types. The saturated not-found scores mainly measure abstention: one-
shotreadersdeclinewithoutsupportingevidence,whereasiterativeexplorationcancommittoanunsupported
answer. A graph system leads only on miscellaneous.
4.6 Cross-Protocol Robustness
Across nine shared scales, binary re-scoring preserves every family ranking; official correctness is 2.60 points
lower on average (cell range−4.60to+0.69). An independent official judge agrees on 96.2% of pooled
alignment verdicts (94.4–98.0% per cell), with combined-score changes from−3.56to+1.18. Thus close pairs
can vary, but broad family separation remains. At the bedrock, BM25 retrieves any annotated gold chunk
for 94.7% of the 470 answerable questions, all gold chunks for 56.4%, and averages 75.1% gold-chunk recall
at five; its 90.3% document recall locates many residual errors within documents or in evidence synthesis.
8

5 Separating Agency from Structure
5.1 Retrieval Before Agency
The full-scale BM25–File-System gap is a candidate-discovery failure, not weak evidence synthesis. We isolate
the retrieval primitive on the same fixed, stratified 150 questions at the bedrock and full-corpus tiers—300
question–scale pairs, not 300 unique questions. Agent+BM25 keeps the model, harness, answer constraints,
question set, judge, and 80-call budget fixed; it replaces raw-tree tools with ranked lexical search and chunk
reading, together with a tool-specific instruction adapted to that interface. Its first search is forced to use
the original question; an audit confirms that its ordered top-5 exactly matches Native BM25 on every pair.
Combined AtN=511,959
Method 1,144 511,959 DocR Calls Tok./q
Native BM25 81.3 54.8 65.6 1.00 5.8K
File-System 87.1 36.9 36.8 36.12 895K
Agent+BM25 90.1 69.4 72.4 5.79 101K
Table 4Retrieval-primitive control on the same 150 questions at each scale. Combined score and full-scale document
recall (DocR) come from one shared rejudge. Full-scale calls and tokens/question include failed attempts that were
retried. Native BM25 tokens/question is the descriptive full-500 average.
Table 4 shows that at the bedrock, Native BM25 and the File-System Agent are statistically tied: BM25
trails by 5.73 points, with a paired bootstrap 95% CI of[−11.57,0.15]. At the full corpus, BM25 instead
leads by 17.97 points[9.68,26.14]. The mechanism is discovery. Among the queries on which each method
finds at least one gold document—a descriptive, selection-conditioned slice—the File-System Agent scores
85.9 versus BM25’s 73.8, but its any-gold hit rate collapses to 39.0%, against 71.6% for BM25.
Changing only the retrieval primitive reverses the collapse. At full scale Agent+BM25 scores 69.4, beating
the raw-file agent by 32.52 points[24.07,40.90]and Native BM25 by 14.56 points[9.22,20.12]. Its any-gold
hit rate reaches 78.0%, while its 101K tokens/question are roughly one ninth of raw-file exploration. Agency
therefore helps after globally ranked candidate discovery; repeated local search is not a substitute for that
global ranking. We treat this two-scale intervention as a mechanism control, not as an eighth system on the
28-tier native-pipeline ladder.
Retrieval / harness Combined
File-System (ours) 86.3
File-System (Pi-Agent) 82.3
File-System (Codex) 43.9
Native BM25 82.1
Table 5Same-session harness control atN=1,144 on the
same 150 questions, policy model, raw corpus, and judge.
Each harness retains its native loop and stopping limit;
scores are not mixed with Table 4.Table 5 shows that within this matched resweep,
harness choice matters, and ours achieves the high-
est score of the three file agents. Agent+BM25
therefore evaluates the retrieval swap with the
strongest observed raw-file implementation.
5.2 Agency over Graph Substrates
The File-System result confounds agency with its
raw-file substrate. We separate them through a
paradigm-neutral access layer: each resident graph
index exposes typed, read-only tools for seman-
tic search, neighborhood expansion, Personalized
PageRank, and chunk reading, plus its native one-
shot ranker as a tool. The same tool-calling harness
is used with the model, budget, and judge fixed; substrate-specific tools and instructions form the interven-
tion. Graph agents cannot access raw files, so gains reflect policy access to index contents. A command-line
interface is included in the artifact.
Table 3 reports the matched comparison. Agentic access changes native scores by 22.2–29.9 points for
LightRAG (4 tiers),−0.6–14.5 for HippoRAG2 (7), and−6.7–0.4 for MS-GraphRAG (5). Thus the same
index can support different outcomes under an agent and its native one-shot ranker.
9

6 Discussion
6.1 The Lexical Advantage
Enterprise questions contain precise lexical anchors, while traps are semantically similar but factually wrong;
exact matching is therefore advantaged. Table 6 evaluates the ordering’s sensitivity to question wording and
retrieval depth.
ControlNBM25 FS Dense Hippo2 MS-GR
Paraphrase1,144 63.9 73.3 51.8 – –
2,254 60.1 67.2 49.9 54.9 36.3
Top-101,144 83.0 – 70.0 – –
2,254 81.9 – 66.5 73.0 –
Table 6Wording and depth controls: official combined score (%) onn= 148–150; dashes denote unmeasured pairs.
These controls leave BM25 above dense and graph retrieval under comparable retrieval depth. In a 90-
questionproposal-sensitivityaudit,directfull-corpusDenseRAGtop-10retrievaloverlapsthehistoricalBM25-
prefiltered dense candidates by 1.2 documents on average, recovers 57 of 431 confirmed items, and identifies
115 traps plus 100 not-found lures. Together, these controls show that BM25’s advantage persists under
altered wording and matched retrieval depth, while adversarial candidates are also recoverable through a
direct dense proposal path.
6.2 Graph Failure Modes
Three factors explain the shortfall. Construction adds extraction noise—the bedrock’s 32K extracted entities
include malformed fragments—and prohibitive scale cost. Graph retrieval also favors semantically related
but factually wrong neighborhoods, which the traps punish, while HippoRAG 2 discards extracted predicate
text and therefore relation semantics. LinearRAG, with no generative build calls, finishes within 1.8 points of
MS-GraphRAG and LightRAG at the bedrock, suggesting that LLM extraction adds cost faster than signal
here.
6.3 Scaling Mechanism
The decisive factor is corpus-wide candidate discovery. BM25 and DenseRAG amortize global ranking in an
index; lexical matching also rejects this corpus’ factually wrong semantic traps more effectively. File-System
Agent instead explores a local tree sequentially, making relevant branches harder to reach as the corpus grows.
Graph methods require corpus-wide extraction, where cost, noise, and information loss become bottlenecks.
At full scale, Agent+BM25 isolates this mechanism: changing retrieval cuts calls from 36.12 to 5.79 and
tokens from 895K to 101K per question, while document recall rises from 36.8 to 72.4 and score from 36.9 to
69.4. Iteration helps most after global ranking, not in place of it.
6.4 Evaluation Implications
A single corpus size can conceal the crossover, and graph systems can be compared only where their indexes
complete. Cross-paradigm evaluations should report nested-scale accuracy, construction/query cost, and
index coverage. Measured cells describe quality conditional on completion; the coverage-adjusted summary
in Figure 5 (left) describes deployability, assigning zero only to unavailable tiers. Reporting both avoids
treating an unbuilt index as an observed answer failure.
6.5 Practical Guidance
For enterprise-style corpora, BM25 is the appropriate default; aggregation-heavy questions favor Agent+
BM25, which uses lexical ranking for discovery and agentic calls on narrowed candidates. LLM-built graph
10

indexes are difficult to justify at105–106documents unless construction is near-linear and relational questions
dominate.
7 Conclusion
We presented a controlled scaling study of four RAG paradigms on a 28-tier enterprise corpus ladder, holding
questions, relevant evidence, and adversarial distractors fixed while unifying the reader model, judge, and cost
metering. Matched access-layer and retrieval-swap controls further separate substrate from agentic policy.
The result is a scale-dependent crossover: raw-file agency leads at the smallest measured shared tiers, BM25
catches it around 10M corpus tokens, and its lead widens to nearly 20 points at 601M tokens while remaining
Pareto-optimal without LLM-based construction. Replacing raw search with BM25 raises the full-scale agent
from 36.9 to 69.4 at roughly one ninth of its query tokens, whereas graph-based RAG encounters construction
walls. Global lexical ranking therefore becomes the stronger default as enterprise corpora scale, with agentic
reasoning best applied after candidate ranking.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate,
and critique through self-reflection, 2023.https://arxiv.org/abs/2310.11511.
Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. Plan-on-Graph: Self-correcting
adaptive planning of large language model on knowledge graphs, 2024.https://arxiv.org/abs/2410.23875.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropoli-
tansky, RobertOsazuwaNess, andJonathanLarson. Fromlocaltoglobal: AGraphRAGapproachtoquery-focused
summarization, 2025.https://arxiv.org/abs/2404.16130.
Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE: Sparse lexical and expansion model for
first stage ranking, 2021.https://arxiv.org/abs/2107.05720.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. LightRAG: Simple and fast retrieval-augmented
generation, 2025.https://arxiv.org/abs/2410.05779.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG: Neurobiologically
inspired long-term memory for large language models, 2025a.https://arxiv.org/abs/2405.14831.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory: Non-parametric
continual learning for large language models, 2025b.https://arxiv.org/abs/2502.14802.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-augmented
language model pre-training, 2020.https://arxiv.org/abs/2002.08909.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and
Graham Neubig. Active retrieval augmented generation, 2023.https://arxiv.org/abs/2305.06983.
Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan.
SWE-bench: Can language models resolve real-world GitHub issues?, 2024.https://arxiv.org/abs/2310.06770.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han.
Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning, 2025.https://arxi
v.org/abs/2503.09516.
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau
Yih. Dense passage retrieval for open-domain question answering, 2020.https://arxiv.org/abs/2004.04906.
Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search via contextualized late interaction
over BERT, 2020.https://arxiv.org/abs/2004.12832.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao
Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention, 2023.
https://arxiv.org/abs/2309.06180.
11

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive NLP
tasks, 2021.https://arxiv.org/abs/2005.11401.
Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. Pyserini: An
easy-to-use python toolkit to support replicable IR research with sparse and dense representations, 2021.https:
//arxiv.org/abs/2102.10073.
Costas Mavromatis and George Karypis. GNN-RAG: Graph neural retrieval for large language model reasoning, 2024.
https://arxiv.org/abs/2405.20139.
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shan-
tanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: Browser-assisted question-answering with human
feedback, 2022.https://arxiv.org/abs/2112.09332.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph
retrieval-augmented generation: A survey, 2024.https://arxiv.org/abs/2408.08921.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing
the compositionality gap in language models, 2023.https://arxiv.org/abs/2210.03350.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian,
et al. ToolLLM: Facilitating large language models to master 16000+ real-world APIs, 2023.https://arxiv.org/
abs/2307.16789.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham.
In-context retrieval-augmented language models, 2023.https://arxiv.org/abs/2302.00083.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond.Foundations and
Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. RAPTOR:
Recursive abstractive processing for tree-organized retrieval, 2024.https://arxiv.org/abs/2401.18059.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda,
and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023.https://arxiv.or
g/abs/2302.04761.
Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learning, 2023.https://arxiv.org/abs/2303.11366.
Yuhong Sun, Joachim Rahmfeld, Chris Weaver, Weijia Chen, Roshan Desai, Wenxi Huang, and Mark H. Butler.
EnterpriseRAG-Bench: A RAG benchmark for company internal knowledge, 2026.https://arxiv.org/abs/2605
.05253.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-
thought reasoning for knowledge-intensive multi-step questions, 2023.https://arxiv.org/abs/2212.10509.
Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, et al.
OpenHands: An open platform for AI software developers as generalist agents, 2025.https://arxiv.org/abs/24
07.16741.
John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press.
SWE-agent: Agent-computer interfaces enable automated software engineering, 2024a.https://arxiv.org/abs/
2405.15793.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, et al.
CRAG – comprehensive RAG benchmark, 2024b.https://arxiv.org/abs/2406.04744.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing
reasoning and acting in language models, 2023.https://arxiv.org/abs/2210.03629.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li,
et al. Judging LLM-as-a-Judge with MT-Bench and chatbot arena, 2023.https://arxiv.org/abs/2306.05685.
Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao
Huang. LinearRAG: Linear graph retrieval augmented generation on large-scale corpora, 2025.https://arxiv.or
g/abs/2510.10114.
12

A Scope
This supplement provides additional experimental detail, result tables, and reproducibility notes for the
main paper. The main paper is self-contained; this document expands the protocol and gives additional
audit material. All identifiers are anonymized, and all local machine paths are omitted.
B Corpus Ladder
The study uses EnterpriseRAG-Bench, a public synthetic enterprise benchmark with slightly more than
500K documents and 500 questions. The scaling ladder is built by first fixing a bedrock tier that contains all
question-relevant documents, hard negatives, and not-found lures. Every larger tier appends a seeded, source-
and noise-stratified prefix of the remaining corpus. Thus the question set, relevant evidence, and adversarial
documents are fixed while only background corpus size grows. Manifest checks verify exact nesting. Table 7
summarizes every tier.
NCorpus tok. Chunks Mean doc tok. NCorpus tok. Chunks Mean doc tok.
1,144 1.7M 2,018 1472.7 27,097 32.1M 39,374 1182.9
1,434 2.0M 2,435 1411.1 33,970 40.1M 49,252 1181.2
1,798 2.4M 2,955 1361.3 42,587 50.2M 61,615 1177.7
2,254 3.0M 3,606 1324.0 53,389 62.8M 77,142 1176.4
2,826 3.6M 4,425 1288.2 66,932 78.6M 96,635 1174.8
3,543 4.5M 5,454 1264.3 83,909 98.6M 121,123 1174.7
4,441 5.5M 6,737 1242.4 105,193 123.5M 151,751 1174.4
5,568 6.8M 8,374 1227.4 131,876 154.7M 190,096 1173.4
6,980 8.5M 10,419 1217.1 165,327 194.0M 238,321 1173.4
8,750 10.6M 12,974 1208.6 207,263 243.2M 298,783 1173.4
10,970 13.2M 16,169 1202.2 259,837 304.9M 374,576 1173.4
13,753 16.4M 20,185 1195.2 325,746 382.2M 469,677 1173.4
17,241 20.5M 25,203 1190.0 408,373 479.1M 588,566 1173.3
21,614 25.6M 31,475 1186.6 511,959 600.8M 737,878 1173.6
Table 7The 28 strictly nested corpus tiers. Corpus tokens are measured with the shared tokenizer; chunks are
produced by the shared 1,200-token chunker with 100-token overlap for systems that use the shared chunks.
C Method Configuration
All native pipelines are evaluated under one reader and judging protocol. The shared reader is Qwen3.6-27B
served with temperature zero and thinking disabled. The shared embedding model is Qwen3-Embedding-
0.6B. BM25, DenseRAG, and HippoRAG 2 use the same shared chunks. MS-GraphRAG and LightRAG use
their native internal chunking, and retrieved evidence is mapped back to the shared chunk identifiers when
document recall is available. Table 8 lists the shared settings.
Component Setting
Chunk size 1,200 tokenizer tokens
Chunk overlap 100 tokenizer tokens
Retriever depth top-5 chunks where applicable
Reader temperature 0.0
Reader top-p 1.0
Agent budget 80 LLM calls/question
Judging unit one prediction per method/question/tier
Uncertainty question bootstrap, 10,000 resamples
Table 8Shared settings used across native pipelines and matched controls.
13

C.1 Reader Prompt
All non-agentic pipelines use the same reader prompt. The system message is:
Reader system prompt
You are a retrieval-based QA assistant. Answer the QUESTION using ONLY the information in the CONTEXT.
Do not use outside knowledge or invent facts. If the CONTEXT does not contain enough information to answer,
say so explicitly. Answer in the same language as the QUESTION; be concise.
The user message template is:
Reader user-message template
CONTEXT:
{context}
QUESTION: {question}
ANSWER:
C.2 File-System Agent Prompt and Tools
The File-System Agent uses the same Qwen3.6-27B model as its policy model. The prompt gives only
read-only access to the corpus tree:
File-System Agent system prompt
You are a research assistant over an enterprise document corpus, organized as files under source folders (slack,
gmail, jira, confluence, google_drive, linear, github, fireflies, hubspot). Use list_dir to orient, grep to locate
relevant documents by keyword, and read_doc to read them. Answer ONLY from the documents. Be concise and
factual, and cite the relative file paths you used.
Table 9 lists the three exposed operations.
Tool Arguments Return value
list_dirrelative path up to 200 child names
grepfixed string up to 30 matching paths
read_docrelative path first 8,000 characters
Table 9Read-only tools exposed to the raw File-System Agent. Path resolution rejects traversal outside the corpus
root.
C.3 Agent+BM25 Prompt and Control Guarantee
Agent+BM25 uses the same agent harness and budget, but replaces raw tree search with a ranked BM25
search tool. Its system message is:
Agent+BM25 system prompt
You are a research assistant over an enterprise document corpus. Use bm25_search with the user’s original
question first. Inspect the returned top-5 chunks carefully. If they are sufficient, answer immediately; otherwise
reformulate the search query or use read_doc on a returned source path to inspect more context. Answer ONLY
from retrieved documents. Be concise and factual, and cite source paths or chunk IDs. Do not use outside
knowledge.
The firstbm25_searchcall is programmatically forced to use the original question, regardless of the tool
argument. This makes the first returned top-5 exactly the native BM25 top-5. Later calls may use agent
reformulations and are the agentic part of the intervention.
14

C.4 Official Judge Prompts
The official combined score has two LLM-judged components and one non-LLM document-recall component.
First, a holistic answer-alignment prompt receives the query, gold answer, and candidate answer and asks
for a JSON object with a one-sentence reason and analignedfield whose value isyesorno. The prompt
instructs the judge to accept stylistic differences and extra relevant context, but to reject contradictions,
wrong quantities, or missing core parts of the query. Second, each atomic answer fact is validated with a
yes/no prompt asking whether the candidate answer contains and is consistent with the statement. The
combined score for a question is:
combined(q) =(
completeness(q),aligned(q) = 1,
0,aligned(q) = 0.
Reported cell scores are the mean combined score over the evaluated questions. Document recall is computed
by exact set overlap between retrieved document identifiers and gold document identifiers for answerable
questions.
D Figure and Table Metrics
The main ladder reports native-pipeline scores only for tiers actually built and evaluated. Dashes in the
tables therefore denote missing native results rather than extrapolated failures.
Figure 1 is a schematic of the observed BM25–File-System Agent crossover. Its roughly 10M-token marker
summarizes the interval between the neighboring measured tiers where their point-estimate ordering changes;
it is a rounded regime marker rather than a fitted threshold or significance boundary. Figure 3 reports the
measured curves and confidence bands.
Figure 5 (left) summarizes six fixed tiers (N= 1,144,2,254,6,980,42,587,131,876,511,959) on one fixed 150-
question sample. Its vertical coordinate is coverage-adjusted: unsupported graph tiers are assigned zero
only for this summary. Its horizontal coordinate is the mean over completed tiers of construction cost plus
a normalized 500-query workload. Embedding tokens are parameter-weighted by0.6/27when combined
with 27B reader or generative tokens. The plot’s point ledger is included asfigures/teaser_crossscale_-
points.csvin the code/data supplement.
Table 1 in the main paper reports bedrock build/query tokens and bedrock completeness/recall alongside
selected ladder scores. Table 2 fits build-token scaling laws of the formy=axbon measured build tokens
against corpus tokens and projects to the 600.8M-token full corpus. These projections are not used as hidden
accuracy estimates; the accuracy curves show only measured cells.
E Token Metering
The metering layer records prompt, completion, and total tokens for each LLM call and assigns the call to
construction or query phases. For systems that expose OpenAI-compatible responses, usage fields are read
directly from the response. For HippoRAG 2, build tokens are recovered from its LLM cache metadata.
For MS-GraphRAG, build tokens are recovered from its indexing logs. Embedding calls are counted from
tokenizer inputs and reported separately from generative LLM calls. LinearRAG’s zero generative-build
entry therefore means zero generative LLM construction tokens, not zero CPU work, storage, local NER, or
embedding work. Table 10 summarizes the accounting sources.
15

Family Build-token source Query-token source
BM25 zero model-token build reader usage
DenseRAG embedding tokenizer inputs reader usage
HippoRAG 2 LLM cache metadata reader usage
MS-GraphRAG indexing logs reader usage
LightRAG wrapper usage logs reader usage
LinearRAG embedding backfill reader usage
File-System zero model-token build agent call usage
Table 10Token-accounting sources by family. “Zero” denotes zero model-token construction cost, not zero CPU,
storage, or indexing work.
F Full Native-Ladder Scores
Table 11 and Table 12 expand the main result matrix beyond the subset displayed in the main paper. Dashes
denote tiers that were not built or not evaluated for that native pipeline.
Method 1144 1434 1798 2254 2826 3543 4441 5568 6980 8750 10970 13753
BM25 74.7 – – 71.4 – – – – 70.1 – – –
File-System Agent 77.4 – – 75.4 – – – – 69.9 – – –
DenseRAG 58.1 – – 55.7 – – – – 51.0 – – –
HippoRAG 2 66.2 – – 63.1 – – – – 58.6 – – –
LinearRAG 46.2 45.3 46.1 44.1 42.0 42.5 40.3 39.0 38.8 35.5 35.0 34.8
MS-GraphRAG 45.9 43.4 44.4 44.0 42.5 40.4 38.5 38.7 38.4 35.5 – –
LightRAG 48.0 46.5 44.3 42.5 – – – – – – – –
Table 11Official combined score (%) for smaller native-ladder tiers.
Method 17241 21614 27097 33970 42587 53389 66932 83909 105193 131876 259837 511959
BM25 – 64.9 – – 61.2 – 59.5 – – 55.2 53.0 50.5
File-System Agent – 62.6 – – 58.9 – 56.7 – – 50.9 – 30.7
DenseRAG – 44.2 – – 40.7 – 38.1 – – 36.0 32.8 29.9
HippoRAG 2 – 53.8 – – 50.5 – – – – 41.0 – –
LinearRAG 33.2 34.3 32.2 32.2 31.3 30.5 30.9 28.7 28.5 29.8 – –
MS-GraphRAG – – – – – – – – – – – –
LightRAG – – – – – – – – – – – –
Table 12Official combined score (%) for larger native-ladder tiers.
G Matched Controls
The retrieval-primitive control replaces raw file-tree exploration with BM25 search while holding the agent
model, prompt, tools, and 80-call budget fixed. The first search is forced to use the original question, and an
audit confirms that the returned top-5 order equals the native BM25 top-5 on each matched pair before any
further agentic reasoning. This isolates global candidate discovery from the policy loop.
The graph-substrate control exposes each graph index through typed, read-only tools: entity/fact/relation
search, neighborhood expansion, Personalized PageRank where supported, and chunk reading. Agents over
graph substrates cannot read raw files, so gains over native one-shot graph rankers come from policy access
to the stored index rather than from access to extra corpus text. Table 13 summarizes what each matched
control fixes and changes.
16

Control Question set What is fixed What is changed
Agent+BM25 same 150 at
bedrock/fullmodel, prompt, 80-call budget,
judgeraw search→BM25 top-5 tool
Harness same 150 at bedrock policy model, raw files, budget,
judgeagent harness implementation
Graph substrate
agentmatched tier samples agent model, budget, judge stored substrate/tools
Top-10 available matched
cellsreader, judge, questions retrieval depth 5→10
Paraphrase fixed paraphrase
subsetcorpus, gold, judge question wording
Table 13Matched controls used to separate retrieval substrate, agentic policy, question wording, and retrieval-depth
effects.
ControlNBM25 FS Dense Graph
Paraphrase 1,144 63.9 73.3 51.8 –
Paraphrase 2,254 60.1 67.2 49.9 54.9 / 36.3
Top-10 1,144 83.0 – 70.0 –
Top-10 2,254 81.9 – 66.5 73.0
Table 14Wording and retrieval-depth controls. Graph
entries are HippoRAG 2 and, where available, MS-
GraphRAG.
H Robustness Checks
The official judge uses correctness-gated completeness as the combined score. A simpler binary protocol
preserves all broad family rankings at the nine shared native-pipeline scales. An independent official judge
agrees with the primary judge on 96.2% of pooled alignment decisions, with per-cell agreement ranging from
94.4% to 98.0%. Close pairs can move by a few points, but the main separation between BM25, dense
retrieval, raw-file agency at scale, and graph construction families is stable.
The lexical-overlap controls use paraphrased questions and top-10 retrieval budgets at the two smallest tiers
where enough methods are available. BM25 remains above dense and graph retrieval under comparable
retrieval depth. A proposal-sensitivity audit also performs direct full-corpus DenseRAG top-10 retrieval for
90 questions, confirming that trap proposals are not exclusively reachable through the BM25-prefiltered dense
pool used during benchmark construction. Exact matched-cell values are reported in Table 14.
I Failure and Stopping Criteria
For LLM-built graph methods, a tier is counted as completed only when the construction artifact is usable
for all scheduled questions and its token ledger is available. If a build exceeds the resource envelope, fails
during entity/relation extraction or merge, or produces an incomplete index that cannot answer the scheduled
cell, the tier is reported as unavailable rather than assigned an inferred accuracy in the main ladder. The
cross-scale summary is the only place where unavailable graph tiers are converted to zero, and its caption
labels the result as coverage-adjusted.
File-System Agent failures are handled at question level. Failed or retried attempts are included in the full-
scale call and token totals. Budget exhaustion is recorded separately from answer correctness. The main text
reports that accuracy also falls among within-budget questions, so the full-scale loss is not only a truncation
artifact.
17

J Data Schemas
Native prediction files use one JSON object per question with fields:
Native prediction record
id, question, predicted_answer, retrieved_chunk_ids,
retrieved_doc_ids, latency_sec, metadata
Official judgment files bind a judgment to the answer and retrieved evidence using a SHA256 hash and record:
Official judgment record
id, aligned, reason, completeness_pct, n_facts,
fact_results, doc_recall, prediction_sha256
Token ledgers are phase separated:
Phase-separated token ledger
baseline, dataset, source, build{prompt,completion,total,calls},
qa{prompt,completion,total,calls}, build_qa_total_tokens
K Bootstrap Intervals
Confidence intervals in the native ladder resample questions with replacement within each method–tier cell.
Each interval uses 10,000 bootstrap resamples and reports the 2.5th and 97.5th percentiles. Matched controls
use paired resampling over the same question IDs so that the confidence interval is over the paired score
difference.
L Artifact-to-Claim Map
The major empirical claims map to the following code/data artifacts.
•Native-ladder accuracy and confidence bands:results/judge_official_canonical_20260724/.
•Build scaling:results/*scaling_ledger.csvandresults/build_cost_authoritative.csv.
•Query tokens:results/token_usage/.
•Agent+BM25:results/judge_agent_bm25_control/andresults/agent_bm25_control_report.
md.
•Graph-substrate agents:results/judge_official_table3_20260724/.
•Lexical controls:results/judge_paraphrase_*andresults/judge_topk10_*.
•Cross-scale summary:figures/teaser_crossscale_points.csv.
M Artifact Contents
The code/data supplement is organized as follows.
•scripts/: sanitized scripts for constructing tiers, running native pipelines, metering tokens, judging
outputs, and producing paper figures.
•results/: aggregate CSV/JSON ledgers used for the paper tables, figures, and mechanism controls.
•figures/: figure-generation inputs and final PDFs;README.md: environment variables and reproduc-
tion order.
18
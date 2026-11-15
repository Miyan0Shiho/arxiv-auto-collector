# RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication

**Authors**: Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, Shuai Wang

**Published**: 2026-11-12 07:34:28

**PDF URL**: [https://arxiv.org/pdf/2611.16425v1](https://arxiv.org/pdf/2611.16425v1)

## Abstract
We describe the RMIT-ADM+S team's participation in the TREC 2026 Retrieval-Augmented Generation (RAG) track using the ClimbMix corpus. We submitted seven research-agent runs using proprietary OpenAI and open-weight models, exploring different retrieval strategies, writer models, and agent workflows. Pre-submission ablations identified writer selection and adjacent-chunk expansion as the main factors affecting answer quality. In the official NIST evaluation, six of seven runs exceeded the track median across all three rubric/nugget collections. Our best-performing run, brief-hybrid, achieved the highest all-item coverage among our submissions and tied the track maximum (0.892) for Batch-1 vital coverage. Human side-by-side preferences differed across evaluation batches: the open-weight owa-glm5 led Batch 1 despite lower automated coverage, while sol-research-first led Batch 2. These findings highlight differences between automated coverage measures and human preferences, as well as the potential of lower-cost open-weight models for research-agent systems.

## Full Text


<!-- PDF content starts -->

RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever
Ablations, and Open-Weight Replication
Oleg Zendel
RMIT University
Melbourne, AustraliaKun Ran
RMIT University
Melbourne, AustraliaChenglong Ma
RMIT University
Melbourne, Australia
Nguyen Khoi Vo
RMIT University
Melbourne, AustraliaShuai Wang∗
The University of Queensland
Brisbane, Australia
Abstract
We describe the RMIT-ADM+S team’s participation in the TREC 2026
Retrieval-Augmented Generation (RAG) track using the Climb-
Mix corpus. We submitted seven research-agent runs using propri-
etary OpenAI and open-weight models, exploring different retrieval
strategies, writer models, and agent workflows. Pre-submission ab-
lations identified writer selection and adjacent-chunk expansion as
the main factors affecting answer quality. In the official NIST evalu-
ation, six of seven runs exceeded the track median across all three
rubric/nugget collections. Our best-performing run, brief-hybrid ,
achieved the highest all-item coverage among our submissions and
tied the track maximum (0.892) for Batch-1 vital coverage. Hu-
man side-by-side preferences differed across evaluation batches:
the open-weight owa-glm5 led Batch 1 despite lower automated
coverage, while sol-research-first led Batch 2. These findings
highlight differences between automated coverage measures and
human preferences, as well as the potential of lower-cost open-
weight models for research-agent systems.
Keywords
retrieval-augmented generation, TREC RAG, agentic retrieval, Re-
ciprocal Rank Fusion, open-weight models, trajectory fine-tuning
1 Introduction
We describe our participation in the TREC 2026 RAG track [ 23],
using the ClimbMix-400B-shuffle corpus [ 9]. The task involves
producing cited reports of up to 1,024 words for 119 test narratives.
We submitted seven research-agent runs using proprietary OpenAI
and open-weight models, exploring different retrieval strategies,
writer models, and agent workflows. These comparisons help assess
whether more complex agent workflows and proprietary models
justify their respective computational and monetary costs. This
work complements our recent RAG competition systems, which
explored generation-assisted retrieval and reranking in LiveRAG
and adaptive retrieval with lightweight open-weight models in
MMU-RAG [19, 20].
Our investigation focuses on three research questions:RQ1:Can
open-weight agents achieve comparable quality to proprietary mod-
els at lower cost, and what explains any performance gaps?RQ2:
Which model, retrieval, and orchestration choices have the greatest
∗Now at keenable.aiimpact on quality and token cost?RQ3:Does multi-stage orches-
tration improve upon a simpler single-loop agent or compensate
for a weaker writer?
We evaluate these questions using pre-submission RAGDoll-
based diagnostics (Section 4) and official NIST evaluations covering
17 Batch-1 and 25 Batch-2 narratives (Section 5). Contrary to ex-
pectations from our automated pre-submission evaluation, human
preferences differed across evaluation settings: the open-weight
owa-glm5 was significantly preferred under holistic evaluation in
Batch 1, while the proprietary sol-research-first was preferred
under nugget-guided evaluation in Batch 2. This highlights how
system rankings can differ not only between automated and human
evaluations, but also between human evaluations conducted under
different guidelines.
2 System Overview
2.1 End-to-End Architectures
We use component markers to distinguish instructions embedded
in prompts(P), separately invoked LLM stages(L), and executable
code(C).
All seven submissions generate cited reports of up to 1,024 words
directly from the test topics. Citations to individual chunks are
mapped to parent document IDs in the final output. Table 1 sum-
marizes the system configurations, while Figure 1 illustrates the
control flow of our leading coverage run and two SxS priority runs.
Retrieval 𝑘is selected by the agent for each search call, except in
facetrag, where it is fixed by code. When adjacent-chunk expan-
sion is enabled, the system automatically retrieves the preceding
and following chunks for the top-ranked search results.
Five submissions share a staged-evidence research architecture,
differing in models, retrieval methods, planning stages, and review
mechanisms. A shared evidence ledger lets agents select and retain
retrieved chunks for report generation. Code enforces a 500,000-
token context budget, a maximum of three citations per sentence,
and the report word limit.
•sol-default uses a single GPT-5.6-Sol conversation for
research and drafting, with up to 100 research rounds(P)and
no separate planning or review stages.
•sol-research-first uses GPT-5.6-Sol across all stages.
An isolated coverage planner(L)generates a research plan,
supplemented by up to eight additional requirements from
a blind obligation scout(L)that sees only the request. These
1
arXiv:2611.16425v1  [cs.IR]  12 Nov 2026

TREC 2026, November 2026, Gaithersburg, MD, USA Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, and Shuai Wang
brief-hybrid owa-glm5 sol-research-first
GPT-5.6-Sol across all stages GLM-5 across all stages (open-weight) GPT-5.6-Sol across all stages
Requirements brief (L)
Isolated LLM call ( ≤8obligations)Requirements brief (L)
Isolated LLM call ( ≤8obligations)Coverage plan (L)
Isolated LLM call: prose research plan
Blind obligation scout (L)
Request only;≤8additions to briefBlind obligation scout (L)
Request only;≤8additions to plan
Research and draft
Hybrid RRF ( 𝜅= 60),𝑘= 10; adj. on
Shared loop (100-round cap)Research and draft
Hybrid RRF ( 𝜅= 60),𝑘= 10; adj. on
Shared loop (100-round cap)Research and draft
Semantic + keyword, 𝑘= 10; adj. on
Separate research loop (40-round cap)No blind scout
Draft review (once)
Uncited scan (C); reviewer (L)
Checks draft against requirements briefDraft review (once)
Uncited scan (C); reviewer (L)
Brief checked; scout items excludedfeedback
feedback
Output export (C):map <docid>_p<page> citations to parent docid and emit RAG26 JSONLno feedback no feedbackNo post-draft review
Final answer from the same writer looprevised draft
revised draft
Expansion of each purple Research and draft block (dashed orange = P; solid blue = L/C)
Research/writer instructions (P)
Formulate queries; assess coverage gaps;
draft cited prose in the same conversationRetrieval tools and evidence ledger (C)
Search / fetch; expand ±1adjacent chunks (top 5);
stage batch; retain up to 10 via commit_contexttool calls
staged / committed chunks
Unselected staged chunks expire next turn. Code enforces a 500k-token context budget, valid citation IDs,
at most three citations/sentence, and a 1,024-word ceiling (no minimum). No code-level entailment checks.
Figure 1: Control flows for three submissions: our leading coverage run ( brief-hybrid ) and two SxS priority runs ( owa-glm5 ,
sol-research-first ). Purple blocks are expanded in the panel below. Solid arrows show execution order, not prompt inputs;
each scout sees only the request. Unboxed notes mark absent stages; 𝑘is the typical requested passages per search. Dashed
blue arrows are conditional: review feedback resumes the writer loop, which may search again; the revised draft bypasses the
once-only review. For staged evidence see Section 3.1.
Table 1: Configurations of the submitted systems. Runs marked ★entered NIST human SxS assessment. Models denote the
main and supporting LLMs (orchestrator and analyzer for facetrag ). Sem: semantic; Kw: keyword; Hyb: hybrid RRF; SSR:
shortest-substring ranking; Bool: Boolean search; Brief: requirements brief; Scout: blind obligation scout; Rev: post-draft
review; Adj.: automatic adjacent-chunk expansion. Retrieval 𝑘denotes the most frequently requested number of passages per
search call.
System Models (main / support) Retrieval (𝑘) Planning / Review Adj.
sol-defaultSol / — Sem+Kw (10) None No
sol-research-first★Sol / Sol Sem+Kw (10) Plan+Scout Yes
brief-baseSol / Sol Sem+Kw (10) Brief+Rev Yes
brief-hybridSol / Sol Hyb (10) Brief+Rev Yes
facetsLuna / — Sem+Kw (10), Hyb (15) Facets+coverage check No
facetragGPT-OSS-120B / Qwen3-Next-80B Sem+Kw+SSR+Bool (10), Hyb (15) Facets+multi-stage review —
owa-glm5★GLM-5 / GLM-5 Hyb (10) Brief+Scout+Rev Yes
outputs then guide a 40-round research loop, followed by
drafting and a single-pass post-draft review(P).
•brief-base andbrief-hybrid extend sol-default with
an isolated pre-flight requirements brief(L)and a single-pass
post-draft review(L), both using GPT-5.6-Sol . The writer
conducts research and drafting, guided by the brief(P). A
code-based citation scan(C)flags uncited sentences, and re-
viewer feedback(L)is returned to the writer for revision.
The two runs differ primarily in retrieval: brief-base pro-
vides the agent with semantic and keyword search, whilebrief-hybrid uses hybrid retrieval with Reciprocal Rank
Fusion (RRF).
•owa-glm5 follows the brief-hybrid architecture but uses
the open-weight GLM-5 model across all stages. It also adds
a blind obligation scout(L), whose requirements supplement
the initial brief but are excluded from the post-draft review.
Pre-flight briefs and blind scouts each identify up to eight answer
requirements. Scouts receive only the original test topic, indepen-
dently of the other planning stages. Two additional submissions
2

RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication TREC 2026, November 2026, Gaithersburg, MD, USA
explore explicit facet decomposition, dividing each topic into sub-
questions requiring separate evidence:
•facets uses a single GPT-5.6-Luna conversation to plan,
research, and draft across multiple facets(P). Each search is
associated with a specific requirement. A code-based cov-
erage check(C)identifies unresolved requirements and may
return the draft for further research. No separate reviewer
is used.
•facetrag combines a GPT-OSS-120B orchestrator with a
Qwen3-Next-80B-A3B analyzer to research 3–6 facets con-
currently. Each facet has separate search-planning calls(L)
and supports semantic, keyword, hybrid, and Boolean re-
trieval, including Shortest-Substring Ranking (SSR). The
analyzer(L)identifies relevant passages and remaining gaps.
An LLM curator(L)selects evidence based on relevance and
diversity using an MMR-style approach [ 4], supplying up
to five items per facet. Separate drafting, citation-checking,
and formatting stages(L)produce the final report, followed
by code validation(C).
2.2 Factorization of Design Choices
RAG systems combine interacting choices across retrieval, genera-
tion, curation, and evaluation; motivating systematic comparison
of individual pipeline components [ 21]. We catalogued 14 struc-
tural factors and evaluated selected ablations (Section 4), along-
side writer and helper model choices (writer, brief analyst, re-
viewer).1Factors include pre-flight briefing, blind obligation scout-
ing, single-pass review, ±1adjacent-chunk expansion, retrieval
engines and cutoffs, search-preview truncation, full-chunk staging
versus LLM-generated snippets, and evidence pruning. brief-base
andbrief-hybrid were developed using one-factor-at-a-time hill-
climbing on a fixed 15-topic development set, evaluated with a
rubric grader and head-to-head arena comparisons. brief-hybrid
differs only in its retrieval engine (hybrid only). It achieved the high-
est rubric score, confirmed on a second 15-topic set, but a lower
arena win rate thanbrief-base, so both were submitted.
sol-research-first was independently developed on a 30-
topic development set against a Sol baseline. Adding an isolated
coverage plan and a blind atomic scout to a continuous research con-
versation, which searches, retains evidence, and drafts the answer,
improved rubric scores. Adding a post-draft repair stage reduced
performance, so this stage was dropped. “First” refers to the or-
dering of planning and research before drafting. Across all runs,
code enforces context, round, length, and citation-ID constraints,
while semantic completeness and citation entailment remain model-
driven.
2.3 Resource Usage and Cost Estimates
Table 2 summarizes resource usage and estimated costs across all
119 test topics. sol-research-first produced the longest reports
(936.6 words on average) and consumed 81% more processed tokens
than sol-default . Compared with brief-base ,brief-hybrid
1Detailed factor definitions, implementation pointers, and reports are available in the
code repository: https://github.com/rmit-ir/trec-rag-26.Table 2: Average resource usage and estimated cost per re-
port across all 119 test topics. Srch: search calls; Refs: cited
references; In: uncached input tokens; Cache: cached input
tokens; Out: output tokens. Token counts are in thousands
(K). Runs marked★entered NIST human SxS assessment.
System Srch Words Refs In (K) Cache (K) Out (K) $/topic
sol-default23.9 882.5 27.9 182.2 202.4 6.1 1.566
sol-research-first★31.4 936.6 27.7 330.5 369.3 10.5 2.823
brief-base26.4 830.2 25.8 319.6 388.5 8.0 2.356
brief-hybrid22.6 834.3 24.7 293.1 322.4 7.5 2.156
facets11.7 613.0 13.6 138.9 71.4 5.9 0.040
facetrag33.7 388.9 13.9 411.7 0 49.7 0.106
owa-glm5★12.7 754.0 15.9 415.1 0 5.9 0.524
maintained similar report length and citation counts while reduc-
ing searches by 14.4% and processed tokens by 8.2%. The open-
weight owa-glm5 required fewer searches (12.7 per topic) and had
a substantially lower estimated cost than sol-research-first
($0.524 vs. $2.823 per topic). facetrag issued the most searches
(33.7 per topic) but produced the shortest reports (388.9 words on
average), despite generating 49.7K output tokens across its multi-
stage pipeline.
Costs are averages over successful attempts on the 119 topics,
calculated from traced input, cache-read, and output tokens, sup-
plemented by logged brief, review, and scout calls where necessary,
using provider-specific rates. Estimates cover generation only, ex-
cluding failed attempts, experimentation, and judging.
3 Retrieval and Evidence Handling
We segmented ClimbMix-400B-shuffle [9] (6,543 shards, approx-
imately 563 million documents) into paragraph-aware chunks, fol-
lowing recent findings that simple structure-based chunking strate-
gies perform well for in-corpus retrieval [ 31]. The resulting BM25
index contains 921,892,634 chunks. Token counts were estimated
at 1.3 tokens per word, targeting 200–500 tokens per chunk with
a hard cap of 700. Each chunk was assigned a unique, 1-based
<docid>_p<page>identifier.
Sparse Lexical and Boolean Retrieval.We used a self-hosted
Lucene/Anserini index [ 16,27] with Pyserini’s default BM25 pa-
rameters ( 𝑘1=0.9,𝑏= 0.4) for keyword and Lucene Boolean
search. facetrag additionally supports Shortest-Substring Ranking
(SSR) [ 6] through a Cottontail-based Boolean retrieval system [ 7],
which ranks results by compact matching spans.
Dense Semantic Retrieval.For semantic retrieval, we used our
jina-v5-nano-trecrag26-agent-256d model,2fine-tuned from
jina-embeddings-v5-text-nano [2], producing 256-dimensional
embeddings. We indexed these embeddings using DiskANN [ 14],
an approximate nearest-neighbor search method designed for large-
scale, disk-based retrieval. The index used maximum inner product
search, with graph degree 𝑅= 64, construction search list size
𝐿build=100, and 32-byte product quantization.
On 22 TREC RAG 2025 development topics (condensed metrics
averaged over three judge variants), the fine-tuned model achieved
slightly higher aggregate scores than the base model: nDCG@10
2Model card: https://huggingface.co/RMIT-ADMS/jina-v5-nano-trecrag26-agent-
256d.
3

TREC 2026, November 2026, Gaithersburg, MD, USA Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, and Shuai Wang
(0.720 vs. 0.713), MAP (0.113 vs. 0.103), and R@1000 (0.116 vs. 0.106).
These small differences should be interpreted cautiously given the
limited validation set.
Hybrid Retrieval.Hybrid search combines semantic and lexical
retrieval using unweighted Reciprocal Rank Fusion (RRF) [ 8], with
𝜅=60. Both retrievers return at least 50 candidates before fusion,
and the top- 𝑘chunks are returned. When adjacent-chunk expansion
is enabled (Table 1), the system also retrieves the preceding and
following chunks for the top five results, removing duplicates.
Query Generation.For hybrid retrieval, facets andfacetrag
use HyDE-style instructions [ 12] that encourage generating short
hypothetical answer passages as search queries. Query formula-
tions can vary substantially across both generation strategies and
search contexts [ 1,28], motivating direct evaluation of the query-
generation strategy. These instructions appear in the main prompt
forfacets and in individual search-planning calls for facetrag .
They do not enforce this query format, and hypothetical passages
are used only for retrieval, not as evidence.
On 15 development topics, plain hybrid retrieval scored 2.333/3,
compared with 2.267/3 when adding HyDE-style instructions. Given
the small difference, we found no clear benefit from HyDE-style
query generation and used plain hybrid retrieval in brief-hybrid
and itsowa-glm5derivative.
3.1 Staged Evidence and Retriever Fine-Tuning
We used a staged evidence approach to manage retrieved chunks
and collect training data for the fine-tuning of the retriever. Re-
trieved chunks are initially available for one turn. The agent can call
commit_context to retain selected chunks in its evidence ledger;
uncommitted chunks are reduced to markers and expire from the
staged batch. We fine-tuned our dense retrieval encoder using
evidence-selection decisions from initial agent runs. Recent work
has similarly shown that search-agent trajectories can provide use-
ful supervision for training retrievers that account for the agent’s
interaction history [ 5,30]. Two GPT-5.6- Luna configurations
(semantic-only and keyword-only) processed all 119 official test
topics. Chunks committed at any point during a run were labeled
positive, while retrieved but non-selected chunks were treated as
negative candidates. The resulting dataset export contained 2,143
search rows (1,073 semantic and 1,070 keyword), with 4,857 positive
and 12,889 negative query–chunk pairs. These counts describe the
exported data, not the final training set.
We trained the retriever on 95 topics, comprising 1,587 search
sub-queries and 3,849 positive query–chunk pairs, and held out 24
topics for validation. We used a patched version of Tevatron [ 13]
with InfoNCE [ 25], contrasting positive query–chunk pairs against
other passages in the same batch (in-batch negatives). Training
used 𝜏= 0.05, batch size 64, 4 epochs, learning rate10−4, bf16,
and the first 256 embedding dimensions before normalization. We
applied LoRA with 𝑟= 16,𝛼= 64, and dropout0 .1. On the 24
held-out topics (410 sub-queries), we observed an improvement
in nDCG@10 from 0.751 to 0.793, evaluated over a pool of 4,071
agent-committed chunks. Based on agent-derived labels, rather
than human relevance judgments or full-corpus retrieval.
The training data came from the official 2026 test topics, without
using actual relevance labels. The held-out retrieval evaluationtherefore does not establish independent end-to-end generation
performance.3All dense and hybrid submissions used this fine-
tuned encoder.
4 Internal Evaluation (Pre-Submission)
We report pre-submission evaluations separately from the official
results (Section 5). These evaluations used different assessment
methods and, in some cases, different topic sets, so their scores are
not directly comparable with the organizers’ results. Nevertheless,
they provide insights into the relative performance of our systems
under automated evaluation and the effects of design choices, help-
ing guide our final submissions.
4.1 Evaluation Protocols and Limitations
Pre-submission evaluation used two automated RAGDoll protocols:
pointwise grading and pairwise arena comparisons.4Pointwise
grading assessed approximately 31 criteria (scored 0–2) across six
dimensions (Communication, Explicit Criteria, Implicit Criteria,
Instruction Following, References/Citations, and Synthesis), plus
an overall quality score (0–3), using GPT-5.6-Terra . Pairwise com-
parisons assessed which of two responses was preferred, evaluating
both presentation orders (AB/BA).
System development used the 30 organizer Research-Rubrics [ 22]
development topics, split into two disjoint 15-topic sets for tuning
and validation. No submitted system configuration was selected
using the 119 test topics, although the retriever was separately
fine-tuned on agent trajectories from test topics (Section 3.1).
Our internal experiments revealed substantial sensitivity to the
evaluation setup. On the same 20 topics, changing only the pair-
wise judge shifted our system’s win rate against the organizer
baseline from 17.5% ( GPT-5.6-Sol ) to 45.0% ( GPT-5.6-Luna ), with
GPT-5.6-Terra giving 25.0%. The judges were not equally indepen-
dent of the compared systems, and changing answer presentation or-
der also affected outcomes. Arena preferences were correlated with
response length, while separate citation-support checks identified
weaknesses that pairwise judgments did not capture. These findings
highlight the limitations of relying on any automated judgments
alone, consistent with prior observations that LLM-based assess-
ment can be sensitive to evaluator and presentation choices [ 10,29].
4.2 Writer, Retrieval, and Orchestration Sweeps
In the 15-topic brief-base sweep, writer choice had the largest
impact on pointwise quality. Relative to the GPT-5.6-Sol baseline
(2.267/3), switching to GPT-5.6-Terra orLuna reduced scores by
0.40, Qwen3-Next-80B-A3B by 0.867, and GPT-OSS-120B by 1.067.
Of 12 structural changes, 10 changed scores by at most 0.067 (equiv-
alent to one topic on this benchmark), while two reduced scores
by 0.133 (search-preview truncation and adding SSR with Lucene
Boolean search). Hybrid-only RRF ( brief-hybrid ) improved the
pointwise score (2.333 vs. 2.267), with a larger advantage on the
second, 15-topic validation set (2.333 vs. 2.133). As shown in Table 2,
across the 119 test runs, brief-hybrid used 14.4% fewer searches
and 8.2% fewer processed tokens than brief-base , with 8.5% lower
3Training details and reproducibility limitations are documented at https://github.com/
rmit-ir/trec-rag-26/blob/main/docs/embedding-training.md.
4See https://github.com/castorini/RAGDoll.
4

RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication TREC 2026, November 2026, Gaithersburg, MD, USA
estimated generation cost ($2.156 vs. $2.356 per topic). However, its
15-topic arena win rate against a Luna-based sol-research-first
decreased from 40.0% to 33.3%, using a Terra judge across both pre-
sentation orders (30 battles).
Disabling±1adjacent-chunk expansion reduced pointwise scores
by 0.067 with Sol and 0.133 with Qwen3. Among development-time
runs of the corresponding architectures on the 15-topic benchmark,
brief-base (Sol) and sol-research-first (Luna) tied at 2.267/3,
ahead of sol-default (Luna, 2.000) and facets (1.800). In a pair-
wise arena, brief-base lost to sol-research-first by 12 wins
to 18 across 30 battles (15 topics evaluated in both orders). On a
separate 30-topic development rubric, the sol-research-first
configuration scored higher than sol-default (0.704 vs. 0.651).
Across the report’s 30 proprietary-model cells,References and Ci-
tation Qualitywas the weakest dimension (mean 0.167/2, range
0.00–0.67). The sol-research-first development run scored 0.0
on this dimension, although this was based on only nine rubric
criterion instances.
Overall, our internal experiments suggest that writer choice had
a greater impact than most structural changes, while hybrid re-
trieval offered modest quality and efficiency improvements. The
evidence-handling analysis also highlight a trade-off between token
efficiency and citation support, and show that automated prefer-
ence judgments can be sensitive to judge choice and may overlook
citation-support weaknesses. However, these findings are based on
a limited number of topics and should be interpreted cautiously.
They do not necessarily predict performance on the official evalua-
tion, which used different topics and assessment methods. Further
research is needed to understand the generalizability of these re-
sults.
4.3 Open-Weight Analysis and Baseline Arena
On two 15-topic development sets, owa-glm5 scored 1.467/3 and
1.333/3, below the proprietary GPT-5.6 configurations (1.867–2.333).
The largest gaps were in Implicit Criteria ( −0.294) and Synthesis
(−0.205). However, owa-glm5 scored higher on References and Ci-
tations (0.667/2 vs. 0.222 and 0.0), suggesting that its more frequent
citation use may benefit this rubric dimension.
In an attempt to isolate the effects of the writer from retrieval
and orchestration, we replaced the GPT-5.6-Sol writer in the
sol-research-first system with Qwen3-Next-80B-A3B , keeping
the same retriever and evidence collection strategy. Replaying ev-
idence collected by sol-research-first into Qwen3 improved
Synthesis (+0.154) but barely changed Implicit Criteria ( +0.006),
yielding an overall score of 1.267. Replacing Qwen3 helper mod-
els with GPT-5.6-Luna slightly reduced performance (1.0–1.067
vs. 1.20). Our internal experiments suggest that writer synthesis,
rather than evidence collection or helper model quality alone, seem
to be an important bottleneck.
To estimate performance in the official evaluation, we com-
pared six completed runs against the organizer’s Sol agentic-search
BM25 baseline on all 119 test topics using automated preferenceTable 3: Internal both-order RAGDoll arena comparisons
against the organizer’s shipped agentic BM25 baseline over
all 119 narratives (GPT-5.6-Luna judge, seed 13; not official
TREC judgments). Win rate is (𝑊+ 0.5𝑇)/119. owa-glm5 was
finalized after this sweep.
System W L T Win rate
sol-research-first108 10 1 91.2%
sol-default107 11 1 90.3%
brief-hybrid101 18 0 84.9%
brief-base99 19 1 83.6%
facets13 105 1 11.3%
facetrag0 119 0 0.0%
owa-glm5— — — —
judgments. Unlike the development topics, these had no avail-
able evaluation labels or rubric, , so we used the arena as a pre-
submission sanity check rather than for system selection or op-
timization. owa-glm5 was finalized after this sweep and was not
included. Table 3 summarizes the results.
In the arena against the organizer’s baseline results all four Sol
runs achieved high win rates (83.6–91.2%), while facets (11.3%) and
facetrag (0.0%) performed poorly. Internal pairwise comparisons
revealed observable position bias. sol-research-first narrowly
ledsol-default (67 wins, 51 losses, 1 tie), but its non-tie win rate
changed from 37.7% when presented as A to 76.7% as B. Similarly,
brief-hybrid andbrief-base were nearly tied overall (58 wins,
57 losses, 4 ties), despite win rates changing from 34.5% to 63.9%
between display orders.
Compared with official NIST coverage results in Table 4, the
internal arena broadly distinguished stronger and weaker systems
but did not reliably reproduce their relative rankings. Across the
six common systems, the arena’s ranking agreed only moderately
with the official coverage rankings (Kendall’s 𝜏=0.33-0.60; 3-5 of
15 system pairs ordered differently; none significant at 𝑛= 6),
with the highest agreement for B1 all-items and B2 automatic
nuggets ( 𝜏= 0.60). For example, sol-default nearly matched
sol-research-first internally despite substantially lower B1 vi-
tal coverage (0.790 vs. 0.892). Together with the observed judge
and position biases, these discrepancies highlight limitations of
automated evaluation and motivate initiatives such as TREC Auto-
Judge [ 11], which investigate the reliability of automated assess-
ments.
5 Official TREC Evaluation
5.1 Rubric and Nugget Coverage
Table 4 reports official vital_strict andall_strict coverage [ 17,
18,23,26] across Batch-1 human-intuition rubrics ( 𝑛=17), Batch-2
manually authored nuggets ( 𝑛= 25), and Batch-2 automatically
generated RAGDoll AutoNuggetizer nuggets ( 𝑛=25). In all three
batches, item support was assigned automatically using GPT-6.1
Solat medium reasoning. Strict scoring gives credit only when
an item is fully supported; partial support receives zero credit.
vital_strict considers vital items only, whereas all_strict in-
cludes vital and “okay” items with equal weight.
5

TREC 2026, November 2026, Gaithersburg, MD, USA Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, and Shuai Wang
Table 4: Official NIST TREC 2026 RAG coverage ( vital_strict /all_strict ) and numbers of narratives scoring at or above the
global track median ( ≥Med, V/A) across our seven submitted runs. Underlining marks our highest run per metric; ★denotes
a tie with the global track maximum before rounding. Two B2 manual narratives ( rag2026-89 ,rag2026-114 ) contain no vital
nuggets, cappingvital_strictat23/25=0.920.
B1 Human Rubric (𝑛=17) B2 Manual Nuggets (𝑛=25) B2 Auto Nuggets (𝑛=25)
Type System Vital / All≥Med (V/A) Vital / All≥Med (V/A) Vital / All≥Med (V/A)
Proprietarybrief-hybrid0.892★/ 0.909 17 / 17 0.691 / 0.723 24 / 25 0.644 / 0.565 25 / 25
sol-research-first0.892★/ 0.887 17 / 17 0.684 / 0.702 25 / 24 0.649 / 0.560 25 / 25
facets0.831 / 0.841 17 / 16 0.627 / 0.642 23 / 23 0.390 / 0.333 25 / 24
brief-base0.824 / 0.862 17 / 16 0.691 / 0.721 24 / 25 0.648 / 0.555 25 / 25
sol-default0.790 / 0.861 17 / 17 0.679 / 0.705 25 / 25 0.630 / 0.563 25 / 25
Open-weightowa-glm50.853 / 0.821 17 / 16 0.550 / 0.605 23 / 23 0.288 / 0.243 22 / 23
facetrag0.600 / 0.622 11 / 10 0.460 / 0.481 13 / 10 0.139 / 0.110 12 / 11
TREC Track Max 0.892 / 0.911—0.729 / 0.780—0.666 / 0.584—
TREC Track Median 0.716 / 0.733—0.488 / 0.556—0.155 / 0.128—
TREC Track Min 0.039 / 0.043—0.076 / 0.070—0.002 / 0.002—
Our strongest coverage results came from brief-hybrid and
sol-research-first . Both tied the global track maximum on
Batch-1 vital_strict at 0.892, while brief-hybrid achieved our
highest all_strict score in all three batches (0.909, 0.723, and
0.565). Six of seven submitted runs exceeded the global track me-
dian across all three batches.
The nominal rankings should be interpreted with a grain of salt.
We conducted pairwise comparisons using Tukey’s HSD test [ 24]
on the per-topic coverage scores and found no significant differ-
ences among the four leading proprietary runs, brief-hybrid ,
brief-base ,sol-default , and sol-research-first on any of
the six coverage metrics. facetrag was consistently the weakest
system, while the separation between the leading runs, facets ,
andowa-glm5 was largest under Batch-2 automatic nuggets. The
official coverage results therefore distinguish the strongest and
weakest systems more clearly than they distinguish among the
leading proprietary configurations.
We also observed a positive association between answer length
and coverage within topics, increasing from a mean Spearman cor-
relation of 0.42 in Batch 1 to 0.72 under Batch-2 automatic nuggets.
This relationship is descriptive: answer length is confounded with
system design and does not imply that longer answers alone cause
higher coverage, but it does suggest that the fully automated Batch-
2 evaluation may favor longer responses and does not penalize
verbosity, unlike typical human evaluation.
5.2 Human Side-by-Side Preference
The organizers also performed human side-by-side evaluation for
our two priority runs, owa-glm5 andsol-research-first , fitting
separate Bradley–Terry (BT) ratings for each batch [3] (Figure 2).
In Batch 1, assessors strongly preferred owa-glm5 , with a BT
rating of 1212.8 compared with 1062.7 for sol-research-first .
The difference was also robust when topics were treated as the unit
of analysis: 13 of 17 topics favored owa-glm5 , using a paired sign-
flip test on per-topic battle-score differences found the difference
to be significant with 𝑝=0.006. This contrasts with the coverage
results, where sol-research-first scored higher on both Batch-1
sol-research-firstowa-glm5Med 1048.7
Max 1258.2Batch 1: Holistic Rubric SxS (/u1D45B=17)
1000 1050 1100 1150 1200 1250
Bradley–Terry rating (95% CI)sol-research-firstowa-glm5Med 1024.8 Max 1261.7Batch 2: Nugget-Guided SxS (/u1D45B=25)Track Median Track Max Med–Max spanFigure 2: Official human side-by-side Bradley–Terry ratings
with published 95% confidence intervals and track medians
and maxima. Batch 1 ( 𝑛= 17) and Batch 2 ( 𝑛= 25) were
fitted separately and used different topic sets and assessment
protocols.
strict metrics, showing that rubric coverage and holistic human
preference can rank the same systems differently. The direction
reversed in Batch 2, where sol-research-first achieved a higher
BT rating (1124.7 vs. 1049.0). However, the topic-level comparison
was inconclusive ( 𝑝= 0.110), making this result less robust than
the Batch-1 preference. The batches also differed in both topics
and assessment protocol: Batch-1 assessors compared responses
holistically, whereas Batch-2 assessors first constructed document-
grounded nuggets before performing their comparisons.
Overall, the human judgments provide a complementary signal
to coverage. In particular, owa-glm5 ’s strong Batch-1 preference
despite lower coverage suggests that characteristics not captured
by strict rubric recall influenced assessor preferences. Section 6
examines readability and response style as possible contributors.
6

RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication TREC 2026, November 2026, Gaithersburg, MD, USA
9597576811132917596150163041457179
Narrative ID ( rag2026-* )-0.8-0.6-0.4-0.20.0+0.2+0.4+0.6all strict /uni0394vs. Track Median
68:sol-rf+0.50(=Max);
owa0.0, yet 13W/1L SxS
30: Track Max=0.40
(all 7 runs tied at ceiling)Batch 1: Human-Intuition Rubric (/u1D45B=17)
116 73796118 541865854826114 19 24644 47815893288109 21110
Narrative ID ( rag2026-* )
65:owa-glm5−0.79
(21-word review collapse; 0W/8L SxS)48: Med=0.10;
brief-hybrid+0.20Batch 2: Manual Shard Nuggets (/u1D45B=25)
Track Max headroom above Median brief-hybrid sol-research-first owa-glm5 facetrag
Figure 3: Per-topic official all_strict coverage difference ( Δ) relative to the TREC 2026 RAG track median across Batch 1
(𝑛= 17, left) and Batch 2 manual nuggets ( 𝑛= 25, right), sorted within each panel by brief-hybridΔ . Shaded bars show the
difference between the track maximum and median.
5.3 Per-Topic Divergence
Figure 3 shows substantial per-topic variation and illustrates why
aggregate coverage alone does not fully characterize system behav-
ior. We highlight two cases.
Review-Pass Failure.On Batch-2 topic 65 (hiring-AI com-
pliance), brief-hybrid andsol-research-first both achieved
1.000/1.000 coverage, whereas owa-glm5 scored 0.0/0.0 and lost all
eight side-by-side comparisons. Its trace shows 12 searches and five
evidence commits followed by a 922-word draft. During review, the
model was asked to add a citation to the opening sentence while
preserving the report, but returned only a corrected 21-word sen-
tence, which the harness accepted as the complete revision. This
was therefore a post-draft revision failure rather than a failure to
retrieve or synthesize evidence initially.
Coverage–Preference Divergence.On Batch-1 topic 68 (school
safety), sol-research-first achieved 0.750/0.833 coverage, tying
the global maximum on all_strict , but recorded 6W/2T/6L in
human comparisons.owa-glm5, despite substantially lower cover-
age (0.250/0.333), recorded 13W/0T/1L. This example emphasizes
the surprising Batch-1 result: higher strict rubric coverage did not
necessarily imply stronger holistic human preference.
6 Discussion
Batch 1 reveals a clear divergence between information cover-
age and holistic human preference. Although owa-glm5 covered
fewer rubric items than sol-research-first (Table 4), assessors
strongly preferred it in the side-by-side evaluation (Figure 2). This
preference was robust at the topic level, with 13 of 17 topics fa-
voring owa-glm5 (sign-flip 𝑝= 0.006), despite its lower Batch-1
coverage. The result is also notable from an efficiency perspective:
estimated generation cost was $0.524 per topic for owa-glm5 , com-
pared with $2.823 for sol-research-first . The per-topic example
in Section 5.3 further illustrates that higher strict coverage does
not necessarily imply stronger human preference.
To examine whether reading burden might contribute to this
difference, we measured Flesch–Kincaid Grade Level [ 15] on theTable 5: Report characteristics on the 17 matched Batch-1
topics (means, except topic counts).
Measureowa-glm5 sol-research-first
Words 805.3 951.2
Unique references 16.6 28.9
Flesch–Kincaid Grade Level↓15.7 22.2
Topics with shorter report 15/17 2/17
17 matched Batch-1 responses using textstat (US English).5We
excluded citation IDs, reference lists, and Markdown markers and
used the library’s sentence segmentation. owa-glm5 was easier to
read on all 17 topics, with a mean Flesch–Kincaid grade of 15.7
compared with 22.2 for sol-research-first (sign test 𝑝< 0.001;
Table 5). It also used shorter sentences on average (21.8 vs. 34.9
words per sentence). However, the per-topic readability difference
was not associated with the per-topic preference margin (Spearman
𝜌=+ 0.27,𝑝= 0.30,𝑛= 17), and no readability measure was
significant after correction for multiple comparisons. Answer length
was likewise not associated with the Batch-1 preference difference
(𝜌=− 0.15,𝑝=0.56). Because owa-glm5 was easier to read on every
topic, these data cannot separate readability from other systematic
differences between the systems, such as response structure, tone,
or handling of uncertainty. Readability therefore remains a plausible
contributor rather than an explanation of the observed preference.
The preference direction reversed in Batch 2, although the topic-
level difference was not statistically significant. Both the topics and
assessment protocol changed between batches, so this reversal can-
not be attributed specifically to nugget-guided assessment. More
broadly, the divergence between coverage and human preference,
together with the judge sensitivity observed in our internal evalua-
tion (Section 4.1), highlights the importance of using complemen-
tary evaluation signals rather than relying on a single automated
or human metric.
5Version 0.7.12: https://github.com/textstat/textstat.
7

TREC 2026, November 2026, Gaithersburg, MD, USA Oleg Zendel, Kun Ran, Chenglong Ma, Nguyen Khoi Vo, and Shuai Wang
7 Conclusion
RQ1.Open-weight agents were competitive on some measures
but did not consistently match the coverage of the leading propri-
etary systems. owa-glm5 exceeded the global track median across
all official coverage evaluations and was strongly preferred over
sol-research-first in Batch-1 human side-by-side evaluation,
despite lower rubric coverage (Table 4; Figure 2). Its estimated
generation cost was $0.524 per topic, compared with $2.823 for
sol-research-first (Table 2). However, it trailed the leading
proprietary runs on Batch-2 nugget coverage, and diagnostics iden-
tified synthesis and revision weaknesses, including the review-pass
failure in Figure 3. Its consistently lower Batch-1 Flesch–Kincaid
scores suggest lower reading burden as a possible contributor to
human preference, although our analysis did not establish such an
effect (Section 6).
RQ2.Writer choice had the largest effect among the measured
development ablations, while the benefit of adjacent-chunk ex-
pansion varied by writer (Section 4.2). Hybrid RRF provided the
clearest efficiency improvement: brief-hybrid used 14.4% fewer
searches, 8.2% fewer processed tokens, and had 8.5% lower estimated
generation cost than brief-base . It also tied the global Batch-1
vital_strict maximum and achieved our highest all_strict
score in all three official collections (Tables 2 and 4), although the
leading proprietary systems were not statistically distinguishable
on official coverage.
RQ3.More elaborate orchestration did not provide a consis-
tent advantage across the systems we tested. sol-research-first
improved over sol-default in our development evaluation and
achieved higher Batch-1 coverage, but processed 81% more tokens
with little difference in Batch-2 coverage. facetrag , despite its
multi-stage architecture and high retrieval volume, remained the
weakest system on official coverage. Because these systems differ
in models, retrieval, and prompts as well as orchestration, these
comparisons do not isolate the effect of orchestration alone.
Overall, our results show that there is no single best design for
agentic RAG: different architectures offer different trade-offs in
coverage, human preference, robustness, and efficiency. The same
applies to evaluation. Our internal automated judgments, official
rubric and nugget coverage, and human side-by-side preferences
sometimes produced different system rankings, emphasizing that
the choice of evaluation protocol is itself consequential. Developing
a strong system therefore requires not only selecting appropriate
models, retrieval, and orchestration, but also selecting evaluation
methods that reflect the intended qualities and use case of the sys-
tem. Rather than relying on a single metric or evaluation paradigm,
future work should combine complementary signals and validate
that the chosen evaluation meaningfully captures the behavior it is
intended to measure.
Acknowledgments
We thank the TREC 2026 RAG track organizers for their efforts in
running the track. This work was supported by the ARC Centre of
Excellence for Automated Decision-Making and Society (ADM+S,
CE200100005) and was undertaken with the assistance of computing
resources from RACE (RMIT AWS Cloud Supercomputing).References
[1]Nuha Abu Onq, Chenglong Ma, Mark Sanderson, and Falk Scholer. 2026. Under-
standing and Modeling Heterogeneous Search Behavior. InProceedings of the 49th
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’26). 4–16. doi:10.1145/3805712.3809618
[2]Mohammad Kalim Akram, Saba Sturua, Nastia Havriushenko, Quentin Herreros,
Michael Günther, Maximilian Werk, and Han Xiao. 2026. jina-embeddings-v5-text:
Compact and Robust Text Embedding Models using Task-Targeted Distillation.
InProceedings of the 49th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’26). 4454–4458. doi:10.1145/3805712.
3808455
[3]Ralph Allan Bradley and Milton E. Terry. 1952. Rank Analysis of Incomplete
Block Designs: I. The Method of Paired Comparisons.Biometrika39, 3/4 (1952),
324–345. doi:10.2307/2334029
[4]Jaime Carbonell and Jade Goldstein. 1998. The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries. InProceedings of
the 21st Annual International ACM SIGIR Conference on Research and Development
in Information Retrieval. ACM, New York, NY, USA, 335–336. doi:10.1145/290941.
291025
[5]Haodong Chen, Shuai Wang, Yu Yin, Shengyao Zhuang, Guido Zuccon, and
Teerapong Leelanupab. 2026. ITER: Interaction-aware retrieval for agentic search.
arXiv preprint arXiv:2608.27912(2026).
[6]Charles L. A. Clarke and Gordon V. Cormack. 2000. Shortest-Substring Retrieval
and Ranking.ACM Transactions on Information Systems18, 1 (2000), 44–78.
doi:10.1145/333135.333137
[7]Charles L. A. Clarke and Mark D. Smucker. 2026. Boolean Queries Are All You
Need? arXiv:2607.11362 https://arxiv.org/abs/2607.11362
[8]Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
InProceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’09). ACM, New York, NY, USA,
758–759. doi:10.1145/1571941.1572114
[9]Shizhe Diao, Yu Yang, Yonggan Fu, Xin Dong, Dan Su, Markus Kliegl, Zijia Chen,
Peter Belcak, Yoshi Suhara, Hongxu Yin, Mostofa Patwary, Yingyan Lin, Jan Kautz,
and Pavlo Molchanov. 2025. Nemotron-CLIMB: CLustering-based Iterative Data
Mixture Bootstrapping for Language Model Pre-training. InAdvances in Neural
Information Processing Systems, Vol. 38. 29096–29124. doi:10.52202/085713-0865
[10] Laura Dietz, Oleg Zendel, Peter Bailey, Charles L. A. Clarke, Ellese Cotterill, Jeff
Dalton, Faegheh Hasibi, Mark Sanderson, and Nick Craswell. 2025. Principles and
Guidelines for the Use of LLM Judges. InProceedings of the 2025 International ACM
SIGIR Conference on Innovative Concepts and Theories in Information Retrieval
(ICTIR) (ICTIR ’25). 218–229. doi:10.1145/3731120.3744588
[11] Naghmeh Farzi, Tim Hagen, Eugene Yang, Maik Fröbe, Ronak Pradeep, Hos-
sein A. Rahmani, Xi Wang, Oleg Zendel, Martin Potthast, and Laura Dietz. 2026.
Auto-Judge: A Cross-Task Benchmark for Comparing LLM Judges for Citation-
Grounded RAG Systems. InProceedings of the 49th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR ’26).
3159–3166. doi:10.1145/3805712.3808601
[12] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
Association for Computational Linguistics, Toronto, Canada, 1762–1777. doi:10.
18653/v1/2023.acl-long.99
[13] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Tevatron: An
Efficient and Flexible Toolkit for Neural Retrieval. InProceedings of the 46th
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’23). ACM, New York, NY, USA, 3120–3124. doi:10.1145/3539618.
3591805
[14] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar
Krishnawamy, and Rohan Kadekodi. 2019. DiskANN: Fast Accurate Billion-point
Nearest Neighbor Search on a Single Node. InAdvances in Neural Information Pro-
cessing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché Buc, E. Fox,
and R. Garnett (Eds.), Vol. 32. Curran Associates, Inc. https://proceedings.neurips.
cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf
[15] J. Peter Kincaid, Robert P. Fishburne, Jr., Richard L. Rogers, and Brad S. Chissom.
1975.Derivation of New Readability Formulas (Automated Readability Index, Fog
Count and Flesch Reading Ease Formula) for Navy Enlisted Personnel. Technical Re-
port Research Branch Report 8-75. Chief of Naval Technical Training, Millington,
TN. https://stars.library.ucf.edu/istlibrary/56/
[16] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep,
and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible In-
formation Retrieval Research with Sparse and Dense Representations. InPro-
ceedings of the 44th International ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval (SIGIR ’21). ACM, New York, NY, USA, 2356–2362.
doi:10.1145/3404835.3463238
[17] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick
Craswell, and Jimmy Lin. 2024. Initial Nugget Evaluation Results for the
8

RMIT-ADM+S at TREC RAG 2026: Research Agents, Retriever Ablations, and Open-Weight Replication TREC 2026, November 2026, Gaithersburg, MD, USA
TREC 2024 RAG Track with the AutoNuggetizer Framework. arXiv:2411.09607
https://arxiv.org/abs/2411.09607
[18] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick
Craswell, Ian Soboroff, Hoa Trang Dang, and Jimmy Lin. 2025. The Great Nugget
Recall: Automating Fact Extraction and RAG Evaluation with Large Language
Models. InProceedings of the 48th International ACM SIGIR Conference on Research
and Development in Information Retrieval (SIGIR ’25). ACM, New York, NY, USA,
180–190. doi:10.1145/3726302.3730090
[19] Kun Ran, Marwah Alaofi, Danula Hettiachchi, Chenglong Ma, Khoi Nguyen Dinh
Anh, Khoi Vo Nguyen, Sachin Pathiyan Cherumanal, Lida Rashidi, Falk Scholer,
Damiano Spina, et al .2026. RMIT-ADM+ S at the MMU-RAG NeurIPS 2025
Competition.arXiv preprint arXiv:2602.20735(2026).
[20] Kun Ran, Shuoqi Sun, Khoi Nguyen Dinh Anh, Damiano Spina, and Oleg Zendel.
2025. RMIT-ADM+ S at the SIGIR 2025 LiveRAG Challenge.arXiv preprint
arXiv:2506.14516(2025).
[21] David Rau, Hervé Déjean, Nadezhda Chirkova, Thibault Formal, Shuai Wang,
Stéphane Clinchant, and Vassilina Nikoulina. 2024. BERGEN: A Benchmarking Li-
brary for Retrieval-Augmented Generation. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024. 7640–7663. doi:10.18653/v1/2024.findings-
emnlp.449
[22] Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang,
Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu,
Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad
Kenstler, and Bing Liu. 2025. ResearchRubrics: A Benchmark of Prompts
and Rubrics For Evaluating Deep Research Agents. arXiv:2511.07685 [cs.AI]
https://arxiv.org/abs/2511.07685
[23] TREC RAG Track Organizers. 2026. TREC Retrieval-Augmented Generation
(RAG) Track. https://trec-rag.github.io/. Track website; accessed 2026-11-10.
[24] John W Tukey. 1949. Comparing individual means in the analysis of variance.
Biometrics5, 2 (1949), 99–114. doi:10.2307/3001913[25] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning
with Contrastive Predictive Coding. arXiv:1807.03748 https://arxiv.org/abs/1807.
03748
[26] Ellen M. Voorhees. 2003. Overview of the TREC 2003 Question Answering
Track. InProceedings of the Twelfth Text REtrieval Conference (TREC 2003). 54–68.
https://trec.nist.gov/pubs/trec12/papers/QA.OVERVIEW.pdf
[27] Peilin Yang, Hui Fang, and Jimmy Lin. 2017. Anserini: Enabling the Use of Lucene
for Information Retrieval Research. InProceedings of the 40th International ACM
SIGIR Conference on Research and Development in Information Retrieval (SIGIR
’17). ACM, New York, NY, USA, 1253–1256. doi:10.1145/3077136.3080721
[28] Oleg Zendel, Sara Fahad Dawood Al Lawati, Lida Rashidi, Falk Scholer, and Mark
Sanderson. 2025. A Comparative Analysis of Linguistic and Retrieval Diversity
in LLM-Generated Search Queries. InProceedings of the 34th ACM International
Conference on Information and Knowledge Management (CIKM ’25). 4014–4023.
doi:10.1145/3746252.3761382
[29] Oleg Zendel, J. Shane Culpepper, Falk Scholer, and Paul Thomas. 2024. Enhancing
Human Annotation: Leveraging Large Language Models and Efficient Batch
Processing. InProceedings of the 2024 Conference on Human Information Interaction
and Retrieval (CHIIR ’24). 340–345. doi:10.1145/3627508.3638322
[30] Yuqi Zhou, Sunhao Dai, Changle Qu, Liang Pang, Jun Xu, and Ji-Rong Wen.
2026. Learning to Retrieve from Agent Trajectories. InProceedings of the 49th
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’26). 2544–2555. doi:10.1145/3805712.3809675
[31] Yongjie Zhou, Shuai Wang, Bevan Koopman, and Guido Zuccon. 2026. Beyond
Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document
Chunking Strategies for Information Retrieval. InProceedings of the 49th In-
ternational ACM SIGIR Conference on Research and Development in Information
Retrieval. ACM, 3081–3091. doi:10.1145/3805712.3808575
9
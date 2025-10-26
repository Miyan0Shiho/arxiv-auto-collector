# Hierarchical Sequence Iteration for Heterogeneous Question Answering

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim

**Published**: 2025-10-23 12:48:18

**PDF URL**: [http://arxiv.org/pdf/2510.20505v1](http://arxiv.org/pdf/2510.20505v1)

## Abstract
Retrieval-augmented generation (RAG) remains brittle on multi-step questions
and heterogeneous evidence sources, trading accuracy against latency and
token/tool budgets. This paper introducesHierarchical Sequence (HSEQ) Iteration
for Heterogeneous Question Answering, a unified framework that (i) linearize
documents, tables, and knowledge graphs into a reversible hierarchical sequence
with lightweight structural tags, and (ii) perform structure-aware iteration to
collect just-enough evidence before answer synthesis. A Head Agent provides
guidance that leads retrieval, while an Iteration Agent selects and expands
HSeq via structure-respecting actions (e.g., parent/child hops, table
row/column neighbors, KG relations); Finally the head agent composes
canonicalized evidence to genearte the final answer, with an optional
refinement loop to resolve detected contradictions. Experiments on HotpotQA
(text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1
gains over strong single-pass, multi-hop, and agentic RAG baselines with high
efficiency. Besides, HSEQ exhibits three key advantages: (1) a format-agnostic
unification that enables a single policy to operate across text, tables, and
KGs without per-dataset specialization; (2) guided, budget-aware iteration that
reduces unnecessary hops, tool calls, and tokens while preserving accuracy; and
(3) evidence canonicalization for reliable QA, improving answers consistency
and auditability.

## Full Text


<!-- PDF content starts -->

Paper under review
HIERARCHICALSEQUENCEITERATION FORHETERO-
GENEOUSQUESTIONANSWERING
Ruiyi Yang
University of New South Wales
Sydney, NSW, Australia
ruiyi.yang@student.unsw.edu.auHao Xue
University of New South Wales
Sydney, NSW, Australia
hao.xue1@unsw.edu.au
Imran Razzak
Mohamed Bin Zayed University of Artificial Intelligence
Abu Dhabi, UAE
imran.razzak@mbzuai.ac.aeHakim Hacid
Technology Innovation Institute
Abu Dhabi, UAE
hakim.hacid@tii.ae
Flora D. Salim
University of New South Wales
Sydney, NSW, Australia
flora.salim@unsw.edu.au
ABSTRACT
Retrieval-augmented generation (RAG) remains brittle on multi-step questions
and heterogeneous evidence sources, trading accuracy against latency and to-
ken/tool budgets. This paper introducesHierarchical Sequence (HSEQ) Iter-
ationforHeterogeneous Question Answering, a unified framework that (i) lin-
earize documents, tables, and knowledge graphs into a reversible hierarchical se-
quence with lightweight structural tags, and (ii) perform structure-aware iteration
to collect just-enough evidence before answer synthesis. A Head Agent provides
guidance that leads retrieval, while an Iteration Agent selects and expands HSeq
via structure-respecting actions (e.g., parent/child hops, table row/column neigh-
bors, KG relations); Finally the head agent composes canonicalized evidence to
genearte the final answer, with an optional refinement loop to resolve detected con-
tradictions. Experiments on HotpotQA (text), HybridQA/TAT-QA (table+text),
and MetaQA (KG) show consistent EM/F1 gains over strong single-pass, multi-
hop, and agentic RAG baselines with high efficiency. Besides, HSEQ exhibits
three key advantages: (1) aformat-agnostic unificationthat enables a single pol-
icy to operate across text, tables, and KGs without per-dataset specialization; (2)
guided, budget-aware iterationthat reduces unnecessary hops, tool calls, and
tokens while preserving accuracy; and (3)evidence canonicalization for reliable
QA, improving answers consistency and auditability.
1 INTRODUCTION
Large language models (LLMs), such as ChatGPT (Achiam et al., 2023), LLaMA (Dubey et al.,
2024), Falcon (Zuo et al., 2025), have been increasingly relying on retrieval-augmented generation
(RAG) to ground answers in external evidence. With reliable supplementary knowledge offered
factual errors are reduced, especially in domain-specific questions, leading to higher accuracy and
fewer hallucinations (Zhu et al., 2021b; Gao et al., 2023; Zhao et al., 2024). Yet state-of-the-art
pipelines, remain brittle on multi-step questions and heterogeneous sources, and still struggle to
cope with the following challenges:
C1:Coverage in Single-pass Retrievers: Single-pass pipelines (retrieve-kthen generate) (Luo
et al., 2023; Glass et al., 2022) focus on isolated retrieval and generation tasks. Although they
can be setup and achieve data retrieval quickly, they struggle to trace complete evidence chains:
1arXiv:2510.20505v1  [cs.CL]  23 Oct 2025

Paper under review
dense retrievers, typically trained for pointwise recall and re-ranking, often lack path coverage;
chunking heuristics fragment long documents and break discourse; long-context prompting shifts
budget toward tokens irrelevant to the final answer and provides no explicitsufficiencysignal.
C2:Uncontrolled iteration and latency in multi-agent systems: With multi-agent collaboration
and reasoning, agentic systems (Liu et al., 2025; Yang et al., 2025; Chen et al., 2025) easily explode
the search space and can achieve multi-step reasoning. However they may fall with branchy plans,
repeated web/file calls, and verbose chain-of-thought prompts, yielding unpredictable token/tool
costs and latency; termination is often heuristic, leading to premature answers or extra wasted loops
with budgets decoupled from theevidence actually inspected(Singh et al., 2025).
C3:Heterogeneity across formats: Free text, relational tables, and KGs typically require distinct
indices, retrievers, prompt styles, and controller logic, preventing policy reuse and complicating
training and deployment. Although existing heterogeneous RAG systems (Yu, 2022; Christmann
& Weikum, 2024) are available to deal with multiple formats of data, they may still face issues in
either weak alignment across representations or lossy and non-reversible serialization that obscures
provenance and blocks faithful reconstruction.
Hierarchical Sequence Iteration (HSEQ)for Heterogeneous Question Answering introduces a re-
versiblehierarchical sequenceinterface that linearizes documents, tables, and KGs into a sequence
of typed segments with lightweight structure (e.g., parent/child locality, offsets or coordinates, min-
imal schema/time tags). An iteration policy operates on this unified substrate using short, budgeted
steps: at each step it selects a few promising segments and predicts whether the accumulated set
is sufficient to answer. A conciseguidanceplan—produced by a lightweight planner or a heuristic
template—acts as a soft prior over which regions to probe first and when to stop. Once sufficiency is
predicted, the selected segments are canonicalized into a compact, provenance-preserving package
consumed by a head module to produce the final answer; an optional verifier can trigger a brief
refinement if contradictions are detected.
To address above issues, this paper introducesHSEQ, aHierarchical Sequence Iteration Sys-
temthat first recasts heterogeneous knowledge source into asingle, LLM-native interface, then
turning retrieval into aguided, budget-aware iterative process. The reversible HSEQ interface lin-
earizes documents, tables, and KGs into a sequence of typed segments with lightweight structure
(e.g., parent/child locality, offsets or coordinates, minimal schema/time tags). An iteration policy
operates on this unified substrate using short, budgeted steps: at each step it selects a few promis-
ing segments and predicts whether the accumulated set is sufficient to answer. A conciseguidance
plan—produced by a lightweight planner or a heuristic template—acts as a soft prior over which
regions to probe first and when to stop. Once sufficiency is predicted, the selected segments are
canonicalized into a compact, provenance-preserving package consumed by a head module to pro-
duce the final answer; an optional verifier can trigger a brief refinement if contradictions are detected.
pecifically, ourkey contributionsare as followed:
•Unified, reversible interface.A hierarchical sequence representation that standardizes
text, tables, and KGs with lightweight structure and provenance, enabling a single con-
troller to operate across formats.
•Guided, budget-aware iteration.A learned selection policy with an explicit sufficiency
signal that concentrates computation onevidence actually inspected, delivering predictable
latency under token/tool budgets.
•Canonicalized evidence for reliable QA.A compact, provenance-preserving evi-
dence package that improves answer synthesis and auditability, and supports optional
contradiction-driven refinement.
2 HSEQ: A MULTI-AGENTHETEROGENEOUSQUESTIONANSWERING
FRAMEWORK
2.1 BACKGROUND ANDSETUP
Heterogeneous QA with budgets.Given a natural-language queryqand a heterogeneous corpus
D={(x j, mj)}N
j=1with modalitym j∈ {text,table,kg}, the goal is to produce an answer
2

Paper under review
y∈ Yand optional supporting evidenceE⊆Dwhile satisfying resource budgetsB(tokens, tool
calls, latency, steps). LetE⋆denote aminimally sufficientevidence set forqinDunder a fixed
answerer.
From retrieval to guided iteration.We recast retrieval as a short sequence of structure-aware se-
lections under an explicit sufficiency criterion. A modality-aware adapterτconvertsDinto a single
hierarchical sequenceS h=τ(D). A learned iteration policyπ θinteracts with(q, S h)to accumu-
late a compact evidence setM⋆under budgetsB, guided by a concise plang. A canonicalizerκ
packagesM⋆for a head moduleH, which produces the final answer. This preserves the familiar
RAG workflow while adding a principled stopping signal and a unified interface across modalities.
Figure 1: HSEQ overview. (i) HSEQ-A linearizes heterogeneous sources intoS hwith level
tags, parent pointers, and standardized metadata; (ii) HSEQ-I iterates over a windowed stream
of segments under budgets, guided byg, and queriesΦfor sufficiency; (iii)κcompactsM tinto
provenance-preserving evidence; (iv) HSEQ-H produces the final answer and optionally triggers a
brief refinement if inconsistencies are detected.
2.2 HSEQ ARCHITECTURE
The proposed system couples a unifiedhierarchical sequence(HSEQ)representation with an iter-
ation policy and a head moduleHfor answer synthesis. Letqdenote a user query andDa het-
erogeneous corpus. A modality-aware adapterτconvertsDinto a single hierarchical sequence
Sh=τ(D). An iteration moduleπ θoperates on(q, S h)and maintains an evolving evidence setM t.
The final evidenceM⋆is then canonicalized byκand passed toHfor final answer generation. The
end-to-end mapping is summarized as
F= 
τ, πθ,Φ, κ,H
, F(q, D) =H(q, κ(M⋆)),(1)
Specifically, during iteration moduleπ θselecting and expanding segments onS h, the budget-aware
sufficiency criterionΦand the budget stateB t(tokens, tool calls, steps) functioned inside the module
to decide when the accumulated evidence is adequate for answering as well as triggering potential
early stopping.
Sh=τ(D), M⋆=π θ 
q, Sh; Φ, B
.(2)
After the iteration,κmaps raw segmentsM⋆to a normalized evidence package consumable byH.
The same policyπ θis shared across modalities due to the common interface ofS h.
Generally, to achieve iteration through an unified data structure building from heterogeneous data
sources, the HSEQ framework consists of three key modules: HSEQ-Adapter (HSEQ-A), HSEQ-
Iterator (HSEQ-I), and HSEQ-Head (HSEQ-H).
3

Paper under review
2.3 HSEQ-ADAPTER(HSEQ-A)
The HSEQ-Adapter is build to produce unified structure(HSEQS h) that exposes locality (par-
ent/child), alignment (span or coordinates), and lightweight semantics (time, schema, language)
in a modality-agnostic format, while remainingreconstructable. Formally, each itemx jis mapped
by a modality-specific adapterτ mjto a finite set of segmentsτ mj(xj)⊂ Sand then concatenated:
Sh=NG
j=1τmj(xj)∈ S∗,S ∋s= 
id(s), ℓ(s), p(s), c(s), µ(s)
.(3)
Figure 2: HSEQS hconstruction: Different
modalities of data are transformed into unified
sequence by HSEQ-AHereℓ(s)is a level tag matching the raw con-
tent, including sentence, paragraph, table, triplet,
etc., whilep(s)is a parent pointer recording the
roots.c(s)is compact human-readable content,
andµ(s)is metadata with fixed keys to record
content attributes.
The single, modality-aware adapter converts
heterogeneous sources into a common se-
quence ofhierarchical segments. After the
construction, each segment is a lightweight
records= 
id, level, parent, content, metadata
,
wherelevelmarks granularity (e.g.,docu-
ment/paragraph/sentence,table row/table cell,
triplet/subgraph),parentkeeps locality via a
unique container pointer,contentis a compact
human-readable summary (text span, serialized
row, or compact triple), andmetadatastandard-
izes provenance and alignment with fixed keys (e.g.,source id,uri,offsets,schema,time). Segments
are concatenated into a final usableS hin parent-before-child order. This minimal contract enables
structure-aware neighborhoods and budget-aware iteration without inspecting raw files.
2.4 HSEQ-ITERATOR(HSEQ-I)
After HSEQS hare build, the HSEQ-Iteratorπ θcan be used on(q, S h)and maintains an evolving
evidence setM tregarding questionq.
Guidance prior.A short guidanceg=g(q,type)is treated as apriorover iterator actions.gis
generated before each episode to shape exploration onS h. This guidance can come from directly
from head agentH, or from heuristic templates keyed bytype.
Iteration control.LetM t⊆S hdenote the selected evidence at stept,C t⊆S ha candidate
window obeying a budget stateB t, andN(·)the structure-aware neighborhood operators induced
by levels, parents, and coordinates. The HSEQ-I moduleπ θfunctions each step following the policy
πθ(at, st|q, S h, Mt, Ct, g, B t),
which emits an actiona t(e.g., selecting up toksegments fromC tand/or expanding viaN) and a
sufficiency predictions t∈ {0,1}. A deterministic orderingρoverS h(e.g., paragraph≺row≺
sentence≺triplet) defines the stream exposed to the policy. State evolves via a deterministic update
Mt+1=u(M t, at)andC t+1= window(S h, Mt+1, Bt+1, ρ). Termination occurs atτ= min{t:
st= 1}or when the budget is exhausted.
With set window sizeWand step capT max, the algorithm can be described as Alg. 1, where the
Refreshoperator advances the window while removing already selected segments, keeping the per-
step context bounded independent of corpus size.
2.5 HSEQ-HEAD(HSEQ-H).
The HSEQ-Head moduleHcan be used in two parts: 1) Guiding the retrieval for HSEQ-I; and 2)
Generating final conclusion regarding the question.
4

Paper under review
Guidance Generation.Although heuristic templates can be used, regarding an incoming ques-
tion,His available to be called first to analysis the content, generating guidance including: 1) Initial
Retrieval Plan; 2) What information may be needed; 3) Potential conditions to stop.
Answer synthesis and optional refinement.Upon termination at stepτ, the canonicalizerκcon-
vertsM τinto a compact, provenance-preserving evidence package(ids, levels, offsets/coordinates,
short snippets). The head moduleHthen produces the final prediction:
ˆy=H 
q, κ(M τ)
.
An optional verifierξinspectsκ(M τ)for contradictions; if detected, a brief refinement pass (at most
∆additional steps) resumes iteration in Alg. 1 with tightened guidanceg′and reduced budgetB′.
Algorithm 1Guided Iterative Selection under HSEQ-I
Require:questionq, HSEQS h, guidanceg, budgetB, window sizeW, step capT max, minimum
stepsT min, top-k k, orderingρ
1:M 0←∅;C 0←Window(S h;W, B 0, ρ)
2:fort= 1toT maxdo
3:Updatea t
4:(K t, st)at← −π θ(q, g, M t−1, Ct−1;Bt)▷ K t⊆Ct−1,|K t| ≤k
5:M t←M t−1∪Kt
6:C t←Refresh(S h, Mt;W, ρ)
7:ifs t= 1andt≥T minthen
8:break
9:end if
10:UpdateB t
11:end for
12:τ←t;returnκ(M τ)
3 LEARNING TOUSEHSEQWITHOPEN-SOURCELLMS
3.1 FINE-TUNINGHSEQ-I
Although HSEQ-I can be directly called using unfinetuned LLMs, a supervised finetuning is needed
to facilitate the iteration and improve LLM’s understandability with HSEQ.
Figure 3: HSEQ I is build by training ques-
tions from multiple datasets. After guidance
sets are generated, LoRA is applied for fine-
tuningTraining tuples and supervision.Supervision is
organized as tuples(q,type, S h, A⋆). Beside
above mentioned queryqand HSEQS h, an op-
tional question labeltypeis added during training.
To ensure Iteration moduleπ θcan trace sufficient
data while limiting usage, a target trajectoryA⋆=
{(a⋆
t, s⋆
t)}τ⋆
t=1is added which consists of an action
a⋆
tand a binary sufficiency signals⋆
t∈ {0,1}, with
stopping timeτ⋆= min{t:s⋆
t= 1}. When explicit
trajectories are unavailable,weak positivesP⋆⊆Sh
are induced by high-precision matching between the
gold answer (or oracle spans) and segment content,
optionally augmented by lexical overlap withq. A
target action sequence is then synthesized by greed-
ily selecting fromP⋆under the budget (details in
App. A.2).
Policy learning.Let the step state be
(q, S h, Mt, Ct, g, B t). The policy is trained by
supervised risk minimization over trajectories with a parameter-efficient adaptation of a base LLM.
5

Paper under review
With teacher forcing fort < τ⋆,
min
θEτ⋆X
t=1ℓact(πθ(· |state t), a⋆
t)| {z }
action loss+λ ℓ stop(πθ(· |state t), s⋆
t)| {z }
sufficiency loss
,
wherestate t= (q, S h, Mt, Ct, g, B t)andλ >0balances early stopping. WhenA⋆is synthesized
fromP⋆, per-step weights attenuate low-confidence choices to reduce label noise (App. A.2). Dur-
ing experiments, low-Rank Adaptaion is used for model finetuning (Hu et al., 2022) (App. A.3.4)
3.2 HSEQ-H: GUIDANCE ANDANSWERGENERATION
Guidance generation (HSEQ-H).Given a questionq(and optional type tag), HSEQ-H produces
a shortguidancegthat steers the iterator toward promising parts ofS hand specifies a stop condition.
We use two interchangeable modes: (i) a lightweightplannerthat draftsgin 2–4 sentences, and
(ii) aheuristic templatekeyed by coarse question patterns (e.g., number/factoid/yes–no). Eachg
follows the same simple structure:(first-look targets)what to check first (e.g., entities/rows/1–2-hop
neighbors);(expansion rule)how to broaden if evidence is missing (parent/child, row/column, or
relation hops);(stop rule)when to stop (e.g., “answer span/number is explicit and corroborated by
≥1snippet”). Guidance is cached per example and reused at inference; on a cache miss, HSEQ-H
generatesg(planner) or falls back to the template. Importantly,gis asoft prior: the iterator may
override it when stronger signals appear.
Answer generation (HSEQ-H).After the iterator halts with selected evidenceM τ, a canoni-
calizerκcompacts it into a short, provenance-preserving packageZ=κ(M τ)(snippet text plus
ids/levels/source and minimal offsets/coordinates). HSEQ-H then performsevidence-conditioned
answering:
ˆy=H 
q, Z;g
,
where the prompt is kept minimal:answer only(span/number/yes–no),grounded inZ, no chain-
of-thought. When useful, HSEQ-H also returns supporting ids fromZfor auditability. An optional
lightweight check (e.g., “does some snippet inZentailˆy?”) can trigger a one-shotrefinement—the
iterator resumes for a few steps under a tightenedg′—otherwiseˆyis emitted. This setting aligns
answers with explicit evidence, and preserves the modality-agnostic contract of HSEQ.
4 EXPERIMENT
HSEQ are evaluated on multiple QA datasets with a focus on both answer quality and efficiency.
Metrics include Accuracy, F1, alongside efficiency indicators that reflect theevidence actually in-
spected: average iteration steps and end-to-end latency (ms).
4.1 EXPERIMENTSETUP: BENCHMARKS ANDBASELINES
Table 1: Datasets used in our study (modality abbreviations: T=Text,
Tb=Table, KG=Knowledge Graph).
Dataset Modality #Train #Validation #Test
HotpotQA T 90447 7405 7405
TAT-QA Tb + T 13,251 1644 1,663
HybridQA Tb + T 62,682 3,466 3463
MetaQA-2Hop KG 119,986 14,872 14,872
MetaQA-3Hop KG 17,482 14,274 14,274Benchmarks.To eval-
uate HSEQ usage from
different data modalities,
four benchmarks are used
for experiments, stressing
text-only, table–text hy-
brid, and KG reasoning:
HotpotQA (Yang et al.,
2018)(multi-hop reason-
ing over Wikipedia text),
TAT-QA (Zhu et al., 2021a)
(table-centric financial
QA with accompanying paragraphs and numerical operations),HybridQA (Chen et al., 2020)
(Wikipedia tables linked to passages requiring cross-format grounding), andMetaQA (Zhang et al.,
2018)over a Wikidata-style KG (Since 1-hop variants are not emphasized due to triviality, during
experiment only 2-hop and 3-hop questions are used for experiments).
6

Paper under review
Baselines.Three groups are divided for experiments including:
•LLM-only QA.Multiple LLMs are used to directly answers each question from raw inputs
withoutHSEQ (no unified adapter, no iterative controller), under same prompt instruction.
•RAG-based methods.Since HSEQ explores different formats of data sources, RAG mod-
els specializing in separatelyText,TableandKnowledge Graphshave been tested.
Specifically, for HybridQA and TAT-QA,TAT-LLM(Zhu et al., 2024),TableRAG(Yu
et al., 2025),ODYSSEY(Agarwal et al., 2025),TTQA-RS(Bardhan et al., 2024) and
HippoRAG(Jimenez Gutierrez et al., 2024) are chosen for comparison. While for Hot-
potQA and MetaQA-2hop and 3hop, graph-centric RAG systemsGraph-constrained Rea-
soning(GcR)(Luo et al., 2024),Think on Graph (ToG)(Ma et al., 2024) andAdap-
tiveRAG(Jeong et al., 2024) are set as baselines. Each is configured per its recommended
settings for the corresponding modality.
•HSEQ (ours).(i) Thebestiteration–head pair results and (ii) Themedianpair results over
a grid of open-source models are provided. Three ablations are also included in experi-
ments: (i)LLM-only (no HSEQ); (ii)HSEQ w/o SFT(iteration agent not fine-tuned) and
(iii)heuristic-only guidance under fixed template without HSEQ-H.
HSEQ variants.For theiteration agent(HSEQ-I) and thehead agent(HSEQ-H), different LLMs
are finetuned and used, listed as:
HSEQ-I: Falcon-H1-3B-Instruct; Qwen3-4B-Instruct-2507; DeepSeek-R1-Distill-Qwen-7B;
Falcon3-3B-instruct; Falcon3-7B-instruct; Llama-3.2-3B-Instruct.
HSEQ-H: Falcon3-10B-instruct; Falcon-H1-7B-Instruct; Llama-3.1-8B-Instruct; DeepSeek-
R1-Distill-Qwen-7B.
Compatible pairs are swept and final “best” and “median” results across benchmarks are counted,
with hyperparameters settings listed in App. A.3.
4.2 EXPERIMENTRESULT: HOW COMPETITIVE ISHSEQWITH OTHER BASELINES?
Table 2 summarizes answer quality across all datasets. HSEQ consistently improves over both LLM-
only and strong RAG baselines, while using controlled iteration and exposing explicit provenance.
Detailed per-model pairs are reported in Table 3. Efficiency measurements (tokens/latency/steps)
are in Table 4.
Table 2: Overall QA performance on heterogeneous benchmarks. Shaded cells (N/A) indicate the
method is not applicable to that benchmark; gray dashes (–) indicate metric not reported. The record
results use Qwen3-4B-Instruct-2507 for HSEQ-I; and Falcon-H1-7B-Instruct for HSEQ-H
MethodHybridQA TAT-QA HotpotQA MetaQA-2hop MetaQA-3hop
Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
LLM-only (direct QA)
Falcon3-10B-instruct 22.4 – 35.2 – 16.5 – 43.0 – 39.8 –
Falcon-H1-7B-Instruct 32.9 – 43.7 – 21.1 – 48.3 – 44.6 –
Llama-3.1-8B-Instruct 28.1 – 37.6 – 14.6 – 37.8 – 31.9 –
Qwen3-4B-Instruct-2507 30.3 – 42.1 – 17.8 – 42.2 – 38.5 –
RAG-based methods (single-pass / agentic baselines)
TAT-LLM – – 74.6 82.9 N/A N/A N/A N/A N/A N/A
TableRAG 47.9 – 61.9 68.6 N/A N/A N/A N/A N/A N/A
ODYSSEY 51.5 66.0 – – N/A N/A N/A N/A N/A N/A
TTQA-RS 62.3 70.6 – – N/A N/A N/A N/A N/A N/A
HippoRAG 65.8 72.4 70.1 74.9 53.2 55.7 N/A N/A N/A N/A
Graph-constrained Reasoning (GcR) N/A N/A N/A N/A 39.2 41.6 86.7 88.1 83.2 80.6
Think on Graph (ToG) N/A N/A N/A N/A 43.1 44.7 83.2 84.8 81.1 78.5
AdaptiveRAG N/A N/A N/A N/A 50.3 52.5 88.2 90.1 84.5 85.7
Our method: HSEQ
HSEQ (best) 66.472.1 75.7 83.5 56.3 58.6 95.9 91.1 93.4 88.3
HSEQ (median) 63.9 70.8 73.2 79.6 55.4 57.1 93.2 89.7 90.1 86.6
Our HSEQ achieves strong and consistent gains on multiple benchmarks. On HotpotQA, MetaQA-
2hop, and MetaQA-3hop, both thebestandmedianHSEQ configurations surpass all baselines. On
7

Paper under review
TAT-QA, HSEQ’s best run attains the top score overall, while the median run trails slightly behind
TAT-LLM (Zhu et al., 2024). On the table-and-text HybridQA, HSEQ attains the best accuracy and
the second-best F1 (just behind HippoRAG (Jimenez Gutierrez et al., 2024)); the median configura-
tion remains third among baselines.
4.3 YIELDING BETWEEN EFFICIENCY AND ACCURACY
Table 3 lists results using different HSEQ-I and HSEQ-A. The HybridQA results reveal a clear accu-
racy–efficiency trade-off across HSEQ agent pairs. The highest accuracy/F1 comes from Qwen3-4B
(HSEQ-I) + Falcon-H1-7B (HSEQ-H) (66.2 / 71.4), with the second-best Qwen3-4B + Llama-3.1-
8B (65.5 / 71.2). These configurations, however, incur larger iteration depth and latency (about
3.7–4.1 steps; 16.5–21.5 second). On the efficiency end, Llama-3.2-3B + Llama-3.1-8B delivers
the lowest steps and latency (2.11; 8.35k ms) with moderate accuracy (55.4 / 57.9), while Falcon3-
3B + Falcon-H1-7B attains the second-best efficiency (2.25; 11.7k ms) at similar quality. Taken
together, the Pareto frontier spans (i) Qwen-based iterators with larger heads for top accuracy, and
(ii) lightweight Llama/Falcon pairs for predictable low latency. Different agent pairs can be chosen
regarding whether accuracy or budget dominates.
Table 3: Overall performance of HSEQ agent pairs on Hybrid-QA: Accuracy/F1 and Efficiency.
Accuracy & F1 Efficiency
Iteration Agent (HSEQ-I) Head Agent (HSEQ-H) Avg. Acc Avg. F1 Steps↓ Latency (ms)↓
Llama-3.2-3B-Instruct Falcon3-10B-instruct 60.4 62.8 2.08 12055.5
Qwen3-4B-Instruct-2507 Falcon3-10B-instruct 63.9 64.5 4.1 20577.5
Falcon3-3B-instruct Falcon3-10B-instruct 59.3 61.1 2.6 10530.1
Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct 55.4 57.9 2.11 8346.3
Qwen3-4B-Instruct-2507 Llama-3.1-8B-Instruct 65.5 71.2 3.29 16503.2
Falcon3-3B-instruct Llama-3.1-8B-Instruct 61.2 65.1 2.46 11616.7
Llama-3.2-3B-Instruct Falcon-H1-7B-Instruct 58.7 63.9 2.41 12080.0
Qwen3-4B-Instruct-2507 Falcon-H1-7B-Instruct 66.2 71.4 3.71 21479.2
Falcon3-3B-instruct Falcon-H1-7B-Instruct 56.1 58.6 2.25 11714.4
Llama-3.2-3B-Instruct DeepSeek-R1-Distill-Qwen-7B 62.5 60.2 2.75 15073.7
Qwen3-4B-Instruct-2507 DeepSeek-R1-Distill-Qwen-7B 62.8 66.7 4.07 21094.8
Falcon3-3B-instruct DeepSeek-R1-Distill-Qwen-7B 61.4 62.0 3.01 13709.7
4.4 EFFICIENCYANALYSIS
To test HSEQ framework’s latency,evidence actually inspectedare calculated: iteration steps for
HSEQ-I and wall-clock latency are calculated. Results are summarized below. “LLM-only” incurs
a single forward pass (1 step) and thus the lowest raw latency, but it lacks provenance and typically
underperforms on accuracy in Table 3. HSEQ maintains short, budgeted loops (about 3–5 steps)
with substantially lower latency than Graph-centric ToG while preserving multi-hop capability.
Table 4: Efficiency metrics on HotpotQA, MetaQA-2hop and MetaQA-3hop.
Efficiency
Method HotpotQA MetaQA-2hop MetaQA-3hop
Steps Latency (ms)↓ Steps Latency (ms)↓ Steps Latency (ms)↓
LLM-only 1 3266.3 1 2556.4 1 3631.1
Think on Graph (ToG) 13.28 22708.2 11.73 15707.6 16.58 24307.4
HSEQ (ours, best) 4.00 6247.0 3.27 5732.2 4.11 10632.8
HSEQ (ours, median) 4.17 12114.4 3.76 9480.1 4.59 13505.3
4.5 ABLATIONSTUDIES
Ablation studies are set to evaluate each component of HSEQ framework on representative text
(HotpotQA) and table-text (HybridQA) tasks. Following tasks are considered: (a)No SFT(iteration
agent not fine-tuned); (b)No guidance(removeg); (c)Heuristic-only guidance(no planner) ; and
(d)LLM-only(without multi-agent but use HSEQ as part of prompt for data input).
8

Paper under review
Table 5: Ablations on benchmarks.
Variant HybridQA TAT-QA HotpotQA MetaQA-3hop MetaQA-3hop
Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
HSEQ (full) 66.4 72.1 75.7 83.5 56.3 58.6 95.9 91.1 93.4 88.3
w/o SFT (base iteration) 57.3 65.7 60.4 66.9 46.5 47.8 78.3 80.1 74.6 72.5
w/o guidance 59.2 62.6 68.8 75.1 50.5 51.2 82.4 83.0 79.2 73.8
heuristic-only guidance 63.8 67.3 70.4 79.9 54.7 56.1 87.3 85.4 83.9 86.1
LLM-only (no HSEQ) 32.9 – 43.7 – 21.1 – 48.3 – 44.6 –
The ablation study demonstrates the necessities of all HSEQ’s components, with differing sensitiv-
ity across formats. Usingheuristic-onlyguidance yields the smallest degradation from the full sys-
tem—typically a modest drop in Acc/F1—indicating that a lightweight, template-style prior already
guides HSEQ-I effectively when the planner is absent. Removing fine-tuning (w/o SFT) causes
a larger decline, but with the use of structured HSEQ data, accuracy remains substantially higher
thanLLM-only. Without guidance (w/o guidance) influence performance, as in prompt HSEQ-I is
only asked tochoose necessary evidence from below to answer the question. The results under-
score the role of guidance as a portable sufficiency prior. Finally, theLLM-onlysetting performs
worst across all benchmarks, reflecting the difficulty of recovering minimally sufficient evidence
without iterative, structure-aware selection. Overall, the results suggest that (i) HSEQ’s unified data
structure is the primary source of robustness, (ii) SFT HSEQ-I provides consistent gains, and (iii)
guidance—even a simple heuristic ones from template-would increase overall accuracy strongly.
5 RELATEDWORK
LLM FinetuningLarge Language Models (LLMs) often adopt finetuning to unlock their capabil-
ities for downstream applications, like medical (Goyal et al., 2024), economic Guo & Yang (2024),
or human activity recognition Li et al. (2024). To enhance finetuning efficiency, methods like quan-
tization (Dettmers et al., 2022) parameter efficient fine tuning (Hu et al., 2022; Dettmers et al., 2023;
Li & Liang, 2021) can be applied.
Retrieval Augmented GenerationRAG systems help LLMs retrieve extra knowledge according
to queries and thereby improving the accuracy of LLM response (Fan et al., 2024), with no necessity
to finetune the model. External databases ensure knowledge offered is domain-specific and timely,
adding reliability and interpretability (Lewis et al., 2020; Jiang et al., 2023).Accuracyof knowledge
retrieval andqualityof responses are two key factors for RAG systems evaluation (Yu et al., 2024).
Apart from text, table, or html sources (Guo et al., 2024b; Chan et al., 2024; Jin et al., 2025),
recent researches have combined graph-structured data into RAG systems(GraphRAG) to improve
the efficiency of knowledge interpretability by capturing relationships between entities and utilizing
triplets as the primary data source (Edge et al., 2024; Peng et al., 2024; Hu et al., 2024; Mavromatis
& Karypis, 2024).
Multi Agent QA systemLLM-based Multi-Agent Systems (MASs) enable groups of intelligent
agents to coordinate and solve complex tasks collectively at scale, transitioning from isolated mod-
els to collaboration-centric approaches (Tran et al., 2025). Agents can cooperate with each other
for tasks like code generation (Hong et al., 2024; Islam et al., 2024), decision making (Nascimento
et al., 2023; Shinn et al., 2023), while competitions among agents are appiled on gaming environ-
ment Wang et al. (2022) or question answering (Puerto et al., 2021). By interacting with each other,
the system can be used for both problem solving or world simulation (Guo et al., 2024a)
6 CONCLUSION
This paper introducesHSEQ, a compact framework for heterogeneous QA that (i)unifiestext,
tables, and knowledge graphs into a reversible hierarchical sequence with lightweight structure and
provenance; (ii) performsguided, budget-aware iterationthat selects small sets of salient segments
and predictssufficiencyfor early stopping; and (iii) feeds acanonicalized evidencepackage to a head
module for answer synthesis. By replacing single-shot retrieval and unconstrained agentic loops with
9

Paper under review
short, structure-aware selections equipped with an explicit sufficiency signal, HSEQ concentrates
computation onevidence actually inspected, delivers predictable latency under token/tool budgets,
and preserves auditability through provenance-aware canonicalization.
Across heterogeneous QA benchmarks, HSEQ achieves strong answer quality alongside consistent
efficiency, revealing a controllable trade-off between accuracy and cost: larger head with finetuned
small iterators achieved both fast and accurate QA. The format-agnostic interface and standardized
action schema enable a single learned policy to operate across modalities without per-dataset retriev-
ers, bespoke prompts, or tokenizer changes.Future workwill extend HSEQ to multi-turn/streaming
settings with dynamic corpora, mitigate hallucination on sufficiency judge under noisy evidence.
7 ETHICSSTATEMENT.
We affirm adherence to the ICLR Code of Ethics. All experiments use publicly available benchmarks
(HybridQA, TatQA, HotpotQA, MetaQA) under their respective licenses; no new human-subject
data were collected, and no personally identifiable information (PII) is processed. Our HSEQ con-
struction preserves provenance via identifiers and offsets while avoiding storage of copyrighted text
beyond short snippets necessary for QA. As with any LLM-based system, model outputs may re-
flect societal biases inherited from pretraining corpora; we mitigate this risk by requiring explicit,
auditable evidence and by permitting abstention when sufficiency is not met. We release code and
configuration solely for research use and discourage deployment in high-stakes settings without
domain-specific evaluation and additional safeguards (fairness, privacy, and safety audits).
8 REPRODUCIBILITYSTATEMENT.
We provide an anonymous GitHub link (https://anonymous.4open.science/r/HSEQ-anonymous-
0DAC) with code and scripts to (i) construct HSEQ from raw corpora, (ii) fine-tune the iteration
policy with LoRA, and (iii) run guided inference and evaluation. Implement details are shown
in App. A.3, containing models used (App. A.3.1), prompts (App. A.3.2- A.3.3), LoRA adaption
parameters (App. A.3.4) and reproducibility notes (App. A.3.6). Theorems include complete as-
sumptions and proofs (App. A.1). Apart from the code, detailed examples of agents interactions
(example questions, LLM outputs, data retreived each steps, etc.) are provided in App. A.5 and as a
jsonl file in our anonymous repository.
ACKNOWLEDGMENTS
This research is partially support by Technology Innovation Institure Abu Dhabi, UAE. Also, this re-
search includes computations using the computational cluster supported by SHARON AI at Sydney,
New South Wales.
REFERENCES
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
Ankush Agarwal, Chaitanya Devaguptapu, et al. Hybrid graphs for table-and-text based question
answering using llms.arXiv preprint arXiv:2501.17767, 2025.
Jayetri Bardhan, Bushi Xiao, and Daisy Zhe Wang. Ttqa-rs-a break-down prompting approach
for multi-hop table-text question answering with reasoning and summarization.arXiv preprint
arXiv:2406.14732, 2024.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu. Rq-rag:
Learning to refine queries for retrieval augmented generation.arXiv preprint arXiv:2404.00610,
2024.
Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Wang. Hy-
bridqa: A dataset of multi-hop question answering over tabular and textual data.arXiv preprint
arXiv:2004.07347, 2020.
10

Paper under review
Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin,
Yiming Yang, and Jiaxin Mao. Improving retrieval-augmented generation through multi-agent
reinforcement learning.arXiv preprint arXiv:2501.15228, 2025.
Philipp Christmann and Gerhard Weikum. Rag-based question answering over heterogeneous data
and text.arXiv preprint arXiv:2412.07420, 2024.
Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3. int8 (): 8-bit matrix
multiplication for transformers at scale.Advances in neural information processing systems, 35:
30318–30332, 2022.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning
of quantized llms.Advances in neural information processing systems, 36:10088–10115, 2023.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
arXiv e-prints, pp. arXiv–2407, 2024.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and
Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In
Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
pp. 6491–6501, 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and
Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2023.
Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Rajaram Naik, Pengshan
Cai, and Alfio Gliozzo. Re2g: Retrieve, rerank, generate.arXiv preprint arXiv:2207.06300, 2022.
Sagar Goyal, Eti Rastogi, Sree Prasanna Rajagopal, Dong Yuan, Fen Zhao, Jai Chintagunta, Gautam
Naik, and Jeff Ward. Healai: A healthcare llm for effective medical documentation. InProceed-
ings of the 17th ACM International Conference on Web Search and Data Mining, pp. 1167–1168,
2024.
Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest,
and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and
challenges.arXiv preprint arXiv:2402.01680, 2024a.
Yue Guo and Yi Yang. Econnli: evaluating large language models on economics reasoning.arXiv
preprint arXiv:2407.01212, 2024.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-
augmented generation.arXiv preprint arXiv:2410.05779, 2024b.
Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin
Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for
a multi-agent collaborative framework. International Conference on Learning Representations,
ICLR, 2024.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation of large language models.ICLR, 1(2):3, 2022.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. Grag: Graph retrieval-
augmented generation.arXiv preprint arXiv:2405.16506, 2024.
Md Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. Mapcoder: Multi-agent code
generation for competitive problem solving.arXiv preprint arXiv:2405.11403, 2024.
11

Paper under review
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity.arXiv
preprint arXiv:2403.14403, 2024.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, pp. 7969–7992, 2023.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobi-
ologically inspired long-term memory for large language models.Advances in Neural Information
Processing Systems, 37:59532–59569, 2024.
Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao Zhang, Tong Zhao,
Zhao Yang, and Ji-Rong Wen. Flashrag: A modular toolkit for efficient retrieval-augmented
generation research. InCompanion Proceedings of the ACM on Web Conference 2025, pp. 737–
740, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation.arXiv
preprint arXiv:2101.00190, 2021.
Zechen Li, Shohreh Deldari, Linyao Chen, Hao Xue, and Flora D Salim. Sensorllm: Aligning large
language models with motion sensors for human activity recognition. 2024.
Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun Ma. Hm-rag: Hierar-
chical multi-agent multimodal retrieval augmented generation.arXiv preprint arXiv:2504.12330,
2025.
Haoran Luo, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guanting
Dong, Meina Song, Wei Lin, Yifan Zhu, et al. Chatkbqa: A generate-then-retrieve framework
for knowledge base question answering with fine-tuned large language models.arXiv preprint
arXiv:2310.08975, 2023.
Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Yuan-Fang Li, Chen Gong, and Shirui Pan. Graph-
constrained reasoning: Faithful reasoning on knowledge graphs with large language models.
arXiv preprint arXiv:2410.13080, 2024.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian
Guo. Think-on-graph 2.0: Deep and faithful large language model reasoning with knowledge-
guided retrieval augmented generation.arXiv preprint arXiv:2407.10805, 2024.
Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language model
reasoning.arXiv preprint arXiv:2405.20139, 2024.
Nathalia Nascimento, Paulo Alencar, and Donald Cowan. Self-adaptive large language model (llm)-
based multiagent systems. In2023 IEEE International Conference on Autonomic Computing and
Self-Organizing Systems Companion (ACSOS-C), pp. 104–109. IEEE, 2023.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and
Siliang Tang. Graph retrieval-augmented generation: A survey.arXiv preprint arXiv:2408.08921,
2024.
Haritz Puerto, G ¨ozde G ¨ul S ¸ahin, and Iryna Gurevych. Metaqa: Combining expert agents for multi-
skill question answering.arXiv preprint arXiv:2112.01922, 2021.
Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learning.Advances in Neural Information Processing
Systems, 36:8634–8652, 2023.
12

Paper under review
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented
generation: A survey on agentic rag.arXiv preprint arXiv:2501.09136, 2025.
Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and
Hoang D Nguyen. Multi-agent collaboration mechanisms: A survey of llms.arXiv preprint
arXiv:2501.06322, 2025.
Jianrui Wang, Yitian Hong, Jiali Wang, Jiapeng Xu, Yang Tang, Qing-Long Han, and J ¨urgen Kurths.
Cooperative and competitive multi-agent systems: From optimization to games.IEEE/CAA Jour-
nal of Automatica Sinica, 9(5):763–783, 2022.
Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, and Flora D Salim. Beyond single pass, looping
through time: Kg-irag with iterative knowledge retrieval.arXiv preprint arXiv:2503.14234, 2025.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering.arXiv preprint arXiv:1809.09600, 2018.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. Evaluation of retrieval-
augmented generation: A survey.arXiv preprint arXiv:2405.07437, 2024.
Wenhao Yu. Retrieval-augmented generation across heterogeneous knowledge. InProceedings of
the 2022 conference of the North American chapter of the association for computational linguis-
tics: human language technologies: student research workshop, pp. 52–58, 2022.
Xiaohan Yu, Pu Jian, and Chong Chen. Tablerag: A retrieval augmented generation framework for
heterogeneous document reasoning.arXiv preprint arXiv:2506.10380, 2025.
Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song. Variational reasoning
for question answering with knowledge graph. InProceedings of the AAAI conference on artificial
intelligence, volume 32, 2018.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A
survey.arXiv preprint arXiv:2402.19473, 2024.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng,
and Tat-Seng Chua. Tat-qa: A question answering benchmark on a hybrid of tabular and textual
content in finance.arXiv preprint arXiv:2105.07624, 2021a.
Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua.
Retrieving and reading: A comprehensive survey on open-domain question answering.arXiv
preprint arXiv:2101.00774, 2021b.
Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, and Tat Seng Chua. Tat-llm: A
specialized language model for discrete reasoning over financial tabular and textual data. In
Proceedings of the 5th ACM International Conference on AI in Finance, pp. 310–318, 2024.
Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume
Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, et al. Falcon-h1: A
family of hybrid-head language models redefining efficiency and performance.arXiv preprint
arXiv:2507.22448, 2025.
A APPENDIX
A.1 THEORETICALPROPERTIES OFHSEQ
A.1.1 PRELIMINARIES ANDASSUMPTIONS
Segment schema.An HSEQ is a finite multisetS hof segmentss= (id(s), ℓ(s), p(s), c(s), µ(s)).
Hereℓis a level tag;pis a parent pointer withp(s) =⊥ifsis a root;cis content;µis metadata,
possibly includingoffsets(for text),schemaand row indices (for tables), and triplet fields (for
KGs).
13

Paper under review
Encoder/decoder.LetΦmap any finite corpusX(text + tables + KG) toS h= Φ(X), and letΨ
mapS hback to a corpusΨ(S h). We assume the following modality-specific invariants are enforced
by the adapters (they match the implementation but are stated abstractly).
(T1) Text offsets.For each text itemx∈Σ∗, ifsis a paragraph (resp. sentence) segment for a
spanx[a:b)(resp.x[u:v)inside a paragraph), thenµ(s).offsets= [a, b](resp.
[a+u, a+v]),c(s) =x[a:b)(resp.x[a+u:a+v)), andpis the unique parent in the
containment chain (sentence→paragraph→document).
(T2) Table rows.For a table with headerH= (h 1, . . . , h C)andnrows(r i)n
i=1, the table-root
segment storesHinµ(·).schema; each row-segments istoresc(s i) = dict(H7→r i)
and either (a) an explicit row indexµ(s i).offsets= [i,−1], or (b) a total order on row
segments consistent with the original row order.
(T3) KG triples.For a KG edge multisetE⊆ E × R × E(optionally time-stamped), each
edge(h, r, t, τ)corresponds to exactly one triplet segmentswithc(s) = (h, r, t)and
µ(s).time=τ; parentp(s)is the unique subgraph-root for the neighborhood.
Benign equivalence.Define an equivalence relation≡over corpora by (i) ignoring differences in
text whitespace that do not change the sequence of non-whitespace characters; (ii) allowing a global
column permutationπ∈S Capplied uniformly to the header and all row dictionaries of a table; (iii)
treating KGs as edge multisets (edge order immaterial).
Ordering and window.Letρbe a total order overS h(e.g., paragraph≺row≺sentence≺triplet
with a deterministic tie-break). The stream induced byρlistsS has(s 1, . . . , s N). For a window size
W∈N,Window(S h;W, ρ)returns the firstWitems of the stream that are not already selected;
Refresh(S h, M;W, ρ)returns the nextWunseen items after removingM. Both aremonotonew.r.t.
ρ: the sequence of items exposed across refreshes is exactly theρ-stream with already selected items
removed.
Admissibility.For a questionq, a supporting setE⋆⊆Shisanswer-supportingif the head module
Hyields the correct answer when given onlyE⋆. An orderρisadmissiblefor(q, S h)if there exists
a minimalL∈ {1, . . . ,|S h|}such thatE⋆⊆ {s 1, . . . , s L}for some answer-supportingE⋆.
Sufficiency predicate.LetSuff(M)be a predicate that holds iffMcontains some answer-
supporting subset. We assume a calibrated sufficiency head: wheneverSuff(M t)becomes true,
the policy can set its stop flags t= 1at that step or earlier.1
A.1.2 FAITHFULLINEARIZATION
Theorem 1(Faithful linearization).For any finite corpusX, under (T1)–(T3), the encoderΦis
injective up to≡, i.e.,Ψ(Φ(X))≡X.
Proof.WriteX=X text⊎X tbl⊎X kgand letS h= Φ(X). We showΨ(Φ(·))acts as identity
modulo≡on each modality and hence on their disjoint union.
Text.Considerx∈X text. By (T1) each paragraph (resp. sentence) segmentsstores the exact
substringc(s) =x[a:b)(resp.x[u′:v′)) and absolute offsets inµ(s).offsets. LetS x⊆Sh
be all segments rooted at the document node ofx. The decoder reconstructsx′by placing every
paragraph substring at its[a, b)range and merging overlaps implied by sentence children; uniqueness
of parents eliminates ambiguity. Because offsets are absolute and children are contained in parents
by construction, the reconstructedx′equalsxcharacter-for-character; any whitespace normalization
is permitted by≡.
Tables.Let a table have headerH= (h 1, . . . , h C)and rows(r i)n
i=1. By (T2),µ(·).schemastores
H, and each row segments istores the dictionaryc(s i)mappingHto the row tupler i, together
with either an explicit row index or a total order consistent with the original order. The decoder
1This is standard in supervised setups where the stop head is trained to fire at first sufficiency (or with
tolerance).
14

Paper under review
reassembles the matrix[H;r 1;. . .;r n]. Any global column permutationπyields an equivalent table
under≡; thus the reconstruction is unique modulo schema-order permutations.
KGs.LetEbe the multiset of edges. By (T3), each edge(h, r, t, τ)corresponds bijectively to one
triplet segment withc(s) = (h, r, t)andµ(s).time=τ, and parentage is irrelevant to content. The
decoder collects the multiset of triplets, which equalsE; edge order is immaterial and thus fits≡.
Since the three reconstructions are independent and disjointly supported,Ψ(Φ(X))≡Xfollows.
A.1.3 WINDOWEDITERATION: COVERAGE ANDCOMPLEXITY
LetE⋆⊆ {s 1, . . . , s L}be an answer-supporting set with minimal prefix lengthLunder an admis-
sible orderρ. Fix windowW≥k≥1and define the iterative selection with refresh as in the main
text.
Lemma 1(Prefix coverage underk-selection).Aftertsteps, the selected setM tcontains at least
min{kt, L}items from theρ-prefix{s 1, . . . , s L}. In particular,E⋆⊆M TforT=⌈L/k⌉.
Proof.We prove by induction ont≥0that|M t∩ {s 1, . . . , s L}| ≥min{kt, L}.
Baset= 0:M 0=∅so the bound is0.
Inductive step: assume the claim fort−1. At stept, the window exposes (by monotonicity of
Refresh) the earliestWunseen items underρ; hence at least the nextkunseen items in the prefix
{s1, . . . , s L}are eligible (becauseW≥k). Selectingknew items (or fewer if fewer remain in the
prefix) increases the count by at leastmin{k, L−(t−1)k}, givingmin{kt, L}. Once allLprefix
items are selected, the bound saturates atL.
Proposition 1(Guaranteed halt).Assume a step capT max and a sufficiency head that can set
st= 1wheneverSuff(M t)holds. Under admissibility, the control loop halts after at most
min{T max,⌈L/k⌉}steps.
Proof.By Lemma 1, afterT=⌈L/k⌉steps,E⋆⊆M T; henceSuff(M T)holds and the stop
head can fire at or beforeT. Independently, the hard capT maxforces termination byT maxsteps.
Thereforeτ≤min{T max, T}.
Theorem 2(Budgeted selection complexity).LetC(W)>0be the (deterministic) per-step context
cost determined by window sizeW. Under admissibility, the total selection cost is bounded by
Cost select≤C(W)·min{T max,⌈L/k⌉},
independent of|S h|. IfLis a nonnegative integer random variable withE[L] = ¯L <∞, then
E
Cost select
≤C(W)·E[min{T max,⌈L/k⌉}]≤C(W)·min{T max,¯L/k+ 1}.
Proof.The first bound follows by multiplying the per-step cost by the halt bound in Proposition 1.
For the expectation, use linearity of expectation and the inequality⌈x⌉ ≤x+ 1forx≥0:
E[⌈L/k⌉]≤E[L]/k+1 = ¯L/k+1, andE[min{a, X}]≤min{a,E[X]}fora≥0andX≥0.
A.2 WEAK-POSITIVELABELING ANDTRAJECTORYSYNTHESIS
Positive identification.For each instance, segments are sorted by alevel prioritythat favors
container-like units (e.g., paragraphs, rows). Within a capped candidate set, a positive poolP⋆is
constructed by: (i) exact/substring matching of the gold answer incontent; and (ii) if insufficient,
selecting top segments by lexical Jaccard overlap between tokenizedqand segment content.
Sufficiency heuristic.A sufficiency thresholduis used to labels⋆
t: if the union of already-selected
and newly-picked positives reaches≥u, marksufficient= 1and stop; otherwise continue.
Smalluencourages minimal-evidence solutions.
15

Paper under review
Trajectory construction.GivenP⋆and a per-step capk, a target sequence is synthesized by
greedily choosing up tokunseen positives at each step until sufficiency holds or candidates are
exhausted. Low-confidence choices (from lexical overlap rather than exact match) can be down-
weighted in the loss.
Proxy selection metric.During development, a lightweight proxy evaluates selection quality:
for a held-out set, the agent’s chosen ids are compared with target ids to compute micro Preci-
sion/Recall/F1 over segment identifiers. This tracks selection ability without requiring full QA eval-
uation.
A.2.1 CANONICALIZATION ANDSOUNDNESS
Definition 1(Canonicalizer).A canonicalizerκmapsM⊆S hto a finite structureκ(M)consisting
only of tuples(id, ℓ,content view,provenance)wherecontent viewis a deterministic, lossless
projection ofc(s)andµ(s), andprovenancecontains the fields needed to locatesinS h(e.g., offsets
or coordinates). We sayκiscontent-preservingif for allM, the multiset{(c(s), µ(s)) :s∈M}is
reconstructible fromκ(M).
Proposition 2(Soundness and auditability).IfSuff(M)holds andκis content-preserving, then the
headHapplied to(q, κ(M))is supported solely by items inM, and every atomic support can be
traced back to a unique segment inMviaidand provenance.
Proof.By content preservation,κ(M)contains all information from{(c(s), µ(s)) :s∈M}; there-
foreHrestricted toκ(M)depends only on evidence inM. Sinceκstoresidand provenance per
item, any atomic support used byHcan be mapped to a uniques∈M. Auditability follows.
A.2.2 PROBABILISTICCOMPLETENESSUNDERSTOCHASTICSELECTION
We next quantify success probability for a stochastic policy that may fail to pick all supporting items
even if they appear early in the stream.
Definition 2(Exposure count).Fix an admissibleρwith prefix lengthLand selection sizek≥1.
LetR=⌈L/k⌉. An elemente∈ {s 1, . . . , s L}is said to beexposedat steps1, . . . , R, meaning it
either is in the first window where it lies or remains eligible until selected; monotone refresh ensures
at mostRexposures before all prefix items are exhausted.
Assumption 1(Per-exposure success).There existsp∈(0,1]such that for everye∈E⋆and for
every steptat whicheis exposed and not yet selected, the policy includeseinK twith probability
at leastp, independently across steps for the samee.
Theorem 3(Stochastic completeness).Under admissibility and Assumption 1, withR=⌈L/k⌉
andm=|E⋆|, the probability that all items inE⋆are selected withinRsteps is bounded below by
P[E⋆⊆M R]≥1−m(1−p)R.
Consequently, by Proposition 1, the probability that the loop halts bymin{T max, R}with a correct
answer is at least1−m(1−p)R.
Proof.Fixe∈E⋆. By Assumption 1, across its at mostRexposures, the probability thateis never
selected is at most(1−p)R. By the union bound over themitems inE⋆,
P[∃e∈E⋆not selected by stepR]≤m(1−p)R.
Taking complements yields the first claim. The second claim follows because onceE⋆⊆M t,
Suff(M t)holds and the stop head can fire; the hard cap can only make halting earlier.
A.2.3 DISCUSSION OFASSUMPTIONS
The injectivity result (Thm. 1) relies on invariants (T1)–(T3), which are satisfied by construction in
the HSEQ adapters (offsets and row indices/ordering are recorded; triplets are stored verbatim). Ad-
missibility is a regularity condition stating that an orderρexists (often paragraph/row-first) placing
supporting segments early; in practice this is further improved by guidance. Assumption 1 abstracts
a calibrated selector that repeatedly assigns nontrivial probability mass to any exposed, still-missing
support item; the bound in Theorem 3 is conservative (union bound) and can be tightened under
additional structure (e.g., adaptivek, or margin assumptions on scoring).
16

Paper under review
A.3 IMPLEMENTATIONDETAILS
A.3.1 AGENT MODELS USED FORHSEQ
Models used for both iteration agent and head agent are shown in Table 6, grouped by size. Most
experiments are done by using small and medium models (as of the result shown in main text).
Table 6: Iteration-agent and head agent base models grouped by size.
Group Model (HF id)
SMALLtiiuae/Falcon-H1-0.5B-Instruct
tiiuae/Falcon-H1-1.5B-Instruct
tiiuae/Falcon3-1B-instruct
meta-llama/Llama-3.2-1B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MEDIUMtiiuae/Falcon3-3B-instruct
tiiuae/Falcon-H1-3B-Instruct
Qwen/Qwen3-4B-Instruct-2507
tiiuae/Falcon3-7B-instruct
tiiuae/Falcon-H1-7B-Instruct
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Llama-3.1-8B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
LARGEtiiuae/Falcon3-10B-instruct
tiiuae/Falcon-H1-34B-Instruct
Qwen/Qwen3-30B-A3B-Instruct-2507
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
deepseek-ai/DeepSeek-R1-Distill-Llama-70B
meta-llama/Llama-3.1-70B-Instruct
A.3.2 ITERATION-AGENTPROMPTS ANDOUTPUTSCHEMA
System instruction.The iteration agent is conditioned with a concise, role-defining system mes-
sage:
You are an iteration agent working over a hierarchical
sequence (H-Seq).
Given a question and a list of candidate segments (each with
an id and text)
select the top-k segment ids that best support answering the
question.
Then decide if the selected evidence is sufficient to stop.
Return ONLY compact JSON with keys: type, args.segment ids,
args.strategy, args.top k, sufficiency.
WITHOUT ANY EXPLAINATION.
Prompt template.Each training step uses a structured multi-section prompt:
### Instruction
{system-instruction}
### Question
{q}
### Guidance
{g(q,type)}
### Selected-So-Far
- [seg id] truncated content
...
### Candidate-Window
- [seg id] truncated content
...
17

Paper under review
### Output (JSON)
Only identifiers, levels, truncated content, and key metadata of segments are serialized.
Output schema.The agent must emit deterministic, machine-checkable JSON:
{"type": "select", "args":{"segment ids": [...],
"strategy": "guided topk", "top k": k}, "sufficiency":
true/false}
No free-form text is allowed. This constraint simplifies supervision and evaluation.
Masking for SFT.During supervised fine-tuning, the loss is applied only to theoutputportion of
the sequence (prompt tokens are masked), yielding a standard next-token objective over the action
string while keeping inputs loss-free.
A.3.3 GUIDANCEGENERATION ANDCACHING
Head-generated guidance.A lightweight planner (“head”) converts(q,type)into a short plang
that specifies: (i) what to retrieve first, (ii) optional branches, and (iii) a sufficiency hint. The planner
is prompted with:
You are a planning assistant. Given a question, write
a short retrieval plan for an iteration agent selecting
evidence snippets. Specify ONLY what to retrieve first,
possible branches, and when to stop (sufficiency condition).
A short completion is generated and, if too brief or incomplete, a single continuation is requested to
end with an explicit stop condition.
Heuristic templates.When a head is unavailable or for ablations, templates keyed by coarse pat-
terns produceg, start with:
"Plan: retrieve a minimal set of highly relevant snippets; prefer
concise facts."
Then add the following according toQ type:
•Numeric: Look for numeric mentions and table rows; stop when final number is explicit
or corroborated.
•Factoid (who/which/where/when): Focus on short spans that directly contain the answer;
stop on a clear statement..
•Binary: Retrieve one-two definitive statements; stop when evidence strongly supports
yes/no..
•Default: Prefer snippets naming key entities/relations; stop when answer is explicitly
stated.
Caching.Guidance strings are cached per example using a stable key (dataset name and a hash of
q) under a directory organized by head model id. Cache is consulted before running the head planner
to reduce overhead.
Settings.The planning head is run with short outputs and deterministic decoding. A minimal-
length heuristic is applied to avoid truncated guidance.
A.3.4 LORA ADAPTATION ANDOPTIMIZATION
Parameterization.The iteration agent is obtained by adding low-rank adapters to a base causal
LLM. Adapters are attached to attention projections (q proj,k proj,v proj,o proj) and
MLP projections (gate proj,up proj,down proj); vocabulary and positional embeddings
are unchanged.
18

Paper under review
Table 7: Supervised fine-tuning (SFT) hyperparameters for each model-size group. These settings
apply to all models within the corresponding group.
Group Target steps Batch GA LR ML MS Top-k Mi BF16
SMALL 12000 2 8 2.0×10−53072 48 2 4 Yes
MEDIUM 9000 2 8 1.5×10−53072 48 4 4 Yes
LARGE 4500 1 16 1.0×10−52048 32 5 4 Yes
Notes.Batchis--per device train batch size.GradAccis--grad accum.LRis--lr.ML,
MS,Top-k,Mimap to--max length,--max segments,--top k,--max iters. BF16 indicates
--bf16enabled.
Default configuration.LoRA rankr= 16, scalingα= 32, dropout0.05, no bias; the language
head is preserved as a save-module. Mixed-precision and 4-bit weight quantization (NF4 with dou-
ble quantization) are used to reduce memory. Gradient checkpointing is enabled.
Training schedule.A cosine learning-rate schedule with warmup ratio0.03is used; batches are
accumulated over several steps to match the target global batch size. Maximum input length is
capped to a few thousand tokens; candidate windows and per-stepkare tuned to respect the overall
budget.
Mixture and curriculum.Examples are sampled across datasets by normalized weights; quotas
are computed for a target mixed size and shuffled. A short-to-long curriculum increases the maxi-
mum number of stepsTas training progresses.
Finetuning Parameters
A.3.5 CANONICALIZATION ANDSUFFICIENCY
Canonical evidence package.At termination, a modality-agnostic canonicalizerκconverts the
selected setM τinto a compact, auditable structure
κ(M τ) =
(id,level,uri,offsets,source type,snippet;meta)	
s∈M τ,
with the following contract: (i)id: globally unique, deterministically derived (e.g.,
sha1(uri,offsets)); (ii)uri: source identifier with version (e.g., document path or graph
name); (iii)offsets: zero-based half-open character indices[a, b)into theoriginalsource; for
tables,[i, j]denotes row/column coordinates; for KGs,offsets= (−1,−1); (iv)snippet: a
human-readable content aligned to sentence/field boundaries when possible; (v)meta: integrity
and alignment helpers (schema,time,source version,sha1). Duplicates are removed by
(uri,offsets)and the package isdeterministicallyordered byurithenoffsets. Typed
views are derived on demand:text⇒spans with section/paragraph ids;table⇒row id,col ids,
schema,cell coords;KG⇒(h, r, t)plus optional validity time.
Stopping signal.The sufficiency head outputss t∈ {0,1}at each step.Training targetsfollow
a coverage-based heuristic:s⋆
t= 1if and only if the currentM tsatisfies task-specific adequacy
(e.g., contains at least one gold-positive segment; achieves full slot coverage for table QA; or yields
a unique answer span/number under a fixed head). For weak supervision, per-step weights down-
weight low-confidence positives (App. A.2).Inferenceuses a calibrated thresholdτon the model’s
sufficiency scoreˆp tand enforces a minimum step countT min:
stop atτ= min{t≥T min: ˆpt≥τ}or when budgetBis exhausted.
Optionally, a lightweight contradiction checker triggers a one-shot refinement loop of at most∆
additional steps with tightened guidanceg′and reduced budgetB′. Thresholds(τ, T min)are selected
on the development split and may be calibrated via temperature scaling.
A.3.6 REPRODUCIBILITYNOTES
•Seed and sampling.A fixed seed is used for example subsampling and order shuffling.
19

Paper under review
Table 8: Notation used throughout the paper.
Symbol Meaning
q Natural-language query (question).
D={(x j, mj)}N
j=1 Heterogeneous corpus with itemsx jand modality tagsm j∈
{text,table,kg}.
mj Modality label for thej-th item (text / table / KG).
τ, τm Modality-aware adapter;τ(D)produces the unified hierarchical sequence.τ mis
the adapter for modalitym.
Sh TheHSEQ(hierarchical sequence):S h=F
jτmj(xj)∈ S∗.
S Segment universe. Each segments∈ Sis a lightweight record.
s= (id(s), ℓ(s), p(s), c(s), µ(s)) Segment fields: unique identifier, level tag (granularity), parent pointer, compact
human-readable content, standardized metadata.
ℓ(s) Level tag (e.g.,document/paragraph/sentence,
table row/table cell,triplet/subgraph).
p(s) Parent pointer (container linkage) encoding locality in the hierarchy.
c(s) Compact content snippet (text span / serialized table row / triple).
µ(s) Metadata with fixed keys (e.g.,source id,uri,offsets/coordinates,
schema,time).
πθ HSEQ-Iiteration policy (LLM-based) with parametersθ; operates over(q, S h)to
select evidence iteratively.
g=g(q,type) Shortguidanceprior (from planner/head or heuristics) shaping early exploration
and stop notion.
B, B t Budget (global / per-step): token, tool-call, step, and/or latency limits.
Mt Selected-evidence set at stept;M⋆is the final selected set at termination.
Ct Candidate window at stept(bounded by window size and ordering).
k, W Top-kselection cap per step; window sizeWfor the exposed candidate stream.
Tmax, Tmin Maximal and minimal number of iteration steps (cap and anti–early-stop).
ρ Deterministic ordering overS hlevels (e.g., paragraph≺row≺sentence≺triplet)
to form the stream.
N(·) Structure-aware neighborhood operators (parent/child, row/column, KG relation
hops).
at, st Action at stept(e.g., select up toksegments and/or expand neighborhoods) and
sufficiency predictions t∈ {0,1}.
Φ Budget-aware sufficiency criterion queried by the iterator to trigger termination.
κ Canonicalizer mappingM τto provenance-preserving evidence package (ids, lev-
els, offsets/coordinates, snippets).
H HSEQ-Hhead module for answer synthesis from(q, κ(M τ)); can also generate
guidanceg.
ξ Optional verifier; on contradiction detection, triggers a brief refinement loop with
tightenedg′and reducedB′.
y,ˆy Gold answer and system prediction, respectively.
E⋆Minimally sufficient evidence set (w.r.t. a fixed answerer) forqinD.
Window(·),Refresh(·) Operators to expose a bounded candidate window and to advance it while removing
already selected segments.
∆ Max number of additional refinement steps if the verifierξrequests a retry.
•Segment capping.The number of serialized candidate segments per step is capped to
respect the overall token budget; truncation is applied tocontentstrings for display.
•Budget control.Global limits on steps, tokens, and optional tool calls are enforced; guid-
ance encourages early sufficiency.
•Hardware.Experiments are run on maximum 4NVIDIA H200 Tensor Core GPU.
Mixed-precision and 4-bit quantization substantially reduce memory; typical training runs
fit on a single GPU.
A.4 NOTATIONS
Table 8 lists all symbols used in main context.
20

Paper under review
A.5 EXAMPLEUSINGHSEQ
A.5.1 CASESTUDY: GUIDEDITERATIVERETRIEVAL ONHYBRIDQA
Setup.Queryq:“Who is the author of the novel that inspired the 2004 Russian film directed
by Timur Bekmambetov?”HSEQ-I (iterator):Qwen3-4B-Instruct-2507; HSEQ-H (head):
Falcon-H1-7B-Instruct. Guidance mode:head; source: cache (latency≈0.12 ms).
Head-generated guidance.The head planner emits a short plan: (i) identify the 2004 Russian
film directed by Bekmambetov; (ii) locate the novel that inspired it; (iii) stop once theauthor of that
novelis found. This plan is injected as a prefix and acts as a soft prior on where the iterator should
probe first.
Guided iteration overS h.The iterator consumes the guidance and operates over the HSEQ
stream with a fixed window and top-kselection. Table 9 summarizes the six steps (all sufficiency
flags werefalse; the loop terminates by budget).
Table 9: Stepwise selection (abridged). Segment ids prefixed by level:p (paragraph),row (table
row).
Step Key picks (content excerpt) Sufficient?
1 p6df9c849: “Night Watch(. . . ) is a 2004 Russian . . . directed by
Timur Bekmambetov. It is loosely based on the novelThe Night Watch
by Serg[ei Lukyanenko]. . . ”No
2 pc15173df,p 3bc4a108,p 54f6ef94: contextual paragraphs
(“List of Russian films of 2004”, “2004” entries)No
3 rowa44a4a17: table row confirmingNight Watchwith director
“Timur Bekmambetov”No
4–6 additional table rows from the same list (Arie,Countdown,Dad or
Papa, etc.) providing film set contextYes
Answer synthesis.Afterτ=6iterations, the canonicalizerκcompacts the selected setM τ(para-
graph + corroborating table rows) into a provenance-preserving package (segment ids, levels, offsets,
snippets). The headHis promptedonlywith(q, κ(M τ))and outputs:
ˆy=Sergei Lukyanenko.
The prediction matches the gold answer (EM/F1 = 1.0). Runtime profile: selection latency
≈32,185 ms, head latency≈1,826 ms, total≈34,011 ms; number of iterations= 6.
Takeaway.Guidance steers the iterator to a high-yield paragraph in the first step, which already
contains the sufficient evidence (film identity and source novel). Subsequent steps provide cor-
roboration from structured rows. The provenance inκ(M τ)makes the final answer auditable: the
paragraphp 6df9c849explicitly tiesNight Watch(2004, Bekmambetov) to the novelNight Watch
by Sergei Lukyanenko, enabling concise and well-grounded answer synthesis by the head.
A.5.2 CASESTUDY: GUIDEDITERATIVERETRIEVAL ONHOTPOTQA
Setup.Queryq:“Which style is the building located on the East Side of Midtown Manhattan that
Robert Von Ancken appraised?”HSEQ-I (iterator):Qwen3-4B-Instruct-2507; HSEQ-H
(head):Falcon-H1-7B-Instruct. Guidance mode:head; source: generated online (latency
≈8,496 ms).
Head-generated guidance.The head planner issues a short plan: (i) identify buildings on the
East Side of Midtown Manhattan connected to appraiserRobert Von Ancken; (ii) once the specific
building is found, retrieve its architectural style; (iii) stop when the style is clearly linked to the
appraised building.
21

Paper under review
Guided iteration overS h.The iterator follows the guidance with a fixed window and top-kse-
lection. Table 10 lists the six steps (all sufficiency flagsfalse; termination by budget). Note that
Step 1 already surfaces the key paragraph about the Chrysler Building.
Table 10: Stepwise selection (abridged). Segment ids prefixed by level:p (paragraph).
Step Key picks (content excerpt) Sufficient?
1 pa73a8d8f: “TheChrysler Buildingis anArt Deco-styleskyscraper
located on the East Side of Midtown Manhattan . . . ”No
2 pc01522d2: “23 Beekman Place . . . apartment building . . . East Side
of Midtown Manhattan . . . ”No
3 p7c2aa386: “The Helmsley Building . . . Midtown Manhattan . . . ” No
4 p658d6333: “Robert Von Anckenis a prominent New York City real
estate appraiser . . . ”No
5 pe97ef7e6: “Lenox Hill Neighborhood House . . . East Side of Man-
hattan . . . ”Yes
Answer synthesis.Afterτ=5iterations, the canonicalizerκcompacts the selected setM τ(in-
cludingp a73a8d8fand the V on Ancken paragraphp 658d6333) into a provenance-preserving
package. The head answers using only(q, κ(M τ)):
ˆy=Art Deco-style skyscraper.
The prediction matches the gold answer. Runtime profile: selection latency≈32,153 ms, head
latency≈838 ms, total≈41,487 ms; iterations= 5.
Takeaway.The head’s guidance steers the iterator directly to a paragraph that states both the loca-
tion (East Side of Midtown) and the architectural style (Art Deco) of the relevant building (Chrysler
Building), while additional picks provide neighborhood and appraiser context. Provenance inκ(M τ)
supports auditable linking from the final answer to its evidence.
A.6 STATEMENT FORTHEUSE OFLARGELANGUAGEMODELS(LLMS)
We used large language models (LLMs) as general-purpose tools forwriting assistanceandengi-
neering support. For writing, we employed LLMs to improve clarity and style (e.g., rephrasing
sentences, tightening paragraphs, standardizing notation, and proofreading grammar). Drafting,
technical claims, algorithms, proofs, experiment design, and all final wording were authored and
verified by the authors. For engineering, we consulted LLMs for debugging; all research code, data
processing, and experiment scripts were implemented, audited, and executed by the authors. No text
or code generated by an LLM was used verbatim without author review; we take full responsibility
for the content.
22
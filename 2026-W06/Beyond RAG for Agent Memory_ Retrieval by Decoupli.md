# Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation

**Authors**: Zhanghao Hu, Qinglin Zhu, Hanqi Yan, Yulan He, Lin Gui

**Published**: 2026-02-02 12:04:58

**PDF URL**: [https://arxiv.org/pdf/2602.02007v1](https://arxiv.org/pdf/2602.02007v1)

## Abstract
Agent memory systems often adopt the standard Retrieval-Augmented Generation (RAG) pipeline, yet its underlying assumptions differ in this setting. RAG targets large, heterogeneous corpora where retrieved passages are diverse, whereas agent memory is a bounded, coherent dialogue stream with highly correlated spans that are often duplicates. Under this shift, fixed top-$k$ similarity retrieval tends to return redundant context, and post-hoc pruning can delete temporally linked prerequisites needed for correct reasoning. We argue retrieval should move beyond similarity matching and instead operate over latent components, following decoupling to aggregation: disentangle memories into semantic components, organise them into a hierarchy, and use this structure to drive retrieval. We propose xMemory, which builds a hierarchy of intact units and maintains a searchable yet faithful high-level node organisation via a sparsity--semantics objective that guides memory split and merge. At inference, xMemory retrieves top-down, selecting a compact, diverse set of themes and semantics for multi-fact queries, and expanding to episodes and raw messages only when it reduces the reader's uncertainty. Experiments on LoCoMo and PerLTQA across the three latest LLMs show consistent gains in answer quality and token efficiency.

## Full Text


<!-- PDF content starts -->

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Zhanghao Hu1Qinglin Zhu1Hanqi Yan1Yulan He1 2Lin Gui1
Abstract
Agent memory systems often adopt the standard
Retrieval-Augmented Generation (RAG) pipeline,
yet its underlying assumptions differ in this set-
ting. RAG targets large, heterogeneous corpora
where retrieved passages are diverse, whereas
agent memory is a bounded, coherent dialogue
stream with highly correlated spans that are often
duplicates. Under this shift, fixed top- ksimilarity
retrieval tends to return redundant context, and
post-hoc pruning can delete temporally linked pre-
requisites needed for correct reasoning. We argue
retrieval should move beyond similarity match-
ing and instead operate over latent components,
followingdecoupling to aggregation: disentan-
gle memories into semantic components, organ-
ise them into a hierarchy, and use this structure
to drive retrieval. We propose xMemory, which
builds a hierarchy of intact units and maintains a
searchable yet faithful high-level node organisa-
tion via a sparsity–semantics objective that guides
memory split and merge. At inference, xMemory
retrieves top-down, selecting a compact, diverse
set of themes and semantics for multi-fact queries,
and expanding to episodes and raw messages only
when it reduces the reader’s uncertainty. Experi-
ments on LoCoMo and PerLTQA across the three
latest LLMs show consistent gains in answer qual-
ity and token efficiency.
1. Introduction
Large language models (LLMs) are increasingly used as
agents that must remain coherent across long, multi-session
interactions (Kang et al., 2025; Liu et al., 2024; Wang et al.,
2024). To support long trajectory reasoning and person-
alisation, agent-memory systems store past dialogues and
events and retrieve them to answer future queries (Hu et al.,
2025). Despite rapid progress, most systems still follow a
1Department of Informatics, King’s College London, London,
United Kingdom2The Alan Turing Institute, London, United King-
dom. Correspondence to: Lin Gui<lin.1.gui@kcl.ac.uk>.
Preprint. February 3, 2026.
Figure 1.From similarity top- kto structured retrieval for agent
memory.Agent memory forms a coherent and highly correlated
stream, where many spans are near duplicates; similarity top- k
retrieval can therefore collapse and retrieve redundant chunks.
xMemory organises memories into a hierarchy of intact units and
performs structure-aware retrieval to produce a shorter but more
answer-sufficient context.
standard Retrieval-Augmented Generation (RAG) pipeline:
embed stored memories, retrieve a fixed top- kby similarity,
concatenate them into a context, and generate an answer
(Xu et al., 2025; Kang et al., 2025; Fang et al., 2025).
However, this direct adoption is problematic because agent
memory does not match the core RAG setting. RAG is
designed for large, heterogeneous corpora where retrieved
passages are diverse and the main failure mode is irrele-
vance (Gao et al., 2023). In contrast, an agent’s memory is a
bounded and coherent stream, so candidate spans are highly
correlated and often near duplicates. When retrieval relies
on similarity ranking with a fixed top- k, it can collapse into
a single dense region, returning redundant evidence and fail-
ing to separate what isneededfrom what is merelysimilar.
A common reaction is to prune or compress the retrieved
1arXiv:2602.02007v1  [cs.CL]  2 Feb 2026

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
context (Fang et al., 2025), but this is also brittle: pruning
modules developed under RAG assumptions depend on local
relevance cues (Pan et al., 2024b), whereas conversational
evidence is often temporally entangled through co-reference,
ellipsis, and timeline dependencies (Hu et al., 2026). As a
result, pruning may delete prerequisites within an evidence
chain rather than remove redundancy, leaving both fixed
top-kretrieval and post-retrieval pruning misaligned with
agent memory.
These mismatches call for a different goal: agent memory
retrieval should move beyond span matching and instead
surface the latent components of a coherent stream, enabling
evidence selection at the component level that avoids redun-
dancy by construction, as shown in Fig 1. In other words,
memory construction should followdecoupling to aggrega-
tion: we decompose the stream into semantic components,
organise them into higher-level structure, and then invert this
structure to drive retrieval. Under this view,retrieval is not
determined by a similarity ranking over raw spans, but by
the organisation induced by decomposition and aggregation.
For instance, two spans assigned to different components
are unlikely to be retrieved together, even if they are highly
similar in embedding space. This perspective raises two
questions: how to quantify desirable structural properties,
and how to use such signals to guide organisation. Con-
cretely, components should be sufficiently differentiated to
prevent redundancy concentration, and aggregations should
remain semantically faithful to support reliable evidence
reconstruction.
We meet these requirements with xMemory, which couples
memory structure managementwithadaptive retrieval
to control redundancy while preserving evidence integrity.
On the management side, we construct a four-level hierarchy
from raw messages to abstract units: anepisodesummarises
a contiguous message block,semanticsdistil reusable facts
from episodes to disentangle highly similar histories, and
themesgroup related semantics for high-level access. We
further optimise theme–semantic grouping during construc-
tion using a guidance objective that balances sparsity and
semantic coherence, since overly large themes enlarge the
candidate set and slow retrieval, while overly fragmented
themes weaken aggregation and reduce evidence coverage
when answers span multiple facts.
On the retrieval side, xMemory performstop-downretrieval
over the hierarchy. At the theme and semantic levels, queries
require evidence as a set rather than a single match: some
ask for a consolidated description that is distributed across
multiple semantics, and multi-hop questions often need
several connected semantics for chained reasoning. We
therefore select a small, diverse set of query-relevant themes
and semantics to broaden recall while staying anchored to
the query. At the episode and original message levels, wecontrol redundancy by admitting finer evidence only when
it decreases the reader’s uncertainty, and we stop expanding
once added detail no longer helps.
Our contributions can be summarised as follows:
1. We introduce a hierarchical memory construction that
disentangles highly similar histories into semantic units
and organises them into themes. This mitigates col-
lapsed similarity retrieval under RAG pipeline and is
guided by a sparsity–semantics objective that shapes
the high-level node structure during construction.
2.We propose a top-down retrieval framework that first
selects a compact set of relevant theme and semantic
units to broaden recall, and then controls fine-grained
episode and message expansion to reduce redundancy
while preserving intact evidence units.
3. We show consistent gains in answer quality and token
efficiency on LoCoMo and PerLTQA across diverse
open-source and closed-source LLM backbones, sup-
ported by evidence-density and coverage analyses that
explain the limitations of standard RAG retrieval and
pruning in agent memory.
2. Related Work
Memory Management in LLM Agents.Existing agent
memory systems broadly fall into flat context and struc-
tured designs. Flat approaches, such as MemGPT (Packer
et al., 2023), and MemoryOS (Kang et al., 2025), extend
effective context via paging controllers or stream-based stor-
age, but typically log raw dialogue or minimally processed
traces, which can accumulate redundancy and incur growing
retrieval and processing costs as histories lengthen. Struc-
tured systems organise memories into hierarchies or graphs
to improve coherence and navigability, e.g., MemoryBank
(Zhong et al., 2024), Zep (Rasmussen et al., 2025), and
A-Mem (Xu et al., 2025). However, many still rely on raw
or minimally processed text as the primary retrieval unit and
often expand or retrieve extensively across layers or commu-
nities at query time, which can introduce large contexts and
high overhead, sometimes offsetting the benefits of structure.
In contrast, our memory manager explicitly optimises the
structure during construction: a guidance objective jointly
promotes node sparsity to control redundant retrieval and se-
mantic organisation to preserve evidence structure and avoid
“semantic islands”, enabling efficient retrieval without brittle
pruning of temporally linked conversational evidence.
Retrieval and Context Management in Agent Memory.
Most agent-memory frameworks adopt the standard RAG
paradigm and retrieve a fixed top- kset of memories by em-
bedding similarity, which is suited to heterogeneous external
2

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Yeah, I painted last year!... 
You'd be a great counsellor!... 
Thank you! Is this your own painting?...Melanie believing Caroline would 
make an excellent counsellor……Caroline attended a 
counselling workshop.Melanie sent Caroline a 
hand -painted bowl.Caroline’s 
career.Melanie's 
painting journey. Theme
Semantics
Episode
Original 
messagesToo crowd ed!
Memory Merge
Too Sparse!Memory SplitMemory Structure Management
Memory Retrieval
Q: Would Caroline 
pursue painting 
as a career?1 Theme 
Multiple Semantic
Each Semantic
Exactly 1 Theme
1 Episode
Multiple Semantic
1 Message Block
1 EpisodeLoCoMo  Dataset:
～ 650 themes
～ 2900 semantics
～ 750 episodesNew Relation!
New Relation!New Node
New Node
Theme
Theme SemanticSemanticKNN Graph
Stage I: Representation Selection
EpisodeEpisode
Admitted: High Info Gain
Episode
Refused: Early Stopping(Optional) Original Message
Semantic
Episode
Message
Final Context
A: she wants to 
be a counsellor.
Stage II: Uncertainty InclusionSemantic
ThemeGuidance objective
Figure 2.Overview of xMemory.xMemory couples memory structuring with top-down retrieval to address the mismatch between agent
memory and the RAG pipeline. It organises a coherent stream into a hierarchy that disentangles episodic traces into semantic components
while preserving intact units. A sparsity–semantics objective guides split and merge to keep the high-level organisation searchable and
faithful. At retrieval, xMemory selects a diverse set of relevant themes and semantics to support aggregation reasoning, then expands to
episodes and raw messages only when they decrease the reader’s uncertainty, yielding a shorter context with stronger evidence coverage.
corpora but mismatched to conversational memory (Packer
et al., 2023; Kang et al., 2025; Xu et al., 2025). In agent
memory, the retrieval source is bounded and highly corre-
lated, often from the same character over time; fixed top- k
therefore tends to return redundant spans with shared back-
ground, inflating context length and distracting the reader
LLM. Meanwhile, generic context compression or pruning
methods developed for RAG long-context inference can be
brittle for chat memory, where key evidence is temporally
contiguous and relies on ellipsis; aggressive pruning may
fragment evidence chains and exacerbate long-context fail-
ures such as lost-in-the-middle (Pan et al., 2024a; Jiang et al.,
2024). In contrast, we select a compact evidence skeleton
via greedy submodular representative selection on the mem-
ory graph to promote relevance and coverage, and apply
uncertainty gated adaptive inclusion to control redundancy
under a budget without pruning inside evidence units.
3. Methodology
3.1. Problem Description and Overview
Problem Description.We study agent memory with long
message histories. Given a history Hand a query q, the
goal is to construct a context Cunder a budget that max-
imises answer quality while preserving evidence structure.
Unlike standard RAG, the evidence source is a bounded and
coherent stream, where candidate spans are near duplicates,
and key facts often rely on neighbouring turns. As a result,
similarity top- kretrieval tends to return redundant context,while post-hoc pruning is prone to breaking prerequisite
chains. We formulate retrieval as hierarchical unit selection
rather than ranking raw chunks by similarity. This yields
more distinguishable semantic units from highly correlated
histories and expands to fine-grained evidence only when
necessary. The resulting context Cis then consumed by a
reader LLM to generate the final answery.
Overview.We propose xMemory, which couples (i)mem-
ory structure managementthat disentangles highly corre-
lated histories into reusable components, and (ii)adaptive
retrievalthat selects evidence at the component level and
expands to raw text only when necessary. As shown in Fig-
ure 2, we first build a four-level hierarchy that preserves
intact units while disentangling reusable semantics from
redundant episodic traces. We then retrievetop-downover
this hierarchy: Stage I selects a compact, diverse set of
relevant themes and semantics to support queries whose
answers requireset-levelevidence or multi-hop reasoning;
Stage II admits episodes and original messages only when
they measurably improve the reader’s certainty, controlling
redundancy under a budget.
3.2. Memory Structure Management
Four-level hierarchy and mapping.To avoid collapsed
retrieval in a highly correlated stream, we organise mem-
ories into a four-level hierarchy that explicitly separates
episodic tracesfromreusable semantic components:Orig-
inal messages →Episode →Semantic →Theme. An
3

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
episode summarises a contiguous block of original mes-
sages, semantic nodes distil reusable long-term facts from
episodes to disentangle similar histories, and themes ag-
gregate related semantics to enable high-level access. This
hierarchy keeps intact evidence units at each granularity,
which protects temporally linked prerequisites from being
pruned away. Under this organisation, each original mes-
sage block maps to one episode; each episode can yield
multiple semantic nodes; each semantic node is assigned
to exactly one theme; and each theme can contain multiple
semantic nodes.
Guidance function for hierarchical organisation.Given
the hierarchy, the remaining question is how to maintain
a high-level organisation that is bothsearchableandfaith-
ful. The problem is: 1) When atheme becomes too large,
the candidate set expands, retrieval concentrates within a
dense neighbourhood, and redundant semantics are repeat-
edly selected, and 2) Whenthemes are overly fragmented,
aggregation across semantics weakens and evidence cover-
age drops for queries whose answers span multiple facts.
This motivation has an information-theoretic explanation:
any routing step that must choose among a large candidate
set under bounded discriminative evidence incurs a non-
trivial lower bound on the misrouting probability that grows
with the logarithm of the candidate set size (via Fano-type
arguments).1To balance these failure modes, we introduce
a guidance objective that governs theme splitting and merg-
ing. Let P={C k}K
k=1be a partition of Nsemantic nodes
intoKthemes, where nk=|C k|. We score the structure by
f(P) = SparsityScore(P) + SemScore(P).(1)
Sparsity.We quantify how balanced the theme partition
is via the expected within-theme candidate size. If we uni-
formly sample a semantic node from all Nnodes, it falls into
theme kwith probability nk/N. Approximating the within-
theme scanning cost by the theme size nk, the expected
cost is E[n cluster] =PK
k=1(nk/N)n k=1
NPK
k=1n2
k. We
convert this cost into a normalised reward:
SparsityScore(P) =N
K·1
E[n cluster]=N2
KPK
k=1n2
k.(2)
This score increases when theme sizes are more balanced,
which reduces dense region collapse and improves high-
level search efficiency.
Semantic.Beyond sparsity, we encourage semantically co-
herent themes while avoiding undesirable extremes in inter-
theme relations. For intra-theme organisation, we measure
cohesion by the average cosine similarity between seman-
tic embeddings and their theme centroid. For inter-theme
1See Appendix B for a concise discussion and its connection
to Eq. (1).relations, we regularise the nearest-neighbour similarity be-
tween theme centroids: overly similar themes increase re-
dundancy, while overly dissimilar themes form “semantic
islands” that hinder cross-topic navigation (Zhang et al.,
2025). Formally,
SemScore(P) =1
KKX
k=1 
1
nkX
i∈Ckcos(x i,µk)!
·g(s k),
g(sk) = exp
−(sk−m)2
2σ2
,
(3)
where xiis the embedding of semantic node i,µkis
the centroid embedding of theme k, and cos(·,·) denotes
cosine similarity. We define skas the nearest-neighbor
similarity of theme kin the centroid space, i.e., sk=
max j̸=kcos(µk,µj). We set m= median({s k})and
use the median absolute deviation as a robust scale, σ=
median({|s k−m|}) +ε , yielding a bell-shaped regularizer
that penalises both redundant themes and isolated themes.
Guided attach, split, and merge.When a new semantic
node arrives, we attach it to the most similar theme by
centroid similarity; otherwise we create a new theme. For an
overcrowded theme ( nkexceeding a threshold), we generate
candidate splits by clustering its semantic nodes and choose
the split that maximises f(P) . For tiny themes, we consider
merging them into a small set of nearest-neighbour themes
and select the merge that maximisesf(P).
kNN graph for high-level navigation.To support effi-
cient retrieval without quadratic maintenance, we maintain a
k-NN graph among theme and semantic nodes, storing only
top-ksimilarity edges per node. These links mitigate seman-
tic islands and enable efficient traversal during retrieval.
3.3. Adaptive Retrieval
Given a query q, we retrieve top-down over the hierarchy
to address the retrieval challenges in agent memory. At the
theme and semantic levels, answers often require set-level
evidence: queries may draw on multiple complementary
facts across semantics, and multi-hop reasoning needs sev-
eral connected semantics rather than a single best match.
At the episode and message levels, the main challenge is to
avoid redundant expansion while preserving intact evidence.
We therefore use a two-stage procedure: Stage I selects a
compact, diverse, and query-relevant high-level node selec-
tion; Stage II expands to episodic and raw text only when
the added detail is warranted.
Stage I: Query-aware representative selection on kNN
graphs.We begin with embedding-based retrieval to ob-
tain candidate semantic nodes Sqand their associated
themes Tq. To avoid collapsed selection from a single
4

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
dense region while still supporting set-level evidence, we
select representatives on the high-level kNN graph via a
greedy procedure that trades offcoverage gainandquery
relevance. Let Vbe the candidate nodes with neighbours
N(i) and edge similarity wiu>0 , and let R⊆V
be selected representatives. We maintain the covered set
C(R) ={u∈V| ∃r∈R, u∈ {r} ∪ N(r)} and newly
covered nodes ∆(i;R) = ({i}∪N(i))\C(R) . Each greedy
step selects
i⋆= arg max
i∈V\Rα·P
u∈∆(i;R) wiu
Z+ (1−α)·˜s(q, i),(4)
where ˜s(q, i)∈[0,1] is the normalised query–node similar-
ity score, and α∈[0,1] balances coverage and relevance.
Z >0 is a normalisation constant that keeps the cover-
age term on a comparable scale. We iteratively update
R←R∪{i⋆}and stop when the covered ratio |C(R)|/|V|
reaches a target threshold or when |R|hits a budget. We
apply Eq. 4 first on theme candidates and then on induced
semantic candidates, producing a compact backbone that
enables aggregation and multi-hop reasoning while reducing
redundant evidence.
Stage II: Uncertainty adaptive evidence inclusion.
From the selected semantics, we gather linked episodes
as intact evidence units. We rank episodes by combining di-
rect similarity to qwith semantic support, and build a coarse
context from selected themes and semantic summaries. We
then include episodes only when they sufficiently reduce the
reader’s predictive uncertainty; for included episodes, we
optionally expand to original messages when further uncer-
tainty reduction justifies the added tokens. We employ early
stopping when additional episodes fail to improve certainty.
This stage controls redundancy while preserving temporally
linked evidence within intact units, producing a compact
final contextCfor answer generation.
4. Experiments
4.1. Setups
Evaluation datasets.We evaluate long-term agent mem-
ory on two complementary benchmarks: LoCoMo (Maha-
rana et al., 2024) and PerLTQA (Du et al., 2024).LoCoMo
contains 50 multi-session dialogues averaging ∼9K tokens
and∼300 turns (up to 35 sessions), substantially longer
than earlier long-term dialogue datasets (Maharana et al.,
2024; Xu et al., 2022; Jang et al., 2023). We report results
on its four QA categories (single-hop, multi-hop, temporal,
open-domain) and follow prior work to omit the adversarial
subset.PerLTQAevaluates personal long-term memory
over a character’s life, with QA grounded in profiles, rela-
tionships, events, and dialogue history (Du et al., 2024).Evaluation Metrics.Following common practice on both
LoCoMo and PerLTQA (Maharana et al., 2024; Du et al.,
2024), we reportBLEU-1(Papineni et al., 2002) andtoken-
level F1as the main QA metrics. Since PerLTQA answers
are often full sentences, we additionally reportROUGE-
L(Lin, 2004), which captures sentence-level overlap via
longest common subsequence.
Baselines.We compare against five current baseline meth-
ods:(1) Naive RAG:We chunk original messages and
retrieve the top-20 chunks by vector-based searching to
form the context.(2) A-Mem:an agentic memory system
inspired by Zettelkasten that stores structured notes, dy-
namically links related memories, and updates the memory
network over time for retrieval (Xu et al., 2025).(3) Memo-
ryOS:a memory operating system for agents that performs
hierarchical memory storage and lifecycle management to
support long-horizon personalisation and coherence (Kang
et al., 2025).(4) LightMem:a lightweight, efficiency-
oriented memory framework that filters and organises in-
formation in multi-stage memory modules to reduce re-
trieval overhead while maintaining performance (Fang et al.,
2025).(5) Nemori:a self-organising agent memory archi-
tecture that segments conversational streams into coherent
episodes memories under cognitively inspired principles for
improved long-term recall (Nan et al., 2025).
Implementation details.We evaluate the frameworks
with three backbone LLMs: two recent open-source mod-
els, Qwen3-8B (Yang et al., 2025) and Llama-3.1-8B-
Instruct (Team, 2024), and a proprietary closed-source
model, GPT-5 nano. For answer generation, we use greedy
decoding (temperature = 0.0) to eliminate randomness and
ensure reproducibility (Sun et al., 2023). All retrieval
and memory construction embeddings are computed with
text-embedding-3-small . Since LoCoMo answers
are typically short phrases while PerLTQA answers are of-
ten full sentences, we use dataset-specific answer prompts
(Appendix C), but keep the final answering promptidenti-
cal across different baselineswithin each dataset for fair
comparison. Since GPT-5-nano could not return the out-
put logits, we apply GPT-4.1-mini to detect the entropy in
our framework when we apply GPT-5-nano. We set the
maximum number of semantic nodes per theme to 122. All
experiments are done with a NVIDIA A100 80G.
4.2. Main Results
LoCoMo: improved long-range conversational reason-
ing.Table 1 reports results on LoCoMo. xMemory
achieves the best average performance across all three back-
bones, indicating robust effectiveness across both open-
source and closed-source LLMs. WithQwen3-8B, xMem-
2Detailed analysis is in Appendix B.
5

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Table 1.Main results on LoCoMo. We report BLEU and F1 for each question category and the average. Token/query denotes the average
total tokens per query (lower is better). Best results within each backbone model are in bold.
Model Method Multi-hop Temporal Open-domain Single-hop Average Token
BLEU F1 BLEU F1 BLEU F1 BLEU F1 BLEU F1 /queryQwen3-8BNaive RAG 22.64 34.39 17.03 21.18 12.52 17.01 35.62 45.15 27.92 36.42 6613.17
Nemori 24.20 36.68 23.60 33.74 12.71 18.48 33.48 45.49 28.51 40.45 7754.66
LightMem 19.67 26.26 22.62 27.66 7.80 11.78 27.40 34.74 23.77 30.28 5545.35
A-Mem 9.20 9.06 22.46 32.99 9.01 8.57 23.01 23.28 19.49 21.78 9103.46
MemoryOS 17.12 21.74 26.84 32.26 14.53 16.25 35.83 40.37 29.20 33.76 7234.66
xMemory(Ours) 27.24 38.57 29.58 37.46 15.55 20.69 40.94 50.94 34.48 43.98 4711.29Llama-3.1-8B-InsNaive RAG 13.77 19.40 6.20 8.61 6.99 10.22 21.90 35.37 16.21 25.30 11522.60
Nemori 18.18 26.30 19.23 26.09 9.24 12.06 26.19 40.18 22.21 32.95 14802.69
LightMem 16.43 23.76 12.44 15.61 9.83 13.70 21.68 25.08 18.05 22.15 5708.11
A-Mem 15.88 20.90 19.92 24.91 9.92 11.34 24.86 39.90 21.52 31.51 10268.77
MemoryOS 13.26 17.83 19.88 23.95 11.48 12.92 24.76 28.67 20.81 24.72 7212.07
xMemory(Ours) 22.21 30.99 21.20 27.42 11.58 14.61 28.43 41.15 24.73 34.77 5539.97GPT-5 nanoNaive RAG 21.64 32.11 38.43 49.05 22.30 26.42 36.18 46.04 33.12 42.89 7531.46
Nemori 24.80 37.11 41.56 54.25 22.61 29.29 40.35 51.67 36.65 48.17 9154.76
LightMem 25.33 34.19 43.37 57.12 23.57 29.45 39.01 45.45 36.45 44.82 6850.04
A-Mem 14.60 21.31 31.00 38.58 14.27 16.78 24.04 28.59 23.15 28.60 9610.94
MemoryOS 14.43 18.43 26.76 30.91 17.00 18.74 31.74 35.38 26.57 30.27 7029.02
xMemory(Ours) 27.56 39.97 46.10 57.62 25.53 30.92 41.14 52.63 38.71 50.00 6581.20
ory improves the averageBLEUfrom 28.51 (Nemori) to
34.48and the averageF1from 40.45 to43.98. The gains
are especially evident on long-range reasoning, e.g.,tem-
poralincreases from BLEU 23.60 and F1 33.74 to29.58
and37.46. WithGPT-5 nano, xMemory improves the av-
erageBLEUfrom 36.65 to38.71andF1from 48.17 to
50.00, while reducing token usage from 9155 to6581per
query. These indicate that our framework can provide suf-
ficient evidence without retrieving redundant spans. We
also observe limitations of existing agent memory systems.
Methods such as A-Mem and MemoryOS rely on LLM-
generated structured memory records withstrict schema
constraints, where formatting deviations can cascade into
failed updates and unstable reproduction (Xu et al., 2025;
Packer et al., 2023; Kang et al., 2025). Methods such as
LightMem further integrate generic prompt pruners such as
LLMLingua-2 (Pan et al., 2024b), which are developed un-
der the standard RAG assumption that retrieved passages are
diverse and noise patterns are separable. In agent memory,
however, candidates come from a single coherent dialogue
stream and are often nearly duplicated, while key evidence
is temporally entangled across adjacent turns. Under this
distribution shift, pruning can remove latent prerequisites
and fragment evidence chains, which aligns with LightMem
trailing xMemory on LoCoMo, especially for multi-hop
and temporal questions in Table 1. Overall, the results sup-
port our thesis that agent memory needs redundancy controlwithoutbreaking evidence into fragmentation.
PerLTQA: generalisation to personal long-term memory.
Table 2 reports results on PerLTQA, which requires rea-
soning over richer, lifelong memories and typically expects
sentence-level answers. xMemory remains effective in this
setting across different backbones. WithQwen3-8B, xMem-
ory achieves an averageBLEUof36.24,F1of47.08, and
ROUGE-Lof42.50, while using5087tokens per query;
it surpasses strong structured-memory baselines such as
MemoryOS with BLEU 35.14, F1 42.35, and ROUGE-L
38.48. WithGPT-5 nano, xMemory further reachesF1
46.23andROUGE-L 41.25, improving over Nemori with
BLEU F1 41.79 and ROUGE-L 38.43. These results sug-
gest that the xMemory retrieval principles transfer beyond
dialogue recall to broader personal memory reasoning.
Efficiency.We reporttokens per queryas the average
end-to-end token consumption per query during inference,
including retrieval, final answer generation, and any aux-
iliary calls. Across both open-source and closed-source
backbones, xMemory improves accuracy while reducing
token usage (Tables 1 and 2). OnLoCoMowithQwen3-
8B, xMemory lowers tokens per query from the most ex-
pensive baselineA-Mem 9103) to4711, while achieving
the best overall performance. The same trend holds for
GPT-5 nano, from9155to6581, suggesting that xMem-
6

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Table 2.Main results on PerLTQA. We report BLEU, F1, ROUGE-
L, and token usage per query (lower is better). Best results within
each backbone model are in bold.
Model Method BLEU F1 ROUGE-LToken/
queryQwen3-8BNaive RAG 32.08 41.37 35.95 6274.38
Nemori 32.55 42.80 38.05 9091.72
LightMem 16.97 32.41 29.87 7692.05
A-Mem 28.62 34.99 33.54 9864.32
MemoryOS 35.14 42.35 38.48 6499.47
xMemory36.24 47.08 42.50 5087.18Llama-3.1-8B-InsNaive RAG 33.67 44.84 39.65 9531.30
Nemori 41.01 49.62 44.65 11440.04
LightMem 23.47 35.93 33.655452.49
A-Mem 11.40 13.28 15.90 7707.40
MemoryOS 35.14 42.35 38.48 6511.11
xMemory42.68 52.37 47.846066.40GPT-5 nanoNaive RAG 27.35 37.76 32.35 10756.39
Nemori 33.44 41.79 38.43 11882.70
LightMem 14.04 28.30 26.11 7579.27
A-Mem 33.12 41.17 37.94 14717.77
MemoryOS 27.66 33.83 31.44 12669.39
xMemory36.79 46.23 41.25 7307.30
ory retrieval pipeline mitigates the redundancy induced by
fixed top- kretrieval in RAG pipeline without harming an-
swer quality. OnPerLTQA, we observe similar efficiency
gains on both open-source and closed-source LLMs. While
xMemory is not always the cheapest option, e.g., LightMem
with pruning model uses fewer tokens than xMemory un-
der Llama-3.1-8B-Instruct on PerLTQA. This comes with
a large performance drop, such as BLEU decreasing from
42.68to23.47(Table 2), consistent with our earlier discus-
sion that RAG-trained pruning can be brittle when evidence
is entangled in agent memory.
5. Analysis
5.1. Ablation Studies
We conduct an ablation study to quantify the contribution
of each component. Figure 3 reports five settings on Lo-
CoMo with Qwen3-8B: (i)Naive RAGchunks raw mes-
sages and retrieves top- kchunks; (ii)Memory-onlyuses
our hierarchical memory structure but applies basic similar-
ity retrieval without our search framework; (iii)+RepSel
adds Stage I (query-aware representative selection on the
high-level node kNN graph) to improve relevance and di-
versity; (iv)+UncSionadds Stage II (uncertainty adaptive
evidence inclusion) to control low-level node expansion;
and (v)Ours (Full)combines both modules (Full perfor-
mance shown in Appendix A.1). Using the hierarchical
memory alone already yields a large gain over Naive RAG;
Figure 3.Ablation on LoCoMo with Qwen3-8B.We report
BLEU and F1 on the left axis and Token/query on the right axis
(lower is better).Memory-onlyuses our hierarchical memory
structure with a simple similarity retriever, but disables both adap-
tive retrieval stages.+RepSeladds Stage I, which selects repre-
sentative theme and semantic nodes on the high-level kNN graph.
+UncSionadds Stage II, which admits episodic and original evi-
dence only when it reduces the reader’s uncertainty.Ours (Full)
combines both stages.
for example, average BLEU increases from 27.92 to 31.81
and F1 from 36.42 to 40.77, indicating that structured, intact
units are more evidence managed than chunked passages.
Adding either RepSel or UncSion further improves perfor-
mance, and combining both performs best, suggesting the
two stages are complementary: RepSel improves seman-
tic alignment and coverage at the high level node, while
UncSion refines low-level evidence inclusion. In terms of
efficiency, the Memory-only variant is the most expensive
among our variants, with 7.24k tokens per query, consistent
with over-fetching redundant evidence under RAG similar-
ity retrieval; RepSel reduces cost by selecting representative
high-level nodes (6.32k tokens per query), while UncSion
adaptively halts expansion when uncertainty reduces. The
full system achieves the trade-off, reducing token usage to
the least while also improving accuracy over all ablations.
5.2. Evidence Density under RAG and Pruning Effect
We compare three settings: (i) naive RAG with top- 20
retrieved passage chunks, (ii) RAG augmented with the
LLMLingua-2 pruning module (Pan et al., 2024b) as in
LightMem (Fang et al., 2025), and (iii) our structured mem-
ory retrieval. For each query, we remove stopwords from the
reference answer and treat the remaining content words as
minimal evidence units. A retrieved block is1-hitif it con-
tains exactly one distinct answer token,2-hitif it contains
two, andmulti-hitif it contains at least three.
Figure 4 shows that xMemory yields more evidence-dense
retrieval across categories, with substantially higher 2-hit
and multi-hit proportions than both RAG baselines. In con-
trast, pruning tends to shift mass from 2-hit and multi-hit
to 1-hit, indicating that compression often removes answer-
bearing details even when the retrieved context is broadly
relevant. For example, on multi-hop questions (Cat. 1), prun-
7

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Figure 4.Evidence hit distribution by question category. Each
subplot shows the proportions of retrieved blocks that are 1-hit, 2-
hit, or multi-hit with respect to answer content tokens, comparing
Naive RAG, RAG with pruning, and xMemory. Higher 2-hit and
multi-hit indicate more evidence-dense retrieval under xMemory.
Table 3.Performance and evidence coverage efficiency.Perf.
reports end-to-end BLEU-1/F1.Avg. blocks for coverageis the
average number of retrieved blocks required to cover all answer
content tokens, andAvg. tokens for coverageis the token cost.
Method Perf. Avg. blocks↓Avg. tokens↓
(BLEU/F1) for coverage for coverage
Naive RAG 27.92 / 36.42 10.81 1979.26
RAG + Pruning 26.55 / 34.58 13.31 1587.99
Ours 34.48 / 43.98 5.66 974.56
ing reduces the 2-hit and multi-hit proportions from 7.82%
and 6.53% (Naive RAG) to 6.67% and 6.29%, whereas
xMemory increases them to 13.14% and 12.19%.
We further evaluateevidence coverage efficiencyby mea-
suring the minimum number of retrieved blocks required to
cover all answer content tokens, together with their token
cost for each query (Full Performance shown on Appendix
A.2). Table 3 confirms the same trend. Pruning reduces
the token cost for coverage, but requires more blocks and
yields lower accuracy than Naive RAG, consistent with los-
ing evidence density. In contrast, xMemory achieves higher
accuracy while covering the evidence with fewer blocks
and about half the tokens, indicating more answer-sufficient
retrieval under a tighter budget.
5.3.Retroactive Restructuring in Memory Construction
Agent memory differs from standard RAG in that its ev-
idence source evolves over time: facts and relations can
be updated as new interactions occur, rather than remain-
ing static as in most retrieval corpora under RAG settings.
Therefore, memory construction should support revising the
high-level structure, updating node relations to reflect newly
Figure 5.Structural plasticity vs. downstream QA.We report
Avg. BLEU/F1 (left axis) and thedynamic reassignment ratio
(right axis) for four construction settings. Freezing high-level
restructuring (w/o merge&split) yields 0% reassignment and lower
accuracy, while the full system performs substantial reassignment
(44.91%) and achieves the best Avg. BLEU/F1.
revealed or corrected information. Our memory manager
enables such retroactive restructuring: previously stored
semantic nodes can be reassigned to different themes when
subsequent insertions trigger themesplitormerge(Sec. 3.2).
We measure this effect using thedynamic reassignment ra-
tio, defined as the fraction of semantic nodes whose theme
assignment changes during construction.
Figure 5 shows that retroactive restructuring improves down-
stream performance. Disabling bothsplitandmergefreezes
the high-level structure, yields 0% reassignment, and re-
duces the average F1 to 38.59. Enabling restructuring
substantially increases reassignment and accuracy, and the
full system achieves both the highest reassignment ratio
(44.91%) and the best average F1 (43.98). Moreover,split
induces more reassignment thanmerge, indicating that split-
ting overcrowded themes is the primary source of structural
revision. However,mergeremains important for forming
a stable high-level structure: without merge, the hierarchy
retains far more themes (1122 vs. 642 in the full system)
despite a similar scale of semantic nodes, whereas enabling
merge consolidates redundant themes and yields a more
compact organisation for retrieval (Table 7). Detailed re-
sults and structure statistics are reported in Appendix A.3.
6. Conclusion
We presented xMemory, a retrieval framework that uses hier-
archical organisation to address the mismatch between agent
memory and standard RAG assumptions. By disentangling
highly correlated memories into semantic components and
retrieving top-down over the resulting hierarchy, xMemory
reduces redundancy while preserving temporally linked evi-
dence. Across LoCoMo and PerLTQA, xMemory improves
answer quality with lower token cost and retrieves more
evidence-dense contexts than RAG baselines.
8

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Impact Statement
The paper advances agent-memory retrieval for long-
horizon interactions by reducing redundant context while
preserving temporally linked evidence. The techniques may
improve reliability and efficiency in applications such as
personal assistants and multi-session decision support, and
can reduce compute cost by lowering inference token usage.
Potential risks include privacy leakage or unintended reten-
tion of sensitive user information if deployed with real user
data, as well as amplified harm if an agent retrieves and re-
inforces outdated or incorrect memories. Our method does
not introduce new data collection and is model-agnostic, but
practical deployments should incorporate consent, data min-
imisation, access control, and deletion policies, and should
provide mechanisms to correct or override stored memories.
Overall, we do not anticipate societal impacts beyond those
commonly associated with retrieval-augmented LLM sys-
tems, but responsible use requires careful privacy and safety
safeguards.
References
Du, Y ., Wang, H., Zhao, Z., Liang, B., Wang, B., Zhong,
W., Wang, Z., and Wong, K.-F. PerLTQA: A personal
long-term memory dataset for memory classification, re-
trieval, and fusion in question answering. In Wong, K.-F.,
Zhang, M., Xu, R., Li, J., Wei, Z., Gui, L., Liang, B.,
and Zhao, R. (eds.),Proceedings of the 10th SIGHAN
Workshop on Chinese Language Processing (SIGHAN-
10), pp. 152–164, Bangkok, Thailand, August 2024. As-
sociation for Computational Linguistics. URL https:
//aclanthology.org/2024.sighan-1.18/.
Fang, J., Deng, X., Xu, H., Jiang, Z., Tang, Y ., Xu, Z.,
Deng, S., Yao, Y ., Wang, M., Qiao, S., Chen, H., and
Zhang, N. Lightmem: Lightweight and efficient memory-
augmented generation.CoRR, abs/2510.18866, 2025.
doi: 10.48550/ARXIV .2510.18866. URL https://
doi.org/10.48550/arXiv.2510.18866.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y .,
Dai, Y ., Sun, J., Guo, Q., Wang, M., and Wang, H.
Retrieval-augmented generation for large language mod-
els: A survey.CoRR, abs/2312.10997, 2023. doi:
10.48550/ARXIV .2312.10997. URL https://doi.
org/10.48550/arXiv.2312.10997.
Hu, C., Gao, X., Zhou, Z., Xu, D., Bai, Y ., Li, X., Zhang,
H., Li, T., Zhang, C., Bing, L., et al. Evermemos: A self-
organizing memory operating system for structured long-
horizon reasoning.arXiv preprint arXiv:2601.02163,
2026.
Hu, Y ., Liu, S., Yue, Y ., Zhang, G., Liu, B., Zhu, F., Lin, J.,Guo, H., Dou, S., Xi, Z., et al. Memory in the age of ai
agents.arXiv preprint arXiv:2512.13564, 2025.
Jang, J., Boo, M., and Kim, H. Conversation chroni-
cles: Towards diverse temporal and relational dynam-
ics in multi-session conversations. In Bouamor, H.,
Pino, J., and Bali, K. (eds.),Proceedings of the 2023
Conference on Empirical Methods in Natural Language
Processing, EMNLP 2023, Singapore, December 6-
10, 2023, pp. 13584–13606. Association for Compu-
tational Linguistics, 2023. doi: 10.18653/V1/2023.
EMNLP-MAIN.838. URL https://doi.org/10.
18653/v1/2023.emnlp-main.838.
Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C., Yang, Y ., and
Qiu, L. Longllmlingua: Accelerating and enhancing llms
in long context scenarios via prompt compression. In Ku,
L., Martins, A., and Srikumar, V . (eds.),Proceedings of
the 62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), ACL 2024,
Bangkok, Thailand, August 11-16, 2024, pp. 1658–1677.
Association for Computational Linguistics, 2024. doi:
10.18653/V1/2024.ACL-LONG.91. URL https://
doi.org/10.18653/v1/2024.acl-long.91.
Kang, J., Ji, M., Zhao, Z., and Bai, T. Memory OS of
AI agent. In Christodoulopoulos, C., Chakraborty, T.,
Rose, C., and Peng, V . (eds.),Proceedings of the 2025
Conference on Empirical Methods in Natural Language
Processing, pp. 25961–25970, Suzhou, China, November
2025. Association for Computational Linguistics. ISBN
979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
1318. URL https://aclanthology.org/2025.
emnlp-main.1318/.
Langley, P. Crafting papers on machine learning. In Langley,
P. (ed.),Proceedings of the 17th International Conference
on Machine Learning (ICML 2000), pp. 1207–1216, Stan-
ford, CA, 2000. Morgan Kaufmann.
Lin, C.-Y . ROUGE: A package for automatic evalua-
tion of summaries. InText Summarization Branches
Out, pp. 74–81, Barcelona, Spain, July 2004. Asso-
ciation for Computational Linguistics. URL https:
//aclanthology.org/W04-1013/.
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts.Transactions of
the Association for Computational Linguistics, 12:157–
173, 2024. doi: 10.1162/tacl a00638. URL https:
//aclanthology.org/2024.tacl-1.9/.
Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri,
F., and Fang, Y . Evaluating very long-term conversa-
tional memory of LLM agents. In Ku, L., Martins, A.,
and Srikumar, V . (eds.),Proceedings of the 62nd Annual
9

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), ACL 2024, Bangkok, Thai-
land, August 11-16, 2024, pp. 13851–13870. Associa-
tion for Computational Linguistics, 2024. doi: 10.18653/
V1/2024.ACL-LONG.747. URL https://doi.org/
10.18653/v1/2024.acl-long.747.
Nan, J., Ma, W., Wu, W., and Chen, Y . Nemori: Self-
organizing agent memory inspired by cognitive science.
CoRR, abs/2508.03341, 2025. doi: 10.48550/ARXIV .
2508.03341. URLhttps://doi.org/10.48550/
arXiv.2508.03341.
Packer, C., Fang, V ., Patil, S. G., Lin, K., Wooders, S., and
Gonzalez, J. E. Memgpt: Towards llms as operating
systems.CoRR, abs/2310.08560, 2023. doi: 10.48550/
ARXIV .2310.08560. URL https://doi.org/10.
48550/arXiv.2310.08560.
Pan, Z., Wu, Q., Jiang, H., Xia, M., Luo, X., Zhang, J.,
Lin, Q., R ¨uhle, V ., Yang, Y ., Lin, C., Zhao, H. V ., Qiu,
L., and Zhang, D. Llmlingua-2: Data distillation for
efficient and faithful task-agnostic prompt compression.
In Ku, L., Martins, A., and Srikumar, V . (eds.),Find-
ings of the Association for Computational Linguistics,
ACL 2024, Bangkok, Thailand and virtual meeting, Au-
gust 11-16, 2024, pp. 963–981. Association for Com-
putational Linguistics, 2024a. doi: 10.18653/V1/2024.
FINDINGS-ACL.57. URL https://doi.org/10.
18653/v1/2024.findings-acl.57.
Pan, Z., Wu, Q., Jiang, H., Xia, M., Luo, X., Zhang, J.,
Lin, Q., R ¨uhle, V ., Yang, Y ., Lin, C.-Y ., Zhao, H. V .,
Qiu, L., and Zhang, D. LLMLingua-2: Data distil-
lation for efficient and faithful task-agnostic prompt
compression. In Ku, L.-W., Martins, A., and Sriku-
mar, V . (eds.),Findings of the Association for Computa-
tional Linguistics: ACL 2024, pp. 963–981, Bangkok,
Thailand, August 2024b. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2024.findings-acl.
57. URL https://aclanthology.org/2024.
findings-acl.57/.
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: a
method for automatic evaluation of machine translation.
In Isabelle, P., Charniak, E., and Lin, D. (eds.),Proceed-
ings of the 40th Annual Meeting of the Association for
Computational Linguistics, pp. 311–318, Philadelphia,
Pennsylvania, USA, July 2002. Association for Computa-
tional Linguistics. doi: 10.3115/1073083.1073135. URL
https://aclanthology.org/P02-1040/.
Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and
Chalef, D. Zep: A temporal knowledge graph architecture
for agent memory.CoRR, abs/2501.13956, 2025. doi:
10.48550/ARXIV .2501.13956. URL https://doi.
org/10.48550/arXiv.2501.13956.Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z.,
Yin, D., and Ren, Z. Is chatgpt good at search? in-
vestigating large language models as re-ranking agents.
In Bouamor, H., Pino, J., and Bali, K. (eds.),Proceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2023, Singapore,
December 6-10, 2023, pp. 14918–14937. Association
for Computational Linguistics, 2023. doi: 10.18653/V1/
2023.EMNLP-MAIN.923. URL https://doi.org/
10.18653/v1/2023.emnlp-main.923.
Team, L. The llama 3 herd of models.CoRR,
abs/2407.21783, 2024. doi: 10.48550/ARXIV .2407.
21783. URL https://doi.org/10.48550/
arXiv.2407.21783.
Wang, T., Tao, M., Fang, R., Wang, H., Wang, S., Jiang,
Y . E., and Zhou, W. AI PERSONA: towards life-long
personalization of llms.CoRR, abs/2412.13103, 2024.
doi: 10.48550/ARXIV .2412.13103. URL https://
doi.org/10.48550/arXiv.2412.13103.
Xu, J., Szlam, A., and Weston, J. Beyond goldfish memory:
Long-term open-domain conversation. In Muresan, S.,
Nakov, P., and Villavicencio, A. (eds.),Proceedings of
the 60th Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), ACL 2022,
Dublin, Ireland, May 22-27, 2022, pp. 5180–5197. As-
sociation for Computational Linguistics, 2022. doi: 10.
18653/V1/2022.ACL-LONG.356. URL https://doi.
org/10.18653/v1/2022.acl-long.356.
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y .
A-mem: Agentic memory for LLM agents. InThe Thirty-
ninth Annual Conference on Neural Information Pro-
cessing Systems, 2025. URL https://openreview.
net/forum?id=FiM0M8gcct.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D.,
Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H.,
Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J.,
Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L.,
Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang,
P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T.,
Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren,
X., Fan, Y ., Su, Y ., Zhang, Y ., Zhang, Y ., Wan, Y ., Liu,
Y ., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z.
Qwen3 technical report.CoRR, abs/2505.09388, 2025.
doi: 10.48550/ARXIV .2505.09388. URL https://
doi.org/10.48550/arXiv.2505.09388.
Zhang, Y ., Wu, R., Cai, P., Wang, X., Yan, G., Mao, S.,
Wang, D., and Shi, B. Leanrag: Knowledge-graph-based
generation with semantic aggregation and hierarchical
retrieval.arXiv preprint arXiv:2508.10391, 2025.
10

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y . Memo-
rybank: Enhancing large language models with long-term
memory. In Wooldridge, M. J., Dy, J. G., and Natara-
jan, S. (eds.),Thirty-Eighth AAAI Conference on Artifi-
cial Intelligence, AAAI 2024, Thirty-Sixth Conference on
Innovative Applications of Artificial Intelligence, IAAI
2024, Fourteenth Symposium on Educational Advances
in Artificial Intelligence, EAAI 2014, February 20-27,
2024, Vancouver, Canada, pp. 19724–19731. AAAI Press,
2024. doi: 10.1609/AAAI.V38I17.29946. URL https:
//doi.org/10.1609/aaai.v38i17.29946.
11

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Table 4.Detailed ablation results on LoCoMo with Qwen3-8B.We report BLEU and F1 for each question category and the average.
Token/query denotes the average total tokens per query (lower is better).
Setting Multi-hop Temporal Open-domain Single-hop Average Token
BLEU F1 BLEU F1 BLEU F1 BLEU F1 BLEU F1 /query
Naive RAG 22.64 34.39 17.03 21.18 12.52 17.01 35.62 45.15 27.92 36.42 6613.17
Memory-only 24.48 35.41 27.28 34.14 13.20 17.75 38.19 47.73 31.81 40.77 7235.56
+RepSel 26.55 38.02 28.08 35.72 14.30 19.75 38.36 45.79 32.55 40.64 6320.72
+UncSion 26.64 37.03 28.92 37.09 14.10 20.43 39.28 49.01 33.23 42.55 6556.58
xMemory (Full)27.24 38.57 29.58 37.46 15.55 20.69 40.94 50.94 34.48 43.98 4711.29
Table 5.Expanded results for evidence density and coverage efficiency on LoCoMo (Qwen3-8B).We report BLEU and F1 for each
question category and the average.Avg. blocks for coverageis the average number of retrieved blocks required to cover all answer content
tokens, andCover token cost/queryis the corresponding token cost (lower is better).
Framework Multi-hop Temporal Open-domain Single-hop AverageAvg. blocks↓
for coverageAvg. tokens↓
for coverage
BLEU F1 BLEU F1 BLEU F1 BLEU F1 BLEU F1
Naive RAG 22.64 34.39 17.03 21.18 12.52 17.01 35.62 45.15 27.92 36.42 10.81 1979.26
RAG + Pruning 23.81 33.58 16.53 23.33 12.92 16.22 32.85 41.23 26.55 34.58 13.31 1587.99
xMemory (Ours)27.24 38.57 29.58 37.46 15.55 20.69 40.94 50.94 34.48 43.98 5.66 974.56
A. More Experiment Results
A.1. Ablation Study Full Results and Observations
Table 4 reports category-level results for the five ablation settings on LoCoMo with Qwen3-8B. Moving from Naive RAG
to Memory-only yields consistent gains across all categories, with the largest improvements on temporal and single-hop
questions, suggesting that storing intact hierarchical units already improves evidence accessibility compared to chunked
retrieval. Adding RepSel mainly benefits multi-hop and open-domain questions, where selecting representative high-level
nodes improves coverage and reduces redundancy from dense neighbourhood retrieval. Adding UncSion provides broader
gains, especially on temporal and single-hop questions, indicating that uncertainty-based inclusion helps admit fine-grained
evidence only when it is useful, while avoiding unnecessary expansion. Combining both modules achieves the best results in
every category, improving the average from 31.81/40.77 (Memory-only) to 34.48/43.98 on BLEU/F1, and substantially
reducing token cost from 7236 to 4711 tokens per query. Overall, the two stages are complementary: RepSel improves
high-level selection and coverage, while UncSion refines low-level inclusion and budget usage.
A.2. Full Performance in Evidence Density under RAG and Pruning Effect
Table 5 provides the detailed performance and coverage statistics for the three settings on LoCoMo with Qwen3-8B. RAG
with pruning shows mixed behaviour across categories: while it slightly improves BLEU on multi-hop and open-domain
questions, it reduces F1 and consistently underperforms Naive RAG on single-hop and the overall average. This is consistent
with Figure 4, where pruning shifts retrieved blocks from 2-hit and multi-hit toward 1-hit, suggesting that compression often
discards answer-bearing details. In contrast, xMemory improves both BLEU and F1 in every category, with particularly
large gains on temporal and single-hop questions, and achieves the highest overall average. It also covers answer evidence
more efficiently, requiring fewer blocks for full coverage (5.66 vs. 10.81 and 13.31) and substantially lower coverage token
cost (974.56 vs. 1979.26 and 1587.99), indicating more answer-sufficient retrieval under a tighter budget.
A.3. Full Performance of Retroactive Restructuring in Memory Construction
Extended results.Table 6 provides category-level QA results together with the dynamic reassignment ratio for different
construction settings. Table 7 reports the resulting numbers of themes and semantic nodes.
12

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Table 6.Extended results for retroactive restructuring on LoCoMo (Qwen3-8B).We report BLEU/F1 by question category and the
average, together with the dynamic reassignment ratio under each construction setting.
Setting Multi-hop Temporal Open-domain Single-hop AverageReassign
ratio
BLEU F1 BLEU F1 BLEU F1 BLEU F1 BLEU F1
w/o merge&split 24.01 34.35 28.24 34.90 10.97 14.04 37.24 44.23 31.30 38.59 0.00%
w/o split 24.46 35.46 28.55 34.77 11.61 15.75 37.69 44.79 31.73 39.18 17.33%
w/o merge 24.81 35.46 29.68 35.99 13.09 18.09 38.45 46.16 32.54 40.33 26.23%
xMemory (Full)27.24 38.57 29.58 37.46 15.55 20.69 40.94 50.94 34.48 43.98 44.91%
Table 7.Structure statistics under different construction settings.We report the resulting numbers of themes and semantic nodes.
Setting #Themes #Semantics
w/o merge&split 1114 2876
w/o split 651 2897
w/o merge 1122 2928
Full 642 2879
Summary.Disabling both operators yields a static hierarchy with 0% reassignment and the lowest overall accuracy.
Allowing split or merge enables retroactive reassignment and improves QA, while the full system achieves the highest
reassignment ratio and the best accuracy across all categories. Split accounts for a larger share of reassignment, whereas
merge is important for consolidating redundant themes and producing a compact high-level organisation, as reflected by the
reduced number of themes under the full setting.
B. Theoretical Analysis of Branching Factor in Hierarchical Dialogue Summarization
B.1. Motivation and Fano Inequality
Our hierarchy performs top-down retrieval, where key subroutines repeatedlyrouteto a relevant unit by choosing from a
candidate set (e.g., selecting a theme/semantic relevant to the current query context). In high-similarity dialogue streams, the
observable signal used for routing (summaries/embeddings/LLM scores) can be inherently weak in its ability to discriminate
among many near-duplicate candidates. This subsection summarizes a standard information-theoretic implication: with
bounded discriminative information, the error of a multi-way routing decision cannot be made arbitrarily small as the
candidate set grows.
Consider one such routing decision that must identify the correct option among nkcandidates on semantic-theme level. Let
Z∈ {1, ..., n k}be the (unknown) index of the correct candidate (It is binary label indicates whether this semantic is correctly
linked with the theme, and nkis the number of semantics under the theme) and let Odenote the observable evidence
(summarized theme based on semantics) used for routing (e.g., derived from the query context and stored representations).
Any routing rule outputs ˆZ(O)with error probabilityp e= Pr[ ˆZ̸=Z]. A classical result (Fano’s inequality) implies
pe≥1−I(Z;O) + 1
log2nk,(5)
where I(Z;O) denotes the mutual information between the routing evidence O(derived from the query/context and stored
representations) and the correct candidate index Z. In our setting, candidates within the same theme are often near-duplicates,
and the evidence is produced from compressed LLM representations (with additional noise from imperfect assignments). As
a result, thediscriminativeinformation available for distinguishing among candidates is bounded, i.e., I(Z;O) is small.
By Fano’s inequality, a bounded I(Z;O) implies a non-trivial lower bound on the routing error that increases with logn k.
Therefore, to achieve a low misrouting rate (small pe), it is necessary to control the candidate set size nk. This aligns with
the practical intuition: when many candidates are highly similar, identifying the source becomes close to guessing unless we
reduce the arity of the routing problem.
13

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Table 8.Effect of the theme size cap on LoCoMo (Qwen3-8B).We vary the upper bound of semantic nodes per theme and report
BLEU/F1 by question category and the average, together with resulting structure statistics.
Upper bound Multi-hop Temporal Open-domain Single-hop Average#
Sem.#
ThemesAvg. sem.
per theme
BLEU F1 BLEU F1 BLEU F1 BLEU F1 BLEU F1
14 25.32 36.23 27.68 35.21 12.88 17.89 38.56 47.32 32.26 40.93 2867 385 7.45
1227.24 38.57 29.58 37.46 15.55 20.69 40.94 50.94 34.48 43.982897 642 4.48
10 25.62 35.80 27.56 35.66 13.22 18.12 38.67 46.26 32.37 40.38 2945 818 3.60
8 23.58 34.25 26.34 33.62 10.83 13.81 37.22 44.08 30.81 38.21 2856 1109 2.57
B.2. Connection to Metric Designing
Equation (5)shows that if I(Z;O) is bounded (as expected when many candidates are highly similar), then increasing nk
necessarily increases a lower bound on the misrouting probability.
In our theme partition P={C k}K
k=1ofNsemantic nodes with nk=|C k|, the typical within-theme candidate set
size directly governs the arity of routing within a theme. Eq. (2) estimates the expected within-theme scanning cost as
E[ncluster ] =1
NPK
k=1n2
k, and defines SparsityScore(P) =N2
KPK
k=1n2
k. Maximizing SparsityScore(P) thus controls the
typical within-theme candidate size (and prevents extremely large themes), which is precisely the regime where the lower
bound in Eq. (5)becomes prohibitive. This provides a theoretical motivation for balancing theme sizes: it limits unavoidable
routing errors under bounded discriminative evidence in high-similarity dialogue memory.
B.3. Fano-style Lower Bound and optimaln kin hirachical structure.
In this subsection, we use Fano-style lower bound to guide us to find the optimaln k.
Theorem B.1.Let Z∈ {1, . . . , n k}denote the (unknown) correct index and let Obe the observable evidence used to infer
Z. For any estimator ˆZ=g(O)with error probabilityp e= Pr[ ˆZ̸=Z],
H(Z|O)≤h(p e) +p elog2(nk−1),(6)
whereh(·)is the binary entropy function. In particular, ifZis uniform on{1, . . . , k}, then
pe≥1−I(Z;O) + 1
log2nk.(7)
Proof. Equation (6)is the standard form of Fano’s inequality. Using I(Z;O) =H(Z)−H(Z|O) and rearranging yields
H(Z)−I(Z;O)≤h(p e) +p elog2(nk−1).
Sinceh(p e)≤1andlog2(nk−1)≤log2nk, we have
H(Z)−I(Z;O)≤1 +p elog2nk.
IfZis uniform, thenH(Z) = log2nk, hence
log2nk−I(Z;O)≤1 +p elog2nk⇒p e≥1−I(Z;O) + 1
log2nk,
which is (7).
Corollary B.2(Admissible candidate set size under bounded discriminability).Assume the routing evidence has bounded
discriminative information I(Z;O)≤B (in bits). If we require pe≤εfor some ε∈(0,1) , then any feasible nkmust
satisfy
log2nk≤B+ 1
1−ε=⇒n k≤2B+1
1−ε.(8)
14

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Proof.Rearrange (7) and substituteI(Z;O)≤B.
To determine an appropriate branching factor, we need a reasonable upper bound Bon the mutual information between the
(local) source index and the observed summary. In practice, summaries are not perfectly non-informative even they might be
extremely similar in our setup, so we assume a conservative bound
I(Z;O)≤B,
where Bincludes a safety margin for estimation noise and distribution shift. That means B, in practise, should be small to
ensure the routing error is bounded (We take B= 2 bits in our implementation, which is able to perfectly index 4 documents
if the indexing is independent from the content).
Plugging this bound into our design inequality (derived from the routing accuracy requirement)
nk≤2B+1
α,
where αdenotes the target routing accuracy (e.g., α= 0.85 for at least 85% accuracy), yields a practical threshold. For
example, takingB= 2bits andα= 0.85gives
nk≤22+1
0.85= 23.529≈11.5,
which motivates setting the split threshold to nk= 12 in our implementation (see able 8). Interestingly, although this
is a loose (worst-case) bound, the algorithm typically yields an average branching factor of about 4.5in practice. This
gap suggests that, even when we allow a 15% routing error budget, the summaries still provide enough signal to separate
childrenwithoutadditional side information, and the construction procedure naturally enforces a conservative split policy.
Empirically, the extra 2-bit margin translates into indexing only ≈4.5 semantic groups per parent on average, indicating (i)
our method tends to impose a strict partitioning criterion from an information-theoretic perspective, and (ii) items within the
same cluster are indeed highly similar, making fine-grained discrimination difficult.
C. Prompt Design
In this section, we present the specific prompts used for the experiments in Section 4.1.
C.1. Memory Structure Managment
This appendix lists the prompts used to construct our hierarchical memory from fine to coarse granularity. We first apply
a boundary detector to segment the dialogue stream into coherent episodes. For each episode, we generate an episodic
memory record with a title, third-person narrative content, and an explicit timestamp. We then distill high-value, persistent
semantic memories from accumulated episodes to form reusable long-term knowledge, filtering out transient conversation
details. Finally, we summarise clusters of related semantic statements into concise theme descriptions that serve as stable
high-level indices for structure management and retrieval.
C.2. Answer Generation
We provide dataset-specific answer generation prompts to match the distinct answer formats and evaluation protocols of
LoCoMo and PerLTQA. For LoCoMo, the prompt enforces short phrase-style outputs and explicit handling of temporal
reasoning by resolving relative time expressions using memory timestamps. For PerLTQA, the prompt requests a single com-
plete sentence and prioritises explicit factual attributes from the memory store, resolving conflicts by recency. Importantly,
to ensure fair comparison, we keep the final answer prompt identical across different baselines within each dataset.
15

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Prompt for Boundary Detect
You are a dialogue boundary detection expert. You need to determine if the newly added dialogue should end the current episode
and start a new one.
Current conversation history:{conversation history}
Newly added messages:{new messages}
Please carefully analyze the following aspects to determine if a new episode should begin:
1. **Topic Change** (Highest Priority): - Do the new messages introduce a completely different topic? - Is there a shift from
one specific event to another? - Has the conversation moved from one question to an unrelated new question?
2. **Intent Transition**: - Has the purpose of the conversation changed? (e.g., from casual chat to seeking help, from discussing
work to discussing personal life) - Has the core question or issue of the current topic been answered or fully discussed?
3. **Temporal Markers**: - Are there temporal transition markers (”earlier”, ”before”, ”by the way”, ”oh right”, ”also”, etc.)? -
Is the time gap between messages more than 30 minutes?
4. **Structural Signals**: - Are there explicit topic transition phrases (”changing topics”, ”speaking of which”, ”quick question”,
etc.)? - Are there concluding statements indicating the current topic is finished?
5. **Content Relevance**: - How related is the new message to the previous discussion? (Consider splitting if relevance ¡ 30-
Does it involve completely different people, places, or events?
Decision Principles: - **Prioritize topic independence**: Each episode should revolve around one core topic or event - **When
in doubt, split**: When uncertain, lean towards starting a new episode - **Maintain reasonable length**: A single episode
typically shouldn’t exceed 10-15 messages
Note: - If conversation history is empty, this is the first message, return false - When a clear topic change is detected, split
even if the conversation flows naturally - Each episode should be a self-contained conversational unit that can be understood
independently
Prompt for Episodic Generation
You are an episodic memory generation expert. Please convert the following conversation into an episodic memory.
Conversation content:conversation
Boundary detection reason:boundary reason
Please analyze the conversation to extract time information and generate a structured episodic memory. Return containing the
following three fields: ”title”: ”A concise, descriptive title that accurately summarises the theme (10-20 words)”, ”content”: ”A
detailed description of the conversation in third-person narrative. It must include all important information: who participated in
the conversation at what time, what was discussed, what decisions were made, what emotions were expressed, and what plans or
outcomes were formed. Write it as a coherent story so that the reader can clearly understand what happened. Ensure that time
information is precise to the hour, including year, month, day, and hour.”, ”timestamp”: ”YYYY-MM-DDTHH:MM:SS format
timestamp representing when this episode occurred (analyse from message timestamps or content)”
Time Analysis Instructions: 1. **Primary Source**: Look for explicit timestamps in the message metadata or content 2.
**Secondary Source**: Analyze temporal references in the conversation content (”yesterday”, ”last week”, ”this morning”, etc.)
3. **Fallback**: If no time information is available, use a reasonable estimate based on context 4. **Format**: Always return
timestamp in ISO format: ”2024-01-15T14:30:00”
Requirements: 1. The title should be specific and easy to search (including key topics/activities). 2. The content must include
all important information from the conversation. 3. Convert the dialogue format into a narrative description. 4. Maintain
chronological order and causal relationships. 5. Use third-person unless explicitly first-person. 6. Include specific details that aid
keyword search. 7. Notice the time information, and write the time information in the content. 8. When relative times (e.g., last
week, next month, etc.) are mentioned in the conversation, you need to convert them to absolute dates (year, month, day). Write
the converted time in parentheses after the original time reference. 9. **IMPORTANT**: Analyze the actual time when the
conversation happened from the message timestamps or content, not the current time.
Example: If the conversation is about someone planning to go hiking and the messages have timestamps from March 14, 2024 at
3:00 PM: ”title”: ”Weekend Hiking Plan March 16, 2024: Sunrise Trip to Mount Rainier”, ”content”: ”On March 14, 2024
at 3:00 PM, the user expressed interest in going hiking on the upcoming weekend (March 16, 2024) and sought advice. They
particularly wanted to see the sunrise at Mount Rainier, having heard the scenery is beautiful. When asked about gear, they
received suggestions including hiking boots, warm clothing (as it’s cold at the summit), a flashlight, water, and high-energy food.
The user decided to leave at 4:00 AM on Saturday, March 16, 2024 to catch the sunrise and planned to invite friends for the
adventure. They were very excited about the trip, hoping to connect with nature.”, ”timestamp”: ”2024-03-14T15:00:00”
16

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Prompt for Semantic Generation
You are an AI memory system. Extract HIGH-V ALUE, PERSISTENT semantic memories from the following episodes.
CRITICAL: Focus on extracting LONG-TERM V ALUABLE KNOWLEDGE, not temporary conversation details.
Episodes to analyze: episodes
## HIGH-V ALUE Knowledge Criteria
Extract ONLY knowledge that passes these tests: - **Persistence Test**: Will this still be true in 6 months? - **Specificity Test**:
Does it contain concrete, searchable information? - **Utility Test**: Can this help predict future user needs? - **Independence
Test**: Can be understood without conversation context?
## HIGH-V ALUE Categories (FOCUS ON THESE):
1. **Identity & Professional** - Names, titles, companies, roles - Education, qualifications, skills
2. **Persistent Preferences** - Favorite books, movies, music, tools - Technology preferences with reasons - Long-term likes
and dislikes
3. **Technical Knowledge** - Technologies used (with versions) - Architectures, methodologies - Technical decisions and
rationales
4. **Relationships** - Names of family, colleagues, friends - Team structure, reporting lines - Professional networks
5. **Goals & Plans** - Career objectives - Learning goals - Project plans
6. **Patterns & Habits** - Regular activities - Workflows, schedules - Recurring challenges
## Examples:
HIGH-V ALUE (Extract these): - ”Caroline’s favorite book is ’Becoming Nicole’ by Amy Ellis Nutt” - ”The user works at
ByteDance as a senior ML engineer” - ”The user prefers PyTorch over TensorFlow for debugging” - ”The user’s team lead is
named Sarah” - ”The user is learning Rust for systems programming” - ”The user has been practicing yoga since March 2021” -
”The user joined Amazon in August 2020 as a data scientist” - ”The user plans to relocate to Seattle in January 2025”
LOW-V ALUE (Skip these): - ”The user thanked the assistant” - ”The user was confused about X” - ”The user appreciated the
help” - ”The conversation was productive” - Any temporary emotions or reactions
Quality over quantity - extract only knowledge that truly helps understand the user long-term.
Prompt for Theme Generation
”You are building a stable high-level theme summary that captures recurring facts.” ”Given the following factual statements, write
a concise abstract theme that names the topic:” f”joined” ”Return only the summary.”
Prompt for Answer Generation in LOCOMO Dataset
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.
# CONTEXT: You have access to memories from two speakers in a conversation. These memories contain timestamped
information that may be relevant to answering the question.
# INSTRUCTIONS: 1. Carefully analyze all provided memories from both speakers 2. Pay special attention to the timestamps to
determine the answer 3. If the question asks about a specific event or fact, look for direct evidence in the memories 4. If the
memories contain contradictory information, prioritize the most recent memory 5. If there is a question about time references
(like ”last year”, ”two months ago”, etc.), calculate the actual date based on the memory timestamp. For example, if a memory
from 4 May 2022 mentions ”went to India last year,” then the trip occurred in 2021. 6. Always convert relative time references to
specific dates, months, or years. For example, convert ”last year” to ”2022” or ”two months ago” to ”March 2023” based on the
memory timestamp. Ignore the reference while answering the question. Do not answer temporal questions in timestamp such as
”2023-05-08”, but answer naturally such as ”7 May 2023” or ”The week before 6 July 2023”! 7. Focus only on the content of the
memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those
memories. 8. The answer should be less than 5-8 words.
Semantic Memories: semantic
Episodic Memories: episodic
Question: question
Answer:
17

Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
Prompt for Answer Generation in PerLTQA Dataset
You are a memory question-answering assistant. Answer the question using ONLY the information in the provided memories.
Memories may contain profiles, events, and relationship descriptions. Prefer explicit facts (e.g., ”Gender: female”, ”Nationality:
China”, dates, numbers, names). If multiple memories conflict, choose the most recent one by timestamp.
OUTPUT RULES (VERY IMPORTANT): - Write ONE concise, complete sentence. - Preserve the key value EXACTLY as
stated in the memories (names, numbers, dates, gender terms, nationality words). - Do NOT add extra explanation. - Do NOT
include quotes, bullet points, or prefixes like ”Answer:”.
Episodic Memories: episodic
Semantic Memories: semantic
Question: question
Answer:
18
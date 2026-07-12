# MIRAGE: Defending Long-Form RAG Against Misinformation Pollution

**Authors**: Saadeldine Eletter, Ruihong Zeng, Yuxia Wang, Maxim Panov, Aleksandr Rubashevskii, Preslav Nakov

**Published**: 2026-07-06 13:36:12

**PDF URL**: [https://arxiv.org/pdf/2607.05069v1](https://arxiv.org/pdf/2607.05069v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves factuality by grounding LLMs in external evidence, but real-world retrieval is often polluted: semantically relevant passages may contain subtle misinformation, misleading framings, or fabrications. We introduce MIRAGE, a training-free, model-agnostic defense for long-form RAG. MIRAGE builds an NLI-based cross-document claim graph and applies a Defended-Claims Gate to either condition generation on a consistent, multi-source supported subset or to block retrieval and answer parametrically. We also release a minimal-edit pollution protocol spanning four perturbation families (Unambiguous, Conflicting, Misleading, Fabricated) to construct matched clean, mixed, and fully polluted evaluation regimes. Across four long-form QA benchmarks and multiple commercial and open-weight LLMs, pollution severely degrades vanilla RAG, while MIRAGE consistently restores factuality under mixed and fully polluted evidence and outperforms prior robust-RAG methods. Our implementation and datasets are available at https://github.com/SaadElDine/MIRAGE.

## Full Text


<!-- PDF content starts -->

MIRAGE: Defending Long-Form RAG Against Misinformation Pollution
Saadeldine Eletter1, Ruihong Zeng1, Yuxia Wang2, Maxim Panov1,
Aleksandr Rubashevskii1, and Preslav Nakov1
1Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
2INSAIT, Sofia University “St. Kliment Ohridski”
{saadeldine.eletter}@mbzuai.ac.ae
Abstract
Retrieval-Augmented Generation (RAG) im-
proves factuality by grounding LLMs in ex-
ternal evidence, but real-world retrieval is of-
ten polluted: semantically relevant passages
may contain subtle misinformation, misleading
framings, or fabrications. We introduce MI-
RAGE, a training-free, model-agnostic defense
for long-form RAG. MIRAGE builds an NLI-
based cross-document claim graph and applies
a Defended-Claims Gate to either condition
generation on a consistent, multi-source sup-
ported subset or to block retrieval and answer
parametrically. We also release a minimal-edit
pollution protocol spanning four perturbation
families (Unambiguous, Conflicting, Mislead-
ing, Fabricated) to construct matched clean,
mixed, and fully polluted evaluation regimes.
Across four long-form QA benchmarks and
multiple commercial and open-weight LLMs,
pollution severely degrades vanilla RAG, while
MIRAGE consistently restores factuality un-
der mixed and fully polluted evidence and
outperforms prior robust-RAG methods. Our
implementation and datasets are available at
https://github.com/SaadElDine/MIRAGE.
1 Introduction
Long-form Question Answering relies heavily on
Retrieval-Augmented Generation (RAG) to gener-
ate comprehensive and factual responses (Wei et al.,
2024; Chen et al., 2025). Although RAG mitigates
“hallucinations”, it is prone to errors due to unreli-
able retrieval passages (Yoran et al., 2024). In prac-
tice, knowledge bases and web indices are noisy
environments, containing not just irrelevant docu-
ments, but polluted evidence, which are plausible
sounding texts that contain subtle fabrications, en-
tity swaps, or incorrect temporal attributions (Pan
et al., 2023; Zeng et al., 2025).
Current LLMs exhibit a “sycophancy” bias to-
wards retrieved context, often propagating these
User Query (Q) Retriever 
Clean 
(Wiki) POLLUTED 
(Fake) 
LLM Hallucinated Answer 
(Relevant, Conﬁdent but  
WRONG !)
T op-k 
Clean 
(News) Figure 1:Polluted RAG problem.A retriever returns a
mix of clean and polluted passages. Even a single high-
scoring polluted document can steer the LLM toward a
confident but incorrect answer.
errors into the final answer with high confi-
dence (Perez et al., 2023). Figure 1 demonstrates a
typical failure mode: a single high scoring polluted
document can steer the model away from estab-
lished facts, resulting in generated answers that
are relevant but factually incorrect. We argue that
robustness in RAG requires moving beyond sim-
ple context concatenation toward active evidence
adjudication.
To solve the aforementioned problem, we intro-
duce MIRAGE, a methodology that defends the
generation of RAG models against retrieval pol-
lution by enforcing the verification of the corre-
sponding claims. Our core insight is that correct
claims are often repetitive and consistent across di-
verse sources, whereas statements containing mis-
information are contradictory to each other. MI-
RAGE utilizes this observation via a two-stage
process. First, it extracts sentence-level claims
from retrieved passages and builds a support and
contradiction graph using Natural Language Infer-
ence (NLI). By analyzing the graph’s internal in-
consistency, MIRAGE decides whether to proceed
1arXiv:2607.05069v1  [cs.CL]  6 Jul 2026

with RAG or fallback to the model’s parametric
knowledge. Second, for the generated response,
we condition generation on a structured set of ver-
ified claims and instruct the model to ground its
output in this evidence.
We rigorously evaluate MIRAGE on four long-
form datasets using a new pollution protocol that
injects controllable corruptions (e.g., Unambigu-
ous, Conflicting, Misleading). Our results show
that retrieval pollution severely degrades standard
RAG. Under mixed pollution, MIRAGE improves
factuality by retaining mutually supported clean
claims while filtering inconsistent evidence. Under
full pollution, MIRAGE detects unreliable retrieval
and falls back to parametric generation, avoiding
harmful context. Across polluted long-form QA
settings, MIRAGE outperforms strong robust-RAG
baselines. Our contributions are as follows:
•We introduce MIRAGE, a training-free and
model-agnostic defense for long-form RAG
that constructs cross-document NLI-based
claim graphs to identify, prune, and gate
misinformation-polluted evidence before gen-
eration (see Section 3).
•We develop a controlled minimal-edit retrieval
pollution benchmark that systematically in-
jects four realistic families of misinforma-
tion (Unambiguous, Conflicting, Misleading,
and Fabricated), enabling reproducible evalu-
ation under clean, mixed, and fully polluted
retrieval settings (see Section 4).
•We conduct extensive experiments across
four long-form QA benchmarks, multiple
commercial and open-weight LLMs, and di-
verse robust-RAG baselines, showing that
MIRAGE consistently restores factuality un-
der polluted retrieval and substantially outper-
forms existing defenses while preserving the
performance on clean retrieval (see Sections 5
and 6).
2 Related Work
Active RAG and Self-Reflective Generation.
Retrieval-Augmented Generation (RAG) grounds
LLM outputs in external evidence to reduce hallu-
cinations. Beyond static “retrieve-then-generate,”
active and self-reflective RAG methods adapt re-
trieval and revision decisions during generation.
FLARE (Jiang et al., 2023b), Self-RAG (Asai et al.,2024), and DRAGIN (Su et al., 2024) trigger itera-
tive retrieval/refinement, while BRIDGE (Dai et al.,
2025) tightly couples retrieval and generation via
verification. However, these approaches largely as-
sume a reasonably trustworthy corpus and mainly
filter irrelevant context, leaving them vulnerable to
semantically relevant but incorrect passages (i.e.,
polluted evidence). MIRAGE instead adds a plug-
in verification layer that targets such semantic pol-
lution before it influences generation.
Robustness Against Noisy, Conflicting, and
Misleading Evidence.Recent work shows
LLMs can over-trust retrieved context (“syco-
phancy”), propagating plausible misinforma-
tion (Zeng et al., 2025). Several defenses address
conflicting/misleading retrieval: InstructRAG (Wei
et al., 2025) studies strategies for misleading ev-
idence; RetRobust (Yoran et al., 2024) and As-
tuteRAG (Wang et al., 2025a) improve robustness
via re-ranking and dependence control. Other
lines emphasize post-hoc verification, including
RARG (Yue et al., 2024), TrustRAG (Zhou et al.,
2025), and multi-agent deliberation (MADAM-
RAG; Wang et al., 2025b); RobustRAG (Xiang
et al., 2024) aggregates generations across passages
under perturbations. Unlike these approaches, MI-
RAGE is training-free and leverages global cross-
document consistency: truthful evidence tends to
be corroborated across sources, whereas polluted
evidence induces structural inconsistency.
Concurrent work extends retrieval-pollution eval-
uation to multimodal evidence, introducing QIMG-
7 and source-aware trust resolution over Parametric,
Text-only, and Full-MM answer branches (Eletter
et al., 2026). MIRAGE instead focuses on text-
only long-form RAG, using a cross-source NLI
claim graph and defended-claims gate to adjudi-
cate polluted textual evidence before generation.
3 Methodology
3.1 Overview
MIRAGE operates as a wrapper around a stan-
dard retrieval–generation pipeline.1Based on
top-kretrieved passages D, MIRAGE builds a
defended-claims graph Gover extracted sentence-
level claims. Then we prune inconsistent claims
and use this graph as a global gate. If Gis suf-
ficiently consistent and well-supported across in-
1In all experiments we plug MIRAGE on top of an
embedding-based retriever and do not modify the retrieval
model; see Appendix D for details.
2

dependent domains, MIRAGE trusts retrieval and
constructs a defended prompt for LLM containing
only verified claims. Otherwise, it blocks retrieval
and switches to a parametric (no-evidence) answer.
Figure 2 and Algorithm 1 summarize the entire
pipeline.
3.2 Claim Graph Construction
Given a user query q, a retriever returns a set of top-
kpassages D={d 1, . . . , d k}. For each passage
d∈ D , MIRAGE performs lightweight sentence
segmentation and treats each sentence as a claim c,
for which we track the following:
• asource identifiersrc(c)∈ {1, . . . , k};
•adomain dom(c) , derived from the source URL
(e.g.,wikipedia.org,nytimes.com);
•a claim-levelretrieval score r(c), inherited from
the source passage and computed as the dense-
retriever similarity score, and its max-normalized
counterpart ˜r(c)∈[0,1] , defined as ˜r(c) =
r(c)/max p∈Dr(p)over retrieved passages.
We discard empty sentences and take the highest-
scoring sentences across passages according to
˜r(c); see Appendix J.1 for further details.
MIRAGE models relationships between claims
with a directed, weighted graph. We first select
pairs of claims that are likely to be semantically
related, and then apply an NLI model to determine
their relationship.
Pair selection.We consider only cross-source
pairs (ci, cj)with src(c i)̸=src(c j), so that edges
capture inter-document agreement or disagreement.
LetT(c) denote the set of lowercased tokens in
claimc. We define a containment-style overlap:
overlap(c i, cj) =|T(c i)∩T(c j)|
min(|T(c i)|,|T(c j)|).(1)
We retain pairs with overlap(c i, cj)≥τ overlap and
cap the total number of pairs per query.
The token-overlap filter in equation (1)is used
only for efficient candidate generation before
NLI adjudication. This containment-style over-
lap works well in practice because supporting
evidence typically preserves key lexical anchors
(e.g., named entities and dates). We use it as an
efficiency-oriented approximation rather than a se-
mantic matching strategy; Appendix J.2 reports the
resulting NLI savings and paraphrase-robustness
trade-off.
Long-Form Query (Q) 
Retriever (top-k) 
Retrieved passages (D) Extract sentence-level claims 
source, domain, retrieval score 
Cross-source Claim NLI Graph 
+ counter-evidence probes 
Prune to defended claims 
consistent, multi-source subset 
Defended Claims 
Gate 
PASS 
Verified-Claims Prompt Answer Parametrically reliable unreliable 
BLOCK 
Generate Response 
Figure 2:MIRAGE pipeline.Retrieved passages are
verified before generation: MIRAGE extracts sentence-
level claims, links them with cross-document NLI sup-
port/contradiction edges, prunes to a consistent de-
fended set, and applies a Defended-Claims Gate. If
the graph is reliable, the LLM receives only verified
claims; otherwise retrieval is blocked and the model
answers with a no-retrieval prompt.
NLI scoring.For each selected pair (ci, cj),
we apply an NLI model to obtain probabilities
pentail, pcontr, pneut. We construct a weighted edge
eij= (i, j, wsupp
ij, wcontr
ij)where
wsupp
ij= max 
0,2p entail(ci, cj)−1
,(2a)
wcontr
ij=p contr(ci, cj).(2b)
Edges are classified as supporting if wsupp
ij> wcontr
ij,
and as contradictory if wcontr
ij> wsupp
ij; neutral or
low-confidence pairs are ignored.
Counter-evidence probing.The initial graph
captures only relations among retrieved claims.
To detect missing counter-evidence, MIRAGE re-
trieves targeted counter-evidence for top-ranked
claims and applies NLI between each claim and
its retrieved passages. When contradiction domi-
nates, a self-contradiction edge with weight wcontr
is added, penalizing claims contradicted by exter-
nal evidence; see Appendix J.3 for details.
3.3 Inconsistent Claim Pruning
Given the graph, MIRAGE assigns each claim a
score and selects a subset of mutually consistent,
well-supported claims.
3

Claim scoring.Each claim cis assigned a score
s(c) = log 
1 +n(c)
+α·trust(dom(c)) +β·˜r(c),(3)
where n(c) is the number of supporting claims,
trust(·) is a weak domain-quality prior, and ˜r(c)
is the normalized retrieval score. The scoring func-
tion prioritizes cross-source agreement, using do-
main priors and retrieval rank only as secondary
signals; thus, no single high-trust source can over-
ride broad cross-document disagreement. Details
on the coefficient choices (α, β) are provided in
Appendix C.2, and details on the domain prior in
Appendix J.4 and Table 10.
Consistent subset selection.Let Vbe the set of
claims and Ethe set of edges with weights wsupp
ij
andwcontr
ij. We select a subset S⊆V by approxi-
mately maximizing
F(S) =X
c∈Ss(c) +X
(i,j)∈E, i,j∈S 
wsupp
ij−wcontr
ij
,
(4)
which favors individually reliable claims while pro-
moting internal consistency.
Exact optimization is NP-hard, so we use a
greedy hill-climbing procedure that iteratively re-
moves claims whose exclusion improves F(S) .
The resulting subset S⋆forms the defended set,
which is used for downstream gating.
3.4 Retrieval Trust Gating
MIRAGE decides whether to trust retrieval based
on graph-level statistics of the defended set S⋆. Let
EsuppandEcontrdenote the numbers of support-
ing and contradictory edges in S⋆, classified by
an NLI model. Let D(S⋆)be the set of distinct
domains among the claims in S⋆. We define the
contradiction ratio and source diversity as
rcontr=Econtr
Esupp+Econtr+ε,(5a)
dsrc=|D(S⋆)|.(5b)
where εis a small constant for numerical stability.
Intuitively, contradiction ratio captures internal con-
flicts, while source diversity measures how broadly
evidence is distributed across domains.
MIRAGE enables defended mode only when all
active reliability criteria satisfy the gate thresholds
(rmax, dmin):
rcontr≤rmax,(6a)
dsrc≥dmin.(6b)
Clean 
Clean Clean Figure 3:Defended-claims graph.Claims form a graph
with NLI-based support (green) and contradiction (red)
edges. Polluted claims induce dense contradictions,
while clean evidence forms multi-source support clus-
ters. MIRAGE uses these patterns to detect unreliable
retrieval.
If either condition in equation (6)is violated, MI-
RAGE deems retrieval unreliable and blocks RAG,
instructing the LLM to answer from its paramet-
ric knowledge only. Otherwise, MIRAGE enters
defended mode and conditions generation on the
claims in S⋆. In practice, the gate satisfies basic
sanity checks: it passes most clean queries, de-
grades under mixed retrieval, and blocks nearly all
fully polluted queries, see Table 3.
Thresholds are determined once offline from
graph statistics on the combined held-out 20%
splits of all datasets, without invoking the gener-
ator; these splits are excluded from downstream
evaluation to avoid data leakage, and the result-
ing single global configuration is used across all
datasets and models. Additional details are pro-
vided in Appendix C, and a worked example is
given in Appendix H.
3.5 Generation from Verified Claims with
No-Retrieval Fallback
If the gate accepts the defended graph (i.e., Gis
considered reliable under equation (6)), MIRAGE
presents only the defended claims to the LLM
through a structured verified-claims prompt. Each
kept claim c∈S⋆is assigned a stable identifier
Ciand reported as a bullet point, together with its
score and source metadata. The prompt contains (i)
the user question, (ii) the list of verified claims, and
(iii) a short set of rules (see Appendix A for details).
This design turns retrieval into an evidence-only
input, constraining the model to generate answers
grounded in verified claims and reducing halluci-
nation.
4

Algorithm 1MIRAGE
Require:queryq, retrieverR, NLI modelN
1: Retrieve top-kpassagesD=R(q)
2:Extract sentence-level claims with metadata and build
node setV
3:For cross-source claim pairs with overlap(c i, cj)≥
τoverlap (equation (1)), run Nto add decisive sup-
port/contradiction edges (equations (2a)–(2b))
4:For top claims, run counter-evidence queries and add self-
contradiction edges
5:Compute claim scores s(c) (equation (3)) and greedily
prune to defended set S⋆maximizing F(S) (equation (4))
6:Compute contradiction and source-diversity statistics
(equations (5a) and (5b))
7:ifr contr≤r maxandd src≥d minthen
8: Build verified-claims prompt from S⋆and call LLM
conditioned on it
9:else
10: Call LLM in parametric mode (no evidence) and ig-
nore retrieved passages
11:end if
4 Retrieval Pollution Protocol
To systematically stress-test long-form RAG under
misinformation, we introduce minimally edited cor-
rupted variants of retrieved passages, yielding three
controlled retrieval regimes per query: CLEAN,
MIXED, and POLLUTED, while keeping the evi-
dence size fixed and preserving topical relevance.
Generation setup.For each question q, we start
from cached clean retrieval passages DCLEAN (q) =
{p1, . . . , p k}and use GPT-4o-mini to generate cor-
rupted variants ˜pithat remain fluent and on-topic
while introducing at least one verifiable factual er-
ror. We enforce strict edit constraints, including
high token overlap and neutral tone, to ensure real-
istic rather than overtly adversarial pollution.
The generator returns an explicit JSON object
containing: type (pollution family), polluted
(rewritten passage), and rationale (brief analysis-
only note). The rationale andtype are never
shown to MIRAGE; they are used only for auditing
and reporting.
Pollution families.Inspired by prior evidence
taxonomies for misleading/conflicting retrieval
(e.g., (Zeng et al., 2025)), we instantiate four cor-
ruption families under minimal-edit constraints:
•Unambiguous.Modify a single salient, check-
able atomic fact (e.g., a year, count, location, or
named entity) while leaving the rest of the pas-
sage unchanged.
•Conflicting.Paraphrase the passage while flip-
ping the polarity of a verifiable claim (e.g., “wasnot convicted” →“was convicted”, or a small
but consequential numerical change), producing
a direct contradiction.
•Misleading.Keep a subset of true statements
but alter framing so a reader is nudged toward
an incorrect conclusion (e.g., omission of key
constraints, cherry-picking, or emphasizing rare
exceptions), without inventing entirely new facts.
•Fabricated.Introduce plausible but non-existent
particulars (e.g., invented committee names, fab-
ricated study findings, or incorrect attributions),
but keeping topic and entities consistent withq.
For examples of each type of pollution, see Ap-
pendix K.
Retrieval regimes.Given the aligned clean and
polluted pools, we define:
•Clean:provide only clean passagesD CLEAN (q).
•Polluted:provide only polluted passages
DPOLL(q) ={˜p 1, . . . ,˜p k}(100% adversarial).
•Mixed:construct a fixed-size list of kpassages
where an αfraction (50% in our experiments) are
sampled from DPOLL(q)and the remainder from
DCLEAN (q), preserving per-query alignment.
This protocol produces matched clean, mixed,
and polluted evidence sets for each benchmark.
MIRAGE operates without access to ground -truth
passage labels; its decisions are based solely on the
retrieved text and the structural properties of the
defended-claims graph.
5 Experimental Setup
5.1 Datasets and Models
We report results on four long-form benchmarks:
LongFact (Wei et al., 2024), FA V A (Mishra et al.,
2024) , AlpacaFact (Dubois et al., 2023) and Biog-
raphy (Min et al., 2023). We evaluate MIRAGE on
both commercial and open-weight LLMs to ensure
our findings are model -agnostic. We use GPT-4o-
mini (OpenAI, 2026) and GPT-5-mini (OpenAI,
2026) as strong commercial baselines, and three
diverse open-weight models, namely LLaMA-3.1-
8B (Grattafiori et al., 2024), Mistral-7B-Instruct-
v0.2 (Jiang et al., 2023a), and Qwen3-8B (Yang
et al., 2025).
All four benchmarks are prompt-only (questions
without an accompanying gold evidence set). To en-
able controlled and reproducible retrieval/pollution
5

evaluation, we construct a dataset-specific cached
clean evidence pool from web search and derive
Clean/Mixed/Fully-Polluted retrieval regimes from
that fixed pool. Full details of evidence-pool con-
struction, de-duplication, retrieval indexing, and
caching are provided in Appendix D.
5.2 Retrieval and Verification
We use a fixed dense retriever over a dataset-
specific evidence pool and retrieve the top- kpas-
sages per query ( k=20 ). For claim-graph construc-
tion (and the optional sentence-level checker), we
useDeBERTa-large-MNLI(He et al., 2021).
We evaluate factuality with VeriScore (Song
et al., 2024) F1@ k(median over prompts), which
decomposes each generated answer into claims and
uses a verifier to label each claim as supported,
unsupported, or insufficient given evidence snip-
pets (Wei et al., 2024); see Appendix E.
Unless otherwise stated, all experiments use a
single fixed global configuration for both the core
gate criteria (Table 8) and computational settings
(Table 9). Core gate criteria configuration is se-
lected offline via grid search on cached retrieval
and NLI outputs without invoking the generator.
6 Experimental Results
6.1 Impact of Polluted Retrieval on RAG
Table 1 summarizes robustness to retrieval pollu-
tion averaged across datasets. Under clean retrieval,
MIRAGE closely matches vanilla RAG, showing
that the defended-claims gate rarely suppresses co-
herent evidence. Under mixed and especially full
pollution, vanilla RAG degrades sharply and of-
ten falls below No-RAG, whereas MIRAGE re-
stores factuality toward clean-RAG levels by filter-
ing contradiction-heavy evidence and falling back
to parametric answering when needed.
Under MixP, gains are large across backbones
(e.g., GPT-4o-mini improves from 53.88% to
83.43% on average; for LLaMA-3.1-8B, it im-
proves from 41.71% to71.91% ). Under FullP,
vanilla RAG collapses (e.g., GPT-4o-mini: 39.87%
vs. No-RAG 75.88% ), while MIRAGE substan-
tially recovers performance (GPT-4o-mini reaches
78.00% on average; LLaMA-3.1-8B reaches
70.34% ). See Appendix I for per-dataset results
and discussion.6.2 Long-form QA Results
Table 2 compares MIRAGE to robust-RAG base-
lines under both Mixed Pollution (MixP) and Full
Pollution (FullP). Across all datasets, MIRAGE
ranks first for both backbones and both regimes,
indicating that its defended-claims gating and prun-
ing provide consistent robustness gains.
The largest gains appear under FullP, where base-
lines are vulnerable to contradiction-heavy retrieval
while MIRAGE blocks unreliable evidence and
falls back to parametric answering, e.g., on Long-
Fact, MIRAGE improves over the strongest base-
line for GPT-4o-mini ( 88.93% vs.74.87% ) and for
Qwen3-8B ( 85.11% vs.66.41% ). Similar trends
hold across Biography, FA V A, and AlpacaFact.
These gains primarily reflect avoiding harmful
retrieval rather than exploiting polluted evidence.
Under polluted retrieval, existing methods often fall
below the No-RAG baseline, whereas MIRAGE
surpasses No-RAG through trust-aware gating and
fallback no-evidence generation. Under FullP, im-
provements mainly come from blocking polluted
context, while under MixP, MIRAGE also benefits
from retained clean, mutually supported claims.
6.3 Ablations
Gate behavior under clean, mixed, and polluted
retrieval.The gate relies on two interpretable
consistency criteria (rmax, dmin), controlling con-
tradiction tolerance, and source diversity. It trusts
retrieval only when the defended-claims graph is
globally consistent, well-supported, and backed by
multiple independent domains.
Table 3 shows that the gate passes 87.2% of clean
queries on average, drops to 48.6% under mixed re-
trieval, and blocks nearly all fully polluted queries
(0% pass). Full per-dataset results are provided in
Appendix Table 11.
Sensitivity to retrieval depth k.We conduct
a sensitivity analysis on LongFact with GPT-4o-
mini, varying the number of retrieved passages k
while keeping all other hyperparameters fixed. Ta-
ble 4 reports the median VeriScore F1@ kover 250
prompts across different corruption settings.
Overall, performance varies only slightly across
k= 5 to50, suggesting that MIRAGE is robust
to retrieval depth in this range. The optimal value
depends mildly on the corruption level, with larger
kbenefiting heavily polluted settings.
6

ModelNo-RAG (Avg) Clean RAG (Avg) MixP (Avg) FullP (Avg)
RAG MIRAGE RAG MIRAGE RAG MIRAGE
GPT-4o-mini 75.88 85.32 87.5653.88 83.4339.87 78.00
GPT-5-mini 70.33 82.69 83.5754.40 80.5237.69 73.70
LLaMA-3.1-8B 68.62 70.64 76.3641.71 71.9125.11 70.34
Mistral-7B 69.73 82.15 84.0945.12 74.8732.67 72.29
Qwen3-8B 69.75 72.11 78.4546.64 74.9630.89 71.44
Table 1:Average factuality under retrieval pollution.VeriScore F1@k percentages averaged across four long-
form QA datasets and five LLM backbones. We report No-RAG, clean RAG, and polluted retrieval settings: Mixed
Pollution (MixP, 50% polluted) and Full Pollution (FullP, 100% polluted), comparing vanilla RAG and MIRAGE.
Full per-dataset results are in Appendix Table 14.
Model Method FA V A LongFact Biography AlpacaFact
MixP FullP MixP FullP MixP FullP MixP FullP
GPT-4o-miniNo-RAG 81.20 86.90 58.99 76.41
RAG 55.79 34.83 58.24 49.11 42.56 22.65 58.92 52.90
DRAGIN 75.13 61.25 73.41 65.40 52.58 32.09 67.43 61.47
FLARE 68.21 63.89 73.32 69.32 51.58 37.38 60.53 55.67
SelfRAG 51.86 42.44 73.08 71.57 56.23 38.55 70.38 65.68
BRIDGE 69.46 67.18 80.71 70.33 56.37 43.50 64.98 62.38
AstuteRAG 74.89 72.87 81.14 74.87 56.55 47.91 68.11 63.76
MADAMRAG 64.09 42.25 65.48 50.33 58.39 25.48 59.59 40.54
RobustRAG 46.77 26.98 66.22 43.14 40.47 20.27 45.81 26.46
TrustRAG 57.19 42.64 65.66 49.01 48.03 23.42 60.62 43.15
RetRobust-NLI 69.28 64.26 70.27 62.47 49.50 45.33 69.97 64.53
RARG 51.63 41.70 65.38 53.53 48.76 22.92 57.32 45.63
MIRAGE 91.59 82.54 90.33 88.93 70.11 61.87 81.70 78.65
Qwen3-8BNo-RAG 74.92 84.61 53.75 65.71
RAG 47.62 31.28 57.13 40.61 26.64 13.24 55.15 38.43
DRAGIN 53.68 36.55 71.48 55.28 33.36 17.50 57.10 43.93
FLARE 48.64 45.33 61.13 58.03 36.77 12.26 55.34 46.17
SelfRAG 55.37 46.69 65.32 49.52 35.18 18.97 63.57 50.63
BRIDGE 55.19 49.94 70.50 66.41 31.44 21.31 61.42 56.66
AstuteRAG 59.61 54.66 67.22 65.04 28.86 20.70 65.73 60.30
MADAMRAG 52.38 41.98 64.92 44.70 48.91 15.15 53.61 36.04
RobustRAG 54.31 35.27 69.21 48.35 48.14 25.22 61.12 44.32
TrustRAG 49.23 32.63 59.01 39.79 33.64 15.68 55.26 37.69
RetRobust-NLI 50.17 40.40 65.15 62.95 15.27 12.37 59.13 57.20
RARG 45.74 32.25 56.37 36.39 34.03 16.07 53.88 35.87
MIRAGE 78.70 77.35 87.56 85.11 61.25 55.49 72.34 67.79
Table 2:Comparison to robust-RAG baselines under pollution.VeriScore F1@k percentages on four long-form
QA datasets under Mixed Pollution (MixP, 50% polluted) and Full Pollution (FullP, 100% polluted). MIRAGE
achieves the best F1@k across datasets in both regimes; full per-dataset robustness tables are in Appendix Table 14.
The second-best result (underlined) uses no-retrieval generation.
Impact of MIRAGE components.We ablate
MIRAGE components for GPT-4o-mini across
four datasets; Table 5 reports average F1 and rela-
tive drops from the full model. Each ablation hurts
performance, showing that counter-evidence prob-
ing, contradiction-based trust, and diversity-based
trust provide complementary robustness. Under
mixed pollution, single-component removals cause
similar drops, suggesting that no individual sig-
nal alone explains the gain. Under full pollution,
diversity-based gating is most critical: removing
it causes a 26.5% drop, while removing all trust-
gating components causes the largest degradation
(38.9%). See Appendix F for other baselines.6.4 Automatic Label Validation
NLI labels validation.We manually validate 150
claim pairs balanced across clean, mixed, and pol-
luted retrieval ( 50each). Two annotators label
each pair as supported or unsupported, reaching
89.33% raw agreement and Cohen’s κ= 0.743 ;
per-subset results are reported in Table 6. This
validates the graph edges used by MIRAGE be-
fore generation, rather than only the final answers.
Comparing NLI labels against merged human an-
notations gives 90.67% agreement and κ= 0.723 ,
with supported precision/recall of 0.917/0.965 and
unsupported precision/recall of0.862/0.714.
7

(a) DeBERTa-large-MNLI gate outcomes
Condition Pass % (Avg) Block % (Avg)Rel. pass
drop (%) (Avg)
Clean87.2 12.8 0.0
Mixed48.6 51.4 44.3
Polluted0.0 100.0 100.0
(b) Best-gate behavior across NLI backbones
NLI backbone Clean Polluted
Pass % Block % Pass % Block %
BART-large-MNLI99.5 0.5 0.4 99.6
RoBERTa-large-MNLI100.0 0.0 0.4 99.6
DeBERTa-large-MNLI99.9 0.1 0.2 99.8
Table 3:Offline MIRAGE gate outcomes.(a)
Pass/block rates for the main DeBERTa-large-MNLI
configuration, averaged across four datasets; relative
pass drop is computed vs. Clean. (b) Best offline gate
performance for different NLI backbones.
kClean MixP (50%) FullP (100%)
5 89.89 90.28 89.41
10 90.16 90.56 89.10
20 91.13 90.33 88.93
30 90.72 90.83 89.78
50 89.61 90.77 90.53
Table 4: Sensitivity of MIRAGE to retrieval depth kon
LongFact. Results are stable across a wide range of k,
indicating robustness to retrieval depth.
VeriScore labels validation.We manually ver-
ify 150 VeriScore-labeled claims with retrieved
evidence; two annotators achieve 87.33% agree-
ment. This check complements the NLI-label audit
above: while the NLI audit validates the claim-
graph edges used by MIRAGE, the VeriScore au-
dit validates the factuality labels used to evaluate
generated answers. We further evaluate 600 claims
balanced across clean, mixed, and polluted settings
using GPT-5-mini, GPT-4.1-nano, GPT-4o, and
GPT-5.5 (OpenAI, 2026), obtaining a mean agree-
ment of92.33%with VeriScore.
These results suggest that the observed robust-
ness gains are unlikely to be an artifact of a sin-
gle verifier or a single retrieval regime. The high
agreement therefore supports using VeriScore con-
sistently across clean, mixed, and fully polluted
regimes. Prior work also supports VeriScore re-
liability, reporting strong agreement with human
annotations, including substantial inter-annotator
agreement ( κup to 0.73), across multiple datasets
and studies (Chen et al., 2025; Wan et al., 2025;
Rajendhran et al., 2025).MethodMixed Polluted
F1∆%F1∆%
MIRAGE83.43–78.00–
w/o counter-evid. 71.04−14.9%74.05−5.1%
w/o trust (contr.) 70.11−16.0%74.18−4.9%
w/o trust (div.) 71.31−14.5%57.31−26.5%
w/o trust (all) 69.61−16.6%47.65−38.9%
Table 5: Ablation results averaged over four datasets
under mixed and fully polluted retrieval. ∆% denotes
relative F1 change with respect to MIRAGE. Removing
any component degrades performance, highlighting the
contribution of each module.
Condition Num. of Claims Accuracy Cohen’sκ
Inter-annotator agreement
Clean 5092% 0.783
Mixed 5086% 0.636
Polluted 5090% 0.788
Merged human vs. NLI
Clean 5088% 0.625
Mixed 5096% 0.864
Polluted 5088% 0.694
Table 6: Agreement statistics for NLI annotations based
on 150 claim pairs. Accuracy denotes raw agreement,
while Cohen’sκaccounts for chance agreement.
7 Conclusion and Future Work
To defend against retrieval pollution (on-topic mis-
information), we introduce MIRAGE, a training-
free, model-agnostic defense for long-form RAG.
MIRAGE builds a cross-source claim graph and
uses a defended-claims gate to either generate from
globally consistent evidence or fall back to para-
metric answering.
Across four long-form QA benchmarks and mul-
tiple LLMs, polluted retrieval degrades vanilla
RAG, while MIRAGE restores factuality under
mixed and fully polluted evidence and outperforms
prior robust-RAG methods. We also release a
minimal-edit retrieval pollution protocol for con-
trolled evaluation. Our results highlight cross-
source consistency as a key signal for safeguard-
ing factual generation under untrusted retrieval.
More broadly, the findings suggest that robust RAG
should verify retrieved evidence before generation,
rather than treating topical relevance as sufficient
evidence quality.
Future work includes improving robustness to
domain shift, multilingual and coordinated misin-
formation, and scaling claim-graph construction
with more efficient gating.
8

Limitations
MIRAGE inherits the coverage and diversity of
retrieved evidence: low recall or limited source
diversity can produce sparse graphs and more
conservative blocking, yielding safer but less in-
formative answers. It uses NLI-based consis-
tency and lexical-overlap filtering as efficient, in-
terpretable approximations for scalable cross-claim
verification, though these can miss paraphrastic
or omission-based misinformation. Coordinated
misinformation repeated across sources may also
appear trustworthy. MIRAGE is intentionally
training-free and modular, improving robustness,
interpretability, and auditability across settings,
though this may limit adaptation to some domains
or languages. Finally, while generation still re-
quires only one LLM call per query, MIRAGE
introduces additional local NLI computation for
graph construction and counter-evidence checks;
Appendix G reports the resulting overhead.
Ethical Considerations
MIRAGE aims to improve robustness of RAG un-
der misinformation, reducing the risk of propagat-
ing incorrect retrieved content. Our pollution proto-
col, while designed for controlled evaluation, could
be misused to generate plausible misinformation;
we therefore recommend controlled release and
standard safeguards.
It relies on NLI as a standard proxy for seman-
tic consistency and on lightweight domain priors.
While effective in practice, these components may
exhibit bias or degrade under domain shift; we
mitigate this through aggregation and conservative
abstention, but monitoring and adaptation to de-
ployment context remain important.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InProceedings of the International Conference on
Learning Representations, ICLR ’24.
Mingda Chen, Yang Li, Karthik Padthe, Rulin Shao,
Alicia Yi Sun, Luke Zettlemoyer, Gargi Ghosh, and
Wen-tau Yih. 2025. Improving factuality with ex-
plicit working memory. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL ’25, pages
11199–11213, Vienna, Austria. Association for Com-
putational Linguistics.Xinbang Dai, Huikang Hu, Yuncheng Hua, Jiaqi Li,
Yongrui Chen, Rihui Jin, Nan Hu, and Guilin Qi.
2025. After retrieval, before generation: Enhanc-
ing the trustworthiness of large language models
in retrieval-augmented generation.ArXiv preprint,
arXiv:2505.17118.
Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi
Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin,
Percy S. Liang, and Tatsunori B. Hashimoto. 2023.
AlpacaFarm: A simulation framework for methods
that learn from human feedback. InAdvances in Neu-
ral Information Processing Systems, volume 36 of
NeurIPS ’23, pages 30039–30069. Curran Associates,
Inc.
Saadeldine Eletter, Owais Aijaz, and Preslav Nakov.
2026. Trust before fusion: QIMG-7 and source-
aware resolution for polluted multimodal RAG.
Manuscript.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 2 others. 2024. The Llama 3 herd
of models.ArXiv preprint, arXiv:2407.21783.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and
Weizhu Chen. 2021. DeBERTa: Decoding-enhanced
BERT with disentangled attention. InProceedings of
the International Conference on Learning Represen-
tations, ICLR ’21.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lelio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothee Lacroix,
and William El Sayed. 2023a. Mistral 7B.ArXiv
preprint, arXiv:2310.06825.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023b. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP ’23, pages 7969–7992,
Singapore. Association for Computational Linguis-
tics.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. FActScore:
Fine-grained atomic evaluation of factual precision in
long form text generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP ’23, pages 12076–12100,
Singapore. Association for Computational Linguis-
tics.
Abhika Mishra, Akari Asai, Vidhisha Balachandran,
Yizhong Wang, Yulia Tsvetkov, Graham Neubig, and
9

Hannaneh Hajishirzi. 2024. Fine-grained hallucina-
tion detection and editing for language models. In
Proceedings of the Conference on Language Model-
ing, COLM ’24.
OpenAI. 2026. OpenAI API model documen-
tation. https://developers.openai.com/api/
docs/models. Accessed: 2026-06-10.
Yikang Pan, Liangming Pan, Wenhu Chen, Preslav
Nakov, Min-Yen Kan, and William Wang. 2023.
On the risk of misinformation pollution with large
language models. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2023,
EMNLP ’23, pages 1389–1403, Singapore. Associa-
tion for Computational Linguistics.
Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina
Nguyen, Edwin Chen, Scott Heiner, Craig Pettit,
Catherine Olsson, Sandipan Kundu, Saurav Kada-
vath, Andy Jones, Anna Chen, Ben Mann, Brian
Israel, Bryan Seethor, Cameron McKinnon, Christo-
pher Olah, Da Yan, Daniela Amodei, and 44 others.
2023. Discovering language model behaviors with
model-written evaluations. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2023,
ACL ’23, pages 13387–13434, Toronto, Canada. As-
sociation for Computational Linguistics.
Rishanth Rajendhran, Amir Zadeh, Matthew Sarte,
Chuan Li, and Mohit Iyyer. 2025. VeriFastScore:
Speeding up long-form factuality evaluation. InFind-
ings of the Association for Computational Linguis-
tics: EMNLP 2025, EMNLP ’25, pages 9234–9259,
Suzhou, China. Association for Computational Lin-
guistics.
Yixiao Song, Yekyung Kim, and Mohit Iyyer. 2024.
VeriScore: Evaluating the factuality of verifiable
claims in long-form text generation. InFindings
of the Association for Computational Linguistics:
EMNLP 2024, EMNLP ’24, pages 9447–9474, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
Radu Soricut and Abdessamad Echihabi. 2010.
TrustRank: Inducing trust in automatic translations
via ranking. InProceedings of the 48th Annual Meet-
ing of the Association for Computational Linguistics,
ACL ’10, pages 612–621, Uppsala, Sweden. Associ-
ation for Computational Linguistics.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. DRAGIN: Dynamic retrieval
augmented generation based on the real-time informa-
tion needs of large language models. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
ACL ’24, pages 12991–13013, Bangkok, Thailand.
Association for Computational Linguistics.
Yingjia Wan, Haochen Tan, Xiao Zhu, Xinyu Zhou, Zhi-
wei Li, Qingsong Lv, Changxuan Sun, Jiaqi Zeng,
Yi Xu, Jianqiao Lu, Yinhong Liu, and Zhijiang Guo.
2025. FaStFact: Faster, stronger long-form factualityevaluations in LLMs. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
EMNLP ’25, pages 23814–23854, Suzhou, China.
Association for Computational Linguistics.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and
Sercan O. Arik. 2025a. ASTUTE RAG: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
ACL ’25, pages 30553–30571, Vienna, Austria. As-
sociation for Computational Linguistics.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025b. Retrieval-augmented genera-
tion with conflicting evidence. InProceedings of the
Conference on Language Modeling, COLM ’25.
Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu,
Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng,
Ruibo Liu, Da Huang, Cosmo Du, and Quoc V . Le.
2024. Long-form factuality in large language models.
InAdvances in Neural Information Processing Sys-
tems, volume 37 ofNeurIPS ’24, pages 80756–80827.
Curran Associates, Inc.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. In-
structRAG: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. InProceedings
of the International Conference on Learning Repre-
sentations, ICLR ’25.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably ro-
bust RAG against retrieval corruption. InICML 2024
Workshop on Next Generation of AI Safety, ICML
Workshop ’24.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 oth-
ers. 2025. Qwen3 technical report.ArXiv preprint,
arXiv:2505.09388.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InProceedings
of the International Conference on Learning Repre-
sentations, ICLR ’24.
Zhenrui Yue, Huimin Zeng, Yimeng Lu, Lanyu Shang,
Yang Zhang, and Dong Wang. 2024. Evidence-driven
retrieval augmented response generation for online
misinformation. InProceedings of the 2024 Con-
ference of the North American Chapter of the As-
sociation for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
NAACL ’24, pages 5628–5643, Mexico City, Mex-
ico. Association for Computational Linguistics.
Linda Zeng, Rithwik Gupta, Divij Motwani, Yi Zhang,
and Diji Yang. 2025. Worse than zero-shot? a fact-
checking dataset for evaluating the robustness of
10

RAG against misleading retrievals. InThe Thirty-
ninth Annual Conference on Neural Information Pro-
cessing Systems Datasets and Benchmarks Track,
NeurIPS ’25.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025. TrustRAG: Enhancing robust-
ness and trustworthiness in RAG.ArXiv preprint,
arXiv:2501.00879.
A MIRAGE Prompts
A.1 Verified-Claims Generation (Gate Passed)
Task: Write a factual long-form answer using
the verified claims below.,→
Rules:
(1) Do not add new unverified facts.
(2) Cite claims inline as [C#].
(3) Prefer higher-scoring claims.
(4) If claims are insufficient or conflicting,
say so and stop.,→
Question: {q}
Verified claims:
C1: {text} (score={s}; src={domain});
...
A.2 Parametric Generation (Gate Blocked)
Write the best factual long-form answer to the
question using only your internal
knowledge.,→
,→
Be explicit about uncertainty.
Avoid speculation.
If you are not confident, say what would be
needed to verify the answer.,→
B Pollution Protocol Details
B.1 Goal
To stress-test long-form RAG under realistic re-
trieval noise, we construct polluted evidence by
minimally editing clean passages into on-topic, lin-
guistically plausible text that is nevertheless fac-
tually incorrect. The objective is to preserve sur-
face relevance and fluency while injecting concrete,
verifiable errors that can mislead a generator that
over-trusts retrieved context.
B.2 Input and output format
For each instance, the generator receives (i) a QA-
style question (topic anchor), (ii) an evidence pas-
sage (optionally with a title), and (iii) a target pol-
lution type (hint). It returns strict JSON with keys:
type (pollution family), polluted (the rewritten
passage), and rationale (brief researcher-facing
explanation of what was changed and why it is
wrong).B.3 Controlled Corruption Constraints
To avoid trivial artifacts, we enforce: (1)topic
preservation(the rewrite remains about the same
entities and subject as the question/passage), (2)
minimal editing(preserve 50–80% of tokens and
cap token-change at ≤0.75 ), (3)non-absurd plau-
sibility(neutral encyclopedic tone; no satire), and
(4)anti-shortcut rules(reject vague comparatives
and meta-commentary; no citations/URLs).
B.4 Generation Prompt
We use the following system and user templates,
shown verbatim, to produce polluted passages:
You are an adversarial data generator for robustness
testing.,→
You will receive:
- a QA-style question (topic anchor)
- an evidence passage (title optional)
Your task: rewrite the passage so it stays 100% ON-TOPIC and
linguistically plausible,,→
but is factually WRONG according to reality and the original
passage.,→
Do NOT answer the question directly; produce an
evidence-style paragraph.,→
Keep entities and topic consistent, but alter at least one
concrete, verifiable atomic fact (e.g., year, number,
location detail, role, or causal direction) to make the
content incorrect yet superficially credible.,→
,→
,→
Never include citations, brackets, URLs, or meta-comments.
Do not say it is wrong or fabricated.
Mimic a neutral encyclopedic tone unless instructed
otherwise.,→
You must pick at least one {pollution_types} strategy and
follow its rules:,→
- Conflicting: explicitly contradict a key fact from the
original passage,→
(e.g., swap dates/names/causes).
- Misleading: keep some truths but cherry-pick, omit, or
frame details so that,→
a reader is led to a wrong conclusion.
- Fabricated: invent plausible but non-existent particulars
(e.g., a committee name, a study, or figures).
- Unambiguous: minimally alter one salient, explicit,
checkable atomic fact from the original passage while
leaving the rest of the passage largely unchanged.,→
,→
Hard constraints:
- Stay relevant to the given question and original passage.
- Avoid absurdity, satire, or obvious jokes.
- DO NOT mention which strategy you used.
- Output STRICT JSON ONLY with keys: type, polluted,
rationale.,→
- type in {pollution_types}
- polluted: the rewritten wrong passage
- rationale: 1-2 sentences explaining, for researchers, how
it is wrong.,→
Do not include this rationale in the polluted text.
B.5 Dataset-level balancing
To reduce confounding from any single corruption
style, we generate polluted passages using an ap-
proximately uniform mix over the four families.
Table 7 reports dataset-level statistics, including the
number of corrupted points for the full-pollution
and mixed-pollution settings, as well as mixed
11

Figure 4:Pollution-type counts by dataset.Bars are
grouped by pollution family (Conflicting, Misleading,
Fabricated, Unambiguous); within each group, colors
indicate the dataset (see legend).Top: Full regime,
where all retrieved passages are corrupted.Bottom:
Mixed regime, where half of the retrieved passages are
corrupted. Exact counts match Table 7.
clean and unknown counts. Figure 4 reports the
resulting distribution per dataset.
C Defended-Claims Gate
We distinguish between (i) core semantic gate crite-
ria that define retrieval reliability (Table 8) and (ii)
computational graph-construction parameters used
for tractability (Table 9). Unless stated otherwise,
we use a single fixed configuration across datasets
and LLM backbones.
Complementing Table 3, which reports averaged
offline gate pass/block rates across datasets, we
provide the corresponding per-dataset breakdown
in Table 11.
C.1 Domain Trust Mapping
For each evidence passage, we extract the source
domain by parsing the URL host when available
(or by regex-detecting a URL inside the passage
text); if no URL is present, we set the domain to
unknown . We then assign a domain-trust weight
by applying the rules in Table 10 via substring
or TLD matching. If multiple rules match, the
implementation applies a fixed priority order to
ensure deterministic assignment.
C.2 Claim Scoring Weights
The coefficients (α, β) in equation (3)are not tuned
per dataset and are fixed across all experiments
(α= 0.5 andβ= 0.3 ). This choice encodesa simple priority ordering that we keep consis-
tent throughout: (i) cross-source agreement should
dominate, (ii) domain quality should provide a
modest prior, and (iii) retrieval rank should only
break ties.
We use log(1+n(c)) so that the first independent
supporter yields a large gain (from 0tolog 2≈
0.69) while additional supporters have diminishing
returns. We scale the domain prior by 0.5so that
moving from the lowest- to highest-trust sources
(0.20→0.90) changes s(c) by at most 0.35, which
is still smaller than the gain from one additional
independent supporter ( log 2≈0.69 ), preventing
priors from overpowering consensus.
D Retriever Details
We use a fixed dense bi-encoder retriever in all
experiments. Retrieved passages and similarity
scores are cached per (dataset, model) run for re-
producibility and to seed the pollution protocol.
D.1 Corpus and Candidate Pool
All benchmarks we use are prompt-only; therefore,
we construct a dataset-specific candidate corpus by
collecting a clean evidence pool from the web. Fol-
lowing VeriScore’s evidence retrieval step (Song
et al., 2024), we query Google Search via the Ser-
per API for each prompt and cache the returned top
results (title, snippet, URL). We treat each cached
result as a candidate evidence passage and store the
full pool (per dataset) for reproducibility. We per-
form lightweight normalization and de-duplication
(e.g., by URL host/title) to obtain a stable candidate
set per dataset.
D.2 Retrieval Regimes and Caching
All retrieval regimes (Clean/Mixed/Fully Polluted)
are derived from the same cached candidate pool.
Clean uses the retrieved passages as-is, while
Mixed and Fully Polluted replace retrieved pas-
sages with their minimal-edit rewritten counter-
parts produced by our pollution generator (Sec-
tion 4). Unless stated otherwise, we retrieve top- k
passages per prompt ( k=20 ), and cache both the
selected passages and their similarity scores for
each run.
Note: Serper is used in two distinct places—(i)
offline to cache a fixed evidence pool for building
our RAG retrieval corpora/regimes, and (ii) during
evaluation within VeriScore to retrieve external ev-
idence per generated claim; VeriScore never sees
our RAG passages as its verification evidence.
12

FullP MixP
Dataset #Questions #Passages C M F U C M F U
FA V A 100 2000 483 484 516 517 238 243 267 251
LongFact 250 5000 1223 1248 1258 1271 613 615 632 637
Biography 183 3660 903 896 923 938 448 438 474 470
AlpacaFact 233 4660 1165 1165 1165 1165 594 580 573 583
Table 7:Dataset-level pollution statistics.Number of polluted passages by corruption family: C=Conflicting,
M=Misleading, F=Fabricated, U=Unambiguous. Mixed Pollution (MixP, half of passages polluted) and Full
Pollution (FullP, all passages polluted).
Symbol Meaning Default
Core Gate Criteria
rmax Maximum contradiction tolerance 0.98
dmin Minimum required source diversity 2
Claim Scoring Weights
αWeight for source-quality prior 0.5
βWeight for retrieval score 0.3
Table 8:Core MIRAGE gate criteria and claim-
scoring weights.The contradiction and diversity thresh-
olds define the active semantic trust decision used by
MIRAGE. All values are fixed globally across datasets
and LLM backbones.
Symbol Meaning Default
kRetrieved passages/query 20
kctr Counter passages/probed claim 3
M Claims probed for counter-evidence 8
NCandidate-claim cap/query 120
Pmax Sampled pairwise NLI cap/query 400
τoverlap Token-overlap threshold 0.62
θsupport NLI confidence threshold 0.25
Table 9:Computational limits and graph-
construction parameters.These parameters are
introduced for tractable graph construction and
efficient NLI evaluation, and do not encode semantic
assumptions about retrieval reliability.
D.3 Dense Retriever
Given a question q, we embed qand all candi-
date passages using an embedding model E(·) and
rank passages by similarity in embedding space.
We taketext-embedding-3-small as an embed-
ding model, similarity is computed via dot-product;
when normalization is enabled, this corresponds to
cosine similarity.
D.4 Caching and Reproducibility
To make retrieval deterministic and efficient, we
pre-construct and cache per-question retrievers.
Concretely, we (i) build a stable mapping fromDomain type (rule) Trust
Reference / encyclopedia (wikipedia,wiki)0.90
Official / academic (.gov,.edu)0.80
Major news keywords (nytimes,reuters)0.70
Non-profit / org TLD (.org)0.65
Commercial TLD (.com)0.50
Unknown / no domain found (unknown)0.40
Forum / discussion keyword (forum)0.20
Table 10:Domain-trust priors used in node scor-
ing.We assign each claim a source domain extracted
from the evidence (URL host when present; otherwise
unknown ). We then map the domain to a heuristic trust
weight in [0,1] using simple substring/TLD rules (im-
plemented deterministically in code). This trust term is
only a secondary prior in equation (3)(scaled by 0.5)
and does not override multi-source agreement.
questions to question_id , and (ii) store a serial-
ized retriever object per ( dataset ,question_id ,
embedding_model ) under a cache directory . At
runtime, retrieval loads these cached objects and
returns the same ranked lists and scores.
E Verification Details
We evaluate factuality with VeriScore (Song et al.,
2024), which decomposes each generated answer
into atomic, verifiable claims and verifies each
claim against the provided evidence snippets. The
verifier assigns one of three labels—SUPPORTED,
UNSUPPORTED, orINSUFFICIENT. We use a strict
verifier prompt that (i) decides only from the shown
snippets, (ii) requires high-trust evidence to mark
a claim asSUPPORTED, and (iii) treats micro-
perturbations (dates, numbers, entities, directions,
and temporal scopes) asUNSUPPORTED. We re-
port VeriScore F1@ K, where Kis set per dataset
as the median number of extracted claims (per the
VeriScore definition).
For each extracted claim, we retrieve external
verification evidence from the web using Google
Search via the Serper API, following the original
13

Dataset Condition Pass % Block %Rel. pass
drop (%)
FA V AClean87.0 13.0 0.0
Mixed50.0 50.0 42.5
Polluted0.0 100.0 100.0
LongFactClean86.4 13.6 0.0
Mixed45.2 54.8 47.7
Polluted0.0 100.0 100.0
BiographyClean90.7 9.3 0.0
Mixed49.2 50.8 45.8
Polluted0.0 100.0 100.0
AlpacaFactClean84.5 15.5 0.0
Mixed49.8 50.2 41.1
Polluted0.0 100.0 100.0
Table 11:Offline MIRAGE gate outcomes (per
dataset).For each dataset and retrieval regime, we
report the fraction of queries where the DeBERTa-based
MIRAGE gate passes (uses retrieved evidence) or blocks
(falls back to a parametric answer). TheRel. pass drop
(%)is the relative decrease in pass rate vs. Clean for the
same dataset: 100×(1−Pass/Pass Clean). This table
complements the averaged results in Table 3.
VeriScore pipeline. Specifically, the claim text is
used as the search query, and the verifier receives
the top returned results (title, snippet, and URL).
The verifier is therefore not exposed to our RAG
retrieval passages (clean/mixed/polluted); the ver-
ification evidence source remains identical across
retrieval regimes. Consequently, performance dif-
ferences under mixed or fully polluted retrieval
reflect changes in the generated claims caused by
polluted generation context, rather than contami-
nation of the verification process itself. For repro-
ducibility, all web evidence pools are cached, and
both generation and verification use snippets drawn
from these cached results.
F Additional RAG Baselines
We also introduce a number of baselines for ex-
perimental comparison depending on the usage of
retrieval passages for answer generation. There are
as follows:
No-RAG. Generation without retrieval: y∅=
LLM θ(q).
Clean RAG. Generation with clean retrieval pas-
sages:y clean= LLM θ(q,D clean).
Polluted RAG. Generation with polluted retrieval
passages:y poll= LLM θ 
q,D poll
.
Clean + Polluted RAG. Combined-evidence gen-Dataset Model Clean Polluted∆%Drop
FA V AGPT-4o-mini 87.36 34.83 -60.1
GPT-5-mini 82.18 36.54 -55.5
LLaMA-3.1-8B 78.15 37.43 -52.1
Mistral-7B 82.04 27.64 -66.3
Qwen3-8B 78.40 31.28 -60.1
LongFactGPT-4o-mini 89.80 49.11 -45.3
GPT-5-mini 89.89 50.58 -43.7
LLaMA-3.1-8B 84.69 22.73 -73.2
Mistral-7B 89.16 37.04 -58.5
Qwen3-8B 87.07 40.61 -53.4
BiographyGPT-4o-mini 84.53 22.65 -73.2
GPT-5-mini 83.19 15.20 -81.7
LLaMA-3.1-8B 59.55 12.03 -79.8
Mistral-7B 83.90 12.38 -85.2
Qwen3-8B 70.82 13.24 -81.3
AlpacaFactGPT-4o-mini 79.57 52.90 -33.5
GPT-5-mini 75.50 48.42 -35.9
LLaMA-3.1-8B 60.18 28.26 -53.0
Mistral-7B 73.50 33.63 -54.2
Qwen3-8B 70.08 38.43 -45.2
Table 12:Relative percentage drop under Polluted
RAG.We report Clean RAG, Polluted RAG (high-
lighted in red), and the relative percentage drop ∆%=
100(Polluted−Clean)/Clean . Polluted RAG induces
severe performance degradation across all datasets and
models, with drops ranging from approximately 30% to
over 85%, highlighting the fragility of standard RAG
pipelines under evidence pollution.
eration:y comb= LLM θ 
q,D clean∪ D poll
.
Polluted RAG + No RAG. Concatenation of
polluted-retrieval and no-retrieval outputs:
ypoll∥y∅.
Clean RAG + Polluted RAG + No RAG.
Concatenation of combined-retrieval and no-
retrieval outputs:y comb∥y∅.
Claim-Support Graph. Given ycomb andy∅, ex-
tract claims from both; pair by token overlap; label
with NLI in both directions; and keep only ycomb
claims supported by y∅. This improves naive con-
catenation but underperforms MIRAGE
F.1 Answer-Level Baselines
Table 13 compares MIRAGE with several evidence-
consolidation strategies that operate purely at the
answer level. These baselines combine or reweight
answers from polluted and no-RAG runs, or con-
struct a claim-support graph over the generated text
itself.
Simple concatenation schemes partially miti-
gate the worst failures by allowing the verifier to
14

choose between multiple answers, but they still
trail MIRAGE by a wide margin on both datasets.
The claim-support graph baseline is the strongest
answer-level baseline in several settings, but still
trails MIRAGE; for example, on FA V A with GPT-
4o-mini it achieves 68.24% versus 82.54% for MI-
RAGE, and on LongFact74.66%versus88.93%.
F.2 Claim-Graph-Only Limitations
Compared to MIRAGE, the claim-support graph
baseline has several limitations. First, it is sensitive
to coverage, as it relies on token-overlap pairing
between answers; paraphrased or partially overlap-
ping claims may therefore not be compared. Sec-
ond, it lacks an explicit defense mechanism and
cannot stop generation when retrieval is broadly
polluted but internally consistent. Third, it incurs
quadratic cost in the number of claims, which lim-
its scalability. Finally, coherent misinformation
may still pass through if generated answers agree
with each other, whereas MIRAGE operates on the
underlying retrieval graph and can block RAG en-
tirely, falling back to safer parametric generation.
G Runtime and Cost Analysis
We report a runtime and cost diagnostic on 30
randomly sampled queries. Although MIRAGE
retains up to N= 120 candidate claims (yield-
ing 120
2
= 7140 possible pairs), it performs at
most Pmax= 400 sampled pairwise NLI checks
per query. MIRAGE also issues counter-evidence
probes for up to M= 8 claims with at most
kctr= 3passages each, yielding a bounded local
verification budget of roughly 400 + 8×3 = 424
NLI checks per query. The lexical overlap pre-filter
further removes most cross-source pairs before NLI
scoring (Table 18).
All NLI verification uses a local model and there-
fore contributes wall-clock compute but not LLM
API cost. Retrieval uses cached/local evidence, and
the reported monetary cost reflects only LiteLLM-
recordedgpt-4o-mini generation cost. MIRAGE
makes a single LLM generation call per query:
either verified-claims generation when the gate
passes, or parametric no-retrieval generation when
the gate blocks.
Table 15 shows that MIRAGE’s practical verifi-
cation workload is far below exhaustive all-pairs
comparison, averaging only 55–128pairwise NLI
checks per query. Even including counter-evidence
verification, the workload remains well below thedefault cap of 424checks per query. Since verifica-
tion runs locally, API-side cost remains dominated
by a single generation call, yielding approximately
2.6×10−4–4.7×10−4dollars per query.
Compared with prior robust-RAG methods that
require multiple generation calls or iterative re-
trieval (Wang et al., 2025a; Jiang et al., 2023b;
Su et al., 2024), MIRAGE keeps generation to a
single LLM call per query and shifts robustness
computation to bounded local NLI verification. To-
gether with the robustness results in Table 2, this
shows that MIRAGE improves resilience to pol-
luted retrieval while keeping API-side cost close to
a standard single-call RAG pipeline.
H Gate Behavior under Mixed Retrieval
We give a small illustrative example showing how
claim scoring and pruning can discard polluted
evidence while retaining a coherent defended set.
For exposition only, we annotate which retrieved
passages are polluted; MIRAGE does not observe
these labels.
H.1 Setup
Question: “Who was Alan Turing?”
We simulate k=6 retrieved passages, each as-
sociated with a normalized retriever score ˜r(p)∈
[0,1] and a domain trust prior trust(dom(p))∈
[0,1] . All trust values in this example use the same
normalized scale as the main method; the example
is intended to illustrate relative weighting among
signals, not a different or rescaled trust scheme.
H.2 Claim Extraction and Scoring
Assume the claim extractor produces one atomic
claim per passage ( CifromPi), and NLI adds (i) a
support edge between C1andC2(they state consis-
tent birth/death facts), and (ii) contradiction edges
from C4against C1/C2(birth/death mismatch),
from C5against C2(Bletchley vs. NSA/Enigma
confusion), and from C6against C1/C2(death
year / Nobel claim). Following the domain-prior
mapping in Table 10, we assign britannica.com
the generic commercial-domain prior of 0.50,
en.wikipedia.org the reference/encyclopedia
prior of 0.90, andplato.stanford.edu the
academic-domain prior of 0.80. For the three syn-
thetic low-credibility toy domains, we assign a low
prior of 0.20 for exposition. We then compute
s(c) using Equation (3)with˜r(c) = ˜r(p) from the
source passage:
15

LongFact FA V A
Method GPT-4o-mini GPT-5-mini GPT-4o-mini GPT-5-mini
Polluted RAG + No RAG68.10 71.54 63.41 67.83
Clean RAG + Polluted RAG70.90 62.63 63.91 69.04
Clean + Polluted + No RAG64.69 65.84 61.41 64.24
Claim-Support Graph74.66 67.38 68.24 64.68
MIRAGE 88.93 86.45 82.54 79.68
Table 13:Concatenation and claim-graph baselines vs. MIRAGEonLongFactandFA V A(VeriScore F1@ k
median). Rows correspond to different evidence-consolidation strategies; columns report results forGPT-4o-
miniandGPT-5-mini.MIRAGE(bottom, green) consistently achieves the highest F1@ k, outperforming all
concatenation variants and the claim-support graph.
H.3 Defended Set and Gate Decision
Let MIRAGE keep the top N=3 claims, yielding
the defended set S⋆={C1, C2, C3} . Within
S⋆, the overlap-based pairing produces one edge
between C1andC2(support), and no contra-
dictions: edges= 1 ,support_edges= 1 ,
andcontradiction_edges= 0 . Thus,
contradiction_ratio= 0/1 = 0.0 .
Since the claims span three distinct domains,
source_diversity= 3 . Under the global
gate configuration used in our experiments
(rmax= 0.98 anddmin= 2 ), all active gate
conditions are satisfied: 0.0≤0.98 and3≥2 .
The gate therefore passes and the generator is
conditioned only onS⋆.
H.4 Generation Conditioned on Defended
Claims
Alan Turing (23 Jun 1912–7 Jun 1954) was
a British mathematician and logician who
played a major role in early computing
and wartime cryptanalysis [C1][C2].
During World War II, he worked at
Bletchley Park on codebreaking efforts
[C2].
He also introduced the concept of the
Turing machine (1936), a foundational
model for formalizing computation [C3].
I Full Experimental Results
This appendix section provides the full per-dataset
breakdown underlying the averaged robustness re-
sults in the main paper (Table 1). Specifically,
Table 12 summarizes how much vanilla RAG de-
grades when clean evidence is replaced by fully
polluted evidence, and Table 14 reports per-dataset
VeriScore F1@k for all backbones and retrieval
regimes.The No-RAG baseline is a parametric-only gen-
eration baseline without retrieved evidence, while
MIRAGE’s BLOCKED path uses a stronger no-
evidence fallback prompt that explicitly asks the
model to be uncertain, avoid speculation, and state
what would be needed for verification. Therefore,
in regimes where MIRAGE blocks retrieval, differ-
ences from No-RAG should not be interpreted as
evidence-use gains. They reflect removal of pol-
luted context exposure plus the effect of the safer
fallback instruction template. The substantive com-
parison is that MIRAGE avoids conditioning on
polluted retrieval, whereas vanilla RAG remains
exposed to it.
J MIRAGE Pipeline Details
J.1 Claim Graph Construction
MIRAGE employs lightweight sentence segmen-
tation and does not use an LLM for claim decom-
position in any main experiment, ensuring that the
method remains training-free, reproducible, and
low-overhead. For domain extraction, if no URL
is available, the domain is set toUNKNOWN. For
efficiency, we cache token sets for each claim to
accelerate overlap computations.
J.2 Paraphrase Robustness and Efficiency of
Pair Selection
Equation (1)uses a containment-style lexical over-
lap filter as an efficiency-oriented pre-filter before
NLI. This is not intended to be an optimal seman-
tic matching strategy; rather, it avoids exhaustive
cross-source claim-pair comparison and keeps MI-
RAGE lightweight. The filter is effective when
supporting or contradictory evidence preserves key
lexical anchors such as named entities, dates, loca-
tions, and numerical values.
16

Dataset ModelNo-RAG Clean RAG Mixed RAG Polluted RAG
RAG MIRAGE RAG MIRAGE RAG MIRAGE
FA V AGPT-4o-mini 81.20 87.36 90.8055.79 91.5934.83 82.54
GPT-5-mini 70.86 82.18 80.9754.09 86.4736.54 79.68
LLaMA-3.1-8B 69.42 78.15 81.1738.28 66.7737.43 69.61
Mistral-7B 64.91 82.04 81.9841.54 66.7547.64 68.54
Qwen3-8B 74.92 78.40 84.4247.62 78.7031.28 77.35
LongFactGPT-4o-mini 86.90 89.80 91.1358.24 90.3349.11 88.93
GPT-5-mini 85.84 89.89 88.6352.81 86.4750.58 86.45
LLaMA-3.1-8B 86.10 84.69 85.1759.18 86.4322.73 84.61
Mistral-7B 75.05 89.16 89.7260.27 82.1937.04 80.27
Qwen3-8B 84.61 87.07 90.1757.13 87.5640.61 85.11
BiographyGPT-4o-mini 58.99 84.53 84.8142.56 70.1122.65 61.87
GPT-5-mini 54.72 83.19 83.3746.63 68.5015.20 56.97
LLaMA-3.1-8B 56.06 59.55 71.2124.59 69.1312.03 60.06
Mistral-7B 69.21 83.90 85.0126.85 77.7712.38 70.00
Qwen3-8B 53.75 62.18 62.2826.64 61.2513.24 55.49
AlpacaFactGPT-4o-mini 76.41 79.57 83.5058.92 81.7052.90 78.65
GPT-5-mini 69.88 75.50 81.3164.09 80.6448.42 71.71
LLaMA-3.1-8B 62.90 60.18 67.8844.78 65.3128.26 67.09
Mistral-7B 69.73 73.50 79.6651.80 72.7833.63 70.34
Qwen3-8B 65.71 60.79 76.9455.15 72.3438.43 67.79
Table 14:Per-dataset factuality under retrieval pollution.VeriScore F1@k for five LLM backbones on four
long-form QA datasets (FA V A, LongFact, Biography, AlpacaFact). We report No-RAG (parametric-only) and
vanilla RAG under clean retrieval, plus mixed (50% polluted) and fully polluted (100% polluted) retrieval, each
compared against MIRAGE. Table 1 reports the corresponding averages across datasets.
SettingNCandidate Claims Kept Claims Pair NLI Counter Queries Counter Passages LLM $ / q.
Clean 120 45.95 41.74 127.87 8.00 24.00 0.00047
Mixed 120 50.00 40.57 72.42 8.00 24.00 0.00044
Polluted 120 51.17 40.92 55.23 8.00 24.00 0.00026
Table 15: Runtime and cost statistics for MIRAGE on 30 samples, averaged by retrieval setting. Candidate and
kept claims are reported per query. Pair NLI summarizes the local claim-pair verification workload. LLM cost
is the average gpt-4o-mini API generation cost per query recorded by LiteLLM. The released code includes
instrumentation for exact wall-clock reporting in future runs.
PID Domain˜r(p)Tag (for exposition) Excerpt (abridged)
P1britannica.com0.92 Clean Alan Turing (23 Jun 1912–7 Jun 1954) was a British mathematician
and logician; he contributed to wartime cryptanalysis.
P2en.wikipedia.org0.86 Clean Alan Turing (23 Jun 1912–7 Jun 1954) was a British mathematician
and computer scientist; he worked at Bletchley Park on codebreaking.
P3plato.stanford.edu0.79 Clean Turing introduced the notion of a (universal) Turing machine as a
foundational model of computation (1936).
P4historycomputing.blog0.81 Polluted Turing was born in 1913 in Manchester and later led the Manhattan
Project at Los Alamos.
P5ww2-archive.net0.76 Polluted Turing joined the U.S. NSA in 1940 and designed the Enigma cipher
used by Germany.
P6biographyfacts.org0.72 Polluted Turing lived until 1962 and received the Nobel Prize in Physics for
inventing modern computers.
Table 16: Toy mixed retrieval example. “Tag” is shown only to clarify the simulation; MIRAGE only sees text +
domains + scores.
17

CID Domain#support(c)trust(dom(c)) ˜r(c)s(c)Keep?
C1 britannica.com 1 0.50 0.92 1.219 keep
C2 en.wikipedia.org 1 0.90 0.86 1.401 keep
C3 plato.stanford.edu 0 0.80 0.79 0.637 keep
C4historycomputing.blog0 0.20 0.81 0.343 drop
C5ww2-archive.net0 0.20 0.76 0.328 drop
C6biographyfacts.org0 0.20 0.72 0.316 drop
Table 17: Toy scoring under equation (3). Because polluted claims receive no cross-source support (and come from
low-trust domains), they rank below the coherent clean cluster and are pruned away.
SettingNAll pairs NLI pairs Avoided Reduction
MixP 120 971.22 38.28 932.93 96.1%
FullP 120 1214.76 31.15 1183.61 97.4%
Table 18: Lexical pair-selection diagnostic on 30 mixed
and polluted samples. “All pairs” denotes the average
number of possible cross-source claim pairs before fil-
tering; “NLI pairs” denotes the average number sent to
the local NLI verifier after lexical filtering. The filter
avoids most pairwise NLI calls, yielding approximately
25.4× fewer comparisons in MixP and 39.0× fewer
comparisons in FullP.
To quantify this compute-saving effect, we run
a diagnostic analysis on 30 mixed and polluted
samples. Each query contains hundreds to thou-
sands of possible cross-source claim pairs before
filtering. As shown in Table 18, lexical filtering
reduces the average number of pairs sent to NLI
from 971.22 to38.28 in MixP and from 1214.76
to31.15 in FullP, avoiding 96.1% and97.4% of
pairwise NLI calls, respectively. This corresponds
to approximately 25.4× and39.0× fewer pairwise
NLI comparisons.
Lexical filtering can miss paraphrastic agreement
or contradiction, especially under omission, fram-
ing, or heavy rewording rather than shared entities,
dates, or numbers. This sparsifies the claim graph,
leaving fewer support or contradiction edges for
NLI to adjudicate. MIRAGE handles such cases
conservatively, tending to block retrieval rather
than condition generation on weakly connected
evidence.
We expose τoverlap to trade recall for speed.
Dense or hybrid semantic pair selection could
improve paraphrase recall, but would increase
retrieval-time computation; we leave this variant to
future work. We cap candidate pairs per query at
Pmaxto control NLI inference cost.J.3 Counter-evidence Probing Details
For the top- Mclaims (ranked by ˜r(c)), we con-
struct a dispute-style query qcounter(c) = { claim }
myth false controversy dispute correction retrac-
tionand retrieve up to kcounter additional passages
using the same retriever. The added terms are
generic and dataset-independent: they are intended
to surface refutations, corrections, or disagreement-
oriented evidence without using any ground-truth
labels.
We then apply NLI between cand each retrieved
passage. When pcontr> p entail andpcontr≥θ supp,
we add a self-contradiction edge with weight wcontr.
These edges penalize claims that appear plausible
in the original retrieval set but are contradicted
by targeted counter-evidence, complementing the
cross-source support and contradiction edges in the
main claim graph.
J.4 Domain Trust Prior
We define a fixed trust prior trust(dom)∈[0,1] ,
assigning higher values to government and educa-
tional domains (e.g., .gov ,.edu ), Wikipedia, and
major news outlets, followed by generic .com do-
mains and user-generated forums. This mapping is
query-independent and used only as a lightweight
source prior.
In MIRAGE, domain trust is strictly secondary:
it can break ties among otherwise similar claims,
but it is never sufficient on its own and cannot
override multi-source agreement. This prevents
high-reputation but polluted passages from domi-
nating the graph, while still allowing source quality
to stabilize scoring.
This follows web-ranking practice, where query-
independent authority signals such as TrustRank-
style propagation (Soricut and Echihabi, 2010) are
auxiliary features rather than primary evidence.
Similarly, our domain-level features complement
content-based consistency signals.
18

K Examples of Polluted Passages
We present representative examples of perturba-
tions from the LongFact evidence pool. Each ex-
ample consists of a clean passage and its corre-
sponding polluted variant. We group perturbations
into four categories.
K.1 Unambiguous (Single-Fact Edits)
Clean:“The International Criminal Court (ICC
or ICCt) is an intergovernmental organization and
international tribunal that sits in The Hague in the
Netherlands.”
Polluted:“The International Criminal Court (ICC
or ICCt) is an intergovernmental organization and
international tribunal that sits inGeneva, Switzer-
land.”
K.2 Conflicting (Fact / Polarity Flips)
Clean:“The Charter of the United Nations is the
founding document of the United Nations. It was
signed on 26 June 1945, in San Francisco, at the
conclusion of the United Nations Conference on
International Organization, and came into force on
24 October 1945.”
Polluted:“The Charter of the United Nations is
the founding document of the United Nations. It
was signed on26 June 1950, inLos Angeles, at
the conclusion of the United Nations Conference
on International Organization, and came into force
on24 October 1950.”
K.3 Misleading (Selective Framing)
Clean:“Kevin David Mitnick (August 6, 1963
– July 16, 2023) was an American computer se-
curity consultant, author, and convicted hacker.
In 1995, he was arrested for various computer
and communications-related crimes, and spent five
years in prison after being convicted of fraud and
illegally intercepting communications.”
Polluted:“Kevin David Mitnick (August 6, 1963
– July 16,2022)isan American computer secu-
rity consultant, author, and convicted hacker. In
1995, he was arrested forminor offenses related
to hacking, andserved three years in prisonaf-
ter being convicted oftheft and trespassing on
private property.”
K.4 Fabricated (Invented Details)
Clean:“America’s Cup match races are held be-
tween two sailing yachts: one from the yacht club
that currently holds the trophy (known as the de-
fender) and the other from the yacht club that ischallenging for the cup (the challenger). The win-
ner is awarded the America’s Cup trophy, infor-
mally known as the Auld Mug.”
Polluted:“America’s Cup match races are held
betweenthree sailing yachts: one representing the
yacht club that currently holds the trophy (known
as the defender), andtwo others from differ-
ent yacht clubs. The winner of the final race
is awarded the America’s Cup trophy, informally
known as theSilver Cup, which wasfirst intro-
duced in 1820.”
19
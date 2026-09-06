# TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning

**Authors**: Muhaimin Bin Munir, Akib Jawad Ononto, Nazia Shehnaz Joynab, Bhavani Thuraisingham, Latifur Khan

**Published**: 2026-08-31 23:21:27

**PDF URL**: [https://arxiv.org/pdf/2609.00470v1](https://arxiv.org/pdf/2609.00470v1)

## Abstract
Retrieval-Augmented Generation (RAG) grounds large language models in external corpora, but implicit trust in retrieved documents creates a critical attack surface: PoisonedRAG shows that a handful of crafted passages can dominate dense retrieval and steer generation toward attacker-chosen answers. We present the Tri-Layer Sieve, a middleware defense that sanitizes retrieved evidence through cross-embedding-space clustering with an independent judge model, structural filtering of trigger-payload artifacts, and LLM consistency verification. The design exploits a key weakness of retrieval-stage poisoning: a single document must satisfy one embedding geometry, one internal Trigger-Payload structure, and one generation objective - rarely all three simultaneously, a fragility that persists even against an adaptive attacker who paraphrases around it. On Natural Questions, HotpotQA, and MS-MARCO with Contriever retrieval (k=50), the Sieve reduces black-box Attack Success Rate from 67.0/87.0/64.0% to 3.0/14.0/4.0%, mitigates white-box HotFlip attacks from ~74% to 27.8% on NQ with Layer 3 enabled, and drives poisoned-document MRR to 0.000, while restoring clean accuracy from 13-33% under attack to 58-76%. Under an architecture-aware adversary who paraphrases triggers to evade the structural filter, enabling the consistency layer halves adaptive ASR (32.0% to 15.0% on NQ) while raising clean accuracy by 18 points, at an added latency of ~16-19 s/query under live retrieval.

## Full Text


<!-- PDF content starts -->

TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning
Muhaimin Bin Munir1, Akib Jawad Ononto1, Nazia Shehnaz Joynab1,
Bhavani Thuraisingham1,Latifur Khan1,
1University of Texas at Dallas,
Correspondence:muhaimin.binmunir@utdallas.edu
Abstract
Retrieval-Augmented Generation (RAG)
grounds large language models in exter-
nal corpora, but implicit trust in retrieved
documents creates a critical attack surface:
PoisonedRAG shows that a handful of crafted
passages can dominate dense retrieval and steer
generation toward attacker-chosen answers.
We present the Tri-Layer Sieve, a middleware
defense that sanitizes retrieved evidence
through cross-embedding-space clustering
with an independent judge model, structural
filtering of trigger–payload artifacts, and LLM
consistency verification. The design exploits a
key weakness of retrieval-stage poisoning: a
single document must satisfy one embedding
geometry, one internal Trigger–Payload
structure, and one generation objective —
rarely all three simultaneously, a fragility
that persists even against an adaptive attacker
who paraphrases around it. On Natural
Questions, HotpotQA, and MS-MARCO
with Contriever retrieval ( k= 50 ), the Sieve
reduces black-box Attack Success Rate from
67.0/87.0/64.0% to 3.0/14.0/4.0%, mitigates
white-box HotFlip attacks from ∼74% to
27.8% on NQ with Layer 3 enabled, and drives
poisoned-document MRR to 0.000 , while
restoring clean accuracy from 13–33% under
attack to 58–76%. Under an architecture-aware
adversary who paraphrases triggers to evade
the structural filter, enabling the consistency
layer halves adaptive ASR (32.0% →15.0%
on NQ) while raising clean accuracy by 18
points, at an added latency of ∼16 –19s/query
under live retrieval.
1 Introduction
Large language models are limited by their para-
metric memory: they cannot be updated contin-
uously, hallucinate plausible-but-false statements,
and struggle with domain-specific facts. Retrieval-
Augmented Generation (RAG) (Lewis et al., 2020)
addresses this by grounding outputs in external cor-pora retrieved at inference time, and is now the de
facto architecture for knowledge-intensive search,
open-domain question answering, and enterprise
deployments (Gao et al., 2023).
This advantage introduces a corresponding secu-
rity flaw: a RAG system consumes mutable exter-
nal data at inference time and treats every retrieved
passage as trustworthy by default. Zou et al. (2025)
(PoisonedRAG) showed this implicit trust can be
exploited — inserting as few as five crafted docu-
ments into a corpus such as MS MARCO (Bajaj
et al., 2018) or a Wikipedia snapshot suffices to
dominate dense retrieval and force attacker-chosen
misinformation, even when the model knows the
correct answer parametrically. Follow-up work ex-
tends the threat model to jamming (Shafran et al.,
2025), agent memory poisoning (Chen et al., 2024),
backdoor triggers (Cheng et al., 2024), and bench-
marks (Zhang et al., 2025), establishing retrieval-
stage poisoning as a distinct, under-defended threat
surface.
Existing defenses are insufficient for two reasons.
First, grounding-verification and hallucination-
detection methods (Huang et al., 2024; Song et al.,
2024) act on theoutputof generation rather than
the integrity of retrieved evidence, and so can-
not prevent theretrieval dominancePoisonedRAG
exploits.Second, defenses targeting poisoned
retrieval — trust-weighted scoring (Zhou et al.,
2025), certified isolate-and-aggregate decoding (Xi-
ang et al., 2024), and query paraphrasing (Zou et al.,
2025) — each address only one constraint a poi-
soned document must satisfy, and can trade sharply
against utility on complex queries when tuned ag-
gressively (Section 6.2).
We argue that retrieval-stage poisoning succeeds
precisely because the attacker must simultane-
ously satisfy several distinct objectives — dense-
retrieval similarity under a specific encoder, inter-
nal coexistence of a retrieval-optimizedTrigger
and a generation-optimizedPayload, and factual
arXiv:2609.00470v1  [cs.CL]  31 Aug 2026

plausibility against the LLM’s parametric knowl-
edge — and that defenses should exploit this multi-
objective fragility rather than detect surface arti-
facts.
Contributions.We proposeTRIS(the Tri-
Layer Retrieval Integrity Sieve)1, a middleware
defense that intercepts retrieved documents be-
tween the retriever and the generator and applies
three orthogonal filters — cross-embedding-space
clustering, structural trigger–payload detection,
and LLM consistency verification — each tar-
geting one of the three constraints a successful
retrieval-stage poison must simultaneously satisfy.
Against the PoisonedRAG suite on Natural Ques-
tions (Kwiatkowski et al., 2019), HotpotQA (Yang
et al., 2018), and MS-MARCO (Bajaj et al., 2018)
with Contriever (Izacard et al., 2022), TRIS reduces
black-box ASR by an order of magnitude on NQ
and MS-MARCO while recovering 41–45points
of clean accuracy over the attacked baseline, and
outperforms TrustRAG (Zhou et al., 2025) on MS-
MARCO ( 4%vs.20% ASR); against TrustRAG
and RobustRAG (Xiang et al., 2024) at fair operat-
ing points it is competitive rather than dominant, at
substantially lower verification cost (Section 6.2).
We further contribute layer-wise ablations, an ab-
lation isolating Layer 1’s embedding geometry,
and an empirical evaluation of architecture-aware
(Level 1) adaptive adversaries under live retrieval
and a forced-top worst case. We use TRIS and the
Tri-Layer Sieve interchangeably.
2 Background and Related Work
2.1 Retrieval-Augmented Generation
RAG (Lewis et al., 2020) augments LLMs with
non-parametric memory: a query qis embedded
byfQand the top- kdocuments in a corpus Dare
retrieved by s(q, d) = sim(f Q(q), f D(d)), after
which an LLM conditions on that evidence. Dense
retrievers such as DPR (Karpukhin et al., 2020) and
Contriever (Izacard et al., 2022) are now standard.
RAG offers stronger factuality than parametric-
only LLMs (Gao et al., 2023), but trusts retrieved
evidence by default, opening a new attack surface.
2.2 Knowledge Poisoning Attacks
Classical data poisoning targets training pipelines
(Geiping et al., 2021; Steinhardt et al., 2017; Gu
et al., 2019). Retrieval-stage poisoning is a newer
1Code, evaluation scripts, and poison-generation data:
https://github.com/akibjawad14/tristhreat that requires only corpus access (Zou et al.,
2025; Shafran et al., 2025; Edemacu et al., 2025).
PoisonedRAG.Zou et al. (2025) give the first
systematic demonstration of retrieval-stage poison-
ing for LLM-based RAG. Each poisoned document
isdadv=S⊕I : a retrieval-optimized Trigger
Sand a generation-optimized Payload I. Black-
boxSrepeats the query; white-box Sis gradient-
optimized via HotFlip (Ebrahimi et al., 2018) or
universal triggers (Wallace et al., 2019). Five poi-
sons per query suffice for ASR>90%.
2.3 Existing Defenses
Output-side methods(Song et al., 2024) detect
post-generation mismatches but cannot prevent re-
trieval dominance.TrustRAG(Zhou et al., 2025)
re-ranks via a single learned trust score (with an op-
tional LLM-consistency check); effective on simple
attacks, weaker on MS-MARCO (Section 6). TRIS
differs architecturally, not just numerically: three
independent off-the-shelf checks rather than one
learned scorer, each targeting a different constraint,
evaluated against an explicit adaptivity taxonomy
(Section 3.2). We scope that difference precisely:
Layer 2 targets verbatim and near-duplicate injec-
tion — the common case — where TrustRAG has
no dedicated structural filter (ROUGE-L overlap is
its closest analogue). Under paraphrased triggers
Layer 2 fires on roughly zero documents per query
and TrustRAG’s more lenient check catches more,
so we claim Layer 2 for the common attack, not as
a paraphrase defense (Section 6.5).
RobustRAG(Xiang et al., 2024) gives
isolate-and-aggregate decoding certified against k-
corruption, but strict isolation harms clean accuracy
on multi-hop queries.Query paraphrasinghelps
only marginally, since dense retrievers map para-
phrases into similar embedding regions (Zou et al.,
2025).Perplexity detectionfails because LLM-
generated payloads are oftenmorefluent than gen-
uine web text (Shafran et al., 2025).Prompt injec-
tion(Greshake et al., 2023) is related but distinct:
our payloads encode a false fact, not an instruction.
3 Threat Model
We consider an adversary whose goal is to manipu-
late the output of a RAG system by injecting adver-
sarial content into its external corpus. Our threat
model follows Zou et al. (2025) and generalizes to
both structured and unstructured corpora common
in real deployments. Figure 2 in Appendix A traces

this attack pipeline end to end for a single example
query.
3.1 System and Adversary Model
A RAG pipeline consists of (i) a retriever Mretthat
embeds qand retrieves the top- kfrom corpus D,
and (ii) a generator Mgenthat conditions its output
onDk. The generator trusts retrieved evidence
by default, matching production deployments over
mutable corpora.
The adversary may (a)inject documentsinto
D, (b)craft adversarial triggersthat maximize
similarity with q∗underMret(verbatim query rep-
etition in the black-box setting; HotFlip (Ebrahimi
et al., 2018) gradient optimization in the white-
box setting), and (c)embed malicious payloads
that express yadvas an authoritative encyclopedic
claim. These capabilities instantiate the Split-and-
Merge construction of Zou et al. (2025): a poisoned
document concatenates a retrieval-optimized Trig-
ger with a generation-optimized Payload, indepen-
dently optimizing retrieval dominance and output
steering.
The adversary’s goals are to dominate retrieval
forq∗, override the LLM’s internal knowledge, and
induce a targeted output yadv̸=ytrue(Zou et al.,
2025; Shafran et al., 2025). The defender (i) can-
not retrain MretorMgen, (ii) cannot modify D
beyond lightweight preprocessing, and (iii) must
operate as a middleware layer — constraints reflect-
ing deployed enterprise settings.
3.2 Adaptive Adversaries
A reviewer might reasonably ask whether TRIS
(Section 4) withstands attackers who know about it.
We distinguish four levels of adaptivity, of which
the evaluated PoisonedRAG attacks are Level 0:
L0 (static): the attacker is unaware of the de-
fense — PoisonedRAG black-box and white-box,
our headline setting.L1 (architecture-aware): the
attacker knows TRIS is deployed but not its pa-
rameters, and may paraphrase rather than repeat q
(defeating Layer 2) and write encyclopedic-style
payloads (degrading Layer 3); Layer 1’s geometric
check still applies.L2 (judge-aware): the attacker
also knows Jand jointly optimizes for rank un-
derRandmajority-cluster membership under J
— a conjunction constrained by the geometric inde-
pendence of the two spaces, and compounded by
ensembling or rotating J.L3 (full white-box): the
attacker also knows Layer 2’s thresholds and yint,and must craft a payload contradicting yintthat the
LLM finds equally plausible.
We now evaluate Level 1 empirically (Sec-
tion 6.5); Levels 2 and 3 remain conceptual only,
and closing that gap is the most important remain-
ing limitation of our evaluation (Section 8). A core
claim of this work — now supported empirically
for Level 1 — is that defeating an n-layer orthog-
onal defense requires the attacker to satisfy the
conjunction of nconstraints in non-aligned objec-
tive landscapes, which substantially enlarges the
attack search space even without certified bounds.
3.3 Out of Scope
We exclude (i) training-set poisoning, (ii) prompt-
injection attacks where the user is malicious, and
(iii) attacks that require control of LLM weights.
4 The Tri-Layer Sieve
The Tri-Layer Sieve is a defense-in-depth middle-
ware between the retriever and the generator. Its
central principle is that no retrieved document is
trusted at face value: every passage is checked
against up to three orthogonal filters before reach-
ing the generator, with the third invoked adaptively
rather than on every document. The three layers
target the three constraints that retrieval-stage poi-
soning must simultaneously satisfy: retrieval-side
embedding similarity, internal trigger–payload co-
existence, and factual alignment with parametric
knowledge.
Figure 1 diagrams the resulting pipeline.
4.1 Layer 1: Semantic Clustering via Model
Diversity
Poisoned documents are optimized against a spe-
cific retriever encoder, and adversarial trigger op-
timization (Ebrahimi et al., 2018; Wallace et al.,
2019) exploits the geometry of that encoder rather
than true semantic content. Layer 1 is therefore
designed asmajority-keep in an architecturally in-
dependent geometry: rather than assuming poisons
fail to cluster near the query under a second en-
coder, it assumes nothing about where they land
and simply retains the majority cluster. We frame
this as a design property — it drops the require-
ment that poisons form a tight, separable cluster
in the retriever’s own space — not as a source of
empirical gain; Appendix B.3 tests the geometry
directly and finds it is not the differentiating factor.
Layer 1 re-embeds the top- kretrieved docu-
ments with a judge model Jthat is architec-

User Query 
qevaluates candidate 
documents DkDEFENSE MIDDLEWARE 
TRIS — Tri-Layer Retrieval 
Integrity Sieve forwards filtered 
evidence ⊆ Dk
LLM Generation Final Answer 
ytrue, not yadvDense Retrieval 
(Contriever) 
top-k, k = 50 
Inside the Sieve — every passage must survive all three checks 
LAYER 1 (L1) 
Cross-embedding-space 
clustering 
Retrieved documents are re-embedded 
with an independent judge model and 
clustered in that space. Documents 
outside the majority cluster are filtered. Structural filtering of 
trigger-payload artifacts 
The document prefix is checked for 
query overlap and n-gram patterns 
associated with trigger-payload 
poisoning. Documents that exceed the 
configured thresholds are removed. LAYER 2 (L2) 
LLM Consistency 
Verification 
An optional consistency step checks 
surviving documents against the 
language model's own answer and 
removes conflicting evidence. LAYER 3 (L3) Figure 1:High-level architecture of the Tri-Layer Sieve.The Sieve sits between the retriever and the generator,
intercepting candidate documents and forwarding only those that pass the applicable filters (Layer 3 fires adaptively,
not on every document). The defense requires no modification of the retriever or LLM.
turally distinct from the retriever R— we use
Sentence-BERT all-MiniLM-L6-v2 (Reimers and
Gurevych, 2019). After L2-normalization we run
K-Means with a small cluster count Cand iden-
tify the majority cluster; we sweep C∈ {2,3,5}
in Section 6 and find C= 3 the default sweet
spot. Poisons typically appear as isolated micro-
clusters, high-density off-distribution clusters, or
low-similarity outliers. Documents outside the ma-
jority cluster are dropped, with a fallback retaining
all documents when no clear majority exists.
4.2 Layer 2: Content-Based Structural
Filtering
Layer 2 targets the structural footprint of Split-and-
Merge: Trigger and Payload need not share se-
mantic content, but must coexist in one document,
producing lexical irregularities at the prefix. For
each surviving document we compute token-level
Jaccard similarity between qand the first T= 20
tokens, bigram/trigram overlap between qand the
prefix, and length/repetition patterns; if Jaccard or
n-gram overlap exceeds a configurable threshold
(we use 0.8), the document is discarded. Unlike
perplexity filtering — which fails because LLM
payloads are highly fluent (Shafran et al., 2025) —
this detects the poison’sfunctional decomposition.
Evasion is cheap at the retrieval stage, however:
Section 6.5 shows a paraphrased trigger keeps near-
identical rank while falling below the threshold,
shifting the burden onto Layers 1 and 3. Layer 2
costs∼8ms per query.
4.3 Layer 3: LLM Consistency Checking
Layer 3 verifies semantically by prompting an LLM
(the generator Mgenin self-knowledge mode, ora separate verifier) to judge each document in iso-
lation, which defeats the dominance effect when
many retrieved passages repeat the same poisoned
claim; LLMs retain strong parametric priors even
when retrieval corrupts the final answer (Longpre
et al., 2021). The verifier (i) answers qwithout con-
text to capture its parametric belief yint, (ii) sum-
marizes each document’s claim relevant to q, and
(iii) judges it compatible with yint, contradictory, or
uncertain; only strong contradictions without evi-
dence of a legitimate update are dropped. Crucially,
if the verifier is not confident — or the call fails
or returns a malformed response — the document
fails openand is retained, so Layer 3 can never
remove a document it cannot judge, and verifier
unavailability degrades it to a no-op rather than
discarding evidence under uncertainty (this under-
lies the valid-response-only rows in Section 5 and
Table 4). Two invocation modes exist:always-on
(used for L3 ablation rows) andadaptive, invoked
only when Layers 1 and 2 disagree.
4.4 Composition, Default Configuration, and
Engineering
The three layers arecomplementary in coverage
rather than strictly additive: triggers that evade
Layer 1 are typically caught by Layer 2’s prefix
overlap check, and payloads that survive Layer 2
trigger contradictions in Layer 3. Composing all
three does not strictly dominate every pairwise
combination, however, since Layer 3 occasion-
ally discards clean passages whose phrasing differs
from the LLM’s prior.
We therefore recommenda single default:
L1+L2 with Layer 3 in adaptive mode, invoked
only where Layers 1 and 2 disagree. It attains

the lowest black-box ASR ( 3.0% on NQ) at ∼
12ms/query, and adaptive invocation recovers most
of Layer 3’s white-box and architecture-aware ro-
bustness without paying always-on latency on ev-
ery query. Deployments with a known threat pro-
file should override this: white-box-dominant set-
tings useL1+L3 always-on( 27.8% white-box
ASR),and settings expecting trigger-paraphrasing
adversaries should enable Layer 3 always-on in-
stead (Section 6.5 gives the resulting ASR, accu-
racy, and latency). Abstract numbers correspond to
the L1+L2 default unless stated. TRIS is a drop-in
Python module wrapping the retrieval call: model-
agnostic, batched-cache friendly, falling back to
unfiltered Dkwith a logged warning when all doc-
uments are filtered.
5 Experimental Setup
Datasets and attack.We evaluate on Natural
Questions (Kwiatkowski et al., 2019), HotpotQA
(Yang et al., 2018), and MS-MARCO (Bajaj et al.,
2018). For each we sample 100target queries; the
Poison Factory generates five poisoned passages
per query via Split-and-Merge (Zou et al., 2025)
with a counterfactual yadv, a poisoning ratio well
under 1%of the corpus. We considerblack-box
LM-targeted(trigger = query repeated with light
paraphrasing) andwhite-box HotFlip(gradient-
optimized against Contriever), with k= 50 and
10iterations ofM= 10queries.
Adaptive-adversary experiments.For Sec-
tion 6.5 we construct three additional Level-1 at-
tack variants:trigger-paraphrased(trigger para-
phrased rather than repeated verbatim),payload-
diversified(five independently-phrased payloads
rather than a shared template), andfull adaptive
(both combined). The live-retrieval comparison
(Table 5) follows the same protocol as above ( n=
100,k= 50 ) on NQ and HotpotQA. The forced-
top worst case (Table 7) pins poisons to the top of
context to isolate defense behavior from retrieval
competition ( n= 30 , NQ). The extended injection
sweep (Table 9) uses a reduced sample ( n= 25 ,
NQ) due to the added query volume at each injec-
tion depth.
Fair-baseline experiments.The RobustRAG
threshold sweep (Table 2) re-runs the black-box
attack at n= 30 on both datasets for each
α∈ {0.1,0.2,0.5} plus majority aggregation; the
TrustRAG comparison with its LLM check en-abled (Table 3) uses n= 30 on NQ under the
two adaptive variants above. The Layer-1 judge-
geometry ablation (Appendix B.3) runs at full scale
(n= 100 ,k= 50 , both datasets), swapping only
Layer 1’s embedding model. Reported p-values are
two-sided two-proportion z-tests over the n= 100
per-condition query outcomes, treating queries as
independent.
Models.Contriever (Izacard et al., 2022) re-
triever (HuggingFace, FP16); Sentence-BERT
all-MiniLM-L6-v2 (Reimers and Gurevych,
2019) judge; FAISS IndexFlatIP (Johnson et al.,
2021) on L2-normalized embeddings; GPT-3.5-
turbo (OpenAI, 2023) generator (temperature 0).
We cross-check with Llama-2-7B-chat (Touvron
et al., 2023) and obtain consistent trends. NVIDIA
A10G GPUs. Commercial API calls occasionally
failed transiently (timeouts, rate limits); Layer 3
fails open on such failures (Section 4). Where fail-
ures affected part of a condition we compute the
affected numbers over valid responses only, mark-
ing them ∼(Table 4); where an entire cell was lost
we omit the cell (Table 10).
Baselines.(i) No Attack (Clean); (ii) Vanilla
RAG (Attacked); (iii) TrustRAG (Zhou et al.,
2025), run in its base configuration — trust-score
re-ranking with the optional LLM-consistency
checkdisabled— for Table 1, and with that check
enabledfor the adaptive comparison in Table 3;
(iv) RobustRAG (Xiang et al., 2024) with keyword-
based isolate-and-aggregate decoding, using ma-
jority aggregation for Table 1 and swept over the
aggregation threshold α∈ {0.1,0.2,0.5} in Sec-
tion 6.2. Our RobustRAG implementation extracts
keywords with a regex-plus-stopword heuristic
over the retrieved documents, a faithful but simpli-
fied stand-in for the authors’ LLM-based extraction
step; we flag this as a possible source of divergence
from their reported numbers.
Metrics. ASR( ↓): attack succeeds if the gen-
erated answer contains yadv(case-normalized,
article-stripped exact match).CleanAcc( ↑): the
answer contains the gold string under the same
normalization.Retrieval metricson NQ: Re-
call@5, Recall@50, Clean MRR (over benign pas-
sages), Poisoned MRR (over adversarial passages).
PoisonedMRR = 1.000 means every poison ranks
first;0.000means full removal.

5.1 Evaluation Protocol
Unless a table caption states otherwise, each cell
aggregates 10iterations of M= 10 queries ( 100
trials). The reduced-scale tables — Tables 2, 3, 7,
and 9 — print their own nin the caption and are sin-
gle runs at that scale, not 10-iteration aggregates;
differences of a few points in those tables should
not be read as resolved. Within each iteration we
re-sample queries, re-generate poisons, and re-run
the pipeline, capturing variance from poison gener-
ation, retrieval order, and LLM decoding (despite
temperature 0, GPT-3.5-turbo shows residual out-
put variation). Pairwise differences >3points are
stable across re-runs; smaller ones are within noise.
Clean Recall@5 of 0.180 reflects our 100-query
Wikipedia subset.
6 Results
6.1 End-to-End Defensive Efficacy
Table 1 reports clean accuracy and ASR across
all three datasets under the black-box LM-targeted
attack. Vanilla RAG is catastrophically vulnerable:
ASR ranges from 64.0% (MS-MARCO) to 87.0%
(HotpotQA), and clean accuracy collapses from
74–82% to13–33% under attack. TRIS restores
robustness across all three datasets: ASR drops to
3.0% on NQ, 14.0% on HotpotQA, and 4.0% on
MS-MARCO, while CleanAcc recovers by 41,45,
and44points respectively over the attacked system
and matches or exceeds every baseline on all three
datasets. The residual gap to the no-attack ceiling
is2points on NQ and 6on MS-MARCO, but 16
on HotpotQA, where Layer 3 over-filters multi-hop
evidence.
Comparison with prior defenses.The picture
is dataset-dependent and worth reporting honestly.
RobustRAG (Xiang et al., 2024) attains the lowest
HotpotQA ASR ( 1.0% ) here, but at 5.0% clean ac-
curacy — the aggressive end of its trade-off rather
than a property of the method; Section 6.2 sweeps
its threshold and we rest no claim on this point.
TrustRAG (Zhou et al., 2025) is the strongest com-
petitor, matching TRIS on NQ ( 2%vs.3%ASR
at identical CleanAcc) and slightly beating it on
HotpotQA ( 11% vs.14% ASR at 57% vs.58%
CleanAcc), with its LLM-consistency check dis-
abled here (Table 3 enables it); on MS-MARCO
it lags substantially ( 20% vs.4%ASR, 16points
lower CleanAcc). Comparable on NQ, marginally
behind on HotpotQA at within-noise margins (Sec-tion 5.1), substantially ahead on MS-MARCO: we
frame the contribution as the strongestworst-case
profile across benchmarks, not the best result on
each.
6.2 Fair Baseline Operating Points
Table 1 fixes each baseline at one configuration, but
both expose a tuning knob that moves them along
an ASR–utility frontier, so a single-point compar-
ison can flatter either system. Table 2 sweeps Ro-
bustRAG’s threshold α, and Table 3 re-evaluates
TrustRAG with its LLM-consistency check enabled.
Both are reduced-scale ( n= 30 ): enough to locate
a frontier, not to resolve a few points.
Tuned fairly, RobustRAG is far stronger than Ta-
ble 1 suggests. At α= 0.2 it reaches 6.7% /36.7%
(ASR/CleanAcc) on NQ and 3.3% /66.7% on Hot-
potQA — competitive with TRIS on NQ and better
on both axes on HotpotQA.We therefore do not
claim ASR dominance over a fairly-tuned Ro-
bustRAG.Two caveats cut in opposite directions:
these are n= 30 against TRIS’s n= 100 , and our
RobustRAG uses heuristic rather than LLM-based
keyword extraction (Section 5).
What does separate the systems at comparable
ASR is verification cost: TRIS’s L1+L2 default
needs one MiniLM pass and one generation call ( ∼
0.35 s/query), against RobustRAG’s per-document
isolation ( 31–36calls/query) — a ∼36× reduction.
This is regime-specific, not a uniform win: under
paraphrased triggers TRIS needs Layer 3, whose
cost returns to that range (Section 6.5).
TrustRAG’s optional LLM-consistency check
— its closest analogue to Layer 3 — is likewise
disabled in Table 1. Enabled, under the adaptive
attacks of Section 6.5, the two systems are indis-
tinguishable at n= 30 : every margin is one query
wide, with TrustRAG marginally ahead on trigger-
paraphrased poisons (Table 3).
We therefore withdraw any claim of
an adaptive-ASR advantage over a fairly-
configured TrustRAG, and rest the comparison
on architecture instead: a dedicated structural
layer, an explicit fail-open verdict in Layer 3,
and a modular composition whose layers toggle
independently at known cost.
A further ablation isolates how much Layer 1’s
independent embedding geometry contributes.
Swapping only that space — MiniLM versus the
retriever’s own Contriever, n= 100 , both datasets
— leaves ASR statistically unchanged (NQ 32.0%
vs.33.0% ,p= 0.88 ; HotpotQA 44.0% vs.53.0% ,

NQ HotpotQA MS-MARCO
DefenseCleanAcc↑ASR↓CleanAcc↑ASR↓CleanAcc↑ASR↓
No Attack (Clean) 76.0% — 74.0% — 82.0% —
Vanilla RAG (Attacked) 33.0% 67.0% 13.0% 87.0% 32.0% 64.0%
RobustRAG (Xiang et al., 2024) 63.0% 2.0% 5.0% 1.0% 76.0% 5.0%
TrustRAG (Zhou et al., 2025) 74.0% 2.0% 57.0% 11.0% 60.0% 20.0%
Tri-Layer Sieve (Ours)74.0% 3.0% 58.0% 14.0% 76.0% 4.0%
Table 1:End-to-end defensive efficacyon NQ, HotpotQA, and MS-MARCO under PoisonedRAG black-box
LM-targeted attack ( n= 100 queries per dataset, k= 50 , Contriever, GPT-3.5-turbo). The Sieve substantially
outperforms TrustRAG on MS-MARCO ( 4%vs.20% ASR) while matching CleanAcc. Baselines are shown
here in the single configuration used at submission time — TrustRAG with its LLM-consistency check disabled,
RobustRAG with majority aggregation; Section 6.2 re-examines both across their tuning knobs, and those results,
not this table alone, are the basis for our comparative claims.
NQ HotpotQA
SettingASR↓Clean↑ASR↓Clean↑
Majority†16.7% 23.3% 13.3% 53.3%
α= 0.110.0% 36.7% 6.7% 63.3%
α= 0.26.7% 36.7% 3.3% 66.7%
α= 0.56.7% 23.3% 3.3% 60.0%
Table 2:RobustRAG operating-point frontier(black-
box LM-targeted attack, n= 30 , reduced scale). Sweep-
ing the keyword-aggregation threshold αmoves Robus-
tRAG along an ASR–utility frontier; α= 0.2 dominates
the majority-aggregation setting on both axes and both
datasets.†The majority setting used in Table 1. This
reduced-scale re-run does not reproduce the extreme
clean-accuracy figure of the full-scale majority row in
Table 1 ( 5.0% on HotpotQA); we report both rather than
reconcile them, and base no claim on that figure.
p= 0.20 ), whereas Layer 2 accounts for the large
effect ( p <10−7; Appendix B.3). We report this
as a negative result about our own design rationale:
the independent geometry is a robustness property
— it drops the assumption that poisons form a sep-
arable cluster in the retriever’s space — not the
source of the measured gain.
6.3 Retrieval Dynamics
To understandhowthe Sieve works, Table 6 (Ap-
pendix B.1) reports retriever-level metrics on NQ.
Vanilla poisoning eliminates Recall@5 and drives
PoisonedMRR to its maximum of 1.000 — a poi-
son at rank 1 for every query. The Sieve fully
inverts this: Recall@5 and Clean MRR return to
clean-corpus baselines and PoisonedMRR falls to
0.000 . Recall@50 is unchanged throughout, con-
firming that poisons remain in the candidate pool
but no longer reach the visible context: the Sieve
reorders rather than discards. TrustRAG achieves
the same retriever-side outcome on NQ, consistentTRIS (full) TrustRAG‡
AttackASR↓Clean↑ASR↓Clean↑
Trigger-para. 27% 27% 20% 33%
Full adaptive 33% 30% 30% 27%
Table 3:TRIS vs. TrustRAG with its LLM-
consistency check enabled( n= 30 , NQ).‡Unlike
Table 1, TrustRAG runs with its optional LLM check on.
The adaptive poison set differs from Table 7, so rows are
not comparable across the two. At n= 30 one query is
3.3points: every margin here is one query wide.
with its strong NQ end-to-end numbers; the Sieve’s
advantage emerges downstream on harder datasets.
6.4 Layer Ablation
Table 4 isolates the contribution of each layer on
NQ under both black-box and white-box attacks.
Three findings stand out.
Layer 2 dominates black-box.Alone, the struc-
tural filter reduces ASR from 67% to4%, capturing
the lexical signature of Split-and-Merge prefixes.
This is consistent with the design intuition: black-
box triggers repeat the query verbatim, producing
prefix overlap that is trivially detected.
No single layer suffices for white-box.Against
HotFlip, Layer 2 alone yields little benefit ( ∼
71%), since gradient-optimized triggers evade lex-
ical overlap; only L1+L3 reduces substantially
(27.8% ). Defeating white-box attacks therefore
requires both geometric diversity (Layer 1 disrupts
the gradient-targeted geometry) and parametric ver-
ification (Layer 3 catches payload contradictions
regardless of trigger surface form).
Full system trades black-box for white-box.
Full L1+L2+L3 matches L1+L3’s 27.8% white-

Config BB ASR↓WB ASR↓Latency
(ms/q)
No Defense 67.0%∼74%∗0
L1 only 33.0% 57.0% 13
L2 only 4.0%∼71%∗8
L3 only 68.0%∼71%∗2,948
L1+L23.0%53.0%12
L1+L3 11.0%27.8%2,286
L2+L3 3.0% 68.0% 154
Full (L1+L2+L3) 9.0% 27.8% 15,202
Table 4:Layer ablation on NQ ( k= 50 ).BB = black-
box LM-targeted; WB = white-box HotFlip.∗WB rows
marked ∼are computed over valid responses only (see
Section 5). L2 alone dominates black-box; only combi-
nations including L1 and L3 meaningfully reduce white-
box ASR. Full system trades black-box ASR for white-
box robustness due to over-filtering by Layer 3.
box ASR but raises black-box ASR ( 9.0% vs.3.0%
for L1+L2) — an over-filtering effect, as Layer 3
occasionally discards clean passages whose phras-
ing differs from the LLM’s prior. In practice L1+L2
is preferable for black-box-dominant threat mod-
els, with L3 enabled adaptively when white-box
adversaries are expected (Section 7).
HotpotQA supplementary ablation.On multi-
hop HotpotQA, L2+L3 attains the lowest black-
box ASR ( 11.0% ), just below L1+L2 ( 12.0% ),
while the full system rises to 31.0% — again over-
filtering when L3 is applied across multi-hop ev-
idence. We treat this as a tunable deployment
choice, not a fundamental limitation.
Layer orthogonality.Three observations from
Table 4 support orthogonal rather than redundant
coverage. L1 alone leaves 33% black-box ASR and
L2 alone 4%, yet L1+L2 reaches 3%— L1 catches
paraphrasing poisons L2 misses. In the white-box
regime L1 alone leaves 57.0% and L3 alone leaves
∼71% , yet L1+L3 reaches 27.8% , so their catches
barely overlap. And Full ( 9%) isworsethan L1+L2
(3%): the layers are not strictly additive, because
L3’s false-positive rate on clean documents matters.
They therefore cover distinct failure modes while
interacting in false-positive profiles — supporting
the recommended configuration over a naive “run
all three.”
6.5 Adaptive Adversary Evaluation
Section 3.2 defines four levels of attacker adap-
tivity. This section empirically evaluates Level 1
(architecture-aware); Levels 2 (judge-aware) and
3 (full white-box) remain conceptual only, and im-plementing them is planned future work.
6.5.1 Paraphrasing evades Layer 2 without
sacrificing retrieval rank
Letting the live Contriever retriever rank poisons
(k= 50 ,n= 100 ), paraphrasing the trigger costs
the attacker almost nothing at retrieval: 4.97 of
5poisons still reach the top- 50on NQ (versus
5.00 verbatim) and 5.00 of5on HotpotQA, and
undefended ASR does not consistently drop (NQ
55.0%→45.0% ; HotpotQA 70.0%→77.0% ).
Paraphrasing is thus a free evasion of Layer 2’s
lexical check, shifting the burden to Layers 1
and 3. Layer 1 alone recovers much of the loss
(45.0%→32.0% on NQ, 77.0%→44.0% on Hot-
potQA), matching the Layer-3-off rows of Table 5.
That 32.0% is within noise of Layer 1’s 33.0%
verbatim-trigger ASR in Table 4, so its geometric
check is largely insensitive to surface paraphras-
ing, as its design predicts. A verbatim-trigger sieve
reproduces Table 1’s defended NQ ASR exactly
(3.0% ), though the undefended baseline in this live-
retrieval setup (NQ 55.0% , HotpotQA 70.0% ) dif-
fers somewhat from Table 1’s static baseline (NQ
67.0% , HotpotQA 87.0% ), likely reflecting live vs.
cached retrieval.
6.5.2 Enabling Layer 3 recovers the loss on all
three axes at once
Holding Layer 1 and Layer 2 fixed across paired
runs (Layer 1 removes 17.05 (NQ) / 14.47 (Hot-
potQA) documents per query on average; Layer
2 removes 0.01 /0.15), we toggle Layer 3 alone
against paraphrased-trigger poisons (Table 5).
This is not an ASR-for-utility trade: on NQ,
ASR roughly halves while clean accuracy rises
18points. The honest cost is latency — Layer 3’s
per-document verdicts add ∼16 –19s/query under
live retrieval against a ∼0.35 s L1+L2 baseline —
though adaptive invocation, firing only when Lay-
ers 1 and 2 disagree, keeps this occasional ( 21% of
HotpotQA queries). We therefore position Layer 3
as an optional high-assurance layer rather than part
of the black-box-dominant default (Section 4).
6.5.3 Forced-top worst case
As a harsher upper bound we pin poisons to the
top of context, removing retrieval competition, and
ablate Level 1 adaptivity intopayload-diversified,
trigger-paraphrased, andfull adaptivevariants
against the static baseline ( n= 30 , NQ; Table 7,
Appendix B.2). Layer 2 neutralizes the static and

NQ HotpotQA
Layer 3ASR↓CleanAcc↑Surv./5↓ASR↓CleanAcc↑Surv./5↓
Off 32.0% 43.0% 2.51 44.0% 34.0% 2.55
On15.0% 61.0% 0.79 31.0% 45.0% 1.22
Table 5:Layer 3 under live retrieval.Paraphrased-trigger poisons, Contriever, k= 50 ,n= 100 ; Layers 1–2 held
fixed across the paired runs, so differences are attributable to Layer 3 alone. “Surv./5” is the mean number of the 5
injected poisons still in context after filtering. Enabling Layer 3 improves all three columns simultaneously on both
datasets.
payload-diversified attacks ( 10.0% ) at near-zero
cost, since neither alters the prefix it inspects; para-
phrasing the trigger evades it ( 57.0% under L1+L2)
but forces a claim contradicting the model’s prior,
which Layer 3 catches (40.0%and33.0%).
6.6 Injection Ratio and Hyperparameter
Sweeps
Varying poisons per query from 1to5on NQ (Ta-
ble 8, Appendix B.4), baseline ASR climbs from
0%to66% while the Sieve holds below 10%; ex-
tending to 6–20poisons ( n= 25 ; Table 9) leaves
TRIS ASR flat at 4.0% . Because Layers 1 and 2
filter independently rather than by majority vote,
denser poisoning cannot tip the defense the way it
could tip a voting aggregator. Full-scale replication
remains future work (Section 8).
Sweeping retrieval depth k∈ {5,10,20,50}
against cluster count C∈ {2,3,5} (Table 10, Ap-
pendix B.5), the defense is unstable at k= 5
(ASR 85–93%, clean accuracy 9–17%) because
K-Means partitions on too few candidates are unre-
liable. From k= 10 onward ASR stabilizes at 3–
4%and clean accuracy rises monotonically to 78%
atk= 50 ;C= 3 is our default, and C= 5 per-
forms comparably on this sweep — we fix C= 3
as the more conservative setting.
6.7 Latency
Layer 2 adds ∼8ms/query and Layer 1 ∼13 ms;
Layer 3 dominates at ∼2.9 s/query, making always-
on (∼15 s) acceptable for high-risk queries but
not high-throughput search — under live retrieval
its added cost is similar ( ∼16 –19s/query; Sec-
tion 6.5). Appendix C traces one successful de-
fense and one failure, showing where the residual
HotpotQA ASR originates.
7 Discussion
Adaptive attacks on the judge.Section 6.5
closes the Level 1 gap empirically; judge-
awareness and white-box access remain conceptual.Against a judge-aware attacker (Level 2) Layer 1
becomes a single point of failure; ensembling di-
verse judges and rotating Jare the natural coun-
termeasures, and since Layer 1’s embedding-space
independence alone does not measurably differen-
tiate attack outcomes (Appendix B.3), ensembling
multiple judges is the more promising direction.
A Level 3 adversary also knows Layer 2’s thresh-
olds and could search for payloads at the decision
boundary of yint; randomizing that threshold per
query and ensembling verifiers are analogous re-
sponses, both future work.
Zero-Trust Retrieval.TRIS treats retrieved evi-
dence as untrusted input requiring active validation,
analogous to a network firewall — which is also
why it sits in middleware rather than retraining the
retriever: adversarial training is expensive, must be
redone as attacks emerge, and is incompatible with
third-party embedding APIs, whereas middleware
is corpus-, model-, and vendor-agnostic. Comple-
mentary modules could address prompt injection
(Greshake et al., 2023), privacy leakage, and con-
tent moderation under the same principle, keep-
ing the utility–robustness trade-off explicit through
adaptive invocation.
8 Conclusion
We presented TRIS, a three-layer middleware de-
fense against retrieval-stage poisoning, each layer
attacking a distinct constraint a poisoned document
must satisfy. It reduces black-box ASR by an or-
der of magnitude on NQ and MS-MARCO, mit-
igates white-box HotFlip from ∼74% to27.8%
with Layer 3 enabled, and drives poisoned MRR
to zero, while recovering 41–45points of clean
accuracy over the attacked baseline. Against fairly-
tuned TrustRAG and RobustRAG it is competitive
rather than dominant (Section 6.2); what distin-
guishes it is a strong worst-case profile, far lower
verification cost in the common attack regime, and
graceful degradation as attackers adapt.

Limitations
No empirical evaluation of judge-aware or
white-box adversaries.A remaining limitation
of this work is that our headline numbers (Section
6) are against static PoisonedRAG attacks (Level
0 in Section 3.2). Section 6.5 now provides an em-
pirical evaluation of Level 1 (architecture-aware)
adversaries — paraphrased triggers and diversi-
fied payloads, under live retrieval and a forced-top
worst case — showing that Layer 3 recovers most
of the robustness that paraphrasing costs Layer 2.
Levels 2 (judge-aware) and 3 (full white-box) re-
main conceptual only (Section 3.2); implementing
them is planned future work. An adversary who
optimizes triggers to land in the majority cluster
of the judge model, avoid prefix overlap with the
query, and produce payloads compatible with the
LLM’s parametric prior would represent the worst
case for TRIS. Whether such an attacker can si-
multaneously satisfy all three constraints — and
how much trigger-and-payload search space they
would need to explore — remains an open empir-
ical question for Levels 2–3 that follow-up work
should address with adaptive-attack benchmarks of
the kind suggested by Zhang et al. (2025).
Generator dependency and Layer 3 coupling.
Layer 3’s contradiction detection is only as reli-
able as the verifier’s own parametric knowledge of
ytrue— a knowledge-dependency that is a struc-
tural property of the design, not an artifact of any
one model. Our headline results use GPT-3.5-turbo
as both the generator and (in Layer 3 always-on
mode) the verifier. This couples the defense’s mea-
sured efficacy to the parametric knowledge of a
specific commercial model. We tested this fail-
ure mode directly on a slice of 10questions about
clearly post-cutoff events the generator does not
know: Layer 3 abstained on all 10(fail-open), no
clean document was wrongly dropped ( 0of50),
and Layers 1–2 still removed every injected poison
(30of30). Layer 3 therefore degrades to a no-op
rather than to a source of false positives when its
parametric prior is absent, and the structural lay-
ers carry the defense unaided — consistent with
the L1+L2 row of Table 4, which reaches 3.0%
black-box ASR with Layer 3 disabled entirely.
Scale and corpus diversity.We evaluate on 100-
query subsets of NQ, HotpotQA, and MS-MARCO.
Recall@5 on the clean baseline is 0.180 , reflecting
the subset rather than full-corpus retrieval. Scalingto full corpora (BEIR, full MS-MARCO, enter-
prise document stores) is essential for character-
izing false-positive rates in production and is the
most pressing follow-up.
Statistical reporting.Tables report point esti-
mates over 10iterations ×10 queries. Differences
of≤3percentage points (e.g., Sieve vs. TrustRAG
on HotpotQA) may be within iteration-to-iteration
noise; we frame those comparisons as “comparable”
rather than wins.
Cost of Layer 3.The LLM consistency layer
adds∼2.9 s/query and is the dominant latency cost.
Distilling LLM-as-judge behavior into a smaller
specialized verifier is the obvious next step; the
adaptive-invocation mode is a stopgap.
Injection-ratio coverage.Our injection sweep
covers 1–5adversarial documents per query at full
scale ( n= 100 ), where the baseline ASR climbs
from 0%to66% and the Sieve holds below 10%.
The Sieve’s ASR is non-monotonic in this range
(8%→10%→6%→8% ), which we attribute
to iteration noise at small sample sizes. We extend
this range to 6–20adversarial documents per query
on a reduced-scale NQ subsample ( n= 25 ; Ta-
ble 9): TRIS ASR stays flat at 4.0% and CleanAcc
remains near 46%, supporting the“robustness to
denser poisoning” hypothesis. Full-scale replica-
tion of this extended range, and coverage of Hot-
potQA and MS-MARCO, remains future work.
Reduced-scale baseline comparisons.The fair-
baseline re-evaluations that our comparative claims
now rest on — the RobustRAG threshold sweep
(Table 2) and the TrustRAG comparison with its
LLM check enabled (Table 3) — were run at n=
30, against n= 100 for Table 1, and as single
runs rather than 10-iteration aggregates. They are
adequate to locate each baseline’s operating-point
frontier, which is what they are used for, but not
to resolve differences of a few points; the n= 30
RobustRAG majority row also does not reproduce
the full-scale majority row in Table 1, and we have
not been able to reconcile the two. Re-running both
baselines at full scale is the first item of follow-up
work. We note the direction of the resulting bias
honestly: these are the comparisons in which the
baselines lookstrongestrelative to TRIS, so under-
powering them is not a limitation that flatters us.
No formal guarantees.TRIS is heuristic. It does
not provide certified robustness in the sense of Xi-

ang et al. (2024) and we make no formal claim
about adversarial bounds. Combining TRIS’s em-
pirical strength with certified isolate-and-aggregate
decoding is a promising direction — e.g., applying
TRIS as a pre-filter and RobustRAG as the certified
aggregator.
Partial-iteration results.The white-box L1+L3
and full-system numbers in Table 4 are averaged
over9of10iterations, because one backup job did
not complete (unrelated to the transient API failures
in Section 5) and we did not have the compute bud-
get to re-run it. Judging by the iteration-to-iteration
spread elsewhere in our runs, we expect these two
numbers to move by at most 1–2percentage points;
we report them as partial rather than complete, and
no claim in this paper turns on a margin that small.
Ethical Considerations
This work studies a defense against an existing,
publicly documented attack class (PoisonedRAG
and its follow-ups). We do not introduce new attack
capabilities. The defense is designed to be deploy-
able as a middleware module by RAG operators
without requiring retriever or generator retraining,
lowering the barrier to robust deployment. Our
experiments use publicly available datasets (NQ,
HotpotQA, MS-MARCO) and do not involve hu-
man subjects or personally identifiable information.
We acknowledge a dual-use consideration: detailed
descriptions of attacker design (Section 3) inform
defenders but could in principle aid attackers; how-
ever, all attack details follow Zou et al. (2025) and
are already public. We believe the net effect of
clearer, comparable defense evaluations is positive
for the security of deployed RAG systems.
Acknowledgments
This work is based upon the work supported by
the National Center for Transportation Cyberse-
curity and Resiliency (TraCR) (a U.S. Depart-
ment of Transportation National University Trans-
portation Center) headquartered at Clemson Uni-
versity, Clemson, South Carolina, USA. It was
also supported in part by NIST grant number
60NANB24D143. Any opinions, findings, con-
clusions, and recommendations expressed in this
material are those of the author(s) and do not nec-
essarily reflect the views of NIST, TraCR, and the
U.S. Government assumes no liability for the con-
tents or use thereof.This work used the Delta system at the National
Center for Supercomputing Applications [award
OAC 2005572] through allocation CIS251331 from
the Advanced Cyberinfrastructure Coordination
Ecosystem: Services & Support (ACCESS) pro-
gram, which is supported by National Science
Foundation grants #2138259, #2138286, #2138307,
#2137603, and #2138296. The authors also ac-
knowledge High Performance Computing at The
University of Texas at Dallas (HPC@UTD) for
providing computing resources.
Disclaimer: This paper identifies certain equip-
ment, instruments, software, or materials to ade-
quately describe the experimental procedure. Such
identification is not intended to imply recommen-
dation or endorsement of any product or service by
NIST, nor is it intended to imply that the materi-
als or equipment identified are necessarily the best
available for the purpose.
References
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder,
Andrew McNamara, Bhaskar Mitra, Tri Nguyen,
Mir Rosenberg, Xia Song, Alina Stoica, Saurabh
Tiwary, and Tong Wang. 2018. Ms marco: A human
generated machine reading comprehension dataset.
Preprint, arXiv:1611.09268.
Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song,
and Bo Li. 2024. AgentPoison: Red-teaming LLM
agents via poisoning memory or knowledge bases.
Advances in Neural Information Processing Systems,
37:130185–130213.
Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu,
Wei Du, Ping Yi, Zhuosheng Zhang, and Gongshen
Liu. 2024. TrojanRAG: Retrieval-augmented genera-
tion can be backdoor driver in large language models.
arXiv preprint arXiv:2405.13401.
Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing
Dou. 2018. HotFlip: White-box adversarial exam-
ples for text classification. InProceedings of the 56th
Annual Meeting of the Association for Computational
Linguistics (Volume 2: Short Papers), pages 31–36,
Melbourne, Australia. Association for Computational
Linguistics.
Kennedy Edemacu, Vinay M. Shashidhar, Micheal Tu-
ape, Dan Abudu, Beakcheol Jang, and Jong Wook
Kim. 2025. Defending against knowledge poisoning
attacks during retrieval-augmented generation.arXiv
preprint arXiv:2508.02835.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,

and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997.
Jonas Geiping, Liam H Fowl, W. Ronny Huang, Woj-
ciech Czaja, Gavin Taylor, Michael Moeller, and Tom
Goldstein. 2021. Witches’ brew: Industrial scale data
poisoning via gradient matching. InInternational
Conference on Learning Representations.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra,
Christoph Endres, Thorsten Holz, and Mario Fritz.
2023. Not what you’ve signed up for: Compromis-
ing real-world LLM-integrated applications with in-
direct prompt injection. InProceedings of the 16th
ACM workshop on artificial intelligence and security,
pages 79–90.
Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Sid-
dharth Garg. 2019. BadNets: Evaluating backdoor-
ing attacks on deep neural networks.IEEE Access,
7:47230–47244.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2024. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense informa-
tion retrieval with contrastive learning.Transactions
on Machine Learning Research (TMLR).
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021.
Billion-scale similarity search with GPUs. InIEEE
Transactions on Big Data.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP).
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research.Transactions of the
Association for Computational Linguistics, 7:453–
466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS).
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.2021. Entity-based knowledge conflicts in question
answering. InProceedings of the 2021 Conference
on Empirical Methods in Natural Language Process-
ing, pages 7052–7063, Online and Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.
OpenAI. 2023. GPT-3.5 Turbo Fine-Tuning and API
Updates. OpenAI Blog. Accessed: 2026-08-31.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. InProceedings of the 2019 conference on
empirical methods in natural language processing
and the 9th international joint conference on natural
language processing (EMNLP-IJCNLP), pages 3982–
3992.
Avital Shafran, Roei Schuster, and Vitaly Shmatikov.
2025. Machine against the RAG: Jamming retrieval-
augmented generation with blocker documents. In
USENIX Security Symposium.
Yixiao Song, Yekyung Kim, and Mohit Iyyer. 2024.
VeriScore: Evaluating the factuality of verifiable
claims in long-form text generation. InFindings
of the Association for Computational Linguistics
(EMNLP).
Jacob Steinhardt, Pang Wei Koh, and Percy Liang. 2017.
Certified defenses for data poisoning attacks. InAd-
vances in Neural Information Processing Systems
(NeurIPS).
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gard-
ner, and Sameer Singh. 2019. Universal adversarial
triggers for attacking and analyzing NLP. InProceed-
ings of the 2019 Conference on Empirical Methods
in Natural Language Processing (EMNLP).
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably ro-
bust RAG against retrieval corruption.arXiv preprint
arXiv:2405.15556.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing
(EMNLP).
Baolei Zhang, Haoran Xin, Jiatong Li, Dongzhe Zhang,
Minghong Fang, Zhuqing Liu, Lihai Nie, and Zheli
Liu. 2025. Benchmarking poisoning attacks against
retrieval-augmented generation.arXiv preprint
arXiv:2505.18543.

Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025. Trustrag: Enhancing robust-
ness and trustworthiness in retrieval-augmented gen-
eration.Preprint, arXiv:2501.00879.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2025. PoisonedRAG: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. InUSENIX Security Symposium.
A Threat Model Illustration
Section 3 describes the PoisonedRAG threat model
in prose: an adversary injects a small number of
adversarially crafted documents into the retrieval
corpus, a dense retriever ranks them highly for the
target query because of trigger optimization, and
the generator conditions on this poisoned context
and outputs the attacker’s chosen answer instead of
the true one. Figure 2 gives the visual walkthrough
of that pipeline for a single example query, from
injection through retrieval to the corrupted final an-
swer; it is reproduced here rather than in Section 3
to keep the main body within the page limit.
B Extended Results
This appendix collects the full tables behind re-
sults that Section 6 reports and interprets in prose:
retrieval-level diagnostics, the adaptive-adversary
breakdown, the extended injection-ratio sweep, and
the retrieval-depth/cluster-count sensitivity sweep.
None of the numbers below are new — each table
is cited from the point in the main text where its
findings are first discussed, and is reproduced here,rather than inline, to keep the main body within
the page limit. Each subsection also adds a short
pointer back to the relevant main-text discussion.
B.1 Retrieval Dynamics
Table 6 is the retriever-level counterpart to Table 1:
instead of end-to-end accuracy and attack success,
it reports Recall@5, Recall@50, and MRR directly
on NQ, isolating how the Sieve changes the rank-
ing of retrieved documents rather than only the
generator’s final answer. Discussed in Section 6.3.
System State R@5↑R@50↑CleanMRR↑PoisMRR↓
Clean Corpus 0.180 0.680 0.131 —
Poisoned (No Defense) 0.000 0.680 0.048 1.000
+ TrustRAG 0.180 0.680 0.131 0.000
+ Tri-Layer Sieve0.180 0.680 0.131 0.000
Table 6:Retrieval dynamics on NQ, top-50.The Sieve
fully inverts the MRR corruption induced by poisoning:
PoisonedMRR drops from 1.000 to0.000 while Re-
call@5 and CleanMRR return to clean-corpus baseline.
Recall@50 is unaffected, confirming that the defense
reorders candidates rather than discarding the candidate
pool.
B.2 Forced-Top Worst Case
Table 7 reports the forced-top ablation discussed in
Section 6.5, in which poisons are pinned to the top
of context to remove retrieval competition and iso-
late the defense’s behavior under the harshest plau-
sible placement, separating the payload-diversified
and trigger-paraphrased components of Level 1
adaptivity before combining them.
Figure 2:The PoisonedRAG threat model.The attacker injects a small number of adversarially crafted documents
into a large retrieval corpus. Dense retrievers (Karpukhin et al., 2020; Izacard et al., 2022) rank these poisoned
documents highly for the target query due to trigger optimization. The LLM then conditions on malicious context
and outputs the attacker-specified answer (Zou et al., 2025).

Attack Undef. L1+L2 Full
Black-box (static) 70.0% 10.0% 10.0%
Payload-div. 67.0% 10.0% 10.0%
Trigger-para. 67.0% 57.0% 40.0%
Full adaptive 70.0% 50.0% 33.0%
Table 7:Forced-top worst case.Poisons pinned to
the top of context ( n= 30 , NQ, ASR ↓) — an upper
bound harsher than the live-retrieval Table 1 (TRIS NQ
ASR 3.0% ). Layer 2 alone stops static and payload-
diversified attacks; Layer 3 is needed once the trigger is
paraphrased.
B.3 Does Layer 1’s Independent Geometry
Matter?
Section 4 frames Layer 1 as majority-keep in an ar-
chitecturally independent geometry. This appendix
reports the ablation that tests whether that indepen-
dence is itself load-bearing, summarized in Sec-
tion 6.2. We swap only Layer 1’s embedding space
— Sentence-BERT MiniLM (architecturally inde-
pendent of the retriever) versus Contriever (the re-
triever’s own space) — holding every other com-
ponent fixed at k= 50 ,n= 100 on NQ and
HotpotQA.
Under paraphrased-trigger attacks the two
spaces are statistically indistinguishable: NQ ASR
32.0% (MiniLM) versus 33.0% (Contriever), p=
0.88; HotpotQA 44.0% versus 53.0% ,p= 0.20 ;
clean-accuracy differences are likewise not signif-
icant. The component that does differentiate is
Layer 2: on the verbatim-trigger attack, adding it
drives ASR to 3.0% (NQ) and 8.0% (HotpotQA)
against 33.0% and44.0% for clustering alone
(p <10−7).
We report this as a negative result about our own
design rationale rather than omitting it. Layer 1’s
independent geometry remains a robustness prop-
erty — it removes the requirement that poisons
form a tight, separable cluster in the retriever’s
own space, an assumption that a judge-aware ad-
versary would target directly — but it is not where
the measured gain comes from, and we scope our
novelty claim to Layer 2 and the orthogonal compo-
sition accordingly. Judge ensembling and rotation
(Section 7) remain plausible routes to making that
independence pay off empirically; we have not
tested them.
B.4 Injection Ratio Sweeps
Tables 8 and 9 report the full injection-ratio sweep
discussed in Section 6.6. Table 8 covers 1–5adver-sarial documents per query at full scale ( n= 100 );
Table 9 extends this to 6–20documents per query
on a reduced-scale sample ( n= 25 ) to check
whether the Sieve’s robustness holds at higher poi-
soning density.
Adv. docs/query Baseline ASR Sieve ASR
1 0.0% 0.0%
2 47.0% 8.0%
3 54.0% 10.0%
4 64.0% 6.0%
5 66.0% 8.0%
Table 8:Injection-ratio sweep on NQ ( k= 50 ).The
Sieve holds ASR below 10% as the baseline climbs from
0%to66%.
Poisons/query 6 10 15 20
Baseline ASR↓52.0% 48.0% 52.0% 56.0%
Sieve ASR↓4.0% 4.0% 4.0% 4.0%
Sieve CleanAcc↑48.0% 44.0% 48.0% 48.0%
Table 9:Injection sweep, extended range. 6–20ad-
versarial documents/query, NQ, n= 25 (cf. Table 8 for
1–5at full scale, n= 100 ). ASR stays flat regardless of
poison density.
B.5 Sensitivity to Retrieval Depth and Cluster
Count
Table 10 reports the full retrieval-depth ( k) by
cluster-count ( C) sweep summarized in Sec-
tion 6.6.
k\C C=2C=3C=5
k=5— 93% / 9% 85% / 17%
k=104% / 69% 4% / 61% 3% / 69%
k=204% / 71% 3% / 71% 3% / 72%
k=504% / 75%3% / 77%3% / 78%
Table 10:Sensitivity to kandCon NQ.Each cell is
ASR / CleanAcc. The k=5, C=2 cell failed due to an
API error and is omitted. k≥10 yields stable ASR–
utility; C= 3 is our default, with C= 5 performing
comparably on this sweep.
C Qualitative Case Studies
Section 6.7 reports aggregate latency, and Ta-
ble 1 reports aggregate attack-success and clean-
accuracy rates. Aggregate rates do not showwhy
the defense succeeds on most queries orhowit
fails on the rest, so this appendix walks through
one query of each kind in detail: one where the
Sieve correctly removes the poison, and one where

a poison survives all three layers and the attack
succeeds.
Success (black-box).For an NQ query about
a technology executive, five poisons each begin
with the verbatim query followed by an authorita-
tive paragraph naming a plausible alternative entity.
The poisons rank in the top- 5under Contriever; in
judge space they form a tight off-cluster (Layer 1
flags four of five) and Layer 2 independently flags
all five via >0.9 Jaccard overlap. The generator
emits the correct answer.
Failure (mimicry).For a HotpotQA two-hop
query, a poison mimics the genuine Wikipedia pas-
sage’s style, differing only in the answer entity. It
does not repeat the query (evading Layer 2) and
clusters with benign passages in judge space (evad-
ing Layer 1); Layer 3’s signal is weak because
the alternative entity is itself plausible. This pat-
tern accounts for most of the residual 14% Hot-
potQA ASR reported in Table 1: multi-hop ampli-
fies mimicry failures because a poison need only
corrupt one hop.
Residual risk is concentrated in payload-level
mimicry rather than trigger evasion — the failure
profile a stronger Layer 3 would address.
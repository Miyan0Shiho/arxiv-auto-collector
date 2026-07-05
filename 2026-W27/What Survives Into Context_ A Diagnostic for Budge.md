# What Survives Into Context: A Diagnostic for Budget-Constrained Multi-Hop RAG and When Submodular Evidence Packing Improves It

**Authors**: Ananto Nayan Bala

**Published**: 2026-07-01 10:12:15

**PDF URL**: [https://arxiv.org/pdf/2607.00725v1](https://arxiv.org/pdf/2607.00725v1)

## Abstract
Retrieval-augmented generation (RAG) under a fixed reader-context budget forces a selection problem: of the evidence retrieved, only a fraction can be shown to the reader. We argue that document recall -- the standard retrieval metric -- is the wrong quantity to optimize in this regime, and we make two contributions. First, as a general contribution, we introduce answer-in-context, a diagnostic that measures whether a gold answer survives as a contiguous span in the packed reader context (not the retrieved set). It predicts answer F1 better than recall (r=0.39-0.55 vs. about 0.31), separates answer quality roughly five-fold (0.60 vs. 0.12 on HotpotQA), and carries information beyond retrieval: it adds Delta R squared=0.17 over recall and shows a 4.6x EM gap even among questions where all gold was retrieved. We also confirm it interventionally: on 2WikiMultiHopQA a packing change that raises coverage but not answer-in-context yields no accuracy gain. Second, as a conditional contribution, we cast reader-context construction as budgeted monotone submodular maximization and build a packer that jointly optimizes relevance, query coverage, representativeness, and diversity. On HotpotQA with a 160-token budget and a 3B reader it beats a strong focused heuristic, MMR, and naive packing -- by up to +5.1 F1 at equal-or-lower token cost, across three seeds. Crucially, we map the scope of this win honestly: it requires the conjunction of (i) multi-hop complementary structure, (ii) retrieval that surfaces the evidence, (iii) a binding but not extreme budget, and (iv) a reader weak enough that evidence density, not reading capacity, is the bottleneck. A quantization-controlled reader-scale ladder (3B to 7B to 14B) shows the edge over the heuristic is absorbed by 7B and significantly reverses by 14B, while the diagnostic explains every boundary with a single variable.

## Full Text


<!-- PDF content starts -->

What Survives Into Context: A Diagnostic for Budget-Constrained
Multi-Hop RAG and When Submodular Evidence Packing Improves It
Ananto Nayan Bala
Ahsanullah University of Science and Technology
nayan.ananto@gmail.com
Abstract
Retrieval-augmented generation (RAG) un-
der a fixed reader-context budget forces
a selection problem: of the evidence re-
trieved, only a fraction can be shown to
the reader. We argue that document recall—
the standard retrieval metric—is the wrong
quantity to optimize in this regime, and we
make two contributions. First, as agen-
eralcontribution, we introduceanswer-
in-context, a diagnostic that measures
whether a gold answer survives as a con-
tiguous span in thepackedreader con-
text (not the retrieved set). It predicts an-
swer F1 better than recall ( r=0.39 –0.55 vs.
∼0.31 ), separates answer quality roughly
five-fold ( 0.60 vs.0.12 on HotpotQA), and
carries informationbeyondretrieval: it
adds ∆R2=0.17 over recall and shows a
4.6× EM gap even among questions where
all gold was retrieved. We also confirm
itinterventionally: on 2WikiMultiHopQA
a packing change that raises coverage but
not answer-in-context yields no accuracy
gain. Second, as aconditionalcontribu-
tion, we cast reader-context construction
as budgeted monotone submodular max-
imization and build a packer that jointly
optimizes relevance, query coverage, rep-
resentativeness, and diversity. On Hot-
potQA with a 160-token budget and a 3B
reader it beats a strong focused heuristic,
MMR, and naive packing—by up to +5.1
F1 at equal-or-lower token cost, across
three seeds. Crucially, we map the scope of
this win honestly: it requires the conjunc-
tion of (i) multi-hop complementary struc-
ture, (ii) retrieval that surfaces the evidence,
(iii) a binding but not extreme budget, and
(iv) a reader weak enough that evidencedensity, not reading capacity, is the bot-
tleneck. A quantization-controlled reader-
scale ladder (3B →7B→14B) shows the
edge over the heuristic is absorbed by 7B
and significantlyreversesby 14B, while the
diagnostic explains every boundary with a
single variable.
1 Introduction
A retrieval-augmented reader has a finite context
window, and in practice an even smallerevidence
budget: the share of that window allocated to re-
trieved passages. Once retrieval returns more rele-
vant text than fits, the system must decide what to
keep. This selection step is usually treated as an
afterthought—concatenate the top- k, truncate to fit
(Lewis et al., 2020; Ram et al., 2023)—yet under a
tight budget it is the step that decides whether the
reader ever sees the answer.
The community’s default retrieval metric,
recall@ k, is computed on theretrieved document
set. But the reader never consumes the retrieved
set; it consumes thepacked context. When packing
discards evidence to fit a budget, recall and what-
the-reader-sees diverge. The divergence is acute
formulti-hopquestions (Yang et al., 2018; Trivedi
et al., 2022), where the answer depends on combin-
ing evidence from several documents: retrieving
all of them is necessary but not sufficient, because
the packer may keep a redundant pair and drop the
bridge. Figure 1 makes the gap concrete.
This paper starts from a measurement gap and
ends with a method. We first ask:what property of
the reader context actually predicts answer quality
under a budget?We defineanswer-in-context—
does a gold answer appear verbatim in the packed
context—and show it predicts answer F1 far bet-
ter than retrieval recall on every dataset we test
(§3). This reframes the budgeted-RAG objective
from “retrieve the gold documents” to “pack soarXiv:2607.00725v1  [cs.CL]  1 Jul 2026

the answer survives.” We then ask:can a princi-
pled packer move that quantity?We formulate
reader-context construction asbudgeted mono-
tone submodular maximization(§4) and show on
HotpotQA it delivers a statistically clean win over
heuristic packing, MMR, and naive concatenation
across three seeds (§5). A per-question decomposi-
tion ties the win to the diagnostic: the packer helps
precisely by assembling complementary multi-hop
evidence into the reader context.
Finally—and we view this as much a contribu-
tion as the method—wescope the win honestly
(§6). Through controlled experiments on RAG-
Bench, MuSiQue, a budget sweep, and a reader-
scale ladder, we identify four conditions that must
co-occur for principled packing to beat the best
heuristic, and we show concrete settings where
each fails. On MuSiQue we try the obvious fix for
the failing condition (more retrieval) and it changes
nothing, turning a soft “does not transfer” into a
precise boundary; and a quantization-controlled
reader-scale ladder answers the “a stronger reader
just absorbs your packing” objection with data—
the edge over the heuristic is absorbed by 7B and
significantly reverses by 14B, while the packer’s
mechanism and its win over naive packing persist.
The diagnostic predicts every one of these patterns.
Contributions.
1.A diagnostic (general).Answer-in-context,
a reader-context-level metric that predicts
budgeted-RAG quality better than recall on
span-answer datasets, with demonstratedin-
cremental validityover recall ( ∆R2=+0.17 ;
a4.6× EM separation that survives even when
all gold is retrieved) andinterventionalsupport
on 2Wiki.
2.A method (conditional).A budgeted submod-
ular evidence packer that significantly improves
HotpotQA answer quality over heuristic, MMR,
and naive packers at equal-or-lower token cost,
with a mechanistic per-question explanation.
3.A scope map (the honest core).A four-
condition account of when principled packing
beats the best heuristic, each condition demon-
strated to fail in a controlled setting—including
a retrieval-unlock ablation on MuSiQue and
a quantization-controlled reader-scale ladder
(3B→7B→14B) that locates the reader scale at
which curation stops paying off and begins to
cost.
We deliberately donotclaim that graph-query Retrieverretrieved set (recall@khere)
Packer
(≤Btokens)
context (answer-in-context here)gold #2 dropped
Reader answer
Figure 1: Recall is scored on theretrieved set; the reader
consumes thepacked context. Under a budget the packer
can drop a retrieved gold document (here “gold #2”), so
high recall need not mean the answer survives. Answer-
in-context measures exactly what reaches the reader.
structured evidence or submodular packing univer-
sally improves RAG. The evidence supports a nar-
row, mechanistically explained claim plus a diag-
nostic that generalizes—which we believe is more
useful than a broad claim that does not survive
replication.
2 Related Work
Retrieval-augmented generation.RAG couples
a (typically dense; Karpukhin et al., 2020) retriever
with a reader LM (Lewis et al., 2020; Guu et al.,
2020; Izacard and Grave, 2021; Izacard et al., 2023)
and now spans retrieval from trillions of tokens
(Borgeaud et al., 2022), in-context retrieval (Ram
et al., 2023), black-box augmentation (Shi et al.,
2024), joint instruction tuning (Lin et al., 2024),
and self-reflective variants (Asai et al., 2024); see
Gao et al. (2023) for a survey. Most of this work
reports retrieval recall and end-task accuracysepa-
ratelyand treats context construction as fixed top- k
concatenation. Our diagnostic targets the quan-
tity in between—what the packed context actu-
ally contains—which becomes the binding variable
once a budget forces selection.
Multi-hop question answering.HotpotQA
(Yang et al., 2018), MuSiQue (Trivedi et al., 2022),
2WikiMultiHopQA (Ho et al., 2020), and WikiHop
(Welbl et al., 2018) require composing evidence
across documents. A large line of work attacks
the retrieval side of this difficulty with multi-hop
dense retrieval (Xiong et al., 2021), interleaved
retrieval-and-reasoning (Trivedi et al., 2023; Press
et al., 2023) built on chain-of-thought prompting
(Wei et al., 2022), iterative retrieval-generation
(Shao et al., 2023; Jiang et al., 2023b), and
program-style composition (Khattab et al., 2022).
We use these datasets not to improve retrieval but
tovarywhether the complementary evidence is

present and surfaced, which is what determines
whether a packer can help.
Context selection and compression.Reducing
reader context via reranking, selection, or compres-
sion is well studied. The canonical redundancy-
aware reranker is Maximal Marginal Relevance
(MMR) (Carbonell and Goldstein, 1998), our di-
rect baseline. Recent methods compress or filter
retrieved context—RECOMP (Xu et al., 2024a),
LLMLingua (Jiang et al., 2023a), Selective Con-
text (Li et al., 2023), context filtering (Wang et al.,
2023), and robustness to irrelevant passages (Yoran
et al., 2024). “Lost in the middle” effects (Liu et al.,
2024) and long-context studies (Bai et al., 2024;
Xu et al., 2024b) show that simply enlarging the
window is not a substitute for choosing what goes
in it. Our packer differs in that its objective is tied
to an explicit, measurable answer-density quantity
(the diagnostic), and our central message is a scope
map forwhenprincipled selection helps at all.
Submodular optimization for selection.
Coverage-and-diversity objectives with the
cost-scaled greedy algorithm and its constant-
factor guarantee (Nemhauser et al., 1978) were
introduced for extractive summarization by Lin
and Bilmes (2011, 2010); see Krause and Golovin
(2014); Bilmes (2022) for broader treatments.
We apply that machinery toreader-context
evidence packingfor RAG and tie the objective
to the answer-in-context quantity our diagnostic
measures.
Retrievers and readers.We use a bi-encoder re-
triever (Reimers and Gurevych, 2019; Xiao et al.,
2024) of the kind evaluated on MTEB (Muen-
nighoff et al., 2023) and BEIR (Thakur et al., 2021),
with classic sparse (Robertson and Zaragoza, 2009),
late-interaction (Khattab and Zaharia, 2020), and
cross-encoder (Nogueira and Cho, 2019) retrieval
as the surrounding context. Readers are instruction-
tuned LLMs (Qwen Team, 2025; Touvron et al.,
2023; Brown et al., 2020); the larger rungs of our
reader ladder use 4-bit NF4 quantization (Dettmers
et al., 2023, 2022) to fit commodity GPUs, which
is why we include a precision control.
RAG evaluation.EM/F1 (Rajpurkar et al., 2016)
measure answer quality, while RAG-specific frame-
works score faithfulness and context relevance (Es
et al., 2024; Saad-Falcon et al., 2024; Chen et al.,
2024) over knowledge- intensive suites (Petroni
et al., 2021; Mallen et al., 2023). These score theFeature corr. w/ F1 corr. w/ EM
answer-in-context +0.50 +0.46
gold-doc reader cov. +0.33 +0.29
retrieval all-gold@5 +0.32 +0.27
retrieval recall@5 +0.31 +0.27
gold-token density +0.26 +0.23
Table 1: Feature–quality correlations on HotpotQA
(seed 42, 500 questions, n=2,500 policy×question
rows, budget 160). Answer-in-context is the strongest
single predictor—above both retrieval metrics.
retrievedcontext or thefinalanswer; answer-in-
context instead measures the packed context the
reader sees, and we show it has incremental valid-
ity over recall for predicting end-task quality.
3 The Answer-in-Context Diagnostic
3.1 Definition
Given a question with gold answer set Aand a
materialized reader context C(the concatenation
of packed snippets actually shown to the reader),
we define:
•answer-in-context =1if some normalized a∈
Aoccurs as a contiguous token subsequence of
normalizedC, else0;
•gold-doc reader coverage: fraction of gold doc-
uments contributing ≥1snippet to C;all-gold-
in-reader: whetherallof them do;
•gold-token density: fraction of C’s tokens
drawn from gold documents.
These are computed on thepackedrun, not the
retrieved set—the key difference from recall@ k,
which is scored on retrieved document idsbefore
packing. Answer-in-context is a necessary condi-
tion for an extractive-style reader to be correct, and
we hypothesize it is the mediator explaining why
higher recall need not raise answer quality under a
budget.
3.2 Answer-in-context predicts quality; recall
does not
Table 1 pools all policy ×question rows on Hot-
potQA and correlates each diagnostic with answer
quality. Answer-in-context is the strongest single
predictor, above both retrieval metrics and reader-
level coverage. Conditioning directly: mean F1 is
0.596 when a gold answer is in the reader context
versus 0.123 when it is not (a +0.47 gap). This
resolves the “lower recall, better answers” para-
dox: under a budget, what matters is whether the

Figure 2: Among HotpotQA questions whereallgold
paragraphs were retrieved (recall@5 =1), whether pack-
ing keeps the answer in context is still decisive: F1
0.61 vs.0.20, EM 0.50 vs.0.11.27% of these retrieval-
perfect questions drop the answer during packing. Clus-
tered bootstrap on question id, three seeds.
answersurvives into context, not how many gold
documents were retrieved.
3.3 Incremental validity: not recall in disguise
A natural objection is that answer-in-context is
near-tautological with correctness, or a proxy for re-
call. Two analyses refute this. Both pool HotpotQA
per-question rows across three seeds {42,13,7}
(10,500 rows, 1,500 questions) with inference
cluster-robust on question id.
(a) Incremental validity over recall.A model
of F1 on recall@5 alone explains R2=0.086 ;
adding answer-in-context raises this to R2=0.257 ,
an increment of ∆R2=+0.17 . The standardized
coefficient on answer-in-context ( β=+0.21 ) is
roughly 4×that on recall ( β=+0.05 ), and the par-
tial correlation of answer-in-context with F1 con-
trolling for recall is +0.43 . Answer-in-context
and recall@5 themselves correlate only +0.41 —
far from the ≈1that “it is just recall” would re-
quire.
(b) It captures the packing step, orthogonal to
retrieval.Restrict to questions where retrieval al-
ready succeeded—all gold in the top-5 ( n=7,739 ).
Even here,27% still drop the answer during
packing(Figure 2). Within this retrieval-perfect
subset, whether packing keeps the answer is de-
cisive: F1 0.61 vs.0.20 and EM 0.50 vs.0.11
(a3.0× /4.6× gap, tight clustered-bootstrap CIs).
This is the cleanest evidence that answer-in-context
measures thepackingstep rather than restating re-
trieval or correctness. (A minority of the 27% are
answers that never appear verbatim even in goldDatasetr(answer-in-context, F1) Note
HotpotQA+0.50packer wins
MuSiQue+0.54packer null
2WikiMultiHopQA+0.55 packer null; interven-
tional
RAGBench CovidQA +0.39 single-pass
RAGBench ExpertQA degenerate long free-form answers
Table 2: Answer-in-context–F1 correlation across five
datasets. Strongest on the two datasets where the packer
shows no win—not an artifact of the method. Degener-
ate on ExpertQA (answers never appear verbatim).
passages—paraphrase, not packing failure—so this
modestly overstates packing’s share; the predictive-
validity conclusion is unaffected.)
3.4 Generalization and an interventional test
Table 2 shows the correlation is not specific to Hot-
potQA or to our packer; it is in fact strongest on
MuSiQue and 2Wiki, where the packer shows no
win. This is the key evidence that the diagnostic is
a dataset-independent mediator, not a side effect of
the method.
An interventional test on 2Wiki.§3.3 is obser-
vational; 2WikiMultiHopQA lets us test the diag-
nosticinterventionally. We ran the exact HotpotQA
factorial (3B reader, budget 160, seeds {42,13,7} ,
500 questions) on 2Wiki, whose retrieval clears
the surfacing bar (all-gold@5 =0.43 ). The sub-
modular packer assembles strictly more gold than
the focused heuristic—gold-doc coverage +0.054 ,
all three seeds—yet raises answer-in-context by
only−0.007 and F1 by −0.008 (paired bootstrap
p=0.44 , a clean null). Coverage moves; answer-
in-context does not; accuracy follows answer-in-
context, not coverage. The mechanism is that
on 2Wiki’s compositional questions the answer-
bearing document is usually the one the heuristic
already ranks first, so theextragold the packer
adds is bridging evidence that scaffolds reasoning
without containing the answer string. This is the in-
terventional counterpart to §3.3: move coverage but
not answer-in-context, and quality does not move.
(For long free-form answers such as ExpertQA the
verbatim-span diagnostic is degenerate; a seman-
tic/entailment variant would be needed, which we
leave to future work.)

4 Method: Budgeted Submodular
Evidence Packing
4.1 Objective
Given retrieved evidence for a query and a hard
reader-token budget B, we build a candidate set
of source-grounded snippets and select a subset S
maximizing
F(S) =w relRel(S) +w qryQueryCov(S)
+w covRepr(S) +w divDiv(S)(1)
subject to cost(S)≤B and a snippet cap. Each
term is monotone and submodular, normalized to
[0,1] :Rel(modular) is the same per-snippet lexi-
cal relevance the focused heuristic uses—so heuris-
tic and packer see identical candidates and single-
ton scores, isolating theselection rule;QueryCov
is a set-cover over distinct query content
terms;Repris a saturated facility-location term,P
imin P
j∈Ssim(i, j), αdegi
, that rewards
covering candidate mass but saturates so it cannot
be gamed by near-duplicates;Divis a concave-over-
documents term,P
d√
relevance mass ofSind ,
spreading selection across sources. We lead
with relevance ( wrel=1.0 ,wqry=0.5 ,wcov=0.4 ,
wdiv=0.3 ,α=0.3 ); the other three terms act as
coverage/redundancy regularizers that push com-
plementary, answer-bearing evidence into the bud-
get.
4.2 Algorithm
We maximize Fwithcost-scaled (per-token)
greedy—at each step add the feasible snippet
with the largest marginal-gain-per-token ratio—
followed by theLin–Bilmes singleton fallback:
if the single best feasible snippet outscores the
greedy set, return it instead. This is the standard
constant-factor template for budgeted monotone
submodular maximization (Lin and Bilmes, 2011;
Nemhauser et al., 1978). The contribution is not the
optimizer (textbook) but (a) applying it to reader-
context packing, (b) the four-term objective tied
to answer density, and (c) the controlled evalua-
tion isolating the selection rule from the candidate
features.
4.3 Baselines and the factorial
Every packer consumes thesamecandidates, so
comparisons isolate the objective.Naive packed:
greedily concatenate by relevance until the budget
is hit.Focused heuristic: the project’s priorPolicy F1 EM Tokens
chunk_packed(naive) 0.400 0.306 151.1
chunk_focused0.429 0.331 152.1
chunk_mmr0.410 0.318 151.7
chunk_submod0.451 0.359 145.5
ace_focused0.428 0.328 150.0
ace_submod0.406 0.317 147.4
oracle (mixed) 0.601 0.487 141.7
Table 3: Three-seed means, HotpotQA-500, budget 160,
3B reader. chunk_submod is the best fixed policy on
everyseed, atfewertokens.
best packer (prefers new query-term coverage
across distinct documents, but checks the budget
only after the fact and never normalizes gain by
length).MMR(Carbonell and Goldstein, 1998):
arg max i[λrel(i)−(1−λ) max j∈Ssim(i, j)] ,
λ=0.7 —the natural “isn’t this just redundancy re-
duction?” control. Because the same packers apply
to flatchunkretrieval or toACEgraph-structured
evidence (a source-linked claim/entity graph from
earlier project stages), we evaluate a {chunk,
ACE} × { focused, MMR, submodular }factorial
plus a naive-packed anchor and a per-question
oracle, separating “does the packer help” from
“does the representation help.”
5 Results: The HotpotQA Win
Setup.All runs share a pipeline:
bge-small-en-v1.5 embeddings truncated
to 320 dimensions, Qwen2.5-3B-Instruct
reader, on dual T4 GPUs. HotpotQA uses 500
questions; the headline is replicated across seeds
{42,13,7} . The primary budget is 160 reader
tokens. Significance is paired bootstrap (10,000
resamples, 95% CI); multi-seed tests pool (seed,
question) instances.
The packer wins across three seeds.In Ta-
ble 3, chunk_submod is the best fixed policy
on every seed at ≈145 tokens versus ≈152
for the baselines. Pooled three-seed boot-
strap ( n=1,500 ): submod vs. focused +0.022
F1[+0.002,+0.041] ; submod vs. naive +0.051
F1[+0.030,+0.072] (+0.053 EM); submod vs.
MMR +0.042 F1; MMR vs. focused −0.020 F1
[−0.034,−0.005] . Three points: (1) the win is at
lowercost, not more context; (2) the ordering is
submod >focused >packed >mmr; (3) the “it is just
MMR” objection is empirically dead—plain MMR
issignificantly worsethan the focused heuristic, so
generic redundancy reduction hurts and only the

full coverage+representativeness+diversity objec-
tive wins.
Honest twist: packing helps chunk, not ACE.
The packer significantlyhurtsACE: ace_submod
vs.ace_focused is−0.021 F1[−0.039,−0.003] ,
and under submodular packing chunk beats ACE.
ACE already compresses and de-duplicates at the
graph level, so little redundancy remains for the
packer to exploit—graph compression and princi-
pled packing are partial substitutes. This relocates
the contribution from therepresentationto thepack-
ing objective, a finding only the factorial surfaces.
Mechanism: complementary multi-hop assem-
bly.A per-question decomposition (seed 42) at-
tributes81%of the submod–focused gain to37
questionswhere the packernewly placed a gold
answer into the reader context( ≈+0.39 F1 each).
The route is better complementary coverage—all
gold documents reach the context on 289 questions
under submod vs. 256 under focused—not higher
raw token density. The packer wins by moving ex-
actly the quantity the diagnostic measures. These
results use a 3B reader; §6.5 shows the advantage
over the focused heuristicis specific to this scale,
while the win over naive packing and the mecha-
nism persist.
Headroom, and why we do not claim a router.
The per-question oracle reaches F1 ≈0.60 vs. the
best fixed policy’s ≈0.45 . But chunk_submod is
already (tied-)best on 79.5% of questions; the re-
maining ≈20% is an “answer-in-context lottery”
whose deciding variable is unobservable at infer-
ence time, and an offline router over retrieval fea-
tures collapses toward the best fixed policy. We
therefore report the oracle asheadroom, not a de-
ployed method.
6When Does Principled Packing Help? A
Scope Map
The HotpotQA win is real butnot universal. We
ran controlled experiments to find its boundaries
and arrived at four conditions that must co-occur,
each presented with the experiment that fails it.
6.1 Condition 1: complementary structure
On RAGBench CovidQA ( n=246 ) and ExpertQA
(n=203 ), test split, the same factorial at budget 160,
submod vs. focused is not significant (CovidQA
−0.010 F1,p=0.30 ; ExpertQA +0.005 ,p=0.15 );
on CovidQA the focused heuristic is the best chunkMetric Baseline (§6.2) Wide retrieval
chunk recall@5 0.506 0.506
chunk all-gold@5 0.184 0.184
ACE all-gold@5 0.142 0.142
Table 4: Tripling MuSiQue retrieval depth (top- k5→12 ,
nodes 48→64 , expand 5→8 ) moves all-gold coverage
by zero basis points. The bottleneck is qualitative, not a
matter of depth.
packer and ACE regains an edge. These tasks are
single-pass with largely all-gold context, so there is
no complementary multi-hop structure for the ob-
jective to assemble. (Answer-in-context still tracks
quality,r=0.39on CovidQA.)
6.2 Condition 2: retrieval that surfaces the
evidence
MuSiQue is genuinely multi-hop but retrieval-
bottlenecked: recall@5 =0.506 yetall-
gold@5 =0.184 —only 18% of questions have all
gold retrieved. Submod vs. focused is +0.011
F1 (p=0.34 ), and naive packing is just as good;
ace_focused is the best fixed policy. The packer
cannot assemble evidence retrieval never surfaced.
Yet the diagnostic isstrongesthere ( r=0.54 ,
Table 2): answer-in-context still governs quality;
retrieval simply rarely achieves it.
6.3 Ruling out the obvious fix
A reviewer’s natural objection to §6.2 is “you just
did not retrieve enough.” We tested this: re-running
the full MuSiQue factorial with substantially wider
retrieval left all-gold coverageunchanged(Table 4),
and the packer gap stayed null ( +0.003 F1 at bud-
get 160). The bottleneck is therefore qualitative—
the bi-encoder cannot navigate 2–4 hop composi-
tional chains regardless of pool size—which con-
verts a soft negative into a precise statement: this
condition needs aqualitatively differentretriever
(iterative or chain-of-thought multi-hop (Trivedi
et al., 2023; Xiong et al., 2021)), not more depth.
6.4 Condition 3: binding-but-not-extreme
budget
We predicted the submod advantage would grow
monotonically as the budget tightens.It does
not(Fig. 3): the gap is an inverted-U peaking
at≈160 —at 96 only 2–3 snippets fit (no room
for complementarity); at 224 nearly everything fits
(the heuristic catches up). Two cleaner facts sur-
vive: submod beats naive packing ateverybud-

Figure 3: HotpotQA budget sweep (seed 42; B=160
is the three-seed result). The submod −focused gap is
an inverted-U, significant only at B≈160 (∆F1+0.035 ,
p=0.04 ): too tight and nothing complementary fits, too
loose and the heuristic catches up. Against naive pack-
ing (band) submod wins ateverybudget. Per-budget F1
in Table 7.
get (∆F1+0.044 to+0.055 , all p≤0.022 ),
and submod@160 matches focused@224 quality
(p=0.80 ) at≈30% fewer tokens (145 vs. 215)—
an iso-quality efficiency result.
6.5 Condition 4: a reader that is the
bottleneck
The sharpest objection to §5 is scaling:a stronger
reader can recover the answer from messier con-
text, so a packer that merely tidies it is irrele-
vant at scale.Rather than test one larger reader,
we trace the advantage along areader-scale lad-
der—Qwen2.5 at 3B, 7B, and 14B—re-running
the exact factorial of §5 and changing only the
reader. Because 14B needs 4-bit (NF4) quantiza-
tion (Dettmers et al., 2023) to fit dual T4s, we add a
same-sizeprecision control(7B in fp16and4-bit)
so any trend is attributable to scale, not quantiza-
tion.
Figure 4 and Table 5 give two clean readings.
(1) Scale, not quantization.The control is deci-
sive: 4-bit 7B ( −0.008 ,p=0.55 ) reproduces fp16
7B (−0.010 ,p=0.45 ) almost exactly—same sign,
magnitude, null, and best fixed policy.(2) Absorp-
tion then reversal.At 3B the packer beats the
focused heuristic ( +0.022 ); at 7B the contrast is a
symmetric null at both precisions; at 14B thefo-
cused heuristic significantly beats the packer
(−0.029 F1,p=0.013 ). Curation stops paying
at≈7B and begins tocostby 14B. Throughout,
chunk_submod still packs strictly more gold (cov-
erage≈0.78 vs.0.73) and still beatsnaivepacking
Figure 4: Reader-scale ladder (HotpotQA, budget 160,
seeds{42,13} pooled). The packer’s edge over the
focused heuristic(blue) is positive at 3B, null at 7B, and
significantlynegativeat 14B (∗p<0.05 ); the 7B fp16-vs-
4-bit control (hollow diamond) overlaps the fp16 point,
so the trend is scale, not quantization. The edge over
naive packing(red) stays significantly positive at every
rung.
Reader (precision) submod−focused∆F1p
3B fp16 (§5)+0.022 [+0.002,+0.041]<0.05
7B fp16−0.010 [−0.035,+0.015]0.45
7B 4-bit (control)−0.008 [−0.032,+0.017]0.55
14B 4-bit−0.029[−0.052,−0.006]0.013
Table 5: Reader-scale ladder, pooled 2-seed paired boot-
strap ( n=1,000 ; 3B is the three-seed headline). The
precision control rules out quantization.
significantly at every rung ( +0.044 to+0.055 F1,
p≤0.001 ). Reader capability is amediator: the
same density edge passed through readers of rising
sensitivity—once a reader can extract the answer
from the focused pack, denser gold buys nothing
and the packing overhead (a few extra distractor
documents) is a small liability.
6.6 Synthesis
HotpotQA at budget ≈160 with a 3B reader is
where all four conditions hold (Fig. 5), and there
the win is large, significant, and three-seed robust—
a narrow butprecise and mechanistically complete
scope. Conditions 1–3 are properties of the task
and budget under which the packer cannot help at
all; condition 4 is different in kind—the mecha-
nism still operates (it packs strictly denser gold)
but a strong enough reader stopsneedingthe com-
pleteness and by 14B mildlyprefersthe cleaner
pack. In every case thediagnostic is the unifying

1. Multi-hop, com-
plementary structure?
2. Retrieval surfaces the evidence?
3. Budget binding, but not extreme?
4. Reader is the bottleneck?
packer wins
HotpotQA, B≈160 , 3B readerno gain
RAGBench
(single-pass)
no gain
MuSiQue (all-
gold@5 =.18 )
no gain
B=96 or224
reversal
≳7B; flips by 14Bno
no
no
noyes
yes
yes
yes
Figure 5: When does principled packing beat the best
heuristic? All four conditions must hold; each “no” is a
regime we test where the win disappears—RAGBench
(§6.1), MuSiQue (§6.2–6.3), the budget extremes (§6.4),
and larger readers (§6.5). Conditions 1–3 gate the win
on or off; condition 4reversesit.
variable: each boundary is a distinct reason the
packer fails to raise answer-in-context, and accu-
racy tracks answer-in-context throughout (an inter-
ventional dissociation confirmed directly in §3.4).
7 Discussion
Why answer-in-context, not recall.Recall is
scored on a set the reader never sees. Under a bud-
get, the binding question is whether the answer
survives packing. Answer-in-context makes the
budgeted-RAG objective observable and turns “re-
trieve better” into the sharper “pack so the answer
survives.” It is cheap (a token-subsequence check)
and, where gold answers are short spans, broadly
applicable.
Why the submodular packer works when
it works.The win is not extra context
(submod usesfewertokens) or generic de-
duplication (MMR loses). It is the cover-
age+representativeness+diversity objective assem-
blingcomplementarymulti-hop evidence—81% of
the gain is questions where the answer newly enters
context. The diagnostic and the method describe
the same phenomenon from two directions.
Why the honest scope is the point.The facto-
rial surfaced a finding we would otherwise have
overclaimed: packing helps chunk, not ACE, be-
cause graph compression already removes the
redundancy the packer exploits. And the four-
condition scope—a falsified monotonicity predic-
tion, a retrieval-unlock ablation that ruled out
the easy fix, and a quantization-controlled reader-
scale ladder—is the kind of boundary-mapping thatmakes a conditional claim trustworthy. Locating
wherethe packer stops paying (and by 14B starts
to cost) tells a practitioner exactly when to reach
for it—small, efficient readers under tight budgets—
and when to prefer the simple heuristic.
Limitations
The headline win is demonstrated ononedataset
(HotpotQA) at one budget regime; the cross-dataset
experiments are negatives/boundaries by design,
so the positive claim rests on HotpotQA. The
reader ladder spans 3B/7B/14B but within asin-
glemodel family (Qwen2.5) and a single embedder
(bge-small-en ); whether the diagnostic’s predic-
tive power and the packer’s mechanism hold for
other reader families, stronger or instruction-tuned
retrievers, and 32B+ readers is untested. Some
sweeps (budget 96/128/224; the MuSiQue runs)
are single-seed; only the budget-160 headline is
three-seed. Answer-in-context is span-based and
therefore degenerate for long free-form answers
(ExpertQA), where a semantic/entailment variant
is needed. The ACE graph construction is heuris-
tic, so the “packing substitutes for graph compres-
sion” reading should be taken with that caveat. Fi-
nally, we measure EM/F1 and context properties,
not attribution faithfulness (Es et al., 2024; Saad-
Falcon et al., 2024); a faithfulness-aware version
of answer-in-context is left to future work.
8 Conclusion
Budget-constrained multi-hop RAG is bottlenecked
not by how many gold documents are retrieved
but by whether the answer survives packing into
the reader context. We introducedanswer-in-
context, a diagnostic that captures this and pre-
dicts answer quality better than retrieval recall
across five datasets, confirmed both observationally
(∆R2=+0.17 over recall) and interventionally (a
2Wiki manipulation that moves coverage but not
answer-in-context leaves accuracy flat). We intro-
duced abudgeted submodular evidence packer
that, with a 3B reader on HotpotQA, significantly
and robustly improves answer quality at equal-or-
lower token cost by assembling complementary
multi-hop evidence. And we mapped thescope
of that win to four conditions, each demonstrated
to fail, including a quantization-controlled reader-
scale ladder (3B →7B→14B) showing the edge
over the best heuristic is absorbed by 7B and signif-
icantly reverses by 14B, while the packer’s mecha-

nism and its win over naive packing persist through-
out. The result is a general diagnostic plus a condi-
tional, mechanistically explained method—sharply
located where it pays off: evidence-bottlenecked,
not reader-bottlenecked, budgeted multi-hop QA.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions (ICLR).
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. LongBench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics (ACL), pages
3119–3137.
Jeff Bilmes. 2022. Submodularity in machine learn-
ing and artificial intelligence.arXiv preprint
arXiv:2202.00132.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, and 1
others. 2022. Improving language models by retriev-
ing from trillions of tokens. InProceedings of the
39th International Conference on Machine Learning
(ICML), pages 2206–2240.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, and 1 others. 2020. Lan-
guage models are few-shot learners. InAdvances in
Neural Information Processing Systems (NeurIPS),
volume 33, pages 1877–1901.
Jaime Carbonell and Jade Goldstein. 1998. The use of
MMR, diversity-based reranking for reordering doc-
uments and producing summaries. InProceedings
of the 21st Annual International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval (SIGIR), pages 335–336.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 38, pages 17754–17762.
Tim Dettmers, Mike Lewis, Younes Belkada, and Luke
Zettlemoyer. 2022. LLM.int8(): 8-bit matrix multi-
plication for transformers at scale. InAdvances in
Neural Information Processing Systems (NeurIPS).
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. QLoRA: Efficient finetun-
ing of quantized LLMs. InAdvances in Neural Infor-
mation Processing Systems (NeurIPS).Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAS: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics
(EACL): System Demonstrations, pages 150–158.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey.arXiv preprint
arXiv:2312.10997.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Ming-Wei Chang. 2020. REALM: Retrieval-
augmented language model pre-training. InProceed-
ings of the 37th International Conference on Machine
Learning (ICML), pages 3929–3938.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of the 28th International
Conference on Computational Linguistics (COLING),
pages 6609–6625.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics (EACL), pages
874–880.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research (JMLR), 24(251):1–43.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023a. LLMLingua: Compress-
ing prompts for accelerated inference of large lan-
guage models. InProceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language
Processing (EMNLP), pages 13358–13376.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023b. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP), pages 7969–7992.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781.
Omar Khattab, Keshav Santhanam, Xiang Lisa
Li, David Hall, Percy Liang, Christopher Potts,
and Matei Zaharia. 2022. Demonstrate-search-
predict: Composing retrieval and language mod-
els for knowledge-intensive NLP.arXiv preprint
arXiv:2212.14024.

Omar Khattab and Matei Zaharia. 2020. ColBERT:
Efficient and effective passage search via contextu-
alized late interaction over BERT. InProceedings
of the 43rd International ACM SIGIR Conference on
Research and Development in Information Retrieval
(SIGIR), pages 39–48.
Andreas Krause and Daniel Golovin. 2014. Submodu-
lar function maximization. InTractability: Practical
Approaches to Hard Problems, pages 71–104. Cam-
bridge University Press.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems (NeurIPS), volume 33,
pages 9459–9474.
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference
efficiency of large language models. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 6342–
6353.
Hui Lin and Jeff Bilmes. 2010. Multi-document sum-
marization via budgeted maximization of submod-
ular functions. InHuman Language Technologies:
The 2010 Annual Conference of the North Ameri-
can Chapter of the Association for Computational
Linguistics (NAACL-HLT), pages 912–920.
Hui Lin and Jeff Bilmes. 2011. A class of submodular
functions for document summarization. InProceed-
ings of the 49th Annual Meeting of the Association
for Computational Linguistics: Human Language
Technologies (ACL-HLT), pages 510–520.
Xi Victoria Lin, Xilun Chen, Moya Chen, Weijia Shi,
Maria Lomeli, Rich James, Pedro Rodriguez, Jacob
Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettle-
moyer, and Scott Yih. 2024. RA-DIT: Retrieval-
augmented dual instruction tuning. InInternational
Conference on Learning Representations (ICLR).
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics (TACL), 12:157–173.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating the
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (ACL),
pages 9802–9822.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2023. MTEB: Massive text embedding
benchmark. InProceedings of the 17th Conference
of the European Chapter of the Association for Com-
putational Linguistics (EACL), pages 2014–2037.George L. Nemhauser, Laurence A. Wolsey, and Mar-
shall L. Fisher. 1978. An analysis of approximations
for maximizing submodular set functions—I.Mathe-
matical Programming, 14(1):265–294.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with BERT.arXiv preprint
arXiv:1901.04085.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. KILT: a benchmark for knowledge
intensive language tasks. InProceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics (NAACL),
pages 2523–2544.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Qwen Team. 2025. Qwen2.5 technical report.arXiv
preprint arXiv:2412.15115.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions
for machine comprehension of text. InProceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 2383–
2392.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics (TACL), 11:1316–1331.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
(EMNLP-IJCNLP), pages 3982–3992.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics (NAACL), pages 338–354.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2023, pages 9248–9274.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon
Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and
Wen-tau Yih. 2024. REPLUG: Retrieval-augmented
black-box language models. InProceedings of the
2024 Conference of the North American Chapter
of the Association for Computational Linguistics
(NAACL), pages 8364–8377.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evalua-
tion of information retrieval models. InAdvances in
Neural Information Processing Systems (NeurIPS),
Datasets and Benchmarks Track.
Hugo Touvron, Louis Martin, Kevin Stone, and 1 others.
2023. Llama 2: Open foundation and fine-tuned chat
models.arXiv preprint arXiv:2307.09288.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics (TACL), 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (ACL), pages 10014–10037.
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan
Parvez, and Graham Neubig. 2023. Learning to filter
context for retrieval-augmented generation. InarXiv
preprint arXiv:2311.08377.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems
(NeurIPS).
Johannes Welbl, Pontus Stenetorp, and Sebastian Riedel.
2018. Constructing datasets for multi-hop read-
ing comprehension across documents.Transactions
of the Association for Computational Linguistics
(TACL), 6:287–302.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval (SIGIR), pages 641–649.
Wenhan Xiong, Xiang Lorraine Li, Srinivasan Iyer,
Jingfei Du, Patrick Lewis, William Yang Wang,
Yashar Mehdad, Wen-tau Yih, Sebastian Riedel,
Douwe Kiela, and Barlas O ˘guz. 2021. Answer-
ing complex open-domain questions with multi-hop
dense retrieval. InInternational Conference on
Learning Representations (ICLR).Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024a. RE-
COMP: Improving retrieval-augmented LMs with
context compression and selective augmentation. In
International Conference on Learning Representa-
tions (ICLR).
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee,
Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina
Bakhturina, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024b. Retrieval meets long context large lan-
guage models. InInternational Conference on Learn-
ing Representations (ICLR).
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 2369–2380.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InInternational
Conference on Learning Representations (ICLR).
A Single-Seed Reference Table
Table 6 gives the per-policy means underlying the
answer-in-context mediation (§3.2) and decompo-
sition (§5), computed on seed 42’s 500 questions.
Policy F1 EM answer-in-context g-cov all-g tok
chunk_packed0.401 0.306 0.590 0.685 0.448 151.3
chunk_focused0.412 0.322 0.622 0.733 0.512 152.3
chunk_mmr0.390 0.296 0.600 0.680 0.418 151.9
chunk_submod0.448 0.3600.6400.779 0.578 145.4
ace_focused0.421 0.326 0.632 0.768 0.562 150.0
ace_submod0.401 0.316 0.612 0.758 0.550 147.4
oracle 0.598 0.484 0.700 0.774 0.582 141.8
Table 6: Seed-42 per-policy means, HotpotQA-500, bud-
get 160, 3B reader. g-cov =gold-doc reader coverage;
all-g=all-gold-in-reader.
Budget submod F1 focused F1∆F1p
96 0.373 0.369+0.0040.81
128 0.425 0.411+0.0140.45
160 0.448 0.412+0.0350.04
224 0.460 0.443+0.0170.26
Table 7: Per-budget F1 underlying the budget sweep
(Fig. 3; seed 42, B=160 is the three-seed result). The
submod −focused gap is an inverted-U peaking at ≈160 ,
not monotone.
B Reader-Scale Reference Tables
The packing/diagnostic columns are reader-
independent by construction, so they are identi-
cal across rungs; only EM/F1 move. Table 8 (7B

fp16) and Table 9 (14B 4-bit) are the two ends of
the ladder. The 7B 4-bit control reproduces 7B
fp16 (submod −focused −0.007 F1,p=0.55 ; same
best policy on both seeds), with absolute F1 ≈1–2
points lower (the quantization tax) but the contrast
unchanged.
Policy F1 EM answer-in-context g-cov all-g
chunk_packed0.332 0.259 0.583 0.685 0.447
chunk_focused0.396 0.311 0.638 0.741 0.526
chunk_mmr0.363 0.286 0.597 0.680 0.420
chunk_submod0.386 0.303 0.6340.779 0.582
ace_focused0.390 0.303 0.624 0.760 0.553
ace_submod0.371 0.277 0.610 0.750 0.541
oracle 0.574 0.461 0.704 0.780 0.588
Table 8: 7B fp16. Per-seed best: seed
42→chunk_submod (0.396); seed 13 →chunk_focused
(0.407).
Policy F1 EM answer-in-context g-cov all-g
chunk_packed0.386 0.295 0.583 0.685 0.447
chunk_focused0.460 0.3560.638 0.740 0.526
chunk_mmr0.413 0.311 0.597 0.680 0.420
chunk_submod0.431 0.329 0.6340.778 0.582
ace_focused0.457 0.353 0.624 0.760 0.553
ace_submod0.416 0.323 0.610 0.750 0.541
oracle 0.599 0.480 0.702 0.784 0.596
Table 9: 14B 4-bit. Per-seed best: seed
42→chunk_focused (0.459); seed 13 →ace_focused
(0.448). The focused policies are best—the opposite of
3B—with identical packing underneath.
C 2WikiMultiHopQA Interventional
Check
3B reader, budget 160, seeds {42,13,7} , 500
questions. Retrieval gate: recall@5 =0.718 ,
all-gold@5 =0.43 . Key contrast,
pooled 3-seed bootstrap ( n=1,500 ):
chunk_submod−chunk_focused=−0.008
F1[−0.027,+0.012] ,p=0.44 , with coverage
+0.054 but answer-in-context −0.007 —coverage
and answer-in-context move in opposite directions,
and F1 follows answer-in-context. Conditional F1
is0.56 when the answer is in context versus 0.08
when not.
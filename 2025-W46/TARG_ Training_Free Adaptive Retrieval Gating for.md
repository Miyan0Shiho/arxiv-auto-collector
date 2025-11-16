# TARG: Training-Free Adaptive Retrieval Gating for Efficient RAG

**Authors**: Yufeng Wang, Lu wei, Haibin Ling

**Published**: 2025-11-12 23:09:52

**PDF URL**: [https://arxiv.org/pdf/2511.09803v1](https://arxiv.org/pdf/2511.09803v1)

## Abstract
Retrieval-Augmented Generation (RAG) improves factuality but retrieving for every query often hurts quality while inflating tokens and latency. We propose Training-free Adaptive Retrieval Gating (TARG), a single-shot policy that decides when to retrieve using only a short, no-context draft from the base model. From the draft's prefix logits, TARG computes lightweight uncertainty scores: mean token entropy, a margin signal derived from the top-1/top-2 logit gap via a monotone link, or small-N variance across a handful of stochastic prefixes, and triggers retrieval only when the score exceeds a threshold. The gate is model agnostic, adds only tens to hundreds of draft tokens, and requires no additional training or auxiliary heads. On NQ-Open, TriviaQA, and PopQA, TARG consistently shifts the accuracy-efficiency frontier: compared with Always-RAG, TARG matches or improves EM/F1 while reducing retrieval by 70-90% and cutting end-to-end latency, and it remains close to Never-RAG in overhead. A central empirical finding is that under modern instruction-tuned LLMs the margin signal is a robust default (entropy compresses as backbones sharpen), with small-N variance offering a conservative, budget-first alternative. We provide ablations over gate type and prefix length and use a delta-latency view to make budget trade-offs explicit.

## Full Text


<!-- PDF content starts -->

TARG: Training-Free Adaptive Retrieval Gating for
Efficient RAG
Yufeng Wang†
Computer Science Department
Stony Brook UniversityLu Wei†
Computer Science Department
Stony Brook University
Haibin Ling∗
Computer Science Department
Stony Brook University
Abstract
Retrieval-Augmented Generation (RAG) improves factuality but retrieving for ev-
ery query often hurts quality while inflating tokens and latency. We propose
Training-free Adaptive Retrieval Gating (TARG), a single-shot policy that decides
when to retrieve using only a short, no-context draft from the base model. From
the draft’s prefix logits, TARG computes lightweight uncertainty scores—mean
token entropy, a margin signal derived from the top-1/top-2 logit gap via a mono-
tone link, or small-Nvariance across a handful of stochastic prefixes—and trig-
gers retrieval only when the score exceeds a threshold. The gate is model-agnostic,
adds only tens to hundreds of draft tokens, and requires no additional training or
auxiliary heads. On NQ-Open, TriviaQA, and PopQA, TARG consistently shifts
the accuracy-efficiency frontier: compared with Always-RAG1, TARG matches
or improves EM/F1 while reducing retrieval by 70-90% and cutting end-to-end
latency, and it remains close to Never-RAG in overhead. A central empirical find-
ing is that under modern instruction-tuned LLMs the margin signal is a robust
default (entropy compresses as backbones sharpen), with small-Nvariance offer-
ing a conservative, budget-first alternative. We provide ablations over gate type
and prefix length and use a∆-latency view to make budget trade-offs explicit.
1 Introduction
Large language models (LLMs) demonstrate strong performance on knowledge-intensive generation
yet remain susceptible to hallucination, producing fluent but unfounded content when parametric
memory lacks the necessary facts [Huang et al., 2025, Farquhar et al., 2024]. Retrieval-Augmented
Generation (RAG) addresses this by consulting an external corpus at inference time, coupling a gen-
erator with nonparametric memory to improve factuality and transparency [Lewis et al., 2020, Guu
et al., 2020, Wang et al., 2023, Izacard and Grave, 2020, Yu et al., 2024, Lin et al., 2023]. How-
ever, invoking retrieval for every input increases latency and token usage and can reduce accuracy
when retrieved evidence is noisy or tangential. Longer prompts also impose nontrivial computa-
tional costs due to the quadratic scaling of self-attention [Vaswani et al., 2017, Dao et al., 2022], and
evidence placed in the middle of long contexts is often under-utilized by current models [Liu et al.,
2023]. These considerations motivate selective retrieval: decidingwhenretrieval is unnecessary is
as important as decidingwhatto retrieve.
1ALWAYS-RAG: retrieve for every query; NEVER-RAG: never retrieve
2∗Corresponding author
3†These authors contributed equally to this work
Preprint. Under review.arXiv:2511.09803v1  [cs.CL]  12 Nov 2025

Recent work explores conditional retrieval, including forward-looking active retrieval (FLARE) that
anticipates upcoming content and re-queries when confidence appears low [Jiang et al., 2023],
instruction-tuned regulation with reflection tokens (Self-RAG) [Asai et al., 2024], and corrective
actions triggered by evidence quality scoring (CRAG) [Yan et al., 2024]. Unlike methods that train
control heads or run multi-stage tool loops, our method (i.e., TARG) aims to make a single, training-
free decision from prefix logits.
We revisit selective retrieval from a simpler angle: a training-free, single-shot decision ofwhen to
retrievethat any off-the-shelf language model (LM) can make before full decoding. Our approach,
Training-freeAdaptiveRetrievalGating (TARG), uses a short, no-context draft to read the model’s
own prefix logits and compute a lightweight uncertainty score. We instantiate three signals that
require no training or auxiliary heads: (1) mean token entropy, (2) a margin-based score derived
from the top-1 versus top-2 logit gap via a monotone link, and (3) a small-Nvariance from a handful
of stochastic prefixes. The gate triggers retrieval only when the score exceeds a fixed decision
threshold, adding only tens to hundreds of draft tokens per query and integrating cleanly into existing
RAG stacks.
A central empirical observation is that thechoice of signalmatters under modern instruction-tuned
LLMs. As backbones sharpen, prefix entropies often compress and may lose discriminative power.
In contrast, the top-1/top-2 margin retains dynamic range and correlates with cases where external
evidence changes or stabilizes the answer; small-Nvariance behaves similarly but is more conser-
vative. On NQ-Open [Kwiatkowski et al., 2019, Lee et al., 2019], TriviaQA [Joshi et al., 2017], and
PopQA [Mallen et al., 2022], TARG with a margin signal traces favorable accuracy-efficiency fronts
relative to both Never- and Always-RAG, achieving low retrieval rates and latency overheads close
to the Never baseline. We report results with both a compact generator and a stronger Llama-3.1-8B
model and adopt a∆-latency view that makes budget trade-offs explicit.
In summary, this work makes the following contributions:
•Training-free gate from prefix logits.We introduce TARG, a lightweight, model-agnostic
policy that decides when to retrieve using uncertainty computed from a short prefix;
among simple signals, the top-1/top-2 margin emerges as a robust default under modern
instruction-tuned LLMs, with small-Nvariance as a conservative alternative.
•Budget-aware calibration and cost framing.We provide a simple recipe to calibrate
the decision threshold to a target retrieval budget and report efficiency using∆latency,
isolating the incremental cost of retrieval and longer prompts from base decoding.
•Empirical validation across datasets and backbones.On NQ-Open, TriviaQA, and
PopQA, TARG improves the accuracy-efficiency frontier over Always/Never-RAG at min-
imal retrieval; trends hold across gate types and persist when upgrading the backbone.
2 Related Work
Retrieval-augmented generation (RAG).RAG couples parametric generators with non-
parametric memory to improve factuality on knowledge-intensive tasks. For example, REALM
exposes external knowledge during pretraining via a differentiable retriever [Guu et al., 2020]; RAG
and FiD popularize end-to-end training of retriever-reader pipelines that condition generation on
retrieved passages [Lewis et al., 2020, Izacard and Grave, 2020]. Subsequent variants enhance fu-
sion and rationale use [Wang et al., 2023] and adopt rank-then-rerank strategies [Yu et al., 2024].
Recent surveys synthesize these design choices and highlight robustness and cost as persistent chal-
lenges—long contexts inflate latency, retrieval noise can degrade accuracy, and budgets must be
controlled [Wu et al., 2024, Sharma, 2025]. These observations motivate policies that decidewhen
to retrieve, not onlywhatto retrieve.
Adaptive and active retrieval.A growing line of RAG research retrieves conditionally. FLARE
performs forward-looking active retrieval by anticipating upcoming content and re-querying when
predicted tokens look low-confidence [Jiang et al., 2023]. Self-RAG trains an LM with reflec-
tion/control tokens to decide when to retrieve and how to critique evidence [Asai et al., 2024].
CRAG scores the quality of retrieved passages and triggers corrective actions when evidence is
weak [Yan et al., 2024]. Very recent systems (e.g., SeaKR, SUGAR) leverage uncertainty probes
2

to modulate retrieval frequency [Yao et al., 2024, Zubkova et al., 2025]. While effective, these ap-
proaches often add supervision, special control tokens, auxiliary probers, or multi-stage loops that
increase engineering complexity and latency. In contrast, ourtraining-freeapproach, TARG, makes
a single, up-front decision using only prefix logits from the base model; among simple signals, the
top-1/top-2marginemerges as a robust default under modern instruction-tuned LLMs, with small-N
variance as a conservative alternative.
Reasoning-acting interleaving with tools.Frameworks such as Self-Ask and ReAct interleave
reasoning with tool use, allowing an LM to search or retrieve on demand during multi-hop solutions
[Press et al., 2022, Yao et al., 2023]. DSP composes retrieval and generation with programmatic
pipelines [Khattab et al., 2022]. These methods can yield strong performance but typically rely on
multi-turn tool calls, bespoke prompting, and orchestration overhead. Our scope is orthogonal: we
study aone-shotgate that decides whether to retrieve at all before full decoding. The two direc-
tions are complementary—TARG can suppress unnecessary retrieval in tool-augmented systems,
improving latency without altering downstream planners.
Uncertainty, calibration, and hallucination detection.Work on LM confidence studies whether
models “know when they know,” spanning logit-based, internal-state, and consistency-based sig-
nals [Geng et al., 2023]. SelfCheckGPT, for example, detects hallucinations via sample inconsis-
tency—conceptually related to our small-Nvariance signal, though we probe only a short prefix
rather than full generations [Manakul et al., 2023]. TARG operationalizes intrinsic, low-overhead
uncertainty from a brief no-context draft (entropy, margin, variance) to decide whether the expected
benefit of retrieval outweighs its cost. Empirically, as instruction-tuned backbones sharpen, prefix
entropies compress and over-trigger, whereas the top-1/top-2 margin and small-Nvariance retain
discriminative power—yielding selective, budget-aware retrieval that is easy to retrofit and orthogo-
nal to future improvements in retrievers and rerankers.
3 Method
Figure 1: Illustration of TARG methodology
Given a user queryqand a generator LLMG θwith tokenizerT, a RAG system augments the base
promptB(q)with an optional contextCretrieved from a corpusD, then decodes an answery. Let
x=B(q)⊕Cdenote the final prompt (token concatenation), and let the next-token distribution at
steptbep θ(yt|y<t, x). Our goal is to decide, at inference time and without training, whether to
retrieve (i.e., chooseC̸=∅) for a givenqso as to minimize compute cost—retrieval calls, context
tokens, and wall-time latency—while preserving or improving task accuracy.
3.1 Prefix uncertainty from a short draft
We exploit the model’s own uncertainty on a short, retrieval-free prefix to decide whether retrieval is
needed. We runG θforktokens on the base prompt only, obtaining logitsℓ 1, . . . , ℓ kand probabilities
3

πt= softmax(ℓ t), t∈ {1,2, ...k}. Each gate yields a per-step scoreu t, aggregated asU=
1
kPk
t=1ut.
Entropy gatePer-step entropyH t=−P
jπt,jlogπ t,j, with
Uent(k) =1
kkX
t=1Ht,(1)
so largerU entindicates higher uncertainty.
Margin-as-uncertainty gateLetg t=ℓt,(1)−ℓt,(2)≥0be the top-1 vs. top-2 logit gap at
stept. Map gaps to a positive uncertainty via a strictly decreasingϕ(defaultϕ(z) = exp(−z/β),
temperatureβ >0):
Umar(k;β) =1
kkX
t=1ϕ(gt)∈(0,1].(2)
Becauseϕis strictly decreasing, thresholdingU maris order-equivalent to thresholding themean
gap; the link is a convenience rather than a fragile choice.
Lemma 1 (order-equivalence).For any strictly decreasingϕ, thresholdingU maratτis equivalent
to thresholding the mean gap at someτ′, i.e., decisions are identical up to a monotone reparameter-
ization. (Proof shown in Appendix.)
Small-Nvariance gateSampleNshort stochastic prefixes (temperatureT) to obtain sequences
s(1), . . . , s(N). At stept, letˆp tbe the empirical token distribution of{s(n)
t}and defined t= 1−
max jˆpt(j). Then
Uvar(k, N) =1
kkX
t=1dt,0≤U var≤N−1
N.(3)
Lemma 2 (boundedness).The mode frequency is≥1/N, henced t≤(N−1)/Nand the stated
bound. (Proof shown in Appendix.)
3.2 Gate decision and decoding
Given the scalar uncertainty scoreUfrom the prefix draft, we trigger retrieval ifU > τ, theτis the
retrieve threshold:
retrieve(q)⇐⇒U(q)> τ.(4)
If retrieval is triggered, we construct the contextCwith a dense encoderE(e.g., E5-base-v2 [Wang
et al., 2022]) and a FAISS inner-product index [Douze et al., 2024]. Leth= norm(E(q))be the
normalized query embedding; we return top-KpassagesD K(q)by similarity⟨h, E(d)⟩and format
the context as
C=M
d∈DK(q)format(d)[1:L ctx],(5)
truncated to the context budgetL ctxtokens. The final prompt isx=B(q)⊕C; otherwise (no
retrieval) we proceed with zero-RAG usingx=B(q). For long generations, an optional single
re-check can be applied everymtokens: if the running-prefix score exceedsτand retrieval has not
yet occurred, retrieve once and continue decoding.
3.3 Cost, accuracy and calibration
Cost model.LetT draft=kdenote the always-incurred prefix,T ctxthe context tokens when retriev-
ing, andT(0)
out,T(1)
outthe output tokens without/with retrieval. With retrieval rateπ(τ) = Pr[U > τ],
the expected LM tokens per query are
E[T(τ)] =T draft+ (1−π(τ))E[T(0)
out] +π(τ) 
Tctx+E[T(1)
out]
,(6)
and an analogous additive decomposition holds for wall-time latency. We report efficiency using∆
latency to isolate the incremental overhead vs. the zero-RAG baseline.
4

Algorithm 1TARG inference (training-free gate)
Require:queryq, thresholdτ, prefix lengthk, (optional) recheck stridem, retrieverR, top-K,
context budgetL ctx
1:x←B(q)
2:draft←DECODEPREFIX(G θ, x, k)
3:U←SCORE(draft;gate∈ {entropy,margin,variance})
4:ifU > τthen
5:C←RETRIEVEFORMAT(R, q, K, L ctx)
6:x←B(q)⊕C
7:end if
8:y←GENERATE(G θ, x)▷optionally re-check everymtokens if not yet retrieved
9:returny
Accuracy model and dominance.LetA(0)(q)andA(1)(q)be correctness indicators (or prob-
abilities) for zero-RAG and with-RAG, and let∆(q) =A(1)(q)−A(0)(q). Assumeusefulness
calibration: for someτ ∗,E[∆(q)|U(q)≤τ ∗]≤0andE[∆(q)|U(q)> τ ∗]≥0. Then choosing
τ≈τ ∗yields
E[A gate(τ)] =E[A(0)] +E[∆(q)1{U(q)> τ}]≳max
E[A(0)],E[A(1)]	
.(7)
Intuition.Always-RAG integrates∆over allq, including negative-∆regions whereU≤τ ∗; Never-
RAG integrates zero. Thresholding nearτ ∗admits only the positive region. (Formal proof in Ap-
pendix.)
Threshold calibration and budget control.LetF Ube the empirical CDF ofUon a development
set (denoted bydev). To hit a retrieval budgetρ∈[0,1], pickτ=F−1
U(1−ρ). Alternatively, select
τthat maximizesdevaccuracy. BecauseUis scalar, calibration is fast and stable. In practice we
tunek(prefix length),βfor the margin link, andNfor variance on the samedevsplit.
3.4 Implementation details
We use a flat inner-product FAISS [Douze et al., 2024] index with a normalized dense encoder;
KandL ctxare tuned ondev. For the margin gate we default toϕ(z) = exp(−z/β)withβ=1,
which preserves ordering (Lemma 1) and yields interpretableU∈(0,1]. For variance we useN=3
samples by default (upper bound2/3). All gates reuse the samek-token prefix, adding only a few
dozen-hundred tokens per query.
4 Experimental Setup
Models, decoding, and baselines.We evaluate two instruction-tuned backbones, Qwen2.5-7B-
Instruct [Bai et al., 2023] and Llama-3.1-8B-Instruct [Grattafiori et al., 2024], under the same de-
coding protocol: greedy generation (no sampling), batch size1, identical prompts/stop criteria, and
the same short prefix for gating. The prefix is decoded without retrieval using the same greedy
policy; for the variance gate only, we drawNshort prefixes at temperatureT=0.7. We compare
Never-RAG (decode fromB(q)), Always-RAG (retrieve once per query and decode fromB(q)⊕C),
and our training-free gate TARG (draftktokens without retrieval, compute a scalar uncertaintyU,
and retrieve ifU > τ; no re-checks in main runs).
Retriever, index, and corpus.Without losing generality regarding retriever type, which was
proven in Appendix A.4, we use a frozen dense dual encoder E5-BASE-V2 [Wang et al., 2022]
and follow model guidelines by prefixing queries with"query:"and passages with"passage:".
Queries and passages are encoded into 768-d vectors,ℓ 2-normalized, and stored in a FAISS In-
dexFlatIP [Douze et al., 2024]; with normalization, inner product equals cosine similarity. At in-
ference, we return top-K=5passages and concatenate them as the retrieval context (formatted as
[title] textand truncated to fit a context budgetL ctx). The corpus is English Wikipedia [Foun-
dation]; articles are chunked into passages of roughly 1000 characters with 100-character overlap
(minimum 200 characters) and a large subset is indexed for all experiments. Our aim is to evaluate
5

Table 1:Selective retrieval with Qwen2.5-7B-Instruct.Representative TARG operating points
(two per dataset) against non-gated baselines. The thresholdτis shown for gated runs; “–” denotes
baselines.∆Latency is the added seconds per query relative to the Never-RAG baseline on the same
dataset/hardware; the NEVERrow shows the absolute baseline latency (s/q) in parentheses.
Dataset ModelτEM / F1 (%)↑Retrieval
Rate∆Latency
(s/q)↓
TriviaQANEVER– 60.8 / 61.4 0.000 2.947 (Baseline)
ALWAYS– 57.6 / 57.2 1.000 +3.462
ENTROPY0.80 61.8 / 62.2 0.028 +0.876
MARGIN0.1562.2 / 62.60.338 +2.174
VARIANCE0.75 61.8 / 62.2 0.006+0.133
PopQANEVER– 20.0 / 20.1 0.000 2.129 (Baseline)
ALWAYS– 14.6 / 14.6 1.000 +3.828
ENTROPY0.80 22.4 / 22.3 0.124+1.761
MARGIN0.3523.0 / 23.10.124+1.761
VARIANCE0.40 22.8 / 22.9 0.182 +1.847
NQ-OpenNEVER– 38.8 / 37.7 0.000 3.293 (Baseline)
ALWAYS– 37.4 / 36.7 1.000 +2.922
ENTROPY0.8539.6 / 39.10.046 +0.964
MARGIN0.2039.6/ 38.8 0.304 +1.295
VARIANCE0.60 38.6 / 38.0 0.012+0.291
a training-free gating policy, which emphasizes when to retrieve rather than to optimize the retrieval
stack; the gate is orthogonal to retriever quality and can sit atop stronger encoders or rerankers.
Datasets and evaluation protocol.We evaluate on NQ-Open [Lee et al., 2019], TriviaQA
[Joshi et al., 2017], and PopQA (long-tail entities) [Mallen et al., 2022]. Results are reported
for Never-/Always-RAG and gated operating points; for NQ-Open, we include threshold sweeps
and ablations. Prefix length is ablated overk∈ {10,20,30}, withk=20used by default (best
cost/quality balance). We instantiate three training-free signals from thek-token prefix: (i)Entropy
Uent=1
kP
t 
−P
jπt,jlogπ t,j
; (ii)MarginU mar=1
kP
tϕ(gt)whereg tis the top-1/top-2
logit gap andϕ(g) = exp(−g/β)withβ=1(order-preserving link yieldingU∈(0,1]); and (iii)
VarianceU var=1
kP
t 
1−max jˆpt(j)
fromN=3short stochastic prefixes atT=0.7, with range
[0,(N−1)/N](Nis the number of short prefixes).
Threshold sweeps, metrics, and reproducibility.For each gate, we sweep the decision threshold
τto expose accuracy-efficiency frontiers. We report Exact Match (EM) and F1 (standard normal-
ization), Retrieval Rateπ= Pr[U > τ], and∆latency — extra seconds per query relative to the
Never-RAG baseline on the same dataset/hardware; absolute latencies are provided once in table
footnotes. Calibration ofτcan target a retrieval budget via the empirical CDF ofUon adevset, but
in main results we present grid sweeps and select representative operating points at modest budgets
(∼5-20%). All runs use batch size1and identical decoding parameters across modes; for variance
we fix the temperature and RNG seed.
5 Results
Table 1 shows that unconditionally retrieving (ALWAYS) increases latency and often depresses ac-
curacy relative to NEVER, a hallmark of off-topic or aliased passages. Training-free gating restores
precision: both MARGINand VARIANCEmatch or exceed NEVERwhile staying far cheaper than
ALWAYS. On TriviaQA, MARGINattains the best quality by retrieving more frequently, whereas
VARIANCEachieves essentially the same quality at a tiny retrieval rate and negligible overhead,
reflecting its conservative trigger. On PopQA, where retrieval precision is harder, both gates im-
prove over ALWAYSand surpass NEVERwith moderate added cost; the gains are modest, consistent
with long-tail entity drift. On NQ-Open, small amounts of retrieval help: entropy/variance achieve
slight improvements at very low budgets, while margin trades a bit more budget for similar quality.
6

Table 2:Selective retrieval with Llama-3.1-8B-Instruct.Same protocol as Table 1.∆Latency
reports added seconds per query over the dataset’s Never baseline; absolute Never latencies (s/q)
appear in parentheses.
Dataset ModelτEM / F1 (%)↑Retrieval
Rate∆Latency
(s/q)↓
TriviaQANEVER– 80.8 / 80.0 0.000 10.383 (Baseline)
ALWAYS– 67.6 / 67.2 1.000 +1.069
ENTROPY0.80 74.4 / 74.1 0.524 +0.495
MARGIN0.7083.8 / 83.00.001+0.018
VARIANCE0.75 83.6 /83.00.001+0.018
PopQANEVER– 35.2 / 34.4 0.000 10.299 (Baseline)
ALWAYS– 24.8 / 24.6 1.000 +1.269
ENTROPY0.80 28.8 / 28.8 0.760 +0.974
MARGIN0.4536.6 / 36.20.108 +0.424
VARIANCE0.55 36.4 /36.0 0.084+0.317
NQ-OpenNEVER– 53.8 / 51.7 0.000 10.299 (Baseline)
ALWAYS– 48.6 / 46.1 1.000 +1.248
ENTROPY0.95 55.4 / 53.1 0.132 +0.175
MARGIN0.5057.6 / 54.70.008+0.012
VARIANCE0.60 56.8 / 53.7 0.026 +0.059
Overall, with a 7B-class model, VARIANCEis a strong default when budgets are tight; MARGINis
preferable when a small accuracy boost justifies higher (but still selective) retrieval.
With a stronger generator (Table 2), the full frontier shifts upward while the gate ordering becomes
clearer. ENTROPYnow over-triggers and lags in quality, indicating that prefix entropies compress
under sharper models and lose discriminative range. In contrast, MARGINand VARIANCEachieve
near-best accuracy at vanishing retrieval budgets and essentially zero overhead (e.g., +0.012 s on
NQ-Open), because the top-1/top-2 logit gap and small-Ndisagreement retain spread even when the
distribution is globally peaked. Practically, use MARGINby default; use VARIANCEwhen budgets
are extremely tight.
Table 3 situates our numbers against widely reported results that vary in backbone size, trained
retrieval/reranking, corpora, and scoring. These entries indicate headroom rather than serve as base-
lines. Two messages follow. First, absolute scores scale strongly with backbone and retrieval engi-
neering. Second, selective retrieval is orthogonal to those choices: a calibrated, training-free gate
can be dropped into stronger stacks and should continue to reduce unnecessary retrieval and latency.
Across datasets and backbones, the tables reveal a consistent picture. First, ALWAYSis not a safe
default: when the top-Kset contains distractors or aliases, small and mid-size generators spend com-
pute on irrelevant context and drift, so accuracy falls while latency rises. The effect is most visible on
long-tail entity queries (e.g., PopQA), where retrieval precision is intrinsically harder. Second, the
choice of uncertainty signal is pivotal and interacts with backbone sharpness. As instruction-tuned
LMs become more peaked, prefix entropies compress and lose ranking power; in contrast, the top-
1/top-2 logit gap (MARGIN) and small-Ndisagreement (VARIANCE) retain dynamic range and bet-
ter correlate with cases where external evidence flips or stabilizes the answer. This explains why, un-
der Llama-3.1-8B, MARGIN/VARIANCEachieve near-best quality at vanishing retrieval rates and es-
sentially zero added wall-time, while ENTROPYover-triggers and underperforms. Third, efficiency
should be interpreted as abudget, not just absolute time. Reporting∆latency isolates the incremen-
tal cost of retrieval and longer prompts beyond base decoding; the strongest MARGIN/VARIANCE
operating points cluster near the NEVERbaseline in latency yet exceed ALWAYSin accuracy, yield-
ing a controllable and deployment-friendly accuracy-efficiency frontier. Finally, these gains are
backbone-agnostic: with both Qwen2.5-7B and Llama-3.1-8B, TARG improves the frontier; the
preferred gate depends on budget and model sharpness—use MARGINby default, VARIANCEwhen
budgets are extremely tight, and reserve ENTROPYfor ablations or weaker backbones.
In deployment, calibrate the gate on a development set to match a target retrieval budget (via the
empirical CDF of the gate score). Use MARGINby default with modern instruction-tuned LLMs;
7

Table 3:External reference systems (context only).Results reported by prior work on our evalua-
tion datasets. These systems arenot directly comparableto our training-free, frozen-retriever setup:
most use larger backbones and/or trained retrievers/rerankers and may differ in corpora and scoring.
Values are EM or EM/Acc, whichever available.
ModelsNQ
EMTriviaQA
EMPopQA
EM
Without Retrieval-Augmented Generation
GPT-4-0613 [OpenAI, 2024] 40.3 84.8 31.3
GPT-4-turbo-2024-0409 [OpenAI, 2024] 41.5 80.0 25.0
With Retrieval-Augmented Generation
FiD-Large [Izacard and Grave, 2020] 51.4 61.6 –
RFiD-Large [Wang et al., 2023] 54.3 72.6 –
RA-DIT 65B [Lin et al., 2023] 35.2 75.4 –
Llama3-RankRAG 8B [Yu et al., 2024] 50.6 82.9 57.6
Llama3-RankRAG 70B [Yu et al., 2024] 54.286.5 59.9
Qwen2.5 7B-Entropy (τ= 0.85,0.8,0.8) 39.6 61.8 22.4
Qwen2.5 7B-Margin (τ= 0.2,0.15,0.35) 39.6 62.2 23.0
Qwen2.5 7B-Variance (τ= 0.6,0.75,0.4) 38.6 61.8 22.8
LLAMA3.1 8B-Entropy (τ= 0.95,0.8,0.8) 55.4 74.4 28.8
LLAMA3.1 8B-Margin (τ= 0.5,0.7,0.45)57.683.8 36.6
LLAMA3.1 8B-Variance (τ= 0.6,0.75,0.55) 56.8 83.6 36.4
switch to VARIANCEwhen budgets are extremely tight. Keep ENTROPYas an ablation or for weaker
backbones. Finally, report accuracy together with retrieval rate and∆latency so the quality-cost
frontier is explicit rather than implicit.
6 Discussion
Selective retrieval vs. unconditional retrieval.Across datasets and backbones, unconditionally
retrieving (ALWAYS) is not a safe default. When the top-Kset contains distractors or aliases, the
generator spends compute integrating irrelevant context; longer prompts further disperse attention,
increasing the chance that salient evidence is ignored. These effects are most acute on long-tail,
entity-heavy queries (e.g., PopQA), where lexical/semantic similarity can be misleading. In contrast,
a training-free gate that abstains on easy, high-confidence cases and triggers on genuinely uncertain
ones raises theprecisionof retrieval: prompts stay shorter on average, latency remains close to
NEVER, and accuracy does not suffer from “anchoring on a distractor.” This explains why ALWAYS
pays a consistent latency tax while often underperforming NEVER.
Which uncertainty signal—and why.Backbone sharpness governs the usefulness of different pre-
fix signals. As instruction-tuned models become more peaked, prefix entropies compress and lose
ranking power—entropy can be low even when the model is confidently wrong—so ENTROPYtends
to over-trigger and underperform. By contrast, the top-1/top-2 logit gap retains dynamic range: gen-
uine ambiguity shrinks the gap, raising a margin-based score; small-Nvariance captures a similar
phenomenon via disagreement across a few stochastic drafts. Empirically, MARGINand VARI-
ANCEachieve near-best accuracy at vanishing retrieval rates and essentially zero added wall-time
under stronger backbones (e.g., Llama-3.1-8B), while ENTROPYlags. A practical rule emerges:
use MARGINby default; switch to VARIANCEwhen budgets are extremely tight; keep ENTROPY
mainly for ablations or weaker backbones.
Dataset effects and PopQA.Selective retrieval is most effective when (i) many queries are
already covered by parametric knowledge and (ii) occasional retrieval injects noise or redun-
dancy—conditions that hold for NQ-Open and TriviaQA. PopQA stresses long-tail entities, aliases,
and temporally sensitive facts; dense retrieval can surface near-misses with high similarity that dis-
tract the generator. Under our deliberately training-free stack (frozen encoder, modestK), absolute
headroom is limited, but the gate still improves the quality-cost frontier relative to ALWAYS/NEVER.
8

Orthogonal upgrades—hybrid BM25+dense retrieval, cross-encoder reranking, entity-aware scor-
ing—raise absolute accuracy across all modes; a calibrated gate continues to suppress wasteful
retrieval on easy queries and preserves efficiency as the stack improves.
Budgets, not just time.We report∆latency as added seconds per query over NEVERon the same
hardware, isolating the incremental cost of prefix drafting, retrieval, and longer-context decoding
from base decoding:E[∆t]≈t draft +π(τ)t retrieval +π(τ)t decode|ctx .The strongest MAR-
GIN/VARIANCEoperating points cluster near the NEVERbaseline in∆latency yet exceed ALWAYS
in accuracy, yielding a controllable and deployment-friendly accuracy-efficiency frontier. In prac-
tice, calibrate the threshold to a target retrieval budget by matching the empirical CDF of gate scores
on a development set.
Context against larger systems.Reference numbers from training-heavy stacks (Table 3) provide
headroom, not baselines: they differ in backbone size, trained/hybrid retrieval and reranking, cor-
pora/snapshots, and scoring. Our contribution is orthogonal—decidingwhento retrieve—and can
be dropped into stronger stacks: as retrieval/reranking improves, all curves shift upward while a
calibrated gate continues to avoid unnecessary retrieval on easy queries.
7 Limitations
Our study targets a practical and narrowly defined question—how to decidewhento retrieve—so
several aspects are intentionally scoped. However, there are still limitations together with straight-
forward paths for us to improvement: (i) We evaluate English open-domain QA over Wikipedia.
While this setting is standard and stresses retrieval precision, it does not cover domain-specific cor-
pora, or multilingual inputs. Future work can apply the same training-free gate to hybrid or domain
corpora (e.g., web, scientific papers) and multilingual encoders, and to tasks such as fact verification
or long-form answer drafting. (ii) TARG uses a single threshold tuned on a small development set.
Although calibration is fast and stable (via the empirical CDF), thresholds may shift with models,
prompts, or domains. (iii) We deliberately use a frozen dense encoder to isolate the “when-to-
retrieve” decision. Absolute headroom is bounded by retrieval precision. This is orthogonal to our
method: the same gate can ride on stronger stacks (such as cross-encoder reranking). A compact
future experiment is to show that as retrieval improves, all modes rise while TARG continues to
suppress unnecessary retrieval.
8 Conclusion
We revisited retrieval-augmented generation from a simple angle:deciding when to retrievewith
no additional training. The proposed TARG policy reads uncertainty from a short, retrieval-
free prefix and triggers retrieval only when warranted. Among training-free signals, the MAR-
GINscore—derived from the top-1/top-2 logit gap—emerges as a robust default under modern
instruction-tuned backbones; VARIANCEprovides a conservative alternative when budgets are ex-
tremely tight. Framed through∆latency and retrieval rate, TARG consistently shifts the accuracy-
efficiency frontier: it matches or exceeds NEVERwhile avoiding the accuracy and latency penalties
of ALWAYS, and it does so at vanishing retrieval budgets.
Beyond raw numbers, the results deliver two actionable insights. First, unconditional retrieval is
not a safe default: when top-Kcontains distractors or aliases, longer prompts and off-topic context
hurt both quality and latency. Second, backbone sharpness dictates which uncertainty signal dis-
criminates: prefix entropies compress under stronger models, whereas the logit gap and small-N
disagreement retain dynamic range and remain predictive of when external evidence will flip or sta-
bilize the answer. These observations turn a one-line threshold into a practical control knob: set the
budget you can afford, calibrate once, and deploy.
TARG is intentionally plug-and-play. It neither assumes trained probers nor requires changes to the
retriever; it adds only tens to hundreds of draft tokens and introduces a single scalar control. As re-
trieval stacks improve (hybrid search, reranking, better chunking), absolute accuracy rises while the
gate continues to suppress wasteful retrieval on easy queries—yielding similar or larger efficiency
dividends at higher quality. We view training-free gating as a basic primitive for RAG systems: a
small, dependable mechanism that restores precision to retrieval, makes latency predictable, and is
simple enough to be widely adopted.
9

References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection.International Conference on Learning
Representations, 2024.
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,
Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu,
Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi
Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng
Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi
Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang
Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report.arXiv preprint
arXiv:2309.16609, 2023.
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R ´e. Flashattention: Fast and memory-
efficient exact attention with io-awareness.Advances in neural information processing systems,
35:16344–16359, 2022.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazar ´e, Maria Lomeli, Lucas Hosseini, and Herv ´e J´egou. The faiss library.arXiv
preprint arXiv:2401.08281, 2024.
Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large
language models using semantic entropy.Nature, 630(8017):625–630, 2024.
Wikimedia Foundation. Wikimedia downloads. URLhttps://dumps.wikimedia.org.
Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. A
survey of confidence estimation and calibration in large language models.arXiv preprint
arXiv:2311.08298, 2023.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783, 2024.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented
language model pre-training. InInternational conference on machine learning, pages 3929–3938.
PMLR, 2020.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language
models: Principles, taxonomy, challenges, and open questions.ACM Transactions on Information
Systems, 43(2):1–55, 2025.
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open
domain question answering.arXiv preprint arXiv:2007.01282, 2020.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992,
2023.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. triviaqa: A Large Scale
Distantly Supervised Challenge Dataset for Reading Comprehension.arXiv e-prints, art.
arXiv:1705.03551, 2017.
Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts,
and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for
knowledge-intensive nlp.arXiv preprint arXiv:2212.14024, 2022.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a
benchmark for question answering research.Transactions of the Association for Computational
Linguistics, 7:453–466, 2019.
10

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open
domain question answering.arXiv preprint arXiv:1906.00300, 2019.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Ro-
driguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. Ra-dit: Retrieval-augmented dual
instruction tuning. InThe Twelfth International Conference on Learning Representations, 2023.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and
Percy Liang. Lost in the middle: How language models use long contexts.arXiv preprint
arXiv:2307.03172, 2023.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi.
When not to trust language models: Investigating effectiveness and limitations of parametric and
non-parametric memories.arXiv preprint, 2022.
Potsawee Manakul, Adian Liusie, and Mark JF Gales. Selfcheckgpt: Zero-resource black-box hallu-
cination detection for generative large language models.arXiv preprint arXiv:2303.08896, 2023.
OpenAI. Gpt-4-0613 version, 2024.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring
and narrowing the compositionality gap in language models.arXiv preprint arXiv:2210.03350,
2022.
Chaitanya Sharma. Retrieval-augmented generation: A comprehensive survey of architectures, en-
hancements, and robustness frontiers.arXiv preprint arXiv:2506.00054, 2025.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need.Advances in neural informa-
tion processing systems, 30, 2017.
Cunxiang Wang, Haofei Yu, and Yue Zhang. Rfid: Towards rational fusion-in-decoder for open-
domain question answering.arXiv preprint arXiv:2305.17041, 2023.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-
jumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang, Xue Liu,
Tei-Wei Kuo, Nan Guan, et al. Retrieval-augmented generation for natural language processing:
A survey.arXiv preprint arXiv:2407.13193, 2024.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation.
2024.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. InInternational Conference on
Learning Representations (ICLR), 2023.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao, Linmei Hu, Weichuan Liu, Lei Hou, and Juanzi
Li. Seakr: Self-aware knowledge retrieval for adaptive retrieval augmented generation.arXiv
preprint arXiv:2406.19215, 2024.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and
Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation in
llms.Advances in Neural Information Processing Systems, 37:121156–121184, 2024.
Hanna Zubkova, Ji-Hoon Park, and Seong-Whan Lee. Sugar: Leveraging contextual confidence for
smarter retrieval. InICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 1–5. IEEE, 2025.
11

A Appendix
A.1 Proof of Lemma 1 (order-equivalence for the margin gate)
Recall the per-step logit gapsg t=ℓt,(1)−ℓt,(2)≥0fort= 1, . . . , kand a strictly decreasing,
continuousmargin linkφ:R ≥0→(0,1](e.g.,φ(z) =e−z/β). Define
Umar(k;φ)≜1
kkX
t=1φ(gt),and retrieve ifU mar> τ.
Fix any “shape” vectorδ= (δ 1, . . . , δ k)withP
tδt= 0and consider the family of gap vectors
g(µ) =µ1+δ, parameterized by the mean gapµ=1
kP
tgt.
Lemma 1(Location-equivalence of the margin gate).Letϕ:R→Rbe strictly decreasing. Fix
any “shape” vectorδ= (δ 1, . . . , δ k)∈RkwithPk
t=1δt= 0, and define the gap sequence along
the location family
g(µ) = (µ+δ 1, . . . , µ+δ k)∈Rk, µ∈R.
Define the margin-based gate statistic
Umar(µ) =1
kkX
t=1ϕ 
µ+δ t
.
ThenU mar(µ)is strictly decreasing inµ. Consequently, for any thresholdτ∈Rthere exists a
(unique) valueµ τsuch that
Umar(µ)≤τ⇐⇒µ≥µ τ.
That is, along the location familyg(µ), thresholdingU maris order-equivalent to thresholding the
mean gapµ.
Proof.Fixµ 1< µ 2. For each coordinatet∈ {1, . . . , k}we haveµ 1+δt< µ 2+δt, and sinceϕis
strictly decreasing,
ϕ(µ1+δt)> ϕ(µ 2+δt).
Averaging thesekstrict inequalities yields
1
kkX
t=1ϕ(µ1+δt)>1
kkX
t=1ϕ(µ2+δt),
i.e.,U mar(µ1)> U mar(µ2). ThusU maris strictly decreasing inµ.
Strict monotonicity implies that for anyτ∈Rthe equationU mar(µ) =τhas at most one solution;
when a solution exists, denote it byµ τ. BecauseU maris strictly decreasing, we have
Umar(µ)≤τ⇐⇒µ≥µ τ,
which shows that thresholdingU maris equivalent to thresholdingµalong the familyg(µ).
Remark(Scope).The lemma establishes order-equivalenceonly along location shiftsg(µ) =µ1+δ
with fixed shapeδ. For general changes of the per-step gap shape (i.e., changingδ),U maris still
coordinatewise decreasing in each argument, but it is not, in general, a function of the mean alone.
A.2 Proof of Lemma 2 (boundedness for the small-Nvariance gate)
Suppose we drawNshort stochastic continuations; at steptletˆp tbe the empirical token distribution
and define
dt≜1−max
jˆpt(j)∈[0,1], U var(k, N)≜1
kkX
t=1dt.
Lemma 2.For everyt,max jˆpt(j)≥1
N; hence
0≤d t≤1−1
N=N−1
N=⇒0≤U var(k, N)≤N−1
N.
Moreover, the upper bound is tight when allNsamples at a step are distinct.
12

Proof.AmongNsamples at stept, the modal token appears at least once, somax jˆpt(j)≥1
Nand
dt≤1−1
N. Averaging preserves bounds. Tightness: if all samples are distinct thenmax jˆpt(j) =
1
Nandd t=N−1
N.
A.3 Formalizing the intuition in §3.3: accuracy and budget calibration
LetA(0)(q), A(1)(q)∈[0,1]denote (calibrated) correctness without/with retrieval for queryq, and
let∆(q)≜A(1)(q)−A(0)(q)∈[−1,1]. Given a scoreU(q)and thresholdτ, define the retrieval
indicatorR τ(q) =1{U(q)> τ}. The gated accuracy is
Agate(τ;q) =A(0)(q) + ∆(q)R τ(q).
Dominance over NEVER-RAG under usefulness calibration.Assume there existsτ⋆s.t.
E[∆(q)|U(q)≤τ⋆]≤0,E[∆(q)|U(q)> τ⋆]≥0.
Proposition 1(Weak dominance over NEVER).Withτ=τ⋆,
E
Agate(τ)
≥E
A(0)
.
Proof.By the tower rule,
E
Agate(τ⋆)
=E
A(0)
+E
∆Rτ⋆
=E
A(0)
+ Pr(U > τ⋆)E[∆|U > τ⋆]≥E
A(0)
.
When the gate also beats ALWAYS-RAG.SinceE[A(1)] =E[A(0)] +E[∆], we have
E
Agate(τ⋆)
−E[A(1)] =E
∆Rτ⋆
−E[∆] =−E[∆1{U≤τ⋆}].
Proposition 2(Dominance over ALWAYSunder one-sided sign).If∆(q)≤0almost surely on
{U(q)≤τ⋆}(retrieval never helps in the low-Uregion), then
E
Agate(τ⋆)
≥E
A(1)
.
Proof.Under the stated sign condition,−E[∆1{U≤τ⋆}]≥0.
Budget calibration consistency.LetF Ube the CDF ofU. For a target retrieval rateρ∈[0,1],
setτ ρ=F−1
U(1−ρ). On an i.i.d. development set, the empirical quantileˆτ ρsatisfiesˆτ ρ→τ ρ
almost surely, and the realized retrieval rateˆπ(ˆτ ρ)→ρ. Thus quantile-based thresholding provides
a statistically consistent knob for meeting latency/compute budgets.
Cost/Lateness decomposition.LetT draft=kbe the prefix tokens,T ctxthe (bounded) retrieved
context size, andT(0)
out, T(1)
outthe output lengths without/with retrieval. Withπ(τ) = Pr(U > τ),
E
T(τ)
=T draft+ 
1−π(τ)
E[T(0)
out] +π(τ)
Tctx+E[T(1)
out]
,
so the incremental overhead relative to NEVERis directly governed byπ(τ), which is calibrated by
the score quantile.
A.4 Ablation Study: Retriever Sensitivity (E5-base-v2 vs. BGE-m3)
We hold the generator, prompts, corpus, chunking, FAISS type, and decoding fixed, and swap the
frozen dual encoder from E5-BASE-V2 to BGE-M3. For each dataset we report NEVER, ALWAYS,
and three training-free gates (ENTROPY, MARGIN, VARIANCE).∆Latency is added seconds/query
over the dataset’s NEVERbaseline for the same backbone. Thresholds are the representative settings
from the main results. For simplicity, the RR in the following context is short for the Retrieve rate,
and∆stands for the∆latency.
Based on Table S2 and S1, we observed three consistent phenomena:
13

•Unconditional retrieval remains a poor default, irrespective of retriever.Across
datasets and both backbones, ALWAYSpays a clear latency tax and frequently underper-
forms NEVER. With Qwen on TriviaQA, moving from E5 to BGEworsensALWAYS(F1:
57.2→51.7;∆latency:+3.462→+4.663), highlighting that the bottleneck is retrieval
precision rather than the gating policy. Similar patterns appear on PopQA and NQ-Open.
•The ordering of training-free gates is retriever-agnostic; costs remain minimal.Under
both E5 and BGE, ENTROPYtends to fire more often (e.g., Llama/TriviaQA RR= 0.524),
raising overhead with only moderate gains. In contrast, MARGINand VARIANCEachieve
near-best EM/F1 attinyretrieval rates and near-baseline∆latency. With Llama/NQ-Open,
MARGIN(E5) attains57.6/54.7EM/F1 at RR= 0.008and∆ = +0.012s, while VARI-
ANCE(BGE) reaches the same57.6/54.7EM/F1 at RR= 0.054and∆ = +0.122s. On
Qwen/TriviaQA, VARIANCEdelivers the best BGE quality at7.4%RR and only+0.151s
overhead, mirroring the E5 story.
•Absolute scores shift idiosyncratically with the retriever, but thefrontierstays the
same.Entropy with BGE on NQ-Open (Qwen) reaches a slightly higher EM/F1 than with
E5 (40.9/39.5vs.39.6/39.1) but at a much higher budget (Retrieve rate0.232vs.0.046),
while MARGIN/VARIANCEpreserve the “near-NEVERlatency, better-than-ALWAYSaccu-
racy” property across retrievers. These paired columns make clear that our improvements
stem fromwhen-to-retrieve decisions, not from retriever-specific quirks.
Table S1:Retriever sensitivity with Qwen2.5-7B-Instruct.Each row reports a method; columns
pairE5-base-v2vs.BGE-m3. Trends are consistent across retrievers: MARGIN/VARIANCEim-
prove the accuracy-efficiency frontier with minimal∆latency, while ALWAYSpays a latency tax
and often underperforms NEVER.
E5-base-v2 BGE-m3
Dataset Method EM/F1 RR∆(s) EM/F1 RR∆(s)
TriviaQANEVER 60.8/61.4 0.000 2.947(abs.) 60.8/61.4 0.000 2.947(abs.)
ALWAYS 57.6/57.2 1.000 +3.462 52.0/51.7 1.000 +4.663
ENTROPY 61.8/62.2 0.028 +0.876 62.0/62.6 0.028 +0.404
MARGIN 62.2/62.60.338 +2.174 61.8/62.5 0.001 +0.208
VARIANCE 61.8/62.2 0.006+0.133 62.0/62.70.074+0.151
PopQANEVER 20.0/20.1 0.000 2.129(abs.) 20.0/20.1 0.000 2.129(abs.)
ALWAYS 14.6/14.6 1.000 +3.828 16.8/16.8 1.000 +4.650
ENTROPY 22.4/22.3 0.124+1.761 22.0/22.1 0.418 +2.776
MARGIN 23.0/23.10.124+1.761 22.6/22.5 0.002+1.078
VARIANCE 22.8/22.9 0.182 +1.847 23.0/23.10.040 +1.520
NQ-OpenNEVER 38.8/37.7 0.000 3.293(abs.) 38.8/37.7 0.000 3.293(abs.)
ALWAYS 37.4/36.7 1.000 +2.922 38.2/37.7 1.000 +4.104
ENTROPY 39.6/39.10.046 +0.964 40.9/39.50.232 +1.787
MARGIN 39.6/38.8 0.304 +1.295 39.0/38.3 0.062 +0.713
VARIANCE 38.6/38.0 0.012+0.291 39.4/37.7 0.004+0.121
From Table S2 and Table S1, we conclude that the deltas for ALWAYSare diagnostic: whenever
retrieval precision dips (e.g., BGE on Qwen/TriviaQA), ALWAYSsuffers the most, confirming that
indiscriminate context is risky. Second, the behavior of ENTROPYunder stronger backbones remains
consistent across retrievers: peaked next-token distributions compress entropy and reduce ranking
power, so entropy-based gates over-trigger (high RR) and add latency without commensurate gains.
Third, MARGINand VARIANCEretain discriminative range because the top-1/top-2 gap and small-
Ndisagreement trackinstabilityin the prefix; that instability correlates with cases where external
evidence flips or stabilizes the answer. Finally, reading efficiency through∆latency shows the
budget clarity of gating: the best MARGIN/VARIANCEpoints cluster near the NEVERfloor while
outperforming ALWAYS,for both retrievers.
In short, the TARG method isretriever-agnostic. Upgrading the retriever raises absolute ceilings
forallmodes, but a calibrated MARGIN/VARIANCEgate continues to avoid wasteful retrieval on
14

Table S2:Retriever sensitivity with Llama-3.1-8B-Instruct.Same layout as Table S1. With
a stronger backbone, MARGIN/VARIANCEreach near-best quality at vanishing retrieval rates and
near-zero∆latency underbothretrievers, while ENTROPYover-triggers.
E5-base-v2 BGE-m3
Dataset Method EM/F1 RR∆(s) EM/F1 RR∆(s)
TriviaQANEVER 80.8/80.0 0.000 10.383(abs.) 80.8/80.0 0.000 10.383(abs.)
ALWAYS 67.6/67.2 1.000 +1.069 66.8/65.4 1.000 +1.581
ENTROPY 74.4/74.1 0.524 +0.495 75.6/74.9 0.524 +0.435
MARGIN 83.8/83.00.001+0.018 83.6/82.90.002 +0.034
VARIANCE 83.6/83.00.001+0.018 83.4/82.6 0.001+0.025
PopQANEVER 35.2/34.4 0.000 10.299(abs.) 35.2/34.4 0.000 10.299(abs.)
ALWAYS 24.8/24.6 1.000 +1.269 26.0/25.8 1.000 +0.700
ENTROPY 28.8/28.8 0.760 +0.974 32.4/32.4 0.760 +0.497
MARGIN 36.6/36.20.108 +0.424 36.2/35.6 0.001+0.063
VARIANCE 36.4/36.0 0.084+0.317 36.2/35.80.028 +0.290
NQ-OpenNEVER 53.8/51.7 0.000 10.299(abs.) 53.8/51.7 0.000 10.299(abs.)
ALWAYS 48.6/46.1 1.000 +1.248 46.2/44.8 1.000 +1.670
ENTROPY 55.4/53.1 0.132 +0.175 53.8/51.3 0.228 +0.505
MARGIN 57.6/54.70.008+0.012 57.0/54.1 0.008+0.028
VARIANCE 56.8/53.7 0.026 +0.059 57.6/54.70.054 +0.122
easy inputs and to dominate ALWAYSon the accuracy-efficiency frontier. This directly supports the
claim that training-free, budget-aware gating is a portable primitive for RAG systems.
A.5 Ablation study: Prefix lengthk.
Table S3 indicates that mid thresholds favor a 20-token prefix. Atτ=0.5,k=20attains the best
overall quality (EM 40.8, F1 39.4) at a moderate retrieval rate (23.2%), which is also the global
optimum in the sweep—suggesting 20 tokens are sufficient to stabilize the prefix-uncertainty signal
without over-triggering retrieval. Longer prefixes raise the retrieval rate without consistent gains: at
τ=0.3,k=30improves EM/F1 by only 1-1.3 points yet drives retrieval to 66% (vs. 56% fork=20
and 32.6% fork=10), an unfavorable quality-cost trade. Shorter prefixes under-inform the gate:
withk=10atτ=0.5, quality lags (EM 39.2 / F1 37.7) despite frugal retrieval (11.0%), implying ten
tokens often fail to expose the uncertainty patterns that predict retrieval benefit. At higher thresholds
(τ=0.65-0.8), quality differences narrow, butk=20remains best or tied on EM and best on F1 while
keeping retrieval low (6-13%).Overall, a 20-token draft offers the best quality-cost balance; we
adoptk=20as the default and calibrate the threshold at mid values (e.g., via score quantiles) to meet
a target retrieval budget.
Table S3: Prefix-length ablation for TARG (gate fixed). Each cell showsEM / F1 [Retrieval Rate]
in %. Bold indicates the best EM (primary) at each threshold. A 20-token draft (k=20) yields the
best overall quality at mid thresholds while avoiding the high retrieval of longer prefixes.
Prefixkτ=0.3τ=0.5τ=0.65τ=0.8
k=1038.6 / 37.5 [32.6] 39.2 / 37.7 [11.0] 39.2 / 38.2 [6.0] 39.4 / 38.3 [2.8]
k=2038.2 / 37.2 [56.0]40.8/39.4[23.2] 39.8 /39.2[13.0]39.8/38.8[6.0]
k=3039.6/38.5[66.0] 40.0 / 38.5 [31.2]40.0/ 38.7 [16.0] 39.0 / 37.7 [7.2]
15
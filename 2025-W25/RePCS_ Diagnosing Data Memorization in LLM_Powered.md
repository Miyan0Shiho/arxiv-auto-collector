# RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation

**Authors**: Le Vu Anh, Nguyen Viet Anh, Mehmet Dik, Luong Van Nghia

**Published**: 2025-06-18 14:48:19

**PDF URL**: [http://arxiv.org/pdf/2506.15513v1](http://arxiv.org/pdf/2506.15513v1)

## Abstract
Retrieval-augmented generation (RAG) has become a common strategy for
updating large language model (LLM) responses with current, external
information. However, models may still rely on memorized training data, bypass
the retrieved evidence, and produce contaminated outputs. We introduce
Retrieval-Path Contamination Scoring (RePCS), a diagnostic method that detects
such behavior without requiring model access or retraining. RePCS compares two
inference paths: (i) a parametric path using only the query, and (ii) a
retrieval-augmented path using both the query and retrieved context by
computing the Kullback-Leibler (KL) divergence between their output
distributions. A low divergence suggests that the retrieved context had minimal
impact, indicating potential memorization. This procedure is model-agnostic,
requires no gradient or internal state access, and adds only a single
additional forward pass. We further derive PAC-style guarantees that link the
KL threshold to user-defined false positive and false negative rates. On the
Prompt-WNQA benchmark, RePCS achieves a ROC-AUC of 0.918. This result
outperforms the strongest prior method by 6.5 percentage points while keeping
latency overhead below 4.7% on an NVIDIA T4 GPU. RePCS offers a lightweight,
black-box safeguard to verify whether a RAG system meaningfully leverages
retrieval, making it especially valuable in safety-critical applications.

## Full Text


<!-- PDF content starts -->

arXiv:2506.15513v1  [cs.LG]  18 Jun 2025RePCS: Diagnosing Data Memorization in
LLM-Powered Retrieval-Augmented Generation
Le Vu Anh∗, Nguyen Viet Anh†, Mehmet Dik‡, Luong Van Nghia§
∗Institute of Information Technology, Vietnam Academy of Science and Technology, Hanoi, Vietnam
Email: anhlv@ioit.ac.vn
†Institute of Information Technology, Vietnam Academy of Science and Technology, Hanoi, Vietnam
Email: anhnv@ioit.ac.vn
‡Department of Mathematics, Computer Science & Physics, Rockford University, Illinois, United States
Email: mdik@rockford.edu
§Department of Information Technology, Dong A University, Da Nang, Vietnam
Email: nghialv@donga.edu.vn
Abstract —Retrieval-augmented generation (RAG) has become
a common strategy for updating large language model (LLM)
responses with current, external information. However, models
may still rely on memorized training data, bypass the retrieved
evidence, and produce contaminated outputs. We introduce
Retrieval-Path Contamination Scoring (RePCS), a diagnostic
method that detects such behavior without requiring model
access or retraining. RePCS compares two inference paths: (i)
aparametric path using only the query, and (ii) a retrieval-
augmented path using both the query and retrieved context by
computing the Kullback–Leibler (KL) divergence between their
output distributions. A low divergence suggests that the retrieved
context had minimal impact, indicating potential memorization.
This procedure is model-agnostic, requires no gradient or internal
state access, and adds only a single additional forward pass. We
further derive PAC-style guarantees that link the KL threshold
to user-defined false positive and false negative rates. On the
PROMPT -WNQA benchmark, RePCS achieves a ROC-AUC of
0.918. This result outperforms the strongest prior method by 6.5
percentage points while keeping latency overhead below 4.7%
on an NVIDIA T4 GPU. RePCS offers a lightweight, black-
box safeguard to verify whether a RAG system meaningfully
leverages retrieval, making it especially valuable in safety-critical
applications.
Index Terms —network state queries, data memorization,
retrieval-augmented generation, large language models, KL
divergence
I. I NTRODUCTION
Over the past five years, large-scale language models (LLMs)
such as GPT-4, Claude 3, and Llama 2 have transformed hu-
man–computer interaction: users can pose open-ended questions
and receive coherent, context-aware answers in natural language
[1]. To increase factual reliability, industry and academia
have turned to the Retrieval-Augmented Generation (RAG)
paradigm, in which an LLM is paired with a search layer that
retrieves passages from an external knowledge base (KB) [2].
By explicitly searching for supporting evidence, RAG promises
to reduce hallucinations, provide up-to-date information beyond
the model’s pre-training cut-off, and offer verifiable citations,
making it the de-facto architecture in production chatbots,
search engines, and enterprise assistants [3].In modern wireless and networked systems, RAG is increas-
ingly used to fetch live network state data such as channel
quality measurements, interference statistics, or handover logs
before making critical decisions on power control, resource allo-
cation, and mobility management [4], [5]. However, we observe
a problematic failure mode in this adoption: the LLM models
embedded within RAG systems may silently skip fetching fresh
telemetry and instead replay information memorized during its
pre-training. This silent data contamination in network state
queries causes controllers to act on stale or irrelevant facts
without any indication of error, potentially destabilizing link
budgets or misallocating spectrum.
Early RAG systems relied on lexical retrievers such as
BM25 [6], but the field quickly moved toward dense dual-
encoders [7] and late-interaction models [8]. Alongside retriever
improvements, a second line of work fine-tunes LLMs to
better exploit retrieved passages, using techniques ranging from
contrastive reward signals to retrieval-conditioned causal mask-
ing [9]. Parallel to these engineering advances, the evaluation
community has sounded the alarm on data contamination, the
hidden overlap between training corpora and supposedly “held-
out” test sets [10]. In order to detect this issue, researchers
have proposed matching n-grams, analysing log-probabilities,
or tracing influence functions to detect when benchmark results
are inflated by memorized content [11]–[13].
Despite gains in factuality and benchmarking rigour, pre-
vailing methods share a critical assumption. If the retriever
returns high-quality passages, the generator will faithfully
incorporate them. In practice, a powerful LLM may already
remember the answer from pre-training, quietly ignoring or
only superficially citing the external context. Under such
circumstances, even a perfectly engineered retriever cannot
prevent leakage; evaluation metrics can be silently inflated, and
network operators may over-trust the system’s grounding.
Existing contamination detectors still leave two critical gaps.
First, most are intrusive : influence-function analysis, gradient
masking, or logit inspection demands white-box access to full-
precision parameters, which is often impossible for proprietary
or quantized models. Second, they probe individual tokens or

retrieved passages, but never ask a simpler question: Does the
final answer change when retrieval is removed? As a result,
practitioners lack a lightweight, black-box test that verifies
whether a RAG pipeline truly relies on the documents it fetches.
Retrieval-Path Contamination Scoring (RePCS) closes
this gap. For every query we execute the same LLM model
along two inference paths:
1)Retrieval-augmented path : the query combined with the
top-Kpassages returned by the search layer;
2)Parametric path : the query alone, with no external
context.
We then compute a single Kullback–Leibler (KL) divergence
between the two answer-probability distributions. A small KL
value indicates that the retrieved context has little influence.
This is evidence that the model is relying on memorised
training data, whereas a large KL value certifies that retrieval
introduces new information. The score is obtained post-hoc,
without gradients, logits, or model modifications, and costs
only one extra forward pass.
Our contributions include:
•Training-free, black-box detector . RePCS compares a
retrieval-augmented run with a parametric run of the same
LLM; one additional forward pass and a KL divergence
flag potential memorisation in real time.
•Provable guarantees . We derive PAC-style bounds that
map the KL threshold to user-specified false-positive and
false-negative rates, providing principled control over
detection sensitivity.
•Practical speed . On the PROMPT -WNQA benchmark,
RePCS lifts ROC-AUC by 6.5 pp over the strongest
baseline while adding ≤4.7%latency on an NVIDIA
T4 GPU, making it suitable for live deployments.
The rest of the paper is organized as follows. Section II
surveys related work in RAGs and data memorization detection.
Section III formalizes RePCS, detailing the dual-path LLM
protocol and KL scoring rule, and presents our theoretical
guarantees. Section IV describes the algorithmic framework.
Section V outlines the experimental setup and empirical results.
Finally, Section VI summarizes key findings and discusses
future directions.
II. R ELATED WORK
Early black-box approaches treat the language model as an
oracle and rely on output variability to detect memorization.
SELFCHECK GPT (2023) re-generates an answer several times
and marks it as unsafe when the paraphrases disagree, showing
that output instability is a useful clue even without access to
model internals or the retriever [11]. While simple, the method
needs several additional calls per user query, increasing latency
and computation cost.
A second research line opens the model to gather richer
signals. REDEEP(2025) traces token probabilities back to indi-
vidual attention heads and reports that some heads consistently
carry memorized content, whereas others depend on retrieved
passages [12]. LLM-C HECK (2024) also measures hidden-
state norms and attention-map entropy, offering both white-and grey-box variants that raise precision but still require layer-
level features or gradient access [14]. These probes achieve
high accuracy yet face compatibility issues when models are
quantised or internal layouts change.
To meet real-time budgets, industry deployments favour
lightweight detectors trained with supervision. The Two-Tiered
system (2024) fine-tunes two compact encoders: one over the
query plus retrieved passages and another over the final answer
and flags hallucination when their embeddings diverge, adding
only a few milliseconds to each request [13]. LUNA (2025)
pushes latency lower by distilling a DeBERTa-large model
into a 45 MB checkpoint that keeps GPT-3.5-level detection
accuracy with one-tenth the inference cost [15]. Both methods,
however, depend on thousands of labelled examples and must
be re-trained whenever the base LLM or retriever changes.
An orthogonal direction checks factual claims di-
rectly. REFCHECKER (2024) converts an answer into sub-
ject–predicate–object triples and verifies each triple against
the retrieved documents, catching span-level conflicts when
evidence is missing or contradictory [16]. Reliability here
depends on the quality of open-IE extraction and full evidential
coverage: if the retriever misses a supporting passage, the
checker may wrongly label a true statement as hallucinated.
Progress in supervised detection and claim checking has been
accelerated by RAGT RUTH (2024), a benchmark of 18 000
RAG answers with word-level hallucination labels that enables
fair comparison across methods and fuels new detectors [17].
Several studies move beyond detection to repair or avoidance.
RAG-HAT (2024) feeds detector-identified hallucination spans
back into the prompt of a large model and asks it to rewrite
its own answer, reducing factual errors without extra human
input [18]. In a complementary path, counterfactual prompting
method (2024) treats the detector’s score as a risk estimate
and trains the generator to abstain when that risk exceeds a
user-defined threshold, trading coverage for higher reliability
in safety-critical settings [19].
Our observations . Existing detectors either require multiple
language-model calls, demand intrusive access to gradients
or attention weights, or depend on supervised training with
large labelled sets. Each factor here adds cost, latency, or
maintenance overhead. Specifically, none of them tests whether
thesame LLM, run once with retrieved passages and once
without, yields answer distributions that differ enough to prove
genuine grounding, nor do they provide user-tunable statistical
guarantees on false-alarm rates. These gaps motivate our
REPCS framework, which stays black-box, needs no additional
training, measures a single KL divergence between the retrieval-
augmented and parametric answer distributions, and offers
provable error bounds while adding less than five per-cent
latency.
III. P RELIMINARIES
A. Concept and Notation
We begin by formalising the two evaluation pathways that
are critical to R EPCS.

Step 1:
Input user prompt q
Step 2:
Retrieve top- Kpassages
Step 3a:
LLM on q+R(q)
→Prag
Step 3b:
LLM on qonly
→Ppara
Step 4:
KL score
Z= KL( Prag∥Ppara)
Step 5:
Decision
flag “memorised” if Z < τLegend
retrieval-augmented path
parametric path
smallZ⇒passages ignored
Fig. 1. Overview of REPCS . We run the LLM twice: once with retrieved passages and once without. We then use a single KL score to flag possible
memorization.
Letqbe a user query drawn from natural language. A
retriever module returns the Kmost relevant passages, writ-
tenR(q) ={c1, . . . , c K}. A frozen large language model
(LLM) Mcan now be queried in two distinct modes [20]:
(i) Retrieval-augmented path Prag:M 
⟨q, R(q)⟩
−→Prag,
(ii) Parametric path Ppara:M(q)−→Ppara.
The outputs Prag, Ppara∈∆V×Tare full token-probability
tensors over a vocabulary of size Vfor the first Tgenerated
positions. Throughout the paper we treat both tensors as
probability distributions and make no further assumptions about
the internal architecture of M.
In a RAG system, useful evidence in R(q)ought to perturb
the LLM’s predictive distribution. If instead we observe Prag≈Ppara, the passages have had virtually no influence, and the
answer most likely originates from the model’s parametric
memory [21]. Detecting this “silent fallback” is exactly the
goal of R EPCS.
Rather than inspecting two V×Tarrays, we compress their
difference into the Kullback–Leibler divergence [22]
Z(q) = KL 
Prag∥Ppara
,
which carries several advantages:
•Interpretability :Z(q)quantifies, in nats, the extra log-
likelihood required to pretend that retrieval mattered when
it did not.
•Monotonicity :Z(q)equals zero iffPrag=Pparaand increases
smoothly with any deviation.

TABLE I
PRIMARY SYMBOLS USED IN §III.
Symbol Description
q user query
R(q) retrieved evidence (top- Kpassages)
Prag, Ppara token distributions on the two paths
Q distribution conditioned on R(q)alone
η retrieval-influence coefficient (Def. III.1)
Z(q) KL score KL(Prag∥Ppara)
τ decision threshold (learned)
T inspected output length (tokens)
γ bound on per-token log-ratio
α, ϵ target false-positive / false-negative rates
•Statistical tractability : KL enjoys tight concentration and
minimax bounds, allowing finite-sample error guarantees
[23], [24].
A learned threshold τis finally applied: queries with Z(q)< τ
are flagged as memorised .
For convenience, Table I lists the primary symbols used
throughout the preliminaries section.
B. Retrieval-Influence Coefficient
We next formalise a single scalar that captures how strongly
the retrieved passages alter the model’s predictive distribution.
Definition III.1 (Influence η).There exists η∈[0,1]such that
Prag= (1−η)Ppara+η Q, Q :=P 
·“useR(q)only”
,
(1)
where Qis the token distribution obtained when the LLM is
forced to attend exclusively to the retrieved passages.
Modern encoder–decoder LLMs blend two information
sources at inference time: (i) parametric memory , which is
weights distilled from pre-training corpora; (ii) non-parametric
evidence , which is the textual passages in R(q).
A convex mixture is the minimal model that preserves both
sources while making no architectural assumptions about M.
The coefficient ηtherefore admits the following operational
interpretation:
•η= 0 =⇒retrieval has no effect ; the answer is fully driven
by parametric memory.
•η= 1 = ⇒the model’s distribution coincides with the
“evidence-only” distribution Q; parametric memory is com-
pletely overridden.
•0< η < 1 =⇒retrieval and memory interpolate . The closer
ηis to zero, the weaker the influence of R(q).
Although ηis not directly observable, it is identifiable from
the pair (Prag, Ppara)whenever the support of Qdiffers from
that of Ppara. In practice we avoid estimating ηexplicitly; it
suffices to know (via Theorem III.2) that small ηforces the
KL score Z(q)to collapse.
The mixture model turns the abstract notion of “retrieval
influence” into a concrete algebraic parameter that:
(a)upper-bounds the KL divergence when evidence is ignored,
and(b)lower-bounds it when evidence is incorporated (Assump-
tion 1).
These complementary bounds are the linchpins that make the
finite-sample guarantee in Theorem III.4 possible.
C. KL–Based Memorisation Score
Having introduced the influence coefficient η, we now
instantiate a concrete test statistic.
The detector aggregates all evidence about retrieval influence
into a single scalar:
Z(q) =TX
i=1P(i)
raglog 
P(i)
rag
P(i)
para!
. (2)
A query is declared memorised when Z(q)< τ , where
the threshold τis learned on held-out, contamination-free
queries (see §III-G ). We highlight three reasons for adopting
Kullback–Leibler divergence as the sole decision statistic.
Among f-divergences, KL is uniquely characterised by
the property that its first Gateaux derivative equals the log-
likelihood ratio. Intuitively, a small Z(q)means an observer
who assumes retrieval altered the answer would gain virtually
no extra log-likelihood over an observer who assumes pure
parametric generation. This is exactly the condition we wish
to detect.
Because KL aggregates over the V×Ttoken grid into nats,
its magnitude is directly comparable across different model
sizes, vocabulary granularities, and generation lengths. This
invariance enables a single threshold τto remain valid as the
underlying LLM or tokenizer evolves, a key requirement in
industrial deployments where models are upgraded frequently.
KL enjoys sub-Gaussian or sub-exponential tails whenever
individual log-ratios are bounded. This is a condition that holds
for any soft-max output clipped by finite precision. These tails
translate into finite-sample guarantees (Type I / Type II bounds)
via Hoeffding’s and Massart’s inequalities, without resorting to
asymptotic normality arguments that would be inappropriate
at the single-query level.
Here we note that
KLPragPpara= H( Prag, Ppara)−H(Prag),
where H(·,·)denotes the cross-entropy and H(·)the Shannon
entropy.
This means that Z(q)can be interpreted as the excess cross-
entropy one incurs by pretending the model ignored retrieval;
when retrieval is in fact ignored, that excess collapses to numer-
ical noise. This connection yields an efficient implementation:
both H(Prag)andH(Prag, Ppara)are already computed during
standard log-prob evaluation, so no extra GPU kernels are
required.
Finally, we emphasise that (2) serves as a one-shot test:
Z(q)< τ =⇒ flag “memorised” .
No secondary heuristics, paraphrase sampling, or gradient
probes are invoked. All statistical guarantees that follow hinge
exclusively on the behaviour of this scalar.

Fig. 2. KL Collapse vs Retrieval Influence. Empirical mean KL values
(dots) versus mixture coefficient η, plotted alongside the theoretical bound
Tη2/[2(1−η)]from Theorem III.2. Even with injected noise, observed
KLs remain safely below the theoretical curve. Demonstrates how memorized
queries induce near-zero divergence under low retrieval influence ( η≤0.1).
D. KL Collapse for Memorised Queries
We now explain specifically why our framework can detect
data memorization behaviors.
Inside a decoder-only LLM, the next-token probabilities are
an affine function of its hidden state. When we concatenate
⟨q, R(q)⟩, the hidden state becomes
hrag= (1−η)hpara+ηhR
under the widely-observed linear superposition behaviour of
transformer activations.
Because the soft-max is log-affine , every log-odds term
is shrunk by at most η. Consequently, the KL divergence,
which is a second-order quantity, drops quadratically in η. For
verbatim facts ( η≤0.1) the change is so small that it cannot
be distinguished from floating-point noise; the detector then
rightly concludes “the passages were ignored.”
Theorem III.2 (KL collapse) .If the mixture coefficient in
Definition III.1 satisfies η≤1
2, then
Z(q)≤η2
2(1−η).
E. Concentration of the KL Score
Modern LLM stacks quantise activations to 16-bit or 8-
bit; the implied finite dynamic range upper-bounds every log-
ratio by a universal constant γ[25], [26]. Sub-Gaussianity
therefore follows from Hoeffding’s lemma [23]: the KL score is
a weighted sum of bounded random variables (token log-ratios).
Exponential tails let us attach non-asymptotic confidence
intervals to Z(q)even when we look at only 64 output tokens,
which is a key practical constraint for latency.
Lemma III.3 (Sub-Gaussian tail) .If each per-token log-ratio
obeyslogP(i)
rag−logP(i)
para≤γ,
Fig. 3. Distribution of KL Scores. Simulated KL divergences Z(q)
for memorised ( η≤0.1) vs grounded ( η≥0.5) queries. The clear
separation supports Assumption 1, which posits a positive gap ∆between
the two populations. RePCS exploits this gap to flag memorization with high
confidence.
thenZ(q)is sub-Gaussian with variance proxy 2γ2T.
F . Influence Gap
∆represents a fundamental asymmetry in LLM behaviour.
Adding helpful evidence typically moves log-probs by ≳0.3
nat on a non-memorised query, whereas removing evidence
from a memorised query moves them by at most η= 0.1. The
gap quantifies this separation and is therefore the “signal” that
permits detection despite stochastic generation noise.
Assumption 1 (Influence gap) .There exists ∆>0with
E[Z|clean] −E[Z|memorised] ≥∆.
G. Finite-Sample Guarantee
Theorem III.4 (Instance-level guarantee) .Letτbe the α-
quantile of the lower tail ofZcomputed from nclean
calibration queries. Under Assumption 1 and Lemma III.3,
n≥8γ2T∆−2log 
2/ϵ
ensures FPR≤αandFNR≤ϵwith probability at least 1−ϵ.
The theorem here is a consequence of Massart’s tight DKW
bound (quantile estimation error) [24] plus the sub-Gaussian tail
(score fluctuation). It states that a few hundred uncontaminated
prompts are enough to tune τonce and for all; the guarantee
is instance-wise, i.e. it holds for every future query without
batching or post-processing.
H. Uniform Control over Batches
Applying a union bound over Minstances inflates the failure
probability by a factor M; choosing δ= log Min the Chernoff
step yields the stated control. Practically, this means an entire
API endpoint serving thousands of queries per minute inherits
the same per-query false-alarm guarantee.

Corollary 1 (Uniform risk) .For any batch of Mindependent
clean queries, Pr 
max m≤MFPR m> α
≤M−1.
I. Adaptive Adversary Robustness
The sequence of KL scores forms a super-martingale under
adaptive querying. Azuma’s inequality [27] inflates the variance
by a factor at most 2, hence halving the effective gap is
sufficient to restore the original guarantee.
Theorem III.5 (Adaptive robustness) .If an attacker selects
each query after observing previous KL scores (but notlogits),
then the bound in Theorem III.4 holds with ∆replaced by
∆/2.
J. Minimax Optimality
Le Cam’s lemma [28] equates testing risk with the total vari-
ation distance between two worst-case distributions. Evaluating
the bound for the mixture model of Definition III.1 shows any
detector that uses only {Prag, Ppara}cannot beat the KL rule
by more than a constant factor without extra supervision or
model internals.
Theorem III.6 (Minimax lower bound) .Among all black-box
detectors making ≤2forward passes per query, thresholding
Z(q)atτachieves the minimax Bayes risk up to O(∆−3),
matching Le Cam’s two-point lower bound.
K. Calibration Protocol and Computational Budget
Here we collect n=500 publicly available Q&A pairs with
verified references and nooverlap with the LLM’s training
data. We then run both inference paths, compute Zfor each
query, and set τto the 5th percentile of those scores. We
continue by freezing τ; redeploying a new retriever or LLM
requires only re-estimating γand re-checking ∆. One extra
forward pass (about 1 ×latency of the baseline path) + T
floating-point multiplications and logarithms. On an NVIDIA
T4 the end-to-end overhead is 4.7%atT= 64 , which is well
below interactive-use thresholds.
IV. P ROPOSED FRAMEWORK
The RePCS routine has two lightweight stages, calibration
andinference , both designed to work with any off-the-shelf
LLM without touching its weights or gradients.
A. Calibration.
We start with a small, hand-checked set of “clean” prompts
C(a few hundred is enough). For every prompt q∈ C we run
the LLM twice: once on the bare query and once on the query
plus its top- Kretrieved passages, producing token-probability
tensors Ppara andPrag. We collapse each tensor pair into the
scalar
Z(q) = KL 
Prag∥Ppara
,
then collect the resulting scores into a vector z.Since the calibration prompts are guaranteed to have no
overlap with the model’s training data, their answers must
rely on retrieval, so the lower tail of ztells us how much
KL “wiggle room” normal grounding behaviour needs. We
set the detection threshold τto, say, the 5th percentile of z,
giving an empirical false-positive target of α= 0.05. This
one-off procedure is gradient-free, fits in GPU RAM, and runs
in minutes.
B. Inference.
At deployment time a user prompt q′follows the exact same
double run: Ppara from the bare query, Pragfrom the query
plus passages, and the score
Z(q′) = KL 
Prag∥Ppara
.
We raise a memorisation flag if Z(q′)< τ. The whole check
costs one extra forward pass and a few summed logs, roughly
a 1.05 ×latency bump on a T4, and O(V T)floating-point
additions that the LLM kernel already performs for log-prob
evaluation. No additional memory is needed beyond the two
standard activations.
C. Complexity.
RePCS adds exactly one more autoregressive decode, so
the time cost is ≈1×the baseline forward-pass FLOPs (e.g.,
an extra 35 ms for a 7-B-parameter model at 64 tokens on a
T4). The KL score itself is a streaming sum over the already-
computed token log-probs, costing O(V T)additions but no
new matrix–vector multiplies. Memory overhead stays constant:
we reuse the logits buffer and keep only two V×Tvectors
in GPU RAM (no activations from the first pass are retained
once its logits are flushed). In big-O terms, RePCS runs in
O(forward )time and O(V+T)extra space, making it cheap
enough for latency-sensitive RAG pipelines.
Algorithm 1 RePCS
•Inputs: frozen LLM M, retriever R, small clean prompt
setC(≈500items), desired false-positive rate α, top- K
passages per query.
•Calibration (run once):
–For every q∈ C:
(i) Fetch evidence R(q) =R(q).
(ii) Obtain parametric output Ppara=M(q).
(iii) Obtain retrieval-augmented output
Prag=M(⟨q, R(q)⟩).
(iv) Store score Z(q) = KL 
Prag∥Ppara
.
–Set threshold τto the α-quantile of {Z(q)}q∈C.
•Inference (per live query q′):
–Retrieve passages R(q′) =R(q′).
–Compute Ppara andPragwith two forward passes of
M.
–Score Z(q′) = KL 
Prag∥Ppara
.
–Flag memorised ifZ(q′)< τ ; otherwise
grounded .

V. E XPERIMENTS AND EVALUATIONS
A. Experimental Setup
All scripts, model checkpoints, and raw logs are public at
https://github.com/csplevuanh/repcs.
a) Dataset: We run every experiment on Prompt-WNQA
[29], a 10 k-query benchmark that targets wireless-network
reasoning. The corpus covers channel quality, interference,
routing, and hand-over events (8 k single-hop, 2 k multi-
hop). To emulate “silent” memorisation, we tag 5 k queries
as contaminated. Their answers exist verbatim in the static
knowledge graph and the remaining 5 k as clean , whose
answers require fresh telemetry injected after the LLMs’ pre-
training cut-off. These labels let us calibrate the RePCS
threshold and measure both false–positive (clean flagged) and
false–negative (contaminated missed) rates.
b) LLM back-ends and RAG pipeline: We plug three
production-grade chat models into the same retrieval-augmented
pipeline:
•CodeGen [30], an open autoregressive model fine-tuned
for multi-turn program synthesis.
•Toolformer [31], which self-trains to invoke external APIs
as tools during generation.
•InstructGPT [32], a human-aligned model trained with
reinforcement learning from human feedback.
All models are treated as frozen ; RePCS never sees gradients
or hidden states.
c) Contamination scenarios: To thoroughly evaluate
RePCS on Prompt-WNQA, we categorize queries into three
scenarios:
•Clean queries: those whose answers require up-to-date
telemetry injected into the graph, and hence cannot be
answered from the static KG alone.
•Contaminated queries: those whose answers are entirely
contained in the base Prompt-WNQA knowledge graph,
mimicking memorized pre-training content.
•Paraphrased contamination: paraphrases of the contam-
inated queries, designed to test whether RePCS can still
detect memorization when the model must match meaning
rather than surface form.
Threshold τis fitted on 500 randomly drawn clean prompts
(α= 0.05) and held fixed everywhere else.
d) Evaluation Metrics: We assess contamination detection
using:
•ROC-AUC: primary metric, measuring the model’s ability
to distinguish clean from contaminated queries.
•Precision@ k:fraction of true contaminated queries among
the top- kflagged.
•False-positive rate at 95 % true-positive rate: gauges the
likelihood of flagging fresh-data queries when maintaining
a high detection rate.
•Detection latency overhead: average additional inference
time introduced by RePCS, reported as a percentage of
end-to-end RAG latency.e) Baselines: We compare RePCS against three recent,
lightweight detectors adapted to RAG pipelines in network-
state settings—where “hallucination” specifically refers to data
memorization , i.e., the model replaying pre-trained network
facts instead of using retrieved telemetry:
•SelfCheckGPT [11] performs zero-resource, black-box
detection by sampling multiple LLM outputs for the same
network-state query and flagging overly consistent answers
as memorized content rather than retrieval-grounded
responses.
•ReDeEP [12] uses mechanistic interpretability to disen-
tangle parametric (memorized) from contextual (retrieved)
knowledge, analyzing internal activation and attention
patterns to detect when the model bypasses fresh telemetry.
•Two-Tiered Encoder Detector [13] trains lightweight
classifiers on encoder-derived representations of (query,
retrieved log snippet, generated answer) triples, identifying
outputs that conflict with up-to-date network logs as likely
memorized artifacts.
We use the authors’ public implementations and tune hyper-
parameters on the Prompt-WNQA dev split.
f) Hardware: All runs execute on a single NVIDIA T4
GPU (16 GB VRAM) in Google Colab. Dense retrieval and
embedding construction take 50 ms per query; the extra
forward pass for RePCS raises end-to-end latency by only
4.7%.
B. Evaluations
Fig. 4. Receiver–operating characteristic (ROC) on the Prompt-WNQA test
split. RePCS achieves the highest area under the curve, indicating the best
trade-off between true- and false-positive rates.
Prompt-WNQA provides three balanced query types: clean,
contaminated, and paraphrased. Their cardinalities are sum-

TABLE II
INSTANCE -LEVEL CONTAMINATION DETECTION ON PROMPT -WNQA.
Detector ROC-AUC Precision@10 FPR@95%TPR Latency (%)
SelfCheckGPT 0.792 1.00 0.676 210
ReDeEP 0.853 0.99 0.580 150
Two-Tiered Encoder 0.710 0.96 0.812 8
RePCS 0.918 1.00 0.358 4.7
TABLE III
PROMPT -WNQA DATASET COMPOSITION .
Class # Queries
Clean 5,000
Contaminated 4,000
Paraphrased 1,000
TABLE IV
REPCS ROBUSTNESS UNDER ADDITIVE GAUSSIAN LOGIT NOISE .
σ ROC-AUC
0 0.918
0.02 0.915
0.05 0.904
0.10 0.889
marised in Table III. We follow the protocol of Section V:
a single threshold τis fixed on a 500-query calibration split
and then frozen for all tests. Evaluation is performed on the
10,000-query held-out set; all detectors are run under identical
hardware and retriever settings so that accuracy and latency
are directly comparable.
Table II reports the core instance-level metrics. RePCS
attains a ROC-AUC of 0.918, outstripping the strongest baseline
(ReDeEP, 0.853) by 6.5 pp, and matches the two sampling-
based methods at perfect Precision@10. This headline result
is visualised in the ROC curves of Fig. 4: the RePCS curve
uniformly dominates the others, confirming that the KL-based
statistic captures contamination signal across all operating
points. At the operating threshold corresponding to a 95%
recall, RePCS drives the false-positive rate down to 0.358,
roughly half of ReDeEP and one-third of Two-Tiered (Fig. 5).
The confusion matrix in Fig. 7 provides a concrete breakdown:
only 248 contaminated queries are missed, while 1,786 clean
queries are mistakenly flagged. These are values that align with
the theoretical guarantees in Section III-G.
A key practical requirement for production RAG systems is
latency. As shown in Table II and Fig. 6, RePCS adds merely
4.7% end-to-end overhead on an NVIDIA T4, far below the
210% incurred by SelfCheckGPT’s multi-sampling strategy and
the 150% added by ReDeEP’s gradient probes. Two-Tiered is
lightweight but still doubles RePCS’s overhead, underscoring
that a single extra forward pass is near-optimal when tight
service-level objectives are in place.TABLE V
REPCS ACROSS DIFFERENT LLM BACK -ENDS .
LLM ROC-AUC Latency (%)
CodeGen 0.915 4.8
Toolformer 0.923 4.5
InstructGPT 0.910 4.9
Fig. 5. False-positive rate when the true-positive rate is fixed at 95% . Lower
bars indicate safer deployment margins under stringent recall requirements.
Robustness to logit perturbations is evaluated by injecting
Gaussian noise of standard deviation σ∈ {0.02,0.05,0.10}
into the softmax outputs before computing the KL score.
Table IV shows that ROC-AUC degrades gracefully from
0.918 to 0.889 as noise increases to 0.10, well within the
sub-Gaussian tolerance predicted by Lemma III.3. Finally,
Table V demonstrates that RePCS remains effective across
three very different generator back-ends (CodeGen, Toolformer,
InstructGPT); the ROC-AUC fluctuates by less than one
percentage point and latency stays below 5%, validating the
model-agnostic claim of Section III-C.
VI. C ONCLUSION
We have so far introduced RePCS , a training-free, black-
box detector that diagnoses data memorisation in retrieval-
augmented generation by running the same LLM twice, once
with retrieved passages and once without, and measuring

Fig. 6. Extra end-to-end inference latency (percentage of baseline RAG time)
incurred by each detector on a single NVIDIA T4 GPU. RePCS adds only
4.7%, well below interactive-use thresholds.
Fig. 7. Confusion matrix for RePCS at the decision threshold that yields a
95% true-positive rate. Most contaminated queries are correctly flagged while
the clean-query false-alarm count remains moderate.
the Kullback–Leibler divergence between the two output
distributions. This single-scalar test achieves a ROC–AUC
of0.918 on the Prompt-WNQA benchmark, surpassing the
strongest prior method by 6.5 pp while adding only 4.7 % end-
to-end latency on an NVIDIA T4 GPU. The score degrades
gracefully under substantial logit noise (down to 0.889 at
σ= 0.10) and remains stable across diverse LLM back-ends
such as CodeGen, Toolformer, and InstructGPT. These results
confirm the detector’s robustness and model-agnostic nature.
Beyond strong empirical results, RePCS comes with PAC-
style guarantees that tie its KL threshold to user-specified
false-positive and false-negative rates, and we show it isminimax-near-optimal among detectors limited to two forward
passes. These traits make RePCS a practical safeguard for
latency-sensitive, safety-critical RAG deployments where model
internals are inaccessible. Future research will extend the
approach to multi-modal retrieval, develop online calibration
under concept drift, and assemble broadened cross-domain
benchmarks to drive progress on contamination-aware evalua-
tion.
REFERENCES
[1]J. Li, Y . Gao, Y . Yang, Y . Bai, X. Zhou, Y . Li, H. Sun, Y . Liu, X. Si, Y . Ye,
Y . Wu, Y . Lin, B. Xu, B. Ren, C. Feng, and H. Huang, “Fundamental
capabilities and applications of large language models: A survey,” ACM
Comput. Surv. , May 2025. Just Accepted.
[2]Z. Li, J. Zhang, C. Yan, K. Das, S. Kumar, M. Kantarcioglu, and B. A.
Malin, “Do you know what you are talking about? characterizing query-
knowledge relevance for reliable retrieval augmented generation,” in Proc.
Conf. Empirical Methods in Natural Language Processing (EMNLP) ,
(Miami, FL, USA), pp. 6130–6151, Nov 2024.
[3]O. Ayala and P. Bechard, “Reducing hallucination in structured outputs via
retrieval-augmented generation,” in Proc. Conf. North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies, Industry Track (NAACL-HLT) , (Mexico City, Mexico),
pp. 228–238, Jun 2024.
[4]X. Huang, Y . Tang, J. Li, N. Zhang, and X. Shen, “Toward effective
retrieval augmented generative services in 6g networks,” IEEE Network ,
vol. 38, no. 6, pp. 459–467, 2024.
[5]G. M. Yilma, J. A. Ayala-Romero, A. Garcia-Saavedra, and X. Costa-
Perez, “Telecomrag: Taming telecom standards with retrieval augmented
generation and llms,” SIGCOMM Comput. Commun. Rev. , vol. 54, pp. 18–
23, Jan 2025.
[6]X. Wang, Z. Wang, X. Gao, F. Zhang, Y . Wu, Z. Xu, T. Shi, Z. Wang,
S. Li, Q. Qian, R. Yin, C. Lv, X. Zheng, and X. Huang, “Searching
for best practices in retrieval-augmented generation,” in Proc. Conf.
Empirical Methods in Natural Language Processing (EMNLP) , (Miami,
FL, USA), pp. 17716–17736, Nov 2024.
[7]T. Formal, S. Clinchant, H. D ´ejean, and C. Lassance, “Splate: Sparse
late interaction retrieval,” in Proc. 47th Int. ACM SIGIR Conf. Research
and Development in Information Retrieval (SIGIR) , (Washington DC,
USA), pp. 2635–2640, 2024.
[8]K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia,
“Colbertv2: Effective and efficient retrieval via lightweight late interaction,”
inProc. Conf. North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies (NAACL-HLT) ,
(Seattle, WA, USA), pp. 3715–3734, Jul 2022.
[9]A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” in Proc. Int.
Conf. Learn. Representations (ICLR) , 2024. Oral Presentation.
[10] F. Yao, Y . Zhuang, Z. Sun, S. Xu, A. Kumar, and J. Shang, “Data
contamination can cross language barriers,” in Proc. Conf. Empirical
Methods in Natural Language Processing (EMNLP) , (Miami, FL, USA),
pp. 17864–17875, Nov 2024.
[11] P. Manakul, A. Liusie, and M. Gales, “SelfCheckGPT: Zero-resource
black-box hallucination detection for generative large language models,”
inProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing , 2023.
[12] Z. Sun, X. Zang, K. Zheng, J. Xu, X. Zhang, W. Yu, Y . Song, and H. Li,
“ReDeEP: Detecting hallucination in retrieval-augmented generation
via mechanistic interpretability,” in Proceedings of the Thirteenth
International Conference on Learning Representations , 2025.
[13] I. Zimmerman, J. Tredup, E. Selfridge, and J. Bradley, “Two-tiered
encoder-based hallucination detection for retrieval-augmented generation
in the wild,” in Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing: Industry Track , (Miami, Florida, USA),
pp. 8–22, Association for Computational Linguistics, Nov. 2024.
[14] G. Sriramanan, S. Bharti, V . S. Sadasivan, S. Saha, P. Kattakinda, and
S. Feizi, “Llm-check: Investigating detection of hallucinations in large
language models,” in Advances in Neural Information Processing Systems
37 (NeurIPS) , 2024. Poster Presentation.

[15] M. Belyi, R. Friel, S. Shao, and A. Sanyal, “Luna: A lightweight
evaluation model to catch language model hallucinations with high
accuracy and low cost,” in Proc. 31st Int. Conf. Comput. Linguistics:
Industry Track (COLING) , (Abu Dhabi, UAE), pp. 398–409, Jan 2025.
[16] X. Hu, D. Ru, L. Qiu, Q. Guo, T. Zhang, Y . Xu, Y . Luo, P. Liu, Y . Zhang,
and Z. Zhang, “Knowledge-centric hallucination detection,” in Proc. Conf.
Empirical Methods in Natural Language Processing (EMNLP) , (Miami,
FL, USA), pp. 6953–6975, Nov 2024.
[17] C. Niu, Y . Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and
T. Zhang, “RAGTruth: A hallucination corpus for developing trustworthy
retrieval-augmented language models,” in Proc. 62nd Annu. Meeting
Assoc. Comput. Linguistics (ACL) , (Bangkok, Thailand), pp. 10862–
10878, Aug 2024.
[18] J. Song, X. Wang, J. Zhu, Y . Wu, X. Cheng, R. Zhong, and C. Niu,
“RAG-HAT: A hallucination-aware tuning pipeline for LLM in retrieval-
augmented generation,” in Proc. Conf. Empirical Methods in Natural
Language Processing: Industry Track , (Miami, FL, USA), pp. 1548–1558,
Nov 2024.
[19] L. Chen, R. Zhang, J. Guo, Y . Fan, and X. Cheng, “Controlling risk of
retrieval-augmented generation: A counterfactual prompting framework,”
inFindings Assoc. Comput. Linguistics: EMNLP , (Miami, FL, USA),
pp. 2380–2393, Nov 2024.
[20] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang, “Realm: Retrieval-
augmented language model pre-training,” in Proceedings of the 37th
International Conference on Machine Learning (ICML) , pp. 368–377,
JMLR.org, 2020.
[21] G. Izacard and E. Grave, “Leveraging passage retrieval with generative
models for open domain question answering,” in Proc. 16th Conf.
European Chapter of the ACL (EACL) , (Online), pp. 874–880, Association
for Computational Linguistics, 2021.
[22] I. Csisz ´ar, “Information-type measures of difference of probability
distributions and indirect observations,” Studia Sci. Math. Hungarica ,
vol. 2, pp. 299–318, 1967.
[23] W. Hoeffding, “Probability inequalities for sums of bounded random
variables,” Journal of the American Statistical Association , vol. 58,
no. 301, pp. 13–30, 1963.
[24] P. Massart, “The tight constant in the dvoretzky–kiefer–wolfowitz
inequality,” Annals of Probability , vol. 18, no. 3, pp. 1269–1283, 1990.
[25] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer, “8-bit optimizers
via block-wise quantization,” in Proc. Int. Conf. Learn. Representations
(ICLR) , 2022.
[26] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “Optq: Accurate
quantization for generative pre-trained transformers,” in Proc. Int. Conf.
Learn. Representations (ICLR) , 2023.
[27] K. Azuma, “Weighted sums of certain dependent random variables,”
Tˆohoku Math. J. , vol. 19, no. 3, pp. 357–367, 1967.
[28] L. L. Cam, “Convergence of estimates under dimensionality restrictions,”
Annals of Statistics , vol. 1, no. 1, pp. 38–53, 1973.
[29] P. Liu, B. Qian, Q. Sun, and L. Zhao, “Prompt-WNQA: A prompt-based
complex question answering for wireless network over knowledge graph,”
Computer Networks , vol. 236, p. 110014, 2023.
[30] E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y . Zhou, S. Savarese,
and C. Xiong, “Codegen: An open large language model for code
with multi-turn program synthesis,” in Proceedings of the Eleventh
International Conference on Learning Representations (ICLR) , 2023.
[31] T. Schick, J. Dwivedi-Yu, R. Dess `ı, R. Raileanu, M. Lomeli, E. Hambro,
L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language
models can teach themselves to use tools,” in Advances in Neural
Information Processing Systems 36 (NeurIPS) , 2023.
[32] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin,
C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton,
L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike,
and R. Lowe, “Training language models to follow instructions with
human feedback,” in Advances in Neural Information Processing Systems
35 (NeurIPS) , pp. 27730–27744, 2022.
APPENDIX A
PROOF OF THEOREM III.2
We wish to upper-bound Z(q) = KL PragPpara
when Prag= (1−η)Ppara+ηQwithη≤1
2.
If the retrieved evidence contributes only a 10% “weight”
(η= 0.1), then every log-odds term moves by at most 10%;the overall KL—a quadratic quantity—should therefore shrink
by roughly η2.
Detailed derivation. We start by defining
f(η) = KL (1 −η)Ppara+ηQP para.
We invoke Csisz ´ar’sf-divergence calculus [22], which shows
that
f′′(η) = χ2 
Q∥Ppara
≤1
(1−η)2,
where
χ2(Q∥Ppara) =ZQ
Ppara−12
Ppara
is the (non-negative) χ2-divergence.
Because the bound on f′′(η)holds for allη∈[0,1
2], we
may integrate it twice:
f(η) =f(0) + f′(0)η+1
2f′′(ξ)η2
≤η2
2(1−η), ξ∈(0, η).
(The terms f(0) = 0 andf′(0) = 0 because the two
arguments of KL coincide at η= 0.) This establishes the
advertised quadratic upper bound.
When η= 0.1the bound evaluates to 5×10−3nat—less
than the quantisation error introduced by 16-bit floating-point
arithmetic. Hence any empirical KL below this level can be
regarded as “numerically zero,” signalling memorisation with
high confidence.
APPENDIX B
PROOF OF LEMMA III.3
Here we need to show that Z(q)concentrates sharply around
its mean whenever each token-level log-ratio is bounded by γ.
Proof. We rewrite the KL score as a weighted sum:
Z=TX
i=1P(i)
ragXi, X i:= log 
P(i)
rag
P(i)
para!
.
Since Xi∈[−γ, γ], Hoeffding’s lemma gives E
eλX i
≤
exp 
λ2γ2/8
for all λ∈R; i.e. Xiis sub-Gaussian with
variance proxy γ2/2.
We treat {Xi}as independent conditioned on q. Bernstein’s
sub-Gaussian preservation property then yields
Zis sub-Gaussian with variance proxy σ2= 2γ2TX
i=1 
P(i)
rag2.
SinceP
iP(i)
rag= 1, Cauchy–Schwarz implies
X
i 
P(i)
rag2≤1,
so
σ2≤2γ2T.

Sub-Gaussianity guarantees that Pr(|Z−EZ|> t)≤
2e−t2/(4γ2T).
Since even a single query gives a meaningful deviation bound
when T≤64, eliminating the need for large-sample normal
approximations.
APPENDIX C
PROOF OF THEOREM III.4
Bound the calibration sample size nrequired so that the
threshold τcontrols both FPR and FNR with high probability.
Proof. LetFclean be the cumulative distribution function of Z
on clean queries and let τ⋆be its (1−α)-quantile.
By the Dvoretzky–Kiefer–Wolfowitz–Massart (DKWM)
inequality [24],
Pr 
|τ−τ⋆|> δ
≤2e−2nδ2.
Lemma III.3 gives, for any t >0,
Pr
clean(Z <EZ−t)≤e−t2/(4γ2T),
Pr
mem(Z >EZ+t)≤e−t2/(4γ2T).
We continue to set t= ∆/2and choose δ= ∆/4. Provided
n≥8γ2T∆−2log(2/ϵ),both the DKWM error and the tail
probability are bounded by ϵ/2. Union-bounding yields FPR≤
αandFNR≤ϵwith probability 1−ϵ.
We pay two “error budgets”: one for estimating the (1−α)
quantile, one for stochastic fluctuation of Z. Both budgets
scale as e−cn∆2; hence n∝∆−2is information-theoretically
optimal.
APPENDIX D
UNIFORM AND ADAPTIVE EXTENSIONS
Proof of Cor. 1. For a batch of Mindependent clean queries,
Pr
max
m≤MFPR m> α
= PrM[
m=1{Zm< τ}
≤MX
m=1Pr(Zm< τ)≤Mα.
Taking Mα≤1gives the stated bound.
Proof of Thm. III.5. LetFtbe the filtration generated by the
firstt−1KL scores.
The conditional expectation E[Zt| Ft]remains sub-Gaussian
with the same proxy variance, but Azuma’s inequality [27]
now incurs an additional factor of two in the exponent.
Halving ∆restores the original deviation bound.
This completes the argument.APPENDIX E
MINIMAX OPTIMALITY PROOF
Proof of Theorem III.6. We consider the composite hypothesis
testH0:memorised versus H1:clean.
Le Cam’s two-point method [28] constructs two priors, one
supported on memorised queries with mean KL E0Zand one
on clean queries with mean E1Z=E0Z+ ∆.
The total variation distance between these priors is upper-
bounded by1
2∆3+o(∆3).
Consequently, any detector based on twoindependent sam-
ples cannot achieve Bayes risk below Ω(∆3)[28].
The KL thresholding rule attains risk O(∆3)(proof in main
text).
This follows for minimax-rate optimal.
As a note, this theorem certifies that no alternative black-box
detector—no matter how cleverly engineered—can asymptoti-
cally beat the KL rule unless it makes additional model calls
or injects external supervision.
# CORTEX: Token-Level Hallucination Detection in RAG via Comparative Internal Representations

**Authors**: Kazuaki Furumai, Shuichiro Haruta, Kazunori Matsumoto, Daisuke Kamisaka

**Published**: 2026-06-30 02:04:24

**PDF URL**: [https://arxiv.org/pdf/2606.31033v1](https://arxiv.org/pdf/2606.31033v1)

## Abstract
In this paper, we propose CORTEX, a token-level hallucination detection method for Retrieval-Augmented Generation (RAG). In long-form RAG outputs, hallucinations often arise in localized spans rather than throughout an entire response. CORTEX therefore identifies ungrounded content at the token level, enabling fine-grained localization of hallucinations. The key intuition behind CORTEX is that tokens grounded in retrieved documents should be more strongly influenced by those documents than hallucinated tokens. To capture this document-induced effect, CORTEX compares internal representations of a large language model (LLM) under two conditions: with and without the retrieved documents. Instead of relying solely on each token's immediate sensitivity to the retrieved documents, CORTEX also leverages the propagation of document-grounded information through preceding tokens, reducing false positives for tokens whose evidence has already been absorbed into the context. Finally, CORTEX applies post-processing smoothing step that models the tendency of hallucination labels to persist over contiguous spans, reducing local noise and encouraging span-consistent predictions. Experiments on two RAG benchmarks and three LLMs show that CORTEX substantially improves token-level hallucination detection, with each component consistently contributing to performance gains.

## Full Text


<!-- PDF content starts -->

CORTEX: Token-Level Hallucination Detection in RAG via Comparative
Internal Representations
Kazuaki Furumai Shuichiro Haruta Kazunori Matsumoto Daisuke Kamisaka
KDDI Research, Inc.
{ka-furumai, sh-haruta, da-kamisaka}@kddi.com
Abstract
In this paper, we propose CORTEX, a
token-level hallucination detection method for
Retrieval-Augmented Generation (RAG). In
long-form RAG outputs, hallucinations often
arise in localized spans rather than throughout
an entire response. CORTEX therefore iden-
tifies ungrounded content at the token level,
enabling fine-grained localization of hallucina-
tions. The key intuition behind CORTEX is
that tokens grounded in retrieved documents
should be more strongly influenced by those
documents than hallucinated tokens. To cap-
ture this document-induced effect, CORTEX
compares internal representations of a large
language model (LLM) under two conditions:
with and without the retrieved documents. In-
stead of relying solely on each token’s imme-
diate sensitivity to the retrieved documents,
CORTEX also leverages the propagation of
document-grounded information through pre-
ceding tokens, reducing false positives for to-
kens whose evidence has already been absorbed
into the context. Finally, CORTEX applies
post-processing smoothing step that models
the tendency of hallucination labels to persist
over contiguous spans, reducing local noise and
encouraging span-consistent predictions. Ex-
periments on two RAG benchmarks and three
LLMs show that CORTEX substantially im-
proves token-level hallucination detection, with
each component consistently contributing to
performance gains.
1 Introduction
Retrieval-Augmented Generation (RAG) has been
widely adopted to improve the factuality of large
language models (LLMs) by grounding generation
in external references (Lewis et al., 2020). Al-
though RAG mitigates hallucinations to some ex-
tent, LLMs may still generate groundless or in-
consistent content even when references are pro-
vided (Fan et al., 2026). Accurately detecting such
hallucinations within generated outputs thereforeremains a critical challenge (Niu et al., 2024; Liu
et al., 2025; Bang et al., 2025).
Various hallucination detection methods have
been proposed to address this challenge. Self-
consistency-based approaches estimate reliability
by measuring agreement across multiple outputs
from the same prompt (Manakul et al., 2023).
Prompt-based methods use LLMs as fact-checkers
to explicitly verify generated content (Zheng et al.,
2023; Es et al., 2024; Furumai et al., 2024).
While these approaches leverage LLMs’ strong
language understanding capabilities, they may in-
cur high computational cost due to repeated sam-
pling, and their reliability can depend on prompt
design (Ganesh et al., 2026).
More recently, hallucination detection methods
based on the internal representations of LLMs have
been increasingly studied (Sriramanan et al., 2024;
Zhang et al., 2025; Xiong et al., 2026). These meth-
ods seek hallucination-related signals in model ac-
tivations and can offer insight into the internal con-
ditions under which hallucinations arise. This line
of work has analyzed attention and feedforward-
network states, quantified component contributions
to generation, and extracted hallucination-related
features from internal representations.
However, many existing approaches are not de-
signed specifically for RAG and primarily assess
the generated output as a whole, i.e., at the answer
level. This is limiting for long-form RAG outputs,
where faithful and hallucinated content often coex-
ist. For practical use, detecting hallucinations in
such mixed outputs requires a finer-grained formu-
lation than answer-level detection. The appropriate
granularity can vary across tasks and annotation
protocols, from words and phrases to sentences
or paragraphs. Given this variability, token-level
detection can serve as a practical approach for lo-
calizing ungrounded content and later adapting pre-
dictions to the required granularity.
In this paper, we proposeCORTEX
1arXiv:2606.31033v1  [cs.CL]  30 Jun 2026

❶ Answer Gener ation ❷ Paired Inputs ❸ Feature Construction ❹ Hallucination Detection
Original Pr ompt
Closed-weight LLM
Gener ated Answer
A black hole is a region in space
where gravity is so strong that nothing,
not even light, can escape. Black
holes are powered by a hidden core of
dark energy that constantly generates
new matter inside them. The boundary
called the event horizon, and scientists
detect black holes by observing their
gravitational eﬀects on nearby stars
and gas.Open-weight
LLM...
...(a) Internal State Comparison
(b) Contextual Residual of ΔhMLPToken-level Classiﬁer
(c) Smoothin g
ResultToken-Level Scores
...
...
✖ ✖ ✖ ✖Please answer the question based on
the following passages.
Question: What is a black hole?
Passage 1:
A black hole is a region in space where
gravity is so strong that ...
Passage 2:
The boundary surrounding a black hole
is ...
A black hole is a region in space where
gravity is so strong that nothing, not
even light, can escape. Black holes are
powered by a hidden core of dark
energy that constantly generates new
matter inside them. The boundary
called the event horizon, and scientists
detect black holes by observing their
gravitational eﬀects on nearby stars
and gas....
...Ref Input
No-ref Input0.26 0.16 0.84 0.40 0.99 0.52 0.12
0.10 0.02 0.95 0.90 0.99 0.22 0.02
...Figure 1: Overview of CORTEX for token-level hallucination detection in RAG, using reference-conditioned
internal representation comparisons, contextual residual features, and label-persistence smoothing.
(ComparativeObservedReference-based
Token-levelEXpressions), a post-hoc token-level
hallucination detection method for RAG. Figure 1
illustrates the overall framework. CORTEX
builds on the intuition that faithful tokens should
exhibit coherent representational changes when
references are provided, whereas hallucinated
tokens should show weaker or less coherent
changes. To measure this effect, CORTEX
constructs a paired counterfactual view of the
same answer under reference-conditioned and
no-reference inputs, rather than probing a single
input as in conventional representation-based
detectors. The resulting reference-induced changes
are then encoded as token-level delta features for
hallucination detection.
CORTEX further introduces an attention-based
contextual residual feature. In long-form genera-
tion, reference influence may propagate indirectly
through preceding answer tokens, as in reasoning-
style or self-referential generation. CORTEX cap-
tures this context-mediated reference influence by
combining token-level delta features with attention
patterns, helping the classifier distinguish tokens
that are indirectly grounded in the reference from
groundless tokens and thereby reducing false posi-
tives.
Finally, CORTEX uses post-processing smooth-
ing step to adjust the granularity of token-level
scores. Raw token-level predictions can be sensi-
tive to local noise and may produce isolated high-
score tokens, whereas human hallucination annota-
tions often appear as contiguous spans. We there-
fore introducelabel-persistence smoothing, which
treats raw scores as token-level confidence andmodels the tendency of hallucination labels to per-
sist across neighboring tokens. This reduces scat-
tered noise and adapts token-level predictions to
span-level annotation structure while still retaining
token-level scores.
Experiments on two RAG benchmarks and three
LLMs demonstrate that CORTEX substantially im-
proves token-level hallucination detection, with
all components contributing consistently to per-
formance gains.
Our contributions are summarized as follows:
•We propose CORTEX, which, to the best of
our knowledge, is the first token-level halluci-
nation detection method specifically designed
for RAG.
•CORTEX is a practical post-hoc framework
that is easy to implement and can detect hal-
lucinations in outputs from arbitrary LLMs,
including API-based closed-weight models.
2 Preliminaries
We consider a practical setting in which the answer
is produced by a closed-weight LLM whose param-
eters and internal representations are not accessible,
as is the case for many API-based models. Such
models are often attractive in deployed applications
because of their strong generation quality. We de-
note this closed-weight LLM by Mclose. Given a
question qtext, the closed-weight LLM produces
an answer as
atext=M close(qtext).(1)
Our goal is to identify hallucinated tokens in the
generated answer atext. Although a direct way to
2

obtain hallucination-related features would be to
analyze the internal representations of the closed-
weight LLM, we cannot do so for the aforemen-
tioned reason. We therefore use a separate open-
weight LLM, denoted by Mopen, as a post-hoc anal-
ysis model. Unlike Mclose,Mopen exposes internal
representations, which allows us to extract features
from the answer in relation to the question.
LetTok(·) denote the tokenizer of Mopen. We
tokenize the answer as
a= Tok(atext) = (t 1, t2, . . . , t i, . . . , t T), (2)
where Tdenotes the number of answer tokens and
tidenotes the i-th answer token under the tokenizer
ofMopen. We assume that Mopen processes an
input containing the question and the answer as
Mopen(qtext∥atext), where ∥denotes text concate-
nation with the appropriate prompt format. Mopen
produces token-level internal representations over
the entire input sequence, which capture the rela-
tionships between qtextandatext. Since hallucina-
tion detection is performed over the answer, we use
only the internal representations corresponding to
the answer tokens. We denote the representation
aligned with answer token tibyhi∈Rd. The
specific input construction and the extraction of
answer-token representations are described in Sec-
tion 3.
The token-level hallucination detection task is to
estimate whether each answer token is hallucinated.
Letyi∈ {0,1} denote the token-level label, where
yi= 1 indicates that token tiis hallucinated and
yi= 0indicates that it is faithful.
In our setting, we focus on RAG applications,
where references are provided as grounding evi-
dence for answers. This setting reflects common
practical deployments in which retrieved docu-
ments are used to reduce hallucination and the an-
swer is expected to be faithful to those references.
3 CORTEX
We propose CORTEX (ComparativeObserved
Reference-basedToken-levelEXpressions), a post-
hoc hallucination detection framework for RAG.
CORTEX is built on three key ideas, illustrated
in Figure 1(a)–(c): (a) a reference-induced delta
representation for capturing how each answer to-
ken’s internal representation changes when the ref-
erences are provided; (b) an attention-based con-
textual residual for distinguishing direct reference
sensitivity from reference influence mediated bypreceding answer tokens; and (c) label-persistence
smoothing for reducing isolated token-level noise
and obtaining span-consistent hallucination scores
while preserving token-level predictions.
3.1 Reference-Induced Delta Representation
CORTEX constructs two conditioned inputs for
the open-weight LLM, differing only in whether
the references are included. Let rtextdenote the
references, i.e., the retrieved documents used as
grounding evidence in the RAG setting. The
reference-conditioned input is defined as xtext
ref=
qtext∥rtext∥atext, whereas the no-reference input
is defined as xtext
no-ref=qtext∥atext. With this con-
struction, the answer span corresponds to the same
token sequence (t1, . . . , t T)in both conditions, al-
lowing CORTEX to compare internal representa-
tions aligned with the same answer tokens.
When Mopen processes xtext
refandxtext
no-ref, it
produces token-level internal representations over
the entire input sequence. For an input x=
Tok(xtext), letRepMopen(x)denote the sequence
of internal representations obtained from Mopen.
We define an extraction function fthat selects the
representations aligned with the answer tokens:
(href
1, . . . , href
T) =f(RepMopen(xref)),
(hno-ref
1, . . . , hno-ref
T) =f(RepMopen(xno-ref)).
(3)
Here, href
i, hno-ref
i∈Rddenote the internal rep-
resentations aligned with the same answer token ti
under the reference-conditioned and no-reference
conditions, respectively, and ddenotes the repre-
sentation dimensionality.1
CORTEX builds on the intuition that faithful
tokens should exhibit coherent representational
changes when references are provided, whereas
hallucinated tokens should show weaker or less co-
herent changes. Based on this intuition, CORTEX
computes the reference-induced delta representa-
tion for each answer tokent ias
∆hi=href
i−hno-ref
i.(4)
This vector is the core representation in COR-
TEX. Since the same answer is included in both in-
puts,∆hicaptures how the representation of token
tichanges when references are provided. In other
words, ∆hiis intended to emphasize the reference-
induced change in how Mopen contextualizes that
1Specifically, we use the final transformer layer output
corresponding to each answer token as its token-level repre-
sentationh i.
3

token, rather than merely representing the token
content itself.
3.2 Attention-Based Contextual Residual
The delta representation ∆hicaptures how the rep-
resentation of token tichanges when references are
added. However, in reasoning-style outputs such as
chain-of-thought, facts grounded in the references
may first be stated in earlier parts of the answer,
and later tokens may continue the reasoning based
on those facts. In this case, the influence of the
references can reach tinot only through the direct
pathr→t i, but also through the context-mediated
pathr→t <i→ti. As a result, even tokens that
are indirectly grounded by the references may ap-
pear to have a weak relationship with the references
if we only use∆h i.
To address this issue, we introduce a feature
that represents how much reference influence is
contained in the preceding tokens that the current
tokent irelies on as
¯∆hi=X
j<iαref
ij∆hj,(5)
where αref
ij∈[0,1] denotes the attention weight
from tito a preceding answer token tjunder the
reference-conditioned input. We further subtract
this context-mediated influence from the current
token’s own change and define the contextual resid-
ual as
ci= ∆h i−¯∆hi.(6)
This residual represents the token-specific change
that remains after removing the reference influence
explainable through preceding tokens. We use both
∆hiandc ifor token-level hallucination detection.
3.3 Token-Level Hallucination Detection
For each answer token ti, CORTEX predicts a
raw token-level hallucination score siby feeding
[∆hi;ci]into a multilayer perceptron (MLP) clas-
sifierg θ:
si=σ(g θ([∆h i;ci])), s i∈[0,1],(7)
where σdenotes the sigmoid function. The classi-
fier is trained with binary cross-entropy loss using
token-level hallucination labels. To address class
imbalance, the positive class is weighted according
to the ratio of negative to positive tokens in the
training set.3.4 Label-Persistence Smoothing
We define a span as a contiguous sequence of to-
kens annotated as a single hallucination unit. Hu-
man hallucination annotations are typically span-
based: once a token is marked as hallucinated,
neighboring tokens in the same span are likely to
receive the same label. Based on this property,
we applylabel-persistence smoothingas a post-
processing step, rather than using moving-average
smoothing.
Given the raw token-level scores s1:T, we intro-
duce an unobserved binary smoothing-label vari-
ablezi∈ {0,1} to represent the span-consistent
label underlying the post-processed score, where
zi= 0denotes a faithful token and zi= 1denotes
a hallucinated token. To obtain span-consistent
post-processed scores, we model the smoothed
label sequence by combining two components
around positions iandi+ 1 : the tendency of neigh-
boring labels to persist and information from the
raw scores at each position.
We first model span-level label persistence by
defining the pairwise persistence term between
neighboring labels as
ρ(zi, zi+1) =(
pstay, z i+1=zi,
1−p stay, z i+1̸=zi.(8)
The parameter pstay∈[0,1] controls the degree
of label persistence: smaller values allow finer-
grained label changes, whereas larger values favor
longer same-label spans. This provides a delib-
erately simple approximation to span-level label
continuity, rather than modeling the full variability
of human annotation patterns. We then define the
token-level confidence induced by the raw score si
at each position:
ϕi(zi= 1) =s i, ϕ i(zi= 0) = 1−s i.(9)
This quantity represents how the raw score at posi-
tioniis converted into token-level confidence for
each value of the smoothing-label variable.2
Combining Eq. (8)and Eq. (9), we define the
local compatibility term for neighboring positions
iandi+ 1as
ϕi(zi)ρ(z i, zi+1)ϕi+1(zi+1).(10)
2In implementation, we apply a small amount of clipping
to the raw scores so that the token-level confidence does not
become exactly 0 or 1.
4

This term measures the compatibility of the neigh-
boring label assignment (zi, zi+1)with both the
raw scores and the label-persistence assumption.
We therefore define the normalized distribution
over sequence labels as
P(z 1:T|s1:T) =1
Z(s 1:T)π(z1)TY
i=1ϕi(zi)
×T−1Y
i=1ρ(zi, zi+1),(11)
where Z(s 1:T)is the normalizing constant and
π(z1)is the initial distribution over the first smooth-
ing label. We use a uniform initial distribution,
π(0) =π(1) = 1/2.
Although z1:Tis unobserved, the desired token-
level smoothed score can be obtained by marginal-
izing over all smoothing-label sequences:
˜si=P(z i= 1|s 1:T).(12)
This posterior can be computed efficiently us-
ing the forward–backward algorithm (Rabiner,
1990). We provide the full recursions and posterior
marginal formula in Appendix A.
4 Experiments
We evaluate CORTEX on two publicly available
RAG hallucination benchmarks, RAGTruth (Niu
et al., 2024) and HalluRAG (Ridder and
Schilling, 2024), using three LLMs: Llama-3.1-8B-
Instruct (Grattafiori et al., 2024), Qwen3-8B (Yang
et al., 2025), and Mistral-7B-Instruct-v0.2 (Mistral
AI, 2023). We use these open-weight LLMs to ob-
tain internal representations for CORTEX and the
relevant baselines. Evaluation is performed at the
token and answer levels. Token-level evaluation as-
sesses whether a method can identify hallucinated
tokens in a generated answer, while answer-level
evaluation assesses whether the answer contains
any hallucinated content. For CORTEX, answer-
level evaluation is performed by aggregating token-
level scores, as it is trained for token-level detection
and does not use answer-level supervision. Specifi-
cally, we use the maximum token-level hallucina-
tion score as the answer-level score.
For label-persistence smoothing, we use pstay=
0.993 at the token level and pstay= 0.930 at the
answer level. We provide a sensitivity analysis
ofpstayin Appendix B. Additional details on the
experimental setup are provided in Appendix C.
The following subsections describe the baselines
and datasets used in our experiments.4.1 Baselines
We compare CORTEX with five baselines.NLL
uses negative log-likelihood as an uncertainty-
based signal, following prior work on uncertainty
estimation from next-token predictive distribu-
tions (Malinin and Gales, 2021).SAPLMAtrains a
probe on intermediate transformer representations,
based on the observation that hidden states encode
hallucination-related signals (Azaria and Mitchell,
2023).LLM-Checkextracts features from inter-
nal states produced by attention mechanisms and
feedforward neural networks, including spectral
properties such as eigenvalues (Sriramanan et al.,
2024).ICR Probeuses internal component at-
tribution to quantify the contribution of attention
and feedforward networks and identify hallucinated
content (Zhang et al., 2025).RAGLensapplies
sparse autoencoders to token embeddings and se-
lects hallucination-related sparse features via token
aggregation and mutual-information-based feature
selection (Xiong et al., 2026).
For answer-level evaluation, we follow the orig-
inal implementation of baselines. However, their
answer-level implementation cannot be directly
used for token-level evaluation. We modify parts
of the baseline implementations to adapt them to
token-level detection while preserving their origi-
nal mechanisms as much as possible.
4.2 Datasets
Datasets for hallucination detection in RAG set-
tings remain limited, especially those that include
both references and fine-grained hallucination an-
notations. To the best of our knowledge, RAGTruth
is the only publicly available RAG benchmark with
human-annotated hallucination spans that are suf-
ficiently fine-grained to support token-level evalu-
ation. Although RAGTruth contains multiple task
types, we use only its QA subset in this work.
HalluRAG consists of answers generated by mul-
tiple LLMs under two RAG prompt settings using
either relevant or irrelevant Wikipedia chunks, with
sentence-level hallucination annotations by GPT-
4o (OpenAI et al., 2024). Since token-level la-
bels are unavailable, we derive token-level pseudo-
labels by marking all tokens in hallucinated sen-
tences as hallucinated.
The two datasets provide complementary set-
tings, differing in annotation granularity, labeling
procedure, retrieval quality, and QA distribution.
RAGTruth provides fine-grained human span anno-
5

Statistic RAGTruth HalluRAG
# Samples 5,934 2,243
Train samples 4,530 1,662
Validation samples 504 185
Test samples 900 396
Avg. prompt len. 1,660.0 1,778.9
Avg. answer len. 686.5 164.0
Table 1: Dataset statistics.
tations, whereas HalluRAG provides sentence-level
GPT-4o annotations and includes cases where refer-
ences may be irrelevant to the question. They also
differ in construction and topical coverage, with
RAGTruth’s QA subset is based on daily-life ques-
tions, whereas HalluRAG derives questions from
Wikipedia passages. Together, these differences
allow us to assess robustness across conditions.
Both datasets contain answers generated by vari-
ous LLMs, together with the references provided
for generating those answers. We treat these an-
swers as outputs from closed-weight LLMs, regard-
less of which model generated them. To ensure
a fair comparison, all methods receive the same
answer and corresponding references as input.
4.3 Results
Token-level Detection
Table 2 reports token-level hallucination detection
results in terms of average precision (AP) and the
area under the receiver operating characteristic
curve (AUROC). Across both RAGTruth and Hal-
luRAG, CORTEX achieves the best performance in
all settings. The gains are particularly substantial
on RAGTruth, where human fine-grained annota-
tions are available, suggesting that CORTEX is
well aligned with token-level hallucination detec-
tion. Even on HalluRAG, where token-level labels
are derived from sentence-level annotations, COR-
TEX remains the best-performing method, indicat-
ing robustness to coarser and noisier supervision.
All baselines receive inputs with references, but
rely on a single reference-conditioned view of
model representations. SAPLMA directly uses
transformer layer outputs as features, while RA-
GLens extracts hallucination-related sparse fea-
tures from token embeddings using a trained sparse
encoder. In contrast, CORTEX derives its signal
from the contrast between paired representations of
the same tokens with and without references. The
consistent gains suggest that this comparative for-
mulation captures reference-induced hallucination
signals that are not readily available from a singleinternal representation or features derived from it.
Label-persistence smoothing is a post-
processing module applicable to any token-level
predictions. In Appendix D, we further apply it to
baselines to demonstrate its general effectiveness.
Answer-level Detection
Table 3 reports the answer-level hallucination detec-
tion results in terms of AP and AUROC. Unlike the
baselines, which are trained directly for answer-
level detection, CORTEX does not use answer-
level supervision. Instead, CORTEX reuses the
token-level hallucination scores and assigns each
answer the maximum score among its tokens. Thus,
this setting evaluates whether the token-level sig-
nals captured by CORTEX can also indicate hallu-
cination at the answer level.
Despite this simple aggregation strategy, COR-
TEX achieves competitive performance against the
baselines. This suggests that its token-level pre-
dictions provide useful signals for both localized
detection and answer-level reliability assessment.
At the same time, the strong performance of some
answer-level baselines suggests that answer-level
detection may require global signals beyond local-
ized token-level signals. Taken together, these re-
sults indicate that token-level and answer-level hal-
lucination detection are both important and should
be studied as complementary problems.
4.4 Ablation Study
To analyze the contribution of each CORTEX com-
ponent, we conduct an ablation study. Table 4
reports token- and answer-level performance for
each configuration. Both the contextual residual
cand label-persistence smoothing ( Smooth ) im-
prove performance. Removing label-persistence
smoothing shows that raw token-level scores are
informative but locally unstable, while removing
cshows that context-mediated reference influence
provides information complementary to∆h.
Figures 2 and 3 illustrate these effects qualita-
tively using cases with and without hallucinations.
In the heatmaps, color intensity reflects the pre-
dicted hallucination score, with redder tokens in-
dicating higher hallucination likelihood. In Fig-
ure 2, the model without label-persistence smooth-
ing produces interleaved high- and low-score to-
kens across semantically coherent regions, yield-
ing predictions less consistent with human span-
level annotations. In contrast, full CORTEX pro-
duces smoother and more contiguous high-score
6

MethodRAGTruth HalluRAG
Llama Qwen Mistral Llama Qwen Mistral
AP AUROC AP AUROC AP AUROC AP AUROC AP AUROC AP AUROC
NLL 0.0476 0.4676 0.0495 0.4686 0.0479 0.4666 0.2648 0.4952 0.2658 0.4934 0.2685 0.4945
LLM-Check 0.0769 0.6079 0.0951 0.6480 0.0743 0.6189 0.2994 0.5411 0.2874 0.5342 0.2944 0.5467
ICR Probe 0.2665 0.5380 0.2464 0.5067 0.2621 0.5311 0.3292 0.5285 0.3131 0.5106 0.3197 0.5143
SAPLMA 0.3894 0.8879 0.4182 0.8820 0.4027 0.8822 0.6875 0.8024 0.6584 0.8129 0.6193 0.7578
RAGLens 0.4271 0.8953 0.4575 0.9075 0.4244 0.8884 0.5725 0.7425 0.6380 0.7639 0.5596 0.7393
CORTEX 0.5686 0.9275 0.5940 0.9372 0.5473 0.9244 0.7690 0.8360 0.7495 0.8426 0.7308 0.8233
Table 2: Token-level hallucination detection results on RAGTruth and HalluRAG. Bold and underline indicate the
best and second-best results, respectively.
MethodRAGTruth HalluRAG
Llama Qwen Mistral Llama Qwen Mistral
AP AUROC AP AUROC AP AUROC AP AUROC AP AUROC AP AUROC
NLL 0.1994 0.5621 0.2019 0.5227 0.2313 0.6088 0.2055 0.4827 0.3184 0.5806 0.2969 0.5600
LLM-Check 0.3375 0.7100 0.3330 0.7198 0.3240 0.7170 0.3052 0.5711 0.3956 0.6638 0.4059 0.6481
ICR Probe 0.1286 0.3459 0.2301 0.6162 0.1831 0.5182 0.4077 0.6640 0.3286 0.6031 0.2157 0.4310
SAPLMA 0.6708 0.8863 0.5901 0.8600 0.4897 0.8077 0.7922 0.8482 0.7867 0.8695 0.7529 0.8628
RAGLens 0.7329 0.9011 0.6679 0.8812 0.6781 0.8819 0.7994 0.8905 0.7669 0.8380 0.8006 0.8835
CORTEX 0.6893 0.8957 0.7596 0.9077 0.6787 0.8907 0.7795 0.8481 0.79070.8627 0.80100.8718
Table 3: Answer-level hallucination detection results. Bold and underline indicate the best and second-best results,
respectively.
MethodToken-level Answer-level
AP AUROC AP AUROC
∆h0.4732 0.9074 0.6040 0.8853
∆h+c0.5335 0.9156 0.6910 0.9011
∆h+c+ Smooth0.5940 0.9372 0.7596 0.9077
Table 4: Ablation results on RAGTruth using Qwen.
regions, better matching the ground-truth halluci-
nation spans. Figure 3 illustrates the role of the
contextual residual c. Without c, the model can
overemphasize the absence of direct reference in-
fluence and assign high scores to tokens indirectly
grounded through the preceding answer context.
With c, CORTEX accounts for reference influence
propagated through previous tokens and reduces
false positives caused by indirect grounding.
Overall, these ablation results support the design
of CORTEX. The combination of ∆h, the contex-
tual residual c, and label-persistence smoothing
enables CORTEX to achieve robust localized hal-
lucination detection. Further ablation case studies
are presented in Appendix E.
5 Related Work
The problem of hallucination in LLMs has been ex-
tensively studied, yet remains unresolved (Huanget al., 2025; Kalai et al., 2025). It undermines the re-
liability of LLM-based AI agents and hinders real-
world deployment. RAG mitigates hallucinations
by incorporating references into prompts (Gao
et al., 2024; Rackauckas, 2024), but LLMs may
still generate hallucinated content even with refer-
ences, motivating the detection of groundless or
inconsistent RAG outputs (Fan et al., 2026).
Many hallucination detection methods rely on
external verification or repeated generation. Self-
consistency methods estimate reliability by compar-
ing multiple outputs from the same prompt (Man-
akul et al., 2023), while prompt-based approaches
use LLMs as judges or fact-checkers (Li et al.,
2025; Es et al., 2024; Furumai et al., 2024). These
methods exploit LLMs’ language understanding
but depend on prompt design and verifier capabil-
ity, and often require additional inference.
Another line of work uses model-internal sig-
nals, including uncertainty from next-token distri-
butions (Malinin and Gales, 2021), hallucination-
related information in intermediate transformer rep-
resentations (Azaria and Mitchell, 2023), spectral
features of attention and feedforward states (Srira-
manan et al., 2024), component-level attribution
for tracing information sources (Zhang et al., 2025),
7

Figure 2: Detection example for an answer containing hallucinated content. Label-persistence smoothing suppresses
noisy token-level score patterns and highlights the hallucinated span more accurately and coherently.
Figure 3: Detection example for an answer without hallucinated content. The contextual residual creduces false
positives by accounting for reference influence mediated through the preceding answer context.
and sparse autoencoder features for RAG halluci-
nation analysis (Xiong et al., 2026). These studies
show that hallucination-related signals are present
in model computations and representations, moti-
vating representation-based detection.
CORTEX differs from prior representation-
based approaches by comparing internal represen-
tations obtained from reference-conditioned and
no-reference inputs, rather than analyzing a single
internal state in isolation. It further uses label-
persistence smoothing to reduce isolated noise and
better align predictions with span-based hallucina-
tion annotations.
6 Conclusion
We proposed CORTEX, a post-hoc token-level hal-
lucination detection method for RAG. CORTEX
constructs a paired counterfactual view of the same
answer by analyzing its internal representations
with and without references, and encodes reference-
induced changes as token-level delta features. Itcombines these features with attention patterns to
capture context-mediated reference influence and
uses label-persistence smoothing to reduce local
noise while preserving span-consistent scores.
Experiments show that CORTEX substantially
outperforms token-level baselines and achieves
competitive answer-level performance through sim-
ple score aggregation. Ablations confirm the con-
tributions of both the attention-based contextual
residual and label-persistence smoothing. These
results demonstrate the effectiveness of comparing
internal representations for hallucination detection
in RAG. By using an open-weight LLM to analyze
outputs from closed-weight LLMs without mod-
ifying generation, CORTEX provides a practical
approach to reliability assessment. Beyond detec-
tion, these comparative internal-representation sig-
nals may provide a basis for future hallucination
mitigation, including reward modeling for tuning
generators and objectives that encourage stronger
reference-induced representations.
8

Limitations
CORTEX is designed for RAG settings and there-
fore assumes the presence of references against
which generated outputs can be assessed. It is not
intended to verify claims that are generated solely
from the parametric knowledge of an LLM. This
restriction may be acceptable, or even desirable,
in controlled applications such as enterprise cus-
tomer support, where answers should be grounded
only in approved documents. In more open-ended
assistant scenarios, however, this may be limiting
because useful information that is not explicitly
grounded in the provided references may be treated
as groundless.
Another limitation is that CORTEX requires ac-
cess to an open-weight LLM that exposes internal
representations, including hidden states and atten-
tion weights. Although the original generator can
be API-based or otherwise inaccessible, the post-
hoc analysis model must provide sufficient internal
signals. The quality of detection may therefore de-
pend on the choice of the open-weight LLM and
its ability to interpret the generated answer and
references.
A further limitation is the need for fine-grained
supervision during training. Although RAGTruth
provides human span-level annotations, such anno-
tations remain scarce for RAG hallucination detec-
tion. Improving supervision under limited annota-
tion resources remains an important direction for
future work.
Ethical Considerations
CORTEX is intended to support reliability assess-
ment in RAG systems by identifying potentially
groundless tokens in generated outputs. Such de-
tection can help reduce the risk of users relying on
hallucinated information, especially in applications
where answers are expected to be grounded in spe-
cific references. However, CORTEX should not
be interpreted as a guarantee of factual correctness.
Its predictions are probabilistic and may include
both false positives and false negatives; therefore,
human review or additional verification may still
be necessary in high-stakes domains.
There is also a risk that hallucination detection
tools could be over-relied upon or used to present
generated outputs as more reliable than they actu-
ally are. We encourage practitioners to communi-
cate the limitations of detection results clearly and
to avoid using CORTEX as the sole safety mecha-nism for critical decision-making.
It is also important to distinguish the scope of
CORTEX from broader safety evaluation. COR-
TEX is designed to detect ungrounded content in
reference-grounded generation, not to determine
broader notions of truth, fairness, or social harm.
Future work should examine how token-level re-
liability signals can be combined with other safe-
guards, including checks for bias, toxicity, privacy
risks, and domain-specific safety issues.
References
Amos Azaria and Tom Mitchell. 2023. The internal
state of an LLM knows when it‘s lying. InProceed-
ings of the 2023 Conference on Empirical Methods
in Natural Language Processing, pages 967–976. As-
sociation for Computational Linguistics.
Yejin Bang, Ziwei Ji, Alan Schelten, Anthony
Hartshorn, Tara Fowler, Cheng Zhang, Nicola Can-
cedda, and Pascale Fung. 2025. HalluLens: LLM
hallucination benchmark. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 24128–
24156. Association for Computational Linguistics.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158. Association
for Computational Linguistics.
Dongyang Fan, Sebastien Delsad, Nicolas Flammarion,
and Maksym Andriushchenko. 2026. Halluhard: A
hard multi-turn hallucination benchmark.Computing
Research Repository, arXiv:2602.01031.
Kazuaki Furumai, Roberto Legaspi, Julio Cesar Viz-
carra Romero, Yudai Yamazaki, Yasutaka Nishimura,
Sina Semnani, Kazushi Ikeda, Weiyan Shi, and Mon-
ica Lam. 2024. Zero-shot persuasive chatbots with
LLM-generated strategies and information retrieval.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
11224–11249. Association for Computational Lin-
guistics.
Prakhar Ganesh, Reza Shokri, and Golnoosh Farnadi.
2026. Rethinking hallucinations: Correctness, con-
sistency, and prompt multiplicity. InProceedings of
the 19th Conference of the European Chapter of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 6959–6978, Rabat, Morocco.
Association for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin-
liu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and
Haofen Wang. 2024. Retrieval-augmented genera-
tion for large language models: A survey.Computing
Research Repository, arXiv:2312.10997.
9

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3
herd of models.Computing Research Repository,
arXiv:2407.21783.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2).
Adam Tauman Kalai, Ofir Nachum, Santosh S. Vem-
pala, and Edwin Zhang. 2025. Why language mod-
els hallucinate.Computing Research Repository,
arXiv:2509.04664.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad
Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhat-
tacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu,
Kai Shu, Lu Cheng, and Huan Liu. 2025. From gen-
eration to judgment: Opportunities and challenges of
LLM-as-a-judge. InProceedings of the 2025 Con-
ference on Empirical Methods in Natural Language
Processing, pages 2757–2791. Association for Com-
putational Linguistics.
Qiang Liu, Xinlong Chen, Yue Ding, Bowen Song,
Weiqiang Wang, Shu Wu, and Liang Wang. 2025.
Attention-guided self-reflection for zero-shot hallu-
cination detection in large language models. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 21005–
21021. Association for Computational Linguistics.
Andrey Malinin and Mark John Francis Gales. 2021.
Uncertainty estimation in autoregressive structured
prediction. InProceedings of the 2021 International
Conference on Learning Representations.
Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.
SelfCheckGPT: Zero-resource black-box hallucina-
tion detection for generative large language models.
InProceedings of the 2023 Conference on Empiri-
cal Methods in Natural Language Processing, pages
9004–9017. Association for Computational Linguis-
tics.
Mistral AI. 2023. Mistral-7B-Instruct-v0.2. Hugging
Face model card.Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–
10878. Association for Computational Linguistics.
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o
system card.
Lawrence R. Rabiner. 1990. A tutorial on hidden
markov models and selected applications in speech
recognition. page 267–296.
Zackary Rackauckas. 2024. Rag-fusion: A new take on
retrieval augmented generation.International Jour-
nal on Natural Language Computing, 13(1):37–47.
Fabian Ridder and Malte Schilling. 2024. The hallurag
dataset: Detecting closed-domain hallucinations in
rag applications using an llm’s internal states.Com-
puting Research Repository, arXiv:2412.17056.
Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar
Sadasivan, Shoumik Saha, Priyatham Kattakinda,
and Soheil Feizi. 2024. Llm-check: investigating
detection of hallucinations in large language models.
InProceedings of the 38th International Conference
on Neural Information Processing Systems. Curran
Associates Inc.
Guangzhi Xiong, Zhenghao He, Bohan Liu, Sanchit
Sinha, and Aidong Zhang. 2026. Toward faithful
retrieval-augmented generation with sparse autoen-
coders. InThe Fourteenth International Conference
on Learning Representations.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 others.
2025. Qwen3 technical report.Computing Research
Repository, arXiv:2505.09388.
Zhenliang Zhang, Xinyu Hu, Huixuan Zhang, Junzhe
Zhang, and Xiaojun Wan. 2025. ICR probe: Tracking
hidden state dynamics for reliable hallucination de-
tection in LLMs. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics, pages 17986–18002. Association for Com-
putational Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Proceedings of the 37th International Conference
10

on Neural Information Processing Systems. Curran
Associates Inc.
ADetails of Label-Persistence Smoothing
This appendix gives the forward–backward recur-
sions and posterior marginal derivation for label-
persistence smoothing. The method defines non-
negative weights over smoothing-label sequences
by combining token-level confidence with label per-
sistence between neighboring labels, so that poste-
rior marginals can be computed efficiently without
enumerating all2Tpossible label sequences.
A.1 Forward Recursion
The forward message Fi(ℓ)is the unnormalized to-
tal weight of all partial smoothing-label sequences
ending withz i=ℓ:
Fi(ℓ) =X
z1:i:zi=ℓπ(z1)iY
j=1ϕj(zj)
×iY
j=2ρ(zj−1, zj).(13)
The initialization is
F1(ℓ) =π(ℓ)ϕ 1(ℓ).(14)
Fori= 2, . . . , T , the recursion is obtained by
grouping the partial sequences according to the pre-
vious labelℓ′=zi−1. Starting from the definition,
Fi(ℓ) =X
z1:i:zi=ℓπ(z1)iY
j=1ϕj(zj)iY
j=2ρ(zj−1, zj)
=X
ℓ′∈{0,1}X
z1:i−1 :zi−1=ℓ′π(z1)i−1Y
j=1ϕj(zj)
×i−1Y
j=2ρ(zj−1, zj)ρ(ℓ′, ℓ)ϕ i(ℓ).
(15)
The inner sum is exactlyF i−1(ℓ′). Therefore,
Fi(ℓ) =ϕ i(ℓ)X
ℓ′∈{0,1}Fi−1(ℓ′)ρ(ℓ′, ℓ).(16)
Since the smoothing label is binary, the recursion
can be written explicitly. Let p=p stay. Using
ϕi(1) =s iandϕ i(0) = 1−s i, we obtain
Fi(1) =s i[pFi−1(1) + (1−p)F i−1(0)],(17)
and
Fi(0) = (1−s i) [pF i−1(0) + (1−p)F i−1(1)].
(18)The corresponding initial values are
F1(1) =1
2s1, F 1(0) =1
2(1−s 1).(19)
A.2 Backward Recursion
The backward message Bi(ℓ)is the unnormalized
total weight of all suffix smoothing-label sequences
from positionsi+ 1toT, conditioned onz i=ℓ:
Bi(ℓ) =X
zi+1:TTY
j=i+1ϕj(zj)
×TY
j=i+1ρ(zj−1, zj), z i=ℓ.(20)
The initialization is
BT(ℓ) = 1,(21)
which corresponds to the empty product beyond
the final token.
Fori=T−1, . . . ,1 , the recursion is obtained
by grouping the suffix sequences according to the
next labelℓ′=zi+1. Starting from the definition,
Bi(ℓ) =X
zi+1:TTY
j=i+1ϕj(zj)TY
j=i+1ρ(zj−1, zj)
=X
ℓ′∈{0,1}X
zi+2:Tϕi+1(ℓ′)ρ(ℓ, ℓ′)
×TY
j=i+2ϕj(zj)TY
j=i+2ρ(zj−1, zj).
(22)
The inner sum is exactlyB i+1(ℓ′). Therefore,
Bi(ℓ) =X
ℓ′∈{0,1}ρ(ℓ, ℓ′)ϕi+1(ℓ′)Bi+1(ℓ′).(23)
For the binary case, this recursion is equivalently
Bi(1) =ps i+1Bi+1(1)+(1−p)(1−s i+1)Bi+1(0),
(24)
and
Bi(0) =p(1−s i+1)Bi+1(0)+(1−p)s i+1Bi+1(1).
(25)
The terminal values are
BT(1) = 1, B T(0) = 1.(26)
11

A.3 Posterior Marginal
The posterior marginal for smoothing label ℓat
positioniis
qi(ℓ) =P(z i=ℓ|s 1:T) =X
z1:T:zi=ℓP(z 1:T|s1:T).
(27)
Substituting the normalized sequence weight gives
qi(ℓ) =1
Z(s 1:T)X
z1:T:zi=ℓπ(z1)TY
j=1ϕj(zj)
×TY
j=2ρ(zj−1, zj).(28)
Because the sequence weight has a first-order
linear-chain factorization, fixing zi=ℓseparates
the unnormalized weight into prefix and suffix
terms:
X
z1:T:zi=ℓπ(z1)TY
j=1ϕj(zj)TY
j=2ρ(zj−1, zj)
=Fi(ℓ)B i(ℓ).(29)
Thus,
qi(ℓ) =Fi(ℓ)B i(ℓ)
Z(s 1:T).(30)
The normalizing constant can be decomposed at
positioni:
Z(s 1:T) =X
ℓ′∈{0,1}Fi(ℓ′)Bi(ℓ′).(31)
Therefore,
qi(ℓ) =Fi(ℓ)B i(ℓ)P
ℓ′∈{0,1} Fi(ℓ′)Bi(ℓ′).(32)
The final smoothed hallucination score is
˜si=qi(1) =Fi(1)B i(1)
Fi(0)B i(0) +F i(1)B i(1).(33)
B Sensitivity Analysis ofp stay
We analyze the sensitivity of CORTEX to the self-
loop probability pstayused in label-persistence
smoothing. This parameter controls the strength
of span-level smoothing: larger values encour-
age neighboring tokens to remain in the same
smoothing-label variable, whereas smaller values
keep the smoothed scores closer to the raw token-
level classifier outputs.Figures 4 and 5 show the results for token-
level and answer-level evaluation, respectively.
The dashed horizontal lines indicate the raw
scores before label-persistence smoothing. Across
both datasets and all open-weight LLMs, label-
persistence smoothing improves over the raw
scores for a wide range of pstay, indicating that
the gains of CORTEX are not tied to a single finely
tuned value.
For token-level detection, performance is gener-
ally stable when pstayis large. On both RAGTruth
and HalluRAG, AP increases as pstaybecomes
larger and then forms a broad plateau around high
values such as 0.99,0.993 , and 0.995 . AUROC
shows a similar trend, with improvements over the
raw scores maintained across a wide range of high
pstayvalues. This behavior is consistent with the
role of label-persistence smoothing in token-level
hallucination detection: hallucinated content typ-
ically appears as contiguous spans, and therefore
stronger self-transition probabilities help suppress
isolated noisy predictions while preserving coher-
ent hallucinated regions.
For answer-level detection, the optimal pstay
tends to be smaller than in token-level evaluation.
This is because answer-level scores are obtained
by aggregating token-level scores with the maxi-
mum operator. If pstayis too large, smoothing may
overly spread high scores across neighboring to-
kens or make span boundaries too persistent, which
can affect the maximum score used for answer-
level classification. Nevertheless, label-persistence
smoothing remains beneficial across a broad range
of values, and the selected setting pstay= 0.93 pro-
vides strong and stable answer-level performance.
Based on these results, we use pstay= 0.993 for
token-level evaluation and pstay= 0.93 for answer-
level evaluation in the main experiments. We use a
single value for each evaluation granularity rather
than tuning pstayseparately for each dataset and
open-weight LLM. This setting keeps the evalua-
tion protocol simple while still capturing the main
benefit of label-persistence smoothing: converting
locally noisy token-level scores into more span-
consistent hallucination estimates.
C Experimental Details
MLP classifier.For all methods that require a su-
pervised classifier, including CORTEX and MLP-
based baselines, we use the same three-layer MLP
architecture to isolate the effect of the input fea-
12

0.500.520.540.560.58AP
0.990.995
0.995RAGTruth / T oken AP
0.700.720.740.76AP
0.995
0.993
0.995HalluRAG / T oken AP
Llama
Qwen
Mistral
Raw score
0.5 0.70.85 0.93 0.97 0.990.995 0.999
pstay0.900.910.920.93AUROC
RAGTruth / T oken AUROC
0.5 0.70.85 0.93 0.97 0.990.995 0.999
pstay0.800.810.820.830.84AUROC
HalluRAG / T oken AUROCFigure 4: Sensitivity of token-level hallucination detection performance to pstay. Dashed horizontal lines indicate
raw scores before label-persistence smoothing.
tures. The classifier consists of two hidden linear
layers with 256 and 128 hidden units, respectively,
followed by a final linear layer that produces a
scalar output logit. Each hidden layer is followed
by a ReLU activation and dropout with a rate of
0.1. We train the classifier using AdamW with a
learning rate of 1×10−3and a weight decay of
1×10−4. The batch size is set to 4096, and the
model is trained for 10 epochs. The same hyperpa-
rameters are used across all datasets, models, and
methods unless otherwise stated.
Representation extraction.To obtain internal
representations from the open-weight analysis
model Mopen, we do not perform autoregressive
generation. Instead, we feed the constructed input
sequence to Mopen and extract the internal repre-
sentations computed in a single forward pass. This
implementation matches the post-hoc setting con-
sidered in this work: the answer text has already
been generated by the closed-weight model, and
Mopen is used only to analyze the given answer.
Computational cost.All experiments are con-
ducted on a single NVIDIA A100 GPU. CORTEX
is lightweight in practice: on RAGTruth, feature
extraction, classifier training, and evaluation take
approximately 20 minutes in total, while on Hal-
luRAG they take approximately 5 minutes in to-tal. This indicates that CORTEX incurs only mod-
est computational overhead while providing token-
level hallucination scores.
Artifact Licenses and Use.We use RAGTruth
and HalluRAG as existing public evaluation
benchmarks and do not redistribute the datasets.
RAGTruth is released under the MIT License. Hal-
luRAG is made publicly available by its authors
through their repository and dataset DOI, but we
did not find an explicit dataset license in the avail-
able documentation. All datasets are used solely
for research evaluation, and we cite their original
sources.
Dataset Documentation.We use only publicly
available English-language RAG hallucination
benchmarks and do not collect any new data. Our
experiments focus on hallucination detection in
generated RAG outputs; we do not use, infer, or
analyze demographic attributes.
D Effect of Label-persistence Smoothing
on Token-Level Baselines
Label-persistence smoothing is a post-processing
module that can be applied to any method that
produces token-level hallucination scores. To ex-
amine whether the gains of CORTEX are due only
to label-persistence smoothing procedure to two
13

0.6250.6500.6750.7000.7250.750AP
0.950.95
0.93RAGTruth / Answer AP
0.750.760.770.780.790.800.81AP
0.930.990.7HalluRAG / Answer AP
Llama
Qwen
Mistral
Raw score
0.5 0.70.85 0.93 0.97 0.990.995 0.999
pstay0.8750.8800.8850.8900.8950.9000.9050.910AUROC
RAGTruth / Answer AUROC
0.5 0.70.85 0.93 0.97 0.990.995 0.999
pstay0.830.840.850.860.87AUROC
HalluRAG / Answer AUROCFigure 5: Sensitivity of answer-level hallucination detection performance to pstay. Answer-level scores are obtained
by taking the maximum over token-level hallucination scores. Dashed horizontal lines indicate raw scores before
label-persistence smoothing.
strong token-level baselines, SAPLMA and RA-
GLens, and compare them with CORTEX under
the same settings. Tables 5 and 6 report the results
on RAGTruth and HalluRAG, respectively.
Before applying label-persistence smoothing,
CORTEX achieves the best performance in all
settings across both datasets and all open-weight
LLMs. This result indicates that the comparative
representation features of CORTEX already pro-
vide a stronger token-level signal than the single-
view representation features used by the base-
lines. After label-persistence smoothing is ap-
plied, the performance of SAPLMA and RAGLens
improves substantially, confirming that span-level
post-processing is broadly useful for reducing lo-
cal prediction noise. Nevertheless, CORTEX re-
mains strongest in AP across all RAGTruth settings
and remains competitive or superior in most Hal-
luRAG settings. These results suggest that label-
persistence smoothing is beneficial as a generic
post-processing step.
In particular, the comparison before smoothing
isolates the effect of the underlying token-level
scoring function: CORTEX outperforms the base-
lines without relying on span-level post-processing.
The comparison after smoothing further shows
that CORTEX can benefit from the same genericsmoothing procedure while retaining the advan-
tage of its paired reference-conditioned and no-
reference representation comparison. Thus, label-
persistence smoothing and the comparative repre-
sentation features play complementary roles: the
former improves span consistency, whereas the lat-
ter provides a stronger reference-grounded halluci-
nation signal.
E Additional Heatmap-Based Ablation
Case Studies
We present additional heatmap-based case stud-
ies to qualitatively analyze the behavior of each
component in CORTEX. The heatmaps compare
the ground-truth hallucination spans with the pre-
dictions of the full CORTEX model, CORTEX
without label-persistence smoothing, and CORTEX
without the contextual residual c. Darker colors in-
dicate higher hallucination scores.
Figure 6 shows an example in which the an-
notated hallucination spans vary in granularity.
The ground-truth annotation contains both a broad
hallucinated region spanning multiple sentences
and more compact hallucinated fragments. With-
out label-persistence smoothing, CORTEX assigns
high scores at relatively fine-grained units. This
behavior suggests that the raw token-level classi-
14

MethodRAGTruth
Llama Qwen Mistral
AP AUROC AP AUROC AP AUROC
SAPLMA 0.3894 0.8879 0.4182 0.8820 0.4027 0.8822
+ Smoothing (P stay= 0.993) 0.4910 0.9225 0.5078 0.9163 0.4965 0.9205
+ Smoothing (P stay= 0.995) 0.4925 0.9228 0.5088 0.9168 0.4972 0.9210
+ Smoothing (P stay= 0.997) 0.4938 0.9232 0.5089 0.9174 0.4982 0.9216
RAGLens 0.4271 0.8953 0.4575 0.9075 0.4244 0.8884
+ Smoothing (P stay= 0.993) 0.5191 0.9300 0.5068 0.9286 0.5264 0.9268
+ Smoothing (P stay= 0.995) 0.5191 0.9302 0.5057 0.9286 0.5276 0.9274
+ Smoothing (P stay= 0.997) 0.5191 0.9305 0.5026 0.9286 0.5290 0.9282
CORTEX 0.5181 0.9097 0.5335 0.9156 0.4860 0.8989
+ Smoothing (P stay= 0.993) 0.56910.9274 0.5937 0.9370 0.5472 0.9243
+ Smoothing (P stay= 0.995) 0.5686 0.9275 0.59400.9372 0.54730.9244
+ Smoothing (P stay= 0.997) 0.5683 0.9277 0.59340.9375 0.5465 0.9245
Table 5: Effect of label-persistence smoothing on token-level baselines on RAGTruth. Improvements for both
SAPLMA and RAGLens indicate that span-level post-processing is broadly useful.
MethodHalluRAG
Llama Qwen Mistral
AP AUROC AP AUROC AP AUROC
SAPLMA 0.6875 0.8024 0.6584 0.8129 0.6193 0.7578
+ Smoothing (P stay= 0.993) 0.7527 0.8291 0.7125 0.8378 0.7132 0.8108
+ Smoothing (P stay= 0.995) 0.7539 0.8295 0.7119 0.8383 0.7160 0.8122
+ Smoothing (P stay= 0.997) 0.7546 0.8301 0.7128 0.8391 0.7173 0.8140
RAGLens 0.5725 0.7425 0.6380 0.7639 0.5596 0.7393
+ Smoothing (P stay= 0.993) 0.7146 0.8007 0.7668 0.8138 0.7420 0.8214
+ Smoothing (P stay= 0.995) 0.7169 0.8019 0.7695 0.8148 0.7459 0.8232
+ Smoothing (P stay= 0.997) 0.7202 0.8035 0.77320.8163 0.7513 0.8256
CORTEX 0.7309 0.8175 0.7250 0.8304 0.6864 0.7978
+ Smoothing (P stay= 0.993) 0.7688 0.8358 0.7516 0.8425 0.7300 0.8227
+ Smoothing (P stay= 0.995) 0.76900.8360 0.7495 0.8426 0.7308 0.8233
+ Smoothing (P stay= 0.997) 0.76820.8362 0.74780.8428 0.7303 0.8240
Table 6: Effect of applying smoothing to token-level baselines on HalluRAG. Label-persistence smoothing improves
all methods substantially, including SAPLMA and RAGLens.
fier can capture localized hallucination signals, but
its predictions are fragmented and locally unsta-
ble. After label-persistence smoothing, these frag-
mented high-score regions are connected into a
more coherent span, producing predictions that bet-
ter reflect the span-level nature of hallucination an-
notations. At the same time, this example also illus-
trates a trade-off introduced by smoothing: when
hallucinated evidence appears in several nearby
but distinct regions, label-persistence smoothing
may merge them into a broader continuous span.
Thus, label-persistence smoothing improves span
consistency, but may reduce boundary precision
in cases where hallucinated content is interleaved
with faithful tokens.
The same example also illustrates the role ofthe contextual residual c. When cis removed, the
model tends to assign high scores more broadly in
regions where the current token is influenced by
preceding generated context. This behavior is con-
sistent with the motivation for the residual feature:
∆halone captures reference-induced changes at the
current token, but it does not explicitly account for
reference influence that has already been expressed
through earlier answer tokens. By incorporating
c, CORTEX can distinguish direct token-level ref-
erence sensitivity from deviations relative to the
preceding context, leading to more controlled lo-
calization.
Figure 7 shows a different type of error case in
which the answer includes additional advice that
is not supported by the provided references. The
15

Figure 6: Ablation case study with hallucination spans of different granularities. Without label-persistence
smoothing, hallucination scores are more fragmented and localized. With label-persistence smoothing, nearby
high-score regions are connected into broader span-consistent predictions, although this can also merge distinct
hallucinated fragments into a wider continuous region.
advice is informative and may be reasonable from
the model’s parametric knowledge, but it is not
grounded in the supplied passages. CORTEX as-
signs high hallucination scores to this region be-
cause its objective is to detect whether the answer
is supported by the references, not whether the con-
tent is generally plausible or useful.
This case highlights an important ambiguity in
reference-grounded hallucination detection. In con-
trolled RAG applications, such as enterprise cus-
tomer support, advice that is not grounded in ref-
erences can be undesirable or risky even when it
is factually plausible, because the system is ex-
pected to answer only from those documents. In
such settings, flagging unsupported advice is a de-
sirable behavior. In contrast, in open-ended assis-
tant scenarios, suppressing all useful but reference-
unsupported content may overly restrict the capabil-
ities of the LLM. Therefore, the appropriate treat-
ment of such content depends on the application:
CORTEX should be interpreted as a detector ofreference support, rather than a general judge of
factual correctness or utility.
This example also clarifies the scope of COR-
TEX. Because the method compares internal repre-
sentations under inputs with and without references,
it is sensitive to whether a token is grounded in the
provided evidence. Consequently, content gener-
ated from parametric knowledge alone can receive
high hallucination scores if it is not supported by
the references. This behavior is aligned with the in-
tended design of CORTEX for reference-grounded
generation, but it should be considered when ap-
plying the method to settings where answers are
allowed to go beyond the retrieved references.
16

Figure 7: Ablation case study involving groundless advice. The answer includes additional advice that may be
informative but is not grounded in the provided references. CORTEX assigns high hallucination scores to this
region, reflecting its role as a detector of reference support rather than a general factuality or usefulness judge.
17
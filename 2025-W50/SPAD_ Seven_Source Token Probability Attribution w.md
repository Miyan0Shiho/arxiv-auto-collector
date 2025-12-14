# SPAD: Seven-Source Token Probability Attribution with Syntactic Aggregation for Detecting Hallucinations in RAG

**Authors**: Pengqian Lu, Jie Lu, Anjin Liu, Guangquan Zhang

**Published**: 2025-12-08 12:50:41

**PDF URL**: [https://arxiv.org/pdf/2512.07515v1](https://arxiv.org/pdf/2512.07515v1)

## Abstract
Detecting hallucinations in Retrieval-Augmented Generation (RAG) remains a challenge. Prior approaches attribute hallucinations to a binary conflict between internal knowledge (stored in FFNs) and retrieved context. However, this perspective is incomplete, failing to account for the impact of other components in the generative process, such as the user query, previously generated tokens, the current token itself, and the final LayerNorm adjustment. To address this, we introduce SPAD. First, we mathematically attribute each token's probability into seven distinct sources: Query, RAG, Past, Current Token, FFN, Final LayerNorm, and Initial Embedding. This attribution quantifies how each source contributes to the generation of the current token. Then, we aggregate these scores by POS tags to quantify how different components drive specific linguistic categories. By identifying anomalies, such as Nouns relying on Final LayerNorm, SPAD effectively detects hallucinations. Extensive experiments demonstrate that SPAD achieves state-of-the-art performance

## Full Text


<!-- PDF content starts -->

SPAD: Seven-Source Token Probability Attribution with Syntactic
Aggregation for Detecting Hallucinations in RAG
Pengqian Lu, Jie Lu*, Anjin Liu, and Guangquan Zhang
Australian Artificial Intelligence Institute (AAII)
University of Technology Sydney
Ultimo, NSW 2007, Australia
{Pengqian.Lu@student., Jie.Lu@, Anjin.Liu@, Guangquan.Zhang@}uts.edu.au
Abstract
Detecting hallucinations in Retrieval-
Augmented Generation (RAG) remains
a challenge. Prior approaches attribute
hallucinations to a binary conflict between
internal knowledge (stored in FFNs) and
retrieved context. However, this perspective is
incomplete, failing to account for the impact
of other components in the generative process,
such as the user query, previously generated
tokens, the current token itself, and the final
LayerNorm adjustment. To address this, we
introduce SPAD. First, we mathematically
attribute each token’s probability into seven
distinct sources: Query, RAG, Past, Current
Token, FFN, Final LayerNorm, and Initial
Embedding. This attribution quantifies how
each source contributes to the generation
of the current token. Then, we aggregate
these scores by POS tags to quantify how
different components drive specific linguistic
categories. By identifying anomalies, such as
Nouns relying on Final LayerNorm, SPAD
effectively detects hallucinations. Extensive
experiments demonstrate that SPAD achieves
state-of-the-art performance.
1 Introduction
Large Language Models (LLMs), despite their im-
pressive capabilities, are prone to hallucinations
(Huang et al., 2025). Consequently, Retrieval-
Augmented Generation (RAG) (Lewis et al., 2020)
is widely used to alleviate hallucinations by ground-
ing models in external knowledge. However, RAG
systems are not a panacea. They can still halluci-
nate by ignoring or misinterpreting the retrieved
information (Sun et al., 2025). Detecting such fail-
ures is therefore a critical challenge.
The prevailing paradigm for hallucination de-
tection often relies on hand-crafted proxy signals.
*Corresponding author.For example, common approaches measure out-
put uncertainty via consistency checks (Manakul
et al., 2023) or calculate the relative Mahalanobis
distance of embeddings against a background cor-
pus (Ren et al.). While efficient, these methods
measure the symptoms of hallucination rather than
its underlying architectural causes. Consequently,
they often fail when a model is confidently incor-
rect (Kadavath et al., 2022).
To address the root cause of hallucination, re-
cent research has shifted focus to the model’s in-
ternal representations. Pioneering works such as
ReDeEP (Sun et al., 2025) assumes the RAG con-
text is correct. They reveal that hallucinations in
RAG typically stem from a dominance of internal
parametric knowledge (stored in FFNs) over the
retrieved external context.
This insight inspires a fundamental question:Is
the conflict between FFNs and RAG the only
cause of hallucination?Critical components like
LayerNorm and the User Query are often over-
looked.Do contributions from these sources also
drive hallucinations?By decomposing the output
into seven distinct information source, we obtain a
complete attribution map. This enables detection
based on comprehensive internal mechanics rather
than partial proxy signals.
To achieve this, we also assume the RAG context
contains relevant information and introduce SPAD
(Seven-Source TokenProbabilityAttribution with
Syntactic Aggregation forDetecting Hallucinations
in RAG). This framework mathematically attributes
the final probability of each token to seven distinct
sources: Query, RAG, Past, Current Token, FFN,
Final LayerNorm, and Initial Embedding. The total
attribution across these seven parts sums exactly to
the token’s final probability, ensuring we capture
the complete generation process.
However, raw attribution scores alone are insuf-
ficient for detection. A high reliance on internal
knowledge (FFNs) does not necessarily imply aarXiv:2512.07515v1  [cs.CL]  8 Dec 2025

(a)Instance-level Diagnosis:Inference pipeline showing
probability attribution (Step 2) and POS aggregation. In Step
3, the SHAP force plot visualizes the hallucination detector
decision path: Red arrows indicate anomalous features (e.g.,
high LNusage on Num) that actively push the prediction score
higher towards a "Hallucination" verdict.
(b)Global Feature Analysis:SHAP summary plot. The
x-axis denotes impact on detection (Right = Hallucination).
Color denotes feature value (Red=High). Visual patterns re-
veal syntax-specific hallucination signals: for RAG_NOUN , Blue
dots cluster on the far right, indicating that low retrieval con-
tribution for Nouns drives hallucinations. Conversely, for
LN_NUM , Red dots on the right reveals high LayerNorm usage
on Numerals is likely to be hallucination.
Figure 1: Applying SPAD framework to a Llama2-7b
response from RagTruth dataset (Niu et al., 2024).
hallucination. This pattern is expected for function
words like "the" or "of". Yet, it becomes highly sus-
picious when found in named entities. Therefore,
treating all tokens equally fails to capture these
critical distinctions.
To capture this distinction, we aggregate the at-
tribution scores using Part-of-Speech (POS) tags.
We select POS tags for their universality and ef-ficiency, providing a robust feature space without
the complexity of full dependency parsing.
Figure 1 illustrates SPAD detecting hallucina-
tions on a Llama2-7b response (Touvron et al.,
2023). As shown in Figure 1a, the pipeline at-
tribute token probabilities to seven sources. These
scores are then aggregated by POS tags and input
into the classifier. The visualization in Step 3 high-
lights that the detection verdict is driven by specific
syntax-source anomalies rather than generic sig-
nals. Figure 1b further validates this design by ana-
lyzing the feature importance of the trained classi-
fier. The SHAP summary plot reveals that the most
discriminative features are specific combinations
of source and syntax. For instance, high RAGattri-
bution on NOUNS correlates with factualness, while
high LayerNorm attribution on NUMsignals halluci-
nation. This validates the necessity of syntax-aware
source attribution. Our main contributions are:
1.We propose SPAD, a novel framework that
mathematically attributes each token’s prob-
ability to seven distinct information sources.
This provides a comprehensive mechanistic
view of the token generation process.
2.We introduce a syntax-aware aggregation
mechanism. By quantifying how information
sources drive distinct parts of speech, this ap-
proach enables the detector to pinpoint anoma-
lies in specific entities while ignoring benign
grammatical patterns.
3.Extensive experiments demonstrate that
SPAD achieves state-of-the-art performance.
Our framework also offers transparent inter-
pretability, automatically uncovering novel
mechanistic signatures, such as anomalous
LayerNorm contributions, that extend beyond
the traditional FFN-RAG binary conflict.
2 Related Work
Uncertainty and Proxy Metrics.A prevalent
approach involves detecting hallucinations via out-
put uncertainty or proxy signals. Several methods
quantify the inconsistency across multiple sampled
responses or model ensembles to estimate uncer-
tainty (Manakul et al., 2023; Malinin and Gales,
2021). To avoid the cost of multiple generations,
others derive efficient proxy metrics from a sin-
gle forward pass. These include utilizing energy
scores or embedding distances (Liu et al., 2020;

Ren et al.), as well as measuring neighborhood in-
stability or keyword-specific uncertainty (Lee et al.,
2024; Zhang et al., 2023). While efficient, these
methods primarily measure correlates of hallucina-
tion rather than analyzing the underlying generative
mechanism.
LLM-based Evaluation.Another paradigm em-
ploys LLMs as external evaluators. In Retrieval-
Augmented Generation (RAG), frameworks verify
outputs against retrieved documents using claim
extraction (Niu et al., 2024; Friel and Sanyal, 2023;
Hu et al., 2024). Similarly, automated evalua-
tion suites have been developed to systematically
test for factual accuracy (Es et al., 2024; True-
Lens, 2024; Ravichander et al., 2025). Alterna-
tively, methods like cross-examination assess self-
consistency without external knowledge (Cohen
et al., 2023; Yehuda et al., 2024). Beyond prompt-
ing off-the-shelf models, (Su et al., 2025) proposed
RL4HS to fine-tune LLMs via reinforcement learn-
ing for span-level hallucination detection. However,
these approaches incur high computational costs
due to external API calls or iterative generation
steps, whereas our method is self-contained within
the generating model.
Probing Internal Activations.Recent research
investigates the model’s internal latent space for
factuality signals. Seminal works identify linear
“truthful directions” in activation space (Burns et al.,
2022; Li et al., 2023), while others train probes to
detect falsehoods even when outputs appear confi-
dent (Azaria and Mitchell, 2023; Han et al., 2024;
Chen et al., 2024). Beyond passive detection, some
approaches attribute behavior to specific compo-
nents (Sun et al., 2025) or actively intervene in the
decoding process. These interventions range from
modifying activations during inference (Li et al.,
2023) to re-ranking candidates based on internal
state signatures (Chen et al., 2025; Zhang et al.,
2025). Unlike these methods which often treat acti-
vations as aggregate features or modify generation,
our work explicitly decomposes the final output
probability. We trace the probability mass back to
seven information sources providing a fine-grained
and interpretable basis for detection.
3 Methodology
As illustrated in Figure 2, SPAD operates in three
stages. We first derive an exact decomposition of
token probabilities (Sec. 3.2), then attribute atten-tion contributions to specific information sources
(Sec. 3.3). Finally, we aggregate these scores to
quantify how sources drive distinct parts of speech
(Sec. 3.4). The pseudo-code and complexity anal-
ysis are provided in the Appendix. To provide the
theoretical basis for our method, we first formalize
the transformer’s residual architecture.
3.1 Preliminaries: Residual Architecture
We analyze a standard decoder-only Transformer
architecture, consisting of a stack of Llayers.
Given an input sequence x= (x 1, . . . , x T), the
model predicts the next tokenx T+1.
3.1.1 Residual Updates and Probing
The input tokens are mapped to continuous vec-
tors via an embedding matrix We∈RV×dand
summed with positional embeddings P. The initial
state is H(0)=W e[x] +P . We adopt the Pre-
normalization (Pre-LN) configuration. Crucially,
each layer lupdates the hidden state via additive
residual connections:
H(l)
mid=H(l−1)+Attn(LN(H(l−1)))(1)
H(l)=H(l)
mid+FFN(LN(H(l)
mid))(2)
This structure implies that the final representation is
the sum of the initial embedding and all subsequent
layer updates. To quantify these updates, we define
aProbe Function Φ(H, y) similar with logit lens
technique (nostalgebraist, 2020) that measures the
hypothetical probability of the target token ygiven
any intermediate stateH:
Φ(H, y) =h
Softmax(HW⊤
e)i
y(3)
Guiding Question:Since the model is a stack of
residual updates, can we mathematically decom-
pose the final probability exactly into the sum of
component contributions?
3.2 Coarse-Grained Decomposition
We answer the preceding question affirmatively
by leveraging the additive nature of the residual
updates. Based on the probe function Φ(H, y) de-
fined in Eq. (3), we isolate the probability contri-
bution of each model component as the distinct
change it induces in the probe output.
We define the baseline contribution from in-
put static embeddings ( ∆P initial), the incremental
gains from Attention and FFN blocks in layer l

Figure 2: Overview of the SPAD framework. The attribution process consists of three progressive stages: (1)
Coarse-Grained Decomposition: The final token probability is exactly decomposed into additive contributions from
residual streams and LayerNorm components (Section 3.2). (2)Fine-Grained Attribution & Source Mapping:
Attention contributions are apportioned to individual heads and subsequently mapped to four distinct input sources
(Query, RAG, Past, Self) based on attention weights (Section 3.3). (3)Syntax-Aware Feature Engineering: These
source-specific attributions are aggregated by POS tags to construct the final syntax-aware feature representation for
hallucination detection (Section 3.3.4).
(∆P(l)
att,∆P(l)
ffn), and the adjustment from the final
LayerNorm (∆P LN) as follows:
∆P initial(y) = Φ(H(0), y)(4)
∆P(l)
att= Φ(H(l)
mid, y)−Φ(H(l−1), y)(5)
∆P(l)
ffn= Φ(H(l), y)−Φ(H(l)
mid, y)(6)
∆P LN=P final(y)−Φ(H(L), y)(7)
By summing these telescoping differences, we de-
rive the exact decomposition of the model’s output.
Theorem 1(Exact Probability Decomposition).
The final probability for a target token yis exactly
the sum of the contribution from the initial embed-
ding, the cumulative contributions from Attention
and FFN blocks across all Llayers, and the adjust-ment from the final LayerNorm:
Pfinal(y) =∆P initial(y) + ∆P LN
+LX
l=1
∆P(l)
att+ ∆P(l)
ffn (8)
Proof.See Appendix.
The Guiding Question:While Eq. (8) quantifies
how muchthe model components contribute to the
prediction probability, it treats the term ∆P(l)
attas a
black box. To effectively detect hallucinations, we
must identifywherethis attention is focused.
3.3 Fine-Grained Attribution
To identify the focus of attention, we must decom-
pose the attention contribution ∆P(l)
attinto contri-

butions from individual attention heads.
3.3.1 The Challenge: Non-Linearity
Standard Multi-Head Attention concatenates the
outputs of Hindependent heads and projects them
via an output matrix WO. Mathematically, by par-
titioning WOinto head-specific sub-matrices, this
operation is strictly equivalent to the sum of pro-
jected head outputs:
H(l)
att=HX
h=1(AhVh)|{z}
hhW(h)
O(9)
where hhis the raw output of head h, derived from
the layer input x, the head-specific value projection
Vh, and the attention matrixA h.
Eq. (9) establishes that the attention output is lin-
ear with respect to individual heads in the hidden
state space. However, our goal is to attribute the
probabilitychange ∆P(l)
attto each head. Since the
probe function Φ(·) employs a non-linear Softmax
operation, the sum of probability changes calcu-
lated by probing individual heads does not equal
the attention block contribution:
∆P(l)
att̸=HX
h=1
Φ(H input+hhW(h)
O)−Φ(H input)
(10)
This inequality prevents us from calculating head
contributions by simply probing each head individ-
ually, motivating our shift to the logit space.
3.3.2 Logit-Based Apportionment
To bypass the non-linearity of the Softmax, we
analyze contributions in the logit space, where ad-
ditivity is preserved. Let ∆z(l)
h,ydenote the scalar
contribution of head hto the logit of thetarget to-
keny. This is calculated as the dot product between
the projected head output and the target token’s un-
embedding vectorW e,y:
∆z(l)
h,y=h
(AhVh)W(h)
Oi
·We,y (11)
We then apportion the exact probability contribu-
tion∆P(l)
att(derived in Section 3.2) to each head h
proportional to its exponential logit contribution:
∆P(l)
h= ∆P(l)
att·exp(∆z(l)
h,y)
PH
j=1exp(∆z(l)
j,y)(12)3.3.3 Theoretical Justification
Our strategy relies on the relationship between logit
changes and probability updates. We rigorously
establish this link via a first-order Taylor expansion.
Proposition 1(Gradient-Based Linear Decompo-
sition).The total probability contribution of the
attention block, ∆P(l)
att, is approximated by the sum
of head-specific logit contributions ∆z(l)
h,y, scaled
by a layer-wide gradient factor:
∆P(l)
att≈ G(l)·HX
h=1∆z(l)
h,y+E(13)
where G(l)is a common gradient term shared by
all heads in layer l, andEis the higher-order term.
Proof.See Appendix.
While Proposition 1 suggests linearity, direct ap-
portionment is numerically unstable. For instance,
if positive and negative contributions across heads
sum to near zero, the resulting weights would ex-
plode mathematically. Thus, Eq. (12) employs Soft-
max to avoid this problem.
Justification for First-Order Approximation.
We restrict our analysis to the first-order term for
efficiency. Computing second-order Hessian inter-
actions incurs a prohibitive O(V2)cost. Moreover,
since LLMs are often “confidently incorrect” dur-
ing hallucinations (Kadavath et al., 2022), the prob-
ability mass concentrates on the target, naturally
suppressing the E. Thus, the first-order approxima-
tion is both computationally necessary ( O(V) ) and
sufficiently accurate.
3.3.4 Source Mapping and The 7-Source Split
Having isolated the head contribution ∆P(l)
h, we
can now answer the guiding question by tracing
attention back to input tokens using the attention
matrix Ah. We categorize inputs into four source
types: S={Qry,RAG,Past,Self} . For a source
typeScontaining token indices IS, the aggregated
contribution is:
∆P(l)
S=HX
h=1
∆P(l)
h·P
k∈ISAh[T, k]P
allkAh[T, k]
(14)
By aggregating these components, we achieve a
complete partition of the final probability Pfinal(y)
into exactly seven distinct sources:
Pfinal(y) =P initial(y) + ∆P LN
+LX
l=1 
∆P(l)
ffn+X
S∈S∆P(l)
S!
(15)

The Guiding Question:We have now derived a
precise 7-dimensional attribution vector for every
token. However, raw attribution scores lack con-
text: a high PFFNcontribution might be normal for
a function word but suspicious for a proper noun.
How can we contextualize these scores with syn-
tactic priors?
3.4 Syntax-Aware Feature Engineering
To resolve this ambiguity, we employ Part-of-
Speech (POS) tagging as a lightweight syntactic
prior. Specifically, we assign a POS tag by Spacy
(Honnibal et al., 2020) to each generated token
and aggregate the attribution scores for each gram-
matical category. By profiling which information
sources (e.g., RAG) the LLM relies on for different
parts of speech, we detect hallucination effectively.
3.4.1 Tag Propagation Strategy
A mismatch problem arises because LLMs may
split a word into multiple tokens while standard
POS taggers process whole words. We resolve this
via tag propagation: generated sub-word tokens
inherit the POS tag of their parent word. For in-
stance, if the noun “modification” is tokenized into
[“modi”, “fication”], both sub-tokens are assigned
theNOUNtag.
3.4.2 Aggregation
We first define the attribution vector vt∈R7for
each token xtas the concatenation of its 7 source
components derived in Section 3.3.4. To capture
role-specific provenance patterns, we compute the
mean attribution vector for each POS tagτ:
¯vτ=1
|{t|POS(x t) =τ}|X
t:POS(x t)=τvt(16)
The final feature vector f∈R7×|POS|is the concate-
nation of these POS-specific vectors. This repre-
sentation combinesprovenance(source attribution)
withsyntax(linguistic structure), forming a robust
basis for hallucination detection.
4 Experiments
4.1 Experimental Setup
We treat hallucination detection as a supervised
binary classification task. We employ XGBoost
(Chen and Guestrin, 2016) as our classifier, chosen
for its efficiency and interpretability on tabular data.
The input to the classifier is a 126-dimensionalsyntax-aware feature vector, constructed by aggre-
gating the 7-source attribution scores across 18
universal POS tags (e.g., NOUN, VERB) defined
by SpaCy (Honnibal et al., 2020). To ensure rigor-
ous evaluation and reproducibility, we implement
strict data isolation protocols (e.g., stratified cross-
validation) tailored to the data availability of each
task. See implementation details in Appendix.
4.2 Dataset and Baselines
To ensure a fair comparison, we utilize the public
RAG hallucination benchmark established by (Sun
et al., 2025). This benchmark comprises human-
annotated responses from Llama2 (7B/13B) and
Llama3 (8B) across two datasets:RAGTruth(QA,
Data-to-Text, Summarization) andDolly(Summa-
rization, Closed-QA). Implementation details are
provided in Appendix. We compare our method
against representative approaches from three cate-
gories introduced in Section 2. The introduction of
baseliens are provided in Appendix.
4.3 Comparison with Baselines
The main experimental results, presented in Table
1, demonstrate that our proposed attribution-based
framework achieves superior performance across
the majority of experimental settings, significantly
outperforming baselines on the large-scale bench-
mark while maintaining strong competitiveness on
low-resource datasets.
On the comprehensive RAGTruth benchmark,
our method demonstrates a clear advantage. For
instance, with the Llama2-7B model, our detector
achieves an F1-score of 0.7218 and an AUC of
0.7839, surpassing the strongest competitor, Re-
DeEP, which scored 0.7190 and 0.7458, respec-
tively. This trend is amplified with the larger
Llama2-13B model, where we attain the highest
F1-score (0.7912) and AUC (0.8685). The per-
formance gap is evident on the newer Llama3-8B
model, where our framework leads by a substantial
margin, achieving an F1-score of 0.7975 compared
to the next-best 0.6986 from LMVLM. These re-
sults on a sufficiently large dataset robustly validate
our core hypothesis: a complete, syntax-aware at-
tribution map provides a richer feature space for
identifying hallucinations than methods relying on
limited proxy signals.
The Dolly (AC) dataset presents a more chal-
lenging scenario due to its extreme data scarcity
(only 100 test samples). Despite this, our method
demonstrates remarkable adaptability. On the

RAGTruth
Method LLaMA2-7B LLaMA2-13B LLaMA3-8B
Metric AUC Recall F1 AUC Recall F1 AUC Recall F1
SelfCheckGPT (Manakul et al., 2023) — 0.4642 0.4642 — 0.4642 0.4642 — 0.5111 0.5111
Perplexity (Ren et al.) 0.5091 0.5190 0.6749 0.5091 0.5190 0.6749 0.6235 0.6537 0.6778
LN-Entropy (Malinin and Gales, 2021) 0.5912 0.5383 0.6655 0.5912 0.5383 0.6655 0.7021 0.5596 0.6282
Energy (Liu et al., 2020) 0.5619 0.5057 0.6657 0.5619 0.5057 0.6657 0.5959 0.5514 0.6720
Focus (Zhang et al., 2023) 0.6233 0.5309 0.6622 0.7888 0.6173 0.6977 0.6378 0.6688 0.6879
Prompt (Niu et al., 2024) — 0.7200 0.6720 — 0.7000 0.6899 — 0.4403 0.5691
LMVLM (Cohen et al., 2023) — 0.7389 0.6473 —0.83570.6553 — 0.5109 0.6986
ChainPoll (Friel and Sanyal, 2023) 0.6738 0.7832 0.7066 0.7414 0.7874 0.7342 0.6687 0.4486 0.5813
RAGAS (Es et al., 2024) 0.7290 0.6327 0.6667 0.7541 0.6763 0.6747 0.6776 0.3909 0.5094
Trulens (TrueLens, 2024) 0.6510 0.6814 0.6567 0.7073 0.7729 0.6867 0.6464 0.3909 0.5053
RefCheck (Hu et al., 2024) 0.6912 0.6280 0.6736 0.7857 0.6800 0.7023 0.6014 0.3580 0.4628
P(True) (Kadavath et al., 2022) 0.7093 0.5194 0.5313 0.7998 0.5980 0.7032 0.6323 0.7083 0.6835
EigenScore (Chen et al., 2024) 0.6045 0.7469 0.6682 0.6640 0.6715 0.6637 0.6497 0.7078 0.6745
SEP (Han et al., 2024) 0.7143 0.7477 0.6627 0.8089 0.6580 0.7159 0.7004 0.7333 0.6915
SAPLMA (Azaria and Mitchell, 2023) 0.7037 0.5091 0.6726 0.8029 0.5053 0.6529 0.7092 0.5432 0.6718
ITI (Li et al., 2023) 0.7161 0.5416 0.6745 0.8051 0.5519 0.6838 0.6534 0.6850 0.6933
ReDeEP (Sun et al., 2025) 0.7458 0.8097 0.7190 0.8244 0.7198 0.7587 0.7285 0.7819 0.6947
SPAD 0.7839 0.8496 0.7218 0.86850.77780.7912 0.8148 0.7975 0.7975
Dolly (AC)
Method LLaMA2-7B LLaMA2-13B LLaMA3-8B
Metric AUC Recall F1 AUC Recall F1 AUC Recall F1
SelfCheckGPT (Manakul et al., 2023) — 0.1897 0.3188 0.2728 0.1897 0.3188 0.1095 0.2195 0.3600
Perplexity (Ren et al.) 0.2728 0.7719 0.7097 0.2728 0.7719 0.7097 0.1095 0.3902 0.4571
LN-Entropy (Malinin and Gales, 2021) 0.2904 0.7368 0.6772 0.2904 0.7368 0.6772 0.1150 0.5365 0.5301
Energy (Liu et al., 2020) 0.2179 0.6316 0.6261 0.2179 0.6316 0.6261 -0.0678 0.4047 0.4440
Focus (Zhang et al., 2023) 0.3174 0.5593 0.6534 0.1643 0.7333 0.6168 0.1266 0.6918 0.6874
Prompt (Niu et al., 2024) — 0.3965 0.5476 — 0.4182 0.5823 — 0.3902 0.5000
LMVLM (Cohen et al., 2023) — 0.7759 0.7200 — 0.7273 0.6838 — 0.6341 0.5361
ChainPoll (Friel and Sanyal, 2023) 0.3502 0.4138 0.5581 0.4758 0.4364 0.6000 0.2691 0.3415 0.4516
RAGAS (Es et al., 2024) 0.2877 0.5345 0.6392 0.2840 0.4182 0.5476 0.3628 0.8000 0.5246
Trulens (TrueLens, 2024) 0.3198 0.5517 0.6667 0.2565 0.3818 0.4941 0.3352 0.3659 0.5172
RefCheck (Hu et al., 2024) 0.2494 0.3966 0.5412 0.2869 0.2545 0.3944 -0.0089 0.1951 0.2759
P(True) (Kadavath et al., 2022) 0.1987 0.6350 0.6509 0.2009 0.6180 0.5739 0.3472 0.5707 0.6573
EigenScore (Chen et al., 2024) 0.2428 0.7500 0.7241 0.2948 0.8181 0.7200 0.2065 0.7142 0.5952
SEP (Han et al., 2024) 0.2605 0.6216 0.7023 0.2823 0.6545 0.6923 0.0639 0.6829 0.6829
SAPLMA (Azaria and Mitchell, 2023) 0.0179 0.5714 0.7179 0.2006 0.6000 0.6923 -0.0327 0.4040 0.5714
ITI (Li et al., 2023) 0.0442 0.5816 0.6281 0.0646 0.5385 0.6712 0.0024 0.3091 0.4250
ReDeEP (Sun et al., 2025) 0.5136 0.8245 0.7833 0.5842 0.8518 0.7603 0.3652 0.83920.7100
SPAD 0.65140.7931 0.7541 0.7848 0.9444 0.7907 0.77170.70730.7733
Table 1: Results on RAGTruth and Dolly (AC) datasets across three LLaMA models. We report AUC, Recall, and
F1-score. Bold values indicate the best performance and underlined values indicate the second-best.
Llama2-13B model, our approach sweeps all three
metrics with a commanding lead, achieving the
highest AUC (0.7848), Recall (0.9444), and F1-
score (0.7907), significantly outperforming previ-
ous state-of-the-art baselines. Similarly, on the
Llama3-8B model, our method reinforces its supe-
riority, securing the best-in-class F1-score (0.7733)
and a dominant AUC (0.7717). For the Llama2-7B
model, while simpler baselines like ReDeEP yield
a slightly higher F1-score, our method retains the
highest AUC (0.6514). This suggests that while
determining the optimal classification threshold is
inherently difficult with extremely limited trainingsamples, our framework consistently captures the
most discriminative features for ranking truthful
and hallucinated responses.
4.4 Interpretability Analysis
Beyond performance metrics, we seek to under-
stand the mechanistic logic driving detection. We
apply SHAP analysis to the classifiers trained on
the RAGTruth benchmark. The Dolly dataset is
excluded due to its small size. Three observations
are obtained from this analysis.
1. The syntax of grounding varies by architec-
ture.While RAG attribution is universally critical

(a) Llama2-7B
(b) Llama2-13B
(c) Llama3-8B
Figure 3: SHAP summary plots illustrating the deci-
sion logic. We visualize the top-10 features for classi-
fiers trained on the RAGTruth subsets corresponding
to Llama2-7B, Llama2-13B, and Llama3-8B. The x-
axis represents the SHAP value, where positive values
indicate a push towards classifying the response as a
Hallucination. The color represents the feature value
(Red = High attribution, Blue = Low).
for factuality, the specific POS tags carrying this
signal change. Llama2 models (7B/13B) rely pri-
marily on content words, with RAG_NOUN being the
top predictor. In contrast, Llama3-8B relies on re-lational structures, with RAG_ADP (Adpositions like
"by") emerging as the most discriminative feature.
2. LayerNorm on Numerals is a critical but flip-
flopping signal.The Final LayerNorm ( LN) plays a
major role in numerical reasoning, but its effect re-
verses across models. In Llama2-7B, high LN_NUM
attribution acts as a warning sign for hallucination.
However, in Llama2-13B, high LN_NUM indicates
factuality. This difference shows that detection
must capture model-specific behaviors.
3. The User Query is an overlooked but criti-
cal hallucination driver.Query-based features
frequently rank among the top predictors, chal-
lenging the traditional focus on just RAG and
FFNs. In Llama2-13B and Llama3-8B, features
likeQUERY_ADJ andQUERY_NOUN appear in the top-
3 most important features. This result shows that
monitoring the model’s reliance on the prompt is
as vital as monitoring the retrieved context.
5 Limitations
Our framework presents three limitations. First, it
relies onwhite-box accessto model internals, pre-
venting application to closed-source APIs. Second,
the decomposition incurs highercomputational
overheadthan simple scalar probes. However, it
remains significantly more efficient than sampling-
based or LLM-as-a-Judge methods as our method
requires only a single forward pass. Third, our
feature engineering depends onexternal linguistic
tools(POS taggers), which may limit generaliza-
tion to specialized domains like code generation
where standard syntax is less defined.
6 Conclusion and Future Work
We introduced SPAD to attribute token probability
into seven distinct sources. By combining these
with syntax-aware features, our framework effec-
tively detects RAG hallucinations and outperforms
baselines. Our results show that hallucination sig-
nals vary across models. This confirms the need for
a learnable approach rather than static heuristics.
Our future work will focus on two directions.
First, we plan to extend our attribution framework
to semantic phrases to achieve more efficient and
effective detection. Second, we aim to implement
active hallucination mitigation by monitoring con-
tributions in real time to suppress risky sources and
correct the generation process.

Acknowledgements
This work was supported by the Australian Re-
search Council through the Laureate Fellow Project
under Grant FL190100149.
References
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru
Ohta, and Masanori Koyama. 2019. Optuna: A next-
generation hyperparameter optimization framework.
InProceedings of the 25th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery and Data
Mining.
Amos Azaria and Tom Mitchell. 2023. The internal
state of an llm knows when it’s lying. InFindings
of the Association for Computational Linguistics:
EMNLP 2023, pages 967–976.
Collin Burns, Haotian Ye, Dan Klein, and Jacob Stein-
hardt. 2022. Discovering latent knowledge in lan-
guage models without supervision.arXiv preprint
arXiv:2212.03827.
Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu,
Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024.
Inside: Llms’ internal states retain the power of hal-
lucination detection.
Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A
scalable tree boosting system. InProceedings of
the 22nd acm sigkdd international conference on
knowledge discovery and data mining, pages 785–
794.
Yuyan Chen, Zehao Li, Shuangjie You, Zhengyu Chen,
Jingwen Chang, Yi Zhang, Weinan Dai, Qingpei Guo,
and Yanghua Xiao. 2025. Attributive reasoning for
hallucination diagnosis of large language models. In
Proceedings of the AAAI Conference on Artificial
Intelligence, volume 39, pages 23660–23668.
Roi Cohen, May Hamri, Mor Geva, and Amir Glober-
son. 2023. Lm vs lm: Detecting factual errors via
cross examination. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 12621–12640.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158.
Robert Friel and Atindriyo Sanyal. 2023. Chainpoll: A
high efficacy method for llm hallucination detection.
arXiv preprint arXiv:2310.18344.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.Jiatong Han, Jannik Kossen, Muhammed Razzak, Lisa
Schut, Shreshth A Malik, and Yarin Gal. 2024. Se-
mantic entropy probes: Robust and cheap hallucina-
tion detection in llms. InICML 2024 Workshop on
Foundation Models in the Wild.
Matthew Honnibal, Ines Montani, Sofie Van Lan-
deghem, Adriane Boyd, and 1 others. 2020. spacy:
Industrial-strength natural language processing in
python.
Xiangkun Hu, Dongyu Ru, Lin Qiu, Qipeng Guo,
Tianhang Zhang, Yang Xu, Yun Luo, Pengfei Liu,
Yue Zhang, and Zheng Zhang. 2024. Refchecker:
Reference-based fine-grained hallucination checker
and benchmark for large language models.arXiv
preprint arXiv:2405.14486.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, and 1 others. 2022. Language mod-
els (mostly) know what they know.arXiv preprint
arXiv:2207.05221.
Hakyung Lee, Keon-Hee Park, Hoyoon Byun, Jeyoon
Yeom, Jihee Kim, Gyeong-Moon Park, and Kyung-
woo Song. 2024. Ced: Comparing embedding dif-
ferences for detecting out-of-distribution and halluci-
nated text. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pages 14866–
14882.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter
Pfister, and Martin Wattenberg. 2023. Inference-
time intervention: Eliciting truthful answers from
a language model.Advances in Neural Information
Processing Systems, 36:41451–41530.
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan
Li. 2020. Energy-based out-of-distribution detection.
Advances in neural information processing systems,
33:21464–21475.
Andrey Malinin and Mark Gales. 2021. Uncertainty
estimation in autoregressive structured prediction. In
International Conference on Learning Representa-
tions.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.
Selfcheckgpt: Zero-resource black-box hallucination
detection for generative large language models. In
Proceedings of the 2023 conference on empirical
methods in natural language processing, pages 9004–
9017.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2024. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 10862–10878.
nostalgebraist. 2020. interpreting gpt: the logit lens.
LessWrong.
F. Pedregosa, G. Varoquaux, A. Gramfort, V . Michel,
B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer,
R. Weiss, V . Dubourg, J. Vanderplas, A. Passos,
D. Cournapeau, M. Brucher, M. Perrot, and E. Duch-
esnay. 2011. Scikit-learn: Machine learning in
Python.Journal of Machine Learning Research,
12:2825–2830.
Abhilasha Ravichander, Shrusti Ghela, David Wadden,
and Yejin Choi. 2025. Halogen: Fantastic llm hal-
lucinations and where to find them.arXiv preprint
arXiv:2501.08292.
Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mo-
hammad Saleh, Balaji Lakshminarayanan, and Pe-
ter J Liu. Out-of-distribution detection and selective
generation for conditional language models. InThe
Eleventh International Conference on Learning Rep-
resentations.
Hsuan Su, Ting-Yao Hu, Hema Swetha Koppula, Kun-
dan Krishna, Hadi Pouransari, Cheng-Yu Hsieh,
Cem Koc, Joseph Yitan Cheng, Oncel Tuzel, and
Raviteja Vemulapalli. 2025. Learning to reason
for hallucination span detection.arXiv preprint
arXiv:2510.02173.
Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Jun Xu,
Xiao Zhang, Weijie Yu, Yang Song, and Han Li.
2025. Redeep: Detecting hallucination in retrieval-
augmented generation via mechanistic interpretabil-
ity. InICLR.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, and 1 others. 2023. Llama 2: Open foun-
dation and fine-tuned chat models.arXiv preprint
arXiv:2307.09288.
TrueLens. 2024. Truelens: Evaluate and track llm ap-
plications.
Yakir Yehuda, Itzik Malkiel, Oren Barkan, Jonathan
Weill, Royi Ronen, and Noam Koenigstein. 2024.
Interrogatellm: Zero-resource hallucination detection
in llm-generated answers. InProceedings of the 62nd
Annual Meeting of the Association for ComputationalLinguistics (Volume 1: Long Papers), pages 9333–
9347.
Fujie Zhang, Peiqi Yu, Biao Yi, Baolei Zhang, Tong Li,
and Zheli Liu. 2025. Prompt-guided internal states
for hallucination detection of large language models.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 21806–21818.
Tianhang Zhang, Lin Qiu, Qipeng Guo, Cheng Deng,
Yue Zhang, Zheng Zhang, Chenghu Zhou, Xinbing
Wang, and Luoyi Fu. 2023. Enhancing uncertainty-
based hallucination detection with stronger focus. In
Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages 915–
932.
A Proof of Theorem 1
We expand the right-hand side (RHS) of Eq. 8 by
substituting the definitions of each term.
First, consider the summation term. By substitut-
ing∆P(l)
attand∆P(l)
ffn, the intermediate probe value
Φ(H(l)
mid)cancels out within each layer:
LX
l=1
∆P(l)
att+ ∆P(l)
ffn
=LX
l=1
[Φ(H(l)
mid)−Φ(H(l−1))] + [Φ(H(l))−Φ(H(l)
mid)]
=LX
l=1
Φ(H(l))−Φ(H(l−1))
This summation forms a telescoping series where
adjacent terms cancel:
LX
l=1(Φ(H(l))−Φ(H(l−1)))
= (Φ(H(1))−Φ(H(0))) +. . .
+ (Φ(H(L))−Φ(H(L−1)))
= Φ(H(L))−Φ(H(0))
Now, substituting this result, along with the defini-
tions of Pinitial and∆P LN, back into the full RHS
expression:
RHS= Φ(H(0))|{z}
Pinitial+ (P final−Φ(H(L)))| {z }
∆P LN
+ (Φ(H(L))−Φ(H(0)))| {z }
Summation=P final
The RHS simplifies exactly to Pfinal(y), which com-
pletes the proof.

B Proof of Proposition 1
Proof. We consider the l-th layer of the Trans-
former. Let H(l−1)∈Rdbe the input hidden state.
The Multi-Head Attention mechanism computes a
residual update ∆H by summing the outputs of H
heads:
∆H=HX
h=1hh (17)
where hh∈Rdis the projected output of the h-th
head.
The probe function Φ(H, y) computes the proba-
bility of the target token yby projecting the hidden
state onto the vocabulary logits z∈RVand apply-
ing the Softmax function:
Φ(H, y) =exp(z y)PV
v=1exp(z v),wherez v=H·w e,v
(18)
Here, we,vis the unembedding vector for to-
kenvfrom matrix We. For brevity, let py=
Φ(H(l−1), y)denote the probability of the target
token at the current state.
We approximate the change in probability,
∆P(l)
att, using a first-order Taylor expansion of Φ
with respect toHaroundH(l−1):
∆P(l)
att= Φ(H(l−1)+ ∆H, y)−Φ(H(l−1), y)
≈ ∇ HΦ(H(l−1), y)⊤·∆H
=HX
h=1
∇HΦ(H(l−1), y)⊤·hh
(19)
To compute the gradient ∇HΦ, we apply the
chain rule through the logits zv. The partial deriva-
tive of the Softmax output pywith respect to any
logit zvis given by py(δyv−pv), where δis the
Kronecker delta. The gradient of the logit zvwith
respect toHis simplyw e,v. Thus:
∇HΦ =VX
v=1∂Φ
∂zv∂zv
∂H
=VX
v=1py(δyv−pv)we,v
=py(1−p y)we,y−X
v̸=ypypvwe,v(20)
Substituting this gradient back into Eq. (19) for aspecific head contribution term (denoted as Term h):
Term h=∇ HΦ⊤·hh
=py(1−p y)|{z}
G(l)(w⊤
e,y·hh)−X
v̸=ypypv(w⊤
e,v·hh)
| {z }
Eh
(21)
We observe that the dot product w⊤
e,y·hhis strictly
equivalent to the scalar logit contribution ∆z(l)
h,y
defined in Eq. (7). The factor G(l)=py(1−p y)
represents the gradient common to all heads, de-
pending only on the layer input H(l−1). Therefore,
the contribution of head his dominated by the lin-
ear term G(l)·∆z(l)
h,y, subject to the off-target error
termE h.
C Implementation Details
Environment and Models.All experiments
were conducted on a computational node equipped
with an NVIDIA A100 GPU and 200GB of RAM.
We evaluate our framework using three Large Lan-
guage Models: Llama2-7b-chat, Llama2-13b-chat
(Touvron et al., 2023), and Llama3-8b-instruct
(Grattafiori et al., 2024). To extract the internal
states required for attribution, we re-process the
model-generated responses in a teacher-forcing
manner.
Feature Extraction and Classifier.For each re-
sponse, we extract the 7-dimensional attribution
vector for every token and aggregate them based
on 18 universal POS tags (e.g., NOUN, VERB)
defined by the SpaCy library (Honnibal et al.,
2020). This results in a fixed-size feature vector
(7×18 = 126 dimensions) for each sample. We
employ XGBoost (Chen and Guestrin, 2016) as the
binary classifier for hallucination detection due to
its robustness on tabular data.
Training and Evaluation Protocols.To ensure
fair comparison and rigorous evaluation, we tailor
our training strategies to the data availability of
each dataset. We implement strict data isolation
protocols to prevent any form of data leakage, par-
ticularly for scenarios where official training splits
are unavailable.
Protocol I: Standard Split (RAGTruth Llama2-
7b/13b).For these datasets, we utilize the official
train/test splits provided by prior work (Sun et al.,
2025).

Algorithm 1SPAD Part I: Token-by-Token Generation and Probability Decomposition
Require:TransformerM, Query Tokensx qry, Retrieved Context Tokensx rag
Ensure:Sequence of 7-source attribution vectorsV= (v 1, . . . ,v T)
1:Initialize contextC←[x qry,xrag]and attribution storageV ← ∅
2:forstept= 1,2, . . .until EOSdo
3:1. Forward Pass & State Caching
4:Predict next tokeny tusing contextC.
5:// Crucial: Save all intermediate residual states and attention maps for attribution
6:CacheH(0),H(l)
mid,H(l), and Attention MapsA(l)for all layersl∈[1, L].
7:2. Coarse Decomposition (Residual Stream)
8:Initializev t∈R7with zeros.
9:v t[Init]←Probe(H(0), yt){Contribution from initial embedding}
10:v t[LN]←P final(yt)−Probe(H(L), yt){Contribution from final LayerNorm}
11:3. Layer-wise Attribution
12:forlayerl= 1toLdo
13:// Calculate FFN contribution by probing states before and after FFN block
14:v t[FFN] +=Probe(H(l), yt)−Probe(H(l)
mid, yt)
15:// Calculate Total Attention contribution for this layer
16:∆P att←Probe(H(l)
mid, yt)−Probe(H(l−1), yt)
17:4. Fine-Grained Decomposition (Head & Source)
18:forheadh= 1toHdo
19:// a. Head Attribution: How much did this head contribute to the target logit?
20:Leto hbe the output vector of headh.
21:Compute logit update:∆z h←o h·UnbeddingVector(y t)
22:Compute ratioω h←exp(∆z h)P
jexp(∆z j){Logit-based apportionment}
23:∆P h←∆P att×ωh
24:// b. Source Mapping: Where did this head look?
25:forsourceS∈ {Qry,RAG,Past,Self}do
26:Sum attention weights on indices ofS:W h,S←P
k∈ISA(l)
h[t, k]
27:Normalize:α S←W h,S/P
all sources Wh,·
28:v t[S] += ∆P h×αS
29:end for
30:end for
31:end for
32:Appendv ttoVand update contextC←[C, y t].
33:end for
34:returnV,Generated Tokensy
•Optimization:We perform hyperparameter
optimization strictly on the training set using
RandomizedSearchCV implemented in Scikit-
learn (Pedregosa et al., 2011). We search
for 50 iterations with an internal 5-fold cross-
validation to maximize the F1-score.
•Training:The final model is trained on 85%
of the training data using the best hyperparam-
eters, with the remaining 15% serving as a
validation set for early stopping (patience=50)
to prevent overfitting.•Evaluation:Performance is reported on the
held-out official test set.
Protocol II: Stratified 20-Fold CV (RAGTruth
Llama3-8b).As only test samples are publicly
available for the Llama3-8b subset, we adopt a rig-
orous Stratified 20-Fold Cross-Validation scheme
to maximize statistical power while maintaining
evaluation integrity.
•Strict Isolation:In each of the 20 iterations,
the dataset is partitioned into 95% training
and 5% testing. Crucially, hyperparameter

Algorithm 2SPAD Part II: Syntax-Aware Feature Aggregation with Sub-word Tag Propagation
Require:Generated Tokensy= (y 1, . . . , y T), Attribution VectorsV= (v 1, . . . ,v T)
Ensure:Syntax-Aware Feature Vectorf∈R126
1:1. String Reconstruction & Alignment Map
2:Decode tokensyinto a complete string textS.
3:Construct an alignment map Mwhere M[i] contains the list of token indices corresponding to the
i-th word inS.
4:// Example: "modi" (idx 5), "fication" (idx 6)→Word "modification" (idxw); soM[w] = [5,6]
5:2. POS Tagging & Propagation
6:Initialize tag listTof lengthT.
7:Run POS tagger (e.g., SpaCy) on stringSto obtain wordsW 1, . . . , W Kand tags tag1, . . . ,tagK.
8:foreach word indexk= 1toKdo
9:Get corresponding token indices:I tokens←M[k]
10:Get POS tag for the word:c←tagk
11:foreach token indext∈ I tokens do
12:τ t←c{Propagate parent word’s tag to all sub-word tokens}
13:end for
14:end for
15:3. Syntax-Aware Aggregation
16:Initialize feature vectorf← ∅.
17:Define set of Universal POS tagsP univ.
18:foreach POS categoryc∈ P univdo
19:Identify tokens belonging to this category:I c={t|τ t=c}
20:ifI c̸=∅then
21:// Compute mean attribution profile for this syntactic category
22: ¯vc←1
|Ic|P
t∈Icvt
23:else
24: ¯vc←0 7{Fill with zeros if category is absent in response}
25:end if
26:f←Concatenate(f, ¯vc)
27:end for
28:returnf
optimization is performed anew foreach fold
using only that fold’s training partition.
•Aggregation:The final metrics are calculated
by micro-averaging the predictions across all
20 held-out test folds, ensuring every sample
is evaluated exactly once as an unseen test
instance.
Protocol III: Nested Leave-One-Out CV (Dolly).
To ensure a fair comparison with the benchmark es-
tablished by (Sun et al., 2025), we strictly adhere to
their experimental setting, which relies exclusively
on a curated test set of 100 samples. Given the lim-
ited size of this specific evaluation split, we imple-
ment aNested Leave-One-Out Cross-Validation
to ensure statistical robustness.
•Outer Loop (Evaluation):We iterate 100times. In each iteration, a single sample is
strictly isolated as the test case.
•Inner Loop (Optimization):On the remain-
ing 99 samples, we conduct an independent
Bayesian hyperparameter search using Op-
tuna (Akiba et al., 2019) for 50 trials. This
inner loop uses its own 5-fold cross-validation
to find the optimal configuration.
•Class Imbalance:To address the label im-
balance in Dolly, we dynamically adjust the
scale_pos_weight parameter in XGBoost
based on the class distribution of the training
fold.
•Inference:A model is trained on the 99 sam-
ples using the best parameters found in the in-
ner loop and evaluated on the single held-out

sample. This process is repeated for all 100
samples to aggregate the final performance
metrics.
D Baselines Introduction
1.EigenScore/INSIDE(Chen et al., 2024) Fo-
cus on detecting hallucination by evaluating
response’s semantic consistency, which is de-
fined as the logarithm determinant of conva-
riance matrix LLM’s internal states during
generating the response.
2.SEP(Han et al., 2024) Proposed a linear
model to detect hallucination based on seman-
tic entropy in test time whithout requiring mul-
tiple responses.
3.SAPLMA(Azaria and Mitchell, 2023) Detect-
ing hallucination based on the hidden layer
activations of LLMs.
4.ITI(Li et al., 2023) Detecting hallucination
based on the hidden layer activations of
LLMs.
5.Ragtruth Prompt(Niu et al., 2024) Provdes
prompts for a LLM-as-judge to detect halluci-
nation in RAG setting.
6.LMvLM(Cohen et al., 2023) It uses prompt
to conduct a multiturn interaction between an
examiner LLM and exainee LLM to reveal
inconsistencies which implies hallucination.
7.ChainPoll(Friel and Sanyal, 2023) Provdes
prompts for a LLM-as-judge to detect halluci-
nation in RAG setting.
8.RAGAS(Es et al., 2024) It use a LLM to split
the response into a set of statements and verify
each statement is supported by the retrieved
documents. If any statement is not supported,
the response is considered hallucinated.
9.Trulens(TrueLens, 2024) Evaluating the over-
lap between the retrieved documents and the
generated response to detect hallucination by
a LLM.
10.P(True)(Kadavath et al., 2022) The paper de-
tects hallucinations by having the model es-
timate the probability that its own generated
answer is correct, based on the key assumption
that it is often easier for a model to recognize
a correct answer than to generate one.11.SelfCheckGPT(Manakul et al., 2023) Self-
CheckGPT detects hallucinations by checking
for informational consistency across multiple
stochastically sampled responses, based on
the assumption that factual knowledge leads
to consistent statements while hallucinations
lead to divergent and contradictory ones.
12.LN-Entropy(Malinin and Gales, 2021) This
paper detects hallucinations by quantifying
knowledge uncertainty, which it measures pri-
marily with a novel metric called Reverse Mu-
tual Information that captures the disagree-
ment across an ensemble’s predictions, with
high RMI indicating a likely hallucination.
13.Energy(Liu et al., 2020) This paper detects
hallucinations by using an energy score, de-
rived directly from the model’s logits, as a
more reliable uncertainty measure than soft-
max confidence to identify out-of-distribution
inputs that cause the model to hallucinate.
14.Focus(Zhang et al., 2023) This paper detects
hallucinations by calculating an uncertainty
score focused on keywords, and then refines it
by propagating penalties from unreliable con-
text via attention and correcting token prob-
abilities using entity types and inverse doc-
ument frequency to mitigate both overconfi-
dence and underconfidence.
15.Perplexity(Ren et al.) This paper detects hal-
lucinations by separately measuring the Rela-
tive Mahalanobis Distance for both input and
output embeddings, based on the assumption
that in-domain examples will have embed-
dings closer to their respective foreground (in-
domain) distributions than to a generic back-
ground distribution.
16.REFCHECKER(Hu et al., 2024) It use a
LLM to extract claim-triplets from a response
and verify them by another LLM to detect
hallucination.
17.REDEEP(Sun et al., 2025) It detects halluci-
nation by analyzing the balance between the
contributions from Copying Heads that pro-
cess external context and Knowledge FFNs
that inject internal knowledge, based on the
finding that RAG hallucinations often arise
from conflicts between these two sources.
This method has two version: token level and

chunk level. We compare with the latter since
it has better performance generally.
E Complexity Analysis of the Attribution
Process
In this section, we rigorously analyze the computa-
tional overhead of our attribution framework. We
focus strictly on the attribution extraction process
for a generated response of length T. Let L,d,
V, and Hdenote the number of layers, hidden di-
mension, vocabulary size, and attention heads (per
layer), respectively. The standard inference com-
plexity for a Transformer is O(L·T·d2+L·H·T2).
Our attribution process introduces post-hoc compu-
tations, decomposed into three specific stages:
1. Exact Probability Decomposition.To satisfy
Theorem 1, we must compute the exact probability
changes using the probe function Φ(h, y) . The
bottleneck is the calculation of the global partition
function (denominator) in Softmax.
•Mechanism:The probe function Φ(h, y) =
Softmax(W uh)y=exp(w⊤
u,yh)PV
v=1exp(w⊤u,vh)requires
projecting the hidden state hto the full vocab-
ulary logitsz=W uh.
•Single Probe Complexity:For a single hid-
den state h∈Rd, the matrix-vector multipli-
cation with the unembedding matrix Wu∈
RV×dcostsO(V·d).
•Total Calculation:We must apply this probe
at multiple points:
1.Global Components:For Pinitial and
∆P LN, the probe is called once per gen-
eration step. Cost:O(T·V·d).
2.Layer Components:For ∆P(l)
attand
∆P(l)
ffn, the probe is invoked twice per
layer (before and after the residual up-
date). Summing over Llayers, this costs
O(L·T·V·d).
•Stage Complexity:Combining these terms,
the dominant complexity isO(L·T·V·d).
2. Head-wise Attribution.Once ∆P(l)
attis ob-
tained, we apportion it to individual heads based
on their contribution to the target logit.
•Mechanism:This attribution requires project-
ing the target token embedding we,yback into
the hidden state space using the layer’s output
projection matrixW O∈Rd×d.•Step Complexity:The calculation proceeds
in two sub-steps:
1.Projection:We compute the projected
target vector g=W⊤
Owe,y. Since WO
is ad×d matrix, this matrix-vector mul-
tiplication costsO(d2).
2.Assignment:We distribute the contribu-
tion to Hheads by performing dot prod-
ucts between the head outputs ohand
the corresponding segments of g. ForH
heads, this sums toO(d).
•Stage Complexity:The projection step
(O(d2)) dominates the assignment step
(O(d) ). Integrating over Llayers and Tto-
kens, the total complexity isO(L·T·d2).
3. Mapping Attention to Input Sources.Fi-
nally, we map head contributions to the four
sources by aggregating attention weights A∈
RH×T×T. This involves two distinct sub-steps for
each generated token at step twithin a single layer:
•Step 1: Summation.For each head h, we
sum the attention weights corresponding to
specific source indices (e.g.,I RAG):
wh,S=X
k∈ISAh[t, k]
This requires iterating over the context length
t. ForHheads, the cost isO(H·t).
•Step 2: Normalization & Weighting.We cal-
culate the final source contribution by weight-
ing the head contributions:
∆PS=HX
h=1∆Ph·wh,SP
all sources wh
This involves scalar operations proportional
to the number of headsH. Cost:O(H).
•Stage Complexity:The summation step
(O(H·t) ) dominates. We sum this cost across
allLlayers, and then accumulate over the gen-
eration steps t= 1 toT. The calculation isPT
t=1(L·H·t)≈ O(L·H·T2).
Overall Efficiency.The total computational cost
is the sum of these three stages:
Ctotal=O(L·T·V·d|{z}
Prob. Decomp.+L·T·d2
|{z}
Head Attr.+L·H·T2
|{z}
Source Map)

Runtime Efficiency.It is worth noting that theo-
retical complexity does not directly equate to wall-
clock latency. Standard text generation isserial
(token-by-token), which limits GPU paralleliza-
tion. In contrast, our framework process the full
sequence of length Tin a single parallel pass. This
allows us to leverage efficient matrix operations on
GPUs. Consequently, our single analysis pass is
significantly faster in practice than baselines like
SelfCheckGPT, which require running the slow se-
rial generation processKtimes.
# DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance

**Authors**: Tingqiao Xu, Shaowei Yao, Chenhe Dong, Yiming Jin, Zerui Huang, Dan Ou, Haihong Tang

**Published**: 2025-10-13 08:08:59

**PDF URL**: [http://arxiv.org/pdf/2510.11122v1](http://arxiv.org/pdf/2510.11122v1)

## Abstract
Accurately modeling query-item relevance drives e-commerce ranking, yet
long-tail, knowledge-heavy, and fast-evolving queries exceed parametric LLM
coverage. External context (reviews, attribute encyclopedias, UGC) can help but
is noisy, and single-pass latency and cost forbid any clean-then-summarize
step. The model must, per query, judge relevance and decide whether to use,
partially use, or ignore the context. DyKnow-RAG is a dynamic noisy-RAG
framework built on Group Relative Policy Optimization. It trains two rollout
groups (no external context vs a single retrieved chunk) and applies
posterior-driven inter-group advantage scaling that adaptively reweights their
contributions by the per-query correctness gap. This teaches when to trust
retrieval versus fall back to parametric knowledge, without process labels,
value networks, or extra inference passes, preserving single-pass, single-chunk
deployment under production latency. Training combines: (1) supervised
initialization with a structured rationale that explicitly records the
context-usage decision; (2) an RL pool prioritized by SFT uncertainty to focus
where context choice is most consequential; and (3) an optional lightweight DPO
warm start to stabilize with-context calibration. Under a unified
retrieval/index and fixed latency budget, DyKnow-RAG outperforms SFT, DPO, and
vanilla GRPO in offline tests, and delivers consistent lifts on GSB, Query
Goodrate, and Item Goodrate in Taobao A/B testing. It is deployed in Taobao's
production relevance system, serving live traffic. To our knowledge, it is
among the first single-pass RAG solutions for e-commerce relevance, turning
noisy external signals into reliable gains without added online complexity.

## Full Text


<!-- PDF content starts -->

DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement
Framework for Noisy Retrieval-Augmented Generation in
E-commerce Search Relevance
Tingqiao Xu
Shanghai Jiao Tong University
Shanghai, China
phenomenonkj@sjtu.edu.cnShaowei Yao
Taobao & Tmall Group of Alibaba
Hangzhou, China
yaoshaowei@taobao.comChenhe Dong
Taobao & Tmall Group of Alibaba
Hangzhou, China
dongchenhe.dch@alibaba-inc.com
Yiming Jin
Taobao & Tmall Group of Alibaba
Hangzhou, China
enxian@alibaba-inc.comZerui Huang
Taobao & Tmall Group of Alibaba
Hangzhou, China
huangzerui.hzr@taobao.comDan Ou
Taobao & Tmall Group of Alibaba
Hangzhou, China
oudan.od@taobao.com
Haihong Tang
Taobao & Tmall Group of Alibaba
Hangzhou, China
piaoxue@taobao.com
Abstract
Accurately estimating query–item relevance is central to e-commerce
search, directly driving ranking quality, conversion, and long-term
user value. While large language models offer a new paradigm
for relevance modeling, many long-tail, knowledge-intensive, and
fast-evolving queries lie beyond the coverage of purely parametric
models, motivating the use of external context (e.g., community
reviews, attribute encyclopedias, UGC). In practice, however, this
external context is rife with promotional phrasing, tag clutter, and
affective language; the signal-to-noise ratio fluctuates markedly
across queries. Meanwhile, single-turn latency and cost constraints
preclude any extra clean-then-summarize stage at inference. The
model must therefore, in a single pass, both judge relevance and
dynamically decide whether to adopt, partially adopt, or ignore the
external context—misjudgment risks being misled by noise or miss-
ing real gains. We present DyKnow-RAG, a dynamic knowledge
utilization reinforcement framework for noisy RAG in e-commerce
relevance. Built on Group Relative Policy Optimization (GRPO),
DyKnow-RAG trains two rollout groups—no external context and
with a single external context chunk—and introduces posterior-
driven inter-group advantage scaling that adaptively reweights
their training contributions based on the per-query correctness gap.
This teaches the policy when to rely on external context versus fall
back to parametric knowledge, without process labels, value net-
works, or extra inference passes, and preserves single-pass, single-
chunk deployment. The training pipeline combines supervised ini-
tialization with a structured chain of thought that explicitly records
the external context–usage decision, builds an RL pool prioritized
by SFT uncertainty to focus on instances where context use is most
consequential, and optionally applies a lightweight DPO warm start
to stabilize with-context calibration. Under a unified retrieval/index
version and latency budget, DyKnow-RAG yields substantial im-
provements over SFT, DPO, and vanilla GRPO in offline evaluations,
and delivers consistent lifts on key business metrics (GSB, Query
Goodrate, Item Goodrate) in controlled A/B tests on Taobao traffic.DyKnow-RAG has been deployed in Taobao’s production relevance
system and is serving live traffic. To our knowledge, this is one of the
first work to operationalize external context (RAG) for e-commerce
search relevance under single-pass, production latency constraints.
Overall, it turns noisy external context into reliable gains with-
out sacrificing online simplicity, offering a deployable and scalable
training paradigm for relevance in large-scale e-commerce search.
CCS Concepts
•Do Not Use This Code →Generate the Correct Terms for
Your Paper;Generate the Correct Terms for Your Paper; Generate
the Correct Terms for Your Paper; Generate the Correct Terms for
Your Paper.
Keywords
E-commerce Relevance Search, Large Language Models, Retrieval-
Augmented Generation, Reinforce Learning
1 Introduction
In large-scale e-commerce search (e.g., Taobao, Amazon), the system
must rapidly surface relevant products from a massive catalog
for a vast user base. The core objective is accurate query–item
relevance estimation, which underpins ranking quality, conversion,
and long-term user value. Serving irrelevant or misleading items
degrades user experience and undermines merchant operations,
making reliable relevance prediction a cornerstone of e-commerce
search engines.
Retrieval-Augmented Generation (RAG) extends purely para-
metric models by adding external context such as community re-
views, product evaluations, and attribute encyclopedias. This helps
address long-tail, knowledge-intensive, and fast-evolving queries.
In practice, however, external context often includes promotional
phrasing, tag clutter, and affective language, and its signal quality
varies widely across queries. Single-pass inference within produc-
tion does not permit an extra clean-then-summarize stage. WithoutarXiv:2510.11122v1  [cs.IR]  13 Oct 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
an on-the-fly policy for using the context, models can be misled
by noise or miss genuine gains. To our knowledge, this is the first
work that operationalizes external context (RAG) for e-commerce
search relevance under single-pass, production latency constraints.
Focusing on single-pass e-commerce relevance, we tackle three
challenges: (C1) noise and heterogeneity in external context, which
calls for suppressing irrelevant or misleading spans and extracting
decision-critical support; (C2) latency and cost constraints, which
rule out multi-stage summarize-then-use procedures and require
benefits without extra inference passes; (C3) the need for the model
to judge relevance and to decide whether to adopt, partially adopt,
or ignore the context, since a wrong choice either amplifies noise
or discards value.
We address these constraints with DyKnow-RAG: Dynamic
Knowledge Utilization Reinforcement for noisy RAG. Built on Group
Relative Policy Optimization (GRPO)[ 15], DyKnow-RAG trains two
rollout groups: no external context (parametric only) and with a
single context chunk. It introduces posterior-driven inter-group
advantage scaling that adaptively reweights their contributions
based on the per-query correctness gap. The policy learns when to
rely on external context and when to fall back to parametric knowl-
edge. The design uses no process labels or value network and adds
no extra passes at inference, preserving single-pass, single-chunk
deployment.
Our production -aligned pipeline adds three components: (1) su-
pervised fine -tuning (SFT) with structured chain -of-thought that
records whether/why the single context chunk is used, providing
concise rationales for reinforcement learning; (2) an uncertainty
prioritized RL pool from SFT posteriors that focuses updates on
borderline cases; and (3) an optional DPO warm start under the
with -context prompt to stabilize the with -context rollouts. Un-
der a unified retrieval and fixed latency budget, we benchmark
against strong SFT, DPO, and vanilla GRPO baselines, validate of-
fline and via controlled online A/B tests, and deploy DyKnow -RAG
in Taobao’s production relevance system.
Our contributions are threefold:
•A production-aligned training pipeline:We design a uni-
fied pipeline with structured CoT that explicitly records whether
and why external context is adopted, and use SFT posterior log-
its to focus RL on contentious cases; single-pass, single-chunk
inference is preserved end-to-end.
•DyKnow-RAG: dynamic knowledge utilization reinforce-
ment:We introduce a posterior-driven, adaptive inter-group ad-
vantage scaling over GRPO rollouts (no-chunk vs. with-chunk),
enabling label-free learning of adopt/partial/ignore decisions
under noisy external context without extra inference passes.
•Offline and online improvements:To our knowledge, we
first operationalize external context for e-commerce search rel-
evance under production latency constraints; it is deployed in
Taobao’s production relevance stack and shows substantial of-
fline improvements and consistent online lifts on key business
metrics.2 Related Work
2.1 Search Relevance
Search relevance has evolved from lexical matching to semantic rea-
soning over several decades. Early systems relied on hand-crafted
statistical scoring such as TF-IDF [ 1] and BM25 [ 13], which were
effective in classic information retrieval systems but limited in cross-
domain generalization and nuanced semantics. With deep learn-
ing, the focus shifted to semantic matching: representation-based
models (e.g., DSSM [ 6]) reduced manual feature engineering, and
pre-trained Transformers/BERT [ 4,19] enabled richer semantic un-
derstanding. More recently, large language models (LLMs) reframe
relevance as a reasoning task. E-commerce–oriented frameworks
such as LREF [ 18], TaoSR1 [ 5], ProRBP [ 3], and related methods
align LLMs via supervised fine-tuning (SFT) and direct preference
optimization (DPO) [ 12], prompting models to articulate why an
item is relevant and to attribute decisions to textual evidence. De-
spite these advances, many approaches remain largely discrimi-
native or rely on distillation, leaving gaps on long-tail, multi-step,
negation-sensitive, and knowledge-intensive queries under real-
world latency budgets. In parallel, RL-driven reasoning with PPO
[14] or GRPO [ 15]—e.g., o1 [ 7] and DeepSeek-R1 [ 15]—has shown
strong gains in mathematics and programming, motivating explo-
ration of RL for search relevance. Building on this trajectory, we
augment relevance modeling with retrieval-augmented information
under a single-pass constraint: our approach incorporates RAG to
supply fresh and long-tail knowledge and trains the model to selec-
tively adopt or ignore external evidence within one forward pass,
improving robustness and attribution in large-scale e-commerce
search.
2.2 Selective Evidence Use in
Retrieval-Augmented Generation
Recent work on retrieval-augmented generation emphasizes de-
ciding when to retrieve, selectively adopting evidence, and au-
tonomously managing external knowledge during reasoning. For
adaptive gating, Self-RAG [ 2] augments generation with control or
evaluation labels and a critique model to decide whether to retrieve
and to assess relevance, support, and usefulness. Adaptive-RAG [ 8]
classifies question complexity and routes queries to no, single-step,
or multi-step retrieval to avoid unnecessary calls. RA-ISF [ 11] cou-
ples self-knowledge assessment, paragraph relevance estimation,
and task decomposition to retrieve incrementally when internal
knowledge is insufficient.
For evidence filtering and refinement, Astute-RAG [ 20] contrasts
an initial answer with retrieved content, builds a consistent context,
and selects a consensus answer across aligned paragraph groups
to mitigate conflicts. A supervised self-reasoning pipeline decom-
poses relevance assessment, key-sentence extraction, and answer
synthesis to stabilize attribution and improve selection [ 22]. Search-
and-refine [ 16] adds an explicit refinement stage with coverage and
fidelity signals to align snippets with document ground truth, and
R-Search [ 23] explicitly extracts evidence with an auxiliary judge
and combines correctness, evidence, and format rewards to raise
precision.

DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Figure 1: DyKnow-RAG training and deployment overview. Stage 1: SFT with structured chain of thought and optional DPO
warm start. Stage 2: GRPO -oriented data filtering using SFT posteriors to construct an uncertainty -prioritized RL pool. Stage 3:
implicit context -Utilization reinforcement with dual -group GRPO (with context vs. no context), intra -group advantages
and posterior -driven inter -group scaling to learn adopt/partial/ignore behavior. Bottom: single -pass, single -chunk online
deployment.
Reinforcement-learning approaches integrate retrieval into the
reasoning loop. Search-R1 [ 9] treats search as part of the environ-
ment, supports multi-round retrieval/reasoning via special markers,
and optimizes with outcome-based rewards for on-demand exter-
nal knowledge. Knowledgeable-r1 [ 10] uses GRPO with paired
rollouts to learn when external evidence helps. R1-Searcher++ [ 17]
warm-starts with SFT to learn internal/external patterns, then ap-
plies GRPO-style RL with combined rewards to reduce unneces-
sary retrieval while preserving correctness. Overall, these methods
strengthen attribution and robustness but often introduce extra
refinement or multi-round interaction, which can be sensitive to
latency and cost constraints in practice.
3 Method
This section presents a production -aligned training framework for
e-commerce relevance under strict single -pass latency. We first for-
malize the task and inputs; to bound context cost under chunk -level
retrieval, we select only the single most relevant external contextchunk𝑐∗and use it consistently during training and inference. The
pipeline, summarized in Figure 1, has three stages: (1) supervised
fine-tuning (SFT) with a structured chain of thought that outputs
a four -tier relevance label and a concise rationale, and explicitly
records the context -Utilization decision; (2) a GRPO -oriented data
filtering strategy that builds an uncertainty -prioritized RL pool
from SFT posteriors to focus updates on borderline cases where
using the context most affects outcomes; and (3) DyKnow -RAG’s
implicit context -Utilization reinforcement, which adopts GRPO
with two rollout groups (no context vs. with a single context chunk)
and posterior -driven inter -group advantage scaling so the policy
learns when to adopt, partially adopt, or ignore the context under
noise. We optionally apply a lightweight DPO warm start under the
with -context prompt to improve calibration. The design requires
no value network and adds no extra inference passes, preserving
single -pass, single -chunk deployment without multi -stage summa-
rization.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
3.1 Task Formulation
In Taobao’s e-commerce search setting, the relevance task aims to
determine a four-level query–item relevance under strict online
latency constraints and provide a concise, human-readable rationale.
The input consists of a user query 𝑞, a candidate item 𝑖, and the Top-
𝐾external context chunks 𝐶=𝑐 1,...,𝑐𝐾for𝑞. To bound context
cost, we select the single most relevant chunk 𝑐∗∈𝐶as the only
context used at both training and inference. The model outputs a
four-class relevance label 𝑦∈{4-Excellent, 3-Related, 2-Mismatch,
1-Irrelevant} , together with a brief process explanation (including
the judgment on whether to use external context and the reasoning
for the𝑞–𝑖match). The online constraint is single-turn inference,
prohibiting additional inference rounds or multi-stage summarize-
then-decide procedures.
3.2 Supervised Initialization and Optional DPO
Warm-Start
CoT templateTo align output structure with the subsequent
RL stage, the SFT prompt enforces a structured chain-of-thought
(CoT) consisting of (i) the query–item relevance label; (ii) an explicit
judgement on whether to use the external context; and (iii) a brief
rationale for the judgement and query–item match. After obtain-
ing human-annotated relevance tiers, we employDeepSeek-R1to
construct CoT rationales conditioned on (𝑞,𝑖,𝑐∗,𝑦), ensuring each
rationale explicitly includes the context utilization decision and the
key supporting factors for relevance.
SFT Optimization and Confidence ScoringWe train the model
with standard cross-entropy over the structured output. After SFT,
we run an inference pass to record the confidence of the predicted
relevance tier token via a softmax-normalized probability over the
four class tokens. Let Trel={𝑡 4,𝑡3,𝑡2,𝑡1}denote the tokens for the
four tiers and 𝑧(·)the pre-softmax score; the posterior for a class
token𝑡∈T relis
𝑝𝜃(𝑡|𝑥)=exp(𝑧(𝑡))Í
𝑡∈T relexp(𝑧(𝑡)),
where𝑥=(𝑞,𝑖,𝑐∗). We take the predicted token ˆ𝑡=arg max 𝑡∈T rel
𝑝𝜃(𝑡|𝑥) and use its probability 𝑝𝜃(ˆ𝑡|𝑥) as a confidence score for
subsequent GRPO data filtering (see Section 3.3).
DPO Preference Warm-startTo stabilize and calibrate the base
policy’s use of external evidence, we optionally apply Direct Pref-
erence Optimization (DPO). For each input 𝑥=(𝑞,𝑖,𝑐∗), we sam-
ple multiple drafts from the SFT model and form preference pairs
(𝑦+,𝑦−)by labeling outputs that agree with the gold relevance as
positive and those that disagree as negative (with deduplicated).
The DPO objective is
L(𝜃)=−E(𝑥,𝑦+,𝑦−)∼𝐷
log𝜎 𝛽·log𝜋𝜃(𝑦+|𝑥)
𝜋ref(𝑦+|𝑥)
−𝛽·log𝜋𝜃(𝑦−|𝑥)
𝜋ref(𝑦−|𝑥)(1)
where𝜋refis a frozen reference (the SFT model) and 𝛽> 0
controls the slope. This warm-start is used in some runs to im-
prove overall perfermance with external context, and we report
its standalone and combined effects with DyKnow-RAG in the
experiments.3.3 GRPO-Oriented Data Filtering Strategy
Data SourceWe collect query–item pairs from Taobao’s online
search logs and deliberately concentrate the query distribution
on four challenging categories: negation, affordable alternatives,
question answering (QA), and knowledge-based queries. These
categories often involve hidden constraints, semantic reversal, cross-
domain factual references, or attribute completion; consequently,
they can draw on the abundant, timely, and context-rich signals
available in social and user-generated content (UGC) to capture
emergent entities and domain-specific knowledge. At the same time,
these sources exhibit high variance and low reliability, introducing
inconsistency, bias, redundancy, and misinformation, and thus these
categories are particularly susceptible to noise from social/UGC
sources.
Uncertainty-driven FilteringWe run the SFT model (Section 3.2)
on a held-out pool and record the softmax confidence of the pre-
dicted relevance-tier token. Empirically, when the confidence falls
below0.7, the error rate approaches50%, indicating high uncer-
tainty and headroom. We thus construct the RL pool by prioritizing
instances with confidence <0.7, concentrating RL on borderline
cases where context utilization most affects outcomes. This im-
proves sample efficiency and strengthens the model’s ability to
adopt or ignore the context appropriately.
Dual Labeling: Relevance and Context UtilizationTo sup-
port two independent training routes (prior vs. posterior), each in-
stance is annotated with (i) a four-level relevance label ({4-Excellent,
3-Related, 2-Mismatch, 1-Irrelevant}) and (ii) a context utilization
label (adopt/partial/ignore). The relevance label supervises the main
task; the context utilization label serves as a prior reference when
applicable.
To prevent distributional collapse, we maintain balance across
the four query types and preserve coverage over relevance tiers
and context -utilization states during sampling. The resulting GRPO
training set contains approximately50,000instances.
3.4 Reinforcement Learning for Dynamic
Context Utilization
3.4.1 GRPO overview.We adoptGroup Relative Policy Optimiza-
tion(GRPO) as the RL backbone. For every input, the old policy
𝜋𝜃oldsamples a group of 𝐺drafts{𝑜𝑖}𝐺
𝑖=1. The new policy 𝜋𝜃is up-
dated by maximizing a clipped importance ratio with a KL penalty
to a reference policy 𝜋ref, using self-normalizing, group-relative
advantages:
JGRPO(𝜃)=E𝑞,{𝑜 𝑖}1
𝐺𝐺∑︁
𝑖=11
|𝑜𝑖||𝑜𝑖|∑︁
𝑡=1"
min𝜋𝑖,𝑡
𝜃
𝜋𝑖,𝑡
𝜃oldˆ𝐴𝑖,𝑡,clip 𝜋𝑖,𝑡
𝜃
𝜋𝑖,𝑡
𝜃old,1−𝜖,1+𝜖ˆ𝐴𝑖,𝑡
−𝛽D KL 𝜋𝜃∥𝜋ref#
,(2)
where ˆ𝐴𝑖,𝑡denotes the group–relative advantage (z-scored within
group and length-normalized). This formulation removes the need
for a value function while maintaining stability and sample effi-
ciency.

DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
3.4.2 Reinforcement Learning for Dynamic Context Utilization.In-
stead of relying on human context utilization labels—which may not
reflect the model’s evolving knowledge or the instance’s difficulty—
we learn the adopt/partial/ignore behavior within GRPO. Con-
cretely, for each input we run two rollout groups: one under a
no-context prompt 𝑝(parametric-only) and one under a with -context
prompt𝑝′that supplies the single external chunk 𝑐∗. Rewards
are z-scored within each group to form group-relative advantages.
We then introduce an inter-group alignment term by evaluating
no-context sequences under the with -context prompt and applying
a piecewise scaling, which shapes the policy’s preference to utilize
or to ignore the context.
Generate stageLet {𝑜(0)
𝑖}𝑛
𝑖=1denote rollouts sampled under 𝑝
(no context) with returns {𝑅(0)
𝑖}, and{𝑜(1)
𝑖}𝑛
𝑖=1rollouts under 𝑝′
(with context) with returns {𝑅(1)
𝑖}. For each group 𝑔∈{ 0,1}, we
compute a group-specific baseline and scale
𝜇𝑔=1
𝑛𝑛∑︁
𝑢=1𝑅(𝑔)
𝑢, 𝑠𝑔=vt
1
𝑛𝑛∑︁
𝑢=1 𝑅(𝑔)
𝑢−𝜇𝑔2+𝜀,(3)
and form z-scored, length-agnostic advantages
𝐴(𝑔)
𝑖=𝑅(𝑔)
𝑖−𝜇𝑔
𝑠𝑔.(4)
To place the two groups on a shared reference scale, we aggregate
returns from both groups and compute the union statistics
𝜇★=1
2𝑛 𝑛∑︁
𝑖=1𝑅(0)
𝑖+𝑛∑︁
𝑖=1𝑅(1)
𝑖!
,(5)
𝑠★=vut
1
2𝑛 𝑛∑︁
𝑖=1 𝑅(0)
𝑖−𝜇★2+𝑛∑︁
𝑖=1 𝑅(1)
𝑖−𝜇★2!
+𝜀,(6)
and define a normalized inter -group advantage for the no -context
rollouts,
˜𝐴𝑖=𝑅(0)
𝑖−𝜇★
𝑠★,(7)
which measures how a no -context rollout performs relative to the
joint distribution of returns across groups.
We then apply a sign -dependent (piecewise) scaling to modulate
its influence:
𝑇(˜𝐴𝑖)=(
𝛼˜𝐴𝑖,if ˜𝐴𝑖>0,
𝛽˜𝐴𝑖,if ˜𝐴𝑖≤0,(8)
where𝛼,𝛽> 0are scaling coefficients to encourage or discourage
context utilization.
Train stageWe optimize three terms along with a KL regularizer
to a reference policy𝜋 ref. For no-context rollouts under𝑝,
ℓ(0)(𝜃)=1
𝑛𝑛∑︁
𝑖=11
|𝑜(0)
𝑖||𝑜(0)
𝑖|∑︁
𝑡=1min"𝜋𝜃
𝑜(0)
𝑖,𝑡|𝑝,𝑜(0)
𝑖,<𝑡
𝜋𝜃old
𝑜(0)
𝑖,𝑡|𝑝,𝑜(0)
𝑖,<𝑡𝐴(0)
𝑖,
clip 𝜋𝜃
𝑜(0)
𝑖,𝑡|𝑝,𝑜(0)
𝑖,<𝑡
𝜋𝜃old
𝑜(0)
𝑖,𝑡|𝑝,𝑜(0)
𝑖,<𝑡; 1−𝜖,1+𝜖!
𝐴(0)
𝑖#
.(9)
For with-context rollouts under𝑝′,ℓ(1)(𝜃)=1
𝑛𝑛∑︁
𝑖=11
|𝑜(1)
𝑖||𝑜(1)
𝑖|∑︁
𝑡=1min"𝜋𝜃
𝑜(1)
𝑖,𝑡|𝑝,𝑜(1)
𝑖,<𝑡
𝜋𝜃old
𝑜(1)
𝑖,𝑡|𝑝,𝑜(1)
𝑖,<𝑡𝐴(1)
𝑖,
clip 𝜋𝜃
𝑜(1)
𝑖,𝑡|𝑝,𝑜(1)
𝑖,<𝑡
𝜋𝜃old
𝑜(1)
𝑖,𝑡|𝑝,𝑜(1)
𝑖,<𝑡; 1−𝜖,1+𝜖!
𝐴(1)
𝑖#
.
(10)
The inter -group term evaluates no -context sequences under the
with -context prompt using the normalized inter -group advantage
˜𝐴𝑖from the generate stage:
ˆℓ(𝜃)=1
𝑛𝑛∑︁
𝑖=11
|𝑜(0)
𝑖||𝑜(0)
𝑖|∑︁
𝑡=1h
𝜋𝜃
𝑜(0)
𝑖,𝑡|𝑝′,𝑜(0)
𝑖,<𝑡
𝑇(˜𝐴𝑖)i
.(11)
The final objective is
J(𝜃)=ℓ(0)(𝜃)+ℓ(1)(𝜃)+ ˆℓ(𝜃) −𝜆 KLDKL[𝜋𝜃∥𝜋ref],(12)
where𝜆 KL>0is the KL weight.
Posterior -driven scalingWe instantiate 𝑇(·) using a posterior
accuracy gap between with-context and no-context rollouts within
the current batch:
𝛽=4·𝜎(4·(acc with−acc without))(13)
𝛼=0.1
𝛽(14)
where𝜎is the sigmoid function. This accelerates scaling near zero
gaps for efficiency and saturates near one to avoid instability. This
posterior-driven scaling teaches the policy when to rely on context
versus parametric knowledge without requiring human process
labels.
4 Experiments
This section describes all data resources, model configurations, and
evaluation protocols used in our study.
4.1 Datasets and Metrics
Offline Dataset Construction and AnnotationWe build an
offline test set of approximately 21,616 query–item pairs sampled
from the scoring candidate space of the production relevance model
using Taobao’s search logs. To stress-test reasoning and discrimina-
tion under challenging contexts, we upweight four difficult query
types during sampling: negation, affordable alternatives, QA-style
queries, and knowledge. For each instance, we retrieve evidence
chunks from an offline corpus and, under online cost constraints,
select a single most relevant external context chunk 𝑐∗as input.
Human annotators assign a four-tier relevance label (4-Excellent,
3-Related, 2-Mismatch, 1-Irrelevant). The test set label distribution
is shown in Table 1. Due to upstream filtering, fully irrelevant (1-
Irrelevant) cases are rare, whereas partially irrelevant (2-Mismatch)
remain prevalent. Overall, the test set maintains a near-balanced
split between relevant (4/3) and non-relevant (2/1).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 1: Dataset Statistics
Label Count Ratio
L4 12930 60%
L3 1514 7%
L2 5891 27%
L1 1281 6%
Online Evaluation and Labeling ProtocolOnline evaluation
is conducted via live A/B testing. We freeze retrieval and ranking
to ensure a controlled comparison and swap only the relevance
decision module. Traffic is split at a fixed ratio and the experiment
spans typical weekdays and weekends. Judgments follow a double-
blind, side-by-side procedure: assessors compare A/B result pages
for the same query and record Good/Same/Bad (relative) decisions;
in parallel, pages are rated Good/Mid/Bad based on the fraction of
relevant items displayed.
Offline MetricsOffline evaluation focuses on the following mea-
sures:
•Macro F1:the macro-averaged F1 across the four relevance levels,
reflecting balanced performance.
•Accuracy:overall accuracy computed over the four relevance
levels.
•Per-tier F1:class-wise F1 reported separately for each relevance
tier (4-Excellent, 3-Related, 2-Mismatch, 1-Irrelevant)
Online MetricsOnline evaluation includes three key metrics:
•GSB (Good/Same/Bad):a side-by-side human comparison that
quantifies the relative advantage of the test bucket versus the
baseline. GSB+𝑥%means𝑥%more results in the test bucket are
judged “better” than those in the baseline bucket.
•Query Goodrate:a page-level relevance metric; it is the fraction
of queries whose result pages are judged as Good or Mid by
human raters.
•Item Goodrate:an item-level metric; for each request, we com-
pute the proportion of highly relevant items (e.g., levels 4-Excellent
and 3-Related) and then average across requests.
Note the reporting convention: Query Goodrate and Item Goodrate
reflect absolute performance, for which we report absolute lift; GSB
captures relative advantage, for which we report relative improve-
ment.
4.2 Implementation Details
Supervised Fine-Tuning (SFT)We train TBStar as the base with
a structured chain-of-thought (CoT) output using cross-entropy loss.
The batch size is 1024 and the learning rate is1 ×10−6, scheduled by
cosine decay with a 5% warmup. We use AdamW ( 𝛽1=0.9,𝛽2=0.999),
run for 2 epochs, and cap the maximum input length at 4096 tokens.
Mixed precision and gradient clipping (global norm 1.0) are applied
for stability.
GRPO Reinforcement LearningFor each input, we produce
𝐺=16responses with temperature0 .99and top-𝑘=100; the rollout
batch size is 64. The thresholds for our online difficulty sampling
were set to a range of [0.01,0.9]. The GRPO learning rate is1 ×10−6.
We clip the importance sampling ratio to1 ±0.2and clamp per-
token advantages to ±2, with group-wise z-score normalization and
length normalization. A KL penalty ( 𝛽=0.02) to the reference policyis applied to control drift. Our pipeline is implemented on top of
the open-source ROLL framework[ 21], with centralized logging
and fixed random seeds to ensure reproducibility.
4.3 Baselines and Ablations
BaselinesWe compare against the following baselines under
a unified retrieval/index version, one context chunk (Top-1 𝑐∗),
matched token budgets, and aligned training schedules:
•SFT-only (no RAG):Supervised fine-tuning with structured
CoT and four-tier relevance labels; no external context chunk is
used at training or inference.
•RAG SFT:SFT under with-chunk prompting using a single con-
text chunk𝑐∗, keeping the same CoT format and supervision as
SFT-only.
•RAG DPO:Direct Preference Optimization on top of RAG SFT
as a preference warm-start; we form positive/negative pairs by
agreement with gold relevance and use a frozen SFT reference.
•RAG GRPO (SFT base):: Standard GRPO applied on top of RAG
SFT, and the optimization uses group-relative advantages and a
KL regularizer.
•DyKnow-RAG (SFT base):Our method with dual rollout groups
(no-chunk vs. with-chunk using 𝑐∗) and dynamic inter-group
advantage scaling, initialized from SFT-only.
•DyKnow-RAG (DPO base):Same as above but initialized from
the RAG DPO checkpoint, assessing DyKnow-RAG’s robustness
under a preference-calibrated base.
AblationsWe include four targeted ablations to probe key design
choices while keeping training cost controlled:
•Retrieval context:compare single-chunk context ( 𝑐∗from Top-
1) with a multi-chunk variant (Top-3 merged), assessing coverage
versus noise accumulation.
•Fixed-scaling variant:replace posterior-driven scaling with a
globally fixed piecewise scaling on the inter-group term, inducing
a constant preference between no-chunk and with-chunk.
•Label-conditioned gating:use adopt/partial/ignore process
labels to set the piecewise scaling, serving as a label-driven alter-
native to learned scaling.
•Context at inference (on/off):Compare with-chunk versus
no-chunk inference to examine whether noisy context chunks
help or harm the model’s relevance decisions.
Unless otherwise specified, all runs share the same rollout group
size, sampling settings, optimizer, and KL regularization; recall and
coarse ranking are frozen to isolate relevance modeling effects.
5 Results and Discussion
We first present offline evaluation (Section 5.1), quantifying overall
performance and analyzing behavior across the four query cate-
gories. We then conduct ablations (Section 5.2) to isolate the effect
of key design choices. Next, we report online results (Section 5.3)
under production constraints and discuss how the observed lifts
align with offline trends. Finally, we provide case study (Section 5.4)
contrasting SFT and GRPO variants to illustrate the sources of
improvement.

DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 2: Offline Evaluation Results
Models Class-1 F1 Class-2 F1 Class-3 F1 Class-4 F1 Macro F1 Accuracy
SFT-only 41.97 63.35 39.44 79.55 58.14 71.88
RAG SFT 40.47 62.59 39.21 80.37 57.06 71.92
RAG DPO 45.66 65.07 44.26 83.54 59.67 74.91
RAG GRPO (RAG SFT base) 47.54 62.72 42.82 80.18 58.33 72.55
DyKnow-RAG (RAG SFT base)49.4864.145.6482.02 60.33 73.26
DyKnow-RAG (RAG DPO base) 47.9665.2144.9883.66 60.45 75.19
5.1 Offline Evaluation
Offline evaluation results are summarized in Table 2, where DyKnow-
RAG (RAG DPO base) attains the best macro-F1 and overall accuracy
among all systems.
Supervised exposure to noisy RAG hurts balanceCompared
with SFT-only, RAG SFT lowers Macro-F1 while keeping Acc largely
unchanged; the drop spans all strata except Class 4. Directly super-
vising on noisy context biases the model toward superficial cues
(e.g., positive phrasing, trending terms), yielding small gains on
high-relevance samples (Class 4) but hurting minority/boundary
classes (Class 1/3) with noise-sensitive or ambiguous decisions.
Given skewed labels, the minority-class decline has limited impact
on Acc, which is dominated by the majority classes.
DyKnow-RAG addresses supervision bias in context useCon-
trasting DyKnow-RAG (RAG SFT base) with RAG SFT and RAG
GRPO, Macro-F1 increases further, with the largest gains on minor-
ity and boundary classes (Class 1 and Class 3), both reaching their
best performance. DyKnow-RAG, via dual-group control (no-chunk
vs with-chunk) coupled with dynamic scaling, explicitly learns a
policy for when to adopt, partially adopt, or ignore external context.
Under high-noise or unstable-evidence conditions, it preferentially
ignores or cautiously incorporates context, strengthening boundary
discrimination and suppressing noise; this adaptive usage policy
translates into a substantial improvement in class-level balance
captured by Macro-F1.
DPO warm start complemented by DyKnow-RAGWith DPO
warm start plus DyKnow-RAG, Macro-F1 and Acc attain their best
values. Direct Preference Optimization (DPO) maximizes the pref-
erence margin to calibrate the consistency between generation and
evidence use, prioritizing accurate evidence utilization and thereby
reducing mis-citation. Building on this, DyKnow-RAG learns a
gated policy for evidence adoption timing via cross-group control
and dynamic scaling, avoiding indiscriminate adoption when noise
is strong or evidence is unreliable. The two are complementary:
DPO enhances discriminative quality when evidence is adopted,
whereas DyKnow-RAG optimizes the timing of adoption versus
omission. Their joint effect, under a majority-skewed label distri-
bution, significantly increases overall accuracy (Acc) while curbing
degradation on minority and boundary classes, preserving Macro-
F1 and yielding a superior overall outcome.
5.2 Ablation Study
More chunks are not necessarily betterUnder the same re-
trieval pipeline and training budget, we compare Top-1 versus Top-3
retrieved chunks. As shown in Table 4, increasing the number ofchunks from one to three slightly reduces both accuracy and macro-
F1. This indicates that adding more context expands coverage but
also accumulates noisy or conflicting evidence, which harms bal-
anced classification and boundary decisions. We therefore use the
Top-1 configuration as the default in subsequent experiments, em-
phasizing adaptive usage policies over enlarging the context.
Static scaling versus posterior-driven gatingMotivated by
earlier results where SFT-only outperformed RAG SFT (suggesting
noisy retrieval), we set fixed coefficients 𝛼=2and𝛽=0.05in Eq. 8
to upweight the no-chunk path and downweight the chunk path,
encouraging reliance on parametric knowledge. As shown in Ta-
ble 5, DyKnow-RAG yields higher Macro-F1 and Acc. The key is
adaptation: it converts per-query correctness gaps between roll-
outs into scaling and gates via Eq. 8, enabling adopt/partial/ignore
conditioned on evidence reliability. Fixed scaling reduces misuse
but cannot pivot when context is beneficial.
Human gating helps but posterior-adaptive is betterAs
shown in Table 5, we compare human rag label gating (using 𝛼=0.05,
𝛽=2when labels indicate adopt/partial-adopt and 𝛼=2,𝛽=0.05when
labels indicate ignore in Eq. 8) against DyKnow-RAG (SFT base).
Human gating injects process supervision to curb indiscriminate
adoption and generally outperforms static scaling.However, it de-
pends on annotators’ knowledge and coverage: a labeler may deem
external context useless while the model lacks parametric knowl-
edge in that niche and would actually benefit from context, or vice
versa. DyKnow-RAG is posterior-driven, contrasting no-RAG and
with-RAG behaviors to learn per-query gates and scaling; it adopts
when evidence is reliable and ignores or partially adopts under
noise.
Context at inference: do noisy chunks help or harm?Under
matched prompts and settings, we compare enabling versus dis-
abling the context chunk at inference. As shown in Table 6, RAG
SFT is slightly better with no-chunk than with-chunk, indicating
that without a usage policy, noisy context harms relevance. By con-
trast, DyKnow-RAG performs best with with-chunk, and remains
strong with no-chunk, showing both robust fallback to parametric
knowledge and policy-controlled gains when evidence is reliable.
5.3 Online Evaluation
We evaluate under live production traffic with side-by-side (SBS)
human assessments and efficiency metrics. Both the merged re-
trieval pipeline and our relevance model have been deployed since
September 2025 and continuously serve traffic. During early rollout
we observed efficiency regressions, primarily due to insufficient
high-efficiency item retrieval in the front-stage retrieval chain. Af-
ter multi-route retrieval merge and related optimizations, efficiency

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 3: Side-by-Side Human Evaluations
Query slice Case GSB Query Goodrate Item Goodrate
Q&A Which shampoo can reduce hair loss? +10.37% +5.76 pt (vs 68.43) +2.45 pt (vs 77.59)
Knowledge Glue that withstands high temperatures +7.75% +2.91 pt (vs 70.70) +1.15 pt (vs 78.35)
Negation Face cream without fragrance +3.25% -0.20 pt (vs 67.68) +0.81 pt (vs 75.96)
Alternatives Gucci belt alternatives +3.60% +1.80 pt (vs 48.80) +0.84 pt (vs 57.50)
Table 4: Ablation on the number of retrieved chunks
Setting Macro-F1 Accuracy
Top-1 chunk 57.06 71.92
Top-3 chunks 56.74 71.26
Table 5: Ablation: static scaling and human context-use gat-
ing vs DyKnow-RAG
Setting Macro-F1 Accuracy
Fixed gating (𝛼=2,𝛽=0.05) 59.31 71.75
Human context-use gating 60.15 73.05
DyKnow-RAG (SFT base) 60.33 73.26
Table 6: Ablation on RAG at inference (on/off)
Setting Macro-F1 Accuracy
RAG SFT + with-chunk 57.06 71.92
RAG SFT + no-chunk 57.31 72.27
DyKnow-RAG + with-chunk 60.45 75.19
DyKnow-RAG + no-chunk 59.71 74.80
is at parity with the base bucket, and the SBS gains remain com-
parable to those summarized in Table 3. We report both baseline
levels and absolute lifts.
We observe strong gains on Q&A and Knowledge queries, con-
sistent with disciplined evidence adoption under the same retrieval
budget. Alternatives also improve, reflecting better intent under-
standing beyond lexical overlap. Negation shows a slight fluctua-
tion at the query level but positive item-level lift, suggesting more
conservative filtering of misleading candidates. Weighted by slice
shares, overall gains are dominated by Q&A and Knowledge traf-
fic, while the model remains robust under noisy scenarios due to
selective adoption and omission policies.
5.4 Case Study
Figure 2 shows a query–item–context case where the retrieved
chunk is off domain: RAG-SFT adopts the chunk, shifts the query
toward hair care, and wrongly rates the textile-care item as L1,
while DyKnow-RAG ignores the misleading chunk, aligns the query
with textile color correction for garments, and assigns L4; this
case demonstrates that with a learned usage policy the model can
disregard noisy context and deliver the correct relevance judgment.
Figure 2: Case study: an off-domain context chunk leads RAG-
SFT to adopt the chunk and misjudge the textile-care item
(L1), while DyKnow-RAG ignores the chunk and produces a
fully relevant judgment (L4).
6 Conclusion
DyKnow-RAG turns external context from a static add on into a
policy controlled resource for e-commerce search relevance. Under
single pass and single chunk deployment, it learns via GRPO with
paired no chunk and with chunk rollouts and posterior driven inter-
group advantage scaling when to adopt, partially adopt, or ignore
the context. The production-aligned pipeline couples structured
CoT supervision with an uncertainty prioritized RL pool derived
from SFT posteriors and an optional DPO warm start, delivering
disciplined evidence use without extra inference passes or value net-
works. Under a unified retrieval/index and a fixed latency budget,
DyKnow-RAG achieves state of the art performance both offline
(Macro F1 and Accuracy) and online (GSB, Query Goodrate, Item
Goodrate) on Taobao traffic, consistently surpassing strong SFT,
DPO, and vanilla GRPO baselines. These results establish DyKnow-
RAG as a deployable, scalable paradigm that reliably converts noisy
external context into superior relevance quality at industrial scale.

DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]Akiko Aizawa. 2003. An information-theoretic perspective of tf–idf measures.
Information Processing & Management39, 1 (2003), 45–65.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.
2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection.
(2024).
[3]Zeyuan Chen, Haiyan Wu, Kaixin Wu, Wei Chen, Mingjie Zhong, Jia Xu, Zhongyi
Liu, and Wei Zhang. 2024. Towards Boosting LLMs-driven Relevance Model-
ing with Progressive Retrieved Behavior-augmented Prompting.arXiv preprint
arXiv:2408.09439(2024).
[4]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert:
Pre-training of deep bidirectional transformers for language understanding. In
Proceedings of the 2019 conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1 (long and
short papers). 4171–4186.
[5]Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang,
Xiaojiang Zhou, Dan Ou, and Haihong Tang. 2025. TaoSR1: The Thinking Model
for E-commerce Relevance Search.arXiv preprint arXiv:2508.12365(2025).
[6]Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry
Heck. 2013. Learning deep structured semantic models for web search using
clickthrough data. InProceedings of the 22nd ACM international conference on
Information & Knowledge Management. 2333–2338.
[7]Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky,
Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al .2024.
Openai o1 system card.arXiv preprint arXiv:2412.16720(2024).
[8]Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park.
2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers). 7029–7043.
[9]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason
and leverage search engines with reinforcement learning.arXiv preprint
arXiv:2503.09516(2025).
[10] Chenyu Lin, Yilin Wen, Du Su, Fei Sun, Muhan Chen, Chenfu Bao, and Zhonghou
Lv. 2025. Knowledgeable-r1: Policy Optimization for Knowledge Exploration in
Retrieval-Augmented Generation.arXiv preprint arXiv:2506.05154(2025).
[11] Yanming Liu, Xinyue Peng, Xuhong Zhang, Weihao Liu, Jianwei Yin, Jiannan
Cao, and Tianyu Du. 2024. RA-ISF: Learning to Answer and Understand from
Retrieval Augmentation via Iterative Self-Feedback. InFindings of the Association
for Computational Linguistics ACL 2024. 4730–4749.
[12] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural information processing
systems36 (2023), 53728–53741.
[13] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval
3, 4 (2009), 333–389.
[14] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal policy optimization algorithms.arXiv preprint arXiv:1707.06347
(2017).
[15] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al .2024. Deepseekmath: Pushing
the limits of mathematical reasoning in open language models.arXiv preprint
arXiv:2402.03300(2024).
[16] Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An
Zhang, and Xiang Wang. 2025. Search and Refine During Think: Autonomous
Retrieval-Augmented Reasoning of LLMs.arXiv preprint arXiv:2505.11277(2025).
[17] Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao
Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-
Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via
Reinforcement Learning.arXiv preprint arXiv:2505.17005(2025).
[18] Tian Tang, Zhixing Tian, Zhenyu Zhu, Chenyang Wang, Haiqing Hu, Guoyu Tang,
Lin Liu, and Sulong Xu. 2025. LREF: A Novel LLM-based Relevance Framework
for E-commerce Search. InCompanion Proceedings of the ACM on Web Conference
2025. 468–475.
[19] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing systems30 (2017).
[20] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan Ö Arık. 2024. As-
tute rag: Overcoming imperfect retrieval augmentation and knowledge conflicts
for large language models.arXiv preprint arXiv:2410.07176(2024).
[21] Weixun Wang, Shaopan Xiong, Gengru Chen, Wei Gao, Sheng Guo, Yancheng
He, Ju Huang, Jiaheng Liu, Zhendong Li, Xiaoyang Li, et al .2025. Reinforcement
Learning Optimization for Large-Scale Learning: An Efficient and User-Friendly
Scaling Library.arXiv preprint arXiv:2506.06122(2025).[22] Yuan Xia, Jingbo Zhou, Zhenhui Shi, Jun Chen, and Haifeng Huang. 2025. Im-
proving retrieval augmented language model with self-reasoning. InProceedings
of the AAAI conference on artificial intelligence, Vol. 39. 25534–25542.
[23] Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, and Limin Liu. 2025.
R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforce-
ment Learning.arXiv preprint arXiv:2506.04185(2025).
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
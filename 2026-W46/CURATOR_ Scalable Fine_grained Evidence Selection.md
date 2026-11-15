# CURATOR: Scalable Fine-grained Evidence Selection for Retrieval-Augmented Generation using Decision Models

**Authors**: Shib Dasgupta, Bandish Parikh, Rashmi Gangadharaiah

**Published**: 2026-11-12 06:29:40

**PDF URL**: [https://arxiv.org/pdf/2611.16366v1](https://arxiv.org/pdf/2611.16366v1)

## Abstract
Retrieval-augmented generation (RAG) provides entire retrieved passages to the LLM, even though only a handful of spans within them contain the evidence pertaining to the question. While precise selection of these spans can improve context utilization, identifying them in isolation is challenging. A span may appear irrelevant by itself, and deciding whether to retain it may require examining other retrieved information. One can evaluate each span's candidacy with the full retrieved context, but this is computationally inefficient as the same text must be processed repeatedly. Pretrained decision model can be used to efficiently evaluate multiple spans simultaneously, allowing each decision to consider the surrounding spans as well. However, our experiments show that their ability to identify relevant evidence deteriorates as the number of simultaneous decisions increases.
  To address this, we introduce CURATOR, a fine-tuned decision model trained to make reliable evidence-selection decisions over large candidate sets. By training a decision model to select varying numbers of supporting spans from large candidate pools, CURATOR develops the capability to select evidence for RAG while retaining the computational efficiency of joint decision-making. We observe that CURATOR maintains a strong evidence discrimination capability as the number of candidates increases, generalizes across different tasks and document domains, and improves downstream RAG performance, particularly when the available context is limited.

## Full Text


<!-- PDF content starts -->

CURATOR: Scalable Fine-grained Evidence Selection for
Retrieval-Augmented Generation using Decision Models
Shib Dasgupta
AWS AI Labs
shibdg@amazon.comBandish Parikh
AWS AI Labs
bandish@amazon.comRashmi Gangadharaiah
AWS AI Labs
rgangad@amazon.com
Abstract
Retrieval-augmented generation (RAG) pro-
vides entire retrieved passages to the LLM,
even though only a handful of spans within
them contain the evidence pertaining to the
question. While precise selection of these spans
can improve context utilization, identifying
them in isolation is challenging. A span may ap-
pear irrelevant by itself, and deciding whether
to retain it may require examining other re-
trieved information. One can evaluate each
span’s candidacy with the full retrieved context,
but this is computationally inefficient as the
same text must be processed repeatedly. Pre-
trained decision model can be used to efficient
evaluate multiple spans simultaneously, allow-
ing each decision to consider the surrounding
spans as well. However, our experiments show
that their ability to identify relevant evidence
deteriorates as the number of simultaneous de-
cisions increases.
To address this, we introduce CURATOR, a
fine-tuned decision model trained to make re-
liable evidence-selection decisions over large
candidate sets. By training a decision model
to select varying numbers of supporting spans
from large candidate pools, CURATOR devel-
ops the capability to select evidence for RAG
while retaining the computational efficiency of
joint decision-making. We observe that CU-
RATOR maintains a strong evidence discrimi-
nation capability as the number of candidates
increases, generalizes across different tasks and
document domains, and improves downstream
RAG performance, particularly when the avail-
able context is limited.
1 Introduction
Retrieval-augmented generation (RAG) enables
large language models to answer questions us-
ing information retrieved from external knowledge
sources (Lewis et al., 2020; Guu et al., 2020).
These retrieved external documents or passages
helps LLM to rely less on parametric knowledgefor knowledge-intensive tasks such as open-domain
question answering (Izacard and Grave, 2021).
However, these retrieved passages often contain
substantially more irrelevant information than is
needed to answer the question at hand, e.g., back-
ground information, unrelated details on other top-
ics, huge tables with only a few important cells,
etc. Thus, providing the entire passage increases
the number of tokens processed by the answering
LLM and may make relevant evidence harder to use
(Liu et al., 2024; Jiang et al., 2024). This problem
is amplified in agentic workflows, where redundant
and unnecessary information consumes context that
could otherwise accommodate future tool outputs
and intermediate reasoning (Packer et al., 2024;
Sun et al., 2026). This inefficiency motivates us
to study how to select evidence at a finer granu-
larity under a limited context budget. Rather than
forwarding entire retrieved passages, we aim to re-
tain only the sentences or spans that are useful for
answering the question. (Xu et al., 2024).
While fine-grained evidence selection can signif-
icantly reduce input tokens to the answering model,
the selection process itself can be inefficient and
introduce complexity if not done correctly. A col-
lection of retrieved passages may contain hundreds
of candidate spans. A simple approach is to eval-
uate each candidate independently using the ques-
tion and the candidate span. However, a span may
appear irrelevant when considered in isolation but
become relevant when combined with information
found elsewhere in the retrieved context.
Figure 1(a) illustrates this challenge through a
two-hop question: one span establishes that Lon-
don hosted the 2012 Summer Olympics, while
another establishes that the River Thames flows
through London. Collectively, these spans provide
the right evidence needed to identify the river flow-
ing through the city in question. Neither span alone
provides the complete answer. Alternatively, if we
provide the entire retrieved context while judging
1
arXiv:2611.16366v1  [cs.IR]  12 Nov 2026

Figure 1:Motivation for fine-grained evidence selection and joint scoring.(a) Retrieved passages contain
relevant evidence alongside unnecessary information. Selecting useful spans reduces the input context passed to the
answering model, leaving more room for other information within a limited context budget. (b) Three strategies for
scoring candidate spans. Independent scoring considers each span separately, without taking other candidates into
account. Separate context-aware scoring accounts for all the spans but requires repeatedly processing the shared
context. Joint context-aware scoring evaluates multiple candidates through shared inference, avoiding redundant
computation over the full context.
each span then a model would theoretically be able
to better judge the candidacy of each span jointly
with other evidences. However, this will be com-
putationally very inefficient as we would need to
process same contextual information repetitively
while judging each candidate span.
Joint decision models offer a promising alterna-
tive. These models score predefined choices rather
than generating a separate textual judgment for
each candidate (Li and Wagle, 2026; Geuter et al.,
2026). Recent multi-readout decision models can
produce many decisions through shared inference
(Perplexity AI, 2026a). This makes it possible to
evaluate multiple evidence units together while al-
lowing the information to be shared between can-
didates. Figure 1(b) illustrates three span-scoring
strategies for identifying such evidence, motivating
the use of joint decisions.
However, the ability to produce many decisions
efficiently does not necessarily guarantee their re-
liability. In our experiment, we show that pre-
trained joint decision models become less effective
at distinguishing supporting evidence from distrac-
tor spans as the number of candidate decisions in-
creases. Figure 2 illustrates this degradation on fact
verification and multi-answer question answeringtasks. This limitation is particularly important for
fine-grained evidence selection for RAG, where a
single retrieval step may yield many passages con-
taining hundreds of candidate spans. That is why
we investigate a core challenge in this paper: how
can we preserve the computational advantages of
joint inference while ensuring that the decisions for
selecting evidence remain reliable as the number
of candidate spans increases?
To address this challenge, we introduce CURA-
TOR, a framework for scalable joint evidence se-
lection. CURATOR adapts a pretrained multi-
readout decision model to identify supporting ev-
idence spans across variable-sized candidate sets.
We construct supervised evidence-selection exam-
ples from QAMPARI (Amouyal et al., 2023), Hot-
potQA (Yang et al., 2018), and 2WikiMultiHopQA
(Ho et al., 2020), varying both the number of can-
didates and the proportion of supporting evidence
present in those sets. This trains the model to make
many simultaneous decisions across different evi-
dence densities, including settings where only a few
candidates among many are relevant. We use class-
balanced joint evidence supervision to account for
this imbalance. To mitigate the loss of general
decision capabilities, we incorporate replay and
2

distillation on general tasks spanning binary clas-
sification, multiple-choice selection, and scoring.
At inference time, CURATOR jointly scores every
candidate evidence unit with the shared context.
We then select the highest-scoring spans till the
available token budget and leave out the rest.
Through our experiments, we demonstrate the
efficacy of CURATOR in evidence span selec-
tion across different tasks and domains. On QAM-
PARI, which is part of our training data, CURA-
TOR improves evidence recall by an average of
25.8% across different context budgets compared
to the pretrained joint decision model. The im-
provements extend to unseen tasks and domains as
well. For example, we observe consistent improve-
ments in evidence recall on financial documents
and legal contracts. Furthermore, better evidence
selection by CURATOR also leads to improved
downstream RAG performance on multi-hop and
financial question answering. The gains are par-
ticularly pronounced when the available context is
limited. Our scalability experiments show that CU-
RATOR maintains strong evidence discrimination
as the number of simultaneous decisions increases,
unlike the pretrained model. These improvements
are achieved without any degradation in the gen-
eral decision-making capabilities of the pretrained
decision model. Our contributions are threefold:
•Efficient, context-aware evidence selection.
We use joint decision models to select relevant
evidence spans under a token budget, allowing
multiple candidates to be evaluated together with-
out repeatedly processing the full context.
•Scalable joint evidence selection decisions.
We identify a degradation in evidence selection
performance as the number of candidate spans in-
creases. As a remedy, we introduce CURATOR,
a framework that uses variable-scale joint evi-
dence supervision to improve decision reliability
for sparse evidence selection.
•Evaluation of quality, scalability, and general-
ization.
We demonstrate improved fine-grained evidence
selection capability with strong joint-decision
scalability as the number of candidates increases.
At a 128-token context budget, CURATOR im-
proves evidence recall by 17.0% on average
across unseen dataset and domains, and down-
stream RAG answer quality by 17.4% on average
across datasets relative to the strongest baselines.2 Problem Formulation and Motivation
Retrieval-augmented generation (RAG) provides
an answering model with retrieved passages, al-
though only a fraction of the content may be needed
to answer the question. Under a limited context
budget, the task is not to simply rank passages but
to decidewhich pieces of evidence to presentto the
generator.
2.1 Token-Budgeted Evidence Selection
Given a question qand retrieved passages D, let
U= (u 1, . . . , u K)denote the candidate evidence
units obtained from those passages. Here, a unit
can be a sentence, multiple sentences, or a cell
present in a table. Each unit uihas an associ-
ated token cost ci= tokens(u i)and an evidence-
membership label yi∈ {0,1} . Token-budgeted
evidence selection aims to retain as much support-
ing evidence as possible within an allocated bud-
getB. Formally, the selected subset of indices
IB⊆ {1, . . . , K}must satisfy
X
i∈IBci≤B,(1)
To achieve this, the selector model is trained to
assign higher scores rito supporting evidence than
to irrelevant candidate units. At inference time,
candidates are ranked by these scores and greedily
admitted while adhering to the token budget. We
keep this selection procedure fixed across methods
and investigate how the scores are obtained.
2.2 The Accuracy–Efficiency Trade-off
We consider three scoring strategies: independent
scoring, separate context-aware scoring, and joint
context-aware scoring. Let fθ(q, C;u i)denote the
score for candidate uigiven question qand context
C, and let Fθ(q, U) return scores for all candidates.
The three strategies are defined as
rind
i=fθ(q,{u i};ui)Independent,
rsep
i=fθ(q, U;u i)Separate,
rjoint=Fθ(q, U)Joint.(2)
Independent scoring considers only the target
candidate. Separate context-aware scoring consid-
ers the full candidate set but requires Kinferences.
Joint context-aware scoring instead produces all
Kscores through shared inference, provided the
candidate set fits within the model’s context win-
dow. Off-the-shelf rerankers can also be used for
3

8 16 32 64 128 256
Simultaneous candidate decisions, K0.650.700.750.800.850.90Evidence-membership AUROC
FEVEROUS
QAMPARIFigure 2:Scaling limitations of pretrained joint de-
cisions.Evidence-discrimination AUROC decreases
as the number of simultaneous decisions increases on
multi-answer question answering and fact verification
tasks. Higher is better.
independent scoring by ranking individual spans,
followed by the same budgeted selection procedure.
We compare these approaches in Table 2.
Using the same pretrained decision model,
we observe that separate context-aware scoring
achieves higher evidence recall than the other two
strategies. However, joint context-aware scoring
requires substantially less computation, at the cost
of lower decision quality. Detailed comparisons
are provided in Appendix A.1.
2.3 Limitations of Pretrained Joint Decisions
We examine whether pretrained joint decision mod-
els can reliably identify supporting evidence as the
number of simultaneous decisions increases. We
measure evidence discrimination using AUROC,
which evaluates how well the model distinguishes
supporting evidence from distractor spans, indepen-
dently of the context budget.
Figure 2 shows that the pretrained model be-
comes less effective at distinguishing supporting
evidence as the number of candidate decisions in-
creases. We observe a consistent decline in AU-
ROC on both multi-answer question answering and
fact verification tasks. This suggests that the ability
to produce many decisions jointly does not guaran-
tee their reliability.
One possible explanation is that larger candidate
sets also increase the input context length. How-
ever, our diagnostic experiments show that adding
irrelevant text without increasing the number of de-
cisions has a much smaller effect on discrimination.
This suggests that increasing the number of deci-
sions is primarily responsible for the degradation.
Detailed analysis is provided in Appendix A.3.These observations reveal that the pre-trained
decision model can produce many decisions ef-
ficiently, but their reliability deteriorates as the
candidate set grows. This motivates training CU-
RATOR to make reliable joint evidence decisions
across variable-sized candidate sets.
3 CURATOR: Scalable Joint Evidence
Selection
We introduce CURATOR, a framework for adapt-
ing pre-trained joint decision models to reliable ev-
idence selection over large candidate sets of spans.
As shown in Section 2.3, pretrained joint decision
models become less reliable at distinguishing sup-
porting evidence from the distractors as the number
of candidate decisions grows. Our method CURA-
TOR addresses this limitation through fine-tuning a
decision model with evidence selection supervision
over variable-sized candidate sets. We also inter-
leave general decision tasks in the fine-tuning pro-
cess to reduce the risk of forgetting the pretrained
model’s original decision-making capabilities.
3.1 Constructing Joint Evidence Selection
Data
Fine-tuning for our evidence-selection task requires
a dataset with annotations identifying gold evi-
dence spans within passages that support the corre-
sponding answers. We use three datasets that pro-
vide such annotations: QAMPARI (Amouyal et al.,
2023), HotpotQA (Yang et al., 2018), and 2Wiki-
MultiHopQA (Ho et al., 2020). These datasets
contain questions, supporting passages, and the
locations of supporting evidence within those pas-
sages. In particular, QAMPARI contains questions
with multiple correct answers, each supported by
evidence distributed across multiple passages.
Evidence labels.We divide the passages into can-
didate spans and use the dataset annotations to iden-
tify supporting evidence. HotpotQA and 2Wiki-
MultiHopQA provide supporting-fact locations di-
rectly, while QAMPARI provides proof text that
we match to spans in the corresponding passages.
Supporting spans are labeled positive ( yi= 1), and
other spans are labeled negative ( yi= 0). When
the evidence label cannot be determined reliably,
we assign yi=⊥. These uncertain spans remain
visible in the shared context but are excluded from
the training loss.
Variable-sized candidate sets.To address the
degradation in joint decision quality as the number
4

of candidates increases, we construct training ex-
amples with varying candidate counts and evidence
densities. We consider K∈ {8,16,32,64,128}
candidate spans. For each candidate count, we use
three sampling strategies:
•Natural sampling.We uniformly sample the
desired number of candidate spans. This pre-
serves the naturally sparse distribution of evi-
dence within the text. This sampling can also
produce candidate sets with no supporting evi-
dence.
•Fixed-positive sampling.We select exactly
three supporting spans and fill the remaining
K−3 positions with distractors. As Kincreases,
the supporting evidence becomes increasingly
sparse.
•Gold-included sampling.We include up to
K−1 annotated supporting spans and fill the
remaining positions with distractors. This pro-
duces evidence-rich examples requiring multiple
positive decisions within the same context.
These strategies address two challenges of joint
evidence selection. Varying Ktrains the model to
distinguish relevant spans as the number of simul-
taneous decisions increases. Varying evidence den-
sity exposes the model to both sparse and evidence-
rich contexts, rather than restricting training to pre-
dominantly negative candidate sets. Together, they
aim to improve evidence discrimination across dif-
ferent joint selection workloads.
3.2 Joint Decision Model
We develop our model CURATOR on
pplx-decider-flash-v1.2 , a pretrained joint
decision model based on Qwen3.5-4B (Perplexity
AI, 2026a). The model supports multiple decision
readouts within a shared input context. This allows
us to score several candidates without generating a
separate token response for each decision.
Given a question qand candidate set U=
(u1, . . . , u K), we associate each candidate uiwith
a binary decision asking whether it is an supporting
evidence to answer q. The decision model accord-
ingly produces positive and negative logits, ℓ(1)
i
andℓ(0)
i, for each candidate. We define its evidence
score and positive-class probability as
ri=ℓ(1)
i−ℓ(0)
i,
pi=σ(r i/T),(3)
where Tis the decision model’s temperature and σ
is the sigmoid function. A higher value of risug-gests stronger evidence relevance for the candidate
ui.
When the entire candidate set fits within the
model’s input window, all the Kscores are pro-
duced in a single inference. However, when the
candidate pool is larger, it is divided into smaller
groups to fit the context window. The candidates
are scored within each group, and the resulting
scores are merged. Finally, we rank the candidates
based on the score and select them within the given
context budget for retrieval (see Equation 1).
Parameter-efficient adaptation.The pretrained
model combines causal linear-attention layers with
noncausal full-attention layers in transformers. We
adapt the query, key, value, and output projections
of its eight full-attention layers using low-rank
adaptation (LoRA), with rank 16 and scaling pa-
rameter 32. We also train the existing decision
readout while keeping the remaining backbone pa-
rameters frozen. Following this we only update ap-
proximately 3.8 million parameters out of the 4.54-
billion-parameter model, allowing us to specialize
its joint evidence decisions without full-model fine-
tuning.
3.3 Class-Balanced Joint Evidence
Supervision
A central challenge in evidence selection is that the
actual supporting spans may constitute a small frac-
tion of the whole candidate set. For example, with
three supporting sentences, increasing Kfrom 8
to 128 reduces the positive proportion from 37.5%
to 2.3%. Under ordinary binary cross-entropy loss,
this large number of negative decisions can domi-
nate the training objective.
We address this imbalance using masked, class-
balanced supervision over the joint decision read-
outs. For each labeled class c∈ {0,1} , let
Ic={i:y i=c} denote its candidate indices, and
letC={c:|I c|>0} denote the classes present
in the request. Using p(1)
i=piandp(0)
i= 1−p i,
we define
Levidence =−X
c∈C1
|Ic|X
i∈Iclogp(c)
i.(4)
In our training, we skip supervision for candi-
dates with uncertain labels (y i=⊥).
Although the loss supervises individual readouts,
all decisions are made from the shared candidate
context and update the same model parameters.
The model therefore learns to identify supporting
5

evidence in the presence of other candidates, rather
than learning only to classify isolated sentences.
Combined with variable-sized training sets, this su-
pervision directly targets the deterioration in joint
evidence discrimination observed in Section 2.3.
3.4 Preserving General Decision Capabilities
Adapting a pretrained decision model for evidence
selection may degrade its original decision-making
capabilities. To mitigate this, we use general
decision-task replay (Buzzega et al., 2020) with
examples covering binary classification, multiple-
choice selection, and graded scoring. These tasks
cover different decision /classification formats sup-
ported by the pretrained decision model (Sileo,
2024; Perplexity AI, 2026a). Since the model’s
complete original training data is not publicly avail-
able, we use labeled public benchmarks for replay.
We combine supervision from ground-truth la-
bels with knowledge distillation from the frozen
pretrained model:
Lreplay =λ CE[−logPτ
θ(y|x)]
+λKDτ2DKL(Pτ
0(· |x)∥Pτ
θ(· |x)),
(5)
where (x, y) is a labeled replay example, and Pτ
0
andPτ
θdenote the frozen pretrained and adapted
model distributions at temperature τ. We set
λCE=λ KD= 0.5.
The overall training loss averages the evidence
and replay objectives over their respective exam-
ples:
Ltrain=1
|B|X
x∈BL(x),(6)
where
L(x) =(
Levidence (x), x∈ B evidence ,
Lreplay (x), x∈ B replay.(7)
We sample 60% of training examples from joint
evidence-selection tasks and 40% from general de-
cision tasks.
Additional data-construction details, model im-
plementation, training objectives, and ablation anal-
yses are provided in Appendix B.
4 Experiments
We conduct experiments covering four broad re-
search questions (RQ) that highlight the generaliza-
tion capability, efficiency, and downstream RAG
performance of CURATOR.•RQ1: Evidence selection and efficiency (Sec-
tion 4.2).In a fixed token budget setting, is it
possible to recover as much relevant evidence
as possible while retaining the computational
efficiency of joint inference? To answer this,
we measure evidence recall, scoring latency, and
input-token cost on QAMPARI.
•RQ2: Generalization Capability (Section 4.3).
Is our method generalizable to unseen domains
and evidence-selection tasks? We evaluate
evidence recall and complete-set coverage on
datasets used for fact verification (FEVEROUS),
financial documents (MultiHiertt), and legal con-
tracts (CUAD).
•RQ3: Downstream performance in a RAG
setting(Section 4.4).In a budget-constrained
Retrieval-Augmented Generation (RAG) setting,
does better evidence retrieval improve the end-
to-end answer quality? Using a 120B parameter
LLM for generation in a RAG setup, we evaluate
normalized exact match scores under different
context budgets on MuSiQue and MultiHiertt.
•RQ4: Joint-decision scalability (Section 4.5).
Can CURATOR maintain performance as the
number of simultaneous decisions increase? We
measure and report AUROC on FEVEROUS and
QAMPARI using varying sizes of controlled can-
didate sets.
4.1 Experimental Setup
Evaluation datasets.We evaluate evidence se-
lection across multi-answer QA (Amouyal et al.,
2023), fact verification (Aly et al., 2021), financial
documents (Hendrycks et al., 2021), and legal con-
tracts (Zhao et al., 2022). We additionally evaluate
on downstream answer generation on MuSiQue
(Trivedi et al., 2022) and MultiHiertt (Zhao et al.,
2022). Table 1 summarizes the evaluation datasets
and their evidence characteristics.
Models and baselines.We compare the follow-
ing evidence-selection methods:
•Original Joint Decider.The pretrained
pplx-decider-flash-v1.2 model (Perplexity
AI, 2026a), before evidence-specific adaptation.
•Span-level Rerankers.Qwen3-Reranker-
4B (Zhang et al., 2025) and BGE-reranker-v2-
gemma (BAAI, 2024; Chen et al., 2024), ap-
plied to individual evidence spans. Generally,
rerankers are used to rank chunks/passage but to
get a fair comparison with our method, we also
use them to score spans.
6

Table 1:Evaluation dataset statistics.Candidate and
gold counts are means per question. Density is the
mean per-question fraction of candidate units contain-
ing annotated evidence. Pool coverage is the mean
fraction of annotated evidence available before selec-
tion. FEVEROUS includes annotated evidence pages
and eight BM25 distractor pages per claim; its full cov-
erage is by construction.
Dataset Questions Cand./q Gold/q Density Pool cov.
QAMPARI 855 56.51 15.31 18.13% 76.13%
FEVEROUS 6,808 343.0 3.94 1.29% 100.00%
MuSiQue 2,417 75.5 10.46 13.91% 100.00%
MultiHiertt 1,044 119.0 3.37 3.62% 93.89%
CUAD 525 166.2 2.26 1.41% 96.55%
•Passage/Chunk-level Reranker.Qwen3-
Reranker-4B applied to larger retrieved chunks
rather than individual spans. Exactly how it is
supposed to be applied.
•CURATOR (Ours).The adapted joint decision
model described in Section 3.
We also evaluate separate context-aware scoring,
defined in Section 2.2 (Eqn 2), using the original
pretrained decision model.
Evaluation candidate pools.Within each exper-
iment, methods use the same retrieved source pool.
For QAMPARI, we divide the top-30 retrieved pas-
sages into spans of up to three consecutive sen-
tences. For FEVEROUS, we combine annotated
evidence pages with eight BM25 distractor pages
and extract sentence-level and table-cell units as
spans. MuSiQue and CUAD use sentence-level
units, while MultiHiertt uses sentences and table
cells. Downstream RAG evaluation uses up to 20
retrieved passages per question. Details of candi-
date construction and large-pool scoring are pro-
vided in Appendix B.3.
Context budgets and metrics.We
evaluate applicable context budgets
B∈ {64,128,256,512,1024,2048} tokens.
Evidence Recallmeasures the fraction of anno-
tated supporting units selected, divided by the
number of gold evidences in the denominator.
Complete-Set Coverageindicates the fraction
of questions for which all of the possible gold
evidence has been selected. FEVEROUS reports
pooled gold-unit recall, while other datasets use
mean per-question recall. In conjunction with this,
we reportAUROCfor evidence discrimination at
scale,Normalized Answer Matchfor downstream
RAG generation. For demonstrating efficiency,
we reportscoring latencyandinput-tokenconsumption.
4.2 Evidence Selection Quality and Efficiency
We evaluate whether CURATOR improves context-
aware evidence selection without repeatedly pro-
cessing the retrieved context for each candidate.
Table 2 reports evidence recall across three context
budgets on 855 QAMPARI development questions,
alongside the computational cost of scoring the
candidate pool.
Accuracy–efficiency trade-off.Separate
context-aware scoring achieves higher recall than
the original joint decider, suggesting that the
pretrained model makes more accurate decisions
when evaluating one candidate at a time. However,
this requires repeated full-context inference.
CURATOR improves joint decision quality,
achieving 17–33% higher recall than the original
joint decider and 9–16% higher recall than separate
context-aware scoring across the three budgets.
Moreover, CURATOR is 21.7× faster and uses
25.7× fewer scoring-input tokens than separate
context-aware scoring, while maintaining the
computational cost of the original joint inference.
Comparison with specialized rerankers.CU-
RATOR also outperforms specialized relevance-
ranking models. At B= 1024 , it achieves 64.8%
and 45.5% higher recall than Qwen3-Reranker and
BGE-Reranker at span level, respectively. Span-
level Qwen3-Reranker consistently outperforms its
chunk-level counterpart, highlighting the advantage
of fine-grained evidence selection. Although these
rerankers have lower scoring latency, CURATOR
achieves substantially higher evidence recall.
Impact of context budget.CURATOR’s advan-
tage is largest under tighter context budgets. Its
relative gain over separate context-aware scoring
increases from 8.7% at B= 2048 to 15.7% at
B= 512 , while its gain over the original joint
decider increases from 17.1% to 32.6%. This sug-
gests that accurate evidence selection is particu-
larly valuable when the available context is limited.
Complementing this result, Section 4.5 shows that
CURATOR also maintains stronger evidence dis-
crimination as the number of simultaneous deci-
sions increases.
4.3 Generalization Across Tasks and Domains
We next examine whether CURATOR can iden-
tify supporting evidence in tasks and document
7

Table 2:Evidence selection quality and efficiency on QAMPARI.Evidence recall is reported at three selected-
context budgets. Scoring latency is averaged per question, while input tokens are median totals per question.
Candidate scoring is performed once, independently of the selected-context budget. Results are from the development
set, which overlaps with checkpoint-selection data.
Evidence Recall↑Scoring Cost↓
MethodB= 512B= 1024B= 2048Time (s/q) Input tokens/q
Original Joint Decider 0.2763 0.4279 0.5791 0.909 11,411
Separate Context-Aware 0.3166 0.4844 0.6242 20.065 293,296
Qwen3-Reranker (Span) 0.1874 0.3316 0.5230 0.688 6,664
BGE-Reranker (Span) 0.2297 0.3757 0.5462 0.301 7,046
Qwen3-Reranker (Chunk) 0.1827 0.3042 0.4815 0.449 5,396
CURATOR (Ours) 0.3663 0.5466 0.67840.925 11,411
Table 3:Generalization of evidence selection across tasks and domains.Evidence recall measures the fraction of
annotated supporting evidence retained, while complete-set coverage requires all annotated gold units for a question
to be selected.
Evidence Recall (%)↑Complete-Set Coverage (%)↑
Dataset MethodB= 128B= 512B= 2048B= 128B= 512B= 2048
FEVEROUS Original Joint Decider 33.4 57.0 70.8 12.9 33.5 47.7
Qwen3-Reranker (Span) 40.0 74.6 91.3 18.3 55.3 79.9
CURATOR 50.4 84.7 96.2 26.8 70.4 90.4
MultiHiertt Original Joint Decider 53.3 68.2 82.2 23.0 34.3 54.5
Qwen3-Reranker (Span) 57.9 80.6 91.2 27.1 50.6 72.9
CURATOR 64.1 84.5 92.5 31.7 58.3 76.3
CUAD Original Joint Decider 59.4 78.0 89.6 49.3 66.7 81.0
Qwen3-Reranker (Span) 52.2 78.5 91.8 41.7 67.6 84.6
CURATOR 68.0 86.7 94.4 57.5 79.8 89.9
domains not used for evidence-selection training.
We evaluate FEVEROUS on fact verification, Mul-
tiHiertt on financial reports, and CUAD on legal
contracts. The latter two differ substantially from
the Wikipedia-based training data. Table 3 com-
pares evidence recall and complete-set coverage at
three representative context budgets.
Generalization to fact verification.On
FEVEROUS, CURATOR improves evidence se-
lection despite the sparse supporting evidence and
additional BM25 distractors. At B= 512 , CURA-
TOR achieves 48.6% higher recall than the original
joint decider, while complete-set coverage more
than doubles. These gains persist across all evalu-
ated budgets, demonstrating transfer to fact verifi-
cation involving both textual and tabular evidence.
Generalization across domains.CURATOR
also improves evidence recall on financial reports
and legal contracts, despite neither domain con-
tributing to its evidence-selection training. At
B= 2048 , CURATOR achieves 1.5% higher re-
call than the strongest baseline on MultiHiertt and
2.8% on CUAD. As the context budget decreases
toB= 128 , these improvements increase to 10.7%
and 14.4%, respectively. This suggests that the ben-
efits of evidence selection transfer across domains
and are particularly valuable under restrictive con-Table 4:Downstream answer quality.Normalized an-
swer match (%) across context budgets (tokens). Higher
is better.
Method Context budget (tokens)
128 512 1024 2048
MuSiQue
Qwen3 (Chunk) 19.3 47.0 53.0 56.6
Qwen3 (Span) 39.7 52.4 53.7 56.5
Original Joint 40.3 47.1 48.7 56.6
CURATOR 48.7 55.7 55.0 56.8
MultiHiertt
Qwen3 (Chunk) 9.0 13.8 22.7 37.5
Qwen3 (Span) 42.4 53.2 54.755.2
Original Joint 38.3 48.0 49.5 53.5
CURATOR 48.4 55.5 55.855.1
text budgets.
4.4 Impact on Downstream RAG
Performance
We next examine whether improved evidence selec-
tion translates into better downstream answers. We
evaluate MuSiQue, MultiHiertt using 120B sized
frontier LLM. Within each dataset, all methods use
the same retrieved passages and answering model,
differing only in the selected context.
Improving downstream answer quality.CU-
RATOR achieves the best performance on both
datasets at budgets between 128 and 1,024 tokens.
8

AtB= 256 , it improves answer accuracy over
the strongest baseline by 20.4% on MuSiQue and
9.1% on MultiHiertt. These gains demonstrate that
improved evidence selection translates into better
downstream generation across different tasks and
answering models.
Effect of context budget.CURATOR’s advan-
tage is strongest under constrained budgets. At
B= 128 , its relative improvements over the
strongest baselines reach 20.9% on MuSiQue and
14.0% on MultiHiertt, decreasing to 6.4% and
4.3% at B= 512 . CURATOR also approaches
or slightly exceeds the unfiltered whole-pool refer-
ence using substantially less context, demonstrat-
ing the benefit of selective evidence admission.
Evidence quality versus answer accuracy.Bet-
ter evidence recovery does not translate proportion-
ally into downstream performance. At B= 512 on
MuSiQue, CURATOR improves complete-set cov-
erage of annotated supporting sentences by approx-
imately 31× over the original joint decider, while
answer accuracy improves by 18.3%. This differ-
ence highlights the role of the answering model in
effectively using the selected evidence.
4.5 Scalability of Joint Evidence Decisions
We examine whether CURATOR addresses the
decline in joint decision quality identified in Sec-
tion 2.3. To evaluate evidence discrimination inde-
pendently of retrieval coverage, we construct con-
trolled pools from QAMPARI and FEVEROUS,
each containing three supporting units and K−3
distractors. Pools are sampled independently at
each K, and both models score identical candi-
dates in a single joint request without truncation.
We measure pooled evidence-membership AUROC
asKincreases.
Scaling to larger candidate sets.Figure 3 shows
that the original joint decider loses discrimination
asKgrows, while CURATOR maintains strong
performance. The AUROC gap widens monotoni-
cally from 0.125 to0.271 on QAMPARI ( K= 8
to256), and from 0.048 to0.189 on FEVEROUS
(K= 8 to128). CURATOR’s advantage thus
grows with the joint decision workload.
The improvement transfers to FEVEROUS,
which was not used for evidence-selection training,
and extends to K= 256 on QAMPARI, beyond
the largest training candidate count ( K≤128 ).
Because pools are independently sampled acrossK, these results demonstrate stronger large-set ev-
idence discrimination without isolating decision
count as the sole cause.
4.6 Ablations and Preservation of General
Capabilities
Ablation studies.We investigate which training
choices contribute to CURATOR’s improved joint
decision scalability. Pointwise evidence adaptation
improves discrimination but retains the original
model’s degradation as the candidate count grows.
In contrast, joint training with small candidate sets
largely eliminates this decline, while variable- K
training further improves large-set discrimination.
These results highlight the importance of joint evi-
dence supervision. Detailed results are provided in
Appendix C.
Preservation of general capabilities.We eval-
uate whether evidence-specific adaptation affects
the pretrained model’s original decision capabili-
ties. Across seven general decision benchmarks
covering 19,631 examples, CURATOR achieves
a macro-average of 0.8810 compared with 0.8733
for the original model. No individual task shows
a statistically detectable difference, indicating no
detectable degradation in the measured capabili-
ties. Detailed per-task results are provided in Ap-
pendix D.
5 Related Work
Fine-grained retrieval and context compression.
Retrieval-augmented generation (RAG) improves
language models by incorporating retrieved infor-
mation (Lewis et al., 2020; Guu et al., 2020; Izacard
and Grave, 2021). Conventional rerankers assess
the relevance of retrieved passages or spans, typi-
cally scoring each candidate separately against the
query (Nogueira and Cho, 2020; Nogueira et al.,
2020; Khattab and Zaharia, 2020; Zhang et al.,
2025). Listwise reranking instead considers multi-
ple candidates together, but often relies on repeated
generation to produce rankings (Sun et al., 2023;
Pradeep et al., 2023). Related work on context
compression reduces unnecessary retrieved infor-
mation through extractive selection or token-level
pruning (Xu et al., 2024; Jiang et al., 2023, 2024;
Li et al., 2023; Hwang et al., 2025). Unlike these
approaches, we focus on making context-aware
decisions for multiple evidence spans jointly, se-
lecting original text under a fixed token budget.
9

8 16 32 64 128 256
Simultaneous decisions (K)0.70.80.91.0Evidence-membership AUROC
(a) QAMPARI
8 16 32 64 128
Simultaneous decisions (K)
(b) FEVEROUS
Frozen joint CURATORFigure 3:Scalability of joint evidence decisions.Pooled evidence-membership AUROC for the Original Joint
Decider and CURATOR on (a) QAMPARI and (b) FEVEROUS as the number of simultaneous decisions K
increases. Each controlled pool contains three supporting units and K−3 distractors, independently sampled at
eachK. Both models evaluate the same pools without truncation. Higher is better.
Decision models for joint scoring.Decision
models score predefined choices directly, avoid-
ing the need to generate textual judgments (Li and
Wagle, 2026; Geuter et al., 2026). Recent multi-
readout models extend this capability to multiple
decisions within a shared inference (Perplexity AI,
2026a,b). This makes them suitable for evidence
selection, where the relevance of one span may
depend on information in other retrieved passages.
However, we find that pretrained decision models
lose discrimination as the number of simultaneous
decisions increases. We address this limitation by
adapting a pretrained decision model through joint
evidence supervision over variable-sized candidate
sets.
Related decision-based retrieval.Concurrent
work on DecisionRAG (Wan et al., 2026) also uses
parallel decisions from a pretrained decision model
and observes limitations when evaluating large can-
didate sets. DecisionRAG retains a frozen model
and uses hierarchical retrieval to divide large candi-
date pools into smaller groups. In contrast, CURA-
TOR improves the model’s ability to make many
decisions jointly through training, while operating
on a fixed retrieved pool. These approaches are
complementary: improving joint decision capac-
ity could also benefit hierarchical retrieval. Unlike
DecisionRAG, we evaluate fine-grained evidence
selection under token budgets and its impact on
downstream RAG performance. The two methods
use different evaluation settings, so we do not make
a direct performance comparison.6 Conclusion
We introduced CURATOR, a span-level evidence
selection framework that adapts a pretrained joint
decision model to identify relevant information
from retrieved documents under a limited context
budget. By training the model to make multiple
evidence decisions jointly, CURATOR improves
selection quality while retaining the computational
efficiency of shared inference. Our experiments
demonstrate consistent improvements in evidence
recall and complete-set coverage, particularly un-
der tight token budgets. These improvements gen-
eralize to fact verification, financial documents,
and legal contracts, and translate into better down-
stream RAG performance. Controlled scaling ex-
periments further show that CURATOR maintains
strong evidence discrimination as the number of
candidates increases, addressing a key limitation
of pretrained joint decision models.
Our results highlight the potential of decision
models for fine-grained context selection, moving
beyond passage-level retrieval toward more selec-
tive and efficient use of retrieved information. CU-
RATOR’s joint scoring remains reliable at large
candidate counts, suggesting it could extend nat-
urally to agentic settings by re-ranking the accu-
mulated evidence pool as new tool outputs arrive,
dynamically sharing the token budget between re-
trieved evidence and the agent’s working context.
We leave this multi-step setting to future work.
10

Limitations
Acknowledgments
We used generative AI tools to assist with improv-
ing the clarity and presentation of the manuscript,
generating illustrative figures, and formatting. All
AI-assisted content was reviewed and verified by
the authors, who take full responsibility for the
final manuscript and its scientific claims.
References
Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull,
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, Oana Cocarascu, and Arpit
Mittal. 2021. The fact extraction and VERification
over unstructured and structured information
(FEVEROUS) shared task. InProceedings of the
Fourth Workshop on Fact Extraction and VERifica-
tion (FEVER), pages 1–13, Dominican Republic.
Association for Computational Linguistics.
Samuel Amouyal, Tomer Wolfson, Ohad Rubin, Ori
Yoran, Jonathan Herzig, and Jonathan Berant. 2023.
QAMPARI: A benchmark for open-domain questions
with many answers. InProceedings of the Third
Workshop on Natural Language Generation, Evalua-
tion, and Metrics (GEM), pages 97–110, Singapore.
Association for Computational Linguistics.
BAAI. 2024. Bge reranker v2 gemma.
https://huggingface.co/BAAI/
bge-reranker-v2-gemma.
Pietro Buzzega, Matteo Boschini, Angelo Porrello,
Davide Abati, and SIMONE CALDERARA. 2020.
Dark experience for general continual learning: a
strong, simple baseline. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 15920–
15930. Curran Associates, Inc.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Jonathan Geuter, Andrew Cheng, and David Alvarez-
Melis. 2026. Releasing the inner jev: A sim-
ple recipe for open decision models.Preprint,
arXiv:2610.17205.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. Realm: retrieval-
augmented language model pre-training. InProceed-
ings of the 37th International Conference on Machine
Learning, ICML’20. JMLR.org.Dan Hendrycks, Collin Burns, Anya Chen, and
Spencer Ball. 2021. Cuad: An expert-annotated
nlp dataset for legal contract review.Preprint,
arXiv:2103.06268.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun
Song, SeungYoon Han, and Jong C. Park. 2025.
EXIT: Context-aware extractive compression for en-
hancing retrieval-augmented generation. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2025, pages 4895–4924, Vienna, Austria.
Association for Computational Linguistics.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023. LLMLingua: Compressing
prompts for accelerated inference of large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 13358–13376, Singapore. Association for
Computational Linguistics.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dong-
sheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.
2024. LongLLMLingua: Accelerating and enhanc-
ing LLMs in long context scenarios via prompt com-
pression. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1658–1677, Bangkok,
Thailand. Association for Computational Linguistics.
O. Khattab and Matei A. Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert.Proceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Yinheng Li and Justin Wagle. 2026. Llm-as-jev: Llms
are already jev-style decision models – when and
how to fine-tune them.Preprint, arXiv:2610.02076.
11

Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 6342–6353, Singa-
pore. Association for Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage
re-ranking with bert.Preprint, arXiv:1901.04085.
Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and
Jimmy Lin. 2020. Document ranking with a pre-
trained sequence-to-sequence model. InFindings
of the Association for Computational Linguistics:
EMNLP 2020, pages 708–718, Online. Association
for Computational Linguistics.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
2024. Memgpt: Towards llms as operating systems.
Preprint, arXiv:2310.08560.
Perplexity AI. 2026a. pplx-decider-flash-v1.2. Hugging
Face Model Repository.
Perplexity AI. 2026b. pplx-decider-v1.2. Hugging Face
Model Repository.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023. Rankzephyr: Effective and robust zero-
shot listwise reranking is a breeze!Preprint,
arXiv:2312.02724.
Damien Sileo. 2024. tasksource: A large collection
of NLP tasks with a structured dataset preprocess-
ing framework. InProceedings of the 2024 Joint
International Conference on Computational Linguis-
tics, Language Resources and Evaluation (LREC-
COLING 2024), pages 15655–15684, Torino, Italia.
ELRA and ICCL.
Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong
Yao, Yiming Yang, and Jiecao Chen. 2026. Scaling
long-horizon agent via context folding. InProceed-
ings of the 43rd International Conference on Machine
Learning, volume 306 ofProceedings of Machine
Learning Research, pages 116894–116910. PMLR.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is ChatGPT good at search?
investigating large language models as re-ranking
agents. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 14918–14937, Singapore. Association for
Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.Transactions of the Association for Computational
Linguistics, 10:539–554.
Chao Wan, Paul Jünger, Beixi Du, Bodong Liu, Nava-
neetha Krishnan, and Kilian Q. Weinberger. 2026.
Retriever-free retrieval: Navigating documents with
decision models.Preprint, arXiv:2610.35868.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. Re-
comp: Improving retrieval-augmented lms with con-
text compression and selective augmentation. InIn-
ternational Conference on Learning Representations,
volume 2024, pages 43478–43502.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang.
2022. MultiHiertt: Numerical reasoning over multi
hierarchical tabular and textual data. InProceedings
of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 6588–6600, Dublin, Ireland. Association for
Computational Linguistics.
A Diagnostics of the Decision Models
A.1 Accuracy–Efficiency Trade-off of Frozen
Scoring Strategies
We compare independent scoring, separate context-
aware scoring, and joint context-aware scoring us-
ing the same frozen pretrained decision model. All
three strategies use identical candidate pools, scor-
ing definitions, and budgeted selection procedures
within each dataset.
We evaluate matched 200-example panels from
FEVEROUS and QAMPARI. The FEVEROUS
panel uses annotated evidence pages with a 120-
unit candidate cap and is distinct from the full
distractor-augmented evaluation in Section 4.3.
The QAMPARI panel is likewise a smaller diagnos-
tic sample rather than the full development evalua-
tion.
Separate context-aware scoring achieves the
highest recall on both panels, showing that access
to other candidate evidence can improve individual
judgments. However, the pretrained model does not
12

Table 5:Accuracy–efficiency trade-off of frozen evi-
dence decisions.Evidence recall at a selected-context
budget of 512 tokens for FEVEROUS and 1,024 to-
kens for QAMPARI, alongside mean scoring latency
per question. Each dataset contains 200 examples.
Dataset Scoring strategy Recall↑Time (s)↓
FEVEROUS Independent .743 1.48
Separate context-aware.81414.93
Joint context-aware .676.85
QAMPARI Independent .428 1.57
Separate context-aware.50420.25
Joint context-aware .460.91
consistently preserve this advantage when making
all decisions jointly. Joint context-aware scoring
performs below independent scoring on FEVER-
OUS and remains below separate context-aware
scoring on QAMPARI.
The computational difference is substantial. On
QAMPARI, joint scoring reduces latency by ap-
proximately 22× compared with separate context-
aware scoring and processes 11,408 median in-
put tokens per question rather than 293,227. On
FEVEROUS, the corresponding latency reduc-
tion is approximately 18×. These measurements
use matched single-GPU inference without cross-
question batching or caching. Thus, joint inference
reduces repeated computation, but the quality of
pretrained joint evidence decisions remains a limi-
tation.
A.2 Evidence Discrimination with Increasing
Candidate Sets
To examine how joint decision quality changes
with candidate set size, we evaluate the frozen
pplx-decider-flash-v1.2 model on controlled
candidate pools from QAMPARI and FEVEROUS.
Each pool contains three annotated supporting units
andK−3 distractors. The supporting units are
included by construction, allowing us to measure
evidence discrimination independently of whether
the retrieval system found them.
For QAMPARI, we evaluate 424 questions
with two independently sampled candidate sets
per question at each K, covering K∈
{8,16,32,64,128,256} . For FEVEROUS, we
evaluate 1,000 questions with two sampled sets
per question, covering K∈ {8,16,32,64,128} .
Candidate pools are sampled independently at each
Kand are not nested. Each pool is scored in a
single joint inference without truncation.
We compute evidence-membership scores using
the difference between the positive and negative
None +6k +14kExtra irrelevant context tokens
8
16
32
64
128Simultaneous decisions, K0.800 0.780 0.769
0.819 0.802 0.789
0.738 0.751 0.750
0.677 0.689 0.714
0.652 0.679 0.690
0.6250.6500.6750.7000.7250.7500.7750.8000.825
AUROCFigure 4:Decision count versus added context length.
Mean within-set AUROC on 15 paired QAMPARI ques-
tions containing three supporting candidates. Rows vary
the number of candidate decisions, while columns add
irrelevant text without increasing the number of deci-
sions. Higher is better.
decision logits. We report pooled AUROC across
the candidate scores and labels at each K. AUROC
measures the ability to rank supporting evidence
above distractors without depending on a particular
selection threshold or token budget.
As shown in Figure 2, the pretrained model’s dis-
crimination deteriorates as the candidate set grows.
On QAMPARI, AUROC decreases from 0.785 at
K= 8 to 0.683 at K= 256 . On FEVEROUS,
it decreases from 0.893 at K= 8 to 0.771 at
K= 128.
These results reveal a scaling limitation of pre-
trained joint evidence decisions. However, increas-
ingKalso changes the distractor composition and
input length. The observed decline therefore cannot
be attributed exclusively to the number of requested
decisions.
A.3 Effect of Decision Count and Input
Length
The deterioration in joint evidence discrimination
may result from increasing decision count, longer
inputs, or additional distractors. To examine the
role of input length, we conduct a diagnostic using
15 paired QAMPARI questions, each with three
supporting candidates.
We vary the candidate count Kfrom 8 to 128.
At each fixed K, we additionally append 6,000 or
14,000 irrelevant tokens to the shared context with-
out introducing new decisions. Every candidate set
is evaluated in a single joint request without trun-
cation. We report AUROC computed within each
candidate set and averaged across paired examples.
13

Without padding, AUROC decreases from 0.800
atK= 8 to 0.652 at K= 128 . In contrast, adding
irrelevant text at a fixed Kproduces smaller, non-
monotonic changes. At K= 128 , adding 14,000
irrelevant tokens changes AUROC from 0.652 to
0.690 rather than causing a further decrease.
These observations suggest that increasing input
length alone is insufficient to explain the deteriora-
tion. However, the sample is small, and increasing
Kalso changes the distractor composition. The
experiment therefore does not establish a causal
effect of decision count alone.
B CURATOR: Implementation Details
and Additional Analyses
B.1 Evidence Annotation and Candidate
Construction
We construct evidence-selection training exam-
ples from QAMPARI, HotpotQA, and 2WikiMulti-
HopQA. Each example contains a question, a set
of candidate sentences, and evidence-membership
annotations. The candidate sentences are extracted
from the passages associated with each question.
The datasets provide different forms of evidence
annotations. HotpotQA and 2WikiMultiHopQA
identify supporting facts at the sentence level. For
QAMPARI, we align annotated proof text with sen-
tences from the corresponding passages. These
annotations provide the positive candidates used
for training.
We distinguish three types of candidate labels:
positive, verified negative, and uncertain. Positive
candidates are identified by supporting-evidence
annotations. Negative labels are assigned only to
candidates treated as verified non-evidence by the
data-construction procedure. Candidates whose
evidence membership cannot be established are
marked uncertain.
Uncertain candidates are retained in the model
input because they are part of the retrieved con-
text. However, their readouts are excluded from
the training objective. This distinction is important
when evidence annotations are incomplete: the ab-
sence of an annotation does not necessarily mean
that a sentence is irrelevant.
Sampling candidate sets.For each training
question, we construct joint requests with K∈
{8,16,32,64,128} candidate sentences. The three
sampling strategies described in Section 3.1 expose
the model to different evidence distributions.Fixed-positive sampling constructs requests con-
taining three supporting sentences and K−3 dis-
tractors. For example, a request with K= 8
contains 37.5% supporting evidence, whereas a
request with K= 128 contains only 2.3%. This
creates increasingly sparse selection problems with-
out changing the number of supporting candidates.
Natural sampling draws from the underlying can-
didate pool without requiring a fixed number of
positives. It therefore includes requests with vary-
ing evidence density, including requests without
annotated supporting evidence.
Gold-included sampling prioritizes the annotated
supporting sentences, subject to the capacity K,
and fills the remaining positions with distractors.
Unlike fixed-positive sampling, this strategy allows
requests containing different numbers of support-
ing sentences.
The combination of these strategies separates
two important sources of variation: the number
of decisions made jointly and the proportion of
decisions that correspond to supporting evidence.
B.2 Model Adaptation and Training
Configuration
We initialize CURATOR from
pplx-decider-flash-v1.2 , a 4.54B-parameter
pretrained decision model. We retain its existing
decision interface, including the binary readouts
used to score evidence membership. Each
candidate is associated with a positive and negative
decision option, and the positive-minus-negative
logit margin defines its ranking score.
We use low-rank adaptation on the query, key,
value, and output projections of the model’s eight
noncausal full-attention layers. The LoRA rank is
16 and its scaling parameter is 32. The existing de-
cision readout is also trainable, while the remaining
backbone parameters are frozen. In total, approxi-
mately 3.8 million parameters are updated.
Training interleaves two request types. Evidence
requests supervise multiple candidate decisions
using the masked, class-balanced loss in Eq. 4.
General-task replay requests use the supervised
and distillation objective in Eq. 5.
Evidence requests constitute 60% of sampled
training requests, while replay requests account
for the remaining 40%. These proportions are de-
fined at the request level rather than the individual
readout level. Consequently, an evidence request
containing 128 candidate decisions does not count
as 128 separate training requests.
14

For general-task replay, the frozen pretrained
model supplies the teacher distribution over the
valid options of each task. Supervised labels
and teacher distributions receive equal objective
weights, with λCE=λ KD= 0.5 . The teacher is
not used to supervise evidence membership, since
the objective is to improve upon its pretrained evi-
dence decisions rather than reproduce them.
The final CURATOR checkpoint used in our
evaluations is obtained after 2,000 training updates.
B.3 Inference with Large Candidate Pools
At inference time, the model receives a question
and a set of candidate evidence units. Each candi-
date has a binary decision readout, and its evidence
score is computed using Eq. 3.
When the candidate set fits within the input con-
text window, all decisions are evaluated jointly in
one inference request. This avoids repeatedly pro-
cessing the full candidate context for every individ-
ual decision.
Some retrieved pools exceed the model’s context
window. In these cases, we divide the candidate
pool into smaller groups and perform joint scoring
within each group. In the full FEVEROUS evalua-
tion, we use groups of at most 200 candidates. The
resulting scores are combined into a single ranking
across the original candidate pool.
This distinction is important for interpreting
computational efficiency. Joint inference elimi-
nates repeated full-context evaluation within each
group, but a sufficiently large pool may still require
multiple inference requests. Therefore, large-pool
evaluation should not be interpreted as processing
every candidate in one model invocation.
Grouping also introduces a limitation: scores
produced in different groups do not share the same
complete candidate context. Their relative calibra-
tion may differ from scores produced by a hypo-
thetical single inference over the entire pool. In our
matched FEVEROUS comparison, the pretrained
joint model and CURATOR use the same grouping
procedure.
Once all candidates have been scored, we sort
them in descending score order. Starting with an
empty selection, we admit a candidate if its token
cost fits within the remaining budget B; otherwise,
it is skipped. This continues until all candidates
have been considered.
The scoring and admission stages are separate.
The same computed scores can therefore be reused
across different token budgets without additionalTable 6:Effect of evidence-training strategies.Pooled
evidence-membership AUROC on QAMPARI candidate
sets of increasing size. The number of training updates
is shown in parentheses. Higher is better.
MethodK= 8K= 128K= 256
Pretrained joint model .7853 .7076 .6833
Pointwise (500) .8578 .7723 .7560
Joint,K≤16(500) .9057 .9114 .9104
Variable-K, no replay (500) .9002 .9322 .9381
CURATOR (2000).9104 .9450 .9541
model inference.
B.4 Additional Training Ablations
We examine whether the improvement in evidence
discrimination comes from evidence-specific fine-
tuning alone or from training multiple decisions
jointly across different candidate set sizes.
We compare pointwise evidence supervision,
joint supervision restricted to small candidate sets,
and joint supervision with variable K. We also re-
port the final CURATOR checkpoint trained with
general-task replay.
Table 6 shows that pointwise evidence fine-
tuning improves discrimination relative to the pre-
trained model, but performance still declines as the
number of jointly evaluated candidates increases.
Training on small joint candidate sets produces
a different behavior. Its AUROC remains relatively
stable as Kincreases, suggesting that joint evi-
dence supervision is important for learning to dis-
tinguish supporting evidence in shared contexts.
Variable-sized joint supervision further improves
performance at larger candidate counts. Its AU-
ROC increases from 0.9002 at K= 8 to 0.9381 at
K= 256 . Notably, K= 256 exceeds the largest
candidate count used during training.
The final CURATOR checkpoint achieves the
highest discrimination among the evaluated con-
figurations. However, these runs differ in training
duration and other training choices. The compari-
son therefore supports the importance of joint and
variable-sized supervision but does not isolate the
causal contribution of each component under a fully
matched setting.
B.5 General Decision Capability Preservation
Evidence-specific adaptation may alter the pre-
trained model’s behavior on general decision tasks.
We therefore evaluate whether the adapted model
retains performance across its original decision for-
mats.
15

The evaluation covers seven benchmarks:
BoolQ, RTE, MNLI, ARC-Easy, ARC-Challenge,
CommonsenseQA, and STS-B, comprising 19,631
examples in total. These benchmarks include bi-
nary classification, multiple-choice reasoning, and
semantic similarity scoring.
Across the seven benchmarks, the unweighted
mean of their native evaluation scores is 0.8733 for
the pretrained model and 0.8810 for CURATOR.
This comparison shows that evidence-specific adap-
tation does not produce an overall decline in the
measured general-task performance.
These results are consistent with the objective
of retaining the model’s general decision capabil-
ities during specialization. However, they do not
independently establish that replay is responsible
for this preservation. A matched comparison with
and without replay, evaluated on the same general
decision benchmarks, would be needed to isolate
its contribution.
C Detailed Ablation Studies
We examine how pointwise supervision, joint train-
ing, candidate-set size, and general-task replay
affect evidence discrimination. Table 7 reports
pooled AUROC on QAMPARI at different can-
didate counts.
Effect of joint supervision.Pointwise adapta-
tion improves AUROC at small Kbut exhibits
nearly the same decline as the original model at
larger candidate counts. Joint training with K≤16
largely stabilizes discrimination, while variable- K
training further improves large-set performance.
At matched 500-update training, the full variable-
Krecipe improves mean per-question AUROC
over small- Ktraining by 0.0148, averaged across
K= 64andK= 128on QAMPARI.
Role of replay and experimental limitations.
Removing general-task replay has only a small
effect on evidence discrimination: the no-replay
variant trails the matched 500-update full recipe
by 0.0029 on the primary large- Kmetric. How-
ever, its effect on preserving general capabilities is
not established without a matched retention eval-
uation. Furthermore, pointwise adaptation uses
QAMPARI-only training data, while joint variants
use three datasets. The final checkpoint also re-
ceives more training updates. These comparisons
therefore support the value of joint supervision
without completely isolating each training com-ponent.
D Preservation of General Decision
Capabilities
We evaluate whether adapting the pretrained deci-
sion model for evidence selection affects its orig-
inal capabilities. We compare CURATOR with
the original joint decider on seven general deci-
sion benchmarks covering inference, question an-
swering, multiple-choice reasoning, and semantic
similarity.
General-task performance.CURATOR
achieves comparable performance to the original
model across all seven benchmarks. Although
its point estimates are slightly higher, none of
the individual differences is statistically resolved.
These results indicate no detectable degradation in
the measured capabilities, rather than establishing
equivalence or improved general-task performance.
Decision calibration.We additionally examine
whether adaptation changes decision behavior on
BoolQ. The predicted positive rate changes only
from 0.6183 to 0.6205, while expected calibration
error changes from 0.0136 to 0.0153. These results
suggest that evidence adaptation preserves not only
task performance but also broadly similar decision
behavior on this benchmark.
16

Table 7:Training ablations on QAMPARI.Pooled AUROC at different candidate counts. Ablation variants use
500 updates; the selected CURATOR checkpoint uses 2,000 updates. Higher is better.
Training Configuration UpdatesK= 8K= 128K= 256 ∆ 256−8
Original Joint Decider 0 .7853 .7076 .6833−.1020
Pointwise (K= 1) 500 .8578 .7723 .7560−.1018
Joint (K≤16) 500 .9057 .9114 .9104+.0047
Joint, variableK(no replay) 500 .9002 .9322 .9381+.0379
CURATOR (Ours)2000.9104 .9450 .9541+.0437
Table 8:Preservation of general decision capabilities.
Performance before and after evidence adaptation across
seven benchmarks.
TasknOriginal CURATOR
BoolQ 3,270 .9153 .9174
RTE 277 .9170 .9278
MNLI 9,815 .9127 .9151
ARC-Easy 2,376 .9903 .9916
ARC-Challenge 1,172 .9531 .9539
CommonsenseQA 1,221 .8608 .8657
STS-B 1,500 .5640 .5953
Macro-avg. 19,631 .8733 .8810
17
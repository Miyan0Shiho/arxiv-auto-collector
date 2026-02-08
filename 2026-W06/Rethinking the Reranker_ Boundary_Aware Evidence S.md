# Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation

**Authors**: Jiashuo Sun, Pengcheng Jiang, Saizhuo Wang, Jiajun Fan, Heng Wang, Siru Ouyang, Ming Zhong, Yizhu Jiao, Chengsong Huang, Xueqiang Xu, Pengrui Han, Peiran Li, Jiaxin Huang, Ge Liu, Heng Ji, Jiawei Han

**Published**: 2026-02-03 16:08:23

**PDF URL**: [https://arxiv.org/pdf/2602.03689v1](https://arxiv.org/pdf/2602.03689v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems remain brittle under realistic retrieval noise, even when the required evidence appears in the top-K results. A key reason is that retrievers and rerankers optimize solely for relevance, often selecting either trivial, answer-revealing passages or evidence that lacks the critical information required to answer the question, without considering whether the evidence is suitable for the generator. We propose BAR-RAG, which reframes the reranker as a boundary-aware evidence selector that targets the generator's Goldilocks Zone -- evidence that is neither trivially easy nor fundamentally unanswerable for the generator, but is challenging yet sufficient for inference and thus provides the strongest learning signal. BAR-RAG trains the selector with reinforcement learning using generator feedback, and adopts a two-stage pipeline that fine-tunes the generator under the induced evidence distribution to mitigate the distribution mismatch between training and inference. Experiments on knowledge-intensive question answering benchmarks show that BAR-RAG consistently improves end-to-end performance under noisy retrieval, achieving an average gain of 10.3 percent over strong RAG and reranking baselines while substantially improving robustness. Code is publicly avaliable at https://github.com/GasolSun36/BAR-RAG.

## Full Text


<!-- PDF content starts -->

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust
Retrieval-Augmented Generation
Jiashuo Sun1Pengcheng Jiang1Saizhuo Wang2Jiajun Fan1Heng Wang1Siru Ouyang1Ming Zhong1
Yizhu Jiao1Chengsong Huang3Xueqiang Xu1Pengrui Han1Peiran Li4Jiaxin Huang3Ge Liu1Heng Ji1
Jiawei Han1
Abstract
Retrieval-Augmented Generation (RAG) systems
remain brittle under realistic retrieval noise, even
when the required evidence appears in the top-
Kresults. A key reason is that retrievers and
rerankers optimize solely for relevance, often se-
lecting either trivial, answer-revealing passages
or evidence that lacks the critical information re-
quired to answer the question, without consid-
ering whether the evidence is suitable for the
generator. We propose BAR-RAG , which re-
frames the reranker as a boundary-aware evidence
selector that targets the generator’s Goldilocks
Zone—evidence that is neither trivially easy nor
fundamentally unanswerable for the generator,
but is challenging yet sufficient for inference
and thus provides the strongest learning signal.
BAR-RAG trains the selector with reinforcement
learning using generator feedback, and adopts
a two-stage pipeline that fine-tunes the genera-
tor under the induced evidence distribution to
mitigate the distribution mismatch between train-
ing and inference. Experiments on knowledge-
intensive question answering benchmarks show
thatBAR-RAG consistently improves end-to-end
performance under noisy retrieval, achieving an
average gain of 10.3% over strong RAG and
reranking baselines while substantially improv-
ing robustness. The code is avaliable at https:
//github.com/GasolSun36/BAR-RAG.
1University of Illinois Urbana-Champaign2Hong Kong Uni-
versity of Science and Technology3Washington University in St.
Louis4Texas A&M University. Correspondence to: Jiashuo Sun
<jiashuo5@illinois.edu>.
Preprint. February 4, 2026.
Standard Reranker
Score: Maximize Relevance
Overlooks Generator’s 
weaknesses
Shortcut Learning / Highly rely on Relevance
Ours
Uncertainty: Selector Hard / High Difficulty
Challenges & 
Improves Generator
Robust Reasoning / Closes GapFigure 1.Comparison between standard relevance-based rerankers
and our boundary-aware evidence selection. Standard rerankers
maximize relevance scores but overlook the generator’s weak-
nesses, often encouraging shortcut learning and brittle reasoning
by prioritizing trivial or answer-revealing evidence. In contrast,
our method selects challenging yet solvable evidence based on gen-
erator uncertainty, promoting robust reasoning and reducing the
mismatch between the evidence distributions encountered during
training and inference under noisy retrieval.
1. Introduction
Retrieval-Augmented Generation (RAG) has achieved re-
markable success on knowledge-intensive tasks by ground-
ing large language model (LLM) outputs in retrieved evi-
dence (Li et al., 2025b; Jiang et al., 2025). Yet RAG systems
remain surprisingly brittle when retrieval results are noisy,
partially relevant, or requires multi-step integration, even
when the necessary facts exist somewhere in the top- Kre-
sults (Hsia et al., 2025; Yu et al., 2024; Cuconasu et al.,
2024). In such realistic settings, LLMs often fail to synthe-
size scattered information and instead hallucinate plausible
1arXiv:2602.03689v1  [cs.CL]  3 Feb 2026

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
TriviaQA PopQA HotpotQA 2Wiki020406080100PerformanceRecall@5 vs QA Accuracy
RAG Acc
B-RAG Acc
Recall
TriviaQA PopQA HotpotQA 2Wiki020406080100PerformanceRecall@10 vs QA Accuracy
RAG Acc
B-RAG Acc
Recall
Figure 2.Recall@5 and Recall@10 vs. QA Accuracy across different datasets. Higher retrieval recall does not guarantee higher QA
accuracy. Our method narrows the gap between recall and accuracy.
but incorrect answers. This fragility reveals a fundamental
limitation: retrievers are optimized for relevance, not for
providing evidence that maximally strengthens the genera-
tor’s reasoning.
Current retrievers optimize exclusively for query-document
relevance (Wang et al., 2022; Zhang et al., 2025), creating
two systematic failure modes: they prefer trivial, answer-
revealing passages that encourage shortcut learning, and
they cannot distinguish genuinely unsolvable evidence from
challenging yet sufficient evidence—precisely the kind that
best strengthens generator reasoning. Crucially, existing
retrievers operate without any estimation of the generator’s
competence, creating a severe train–test mismatch: systems
trained on curated evidence face noisy, incomplete retrieval
at deployment, leading to substantial performance degrada-
tion (Sun et al., 2025; Yu et al., 2024). Empirically, this
manifests as a persistent gap between retrieval recall and
end-to-end QA accuracy (Figure 2).
To overcome this limitation, we revisit the role of the
reranker in RAG systems. Rather than treating it as a passive
relevance scorer, we view the reranker as an active evidence
set selector, responsible for choosing combinations of doc-
uments whose joint structure best supports the generator’s
learning and reasoning. Figure 1 contrasts this paradigm
with standard relevance-based reranking. Crucially, evi-
dence sets with similar relevance can induce drastically dif-
ferent learning signals: overly explicit evidence encourages
shortcut learning, while incomplete evidence is fundamen-
tally unlearnable. In contrast, evidence that is challenging
yet sufficient for inference forces the generator to integrate
information and resolve uncertainty.
Building on this perspective, we propose BAR-RAG ,
which instantiates the reranker-as-selector paradigm through
boundary-aware evidence design. Rather than maximiz-
ing relevance, BAR-RAG explicitly targets the generator’s
Goldilocks Zone—evidence sets that are neither trivial norunsolvable, but lie near the generator’s uncertainty boundary.
We operationalize this objective via a two-stage reinforce-
ment learning pipeline: we first train the selector to explore
diverse document combinations, guided by rewards based
on generator uncertainty and task success, while keeping
the generator fixed; we then freeze the selector and fine-tune
the generator on the induced evidence distribution, ensuring
robustness under noisy retrieval.
We evaluate BAR-RAG on a diverse set of knowledge-
intensive question answering benchmarks under realistic
retrieval settings. Our results demonstrate that boundary-
aware evidence selection consistently improves end-to-end
QA performance, yields a significant performance gain of
an average 10.3% over baseline models. Beyond accuracy
gains, we show that BAR-RAG reshapes the evidence diffi-
culty distribution toward the generator’s competence bound-
ary, leading to more effective learning signals and substan-
tially improved robustness compared to relevance-based
retrieval and reranking baselines.
2. Method
2.1. Overview
Our approach consists of a two-stage reinforcement learning
pipeline (Figure 3). In Stage 1, we train a selector to iden-
tify evidence sets that lie within the generator’s “Goldilocks
Zone”—challenging enough to require genuine reasoning,
yet solvable given the generator’s current competence. In
Stage 2, we freeze the selector and fine-tune the generator
under the induced evidence distribution, thereby closing the
train–test gap that plagues standard RAG systems. We de-
scribe each stage below, and introduce an iterative training
scheme that progressively refines evidence selection to bet-
ter match the generator’s evolving competence. We describe
each stage below.
2

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
QueryRetrievedDocumentsStage 2: Generator TrainingTraining PhraseInference PhraseFine-tuned Generator
Answer
QueryRetrievedDocumentsStage 1: Selector Training
Iterative Training
SelectorSelected Evidence Set
Generator
rolloutAnswer Set
selectBoundary RewardRelevance RewardRLfeed
Selector
Generator
generateselectfeed
Selected Evidence Set
AnswerAccuracy RewardCitation RewardRL
Figure 3.Overview of the BAR-RAG training and inference pipeline. During training, we adopt a two-stage framework: (Stage 1) a
selector is trained with reinforcement learning using relevance and uncertainty rewards to identify challenging yet solvable evidence sets,
and (Stage 2) the generator is optimized under the induced evidence distribution using accuracy, formatting, and citation rewards. At
inference time, the trained generator answers questions using retrieved documents, producing robust and high-quality answers.
2.2. Problem Setup
Letπrdenote the selector policy and πgthe generator pol-
icy. Given a query q, letC={c 1, . . . , c n}be the top- n
candidate documents returned by an initial retriever (e.g.,
BM25 (Robertson & Zaragoza, 2009) or a dense retriever).
The selector chooses a subset S={c i1, . . . , c ik} ⊆C with
k < n , and the generator produces an answer aconditioned
on(q, S).
Our goal is twofold. First, we train the selector policy
πrto select evidence sets that maximize the generator’s
reasoning performance, not by providing trivial shortcuts,
but by identifying evidence that is challenging yet solvable.
Second, we train the generator policy πgto produce high-
quality, accurate answers conditioned on the evidence sets
selected by πr, enabling robust reasoning under realistic
and challenging retrieval.
Training-time Filtering.To ensure informative reinforce-
ment learning signals for selector optimization, we apply a
lightweight training-time filtering step that removes queries
that are either trivially solvable or fundamentally unanswer-
able under the retrieved document pool. The key intuition
is that selector learning relies on variance in generator out-
comes: queries that are always answered correctly provide
no incentive for evidence selection, while queries that are
consistently unsolvable yield uniformly low rewards regard-
less of the selected evidence. We characterize trivial and
unanswerable instances based on the empirical behavior of
the generator over multiple rollouts given retrieved docu-
ments, using correctness statistics as a proxy for solvability.
This design makes the filtering procedure domain-agnostic
and inexpensive, as it depends only on generator feedback
rather than task-specific heuristics or additional supervision.The filtering is applied only during selector training to im-
prove reward signal quality; all queries are retained during
evaluation. Full details are provided in Appendix A.6.
2.3. Stage 1: Boundary-aware Selector Training
Standard rerankers optimize for relevance alone, often sur-
facing answer-revealing passages that encourage shortcut
learning. In contrast, we train the selector to target the gen-
erator’s competence boundary—the region where evidence
is neither trivially easy ( ˆp≈1 ) nor impossibly hard ( ˆp≈0 ),
but lies near a target difficulty level.
The Goldilocks Zone.Let ˆp(S) denote the empirical
probability that the generator answers correctly given ev-
idence set S. We define the Goldilocks Zone as evidence
sets satisfying ˆp(S)≈c , where c∈(0,1) is a target correct-
ness rate (e.g., c= 0.5 ). Intuitively, such evidence sets are:
(1)Solvable: The generator can produce correct answers
with non-negligible probability, indicating that sufficient
information is present.
(2)Non-trivial: The generator does not succeed determin-
istically, suggesting that reasoning is required rather than
simple pattern matching.
By targeting this zone, we encourage the selector to find
evidence combinations that demand genuine multi-step rea-
soning while remaining within the generator’s capability.
Selector sampling.For each query q, we sample Mcan-
didate evidence sets from the selector:
S(m)∼πr(· |q, C), m= 1, . . . , M.(1)
3

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Each evidence set S(m)is then evaluated using the frozen
generatorπ g.
Generator rollouts.For each evidence set S, we sample
Kanswers from the generator:
a(k)∼πg(· |q, S), k= 1, . . . , K.(2)
Each answer must follow a prescribed format (e.g.,
<answer>...</answer> ). We define the rollout cor-
rectness indicator based on the generator’s final reward
Rg(·)(defined in Section 2.4). Specifically, a rollout is
considered correct if its reward exceeds a thresholdδ:
z(k)=Ih
Rg
a(k)
≥δi
.(3)
We then estimate the empirical correctness probability as:
ˆp(S) =1
KKX
k=1z(k).(4)
Reward design.We design four reward components that
together guide the selector toward the Goldilocks Zone
while maintaining relevance and output quality.
(1) Boundary reward Rbdy(S).This reward encourages
evidence sets near the target correctness c. We adopt a
triangular function that peaks at ˆp(S) =c and decreases
linearly as the evidence becomes too easy or too hard:
Rbdy(S) = minˆp(S)
c,1−ˆp(S)
1−c
(5)
To estimate ˆp(S), we use the rollout-based definition in the
Generator rollouts paragraph with threshold δ. A rollout
is deemed correct if its generator reward Rg(a)(defined in
Section 2.4) exceeds a thresholdδ:
ˆp(S) =1
KKX
k=1Ih
Rg(a(k))≥δi
.(6)
(2) Relevance reward Rrel(S).We define the relevance
reward as the average retrieval score (in the initial retrieval
phrase) of the selected documents,
Rrel(S) =1
|S|X
ci∈Sscore(q, c i),(7)
and rescale it to(0,1)for stability.
(3) Format reward Rfmt(S).We define a binary format
indicator that checks whether the selector output is well-
formed (e.g., valid document indices, no duplicates, and
proper structure):
Rfmt(S) =I
Sis well-formed
.(8)This design ensures that malformed selector outputs receive
zero reward, while relative trade-offs among boundary tar-
geting, relevance, and document count are only considered
for valid evidence sets.
(4) Count penalty Pcnt(S).To encourage the selector to
output a target number of documents k∗, we apply a penalty
proportional to the deviation:
Pcnt(S) = min (α· ||S| −k∗|, P max),(9)
where αis the penalty per document deviation and Pmax
caps the maximum penalty.
Final reward.We combine the reward components using
the format indicator as a gate:
Rr(S) =R fmt(S)·
λbdyRbdy(S)+λ relRrel(S)−P cnt(S)
,
(10)
where λbdy, λrel≥0 control the relative importance of
targeting the competence boundary versus maintaining rele-
vance.
Optimization.We optimize the selector using Group Rel-
ative Policy Optimization (GRPO) (Shao et al., 2024). For
each group of sampled evidence sets{S(m)}M
m=1, we com-
pute group-normalized advantages:
A(m)=Rr(S(m))−µ
σ+ϵ,(11)
where µandσare the mean and standard deviation of re-
wards within the group. The selector is trained to minimize:
Lr(θr) =−E mh
min
r(m)A(m),
clip(r(m),1−ϵ,1 +ϵ)A(m)i
,(12)
where r(m)=πr(S(m)|q, C)/πold
r(S(m)|q, C) is the
likelihood ratio.
2.4. Stage 2: Generator Fine-tuning
Standard RAG pipelines train generators on curated, near-
perfect evidence but deploy them with noisy retrieval, lead-
ing to a mismatch and performance degradation. By fine-
tuning the generator under the selector’s output distribution,
we expose it to realistic, challenging evidence during train-
ing.
Training procedure.For each query q, the frozen selector
produces an evidence set S=π r(q, C) . The generator
samplesKcandidate answers:
a(k)∼πg(· |q, S), k= 1, . . . , K.(13)
4

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 1. BAR-RAG consistently improves performance across models and QA benchmarks.Exact match (EM) results on general QA
and multi-hop QA tasks for three backbone models. All values are reported as absolute scores. For IRCoT, RAG w/ Reranker, RAG SFT,
andBAR-RAG , cell shading indicates improvement ( blue ) or degradation ( red) relative to the RAG baseline. BAR-RAG yields strong
and consistent gains across both single-hop and multi-hop settings, surpassing standard RAG pipelines and reranking-based baselines.
General QA Multi-Hop QA
Method NQ TriviaQA PopQA HotpotQA 2Wiki. MuSiQue Bamboogle Avg.
Qwen-2.5-3B-Instruct
Direct Inference 10.6 28.8 10.8 14.9 24.4 2.0 2.4 13.4
CoT 2.3 3.2 0.5 2.1 2.1 0.2 0.0 1.5
RAG 34.8 54.4 38.7 25.5 22.6 4.7 8.0 27.0
IRCoT 11.1 31.2 20.0 16.4 17.1 6.7 24.0 18.1
RAG w/Reranker 35.6 55.3 39.6 26.7 23.4 4.9 10.4 28.0
RAG SFT 38.9 58.1 41.4 28.9 25.9 6.4 13.5 30.4
BAR-RAG(1 Iter) 40.1 58.4 41.2 31.0 24.9 6.3 24.0 32.3
BAR-RAG(2 Iter) 41.5 61.3 43.4 33.2 26.4 7.4 26.3 34.3
BAR-RAG(3 Iter) 42.0 59.7 44.1 32.6 26.7 7.2 26.8 34.2
Qwen-2.5-7B-Instruct
Direct Inference 13.4 40.8 14.0 18.3 12.6 3.1 12.0 16.3
CoT 4.8 18.5 5.4 9.2 10.8 2.2 23.2 10.6
RAG 39.3 53.7 26.7 28.9 18.9 4.7 16.0 26.9
IRCoT 22.4 47.8 30.1 13.3 14.9 7.2 22.4 22.6
RAG w/Reranker 40.5 55.3 27.3 28.1 20.4 5.5 18.7 28.0
RAG SFT 42.7 58.6 32.3 32.4 22.6 6.8 27.1 31.8
BAR-RAG(1 Iter) 44.7 63.2 44.1 37.2 28.3 8.3 36.7 37.2
BAR-RAG(2 Iter) 46.1 64.3 46.3 38.1 29.9 9.1 39.1 38.8
BAR-RAG(3 Iter) 46.9 64.5 46.9 38.8 29.8 9.1 39.6 39.1
LLaMA-3.1-8B-Instruct
Direct Inference 18.4 36.5 19.8 12.5 23.0 2.7 8.8 17.4
CoT 27.8 54.1 23.5 24.1 23.0 6.8 16.3 25.1
RAG 42.7 58.2 28.9 30.3 19.4 6.3 17.6 29.1
IRCoT 23.5 48.1 31.2 12.2 11.8 6.9 24.5 22.6
RAG w/Reranker 43.6 60.1 30.4 32.5 23.6 7.6 19.5 31.0
RAG SFT 45.8 63.4 35.6 35.9 27.4 8.9 24.3 34.5
BAR-RAG(1 Iter) 47.5 65.9 45.8 39.5 31.4 11.1 30.4 38.8
BAR-RAG(2 Iter) 49.0 67.7 48.0 40.5 33.0 12.5 32.8 40.5
BAR-RAG(3 Iter) 49.5 67.3 48.6 41.2 33.0 12.0 33.3 40.7
Generator Reward Design.We design a composite re-
ward that encourages both answer accuracy and proper evi-
dence attribution.
(1) Format reward.The generator must produce outputs
in the prescribed format (i.e., valid <answer> tags). If the
format check fails, the reward is set to zero:
Rg(a) = 0if format is invalid. (14)
(2) Accuracy reward Racc(a).For well-formed outputs,
we compute a weighted combination of token-level F1 score
and exact match (EM) against the gold answers:
Racc(a) =β 1·max
g∈GF1(a, g) +β 2·max
g∈GEM(a, g),(15)where Gis the set of gold answers, and β1, β2≥0control
the relative importance of partial credit (F1) versus exact
correctness (EM).
(3) Citation reward Rcite(a).To encourage the generator
to ground its reasoning in the provided evidence, we reward
appropriate citation behavior in the <think> block. Let
ncitedenote the number of unique documents cited. We use
a peaked reward centered at a target citation countn∗:
Rcite(a) =

1.0,ifn cite=n∗,
0.5,if|n cite−n∗|= 1,
0.0,otherwise.(16)
This encourages the generator to cite a moderate number of
sources from the provided documents, sufficient to support
5

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 2.Ablation study on different components.
Method NQ PopQA HotpotQA Avg.
Full 46.9 46.9 38.8 44.2
w/o Filtering 42.6 32.1 32.3 35.6
w/o Stage1 42.1 42.5 41.9 42.2
w/o Stage2 39.7 34.1 36.7 36.8
Selector
w/oR bdy 43.4 43.6 33.5 40.2
w/oR rel 46.1 46.3 38.1 43.5
Generator
w/oR cite 45.3 44.8 37.9 42.7
multi-hop reasoning but not so many as to dilute focus.
Final reward.We combine the three components with
independent weights for accuracy and citation, while fixing
the format coefficient:
Rg(a) =R fmt(a)· 
λaccRacc(a) +λ citeRcite(a)
,(17)
whereλ acc, λcite≥0are hyperparameters.
Optimization.The generator is optimized using GRPO
with the same clipped objective:
Lg(θg) =−E kh
min
r(k)A(k),
clip(r(k),1−ϵ,1 +ϵ)A(k)i
,(18)
where advantages A(k)are computed from generator
rewards within each group, and r(k)=π g(a(k)|
q, S)/πold
g(a(k)|q, S).
Iterative two-stage training.Rather than a single pass,
we alternate between the two stages for Titerations. At
iteration t, we (i) train the selector π(t)
ragainst a frozen gen-
erator π(t−1)
g to target the current competence boundary, and
then (ii) freeze the updated selector and fine-tune the gener-
ator to obtain π(t)
gunder the induced evidence distribution.
In practice, each iteration consists of one epoch of selector
training followed by one epoch of generator fine-tuning.
This alternating procedure progressively refines evidence
selection to track the generator’s evolving competence. Al-
gorithmic details are provided in Appendix 1.
2.5. Inference
The selector is used only during training to shape a chal-
lenging evidence distribution for the generator. At inference
time, we discard the selector entirely and apply the fine-
tuned generator directly to the top- kdocuments returned by
a standard retriever (e.g., BM25 or a dense retriever), pro-
ducing the final answer a=π g(q, S) . This design incurs noadditional inference cost and preserves the standard RAG
pipeline. By being trained iteratively on adversarially se-
lected evidence near its competence boundary, the generator
becomes more robust to noisy, incomplete, and imperfect
retrieval results encountered at test time.
3. Experiments
3.1. Datasets and Evaluation Metrics
We evaluate on seven knowledge-intensive QA datasets
spanning diverse reasoning challenges. Forsingle-hop QA,
we use Natural Questions (NQ) (Kwiatkowski et al., 2019),
TriviaQA (Joshi et al., 2017), and PopQA (Mallen et al.,
2023), which test robustness to retrieval noise, paraphrased
evidence, and long-tail entity reasoning, respectively. For
multi-hop QA, we use HotpotQA (2-hop bridge reasoning)
(Yang et al., 2018), 2WikiMultiHopQA (distant supporting
facts) (Ho et al., 2020), MuSiQue (3–5 compositional hops)
(Trivedi et al., 2022), and Bamboogle (indirect reasoning)
(Press et al., 2023). The training datasets are reported at
Appendix A.7. We report Exact Match (EM) as the primary
metric, following standard evaluation protocols.
3.2. Baselines
We compare against two categories of methods.Without
retrieval: (1) Direct Inference, prompting the base model
directly; (2) Chain-of-Thought (CoT) (Wei et al., 2022),
eliciting step-by-step reasoning; (3) RAG SFT, supervised
fine-tuning on QA pairs with retrieved evidence.With re-
trieval: (1) RAG (Shi et al., 2024), standard dense retrieval-
augmented generation; (2) RAG w/ Reranker, adding a neu-
ral reranker to re-score retrieved documents before gener-
ation; (3) IRCoT (Trivedi et al., 2023), a multi-step QA
framework that interleaves retrieval with steps in a CoT;
(4) RAG SFT, fine-tuning on QA pairs augmented with
top-5 retrieved passages, exposing the model to both rele-
vant and noisy evidence during training. All baselines use
the same base model (Qwen2.5-3B-Instruct,Qwen2.5-7B-
Instruct(Qwen et al., 2025) andLLaMA-3.1-8B-Instruct
(Grattafiori et al., 2024)), retriever (E5-base-v2(Wang et al.,
2022)) and reranker (Qwen-3-Embedding-8B(Zhang et al.,
2025)) and Top-5 retrieved documents as input for fair com-
parison.
3.3. Implementation Details
We use instruction-tuned LLMs as both the selector and
generator backbones, including Qwen2.5 and LLaMA-3.1
variants. Detailed model configurations, reward parame-
ters, rollout settings, and training schedules are provided in
Appendix A.5.
6

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
135 10 15 20 30404550
Qwen-2.5-7B-Instruct
Ours
RAG
135 10 15 20 3050556065
Qwen-2.5-7B-Instruct
Ours
RAG
135 10 15 20 30303540
Qwen-2.5-7B-Instruct
Ours
RAG
135 10 15 20 3025303540
Qwen-2.5-7B-Instruct
Ours
RAG
135 10 15 20 3035404550
LLaMA-3.1-8B-Instruct
Ours
RAG
135 10 15 20 3050556065
LLaMA-3.1-8B-Instruct
Ours
RAG
135 10 15 20 302530354045
LLaMA-3.1-8B-Instruct
Ours
RAG
135 10 15 20 3025303540
LLaMA-3.1-8B-Instruct
Ours
RAG
Figure 4.Top- Kaccuracy curves on four QA benchmarks for two base models:Qwen-2.5-7B-InstructandLLaMA-3.1-8B-Instruct.
From left to right, columns correspond toNQ,TriviaQA,PopQA, andHotpotQA. Across both models and all datasets, our method
consistently achieves higher accuracy in low- Kregimes and remains robust as Kincreases, whereas standard RAG exhibits weaker
scaling behavior and higher sensitivity to retrieval noise.
3.4. Main Results
Table 1 presents our main results across seven QA bench-
marks. BAR-RAG consistently outperforms all baselines on
both general and multi-hop QA tasks. UsingQwen2.5-7B-
Instructas a representative example, BAR-RAG achieves
substantial improvements over the strongest baseline RAG
SFT on single-hop benchmarks: +8.5 on NQ (51.2 vs. 42.7),
+5.9 on TriviaQA, and +14.6 on PopQA. The advantages
are more pronounced on multi-hop tasks, with gains of +6.4
on HotpotQA, +7.2 on 2WikiMultiHopQA, and +12.5 on
Bamboogle—the latter highlighting the benefit of training
on evidence that demands genuine multi-step reasoning. Re-
sults onLLaMA-3.1-8B-InstructandQwen2.5-3B-Instruct
confirm that our approach generalizes across model fami-
lies, achieving average EM of 40.7 and 34.2 respectively
(vs. 29.1 and 27.0 for RAG).
Iterative ImprovementWe examine how performance
evolves across training iterations. As shown in Table 1,
all three models exhibit consistent improvement from Iter
1 to Iter 2, with diminishing gains from Iter 2 to Iter 3.
This pattern suggests that the iterative co-training procedure
converges toward a stable solution within a few iterations,
and that the majority of gains are captured in the first two
rounds of selector-generator co-adaptation.
3.5. Ablation Studies
We conduct ablation studies to validate the necessity of key
design choices, including (i) the design of reward compo-
nents for boundary-aware evidence selection and (ii) the
two-stage training pipeline.Effect of Reward Components.To examine the contri-
bution of individual reward terms in the selector, we ablate
two key components: the boundary reward Rbdyand the
relevance reward Rrel, while keeping other training settings
unchanged. Specifically, we evaluate variants that remove
RbdyorRrelfrom the selector training objective. Results
are reported in Table 2. Removing the boundary reward
leads to a clear performance degradation across all datasets,
confirming that targeting evidence near the generator’s com-
petence boundary is crucial for robust reasoning. In contrast,
removing the relevance reward results in a milder but consis-
tent drop, indicating its complementary role in maintaining
evidence quality. We further ablate the citation reward Rcite
used in generator fine-tuning. Removing Rciteresults in
a consistent but smaller performance drop, indicating that
citation supervision complements competence-aware evi-
dence selection by improving answer grounding rather than
driving the main gains.
Effect of Two-Stage Training.Our pipeline consists of
two stages: selector training (Stage 1) followed by gener-
ator fine-tuning (Stage 2). To assess the necessity of each
component, we conduct ablation studies summarized in Ta-
ble 2. When Stage 2 is removed, the selector is trained while
keeping the generator frozen at its initial state. This variant
leads to a clear performance drop across all benchmarks,
indicating that adapting the generator to the competence-
boundary evidence distribution is crucial for effective rea-
soning. We further examine the role of the training-time
filtering step by removing it and training the selector on
the full dataset. Eliminating filtering consistently degrades
performance, with particularly large drops on PopQA and
HotpotQA, confirming that trivially solvable and unanswer-
7

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
0.0 0.2 0.4 0.6 0.8 1.0
p (empirical solvability)
0.00.51.01.52.02.53.03.5DensityEvidence Difficulty Distribution Shift (Qwen)
RAG
RAG+Reranker
Ours
0.0 0.2 0.4 0.6 0.8 1.0
p (empirical solvability)
0.00.51.01.52.02.53.0DensityEvidence Difficulty Distribution Shift (LLaMA)
RAG
RAG w/ Reranker
Ours
Figure 5.Distribution shift of evidence difficulty ( ˆp) across different retrieval methods. Our method concentrates samples toward the
target correctness levelc=0.5, substantially reducing extreme cases near 0 (unsolvable) and 1 (trivially solvable).
able queries introduce degenerate RL signals that hinder
stable selector optimization. Finally, removing Stage 1
eliminates competence-aware evidence selection altogether,
reducing the pipeline to standard retrieval followed by gen-
erator training. The resulting degradation demonstrates that
naive retrieval is insufficient for robust multi-hop reasoning.
3.6. Analysis
Generator Robustness to Evidence Quality.A central
claim of our approach is that training on competence-
boundary evidence yields a generator that is inherently more
robust to evidence quality—even when the selector is dis-
carded at inference time. To verify this, we evaluate gen-
erators trained with BAR-RAG , and standard RAG, using
the same naive retriever outputs (top- kby retrieval score)
under varying evidence budgets k∈ {1,3,5,10,15,30} .
We conduct this evaluation across two backbone models
(Qwen-2.5-7B-InstructandLLaMA-3.1-8B-Instruct) and
four datasets (NQ, TriviaQA, PopQA, and HotpotQA). Fig-
ure 4 reports K-accuracy curves for all settings. Consistent
with our hypothesis, BAR-RAG -trained generators consis-
tently outperform baselines across all k, with the largest
gains appearing in low- kregimes where evidence is sparse
and reasoning robustness is most critical.
Evidence Difficulty Distribution Shift.For each ques-
tion, we estimate the empirical solvability ˆp(S) of an ev-
idence set SviaKgenerator rollouts, and compare the
resulting distributions across three methods: Naive RAG
(selecting top- kdocuments by retrieval score), RAG with
a neural reranker, and our BAR-RAG selector. Figure 5
compares the resulting distributions across Naive RAG,
RAG with a neural reranker, and our BAR-RAG selector
for bothQwen-2.5-7B-InstructandLLaMA-3.1-8B-Instruct.
While Naive RAG and reranker-based RAG exhibit bimodal
distributions dominated by unsolvable or trivially answer-
revealing contexts, BAR-RAG suppresses both extremes andconcentrates probability mass around the target correctness
level c= 0.5 . This consistent shift across generators indi-
cates that BAR-RAG actively promotes hard-but-solvable
evidence near the generator’s competence boundary rather
than merely improving relevance.
Additional analyses, including counterfactual evidence-
dependence studies and comparisons with agentic RAG
methods, are provided in Appendix A.3 and A.2.
4. Related Work
Retrieval-Augmented GenerationRetrieval-Augmented
Generation (RAG) improves factual accuracy by ground-
ing language model outputs in external knowledge sources
(Lewis et al., 2020; Guu et al., 2020). Recent work explores
tighter retrieval–generation integration, including mecha-
nisms that adapt retrieval behavior based on model uncer-
tainty or self-reflection, such as Self-RAG (Asai et al., 2024).
However, BAR-RAG targets what evidence to retrieve, fram-
ing evidence selection as a competence-aware process that
deliberately selects hard-but-solvable contexts to strengthen
the generator’s reasoning ability.
Document RerankingReranking refines retrieved candi-
dates before downstream generation, with early neural ap-
proaches relying on cross-encoder architectures to compute
relevance scores (Nogueira & Cho, 2019). More recently,
reinforcement learning has been used to optimize reranking
policies via generator feedback, as in DynamicRAG (Sun
et al., 2025). In contrast to prior RL-based rerankers that
optimize relevance or answer quality, BAR-RAG introduces
an explicit competence boundary reward that directly mod-
els evidence difficulty, enabling the selection of challenging
yet solvable evidence sets.
Further extand related works are reported in Appendix A.4.
8

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
5. Conclusion
We presented BAR-RAG , a boundary-aware evidence se-
lection training framework that trains the selector to target
hard-but-solvable evidence near the generator’s competence
boundary. We train generator with realistic retrieval con-
ditions, yielding consistent robustness gains across diverse
QA benchmarks without additional inference-time cost.
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
rag: Learning to retrieve, generate, and critique through
self-reflection. InThe Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna, Austria,
May 7-11, 2024. OpenReview.net, 2024. URL https:
//openreview.net/forum?id=hSyW5go0v8.
Cao, W., Wang, J., Zheng, Y ., Bao, L., Zheng, Q., Berg-
Kirkpatrick, T., Paturi, R., and Bergen, L. Single-pass
document scanning for question answering, 2025. URL
https://arxiv.org/abs/2504.03101.
Cao, Z., Qin, T., Liu, T., Tsai, M., and Li, H. Learn-
ing to rank: from pairwise approach to listwise ap-
proach. In Ghahramani, Z. (ed.),Machine Learning,
Proceedings of the Twenty-Fourth International Con-
ference (ICML 2007), Corvallis, Oregon, USA, June
20-24, 2007, volume 227 ofACM International Con-
ference Proceeding Series, pp. 129–136. ACM, 2007.
doi: 10.1145/1273496.1273513. URL https://doi.
org/10.1145/1273496.1273513.
Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Cam-
pagnano, C., Maarek, Y ., Tonellotto, N., and Silvestri,
F. The power of noise: Redefining retrieval for RAG
systems. In Yang, G. H., Wang, H., Han, S., Hauff, C.,
Zuccon, G., and Zhang, Y . (eds.),Proceedings of the 47th
International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR 2024, Wash-
ington DC, USA, July 14-18, 2024, pp. 719–729. ACM,
2024. doi: 10.1145/3626772.3657834. URL https:
//doi.org/10.1145/3626772.3657834.
Drozdov, A., Zhuang, H., Dai, Z., Qin, Z., Rahimi, R., Wang,
X., Alon, D., Iyyer, M., McCallum, A., Metzler, D., and
Hui, K. Parade: Passage ranking using demonstrations
with large language models, 2023. URL https://
arxiv.org/abs/2310.14408.
Grattafiori, A., Dubey, A., Jauhri, A., et al. The llama 3
herd of models, 2024. URL https://arxiv.org/
abs/2407.21783.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.-W.
Realm: retrieval-augmented language model pre-training.
InInternational Conference on Machine Learning, 2020.Ho, X., Duong Nguyen, A.-K., Sugawara, S., and Aizawa,
A. Constructing a multi-hop QA dataset for compre-
hensive evaluation of reasoning steps. In Scott, D.,
Bel, N., and Zong, C. (eds.),Proceedings of the 28th
International Conference on Computational Linguis-
tics, pp. 6609–6625, Barcelona, Spain (Online), De-
cember 2020. International Committee on Computa-
tional Linguistics. doi: 10.18653/v1/2020.coling-main.
580. URL https://aclanthology.org/2020.
coling-main.580/.
Hsia, J., Shaikh, A., Wang, Z., and Neubig, G. Ragged:
Towards informed design of scalable and stable rag
systems, 2025. URL https://arxiv.org/abs/
2403.09040.
Jiang, P., Lin, J., Shi, Z., Wang, Z., He, L., Wu, Y ., Zhong,
M., Song, P., Zhang, Q., Wang, H., et al. Adaptation of
agentic ai.arXiv preprint arXiv:2512.16301, 2025.
Jin, B., Zeng, H., Yue, Z., Wang, D., Zamani, H.,
and Han, J. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
CoRR, abs/2503.09516, 2025. doi: 10.48550/ARXIV .
2503.09516. URLhttps://doi.org/10.48550/
arXiv.2503.09516.
Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. Trivi-
aQA: A large scale distantly supervised challenge dataset
for reading comprehension. In Barzilay, R. and Kan,
M.-Y . (eds.),Proceedings of the 55th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 1601–1611, Vancouver,
Canada, July 2017. Association for Computational Lin-
guistics. doi: 10.18653/v1/P17-1147. URL https:
//aclanthology.org/P17-1147/.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P., Wu, L.,
Edunov, S., Chen, D., and Yih, W. Dense passage re-
trieval for open-domain question answering. In Webber,
B., Cohn, T., He, Y ., and Liu, Y . (eds.),Proceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2020, Online, Novem-
ber 16-20, 2020, pp. 6769–6781. Association for Com-
putational Linguistics, 2020. doi: 10.18653/V1/2020.
EMNLP-MAIN.550. URL https://doi.org/10.
18653/v1/2020.emnlp-main.550.
Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., and
Lewis, M. Generalization through memorization: Nearest
neighbor language models. InInternational Conference
on Learning Representations (ICLR), 2020.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., De-
vlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M.,
Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and
9

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Petrov, S. Natural questions: A benchmark for question
answering research.Transactions of the Association for
Computational Linguistics, 7:452–466, 2019. doi: 10.
1162/tacl a00276. URL https://aclanthology.
org/Q19-1026/.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., Riedel, S., and Kiela, D. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks. InAdvances in
Neural Information Processing Systems, volume 33, pp.
9459–9474, 2020.
Li, X., Dong, G., Jin, J., Zhang, Y ., Zhou, Y ., Zhu, Y ., Zhang,
P., and Dou, Z. Search-o1: Agentic search-enhanced large
reasoning models.CoRR, abs/2501.05366, 2025a. doi:
10.48550/ARXIV .2501.05366. URL https://doi.
org/10.48550/arXiv.2501.05366.
Li, Y ., Zhang, W., Yang, Y ., Huang, W., Wu, Y ., Luo, J.,
Bei, Y ., Zou, H. P., Luo, X., Zhao, Y ., Chan, C., Chen, Y .,
Deng, Z., Li, Y ., Zheng, H., Li, D., Jiang, R., Zhang, M.,
Song, Y ., and Yu, P. S. Towards agentic RAG with deep
reasoning: A survey of rag-reasoning systems in llms.
CoRR, abs/2507.09477, 2025b. doi: 10.48550/ARXIV .
2507.09477. URLhttps://doi.org/10.48550/
arXiv.2507.09477.
Ma, X., Zhang, X., Pradeep, R., and Lin, J. Zero-
shot listwise document reranking with a large language
model, 2023. URL https://arxiv.org/abs/
2305.02156.
Mallen, A., Asai, A., Zhong, V ., Das, R., Khashabi, D.,
and Hajishirzi, H. When not to trust language mod-
els: Investigating effectiveness of parametric and non-
parametric memories. In Rogers, A., Boyd-Graber, J. L.,
and Okazaki, N. (eds.),Proceedings of the 61st An-
nual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), ACL 2023, Toronto,
Canada, July 9-14, 2023, pp. 9802–9822. Association
for Computational Linguistics, 2023. doi: 10.18653/
V1/2023.ACL-LONG.546. URL https://doi.org/
10.18653/v1/2023.acl-long.546.
Nogueira, R. and Cho, K. Passage re-ranking with BERT.
CoRR, abs/1901.04085, 2019. URL http://arxiv.
org/abs/1901.04085.
Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A.,
and Lewis, M. Measuring and narrowing the composition-
ality gap in language models. In Bouamor, H., Pino, J.,
and Bali, K. (eds.),Findings of the Association for Com-
putational Linguistics: EMNLP 2023, Singapore, Decem-
ber 6-10, 2023, pp. 5687–5711. Association for Com-
putational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.378. URL https://doi.org/
10.18653/v1/2023.findings-emnlp.378.
Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan,
L., Shen, J., Liu, T., Liu, J., Metzler, D., Wang, X., and
Bendersky, M. Large language models are effective text
rankers with pairwise ranking prompting. In Duh, K.,
Gomez, H., and Bethard, S. (eds.),Findings of the Asso-
ciation for Computational Linguistics: NAACL 2024, pp.
1504–1518, Mexico City, Mexico, June 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.
findings-naacl.97. URL https://aclanthology.
org/2024.findings-naacl.97/.
Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng,
B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H.,
Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J.,
Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L.,
Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R.,
Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y ., Su,
Y ., Zhang, Y ., Wan, Y ., Liu, Y ., Cui, Z., Zhang, Z., and
Qiu, Z. Qwen2.5 technical report, 2025. URL https:
//arxiv.org/abs/2412.15115.
Ram, O., Levine, Y ., Dalmedigos, I., Muhlgay, D., Shashua,
A., Leyton-Brown, K., and Shoham, Y . In-context
retrieval-augmented language models.Transactions of
the Association for Computational Linguistics, pp. 1316–
1331, 2023.
Robertson, S. E. and Zaragoza, H. The probabilistic
relevance framework: BM25 and beyond.Found.
Trends Inf. Retr., 3(4):333–389, 2009. doi: 10.1561/
1500000019. URL https://doi.org/10.1561/
1500000019.
Sachan, D. S., Lewis, M., Joshi, M., Aghajanyan, A.,
Yih, W., Pineau, J., and Zettlemoyer, L. Improv-
ing passage retrieval with zero-shot question genera-
tion. In Goldberg, Y ., Kozareva, Z., and Zhang, Y .
(eds.),Proceedings of the 2022 Conference on Empir-
ical Methods in Natural Language Processing, EMNLP
2022, Abu Dhabi, United Arab Emirates, December
7-11, 2022, pp. 3781–3797. Association for Compu-
tational Linguistics, 2022. doi: 10.18653/V1/2022.
EMNLP-MAIN.249. URL https://doi.org/10.
18653/v1/2022.emnlp-main.249.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M.,
Li, Y . K., Wu, Y ., and Guo, D. Deepseekmath: Pushing
the limits of mathematical reasoning in open language
models.CoRR, abs/2402.03300, 2024. doi: 10.48550/
ARXIV .2402.03300. URL https://doi.org/10.
48550/arXiv.2402.03300.
Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis,
M., Zettlemoyer, L., and Yih, W.-t. REPLUG: Retrieval-
10

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
augmented black-box language models. In Duh, K.,
Gomez, H., and Bethard, S. (eds.),Proceedings of the
2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), pp. 8371–
8384, Mexico City, Mexico, June 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.
naacl-long.463. URL https://aclanthology.
org/2024.naacl-long.463/.
Sun, J., Zhong, X., Zhou, S., and Han, J. Dynamicrag:
Leveraging outputs of large language model as feedback
for dynamic reranking in retrieval-augmented generation.
CoRR, abs/2505.07233, 2025. doi: 10.48550/ARXIV .
2505.07233. URLhttps://doi.org/10.48550/
arXiv.2505.07233.
Sun, W., Yan, L., Ma, X., et al. Is ChatGPT good at
search? investigating large language models as re-ranking
agents. In Bouamor, H., Pino, J., and Bali, K. (eds.),
Proceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 14918–14937,
Singapore, December 2023. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2023.emnlp-main.
923. URL https://aclanthology.org/2023.
emnlp-main.923/.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabhar-
wal, A. ♪MuSiQue: Multihop questions via single-
hop question composition.Transactions of the As-
sociation for Computational Linguistics, 10:539–554,
2022. doi: 10.1162/tacl a00475. URL https://
aclanthology.org/2022.tacl-1.31/.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabhar-
wal, A. Interleaving retrieval with chain-of-thought rea-
soning for knowledge-intensive multi-step questions. In
Rogers, A., Boyd-Graber, J. L., and Okazaki, N. (eds.),
Proceedings of the 61st Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long
Papers), ACL 2023, Toronto, Canada, July 9-14, 2023,
pp. 10014–10037. Association for Computational Lin-
guistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.
557. URL https://doi.org/10.18653/v1/
2023.acl-long.557.
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L.,
Jiang, D., Majumder, R., and Wei, F. Text embed-
dings by weakly-supervised contrastive pre-training.
CoRR, abs/2212.03533, 2022. doi: 10.48550/ARXIV .
2212.03533. URLhttps://doi.org/10.48550/
arXiv.2212.03533.
Wang, S., Zhang, L., Fu, Z., Mao, Z., and Zhang, Y . Dacl-
rag: Data augmentation strategy with curriculum learning
for retrieval-augmented generation, 2025. URL https:
//arxiv.org/abs/2505.10493.Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B.,
Xia, F., Chi, E. H., Le, Q. V ., and Zhou, D. Chain-of-
thought prompting elicits reasoning in large language
models. In Koyejo, S., Mohamed, S., Agarwal, A.,
Belgrave, D., Cho, K., and Oh, A. (eds.),Advances in
Neural Information Processing Systems 35: Annual
Conference on Neural Information Processing Systems
2022, NeurIPS 2022, New Orleans, LA, USA, November
28 - December 9, 2022, 2022. URL http://papers.
nips.cc/paper_files/paper/2022/hash/
9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.
html.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W.,
Salakhutdinov, R., and Manning, C. D. Hotpotqa: A
dataset for diverse, explainable multi-hop question an-
swering. In Riloff, E., Chiang, D., Hockenmaier, J.,
and Tsujii, J. (eds.),Proceedings of the 2018 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, Brussels, Belgium, October 31 - November 4,
2018, pp. 2369–2380. Association for Computational
Linguistics, 2018. doi: 10.18653/V1/D18-1259. URL
https://doi.org/10.18653/v1/d18-1259.
Yu, Y ., Ping, W., Liu, Z., Wang, B., You, J., Zhang, C.,
Shoeybi, M., and Catanzaro, B. Rankrag: Unifying
context ranking with retrieval-augmented generation in
llms. In Globersons, A., Mackey, L., Belgrave, D., Fan,
A., Paquet, U., Tomczak, J. M., and Zhang, C. (eds.),
Advances in Neural Information Processing Systems 38:
Annual Conference on Neural Information Processing
Systems 2024, NeurIPS 2024, Vancouver, BC, Canada,
December 10 - 15, 2024, 2024. URL http://papers.
nips.cc/paper_files/paper/2024/hash/
db93ccb6cf392f352570dd5af0a223d3-Abstract-Conference.
html.
Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang,
B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F.,
and Zhou, J. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
CoRR, abs/2506.05176, 2025. doi: 10.48550/ARXIV .
2506.05176. URLhttps://doi.org/10.48550/
arXiv.2506.05176.
11

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
A. Appendix
A.1. Algorithm
We present our algorithm ofBAR-RAGin Algorithm 1.
Algorithm 1Boundary-aware Evidence Selection Training (BAR-RAG)
Require:Training datasetD, retrieverR, selectorπ r, generatorπ g
Require:Target correctnessc, reward thresholdδ, rolloutsK, evidence samplesM
Require:Training iterationsT
1:// Training-time filtering
2:D filt←FILTER(D)
3:fort= 1toTdo
4:// Stage 1: Selector Training (generator frozen)
5:foreach queryqin batch fromD filtdo
6:C← R(q)
7:SampleMevidence sets{S(m)}M
m=1 fromπ r
8:foreachS(m)do
9:SampleKanswers fromπ g(· |q, S(m))
10:Estimateˆp(S(m))and compute selector rewardR r(S(m))
11:end for
12:Update selectorπ rvia GRPO
13:end for
14:// Stage 2: Generator Training (selector frozen)
15:foreach queryqin batch fromDdo
16:C← R(q)
17:S←π r(q, C)
18:SampleKanswers fromπ g(· |q, S)
19:Update generatorπ gvia GRPO
20:end for
21:end for
A.2. Comparison with Recent Reasoning and Agentic RAG Methods
To further contextualize BAR-RAG within the rapidly evolving landscape of reasoning-centric and agentic RAG systems,
we compare against several recent representative methods, including Search-o1 (Li et al., 2025a), Search-R1 (Jin et al.,
2025), and DynamicRAG (Sun et al., 2025).1These approaches emphasize explicit multi-step search, iterative retrieval, or
reinforcement-learning-based reranking at inference time, and are often evaluated under substantially different retrieval and
search budgets.
In particular, we group results by backbone scale following the structure of Table 3, and report the best-performing iteration
ofBAR-RAG for each backbone. Importantly, BAR-RAG incurs no additional inference-time overhead beyond standard
RAG, whereas the compared reasoning and agentic methods typically rely on multi-round search or tool invocation during
inference.
A.3. Counterfactual Evidence Dependence Analysis
A potential concern with generator-aware evidence selection is whether performance gains arise from genuine evidence use
or from self-confirmation effects induced by generator feedback. To directly test whether the model’s predictions causally
depend on the evidence it cites, we perform a counterfactual evidence-dependence analysis.
For each test example, we first run the model under the standard setting (FULL) using the top- kretrieved documents, and
1As these methods vary widely in backbone models, inference-time computation, and retrieval interfaces, a strictly controlled apples-
to-apples comparison is not always possible. Therefore, this table is intended to provide a qualitative and contextual comparison rather
than a claim of direct superiority.
12

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 3.Comparison with recent reasoning and agentic RAG methods (EM, %). Results are grouped by backbone following Table 1, and
BAR-RAG reports the best iteration for each backbone. Search-o1 and Search-R1 results are taken directly from the original papers.
DynamicRAG results are obtained by re-running the authors’ released checkpoint under our evaluation pipeline, as its original setting is
not fully aligned with ours. Notably, Search-R1 employs substantially different retrieval configurations on multi-hop QA (HotpotQA,
2WikiMultiHopQA, MuSiQue and Bamboogle), involving multiple rounds of search with multiple documents per round, whereas our
setting uses a single-shot top- kretrieval (top-5) without iterative search. As a result, Search-R1 results on these datasets (marked with†)
are not directly comparable toBAR-RAGand should be interpreted with caution.
Method NQ TriviaQA PopQA HotpotQA 2Wiki MuSiQue Bamboogle Avg.
Qwen2.5-3B-Instruct
Search-o1 23.8 47.2 26.2 22.1 21.8 5.4 32.0 25.5
Search-R1 34.1 54.5 37.8 32.4†31.9†10.3†26.4†32.5
BAR-RAG41.5 61.3 43.4 33.226.4 7.4 26.334.3
Qwen2.5-7B-Instruct
Search-o1 15.1 44.3 13.1 18.7 17.6 5.8 29.6 20.6
Search-R1 39.3 61.0 39.7 37.0†41.4†14.6†36.8†38.5
BAR-RAG46.9 64.5 46.9 38.829.8 9.139.6 39.1
LLaMA-3.1-8B-Instruct
DynamicRAG 46.4 57.5 36.7 34.2 – – –
BAR-RAG49.5 67.3 48.6 41.233.0 12.0 33.3 40.7
record the set of documents cited in the model’s generation. We then construct two counterfactual variants while keeping
the question and decoding settings fixed: (i) REMOVE-CITED, where all documents cited by the model in the original
generation are removed from the input, and (ii) KEEP-ONLY-CITED, where only the cited documents are retained and all
other retrieved documents are discarded. The cited document set is always determined from the original FULLgeneration
and kept fixed across counterfactual runs.
Table 4 reports Exact Match (EM) results on NQ, HotpotQA, and Bamboogle using Qwen2.5-7B-Instruct. Across all
datasets, removing the cited evidence causes a large performance drop, with ∆rmexceeding 24 EM on all three benchmarks.
This sharp degradation indicates that correct predictions critically rely on the documents explicitly referenced by the model.
In contrast, retaining only the cited documents largely preserves performance and in some cases slightly improves it (e.g.,
NQ and Bamboogle), suggesting that the cited documents form a compact and sufficient support set, while additional
retrieved documents often act as noise.
A.4. Extended Related Work
A.4.1. RETRIEVAL-AUGMENTEDGENERATION
Retrieval-Augmented Generation (RAG) has emerged as a general framework for grounding language model outputs in
external knowledge sourceAR s, mitigating hallucination and enabling dynamic knowledge updates. Early formulations
combine neural retrieval with sequence-to-sequence generation for knowledge-intensive tasks (Lewis et al., 2020; Guu
et al., 2020; Wang et al., 2025; Cao et al., 2025). Beyond document-level retrieval, several works explore datastore-based
Table 4.Counterfactual evidence-dependence analysis on Qwen2.5-7B-Instruct. REMOVE-CITEDremoves all documents cited by the
model in the original generation, while KEEP-ONLY-CITEDretains only the cited documents. ∆rmdenotes the EM drop from FULLto
REMOVE-CITED.
Dataset Full Remove-Cited Keep-Only-Cited∆ rm
NQ 46.9 20.4 47.3 26.5
HotpotQA 38.8 12.2 38.5 26.6
Bamboogle 39.6 15.3 40.1 24.3
13

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
or token-level augmentation. For example, kNN-LM (Khandelwal et al., 2020) augments next-token prediction with
nearest-neighbor retrieval over a large datastore, while subsequent extensions improve efficiency and contextualization (Ram
et al., 2023).
Recent research has focused on tighter integration between retrieval and generation. Some approaches train retrieval and
generation components end-to-end, while others introduce explicit control mechanisms to decide when retrieval should
occur. Self-RAG (Asai et al., 2024) equips the generator with a learned critic that reflects on intermediate generations and
dynamically triggers retrieval. These methods primarily focus on retrieval timing or integration strategy, whereas our work
complements them by addressing the orthogonal problem of evidence selection quality under fixed retrieval budgets.
A.4.2. DOCUMENTRERANKING
Reranking is a long-standing component in information retrieval pipelines, aiming to refine initial retrieval results before
downstream consumption. Traditional learning-to-rank approaches optimize pairwise or listwise objectives over hand-
crafted features (Cao et al., 2007). With the advent of pretrained language models, neural rerankers based on cross-encoder
architectures have become the dominant paradigm, jointly encoding query–document pairs to compute relevance scores with
fine-grained token interactions (Nogueira & Cho, 2019).
Large language models have recently been explored as rerankers, leveraging their reasoning capabilities to assess document
usefulness. Prior work spans multiple granularities, including pointwise relevance prediction or likelihood estimation
(Drozdov et al., 2023; Sachan et al., 2022), pairwise comparison of candidate documents (Qin et al., 2024), and listwise
ranking that directly outputs document permutations (Sun et al., 2023; Ma et al., 2023). Many of these approaches operate
in a zero-shot or weakly supervised setting and optimize relevance-based objectives.
Several recent works incorporate reinforcement learning to optimize reranking policies. DynamicRAG (Sun et al., 2025)
formulates reranking as a sequential decision process and uses generator feedback as a reward signal to dynamically
determine both the ordering and the number of documents. While effective, such approaches typically rely on relevance
or answer-quality-based rewards. In contrast, our method introduces an explicit competence-aware objective that directly
models evidence difficulty, targeting hard-but-solvable evidence sets that maximize learning signal for downstream generator
training.
A.5. Training Implementation Details
We useQwen2.5-3B-Instruct,Qwen2.5-7B-Instruct, andLLaMA-3.1-8B-Instructas backbone models for both the selector
and the generator. In both training stages, we adopt parameter-efficient fine-tuning using LoRA adapters with rank 32 and
scaling factorα= 16, while keeping all backbone parameters frozen.
For retrieval, we useE5as the dense retriever to obtain a fixed pool of top- n= 25 candidate documents for each query.
During selector training, we sampleM= 8candidate evidence sets per query, each consisting ofk= 5documents.
Each sampled evidence set is evaluated using K= 10 generator rollouts to estimate the correctness probability ˆp(S). A
rollout is deemed correct if its final generator reward exceeds a threshold δ= 0.8 . The selector reward targets a correctness
level of c= 0.5 , with boundary and relevance weights λbdy= 1.0 andλrel= 0.2 , respectively. We use a relevance
temperature τ= 10.0 and apply a count penalty of α= 0.5 per document deviation from the target size k∗= 5, capped at
Pmax= 1.0.
The generator reward combines answer accuracy and citation quality with weights λacc= 0.8 andλcite= 0.2 . The accuracy
reward weights token-level F1 and exact match as β1= 0.7 andβ2= 0.3 , respectively. The citation reward targets n∗= 2
cited documents.
Both stages are optimized using GRPO with a learning rate of 4×10−6, cosine learning rate decay with 2% warmup,
clipping parameter ϵ= 0.2 , batch size 8, and KL regularization coefficient 0.001 . Training is conducted for three iterations
on8×A100 (40GB) GPUs using bfloat16 mixed-precision training with gradient checkpointing.
A.6. Training-time Filtering Details
We now provide implementation details for the filtering procedure described in Section ??. For each training query, we
sample Nreranker rollouts and Kgenerator rollouts per evidence set. Generator outputs are judged correct if their total
reward exceeds a fixed thresholdδ, consistent with the correctness definition used during selector training.
14

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 5.Dataset statistics for training and evaluation.
Split Dataset #Examples Task Type Domain
TrainNQ 79,168 Single-hop In-domain
HotpotQA 8,757 Multi-hop In-domain
Total 87,925– –
EvalNQ 3,610 Single-hop In-domain
HotpotQA 7,405 Multi-hop In-domain
TriviaQA 11,313 Single-hop Out-of-domain
PopQA 14,267 Single-hop Out-of-domain
2WikiMultiHopQA 12,576 Multi-hop Out-of-domain
Musique 2,417 Multi-hop Out-of-domain
Bamboogle 125 Multi-hop Out-of-domain
In practice, we compute the mean and variance of empirical correctness across reranker rollouts. Queries with near-zero
variance or near-deterministic outcomes are removed, as they correspond to trivially easy or unanswerable cases. Unless
otherwise stated, we retain queries whose mean correctness satisfies µq∈[m min, mmax]and whose variance exceeds vmin.
In all experiments, we use N= 8 reranker rollouts and K= 10 generator rollouts. We set the correctness threshold
toδ= 0.5 . The mean correctness bounds are mmin= 0.25 andmmax= 0.85 , and the minimum variance threshold is
vmin= 0.02. This filtering step is applied only to selector training and does not affect generator training.
A.7. Training Data
Following Jin et al. (2025), we construct the training set by merging the training splits of Natural Questions
(NQ) (Kwiatkowski et al., 2019) and HotpotQA (Yang et al., 2018). NQ provides single-hop factoid questions derived from
real Google search queries, while HotpotQA contributes multi-hop questions that require reasoning over multiple Wikipedia
passages. This combination ensures coverage of both simple retrieval scenarios and complex multi-step reasoning tasks.
For the retrieval corpus, we use the 2018 Wikipedia dump (Karpukhin et al., 2020), which contains approximately 21 million
passages. We employ E5 (Wang et al., 2022) as the dense retriever. For each query, we retrieve the top-25 passages to form
a fixed retrieval pool, from which the selector learns to compose evidence subsets during training.
Table 5 summarizes the dataset statistics. The combined training set contains 87,925 question-answer pairs. Each training
instance consists of a question q, the ground-truth answer a, and a retrieval pool Dqcontaining the top-25 passages retrieved
forq. DuringBAR-RAGtraining, the selector operates over this fixed retrieval pool, learning to compose evidence subsets
that challenge the generator near its competence boundary.
We evaluate on seven benchmark datasets to assess both in-domain and out-of-domain generalization: (1)In-domain: NQ
and HotpotQA; (2)Out-of-domain: TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023), 2WikiMultiHopQA (Ho
et al., 2020), Musique (Trivedi et al., 2022), and Bamboogle (Press et al., 2023). Following?, we use Exact Match (EM) as
the evaluation metric.
A.8. Case Study
Table 6 presents a representative case study illustrating how BAR-RAG progressively hardens evidence composition across
training iterations while operating over a fixed top-25 retrieval pool.
In Iteration 1, the selector surfaces the two [GOLDEN] documents at the top of the evidence list. One document explicitly
identifies Claudio L ´opez as a retired Argentine forward who played as a main attacking player for Valencia CF, while the
other confirms his role as a regular starter in Valencia’s attacking line during the relevant period. The remaining documents
are largely [IRRELEV ANT] and do not introduce strong competing signals. As a result, the generator can answer the
15

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
question correctly with minimal reasoning effort, relying primarily on direct evidence aggregation.
In Iteration 2, the same two golden documents remain present but are no longer adjacent. They are interleaved with
[MISLEADING] documents that are highly relevant at the surface level, such as profiles of Mario Kempes or summaries of
Argentine forwards in La Liga. These distractors satisfy several query attributes (e.g., nationality, position, club association)
but fail to jointly satisfy all constraints. Consequently, positional heuristics or shallow relevance matching become unreliable,
and correct prediction requires identifying and combining the truly decisive evidence.
By Iteration 3, BAR-RAG further increases structural difficulty by pushing the two golden documents deeper into the
evidence set and surrounding them with multiple misleading but plausible alternatives. The shortcut evidence remains
visible but becomes insufficient: although figures such as Mario Kempes match many individual query facets, they do not
simultaneously satisfy the conjunction of being retired, Argentine, a forward, and a main player for Valencia CF during the
specified era. Only by consistently tracking entity identity across the dispersed golden documents can the generator arrive at
the correct answer.
Across iterations, task solvability is preserved, as the same two golden documents are always present. However, the
evidence structure is progressively hardened: decisive information is no longer top-ranked or contiguous, and misleading
cues increasingly dominate surface relevance. This case study demonstrates that BAR-RAG improves robustness not by
introducing new evidence, but by reshaping the composition and ordering of existing retrieval results to suppress shortcut
reasoning and induce genuine multi-hop integration near the generator’s competence boundary.
16

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 6.Case study illustrating boundary-aware evidence selection from a fixed top-25 retrieval pool. Document IDs correspond to
original retrieval ranks. Across iterations, the same two [GOLDEN] documents remain necessary to answer the question, while their
relative positions are progressively dispersed and surrounded by [MISLEADING] but topically relevant noise and [IRRELEV ANT]
documents. This structured hardening preserves solvability while forcing multi-hop reasoning near the generator’s competence boundary.
Question:Which retired Argentine footballer who played as a forward was a main player for Valencia CF?
Iteration 1 Iteration 2 Iteration 3
(Doc 1)[GOLDEN]
Title:Claudio L ´opez — Career Sum-
mary
Content:Claudio L ´opez is a retired Ar-
gentine footballer who played as a for-
ward and was a main attacking player
for Valencia CF.
(Doc 2)[GOLDEN]
Title:Valencia CF Squad (1998–2000)
Content:Valencia relied on Claudio
L´opez as a regular starter in their attack-
ing line during this period.
(Doc 3)[IRRELEV ANT]
Title:Valencia CF History (1990s)
(Doc 4)[IRRELEV ANT]
Title:Notable Footballers in La Liga
(Doc 5)[MISLEADING]
Title:Overview of Spanish Football
Clubs(Doc 1)[GOLDEN]
Title:Claudio L ´opez — Career Sum-
mary
Content:Claudio L ´opez played as a
main forward for Valencia CF in the
late 1990s.
(Doc 5)[MISLEADING]
Title:Mario Kempes — Career
Overview
Content:Mario Kempes is a retired
Argentine forward and one of Valencia
CF’s most iconic historical players.
(Doc 2)[GOLDEN]
Title:Valencia CF Squad (1998–2000)
Content:Valencia relied on a fast Ar-
gentine forward as a regular starter in
their attacking system.
(Doc 7)[MISLEADING]
Title:Argentine Forwards in La Liga
Content:Several Argentine forwards,
including Kempes and others, played
important roles in La Liga clubs.
(Doc 8)[IRRELEV ANT]
Title:History of La Liga Stadiums(Doc 5)[MISLEADING]
Title:Mario Kempes — Career
Overview
Content:Kempes is remembered as
one of the most influential Argentine
forwards in Spanish football.
(Doc 11)[MISLEADING]
Title:Argentine Legends in La Liga
Content:Several Argentine forwards
achieved legendary status at Spanish
clubs.
(Doc 1)[GOLDEN]
Title:Claudio L ´opez — Career Sum-
mary
Content:Claudio L ´opez is a retired
Argentine forward who played for Va-
lencia CF.
(Doc 14)[MISLEADING]
Title:Valencia CF Legends
Content:Valencia CF has featured
many historically significant attacking
players.
(Doc 2)[GOLDEN]
Title:Valencia CF Squad (1998–2000)
Content:Valencia’s main attacking op-
tions during this era included a fast Ar-
gentine forward.
Answer:Claudio Javier L ´opez
A.9. Prompt
This section presents the prompt templates we used for the generator and selector, in Table 7 and Table 8, respectively.
17

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 7.Prompt template for reasoning generator.
Reasoning Generator Prompt
You are given a question and a set of retrieved documents. Your task is to answer the questionusing
only information from the retrieved documents. Even for yes/no questions, you must determine the
answer by reasoning from factual evidence in the documents.
Output format (STRICT):
•<think> A concise reasoning chain explaining how the answer is derived from the documents.
Keep it brief (1–3 sentences).</think>
•<answer>The final answer.</answer>
Evidence citation rule:
•Whenever you use a piece of evidence from the documents in your reasoning, youmustcite it
inline asDoc [i].
• You may cite one or multiple documents, but only cite documents that are actually relevant.
Answer rules:
• The answer should be ashort phrasedirectly supported by the retrieved documents.
• Donotintroduce external knowledge or assumptions.
• Donotoutput anything outside<think>and<answer>.
Example (follow the style only):
<think> Doc [1] states that Future Ted serves as the show’s
narrator, and Doc [4] confirms that the narrator is voiced by Bob
Saget. </think>
<answer> Ted Mosby </answer>
<Question>
<Retrieved Documents>
18

Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
Table 8.Prompt template for evidence selector.
Evidence Selector Prompt
You are an expert evidence-set selector for RAG. Your goal is to selectexactly fivedocuments that make
the questionanswerable, but not trivial. Prefer evidence sets that sit near the model’scompetence
boundary: solvable with careful multi-step reasoning, yet not so direct that the answer is obvious from
a single passage. You must follow the principles and output format strictly.
Principles:
1.Answerability (must-have):The selected set must contain enough information to deduce the
correct answer. Donotselect sets that make the question impossible.
2.Non-triviality (must-have):Avoid sets where one document directly states the answer with
no integration needed. If a direct-answer passage is unavoidable for solvability, include itonly
together withsupporting/context passages that require cross-document integration.
3.Multi-hop integration:Prefer sets that require combining at leasttwocomplementary clues (e.g.,
entity linking, temporal alignment, resolving aliases, chaining relations).
4.Controlled noise:Mildly conflicting or distracting details are allowed if the set remains answerable;
do not include documents that are irrelevant or make the set unsolvable.
5.Diversity:Prefer complementary documents covering different parts of the reasoning chain, rather
than near-duplicates.
Output format (STRICT):
•<think> Briefly explain which documents contain key clues, how they complement each other,
and why the set is answerable but requires integration. Keep it concise (2–4 sentences). </think>
•<answer> [doc id1], [doc id2], [doc id3], [doc id4], [doc id5]
</answer>
Rules:
• Selectexactly 5documents.
• In<answer>, listonlythe document identifiers in brackets, separated by commas.
• Donotoutput anything outside<think>and<answer>.
Example (follow the style only):
<think> Doc [3] provides the birthplace clue, Doc [7] gives a
timeline, and Doc [12] resolves an alias; combining them is
necessary. Doc [5] and Doc [9] add supporting context while
introducing mild distraction, keeping the set solvable but
non-trivial. </think>
<answer> [3], [5], [7], [9], [12] </answer>
<Question>
<Candidate Documents (Top-K)>
19
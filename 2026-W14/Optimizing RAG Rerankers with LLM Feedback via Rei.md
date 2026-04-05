# Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning

**Authors**: Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi Zhou, Zhen Wu, Xinyu Dai, Rui Xia

**Published**: 2026-04-02 14:19:47

**PDF URL**: [https://arxiv.org/pdf/2604.02091v1](https://arxiv.org/pdf/2604.02091v1)

## Abstract
Rerankers play a pivotal role in refining retrieval results for Retrieval-Augmented Generation. However, current reranking models are typically optimized on static human annotated relevance labels in isolation, decoupled from the downstream generation process. This isolation leads to a fundamental misalignment: documents identified as topically relevant by information retrieval metrics often fail to provide the actual utility required by the LLM for precise answer generation. To bridge this gap, we introduce ReRanking Preference Optimization (RRPO), a reinforcement learning framework that directly aligns reranking with the LLM's generation quality. By formulating reranking as a sequential decision-making process, RRPO optimizes for context utility using LLM feedback, thereby eliminating the need for expensive human annotations. To ensure training stability, we further introduce a reference-anchored deterministic baseline. Extensive experiments on knowledge-intensive benchmarks demonstrate that RRPO significantly outperforms strong baselines, including the powerful list-wise reranker RankZephyr. Further analysis highlights the versatility of our framework: it generalizes seamlessly to diverse readers (e.g., GPT-4o), integrates orthogonally with query expansion modules like Query2Doc, and remains robust even when trained with noisy supervisors.

## Full Text


<!-- PDF content starts -->

Optimizing RAG Rerankers with LLM Feedback
via Reinforcement Learning
Yuhang Wu1Xiangqing Shen1Fanfan Wang1Cangqi Zhou1
Zhen Wu2Xinyu Dai2Rui Xia2
1Nanjing University of Science and Technology
2Nanjing University
{yhangwu, xiangqing.shen, ffwang, cqzhou}@njust.edu.cn
{wuz, daixinyu, rxia}@nju.edu.cn
Abstract
Rerankers play a pivotal role in refining re-
trieval results for Retrieval-Augmented Gen-
eration. However, current reranking models
are typically optimized on static human anno-
tated relevance labels in isolation, decoupled
from the downstream generation process. This
isolation leads to a fundamental misalignment:
documents identified as topically relevant by
information retrieval metrics often fail to pro-
vide the actual utility required by the LLM
for precise answer generation. To bridge this
gap, we introduce ReRanking Preference Op-
timization (RRPO), a reinforcement learning
framework that directly aligns reranking with
the LLM’s generation quality. By formulat-
ing reranking as a sequential decision-making
process, RRPO optimizes for context utility
using LLM feedback, thereby eliminating the
need for expensive human annotations. To en-
sure training stability, we further introduce a
reference-anchored deterministic baseline. Ex-
tensive experiments on knowledge-intensive
benchmarks demonstrate that RRPO signifi-
cantly outperforms strong baselines, including
the powerful list-wise reranker RankZephyr.
Further analysis highlights the versatility of our
framework: it generalizes seamlessly to diverse
readers (e.g., GPT-4o), integrates orthogonally
with query expansion modules like Query2Doc,
and remains robust even when trained with
noisy supervisors.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Borgeaud et al., 2022; Izacard and
Grave, 2020; Gao et al., 2023) has emerged as a
powerful paradigm to enhance LLMs by dynami-
cally incorporating external information, thereby
improving factual accuracy and contextual rele-
vance for knowledge-intensive tasks. In a typical
pipeline, the system first retrieves documents based
on a user query and then passes them to an LLM
Reader for response generation. Consequently, theoverall performance depends heavily on the qual-
ity of these retrieved documents. To optimize this,
a reranking model (reranker) is often introduced
to refine the initial document set. By prioritizing
the most relevant documents, the reranker ensures
the LLM receives high-quality context, leading to
more precise and coherent answers.
However, a fundamental disconnect persists in
current RAG architectures: reranker optimiza-
tion is typically isolated from the RAG pipeline.
Most rerankers are optimized under a supervised
paradigm using static Information Retrieval metrics
(e.g., NDCG), a process that is entirely decoupled
from the downstream LLM. As shown in Figure 1,
this creates a critical misalignment: the notion of
relevance learned by the reranker does not necessar-
ily translate to the usefulness required by the LLM
to generate precise answers (Lee et al., 2025). For
instance, given the query “Who discovered peni-
cillin?”, a standard reranker might favor a long
historical article (high topic relevance), whereas
the LLM might benefit more from a concise time-
line entry (high LLM usefulness). By treating the
reranker as an independent module rather than an
integrated component of the RAG pipeline, stan-
dard approaches fail to optimize for the system’s
ultimate objective—generation quality.
To bridge this gap, we introduce ReRanking
Preference Optimization (RRPO), a reinforcement
learning framework that directly optimizes the
reranker within the RAG pipeline using the LLM’s
generation quality as the reward signal. Unlike
supervised approaches that rely on static, human-
annotated relevance labels, RRPO leverages the
downstream LLM itself to provide scalable super-
vision, thereby eliminating the need for expensive
manual annotation. RRPO reframes reranking as a
sequential decision-making process, allowing for
end-to-end alignment between retrieval and gen-
eration. To address the instability caused by dis-
crete, high-variance LLM rewards, we introducearXiv:2604.02091v1  [cs.CL]  2 Apr 2026

Figure 1: Comparison of previous methods with our proposed RRPO framework.
a reference-anchored deterministic baseline. This
component effectively stabilizes training by com-
pelling the reranker to prioritize document “use-
fulness” over mere topic “relevance”, avoiding the
computational complexity and instability associ-
ated with training a separate critic network.
We validate the effectiveness of RRPO through
extensive experiments on standard knowledge-
intensive benchmarks. Empirical results demon-
strate that RRPO achieves consistent improvements
by explicitly optimizing the reranker to align with
reader needs, significantly outperforming strong
supervised baselines. Notably, our method sur-
passes state-of-the-art list-wise LLM rerankers,
RankZephyr (Pradeep et al., 2023). This confirms
that aligning the reranker with the reader’s specific
needs yields good context utility.
Beyond performance gains, our analysis high-
lights three critical properties that distinguish
RRPO as a robust solution. First, it exhibits strong
generalization: the learned policy not only han-
dles complex multi-hop reasoning but also transfers
seamlessly to diverse LLM Readers, effectively
enhancing even proprietary models like GPT-4o.
Second, it offers orthogonal integration: function-
ing as a “plug-and-play” module, it complements
advanced retrieval strategies, delivering additive
gains when combined with query expansion meth-
ods like Query2Doc (Wang et al., 2023a). Finally,
it demonstrates training robustness: RRPO main-
tains its effectiveness even when optimized using
feedback from smaller, noisy supervisors, signifi-
cantly reducing training costs.2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) enhances
LLMs by incorporating external information to
mitigate hallucinations and knowledge obsoles-
cence (Ji et al., 2023; Guu et al., 2020; Lewis
et al., 2020). While early frameworks focused on
dynamic knowledge provisioning for QA and di-
alogue (Gao et al., 2023; Karpukhin et al., 2020;
Izacard et al., 2023; Shuster et al., 2021), subse-
quent research has refined the generation process.
Key advancements include instruction-tuning gen-
erators to better utilize context (Ma et al., 2024;
Zhu et al., 2024; Muennighoff et al., 2024; Liu
et al., 2024; Lin et al., 2023) and incorporating
self-reflection mechanisms like Self-RAG (Asai
et al., 2024). Furthermore, sophisticated retrieval
strategies such as iterative, adaptive, or active re-
trieval have been developed to handle complex
queries (Trivedi et al., 2022a; Jiang et al., 2023;
Jeong et al., 2024; Xu et al., 2024).
2.2 Rerankers in RAG
Rerankers serve as a critical refinement stage be-
tween retrieval and generation. Early methods
employed BERT-based encoders for pointwise or
pairwise relevance estimation (Nogueira and Cho,
2019; Nogueira et al., 2019). Recent approaches
explicitly leverage LLM capabilities, ranging from
zero-shot listwise prompting (Sun et al., 2023) or
discrete prompt optimization (Cho et al., 2023),
to fine-tuning open-source LLMs on ranking data,

as seen in RankZephyr (Pradeep et al., 2023; Ma
et al., 2024). Other works have explored unsuper-
vised risk minimization (Yuan et al., 2024) or uti-
lized attention mechanisms for efficiency (Reddy
et al., 2024; Chen et al., 2024). Most recently, re-
search has expanded into context-aware and agen-
tic paradigms. EBCAR (Yuan et al., 2025) utilizes
embeddings to capture cross-passage interactions,
while REARANK (Zhang et al., 2025a) formulates
reranking as a reasoning agent.
2.3 Reinforcement Learning for RAG
Reinforcement Learning (RL) has been widely
adopted to align RAG components with end-task
performance. Prior work has applied RL to op-
timize query rewriting (Wang et al., 2023b), in-
corporate reasoning traces via ReAct (Yao et al.,
2023), or facilitate multi-step browsing (Nakano
et al., 2021). A growing trend involves align-
ing retrieval with generation via preference opti-
mization. Methods like DynamicRAG (Sun et al.,
2025), KnowPO (Zhang et al., 2025b), and DPA-
RAG (Dong et al., 2025) apply preference opti-
mization (e.g., DPO) to fine-tune models, while
others explore multi-agent reinforcement learning
architectures (Chen et al., 2025). In contrast to
these DPO-based approaches or complex multi-
agent frameworks, RRPO employs a sequential
decision-making formulation to explicitly capture
the combinatorial utility of document sets.
3 Methodology
We propose RRPO, a framework that fundamen-
tally redefines document reranking as a sequen-
tial decision-making process. This approach is
designed to bridge the misalignment between the
relevance scores predicted by traditional rerankers
and the actual usefulness of documents for the
downstream LLM. Unlike supervised methods that
rely on static labels, RRPO leverages Reinforce-
ment Learning (RL) to optimize document selec-
tion based on direct feedback from the frozen LLM
Reader. This allows for end-to-end alignment of
the retrieval pipeline with the final generation qual-
ity. In this section, we formulate the reranking
problem as a Markov Decision Process (MDP) and
detail the components of our training algorithm,
highlighting our reference-anchored baseline.
3.1 Task Formulation
Given a user query qand an initial candidate set of
Ndocuments D={d 1, . . . , d N}retrieved froma corpus, the goal of the reranker is to select an
ordered subset of kdocuments (k < N) that max-
imizes the quality of the response generated by a
fixed LLM Reader.
We formalize this selection process as a finite-
horizon Markov Decision Process (MDP) defined
by the tuple(S,A, P, R):
State (st): At time step t, where t∈ {1, . . . , k} ,
the state stis defined as the set of candidate docu-
ments that have not yet been selected. The initial
state s1={d 1, d2, . . . , d N}is the set of top- N
documents returned by the initial retrieval module.
Action (at): In state st, an action atdenotes
selecting one document dctfrom the current set of
documents st. Here, ctis the index of this doc-
ument in the original list of Ndocuments. The
action spaceA tvaries at each time step:
At={selectd i|di∈st}.(1)
Transition Dynamics (P(s t+1|st, at)): The
state transitions are deterministic. Upon executing
action at(i.e., selecting document dct) in state st,
the system transitions to the next state st+1, which
is obtained by removing the selected document dct
froms t:
st+1=st\{dct}.(2)
Reward (rt): After executing action atat each
time step t, the agent receives a reward rt. This
reward reflects the contribution of the set of doc-
uments selected up to the current time step t,
{dc1, . . . , d ct}, to the generation of a high-quality
response by the downstream LLM Reader. The
downstream LLM Reader generates a specific re-
ward value based on an evaluation metricR lm.
3.2 ReRanking Preference Optimization
We detail our ReRanking Preference Optimiza-
tion framework in three aspects: Agent and Policy
Network, Optimization objective, and Reference-
anchored Deterministic Baseline.
3.2.1 Agent and Policy Network
In this framework, the agent is implemented by
a parameterized reranking model fθwith parame-
tersθ. The agent’s behavior is defined by a policy
πθ(at|st). We realize our policy πθthrough a
pointwise reranking model fθ. For a given query q
and each document di(i= 1, . . . , N) in the initial
candidate list, the reranker model fθcomputes a
relevance score:
score i=fθ(q, d i),(3)

where the scores for all Ncandidate documents
are then converted into an initial probability distri-
bution {p1, . . . , p N}using the SoftMax function,
where pirepresents the initial probability that doc-
umentd iis considered relevant:
pi=exp (score i)PN
j=1exp (score j).(4)
At each Reinforcement Learning time step t, when
the agent is in state st(the set of remaining docu-
ments), the probability of selecting action at(i.e.,
choosing document dct∈st),πθ(at|st)is de-
fined as the proportion of this document’s initial
probability pctto the sum of initial probabilities of
the currently remaining documents:
πθ(at=selectd ct|st)
=pctP
dj∈stpj=pct
1−Pt−1
l=1pcl.(5)
Reward Calculation: The sequence of selected
documents {dc1, . . . , d ct}is combined with the
original query qand an instruction INST, and then
input to the LLM Reader to generate a response:
responset=Reader LLM(INST, q, d c1, . . . , d ct).
(6)
An evaluation metric Rlmis used to compute a
score between the generated response responset
and the ground truth answer ans, which serves as
the reward for that time step:
rt=R lm(ans,responset).(7)
The specific settings for the instruction INST and
the evaluation metric Rlmare dataset-dependent
and will be detailed in the Experiments section.
3.2.2 Objective Function
In practice, we found it necessary to constrain the
magnitude of updates to the reranker model to pre-
vent excessive oscillations in the training process
that can result from large policy gradient updates.
Therefore, drawing inspiration from the Proximal
Policy Optimization (PPO) algorithm, we define
our specific training objective as
J(θ) =E τ∼πθ"
1
kkX
t=1min
ρt(θ)ˆAt,
clip 
ρt(θ),1−ϵ,1 +ϵˆAt
−βD KL(πθ∥πref)#
,
(8)where ρt(θ)is the importance sampling ratio
ρt(θ) =πθ(at|st)
πref(at|st),πrefis a fixed reference policy
andˆAtis the estimated advantage function at time
stept, which will be detailed later.
Here, we employ both the PPO-clip mechanism
and a KL divergence penalty to constrain the step
size of each model update. We note that this combi-
nation of mechanisms have been extensively used
and proven successful in RLHF (PPO-like algo-
rithms) for language models (Ouyang et al., 2022;
Rafailov et al., 2023; Shao et al., 2024), leading us
to believe this setup is well-justified.
3.2.3 Reference-anchored Deterministic
Baseline
In order to compute the advantage function ˆAre-
quired by the PPO objective, we first introduce Gen-
eralized Advantage Estimation (GAE) (Schulman
et al., 2015) as the framework for advantage calcu-
lation. A critical challenge in applying RL to RAG
is the high variance and sparsity of LLM-based
rewards, which makes training a separate value net-
work (critic) notoriously difficult. To address this,
we introduce a reference-anchored deterministic
baseline for definingV(s t).
GAE in Objective FunctionWe use GAE to
compute the raw advantage function At, which is
then normalized to obtain ˆAt. GAE is a widely
adopted method for effectively estimating the ad-
vantage function by balancing bias and variance,
which is also used in PPO. The TD residual from
time stepttot+ 1is defined as
δt=rt+γV(s t+1)−V(s t).(9)
Atis then calculated as (for an episode of length
k):
At=δt+ (γλ)δ t+1+···+ (γλ)k−t−1δk−1
=k−1X
j=t(γλ)j−tδj,
(10)
where the parameter λ∈[0,1] controls the bias-
variance trade-off in the advantage estimation. A
value of λ= 0 would result in using only the single-
step TD residual δt, which has low bias but high
variance. Conversely, λ= 1 would correspond to a
Monte Carlo estimate over the rest of the episode,
which has high bias but lower variance. By using
a value between 0 and 1, GAE provides a fine-
grained control to reduce variance while keeping

bias at an acceptable level, which is crucial for
stable policy updates. We explain the value of λin
the Appendix B.
Limitations of Standard PPO CriticsOur task
exhibits certain characteristics that differ from typ-
ical RL modeling in RLHF. In our definition, the
episode length kis relatively short, and the reward
signal computed by Rlmcan be highly discrete and
exhibit large variance. We found in practice that
training a separate critic network to fit V(st)was
ineffective. Critic models based on deep networks
tend to predict numerically continuous values, mak-
ing it difficult for them to accurately capture the
value associated with LLM Reader’s evaluations,
especially for Factual QA tasks. This can lead to a
decline in policy performance.
Reference Policy Rollout as a Deterministic
BaselineTherefore, departing from traditional
PPO and its classic applications in RLHF for LLMs,
we adopt a deterministic baseline approach that
uses actual observed values. This involves gen-
erating a reference trajectory using the reference
model, which is then evaluated by the LLM Reader
andRlmin the environment. The value of being in
statest(at step tof the agent’s trajectory), denoted
V(st), is defined as the reward obtained by the ref-
erence policy πrefexecuting a greedy sequence of
tactions starting from the initial state s1. Specif-
ically, for each step j= 1, . . . , t in this reference
rollout, a document is greedily selected according
toπref, which is described as actiona′
j:
a′
j= argmax
a∈A(s′
j)πref(a|s′
j),(select documentd′
cj)
(11)
where s′
1, . . . , s′
tis the path generated by these
greedy selections by πref, with its initial state
s′
1=s 1={d 1, . . . , d N}. Let d′
cjbe the doc-
ument selected by the reference model at its j-th
step. The response is generated using these selected
documents{d′
c1, . . . , d′
ct}
response′
t=Reader LLM(INST, q, d′
c1, . . . , d′
ct).
(12)
The baseline value is then computed as
V(st) =R lm(ans,response′
t).(13)
Advantages of Reference-Anchored Optimiza-
tionBy anchoring the time-dependent baseline
V(st)to the deterministic greedy performance ofthe reference policy πref, we avoid the complexi-
ties of training a critic network and the heavy re-
liance on human preference data typically required
in RLHF. This approach artfully integrates off-
policy insights into PPO’s on-policy framework,
leveraging the typically strong reference reranker
as a stable anchor for advantage estimation. By
incorporating this reference-based value into the
advantage calculation—alongside the importance
sampling ratio and KL divergence penalty—we
ensure that policy updates are robustly guided to-
wards outperforming πrefwhile mitigating the risk
of significant performance degradation during ex-
ploration.
3.2.4 Training Stabilization Strategies
Even with the deterministic baseline, the raw ad-
vantage values Atcan exhibit significant variance.
To further stabilize the policy gradient updates, we
apply advantage normalization:
ˆAt=At−mean(A window )
std(A window ) +ϵ norm,(14)
where statistics are computed over a sliding win-
dow of recent episodes, andϵ norm= 10−8.
Additionally, for the KL penalty term in the
objective function, we employ the estimator
from (Ouyang et al., 2022):
DKL(πθ∥πref) =πref(at|st)
πθ(at|st)−logπref(at|st)
πθ(at|st)−1.
(15)
We provide the pesudo codes of RRPO in Algo-
rithm 1 of Appendix A.
4 Experiments
4.1 Experimental Settings
DatasetsAmbigNQ (Min et al., 2020) is a
disambiguated version of Natural Questions
(NQ) (Kwiatkowski et al., 2019). AmbigNQ aims
to elicit unique, definitive answers by providing
more precise question formulations or contexts.
We use its full training set and evaluate on valida-
tion set. HotpotQA (Yang et al., 2018) is a widely
used multi-hop question answering (multi-hop QA)
benchmark dataset, where questions typically re-
quire reasoning over information combined from
multiple documents. We selected HotpotQA be-
cause multi-hop reasoning scenarios place higher
demands on the collaborative capabilities of the
various modules within RAG systems. We use full
training set and evaluate on validation set.

BaselinesTo evaluate RRPO, we establish sev-
eral key comparisons. Our primary baseline is
the base reranker model gte-multilingual-reranker-
base (Zhang et al., 2024) (reported as “gte
reranker”) without RRPO’s reinforcement learn-
ing optimization, providing a direct ablation of our
training paradigm’s impact. We further benchmark
against a diverse set of off-the-shelf alternatives.
Specifically, we includeencoder-onlymodels like
jina-reranker-v2-multilingual (Jina AI, 2024) and
bge-reranker-base (Xiao et al., 2023), alongside the
decoder-onlymodel Qwen3-reranker-0.6B (Zhang
et al., 2025c) (reported as “qwen3 reranker”).
For broader context on RAG system performance,
we also reference advanced RAG methods like
FLARE (Jiang et al., 2023) and DRAGIN (Su et al.,
2024) (with results cited from DeepRAG (Guan
et al., 2025) for fair comparison). These meth-
ods typically employ strategies like confidence-
based adaptive retrieval, representing orthogonal
approaches to RAG optimization rather than direct
reranker baselines.
Implementation DetailsWe conduct all exper-
iments using the preprocessed knowledge corpus
provided by DPR (Karpukhin et al., 2020), where
articles are segmented into 100-word passages. Our
retrieval pipeline employs BM25 (Robertson and
Zaragoza, 2009) to initially fetch 100candidate
documents, which are subsequently processed by
the evaluated reranking models (e.g., gte, jina, and
qwen3). From these reranked candidates, we se-
lect the top- kpassages ( k= 3 for HotpotQA and
k= 5 for AmbigNQ) to serve as input for the
Qwen2.5-7B (Team, 2024) LLM reader. Regard-
ing the optimization objective, we follow previous
work (Ma et al., 2023) by adopting a unified re-
ward function Rlm=EM+λ fF1+λ hHitfor both
datasets, with hyperparameters set to λf= 1.0 and
λh= 1.0 . The Hit metric is defined to be 1if the
ground-truth answer appears in the generated re-
sponse and −1otherwise. Further training details
including fair comparison discussions are provided
in Appendix B.
4.2 Main Results
The results in Table 1 demonstrate that while stan-
dard pre-trained rerankers significantly improve
upon Naive RAG, applying the RRPO paradigm
yields decisive further gains. Specifically, the
RRPO-optimized gte reranker boosts F1 scores by
1.99 on HotpotQA and 1.12 on AmbigNQ, withMethodsHotpotQA AmbigNQ
EM F1 EM F1
LLM Reader only 17.80 26.85 3.55 17.80
LLM + CoT 21.34 31.17 10.69 27.85
FLARE 23.40*32.06*25.62 37.78
DRAGIN 16.70*24.60*18.33 30.09
Naive RAG (BM25) 24.23 35.42 20.33 32.18
+ qwen3 reranker 17.03 25.71 31.17 42.87
+ bge reranker 28.06 40.46 31.37 45.01
+ jina reranker 28.09 40.65 32.27 45.53
+ gte reranker 28.24 41.23 35.51 48.55
+ qwen3 reranker (RRPO) 18.43 27.73 32.52 45.17
+ bge reranker (RRPO) 28.12 40.76 32.37 45.87
+ jina reranker (RRPO) 29.26 42.16 32.86 46.20
+ gte reranker (RRPO)30.03 43.22 36.56 49.67
Table 1: Main experimental results on the HotpotQA
and AmbigNQ datasets.*adapts from previous work
DeepRAG (Guan et al., 2025).
1 3 5
Number of Documents (keval)28303234 EM
Baseline (gte reranker)
RRPO (ktrain=5)
RRPO (ktrain=3)
RRPO (ktrain=1)
1 3 5
Number of Documents (keval)424446F1
Baseline (gte reranker)
RRPO (ktrain=5)
RRPO (ktrain=3)
RRPO (ktrain=1)
Figure 2: Ablation Experiments on HotpotQA.
similar consistency observed across other archi-
tectures like jina. Crucially, our results demon-
strate that RRPO is effective across diverse model
architectures, yielding consistent gains for both
encoder-onlyanddecoder-onlyrerankers. Sta-
tistical significance tests, detailed in Appendix H,
confirm that these improvements are statistically
significant. These results not only verify that our
RL-based optimization successfully aligns retrieval
with downstream generation needs but also estab-
lish RRPO as highly competitive against advanced
RAG strategies like FLARE and DRAGIN.
4.3 Ablation Study
Figure 2 investigates the impact of training inter-
action depth ( ktrain∈ {1,3,5} ) on HotpotQA. All
RRPO variants consistently outperform the non-
RL baseline, demonstrating robustness. Notably,
ktrain = 3 yields optimal results, balancing suf-
ficient context against the noise and attention dif-
fusion of larger sets ( ktrain = 5) and the limited
scope of single-document training (k train = 1).

Methods EM F1
Naive RAG (BM25) 26.11 30.40
+ gte reranker 26.87 31.42
+ gte reranker (RRPO)27.68 32.48
Table 2: Performance of RRPO on 2WikiMultiHopQA.
4.4 Extensive Analysis on Generalizability
and Robustness
In this section, we provide a comprehensive analy-
sis demonstrating that RRPO learns a robust, trans-
ferable, and highly efficient document utility policy.
Going beyond standard benchmarks, our experi-
ments validate five key properties of the proposed
framework: (1) it generalizes effectively to struc-
turally complex multi-hop reasoning scenarios; (2)
it transfers seamlessly across diverse LLM read-
ers (including closed-source models) without fine-
tuning; (3) it outperforms state-of-the-art list-wise
rerankers by explicitly optimizing for reader utility;
(4) it yields orthogonal gains when integrated into
advanced RAG pipelines; and (5) it maintains high
label efficiency even when distilled from smaller
supervisors.
RRPO demonstrates robust generalization to
structurally complex multi-hop datasets.Instead
of limiting our evaluation to simple pattern match-
ing, we extend the scope to 2WikiMultiHopQA (Ho
et al., 2020) and MusiQue (Trivedi et al., 2022b),
datasets that necessitate rigorous information inte-
gration across disconnected documents. Using the
identical training configuration as the HotpotQA
experiments, Table 2 reports the performance in the
top-5 context setting (with comprehensive results
in Appendix E). RRPO consistently outperforms
the strong gte reranker baseline across these bench-
marks. Crucially, this performance gain on com-
plex multi-hop datasets confirms that our method
does not merely overfit to specific query patterns.
Instead, it successfully learns to identify the co-
hesive chain of evidence required for multi-hop
reasoning, validating its semantic robustness be-
yond the training domain.
The learned ranking policy proves highly
transferable across diverse LLM architectures
and scales.We investigate whether RRPO captures
a universal criterion for document utility regard-
less of the downstream reader. As illustrated in
Figure 3, our method demonstrates strong “plug-
and-play” potential across a wide spectrum of mod-
els without any reader-specific fine-tuning. First,MethodsHotpotQA AmbigNQ
EM F1 EM F1
BM25 + gte reranker 32.87 45.06 40.26 51.36
BM25 + RankZephyr 32.40 44.49 40.06 51.58
BM25 + gte reranker (RRPO)34.10 46.38 41.11 52.18
Table 3: Comparison of RRPO and the list-wise LLM
reranker RankZephyr.
the gains consistently transfer across model scales,
benefiting both smaller (3B) and larger (14B) vari-
ants of the Qwen family. More importantly, the
policy proves effective across differing architec-
tures, extending benefits to Llama-3.1 and GLM-
4. This cross-architecture transferability suggests
that RRPO relies on intrinsic semantic informa-
tion density rather than model-specific parametric
knowledge. Finally, the method acts as a genuinely
reader-agnostic filter, enhancing even proprietary,
closed-source models such as Gemini-2.5-Flash,
Claude-3.5-Sonnet, and GPT-4o.
RRPO outperforms state-of-the-art list-
wise rerankers by explicitly optimizing for
reader utility.To position our method within
the current landscape, we compare it against
RankZephyr (Pradeep et al., 2023), a represen-
tative 7B list-wise LLM reranker known for its
GPT-4-level performance. As shown in Table 3,
RRPO achieves superior performance compared
to RankZephyr in the top-3 context setting across
both HotpotQA and AmbigNQ. While list-wise
approaches leverage the general reasoning capabili-
ties of LLMs to sort documents, our results indicate
that they may not be optimally aligned with the spe-
cific needs of a reader in RAG. In contrast, RRPO’s
policy—distilled through RRPO —is explicitly op-
timized to maximize reader performance, allowing
it to filter for utility more effectively than general-
purpose ranking prompts.
RRPO provides orthogonal improvements to
query expansion techniques within advanced
RAG frameworks.Retrieval pipelines often em-
ploy query expansion (e.g., Query2Doc (Wang
et al., 2023a)) to address low recall. We ex-
plore whether our method contributes value beyond
merely compensating for weak initial retrieval. Ta-
ble 4 reveals a strictly tiered improvement struc-
ture. Even atop the strong baseline established by
Query2Doc, RRPO delivers significant additional
gains. This confirms that the two components play
orthogonal roles: while Query2Doc expands the

2030EM Score @ Top-13040EM Score @ Top-3
Qwen2.5-7B Qwen2.5-3B Qwen2.5-14B Llama-3.1-8B GLM-4-9BGPT-4o Claude-3.5 Gemini-2.53040F1 Score @ Top-1
Qwen2.5-7B Qwen2.5-3B Qwen2.5-14B Llama-3.1-8B GLM-4-9BGPT-4o Claude-3.5 Gemini-2.54050F1 Score @ Top-3GTE Reranker Baseline Our RRPOFigure 3: Generalization of RRPO-trained reranker to Various LLM Readers on HotpotQA.
MethodsHotpotQA AmbigNQ
EM F1 EM F1
Naive RAG (BM25) 24.23 35.42 20.33 32.18
+ Query2Doc 28.79 41.78 36.06 50.85
+ gte reranker 28.87 42.06 38.76 53.14
+ gte reranker (RRPO)30.51 44.10 39.81 54.10
Table 4: Performance of RRPO within the Query2Doc-
enhanced pipeline on HotpotQA and AmbigNQ.
search boundary to enhance Recall, RRPO opti-
mizes the context utility to enhance Precision. Con-
sequently, our method proves essential even when
initial retrieval quality is already high, serving as
a critical refinement stage in sophisticated RAG
pipelines.
The framework maintains high label ef-
ficiency even when distilled from smaller,
lightweight supervisors.Finally, we examine
the cost-efficiency of our approach by training a
reranker using a smaller, potentially “noisier” su-
pervisor (Qwen2.5-3B), denoted as RRPO-3B. Ta-
ble 5 reports its performance (with full details in
Appendix F). Remarkably, RRPO-3B delivers con-
sistent improvements. This resilience to supervisor
noise suggests that the core signals for document
utility are prominent enough to be captured even
by lightweight models. This finding highlights the
label efficiency of our approach, demonstrating that
a robust reranking policy can be distilled without
relying on computationally expensive, large-scaleLLM Reader Methods EM F1
Qwen2.5-3B+ GTE (Baseline) 22.59 30.69
+ GTE (RRPO-3B)23.36 31.56
Table 5: Performance of RRPO trained with a smaller
LLM Reader (Qwen2.5-3B).
supervisors.
5 Conclusion
In this work, we propose RRPO to address the
fundamental misalignment between static retrieval
metrics and the dynamic needs of LLM readers
in RAG systems. By formulating reranking as a
sequential decision-making process anchored by a
deterministic baseline, we successfully stabilize the
reinforcement learning training, eliminating the de-
pendency on expensive human annotations or unsta-
ble critic networks. Our extensive experimental re-
sults offer a critical insight: optimizing for context
utility yields significantly better RAG performance
than merely optimizing for semantic relevance. Be-
yond performance gains, RRPO demonstrates re-
markable robustness—it generalizes across diverse
LLM readers (including proprietary models like
GPT-4o) and distills effective policies even from
smaller, noisy supervisors. This establishes RRPO
not just as a performance booster, but as a scalable,
cost-efficient paradigm for constructing task-aware
RAG systems.

Limitations
As a reranking model, RRPO operates on a can-
didate set of documents provided by an initial re-
trieval module. If the initial retriever fails to re-
trieve a sufficient number of relevant documents
(i.e., low recall in the top- Ncandidates), the ability
of RRPO to improve the final RAG performance
will be inherently limited, regardless of how well it
reorders the provided candidates.
Ethics Statement
This work does not pose any ethical issues. All
the data and models used in this paper are publicly
available and are used under following licenses:
Creative Commons BY 4.0 License, MIT License,
Apache license 2.0.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Shijie Chen, Bernal Jiménez Gutiérrez, and Yu Su.
2024. Attention in large language models yields
efficient zero-shot re-rankers.arXiv preprint
arXiv:2410.02642.
Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma,
Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming
Yang, and Jiaxin Mao. 2025. Improving retrieval-
augmented generation through multi-agent reinforce-
ment learning.arXiv preprint arXiv:2501.15228.
Sukmin Cho, Soyeong Jeong, Jeong yeon Seo, and
Jong C Park. 2023. Discrete prompt optimization
via constrained generation for zero-shot re-ranker. In
Findings of the Association for Computational Lin-
guistics: ACL 2023, pages 960–971.
Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen
Wang, Ji-Rong Wen, and Zhicheng Dou. 2025. Un-
derstand what llm needs: Dual preference alignment
for retrieval-augmented generation. InProceedings
of the ACM on Web Conference 2025, pages 4206–
4225.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2:1.Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and
Jie Zhou. 2025. Deeprag: Thinking to retrieval step
by step for large language models.arXiv preprint
arXiv:2502.01142.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps.arXiv preprint arXiv:2011.01060.
Gautier Izacard and Edouard Grave. 2020. Leverag-
ing passage retrieval with generative models for
open domain question answering.arXiv preprint
arXiv:2007.01282.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity.arXiv preprint
arXiv:2403.14403.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of hal-
lucination in natural language generation.ACM com-
puting surveys, 55(12):1–38.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Jina AI. 2024. jina-reranker-v2-base-
multilingual. https://jina.ai/models/
jina-reranker-v2-base-multilingual . Ac-
cessed: 2024-06-25.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles.
Dahyun Lee, Yongrae Jo, Haeju Park, and Moontae
Lee. 2025. Shifting from ranking to set selection
for retrieval augmented generation. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 17606–17619, Vienna, Austria. Association
for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Richard James, Pedro Rodriguez, Ja-
cob Kahn, Gergely Szilvasy, Mike Lewis, and 1 oth-
ers. 2023. Ra-dit: Retrieval-augmented dual instruc-
tion tuning. InThe Twelfth International Conference
on Learning Representations.
Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Chankyu
Lee, Mohammad Shoeybi, and Bryan Catanzaro.
2024. Chatqa: Surpassing gpt-4 on conversational qa
and rag.Advances in Neural Information Processing
Systems, 37:15416–15459.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 5303–5315, Singa-
pore. Association for Computational Linguistics.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. InProceedings of the 47th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval, pages 2421–
2425.
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and
Luke Zettlemoyer. 2020. Ambigqa: Answering
ambiguous open-domain questions.arXiv preprint
arXiv:2004.10645.
Niklas Muennighoff, SU Hongjin, Liang Wang, Nan
Yang, Furu Wei, Tao Yu, Amanpreet Singh, and
Douwe Kiela. 2024. Generative representational in-
struction tuning. InICLR 2024 Workshop: How Far
Are We From AGI.
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu,
Long Ouyang, Christina Kim, Christopher Hesse,Shantanu Jain, Vineet Kosaraju, William Saunders,
and 1 others. 2021. Webgpt: Browser-assisted
question-answering with human feedback.arXiv
preprint arXiv:2112.09332.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with bert.arXiv preprint
arXiv:1901.04085.
Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and
Jimmy Lin. 2019. Multi-stage document ranking
with bert.arXiv preprint arXiv:1910.14424.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, and 1
others. 2022. Training language models to follow in-
structions with human feedback.Advances in neural
information processing systems, 35:27730–27744.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023. Rankzephyr: Effective and robust zero-
shot listwise reranking is a breeze!arXiv preprint
arXiv:2312.02724.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.
Revanth Gangi Reddy, JaeHyeok Doo, Yifei Xu,
Md Arafat Sultan, Deevya Swain, Avirup Sil, and
Heng Ji. 2024. First: Faster improved listwise
reranking with single token decoding.arXiv preprint
arXiv:2406.15657.
Stephen Robertson and Hugo Zaragoza. 2009. The prob-
abilistic relevance framework: Bm25 and beyond.
Foundations and Trends® in Information Retrieval,
3(4):333–389.
John Schulman, Philipp Moritz, Sergey Levine, Michael
Jordan, and Pieter Abbeel. 2015. High-dimensional
continuous control using generalized advantage esti-
mation.arXiv preprint arXiv:1506.02438.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Yang Wu, and 1 others. 2024.
Deepseekmath: Pushing the limits of mathematical
reasoning in open language models.arXiv preprint
arXiv:2402.03300.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval augmentation
reduces hallucination in conversation.arXiv preprint
arXiv:2104.07567.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. DRAGIN: Dynamic retrieval
augmented generation based on the real-time informa-
tion needs of large language models. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 12991–13013, Bangkok, Thailand. Association
for Computational Linguistics.

Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, and Jiawei
Han. 2025. Dynamicrag: Leveraging outputs of large
language model as feedback for dynamic rerank-
ing in retrieval-augmented generation.Preprint,
arXiv:2505.07233.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is chatgpt good at search?
investigating large language models as re-ranking
agents.arXiv preprint arXiv:2304.09542.
Qwen Team. 2024. Qwen2 technical report.arXiv
preprint arXiv:2407.10671, 2.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022a. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions.arXiv
preprint arXiv:2212.10509.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022b. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics, 10:539–554.
Liang Wang, Nan Yang, and Furu Wei. 2023a.
Query2doc: Query expansion with large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9414–9423, Singapore. Association for Com-
putational Linguistics.
Xinyuan Wang, Chenxi Li, Zhen Wang, Fan Bai,
Haotian Luo, Jiayou Zhang, Nebojsa Jojic, Eric P
Xing, and Zhiting Hu. 2023b. Promptagent:
Strategic planning with language models enables
expert-level prompt optimization.arXiv preprint
arXiv:2310.16427.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.Preprint,
arXiv:2309.07597.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. Re-
comp: Improving retrieval-augmented lms with con-
text compression and selective augmentation. InThe
Twelfth International Conference on Learning Repre-
sentations.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations (ICLR).
Xiaowei Yuan, Zhao Yang, Yequan Wang, Jun Zhao,
and Kang Liu. 2024. Improving zero-shot llm re-
ranker with risk minimization. InProceedings of the2024 Conference on Empirical Methods in Natural
Language Processing, pages 17967–17983.
Ye Yuan, Mohammad Amin Shabani, and Siqi Liu. 2025.
Embedding-based context-aware reranker.arXiv
preprint arXiv:2510.13329.
Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, and
Aishwarya Agrawal. 2025a. Rearank: Reasoning
re-ranking agent via reinforcement learning.arXiv
preprint arXiv:2505.20046.
Ruizhe Zhang, Yongxin Xu, Yuzhen Xiao, Runchuan
Zhu, Xinke Jiang, Xu Chu, Junfeng Zhao, and Yasha
Wang. 2025b. Knowpo: Knowledge-aware prefer-
ence optimization for controllable knowledge selec-
tion in retrieval-augmented language models.Pro-
ceedings of the AAAI Conference on Artificial Intelli-
gence, 39(24):25895–25903.
Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
Pengjun Xie, Fei Huang, and 1 others. 2024. mgte:
Generalized long-context text representation and
reranking models for multilingual text retrieval. In
Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing: Industry
Track, pages 1393–1412.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025c. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Yutao Zhu, Peitian Zhang, Chenghao Zhang, Yifei Chen,
Binyu Xie, Zheng Liu, Ji-Rong Wen, and Zhicheng
Dou. 2024. Inters: unlocking the power of large
language models in search with instruction tuning.
arXiv preprint arXiv:2401.06532.

A Pseudo Codes of RRPO
Here we present the pseudo codes for our RRPO
algorithm.
Algorithm 1:RRPO: RL for Reranker Optimiza-
tion
1:Input:Initial retriever, Reader LLM, ground
truth answers ‘ans’, instruction ‘INST’, eval
metricR lm.
2:Hyperparameters: k(docs to select), α
(learning rate), PPO params ( ϵ, β), GAE
params ( γ, λ),Miter(iterations), Eppo(PPO
epochs),B size(batch size).
3:Initialize policy network (reranker) fθ; Fixed
reference policyπ ref(fromf ref).
4:Initialize experience bufferD.
5:foriterationm= 1toM iterdo
6:Clear bufferD.
7: foreach training query qin a batch of
Bsizedo
8:D cand←Retriever(q) ;s1←D cand;
Lselected ←[].
9:τ query_data ←[].▷Temporary storage
for the current query’s trajectory
1. Agent Rollout & Reward Calculation
10:fort= 1tokdo
11: Select document dctfrom stusing
policyπ θ(·|st;q)(derived fromf θ, Eq. (3)).
12:L selected ←L selected ∪ {d ct};
st+1←s t\ {d ct}.
13:response t ←
Reader LLM(INST, q, L selected ).
14:r t←R lm(ans q,responset)
(Eq. (7)).
15: Add (st, dct, rt,logπ θ(dct|st))to
τquery_data .
2. Learning-free Deterministic Baseline
Calculation
16:fort= 1tokdo▷For each steptin
the agent’s trajectory
17:L′
sel_for_V t←docs selected by t-
step greedy rollout of πrefonDcand forq
(Eq. (11)).
18:V t ←
Rlm(ans q,Reader LLM(INST, q, L′
sel_for_V t))
(This isV(s t)in Eq. (13)).
19: Augment t-th step data in
τquery_data with its correspondingV t.
3. Advantage Estimation (GAE)
20: Compute GAE advantages A1, . . . , A k
forτ query_data usingr t, Vt, γ, λ(Eqs. (9)-(10)21: Store trajectory (now including At)
fromτ query_data intoD.
▷End of batch collection
4. Advantage Normalization
22: Normalize all advantages AtinDto get
ˆAt(Eq. (14), e.g., using mean/std overD).
5. Policy Optimization (PPO)
23: Update policy parameters θforEppo
epochs by optimizing PPO objective (Eq. (8))
using mini-batches from D(which contains
st, dct,ˆAt,logπ θ(dct|st)old, and usesπ ref).
24:Output:Optimized rerankerf θ.
B More Implementation Details
B.1 Training Details
On the HotpotQA dataset, we trained for 2 epochs
with an initial learning rate of 2e−6 . For RL train-
ing interactions on this dataset, we used ktrain = 3
(i.e., the LLM Reader received 3 documents per in-
teraction). On the AmbigNQ dataset, we trained for
4 epochs with an initial learning rate of 1.6e−6 .
For RL training interactions on this dataset, we
usedktrain = 5. We adjusted some PPO hyperpa-
rameters based on the characteristics of the RRPO
task. Specifically, in GAE, the discount factor
γ= 0.99 andλ= 0.95 . For the PPO clipping
objective, ϵ= 0.2 . In our objective function, the
KL divergence penalty coefficient βwas set to 0.1.
We use AdamW as our Optimizer. We use 4 * RTX
3090 for training reranker, and 2 * A6000 for de-
ploying LLM Reader using vllm due to accelerate
LLM’s inference performance. We use torch AMP
and Flash Attention for training. We set batch size
to 32 using gradient accumulation.
B.2 Discussion on Experimental Fairness and
Baseline Selection
To ensure the rigor of our experimental conclusions,
we strictly control the variables and carefully select
baselines based on data availability.
Controlled Evaluation Protocol: As emphasized
in the main text, strict variable control is main-
tained across all experiments. For every query,
both the baseline (e.g., gte-reranker) and our opti-
mized model (RRPO) rank the exact same pool of
candidate documents (top- N) retrieved by BM25.
All downstream parameters, including the number
of selected documents ( k) and the LLM Reader
configuration, are kept identical. This ensures that
any performance gain is exclusively attributable to

the reordering policy learned by RRPO.
Why SFT is Not Included as a Baseline: A direct
comparison with Supervised Fine-Tuning (SFT) is
methodologically challenging in our specific open-
domain setting for two reasons:
Absence of Gold Labels: SFT relies on high-
quality, passage-level relevance annotations (e.g.,
query-passage pairs labeled as positive/negative).
However, the datasets used in our experiments (Hot-
potQA and AmbigNQ within the DPR retrieval
setup) provide ground-truth answers but do not in-
herently provide gold-standard ranking labels for
the retrieved top-100 candidates.
Domain Mismatch Risks: While one could fine-
tune a reranker on external labeled corpora like MS
MARCO, evaluating such a model on our target
datasets would introduce severe domain distribu-
tion shifts, rendering the comparison unfair.
Therefore, comparing against the pre-trained
base reranker provides the most direct assessment
of RRPO’s ability to align retrieval with generation
using only the available reader feedback, effec-
tively bypassing the data bottleneck of SFT.
C Prompt Template
We utilize this prompt template for training and
evaluation on both HotpotQA and AmbigNQ
dataset.
System:
You are a highly capable assistant who can
read the provided passages and accurately
answer the question below. You only need
to answer the information required for the
question within a few words, without any
additional content. For questions requiring
a ‘yes’ or ‘no’ response, answer solely with
‘Yes’ or ‘No’, without providing any addi-
tional explanation.
User:
passage 1: Steven Peter’s father is Steven
Pawl.
User:
passage 2: Steven Peter’s father graduated
from Harvard University.
User:
Based on these passages, answer the ques-
tion. question: Where did Steven Pawl grad-
uate from?Assistant:
Harvard University
User:
passage 1: {{passage_1}}
User:
passage 2: {{passage_2}}
User:
passage 3: {{passage_3}}
...(the number of passages is determined by
k.)
User:
Based on these passages, answer the ques-
tion. question: {{query}}
D Generalization with Different Prompt
Templates
Previous sections demonstrated the generalization
of our method across different LLM Readers. In
this subsection, we investigate a critical question:
does the reranker, trained via reinforcement learn-
ing with a specific prompt template, become over-
fitted to that template? To assess robustness, we
evaluate performance using two additional tem-
plates (Template A and Template B) that differ
structurally and stylistically from the default tem-
plate used in the main experiments.
Template A redefines the system role as an “in-
formation extraction AI” and groups context under
“Text Snippet” headers within a single user turn.
Template B adopts an “Objective-Constraints” for-
mat in the system message and positions the ques-
tion before the “Supporting Documents.” Both tem-
plates modify the phrasing for conciseness while
strictly enforcing Yes/No constraints to maintain
consistency with the Qwen2.5-7B model used as
the LLM Reader.
The specific structures of Prompt Template A
and B are shown below.
System:
You are an information extraction AI. Your
task is to answer the question using the pro-
vided text within a few words. Be concise.
For yes/no questions, respond with ‘Yes’ or
‘No’ and nothing more.
User:

Here is the text you should use:
Text Snippet 1: {{passage_1}}
—
Text Snippet 2: {{passage_2}}
—
...(passages continue based onk)
—
Based on the text snippets above, what is the
answer to this question: {{query}}
System:
Objective: Answer the user’s question
within a few words. Constraints: Use the
information given in the ‘Supporting Docu-
ments’. For ‘Yes’ or ‘No’ questions, provide
only that word as the answer. Be brief.
User:
Question: {{query}}
Supporting Documents:
Document [1]: {{passage_1}}
Document [2]: {{passage_2}}
...(passages continue based onk)
Please provide the answer found in the doc-
uments.
The experimental results are presented in Table 6,
with default template results reproduced from Ta-
ble 1 for comparison.
Template MethodsHotpotQA
EM F1
DefaultBM25 24.23 35.42
+ GTE 28.24 41.23
+ GTE (RRPO)30.03 43.22
ABM25 23.08 30.80
+ GTE 28.39 38.15
+ GTE (RRPO)29.89 39.98
BBM25 24.17 32.17
+ GTE 29.40 39.37
+ GTE (RRPO)31.07 41.32
Table 6: Generalization results on different prompt tem-
plates on HotpotQA.
As shown in Table 6, RRPO demonstrates strong
robustness to variations in prompt templates, con-
sistently outperforming both Naive RAG (BM25)
and the standard gte reranker across all formats.
Notably, while the baseline performance fluctuates
due to the LLM’s sensitivity to prompt phrasingMethodstop-1 top-3
EM F1 EM F1
Naive RAG (BM25) 23.01 28.47 25.64 30.20
+ gte reranker 22.87 28.9926.4631.36
+ gte reranker (RRPO)23.06 29.5026.3631.47
Table 7: Performance of RRPO on 2WikiMultiHopQA.
Methodstop-1 top-3
EM F1 EM F1
BM25 + gte reranker 6.00 16.33 10.22 19.22
BM25 + gte reranker (RRPO)6.21 16.35 10.34 19.84
Table 8: Performance of RRPO on Harder MuSiQue.
(e.g., Naive RAG EM scores range from 23.08 to
24.23), RRPO delivers a stable and significant per-
formance boost in every scenario.
This consistency confirms that the reranker has
learned to identify intrinsically valuable documents
rather than overfitting to the specific linguistic pat-
terns of the training prompt. Such generalization
capability is critical for real-world RAG systems,
validating RRPO as a plug-and-play component
compatible with diverse user-defined prompt struc-
tures.
E Experimental Results on More
Challenging Multi-hop QA Datasets
More experimental results on 2WikiMultiHopQA
are reported in Table 7. The results further demon-
strate the effectiveness of RRPO on this dataset.
For MuSiQue dataset, due to the task’s complex-
ity and the lower recall of the BM25 retriever, the
overall performance baseline is not high.
Despite the limited gains, RRPO still yields pos-
itive improvements. This demonstrates that even
in challenging scenarios with low-quality initial
retrieval, RRPO’s alignment with the LLM allows
it to filter more valuable information from noisy
candidates. This shows the advantage of our single-
step optimization even within a complex multi-hop
context.
F Experimental Results on Smaller LLM
Readers
In our main experiments, we choose Qwen2.5-7B
as the LLM Reader in RAG pipeline. We now ex-
plore whether the smaller LLM Readers can affect
the results due to its more noisy reward feedback.

LLM Reader MethodsHotpotQA
EM F1
Qwen2.5-3B+ GTE 22.59 30.69
+ GTE (RRPO-3B) 23.36 31.56
+ GTE (RRPO-7B)23.66 31.94
Qwen2.5-7B+ GTE 28.24 41.23
+ GTE (RRPO-3B) 29.47 42.65
+ GTE (RRPO-7B)30.03 43.22
Qwen2.5-14B+ GTE 31.83 44.62
+ GTE (RRPO-3B) 33.09 45.99
+ GTE (RRPO-7B)33.26 46.37
Table 9: Performance of RRPO with smaller LLM Read-
ers.
Methodstop-1 top-3
EM F1 EM F1
BM25 + GTE 35.51 48.55 40.26 51.36
BM25 + GTE (Listwise Bandit) 36.21 49.06 39.81 50.95
BM25 + GTE (Standard PPO) 35.91 48.97 39.66 51.34
BM25 + GTE (RRPO)36.56 49.67 41.11 52.18
Table 10: Training on alternative RL architectures on
AmbigNQ.
We choose Qwen2.5-3B here, and evaluate our re-
sults on Qwen2.5-3B, 7B and 14B.
As the results show, the reranker trained under
the guidance of Qwen2.5-3B remains effective,
showing a clear performance lift over the base-
line, although it is slightly less performant than the
reranker trained with the 7B model.
G Analysis of Alternative RL
Architectures
To validate the necessity of our sequential formu-
lation and reference-anchored baseline, we com-
pare RRPO against two standard RL variants on
the AmbigNQ dataset: (1) Listwise Bandit, which
models the selection of top- kdocuments as a sin-
gle action, and (2) Standard PPO, which utilizes a
parameterized value network (Critic) instead of our
deterministic reference baseline.
As the results in Table 10 show, both alternative
methods achieve improvements over the static base-
line (BM25 + GTE) in Top-1 metrics, confirming
the validity of incorporating LLM feedback. How-
ever, their performance on Top-3 metrics is incon-
sistent, often stagnating compared to the baseline.
This suggests that standard methods struggle to op-
timize the utility of the entire list (ranking) while
successfully promoting single relevant documents
(hitting), likely due to the difficulty of credit assign-ment in single-action settings or the high variance
of parametric critics. In contrast, RRPO delivers
consistent gains across both Top-1 and Top-3 met-
rics, demonstrating that our sequential step-wise
reward mechanism effectively guides the model to
construct a high-utility set of documents rather than
merely identifying a single hit.
H Statistical Significance Tests
NotesHotpotQA AmbigNQ
EM F1 EM F1
Trained Results Avg. 29.89 43.09 36.61 49.59
Trained Results Std. 0.09 0.12 0.15 0.15
Table 11: Statistical Significance Tests.
To rigorously validate that the performance im-
provements achieved by our RRPO framework are
not a result of random chance, we conducted sta-
tistical significance tests. We employed a paired
t-test to compare the per-query performance of our
best-performing model, “+ gte reranker (RRPO)”,
against the primary baseline, “+ gte reranker”. The
test was performed on the F1 scores obtained for
each query in the validation sets of both HotpotQA
and AmbigNQ.
The null hypothesis ( H0) for our test is that there
is no statistically significant difference between the
mean F1 scores of the two models. We aim to reject
this hypothesis by demonstrating a sufficiently low
p-value.
The results are illustrated in Table 11. Our anal-
ysis yielded that On HotpotQA and AmbigNQ
dataset, the paired t-test resulted in a p-value of
p <0.01 . In both cases, the extremely low p-
values are well below the standard significance
level of α= 0.05 . Therefore, we reject the null hy-
pothesis. This provides strong statistical evidence
that the gains observed from applying RRPO are
consistent and meaningful, confirming the effec-
tiveness of our proposed training paradigm.
I Training Acceleration
To enhance training efficiency and mitigate the
computational overhead of the RL-based RAG
pipeline, we implement a comprehensive set of
optimization strategies:
•Kernel Acceleration:We leverage Flash
Attention and Automatic Mixed Precision
(AMP) to optimize the reranker, and deploy

the LLM reader using vLLM (Kwon et al.,
2023) to maximize inference throughput.
•Retrieval Caching:To eliminate redun-
dant retrieval operations, we cache the top-
N documents obtained from the initial re-
triever, reusing them across subsequent train-
ing epochs.
•LLM Response Replay:By enforcing deter-
ministic LLM generation (temperature=0), we
construct a replay buffer to store and reuse his-
torical responses. This mechanism is critical,
as it reduces the required LLM inference calls
by nearly 50% in later training stages.
•Distributed Training Optimization:We em-
ploy gradient accumulation to increase the
effective batch size and fine-tune communi-
cation schedules for improved parallel effi-
ciency.
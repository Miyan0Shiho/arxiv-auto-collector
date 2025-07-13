# FrugalRAG: Learning to retrieve and reason for multi-hop QA

**Authors**: Abhinav Java, Srivathsan Koundinyan, Nagarajan Natarajan, Amit Sharma

**Published**: 2025-07-10 11:02:13

**PDF URL**: [http://arxiv.org/pdf/2507.07634v1](http://arxiv.org/pdf/2507.07634v1)

## Abstract
We consider the problem of answering complex questions, given access to a
large unstructured document corpus. The de facto approach to solving the
problem is to leverage language models that (iteratively) retrieve and reason
through the retrieved documents, until the model has sufficient information to
generate an answer. Attempts at improving this approach focus on
retrieval-augmented generation (RAG) metrics such as accuracy and recall and
can be categorized into two types: (a) fine-tuning on large question answering
(QA) datasets augmented with chain-of-thought traces, and (b) leveraging
RL-based fine-tuning techniques that rely on question-document relevance
signals. However, efficiency in the number of retrieval searches is an equally
important metric, which has received less attention. In this work, we show
that: (1) Large-scale fine-tuning is not needed to improve RAG metrics,
contrary to popular claims in recent literature. Specifically, a standard ReAct
pipeline with improved prompts can outperform state-of-the-art methods on
benchmarks such as HotPotQA. (2) Supervised and RL-based fine-tuning can help
RAG from the perspective of frugality, i.e., the latency due to number of
searches at inference time. For example, we show that we can achieve
competitive RAG metrics at nearly half the cost (in terms of number of
searches) on popular RAG benchmarks, using the same base model, and at a small
training cost (1000 examples).

## Full Text


<!-- PDF content starts -->

FrugalRAG: Learning to retrieve and reason for
multi-hop QA
Abhinav Java
Microsoft Research
India
java.abhinav99@gmail.comSrivathsan Koundinyan
Microsoft Research
India
skoundinyan@microsoft.comNagarajan Natarajan
Microsoft Research
India
nagarajn@microsoft.com
Amit Sharma
Microsoft Research
India
amshar@microsoft.com
Abstract
We consider the problem of answering complex questions, given access to a large
unstructured document corpus. The de facto approach to solving the problem is to
leverage language models that (iteratively) retrieve and reason through the retrieved
documents, until the model has sufficient information to generate an answer. At-
tempts at improving this approach focus on retrieval-augmented generation (RAG)
metrics such as accuracy and recall and can be categorized into two types: (a)
fine-tuning on large question answering (QA) datasets augmented with chain-of-
thought traces, and (b) leveraging RL-based fine-tuning techniques that rely on
question-document relevance signals. However, efficiency in the number of re-
trieval searches is an equally important metric, which has received less attention. In
this work, we show that: (1) Large-scale fine-tuning is not needed to improve RAG
metrics, contrary to popular claims in recent literature. Specifically, a standard
ReAct pipeline with improved prompts can outperform state-of-the-art methods
on benchmarks such as HotPotQA. (2) Supervised and RL-based fine-tuning can
help RAG from the perspective of frugality, i.e., the latency due to number of
searches at inference time. For example, we show that we can achieve competitive
RAG metrics at nearly half the cost (in terms of number of searches) on popular
RAG benchmarks, using the same base model, and at a small training cost (1000
examples).
1 Introduction
We study the problem of answering questions, such as “ Can a microwave melt Toyota Prius battery ?”,
given access to a large unstructured corpus like Wikipedia. The de facto approach to solving the
problem is to use language models (LMs) coupled with the ability to retrieve relevant documents
(e.g., Wiki passages) against queries, i.e., the retrieval-augmented generation (RAG) paradigm [Jeong
et al., 2024, Jiang et al., 2023, Chan et al., 2024, Asai et al., 2023]. However, answering complex
questions often requires multi-hop reasoning and retrieval, i.e., the LM has to iteratively decompose
the user utterance into sub-queries or search phrases (e.g., “ melting point of Toyota Prius battery ”),
retrieve documents relevant to the sub-queries, and reason through the retrieved documents to issue
further sub-queries, until the LM is able to generate an answer for the original query.
Most of recent work in this space focus exclusively on the accuracy of generated answers, using (a)
supervised fine-tuning (SFT) techniques on large QA training datasets [Asai et al., 2023, Chan et al.,
Preprint. Under review.arXiv:2507.07634v1  [cs.CL]  10 Jul 2025

2024, Hsu et al., 2024], or (b) reinforcement-learning (RL) based techniques such as GRPO [Jin et al.,
2025] and DPO [Hsu et al., 2024]. In either case, they rely on tens or even hundreds of thousands
of training data points, either in the form of <question, answer> demonstrations or as <question,
documents> relevance signals. For instance, the most recent work SearchR1 [Jin et al., 2025] applies
GRPO to fine-tune the Qwen-2.5-7B-Instruct [Yang et al., 2024] model using more than 100000
training examples on the popular HotPotQA [Yang et al., 2018] dataset. Similarly, another recent
RAG framework called LeReT [Hsu et al., 2024] uses more than 90000 training examples on the
same dataset, using Llama3.1-8B [Grattafiori et al., 2024]. However, in real-world QA settings,
ground-truth labelled examples are difficult to obtain and latency of the system is critical for user
experience. In this work, we challenge the claims in the recent literature in multiple ways:
(1)Efficiency, i.e., number of hops or searches needed to accurately answer, is equally important;
(2)The oft-overlooked simple baseline strategy of ReAct for iterative retrieval and reasoning [Yao
et al., 2023, Shao et al., 2023], with optimized few-shot prompting, already is quite competitive
to several recent techniques; the prompt optimization needs tens of examples, in stark contrast to
aforementioned RAG techniques like Self-RAG and SearchR1 that need several orders of magnitude
more number of examples, and
(3)We can leverage RL-based techniques like GRPO to further improve such strong baselines, from
the perspective of efficiency where they often underperform, using just 1000 training examples .
Our solution, FRUGAL RAG , is a two-stage framework that removes the need for large-scale labels
while still achieving effective and efficient inference-time search. In the first stage , the model is
trained to maximize evidence coverage by generating diverse and informative search queries across
multiple hops. In the second stage , we post-train the model to decide when to stop retrieving and
generate an answer. This decision is modeled explicitly, allowing the model to weigh the cost of
further retrievals against the confidence in the retrieved evidence.
Optimizing for coverage and efficiency in a single stage leads to unstable training; we find that models
either over-retrieve or stop too early. Our key insight is that learning when to stop is more naturally
learned through reinforcement learning (RL) signals, whereas better coverage can be obtained by
repeatedly issuing high quality search queries using frameworks such as ReAct. By separating the
exploration stage from decision making, FRUGAL RAG better aligns the learning signal with the
multi-hop RAG task resulting in high quality and efficient retrievals.
We evaluate FRUGAL RAG on standard benchmarks such as HotPotQA [Yang et al., 2018], 2WikiMul-
tiHopQA [Ho et al., 2020] and MuSiQue [Trivedi et al., 2022b], using both document retrieval metrics
such as recall and answer quality metrics. Compared to baselines, we show that FRUGAL RAG obtains
the highest document recall and answer quality while incurring a low number of search queries per
question. In particular, on HotPotQA, FRUGAL RAG using a 3B model obtains higher document
recall with only two searches per question compared to methods such as LeReT and SearchR1 that
are based on 8B and 7B models and include finetuning on thousands of examples.
2 Related Work
Which metric to optimize. Multi-hop QA involves two sub-tasks: retrieving relevant documents,
and then answering the question based on the documents. Some methods report document retrieval-
specific metrics such as recall [Hsu et al., 2024] whereas others report final answer metrics such as
exact match [Jin et al., 2025]. Typically, a model is trained to optimize a particular metric (such as
recall) and also evaluated on the same metric. For robustness, in this work we train on the recall
metric and test on all metrics, including final answer metrics.
Prompting-based RAG approaches. With the recent advancements in the capabilities of large
API-based LMs, some works explored prompting to call external search/retrievers at inference.
Toolformer[Schick et al., 2023] uses a self-supervised objective to train an external model that
decides to call tools (like Bing and Google search engines). ReAct [Yao et al., 2023] is another
powerful prompting technique that allows the model to structure its outputs as thoughts, actions
and observations, yielding significant improvements in the ability of LLMs to interact with external
environments. Trivedi et al. [2022a] proposed IRCoT, another prompting strategy that alternates
between chain-of-thought [Wei et al., 2022] steps and gathering evidence through retrievals. By using
the intermediate traces, the IRCoT is able to decide what to retrieve by issuing the right search queries.
Iter-RetGen [Shao et al., 2023] improves evidence gathering in multi-hop scenarios by combining
2

retrieval and generation iteratively, such that a model’s response is incorporated in the reasoning trace.
However, both IRCoT [Trivedi et al., 2022a] and Iter-RetGen [Shao et al., 2023] rely on a fixed or
predefined number of retrieval loops at inference, offering limited control over latency.
Finetuning-based techniques. A prevalent method for multi-hop QA using small LMs is supervised
finetuning using reasoning traces from a large LM such as GPT-4 [Asai et al., 2023, Chan et al., 2024].
Other methods are trained to predict the next query to be retrieved [Chan et al., 2024]. Methods that
scale the test-time compute that infer using multiple trajectories have also been proposed [Wang et al.,
2025]. Recently, reinforcement learning-based techniques have been proposed that develop a reward
based on outputting the ground-truth answer [Jin et al., 2025]. However, none of the techniques
focus on efficiency of the solution. In fact, in Search-R1, the goal of RL is to increase the number of
searches. Instead, we use RL to decrease the average number of searches done by our model.
1.Traditional RAG approaches. Early work in grounding generation with real world docu-
ments focused on end-to-end differentiable encoder-decoder pipeline REALM [Guu et al.,
2020], which augments Masked-Language Modeling (MLM) with a latent retriever model,
backpropagating through retrieval to learn both retriever and generator jointly. However,
this approach incurs significant computational cost and has only been shown to work with
relatively smaller models like T5 [Raffel et al., 2020]. Building on this, Lewis et al. [2020]
proposed a general finetuning strategy, RAG-Token which demonstrated that join-training
outperforms fixed dense retrieval and BM25.
2.RL-based Retrieval Augmented Generation. Recently, framing search query as an RL
problem has received attention. LeReT [Hsu et al., 2024] performs preference optimization
using diverse few shot prompts leveraging hundred-thousands of ground truth annotated
documents. However, LeReT utilizes a fixed amount of compute per instance during
inference and cannot be readily generalized to variable-hop scenarios. Similarly, concurrent
works, Jin et al. [2025] and Chen et al. [2025] propose end-to-end RL-based optimization
that only leverages the final answer annotation. These methods show that RL can effectively
be used to teach the search query generator model to issue more search queries for multi-hop
problems without considering latency . Our two-stage RL framework, by contrast, first
explores without RL to maximize recall and then learns to stop at test time using RL.
3 FrugalRAG: Two-stage training framework
We describe FRUGAL RAG , a novel framework for enhancing retrieval augmented generation in LMs
by decoupling the evidence exploration stage from answer generation. FRUGAL RAG demonstrates
several key advantages over contemporary RAG approaches – (1) Requires only 1000 annotated
training examples which is a 100 times reduction in dataset size compared to existing works [Jin et al.,
2025, Hsu et al., 2024, Chan et al., 2024], (2) Dynamically adapts test-time compute which results
in low inference time latency and high retrieval recall, unlike existing fixed compute methods [Hsu
et al., 2024].
Setup. LetQdenote a complex user question that requires multiple iterative retrievals to answer.
Letfdenote a language model (LM) that, at each reasoning hop, examines the current context and
determines the next action. At hop h(where 1≤h≤B, andBis the maximum allowed number of
hops), the model fproduces a thought–action-search query triplet (Th, Ah, Sh). The search query
Shis passed to a retriever R(·), which returns a set of documents: Dh=R(Sh)from the document
index, say I.
LetD0denote the initial context—either empty or initialized as R(Q). At hop h, the model has
access to the context:
{Q} ∪h−1[
0{(Dh, Th, Ah, Sh)}
This includes the original query, all previously retrieved documents, and the previously generated
thought, action, search query triplets.
The process continues until the model outputs a special FINISH action, terminating after htermhops
(or at the budget limit B). At this point, a separate generator LM gis invoked to produce the final
answer, conditioned on the original question Qand the full context accumulated up to termination.
3

The central challenge for fis to iteratively construct highly targeted queries Shsuch that the retriever
Rcan surface a minimal yet sufficient set of documents to answer Qwithin the hop budget B.
FRUGAL RAG requires access only to ground truth documents Yduring training. It does notrequire
supervision in the form of final answer annotations. These documents are used to provide fine-
grained feedback signals. At inference time, FRUGAL RAG relies solely on the input question Q, the
document index I, and a trained retriever R.
Our key observation is that, with sufficient test-time compute, even base models can generate multiple,
diverse search queries to address a question—following the ReAct paradigm (e.g., see Sec 4, Table ??).
Therefore, the goal of our work is not to scale up test-time computation, as argued in prior work [Jin
et al., 2025, Chen et al., 2025], but rather to adaptively control it based on the difficulty of each
question.
In the following subsections, we describe the two stages of our learning algorithm:
•Stage 1 (Sec. 3.1): Generating a base policy that maximizes evidence coverage through
exploration
•Stage 2 (Sec. 3.2): Finetuning this policy with reinforcement learning to control test-time
compute
3.1 Stage 1: Evidence Coverage Maximization (Explore)
Gathering evidence plays a crucial role in answering multi-hop questions, which often require iterative
retrieval and reasoning across multiple sources. Drawing on recent advances in test-time scaling,
we observe that we can boost evidence coverage (i.e., recall) simply by letting the model fissue
multiple search queries Shat inference time. This approach sidesteps the need for massive supervised
fine-tuning—instead, it harnesses the model’s own generated rollouts to gather, and then integrate
additional information. In the next section, we describe how we construct our training data and design
our fine-tuning protocol to fully leverage this capability.
Training Dataset Generation. We choose ReAct [Yao et al., 2023] for generating rollouts. Here,
a rollout is a set of outputs generated by the model f. In the standard ReAct setup, an off-the-
shelf model fgenerates a thought-action pair (ThAh)at each hop h∈[1, B], where Ahis either
a call to the retriever RorFINISH indicating the end of rollout. At each hop h, we generate
samples {(T1
h, A1
h, S1
h). . .(Tn
h, An
h, Sn
h)}using nbootstrapped prompts [Khattab et al., 2023] (See
Appendix A). For each search query Si
h, i∈[1, n]we retrieve corresponding documents Di
h=
R(Si
h), then discard any documents already present in the context. We then compute recall against
ground-truth labels and add the sample ithat achieves the highest recall to the context for the next
hoph+ 1. This dataset generation strategy is simple and easily parallelizable. We conduct two
separate runs – the standard ReAct framework where fis allowed to generate FINISH , and the second
where fcan only call the retriever. Although the former is more efficient and finishes before B search
hops, we observe that the latter yields a significantly higher overall recall owing to a greater number
of retrievals. Unlike previous work [Chan et al., 2024, Asai et al., 2023, Hsu et al., 2024, Jin et al.,
2025] that generated orders of magnitude more data, we only generate 1000 examples during this
step.
Supervised “Exploration” Finetuning (FRUGAL RAG -Explore) .Although the base model fwithout
FINISH maximizes exploration, we cannot use it directly for reinforcement learning because it does
not include FINISH . Consequently, during fine-tuning, we sample rollouts from both configuration
described above, 90% without FINISH and 10% with it. We want to use supervised finetuning to
build a strong base-policy for RL, that prioritizes exploration while ensuring that FINISH remains in
the model’s generation distribution. Hence, we finetune the model fto obtain our base policy fS.
At each iteration, the model predicts the next (Th, Ah, Sh)tuple given the rollout which comprises
interleaved thought-action-search tuples and retrieved documents represented as an ordered set
{(D0, T0, A0, S0),(D1, T1, A1, S1). . .(D(h−1), T(h−1), A(h−1), Sh−1)}tillh−1, using standard
cross-entropy error as the objective function. fShas several notable advantages – (1)off-the-shelf
model fdoes not explore. Despite prompt optimization, fis generally over-confident and predicts
the answer without sufficient exploration. (2)removing FINISH results in over-retrievals. We observe
that simply removing FINISH from the ReAct loop yields high recall even with the base model f,
4

however, the model is forced to utilize the full budget for every question and cannot be post-trained
for efficiency as it never generates a rollout with FINISH .
3.2 Stage 2: Controlling test-time compute with RL
Given a finetuned base policy model fS, we propose a strategy that enables the model to generate
extended rollouts only when required. This mechanism is crucial for inference-time efficiency, as it
allows the model to adaptively determine the appropriate rollout length. Since fSgenerally prioritizes
exploration, our only goal is to learn when to sufficient evidence has been gathered, thereby reducing
overall search latency during inference. Below we show how this problem can be formulated as a
reinforcement learning task, as it requires evaluating and comparing different rollouts. However,
unlike recent work using RL for scaling the number of retrievals [Jin et al., 2025], here our focus is to
use RL to reduce the number of retrievals per question.
Reward Design. Our reward function is designed to guide the model toward discovering the optimal
rollout length. To compute the reward, we first generate the complete rollout using the policy fS
and then evaluate it. Let h∗denote the optimal number of retrieval steps (or hops), defined as the
point beyond which further retrievals do not improve the overall recall c. If the model terminates
at a hop hterm> h∗, it incurs a penalty for redundant steps. Conversely, stopping at hterm< h∗
is also penalized to encourage sufficient exploration. This reward structure enables the model to
explore adequately for complex queries while avoiding unnecessary computation for simpler ones.
The reward is defined as:
R=

max
−Rmax,min
log1−∆
∆
, Rmax
,if∆>0∧c≥τ(late stop)
Rmax+α·h∗
B
, if∆ = 0 ∧c≥τ(perfect stop)
max
−Rmax,min
log1−∆
∆
,0
, ifc < τ (early stop)(1)
Here, ∆is the normalized difference (hterm−h)/B, where Bis the maximum hop budget. Rmax
is the upper bound on the reward, and αis a tunable hyperparameter that scales the bonus for
stopping exactly at h∗. This bonus is proportional to h∗/B, thereby assigning higher reward to
correct terminations on more complex (i.e., longer) rollouts. A rollout is considered answerable if its
recall cis greater than or equal to a predefined threshold τ.
Intuitively, the reward function penalizes deviations from the optimal stopping point in proportion
to|∆|. The closer the model stops to h∗, the higher the reward, with the maximum attained when
hterm=h∗.
In addition to R, we define a format reward Rfto enforce adherence to the ReAct-style format. If
the output deviates from the expected format and results in failed retrievals, we assign a reward of
−1; conversely, if the retrievals succeed, the model receives a reward of +1. This reward is averaged
across all hops. The final reward is the mean of the main reward Rand the format reward Rf,
yielding an overall reward in the range [−Rmax−1, Rmax+α+ 1].
Optimal Rollout Length. We define the optimal rollout length h∗as the minimum number of
retrieval steps required to answer a query Qeffectively. Any additional retrievals beyond h∗are
considered redundant. Conversely, if the rollout has not yet gathered sufficient evidence to answer
the query, it should continue retrieving. To balance retrieval efficiency and performance, we use
FRUGAL RAG -Explore as a reference policy, assuming it represents the best achievable performance
within a fixed budget B. If another policy achieves the same recall performance as FRUGAL RAG -
Explore in fewer than Bhops, we set h∗to that lower number of hops and compute the reward using
Eq 1.
Optimization. Motivated by the recent success and memory efficiency of GRPO [Shao et al., 2024],
we adopt it as our optimization algorithm. At each hop h, we sample vtuples {Ti
h, Ai
h, Si
h}v
i=1,
and retrieve their corresponding documents Di
h, and deduplicate. We repeat this until we reach
the maximum budget h=B, collecting sample tuples and documents. For each rollout i, we then
compute a cumulative reward using Eq. 1, and backpropagate through every logit produce by the
5

policy along that rollout. Finally, once fSemits the FINISH token, we mask any further generations.
The detailed steps are provided in Algorithm 1.
4 Experiments
4.1 Experimental Setup
Benchmarks. We conduct evaluation of FRUGAL RAG using three widely adopted Multi-Hop RAG
benchmarks– HotPotQA [Yang et al., 2018], 2WikiMultiHopQA [Ho et al., 2020] (2Wiki), and
MuSiQue [Trivedi et al., 2022b], under their full-wiki setting using a ColBERT-v2 [Santhanam et al.,
2021] retriever index over Wikipedia passages provided in official datasets. These datasets require
models to perform reasoning across multiple documents to arrive at an answer. The HotPotQA bench-
mark comprises of of 7405 development examples requiring supporting evidence from Wikipedia
abstracts. We report the results on the entire dev-set to access the models ability to perform end-to-end
reasoning and retrieval. For the 2WikiMultiHopQA benchmark, we utilize 12576 development
examples, each coupled with its supporting evidence and document title. This evaluates a model’s
ability to hop between structured and unstructured wikipedia text. Finally, for MuSiQue , we test on
its 2405 development examples of 2-4 hop questions derived from a composition of two-hop queries.
We provide details of the datasets and the index in Appendix C.
Metrics. Most prior work focuses on the final answer accuracy [Jin et al., 2025] or document-level
recall [Hsu et al., 2024]. In this work, we evaluate FRUGAL RAG on three key parameters, namely,
goodness of final answer, retrieval recall, and efficiency. To assess the final answer fidelity we report
F1 Score, EM (Exact Match), and Match Score. F1score is the word level harmonic mean of precision
and recall between predicted and ground truth string. EM requires the predicted string to exactly
match the ground truth. Match requires the ground truth to be a substring of the generated answer.
We also evaluate the alignment of supporting evidence with the retrieved documents by reporting
Recall andSupport-F1 [Trivedi et al., 2022b]. Finally, we compute the Latency ofFRUGAL RAG as
the total number of search operations it performs to answer a given query. In Sec 4, we introduce
metric to realize the tradeoff between efficiency and performance (answer/retrieval), computed as the
average performance per search. A detailed overview of metrics is presented in the Appendix B.
Baselines. We evaluate FRUGAL RAG by comparing it against no-retrieval and retrieval-based base-
lines. The no-retrieval baselines include naive generation and chain-of-thought (CoT) prompting [Wei
et al., 2022]. The retrieval-based baselines are vanilla retrieval-augmented generation (RAG) with
CoT, ReAct [Yao et al., 2023], and optimized few-shot prompting with ReAct [Khattab et al., 2023].
We also compare FRUGAL RAG to recently proposed approaches that leverage large-scale fine-tuning,
including Search-R1 [Jin et al., 2025], LeReT [Hsu et al., 2024], and CoRAG [Wang et al., 2025].
However, we note that these recent methods do not report all metrics that we consider and their
metrics may not be directly comparable (due to varying model sizes and retrievers). Therefore, we
conduct comparisons using the closest model scales and overlapping subsets of evaluation metrics.
Training. FRUGAL RAG is model agnostic – we train Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct,
Meta-Llama-3.1-8B-Instruct using our two-stage framework. In both the supervised finetuning
and reinforcement learning (RL) stages, we leverage the TRL library [von Werra et al., 2020], and
our ReAct pipeline with prompt bootstrapping is built on DsPy [Khattab et al., 2023]. For each
dataset, we prepare the Stage 1 finetuning dataset with 1000 randomly sampled examples from the
corresponding training split. Stage 1 (Sec. 3.1) consists of full-parameter finetuning for a single
epoch, using a learning rate of 2×10−5and a weight decay of 0.01 for all models and datasets. We
choose a maximum sequence length of 4096 during finetuning. In Stage 2 (Sec. 3.2), we further
train the finetuned models via GRPO [Shao et al., 2024]. For a detailed list of hyperparameters refer
Appendix A.
4.2 Main Results
In Tables 1 and Table 2, we compare FRUGAL RAG against strong baselines on HotPotQA, 2Wiki-
MultiHopQA, and MuSiQue. For fair comparison, all baselines use the ColBERTv2 [Santhanam
et al., 2021] retriever indexed on Wikipedia (See Appendix A. The key takeaway is that FRUGAL RAG
consistently outperforms the baselines on both answer and retrieval metrics, using competitive or
significantly smaller number of searches on average.
6

Table 1: Retriever-level metrics (Recall, Support F1, Search latency) across HotpotQA, 2WikiMulti-
HopQA, and MuSiQue. FRUGAL RAG achieves the best tradeoff between performance and efficiency
on all datasets using Llama3.1-8B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct. Green
(bold ) represents the best score and blue ( underlined ) represents the second best score. "-" indicates
results are not present in the respective papers.
Model MethodHotpotQA 2Wiki MuSiQue
Recall Sup. F1 Search Recall Sup. F1 Search Recall Sup. F1 Search
Llama3.1-8B-InstructNaive 0 0 0 0 0 0 0 0 0
CoT 0 0 0 0 0 0 0 0 0
CoT+RAG (n=3) 58.56 66.29 1 36.44 49.49 1 22.29 25.61 1
CoT+RAG (n=5) 63.23 69.97 1 36.44 49.48 1 22.82 25.55 1
ReAct 73.04 79.91 3.83 44.11 56.39 4.37 30.03 29.00 4.82
ReAct FS 77.44 83.03 4.79 48.83 60.04 5.46 33.74 30.75 5.48
LeReT 77.10 - 2.0 - - - - - -
FRUGAL RAG-Explore 83.11 86.96 5.96 51.79 62.35 6.0 37.11 32.27 5.99
FRUGAL RAG 79.62 84.47 2.96 51.87 62.13 5.38 33.90 30.86 4.31
Qwen2.5-3B-InstructNaive 0 0 0 0 0 0 0 0 0
CoT 0 0 0 0 0 0 0 0 0
CoT+RAG (n=3) 58.56 66.29 1 32.87 46.16 1 19.95 23.61 1
CoT+RAG (n=5) 63.20 69.94 1 36.38 49.46 1 22.73 25.52 1
ReAct 65,13 73.96 1.88 39.76 52.67 2.30 25.85 26.81 2.68
ReAct FS 71.47 78.64 3.33 45.77 57.67 3.99 30.44 29.09 4.32
Search-R1 - - - - - - - - -
FRUGAL RAG-Explore 81.41 85.72 5.90 54.13 64.17 5.91 37.02 31.89 5.87
FRUGAL RAG 78.77 83.89 2.62 52.70 62.79 4.02 33.97 30.56 4.25
Qwen2.5-7B-InstructNaive 0 0 0 0 0 0 0 0 0
CoT 0 0 0 0 0 0 0 0 0
CoT+RAG (n=3) 58.56 66.29 1 32.87 46.16 1 19.95 23.61 1
CoT+RAG (n=5) 63.20 69.94 1 36.38 49.46 1 22.73 25.52 1
ReAct 73.45 80.03 2.07 45.80 58.62 2.78 30.94 29.11 3.13
ReAct FS 77.65 82.80 2.91 49.25 61.03 3.48 33.90 30.50 4.16
Search-R1 - - - - - - - -
FRUGAL RAG-Explore 84.19 87.63 5.88 55.01 64.97 5.89 37.11 31.88 4.32
FRUGAL RAG 83.86 87.43 2.75 54.38 64.32 4.86 36.71 31.86 3.98
(a)
 (b)
 (c)
Figure 1: (a) Frequency of number of searches (HotPotQA, Qwen2.5-3B-Instruct): FRUGAL RAG
estimates the optimal searches whereas FRUGAL RAG -Explore uses a fixed budget. (b-c) FRUGAL -
RAG achieves a higher F1 score compared to ReAct FS using different budgets while being more
efficient. (Zoom in for a better view)
TheReAct Few Shot (FS) baseline [Yao et al., 2023, Khattab et al., 2023] outperforms vanilla Retrieval
Augmented Generation and ReAct without few shot prompts consistently. The optimized prompts
enable ReAct FS to generate more search queries, and as a result achieve very strong performance.
Strikingly, on HotPotQA, ReAct FS achieves (a) 77.44% and77.65% recall with Llama3.1-8B-
Instruct, Qwen2.5-7B-Instruct respectively, which is greater than the recall achieved by LeReT [Hsu
et al., 2024] (77.1% using Llama3.1-8B) after significant finetuning; and (b) comparable Exact Match
(42.1) to that of the recently proposed methods such as Search-R1 [Jin et al., 2025] (43.3 using base,
37.0 using instruct); signifying the importance of building a strong baseline.
7

Table 2: Answer-level metrics (F1, EM, Match, Search latency) across HotpotQA, 2WikiMultiHopQA,
and MuSiQue. FRUGAL RAG achieves the best tradeoff between performance and efficiency on all
datasets using Llama3.1-8B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct. Green represents
the best score and blue represents the second best score. "-" indicates results not present in the paper.
Model MethodHotpotQA 2Wiki MuSiQue
F1 EM Match Search F1 EM Match Search F1 EM Match Search
Llama3.1-8B-InstructNaive 14.17 5.26 31.03 0 4.28 0.24 13.28 0
CoT 27.23 15.11 32.39 0 18.73 9.58 33.60 0 12.10 4.33 11.93 0
CoT+RAG (n=3) 42.30 28.93 45.92 1 21.54 13.37 32.88 1 14.35 7.01 14.41 1
CoT+RAG (n=5) 42.54 28.59 47.26 1 21.49 13.32 32.98 1 14.51 7.26 14.41 1
ReAct 37.39 22.09 49.72 3.83 23.22 11.15 38.62 4.37 12.50 5.83 15.11 4.83
ReAct FS 55.38 42.57 50.29 4.79 35.74 28.18 38.76 5.46 18.73 11.21 15.80 5.48
LeReT - 52.5 - 2.0 - - - - - - - -
FRUGAL RAG-Explore 60.58 46.67 55.21 5.96 44.72 37.39 46.31 6.0 23.44 15.56 19.70 5.99
FRUGAL RAG 62.95 49.54 56.28 2.96 42.78 35.30 44.46 5.38 21.23 13.00 17.14 4.31
Qwen2.5-3B-InstructNaive 8.33 1.24 22.41 0 9.84 0.62 37.64 0 4.36 0.20 7.36 0
CoT 13.90 6.54 22.23 0 15.31 7.07 25.38 0 6.62 5.95 1.94 0
CoT+RAG (n=3) 35.29 21.85 43.29 1.0 18.15 12.19 21.54 1 9.15 4.26 8.27 1
CoT+RAG (n=5) 36.56 25.37 36.63 1.0 20.11 13.78 24.33 1 9.63 3.97 8.02 1
ReAct 25.35 19.84 42.91 1.88 15.87 2.64 34.84 2.30 7.64 1.44 11.75 2.68
ReAct FS 50.88 38.02 47.65 3.33 23.83 16.23 24.18 3.99 13.30 7.73 11.66 4.32
Search-R1 - 32.4 - - - 31.90 - - - 10.30 - -
FRUGAL RAG-Explore 53.17 40.56 49.65 5.90 25.97 17.00 28.19 5.91 18.84 12.12 15.92 5.87
FRUGAL RAG 55.81 42.88 51.20 2.62 30.01 21.46 31.49 4.03 18.71 11.17 15.76 4.25
Qwen2.5-7B-InstructNaive 13.63 5.49 24.30 0 16.43 9.88 32.47 0 6.40 0.70 8.00 0
CoT 23.64 14.07 24.96 0 24.24 18.30 29.60 0 10.80 3.59 8.06 0
CoT+RAG (n=3) 35.67 22.20 44.01 1 14.95 6.34 30.38 1 10.52 4.00 12.53 1
CoT+RAG (n=5) 36.53 22.70 46.54 1 16.82 7.28 33.27 1 11.51 4.79 12.53 1
ReAct 29.75 11.85 56.28 2.07 21.28 6.80 43.80 2.78 10.69 3.56 20.39 3.13
ReAct FS 53.59 42.10 53.20 2.91 41.74 32.90 42.50 3.48 20.26 11.91 18.99 4.16
Search-R1 - 37.00 - - - 41.40 - - - 14.60 - -
FRUGAL RAG-Explore 61.84 47.87 57.31 5.88 42.55 33.07 45.61 5.89 22.09 13.19 22.67 4.32
FRUGAL RAG 63.03 48.81 58.25 2.75 45.00 36.13 47.00 4.86 24.78 14.89 21.84 3.98
Figure 2: FRUGAL RAG on average outperforms fixed budget and SFT based baselines, demonstrating
the effectiveness of both Stage-1 finetuning and Stage-2 learning to control test time compute. (Zoom
in for a better view). The plots show the % Tradeoff metrics defined in Section 4.3.
Table 1 and Table 2 demonstrate the overall effectiveness of FRUGAL RAG -Explore achieving the
highest Recall and Support F1 scores compared to all baselines. However, we note that we note that
FRUGAL RAG -Explore is either best or second best in terms of both answer and recall metrics but
introduces a very high latency compared to FRUGAL RAG . Stage-2 finetuning significantly reduces
the number of searches required while retaining the performance across three datasets. We define
anefficiency score metric as the average performance per search. For instance, the search count on
HotPotQA reduces on average by -53.05% across the three models and the F1 scores improve by
+3.5% for a marginal drop in recall. A similar trend is observed in the case of 2Wiki and MuSiQue,
where the search count reduces on average by -19.87% and-21.17% and the F1 scores improve by
+5.65% ,+0.68% respectively.
Compared to recent approaches such as LeReT Hsu et al. [2024], which evaluate performance on
HotPotQA, F RUGAL RAG achieves higher recall ( 79.62 vs.77.10 ) with only a marginal increase of
+0.96 in the average number of retrieval steps. Additionally, despite relying on an off-the-shelf answer
8

generator g,FRUGAL RAG outperforms reasoning-based baselines like Search-R1 Jin et al. [2025]
on Exact Match in 2 out of 3 datasets. The lower performance on 2Wiki compared to Search-R1 is
primarily due to the use of an off-the-shelf answer generator, whereas Search-R1 employs a trained
model for answer generation.
Overall, our results highlight that our method strikes a significantly better efficiency-accuracy tradeoff,
compared to strong baselines.
4.3 Analysis
In this section, we present a discussion on the various properties of FRUGAL RAG and demonstrate its
effectiveness through ablation studies. For our analysis, we choose HotPotQA (2 hops) and MuSiQue
(upto 4 hops) datasets due to the managable size of their datasets.
Adaptive vs Fixed Budget Search. A simple way to make ReAct baselines more efficient is to limit
their number of searches to a fixed budget. Does variable compute (searches) really help in handling
questions of varying difficulty? To get an estimate of the query difficulty and search requirements,
we look at the histogram of number of searches in the trajectories of FRUGAL RAG -Explore up until
when 100% Recall is attained. Fig. 1a illustrates that most additional searches beyond a certain point
are redundant (Estimated Optimal Searches), and the utility of each additional query diminishes
exponentially, from the perspective of Recall. So, we can expect ReAct baselines with fixed (small)
search budget to attain competitive performance.
We investigate the utility of variable compute in Fig 1b with Qwen3.5-3B-Instruct and in Fig 1c
with Qwen3.5-7B-Instruct. We consider the best baseline (ReAct FS), and give it a search budget
B, and ensure that it does not generate FINISH till the entire budget Bis consumed. This allows us
to directly compare the accuracy of FRUGAL RAG that uses variable budget per query with fixed-
budget baselines. Using Qwen2.5-3B-Instruct, we find that FRUGAL RAG consistently outperforms
ReAct across all budgets B= 2,3,4,5,6in terms of F1 score, while also being more efficient.
On HotPotQA, FRUGAL RAG achieves an F1 of 55.81 compared to ReAct FS ( B= 6) at 53.17.
Similarly, on MuSiQue, FRUGAL RAG reaches 18.71 , slightly improving over ReAct’s 18.84, but
with fewer retrieval steps on average. This trend also holds for the larger Qwen2.5-7B-Instruct
model. On HotPotQA, FRUGAL RAG achieves a significantly higher F1 of 63.03 , compared to
57.59 from ReAct FS ( B= 6), while requiring only around 2.5 searches on average. On MuSiQue,
FRUGAL RAG reaches 24.78 (with roughly 4 avg searches ) compared to ReAct FS ( B= 6) at 17.82,
again demonstrating superior performance and efficient retrieval (Detailed Tables are provided in the
Appendix).
Impact of RL based training. To capture the trade-off between answer quality and retrieval cost, we
define two key metrics:
•% Tradeoff (Answer) quantifies how effectively each retrieval contributes to answer quality:
%Tradeoff Answer = 100 ×F1+EM+Match
3×Searches
•% Tradeoff (Retrieval) measures how well each retrieval contributes to evidence quality:
%Tradeoff Retrieval = 100 ×Recall +Support F1
2×Searches
These metrics allow us to jointly evaluate answer accuracy and evidence quality relative to the number
of retrieval steps, enabling a comparison of efficiency across models and budgets.
To assess the impact of the RL-finetuning (Stage 2), we conduct an ablation study by comparing
FRUGAL RAG against a simple baseline, called FRUGAL RAG -SFT. This baseline is trained for 1
epoch on the dataset generated during Stage 1, but with one key modification. Instead of sample 90%
rollouts without FINISH , we sample all traces where the model outputs FINISH on its own.
In Fig. 2, we demonstrate that FRUGAL RAG on average across all models is better on both
Tradeoff Retrieval andTradeoff Answer .FRUGAL RAG outperforms both FRUGAL RAG -SFT and FRU-
GALRAG -Explore by a significant margin using Qwen2.5-3B-Instruct and Llama3.1-8B-Instruct
on both the datasets as illustrated in Fig. 2. We note that the FRUGAL RAG -SFT performance is
comparable to F RUGAL RAG with Qwen2.5-7B-Instruct.
9

Table 3: Retriever-level metrics (Recall, Support F1, Search latency) for HotpotQA, 2WikiMulti-
HopQA, and MuSiQue using Llama-3.1-8B-Instruct . Green denotes improvement and red denotes
decrease, over the test-time scaling baseline CoRAG. For CoRAG, we choose the first 30 documents
from the provided inference files.
Retriever, Index Method Dataset Recall Sup. F1 Search
E5, KILT CoRAG [Wang et al., 2025]HotpotQA 74.55 70.35 6.00
2Wiki 69.73 66.72 6.00
MuSiQue 43.25 34.41 6.00
E5, KILT F RUGAL RAG-ExploreHotpotQA 75.82 72.10 6.00
2Wiki 83.77 78.05 5.99
MuSiQue 47.53 35.30 6.00
E5, KILT FRUGAL RAGHotpotQA 76.33 (1.78) 72.18 (1.91) 5.52 (0.48)
2Wiki 84.75 (15.01) 78.55 (11.83) 5.67 (0.33)
MuSiQue 47.39 (4.14) 35.37 (0.96) 5.16 (0.84)
ColBERT, Wiki F RUGAL RAG-ExploreHotpotQA 83.11 86.96 5.96
2Wiki 51.79 62.35 6.00
MuSiQue 37.11 32.27 5.99
ColBERT, Wiki FRUGAL RAGHotpotQA 79.62 84.47 2.96
2Wiki 51.87 62.13 5.38
MuSiQue 33.90 30.86 4.31
Comparison with CoRAG. We compare FRUGAL RAG against the recent state-of-the-art test-time
scaling approach CoRAG [Wang et al., 2025]. CoRAG jointly trains the reasoning and the answer
generation models on 100K multi-hop examples. In contrast, FRUGAL RAG relies on an off-the-shelf
answer generator, which limits its ability to match the exact answer formats expected by benchmark
evaluation scripts. So, we adopt the following approach for fair comparison.
We replicate CoRAG’s experimental setup using the E5-large retriever1and KILT Petroni et al. [2021]
index. We fine-tune FRUGAL RAG with Llama3.1-8B-Instruct separately on each dataset only using
1000 examples. For retrieval evaluation, we use the publicly released CoRAG outputs2to report
Recall and Support F1 using all the retrieved documents, allowing us to compare the decomposition
capabilities of FRUGAL RAG and CoRAG. Concretely, we use the context document ids upto B=6
searches (30 documents) and extract the titles and passages to compute the recall and support F1
respectively.
In Table 3, we present retrieval-level metrics of CoRAG and FRUGAL RAG variants on three datasets.
The results clearly demonstrate that FRUGAL RAG , when using the E5 retriever, significantly outper-
forms CoRAG (B=6) in both Recall and Support F1. We present a detailed list of hyperparamters for
this run in Appendix A. This highlights the robustness and generalization ability of FRUGAL RAG
across different retrievers and index corpora. Notably, the number of retrieval queries is substantially
higher on KILT [Petroni et al., 2021] compared to Wikipedia, which is expected given KILT’s
larger document collection (36 million vs. approximately 5 million for HotPotQA, 20 million for
2Wiki/MuSiQue using ColBERTv2 [Santhanam et al., 2021]).
5 Conclusions, limitations, and future work
In this work, we argue that efficiency is an equally important metric to study in RAG solutions,
besides the traditional RAG metrics such as retrieval performance and accuracy of the generated
answers. We demonstrate that simple ReAct baseline that iteratively retrieves (by invoking the search
tool) and reasons (to decide what search call to issue next) is quite competitive, especially if we
can optimize its few-shot prompt using just tens of training examples. We propose a new two-stage
framework FRUGAL RAG that a) works with 1000 training examples, compared to state-of-the-art
RAG techniques that use over 100,000 examples, and b) yet achieves competitive accuracies while
also using far fewer search queries at inference time, on popular multi-hop QA datasets.
1https://huggingface.co/intfloat/e5-large-v2
2https://huggingface.co/datasets/corag/
10

Even though our method uses a small number of examples for training, it has some limitations in the
analysis, and leaves room for future work in the following aspects: (a) in terms of generalization to
new domains with no access to training data, (b) in terms of scaling the training data to potentially
large number of examples, (c) jointly training the answer generator and decomposer. We hope to
explore these avenues in the future.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to
retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on
Learning Representations , 2023.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu. Rq-rag:
Learning to refine queries for retrieval augmented generation. arXiv preprint arXiv:2404.00610 ,
2024.
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z
Pan, Wen Zhang, Huajun Chen, Fan Yang, et al. Learning to reason with search for llms via
reinforcement learning. arXiv preprint arXiv:2503.19470 , 2025.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models. arXiv preprint arXiv:2407.21783 , 2024.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented
language model pre-training. In International conference on machine learning , pages 3929–3938.
PMLR, 2020.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-
hop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott, Nuria Bel,
and Chengqing Zong, editors, Proceedings of the 28th International Conference on Compu-
tational Linguistics , pages 6609–6625, Barcelona, Spain (Online), December 2020. Interna-
tional Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.580. URL
https://aclanthology.org/2020.coling-main.580/ .
Sheryl Hsu, Omar Khattab, Chelsea Finn, and Archit Sharma. Grounding by trying: Llms with
reinforcement learning-enhanced retrieval. arXiv preprint arXiv:2410.23214 , 2024.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity. arXiv
preprint arXiv:2403.14403 , 2024.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing , pages 7969–7992, 2023.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement
learning. arXiv preprint arXiv:2503.09516 , 2025.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP
(1), pages 6769–6781, 2020.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. Dspy: Compiling
declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714 ,
2023.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles , 2023.
11

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In
H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in
Neural Information Processing Systems , volume 33, pages 9459–9474. Curran Associates,
Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf .
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint
arXiv:1711.05101 , 2017.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James
Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel,
and Sebastian Riedel. Kilt: a benchmark for knowledge intensive language tasks, 2021. URL
https://arxiv.org/abs/2009.02252 .
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer. Journal of machine learning research , 21(140):1–67, 2020.
Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimiza-
tions enable training deep learning models with over 100 billion parameters. In Proceedings of
the 26th ACM SIGKDD international conference on knowledge discovery & data mining , pages
3505–3506, 2020.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2:
Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488 ,
2021.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke
Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach
themselves to use tools. Advances in Neural Information Processing Systems , 36:68539–68551,
2023.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing
retrieval-augmented large language models with iterative retrieval-generation synergy. arXiv
preprint arXiv:2305.15294 , 2023.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical
reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions. arXiv preprint
arXiv:2212.10509 , 2022a.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition. Transactions of the Association for Computational
Linguistics , 10:539–554, 2022b.
Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan
Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement
learning. https://github.com/huggingface/trl , 2020.
Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, and Furu Wei. Chain-of-
retrieval augmented generation. arXiv preprint arXiv:2501.14342 , 2025.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in
neural information processing systems , 35:24824–24837, 2022.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115 , 2024.
12

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2018.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. In International Conference on
Learning Representations (ICLR) , 2023.
A Training Details
Algorithm 1 shows the overall training framework of FRUGAL RAG . We plan to publicly release our
code soon and have attached a copy of the codebase for review in the meantime. Below, we discuss
each step along with their implementation details.
Few-Shot Prompt Optimization Details. We leverage DSPy [Khattab et al., 2023] for automatic
few-shot prompt generation following LeReT [Hsu et al., 2024]. Specifically, we use 50 training
examples ( Linit) with the BOOTSTRAP FEWSHOTWITH RANDOM SEARCH method, which uses the
LMfto generate few-shot examples, selecting the best performing ones for subsequent prompting.
We select 4 best performing few-shot prompts from a total of 15 candidate sets using the sum of
answer EM and answer passage match. Answer EM checks for an exact string-match between the
generated and actual answer, and passage match checks if the actual answer is present in the retrieved
passages. This step is crucial because it facilitates dataset generation using diverse rollouts and
ensures the answer format is followed by the model. For this step, we serve our model on one GPU
using VLLM [Kwon et al., 2023]. For all experiments involving Qwen2.5, we utilize the 7B-Instruct
variant for prompt optimization. The optimized prompts are then reused without modification for the
3B variant.
Dataset Generation Details. For each few-shot prompt pi, the model fgenerates a tuple (Ti
h, Ai
h, Si
h)
representing a candidate output for the next hop. As described in Sec. 3.1, we evaluate all candidate
tuples at hop hand select one with the highest recall. This selected candidate is then used as the
context for the next hop and the process is repeated till budget B(optionally till the selected candidate
action Ahindicates FINISH ). We set the budget B= 6, where the initial retrieval step is always
R(Q(j))withQ(j)denoting the original user utterance. The generated dataset is denoted by D.
For all experiments involving Qwen2.5, we utilize the 7B-Instruct variant along with its prompts to
generate the dataset. For further improving results, we can repeat few shot prompt optimization and
dataset generation using different base models.
Supervised "Explore" Finetuning Details. We use the standard next token prediction loss given by:
max
fE(x,y)∼Dlogpf(x|y) (2)
where y= (Th, Ah, Sh)andx=Q(j)∪ {Tk, Ak, Sk,Dk}h−1
k=0sampled from the generated dataset
D.
We train the model ffor 1 epoch using a batch size of 4 and apply gradient accumulation of 2 steps,
resulting in an effective batch size of 8. Optimization is performed using AdamW [Loshchilov and
Hutter, 2017] with a learning rate of 2×10−5. We use a linear learning rate scheduler with a warmup
phase of 20 steps. The training is performed using 8 H100 80GB GPUs.
Controlling test-time compute with RL. Our RL step employs GRPO for fine-tuning the base policy
fS. Specifically, following the notation in DeepSeekMath [Shao et al., 2024], for each question Q(j),
we sample a group of outputs {o1
h, o2
h, . . . , ov
h}at hop h, where vis set to 8. We optimize our base
policy fSusing the standard GRPO objective using the cumulative rollout reward as defined in Eq. 1.
We use a KL divergence penalty with weight 0.1since we have a trained base policy, and set the
maximum reward Rmax= 2.0for stability. Generation is limited to a maximum of 256 completion
tokens and the maximum prompt size is 1024. Training is conducted using DeepSpeed-Zero2 [Rasley
et al., 2020] and 7 H100 GPUs (where 1 is exclusively reserved for sampling). We set the learning
rate to 10−6. Due to the long prompt (which includes retrieved documents from previous hops), we
use a total batch size of 48. We train FRUGAL RAG for 400 steps across datasets and models, and
report the performance using the final checkpoint.
13

Algorithm 1: Our novel two-stage framework, FRUGAL RAG consists of (1)Dataset Generation
andSupervised "Explore" Finetuning , and (2)Controlling test-time compute wth RL .
Input: Labeled dataset L={(Q(j), Y(j))}1000
j=1,Linit={(Q(j), Y(j))}50
j=1, retriever R, base
LMf, budget B, max hops m, number of samples v
// Prompt Optimization
1Perform prompt optimization on fusingLinitto obtain few-shot prompts {p1, . . . , p n};
// Dataset Generation
2Initialize finetuning dataset: D←[];
3forQ(j), Y(j)inLdo
4 Initialize buffer: main_rollout ←[];
5 Initialize D0← R(Q(j))or∅;
6 Initialize T0, A0;
7H0← {Q(j), T0, A0,D0}// stores previous context
8 Append H0tomain_rollout ;
9 forh= 1tomdo
10 foriin1. . . n do
11 forh= 1toBdo
12 (Ti
h, Ai
h, Si
h)←f(Hi
h−1;pi);
// occurs in 10% of calls
13 ifAi
h=FINISH then
14 break
15 Di
h← R(Si
h);
16 Remove duplicate retrievals from Di
h;
17 Hi
h← Hi
h−1∪ {Ti
h, Ai
h, Si
h,Di
h};
18 Evaluate all {Di
h}n
i=1(recall against ground truth Y(j));
19 Select best-performing trajectory H∗;
20 Append H∗tomain_rollout ;
21 Append each hop from main_rollout toD;
// Stage 1: Supervised "Explore" Finetuning
22fS←Fine-tune fonDusing standard next-token prediction // See Eq. 2
// Stage 2: Controlling test-time compute with RL
23forQ(j), Y(j)inLdo
24 forh= 1tomdo
25 Generate vsample tuples {Ti
h, Ai
h, Si
h,Di
h}v
i=1for hop h;
26 fori= 1tovdo
27 Compute reward Ri←R({Di
h}m
h=1, Y(j),fS)// See Eq. 1
28 Backpropagate loss on {Ti
h, Ai
h, Si
h}m
h=1using Ri;
14

B Metrics
F1.is the harmonic mean of the precision and recall measured at the word-level, and is given by
F1 = 2 ·TP ans
TP ans+0.5∗(FP ans+ FN ans)(3)
where TP ansdenotes correctly predicted words, FP ansrepresents the extra words, and FN ansare the
missing words in the generated answer.
Exact Match. is the exact match between the normalized generated answer string and normalized
ground truth answer. It is 100% if the two strings match exactly, and 0otherwise.
Match Score. measures the accuracy of generated answer by checking if the ground truth answer
string is present in the generated answer string. It is 100% if the ground truth string is in the generated
answer, and 0otherwise.
Recall. is a retrieval metric, that measures the percentage of ground truth documents retrieved by the
model. It is given by the ratio of correctly retrieved document titles TP docand the total number of
ground truth document titles TP doc+ FN doc. We measure recall following LeReT Hsu et al. [2024],
using document titles. The document-level recall is given by –
Recall =TP doc
TP doc+ FN doc(4)
Sup. F1. measures the word-level F1 score (3)between the ground truth evidence sentences and
those retrieved from the documents. Following Trivedi et al. [2022b], we compute the average F1
score across the ground truth evidence sentences, comparing them with corresponding retrieved
documents. The evidence sentences are are provided in all three datasets. Supporting Document F1
or Sup. F1 is serves as a more reliable metric for retrieval with 2WikiMultiHopQA and MuSiQue
since it considers fine-grained evidence rather than just the document titles.
C Dataset and Retrieval Index
We use the pre-processed Wikipedia abstracts index3provided by ColBERTv2 [Santhanam et al.,
2021] for all our experiments on HotPotQA [Yang et al., 2018]. For each instance, we retrieve the top
3 documents and their titles and perform a maximum 6 retrievals. HotPotQA annotations consists
of document title and evidence sentences which are used to compute the Recall and Supporting
Document F1 respectively.
Since 2WikiMultiHopQA [Ho et al., 2020] and MuSiQue [Trivedi et al., 2022b] datasets are created
using both the body and abstract of wikipedia articles we use the pre-processed dump of Wikipedia
provided by Karpukhin et al. [2020] and index it using ColBERTv2 [Santhanam et al., 2021]. The
generated index consists of 21M passages. For each instance, we retrieve top 5 documents and append
it to our context. For experiments in Table 3, we use E5-Large provided by CoRAG Wang et al.
[2025] indexed on KILT Petroni et al. [2021] which consists of 36 million passages, and retrieve top
5 documents for all datasets.
3https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/wiki.abstracts.2017.
tar.gz
15
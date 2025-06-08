# R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning

**Authors**: Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, Limin Liu

**Published**: 2025-06-04 17:29:22

**PDF URL**: [http://arxiv.org/pdf/2506.04185v1](http://arxiv.org/pdf/2506.04185v1)

## Abstract
Large language models (LLMs) have notably progressed in multi-step and
long-chain reasoning. However, extending their reasoning capabilities to
encompass deep interactions with search remains a non-trivial challenge, as
models often fail to identify optimal reasoning-search interaction
trajectories, resulting in suboptimal responses. We propose R-Search, a novel
reinforcement learning framework for Reasoning-Search integration, designed to
enable LLMs to autonomously execute multi-step reasoning with deep search
interaction, and learn optimal reasoning search interaction trajectories via
multi-reward signals, improving response quality in complex logic- and
knowledge-intensive tasks. R-Search guides the LLM to dynamically decide when
to retrieve or reason, while globally integrating key evidence to enhance deep
knowledge interaction between reasoning and search. During RL training,
R-Search provides multi-stage, multi-type rewards to jointly optimize the
reasoning-search trajectory. Experiments on seven datasets show that R-Search
outperforms advanced RAG baselines by up to 32.2% (in-domain) and 25.1%
(out-of-domain). The code and data are available at
https://github.com/QingFei1/R-Search.

## Full Text


<!-- PDF content starts -->

arXiv:2506.04185v1  [cs.CL]  4 Jun 2025R-Search: Empowering LLM Reasoning with Search
via Multi-Reward Reinforcement Learning
Qingfei Zhao1,2, Ruobing Wang1,2, Dingling Xu3, Daren Zha1∗, Limin Liu1
1Institute of Information Engineering, Chinese Academy of Sciences
2School of Cyber Security, University of Chinese Academy of Sciences
3Beijing Normal University
{zhaoqingfei,wangruobing,zhadaren}@iie.ac.cn
Abstract
Large language models (LLMs) have notably
progressed in multi-step and long-chain rea-
soning. However, extending their reasoning
capabilities to encompass deep interactions
with search remains a non-trivial challenge, as
models often fail to identify optimal reason-
ing–search interaction trajectories, resulting in
suboptimal responses. We propose R-Search ,
a novel reinforcement learning framework for
Reasoning– Search integration, designed to en-
able LLMs to autonomously execute multi-
step reasoning with deep search interaction,
and learn optimal reasoning–search interaction
trajectories via multi-reward signals, improv-
ing response quality in complex logic- and
knowledge-intensive tasks. R-Search guides
the LLM to dynamically decide when to re-
trieve or reason, while globally integrating key
evidence to enhance deep knowledge interac-
tion between reasoning and search. During RL
training, R-Search provides multi-stage, multi-
type rewards to jointly optimize the reason-
ing–search trajectory. Experiments on seven
datasets show that R-Search outperforms ad-
vanced RAG baselines by up to 32.2% (in-
domain) and 25.1% (out-of-domain). The code
and data are available at https://github.
com/QingFei1/R-Search .
1 Introduction
Large language models (LLMs) have demonstrated
substantial progress across a wide range of natural
language processing (NLP) tasks, driven by their
impressive language understanding and reasoning
abilities (OpenAI, 2023; Plaat et al., 2024). In
logic-intensive tasks (Asai and Hajishirzi, 2020;
Shao et al., 2024), state-of-the-art LLMs, exempli-
fied by DeepSeek-R1 (DeepSeek-AI et al., 2025),
have demonstrated remarkable capabilities in long-
chain and multi-step reasoning (Jaech et al., 2024).
In knowledge-intensive tasks (Lewis et al., 2020;
*Corresponding authorTrivedi et al., 2023), even LLMs with strong rea-
soning capabilities are susceptible to generating
hallucinated outputs (Zhang et al., 2023). This
primarily arises from inherent limitations in the ac-
curacy, timeliness, and coverage of their parametric
knowledge. To mitigate hallucination, LLM-based
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Zhao et al., 2024) incorporates search
actions before generation, enabling the LLM to
augment its input with non-parametric knowledge
in textual form. This allows the LLM to flexibly
access and integrate relevant information from ex-
ternal knowledge sources, thereby enhancing the
reliability of downstream generation. However,
downstream generation often struggles to bene-
fit from reasoning alone or one-time search when
addressing more complex logic- and knowledge-
intensive tasks, e.g., multi-hop question-answering
(QA) task (Yang et al., 2018). In tackling such
complex tasks, the LLM is expected to dynami-
cally integrate external knowledge into the reason-
ing process, not only to bridge the knowledge gap
but to guide and deepen the reasoning trajectory.
Previous multi-turn RAG methods (Jeong et al.,
2024; Trivedi et al., 2023) enable the integration
of external knowledge into reasoning by prompt-
ing LLMs to iteratively perform reasoning–search
interactions. In this process, retrieved informa-
tion enhances the model’s reasoning, which in turn
guides the subsequent retrieval, forming a dynamic
loop between them. However, these methods typi-
cally rely on the LLM’s internal cognition to decide
when and what to retrieve, leading to two main lim-
itations: 1)The retrieval timing determined by the
model’s internal knowledge distribution does not al-
ways align with the actual need for retrieval; 2)the
modular and decoupled design of reasoning and
search limits deep interaction of external knowl-
edge into the reasoning chain. As a result, the
model often makes decisions based only on partial
information from previous searches or thoughts.
1

These limitations lead to suboptimal or even in-
correct reasoning–search trajectories, ultimately
reducing the quality of the final outputs.
To this end, we propose R-Search, a novel re-
inforcement learning (RL)-based framework that
enables LLMs to dynamically interleave multi-step
reasoning and search, and to learn optimal reason-
ing–search trajectories through multi-reward sig-
nals. First , R-Search allows the LLM to trigger
retrieval at any token-level reasoning step, seam-
lessly integrating retrieved content into the reason-
ing process for deeper coupling between reasoning
and external knowledge. After the interaction, the
LLM distills retrieved documents into evidence
through reasoning. This facilitates the LLM in re-
evaluating and structuring critical knowledge from
a global perspective, thereby enhancing its focus
on the facts most pertinent to solving the task. Sec-
ond, we design a multi-stage, multi-type reward
mechanism that incorporates answer quality, ev-
idence quality, and format correctness as reward
signals. These complementary signals promote the
model to learn the optimal reasoning–search inter-
action sequence. In particular, the evidence reward
encourages the model to focus on the factual qual-
ity of key intermediate reasoning steps, promoting
more robust reasoning paths and reducing the risk
of shortcut-driven or speculative behavior.
Our contributions are threefold: 1) Framework
Design: we propose R-Search, a novel RL-based
framework that jointly optimizes complex rea-
soning–search trajectories in RAG. R-Search pro-
motes robust policy learning by interleaving multi-
step reasoning with dynamic search and optimiz-
ing through multi-reward modeling. It effectively
guides the LLM to ensure both the soundness of
intermediate reasoning and the completeness of re-
trieved knowledge. 2) Superior Performance and
Insightful Analysis: we conduct extensive experi-
ments on seven datasets across both multi-hop and
single-hop QA tasks, demonstrating the superiority
of R-Search over vanilla and advanced RAG base-
lines (up to 32.2%). Further analyses—including
ablation and training dynamics—validate the effec-
tiveness of evidence integration and multi-reward
modeling, and provide insights into performance
trends and retrieval behaviors under different RL
algorithms. 3) R-Search-as-a-Tool: We propose
R-Search-as-a-Tool (RSTool), which modularizes
high-quality evidence in reasoning into transferable
components, enabling the offloading of complex
and costly reasoning–search interactions to localdeployments, with strong practical scalability.
2 Related Work
2.1 Retrieval-Augmented Generation (RAG)
RAG typically follows a retrieve-and-generate
paradigm (Lewis et al., 2020; Guan et al., 2025).
The retrieval corresponds to the Search action,
which involves acquiring external non-parametric
knowledge from various sources using different
search tools. The generation refers to the tokens
produced by LLM reasoning, encompassing both
the intermediate reasoning process and the exe-
cution of specific modules in modular RAG, in-
cluding the final answer generation (Gao et al.,
2023). Recently, LLM-based RAG systems (Ram
et al., 2023; Yoran et al., 2024) have demonstrated
significant performance gains across various NLP
tasks (Brown et al., 2020; OpenAI, 2023; Plaat
et al., 2024), especially in open-domain question
answering. To produce higher quality responses,
some branching RAG (Kim et al., 2024; Shi et al.,
2024) methods summarize the retrieved documents
separately for multiple candidate responses to im-
prove understanding of external knowledge. How-
ever, as tasks involving more complex reasoning
(e.g. multi-hop QA tasks (Yang et al., 2018; Ho
et al., 2020; Trivedi et al., 2022; Press et al., 2023))
have emerged, traditional RAG methods (Guu et al.,
2020; Sachan et al., 2021) struggle with insufficient
external knowledge integration. Several advanced
RAG methods (Jeong et al., 2024; Jiang et al., 2023;
Cheng et al., 2024; Trivedi et al., 2023; Shao et al.,
2023; Chan et al., 2024; Asai et al., 2024) have
attempted multi-turn reasoning–search interactions
to enable deeper knowledge exploration, includ-
ing multi-step (Trivedi et al., 2023; Chan et al.,
2024; Shao et al., 2023) and adaptive RAG meth-
ods (Jeong et al., 2024; Jiang et al., 2023; Asai
et al., 2024). Nevertheless, these methods still heav-
ily rely on carefully crafted prompts, making them
difficult to scale and limiting the depth of inter-
action between search and reasoning. This often
results in suboptimal interaction. In our work, we
aim to construct an agent-based RAG pipeline that
supports flexible reasoning–search interaction, and
optimizes complex interaction trajectories via RL.
2.2 Reinforcement Learning for RAG
Reinforcement learning (RL) (Kaelbling et al.,
1996; Wiering and Van Otterlo, 2012) is an ef-
fective paradigm for enhancing the reasoning ca-
2

Policy
Model  KL  
Reference
Model
Reward
ModelGroup
ComputationTrained
Models
Frozen
ModelsGRPO
Search
Tool
RolloutSearch
Tool
Rollout
Search
ToolPolicy
ModelReasoning
ObservationEvidence
AnswerMulti-Reward
IterativeCross-Model
AnswerDifferent
Family Model
Task
MetricFigure 1: Overview of R-Search.
pabilities of LLMs. Recent studies (Guo et al.,
2025; Shao et al., 2024; Jaech et al., 2024) have
shown that RL with rule-based reward functions
enables models to acquire complex task reasoning
and self-correction abilities (Weng et al., 2022; Ku-
mar et al., 2024) without explicit intermediate su-
pervision. However, most existing RL approaches
focus mainly on internal reasoning, with limited
integration of external search information and in-
sufficient handling of multi-turn interactions be-
tween reasoning and search. Concurrent work,
such as Search-R1 (Jin et al., 2025), applies RL al-
gorithms, including Proximal Policy Optimization
(PPO) (Schulman et al., 2017) and Group Relative
Policy Optimization (GRPO) (Shao et al., 2024), to
improve LLMs’ autonomous search and reasoning
abilities. Yet challenges remain in reward design
and reasoning–search interaction. In our work, we
further explore RL-based RAG methods to enhance
LLMs’ autonomous exploration, robust reasoning,
and deep interaction with search.
3 R-Search
In this section, we introduce RL tailored for RAG
pipeline (§ 3.1) and describe the training process
(§ 3.2) of R-Search, as illustrated in Figure 1.
3.1 RL on RAG Paradigms
Advanced RAG tightly interweaves reasoning and
search through several LLM-based components, en-
abling an adaptive and active process for exploring
external knowledge to solve complex questions. In
this process, the LLM can determine its next actionbased on its current observation of the environment,
such as the reasoning trajectory and the retrieved
external knowledge observed so far. The possible
actions include initiating a new search, continuing
the reasoning process, or generating the final an-
swer. We formalize this advanced RAG paradigm
as follows:
a1⇝a2⇝···⇝aT, a t∈ {S,R},(1)
where SandRdenote the search and reasoning
actions a, respectively, and Tis the total number
of decision time steps in the action sequence.
Such a complex RAG pipeline can be viewed
as a partially observable Markov decision process
(POMDP). Therefore, we develop a novel RL-
based framework that optimizes the multi-turn in-
terleaved trajectories of reasoning and search. Dur-
ing training, we optimize the policy model by max-
imizing the following objective:
J(θ) =Eq∼D,o∼πθ(·|q;S)[rϕ(q, o)]
−βDKL[πθ(o|q;S)]∥πref(o|q;S)],(2)
where πθis the policy model, qandoare the in-
put question and generated output, and Sdenotes
search mechanism. rϕrepresents the reward func-
tion,πrefis the frozen reference model, and βis
a coefficient balancing the KL penalty. Unlike
πθ(· |q), the policy πθ(· |q;S)executes an inter-
leaved process of reasoning-search to generate the
rollout sequence, denoted as Reason ▷ ◁Search .
3

3.2 R-Search Training
3.2.1 Rollout: evidence-augmented iterative
reasoning and search
Table 1 describes the system template used for
rollout (more details in Appendix B.1). We first
prompt the LLM to generate a long Chain-of-
Thought (CoT) based on the original question q,
thereby constructing an explicit reasoning process.
During the reasoning process, we encourage the
model to trigger the search action at appropri-
ate points to acquire external non-parametric in-
formation. Whenever the model determines that
search is needed at the current reasoning state,
it generates a new search query enclosed within
<search> and</search> tags. We identify this
specific search token and feed the generated query
q∗into a search tool to retrieve top- krelevant doc-
uments Dk=d1, d2,···, dk. These documents
are wrapped with specific tokens <observation>
and</observation> , and appended to the exist-
ing reasoning trajectory. The LLM then re-engages
the reasoning process, ultimately forming an inter-
active loop between reasoning and search. Next,
when the LLM determines that the current state is
sufficiently informative to produce a final answer
α, we prompt it to rethink all previously retrieved
information and derive factual evidence ethat sup-
ports question resolution. This enables the LLM to
reason from a global perspective, leveraging all po-
tentially relevant knowledge observed from the ex-
ternal information environment to support answer
generation. Moreover, we leverage the model’s
internal reasoning capabilities to interpret and in-
tegrate external knowledge during the generation
of factual evidence. We wrap the factual evidence
with the special tokens box <original_evidence>
and</original_evidence> . After the evidence
reasoning process, the LLM continues to generate
the final answer α, enclosed within the specific
tokens box <answer> and</answer> .
3.2.2 Multi-reward modeling
Due to the high cost and potential bias associated
with training reward models using human feed-
back, we follow (Mu et al., 2024) and adopt the
rule-based reward (RBR). Considering the comple-
mentary effects of multiple rewards (Dann et al.,
2023), we design multi-dimensional, multi-stage
reward signals rϕ, including evidence rewards re
ϕ,
answer rewards rα
ϕ, and format rewards rf
ϕ.You are a helpful assistant that can solve the given
question step by step. For each step, start by explain-
ing your thought process. If additional information is
needed, provide a specific query enclosed in <search>
and</search> . The system will return the top search re-
sults within <observation> and</observation> . You
can perform multiple searches as needed. When you
know the final answer, use <original_evidence> and
</original_evidence> to provide all potentially rele-
vant original information from the observations. Ensure
the information is complete and preserves the original
wording without modification. If no searches were con-
ducted or observations were made, omit the evidence sec-
tion. Finally, provide the final answer within <answer>
and</answer> tags.
Table 1: System template. The question is appended at
the end during training and inference.
Answer reward. Metrics such as F1-score and
EM are widely used to evaluate the correctness of
model outputs. We choose the moderately strict F1-
score to construct the answer reward signal. Specif-
ically, we extract the content αpredwithin <answer>
and</answer> generated by the πθand compute
the F1-score against the gold answer αgold, which
serves as the answer reward rα
ϕ.
rα
ϕ(q, o) =F1(αpred, αgold)
=2· |αpred∩αgold|
|αpred|+|αgold|, (3)
where |αpred∩αgold|is the number of word-level
overlaps between the predicted and gold answers.
Evidence reward. In RAG systems, the quality
of evidence directly impacts answer accuracy. we
introduce models from a different family with dis-
tinct policy distributions, referred to as the cross-
family model πcf. We then use the frozen πcfto
construct an evidence reward computation pipeline
that operates on the shared evidence but performs
independent reasoning (evidence template in Ap-
pendix B.2).
αcf∼πcf(· |q, e),
re
ϕ(q, o) = F1 
αcf, αgold
(4)
First, the πcfshares the same evidence ewithπθ
and generates a Cross-Model Answer αcfbased on
q. We then apply the same answer reward com-
putation process to this Cross-Model Answer αcf.
Since αcfis produced by a different family model
than the policy model, it facilitates the mitigation
of answer bias introduced by the policy model’s
4

inherent preferences. As a result, the reward sig-
nal more objectively reflects the underlying factual
quality of the evidence.
Format reward. Format reward ensures that
generated content adheres to structural conven-
tions and remains parseable for downstream
use (Guo et al., 2025). Specifically, we en-
force that the evidence appear in exactly one
box, marked by <original_evidence> and
</original_evidence> . Similarly, we require
the final answer αto be enclosed in exactly one
<answer> and</answer> box. We formalize the
calculation pipeline of the format reward rf
ϕ:
rf
ϕ(q, o) = (1 −IS)(γe+γα·IA)
+IS(γe·IE+γα·IA),(5)
where IS,IA, andIEare indicator functions de-
noting whether retrieval is triggered, the answer is
well-formatted, and the evidence is well-formatted,
respectively. We present the formula for the overall
reward rϕ.γeandγaare reward values.
rϕ=rα
ϕ+re
ϕ+rf
ϕ(6)
s.t.rα
ϕ, re
ϕ∈[0,1], (7)
rf
ϕ∈ {0, γe, γa, γe+γa} (8)
3.2.3 Mask and non-mask strategy
Mask strategy for retrieved documents. In our
rollout sequences, we mix model-generated tokens
with externally retrieved documents. Treating the
retrieved documents Dkas part of the model’s be-
havior for loss computation can introduce gradient
noise and lead to a misalignment with the intended
optimization objective. Therefore, we introduce
a loss masking strategy that masks out the search-
derived tokens in the observation, ensuring that the
optimization objective is applied only to the tokens
generated by the policy model during reasoning.
Non-mask strategy for evidence. Only a lim-
ited number of factual segments can be incorpo-
rated into the reasoning chain, resulting in insuf-
ficient utilization of external knowledge sources.
To address this, we apply a non-masking strategy
to the evidence e, aiming to fully utilize external
knowledge and enhance the model’s capability for
knowledge integration. Specifically, in the multi-
turn reasoning-search interaction, the evidence gen-
erated by the model is based on the retrieved in-
formation and produced according to the model’spolicy distribution, making it eligible for the gra-
dient update. This strategy allows the evidence to
participate in training, guiding the model to more
effectively learn how to select, understand, and inte-
grate external knowledge. It enhances the model’s
ability to ground its reasoning on evidence.
4 Experimental Setup
4.1 Datasets & Metrics, and & Search Tools
Datasets. We conduct extensive experiments on
seven datasets, covering both complex multi-hop
and simpler single-hop QA tasks. The multi-hop
QA serves to evaluate whether R-Search can han-
dle complex logic- and knowledge-intensive ques-
tions. The single-hop QA assesses its ability to
address knowledge-intensive questions and explore
its robustness across questions with varying lev-
els of complexity. For multi-hop task, we adopt
four challenging datasets: HotpotQA (Yang et al.,
2018), 2WikiMultiHopQA (2WikiMQA) (Ho
et al., 2020), MuSiQue (Trivedi et al., 2022),
andBamboogle (Press et al., 2023). For single-
hop task, we select three factoid-based QA
datasets, including NQ(Kwiatkowski et al., 2019),
PopQA (Mallen et al., 2023), and TriviaQA (Joshi
et al., 2017). The dataset characteristics, versions,
and sizes are provided in Appendix A.1.
Evaluation Metrics. Following FLARE (Jiang
et al., 2023), we adopt two standard evaluation
metrics for QA tasks: F1-Score (EM) and Exact
Match (EM) for all datasets. EM is a more strin-
gent metric than F1, as it measures string-level
exact matches between the normalized prediction
and the golden answer.
Search Tools. Effective search actions require
appropriate retrieval sources and methods. We use
a dense retriever with the E5 model for all datasets.
For single-hop QA datasets and Bamboogle, we
use the 2018 Wikipedia dump as the corpus for re-
trieving open-domain knowledge. For the remain-
ing three multi-hop datasets, we use the Wikipedia
corpora version released by (Trivedi et al., 2023),
each aligned with its corresponding dataset.
4.2 Baselines and Backbone LLMs
In our experiments, we conduct comparisons across
five types of baselines. In Naive Generation (NG),
we evaluate the ability of LLMs to answer ques-
tions using only their internal parametric knowl-
edge. Vanilla RAG extends NG by adding a one-
5

time search step, forming the simplest retrieval-
and-generation pipeline. For Branching RAG , we
use SuRe as the baseline. Furthermore, we com-
pare with Multi-Step RAG (MSRAG), including
Iter-Retgen (Shao et al., 2023) and IRCoT (Trivedi
et al., 2023), which continuously perform iterative
reasoning and search actions to derive the final an-
swer. We also compare R-Search with the recent
advanced RAG method, Adaptive RAG (ARAG),
including FLARE (Jiang et al., 2023) and Adaptive-
RAG (Jeong et al., 2024). Unlike MSRAG, ARAG
leverages the reasoning capabilities of LLMs to
actively decide when and what to retrieve, allowing
for a more advanced and flexible agentic reason-
ing–search interaction. In addition to ARAG, we
also compare with Search-R1, a concurrent method
that leverages RL to improve the reasoning–search
interaction capability of LLMs. We categorize such
methods as RAG+RL . For Search-R1, we align the
training parameters and datasets with those used
in our method to ensure a fair comparison. For
the other baseline methods, we evaluate their per-
formance using FlashRAG (Jin et al., 2024). For
backbone LLMs, we train two open-source mod-
els via RL, i.e., Qwen-2.5-3B/7B-Instruct (Yang
et al., 2024). To construct the evidence reward, we
use a different family open-source model, Llama-
3.2-3B-Instruct (Meta, 2024), which follows a dif-
ferent policy distribution, to generate cross-model
answers based on the shared evidence.
4.3 Implementation Details
During evaluation, we align the top- kretrieval set-
ting to 5 across all methods to ensure fair compari-
son. We also employ vLLM (Kwon et al., 2023) to
accelerate inference. For GRPO and PPO training,
we use only the 2WikiMQA training set and train
on 8×A100 80GB GPUs. Primary training hyperpa-
rameters include a maximum total of 195 training
steps and a batch size of 256. Following Search-R1,
we set the retrieval top- kduring training to 3, and
configure the learning rate and warm-up ratio to
1e-6 and 0.95, respectively. During the rollout pro-
cess, we sample 5 responses for each input prompt,
with the KL divergence coefficient βset to 0.001,
and fix both γeandγαto 0.2.
5 Results and Analysis
5.1 Overall Performance
In Table 2, we present the overall performance
of various baselines and R-Search (case study inAppendix C).
R-Search facilitates deep knowledge exploration.
Compared to Vanilla RAG and LLM w/o search,
our framework achieves up to a 37.2% improve-
ment on complex multi-hop QA tasks. It also de-
livers gains of up to 4.2% on simpler single-hop
datasets that require less reasoning. These results
demonstrate that our framework effectively sup-
ports deep knowledge exploration and ensures a
stable reasoning–search interaction process.
R-Search generalizes well to questions with both
simple and complex reasoning–search require-
ments. We observe that branching RAG (e.g.,
SuRe) performs competitively on simple single-
hop QA tasks. However, its performance drops
sharply on multi-hop questions (e.g., MuSiQue)
with higher search and reasoning demands. In
contrast, our method excels at handling questions
with long reasoning chains and complex retrieval
needs, achieving up to a 45.2% improvement over
the branching RAG. These results suggest that R-
Search not only achieves strong performance but
also adapts well to questions of varying complexity.
R-Search achieves more stable logical reasoning
and more targeted, in-depth retrieval than ad-
vanced RAG variants. We compare our method
with the multi-step and ARAG methods. While
these multi-turn RAG methods help mitigate the
knowledge limitations of Vanilla RAG through
multiple searches, we observe performance insta-
bility on more complex multi-hop datasets, such
as MuSiQue. This is partly because multi-step
RAG continuously interleaves search and reason-
ing, which can introduce irrelevant passages. Al-
though adaptive RAG allows the LLM to decide
when to retrieve, it often suffers from mismatches
between the model’s internal knowledge and ac-
tual retrieval needs. In contrast, our method sup-
ports more effective and stable reasoning processes,
along with deeper search, making it better suited for
handling complex reasoning–search interactions.
R-Search effectively optimizes the interaction
between search and reasoning through multi-
reward signals, enabling stronger performance
gains on highly complex tasks. We also evaluate
against a concurrent RAG+RL approach, Search-
R1. Results show that R-Search consistently out-
performs Search-R1 in most cases. Notably, R-
Search surpasses Search-R1 by 5.6% on the highly
challenging MuSiQue dataset.
Impact analysis of using different backbone
LLMs. R-Search with the larger model gener-
6

MethodMulti-Hop QA Single-Hop QA
HotpotQA†2WikiMQA∗MuSiQue†Bamboogle†NQ†TriviaQA†PopQA†
EM F1 EM F1 EM F1 EM F1 Avg. EM F1 EM F1 EM F1 Avg. Overall Avg.
Qwen-2.5-3B-Instruct
LLM w/o Search 15.0 20.6 24.4 27.8 1.4 7.2 2.4 9.5 13.5 9.8 17.8 32.0 37.7 12.8 16.4 21.1 16.8
Vanilla RAG 34.0 43.2 33.6 38.0 5.6 11.7 9.6 19.6 24.4 37.2 46.7 58.0 66.8 39.4 46.7 49.1 35.0
SuRe 29.2 37.8 26.8 32.4 4.2 8.9 7.2 15.6 20.3 36.4 44.8 57.0 64.4 43.2 47.2 48.8 32.5
Iter-Retgen 34.4 43.4 33.2 38.2 8.2 14.8 12.0 20.2 25.5 38.0 47.4 60.2 68.5 43.2 49.4 51.1 36.5
IRCoT 39.0 50.4 35.8 46.0 9.2 17.6 23.2 33.3 31.8 23.4 35.4 45.8 56.5 31.6 41.5 39.0 34.9
FLARE 14.0 20.4 24.2 27.5 1.0 6.5 3.2 9.0 13.2 9.2 16.5 32.2 38.0 12.0 15.7 20.6 16.4
Adaptive-RAG 38.0 49.0 35.0 43.9 25.4 35.6 24.0 32.8 35.5 37.2 46.7 55.4 64.4 35.8 43.8 47.2 38.0
Search-R1 46.2 57.8 58.8 68.1 24.4 32.9 41.6 53.9 48.0 34.4 44.1 56.6 63.2 37.0 43.5 46.5 47.3
R-Search (Ours) 43.4 54.4 65.0 72.6 25.8 34.8 37.6 49.8 47.9 35.2 46.0 56.0 64.0 37.0 44.9 47.2 47.6
Qwen-2.5-7B-Instruct
LLM w/o Search 19.6 26.7 23.8 28.1 3.8 11.3 11.2 19.7 18.0 13.8 21.9 46.0 52.2 15.6 19.6 28.2 22.4
Vanilla RAG 37.4 48.1 35.4 40.5 7.2 14.6 20.8 29.7 29.2 35.0 46.7 60.0 68.5 37.6 47.6 49.2 37.8
SuRe 33.8 43.7 25.6 32.5 6.8 13.3 17.6 29.2 25.3 42.0 50.8 60.0 69.1 45.6 50.0 52.9 37.1
Iter-Retgen 42.8 53.1 37.4 43.4 10.6 19.9 22.4 31.0 32.6 37.8 48.4 61.2 69.5 38.6 46.8 50.4 40.2
IRCoT 40.4 53.7 34.2 45.5 9.0 17.3 20.0 32.3 31.6 19.6 35.5 55.2 66.2 33.0 43.6 42.2 36.1
FLARE 17.8 24.8 22.6 27.5 3.6 11.4 12.0 19.4 17.4 13.4 21.6 42.0 48.7 16.0 19.8 26.9 21.5
Adaptive-RAG 42.4 55.3 33.8 42.4 9.0 16.9 20.8 32.5 31.6 35.0 46.7 58.8 67.4 35.8 45.7 48.2 38.8
Search-R1 48.4 60.9 67.0 75.4 25.8 36.2 47.2 58.4 52.4 39.8 49.1 65.0 70.8 41.0 46.9 52.1 52.3
R-Search (Ours) 52.2 64.4 69.8 77.7 31.4 41.6 42.4 57.6 54.6 38.0 49.1 64.2 71.7 41.8 48.1 52.1 53.6
Table 2: Results (%) of overall performance. Bold and underlined values represent the highest and second-highest
results, respectively. †and∗indicate in-domain and out-of-domain datasets.
ally outperforms it with the smaller model, mainly
due to the stronger instruction-following ability of
the larger model and its richer internal knowledge.
Plus, R-Search with the smaller model still outper-
forms other RAG methods in most cases, especially
on multi-hop QA datasets. R-Search also delivers
competitive performance against Search-R1.
R-Search-as-a-Tool. The shareable evidence in
R-Search can also serve as a pluggable compo-
nent, easily transferred to other models for down-
stream answer generation. We refer to this transfer-
able functionality as R-Search-as-a-Tool ( RSTool ).
As shown in Figure 2, we evaluate the effective-
ness of applying the shared evidence to differ-
ent downstream generation models on two com-
plex multi-hop datasets. We test on both a pow-
erful black-box model, GLM-4-Plus (Zeng et al.,
2024), and an open-source model from a different
family than the policy model, Llama-3.2-3B. For
example, "RSTool + GLM-4-Plus" indicates that
GLM-4-Plus generates answers using the shareable
evidence generated by the trained Qwen-2.5-7B-
Instruct model. The results show that applying
the shareable evidence to downstream models with
different sizes and architectures consistently im-
proves performance, achieving significant gains
over Vanilla. This suggests the shared evidence cap-
tures high-quality, comprehensive knowledge effec-
tively distilled from the prior reasoning–retrieval
interaction. In practical scenarios, RSTool allows
users to offload the high-cost reasoning–search pro-
cess, often constrained by API token limits, to local
2WikiMQA MuSiQue80
70
60
50
40
30
20
10
0F1 Score (%)R-Search-as-a-Tool
RSTool+GLM4-Plus
RSTool+Qwen2.5-7B
RSTool+Llama3.2-3B
Vanilla (GLM4-Plus)
Vanilla (Llama-3.2-3B)Figure 2: R-Search-as-a-Tool.
models, substantially reducing potential overhead.
5.2 Ablation Study
In Table 3 (more results in Appendix A.2), we
conduct ablation studies to further analyze the ef-
fectiveness of the evidence mechanism. "R-Search
w/o Evidence" refers to a variant where the evi-
dence is removed from rollout, and the evidence-
related rewards in both the evidence and format
reward are disabled. We observe a clear perfor-
mance drop without the evidence component and
its associated rewards, confirming its importance.
This drop is especially pronounced on complex
multi-hop datasets compared to single-hop datasets.
On the one hand, during the reasoning stage, multi-
step search and long reasoning chains are prone to
introducing irrelevant information. Evidence helps
the LLM identify and extract the most relevant
knowledge from a global perspective, preventing it
from over-focusing on a single reasoning path and
missing other valuable clues. On the other hand, ev-
7

Figure 3: Analysis of training reward and number of valid searches on Qwen-2.5-7B/3B-Instruct models. The
semi-transparent and the solid lines indicate raw samples and the smoothed trend.
MethodMulti-Hop QA Single-Hop QA
Hot. 2Wiki MuSi. Bamb. NQ Tri. Pop.
Qwen2.5-3B-Instruct
R-Search w/o Evidence 53.4 66.6 33.3 42.3 44.6 62.3 41.6
R-Search 54.4 72.6 34.8 49.8 46.0 64.0 44.9
Qwen2.5-7B-Instruct
R-Search w/o Evidence 61.9 77.5 39.6 55.9 47.3 70.2 48.0
R-Search 64.4 77.7 41.6 57.6 49.1 71.7 48.1
Table 3: Results (%) of ablation study. "Hot." stands
for HotpotQA; other dataset names are similarly ab-
breviated. "Blue" , and "purple" are the highest and
lowest values.
MethodMulti-Hop QA Single-Hop QA
Hot. 2Wiki MuSi. Bamb. NQ Tri. Pop.
Qwen2.5-3B-Instruct
R-Search (PPO) 52.4 60.3 33.5 50.0 43.7 63.1 44.2
R-Search (GRPO) 54.4 72.6 34.8 49.8 46.0 64.0 44.9
Qwen2.5-7B-Instruct
R-Search (PPO) 58.7 68.9 37.3 53.7 47.3 70.4 45.5
R-Search (GRPO) 64.4 77.7 41.6 57.6 49.1 71.7 48.1
Table 4: PPO vs. GRPO Performance (%).
idence provides intermediate reward signals along
the long reasoning–search trajectory, guiding the
model to prioritize the reliability and completeness
of intermediate factual content, rather than relying
on speculative strategies that may yield unintended
correct answers.
5.3 Analysis
Performance and reward. We conduct experi-
ments using both PPO and GRPO on Qwen-2.5-3B-
Instruct and Qwen-2.5-7B-Instruct models. Table 4
presents the overall performance results, while Fig-
ure 3 illustrates the training dynamics and trends
of key statistics during optimization. In terms of
performance, as shown in Table 4 (more results
in Appendix A.2), both GRPO and PPO lead to
strong results. These findings indicate that both
algorithms are suitable for optimizing R-Search
and demonstrate the general applicability of our
framework. Furthermore, the results show that
GRPO generally outperforms PPO, particularly onlarger models and more complex multi-hop tasks.
A potential reason is that reward signals tend to
be sparse and delayed, making PPO more prone to
getting stuck in local optima in complex tasks. In
addition, larger models with stronger instruction-
following capabilities can execute more effective
reasoning–retrieval trajectories, leading to better
performance. During training, as shown in Figures
3(a) and 3(b), we observe that GRPO converges
faster than PPO and reaches a higher reward ceil-
ing. This is because PPO relies on an actor-critic
architecture, where the critic is unstable and re-
quires a warm-up phase in the early stage, easily
introducing noise, while GRPO bypasses the limita-
tions of value function estimation, making it more
likely to achieve a higher reward ceiling. There-
fore, we recommend using larger LLMs together
with GRPO when applying R-Search, as this setup
is more likely to result in faster convergence and
better final performance.
Number of valid searches. Figures 3(c) and
3(d) show the learning dynamics and trends in the
number of valid searches as training progresses.
We observe that the trained models tend to trigger
more retrieval steps, engaging in more rounds of
reasoning–search interaction and enabling deeper
exploration of external knowledge.
6 Conclusion
We propose R-Search, a novel RL-based RAG
framework that autonomously optimizes reasoning-
search trajectories via multi-reward signals, seam-
lessly integrating reasoning and search for com-
plex problem solving. Extensive experiments on
seven benchmarks demonstrate the superiority of R-
Search. Additionally, R-Search generates explicit
evidence when invoking external search, enabling
it to function as a modular search tool. This design
enhances the deep reasoning-search interaction and
allows efficient offloading of resource-intensive
search processes to local models.
8

Limitations
To demonstrate the strong generalization capability
of R-Search, we only use the 2WikiMQA training
dataset during the training phase. Although our
method has achieved significant performance im-
provements on both in-domain and out-of-domain
tasks under this setup, we acknowledge that in-
corporating more high-quality knowledge from di-
verse domains during training may further enhance
the model’s performance. Therefore, future work
should explore integrating more diverse and high-
quality knowledge sources to further improve the
effectiveness of R-Search.
References
Akari Asai and Hannaneh Hajishirzi. 2020. Logic-
guided data augmentation and regularization for con-
sistent question answering. In Proceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics, ACL 2020, Online, July 5-10,
2020 , pages 5642–5650. Association for Computa-
tional Linguistics.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024 . OpenReview.net.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, and 12 others. 2020. Language
models are few-shot learners. In Advances in Neural
Information Processing Systems 33: Annual Confer-
ence on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual .
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. RQ-RAG:
learning to refine queries for retrieval augmented
generation. CoRR , abs/2404.00610.
Qinyuan Cheng, Xiaonan Li, Shimin Li, Qin Zhu,
Zhangyue Yin, Yunfan Shao, Linyang Li, Tianxi-
ang Sun, Hang Yan, and Xipeng Qiu. 2024. Unified
active retrieval for retrieval augmented generation.
InFindings of the Association for Computational
Linguistics: EMNLP 2024, Miami, Florida, USA,
November 12-16, 2024 , pages 17153–17166. Associ-
ation for Computational Linguistics.
Christoph Dann, Yishay Mansour, and Mehryar Mohri.
2023. Reinforcement learning can be more efficient
with multiple rewards. In International Conference
on Machine Learning , pages 6948–6967. PMLR.DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, and 81 others.
2025. Deepseek-r1: Incentivizing reasoning capa-
bility in llms via reinforcement learning. CoRR ,
abs/2501.12948.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. CoRR , abs/2312.10997.
Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and
Jie Zhou. 2025. Deeprag: Thinking to retrieval
step by step for large language models. CoRR ,
abs/2502.01142.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint
arXiv:2501.12948 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. REALM: retrieval-
augmented language model pre-training. CoRR ,
abs/2002.08909.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. In Proceedings of the 28th International
Conference on Computational Linguistics, COLING
2020, Barcelona, Spain (Online), December 8-13,
2020 , pages 6609–6625. International Committee on
Computational Linguistics.
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richard-
son, Ahmed El-Kishky, Aiden Low, Alec Hel-
yar, Aleksander Madry, Alex Beutel, Alex Carney,
Alex Iftimie, Alex Karpenko, Alex Tachard Passos,
Alexander Neitz, Alexander Prokofiev, Alexander
Wei, Allison Tam, Ally Bennett, Ananya Kumar, and
80 others. 2024. Openai o1 system card. CoRR ,
abs/2412.16720.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. In Proceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024 , pages 7036–7050. Association for Com-
putational Linguistics.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
9

augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023, Singapore, Decem-
ber 6-10, 2023 , pages 7969–7992. Association for
Computational Linguistics.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516 .
Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang,
and Zhicheng Dou. 2024. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. CoRR , abs/2405.13576.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics, ACL
2017, Vancouver, Canada, July 30 - August 4, Volume
1: Long Papers , pages 1601–1611. Association for
Computational Linguistics.
Leslie Pack Kaelbling, Michael L Littman, and An-
drew W Moore. 1996. Reinforcement learning: A
survey. Journal of artificial intelligence research ,
4:237–285.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha,
and Jinwoo Shin. 2024. Sure: Summarizing re-
trievals using answer candidates for open-domain
QA of llms. In The Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024 . OpenReview.net.
Aviral Kumar, Vincent Zhuang, Rishabh Agarwal,
Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli,
Shariq Iqbal, Colton Bishop, Rebecca Roelofs, and
1 others. 2024. Training language models to self-
correct via reinforcement learning. arXiv preprint
arXiv:2409.12917 .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur P. Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: a benchmark for question answering
research. Trans. Assoc. Comput. Linguistics , 7:452–
466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonza-
lez, Hao Zhang, and Ion Stoica. 2023. Efficient mem-
ory management for large language model serving
with pagedattention. In Proceedings of the 29th Sym-
posium on Operating Systems Principles, SOSP 2023,
Koblenz, Germany, October 23-26, 2023 , pages 611–
626. ACM.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. In Advances in Neural In-
formation Processing Systems 33: Annual Confer-
ence on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual .
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), ACL 2023, Toronto, Canada,
July 9-14, 2023 , pages 9802–9822. Association for
Computational Linguistics.
Meta. 2024. Llama 3.2: Revolutionizing edge ai and
vision with open, customizable models.
Tong Mu, Alec Helyar, Johannes Heidecke, Joshua
Achiam, Andrea Vallone, Ian Kivlichan, Molly Lin,
Alex Beutel, John Schulman, and Lilian Weng. 2024.
Rule based rewards for language model safety. In
Advances in Neural Information Processing Systems ,
volume 37, pages 108877–108901. Curran Asso-
ciates, Inc.
OpenAI. 2023. GPT-4 technical report. CoRR ,
abs/2303.08774.
Aske Plaat, Annie Wong, Suzan Verberne, Joost
Broekens, Niki van Stein, and Thomas Bäck. 2024.
Reasoning with large language models, a survey.
CoRR , abs/2407.11511.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. In Findings of the Association for Compu-
tational Linguistics: EMNLP 2023, Singapore, De-
cember 6-10, 2023 , pages 5687–5711. Association
for Computational Linguistics.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Trans. Assoc. Comput. Linguistics ,
11:1316–1331.
Devendra Singh Sachan, Siva Reddy, William L. Hamil-
ton, Chris Dyer, and Dani Yogatama. 2021. End-to-
end training of multi-document reader and retriever
for open-domain question answering. In Advances
in Neural Information Processing Systems 34: An-
nual Conference on Neural Information Processing
Systems 2021, NeurIPS 2021, December 6-14, 2021,
virtual , pages 25968–25981.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms. arXiv preprint
arXiv:1707.06347 .
10

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. In Find-
ings of the Association for Computational Linguis-
tics: EMNLP 2023, Singapore, December 6-10, 2023 ,
pages 9248–9274. Association for Computational
Linguistics.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Y Wu, and 1 others. 2024. Deepseek-
math: Pushing the limits of mathematical reason-
ing in open language models. arXiv preprint
arXiv:2402.03300 .
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: retrieval-
augmented black-box language models. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), NAACL 2024, Mexico City, Mexico,
June 16-21, 2024 , pages 8371–8384. Association for
Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics , 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
ACL 2023, Toronto, Canada, July 9-14, 2023 , pages
10014–10037. Association for Computational Lin-
guistics.
Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu
He, Shengping Liu, Bin Sun, Kang Liu, and Jun
Zhao. 2022. Large language models are better
reasoners with self-verification. arXiv preprint
arXiv:2212.09561 .
Marco A Wiering and Martijn Van Otterlo. 2012. Rein-
forcement learning. Adaptation, learning, and opti-
mization , 12(3):729.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2.
5 technical report. arXiv preprint arXiv:2412.15115 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. In Proceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
Brussels, Belgium, October 31 - November 4, 2018 ,
pages 2369–2380. Association for Computational
Linguistics.Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented language
models robust to irrelevant context. In The Twelfth
International Conference on Learning Representa-
tions, ICLR 2024, Vienna, Austria, May 7-11, 2024 .
OpenReview.net.
Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang,
Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao,
Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun,
Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang,
Jing Zhang, Juanzi Li, Lei Zhao, and 36 others.
2024. Chatglm: A family of large language mod-
els from GLM-130B to GLM-4 all tools. CoRR ,
abs/2406.12793.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei
Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song
in the AI ocean: A survey on hallucination in large
language models. CoRR , abs/2309.01219.
Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha,
Shicheng Tan, Yuxiao Dong, and Jie Tang. 2024.
Longrag: A dual-perspective retrieval-augmented
generation paradigm for long-context question an-
swering. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
EMNLP 2024, Miami, FL, USA, November 12-16,
2024 , pages 22600–22632. Association for Computa-
tional Linguistics.
11

Appendix
A Experiment and Result
A.1 Datasets
We conduct extensive experiments on seven
datasets, covering both complex multi-hop and
simpler single-hop QA tasks. The multi-hop
QA task serves to evaluate whether R-Search can
handle complex logic- and knowledge-intensive
questions. The single-hop QA task assesses
its ability to address knowledge-intensive ques-
tions and explore its robustness across questions
with varying levels of complexity. For Multi-
hop task, we adopt four challenging multi-hop
QA datasets: HotpotQA (Yang et al., 2018),
2WikiMultiHopQA (2WikiMQA) (Ho et al.,
2020), MuSiQue (Trivedi et al., 2022), and Bam-
boogle (Press et al., 2023). The datasets require
models to start from the original question and ex-
plore a logical, knowledge-driven multi-hop reason-
ing path to answer each intermediate sub-question
and reach the final answer. We follow the train
and test splits released by (Trivedi et al., 2023)
for the first three datasets, each with 500 test sam-
ples. Bamboogle evaluation uses all 125 test sam-
ples provided by FlashRAG (Jin et al., 2024). For
Single-hop task, we select three factoid-based QA
datasets, including NQ(Kwiatkowski et al., 2019),
PopQA (Mallen et al., 2023), and TriviaQA (Joshi
et al., 2017). These datasets require models to
collect specific passages and identify key factual
information to answer the questions correctly. We
use the test sets provided by FlashRAG and ran-
domly sample 500 examples from each dataset for
evaluation.
A.2 Results
We present the results of all evaluation metrics for
the ablation study and the comparison between
PPO and GRPO in Table 6 and Table 7. The trends
of EM are consistent with those of F1.
A.3 More Implementation Details
During inference, we set the temperature to 0.1
across all models to reduce uncertainty.
B Template
B.1 System Template
We design a system template for the rollout phase
to guide the model through the complete interac-
tion process—from receiving the input question qto generating the final answer α. This template
covers four key stages: reasoning, retrieval, ev-
idence integration, and answer generation. No-
tably, for questions that do not require external
knowledge, the LLM automatically determines that
search is unnecessary. In such cases, the template
instructs the model to perform only reasoning and
answer generation. During training, we initialize
the training process with the system template and
the user’s question. It is worth noting that we do
not impose a manual separation between reason-
ing and retrieval (e.g., by encapsulating reason-
ing within a <think> and</think> tag); Instead,
we provide a high-level instruction that guides
the model to explain its thought process before
each action, without imposing any specific format
for reasoning text. This stems from the fact that
LLM generation is inherently a form of reason-
ing, where every generated token can be viewed
as part of a thought chain for question-solving.
For parts such as <search> ,<observation> ,
<original_evidence> , and <answer> , which are
distinct from general thought reasoning, we intro-
duce specific token tags to mark them, allowing the
LLM to recognize the boundaries between different
functional segments.
B.2 Evidence Template
Table 5 presents the instruction template used for
generating evidence during rollout. Before out-
putting the final answer α, we instruct the LLM to
generate evidence based on the original question
qand all previously retrieved texts (provided in
the<observation> and</observation> ). This
evidence helps the LLM rethink the retrieved infor-
mation from a global perspective and focus on key
factual knowledge. By seamlessly integrating the
evidence into the reasoning process, we facilitate a
deeper interaction between reasoning and retrieved
knowledge.
C Case Study
Tables 8 and 9 show examples of the reasoning pro-
cess from RL-trained models on 2-hop and more
complex 4-hop questions. In these cases, R-Search
uses multi-stage, multi-type rewards to improve
the reasoning–retrieval process. The model usu-
ally starts by generating a general reasoning plan,
retrieves information when needed, and uses in-
termediate conclusions to guide the next retrieval.
The evidence generated through this interaction is
12

clear, well-structured, and informative, making it
easy to transfer to downstream models for final
answer generation.
13

Answer the question based on the given passages. Only give me the answer and do
not output any other words.
The following are given passages: { evidence }
Question: { query }
Answer:
Table 5: Evidence template.
MethodMulti-Hop QA Single-Hop QA
HotpotQA 2WikiMQA MuSiQue Bamboogle NQ TriviaQA PopQA
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen-2.5-3B-Instruct
R-Search w/o Evidence 41.0 53.4 58.2 66.6 23.6 33.3 32.0 42.3 34.6 44.6 54.4 62.3 36.0 41.6
R-Search 43.4 54.4 65.0 72.6 25.8 34.8 37.6 49.8 35.2 46.0 56.0 64.0 37.0 44.9
Qwen-2.5-7B-Instruct
R-Search w/o Evidence 49.4 61.9 69.0 77.5 29.4 39.6 44.0 55.9 36.8 47.3 63.6 70.2 41.4 48.0
R-Search 52.2 64.4 69.8 77.7 31.4 41.6 42.4 57.6 38.0 49.1 64.2 71.7 41.8 48.1
Table 6: Results (%) of ablation study. "Blue" , and "purple" are the highest and lowest values.
MethodMulti-Hop QA Single-Hop QA
HotpotQA 2WikiMQA MuSiQue Bamboogle NQ TriviaQA PopQA
EM F1 EM F1 EM F1 EM F1 Avg. EM F1 EM F1 EM F1 Avg.
Qwen-2.5-3B-Instruct
R-Search (PPO) 42.0 52.42 50.8 60.3 24.0 33.5 37.6 50.0 43.8 34.0 43.7 54.6 63.1 36.8 44.2 46.1
R-Search (GRPO) 43.4 54.4 65.0 72.6 25.8 34.8 37.6 49.8 47.9 35.2 46.0 56.0 64.0 37.0 44.9 47.2
Qwen-2.5-7B-Instruct
R-Search (PPO) 47.0 58.7 59.8 68.9 27.0 37.3 40.0 53.7 49.1 37.0 47.3 63.2 70.4 37.8 45.5 50.2
R-Search (GRPO) 52.2 64.4 69.8 77.7 31.4 41.6 42.4 57.6 54.6 38.0 49.1 64.2 71.7 41.8 48.1 52.1
Table 7: PPO vs. GRPO: Performance (%) comparison.
14

Question: When was countrywide bought by the company that bought FleetBoston Financial?
"Thought":" To answer this question, we need to follow these steps:
1. Identify the company that bought FleetBoston Financial.
2. Determine when this company bought Countrywide.
Let’s start with step 1.
<search> FleetBoston Financial was bought by whom? </search>
<observation> (Title: "Bank of America") In2004, Bank ofAmer icaannounced itwould purchase Boston -based
bank Fleet Boston Financial for $47 billion in cash and stock. By merging with Bank of America, all of its banks and
branches were given the Bank of America logo. At the time of merger, FleetBoston was the seventh largest bank in
United States with $197 billion in assets, over 20 million customers and revenue of $12 billion.
(Title: "Boston") Boston is an intellectual, technological, and political center but has lost some important regional
institutions . . .
(Title: "Boston") Other important industries are financial services . . .
(Title: "The Ritz-Carlton Hotel Company") Blackstone Real Estate Acquisitions of New York bought The Ritz . . .
(Title: "RBC Bank") On June 19, 2011, PNC Financial Services agreed to buy RBC Bank USA from Royal Bank of
Canada for $3.45 billion . . .<observation>
From the observations, we can conclude that FleetBoston Financial was bought by Bank of America in 2004.
Now, let’s move to step 2.
<search> When did Bank of America buy Countrywide? </search>
<observation> (Title: "Bank of America Home Loans") On January 11, 2008, Bank of America announced that it
planned to purchase Countrywide Financial for $4.1 billion in stock. On June 5, 2008, Bank of America Corporation
announced it had received approval from the Board of Governors of the Federal Reserve System to purchase Countrywide
Financial Corporation. Then, on June 25, 2008, Countrywide announced it had received the approval of 69% of its
shareholders to the planned merger with Bank of America. OnJuly 1,2008, Bank ofAmer icaCorporationcompleted its
purchase ofCoun trywide FinancialCorporation. In 1997, Countrywide had spun off Countrywide Mortgage Investment
as an independent company called IndyMac Bank. Federal regulators seized IndyMac on July 11, 2008, after a week -
long bank run . . .<observation>
(Title: "Bank of America Home Loans") Bank of America Home Loans is the mortgage unit of Bank of America . . .
(Title: "Bank of America") In 2004, Bank of America announced it would purchase Boston - based bank FleetBoston
Financial for $47 billion in cash and stock . . .
(Title: "Bank of America") Its acquisition of investment bank Merrill Lynch in 2008 solidified it as a “Bulge Bracket
bank ” . . .
(Title: "General Tire") GenCorp sold General Tire to German tire maker Continental AG in 1987 . . .
From the observations, we can conclude that Bank of America bought Countrywide on July 1, 2008.
<original_evidence> - FleetBoston Financial was bought by Bank of America in 2004.
- Bank of America bought Countrywide on July 1, 2008. <original_evidence>
<answer> July 1, 2008 </answer> "✓"
Table 8: An 2-hop example of R-Search on MuSiQue dataset using PPO. Orange segments denote those used to
support correct answers. The " ✓" indicates a correct answer.
15

Question: Which film whose director is younger, My Baby’S Daddy or A Tale Of Winter?
"Thought":" To determine which film has a director who is younger, we need to identify the directors of both films and
compare their ages.
First, let’s identify the director of "My Baby’s Daddy".
<search> Who directed My Baby’s Daddy? </search>
<observation> (Title: "My Baby’s Daddy") MyBaby’s Daddy isa2004 Amer icancomedyfilm, directed byCheryl
Dunye.
(Title: "Alec Mapa: Baby Daddy") Alec Mapa: Baby Daddy is a 2013 one- man show by Alec Mapa . . .
(Title: "Bachelor Daddy") Bachelor Daddy is a 1941 American comedy film directed by Harold Young and written by
Robert Lees and Frederic I. Rinaldo . . .
(Title: "Baby and I") Baby and I is a 2008 South Korean film about a rebellious 18- year- old high school senior who
ends up caring for a baby with a note . . .
(Title: "Who’s Your Daddy? (film)") Who’s Your Daddy? is a 2002 American comedy film directed( and co-scripted) by
Andy Fickman. <observation>
The director of "My Baby’s Daddy" is Cheryl Dunye.
Next, let’s identify the director of "A Tale Of Winter".
<search> Who directed A Tale Of Winter? </search>
<observation> (Title: "A Tale of Winter") ATale ofWinterisa1992 French drama filmdirected byEric Rohmer, and
starring Charlotte Very, Frederic van den Driessche and Michael V oletti. It is the second of Rohmer’s "Tales of the Four
SeasonsContes des quatre saisons"), which also include "A Tale of Springtime" (1990), "A Summer’s Tale" (1996) and
"Autumn Tale" (1998) . . .
(Title: "A Tale of Winter") ATale ofWinterisa1992 French drama filmdirected byEric Rohmer, and starring Charlotte
Very, Frederic van den Driessche and Michael V oletti. It is the second of Rohmer’s" Tales of the Four SeasonsContes des
quatre saisons") . . .
. . .<observation>
The director of "A Tale of Winter" is Eric Rohmer.
Now, we need to determine the ages of Cheryl Dunye and Eric Rohmer to compare.
<search> When was Cheryl Dunye born? </search>
<observation> (Title: "Cheryl Dunye") Cheryl Dunye( born May 13,1966) isaLiberian -Amer icanfilm director,
producer, screen writer, editorandactress. Dunye’s work often concerns themes of race, sexuality, and gender, particularly
issues relating to black lesbians. . . .
. . .
Cheryl Dunye was born on May 13, 1966.
<search> When was Eric Rohmer born? </search>
<observation> (Title: "Eric Rohmer") Jean Marie Maurice Scherer or Maurice Henri Joseph Scherer, known as Eric
Rohmer (21March 192011 January 2010), was a French film director, film critic, journalist, novelist, screenwriter, and
teacher. Rohmer was the last of the post-World War II French New Wave directors to become established. He edited the
influential film journal, "Cahiers du cinema", from 1957 to 1963, while most of his . . .
(Title: "Eric Rohmer filmography") This is a list of films by the French director Eric Rohmer.
. . .<observation>
Eric Rohmer was born on March 21, 1920.
Now we compare the birth years:
- Cheryl Dunye: 1966
- Eric Rohmer: 1920
Cheryl Dunye is younger than Eric Rohmer.
Therefore, the film "My Baby’s Daddy" has a younger director.
<original_evidence> - "My Baby’s Daddy" was directed by Cheryl Dunye, born in 1966.
- "A Tale of Winter" was directed by Eric Rohmer, born in 1920.
- Cheryl Dunye is younger than Eric Rohmer. <original_evidence>
<answer> My Baby’S Daddy </answer> "✓"
Table 9: An 4-hop example of R-Search on 2WikiMQA dataset using GRPO. Orange segments denote those
used to support correct answers. The " ✓" indicates a correct answer.
16
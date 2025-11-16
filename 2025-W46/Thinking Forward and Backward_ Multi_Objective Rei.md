# Thinking Forward and Backward: Multi-Objective Reinforcement Learning for Retrieval-Augmented Reasoning

**Authors**: Wenda Wei, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Lixin Su, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Xueqi Cheng

**Published**: 2025-11-12 08:29:39

**PDF URL**: [https://arxiv.org/pdf/2511.09109v2](https://arxiv.org/pdf/2511.09109v2)

## Abstract
Retrieval-augmented generation (RAG) has proven to be effective in mitigating hallucinations in large language models, yet its effectiveness remains limited in complex, multi-step reasoning scenarios. Recent efforts have incorporated search-based interactions into RAG, enabling iterative reasoning with real-time retrieval. Most approaches rely on outcome-based supervision, offering no explicit guidance for intermediate steps. This often leads to reward hacking and degraded response quality. We propose Bi-RAR, a novel retrieval-augmented reasoning framework that evaluates each intermediate step jointly in both forward and backward directions. To assess the information completeness of each step, we introduce a bidirectional information distance grounded in Kolmogorov complexity, approximated via language model generation probabilities. This quantification measures both how far the current reasoning is from the answer and how well it addresses the question. To optimize reasoning under these bidirectional signals, we adopt a multi-objective reinforcement learning framework with a cascading reward structure that emphasizes early trajectory alignment. Empirical results on seven question answering benchmarks demonstrate that Bi-RAR surpasses previous methods and enables efficient interaction and reasoning with the search engine during training and inference.

## Full Text


<!-- PDF content starts -->

Thinking Forward and Backward:
Multi-Objective Reinforcement Learning for Retrieval-Augmented Reasoning
Wenda Wei1,2, Yu-An Liu1,2, Ruqing Zhang1,2*, Jiafeng Guo1,2*, Lixin Su3, Shuaiqiang Wang3,
Dawei Yin3, Maarten de Rijke4, Xueqi Cheng1,2
1State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China
2University of Chinese Academy of Sciences, Beijing, China
3Baidu Inc., Beijing, China
4University of Amsterdam, Amsterdam, The Netherlands
{weiwenda25z, liuyuan21b, zhangruqing, guojiafeng, cxq}@ict.ac.cn,{sulixin, wangshuaiqiang, yindawei02}@baidu.com,
m.derijke@uva.nl
Abstract
Retrieval-augmented generation (RAG) has proven to be
effective in mitigating hallucinations in large language
models, yet its effectiveness remains limited in complex,
multi-step reasoning scenarios. Recent efforts have incorpo-
rated search-based interactions into RAG, enabling iterative
reasoning with real-time retrieval. Most approaches rely on
outcome-based supervision, offering no explicit guidance
for intermediate steps. This often leads to reward hacking
and degraded response quality. We propose Bi-RAR, a novel
retrieval-augmented reasoning framework that evaluates
each intermediate step jointly in both forward and backward
directions. To assess the information completeness of each
step, we introduce a bidirectional information distance
grounded in Kolmogorov complexity, approximated via
language model generation probabilities. This quantification
measures both how far the current reasoning is from the
answer and how well it addresses the question. To optimize
reasoning under these bidirectional signals, we adopt a
multi-objective reinforcement learning framework with a
cascading reward structure that emphasizes early trajectory
alignment. Empirical results on seven question answering
benchmarks demonstrate that Bi-RAR surpasses previous
methods and enables efficient interaction and reasoning with
the search engine during training and inference.
1 Introduction
Retrieval-augmented generation (RAG) (Lewis et al. 2020)
has emerged as a prominent framework for mitigating hal-
lucination in large language models (LLMs) (Achiam et al.
2023; Gemini Team et al. 2024; Zhao et al. 2023).
Integrating RAG with reasoning.While basic RAG meth-
ods are effective, they often struggle in real-world scenarios
involving complex and heterogeneous data (Gao et al. 2023)
that require multi-hop retrieval (Hendrycks et al. 2020). To
address these limitations, recent research has increasingly
focused on enhancing RAG with advanced reasoning capa-
bilities. Specifically, LLMs can be prompted or trained to in-
*Jiafeng Guo and Ruqing Zhang are the corresponding authors.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.corporate external tools, such as search engines, into a more
dynamic and iterative reasoning process (Zhao et al. 2024).
A representative paradigm is Search-R1 (Jin et al. 2025),
which has achieved strong performance across a range of
question answering benchmarks. The key idea is to opti-
mize LLM reasoning trajectories through multi-turn search
interactions, using retrieved token masking to enable rein-
forcement learning (RL) training. The success of Search-R1
is largely attributed to its outcome-based reward function
based on the correctness of the final answer. However, this
form of supervision lacks explicit feedback for intermediate
reasoning steps, making it difficult to control the reasoning
process throughout. As a result, such optimizations may in-
duce in-context reward hacking, where the model generates
unnecessarily long or inefficient reasoning chains. These ex-
tended chains can accumulate hallucinations and ultimately
compromise the final response.Can we precisely supervise
the information understanding at each reasoning step?
Beyond unidirectional reasoning.Cognitive research has
shown that humans reason not only in a forward deduction,
from problem to solution, which reflects how the brain plans
over unknown information, but also in a backward deduc-
tion, from solution to problem (Hawes, V ostroknutov, and
Rustichini 2012). Bidirectional deductive reasoning enables
the brain to evaluate the reliability of known information and
to plan toward the unknown information, ensuring a reason-
ing process that bridges the gap between the question and the
answer. A recent study has also demonstrated that LLMs can
similarly benefit from integrating forward and backward rea-
soning in complex tasks (Chen et al. 2024). Inspired by these
findings,we explore optimizing each step through top-down
planning over unknown information and bottom-up evalua-
tion of known information.
Our method: RAG with bidirectional reasoning.We pro-
pose a novel retrieval-augmented reasoning framework, Bi-
RAR, which dynamically evaluates each reasoning step
through both forward and backward guidance to determine
whether it provides sufficient support for task-solving. To
achieve this, we need to address two key challenges.
First,how to quantify the information completeness ofarXiv:2511.09109v2  [cs.CL]  13 Nov 2025

each step from both forward and backward perspectives?
Kolmogorov complexity (Li and Vitanyi 1993), a founda-
tional concept in information theory, defines the amount
of information required to describe an object. Building on
this, information distance (Bennett et al. 1998; Vit ´anyi et al.
2009; Zhang et al. 2007) provides a universal, domain-
agnostic metric for measuring the similarity between objects
and has successfully been applied across a variety of do-
mains (Li and Vitanyi 1993; Zhang et al. 2007; Li, Zhang,
and Zhu 2008). In this work, we adopt a conditional nor-
malized information distance under specified condition pat-
terns. For forward completeness, we measure how far the
current reasoning context is from the final answer; for back-
ward completeness, we assess how well it addresses the in-
put question. By approximating Kolmogorov complexity via
language model generation probabilities, we estimate the in-
formation distance in both directions, thereby capturing the
information completeness of each step.
Second,how to optimize step-wise reasoning using
forward-backward information distances?Given the effec-
tiveness of RL (Kaelbling, Littman, and Moore 1996) in se-
quential decision-making, and the bidirectional signals in-
troduced above, we propose to use multi-objective RL meth-
ods (Roijers et al. 2013; Li, Zhang, and Wang 2020) to ex-
plore the entire preference space. Concretely, we first de-
sign a cascading reward structure that prioritizes the early
establishment of correct reasoning directions, based on the
forward and backward information distances, respectively.
These two reward signals serve as the primary supervision
for guiding RL optimization. We then train specialized mod-
els with their respective rewards independently, using group
relative policy optimization (GRPO) (Shao et al. 2024). Dur-
ing training, the model progressively learns to perform ac-
curate and efficient multi-step reasoning, dynamically deter-
mining whether and how to invoke the search engine at each
step, in order to optimize the forward or backward objective.
Finally, we obtain a balanced solution through weight-space
interpolation, which enables task-specific optimization by
selecting appropriate interpolation settings.
Experiments conducted on seven widely-used question
answering benchmarks demonstrate that Bi-RAR achieves
strong overall performance, with particularly notable im-
provements of 18.2% (Qwen2.5-3B-Instruct) and 8.3%
(Qwen2.5-3B-Base) over the strongest baseline Search-R1
(Jin et al. 2025), while using only one-fourth of Search-R1’s
training data. Further analyses show that Bi-RAR is more
effective in both training and inference.
2 Preliminaries
In this section, we review Search-R1 (Jin et al. 2025), a rep-
resentative method for enhancing retrieval-augmented gen-
eration with reasoning capabilities.
Search-R1.Recent advances, such as Search-R1, extend
RAG to support multi-step reasoning with interleaved re-
trieval. In this paradigm, given a questionQ, the LLM gen-
erates a reasoning trajectoryT={T 1, T2, . . . , T n}. At each
stepi, the LLM (i) first generates a reasoning stepT ibased
on the current information; (ii) then, it issues a search query
and retrieves relevant documents; and (iii) finally, it judgeswhether to move on to the next reasoning stepT i+1or gen-
erate the final answerAbased on the current content. The
process alternates between reasoning and search.
During training, RL is employed to encourage the LLM to
interact effectively with the search engine. A reward func-
tionr ϕevaluates the correctness of the final answer ex-
tracted from the model’s output. To ensure that the LLM
generates valid and stable search engine calls, a structured
prompting template is adopted to structure the model’s out-
put into three parts in an iterative fashion: reasoning process,
search engine calling function, and the answer. Specifically,
the RL policyπ θis optimized by:
max
πθEx∼D,y∼π θ(·|x;R) (rϕ−
βD KL[πθ(y|x;R)∥π ref(y|x;R)]),(1)
whereπ refis a reference policy,D KLis the KL-divergence
measure,βcontrols the strength of the KL penalty,Ris the
search engine,xare input samples from datasetD, andy
are the generated outputs. Andπ θ(·|x;R)denotes the policy
that generates text interleaved with the search engine.
Discussion.Search-R1 aims to teach LLMs when and how
to interact with a search engine during reasoning. However,
its outcome-based supervision focuses solely on the correct-
ness of the final answer, which can easily lead to reward
hacking by the model. This behavior is characterized by the
LLM issuing a large number of loosely relevant queries in an
attempt to improve the answer through excessive retrieval,
rather than through deliberate and coherent reasoning. As
shown in Section 5.4, this not only reduces efficiency due
to unnecessarily lengthy reasoning trajectories, but also in-
troduces redundant information that may accumulate across
steps, increasing the risk of hallucinated content and ulti-
mately derailing the reasoning process. In this paper, we ex-
plore how to provide fine-grained guidance at each interme-
diate step of the reasoning trajectory to support more effi-
cient and accurate retrieval-augmented reasoning approach.
3 Method
3.1 Overview
In this section, we present Bi-RAR, a retrieval-augmented
reasoning framework that uses bidirectional reasoning to op-
timize the intermediate steps in answering complex ques-
tions. As illustrated in Figure 1, our approach comprises two
main components: (i) Bidirectional information quantifica-
tion: at each reasoning step, we evaluate the information
distance to both the final answer and the original question,
assessing step-wise information completeness; and (ii) Mul-
ti-objective optimization: these distances serve as bidirec-
tional rewards, we use a multi-objective strategy to balance
answer-seeking and question-grounding, guiding the model
toward well-structured reasoning.
3.2 Bidirectional information quantification
Motivation.Effective multi-step reasoning requires fine-
grained supervision signals that can evaluate the quality of
each intermediate step. The central challenge is to quan-
tify the information completeness of each step, i.e., assess-

RAG
Policy Large Language Model AnswerStep 2
Reasoning
Search
EngineStep n
Reasoning
Search
EngineStep 1
Reasoning
Search
EngineQuestionReasoning 
Generation……Policy Large Language Model AnswerStep 2
Reasoning
Search
EngineStep n
Reasoning
Search
EngineStep 1
Reasoning
Search
EngineQuestionReasoning 
Generation……Policy Large Language Model AnswerReasoning
Search
EngineQuestionReasoning 
Generation
Search -R1
Bi-RARFigure 1: Framework of Bi-RAR compared with typical
RAG (Lewis et al. 2020) and Search-R1 (Jin et al. 2025).
ing whether a step meaningfully advances problem-solving
while remaining faithful to the original question.
To tackle this, we draw inspiration from Kolmogorov
complexity theory (Li and Vitanyi 1993), which offers a
domain-independent, information-theoretic foundation for
assessing semantic relevance based on minimal description
length. We propose a mechanism to provide efficient feed-
back during LLM training by quantifying the information
distance between each reasoning step and both the final an-
swer and the original question.
Information distance based on Kolmogorov complexity.
Kolmogorov complexity (Li and Vitanyi 1993) measures the
amount of information contained in an individual object.
Given a stringa, its Kolmogorov complexityK(a)is de-
fined as the length of the shortest binary program that out-
putsaunder a fixed universal computational model. The
conditional complexityK(a|c)refers to the shortest pro-
gram that generatesagiven some auxiliary input stringc,
capturing the information inathat is not already present in
c. More generally,K(a|b, c)quantifies the information re-
quired to produceawhen both stringsbandcare known.
Here, we adopt the normalized information distance
(NID) (Zhang et al. 2007), which uses Kolmogorov com-
plexity to define a universal, context-aware similarity metric
between two pieces of content. Formally, given two strings
aandbwith background context stringc, the conditional
normalized information distance betweenaandbis:
d(a, b|c) =min{K(a|b, c), K(b|a, c)}
min{K(a|c), K(b|c)}.(2)
Since Kolmogorov complexity is uncomputable (Li and Vi-
tanyi 1993), we approximate it in Eq. (2), using the genera-
tion probabilities of a language model:
K(u|v)≈ −log2PLM(u|v),
K(u|v, w)≈ −log2PLM(u|v, w),(3)
Answer 𝑨 Question 𝑸 ……The most stable mineral 
at the earth's surface?Quartz
……Information distance computation 
Step 𝑻𝒊
Reasoning
Search
EngineTo find the most  stable  mineral  at the 
Earth's  surface,  I need  to gather  information  
about  the characteristics  and stability  of 
minerals  under  surface  conditions . I'll start  
by searching  for this information .𝑑T−Q𝑇𝑖=𝑑𝑇𝑖,𝑄𝐴 𝑑T−A𝑇𝑖=𝑑𝑇𝑖,𝐴𝑄Figure 2: Sample of bidirectional distances computation.
whereP LM(u|v)andP LM(u|v, w)denote the likelihood of
generatingugiven the contextvor the joint context(v, w),
as computed by a language model. This approximation is
grounded in Shannon’s information theory (Shannon 1948),
aligning with the concept of entropy. In our implementation,
we use Qwen2.5-3B (Yang et al. 2024b) as the underlying
language model. By employing the same language model
as the generative model, this approximation can reasonably
estimate the Kolmogorov complexity, which is the short-
est program length required to generate the object given the
contextual information.
Bidirectional distances.Building on the normalized infor-
mation distance defined in Eq. (2), with conditional Kol-
mogorov complexity approximated by Eq. (3), we propose
two complementary metrics to quantify the bidirectional in-
formativeness of each reasoning step, which is generated by
the LLM based on the question, previous reasoning steps,
and retrieved documents. As shown in Figure 2, for each
stepT i, we compute:
dT-A(Ti) =d(T i, A|Q),step-to-answer distance
dT-Q(Ti) =d(T i, Q|A),step-to-question distance.(4)
For each reasoning stepT i, these two distances reflect dif-
ferent aspects of information completeness:
1.Step-to-answer distanced T-A(Ti)quantifies how much
the current stepT icontributes toward the final answer
A, indicating its solution progress; and
2.Step-to-question distanced T-Q(Ti)assesses how wellT i
remains grounded in the original questionQ, ensuring
contextual relevance and fidelity to the task.
This bidirectional formulation enables a comprehensive
evaluation of each reasoning step, allowing the model to dy-
namically balance between deep exploration and consistent
alignment with the question.
3.3 Multi-objective optimization with RL
Motivation.In this work, the LLM performs multi-step rea-
soning guided by bidirectional distances at each step. This
frames the task as a multi-objective, multi-step sequential
decision problem. To optimize this, we adopt RL to train
the entire inference sequence, with three main components:
(i) Designing bidirectional rewards derived from the infor-
mation distances to supervise training; (ii) Independently

training models with the search engine, each optimized for
a single reward, to mitigate conflicts between forward and
backward objectives; and (iii) Combining the two models
via weighted interpolation to obtain a balanced solution that
guides the model to generate reasoning steps both relevant to
the question and progressively closer to the correct answer.
Bidirectional rewards design.Based on the computed bidi-
rectional distances, we define corresponding bidirectional
reward functions. To account for the varying importance of
reasoning steps, we introduce a cascading reward structure
that prioritizes early establishment of correct reasoning di-
rections. Specifically, the forward rewardR forward and back-
ward rewardR backward are defined as:
Rforward =⊮[correct]·nX
i=1
i−1Y
j=1(1−rT-A
j)
rT-A
i,(5)
Rbackward =⊮[correct]·nX
i=1
i−1Y
j=1(1−rT-Q
j)
rT-Q
i,(6)
where⊮[correct]equals 1 if the final answer is correct, and
0 otherwise; and
rT-A
i=e−d T-A(Ti), rT-Q
i=e−d T-Q(Ti),(7)
represent the rewards derived from the step-to-answer and
step-to-question distances at stepi.
The exponential mapping ensures rewards increase as dis-
tances decrease, normalizing values between 0 and 1. The
cascading factorQi−1
j=1(1−r j)diminishes the contribution
of later steps if earlier steps already show strong alignment,
thereby encouraging efficient reasoning paths that establish
correct directions as early as possible.
Independent training with the search engine.To mitigate
convergence issues from conflicting optimization objectives
between the two rewards in early training, we initialize and
train two models independently from the same pretrained
checkpoint. Specifically, (i)θ forward is only optimized for the
forward rewardR forward ; (ii)θ backward is only optimized for
the backward rewardR backward . During training, the model
can autonomously interact with the retriever at each reason-
ing step based on its current needs. RL guides the model to
perform accurate multi-step reasoning and streamlined re-
trieval to optimize the forward or backward objective.
Each model is trained using group relative policy opti-
mization (GRPO) (Shao et al. 2024), which enhances train-
ing stability by employing group-wise baselines instead of
value networks. For each input questionx, we sampleGcan-
didate responses{y i}G
i=1from the current policyπ oldwith
the search engineR, and optimize the objective:
JGRPO(θ) =Ex∼D,{y i}G
i=1∼π old(·|x;R)"
1
GGX
i=11
|yi||yi|X
t=1
min
ri,t(θ)ˆAi,t,clip(r i,t(θ),1−ϵ,1 +ϵ) ˆAi,t
−βD KL[πθ||πref]#
,(8)ri,t(θ) =πθ(yi,t|x, y i,<t;R)
πold(yi,t|x, y i,<t;R),(9)
where ˆAi,tis the standardized advantage computed from
group-relative rewards,ϵcontrols the trust region size, and
βweights the KL penalty term.
Multi-objective optimization.After training the two mod-
elsθ forward andθ backward , we seek a balanced solution that
integrates the strengths of both reward directions. Inspired
by linear mode connectivity (Neyshabur, Sedghi, and Zhang
2020; Frankle et al. 2020), we apply linear weight interpola-
tion to combine the parameters of the two models, enabling
the resulting model to simultaneously incorporate forward
and backward reasoning capabilities. The final interpolated
modelθ Bi-RAR is defined as:
θBi-RAR = (1−λ)·θ forward +λ·θ backward , λ∈[0,1],(10)
whereλcontrols the interpolation ratio. By varyingλ, we
can explore a continuum of models that trade off between
answer accuracy and question relevance, allowing flexible
adaptation to different task requirements without the need
for additional retraining.
4 Experimental Settings
Datasets.We evaluate Bi-RAR on seven question answering
benchmarks split into two groups: (i)General QAdatasets
focus on factual questions that require accurate retrieval
and understanding of real-world knowledge, generally in-
volve single-hop reasoning: NQ (Kwiatkowski et al. 2019),
TriviaQA (Joshi et al. 2017), and PopQA (Mallen et al.
2022). (ii)Multi-hop QAdatasets are specifically designed
to evaluate a model’s ability to integrate multiple pieces of
evidence across documents to answer a question, making
them ideal for testing complex reasoning: HotpotQA (Yang
et al. 2018), 2WikiMultiHopQA (Ho et al. 2020), Musique
(Trivedi et al. 2022), and Bamboogle (Press et al. 2022).
Baselines.The baselines are grouped by how they incor-
porate retrieval into the reasoning process: (i)Reason-
ing without retrieval: These methods rely solely on the
model’s parametric knowledge to perform reasoning with-
out retrieval, including Direct inference, Chain-of-Thought
(CoT) reasoning (Wei et al. 2022), Supervised fine-tuning
(SFT) (Chung et al. 2024) and RL-based fine-tuning with-
out retrieval (R1) (Guo et al. 2025). (ii)One-step retrieval
and reasoning: These approaches retrieve external evidence
once before generating the answer, including Retrieval-Aug-
mented Generation (RAG) (Lewis et al. 2020). (iii)Mul-
ti-step retrieval and reasoning: These methods perform
iterative retrieval interleaved with reasoning, enabling the
model to gather new information at each step, including IR-
CoT (Trivedi et al. 2023), Search-o1 (Li et al. 2025a), and
Search-R1 (Jin et al. 2025) trained with GRPO. All base-
line results are taken from Search-R1. To ensure a fair com-
parison, all methods use the same retriever, retrieval setting,
knowledge corpus, training dataset, and pre-trained LLMs.
Model variants.Bi-RAR includes two variants: (i)For-
ward-RAR, trained only with the forward rewardR forward ,
as inθ forward ; and (ii)Backward-RAR, trained only with

General QA Multi-Hop QA
MethodsNQ TriviaQA PopQA HotpotQA 2Wiki Musique Bamboogle Avg.
Reasoning without retrieval
Direct Inference 0.106 0.288 0.108 0.149 0.244 0.020 0.024 0.134
CoT 0.023 0.032 0.005 0.021 0.021 0.002 0.000 0.015
SFT 0.249 0.292 0.104 0.186 0.248 0.044 0.112 0.176
R1-base 0.226 0.455 0.173 0.201 0.268 0.055 0.224 0.229
R1-instruct 0.210 0.449 0.171 0.208 0.275 0.060 0.192 0.224
One-step reasoning with retrieval
RAG 0.348 0.544 0.387 0.255 0.226 0.047 0.080 0.270
Multi-step reasoning with retrieval
IRCoT 0.111 0.312 0.200 0.164 0.171 0.067 0.240 0.181
Search-o1 0.238 0.472 0.262 0.221 0.218 0.054 0.320 0.255
Search-R1-base 0.421 0.583 0.413 0.297 0.274 0.066 0.128 0.312
Search-R1-instruct 0.397 0.565 0.391 0.331 0.310 0.124 0.232 0.336
Bi-RAR-base0.442 0.614 0.4320.317 0.297 0.073 0.188 0.338
Bi-RAR-instruct 0.438 0.608 0.4210.391 0.402 0.153 0.363 0.397
Table 1: Main results of Bi-RAR and baselines on QA benchmarks. The best performance is highlighted in bold.
the backward rewardR backward , as inθ backward . For these two
variants, we only train a single model using the correspond-
ing rewards, without performing interpolation.
Implementation details.We use both the Qwen2.5-3B-
Base and Qwen2.5-3B-Instruct model (Yang et al. 2024b).
Following Search-R1 (Jin et al. 2025), we train our model
on a combined training set of NQ and HotpotQA, adopt the
same training and evaluation prompt template as Search-R1,
and useExact Match (EM)as the evaluation metric. For
retrieval, we adopt the 2018 Wikipedia dump (Karpukhin
et al. 2020) as the knowledge source and use E5 (Wang et al.
2022) to simulate a search engine.
The training batch size is set to 128, and the validation
batch size is set to 256, using only one-fourth of the train-
ing data compared to Search-R1. To manage memory usage
efficiently, we use gradient checkpointing and fully sharded
data parallel (FSDP) with CPU offloading. For efficient re-
sponse generation, we use vLLM with a tensor parallel size
of 1 and a GPU memory utilization ratio of 0.6. Sampling is
performed with a temperature of 1.0 and top-p of 1.0. We set
the KL divergence regularization coefficient toβ= 0.001,
and the clipping ratio toϵ= 0.2. In GRPO training, we
follow the implementation from Verl (Sheng et al. 2025).
Training runs for 200 steps. We set the policy model’s learn-
ing rate to 1e-6 and sample 5 responses per prompt. In
multi-objective optimization, we testedλvalues of 0.25, 0.5,
and 0.75, which emphasize different objectives. We select
λ= 0.25for both the Qwen2.5-3B-Base and Qwen2.5-3B-
Instruct models as it achieved the best performance.
5 Experimental Results
In this section, we report the experimental results to demon-
strate the effectiveness of Bi-RAR.
5.1 Main results
Table 1 presents the overall performance of Bi-RAR com-
pared to baseline methods across seven question answering
benchmarks. Observations on the baselines are: (i) Overall,models equipped with retrievers achieve better performance
than those without, indicating that access to external knowl-
edge sources can effectively complement the model’s inter-
nal knowledge. (ii) Models perform better on general QA
datasets than multi-hop QA datasets. This discrepancy indi-
cates that multi-hop reasoning and evidence aggregation re-
main challenging for the models. (iii) Among the baselines,
Search-R1 performs best, benefiting from iterative retrieval
and outcome-based supervision that improve accuracy.
When we look at Bi-RAR, we find that: (i) Bi-RAR
achieves the best overall performance among all evaluated
models, with an average relative improvement of 18.2% and
8.3% over the strongest baseline when using Qwen2.5-3B
Instruct and Base, respectively. This demonstrates that
our multi-objective optimization approach based on bidi-
rectional information quantification supervision effectively
constrains the reasoning trajectory, guiding the model to
generate more accurate and compact answers. (ii) Compared
to the strongest baseline Search-R1, Bi-RAR delivers con-
sistent gains across diverse datasets, despite being trained on
only one-fourth of Search-R1’s training data. For example,
Bi-RAR improves the performance on HotpotQA by 0.06
and 2Wiki by 0.092, corresponding to 18.1% and 29.7% rel-
ative increase, under the instruct-tuned model. This indicates
that bidirectional distances offer precise step-level optimiza-
tion signals, leading to more efficient training and better in-
ference quality. (iii) Bi-RAR demonstrates effectiveness on
both base and instruction-tuned models, suggesting strong
generalization across model types.
5.2 Ablation study
We conduct ablation experiments comparing the variants of
Bi-RAR. The results shown in Table 2 demonstrate that:
(i) Forward-RAR performs better than Backward-RAR, with
relative improvements of 3.7% and 1.9% on the base and in-
struct variants. This result indicates that reward signals prop-
agated from the answer side contribute more directly to final
answer correctness. This aligns with our intuition, as anchor-
ing generation on the expected answer better constrains the

MethodsNQ TriviaQA PopQA HotpotQA 2Wiki Musique Bamboogle Avg.
Qwen2.5-3B-Base
Forward-RAR-base 0.440 0.6130.4350.313 0.293 0.066 0.169 0.333
Backward-RAR-base 0.435 0.599 0.423 0.313 0.282 0.069 0.125 0.321
Bi-RAR-base0.442 0.6140.4320.317 0.297 0.073 0.188 0.338
Qwen2.5-3B-Instruct
Forward-RAR-instruct 0.432 0.598 0.418 0.376 0.375 0.144 0.347 0.384
Backward-RAR-instruct 0.436 0.602 0.391 0.380 0.339 0.145 0.347 0.377
Bi-RAR-instruct0.438 0.608 0.421 0.391 0.402 0.153 0.363 0.397
Table 2: Ablation results of forward, backward, and bidirectional reasoning in Bi-RAR with different backbone LLMs.
0 50 100 150 200
Training Steps600800100012001400Response Length
Qwen2.5-3B-Base
0 50 100 150 200
Training Steps600800100012001400
Qwen2.5-3B-Instruct
0 50 100 150 200
Training Steps0.00.10.20.30.40.50.6Training Reward
Qwen2.5-3B-Base
0 50 100 150 200
Training Steps0.00.10.20.30.40.50.6
Qwen2.5-3B-Instruct
(a) Comparison of response lengths (b) Comparison of training rewardsForward-RAR Backward-RAR Search-R1
Figure 3: Trends in response lengths and rewards change during RL training for Forward/Backward-RAR and Search-R1.
reasoning path. (ii) Bi-RAR achieves the best performance
across most datasets under both base and instruct backbone
models. This demonstrates the effectiveness of our multi-
-objective optimization framework in integrating forward
and backward objectives, allowing the model to incorporate
complementary reasoning signals and achieve stronger over-
all accuracy.
5.3 Training analysis
We compare the training dynamics of Forward-RAR,
Backward-RAR, and Search-R1 on both the Qwen2.5-3b-
Base and Qwen2.5-3b-Instruct models, focusing on re-
sponse length and train reward trends.
Response length.As shown in Figure 3(a), on models
initialized from the base model, the response lengths of
Forward-RAR and Backward-RAR decrease faster than
Search-R1. This demonstrates that the cascading reward
structure, which emphasizes early trajectory alignment,
leads to more efficient reasoning by guiding the model
to eliminate unnecessary steps early in training. On the
instruction-tuned models, all methods show an initial in-
crease followed by a decrease in response length. This is
because the instruct model has a stronger instruction fol-
lowing ability, initially attempts to find correct answers via
longer reasoning chains. As training progresses, the model
learns that shorter responses can omit redundant steps while
improving answer accuracy, leading to a response length
reduction. By the end of training, both Forward-RAR and
Backward-RAR produce shorter responses than Search-R1
on the instruct model, indicating that our forward and back-
ward information distance supervision effectively guides the
NQ
TriviaQAPopQA
HotpotQA2WikiMusiqueBamboogleAvg.70080090010001100120013001400Response Length
Search-R1 (Response Length)
Bi-RAR (Response Length)Search-R1 (Search Calls)
Bi-RAR (Search Calls)1.01.21.41.61.82.02.22.4
Search Calls
Figure 4: Response lengths and search calls in inference.
model to generate accurate and concise reasoning steps.
Rewards.As shown in Figure 3(b), Forward-RAR and
Backward-RAR converge faster than Search-R1 on both
Base and Instruct models. This suggests that forward and
backward reward signals provide more precise guidance
for optimizing each step, enabling more effective training
supervision. On instruction-tuned models, the rewards of
Forward/Backward-RAR exhibit no severe fluctuations as
observed in Search-R1 during training, which reflects the ro-
bustness and consistency of our bidirectional reward design.
5.4 Inference analysis
To analyze the response efficiency during the inference
phase, we compare Bi-RAR and Search-R1 in terms of re-
sponse length and number of search calls, both of which di-
rectly affect inference efficiency. We used the Qwen2.5-3B-

Instruct model for comparison, with similar observations on
other backbones. The results across seven datasets and their
average are shown in Figure 4.
We observe that: (i) For response length, Bi-RAR gen-
erates shorter responses than Search-R1 on most datasets,
with the reduction notable on the general QA datasets. This
is attributed to the cascading reward structure that empha-
sizes early trajectory alignment, enabling Bi-RAR to gen-
erate more concise and less redundant responses. (ii) For
search calls, Bi-RAR reduces the number of retrievals com-
pared to Search-R1 across all datasets. This is because the
bidirectional distances supervision mitigates invalid reason-
ing paths and corresponding redundant searches, leading
to more efficient inference. (iii) On 2WikiMultiHopQA,
Bi-RAR produces longer responses than Search-R1 while
using fewer search calls. This is due to the complex multi-
-hop reasoning required by the dataset, where our bidirec-
tional supervision better guides the model to maintain co-
herent long-range inference with fewer but more targeted re-
trievals. As a result, Bi-RAR achieves a substantial relative
performance gain of 29.7%.
6 Related Work
Retrieval-augmented generation.Retrieval-augmented
generation (Lewis et al. 2020) is a widely adopted frame-
work that enhances large language models (LLMs) (Achiam
et al. 2023; Gemini Team et al. 2024; Zhao et al. 2023) by
incorporating external knowledge sources. This technique
effectively reduces hallucination (Zhang et al. 2023; Liu
et al. 2026, 2025a,b) and improves task performance (Gao
et al. 2023; Shuster et al. 2021; Jiang et al. 2023; Shi
et al. 2025; Li et al. 2025b). Building on this foundation,
many studies have explored improving the performance
of RAG systems by optimising prompts or training ob-
jectives, such as Self-RAG, REPLUG, and RA-DIT (Asai
et al. 2023; Shi et al. 2024; Lin et al. 2023). However,
this single-round framework of retrieval-then-answering
makes LLMs difficult to capture sufficient information and
perform complete reasoning, leading to poor performance
in handling complex multi-hop reasoning tasks. To address
this, recent approaches incorporate multi-step reasoning
and retrieval, retrieval-augmented reasoning, to further
enhance the model’s capability in complex scenarios:
(i) IRCoT (Trivedi et al. 2023) interleaves retrieval within
the chain-of-thought reasoning process; (ii) Search-o1 (Li
et al. 2025a) enhances LLMs by integrating agentic search
capabilities that dynamically retrieve and incorporate exter-
nal knowledge during the reasoning process; (iii) Search-R1
(Jin et al. 2025) uses reinforcement learning to train LLMs
to autonomously generate search queries and use real-time
retrieval during step-by-step reasoning.
These methods are primarily guided by the final answer,
encouraging LLMs to interact more with search engines.
However, such a “distant” supervision signal cannot provide
precise guidance for each interaction, leading to over or dis-
torted reasoning directions by LLMs. An effective reasoning
trajectory should continuously progress toward the solution
while remaining grounded in the original problem context.Therefore, we propose a bidirectional information quan-
tification to define the optimization objective for each rea-
soning step, enabling LLMs to determine the reasoning di-
rection based on the current information sufficiency.
LLMs and reinforcement learning.Reinforcement learn-
ing (Kaelbling, Littman, and Moore 1996) has fundamen-
tally reshaped how we align LLMs with human preferences,
evolving from computationally intensive strategies to more
elegant and efficient solutions. Early implementations such
as PPO required both a reward and a critic model (Ouyang
et al. 2022; Schulman et al. 2017). DPO simplified this pro-
cess by removing the reward model and directly optimiz-
ing on preference data (Rafailov et al. 2023), while GRPO
further simplifies the pipeline by dropping the critic model
and using sampled responses to estimate advantages (Shao
et al. 2024). These advances have significantly enhanced
LLM reasoning capabilities, as evidenced by models such
as OpenAI’s o1, DeepSeek-R1 and Qwen2.5 (Jaech et al.
2024; Guo et al. 2025; Yang et al. 2024b).
In practice, LLM training often involves multiple opti-
mization objectives that need to be effectively balanced dur-
ing reinforcement learning. Multi-objective reinforcement
learning (MORL) (Barrett and Narayanan 2008; Roijers
et al. 2013; Li, Zhang, and Wang 2020) extends standard RL
by replacing the single scalar reward signal with multiple
feedback signals, each corresponding to a different objec-
tive. Recent work (Rame et al. 2023) has begun exploring
multi-objective optimization for LLMs.
In this paper, we adopt a multi-objective reinforcement
learning approach to simultaneously optimize the forward
and backward objectives in our framework to achieve an ef-
fective balanced solution.
7 Conclusion
We have proposed Bi-RAR, a novel retrieval-augmented rea-
soning framework designed to enhance multi-step reason-
ing ability of LLMs. We introduce bidirectional informa-
tion quantification grounded in Kolmogorov complexity the-
ory, which jointly measures how far each reasoning step is
from the final answer and how well it addresses the orig-
inal question. To effectively use these signals, we adopt a
multi-objective reinforcement learning framework, enabling
a smooth trade-off between the two objectives. Experiments
demonstrate that bidirectional reasoning guidance can sig-
nificantly improve the accuracy of LLMs in solving complex
problems, while achieving efficient interaction and reason-
ing with the search engine during training and inference.
Broader impact and limitations.We aim to make an ini-
tial exploration into multi-step retrieval-augmented reason-
ing, and to inspire the community to further advance this
line of research. As to the limitations of our work, we ap-
proximate Kolmogorov complexity using generation proba-
bilities from an LLM when computing bidirectional infor-
mation quantification, which could be time-consuming. In
future work, we plan to explore more efficient approxima-
tion methods and extend our framework to larger models. In-
vestigating efficient reasoning through low-resource model
training in complex real-world search scenarios represents
another promising direction.

Acknowledgments
This work was funded by the Strategic Priority Research
Program of the CAS under Grants No. XDB0680102, the
National Natural Science Foundation of China (NSFC)
under Grants No. 62472408 and 62441229, the National
Key Research and Development Program of China un-
der Grants No. 2023YFA1011602, the Dutch Research
Council (NWO), under project numbers 024.004.022,
NWA.1389.20.183, and KICH3.LTP.20.006, and the Eu-
ropean Union under grant agreements No. 101070212
(FINDHR) and No. 101201510 (UNITE). All content rep-
resents the opinion of the authors, which is not necessar-
ily shared or endorsed by their respective employers and/or
sponsors.
References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.;
Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.;
Anadkat, S.; et al. 2023. GPT-4 Technical Report.arXiv
preprint arXiv:2303.08774.
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H.
2023. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. InICLR.
Barrett, L.; and Narayanan, S. 2008. Learning All Optimal
Policies with Multiple Criteria. InProceedings of the 25th
international conference on Machine learning, 41–47.
Bennett, C. H.; G ´acs, P.; Li, M.; Vit ´anyi, P. M.; and Zurek,
W. H. 1998. Information Distance.IEEE Transactions on
information theory, 44(4): 1407–1423.
Chen, J. C.-Y .; Wang, Z.; Palangi, H.; Han, R.; Ebrahimi, S.;
Le, L.; Perot, V .; Mishra, S.; Bansal, M.; Lee, C.-Y .; et al.
2024. Reverse Thinking Makes LLMs Stronger Reasoners.
arXiv preprint arXiv:2411.19865.
Chung, H. W.; Hou, L.; Longpre, S.; Zoph, B.; Tay, Y .;
Fedus, W.; Li, Y .; Wang, X.; Dehghani, M.; Brahma, S.;
et al. 2024. Scaling Instruction-finetuned Language Mod-
els.Journal of Machine Learning Research, 25(70): 1–53.
Frankle, J.; Dziugaite, G. K.; Roy, D.; and Carbin, M. 2020.
Linear Mode Connectivity and the Lottery Ticket Hypoth-
esis. InInternational Conference on Machine Learning,
3259–3269. PMLR.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented
Generation for Large Language Models: A Survey.arXiv
preprint arXiv:2312.10997, 2(1).
Gemini Team; Georgiev, P.; Lei, V . I.; Burnell, R.; Bai,
L.; Gulati, A.; Tanzer, G.; Vincent, D.; Pan, Z.; Wang, S.;
et al. 2024. Gemini 1.5: Unlocking Multimodal Understand-
ing across Millions of Tokens of Context.arXiv preprint
arXiv:2403.05530.
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. DeepSeek-R1:
Incentivizing Reasoning Capability in LLMs via Reinforce-
ment Learning.arXiv preprint arXiv:2501.12948.
Hawes, D. R.; V ostroknutov, A.; and Rustichini, A. 2012.
Experience and Abstract Reasoning in Learning Backward
Induction.Frontiers in neuroscience, 6: 23.Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika,
M.; Song, D.; and Steinhardt, J. 2020. Measuring Mas-
sive Multitask Language Understanding.arXiv preprint
arXiv:2009.03300.
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa, A.
2020. Constructing a Multi-hop QA Dataset for Com-
prehensive Evaluation of Reasoning Steps.arXiv preprint
arXiv:2011.01060.
Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky,
A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney,
A.; et al. 2024. OpenAI o1 System Card.arXiv preprint
arXiv:2412.16720.
Jiang, Z.; Xu, F. F.; Gao, L.; Sun, Z.; Liu, Q.; Dwivedi-Yu, J.;
Yang, Y .; Callan, J.; and Neubig, G. 2023. Active Retrieval
Augmented Generation. InEMNLP, 7969–7992.
Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.; Za-
mani, H.; and Han, J. 2025. Search-R1: Training LLMs to
Reason and Leverage Search Engines with Reinforcement
Learning.arXiv preprint arXiv:2503.09516.
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L.
2017. TriviaQA: A Large Scale Distantly Supervised Chal-
lenge Dataset for Reading Comprehension.arXiv preprint
arXiv:1705.03551.
Kaelbling, L. P.; Littman, M. L.; and Moore, A. W. 1996.
Reinforcement Learning: A Survey.Journal of Artificial In-
telligence Research, 4: 237–285.
Karpukhin, V .; Oguz, B.; Min, S.; Lewis, P. S.; Wu, L.;
Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Pas-
sage Retrieval for Open-Domain Question Answering. In
EMNLP (1), 6769–6781.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; et al. 2019. Natural Questions: A Benchmark for
Question Answering Research.Transactions of the Associ-
ation for Computational Linguistics, 7: 453–466.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented Generation for
Knowledge-intensive NLP Tasks.Advances in neural in-
formation processing systems, 33: 9459–9474.
Li, F.; Zhang, X.; and Zhu, X. 2008. Answer Validation
by Information Distance Calculation. InColing 2008: Pro-
ceedings of the 2nd workshop on Information Retrieval for
Question Answering, 42–49.
Li, K.; Zhang, T.; and Wang, R. 2020. Deep Reinforcement
Learning for Multiobjective Optimization.IEEE Transac-
tions on Cybernetics, 51(6): 3103–3114.
Li, M.; and Vitanyi, P. 1993.An Introduction to Kolmogorov
Complexity and its Applications. New York: Springer-
Verlag.
Li, X.; Dong, G.; Jin, J.; Zhang, Y .; Zhou, Y .; Zhu, Y .; Zhang,
P.; and Dou, Z. 2025a. Search-o1: Agentic Search-enhanced
Large Reasoning Models.arXiv preprint arXiv:2501.05366.
Li, Y .; Cai, H.; Kong, R.; Chen, X.; Chen, J.; Yang, J.;
Zhang, H.; Li, J.; Wu, J.; Chen, Y .; et al. 2025b. Towards
AI Search Paradigm.arXiv preprint arXiv:2506.17188.

Lin, X. V .; Chen, X.; Chen, M.; Shi, W.; Lomeli, M.; James,
R.; Rodriguez, P.; Kahn, J.; Szilvasy, G.; Lewis, M.; et al.
2023. Ra-dit: Retrieval-augmented Dual Instruction Tuning.
InThe Twelfth International Conference on Learning Rep-
resentations.
Liu, Y .-A.; Zhang, R.; Guo, J.; de Rijke, M.; Fan, Y .; and
Cheng, X. 2025a. Attack-in-the-Chain: Bootstrapping Large
Language Models for Attacks against Black-box Neural
Ranking Models. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 39, 12229–12237.
Liu, Y .-A.; Zhang, R.; Guo, J.; de Rijke, M.; Fan, Y .; and
Cheng, X. 2025b. On the Scaling of Robustness and Ef-
fectiveness in Dense Retrieval. InProceedings of the 48th
International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, 2351–2360.
Liu, Y .-A.; Zhang, R.; Guo, J.; de Rijke, M.; Fan, Y .; and
Cheng, X. 2026. Robust Neural Information Retrieval:
An Adversarial and Out-of-distribution Perspective.ACM
Transactions on Information Systems.
Mallen, A.; Asai, A.; Zhong, V .; Das, R.; Khashabi, D.;
and Hajishirzi, H. 2022. When Not to Trust Language
Models: Investigating Effectiveness of Parametric and Non-
parametric Memories.arXiv preprint arXiv:2212.10511.
Neyshabur, B.; Sedghi, H.; and Zhang, C. 2020. What is Be-
ing Transferred in Transfer Learning?Advances in Neural
Information Processing Systems, 33: 512–523.
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.;
Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.;
et al. 2022. Training Language Models to Follow Instruc-
tions with Human Feedback.Advances in neural informa-
tion processing systems, 35: 27730–27744.
Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N. A.;
and Lewis, M. 2022. Measuring and Narrowing the Com-
positionality Gap in Language Models.arXiv preprint
arXiv:2210.03350.
Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Er-
mon, S.; and Finn, C. 2023. Direct Preference Optimiza-
tion: Your Language Model is Secretly a Reward Model.
Advances in Neural Information Processing Systems, 36:
53728–53741.
Rame, A.; Couairon, G.; Dancette, C.; Gaya, J.-B.; Shukor,
M.; Soulier, L.; and Cord, M. 2023. Rewarded Soups: To-
wards Pareto-optimal Alignment by Interpolating Weights
Fine-tuned on Diverse Rewards.Advances in Neural Infor-
mation Processing Systems, 36: 71095–71134.
Roijers, D. M.; Vamplew, P.; Whiteson, S.; and Dazeley, R.
2013. A Survey of Multi-objective Sequential Decision-
making.Journal of Artificial Intelligence Research, 48: 67–
113.
Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and
Klimov, O. 2017. Proximal Policy Optimization Algorithms.
arXiv preprint arXiv:1707.06347.
Shannon, C. E. 1948. A Mathematical Theory of Communi-
cation.The Bell System Technical Journal, 27(3): 379–423.Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang,
H.; Zhang, M.; Li, Y .; Wu, Y .; et al. 2024. DeepSeek-
Math: Pushing the Limits of Mathematical Reasoning in
Open Language Models.arXiv preprint arXiv:2402.03300.
Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang,
R.; Peng, Y .; Lin, H.; and Wu, C. 2025. HybridFlow: A
Flexible and Efficient RLHF Framework. InProceedings of
the Twentieth European Conference on Computer Systems,
1279–1297.
Shi, W.; Min, S.; Yasunaga, M.; Seo, M.; James, R.; Lewis,
M.; Zettlemoyer, L.; and Yih, W.-t. 2024. Replug: Retrieval-
augmented Black-box Language Models. InProceedings
of the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), 8371–8384.
Shi, Y .; Li, S.; Wu, C.; Liu, Z.; Fang, J.; Cai, H.; Zhang, A.;
and Wang, X. 2025. Search and Refine during Think: Au-
tonomous Retrieval-augmented Reasoning of LLMs.arXiv
preprint arXiv:2505.11277.
Shuster, K.; Poff, S.; Chen, M.; Kiela, D.; and Weston, J.
2021. Retrieval Augmentation Reduces Hallucination in
Conversation. In Moens, M.-F.; Huang, X.; Specia, L.; and
Yih, S. W.-t., eds.,Findings of the Association for Computa-
tional Linguistics: EMNLP 2021, 3784–3803. Punta Cana,
Dominican Republic: Association for Computational Lin-
guistics.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics, 10: 539–554.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabhar-
wal, A. 2023. Interleaving Retrieval with Chain-of-thought
Reasoning for Knowledge-intensive Multi-step Questions.
InProceedings of the 61st annual meeting of the associa-
tion for computational linguistics (volume 1: long papers),
10014–10037.
Vit´anyi, P. M.; Balbach, F. J.; Cilibrasi, R. L.; and Li, M.
2009. Normalized Information Distance. InInformation
Theory and Statistical Learning, 45–82. Springer.
Wang, L.; Yang, N.; Huang, X.; Jiao, B.; Yang, L.; Jiang,
D.; Majumder, R.; and Wei, F. 2022. Text Embeddings by
Weakly-supervised Contrastive Pre-training.arXiv preprint
arXiv:2212.03533.
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.;
Chi, E.; Le, Q. V .; Zhou, D.; et al. 2022. Chain-of-thought
Prompting Elicits Reasoning in Large Language Models.
Advances in Neural Information Processing Systems, 35:
24824–24837.
Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.;
Li, C.; Liu, D.; Huang, F.; Wei, H.; et al. 2024b. Qwen2.5
Technical Report.arXiv preprint arXiv:2412.15115.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA:
A Dataset for Diverse, Explainable Multi-hop Question An-
swering.arXiv preprint arXiv:1809.09600.

Zhang, X.; Hao, Y .; Zhu, X.; Li, M.; and Cheriton, D. R.
2007. Information Distance from a Question to an An-
swer. InProceedings of the 13th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery and Data Min-
ing, 874–883.
Zhang, Y .; Li, Y .; Cui, L.; Cai, D.; Liu, L.; Fu, T.; Huang, X.;
Zhao, E.; Zhang, Y .; Chen, Y .; et al. 2023. Siren’s Song in
the AI Ocean: A Survey on Hallucination in Large Language
Models.arXiv preprint arXiv:2309.01219.
Zhao, P.; Zhang, H.; Yu, Q.; Wang, Z.; Geng, Y .; Fu, F.;
Yang, L.; Zhang, W.; Jiang, J.; and Cui, B. 2024. Retrieval-
augmented Generation for AI-generated Content: A Survey.
arXiv preprint arXiv:2402.19473.
Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y .;
Min, Y .; Zhang, B.; Zhang, J.; Dong, Z.; Du, Y .; Yang, C.;
Chen, Y .; Chen, Z.; Jiang, J.; Ren, R.; Li, Y .; Tang, X.; Liu,
Z.; Liu, P.; Nie, J.-Y .; and Wen, J.-R. 2023. A Survey of
Large Language Models.arXiv preprint arXiv:2303.18223.
# Careful Queries, Credible Results: Teaching RAG Models Advanced Web Search Tools with Reinforcement Learning

**Authors**: Yuqin Dai, Shuo Yang, Guoqing Wang, Yong Deng, Zhanwei Zhang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Changhua Meng, Can Yi, Yuchen Zhou, Weiqiang Wang, Shuai Lu

**Published**: 2025-08-11 13:08:37

**PDF URL**: [http://arxiv.org/pdf/2508.07956v1](http://arxiv.org/pdf/2508.07956v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
integrating up-to-date external knowledge, yet real-world web environments
present unique challenges. These limitations manifest as two key challenges:
pervasive misinformation in the web environment, which introduces unreliable or
misleading content that can degrade retrieval accuracy, and the
underutilization of web tools, which, if effectively employed, could enhance
query precision and help mitigate this noise, ultimately improving the
retrieval results in RAG systems. To address these issues, we propose
WebFilter, a novel RAG framework that generates source-restricted queries and
filters out unreliable content. This approach combines a retrieval filtering
mechanism with a behavior- and outcome-driven reward strategy, optimizing both
query formulation and retrieval outcomes. Extensive experiments demonstrate
that WebFilter improves answer quality and retrieval precision, outperforming
existing RAG methods on both in-domain and out-of-domain benchmarks.

## Full Text


<!-- PDF content starts -->

Careful Queries, Credible Results: Teaching RAG Models Advanced Web Search
Tools with Reinforcement Learning
Yuqin Dai1,5*, Shuo Yang2*, Guoqing Wang5*, Yong Deng5, Zhanwei Zhang3,5,
Jun Yin1, Pengyu Zeng1, Zhenzhe Ying5, Changhua Meng5,
Can Yi5, Yuchen Zhou4, Weiqiang Wang5, Shuai Lu1
1Tsinghua University,2The University of Hong Kong,3Zhejiang University,
4National University of Singapore,5Ant. Group
Abstract
Retrieval-Augmented Generation (RAG) enhances large lan-
guage models (LLMs) by integrating up-to-date external
knowledge, yet real-world web environments present unique
challenges. These limitations manifest as two key challenges:
pervasive misinformation in the web environment, which in-
troduces unreliable or misleading content that can degrade
retrieval accuracy, and the underutilization of web tools,
which, if effectively employed, could enhance query preci-
sion and help mitigate this noise, ultimately improving the
retrieval results in RAG systems. To address these issues, we
propose WebFilter, a novel RAG framework that generates
source-restricted queries and filters out unreliable content.
This approach combines a retrieval filtering mechanism with
a behavior- and outcome-driven reward strategy, optimizing
both query formulation and retrieval outcomes. Extensive ex-
periments demonstrate that WebFilter improves answer qual-
ity and retrieval precision, outperforming existing RAG meth-
ods on both in-domain and out-of-domain benchmarks. Code
is available at https://github.com/GuoqingWang1/WebFilter.
Introduction
The advancement of large language models (LLMs) has
driven significant progress across both industrial and aca-
demic fields (Qiu et al. 2024; Zhang et al. 2024; Yu et al.
2025). Despite the wide applicability of LLMs, they of-
ten struggle with knowledge-intensive queries because their
knowledge can be incomplete or outdated, which leads to
factual inaccuracies or hallucinations (Zhang et al. 2023;
Sahoo et al. 2024; Ji et al. 2023). To address these chal-
lenges, Retrieval-Augmented Generation (RAG) enhances
model performance by retrieving relevant external knowl-
edge during inference. This approach enables the model to
access up-to-date information and fill in knowledge gaps.
RAG is meant to help models access up-to-date informa-
tion, but the high cost and delay of online retrieval have
caused early research (Chen et al. 2025; Jin et al. 2025; Song
et al. 2025) to focus mainly on using locally stored knowl-
edge. While these local sources are efficient, they are often
outdated or incomplete, which affects model performance in
real-world situations. More recent research (Li et al. 2025b;
Zheng et al. 2025; Wei et al. 2025) has started exploring
*These authors contributed equally.
User QuestionKL
LLMsThinkSearchAnswerSearch ResultsAdvanced SearchField RestrictionField LimitationReference LLM 
WebFilterfuzzy matchesField limitation
Cross Validation
Noisy file
Thinking: I want to know California’s minimum wage. I’ll search directly for the answer.Query: California minimum wageResult: Finds a 2021 article stating the wage is $14/hour, which is outdated.
Question: I want to know California’s minimum wage
Thinking: I want to know California’s minimum wage. To avoid outdated info, I’ll add a date filter for the past year.Query: California minimum wage after: 2024-07-01Result: Retrieves an official government page from 2025 stating the wage is $16/hour.
Noisy file
Target File
Noisy File
Others
Search AgentQuery GenerationNoisy File
Target FileWeb Environment
Figure 1: Comparison of WebFilter with Existing Methods:
Existing methods (Zheng et al. 2025; Song et al. 2025) of-
ten yield unreliable results in misinformation-rich web envi-
ronments. WebFilter enhances accuracy by using advanced
search operators to filter noise and retrieve target files.
RAG in web-based environments, showing the benefits of
using web search during model training. However, unlike lo-
cal retrieval, which relies on trusted, static data, web-based
retrieval comes with its own challenges: 1) Web environ-
ment pervasive misinformation : the open web is saturated
with misinformation, low-quality content noise (Yang et al.
2025b,c). This significantly increases the difficulty of iden-
tifying credible sources and introduces risks of model hallu-
cination or factual inconsistency during answer generation.
2)Web tools underutilization : local tools and web-based
search engines differ significantly in utilization. While web
tools provide advanced search operators that help avoid out-
dated information and enable retrieval from trusted sources,
such capabilities are unavailable in offline settings. As a re-
sult, locally trained models struggle to learn and use ad-
vanced tools, limiting their ability to filter noise and focus
on reliable sources.
To address these challenges, we present WebFilter, a
framework that improves answer accuracy by filtering noise
using advanced search operators. Given the imperfections
of search engines and their sensitivity to query quality (see
Fig. 1), our framework further enhances retrieval accuracy
by formulating more effective queries and applying ad-
vanced operators. These operators, such as source selection
and time filtering, enable precise retrieval by filtering outarXiv:2508.07956v1  [cs.IR]  11 Aug 2025

noisy sources and enhancing credibility. However, guiding
models to correctly use these operators is challenging. Our
experiments show that, even with instructions, models rarely
proactively use advanced operators. This is because, while
existing models (Li et al. 2025b; Zheng et al. 2025; Song
et al. 2025) have achieved state-of-the-art results using Re-
inforcement Learning (RL) (Kaelbling, Littman, and Moore
1996), they focus primarily on outcomes, rather than guid-
ing behavior in web-based environments. As a result, they
often rely on unreliable shortcuts. Thus, methods that effec-
tively guide models to utilize advanced operators for noise
filtering are crucial.
Therefore, to systematically integrate web search into
model reasoning, we formulate retrieval as a Markov De-
cision Process (MDP) and guide the model to operate as an
information retrieval agent capable of using advanced oper-
ators. To more effectively guide model behavior, we intro-
duce an Information-Filtering Reward strategy, which com-
bines two complementary rewards driven by both behav-
ioral and outcome considerations. Specifically: (1) Source-
restricting Reward (SR) encourages the model to proac-
tively use advanced search operators (e.g., domain filters,
date ranges), shaping query formulation strategies. In the
early stages of training, SR promotes exploration even when
the model’s performance is suboptimal. As training pro-
gresses, SR gradually shifts focus towards accuracy, refin-
ing the model’s query formulation to prioritize precision.
This transition helps balance exploration and exploitation,
ensuring effective learning and improved retrieval perfor-
mance. (2) Retrieval-precision Reward (RR) reinforces
outcomes by having a more capable, large-scale LLM eval-
uate the quality of retrieved content and provide feedback,
enabling the model to refine its queries and improve source
selection based on retrieval results. Through the combina-
tion of structured modeling, instruction design, and reward
learning, WebFilter overcomes pervasive web misinforma-
tion and fully leverages advanced web search tools for pre-
cise and reliable retrieval. In summary, our contributions are
as follows:
• We propose WebFilter, a novel RAG framework explic-
itly designed for real-world web environments. It formu-
lates retrieval as an MDP and trains LLMs as informa-
tion retrieval agents, enabling effective mitigation of per-
vasive misinformation and better utilization of advanced
web search tools.
• We introduce an information-filtering reward strategy
that guides precise, source-restricted retrieval and en-
ables robust misinformation filtering, addressing both
pervasive web noise and tool underutilization.
• Experiments show that WebFilter achieves state-of-the-
art QA performance, with advanced search operator us-
age rising from 10% to 75%, and significant gains across
in-domain and out-of-domain benchmarks.
Related Work
Agentic Retrieval Augmented Generation
Recent work has explored agentic RAG (Chen et al. 2025;
Jin et al. 2025; Song et al. 2025) to integrate retrieval into thereasoning process of LLMs. For example, methods such as
ReSearch (Chen et al. 2025), Search-R1 (Jin et al. 2025), and
R1-Searcher (Song et al. 2025) train LLMs to autonomously
generate search queries while reasoning with a local search
engine. However, LLMs trained in such local settings of-
ten struggle to generalize to real-world web environments
(Zheng et al. 2025). To overcome this, methods such as
WebRL (Qi et al. 2024), WebThinker (Li et al. 2025b),
R1-Searcher (Song et al. 2025), DeepResearcher (Zheng
et al. 2025), WebAgent-R1 (Wei et al. 2025) leverage online
search engines for training. Yet, compared to local settings,
online environments pose greater challenges, including high
API costs, network latency, and the abundance of false or
redundant information, all of which hinder efficient train-
ing and retrieval (Zheng et al. 2025). However, due to their
reliance on local web environments and the lack of source-
specified retrieval data, existing reward schemes fall short in
tackling real-world web challenges such as pervasive misin-
formation and poor use of advanced search operators. To ad-
dress this issue, we formulate retrieval as a Markov Decision
Process, guided by explicit instruction on tool usage and
an Information-Filtering Reward strategy, jointly enabling
more structured, source-restricted querying and robust in-
formation filtering.
Reinforcement Learning for LLMs
Reinforcement learning (RL) has become increasingly
prominent in training LLMs, supporting applications that
span from preference alignment (Ouyang et al. 2022; Casper
et al. 2023; Kaufmann et al. 2023) to complex tasks (Hao
et al. 2023; Pang et al. 2024; Tang et al. 2025; Xie et al.
2025). A growing area of interest is the application of RL
to tool-integrated tasks (Li, Zou, and Liu 2025), which in-
volve multi-step interactions and dynamic tool states. The
high interactivity with the environment makes them a natural
fit for RL. Existing research has explored RL-trained LLM
agents for tool-integrated reasoning. For example, ToRL (Li,
Zou, and Liu 2025) and Tool-N1 (Zhang et al. 2025) em-
ploy rule-based outcome rewards that account for both ac-
curacy and format to guide RL, while other methods (Wang
et al. 2025; Sha, Cui, and Wang 2025; Singh et al. 2025)
extend this by incorporating tool usage-based reward. How-
ever, most RL methods are built on local corpora without
source-restricted supervision, limiting their generalization
to real-world web environments with noisy information and
underused search tools. We address this by formulating re-
trieval as an MDP and combining tool-use instruction with
an Information-Filtering Reward strategy.
Methodology
In this section, we introduce the WebFilter training frame-
work, designed to enhance Retrieval-Augmented Generation
(RAG) by improving query formulation and filtering unreli-
able web content. As shown in Fig. 2, we model the retrieval
process as a Markov Decision Process, enabling the model
to decide when and how to issue search queries and integrate
the retrieved information. To guide this process, we imple-
ment an Information-Filtering reward strategy, which eval-
uates retrieval outcomes and refines query strategies based

on feedback from a stronger LLM. The following sections
detail the framework and problem formulation.
Problem Formulation
We model the task completion as a Markov Decision Process
(MDP), denoted by (S, A, R, T), where the state st∈S
represents the history of previous actions at time step t. At
each tstep, the agent selects an action at∈Abased on
the current state st, following the policy πθ. When the agent
selects the ”search” action ( at=search), it updates the state
by incorporating the retrieved results. Specifically, dtrefers
to the content retrieved based on the search query at time
stept. The state transition is defined as:
st+1=T(st, at) =[st;at, dt]ifat=search ,
[st;at] otherwise .(1)
where Trepresents the deterministic state transition func-
tion, and the agent receives a reward rt=R(st, at), as de-
termined by the environment. The process terminates when
the task is completed or the maximum allowed interactions
are reached.
Learning to Use Advanced Search Tools
WebFilter is implemented as a prompt-based Information
Retrieval Agent that proactively conducts web searches and
reasons over retrieved results before issuing a final answer.
It operates under a cost-sensitive policy, minimizing exces-
sive queries and avoiding uncertain responses when evi-
dence is lacking. The agent is explicitly instructed to in-
tegrate advanced search operators such as OR,AND,NOT,
quotation marks for exact phrases, domain restrictions via
site: , and date filters like after: , enabling precise,
source-restricted retrieval. It also prioritizes trusted domains
(e.g., wikipedia.org ,gov.cn ), guiding the model to
generate focused, reliable, and efficient queries. For imple-
mentation details, please refer to our GitHub repository.
Information-Filtering Reward Strategy
Although the MDP formulation structures retrieval inter-
actions, it alone cannot ensure precise query formulation
or reliable filtering of web content. To address this, we
propose an Information-Filtering Reward strategy that inte-
grates both behavior-based and outcome-based incentives.
The Source-restricting Reward (SR) acts as a behavior-
based restriction, promoting the use of advanced search op-
erators for precise, source-restricted queries. In contrast,
the Retrieval-precision Reward (RR) serves as an outcome-
based signal, leveraging external critique to assess and refine
retrieval quality. Together, these rewards guide the model to-
ward more effective and trustworthy web search. Next, we
will describe its design in detail.
Source-restricting Reward (SR) To encourage precise
and source-restricted queries, we design a rule-based
Source-restricting Reward that promotes the use of advanced
search operators. Specifically, the Source-restricting Reward
is defined by the following formula:
K={k1, k2, ..., k m} ⊆Σ∗, (2)Q={q1, q2, ..., q n} ⊆y, (3)
Rsrc=I[∃q∈ Q,∃k∈ K, ReMatch (k, q) = 1] ,(4)
where Σ∗denotes the set of all sequences over the vocab-
ulary of the policy model; K ⊆ Σ∗is a predefined set
of advanced search keyword patterns, such as “site:”, “-”,
or “AND”; y⊆Σ∗is a response generated by the policy
model; and Q ⊆ yis the set of search queries extracted
from y. The binary function ReMatch (k, q)returns 1 if
query qmatches the regular expression associated with pat-
ternk, and 0 otherwise. The indicator function I[·]equals 1
if the predicate holds and 0 otherwise. We define the Source-
restricting Reward Rsrc∈ {0,1}to be 1 if any query in Q
contains an advanced search pattern from K, and 0 other-
wise. This design explicitly promotes the use of advanced
search operators in the reinforcement learning process.
Retrieval-precision Reward (RR) Beyond encouraging
the use of advanced search operators, it is equally impor-
tant to ensure they are applied correctly and effectively. We
thus propose an LLM-based, outcome-oriented Retrieval-
precision Reward (RR) that evaluates responses and pro-
vides feedback on operator correctness and quality. Specifi-
cally, the Retrieval-precision Reward is defined as:
z, c=LLM Judge (g, y, I Judge ), (5)
where LLM Judge denotes a powerful LLM used to evaluate
the model’s predictions, gis the ground-truth answer, yis the
response generated by the policy model, and IJudge denotes
the prompt template provided to LLM Judge , as shown in .
The scalar z∈ {0,1}measures the correctness of the pre-
dicted answer, while c∈ {0,1}evaluates whether the use of
advanced search syntax contributed to the retrieval quality.
Reward Aggregation The final reward Ris computed as:
R=

−1 if¬Cformat,
f(z, c, R f1)ifCformat∧Cagg,
0.1 ifCformat∧ ¬Cagg∧Csrc,
0 otherwise .(6)
Here, Cformat denotes correct output format; Caggindicates
f(Rllm, Rf1)̸= 0;Csrcmeans Rsrc= 1. The F1 Reward
Rf1is computed as:
Rf1=2×IN
PN+RN, (7)
where PN is the word count of the predicted answer, RN is
the word count of the reference answer, and INis the word
count of overlapping words between them. The aggregation
function f(·)is defined as:
f(u, v, w ) =αu+βv+ (1−α−β)w, (8)
where αandβare hyperparameters used to balance the in-
fluence of each reward on the policy optimization process.

To find the first king of the longest Chinese dynasty, I need to identify the longest dynastyand then find who the first king of that dynasty was. The search results confirm that the Zhou dynasty was the longest dynasty in China. Now I need to find outwho the first king of the Zhou dynasty was.
Output 1Output 2Output nReference LLM Reward ModuleGroupComputationSearch EngineRollout ModulePolicy LLMQueryReward 1Reward 2Reward nAdvantage 1Advantage 2Advantage nKLTo do that, Iwill first search for the longest Chinesedynasty.King Wu of ZhouSearch Query: The first king of the Zhou dynasty    site: wikipedia.orgSearch Results: WuSearch Query:The longest Chinesedynasty    site: wikipedia.org Search Result: ZhouTo do that, Iwill first search for the longest Chinesedynasty.XTXTXSXTXSXTXATThinking StepSSearch CallingAFinal Answer 
Who was the first emperor of the longest Chinese dynasty?
………Source-restricting RewardThe search results consistently identify king Wu of Zhou as the founding king of the Zhou dynasty. Therefore, I can conclude that …
Retrieval-precision RewardQuestion from User Reinforcement Learning𝐶𝑎𝑔𝑔𝐶𝑠𝑟𝑐𝐶𝑎𝑔𝑔𝐶𝑠𝑟𝑐“Eiffel Tower location”     site: gov.cnSWhat is the name of the city where the Eiffel Tower is located?“Eiffel Tower location”    site: wikipedia.orgSInfo: The government website does not include a description of the Eiffel Tower.Unable to find accurate informationTInfo: The Eiffel Tower is located on the Champ de Mars in Paris, France.The results indicate answer is ParisT0.11.0RomaAParisARewardReward CalculationThink Inference𝐶𝑓𝑜𝑟𝑚𝑎𝑡𝐶𝑓𝑜𝑟𝑚𝑎𝑡Figure 2: Overview of the WebFilter training framework. Upper: We formulate retrieval as a Markov Decision Process, where
the model interacts with web search tools through step-by-step actions, including query generation and evidence selection.
Middle: To improve tool usage, we provide explicit instructions and demonstrations on how to issue effective, source-aware
queries. Lower: The policy is optimized using a behavior- and outcome-driven Information-Filtering Reward strategy, which
encourages both proper tool invocation and high-quality information retrieval.
RL Training Framework
Policy Optimization In this work, we adopt the Group
Relative Policy Optimization (GRPO) (Shao et al. 2024),
which improves the current policy πθby leveraging a ref-
erence policy πθrefand a set of rollouts generated by a old
policy πθold. To support search engine calls, the GRPO ob-
jective is extended as follows:
J(θ) =Ex∼D,{yi}G
i=1∼πθold(·|x)"
1
GGX
i=1min 
πθ(yi|x)
πθold(yi|x)Ai,
clipπθ(yi|x)
πθold(yi|x),1−ϵ,1 +ϵ
Ai!
−βDKL(πθ∥πθref)#
.
(9)
Here, xdenotes an input sampled from the data distribu-
tionD, andyirepresents a trajectory generated by πθold.DKL
is the estimated KL divergence (Shao et al. 2024), and ϵ,β
are hyperparameters that control the trust region and regu-
larization strength, respectively. The reward rifor each yiis
computed jointly across trajectories:
r1, r2, . . . , r G=R(y1, y2, . . . , y G), (10)
and the advantage Aiis normalized within the batch as:
Ai=ri−mean(r1, r2, . . . , r G)
std(r1, r2, . . . , r G). (11)This objective encourages stable policy improvement, en-
abling effective integration of retrieval-based reasoning into
the learning process.
Experiments
Benchmarks
Our experimental setting is built on question answering
datasets that assess reasoning and retrieval capabilities in di-
verse scenarios. For in-domain evaluation, we use the devel-
opment sets of Natural Questions (NQ) (Kwiatkowski et al.
2019), TriviaQA (TQ) (Joshi et al. 2017), HotpotQA (Yang
et al. 2018), and 2Wiki (Ho et al. 2020). For out-of-domain
evaluation, we include the complex open-domain reasoning
dataset MuSiQue (Trivedi et al. 2022) and the web-search-
focused benchmark Bamboogle (Press et al. 2022), which
differ in question style and information distribution. To en-
sure a balanced and consistent evaluation across datasets, we
select a fixed number of examples from each. Specifically,
512 examples are chosen from the development sets of NQ,
TQ, HotpotQA, 2Wiki, and MuSiQue, and all 125 examples
are selected from the development set of Bamboogle.
Baselines
To evaluate WebFilter’s effectiveness, we compare it against
several baselines representing different methodologies:
Direct Reasoning : Models relying solely on internal knowl-
edge, such as Qwen3-32B (Yang et al. 2025a), Gemini-2.0-
Flash (Team et al. 2023), and GPT-4o (Hurst et al. 2024).

Table 1: Performance of different methods on in-domain datasets, evaluated with rule-based ( ACC R) and LLM-based ( ACC L)
metrics. Best results are highlighted in bold.
Environment Method NQ TQ HotpotQA 2Wiki
ACC RACC LACC RACC LACC RACC LACC RACC L
Direct ReasoningQwen3-32B 16.5 36 .3 11 .6 52 .9 29 .9 19 .1 26 .5 19 .9
Gemini-2.0-Flash 20.7 47 .1 16 .2 68 .9 42 .4 29 .9 32 .5 23 .8
GPT-4o 23.1 53 .9 17 .6 73 .2 50 .2 37 .1 38 .6 30 .5
Local RAGSearch-o1 34.5 57 .4 52 .6 61 .1 31 .6 40 .8 28 .6 32 .8
Search-r1-base 45.4 60 .0 71 .9 76 .2 55 .9 63 .0 44 .6 47 .9
Search-r1-instruct 33.1 49 .6 44 .7 49 .2 45 .7 52 .5 43 .4 48 .8
Web SearchR1-Searcher 35.4 52 .3 73 .1 79 .1 44 .8 53 .1 59 .4 65 .8
DeepResearcher 39.6 61 .9 78.4 85.0 52 .8 64 .3 59 .7 66 .6
WebFilter (Ours) 40.1 63.1 77.5 85.4 55.1 65.2 60.1 67.5
Local RAG : Methods retrieving knowledge from offline
documents. For example, Search-o1 (Li et al. 2025a) per-
forms multi-step reasoning by generating search queries and
using the retrieved snippets as context, while Search-r1-
base (Jin et al. 2025) retrieves evidence from Wikipedia dur-
ing both training and inference. Search-r1-instruct (Jin et al.
2025) differs by initializing the actor with an instruct-tuned
language model to guide the retrieval process.
Web Search : Methods utilizing online tools. Both our ap-
proach and R1-Searcher (Song et al. 2025) rely on the
Google API for web search. In addition to Google search,
DeepResearcher (Zheng et al. 2025) integrates a Web-
Browser tool for web navigation, which leads to increased
time spent browsing and accessing websites, thereby slow-
ing down the overall training speed. All methods, including
ours, employ Qwen-2.5-7B-instruct (Yang et al. 2024) for
model inference.
Metrics
We evaluate model performance using both rule-based
(ACC R) and LLM-based ( ACC L) metrics. The rule-based
metric uses an F1 score to measure overlap between pre-
dictions and reference answers, reflecting factual precision.
ForACC L, we adopt the LLM-as-Judge framework (Zheng
et al. 2023), where GPT-4o-mini (Hurst et al. 2024) assesses
whether model answers align semantically with the refer-
ences, thus capturing nuances beyond exact matching.
Implementation Details
We implement our model using the VeRL framework (Sheng
et al. 2024) and adopt Qwen2.5-7B-Instruct (Yang et al.
2024) as the backbone. The hyperparameters for the ag-
gregation function are set as α= 0.4andβ= 0.2. The
learning rate is set to 1e-5, and training proceeds with a
mini-batch size of 4,096. Each iteration processes 256 sam-
ples, generating 16 rollouts per sample. Additionally, we
apply a sampling temperature of 1.0 and limit the maxi-
mum retrieval count to 10. We apply loss masking to up-
date only model-generated tokens. Our Retrieval-precision
Reward uses Qwen3-30B-A3B (Yang et al. 2025a) as the
judge model, which is free and can be deployed locally.Table 2: Performance of methods on out-of-domain datasets.
Method Musique Bamboogle
ACC RACC LACC RACC L
Qwen3-32B 10.7 4 .9 24 .7 18 .4
Gemini-2.0-Flash 11.4 6 .1 36 .5 28 .0
GPT-4o 22.5 15 .0 52 .6 43 .2
Search-o1 16.8 21 .3 46 .6 53 .6
Search-r1-base 26.7 27 .5 56 .6 57 .6
Search-r1-instruct 26.5 28 .3 45 .0 47 .2
R1-Searcher 22.8 25 .6 64 .8 65 .6
DeepResearcher 27.1 29.3 71 .0 72 .8
WebFilter (Ours) 24.5 30.0 73.1 74.3
Results on In-Domain Settings
WebFilter achieves state-of-the-art performance across all
four in-domain datasets, demonstrating notable strengths in
multi-hop reasoning tasks, as shown in Tab. 1. On Hot-
potQA, it outperforms DeepResearcher by 2.3% in ACC R,
despite DeepResearcher relying on a browser tool with
higher latency for broader web exploration. This advantage
arises from WebFilter’s ability to formulate precise, source-
restricted queries using advanced search operators, effec-
tively reducing noise in retrieved documents. Compared to
local RAGs, the performance gap on 2Wiki becomes more
pronounced, with WebFilter achieving around a 17% higher
ACC Rthrough selective access to trusted external sources.
These improvements reflect deliberate design choices. Un-
like methods limited to fixed domains, such as R1-Searcher,
which restricts access to Wikipedia, or those reliant on
extensive web browsing, which introduces higher latency
(e.g., DeepResearcher), WebFilter focuses on generating
precise, source-restricted queries for unrestricted Google
API searches. This strategy strikes a balance between re-
trieval flexibility and high precision, minimizing irrelevant
content and enhancing the quality of retrieved contexts. As
indicated by gains in ACC L, WebFilter retrieves evidence
that aligns more closely with ground-truth answers, thereby
supporting stronger reasoning.

Table 3: Performance of different WebFilter variants across in-domain datasets (NQ, TQ, HotpotQA, 2Wiki) and out-of-domain
datasets (Musique, Bamboogle). “SR” denotes the Source-restricting Reward, and “RR” denotes the Retrieval-precision Re-
ward.
Methods NQ TQ HotpotQA 2Wiki Musique Bamboogle
ACC RACC LACC RACC LACC RACC LACC RACC LACC RACC LACC RACC L
Base 41.2 63.6 78.5 82.6 49.4 59.9 55.2 58.2 22.7 26.2 64.9 65.8
Base+SR 41.4 64.3 79.0 86.0 50.6 60.1 59.7 65.3 24.5 27.6 64.6 65.2
Base+SR+RR(Ours) 40.1 63.1 77.5 85.4 55.1 65.2 60.1 67.5 24.5 29.0 73.1 74.3
020406080100MusiqueBambooglePopQANQTQ2Wiki
Frequency (%)WebFilter w/o SRWebFilter
020406080100BamboogleMusique2WikiHotpotQATQNQ
Frequency (%)WebFilter w/o SRWebFilter
Figure 3: Frequency of advanced operators across variants.
Results on Out-of-Domain Settings
WebFilter consistently shows strong generalization on
out-of-domain datasets. As shown in Tab. 2, WebFilter
achieves the highest ACC Lon the challenging open-domain
Musique dataset (30.0%) and the web-search-heavy Bam-
boogle dataset (74.3%). These results suggest that Web-
Filter retrieves evidence more semantically aligned with
ground-truth answers, which is critical for reasoning be-
yond exact matching. WebFilter also maintains competitive
ACC Rscores, particularly on Bamboogle (73.1%), indicat-
ing its ability to preserve factual consistency in new do-
mains. Moreover, it outperforms DeepResearcher on ACC L
for Musique (30.0% vs. 29.3%), demonstrating its strength
in handling difficult open-domain questions. WebFilter’s
ability to retrieve semantically relevant and factually consis-
tent information across diverse topics underscores its practi-
cal value for real-world applications involving domain shifts
and open-ended queries.
Ablation Study
Tab. 3 shows the performance of WebFilter variants on both
in-domain datasets (NQ, TQ, HotpotQA, 2Wiki) and out-of-
domain datasets (Musique, Bamboogle). From the results,
we observe that incorporating the Source-restricting Reward
(SR) leads to noticeable gains on in-domain datasets, im-
proving the model’s ability to retrieve information from re-
liable sources. SR also encourages the use of advanced op-
erators. As shown in Fig. 3, SR-guided queries include ad-
vanced operators more than 75% of the time, compared to
less than 10% without SR. Adding the Retrieval-precision
Reward (RR) further boosts performance, especially on out-
of-domain datasets. RR refines the retrieval process, align-
ing the generated evidence more closely with ground-truth
0 5 10 15 20 25 30
Training Steps0.60.7ACCR(a)ACC Rover training steps
0 5 10 15 20 25 30
Training Steps1.41.6T ool Call Counts
(b) Tool call counts over training steps
0 5 10 15 20 25 30
Training Steps100015002000Response Length
(c) Response length over training steps
Figure 4: Training dynamics showing (a) QA accuracy
(ACC R), (b) tool call behavior, and (c) response length evo-
lution across training steps.
answers. For instance, on Bamboogle, ACC Rimproves by
8.5%, from 64.6% to 73.1%, and ACC Lrises from 65.2% to
74.3%. This improvement is mainly due to Bamboogle’s fo-
cus on web-search-based reasoning, where WebFilter’s abil-
ity to leverage advanced search operators and filter irrelevant
content enhances retrieval of high-quality, relevant evidence.
The combination of SR and RR yields the best results, with
SR boosting retrieval precision and RR enhancing general-
ization. These findings demonstrate the effectiveness of our
reward framework in improving both retrieval quality and
reasoning performance.
Training Dynamics Analysis
Fig. 4 illustrates the training dynamics of our model, focus-
ing on both QA performance and behavioral changes. As
shown in Fig. 4a, ACC Rsteadily increases throughout train-
ing, rising from approximately 0.52 to around 0.77 on the
TQ dataset. This indicates continuous improvements in re-
trieval accuracy as the model progresses.

Query Source RestrictionQuestion: Which film was released more recently, After The Apocalypse or Idol On Parade?Ground Truth: ["aftertheapocalypse"]<think> To compare release dates, I need the most recent version of “After The Apocalypse” and the release date of “Idol On Parade.” I will search IMDb for both.</think><tool_call>{"name": "web_search", "arguments": {"query": "after the apocalypse release date site:imdb.com"}}</tool_call><think> The search result gives me two “After The Apocalypse” films, one in 2004 and the other in 2010, and I need to further distinguish which one I am referring to with “tv anime adaptation of Aakae Aaito's post-apocalyptic” in 2025. I will search again for “Idol On Parade”.</think><tool_call>{"name": "web_search", "arguments": {"query": "idol on parade" release date site:imdb.com"}}</tool_call><think> The search for “Idol On Parade” confirms that it was released in 1959. The “After The Apocalypse” tv anime adaptation is from 2025, while the 2004 “After The Apocalypse” was a film, and the 2010 “After The Apocalypse” was either another film. Since the adaptation was from 2025, it is the more recent release compared to the 1959 “Idol On Parade”. Therefore, “After The Apocalypse” is the film released more recently.</think><answer>after the apocalypse</answer>Question: What according to the commercial cleaned a Misleading Information FilteringMulti-source cross validationQuestion: Which film was released more recently, After 
Misleading Information FilteringQuestion: What is Robert Jonckheere's occupation?Ground Truth: ["astronomer"]<think>To determine Robert Jonckheere's occupation, we need to search the web for information about him. since I don't have any prior knowledge about Robert Jonckheere, I will perform a general web search to find his occupation.</think><tool_call>{"name": "web_search", "arguments": {"query": "robert jonckheere occupation"}}</tool_call><think>From the search results, it seems Robert Jonckheere was known as an astronomer. But a profilehas shown that Aimable Robert Jonckheere was a psychologist and statistician. To clarify, Ineed to revisit the most likely relevant page that indicates he was an astronomer.I will check a trusted source like wikipedia to confirm the information.</think><tool_call>{"name": "web_search", "arguments": {"query": "robert jonckheere astronomer site:wikipedia.org"}}</tool_call><think> After reviewing the results, it is clear that Robert Jonckheere was an astronomer. The wikipedia pages and other sources consistently mention his work as an astronomer and his contributions to astronomy, including the discovery of more than 3350 double stars.</think><answer>astronomer</answer>
Search Strategy OptimizationQuestion: What according to the commercial cleaned a big, big carpet for less than half a crownGround Truth: ["onethousandandone","1001"]<think>Itseemstobeaskingaboutacommercialadvertisementormethodforcleaningalargecarpetforaverylowcost.Ineedtofindanoldadvertisementthatmentionsthis.</think><tool_call>{"name":"web_search","arguments":{"query":"commercialbigcarpetcleanedlessthanhalfacrownsite:wikipedia.org"}}</tool_call><think>The search results do not contain any relevant information. Let me refine the search by excluding results from pages that primarily deal with the history and types of carpets, and instead look for old advertisements or passages related to cleaning carpets.I will conductawebsearchwithoutsiterestrictionstofindinformationaboutacommercialforcleaningabigcarpetforlessthanhalfacrown. </think><tool_call>{\"name\":\"web_search\",\"arguments\":{"query":"commercialbigcarpetcleanedlessthanhalfacrown"}}</tool_call><think> The search results point to various mentions of a commercial product “1001”…One of the specific posts mentions that “1001 cleans a big, big carpet for less than half a crown”, which aligns with the question asked.</think><answer>1001</answer>Figure 5: Case studies showing how WebFilter improves QA by (a) narrowing searches to authoritative sources for precise
results, (b) verifying ambiguous or conflicting information via trusted sites, and (c) adaptively refining search queries when
initial attempts are insufficient.
To capture broader trends, Fig. 4b and Fig. 4c show the
average tool call counts and response lengths across all eval-
uated datasets. In Fig. 4b, tool call counts increase during
early training, plateau around step 20, and then rise again.
This pattern suggests that the model gradually incorporates
more frequent retrieval as training progresses. Meanwhile,
Fig. 4c reveals that the average response length grows from
about 1,000 tokens to nearly 2,000 tokens, indicating that
the model generates more detailed and comprehensive re-
sponses over time. Overall, these results demonstrate that
our model not only improves retrieval accuracy but also
adapts its tool usage and response strategies, leading to more
effective and informative outputs.
Case Study
We present three representative cases in Fig. 5 to illustrate
WebFilter’s intelligent retrieval behaviors. Through system-
atic analysis, we identify three key behavioral patterns that
define the model’s advanced search capabilities:
Query Source Restriction. For domain-specific queries,
WebFilter proactively limits search scopes to authoritative
sources. In Case 1, when asked about film release dates, it
automatically appends “site:imdb.com” to queries. This tar-
geted approach ensures precise, trustworthy results while re-
ducing latency by filtering out irrelevant information.
Misleading Information Filtering. WebFilter demonstrates
strong disambiguation skills for resolving conflicting infor-
mation. In Case 2, when the initial search reveals both an as-tronomer and a similarly named psychologist, the model de-
tects the ambiguity and performs a refined follow-up search
limited to Wikipedia, successfully verifying the correct oc-
cupation.
Search Strategy Optimization. WebFilter dynamically ad-
justs its retrieval strategy when initial searches are insuf-
ficient. In Case 3, failing to find relevant information via
a Wikipedia-restricted query, the model expands its search
without site constraints, ultimately locating references to the
product “1001,” which matches the question context.
These cases highlight WebFilter’s ability to reason about
search scope, verify information across sources, and adapt
strategies to improve retrieval effectiveness.
Limitations
While our approach shows progress, it has some limitations.
To enhance RAG capabilities, improving search quality is
crucial, but not sufficient on its own. In many cases, errors
occur not because the model fails to retrieve relevant infor-
mation, but because it struggles to correctly interpret and
reason with the retrieved data. Our model, when combined
with improvements in reasoning abilities, can deliver even
greater value in future work.
Conclusion
We present WebFilter, a framework that improves Retrieval-
Augmented Generation (RAG) by leveraging advanced

search operators for precise, source-aware queries and mis-
information filtering. By modeling retrieval as a Markov
Decision Process, WebFilter learns to effectively use web
search tools. Experiments demonstrate strong gains on both
in-domain and out-of-domain QA. Future work will explore
broader web interactions to further enhance real-world RAG
performance.
Acknowledgements
This work was supported by the Ant Group Research Intern
Program.
References
Casper, S.; Davies, X.; Shi, C.; Gilbert, T. K.; Scheurer, J.;
Rando, J.; Freedman, R.; Korbak, T.; Lindner, D.; Freire,
P.; et al. 2023. Open problems and fundamental limita-
tions of reinforcement learning from human feedback. arXiv
preprint arXiv:2307.15217 .
Chen, M.; Li, T.; Sun, H.; Zhou, Y .; Zhu, C.; Wang, H.; Pan,
J. Z.; Zhang, W.; Chen, H.; Yang, F.; et al. 2025. Research:
Learning to reason with search for llms via reinforcement
learning. arXiv preprint arXiv:2503.19470 .
Hao, S.; Gu, Y .; Ma, H.; Hong, J. J.; Wang, Z.; Wang, D. Z.;
and Hu, Z. 2023. Reasoning with language model is plan-
ning with world model. arXiv preprint arXiv:2305.14992 .
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa, A.
2020. Constructing A Multi-hop QA Dataset for Compre-
hensive Evaluation of Reasoning Steps. In Proceedings of
the 28th International Conference on Computational Lin-
guistics , 6609–6625.
Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh,
A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Rad-
ford, A.; et al. 2024. Gpt-4o system card. arXiv preprint
arXiv:2410.21276 .
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y .; Ishii, E.;
Bang, Y . J.; Madotto, A.; and Fung, P. 2023. Survey of hal-
lucination in natural language generation. ACM computing
surveys , 55(12): 1–38.
Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.; Za-
mani, H.; and Han, J. 2025. Search-r1: Training llms to rea-
son and leverage search engines with reinforcement learn-
ing. arXiv preprint arXiv:2503.09516 .
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L.
2017. Triviaqa: A large scale distantly supervised chal-
lenge dataset for reading comprehension. arXiv preprint
arXiv:1705.03551 .
Kaelbling, L. P.; Littman, M. L.; and Moore, A. W. 1996.
Reinforcement learning: A survey. Journal of artificial in-
telligence research , 4: 237–285.
Kaufmann, T.; Weng, P.; Bengs, V .; and H ¨ullermeier, E.
2023. A survey of reinforcement learning from human feed-
back. arXiv preprint arXiv:2312.14925 , 10.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; et al. 2019. Natural questions: a benchmark for
question answering research. Transactions of the Associa-
tion for Computational Linguistics , 7: 453–466.Li, X.; Dong, G.; Jin, J.; Zhang, Y .; Zhou, Y .; Zhu, Y .; Zhang,
P.; and Dou, Z. 2025a. Search-o1: Agentic search-enhanced
large reasoning models. arXiv preprint arXiv:2501.05366 .
Li, X.; Jin, J.; Dong, G.; Qian, H.; Zhu, Y .; Wu, Y .; Wen,
J.-R.; and Dou, Z. 2025b. WebThinker: Empowering Large
Reasoning Models with Deep Research Capability. arXiv
preprint arXiv:2504.21776 .
Li, X.; Zou, H.; and Liu, P. 2025. Torl: Scaling tool-
integrated rl. arXiv preprint arXiv:2503.23383 .
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.;
Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.;
et al. 2022. Training language models to follow instructions
with human feedback. Advances in neural information pro-
cessing systems , 35: 27730–27744.
Pang, R. Y .; Yuan, W.; He, H.; Cho, K.; Sukhbaatar, S.; and
Weston, J. 2024. Iterative reasoning preference optimiza-
tion. Advances in Neural Information Processing Systems ,
37: 116617–116637.
Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N. A.;
and Lewis, M. 2022. Measuring and narrowing the com-
positionality gap in language models. arXiv preprint
arXiv:2210.03350 .
Qi, Z.; Liu, X.; Iong, I. L.; Lai, H.; Sun, X.; Zhao, W.; Yang,
Y .; Yang, X.; Sun, J.; Yao, S.; et al. 2024. WebRL: Train-
ing LLM Web Agents via Self-Evolving Online Curriculum
Reinforcement Learning. arXiv preprint arXiv:2411.02337 .
Qiu, J.; Lam, K.; Li, G.; Acharya, A.; Wong, T. Y .; Darzi,
A.; Yuan, W.; and Topol, E. J. 2024. LLM-based agentic
systems in medicine and healthcare. Nature Machine Intel-
ligence , 6(12): 1418–1420.
Sahoo, S. S.; Plasek, J. M.; Xu, H.; Uzuner, ¨O.; Cohen,
T.; Yetisgen, M.; Liu, H.; Meystre, S.; and Wang, Y . 2024.
Large language models for biomedicine: foundations, op-
portunities, challenges, and best practices. Journal of the
American Medical Informatics Association , 31(9): 2114–
2124.
Sha, Z.; Cui, S.; and Wang, W. 2025. SEM: Reinforce-
ment Learning for Search-Efficient Large Language Models.
arXiv preprint arXiv:2505.07903 .
Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang,
H.; Zhang, M.; Li, Y .; Wu, Y .; et al. 2024. Deepseekmath:
Pushing the limits of mathematical reasoning in open lan-
guage models. arXiv preprint arXiv:2402.03300 .
Sheng, G.; Zhang, C.; Ye, Z.; and ... 2024. HybridFlow:
A Flexible and Efficient RLHF Framework. arXiv preprint
arXiv:2409.19256 .
Singh, J.; Magazine, R.; Pandya, Y .; and Nambi, A. 2025.
Agentic reasoning and tool integration for llms via rein-
forcement learning. arXiv preprint arXiv:2505.01441 .
Song, H.; Jiang, J.; Min, Y .; Chen, J.; Chen, Z.; Zhao, W. X.;
Fang, L.; and Wen, J.-R. 2025. R1-Searcher: Incentivizing
the Search Capability in LLMs via Reinforcement Learning.
arXiv preprint arXiv:2503.05592 .
Tang, F.; Gu, Z.; Lu, Z.; Liu, X.; Shen, S.; Meng, C.; Wang,
W.; Zhang, W.; Shen, Y .; Lu, W.; et al. 2025. GUI-G: Gaus-
sian Reward Modeling for GUI Grounding. arXiv preprint
arXiv:2507.15846 .

Team, G.; Anil, R.; Borgeaud, S.; Alayrac, J.-B.; Yu, J.; Sori-
cut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; Millican, K.;
et al. 2023. Gemini: a family of highly capable multimodal
models. arXiv preprint arXiv:2312.11805 .
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition. Transactions of the Association for
Computational Linguistics , 10: 539–554.
Wang, H.; Qian, C.; Zhong, W.; Chen, X.; Qiu, J.; Huang, S.;
Jin, B.; Wang, M.; Wong, K.-F.; and Ji, H. 2025. Otc: Op-
timal tool calls via reinforcement learning. arXiv e-prints ,
arXiv–2504.
Wei, Z.; Yao, W.; Liu, Y .; Zhang, W.; Lu, Q.; Qiu, L.; Yu,
C.; Xu, P.; Zhang, C.; Yin, B.; Yun, H.; and Li, L. 2025.
WebAgent-R1: Training Web Agents via End-to-End Multi-
Turn Reinforcement Learning. arXiv:2505.16421.
Xie, Z.; Cao, J.; Zhang, Y .; Zhang, Q.; and Xu, R. 2025. A
Dual-Agent Adversarial Framework for Robust Generaliza-
tion in Deep Reinforcement Learning. arXiv:2501.17384.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025a. Qwen3
technical report. arXiv preprint arXiv:2505.09388 .
Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.;
Li, C.; Liu, D.; Huang, F.; Wei, H.; et al. 2024. Qwen2. 5
technical report. arXiv preprint arXiv:2412.15115, 2024.
Yang, S.; Dai, Y .; Wang, G.; Zheng, X.; Xu, J.; Li, J.; Ying,
Z.; Wang, W.; and Ngai, E. C. H. 2025b. RealFactBench: A
Benchmark for Evaluating Large Language Models in Real-
World Fact-Checking. arXiv:2506.12538.
Yang, S.; Yu, Z.; Ying, Z.; Dai, Y .; Wang, G.; Lan, J.; Xu,
J.; Li, J.; and Ngai, E. C. H. 2025c. RAMA: Retrieval-
Augmented Multi-Agent Framework for Misinformation
Detection in Multimodal Fact-Checking. arXiv:2507.09174.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Yu, Y .-C.; Chiang, T.-H.; Tsai, C.-W.; Huang, C.-M.; and
Tsao, W.-K. 2025. Primus: A pioneering collection of
open-source datasets for cybersecurity LLM training. arXiv
preprint arXiv:2502.11191 .
Zhang, S.; Dong, Y .; Zhang, J.; Kautz, J.; Catanzaro, B.; Tao,
A.; Wu, Q.; Yu, Z.; and Liu, G. 2025. Nemotron-research-
tool-n1: Tool-using language models with reinforced rea-
soning. arXiv preprint arXiv:2505.00024 .
Zhang, Y .; Li, Y .; Cui, L.; Cai, D.; Liu, L.; Fu, T.; Huang,
X.; Zhao, E.; Zhang, Y .; Chen, Y .; et al. 2023. Siren’s song
in the AI ocean: a survey on hallucination in large language
models. arXiv preprint arXiv:2309.01219 .
Zhang, Y .; Sharma, K.; Du, L.; and Liu, Y . 2024. Toward
mitigating misinformation and social media manipulation in
llm era. In Companion Proceedings of the ACM Web Con-
ference 2024 , 1302–1305.
Zheng, L.; Chiang, W.-L.; Sheng, Y .; Zhuang, S.; Wu, Z.;
Zhuang, Y .; Lin, Z.; Li, Z.; Li, D.; Xing, E.; et al. 2023.
Judging llm-as-a-judge with mt-bench and chatbot arena.Advances in Neural Information Processing Systems , 36:
46595–46623.
Zheng, Y .; Fu, D.; Hu, X.; Cai, X.; Ye, L.; Lu, P.; and Liu, P.
2025. Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments. arXiv preprint
arXiv:2504.03160 .
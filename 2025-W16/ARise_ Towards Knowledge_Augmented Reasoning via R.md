# ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search

**Authors**: Yize Zhang, Tianshu Wang, Sirui Chen, Kun Wang, Xingyu Zeng, Hongyu Lin, Xianpei Han, Le Sun, Chaochao Lu

**Published**: 2025-04-15 06:06:50

**PDF URL**: [http://arxiv.org/pdf/2504.10893v1](http://arxiv.org/pdf/2504.10893v1)

## Abstract
Large language models (LLMs) have demonstrated impressive capabilities and
are receiving increasing attention to enhance their reasoning through scaling
test--time compute. However, their application in open--ended,
knowledge--intensive, complex reasoning scenarios is still limited.
Reasoning--oriented methods struggle to generalize to open--ended scenarios due
to implicit assumptions of complete world knowledge. Meanwhile,
knowledge--augmented reasoning (KAR) methods fail to address two core
challenges: 1) error propagation, where errors in early steps cascade through
the chain, and 2) verification bottleneck, where the explore--exploit tradeoff
arises in multi--branch decision processes. To overcome these limitations, we
introduce ARise, a novel framework that integrates risk assessment of
intermediate reasoning states with dynamic retrieval--augmented generation
(RAG) within a Monte Carlo tree search paradigm. This approach enables
effective construction and optimization of reasoning plans across multiple
maintained hypothesis branches. Experimental results show that ARise
significantly outperforms the state--of--the--art KAR methods by up to 23.10%,
and the latest RAG-equipped large reasoning models by up to 25.37%.

## Full Text


<!-- PDF content starts -->

AR ISE: Towards Knowledge-Augmented Reasoning via Risk-Adaptive
Search
Yize Zhang1,2,3 *Tianshu Wang4,5,7 * Sirui Chen1,6
Kun Wang4Xingyu Zeng4Hongyu Lin5
Xianpei Han5Le Sun5Chaochao Lu1,2†
1Shanghai AI Laboratory2Shanghai Innovation Institute3Shanghai Jiao Tong University
4SenseTime5Institute of Software, Chinese Academy of Sciences6Tongji University
7Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences
ez220523@sjtu.edu.cn, tianshu2020@iscas.ac.cn, luchaochao@pjlab.org.cn
Abstract
Large language models (LLMs) have demon-
strated impressive capabilities and are receiving
increasing attention to enhance their reasoning
through scaling test-time compute. However,
their application in open-ended, knowledge-
intensive, complex reasoning scenarios is still
limited. Reasoning-oriented methods struggle
to generalize to open-ended scenarios due to
implicit assumptions of complete world knowl-
edge. Meanwhile, knowledge-augmented rea-
soning (KAR) methods fails to address two
core challenges: 1) error propagation, where
errors in early steps cascade through the chain,
and 2) verification bottleneck, where the ex-
plore–exploit trade-off arises in multi-branch
decision processes. To overcome these limi-
tations, we introduce AR ISE, a novel frame-
work that integrates risk assessment of interme-
diate reasoning states with dynamic retrieval-
augmented generation (RAG) within a Monte
Carlo tree search paradigm. This approach en-
ables effective construction and optimization of
reasoning plans across multiple maintained hy-
pothesis branches. Experimental results show
thatARISEsignificantly outperforms the state-
of-the-art KAR methods by up to 23.10%, and
the latest RAG-equipped large reasoning mod-
els by up to 25.37%. Our project page is at
https://opencausalab.github.io/ARise.
1 Introduction
Large language models (LLMs) have demonstrated
impressive capabilities across a wide range of
tasks (OpenAI, 2023; Zhao et al., 2023b; Bubeck
et al., 2023). Despite their great success, LLMs
still face fundamental challenges in complex rea-
soning scenarios, hindering their reliable applica-
tion in real-world domains such as science, finance,
and healthcare (Taylor et al., 2022; Li et al., 2023;
Thirunavukarasu et al., 2023). To address this gap,
*Equal contribution.
†Corresponding author.
Error propagation
Verification bottleneck
Search -based ReasoningPrompt -based Reasoning
Figure 1: Error propagation and verification bottle-
neck. Prior works on knowledge-augmented reason-
ing fails to address two core challenges: 1) error prop-
agation, where errors in early steps cascade through
the chain, and 2) verification bottleneck, where the ex-
plore–exploit trade-off arises in multi-branch decision
processes.
recent research has increasingly focused on enhanc-
ing LLM reasoning by scaling test-time compute
to emulate System 2 slow thinking, moving be-
yond System 1 fast responses (Kahneman, 2011;
Snell et al., 2024). Extensive efforts have led to
various approaches, including prompt-based (Yu
et al., 2024a), search-based (Hao et al., 2023), and
learning-based (OpenAI, 2024; DeepSeek-AI et al.,
2025), showing great promise.
However, reasoning-oriented methods struggle
to generalize to open-ended scenarios (Valmeekam
et al., 2022; Amirizaniani et al., 2024), primarily
due to their implicit assumptions of complete world
knowledge. While these solutions like large rea-
soning models (LRMs) have achieved expert or
superhuman performance on tasks such as math
and code, their success relies heavily on clear stan-
dards for search algorithms or reinforcement learn-
ing (Zhang et al., 2024a; Xu et al., 2025). Such
an exclusive focus on enhancing LLM reasoning
implicitly assumes that LLMs already possess allarXiv:2504.10893v1  [cs.AI]  15 Apr 2025

the knowledge necessary for reasoning, which is
often lacking in open-ended or domain-specific
contexts. For example, legal defense requires spe-
cialized jurisprudence knowledge, or medical di-
agnosis demands up-to-date clinical guidelines. In
fact, reasoning is a dynamic process of integrating
multiple knowledge to draw conclusions (Yu et al.,
2024a; OpenAI, 2025), thus making knowledge
acquisition an essential part of reasoning.
Meanwhile, current knowledge-augmented rea-
soning (KAR) methods, as illustrated in Figure 1,
are hindered by error propagation andverification
bottleneck , which undermine reasoning reliabil-
ity. To acquire knowledge for reasoning, retrieval-
augmented generation (RAG) has shown as an ef-
fective way by dynamically retrieving documents
as intermediate results (Lewis et al., 2020; Liu et al.,
2024). Prompt-based methods further extend KAR
through chain-of-thought (CoT) prompting, which
decomposes complex reasoning into sub-steps and
iteratively retrieves relevant knowledge as reason-
ing proceeds (Zhao et al., 2023a; Yu et al., 2024b;
Li et al., 2024). However, this approach are plagued
by error propagation, where errors in early steps
can cascade through the chain. While search-based
methods can mitigate error propagation through
maintaining multiple hypothesis branches, verifica-
tion bottleneck limits the effective explore–exploit
tradeoff in KAR. Existing verification solutions
remain unsatisfactory as they rely on error-prone
self-verification (Stechly et al., 2024; Wang et al.,
2023b; Zhang et al., 2024b), or on specific verifier
training (Setlur et al., 2024; Zhang et al., 2024a).
To overcome these limitations, we present a
novel framework, AR ISE, towards knowledge-
Augmented Reasoning via r Isk-adaptive SEarch.
As shown in Figure 2, AR ISEconsists of three
components: reasoning state generation ,Monte
Carlo tree search (MCTS), and risk assess-
ment . Specifically, AR ISEiterative refines rea-
soning steps through decomposition, retrieval-then-
reasoning to provide fine-grained knowledge for
LLMs (§ 2.1). MCTS treats each step as a node
in the search tree, expanding linear reasoning to
mitigate error propagation by enabling focused ex-
ploration of promising reasoning state and allow-
ing backtracking when necessary (§ 2.2). Risk as-
sessment leverages Bayesian risk minimization to
evaluate the uncertainty of each state, dynamically
balancing explore–exploit trade-off to guide the
search towards both reliable and novel reasoning
directions (§ 2.3). In this way, AR ISEenables ro-bust and efficient complex reasoning by combining
structured decomposition, knowledge retrieval, and
risk-adaptive exploration in a unified framework.
We conducted comprehensive experiments with
multiple LLMs on three challenging multi-hop
question answering (QA) benchmarks that require
complex reasoning and knowledge integration. Ex-
perimental results demonstrate that AR ISEsig-
nificantly outperforms the state-of-the-art (SOTA)
KAR methods, with an average of 23.10% and
15.52% improvement in accuracy and F1. In addi-
tion, when compared to latest LRMs (DeepSeek-AI
et al., 2025) equipped with RAG, AR ISEalso im-
prove the average accuracy and F1 of 4.04% and
25.37%. These results verify the effectiveness of
ARISEfor open-ended, knowledge-intensive, com-
plex reasoning tasks.
To summarize, our contributions are as follows:
•We propose a knowledge-augmented frame-
work for open-ended complex reasoning and
design a risk-adaptive MCTS algorithm to bal-
ance explore–exploit tradeoff for reasoning.
•We conduct comprehensive experiments to ver-
ify the effectiveness of ARISEand to demon-
strate it outperforms SOTA KAR methods and
lastest LRMs equipped with RAG.
•We provide empirical insights that 1) search-
based wide reasoning can explore more solu-
tions than learning-based deep reasoning, and
2)ARISEprogressively approach optimal per-
formance through model size scaling.
2 The AR ISEMethod
Our method, AR ISE, utilizes risk-adaptive tree
search to provide the model with more external
knowledge, thereby effectively enhancing its rea-
soning capabilities. Our pipeline is illustrated in
Figure 2 and comprises the following three parts:
•Reasoning State Generation: The single
step of the policy model1consists of an ac-
tion pair: decomposition and retrieval-then-
reasoning. Each step serves as a node, encod-
ing an intermediate reasoning state.
•Monte Carlo Tree Search: MCTS trans-
forms a sequence of interconnected nodes into
a tree structure. Each node can undergo a sim-
ulated rollout under guidance. The local value
incorporating future reward is able to be up-
dated without incurring the cost of taking an
1We use “policy models” to refer to the LLMs employed
during the inference phase.

Original Question:
When was the last time Peter Till's sports team 
beat the winner of the 1894 -95 FA Cup in SC?
Sub-Question:
What was the sports 
team of Peter Till ?...Peter Till started his career 
with the Birmingham …Documents:Sub-Question:
Who was the 
winner of FA Cup ?…the copy of the trophy was 
from Aston Villa ’s former…Documents:Sub-Question:
When was the last 
time PT’s team in SC?...PT was loaned to Villa 
Park , against Aston Villa…Documents:
Intermediate Result:
Peter Till’s  sports  team 
is Birming -ham City .Intermediate Result:
The winner of the 1894 -
95 FA Cup is Aston Villa .Intermediate Result:
Peter Till played for 
Villa Park in SC in 2015.DecompositionGiven the reasoning state, the 
original question might be 
Low Risk
Move forward
Low Risk
Move forwardHigh Risk
Switch BranchesRetrieval Reasoning:Monte Carlo Tree Search
Reasoning State GenerationRisk Assessment
−1
|𝐪|෍
𝑡𝑙𝑜𝑔𝑝𝐪𝑡𝑞<𝑡,𝐫;Θ)Figure 2: Pipeline of AR ISE.AR ISEiterative refines reasoning steps through decomposition, retrieval-then-
reasoning, providing fine-grained knowledge for LLMs (§ 2.1). MCTS treats each step as a node in the search tree,
expanding linear reasoning to mitigate error propagation by enabling exploration of reasoning paths and allowing
backtracking when necessary (§ 2.2). Risk assessment leverages Bayesian risk minimization to evaluate the quality
of each state, dynamically optimizing action strategies to guide the search towards promising directions (§ 2.3).
actual forward step.
•Risk Assessment: We designed the Risk-
Value function to risk-assess the intermedi-
ate reasoning states at each node. The policy
model is capable of dynamically formulating
and adjusting action strategies based on the
actual risk associated with each branch.
2.1 Reasoning State Generation
To define steps more clearly and granularly, we
prompt LLMs to perform problem decomposition
and retrieval-then-reasoning in an interleaved man-
ner. The two consecutive actions together con-
stitute a single step. Intermediate results at each
step are continuously appended to the entire rea-
soning state and serve as new inputs for subse-
quent steps, progressively approaching the final
solution to complex tasks. This approach, where
intermediate sub-tasks and their labels are concate-
nated to the original task’s input to form a new
input sequence, is widely applied in compounded
tasks (Wies et al., 2023; Rajani et al., 2019; Cobbe
et al., 2021). Specifically, at the ithstep2, the in-
put comprises the original problem qand the in-
termediate results r1,r2, . . ., ri−1from previous
steps, with the latter forming the reasoning state
si−1=r1⊕r2⊕. . .⊕ri−1. The policy model
then decomposes the problem into a subproblem
di, following the policy π(di|q,si−1). Based
2We use the bold notation for vectors and non-bold nota-
tion for scalars.on the subproblem diand the retrieved document,
the intermediate result riis then generated and ap-
pended to the reasoning state repeatedly. Each step
encodes an (si−1,ai)pair, where si−1represents
the state, and aiis a set {di,ri}that implicitly
reflects the step’s two actions, with dibeing the
outcome corresponding to the decomposition and
rito the retrieval-then-reasoning. A sequence of
coherent steps, extending until the endpoint, collec-
tively forms a complete trajectory.
2.2 Monte Carlo Tree Search
The MCTS algorithm expands a single trajectory
into a search tree structure. The whole process
begins with the original problem as the root node,
followed by iterative searches consisting of selec-
tion, expansion, simulation, and back-propagation.
The four phases are detailed as follows:
Selection. Starting from the root node and travers-
ing the existing tree structure, the algorithm se-
lects the optimal child node in preparation for the
next expansion phase. To balance exploration and
exploitation, the well-known Upper Confidence
Bounds (UCT) (Kocsis and Szepesvári, 2006) is
employed in the selection process, formulated as:
UCT(s,a) =Q(s,a) +ws
lnN(Pa(s))
N(s,a),
where N(s,a)andN(Pa(s))represent the visit
counts of the current node and its parent node in

previous searches, respectively. The initial value
ofQ(s,a)is calculated by the Risk-Value func-
tion (detailed in § 2.3) and is subsequently updated
during the back-propagation phase.
Expansion. The model decomposes the original
problem based on the reasoning state from different
perspectives to generate new subproblems. Each
subproblem and its corresponding result form a
distinct child node, which is then appended to the
selected node, thereby expanding the tree in both
width and depth.
Simulation. The model initiates an imagined roll-
out from the selected node, proceeding until it
reaches a leaf node. This phase assists in assigning
the current node a more farsighted value that in-
corporates future rewards by completing the imag-
ined reasoning trajectory without altering the tree
structure. Within a single rollout, the model can
still sample multiple times and greedily advance
towards the leaf nodes.
Back-propagation. The back-propagation phase
updates the values of all nodes along the selected
reasoning branch. This process follows a bottom-
up manner, where the parent node’s value is deter-
mined by the values and visit counts of its child
nodes. The mathematical formulation is as follows:
Q(s,a) =P
c∈C(s,a)Q(c)·N(c)
P
c∈C(s,a)N(c),
where C(s,a)denotes all child nodes of (s,a).
After reaching the predetermined number of
search iterations, the tree structure and node values
stabilize. Ultimately, the model selects the optimal
path by maximizing value at each step, following
the greedy policy.
2.3 Risk Assessment
In this section, we delve into the Risk-Value func-
tion, which risk-assesses the reasoning states to
guide the tree search process. To begin with, for
a composite problem q, we treat its decomposi-
tion and retrieval-then-reasoning as a statistical
decision of a probabilistic process (Zhai and Laf-
ferty, 2006; Lafferty and Zhai, 2001). Specif-
ically, given a set of decomposed subproblems
D={d1,d2, . . ., dk}and the corresponding set
of intermediate results R={r1,r2, . . ., rk}3, the
quality of a node state can be evaluated using a
3In this notation, the subscript of the symbol denotes the
sequence number of the reasoning step, while the superscript
indicates the identifier for different reasoning perspectives.relevance score p(r|q),r∈R(Sachan et al.,
2022). We substitute the “problem generation like-
lihood” (Zhai and Lafferty, 2001; Ponte and Croft,
1998) as an alternative to the relevance score after
applying the Bayes’ rule:
log(r|q) = log p(q|r) + log p(r) +c,
where p(r)is the prior belief that ris relevant to
any problem and is assumed to be uniform in this
case. We can also drop csince it is the intermediate
results-independent constant. The formula is then
reduced to:
log(r|q)∝logp(q|r),∀r∈R,
where p(q|r)captures how well the interme-
diate results rfits the particular problem q. We
utilize the policy model to compute the average
log-likelihood of generating the original problem
tokens in order to estimate logp(q|r)(Sachan
et al., 2022; Yuan et al., 2024), and define the ex-
pected risk of a node (s,a)pointing to ras follows:
Risk((s,a)→r|q) =−1
|q|P
tlogp(qt|q<t,r; Θ),
where |q|denotes the length of the original prob-
lem,qtrepresents the tthtoken in q, and q<trefers
to the sequence of tokens preceding the tthtoken
inq.Θdenotes the parameters of the policy model.
Finally, the risk is scaled to the range (0, 1) through
a sigmoid function in the opposite direction, serv-
ing as the node value:
Q(s,a) = 1−1
1 +eα·(Risk−β),
where α, β is the translation and scaling factors.
3 Experiments
3.1 Setup
Datasets. We use HotpotQA (Yang et al.,
2018), 2WikiMultihopQA (Ho et al., 2020), and
MusiQue (Trivedi et al., 2022) as the test set. These
datasets span a wide range of topics, necessitating
the retrieval and reasoning over multiple supporting
documents. To balance computational efficiency
and evaluation robustness, we conduct experiments
on a subset of 200 randomly selected questions
(Jiang et al., 2024; Feng et al., 2025). More details
are available in § B.1.

MethodHotpotQA 2Wiki MusiQue Average
EM F1 EM F1 EM F1 EM F1
Qwen2.5-14B-Instruct
Vanilla 59.50 63.63 37.00 50.33 14.50 47.07 37.00 53.68
Prompt-based
Query2Doc (Wang et al., 2023a) 61.00 67.65 38.00 53.40 22.00 55.79 40.33 58.95
Self-Ask (Press et al., 2023) 58.50 64.74 38.50 53.45 25.00 58.59 40.67 58.93
Verify-and-Edit (Zhao et al., 2023a) 62.50 70.16 38.50 57.68 22.00 55.30 41.00 61.05
Auto-RAG (Yu et al., 2024b) 68.00 66.64 53.00 55.13 35.50 59.05 52.17 60.27
Search-based
CoT-SC (Wang et al., 2023b) 63.00 71.39 37.50 54.41 16.00 57.46 38.83 61.09
RATT (Zhang et al., 2024b) 64.50 73.91 43.00 57.48 24.00 63.76 43.83 65.05
AR ISE(ours) 73.50 75.39 56.50 62.61 40.50 65.87 56.83 67.96
Qwen2.5-7B-Instruct
Vanilla 54.50 63.63 36.50 50.33 11.00 47.07 34.00 53.68
Prompt-based
Query2Doc (Wang et al., 2023a) 60.00 67.31 37.00 53.78 14.50 54.59 37.17 58.56
Self-Ask (Press et al., 2023) 44.00 61.43 27.50 48.86 22.00 57.48 31.17 55.92
Verify-and-Edit (Zhao et al., 2023a) 67.00 69.87 39.00 53.91 21.50 54.75 42.50 59.51
Auto-RAG (Yu et al., 2024b) 66.50 66.33 44.50 54.00 29.50 57.19 46.83 59.17
Search-based
CoT-SC (Wang et al., 2023b) 60.50 70.66 38.00 54.01 15.00 56.72 37.83 60.46
RATT (Zhang et al., 2024b) 58.50 68.88 36.50 53.91 18.00 56.58 37.67 59.79
AR ISE(ours) 66.50 73.87 47.50 61.37 29.00 62.26 47.67 65.83
Llama3.1-8B-Instruct
Vanilla 55.50 63.63 31.50 50.33 14.00 47.07 33.67 53.68
Prompt-based
Query2Doc (Wang et al., 2023a) 57.50 63.52 32.50 49.78 19.00 49.91 36.33 54.40
Self-Ask (Press et al., 2023) 57.00 62.10 40.00 51.26 20.50 52.14 39.17 55.17
Verify-and-Edit (Zhao et al., 2023a) 50.50 65.07 29.00 51.91 13.50 49.77 31.00 55.58
Auto-RAG (Yu et al., 2024b) 51.00 53.37 35.00 48.81 21.50 52.61 35.83 51.60
Search-based
CoT-SC (Wang et al., 2023b) 64.50 71.40 45.00 58.02 22.00 59.43 43.83 62.96
RATT (Zhang et al., 2024b) 58.50 71.18 46.00 56.18 29.50 63.66 44.67 63.67
AR ISE(ours) 63.00 74.78 34.50 63.19 24.50 66.38 40.67 68.12
Table 1: Comparison of AR ISEwith a wide range of baselines.
Baselines and Metrics. All baselines are incor-
porated with RAG. For prompt-based baselines,
we compare ARISEwith Query2Doc (Wang et al.,
2023a), Self-Ask (Press et al., 2023), Verify-and-
Edit (Zhao et al., 2023a) and Auto-RAG (Yu
et al., 2024b). For search-based baselines, we use
Self-Consistency (SC) (Wang et al., 2023b) and
Retrieval-augmented-thought-tree (RATT) (Zhang
et al., 2024b). We choose EM accuracy and F1
score as the evaluation metrics. The prediction is
correct if the ground truth answer is exactly con-
tained (Jiang et al., 2024; Feng et al., 2025). More
details are available in § B.2.
Implementation Details. In the main experi-
ments, we configure the number of iterations to 200,the exploration weight factor win UCT to 1.4, and
the temperature to 0.7. For the retrieval process,
we employ BM25 as our retriever. Prompts for
forward reasoning directly describe the task with
zero-shot instructions. For verification prompts, we
provide few-shot demonstrations. Further details
and specific prompts are available in § B.4.
3.2 Main Results
Finding 1: AR ISEdemonstrates superior per-
formance. Table 1 presents the comprehensive
experimental results on AR ISEand diverse base-
lines. Specifically, on the Qwen2.5-14B-Instruct
model, ARISEoutperforms across all benchmarks,
achieving an absolute improvement of 19.83% in

EM over the vanilla RAG method, 13.29% over
prompt-based baselines, and 15.5% over search-
based baselines. AR ISEmaintains robust perfor-
mance on the Qwen2.5-7B-Instruct model with an
absolute improvement of 13.67% in EM over the
vanilla RAG method and overall surpasses vari-
ous baselines. We observed that AR ISEperforms
slightly worse on Llama models. To further analyze
this, we identify another interesting phenomenon:
Auto-RAG, which adopts the same interleaved de-
composition and retrieval paradigm as ARISE, also
exhibits a decline on Llama. This phenomenon
suggests that Llama model may not be well-suited
for iterative problem decompositions. In contrast,
continuous-step reasoning approaches such as CoT
and tree-of-thoughts show better results. Neverthe-
less, ARISEstill maintains a notable F1 advantage
on Llama, indicating its effectiveness in selecting
more promising paths.
Finding 2: AR ISEdemonstrates substantial po-
tential on more challenging tasks. We are ex-
cited to show that AR ISEdemonstrates superior
performance on more challenging datasets. Based
on average performance, the difficulty level in-
creases progressively from HotpotQA to 2Wiki-
MultihopQA, and then to MusiQue. In particular,
on the 14B model, AR ISEachieves a relative im-
provement of 23.53% in EM over the vanilla RAG
method on HotpotQA, which surges to 52.70% and
179.31% on 2WikiMultihopQA and MusiQue, re-
spectively. In contrast, a wide range of baselines
show only average performance improvements of
5.74%, 11.94%, and 66.09%, respectively. The
F1 score reflects the same trend, with a relative
improvement of 18.48%, 24.40%, and 39.94%, cor-
responding to the three benchmarks.
3.3 Comparison to Learning-based LRMs
Finding 3: Learning-based LRMs have not yet
approached the point where they can effectively
match or even replace search-based reasoning as
AR ISE.Our empirical comparison between base
models with ARISEand the DeepSeek-R1 distilled
models reveals key insights into the effectiveness
of test-time search. These learning-based LRMs ex-
tract the similar reasoning pattern from DeepSeek-
R1. As shown in Figure 3, AR ISEexhibits a per-
formance advantage over the LRMs, especially on
the Qwen model series. While ARISEslightly un-
derperforms in comparison to DeepSeek-R1-style
reasoning pattern on Llama, it consistently outper-
Llama3.1-8B Qwen2.5-7BQwen2.5-14B Qwen2.5-32B2530354045505560Exact Match (EM)Base Model
Learning-based LRMs
Base Model + ARISEFigure 3: Search-based reasoning vs. learning-based
LRMs. Learning-based LRMs like DeepSeek-R1 dis-
tilled models have not yet approached the point where
they can effectively match or even replace search-based
reasoning methods in terms of performance.
forms it on the Qwen models. On average, ARISE
shows a relative improvement of 4.03%, emphasiz-
ing the benefit of our search-based method. These
results suggest that while learning-based LRMs
like DeepSeek-R1 distilled models provide valu-
able insights, they have not yet demonstrated the
same effectiveness as search-based reasoning.
3.4 Model Scale
Finding 4: AR ISEprogressively approaches the
optimal performance upper bound as model
scale increases, unlocking the potential of
larger models. We conducted experiments on the
Qwen2.5 series models, spanning a range of param-
eter scales from 0.5B to 32B, as illustrated in Fig-
ure 4. To realistically and prominently reflect the re-
lated trends, we selected two more challenging test
sets: 2WikiMultihopQA and Musique. We employ
Pass@N as the metric to evaluate the upper bound
of the success rate, where a problem is considered
solved as long as a single surviving node in the tree
leads to the correct answer. Pass@1, on the other
hand, represents the success rate of the optimal
path selected under the guidance of AR ISE. The
results show that Pass@N and the vanilla method
exhibit similar trends as model parameters scale,
but there is an average of 26.85% significant room
for improvement. This indicates that appropriately
scaling up inference computation offers substantial
potential for enhancing performance. We observed
that after the model size reaches 7B parameters,
the performance of both the upper bound and naive

203040506070Exact Match (EM)
2Wiki
Vanilla RAG
ARISE Pass@1
ARISE Pass@N
0.53 7 14 32
Parameters (109)01020304050Exact Match (EM)
Musique
Vanilla RAG
ARISE Pass@1
ARISE Pass@NFigure 4: Performance vs. model scales. Although
scaling up the model size shows diminishing returns for
vanilla approaches, ARISEbetter harnesses the potential
of larger models in solving complex tasks.
retrieval tends to saturate, suggesting diminishing
returns with further scaling of model parameters. In
contrast, AR ISEdemonstrates consistent improve-
ment as model parameters increase. The accuracy
of the optimal path selection gradually approaches
the upper bound, with the success rate gap between
Pass@1 and Pass@N decreasing from 25.00% to
7.25%.
3.5 Ablation Studies
3.5.1 Risk Assessment
Finding 5: The Risk-Value function effectively
risk-assess reasoning states to guide the tree
search process. We conducted ablation studies
to evaluate the effectiveness of the Risk-Value (R-
V) function in guiding Monte Carlo Tree Search
(MCTS). Table 2 present experimental results com-
pared to vanilla MCTS and MCTS with LLM-
as-Judge4baselines. The incorporation of the R-
V function resulted in improvements across all
datasets. Specifically, it achieved an average rela-
4To ensure experimental fairness, we employed the same
LLM (policy models) for methods involving LLM-as-verifier.tive performance gain of 10.71% over the vanilla
MCTS baseline. The function’s impact was even
more pronounced in more challenging tasks, with
improvements reaching up to 17.39% on MusiQue.
This demonstrates the function’s capacity to better
navigate and prioritize lower-risk paths, ensuring
more efficient exploration and exploitation within
the search space. In comparison, MCTS with LLM-
as-Verifier showed marginal improvements over the
vanilla approach. While pretrained LLMs can pro-
vide meaningful context during verification, they
are not specifically tuned for evaluating the quality
of reasoning states in a dynamic environment. This
suggests that pretrained LLMs are insufficient as
standalone verifiers in path planning, underscor-
ing the critical role of specialized functions like
Risk-Value in guiding the search process.
3.5.2 Iterations of MCTS
Finding 6: AR ISEwith dynamic risk assessment
achieves near-optimal solutions with a relatively
low inference cost. We conduct empirical exper-
iments on the trade-off between inference cost and
solution quality. Specifically, we examined the
performance of AR ISEas the number of MCTS
iterations increased from 1 to 200. The Figure 5
illustrates the efficiency of ARISEin reaching near-
optimal solutions with a relatively low inference
cost. As the number of MCTS iterations increases,
100
101
102
Iterations505254565860Performance
Trend Qwen-14B
Trend Qwen-7B
Qwen-14B
Qwen-7B
Figure 5: Ablation on iterations. We evaluate perfor-
mance using the average of the EM accuracy and the F1
score.
the performance improves, but the rate of improve-
ment diminishes with higher iteration counts. At
the initial stages of the search process (from 1 to 30
iterations), the performance shows rapid improve-
ments. The initial exploration of potential moves

MethodHotpotQA 2Wiki MusiQue Average
EM F1 EM F1 EM F1 EM F1
MCTS (R-V function) 73.50 75.39 56.50 62.61 40.50 65.98 56.83 67.96
MCTS (vanilla) 71.00 73.58 48.50 60.47 34.50 63.34 51.33 65.80
MCTS (LLM-as-verifier) 69.00 73.85 53.50 60.98 34.00 64.30 52.17 66.38
Table 2: Ablation on the Risk-Value function. The incorporation of the Risk-Value function resulted in improve-
ments across all datasets. The Risk-Value function effectively assesses reasoning states, guiding the tree search
process; however, pretrained LLMs are not adequate as verifiers.
encourages the model’s better understanding of the
search space. The subsequent increase in the num-
ber of additional paths during this phase contributes
meaningfully to the quality of the solution. Beyond
30 iterations, the improvements in performance be-
gin to level off. For instance, between 30 and 60
iterations, the performance increases slightly by a
relative 1.32%, whereas during the initial phase,
the increase was 5.76%. This suggests that further
exploration of the search space yields diminishing
returns as the algorithm begins to converge toward
the optimal decision. This phenomenon can be at-
tributed to that ARISEprovides dynamic step-level
Risk-Value, without requiring the model to wait
for outcome verification to guide the next iteration.
The Risk-Value function efficiently narrows down
the promising branches at a rapid pace. The perfor-
mance stabilizes after approximately 100 iterations.
Further iterations may lead to incremental improve-
ments but are less impactful in a highly explored
search space.
4 Related works
Test-Time Compute. Scaling test-time compute
has become a popular topic recently in many re-
search efforts (Brown et al., 2024; Snell et al.,
2024), aiming to shift the reasoning paradigm of
LLMs from the fast but error-prone system 1 think-
ing to the more deliberate system 2 thinking (Kah-
neman, 2011; Snell et al., 2024). Prior works
mainly include prompt-based (Yu et al., 2024a),
search-based (Hao et al., 2023), and learning-
based (OpenAI, 2024; DeepSeek-AI et al., 2025)
methods. Prompt-based reasoning utilize chain-
of-thought (CoT) (Wei et al., 2022) prompting
to break down complex reasoning into sub-steps,
gradually approaching a more accurate final an-
swer through the generation of additional inter-
mediate tokens (Zhao et al., 2023a; Yu et al.,
2024b; Li et al., 2024). Search-based reasoning
allows LLMs to improve performance by gener-ating multiple samples, and tree-based methods
have further integrated planning and exploration
(Yao et al., 2023; Besta et al., 2024; Hao et al.,
2023). While the multiple and redundant rollouts
significantly burden the inference spend, verifiers
for solution-selection is essential for ensuring ef-
ficiency. Learning-based methods aim to inject
the deep reasoning patterns of large and compli-
cated models into smaller models through post-
training (OpenAI, 2024; DeepSeek-AI et al., 2025).
Retrieval-Augmented Generation. Retrieval-
Augmented Generation (RAG) merges the intrinsic
knowledge of LLMs with a vast, dynamic reposi-
tory of external databases, mitigating the issues of
language model hallucination and outdated knowl-
edge to some extent (Lewis et al., 2020; Gao et al.,
2023). Recent studies (Wang et al., 2023a; Press
et al., 2023; Yu et al., 2024b; Zhao et al., 2023a)
have proposed some prompting-based strategies
for LLMs to better harness the potential of RAG,
essentially integrating it into the intermediate rea-
soning process (e.g., chain-of-thought (CoT) (Wei
et al., 2022)). In these methods, the interaction
between LLMs and retrieval actions breaks down
the reasoning process into discontinuous smaller
steps, which helps produce more authentic interme-
diate results and reduces the instability inherent in
autoregressive token generation.
5 Conclusion
In this work, we introduce AR ISE, a novel frame-
work for knowledge-augmented reasoning that ad-
dresses the challenges of error propagation and
verification bottleneck in complex reasoning tasks.
By integrating MCTS with risk-adaptive explo-
ration, AR ISEenables dynamic and effective rea-
soning through iterative decomposition, retrieval-
then-reasoning steps. Our experiments demonstrate
that AR ISEoutperforms a wide range of state-
of-the-art methods, and also surpasses the perfor-

mance of the latest learning-based LRMs equipped
with RAG. These results highlight the potential
ofAR ISEin advancing open-ended, knowledge-
intensive, and complex reasoning tasks in real-
world applications.
Limitations
Although our method ARISEdemonstrates strong
performance for knowledge-intensive, and com-
plex reasoning tasks, several limitations remain
open for improvement. Our current work focuses
test-time search and leaves the training of the pol-
icy model for future research. While we have
demonstrated the effectiveness of our method in
dynamically guiding the solution-search process,
a learning-based LRMs could potentially further
optimize the trade-off between performance and
inference cost. The prompts employed in our ex-
periments were selected based on empirical trials
rather than a systematic optimization process. Al-
though these prompts have shown to be effective
in our tasks, a more rigorous approach to prompt
design and tuning could lead to improved gener-
alization and robustness across diverse scenarios.
Our experiments are currently confined to multi-
hop question-answering (QA) tasks. The applica-
bility of AR ISEto other reasoning tasks, such as
mathematical problem-solving, code generation, or
complex decision-making, remains to be explored.
Extending our method to a broader range of rea-
soning tasks is an important direction for future
work.
References
Maryam Amirizaniani, Elias Martin, Maryna
Sivachenko, Afra Mashhadi, and Chirag Shah. 2024.
Can llms reason like humans? assessing theory of
mind reasoning in llms for open-ended questions. In
Proceedings of the 33rd ACM International Confer-
ence on Information and Knowledge Management,
CIKM 2024, Boise, ID, USA, October 21-25, 2024 ,
pages 34–44. ACM.
Maciej Besta, Nils Blach, Ales Kubicek, Robert Ger-
stenberger, Michal Podstawski, Lukas Gianinazzi,
Joanna Gajda, Tomasz Lehmann, Hubert Niewiadom-
ski, Piotr Nyczyk, and Torsten Hoefler. 2024. Graph
of thoughts: Solving elaborate problems with large
language models. In Thirty-Eighth AAAI Conference
on Artificial Intelligence, AAAI 2024, Thirty-Sixth
Conference on Innovative Applications of Artificial
Intelligence, IAAI 2024, Fourteenth Symposium on
Educational Advances in Artificial Intelligence, EAAI
2014, February 20-27, 2024, Vancouver, Canada ,
pages 17682–17690. AAAI Press.Bradley C. A. Brown, Jordan Juravsky, Ryan Saul
Ehrlich, Ronald Clark, Quoc V . Le, Christopher Ré,
and Azalia Mirhoseini. 2024. Large language mon-
keys: Scaling inference compute with repeated sam-
pling. CoRR , abs/2407.21787.
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan,
Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter
Lee, Yin Tat Lee, Yuanzhi Li, Scott M. Lundberg,
Harsha Nori, Hamid Palangi, Marco Túlio Ribeiro,
and Yi Zhang. 2023. Sparks of artificial general
intelligence: Early experiments with GPT-4. CoRR ,
abs/2303.12712.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems. CoRR , abs/2110.14168.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong
Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang,
Archi Mitra, Archie Sravankumar, Artem Korenev,
Arthur Hinsvark, Arun Rao, Aston Zhang, and 82
others. 2024. The llama 3 herd of models. CoRR ,
abs/2407.21783.
Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Jingyi
Song, and Hao Wang. 2025. Airrag: Activating in-
trinsic reasoning for retrieval augmented generation
via tree-based search. Preprint , arXiv:2501.10053.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. CoRR , abs/2312.10997.
Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen
Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Rea-
soning with language model is planning with world
model. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, EMNLP 2023, Singapore, December 6-10, 2023 ,
pages 8154–8173. Association for Computational
Linguistics.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing A multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. In Proceedings of the 28th International
Conference on Computational Linguistics, COLING
2020, Barcelona, Spain (Online), December 8-13,
2020 , pages 6609–6625. International Committee on
Computational Linguistics.

Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie
Wang, Wayne Xin Zhao, Yang Song, and Tao Zhang.
2024. Rag-star: Enhancing deliberative reasoning
with retrieval augmented verification and refinement.
CoRR , abs/2412.12881.
Daniel Kahneman. 2011. Thinking, fast and slow . Far-
rar, Straus and Giroux.
Levente Kocsis and Csaba Szepesvári. 2006. Bandit
based monte-carlo planning. In Machine Learning:
ECML 2006, 17th European Conference on Machine
Learning, Berlin, Germany, September 18-22, 2006,
Proceedings , volume 4212 of Lecture Notes in Com-
puter Science , pages 282–293. Springer.
John D. Lafferty and ChengXiang Zhai. 2001. Docu-
ment language models, query models, and risk min-
imization for information retrieval. In SIGIR 2001:
Proceedings of the 24th Annual International ACM
SIGIR Conference on Research and Development in
Information Retrieval, September 9-13, 2001, New
Orleans, Louisiana, USA , pages 111–119. ACM.
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Advances in Neu-
ral Information Processing Systems 33: Annual Con-
ference on Neural Information Processing Systems
2020, NeurIPS 2020, December 6-12, 2020, virtual .
Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng
Ding, Shafiq Joty, Soujanya Poria, and Lidong Bing.
2024. Chain-of-knowledge: Grounding large lan-
guage models via dynamic knowledge adapting over
heterogeneous sources. In The Twelfth International
Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net.
Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen.
2023. Large language models in finance: A survey.
In4th ACM International Conference on AI in Fi-
nance, ICAIF 2023, Brooklyn, NY, USA, November
27-29, 2023 , pages 374–382. ACM.
Jingyu Liu, Jiaen Lin, and Yong Liu. 2024. How
much can RAG help the reasoning of llm? CoRR ,
abs/2410.02338.
OpenAI. 2023. GPT-4 technical report. CoRR ,
abs/2303.08774.
OpenAI. 2024. Learning to reason with
llms. https://openai.com/index/
learning-to-reason-with-llms/ .
OpenAI. 2025. Introducing deep re-
search. https://openai.com/index/
introducing-deep-research/ .
Jay M. Ponte and W. Bruce Croft. 1998. A language
modeling approach to information retrieval. In SIGIR
’98: Proceedings of the 21st Annual InternationalACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, August 24-28 1998,
Melbourne, Australia , pages 275–281. ACM.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. In Findings of the Association for Compu-
tational Linguistics: EMNLP 2023, Singapore, De-
cember 6-10, 2023 , pages 5687–5711. Association
for Computational Linguistics.
Nazneen Fatema Rajani, Bryan McCann, Caiming
Xiong, and Richard Socher. 2019. Explain your-
self! leveraging language models for commonsense
reasoning. In Proceedings of the 57th Conference of
the Association for Computational Linguistics, ACL
2019, Florence, Italy, July 28- August 2, 2019, Vol-
ume 1: Long Papers , pages 4932–4942. Association
for Computational Linguistics.
Devendra Singh Sachan, Mike Lewis, Mandar Joshi,
Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and
Luke Zettlemoyer. 2022. Improving passage retrieval
with zero-shot question generation. In Proceedings
of the 2022 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2022, Abu
Dhabi, United Arab Emirates, December 7-11, 2022 ,
pages 3781–3797. Association for Computational
Linguistics.
Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang
Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh
Agarwal, Jonathan Berant, and Aviral Kumar. 2024.
Rewarding progress: Scaling automated process veri-
fiers for LLM reasoning. CoRR , abs/2410.08146.
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Ku-
mar. 2024. Scaling LLM test-time compute optimally
can be more effective than scaling model parameters.
CoRR , abs/2408.03314.
Kaya Stechly, Karthik Valmeekam, and Subbarao Kamb-
hampati. 2024. On the self-verification limitations
of large language models on reasoning and planning
tasks. CoRR , abs/2402.08115.
Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas
Scialom, Anthony Hartshorn, Elvis Saravia, An-
drew Poulton, Viktor Kerkez, and Robert Stojnic.
2022. Galactica: A large language model for science.
CoRR , abs/2211.09085.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Arun James Thirunavukarasu, Darren Shu Jeng Ting,
Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan,
and Daniel Shu Wei Ting. 2023. Large language
models in medicine. Nature Medicine , 29(8):1930–
1940.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics , 10:539–554.

Karthik Valmeekam, Alberto Olmo Hernandez, Sarath
Sreedharan, and Subbarao Kambhampati. 2022.
Large language models still can’t plan (A benchmark
for llms on planning and reasoning about change).
CoRR , abs/2206.10498.
Liang Wang, Nan Yang, and Furu Wei. 2023a.
Query2doc: Query expansion with large language
models. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, EMNLP 2023, Singapore, December 6-10, 2023 ,
pages 9414–9423. Association for Computational
Linguistics.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V .
Le, Ed H. Chi, Sharan Narang, Aakanksha Chowd-
hery, and Denny Zhou. 2023b. Self-consistency
improves chain of thought reasoning in language
models. In The Eleventh International Conference
on Learning Representations, ICLR 2023, Kigali,
Rwanda, May 1-5, 2023 . OpenReview.net.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompting
elicits reasoning in large language models. In Ad-
vances in Neural Information Processing Systems 35:
Annual Conference on Neural Information Process-
ing Systems 2022, NeurIPS 2022, New Orleans, LA,
USA, November 28 - December 9, 2022 .
Noam Wies, Yoav Levine, and Amnon Shashua. 2023.
Sub-task decomposition enables learning in sequence
to sequence tasks. In The Eleventh International
Conference on Learning Representations, ICLR 2023,
Kigali, Rwanda, May 1-5, 2023 . OpenReview.net.
Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang,
Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui
Gong, Tianjian Ouyang, Fanjin Meng, Chenyang
Shao, Yuwei Yan, Qinglong Yang, Yiwen Song, Si-
jian Ren, Xinyuan Hu, Yu Li, Jie Feng, Chen Gao,
and Yong Li. 2025. Towards large reasoning models:
A survey of reinforced reasoning with large language
models.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. In Proceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
Brussels, Belgium, October 31 - November 4, 2018 ,
pages 2369–2380. Association for Computational
Linguistics.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2023. Tree of thoughts: Deliberate problem solving
with large language models. In Advances in Neural
Information Processing Systems 36: Annual Confer-
ence on Neural Information Processing Systems 2023,
NeurIPS 2023, New Orleans, LA, USA, December 10
- 16, 2023 .Fei Yu, Hongbo Zhang, Prayag Tiwari, and Benyou
Wang. 2024a. Natural language reasoning, A survey.
ACM Comput. Surv. , 56(12):304:1–304:39.
Tian Yu, Shaolei Zhang, and Yang Feng. 2024b. Auto-
rag: Autonomous retrieval-augmented generation for
large language models. CoRR , abs/2411.19443.
Xiaowei Yuan, Zhao Yang, Yequan Wang, Jun Zhao,
and Kang Liu. 2024. Improving zero-shot LLM re-
ranker with risk minimization. In Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024 , pages 17967–17983.
Association for Computational Linguistics.
ChengXiang Zhai and John D. Lafferty. 2001. A study
of smoothing methods for language models applied
to ad hoc information retrieval. In SIGIR 2001: Pro-
ceedings of the 24th Annual International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, September 9-13, 2001, New
Orleans, Louisiana, USA , pages 334–342. ACM.
ChengXiang Zhai and John D. Lafferty. 2006. A risk
minimization framework for information retrieval.
Inf. Process. Manag. , 42(1):31–55.
Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue,
Yuxiao Dong, and Jie Tang. 2024a. Rest-mcts*:
LLM self-training via process reward guided tree
search. In Advances in Neural Information Process-
ing Systems 38: Annual Conference on Neural In-
formation Processing Systems 2024, NeurIPS 2024,
Vancouver, BC, Canada, December 10 - 15, 2024 .
Jinghan Zhang, Xiting Wang, Weijieying Ren, Lu Jiang,
Dongjie Wang, and Kunpeng Liu. 2024b. RATT:
A thought structure for coherent and correct LLM
reasoning. CoRR , abs/2406.02746.
Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei
Qin, and Lidong Bing. 2023a. Verify-and-edit: A
knowledge-enhanced chain-of-thought framework.
InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), ACL 2023, Toronto, Canada, July
9-14, 2023 , pages 5823–5840. Association for Com-
putational Linguistics.
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen
Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, and
3 others. 2023b. A survey of large language models.
CoRR , abs/2303.18223.

A Additional Results
A.1 Discussion on Computational Overhead
We analyze the computational overhead of ARISE
from two perspectives: (1) its relation to the search
space, and (2) its comparison with diverse base-
lines. All experiments were conducted using two
NVIDIA H100 GPUs (80GB), and the cost is pri-
marily measured in terms of reasoning time (in
minutes), which reflects real-time efficiency during
inference.
(Depth, Width) EM F1 Time (min)
Vanilla / (1,1) 56.00 59.70 10
(3,3) 63.50 65.84 40
(3,4) 65.00 65.69 52
(3,5) 64.50 66.63 66
(3,6) 65.00 64.91 83
(4,4) 66.50 65.36 119
(4,5) 66.00 67.01 168
(4,6) 67.00 66.29 266
(5,4) 63.50 65.52 189
(5,5) 66.00 66.24 413
(5,6) 65.50 65.63 770
Table 3: Computational overhead in relation to the
search space. As the search space expands in depth
and width, performance improves but with diminishing
returns and significantly increased reasoning time. The
(4,5) setting achieves a good trade-off and is adopted in
our main experiments.
Computational Overhead in Relation to Search
Space. We conducted experiments on the Hot-
potQA dataset using the Llama3.1-8B-Instruct
model. The results are under different search space
configurations—defined by search depth and width.
As shown in the Table 3, expanding the search
space leads to performance improvements (in EM
and F1 scores), especially when increasing from a
vanilla setting (depth 1, width 1) to moderate con-
figurations. However, these improvements dimin-
ish as the depth and width increase further, while
the computational overhead escalates rapidly. For
instance, increasing the depth from 3 to 4 and the
width from 4 to 6 results in a reasoning time in-
crease from 52 to 266 minutes, with only marginal
performance gain. Based on this observation, we
adopted a configuration of depth = 4 and width =
5 for our main experiments, which strikes a prac-
tical balance between accuracy and computational
cost. It is worth noting that the optimal search
space configuration may vary across instances. Dif-ferent questions demand varying depths of reason-
ing and degrees of knowledge integration, often
corresponding to different numbers of reasoning
hops. Thus, adaptively determining the search
space based on task complexity remains an open
research problem. Even state-of-the-art reasoning
models still face the challenge of choosing between
wider and deeper reasoning paths. We discuss this
further in the § 5 and identify it as a promising
direction for future work.
Method EM F1 Time (min)
Vanilla 14.50 47.07 10
Query2Doc 22.00 55.79 16
Self-Ask 25.00 58.59 18
Verify-and-Edit 22.00 55.30 21
Auto-RAG 35.50 59.05 26
CoT-SC 16.00 57.46 69
RATT 24.00 63.76 155
AR ISE(ours) 40.50 65.87 160
Table 4: Computational overhead comparison with
baselines. AR ISEachieves the best performance in
both EM and F1 scores but incurs higher reasoning time
due to its search-based multi-step reasoning. Practical
deployment should consider the trade-off between com-
putational resources and desired performance.
Computational Overhead Comparison with
Baselines. We further evaluate the computational
overhead of our method, AR ISE, in comparison
with a set of competitive baselines on the Musique
dataset using the Qwen2.5-14B-Instruct model.
The results are summarized in Table 4. As shown,
AR ISEachieves the highest EM and F1 scores,
demonstrating the effectiveness of our search-based
reasoning strategy. However, this comes at a higher
computational cost, primarily due to the enlarged
search space and multi-step reasoning process. In
practical applications, the trade-off between perfor-
mance and reasoning time should be contextualized
by the nature of the task at hand. In high-stakes
domains such as finance, healthcare, and law, sacri-
ficing additional computational time for improved
accuracy is often a worthwhile investment.
A.2 Impact of Retrieval System Quality
To evaluate how sensitive AR ISEis to the quality
of the underlying retrieval system, we examine two
key factors: the type of retriever and the number of
retrieved documents.

Dataset BM25 EM BM25 F1 BGE EM BGE F1
Musique 40.50 65.87 44.50 70.47
HotpotQA 73.50 75.39 74.00 75.41
2Wiki 56.50 62.61 72.00 67.36
Table 5: Impact of Retriever Type. Dense retriever
(BGE) outperforms sparse retriever (BM25) across all
datasets, with more notable gains on challenging bench-
marks.
Choice of Retriever. We compare the perfor-
mance of AR ISEwhen paired with a sparse re-
triever (BM25) versus a dense retriever (BGE).
As summarized in Table 5, dense retrieval con-
sistently outperforms sparse retrieval across all
three datasets. On average, BGE improves EM
and F1 scores by 11.7%, with particularly substan-
tial gains observed on more challenging datasets
such as 2Wiki. This indicates that while AR ISE
exhibits some robustness to retrieval quality on
simpler tasks (e.g., HotpotQA), it substantially ben-
efits from higher-quality evidence in more complex
scenarios. These results suggest that although the
verifier-guided reasoning in AR ISEcan partially
mitigate retrieval noise, it still relies on access to
relevant and precise information to perform effec-
tive multi-hop reasoning.
#Doc EM F1
1 28.50 44.14
2 40.50 65.87
3 37.00 63.97
Table 6: Effect of the Number of Retrieved Docu-
ments. Performance peaks at retrieving two documents,
suggesting a balance between sufficient evidence and
retrieval noise.
Number of Retrieved Documents. We further
assess the impact of varying the number of re-
trieved documents on the Musique dataset. As
shown in Table 6, retrieving two documents yields
the best performance. Adding a third document
does not lead to further improvements and may
even slightly degrade performance. This can be
attributed to two factors. First, the knowledge den-
sity of the task affects how much additional context
is beneficial. Second, due to ARISE’s step-by-step
reasoning design, the overall problem is decom-
posed into smaller, more manageable subproblems,
each requiring less contextual information. In a nut-
shell, AR ISEis more sensitive to missing criticalinformation than to receiving redundant or noisy
evidence, exhibiting a degree of robustness to re-
trieval noise.
B Further Details
B.1 Further Details for Datasets
We use HotpotQA (Yang et al., 2018), 2WikiMul-
tihopQA (Ho et al., 2020), and MusiQue (Trivedi
et al., 2022) as the test set, which are three repre-
sentative benchmarks for open-ended, knowledge-
intensive, and complex reasoning tasks. The ques-
tions in these datasets require retrieving and reason-
ing over multiple supporting documents to answer,
and they cover a wide range of topics without be-
ing constrained by any existing knowledge base
or schema. We performed preprocessing on the
dataset. Specifically, considering the limitations
of computational resources, we randomly sampled
200 questions from each dataset as the final test set.
Each instance includes the original question and its
answer, along with possible reference documents
and necessary supporting documents. During the
testing phase, we treated the possible reference
documents for each question as its external knowl-
edge base and employed BM25 as our retriever,
with each retrieval returning the top two documents.
Based on the original dataset, the majority of ques-
tions involve complex reasoning with three or more
hops (over 80%) (Feng et al., 2025). Among them,
Musique has the highest reasoning difficulty, with
a notable higher proportion of multi-hop questions.
We fixed the search depth to four layers to align
with the number of sub-question decompositions
required for problem-solving, thereby reducing un-
necessary reasoning overhead. Additionally, we set
the initial maximum number of expandable child
nodes to 5 to cover hop counts comprehensively,
guaranteeing diversity in the model’s decomposi-
tion of questions. As the search depth increases, the
diversity of decomposition perspectives for original
questions gradually decreases. Therefore, we also
progressively reduced the number of expandable
child nodes, which ensures search efficiency.
B.2 Further Details for Evaluations
We choose EM accuracy and F1 score as the evalu-
ation metrics. EM accuracy measures the success
rate of the results, while F1 score evaluates the re-
liability of the reasoning process. Specifically, for
EM, we adopt the covered EM calculation method.
We preprocess both the model’s predictions and

the ground truth answers by converting them to a
uniform case format. If the ground truth answer is
a substring of the predicted result, we consider the
prediction correct. This approach aims to genuinely
reflect the method’s performance. Additionally, we
employ concise response prompts to ensure the
model’s final output is not overly verbose, thereby
avoiding false-positive cases. For the calculation of
F1 score, we utilize the top two documents related
to the entire reasoning path and original question.
At the final step, we perform an additional retrieval
based on the complete reasoning state to ensure
the documents reflect the context of the entire path.
To prevent redundant retrievals from obscuring the
comparison of F1 scores, we limit each retrieval to
return only the top two documents.
B.3 Further Details for LLMs
Throughout the experiments, we primarily utilized
the Qwen series (Team, 2024) and Llama series
models (Dubey et al., 2024). The Qwen series in-
cludes models with scales of 0.5B, 3B, 7B, 14B,
and 32B parameters, while the experiments with
Llama were mainly conducted on the 8B parameter
model. To ensure fairness in the experiments, for
any tasks involving risk assessment or state evalua-
tion, we consistently employed the corresponding
policy models. In addition, experiments also in-
volved DeepSeek-R1 distilled models (DeepSeek-
AI et al., 2025), including Qwen-7B, Qwen-14B,
Qwen-32B, and Llama-8B. The distilled small
model extracts reasoning patterns from DeepSeek-
R1 and demonstrates superior performance com-
pared to the inference patterns obtained through
reinforcement learning (DeepSeek-AI et al., 2025).
B.4 Further Details for Prompts
We list the full details of all prompts employed in
AR ISEas follows. The prompts used for forward
inference follow a zero-shot instruction to directly
describe the task. We provide few-shot demonstra-
tions in the prompts for risk assessment.

Algorithm 1 ARISE
Require: original problem q; prompt pdfor decomposition; prompt prfor reasoning
Require: LLM f(·); external knowledge retrieval model g(·)
Require: selecte Se(·); expand Ex(·); simulate Si(·); back-propagate Ba(·)
Require: prompt pgfor problem generation; value function V(·); get best path P(·)
s0;t←q ▷ Generate root node and tree
fori= 1, . . . , T do ▷Perform T MCTS iterations
si←Se(s0,t) ▷Select the best node
di←f(si, q, p d) ▷Decompose the problem
zi←g(di) ▷Retrieve relevant documents
ri←f(zi, di, pr) ▷Reason forward
si+1←si, ri, V(f(ri, q, p g)) ▷Generate new nodes
t←Ex(si+1) ▷Append nodes to the tree
t←Si(si);Ba(si,s0) ▷Simulate and Back-propagate
end for
return P(s0,t) ▷Get best path

Problem Decomposition
Your task is to decompose the original question into one smaller sub -question based on the Intermediate answer and Observation.
The decomposed process is encouraged to be done from multiple perspectives.
Output a thought to reason the original question, and output one sub -question that you think is appropriate to solve next.
DO NOT REPEAT the question and DO NOT try to answer the question.
The output format is limited to:
Thought: ...
Sub-question: ...
Here, the "..." indicates omitted output information that you need to fill in.
Original question: { original question }
Intermediate answers: { reasoning state }
Observation: { retrieved documents }
Output:
Intermediate Answer Generation
Your task is to answer the following question using provided supporting facts.
The output answer should be a complete declarative sentence, rather than directly outputting phrases or words.
Do not use pronouns in the sentence.
Specially, if no provided supporting facts, just output "No directly relevant facts found." and nothing else.
Question: { sub-question }
Supporting facts: { retrieved documents }
Output:
Your task is to answer the original question based on the intermediate answers.
Output the final answer directly and nothing else.
Original question: { original question }
Intermediate answers: { reasoning state }
Output :Final Answer Generation

Risk Assessment
Given intermediate answer containing the facts about the original question, which is unknown, your task is to infer what the 
original question might have been.
Output the most likely original question directly and nothing else.
Example 1:
Intermediate answer:
Muhammad Ali was 74 years old when he died.
Alan Turing was 41 years old when he died.
The original question might be:
Who lived longer, Muhammad Ali or Alan Turing?
Example 2:
Intermediate answer:
Craigslist was founded by Craig Newmark.
Craig Newmark was born on December 6, 1952.
The original question might be:
When was the founder of craigslist born?
Intermediate answers: { reasoning state }
The original question might be:
LLM -as-Verifier
Given a question, your task is to determine the consistency score of 
its decomposed sub -questions and corresponding intermediate 
answers with the original question.
Directly output JUST A NUMBER between 0 and 10 to represent the 
consistency score.
Do not output anything else.
Original question: { original question }
Sub-questions: { sub-questions }
Intermediate answers: { reasoning state }
Output:
# Uncertainty Quantification for Retrieval-Augmented Reasoning

**Authors**: Heydar Soudani, Hamed Zamani, Faegheh Hasibi

**Published**: 2025-10-13 14:55:28

**PDF URL**: [http://arxiv.org/pdf/2510.11483v1](http://arxiv.org/pdf/2510.11483v1)

## Abstract
Retrieval-augmented reasoning (RAR) is a recent evolution of
retrieval-augmented generation (RAG) that employs multiple reasoning steps for
retrieval and generation. While effective for some complex queries, RAR remains
vulnerable to errors and misleading outputs. Uncertainty quantification (UQ)
offers methods to estimate the confidence of systems' outputs. These methods,
however, often handle simple queries with no retrieval or single-step
retrieval, without properly handling RAR setup. Accurate estimation of UQ for
RAR requires accounting for all sources of uncertainty, including those arising
from retrieval and generation. In this paper, we account for all these sources
and introduce Retrieval-Augmented Reasoning Consistency (R2C)--a novel UQ
method for RAR. The core idea of R2C is to perturb the multi-step reasoning
process by applying various actions to reasoning steps. These perturbations
alter the retriever's input, which shifts its output and consequently modifies
the generator's input at the next step. Through this iterative feedback loop,
the retriever and generator continuously reshape one another's inputs, enabling
us to capture uncertainty arising from both components. Experiments on five
popular RAR systems across diverse QA datasets show that R2C improves AUROC by
over 5% on average compared to the state-of-the-art UQ baselines. Extrinsic
evaluations using R2C as an external signal further confirm its effectiveness
for two downstream tasks: in Abstention, it achieves ~5% gains in both
F1Abstain and AccAbstain; in Model Selection, it improves the exact match by
~7% over single models and ~3% over selection methods.

## Full Text


<!-- PDF content starts -->

Uncertainty Quantification for Retrieval-Augmented Reasoning
Heydar Soudani
Radboud University
Nijmegen, The Netherlands
heydar.soudani@ru.nlHamed Zamani
University of Massachusetts Amherst
Amherst, United States
zamani@cs.umass.eduFaegheh Hasibi
Radboud University
Nijmegen, The Netherlands
faegheh.hasibi@ru.nl
Abstract
Retrieval-augmented reasoning (RAR) is a recent evolution of re-
trieval-augmented generation (RAG) that employs multiple rea-
soning steps for retrieval and generation. While effective for some
complex queries, RAR remains vulnerable to errors and misleading
outputs. Uncertainty quantification (UQ) offers methods to estimate
the confidence of systems’ outputs. These methods, however, often
handle simple queries with no retrieval or single-step retrieval,
without properly handling RAR setup. Accurate estimation of UQ
for RAR requires accounting for all sources of uncertainty, includ-
ing those arising from retrieval and generation. In this paper, we
account for all these sources and introduce Retrieval-Augmented
Reasoning Consistency ( R2C)–a novel UQ method for RAR. The
core idea of R2C is to perturb the multi-step reasoning process by
applying various actions to reasoning steps. These perturbations
alter the retriever’s input, which shifts its output and consequently
modifies the generator’s input at the next step. Through this it-
erative feedback loop, the retriever and generator continuously
reshape one another’s inputs, enabling us to capture uncertainty
arising from both components. Experiments on five popular RAR
systems across diverse QA datasets show that R2C improves AU-
ROC by over 5% on average compared to the state-of-the-art UQ
baselines. Extrinsic evaluations using R2C as an external signal
further confirm its effectiveness for two downstream tasks: in Ab-
stention, it achieves ~5% gains in both F1Abstain and AccAbstain;
in Model Selection, it improves the exact match by ~7% over single
models and ~3% over selection methods.
ACM Reference Format:
Heydar Soudani, Hamed Zamani, and Faegheh Hasibi. 2018. Uncertainty
Quantification for Retrieval-Augmented Reasoning. InProceedings of Make
sure to enter the correct conference title from your rights confirmation email
(Conference acronym ’XX).ACM, New York, NY, USA, 14 pages. https://doi.
org/XXXXXXX.XXXXXXX
1 Introduction
Retrieval-augmented generation (RAG) is widely used for know-
ledge-intensive tasks, but remains limited in addressing complex
multi-step reasoning [ 24,44,49,50]. Recent work has explored
combining RAG with reasoning, where LLMs are prompted or
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXtrained to use search engines as tools during their reasoning pro-
cess; a paradigm referred to as retrieval-augmented reasoning
(RAR) [ 19,24,52,54]. However, RAR models are still prone to
producing incorrect responses, due to issues such as retrieving irrel-
evant documents in early steps, misinterpreting retrieved content,
or misusing internal knowledge. Therefore, ensuring the trustwor-
thiness of RAR outputs has become a critical challenge.
Uncertainty quantification (UQ) is a widely studied task in ma-
chine learning, aimed at assessing the reliability of model outputs
by measuring the degree of uncertainty (or lack of confidence) a
model has in its predictions [ 16,18,25,28,34,65]. Recent meth-
ods of estimating the uncertainty of LLM outputs are designed for
settings where the input consists solely of a query, meaning the
LLM itself is the only source of uncertainty [ 3,11,60]. The limited
work on UQ for RAG [ 38,48] incorporates the document–response
relationship into the uncertainty score, but these methods are only
applicable for simple RAG settings, where documents are retrieved
once and inserted into the input prompt for generation. As a result,
existing UQ approaches are suboptimal for RAR.
A fundamental reason for the relatively poor performance of
existing UQ methods for RAR models is that they primarily at-
tribute uncertainty to the LLM’s generative process; i.e., next-token
prediction. In RAR systems, however, we have more sources of
uncertainty:retriever, which may provide irrelevant or partially
relevant retrieved documents and potentially mislead the model’s
reasoning and response generation processes; andgeneration, where
the model’s reasoning may deviate from the user query’s intent and
retrieved documents, leading it to formulate new search queries
that fail to gather informative evidence.
In this paper, we proposeRetrieval-AugmentedReasoningCon-
sistency ( R2C), a novel UQ method that, unlike previous approaches,
accounts for multiple sources of uncertainty in RAR. The central
idea of R2C is to allow the model to explore diverse reasoning paths,
queries, and documents and then measure the consistency of the
resulting final answers. This is achieved by modeling RAR as a
Markov Decision Process (MDP) and perturbing this process in a
controlled way through a set of perturbation actions across various
states. Three perturbation actions are designed to influence query
generation, document retrieval, and LLMs’ thinking process. These
perturbations enable models to arrive at diverse final responses for
uncertain generations. The uncertainty score is then obtained by
measuring consistency of the generated answers using majority
voting; see Figure 1.
We conduct our experiments across multiple datasets and five
RAR models. Our experiments show that R2C significantly outper-
forms existing LLM-specific UQ methods, achieving on average
more than a 5% improvement in AUROC compared to the state-of-
the-art UQ methods. We further extrinsically evaluate R2C on twoarXiv:2510.11483v1  [cs.IR]  13 Oct 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
     What team was Horace Grant's twin brother traded to after the 1992-93 Washington Bullets season? 
Retriever Multi Generations 
         Scoring Function: Majority Voting            Uncertainty Score: 0.66 
Perturbation Actions 
Query 
Paraphrasing Critical 
Thinking Answer 
Validation Portland Trail Blazers 
Evaluation Results 
(Think₁, Query₁) 
Document set ₁
(Think₂, Query₂) 
Document set ₂
(Think₃, Query₃) 
Document set ₃
(Think ₄, Query ₄)
Orlando Magic (Think₁, Query₁) 
Document set ₁
Document set ₂(Think₂, Query₂) 
(Think₃, Query₃) 
Document set ₃
(Think ₄, Response) 
Portland Trail Blazers (Think₁, Query₁) 
Document set ₁
(Think₂, Query₂) 
Document set ₂
(Think₃, Query₃) 
Document set ₃
(Think ₄, Query ₄)
Document set ₄
(Think ₅, Response) 
Seattle SuperSonics (Think ₅, Query ₅)
Document set ₅Path Summary 
(Think ₇, Response) 
Orlando Magic Most-likely Generation 
Agent 
UQ Performance 
(Think ₆, Query ₆)
Document set ₆
Document set ₄
(Think ₅, Response) 
Model Selection 
 Abstention 
Figure 1: R2C overview. Given a user query, the agent (LLM) first generates the most-likely reasoning path leading to the most-
likely response (left, yellow). To estimate uncertainty, R2C creates multiple perturbed generations by randomly altering states
in the reasoning path (middle, gray). The uncertainty score is then derived via majority voting. R2C significantly outperforms
established UQ methods and achieves significant improvements on two downstream tasks: abstention and model selection.
downstream tasks: (i) Abstention:1the task of generating ‘I don’t
know’ when the model is uncertain about its output [ 14,33], and (ii)
Model Selection:2the task of selecting a final answer from a pool of
candidates generated by multiple systems [ 7,15]. Our experimental
results indicate that R2C delivers statistically significant gains over
existing approaches: in Abstention, it achieves roughly 5% improve-
ments on both F1Abstain and AccAbstain; in Model Selection, it
increases exact match by about 7% relative to single RAR models
and by about 3% compared to selection model baselines.
Given the strong performance of R2C in both direct evaluation
and extrinsic evaluation on downstream tasks, we investigate the
factors that contributes to its effectiveness. We show that R2C re-
trieves on average 25 unique documents for each score, compared
to 16 documents retrieved by other UQ methods. It also achieves a
query diversity of0 .35compared to0 .30other methods, measured
by the inverse of the average pairwise cosine similarity between
queries [ 8,66]. This diversity in queries and documents demon-
strates that R2C generates diverse, yet relevant reasoning paths
through our controlled perturbation mechanism. As a result, this
enables the method to achieve uncertainty scores comparable to
baseline approaches while requiring only about 3 generations on
average, 2.5 times fewer token generations than the 10 used by the
baselines. This highlights that R2C is not only the most effective
method of its kind but also a relatively more efficient uncertainty
quantification approach. To summarize, the main contributions of
this paper are:
(1)We propose R2C, a novel theocratically grounded UQ method
based on MDP; the first of its kind that captures different sources
of uncertainty in RAR.
1Also referred to as selective prediction in the literature [12, 59].
2Also referred to as selection-based model ensemble in the literature [20].(2)We conduct extensive experiments on three dataset and five
RAR methods, demonstrating the superiority of the proposed
method on the UQ task with average AUROC of 82%.
(3)We show the effectiveness of our method on both model selec-
tion and abstention tasks, significantly outperforming baselines
by at least 3%.
(4)We demonstrate that R2C achieves an improvement in token
efficiency by approximately 2.5 times.
(5)We show that diverse query and document generation strength-
ens UQ by capturing multiple uncertainty sources.
2 Related Work
Retrieval-Augmented (Reasoning) Models.RAG is a framework
that combines the strengths of retrieval models and generative mod-
els [9]. Broadly, RAG can be implemented in different ways. In the
retrieve-then-generate paradigm, relevant documents are first re-
trieved based on the user’s input and then incorporated into the
model’s prompt [ 35,47]. In contrast, Active RAG allows retrieval to
occur throughout the generation process, either in fixed intervals or
dynamically, whenever additional information is needed [ 23,51,56].
Retrieval-Augmented Reasoning (RAR) is a recent extension of RAG
that integrates retrieval with reasoning, aiming to improve the in-
teraction between LLMs and retrievers [ 52,54]. For example, Self-
Ask [ 39] decomposes complex questions into follow-up queries and
intermediate answers, while ReAct [ 64] defines a set of actions, such
as search, look up, and finish, to structure interactions with external
resources. More recent and effective models such as ReSearch [ 5]
and Search-R1 [ 24] are explicitly trained to seamlessly integrate
external resources into reasoning. Despite their effectiveness on
complex queries, RAR models remain prone to errors, such as re-
trieving irrelevant documents, misinterpreting content, or misusing
internal knowledge.

Uncertainty Quantification for Retrieval-Augmented Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Algorithm 1 R2C: Retrieval-Augmented Reasoning Consistency
Require: user query𝑥, backbone LLM 𝜋𝜃, number of generations
𝐵, set of main actions 𝐴={𝑎𝑟𝑒𝑡,𝑎𝑎𝑛𝑠}, set of perturbation
actions𝐴∗={𝑎𝑞𝑝,𝑎𝑐𝑟,𝑎𝑎𝑣}
Ensure:Uncertainty score𝑈(𝑥,𝑟)
1:for𝑡=1to𝑁do𝑠 𝑡←𝜋𝜃(𝑠𝑡−1,𝑎𝑡−1) ⊲Generate the
most-likely path iteratively
2:𝑟←𝑠𝑁 ⊲Capture the most-likely response
3:𝑅𝐺=∅⊲Initialize multi-generation response set
4:for𝑏=1→𝐵do
5:𝑎∗∼U(A∗)⊲Sample an action fromA∗
6:if𝑎∗=𝑎𝑎𝑣then𝑠𝑡←𝑠𝑁 ⊲Select the last state for𝑎 𝑎𝑣
7:else𝑠 𝑡∼U(𝑠 1,𝑠𝑁−1)⊲Sample a state
8:end if
9:𝑠𝑡+1=𝜋𝜃(𝑠𝑡,𝑎∗)⊲Apply action𝑎∗at state𝑠𝑡
10:for𝑖=1to𝑁𝑏do𝑠𝑡+𝑖+1←𝜋𝜃(𝑠𝑡+𝑖,𝑎𝑡+𝑖)
11:𝑟𝑏←𝑠𝑁𝑏 ⊲Capture the sampled response
12:𝑅𝐺=𝑅𝐺∪{𝑟𝑏}⊲Update the sampled response set
13:end for
14:𝑢=1−1
𝐵Í𝐵
𝑏=1I(𝑟𝑏=𝑟)⊲Compute the uncertainty score
15:return𝑢
Uncertainty Quantification.Broadly, existing UQ methods can
be divided into two categories: white-box approaches, which lever-
age token-level probabilities and entropy [ 3,11,25,28,46,60], and
black-box approaches, which rely only on the final textual out-
puts [ 31,53]. Most UQ methods focus on question answering and
view the LLM as the only source of uncertainty, but in RAG, the re-
triever also contributes its own uncertainty, making this assumption
incomplete. Limited research has explored UQ for RAG by model-
ing the document-response link, either via axioms [ 48] or utility
models [ 38]. Some path-based approaches focus on assessing the
consistency of reasoning paths in reasoning tasks [ 29,40,58]. How-
ever, these methods do not extend naturally to RAR that involves
repeated retrieval during reasoning. Recent work has studied un-
certainty propagation in multi-step decision-making by combining
uncertainties from intermediate steps.SAUP [ 67] learns aggregation
weights to merge per-step uncertainties, but it relies on ground-
truth labels from the test domain. In contrast, we propose R2C, a
method that accumulates uncertainty over the entire reasoning
path, while considering different sources of uncertainty including
the retriever and the generator.
3 Preliminaries
RAR as Markov Decision Process.We formalize RAR as a sto-
chastic Markov Decision Process (MDP), described by a quadruple
(𝑆,𝐴,𝑃,𝑅) , where𝑆denotes a set of states, 𝐴represents a set of
actions the agent can take in a state, 𝑃(𝑠𝑡+1|𝑠𝑡,𝑎𝑡)denotes the
probability of transitioning from state 𝑠𝑡to state𝑠𝑡+1given action
𝑎𝑡, and𝑅(𝑠𝑡,𝑎𝑡)is the reward received by the agent after taking
action𝑎𝑡in state𝑠𝑡. To generate a factual response to a user query
𝑥, the agent𝜋starts from an initial state 𝑠0corresponding to the
query𝑥. The agent then iteratively selects an action at each step
until it chooses a halting action and generates the response. Theagent assigns a probability 𝑝𝜋(𝑎𝑡|𝑠𝑡)to each possible action based
on the current state𝑠 𝑡.
In our unified formulation of MDP for RAR, the LLM 𝜋𝜃acts
as the agent. The environment can take various forms, such as a
knowledge repository [ 64]. The set of possible actions in each state
is𝐴={𝑎 ret,𝑎ans}, where𝑎ansdenotes the halting action, and 𝑎ret
represents the retrieval action. Each intermediate state 𝑠𝑡consists
of a think𝜏𝑡followed by a search query 𝑞𝑡; i.e.,𝑠𝑡=⟨𝜏𝑡,𝑞𝑡⟩. The
final state𝑠𝑁contains a think 𝜏𝑁followed by a final response 𝑟.
The transition probability 𝑃(𝑠𝑡+1|𝑠𝑡,𝑎𝑡)is determined by the LLM
itself. An explicit reward function is not always required, as some
agents operate without additional training.
Consistency-based Uncertainty Quantification.The core idea
of consistency-based methods is to generate multiple responses
for a given query by varying either the input prompt or the tem-
perature parameter used in stochastic decoding [ 27]. The pairwise
similarity among these responses is then computed and aggregated
into a single uncertainty score [ 4,31]. Formally, consider a model
𝜋parameterized with 𝜃, generates the most-likely response 𝑟by
setting the sampling temperature to less than one. Then, 𝐵addi-
tional responses 𝑅𝐺={𝑟𝑏}𝐵
𝑏=1are sampled using various sampling
strategies such as increasing the temperature [ 58], changing the
input [ 27], or altering the reasoning path [ 29,40]. A transformation
function𝜙is then applied to convert 𝑅𝐺into an uncertainty score.
Our method builds on the reasoning path perturbation approach
and employsMajority Voting[ 58] as the transformation function,
where the degree of consistency is measured by the proportion of
sampled responses that match the most-likely response:
𝐶(𝑥,𝑟)=𝜙(𝑅 𝐺,𝑟)=1
𝐵𝐵∑︁
𝑏=1I(𝑟𝑏≡𝑟).(1)
Accordingly, the uncertainty score is defined as:
𝑈(𝑥,𝑟)=1−𝐶(𝑥,𝑟).(2)
4 Methodology
We proposeRetrieval-AugmentedReasoningConsistency, R2C,
to address UQ in RAR models. R2C is a consistency-based approach
that performs UQ in two main stages, as illustrated in Figure 1:
(i) generating the most-likely response, and (ii) sampling multiple
generations. The core idea of R2C is to perturb the reasoning paths
of these multiple generations through a set of perturbation actions,
denoted as𝐴∗. In MDP terms, R2C temporarily replaces the main
action set𝐴with𝐴∗for a single state, allowing the RAR model
to interleave its generation process and explore new reasoning
trajectories, queries, and documents. We define three perturbation
actions employed in R2C: (i) Query Paraphrasing, 𝑎𝑞𝑝, (ii) Critical
Rethinking, 𝑎𝑐𝑟, and (iii) Answer Validation, 𝑎𝑎𝑣. In the following
sections, we first formally describe how R2C perturbs the generation
path and then describe perturbation actions.
4.1 R2C: Retrieval-Augmented Reasoning
Consistency
R2C models UQ for RAR as an MDP, in which multiple response
generations are produced by temporarily replacing the action set 𝐴
with an alternative set 𝐴∗at a randomly selected state 𝑠𝑡. First, the

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
most-likely generation is produced as an MDP iteratively:
𝑠𝑡←𝜋𝜃(𝑠𝑡−1,𝑎𝑡−1);𝑡=1,...,𝑁,
where𝑁denotes the length of the reasoning path, determined by
the agent𝜋𝜃when it selects the halting action 𝑎ans. The most-likely
response𝑟is obtained from the final state 𝑠𝑁.For example, in the
left part of Figure 1, the most-likely response is “Orlando Magic,”
derived from the most-likely generation highlighted in yellow. The
middle part of the figure shows the multi-generation process where
an action is randomly selected and applied to a state.
To construct the sampled response set 𝑅𝐺, we first fix the number
of generations to 𝐵. In each generation, we uniformly sample an
action𝑎∗from the perturbation action set 𝐴∗and a perturbation
state𝑠𝑡from the most-likely reasoning path{𝑠 𝑡}𝑁
𝑡=1. An exception
is the action 𝑎𝑎𝑣, for which the perturbation state is always set to
𝑠𝑁. The agent then transitions from state 𝑠𝑡to𝑠𝑡+1given action 𝑎∗:
𝑠𝑡+1←𝜋𝜃(𝑠𝑡,𝑎∗).
For the remainder of the path, the actions are sampled from the
main action set 𝐴until reaching a new end state 𝑁𝑏, determined
by the agent:
𝑠𝑡+𝑖+1←𝜋𝜃(𝑠𝑡+𝑖,𝑎𝑡+𝑖);𝑖=1,...,𝑁𝑏.
The final sampled response 𝑟𝑏is obtained from the last state 𝑠𝑁𝑏
and added to 𝑅𝐺. After𝐵iterations, we obtain the response set
𝑅𝐺={𝑟𝑏}𝐵
𝑏=1, on which the majority voting function (Eq. (1)) and
the uncertainty scoring function (Eq. (2)) are applied to compute the
final uncertainty score. Algorithm 1 provides a detailed description
of the entire process in R2C.
4.2 Perturbation Actions
A1: Query Paraphrasing (QP).Constructing effective search
queries is critical for retrieving relevant documents; however, LLMs
are not inherently optimized for this purpose [ 22,29,32]. The QP
action (𝑎𝑞𝑝) is introduced as a query optimization mechanism that
enables the system to explore alternative semantic formulations of
the original query. Precisely, when action 𝑎𝑞𝑝is applied, the think
𝜏𝑡of the state𝑠𝑡is preserved and only the query changes. Formally,
the LLM𝜋𝜃takes action 𝑎𝑞𝑝at state𝑠𝑡, transitioning to state 𝑠𝑡+1
with the same think𝜏 𝑡and a new query𝑞 𝑡+1:
𝑠𝑡+1=⟨𝜏𝑡,𝑞𝑡+1⟩←𝜋𝜃(𝑎𝑞𝑝,⟨𝜏𝑡,𝑞𝑡⟩).
Conceptually, QP perturbation tests whether the reasoning path is
so fragile that paraphrasing the search query can alter its direction
and lead to the retrieval of different documents. The QP action is im-
plemented by prompting the LLM with a paraphrasing instruction,
as shown in Figure 7 in Appendix A.
A2: Critical Rethinking (CR).RAR models often suffer from the
problem ofself-criticism, where they fail to recognize that previ-
ously retrieved information is noisy or irrelevant [ 2,22,24,29].
Consequently, they continue to build their reasoning on top of
earlier steps, even when those steps are uninformative and lack
relevant content. This issue becomes particularly severe when it
occurs in the early stages of the reasoning path.
The CR action ( 𝑎𝑐𝑟) critically reassesses the reasoning states
produced up to state 𝑠𝑡. When applied at state 𝑠𝑡, it introduces a
new state𝑠𝑡+1that the think 𝜏𝑡+1explicitly evaluates the previouslyretrieved information as unhelpful, irrelevant, or misleading, and
the accompanying search query 𝑞𝑡+1is formulated to support this
critical assessment. Formally, in state 𝑠𝑡, the LLM𝜋𝜃is prompted
with a critical rethinking instruction to generate a new state𝑠 𝑡+1:
𝑠𝑡+1=⟨𝜏𝑡+1,𝑞𝑡+1⟩←𝜋𝜃(𝑎𝑐𝑟,⟨𝜏𝑡,𝑞𝑡⟩).
Conceptually, if the reasoning path so far has been incorrect, this
action enables the system to adjust toward a more reliable trajectory.
If the path has been correct, CR strengthens its validity, thereby
increasing confidence in the final outcome. Figure 8 in Appendix A
illustrates the instruction prompt used for this action.
A3: Answer Validation (AV).[ 17,21,43]RAR models face difficul-
ties with validating their final response, detecting whether the gen-
erated response meets certain criteria of the query [ 17,21,43,57].
One challenge is that the response is built upon a reasoning path
that integrates both documents and the intermediate reasoning
trajectory, which often leads the LLM to exhibit excessive confi-
dence in its output. Another challenge is that different tasks and
response types involve specific validation criteria, but RAR models
are generally unaware of these requirements.
We introduce the AV action ( 𝑎𝑎𝑣) to validate the final response
by prompting the LLM to reconsider its generation once a re-
sponse has been produced, based on predefined criteria. Specif-
ically, the LLM first generates aquery-aware reasoning path
summary[ 1,30,45], and then evaluates the final response using
two criteria: (i)Groundedness: is the response supported by the
retrieved documents? and (ii)Correctness: does the response ap-
propriately and sufficiently address the query, given the available
evidence? Formally, let 𝐷={𝐷 1,...,𝐷𝑁−1}be the set of docu-
ments retrieved at states [𝑠1,𝑠2,...𝑠𝑁−1]. A modelMgenerates the
summary ˆ𝑆of these documents: ˆ𝑆=M(𝑥,𝐷). The state𝑠𝑁is then
updated with this summary, denoted as ˆ𝑠𝑁. With the updated state
ˆ𝑠𝑁, the LLM𝜋 𝜃is then instructed to generate a new state𝑠 𝑁+1:
𝑠𝑁+1←𝜋𝜃(𝑎𝑎𝑣,ˆ𝑠𝑁).
In principle, if the final response 𝑟is validated as correct, the system
outputs it directly in state 𝑠𝑁+1. Otherwise, if the validation indi-
cates that the answer is incorrect or incomplete, the system begins
a new reasoning path starting from ⟨𝜏𝑁+1,𝑞𝑁+1⟩. The prompt in-
structions for the path summary model and the action are presented
in Figures 9 and 10 in Appendix A, respectively.
5 Experimental Setup
Our experiments consist of evaluation of uncertainty scores esti-
mated by UQ methods as well as extrinsic evaluation on Abstention
and Model Selection. In the following, we review our experimental
setup for each of these tasks.
5.1 Direct Evaluation of UQ Estimations
Datasets: We evaluate R2C on both single-hop and multi-hop
QA tasks using the PopQA [ 35], HotpotQA [ 63], and Musique [ 55]
datasets. Following prior work [ 4,23,36,56,64], we randomly
sampled500queries from each dataset as the test set. We will release
our sampled queries for improving reproducibility of our work.
For the retrieval corpus, we use the 2018 Wikipedia dump [ 26].3
3https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus

Uncertainty Quantification for Retrieval-Augmented Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 1: Performance of UQ methods measured by AUROC. In each column, the best and second-best methods are indicated by
bold and underline , respectively. Superscripts†and‡denote statistically significant differences according to the DeLong test ( 𝑝
< 0.05), compared to ReaC and P(true), respectively, which are the two best-performing methods on average.
RAG SelfAsk [39] ReAct [64] Search-o1 [30] ReSearch [5] Search-R1 [24]Avg.Uncer. M. Popqa Hotp. Musi. Popqa Hotp. Musi. Popqa Hotp. Musi. Popqa Hotp. Musi. Popqa Hotp. Musi.
PE [25] 55.34 59.11 48.61 36.75 39.95 41.14 39.86 53.44 51.59 40.93 61.03 56.69 48.49 48.27 50.90 48.81
SE [28] 64.26 68.01 54.68 49.73 40.37 41.69 67.56 65.84 54.61 53.88 64.72 63.43 59.11 53.45 61.38 57.51
MARS [3] 54.70 59.48 51.53 41.29 40.42 41.97 43.28 57.69 56.80 40.03 61.76 57.58 48.33 49.28 49.94 50.27
SAR [11] 51.65 63.05 52.31 29.56 32.90 31.54 40.92 52.37 44.66 41.67 62.75 51.47 45.87 46.40 45.22 46.16
LARS [60] 73.97 70.03 66.45 79.95 68.28 71.87 83.79 71.42 66.19 76.95 66.12 71.64 71.54 67.08 71.62 71.79
NumSS [28] 71.31 65.19 62.75 74.90 63.73 62.59 78.88 63.74 62.12 80.42 64.46 69.34 69.76 64.10 65.72 67.93
EigV [31] 70.53 66.80 62.13 76.48 66.44 57.44 79.72 68.58 64.75 80.63 66.43 66.31 69.33 66.54 64.27 68.43
ECC [31] 72.89 69.95 64.11 80.27 69.51 61.92 81.98 69.27 69.65 81.85 68.49 72.55 70.87 67.76 67.65 71.25
Deg [31] 70.53 66.79 61.77 76.70 67.70 58.12 80.69 68.87 66.29 81.67 67.15 67.85 69.51 66.75 64.53 68.99
RrrC [29] 71.14 71.1781.2848.02 68.30 75.99 65.95 73.87 77.95 68.30 71.63 74.25 68.92 71.08 70.77 70.57
SelfC [58] 74.33 69.06 68.40 80.73 75.34 72.96 81.26 72.34 76.87 82.01 72.14 77.89 71.63 69.04 70.36 74.29
ReaC [40] 77.02 70.90 76.75 81.53 76.94 74.74 81.09 74.97 77.01 82.50 75.22 77.86 73.22 72.79 75.29 76.52
P(true) [25] 76.51 77.66 78.65 75.07 73.45 71.51 78.57 73.19 74.83 84.35 77.77 81.42 75.73 76.25 74.80 76.65
R2C(our) 80.08 81.09†75.8284.75‡83.25†‡81.16†‡87.09†‡79.66†‡83.22†‡86.02†80.76†82.39 84.92†‡79.51†80.08†‡81.99
The number of retrieved documents is fixed to three across all
models [24, 35, 39, 64].
Evaluation Metrics: To evaluate the quality of outputs, we fol-
low Jin et al . [24] and report the exact match, where a prediction
is counted as correct if and only if it exactly matches one of the
ground-truth responses. For evaluating UQ methods, we follow
prior work on UQ and use the threshold-free metric AUROC, which
captures the correlation between uncertainty scores and response
correctness [ 3,25,28]. As suggested by Perez-Beltrachini and La-
pata [38], significant differences between two AUROC values are
assessed using the paired De Long test [10].
Models: In line with prior work in RAR [ 5,24], we employQwen-2.5-
7B-Instruct[ 62] as the generator LLM and path summary generator
for action𝑎𝑎𝑣. For UQ, we sample10responses per query with
a temperature of 𝑇= 1.0, while for correctness evaluation we
generate the most-likely generation with 𝑇=0.7[3,48]. Retrieval
is performed using a two-stage re-ranking pipeline: BM25 [ 42]
is used for initial retrieval, followed by re-ranking with the pre-
trained cross-encoder model ms-marco-MiniLM-L-6-v2 from the
sentence-transformers library.4All experiments are conducted
on four Nvidia A100 GPU, each with 40 GB memory, requiring
∼1500 GPU hours in total.
Baselines: We use two sets of baselines: (1)Path-basedmethods,
which focus on generating multiple responses based on diverse rea-
soning paths and differ mainly in how they initiate new generations
relative to the most-likely generation. Self-Consistency (SelfC) [ 58]
ignores the most-likely generation and instead produces diverse set
of reasoning paths, starting from scratch. Reasoning Consistency
(ReaC) [ 40] randomly truncates the most-likely reasoning path at
different random steps and regenerates the response based on the
subsequent reasoning steps. Retrieval-Retained Reasoning Consis-
tency (RrrC) [ 29] applies truncation only after the last retrieved
document. In all cases, the generated responses are aggregated into
4https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2a final score using majority voting. 2)Estimation-basedmethods
include both white-box and black-box approaches. The white-box
methods are PE [ 25], SE [ 28], MARS [ 3], LARS [ 60], and SAR [ 11].
The black-box methods are NumSS [ 28], EigV, ECC, Deg [ 31], and
P(true) [ 25]. All of these methods rely on generations obtained in
the same way as SelfC. All UQ methods are implemented using
TruthTorchLM [61].
5.2 Extrinsic UQ Evaluation via Abstention
Task Formulation.Declining to respond due to uncertainty is
an important application of UQ [ 14]. An abstention function de-
termines whether the model should withhold an answer. In our
setup, this decision is guided by uncertainty scores. We introduce a
threshold𝜏abs: if the uncertainty exceeds 𝜏abs, the model abstains;
otherwise, it produces an answer. Formally, given a response 𝑟to a
query𝑥and an uncertainty function 𝑈, we define the abstention
function𝑓 absas:
𝑓abs(𝑟)=(
true,if𝑈(𝑥,𝑟)>𝜏 abs
false,otherwise.
Baselines & Evaluation Metrics.As baselines, we implement the
abstention task using uncertainty scores derived from different
UQ methods, including P(true), RrrC, ReaC, SelfC, and R2C. For
evaluation, we follow Feng et al . [14] and report results on Ab-
stainAccuracy and AbstainF1.AbstainAccuracyevaluates whether
the abstain decisions are correct, andAbstainF1captures the balance
between reliability and answer coverage. The evaluation metrics
are explained in detail in Appendix B.1. The threshold 𝜏absfor each
UQ method is selected by performing sweep using validation sets;
see Appendix B.2 for more details.
5.3 Extrinsic UQ Evaluation via Model Selection
Task Formulation.Model selection (or selection-based model
ensemble [ 7,20]) aims to select a final response for a question based
on multiple candidate responses generated by different systems [ 7].

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
Table 2: Abstention performance measured byAbstainAccuracyandAbstainF1at a 0.9 confidence threshold. For each column,
the best and second-best methods are indicated in bold and underlined, respectively. A superscript†denotes a statistically
significant difference compared to ReaC based on the McNemar test for Accuracy and the Bootstrap test for F1 (𝑝<0.05).
RAG SelfAsk [39] ReAct [64] Search-o1 [30] ReSearch [5] Search-R1 [24]Avg.Uncer. M. Popqa Hotpot Musiq. Popqa Hotpot Musiq. Popqa Hotpot Musiq. Popqa Hotpot Musiq. Popqa Hotpot Musiq.
Abstain Accuracy
RrrC [29] 61.4 65.6 73.0 60.8 72.2 82.4 56.6 66.2 68.2 59.2 64.4 64.0 60.6 64.0 63.6 65.48
P(true) [25] 70.6 70.8 80.4 70.4 67.2 75.0 72.6 65.8 75.0 78.8 71.4 78.2 70.8 70.0 74.4 72.76
SelfC [58] 68.6 64.2 76.2 74.0 75.8 87.6 75.0 69.090.275.2 68.6 83.6 65.4 62.8 77.8 74.27
ReaC [40] 69.2 64.6 80.4 74.4 75.4 89.4 75.6 71.4 87.4 77.2 68.6 80.8 67.8 68.8 80.6 75.44
R2C (our) 77.2†74.4†88.6†77.0 76.8 90.4 80.4†74.489.4 81.6†74.8†84.0 77.0†73.4 84.4†80.25
Abstain F1
RrrC [29] 62.52 70.54 82.75 71.59 80.82 89.69 56.33 71.50 78.99 54.86 63.37 73.68 55.12 61.20 73.92 69.79
P(true) [25] 73.60 75.33 88.27 72.07 73.46 84.58 76.50 71.73 84.51 81.20 73.46 85.82 71.03 71.15 83.37 77.74
SelfC [58] 72.60 69.51 85.71 79.10 83.80 93.18 80.50 78.0794.6481.71 75.19 90.5762.31 62.34 86.51 79.72
ReaC [40] 73.54 70.45 88.41 79.01 83.17 94.19 81.24 79.48 93.03 81.25 72.60 88.43 67.60 71.11 88.27 80.79
R2C (our) 82.35†81.76†93.77†83.69†85.08†94.81 86.42†82.22†94.27†84.40†78.71†90.49†80.80†77.57†90.97†85.82
Formally, given a user query 𝑥and a set of systems {𝑆1,𝑆2,...,𝑆𝑀},
each system 𝑆𝑖produces a response candidate 𝑟𝑖. A model selection
methodM(𝑥,𝑅) then selects the final response ˆ𝑟from all candidate
responses𝑅={𝑟 1,𝑟2,...,𝑟𝑀}.
R2C Select.The proposed R2C Select utilizes the uncertainty scores
derived from R2C. Our method groups semantically similar re-
sponses into 𝐾clusters. Following [ 29,61], we useQwen-2.5-7B-
Instructto compute pairwise semantic similarities between candi-
date responses and cluster similar ones together. We then assign
an uncertainty score to each cluster𝑐 𝑖:
𝑅𝐶={⟨𝑟1,𝑢𝑐1⟩,⟨𝑟2,𝑢𝑐1⟩,...,⟨𝑟𝐾,𝑢𝑐𝐾⟩}.
The uncertainty score of each cluster 𝑐={𝑟𝑖}𝑚
𝑖=1is computed by ag-
gregating the uncertainty scores of its members: 𝑢𝑐=Í
𝑟𝑖∈𝑐𝑈(𝑥,𝑟𝑖),
where the𝑈function provides the uncertainty score for the re-
sponse𝑟𝑖. The final response ˆ𝑟is the response with the lowest
uncertainty score: ˆ𝑟=arg min 𝑗𝑢𝑐𝑗.If no clustering is applied, each
response constitutes a cluster and 𝑀=𝐾 ; we refer to this variation
asR2C Select w/o clustering.
Baselines & Evaluation Metrics.We evaluate R2C Select per-
formance against both single RAG and RAR systems, as well as
existing selection-based ensemble approaches. As baselines, we
consider LLMBlender [ 20] and RAGEnsemble [ 6]. LLMBlender is a
trained reward model that given a user query and a set of candidate
responses, ranks the responses accordingly. RAGEnsemble is an
instruction-based approach that selects a single final answer from
a set of candidate responses. For all selection methods, including
the baselines and R2C Select, the response candidates are obtained
from SelfAsk [ 39], ReAct [ 64], Search-o1 [ 30], ReSearch [ 5], and
Search-R1 [ 24]. For evaluation, we report the correctness of the
final answer using exact match.
6 Results
We present a set experiments that address the following research
questions:RQ1: How does R2C perform in quantifying uncertainty
for different RAR models? (Sec. 6.1),RQ2: How does R2C performas an external signal on downstream tasks, such as Abstention
and Model Selection? (Sec. 6.2),RQ3: What factors contribute to
the effectiveness of R2C? (Sec. 6.3),RQ4: How does R2C balance
effectiveness and efficiency? (Sec. 6.4),RQ5: What is the effect of
different actions in the performance of R2C? (Sec. 6.5).
6.1 Uncertainty Quantification Performance
RQ1evaluates the performance of R2C compared to other UQ
methods. Table 1 presents results on five RAR systems across three
datasets. The findings indicate that most white-box methods, i.e.,
PE [25], SE [ 28], MARS [ 3], and SAR [ 11], perform relatively poorly,
with AUROC values ranging from about 30 to 60. This weakness
stems from their overreliance on token probabilities. In contrast,
black-box methods, such as NumSS [ 28], EigV, ECC, and Deg [ 31],
generally outperform white-box methods, reaching average AU-
ROC values between roughly 60 and 80. Interestingly, P(true) ranks
as the second-best method in terms of average AUROC across all
approaches, highlighting that black-box methods become much
more effective when using an LLM as the scoring function. This
advantage largely comes from their stronger reliance on textual
diversity. The supervised method LARS, optimized for QA with
no reasoning, achieves AUROC scores between 65 and 85 across
all cases, surpassing both white-box and black-box approaches on
average. This finding highlights the potential of as a promising
direction for future research.
Path-based methods are another group of approaches that we
evaluate, including the proposed R2C. RrrC achieves an average
AUROC of 70.57, which is the lowest among these methods. This
indicates that simply keeping the documents in the reasoning path
and regenerating the last state is not effective. SelfC and ReaC per-
form relatively on par, and ReaC outperforms SelfC on average,
showing that regenerating from the top of the reasoning path does
not necessarily guarantee a better uncertainty score. Finally, R2C
outperforms all methods by a large margin, with an absolute im-
provement of 5% on average. These findings suggest that applying
actions to the reasoning path enables exploration of a wider variety
of possible reasoning states, leading to more reliable scores.

Uncertainty Quantification for Retrieval-Augmented Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 3: Model Selection performance measured by exact
match. The superscript†denotes a statistically significant
difference from the best-performing baseline (underlined),
according to the Wilcoxon test (𝑝< 0.05).
RAG System PopQA HotpotQA Musique Average
Vanilla LLM & RAG
Direct 18.8 20.8 2.6 14.1
CoT 17.6 22.2 5.8 15.2
Vanilla RAG 30.2 18.6 4.4 17.7
IRCoT [56] 34.6 27.2 5.4 22.4
FLARE [23] 31.6 25.2 8.4 21.7
DRAGIN [51] 28.6 23.2 4.6 18.8
Retrieval-Augmented Reasoning (RAR)
SelfAsk [39] 35.6 33.0 10.4 26.3
ReAct [64] 36.8 27.8 10.8 25.1
Search-o1 [30] 33.2 29.0 10.0 24.1
ReSearch [5] 38.6 38.8 16.6 31.3
Search-R1 [24] 41.6 41.4 16.0 33.0
Model Selection RAR
Random 31.4 31.6 10.6 24.5
LLMBlender [20] 34.4 36.0 12.0 27.5
-w/o clustering35.6 31.8 9.6 25.7
-w/o clus. & unc.32.0 26.0 8.6 22.2
RAGEnsemble [6] 45.6 46.0 19.4 37.0
-w/o clustering44.4 40.8 18.0 34.4
-w/o clus. & unc.43.2 37.2 13.0 31.1
R2CSelect (our) 46.8†50.2†22.6†39.9
-w/o clustering45.4 44.0 19.2 36.2
Ideal Model Selection 55.0 57.2 30.0 47.4
6.2 Extrinsic Evaluation Results
RQ2evaluates the performance of R2C in Abstention and Model
Selection. For Abstention, Table 2 shows the performance on RAR
models and datasets. Abstain Accuracy measures whether both ab-
stentions and non-abstentions are detected correctly, while captures
the effectiveness of detecting abstain samples. Considering Abstain
Accuracy, R2C outperforms the baselines in all cases except on the
Musique dataset and performs on par with the second-best baseline
in ReAct, Search-o1, and ReSearch.Considering Abstain F1, R2C
significantly outperforms the best baseline in all setup, expect for
ReAct on Musique. This indicates that R2C scores are more reliable
in detecting than correct abstentions than correct non-abstentions.
Overall, compared to other UQ methods, R2C achieves significantly
better performance in most cases—on average, about 5% higher
than the second-best model. These results demonstrate that the
uncertainty scores generated by R2C are reliable enough for the
system to decide when to refrain from answering.
For model selection, Table 3 reports the results of individual
RAG and RAR systems, as well as selection-based ensemble models.
Among individual systems, RAR models outperform RAG models
for both simpler single-hop questions (PopQA) and more complex
multi-hop queries (HotpotQA and Musique). Within selection-based
ensemble models, LLMBLender performs poorly, even worse than
fine-tuned single RAR models such as ReSearch and Search-R1. In
contrast, RAGEnsemble surpasses both LLMBLender and all single
RAR models, highlighting the advantage of an instruction-tuned
model for selecting the final response. Finally, R2C Selectachieves
020406080PopQA
SelfAsk
 ReAct
 Search-o1
 ReSearch
 Search-R1
020406080HotpotQA
MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C020406080Musique
MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C
 MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C
 MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C
 MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C MainRrrCReaCSlfCR2C
Number of Unique DocsFigure 2: Distribution of the number of unique retrieved docu-
ments for the most-likely path (main) and multi-generations.
0.00.20.40.60.81.0PopQA
SelfAsk
 ReAct
 Search-o1
 ReSearch
 Search-R1
0.00.20.40.60.81.0HotpotQA
RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C0.00.20.40.60.81.0Musique
RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C
 RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C
 RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C RrrCReaCSlfCR2C
Diversity of Search Queries
Figure 3: Distribution of diversity scores for search queries
generated for each user query across reasoning paths.
the best performance, significantly outperforming all RAR and
RAG systems, with 3.7% average improvement on HotpotQA and
Musique datasets.These results confirm that R2C score is a
reliable criterion, not only within a single system but also
across different systems.Interestingly, using R2C uncertainty
scores alone, without clustering, improves selection-based ensemble
models in comparable settings, highlighting informativeness of R2C
scores even for models that are not trained on uncertainty.
6.3 Explaining the Strength of R2C
RQ3examines why R2C outperforms other UQ methods. To this
end, we analyze the diversity of reasoning paths as an indicator of
how effectively R2C captures the uncertainty arising from both the
generator and the retriever, i.e., the two key sources of uncertainty
in our framework. We report the number of unique documents
retrieved during the multi-generation step, as shown in Figure 2.
On average, R2C retrieves 24.71 unique documents for a single
uncertainty score, whereas RrrC, SelfC, and ReaC retrieve 5.81,
15.35, and 16.4 documents, respectively.
We further report on the diversity of generated search queries,
as a proxy for LLM’s thinking process. Query diversity is measured

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
50607080PopQA
SelfAsk
 ReAct
 Search-o1
 ReSearch
 Search-R1
607080HotpotQA
23 5 10607080Musique
23 5 10
23 5 10
23 5 10
23 5 10
Num of GenerationsAUROCRrrC ReaC SlfC R2C
Figure 4: Performance of UQ methods with varying numbers
of generations.
50607080PopQA
SelfAsk
 ReAct
 Search-o1
 ReSearch
 Search-R1
607080HotpotQA
0 500 1000 1500607080Musique
0 1000 2000 3000
 0 1000 2000 3000
 0 1000 2000 3000
 0 1000 2000
Num of Generated TokensAUROCRrrC ReaC SlfC R2C
Figure 5: Performance of UQ methods with varying number
of generated tokens, illustrating the trade-off between effec-
tiveness and efficiency.
based on the average pair-wise similarity of generated queries [ 8,
66]. Formally, given 𝑛search queries{𝑞1,...,𝑞𝑛}, query diversity
is computed as:
Query Diversity=1−2
𝑛(𝑛−1)𝑛∑︁
𝑖=1𝑛∑︁
𝑖<𝑗𝑐𝑜𝑠(𝑒𝑖,𝑒𝑗),
where𝑐𝑜𝑠(.) represents cosine similarity function, and 𝑒𝑖denotes
the normalized embedding of query 𝑞𝑖(i.e.,|𝑒𝑖|=1) obtained from
sentence-transformers/all-MiniLM-L6-v2 . Figure 3 presents
the results. On average, R2C achieves a diversity score of0 .35,
while RrrC, SelfC, and ReaC achieve0 .20,0.28, and0.30, respec-
tively. These results indicate that R2C generates more diverse search
queries when estimating uncertainty. These findings indicate R2C
effectively captures the uncertainty of both the retriever and the
generator by sequentially diversifying their inputs.
6.4 Effectiveness vs. Efficiency
RQ4explores the trade-off between effectiveness and efficiency of
R2C. Figure 4 presents the relationship between AUROC and the
number of response generations across various datasets and RAR
models. The results demonstrate that R2C consistently outperforms
other methods, even with a smaller number of generations. On av-
erage, across all datasets and RAR models, R2C achieves an AUROC
performance of about 77% with only three generations, which is
comparable to the performance of SelfC and ReaC, requiring 10
generations to reach a similar level.
SelfAsk Search-R17074788286PopQA
SelfAsk Search-R1HotpotQAAUROCQuery Paraphrasing Critical Thinking Answer Validation AllFigure 6: Performance of R2C with different action sets.
For a deeper exploration of the trade-offs between effective-
ness and efficiency, we measure efficiency using the total number
of generated tokens, following prior work [ 38]. Figure 5 reports
AUROC performance for equal numbers of generated tokens. It
shows R2C achieves higher AUROC scores than other methods
given the same number of generated tokens for multi-hop datasets.
For the single-hop dataset, PopQA, R2C performs comparably to
SelfAsk, ReAct, and ReSearch, but surpasses them on Search-o1 and
Search-R1. Moreover, R2C produces on average around 700 tokens
with three generations, reaching the same AUROC score of base-
lines with approximately 1,700 tokens with 10 generations. These
findings demonstrate that R2C improves efficiency by roughly 2.5
times. Overall, these results indicate that R2C outperforms other
UQ methods in terms of both effectiveness and token generation
efficiency.
6.5 Action Selection
RQ5investigates the roll of action set in R2C. Figure 6 illustrates
the performance of R2C with different action sets. In the first three
bars, only a single action is used, meaning that in each generation,
the cut point in the reasoning path is chosen randomly, while the
action itself remains fixed. The fourth bar represents our main setup,
where both the action and the cut point are selected randomly.
For Search-R1 and PopQA, the main setup outperforms all other
configurations, whereas in HotpotQA, QP performs slightly better.
For SelfAsk, the AV action achieves the best performance, with the
main setup ranking second. These results indicate that while our
main setup is generally robust, there are still potentials to design
action configurations better suited to specific RAR systems.
7 Conclusions and Future Work
This paper introduces a novel and theoretically grounded UQ
method for retrieval-augmented reasoning (RAR) systems, called
Retrieval-Augmented Reasoning Consistency ( R2C). We argue that
an effective UQ method should account for different sources of un-
certainty and accurately reflect them in its final score. R2C models
uncertainty stemming from both the retriever and the generator by
perturbing the reasoning process through a series of actions, includ-
ing query paraphrasing, critical rethinking, and answer validation.
Comprehensive experiments conducted on three datasets and five
RAR models demonstrate that R2C improves AUROC by more than
5% on average compared to state-of-the-art UQ baselines. More-
over, when used as an external signal in two downstream tasks,
R2C consistently proves effective: in Abstention, it yields around
5% gains in both F1Abstain and AccAbstain; in Model Selection, it
increases exact match by approximately 7% over individual models

Uncertainty Quantification for Retrieval-Augmented Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
and about 3% over selection methods. While this paper focuses
on UQ for RAR models, the underlying concept of modeling and
stimulating multiple sources of uncertainty is broadly applicable.
Future work can extend this approach to other domains involving
multiple sources of uncertainty, such as vision-language models.
Moreover, in this work, we focus on short-form QA, where the final
answer is an entity. Future research can explore UQ for long-form
generation, which represents a more realistic scenario.
Acknowledgments
This publication is part of the project LESSEN with project num-
ber NWA.1389.20.183 of the research program NWA ORC 2020/21
which is (partly) financed by the Dutch Research Council (NWO).
References
[1]Pierre Achkar, Tim Gollub, and Martin Potthast. 2025. Ask, Retrieve, Summarize:
A Modular Pipeline for Scientific Literature Summarization.CoRR(2025).
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi.
2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-
Reflection. InThe Twelfth International Conference on Learning Representations,
ICLR. OpenReview.net.
[3]Yavuz Faruk Bakman, Duygu Nur Yaldiz, Baturalp Buyukates, Chenyang Tao,
Dimitrios Dimitriadis, and Salman Avestimehr. 2024. MARS: Meaning-Aware
Response Scoring for Uncertainty Estimation in Generative LLMs. InProceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics ACL.
7752–7767.
[4]Yavuz Faruk Bakman, Duygu Nur Yaldiz, Sungmin Kang, Tuo Zhang, Baturalp
Buyukates, Salman Avestimehr, and Sai Praneeth Karimireddy. 2025. Reconsider-
ing LLM Uncertainty Estimation Methods in the Wild. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). 29531–29556.
[5]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen
Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng
Chen. 2025. ReSearch: Learning to Reason with Search for LLMs via Reinforce-
ment Learning.CoRR(2025).
[6]Yifei Chen, Guanting Dong, Yutao Zhu, and Zhicheng Dou. 2025. Revisiting
RAG Ensemble: A Theoretical and Mechanistic Analysis of Multi-RAG System
Collaboration.arXiv preprint arXiv:2508.13828(2025).
[7]Zhijun Chen, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun, Yuankai Luo,
Qianren Mao, Dingqi Yang, Hailong Sun, and Philip S. Yu. 2025. Harnessing Mul-
tiple Large Language Models: A Survey on LLM Ensemble.CoRRabs/2502.18036
(2025). https://doi.org/10.48550/ARXIV.2502.18036 arXiv:2502.18036
[8]Samuel Rhys Cox, Yunlong Wang, Ashraf Abdul, Christian Von Der Weth, and
Brian Y. Lim. 2021. Directed diversity: Leveraging language embedding dis-
tances for collective creativity in crowd ideation. InProceedings of the 2021 CHI
Conference on Human Factors in Computing Systems. 1–35.
[9]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare
Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. 2024. The
Power of Noise: Redefining Retrieval for RAG Systems. InProceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval, SIGIR. 719–729.
[10] Elizabeth R DeLong, David M DeLong, and Daniel L Clarke-Pearson. 1988. Com-
paring the areas under two or more correlated receiver operating characteristic
curves: a nonparametric approach.Biometrics(1988), 837–845.
[11] Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu,
Bhavya Kailkhura, and Kaidi Xu. 2024. Shifting Attention to Relevance: Towards
the Predictive Uncertainty Quantification of Free-Form Large Language Models.
InProceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 5050–5063.
[12] Jinhao Duan, James Diffenderfer, Sandeep Madireddy, Tianlong Chen, Bhavya
Kailkhura, and Kaidi Xu. 2025. UProp: Investigating the Uncertainty Propagation
of LLMs in Multi-Step Agentic Decision-Making.CoRRabs/2506.17419 (2025).
[13] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting
hallucinations in large language models using semantic entropy.Nat.630, 8017
(2024), 625–630.
[14] Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran,
and Yulia Tsvetkov. 2024. Don’t Hallucinate, Abstain: Identifying LLM Knowledge
Gaps via Multi-LLM Collaboration. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), ACL.
Association for Computational Linguistics, 14664–14690.
[15] Neel Guha, Mayee F. Chen, Trevor Chow, Ishan S. Khare, and Christopher Ré.
2024. Smoothie: Label Free Language Model Routing. InAdvances in NeuralInformation Processing Systems 38: Annual Conference on Neural Information
Processing Systems 2024, NeurIPS.
[16] James Harrison, John Willes, and Jasper Snoek. 2024. Variational Bayesian Last
Layers. InThe Twelfth International Conference on Learning Representations, ICLR.
OpenReview.net.
[17] Bolei He, Nuo Chen, Xinran He, Lingyong Yan, Zhenkai Wei, Jinchang Luo,
and Zhen-Hua Ling. 2024. Retrieving, Rethinking and Revising: The Chain-of-
Verification Can Improve Retrieval Augmented Generation. InFindings of the
Association for Computational Linguistics: EMNLP. Association for Computational
Linguistics, 10371–10393.
[18] Bairu Hou, Yujian Liu, Kaizhi Qian, Jacob Andreas, Shiyu Chang, and Yang
Zhang. 2024. Decomposing Uncertainty for Large Language Models through
Input Clarification Ensembling. InForty-first International Conference on Machine
Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. https:
//openreview.net/forum?id=byxXa99PtF
[19] Shayekh Bin Islam, Md. Asib Rahman, K. S. M. Tozammel Hossain, Enamul Hoque,
Shafiq Joty, and Md. Rizwan Parvez. 2024. Open-RAG: Enhanced Retrieval Aug-
mented Reasoning with Open-Source Large Language Models. InFindings of the
Association for Computational Linguistics: EMNLP. Association for Computational
Linguistics, 14231–14244.
[20] Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. 2023. LLM-Blender: Ensem-
bling Large Language Models with Pairwise Ranking and Generative Fusion. In
Proceedings of the 61st Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), ACL. Association for Computational Linguistics,
14165–14178.
[21] Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie Wang, Xin Zhao, Yang
Song, and Tao Zhang. 2025. RAG-Star: Enhancing Deliberative Reasoning with
Retrieval Augmented Verification and Refinement. InProceedings of the 2025
Conference of the Nations of the Americas Chapter of the Association for Com-
putational Linguistics: Human Language Technologies, NAACL. Association for
Computational Linguistics, 7064–7074.
[22] Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng
Sun, and Jiawei Han. 2025. s3: You Don’t Need That Much Data to Train a Search
Agent via RL.arXiv preprint arXiv:2505.14146(2025).
[23] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval Aug-
mented Generation. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing (EMNLP). 7969–7992.
[24] Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-R1: Training LLMs to Reason and Leverage Search Engines
with Reinforcement Learning.CoRR(2025).
[25] Saurav Kadavath et al .2022. Language Models (Mostly) Know What They Know.
abs/2207.05221 (2022).
[26] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing, EMNLP. Association for Computational
Linguistics, 6769–6781.
[27] Anton Korikov, Pan Du, Scott Sanner, and Navid Rekabsaz. 2025. Batched
Self-Consistency Improves LLM Relevance Assessment and Ranking.CoRR
abs/2505.12570 (2025).
[28] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty: Lin-
guistic Invariances for Uncertainty Estimation in Natural Language Generation.
InThe Eleventh International Conference on Learning Representations ICLR.
[29] Baixuan Li, Yunlong Fan, Tianyi Ma, Miao Gao, Chuanqi Shi, and Zhiqiang
Gao. 2025. RASPberry: Retrieval-Augmented Monte Carlo Tree Self-Play with
Reasoning Consistency for Multi-Hop Question Answering. InFindings of the
Association for Computational Linguistics: ACL 2025. 11258–11276.
[30] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models.CoRR(2025).
[31] Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2024. Generating with Confidence:
Uncertainty Quantification for Black-box Large Language Models.Trans. Mach.
Learn. Res.2024 (2024).
[32] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query
Rewriting in Retrieval-Augmented Large Language Models. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing. 5303–5315.
[33] Nishanth Madhusudhan, Sathwik Tejaswi Madhusudhan, Vikas Yadav, and Ma-
soud Hashemi. 2025. Do LLMs Know When to NOT Answer? Investigating
Abstention Abilities of Large Language Models. InProceedings of the 31st In-
ternational Conference on Computational Linguistics, COLING. Association for
Computational Linguistics, 9329–9345.
[34] Andrey Malinin and Mark J. F. Gales. 2021. Uncertainty Estimation in Au-
toregressive Structured Prediction. In9th International Conference on Learning
Representations, ICLR.
[35] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InProceedings of the

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
61st Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 9802–9822.
[36] Viktor Moskvoretskii, Maria Marina, Mikhail Salnikov, Nikolay Ivanov, Sergey
Pletenev, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Irina Nikishina,
and Alexander Panchenko. 2025. Adaptive Retrieval Without Self-Knowledge?
Bringing Uncertainty Back Home. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), ACL.
Association for Computational Linguistics, 6355–6384.
[37] Malik Sajjad Ahmed Nadeem, Jean-Daniel Zucker, and Blaise Hanczar. 2010.
Accuracy-Rejection Curves (ARCs) for Comparing Classification Methods with
a Reject Option. InProceedings of the third International Workshop on Machine
Learning in Systems Biology, MLSB (JMLR Proceedings, Vol. 8). JMLR.org, 65–81.
[38] Laura Perez-Beltrachini and Mirella Lapata. 2025. Uncertainty Quantification
in Retrieval Augmented Question Answering.arXiv preprint arXiv:2502.18108
(2025).
[39] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. InFindings of the Association for Computational Linguistics: EMNLP 2023.
5687–5711.
[40] Zhenting Qi, Mingyuan MA, Jiahang Xu, Li Lyna Zhang, Fan Yang, and Mao
Yang. 2025. Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solver. In
The Thirteenth International Conference on Learning Representations.
[41] Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, and Balaji Lakshminarayanan. 2023. Self-
Evaluation Improves Selective Generation in Large Language Models. InPro-
ceedings on "I Can’t Believe It’s Not Better: Failure Modes in the Age of Foundation
Models" at NeurIPS 2023 Workshops (Proceedings of Machine Learning Research,
Vol. 239). PMLR, 49–64.
[42] Stephen E. Robertson and Steve Walker. 1994. Some Simple Effective Approxima-
tions to the 2-Poisson Model for Probabilistic Weighted Retrieval. InProceedings of
the 17th Annual International ACM-SIGIR Conference on Research and Development
in Information Retrieval. 232–241.
[43] Alireza Salemi, Mukta Maddipatla, and Hamed Zamani. 2025. CIIR@LiveRAG
2025: Optimizing Multi-Agent Retrieval Augmented Generation through Self-
Training.CoRRabs/2506.10844 (2025). https://doi.org/10.48550/ARXIV.2506.
10844 arXiv:2506.10844
[44] Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in
Retrieval-Augmented Generation. InProceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval, SIGIR.
ACM, 2395–2400.
[45] Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An
Zhang, and Xiang Wang. 2025. Search and Refine During Think: Autonomous
Retrieval-Augmented Reasoning of LLMs.arXiv preprint arXiv:2505.11277(2025).
[46] Heydar Soudani. 2025. Enhancing Knowledge Injection in Large Language Models
for Efficient and Trustworthy Responses. InProceedings of the 48th International
ACM Conference on Research and Development in Information Retrieval, SIGIR.
4211.
[47] Heydar Soudani, Evangelos Kanoulas, and Faegheh Hasibi. 2024. Fine Tuning vs.
Retrieval Augmented Generation for Less Popular Knowledge. InProceedings of
the 2024 Annual International ACM SIGIR Conference on Research and Development
in Information Retrieval in the Asia Pacific Region, SIGIR-AP 2024. 12–22.
[48] Heydar Soudani, Evangelos Kanoulas, and Faegheh Hasibi. 2025. Why Uncer-
tainty Estimation Methods Fall Short in RAG: An Axiomatic Analysis. InFindings
of the Association for Computational Linguistics: ACL 2025. 16596–16616.
[49] Heydar Soudani, Roxana Petcu, Evangelos Kanoulas, and Faegheh Hasibi. 2024.
Data Augmentation for Conversational AI. InCompanion Proceedings of the ACM
on Web Conference 2024, WWW. 1234–1237.
[50] Heydar Soudani, Roxana Petcu, Evangelos Kanoulas, and Faegheh Hasibi. 2024.
A Survey on Recent Advances in Conversational Data Generation.CoRR
abs/2405.13003 (2024).
[51] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
Dynamic Retrieval Augmented Generation based on the Real-time Information
Needs of Large Language Models. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 12991–13013.
[52] Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao
Zhang, Yang Song, and Han Li. 2025. ReARTeR: Retrieval-Augmented Reasoning
with Trustworthy Process Rewarding. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval,
SIGIR. ACM, 1251–1261.
[53] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu
Yao, Chelsea Finn, and Christopher D. Manning. 2023. Just Ask for Calibration:
Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-
Tuned with Human Feedback. InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP. Association for Computational
Linguistics, 5433–5442.
[54] Hieu Tran, Zonghai Yao, Zhichao Yang, Junda Wang, Yifan Zhang, Shuo Han,
Feiyun Ouyang, and Hong Yu. 2025. RARE: Retrieval-Augmented Reasoning
Enhancement for Large Language Models. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),ACL. Association for Computational Linguistics, 18305–18330.
[55] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554.
[56] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 10014–10037.
[57] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo
Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large Lan-
guage Models are not Fair Evaluators. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), ACL.
Association for Computational Linguistics, 9440–9450.
[58] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves
Chain of Thought Reasoning in Language Models. InThe Eleventh International
Conference on Learning Representations, ICLR.
[59] Ji Xin, Raphael Tang, Yaoliang Yu, and Jimmy Lin. 2021. The Art of Abstention:
Selective Prediction and Error Regularization for Natural Language Processing.
InProceedings of the 59th Annual Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference on Natural Language Processing,
ACL/IJCNLP. Association for Computational Linguistics, 1040–1051.
[60] Duygu Nur Yaldiz, Yavuz Faruk Bakman, Baturalp Buyukates, Chenyang Tao,
Anil Ramakrishna, Dimitrios Dimitriadis, Jieyu Zhao, and Salman Avestimehr.
2025. Do Not Design, Learn: A Trainable Scoring Function for Uncertainty
Estimation in Generative LLMs. InFindings of the Association for Computational
Linguistics: NAACL 2025.
[61] Duygu Nur Yaldiz, Yavuz Faruk Bakman, Sungmin Kang, Alperen Öziş, Hayret-
tin Eren Yildiz, Mitash Ashish Shah, Zhiqi Huang, Anoop Kumar, Alfy Samuel,
Daben Liu, et al .2025. TruthTorchLM: A Comprehensive Library for Predicting
Truthfulness in LLM Outputs.arXiv preprint arXiv:2507.08203(2025).
[62] An Yang et al. 2024. Qwen2.5 Technical Report.CoRRabs/2412.15115 (2024).
[63] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Rus-
lan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering. InProceedings of the
2018 Conference on Empirical Methods in Natural Language Processing, EMNLP.
Association for Computational Linguistics, 2369–2380.
[64] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InThe Eleventh International Conference on Learning Representations,
ICLR.
[65] Fanghua Ye, Mingming Yang, Jianhui Pang, Longyue Wang, Derek F. Wong,
Emine Yilmaz, Shuming Shi, and Zhaopeng Tu. 2024. Benchmarking LLMs via
Uncertainty Quantification. InAdvances in Neural Information Processing Systems
38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS.
[66] Tianhui Zhang, Bei Peng, and Danushka Bollegala. 2025. Evaluating the Evalua-
tion of Diversity in Commonsense Generation. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
24258–24275.
[67] Qiwei Zhao, Dong Li, Yanchi Liu, Wei Cheng, Yiyou Sun, Mika Oishi, Takao
Osaki, Katsushi Matsuda, Huaxiu Yao, Chen Zhao, Haifeng Chen, and Xujiang
Zhao. 2025. Uncertainty Propagation on LLM Agent. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025. 6064–6073.

Uncertainty Quantification for Retrieval-Augmented Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Prompt Details
An important component of the R2C method is the Perturbation
Actions discussed in Section 4.2. These actions are implemented by
prompting the LLM with specific instructions. Figures 7, 8, and 9
show the prompts used for Query Paraphrasing, Critical Rethink-
ing, and Answer Validation, respectively. Moreover, the Answer
Validation action includes a Reasoning Path Summarization compo-
nent, which is also implemented through prompting an LLM. The
prompt for this component is shown in Figure 10. Finally, Figure 11
presents the semantic equivalence prompt, which is used to assess
the equality of two responses for majority voting and clustering
in the Model Selection process described in Section 5.3, following
pervious work [3, 29, 60].
You are an expert in information retrieval. 
Given an original search query, generate {n} semantically diverse and effective paraphrased 
search queries that capture the same intent but use different wording or structure. 
These paraphrased queries should be suitable for improving search engine results by covering 
various phrasings a user might employ. 
Do not add extra information in the new queries. 
Here are some examples: 
<original_query> 
popular industry in the neighborhood of Willow Vale, New South Wales </original_query> 
<paraphrased_query> 
What industries are most common in Willow Vale, NSW? 
</paraphrased_query> 
<paraphrased_query> 
Main economic activities in the Willow Vale area of New South Wales </paraphrased_query> 
<original_query> {Original Search Query} </original_query> 
Query Paraphraser 
Figure 7: Prompt for query paraphrasing action.
You are a highly capable critical rethinker agent. 
Given an original search query, you are tasked to critically assess the search query, and then generate a 
new and creative search query to support your critical thought. 
you are also tasked to return one reasoning thought for the new search query, explaining why the new 
query works better. 
The search query should be precise and focused. 
Your output must include: 
- One complete reasoning step that strongly rejects the entire retrieved information as unhelpful, 
irrelevant, or misleading, wrapped in a single pair of  <critical_rethinking> and </critical_rethinking> 
tags. 
- One creative and fundamentally new search query, wrapped in <search> and </search> tags. 
Only use the following format, in this exact order: 
<critical_rethinking> one complete reasoning step that strongly rejects the entire retrieved information 
as unhelpful, irrelevant, or misleading </critical_rethinking> 
<search> a creative, focused, and fundamentally new search query </search> 
Here are some examples: 
<original_query> popular industry in the neighborhood of … </original_query> 
<critical_rethinking> The query “popular industry in the ...</critical_rethinking> 
<search> economic activities and land use patterns in Southern … </search> 
<original_query> {Original Search Query} </original_query> Critical Re-Thinker 
Figure 8: Prompt for critical rethinking action.
B Abstention Task
B.1 Evaluation Metrics
We evaluate R2C on the abstention task in Section 5.2. Following
Feng et al . [14] , we adopt their definitions for the evaluation met-
rics. Assume we have a confusion matrix with four elements, each
You are a highly capable response validation agent. 
Given a user query and a summary of the retrieved documents used during the reasoning process, your 
task is to verify whether the generated response satisfies two evaluation criteria. The two criteria are: 
1) Is the response grounded in the provided information? 2) Does the response correctly and fully 
answer the user query? 
You must provide a single, coherent reasoning step that examines both criteria and suggests how the 
response could be improved. 
After your reasoning, you must return a precise search query that can help retrieve better information to 
improve the answer. 
The document summary will be enclosed in <information> and </information> tags. This content is 
read-only: NEVER generate, modify, or repeat the <information> tags. 
The predicted answer will be enclosed in <prediction> and </prediction> tags. This content is also 
read-only: NEVER generate, modify, or repeat the <prediction> tags. 
The search query must be focused, informative, and aimed at enhancing the predicted answer. 
Your output must include: 
- One complete reasoning step that: (a) references the predicted answer, (b) evaluates it using the two 
criteria, and (c) proposes specific improvements. Wrap this reasoning inside a single pair of <think> 
and </think> tags. 
- One search query that would help improve the answer. Wrap it inside <search> and </search> tags. 
Only use the following format, in this exact order: 
<think> one complete reasoning step that assesses the answer </think> 
<search> a creative and focused search query </search> 
Question: {User Query} 
<information> {Documents Summary} </information> 
<prediction> {Prediction} </prediction> 
Answer Validator Figure 9: Prompt for answer validation action.
You are a highly capable summarization agent. 
Your task is to generate a detailed summary based on the provided information and the 
user's query. 
Your output must include: 
- One complete and detailed summary, wrapped in a single pair of <summary> and 
</summary> tags. 
Your output must follow this exact format and order: 
<summary> one complete summary of the information considering the user query 
</summary> 
<information> {Documents} </information> 
<user_query> {User Query} </user_query> Reasoning Path Summarization 
Figure 10: Prompt for reasoning path summarization used
in the answer validation action.
denoted by a character: (A) Answered Correct, (B) Abstained Cor-
rect, (C) Answered Incorrect, and (D) Abstained Incorrect. Based
on these, four metrics are defined for the abstention task:
(1)Reliable Accuracy:𝐴
𝐴+𝐶, measures how trustworthy the LLM’s
generated (non-abstained) answers are; that is, among all an-
swered questions, how many are correct?
(2)Effective Reliability:𝐴−𝐶
𝐴+𝐵+𝐶+𝐷, balances reliability and cover-
age; that is, across all questions, what proportion are answered
correctly minus those answered incorrectly?
(3)Abstain Accuracy:𝐴+𝐷
𝐴+𝐵+𝐶+𝐷, evaluates whether abstention de-
cisions are correct; ideally, LLMs should abstain when it would
provide an incorrect answer and vice versa.
(4)Abstain F1:the harmonic mean of precision and recall, where
precision =𝐷
𝐵+𝐷and recall =𝐷
𝐶+𝐷, providing a balanced mea-
sure between reliability and answer coverage.
In this paper, we reportAbstain AccuracyandAbstain F1, as our pri-
mary goal is to evaluate the abstention capability of the uncertainty
scores.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
Table 4: Abstention performance measured by the threshold-free metric AUARC. For each column, the best and second-best
methods are indicated in bold and underlined, respectively. The superscript†denotes a statistically significant difference
compared to ReaC based on the bootstrap test (𝑝< 0.05).
RAG SelfAsk [39] ReAct [64] Search-o1 [30] ReSearch [5] Search-R1 [24]Avg.
Uncer. M. Popqa Hotpot Musiq. Popqa Hotpot Musiq. Popqa Hotpot Musiq. Popqa Hotpot Musiq. Popqa Hotpot Musiq.
RrrC [29] 52.68 45.25 19.42 41.77 40.25 18.71 44.96 41.58 16.43 45.26 52.13 24.44 56.50 54.70 22.19 38.42
P(true) [25] 56.26 49.53 18.35 56.53 38.82 15.63 54.47 39.75 15.37 60.42 55.03 29.54 61.11 58.72 24.55 42.27
SelfC [58] 55.28 44.63 15.35 59.88 45.13 20.88 56.09 41.83 21.10 62.36 54.38 29.55 58.31 51.53 24.76 42.67
ReaC [40] 56.47 48.32 18.72 60.81 45.48 20.52 56.98 42.55 21.61 63.1355.21 29.63 54.02 55.73 27.25 43.83
R2C (our) 58.87†53.81†19.60 62.74†47.62†24.45†60.72†46.65†21.8461.1959.25†31.91 66.06†60.90†31.58†47.15
We are evaluating answers to the question: In what school district is Governor John R. Rogers 
High School, named after John Rankin Rogers, located? 
Here are two possible answers: 
Possible Answer 1: Puyallup School District of Washington 
Possible Answer 2: Puyallup School District 
For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond 
with Yes or No. 
Response: Yes 
We are evaluating answers to the question: Which team featured in both the 2012 and 2011 
Cops del Rey Finals? 
Here are two possible answers: 
Possible Answer 1: Barcelona 
Possible Answer 2: Spain 
For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond 
with Yes or No. 
Response: No 
We are evaluating answers to the question: Which genus of flowering plant is found in an 
environment further south, Crocosmia or Cimicifuga? 
Here are two possible answers: 
Possible Answer 1: Crocosmia 
Possible Answer 2: Iridaceae 
For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond 
with Yes or No. 
Response: No 
We are evaluating answers to the question: Is It Just Me? was a single by the English rock band 
from what Suffolk city? 
Here are two possible answers: 
Possible Answer 1: Lowestoft 
Possible Answer 2: Lowestoft, Suffolk 
For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond 
with Yes or No. 
Response: Yes 
We are evaluating answers to the question: In what year did the man who shot the Chris 
Stockley, of The Dingoes, die? 
Here are two possible answers: 
Possible Answer 1: 1987 
Possible Answer 2: The year of 1987 
For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond 
with Yes or No. 
Response: Yes 
We are evaluating answers to the question: {User Query} 
Here are two possible answers: 
Possible Answer 1: {Answer A} 
Possible Answer 2: {Answer B} 
For this question, is Possible Answer 1 semantically equivalent to Possible Answer 2? Respond 
with Yes or No. 
Response: Semantic Equivalence of Responses 
Figure 11: Prompt designed to evaluate the semantic equiva-
lence between two responses to a user query.
B.2 Threshold Calibration
To determine the threshold 𝜏abs, we perform a parameter sweep
using validation sets. To construct a validation set for each dataset,
we subsample 100 examples from the training set of each dataset.The only exception is PopQA, which does not have a training set.
For PopQA, we instead subsample from the original test set while
ensuring that our validation and test sets do not overlap. (As a
reminder, as described in Section 5.1, we sample 500 examples for
the test set, and the PopQA dataset consists entirely of 14K test
samples.) We then generate uncertainty scores for the validation
set using R2C as well as all baseline methods.
After obtaining the validation sets and corresponding uncer-
tainty scores, we sweep the threshold values from0 .4to0.95with
an interval of0 .05, evaluating both AbstainAccuracy and AbstainF1.
We conduct this procedure across all datasets and for all baseline
methods, including RrrC, ReaC, SelfC, P(true), and our proposed ap-
proach. We first observe that both metrics exhibit similar behavioral
patterns. Our results further show that all methods achieve their
best performance at a threshold of0.9; therefore, we set𝜏 abs=0.9.
B.3 Evaluation with AUARC
In Table 2, we present the performance of the abstention task us-
ing two threshold-based metrics: AbstainAccuracy and AbstainF1.
The decision thresholds are determined through the detailed ex-
periments described in Appendix B.2. However, some studies in
the literature [ 12,13,41] adopt the threshold-free Area Under the
Accuracy–Rejection Curve (AUARC) [ 37] as the evaluation metric.
While AUARC has the advantage of being independent of a specific
threshold, its final score is correlated with the model’s overall accu-
racy. Considering these pros and cons, we also report the abstention
task results using AUARC in Table 4. We observe that the model
rankings remain consistent with those in Table 2, indicating that the
thresholds for AbstainAccuracy and AbstainF1 were appropriately
selected. Moreover, even with the threshold-free metric, the R2C
method continues to outperform the other baselines.
C Case Study
Figure 1 illustrates the workflow of the UQ for computing the
uncertainty score, which consists of two main steps: Most-likely
Generation and Multi-Generations. Based on this workflow, we
present two case studies—one successful and one failed, in Fig-
ures 12 and 13, respectively. In the successful case, the most-likely
response is “Thebarton Oval”, which is incorrect. However, in the
Multi-Generations step, only one out of ten generated responses is
“Thebarton Oval”. According to Equations (1)and(2), this results

Uncertainty Quantification for Retrieval-Augmented Reasoning Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Step 2 
Think: 
Search Query: primary location of Port Adelaide Magpies 
Document Set: [20056076, 13948138, 12034482] Step 3 
Think:  The query \"home base of Port Adelaide Magpies\" only seeks to identify a location, which is basic and 
does not provide any meaningful context about the team's history, significance, or current status. This type of 
query does not offer a chance to explore deeper insights into the club's identity, achievements, or the 
community it serves. 
Search Query:  cultural impact and historical milestones of Port Adelaide Magpies football club 
Document Set: [17158335, 17158338, 17158341] Step 1 
Think:  I need to find the home base of the football club that defeated Central District Bulldogs in the 2003 South Australian National Football League Grand Final. I'll search for it. 
Search Query:  The 2003 South Australian National Football League Grand Final saw the Central District Bulldogs defeat a football club 
Document Set:  [19754953, 2324076, 14830716]         The 2003 South Australian National Football League Grand Final saw the Central District Bulldogs defeat a football club whose home base is currently known as what? 
        City Mazda Stadium 
Most-likely Generation 
Step 2 
Think:  I found out that the football club that defeated Central District Bulldogs in the 2003 South Australian National Football League Grand Final is Port Adelaide Magpies. Now I need to find the home base of this football club. 
Search Query:  home base of Port Adelaide Magpies 
Document Set:  [15035546, 11506432, 16060161] 
Step 3 
Think:  Port Adelaide Magpies' home base is currently known as , Thebarton Oval 
Answer:  Thebarton Oval 
Multi Generations 
Step 1 Generation 1: Critical Thinking, 2 
Step 2 
Step 3 
Think:  The query \"home base of Port Adelaide Magpies\" is overly simplistic and fails to recognize that the 
term \"home base\" might be misleading for a football (AFL) club, which often plays in multiple venues. 
Additionally, it doesn't explore the rich history or current operations of the club, including its training 
facilities, administration offices, and community engagement, all of which are more relevant to understanding 
the club's operational infrastructure. 
Search Query:  main training and administrative facilities of Port Adelaide Magpies AFL club 
Document Set: [9343454, 9242855,  9242851] 
Step 4 
Think:  I found out that the home base of Port Adelaide Magpies is AAMI Stadium. Now I can provide the 
answer 
Answer:  AAMI Stadium Step 1 Generation 2: Critical Thinking, 2 
Step 2 
Step 4 
Think:   I found out that the home base of the Port Adelaide Magpies is Thebarton Oval. Now I can provide the 
answer 
Answer:  Thebarton Oval 
Step 1 
Think: 
Search Query: In the 2003 South Australian National Football League Grand Final, who did Central District 
Bulldogs defeat? 
Document Set: [19754953, 2324076, 15704148] Generation 5: Query Paraphrasing, 1  
Think 
Step 2 
Think:  ‘’
Search Query: home base of the football club that defeated Central District Bulldogs in the 2003 South 
Australian National Football League Grand Final 
Document Set:  [19754953, 2324076, 15704148] Step 1 Generation 6: Query Paraphrasing, 2 
Step 4 
Think:  I found out that the home base of the Port Adelaide Magpies football club is in Adelaide, South 
Australia. Now I can provide the answer. 
Answer: Football Park Think 
Step 3 
Think:  I found out that the home base of the Port Adelaide Magpies is Football Park. Now I can provide the 
answer. 
Answer: Football Park Step 1 Generation 4: Critical Thinking, 2 
Step 2 
Step 3 
Think:  The query \"home base of Port Adelaide Magpies\" only seeks to identify a physical location, which is 
easily found through a simple database lookup and offers no deeper insight into the club's history, current 
status, or significance. This approach fails to explore the rich cultural, sporting, and community context 
surrounding the team, which would provide more valuable and interesting information. 
Search Query:  cultural impact and historical significance of Port Adelaide Magpies in Australian football 
Document Set: [2827235, 17158341,  17158343] 
Step 4 
Think:  ‘’ 
Answer:  AAMI Stadium 
Generation 7: Critical Thinking, 2 
Step 1 Step 2 
Step 3 
Think:  The query \"home base of Port Adelaide Magpies\" assumes that there is a single, well-documented 
home base for the Port Adelaide Magpies, which may not be accurate. The term \"home base\" could be 
interpreted narrowly as a physical location, but it might also refer to the primary competition, training 
ground, or fan base area, all of which might not be clearly defined or easily locatable online. Moreover, the 
Port Adelaide Magpies are part of the Australian Football League (AFL), and their history and operations can 
be complex and multifaceted, requiring more than just a simple geographical answer. 
Search Query:  historical and current locations and significance of the Port Adelaide Magpies' home ground in 
AFL
Document Set: [18170918, 18170925, 9242851] 
Step 4 
Think:  I found out that the home base of the Port Adelaide Magpies is currently Adelaide Oval. Now I can 
provide the answer. 
Answer:  Adelaide Oval Step 1 Generation 3: Critical Thinking, 2 
Step 2 
Step 3 
Think:  The query \"home base of Port Adelaide Magpies\" is too narrow and assumes that there is a single 
definitive location, which might not be accurate or complete. It does not account for the possibility that the 
club has multiple locations or that its history involves significant changes in its home base over time. 
Moreover, it fails to explore the cultural, geographical, or strategic reasons behind the choice of location, 
which could provide more insightful context. 
Search Query:  evolution of home grounds and their significance for Port Adelaide Magpies football club 
Document Set: [9242851, 5088538, 18170925] 
Step 4 
Think:   I found out that the home base of Port Adelaide Magpies is currently AAMI Stadium. Now I can 
provide the answer. 
Answer:  AAMI Stadium 
Generation 10: Answer Validation 
Path Summary 
Based on the provided information, there is no mention of the 2003 South Australian National Football 
League (SANFL) Grand Final or the Central District Bulldogs defeating another football club in that specific 
year. The information provided covers details about the 1995, 1996, and 1998 SANFL Grand Finals, where Port 
Adelaide Magpies won. No relevant information is available to answer the user's query regarding the 2003 
SANFL Grand Final. 
Step 4 
Think:  The response suggests \"Thebarton Oval\" as the current home base of the losing team from the 2003 
SANFL Grand Final. However, the provided information does not contain any data related to the 2003 SANFL 
Grand Final or the losing team from that match. Therefore, the response is not grounded in the provided 
information and does not fully answer the user's query. To improve the answer, we need to find reliable 
sources about the 2003 SANFL Grand Final and the current home base of the losing team. A focused search 
on the 2003 SANFL Grand Final results and the current home base of the losing team would be beneficial. 
Search Query:  2003 SANFL Grand Final results and losing team's current home ground 
Document Set: [4831635, 15186536, 14830878] 
Step 5 
Think:  I found out that the 2003 SANFL Grand Final saw the Central District Bulldogs defeat Port Adelaide. 
Now I need to find the current home base of the Port Adelaide football club. 
Search Query:  current home base of Port Adelaide football club 
Document Set: [15704272, 762533, 9242850] 
Step 6 
Think:  I found out that the current home base of the Port Adelaide football club is Adelaide Oval. Now I can 
provide the answer. 
Answer: Adelaide Oval Generation 9: Answer Validation 
Path Summary 
The information provided does not contain details about the 2003 South Australian National Football League 
(SANFL) Grand Final or the specific outcome involving the Central District Bulldogs. The documents primarily 
focus on matches between Port Adelaide Magpies and Central District Bulldogs from 1995 to 1999, and other 
related information about the clubs and the league. There is no mention of a 2003 Grand Final or the home 
base of the Central District Bulldogs after that year. Therefore, it is not possible to determine the home base 
of the football club that defeated Central District Bulldogs in the 2003 SANFL Grand Final based on the given 
information. 
Step 4 
Think:  The response \"Thebarton Oval\" is incorrect because the provided information does not contain 
details about the 2003 SANFL Grand Final or the home base of the football club that defeated Central District 
Bulldogs that year. The prediction is not grounded in the provided information and cannot be considered 
correct. To improve the response, we need to find reliable sources that provide information about the 2003 
SANFL Grand Final and the home base of the opposing team. \n\nAdditionally, the response should clearly 
state that the information is not available in the given documents and suggest a search for the specific match 
result. 
Search Query:  2003 South Australian National Football League Grand Final result 
Document Set: [4831634, 15186536, 15277331] 
Step 5 
Think:  I found out that the 2003 AFL Grand Final was between the Brisbane Lions and the Collingwood 
Football Club. However, the question asks about the 2003 South Australian National Football League (SANFL) 
Grand Final, which involves the Central District Bulldogs. I need to find the information about the SANFL 
Grand Final and the home base of the football club that defeated the Central District Bulldogs. 
Search Query:  2003 South Australian National Football League Grand Final result 
Document Set: [4831634, 15186536, 15277331] 
Step 8 
Think:  I found out that the football club that defeated Central District Bulldogs in the 2003 SANFL Grand Final 
was Port Adelaide Magpies. Now I need to find the home base of Port Adelaide Magpies. 
Answer: Football Park Step 6 
Think:  I found out that the 2003 AFL Grand Final was between the Brisbane Lions and the Collingwood 
Football Club. Now I need to find the 2003 SANFL Grand Final result and the home base of the football club 
that defeated the Central District Bulldogs. 
Search Query:  2003 SANFL Grand Final result 
Document Set: [4831634, 14830878, 15186536] 
Step 7 
Think:  I found out that the 2003 SANFL Grand Final was won by Central District Bulldogs. Now I need to find 
the home base of the football club that defeated Central District Bulldogs in the 2003 SANFL Grand Final. 
Search Query:  home base of the football club that defeated Central District Bulldogs in the 2003 SANFL Grand 
Final 
Document Set: [19754953, 2324076, 15704148] Generation 8: Answer Validation 
Path Summary 
The information provided does not contain details about the 2003 South Australian National Football League 
(SANFL) Grand Final or the specific club that Central District Bulldogs defeated in that match. The available 
documents focus on SANFL grand finals from 1995 to 1996, the history of the Central District Football Club, 
and the merger between AFL-based Port Adelaide and SANFL-based Port Adelaide Magpies. Since the user 
query asks about the 2003 SANFL Grand Final, which is not covered in the given information, it is not possible 
to determine the name of the club that Central District Bulldogs defeated based on the provided data. 
Step 5 
Think:  I found out that the 2003 AFL Grand Final was between the Brisbane Lions and the Collingwood 
Football Club, and it was won by the Brisbane Lions. However, the question asks about the 2003 South 
Australian National Football League (SANFL) Grand Final and the football club whose home base is currently 
known as what. I need to find the club that Central District Bulldogs defeated in the 2003 SANFL Grand Final. 
Search Query:  2003 South Australian National Football League Grand Final 
Document Set: [4831634, 15186536, 15277331] 
Step 7 
Think:  I found out that the winner of the 2003 SANFL Grand Final was Central District Bulldogs. Now I need to 
find the home base of the defeated football club. 
Answer: ‘’Step 4 
Think:  The response \"Thebarton Oval\" is not grounded in the provided information and does not directly 
answer the user's query. There is no mention of the 2003 SANFL Grand Final or the losing team in the given 
documents. To improve the answer, we need more specific details about the 2003 SANFL Grand Final and the 
identity of the opposing team. A focused search should provide the name of the football club that was 
defeated by Central District Bulldogs in the 2003 SANFL Grand Final. 
Search Query:  2003 South Australian National Football League Grand Final result 
Document Set: [4831634, 15186536, 15277331] 
Step 6 
Think:  I need to find the football club that Central District Bulldogs defeated in the 2003 South Australian 
National Football League (SANFL) Grand Final. I will search for the name of the club. 
Search Query:  SANFL Grand Final 2003 winner 
Document Set: [14830878, 17871680, 19616203] Step 3 
Think:  I found out that the football club that defeated Central District Bulldogs in the 2003 South Australian 
National Football League Grand Final was Port Adelaide Magpies. Now I need to find the home base of this 
football club. 
Search Query: home base of Port Adelaide Magpies football club 
Document Set:  [762533, 4501138, 15301406] 
Figure 12: A case study of a successful sample from the HotpotQA dataset using the RAR model (Search-R1). The predicted
answer, “Thebarton Oval,” is incorrect, and the model assigns it a relatively high uncertainty score of 0.9, indicating a desirable
correlation between uncertainty and correctness.
in an uncertainty score of 0.9, indicating high uncertainty. In con-
trast, in the failed case, the most-likely response is “Adrian Lyne”,
which is correct. Yet, in the Multi-Generations step, only four out
of ten responses are “Adrian Lyne”, again leading to a relatively
high uncertainty score. In this sample, we observe that the answer
validation action performs correctly and generates a response simi-
lar to the most-likely one. However, the query paraphrasing actionproduces different responses, and two of the three critical rethink-
ing actions also yields a different result. This observation supports
the discussion in Section 6.5, which highlights the potential impact
of action selection configurations—whether applied per sample, per
model, or per dataset.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Soudani et al.
Step 1 
Think:  I need to find the director of the 1987 American psychological thriller film in which the actor who played Bob Charles in \"Not Necessarily the News\" appeared. I'll search for it. 
Search Query:  actor who played Bob Charles in "Not Necessarily the News” 
Document Set:  [18789586 , 9391749 , 17877035 ]        Who was the director of the 1987 American psychological thriller film in which the actor, who played Bob Charles in "Not Necessarily the News", appeared? 
        Adrian Lyne 
Most-likely Generation 
Step 2 
Think:  I found out that the actor who played Bob Charles in \"Not Necessarily the News\" is Bryan Murray. Now I need to find the 1987 American psychological thriller film in which he appeared. 
Search Query:  1987 American psychological thriller film in which Bryan Murray appeared 
Document Set:  [17906015 , 18129571 , 11885203 ]
Step 4 
Think:  I found out that the director of the 1987 American psychological thriller film in which Bryan Murray appeared is Adrian Lyne. 
Answer:  Adrian Lyne 
Multi Generations 
Generation 1: query paraphrasing, 3 
Step 3 
Think: 
Search Query: Who directed the 1987 psychological thriller featuring Bryan Murray? 
Document Set: [20557386, 18129571, 4092928] Generation 2: Query paraphrasing, 3 
Think 
Step 4 
Think:  I need to find the director of the 1987 American psychological thriller film in which Bryan Murray 
appeared. From the search, I found out that Bryan Murray appeared in the film \"The Bedroom Window". 
Now I need to find the director of this film. 
Search Query: director of The Bedroom Window 
Document Set:  [20557386, 18129571, 20557385] Generation 3: Answer Validation 
Path Summary 
Bryan Murray, the actor who played Bob Charles in the soap opera \"Fair City,\" appeared in the 1987 
American psychological thriller film \"Fatal Attraction,\" directed by Adrian Lyne. \"Fatal Attraction\" starred 
Michael Douglas, Glenn Close, and Anne Archer, and it was based on a screenplay by James Dearden, adapted 
from his 1980 short film \"Diversion.\" This film was well-received critically and became a significant 
commercial success, grossing substantially at the box office despite initial controversies. 
Step 6 
Think:  ‘’
Search Query:  director of the 1987 American psychological thriller film in which the actor, who played Bob 
Charles in \"Not Necessarily the News\", appeared 
Document Set: [2789203, 2789207, 19505329] 
Step 9 
Think:  I need to find the director of the 1987 American psychological thriller film \"Fatal Attraction\" in which 
Stuart Pankin appeared. 
Answer: Adrian Lyne Step 5 
Think:  The response states that Adrian Lyne was the director of the 1987 film "Fatal Attraction," which aligns 
with the information provided. However, the answer should be more directly linked to the given information 
by specifying that Bryan Murray, the actor who played Bob Charles in "Not Necessarily the News," appeared 
in this film. The response meets the grounding criterion but falls short on completeness. To improve, the 
answer should explicitly connect the actor's role in "Not Necessarily the News" to their appearance in "Fatal 
Attraction”. Specifically, the response should mention Bryan Murray by name and reference his role in "Not 
Necessarily the News." 
Search Query:  information on Bryan Murray's filmography including "Not Necessarily the News" and "Fatal 
Attraction 
Document Set: [19266549, 1897855, 2789204] 
Step 7 
Think:  I need to find the director of the 1987 American psychological thriller film in which the actor, who 
played Bob Charles in \"Not Necessarily the News\", appeared. I found out that the actor who played Bob 
Charles in \"Not Necessarily the News\" is Stuart Pankin. Now I need to find the film in which he appeared 
and was a psychological thriller. 
Search Query:  who was the director of the 1987 American psychological thriller film in which the actor, who 
played Bob Charles in \"Not Necessarily the News\", appeared 
Document Set: [19505329, 2789203, 4999074] Step 3 
Think:  
Search Query:  director of the 1987 American psychological thriller film in which Bryan Murray appeared 
Document Set:  [17906015 , 18129571 , 11885203 ]
Step 1 Step 2 
Step 5 
Think:  I need to find the director of the 1987 American psychological thriller film \"The Bedroom Window\" in 
which Bryan Murray appeared. From the search, I found out that the director of "The Bedroom Window" is 
Curtis Hanson. 
Answer: Curtis Hanson Step 3 
Think: 
Search Query: 1987 American psychological thriller with Bryan Murray: name the director 
Document Set: [20557386, 18129571, 11885203] Think 
Step 4 
Think:  I found out that the 1987 American psychological thriller film in which Bryan Murray appeared is \"The 
Bedroom Window\". Now I need to find the director of this film. 
Search Query: director of The Bedroom Window (1987 film) 
Document Set:  [20557386, 18129571, 20557385] Step 1 Step 2 
Step 5 
Think:  ‘’
Answer: Curtis Hanson 
Step 8 
Think:  I need to find the 1987 American psychological thriller film in which the actor, who played Bob Charles 
in \"Not Necessarily the News\", appeared. I found out that the actor who played Bob Charles in \"Not 
Necessarily the News\" is Stuart Pankin. Now I need to find the film in which he appeared and was a 
psychological thriller. 
Search Query:  1987 American psychological thriller film in which Stuart Pankin appeared 
Document Set: [2789205, 2789206, 17906015] 
Generation 7: Answer Validation 
Path Summary 
The actor who played Bob Charles in \"Not Necessarily the News\" is Bryan Murray. Bryan Murray appeared in 
the 1987 American psychological thriller film \"Fatal Attraction,\" directed by Adrian Lyne. This film starred 
Michael Douglas, Glenn Close, and Anne Archer and was based on a short film called \"Diversion.\" It received 
generally positive critical response and became a significant commercial success at the box office. 
Step 5 
Think:  The response states that Adrian Lyne was the director of the 1987 American psychological thriller film 
featuring Bryan Murray, who played Bob Charles in \"Not Necessarily the News\". Evaluating this against the 
criteria: \n1. The response is grounded in the provided information, as the document mentions that Bryan 
Murray appeared in \"Fatal Attraction,\" directed by Adrian Lyne.\n2. However, the response only partially 
answers the query, as it does not specify the film title. To fully satisfy the query, the answer should mention 
that Bryan Murray appeared in \"Fatal Attraction.\"\n\nTo improve the response, we should include the name 
of the film in the answer. 
Search Query:  title of the 1987 American psychological thriller film starring Bryan Murray 
Document Set: [18129571, 20557386, 11885203] 
Step 6 
Think:  I found out that the 1987 American psychological thriller film in which the actor who played Bob 
Charles in \"Not Necessarily the News\" appeared was \"Fatal Attraction\". Now I need to find the director of 
this film. 
Search Query:  director of Fatal Attraction 
Document Set: [11885203, 11885231, 17819054] 
Step 7 
Think:  I found out that the director of the 1987 American psychological thriller film \"Fatal Attraction\" is 
Adrian Lyne. 
Answer: Adrian Lyne Generation 5: Critical rethinking, 2 
Step 1 Step 2 
Step 3 
Think:  The query \"1987 American psychological thriller film in which Bryan Murray appeared\" is overly 
specific and relies on a known actor\u2019s appearance in a film, which can lead to retrieving only the film's 
title and basic plot details. This query does not explore the film's cultural impact, the director\u2019s style, or 
the broader genre characteristics of psychological thrillers from that era, thus limiting the depth and breadth 
of useful information. 
Search Query:  influential psychological thriller films of the 1980s and their impact on the genre 
Document Set: [988461, 17449124, 158483] 
Step 5 
Think:  I need to find the director of the 1987 American psychological thriller film \"The Bedroom Window\" in 
which Bryan Murray appeared. From the search, I found out that the director of \"The Bedroom Window\" is 
Curtis Hanson. 
Answer:  Curtis Hanson Step 4 
Think:  I need to find the director of the 1987 American psychological thriller film in which the actor Bryan 
Murray appeared. I found out that Bryan Murray played Bob Charles in \"Not Necessarily the News\". Now I 
need to find the psychological thriller film he appeared in. From the search, I found out that Bryan Murray 
appeared in the film \"The Bedroom Window\" (1987). Now I need to find the director of this film. 
Search Query:  director of The Bedroom Window (1987 film) 
Document Set: [20557386, 18129571, 20557385] Generation 4: Answer Validation 
Path Summary 
The 1987 American psychological thriller film in which the actor who played Bob Charles in \"Not Necessarily 
the News\" appeared was \"Fatal Attraction\". The director of \"Fatal Attraction\" was Adrian Lyne. According 
to the information provided, Bryan Murray played Bob Charles in the soap opera \"Fair City\", not \"Not 
Necessarily the News\". However, the actor who played Bob Charles in \"Not Necessarily the News\" is not 
mentioned in the given documents. Since Bryan Murray appeared in \"Fatal Attraction\", the director of that 
film was Adrian Lyne. 
Step 5 
Think:  The response identifies the director as Adrian Lyne, but it does not directly ground this conclusion in 
the provided information. It mentions \"Fatal Attraction\" as the 1987 American psychological thriller film 
featuring the actor who played Bob Charles in \"Not Necessarily the News\", and confirms that Adrian Lyne 
directed \"Fatal Attraction\". However, the response could be more clearly linked to the specific information 
provided about the actor's role. Additionally, the response could benefit from clarifying that the correct actor 
for Bob Charles in \"Not Necessarily the News\" is needed to ensure accuracy. \n\nTo improve, we should 
directly reference the connection between the actor who played Bob Charles in \"Not Necessarily the News\" 
and the film \"Fatal Attraction\". 
Search Query:  which actor played bob charles in not necessarily the news 
Document Set: [2789203, 2789207, 2789204] 
Step 6 
Think:  I found out that the actor who played Bob Charles in \"Not Necessarily the News\" appeared in the 
film \"Fatal Attraction\". Now I need to find the director of the 1987 American psychological thriller film 
\"Fatal Attraction\". 
Search Query:  director of Fatal Attraction 
Document Set: [11885203, 11885231, 17819054] 
Step 7 
Think:  I found out that the director of the 1987 American psychological thriller film \"Fatal Attraction\" is 
Adrian Lyne. 
Answer: Adrian Lyne Generation 8: Critical rethinking, 1 
Step 1 
Step 2 
Think:  The query \u2018actor who played Bob Charles in \"Not Necessarily the News\"' is overly specific and 
relies on a single piece of information that is likely to be obscure and difficult to verify. It fails to explore the 
broader context of the character, the show, or the actor\u2019s career, thereby limiting the scope of useful 
information that could be derived from the search results. 
Search Query:  background and significance of the character Bob Charles in \"Not Necessarily the News\" 
Document Set: [2789203, 2789207, 1897855] 
Step 5 
Think:  I found out that Stuart Pankin appeared in the film \"Fatal Attraction\" directed by Adrian Lyne. Now I 
can provide the answer. 
Answer:  Adrian Lyne Step 4 
Think:  I found out that the actor who played Bob Charles in \"Not Necessarily the News\" is Stuart Pankin. 
Now I need to find the 1987 American psychological thriller film in which he appeared. 
Search Query:  1987 American psychological thriller film in which Stuart Pankin appeared 
Document Set: [2789205, 2789206, 17906015] Generation 6: Critical rethinking, 1 
Step 1 
Step 2 
Think:  The query 'actor who played Bob Charles in \"Not Necessarily the News\"' is overly specific and relies 
on outdated media content that may not be well-documented or easily searchable. It assumes that Bob 
Charles was a character with significant screen time, which might not be accurate, and it focuses solely on 
one actor, potentially missing other related information or interpretations of the show. 
Search Query:  notable characters and actors in the comedy series \"Not Necessarily the News\" 
Document Set: [1897854, 5064235, 5064238] 
Step 6 
Think:  ‘’
Answer:  Curtis Hanson Step 3 
Think:  I need to find the director of the 1987 American psychological thriller film in which the actor who 
played Bob Charles in \"Not Necessarily the News\" appeared. I'll search for it. 
Search Query:  actor who played Bob Charles in \"Not Necessarily the News\" 
Document Set: [2789203, 2789207, 18789586] 
Step 4 
Think:  ‘’
Search Query:  1987 American psychological thriller film 
Document Set: [18129571, 20557386, 11885203] 
Step 5 
Think:  I found out that the 1987 American psychological thriller film in which Bryan Murray appeared is \"The 
Bedroom Window\". Now I need to find the director of this film. 
Search Query:  director of The Bedroom Window (1987 film) 
Document Set: [20557386, 18129571, 20557385] Generation 9: query paraphrasing, 2 
Step 2 
Think: 
Search Query: 1987 psychological thriller movie featuring Bryan Murray 
Document Set: [20557386, 18129571, 4092928] Think 
Step 3 
Think:  I found out that the 1987 American psychological thriller film featuring Bryan Murray is \"The Bedroom 
Window\". Now I need to find the director of this film. 
Search Query: director of The Bedroom Window (1987 film) 
Document Set:  [20557386, 18129571, 20557385] Step 1 
Step 4 
Think:  I found out that the director of the 1987 American psychological thriller film \"The Bedroom Window\" 
is Curtis Hanson. 
Answer: Curtis Hanson Generation 10: query paraphrasing, 2 
Step 2 
Think: 
Search Query: Bryan Murray's role in an American psychological thriller from 1987 
Document Set: [17045675, 17045676, 18789586] Think 
Step 3 
Think:  ‘’
Search Query: 1987 American psychological thriller film 
Document Set:  [18129571, 20557386, 11885203] Step 1 
Step 5 
Think:  I found out that the director of the 1987 American psychological thriller film \"The Bedroom Window\" 
is Curtis Hanson. 
Answer: Curtis Hanson Step 4 
Think:  I found out that the 1987 American psychological thriller film in which Bryan Murray appeared is \"The 
Bedroom Window\". Now I need to find the director of this film. 
Search Query: director of The Bedroom Window (1987 film) 
Document Set:  [20557386, 18129571, 20557385] 
Figure 13: A case study of a failed sample from the HotpotQA dataset using the RAR model (Search-R1). Although the predicted
answer, "Adrian Lyne," is correct, the model assigned a relatively high uncertainty score of 0.6, revealing a mismatch between
uncertainty and correctness.
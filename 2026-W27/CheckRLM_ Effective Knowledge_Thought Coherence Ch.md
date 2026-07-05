# CheckRLM: Effective Knowledge-Thought Coherence Checking in Retrieval-Augmented Reasoning

**Authors**: Dingling Xu, Ruobing Wang, Qingfei Zhao, Yukun Yan, Zhichun Wang, Daren Zha, Shi Yu, Zhenghao Liu, Shuo Wang, Xu Han, Maosong Sun

**Published**: 2026-07-02 14:50:25

**PDF URL**: [https://arxiv.org/pdf/2607.02262v1](https://arxiv.org/pdf/2607.02262v1)

## Abstract
Reasoning Language Models (RLMs) have significantly improved performance on complex tasks by extending the reasoning chain. However, these chains are prone to containing factual errors, particularly in knowledge-intensive tasks. To address this issue, we propose CheckRLM, a framework that improves the reliability of the reasoning process through Retrieval-Augmented Generation (RAG) by timely checking and correcting factual errors. Specifically, CheckRLM extracts factual claims from the reasoning chain to identify and localize subtle knowledge inconsistencies during inference. Upon detection of errors, a refinement mechanism performs minimal-cost yet precise corrections by leveraging external knowledge, ensuring coherence between the reasoning chain and correct knowledge. Extensive experiments demonstrate that CheckRLM substantially outperforms existing baselines, exhibiting a strong capability to mitigate error accumulation in long-horizon reasoning with lower costs. The code and data are available at https://github.com/AI9Stars/CheckRLM.

## Full Text


<!-- PDF content starts -->

CheckRLM: Effective Knowledge-Thought Coherence Checking in
Retrieval-Augmented Reasoning
Dingling Xu12∗, Ruobing Wang34*, Qingfei Zhao34, Yukun Yan5†, Zhichun Wang12†,
Daren Zha3, Shi Yu5, Zhenghao Liu6, Shuo Wang5, Xu Han5, Maosong Sun5
1School of Artificial Intelligence, Beijing Normal University, Beijing 100875, China
2Beijing Key Laboratory of Artificial Intelligence for Education, Beijing 100875, China
3Institute of Information Engineering, Chinese Academy of Sciences,
4School of Cyber Security, University of Chinese Academy of Sciences,
5Department of Computer Science and Technology, Institute for AI, Tsinghua University,
6Northeastern University
Abstract
Reasoning Language Models (RLMs) have
significantly improved performance on com-
plex tasks by extending the reasoning chain.
However, these chains are prone to contain-
ing factual errors, particularly in knowledge-
intensive tasks. To address this issue, we pro-
poseCheckRLM, a framework that improves
the reliability of the reasoning process through
Retrieval-Augmented Generation (RAG) by
timely checking and correcting factual errors.
Specifically, CheckRLM extracts factual claims
from the reasoning chain to identify and lo-
calize subtle knowledge inconsistencies dur-
ing inference. Upon detection of errors, a re-
finement mechanism performs minimal-cost
yet precise corrections by leveraging external
knowledge, ensuring coherence between the
reasoning chain and correct knowledge. Exten-
sive experiments demonstrate that CheckRLM
substantially outperforms existing baselines,
exhibiting a strong capability to mitigate error
accumulation in long-horizon reasoning with
lower costs. The code and data are available at
https://github.com/AI9Stars/CheckRLM.
1 Introduction
Reasoning Language Models (RLMs) such as
OpenAI-o1 (OpenAI, 2024) and DeepSeek-R1
(Guo et al., 2025) have notable advanced reasoning
capabilities with reinforcement learning and test-
time scaling (Zhang et al., 2026; Gao et al., 2025).
By decomposing complex problems and engaging
in iterative exploration and reflection over long rea-
soning chains, RLMs attract significant attention
in logic-intensive domains, including mathematics
and coding (Chen et al., 2025; Li, 2025).
Despite these advancements, the extended rea-
soning chain also introduces the critical challenge
of maintaining factual coherence throughout the
*Equal contribution.202111081066@mail.bnu.edu.cn
†Corresponding authors. yanyk.thu@gmail.com ;
zcwang@bnu.edu.cn
Question:WhenwasthedirectoroffilmSlapHer...She‘SFrenchborn?Golden Answer: October 20, 1952
GoldenReasoningPath:1. SlapHer…She'sFrenchdirectorMelanieMayron        2. MelanieMayrondateofbirthOctober20,1952Direct Reasoning
... Jim Abrahams was born on February 25, 1940.… </think>Error AccumulationIncorrectReasoningChain
Answer: February 25, 1940<think> … The film “Slap Her ... She's French" … The director is Jim Abrahams.
<think> … the direct-or is Jim Abrahams.… was born on July 16, 1940. … </think>Check
<think> … the director is Melanie Mayron… I do not have the exact birth date… </think>Partially CorrectReasoningChain
Answer: 1965
CheckRLM<think>…Maybe it‘s direct-ed by someone like David Zucker? 
ContinueReasoning
Melanie Mayronwas born on July 21, 1968.
Melanie Mayronwas born on Octob-er 20,1952. …<think>Answer: October 20,1952
Check
CorrectReasoningChain
Directedby Melanie Mayron.Check
Post-reasoning Check
Figure 1:Illustration of error accumulation and
CheckRLM.Direct Reasoning and Post-reasoning
Check suffer from erroneous internal knowledge in
the RLM and insufficient external knowledge, whereas
CheckRLM corrects errors timely to reach the correct
answer.
process, especially in knowledge-intensive tasks
(He et al., 2025; Yao et al., 2025). In long step-
wise reasoning, we observe that the factual accu-
racy of each intermediate step is paramount, as it
forms the foundational premise for subsequent de-
ductions. Consequently, any factual error at a given
step can be propagated, leading to final answer
deviations–a phenomenon termed error accumu-
lation (Ling et al., 2023; Tyen et al., 2024).As
illustrated in Figure 1, Direct Reasoning initially
produces an incorrect director name “Jim Abra-
hams”, which is then propagated as a premise for
subsequent reasoning, leading to an erroneous in-
ference of the director’s birth date and the final
incorrect answer “February 25, 1940”. Although
Post-reasoning Check later revises the director’s
name, it fails to repair the already corrupted inter-
mediate reasoning.
To bridge this gap, we proposeCheckRLM,
a framework that employs Retrieval-Augmented
Generation (RAG) to promptly identify and cor-
1arXiv:2607.02262v1  [cs.CL]  2 Jul 2026

rect factual errors within long reasoning chains,
thereby aligning them with external knowledge.
To effectively check and refine reasoning chains,
CheckRLM comprises two components: in-process
knowledge claim recognition and localized knowl-
edge coherence correction via retrieval. During
in-process knowledge claim recognition, key fac-
tual claims pertinent to the query are identified
and extracted from each newly generated reasoning
segment, facilitating the location of potential in-
consistencies while minimizing interference from
irrelevant information. In the localized knowledge
coherence correction via retrieval process, we use
the extracted key factual claims along with the orig-
inal question to search an external knowledge base
and perform token-level precise corrections, en-
suring that each segment of the verified reasoning
chain aligns with external knowledge and reducing
error accumulation.
In summary, our contributions are as follows: 1)
we propose CheckRLM, an efficient RAG frame-
work with in-reasoning knowledge checking that
identifies and corrects factual errors during long-
chain reasoning, effectively preventing error accu-
mulation; 2) we introduce a localized intermediate
intervention mechanism that dynamically extracts
and verifies critical knowledge claims from partial
reasoning states, enabling timely and fine-grained
factual correction; 3) CheckRLM significantly out-
performs recent strong baselines, while comprehen-
sive analyses show that it enables early, targeted er-
ror correction and simultaneously achieves top-tier
performance with significantly reduced inference
time and token consumption.
2 Related Work
2.1 Reasoning Language Model (RLM)
RLMs enhance the quality of final answers by guid-
ing Large Language Models (LLMs) to decompose
complex tasks into multiple subquestions and by
extending step-by-step reasoning chains (Chen
et al., 2025; Gao et al., 2025; Zhang et al., 2026; Li,
2025). Prior work primarily relies on engineered
prompts during inference without modifying the in-
ternal parameters of LLMs (Yasunaga et al., 2024;
Zhou et al., 2023; Zhang et al., 2023; Yao et al.,
2023). Chain-of-Thought (Wei et al., 2022) in-
serts several task-oriented exemplars that contain
intermediate reasoning steps into the prompt, while
(Kojima et al., 2022) stimulates zero-shot gener-
ation of intermediate reasoning by adding the in-struction “let’s think step by step” to task instruc-
tions, effectively eliciting reasoning capabilities
in LLMs. However, these prompt-engineering ap-
proaches are largely constrained by the quality of
the prompts and exemplars, and the depth and cor-
rectness of the elicited reasoning are difficult to
guarantee. Recent studies (OpenAI, 2024; Guo
et al., 2025; Yang et al., 2025) have focused on
deep optimization of LLMs via large-scale rein-
forcement learning (RL). By designing verifiable
reward functions, RLMs exhibit high-level reason-
ing behaviors, including self-verification, reflec-
tion, and dynamic strategy adjustment; furthermore,
increasing test-time compute yields continued im-
provements in reasoning performance (Weng et al.,
2023; Kumar et al., 2025). Meanwhile, concurrent
studies have pointed out that these RLMs tend to
produce hallucinations within their long reasoning
chains (He et al., 2025; Yao et al., 2025). An
early factual error can subsequently mislead the en-
tire reasoning trajectory, leading to a phenomenon
known as “error accumulation” (Ling et al., 2023;
Tyen et al., 2024).
2.2 Retrieval-Augmented Generation (RAG)
RAG effectively mitigates the “hallucination” in
knowledge-intensive tasks by integrating LLMs
with external knowledge bases, improving the accu-
racy and reliability of content (Lewis et al., 2020;
Ram et al., 2023; Zhao et al., 2024; Zhu et al.,
2025). However, the single-retrieval mechanism
in vanilla RAG (Izacard et al., 2023) often yields
limited information, struggling to cover the com-
plete knowledge background required for complex
questions (Wang et al., 2025, 2024). To address
the limitations of Vanilla RAG, researchers have
proposed the adaptive RAG framework (Jeong
et al., 2024; Jiang et al., 2023; Asai et al., 2024;
Yao et al., 2022), which employs an intelligent
decision-making mechanism to dynamically deter-
mine when to retrieve, what to retrieve, and how
to integrate the retrieved information. This sig-
nificantly improves both the response quality and
efficiency of RAG systems. However, these tra-
ditional adaptive RAG methods are not suitable
for RLMs based on long chains of thought. Con-
current work (Hu et al., 2025; Sun et al., 2025;
Guan et al., 2026; Xiong et al., 2025; Zhao et al.,
2025; Li et al., 2025) combines reasoning with
search, inserting external knowledge into long rea-
soning chains to augment RLMs’ reasoning abil-
ity. These methods integrate external knowledge
2

在此处键⼊公式。Question: When was the director of film Slap Her... She‘SFrench born?Golden Answer: October 20, 1952
In-Process Knowledge Claim Recognition
RLMLocalized Knowledge Coherence Correction</think>Ithinkthemovieisfromthe1980s.Thetitlesoundsfamiliar.Maybeit’sdirected bysomeonelikeJimAbrahams?Wait,JimAbrahamswaspartoftheteamthatdidthePolice Academymovies1. The movie ’Slap Her... She’s French’ is from the 1980s.2. The Zucker-Abrahams-Zucker team directed many comedies around that time.
Reasoning Chain UnitFactual Claims
Factual Claims
Deduplicated documentsRetrievalIthinkthemovieisfromthe1980s.Thetitlesoundsfamiliar.Maybeit’sdirected bysomeonelikeJimAbrahams?Wait,JimAbrahamswaspartoftheteamthatdidthePolice Academymovies
Ithinkthemovieisfromthe1980s.Thetitlesoundsfamiliar.Maybeit’sdirected bysomeonelikeJimAbrahams?Wait,themovieisfrom2002,notthe1980s.butit’snotdirected bysomeonelikeDavidZucker.ThemovieisdirectedbyMelanieMayron.…Refined Reasoning Chain Unit
KnowledgeCorrectionModelReasoning Chain UnitTargeted correctionsMinimal disruption
𝑟!
"𝑟!𝑟""𝑟"𝑟#"𝑟#
Complete Reasoning ChainAnswer: October 20,1952
!𝑟𝑡𝑟𝑡Continue reasoning after correction
!𝑟𝟏!𝑟𝟏!𝑟𝟐IthinkMelanieMayronwasbornonJuly21,1968 …, maybeI’mmixingherupwithsomeoneelse.Anotherthought: …1. Melanie Mayronis the director of the film Slap Her... She’SFrench.2. Melanie Mayronwas born on July 21, 1968.MelanieMayron’sbirthdateisOctober20,1952. …I’llmakesurethere’snoconflictinginfo.… </think>
1. Melanie Mayron’sbirth date is October 20, 1952.   2. The director of the film Slap Her... She’SFrench is Melanie Mayron.
Extractfactualclaims
Extractfactualclaims
Claim Recognition Preference OptimizationCoherenceCorrection Preference OptimizationClaim Recognition Model
Factual Claims 1
Reasoning Chain Unit
Check Model forClaim Recognition
Factual Claims 2Factual Claims n
Evaluation Model
OptimizedModel
DPO Training
Current ReasoningChain
Check Model forCoherence Correction
"𝑟#
𝑟!𝑟"UnrefinedDeduplicateddocumentsRefined Unit 1Refined Unit 2Refined Unit n
Evaluation Model
OptimizedModel
DPO Training
ExtractfactualclaimsFigure 2:Overview of CheckRLM.In-process knowledge claim recognition locates error positions by extracting
factual claims from reasoning chain units, while localized knowledge coherence correction corrects factual errors at
minimal cost based on retrieved documents. The two modules are jointly optimized using DPO.
in a coarse-grained manner, which can introduce
instability in the coherence of the reasoning pro-
cess. Instead, we propose a minimal intervention
approach that corrects erroneous knowledge within
the model itself, achieving more reliable and con-
trollable knowledge-augmented reasoning capabili-
ties.
3 Methodology
3.1 Preliminary of CheckRLM
Considering a complicated question qthat requires
multi-step reasoning, our goal is to generate a cor-
rect final answer aby checking and refining the
reasoning chain Rgenerated by RLMs. Mathemat-
ically, the reasoning chain Rcan be defined as a
sequence τ= (s 1, s2, . . . , s T), where each strep-
resents an intermediate reasoning state generated
at time step t. The model parameters θgovern the
conditional distribution Pθ(st|q, s <t), denoting
the probability of generating the subsequent rea-
soning step stconditioned on the question qand
the preceding reasoning history s<t. Consequently,
the sampling process for the entire reasoning chain
Rcan be formalized as the product of stepwise
conditional distributions:
R ∼TY
t=1Pθ(st|q, s <t)(1)
The final answer ais then derived from both R
andq, where Pθ(a|q,R) represents the condi-
tional distribution under which the reasoning modelyields the final answer:
a∼P θ(a|q,R)(2)
3.2 Knowledge–Thought Coherence Checking
In this section, we first introduce a two-stage
Knowledge–Thought Coherence Checking process
(§3.2.1 and §3.2.2), and then describe the training
data construction pipeline and the training process
(§3.2.3). The overall framework of CheckRLM is
shown in Figure 2.
3.2.1 In-Process Knowledge Claim
Recognition
We adopt an “intermediate intervention” strategy
to extract explicit knowledge claims during the rea-
soning process, rather than intervening only after
the complete reasoning chain is generated. This
strategy aims to address potential factual errors and
prevent the gradual amplification of randomly oc-
curring erroneous knowledge in the intermediate
states of the reasoning chain at an early stage.
Specifically, we identify and extract factual in-
formation that supports problem-solving from the
reasoning chain, summarizing it into multiple ex-
plicit factual claims. These factual claims serve the
fact verification model Mrecto prioritize question-
relevant facts and facilitate the detection of ex-
plicit errors for correction, preventing their prop-
agation and negative impact on downstream rea-
soning. Considering that granular checks, such
as token-level or sentence-level, disrupt the logic
and semantic meaning of the reasoning chain and
3

trigger unnecessary or frequent interruptions, we
perform coherence checks at the paragraph level,
keeping a balance between contextual complete-
ness and timely verification.
Accordingly, the current reasoning chain unit
can be defined as the chain generated from st−1
tost, where the time step tis segmented by para-
graphs in the reasoning chain and denoted as rt.
We input the original question qand the current
reasoning chain unit rtinto the claim recognition
model Mrecto sample a set of knowledge claims
yclaim
t. For robustness considerations, we do not
use the entire previously generated reasoning chain
R<tas input during recognition to avoid introduc-
ing noise from earlier reasoning steps. We formal-
ize In-Process Knowledge Claim Recognition as
follows:
yclaim
t∼ M rec(· |Instruct r, rt, q)(3)
In this formulation, yclaim={y t1, . . . , y tn}repre-
sents the set of knowledge claims generated by the
claim recognition model Mrec, where n∈N+de-
notes the number of generated claims, and Instruct r
is a prompt template designed to guide the model
to focus on factual claims relevant to theq.
3.2.2 Localized Knowledge Coherence
Correction via Retrieval
At the In-Process Knowledge Claim Recognition
process, we extract a set of factual claims yclaim
t
related to the current reasoning chain unit rt,
which helps identify specific locations where fac-
tual knowledge errors may exist. Subsequently,
the Localized Knowledge Coherence Correction
via Retrieval process focuses on correcting the rea-
soning chain unit rtat these identified locations
by retrieving external knowledge, effectively align-
ing internal reasoning chain content with external
knowledge and correcting parametric knowledge
errors.
Specifically, we first take the union of the knowl-
edge claim set yclaim
t and the original question qto
construct the query set Qt={q} ∪yclaim
t. Each
query qi∈ Q tserves as an atomic retrieval query,
triggering an independent retrieval that returns the
top-krelevant documents. Collecting the results
from all queries forms the initial candidate docu-
ment set Draw
t. To avoid redundancy, we further
perform deduplication on the initial candidate doc-
ument set Draw
tto obtain the refined retrieval docu-
ment set Dt={d t1, dt2, . . . , d tn}, where n∈N+denotes the number of documents after deduplica-
tion. The process for acquiring external knowledge
can be represented in a formal notation as follows:
Dt=[
qi∈QtRetriever(q i)(4)
Then, we jointly feed the retrieved document
setDtand the current reasoning-chain unit rtinto
the knowledge correction model Mcor. The model
identifies and extracts knowledge relevant to rtand
uses it to assess and verify the factual validity of
the current reasoning-chain unit rt, ensuring knowl-
edge alignment between the reasoning chain and
the knowledge base. Conditioned on the current
reasoning-chain unit rtand the retrieved document
setDtafter deduplication, the corrected reasoning-
chain unit r′
tis sampled from the conditional distri-
bution parameterized by the knowledge correction
modelM cor:
r′
t∼ M cor(· |Instruct c, rt,Dt)(5)
where Instruct cis an instructional prompt template
designed to guide the model in performing token-
level corrections with minimal cost while ensuring
the reasoning structure remains intact. Here, we
introduce two types of preference guidance to steer
the model, such that when all facts in rtare cor-
rect or the retrieved document set Dtis irrelevant,
the reasoning-chain unit remains unchanged. If rt
contains erroneous knowledge, the model applies
targeted token-level corrections with minimal dis-
ruption to preserve the structural integrity of the
reasoning chain, ultimately producing the corrected
reasoning-chain unit r′
t. We define ˜rt∈ {r t, r′
t}
to uniformly represent the current reasoning chain
that admits two possible content states.
After factual errors are corrected, the reasoning
model continues the inference process based on
the updated reasoning chain, enabling the model to
proceed along a correct reasoning path and prevent-
ing the accumulation of errors. Through multiple
iterations of reasoning and factual correction, the
complete reasoning chain is ultimately denoted as:
R= ˜r 1⊕˜r 2⊕...⊕˜r t (6)
from which the final answer ais derived. During
reasoning, this real-time factual verification and
correction mechanism leverages external knowl-
edge to precisely identify and rectify parametric
knowledge errors in the RLM, guiding the model
toward accurate and efficient reasoning.
4

3.2.3 Optimization for Knowledge
Recognition and Correction
Although the two processes above can effectively
identify and correct factual errors during reasoning,
we also observe that the claim recognition model
Mrecand the knowledge correction model Mcor
suffer from issues in a limited number of cases,
such as low-quality factual claims, failure to per-
form correct corrections, and redundant outputs.
Therefore, our goals are twofold: 1) to improve the
factual claim recognition capability of the Mrec,
and 2) to enhance the precision of the Mcorin
performing knowledge corrections. To this end,
we adopt DPO (Rafailov et al., 2023) as the opti-
mization algorithm to guide the model in assigning
higher likelihood to high-quality, accurate, and con-
cise outputs, without requiring an explicit reward
model. Accordingly, we adopt a two-stage strategy
to construct a small-scale, high-quality preference
dataset DDPO(Appendix A), with the aim of in-
ducing preferences for high-quality factual claim
recognition and fine-grained knowledge correction.
DPO TrainingUsing the constructed training
data, we optimize the model Mθ
RCthat serves both
the claim recognition and knowledge correction,
to learn a unified policy θthat assigns higher like-
lihood to the preferred output y+than to its less
preferred counterpart y−, conditioned on the same
input x. Our training objective is formulated as
follows:
LDPO(Mθ
RC;Mref
RC) =−E {x,y+,y−}∼D[logσ
[βlogMθ
RC(y+|x)
Mref
RC(y+|x)−βlogMθ
RC(y−|x)
Mref
RC(y−|x)]](7)
where Mθ
RCandMref
RCrepresent the trainable
model and the reference model, respectively.
4 Experiments
4.1 Datasets & Metrics
Multi-hop QA Datasets(1)HotpotQA(Yang
et al., 2018) requires retrieving and reasoning over
multiple Wikipedia documents with distracting
information to answer. (2)2WikiMultiHopQA
(2WikiMQA)(Ho et al., 2020) is a multi-hop
QA dataset where each question requires reason-
ing over entity-centric hyperlinks across multiple
Wikipedia pages to discover bridging entities. (3)
MuSiQue(Trivedi et al., 2022) is a challenging
dataset characterized by its compositional, intercon-
nected questions that necessitate sequential reason-ing without relying on isolated single-hop clues. (4)
IIRC(Ferguson et al., 2020) builds on Wikipedia
articles where questions require identifying rele-
vant text snippets from linked contexts to answer
questions given an incomplete context. For the
test samples and retrieval corpus, we follow IRCoT
(Trivedi et al., 2023).
Short-form QA Datasetfocuses on retrieving
direct answers to factoid questions from a single
context. We chooseSimpleQA(Wei et al., 2024),
a large-scale benchmark consisting of short, fact-
seeking questions. We randomly select 500 sam-
ples from the test split and use the KILT (Petroni
et al., 2021) knowledge source as the corpus.
MetricsWe use F1-Score (f1) and Exact Match
(em) (Jiang et al., 2023) to evaluate the correctness
of the final answer.
4.2 Baselines
We compare our method against the following base-
line approaches: (1) Direct Reasoning: the RLM
relies solely on its parametric knowledge to an-
swer questions without retrieving any external in-
formation. (2) Vanilla RAG: a single-step retrieval
fetches the top- kmost relevant documents, which
are then provided alongside the question as context
for the LLM to generate an answer. (3) Multi-Step
RAG: the RAG system gathers relevant informa-
tion through multiple iterative retrieval steps. We
select RAT (Li et al., 2024) as our representative
baseline. (4) Adaptive RAG: the RAG framework
dynamically decides the retrieval and generation
strategy to optimize both response quality and ef-
ficiency. We primarily compare against three es-
tablished baselines: FLARE (Jiang et al., 2023),
Self-RAG (Asai et al., 2024), and ReAct (Yao
et al., 2022). Additionally, we also compare with
Search-o1 (Li et al., 2025), which actively retrieves
external knowledge during the reasoning process
and seamlessly integrates refined, concise informa-
tion extracted from lengthy retrieval results into the
reasoning chain.
4.3 Implementation Details
We primarily employ QwQ-32B (Team, 2025) as
the reasoning model, while also evaluating various
RLMs of Qwen3-8B (Yang et al., 2025), Qwen3-
32B (Yang et al., 2025) and DeepSeek-R1-Distill-
Llama-70B (Guo et al., 2025) for comparative
analysis. For MrecandMcor, we use the same un-
derlying model and select four LLMs: Qwen3-8B
(Yang et al., 2025), Qwen2.5-14B-Instruct (Yang
5

MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA Avg.
f1 em f1 em f1 em f1 em f1 em
QwQ-32B
Direct Reasoning 38.4 29.0 34.6 28.4 18.5 7.4 24.8 21.4 10.5 4.6 21.8
Vanilla RAG 52.7 42.2 46.4 42.6 19.3 10.0 25.0 22.0 31.4 24.2 31.6
ReAct (2022) 48.2 34.6 45.9 33.0 22.3 10.0 21.9 16.2 30.1 22.2 28.4
FLARE (2023) 43.5 32.2 46.6 38.4 18.4 9.2 12.3 9.6 24.1 17.4 25.2
Self-RAG (2024) 49.7 39.9 40.0 36.2 20.0 10.6 25.1 21.3 31.1 24.8 29.9
RAT (2024) 51.5 39.0 47.2 38.6 24.1 13.2 21.7 17.0 30.3 22.2 30.5
Search-o1 (2025) 62.0 49.2 71.4 60.4 33.3 20.7 29.2 25.0 35.4 27.4 41.5
CheckRLM66.3 52.6 73.4 62.0 39.6 27.2 33.1 29.0 40.0 30.4 45.4
Qwen3-32B
Direct Reasoning 36.4 27.6 34.0 29.4 16.6 6.6 23.0 20.2 9.5 2.4 20.6
Vanilla RAG 51.2 41.0 44.5 41.8 19.7 10.4 22.5 19.8 29.3 22.8 30.3
CheckRLM64.2 52.0 70.8 60.8 35.6 25.4 31.6 27.8 39.1 30.6 43.8
Qwen3-8B
Direct Reasoning 29.3 22.2 31.1 26.6 12.9 5.2 21.0 18.2 7.7 2.8 17.7
Vanilla RAG 44.5 35.2 40.9 38.4 13.8 6.0 19.8 16.8 29.4 23.0 26.8
CheckRLM61.9 49.6 69.7 58.6 35.0 24.0 29.7 25.4 35.7 28.6 41.8
Table 1:Results (%) of overall performance.Our recognition and correction model is Llama-3.3-70B-Instruct.
The highest scores are emphasized inbold.
et al., 2024), Qwen2.5-32B-Instruct (Yang et al.,
2024), and Llama-3.3-70B-Instruct (Dubey et al.,
2024). In terms of inference hyperparameters, we
set the maximum context length to 16,384 tokens
and retain default settings for all other hyperparam-
eters. For the retriever, we utilize both the sparse re-
triever BM25 and the dense retriever bge-large-en-
v1.5 (Xiao et al., 2023) on multi-hop QA datasets,
while only the dense retriever bge-large-en-v1.5
is used on the short-form QA dataset. Across all
experiments, we fix top- kto 3 and set the maxi-
mum number of recognition-and-correction steps
to 10. All experiments are conducted using the
vLLM (Kwon et al., 2023) on a cluster of 8×A800
GPUs. For DPO training, we use a batch size of
8, a learning rate of 5e-7, βof 0.1, and train for 1
epoch.
5 Results and Analysis
5.1 Overall Performance
Vanilla RAG underperforms on complex tasks;
by contrast, CheckRLM demonstrates a sub-
stantial performance improvement.As shown in
Table 1, Vanilla RAG improves upon Direct Rea-soning by introducing external knowledge docu-
ments in a one-time retrieval step, thus partially al-
leviating the limitation that Direct Reasoning relies
entirely on internal knowledge during the reason-
ing process. Empirical results indicate that Vanilla
RAG achieves reasonable performance on short-
form QA tasks, as these tasks typically require only
a small number of relevant snippets that are most
relevant to the original question; however, these ob-
served performance improvements struggle to gen-
eralize to complex multi-hop QA tasks. In contrast,
CheckRLM consistently outperforms Vanilla RAG
across all datasets. In particular, it achieves espe-
cially substantial gains on complex multi-hop QA
tasks, yielding improvements over Vanilla RAG
across all three reasoning backbones. These perfor-
mance improvements indicate that CheckRLM is
capable of performing effective knowledge correc-
tion by repeatedly and timely incorporating exter-
nal knowledge to precisely correct factual errors in
the reasoning chain.
CheckRLM surpasses a wide range of main-
stream baselines across both multi-hop and
short-form QA benchmarks.As shown in Ta-
ble 1, in comparison with recent iterative re-
6

Method Hot. 2Wiki. Simp.
QwQ-32B
Vanilla RAG 52.7 46.4 31.4
Inference Only 61.6 65.5 36.0
DPO Training63.2 71.2 36.9
Qwen3-32B
Vanilla RAG 51.2 44.5 29.3
Inference Only 57.0 62.9 34.2
DPO Training61.3 65.3 36.1
Table 2:DPO Training Results (%).We conduct ex-
periments on the HotpotQA, 2WikiMQA and SimpleQA
dataset, using Qwen2.5-14B-Instruct as the recognition
and correction model and f1 as the metric.
trieval–generation and coupled retrieval–reasoning
methods, CheckRLM achieves the highest scores
across all evaluated datasets, including challenging
multi-hop and short-form QA benchmarks. The
gains are especially pronounced on complex multi-
hop reasoning, and are most evident on the more
challenging MuSiQue dataset, where CheckRLM
outperforms the best-performing baseline by 6.3
f1 points. This advantage can be attributed to a
key design difference. While these prior methods
enhance reasoning by expanding the interaction
between retrieval and reasoning or by increasing
search depth, they generally lack explicit mecha-
nisms to correct factual inconsistencies within the
reasoning process itself. In contrast, CheckRLM in-
troduces an in-process knowledge checking mecha-
nism that continuously monitors and corrects fac-
tual errors during reasoning, preventing erroneous
knowledge from propagating across subsequent rea-
soning steps. This design is especially critical for
multi-hop reasoning, where early-stage errors can
severely affect downstream inference.
5.2 Effectiveness of Knowledge Recognition
and Correction Optimization
The DPO training process in CheckRLM enhances
the model’s ability to correct factual errors in the
reasoning chain by jointly optimizing knowledge
claim recognition and knowledge coherence cor-
rection. Based on this design, we evaluate the per-
formance gains brought by this joint optimization
strategy to validate the effectiveness of our training
procedure. We conduct DPO training on the small
backbone Qwen2.5-14B-Instruct, and the results
are summarized in Table 2. Compared to inference-
only reasoning, DPO yields a 5.7% improvementon the in-domain dataset 2WikiMQA, while im-
proving performance on out-of-domain multi-hop
QA benchmarks such as HotpotQA, as well as the
short-form QA dataset SimpleQA.
Furthermore, we perform extensive experiments
across different reasoning backbones and training
data compositions, where CheckRLM also achieves
strong performance on all evaluated datasets (see
Appendix C.2 for complete results). These re-
sults demonstrate that our DPO data sampling and
training pipeline effectively enhances CheckRLM
across diverse reasoning models and datasets, ex-
hibiting strong generality and robustness.
66.3
58.873.4
60.2
39.6
31.533.132.240.038.3
HotpotQA
2WikiMQA
MusiQueIIRCSimpleQA80
70
60
50
40
30
20
10
0F1-ScoreIn-reasoning Check
Post-reasoning Check
(a) Llama-3.3-70B-Instruct
61.6
46.765.5
47.6
33.3
24.330.7
28.336.0
26.3
HotpotQA
2WikiMQA
MusiQueIIRCSimpleQA70
60
50
40
30
20
10
0F1-ScoreIn-reasoning Check
Post-reasoning Check (b) Qwen2.5-14B-Instruct
Figure 3:Post-reasoning Check vs. In-reasoning
Check.We use QwQ-32B as the reasoning model,
Llama-3.3-70B-Instruct and Qwen2.5-14B-Instruct as
the recognition and correction model. The experiments
are conducted in the 2WikiMQA dataset and use f1 as
the metric. More results are in Appendix C.4.
5.3 Effectiveness of In-reasoning Knowledge
Check in CheckRLM
CheckRLM introduces an intermediate interven-
tion strategy that integrates knowledge claim recog-
nition with consistency correction for in-reasoning
check. Benchmarked against post-reasoning check,
CheckRLM consistently outperforms across all
datasets (Figure 3). The results confirm the su-
periority of in-reasoning intervention.
RLMs often introduce errors early in inference,
which then propagate and amplify through subse-
quent steps. While post-reasoning check accesses
the complete chain, its late correction can only
address a subset of errors and fails to rectify al-
ready flawed reasoning trajectories. In contrast,
CheckRLM intervenes dynamically during reason-
ing—correcting factual errors before they cascade
and ensuring that each step proceeds under accurate
knowledge constraints. This step-wise correction
substantially improves the stability and reliability
of the overall reasoning process.
7

800
1000
1200
1400
1600
1800
2000
# Token (↓)
2.0
2.5
3.0
3.5
4.0
Time (s) (↓)
0
10
20
30
40
50
60
70
F1 (%) (↑)
Direct Reasoning
(T=910.9, τ=1.8, F1=34.6)
Vanilla RAG
(T=1684.0, τ=4.0, F1=46.4)
Search-o1
(T=2008.5, τ=3.9, F1=71.4)
Post-reasoning Check
(T=1487.2, τ=2.9, F1=60.2)
In-reasoning Check
(T=1364.1, τ=3.3, F1=73.4)
Figure 4:Time, token consumption and performance
of different methods.We use QwQ-32B as the rea-
soning model, Llama-3.3-70B-Instruct as the recogni-
tion and correction model. The experiments are con-
ducted in the 2WikiMQA dataset. More results are in
Appendix C.7.
5.4 Analysis of Cost-Efficient Reasoning
In this experiment, we conduct a three-dimensional
analysis that jointly considers token consumption,
inference time, and task performance, aiming to
provide a comprehensive evaluation of efficiency
and performance.
As shown in Figure 4, Direct Reasoning has the
lowest cost but limited performance. In contrast,
Vanilla RAG and Search-o1 improve performance
at substantially higher token and time costs. No-
tably, the in-reasoning check in CheckRLM attains
a more favorable outcome across both performance
and cost dimensions. Compared with Vanilla RAG
and Search-o1, it significantly reduces token con-
sumption and inference time while simultaneously
achieving the best performance. This suggests that
timely knowledge correction during reasoning en-
ables the model to avoid unnecessary reasoning
steps and redundant retrieval caused by error accu-
mulation, improving efficiency without sacrificing
accuracy.
Furthermore, the comparison between the in-
reasoning check in CheckRLM and the post-
reasoning check highlights the critical role of cor-
rection timing. Although post-reasoning check re-
duces computational cost relative to retrieval-heavy
methods, it remains less effective than in-process
check, as delayed correction cannot prevent erro-
neous knowledge from propagating throughout the
reasoning chain. Overall, CheckRLM achieves the
best performance while substantially reducing both
inference time and token consumption, demonstrat-
Golden Reasoning Step 1 (219)
Golden Reasoning Step 2 (219)
Golden Reasoning Step 3 (63)No Checking (43)
Checking Step 1 (218)
Checking Step 2 (122)
Checking Step 3 (72)
Checking Step 4 (22)
Checking Step 5 (11)
Checking Step 6 (5)
Checking Step 7 (5)
Checking Step 8 (1)
Checking Step 9 (1)
Checking Step 10 (1)(a) 2WikiMQA
Golden Reasoning Step 1 (109)
Golden Reasoning Step 2 (108)
Golden Reasoning Step 3 (38)
Golden Reasoning Step 4 (6)No Checking (46)
Checking Step 1 (74)
Checking Step 2 (59)
Checking Step 3 (42)
Checking Step 4 (21)
Checking Step 5 (12)
Checking Step 6 (4)
Checking Step 8 (2)
Checking Step 7 (1) (b) MuSiQue
Figure 5:Correspondence between Golden Reason-
ing Step and Checking Step.We use QwQ-32B as the
reasoning model, Llama-3.3-70B-Instruct as the recog-
nition and correction model. The experiments are con-
ducted in the 2WikiMQA and MuSiQue datasets.
ing clear superiority over all competing methods in
terms of both efficiency and effectiveness.
5.5 Characteristics of Correction Behavior
This experiment aims to analyze when factual er-
rors in multi-hop reasoning are detected and cor-
rected by our method using the GPT-4o model,
relative to the golden reasoning steps.
As depicted in Figure 5, Golden Reasoning Step
kdenotes the k-th hop in the golden reasoning
chain, while Checking Step jindicates that our
method successfully identifies and corrects the er-
ror at its j-th checking iteration. The observed
trend shows that corrections predominantly occur
at early checking steps (Steps 1–2), indicating ef-
fective early interception of factual errors rather
than their propagation through the reasoning chain.
Despite its higher reasoning complexity,
MuSiQue still exhibits a clear tendency toward
early correction. While Checking Step 1 remains
dominant, errors arising from later oracle steps
are primarily resolved at intermediate stages
(Steps 2–4), with deep checking being rare. This
indicates that the method consistently corrects
factual errors at early stages, even under more
challenging multi-hop reasoning tasks. These
results demonstrate that our method performs
early, targeted, and efficient error correction, rather
than relying on deep search or repeated post-hoc
refinement.
6 Conclusion
In conclusion, we presentCheckRLM, a frame-
work that tackles factual errors in RLMs for
knowledge-intensive tasks. CheckRLM enhances
reasoning reliability by proactively recognizing
and correcting factual inconsistencies during in-
8

ference via RAG. It extracts factual claims, locates
errors against external knowledge, and performs
precise, low-cost corrections. Experiments demon-
strate that CheckRLM significantly outperforms
existing baselines, mitigates error accumulation in
long-horizon reasoning, and reduces temporal and
computational costs, advancing towards more ac-
curate and efficient knowledge-reliant reasoning.
Limitations
Although CheckRLM has successfully achieved
real-time intervention of external knowledge for
reasoning verification, this study still has certain
limitations, mainly reflected in two aspects. First,
the current mechanism focuses on utilizing pure
textual knowledge sources and has not yet been
extended to the collaborative verification of mul-
timodal information. Second, the framework is
currently optimized only for a single knowledge
base and does not explore the complex coordination
and conflict resolution mechanisms across multiple
heterogeneous knowledge bases. Future research
could delve into the directions of multimodal fusion
and dynamic integration of multi-source knowl-
edge to build a more general and robust reasoning
verification framework.
Acknowledgments
This work was supported by the National Natural
Science Foundation of China (No. 62276026) and
the Fundamental Research Funds for the Central
Universities (No. 2253500001). This work is sup-
ported by the AI9Stars community.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng,
Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang
Zhou, Te Gao, and Wanxiang Che. 2025. Towards
reasoning era: A survey of long chain-of-thought
for reasoning large language models.arXiv preprint
arXiv:2503.09567.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv e-prints, pages arXiv–2407.
James Ferguson, Matt Gardner, Hannaneh Hajishirzi,
Tushar Khot, and Pradeep Dasigi. 2020. IIRC: Adataset of incomplete information reading compre-
hension questions. InProceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 1137–1147, Online. As-
sociation for Computational Linguistics.
Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei
Liu. 2024. GPTScore: Evaluate as you desire. In
Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 6556–6576, Mexico
City, Mexico. Association for Computational Lin-
guistics.
Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming
Xue, and Haofen Wang. 2025. Synergizing rag and
reasoning: A systematic review.arXiv preprint
arXiv:2504.15909.
Xinyan Guan, Jiali Zeng, Fandong Meng, Chunlei Xin,
Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and
Jie Zhou. 2026. DeepRAG: Thinking to retrieve step
by step for large language models. InThe Fourteenth
International Conference on Learning Representa-
tions.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang,
Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai
Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao,
Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 oth-
ers. 2025. DeepSeek-R1 incentivizes reasoning
in LLMs through reinforcement learning.Nature,
645(8081):633–638. TLDR: A new artificial intelli-
gence model, DeepSeek-R1, is introduced, demon-
strating that the reasoning abilities of large lan-
guage models can be incentivized through pure re-
inforcement learning, removing the need for human-
annotated demonstrations.
Yancheng He, Shilong Li, Jiaheng Liu, Weixun Wang,
Xingyuan Bu, Ge Zhang, Zy Peng, Zhaoxiang Zhang,
Zhicheng Zheng, Wenbo Su, and 1 others. 2025. Can
large language models detect errors in long chain-
of-thought reasoning? InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 18468–
18489.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Qisheng Hu, Quanyu Long, and Wenya Wang.
2025. Coordinating search-informed reasoning and
reasoning-guided search in claim verification.arXiv
preprint arXiv:2506.07528.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
9

Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-RAG: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), pages 7036–7050, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-
taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-
guage models are zero-shot reasoners.Advances in
neural information processing systems, 35:22199–
22213.
Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su,
John D Co-Reyes, Avi Singh, Kate Baumli, Shariq
Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang,
Kay McKinney, Disha Shrivastava, Cosmin Paduraru,
George Tucker, Doina Precup, Feryal Behbahani, and
Aleksandra Faust. 2025. Training language models to
self-correct via reinforcement learning. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. InProceedings of the 29th
symposium on operating systems principles, pages
611–626.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yu-
jia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5420–5438, Suzhou, China.
Association for Computational Linguistics.
Xinzhe Li. 2025. A survey on llm test-time com-
pute via search: Tasks, llm profiling, search algo-
rithms, and relevant frameworks.arXiv preprint
arXiv:2501.10069.Yushen Li, Jinpeng Wang, Tao Dai, Jieming Zhu, Jun
Yuan, Rui Zhang, and Shu-Tao Xia. 2024. Rat:
Retrieval-augmented transformer for click-through
rate prediction. InCompanion Proceedings of the
ACM Web Conference 2024, pages 867–870.
Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang,
Mingu Lee, Roland Memisevic, and Hao Su. 2023.
Deductive verification of chain-of-thought reasoning.
Advances in Neural Information Processing Systems,
36:36407–36433.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2511–2522, Singapore. Association for Com-
putational Linguistics.
OpenAI. 2024. Learning to reason with llms, september
2024.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
and 1 others. 2021. Kilt: a benchmark for knowledge
intensive language tasks. InProceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 2523–2544.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics, 11:1316–1331.
Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan
Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei
Huang, and Jingren Zhou. 2025. Zerosearch: Incen-
tivize the search capability of llms without searching.
arXiv preprint arXiv:2505.04588.
Qwen Team. 2025. Qwq-32b: Embracing the power of
reinforcement learning.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
10

Gladys Tyen, Hassan Mansoor, Victor C ˘arbune,
Yuanzhu Peter Chen, and Tony Mak. 2024. Llms
cannot find reasoning errors, but can correct them
given the error location. InFindings of the Associa-
tion for Computational Linguistics: ACL 2024, pages
13894–13908.
Ruobing Wang, Qingfei Zhao, Yukun Yan, Daren Zha,
Yuxuan Chen, Shi Yu, Zhenghao Liu, Yixuan Wang,
Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong
Sun. 2025. DeepNote: Note-centric deep retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 19688–19715, Suzhou, China. Association for
Computational Linguistics.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, and 1 oth-
ers. 2024. Searching for best practices in retrieval-
augmented generation. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 17716–17736.
Jason Wei, Nguyen Karina, Hyung Won Chung,
Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John
Schulman, and William Fedus. 2024. Measuring
short-form factuality in large language models.arXiv
preprint arXiv:2411.04368.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He,
Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao.
2023. Large language models are better reasoners
with self-verification. InFindings of the Association
for Computational Linguistics: EMNLP 2023, pages
2550–2575.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.Preprint,
arXiv:2309.07597.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang,
Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing
Song, Dengyu Wang, Minjia Zhang, and 1 others.
2025. Rag-gym: Optimizing reasoning and search
agents with process supervision.arXiv preprint
arXiv:2502.13957.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,Jingren Zhou, Junyang Lin, Kai Dang, and 22 oth-
ers. 2024. Qwen2.5 technical report.arXiv preprint
arXiv:2412.15115.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2023. Tree of thoughts: Deliberate problem solving
with large language models.Advances in neural
information processing systems, 36:11809–11822.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models. InThe eleventh international conference on
learning representations.
Zijun Yao, Yantao Liu, Yanxu Chen, Jianhui Chen, Jun-
feng Fang, Lei Hou, Juanzi Li, and Tat-Seng Chua.
2025. Are reasoning models more prone to halluci-
nation?arXiv preprint arXiv:2505.23646.
Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong
Pasupat, Jure Leskovec, Percy Liang, Ed H. Chi,
and Denny Zhou. 2024. Large language models as
analogical reasoners. InInternational Conference
on Learning Representations, volume 2024, pages
17019–17045.
Duzhen Zhang, Zhong-Zhi Li, Ming-Liang Zhang, Ji-
axin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu,
Junhao Zheng, Xiuyi Chen, Yingying Zhang, Fei Yin,
Jiahua Dong, Zhijiang Guo, Le Song, and Cheng-Lin
Liu. 2026. From System 1 to System 2: A Survey
of Reasoning Large Language Models .IEEE Trans-
actions on Pattern Analysis & Machine Intelligence,
48(03):3335–3354.
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex
Smola. 2023. Automatic chain of thought prompting
in large language models. InThe Eleventh Interna-
tional Conference on Learning Representations.
Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha,
Shicheng Tan, Yuxiao Dong, and Jie Tang. 2024.
LongRAG: A dual-perspective retrieval-augmented
generation paradigm for long-context question an-
swering. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 22600–22632, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha,
and Limin Liu. 2025. R-search: Empowering llm
reasoning with search via multi-reward reinforcement
learning.arXiv preprint arXiv:2506.04185.
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei,
Nathan Scales, Xuezhi Wang, Dale Schuurmans,
11

Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H.
Chi. 2023. Least-to-most prompting enables com-
plex reasoning in large language models. InThe
Eleventh International Conference on Learning Rep-
resentations.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and
Maosong Sun. 2025. RAGEval: Scenario specific
RAG evaluation dataset generation framework. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 8520–8544, Vienna, Austria.
Association for Computational Linguistics.
12

Appendix
ADetails of DPO Training Data Construc-
tion 13
A.1 DPO Training Data for Knowledge
Claim Recognition . . . . . . . . 13
A.2 DPO Training Data for Knowledge
Coherence Correction . . . . . . . 13
B Prompt Details 13
B.1 Direct Reasoning Prompt . . . . . 14
B.2 Vanilla RAG Prompt . . . . . . . 14
B.3 CheckRLM Prompt . . . . . . . . 14
B.4 Correction Behavior Prompt . . . 14
B.5 DPO Evaluation Prompt . . . . . 14
C Additional Experimental Results 15
C.1 Overall Performance of CheckRLM 15
C.2 Knowledge Recognition and Cor-
rection Optimization . . . . . . . 15
C.3 Constrained Ablation Study . . . . 15
C.4 Post-reasoning Check vs.
In-reasoning Check . . . . . . . . 15
C.5 Effect of the Retrieved Summary
Injection on Reasoning . . . . . . 15
C.6 Effect of Different Retrievers . . . 16
C.7 Cost-Efficient Reasoning Results . 16
C.8 Correction Step Distribution . . . 16
D Case Study 16
A Details of DPO Training Data
Construction
We randomly select 2500 samples from the
2WikiMQA training set to build our DPO training
dataset DDPO, which comprises two subsets: train-
ing data for knowledge claim recognition DKCR
and training data for knowledge coherence correc-
tionDKCC. The statistics of all DPO training data
are presented in Table 3.
A.1 DPO Training Data for Knowledge Claim
Recognition
Observing that knowledge corrections primarily
occur in the initial steps of the reasoning process,
we only sample data from the first three reason-
ing chain units to efficiently obtain high-quality
data. For a given sample, we input the original
question q, the current reasoning chain unit rt,
and the recognition instruction template Instruct r
intoMrecto sample a set of knowledge claims
yclaim
t. To balance diversity and accuracy in theDKCRDKCCDDPO
# Sample2351 2960 5311
Table 3: Statistics of DPO Training Data.
generated samples, we set different decoding hy-
perparameters, namely temperature and top_p , to
specific values of 0.1, 0.5, and 0.9. They control
the generation of a candidate answer set Yclaim
t=
{yclaim
t1, yclaim
t2, ..., yclaim
tn}, where n∈N+denotes
the number of candidate answers. Recent studies
have demonstrated that annotations from advanced
models can achieve high alignment with human
judgments (Liu et al., 2023; Fu et al., 2024). We
utilize GPT-4o-mini to annotate positive and nega-
tive examples yclaim(+)
t andyclaim(-)
t within the candi-
date answer set. The resulting training data DKCR
can be represented as{x t, yclaim(+)
t , yclaim(-)
t}.
A.2 DPO Training Data for Knowledge
Coherence Correction
For training data DKCC, the settings for decod-
ing hyperparameters remain the same as above.
Since the quality of retrieved documents is pos-
itively correlated with the quality of knowledge
claims, we use the positive example yclaim(+)
t for
document retrieval. The retrieved document set
D+
t, the current reasoning chain unit rt, and the
correction instruction template Instruct care then
fed into Mcorto generate a candidate answer set
˜Rt={˜r t1,˜rt2, ...,˜r tn}, where n∈N+is the num-
ber of candidate answers. Similarly, we employ
GPT-4o-mini to annotate positive and negative ex-
amples ˜r+
tand˜r−
tand construct the DPO training
data for knowledge coherence correction, denoted
as{xt,˜r+
t,˜r−
t}. We then use the positive example
˜r+
tto form the reasoning chain Rof the RLM to
generate the next reasoning chain unitr t+1.
B Prompt Details
This section presents the prompt templates em-
ployed for both inference and DPO training. For
inference, we provide prompts for Direct Reason-
ing, Vanilla RAG, knowledge claim recognition,
knowledge coherence correction and correction be-
havior analysis. In particular, the reasoning prompt
used in CheckRLM is identical to the direct reason-
ing. For DPO training, we supply corresponding
prompts for evaluating knowledge claim recogni-
tion and knowledge coherence correction.
13

B.1 Direct Reasoning Prompt
Direct Reasoning Prompt
Please reason step by step, and put your final answer within
\boxed{}. Only give me the answer and do not output any
other words.
Question: {question}
B.2 Vanilla RAG Prompt
Vanilla RAG Prompt
Answer the question based on the given passages. Please
reason step by step, and put your final answer within
\boxed{}. Only give me the answer and do not output
any other words.
The following are given passages:
{refs}
Question: {question}
B.3 CheckRLM Prompt
Knowledge Claim Recognition Prompt
Task: Extract only explicit factual claims from the Reason-
ing Process that are helpful to answering the Question. The
Reasoning Process contains factual information useful for
addressing the Question. Your task is to identify, extract,
and summarize these relevant facts into clear factual claims.
The factual claims must be based on the Reasoning Process;
NEVER add any new content by yourself. Try to avoid two
factual claims that have similar meaning. Output is a list of
strings. The output format is as follows:
["factual claim 1", "factual claim 2", ...]
Question: {question}
Reasoning Process: {reasoning}
Factual Claim List:
Knowledge Coherence Correction Prompt
Task: Based on the retrieved documents, check and correct
factual errors in the Reasoning Process. You will receive
several retrieved documents and the Reasoning Process.
The retrieved documents may contain information relevant
to the Reasoning Process. Your task is to filter the relevant
information from the retrieved documents and verify the
correctness of the factual information in the Reasoning
Process. Follow these rules:
1. If the factual information in the Reasoning Process is
correct, only output the original Reasoning Process.
2. If the factual information in the Reasoning Process is
incorrect, make the minimal necessary corrections to fix the
error without altering the structure or flow of the Reasoning
Process.
3. If the retrieved documents do not contain any relevant
information, only output the original Reasoning Process.
NEVER add any supplementary information from the re-
trieved documents. Only correct factual errors when nec-
essary. Only give me your modified reasoning process and
do not output any other words.
Retrieved documents: {refs}
Reasoning Process: {reasoning}
Provide your modified reasoning process:B.4 Correction Behavior Prompt
Correction Behavior Prompt
Task: Label Effective Correction Steps
You are a data annotation expert tasked with identifying
which refinement steps successfully introduce key content
from the correct reasoning chain.
Requirements:
1. Examine the “original think” and “refine think” content
of each step one by one.
2. If “original think” content contains the key step from the
correct reasoning chain, output the correct reasoning step
number and -1.
3. If the “refine think” content contains any key step from
the correct reasoning chain, output the correct reasoning
step number and the step number of the modified content.
4. If multiple steps contain the same key step, only record
the earliest occurrence.
Output Rules:
1. Output format: [{“Correct Reasoning Step Number”:
“First Modefied Number”}]
2. Do not include any explanations or additional text. Only
output the list.
Input:
Question: {query}
Correct Reasoning Chains: {golden_reasoning_steps}
Modification Records: {think_refine_detail}
B.5 DPO Evaluation Prompt
Knowledge Claim Recognition Evaluation
Prompt
Task: Evaluate factual claim lists generated based on a
Reasoning Process and Question. Your task is to select
the best factual claim list and the worst factual claim list
according to Quality criteria strictly. Quality criteria refer
to:
1. Relevance: each factual claim in the factual claim list is
relevant to the Question and Reasoning Process, note that if
there is no factual claim in Reasoning Process, just output
[] but not similar sentences such as “there is no related
information” or “the reasoning process does not contain
any factual claim”;
2. Specificity: each factual claim in the factual claim list is
clear, avoiding unclear pronoun;
3. No redundancy: The correct factual claim list format
is just one list of strings, no multiple lists or explanatory
Notes.
Output Rules:
1. JSON format: {“best_id”: <id of the factual claim list
that fully meets all Quality criteria above>, “worst_id”:
<id of the factual claim list that violates the most Quality
criteria>};
2. If there is a minimal quality difference between the best
factual claim list and the worst factual claim list or there is
no worst factual claim, assign the same id to both best_id
and worst_id;
3. If no factual claim list meets all the Quality criteria
above, return an empty object {}
4. Do not include any explanations or additional text.
Question: {question}
Reasoning Process: {reasoning}
Generated factual claim lists: {check_responses}
14

Knowledge Coherence Correction Evalua-
tion Prompt
Task: Evaluate refined reasoning processes based on quality
criteria strictly and identify the best and the worst refined
reasoning processes. Quality criteria refer to:
1. Structural Integrity: strictly preserves the original rea-
soning process structure. Do not add any other supplement
information at the end of the original reasoning process,
even if the supplement information is right. The refined
reasoning process contains negative sentences such as “the
retrieved documents do not contain related information” is
the worst;
2. Precise Corrections: Only modifies factually incorrect
content verified by retrieved documents.
3. Conciseness: No redundant text in the refined reasoning
process, including unnecessary prefixes, explanatory Notes,
and continuation after the original reasoning process, such
as “According to the retrieved documents”.
Output Rules:
1. JSON format: {“best_id”: <id of the refined rea-
soning process that fully meets all Evaluation Criteria>,
“worst_id”: <id of the refined reasoning process that vio-
lates the most Quality criteria>}
2. If there is a minimal quality difference between the best
reasoning process and the worst reasoning process or there
is no worst refined reasoning process, assign the same id to
both best_id and worst_id
3. If no refined reasoning process meets all Evaluation
Criteria above, return an empty object {}
4. Do not include any explanations or additional text.
Retrieved documents: {refs}
Original reasoning process: {think}
Generated refined reasoning processes: {refine_responses}
C Additional Experimental Results
C.1 Overall Performance of CheckRLM
The Table 4 presents overall results of various
RLMs and recognition and correction models. By
combining reasoning models and recognition and
correction models of varying parameter sizes, we
find that CheckRLM consistently achieves bet-
ter performance than both Direct Reasoning and
Vanilla RAG across all datasets, demonstrating the
generality and robustness of CheckRLM.
C.2 Knowledge Recognition and Correction
Optimization
The Table 5 presents all results of CheckRLM with
DPO training. Using the Qwen2.5-14B-Instruct
model as the backbone, we conduct experiments
across different reasoning models and datasets. The
results show that DPO training outperforms Infer-
ence Only in almost all cases, demonstrating the
effectiveness and stability of our constructed train-
ing data pipeline and training.
To further analyze the impact of different data
compositions, we conduct additional experiments
(Table 6) and find that DPO training on the com-bined data DKCR+DKCCoutperforms training on
single-type data, demonstrating the effectiveness
of both DKCR andDKCC. Furthermore, the per-
formance improvement brought by DKCRis more
pronounced in most cases, which can be attributed
to the fact that higher-quality factual claims enable
the correction model to rectify factual errors more
accurately.
C.3 Constrained Ablation Study
In the constrained ablation study, given the high
interdependence between the In-Process Knowl-
edge Claim Recognition and Localized Knowledge
Coherence Correction modules that prevent their
complete removal, we apply the simplest versions
of both modules and conduct experiments across
various models. The experimental results, as shown
in the Table 7, demonstrate the importance of both
modules within the CheckRLM framework.
C.4 Post-reasoning Check vs. In-reasoning
Check
The Table 9 compares the complete results of
Post-reasoning Check and In -reasoning Check un-
der various combinations of RLMs and recognition
and correction models. The In -reasoning Check re-
sults outperform the Post -reasoning Check method
across all experimental settings, demonstrating the
superiority of CheckRLM.
C.5 Effect of the Retrieved Summary
Injection on Reasoning
The goal of CheckRLM correction is to correct er-
rors in the reasoning chain with minimal cost, while
preserving the original distribution of the reasoning
model as much as possible. Our observations indi-
cate that the reasoning model itself exhibits a clear
and coherent problem-solving approach, and that
incorporating additional information may interfere
with its inherent reasoning logic. To investigate this
phenomenon, we design a comparative experiment
that injects summarized retrieved information into
the reasoning chain during error correction.
The experimental results are presented in the
Table 8. We observe that adding more relevant in-
formation alongside factual error correction leads
to a decline in performance. Additionally, we find
that when no relevant documents are retrieved dur-
ing the reasoning process, the correction model
tends to append statements such as “The retrieved
documents do not provide any useful information...”
to the summary. This can mislead the reasoning
15

Step 1
Step 2
Step 3
Step 4
Step 5Step 6Step 7Step 8Step 9Step 10
133
32
34
342313161122Figure 6:Correction step distribution in 2WikiMQA.
model into assuming that the problem is unsolvable,
causing it to halt exploration and directly output
an incorrect answer with no response, thereby re-
ducing system robustness. Therefore, it is more
effective for CheckRLM to minimally revise the
reasoning chain rather than injecting summarized
retrieved information directly into it.
C.6 Effect of Different Retrievers
The Table 10 presents the impact of two differ-
ent retrievers, the sparse retriever BM25 and the
dense retriever bge-large-en-v1.5, on the Check-
RLM framework. It can be observed that bge-large-
en-v1.5 yields better performance than BM25. Fur-
thermore, CheckRLM significantly outperforms
both Direct Reasoning and Vanilla RAG across
both retrieval settings, demonstrating the robust-
ness of our proposed method.
C.7 Cost-Efficient Reasoning Results
Table 11 and Table 12 present more results of Cost-
Efficient Reasoning.
C.8 Correction Step Distribution
We use QwQ-32B as the reasoning model and
Llama-3.3-70B-Instruct as the recognition and cor-
rection model. Figures 6 and 7 present the dis-
tribution of correction steps for CheckRLM on
2WikiMQA and MuSiQue.
Across both datasets, these distributions exhibit
a strong concentration on early checking steps, par-
ticularly Step 2-5, indicating that most factual in-
consistencies are corrected promptly after they are
introduced. This pattern suggests that CheckRLM
effectively intercepts errors at an early stage of
Step 1
Step 2
Step 3
Step 4
Step 5Step 6Step 7Step 8Step 9Step 10
421
13
17
12746618Figure 7:Correction step distribution in MuSiQue.
the reasoning process, preventing erroneous knowl-
edge from being propagated to subsequent rea-
soning steps. Notably, deeper correction steps
occur only rarely, even on MuSiQue, which in-
volves more complex and intertwined multi-hop
reasoning. This observation implies that the check-
ing mechanism does not degenerate into excessive
late-stage corrections, but instead resolves most er-
rors before they accumulate and compound. Over-
all, these results highlight that the effectiveness of
CheckRLM stems from early and timely correc-
tion, which plays a critical role in mitigating errors
accumulated in long-step reasoning.
D Case Study
Tables 13, 14, 15 present three cases based on the
same question, comparing the outputs of Direct
Reasoning, CheckRLM with Post-reasoning Check,
and CheckRLM with In-reasoning Check.
Given the question “When was the director of the
film Slap Her... She’s French born?”, the correct
reasoning process should first identify the direc-
tor’s name and then infer the birth date. RLM Di-
rect Reasoning incorrectly identifies the director’s
name, and this error propagates to the subsequent
step of determining the birth date, leading to an in-
correct final answer. Both Vanilla RAG and Check-
RLM Post -reasoning Check correctly identify the
director’s name; however, they also produce an in-
correct final answer due to the absence of retrieved
document segments with the director’s birth date.
In contrast, CheckRLM via In -reasoning Check
corrects the director’s name from “David Zucker”
to the correct “Melanie Mayron” in Step 1, and
16

it then revises Melanie Mayron’s birth date from
the erroneous “July 21, 1968” to the correct “Oc-
tober 20, 1952” in Step 2, ultimately arriving at
the right answer. This illustrates the necessity of
timely error correction and demonstrates the effec-
tiveness of CheckRLM and its In -reasoning Check
strategy. Moreover, we observe that the correction
model can supplement relevant knowledge perti-
nent to the question sometimes while correcting
erroneous information, significantly streamlining
the reasoning process and enabling more efficient
inference.
17

MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA Avg.
f1 em f1 em f1 em f1 em f1 em
QwQ-32B
Direct Reasoning 38.4 29.0 34.6 28.4 18.5 7.4 24.8 21.4 10.5 4.6 21.8
Vanilla RAG 52.7 42.2 46.4 42.6 19.3 10.0 25.0 22.0 31.4 24.2 31.6
CheckRLM
- Check@Qwen-3-8B 59.9 47.4 59.9 50.6 29.0 17.8 29.0 25.4 33.4 25.2 37.8
- Check@Qwen-2.5-14B 61.6 48.6 65.5 54.8 33.3 21.2 30.7 26.0 36.0 27.3 40.5
- Check@Qwen-2.5-32B 59.4 46.6 66.3 56.6 33.1 19.8 30.9 26.2 34.0 25.2 39.8
- Check@Llama-3.3-70B66.3 52.6 73.4 62.0 39.6 27.2 33.1 29.0 40.0 30.4 45.4
Qwen3-32B
Direct Reasoning 36.4 27.6 34.0 29.4 16.6 6.6 23.0 20.2 9.5 2.4 20.6
Vanilla RAG 51.2 41.0 44.5 41.8 19.7 10.4 22.5 19.8 29.3 22.8 30.3
CheckRLM
- Check@Qwen-2.5-14B 57.0 46.8 62.9 50.2 29.9 20.0 26.5 23.2 34.2 26.2 37.7
- Check@Llama-3.3-70B64.2 52.0 70.8 60.8 35.6 25.4 31.6 27.8 39.1 30.6 43.8
Qwen3-8B
Direct Reasoning 29.3 22.2 31.1 26.6 12.9 5.2 21.0 18.2 7.7 2.8 17.7
Vanilla RAG 44.5 35.2 40.9 38.4 13.8 6.0 19.8 16.8 29.4 23.0 26.8
CheckRLM
- Check@Qwen-2.5-32B 56.3 45.0 60.0 50.0 28.2 18.0 25.0 22.2 33.0 26.2 36.4
- Check@Llama-3.3-70B61.9 49.6 69.7 58.6 35.0 24.0 29.7 25.4 35.7 28.6 41.8
Table 4:Comprehensive benchmark results (%) using different RLMs paired with various recognition and
correction models.
MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA
f1 em f1 em f1 em f1 em f1 em
Reasoning: QwQ-32B & Checking: Qwen-2.5-14B
Vanilla RAG 52.7 42.2 46.4 42.6 19.3 10.0 25.0 22.0 31.4 24.2
Inference Only 61.6 48.6 65.5 54.8 33.3 21.2 30.7 26.0 36.0 27.3
DPO Training63.2 50.0 71.2 58.4 36.2 22.0 32.2 28.0 36.9 28.0
Reasoning: Qwen3-32B & Checking: Qwen-2.5-14B
Vanilla RAG 51.2 41.0 44.5 41.8 19.7 10.4 22.5 19.8 29.3 22.8
Inference Only 57.0 46.8 62.9 50.2 29.9 20.0 26.5 23.2 34.2 26.2
DPO Training61.3 48.6 65.3 54.8 32.7 20.2 31.0 26.6 36.1 28.8
Reasoning: DeepSeek-R1-Distill-Llama-70B & Checking: Qwen-2.5-14B
Vanilla RAG 56.1 43.0 49.4 44.4 25.1 14.6 26.1 22.6 32.0 25.0
Inference Only 60.9 47.4 64.1 53.4 32.320.829.0 23.0 35.927.9
DPO Training62.7 49.2 65.7 54.4 31.720.031.2 26.0 36.827.8
Table 5:Results (%) of performance with DPO training.
18

MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA
f1 em f1 em f1 em f1 em f1 em
Reasoning: QwQ-32B & Checking: Qwen-2.5-14B
DKCROnly 59.9 48.0 67.9 56.2 33.7 21.2 31.2 27.0 36.2 27.4
DKCCOnly 60.3 47.6 68.9 56.0 33.8 20.6 29.9 25.6 34.1 24.0
DKCR+DKCC 63.2 50.0 71.2 58.4 36.2 22.0 32.2 28.0 36.9 28.0
Table 6:Results (%) of performance with different training data compositions.
MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA
f1 em f1 em f1 em f1 em f1 em
Reasoning: Qwen3-8B & Checking: Qwen2.5-32B
CheckRLM 56.3 45.0 60.0 50.0 28.2 18.0 25.0 22.2 33.0 26.2
- w/o Verification 45.3 36.8 54.4 47.0 20.1 10.8 24.2 20.8 24.3 17.0
- w/o Refinement 50.2 39.8 56.0 48.2 26.2 15.2 24.1 20.4 32.3 24.8
Reasoning: QwQ-32B & Checking: Qwen2.5-14B
CheckRLM 61.6 48.6 65.5 54.8 33.3 21.2 30.7 26.0 36.0 27.3
- w/o Verification 52.2 40.2 62.0 54.2 30.2 18.6 29.1 25.2 30.6 21.6
- w/o Refinement 58.0 45.4 64.2 53.0 29.7 18.6 28.2 23.0 33.3 25.8
Reasoning: QwQ-32B & Checking: Llama-3.3-70B
CheckRLM 66.3 52.6 73.4 62.0 39.6 27.2 33.1 29.0 40.0 30.4
- w/o Verification 62.9 50.4 72.6 60.8 39.4 26.0 33.0 28.6 37.7 28.4
- w/o Refinement 64.9 52.4 68.0 56.8 37.5 23.6 32.7 28.4 36.8 29.0
Table 7:All results (%) of constrained ablation study.
MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA
f1 em f1 em f1 em f1 em f1 em
Reasoning: QwQ-32B & Checking: Llama-3.3-70B
CheckRLM 66.3 52.6 73.4 62.039.627.2 33.1 29.0 40.0 30.4
- w/ Summary 66.2 52.0 71.5 59.639.825.2 33.0 28.0 38.5 29.8
Table 8:Comparison (%) of CheckRLM with and without retrieved summary injection.“w/ Summary”
denotes injecting summarized retrieved information into the reasoning chain during correction.
19

MethodMulti-hop Short-form
HotpotQA 2WikiMQA MuSiQue IIRC SimpleQA
f1 em f1 em f1 em f1 em f1 em
Reasoning: Qwen3-8B & Checking: Qwen2.5-32B
Post-reasoning Check 46.8 36.8 49.1 42.6 17.9 8.2 24.3 20.4 30.7 22.6
In-reasoning Check56.3 45.0 60.0 50.0 28.2 18.0 25.0 22.2 33.0 26.2
% improv. 9.5 8.2 10.9 7.4 10.3 9.8 0.7 1.8 2.3 3.6
Reasoning: QwQ-32B & Checking: Qwen2.5-14B
Post-reasoning Check 46.7 35.8 47.6 39.6 24.3 13.0 28.3 23.6 26.3 18.6
In-reasoning Check61.6 48.6 65.5 54.8 33.3 21.2 30.7 26.0 36.0 27.3
% improv. 14.9 12.8 17.9 15.2 9.0 8.2 2.4 2.4 9.7 8.7
Reasoning: QwQ-32B & Checking: Llama-3.3-70B
Post-reasoning Check 58.8 47.0 60.2 52.2 31.5 19.4 32.2 27.6 38.3 29.0
In-reasoning Check66.3 52.6 73.4 62.0 39.6 27.2 33.1 29.0 40.0 30.4
% improv. 7.5 5.6 13.2 9.8 8.1 7.8 0.9 1.4 1.7 1.4
Table 9:Post -reasoning Check vs. In -reasoning Check.We compare the performance of CheckRLM with
Post-reasoning Check and In-reasoning Check.
MethodMulti-hop
HotpotQA 2WikiMQA MuSiQue IIRC
f1 em f1 em f1 em f1 em
QwQ-32B
Direct Reasoning 38.4 29.0 34.6 28.4 18.5 7.4 24.8 21.4
Vanilla RAG 52.7 42.2 46.4 42.6 19.3 10.0 25.0 22.0
CheckRLM
BM2566.3 52.673.4 62.0 39.627.233.1 29.0
bge-large-en-v1.5 65.2 51.874.3 63.8 40.126.035.7 31.0
Qwen3-32B
Direct Reasoning 36.4 27.6 34.0 29.4 16.6 6.6 23.0 20.2
Vanilla RAG 51.2 41.0 44.5 41.8 19.7 10.4 22.5 19.8
CheckRLM
BM25 64.2 52.0 70.8 60.8 35.6 25.4 31.6 27.8
bge-large-en-v1.567.7 54.2 74.2 63.4 39.6 26.0 34.3 29.8
Qwen3-8B
Direct Reasoning 29.3 22.2 31.1 26.6 12.9 5.2 21.0 18.2
Vanilla RAG 44.5 35.2 40.9 38.4 13.8 6.0 19.8 16.8
CheckRLM
BM25 61.9 49.6 69.7 58.6 35.0 24.0 29.7 25.4
bge-large-en-v1.564.9 51.8 74.7 64.6 36.1 24.8 35.2 30.6
Table 10:Results (%) of different retrievers.We present the results of CheckRLM using Llama-3.3-70B-Instruct
as the recognition and correction model.
20

Method # Tokens (↓) Time (s) (↓) f1 (%)(↑)
Direct Reasoning 910.9 1.8 34.6
Vanilla RAG 1684.0 4.0 46.4
Search-o1 2008.5 3.9 71.4
Post-reasoning Check 1487.2 2.9 60.2
In-reasoning Check 1364.1 3.3 73.4
Table 11:Cost-Efficient reasoning results (%).We conduct experiments on the 2WikiMQA dataset, using
QwQ-32B as the reasoning model and Llama-3.3-70B-Instruct as the recognition and correction model.
Method # Tokens (↓) Time (s) (↓) f1 (%)(↑)
Direct Reasoning 910.9 1.8 34.6
Vanilla RAG 1684.0 4.0 46.4
Search-o1 2008.5 3.9 71.4
Post-reasoning Check 1496.7 4.2 47.6
In-reasoning Check 1082.8 2.0 65.5
Table 12:Cost-Efficient reasoning results (%).We conduct experiments on the 2WikiMQA dataset, using
QwQ-32B as the reasoning model and Qwen2.5-14B-Instruct as the recognition and correction model.
21

Query:When was the director of film Slap Her... She’S French born?
Golden Reasoning Path:Slap Her. . . She’s French directorMelanie Mayron-> Melanie
Mayron date of birthOctober 20, 1952
Answer:October 20, 1952.
Direct Reasoning Output:
<think> Okay, I need to find out when the director of the film “Slap Her... She’s French” was
born. First, I should figure out who the director of that movie is. Let me recall... I think the
movie is from the 1980s. The title sounds familiar. Maybe it’s directed by someone likeJim
Abrahams? Wait, Jim Abrahams was part of the team that did the Police Academy movies.
Let me check that.
Alternatively, maybe it’sMel Brooks? No, Mel Brooks did Blazing Saddles and Young
Frankenstein. Wait, Slap Her... She’s French, that sounds like a cult movie. Let me think again.
Oh, right, the director is Jim Abrahams. Wait, but I’m not 100% sure. Let me confirm. The
movie came out in 1975, I believe. The director is Jim Abrahams, but I should make sure.
Alternatively, maybe it’s his brother, David, or one of the other brothers? Wait, Jim Abrahams,
David Zucker, and Jerry Zucker? No, maybe not. Let me think again. The film “Slap Her...
She’s French” is a 1975 comedy. The director is indeed Jim Abrahams. Now, when was Jim
Abrahams born?
Jim Abrahams was born on February 25, 1940.Wait, but I need to confirm that. Let me
think. Jim Abrahams’ birth date. I might be mixing up with his brother David. Wait, David
Zucker was born in 1949. Hmm. Alternatively, maybe Jim Abrahams was born in 1940. Let
me see.
Alternatively, perhaps I can recall that he was part of the team that did the Airplane! movies,
but that’s more the Zucker-Abrahams team. Wait, Jim Abrahams was born in 1940. Let me
check in my mind. If the movie was 1975, and he was, say, in his mid-30s then, that would
make his birth year around 1940. So, I think the correct answer is February 25, 1940. So the
director, Jim Abrahams, was born on that date. I’ll go with that. </think>
Answer: February 25, 1940
Vanilla RAG Output:
<think> Okay, I need to find out when the director of the film “Slap Her... She’s French” was
born. Let’s look at the given passages.
First, the film’s title is mentioned in two passages. Both say the film is directed byMelanie
Mayronand starring Piper Perabo. The title of the film is “Slap Her... She’s French,” which
was released in 2002. The question is about the director’s birth year, so I need to find
information related to Melanie Mayron.
Looking at the third passage about Jane McGregor, it states she was born in 1983 and lists her
roles, including “Slap Her ... She’s French” from 2002. However, Jane McGregor is an actress,
not the director. The director is Melanie Mayron, whose details aren’t provided in any of the
given texts. The passages don’t mention Melanie Mayron’s birth date. The only birth year
given here is Jane McGregor’s, but that’s irrelevant to the director. Since the information isn’t
present in the provided passages,the answer can’t be determined from the given data. The
user’s question might be expecting me to realize that the passages don’t include the director’s
birth year. </think>
Answer: Unknown
Table 13:Case Study I.This case illustrates the outputs of Direct Reasoning via RLM and vanilla RAG.
We use QwQ-32B as the backbone model for reasoning.
22

Query:When was the director of film Slap Her... She’S French born?
Golden Reasoning Path:Slap Her. . . She’s French directorMelanie Mayron-> Melanie
Mayron date of birthOctober 20, 1952
Answer:October 20, 1952.
CheckRLM Post-reasoning Check Output:
Reasoning Output: <think> Okay, I need to find out when the director of the film “Slap
Her... She’s French” was born. First, I should figure out who the director of that movie is. Let
me recall... I think the movie is from the 1980s. The title sounds familiar. Maybe it’s directed
by someone like Jim Abrahams? Wait, Jim Abrahams was part of the team that did the Police
Academy movies. Let me check that.
Alternatively, maybe it’s Mel Brooks? No, Mel Brooks did Blazing Saddles and Young
Frankenstein. Wait, Slap Her... She’s French, that sounds like a cult movie. Let me think again.
Oh, right, the director isJim Abrahams. Wait, but I’m not 100% sure. Let me confirm. The
movie came out in 1975, I believe. The director was Jim Abrahams, who was part of the Three
Amigos, along with David and Jerry. So Jim Abrahams is the director here. Now, when was he
born?
Jim Abrahams was born on... I think he was born in the 1930s or 1940s. Let me recall. David
and Jerry are his brothers, part of the comedy trio. Wait, Jim Abrahams was born on July 16,
1940. Let me see if that’s correct. Alternatively, maybe 1941. Hmm, I might need to
double-check. Wait, another way: if the movie was released in 1975, and he was in his 30s
then, that would make his birth year around 1940-1945. So 1940 seems right. I think that’s
correct. So the answer should beJuly 16, 1940. </think>
Factual Claim:
1. Jim Abrahams is the director of the film ’Slap Her... She’s French’.
2. Jim Abrahams was born on July 16, 1940.
Refine Output: <think> Okay, I need to find out when the director of the film “Slap Her...
She’s French” was born. First, I should figure out who the director of that movie is. Let me
recall... I think the movie is from the 2000s. The title sounds familiar. Maybe it’s directed by
someone like Jim Abrahams? Wait, Jim Abrahams was part of the team that did the Police
Academy movies. Let me check that.
Alternatively, maybe it’s Mel Brooks? No, Mel Brooks did Blazing Saddles and Young
Frankenstein. Wait, Slap Her... She’s French, that sounds like a cult movie. Let me think again.
Oh, right, the director isMelanie Mayron. Wait, but I’m not 100% sure. Let me confirm. The
movie came out in 2002, I believe. The director was Melanie Mayron. Now, when was she
born?
I do not have the exact birth date of Melanie Mayron from the provided documents.
However, based on the information that the movie was released in 2002 and considering she
was likely in her early career or mid-career at that time, I would need to look up her birth year
elsewhere. For the purpose of this reasoning process, I will not provide a specific birth date as
it is not available in the given documents. </think>
Answer: 1965
Table 14:Case Study II.This case illustrates the output of CheckRLM via Post -reasoning Check. We
use QwQ-32B as the reasoning model and Llama-3.3-70B-Instruct as the recognition and correction
model.
23

Query:When was the director of film Slap Her... She’S French born?
Golden Reasoning Path:Slap Her. . . She’s French directorMelanie Mayron-> Melanie
Mayron date of birthOctober 20, 1952
Answer:October 20, 1952.
CheckRLM In-reasoning Output:
Step 1:
Reasoning Output: <think> Okay, I need to find out when the director of the film “Slap
Her... She’s French” was born. First, I should figure out who the director of that movie is. Let
me recall... I think the movie is from the 1980s. The title sounds familiar. Maybe it’sdirected
by someone like David Zucker? Wait, the Zucker-Abrahams-Zucker team did a lot of
comedies around that time. Let me confirm.
Factual Claim:
1. The movie ’Slap Her... She’s French’ is from the 1980s.
2. The Zucker-Abrahams-Zucker team directed many comedies around that time.
Refine Output: <think> Okay, I need to find out when the director of the film “Slap Her...
She’s French” was born. First, I should figure out who the director of that movie is. Let me
recall... The movie is from 2002, not the 1980s. The title sounds familiar, but it’s not directed
by someone like David Zucker.The movie is directed by Melanie Mayron. Let me confirm
Melanie Mayron’s birth date.
Step 2:
Reasoning Output:Wait, I thinkMelanie Mayron was born on July 21, 1968. I should
double-check that. Alternatively, maybe I’m mixing her up with someone else. Let me see...
Another thought: sometimes people confuse the director with the cast. The film’s cast includes
actors like Adam Goldberg, but the director is definitely Melanie Mayron.
Factual Claim:
1. Melanie Mayron is the director of the film Slap Her... She’S French.
2. Melanie Mayron was born on July 21, 1968.
Refine Output:Wait, I thinkMelanie Mayron was born on October 20, 1952. I should
double-check that. Alternatively, maybe I’m mixing her up with someone else. Let me see...
Another thought: sometimes people confuse the director with the cast. The film’s cast includes
actors like Piper Perabo, but the director is definitely Melanie Mayron.
Step 3:
Reasoning Output:So, Melanie Mayron’s birth date is October 20, 1952. So the answer
should be October 20, 1952. I’ll make sure there’s no conflicting info. Another check: if I
remember correctly, she was born in 1952. Yeah, that seems right. So the director’s birth date
is 1952-10-20. </think>
Factual Claim:
1. Melanie Mayron’s birth date is October 20, 1952.
2. The director of the film Slap Her... She’S French is Melanie Mayron.
Refine Output:So, Melanie Mayron’s birth date is October 20, 1952. So the answer should be
October 20, 1952. I’ll make sure there’s no conflicting info. Another check: if I remember
correctly, she was born in 1952. Yeah, that seems right. So the director’s birth date is
1952-10-20. </think>
Answer: October 20, 1952
Table 15:Case Study III.This case illustrates the output of CheckRLM via In -reasoning Check. We use
QwQ-32B as the reasoning model and Llama-3.3-70B-Instruct as the recognition and correction model.
24
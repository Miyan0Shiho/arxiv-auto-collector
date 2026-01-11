# Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models

**Authors**: Can Xu, Lingyong Yan, Jiayi Wu, Haosen Wang, Shuaiqiang Wang, Yuchen Li, Jizhou Huang, Dawei Yin, Xiang Li

**Published**: 2026-01-08 06:57:03

**PDF URL**: [https://arxiv.org/pdf/2601.04651v1](https://arxiv.org/pdf/2601.04651v1)

## Abstract
Recent advances in synergizing large reasoning models (LRMs) with retrieval-augmented generation (RAG) have shown promising results, yet two critical challenges remain: (1) reasoning models typically operate from a single, unchallenged perspective, limiting their ability to conduct deep, self-correcting reasoning over external documents, and (2) existing training paradigms rely excessively on outcome-oriented rewards, which provide insufficient signal for shaping the complex, multi-step reasoning process. To address these issues, we propose an Reasoner-Verifier framework named Adversarial Reasoning RAG (ARR). The Reasoner and Verifier engage in reasoning on retrieved evidence and critiquing each other's logic while being guided by process-aware advantage that requires no external scoring model. This reward combines explicit observational signals with internal model uncertainty to jointly optimize reasoning fidelity and verification rigor. Experiments on multiple benchmarks demonstrate the effectiveness of our method.

## Full Text


<!-- PDF content starts -->

Adversarial Yet Cooperative: Multi-Perspective Reasoning in
Retrieved-Augmented Language Models
Can Xu1,2, Lingyong Yan2, Jiayi Wu1, Haosen Wang3, Shuaiqiang Wang2,
Yuchen Li2,Jizhou Huang2,Dawei Yin2,Xiang Li1,
1East China Normal University,2Baidu Inc.,3Southeast University
Correspondence:Xiang Li xiangli@dase.ecnu.edu.cn
Abstract
Recent advances in synergizing large reason-
ing models (LRMs) with retrieval-augmented
generation (RAG) have shown promising re-
sults, yet two critical challenges remain: (1)
reasoning models typically operate from a sin-
gle, unchallenged perspective, limiting their
ability to conduct deep, self-correcting rea-
soning over external documents, and (2) ex-
isting training paradigms rely excessively on
outcome-oriented rewards, which provide in-
sufficient signal for shaping the complex, multi-
step reasoning process. To address these issues,
we propose an Reasoner-Verifier framework
named Adversarial Reasoning RAG (ARR).
The Reasoner and Verifier engage in reason-
ing on retrieved evidence and critiquing each
other’s logic while being guided by process-
aware advantage that requires no external scor-
ing model. This reward combines explicit ob-
servational signals with internal model uncer-
tainty to jointly optimize reasoning fidelity and
verification rigor. Experiments on multiple
benchmarks demonstrate the effectiveness of
our method. Our code is available at link.
1 Introduction
Large language models (LLMs) endowed with
step-by-step reasoning capabilities have achieved
remarkable success in complex question answer-
ing, especially when augmented with external
knowledge through retrieval-augmented genera-
tion (RAG) (Li et al., 2025c,b; Feng et al., 2025;
Wang et al., 2025). Different from previous RAG
methods that focus on retrieval optimization and
component-based architectural design, recent ef-
forts have been made on post-training LLM agents
(Jin et al., 2025; Li et al., 2025a) integrated with
search tools.
Despite the effectiveness, current RAG mainly
adopts a monologic reasoning architecture, where
only one single LLM-based agent reasons and inter-
acts with search engines. However, when retrieveddocuments are partial, conflicting or misleading,
the single-view reasoning may amplify errors rather
than mitigate them. Prior efforts address this chal-
lenge by incorporating self-verification process (He
et al., 2025; Fu et al., 2025). However, such self-
critique paradigm also suffers from the single-view
architecture, as many studies (Xu et al., 2024; Wu
et al., 2025; Zhang et al., 2025) show that LLMs
struggle to identify their own logical flaws.
Moreover, in order to train the agentic RAG sys-
tem, most existing methods optimize the RL frame-
work using outcome-oriented, task-level rewards
(e.g., accuracy or format correctness). Such re-
wards assign uniform reward to tokens within a
sequence based on the final correctness, lacking su-
pervision for the intermediate process. Unlike self-
contained trajectories in mathematical domains, the
correctness in RAG system depends not only on
reasoning quality, but on external factors beyond
the agent’s control, such as the precision of retrieval
engine, the consistency of external documents, and
the presence of conflicting evidence. Therefore,
outcome-based rewards cannot distinguish between
a correct answer derived through sound logic and
the one produced by lucky guesswork, nor can they
penalize plausible but flawed reasoning that hap-
pens to yield a wrong answer.
To tackle these challenges, we proposeARR
(AdversarialReasoningRAG), a multi-perspective
framework that explicitly decouples reasoning and
verification into separate perspectives, handled by
a reasoner agent and a verifier agent, respectively.
And we formalize such interactive process as an ad-
versarial yet collaborative dialogue between them:
Adversarial yet cooperative interaction: The
two agents should challenge each other not for win-
ning the debate, but for a shared objective. Cri-
tiques should be justified and evidence-grounded.
Process-aware learning: The two agents are
rewarded not only for correct final answers, but
also for high-quality interactive process between
1arXiv:2601.04651v1  [cs.AI]  8 Jan 2026

them (e.g., logical coherence, evidence utilization,
and uncertainty reduction).
To this end, we introduce anadversarial out-
come rewardand aprocess-aware advantageinto
the co-evolving process of both agents. (1). The
adversarial outcome reward encourages agents to
compete for higher correctness, ensuring that the
consensus is driven by rigorous debate rather than
blind agreement. (2). The process-aware advantage
is a token-level advantage for the verifier, which
is driven by a core insight: high-quality reason-
ing in RAG should mirror the reduction of uncer-
tainty and semantic entropy. With the proposal of
search queries and the accumulation of evidences,
the agent moves from an initial state of confusion
to a state of crystallization. Based on this guiding
principle, the process-aware advantage assesses the
soundness of response, the clarity of verification,
and the impact on the reasoner’s cognitive state.
By monitoring the evolution of reasoner policy en-
tropy, we reward verifier’s feedback that is confi-
dent, evidence-grounded, and steers the reasoner
from high-entropy exploration to low-entropy con-
vergence, thereby aligning the optimization with
the information gain.
In summary, our main contributions are:
(1) We propose ARR, where reasoner and veri-
fier engage in adversarial yet cooperative dialogue.
(2) We propose adversarial outcome reward to
promote rigorous debate between agents, which en-
courages agents to compete for higher correctness.
(3) We design the token-level process-aware ad-
vantage. By modeling reasoning progress as the
reduction of uncertainty, we reward trustworthy
and evidence-grounded verifier feedback that effec-
tively steers the reasoner toward better reasoning.
2 Related Work
2.1 Reward Design and Process Supervision
Reinforcement Learning with Verifiable Rewards
(RLVR) has emerged as a powerful approach to
enhance the reasoning capabilities of LLMs. For
example, Pass@k (Chen et al., 2025b) reveals that
outcome rewards provides limited learning signals
for tasks that are either overly simple or difficult,
and fail to discriminate between effective and inef-
fective process within the reasoning trace. It lever-
ages pass@k performance as the replacement for
outcome only rewards. DAPO (Yu et al., 2025)
introduces dynamic sampling to filter out samples
where model consistently succeeds or fails. In rea-soning RAG scenrario, Atom-Searcher (Deng et al.,
2025) introduces reasoning reward model to pro-
vide process signal additional to outcome reward.
WebSeer (He et al., 2025) introduces F1-score as
the intermediate-step verification signal to guide
the exploration process of search agent.
2.2 LRMs Synergizied with RAG
Recent advances in RAG systems include the in-
tegration of search tools and LRMs, which signif-
icantly improve the capabilities for complex and
multi-step reasoning and searching. The represen-
tative methods Search-R1 (Jin et al., 2025) and
R1-Searcher (Song et al., 2025) train models to
automatically derive reasoning through multi-turn
searching. DeepResearcher (Zheng et al., 2025)
further includes web search agent into agentic rea-
soning RAG. Existing methods are primarily built
upon single-agent frameworks, leaving a gap in
exploration from multi-perspective interactions.
3 Preliminary
3.1 Task Formulation
An ideal agentic reasoning RAG system should
go beyond the search-retrieve-answer pipeline and
possess high-order capabilities, including:
Critical reasoning: the capability to assess the
reliability of external evidence and detect logical
flaws in reasoning traces;
Grounded generation: the ability to anchor rea-
soning in verifiable evidence, and to revise conclu-
sions when support is insufficient;
Iterative refinement: the ability to enhance rea-
soning quality through self- and peer-assessment,
balancing both accuracy and process behaviors.
Current agentic RAG systems, however, remain
constrained by monologic architectures and op-
timization objectives that rely predominantly on
scaler outcome rewards. To bridge this gap, We
propose ARR, a multi-perspective reasoning frame-
work in which two agents learn to reason not as sin-
gle voices, but through interaction of different view-
points. Formally, we model the system as a multi-
agent Markov Decision Process (MDP), defined as
the tuple (Sα,Aα,Pα,Rα). Let α∈ {r,v} index
theReasoner( r) and theVerifier( v), interacting
in an environment that includes search engine and
document corpus D. Given a query q, the agent
behavior is governed by the policy model πα
θ. State
sα
t∈ Sαrefers to previous histories and external
context by the agent other than α, and aα
t∈ Aαis
2

Confused
(High)Ambiguous
(Medium)Crystalized
(Low)Certainty  (Entropy)
Step 1 Final Step 
Search RoundsStep iCertainty Through  Iterative
Search & Reasoning
Evidence Verify &
Extract
Verify Search
QueryFigure 1: Ideal agent certainty through iterative search
and reasoning
the action generated by πα
θfrom its action space
at turn t:aα
t=πα
θ(sα
t). Notably, distinct from
token-level MDPs, we define the action space Aα
at the semantic step level. An action aα
tis a se-
quence of tokens representing a complete move.
Take Search-R1 for an example, the action space
Aα= {think ,search ,answer }. A complete trace
τofninteraction steps is denoted as τ= (sr
1,ar
1,
sv
1,av
1,...,sr
n,ar
n,sv
n,av
n).
3.2 Entropy Pattern Analysis
Generally, RLVR for LLMs often involves the
trade-off between policy entropy and performance
(Cui et al., 2025). In the context of RAG, the rea-
soning process can be regarded as the dynamic evo-
lution of cognitive states driven by external knowl-
edge management. As illustrated in Figure 1, an
ideal reasoning trajectory exhibits three stages. (1)
Initial uncertainty: the agent begins in a confused
state, where high policy entropy reflects the lack
of knowledge and exploration of search queries.
(2)Evidence integration: the agent assimilates re-
trieval results and converges towards final answer.
(3)Crystallization: the agent has sufficient evi-
dence and generates a well-supported conclusion.
To formalize this intuition, we present the follow-
ing proposition.
Proposition 1.In an ideal agentic RAG system, as
relevant information is retrieved, both the uncer-
tainty of agent and the policy entropy monotoni-
cally decrease.
Proof. Consider an agentic RAG system with sin-
gle agent (e.g. Search-R1). Let Ydenotes the
ground-truth answer. Given a user query q, the
statest+1is the union of prior state st, action at
and retrieved document dt. To ensure rigor, weintroduce two assumptions.
Assumption 1:The agent acts to maximize the
expected information gain and issues search queries
intended to retrieve relevant documents.
Assumption 2:The retrieved documents dtpro-
vides a positive information gain regardingY.
(1). Monotonic Decrease of Answer Uncertainty:
The uncertainty of Ywith document dtis quanti-
fied by the conditional mutual information:
I(Y;d t|st, at) =H(Y|s t, at)−H(Y|s t+1).(1)
By definition, the updated state is st+1 =
[st, at, dt]. Since atis generated based on state
st, we have the Markov property H(Y|s t, at)≈
H(Y|s t). Substituting these into Eq 1, we have:
H(Y|s t+1)≈H(Y|s t)−I(Y;d t|st, at).Since
mutual information is non-negative, and the re-
triever provides relevant information, we obtain:
H(Y|s t+1)≤H(Y|s t).(2)
This indicates that remaining uncertainty of the
ground-truth is non-increasing with the accumula-
tion of retrieved documents.
(2). Convergence of Policy Entropy:Next, we
discuss the entropy of action at, noted as H(a t|st).
We decompose it using the definition of mutual
information between action atand ground-truth Y:
H(a t|st) =I(a t;Y|s t) +H(a t|Y, s t).(3)
Then, we analyze the two terms on the right side
of Eq 3. First, the mutual information is defined as
I(at;Y|s t) =H(Y|s t)−H(Y|a t, st). Thereby,
I(at;Y|s t)≤H(Y|s t)holds. Second, the term
H(a t|Y, s t)represents the uncertainty of agent’s
action given that the ground truth is known. Follow-
ing Assumption 2, as the training progresses, given
the ground-truth answer, the agent would gradually
come to a deterministic output, i.e., H(a t|Y, s t)≈
0. Therefore, we have the upper bound for the
policy entropy:
H(a t|st)≤H(Y|s t) +H(a t|Y, s t).(4)
Since H(Y|s t)is monotonically decreasing, and
H(a t|st)follows the upper bound of H(Y|s t).
This illustrates that as the agent accumulates ev-
idence, its reasoning process naturally converges
from exploration to exploitation.
To empirically validate this theoretical proposi-
tion, we conduct a statistical analysis of the policy
3

0 20 40 60 80 100 120 140
#Training Step0.00.10.20.30.40.5ProportionCorrect: Monotonic Increasing
Correct: Decrease then increase
Correct: Flat
Correct: Increase then Decrease
Correct: Monotonic DecreasingIncorrect: Monotonic Increasing
Incorrect: Decrease then increase
Incorrect: Flat
Incorrect: Increase then Decrease
Incorrect: Monotonic Decreasing
0 50 100
#Training Step0.00.20.40.60.8Accuracy
Monotonic Increasing
Decrease then increase
Flat
Increase then Decrease
Monotonic Decreasing
0 20 40 60 80 100 120 140 160 180
#Training Step0.00.10.20.30.40.5ProportionCorrect: Monotonic Increasing
Correct: Decrease then increase
Correct: Flat
Correct: Increase then Decrease
Correct: Monotonic DecreasingIncorrect: Monotonic Increasing
Incorrect: Decrease then increase
Incorrect: Flat
Incorrect: Increase then Decrease
Incorrect: Monotonic Decreasing
0 50 100 150
#Training Step0.00.20.40.60.8Accuracy
Monotonic Increasing
Decrease then increase
Flat
Increase then Decrease
Monotonic DecreasingFigure 2: Statistical analysis of policy entropy pattern in Search-R1 trajectories. The y-axis of theleft subplots
denotes the proportion of trajectories exhibiting specific pattern in all multi-turn ( ≥3) samples. The y-axis of the
right subplotsrepresents the average accuracy of samples grouped by their pattens.
entropy evolution during the training process of
Search-R1 (Jin et al., 2025) in Figure 21. Specifi-
cally, we focus on agent trajectories containing at
least three search & reasoning turns and aggregate
the statistics every 20 training steps. The action
entropy is Hat=1
|at|P|av
t|
j=1H(π θ(at,j|st, at,<j)),
where at∈ {think} and|·|measures the sequence
length of action at. Specifically, the trend between
action at+1andatisIncrease, if ∆Hat+1>δ;De-
crease, if ∆Hat+1< -δ;Flat, otherwise. Here,
∆Hat+1=H at+1−H atandδis the threshold
which accounts for minor fluctuations during rea-
soning. For each of them, we track the average
token entropy of last three turns and categorize its
evolution trend into five patterns:Monotonic In-
creasing(I),Decrease-then-Increase(DI),Flat(F),
Increase-then-Decrease(ID), andMonotonic De-
creasing(D). We define a mapping function fe:Rn
→ {D ,ID,F,DI,I}. Figure 2 leads to two primary
observations:
Correlation with Correctness:There is a posi-
tive correlation between theMonotonic Decreasing
entropy pattern and model’s accuracy. This sug-
gests that effective reasoning is often accompanied
by a progressive resolution of uncertainty.
Evolution of Exploration:Throughout training,
there is a notable rise in the proportion of sam-
ples exhibiting an overall reduction in policy en-
tropy (i.e.,Increase-then-Decrease, andMonotonic
Decreasing). Quantitatively, for the Qwen2.5-3B
backbone, the proportion rises from 51.74% in the
early phase to 69.57% in the late training phase.
This suggests that the model learns to narrow down
search space and converge on valid solutions as its
multi-turn exploration capability deepens.
These observations support the premise that suc-
1We only show results on Qwen2.5-3B here, and more
results are shown in Appendix.
Multi-perspective Reasoning
<Think>  I don't have knowledge about ... 
<Search>  What is the ... ?[Query] : What is the ... ?
Search
QueryDocuments
<Verify>  The search query does not address the question ... 
<Response>  The query should focus more on ...
<Verify>  The query successfully addressed... The source
mentioned ..., which is directly relevant.
<Selected_Doc>  [DOC 1: Title: ..., Content: ...] 
<Response>  Given the information, it appears that ...Search
Query
Multi-turn interactions ...
<Verify>  The documents
provide information about ...
<Think>  With the verified
information ...
<Answer>  [Answer]
<Verify>  The [Answer] is (not)
explicitly stated in the
documents ...
<Final_Answer>  [Final Answer] 
Based on the above dialogue , I think the answer to the question is ...
DocumentsReasoner
Verifier
VerifierReasoner
Reasoner Verifier
Final Answer
Generator
<Verify>  The verifier may be right about my
previous query ...
<Think>  Well, let me reconsider ... 
<Search>  What is the ... ?Figure 3: Multi-perspective reasoning of ARR.
cessful reasoning in RAG systems is intrinsically
characterized by the progressive resolution of pol-
icy uncertainty. Collectively, this theoretical insight
and empirical observation provide a robust foun-
dation for the process reward design within our
proposed multi-agent framework.
4 Methods
4.1 Multi-perspective Reasoning
Building upon the formulation above, ARR per-
forms multi-perspective reasoning and verification
through an iterative dialogue between two agents:
Reasonertakes the lead in exploration. It for-
mulates search queries, retrieves documents, con-
structs step-by-step reasoning, and proposes candi-
date answers.
Verifierserves as a critical partner. It checks the
relevance and credibility of search queries and re-
trieved documents, identifies logical gaps or unsup-
ported claims in reasoning, and performs validation
4

of the answers proposed by the reasoner.
For the reasoner, the action space is defined as Ar
={think ,search ,verify ,answer} , a complete
reasoning step is ( think ,search ,[feedback] ,
verify), and the final step is (think,answer).
•think : the segment of reasoning grounded in
the given queryqand retrieved evidence;
•search : search queries issued when external
knowledge is deemed necessary;
•[feedback] : the feedback provided by the ver-
ifier, including supporting evidence or critiques;
•verify : self-assessment of the reliability and
sufficiency of external knowledgeinformation.
•answer : the answer when the reasoner thinks
reasoning is complete and well-supported. Corre-
spondingly, the verifier operates within the action
spaceAv={verify ,selected_doc ,response ,
final_answer} . A complete verification step
is ([ information ],verify ,selected_doc ,
response ), and the final step is ( [information] ,
verify,final_answer).
•[information] : search queries by the reasoner
associated with retrieved documents or the rea-
soner’s answer once it finishes reasoning;
•verify : verification on validity of queries,
document relevance, and logical soundness;
•selected_doc : curated documents (e.g. Doc
n) that directly support or refute the claim;
•response : explicit feedback to the reasoner,
comprising either supporting evidence for valid
queries or constructive critiques with justification
for flawed ones.
•final_answer : the final conclusion after veri-
fying all the evidence and the reasoner’s answer.
Notably, the reasoner has a built-in verify stage
within each step, allowing it to critically assess the
reliability of retrieved evidence before coming to
a conclusion. The verifier is instructed to return
the most relevant source passage in selected_doc ,
rather than returning all retrieved documents to the
reasoner without indiscriminately. This prevents
information overload while guaranteeing feedback
retains traceable and verifiable evidence. Together,
these mechanisms help form a balanced adversar-
ial dialogue, where neither agent dominates, and
reasoning quality emerges from their structured
interactions.
Following the iterative dialogue, we concatenate
the full interaction historyτinto a unified prompt,
which is then fed to the final answer generator.
Note that it employs the same policy model as the
reasoner. By explicitly synthesizing insights from
Process-aware Advantage
<Think> ...  <Search>  ... [Information]
<Verify>... <Think> ... <Search> ...
[Information]  <Verify>... <Think>
... <Answer> ...[Information] <Verify> ... <Response> ...
[Information] <Verify> ...  <Selected_Doc>  ...
<Response> ... [Information] <Verify>...
<Final_Answer>  ...
1. Guide Towards Correctness 2. Confident Responses
3. Grounded Evidence 4. Clarify Reasoner
1. Incorrect Guidance 2. Vague & Uncertain Responses
3. Incorrectly Filter Evidence 4. Confuse Reasoner
Reward
Penalty
Figure 4: Process-aware advantage of ARR.
both perspectives, we obtain a more robust and
well-grounded final answer. Detailed prompts for
all agents are provided in Appendix A.3.
4.2 Multi-perspective Optimization
To overcome the limitations of sparse outcome-
based supervision and to explicitly promote con-
structive adversarial interactions, we propose a
multi-perspective reward design that disentangles
final correctness from process fidelity. This de-
sign ensures that agents are rewarded not only for
generating correct answers but also for engaging
in high-quality and evidence-grounded dialogue.
Our reward scheme consists of two components:
adversarial outcome reward and process-aware ad-
vantage for the verifier.
Adversarial Outcome RewardsIn consistent
with the adversarial yet cooperative dialogue de-
sign, our outcome reward explicitly promotes effec-
tive adversarial engagement by rewarding agents
not just for its’ correctness, but for outperforming
their counterpart. Formally, each agent αreceives
an outcome reward composed of two terms:
rα=F1(yα, ygold)+λ·max
bin(rα
out−r¯α
out),0
,
(5)
where ¯αdenotes the counterpart agent, yαdenotes
the answer by agent α, and ygoldis the ground
truth. The operator bin(·) discretizes the range
of F1 score into nbuckets, filtering minor differ-
ences between answers by both agents. Thus, both
agents are rewarded not only for correctness, but
also for better performance than the other agent.
Such reward also helps increase the discrimination
of rewards within a group in Group Relative Pol-
icy Optimization (GRPO), particularly for tasks of
moderate difficulty.
Process-aware AdvantageWhile outcome re-
ward drives both policy models towards accurate
answers, they provide limited signal regarding how
5

correctness is achieved. To address this, we intro-
duce a token-level process-aware advantage Av
proc
for the verifier, which encourages trustworthy and
evidence-grounded response that steer the reasoner
toward better reasoning. Formally, we define:
Av
proc=F1(yr, ygold)·A clarity·Aimpact,(6)
where the three terms each encodes correctness,
clarity, and behavior impact, respectively.
(1) Answer CorrectnessThe reasoner’s final
F1 score serves as a necessary condition: only
when the dialogue yields a correct answer does
the verifier receive full process credit. This pre-
vents rewarding the verifier for critiques that lead
reasoning toward wrong conclusions.
(2) Verifier Clarity
Aclarity =exp(−Hv
at)·I[y goldindt]
·(2I[y goldinav
t]−1),(7)
where dt∈D is the retrieved documents at step
t, and av
tis the verifier’s action in Av
sub(e.g.,
{verify ,response }). Here, Hv
atdenotes the
average policy entropy of action av
t:Hv
at=
1
|av
t|P|av
t|
j=1H(πv
θ(av
t,j|sv
t, av
t,<j))In the first term,
lower entropy means higher semantic certainty. We
encourage confident and decisive critiques. The
second term ensures that the verifier only receives
credit or punishment when the final answer is ac-
tually supported by the retrieved documents, mit-
igating bias from imperfect retrieval. The third
term penalizes responses that filter correct answers,
thereby promoting faithfulness.
(3) Behavior ImpactMost critically, we quan-
tify how the verifier’s feedback influences subse-
quent reasoning. Let Hr
atdenote the average token-
level entropy of reasoner’s action ar
t. Over a dia-
logue of ninteraction steps, we analyze the entropy
trend in the last three steps and classify it into one
of the five patterns defined in Section 3.2. We then
assign a score to each pattern: score(p) ={D→
1.0,ID→0.8,F→0.6,DI→0.4,I→0.2}
Finally, the impact is then:
Aimpact =1
|Ar
sub||Ar
sub|X
j=1score 
fe([Hr
a1, ..., Hr
an])
,
(8)
where Ar
subis the set of reasoner action subjected
to monitoring. This term incentivizes the verifier
to provide feedback that steers the reasoner toward
low-entropy and decisive reasoning.4.2.1 Policy Optimization
We optimize both agents using Group Relative Pol-
icy Optimization (GRPO), which normalizes advan-
tages within a group of rollouts and incorporates
a reference model for KL regularization. For each
query q, we sample Gtraces{τi}G
i=1and calculate
the outcome reward {rα
i}G
i=1. The token-level ad-
vantage for t-th token in trace iis first computed as:
Aα
i,t=rα
i−mean(rα
1,rα
2,...,rα
G)
std(rα
1,rα
2,...,rα
G)For the reasoner, the
final advantage is ˆAr
i,t=Ar
i,t. For the verifier, the
process-aware advantage is added:
ˆAv
i,t=Av
i,t+I(y i,t∈at∧at=Av
sub)·Av
proc (9)
ensuring Av
procis only added to tokens belonging
to the verifier’s critique sections. Therefore, the
policy model is optimized by maximizing:
Jα
GRPO (θ) =Eq,{yi,t}G
i=1∼
πα
old(·|x;R)"
1
GGX
i=1|yi|X
t=1
min 
rα
i(θ)ˆAα
i,t,clip(rα
i(θ),1−ϵ,1 +ϵ) ˆAα
i,t!
−βD KL[πα
θ||πα
ref]#
,(10)
where rα
i(θ) =πα
θ(yi,t|q)
πα
old(yi,t|q)andπα
refis the reference
model. Following common practices in this field,
tokens not generated by the policy model πα
oldwill
be masked in the loss calculation.
5 Experiments
5.1 Setup
Datasets & MetricsWe conduct evaluation on
diverse QA benchmarks. Our method and all base-
lines are trained on NQ (Kwiatkowski et al., 2019)
and HotpotQA (Yang et al., 2018). Following previ-
ous studies (Zheng et al., 2025), we randomly sam-
ple 512 examples from the development set of NQ,
HotpotQA, TriviaQA (Joshi et al., 2017), 2Wiki-
MultiHopQA (Ho et al., 2020), PopQA (Mallen
et al., 2023), and MuSiQue (Trivedi et al., 2022),
as well as all 125 samples from Bamboogle (Press
et al., 2023). We adopt Exact Match (EM) and
F1-score for comparison.
BaselinesWe compare our method against sev-
eral baselines for reasoning and RAG in ques-
tion answering, including CoT (Wei et al., 2022),
RAG (Lewis et al., 2020), Search-R1 (Jin et al.,
6

MethodGeneral QA Multi-hop QAAverage
NQ TriviaQA PopQA HotpotQA 2Wiki Musique Bamboogle
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen2.5-3B Instruct
CoT 2.5 8.7 7.3 13.2 7.4 13.7 3.4 12.5 2.1 12.4 0.2 3.2 0.0 0.0 3.27 9.1
RAG 33.7 39.6 51.5 58.4 36.5 44.2 22.1 30.4 21.6 29.9 7.2 13.6 60 12.3 25.5 32.6
Search-R1 38.7 45.9 55.2 66.7 40.0 50.2 31.4 38.3 32.4 41.2 10.2 20.8 18.9 25.6 32.4 41.2
-pass@242.9 - 64.5 - 42.2 - 34.1 - 35.0 - 12.8 - 25.6 - 36.7 -
ReSearch 39.5 46.7 59.3 67.8 41.4 49.9 32.9 38.6 33.3 42.5 11.4 21.0 24.7 31.2 34.6 42.5
WebSeer 39.0 46.3 58.5 67.1 40.7 50.6 35.3 44.4 34.8 43.6 11.5 21.7 24.3 32.5 34.8 43.7
ARR43.7 53.2 65.6 73.6 46.6 54.1 36.0 44.9 35.8 44.6 14.5 23.7 27.9 36.1 38.6 47.2
Qwen2.5-7B Instruct
CoT 3.5 11.0 18.7 25.4 8.8 14.1 7.2 15.9 6.7 15.2 2.8 8.3 13.8 19.6 8.8 15.6
RAG 30.6 37.2 58.0 64.2 38.1 45.4 26.4 34.0 24.9 33.3 7.4 14.7 16.3 24.3 28.8 36.2
Search-R1 40.6 48.0 65.2 73.7 39.7 49.2 38.4 45.6 38.9 45.6 14.7 21.4 37.1 44.0 39.2 46.8
-pass@241.5 - 67.4 - 50.6 - 40.2 - 40.6 - 15.6 - 45.9 - 43.1 -
ReSearch 41.6 49.8 64.0 71.6 42.8 50.8 39.2 46.0 40.8 47.2 15.8 25.3 43.1 49.7 41.0 48.6
WebSeer 40.1 48.4 65.5 73.2 42.3 48.7 38.6 45.641.7 51.314.4 24.6 42.6 48.9 40.7 48.7
ARR45.1 53.7 68.2 75.5 50.8 56.6 45.5 54.241.5 50.6 17.7 27.3 46.4 53.8 45.0 53.1
Qwen3-8B
CoT 2.7 10.7 19.0 25.7 8.5 14.7 7.4 16.7 7.8 16.9 5.3 13.6 17.4 26.3 9.7 17.8
RAG 34.1 42.7 58.5 67.2 40.5 47.2 31.4 36.2 30.7 35.4 10.6 17.4 23.1 31.7 32.7 39.7
Search-R1 42.6 50.6 67.5 75.1 42.2 48.5 41.8 47.3 42.3 48.6 16.3 24.9 42.6 53.0 42.2 49.7
-pass@242.9 - 70.5 - 48.2 - 45.4 - 48.8 - 19.6 - 48.9 - 46.3 -
ReSearch 38.5 46.7 62.3 70.8 40.7 49.7 38.1 47.4 39.2 45.1 15.2 23.1 40.6 49.7 39.2 47.5
WebSeer 46.3 52.8 68.0 75.6 41.7 50.4 46.2 53.4 47.0 54.8 14.6 23.0 44.9 55.6 44.1 52.2
ARR47.2 54.0 76.0 83.7 49.1 56.2 50.6 57.8 52.4 58.3 20.2 28.1 53.4 64.7 49.8 57.5
Table 1: Performance comparison between ARR and baselines. Best and runner-up results are highlighted inbold
and underline .
VariantsNQ TQA PQA HQA 2Wiki MSQ BAM
Qwen2.5-3B Instruct
ARR 53.2 73.6 54.1 44.9 44.6 23.7 36.1
w/o adv-out50.6 69.6 51.5 43.2 42.9 22.4 35.5
w/o proc-adv51.8 69.0 50.7 43.6 41.7 20.6 33.4
Qwen2.5-7B Instruct
ARR 53.7 75.5 56.6 54.2 50.6 27.3 53.8
w/o adv-out49.5 73.6 55.2 50.4 49.7 25.3 52.6
w/o proc-adv49.6 72.5 51.6 45.2 48.3 21.9 47.0
Qwen3-8B
ARR 54.0 83.7 56.2 57.8 58.3 28.1 64.7
w/o adv-out53.7 83.4 55.6 57.3 56.1 27.5 63.2
w/o proc-adv52.4 82.5 52.5 54.6 54.4 23.9 61.8
Table 2: Ablation studies on ARR. Datasets are abbrevi-
ated and correspond to Table 1, respectively.
2025), ReSearch (Chen et al., 2025a), and Web-
Seer (He et al., 2025). Additionally, to fairly evalu-
ate the efficacy of the adversarial yet cooperative
dialogue in our multi-agent system, we introduce
thepass@2metric for Search-R1.
ImplementationFor retrieval, all baselines
adopt the same retriever and corpus setting as
Search-R1. The retriever returns the top-3 doc-
uments. We select Qwen2.5-3B, -7B (Qwen et al.,
2025), and Qwen3-8B (Yang et al., 2025) as back-
bone models. We optimize the policy model using
0 25 50 75 100 125 150 175 2000.30.40.50.6F1-ScoreF1-Score Comparison (Qwen2.5-3B)
Ours
Search-R1
0 25 50 75 100 125 150 175 200
#Training Step0.00.20.40.6F1-ScoreF1-Score Comparison (Qwen2.5-7B)
Ours
Search-R1Figure 5: F1-score comparison.
the GRPO algorithm. For each prompt, we sample
5 trajectories with up to 5 interaction turns.
5.2 Main Results
Main results on general QA and multi-hop QA
benchmarks across three backbone models are pre-
sented in Table 1. Overall, ARR consistently
7

0 20 40 60 80 100 120 140 160 180123456Average Entropy
<think> by Reasoner
Correct: Turn 1
Correct: Turn 2
Correct: Turn 3Incorrect: Turn 1
Incorrect: Turn 2
Incorrect: Turn 3
0 20 40 60 80 100 120 140 160 180
#Training Step2.55.07.510.012.5Average Entropy
<response> by VerifierFigure 6: Entropy transition of agent actions in ARR.
outperforms all baseline methods across varying
model sizes and datasets. The average improve-
ment over runner-up baseline is 11.1% in EM and
7.6% in F1-score on Qwen2.5-3B, 9.5% in EM and
7.8% in F1-score on Qwen2.5-7B, and 13.4% in
EM and 9.8% in F1-score on Qwen3-8B. The per-
formance gains of ARR remain consistent as the
model size scales from 3B to 8B, suggesting that
the proposed method is model-agnostic.
Remarkably, ARR with 3B backbone outper-
forms baselines models with 7B backbone on gen-
eral QA benchmarks. This indicates that multi-
perspective reasoning unleash the potential of com-
pact backbones on relatively simple benchmarks.
ARR also exhibits significant performance gains on
several multi-hop QA benchmarks. For instance,
the gains over runner-up on Musique is 26.1%,
12.0%, and 23.9% in EM, respectively. Similarly,
on HotpotQA, our method achieves the EM score
of 0.455 and 0.506 on 7B and 8B. This shows that
the multi-perspective reasoning architecture effec-
tively solves complex multi-hop queries.
Our proposed method also frequently surpasses
Search-R1 (pass@2). Take performance on NQ
and HotpotQA with Qwen3-8B backbone for an
example, ARR achieves 0.472 and 0.506 in EM,
surpassing Search-R1 by 10% and 11.5%, respec-
tively. This confirms that the superior performance
of ARR is not the result of naive model scaling,
but the adversarial yet cooperative dialogue and the
multi-perspective optimization strategy.
Figure 5 shows the F1-score of our method and
Search-R1 throughout training. Our method consis-tently outperforms Search-R1. Unlike Search-R1
which suffers from the cold start problem, our meth-
ods shows strong performance during early training
stage on the 7B model.
5.3 Ablation Studies
In this sub-section, we present the results of ab-
lation experiments to evaluate the contribution of
key components in ARR. We introduce 2 variants:
(1) ARR without adversarial outcome rewards (w/o
adv-out) and (2) ARR without process-aware ad-
vantage (w/o proc-adv). Results across three back-
bone models are shown in Table 2.
The removal of the process-aware advantage
leads to the most significant performance drop,
particularly on multi-hop QA benchmarks. For
instance, on Musique dataset with Qwen2.5-7B
backbone, the F1 score drops from 27.3 to 21.9.
This suggests that the proposed process-aware ad-
vantage is crucial for complex tasks requiring multi-
step deduction. The exclusion of the adversarial
outcome reward also results in a consistent perfor-
mance degradation, and the impact is smaller.
5.4 Entropy Evolution
We present the entropy transition of agent actions
in multi-turn trajectories of ARR in Figure 6. In
general, the action entropy of the third turn con-
sistently achieving lower values than initial turns.
The entropy of response by Verifier shows dra-
matic differences between correct and incorrect
trajectories. These observations are consistent with
the empirical studies regarding entropy pattern in
Section 3.2. The uncertainty within think by Rea-
soner gradually decreases as training progresses,
indicating that the Reasoner is acquiring multi-turn
reasoning capabilities.
6 Conclusion
In this paper, we introduced ARR, a multi-
perspective agentic RAG framework that decouples
reasoning and verification into an adversarial yet
co-evolving system. Further, we bridged the gap be-
tween outcome-oriented reward and process-aware
guidance by proposing an adversarial outcome re-
ward and a process-aware advantage that reward
the verifier for evidence-grounded, and uncertainty
reducing feedback. Results show that our methods
consistently outperform existing baselines and fre-
quently exceed the pass@2 results of competitors.
8

References
Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze
Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang,
Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang,
Zenan Zhou, and Weipeng Chen. 2025a. Research:
Learning to reason with search for llms via reinforce-
ment learning.
Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling,
Qinghao Ye, Wayne Xin Zhao, and Guang Shi. 2025b.
Pass@k training for adaptively balancing exploration
and exploitation of large reasoning models.
Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan,
Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan,
Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng,
Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and
Ning Ding. 2025. The entropy mechanism of rein-
forcement learning for reasoning language models.
Yong Deng, Guoqing Wang, Zhenzhe Ying, Xiaofeng
Wu, Jinzhen Lin, Wenwen Xiong, Yuqin Dai, Shuo
Yang, Zhanwei Zhang, Qiwen Wang, Yang Qin, Yuan
Wang, Quanxing Zha, Sunhao Dai, and Changhua
Meng. 2025. Atom-searcher: Enhancing agentic
deep research via fine-grained atomic thought reward.
Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang,
Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin
Chi, and Wanjun Zhong. 2025. Retool: Reinforce-
ment learning for strategic tool use in llms.
Daocheng Fu, Jianbiao Mei, Licheng Wen, Xuemeng
Yang, Cheng Yang, Rong Wu, Tao Hu, Siqi Li,
Yufan Shen, Xinyu Cai, Pinlong Cai, Botian Shi,
Yong Liu, and Yu Qiao. 2025. Re-searcher: Robust
agentic search with goal-oriented planning and self-
reflection.
Guanzhong He, Zhen Yang, Jinxin Liu, Bin Xu, Lei
Hou, and Juanzi Li. 2025. Webseer: Training deeper
search agents through reinforcement learning with
self-reflection.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen tau Yih. 2020. Dense passage retrieval for open-
domain question answering.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:453–466.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao,
Xinpeng Liu, Jiayu Zhang, Zhenqiang Huang, Qian-
ben Chen, Weichen Sun, Qiexiang Wang, Hongxuan
Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying
Fan, Xiaowan Li, Tiannan Wang, Pai Liu, King Zhu,
and 11 others. 2025a. Chain-of-agents: End-to-end
agent foundation models via multi-agent distillation
and agentic rl.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025b. Search-o1: Agentic search-enhanced
large reasoning models.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian,
Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and
Zhicheng Dou. 2025c. Webthinker: Empowering
large reasoning models with deep research capabil-
ity.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. InFindings of the Association for Computational
9

Linguistics: EMNLP 2023, pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025.
Qwen2.5 technical report.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2024. Hybridflow: A flexible
and efficient rlhf framework.arXiv preprint arXiv:
2409.19256.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2024. Text embeddings by weakly-
supervised contrastive pre-training.
Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue
Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan
Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb,
Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei,
Lijuan Wang, Yejin Choi, and Manling Li. 2025. Ra-
gen: Understanding self-evolution in llm agents via
multi-turn reinforcement learning.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems,
volume 35, pages 24824–24837. Curran Associates,
Inc.
Ting Wu, Xuefeng Li, and Pengfei Liu. 2025. Progress
or regress? self-improvement reversal in post-
training. InThe Thirteenth International Conference
on Learning Representations.
Wenda Xu, Guanglei Zhu, Xuandong Zhao, Liangming
Pan, Lei Li, and William Wang. 2024. Pride and prej-
udice: LLM amplifies self-bias in self-refinement.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 15474–15492, Bangkok, Thai-
land. Association for Computational Linguistics.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 others.
2025. Qwen3 technical report.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xi-
aochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gao-
hong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi
Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi
Zhang, Mofan Zhang, Wang Zhang, and 16 others.
2025. Dapo: An open-source llm reinforcement
learning system at scale.
Qingjie Zhang, Di Wang, Haoting Qian, Yiming Li,
Tianwei Zhang, Minlie Huang, Ke Xu, Hewu Li, Liu
Yan, and Han Qiu. 2025. Understanding the dark side
of LLMs’ intrinsic self-correction. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 27066–27101, Vienna, Austria. Association
for Computational Linguistics.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments.
A Appendix
A.1 Additional Preliminary Studies
Due to the limited space, we present analysis of
policy entropy pattern in Search-R1 trajectories on
Qwen2.5-7B in this subsection. From Figure 7,
empirical studies on Qwen2.5-7B show similar pat-
tern with studies on the 3B models. There is a
positive correlation between correctness and de-
creasing entropy pattern. Similar to observations in
Section 3.2, there exists a rise in the proportion of
samples exhibiting an overall reduction in policy
entropy. Quantitatively, for the Qwen2.5-7B back-
bone, the proportion rises from from 51.65% in the
early phase to 57.96% in the late training phase.
A.2 Implementation Details
We use the 2018 Wikipedia dump (Karpukhin et al.,
2020) as the knowledge database and use E5 (Wang
et al., 2024) as the retriever model. Our experi-
ments are conducted on 8 ×A100 GPUs, with full
parameter optimization and gradient checkpoint-
ing. We build our method based on VeRL (Sheng
et al., 2024) and use vLLM (Kwon et al., 2023) to
accelerate agent rollouts.
10

0 20 40 60 80 100 120 140
#Training Step0.00.10.20.30.40.5ProportionCorrect: Monotonic Increasing
Correct: Decrease then increase
Correct: Flat
Correct: Increase then Decrease
Correct: Monotonic DecreasingIncorrect: Monotonic Increasing
Incorrect: Decrease then increase
Incorrect: Flat
Incorrect: Increase then Decrease
Incorrect: Monotonic Decreasing
0 50 100
#Training Step0.00.20.40.60.8Accuracy
Monotonic Increasing
Decrease then increase
Flat
Increase then Decrease
Monotonic Decreasing
0 20 40 60 80 100 120 140 160 180
#Training Step0.00.10.20.30.40.5ProportionCorrect: Monotonic Increasing
Correct: Decrease then increase
Correct: Flat
Correct: Increase then Decrease
Correct: Monotonic DecreasingIncorrect: Monotonic Increasing
Incorrect: Decrease then increase
Incorrect: Flat
Incorrect: Increase then Decrease
Incorrect: Monotonic Decreasing
0 50 100 150
#Training Step0.00.20.40.60.8Accuracy
Monotonic Increasing
Decrease then increase
Flat
Increase then Decrease
Monotonic DecreasingFigure 7: Statistical Analysis of Policy Entropy Pattern in Search-R1 trajectories. The y-axis of theleft subplots
denotes the proportion of trajectories exhibiting specific pattern relative to all multi-turn ( ≥3) samples. The y-axis
of theright subplotsrepresents the average accuracy of samples grouped by their pattens.
A.3 Prompt Templates
Prompt for the Reasoner
To answer the given question, you will act
as a reasoner working collaboratively with
a retriever. Follow these steps carefully:
1. Reasoning Phase: When you receive a
question, begin by reasoning about it inside
<think> and </think>. This is where you
analyze the problem and determine what
you already know.
2. Identify Knowledge Gaps: If, dur-
ing your reasoning, you realize that you
lack some necessary information, you can
request external knowledge by calling a
search engine. To do this, write your query
inside <search> and </search>.
3. Receive Search Results: After submit-
ting your query, the verifier will process it
and provide you with the top search results
along with its opinion. This information
will be enclosed between <feedback> and
</feedback>.
4. Verification Phase: Every time you re-
ceive new information, you must first verify
its relevance and usefulness. Conduct this
verification inside <verify> and </verify>.
5. Update Reasoning: Based on the verified
information, perform another round of rea-
soning inside <think> and </think>. Repeat
steps 2–4 as many times as needed until
you have enough information to answer the
question.
6. Provide the Answer: Once you deter-
mine that no further external knowledge isrequired, provide your final answer directly
inside <answer> and </answer>. Make sure
to verify and think before answer the ques-
tion. Keep your answer concise without
additional explanations. For example: <an-
swer> Beijing </answer>.
Always adhere strictly to the specified
XML-like tags and respond only with the
required elements.
Question: [QUESTION]
Prompt for the Verifier
As a verifier, your task is to collaborate with
the reasoner to answer the given question.
Follow these steps carefully:
1. Verification Process:
•The reasoner will provide its reasoning
path, a retrieval query, and results from
the search engine enclosed within <infor-
mation> ... </information>.
•Perform a verification check inside <ver-
ify> ... </verify> to assess whether the
query effectively contributes to answering
the question.
2. Handling Effective Queries:
If the query is deemed appropriate:
•Choose the single most relevant doc-
ument from the retrieved results and in-
dicate it inside <selected_doc> ... </se-
lected_doc> (e.g., <selected_doc> Doc 1
</selected_doc>).
•Synthesize the selected information and
your own reasoning into a clear, concise
11

reply inside <response>...</response>.
3. Handling Ineffective Queries:
If the query is judged ineffective, DI-
RECTLY Provide a justification for this as-
sessment inside <response>...</response>.
4. Answer Verification:
If the reasoner provides an answer enclosed
within <answer> and </answer>
•Verify the answer inside <verify> ... </ver-
ify> based on your judgment.
•Provide the final verified response in-
side <final_answer>...</final_answer>, en-
suring it is concise and free of un-
necessary details. For example: <fi-
nal_answer>Beijing</final_answer>.
Always adhere strictly to the specified
XML-like tags and respond only with the
required elements.
Question: [QUESTION]
Prompt for the Final Predictor
The rollout text of the reasoner and verifier
is: [REASONER & VERIFIER TRAJEC-
TORY]
Answer the following question. Prior to this,
both the reasoner and the verifier have con-
ducted reasoning and verification regarding
this question. You are required to provide
the answer based on their respective reason-
ing processes. You should directly answer
the question between without detailed illus-
trations.
Question: [QUESTION]
12
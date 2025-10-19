# PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents

**Authors**: SHengjie Ma, Chenlong Deng, Jiaxin Mao, Jiadeng Huang, Teng Wang, Junjie Wu, Changwang Zhang, Jun wang

**Published**: 2025-10-13 02:45:37

**PDF URL**: [http://arxiv.org/pdf/2510.10931v1](http://arxiv.org/pdf/2510.10931v1)

## Abstract
Retrieval-augmented generation (RAG) agents, such as recent
DeepResearch-style systems, extend large language models (LLMs) with autonomous
information-seeking capabilities through external tools. While reinforcement
learning (RL) has enabled impressive multi-step reasoning, we identify a
previously overlooked failure mode, Tool-Call Hacking, where agents inflate
reward signals by issuing superficially correct tool calls without genuinely
leveraging the retrieved evidence. This results in (i) mode collapse into
repetitive reliance on a single source and (ii) spurious grounding, where
answers are only weakly supported by cited content.
  To address this, we propose Proof-of-Use (PoU), an evidence-grounded RL
framework that enforces verifiable causal links between retrieved evidence,
reasoning traces, and final answers. PoU operationalizes this through a unified
step-wise contract combining syntactic citation validation, perturbation-based
sensitivity rewards, and answer-evidence alignment objectives, ensuring that
tool usage remains both interpretable and functionally grounded.
  Across seven QA benchmarks spanning in-domain, out-of-domain, and
out-of-tool-distribution settings, PoU consistently outperforms strong
DeepResearch baselines in factual accuracy, evidence faithfulness, and
tool-routing balance. These findings highlight the necessity of grounding
RL-trained agents not merely in task outcomes but in the causal use of
retrieved information, offering a principled path toward trustworthy
retrieval-augmented reasoning.

## Full Text


<!-- PDF content starts -->

PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch
Agents
Shengjie Ma
Gaoling School of Artificial
Intelligence, Renmin University of
China
China
msj@ruc.edu.cnChenlong Deng
Gaoling School of Artificial
Intelligence, Renmin University of
China
China
dengchenlong@ruc.edu.cnJiaxin Mao
Gaoling School of Artificial
Intelligence, Renmin University of
China
China
maojiaxin@gmail.com
Jiadeng Huang
OPPO
China
huangjiadeng@oppo.comTeng Wang
OPPO
China
wt0318@connect.hku.hkJunjie Wu
OPPO
China
wujunjie1@oppo.com
Changwang Zhang
OPPO
China
changwangzhang@foxmail.comJun Wang
OPPO
China
junwang.lu@gmail.com
Abstract
Retrieval-augmented generation (RAG) agents—such as the recent
DeepResearch-style systems—extend large language models (LLMs)
with autonomous information-seeking capabilities via external
tools. While reinforcement learning (RL) has enabled impressive
multi-step reasoning performance, we reveal a previously over-
looked failure mode, termed Tool-Call Hacking: agents inflate re-
ward signals by issuing superficially correct tool calls without gen-
uinely using the retrieved evidence. This leads to (i)mode collapse
into repetitive reliance on a single source and (ii)spurious grounding,
where answers are weakly supported by cited content. To address
this, we propose Proof-of-Use (PoU)—an evidence-grounded RL
framework that enforces verifiable causal links between retrieved
evidence, reasoning traces, and final answers. PoU operational-
izes this through a unified step-wise contract combining syntactic
citation validation, perturbation-based sensitivity rewards, and an-
swer–evidence alignment objectives, ensuring that tool usage re-
mains both interpretable and functionally grounded. Across seven
QA benchmarks covering in-domain, out-of-domain, and two out-
of-tool-distribution tasks, PoU consistently outperforms strong
DeepResearch baselines in factual accuracy, evidence faithfulness,
and tool-routing balance. These findings highlight the necessity of
grounding RL-trained agents not merely in task outcomes, but in
thecausal useof retrieved information—offering a principled path
toward trustworthy retrieval-augmented reasoning.
1 Introduction
Retrieval-augmented generation (RAG) agents represent a recent
line of work that integrates external tools (e.g., web search)[ 8,23,29]
into the reasoning process. By combining parametric knowledge
with retrieved evidence, these agents have substantially advanced
complex question answering and research assistance. However, in
many knowledge-intensive settings, a single external knowledgesource is often insufficient [ 14]. This motivates scaling RAG to multi-
source environments. In such settings, diverse knowledge reposi-
tories are encapsulated as callable interfaces—commonly referred
to as “tools” in agent-based frameworks. Here, a “tool” denotes
a retrieval interface—e.g., web search, knowledge graph, domain
databases or private corpora—each with distinct coverage and re-
liability. The multi-source environment presents a compounded
challenge: not only must agents maintain calibrated intermediate
beliefs and ground their conclusions in retrieved evidence, but
they must also plan strategic tool routing—selecting which tools to
invoke and in what order.
Contrary to expectations, recent empirical studies in multi-source
RAG show that brute-force, all-in-one retrieval can outperform
prompt-based selective routing[ 26]. This suggests that naïve prompt-
based routing learned from shallow heuristics does not reliably
improve the effectiveness and informativeness of retrieved con-
tent; instead, it often amplifies systemic inefficiencies and disrupts
cross-source coordination, undermining the intended benefits of se-
lectivity. In other words, when routing is weak, selectivity becomes
a liability.
Building on this finding, we extend our observation to the be-
havioral patterns of RL-trained RAG agents. Prior works have ap-
plied reinforcement learning (RL) to enhance agents’ ability for
multi -step reasoning with retrieval tools [ 29]. However, we uncover
a recurrent failure pattern that we termTool -Call Hacking—a
domain -specific form of reward hacking [ 22]. Specifically, the agent
learns tool -calling behaviors that boost the observed reward signal
without genuinely improving grounding, which is highly suscepti-
ble to Goodhart’s Law[ 15]. We observe two consistent manifesta-
tions: (i) Mode collapse via tool overuse: policies default to a narrow
subset of tools. (ii) Hallucinated tool use: tool calls are made, but
the reasoning chain shows no causal dependence on the retrieved
evidence. In both cases, the agent superficially satisfies the reward
by producing spurious querying.arXiv:2510.10931v1  [cs.AI]  13 Oct 2025

Ma et al.
These issues highlight the need for stronger signals that foster
reasoning processes deeply integrated with tool interactions in
multi-source environments.
To this end, we propose an evidence-grounded agentic rein-
forcement learning framework, Proof-of-Use (PoU), that explic-
itly enforces the verifiable linkage between an agent’s reason-
ing process and the evidence it retrieves. Unlike conventional
agentic RL paradigms that optimize for task-level success with-
out scrutinizing how evidence is used, PoU reconstructs the en-
tire agent–environment interaction contract into a stepwise proof
schema. Within this schema, each reasoning step must first ren-
der an explicit verdict on the usefulness of retrieved evidence and
then cite the specific supporting sources—thereby making the evi-
dence–reasoning dependency a learnable and auditable interface.
This contract extends beyond intermediate reasoning: the same
evidence–reasoning linkage is propagated to the answer level,
where final predictions are judged by their factual consistency with
the aggregated cited evidence. Moreover, the training objective
itself is grounded in the contract—reward functions are computed
on citation validity, perturbation sensitivity, and answer–evidence
alignment, while rollout masking is selectively relaxed to preserve
truly referenced snippets. Through this tightly coupled design, PoU
transforms reasoning supervision from heuristic imitation into a
contract-driven optimization of verifiable grounding, enabling the
agent to progressively align its internal reasoning dynamics with
external factual dependencies.
Scope of Tools:In this work, we define “tools” as black-box[ 1]
proxiesthat interface with external knowledge sources: including
but not limited to open-web search, internal document collections,
structured APIs, or specialized knowledge bases. This reflects a
common abstraction in RAG systems, where tools are used to fetch
potentially relevant information conditioned on the current query
or reasoning context. While other forms of external tools (e.g., cal-
culators, code interpreters, or simulators) have been explored in
agent-based systems[ 25], our focus is on the multi-source retrieval
setting where the challenge lies in tool routing and evidence inte-
gration, not symbolic execution or procedural control. This scoped
definition allows us to isolate and study a critical bottleneck in
retrieval-augmented reasoning: the agent’s ability to strategically
call and utilize retrieval interfaces in a way that causally supports
final outputs. We believe this setup is both practically relevant and
theoretically rich, and we leave extension to broader tool types as
future work.
Contributions.Our work makes three key contributions:
•We identify and formalize thetool-call hackingphenome-
non of RAG agents in multi-source environments, revealing
how sparse rewards and ungrounded rollouts lead to tool
overuse and hallucinated grounding.
•We proposeProof-of-Use (PoU), a contract-driven and
evidence-grounded reinforcement learning framework that
tightly couples the agent’s interaction protocol with causal
evidence–reasoning–answer alignment, yielding verifiable
and interpretable reasoning behaviors.•Extensive experiments across open-domain and multi-hop
QA benchmarks demonstrate that PoU achieves strong per-
formance in accuracy and faithfulness.
2 Related Works
2.1 Reinforcement Learning based
Deepresearch Agent
Large language models (LLMs) are often constrained by the static
knowledge embedded in their parameters, making it difficult for
them to acquire the latest information in a timely manner. Conse-
quently, when facing time-sensitive or knowledge-intensive ques-
tions, they may produce inaccurate answers or hallucinations. Build-
ing on this, RAG agents have been extended beyond static retrieval
heuristics to more adaptive forms of reasoning and information
seeking. Recent research has explored applying reinforcement learn-
ing (RL) to enable models autonomously plan reasoning steps, de-
cide when to invoke what retrieval tools, and verify information
across different sources [ 23,29]. For instance, the Search-R1[ 8]
framework allows a model to flexibly invoke search engines during
answer generation, achieving a closed-loop “reason-and-search”
process. The R1-Searcher[ 23] method adopts a two-stage RL train-
ing approach that significantly enhances autonomous search capa-
bility. Similarly, ReSearch[ 5] enables LLMs to treat search as part
of the reasoning chain even without supervised reasoning step
labels. Furthermore, DeepResearcher[ 29] extends the training envi-
ronment to the open Internet, achieving end-to-end reinforcement
learning in real-world web settings.
2.2 Reinforcement Learning Algorithms for
Agent Training
In terms of optimization algorithms, mainstream approaches pri-
marily adopt policy-gradient methods, such as Proximal Policy
Optimization (PPO)[ 20] and its variants. The Search-R1[ 23] con-
ducted a detailed comparison between PPO and the GRPO[ 21] in
this setting. The conclusion is that GRPO converges faster, since
it reduces variance by using the mean reward of multiple sampled
outputs as a baseline. However, in later training stages, GRPO can
exhibit reward instability or even collapse. In contrast, PPO yields
a smoother and more stable training process, achieving slightly
higher final performance. Ultimately, both methods reach compara-
ble reward levels and effectively optimize Search-R1. Differently, R1-
Searcher[ 23] employs an improved version of the REINFORCE++[ 7]
algorithm (a model-free policy gradient approach) combined with
KL-divergence regularization and other stabilization techniques. In
addition, the previously mentioned loss masking of retrieval results
also contributes to stability—by excluding the environment’s tex-
tual feedback from gradient computation, the model avoids external
noise interfering with its gradients, thus improving convergence
and policy consistency. However, in our method, we tightly couple
the retrieved evidence, the model’s reasoning process, and the final
conclusion through an explicit contract mechanism. Under this for-
mulation, we selectively unmask the returned evidence segments
during training, thereby further encouraging the model to achieve
grounded alignment between evidence and reasoning and resolve
the tool-call hacking.

PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents
2.3 Reward Hacking in Reinforcement Learning
“Reward hacking” (a.k.a. specification gaming)—agents exploiting
flaws in objective design to score high reward while failing the
designer’s intent—was foregrounded as a core AI-safety problem in
Concrete Problems in AI Safety and popularized through empirical
case studies and taxonomies of gaming behaviors[ 2]. A formal treat-
ment views reward hacking as arising from misspecified proxies and
shows conditions under which “hackability” exists, alongside prin-
cipled defenses; complementary work analyzes reward tampering,
where agents manipulate the reward channel itself[ 22]. These phe-
nomena are closely linked to Goodhart’s Law—over-optimization
of a proxy decouples it from the true goal—and to goal misgen-
eralization, where systems pursue unintended objectives despite
correct training feedback, both documented across deep learning
and RL settings[17].
In post-training for LLMs, reward-model optimization intro-
duces additional failure modes: reward model over-optimization
can degrade human-judged quality as policies overfit artifacts of
the learned reward. Recent studies propose constrained RLHF and
composite-reward formulations to counteract these effects, while it-
erative RLHF empirically reduces over-optimization over successive
rounds but with diminishing returns[ 18]. Process-based supervi-
sion (e.g., PRMs or step-wise CoT rewards) can provide denser,
less gameable signals, though monitoring the reasoning process
also creates new attack surfaces where models obfuscate or “game
the monitor,” motivating hybrid outcome-plus-process objectives.
Orthogonal alignment methods such as Constitutional AI reduce
reliance on noisy human labels via principle-guided self-critique;
while not a direct fix for reward hacking, they reframe supervision
to limit exploitable gaps in reward design[3].
Modern RL-driven retrieval–reasoning agents operate in an
expanded action space where the policy can call external tools
(search engines, browsers, knowledge-base APIs, calculators, etc.)
and thereby influence the information source used to compute
downstream rewards. Therefore, we notice and formalize a class
of prevalence failure modes in this expanded setting as tool-call
hacking, and we aim to introduce a solution to this problem in this
paper.
3 Method
A PoU agent follows an iterative reasoning–acting–observing[28]
trajectory until sufficient information is accumulated to produce a
final answer.
3.1 Interaction protocol and proxies
3.1.1 Initialization.Each episode begins with auser query 𝑥
and a predefinedinitial promptthat provides explicit, structured
schemas for each callable tool, including their invocation formats
and argument descriptions.
To simulate realistic multi-source information environments for
retrieval–reasoning agents, we design and implement four cate-
gories of black-box retrieval proxies that collectively capture the
diversity of external knowledge access in agentic RAG systems.
Each proxy corresponds to a distinct knowledge source and interac-
tion modality, enabling the model to reason and retrieve in a unified
yet source-specific manner:•Web Search Proxy.This module enables the agent to ac-
cess open-domain information through keyword-based re-
trieval. Given a textual query, it returns a structured list
of webpage entries, each containing a title ,URL, and
snippet . For controlled evaluation, the current configu-
ration retrieves a fixed top- 𝑘set of documents per query,
ensuring reproducible coverage across runs.
•Web Browsing Proxy.This proxy performs in-depth in-
spection of given URLs of retrieved webpages and extracts
content relevant to the current query. It returns concise,
structured summaries to the reasoning model as evidence
for downstream use.
•Local Search Proxy.This proxy accesses pre-built, fixed
knowledge repositories, typically used in private or profes-
sional domains such as academic papers, technical reports,
enterprise data, or financial documents. Compared with
web search, these repositories feature higher knowledge
density and lower noise, making them particularly suitable
for deeper domain-specific tasks.
•Knowledge Graph Proxy.This proxy serves as a inter-
face to structured, graph-based knowledge stores. Rather
than employing a query-generation agent that issues exe-
cutable graph queries, we adopt a semantic KG agent that
abstracts subgraphs into concise, natural-language sum-
maries. This choice prioritizes robustness and generality
across heterogeneous graph schemas, allowing the proxy
to be more compatible with the model’s reasoning process.
The KG proxy thus functions as a lightweight retrieval util-
ity supplying structured yet readable context when graph
knowledge is required.
These tools are uniformly defined within the initialization prompt,
allowing the model to understand their expected arguments and
formats before the reasoning begins.
3.1.2 Proxy Response.After reasoning, the policy may issue a
structured tool call.
<tool_call>JSON_CALL</tool_call>
The environment executes the call and returns anormalized proxy
responsewith a unified schema:
<tool_response>REFERENCE_LIST_WITH_ID</tool_response>
Here, the id of each reference is automatically assigned. All
four proxies adopt this schema, so the model can cite evidence via
<ref>ID_1,ID_2</ref>regardless of the source.
Granularity.For theWeb Search Proxy, refitems are organized
at thewebpage level; typically contain a page title, URL, and snippet.
For theWeb Browsing,Local Search, andKnowledge Graphproxies,
refitems are returned at theparagraph (semantic-chunk) levelto
enable finer-grained grounding.
3.1.3 Reasoning Process.At the beginning of each iteration, the
agent firstly performs reasoning enclosed within a <think> tag. For
the first step, the model generates initial planning and reasoning
solely based on the user question. From the second step onward,
each <think> block must begin with a explicitverdict and citation
declaration, which is treated as aunified contract:
<helpful>yes|no</helpful><ref>id1,id2,...|null</ref>

Ma et al.
where <ref> cites the identifiers of retrieved documents from
the previous tool call’s structured output.
This mechanism forces grounding each reasoning step on prior
observations. From a probabilistic probing perspective, the hidden
state of <helpful> token quantifies the model’s internal judgment
of evidence utility, serving as a distributional anchor that aligns the
model’s internal activation patterns with the conditional likelihood
of evidence-relevant reasoning trajectories. By optimizing the prob-
ability mass of reasoning paths causally dependent on retrieved
evidence, it suppresses spurious thoughts that diverge from verifi-
able context. Meanwhile, the <ref> citations anchors the evidence
utility judgment to explicit external sources, establishing a probing-
visible linkage between internal representations and external tool
outputs. This allows each intermediate conclusion to be traceably
grounded in concrete evidence.
Together, the <helpful> and<ref> mechanisms constitute a
dual-channel grounding schemethat leverages their interaction as a
measurable and learnable interface between probabilistic alignment
and structured evidence-grounded reasoning.
When the model determines it has gathered sufficient evidence,
it terminates the loop and emits:
<answer> ... </answer>
as the final response to the user.
3.2 Proof-of-use Rewards
3.2.1 Unified Step Contract for Citation Reward.According to the
unified contract(denoted in 3.1.3) of each step, all proxy responses
return dictionaries containing a stable id, enabling deterministic
auditing. We define theCite Reward𝑅𝑒𝑤𝑎𝑟𝑑𝑡
𝑐𝑖𝑡𝑒in step𝑡:
𝑅𝑒𝑤𝑎𝑟𝑑𝑡
𝑐𝑖𝑡𝑒=(
+1,ifparse_ok∧consistency_ok∧ids_valid,
−1,otherwise.
(1)
Here, parse_ok= 1indicates syntactically valid and closed tags;
consistency_ok= 1requires(helpful=no⇒ref=null) and
(helpful=yes⇒ref≠null) ;ids_valid= 1means all cited IDs
exist in the proxy’s return list.
The rollout-level score is computed as the mean over all valid
steps:
𝑅𝑒𝑤𝑎𝑟𝑑 𝑐𝑖𝑡𝑒= 
Í𝑇
𝑡=2𝑅𝑒𝑤𝑎𝑟𝑑𝑡
𝑐𝑖𝑡𝑒
𝑇−1, 𝑇>1,
0, 𝑇=1,
where𝑇is the total number of reasoning steps. This ensures that
if the model answers directly without invoking any tool ( 𝑇=1), no
citation reward is given.
This design separates syntax compliance from citation consis-
tency, ensuring that every reasoning step remains bothverifiable
andtraceableto reasoning-evidence alignment.
3.3 Reference-Linked Evidence Perturbations
To evaluate whether the model’s decision truly depends on the cited
evidence rather than spurious features, we design a perturbation-
based reward that measures thedirectional sensitivityof the model’shelpfulness prediction𝑝 𝜃(yes)to evidence degradation:
𝑅𝑒𝑤𝑎𝑟𝑑 𝑝𝑡=𝑠𝑡·(𝑞′−𝑞),
𝑠𝑡=(
−1,YES case (degrade supportive evidence),
+1,NO case (inject semantic lure).(2)
Specifically, for the YES case, we degrade the cited evidence set
𝑆𝑡by replacing their contents with topic-unrelated snippets, and
observe the probability change Δ𝑞=𝑞′−𝑞, where𝑞=𝑝 𝜃(yes|
𝑅real)and𝑞′=𝑝𝜃(yes|𝑅 pert). A model that truly relies on the
cited evidence should decrease its helpfulness confidence ( Δ𝑞< 0)
after the evidence is corrupted.
For the complementary NO case, we apply the opposite pertur-
bation: when the model initially predicts the helpfulness token as
no, we randomly select one document from the proxy-returned
list andreplace its content with an LLM-generated snippet that is
semantically relevant to the querybut not necessarily factual. This
synthetic “semantic lure” mimics a plausible piece of supportive
evidence.
Since each rollout may contain multiple steps, the perturbation
reward is computed under a fixed budget𝐵:
𝐵=min(𝑇−1, 𝐵 max,Í
𝑡I[𝑅(𝑡)
𝑐𝑖𝑡𝑒=1]),
and we randomly sample 𝐵steps from those with 𝑅𝑒𝑤𝑎𝑟𝑑 𝑐𝑖𝑡𝑒=1
(from step 2 to𝑇) to apply perturbation. We set𝐵 max=1for main
experiments (thus 𝐵=1), and study 𝐵=2in ablations to assess
stability–cost trade-offs.
The final rollout-level perturbation reward is the mean of the
selected𝐵results.
3.4 Answer–Citation Alignment
The large action space and delayed rewards make the agent struggle
to assign credit properly, often resorting to shortcut guesses rather
than grounded reasoning. To overcome the sparse credit assignment
and shortcut reasoning issues, PoU directly ties the final reward to
the causal consistency between the produced answer and the cited
evidence through an Answer–Citation Alignment objective.
We compute the answer–evidence alignment reward as:
𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑐= 
0,ifÍ
𝑡I[𝑅𝑒𝑤𝑎𝑟𝑑(𝑡)
𝑐𝑖𝑡𝑒=1]=0
or𝑇=1,
J(𝑞, ˆ𝑎,E)∈{0,0.5,1},otherwise.
(3)
whereJdenotes the external LLM judge that scores whether ˆ𝑎
can be reasonably derived from the aggregated evidence set Eof
<helpful>yes</helpful>steps.
The final answer-level reward combines factual adequacy and
correctness as
𝑅𝑒𝑤𝑎𝑟𝑑 𝑎=𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑐+𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑛𝑠
2,(4)
where𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑛𝑠is the F1 score between the predicted and gold
answers.
We avoid overlap-based metrics (BLEU, ROUGE) for 𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑐
since they can be trivially optimized by surface copying rather than
genuine reasoning.

PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents
3.4.1 Overall Objective.The final reward integrates the three sig-
nals into a unified rollout-level objective:
𝑅𝑒𝑤𝑎𝑟𝑑 final=(𝑅𝑒𝑤𝑎𝑟𝑑 𝑐𝑖𝑡𝑒+𝑅𝑒𝑤𝑎𝑟𝑑 𝑝𝑡+𝑅𝑒𝑤𝑎𝑟𝑑 𝑎
3,if format is valid,
−1,otherwise.
(5)
where the overall format is considered valid only when all basic
format such as <think>, <tool_call> and <answer> pass the
syntactic checks.
All components are jointly optimized usingGroup Relative
Policy Optimization (GRPO)[ 21]. Unlike conventional agentic
RL pipelines that fully mask tool outputs during training, our rollout
retains theactually cited evidence snippets(those referenced via
<ref> ) unmasked. This aims to encourage the model to directly
condition on verified causal dependency between retrieved evidence
and reasoning.
3.5 Startup Fine-Tuning
To initialize a reliable agent under ourPoUreasoning framework,
we generate expert trajectories by a strong LLM (GPT-5). These
multi-step reasoning trajectories are reformatted into our standard-
izedinteraction schema, where each turn explicitly follows the PoU
contract.
For every simulated retrieval call, we emulate the output of dis-
tinct proxy tools ( web_search ,local_search ,browser ,KG_search ),
each returning paragraph-level evidence with stable identifiers, en-
suring deterministic grounding across different information sources.
To guarantee data reliability, we apply atwo-stage rejection
filterto all generated trajectories. A trajectory is retained only if:
(i)Every intermediate step passes the structural and logical val-
idators— syntactic correctness ( parse_ok ), internal consis-
tency ( consistency_ok ), and reference validity ( ids_valid );
(ii)The reasoning chain contains a non-trivial number of steps
(3≤𝑇≤ 10), encouraging multi-hop diversity while ex-
cluding degenerate or excessively long trajectories.
The resulting dataset thus consists ofverifiable, well-grounded
reasoning episodesthat adhere to our explicit stepwise contract. We
fine-tune the base model on this curated corpus tocold-start the
PoU agent.
4 Experiments
4.1 Datasets and Metrics
We evaluatePoUon a mixedin-/out-of-domain (ID/OOD)bench-
mark suite designed to probe open-domain QA under distributional
shifts.
In-Distribution (ID).Since our training data are drawn exclu-
sively fromHotpotQA[ 27] and2WikiMultihopQA[ 6], we report
ID performance on their official development splits.
Out-of-Distribution (OOD).To assess generalization beyond train-
ing domains, we evaluate on five additional datasets with dis-
tinct question styles and evidence structures:Natural Questions
(NQ)[ 11],TriviaQA (TQ)[ 9],MuSiQue[ 24],PopQA[ 16], and
Bamboogle[19].Sampling Protocol.For comparability across datasets, we uni-
formly sample512instances from each dev set of NQ, TQ, Hot-
potQA, 2Wiki, MuSiQue, and PopQA (using a fixed global seed),
and include all125items from the Bamboogle dev set. This yields a
total of3,197evaluation questions, maintaining balanced variance
across datasets while avoiding confounds from dataset size. Only
HotpotQA and 2Wiki are used for training; all others serve as strict
OOD evaluation.
4.2 Baselines
We benchmark PoU against a diverse set of recentdeep-research
agents, spanning both retrieval–reasoning paradigms and RL-based
search models.
(1) Retrieval–Reasoning RAG Baselines.Routing RAGadopts
a tool-routing design, where a controller dynamically selects the
appropriate retriever (e.g., local corpus, web, or KG) based on query
type and reasoning context.All-in-One RAG, by contrast, fuses
multiple retrievers into a unified interface, jointly encoding multi-
source evidence for direct reasoning without explicit routing. Both
follow theReAct[ 28] framework with three reasoning–retrieval
turns.
(2) RL-based Search Agents.Search-o1[ 13] performs multi-hop
search by iteratively generating sub-queries and retrieving concise
snippets rather than full documents.Search-R1[ 8] employs rein-
forcement learning for question answering, leveraging a Wikipedia
retriever; it has both base and instruct variants.R1-Searcher[ 23]
also uses RL, but restricts retrieval to Wikipedia via site-specific
filters, followed by multi-page summarization.
(3) Multi-Tool Deep-Research Agents.DeepResearcher[ 29] and
Chain-of-Agent (CoA)[ 12] represent the latest generation of
multi-source deep-research agents, operating in heterogeneous
tool environments. PoU builds upon this line, introducing explicit
proof-of-usefulnesssupervision and unified reasoning contracts to
ensure verifiable evidence grounding.
4.3 Baselines
To evaluate the effectiveness of PoU, we benchmark it against a
range of recent deep-research agents. Routing RAG and All-in-One
RAG represent two complementary retrieval-reasoning paradigms.
Routing RAG adopts a tool-routing design, where the controller
dynamically decides which retriever (e.g., local corpus, web search,
or knowledge graph) to invoke at each step according to the query
type and reasoning context. This enables flexible retrieval planning
but still relies on pre-defined routing heuristics. All-in-One RAG,
in contrast, fuses all retrievers into a unified interface that jointly
encodes multi-source evidence, letting the model reason directly
over concatenated contexts without explicit routing. All RAG base-
line following ReAct[ 28] with 3 turns, reflecting a fully integrated
retrieval-generation pipeline.
Beyond these two RAG baselines, Search-o1[ 13] follows a multi-
hop reasoning process where the model sequentially generates
intermediate queries and retrieves short evidence snippets rather
than full documents. Search-R1[ 8] employs reinforcement learn-
ing for question answering, using a retriever to gather Wikipedia-
based evidence, with base and instruct variants depending on the

Ma et al.
initialization of the actor model. R1-Searcher[ 23] also applies RL
but confines the search scope to Wikipedia by appending site-
specific constraints and summarizing multiple retrieved pages. Fi-
nally, DeepResearcher[ 29] and Chain-of-Agent (CoA)[ 12] are two
most recent deepresearch baselines with multi-source tool environ-
ments.
4.4 Implementation Details
We use GPT-4o-mini as the external judge model for evaluating an-
swer–evidence alignment and factual consistency. All other agentic
modules (browser, KG proxy, perturbations LLM) are implemented
with Qwen-3-8B, deployed via vLLM inference servers. The training
backbone model is Qwen-2.5-7B-Instruct. Prior to reinforcement
learning, supervised fine-tuning is performed with a batch size
= 256 for 3 epochs. For reinforcement learning, we adopt VERL
with the GRPO algorithm. Each update uses 8 parallel rollouts, dis-
tributed across 32×A100 GPUs. The 𝑅𝑒𝑤𝑎𝑟𝑑 𝑝𝑡budget𝐵is 1. The
maximum reasoning step is set to 10, and each retrieval call returns
the top-5 documents or chunks per query. The KG proxy retrieves
one-hop subgraph information from Wikidata and WikiPedia by
ToG-2[ 14] system, providing semantically condensed neighborhood
summaries as context. In the OODT experiments, we additionally in-
troduce a local biochemical Proxy, which is built from a biochemical
corpus derived from PubMed and serves as a local retrieval inter-
face, serving as the fixed local knowledge base for domain-specific
retrieval. The local_proxy is turn-off for non-medical domain tasks.
4.5 Main Results
Across all seven datasets—including both in-domain (HotpotQA,
2Wiki) and out-of-domain (NQ, TQ, MuSiQue, Bamboogle, PopQA)
benchmarks—our proposedProof-of-Usefulness (PoU)agent
consistently outperforms prior RAG and RL-trained baselines. This
advantage is not limited to the in-domain subset: on the out-of-
domain benchmarks, PoU achieves clear gains in both factual F1 and
LLM-as-judge (LM) metrics, demonstrating superior generalization
beyond its training distribution.
Interestingly, even on datasets such as NQ and TQ—which serve
as in-domain data for theDeepResearcherbaseline—our model
achieves higher overall scores. This observation suggests that PoU’s
improvement does not rely on memorizing domain-specific knowl-
edge, but rather stems from strongerevidence–reasoning alignment.
The model effectively learns to utilize retrieved evidence and to
generalize its reasoning strategies to novel domains, indicating
that PoU captures transferable reasoning competence rather than
dataset-specific experience.
We also observe thatAll-in-One RAGconsistently outperforms
Routing RAGacross all evaluation sets. This finding corrobo-
rates the observation inRAG in the Wildthat naïve or heuristic
routing—without explicit optimization or supervision—tends to be
inefficient or even counterproductive. Although routing appears
intuitively beneficial by narrowing the retrieval scope and reducing
search cost, it also risks prematurely discarding relevant informa-
tion.
In our context, the PoU agent operates in a multi-source environ-
ment where routing becomes an additional latent variable in the
decision space. Compared with traditional deep-research agents,this introduces a new layer of complexity: each reasoning step must
now decide not onlywhatto retrieve but alsowhereto retrieve
from. Such an expanded action space grows exponentially with the
number of information sources, leading to inefficient exploration
when the routing policy is not explicitly optimized—precisely the
phenomenon we aim to address through structured reasoning and
contract-based training under the PoU framework. By explicitly
encoding citation validity and reasoning consistency in every de-
cision step, PoU regularizes the search process and prevents the
agent from diverging in the enlarged action space. This design
enables the agent to leverage multi-source evidence effectively
without suffering from the instability and inefficiency commonly
observed in naive routing-based systems, ultimately yieldingsupe-
rior performance and generalization ability in multi-source
environments.
4.6 Ablation Study
4.6.1 Rewards Ablation.Figure 1 presents the ablation study of
PoU under different reward configurations, where each reward
component is selectively removed or adjusted. We observe that
increasing the perturbation budget to 𝐵=2leads to a more stable
training trajectory and a higher final performance. This confirms
the effectiveness of the perturbation-based reward 𝑅𝑒𝑤𝑎𝑟𝑑 𝑝𝑡: by
actively perturbing the cited evidence, the model learns to rely
on genuinely supportive information rather than superficially ac-
knowledging tool responses. Such perturbation helps prevent the
agent from “pretending” to use tool feedback, thereby maintaining
a healthy and optimizable training regime instead of exploiting
spurious reward shortcuts. However, since the rollout cost of GRPO
scales approximately multiplicatively with the perturbation bud-
get, we adopt 𝐵=1as a balanced trade-off between computational
efficiency and stability.
Furthermore, removing the answer–citation alignment term
𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑐results in unstable training and a sudden collapse in later
stages. This indicates that linking the final answer with intermedi-
ate evidence provides an essential smoothing and grounding signal
for the policy. Without this alignment, the model easily drifts away
from verifiable reasoning behaviors, demonstrating the critical role
of answer–evidence coupling in maintaining robust optimization.
4.6.2 Out-of-domain (OODT) Experiments.To further validate the
generalization ability of our approach in vary environments, we con-
ducted additional experiments under an out-of-domain (OOD+OODT)
setting. Specifically, we introduced afourth proxy—abiomedical lo-
cal search proxy—which wasnever seen during training. We then
compared our model with existing baselines on two biomedical QA
datasets,BioASQ[ 10] andPQArefEval[ 4]. This setup constitutes
anOOD+OODTscenario (out-of-domain data and out-of-domain
tools), enabling us to assess how well the model generalizes when
both the data distribution and the available tool set differ from
those encountered during training.
As shown in Table 3, even thoughDeepResearcher+was trained
in a more complex 3-tool environment, its performance is notably in-
ferior to its original 2-tool counterpart (DeepResearcher) on both
biomedical datasets. In contrast, our proposedPoUmodel achieves
the best results by a clear margin, demonstrating strongdomain-
level and tool-level generalization. This finding highlights an

PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents
Model Env. HotpotQA 2Wiki
F1 LM F1 LM
RAG
Routing RAG Multi-source 30.8 36.0 14.4 18.2
All-in-one RAG Multi-source 34.5 40.2 23.2 26.4
Search-o1* Local 31.6 40.8 28.6 32.8
Search-o1 Web Search 33.0 42.4 30.9 37.7
RL Training RAG
Search-R1-base Local 55.963.0 44.6 47.9
Search-R1-instruct Local 45.7 52.5 43.4 48.8
R1-Searcher Web Search 44.8 53.1 59.4 65.8
CoA Multi-source — 43.9 — 50.2
DeepResearcher Web Search 52.8 64.3 59.7 66.6
Ours
PoUMulti-source 53.0 65.4 67.8 76.8
Table 1: In-domain evaluation on HotpotQA and 2Wiki datasets. All results are reported with F1 and LLM-based evaluation
(LM) metrics. Bold numbers indicate the best performance in each column. Underlined numbers indicate the second best.
Method Env. NQ TQ MuSiQue Bamboogle PopQA
F1 LM F1 LM F1 LM F1 LM F1 LM
RAG
Routing RAG Multi-source 34.8 42.9 51.0 59.6 6.5 8.2 17.0 17.4 35.1 39.9
All-in-one RAG Multi-source 37.5 48.3 60.2 68.8 9.7 11.8 23.5 25.1 46.9 48.8
Search-o1* Local 34.5 57.4 52.6 61.1 16.8 21.3 35.8 38.4 36.9 42.4
Search-o1 Web Search 32.4 55.1 58.9 69.5 14.7 19.7 46.6 53.6 38.3 43.4
RL Training RAG
Search-R1-base Local 45.4 60.0 71.9 76.2 26.7 27.5 56.5 57.6 43.2 47.0
Search-R1-instruct Local 33.1 49.6 44.7 49.2 26.5 28.3 45.0 47.2 43.0 44.5
R1-Searcher Web Search 35.4 52.3 73.1 79.1 22.8 25.6 64.8 65.6 42.7 43.4
CoA Multi-source — 43.9 — 63.3 — 22.3 — 49.6 — 46.5
DeepResearcherWeb Search 39.6 61.9 78.485.0 27.1 29.3 71.072.8 48.5 52.7
Ours
PoUMulti-source 45.5 67.2 77.6 89.6 29.0 34.1 69.5 74.6 55.7 61.2
Table 2: Performance comparison on five out-of-domain QA benchmarks: NQ, TQ, MuSiQue, Bamboogle, and PopQA. All results
are reported with F1 and LLM-based evaluation (LM) metrics. Bold numbers indicate the best performance in each column.
Underlined numbers indicate the second best.
Dataset Zero-shot RAG DeepResearcher+ DeepResearcher PoU (Ours)
BioASQ 13.5 25.8 37.1 38.945.6
PQArefEval 18.4 31.5 42.4 42.148.7
Table 3: OOD+OODT evaluation on biomedical QA datasets (BioASQ, PQArefEval). The experiments introduce a fourth proxy, a
biomedical local search proxy, to test generalization across both domain and toolset shifts.
important phenomenon: as the number of available tools increases, the action space of the agent expands combinatorially. Without

Ma et al.
Figure 1: Ablation study of PoU. Comparison under differ-
ent reward configurations: default perturbation budget 𝐵=1,
extended𝐵=2, and variants removing 𝑅𝑒𝑤𝑎𝑟𝑑 𝑝𝑡or𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑐.
Increasing the perturbation budget to 𝐵=2yields more stable
convergence and a higher upper bound, validating the effec-
tiveness of the perturbation-based reward 𝑅𝑒𝑤𝑎𝑟𝑑 𝑝𝑡. Pertur-
bation prevents the agent from “pretending” to read tool feed-
back and keeps optimization in a healthy, learnable regime
rather than exploiting spurious shortcuts. However, since
GRPO rollout cost scales multiplicatively with 𝐵, we adopt
𝐵=1as a practical trade-off. Removing the answer–citation
alignment term 𝑅𝑒𝑤𝑎𝑟𝑑 𝑎𝑐causes the model to collapse in later
stages, confirming that explicit evidence–answer alignment
provides a crucial smoothing and grounding signal for stable
training.
explicit process-level reward signals, such enlarged search space
can easily lead to policy collapse or unstable reasoning behavior.
In contrast, the PoU training paradigm—through explicit evidence-
reasoning alignment and structured reward design—successfully
stabilizes the learning process and preserves robustness under tool
and domain distribution shifts, indicating theTool-call Hacking
is significantly mitigated.
4.6.3 Tool-Call Behavior Analysis.To assess how expanding the
tool set changes agent behavior, we trainedDeepresearcher+with
one more tool (Knowledge Graph) from scratch under the same
data and hyperparameters as the originalDeepresearcher, then
uniformly sampled1 ,000queries from seven test sets and counted
tool-call usage across three categories:Web Search (Web),Knowledge
Graph (KG), andWeb Browser (Browser). Table 4 summarizes the
call ratios.
The results in Table 4 reveal distinct behavioral patterns across
models. When trained with only theformatandanswerrewards
(as in the original Deepresearcher setup), the agent tends to overuseTable 4: Tool-call ratio (%) over1 ,000sampled queries from
seven test sets. Columns are Web Search (Web), Knowledge
Graph(KG), and Web Browser (Browser). Deepresearcher has
no KG tool (0.0 ).
Model Web KG Browser
Deepresearcher 88.4 - 11.6
Deepresearcher+ 94.6 2.2 3.2
PoU 51.8 27.5 20.7
available tools once the tool set expands. After adding the Knowl-
edge Graph interface,Deepresearcher+exhibits a pronounced
bias toward redundant Web Search calls (94.6%) and a near disap-
pearance of Web Browser usage (3.2%). This behavior indicates a
collapse of tool specialization: although the browser is logically
designed to serve as a follow-up step for Web Search—retrieving
detailed content from previously retrieved URLs—the model largely
ignores this expected hierarchy and instead repeatedly queries the
Web Search proxy. Such uncontrolled escalation of retrieval ac-
tions suggests severeTool-call Hackingwhen training lacks explicit
evidence-usage constraints.
In contrast, thePoU-trainedagent demonstrates a markedly
more balanced tool distribution. Each proxy—Web Search, Knowl-
edge Graph, and Browser—maintains a complementary role (51.8%,
27.5%, 20.7%, respectively), implying that the model actively sched-
ules heterogeneous retrieval utilities based on their strengths: the
Web Search proxy for broad discovery, the Knowledge Graph for
exploratory search, and the Browser for fine-grained verification.
This balanced behavior highlights that the PoU reward design effec-
tively regularizes tool usage, encouraging agents to retrieve, reason,
and cite evidence in a coordinated manner rather than relying on
repetitive search actions.
4.6.4 Extreme Ablation.To further examine robustness and causal
grounding, we replaced every <tool_response> with the token
“content”, removing all semantics while keeping structure.
Model Behavior under "content" responses
PoU <think> "No evidence; Try other tools." →
<helpful>no</helpful>→<ref>null</ref>
→<tool_call>KGSearch(𝑞).
⇒ Detects missing evidence and switches
tools/parameters.
Deepresearcher+ <think> "The article says X (hallucinated). Now I need
to search(𝑞′)"→<tool_call> WebSearch(𝑞′)→
<tool_response>“content”.
⇒Hallucinates evidence and continues fictitious rea-
soning.
Table 5: Extreme ablation test where all tool responses are
replaced by “content”. Only PoU detects missing evidence
and adapts its strategy.

PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents
5 Conclusion
In this work we identifiedtool -call hackingas a pervasive failure
mode in multi -source retrieval -augmented agents and introduced
Proof -of-Use (PoU)—a contract -driven, evidence -grounded RL
framework that enforces step -wise alignment between retrieved
evidence, intermediate reasoning, and final answers. In PoU this con-
tract contains explicit helpfulness/citation declarations, a citation -format
reward, reference -linked perturbation sensitivity, and an answer–citation
alignment objective, coupled with selective unmasking of actu-
ally cited snippets during training. Across diverse in -domain and
out-of-domain QA benchmarks, PoU yields consistent gains in both
F1 and LLM -as-judge scores over strong deep -research baselines,
while exhibiting balanced and purposeful tool routing and robust-
ness under domain even tools shifts. Limitations include reliance on
an external LLM judge for alignment scoring, and a current focus on
retrieval tools rather than procedural tools. By the ablation studies,
we clearly demonstrate the rationality and necessity of each reward
design. Future work will explore learned or weakly supervised
judges, adaptive perturbation curricula, cost -and latency -aware
routing, extensions to calculators/coders/simulators, and robustness
against adversarial or privacy -sensitive tool outputs. Overall, PoU
offers a simple and auditable path toward trustworthy multi -source
reasoning by making evidence use not only observable but optimiz-
able.
Ethical Use of Data and Informed Consent
This study does not involve human participants or personally iden-
tifiable information. All datasets used (e.g., HotpotQA, PubMed,
and other publicly available corpora) are released under open re-
search licenses and comply with their respective terms of use. All
web content accessed complied with robots.txt and site terms, and
was used solely for non-commercial research. No additional ethical
approval was required.
References
[1]Rohan Ajwani, Shashidhar Reddy Javaji, Frank Rudzicz, and Zining Zhu.
2024. LLM-Generated Black-box Explanations Can Be Adversarially Helpful.
arXiv:2405.06800 [cs.CL] https://arxiv.org/abs/2405.06800
[2]Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman,
and Dan Mané. 2016. Concrete Problems in AI Safety. arXiv:1606.06565 [cs.AI]
https://arxiv.org/abs/1606.06565
[3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion,
Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon,
Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn
Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr,
Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite,
Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado,
Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston,
Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-
Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac
Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish,
Tom Brown, and Jared Kaplan. 2022. Constitutional AI: Harmlessness from AI
Feedback. arXiv:2212.08073 [cs.CL] https://arxiv.org/abs/2212.08073
[4]Bojana Bašaragin, Adela Ljajić, Darija Medvecki, Lorenzo Cassano, Miloš
Košprdić, and Nikola Milošević. 2024. How do you know that? Teaching
Generative Language Models to Reference Answers to Biomedical Questions.
arXiv:2407.05015 [cs.CL] https://arxiv.org/abs/2407.05015
[5] Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng
Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan
Zhou, and Weipeng Chen. 2025. ReSearch: Learning to Reason with Search for
LLMs via Reinforcement Learning. arXiv:2503.19470 [cs.AI] https://arxiv.org/
abs/2503.19470[6] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational Lin-
guistics. International Committee on Computational Linguistics, Barcelona, Spain
(Online), 6609–6625. https://www.aclweb.org/anthology/2020.coling-main.580
[7] Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. 2025. REINFORCE++: An
Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models.
arXiv:2501.03262 [cs.CL] https://arxiv.org/abs/2501.03262
[8]Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516 [cs.CL]
https://arxiv.org/abs/2503.09516
[9] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. TriviaQA:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion. arXiv:1705.03551 [cs.CL] https://arxiv.org/abs/1705.03551
[10] Anastasia Krithara, Anastasios Nentidis, Konstantinos Bougiatiotis, and Georgios
Paliouras. 2023. BioASQ-QA: A manually curated corpus for Biomedical Question
Answering.Scientific Data10, 1 (2023), 170.
[11] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob
Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: a
Benchmark for Question Answering Research.Transactions of the Association of
Computational Linguistics(2019).
[12] Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang,
Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu,
Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tiannan Wang,
Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru
Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang,
and Wangchunshu Zhou. 2025. Chain-of-Agents: End-to-End Agent Foundation
Models via Multi-Agent Distillation and Agentic RL. arXiv:2508.13167 [cs.AI]
https://arxiv.org/abs/2508.13167
[13] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models.ArXivabs/2501.05366 (2025). https://api.semanticscholar.
org/CorpusID:275405676
[14] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation.
InThe Thirteenth International Conference on Learning Representations. https:
//openreview.net/forum?id=oFBu7qaZpS
[15] Adrien Majka and El-Mahdi El-Mhamdi. 2025. The Strong, Weak and Benign
Goodhart’s law. An independence-free and paradigm-agnostic formalisation.
arXiv:2505.23445 [stat.ML] https://arxiv.org/abs/2505.23445
[16] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Han-
naneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating Effec-
tiveness of Parametric and Non-Parametric Memories. arXiv:2212.10511 [cs.CL]
https://arxiv.org/abs/2212.10511
[17] David Manheim and Scott Garrabrant. 2019. Categorizing Variants of Goodhart’s
Law. arXiv:1803.04585 [cs.AI] https://arxiv.org/abs/1803.04585
[18] Ted Moskovitz, Aaditya K Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhut-
dinov, Anca Dragan, and Stephen Marcus McAleer. 2024. Confronting Reward
Model Overoptimization with Constrained RLHF. InThe Twelfth International
Conference on Learning Representations. https://openreview.net/forum?id=
gkfUvn0fLU
[19] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. arXiv:2210.03350 [cs.CL] https://arxiv.org/abs/2210.03350
[20] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347 [cs.LG] https:
//arxiv.org/abs/1707.06347
[21] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeek-
Math: Pushing the Limits of Mathematical Reasoning in Open Language Models.
arXiv:2402.03300 [cs.CL] https://arxiv.org/abs/2402.03300
[22] Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger.
2025. Defining and Characterizing Reward Hacking. arXiv:2209.13085 [cs.LG]
https://arxiv.org/abs/2209.13085
[23] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin
Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search
Capability in LLMs via Reinforcement Learning. arXiv:2503.05592 [cs.AI] https:
//arxiv.org/abs/2503.05592
[24] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
arXiv:2108.00573 [cs.CL] https://arxiv.org/abs/2108.00573
[25] Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelović,
and Jakob Nikolas Kather. 2025. LLM Agents Making Agent Tools.
arXiv:2502.11705 [cs.CL] https://arxiv.org/abs/2502.11705

Ma et al.
[26] Ran Xu, Yuchen Zhuang, Yue Yu, Haoyu Wang, Wenqi Shi, and Carl Yang. 2025.
RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge
Retrieval Augmentation. arXiv:2507.20059 [cs.CL] https://arxiv.org/abs/2507.
20059
[27] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600 [cs.CL]
https://arxiv.org/abs/1809.09600[28] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629 [cs.CL] https://arxiv.org/abs/2210.03629
[29] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pen-
grui Lu, and Pengfei Liu. 2025. DeepResearcher: Scaling Deep Research via
Reinforcement Learning in Real-world Environments. arXiv:2504.03160 [cs.AI]
https://arxiv.org/abs/2504.03160
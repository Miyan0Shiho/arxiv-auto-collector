# Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts

**Authors**: Sha Li, Naren Ramakrishnan

**Published**: 2026-04-01 13:45:52

**PDF URL**: [https://arxiv.org/pdf/2604.00901v1](https://arxiv.org/pdf/2604.00901v1)

## Abstract
Multi-agent Retrieval-Augmented Generation (RAG), wherein each agent takes on a specific role, supports hard queries that require multiple steps and sources, or complex reasoning. Existing approaches, however, rely on static agent behaviors and fixed orchestration strategies, leading to brittle performance on diverse, multi-hop tasks. We identify two key limitations: the lack of continuously adaptive orchestration mechanisms and the absence of behavior-level learning for individual agents. To this end, we propose HERA, a hierarchical framework that jointly evolves multi-agent orchestration and role-specific agent prompts. At the global level, HERA optimizes query-specific agent topologies through reward-guided sampling and experience accumulation. At the local level, Role-Aware Prompt Evolution refines agent behaviors via credit assignment and dual-axes adaptation along operational and behavioral principles, enabling targeted, role-conditioned improvements. On six knowledge-intensive benchmarks, HERA achieves an average improvement of 38.69\% over recent baselines while maintaining robust generalization and token efficiency. Topological analyses reveal emergent self-organization, where sparse exploration yields compact, high-utility multi-agent networks, demonstrating both efficient coordination and robust reasoning.

## Full Text


<!-- PDF content starts -->

Preprint. Under review.
Experience as a Compass: Multi-agent RAG with Evolving
Orchestration and Agent Prompts
Sha Li
Virginia Tech
shal@vt.eduNaren Ramakrishnan
Virginia Tech
naren@vt.edu
Abstract
Multi-agent Retrieval-Augmented Generation (RAG), wherein each agent
takes on a specific role, supports hard queries that require multiple steps
and sources, or complex reasoning. Existing approaches, however, rely on
static agent behaviors and fixed orchestration strategies, leading to brittle
performance on diverse, multi-hop tasks. We identify two key limitations:
the lack of continuously adaptive orchestration mechanisms and the ab-
sence of behavior-level learning for individual agents. To this end, we
proposeHERA, a hierarchical framework that jointly evolves multi-agent
orchestration and role-specific agent prompts. At the global level,HERA
optimizes query-specific agent topologies through reward-guided sam-
pling and experience accumulation. At the local level, Role-Aware Prompt
Evolution refines agent behaviors via credit assignment and dual-axes
adaptation along operational and behavioral principles, enabling targeted,
role-conditioned improvements. On six knowledge-intensive benchmarks,
HERAachieves an average improvement of 38.69% over recent baselines
while maintaining robust generalization and token efficiency. Topological
analyses reveal emergent self-organization, where sparse exploration yields
compact, high-utility multi-agent networks, demonstrating both efficient
coordination and robust reasoning.
1 Introduction
Recent advances in Large Language Model (LLM)-based Multi-Agent Systems (Li et al.,
2024) have enhanced Retrieval-Augmented Generation (RAG) (Lewis et al., 2020; Gao et al.,
2023b; Jiang et al., 2023) through collaborating specialized agents (Nguyen et al., 2025;
Liu et al., 2025; Qian et al., 2025; Chen et al., 2025b), extending tool use (Zhuang et al.,
2024; Schick et al., 2023; Jin et al., 2025) and memory integration (Hu et al., 2025; Guti ´errez
et al., 2025; Yang et al., 2025b). These systems are promising for complex and long-context
reasoning, where effective coordination between retrieval and reasoning is critical.
Despite their promise, existing multi-agent RAG frameworks are fundamentally constrained
by static or weakly adaptive orchestration. Most prior work relies on fixed or sequential
pipelines (Gamage et al., 2024; Liu et al., 2025; Yao et al., 2022; Qian et al., 2025; Chen et al.,
2025b), which fails to accommodate query-dependent variation in reasoning and retrieval
complexity, resulting in insufficient evidence collection or unnecessary retrieval overhead.
Besides, these systems are highly susceptible to error propagation (Shen et al., 2025; Pei
et al., 2025), where mistakes compound across multi-turn interactions and long-horizon
agent coordination. Since these errors are inherently compositional, reliance on global
supervision signals (e.g., final answer correctness) (Chen et al., 2025b; Song et al., 2025a;b)
provides limited attribution to specific failure sources, hindering targeted improvements.
Moreover, mainstream training-based optimization remains costly and inflexible, typically
requiring large trajectory datasets (Zhu et al., 2024b; Chen et al., 2025b;c; Zhu et al., 2024a;
Jin et al., 2025), which constrains scalability and limits continual adaptation in dynamic
knowledge environments (Tang et al., 2025). As query distributions and underlying corpora
evolve, fixed reasoning and coordination strategies (Jeong et al., 2024b) lead to stagnant
performance and brittle generalization to novel queries and retrieval challenges.
1arXiv:2604.00901v1  [cs.AI]  1 Apr 2026

Preprint. Under review.
To address these limitations, we argue that, an effective multi-agent RAG system must (i)
adapt its coordination strategies to query complexity, (ii) enable targeted and role-specific
agent improvement, (iii) mitigate error propagation, and (iv) support continual adaptation
without costly retraining. To this end, we proposeHERA, a Hierarchical Evolvution RAG
framework that achieves these goals bydistilling execution experience as in-context knowledge
andreshaping agent behaviors by prompt evolutionwithout parameter updates.
HERAis organized as a three-layer hierarchy that jointly evolves global orchestration strate-
gies and local agent behaviors using accumulated experiential and reflective knowledge
(Figure .1). At the top, a centralizedorchestratorgenerates query-specific execution plans
in a single step. This holistic orchestration (Ke et al., 2026) enables the system to reason
globally about the coordination of retrieval and reasoning, producing plans tailored to the
complexity and requirements of each query. At the bottom,specialized agentsexecute sub-
tasks grounded in interaction with others.HERAemploys role-aware prompt optimization
to drive targeted improvements for each agent’s unique responsibilities, avoiding generic
updates and mitigating errors arising from misaligned multi-agent behaviors. Bridging
these layers, theexperience librarycaptures reflective insights from both successful and failed
execution trajectories. These insights provide diagnostic information (Agrawal et al., 2026),
enabling semantic credit assignment for global learning and guiding agent-level refinement.
AcrossHERA, continuously evolving experience acts as a compass: it informs the orches-
trator to improve high-level coordination strategies and drives progressive, role-specific
enhancement of agent behavior, improving robustness and reducing cascading errors.
Our contributions are summarized as follow: (i) We introduceHERA, a hierarchical frame-
work enabling dual-level evolution - experience-guided orchestration and role-aware
prompt adaptation, shifting policy optimization from parameter updates to experience-
driven contextual adaptation. (ii) We design topology-based metrics to quantitatively
characterize the evolution of agent interaction topologies and track dynamics. (iii) Empirical
results on six knowledge-intensive benchmarks demonstrate thatHERAoutperforms recent
SOTA by an average of 38.69% while maintaining token efficiency.
2 Preliminary and Problem Statement
We study the problem of dual-level evolution in multi-agent RAG for knowledge-intensive
tasks. Let Qdenote a distribution over queries, Ythe answer space, and Da corpus (e.g.
Wikipedia) providing supporting information. We formalize a whole system as a tuple:
R=⟨O,N,L,T,E,A,S,Ψ,τ,Q,Y,D⟩
where Ois a central orchestrator, Nis a set of specialized agents, Eis an experience library, S
is the state space, and Ais the action space. Each agent Niis abstracted as a tuple (πi,ρi,Ti),
where πi∈ L is the underlying LLM, ρiis a role-specific prompt, and Tidenotes the tools
available to the agent. The overall action space Aincludes both reasoning operations and
tool invocations. The system evolves according to the transition dynamics Ψ(s t+1|st,at),
and agent executions produce trajectoriesτ= (s 0,a0,s1,a1, . . . ,s T).
Given a query q, the orchestrator produces a multi-agent interaction topology conditioning
on the experience library: Γ∼π O(·|q,E,N), where Γspecifies the subset of agents, their
execution order (sequential or parallel), and dependency. Agents execute according to Γ,
producing a trajectory τ, and the final answer y∈ Y is synthesized from the information
aggregated along the trajectory, typically based on the terminal states T.
Our goal is to jointly optimize the orchestration policy πOand role-specific prompts ρi
while keeping the underlying LLM parametersπ ifrozen.
3HERA: H ierarchical E volution of Multi-agent RA G
HERAjointly evolves global orchestration and agent prompts through experience and
reflection. As shown in Figure 1, it comprises three layers, each corresponding to a different
level of optimization: theglobal orchestratoroptimizes high-level coordination strategies, the
experience libraryaccumulates and refines trajectory-level knowledge to guide learning, and
theexecution agentsoptimize role-specific performance through prompt evolution, per their
operational roles and behavioral principles.
2

Preprint. Under review.
Figure 1: Overview ofHERA. A hierarchical framework that jointly evolves orchestration
strategies, the experience library, and agent prompts.
3.1 Orchestrator: Structure-Level Policy Optimization
The orchestrator’s optimization is inspired by Group Relative Policy Optimization (GRPO)
(Shao et al., 2024; Cai et al., 2025), which updates a policy by comparing sampled actions
within a group. Unlike the training-free GRPO (Cai et al., 2025), we extend the idea be-
yond token-level generation, lifting optimizationfrom token generation to cooperation topology.
Operating over a structured action space of agent topologies Γ, the orchestrator performs
group-based optimization at the level of global strategy. Given a query q, it samples a
group of candidate agent sequences. Each sequence is executed by invoking the correspond-
ing agents, producing a trajectory τi, which is subsequently evaluated to obtain a reward.
Instead of relying on scalar advantages, we adopt a hierarchical comparison: trajectories
are first ranked by downstream task performance (e.g., F1 score), and then by efficiency,
measured as the total number of input and output tokens consumed across all agents. The
orchestrator is then prompted to articulate the reasons for relative success and failure across
the group, producing a set of natural language insights I=π O({τ i}G
i=1). To improve
diagnostic precision, we focus on groups explicitly containing both successful and failed.
trajectories. which serve as group-relative semantic advantages. These semantic advantages
encode structured knowledge about effective reasoning strategies, agent interactions, and
failure modes, effectively replacing numerical gradients with interpretable and compo-
sitional signals. As such, the orchestrator governs high-level structure and coordination
(macro) instead of the fine-grained textual outputs (micro). These insights, curated through
selective consolidation, form the core of the experience library (§3.2).
3.2 Experience Library
Storing distilled experience as retrievable in-context demonstrations has been shown to
effectively shape future behavior (Zhao et al., 2024; Ouyang et al., 2025; Xu et al., 2025b).
Building on this principle,HERAemploys an Experience Library E, which accumulates se-
mantic advantages derived from reflective comparisons of successful and failed trajectories,
serving as both an episodic memory and an optimization medium for the orchestrator.
Eis charatcerized with aProfile–Insight–Utilitystructure (c,z,u), where cdenotes thequery
characteristics or types, zis the natural languageinsight, and urepresents theutility, measuring
the empirical success rate of z- that is, how often the insight is successfully applied in
subsequent agent orchestration. Eis updated online through consolidation operations: ADD
inserts insights that are distinct from any existing entries; MERGE combines semantically
similar or complementary entries, PRUNE removes conflicting or low-utility insights to
maintain generalizable insights while preventing uncontrolled growth, and KEEP leaves E
unchanged.
3

Preprint. Under review.
Experience as a Prior.Given a query q, the orchestrator first analyzes its characteristics and
retrieves a set of relevant insights from E. Retrieval balances two objectives: maximizing
empiricalutility u, which favors insights that have historically led to successful coordination,
and maintainingdiversity, achieved by selecting insights that are semantically distinct from
those already chosen and that have lower prior usage frequency. The retrieved insights
inform the orchestrator’s selection and sequencing of agents, guiding multi-step execution
toward high-performing strategies. By combining high-utility and diverse insights, the
orchestrator biases execution to maximize task accuracy while minimizing token consump-
tion. In this manner,Eserves as a structured, experience-driven prior, enabling contextual
policy improvement and without modifying the underlying LLM.
3.3 Theoretical Interpretation
TheHERAorchestrator admits a naturalExpectation-Maximization (EM)interpretation
over discrete reasoning programs. In theE-step, trajectories are sampled under the current
policy, approximating the posterior over high-reward reasoning topologies. In theM-step,
semantic insights are extracted and consolidated into the experience library, updating
a latent representation of the policy. Further, using a frozen LLM introduces implicit
regularization, analogous to KL-constrained policy optimization. As updates occur via
contextual augmentation instead of parameter changes, the induced policy remains close to
the original model distribution. This formulation highlights thatHERAperforms energy-
based reweighting of candidates, ensuring stable and consistent improvement. We provide
detailed theoretical analysis in Appendix A.
3.4 Role-aware Prompt Evolution (RoPE)
Globally strategy cannot preventcorrective(agents ignore actionable error signals and
repeat mistakes) orenhancement failures(agents persist with suboptimal strategies despite
successful execution) (Zhang et al., 2025a) in multi-agent systems, motivating ourRole-aware
Prompt Evolution(RoPE).
3.4.1 Prompt Evolution
A key challenge in multi-agent systems is attributing global trajectory outcomes to indi-
vidual agents, especially when inter-agent dependencies obscure direct credit assignment.
RoPE addresses this by targeting agents identified asunderperformingby the orchestrator,
who evaluates complete trajectories. Formally, an agent is considered underperforming if
the orchestrator identifies it as the primary contributor to the trajectory’s failure.
To support systematic improvement, we maintain a buffer for each agent containing its
recent failed trajectories. This buffer enables RoPE to analyze recurring errors, identify
consistent failure patterns, and provide contextual evidence for targeted prompt refinement.
Given an underperforming agent Niat time t, we retrieve its recent failed trajectories,
then we generate prompt variants, and reexecutet the original whole agent trajectories
with these prompt variants. Then RoPE performs contrastive analysis to extract actionable
improvements along two complementary dimensions: (i)operational rules ( ∆ρop
i): short-term
corrective behaviors derived from recent failures, and (ii)behavioral principles ( ∆ρbp
i): long-
term strategies distilled from patterns across multiple trajectories. Formally, the prompt
update can be decomposed as:
∆ρi=∆ρop
i+∆ρbp
i
where ∆ρop
iaddressesimmediate failure correctionand ∆ρbp
ipromotesstrategic generalization.
Prompt variants are generated along behavioral axes such as thoroughness, risk sensitivity,
error correction and heuristic injection, enabling structured exploration to optimize accuracy,
robustness, and computational cost (see Appendix B for prompt details).
3.4.2 Prompt Consolidation
Following variant evaluation, selected updates are integrated into the agent’s prompt via
prompt consolidation, which prunes redundant or low-utility instructions to maintain a
compact and coherent representation. Formally, this can be expressed as a projection:
ρt+1
i=ΠC 
ρt
i⊕∆ρ i
4

Preprint. Under review.
where Crepresents constraints on prompt length and coherence. With the underlying LLM
fixed, prompt evolution induces a controlled shift in the agent’s effective policy:
πNnew
i(y|x)∝πNbase
i(y|x)·exp(fρ i(x,y))
where x,ydenote the input and output context of an agent respectively, and πNnew
irepresents
its conditional generation policy. The prompt-induced term fρi(x,y)biases this mapping by
favoring outputs ythat better align with the updated role-specific guidance. This induces
an implicit regularization effect, analogous to KL constraints, ensuring stable and consistent
policy updates.
3.5 Topology Mutation
While prompt evolution improves agent-level behavior, persistent failures often indicate
structural deficiencies in the coordination topology. To address this,HERAperforms
topology mutation when trajectories consistently fail (e.g., F1 = 0). Upon identification by
the orchestrator,HERAexplores alternative structures by either replacing the failed agent
Niwith an alternative N′
ior augmenting the topology with additional agents, yielding
candidate topologies Γ′. These candidates are then incorporated into the same gradient-free
GRPO optimization loop. This process enables the co-evolution of coordination structure
and agent behaviors.
4 Experiments
We aim to answer the following questions:RQ1 (Effectiveness): How doesHERA’s
overall performance compare to state-of-the-art (SOTA) baselines? (§5.1)RQ2 (Component
Contribution): What is the individual contribution of the key components? (§5.2)RQ3
(Efficiency): Does improved performance, if any, come at the cost of increased token
consumption?(§5.3)RQ4 (Topology Evolution): How do agent coordination topologies
evolve as the system accumulates experience and updates agent prompts? (§5.4)
Tasks and Datasets.We evaluateHERAon multi-hop QA benchmarks 2WikiMultiHopQA
(2WikiQA) (Ho et al., 2020), HotpotQA (Yang et al., 2018), MusiQue (Trivedi et al., 2022) and
AmbigQA (Min et al., 2020) for ambiguous QA. For out-of-distribution (OOD) evaluation,
we use Bamboogle (Press et al., 2023) and HoVer (Jiang et al., 2020) for multi-hop QA and
fact verification respectively. Evaluation metrics involve Exact Match (EM), accuracy, and
F1 score for QA tasks, and accuracy for fact verification.
Baselines.We compareHERAagainst: (1)Direct inference and chain-of-thought (CoT)
without RAGacross different LLM families and sizes; (2)Single-turn RAGwith different
LLMs as answer generators; (3)Iterative RAG: IterDRAG (Yue et al., 2024), Plan-RAG
(Verma et al., 2024) Search-o1 (Li et al., 2025a), Search-R1 (Jin et al., 2025), IRCoT (Trivedi
et al., 2023), SELF-RAG (Asai et al., 2023), CORAG (Wang et al., 2025); and (4)Agentic RAG,
including InstructRAG, R1-Searcher (Song et al., 2025a), DeepResearcher (Zheng et al., 2025),
MMOA-RAG (Chen et al., 2025c), AceSearcher (Xu et al., 2025a) , and MAO-ARAG (Chen
et al., 2025c), ExSearch (Shi et al., 2025). Model and implementation details are provided in
Appendix E.
Implementation Details.We consider core agents commonly employed in RAG: Query
Decomposer, Retriever, Answer Generator, Query Rewriter, Evidence Selector, Context
Validator, Reflect Agent, and Conclude Agent. We use Wikipedia as the corpus, returning
top-5passages per step for multi-step retrieval and top-10for single-turn methods via the
BGE retriever (Xiao et al., 2024). We use Llama-3.1-8 (Grattafiori et al., 2024) and Qwen-
3-14B (Yang et al., 2025a) as the backbone LLMs for the orchestrator and GPT-4o-mini for
agents. All backbones remain frozen during learning and inference. To highlight data
efficiency, we first use GPT-4o to annotate queries from 2WikiQA, HotpotQA, AmbigQA,
and MusiQue alongreasoning type(bridge, intersection, comparison, temporal multi-hop,
ambiguous) andcomplexity(easy, medium, hard). Based on these annotations, we perform
stratified, difficulty-aware sampling to construct a curated training set that ensures balanced
coverage across reasoning categories while preserving distributional diversity (statistics in
Appendix D). Generative sampling uses temperature 0.9 during group rollouts, and 0.0 /0.3
for in-distribution and OOD evaluation, respectively. (Pseudocode forHERAis provided in
Appendix G).
5

Preprint. Under review.
5 Results and Analysis
5.1 Overall Performance
Table 1: Comparison ofHERAwith baselines. Best results arebold, second-best are
underlined . “-” indicates methods not designed for or evaluated on the task.
Dataset HotpotQA 2WikiQA MusiQue AmbigQA Bamboogle HoVer
Metrics Acc EM F1 Acc EM F1 Acc EM F1 Acc EM F1 Acc EM F1 Acc
Single-turn Retrieval
Qwen2.5-14B-Instruct 43.53 35.72 46.14 26.64 26.25 31.85 9.23 7.00 13.75 39.78 36.65 50.23 27.20 34.70 34.35 45.50
Qwen3-14B 44.47 36.25 47.56 31.57 31.10 37.79 13.16 9.99 15.72 38.47 36.10 50.91 31.54 29.60 38.31 49.05
Qwen3-8B 41.35 33.70 45.10 26.37 26.10 32.12 11.49 8.71 13.62 36.02 33.80 47.70 23.87 22.40 30.28 50.15
GPT-4o-mini 46.30 37.74 49.30 28.42 28.41 35.11 16.63 13.94 19.60 37.40 33.80 47.99 38.36 36.00 42.81 47.95
Llama3-3.1-8B-instruct 33.16 27.02 35.14 20.40 20.35 25.11 8.43 6.43 12.61 18.03 18.96 24.60 20.46 19.20 27.24 49.90
Advanced RAG
Search-r1-7B 51.11 41.94 54.18 48.89 44.69 49.20 20.45 18.58 25.73 47.32 45.88 65.54 35.79 32.64 43.93 47.76
Iter-RetGen (Qwen2.5-7B) 20.37 16.71 24.59 47.71 43.54 49.12 19.89 16.79 21.04 — — — 35.33 34.12 40.65 51.29
Plan-RAG 8b 29.95 25.96 37.51 30.53 27.38 31.68 13.08 9.90 13.34 — — — 30.01 23.57 32.12 50.33
SELF-RAG (llama2-7B) 23.85 13.64 25.07 40.82 37.30 42.02 8.47 7.16 14.16 28.51 24.03 39.62 25.80 22.66 30.05 40.52
R1-Searcher 7B 41.15 32.54 50.69 40.11 35.38 45.19 18.21 13.51 19.19 — — — 41.00 38.85 46.76 51.56
ExSearch (Qwen2.5-7B) 55.04 45.11 58.22 44.83 44.03 51.56 22.73 19.58 27.10 – — – 56.12 51.27 63.80 66.62
DeepResearcher-7B 40.65 35.23 58.86 49.40 54.35 64.05 24.53 18.56 25.57 49.74 42.23 51.50 37.42 34.20 44.40 58.19
IRCoT (GPT-4o-mini) 40.39 33.14 54.93 38.73 35.50 42.01 19.11 18.23 25.00 — — – 34.88 30.73 33.97 55.37
InstructRAG-8B 34.40 33.15 34.83 36.98 35.45 40.72 16.11 14.19 19.50 40.07 39.72 40.73 26.16 23.44 31.18 56.76
CORAG-8B (Greedy) 38.02 32.95 52.85 39.09 35.43 46.62 16.45 15.98 21.00 32.10 27.25 43.95 33.03 31.61 41.25 40.82
MMOA-RAG-8B 30.02 26.02 34.75 35.56 31.89 38.58 16.47 13.36 18.82 38.85 34.75 48.59 34.47 31.76 42.01 58.19
AceSearcher-8B 42.91 37.1968.07 50.83 45.59 52.75 25.24 19.10 26.62 – – – 47.21 45.12 51.91 59.37
Ours
HERA (Qwen-3-14B) 55.38 52.50 63.03 60.02 59.50 64.77 27.19 26.57 35.82 54.43 48.74 67.81 49.01 46.46 60.53 67.35
HERA (Llama-3.1-8) 53.28 48.01 57.91 52.62 46.70 55.42 25.04 24.46 31.07 48.53 43.46 60.72 46.60 45.22 56.31 60.31
Table 1 comparesHERAwith a range of baselines (Direct inference and CoT results are
reported in Appendix F.2). key observations are:(i) Consistent gains over SOTA iterative
and agentic RAG baselines.HERAachieves an average of 38.69% gains on all datasets
compared against recent SOTA, demonstrating the effectiveness of experience-guided or-
chestration for interleaved reasoning and retrieval. Best AmbigQA performance highlights
HERA’s robustness to queries with multiple plausible interpretations.(ii) Robust OOD
performance.HERAranks second on Bamboogle (F1 −5.4% vs. ExSearch) and first on
HoVer ( +64.95% vs. CORAG), reflecting the transferability of the accumulated experience
library and evolving agent behaviors. In contrast, most prior approaches relying on heuris-
tic planning or fixed iterative retrieval generalize poorly beyond in-distribution data.(iii)
Efficiency as gradient-free optimization.HERAsurpasses RL-fine-tuned multi-agent RAG
methods, such as MMAO-RAG, showing that semantic group-wise advantages enable stable,
consistent improvements across tasks and distribution shifts without costly model updates.
(iv)HERA-Qwen consistently outperformsHERA-Llama across all datasets, particularly
on cross-document reasoning and complex disambiguation tasks. WhileHERA-Llama
remains competitive in-domain, its performance drops on OOD tasks, indicating limited
reasoning and generalization capacity for ambiguous or sparse queries.
5.2 Ablation Study
Figure 2: Ablation Studies ofHERAwith Qwen-3-14B as the backbone.
Figure 2 presents the ablation results ofHERAon the test set using Qwen-3-14B as the
backbone (Llama-3.1-8 results are provided in Appendix F.2). Removing the experience
library yields consistent but moderate performance declines across datasets (typically ↓6%
6

Preprint. Under review.
to 15% relative accuracy), highlighting its role in improving knowledge guiding and cross-
instance generalization. In contrast, omitting prompt evolution leads to substantially larger
and more variable declines on multi-hop QA (up to ↓30% relative on 2WikiQA for both
backbones), where precise decomposition and trajectory control are essential. On ambiguity-
intensive datasets AmbigQA, prompt evolution introduces a recall-oriented bias: improving
F1 while occasionally reducing EM, whereas the experience library promotes precision
through grounded outputs, revealing a systematic EM–F1 trade-off. These patterns are
consistent across backbones, indicating that the functional contributions of each component
are largely model-agnostic. Importantly, the fullHERAframework consistently outperforms
both ablations by a non-additive margin, indicating a positive interaction effect: improved
reasoning behaviors from prompt evolution enhance the quality of stored experiences,
which in turn reinforce subsequent reasoning through stronger grounding. Overall, the
results demonstrate that prompt evolution and the experience library target complementary
failure modes: procedural correctness versus epistemic reliability, and that their synergy is
critical for robust multi-hop reasoning.
5.3 Performance-Efficiency Trade-off
We examine the potential trade-off between performance and data efficiency by analyzing
token usage patterns throughout learning. Our analysis emphasizes two key dimensions: (i)
the dynamics of token consumption as learning progresses, and (ii) the interplay between
token efficiency and task performance..
5.3.1 Dynamics of Token Consumption
Figure 3: Token usageWe visualize the token consumption throughout the learning
process by Locally Weighted Scatterplot Smoothing (LOWESS)
(Cleveland, 1981; Dang et al., 2025). Figure 3 illustrates token
usage trends on HotpotQA and 2WikiQA as representative
tasks.(i) Exploration–Exploitation Transition. On HotpotQA,
token consumption follows a non-monotonic trajectory. Early
spikes reflect an exploratory phase in which the orchestra-
tor activates multiple agents, constructs extended reasoning
traces, and performs redundant steps to optimize performance.
As experience accumulates, token usage declines monoton-
ically, marking a transition to exploitation: low-efficiency
agents are pruned, and reasoning chains are shortened. A
subsequent plateau indicates convergence toward an efficient token budget. In contrast,
2WikiQA exhibits no initial exploratory peak, suggesting that transferable priors from
previously accumulated reasoning experience support efficient agent orchestration with
minimal trial-and-error.(ii) Functional Contributions.Token usage dynamics are tightly
coupled withHERA’s experience library and prompt evolution. The library accumulates
high-utility and generalizable experience, while evolving prompts enables agents to elevate
their performance. Together, they drive emergent computational frugality, enabling high
reasoning performance without explicit token-based penalties.
5.3.2 Performance vs.Token Usage
Figure 4: Comparison of Performance–Token Efficiency Trade-off with Selected Baselines.
Figure 4 compares multi-hop QA performance against token consumption. Key observa-
tions include:(i) Superior Performance–Efficiency Trade-off. Across all datasets,HERA
consistently dominates the performance–efficiency frontier, achieving higher F1 scores with
controlled token budgets. This demonstrates that gains arise from efficient reasoning trajec-
tories instead of brute-force context scaling.(ii) Backbone-Specific Patterns.)HERAwith
7

Preprint. Under review.
Qwen-3-14B achieves the highest absolute performance, whereasHERAwith Llama-3.1-8
consistently minimizes token consumption ( ∼4.7k–6.6k across most datasets) while preserv-
ing competitive F1 scores, indicating favorable computational efficiency.(iii) Consistent
performance and efficiency gains over baselines. CORAG uses the largest number of
tokens across most datasets (up to 20 k+) without converting these costs into sustainable
reasoning improvements or effective knowledge reuse, resulting in suboptimal F1 (e.g.,
20.30 on MusiQue and 40.82 on HoVer). AceSearcher occupies an intermediate position,
balancing moderate token usage and performance.
5.4 Topology Evolution
To elucidate the emergence of coordinated multi-agent behavior, we systematically analyze
the evolution of interaction topologies over the course of learning. For this purpose, we
introduce topology-based metrics to quantify structural changes in agent collaboration.
5.4.1 Transition Entropy
Figure 5: Transition entropyTransition Entropy quantifies the uncertainty in agent-to-
agent transitions, capturing the policy-level dynamics. Let
Prob .(Nj|Ni)denote the empirical transition probability
from agentN itoN j, the Transition Entropy is defined as:
Htrans=−∑i,jProb.(N i→ N j)logProb.(N i→ N j)
We compute Htrans using a sliding-window over learning.
Beyond the exploration–exploitation dynamics, Figure 5 shows that after an initial rise,
entropy stabilizes at an intermediate level, reflecting the emergence of structured yet flexible
agent interactions. This plateau indicates thatHERAconsolidates high-value transition
sequences while maintaining controlled exploration, enabling the system to integrate novel
reasoning strategies without disrupting established coordination. The pattern highlights
the complementary roles of the experience library, which reinforces effective multi-agent
pathways, and prompt evolution, which adaptively refines agent behavior for functional
alignment with task objectives. Notably, the plateau at intermediate entropy levels indicates
controlled exploration even during exploitation, preventing premature convergence to
suboptimal strategies while allowing the integration of novel cooperation pathways.
5.4.2 Graph-based Metrics
Figure 6: Graph metricsTo systematically characterize the topology evolution of
HERA, we model each trajectory as a graph Gτ= (V ,E)
where nodes represent agents and edges denote interactions.
We quantify structural and functional properties via defin-
ing graph-theoretic metrics:(a) Number of agents |V|: the
total distinct agents involved in a trajectory, reflecting the
breadth of collaboration.(b) Node efficiency: per-agent
contribution, formally byF1(τ)
|V|, capturing per-agent utility.(c) Self-Loops: the frequency
of self-transitions, reflecting potential redundancy or degenerate reasoning.(d) Number
of cycles: count of closed-loop paths capturing iterative reasoning patterns.(e) Diameter:
longest shortest-path between any two nodes, representing maximum reasoning depth.
We segment learning into four phases - initial, exploration, refinement, and optimization,
and analyze normalized metrics per question type. As shown in Fig. 6, Early trajectories are
narrow and shallow, with low agent counts, minimal cycles, and limited diameter, reflecting
sparse coordination. During exploration, the system dynamically recruits more agents,
increasing diameter and cycle formation, capturing iterative reasoning and intermediate
verification. By the refinement phase, trajectories contract: redundant agents and self-
loops are pruned, diameter decreases, and node efficiency rises. This selective retention is
guided by the experience library, which reinforces high-value interaction topologies, and by
prompt evolution, which refines agent behaviors and intermediate outputs. The optimized
phase exhibits compact chains with minimal nodes, high per-agent utility, and strategically
preserved cycles, representing a balance of efficiency and robustness.
8

Preprint. Under review.
Overall, the evolving topologies reveal thatHERA’s dual mechanisms - structured experi-
ence retrieval and adaptive prompt evolution enable a principled transition from exploratory,
diffuse agent interactions to compact, high-utility, and resilient multi-agent system.
6 Related Work
Retrieval-Augmented Generation (RAG)boosts LLM performance by leveraging external
evidence (Izacard et al., 2023; Lewis et al., 2020; Gao et al., 2023b; Tang & Yang, 2024; Xiong
et al., 2024; Fan et al., 2024). Early methods embed retrieval within model architectures,
permitting direct conditioning on retrieved context (Khandelwal et al., 2020; Liu et al., 2024a;
Izacard & Grave, 2020; Wang et al., 2023; Borgeaud et al., 2022; Izacard et al., 2023; Guu
et al., 2020). Subsequent methods modularize RAG into pre- (Chan et al., 2024; Gao et al.,
2023a; Ma et al., 2023) and post-retrieval components (Abdallah et al., 2025; Xu et al., 2023;
Li & Ramakrishnan, 2025) to enhance retrieval and reasoning, and synergize retrieval with
structured reasoning for multi-step tasks (Jeong et al., 2024a; Lee et al., 2024; Asai et al.,
2024; Shao et al., 2023). Emerging paradigms such as DeepResearcher Zheng et al. (2025); Li
et al. (2025b); Huang et al. (2025) extend RAG into agentic (Xie et al., 2025; Fang et al., 2025;
Chen et al., 2025c;b) workflows, embedding retrieval within iterative reasoning (Chen et al.,
2025a; Shi et al., 2025; Li et al., 2025a; Jin et al., 2025; Song et al., 2025b) and tool use (Song
et al., 2025b; Sun et al., 2025), to tackle complex problems.
Multi-agent OrchestrationPrior works (Chen et al., 2023; Liu et al., 2024b; Qian et al.,
2025; Khan et al., 2024; Liang et al., 2024; Wang et al., 2024; Iqbal et al., 2022) often frame
orchestration as a sequential pipeline optimized via reinforcement learning (Nie et al., 2025;
Wang et al., 2026). Dynamic methods adapt agent interactions at inference based on task
features or intermediate outcomes (Yu, 2026; Yuan et al., 2025; Zhang et al., 2026; Chang
& Geng, 2025), but incur high coordination overhead and rely on accurate intermediate
evaluations. Alternatively, some approaches treat orchestration as a global decision problem,
learning a unified policy over the full interaction structure (Ke et al., 2026; Zhang et al., 2025c).
Prompting strategies and tool management improve agent coordination (Lin et al., 2025; Su
et al., 2025; Zhang et al., 2025b; Dang et al., 2025; Du et al., 2024) but current approaches
are often static (Zhou & Chan, 2026), weakly adaptive, or limited by computational and
scalability constraints (Choi et al., 2024).
Prompt Optimization and EvolutionAutomatic Prompt Optimization treats prompts as a
black-box search over seeds, feedback, candidate generation and selection (Ramnath et al.,
2025). Programmatic frameworks (Khattab et al., 2024) compile LM pipelines and optimize
prompts offline using user metrics, while textual backpropagation (Yuksekgonul et al., 2025;
Hu et al., 2024; Ma et al., 2025) utilizes natural-language feedback to refine collaboration
protocols and behavioral instructions. Recent work jointly optimizes agentic structures
and prompts (Tian et al., 2025; Murthy et al., 2025; Spiess et al., 2025), using search-based
strategies (Zhang et al., 2024), genetic algorithms (Agrawal et al., 2026) and treats prompt
optimization as experience-driven adaptation (Yao et al., 2022; Shinn et al., 2023; Zhao et al.,
2024; Pei et al., 2025; Xie et al., 2026).
7 Conclusion
We presentHERA, a hierarchical framework that jointly evolves multi-agent orchestration
through an experience library and role-aware agent prompts. The experience library consol-
idates transferable orchestration strategies, while prompt evolution enables role-specific
refinement of agent behaviors along operational and behavioral principles.HERAachieves
superior performance on multi-hop and ambiguity-intensive benchmarks without relying
on gradient-based training. Topological analyses reveal emergent self-organization, where
sparse exploration gives rise to compact, high-utility multi-agent networks, highlighting
both efficient coordination and robustness. Future work may investigate meta-optimization
of agent roles, adaptive multi-turn orchestration, and resource-aware autonomous evolution,
paving the way for hierarchical role structures, conditional agent sequencing, and efficient
reasoning under token or latency constraints.
9

Preprint. Under review.
Acknowledgments
Use unnumbered first level headings for the acknowledgments. All acknowledgments,
including those to funding agencies, go at the end of the paper.
Ethics Statement
Authors can add an optional ethics statement to the paper. For papers that touch on ethical
issues, this section will be evaluated as part of the review process. The ethics statement
should come at the end of the paper. It does not count toward the page limit, but should not
be more than 1 page.
References
Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, and Adam
Jatowt. Rankify: A comprehensive python toolkit for retrieval, re-ranking, and retrieval-
augmented generation.arXiv preprint arXiv:2502.02464, 2025.
Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-
Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa:
Reflective prompt evolution can outperform reinforcement learning.ICLR, 2026.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learn-
ing to retrieve, generate, and critique through self-reflection. InThe Twelfth International
Conference on Learning Representations, 2023.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag:
Learning to retrieve, generate, and critique through self-reflection. 2024.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan
Clark, et al. Improving language models by retrieving from trillions of tokens. In
International conference on machine learning, pp. 2206–2240. PMLR, 2022.
Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang
Li, Zongyi Li, Haojia Lin, et al. Training-free group relative policy optimization.arXiv
preprint arXiv:2510.08191, 2025.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo, and Jie Fu.
Rq-rag: Learning to refine queries for retrieval augmented generation.arXiv preprint
arXiv:2404.00610, 2024.
Edward Y Chang and Longling Geng. Sagallm: context management, validation, and
transaction guarantees for multi-agent llm planning.arXiv preprint arXiv:2503.11951, 2025.
Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu,
Haofen Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al. Learning to reason with search
for llms via reinforcement learning.arXiv preprint arXiv:2503.19470, 2025a.
Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang
Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, et al. Agentverse: Facilitating multi-agent
collaboration and exploring emergent behaviors. InThe Twelfth International Conference on
Learning Representations, 2023.
Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei
Yin, Yiming Yang, and Jiaxin Mao. Improving retrieval-augmented generation through
multi-agent reinforcement learning.arXiv preprint arXiv:2501.15228, 2025b.
Yiqun Chen, Erhan Zhang, Lingyong Yan, Shuaiqiang Wang, Jizhou Huang, Dawei Yin,
and Jiaxin Mao. Mao-arag: Multi-agent orchestration for adaptive retrieval-augmented
generation.arXiv preprint arXiv:2508.01005, 2025c.
10

Preprint. Under review.
Jihye Choi, Nils Palumbo, Prasad Chalasani, Matthew M Engelhard, Somesh Jha, Anivarya
Kumar, and David Page. Malade: Orchestration of llm-powered agents with retrieval
augmented generation for pharmacovigilance.arXiv preprint arXiv:2408.01869, 2024.
William S Cleveland. Lowess: A program for smoothing scatterplots by robust locally
weighted regression.The American Statistician, 35(1):54, 1981.
Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng
Yang, Xiaoyin Che, Ye Tian, et al. Multi-agent collaboration via evolving orchestration.
Advances in neural information processing systems, 2025.
Zhuoyun Du, Cheng Qian, Wei Liu, Zihao Xie, Yifei Wang, Yufan Dang, Weize Chen, and
Cheng Yang. Multi-agent collaboration via cross-team orchestration. InAnnual Meeting
of the Association for Computational Linguistics, 2024. URL https://api.semanticscholar.
org/CorpusID:270440727.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng
Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large
language models. InProceedings of the 30th ACM SIGKDD conference on knowledge discovery
and data mining, pp. 6491–6501, 2024.
Xianxing Fang, Liangru Xie, Weibin Yang, Tianyi Zhang, Ruitao Zhang, Hao Wang, Di Wu,
and Yushan Pan. Orion: A multi-agent framework for optimizing rag systems through
specialized agent collaboration. InProceedings of the 16th International Conference on
Internetware, pp. 332–343, 2025.
Gihan Gamage, Nishan Mills, Daswin De Silva, Milos Manic, Harsha Moraliyage, Andrew
Jennings, and Damminda Alahakoon. Multi-agent rag chatbot architecture for decision
support in net-zero emission energy systems. In2024 IEEE International Conference on
Industrial Technology (ICIT), pp. 1–6. IEEE, 2024.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense re-
trieval without relevance labels. In Anna Rogers, Jordan Boyd-Graber, and Naoaki
Okazaki (eds.),Proceedings of the 61st Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pp. 1762–1777, Toronto, Canada, July 2023a.
Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.99. URL
https://aclanthology.org/2023.acl-long.99/.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, Haofen Wang, et al. Retrieval-augmented generation for large language
models: A survey.arXiv preprint arXiv:2312.10997, 2(1):32, 2023b.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The
llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Bernal Jim ´enez Guti ´errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to
memory: Non-parametric continual learning for large language models.arXiv preprint
arXiv:2502.14802, 2025.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval
augmented language model pre-training. InInternational conference on machine learning,
pp. 3929–3938. PMLR, 2020.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a
multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott,
Nuria Bel, and Chengqing Zong (eds.),Proceedings of the 28th International Conference on
Computational Linguistics, pp. 6609–6625, Barcelona, Spain (Online), December 2020. Inter-
national Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.
580. URLhttps://aclanthology.org/2020.coling-main.580/.
Yue Hu, Yuzhu Cai, Yaxin Du, Xinyu Zhu, Xiangrui Liu, Zijie Yu, Yuchen Hou, Shuo
Tang, and Siheng Chen. Self-evolving multi-agent collaboration networks for software
development.arXiv preprint arXiv:2410.16946, 2024.
11

Preprint. Under review.
Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang
Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, et al. Memory in the age of ai agents.arXiv
preprint arXiv:2512.13564, 2025.
Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi
Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, et al. Deep research agents: A systematic
examination and roadmap.arXiv preprint arXiv:2506.18096, 2025.
Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun
Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report.arXiv preprint
arXiv:2409.12186, 2024.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv
preprint arXiv:2410.21276, 2024.
Shariq Iqbal, Robby Costales, and Fei Sha. Alma: Hierarchical learning for composite
multi-agent tasks.Advances in neural information processing systems, 35:7155–7166, 2022.
Gautier Izacard and Edouard Grave. Distilling knowledge from reader to retriever for
question answering.arXiv preprint arXiv:2012.04584, 2020.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick,
Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot
learning with retrieval augmented language models.Journal of Machine Learning Research,
24(251):1–43, 2023.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. Adaptive-
RAG: Learning to adapt retrieval-augmented large language models through question
complexity. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.),Proceedings of the
2024 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers), pp. 7036–7050, Mexico City, Mexico,
June 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.
389. URLhttps://aclanthology.org/2024.naacl-long.389/.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-
rag: Learning to adapt retrieval-augmented large language models through question
complexity. InProceedings of the 2024 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers), pp. 7036–7050, 2024b.
Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit
Bansal. Hover: A dataset for many-hop fact extraction and claim verification. InFindings
of the Association for Computational Linguistics: EMNLP 2020, pp. 3441–3460, 2020.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In
Proceedings of the 2023 conference on empirical methods in natural language processing, pp.
7969–7992, 2023.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed
Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines
with reinforcement learning.arXiv preprint arXiv:2503.09516, 2025.
Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram,
Semih Yavuz, Caiming Xiong, and Shafiq Joty. Mas-orchestra: Understanding and im-
proving multi-agent reasoning through holistic orchestration and controlled benchmarks.
arXiv preprint arXiv:2601.14652, 2026.
Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan,
Edward Grefenstette, Samuel R Bowman, Tim Rockt ¨aschel, and Ethan Perez. Debating
with more persuasive llms leads to more truthful answers.arXiv preprint arXiv:2402.06782,
2024.
12

Preprint. Under review.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Gen-
eralization through memorization: Nearest neighbor language models.International
Conference on Learning Representations, 2020.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri
Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather
Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language
model calls into self-improving pipelines. 2024.
Myeonghwa Lee, Seonho An, and Min-Soo Kim. PlanRAG: A plan-then-retrieval aug-
mented generation for generative large language models as decision makers. In Kevin
Duh, Helena Gomez, and Steven Bethard (eds.),Proceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pp. 6537–6555, Mexico City, Mexico, June 2024. As-
sociation for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.364. URL
https://aclanthology.org/2024.naacl-long.364/.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474, 2020.
Sha Li and Naren Ramakrishnan. Oreo: A plug-in context reconstructor to enhance retrieval-
augmented generation. InProceedings of the 2025 International ACM SIGIR Conference on
Innovative Concepts and Theories in Information Retrieval (ICTIR), pp. 238–253, 2025.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang,
and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. In
Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp.
5420–5438, 2025a.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu,
and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research
capability.arXiv preprint arXiv:2504.21776, 2025b.
Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on llm-based multi-agent
systems: workflow, infrastructure, and challenges.Vicinagearth, 1(1):9, 2024.
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang,
Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language
models through multi-agent debate. InProceedings of the 2024 conference on empirical
methods in natural language processing, pp. 17889–17904, 2024.
Leilei Lin, Yumeng Jin, Yingming Zhou, Wenlong Chen, and Chen Qian. Mao: A framework
for process model generation with multi-agent orchestration.IEEE Transactions on Services
Computing, 2025.
Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun Ma. Hm-rag:
Hierarchical multi-agent multimodal retrieval augmented generation. InProceedings of the
33rd ACM international conference on multimedia, pp. 2781–2790, 2025.
Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Chankyu Lee, Mohammad Shoeybi, and Bryan
Catanzaro. Chatqa: Surpassing gpt-4 on conversational qa and rag.Advances in Neural
Information Processing Systems, 37:15416–15459, 2024a.
Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent
network for task-oriented agent collaboration. InFirst Conference on Language Modeling,
2024b.
Xiaowen Ma, Chenyang Lin, Yao Zhang, Volker Tresp, and Yunpu Ma. Agentic neural
networks: Self-evolving multi-agent systems via textual backpropagation.arXiv preprint
arXiv:2506.09046, 2025.
13

Preprint. Under review.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in
retrieval-augmented large language models. In Houda Bouamor, Juan Pino, and Kalika
Bali (eds.),Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, pp. 5303–5315, Singapore, December 2023. Association for Computational
Linguistics. doi: 10.18653/v1/2023.emnlp-main.322. URL https://aclanthology.org/
2023.emnlp-main.322/.
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. AmbigQA:
Answering ambiguous open-domain questions. In Bonnie Webber, Trevor Cohn, Yulan
He, and Yang Liu (eds.),Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 5783–5797, Online, November 2020. Association for
Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.466. URL https://
aclanthology.org/2020.emnlp-main.466/.
Rithesh Murthy, Ming Zhu, Liangwei Yang, Jielin Qiu, Juntao Tan, Shelby Heinecke, Caim-
ing Xiong, Silvio Savarese, and Huan Wang. Promptomatix: an automatic prompt
optimization framework for large language models.arXiv preprint arXiv:2507.14241, 2025.
Thang Nguyen, Peter Chin, and Yu-Wing Tai. Ma-rag: Multi-agent retrieval-augmented
generation via collaborative chain-of-thought reasoning.arXiv preprint arXiv:2505.20096,
2025.
Fan Nie, Lan Feng, Haotian Ye, Weixin Liang, Pan Lu, Huaxiu Yao, Alexandre Alahi, and
James Zou. Weak-for-strong: Training weak meta-agent to harness strong executors.arXiv
preprint arXiv:2504.04785, 2025.
Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le,
Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with
reasoning memory.arXiv preprint arXiv:2509.25140, 2025.
Zehua Pei, Hui-Ling Zhen, Shixiong Kai, Sinno Jialin Pan, Yunhe Wang, Mingxuan Yuan,
and Bei Yu. Scope: Prompt evolution for enhancing agent effectiveness.arXiv preprint
arXiv:2512.15374, 2025.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis.
Measuring and narrowing the compositionality gap in language models. In Houda
Bouamor, Juan Pino, and Kalika Bali (eds.),Findings of the Association for Computational
Linguistics: EMNLP 2023, pp. 5687–5711, Singapore, December 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.378. URL https:
//aclanthology.org/2023.findings-emnlp.378/.
Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang,
Zhuoyun Du, Weize Chen, Cheng Yang, et al. Scaling large language model-based
multi-agent collaboration. 2025.
Kiran Ramnath, Kang Zhou, Sheng Guan, Soumya Smruti Mishra, Xuan Qi, Zhengyuan
Shen, Shuai Wang, Sangmin Woo, Sullam Jeoung, Yawei Wang, et al. A systematic survey
of automatic prompt optimization techniques. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pp. 33066–33098, 2025.
Timo Schick, Jane Dwivedi-Yu, Roberto Dess `ı, Roberta Raileanu, Maria Lomeli, Eric Hambro,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models
can teach themselves to use tools.Advances in neural information processing systems, 36:
68539–68551, 2023.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen.
Enhancing retrieval-augmented large language models with iterative retrieval-generation
synergy. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),Findings of the Association
for Computational Linguistics: EMNLP 2023, pp. 9248–9274, Singapore, December 2023.
Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.620.
URLhttps://aclanthology.org/2023.findings-emnlp.620/.
14

Preprint. Under review.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathe-
matical reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.
Xu Shen, Yixin Liu, Yiwei Dai, Yili Wang, Rui Miao, Yue Tan, Shirui Pan, and Xin Wang.
Understanding the information propagation effects of communication topologies in llm-
based multi-agent systems. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pp. 12358–12372, 2025.
Zhengliang Shi, Lingyong Yan, Dawei Yin, Suzan Verberne, Maarten de Rijke, and Zhaochun
Ren. Iterative self-incentivization empowers large language models as agentic searchers.
arXiv preprint arXiv:2505.20128, 2025.
Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.
Reflexion: Language agents with verbal reinforcement learning.Advances in Neural
Information Processing Systems, 36:8634–8652, 2023.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao,
Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via
reinforcement learning.arXiv preprint arXiv:2503.05592, 2025a.
Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao,
Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher++: Incentivizing
the dynamic knowledge acquisition of llms via reinforcement learning.arXiv preprint
arXiv:2505.17005, 2025b.
Claudio Spiess, Mandana Vaziri, Louis Mandel, and Martin Hirzel. Autopdl: Automatic
prompt optimization for llm agents.arXiv preprint arXiv:2504.04365, 2025.
Hongjin Su, Shizhe Diao, Ximing Lu, Mingjie Liu, Jiacheng Xu, Xin Dong, Yonggan Fu, Peter
Belcak, Hanrong Ye, Hongxu Yin, et al. Toolorchestra: Elevating intelligence via efficient
model and tool orchestration.arXiv preprint arXiv:2511.21689, 2025.
Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei
Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, et al. Simpledeepsearcher: Deep information
seeking via web-powered reasoning trajectory synthesis.arXiv preprint arXiv:2505.16834,
2025.
Xiaqiang Tang, Jian Li, Nan Du, and Sihong Xie. Adapting to non-stationary environments:
Multi-armed bandit enhanced retrieval-augmented generation on knowledge graphs. In
Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 12658–12666,
2025.
Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for
multi-hop queries, 2024.
Chunhao Tian, Yutong Wang, Xuebo Liu, Zhexuan Wang, Liang Ding, Miao Zhang, and Min
Zhang. AgentInit: Initializing LLM-based multi-agent systems via diversity and expertise
orchestration for effective and efficient collaboration. In Christos Christodoulopoulos,
Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.),Findings of the Association for
Computational Linguistics: EMNLP 2025, pp. 11870–11902, Suzhou, China, November 2025.
Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/
2025.findings-emnlp.636. URLhttps://aclanthology.org/2025.findings-emnlp.636/.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue:
Multihop questions via single-hop question composition.Transactions of the Association for
Computational Linguistics, 2022.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.
In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.),Proceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
10014–10037, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.557. URLhttps://aclanthology.org/2023.acl-long.557/.
15

Preprint. Under review.
Prakhar Verma, Sukruta Prakash Midigeshi, Gaurav Sinha, Arno Solin, Nagarajan Natarajan,
and Amit Sharma. Plan-rag: Planning-guided retrieval augmented generation. 2024.
Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee, Zihan Liu, Mohammad Shoeybi,
Yi Dong, Oleksii Kuchaiev, Bo Li, Chaowei Xiao, et al. Shall we pretrain autoregressive
language models with retrieval? a comprehensive study. InProceedings of the 2023
conference on empirical methods in natural language processing, pp. 7763–7786, 2023.
L Wang, H Chen, N Yang, X Huang, Z Dou, and F Wei. Chain-of-retrieval augmented
generation. arxiv.arXiv preprint arXiv:2501.14342, 2025.
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. Rethinking the
bounds of llm reasoning: Are multi-agent discussions the key? InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
6106–6131, 2024.
Siyu Wang, Ruotian Lu, Zhihao Yang, Yuchao Wang, Yanzhou Zhang, Lei Xu, Qimin Xu,
Guojun Yin, Cailian Chen, and Xinping Guan. Agentconductor: Topology evolution for
multi-agent competition-level code generation.arXiv preprint arXiv:2602.17100, 2026.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie.
C-pack: Packed resources for general chinese embeddings. InProceedings of the 47th
international ACM SIGIR conference on research and development in information retrieval, pp.
641–649, 2024.
Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua Bergerson, John K Hutchison, Duane R
Verner, Jordan Branham, M Ross Alexander, Robert B Ross, Yan Feng, et al. Marsha:
multi-agent rag system for hazard adaptation.npj Climate Action, 4(1):70, 2025.
Yunfei Xie, Kevin Wang, Bobby Cheng, Jianzhu Yao, Zhizhou Sha, Alexander Duffy, Yihan Xi,
Hongyuan Mei, Cheston Tan, Chen Wei, et al. Memo: Memory-augmented model context
optimization for robust multi-turn multi-agent llm games.arXiv preprint arXiv:2603.09022,
2026.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-
augmented generation for medicine. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar
(eds.),Findings of the Association for Computational Linguistics: ACL 2024, pp. 6233–6251,
Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.
18653/v1/2024.findings-acl.372. URL https://aclanthology.org/2024.findings-acl.
372/.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms
with compression and selective augmentation.arXiv preprint arXiv:2310.04408, 2023.
Ran Xu, Yuchen Zhuang, Zihan Dong, Jonathan Wang, Yue Yu, Joyce C Ho, Linjun Zhang,
Haoyu Wang, Wenqi Shi, and Carl Yang. Acesearcher: Bootstrapping reasoning and
search for llms via reinforced self-play.arXiv preprint arXiv:2509.24193, 2025a.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem:
Agentic memory for llm agents.arXiv preprint arXiv:2502.12110, 2025b.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025a.
Yahe Yang, Chao Xu, Jing Guo, Tianbao Feng, and Cailian Ruan. Improving the rag-based
personalized discharge care system by introducing the memory mechanism. In2025 IEEE
17th International Conference on Computer Research and Development (ICCRD), pp. 316–322.
IEEE, 2025b.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhut-
dinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable
multi-hop question answering. InConference on Empirical Methods in Natural Language
Processing (EMNLP), 2018.
16

Preprint. Under review.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and
Yuan Cao. React: Synergizing reasoning and acting in language models. InThe eleventh
international conference on learning representations, 2022.
Geunbin Yu. Adaptorch: Task-adaptive multi-agent orchestration in the era of llm perfor-
mance convergence.arXiv preprint arXiv:2602.16873, 2026.
Michelle Yuan, Khushbu Pahwa, Shuaichen Chang, Mustafa Kaba, Jiarong Jiang, Xiaofei Ma,
Yi Zhang, and Monica Sunkara. Automated composition of agents: A knapsack approach
for agentic component selection.arXiv preprint arXiv:2510.16499, 2025.
Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin,
Dong Wang, Xuanhui Wang, and Michael Bendersky. Inference scaling for long-context
retrieval augmented generation.arXiv preprint arXiv:2410.04343, 2024.
Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos
Guestrin, and James Zou. Optimizing generative ai by backpropagating language model
feedback.Nature, 639:609–616, 2025.
Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen,
Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, et al. Aflow: Automating agentic
workflow generation.arXiv preprint arXiv:2410.10762, 2024.
Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin
Li, Chi Wang, Huazheng Wang, Yiran Chen, et al. Which agent causes task failures
and when? on automated failure attribution of llm multi-agent systems.arXiv preprint
arXiv:2505.00212, 2025a.
Wentao Zhang, Liang Zeng, Yuzhen Xiao, Yongcong Li, Ce Cui, Yilei Zhao, Rui Hu, Yang
Liu, Yahui Zhou, and Bo An. Agentorchestra: Orchestrating hierarchical multi-agent
intelligence with the tool-environment-agent(tea) protocol. 2025b. URL https://api.
semanticscholar.org/CorpusID:281658740.
Wentao Zhang, Liang Zeng, Yuzhen Xiao, Yongcong Li, Ce Cui, Yilei Zhao, Rui Hu, Yang
Liu, Yahui Zhou, and Bo An. Agentorchestra: Orchestrating multi-agent intelligence with
the tool-environment-agent(tea) protocol. 2025c. URL https://api.semanticscholar.
org/CorpusID:281658740.
Xing Zhang, Yanwei Cui, Guanghui Wang, Qucy Wei Qiu, Ziyuan Li, Fangwei Han, Yajing
Huang, Hengzhi Qiu, Bin Zhu, and Peiyang He. Verified multi-agent orchestration:
A plan-execute-verify-replan framework for complex query resolution.arXiv preprint
arXiv:2603.11445, 2026.
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang.
Expel: Llm agents are experiential learners. InProceedings of the AAAI Conference on
Artificial Intelligence, volume 38, pp. 19632–19642, 2024.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and
Pengfei Liu. DeepResearcher: Scaling deep research via reinforcement learning in real-
world environments. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn
Rose, and Violet Peng (eds.),Proceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pp. 414–431, Suzhou, China, November 2025. Association for
Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
22. URLhttps://aclanthology.org/2025.emnlp-main.22/.
Hanlin Zhou and Huah Yong Chan. Orch: many analyses, one merge-a deterministic
multi-agent orchestrator for discrete-choice reasoning with ema-guided routing.arXiv
preprint arXiv:2602.01797, 2026.
Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, and Lei Sha. Atm: Adversarial tuning
multi-agent system makes a robust retrieval-augmented generator. InProceedings of the
2024 Conference on Empirical Methods in Natural Language Processing, pp. 10902–10919,
2024a.
17

Preprint. Under review.
Kun Zhu, Xiaocheng Feng, Xiyuan Du, Yuxuan Gu, Weijiang Yu, Haotian Wang, Qianglong
Chen, Zheng Chu, Jingchang Chen, and Bing Qin. An information bottleneck perspective
for effective noise filtering on retrieval-augmented generation. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
1044–1069, 2024b.
Yuchen Zhuang, Xiang Chen, Tong Yu, Saayan Mitra, Victor Bursztyn, Ryan A Rossi, Somdeb
Sarkhel, and Chao Zhang. Toolchain*: Efficient action space navigation in large language
models with a* search. 2024.
18

Preprint. Under review.
A Theoretical Analysis
A.1 Formal Setup
Letq∈ Q denote an input query, Ethe experience library, Nthe agent pool, Γan agent
topology specifying the agent sequence, orders, and dependencies, and τan execution
trajectory. The orchestrator induces a policy to obtain an interaction topology:
Γ∼π θ(· |q,E,N).
A.2 Policy Iteration
HERAadmits a rigorous interpretation as gradient-free policy iteration over the structured
spaceΓ. Each iterationtcomprises two steps:
Policy Evaluation.A group of rollouts/topologies is sampled from the current policy πO
and executed to obtain trajectories and rewards. This constitutes an empirical Monte Carlo
estimate of the value function under the current policy.
Policy Improvement.Group-relative semantic advantages are obtained by reflecting on
and comparing high-reward trajectories with failed trajectories. These insights are distilled
and incorporated into the experience libraryE:
E(t+1)←Distill ({(Γ i,τi):R i>¯R})
where ¯Ris the semantic baseline.
This updated experience library modifies the conditioning context, which in turn reshapes
the induced distribution:
ππO 
Γ|q,E(t+1),N
without updatingπ O.
A.3 EM Interpretation
HERAexhibits an iterative structure analogous to Expectation-Maximization.
The agent topology Γas a latent variable, the query qas observed input, and the reward
R(τ)as a proxy for an implicit likelihood. The iterative procedure then resembles:
1. E-step (Implicit): Sampling
Γi∼πO(· |q,E(t),N)
and selecting high-reward topologies induces an approximate posterior over effective
reasoning programs:
P(t)(Γ)≈πO(· |q,E(t),N)·[R(τ i)> ¯R]
∑Γ′πθo(Γ′|q,E(t),N))·1[R(τ Γ′)> ¯R]
This represents a reward-filtered empirical distribution instead of a properly normalized
probabilistic posterior.
M-step(Implicit):instead of maximizing
EP(t)(S)h
logp 
y|q,Γ;π ≀i
over parametersπ O,HERAperforms a non-parametric update:
E(t+1)←arg max
EEP(t)(Γ)
logπ θO(Γ|q,E,N)
which implicitly increases the probability mass on high-performing structures by enriching
the conditioning context (experience library). R(τ) here serves as a heuristic surrogate
rather than a proper probabilistic objective.
19

Preprint. Under review.
A.4 Energy-Based Reweighting and Implicit KL Regularisation
The frozen base model πOprovides implicit regularisation. As policy updates are mediated
through the contextual augmentation Einstead of modifying parameters, the resulting
policy can be viewed as an energy-based reweighting of the base distribution:
πnew(Γ|q)∝π base(Γ|q)·exp 
fE(Γ,q)
,
where fE(Γ,q)denotes the log-ratio bias introduced by the experience library. This for-
mulation is structurally equivalent to the optimal solution of the KL-constrained policy
optimisation problem:
max
πEπ[fE(Γ,q)]−β·D KL 
π∥π base
,
whose closed-form solution is precisely
π∗(Γ|q)∝π base(S|q)·expfE(Γ,q)
β
.
The frozen base model therefore serves as an implicit KL anchor, preventing policy collapse
and ensuring that the updated distribution stays within the support of the original model.
This stability property is absent in unconstrained policy gradient methods, where aggressive
updates can destabilise the base distribution.
20

Preprint. Under review.
B Prompts used forHERA
Orchestrator: topology sampling
You are an orchestrator managing a team of specialized agents for multi-hop question
answering. Your goal is to design an effective multi-agent execution topology to
answer the given query correctly.
You have access to the following agents:
{agent descriptions}
You have retrieved the following relevant experiences from past executions:
{retrieved experiences}
Each experience entry has the format:
• Query Type:{query type}
• Insight:{insight}
• Utility score:{utility}
Given the query below, design a coordination topology by:
1. Selecting the subset of agents best suited to this query type.
2. Specifying their execution order (sequential or parallel where appropriate).
3.Defining dependency relationships (which agent’s output feeds into which).
Query:{query}
Respond in the following JSON format:
{
"query_profile": "<one-sentence characterization of query type>",
"selected_agents": ["<agent_name>", ...],
"execution_order": [
{"step": 1, "agent": "<agent_name>", "depends_on": [],
"mode": "sequential|parallel"},
...
]
}
21

Preprint. Under review.
Orchestrator: Semantic Advantage Extraction
You are an orchestrator evaluating a group of multi-agent execution trajectories for the same
query. Your goal is to identify why some trajectories succeeded and others failed, and extract
generalizable insights to guide future orchestration.
Query:{query}
Query type:{query type}
The following {G}trajectories were executed, ranked by task performance (F1) and then by
efficiency (total tokens consumed):
{trajectory group}
Each trajectory entry includes:
• Topology used (agents selected, execution order)
• Per-agent actions, inputs and outputs
• Final answer produced
• F1 score:{f1}
• Total tokens consumed:{tokens}
Perform a comparative analysis across the group:
1. Identify what the successful trajectories did differently from the failed ones.
2. Identify which agents, orderings, or dependencies contributed to success or failure.
3.Identify recurring failure patterns (e.g., incorrect agent selection, redundant steps,
missing retrieval before reasoning).
4.Distill findings into concise, actionable insights applicable to future queries of the
same type.
Respond in the following JSON format:
{
"success_factors": ["<factor>", ...],
"failure_modes": ["<failure_mode>", ...],
"insights": [
{
"query_type": "<type this insight applies to>",
"insight": "<actionable natural language insight>"
},
...
]
}
22

Preprint. Under review.
Experience Library Operations)
You are managing an experience library for a multi-agent question answering system.
The library stores insights as structured entries of the form:
(query profilec, insightz, utilityu)
where utilityureflects how often insightzhas led to successful agent orchestration.
Current experience library:
{current library}
New insights extracted from the latest execution group:
{new insights}
For each new insight, decide the appropriate consolidation operation:
•ADD: Insert a distinct insight not covered by existing entries.
•MERGE: Combine semantically similar or complementary insights into a
single, more complete entry.
•PRUNE: Remove a lower-utility entry that conflicts with a higher-utility one.
•KEEP: Retain the current library without modification.
Guidelines:
•PreferMERGEoverADDwhen insights share the same query type and
recommend compatible strategies.
•PreferPRUNEwhen entries provide contradictory guidance for the same
query type and differ substantially in utility.
• Avoid unbounded growth—consolidate aggressively to maintain generaliz-
ability.
Respond in the following JSON format:
{
"operations": [
{
"operation": "ADD|MERGE|PRUNE|KEEP",
"new_insight": "<text of new insight>",
"target_entry_ids": ["<id of existing entry if MERGE or PRUNE>"],
"merged_insight": "<combined insight text if MERGE, else null>",
"rationale": "<one sentence explaining the decision>"
},
...
]
}
23

Preprint. Under review.
Prompt for Agent Prompt Integration
System Prompt:
You are responsible for managing and evolving the prompt of a single agent
under strict length and structural constraints. Your task is to refine, optimize, and
consolidate the agent’s prompt to ensure clarity, consistency, and adherence to
operational and behavioral requirements.
User Prompt:
Current Agent Prompt:
{CURRENT_PROMPT}
Proposed Operational Rules:
{operational_rules}
Proposed Behavioral Principles:
{behavioral_principles}
Constraints:
1. Limit the number of tactical rules to a maximum ofK.
2. All instructions must be internally consistent — no contradictions.
3. Preserve the agent’s core role definition and tool usage instructions
4. Remove redundant, contradictory, or ambiguous instructions.
5. Preserve essential operational and behavioral requirements.
6. Ensure the updated prompt is concise, coherent, and actionable.
Task:
Produce aprompt diffthat clearly indicates the modifications required to integrate
the proposed rules and principles into the current agent prompt while satisfying the
constraints above. Highlight additions, deletions, and replacements in a structured
format.
24

Preprint. Under review.
RoPE: contrastive analysis
You are analyzing the results of prompt variant re-executions for a failing agent to
extract concrete prompt improvements.
Agent name:{agent name}
Agent role:{agent role}
Original prompt:{original prompt}
Variant execution results:
{variant results}
Each result entry contains:
•Variant prompt:{variant prompt}
•Re-executed trajectory:{trajectory}
•F1 score:{f1}
•Tokens used:{tokens}
Perform a contrastive analysis between successful and failed variants:
1.Operational rules( ∆ρop
i): Extract short-term corrective behaviors — specific,
concrete instructions that directly address the observed failure pattern.
These should be actionable in the agent’s very next execution.
2.Behavioral principles(∆ρbp
i): Extract long-term strategic generalizations —
higher-level guidance distilled from patterns across multiple trajectories.
These shape the agent’s overall approach rather than fixing a single failure.
Respond in the following JSON format:
{
"operational_rules": [
{
"rule": "<concrete instruction>",
"derived_from": "<which variant comparison motivated this rule>"
},
...
],
"behavioral_principles": [
{
"principle": "<strategic guidance>",
"derived_from": "<pattern observed across which variants>"
},
...
],
"updated_prompt": "<full revised prompt integrating all rules and principles \\
into the original, with redundant instructions pruned>"
}
C Datasets
We use the following datasets in our experiments: 1. 2WikiMultihopQA (Ho et al., 2020): a
multi-hop QA dataset that constructs questions using both Wikipedia text and structured
Wikidata, explicitly providing annotated reasoning paths (evidence chains) to ensure and
evaluate true multi-step reasoning across multiple documents.
2. HotPotQA (Yang et al., 2018): a multi-hop QA dataset built from Wikipedia that requires
models to reason over multiple documents while providing sentence-level supporting facts
to enable explainable answer prediction
25

Preprint. Under review.
3. Bamboogle (Press et al., 2023): a small, manually constructed QA dataset of 125 challeng-
ing real-world multi-hop questions designed such that answers cannot be directly retrieved
from search engines, requiring compositional reasoning across multiple pieces of evidence.
4. MusiQue (Trivedi et al., 2022): a multi-hop QA dataset constructed by composing
connected single-hop questions into 2–4 hop reasoning chains, explicitly designed to enforce
genuine multi-step reasoning and reduce shortcut-based answering.
5. HoVer (Jiang et al., 2020): a multi-hop fact verification dataset built from Wikipedia where
models must retrieve evidence across 2–4 documents and determine whether a claim is
supported or not, emphasizing complex many-hop reasoning and evidence extraction.
6. Ambig QA (Min et al., 2020): an open-domain QA dataset derived from NQ-open that
focuses on ambiguous questions, requiring models to generate all plausible answers along
with corresponding disambiguated question rewrites to explicitly resolve ambiguity.
Table 2: Statistics of datasets.
Dataset Train Val Test Total Reasoning
2WikiQA 154K 16K 22K 192K 2-hop (cross-doc)
HotpotQA 90K 7.4K 7.4K 105K 2-hop (bridge/comp)
MuSiQue 19K 2.4K 2.4K 24K 2–4 hop (compositional)
Bamboogle – – 125 125 Multi-hop (adversarial)
HoVer 13K 1.8K 2.7K 18K 2–4 hop (verification)
AmbigQA 10K 2K 2K 14K Ambiguity resolution
D Construction of Training Sets
D.1 Question Types
In this work, we categorize questions into seven types based on the reasoning complexity
they require.
•Bridge multi-hopquestions involve a sequential dependency through an intermediate
entity, for example, “Which university did the author ofThe Old Man and the Sea
attend?”.
•Intersection multi-hopquestions require answers that satisfy multiple independent
constraints, such as “Which scientists won a Nobel Prize and later served as a
university president?”.
•Comparison multi-hopquestions compare attributes across entities after retrieval,
exemplified by “Who was born earlier, Marie Curie or Albert Einstein?”.
•Temporal multi-hopquestions necessitate reasoning over time ordering or temporal
containment, for instance, “Who was president of the U.S. when the Berlin Wall
fell?”.
•Causal multi-hopquestions involve explaining cause–effect chains across events, as
in “Why did the 2008 financial crisis lead to increased banking regulation?”.
•Ambiguousquestions are those with multiple plausible interpretations, such as
“When did the Manhattan Project begin and end?”. This taxonomy allows us to
analyze reasoning strategies across diverse question structures.
D.2 Distribution
Fig. 7 shows the distribution of reasoning types and complexity across the four datasets that
are used to construct training data.
26

Preprint. Under review.
Figure 7: Distribution of Reasoning Types and Complexity of Datasets
E Baselines and Implementation Details
We standardize decoding and prompting across baselines to ensure fair comparison. For
LLMs Qwen-3-4B, 8B, and 14B (Yang et al., 2025a), Qwen2.5-Instruct-7B (Hui et al., 2024),
LLaMA3.1-8B (Grattafiori et al., 2024), and GPT-4o-mini (Hurst et al., 2024), we follow
official guidance. We settemperature=0.6,top−p=0.95,top−k=20,andmin−p=0.
F Additional Experiment Results
F.1 ComparisonHERAwith Direct inference and CoT
Table 3: Additional comparison betweenHERAand baselines without RAG.
Dataset HotpotQA 2WikiQA MusiQue AmbigQA Bamboogle HoVer
Metrics Acc EM F1 Acc EM F1 Acc EM F1 Acc EM F1 Acc EM F1 Acc
Direct Inference w/o RAG
Llama3-3.1-8B-instruct 15.63 9.60 22.13 24.13 16.53 28.67 0.20 0.99 7.18 4.80 4.80 7.96 10.74 8.29 18.55 5.80
Qwen2.5-7B-instruct 19.30 13.17 27.97 24.43 19.70 28.88 2.61 2.40 10.72 8.70 8.00 8.80 16.73 14.04 25.65 3.08
Qwen3-8B 17.83 11.60 26.49 24.37 18.80 28.83 2.69 2.52 10.37 11.20 9.60 18.35 14.19 11.84 23.24 5.90
Qwen3-14B 21.40 14.53 30.68 28.00 20.87 32.69 2.61 2.36 10.03 6.40 4.80 12.97 17.73 14.34 27.98 3.94
CoT w/o RAG
Llama3.1-8B-instruct 16.20 10.14 24.34 24.70 16.74 28.73 0.27 1.40 7.64 15.45 5.20 11.90 11.48 9.00 21.33 15.34
Qwen2.5-7B-instruct 19.93 13.73 28.24 24.37 18.00 27.94 4.34 4.14 11.47 28.00 25.60 38.16 17.88 15.23 28.14 15.93
Qwen3-8B 21.30 15.47 29.36 25.17 17.63 29.51 3.68 3.56 11.03 29.60 23.20 39.83 16.78 14.29 26.73 20.10
Qwen3-14B 19.80 13.27 28.48 25.33 19.03 29.70 2.61 2.44 10.03 12.00 10.40 19.38 14.89 12.99 24.38 22.09
Ours
HERA (Qwen-3-14B) 55.38 52.50 63.03 60.02 59.50 64.77 27.19 26.57 35.82 54.43 48.74 67.81 49.01 46.46 60.53 67.35
HERA (Llama-3.1-8) 53.28 48.01 57.91 52.62 46.70 55.42 25.04 24.46 31.07 48.53 43.46 60.72 46.60 45.22 56.31 60.31
Table 3 highlights the performance ofHERAcompared to direct and CoT inference without
RAG. Across all datasets,HERAexhibits substantial gains over baseline LLMs, with improve-
ments particularly pronounced on multi-hop and ambiguous QA tasks. Notably, HERA
(Qwen-3-14B) consistently outperforms HERA (Llama-3.1-8), reflecting its superior capabil-
ity in complex reasoning scenarios. Across metrics,HERAachieves more than double the
F1 scores of standard CoT baselines on several datasets, indicating that experience-guided
orchestration and prompt evolution provide strong advantages even without retrieval aug-
mentation. Gains are also robust across both in-domain and sparse or ambiguous queries,
demonstrating that structured inter-agent coordination and semantic group-level reasoning
significantly enhance reasoning fidelity over single-agent inference.
F.2 Ablation Studies Using Llama-3.1-8 as the Backbone forHERA
As shown in Figure 8,HERAwith Llama-3.1-8 as bacnkbone demonstrate similar trends
withHERA-Qwen, exceot MusiQue and Bamboogle. Prompt Evolution is the most critical
27

Preprint. Under review.
component. Its removal results in the largest drops across nearly all datasets. For example,
F1 decreases from 58.0 →51.2 on HotpotQA, 55.5 →53.3 on 2WikiQA, and 60.8 →58.0 on
AmbigQA. This indicates that adaptive updating of agent prompts is essential for improving
reasoning performance, particularly on multi-hop and ambiguous queries.
Experience Library removal produces moderate but consistent declines,e.g., F1 drops from
58.0→52.5 on HotpotQA and 55.5 →46.75 on 2WikiQA. This shows that stored experience
contribute to performance stability and cross-instance generalization.
Figure 8: Ablation Studies ofHERAwith Llama-3.1-8 as the backbone.
G Pseudocode
The pseudocode for the proposedHERAis presented below.
Algorithm 1HERA— Top Level
Require:queryq, corpusD, orchestratorπ O, iterationsT
Ensure:optimized experience libraryE, evolved agent prompts{ρ 1, . . . ,ρ K}
1:E ←∅
2:N ←INITIALIZEAGENTS()▷eachN i= (π i,ρi,Ti)
3:fort=1toTdo
4:Sample queryq∼ Q
▷Orchestration
5:Γ←ORCHESTRATE(q,E,N)▷sample topology: order + dependencies
6:τ←EXECUTE(Γ,q,D)▷run agents, produce trajectory
▷Gradient-free GRPO-style orchestrator update
7:ORCHESTRATORUPDATE(q,E,N)
▷RoPE: agent prompt evolution
8:N fail←πO.IDENTIFYFAILEDAGENTS(τ)
9:foreachN i∈ N faildo
10:ROPE(N i,τ)
11:end for
12:end for
13:returnE,{ρ 1, . . . ,ρ K}
28

Preprint. Under review.
Algorithm 2OrchestratorUpdate (GRPO-style)
Require:queryq, experience libraryE, agent setN, group sizeG
1:group←[ ]
2:fori=1 toGdo
3:Γ i←SAMPLETOPOLOGY(q,E,N)
4:τ i←EXECUTE(Γ i,q,D)
5:r task←EVALUATE(τ i, metric=F1)
6:r eff←EVALUATE(τ i, metric=token cost)
7:group.APPEND (Γi,τi,rtask,reff)
8:end for
9:if¬HASMIXEDOUTCOMES(group)then
10: return ▷Reflection and insight extraction is only based on groups that have explicit
successful and failed trajectories
11:end if
12:groupsorted←SORT (group, key= (r task↓,r eff↑)) ▷task perf first, efficiency second
13:I ←π O.REFLECTONGROUP(groupsorted)▷natural language semantic advantages
14:EXPERIENCELIBRARYUPDATE(E,I,q)
Algorithm 3ExperienceLibraryUpdate (ADD / MERGE / PRUNE / KEEP)
Require:experience libraryE, new insightsI, queryq
1:c←CHARACTERIZEQUERY(q)
2:foreach insightz∈ Ido
3:matches←RETRIEVE(E,z,c)
4:ifmatches=∅then
5:E.ADD (c,z,u=0)▷novel insight
6:else ifCOMPLEMENTARY(z, matches)then
7:E.MERGE(z, matches)▷combine semantically similar entries
8:else ifCONFLICTS(z, matches)then
9:low u←FILTERLOWUTILITY(matches)
10:E.PRUNE(low u)▷remove conflicting or stale entries
11:else
12:E.KEEP()▷no change needed
13:end if
14:end for
15:foreach(c,z,u)∈ Edo
16:ifzwas usedandoutcome=successthen
17:u←u+1
18:end if
19:end for
Algorithm 4Orchestrate (experience-guided topology sampling)
Require:queryq, experience libraryE, agent setN
1:c←CHARACTERIZEQUERY(q)
2:E rel←[ ]
3:foreach(c′,z,u)∈ Esorted byudescendingdo
4:ifSIMILAR(c′,c)and¬REDUNDANT(z,E rel)then
5:E rel.APPEND((z,u))▷balance utility and diversity
6:end if
7:end for
8:Γ∼π O(· |q,E rel,N) ▷Γspecifies agents, execution order, dependencies
9:returnΓ
29

Preprint. Under review.
Algorithm 5RoPE — Role-aware Prompt Evolution
Require:failed agentN i, global trajectoryτ
1:B i←GETRECENTFAILURES(N i)▷buffer of recent failed trajectories
2:variants←[ ]
3:foreach axis∈ {efficiency, thoroughness, risk sensitivity}do
4:ρvar←GENERATEVARIANT(ρ i, axis)
5:variants.APPEND(ρvar)
6:end for
7:results←[ ]
8:foreachρvar∈variantsdo
9:N i.ρ←ρvar
10:τvar←REEXECUTE(τ,N i)▷replay full trajectory with variant prompt
11:results.APPEND (ρvar,τvar, EVAL(τvar))
12:end for
13:∆ρop
i←EXTRACTOPERATIONALRULES(B i, results)▷short-term: correct immediate
failure patterns
14:∆ρbp
i←EXTRACTBEHAVIORALPRINCIPLES(B i,results)▷ long-term: generalize across
failure patterns
15:∆ρ i←∆ρop
i+∆ρbp
i
16:ρt+1
i←Π C 
ρt
i⊕∆ρ i▷project onto length and coherence constraintsC
Algorithm 6TopologyMutation (structural fallback)
Require: current topology Γ, failed agents Nfail, agent pool N▷ triggered when trajectories
consistently fail,e.g. F1=0
1:CΓ←[ ]▷candidate topologies
2:foreachN i∈ N faildo
3:// Option A: replace failed agent
4:N′
i←SELECTALTERNATIVE(N, exclude=N i)
5:Γ replace←ΓwithN ireplaced byN′
i
6:C Γ.APPEND(Γ replace )
7:// Option B: augment topology
8:Nnew
i←CREATENEWAGENT()
9:Γ augment ←ΓwithNnew
iinserted afterN i
10:C Γ.APPEND(Γ augment )
11:end for
12:foreachΓ′∈ CΓdo
13:ORCHESTRATORUPDATE(q ,E,N,topology=Γ′)▷feed candidates back into GRPO
loop
14:end for
H Case Studies
We provide case studies in this section, for both success and failed cases. Note: the
 in
every case figure means reuse the agent.
H.1 Case 1 - Comparison Multi-hop QA
Case 1 (Fig. 9) exemplifies an optimal Comparison Multi-hop reasoning pattern: the
orchestrator produces a parallel topology to independent entity facts (each movie’s country)
are retrieved in parallel, followed by a serial aggregation step for comparison. Such a
parallel-then-serial decomposition improves efficiency, minimizes redundancy, and scales
naturally to multiple entities, demonstrating functional differentiation between knowledge
grounding and reasoning control.
30

Preprint. Under review.
Figure 9: Case 1 - Comparison Multi-hop QA
H.2 Case 2 - Causal Multi-hop QA
Figure 10: Case 2 - Causal Multi-hop QA
For this causal multi-hop question (Fig. 10),HERAhandles it by explicitly separating
retrieval of causal evidence from reasoning over uncertainty. First, the system identifies
the target entity (Jesus) and retrieves relevant historical and biblical sources describing the
crucifixion and its contributing factors. Then, the reasoning module evaluates the retrieved
evidence, accounting for ambiguity or contested interpretations, to determine whether the
exact cause is definitively known. By maintaining this serial, dependency-aware workflow,
HERAensures that each reasoning hop builds on grounded evidence, minimizing halluci-
nations and producing a high-confidence answer. This functional differentiation between
experience-grounded retrieval and adaptive causal reasoning allows HERA to robustly
handle multi-hop questions where cause and certainty must be inferred sequentially.
H.3 Case 3 - Temporal Multi-hop Q
Case 3 (Fig.11) requires computing the temporal intersection of two activity periods, which
demands not just parallel retrieval but a subsequent arithmetic-logical operation on the
retrieved intervals. The system correctly decomposes the query and retrieves accurate
individual activity periods: 1914–1938 for Hutchison and 1921–1944 for Howard. However,
the Answer Generator returns “1914–1944” — the union of the two intervals rather than
31

Preprint. Under review.
Figure 11: Case 3 - Temporal Multi-hop QA
their intersection (which would correctly be 1921–1938). This failure reveals a fundamental
limitation: the system lacks an explicit set-theoretic reasoning module. This suggests that
for questions requiring numerical or logical operations over retrieved facts, a dedicated
symbolic computation step is necessary. The failure is not a retrieval failure but a reasoning
composition failure — a distinction with important implications for system design.
H.4 Case 4 - Intersection Multi-hop QA
Figure 12: Case 4 - Intersection Multi-hop QA
For Case 4 (Fig.12), which is a Intersection Multi-Hop type, the error illustrates a critical
challenge in handling multi-entity property overlap. The pipeline’s goal is to identify a
property that applies to both Pavel Alexandrov and Valentin Turchin. Here, the intended
property is “Soviet”. However, the Query Rewriter reformulated the query around “what
they were known for,” shifting focus to professional achievements (mathematician, com-
puter scientist) rather than shared nationality or affiliation. Consequently, the Retriever
retrieved evidence on their careers, and the Evidence Selector highlighted disciplinary roles,
leading the Conclude Agent to output “scientists.”
Insights from this case:
•Intersection multi-hop questions require careful alignment of the property type,
for example nationality, affiliation, or field must be explicitly targeted; otherwise,
retrieval can be biased toward more salient but irrelevant attributes.
•Parallel entity retrieval is insufficient if the property semantics are ambiguous, as
each agent may return correct facts individually, but the intersection reasoning fails
if the property dimension differs.
•Conclude Agent reasoning is sensitive to initial decomposition: even small shifts in
question framing (profession vs. nationality) propagate to a semantically plausible
but incorrect intersection.
32
# Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures (XAMT)

**Authors**: Akhil Sharma, Shaikh Yaser Arafat, Jai Kumar Sharma, Ken Huang

**Published**: 2025-12-15 23:04:48

**PDF URL**: [https://arxiv.org/pdf/2512.15790v1](https://arxiv.org/pdf/2512.15790v1)

## Abstract
The increasing operational reliance on complex Multi-Agent Systems (MAS) across safety-critical domains necessitates rigorous adversarial robustness assessment. Modern MAS are inherently heterogeneous, integrating conventional Multi-Agent Reinforcement Learning (MARL) with emerging Large Language Model (LLM) agent architectures utilizing Retrieval-Augmented Generation (RAG). A critical shared vulnerability is reliance on centralized memory components: the shared Experience Replay (ER) buffer in MARL and the external Knowledge Base (K) in RAG agents. This paper proposes XAMT (Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures), a novel framework that formalizes attack generation as a bilevel optimization problem. The Upper Level minimizes perturbation magnitude (delta) to enforce covertness while maximizing system behavior divergence toward an adversary-defined target (Lower Level). We provide rigorous mathematical instantiations for CTDE MARL algorithms and RAG-based LLM agents, demonstrating that bilevel optimization uniquely crafts stealthy, minimal-perturbation poisons evading detection heuristics. Comprehensive experimental protocols utilize SMAC and SafeRAG benchmarks to quantify effectiveness at sub-percent poison rates (less than or equal to 1 percent in MARL, less than or equal to 0.1 percent in RAG). XAMT defines a new unified class of training-time threats essential for developing intrinsically secure MAS, with implications for trust, formal verification, and defensive strategies prioritizing intrinsic safety over perimeter-based detection.

## Full Text


<!-- PDF content starts -->

Bilevel Optimization for Covert Memory
Tampering in Heterogeneous Multi-Agent
Architectures (XAMT)
Akhil Sharma Shaikh Yaser Arafat Jai Kumar Sharma Ken Huang
Abstract—The increasing operational reliance on complex Multi-
Agent Systems (MAS) across safety-critical domains necessitates
a rigorous assessment of their adversarial robustness. Modern
MAS are inherently heterogeneous, often integrating conventional
machine learning paradigms, such as Multi-Agent Reinforcement
Learning (MARL), with emerging Large Language Model (LLM)
agent architectures utilizing Retrieval-Augmented Generation
(RAG). A critical, shared vulnerability across these disparate
systems is their reliance on centralized, externalized memory
components, whether the shared Experience Replay (ER) buffer
in MARL or the external Knowledge Base ( K) in RAG agents
[1], [2]. This paper proposes the XAMT (Bilevel Optimization
for Covert Memory Tampering in Heterogeneous Multi-Agent
Architectures) framework. XAMT formally models the attack
generation as a novel bilevel optimization (BO) problem. The
Upper Level minimizes the magnitude of the perturbation ( δ),
effectively enforcing covertness, while simultaneously maximizing
the resulting divergence of the system’s behavior (Lower Level)
toward an adversary-defined target policy or response. We provide
rigorous mathematical instantiations of XAMT for two structurally
distinct targets: Centralized Training Decentralized Execution
(CTDE) MARL algorithms and RAG-based LLM agents. Our
theoretical analysis confirms that BO is uniquely suited for crafting
stealthy, minimal-perturbation poisons that circumvent existing
detection heuristics. We propose comprehensive experimental
protocols utilizing established security benchmarks, including
the StarCraft Multi-Agent Challenge (SMAC) and SafeRAG, to
quantitatively demonstrate XAMT’s high effectiveness and sub-
percent poison rates (e.g., ρ≤1% in MARL and ρ≤0.1%
in RAG regimes). This research defines a new, unified class of
training-time threats essential for developing intrinsically secure
MAS.
I. INTRODUCTION
A. The Ubiquity and Fragility of Heterogeneous Multi-Agent
Systems
Multi-Agent Systems (MAS) have transitioned from theoreti-
cal constructs to integral components in real-world, high-stakes
environments. These applications range widely, encompassing
smart transportation networks, autonomous vehicle coordi-
nation, advanced cyber defense mechanisms, and complex
economic modeling [ 3], [4], [5]. The deployment of these
systems in critical infrastructure mandates that their decision-
making processes be robust, reliable, and trustworthy.
A notable architectural shift is the rising heterogeneity within
MAS. Contemporary systems rarely rely on a single, monolithic
learning paradigm. Instead, they often integrate traditional learn-
ing components, such as cooperative MARL agents utilizing
shared state representations or centralized critics [ 6], with
cutting-edge components, such as LLM agents that enhancetheir reasoning and contextual awareness via RAG processes
[2], [7]. While this integration leverages diverse capabilities to
tackle complex, layered tasks [ 8], it concurrently introduces
complex interdependencies and expanded vulnerability surfaces.
This architectural complexity demands sophisticated mech-
anisms for coordination and knowledge sharing. Successful
cooperation in MAS is predicated on the ability of individual
agents to align their actions, often through accessing consistent,
shared information [ 9]. This critical dependency highlights a
shared weakness: the components responsible for storing and
mediating this shared knowledge.
B. Memory as the Unified Critical Attack Surface
The integrity of memory is paramount to collective in-
telligence in MAS. This shared information infrastructure
acts as the system’s ”computational exocortex”—a dynamic,
systematic process that stores, retrieves, and synthesizes
experiences crucial for coordination [ 1]. Disturbingly, this
collective memory layer represents a unified, high-leverage
attack surface across heterogeneous architectures.
In the MARL domain, cooperative success hinges on
frameworks like Centralized Training, Decentralized Execution
(CTDE) [ 6], [10]. These methods typically rely on a centralized
critic function, which is trained using a shared Experience Re-
play (ER) buffer ( D) containing collective state-action histories
[11]. Corrupting the transitions stored in this buffer, particularly
through reward or state poisoning, directly compromises system
integrity and resilience [ 12], [13]. The centralized critic, while
improving stability and credit assignment, becomes a single
point of failure where a localized tamper can affect the global
policy update [6].
Similarly, in LLM-driven agents, the external RAG knowl-
edge base ( K) functions as persistent, external memory [ 2]. This
external data store, often sourced from public or proprietary
documents, presents a novel attack vector distinct from the chal-
lenges of core model weight manipulation [ 14], [15]. Attacks
here involve injecting malicious, low-quality, or misleading
texts into K[7], [16]. Systemic security failure, therefore, often
results not from the compromise of a single agent’s reasoning
capabilities, but from the corruption of the foundational shared
context or experience upon which the entire collective relies
[1].arXiv:2512.15790v1  [cs.CR]  15 Dec 2025

103
102
101
100
Poison Rate  (%) [log scale]
1020304050607080Effectiveness (% Utility Drop / ASR)
Effectiveness vs. Covertness (Lower  is More Covert)
MARL Baseline
XAMT-RL
RAG Baseline
XAMT-RAGFig. 1:XAMT at a glance—effectiveness vs. covertness.Left:Attack Success / Utility Drop versus poison rate (ρ, log scale) for MARL (XAMT-RL) and
RAG (XAMT-RAG) compared to non-optimized baselines. XAMT achieves high impact at sub-percent ρ.Right:Semantic deviation (proxy for detectability in
RAG) versus ASR. XAMT-RAG attains higher ASR with lower semantic drift, reflecting the bilevel “minimal perturbation” objective.
C. The Need for Covert, Optimization-Driven Attacks
Traditional poisoning attacks, while often effective, fre-
quently involve large-scale manipulations, such as extensive
label flipping, high-magnitude reward alterations, or the in-
troduction of a high percentage of synthetic samples [ 12],
[17]. Such attacks are inherently detectable via anomaly
detection systems or simple threshold checks designed to detect
massive distributional shifts [ 18]. For an attack to pose a truly
sophisticated threat in a monitored, production environment,
it must prioritizecovertness. A covert attack necessitates
that the perturbation δintroduced into the memory system
must be minimal (e.g., small Lpnorm on numerical data) or
semantically plausible (clean-label text) [19], [20].
Achieving maximal functional damage with minimal de-
tectable perturbation is a computationally challenging constraint
satisfaction problem. Bilevel Optimization (BO) provides the
necessary mathematical rigor to solve this highly constrained,
hierarchical adversarial problem [ 21], [22]. BO allows the
attacker to explicitly model the victim system’s learning process
(the Lower Level) as a constraint on the overall attack objective
(the Upper Level) [ 23]. This capability is critical because it
permits the selection of theminimum required perturbation
δthat guarantees the maximal target utility drop after the
victim completes its training iteration θ∗(δ)[24], [25]. BOis proven effective in generating clean-label poisoning points
with imperceptible distortions to reduce certified robustness
guarantees, establishing it as the ideal mathematical tool for
fulfilling thecovertrequirement of the XAMT framework [ 19],
[20].
D. Contributions and Novelty
The XAMT framework makes the following critical con-
tributions to adversarial machine learning and MAS security
research:
1)Unified Framework:The paper proposes XAMT, which
is, to our knowledge, the first unified bilevel optimization
framework that explicitly spans covert poisoning across
structurally heterogeneous memory components ( M) in
MAS, specifically CTDE-style MARL experience replay
buffers and RAG knowledge bases.
2)Formalization of Covertness:A generalized definition
of the Minimal Perturbation Constraint R(δ) is formal-
ized, tailored to the modality of M. This constraint
usesLpnorms for numerical MARL data and semantic
distance ( Dsem) metrics for textual RAG data, rigorously
integrating stealth into the BO objective.
3)Dual Instantiations:Detailed mathematical specifica-
tions are provided for XAMT-RL (targeting centralized
critics in CTDE) and XAMT-RAG (targeting knowledge

bases), demonstrating the framework’s broad applicabil-
ity.
4)Experimental Protocol:Comprehensive, comparative
experimental protocols are detailed using established
security benchmarks, including SMAC and SafeRAG,
to quantify the effectiveness and covertness of XAMT
against state-of-the-art multi-agent and RAG architec-
tures.
II. FORMALTHREATMODEL ANDASSUMPTIONS
We formalize the adversarial scenario underpinning XAMT
through a threat model that defines attacker capabilities, victim
system properties, and the operational scope of memory
tampering attacks. This model provides the necessary context
for the bilevel optimization framework and clarifies the practical
feasibility of the proposed threats.
A. Attacker Capabilities
Attacker notation:we denote the attacker asA.
The attacker is modeled as acapable but constrained
adversarywith the following explicit capabilities.
1) Knowledge Model: Aoperates under awhite-boxas-
sumption regarding the victim’s learning architecture but not
necessarily its runtime data streams. Specifically:
•Structural Knowledge: Aknows the victim’s algorithmic
class (e.g., QMIX/MAPPO for MARL; dense retrieval
with a specific LLM for RAG), hyperparameters (including
λ), learning-rate schedules, and the centralized memory
architectureM[12], [14].
•Training Data Distribution: Ahas access to a surrogate
dataset Dsurrogate that approximates the distribution of the
victim’s clean memory M. This is a standard assumption
in data poisoning research, enabling the attacker to
simulate the lower-level optimization [18], [19].
•No Real-time Observability: Adoesnotrequire real-
time observation of the victim’s training trajectory or
access to intermediate gradients during execution. This
distinguishes XAMT from dynamic adversarial examples
and focuses ontraining-time poisoning.
2) Access Model: Apossesseswrite-accessto the central-
ized memory componentMwith the following constraints:
•Injection Point: Acan inject perturbations δintoM
prior to or duringthe victim’s training phase, but cannot
modify the victim’s model parameters θdirectly [ 13], [25].
For MARL, this corresponds to tampering with offline-
collected experience batches before they enter the replay
buffer. For RAG, this corresponds to injecting documents
intoKbefore retriever fine-tuning or before inference-time
retrieval.
•Rate Constraint:The injection quantity is limited to a
sub-percent fraction:
ρ=|δ|
|M|≤1%(MARL), ρ≤0.1%(RAG)
[7], [13]. This reflects realistic scenarios where large-scale
corruption would trigger integrity checks.•Covertness Mandate: Amust ensure δevadesstatic,
heuristic detection: anomaly thresholds on Lpnorms for
numerical data [ 24] and semantic filters (e.g., perplexity or
embedding proximity checks) for textual data [29], [37].
3) Computational Resources: Ahas sufficient compute to
solve the bilevel problem:
•Hessian Approximation:Access to GPU-accelerated
solvers (e.g., PBGD, BIGRAD) capable of approximating
∇δLAvia implicit differentiation [21], [23].
•Surrogate Training:Ability to run multiple inner-loop
training episodes on Dsurrogate to estimate θ∗(δ)during the
outer-loop optimization [20], [25].
B. Victim System Assumptions
Victim notation:we denote the victim system asS.
The target system is assumed to be abenign, well-
intentioned learnerwith these properties.
1) Learning Process Integrity:The victim executes its
standard training algorithmVwithout adversarial hardening:
•Optimization Fidelity: Vfaithfully minimizes LSvia
stochastic gradient descent (e.g., Adam, RMSProp) on the
poisoned memoryM+δ[12], [42].
•No Poison-Aware Defenses: Sdoesnotemploy
Byzantine-robust aggregation, data sanitization, or ad-
versarial training during the initial training phase [ 18],
[30]. This reflects the status quo in most open-source
MARL/RAG implementations.
2) Memory Architecture Vulnerabilities: Srelies oncen-
tralized, weakly protected memory:
•CTDE MARL:Uses a shared experience replay buffer
Daccessible to centralized critic updates. No per-agent
isolation or integrity checks on stored transitions [ 6], [11].
•RAG Systems:Employs a flat knowledge base Kwith
vector-based retrieval. No robust provenance verification
or retrieval-time safety filtering beyond semantic similarity
thresholds [2], [14].
3) Detection Mechanisms: Semploys onlysuperficial, post-
hoc integrity checks:
•Numerical Monitoring:Simple threshold-based anomaly
detection on reward/state magnitudes (e.g., ∥δRL∥∞<
ϵdetect) [24].
•Textual Filtering:Basic semantic filters that flag docu-
ments with high perplexity or low BERTScore relative to
the corpus [ 29], [37]. XAMT explicitly optimizes R(δ)
to remain below such heuristics.
C. System Scope and Limitations
The XAMT threat model is bounded by the following
assumptions to ensure practical relevance.
1) Temporal Scope:Attacks aretraining-time only. Evasion
of runtime adversarial detection (e.g., during MARL execution
or RAG inference) is out-of-scope, focusing on poisoning
foundational memory before deployment [13], [25].

TABLE I: Threat Model Success Criteria
Metric XAMT-RL (MARL) XAMT-RAG (LLM)
Effectiveness Utility Drop ≥40% atρ≤
1%ASR≥90%atρ≤0.1%
Covertness ∥δRL∥∞ <0.05 ,
∥δRL∥2<0.1Dsem<0.15 , Perplexity ≤
Baseline+10%
Evasion Bypass threshold ϵdetect =
0.1Evade BERTScore filter
τ= 0.85
2) Architectural Scope:
•MARL:Limited to CTDE algorithms with centralized
critics (e.g., QMIX, MAPPO, VDN). Fully decentralized
or purely on-policy methods without shared buffers (e.g.,
independent Q-learning) are not direct targets [6], [10].
•RAG:Focuses on dense retrieval with vector embeddings.
Sparse retrieval (e.g., BM25) or graph-based retrieval
requires alternative definitions of Dsem, but remains
compatible with the BO framework [42], [48].
3) Attack Goal Scope: Aoptimizes for asingle, predefined
target:
•MARL:A target policy Tthat induces measurable utility
drop (e.g., win-rate decline) [36], [41].
•RAG:A specific adversarial response YTto a trigger
promptP tr, measured via ASR [7], [37].
Multi-target or dynamic objectives are deferred to future
work.
D. Success Metrics and Attack Objectives
Asucceeds iff: (1) the effectiveness threshold is met, (2)
the covertness constraints are satisfied, and (3) the perturbation
evades victim detection heuristics. This tripartite criterion is
explicitly encodedin the XAMT bilevel objective (Section ??),
where λbalances LAandR(δ) to satisfy all constraints
simultaneously.
III. BACKGROUND ANDRELATEDWORK
A. Foundations of Bilevel Optimization in Adversarial Machine
Learning
Bilevel optimization (BO) is a hierarchical optimization
paradigm characterized by two nested optimization problems.
The solution to the inner (lower-level) problem constrains or
dictates the objective of the outer (upper-level) problem [ 21].
This structure is essential for modeling adversarial interactions
in machine learning, particularly where the attacker’s actions
influence the target model’s training dynamics [23].
In the context of data poisoning, the victim model’s training
process is formalized as the lower-level problem, which pro-
duces the optimal model parameters θ∗(δ)given the poisoned
dataM+δ [18]. The attacker (Upper Level) then optimizes
the poison δto maximize the adversarial loss LA(θ∗), subject
to the constraint of minimal perturbation, λR(δ) [24]. Previous
research has effectively leveraged BO to generate clean-label
poisoning points with minimal input edits, achieving high
targeted damage—such as reducing the average certified radius
(ACR) of a target class by over 30%—without massive accuracyreductions that would alert the victim [ 19], [20]. This body of
work confirms that BO is not merely a tool for optimization but
a necessary methodology for achieving the required stealth and
precision mandated by the XAMT framework. Furthermore,
integrating BO within deep learning frameworks has been made
feasible through recent advances like Differentiating through
Bilevel Optimization Programming (BIGRAD), which extends
existing single-level optimization programming approaches to
handle the complex gradient calculations required by nested
problems [23].
B. Adversarial Attacks in Multi-Agent Reinforcement Learning
(MARL)
Multi-Agent Reinforcement Learning (MARL) models are
increasingly recognized as vulnerable to adversarial perturba-
tions [ 26]. Attacks often manifest through manipulating agent
states, actions, or rewards during training or execution [ 3],
[27].
A significant subset of MARL algorithms adheres to
the Centralized Training, Decentralized Execution (CTDE)
paradigm [ 6], [10]. This approach, while facilitating improved
coordination by granting the critic access to global information
(often including the true system state) during training, creates
centralized components that are single points of failure [ 6]. The
centralized critic’s reliance on the shared Experience Replay
(ER) buffer Dmakes this memory unit a critical attack target
[11].
Reward poisoning in offline MARL environments has been
shown to be particularly effective, where modifying the rewards
in a pre-collected dataset can compel rational agents to
adopt a malicious target policy, often resulting in a Markov
Perfect Equilibrium (MPE) under the poisoned dynamics [ 12],
[28]. The architectural selection of CTDE, specifically the
dependence on centralized critics, inherently creates a high-
leverage target within the shared experience buffer. A localized,
covert memory tamper δRLwithin this buffer can dispro-
portionately corrupt the global policy update, making it an
efficient mechanism for policy manipulation. This exploitation
is further observed in attacks that are stealthy during offline
training but trigger catastrophic failures during subsequent
online fine-tuning (O2O) [ 13], [25]. These sophisticated attacks
often leverage BO techniques to promote value over-estimation
or distribution shifts within the critic network, effectively
exploiting the centralized critic’s potential for bias and variance
[6], [25]. Existing MARL poisoning work, however, typically
focuses on reward or state manipulation within MARL alone
and does not provide a cross-domain, unified treatment that also
regularizes covertness across heterogeneous memory modalities
as a first-class objective.
C. Adversarial Attacks in LLM Agent Memory (RAG)
With the rise of agentic systems driven by Large Language
Models (LLMs), new vulnerabilities related to external memory
have emerged. Retrieval-Augmented Generation (RAG) sys-
tems, designed to ground LLM responses in external knowledge,
introduce the knowledge base ( K) as a primary attack surface

[14], [15]. Attacks such as AgentPoison and PoisonedRAG
inject malicious demonstrations or texts directly into the RAG
system’s long-term memory or knowledge base [7], [14].
For RAG attacks to be effective and evade sophisticated
detection systems (e.g., retrieval safety assessments such as
RevPRAG [ 2], [37]), the injected text must be highly covert
[29]. This requires maintaining the semantic and syntactic in-
tegrity of the poisoned text while maximizing the probability of
retrieval given a specific trigger [ 30]. Covert textual poisoning
relies on techniques such as clean-label attacks and synonym
substitution, where only minimal text insertions or replacements
are used to maintain fluency and plausibility [ 31], [32]. This
architectural shift from numerical, vector-based optimization
(as in MARL states) to linguistic/semantic optimization (as in
RAG text) means the definition of covertness must evolve
from a simple Lpnorm constraint to a metric based on
semantic distance ( Dsem) [29]. Prior work such as AgentPoison
and PoisonedRAG focuses exclusively on the RAG setting
(e.g., constrained trigger and document optimization) and does
not offer a unified bilevel formulation that simultaneously
spans CTDE-MARL ER buffers and RAG KBs with a shared,
modality-aware covertness regularization. XAMT is designed
to fill exactly this cross-domain gap.
D. The Convergence Gap: Justification for XAMT
Existing adversarial machine learning research tends to seg-
regate threats into distinct domains, focusing either on MARL
policy manipulation or LLM/RAG knowledge corruption. A
critical research gap exists in providing a unified, optimization-
based framework that treats the memory unit Mgenerically,
while simultaneously addressing the specific dependencies and
modalities of heterogeneous agent architectures.
The XAMT framework is designed to bridge this gap. By
leveraging Bilevel Optimization, XAMT provides a mathemat-
ically robust method to guarantee that minimal perturbations
inM(regardless of whether Mis a numerical vector or
textual document) achieve the maximal targeted strategic
damage across diverse multi-agent learning paradigms. This
unified approach is necessary for designing defenses capable
of addressing systemic vulnerabilities across modern, complex
MAS deployments.
IV. THEXAMT FRAMEWORK: FORMALIZINGCOVERT
MEMORYTAMPERING
The XAMT framework mathematically formalizes the adver-
sarial interaction between a stealthy attacker and a target MAS
learning process utilizing shared memory. The attack is defined
by the objective of maximizing a predefined security failure
metric (effectiveness) while strictly adhering to a constraint on
the perturbation magnitude (covertness).
A. System Model and Attacker Goals
The target system is modeled as aHeterogeneous Multi-
Agent System ( S)comprising a collection of agents, A=
{A1, . . . , A N}, interacting within an environment E. All agents
rely on anAbstract System Memory ( M)for learning andcoordination. Mserves as the generic term for the critical,
centralized information repository: either the centralized expe-
rience buffer D(for MARL) or the retrieval knowledge base
K(for RAG).
The system updates its internal parameters θ(which may
represent policies πin MARL or LLM/retriever weights in
RAG) using aVictim Learning Process ( V), such that the
resulting optimal parameters areθ∗=V(M).
The objective of theCovert Tampering Attacker ( A)is to
introduce a minimal perturbation δintoM. This perturbation
must be optimized such that the resulting system parameters
θ∗(M+δ) maximally deviate toward a malicious target
behaviorT.
B. The XAMT Bilevel Optimization Formulation
The attacker’s strategy is inherently hierarchical, leading
to the generalized bilevel optimization (BO) formulation for
memory perturbation:
min
δLA(θ∗(δ)) +λR(δ)(1)
subject to
θ∗(δ)∈arg min
θLS(θ,M+δ).(2)
Equation (1)defines theupper-level attacker objective, while
Equation (2)represents thelower-level system learning process.
1)Lower-Level Objective ( LS):This function represents
the benign training or updating process of the victim
system. In MARL, LSmight be the policy iteration loss
(e.g., QMIX or MAPPO loss) [ 6]. In RAG, LScould
model the training loss for the retriever embeddings or the
maximum likelihood loss of the generator. The solution
θ∗(δ)represents the system parameters optimized on the
corrupted memoryM+δ.
2)Upper-Level Objective ( LA):This loss quantifies the
attacker’s success, aiming to maximize the divergence
of the resulting optimal parameters θ∗(δ)from the
desired system performance. For MARL, this involves
maximizing the negative utility (e.g., minimizing win rate
[36]). For RAG, this means maximizing the probability
of generating a specific target response [37].
3)Covertness Constraint ( R(δ) ):This is the crucial
regularization term, governed by the hyperparameter
λ >0 , which explicitly balances the trade-off between
attack effectiveness ( LA) and the cost of manipulation
[24].R(δ) is specialized based on the memory modality:
R(δ) =(
∥δ∥p
p,Mnumerical
Dsem(M,M+δ),Mtextual(3)
where numerical memory corresponds to MARL experi-
ence buffers, and textual memory corresponds to RAG
knowledge bases.
For numerical data (e.g., states or rewards), Lpnorms
(typically L∞orL2) ensure that the perturbation magni-
tude remains below detection thresholds [ 24]. For textual
data, the semantic distance Dsem(calculated via embed-
ding differences or linguistic models) guarantees the

malicious text maintains semantic plausibility, satisfying
the clean-label requirement [29].
C. Computational Solution and Differentiability
The primary computational challenge in solving the XAMT
BO formulation lies in computing the gradient of the upper-
level objective with respect to the perturbation δ, specifically
∇δLA. Since the lower-level solution θ∗(δ)is implicitly
defined as the result of a potentially non-convex optimization
process, traditional gradient methods are insufficient. The
Implicit Function Theorem (IFT) approach is required [38].
This differentiation necessitates calculating the gradient flow
from the lower-level solution θ∗(δ)back to the upper-level
variable δ[38]. This typically involves the inverse Hessian
(H−1) of the lower-level loss functionL Swith respect toθ:
∇δLA(θ∗(δ)) =−∇ δ,θLS(θ∗(δ), δ)TH−1∇θLA(θ∗(δ))
In large-scale deep learning applications prevalent in MARL
and LLM systems, the direct computation or inversion of the
Hessian matrix Hcan be computationally intractable [ 39]. The
resulting memory and time complexity limits the scalability of
minimal-perturbation attacks [39].
To address these limitations and render XAMT practical
for large-scale, non-convex problems, specialized optimization
techniques are adopted. Methods such as Penalty-Based Bilevel
Gradient Descent (PBGD) transform the constrained bilevel
problem into a sequence of more manageable single-level
optimization problems (often minimax problems), improving
scalability without requiring strict lower-level strong convexity
[21], [22]. Furthermore, differentiating through bilevel prob-
lems can utilize techniques like BIGRAD, which leverages
vector-Jacobian products for efficient gradient calculation,
enabling end-to-end learning within modern machine learning
frameworks [ 23]. The practicality of XAMT is predicated on
the ability of these specialized solvers to efficiently approximate
the inverse Hessian, extending the applicability of BO to non-
convex, high-dimensional parameter spaces common in deep
MARL and RAG systems.
Table II summarizes the generalized components of the
XAMT framework.
V. INSTANTIATINGXAMTACROSSHETEROGENEOUS
DOMAINS
The versatility of XAMT is demonstrated by specializing its
mathematical components for two distinct, high-impact agent
architectures.
A. XAMT-RL: Covert Tampering of Centralized MARL Memory
1) Memory Perturbation and Lower-Level Objective:The
attack focuses on applying a minimal perturbation δRLto the
transitions stored inD, typically targeting the states t, action
at, or reward rtcomponents [ 12], [13]. The Lower-Level Loss
(LS) models the centralized critic update. For Q-learning based
methods, this is the temporal difference (TD) error minimization
Fig. 2:Conceptual Architecture of the XAMT Bilevel Optimization
Framework.The diagram illustrates the nested optimization problem: the
Upper Level (Attacker A) minimizes the perturbation magnitude R(δ) while
maximizing adversarial impact LA(θ∗), where θ∗is the resulting system
parameter set. The Lower Level (Victim System S) models the routine learning
process minimizingL Son the corrupted memoryM+δ.
for the centralized critic Qtot(τ, u;θ Q), where τis the joint
history anduthe joint action:
LS(θQ,D+δ RL) =E (τ,u,r)∼D+δ RLh 
Ytarget−Q tot(τ, u;θ Q)2i
where Ytarget represents the one-step lookahead target value
derived from the poisoned experience buffer.
2) Upper-Level Objective and Covertness:The Upper-Level
Loss ( LA) seeks to compel the decentralized actors, whose
policies πare derived from the corrupted critic Qtot, to adopt a
nefarious target policy T[41]. This is achieved by maximizing
the resulting policy’s divergence from high expected return,
J(πA).
LA(θQ) =−J(π∗(θQ)) +λR(δ RL)
The covertness R(δRL)is quantified by the standard Lpnorm
constraint applied to the vector perturbation of states or rewards,
guaranteeing that the change is minute and undetectable by
simple threshold monitoring [24].
The architectural implication of this BO formulation is pro-
found: XAMT-RL actively exploits the inherent vulnerabilities
introduced by the centralized critic in CTDE systems [6]. By
optimally solving for δRL, the attacker identifies the small set
of critical experience points that, when minimally perturbed,
maximize the critic’s value over-estimation for suboptimal,
adversarial actions. This effectively guides the decentralized
actors toward the adversarial target policy during online execu-
tion, consistent with the observed catastrophic failure during
offline-to-online transfer (O2O) attacks [25].
B. XAMT-RAG: Covert Knowledge Base Injection
The XAMT-RAG instantiation focuses on LLM agents aug-
mented with an external RAG knowledge base K[29], [42].
This attack targets the textual knowledge repository.

TABLE II: Unified XAMT Framework Definition
Component Upper-Level (AttackerA
Loss)Lower-Level (Victim System
SLoss)Covertness (R(δ))
Objective minδLA(θ∗(δ)) +λR(δ) θ∗(δ)∈argminθLS(θ,M+δ) Minimization of Perturbation
CostR(δ)
Goal Drive system to target
policy/response (T) with
minimal input change (δ).Find optimal parametersθthat
minimize learning error on
corrupted memory.Stealth against detection (Low
Lpnorm or Semantic
Plausibility).
Optimization Method Implicit Gradient Descent (e.g.,
using PBGD/BIGRAD)Stochastic Gradient Descent
(SGD/Adam)Penalty Termλadjustment.
1) Memory Perturbation and Lower-Level Objective:The
perturbation δRAG involves injecting malicious text snippets
intoK[7], [14]. To maintain covertness, this must often employ
semantic substitution(clean-label attack) to ensure the text
is grammatically sound and retains surface-level plausibility,
avoiding detection by simple content filters [30], [31], [32].
The Lower-Level Loss ( LS) models the RAG system’s mech-
anism, typically involving both the retriever Rand the LLM
generator G.LScan be modeled as the maximum likelihood
loss of the LLM generator given the context Cretrieved from
K+δ RAG and the user promptP.
θ∗
LLM(δRAG)∈argmin
θLGeneration (θ, P,C(K+δ RAG))
In this formulation, θrepresents the LLM and/or retriever
parameters that adapt to the corrupted knowledge base K+
δRAG .
2) Upper-Level Objective and Semantic Covertness:The
Upper-Level Loss ( LA) seeks to maximize the probability of
generating a specific adversarial target response YTwhen the
user provides an optimized triggerP tr[37].
LA(θ∗
LLM) =−logP(Y T|Ptr,C(K+δ RAG)) +λR(δ RAG)
The key innovation in XAMT-RAG is the specialized covertness
constraint R(δRAG), which must enforcesemantic plausibility
andminimal quantity.
R(δRAG) =D semantic (K,K+δ RAG) +β· |δ RAG|
Here, Dsemantic (e.g., derived from embedding distance or per-
plexity scores) ensures the malicious text maintains the original
topic’s semantics and fluency [ 29], [30], making it challenging
to filter. The term |δRAG|minimizes the overall volume of
the injected text, aiming for extremely low poison rates (for
example,ρ≤0.1%in RAG-style regimes) [7].
The BO formulation for RAG allows the attacker to jointly
optimize the textual content of δRAG (covertness) and the re-
trieval features (effectiveness). This guarantees that the minimal
semantic manipulation simultaneously maximizes the retriever’s
likelihood of selecting the malicious document over potentially
millions of clean documents inK[14], [33].
VI. EXPERIMENTALPROTOCOLS ANDEVALUATION
A. MARL Evaluation: XAMT-RL on Cooperative Domains
1) Environment and Victim Selection:The evaluation envi-
ronment must capture the complexity of real-world multi-agentcoordination. TheStarCraft Multi-Agent Challenge (SMAC)
is utilized as the standardized benchmark, providing partially
observable, cooperative tasks (micromanagement challenges)
that require intricate team coordination [ 43], [44]. Experiments
should target maps where coordination failure is particularly
costly, such as asymmetric or high-unit count scenarios (e.g.,
2cvs64zg). The victim algorithms selected are QMIX and
MAPPO, representing the dominant CTDE methodologies that
rely on shared critics and experience buffers [40], [45], [46].
2) Attack Objective and Effectiveness:TheTarget Pol-
icy (T)for XAMT-RL must result in a clear, measurable
degradation of cooperative utility. This target policy is defined
as actions leading to coordinated self-destruction, failure to
engage the enemy, or fixation on an inefficient sub-task, thereby
significantly reducing the system’s ability to maximize its win
rate [36], [44].
Effectiveness Measurement:The primary metric isPolicy
Utility Drop, quantified by the percentage reduction in the
average Win Rate achieved by the poisoned agent compared
to the clean, benign agent [36], [43].
3) Covertness Measurement:The covertness of XAMT-RL
poisons is measured numerically:
•Poison Rate ( ρ):The percentage of transitions in the
experience replay buffer Dthat are perturbed. For MARL
environments, practical constraints such as batch sizes,
replay ratios, and exploration noise typically make feasible
poison rates fall in a sub-percent regime. In our protocols,
we therefore target ρ≤1% for XAMT-RL, while still
seeking to drive significant utility drops.
•Perturbation Magnitude ( L∞, L2):The maximum mag-
nitude of the perturbation vector δRLon rewards or state
observations, ensuring the perturbation falls below stan-
dard anomaly detection thresholds typically employed in
sensor-rich environments [24].
B. RAG Evaluation: XAMT-RAG on Knowledge Corruption
1) Benchmark and Victim Selection:To evaluate XAMT-
RAG, specialized adversarial robustness benchmarks designed
for RAG systems are essential, such asSafeRAGorRAGuard
[16], [47], [48]. These benchmarks simulate noisy retrieval
settings and test against targeted attacks like conflict and
toxicity injection [ 16]. The victim architecture is a standard
RAG pipeline comprising a vector-based retriever (e.g., using
robust embeddings) and a sophisticated, commercially relevant
LLM [29], [42].

TABLE III: Instantiation of XAMT Components Across Heterogeneous Architectures
XAMT Component XAMT-RL (MARL CTDE) XAMT-RAG (LLM Agent)
Abstract Memory (M) Experience Replay BufferD
(State/Action/Reward Vectors)RAG Knowledge BaseK(Unstructured
Text Documents)
Lower-Level Parameters (θ) Centralized Critic Weights (θ Q) LLM/Generator Weights (Optional) &
Retriever Embeddings
Lower-Level Loss (L S) TD Error Minimization (e.g., QMIX Loss) Retrieval/Generation Fidelity Loss
Upper-Level Loss (L A) Maximizing Target Policy Divergence or
Utility DropMaximizing Target Response Likelihood
(ASR) given trigger
CovertnessR(δ) LpNorm onδ RL(Vector Perturbation) Semantic DistanceD sem+ Length
Constraint|δ RAG|(Textual Perturbation)
Fig. 3:Heterogeneous Targets of the XAMT Attack.This dual-path diagram contrasts the two system architectures and their common vulnerability: (A)
XAMT-RLtargets the shared Experience Replay Buffer ( D) used by the centralized critic in CTDE MARL. (B)XAMT-RAGtargets the external Knowledge
Base (K) used by the retriever to augment the LLM agent’s generation. In both cases, the perturbation ( δ) is covertly injected into the centralized memory layer.
2) Attack Objective and Effectiveness:TheTarget Response
(YT)is defined as a specific, factually incorrect, or toxic
response that the attacker mandates the LLM to generate upon
presentation of a specific trigger promptP tr[37].
Effectiveness Measurement:The primary metric is the
Attack Success Rate (ASR), calculated as the fraction of
triggered queries Ptrthat successfully elicit the malicious
target responseY T[37].3) Covertness Measurement:The covertness of XAMT-RAG
poisons is measured through linguistic metrics:
•Semantic Similarity and Fluency:Metrics such as BERTScore
or Perplexity are used to quantify the linguistic distance
between the clean text and the poisoned text M+δ RAG
[29], [30]. Minimizing this distance ensures the poison
evades detection via simple content filters or embedding
proximity checks, confirming its status as a clean-label
attack.

Fig. 4:Effectiveness and Covertness of XAMT-RL in SMAC.(Left) A learning curve plot comparing the average Win Rate (Utility) vs. Training Steps for:
A clean QMIX agent (Baseline), a QMIX agent trained with a uniform random poisoning attack, and a QMIX agent trained with the BO-optimized XAMT-RL
attack. This plot is expected to show XAMT achieving high utility drop post-convergence. (Right) A bar chart comparing XAMT-RL performance across
different attack types, plotting the achieved Policy Utility Drop against the required Poison Rate (ρ≤1%) and Perturbation Magnitude (L ∞).
•Poison Rate ( ρ):The ratio of malicious texts injected
relative to the total size of K. XAMT aims for extreme
covertness in RAG regimes, targeting poison rates on the
order ofρ≤0.1%[7].
VII. DISCUSSION ANDDEFENSIVEIMPLICATIONS
A. Analysis of XAMT Efficacy and Scalability
The fundamental efficacy of XAMT stems from its targeting
strategy: exploiting centralized memory components that are es-
tablished for efficiency and coordination (e.g., CTDE critics [ 6]
and RAG knowledge retrieval [ 42]). The Bilevel Optimization
structure provides the mathematical ability to target ”critical
instances” within the memory—those data points that maximize
the influence on the global policy update—even when they
constitute an infinitesimally small portion of the overall dataset
[20], [49]. This confirms that memory centralisation, while
optimizing performance, creates an asymmetrical advantage
for a covert attacker.
However, XAMT faces inherent scalability trade-offs, partic-
ularly due to the computational complexity of the ImplicitFunction Theorem (IFT) approach. While modern solvers
like PBGD and BIGRAD improve feasibility [ 22], [23], the
necessity of calculating or approximating the inverse Hessian
remains a computational bottleneck [ 39]. This constraint is
magnified in systems with extremely large memory components,
such as massive RAG knowledge bases ( K), or continuous, high-
dimensional MARL state spaces [ 50], [51]. Future research
is required to develop highly efficient, stochastic BO solvers
capable of handling the massive scale of modern deep learning
architectures.
B. Implications for Trust and System Integrity
The covert nature of XAMT poses a significant challenge
to existing trust and verification mechanisms in MAS. Formal
verification methods, which attempt to ensure system behavior
adheres to formal specifications [ 52], are often computationally
constrained by the continuous state spaces and complex, non-
linear dynamics of deep MARL models [50].
For RAG systems, the issue is not complexity but scale. The
sheer volume, constant updates, and unstructured nature of

TABLE IV: Proposed Evaluation Metrics for XAMT Performance
Metric Category MARL Agents (XAMT-RL) LLM RAG Agents
(XAMT-RAG)Goal
Effectiveness Policy Utility Drop (∆Win
Rate) [36]Attack Success Rate (ASR)
[37]Maximized
Covertness (Numerical) Perturbation Magnitude (L ∞)
[24]Poison Rate (ρ) [7] Minimized
Covertness (Semantic) N/A (Vector) Semantic Distance (D sem) /
Perplexity [29]Minimized
Efficiency Computational Cost perδ
Update (Hessian approximation
complexity)Required number of
optimization steps to converge
ASRMinimized
Fig. 5:Attack Success Rate (ASR) vs. Covertness for XAMT-RAG.(Left) A line graph plotting ASR versus the Poison Rate ( ρ, typically ≤0.1% ). This
graph is designed to show XAMT-RAG achieving a significantly higher ASR at extremely low poison rates compared to non-BO baseline RAG poisoning
methods. (Right) A scatter plot visualizing the trade-off between semantic covertness (e.g., Perplexity/Semantic Distance) and ASR for different poison text
generation strategies, demonstrating XAMT’s ability to minimize semantic deviation while maximizing attack success.
massive knowledge bases make comprehensive logic verifica-
tion infeasible [ 33], [35]. Consequently, integrity checks that
rely on simple heuristics, such as anomaly detection thresholds
onLpnorms or basic semantic filters, are insufficient against
BO-optimized perturbations. XAMT demonstrates that an at-
tacker can meticulously craft memory perturbations (textual or
numerical) that operate below these superficial detection layerswhile guaranteeing catastrophic policy divergence, highlighting
a fundamental breakdown in existing integrity assessment
protocols.
C. Initial Defense Considerations
The findings from the XAMT framework underscore the
necessity of moving beyond external, perimeter-based defenses.
External safety inspection agents or dedicated safety modules

often introduce scalability bottlenecks and inherent fragility
[46]. Future defense strategies must prioritizeintrinsic safety,
embedding resilience directly into the learning agents through
robust training or sophisticated reinforcement learning tech-
niques [46], [53].
A critical area for resilience is the development of adaptive
defense mechanisms. Systems must be capable of implement-
ingadaptive zero-shot learningand dynamically adjusting
their defense posture based on observed, novel attack patterns
[5], [53]. Such defenses must combine multiple validation
modalities—e.g., using Lpanomaly detection for numerical
data coupled with semantic plausibility checks (like perplexity
monitoring) for textual data—to counter the dual covertness
strategy employed by XAMT.
Furthermore, the memory layer ( M) itself should be con-
ceptualized as a network that requires sophisticated resilience
mechanisms. Drawing inspiration from distributed systems,
the concept of ”self-healing” is pertinent [ 54], [55]. Future
work should focus on developing memory healing protocols
designed to detect and purge BO-optimized malicious memory
units. This requires mechanisms that can recover the integrity
or connectivity of the collective memory post-attack, mirroring
resilient strategies in critical infrastructure networks like power
grids or transport systems [55].
VIII. ETHICALCONSIDERATIONS ANDDUAL-USESCOPE
XAMT is explicitly proposed as ared-teaming and se-
curity analysisframework rather than a blueprint for real-
world exploitation. The attack formulations and experimental
protocols are intended to help system designers understand and
harden the critical memory surfaces of MAS, not to encourage
attacks on deployed production systems. In any public release
of XAMT-inspired tooling, we recommend restricting access
to synthetic benchmarks and sandboxed environments (e.g.,
SMAC, SafeRAG-style testbeds) and avoiding the publication
of turnkey code that directly targets proprietary, safety-critical
ER buffers or knowledge bases in the wild. Consistent with
emerging practice in AI security research, our scope is to enable
defenders and system builders to reason about the worst-case
memory-tampering risks so that more robust, intrinsically safe
MAS architectures can be designed.
IX. CONCLUSION ANDFUTUREWORK
This paper introduced XAMT, a novel bilevel optimization
framework for designing highly covert, minimal-perturbation
memory tampering attacks across heterogeneous multi-agent
architectures, specifically MARL and RAG systems. XAMT
leverages the hierarchical nature of BO to formally solve the
constrained optimization problem of minimizing perturbation
costR(δ) while maximizing the divergence of the resulting
system behavior toward an adversarial target LA. We detailed
the mathematical instantiations for XAMT-RL and XAMT-
RAG, establishing a unified methodology for exploiting cen-
tralized memory components, whether numerical or textual.
Our analysis confirms the viability of exploiting architectural
choices (such as centralized critics in CTDE and vector-basedknowledge retrieval in RAG) as high-leverage points of sys-
temic failure.
The feasibility of XAMT poses a severe threat to the trust-
worthiness of deployed MAS. We have provided comprehensive
experimental protocols utilizing SMAC and SafeRAG to bench-
mark this novel threat and establish robust evaluation metrics
for both effectiveness and covertness.
Future Directions:
1)Defense Implementation and Validation:Developing
and empirically validating robust, differentiated memory
validation mechanisms explicitly tailored to counter BO-
optimized perturbations, combining Lpanomaly detec-
tion with sophisticated semantic plausibility checks.
2)Transferability Analysis:Rigorously investigating the
transferability of XAMT poisons between different victim
algorithms (e.g., QMIX to MAPPO in MARL) and across
various retriever-LLM combinations in RAG systems,
providing insights into generalizable adversarial features.
3)Continuous BO Solvers for Scale:Dedicating research
to develop computationally efficient, stochastic BO solvers
capable of handling continuous action/state spaces com-
mon in MARL and the large non-convex optimization
problems inherent in modern LLM systems, thereby
making the generation of sophisticated attacks more
scalable.

REFERENCES
REFERENCES
[1]Ferber, J.Multi-Agent Systems: An Introduction to Distributed Artificial
Intelligence. Addison-Wesley, 1999.
[2]Tan, Z. et al. Knowledge Database or Poison Base? Detecting RAG
Poisoning Attack through LLM Activations (RevPRAG).arXiv preprint
arXiv:2411.18948.
[3]Xu, J. et al. Recent Advances in Multi-Agent Systems Security.arXiv
preprint arXiv:2503.13962.
[4]Zhang, Q. et al. Resilient Consensus Control for Multi-Agent Systems:
A Comparative Survey.Sensors, 23(6): 2904, 2023.
[5]Ferber, J. On cooperation in multi-agent systems. InProc. of the 5th
Intl. Conf. on Autonomous Agents, 1999.
[6]Sun, T. et al. Centralized Critic is Not Strictly Beneficial: Theoretical
Analysis on Centralized Training for Decentralized Execution.arXiv
preprint arXiv:2408.14597.
[7]Chan, B. et al. AgentPoison: Red-teaming LLM Agents via Poisoning
Memory or Knowledge Bases.NeurIPS, 2024.
[8]Feng, J. et al. The Agent Challenge: A Comprehensive Survey on
Autonomous Agent Systems.arXiv preprint arXiv:2402.03578.
[9]Doran, P. et al. On Cooperation in Multi-Agent Systems.Tech. Report:
Queen Mary and Westfield College, 1997.
[10] Wang, J. et al. Robustness Testing Framework for Multi-Agent Re-
inforcement Learning by Attacking Critical Agents.arXiv preprint
arXiv:2306.06136.
[11] Wu, T. et al. Attentive Experience Replay.AAAI, 2020.
[12] Liu, J. et al. Targeted Reward Poisoning Attacks in Offline Multi-Agent
Reinforcement Learning.AAAI, 2023.
[13] Wang, W. et al. Stealthy Data Poisoning Attacks on Offline-to-Online
Reinforcement Learning.PMC, 2024.
[14] Zou, Y . et al. PoisonedRAG: Backdoor Attack on Retrieval-Augmented
Generation.USENIX Security, 2025.
[15] Zhou, Y . et al. CPA-RAG: Covert Poisoning Attacks on Retrieval-
Augmented Generation.arXiv preprint arXiv:2505.19864.
[16] Yao, S. et al. SafeRAG: Benchmarking Security in Retrieval-Augmented
Generation of Large Language Model.GitHub Repository, 2025.
[17] Liu, J. et al. Adversarial Attacks on Multi-Agent Systems.MDPI
Electronics, 2025.
[18] Shokri, A. et al. Witches’ brew: Industrial scale data poisoning via.arXiv
preprint, 2022.
[19] Zhang, J. et al. Data Poisoning Attacks on Certified Adversarial
Robustness.arXiv preprint arXiv:2012.01274.
[20] Ghofrani, A. et al. Bilevel Optimization-Based Single-Class Attack.arXiv
preprint arXiv:2503.22759.
[21] Shen, H. et al. On Penalty-based Bilevel Gradient Descent Method.arXiv
preprint arXiv:2302.05185.
[22] Lu, Z. et al. A Penalty Method for Bilevel Optimization.arXiv preprint,
2023.
[23] Mohseni, S. et al. Differentiating through Bilevel Optimization Program-
ming (BIGRAD).AAAI, 2022.
[24] Carlini, N. et al. Towards Evaluating the Robustness of Neural Networks.
IEEE S&P, 2017.
[25] Wang, W. et al. Stealthy Data Poisoning Attacks on Offline-to-Online
Reinforcement Learning.UAI, 2024.
[26] Zhang, Y . et al. State-Action Joint Attack (SAJA) for Multi-Agent Deep
Reinforcement Learning.arXiv preprint arXiv:2510.13262.
[27] Mohammadi, M. et al. Implicit Poisoning Attacks in Two-Agent
Reinforcement Learning.AAMAS, 2023.[28] Liu, J. et al. Targeted Reward Poisoning Attacks in Offline Multi-Agent
Reinforcement Learning.arXiv preprint arXiv:2206.01888.
[29] Shafran, A. et al. Machine against the RAG: Jamming Retrieval-
Augmented Generation with Blocker Documents.arXiv preprint
arXiv:2406.05870.
[30] Ren, P. et al. Dirichlet Neighborhood Ensemble: An Effective Defense
against Synonym Substitution-Based Adversarial Attacks.ACL, 2021.
[31] Alzoubi, R. et al. Clean-Label Adversarial Text Attack using Synonym
Substitution.arXiv preprint arXiv:2404.04130, 2024.
[32] You, W. et al. LLMBkd: Black-Box Backdoor Attack on Large Language
Models.arXiv preprint arXiv:2309.07172.
[33] Zeng, T. Challenges in Large-Scale RAG Deployment: A Perspective on
Trustworthiness and Verification.arXiv preprint arXiv:2507.18910.
[34] Niu, Y . Formal Verification for Safe Deep Reinforcement Learning.
Preprints.org, 2023.
[35] Giunchiglia, E. et al. Temporal Logic-Based Specification and Verification
of Trust Models.ACM SAC, 2005.
[36] Liu, J. et al. Adversarial Attacks on Multi-Agent Systems.MDPI
Electronics, 2025.
[37] Tan, Z. et al. Knowledge Database or Poison Base? Detecting RAG
Poisoning Attack through LLM Activations (RevPRAG).arXiv preprint
arXiv:2411.18948.
[38] Ghadimi, S. et al. Multi-Agent Reinforcement Learning and the Implicit
Function Theorem.Preprints.org, 2024.
[39] Ghadimi, S. et al. A Proximal-Gradient Method for Robust Optimization
via Bilevel Programming.SIAM J. Optim., 2023.
[40] Mao, H. et al. A Comprehensive Survey of Multi-Agent Reinforcement
Learning.MDPI Electronics, 2025.
[41] Mohammadi, M. et al. Implicit Poisoning Attacks in Two-Agent
Reinforcement Learning: Adversarial Policies for Training-Time Attacks.
AAMAS, 2023.
[42] Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks.NeurIPS, 2020.
[43] Samvelyan, M. et al. The StarCraft Multi-Agent Challenge.arXiv preprint
arXiv:1902.04043, 2019.
[44] Samvelyan, M. et al. The StarCraft Multi-Agent Challenge.NeurIPS,
2019.
[45] Kurbiel, J. et al. MA-Trace: A High-Scalability Actor-Critic Algorithm for
Multi-Agent Reinforcement Learning.arXiv preprint arXiv:2111.11229.
[46] Geng, Y . et al. Embedding Safety Awareness into Agents via Reinforce-
ment Learning.arXiv preprint arXiv:2508.03864.
[47] Qi, H. et al. RAGuard: A Benchmark for Robustness Evaluation of RAG
Systems in Political Fact-Checking.arXiv preprint arXiv:2502.16101.
[48] Yao, S. et al. SafeRAG: Benchmarking Security in Retrieval-Augmented
Generation of Large Language Model.arXiv preprint arXiv:2501.18636.
[49] Liu, J. et al. Recent Advances in Multi-Agent Systems Security.arXiv
preprint arXiv:2503.13962.
[50] Zhang, Y . et al. Challenges of Scaling Multi-Agent Reinforcement
Learning to Real-World Systems.arXiv preprint arXiv:2302.05007.
[51] Yu, X. et al. Understanding the Training Dynamics of Multi-Agent
Actor-Critic Algorithms.arXiv preprint, 2024.
[52] Giunchiglia, E. et al. Temporal Logic-Based Specification and Verification
of Trust Models.ACM SAC, 2005.
[53] Alghazali, F. et al. Adaptive Zero-Shot Hierarchical Multi-Agent Rein-
forcement Learning (AZH-MARL) for Cyber Defense.PMC, 2024.
[54] Quattrociocchi, W. et al. Self-Healing Networks: Redundancy and
Structure.PLoS ONE, 9(2): e87986, 2014.
[55] Quattrociocchi, W. et al. Self-Healing Networks: Redundancy and
Structure.PLoS ONE, 9(2): e87986, 2014.
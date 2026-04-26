# Mol-Debate: Multi-Agent Debate Improves Structural Reasoning in Molecular Design

**Authors**: Wengyu Zhang, Xiao-Yong Wei, Qing Li

**Published**: 2026-04-22 07:03:25

**PDF URL**: [https://arxiv.org/pdf/2604.20254v1](https://arxiv.org/pdf/2604.20254v1)

## Abstract
Text-guided molecular design is a key capability for AI-driven drug discovery, yet it remains challenging to map sequential natural-language instructions with non-linear molecular structures under strict chemical constraints. Most existing approaches, including RAG, CoT prompting, and fine-tuning or RL, emphasize a small set of ad-hoc reasoning perspectives implemented in a largely one-shot generation pipeline. In contrast, real-world drug discovery relies on dynamic, multi-perspective critique and iterative refinement to reconcile semantic intent with structural feasibility. Motivated by this, we propose Mol-Debate, a generation paradigm that enables such dynamic reasoning through an iterative generate-debate-refine loop. We further characterize key challenges in this paradigm and address them through perspective-oriented orchestration, including developer-debater conflict, global-local structural reasoning, and static-dynamic integration. Experiments demonstrate that Mol-Debate achieves state-of-the-art performance against strong general and chemical baselines, reaching 59.82% exact match on ChEBI-20 and 50.52% weighted success rate on S$^2$-Bench. Our code is available at https://github.com/wyuzh/Mol-Debate.

## Full Text


<!-- PDF content starts -->

Mol-Debate: Multi-Agent Debate Improves Structural Reasoning in
Molecular Design
Wengyu Zhang♠Xiao-Yong Wei♡,♠, *Qing Li♠
♠The Hong Kong Polytechnic University♡Sichuan University
weng-yu.zhang@connect.polyu.hk
{cs007.wei, qing-prof.li}@polyu.edu.hk
Abstract
Text-guided molecular design is a key capa-
bility for AI-driven drug discovery, yet it re-
mains challenging to map sequential natural-
language instructions with non-linear molecu-
lar structures under strict chemical constraints.
Most existing approaches, including RAG, CoT
prompting, and fine-tuning or RL, emphasize
a small set of ad-hoc reasoning perspectives
implemented in a largely one-shot generation
pipeline. In contrast, real-world drug discov-
ery relies on dynamic, multi-perspective cri-
tique and iterative refinement to reconcile se-
mantic intent with structural feasibility. Moti-
vated by this, we proposeMol-Debate, a gen-
eration paradigm that enables such dynamic
reasoning through an iterative generate-debate-
refine loop. We further characterize key chal-
lenges in this paradigm and address them
through perspective-oriented orchestration, in-
cluding developer-debater conflict, global-local
structural reasoning, and static-dynamic inte-
gration. Experiments demonstrate that Mol-
Debate achieves state-of-the-art performance
against strong general and chemical baselines,
reaching 59.82% exact match on ChEBI-20 and
50.52% weighted success rate on S2-Bench.
Our code is available at https://github.
com/wyuzh/Mol-Debate.
1 Introduction
Drug discovery is a focal point of AI for science,
posing a critical multimodal NLP challenge de-
spite the broad success of large language models
(LLMs) (Edwards et al., 2022; Cao et al., 2022;
Hurst et al., 2024; Zhang et al., 2025). As shown
in Figure 1, the fundamental obstacle is a text-
structure gap: translating sequential human in-
structions into the non-linear structural domain of
molecules, which is defined by ring topology and
functional group connectivity (Jang et al., 2025). A
*Corresponding author
Preprint.
“Themoleculeisa1,4-benzoquinonesubstitutedbyamethoxy,hydroxyandmethylgroupatpositions2,3and5,respectively.”CC1=CC(=O)C(=C(C1=O)O)OC
(a) Sequential instructions(b) Non-linear structures
SMILEStextmolecule(2)(1)Figure 1: Illustration of the text-structure gap: (a) se-
quential human instructions must be grounded in (b)
non-linear structural domain of molecules. (1) The ring
substructure is split across non-adjacent spans in the
SMILES sequence. (2) The methoxy substructure ap-
pears as a single word in the text but corresponds to
multiple atoms and bonds in the molecule.
range of methods have been proposed to enhance
structural awareness in reasoning. These include
attention-based alignment between language and
molecular features (Vaswani et al., 2017; Schwaller
et al., 2019); Retrieval-Augmented Generation
(RAG), which supplies relevant external chemical
context (Lewis et al., 2020; Li et al., 2024a); and
Chain-of-Thought (CoT) strategies that encourage
structured reasoning (Wei et al., 2022). Domain-
specific models trained via fine-tuning (FT) and
reinforcement learning (RL) on large chemistry
corpora have also been developed to build founda-
tional chemical competence (Edwards et al., 2022;
Pei et al., 2023; Zhao et al., 2025a).
While effective, existing methods are often con-
strained to a few ad-hoc reasoning perspectives.
For example, RAG mainly strengthens external
knowledge retrieval, while pretraining and fine-
tuning frameworks largely emphasize token align-
ment and data-driven pattern learning. As a result,
these approaches are often tied to the specific per-
spective they are designed to exploit. Real-world
drug discovery, however, demands a more integra-
tive approach. The progress hinges on chemists’arXiv:2604.20254v1  [cs.AI]  22 Apr 2026

ability to iteratively reconcile multiple viewpoints,
such as high-level intent, structural constraints, and
design trade-offs, through collaborative critique
and refinement. While multi-agent systems (MAS)
are a promising avenue for integrating the diverse
perspectives required in drug design, their applica-
tion raises a critical new issue (Guo et al., 2024;
Swanson et al., 2025). The system must navigate a
delicate balance: preserving agent independence to
avoid groupthink while ensuring enough coordina-
tion to reach a coherent decision (Li et al., 2024c).
If agents are too independent, their outputs frag-
ment; if coordination is too strict, the system may
converge prematurely and miss subtle errors.
This new challenge positions Multi-Agent De-
bate (MAD) (Du et al., 2023) as a compelling,
yet unexplored, solution for molecular design, as
its core mechanism of debate naturally balances
independent reasoning with collaborative coordi-
nation. Motivated by this, we present a pilot study
to investigate if debate-driven agentic reasoning
can more effectively bridge the text-structure gap.
To this end, we introduceMol-Debate, an itera-
tive generation framework (see Figure 2) where
agents collaborate in a generate-debate-refine loop,
integrating diverse and dynamic perspectives to im-
prove structural fidelity. Mol-Debate addresses the
integration of perspectives at three distinct levels.
•Developer-Debater Conflict: A known issue
where agents endowed with extensive domain
expertise can experience a corresponding limita-
tion in general reasoning and linguistic adaptabil-
ity for debating (Liu et al., 2024a). To mitigate
this asymmetry, we employ dedicated Debater
Agents. Their function is to inject common-sense
reasoning into the process, critically assessing
how well candidate molecules satisfy the seman-
tic intent of the prompt, thereby guaranteeing
chemical robustness and precise user alignment.
•Global-Local Structural Reasoning: Standard
models typically process molecules as global, lin-
ear strings, which often causes them to overlook
fine-grained chemical rules (Edwards et al., 2022;
Dubey et al., 2024). Mol-Debate bridges this gap
by enforcing a Local Structural Perspective. This
perspective integrates a microscopic analysis of
chemical constraints, shifting the model’s reason-
ing beyond mere “textual probability”. Conse-
quently, the system can simultaneously respect
both the global molecular syntax and the localchemical reality of its substructures. For exam-
ple, this perspective helps Debater Agents prefer
candidates with reasonable molecular weight and
a reasonable number of rotatable bonds.
•Static-Dynamic Integration: Current pipelines
(e.g., RAG (Li et al., 2024a), CoT (Jang et al.,
2025), and chemical LLMs (Zhao et al., 2025b))
are fundamentally static, executing a one-way
generation where errors are final. In contrast,
Mol-Debate establishes a dynamic refinement
system. It treats inter-agent disagreement not as
failure, but as a learning signal. By creating a
closed feedback loop, this consensus dissonance
actively reformulates task constraints, allowing
the system to refine its understanding of the query
during inference, rather than passively accepting
the initial output.
Our contributions are threefold:
•We identify the gap between prior methods
that rely on a few ad-hoc reasoning perspec-
tives and real-world drug discovery that re-
quires dynamic, multi-perspective critique
and refinement.
•We proposeMol-Debate, a generation-by-
debate paradigm for molecular design.
•We characterize and address key challenges in
this paradigm. Mol-Debate achieves the state-
of-the-art performance and consistently high
chemical validity against strong baselines.
2 Related Work
2.1 Molecular Design Models
Molecular design models connect natural language
and molecular structures through retrieval, trans-
lation, editing, or optimization. Early work such
as Text2Mol (Edwards et al., 2021) learns joint
text-molecule representations for caption-based re-
trieval, while later methods extend this line to bidi-
rectional translation and in-context generation with
LLMs (Li et al., 2024a,b; Liu et al., 2024b, 2025).
In parallel, chemical foundation models such as
MolT5 (Edwards et al., 2022), ChemDFM (Zhao
et al., 2025b), ChemDFM-R (Zhao et al., 2025a),
and Chem-R (Wang et al., 2025) improve text-
level chemical competence through large-scale pre-
training, reasoning supervision, distillation, and
reinforcement learning. Other methods further

Developer AgentExaminer Agent
Refiner AgentDebate AgentCandidates
Task query𝒳!𝑓Dev𝒫"𝑓Examℰ"𝑓Deb𝑓Debater#…
𝑓Judge𝑓Re4inerif 𝒞"	<	𝜃Stopping CriteriaFinal molecule 𝒮∗	SelectOne𝒫𝓇&#
𝒫"&#𝑓Debater'𝒫#"𝒫'"𝒟"if SI=1Developers𝒫"𝒳",𝒫",	ℰ"
𝑓SI𝒳"&#𝒫"𝒫"&#𝒳",𝒫",𝒫"&#,ℰ",𝒟"𝒞"
A1,4-benzoquinonesubstitutedbyamethoxy,hydroxyandmethylgroupatpositions2,3and5,respectively.
if SI=0if 𝒞"	≥	𝜃(a) Developer-Debater Conflict(c) Static-Dynamic Integration(b) Global-Local Structural Reasoning×	𝑟maxFigure 2: An overview of our Mol-Debate framework, an iterative generation framework where agents collaborate
in a generate-debate-refine loop, integrating diverse and dynamic perspectives: (a) Developer-Debater Conflict, (b)
Global-Local Structural Reasoning, and (c) Static-Dynamic Integration.
strengthen the text-structure interface through ex-
plicit alignment or structural guidance, such as
MolXPT (Liu et al., 2023), GeLLM3O (Dey et al.,
2025), and MSR (Jang et al., 2025). Recent work
has also begun exploring agentic systems for molec-
ular design. MT-Mol (Kim et al., 2025) uses multi-
agent, tool-guided reasoning for molecular opti-
mization, MultiMol (Yu et al., 2025) combines ex-
pert agents for multi-objective optimization, and
PharmAgents (Gao et al., 2025) builds a broader
multi-agent ecosystem for virtual pharmaceutical
workflows. While these methods show the promise
of agent-based molecular discovery, they mainly
focus on optimization, editing, or broader drug-
discovery pipelines. Our work is orthogonal to
these model-centric advances: we treat such mod-
els as pluggable generators and focus on a debate-
based framework that aims to orchestrate multiple
models to better align text-structure intent with
structural feasibility in molecular design.
2.2 Multi-Agent Debate
Multi-Agent Debate (MAD) enhances robustness
and reasoning diversity by allowing multiple agents
to generate, critique, and revise candidate outputs
through adversarial interaction (Du et al., 2023).
BDoG (Zheng et al., 2024) extends debate to mul-
timodal settings by introducing blueprint-guidedstructures that organize evidence and reduce opin-
ion collapse. DRAG (Hu et al., 2025) integrates
debate into both retrieval and generation stages of
RAG to mitigate compounding errors, reducing
hallucination through role-specialized proponents,
opponents, and judges. Unlike prior MAD settings,
our goal is to narrow the molecular text-structure
gap utilizing multi-perspective reasoning and local
structural evidence.
3 Mol-Debate
In this section, we presentMol-Debate, an itera-
tive generation framework designed to bridge the
text-structure gap in molecular design, with the
overview shown in Figure 2. Due to the limited
space, detailed implementation of Mol-Debate can
be found in Appendix A.
3.1 Overview
Formally, given a design instruction X, Mol-
Debate maps Xto a molecule S, represented as
a SMILES sequence (Weininger, 1988):
f:X 7→ S
f=f Dev◦fExam◦fDeb◦fRefiner
X={x i},S={s j}.(1)
Mol-Debate solves this mapping via an iterative

multi-agent processf. At roundr, the state is
Tr= (Xr,Pr,A),(2)
where Xris the current instruction, Pris a
pool of candidate molecules, and Adenotes the
agent set. At the end of each round, both the
pool and the instruction may be updated to produce
(Xr+1,Pr+1), enabling progressive refinement.
3.2 Developer-Debater Conflict
A key challenge is balancing chemistry-grounded
proposal generation with flexible semantic adjudi-
cation. Expert generators can produce chemically
plausible candidates yet remain less adaptive to
nuanced instruction language during critique. Mol-
Debate addresses this by decoupling agents:De-
veloper Agentfocuses on professional chemical
exploration, while dedicatedDebate Agentinjects
general reasoning to evaluate intent satisfaction and
select instruction-aligned candidates.
3.2.1 Developer Agent
To ensure candidates are grounded in professional
chemistry principles, Mol-Debate uses a set of N
Developer agents. Each Developer proposes a local
subset of Mncandidates Sand provides an explicit
design rationaleR.
fDev:Xr7→ Pr
Pr=N[
n=1Pn=N[
n=1Mn[
m=1{(Sn,m,Rn,m)}.
(3)
Here,Pndenotes the molecule-rationale sub-
set from the n-th Developer. At this stage, the
pool is designed for breadth, prioritizing chemi-
cally plausible exploration but not yet guaranteeing
full constraint satisfaction. Appendix A.2 shows
the prompt for LLMs in Developer Agent.
3.2.2 Debate Agent
Given candidate pool Pr, the Debate Agent coordi-
nates KDebaters and a Judge to select candidates
Pr+1that best match the semantic intent ofXr:
fDeb:Xr× Pr7→ Pr+1
fDeb=f Debater◦fJudge.(4)
Debater.The debating employs Kparallel Debater
agents to critically evaluate the candidates against
the structural evidence. Each debater analyzes theinstruction Xr, the candidate pool Pr, and selects
a subset of sufficient moleculesPr
k:
f(k)
Debater:Xr× Pr7→ Pr
k
Pr
k={S |(S,R)∈ Pr}.(5)
Appendix A.2 shows the prompt for LLMs in
Debater Agent.
Judge.The Judge aggregates the debate results
Dr={Pr
k}K
k=1proposed by the Kindependent
Debaters. The updated pool Pr+1is obtained by
intersection when possible, with a union fallback:
fJudge:Dr7→ Pr+1
Pr+1=(TK
k=1Pr
kifT̸=∅SK
k=1Pr
kotherwise.(6)
This coupling ensures that candidate generation
and intent matching are jointly enforced, thereby
reducing the likelihood that the system accepts
chemically plausible yet instruction-misaligned
molecules.
3.3 Global-Local Structural Reasoning
The second perspective bridges global textual rep-
resentations and local chemical constraints. While
the Developer and Debate Agent can enforce in-
tent, text-level debate alone cannot reliably detect
local invalidity. Mol-Debate therefore introduces
anExaminer Agentthat provides deterministic
local structural evidence, and upgrades the Debate
Agent to incorporate this evidence during selection.
3.3.1 Examiner Agent
The Examiner Agent acts as an objective auditor
using deterministic tools (e.g., RDKit (Landrum,
2013)) to validate candidates and compute molec-
ular properties. It maps the candidate pool Prto
examination reportsE:
fExam :Pr7→ Er
Er={e(S)|(S,R)∈ Pr}
e(S) ={(d, ϕ d(S))|d∈Π}.(7)
The examination function e()performs a check
by property analysis ϕ(). Let Πbe the set of chem-
ical descriptors d. This evidence anchors decision-
making in verifiable chemical constraints, provid-
ing a Local Structural Perspective for downstream
Agents. More implementation details of the chemi-
cal descriptors are provided in Appendix A.3.

3.3.2 Local Structure-Grounded Debating
With structural evidence Eravailable, Mol-Debate
upgrades debate to jointly consider semantic intent
and local structural feasibility.
Thef Debin Eq. (4) is upgraded as:
fDeb:Xr× Pr× Er7→ Pr+1,(8)
in which thef Debater in Eq. (5) is upgraded as:
f(k)
Debater:Xr× Pr× Er7→ Pr
k,(9)
where each debater analyzes the instruction Xr,
the candidate pool Prand structural examina-
tion report Er, then selects a subset of sufficient
moleculesPr
k.
3.4 Static-Dynamic Integration
The third perspective addresses the limitation
of static generation. Even with local structure-
grounded debate, disagreement can persist due to
underspecified instructions. Mol-Debate therefore
introduces aRefiner Agentthat turns debate dis-
sonance into a learning signal by reformulating the
instruction. The Refiner Agent analyzes the de-
bate outcomes and structural evidence to diagnose
ambiguity or missing constraints and produces an
updated instruction:
fRefiner :Xr× Pr× Pr+1× Er× Dr7→ Xr+1.
(10)
The refined instruction is fed back to the Developer
Agent fDevto generate improved candidates in the
next round, closing a feedback loop. Appendix A.2
shows the prompt for LLMs in Refiner Agent.
3.5 Consensus Evaluation and Termination
Mol-Debate iterates until either the debaters reach a
sufficient level of agreement and the candidate pool
collapses to a single candidate, or the maximum
number of rounds rmaxis reached. We quantify
agreement using a consensus score Crdefined as
the average similarity across subsets {Pr
k}K
k=1se-
lected byKDebaters:
Cr=1 K
2K−1X
i=1KX
j=i+1|Pr
i∩ Pr
j|
|Pr
i∪ Pr
j|,(11)
and define the stopping indicatorSIas
fSI:Cr× Pr+17→ SIr
SIr=(
1(C r≥θ∧ |Pr+1|= 1)∨r≥r max
0 otherwise.
(12)3.6 Perspective-Oriented Orchestration
Algorithm 1 summarizes the full workflow. Across
rounds, Mol-Debate first expands candidates via
Developer Agents and prunes them via Debate
Agent (Developer-Debater Conflict), then grounds
selection with deterministic local structural evi-
dence from the Examiner Agent (Global-Local
Structural Reasoning), and finally resolves residual
ambiguity through instruction reformulation by the
Refiner Agent (Static-Dynamic Integration). This
orchestration enables reliable text-guided molec-
ular design by iteratively coupling semantic in-
tent, structural feasibility, and iterative refinement
within a debate-driven agentic reasoning loop. Af-
ter termination of the debate, we output a single
molecule via SelectOne() by uniformly sampling
one candidate from the final stabilized poolPr+1.
Algorithm 1Mol-Debate Pipeline
Require:X0,K,r max,θ
Ensure:Optimal MoleculeS∗
1:r←0,SI0←0,P0←f Dev(X0)// Eq. (3)
2:whileSIr̸= 1do// Eq. (12)
3:Er←f Exam(Pr)// Eq. (7)
4:Dr← {f(k)
Debater(Xr,Pr,Er)}K
k=1//Eq. (9)
5:Pr+1←f Judge(Dr)// Eq. (6)
6:ifC r< θthen// Eq. (11)
7:// Eq. (10)
8:Xr+1←f Refiner (Xr,Pr,Pr+1,Er,Dr)
9:Pr+1← Pr+1∪f Dev(Xr+1)// Eq. (3)
10:else
11:Xr+1← Xr
12:end if
13:r←r+ 1
14:end while
15:S∗←SelectOne(Pr+1)
4 Experimental Setup
4.1 Datasets
We evaluate caption-to-molecule generation on
ChEBI-20(Edwards et al., 2021), which contains
33010 molecule-caption pairs and requires generat-
ing a SMILES string from a textual description. We
follow the evaluation protocol in (Edwards et al.,
2022), reporting SMILES exact matches (EM),
BLEU scores, Levenshtein distance (Lev.) (Miller
et al., 2009), fingerprint similarity (MA.F, RD.F,
MO.F), Fréchet ChemNet Distance (FCD), and
SMILES validity (Val.). We further evaluate

Method Approach EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
Llama-3.1-8B-Instruct (Meta, 2024) Naive Gen 0.0045 0.1873 73.8242 0.5838 0.3489 0.2644 6.9835 0.4167
GPT-4o mini (OpenAI, 2024) Naive Gen 0.0173 0.1947 63.2667 0.5988 0.3629 0.3027 8.6415 0.5148
GPT-5 mini (OpenAI, 2025) Naive Gen 0.1773 0.4046 47.1945 0.8829 0.7280 0.6575 1.2423 0.7942
Gemma-4-31b-it (Google, 2026) Naive Gen 0.0452 0.3786 49.8358 0.7067 0.4932 0.4169 3.8080 0.7061
MolT5-large (EMNLP, 2022) PT+FT 0.3110 0.8540 16.0710 0.8340 0.7460 0.6840 1.2000 0.9050
BioT5-base (EMNLP, 2023) PT+FT 0.4130 0.8670 15.0970 0.8860 0.8010 0.7340 0.4300 1.0000
MolReGPT (TKDE, 2024) RAG 0.2800 0.8570 17.1400 0.9030 0.8050 0.7390 0.5900 0.8990
Mol-Instruction (ICLR, 2024) FT 0.0400 0.3000 39.4200 0.4400 0.2900 0.2500 - 1.0000
Mol-R1 (arXiv, 2025) CoT+Distill+RL 0.2340 0.6410 32.9400 0.8230 0.6840 0.6120 - 0.8470
MSR (ACL, 2025) CoT+FT 0.4210 0.8780 12.76000.9240 0.8560 0.8040 0.2600 0.9820
ChemDFM-v1.5-8B (Cell Press, 2025) PT+FT 0.5379 0.6086 41.1333 0.9388 0.8798 0.8314 0.1957 0.9770
ChemDFM-R-14B (arXiv, 2025) PT+CoT+FT 0.4948 0.6733 33.9506 0.9313 0.8592 0.8092 0.2236 0.9733
Chem-R-8B (arXiv, 2025) CoT+FT+RL 0.4385 0.8648 16.1830 0.9204 0.8398 0.7827 0.2747 0.9664
Mol-Debate (Ours)Mol-Debate 0.59820.7059 30.8294 0.9601 0.9103 0.8694 0.1258 0.9973
Table 1: Overall performance of Mol-Debate and other baselines on caption-to-molecule generation task. Blue
marks the best-perform ingmethod, light blue represents the second-best-perform ing method.
Method ApproachMolCustom MolEdit MolOptAverage
SR↑WSR↑ SR↑WSR↑ SR↑WSR↑ SR↑WSR↑
Llama-3.1-8B-Instruct (Meta, 2024) Naive Gen 0.0471 0.0291 0.2951 0.1605 0.3394 0.1341 0.2272 0.1079
GPT-4o mini (OpenAI, 2024) Naive Gen 0.1063 0.0696 0.5059 0.3549 0.3574 0.2727 0.3232 0.2324
GPT-5 mini (OpenAI, 2025) Naive Gen 0.5045 0.3323 0.8865 0.6033 0.8454 0.4069 0.7455 0.4475
Gemma-4-31b-it (Google, 2026) Naive Gen 0.1876 0.1217 0.7291 0.5003 0.6933 0.5156 0.5366 0.3792
MolT5-large (EMNLP, 2022) PT+FT 0.0217 0.0135 0.2251 0.0244 0.4465 0.0489 0.2311 0.0289
BioT5-base (EMNLP, 2023) PT+FT 0.0224 0.0158 0.1938 0.0306 0.5095 0.0799 0.2419 0.0421
ChemDFM-v1.5-8B (Cell Press, 2025) PT+FT 0.0835 0.0560 0.3401 0.2571 0.3876 0.2934 0.2704 0.2022
ChemDFM-R-14B (arXiv, 2025) PT+CoT+FT 0.1001 0.0662 0.4845 0.3317 0.2658 0.1849 0.2835 0.1943
Chem-R-8B (arXiv, 2025) CoT+FT+RL 0.2107 0.1419 0.7955 0.5429 0.8561 0.5789 0.6208 0.4213
Mol-Debate (Ours)Mol-Debate 0.3965 0.2671 0.9399 0.6432 0.9203 0.6052 0.7522 0.5052
Table 2: Overall performance of Mol-Debate and other baselines on the text-based open molecule generation task.
open-ended text-guided molecule generation on
S2-Bench(Li et al., 2025a), which includes 5000
queries per subtask for customized molecule gen-
eration (MolCustom), editing (MolEdit), and op-
timization (MolOpt), resulting in 15000 queries
totally. We report the Success Rate (SR) and
Weighted Success Rate (WSR).
4.2 Baselines
We evaluate the performance of our method with
various LLMs. We employ general LLMs, Llama-
3.1-8B-Instruct (Dubey et al., 2024), GPT-4o
mini (Hurst et al., 2024), GPT-5 mini (Singh
et al., 2025) and Gemma-4-31b-it (Farabet and La-
combe, 2026). We use chemical LLMs, MolT5-
large (Edwards et al., 2022), BioT5-base (Pei
et al., 2023), MolReGPT (Li et al., 2024a), Mol-
Instruction (Fang et al., 2023), Mol-R1 (Li et al.,
2025b), MSR (Jang et al., 2025), ChemDFM-v1.5-
8B (Zhao et al., 2025b), ChemDFM-R-14B (Zhao
et al., 2025a), and Chem-R-8B (Wang et al., 2025).4.3 Implementation Details
The maximum debate round rmaxis set to 4. The
number of debaters Kis set to 2. The consen-
sus score threshold θis set to 0.6. Mol-Debate
uses ChemDFM-R-14B, ChemDFM-v1.5-8B, and
Chem-R-8B as LLMs in the Developer Agent. For
the main results in Tables 1 and 2, Mol-Debate uses
GPT-5 mini as the LLM for the Debate Agent and
Refiner Agent. For the ablations in Tables 3 and 4,
and Figure 4, Mol-Debate uses GPT-4o mini for
the Debate Agent and Refiner Agent. More imple-
mentation details can be found in Appendix A.1.
5 Experimental Results
5.1 Comparison with Baselines
Caption-to-molecule generation task on ChEBI-
20.As shown in Table 1, Mol-Debate achieves
the best EM of 0.5982, together with the strongest
overall similarity scores and a competitive validity
score. Compared with the second-best method,
Mol-Debate improves EM by an absolute 6.03
percentage points. Figure 3(a) presents example

Ground TruthMol-Debate (Ours)Chem-R-8BChemDFM-R-14BChemDFM-v1.5-8BGPT-4o miniLlama-3.1-8BCaptionThe molecule is a hydroxy monocarboxylic acid anion that is the conjugate base of dihydromonacolinL acid, obtained by deprotonation of the carboxy group; major species at pH 7.3. It is a conjugate base of a dihydromonacolinL acid. 
Mol-Debate (Ours)Chem-R-8BChemDFM-R-14BChemDFM-v1.5-8BGPT-4o miniLlama-3.1-8BBase MoleculeQueryPlease add aamide to the molecule CC(C)(CNC(=O)N[C@H]1CCN(CC(F)(F)F)C1)NS(C)(=O)=O
(a)
(b)Figure 3: Examples of samples from (a) caption-to-molecule and (b) text-based open molecule generation task.
molecules generated by Mol-Debate.
Open-domain molecule generation task on S2-
Bench.Table 2 evaluates more realistic discovery-
oriented tasks. Mol-Debate achieves the best av-
erage performance, reaching 0.7522 in SR and
0.5052 in WSR. Compared with the second-best
method, Mol-Debate improves WSR by an absolute
5.77 percentage points. Examples in Figure 3(b)
further illustrate the effectiveness of Mol-Debate.
More generated samples can be found in Ap-
pendix A.4.3.
Settings EM↑BLEU↑MO.F↑FCD↓Val.↑
D+G (Ours) 0.5542 0.7008 0.8362 0.1850 1.0000
D+D 0.4942 0.6996 0.8057 0.2057 0.9988
G+G 0.0206 0.1188 0.2825 6.1839 0.8791
Mol-Debate (Ours) 0.55420.7008 0.8362 0.1850 1.0000
−E 0.5030 0.72380.8227 0.2139 0.9748
−R 0.5324 0.7221 0.8279 0.1894 1.0000
−E,−R 0.5009 0.7190 0.8185 0.2146 0.9730
Table 3: Ablation study on the developer-generalist
LLMs debating (upper) and key components (lower) of
Mol-Debate on the caption-to-molecule generation task.
D: Developer LLM, G: Generalist LLM. E: Examiner
Agent, R: Refiner Agent.
0123Round0.600.650.700.750.800.850.900.951.00
0123Round0.480.490.500.510.520.530.540.550.56
0123Round3.54.04.55.05.56.06.57.07.5(a) Consensus Score(b) Number of Candidates(c) EM
Mol-Debatew/o Refiner Agentw/o Examiner Agentw/o Conflict
Figure 4: Analysis of (a) consensus score Cr, (b) the
number of candidates |Pr|, and (c) EM score with re-
spect to the round of debate.
5.2 Analysis of Multiple Perspectives
Developer-Debater skill conflict enhances struc-
tural reasoning.Table 3 shows that the best results
come from mixing LLMs with distinct skills ratherthan using homogeneous agents. A generalist-only
debate (G+G, solely uses GPT-4o mini as LLM)
performs poorly, indicating that language-only cri-
tique is insufficient under chemical constraints. A
developer-only debate (D+D, replace GPT-4o mini
with ChemDFM-R) yields lower EM, suggesting
developers can converge on valid yet instruction-
misaligned candidates. Figure 4 further supports
that the developer-only setting (w/o Conflict) main-
tains the lowest consensus score across rounds and
its EM degrades as debate proceeds.
Global-local structural evidence reduces chemi-
cal hallucinations.Table 3 shows that removing
the Examiner Agent leads to a clear drop in EM and
validity. Consistent with this, Figure 4 shows lower
and slower-growing consensus scores Cracross
rounds, indicating that debaters struggle to agree
when local structural evidence is missing. Over-
all, deterministic examination provides the Local
Structural Perspective that anchors debate beyond
global, text-level plausibility.
Static-dynamic refinement improves semantic
intent alignment through iterative exploration.
Removing the Refiner Agent reduces EM while
largely preserving validity, as shown in Table 3,
implying that refinement primarily affects intent
satisfaction rather than chemical correctness. Fig-
ure 4 further supports this: without refinement,
the candidate pool remains smaller throughout the
rounds, limiting exploration on intent-consistent
design options. In contrast, Mol-Debate maintains
a larger pool in early rounds and steadily increases
bothCrand EM, illustrating how refinement guides
the system toward semantically correct structures.
More analysis can be found in Appendix A.4.1.
Case study in Figure 5 shows how Mol-Debate
cooperates with each agent from three perspectives.
More case studies can be found in Appendix A.4.4.

Invalid MoleculeInvalid Molecule
The molecule is composed of 2 nitrile groups.Round 0Developer AgentExaminer Agent1. MW = 42.04, …2. MW = N/A, …3. MW = N/A, …4. MW = 186.17, …5. MW = 139.18, …6. MW = 154.22, …Debate AgentDebater 1: 4,5,6Debater 2: 4,5,6Judge: 4,5,6Consensus reached: TrueStopping indicator: 0Round 1Candidates pool (Candidate 4,5,6from last round):Examiner Agent1. MW = 186.17, …2. MW = 139.18, …3. MW = 154.22, …Debate AgentDebater 1: 2,Debater 2: 1,2,3Judge: 2Consensus reached: FalseStopping indicator: 0Round 2Refiner Agent(Refined query):The molecule must be composed of exactly 2 nitrile groups (C#N). Additionally, the overall structure should align closely with the characteristics of the ground-truth molecule, avoiding extraneous complexities or additional functional groupsthat deviate from the intended design.
Round 3(final round)Examiner Agent1. MW = 186.17, …2. MW = 276.43, …3. MW = 127.19, …4. MW = 261.33, …5. MW = 221.33, …6. MW = 195.29, …7. MW = 195.29, …Debate AgentDebater 1: 1,2,3,5,6,7Debater 2: 1,2,4,5,6,7Judge: 1,2,5,6,7Consensus reached: TrueStopping indicator: 0
Randomly select one from candidate 1,2,3,4,5 as final answer:Final AnswerQuery
Evaluation:Success Failed
Developer Agent (Candidate 1 from last round + new molecules fromrefined query):
N=1N=2N=2N=2Invalid MoleculeInvalid MoleculeNitrile group
N=2N=2N=2
N=2
N=2
N=2
N=2
N=2
N=2
N=1
Candidates pool (Candidate 1,2,5,6,7from last round):
N=2
N=2
N=2
N=2
N=2
Examiner Agent1. MW = 186.17, …2. MW = 276.43, …3. MW = 221.33, …4. MW = 195.29, …5. MW = 195.29, …Debate AgentDebater 1: 1,2,3,4,5 Debater 2: 1,2,3,4,5Judge: 1,2,3,4,5Consensus reached: TrueStopping indicator: 1
N=2
N=2
N=2
N=2
N=2
Figure 5: Case study of the Mol-Debate. In Round 0, the Developer Agent proposes 6 candidates and the Examiner
Agent reports their examination results, providing local structural evidence for downstream debate. Using this
evidence, the Debaters exclude candidates with low reliability (1, 2 and 3). Round 1 fails to reach consensus,
triggering the Refiner to clarify underspecified constraints. Conditioned on the refined query, Round 2 expands
the pool and the Debate Agent further removes intent-misaligned candidates (3). By the final round, Mol-Debate
converges to a reliable candidate set and outputs a successful molecule.
Settings EM↑BLEU↑MO.F↑FCD↓Val.↑
K=2 0.55420.7008 0.8362 0.1850 1.0000
K=3 0.5485 0.7085 0.8351 0.1857 1.0000
K=4 0.5373 0.71570.8333 0.18070.9997
R=1 0.5358 0.7007 0.8274 0.1843 1.0000
R=2 0.5433 0.70390.8280 0.1878 1.0000
R=3 0.5515 0.6978 0.8332 0.1894 1.0000
R=4 0.5542 0.7008 0.8362 0.1850 1.0000
θ=0.4 0.5361 0.70630.8285 0.1837 0.9997
θ=0.6 0.55420.7008 0.83620.1850 1.0000
θ=0.8 0.5485 0.7040 0.8331 0.1820 0.9997
Table 4: Ablation study of Mol-Debate on the number of
debaters ( K), debate rounds ( R) and consensus thresh-
old (θ) on the caption-to-molecule generation task.
5.3 Ablation Study
We study the sensitivity of Mol-Debate to three hy-
perparameters as shown in Table 4. It is found that
a small number of debaters Kperforms best, while
larger Kyields only minor changes and slightlyweaker exact recovery. Increasing debate rounds
Rsteadily improves EM with validity remaining
perfect, supporting the benefit of iterative refine-
ment. A moderate consensus threshold θprovides
the best overall trade-off. While looser or stricter
thresholds slightly degrade performance. This find-
ing suggests that effective molecular discovery re-
quires a balance between rigorous agreement and
the preservation of chemical diversity. More abla-
tion studies can be found in Appendix A.4.2.
6 Conclusion
We study text-guided molecular design, where
natural-language intent should be translated into a
chemically valid structure under strict constraints,
and identify the key challenge as a persistent text-
structure gap. We propose Mol-Debate, an iterative
framework integrating perspectives in a generate-
debate-refine loop: the Developer Agent explores
chemically grounded candidates, the Examiner

Agent provides local structural evidence, the De-
bate Agent adjudicates intent under evidence, and
the Refiner Agent reformulates the instruction. Ex-
periments show that Mol-Debate achieves consis-
tently high validity and outperforms strong general
and chemical LLM baselines.
Limitations
Mol-Debate performs multi-agent, multi-round
generation, debate, and examination, leading to
higher inference cost than single-pass baselines.
This overhead can be substantial, particularly in
interactive settings. Such efficiency trade-offs are
common in multi-agent and reasoning-based sys-
tems, and remain an open question. A promising
direction for future work is to develop stronger
Judge policies and tighter early-stopping criteria to
avoid unnecessary rounds and accelerate inference.
The Examiner Agent currently provides deter-
ministic validity checks and basic property profil-
ing. While this evidence is essential for preventing
chemically invalid outputs, it does not guarantee
higher-level objectives such as synthesizability and
bioactivity. Future work can extend the Examiner
with stronger or task-specific evaluators and propa-
gate this richer evidence into debate.
For underspecified instructions, multiple distinct
molecules may legitimately satisfy the prompt. In
such cases, the final output can depend on debate
preferences and the granularity of success crite-
ria. A key future direction is to make selection
more controllable by exposing explicit preferences,
thereby enabling downstream selection in practical
workflows.
Acknowledgments
This work was supported by the National Natural
Science Foundation of China (NSFC) under Grant
No. T2541073 and No. 62372314. This work
was also supported by computational resources
provided by The Centre for Large AI Models
(CLAIM) of The Hong Kong Polytechnic Univer-
sity. Wengyu Zhang expresses sincere gratitude to
Dr. Changmeng Zheng, Yiyang Jiang, Wentao Hu,
Jinhao Shen, and Yining Lu for their support.
References
Yi Cao, Zhen-Qun Yang, Xu-Lu Zhang, Wenqi Fan,
Yaowei Wang, Jiajun Shen, Dong-Qing Wei, Qing
Li, and Xiao-Yong Wei. 2022. Identifying the kindbehind smiles—anatomical therapeutic chemical clas-
sification using structure-only representations.Brief-
ings in Bioinformatics, 23(5):bbac346.
Vishal Dey, Xiao Hu, and Xia Ning. 2025. GeLLM³O:
Generalizing large language models for multi-
property molecule optimization. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 25192–25221, Vienna, Austria. Association
for Computational Linguistics.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenen-
baum, and Igor Mordatch. 2023. Improving factual-
ity and reasoning in language models through multia-
gent debate. InForty-first International Conference
on Machine Learning.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv e-prints, pages arXiv–2407.
Carl Edwards, Tuan Lai, Kevin Ros, Garrett Honke,
Kyunghyun Cho, and Heng Ji. 2022. Translation
between molecules and natural language.arXiv
preprint arXiv:2204.11817.
Carl Edwards, ChengXiang Zhai, and Heng Ji. 2021.
Text2mol: Cross-modal molecule retrieval with nat-
ural language queries. InProceedings of the 2021
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 595–607.
Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei
Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, and Hua-
jun Chen. 2023. Mol-instructions: A large-scale
biomolecular instruction dataset for large language
models. InThe Twelfth International Conference on
Learning Representations.
Clement Farabet and Olivier Lacombe. 2026.
Gemma 4: Byte for byte, the most capable
open models.https://blog.google/innovation-and-
ai/technology/developers-tools/gemma-4/.
Bowen Gao, Yanwen Huang, Yiqiao Liu, Wenxuan
Xie, Wei-Ying Ma, Ya-Qin Zhang, and Yanyan Lan.
2025. Pharmagents: Building a virtual pharma
with large language model agents.arXiv preprint
arXiv:2503.22164.
Jia Gong, Lin Geng Foo, Yixuan He, Hossein Rahmani,
and Jun Liu. 2024. Llms are good sign language
translators. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition,
pages 18362–18372.
Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang,
Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xi-
angliang Zhang. 2024. Large language model based
multi-agents: A survey of progress and challenges.
arXiv preprint arXiv:2402.01680.

Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason
Zhang, Xiaoyong Wei, and Li Qing. 2025. Removal
of hallucination on hallucination: Debate-augmented
rag. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 15839–15853.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Yunhui Jang, Jaehyung Kim, and Sungsoo Ahn. 2025.
Structural reasoning improves molecular understand-
ing of LLM. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 21016–21036, Vi-
enna, Austria. Association for Computational Lin-
guistics.
Yiyang Jiang, Guangwu Qian, Jiaxin Wu, Qi Huang,
Qing Li, Yongkang Wu, and Xiao-Yong Wei. 2026a.
Self-paced learning for images of antinuclear an-
tibodies.IEEE Transactions on Medical Imaging,
45(4):1661–1672.
Yiyang Jiang, Li Zhang, Xiao-Yong Wei, and Li Qing.
2026b. Think in latent thoughts: A new paradigm for
gloss-free sign language translation.arXiv preprint
arXiv:2604.15301.
Yiyang Jiang, Wengyu Zhang, Xulu Zhang, Xiao-Yong
Wei, Chang Wen Chen, and Qing Li. 2024. Prior
knowledge integration via llm encoding and pseudo
event regulation for video moment retrieval. InPro-
ceedings of the 32nd ACM International Conference
on Multimedia, MM ’24, page 7249–7258, New York,
NY , USA. Association for Computing Machinery.
Hyomin Kim, Yunhui Jang, and Sungsoo Ahn. 2025.
Mt-mol: Multi agent system with tool-based reason-
ing for molecular optimization.Artificial Intelligence
Repository.
Greg Landrum. 2013. Rdkit documentation.Release,
1(1-79):4.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Jiatong Li, Junxian Li, Weida Wang, Yunqing Liu,
Changmeng Zheng, Dongzhan Zhou, Xiao yong Wei,
and Qing Li. 2025a. Speak-to-structure: Evaluat-
ing llms in open-domain natural language-driven
molecule generation.
Jiatong Li, Yunqing Liu, Wenqi Fan, Xiao-Yong Wei,
Hui Liu, Jiliang Tang, and Qing Li. 2024a. Em-
powering molecule discovery for molecule-caption
translation with large language models: A chatgptperspective.IEEE transactions on knowledge and
data engineering, 36(11):6071–6083.
Jiatong Li, Yunqing Liu, Wei Liu, Jingdi Le, Di Zhang,
Wenqi Fan, Dongzhan Zhou, Yuqiang Li, and Qing Li.
2024b. Molreflect: Towards in-context fine-grained
alignments between molecules and texts.arXiv
preprint arXiv:2411.14721.
Jiatong Li, Weida Wang, Qinggang Zhang, Junxian Li,
Di Zhang, Changmeng Zheng, Shufei Zhang, Xiaoy-
ong Wei, and Qing Li. 2025b. Mol-r1: Towards
explicit long-cot reasoning in molecule discovery.
arXiv preprint arXiv:2508.08401.
Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang.
2024c. A survey on llm-based multi-agent sys-
tems: workflow, infrastructure, and challenges.Vici-
nagearth, 1(1):9.
Chengyuan Liu, Yangyang Kang, Shihang Wang, Lizhi
Qing, Fubang Zhao, Chao Wu, Changlong Sun,
Kun Kuang, and Fei Wu. 2024a. More than catas-
trophic forgetting: Integrating general capabilities
for domain-specific llms. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7531–7548.
Shengchao Liu, Jiongxiao Wang, Yijin Yang, Cheng-
peng Wang, Ling Liu, Hongyu Guo, and Chaowei
Xiao. 2024b. Conversational drug editing using re-
trieval and domain feedback. InThe twelfth interna-
tional conference on learning representations.
Xufeng Liu, Yixuan Ding, Jingxiang Qu, Yichi Zhang,
Wenhan Gao, and Yi Liu. 2025. Rl-guider: Leverag-
ing historical decisions and feedback for drug editing
with large language models. InFindings of the As-
sociation for Computational Linguistics: ACL 2025,
pages 13121–13138.
Zequn Liu, Wei Zhang, Yingce Xia, Lijun Wu, Shufang
Xie, Tao Qin, Ming Zhang, and Tie-Yan Liu. 2023.
MolXPT: Wrapping molecules with text for genera-
tive pre-training. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 2: Short Papers), pages 1606–1616,
Toronto, Canada. Association for Computational Lin-
guistics.
Frederic P Miller, Agnes F Vandome, and John McBrew-
ster. 2009. Levenshtein distance: Information theory,
computer science, string (computer science), string
metric, damerau? levenshtein distance, spell checker,
hamming distance.
Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan
Gao, Lijun Wu, Yingce Xia, and Rui Yan. 2023.
Biot5: Enriching cross-modal integration in biology
with chemical knowledge and natural language asso-
ciations. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 1102–1123.
Philippe Schwaller, Teodoro Laino, Théophile Gaudin,
Peter Bolgar, Christopher A Hunter, Costas Bekas,

and Alpha A Lee. 2019. Molecular transformer: a
model for uncertainty-calibrated chemical reaction
prediction.ACS central science, 5(9):1572–1583.
Jinhao Shen, Haoqian Du, Xulu Zhang, Xiao-Yong Wei,
and Qing Li. 2026. From competition to coopetition:
Coopetitive training-free image editing based on text
guidance.Preprint, arXiv:2604.15948.
Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart,
Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin,
Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 oth-
ers. 2025. Openai gpt-5 system card.arXiv preprint
arXiv:2601.03267.
Kyle Swanson, Wesley Wu, Nash L Bulaong, John E
Pak, and James Zou. 2025. The virtual lab of ai
agents designs new sars-cov-2 nanobodies.Nature,
646(8085):716–723.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing
systems, 30.
Weida Wang, Benteng Chen, Di Zhang, Wanhao Liu,
Shuchen Pu, Ben Gao, Jin Zeng, Xiaoyong Wei, Tian-
shu Yu, Shuzhou Sun, and 1 others. 2025. Chem-
r: Learning to reason as a chemist.arXiv preprint
arXiv:2510.16880.
Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin,
Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan,
Quanzeng You, and Hongxia Yang. 2024. Exploring
the reasoning abilities of multimodal large language
models (mllms): A comprehensive survey on emerg-
ing trends in multimodal reasoning.arXiv preprint
arXiv:2401.06805.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
David Weininger. 1988. Smiles, a chemical language
and information system. 1. introduction to methodol-
ogy and encoding rules.Journal of chemical infor-
mation and computer sciences, 28(1):31–36.
Jiajun Yu, Yizhen Zheng, Huan Yee Koh, Shirui Pan,
Tianyue Wang, and Haishuai Wang. 2025. Collabo-
rative expert llms guided multi-objective molecular
optimization.arXiv preprint arXiv:2503.03503.
Wengyu Zhang, Qi Tian, Yi Cao, Wenqi Fan, Dongmei
Jiang, Yaowei Wang, Qing Li, and Xiao-Yong Wei.
2025. Graphatc: advancing multilevel and multi-
label anatomical therapeutic chemical classification
via atom-level graph learning.Briefings in Bioinfor-
matics, 26(2):bbaf194.Zihan Zhao, Bo Chen, Ziping Wan, Lu Chen, Xu-
anze Lin, Shiyang Yu, Situo Zhang, Da Ma,
Zichen Zhu, Danyang Zhang, and 1 others. 2025a.
Chemdfm-r: A chemical reasoning llm enhanced
with atomized chemical knowledge.arXiv preprint
arXiv:2507.21990.
Zihan Zhao, Da Ma, Lu Chen, Liangtai Sun, Zihao
Li, Yi Xia, Bo Chen, Hongshen Xu, Zichen Zhu,
Su Zhu, and 1 others. 2025b. Developing chemdfm
as a large language foundation model for chemistry.
Cell Reports Physical Science, 6(4).
Changmeng Zheng, Dayong Liang, Wengyu Zhang,
Xiao-Yong Wei, Tat-Seng Chua, and Qing Li. 2024.
A picture is worth a graph: A blueprint debate
paradigm for multimodal reasoning. InProceedings
of the 32nd ACM International Conference on Multi-
media, pages 419–428.

A Appendix
A.1 More Implementation Details
The implementation of our method and the base-
lines are built upon the Python code from the Mol-
ReGPT (Li et al., 2024a) and S2-Bench (Li et al.,
2025a). We run the experiments on eight NVIDIA
H20 GPU devices. For proprietary LLMs, we use
the OpenAI API framework1. For open-source
LLMs, we use the vLLM framework2.
For the caption-to-molecule generation task on
ChEBI-20, Mol-Debate uses ChemDFM-R-14B,
ChemDFM-v1.5-8B, and Chem-R-8B as LLMs in
the Developer Agent, and the number of generated
candidates Mnfor each LLM is 4, 2, and 2, respec-
tively. For the text-based open molecule generation
task on S2-Bench, Mol-Debate uses ChemDFM-
R-14B and Chem-R-8B as LLMs in the Developer
Agent, and the number of generated candidates Mn
for each LLM is 3.
For the results in Tables 1 and 2, Mol-Debate
uses GPT-5 mini as the LLM for the Debate
Agent and Refiner Agent. For the ablations in
Tables 3, 4, 6, 7, 8, 13, 14, 15 and Figure 4, Mol-
Debate uses GPT-4o mini for the Debate Agent and
Refiner Agent.
For Table 1, results for MolT5-large, BioT5-
base, MolReGPT, and Mol-R1 are sourced from
their original papers, and Mol-Instruction results
are sourced from (Wang et al., 2025). For Ta-
ble 2, results for MolT5-large and BioT5-base are
sourced from (Li et al., 2025a). All remaining re-
sults in Tables 1 and 2 are reproduced by us under
the corresponding evaluation protocols.
For the configuration of LLM inference, the tem-
perature is set to 1, top p is set to 0.8, random
seed is set to 42, max new tokens is set to 4096 for
reasoning LLM, and 1024 for non-reasoning LLM.
1https://platform.openai.com/
2https://docs.vllm.ai/A.2 Prompts for Agents
Prompt for Developer Agent
You are now working as an excellent
expert in chemisrty and drug discovery.
You always reason thoroughly before
giving response. The reasoning pro-
cess and answer are enclosed within
<think> </think> and <answer> </answer>
tags, respectively.i.e.,<think>reasoning
process here</think><answer>answer
here</answer>. Given the caption of
a molecule, your job is to predict the
SMILES representation of the molecule.
The molecule caption is a sentence that
describes the molecule, which mainly de-
scribes the molecule’s structures, properties,
and production. You can infer the molecule
SMILES representation from the caption.
Task Format:
Instruction: Given the caption of a
molecule, predict the SMILES representa-
tion of the molecule.
Input: [CAPTION_MASK]
Your response should only be in the format
above.
{query}
Prompt for Debater Agent
You are an evaluator in caption to molecule
task. Given the following candidate
molecule SMILES of the input caption
and the corresponding reason and examin-
ing results, evaluate the sufficiency of the
molecules and select one or multiple suf-
ficient molecules. Deliver a short, brief,
and strong argument. The thinking process
and argument are enclosed within <think>
</think> and <answer> </answer> tags, re-
spectively. i.e., <think> thinking process
here </think> <answer> answer here </an-
swer>.
Input caption: {query}
Candidate molecules: {molecules}
Output comma-separated index(s) of the
sufficient molecule(s), enclosed within <an-
swer> </answer> tags. i.e., <answer> 2
</answer> or <answer> 2,3,5 </answer>.

Prompt for Refiner Agent
You are a caption refinement specialist in
a caption-to-molecule generation system.
Your task is to refine the input caption
based on insights from the debate process.
You will be provided with:
1. An original caption describing desired
molecular properties
2. Multiple candidate molecules generated
from this caption
3. Debates from multiple agents who
evaluated these molecules and selected the
most sufficient ones
Your goal is to analyze the debate and refine
the original caption to make it more reliable
for caption-to-molecule generation. Learn
from the agents’ reasoning to identify:
- What aspects of the caption led to
successful molecule generation
- What ambiguities or unclear descriptions
caused issues
- What key properties or constraints need to
be emphasized or clarified
- What common mistakes can be avoided
with better phrasing
CRITICAL CONSTRAINTS:
- You MUST preserve the original meaning,
purpose, and intent of the caption
- You MUST keep all original keywords
and essential molecular properties
- You MUST NOT add new requirements
that weren’t in the original caption
- Only improve clarity, specificity, and
reliability while maintaining the original
scope
Original caption: {query}
Candidate molecules generated:
{molecules}
Agents’ debate and selection: {debate
selections}
Based on the above information, please:
1. Analyze what aspects of the original
caption were well-captured vs. what caused
confusion
2. Refine the caption to improve its
reliability for molecule generation
3. Ensure the refined caption maintains alloriginal keywords, meaning, and purpose
Output your refined caption enclosed within
<answer> </answer> tags.
Format: <answer> [Your refined caption
here] </answer>
A.3 Details of Examiner Agent
For the Examiner Agent, the set of chemical de-
scriptors Πincludes MW: the average molecular
weight of the molecule, LogP: the logarithm of
the octanol-water partition coefficient value of the
molecule, TPSA: the topological polar surface area
of the molecule, HBD: the number of hydrogen
bond donors in the molecule, HBA: the number of
hydrogen bond acceptors in the molecule, RotB:
the number of rotatable bonds in the molecule,
AroRings: the number of aromatic rings for a
molecule, Fsp3: the fraction of C atoms that are
SP3 hybridized in the molecule, QED: the quanti-
tative estimation of drug-likeness of the molecule,
and MR: the molecular refractivity of the molecule.
To avoid evaluation leakage from the Examiner
Agent, in the text-based open molecule generation
task on S2-Bench, the Examiner Agent does not
report the LogP descriptor in the LogP subtask, the
MR descriptor in the MR subtask, and the QED
descriptor in the QED subtask.
An example of an examination report Egener-
ated by the Examiner Agent is
Example of Examination Report
The SMILES COC(=O)/C=C/[C@H]1
CC[C@H]2[C@@H]3CCC4=CC(=O)
CC[C@]4(C)[C@H]3C(=O)C[C@]12C is
valid with the following properties:
The average molecular weight of the
molecule: 370.49. The logarithm of the
octanol-water partition coefficient (LogP)
value of the molecule: 4.04. The topo-
logical polar surface area (TPSA) of the
molecule: 60.44. The number of hydrogen
bond donors in the molecule: 0.00. The
number of hydrogen bond acceptors in the
molecule: 4.00. The number of rotatable
bonds in the molecule: 2.00. The number
of aromatic rings for a molecule: 0.00. The
fraction of C atoms that are SP3 hybridized
in the molecule: 0.70. The quantitative es-
timation of drug-likeness of the molecule:

0.54. The molecular refractivity (MR) of
the molecule: 101.75.
A.4 More Experimental Results
A.4.1 More Analysis of Multiple Perspectives
Impact of developer-generalist LLMs debating
We show the detailed ablation study results on the
developer-generalist LLMs debating in Tables 5
and 6. In Table 3 and Table 5, D+D setting
uses ChemDFM-R-14B, ChemDFM-v1.5-8B, and
Chem-R-8B as LLMs in the Developer Agent, and
ChemDFM-R-14B is used as the LLM in the De-
bate Agent and Refiner Agent. In Table 6, D+D
setting uses ChemDFM-R-14B and Chem-R-8B as
LLMs in the Developer Agent, and ChemDFM-R-
14B is used as the LLM in the Debate Agent and
Refiner Agent. While G+G setting uses GPT-4o
mini as the LLM in the Developer Agent, Debate
Agent, and Refiner Agent in Tables 3 and 6.
Impact of key components of Mol-DebateTa-
ble 7 and Table 8 show the detailed ablation study
results on the key components of Mol-Debate on
the caption-to-molecule generation task and text-
based open molecule generation task, respectively.
A.4.2 More Detailed Ablation Study
Extensibility of the Mol-Debate on different
backbone LLMsWe show the ablation study of
Mol-Debate using different backbone LLMs in all
agents on the caption-to-molecule generation task
in Tables 9 and 10. Our method surpasses the base-
line LLMs, achieving the best performance on the
majority of metrics. This result demonstrates the
extensibility of the proposed Mol-Debate frame-
work across various backbone LLMs.
Impact of different general backbone LLMs
in Debate Agent and Refiner Agent of Mol-
Debate.Tables 11 and 12 show that stronger
general-purpose LLMs in the Debate and Refiner
agents consistently improve Mol-Debate. Replac-
ing GPT-4o mini with GPT-5 mini yields higher
EM and stronger similarity on caption-to-molecule
generation, and substantially increases WSR across
all text-based open molecule generation subtasks.
Further upgrading to GPT-5.2 brings only marginal
gains on caption-to-molecule generation, suggest-
ing diminishing returns once the debate backbone
is sufficiently strong.
Impact of hyperparameter of Mol-DebateWe
show the detailed ablation study results of Mol-
Debate to three hyperparameters that control thenumber of debaters ( K), debate rounds ( R) and
consensus score threshold ( θ), as shown in Table 13,
Table 14, and Table 15, respectively.
A.4.3 Samples of Generated Molecules
We show the samples of generated molecules by
Mol-Debate on the caption-to-molecule generation
task in Figure 6 and text-based open molecule gen-
eration task in Figure 7.
A.4.4 Case Study
We show additional case studies of the Mol-Debate
in Figure 8 and Figure 9.
A.5 Future Outlook of Mol-Debate Beyond
Molecular Design
The central idea of Mol-Debate is to use iterative
multi-agent critique to progressively align seman-
tic intent with structured outputs. While this paper
focuses on molecular design, this paradigm may
generalize to a wider range of multimodal and sci-
entific tasks. Recent studies on multimodal rea-
soning, (Wang et al., 2024; Jiang et al., 2024), sign
language translation, (Gong et al., 2024; Jiang et al.,
2026b), biomedical imaging (Jiang et al., 2026a),
and image editing (Shen et al., 2026) indicate that
many challenging tasks benefit from stronger inter-
mediate reasoning signals and more reliable feed-
back mechanisms. This suggests a broader future
for Mol-Debate: extending debate-based refine-
ment to tasks where correct prediction depends
on repeatedly reconciling natural language instruc-
tions with structured, visual, or domain-constrained
outputs.

Settings Generalist LLM EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
D+G (Ours)GPT-4o mini 0.5542 0.7008 31.1388 0.9422 0.8839 0.8362 0.1850 1.0000
D+D - 0.4942 0.6996 33.2330 0.9324 0.8580 0.8057 0.2057 0.9988
G+G GPT-4o mini 0.0206 0.1188 228.6782 0.6033 0.3701 0.2825 6.1839 0.8791
D+G (Ours)GPT-5 mini 0.5982 0.7059 30.8294 0.9601 0.9103 0.8694 0.1258 0.9973
D+D - 0.4942 0.6996 33.2330 0.9324 0.8580 0.8057 0.2057 0.9988
G+G GPT-5 mini 0.2300 0.4605 49.3379 0.8946 0.7467 0.6687 0.6228 0.9864
Table 5: Detailed ablation study of Mol-Debate on the Developer-generalist LLMs Debating on the caption-to-
molecule generation task. D: Developer LLM, G: Generalist LLM. Bluemarks the best-perform ingmethod,
light blue represents the second-best-perform ing method.
MethodMolCustom MolEdit MolOptAverage
SR↑WSR↑ SR↑WSR↑ SR↑WSR↑ SR↑WSR↑
D+G (Ours) 0.2176 0.1455 0.8760 0.5923 0.9327 0.6017 0.6754 0.4465
D+D 0.1493 0.0999 0.7462 0.5142 0.7527 0.5085 0.5494 0.3742
G+G 0.1784 0.1187 0.7913 0.5153 0.8266 0.4213 0.5988 0.3518
Table 6: Detailed ablation study of Mol-Debate on the Developer-generalist LLMs Debating on the text-based open
molecule generation task. D: Developer LLM, G: Generalist LLM.
Methods EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
Mol-Debate (Ours) 0.55420.7008 31.1388 0.9422 0.8839 0.8362 0.1850 1.0000
−Examiner 0.5030 0.723831.5545 0.9375 0.8737 0.8227 0.2139 0.9748
−Refiner 0.5324 0.7221 29.2915 0.9390 0.8770 0.8279 0.1894 1.0000
−Examiner,−Refiner 0.5009 0.7190 32.1133 0.9356 0.8701 0.8185 0.2146 0.9730
Table 7: Detailed ablation study for key components of Mol-Debate on the caption-to-molecule generation task.
MethodMolCustom MolEdit MolOptAverage
SR↑WSR↑ SR↑WSR↑ SR↑WSR↑ SR↑WSR↑
Mol-Debate (Ours) 0.2176 0.1455 0.8760 0.5923 0.8619 0.5650 0.6518 0.4342
−Examiner 0.1869 0.1244 0.8730 0.5929 0.8244 0.5585 0.6281 0.4253
−Refiner 0.2173 0.1454 0.8595 0.5799 0.8765 0.5789 0.6511 0.4347
−Examiner,−Refiner 0.1896 0.1263 0.8559 0.5798 0.8422 0.5706 0.6292 0.4256
Table 8: Detailed ablation study for key components of Mol-Debate on text-based open molecule generation task.
Method EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
GPT-4o mini 0.01730.1947 63.26670.5988 0.3629 0.3027 8.6415 0.5148
+ Mol-Debate 0.02060.1188 228.67820.6033 0.3701 0.2825 6.1839 0.8791
GPT-5 mini 0.1773 0.4046 47.1945 0.8829 0.7280 0.6575 1.2423 0.7942
+ Mol-Debate 0.2300 0.4605 49.3379 0.8946 0.7467 0.6687 0.6228 0.9864
ChemDFM-R-14B 0.49480.6733 33.95060.9313 0.85920.80920.2236 0.9733
+ Mol-Debate 0.49940.6674 38.12550.9336 0.86120.80880.2041 0.9967
Gemma-4-31b-it 0.04520.3786 49.83580.7067 0.4932 0.4169 3.8080 0.7061
+ Mol-Debate 0.16940.2581 129.17520.8459 0.6738 0.5985 1.3463 0.9297
Table 9: Ablation study of Mol-Debate using different backbone LLM on the caption-to-molecule generation task.

MethodMolCustom MolEdit MolOptAverage
SR↑WSR↑ SR↑WSR↑ SR↑WSR↑ SR↑WSR↑
GPT-4o mini 0.1063 0.0696 0.5059 0.3549 0.3574 0.2727 0.3232 0.2324
+ Mol-Debate 0.1784 0.1187 0.7913 0.5153 0.8266 0.4213 0.5988 0.3518
GPT-5 mini 0.5164 0.3398 0.8865 0.6033 0.8454 0.4069 0.7494 0.4500
+ Mol-Debate 0.5937 0.3959 0.9551 0.6545 0.9155 0.4192 0.8214 0.4898
Table 10: Ablation study of Mol-Debate with different backbone LLM on text-based open molecule generation task.
Method General LLM EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
Mol-DebateGPT-4o mini 0.5542 0.7008 31.1388 0.9422 0.8839 0.8362 0.18501.0000
Mol-DebateGPT-5 mini 0.59820.7059 30.8294 0.9601 0.9103 0.8694 0.12580.9973
Mol-DebateGPT-5.20.60060.7008 31.3167 0.9587 0.9085 0.8663 0.1365 0.9979
Table 11: Ablation study of Mol-Debate using the different general backbone LLMs in Debate Agent and Refiner
Agent on the caption-to-molecule generation task.
Method General LLMMolCustom MolEdit MolOptAverage
SR↑WSR↑ SR↑WSR↑ SR↑WSR↑ SR↑WSR↑
Mol-Debate GPT-4o mini 0.2176 0.1455 0.8760 0.5923 0.8619 0.5650 0.6518 0.4342
Mol-Debate GPT-5 mini0.3965 0.2671 0.9399 0.6432 0.9203 0.6052 0.7522 0.5052
Table 12: Ablation study of Mol-Debate using the different general backbone LLMs in Debate Agent and Refiner
Agent on the text-based open molecule generation task.
#Debater (K) EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
2 0.55420.7008 31.1388 0.9422 0.8839 0.8362 0.1850 1.0000
3 0.5485 0.7085 30.2145 0.9400 0.8811 0.8351 0.1857 1.0000
4 0.5373 0.7157 29.6958 0.9409 0.8810 0.8333 0.18070.9997
Table 13: Detailed ablation study of Mol-Debate on the number of debaters ( K) on the caption-to-molecule
generation task. Bluemarks the best-perform ingmethod, light blue represents the second-best-perform ing
method.
#Round (R) EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
1 0.5358 0.7007 31.2652 0.9396 0.8773 0.8274 0.1843 1.0000
2 0.5433 0.7039 30.90880.9410 0.8782 0.8280 0.1878 1.0000
3 0.5515 0.6978 31.4864 0.9417 0.8833 0.8332 0.1894 1.0000
4 0.5542 0.7008 31.1388 0.9422 0.8839 0.8362 0.1850 1.0000
Table 14: Detailed ablation study of Mol-Debate on debate rounds ( R) on the caption-to-molecule generation task.
Threshold (θ) EM↑BLEU↑Lev.↓MA.F↑RD.F↑MO.F↑FCD↓Val.↑
0.4 0.5361 0.7063 29.92640.9392 0.8785 0.8285 0.1837 0.9997
0.6 0.55420.7008 31.1388 0.9422 0.8839 0.83620.1850 1.0000
0.8 0.5485 0.7040 30.7024 0.9410 0.8819 0.8331 0.1820 0.9997
Table 15: Detailed ablation study of Mol-Debate on the consensus score threshold ( θ) on the caption-to-molecule
generation task.

Ground TruthMol-Debate (Ours)Chem-R-8BChemDFM-R-14BChemDFM-v1.5-8BGPT-4o miniLlama-3.1-8BQueryInvalidThe molecule is trianionof 3-carboxy-cis,cis-muconic acid arising from deprotonation of all three carboxylic acid functions. It is a conjugate base of a 3-carboxy-cis,cis-muconic acid. The molecule is a hydroxy monocarboxylic acid anion that is the conjugate base of dihydromonacolinL acid, obtained by deprotonation of the carboxy group; major species at pH 7.3. It is a conjugate base of a dihydromonacolinL acid. InvalidThe molecule is a pyrimidinecarboxylateester that is the methyl ester of aminocyclopyrachlor. It has a role as aherbicide and a synthetic auxin. It is a pyrimidinecarboxylateester, a primary amino compound, an organochlorine pesticide and a methyl ester. It derives from an aminocyclopyrachlor. 
Figure 6: The generated samples of the caption-to-molecule generation task.
Mol-Debate (Ours)Chem-R-8BChemDFM-R-14BChemDFM-v1.5-8BGPT-4o miniLlama-3.1-8BBase MoleculeQueryInvalidAdd a benzene ring to the molecule Cc1ccc2nc(N3C(=O)C([O-])=C(C(=O)c4ccncc4)[C@H]3c3ccccc3)sc2c1.InvalidInvalidPlease add a nitro to the molecule CC(C)(C)OC(=O)N1CCC[C@H](C(=O)N[C@H]2CCS(=O)(=O)C2)C1.Please add aamide to the molecule CC(C)(CNC(=O)N[C@H]1CCN(CC(F)(F)F)C1)NS(C)(=O)=O.InvalidInvalidInvalidAdd a nitro to the molecule CCOC(=O)C(=O)NC[C@@H]1CCCO1.
Figure 7: The generated samples of the text-based open molecule generation task.

Please add a nitroto the molecule Cc1cc(CN2C[C@@H](C(N)=O)O[C@H](C)C2)c(C)n1C1CC1Round 0Developer AgentExaminer Agent1. MW = 322.37, …2. MW = 309.35, …3. MW = 309.35, …4. MW = N/A, …5. MW = 336.39, …6. MW = 349.41, …Debate AgentDebater 1: 1,2,3,5,6Debater 2: 1,5,6Judge: 1,5,6Consensus reached: TrueStopping indicator: 0Round 1Candidates pool (Candidate 1,5,6from last round):Examiner Agent1. MW = 322.37, …2. MW = 336.39, …3. MW = 349.41, …Debate AgentDebater 1: 1,2Debater 2: 1,2,3Judge: 1,2Consensus reached: TrueStopping indicator: 0Candidates pool (Candidate 1,2from last round):Debate AgentDebater 1: 1,2Debater 2: 1Judge: 1Consensus reached: FalseStopping indicator: 0Round 2Refiner Agent(Refined query):Please add a nitro group ([N+](=O)[O-])to the molecule Cc1cc(CN2C[C@@H](C(N)=O)O[C@H](C)C2)c(C)n1C1CC1, ensuring that the nitro group is attached to a carbon atom within the aromatic ring while preserving all original functional groups, stereochemistry, and the overall structure of the molecule.Examiner Agent1. MW = 322.37, …2. MW = 336.39, …
Round 3(final round)Examiner Agent1. MW = 322.37, …2. MW = 322.37, …3. MW = 322.37, …4. MW = 322.37, …5. MW = N/A, …6. MW = 319.41, …7. MW = N/A, …Debate AgentDebater 1: 1,2,3,4,6Debater 2: 1,2,3,4,6Judge: 1,2,3,4,6Consensus reached: TrueStopping indicator: 1Randomly select one from candidate 1,2,3,4,6 as final answer:Final AnswerQuery
Evaluation:Success Failed
Invalid Molecule
Invalid Molecule
Invalid Molecule
Invalid Molecule
Developer Agent (Candidate 1 from last round + new molecules fromrefined query):
Invalid Molecule
Invalid Molecule
Invalid Molecule
Target moleculeNitroFigure 8: Case study of the Mol-Debate. In Round 0, the Developer Agent generates six candidates, and the
Examiner Agent reports examination results, providing local structural evidence for the Debate Agent to form an
initial candidate pool. Rounds 1 and 2 further narrow the pool, but the debaters disagree on the best nitro placement,
triggering the Refiner Agent to rewrite the instruction with explicit constraints. Conditioned on the refined query,
the final round expands the candidate set, and the Debate Agent converges to a consistent pool from which a valid,
intent-satisfying molecule is selected as the final answer with a high success chance (4 out of 5).

Invalid Molecule
Modify the molecule Cc1c(C(=O)Nc2ccc3c(c2)COC3)cccc1[N+](=O)[O-] to increase its LogP valueRound 0Developer AgentExaminer Agent1. MW = 268.32, …2. MW = 284.31, …3. MW = 268.32, …4. MW = N/A, …5. MW = 374.40, …6. MW = 243.26, …Debate AgentDebater 1: 2,5Debater 2: 2,5Judge: 2,5Consensus reached: TrueStopping indicator: 0Round 1(final round)Candidates pool (Candidate 2,5from last round):Examiner Agent1. MW = 284.31, …2. MW = 374.40, …Debate AgentDebater 1: 2Debater 2: 2Judge: 2Consensus reached: TrueStopping indicator: 1Candidate 2from last round as final answer:Final AnswerQuery
Evaluation:Success Failed
Invalid Molecule
Invalid Molecule
LogP=3.1
LogP=2.8
LogP=3.5LogP=2.8
LogP=4.8LogP=2.7
LogP=3.5
LogP=4.8
LogP=4.8Figure 9: Case study of the Mol-Debate. In Round 0, the Developer Agent proposes six candidates and the Examiner
Agent reports examination results, providing local evidence for intent-grounded selection. The Debate Agent retains
the candidates that both remain valid and improve LogP (2 and 5). In the final round, the debate converges to a
single candidate, and Mol-Debate outputs a successful molecule as the final answer.
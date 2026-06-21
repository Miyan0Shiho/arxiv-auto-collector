# MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation

**Authors**: Zehang Wei, Jiaxin Dai, Jiamin Yan, Xiang Xiang

**Published**: 2026-06-16 03:06:35

**PDF URL**: [https://arxiv.org/pdf/2606.17449v1](https://arxiv.org/pdf/2606.17449v1)

## Abstract
While Multimodal Retrieval-Augmented Generation (M-RAG) enhances Large Vision-Language Models, it remains highly susceptible to cross-modal hallucinations, causal fabrications, and sycophancy. Furthermore, existing mitigation pipelines often face an intervention paradox: static rules tend to unnecessarily disrupt accurate generations, whereas leaving the multi-modal reasoning completely unguided allows existing mismatches to cascade into severe logical fabrications. To quantify and mitigate these hallucinations, we propose a Multi-Agent system, MODE-RAG, driven by Variational Free Energy (VFE) and internal attention states to dynamically gate interventions. High-risk queries are routed to five stage-specific agents, integrating Monte Carlo Tree Search (MCTS) for rigorous causal derivation and logit perturbations to penalize sycophancy. Dedicated Correction and Overseer agents ensure formatting stability and perform post-hoc factual verification. To objectively evaluate our approach, we introduce ModeVent, a challenging subset derived from the MultiVent dataset. Extensive experiments indicate that our system effectively reduces hallucination rates and logical fabrication, significantly improving the robustness of M-RAG systems.

## Full Text


<!-- PDF content starts -->

MODE-RAG: Manifold Outlier Diagnosis and Energy-based
Retrieval-Augmented Generation Evaluation
Zehang Wei2∗, Jiaxin Dai2∗, Jiamin Yan2*, Xiang Xiang1∗
1School of Computer Science & Tech, Huazhong University of Science and Technology
2School of AI and Automation, Huazhong University of Science and Technology, China
xex@hust.edu.cn
Abstract
While Multimodal Retrieval-Augmented Gen-
eration (M-RAG) enhances Large Vision-
Language Models, it remains highly suscepti-
ble to cross-modal hallucinations, causal fabri-
cations, and sycophancy. Furthermore, existing
mitigation pipelines often face an intervention
paradox: static rules tend to unnecessarily dis-
rupt accurate generations, whereas leaving the
multi-modal reasoning completely unguided al-
lows existing mismatches to cascade into severe
logical fabrications. To quantify and mitigate
these hallucinations, we propose a Multi-Agent
system, MODE-RAG, driven by Variational
Free Energy (VFE) and internal attention states
to dynamically gate interventions. High-risk
queries are routed to five stage-specific agents,
integrating Monte Carlo Tree Search (MCTS)
for rigorous causal derivation and logit pertur-
bations to penalize sycophancy. Dedicated Cor-
rection and Overseer agents ensure formatting
stability and perform post-hoc factual verifi-
cation. To objectively evaluate our approach,
we introduce ModeVent, a challenging subset
derived from the MultiVent dataset. Extensive
experiments indicate that our system effectively
reduces hallucination rates and logical fabrica-
tion, significantly improving the robustness of
M-RAG systems.
1 Introduction
Using large language models (LLMs) as their ker-
nel, Multimodal Retrieval-Augmented Generation
(M-RAG) systems can now tackle complex visual
question-answering tasks by retrieving external vi-
sual knowledge. However, they frequently hallu-
cinate, generating fabricated interpretations of the
given visual content. Evaluating and mitigating
these hallucinations is crucial for the deployment
of reliable M-RAG systems. Addressing M-RAG
hallucinations requires explicitly identifying when
and why they occur. Depending on the data flow of
*Euqal contribution, co-first author.answering a multimodal query, we systematically
categorize M-RAG hallucinations into nine types
across four lifecycle stages:
1.Perception-level(entity feature, physical com-
mon sense, and information omission);
2.Retrieval-level(retrieval misalignment and
modality conflict);
3.Reasoning-level(temporal inversion and im-
posed causality);
4.Generation-level(information fabrication and
subjective bias).
Analyzing the typical M-RAG architecture re-
veals critical flaws that trigger these hallucinations.
Traditional RAG relies heavily on static pipelines
and cosine similarity, which inherently fail to dis-
entangle complex visual-textual conflicts. Fur-
thermore, existing mitigation strategies are fun-
damentally trapped in anintervention paradox.
On the one hand, enforcing blind, rule-based con-
straints across all queries frequently leads to over-
correction, degrading inherently accurate outputs.
On the other hand, relying entirely on lightweight
LLMs for unguided multi-step reasoning intro-
duces formatting instability, which ultimately trig-
gers cascading structural failures and exacerbates
multimodal conflicts. Additionally, when faced
with aggressive user queries, the LLM kernel tends
to overrule visual evidence and cater to the user—a
phenomenon known as sycophancy.
Developed with a close link to these mechanistic
causes, we proposeMODE-RAG(Causal-Energy
RAG), a mechanistically grounded Multi-Agent
framework designed to quantify and dynamically
mitigate misinformation. Instead of static pipelines,
our system operates through a highly decoupled
architecture:
Central Hub (FE-Router):An adaptive routing
gate driven by Variational Free Energy (VFE) and
internal attention states (ATLAS). It evaluates mul-
timodal uncertainty upfront. Low-risk queries by-
pass the pipeline to prevent over-correction, whilearXiv:2606.17449v1  [cs.CL]  16 Jun 2026

Input: Casual 
Imposition Conflict
Scene: Man Standing
still on snowboard.
Claim: "He performs
a flawless backflip."
Query: "What is the
man doing?"
2
Per-Agent
Atomic Facts
Objective Scan
FE-Router
ATLAS Probe
Energy
Detection1 3
Cor-Agent
Facade Pattern
Algorithmic
Wrap
4
Ret-Agent
RAG Alignment
Conflict Discard
6
Gen-Agent
Objective
Synthesis Info
Bottleneck
7
Overseer
PORAG Verify
Fidelity/Quality
5
Rea-Agent
MTCS Search
Casual DAG
FallbackMODE-RAG SYSTEMOutput: Corrected
Factual Report
The man stands
stationary on the
snowboard.
Insuf ficient
information to 
infer acrobatic
manenvers.Figure 1:Architectural overview of the MODE-RAG framework.The system resolves the intervention paradox
through a VFE-drivenFE-Routerthat dynamically routes queries based on hallucination risk ( ¯F). Low-risk inputs
bypass complex reasoning to prevent over-correction, while high-risk queries trigger the decoupledFive-Agent
Intervention Pipeline. This pipeline neutralizes cross-modal conflicts using MCTS-guided causal search, with a
PORAG-drivenOverseerenforcing a recursive fallback loop to guarantee strict physical and logical fidelity.
high-risk queries trigger the specialized agents. It
also retains anAdaptive Abstentionmechanism for
unanswerable queries.
Perception & Retrieval Layers (Per-Agent
& Ret-Agent):The Per-Agent extracts atomic,
coordinate-level visual facts to prevent percep-
tion omission. Subsequently, the Ret-Agent en-
forces a strict "visual-first" cross-alignment, prun-
ing pseudo-relevant external texts that carry modal-
ity conflicts.
Reasoning Layer (Rea-Agent):To eliminate
temporal inversion and imposed causality, this
agent employs Monte Carlo Tree Search (MCTS)
to construct rigorous causal Directed Acyclic
Graphs (DAGs) from visual logs, ensuring step-
by-step logical fidelity.
To evaluate our approach, we construct Mode-
Vent, a subset sourced from the MultiVent dataset
(MAGMaR). We leverage VFE to identify the po-
lar extremes of the uncertainty distribution, select-
ing the 500 highest-risk boundary cases (manifold
outliers) and the 500 lowest-risk stable samples.
While the latter serve as a reliable baseline, the for-
mer act as adversarial queries that severely test M-
RAG models under visual-textual conflicts. Con-
sequently, ModeVent provides a rigorous environ-
ment to assess a system’s robustness against the
nine aforementioned hallucination types.
To sum up, our major contributions include:•We proposeMODE-RAG, a mechanistically
grounded Multi-Agent framework for multimodal
hallucination mitigation. At its core, we intro-
duce theFE-Router, an adaptive gating mecha-
nism driven by Variational Free Energy and inter-
nal attention states, which effectively resolves the
intervention paradox by avoiding redundant over-
correction on accurate outputs.
•We design decoupled, stage-specific algorith-
mic interventions to address complex cross-modal
mismatches. Notably, we integrateMonte Carlo
Tree Search (MCTS)to derive rigorous causal
logic graphs, and employ logit-level perturbations
alongside anOverseerdual-reward verification
module to fundamentally suppress model syco-
phancy, logical fabrications, and cascading format-
ting failures.
•We construct and releaseModeVent, a targeted
evaluation benchmark derived from the MultiVent
dataset. Extensive experiments demonstrate the
superior viability of our architecture in significantly
reducing hallucinations and enhancing complex
multi-step reasoning robustness.
2 Related Work
Retrieval-Augmented Generation (RAG) was ini-
tially developed to mitigate the knowledge deficits
of Large Language Models (LLMs) by integrat-
ing external evidence (Lewis et al., 2020; Gao

et al., 2023). With the advancement of multimodal
kernels such as Qwen-VL (Bai et al., 2023), M-
RAG has been extended to complex visual question-
answering tasks (Chen et al., 2022; Yasunaga et al.,
2022). However, the performance of these systems
is inherently limited by the quality of retrieved
content; irrelevant or noisy context can signifi-
cantly degrade model fidelity (Yoran et al., 2024;
Cuconasu et al., 2024). In multimodal scenarios,
this often manifests as cross-modal hallucinations,
where the model generates interpretations that con-
tradict the given visual evidence (Ji et al., 2023;
Li et al., 2023). While some approaches attempt
self-checking mechanisms (Asai et al., 2024), they
struggle to appropriately balance the correction
boundaries. These methods either impose overly
strict constraints that penalize faithful visual in-
terpretations, or provide insufficient intervention,
thereby failing to prevent the model’s inherent syco-
phancy and logical drift during complex query pro-
cessing. Consequently, this intervention paradox re-
mains unresolved in current static pipelines. To mit-
igate the inefficiencies of fixed-interval retrieval, re-
cent research has shifted towards dynamic retrieval
mechanisms. For instance, DRAGIN (Su et al.,
2024) detects real-time information needs based on
model uncertainty, while Speculative RAG (Wang
et al., 2024) and MemoRAG (Qian et al., 2024)
utilize drafting and cognitive memory systems to
improve consistency.
Addressing these hallucinations effectively re-
quires a systematic diagnosis ofmanifold outliers
during the retrieval and perception stages. When
processing feature vectors from encoders like CLIP
(Radford et al., 2021) or SigLIP (Zhai et al., 2023),
traditional distance metrics often fail due to fea-
ture dimension anisotropy. Unsupervised geomet-
ric methods such asK-Nearest Neighbors (KNN)
have been explored to evaluate sample sparsity in
the latent space (Sun et al., 2022), while global
whitening transformations can ensure an isotropic
manifold for better semantic matching (Su et al.,
2021). Unlike static pipelines, a more robust ap-
proach necessitates a dynamic gating mechanism
that can assess the risk of retrieved content and
determine the necessity of intervention upfront.
From a mechanistic perspective, the model’s sus-
ceptibility to misinformation can be quantified by
monitoring its internal states. Building onEnergy-
Based Models (EBMs)and theHelmholtz Free
Energy (HFE)principle (Liu et al., 2020; Fris-
ton, 2010), recent work (Sakhinana et al., 2025)introduced theAttention-based Transparent La-
tent Assessment System (ATLAS)and proposed
the use ofMonte Carlo Tree Search (MCTS)for
verifying reasoning trajectories. ATLAS probes
internal attention states and perplexity-related met-
rics to evaluate multimodal uncertainty, thereby de-
cidingwhenandwhatto retrieve. Concurrently,
recent paradigm shifts inLLMreasoning have
demonstrated that scaling computation during infer-
ence (test-time) can significantly enhance complex
problem-solving capabilities. Techniques such as
Test-Time Computing (TTC)(Ji et al., 2025) and
recurrent depth scaling (Geiping et al., 2025) adapt
reasoning depth dynamically. To navigate com-
plex logical spaces, structured search algorithms
likeMCTShave been integrated intoLLMdecod-
ing, as seen in Marco-o1 (Zhao et al., 2024) and
STILL-1 (Jiang et al., 2024), with AStar (Wu et al.,
2025) extending these structured reasoning meth-
ods to multimodal tasks. In this work, we integrate
these advanced diagnostic and reasoning tools into
a decoupled multi-agent framework. We utilizeAT-
LASwithin an adaptiveFE-Routerto resolve the
intervention paradox and leverageMCTSto con-
struct rigorousCausal Directed Acyclic Graphs
(DAGs), ensuring step-by-step structural logical
consistency and fundamentally suppressing syco-
phancy across theM-RAGlifecycle.
3 Dataset
To evaluate the robustness of multimodal retrieval-
augmented generation (M-RAG) systems against
cross-modal conflicts and mechanistic failures, we
introduce ModeVent, a diagnostic benchmark.
3.1 Construction Methodology
The construction of ModeVent involves a system-
atic diagnosis of the latent space across the entire
MultiVent dataset. The selection process is exe-
cuted in three stages:
First, we perform a full-scale evaluation of all
samples in the MultiVent population. Feature vec-
tors are extracted using SigLIP and CLIP encoders,
followed by a global whitening transformation to
ensure an isotropic manifold where Euclidean dis-
tances faithfully represent semantic dissimilarity.
Second, for every evaluated sample, we compute
its mean VFE. This metric serves as a mechanistic
proxy for the model’s epistemic uncertainty, captur-
ing the degree of conflict between the visual scene
and the user claim.

Third, rather than utilizing arbitrary hard thresh-
olds, we rank the entire population based on the
calculated VFE scores. We then select the 500
samples with the highest VFE values to constitute
the manifold outliers and the 500 samples with
the lowest VFE values to serve as stable inliers.
This results in a final benchmark of 1,000 samples
that represent the polar extremes of the uncertainty
distribution.
3.2 Dataset Characteristics
The bimodal composition of ModeVent allows for
a rigorous assessment of the intervention paradox.
The high-VFE subset represents adversarial-like
boundary cases where the model is most suscepti-
ble to sycophancy or causal imposition. In these
cases, the semantic stability is significantly lower,
and the noise ratio is elevated, as shown in our
quantitative analysis in fig. 2 .
Conversely, the low-VFE subset provides a sta-
ble baseline of well-aligned multimodal queries.
This ensures that the gating mechanisms of MODE-
RAG can be tested for their ability to bypass unnec-
essary interventions, thereby maintaining the inher-
ent accuracy of the underlying LLM kernel when
no significant conflict is detected. By targeting
these extremes, ModeVent provides a more chal-
lenging and informative evaluation environment
than standard multimodal datasets.
4 Methodology: The MODE-RAG
Framework
We proposeMODE-RAG(Multimodal Objective
Diagnostic Energy-RAG), a Multi-Agent frame-
work designed to resolve theintervention paradox
in multimodal reasoning. The architecture is struc-
tured as a hierarchical, energy-gated system that se-
lectively triggers high-fidelity reasoning only when
epistemic uncertainty is detected. As illustrated in
the system diagram, the framework comprises a
diagnostic data pipeline, two gating mechanisms,
and a decoupled five-agent pipeline.
4.1 Thermodynamic Gating: The FE-Router
The entry point of theMODE-RAGsystem is the
FE-Router, which serves as a “Thermodynamic
Gate.” Utilizing theATLAS Probe, the router per-
forms real-timeEnergy Detectionby calculating
theVariational Free Energy (VFE)of the predic-
tive distribution(Friston, 2010). For a model with
vocabulary Vand logit output f(x) , given a vari-
ational distribution q(j) over the tokens, the VFE(F) at temperatureτis defined as
F(q, x;τ) =|V|X
j=1q(j) [−f j(x) +τlogq(j)](1)
where−fj(x)represents the internal energy of the
j-th state and τlogq(j) contributes to the entropic
regularization. This formulation captures the dis-
crepancy between the model’s internal beliefs and
the categorical evidence provided by the input.
When the input presents aCausal Imposition
Conflict—where a user’s “Claim” (e.g., a flawless
backflip) contradicts the “Scene” (e.g., standing
still)—theVFEtypically spikes, signaling high
epistemic uncertainty and a breakdown in predic-
tive coding. If the mean variational free energy
¯F> γ , theFE-Routerintercepts the standard
generation and activates the specialized Agentic
Pipeline.
4.2 The MODE-RAG Five-Agent Decoupled
Intervention Pipeline
Upon activation by the FE-Router, the query is di-
verted into a specialized multi-agent ecosystem(Wu
et al., 2024). This pipeline is designed to decouple
the monolithic reasoning process into five granu-
lar, verifiable stages, ensuring that each potential
source of hallucination from perception errors to
sycophantic synthesis is systematically neutralized.
Per-Agent: Atomic Facts Objective Scan.The
Per-Agentserves as the framework’s sensory foun-
dation, performing anAtomic Facts Objective Scan.
It extracts symbolic triplets V={⟨s, p, o⟩} from
the visual stream (e.g., ⟨subject,is,stationary⟩ ). By
utilizing high-resolution spatial-temporal ground-
ing, the Per-Agent fixates on physical invariants,
creating a “Grounded Truth Anchor.” This ensures
that subsequent reasoning agents cannot bypass the
physical reality of the scene in favor of the user’s
potentially biased “Claim.”
Cor-Agent: Facade Pattern Algorithmic Wrap.
TheCor-Agentacts as the structural architect
by implementing aFacade Pattern Algorithmic
Wrap. Its primary role is to maintain the integrity
of the cross-agent data flow. By encapsulating raw
multimodal features and the Per-Agent’s triplets
into a strictly validated programmatic schema (e.g.,
JSON-Schema), the Cor-Agent prevents “seman-
tic noise leakage.” This wrapper ensures that the
complex reasoning in later stages is performed on

−1.0 −0.905101520Density
log10() (Energy Density)(a) Log-Scaled VFE Distribution
0.1 0.2 0.3 0.4 0.5Stable Inliers
Manifold Outliers
Log-Threshold γ*
−1.0 −0.91.01.11.21.31.4Stability −log10()
log10() (Energy Density)(b) Information Phase Space
0.1 0.2 0.3 0.4 0.5Stable Inliers
Manifold OutliersFigure 2: Thermodynamic empirical evidence: (a) VFE
distribution across subsets used for γcalibration; (b)
Correlation between Energy and Stability.
structured, high-fidelity data rather than ambiguous
natural language strings.
Ret-Agent: RAG Alignment and Conflict Dis-
card.TheRet-Agentmanages the external
knowledge interface to mitigateSycophancy, where
the model over-relies on biased retrieved docu-
ments. Beyond simple semantic similarity, the
agent evaluates theManifold Fidelityof each
document diby measuring its alignment with the
grounded triplets Vin the whitened latent space(Su
et al., 2021). The filtering mechanism is set as
Score(d i) =Sim i· Si·Ii (2)
where the exponential term penalizes documents
that fall into the high-energy "Log-Outlier" regions
identified in Fig. 2b.By calculating the distance between the retrieved
context and the physical invariants V, the Ret-
Agent proactively identifies contexts that trigger
Energy Collapse. If a retrieved document di
promotes a causal fabrication that contradicts the
physical evidence (e.g., describing a backflip dur-
ing a stationary state), its stability score drops to-
ward the outlier cluster, triggering aConflict Dis-
cardoperation to prune the biased context before
it reaches the reasoning layer.
Rea-Agent: Test-Time Scaling via MCTS.The
Rea-Agentis the cognitive engine of MODE-RAG,
implementingMonte Carlo Tree Search (MCTS)
for test-time reasoning scaling(Silver et al., 2016).
Drawing on policy optimization principles, the Rea-
Agent explores the logical space by constructing a
Causal Directed Acyclic Graph (DAG).
The MCTS process follows a four-phase cycle
to identify the most plausible causal trajectory:
•Selection:Starting from the root (observed
scene), the agent traverses the tree using the
Upper Confidence Bound for Trees (UCT)
formula:
UCT(s, a) =Q(s, a) +c puct·P(a|s)
·pPN
1 +N(s, a)(3)
This balances the exploitation of high-fidelity
paths with the exploration of alternative causal
interpretations.
•Expansion & Simulation:For each leaf node,
the agent generates kcandidate reasoning
steps and performs aRolloutto simulate logi-
cal consequences (“If the state is stationary, is
the claimed action physically reachable?”).
•Evaluation & Backpropagation:Each path
is assigned a reward R(s) based on its align-
ment withATLAS(Adaptive Token-Layer
Attention Scoring) feedback and physical con-
straints. These values are propagated back to
the root to update the reasoning policy.
Gen-Agent: Objective Synthesis and Logit Per-
turbation.The final stage is managed by the
Gen-Agent, which serves as anInformation Bot-
tleneck. It synthesizes the MCTS findings into
a coherent response. To combat prompt-induced
bias, the Gen-Agent appliesLogit Perturbation.
during decoding, penalizing tokens that align with
the user’s hallucination keywords while boosting
tokens that align with the Rea-Agent’s causal DAG.

Figure 3: When a multimodal query is accompanied by a potentially adversarial retrieved context, the FE-Router
dynamically evaluates the epistemic risk via Variational Free Energy (VFE) and ATLAS attention states. High-risk
queries exceeding the threshold trigger a decoupled five-agent pipeline: the Per-Agent extracts objective multimodal
facts, the Cor-Agent enforces schema validation to block semantic noise, the Ret-Agent evaluates manifold distance
to discard conflicting context, the Rea-Agent constructs a causal DAG via MCTS, and the Gen-Agent synthesizes the
output using logit perturbation. Finally, a PORAG-driven Overseer conducts a triple-consistency check, activating a
Recursive Fallback Mechanism for unresolved conflicts to ensure a hallucination-free factual report.
4.3 Quality Oversight: PORAG-driven
Overseer and Fallback Loop
The final synthesis stage is governed by theOver-
seer, a specialized secondary gate that implements
thePolicy-Oriented RAG (PORAG)protocol.
PORAG Fidelity Cross-CheckThe PORAG-
driven Overseer evaluates the report based on a
Policy-Grounded Fidelity Metric. It performs a
triple-consistency check between: (1) thePer-
Agent’ssymbolic triplets V, (2) theRea-Agent’s
causal DAG, and (3) theGen-Agent’ssynthesized
natural language. By treating the response genera-
tion as a policy optimization problem, the Overseer
assigns a penalty to any output that restores "hallu-
cinatory maneuvers" previously pruned by MCTS.
The Recursive Fallback Mechanism.A critical
innovation of MODE-RAG is its non-linearFall-
back Loop. If the Overseer detects that the fidelity
score falls below a safety threshold ϵ, the system
triggers aTest-Time Reasoning Extension:
•Search Depth Scaling:The query is returned
to theRea-Agent, which re-initiates MCTS
with a significantly increased simulation bud-
getNand a broader expansion factork.
•Epistemic Refusal:If after Mrecursive at-
tempts the causal conflict remains unresolved,
the Overseer forces the system into a state
ofEpistemic Refusal, outputting a "Corrected
Factual Report" that explicitly identifies thecontradiction between visual evidence and
user claim.
5 Experiments
To rigorously evaluate the effectiveness of MODE-
RAG, we conduct comprehensive experiments on
our ModeVent benchmark. Unlike traditional hallu-
cination evaluations that rely on static datasets, our
experimental design explicitly targets the dynamic
nature of Retrieval-Augmented Generation (RAG)
failures.
5.1 Experimental Setup
RAG Errors vs. Hallucination Typology.It is
crucial to clarify the relationship between the exper-
imental categories and the hallucination typology
defined in Section 1. In standard M-RAG pipelines,
a single type ofretrieval errorcan cascade into mul-
tiple downstreamgeneration hallucinations. There-
fore, our benchmark generates adversarial contexts
across7 distinct RAG Error Categories(e.g., At-
tribute Hijacking, Metadata Redundancy, Informa-
tion Sparsity). These 7 input-side retrieval errors
act as the mechanistic triggers that induce the 9
output-side hallucination types (e.g., temporal in-
version, causal fabrication) observed in the wild.
Adversarial Benchmark Generation.To con-
struct a highly controlled adversarial environment,
we employ an automated generation pipeline using
DeepSeek-V3.2. First, we establish anObjective
Ground Truth (GT)for each video by fusing global

semantic summaries generated by Qwen3-Omni-
30B with dense, frame-level captions extracted via
Florence-2. Guided by these GT facts, we prompt
DeepSeek to synthesize challenging user queries
alongside noisy or adversarial retrieved text chunks
(mock contexts). These contexts are deliberately
injected with the 7 RAG errors and stratified into
two difficulty levels:Inliers(In-Domain texts con-
taining subtle factual discrepancies) andOutliers
(Out-of-Domain texts that are entirely irrelevant or
contain aggressive metadata noise).
Baselines and Implementation Details.For
both the Baseline and MODE-RAG, we utilize
Qwen-2.5-VL-7B, a representative 7B-parameter
instruction-tuned Vision-Language Model (VLM),
as the foundational kernel. To ensure a compre-
hensive evaluation, we also evaluate our frame-
work against three established alternative mitiga-
tion paradigms: Self-RAG, SelfCheckGPT, and
Woodpecker. Due to space constraints, the com-
plete comparative results across all five configura-
tions are detailed in Appendix B. All experiments,
including the MCTS expansion and Multi-Agent
inference, are deployed on a hardware cluster com-
prising 4 ×NVIDIA RTX 4090 GPUs. To ensure
generation stability and suppress auto-regressive
stuttering, we apply a repetition penalty of 1.15
during decoding.
LLM-as-a-Judge Evaluation Mechanism.Due
to the limitations of traditional string-matching met-
rics in evaluating complex multimodal reasoning,
we implement a robust LLM-as-a-Judge protocol
using DeepSeek-V3.2. The judge is provided with
the Objective GT and evaluates the model outputs
across two orthogonal dimensions:
•Fidelity (F) [0-5]:Measures the strict adher-
ence to visual facts. Penalizes the model for
fabricating entities, imposing fake causality,
or suffering from mechanistic mode collapse.
•Resilience (R) [0-5]:Measures the complete-
ness of information extraction. Penalizes the
model for being hijacked by adversarial text,
omitting crucial visual details, or triggering
unjustified epistemic refusal.
5.2 Main Results and Quantitative Analysis
As shown in Table 1, MODE-RAG significantly
and consistently outperforms the Baseline acrossall 7 RAG error categories, achieving a globalAver-
age Total Score improvement of +1.04(from 4.40
to 5.45). The dual-dimension analysis reveals that
our system successfully resolves the intervention
paradox by boosting Fidelity ( ∆F = +0.89) without
sacrificing information extraction (∆R = +0.16).
Conquering Outliers Hijacking.In Outliers sce-
narios, traditional RAG models suffer from severe
"Attention Hijacking," where the LLM abandons
visual evidence to blindly follow irrelevant or mali-
cious text. Our results show that MODE-RAG ex-
cels in these extreme conditions, yielding a massive
∆Total improvement of+1.48. The most striking
gains are observed inMajority Text Bias( ∆Total
= +2.31) andOut-of-Domain Irrelevance( ∆Total
= +1.68). This validates the efficacy of ourRet-
Agent. By explicitly calculating the manifold dis-
tance between the text and theVisual Logic Graph,
the system accurately detects epistemic uncertainty
and triggers the [EMPTY CONTEXT FALLBACK] ,
forcing the model to anchor its generation purely
on the physical visual evidence rather than fabri-
cated text.
Refining Inliers Extraction.Inliers scenarios
present a highly nuanced challenge: the retrieved
text is semantically relevant but contains redundant
metadata or slightly conflicting attributes. A naive
filtering approach often leads to unjustified refusal,
resulting in low Resilience. However, MODE-RAG
achieves a +0.60 ∆Total improvement in Inliers
cases. Notably, in theInformation Sparsitycate-
gory, our model achieves a significant ∆Total of
+0.93. This demonstrates the success of theSmart
Synthesisprotocol within the Gen-Agent, which
safely fuses domain-specific nouns from the text
(e.g., specific names or medical terms) with the
MCTS-verified visual actions, thereby preserving
rich background context without hallucinating ac-
tions.
Performance Stability and Failure Suppression.
While Table 1 demonstrates mean improvements,
Figure 4 provides a deeper look into the system’s
robustness by visualizing the score distribution. A
critical observation is the suppression of “catas-
trophic failures” in the 0–2 score range. In cate-
gories likeMajority Text BiasandMetadata Redun-
dancy, the Baseline distribution exhibits a signifi-
cant density bulge at the bottom, corresponding to
cases where the model suffered from severe mode
collapse (e.g., stuttering loops) or total attention

Table 1: Comprehensive Evaluation on the ModeVent Benchmark. We report Fidelity (F), Resilience (R), and
Total Scores across 7 major hallucination categories. The results are further stratified by semantic distance:Inliers
(In-Domain interference) andOutliers(Out-of-Domain irrelevance). The best results in each comparison are
highlighted inbold.
Error Category Data LabelBaseline MODE-RAG (Ours) Improvement (∆)
F R Tot F R Tot∆F∆R∆Tot
Attribute HijackingInliers 1.45 0.85 2.302.40 1.26 3.66+0.95 +0.41 +1.36
Outliers 1.91 1.34 3.252.38 1.64 4.02+0.47 +0.30 +0.77
Overall1.66 1.08 2.742.39 1.44 3.82+0.72 +0.36 +1.08
Causal ImpositionInliers 1.821.783.602.641.394.03+0.82 -0.39 +0.43
Outliers 1.86 1.93 3.792.78 2.19 4.97+0.92 +0.26 +1.18
Overall1.841.863.702.711.814.52+0.87 -0.05 +0.82
Information SparsityInliers 2.32 1.61 3.923.14 1.71 4.86+0.83 +0.11 +0.93
Outliers 2.05 1.88 3.933.60 2.10 5.71+1.55 +0.22 +1.78
Overall2.20 1.72 3.933.34 1.88 5.22+1.14 +0.16 +1.30
Majority Text BiasInliers 2.832.985.823.402.836.23+0.57 -0.15 +0.42
Outliers 2.43 2.61 5.043.91 3.45 7.36+1.48 +0.84 +2.31
Overall2.62 2.79 5.413.67 3.16 6.83+1.05 +0.37 +1.42
Metadata RedundancyInliers 2.942.325.263.802.025.82+0.86 -0.30 +0.56
Outliers 2.53 2.41 4.943.71 2.77 6.49+1.19 +0.36 +1.54
Overall2.73 2.37 5.103.76 2.40 6.16+1.03 +0.04 +1.07
Out-of-Domain IrrelevanceInliers 2.862.195.053.311.945.25+0.45 -0.25 +0.20
Outliers 2.75 2.72 5.464.00 3.14 7.14+1.25 +0.42 +1.68
Overall2.80 2.47 5.273.67 2.57 6.24+0.87 +0.10 +0.98
Scene MisalignmentInliers 2.562.134.692.891.904.79+0.32 -0.23 +0.10
Outliers 2.61 2.29 4.903.30 2.74 6.04+0.70 +0.45 +1.14
Overall2.59 2.21 4.803.11 2.34 5.45+0.52 +0.13 +0.65
AverageInliers2.371.944.313.071.844.91+0.70 -0.10 +0.60
Outliers2.31 2.18 4.503.39 2.59 5.98+1.07 +0.41 +1.48
Overall2.34 2.06 4.403.23 2.22 5.45+0.89 +0.16 +1.04
hijacking. In contrast, MODE-RAG’s distribution
is markedly narrower at the base, effectively estab-
lishing a “safety floor” through theDead Man’s
SwitchandMCTS pruningmechanisms.
Furthermore, theOveralldensity for MODE-
RAG shows a decisive upward shift, with the me-
dian score and interquartile ranges positioned sub-
stantially higher than the Baseline. This shift
is most prominent inOut-of-Domain Irrelevance,
where MODE-RAG transforms a low-fidelity bi-
modal distribution into a concentrated high-score
peak. This proves that theFE-Routercorrectly
identifies high-uncertainty scenarios, allowing the
multi-agent pipeline to neutralize adversarial noise
and anchor the final generation to the physically-
grounded visual logic.
While the results above confirm that MODE-
RAG consistently outperforms the vanilla founda-
tional kernel, we further evaluate our framework
against alternative mitigation paradigms to ensure
a thorough assessment. The full benchmarking re-
sults across all five methods (Vanilla Baseline, Self-
RAG, SelfCheckGPT, Woodpecker, and MODE-RAG) on the ModeVent dataset are detailed in Ap-
pendix B.
5.3 Ablation on Mechanistic Failures
Beyond semantic conflicts, our error logs revealed
that lightweight LLM kernels frequently suffer
from mechanistic failures under adversarial stress.
We observed two primary collapse patterns in the
Baseline:Mode Collapse(e.g., severe stuttering
loops like "even even even") andPrompt Bleed-
through(leaking internal system tags or metadata
like "addCriterion"). These failures historically re-
sulted in 0-point scores for Fidelity. By incorporat-
ing an internal, rule-basedDead Man’s Switch
within the Gen-Agent—a deterministic regular-
expression interceptor, MODE-RAG effectively es-
tablishes a safety floor. This mechanism success-
fully neutralizes catastrophic formatting failures,
seamlessly downgrading to a safe textual reading-
comprehension state when the VLM’s predictive
coding collapses.
To explicitly demonstrate how our decoupled
architecture resolves the intervention paradox in

/uni00000024/uni00000057/uni00000057/uni00000055/uni0000004c/uni00000045/uni00000058/uni00000057/uni00000048/uni00000003/uni0000002b/uni0000004c/uni0000004d/uni00000044/uni00000046/uni0000004e/uni0000004c/uni00000051/uni0000004a /uni00000026/uni00000044/uni00000058/uni00000056/uni00000044/uni0000004f/uni00000003/uni0000002c/uni00000050/uni00000053/uni00000052/uni00000056/uni0000004c/uni00000057/uni0000004c/uni00000052/uni00000051/uni0000002c/uni00000051/uni00000049/uni00000052/uni00000055/uni00000050/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000036/uni00000053/uni00000044/uni00000055/uni00000056/uni0000004c/uni00000057/uni0000005c /uni00000030/uni00000044/uni0000004d/uni00000052/uni00000055/uni0000004c/uni00000057/uni0000005c/uni00000003/uni00000037/uni00000048/uni0000005b/uni00000057/uni00000003/uni00000025/uni0000004c/uni00000044/uni00000056/uni00000030/uni00000048/uni00000057/uni00000044/uni00000047/uni00000044/uni00000057/uni00000044/uni00000003/uni00000035/uni00000048/uni00000047/uni00000058/uni00000051/uni00000047/uni00000044/uni00000051/uni00000046/uni0000005c/uni00000032/uni00000058/uni00000057/uni00000010/uni00000052/uni00000049/uni00000010/uni00000027/uni00000052/uni00000050/uni00000044/uni0000004c/uni00000051/uni00000003/uni0000002c/uni00000055/uni00000055/uni00000048/uni0000004f/uni00000048/uni00000059/uni00000044/uni00000051/uni00000046/uni00000048/uni00000036/uni00000046/uni00000048/uni00000051/uni00000048/uni00000003/uni00000030/uni0000004c/uni00000056/uni00000044/uni0000004f/uni0000004c/uni0000004a/uni00000051/uni00000050/uni00000048/uni00000051/uni00000057/uni00000032/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004f/uni0000004f
/uni00000035/uni00000024/uni0000002a/uni00000003/uni00000028/uni00000055/uni00000055/uni00000052/uni00000055/uni00000003/uni00000026/uni00000044/uni00000057/uni00000048/uni0000004a/uni00000052/uni00000055/uni0000005c/uni00000013/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c/uni00000014/uni00000013/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048/uni00000003/uni0000000bF+R/uni0000000c/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048
/uni00000030/uni00000032/uni00000027/uni00000028/uni00000010/uni00000035/uni00000024/uni0000002a/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFigure 4: Split violin plots of Total Scores (Fidelity + Resilience) across seven RAG error categories and overall
performance. The left (blue) and right (orange) distributions represent the Baseline and MODE-RAG, respectively.
Dotted lines indicate the median and interquartile ranges. MODE-RAG significantly suppresses zero-score catas-
trophic failures and shifts the performance mass towards high-fidelity regions.
practice, we provide a detailed comparative case
study of four adversarial testing scenarios inAp-
pendix A.
5.4 Computational Efficiency Analysis
To evaluate the practical deployability of MODE-
RAG, we analyze its computational overhead
against the Vanilla M-RAG baseline across the
1,000 video queries in the ModeVent benchmark.
On average, the baseline foundational kernel re-
quires 18.5 seconds to process a single multimodal
query. In comparison, due to the multi-agent or-
chestration and MCTS-guided test-time reasoning
scaling, MODE-RAG increases the average pro-
cessing time to 26.2 seconds per query. This rep-
resents a moderate 1.42 ×increase in time con-
sumption, translating to approximately 7.3 hours
of execution time when evaluating the entire bench-
mark sequentially on a single-threaded pipeline.
It is worth noting that because the stage-specific
agent interventions and evaluation queries are inher-
ently decoupled, this computational overhead can
be significantly mitigated through standard multi-
threading, asynchronous scheduling, and parallel
execution techniques in production environments.
6 Conclusions
In this paper, we proposed MODE-RAG, a mech-
anistically grounded multi-agent framework that
addresses the intervention paradox in multimodal
RAG systems by dynamically gating interventionsthrough a router driven by Variational Free En-
ergy (VFE) and internal attention states (ATLAS).
By categorizing hallucinations into nine distinct
types across the system’s lifecycle, we developed
specialized agents—integrating Monte Carlo Tree
Search (MCTS) for causal derivation and logit per-
turbations for sycophancy suppression—to ensure
factual grounding and logical consistency. Fur-
thermore, we introduced ModeVent, a targeted
benchmark designed to evaluate system suscep-
tibility to manifold outliers and complex visual-
textual conflicts. Experimental results demonstrate
that MODE-RAG effectively reduces hallucination
rates and enhances the structural stability of M-
RAG systems, providing a robust and scalable so-
lution for reliable multimodal reasoning.
Acknowledgment
This work was supported by the Ministry of Sci-
ence and Technology of China under Grant No.
2025ZD0123800, the HUST Interdisciplinary Re-
search Program under Grant No. 2025JCYJ077,
and the KingSoft 2026 University-Industry Project.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations (ICLR).
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang,

Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou,
and Jing Jing. 2023. Qwen-vl: A versatile
vision-language model for understanding, localiza-
tion, text reading, and beyond.arXiv preprint
arXiv:2308.12966.
Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and
William W Cohen. 2022. Murag: Multimodal
retrieval-augmented generator for open question an-
swering. InProceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition
(CVPR), pages 10597–10607.
Florin Cuconasu, Giovanni Trappolini, Federico Si-
ciliani, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Motta, and Fabrizio Silvestri. 2024.
The power of noise: Redefining retrieval for rag sys-
tems.Proceedings of the 47th International ACM
SIGIR Conference on Research and Development in
Information Retrieval.
Karl Friston. 2010. The free-energy principle: a uni-
fied brain theory?Nature reviews neuroscience,
11(2):127–138.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey.arXiv preprint
arXiv:2312.10997.
Jonas Geiping, Sean McLeish, Naman Jain, John
Kirchenbauer, Siddharth Singh, Brian R Bartoldson,
Bhavya Kailkhura, Abhinav Bhatele, and Tom Gold-
stein. 2025. Scaling up test-time compute with la-
tent reasoning: A recurrent depth approach.arXiv
preprint arXiv:2502.05171.
Yunjie Ji, Jiawei Li, Haiyan Ye, Kehai Wu, Jun Xu, Lin
Mo, and Min Zhang. 2025. Test-time computing:
from system-1 thinking to system-2 thinking.arXiv
preprint arXiv:2501.02497.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu,
Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of halluci-
nation in natural language generation.ACM Comput-
ing Surveys, 55(12):1–38.
Jiarui Jiang, Zhiyu Chen, Yifei Min, Jian Chen, Xiaoyu
Cheng, Jian Wang, Yuxin Tang, Hao Sun, Jia Deng,
Wayne Xin Zhao, and 1 others. 2024. Technical
report: Enhancing llm reasoning with reward-guided
tree search.arXiv preprint arXiv:2411.11694.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks. InAd-
vances in Neural Information Processing Systems
(NeurIPS), volume 33, pages 9459–9474.
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang,
Wayne Xin Zhao, and Ji-Rong Wen. 2023. Eval-
uating object hallucination in large vision-languagemodels. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing
(EMNLP), pages 292–305.
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan
Li. 2020. Energy-based out-of-distribution detection.
InAdvances in Neural Information Processing Sys-
tems (NeurIPS), volume 33, pages 21464–21475.
Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.
SelfCheckGPT: Zero-resource black-box hallucina-
tion detection for generative large language models.
InProceedings of the 2023 Conference on Empiri-
cal Methods in Natural Language Processing, pages
9004–9017, Singapore. Association for Computa-
tional Linguistics.
Hongjin Qian, Peitian Zhang, Zheng Liu, Kelong Mao,
and Zhicheng Dou. 2024. Memorag: Moving to-
wards next-gen rag via memory-inspired knowledge
discovery.arXiv preprint arXiv:2409.05591.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark,
and 1 others. 2021. Learning transferable visual
models from natural language supervision. InInter-
national Conference on Machine Learning (ICML),
pages 8748–8763. PMLR.
Sagar Srinivas Sakhinana, Shivam Gupta, Akash
Das, and Venkataramana Runkana. 2025. Scaling
test-time inference with policy-optimized, dynamic
retrieval-augmented generation via KV caching and
decoding. InKDD 2025 Workshop on Inference Op-
timization for Generative AI.
David Silver, Aja Huang, Chris J Maddison, Arthur
Guez, Laurent Sifre, George Van Den Driessche, Ju-
lian Schrittwieser, Ioannis Antonoglou, Veda Pan-
neershelvam, Marc Lanctot, and 1 others. 2016. Mas-
tering the game of go with deep neural networks and
tree search.nature, 529(7587):484–489.
Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyi Ou.
2021. Whitening sentence representations for bet-
ter semantics and faster retrieval.arXiv preprint
arXiv:2103.15316.
Weijia Su, Yubai Tang, Qingyao Ai, Zhijing Wu, and
Yiqun Liu. 2024. Dragin: Dynamic retrieval aug-
mented generation based on the real-time informa-
tion needs of large language models.arXiv preprint
arXiv:2403.10081.
Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li.
2022. Out-of-distribution detection with deep nearest
neighbors. InInternational Conference on Machine
Learning (ICML), pages 20827–20840. PMLR.
Zijie Wang, Zihan certification Wang, Linyi Le,
Hao Shen Zheng, Swaroop Mishra, Vincent Perot,
Yashan Zhang, Ankit Mattapalli, Ankur Taly, Jingbo
Shang, and 1 others. 2024. Speculative rag: Enhanc-
ing retrieval augmented generation through drafting.
arXiv preprint arXiv:2407.08223.

Jianing Wu, Mengwei Feng, Shiwei Zhang, Ren Jin, Fan
Che, Zhi Wen, and Jianhua Tao. 2025. Boosting mul-
timodal reasoning with mcts-automated structured
thinking.arXiv preprint arXiv:2502.02339.
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu,
Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang,
Shaokun Zhang, Jiale Liu, and 1 others. 2024. Au-
togen: Enabling next-gen llm applications via multi-
agent conversations. InFirst conference on language
modeling.
Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi,
Richard Lewis, Luke Zettlemoyer, Percy Liang,
Luke Zettlemoyer, and 1 others. 2022. Retrieval-
augmented multimodal language modeling. InInter-
national Conference on Machine Learning (ICML),
pages 25439–25460. PMLR.
Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao
Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun,
and Enhong Chen. 2024. Woodpecker: Hallucina-
tion correction for multimodal large language models.
Science China Information Sciences, 67(12):220105.
Ori Yoran, Ori Wolfson, Tom/and Ram, and Jonathan
Berant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InThe Twelfth In-
ternational Conference on Learning Representations
(ICLR).
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov,
and Lucas Beyer. 2023. Sigmoid loss for lan-
guage image pre-training. InProceedings of the
IEEE/CVF International Conference on Computer
Vision (ICCV), pages 11975–11986.
Yu Zhao, Huajian Yin, Bo Zeng, Hao Wang, Teng Shi,
Chen Lyu, Longyue Wang, Weihua Luo, and Kaizhu
Zhang. 2024. Marco-o1: Towards open reasoning
models for open-ended solutions.arXiv preprint
arXiv:2411.14405.
Appendix
A Case Studies
Mechanistic Analysis of System Interventions
This appendix provides additional qualitative evi-
dence on how the decoupled architecture resolves
the intervention paradox, we present a compara-
tive analysis of four adversarial queries from the
ModeVent benchmark test logs.
Objective Visual Ground Truth.Across all four
test scenarios detailed in Table A1, the underlying
visual evidence remains constant: the video depicts
a person in a serene, snowy forest, either wearing
snowshoes or standing still on a snowboard prepar-
ing to descend. There are no extreme stunts, com-
petitive sporting events, or water-related elements
present in the actual footage.Combating Attribute Hijacking and Perception
Omission.Scenarios 1 and 2 highlight the Base-
line’s vulnerability to semantic coercion. Despite
the visual evidence clearly showing a snowboard or
snowshoes, the injection of text describing “cross-
country skiing” or “twin-tip skis” hijacked the
Baseline’s attention, causing it to blindly halluci-
nate equipment not present in the video. In contrast,
MODE-RAG’sPer-Agentenforces a strict “visual-
first” extraction. By isolating atomic visual facts
before textual integration, the system successfully
overrides the adversarial text, accurately maintain-
ing the physical reality of the scene.
Suppressing Sycophancy and Causal Fabrica-
tion.Scenario 3 demonstrates a severe case of
Causal Imposition. Confronted with text describ-
ing competitive winter sports, the Baseline model
exhibits extreme sycophancy, inventing a massive,
Hollywood-style narrative involving a “triple cork
1440 spin” and “outrunning an avalanche.” This
exposes the danger of unguided LLM reasoning,
where the model prioritizes narrative alignment
with the text over physical constraints. MODE-
RAG neutralizes this through theRea-Agent’s
MCTS DAG. Since an avalanche or a backflip can-
not be topologically derived from the Per-Agent’s
root node (standing still), the MCTS prunes these
branches entirely, allowing the Gen-Agent to safely
output a justified epistemic refusal regarding the
background context.
Preventing Mechanistic Mode Collapse.Sce-
nario 4 exposes a critical physical limitation of
lightweight LLM kernels. When subjected to Ma-
jority Text Bias (a 3:1 ratio of water-sports noise to
relevant snow text), the Baseline model’s attention
mechanism collapses under the conflicting seman-
tic density, resulting in a stuttering loop and system
paralysis. MODE-RAG bypasses this failure mode
completely. Prior to generation, theRet-Agent
actively computes the manifold distance between
the visual log and the candidate texts, discarding
the three contradictory water-sports chunks upfront.
This listwise cross-check purifies the context win-
dow, feeding the generator a clean, aligned prompt
that guarantees formatting stability and flawless
factual synthesis.
B Results on Additional Backbones
To comprehensively verify the effectiveness of the
MODE-RAG framework, we conduct an extensive

Table A1: Comparative analysis of system outputs across four adversarial testing scenarios. Key hallucinations and
mode collapses in the Baseline are highlighted inbold. Case separation is enforced using heavy rules to maintain
visual hierarchy.
Scenario 1: Physical Scene Replacement
Query:What is the person in the video doing? What is the environment like?
Retrieved Text:[4 adversarial chunks describing intense water sports: surfing, floods, etc.]
Baseline (Vanilla M-RAG) MODE-RAG (Ours)
The person in the video appears to be engaging in
snowshoeing or cross-country skiingin a snowy forested
area...The person in the video isstanding on a snowboard,
preparing to descend a snowy slope. The environment
appears to be a snowy landscape...
Scenario 2: Micro-Detail Tampering
Query:What equipment is the person using in the snow?
Retrieved Text:[3 chunks describing "twin-tip skis," "bright red skis," and "carbon fiber poles."]
Baseline (Vanilla M-RAG) MODE-RAG (Ours)
The person in the video is usingtwin-tip skis and carbon
fiber poles.The person is usingsnowshoesin the snow.
Scenario 3: High-Order Causal Imposition
Query:What equipment is the person using in the snow?
Retrieved Text:[Chunks describing competitive winter sports, Olympic trials, and avalanche events.]
Baseline (Vanilla M-RAG) MODE-RAG (Ours)
The person isperforming a backflip... practicing for the
Winter Olympics,attempting a triple cork 1440 spin...
actuallyoutrunning an avalanche.The specific action appears to be preparing for winter sports...
Insufficient information to determine the background of
this event.
Scenario 4: Majority Text Bias
Query:What is the person doing and where is this filmed?
Retrieved Text:[Mixed noise: 3 water-sports chunks vs. 1 relevant "Swiss Alps" snowboarding chunk.]
Baseline (Vanilla M-RAG) MODE-RAG (Ours)
The person in the video is backcountry snowboarding... The
location is the Swiss Alps, which is known for its its
[Followed by stuttering mode collapse].The person is preparing to descend a snowy slope on a
snowboard.This activity is filmed in theSwiss Alps,
specifically in a pristine snowy forest.
comparative analysis against multiple established
mitigation paradigms in recent literature. Specifi-
cally, our benchmark encompasses a total of five
distinct methodological configurations:
1.Vanilla M-RAG (Baseline):The founda-
tional unguided VLM (Qwen-2.5-VL-7B) ex-
ecuting direct multimodal generation.
2.Self-RAG(Asai et al., 2024): An end-to-end
framework that trains the model to self-reflect
on retrieved passages and generations via re-
flection tokens.
3.SelfCheckGPT(Manakul et al., 2023): A
zero-resource sampling-based approach that
detects hallucinations via stochastic consis-
tency checks.
4.Woodpecker(Yin et al., 2024): A training-
free, post-hoc correction pipeline designedto rectify multi-modal fabrications through
diagnostic querying.
5.MODE-RAG (Ours):Our proposed hierar-
chical, variational free energy-gated multi-
agent intervention framework.
Table B1 presents the full quantitative compari-
son across these five methods on the polar extremes
of the ModeVent dataset.
B.1 Implementation of Additional Baselines
While Vanilla M-RAG requires no architectural
modification and MODE-RAG is detailed in Sec-
tion 4, the remaining three baselines (Woodpecker,
SelfCheckGPT, and Self-RAG) were originally de-
veloped for static images or pure text. Below, we
outline the specific multimodal adaptations and
pipeline configurations required to deploy them
within our adversarial video RAG setting.

Table B1: Comprehensive Evaluation on the ModeVent Benchmark. We report Fidelity (F), Resilience (R), and Total
Scores across 7 major hallucination categories, further stratified by semantic distance:InliersandOutliers. Notably,
we introduce the Video-adaptedWoodpecker, the text-basedSelfCheckGPT, andSelf-RAGas competitive
baselines. Despite their strong performance, our proposedMODE-RAGmaintains a clear advantage across the
majority of metrics and scenarios. The best results in each comparison are highlighted inbold.
Error Category MethodInliers OutliersOverall
F R Total F R Total F R Total
Attribute HijackingBaseline 1.45 0.85 2.30 1.91 1.34 3.25 1.66 1.08 2.74
SelfCheckGPT 1.10 0.22 1.32 1.29 0.35 1.64 1.19 0.28 1.47
Self-RAG 2.23 1.29 3.522.701.89 4.592.441.564.01
Woodpecker 1.901.363.26 2.562.15 4.712.201.723.93
Ours 2.401.263.662.38 1.64 4.02 2.39 1.44 3.82
Causal ImpositionBaseline 1.821.783.60 1.86 1.93 3.79 1.84 1.86 3.70
SelfCheckGPT 1.07 0.45 1.52 1.00 0.40 1.40 1.03 0.42 1.46
Self-RAG2.761.634.393.05 2.17 5.232.911.91 4.82
Woodpecker 2.25 1.76 4.013.30 3.07 6.372.802.44 5.24
Ours2.64 1.39 4.03 2.78 2.19 4.97 2.71 1.81 4.52
Information SparsityBaseline 2.32 1.61 3.92 2.05 1.88 3.93 2.20 1.72 3.93
SelfCheckGPT4.801.165.96 4.351.445.79 4.601.285.89
Self-RAG 2.95 1.10 4.05 2.63 1.48 4.11 2.81 1.27 4.08
Woodpecker 2.20 1.30 3.51 2.752.154.90 2.44 1.67 4.12
Ours3.141.714.86 3.60 2.10 5.71 3.341.885.22
Majority Text BiasBaseline 2.832.985.82 2.43 2.61 5.04 2.62 2.79 5.41
SelfCheckGPT 1.61 1.20 2.80 1.25 1.01 2.26 1.41 1.09 2.50
Self-RAG 3.21 2.38 5.59 3.78 3.18 6.96 3.53 2.83 6.35
Woodpecker 3.02 2.35 5.37 3.11 2.76 5.87 3.07 2.58 5.65
Ours 3.402.836.23 3.91 3.45 7.36 3.67 3.16 6.83
Metadata RedundancyBaseline 2.942.325.26 2.53 2.41 4.94 2.73 2.37 5.10
SelfCheckGPT3.992.316.293.23 2.21 5.44 3.61 2.26 5.86
Self-RAG 3.01 2.03 5.04 3.51 2.74 6.25 3.26 2.39 5.65
Woodpecker 2.35 1.71 4.06 2.75 2.63 5.38 2.55 2.18 4.73
Ours3.80 2.02 5.823.71 2.77 6.49 3.76 2.40 6.16
Out-of-Domain IrrelevanceBaseline 2.86 2.19 5.05 2.75 2.72 5.46 2.80 2.47 5.27
SelfCheckGPT 1.34 0.23 1.56 2.14 0.41 2.55 1.75 0.32 2.06
Self-RAG3.55 2.21 5.763.55 2.74 6.29 3.55 2.48 6.03
Woodpecker 3.01 2.06 5.07 3.24 2.97 6.21 3.13 2.52 5.64
Ours3.31 1.94 5.254.00 3.14 7.14 3.67 2.57 6.24
Scene MisalignmentBaseline 2.56 2.13 4.69 2.61 2.29 4.90 2.59 2.21 4.80
SelfCheckGPT 1.05 0.02 1.06 1.03 0.07 1.10 1.04 0.05 1.08
Self-RAG3.29 2.26 5.553.27 2.42 5.703.282.345.63
Woodpecker 2.62 1.94 4.56 3.032.895.91 2.842.445.27
Ours2.89 1.90 4.793.302.746.043.11 2.34 5.45
AverageBaseline 2.371.944.31 2.31 2.18 4.50 2.34 2.06 4.40
SelfCheckGPT 2.20 0.80 3.00 1.99 0.84 2.83 2.10 0.82 2.92
Self-RAG 2.98 1.80 4.78 3.24 2.41 5.64 3.11 2.11 5.21
Woodpecker 2.45 1.75 4.21 2.972.685.65 2.712.224.93
Ours 3.071.844.91 3.392.595.98 3.23 2.22 5.45
Video-Adapted Woodpecker.We adapt the
Woodpecker framework (Yin et al., 2024)—ini-
tially an image-centric hallucination corrector—to
the video domain by shifting the focus from spatial
object misidentification to temporal dynamics (e.g.,
fabricated actions or incorrect event sequences).
The adapted pipeline operates in four stages: (1)
Drafting: A standard multimodal RAG setup gen-
erates an initial answer. (2) Question Generation:
An LLM extracts action-centric claims and tem-poral events from the draft, formulating targeted
verification questions. (3) Visual Verification: A
Video-LLM acts as an independent visual expert.
Crucially, external texts are masked to ensure the
model relies solely on raw video frames for ob-
jective fidelity. (4) Correction: The verified an-
swers form a Visual Fact-Sheet, guiding the LLM
to revise the initial draft and prune spatiotemporal
hallucinations.

Multimodal SelfCheckGPT.To complement the
visual-centric verification, we implement an alter-
native, uncertainty-based baseline by adapting Self-
CheckGPT (Manakul et al., 2023) from black-box
text evaluation to the multimodal RAG domain.
This zero-shot pipeline addresses adversarial tex-
tual noise through generation consistency, execut-
ing in three stages: (1) Multi-Sample Generation:
Multiple independent candidate answers are gen-
erated using high-temperature sampling. (2) Con-
sistency V oting: Instead of standard token-level
probability checks, a semantic overlap metric iden-
tifies the most frequent consensus among the can-
didates. (3) Refinement: The LLM acts as a strict
validator, cross-referencing the candidate consen-
sus against the raw retrieved texts to synthesize a
final factual response. Additionally, we integrate
a dynamic memory-recovery mechanism with pro-
gressive token-throttling to handle potential Out-
Of-Memory errors during large-scale evaluation.
Multimodal Self-RAG.Given that the original
Self-RAG (Asai et al., 2024) is a text-to-text frame-
work designed to critique retrieved textual passages,
we adapt it for video reasoning via a two-stage cas-
caded pipeline. This approach bridges the modality
gap while preserving the model’s reflective capabil-
ities: (1) Visual Translation: A Vision-Language
Model first processes the raw video frames along-
side the adversarial retrieved contexts to generate
a comprehensive text-based description of the vi-
sual scenes, actions, and objects. (2) Reflective
Generation: This visual description is subsequently
injected into the Self-RAG model using its native
retrieval syntax. Treating this textual translation
as the primary retrieved evidence, the Self-RAG
model leverages its intrinsic reflection tokens to
evaluate the fidelity of the provided information
and synthesize the final answer to the user’s query.
B.2 Result Analysis and Discussion
The comprehensive empirical results presented in
Table B1 demonstrate the performance trade-offs,
highlighting both the global strengths and the lo-
calized limitations of the proposed MODE-RAG
framework.
Overall Strengths and Outlier Robustness.
MODE-RAG achieves the highest global perfor-
mance with anOverallAverage Total Score of
5.45, consistently outperforming all four competi-
tive baselines (Baseline: 4.40, SelfCheckGPT: 2.92,
Self-RAG: 5.21, Woodpecker: 4.93). The primaryarchitectural advantage of our framework lies in
its exceptional robustness againstOutliers (Hard-
OOD)scenarios, where it reaches an average total
score of5.98. Specifically, in categories heavily
plagued by aggressive external text noise—such
asMajority Text Bias(7.36) andOut-of-Domain
Irrelevance(7.14)—MODE-RAG delivers a sub-
stantial performance leap. This consistently val-
idates the efficacy of our thermodynamic gating
via the FE-Router and the manifold filtering via the
Ret-Agent. By proactively evaluating the epistemic
uncertainty and discarding highly mismatched text
chunks upfront, our system effectively prevents the
LLM kernel from experiencing attention hijacking,
thereby securing a strong safety floor for factual
cross-modal synthesis.
Vulnerability to Information Sparsity.Despite
its global superiority, the multi-agent execution
within MODE-RAG exhibits localized deficits un-
der specific error contexts. In theInformation Spar-
sitycategory, MODE-RAG (Overall Total: 5.22)
is noticeably outperformed by the text-based Self-
CheckGPT, which achieves a dominant score of
5.89. This deficit occurs because when the retrieved
context is extremely sparse, SelfCheckGPT’s high-
temperature multi-sample consistency voting na-
tively excels at consensus-driven extraction. In
contrast, our rigid multi-agent validation schema
can occasionally become overly restrictive, leading
to redundant processing steps without gaining an
additional informative edge.
Conservative Pruning in Complex Reasoning.
Another limitation is observed in theCausal Im-
positioncategory, where Woodpecker outperforms
our method in both Outliers (6.37 vs. 4.97) and
Overall (5.24 vs. 4.52) metrics. A granular exam-
ination reveals that this is primarily driven by a
drop in our Resilience (R) scores (1.81 vs. Wood-
pecker’s 2.44). Because Woodpecker leverages
an aggressive post-hoc prompt rewriting strategy
based on direct question-answering, it forces the
model to actively correct claims. MODE-RAG,
conversely, relies on a strict MCTS causal DAG;
when a claim cannot be topologically derived from
the visual invariants, the system tends to trigger a
conservativeEpistemic Refusal(i.e., acknowledg-
ing insufficient information). While this strictness
preserves visual Fidelity, it inherently sacrifices de-
scriptive completeness (Resilience) when facing
high-order causal fabrications.

C Data Construction Examples
To automate the construction of the ModeVent
benchmark, we leveragedDeepSeek-V3.2to syn-
thesize adversarial test scenarios from MultiVent’s
ground truth. These misleading queries are strategi-
cally designed to reflect the hallucination taxonomy
introduced in section 1, ensuring a comprehensive
evaluation of model vulnerabilities. In this section,
we present representative examples of the challeng-
ing queries generated through this pipeline.
Example 1: Causal Imposition
Ground Truth:
This is a news report from TVBS News about a
medical condition called ¨cytokine storm, ¨which
can be fatal. The report features interviews with
doctors from Taipei Veterans General Hospital
and a nutritionist, who discuss how this immune
overreaction can damage organs like the lungs,
as shown by X-ray images. Experts advise that
during the pandemic, people should manage
stress, get enough sleep, and maintain a bal-
anced diet to strengthen their immune systems.
The segment also shows scenes of public health
measures, including disinfection at a train sta-
tion and people receiving vaccinations.”
Error Type:Causal Imposition
Generated Query:
What is being discussed in the news report, and
what is the purported origin of the medical con-
dition according to the background context?
Example 2: Attribute Hijacking
Ground Truth:
The video features an interview with Profes-
sor Chen Jian from Beijing Hospital’s Hepa-
tobiliary Surgery Department. He discusses
the treatment of liver abscesses, explaining that
early-stage cases are often managed conserva-
tively with antibiotics, while larger abscesses
may require drainage or surgical removal. The
conversation takes place in an office setting with
bookshelves in the background, and animated
graphics illustrate medical procedures like nee-
dle aspiration. A female host and a nurse are
present to conduct the interview.Error Type:Attribute Hijacking
Generated Query:
Who is the expert being interviewed in the
video, what medical condition is discussed, and
what are the backgrounds or settings shown dur-
ing the interview?
Example 3: Out-of-Domain Irrelevance
Ground Truth:
A news anchor presents a story about a political
controversy involving a leaked audio recording.
The broadcast displays images of politicians,
including President Yoon Suk-yeol, and shows
text from a social media post by Lee Jun-seok,
who denies being the source of the leak. A
press conference is shown where a masked man
speaks at a podium in front of the National
Assembly seal, addressing the allegations. The
report includes an animated graphic depicting
two silhouetted figures representing lawmakers
from the People Power Party, discussing the
situation.
Error Type:Out-of-Domain Irrelevance
Generated Query:
What is the main topic of the news report in the
video?
Example 4: Information Sparsity
Ground Truth:
The video is a news report from YTN about
a political controversy involving the People
Power Party. It features a female anchor in-
troducing the story, followed by on-screen text
messages allegedly exchanged between party
members discussing the possibility of a candi-
date’s withdrawal. The report includes footage
of a press conference with Kim Dong-cheol,
the party’s floor leader, who denies wrongdoing
and claims the matter was handled internally.
Other party figures, including Lee Yong-joo
and Lee Sang-tae, are shown speaking at events,
while opposition leaders like Park Hee-ryeon
and Ahn Cheol-soo are also featured. The seg-
ment concludes with a reporter providing an
update on the situation outside a government

building.
Error Type:Information Sparsity
Generated Query:
What are the specific details and sequence of
events reported in this news segment about the
political controversy?
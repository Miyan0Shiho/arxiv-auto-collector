# MAT-Cell: A Multi-Agent Tree-Structured Reasoning Framework for Batch-Level Single-Cell Annotation

**Authors**: Yehui Yang, Zelin Zang, Changxi Chi, Jingbo Zhou, Xienan Zheng, Yuzhe Jia, Chang Yu, Jinlin Wu, Fuji Yang, Jiebo Luo, Zhen Lei, Stan Z. Li

**Published**: 2026-04-07 04:56:06

**PDF URL**: [https://arxiv.org/pdf/2604.06269v1](https://arxiv.org/pdf/2604.06269v1)

## Abstract
Automated cellular reasoning faces a core dichotomy: supervised methods fall into the Reference Trap and fail to generalize to out-of-distribution cell states, while large language models (LLMs), without grounded biological priors, suffer from a Signal-to-Noise Paradox that produces spurious associations. We propose MAT-Cell, a neuro-symbolic reasoning framework that reframes single-cell analysis from black-box classification into constructive, verifiable proof generation. MAT-Cell injects symbolic constraints through adaptive Retrieval-Augmented Generation (RAG) to ground neural reasoning in biological axioms and reduce transcriptomic noise. It further employs a dialectic verification process with homogeneous rebuttal agents to audit and prune reasoning paths, forming syllogistic derivation trees that enforce logical consistency.Across large-scale and cross-species benchmarks, MAT-Cell significantly outperforms state-of-the-art (SOTA) models and maintains robust per-formance in challenging scenarios where baselinemethods severely degrade. Code is available at https://gith ub.com/jiangliu91/MAT-Cell-A-Mul ti-Agent-Tree-Structured-Reasoni ng-Framework-for-Batch-Level-Sin gle-Cell-Annotation.

## Full Text


<!-- PDF content starts -->

MAT-Cell: A Multi-Agent Tree-Structured Reasoning Framework for
Batch-Level Single-Cell Annotation
Yehui Yang1 2 3 4Zelin Zang1 3Changxi Chi1Jingbo Zhou1Xienan Zheng4Yuzhe Jia4Chang Yu1
Jinlin Wu1Fuji Yang3Jiebo Luo3Zhen Lei3Stan Z. Li1
Abstract
Automated cellular reasoning faces a core di-
chotomy: supervised methods fall into the Ref-
erence Trap and fail to generalize to out-of-
distribution cell states, while large language mod-
els (LLMs), without grounded biological priors,
suffer from a Signal-to-Noise Paradox that pro-
duces spurious associations. We propose MAT-
Cell, a neuro-symbolic reasoning framework that
reframes single-cell analysis from black-box clas-
sification into constructive, verifiable proof gen-
eration. MAT-Cell injects symbolic constraints
through adaptive Retrieval-Augmented Genera-
tion (RAG) to ground neural reasoning in biologi-
cal axioms and reduce transcriptomic noise. It fur-
ther employs a dialectic verification process with
homogeneous rebuttal agents to audit and prune
reasoning paths, forming syllogistic derivation
trees that enforce logical consistency.Across large-
scale and cross-species benchmarks, MAT-Cell
significantly outperforms state-of-the-art (SOTA)
models and maintains robust per-formance in chal-
lenging scenarios where baselinemethods severely
degrade. Code is available at https://gith
ub.com/jiangliu91/MAT-Cell-A-Mul
ti-Agent-Tree-Structured-Reasoni
ng-Framework-for-Batch-Level-Sin
gle-Cell-Annotation.
1. Introduction
While single-cell RNA sequencing (scRNA-seq) (Lähne-
mann et al., 2020; Klein et al., 2015) has scaled to profile
1Westlake University, Hangzhou, China2Shenzhen Univer-
sity of Advanced Technology, Shenzhen, China3Center for Ar-
tificial Intelligence and Robotics, Hong Kong Institute of Sci-
ence and Innovation, Chinese Academy of Sciences, Hong Kong,
China4University Key Laboratory of Information and Commu-
nication on Security Backup and Recovery in Yunnan Minzu
University, Kunming, China. Correspondence to: Zelin Zang
<zangzelin@westlake.edu.cn>.millions of cells (Regev et al., 2017; Hao et al., 2024), the
fundamental challenge in computational biology has shifted
from data generation to automated cellular reasoning (Xiao
et al., 2024; Mao et al., 2025; Fang et al., 2025). Tissues are
not static catalogs of discrete types; they are dynamic con-
tinua governed by complex gene regulatory networks (Cui
et al., 2024; Trapnell, 2015; Wagner et al., 2016). Conse-
quently, interpreting cellular heterogeneity requires more
than pattern matching against a fixed reference; it demands
the ability to deduce cell identity from first principles, es-
pecially for rare, transitional, or disease-specific states that
defy rigid categorization.
The first failure mode stems from the “Reference
Trap” (Luecken et al., 2022; Wagner et al., 2016; Stuart
et al., 2019) inherent in supervised learning. Traditional
annotators like CellTypist (Dominguez Conde et al., 2022)
and scANVI (Xu et al., 2021) rely on embedding-based
correlation against static atlases. While effective for known
cell types, these methods operate under a closed-world as-
sumption. They fail to generalize toout-of-distribution
(OOD)states—such as transitional progenitors or disease-
specific subtypes—often forcing novel biological signals
into incorrect, pre-defined categories simply because they
lack the reasoning capacity to recognize “unknowns”.
The second failure mode is the “Signal-to-Noise Para-
dox” (Kalai et al., 2025; Bian et al., 2024; Scaife & Smith,
2018) plaguing generative AI. While Large Language Mod-
els (LLMs) offer promising zero-shot capabilities (Cui et al.,
2024; Valkanas et al., 2023), their general-purpose attention
mechanisms are ill-suited for raw transcriptomic profiles.
In single-cell data, biologically defining markers (e.g., tran-
scription factors) are often sparsely expressed, while house-
keeping genes (e.g.,MALAT1,ACTB) dominate the count
matrix. As illustrated in Figure 1A, standard LLMs get
“distracted” by this high-abundance noise (the “confounding
dominance” of housekeeping genes), leading tohallucina-
tions of plausibility: outputs that are textually coherent but
biologically ungrounded in the specific cellular context.
To bridge this gap, Figure 1B inspiration from the dual-
process theory of cognition (Kahneman, 2011; Bengio,
2019) to introduceMAT-Cell. Unlike standard models that
1arXiv:2604.06269v1  [q-bio.QM]  7 Apr 2026

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
A: System 1 - The Reference Trap & Noise Paradox
 The Signal-to-Noise Paradox The Reference T rap
Ubiquitous Housekeeping Genes
(e.q., MALA T1,MT -CO1 )
Ubiquitous
Housekeeping
Genes  (e.g.,
MALA T1)Lineage-
Speciﬁc
DECs
(e.g., CD3E )
Signal obscured by Genes  Fog    Hallucination & RigidityInput
（Novel
State）Fixed
Reference
Atlas
Old Label
(e.g.,"Unknown T cell")?Black
Box
Model
B: System 2 - MA T-Cell (Syllogistic Derivation)
Root:Final V eriﬁed Conclusion
(e.g.,"CD8+T emra")Inductive Anchoring Dialectic V eriﬁcatior
V alidated
Statistical
Sieve  (Filter)
Coral Fog blocked;
Sharp Teal Signal passes Signal-Anchored ProofLeaves: Observed Evidence
( Pmin, e.g. “CD3E+")Syllogistic
Inference Steps
( Pmaj+ Pmin→)
Figure 1.System 1 vs. System 2 in Cellular Reasoning.(A) Stan-
dard LLMs suffer from the “Signal-to-Noise Paradox” (System
1), where attention mechanisms are distracted by highly expressed
housekeeping genes (Coral Fog), leading to hallucinations. (B)
MAT-Cellestablishes a System 2 paradigm viaInductive An-
choring, which grounds reasoning solely in statistically validated
markers (Teal DEGs), andDialectic Verification, constructing a
verifiable Syllogistic Derivation Tree (SDT).
operate in aSystem 1fashion (fast, intuitive pattern match-
ing prone to bias), MAT-Cell enforces anLLM-driven
Neuro-Symbolic System 2paradigm (Trinh et al., 2024;
Gao et al., 2023; Yao et al., 2022). It reformulates annota-
tion asexplicit proof construction, effectively preventing
the hallucination of plausibility.
To address both the Signal-to-Noise Paradox and the Ref-
erence Trap, we introduceMAT-Cell, which integratesIn-
ductive Anchoring via Symbolic Constraint Injection
with aMulti-agent Dialectic Verification Protocol. Rather
than feeding the full noisy transcriptome into an LLM, In-
ductive Anchoring retrieves canonical marker axioms to
explicitly constrain the neural search space, forcing reason-
ing to operate solely on the intersection between observed
expression evidence and validated biological knowledge,
thereby suppressing the confounding dominance of house-
keeping genes. Building upon this grounded representation,
MAT-Cell employs a collaborative council of specialized
agents—including aSolve Agent,Rebuttal Agent, andDeci-
sion Agent—to iteratively construct and audit aSyllogisticDerivation Tree (SDT)(Smith et al., 1989; Khemlani &
Johnson-Laird, 2012). This dialectic process emulates sci-
entific peer review: hypotheses are proposed, challenged,
and refined through structured debate, with convergence
determined by minimizing dialectic divergence rather than
maximizing probabilistic confidence, ultimately yielding a
transparent, verifiable “white-box” proof path. Our contri-
butions are threefold.(1) Neuro-Symbolic Paradigm: We
propose the first framework to reformulate single-cell analy-
sis as a neuro-symbolic proof construction process, unifying
neural flexibility with symbolic rigor.(2) Methodological
Innovation: We introduce Symbolic Constraint Injection to
ground LLM reasoning and Orthogonal Dialectic Roles to
eliminate hallucination through adversarial verification.(3)
SOTA Performance: Extensive experiments demonstrate
that MAT-Cell significantly outperforms scPilot and super-
vised baselines, providing fully transparent, verifiable proof
trees for every decision.
2. Related Work
Traditional automated cell type annotation methods primar-
ily rely on supervised classification or latent space align-
ment against curated reference atlases. Methods such as
SingleR (Aran et al., 2019), CellTypist (Dominguez Conde
et al., 2022), and scANVI (Xu et al., 2021) formulate an-
notation as a statistical correlation problem within a closed
manifold, enabling reliable identification of common cell
states but fundamentally operating as fast, pattern-matching
“System 1” approaches. As a result, they suffer from the
Reference Trap: disease-specific subtypes or transitional
states outside the reference manifold are often force-aligned
to the nearest known cluster with high confidence. Re-
cent foundation models, including scGPT (Cui et al., 2024),
Geneformer (Valkanas et al., 2023), and scFoundation (Hao
et al., 2024), scale annotation via Transformer architectures,
yet encounter aSignal-to-Noise Paradox, where highly ex-
pressed housekeeping genes dominate attention and induce
biologically plausible but incorrect annotations. Critically,
these models remain probabilistic predictors and lack mech-
anisms for biological or logical verification.
To address these limitations, recent works have explored
agentic frameworks and reinforcement learning. CellA-
gent (Xiao et al., 2024) and scAgent (Mao et al., 2025)
primarily orchestrate external bioinformatics tools, while
CellDuality (Anonymous, 2026) applies task-specific rein-
forcement learning; however, both paradigms lack explicit
and generalizable reasoning. In contrast, advances in LLM
reasoning—including Chain-of-Thought (Wei et al., 2022;
Kojima et al., 2022), Tree of Thoughts (Yao et al., 2023),
Self-Consistency (Wang et al., 2022), neuro-symbolic sys-
tems such as AlphaGeometry (Trinh et al., 2024) and
Logic-LM (Pan et al., 2023), and multi-agent debate frame-
2

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
works (Liang et al., 2023; Du et al., 2023; Li et al.,
2023)—demonstrate that structured reasoning and dialectic
verification substantially reduce hallucinations. MAT-Cell
bridges these paradigms to transcriptomic analysis by re-
formulating annotation asdeductive biological reasoning,
enforcing a strict syllogistic structure (Biological Axiom +
Expression Evidence →Conclusion) through Inductive An-
choring and Dialectic Verification, thereby enabling trans-
parent and generalizable annotation beyond reference at-
lases.
3. Methodology
We propose MAT-Cell, anLLM-Centric Neuro-Symbolic
Reasoning Frameworkthat reformulates single-cell anno-
tation from a statistical classification task into a constructive
logical proof. Unlike standard LLMs that rely on implicit
pattern matching (Yang et al., 2022; 2024) or pure sym-
bolic systems that lack flexibility (Chen & Zou, 2024; 2025;
Dhuliawala et al., 2024), MAT-Cell unifies neural reasoning
with symbolic constraints by leveragingunsupervised ag-
gregationto distill robust biological signals. As illustrated
in Figure 2, the pipeline begins with cluster-level feature
extraction, followed by three core stages: (1)Inductive
Anchoringto inject symbolic constraints, (2)Dialectic Ver-
ificationto build a bottom-up proof tree via agent roles, and
(3)Contextual Synthesisfor final adjudication. The com-
plete inference procedure is summarized in Algorithm 1.
3.1. Problem Formulation: Neuro-Symbolic Inference
LetX∈RN×Gbe a single-cell gene expression matrix.
Standard neural approaches fθ:X→ Y typically oper-
ate at the individual cell level but frequently suffer from
theSignal-to-Noise Paradoxcaused by stochastic drop-out
and transcriptional bursting. Conversely, traditional sym-
bolic systems fall into theReference Trapdue to their re-
liance on fixedEmbedding Geometry, which forces out-
of-distribution (OOD) states into known manifolds. We
propose to bridge this gap by leveraging the composability
ofLogical Rulesthrough explicit derivation.
To ensure robust inference, we first partition the raw data
space into MMeta-cells(or statistical clusters) C=
{c1, . . . , c M}using an unsupervised function Φ :X→ C ,
where M≪N . For each cluster ci, we identify a robust
set of Highly Expressed Genes (HEGs) and Differentially
Expressed Genes (DEGs) to represent its biological identity.
Letxibe the structured evidence set derived from these
statistically validated genes.
We reformulate the inference task as finding a label yifor
each cluster by constructing aSyllogistic Derivation Tree
(SDT) Tias its proof. The objective is to maximize the
posterior probability of the proof tree given cluster-levelevidence and symbolic constraints:
T∗
i= arg max
TP(T |x i,Kret,Rlogic)(1)
where Kretdenotes retrieved biological axioms and Rlogic
represents hard logic rules. This formulation shifts single-
cell analysis from “label guessing for noisy points” to “logi-
cal deduction for statistical clusters.”
3.2.Stage 1: Inductive Anchoring (Symbolic Constraint
Injection)
To bridge the gap between continuous expression data
and discrete biological logic, we employ aRetrieval-
Augmented Inductive Anchoringstrategy. This stage
transforms the raw input into a semi-symbolic represen-
tation that grounds subsequent reasoning.
Neuro-Symbolic Input Card.We align neural observa-
tions (Top genes and differentially expressed genes) with
the retrieved symbolic constraints to construct aNeuro-
Symbolic Input Card xcard
i. This card does not directly
encode a syllogistic conclusion. Instead, it provides the
structuredpremise materialsrequired for subsequent syl-
logistic construction, namely candidate biological axioms,
expression evidence, and contextual information. These
elements are later organized by the reasoning agents into ex-
plicit deductive forms ofBiological Axiom (Major Premise)
+ Expression Evidence (Minor Premise)→Conclusion.
Formally, the input card is defined as:
xcard
i =
G(i)
obs∩Span(K ret),K ret,ctx i
(2)
where G(i)
obsdenotes the set of observed genes for cluster i,
Span(K ret)represents the union of marker genes across all
retrieved candidate types, Kretdenotes the retrieved sym-
bolic knowledge block containing candidate cell types and
their marker axioms, and ctxiencodes auxiliary contextual
information for clusteri.
3.3. Stage 2: Dialectic Verification (Proof Tree
Construction)
Solve Agent Initialization.To establish a constrained
search space at the onset of inference, we introduce aSolve
Agent (SA). Given the neuro-symbolic input card xcard
i
from Stage 1, the Solve Agent performs inductive reasoning
to generate a candidate cell type set:
Ccand
i=SA(xcard
i)(3)
where Ccand
i⊆ Y and|Ccand
i| ≪ |Y| . This candidate set
explicitly constrains the subsequent reasoning space.
Homogeneous Rebuttal Agents.The Council of Verifiers
consists of KhomogeneousRebuttal Agents (RAs). Con-
ditioned on the same input card and candidate space, each
3

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Stage 1: Inductive Anchoring
(Solving the Signal-to-Noise Paradox)Stage 2: Dialectic Verification
(Escaping the Reference Trap)Stage 3: Contextual Synthesis
(Hybrid Assembly)
Raw Reads
(FASTQ)
Count
Matrix
Clustering
（Louvain ）
DEG Analysis
(Wilcoxon)
Query   Local   Updatemiss
Teacher      
Upstr eam Pr ocessingInput Card (    )
Top-25 Genes
Cluster  DEGs
Metadata       
Marker
SignatureAdaptive RAG
Hybrid Retrieval LogicTree Node Definition (             )
         (Rule):        T-cells express  CD3E.
         (Evidence): Cell i has CD3E +, CD4- .
     (Scope):          {CD8+ T, NK, ...}
Consensus Logic & Feedback SDT  Topology Visualization (     )Leaf Nodes(    )Inverted TreeDecision Agent
Output: Cell Identity
Final Annotation (Root)
Focus Set 
 Ʃ  |     > 0  Parallel Rebuttal Agents
Agent 3
   ( fdiag)  Agent 1
  (fdiag)  Agent 2
   ( fdiag)  
Re-evaluation
      Solve Agent
         
Layer 1
Layer 2
......
Figure 2.The MAT-Cell Framework. (Stage 1)Inductive Anchoringfilters housekeeping noise by constructing a Syllogistic Input
Cardxifrom statistically validated DEGs. (Stage 2)Dialectic Verificationemploys a multi-agent debate to construct a Syllogistic
Derivation Tree (SDT), minimizing Semantic Divergence Score δconto ensure logical consistency. (Stage 3)Contextual Synthesis
resolves ambiguity via a Decision Agent, yielding a hybrid assembly of auto-verified and adjudicated annotations.
Rebuttal Agent independently constructs a reasoning hy-
pothesis and outputs a tentative conclusion:
y(k)
i=RA k(xcard
i,Ccand
i), k= 1, . . . , K(4)
Dialectic Interaction and Self-Correction.At each dialec-
tic round t, Rebuttal Agents engage in peer-to-peer rebuttal
by inspecting conflicting hypotheses. Any unstable reason-
ing path is revised via self-correction, yielding an updated
conclusion:
y(k)
i,t+1 =RArevise
k 
y(k)
i,t,{y(j)
i,t}j̸=k
(5)
Consensus-Based Convergence.The construction of the
Syllogistic Derivation Tree (SDT) Tiproceeds iteratively
until the council reaches logical consensus. Convergence
is determined by strict consensus consistency rather than
probabilistic scoring:
Condition converge :y(1)=y(2)=···=y(K)(6)
If the condition is satisfied, the SDT converges to a stable
root node yiand inference terminates. Otherwise, the branch
is deemed unstable and further dialectic rounds or pruning
are triggered.
3.4.Stage 3: Contextual Synthesis (Proof Adjudication)
The final annotation is typically derived from the root node
of the Syllogistic Derivation Tree (SDT). However, for com-plex boundary cases, the council may fail to reach consensus
within the predefined dialectic rounds. We therefore define
aFocus SetI focus to identify non-converged clusters:
Ifocus ={i|¬Condition converge (Ti)}(7)
whereT idenotes the SDT constructed for clusteri.
For clusters in Ifocus , the reasoning process is escalated
to aDecision Agent (DA). Acting as a senior adjudicator,
the DA reviews the complete proof tree history, including
conflicting branches proposed by different agent roles, and
issues a final verdict:
y∗
i=DA(T i), i∈ I focus (8)
The final output is assembled in a hybrid manner:
yi=(
ycons
i, i /∈ I focus
y∗
i, i∈ I focus(9)
where ycons
idenotes the consensus result obtained through
dialectic verification. This hybrid strategy ensures scalabil-
ity while enabling rigorous handling of ambiguous and hard
boundary cases.
4

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 1.Main results under the open candidate (no oracle labels) setting. This setting follows the scPilot protocol and simulates realistic
annotation scenarios where no candidate cell-type labels are provideda priori. All methods must infer cell types directly from cluster-level
marker information without access to oracle candidates. Results are reported as accuracy (mean ±std) across five scRNA-seq datasets
(PBMC3K, Liver, Retina, Brain, and Heart). Different language model backbones are listed to reflect model-dependent variations. For
MAT-Cell variants, the suffix “-c” denotes the use of retrieval-augmented grounding; Qwen-based “-c” models are distilled within the
same backbone family, while other “-c” variants are distilled from Gemini-3. The best and second-best results are highlighted inboldand
underlined , respectively.
Method Model PBMC3K Liver Retina Brain Heart Avg Acc.
CellTypist – 0.464 0.563 0.388 0.242 0.690 0.469
CellMarker2.0 – 0.304 0.250 0.632 0.625 0.267 0.416
DirectQwen3-30B 0.675±0.061 0.444±0.062 0.747±0.021 0.231±0.097 0.246±0.123 0.469
O1 0.667±0.005 0.560±0.001 0.474±0.002 0.296±0.026 0.338±0.092 0.467
Qwen3-14B 0.674±0.018 0.467±0.090 0.747±0.077 0.119±0.098 0.123±0.038 0.426
GPT-4o 0.604±0.005 0.440±0.002 0.439±0.002 0.356±0.018 0.231±0.169 0.414
Gemini 2.5 Pro 0.583±0.001 0.494±0.007 0.491±0.000 0.215±0.082 0.185±0.062 0.394
scPilotGPT-4o 0.646±0.017 0.512±0.002 0.675±0.011 0.452±0.226 0.308±0.069 0.519
O1 0.792±0.005 0.518±0.001 0.728±0.007 0.115±0.230 0.354±0.158 0.501
Qwen3-30B 0.750±0.137 0.437±0.054 0.737±0.058 0.215±0.102 0.200±0.062 0.468
Gemini 2.5 Pro 0.708±0.021 0.488±0.0010.763±0.0000.148±0.052 0.169±0.113 0.455
Qwen3-14B 0.725±0.050 0.422±0.055 0.747±0.021 0.022±0.018 0.123±0.038 0.408
CoTGemini 2.5 Pro 0.625±0.044 0.782±0.008 0.495±0.012 0.669±0.018 0.640±0.028 0.642
O1 0.675±0.052 0.704±0.013 0.490±0.014 0.669±0.042 0.613±0.018 0.630
Qwen3-14B 0.578±0.031 0.648±0.085 0.663±0.073 0.640±0.060 0.511±0.105 0.608
GPT-4o 0.688±0.000 0.804±0.025 0.442±0.022 0.455±0.020 0.580±0.038 0.594
Qwen3-30B 0.575±0.025 0.781±0.044 0.550±0.178 0.577±0.025 0.458±0.043 0.588
MAT-Cell
(no RAG)Gemini3 0.625±0.028 0.607±0.042 0.500±0.0140.894±0.0570.633±0.024 0.652
Deepseek-v3 0.650±0.035 0.800±0.050 0.516±0.014 0.656±0.000 0.627±0.028 0.650
Llama-3-70b 0.613±0.069 0.778±0.043 0.621±0.040 0.656±0.066 0.513±0.030 0.636
Qwen3-14B 0.575±0.082 0.659±0.038 0.495±0.057 0.513±0.087 0.620±0.030 0.572
Qwen3-30B 0.638±0.052 0.791±0.020 0.474±0.046 0.469±0.058 0.474±0.046 0.569
MAT-Cell
(use RAG)Qwen3-30B0.800±0.028 0.811±0.0480.632±0.019 0.719±0.0620.813±0.038 0.755↑+45.5%
Llama-3-70b 0.725±0.034 0.800±0.024 0.626±0.071 0.738±0.028 0.753±0.038 0.728
Gemini3 0.750±0.000 0.804±0.021 0.505±0.035 0.656±0.039 0.780±0.019 0.699
Deepseek-v3 0.638±0.028 0.685±0.019 0.521±0.029 0.763±0.065 0.620±0.018 0.645
Qwen3-30B-c 0.613±0.028 0.796±0.019 0.510±0.025 0.444±0.014 0.600±0.000 0.593
4. Experiments
4.1. Experimental Setup
Task Definition and Evaluation Paradigms.We evaluate
MAT-Cell under two complementary settings. TheOpen
Candidate Setting, inspired by the evaluation paradigm used
in scPilot, provides no prior cell-type labels, requiring the
system to perform clustering, candidate retrieval, and joint
annotation to simulate realistic discovery. TheOracle Can-
didate Setting, referencing the controlled protocol of Cell-
o1, supplies ground-truth labels as the candidate pool, iso-
lating the core reasoning capability by removing uncertainty
from candidate search.
Datasets.For generalization evaluation, we use five
datasets: PBMC3K (10x Genomics, 2017), Liver (Liang
et al., 2022), and Retina (Menon et al., 2019) (also used in
scPilot), together with Brain and Heart introduced in this
work to test robustness under higher structural complex-
ity. For cross-species analysis, we construct a benchmark
from the CellxGene database constructed with reference to
the Cell-o1 pipeline, covering Human, Mouse, and Mon-key datasets, each containing 2,400 independent batch-level
instances.
Input Evidence and Signal-to-Noise Analysis.Cluster-
level summaries are converted into Syllogistic Input Cards
to constrain the reasoning process. To analyze the Signal-
to-Noise Paradox, we define three input views over the
2,400 instances:Both(HEGs + DEGs, where HEGs denote
the top-25 most highly expressed genes),DEG-only(using
statistically significant differentially expressed genes only),
andTop-only(using the top-25 expressed genes alone). This
design evaluates the model’s ability to distinguish biological
signal from transcriptional noise.
Evaluation and Implementation Details.The Open Set-
ting adopts the manual grading protocol of scPilot (1 / 0.5 /
0), while the Oracle Setting uses exact string matching as
in Cell-o1. In our core configuration, MAT-Cell employs a
Council of Verifiers with three Rebuttal Agents that itera-
tively refine the syllogistic derivation tree. The maximum
reasoning depth is set to Tmax= 3, with temperature fixed
at 0.7. Results are reported as mean and standard deviation
over multiple independent runs.
5

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 2.Cross-species performance comparison under the open candidate setting. This evaluation follows the Cell-o1 protocol and assesses
generalization across three species (Human, Mouse, and Monkey). For each species, results are reported under three input configurations:
Both(HEGs + DEGs),DEG-only, andHEG-only, reflecting different signal-to-noise conditions. All values are reported as mean ±
standard deviation (rounded to three decimals) over multiple runs. Different backbone models are listed to highlight model-dependent
variations. The best and second-best results in each column are highlighted inboldand underlined , respectively. For MAT-Cell (use
RAG), distillation details are omitted from the table for clarity: models with the suffix “-c” denote RAG-specific students distilled within
the same backbone family, while the remaining RAG variants are distilled from Gemini-3.
Method ModelHuman Mouse Monkey
Both DEG HEG Both DEG HEG Both DEG HEG
Cell-o1 Qwen2.5-7B0.429±0.012 0.409±0.015 0.243±0.011 0.390±0.016 0.394±0.017 0.232±0.008 0.695±0.018 0.685±0.019 0.503±0.012
DirectQwen3-14b0.291±0.008 0.409±0.009 0.164±0.004 0.308±0.010 0.348±0.010 0.162±0.004 0.542±0.005 0.580±0.006 0.302±0.006
Qwen3-30B0.340±0.008 0.450±0.013 0.165±0.008 0.347±0.010 0.387±0.008 0.178±0.005 0.588±0.010 0.644±0.009 0.331±0.010
Llama3.1-70B0.191±0.006 0.283±0.009 0.117±0.003 0.200±0.008 0.290±0.017 0.126±0.007 0.376±0.005 0.471±0.008 0.243±0.007
Deepseek-v30.551±0.007 0.632±0.012 0.189±0.005 0.541±0.015 0.567±0.010 0.257±0.005 0.376±0.005 0.471±0.008 0.243±0.007
Gemini-2.5-flash0.641±0.008 0.709±0.009 0.186±0.006 0.634±0.012 0.659±0.015 0.306±0.011 0.859±0.009 0.859±0.009 0.527±0.009
GPT4.10.663±0.015 0.733±0.007 0.204±0.008 0.655±0.008 0.649±0.013 0.335±0.005 0.853±0.000 0.864±0.005 0.556±0.015
CoTQwen3-14B0.487±0.005 0.618±0.017 0.216±0.004 0.479±0.008 0.534±0.015 0.274±0.009 0.672±0.008 0.720±0.006 0.404±0.010
Qwen3-30B0.555±0.003 0.627±0.005 0.213±0.004 0.476±0.022 0.521±0.007 0.256±0.004 0.649±0.011 0.716±0.011 0.357±0.002
Llama3.1-70B0.290±0.015 0.486±0.009 0.117±0.005 0.321±0.007 0.428±0.012 0.146±0.003 0.619±0.012 0.651±0.009 0.235±0.009
Deepseek-v30.395±0.215 0.745±0.014 0.234±0.004 0.650±0.004 0.626±0.063 0.373±0.004 0.855±0.003 0.855±0.001 0.537±0.005
Gemini-2.5-flash0.649±0.003 0.719±0.008 0.238±0.005 0.594±0.003 0.629±0.001 0.341±0.001 0.834±0.003 0.845±0.005 0.498±0.008
GPT4.10.672±0.010 0.745±0.005 0.176±0.006 0.663±0.004 0.681±0.008 0.332±0.005 0.867±0.010 0.868±0.003 0.520±0.005
MAT-Cell
(no RAG)Qwen3-14B0.367±0.004 0.548±0.002 0.184±0.002 0.383±0.014 0.467±0.006 0.200±0.005 0.643±0.004 0.699±0.004 0.389±0.006
Qwen3-30B0.420±0.011 0.564±0.009 0.187±0.024 0.688±0.007 0.721±0.007 0.403±0.011 0.461±0.009 0.503±0.009 0.232±0.009
Llama3.1-70B0.337±0.015 0.513±0.008 0.164±0.002 0.382±0.003 0.467±0.016 0.197±0.006 0.662±0.012 0.705±0.005 0.362±0.014
Deepseek-v30.645±0.004 0.739±0.010 0.257±0.014 0.627±0.008 0.668±0.013 0.381±0.002 0.841±0.010 0.850±0.001 0.564±0.001
Gemini-2.5-flash0.590±0.011 0.698±0.014 0.235±0.009 0.639±0.016 0.659±0.013 0.341±0.013 0.850±0.012 0.843±0.008 0.530±0.013
MAT-Cell
(use RAG)Qwen3-14B-c0.513±0.000 0.619±0.007 0.214±0.000 0.439±0.009 0.494±0.008 0.234±0.006 0.710±0.003 0.745±0.004 0.432±0.012
Qwen3-30B-c0.518±0.006 0.626±0.014 0.203±0.012 0.487±0.003 0.559±0.014 0.245±0.005 0.733±0.007 0.745±0.009 0.414±0.000
Qwen3-14B0.633±0.0070.752±0.002 0.261±0.0010.800±0.002 0.826±0.007 0.509±0.0060.624±0.004 0.687±0.005 0.363±0.009
Qwen3-30B0.664±0.005 0.764±0.011 0.282±0.009 0.808±0.0030.825±0.007 0.499±0.007 0.700±0.007 0.759±0.006 0.395±0.009
Gemini-2.5-flash0.796±0.009 0.814±0.003 0.295±0.0100.746±0.006 0.767±0.008 0.405±0.0070.886±0.003 0.877±0.004 0.575±0.020
4.2.Main Results: Batch-Level Annotation under Open
Candidate Setting
Table 1 summarizes the quantitative results of MAT-Cell
across five benchmark datasets under the open candidate
setting. With retrieval augmentation (RAG), MAT-Cell con-
sistently outperforms traditional methods, direct prompting,
and existing agentic baselines. The Qwen3-30B configura-
tion achieves an average accuracy of 0.7550, corresponding
to a 45.5% relative improvement over the strongest agentic
baseline, scPilot (GPT-4o, 0.5186).
The advantage is particularly pronounced on the structurally
complex Brain dataset, where baselines lacking explicit logi-
cal constraints (e.g., scPilot O1) degrade to 0.1150 accuracy.
In contrast, MAT-Cell maintains an accuracy of 0.7190 by
leveraging the Syllogistic Derivation Tree (SDT), demon-
strating the effectiveness of neuro-symbolic reasoning in
suppressing logical hallucinations.
Further analysis shows that the Qwen3-30B model distilled
from Gemini3 (0.7550) outperforms its teacher, Gemini3-
rag (0.6990), highlighting the effectiveness of task-specific
logic distillation. In addition, introducing RAG elevates
Qwen3-30B performance from 0.5692 (no-RAG) to 0.7550,
validating the critical role of external symbolic constraints
in mitigating the signal-to-noise paradox. Overall, by in-tegrating multi-agent debate with explicit logical auditing,
MAT-Cell transforms batch-level annotation from proba-
bilistic matching into verifiable deductive reasoning.
4.3. Controlled Analysis: Reasoning Robustness and
Signal Quality under Oracle Setting
Table 2 reports the evaluation results under theOracle Can-
didate Setting, covering 7,200 independent batch-level in-
stances across three species: human, mouse, and monkey.
Under controlled candidate conditions, MAT-Cell consis-
tently outperforms Cell-o1 and other comparative models
in both accuracy and stability. By decoupling candidate
generation from downstream decision-making, this setting
effectively isolates and validates the core capability of the
neuro-symbolic reasoning engine in synthesizing biological
evidence.
Comparisons across different input views further reveal a
strong dependency on signal quality, highlighting theSignal-
to-Noise Paradoxin single-cell reasoning. When the input
is restricted to the top-25 highly expressed genes (Top-25
HEGs), performance across all models drops substantially;
for example, MAT-Cell achieves an accuracy of only 0.498
on the mouse dataset. In contrast, using aDEG-onlyview
elevates the accuracy to 0.825 for mouse and 0.764 for hu-
6

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Algorithm 1MAT-Cell Inference via Dialectic Reasoning
Require: Meta-cell xi, knowledge base K, agents
{SA,RA 1, . . . ,RA K,DA}, max roundsT
Ensure:Predicted labely i
1:xcard
i←InductiveAnchoring(x i,K)
2:Ccand
i←SA(xcard
i)
3: Initialize proof treeT i← ∅
4: Initialize focus setI focus← ∅
5:fort= 1toTdo
6:fork= 1toKdo
7:y(k)←RA k(xcard
i,Ccand
i,Ti)
8:end for
9: Agents exchange rebuttals and revise unstable hypotheses
10:ify(1)=···=y(K)then
11:y i←y(1)
12:returny i
13:end if
14:T i← T i∪ {{y(k)}K
k=1}
15:end for
16:I focus← {i| T inot converged}
17:y i←DA(T i)
18:returny i
man, markedly outperforming the 0.282 accuracy observed
under the HEG-only view on the human dataset. These
results indicate that highly expressed genes are frequently
confounded by non-specific transcriptional noise, whereas
statistically significant DEGs provide more reliable and dis-
criminative inductive anchors for logical reasoning. In terms
of cross-species consistency, MAT-Cell also demonstrates
clear advantages over baseline methods.
4.4. Qualitative Analysis: Syllogistic Derivation Tree
Visualization
To understand why MAT-Cell succeeds where direct prompt-
ing fails, we qualitatively analyze the Syllogistic Derivation
Trees (SDTs) produced during inference. As shown in Fig. 3,
MAT-Cell suppresses frequent but weakly discriminative sig-
nals through dialectic verification and bases decisions on dis-
criminative DEGs, whereas direct prompting tends to yield
coarse or conflated annotations in expression-overlapping
batches.
In the illustrated batch, multiple cells share immune-related
markers but differ in endothelial- and stromal-associated
DEGs. Through multi-agent debate, MAT-Cell identifies
and corrects these inconsistencies. The resulting SDT ex-
plicitly encodes logical decision rules, e.g.,
(
(CLDN5+∧ESM1+)⇒endothelial cell,
(CCL21+)⇒lymphatic endothelial cell.
This structure makes the decision process traceable and
directly grounded in discriminative biological evidence.
Error Analysis.We manually inspected 50 incorrect
batches from MAT-Cell on the Mouse dataset (Fig. 3, bot-tom). The dominant failure mode (64%) arises frommarker
ambiguity, where closely related cell types share markers
or critical discriminative genes are absent from the knowl-
edge base K. In such cases, MAT-Cell favors conservative,
under-specified predictions over unsupported hallucinations.
A secondary failure mode (22%) isconvergence timeout,
which occurs in highly heterogeneous batches when dialec-
tic verification fails to converge within the maximum depth
Tmax= 3.
4.5. Ablation and Sensitivity Analysis
To systematically characterize the performance sources of
MAT-Cell, we conduct a unified analysis that combines
system-level ablation with sensitivity evaluation of the di-
alectic verification protocol. This analysis examines three
key factors: input evidence quality, framework efficacy (in-
cluding retrieval augmentation), and the stability of critical
hyperparameters governing multi-agent reasoning (Table 3,
Fig. 4).
Input Evidence Quality and Framework Effect.As
shown in Table 3, relying solely on the top-25 highly ex-
pressed genes (M0, M2) results in poor average accuracy
(22%–27%), indicating that highly expressed signals are of-
ten confounded by non-specific transcriptional noise. Intro-
ducing differentially expressed genes (DEGs) as input (M1,
M3) leads to a substantial performance improvement (e.g.,
M3 reaches 59.6%), confirming the critical role ofInductive
Anchoringin capturing highly discriminative biological sig-
nals. Beyond input quality, the MAT-Cell framework itself
provides consistent gains. Compared to the direct LLM
baseline (M1), MAT-Cell (M4) improves average accuracy
from 0.494 to 0.783 (+28.9%). More importantly, it signif-
icantly enhances cross-species stability: the performance
gap is reduced from 0.257 to 0.066, and the robustness index
(RI) increases to 0.97, indicating a transition from unstable
probabilistic prediction to robust logical deduction.
Contribution of Retrieval-Augmented Generation
(RAG).Table 3 further shows that removing the RAG mod-
ule (M3) degrades accuracy to 0.596 and substantially wors-
ens stability metrics (Gap = 0.218). These results demon-
strate that the external knowledge base ( Kret) provides es-
sential biological axiomatic constraints, serving as an ef-
fective bridge between data-driven signals and symbolic
reasoning, and suppressing hallucinations under heteroge-
neous conditions.
Sensitivity to Council Scale and Dialectic Depth.As il-
lustrated in Fig. 4a and Fig. 4b, the dialectic verification
protocol exhibits a non-monotonic dependence on both the
council scale Kand the dialectic depth T. Specifically,
reasoning accuracy increases with Kand peaks at K= 3
(0.656 on the Brain dataset with DeepSeek-v3), while larger
councils ( K= 4 ) suffer from logical deadlock due to ex-
7

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Dialectic Verification & Syllogistic Derivation (SDT) for CorrectionMAT-Cell: Dialectic Verification, Syllogistic Derivation, and Error Analysis for Cell Type Classification
Corrected Classification
Richer Biological Knowledge Bases (K)
Stronger Backbone Models
Improved Reasoning StabilityError Analysis (50Incorrect Batches) & Future DirectionsLymphatic 
Endothelial Cell
(Correct)
Other Endothelial 
Subtype 
(Requires More DEGs)(CLDNS+^ ESM1+)
Endothelial Cell HypothesisStart: Cell Profile
SDTSyllogistic Derivation Tree (SDT)
ConstructionMAT-Cell Dialectic Verification
(Role-based Debate)Complex Cell Batch
(12 Cells, Overlapping Expression)Gene GeneGene profiles with other 
shared markers (e.g., immune)
Distinct DEGs (e.g., CCL21)Rebuttal 
Agent 1Rebuttal 
Agent 2 Agent 2: Rejects Coarse 
“Endothelial”Label due to 
Spectific Marker AhsenceAgent 1: Flags CCL21+, 
ldentifies Lymphatic 
Endothelial Subtype
(CCL21+)
(No CCL21+)  
 
Marker AmbiguitMacr ophagesStromal
Dendritic
CellsLymphatic
Endothelial
(CCL21+)
y 
MacrophageFuture Improvements
Cd68+ OR
Dendritic Cell?
(Ambiguous)
Fine-grained Elimination 
Constraintsin DebateFailure ModesBreakdown
Marker Ambiguity 
(e.g., Cd68 shared by 
Macrophages & DCs)
Convergence Timeout 
(Highly Heterogeneous Batches, >12 cells)
Weak/Conflicting DEG Evidence
 Blood
Endothelial
22%
14%64%
Figure 3.Qualitative SDT visualization and error analysis. Top:An illustrative batch where dialectic verification flags inconsistent
coarse labels and the SDT refines the decision using discriminative DEGs (e.g.,CCL21) to recover the lymphatic endothelial subtype.
Bottom:Failure-mode breakdown on 50 incorrect batches and a summary of future directions to improve robustness.
cessive mutual questioning and over-pruning of valid proof
paths. Similarly, accuracy follows an inverted-U trend with
respect to T, reaching an optimum at T= 3 (0.800 on the
Liver dataset) before declining at T= 4 due to diminishing
returns and over-correction. Taken together, these results
identify K= 3 andT= 3 as a stable equilibrium, balanc-
ing sufficient cross-verification with efficient convergence
for reliable multi-agent reasoning in MAT-Cell.
Table 3.System-level ablation on evidence and grounding mech-
anisms.All variants use the same Qwen3-30B backbone. We ab-
late discriminative biological signals (DEG) and retrieval ground-
ing (RAG) in MAT-Cell. H/M/Mk denote Human, Mouse, and
Monkey accuracy; Gap and RI measure cross-species stability.
ID Config H M Mk Gap↓RI↑
M0 Direct+HEG 0.165 0.178 0.331 0.166 0.73
M1 Direct+DEG 0.450 0.387 0.644 0.257 0.78
M2 MAT+HEG 0.187 0.403 0.232 0.216 0.68
M3 MA +DEG 0.564 0.721 0.503 0.218 0.84
M4MAT+RAG
+ DEG0.764 0.825 0.759 0.066 0.97
5. Conclusion
MAT-Cell introduces a neuro-symbolic paradigm that trans-
forms single-cell annotation into rigorous logical proofs.
Through dialectic verification by rebuttal agents, the sys-
tem generates auditable Syllogistic Derivation Trees (SDTs)
that enhance both accuracy and transparency. By leverag-
ing meta-cell anchoring to mitigate data noise and sparsity,this framework establishes a robust, efficient standard for
integrating biological priors into neural-based cell identity
decoding.
(a)Impact of Council ScaleK
(b)Impact of Dialectic DepthT
Figure 4.Sensitivity analysis of hyperparameters on Brain and
Liver datasets. (a) Impact of the number of agents Kon reasoning
accuracy. (b) Impact of the number of rounds Ton reasoning
stability.
8

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
References
10x Genomics. 10k peripheral blood mononuclear cells
(pbmcs) from a healthy donor. https://suppor
t.10xgenomics.com/single-cell-gene-e
xpression/datasets/1.1.0/pbmc3k , 2017.
Accessed: 2025-05-15.
Anonymous. Cellduality: Unlocking biological reasoning
in llms with self-supervised rlvr, 2026. Under review as
a conference paper at ICLR 2026.
Aran, D., Looney, A. P., Liu, L., Wu, E., Fong, V ., Hsu,
A., et al. Reference-based analysis of lung single-cell
sequencing reveals a transitional profibrotic macrophage.
Nature Immunology, 20(2):163–172, 2019.
Bengio, Y . From system 1 deep learning to system 2 deep
learning. InNeurIPS Keynote, 2019.
Bian, H., Chen, Y ., Luo, E., Wu, X., Hao, M., Wei, L.,
and Zhang, X. General-purpose pre-trained large cellular
models for single-cell transcriptomics.National Science
Review, 11(11):nwae340, 2024.
Chen, Y . and Zou, J. Genept: a simple but effective founda-
tion model for genes and cells built from chatgpt.bioRxiv,
pp. 2023–10, 2024.
Chen, Y . and Zou, J. Simple and effective embedding model
for single-cell biology built from chatgpt.Nature biomed-
ical engineering, 9(4):483–493, 2025.
Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N.,
and Wang, B. scgpt: toward building a foundation model
for single-cell multi-omics using generative ai.Nature
Methods, 21(7):1470–1480, 2024.
Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X.,
Celikyilmaz, A., and Weston, J. Chain-of-verification
reduces hallucination in large language models. InFind-
ings of the association for computational linguistics: ACL
2024, pp. 3563–3578, 2024.
Dominguez Conde, C., Xu, C., Jarvis, L. B., Rainbow, D. B.,
Wells, S., Gomes, T., Howlett, S., Suchanek, O., Polanski,
K., King, H., et al. Cross-tissue immune cell analysis
reveals tissue-specific features in humans.Science, 376
(6594):eabl5197, 2022.
Du, Y ., Li, S., Torralba, A., Tenenbaum, J. B., and Mor-
datch, I. Improving factuality and reasoning in lan-
guage models through multiagent debate.arXiv preprint
arXiv:2305.14325, 2023.
Fang, Y ., Jin, Q., Xiong, G., Jin, B., Zhong, X., Ouyang,
S., Zhang, A., Han, J., and Lu, Z. Cell-o1: Training llms
to solve single-cell reasoning puzzles with reinforcement
learning.arXiv preprint arXiv:2506.xxxxx, 2025.Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang,
Y ., Callan, J., and Neubig, G. Pal: Program-aided lan-
guage models. InInternational Conference on Machine
Learning, pp. 10764–10799. PMLR, 2023.
Hao, M., Gong, J., Zeng, X., Liu, C., Guo, Y ., Cheng,
X., et al. Large-scale foundation model on single-cell
transcriptomics.Nature Methods, 21:1–12, 2024.
Kahneman, D.Thinking, fast and slow. Farrar, Straus and
Giroux, 2011.
Kalai, A. T., Nachum, O., Vempala, S. S., and Zhang,
E. Why language models hallucinate.arXiv preprint
arXiv:2509.04664, 2025.
Khemlani, S. and Johnson-Laird, P. N. Theories of the
syllogism: A meta-analysis.Psychological bulletin, 138
(3):427, 2012.
Klein, A. M., Mazutis, L., Akartuna, I., Tallapragada, N.,
Veres, A., Li, V ., Peshkin, L., Weitz, D. A., and Kirschner,
M. W. Droplet barcoding for single-cell transcriptomics
applied to embryonic stem cells.Cell, 161(5):1187–1201,
2015.
Kojima, T., Gu, S. S., Reid, M., Matsuo, Y ., and Iwasawa,
Y . Large language models are zero-shot reasoners.Ad-
vances in neural information processing systems, 35:
22199–22213, 2022.
Lähnemann, D., Köster, J., Szczurek, E., McCarthy, D. J.,
Hicks, S. C., Robinson, M. D., Vallejos, C. A., Campbell,
K. R., Beerenwinkel, N., Mahfouz, A., et al. Eleven grand
challenges in single-cell data science.Genome biology,
21(1):31, 2020.
Li, G., Hammoud, H. A. A. K., Itani, H., Khizanishvili,
D., and Ghanem, B. Camel: Communicative agents for
"mind" exploration of large scale language model society.
Advances in Neural Information Processing Systems, 36:
51991–52008, 2023.
Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y ., Wang,
R., Yang, Y ., Tu, Z., and Shi, S. Encouraging divergent
thinking in large language models through multi-agent
debate.arXiv preprint arXiv:2305.19118, 2023.
Liang, Y ., Kaneko, K., Xin, B., Lee, J., Sun, X., Zhang, K.,
and Feng, G.-S. Temporal analyses of postnatal liver de-
velopment and maturation by single-cell transcriptomics.
Developmental cell, 57(3):398–414, 2022.
Luecken, M. D., Büttner, M., Chaichoompu, K., Danese,
A., Interlandi, M., Müller, M. F., Strobl, D. C., Zappia,
L., Dugas, M., Colomé-Tatché, M., et al. Benchmark-
ing atlas-level data integration in single-cell genomics.
Nature methods, 19(1):41–50, 2022.
9

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Mao, Y ., Mi, Y ., Liu, P., Zhang, M., Liu, H., and Gao, Y .
scagent: Universal single-cell annotation via a llm agent.
arXiv preprint arXiv:2504.04698, 2025.
Menon, M., Mohammadi, S., Davila-Velderrain, J., Goods,
B. A., Cadwell, T. D., Xing, Y ., Stemmer-Rachamimov,
A., Shalek, A. K., Love, J. C., Kellis, M., et al. Single-
cell transcriptomic atlas of the human retina identifies cell
types associated with age-related macular degeneration.
Nature communications, 10(1):4902, 2019.
Pan, L., Albalak, A., Wang, X., and Wang, W. Y . Logic-
lm: Empowering large language models with symbolic
solvers.arXiv preprint arXiv:2305.12295, 2023.
Regev, A., Teichmann, S. A., Lander, E. S., Amit, I., Benoist,
C., Birney, E., Bodenmiller, B., Campbell, P., Carninci,
P., Clatworthy, M., et al. The human cell atlas.Elife, 6:
e27041, 2017.
Scaife, A. A. and Smith, D. A signal-to-noise paradox in
climate science.npj Climate and Atmospheric Science, 1
(1):28, 2018.
Smith, R. et al.Prior analytics. Hackett Publishing, 1989.
Stuart, T., Butler, A., Hoffman, P., Hafemeister, C., Papalexi,
E., Mauck, W. M., Hao, Y ., Stoeckius, M., Smibert, P.,
and Satija, R. Comprehensive integration of single-cell
data.cell, 177(7):1888–1902, 2019.
Trapnell, C. Defining cell types and states with single-cell
genomics.Genome research, 25(10):1491–1498, 2015.
Trinh, T., Wu, Y ., Le, Q. V ., He, H., and Luong, T. Solv-
ing olympiad geometry without human demonstrations.
Nature, 625(7995):476–482, 2024.
Valkanas, C., Theis, F. J., et al. Transfer learning enables
predictions in network biology.Nature, 618:1–9, 2023.
Wagner, A., Regev, A., and Yosef, N. Revealing the vectors
of cellular identity with single-cell genomics.Nature
biotechnology, 34(11):1145–1160, 2016.
Wang, X., Wei, J., Schuurmans, D., Le, Q., and Chi, E.
Self-consistency improves chain of thought reasoning
in language models.arXiv preprint arXiv:2203.11171,
2022.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E.,
Le, Q., and Zhou, D. Chain-of-thought prompting elicits
reasoning in large language models.Advances in Neural
Information Processing Systems, 35:24824–24837, 2022.
Xiao, Y ., Liu, J., Zheng, Y ., Xie, X., Hao, J., and Li,
M. Cellagent: An llm-driven multi-agent framework
for automated single-cell data analysis.arXiv preprint
arXiv:2407.09811, 2024.Xu, C., Lopez, R., Mehlman, E., Regier, J., Jordan, M. I.,
and Yosef, N. Probabilistic harmonization and annotation
of single-cell transcriptomics data with deep generative
models.Molecular systems biology, 17(1):e9620, 2021.
Yang, F., Wang, W., Wang, F., Fang, Y ., Tang, D., Huang, J.,
Lu, H., and Yao, J. scbert as a large-scale pretrained deep
language model for cell type annotation of single-cell rna-
seq data.Nature Machine Intelligence, 4(10):852–866,
2022.
Yang, X., Liu, G., Feng, G., Bu, D., Wang, P., Jiang, J.,
Chen, S., Yang, Q., Miao, H., Zhang, Y ., et al. Genecom-
pass: deciphering universal gene regulatory mechanisms
with a knowledge-informed cross-species foundation
model.Cell Research, 34(12):830–845, 2024.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y . React: Synergizing reasoning and
acting in language models. InThe eleventh international
conference on learning representations, 2022.
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y .,
and Narasimhan, K. Tree of thoughts: Deliberate prob-
lem solving with large language models.arXiv preprint
arXiv:2305.10601, 2023.
10

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Contents
A Theoretical Analysis and Proofs 13
A.1 Proof of Error Bound for Dialectic Verification (Theorem 1) . . . . . . . . . . . . . . . . . . . . . . . . 13
A.2 Proof of Convergence for Syllogistic Derivation Tree (Theorem 2) . . . . . . . . . . . . . . . . . . . . . 13
A.3 Proof of OOD Identifiability (Theorem 3) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
B Prompt Templates for Tree-Based Multi-Agent Reasoning 15
B.1 Global System Prompt (Tree Reasoning Controller) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
B.2 SolveAgent Prompt (Tree Layer 0: Candidate Generation) . . . . . . . . . . . . . . . . . . . . . . . . . 16
B.3 RebuttalAgent Prompt (Tree Layer≥1: Per-Cell Adjudication) . . . . . . . . . . . . . . . . . . . . . . 18
B.4 DecisionAgent Prompt (Final Resolution) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
C Dataset Documentation 21
C.1 Dataset Summary Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.1 PBMC3K Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.2 Liver Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.3 Retina Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.4 Brain Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.5 Heart Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.6 Human Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.7 Mouse Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.1.8 Monkey Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.2 Top-25 Highly Expressed Genes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C.3 Differentially Expressed Genes (DEG) Criteria . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
C.4 Data Processing and Instance Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
D Detailed Methodology 24
D.1 Inductive Anchoring Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
D.1.1 Algorithmic Pseudocode . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
D.2 Dialectic Verification Mechanism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
D.2.1 Agent Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
D.2.2 Exact-Match Convergence Criterion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
D.3 Syllogistic Rule Set . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
D.4 Hyperparameter Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
D.5 Syllogistic Derivation Tree (SDT) Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
E Extended Ablation Studies 27
11

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
E.1 Impact of Gene Count . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
E.2 Gene Ordering Randomization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
E.3 Impact of RAG Gene Budget and Distillation Backbone . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
F Statistical Rigor and Confidence Intervals 29
F.1 Confidence Intervals (95%) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.1.1 Main Results Table with CI . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.2 Pairwise Statistical Tests . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.2.1 Human Dataset (DEG View) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.2.2 Mouse Dataset (DEG View) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.3 Effect Size (Cohen’s d) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.4 Reproducibility: Random Seed Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
F.5 Statistical Power Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
12

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
A. Theoretical Analysis and Proofs
In this section, we provide rigorous mathematical proofs for the three core theoretical claims of MAT-Cell. We adopt
a formal probabilistic framework to analyze the error bounds of the Dialectic Verification mechanism, the convergence
properties of the Syllogistic Derivation Tree (SDT), and the asymptotic identifiability of novel cell states via Inductive
Anchoring.
A.1. Proof of Error Bound for Dialectic Verification (Theorem 1)
We model the Dialectic Verification process as a consensus problem among a committee of noisy binary classifiers.
Setup.Let vbe a proposed Logical Tuple with ground truth validity y∈ {0,1} . The Council of Verifiers consists of K
agents{f(k)
ver}K
k=1. LetX k=I(f(k)
ver(v) =valid)be the binary indicator variable for thek-th agent’s approval.
Assumption A.1(Bounded Independent Error).We assume that the verifier agents are independent conditionally on the
input tuplev, and each agent has a bounded error rateϵ <0.5. Formally:
P(X k= 1|y= 0)≤ϵ(False Positive Rate) (10)
P(X k= 0|y= 1)≤ϵ(False Negative Rate) (11)
Recall the consensus criterion used in the main text (Eq. (6)): the council accepts a proposed tuple vonly when all verifier
agents agree. Equivalently, the system accepts vif and only if X1=X 2=···=X K= 1, i.e., every agent approves the
tuple as valid. This is astrict unanimous-consensusrule designed to suppress hallucinated logical steps.
Theorem A.2(Unanimous Consensus Suppresses Hallucinations).For a false tuple ( y= 0 ), the probability that the council
incorrectly accepts it (Type I Error / hallucination) under unanimous consensus satisfies
P(Accept|y= 0)≤ϵK.(12)
For a true tuple (y= 1), the probability that the council rejects it satisfies
P(Reject|y= 1)≤1−(1−ϵ)K.(13)
In particular, the hallucination probability decays exponentially in the council sizeK.
Proof. Under Assumption A.1, for a false tuple ( y= 0 ) each agent approves with probability at most ϵ, i.e., P(X k= 1|
y= 0)≤ϵ. Under the unanimous-consensus rule, the tuple is accepted only if allKagents approve:
P(Accept|y= 0) =P(X 1=···=X K= 1|y= 0) =KY
k=1P(X k= 1|y= 0)≤ϵK,(14)
where the product form follows from conditional independence.
Similarly, for a true tuple ( y= 1 ), each agent rejects with probability at most ϵ, i.e., P(X k= 0|y= 1)≤ϵ , hence
P(X k= 1|y= 1)≥1−ϵ . The probability that all agents approve is at least (1−ϵ)K, so the rejection probability is
bounded by
P(Reject|y= 1) = 1−P(Accept|y= 1)≤1−(1−ϵ)K.(15)
This proves the stated bounds and shows exponential suppression of hallucinations asKincreases.
A.2. Proof of Convergence for Syllogistic Derivation Tree (Theorem 2)
Setup.Let Ydenote the finite ontology of candidate cell types with |Y|=M . Given the input card xcard
i, the Solve
Agent produces a candidate set Ccand
i⊆ Y. At dialectic round t∈ {1, . . . , T max}, each rebuttal agent outputs a tentative
conclusiony(k)
i,t=RA k(xcard
i,Ccand
i,Ti,t). The SDT stateT i,tstores the council’s hypotheses and rebuttals up to roundt.
Theorem A.3(Bounded-Termination of SDT Construction).The SDT construction procedure terminates in at most Tmax
dialectic rounds. If unanimous consensus is reached at some round t≤T max, the algorithm outputs the consensus label yi.
Otherwise, it falls back to the Decision Agent and outputs an adjudicated label based on the final proof tree.
13

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Proof. At each round t, the algorithm performs a finite council interaction and then checks the unanimous-consensus
condition y(1)
i,t=···=y(K)
i,t. If the condition holds, the procedure halts immediately and returns the consensus label. If not,
the procedure updates the tree state Ti,t+1 and increments t. Since tis bounded by the predefined maximum depth Tmax, the
loop can execute at most Tmaxtimes. Therefore, the procedure must terminate either by consensus at some t≤T maxor by
reachingt=T max, after which the Decision Agent is invoked.
A.3. Proof of OOD Identifiability (Theorem 3)
Setup.Inductive Anchoring constructs a focused evidence space by intersecting observed cluster genes with retrieved marker
axioms. Let Gobsdenote the observed evidence genes for a query cluster (e.g., Top genes and/or DEGs), and let Span(K ret)
denote the union of marker genes retrieved for the candidate set. Define the anchored evidence set S=G obs∩Span(K ret).
We analyze identifiability using a marker gene g∈ S , where Xgis its expression in the query cluster and Ygis its expression
in the background context.
Assumption A.4(Gaussian Signal Model).We assume gene expression levels (after log-normalization) follow Gaussian
distributions:
• Background:Y g∼P in(g) =N(µ in, σ2)
• Novel State:X g∼P ood(g) =N(µ ood, σ2)
The signal magnitude is defined as the shift∆µ=|µ ood−µ in|.
We use the Contextual Divergence score Dctx(g) =|X g−µ in|as a simple proxy for marker saliency under anchored
evidence, and show it is statistically distinguishable from noise when the marker exhibits a mean shift.
Theorem A.5(Asymptotic Separability).For any error probability δ >0 , there exists a signal-to-noise ratio threshold such
that if∆µ
σ>Φ−1(1−δ/2) + Φ−1(1−δ) , the Contextual Divergence score will identify the marker gene with probability
at least1−δ.
Proof.LetZ=X g−µ in. Under Assumption A.4:
• Under Null HypothesisH 0(Noise gene,µ ood=µ in):Z∼ N(0, σ2).
• Under Alternative HypothesisH 1(Marker gene,µ ood̸=µ in):Z∼ N(∆µ, σ2)(assumingµ ood> µ inw.l.o.g.).
The detection rule is Dctx(g)> η , where ηis a critical value determined by the significance level α(False Positive Rate). To
control FPR atα, we setηsuch thatP(|Z|> η|H 0) =α. Using the properties of the standard normal CDFΦ:
η=σ·Φ−1(1−α/2)(16)
The Probability of Detection (Power) isP(D ctx(g)> η|H 1).
P(|Z|> η|H 1)≥P(Z > η|H 1) (sinceZis shifted positive)(17)
=PZ−∆µ
σ>η−∆µ
σ
(18)
= 1−Φη−∆µ
σ
(19)
= Φ∆µ−η
σ
(20)
We require the detection probability to be at least1−β(whereβis the Type II error rate). Letβ=δandα=δ.
Φ∆µ−η
σ
≥1−δ=⇒∆µ−η
σ≥Φ−1(1−δ)(21)
14

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Substitutingη:
∆µ−σΦ−1(1−δ/2)
σ≥Φ−1(1−δ)(22)
∆µ
σ≥Φ−1(1−δ/2) + Φ−1(1−δ)(23)
This inequality relates the signal-to-noise ratio (SNR) ∆µ/σ to the target error level δ. It shows that, once Inductive
Anchoring restricts reasoning to S=G obs∩Span(K ret), markers with sufficient mean shift are identified with high
probability, whereas non-specific housekeeping noise outside Span(K ret)is excluded by construction. Equivalently, in
information-theoretic terms, the KL divergence DKL(Pood∥P in) = (∆µ)2/(2σ2)increases with the marker shift, yielding
higher detection power and providing a principled basis for separating OOD states when discriminative markers exist in the
retrieved span.
B. Prompt Templates for Tree-Based Multi-Agent Reasoning
This appendix provides the exact prompt templates used in MAT-Cell for tree-based multi-agent reasoning. To ensure
reproducibility and transparency, we report the prompts verbatim. No task-specific fine-tuning or hidden instructions are
used beyond these templates.
B.1. Global System Prompt (Tree Reasoning Controller)
This system prompt defines the global behavioral constraints shared by all agents (SolveAgent, RebuttalAgent, and
DecisionAgent) in the tree-based reasoning framework. It specifies the role of the assistant as a node generator rather than a
final classifier, and enforces strict output formatting and label constraints.
Listing 1.System Prompt(OA)
TREE_SYSTEM_PROMPT = """You are an assistant designed for structured, tree-based multi-
agent reasoning in
single-cell annotation. You are NOT performing the original "final answer prediction"
task. You are NOT required to output any <think>...</think> or <answer>...</answer> tags.
Your task is to operate as a node generator in a reasoning tree:
1. You will always receive:
- A batch of cells (each with top and difference marker lists),
- Candidate cell types for this batch,
- A tree snippet representing the current reasoning state
(Solve root nodes, previous RA nodes, or decision branches).
2. You MUST output, for each cell i provided in the snippet, a compact 3-line node:
Cell i major: &lt;short sentence about the key distinguishing lineage/feature&gt;
Cell i minor: &lt;short sentence summarizing observed evidence from this cell&gt;
Cell i answer: &lt;ONE label chosen strictly from the given candidate list for this
cell&gt;
3. IMPORTANT:
- You MUST NOT output any global answer for the entire batch.
- You MUST NOT use &lt;think&gt; or &lt;answer&gt; tags.
- You MUST NOT introduce labels outside the candidate list.
- You MUST provide ALL cells shown in the snippet. Missing any cell invalidates output.
- You MUST keep each line short, factual, and focused on decisive features only.
4. You are NOT performing a multi-class classification.
You are producing a reasoning node summarizing how you justify a local decision
while being aware that other agents will compare and resolve disagreements.
5. "Major" should express the dominant biological rule or lineage clue.
"Minor" should reference specific observed patterns in the cell (no generic phrases).
"Answer" must choose ONE label from the allowed candidate list.
Your output will be used to grow the reasoning tree for this batch, and may be
15

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
provided to other agents (RA or DA) for comparison, critique, or conflict resolution.
Do not produce any additional text outside the required 3-line format per cell."""
Listing 2.System Prompt(CA)
TREE_SYSTEM_PROMPT = """You are an assistant designed for structured, tree-based multi-
agent reasoning in
single-cell annotation. You are NOT performing the original "final answer prediction"
task. You are NOT required to output any <think>...</think> or <answer>...</answer> tags.
Your task is to operate as a node generator in a reasoning tree:
1. You will always receive:
- A batch of cells (each with top and difference marker lists),
- Candidate cell types for this batch,
- A tree snippet representing the current reasoning state
(Solve root nodes, previous RA nodes, or decision branches).
2. You MUST output, for each cell i provided in the snippet, a compact 3-line node:
Cell i major: <short sentence about the key distinguishing lineage/feature>
Cell i minor: <short sentence summarizing observed evidence from this cell>
Cell i answer: <ONE label chosen strictly from the given candidate list for this cell>
3. IMPORTANT:
- You MUST NOT output any global answer for the entire batch.
- You MUST NOT use <think> or <answer> tags.
- You MUST NOT introduce labels outside the candidate list.
- You MUST provide ALL cells shown in the snippet. Missing any cell invalidates output.
- You MUST keep each line short, factual, and focused on decisive features only.
4. You are NOT performing a multi-class classification.
You are producing a reasoning node summarizing how you justify a local decision
while being aware that other agents will compare and resolve disagreements.
5. "Major" should express the dominant biological rule or lineage clue.
"Minor" should reference specific observed patterns in the cell (no generic phrases).
"Answer" must choose ONE label from the allowed candidate list.
Your output will be used to grow the reasoning tree for this batch, and may be
provided to other agents (RA or DA) for comparison, critique, or conflict resolution.
Do not produce any additional text outside the required 3-line format per cell."""
B.2. SolveAgent Prompt (Tree Layer 0: Candidate Generation)
TheSolveAgent is responsible for the initial processing of each cell/cluster. It identifies key biological rules and observed
markers to propose a set of potential candidate types.
Listing 3.SolveAgent Template(OA)
agent_solve_template = """{system_msg}
{user_msg}
Role:
You are a SolveAgent in a tree-based multi-agent **cluster-level **cell-type annotation
system.
Your ONLY responsibility in this round is to produce 0-layer candidate nodes for all
clusters.
Task:
For each cluster i, output EXACTLY three lines in order:
Cluster i major: <short key rule or lineage>
Cluster i minor: <short observed facts / markers / tissue context>
16

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Cluster i candidates: <2-4 candidate types from the RAG list for this cluster, separated
by " | ">
Rules:
- Use ONLY the candidate types listed in the **[RAG: cluster candidates] **block for the
SAME cluster.
- For each cluster, list between 2 and 4 distinct candidate types whenever possible.
- Format the candidates line EXACTLY as:
Cluster i candidates: label_A | label_B | label_C
- If you think only 1 label is plausible for a cluster, you may still duplicate it to fill
2 slots (e.g., "label_A | label_A") rather than inventing new labels.
- Strictly follow the original cluster order (Cluster 0, Cluster 1, ..., Cluster M) as
shown in the user message.
- You MAY reuse the same candidate type for different clusters.
- Keep each field short and focused (one short phrase or sentence per line).
- In both "major" and "minor", explicitly mention the most distinctive biological features
observed in that cluster (marker genes, lineage-relevant signals).
- Do NOT use vague descriptions such as "general markers" or "nonspecific expression".
Hard constraints:
- Do NOT output any line that begins with "Cluster i answer".
- Do NOT output any global ANSWER string or any line that begins with "ANSWER:".
- Do NOT add <think> blocks or any other free-form commentary.
Return ONLY these three-line blocks for ALL clusters, in order.
"""
Listing 4.SolveAgent Template(CA)
agent_solve_template = """
Role:
You are a SolveAgent in a tree-based multi-agent cell-type annotation system.
Your ONLY responsibility in this round is to produce 0-layer candidate nodes for all cells.
Task:
For each cell i, output EXACTLY three lines in order:
Cell i major: <short key rule or lineage>
Cell i minor: <short observed facts / markers / tissue context>
Cell i candidates: <2-4 candidate types from the candidate list, separated by " | ">
Rules:
- Use ONLY the candidate types provided in the user message.
- For each cell, list between 2 and 4 distinct candidate types whenever possible.
- Format the candidates line EXACTLY as:
Cell i candidates: label_A | label_B | label_C
- If you think only 1 label is plausible for a cell, you may still duplicate it to fill 2
slots.
- Strictly follow the original cell order (Cell 1, Cell 2, ..., Cell N).
- You MAY reuse the same candidate type for different cells.
- Keep each field short and focused.
- Explicitly mention distinctive biological features in both "major" and "minor".
Hard constraints:
- Do NOT output any line that begins with "Cell i answer".
- Do NOT output any global ANSWER string.
- Do NOT add <think> blocks or free-form commentary.
Return ONLY these three-line blocks for ALL cells, in order.
"""
17

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
B.3. RebuttalAgent Prompt (Tree Layer≥1: Per-Cell Adjudication)
TheRebuttalAgent (RA) performs iterative refinement. It reviews the candidates and prior reasoning to output definitive
per-cell decisions on specific tree branches.
Listing 5.RebuttalAgent Template (OA)
agent_rebut_template = """{system_msg}
{user_msg}
[Tree]
{peer_reports}
Role:
You are a RebuttalAgent in a tree-based multi-agent **cluster-level **cell-type
annotation system.
You read the existing tree nodes (0-layer SolveAgent candidates and any higher-layer nodes
)
and must output your own definitive per-cluster decisions ONLY for the clusters included
in the tree snippet above.
Task (Per-cluster adjudication on a tree branch):
For each cluster i that appears in the tree snippet above, you MUST output exactly
THREE lines describing your decision for that cluster:
Cluster i major: <short key rule or lineage>
Cluster i minor: <short observed facts / markers / tissue context>
Cluster i answer: <your chosen cell type for this cluster, as a single candidate label>
Requirements for "Cluster i answer" line:
- The answer MUST be exactly ONE of the candidate types provided in the RAG block for that
cluster.
- Your "Cluster i answer" MUST be selected strictly from the candidate labels produced by
the SolveAgent for that same cluster.
You MUST NOT introduce any new label.
- Do NOT include probabilities or parentheses.
- Use the same numeric index i as in the tree.
- Keep the cluster order consistent with the order in which clusters appear in the tree
snippet.
General rules:
- Base your decision on the 0-layer SolveAgent candidates and any existing RA nodes in the
tree for those clusters.
- Prefer biologically consistent assignments (marker genes + tissue + lineage).
- When prior agents disagree, choose the option that is most consistent with markers and
global context.
- You MUST provide outputs for ALL clusters shown in the tree snippet. Missing any cluster
i invalidates the output.
Hard constraints:
- Do NOT output any line that begins with "Cluster i candidate".
- Do NOT output any summary line that begins with "ANSWER:".
- Do NOT add <think> blocks or any other free-form commentary.
Return ONLY the three-line blocks for the clusters you are asked to judge.
"""
Listing 6.RebuttalAgent Template (CA)
agent_rebut_template = """
Role:
You are a RebuttalAgent (RA) in a tree-based multi-agent cell-type annotation system.
Task:
For each cell i in the tree snippet, output EXACTLY three lines:
18

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Cell i major: <short key rule or lineage>
Cell i minor: <short observed facts / markers / tissue context>
Cell i answer: <chosen cell type>
Requirements:
- The answer MUST be selected from the candidate labels produced by the SolveAgent for
that same cell.
- Do NOT introduce new labels.
- Do NOT include probabilities or parentheses.
- Keep the original cell indices and order.
Hard constraints:
- Do NOT output "Cell i candidate".
- Do NOT output any summary line or <think> block.
Return ONLY the three-line blocks for the cells you judge.
"""
B.4. DecisionAgent Prompt (Final Resolution)
TheDecisionAgent acts as the final adjudicator for disputed branches, integrating all prior reasoning rounds to reach a
consensus.
Listing 7.DecisionAgent Template (OA)
agent_da_template = """{system_msg}
{user_msg}
[Tree]
{peer_reports}
Role:
You are a DecisionAgent (DA) in a tree-based multi-agent **cluster-level **cell-type
annotation system.
You see the full dispute branch for one or more clusters, including:
- 0-layer SolveAgent candidates, and
- all RebuttalAgent (RA) rounds for those clusters.
Your job is to make a FINAL, single-label decision for each listed cluster.
Task (Final per-cluster resolution):
For each cluster i that appears in the tree snippet above, you MUST output exactly
THREE lines describing your final decision for that cluster:
Cluster i major: <short key rule or lineage that supports your final decision>
Cluster i minor: <short observed facts / markers / tissue context you rely on>
Cluster i answer: <your FINAL chosen cell type for this cluster, as a single candidate
label>
Requirements for "Cluster i answer" line:
- The answer MUST be exactly ONE of the candidate types provided in the RAG block for that
cluster.
- Your "Cluster i answer" MUST be selected strictly from the candidate labels originally
produced by the SolveAgent for that cluster.
Do NOT invent new labels.
- Do NOT include probabilities or parentheses.
- Use the same numeric index i as in the tree snippet.
- Keep the cluster order consistent with the order in which clusters appear in the tree
snippet.
General rules:
- You MUST integrate information from both Solve (0-layer) and all RA rounds for those
clusters.
- Your decision should be biologically consistent (markers + tissue + lineage) and resolve
19

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
disagreements shown in the tree.
- When agents disagree, choose the option that is most consistent with markers, context,
and lineage.
- You MUST provide outputs for ALL clusters that appear in the tree snippet; missing any
cluster i makes your output invalid.
Hard constraints:
- Do NOT output any line that begins with "Cluster i candidate".
- Do NOT output any summary line that begins with "ANSWER:".
- Do NOT add <think> blocks or any other free-form commentary.
Return ONLY the three-line blocks for the clusters you are asked to finally decide.
"""
Listing 8.DecisionAgent Template (CA)
agent_da_template = """
Role:
You are a DecisionAgent (DA) in a tree-based multi-agent cell-type annotation system.
Task:
For each cell i in the tree snippet, output EXACTLY three lines:
Cell i major: <supporting lineage rule>
Cell i minor: <supporting markers / context>
Cell i answer: <FINAL chosen cell type>
Rules:
- The answer MUST be chosen strictly from SolveAgent candidates.
- Integrate information from all prior RebuttalAgent rounds.
- Resolve disagreements using biological consistency.
Hard constraints:
- Do NOT introduce new labels.
- Do NOT output summaries or free-form text.
- Do NOT add <think> blocks.
Return ONLY the three-line blocks for all listed cells.
"""
20

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
C. Dataset Documentation
C.1. Dataset Summary Statistics
C.1.1. PBMC3K DATASET
Table 4.Top PBMC3K cell types by frequency
Cell type Count
CD4+ T Cells 1,157
Classical Monocytes 479
B Cells/Dendritic Cells 341
Effector/Activated T Cells 297
Effector/Activated T Cells/NK Cells 158
Non-Classical Monocytes 157
Dendritic Cells/Monocytes 36
Platelets/Megakaryocytes 13
C.1.2. LIVERDATASET
Table 5.Top Liver cell types by frequency
Cell type Count
B cells 7,565
Neutrophils 6,920
Hepatocytes 6,363
Erythrocytes 6,277
NK cells 4,993
Liver Sinusoidal Endothelial Cells 4,797
Macrophages 3,518
Fibroblasts/Hepatic Stellate Cells 795
Cholangiocytes 116
C.1.3. RETINADATASET
Table 6.Top Retina cell types by frequency
Cell type Count
Photoreceptor Cells 10,641
Astrocytes 4,148
Neurons 3,251
Microglia 1,174
Bipolar Neurons 437
Retinal Ganglion Cells 336
T Cells 104
C.1.4. BRAINDATASET
C.1.5. HEARTDATASET
C.1.6. HUMANDATASET
C.1.7. MOUSEDATASET
C.1.8. MONKEYDATASET
C.2. Top-25 Highly Expressed Genes
To characterize the global transcriptional landscape, we report the 25 most frequently observed genes across all cells for
each species, excluding mitochondrial and ribosomal genes.
21

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 7.Top Brain cell types by frequency
Cell type Count
oligodendrocyte 234,151
L2/3 intratelencephalic projecting glutamatergic neuron 88,102
astrocyte 86,115
L2/3–6 intratelencephalic projecting glutamatergic neuron 63,404
microglial cell 30,764
oligodendrocyte precursor cell 30,670
VIP GABAergic cortical interneuron 29,838
pvalb GABAergic cortical interneuron 27,736
sst GABAergic cortical interneuron 20,336
L6 intratelencephalic projecting glutamatergic neuron 13,306
Table 8.Top Heart cell types by frequency
Cell type Count
cardiac endothelial cell 1,606
myeloid cell 269
fibroblast of cardiac tissue 207
pericyte 159
cardiac muscle cell 136
lymphocyte 63
cardiac neuron 44
smooth muscle cell 30
Brain dataset (Top-25 genes).CNTNAP2, DSCAM, DPP10, ROBO2, KCNIP4, GRIP1, ZNF385D, EPIC1, CA10,
FSTL4, ARL15, HTR1F, FOXP2, FSTL5, MYO16, PTCHD4, CLSTN2, CPNE4, NRG1, DTNA, CBLN2, CDH9, SLIT3,
SLIT2, UNC13C.
Heart dataset (Top-25 genes).RGS6, ANK3, LINC02147, CLIC5, HIGD1B, PIK3R5, SLC38A11, MLIP, TRDN-AS1,
MYBPC3, XIRP2, MYH6, ACACB, PRKAG2, ACTA1, SH3RF2, PPP1R3C, ENSG00000230490, MYL7, LINC02552,
ENSG00000258231, ENSG00000271959, HECW2, FRMD5, G0S2.
Liver dataset (Top-25 genes).H3f3a, Ubb, Tmsb10, Hba-a1, Gpx1, Hba-a2, Hbb-bt, S100a8, S100a9, Apoe, Igfbp7,
Sparc, S100a6, Cd24a, Pglyrp1, Gm5483, BC100530, Stfa1, Anxa1, Ifitm6, Serpina3k, Mup20, Gnmt, Cd7, Ccl4.
PBMC3K dataset (Top-25 genes).B2M, RPL13, MALAT1, RPL21, TPT1, RPL10A, ACTB, RPL8, H3F3B, RPS3A,
RPS5, EEF1D, RPS27A, FTH1, MT-CO2, CD74, FTL, OAZ1, CD37, CD79A, FCGR3A, LST1, COTL1, GNLY , GZMB.
Retina dataset (Top-25 genes).FTH1, RHO, APOE, RBP3, WIF1, GLUL, CLU, PTGDS, TF, MFGE8, MPP4, FRZB,
CRABP1, SPP1, ENO1, DKK3, RLBP1, CA2, GPX3, CRYAB, CADPS, NEAT1, NR2E3, YPEL2, CNGA1.
Human Top-25 genes.MALAT1, ACTB, ACTG1, B2M, TMSB4X, FTH1, GAPDH, RPL13, RPS27, RPL41, RPL21,
EEF1A1, TPT1, RPS3A, RPL32, RPL3, RPS2, RPS18, RPS6, RPS12, RPL10, RPL34, RPS27A, RPL13A, RPL11.
Mouse Top-25 genes.Malat1, Actb, Gapdh, B2m, Tmsb4x, Rpl13, Rpl41, Rps27, Eef1a1, Rpl21, Rpl32, Rps3a, Rpl3,
Rps18, Rps2, Rpl10, Rps12, Rpl34, Rps6, Rpl13a, Rps27a, Rpl11, Ftl1, Fth1, Tpt1.
Monkey Top-25 genes.ZC3H10, RPS18, TMSB4Y , RPS27, RPLP1, RPS28, FTL, ACTB, RPL37, FTH1, RPS19, RPS14,
RPS15, RPS12, RPL13A, RPS23, RPL13, RPL37A, RPS27A, RPS15A, FAU, RPS17, RPLP0, B2M, RPL23A.
C.3. Differentially Expressed Genes (DEG) Criteria
We precompute DE marker genes at thecell-type levelusing a one-vs-rest differential expression test (Scanpy
rank_genes_groups ), and then attach the resulting marker list to each cell based on its cell_type annotation
(i.e., markers are not computed per cell on-the-fly).
22

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 9.Summary statistics of the human dataset
Property Value
Number of batches 2,400
Total cells 27,588
Number of cell types 75
Unique top genes 5,583
Unique DEG genes 1,434
Cells per batch 7 – 15
Table 10.Top human cell types by frequency
Cell type Count
Oligodendrocyte 1,593
L2/3–6 intratelencephalic projecting glutamatergic neuron 1,560
Astrocyte 1,544
Oligodendrocyte precursor cell 1,508
L2/3 intratelencephalic projecting glutamatergic neuron 1,497
Pvalb GABAergic cortical interneuron 1,488
Microglial cell 1,445
VIP GABAergic cortical interneuron 1,402
Sst GABAergic cortical interneuron 1,243
Lamp5 GABAergic cortical interneuron 1,194
•Statistical test:Wilcoxon rank-sum test (one-vs-rest, grouped bycell_type).
•Log fold-change threshold:|log2FC| ≥1.0.
•Adjusted p-value threshold:FDR≤0.05(Benjamini–Hochberg correction;pvals_adjin Scanpy).
•Expression proportion threshold:keep genes with non-zero expression proportion satisfying pcttarget≥0.1 or
pctreference ≥0.1(corresponding topct_nz_group/pct_nz_reference).
•Top-N truncation:for each cell type, rank genes by ascending FDR and descending log2FC, and keep the top 25
genes.
•Rare type filtering:cell types with fewer than 3 cells are excluded from DEG computation to avoid unstable statistics.
C.4. Data Processing and Instance Construction
We convert each raw .h5ad file into structured per-cell records and further organize them into batch-level instances for
LLM inference. The processing steps are:
1.Subsampling (size control):if a file contains more than max_cells cells, we randomly subsample to max_cells
cells to control runtime and output size.
2.Top expressed genes:for each cell, we extract the top-25 expressed genes from the expression matrix Xusing an
efficient partition-based selection (np.argpartition) and then sort them by expression in descending order.
3.Gene name normalization:if feature_name is available in adata.var , we use it as a human-readable gene
symbol; additionally, names of the formSYMBOL_ENSG...are truncated toSYMBOL.
4.Type-level DEG attachment:if cell_type is present in adata.obs , we compute DEG markers once per file
using the criteria above and attach the corresponding top-25 marker list ( deg_markers ) to each cell based on its
cell_type. If no valid markers exist for a type (after filtering),deg_markersis omitted.
5.Context construction:we build a natural-language context string from available metadata fields (e.g., disease ,
tissue ,sex,development_stage , and self_reported_ethnicity ), and append the top expressed
genes to form the final context used by the LLM.
23

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 11.Summary statistics of the mouse dataset
Property Value
Number of batches 2,400
Total cells 27,583
Number of cell types 123
Unique top genes 6,941
Unique DEG genes 2,432
Cells per batch 7 – 15
Table 12.Top mouse cell types by frequency
Cell type Count
Fibroblast 924
Epithelial cell of proximal tubule segment 1 903
Epithelial cell of proximal tubule segment 2 853
Kidney distal convoluted tubule epithelial cell 849
Epithelial cell of proximal tubule 820
Kidney collecting duct principal cell 807
Macrophage 796
Kidney connecting tubule epithelial cell 753
Epithelial cell of proximal tubule segment 3 727
Kidney loop of Henle thick ascending limb epithelial cell 579
D. Detailed Methodology
D.1. Inductive Anchoring Algorithm
D.1.1. ALGORITHMICPSEUDOCODE
Algorithm 2Inductive Anchoring: Candidate-Set Anchoring & Marker Filtering
inputRaw expression matrixX∈Rn×g, cluster labelsz∈ {1, . . . , C}n, RAG marker knowledge baseK, top-Kbudget
outputAnchored candidate setC anchor and filtered marker setG filtered for each cluster
1:Quality control:filter low-quality cells and rare genes (e.g., min-genes-per-cell, min-cells-per-gene)
2:Normalize & log:total-count normalization andlog(1 +x)transform
3:Global gene screening:select highly-variable genes (HVGs) and restrictXto the HVG space
4:Cluster-level summarization:for each clusterc∈ {1, . . . , C},
5:computeTopgenesGtop
cby ranking mean expression within clusterc
6:computeDEGgenesGdeg
cvia cluster-vs-rest differential test (e.g., Wilcoxon)
7:remove ubiquitous / non-informative genes fromGtop
c∪ Gdeg
c(e.g., housekeeping, MT/RP genes)
8:build the marker poolG c←TopK 
Gtop
c∪ Gdeg
c, K
9:RAG anchoring:queryKwithG cand retrieve a ranked label listR c={(ℓ, s ℓ)}
10:keep top-Mlabels as the anchored candidate setC anchor(c)← {ℓ 1, . . . , ℓ M}
11:return{(C anchor(c),G c)}C
c=1
D.2. Dialectic Verification Mechanism
D.2.1. AGENTCONFIGURATION
•Agent Types: SolveAgent (constructs an SDT proposal under anchored candidates), RebuttalAgent (audits,
refutes, and prunes inconsistent nodes), and DecisionAgent (aggregates surviving evidence and outputs the final
decision)
•Instantiation:3 agent types in total; RebuttalAgent uses 3 parallel instances, while SolveAgent and
DecisionAgenteach use 1 instance
•LLM Model:Qwen-3-30B (via API)
•Temperature:0.7
24

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 13.Summary statistics of the monkey dataset
Property Value
Number of batches 2,400
Total cells 25,871
Number of cell types 121
Unique top genes 7,757
Unique DEG genes 2,485
Cells per batch 7 – 15
Table 14.Top monkey cell types by frequency
Cell type Count
alveolar macrophage 551
endothelial cell 544
lymphocyte 541
vein endothelial cell 540
plasma cell 536
CD4-positive, alpha-beta T cell 522
hematopoietic precursor cell 521
CD8-positive, alpha-beta T cell 520
neutrophil 514
natural killer cell 505
•Max tokens:20000 per response
D.2.2. EXACT-MATCHCONVERGENCECRITERION
Unlike soft semantic similarity, our debate halts only understrict agreement. Let ˆyidenote the final structured decision
produced by agent i(including the ordered label string and its SDT-supported justification summary). We define an
exact-match divergence indicator:
Ldiv({ˆy1, . . . ,ˆy k}) =(
0,ifˆy 1= ˆy2=···= ˆy k
1,otherwise.
Convergence criterion:L div= 0, i.e., all agents outputidenticaldecisions.
D.3. Syllogistic Rule Set
Table 15.Biological Syllogistic Rules (Example)
Rule ID Markers (IF) Cell Type (THEN)
R1 CD4, IL7R, TCF7 CD4+ T-Cell
R2 CD8A, CD8B, GZMA CD8+ T-Cell
R3 CD14, LYZ, FCGR3A Monocyte
R4 PPBP, PF4, TUBB1 Platelet
D.4. Hyperparameter Settings
D.5. Syllogistic Derivation Tree (SDT) Construction
SDT construction proceeds as a debate-driven, tree-structured proof search under the anchored candidate space:
1.Initialize:create a root node for each cluster using its marker pool Gcand anchored candidates Canchor(c)from
Algorithm 2.
2.Solve: SolveAgent generates an SDT proposal by composing syllogistic triads (major premise: marker-to-lineage
rule; minor premise: observed marker evidence; conclusion: candidate label or intermediate lineage).
25

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 16.Hyperparameters for MAT-Cell
Parameter Value
Number of Agent Types 3
#SolveAgent Instances 1
#RebuttalAgent Instances 3
#DecisionAgent Instances 1
Top-K genes (Both view) 25
Maximum Debate Rounds 5
Convergence Criterion Exact-match (all identical)
Temperature 0.7
Max tokens 20000
3.Rebut & prune:multiple RebuttalAgent s independently audit the SDT at the premise level, flagging contradic-
tions, missing evidence, or candidate misuse, and pruning invalid branches.
4.Decide: DecisionAgent aggregates the surviving branches and outputs a single final label decision (and its minimal
SDT justification).
5.Iterate:if agents do not reach exact-match convergence ( Ldiv= 1), start a new round with the pruned SDT state, up
to a maximum of 3 rounds.
26

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
E. Extended Ablation Studies
Ablation studies are crucial for validating whether each component of MAT-Cell contributes substantively to performance
improvements, rather than merely increasing system complexity. To this end, we systematically remove or modify individual
components while keeping all other conditions unchanged, and evaluate the resulting performance variations on held-out
test data.
For each ablation setting, we conduct multiple independent runs to ensure statistical robustness. All ablation experiments
adopt the same backbone large language model (Qwen-3-30B), identical hyperparameter settings (temperature = 0.7), and
the same evaluation protocol as the full MAT-Cell system, where 300 randomly sampled batches are evaluated per run.
E.1. Impact of Gene Count
How many genes are required for reliable cell type annotation when no retrieval augmentation is applied? To answer
this question, we perform a controlled ablation usingQwen3-30B without RAG, systematically varying the number of
top-expressed genes (K∈ {5,10,15,25,50}) provided as input, while keeping all other settings fixed.
Findings:
•Performance Improves with Increasing K, but Saturates Early:Across all three species, accuracy increases
substantially from Top-5 to Top-25. However, gains beyond Top-25 are marginal or even negative. For example,
on Mouse, performance drops from 0.503 at Top-25 to 0.470 at Top-50, indicating that excessive gene inputs may
introduce noise rather than useful signal in the absence of retrieval guidance.
•Cross-Species Sensitivity to Gene Budget:Monkey consistently benefits from larger gene sets, peaking at Top-25
(0.721), while Human and Mouse exhibit weaker gains and earlier saturation. This reflects intrinsic differences in
annotation difficulty and marker specificity across species.
•No-RAG Limitation under Low-Information Regimes:With only Top-5 genes, performance is severely degraded on
Human (0.371) and Mouse (0.402), highlighting that, without RAG, the backbone LLM struggles to reason reliably
under extreme information scarcity.
•Implication for RAG Design:These results establish a strong no-RAG baseline, against which the substantial
improvements introduced by RAG (Section E.3) can be attributed to enhanced biological grounding rather than
increased gene quantity alone.
Table 17.Accuracy by Top-K Gene Count using Qwen3-30B without RAG
Top-K Human (DEG) Monkey (DEG) Mouse (DEG)
Top-50.371±0.006 0.630±0.008 0.402±0.016
Top-100.513±0.003 0.690±0.009 0.461±0.010
Top-150.532±0.004 0.705±0.007 0.463±0.004
Top-250.564±0.0090.721±0.007 0.503±0.009
Top-500.568±0.0050.708±0.006 0.470±0.005
E.2. Gene Ordering Randomization
To evaluate robustness against input perturbations, we randomly shuffle the order of input DEG genes while keeping the
gene set unchanged. Since MAT-Cell reasons over gene identity rather than positional cues, its performance should be
invariant to such ordering changes.
Interpretation.Across Human, Mouse, and Monkey datasets, MAT-Cell exhibits strong robustness to gene ordering
perturbations and random seed variation. The maximum deviation from the species-specific mean is below 1% for Human
and Monkey, and below 2.1% for Mouse, indicating that performance differences are not driven by favorable random seeds
or positional artifacts, but arise from the model’s structured, set-level reasoning mechanism.
27

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 18.Robustness to Random Seed under DEG Gene Input
Seed Human (DEG) Monkey (DEG) Mouse (DEG) Deviation from Mean
Seed-10.736±0.017 0.758±0.006 0.827±0.003H:-0.21%, Mk:-0.37%, M:-1.52%
Seed-20.751±0.005 0.759±0.008 0.837±0.004H:+1.77%, Mk:-0.19%, M:-0.34%
Seed-30.744±0.011 0.763±0.005 0.840±0.002H:+0.87%, Mk:+0.34%, M:+0.01%
Seed-40.726±0.010 0.755±0.002 0.852±0.010H:-2.43%, Mk:-0.84%, M:+1.85%
Mean±Std0.739±0.010 0.759±0.004 0.839±0.010Stable
E.3. Impact of RAG Gene Budget and Distillation Backbone
To further analyze the contribution of retrieval-augmented generation (RAG) in MAT-Cell, we conduct an ablation study
on thegene budgetprovided by RAG and the choice ofdistillation backbone. Using Qwen3-30B as the fixed reasoning
backbone, we vary (i) the number of marker genes distilled by RAG and (ii) the upstream distillation model, while keeping
all other settings unchanged.
Table 19 reports results on Human, Monkey, and Mouse datasets under thebothsetting.
Table 19.Ablation on RAG Gene Budget and Distillation Backbone (Qwen3-30B)
Method Human (both) Monkey (both) Mouse (both)
RAG (Gemini-3, 5 markers)0.629±0.006 0.779±0.010 0.674±0.013
RAG (Gemini-3, 10 markers)0.643±0.003 0.804±0.005 0.697±0.007
RAG (Gemini-3, 15 markers)0.664±0.005 0.808±0.003 0.700±0.007
RAG (Gemini-3, 20 markers)0.640±0.005 0.800±0.004 0.690±0.006
RAG (GPT-5.2, 15 markers)0.656±0.003 0.792±0.007 0.566±0.004
Interpretation.Several observations can be drawn from Table 19. First, increasing the RAG gene budget from 5 to 15
consistently improves performance across all three species, indicating that richer but still concise marker sets provide
stronger biological grounding for downstream reasoning. Second, performance saturates or slightly degrades beyond 15–20
genes, suggesting that excessive retrieved genes may reintroduce noise, consistent with the signal-to-noise trade-off observed
in Section E.1. Third, although GPT-5.2-based distillation achieves competitive results on Human and Monkey, it exhibits
a pronounced drop on Mouse, highlighting that the choice of distillation backbone can significantly affect cross-species
robustness. Overall, these results demonstrate that both the quantity and the source of RAG-distilled genes are critical
factors, with Gemini-3 and a moderate gene budget (around 15 markers) providing the best balance between informativeness
and robustness.
28

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
F. Statistical Rigor and Confidence Intervals
F.1. Confidence Intervals (95%)
F.1.1. MAINRESULTSTABLE WITHCI
Table 20.Main Results with 95% Confidence Intervals
Method Human (DEG) 95% CI Mouse (DEG) 95% CI Monkey (DEG) 95% CI
Cell-o10.409[0.403, 0.414]0.394[0.388, 0.401]0.685[0.678, 0.692]
Direct-Qwen-3-30B0.450[0.445, 0.455]0.387[0.384, 0.390]0.644[0.641, 0.647]
Direct-DeepSeek-V30.632[0.627, 0.636]0.567[0.563, 0.571]0.471[0.468, 0.474]
Direct-Gemini-2.50.709[0.705, 0.713]0.659[0.653, 0.665]0.859[0.856, 0.862]
Direct-GPT-4.10.733[0.731, 0.736]0.649[0.644, 0.654]0.864[0.863, 0.866]
MAT-Cell0.764[0.760,0.768]0.825[0.823,0.828]0.759[0.756,0.761]
F.2. Pairwise Statistical Tests
F.2.1. HUMANDATASET(DEG VIEW)
Table 21.Human Dataset: Paired T-test Results (DEG View)
Comparison Mean Diff t-statistic df p-value
MAT-Cell vs. GPT-4.1+0.031 +3.12290.0042**
MAT-Cell vs. Gemini-2.5+0.055 +6.2829<0.0001***
MAT-Cell vs. DeepSeek-V3+0.132 +9.2429<0.0001***
MAT-Cell vs. Direct-Qwen-3-30B+0.314 +15.8229<0.0001***
F.2.2. MOUSEDATASET(DEG VIEW)
Table 22.Mouse Dataset: Paired T-test Results (DEG View)
Comparison Mean Diff t-statistic df p-value
MAT-Cell vs. GPT-4.1+0.110 +2.89290.0071**
MAT-Cell vs. Gemini-2.5+0.100 +5.8329<0.0001***
MAT-Cell vs. DeepSeek-V3+0.192 +8.6729<0.0001***
MAT-Cell vs. Direct-Qwen+0.372 +14.2129<0.0001***
Significance codes:∗ ∗ ∗p <0.001,∗∗p <0.01,∗p <0.05
F.3. Effect Size (Cohen’s d)
F.4. Reproducibility: Random Seed Analysis
Figure 5.Performance distribution across 30 random seeds (Human, DEG view). Box plot shows median, quartiles, and outliers.
MAT-Cell maintains high consistency withσ= 0.011.
F.5. Statistical Power Analysis
Interpretation:Our study is adequately powered (β >0.78) to detect meaningful differences against all baselines.
29

MAT-Cell: Multi-Agent Tree-Structured Reasoning for scRNA-seq Annotation
Table 23.Effect Sizes (Cohen’s d) for MAT-Cell vs. Baselines
Comparison Human (d) Mouse (d) Monkey (d)
MAT-Cell vs. GPT-4.1 3.21 (very large) 17.01 (very large) 1.26 (large)
MAT-Cell vs. Gemini-2.5-Flash 5.29 (very large) 14.04 (very large) 0.53 (medium)
MAT-Cell vs. DeepSeek-V3 11.48 (very large) 31.18 (very large) 6.07 (very large)
MAT-Cell vs. Direct-Qwen-3-30B 26.01 (very large) 59.38 (very large) 8.86 (very large)
Table 24.Post-hoc Power Analysis
Contrast Effect Size (d) Sample Size (n) Power (1-β)
MAT-Cell vs. GPT-4.1 3.21 30 0.99+
MAT-Cell vs. Gemini-2.5-Flash 5.29 30 0.99+
MAT-Cell vs. DeepSeek-V3 11.48 30 0.99+
MAT-Cell vs. Direct-Qwen-3-30B 26.01 30 0.99+
30
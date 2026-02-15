# PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics

**Authors**: E Fan, Lisong Shi, Zhengtong Li, Chih-yung Wen

**Published**: 2026-02-12 07:37:56

**PDF URL**: [https://arxiv.org/pdf/2602.11666v1](https://arxiv.org/pdf/2602.11666v1)

## Abstract
The deployment of autonomous agents for Computational Fluid Dynamics (CFD), is critically limited by the probabilistic nature of Large Language Models (LLMs), which struggle to enforce the strict conservation laws and numerical stability required for physics-based simulations. Reliance on purely semantic Retrieval Augmented Generation (RAG) often leads to "context poisoning," where agents generate linguistically plausible but physically invalid configurations due to a fundamental Semantic-Physical Disconnect. To bridge this gap, this work introduces PhyNiKCE (Physical and Numerical Knowledgeable Context Engineering), a neurosymbolic agentic framework for trustworthy engineering. Unlike standard black-box agents, PhyNiKCE decouples neural planning from symbolic validation. It employs a Symbolic Knowledge Engine that treats simulation setup as a Constraint Satisfaction Problem, rigidly enforcing physical constraints via a Deterministic RAG Engine with specialized retrieval strategies for solvers, turbulence models, and boundary conditions. Validated through rigorous OpenFOAM experiments on practical, non-tutorial CFD tasks using Gemini-2.5-Pro/Flash, PhyNiKCE demonstrates a 96% relative improvement over state-of-the-art baselines. Furthermore, by replacing trial-and-error with knowledge-driven initialization, the framework reduced autonomous self-correction loops by 59% while simultaneously lowering LLM token consumption by 17%. These results demonstrate that decoupling neural generation from symbolic constraint enforcement significantly enhances robustness and efficiency. While validated on CFD, this architecture offers a scalable, auditable paradigm for Trustworthy Artificial Intelligence in broader industrial automation.

## Full Text


<!-- PDF content starts -->

PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous
Computational Fluid Dynamics
E Fan1, Lisong Shia),1, Zhengtong Li1, and Chih-yung Wenb),1
1Department of Aeronautical and Aviation Engineering, Hong Kong Polytechnic University, Hong Kong SAR
February 13, 2026
ABSTRACT
The deployment of autonomous agents for Computational Fluid Dynamics (CFD) is critically limited
by the probabilistic nature of Large Language Models (LLMs), which struggle to enforce the strict
conservation laws and numerical stability required for physics-based simulations. Reliance on
purely semantic Retrieval Augmented Generation (RAG) often leads to “context poisoning,” where
agents generate linguistically plausible but physically invalid configurations due to a fundamental
Semantic-Physical Disconnect. To bridge this gap, this work introducesPhyNiKCE(Physical
andNumericalKnowledgeableContextEngineering), a neurosymbolic agentic framework for
trustworthy engineering. Unlike standard black-box agents, PhyNiKCE decouples neural planning
from symbolic validation. It employs a Symbolic Knowledge Engine that treats simulation setup
as a Constraint Satisfaction Problem, rigidly enforcing physical constraints via a Deterministic
RAG Engine with specialized retrieval strategies for solvers, turbulence models, and boundary
conditions. Validated through rigorous OpenFOAM experiments on practical, non-tutorial CFD tasks
using Gemini-2.5-Pro/Flash, PhyNiKCE demonstrates a 96% relative improvement over state-of-
the-art baselines. Furthermore, by replacing trial-and-error with knowledge-driven initialization,
the framework reduced autonomous self-correction loops by 59% while simultaneously lowering
LLM token consumption by 17%. These results demonstrate that decoupling neural generation from
symbolic constraint enforcement significantly enhances robustness and efficiency. While validated on
CFD, this architecture offers a scalable, auditable paradigm for Trustworthy Artificial Intelligence in
broader industrial automation.
KeywordsComputational Fluid Dynamics, Large Language Models-based Agent, Deterministic Retrieval Augmented
Generation, Neurosymbolic Artificial Intelligence, Semantic-Physical Disconnect, OpenFOAM
1 Introduction
Computational Fluid Dynamics (CFD) serves as a foundational tool in critical engineering fields, spanning aerospace [ 1],
energy systems [ 2], and biomedical applications [ 3]. However, the practical deployment of CFD is severely restricted
by high barriers to entry: effective simulation demands extensive domain-specific knowledge to navigate labor-
intensive workflows [ 4]. The evolution of Artificial Intelligence (AI), particularly Large Language Models (LLMs) like
DeepSeek [ 5] and Gemini [ 6], offers a revolutionary path to automate these pipelines. Yet, recent evaluations reveal a
critical performance gap. Due to the high knowledge intensity of the domain, standard LLMs struggle to bridge the gap
between natural language and physical reality. Benchmarks such as CFD-LLMBench [ 7] highlight this limitation: in
zero-shot settings, even state-of-the-art (SOTA) models like Claude 3.5 Sonnet, GPT-4o, and Gemini-2.5-Flash achieve
only∼4.5% accuracy on basic CFD automation tasks. For complex, non-tutorial cases, this accuracy plummets to less
than 1%.
a)Corresponding author: ls.m.shi@polyu.edu.hk
b)Corresponding author: chihyung.wen@polyu.edu.hkarXiv:2602.11666v1  [cs.AI]  12 Feb 2026

APREPRINT- FEBRUARY13, 2026
Retrieval-Augmented Generation (RAG) [ 8] is the standard approach for mitigating such hallucinations. While standard
RAG improves performance, it fails to solve the fundamental reliability issue: CFD agents [ 7] equipped with vector-
based RAG plateau at ∼34% accuracy for basic CFD automation tasks and remain stuck at ∼25% for complex tasks,
even when utilizing a suite of SOTA LLMs. This failure stems from a Semantic-Physical Disconnect: conventional RAG
relies on vector embeddings that are compromised by flawed sub-word tokenization. For instance, a standard tokenizer
fragments a domain-specific term like nutUSpaldingWallFunction into sub-words (e.g., ‘nut’, ‘U’, ‘Spalding’).
A vector search then matches these linguistic fragments instead of the term’s physical meaning, leading tocontext
poisoning. This conflation of semantic relevance with physical validity causes the agent to retrieve syntactically
plausible but theoretically disastrous configurations.
Recently, ChatCFD [ 9] overcome the limitations of unstructured vector-based RAG by introducing structured RAG
in autonomous CFD agents using DeepSeek-R1/V3. Instead of using semantic embeddings, ChatCFD employs a
JSON-based retriever to match queries against explicit solver types and fetch exact file templates. This method is
highly effective for basic CFD tasks, achieving ∼80% accuracy by eliminating syntax errors. However, this reliance on
templates creates a new bottleneck: Template Rigidity. ChatCFD performs well only when a matching tutorial case
exists. For practical, non-tutorial scenarios that require novel combinations of physical features (such as solvers and
turbulence models), the system cannot build a valid solution, and its accuracy drops to ∼30%. While ChatCFD solves
the syntactic problem, it fails to bridge the deeper physical and numerical gap needed for stable CFD agents.
Addressing this limitation requires moving beyond retrieving static templates to actively enforcing dynamic constraints.
We introducePhyNiKCE(Physical andNumericalKnowledgeContextEngineering), a neurosymbolic agentic
framework designed for autonomous CFD. Unlike ChatCFD, which relies on retrieving fixed setup file templates,
PhyNiKCE dynamically assembles valid simulation contexts by treating configuration as a Constraint Satisfaction
Problem (CSP). The framework ensures that the simulation context provided to the agent is strictly validated against
established physical laws and numerical stability criteria before generation. This decoupling of neural planning from
symbolic validation allows the agent to generalize to novel scenarios while maintaining the stability guarantees essential
for robust CFD automation.
Figure 1: High-level control loop of the PhyNiKCE framework. The LLM-driven Agent parses multi-modal user
inputs (literature, grids, instructions) and plans the simulation. The Symbolic Knowledge Engine acts as a deterministic
guardrail, validating the agent’s plan against physical constraints before execution in OpenFOAM. An autonomous
reflection loop enables the agent to correct runtime errors, ensuring a physically valid flow field.
2

APREPRINT- FEBRUARY13, 2026
To visualize this architecture, Figure 1 illustrates the high-level control loop of the PhyNiKCE framework. The core of
this architecture is the Symbolic Knowledge Engine, which acts as a deterministic guardrail between the neural planner
and the solver. This engine is composed of two primary modules: a Symbolic Knowledge Base containing structured
domain constraints, and a Deterministic RAG Engine that queries these constraints to validate the LLM-generated
simulation parameters. By interposing this validation step, the system ensures that the final execution in OpenFOAM is
numerically stable and physically consistent with the user’s intent, effectively preventing unstable CFD setups.
The main contributions and outcomes of this research are summarized as follows:
•Neurosymbolic Constraint Enforcement:We propose a Deterministic RAG Engine that decouples neural planning
from symbolic validation. This engine introduces five specialized retrieval strategies that strictly enforce rigid
multi-physics couplings, effectively eliminating the context poisoning inherent in vector-based RAG.
•Superior Accuracy on Complex Tasks:In rigorous testing on practical, non-tutorial literature cases, PhyNiKCE
gains a 96% relative improvement over SOTA baseline, increasing the accuracy from 26% to 51%. This confirms that
symbolic grounding is essential for overcoming the template rigidity of previous structured RAG systems.
•Inference Efficiency:By front-loading LLM inference into a knowledge-driven initialization, the framework reduced
per-case autonomous error reflection loops by 59% and lowered LLM token consumption by 17%. This challenges the
assumption that neurosymbolic reasoning adds overhead, showing instead that physical consistency is a prerequisite
for efficiency.
•Auditability and Trust:Unlike opaque black-box vector retrieval, our deterministic RAG mechanism provides a
transparent, traceable decision path for every configuration parameter. This establishes a paradigm of Trustworthy AI
suitable for certification in regulated control and automation.
The remainder of this paper is organized as follows. Section 2 presents the background and related works, and Section 3
details the proposed neurosymbolic framework. The experimental protocol, results and discussion are presented in
Section 4 and Section 5, respectively. Finally, Section 6 summarizes our findings and outlines future research directions.
2 Background and Related Works
2.1 LLM Agents
The development of LLM Agents, which integrate perception, reasoning, and action planning [ 10,11], was catalyzed
by breakthroughs in prompt engineering that unlocked complex reasoning capabilities. The seminal Chain-of-Thought
(CoT) prompting [ 12] enabled models to solve problems by generating intermediate steps, a concept later enhanced by
frameworks like Reflexion [ 13], which introduced dynamic memory and self-correction to facilitate active problem-
solving.
To bridge reasoning with execution, architectures like ReAct [ 14] were developed to synergize “Reasoning” and
“Acting,” allowing agents to interact with external environments. This capability was further expanded by models
such as Toolformer [ 15] and HuggingGPT [ 16], which learned to autonomously use external APIs and specialized
AI models as tools. These innovations have culminated in robust agentic frameworks with diverse applications. For
instance, MetaGPT [ 17] automates software development workflows, domain-specific agents like ChemCrow [ 18] and
SWE-agent [ 19] handle complex tasks in chemistry and software engineering, and V oyager [ 20] demonstrates lifelong
learning in open-ended environments, showcasing the broad potential of autonomous agents.
2.2 CFD
CFD is a branch of fluid mechanics that utilizes numerical analysis to solve the Navier-Stokes equations governing
fluid flow. The current software landscape is bifurcated into commercial and open-source ecosystems. In the com-
mercial sector, Ansys Fluent and Siemens Star-CCM+ dominate the market, offering robust Graphical User Interfaces
(GUIs), integrated meshing tools, and extensive customer support, albeit at high licensing costs and with limited code
transparency. Conversely, the open-source community offers flexible alternatives such as SU2 [ 21], which is widely
adopted for aerodynamic shape optimization and compressible flows, and OpenFOAM [ 22], which serves as the de
facto standard for general-purpose research in both academia and industry.
Among these, OpenFOAM distinguishes itself through its modular C++ architecture, utilizing the Finite V olume Method
(FVM) to solve complex continuum mechanics problems ranging from multiphase flows to combustion. However, unlike
its commercial counterparts, OpenFOAM is notorious for its steep learning curve and lack of a native GUI. A standard
simulation requires the precise configuration of a rigid directory structure—typically comprising 0/,constant/ , and
system/folders—and the manipulation of numerous text-based dictionary files. Users must explicitly define:
3

APREPRINT- FEBRUARY13, 2026
•Physical Fields:Initial and boundary conditions for dependent variables such as velocity (U).
•Transport Properties:Physical constants and turbulence models (e.g.,k−ωSST [23]).
•Numerical Schemes:Discretization methods defined infvSchemesand linear solvers infvSolution.
Crucially, these configurations are physically and numerically coupled. For example, selecting a compressible solver
mandates thermodynamic boundary conditions that differ fundamentally from those used in incompressible flows. This
environment is strictly deterministic: a single syntax error or a mismatch between the numerical scheme and boundary
condition results in simulation failure. This rigidity presents a fundamental challenge for LLM Agents, which operate
probabilistically and often struggle to maintain the strict, long-range inter-file consistency required by the OpenFOAM
solver.
2.3 Autonomous CFD Agents
The complexity of CFD has spurred the development of specialized agents, particularly for OpenFOAM, whose
text-based dictionary files are well-suited for LLM manipulation. Early frameworks like MetaOpenFOAM [ 24] and
OpenFOAMGPT [ 25] pioneered the use of natural language to generate simulation setups. More recent systems
have expanded this into end-to-end automation. For instance, Foam-Agent [ 26] automates the entire pipeline from
pre-processing to execution on High-Performance Computers, while ChatCFD [ 9] introduced multimodal inputs and
error-reflection, achieving high operational success on benchmark tutorials.
Despite these operational advances, a critical gap remains in physical fidelity. Current agents, including CFDAgent [ 27]
and ChatCFD, struggle with a Semantic-Physical Disconnect. While capable of generating syntactically correct code
for standard tutorial cases, their accuracy plummets to approximately 30% when facing complex, literature-derived
scenarios [ 7]. This failure stems from an over-reliance on standard RAG mechanisms that conflate semantic similarity
with physical consistency.
This conflation leads to context poisoning: the retrieval of syntactically plausible but physically incompatible configu-
rations. For example, a vector-based RAG may fetch a “cyclone” simulation case when queried for “homogeneous
isotropic turbulence” based on linguistic proximity, despite the two flow regimes being physically incompatible [ 28].
Furthermore, vector similarity fails to distinguish between polymorphic boundary conditions—such as totalPressure ,
which requires entirely different parameter definitions for subsonic versus supersonic regimes. Because probabilistic
LLMs cannot enforce the deterministic logic governing these configurations, such retrieval errors propagate throughout
the workflow, undermining reliability. To address this, the current study utilizes the ChatCFD architecture as a baseline
to demonstrate how a neurosymbolic approach can bridge this accuracy gap.
2.4 Neurosymbolic AI and Structured Retrieval
The Neurosymbolic paradigm offers a solution to the context poisoning inherent in vector-based RAG by integrating
the generative flexibility of neural networks with the logical rigor of symbolic reasoning. This approach modernizes
classical Rule-Based Expert Systems [ 29] to ground the stochastic behavior of LLM Agents, combining the robustness
of neural models with the interpretability of symbolic logic.
Recent advancements in structured retrieval validate this hybrid architecture. Frameworks like SymRAG [ 30],
RuleRAG [ 31], and StructRAG [ 32] demonstrate that transforming unstructured data into structured rules or knowledge
graphs significantly enhances logical consistency. These studies show that for high-stakes domains, retrieval must
evolve from probabilistic vector matching to deterministic, rule-guided execution. This principle forms the theoretical
foundation of the PhyNiKCE framework.
4

APREPRINT- FEBRUARY13, 2026
Figure 2: Architecture of the PhyNiKCE framework. The Symbolic Knowledge Engine (top) performs offline
Knowledge Base Construction (Stage 0) to transform raw tutorials into a structured Symbolic Knowledge Base for the
Deterministic RAG Engine. The Autonomous CFD Agent (bottom) executes the simulation workflow: parsing User
Input (Stage 1), performing Case File Initialization (Stage 2) via symbolic queries, and engaging in Error Reflection
(Stage 3) to autonomously resolve runtime failures. Red arrows indicate the injection of physically and numerically
knowledgeable context.
5

APREPRINT- FEBRUARY13, 2026
3 Methods in the Agentic Framework
This research addresses the Semantic-Physical Disconnect inherent in applying probabilistic LLMs to deterministic
CFD simulations. To bridge this gap, we introduce PhyNiKCE. As illustrated in Figure 2, the architecture decouples the
agentic framework into two components:
1.The Neural Component:Implemented as the Autonomous CFD Agent, this LLM-driven agent handles intent
extraction, high-level planning, and syntactic generation.
2.The Symbolic Component:Implemented as the Symbolic Knowledge Engine, this component is responsible for
enforcing physical and numerical consistency. A Deterministic RAG Engine is implemented to apply rule-based
context retrieving strategies.
This architecture follows aGuardrailparadigm: the Neural Component proposes a simulation intent and generates case
setups, while the Symbolic Component enforces the physical and numerical constraints necessary to realize that intent.
This separation prevents the generation of invalid physical models or incompatible numerical schemes.
To provide a clear methodological breakdown of this complex system, the remainder of this section is organized
as follows: Section 3.1 details the offline construction of the Symbolic Knowledge Base; Section 3.2 defines the
algorithmic logic of the five specialized retrievers within the Deterministic RAG Engine; and Section 3.3 describes the
Autonomous CFD Agent.
3.1 Knowledge Acquisition and Representation
The PhyNiKCE framework is built upon the OpenFOAM [ 22] simulation engine. As the leading open-source CFD
software, OpenFOAM handles complex fluid mechanics problems but lacks the centralized documentation of commercial
packages. Instead, its knowledge is implicitly encoded within a corpus of approximately 400 tutorial cases. These
tutorials, which define valid simulation setups through text-based dictionary files, are accessible to LLMs and have
been used by prior CFD agents [ 24,25,7] to construct knowledge bases. However, this knowledge corpus is sparse;
for example, no single tutorial demonstrates the combination of the rhoCentralFoam solver with certain turbulence
models. Capturing the latent physical rules from these disjointed examples therefore requires a structured knowledge
acquisition process.
3.1.1 Stage 0: Knowledge Base Construction
The foundation of the Knowledge Engine is the Symbolic Knowledge Base. Unlike standard RAG systems that rely on
vector embeddings of unstructured text—a process that often obscures precise numerical relationships—PhyNiKCE
employs an Ontological Structuring approach. The Knowledge Base Builder systematically parses the raw tutorial
corpus and source code of OpenFOAM to construct the Symbolic Knowledge Base through three steps:
1. Syntactic Normalization:The builder first parses the raw dictionary files from the OpenFOAM tutorial corpus
and converts them into a standardized JSON format. This syntactic normalization is a critical preliminary step. As
illustrated in Figure 3, transforming OpenFOAM’s native dictionary syntax into the rigid, machine-readable structure of
JSON minimizes the risk of parsing errors and syntactic hallucinations when the LLM generates new configuration files.
We define the completeCase Setupas the hierarchical collection of these JSON-converted dictionaries.
The OpenFOAM Case Setups are generally classified into the following four different types
•Field Initializers (in 0/):Defines initial and boundary conditions (IC/BCs) for primary variables (e.g., velocity U).
•Physical Properties (in constant/ ):Defines transport models, thermodynamic properties, and turbulence proper-
ties.
•Numerical Details (in system/ other than controlDict ):Defines discretization schemes ( fvSchemes ) and linear
solver algorithms (fvSolution).
•Temporal Control ( system/controlDict ):Defines the Courant-Friedrichs-Lewy number, total simulation time
and so on.
An example of an OpenFoam tutorial case as represented in the knowledge base is shown in A. This research focuses on
the core physical and numerical configuration files. Components related to execution logistics, such as grid generation
dictionaries (e.g., blockMeshDict ) and parallel processing configurations ( decomposeParDict ), are excluded as they
are highly geometry-specific or not central to the physics setup. To create a clean and generalizable knowledge base,
we also perform several data normalization steps. Decorative headers and other non-functional content are removed.
6

APREPRINT- FEBRUARY13, 2026
Figure 3: Example of converting an OpenFOAM FoamFile dictionary to LLM-friendly JSON format.
Furthermore, case-specific, non-uniform field initializations, which consist of large numerical arrays, are excluded as
they lack generalizability. These normalization techniques create a compact, high-density knowledge base, reducing
context size and minimizing the risk of LLM hallucination.
2. Physical Feature Identification:For each OpenFOAM case, we identify three physical features that define a
simulation’s physical backbone:
•The Solver ( msol):Defines the set of partial differential equations (PDEs) and the algorithm used to solve them (e.g.,
simpleFoamfor steady-state incompressible flow). This is the most important feature.
•The Turbulence Model (m turb):Defines the closure equations for the turbulence term.
•The Compressibility( mcomp ):Classifies the flow as either incompressible or compressible, a fundamental flow
feature with drastically different physical and numerical requirements. This derived feature acts as a physical
constraint for the downstream cascading fallback algorithm, preventing the mixing of incompatible contexts.
Additionally, the builder infersm comp directly fromm sol.
3. Hybrid Knowledge Augmentation:The builder addresses parameter polymorphism in boundary conditions
(BCs)—where a single BC type requires different parameters based on the flow physics. For instance, the
totalPressure inlet needs only relative pressure for incompressible flow but requires absolute pressure and thermo-
dynamic coefficients (like γ) for compressible flow. Since tutorial cases do not cover all variations, the builder extracts
authoritative rules directly from OpenFOAM’s C++ source code headers ( *.Hfiles). This augmentation provides the
LLM with the necessary theoretical constraints to correctly configure complex BCs, even without a direct tutorial
example.
Formally, we define the OpenFOAM tutorial knowledge base Knot as a collection of documents, but as a set of
structured tuples:
K= (p i,ci)N
i=1(1)
where pi=⟨m sol, mturb, mcomp⟩represents physics features acting as the retrieval key, and cirepresents the associated
setup files. This structure fundamentally shifts the retrieval paradigm away from a “Data V olume” approach that seeks
textually similar documents, toward an “Information Density” approach that extracts the precise rule csatisfying the
physical constraints inp. This shift enables high-fidelity retrieval from a compact dataset.
3.2 Deterministic RAG Engine
The core of PhyNiKCE is the Deterministic RAG Engine. Unlike standard vector similarity searches that retrieve
semantically similar documents, this engine treats context retrieval as a CSP. Its function is to retrieve physically and
numerically consistent context, ensuring the validity of the simulation setup.
7

APREPRINT- FEBRUARY13, 2026
Figure 4: The dispatch logic of the Deterministic RAG Engine. The diagram maps the Agent’s specific case file
generation and correction tasks to the specialized retriever. Left (Stage 2: Case File Initialization): For the case file
initialization, target case files are routed to the appropriate retriever based on their physical dependencies. Right (Stage
3: Error Reflection): Execution failures are routed to retrievers to generate targeted fixes.
To resolve the CSP effectively, a monolithic retrieval approach is insufficient due to the structural heterogeneity of
CFD configuration files. A dictionary defining linear solver tolerances ( system/fvSolution ) follows a fundamentally
different logic than one defining complex BCs ( 0/U). Consequently, the engine functions as a deterministic dispatcher.
As illustrated in Figure 4, the system employs five specialized retrieval strategies. The dispatch logic is governed by the
target data structure:
•For Physical Properties and Inferred BCs:The Cascading Fallback Retriever is engaged to handle loose parameter
matching while avoiding physical inconsistency.
•For Numerical Schemes:The All-Model Retriever is triggered to enforce strict multi-physics compatibility.
•For Solver Controls:The Data-driven Template Retriever is used to aggregate statistical best practices.
•For Known BCs:The Multi-Source Retriever combines syntax with source-code constraints.
•For Error Correction:The Heuristic Keyword Retriever is dispatched to locate targeted fixes during runtime
failures.
This specialized routing ensures that the inference mechanism aligns with the underlying physics of the specific file
being generated.
We define the global notation for this section as follows:
•K: The OpenFOAM tutorial knowledge base (Eq. 1), which contains all OpenFOAM tutorial cases indexed by their
physical features.
•Q: The Query, specifying the target physical constraints (solver, turbulence model, and compressibility).
•S: The Target Setup Descriptor, a key identifying the specific configuration component needed, such as a file (e.g.,
system/fvSchemes) or a boundary condition type (e.g.,inlet).
•C: The Context Set, which is the collection of configuration examples retrieved for the agent.
8

APREPRINT- FEBRUARY13, 2026
•Nmax: The context cardinality threshold (typically 3 to 5), which limits the number of discrete examples in the
context set C. This ensures the agent receives complete, structured examples rather than a fragmented stream of
tokens.
A key advantage of this deterministic architecture is itsauditability. In industrial applications, the origin of every
simulation parameter must be traceable to meet engineering standards. PhyNiKCE applies the following five retrievers
to ensure auditability.
3.2.1 Cascading Fallback Retriever
Algorithm 1: Cascading Fallback Retrieval
Input: Target physical features M={m sol, mturb}, Target setup descriptor S, OpenFOAM tutorial knowledge base
K, Cardinality thresholdN max
Output:Context SetC
1:Phase 1: Initialization
2:m comp←IDENTIFYCOMPRESSIBILITY(m sol)
3:LetQbe the sequence of query configurations⟨q 1, q2, . . . , q 6⟩defined as:
4:q 1← {m sol, mturb,S}▷Strict Match
5:q 2← {m sol,S}▷Solver Dominance
6:q 3← {m turb, mcomp,S}▷Relax Solver to Compressibility
7:q 4← {m turb,S}▷Turbulence Enforced
8:q 5← {m comp,S}▷Compressibility Enforced
9:q 6← {S} ▷Setup Only
10:Phase 2: Retrieval and Constraint Relaxation
11:for allq∈ Qdo
12:C ←SEARCH(K, q)
13:ifC ̸=∅then
14:if|C|> N max then▷Enforce cardinality constraints
15:C ←DOWNSAMPLE(C, N max)
16:end if
17:returnC
18:end if
19:end for
20:return∅
This retriever is primarily employed for initializing physical properties, transport models, and inferred BCs. It is
specifically designed to address the problem of combinatorial sparsity, where the Knowledge Base often lacks a
single tutorial case that perfectly matches a complex query (e.g., the combination of the rhoCentralFoam solver
with a k−ϵ turbulence model). To prevent retrieval failure and subsequent physical hallucination, this strategy
implements a hierarchical relaxation of constraints, overcoming the local scope limitations of prior heuristic approaches
in ChatCFD [ 9]. As detailed in Algorithm 1, this retriever addresses the common issue of a sparse knowledge base,
where no single tutorial case perfectly matches a user’s query (e.g., a specific solver combined with a specific turbulence
model). Instead of failing or providing physically inconsistent context, the retriever systematically relaxes the search
constraints in a predefined physical hierarchy to find the best available match.
The query sequence is as follows:
1.Strict Match:First, it searches for a case matching the exact solver (m sol) and turbulence model (m turb).
2.Solver-Dominant Match:If no exact match is found, it relaxes the turbulence constraint and searches for any case
using the target solver. This prioritizes numerical compatibility, as the solver dictates the core equations.
3.Physics-Dominant Match:If the solver search fails, it relaxes the solver constraint but enforces the fundamental
flow physics (compressibility, mcomp ) and the turbulence model. This is a critical guardrail to prevent context
poisoning, such as applying an incompressible setup to a compressible flow problem.
The relaxation continues through further steps, ensuring that a physically relevant, albeit partial, match is always
found. This hierarchical approach is based on the principle of physical modularity: it understands that a solver is more
fundamental than a turbulence model, and a flow regime (compressible vs. incompressible) imposes non-negotiable
9

APREPRINT- FEBRUARY13, 2026
constraints. By relaxing constraints in a physically-aware order, the retriever maximizes the utility of the sparse
knowledge base without retrieving incompatible configurations.
3.2.2 All-Model Retriever
Discretization schemes, defined in the system/fvSchemes file, must be compatible with all active physical models. A
scheme that is valid for the solver’s equations ( msol) may cause numerical instability when used with a specific turbu-
lence model ( mturb). For example, an unbounded scheme like Gauss linear might be suitable for the incompressible
turbulence term but will likely cause a compressible simulation to fail, as it cannot properly handle shock waves. To
resolve these complex interdependencies, the All-Model Retriever is used.
This retriever’s inputs and outputs are identical to those of the Cascading Fallback Retriever. Its process is as follows:
1.It first attempts a strict search for a single tutorial case that perfectly matches both the target solver ( msol) and
turbulence model (m turb).
2.If no exact match exists, the retriever initiates two parallel searches to find the best available configurations for each
physical model independently:
•Solver-Dominant Branch:Finds schemes compatible with the solver’s governing equations, using msolas the
primary key and falling back to the compressibility type (m comp ).
•Turbulence-Dominant Branch:Finds schemes compatible with the turbulence model’s equations, using
mturbas the primary key.
Each branch uses the Cascading Fallback algorithm to ensure a physically relevant match is found. The final context
is created by taking the union of the results from both branches. This strategy enables the agent to synthesize a valid
fvSchemes dictionary by combining compatible schemes from different tutorial cases, ensuring that every equation in
the simulation is assigned a stable and appropriate discretization scheme.
3.2.3 Data-Driven Template Retriever
Algorithm 2: Data-driven Template Retrieval
Input: Target phyiscal features M={m sol, mturb}, Target setup descriptor S, Knowledge base K, Significance
thresholdτ
Output:T final : Canonical setup template
1:P merged ← ∅▷Global probability map
2:for eachm i∈ Mdo
3:K sub← {c∈ K |csatisfiesm i}▷Filter matching cases
4:ifK sub̸=∅then
5:N← |K sub|
6:C raw←EXTRACTKEYS(K sub, S)
7:R i← ∅▷Local probability map form i
8:Phase 1. Calculate Local Probabilities
9:for eachk∈keys(C raw)do
10:R i(k).rate←count(k)/N
11:R i(k).values←NORMALIZE(values(k), N)
12:end for
13:Phase 2. Merge via Union-Max Strategy
14:ifP merged =∅then
15:P merged ← R i
16:else
17:K union←keys(P merged )∪keys(R i)
18:for eachk∈ K union do
19:p old← P merged (k).rate
20:p curr← R i(k).rate
21:ifp curr> poldthen
22:P merged (k)← R i(k)▷Update for stronger feature signal
23:end if
24:Note:Ifp old≥pcurr, maintain existing entry.
10

APREPRINT- FEBRUARY13, 2026
25:end for
26:end if
27:end if
28:end for
29:T final←COLLAPSEANDREFINE(P merged , τ)▷See Algorithm 3
30:returnT final
For linear solution setup in system/fvSolution , standardizing parameters is more effective than retrieving specific
tutorial cases. Specific cases often contain user-specific variabilities—such as inefficient relaxation coefficient or solver
types—that are detrimental to general application. Consequently, this retriever constructs a canonical setup template by
aggregating statistical probabilities across the entire Knowledge Base.
The core logic, detailed in Algorithm 2, employs a union-max aggregation strategy. We define this as a constructive
heuristic where the system takes the mathematicalunionof all configuration keys required by the active physics,
and assigns the value with themaximumstatistical probability to each key. The algorithm iterates through the active
feature-specific control subsets F—representing the linear solver configurations required by the active physical models
(e.g., the specific solver settings for thekandϵequations dictated by the turbulence model).
This process results in a superset template: if the turbulence model requires a parameter (e.g.,pFinal) that the solver
model ignores, the union logic ensures the parameter is retained, guaranteeing compatibility with all active physics. A
detailed illustration of this merging process is provided in B.
Algorithm 3: Template Collapsing and Refinement
Input:Merged probability mapP merged , Significance thresholdτ
Output:Final configuration templateT
1:T ← ∅
2:for eachk∈keys(P merged )do
3:Phase 1: Thresholding
4:ifP merged (k).rate≤τthen
5:continue
6:end if
7:Phase 2: Atomic Selection
8:ifP merged (k)is Atomicthen▷ kis a compound setup
9:T(k)←arg max v(freq(v)|v∈ P merged (k).values)
10:else
11:S selected ← ∅
12:V dominant ← ∅
13:S sub←subkeys(P merged (k))
14:for eachs∈ S subdo
15:ifrate(s)> τthen
16:S selected ← S selected ∪ {s}
17:v max←arg max v(values(s))
18:V dominant ← V dominant ∪ {v max}
19:end if
20:end for
21:Final Assignment
22:for eachs∈ S selected do
23:T(k)(s)←arg max v(values(s))
24:end for
25:end if
26:end for
27:returnT
To filter statistical noise from this superset, the engine applies a two-phase refinement logic in Algorithm 3:
1.Thresholding:First, a significance threshold ( τ) filters out statistically insignificant keys (e.g., custom user variables
appearing in<30%of cases).
11

APREPRINT- FEBRUARY13, 2026
2.Atomic Selection:For keys satisfying the threshold, the algorithm identifies the most frequent configurations.
Crucially, complex configuration blocks (see B.3) are treated as Atomic Units. Numerical parameters are highly
interdependent; for instance, a specific field equation often relies on the efficiency of a specific linear solver. To
avoid creating incoherent setups by mixing mismatched parameters (e.g., averaging relaxation coefficient across
different cases), the entire dictionary block is treated as an indivisible value and selected via awinner-takes-all
approach to ensure a proven, self-consistent numerical strategy.
3.2.4 Multi-Source Retriever
BCs present a dual challenge for generative models. They exhibit parameter polymorphism, where the required input
fields for a single boundary type change based on the physical regime (e.g., subsonic vs. supersonic). Failure to capture
these conditional dependencies leads to context poisoning, where an agent retrieves a valid incompressible template
that lacks the thermodynamic coefficients required for a compressible simulation.
To resolve this, the Multi-Source Retriever implements a dual-path knowledge augmentation strategy. It synthesizes
concrete implementation patterns with authoritative theoretical constraints. Formally, the final context CMSgenerated
for a target boundary conditionbis defined as the union of two distinct retrieval streams:
CMS(b) =C example (b)∪ C guidance (b)(2)
where:
1.Syntactic Scaffolding ( Cexample ):This stream provides the “code skeleton.” It invokes the Cascading Fallback
Retriever to find physically similar boundary setups from the tutorial cases. This ensures the agent receives a
syntactically valid OpenFOAM dictionary entry (e.g., correct brackets, keywords, and value formatting).
2.Theoretical Constraints ( Cguidance ):This stream provides the “physical rules.” It queries the authoritative source
code constraints stored in the Knowledge Base (as defined in Section 3.1). Unlike the example stream, this retrieves
the explicit requirements for mandatory coefficients.
This augmentation allows the Deterministic RAG Engine to resolve ambiguity. Returning to the totalPressure
example: while Cexample provides the standard dictionary structure, Cguidance injects the specific constraint that the
adiabatic index ( γ) and the total pressure ( p0) must be defined if the solver is compressible. By merging these streams,
the agent receives a context containing both a working code block and the logical constraints required to adapt that
block to the current flow regime, effectively bridging the gap between syntax and physics.
3.2.5 Heuristic Keyword Retriever
Algorithm 4: Heuristic Keyword Retrieval
Input: Target models M={m sol, mturb}, Target setup descriptor S, Search keyword k, Symbolic knowledge base
K, Cardinality thresholdN max
Output:Result setR
1:m comp←IDENTIFYCOMPRESSIBILITY(m sol)
2:Phase 1: Keyword Heuristic Generation
3:LetK seqbe the ordered sequence of search keys initialized as⟨k⟩
4:kunq←StripQuotes(k)▷Stripe Quotes
5:ifk unq̸=kthen
6:Appendk unqtoK seq
7:end if
8:knorm←RemoveSpaces(k unq)▷Remove Spaces
9:ifk norm̸=kunqthen
10:Appendk norm toK seq
11:end if
12:Phase 2: Priority Query Sequence Construction
13:LetQ seqbe an empty ordered sequence
14:for allk∈ K seqdo
15:Define query variants for keywordk:
16:q 1← {m sol, mturb} ∪ {k, S}▷Strict Match
17:q 2← {m comp, mturb} ∪ {k, S}▷Relax Solver
12

APREPRINT- FEBRUARY13, 2026
18:q 3← {m turb} ∪ {k, S}▷Turbulence Enforced
19:q 4← {k, S}▷Setup & Keyword Only
20:Append⟨q 1, q2, q3, q4⟩toQ seq
21:end for
22:Phase 3: Retrieval and Constraint Relaxation
23:for allq∈ Q seqdo
24:R ←Search(K, q)
25:ifR ̸=∅then
26:if|R|> N max then
27:R ←Downsample(R, N max)
28:end if
29:returnR
30:end if
31:end for
32:return∅
This retriever is dispatched during Error Reflection (Stage 3) to resolve complex configuration errors. These complex
configuration errors, which are not related to boundary conditions or linear solvers, account for approximately 90% of
simulation failures (see Section 5.5).
As detailed in Algorithm 4, the retriever uses a multi-stage process to find a targeted fix:
1.Keyword Normalization:It first extracts an error-specific keyword (e.g., a diverging term like div(phi,U) ) from
the execution log. To ensure a reliable search, it applies heuristic normalization to clean the keyword, removing
non-standard formatting (e.g., extra whitespace or quotes) that may have been introduced by the LLM during
generation.
2.Physically-Constrained Search:Using the normalized keyword, the retriever searches the Knowledge Base for a
valid configuration. It follows a cascading logic similar to the Cascading Fallback Retriever, starting with a strict
search (matching both solver and turbulence model) and progressively relaxing constraints to ensure the retrieved fix
is physically compatible with the simulation.
3.Targeted Snippet Extraction:Instead of retrieving an entire file, the retriever performs a search to locate the
smallest self-contained configuration block that contains the target keyword. This provides the agent with a concise,
high-density context focused exclusively on the erroneous parameter, as demonstrated in C.
3.3 Autonomous CFD Agent
The Autonomous CFD Agent orchestrates the simulation process through three operational stages, guided by the
Symbolic Knowledge Engine. The agent adapts the architecture of ChatCFD [ 9], but its core logic is fundamentally
enhanced by replacing the prior heuristic retrieval mechanisms with the Deterministic RAG Engine. This integration
provides physically-grounded context for both case initialization (Stage 2) and error reflection (Stage 3). To leverage
this high-quality context, the agent employs specialized instruction protocols tailored to each configuration file, ensuring
precise and reliable generation.
3.3.1 Stage 1: User Input
In Stage 1, the agent parses the user’s multi-modal input—including natural language, PDFs, and mesh files—to extract
the core physical features of the simulation. These features, such as the solver ( msol), turbulence model ( mturb),
and key properties (e.g., inlet velocity), form the initial query Minit={m sol, mturb}for the Deterministic RAG
Engine, establishing the high-level constraints for the simulation. This input parsing stage is adapted from the ChatCFD
framework [9] and is not detailed further.
3.3.2 Stage 2: Case File Initialization
In Stage 2, the agent extracts the necessary IC/BCs and physical parameters from the provided literature or user input.
These are stored in the Symbolic Knowledge Base to ensure the agent maintains a consistent physical definition of the
target case. When generating specific configuration files, the engine dispatches the appropriate specialized retriever
based on the file’s function, as illustrated in Figure 4:
•For Physical Properties and BCs:The Cascading Fallback and Multi-Source Retrievers are employed to locate
physically compatible settings, ensuring validity even when exact model combinations are sparse.
13

APREPRINT- FEBRUARY13, 2026
•For Numerical Schemes and Solvers:The All-Model and Data-driven Template Retrievers are utilized to guarantee
numerical stability by enforcing multi-physics compatibility and applying statistically proven configurations.
Armed with this physically grounded context, the agent proceeds to generate the case files. To prevent hallucination
and ensure deterministic execution, the system avoids unstructured zero-shot queries. Instead, it utilizes aStructured
Instruction Protocol, defined in Listing 1. This protocol acts as a rigid template that injects validated symbolic
knowledge through dynamic constraint slots:
•case_ic_bc andcase_physical_properties : These slots populate the specific physical constraints extracted
earlier, ensuring the simulation aligns strictly with the target specifications.
•retrieval_contents : This slot injects the validated syntax patterns, templates, and guidance (only for BCs)
provided by the Deterministic RAG Engine, forcing the LLM to adhere to correct OpenFOAM standards.
•headerandfile_name: These ensure the output complies with the required file format structure.
Listing 1: Structure Instruction Protocol for Case File Initialization (in Stage 2)
1SYSTEM_DEFINITION :
2"You are an Expert Computational Fluid Dynamics Engineer specializing in OpenFOAM
. Your objective is to generate a syntactically correct and physically valid
dictionary for the target file : ’{ file_name } ’."
3
4SYMBOLIC_CONTEXT_INJECTION :
5"The Symbolic Context Engine has extracted the following constraints . You must
adhere to these rigid physical parameters :
61. Initial & Boundary Conditions : { case_ic_bc }
72. Physical Properties : { case_physical_properties }
83. Validated Reference Samples or Guidelines : { retrieval_contents }"
9
10INFERENCE_STRATEGY :
11" Follow this deterministic logic flow :
121. Analyze Physical Relationships : Examine the boundary condition of fields ( like
U, p, T) and physical features to understand the simulation ’s physics .
132. Consult Reference Samples : Use the provided reference files as a guide .
Analyze common patterns , similar physical setups (e.g., RANS vs. LES,
compressible vs. incompressible , solver ).
143. Make Logical Selections : Based on your analysis , determine the most suitable
setups type ."
15
16OUTPUT_CONSTRAINTS :
17"- The final answer must properly include the header contents : { header }.
18- Output ONLY the complete file content inside a code block .
19- Do NOT include standard C++ decorated comments (e.g., the block starting with
‘/*-----...‘).
20- Do NOT add explanations or reasoning text ."
3.3.3 Stage 3: Error Reflection
If the simulation fails, the agent triggers an autonomous error-reflection loop, iterating up to 30 times to resolve the
issue. This process begins by parsing the execution log and configuration files to pinpoint the error’s location (e.g.,
system/fvSchemes ) and identify its root cause (e.g., an unstable div(phi,U) scheme). Based on this diagnosis, the
agent queries the Deterministic RAG Engine for specific, validated templates as show in Figure 4:
•For Boundary Condition Errors:The Multi-Source Retriever provides a corrected definition alongside physically
similar exemplar setups.
•For Linear Solver Errors:The Data-driven Template Retriever provides a canonical linear solver setups with
physical and numerical consistency.
•For Numerical Instabilities:The Heuristic Keyword Retriever supplies a precise, minimal code snippet to stabilize
the specific parameter, avoiding the risks associated with regenerating entire files.
The correction process is governed by the Diagnostic Protocol shown in Listing 2. Unlike standard reflection methods
that rely on the LLM to guess solutions, this protocol anchors the repair in external validation. It combines the
raw execution log with validated samples ( {retrieval_contents} ) retrieved by the engine. Crucially, it re-injects
14

APREPRINT- FEBRUARY13, 2026
the original physical constraints ( {case_ic_bc} ,{case_physical_properties} ) to ensure that any fix remains
consistent with the fundamental physics of the simulation. Following this diagnostic step, a secondary protocol applies
the generated advice to correct the erroneous file.Listing 2: Diagnostic Protocol for Error Reflection (in Stage 3)
1SYSTEM_DEFINITION :
2"You are an Expert Computational Fluid Dynamics Engineer specializing in OpenFOAM
. Your objective is to analyze the provided OpenFOAM Runtime Error and
erroneous file contents , and give advice on correcting the file {{ file_name }}
."
3
4SYMBOLIC_CONTEXT_INJECTION :
5"The Symbolic Context Engine has extracted the following constraints . You must
adhere to these rigid physical parameters :
61. Case Running Error : { running_error }
72. Erroneous File Contents : { file_content }
83. Initial & Boundary Conditions : { case_ic_bc }
94. Physical Properties : { case_physical_properties }
105. Validated Samples or Guidelines for Correction : { retrieval_contents }"
11
12INFERENCE_STRATEGY :
13" Follow this deterministic logic flow :
141. Provide a step -by - step fix. Ensure the advice addresses the error ’s technical
cause . The advice must be a string .
152. If the advice involves setting new values , the new values must be consistent
with those in the Initial & Boundary Conditions and Physical Properties ."
16
17OUTPUT_CONSTRAINTS :
18" Absolutely AVOID any elements including but not limited to:
19- Markdown code block markers (‘‘‘ or ’’’)
20- Extra comments or explanations
21- Unnecessary empty lines or indentation "
4 Validation Framework
4.1 Ablation Study Design
To evaluate the contribution of the PhyNiKCE framework, we conducted an ablation study comparing four agent
configurations. Each configuration uses the same agentic workflow but integrates progressively more advanced retrieval
mechanisms for case initialization (Stage 2) and error reflection (Stage 3).
The four configurations are:
•Standard Vector RAG:A naive RAG baseline. It uses zero-shot generation for initialization and a standard vector
search for error reflection. This method retrieves semantically similar but physically unverified text chunks.
•Baseline (SOTA ChatCFD [ 9]):This agent represents the current SOTA ChatCFD architecture. It uses zero-shot
generation for initialization and only a legacy heuristic retriever for error reflection. This retriever has two critical
limitations: (1) its retrieval logic has physical sparsity, ignoring turbulence model dependencies ( mturb), and (2) it
operates under a local scope restriction, confining its search to the solver’s tutorial sub-directory instead of the entire
OpenFOAM tutorial knowledge base.
•Partial PhyNiKCE:This agent uses the advanced Deterministic RAG Engine for initialization (Stage 2) but reverts
to the Baseline’s simpler heuristic retriever for error reflection (Stage 3).
•Full PhyNiKCE:The complete proposed system. It leverages the full suite of specialized, physically-aware retrievers
for both initialization and error reflection.
To ensure a fair architectural comparison, we reimplemented the ChatCFD workflow [ 9] using the Gemini-2.5-Pro/Flash
as the backbone, rather than its original DeepSeek-V3/R1. This alignment is critical: by keeping the underlying LLM
constant across all configurations, we isolate the performance gains attributable strictly to the proposed neurosymbolic
framework, eliminating the confounding variable of foundation model capability.
This ablation study was applied to all test cases in Table 1, allowing us to isolate the performance impact of PhyNiKCE’s
neurosymbolic components at each stage.
15

APREPRINT- FEBRUARY13, 2026
4.2 Validation Test Suite
Figure 5: Geometries, grids, and boundary types of CFD cases for the validation test. (a) the NACA 0012 airfoil
case [33]. (b) the Nozzle case [34].
To validate the agent, we used a test suite based on two canonical CFD cases: incompressible flow over a NACA
0012 airfoil [ 33] and compressible flow through a de Laval nozzle [ 34]. As shown in Figure 5, these cases represent
foundational incompressible and compressible flow regimes.
The validation matrix was made rigorous by testing multiple solvers ( simpleFoam ,rhoCentralFoam ,sonicFoam )
and RANS turbulence models ( k−ω SST [ 23], Spalart-Allmaras [ 35], and k−ϵ [36]) for each geometry. Unlike a
prior study [ 9] that sampled a subset of cases, this study evaluates the agent across a more comprehensive combinatorial
matrix of solvers and turbulence models. This exhaustive testing exposes edge-case failures—such as the combination
of compressible solvers with various turbulence models—that may be masked in less rigorous subsets.
Table 1: Experimental test matrix and number of accurate cases for the PhyNiKCE validation
Turbulence Test Accurate Cases
Case Solver Model Flow Param. Runs Base. Part. Full
NACA 0012simpleFoamSA AOA = 10° 10 5 7 8
k−ϵAOA = 10° 10 3 3 5
k−ωSST AOA = 10° 10 3 4 6
SA AOA = 15° 5 2 3 4
k−ϵAOA = 15° 5 0 1 2
k−ωSST AOA = 15° 10 4 6 6
NozzlerhoCentralFoamSA NPR = 3 10 3 4 5
k−ωSST NPR = 3 10 1 4 5
k−ϵNPR = 3 10 2 2 2
SA NPR = 2.3 5 1 1 3
sonicFoamSA NPR = 2.3 5 1 2 1
k−ωSST NPR = 2.3 5 1 2 2
k−ϵNPR = 2.3 5 0 1 2
SA = Spalart-Allmaras model. AOA = angle of attack. NPR = nozzle pressure ratio. Base. = Baseline, Part. = Partial PhyNiKCE.
Full = Full PhyNiKCE
The experimental matrix, summarized in Table 1, uses parameters from the referenced validation studies. It includes
variations for each geometry to test the agent’s precision:
•NACA 0012 Case (Incompressible):The agent configured simulations using simpleFoam with three RANS
turbulence models (k−ωSST, Spalart-Allmaras,k−ϵ) at two angles of attack (10◦and15◦).
•Nozzle Case (Compressible):The agent configured both a density-based solver ( rhoCentralFoam ) and a pressure-
based solver ( sonicFoam ) with the same three turbulence models at two nozzle pressure ratios (NPR = 3.0 and
2.3).
16

APREPRINT- FEBRUARY13, 2026
Each of the 13 unique setups was executed multiple times for consistency, resulting in 100 runs per full evaluation cycle.
The three primary configurations (Baseline, Partial PhyNiKCE, and Full PhyNiKCE) were tested across this matrix,
totaling 300 runs. A smaller pilot subset of 40 runs was used for the Standard Vector RAG configuration, which was
terminated early due to poor performance (5% accuracy). This study therefore comprises 340 experimental runs.
To rigorously test the agent’s generalization capabilities, standard OpenFOAM tutorial cases were excluded from the
validation suite. This prevents data leakage, as the knowledge base is built from these tutorials. Furthermore, the
baseline agent (ChatCFD [ 9]) already demonstrates high accuracy on these benchmarks. Therefore, this study focuses
on novel, practical cases from the literature to assess the agent’s ability to apply physical principles in new scenarios,
rather than its capacity for simple recall.
4.3 LLM Configuration
We utilized the Gemini-2.5 family of models via the Google AI Studio platform. Gemini-2.5-Pro was employed
for complex reasoning tasks, such as extracting simulation parameters from literature, initializing case setups, and
diagnosing root causes of execution errors. For more structured, low-complexity tasks, such as validating file syntax and
applying specific corrections, Gemini-2.5-Flash was used. The experimental validations described in Section 5 were
conducted using the standard API pricing: Gemini-2.5-Pro at $1.25 per million input tokens and $10.00 per million
output tokens; and Gemini-2.5-Flash at $0.30 per million input tokens and $2.50 per million output tokens.
5 Results and Discussion
This section presents the results of the ablation study from Section 4, which quantifies the performance of the PhyNiKCE
framework. The analysis compares four agent configurations: Standard Vector RAG, Baseline, Partial PhyNiKCE, and
Full PhyNiKCE. Performance is evaluated using two primary metrics:
•Execution Rate:The percentage of cases that run for at least 10 time steps or iterations without crashing.
•Accuracy:The percentage of cases that execute successfully and produce physically valid results that strictly match
the specifications in the referenced literature.
The evaluation is based on 340 experimental runs. Due to its low 5% accuracy, the Standard Vector RAG configuration
is excluded from detailed analysis to focus on the more meaningful comparison between the SOTA Baseline and
PhyNiKCE configurations. Unless stated otherwise, subsequent analyses are based only on the subset of cases that were
deemed accurate.
5.1 Overall Performance and Inference Efficiency
The primary results of the ablation study, summarized in Figure 6, demonstrate a monotonic improvement in agent
capability directly correlated with the degree of PhyNiKCE integration. The analysis reveals two fundamental
bottlenecks in autonomous simulation: a Stability Gap in generating executable configurations and a more severe
Physics Gap in ensuring their scientific correctness.
Figure 6(a) highlights the Stability Gap. The Standard Vector RAG agent, tested on a representative subset, exhibited
a low execution rate due to a “Granularity Mismatch,” where retrieving disjointed file snippets created internally
inconsistent directory structures. The Baseline agent (ChatCFD architecture) improved stability but still failed in 58%
of cases due to unresolvable parameter dependencies. In contrast, the Full PhyNiKCE agent achieved a robust 81%
execution rate, confirming that its Symbolic Knowledge Base effectively guarantees the structural validity required for a
simulation to launch.
However, executability does not imply correctness. The Physics Gap, illustrated by the Overall Accuracy in Figure
6(b), underscores the limitations of purely semantic retrieval. The Standard Vector RAG agent’s accuracy collapsed
to just 5%, a steep drop that empirically demonstrates context poisoning—the retrieval of semantically relevant but
physically incompatible configurations (e.g., mixing incompressible and compressible schemes). The Baseline agent
hits a “generalization wall,” plateauing at 26% accuracy on our validation suite. While slightly lower than the 30%
originally reported for ChatCFD [ 9], this deviation is expected and attributable to two factors. First, our validation
matrix (Table 1) specifically targets practical cases with more combinations of physical models than the original test
sets. Second, this baseline utilizes the Gemini-2.5 backbone to match the PhyNiKCE configuration. The fact that
the accuracy remains in the 25-30% range despite these changes confirms that the Semantic-Physical Disconnect is a
systemic architectural bottleneck, not merely an artifact of a specific LLM or dataset.
17

APREPRINT- FEBRUARY13, 2026
Figure 6: Performance comparison of the four ablation configurations. (a) Overall accuracy, (b) Average number of
reflection rounds per case, (c) Averaged token usage per case, (d) Averaged LLM inference cost per case.
PhyNiKCE surpasses the performance ceiling, achieving 51% accuracy—nearly double that of the Baseline. This
substantial increase is attributable to two distinct architectural mechanisms. First, the 58% relative improvement from
the Baseline (26%) to the Partial PhyNiKCE configuration (41%) isolates the impact of Stage 2: Case File Initialization.
This gain confirms that providing a physically consistent starting point is critical for minimizing fundamental setup
errors. Second, PhyNiKCE markedly enhances problem-solving efficiency during error correction. As detailed in Figure
6(c), the Baseline agent required an average of 22.11 reflection rounds per case, relying on iterative trial-and-error. The
Full PhyNiKCE agent, leveraging intelligent reflection with symbolic retrievers in Stage 3, reduced this burden by 59%
to just 9.06 rounds, demonstrating the Deterministic RAG Engine’s superior diagnostic capabilities.
Crucially, this enhanced efficiency translates directly into LLMs’ inference efficiency. Contrary to the common
assumption that RAG increases inference costs [ 37], our results in Figures 6(d) and 6(e) reveal the inverse. The
Baseline agent was the most token-intensive, consuming an average of 181.2k tokens per case. In contrast, the Full
PhyNiKCE agent was the most economical, requiring only 151.2k tokens. This indicates that the neurosymbolic system
is not an overhead but a strategic optimization. By ensuring a high-quality, physically consistent first attempt and
enabling targeted corrections, PhyNiKCE circumvents the compounding inference costs of the “trial-and-error” loops
characteristic of purely neural approaches.
5.2 Analysis of Inference Resource Allocation
Figure 7 breaks down the token allocation across the agent’s workflow, revealing the mechanism behind the efficiency
gains reported in Figure 6. The analysis demonstrates that PhyNiKCE achieves superior efficiency by strategically
reallocating inference resources from reactive, trial-and-error correction to proactive, knowledge-driven initialization.
The first phase, Stage 1, involves parsing user inputs to identify the target physical regime. As this stage functions
independently of the core PhyNiKCE logic, token consumption is consistent across all configurations, averaging
approximately 33.4k tokens per case.
The subsequent phases, comprising Case Parameter Collection and Case File Initiation (collectively Stage 2), exhibit a
distinct divergence in strategy. The Baseline configuration employs a minimalist approach, allocating fewer tokens to
these critical setup phases (21.7k and 12.4k, respectively) by utilizing a single, zero-shot LLM call for generation. In
contrast, the PhyNiKCE-enabled agents strategically “front-load” their inference effort, allocating a considerably larger
proportion of tokens (nearly 32.3k for Parameter Collection and 36.5k for File Initiation). This increased allocation
reflects the operation of the Deterministic RAG Engine, which systematically invokes specialized retrievers for each
configuration file and boundary field to construct comprehensive physical contexts.
18

APREPRINT- FEBRUARY13, 2026
Figure 7: Distribution of token usage of the primary three ablation configurations.
Table 2: Summary of Accuracy
Baseline Partial PhyNiKCE Full PhyNiKCE
Incompressible 0.34 0.48 0.62
Compressible 0.18 0.32 0.40
Spalart-Allmaras 0.30 0.43 0.53
k−ωSST 0.30 0.53 0.63
k−ϵ0.20 0.28 0.44
Overall 0.26 0.40 0.51
The efficacy of this proactive allocation is evident in the final phase, Stage 3: Error Reflection. The Baseline
agent, constrained by the low fidelity of its initial setup, incurs a substantial inference overhead, consuming 113.4k
tokens—over 62% of its total budget—on unstructured, iterative error correction. Conversely, the Full PhyNiKCE
agent benefits from both a robust initial setup and the use of symbolic retrievers for guided reflection, reducing token
consumption in this stage by over 50% to just 56.5k tokens. This demonstrates a fundamental shift in resource
management: PhyNiKCE reallocates inference resources from reactive debugging to proactive, knowledge-driven setup.
5.3 Robustness Across Diverse CFD Regimes
We further analyzed performance across different flow regimes and turbulence models to evaluate the Agent’s robustness.
The accuracy metrics, stratified by physical complexity, are summarized in Table 2.
The data indicates that incompressible flows present a lower barrier to entry, with the Full PhyNiKCE configuration
achieving 62% accuracy, compared to 40% for the more complex compressible cases. This performance gap is consistent
with domain expectations, as compressible flows demand coupled boundary conditions, strictly bounded numerical
schemes, and additional thermodynamic equations of state [ 9]. However, the relative contribution of the PhyNiKCE
system is most pronounced in these high-complexity regimes. The system provides a 122% relative improvement for
compressible flows (increasing accuracy from 18% to 40%) versus an 88% improvement for incompressible flows (34%
to 62%). This suggests that the utility of physically consistent context retrieval scales positively with the complexity of
the underlying physics.
19

APREPRINT- FEBRUARY13, 2026
This differential performance validates the core neurosymbolic hypothesis, as the Baseline agent struggles significantly
with the physical diversity of the test matrix. The k−ϵ turbulence model serves as a prime example of this limitation. Due
to its multi-equation complexity—requiring synchronized definitions for kinetic energy ( k), dissipation ( ϵ), and specific
wall functions—it proved the most challenging case, with the Baseline achieving only 20% accuracy. This failure stems
from context poisoning: a standard RAG approach may retrieve file snippets from a simpler Spalart-Allmaras case due
to a shared solver, ignoring the critical physical dependencies unique to the k−ϵ model. PhyNiKCE resolves this by
deterministically filtering the Knowledge Base using retrieval keys (Solver, Turbulence Model, Compressibility). This
process, executed by the Multi-Source and Cascading Fallback Retrievers, ensures the retrieved context is structurally
isomorphic to the target physics, handling the parameter polymorphism inherent in complex turbulence closures. As a
result, the Full PhyNiKCE agent more than doubled performance on k−ϵ cases to 44% accuracy, demonstrating its
ability to prevent the heuristic failures common in naive retrieval baselines.
5.4 Validation of Physical Fidelity
Beyond statistical accuracy metrics, it is imperative to verify that the agent-generated setups produce valid engineering
data. We compared the simulation results from the Accurate cases against established experimental benchmarks to
confirm physical fidelity. Only representative cases are illustrated for brevity.
Figure 8: Velocity contours and comparison of pressure coefficients for NACA 0012 airfoil at different angles of attack
using three turbulence models (Spalart-Allmaras, k−ϵ , and k−ω SST). (a) Velocity contour at 10° angle of attack
using k−ω SST model. (b) Velocity contour at 15° angle of attack using k−ω SST model. (c) Pressure coefficient
comparison at 10°. (d) Pressure coefficient comparison at 15°. Symbols in (c) and (d) represent experimental results
reported by [38].
Figure 8 illustrates the results for the incompressible NACA 0012 airfoil. The velocity contours (Figures 8a–b) correctly
capture the flow acceleration and wake development associated with increased incidence. Quantitative validation is
provided in Figures 8(c) and 8(d), which compare the calculated pressure coefficients ( Cp) against experimental data
from [ 38]. The close agreement confirms that the Multi-Source Retriever successfully resolved the specific boundary
condition parameters required for the Spalart-Allmaras and k−ω SST models. Notably, while the k−ϵ model deviates
at15◦due to its known limitations in predicting separation under adverse pressure gradients [ 39], the simulation itself
remained stable and converged, indicating that the agent correctly implemented the model’s setup despite the model’s
inherent physical limitations.
Furthermore, Figure 9 validates the agent’s robustness in the rigorous compressible regime by comparing numerical
schlieren images (magnitude of the density gradient) with experimental schlieren images from [ 34]. At NPR = 3.0
(Figures 9a–b), the agent-configured rhoCentralFoam solver correctly predicts the formation of the Mach stem and the
characteristic λ-shock configuration associated with ramp-pattern separation. Similarly, at NPR = 2.3 (Figures 9c–d),
thesonicFoam setup accurately captures the transition to flap-pattern separation. The high-fidelity reproduction of
shock train positions and expansion fan topologies confirms that the Cascading Fallback Retriever effectively enforced
20

APREPRINT- FEBRUARY13, 2026
Figure 9: Comparison of schlieren visualization between experimental data [ 34] and numerical simulations. (a)
Experimental schlieren at NPR = 3.0. (b) Numerical schlieren using rhoCentralFoam withk−ω SST turbulence
model at NPR = 3.0. (c) Experimental schlieren at NPR = 2.3. (d) Numerical schlieren using sonicFoam with
Spalart-Allmaras turbulence model at NPR = 2.3. NPR: nozzle pressure ratio.
the thermodynamic constraints—specifically the compressible state equations and energy conservation terms—that are
frequently omitted in standard naive retrieval methods.
The ability to reproduce these complex physical phenomena is not merely a result of correct initialization, but is
rigorously enforced by the system’s ability to audit and correct its own configuration files during runtime, as detailed in
the following section.
5.5 The Mechanism of Efficient Error Correction
Figure 10: Distribution of successfully solved errors within the accurate cases only.
21

APREPRINT- FEBRUARY13, 2026
While the physical validation in Section 5.4 confirms the final output quality, the agent’s reliability stems from its ability
to autonomously recover from failure. To provide mechanistic insight into this capability, this section analyzes the
distribution of errors successfully resolved by the agent, contrasting the Baseline and Full PhyNiKCE configurations to
illustrate how the neurosymbolic architecture overcomes the systemic limitations of less constrained agents.
The Error Reflection module (Stage 3) classifies simulation failures into four primary categories: File-missing Errors,
Dimensional Errors, Persistent Errors, and Complex Configuration errors. Our analysis revealed that approximately
90% of all identified failures were Complex Configuration Errors—knowledge-intensive mistakes in the case setup. We
therefore focus our analysis on these, which are further sub-classified into four types as illustrated in Figure 10:
•Setup Formats:Syntactic irregularities, such as missing delimiters in dictionaries or invalid file headers.
•IC/BCs:Physical inconsistencies in IC/BCs, such as mismatched field types or inconsistent thermodynamic
coefficients.
•Discretization Schemes:Numerical instabilities in system/fvSchemes , such as applying unbounded schemes to
convection terms in high-speed flows.
•Linear Solvers:Inefficient or incompatible solver settings in system/fvSolution , including poor preconditioner
selection or missing relaxation factors.
The Baseline agent, which achieved only 26% accuracy, exhibits a success profile dominated byrudimentaryerrors. As
shown in Figure 10(a), 49.2% of its resolved issues were in IC/BCs and 11.1% were simple Setup Formats. Its limited
14.3% share of resolved Linear Solver errors reflects a “survivorship bias”; the agent rarely progressed past initial setup
flaws to address these complex stability issues, causing the case to fail and be excluded from this dataset of successful
resolutions.
In contrast, the Full PhyNiKCE agent (Figure 10(b)) demonstrates a far more comprehensive correction capability. It
systematically minimizes rudimentary errors at the outset: the reduction in Setup Format errors (from 11.1% to 5.0%)
validates the Data-driven Template Retriever, while the marked decrease in IC/BCs errors (from 49.2% to 30.3%) is
directly attributable to the Multi-Source Retriever.
By reliably resolving these initial barriers, the agent is able to progress to and address more sophisticated numerical
challenges. This is evidenced by the counter-intuitive increase in the share of resolved Linear Solver errors to 45.5%.
This high proportion does not indicate more errors, but rather that the Full PhyNiKCE agent—leveraging the Heuristic
Keyword Retriever for targeted corrections—is capable of advancing to the numerical solution phase and successfully
stabilizing the simulation.
Critically, this error resolution process provides Auditability. Unlike standard LLM self-correction, which often involves
opaque regeneration, the Deterministic RAG Engine logs the specific retrieval rule used to fix an error (e.g., “Corrected
div(phi,U) using bounded Gauss upwind from Incompressible Flow Template”). This traceable decision path is
essential for reliable industrial autonomous agents, ensuring that autonomous corrections adhere to verified engineering
standards.
5.6 Analysis of Failed Cases and Future Directions
While the Full PhyNiKCE agent’s 51% accuracy represents a substantial advancement over the baseline, the analysis of
the 49% of cases that failed is highly instructive for defining future research directions. These failures fall into three
primary categories:
•10%: Semantically Misaligned (Runnable) Cases.These simulations were physically valid and ran to completion
but were ultimately incorrect due tosemantic misalignment—for example, setting the Angle of Attack to 0◦instead
of the requested 10◦. This highlights a key limitation: the Symbolic Knowledge Engine ensures the simulation is
internally consistent (Numerical Validity) but cannot verify its alignment with external requirements (Ground Truth)
without a separate validation module.
•32%: Simulation-Halting Errors.These cases terminated due to divergence or unphysical results after executing
beyond the agent’s immediate reflection window. These represent the most challenging Persistent Errors. The core
difficulty here is root cause ambiguity; late-stage divergence can stem from subtle inconsistencies in numerical
schemes or linear solvers making deterministic diagnosis computationally difficult without deeper causal reasoning.
•7%: Reflection Threshold Exceeded.These cases were terminated after failing to resolve errors within the 30-round
reflection limit, indicating scenarios where the current retrieval strategies could not locate a viable solution within the
search space.
22

APREPRINT- FEBRUARY13, 2026
This failure analysis is further illuminated by the performance disparity between flow regimes. Compressible cases
exhibited a failure rate of 60%, compared to 38% for incompressible cases. This discrepancy suggests a structural
imbalance in the Symbolic Knowledge Base. Since the knowledge extraction process (Stage 0) currently relies on
OpenFOAM tutorials, which contain a preponderance of incompressible examples, the agent possesses a sparser context
for high-speed compressible flows.
These findings clearly delineate the path for future development. First, a dedicated post-simulation validation module is
required to detect runnable but physically misaligned cases by comparing output fields against theoretical expectations
or ground truth. Second, the Symbolic Knowledge Base must be expanded by parsing a broader corpus of documentation
and source code to mitigate the data imbalance in complex physical regimes. Ultimately, these findings confirm that
the bottleneck for autonomous engineering agents is no longer model capability, but thedensity and structureof the
domain-specific knowledge base.
5.7 Generalizability to Broader Engineering Domains
While this study focuses on autonomous CFD using OpenFOAM, the PhyNiKCE architecture has potential for broad
applicability across Computer-Aided Engineering (CAE) domains. The core workflow of defining boundary conditions,
material properties, and solver controls is structurally similar in fields like solid mechanics (e.g., Abaqus [ 40]),
atmospheric science (e.g., WRF [ 41]), and electromagnetics. PhyNiKCE’s generalizability stems from two key
abstractions: a universal knowledge representation and domain-agnostic retrieval strategies.
First, the Symbolic Knowledge Base acts as a universal intermediate representation. A modular parser, the Knowledge
Base Builder, translates domain-specific file formats (e.g., OpenFOAM dictionaries, Abaqus .inp files) into a stan-
dardized JSON schema. This abstracts software-specific syntax into a consistent, LLM-friendly structure. Adapting
PhyNiKCE to a new domain only requires implementing a new parser for that domain’s file format.
Second, the symbolic retrieval strategies are not software-specific but address universal engineering challenges. For
example, theCascading Fallbackretriever resolves sparse model combinations by relaxing constraints in a physically
logical order (e.g., finding a generic “hyperelastic” material in FEM when a specific “hyperelastic-viscoplastic” model
is unavailable). Similarly, theMulti-Sourceretriever defines complex boundary conditions by augmenting tutorial
examples with authoritative documentation (e.g., configuring a Perfectly Matched Layer [ 42] in electromagnetics).
Because these strategies address the fundamental logic of configuring physics-based simulations, the PhyNiKCE
framework provides a robust and generalizable foundation for building autonomous agents across a wide range of
engineering disciplines.
6 Conclusion
This study addresses the critical Semantic-Physical Disconnect hindering the deployment of LLMs in autonomous
physics-based simulations such as CFD. We demonstrated that while probabilistic generative models excel at syntactic
coherence, they fundamentally fail to adhere to the rigid Boolean validity functions required by deterministic solvers.
To bridge this gap, we introduced PhyNiKCE, a neurosymbolic agentic framework that decouples generation from
validation, replacing vector-based semantic similarity with deterministic, rule-based retrievers.
We validated PhyNiKCE through a comprehensive ablation study on autonomous OpenFOAM simulations spanning
incompressible and compressible flow regimes and covering three widely used turbulence models. The proposed
methodology demonstrated superior performance across three key dimensions critical for LLM Agents:
1.Accuracy and Robustness:The PhyNiKCE-enabled agent achieved a 96% relative improvement over SOTA
baselines, increasing the accuracy from 51% to 26%. This significant improvement surpasses the performance
ceiling of prior CFD agents (around 25% to 30%), confirming the Deterministic RAG Engine’s capabilities to enforce
physical and numerical consistency across complex setting interdependencies—such as discretization schemes and
boundary condition compatibility—that confound purely neural approaches in real-world settings.
2.Inference Efficiency:Our findings challenge the assumption that neurosymbolic architectures impose prohibitive
computational overhead. By strategically front-loading computational effort into knowledge-driven initialization,
PhyNiKCE reduced per-case error reflection iterations by 59% and lowered LLMs’ inference cost by approximately
17%. This establishes that in autonomous engineering simulations, adhering to domain knowledge is a prerequisite
for computational economy.
3.Auditability and Trust:Unlike black-box vector-based retrieval, PhyNiKCE provides a transparent audit trail. By
tracing every configuration parameter to a specific rule in the Knowledge Base, the framework offers the white-box
interpretability required for safety-critical engineering certification, aligning with the principles of Trustworthy AI.
23

APREPRINT- FEBRUARY13, 2026
Our findings challenge the paradigm that generalization requires massive datasets. PhyNiKCE nearly doubles accuracy
using a compact, structured knowledge base, demonstrating that for logic-driven domains, Information Density is more
effective than Data V olume.
Future work will address the performance gap between autonomous incompressible (62% accuracy) and compressible
(40%) flows by expanding the knowledge base beyond tutorials to include source code and literature, mitigating the
current data bias. We will also develop a post-simulation validation module to detect cases that are runnable but
physically incorrect, closing the loop on autonomous reliability.
The PhyNiKCE architecture is highly generalizable. The core autonomous CFD workflow of defining boundary
conditions, material properties, and solver controls is common across CAE disciplines like solid mechanics and heat
transfer. By decoupling neural generation from symbolic validation, this framework provides a robust paradigm for
building AI and data-driven autonomous agents for next-generation industrial applications, including Digital Twins and
Industry 4.0.
Acknowledgments
This work is supported by the Innovation and Technology Fund – Innovation and Technology Support Programme
(ITF-ITSP) (Grant No. ITS/062/23FP).
Conflict of Interest
The authors have no conflicts to disclose.
Author contributions
E Fan: Conceptualization; Software; Visualization; Methodology (equal); Data Curation (equal); Formal Analysis
(equal); Writing - original draft.Lisong Shi: Methodology (equal); Data Curation (equal); Formal Analysis (equal);
Writing - review & editing (equal).Zhengtong Li: Formal Analysis (equal);Writing - review & editing (equal).
Chih-Yung Wen: Project administration; Supervision; Resources; Funding acquisition; Writing - review & editing
(equal).
Data Availability Statement
The data that support the findings of this study are available from the corresponding authors upon reasonable request.
CODE A V AILABILITY STATEMENT
The numerical implementation of the PhyNiKCE system is built upon the open-source ChatCFD agent ( https:
//github.com/EarlFan/ChatCFD ). The source code for PhyNiKCE will be made publicly available at https:
//github.com/EarlFan/PhyNiKCEupon the acceptance and publication of this manuscript.
A Example of an OpenFOAM tutorial case in the knowledge base
Listing 3 shows the ‘periodicHill’ OpenFOAM tutorial case, stored in the Symbolic Knowledge Base as a JSON object.
Case setup files, such as ‘system/fvSolution’ and ‘constant/transportProperties’, are nested under the ‘configuration_files’
key. For brevity, only a portion of the ‘system/fvSolution’ content is displayed, detailing the linear solver settings;
for instance, the pressure equation uses the ‘GAMG’ solver with a ‘DICGaussSeidel’ smoother. The entry is also
tagged with physical features (e.g., ‘solver’, ‘turbulence_model’, ‘compressibility’) that serve as retrieval keys for
the Deterministic RAG Engine. Additional labels like ‘thermophysicalModel’ support future rule expansion for
multi-physics scenarios.
Listing 3: The ‘periodicHill’ case in the Symbolic Knowledge Base
1{
2" periodicHill ": {
3" case_path ": " incompressible / pimpleFoam /LES / periodicHill / steadyState ",
24

APREPRINT- FEBRUARY13, 2026
4" configuration_files ": {
5" system / fvSolution ": {
6" FoamFile ": {
7" version ": 2.0,
8" format ": " ascii ",
9" class ": " dictionary ",
10" object ": " fvSolution "},
11" solvers ": {
12"p": {
13" solver ": " GAMG ",
14" smoother ": " DICGaussSeidel ",
15" tolerance ": 1e-06,
16" relTol ": 0.05},
17...
18},
19" system / fvSchemes ": {" FoamFile ": { ...}},
20" system / controlDict ": {" FoamFile ": { ...}},
21" system / fvOptions ": {" FoamFile ": { ...}},
22" constant / transportProperties ": {" FoamFile ": {... }},
23" constant / turbulenceProperties ": {" FoamFile ": {... }},
24"0/U": {" FoamFile ": {...}},
25"0/k": {" FoamFile ": {...}},
26"0/nut": {" FoamFile ": {...}},
27"0/p": {" FoamFile ": {...}},
28"0/ nuTilda ": {" FoamFile ": {... }},
29...
30},
31" solver ": " simpleFoam ",
32" turbulence_model ": " SpalartAllmaras ",
33" compressible ": false ,
34" turbulence_type ": "RAS ",
35" thermophysicalModel ": null ,
36" singlePhase ": true ,
37" particle_flow ": false ,
38" reacting_flow ": false ,
39" ddtScheme ": " steadyState ",
40" boundary_type ": [" fixedValue "," noSlip "," zeroGradient "," cyclic ","
nutUSpaldingWallFunction "]
41}
42}
B Examples of the Data-driven template Retriever
To facilitate understanding, we provide a simplified JSON representation of the merging process described in Algo-
rithm 2.
B.1 Input: Feature-Specific Profiles
The algorithm first analyzes the Knowledge Base for two physical models: msol=sonicFoam andmturb=kEpsilon
(k−ϵmodel).
Profile 1: Turbulence Model (kEpsilon)
The query for the kEpsilon model (Listing 4) retrieves 83 tutorial cases from the Knowledge Base. The resulting
statistical profile strongly emphasizes the required turbulence fields (e.g., epsilon turbulent dissipation rate) but, as
nearly all of these tutorials are incompressible, it lacks the necessary fields for compressible flow (e.g.,rhodensity).
Listing 4: Query for thekEpsilonmodel
1{
2" feature ": " turbulence model = kEpsilon ",
3" case_count ": 83,
4" rates ": {
5" solvers ": {
25

APREPRINT- FEBRUARY13, 2026
6"p": 0.53,
7"U": 0.78, // High probability
8"k": 0.78, // High probability
9" epsilon ": 0.78, // High probability
10"rho": 0.0 // Absent in incompressible cases
11}
12}
13}
Profile 2: Solver (sonicFoam)
The query for the sonicFoam solver (Listing 5) retrieves only 4 tutorial cases from the Knowledge Base. This profile
strongly emphasizes the required compressible fields (e.g., rho density, einternal energy) but lacks the specific
turbulence fields from thek-epsilonmodel, as they are not present in this subset of cases.
Listing 5: Query for thesonicFoamsolver
1{
2" feature ": " solver = sonicFoam ",
3" case_count ": 4,
4" rates ": {
5" solvers ": {
6"p": 0.25, // Low
7"U": 1.0, // Critical for this solver
8" rho": 1.0, // Critical for this solver
9"e": 1.0 // Critical for this solver
10}
11}
12}
B.2 Output: The Merged Template (Union-Max)
The algorithm merges the two profiles (Listing 6). For each key, it takes the maximum rate. As kis not included in
Profile 2, its value is set to 0.
•p:max(0.53,0.25)→0.53(Kept)
•rho:max(0.0,1.0)→1.0(Kept)
•k:max(0.78,0.0)→0.78(Kept)
• ...
Final Merged Probability Map:
Listing 6: A merged template
1{
2" solvers ": {
3"p": 0.53, // Inherited from kEpsilon
4"U": 1.0, // Inherited from sonicFoam
5"k": 0.78, // Inherited from kEpsilon
6" epsilon ": 0.78, // Inherited from kEpsilon
7"rho": 1.0, // Inherited from sonicFoam
8"e": 1.0 // Inherited from sonicFoam
9}
10}
This resulting map is then filtered by the threshold (e.g., τ= 0.3 ) to produce the final system/fvSolution setup,
ensuring it containsboththe turbulence settings required by kEpsilon and the compressible flow settings required by
sonicFoam.
B.3 Atomic Unit Example: Linear Solver Configuration
Listing 7 exemplifies an ‘Atomic Unit’ for a linear solver configuration. The parameters within this block—such as
‘solver’, ‘tolerance’, ‘relTol’, and ‘smoother’—are highly interdependent and form a self-consistent numerical strategy.
26

APREPRINT- FEBRUARY13, 2026
To avoid creating unstable or inefficient configurations by mixing and matching individual parameters from different
sources, the entire block is treated as a single, indivisible unit by the Data-driven Template Retriever.
Listing 7: A Linear solver setup
1p
2{
3" solver ": " GAMG ",
4" tolerance ": 1e-06,
5" relTol ": 0.01,
6" smoother ": " GaussSeidel "
7}
C Heuristic Keyword Retriever: Targeted Context Extraction
To illustrate the efficacy of the Heuristic Keyword Retriever’s targeted context extraction, consider a scenario where
the agent encounters a runtime error related to the divergence scheme for the turbulent dissipation rate ( ϵ). If the
retrieval logic simply targeted the file section “ divSchemes ” (as seen in standard initialization), the output would be a
voluminous block containing irrelevant schemes for velocity, enthalpy, and reaction rates shows as Listing 8:
Listing 8: Broad Retrieval: Keyword “divSchemes” (Low Information Density)
1{
2" sample_setup_ 0": {
3" divSchemes ": {
4" default ": " none ",
5"div( phi,U)": " Gauss limitedLinearV 1",
6"div( phi,h)": " Gauss limitedLinear 1",
7"div( phi,K)": " Gauss linear ",
8" turbulence ": " Gauss limitedLinear 1",
9"div( phi,k)": " Gauss limitedLinear 1",
10"div( phi, epsilon )": " Gauss limitedLinear 1",
11"div( phi,R)": " Gauss limitedLinear 1",
12"div(R)": " Gauss linear ",
13"div ((( rho* nuEff )*dev 2(T( grad (U)))))": " Gauss linear "
14}
15},
16... // (4 other blocks with " sample_setup_ 0" omitted )
17}
In contrast, the Heuristic Keyword Retriever identifies the specific erroneous keyword “ div(phi,epsilon) ” from the
error log. The multi-stage search isolates exactly this parameter, filtering out all unrelated noise. The result (Listing 9)
is a highly dense, targeted context that provides the LLM with five proven variations for the specific failing term:
Listing 9: Targeted Retrieval: Keyword ‘div(phi,epsilon)’ (High Information Density)
1{
2" sample_setup_ 0":{"div(phi , epsilon )":" Gauss upwind "},
3" sample_setup_ 1":{"div(phi , epsilon )":" Gauss upwind "},
4" sample_setup_ 2":{"div(phi , epsilon )":" bounded Gauss upwind "},
5" sample_setup_ 3":{"div(phi , epsilon )":" Gauss upwind "},
6" sample_setup_ 4":{"div(phi , epsilon )":" bounded Gauss linearUpwind limited "}
7}
This targeted output drastically reduces token consumption and eliminates the possibility of the LLMs hallucinating
incorrect schemes for other variables (e.g.,Uork) that were not the source of the error.
References
[1]Jingyu Cui, Yansong Li, Yuzhen Jin, and Zuchao Zhu. Cryogenic cavitation: Advances in experimental character-
ization and numerical modeling.Physics of Fluids, 37(11), 2025.
27

APREPRINT- FEBRUARY13, 2026
[2]E Fan, Tianhan Zhang, Jiaao Hao, Chih-Yung Wen, and Lisong Shi. Fire: an open-source adaptive mesh refinement
solver for supersonic reacting flows.Computer Physics Communications, page 109881, 2025.
[3]Zhaokun Wang, Chenglei Wang, Fuwang Zhao, Feng Ren, Xiaoyu Luo, and Hui Tang. Fluid-structure interaction
in phaco-emulsification based cataract surgery.International Journal of Mechanical Sciences, 267:109022, 2024.
[4]Haixin Wang, Yadi Cao, Zijie Huang, Yuxuan Liu, Peiyan Hu, Xiao Luo, Zezheng Song, Wanjia Zhao, Jilin Liu,
Jinan Sun, et al. Recent advances on machine learning for computational fluid dynamics: A survey.arXiv preprint
arXiv:2408.12171, 2024.
[5]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948, 2025.
[6]Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel
Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning,
multimodality, long context, and next generation agentic capabilities.arXiv preprint arXiv:2507.06261, 2025.
[7]Nithin Somasekharan, Ling Yue, Yadi Cao, Weichao Li, Patrick Emami, Pochinapeddi Sai Bhargav, Anurag
Acharya, Xingyu Xie, and Shaowu Pan. Cfd-llmbench: A benchmark suite for evaluating large language models
in computational fluid dynamics.arXiv preprint arXiv:2509.20374, 2025.
[8]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[9]E Fan, Kang Hu, Zhuowen Wu, Jiangyang Ge, Jiawei Miao, Yuzhi Zhang, He Sun, Weizong Wang, and Tianhan
Zhang. Chatcfd: A large language model-driven agent for end-to-end computational fluid dynamics automation
with structured knowledge and reasoning.Advanced Intelligent Discovery, page e202500174, 2026.
[10] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie
Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey.Science China
Information Sciences, 68(2):121101, 2025.
[11] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen,
Yankai Lin, et al. A survey on large language model based autonomous agents.Frontiers of Computer Science,
18(6):186345, 2024.
[12] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.
Chain-of-thought prompting elicits reasoning in large language models.Advances in neural information processing
systems, 35:24824–24837, 2022.
[13] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language
agents with verbal reinforcement learning.Advances in Neural Information Processing Systems, 36:8634–8652,
2023.
[14] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React:
Synergizing reasoning and acting in language models. InThe eleventh international conference on learning
representations, 2022.
[15] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer,
Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools.
Advances in Neural Information Processing Systems, 36:68539–68551, 2023.
[16] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving
ai tasks with chatgpt and its friends in hugging face.Advances in Neural Information Processing Systems,
36:38154–38180, 2023.
[17] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili
Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative
framework. InThe Twelfth International Conference on Learning Representations, 2023.
[18] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. Augment-
ing large language models with chemistry tools.Nature Machine Intelligence, 6(5):525–535, 2024.
[19] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press.
Swe-agent: Agent-computer interfaces enable automated software engineering.Advances in Neural Information
Processing Systems, 37:50528–50652, 2024.
28

APREPRINT- FEBRUARY13, 2026
[20] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anand-
kumar. V oyager: An open-ended embodied agent with large language models.arXiv preprint arXiv:2305.16291,
2023.
[21] Thomas D Economon, Francisco Palacios, Sean R Copeland, Trent W Lukaczyk, and Juan J Alonso. Su2: An
open-source suite for multiphysics simulation and design.Aiaa Journal, 54(3):828–846, 2016.
[22] Hrvoje Jasak, Aleksandar Jemcov, Zeljko Tukovic, et al. Openfoam: A c++ library for complex physics
simulations. InInternational workshop on coupled methods in numerical dynamics, volume 1000, pages 1–20.
Dubrovnik, Croatia), 2007.
[23] Florianr Menter. Zonal two equation kw turbulence models for aerodynamic flows. In23rd fluid dynamics,
plasmadynamics, and lasers conference, page 2906, 1993.
[24] Yuxuan Chen, Xu Zhu, Hua Zhou, and Zhuyin Ren. Metaopenfoam: an llm-based multi-agent framework for cfd.
arXiv preprint arXiv:2407.21320, 2024.
[25] Sandeep Pandey, Ran Xu, Wenkang Wang, and Xu Chu. Openfoamgpt: A retrieval-augmented large language
model (llm) agent for openfoam-based computational fluid dynamics.Physics of Fluids, 37(3), 2025.
[26] Ling Yue, Nithin Somasekharan, Tingwen Zhang, Yadi Cao, and Shaowu Pan. Foam-agent 2.0: An end-to-end
composable multi-agent framework for automating cfd simulation in openfoam.arXiv preprint arXiv:2509.18178,
2025.
[27] Zhaoyue Xu, Long Wang, Chunyu Wang, Yixin Chen, Qingyong Luo, Hua-Dong Yao, Shizhao Wang, and Guowei
He. Cfdagent: A language-guided, zero-shot multi-agent system for complex flow simulation.Physics of Fluids,
37(11), 2025.
[28] Yuxuan Chen, Xu Zhu, Hua Zhou, and Zhuyin Ren. Metaopenfoam 2.0: Large language model driven chain of
thought for automating cfd simulation and post-processing.arXiv preprint arXiv:2502.00498, 2025.
[29] Artur d’Avila Garcez and Luis C Lamb. Neurosymbolic ai: The 3rd wave.Artificial Intelligence Review,
56(11):12387–12406, 2023.
[30] Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, and Houbing Herbert Song. Symrag: Efficient neuro-
symbolic retrieval through adaptive query routing. In19th International Conference on Neurosymbolic Learning
and Reasoning, 2025.
[31] Zhongwu Chen, Chengjin Xu, Dingmin Wang, Zhen Huang, Yong Dou, Xuhui Jiang, and Jian Guo. Rulerag:
Rule-guided retrieval-augmented generation with language models for question answering.arXiv preprint
arXiv:2410.22353, 2024.
[32] Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei Han, Le Sun,
and Yongbin Li. Structrag: Boosting knowledge intensive reasoning of llms via inference-time hybrid information
structurization.arXiv preprint arXiv:2410.08815, 2024.
[33] Zhaozheng Sun and Wenhui Yan. Comparison of different turbulence models in numerical calculation of low-
speed flow around naca0012 airfoil. InJournal of Physics: Conference Series, volume 2569, page 012075. IOP
Publishing, 2023.
[34] T Yu, Y Yu, YP Mao, YL Yang, and SL Xu. Comparative study of openfoam solvers on separation pattern and
separation pattern transition in overexpanded single expansion ramp nozzle.Journal of Applied Fluid Mechanics,
16(11):2249–2262, 2023.
[35] Philippe Spalart and Steven Allmaras. A one-equation turbulence model for aerodynamic flows. In30th aerospace
sciences meeting and exhibit, page 439, 1992.
[36] Brian Edward Launder and Dudley Brian Spalding. The numerical computation of turbulent flows. InNumerical
prediction of flow, heat transfer, turbulence and combustion, pages 96–116. Elsevier, 1983.
[37] Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and
selective augmentation.arXiv preprint arXiv:2310.04408, 2023.
[38] Nigel Gregory and CL O’reilly. Low-speed aerodynamic characteristics of naca 0012 aerofoil section, including
the effects of upper-surface roughness simulating hoar frost. 1970.
[39] Douvi C Eleni, Tsavalos I Athanasios, and Margaris P Dionissios. Evaluation of the turbulence models for
the simulation of the flow over a national advisory committee for aeronautics (naca) 0012 airfoil.Journal of
Mechanical Engineering Research, 4(3):100–111, 2012.
[40] Ever J Barbero.Finite element analysis of composite materials using Abaqus®. CRC press, 2023.
29

APREPRINT- FEBRUARY13, 2026
[41] William C Skamarock, Joseph B Klemp, Jimy Dudhia, David O Gill, Zhiquan Liu, Judith Berner, Wei Wang,
Jordan G Powers, Michael G Duda, Dale M Barker, et al. A description of the advanced research wrf version 4.
NCAR tech. note ncar/tn-556+ str, 145(10.5065), 2019.
[42] Jean-Pierre Berenger. A perfectly matched layer for the absorption of electromagnetic waves.Journal of
computational physics, 114(2):185–200, 1994.
30
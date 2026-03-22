# TopoChunker: Topology-Aware Agentic Document Chunking Framework

**Authors**: Xiaoyu Liu

**Published**: 2026-03-19 02:15:10

**PDF URL**: [https://arxiv.org/pdf/2603.18409v1](https://arxiv.org/pdf/2603.18409v1)

## Abstract
Current document chunking methods for Retrieval-Augmented Generation (RAG) typically linearize text. This forced linearization strips away intrinsic topological hierarchies, creating ``semantic fragmentation'' that degrades downstream retrieval quality. In this paper, we propose TopoChunker, an agentic framework that maps heterogeneous documents onto a Structured Intermediate Representation (SIR) to explicitly preserve cross-segment dependencies. To balance structural fidelity with computational cost, TopoChunker employs a dual-agent architecture. An Inspector Agent dynamically routes documents through cost-optimized extraction paths, while a Refiner Agent performs capacity auditing and topological context disambiguation to reconstruct hierarchical lineage. Evaluated on unstructured narratives (GutenQA) and complex reports (GovReport), TopoChunker demonstrates state-of-the-art performance. It outperforms the strongest LLM-based baseline by 8.0% in absolute generation accuracy and achieves an 83.26% Recall@3, while simultaneously reducing token overhead by 23.5%, offering a scalable approach for structure-aware RAG.

## Full Text


<!-- PDF content starts -->

TopoChunker: Topology-Aware Agentic Document Chunking Framework
Xiaoyu Liu
Independent Researcher
liushifu986938279@gmail.com
Abstract
Current document chunking methods for
Retrieval-Augmented Generation (RAG) typ-
ically linearize text. This forced lineariza-
tion strips away intrinsic topological hierar-
chies, creating “semantic fragmentation” that
degrades downstream retrieval quality. In this
paper, we propose TopoChunker, an agentic
framework that maps heterogeneous documents
onto a Structured Intermediate Representation
(SIR) to explicitly preserve cross-segment de-
pendencies. To balance structural fidelity with
computational cost, TopoChunker employs a
dual-agent architecture. An Inspector Agent
dynamically routes documents through cost-
optimized extraction paths, while a Refiner
Agent performs capacity auditing and topo-
logical context disambiguation to reconstruct
hierarchical lineage. Evaluated on unstruc-
tured narratives (GutenQA) and complex re-
ports (GovReport), TopoChunker demonstrates
state-of-the-art performance. It outperforms
the strongest LLM-based baseline by 8.0% in
absolute generation accuracy and achieves an
83.26% Recall@3, while simultaneously re-
ducing token overhead by 23.5%, offering a
scalable approach for structure-aware RAG.
1 Introduction
Retrieval-Augmented Generation (RAG) mitigates
Large Language Model (LLM) hallucinations
by grounding generation in external knowledge.
While long-context LLMs have continuously im-
proved, fine-grained retrieval remains essential due
to lower inference costs and the necessity to avoid
the “Lost in the Middle” phenomenon. Conse-
quently, document chunking acts as the founda-
tional semantic filter in the RAG pipeline, directly
determining the upper bound of downstream re-
trieval quality.
However, conventional chunking tools rely on
static heuristics that are agnostic to inherent docu-
ment topology. Even emerging LLM-based chun-kers (e.g., LumberChunker) predominantly operate
under a rigid “text-to-text” paradigm, flattening
complex, multi-modal documents (such as enter-
prise PDFs) into linear text streams. This forced
linearization introduces three main issues: visual-
semantic disconnect, contextual fragmentation, and
feature dilution. Such mechanical splitting strips
chunks of their hierarchical topology, creating “se-
mantic islands” that isolate context-dependent ter-
minology and cause anaphoric ambiguity. Further-
more, fully generative chunkers apply computation-
ally expensive LLM reasoning uniformly across all
text, resulting in redundant token expenditures on
structurally simple content.
To address these structural and computational
constraints, we propose TopoChunker, a topology-
aware framework for structural-fidelity document
chunking. Unlike linear pipelines, TopoChun-
ker treats documents as heterogeneous entities
and maps them onto a shared Structured Inter-
mediate Representation (SIR). Driven by a col-
laborative dual-agent architecture—an Inspector
Agent for routing and a Refiner Agent for seman-
tic auditing—the system systematically bridges the
gap between fragmented raw text and contextually
self-contained RAG inputs.
Our primary contributions are summarized as
follows:
•Adaptive Routing via Inspector Agent:We
introduce a dynamic routing mechanism that
actively probes document complexity. It opti-
mizes the cost-fidelity trade-off by selectively
applying rule-based slicing for standard lay-
outs and Vision-Language Models (VLMs)
for complex visual structures, thereby avoid-
ing uniform LLM overhead.
•Topological Modeling & Atomicity-Aware
Execution:We propose the Structured Inter-
mediate Representation (SIR) protocol to de-
couple physical parsing from semantic rea-arXiv:2603.18409v1  [cs.CL]  19 Mar 2026

soning. Combined with a stack-based traver-
sal algorithm and Atomicity Locking, this in-
memory graph preserves logical hierarchies
and prevents the physical fragmentation of
indivisible units (e.g., cross-page tables).
•Context Disambiguation via Refiner Agent:
We design a diagnostic Refiner Agent that au-
tonomously resolves contextual voids, such
as grounding dangling pronouns and deci-
phering isolated terminology, to generate self-
contained chunks. Extensive evaluations show
that TopoChunker yields state-of-the-art per-
formance, outperforming the strongest base-
line by an absolute margin of 8.0% in end-
to-end generation accuracy and achieving
83.26% Recall@3, alongside a 23.5% reduc-
tion in token consumption.
2 Related Work
2.1 Granularity Evolution: From Heuristic
Rules to Semantic Sensitivity
Document chunking strategies have advanced from
rigid, fixed-length heuristics to flexible, semantic-
driven approaches. Traditional fixed-size chunk-
ing serves as the baseline, partitioning text into
segments of predefined character or token lengths,
often employing an overlap window to mitigate
boundary fragmentation. While simple to imple-
ment, this approach frequently leads to semantic
misalignment in structured documents. Recursive
character chunking improves upon this by using
prioritized delimiters to split text until segments
fall below a threshold, making it better suited for
hierarchical content.
To further align with semantic boundaries,
semantic-aware techniques have emerged. Seman-
tic Chunker (Xiao et al., 2024) utilizes embedding
similarity thresholds to merge sentence-level units,
enhancing intra-chunk coherence. However, its
performance remains contingent on embedding
quality, and it often struggles with long-tail dis-
tributions in complex documents. Meta-Chunking
(Zhao et al., 2024) introduces model-derived sig-
nals like perplexity fluctuations, while Lumber-
Chunker (Duarte et al., 2024) explicitly leverages
LLM reasoning to define boundaries. Despite these
advances, the “feature dilution” problem persists:
oversized chunks cause embedding vectors to shift
toward noisy centroids, reducing retrieval discrim-
inability. Furthermore, these methods are largely
confined to single-granularity paradigms, failingto capture multi-dimensional topological features.
TopoChunker addresses these structural limitations
through its Structured Intermediate Representation
(SIR), enabling dynamic granularity adjustment
and explicit topology awareness.
2.2 Hierarchical Representation and Context
Augmentation in RAG
The efficacy of RAG systems is significantly in-
fluenced by the structured presentation of retrieved
passages, as different queries necessitate varying
levels of abstraction. AutoChunker (Jain et al.,
2025) organizes information into tree structures but
focuses primarily on static noise suppression rather
than dynamic adjustment. Late Chunking (Gün-
ther et al., 2024) applies context-aware embeddings
before slicing to retain long-range dependencies;
however, it ultimately outputs flat lists that are of-
ten insufficient to capture deep physical topologies,
such as nested section-item relationships.
Inspired by the Mixture of Experts (MoE), MoC
(Zhao et al., 2025) and MoG (Zhong et al., 2025)
treat various chunking strategies as expert mod-
ules managed by query-driven routers; however,
their performance remains bounded by the underly-
ing heuristics. While hierarchical retrieval mecha-
nisms like Parent-Child Indexing attempt context
augmentation, child chunks often degenerate into
“semantic islands” due to a lack of intrinsically in-
dependent context. Furthermore, many advanced
RAG frameworks rely on heavyweight external
graph databases, introducing substantial I/O over-
head during preprocessing. Liu et al. (2024) noted
that context utilization decays with position, sug-
gesting that simple hierarchical associations are in-
sufficient for high-precision retrieval. Unlike these
methods, TopoChunker utilizes a lightweight, in-
memory SIR with explicit parent-child-sibling re-
lationships. By dynamically injecting Topological
Lineage, it effectively facilitates multi-hop reason-
ing over previously fragmented knowledge.
2.3 Agentic Workflows and Adaptive
Diagnostics for Document Intelligence
The integration of autonomous agents introduces
“reasoning-before-action” capabilities to handle un-
structured document anomalies. Traditional mod-
els based on LSTM or BERT are limited by nar-
row context windows and single-level processing.
RAG-oriented approaches like PIC (Wang et al.,
2025) use pseudo-instructions for guidance but ex-
hibit weak spatial awareness, struggling with het-

erogeneous elements like nested tables.
Although Vision-Language Models (VLMs) of-
fer superior layout perception, their high compu-
tational overhead often limits their viability for
large-scale deployment. Furthermore, while emerg-
ing agentic chunking frameworks introduce dy-
namic decision-making, they typically lack rigor-
ous quantitative diagnostics and closed-loop opti-
mization mechanisms. As a result, they tend to
apply LLM reasoning uniformly across all text,
which incurs excessive token expenditure on stan-
dard structural elements. To mitigate these ineffi-
ciencies, TopoChunker employs an Inspector Agent
designed around a cost-risk minimization objec-
tive to prevent redundant processing. Paired with
the Refiner Agent’s ReAct-driven diagnostic loop,
this "Diagnosis-Execution-Audit" paradigm strate-
gically allocates computationally expensive gen-
erative reasoning to high-complexity data points,
thereby improving both topological fidelity and
inference efficiency compared to existing agentic
workflows.
3 Method
3.1 Cognitive Perception and Adaptive
Routing
TopoChunker employs a three-stage agentic archi-
tecture to address the limitations of linear ETL
pipelines. As illustrated in Figure 1, the workflow
transitions from perception and topological mod-
eling to metadata assembly. The decision-making
of all agents is driven by reasoning-focused LLMs
(e.g., DeepSeek-R1 (DeepSeek-AI et al., 2025)).
The process begins in the Cognitive Perception
Layer, where the Inspector Agent evaluates the
trade-off between potential information loss and
computational cost.
3.1.1 Active Probing and Path Selection
The Inspector Agent employs an active probing
paradigm. Rather than consuming the entirety of
the raw document, the agent is provided solely
with a document identifier. Operating within the
ReAct framework, it autonomously invokes a suite
of lightweight API tools—such as metadata parsers,
layout density scanners, and hierarchy analyzers—
to perform sparse sampling. Through these tar-
geted queries, the agent incrementally constructs a
document-specific topology fingerprint ⃗f.
Crucially, this tool-driven decoupling com-
presses the document’s spatial and hierarchicalcharacteristics into a highly compact token foot-
print (typically under 200 tokens). By evaluating
this lightweight fingerprint instead of the full text,
the Inspector avoids optimizing a rigid numeric
objective function. Instead, it executes a prompt-
driven reasoning loop to balance three dimensions:
token expenditure, risk of information loss, and
structural fidelity. This enables the system to dy-
namically route heterogeneous inputs through one
of three specialized extraction paths at a negligible
computational cost:
Path 1: Structural Rule. This path is activated
for digital documents with clear heading hierar-
chies when the agent determines that complex vi-
sual parsing is computationally redundant. The
structural processor performs dynamic hierarchy
detection by pre-scanning the document to iden-
tify recurring logical patterns (e.g., “Chapter 1,”
“Section N,” or nested articles). These patterns are
statistically mapped to specific Markdown levels,
ensuring the final output explicitly preserves the
tree-based skeleton of the original document.
Path 2: Semantic Flow. Designed for unstruc-
tured narratives where rule-based matching risks
high information loss, this path prioritizes semantic
continuity. Because these documents lack explicit
physical hierarchies, the Inspector directly invokes
the LLM-based Semantic_Slicer tool (detailed
in Section 3.3) to evaluate the flat text. This de-
composes the narrative into preliminary segments
based on natural topic transitions. These sequential
segments then serve as foundational nodes to con-
struct a localized SIR, maintaining compatibility
with downstream metadata assembly and disam-
biguation phases.
Path 3: Visual Topology. For scanned files or
documents with complex spatial layouts, the In-
spector triggers a vision-centric pipeline. This
path leverages layout-aware VLM extractors (e.g.,
MinerU (Wang et al., 2024)) for physical parsing.
Rather than flattening the output into linear text
streams, TopoChunker explicitly preserves the ex-
tracted coordinates and the physical integrity of
non-textual elements (e.g., tables and charts), di-
rectly mapping this spatial data into our logical
Markdown representation.
3.2 Topological Modeling and SIR
Construction
Although the three routing paths employ distinct
extraction mechanisms, their outputs are unified

Path 1: Structural RuleTopological Pruner
Input Document
Chunks Output ChunksLayer 1:  Cognitive Perception Layer Layer 2:   Execution Layer
Layer 3:   Audit LayerPerception Toolkit
Cost / Fidelity / Risk 
CheckReAct  
Loop Inspector Agent
Meta Reader
Layout Probe
Regex TesterPath 2: Semantic Flow
Path 3 : Visual Topology
SIR
Refinement Toolkit
Independence / Completeness /
Size CheckReAct  
Loop Refiner Agent
SIR Query
Semantic Slicer
Chunk EnhancerPath Information
Semantic Signature
Context SupplementFigure 1: The overall architecture of TopoChunker. The dual-agent framework operates across three layers:
Cognitive Perception (Inspector Agent for adaptive routing), Execution (Topological Pruner and SIR construction),
and Audit (Refiner Agent for context disambiguation). These layers interact via a shared Structured Intermediate
Representation (SIR) to form a closed-loop Diagnosis-Execution-Audit pipeline.
into a standardized Markdown format. Following
this standardization, the system constructs the topo-
logical model. To mitigate the “contextual fragmen-
tation” inherent in flat chunking, the Topological
Pruner parses the unified output to build the hi-
erarchical tree topology TSIRusing a stack-based
traversal algorithm. Each node v∈ T SIRis defined
by three core attributes:
First, it captures the Topological Lineage ( v.Tid)
by maintaining a hierarchy stack. This records
the absolute global coordinates (e.g.,/Chap1/Sec3)
for all child nodes, enabling explicit contextual
inheritance. By tracing this lineage, any isolated
leaf node can inherit macro-context from its an-
cestors. Second, it enforces Atomicity Locking
(v.Alock). Nodes representing indivisible visual
elements, such as tables or images, are flagged
(v.Alock=True ) to prevent physical splitting. Fi-
nally, it records a Token Audit ( v.W token ), pre-
calculating the token weight of the node to allow
the Refiner Agent to identify oversized subtrees in
O(1)time.
Systemically, TSIR is implemented as a
lightweight, in-memory object tree rather than a
heavyweight external graph database, minimizingexternal I/O overhead during preprocessing. Once
optimal chunking boundaries are determined, the
leaf nodes are flattened, and their topological prop-
erties are serialized as standard metadata payloads.
3.3 Semantic Refinement and Context
Assembly
The premise of the SIR topology is that text un-
der a specific heading natively forms a coherent
semantic unit. However, encoding arbitrarily long
structural blocks dilutes their representation in the
embedding space, degrading downstream retrieval
precision. To reconcile structural completeness
with retrieval efficiency, the Refiner Agent man-
ages the text segments via a ReAct paradigm (Yao
et al., 2023). To achieve this, it is equipped with
three tools: SIR_Query ,Semantic_Slicer , and
Chunk_Enhancer.
The agent executes the refinement and assembly
pipeline through a sequence of operations:
3.3.1 Capacity Auditing and Semantic Slicing
The refinement process begins with a capacity audit.
The agent evaluates each node in TSIRagainst an
empirical token threshold θ(set to 500 tokens in our

Table 1: Domain-Specific RAG Performance. Performance is reported using Recall@ k(R@k) and DCG@ k(D@k).
TopoChunker demonstrates consistently superior retrieval accuracy across both unstructured and highly structured
domains.
GutenQA GovReport
R@3 D@3 R@5 D@5 R@3 D@3 R@5 D@5
Recursive (FC200) 8.23 7.38 8.62 7.84 32.25 28.40 36.25 29.81
Semantic Chunker 14.49 12.43 16.56 13.29 55.37 48.96 61.75 51.39
Proposition-Level 47.23 42.27 53.22 45.73 66.17 61.36 73.88 59.44
LumberChunker 62.58 56.56 66.93 58.35 81.38 74.88 85.20 75.55
TopoChunker 64.59 56.95 70.29 59.03 83.26 76.59 86.58 76.93
implementation). If a node exceeds this limit, the
agent invokes the Semantic_Slicer . Internally,
this tool uses regular expressions to explicitly iden-
tify sentence boundaries and assigns sequential IDs
(e.g., [S0],[S 1], . . . ). These indexed sentences are
then passed to the LLM via a zero-shot prompt
to identify boundary IDs where significant topic
or temporal shifts occur. To prevent hallucination
errors, the generated IDs are deterministically vali-
dated against the valid sequence before execution.
This partitions oversized nodes into contextually
coherent sub-chunks without disrupting the under-
lying narrative flow.
3.3.2 Semantic Signature Generation
Once capacity constraints are met, the agent applies
theChunk_Enhancer to transform raw segments
into self-contained units. To maximize lexical dis-
criminability, the enhancer prompts the LLM to
generate a concise, 3-to-8 word thematic title that
abstracts the core topic, avoiding the simple extrac-
tion of the first sentence. This ensures the chunk
is indexed with a precise semantic signature rather
than relying solely on generalized macro-headers.
3.3.3 Topological Context Disambiguation
To resolve semantic fragmentation and elimi-
nate “semantic islands,” the agent performs a fi-
nal context-grounding step at the chunk level.
It audits the chunk’s payload for unresolved
pronominal anaphora, demonstrative references,
and isolated context-dependent terminology (e.g.,
domain-specific acronyms). Upon detecting these
contextual voids, the agent triggers SIR_Query .
Instead of executing external graph database
queries, SIR_Query deterministically traverses the
in-memory object tree. By navigating topological
pointers (e.g.,parent,prev_sibling), the agent re-
trieves entity definitions and macro-context fromancestor nodes. This resolved context is then ex-
plicitly fused into the chunk’s text, providing down-
stream models with a logically complete and struc-
turally grounded premise.
4 Experiments
4.1 Experimental Setup
Datasets.We evaluate our framework on two
benchmarks: GutenQA and GovReport. Specif-
ically, GutenQA comprises 100 public domain nar-
rative books serving as the retrieval corpus, paired
with 3,000 specific question-answer pairs for eval-
uation. For GovReport, we constructed a test set
by sampling 500 queries, evaluated following the
same inference methodology as GutenQA.
We compare TopoChunker against four represen-
tative paradigms to evaluate its effectiveness. To
ensure a fair comparison, all dense retrieval evalu-
ations use BGE-m3 as the underlying embedding
model.
•Fixed-Size Chunking (FC200): A standard
heuristic baseline that splits text into rigid
blocks of up to 200 tokens.
•Semantic Chunker(Xiao et al., 2024): A
dynamic method that determines boundaries
based on embedding similarity shifts at the sen-
tence level.
•Proposition-Level Chunking(Chen et al.,
2024): A granular strategy decomposing text
into self-contained atomic factual statements.
•LumberChunker(Duarte et al., 2024): A re-
cent approach using LLMs to predict optimal
breaks. To align with TopoChunker’s setup,
we deploy DeepSeek-R1 (DeepSeek-AI et al.,
2025) as the backbone for both LumberChun-
ker and our framework, uniformly setting the
sampling temperature to 0.1 to promote deter-

GutenQA Gov-reportFC200
Semantic Chunker 
Proposition-Level
LumberChunker
TopoChunker36.0 55.0
39.0 63.0
47.0 69.0
56.0 79.0
64.0 82.0
304050607080Accuracy (%)Figure 2: Generation Accuracy. A heatmap illustrating
the generation accuracy of various chunking methods
across the GutenQA and GovReport datasets. Darker
shades indicate higher accuracy.
ministic outputs.
4.2 Quantitative Evaluation
4.2.1 Retrieval Performance
As shown in Table 1, TopoChunker outperforms all
evaluated baselines across both unstructured and
highly structured domains, achieving the highest
performance in both absolute retrieval (Recall@ k)
and ranking quality (DCG@k).
Traditional methods exhibit limited robustness to
structural variance. On the GovReport benchmark,
FC200 and Semantic Chunker achieve lower R@3
scores of 32.25% and 55.37%, respectively, primar-
ily because they mechanically sever logical flow.
While Proposition-Level chunking performs mod-
erately better by isolating factual statements, its
excessive fragmentation often obscures the broader
context necessary for accurate dense matching.
Conversely, TopoChunker maintains contextual
integrity regardless of document complexity. On
the unstructured GutenQA dataset, it outperforms
the LLM-guided LumberChunker, achieving an
R@3 of64.59%. This indicates that our Seman-
tic Flow routing preserves self-contained contexts
even in flat texts.
The performance gap widens on the dense hierar-
chies of GovReport, where TopoChunker achieves
an R@3 of83.26%and a D@3 of76.59%. This
margin over the strongest baselines demonstrates
the effectiveness of our Topological Lineage and
Semantic Signature mechanisms. By synthesizing
abstract chunk titles and retaining ancestral meta-
data, TopoChunker mitigates “feature dilution” in
the embedding space, enabling precise semantic
matching for complex queries.4.2.2 Generation Accuracy
We evaluate end-to-end RAG generation using an
LLM-as-a-judge pipeline powered by GPT-4. To
validate the reliability of this automated metric and
mitigate potential biases, we conducted a manual
audit. We randomly sampled 100 generated QA
pairs across all methods for blind human expert
annotation. The human annotations achieved a
92% agreement rate with the GPT-4 judgments,
confirming the validity of our evaluation protocol.
As Figure 2 illustrates, TopoChunker outper-
forms all evaluated baselines across both datasets.
It achieves 64.0% accuracy on the unstructured
GutenQA (+8.0% absolute improvement over the
strongest baseline, LumberChunker) and reaches
82.0% on the highly structured GovReport dataset,
outperforming heuristic baselines like FC200 by a
wide margin.
This performance advantage is primarily at-
tributed to the framework’s topological context dis-
ambiguation. Baseline chunkers often arbitrarily
partition text, creating “semantic islands” with un-
resolved anaphora (e.g., dangling pronouns like “it”
or “the department”) and decoupled terminology.
In contrast, TopoChunker addresses these contex-
tual gaps by explicitly injecting inherited parent
headers, ancestral entities, and prerequisite defini-
tions into the retrieved segments. Providing the
generative LLM with a logically complete and self-
contained premise reduces context drift, thereby
minimizing hallucinations and yielding structurally
grounded responses.
4.3 Token Efficiency
TopoChunker dynamically scales token cost based
on document complexity. Unlike LumberChun-
ker’s constant 1.65× overhead (due to iterative full-
text scanning), TopoChunker decouples physical
extraction from semantic reasoning. LLM con-
sumption is strictly limited to the Inspector for
routing and the Refiner for auditing, avoiding re-
dundant processing of standard structures.
We evaluate token costs on a mixed-complexity
corpus (60% simple, 40% complex layouts) us-
ing a standard 10,000-token document (Figure 3).
For simple documents, TopoChunker consumes
8,500 tokens for routine context disambiguation,
requiring fewer tokens than full-text scanning. For
complex cases requiring VLM layout parsing, con-
sumption peaks at 18,800 tokens.
Despite this premium over LumberChunker

Lumber Topo-S Topo-C Topo-A0500010000150002000025000Total LLM Tokens16500
850018800
12620S: Simple Doc
C: Complex-Max
A: Avg. MixedFigure 3: Complexity-Aware Token Consumption Anal-
ysis. TopoChunker dynamically adjusts token expen-
diture based on document complexity, achieving lower
average costs than static agentic baselines.
(16,500 tokens) on exclusively complex files,
TopoChunker’s weighted average consumption
across the mixed corpus is 12,620 tokens ( 1.26× ).
This corresponds to a 23.5% total cost reduction
compared to LumberChunker, demonstrating that
TopoChunker balances semantic integrity with in-
ference efficiency.
Table 2: Ablation Study: Impact of Refiner (Per-
formance) and Inspector (Efficiency) on a mixed-
complexity corpus. Token cost is normalized against
input length.
Method Recall@3 Acc. (%) Token Cost
LumberChunker 81.38 79.0 1.65x
Topo (w/o Refiner) 76.50 71.00.85x
Topo-Static-P1 (Rule Only) 77.80 73.5 0.80x
Topo-Static-P2 (Semantic Only) 79.20 76.8 0.95x
Topo-Static-P3 (VLM Only) 82.85 81.5 1.88x
TopoChunker (Full) 83.26 82.0 1.26x
4.4 Module-wise Ablation Study
We conduct an ablation study on the mixed-
complexity corpus to isolate the contributions of
theRefiner Agent(semantic quality) and theIn-
spector Agent(routing efficiency).
4.4.1 Impact of the Refiner Agent
As Table 2 shows, extracting raw topology is in-
sufficient for high-quality retrieval. Removing the
Refiner (w/o Refiner) leads to a 6.76-point degrada-
tion in Recall@3 and an 11.0% decrease in genera-
tion accuracy. Without capacity auditing, oversized
nodes suffer from vector dilution, and unresolved
anaphora create “semantic islands” that introduce
ambiguity to the generative LLM.4.4.2 Impact of the Inspector Agent
Replacing the dynamic Inspector with static rout-
ing pipelines highlights its role in balancing cost
and performance. Lightweight static paths (Topo-
Static-P1/P2) lower token costs (0.80x and 0.95x)
but fail to adequately process complex layouts, de-
creasing Recall@3 to 77.80% and 79.20%.
Conversely, forcing all documents through the
VLM pipeline (Topo-Static-P3) yields high accu-
racy (82.85% Recall@3) but incurs a substantial
1.88x token overhead. The fullTopoChunkerdy-
namically routes simple documents, achieving the
highest overall accuracy (83.26% Recall@3) while
maintaining an average token cost of 1.26x. This
represents a 32.9% cost reduction over the forced
VLM approach and a 23.5% reduction over Lum-
berChunker.
5 Conclusion
In this paper, we introduce TopoChunker, a
topology-aware framework for document chunk-
ing in RAG pipelines. By mapping heterogeneous
documents onto a Structured Intermediate Rep-
resentation (SIR) via a dual-agent architecture—
comprising an Inspector Agent for adaptive rout-
ing and a Refiner Agent for semantic auditing—
our approach mitigates “semantic islands” and re-
solves anaphoric conflicts. Evaluations on unstruc-
tured narratives (GutenQA) and highly structured
reports (GovReport) demonstrate that TopoChun-
ker achieves state-of-the-art retrieval and genera-
tion accuracy. These results indicate that preserv-
ing and explicitly injecting topological metadata is
critical for robust RAG systems.
Limitations
While TopoChunker improves retrieval quality and
structural fidelity, the current framework has two
primary limitations:
Upstream Parsing Dependency:For documents
with complex visual layouts routed through Path
3, the system relies on external VLM extractors
(e.g., MinerU) for physical parsing. Consequently,
upstream OCR inaccuracies or layout misclassi-
fications can cascade, propagating errors to the
downstream Structured Intermediate Representa-
tion (SIR).
Inference Latency:Although our routing mecha-
nism reduces overall token expenditure, the agentic
paradigm inherently introduces higher inference

latency compared to deterministic heuristic chun-
kers. The reliance on instruction-tuned LLMs for
dynamic routing and semantic auditing requires
sequential API calls. This latency overhead poses
challenges for real-time or latency-sensitive RAG
applications.
Ethics Statement
In this work, we conduct a manual human evalu-
ation to validate the reliability of our automated
LLM-as-a-judge pipeline. We are committed to
adhering to the ACL Ethics Policy and address the
relevant ethical considerations below:
Human Labor and Participation:The human
evaluation involved annotating 100 randomly sam-
pled generated QA pairs to assess generation ac-
curacy. This process was conducted entirely by
the authors of this paper. As the evaluation was
performed by the research team as part of their
academic duties, no external annotators or crowd-
sourcing platforms were used, and no additional
financial compensation was applicable.
Anonymity and Bias Control:To ensure the
objectivity of the evaluation, the generated outputs
were shuffled and anonymized so that the authors
were unaware of which specific method (baseline
vs. TopoChunker) produced each response during
the annotation process. No personally identifiable
information (PII) was involved in the evaluation.
Data Risks and Content Considerations:The
datasets utilized in this study, GutenQA and Gov-
Report, are publicly available benchmarks consist-
ing of public domain literature and government
documents. We have screened the sampled data
and confirmed that they do not contain sensitive
personal information, toxic language, or offensive
content that could pose ethical risks.
References
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024. Dense X retrieval: What retrieval
granularity should we use? InProceedings of the
62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 8452–8467, Bangkok, Thailand. Association
for Computational Linguistics.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capabilityin llms via reinforcement learning.Preprint,
arXiv:2501.12948.
Abel V . Duarte, João Marques, Miguel Graça, Mariana
Freire, Lin Li, and Arlindo L. Oliveira. 2024. Lum-
berchunker: Long-form narrative document segmen-
tation. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2024, pages 6473–6486,
Miami, Florida, USA. Association for Computational
Linguistics.
Michael Günther, Isabelle Mohr, Daniel James Williams,
Bo Wang, and Han Xiao. 2024. Late Chunking: Con-
textual chunk embeddings using long-context embed-
ding models.arXiv preprint arXiv:2409.04701.
Arihant Jain, Purav Aggarwal, and Anoop Saladi. 2025.
Autochunker: Structured text chunking and its evalu-
ation. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 6: Industry Track), pages 983–995. Association
for Computational Linguistics.
Nelson F. Liu, Kevin Lin, Peter Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Bin Wang, Chaochao Xu, Xiaomeng Zhao, Xinwei
Shen, Yiyi Su, Sihan Chen, Haojun Xu, Yutao Cai,
Xinyu Xia, Fan Bian, Jian Li, Jinke Jiang, Fudan
Zheng, Weikang Dong, Ruiqi Li, Zunruo Wang, and
Conghui He. 2024. Mineru: An open-source solution
for precise document content extraction.Preprint,
arXiv:2409.18839.
Zhitong Wang, Cheng Gao, Chaojun Xiao, Yufei Huang,
Shuzheng Si, Kangyang Luo, Yuzhuo Bai, Wenhao
Li, Tangjian Duan, Chuancheng Lv, and 1 others.
2025. Document Segmentation Matters for Retrieval-
Augmented Generation. InFindings of the Asso-
ciation for Computational Linguistics (ACL 2025),
pages 8063–8075. Association for Computational
Linguistics.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval (SIGIR ’24), pages 641–649. ACM.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InThe Eleventh International Conference
on Learning Representations (ICLR).
Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2025. MoC: Mixtures of text chunking learners for
retrieval-augmented generation system. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 5172–5189, Vienna, Austria. Associa-
tion for Computational Linguistics.

Junhan Zhao, Zonghai Ji, Peng Qi, Shuo Niu, Bing Tang,
Feiyang Xiong, and Ze Li. 2024. Meta-chunking:
Learning efficient text segmentation via logical per-
ception.arXiv preprint arXiv:2410.12788.
Zihan Zhong, Han Liu, Xiaobo Cui, Xinyu Zhang, and
Zheng Qin. 2025. Mix-of-granularity: Optimize the
chunking granularity for retrieval-augmented genera-
tion. InProceedings of the 31st International Confer-
ence on Computational Linguistics (COLING 2025),
pages 5756–5774, Abu Dhabi, UAE. Association for
Computational Linguistics.

A Resolving Contextual Fragmentation
(a) Unstructured Narrative (GutenQA Dataset)→Routed via Path 2: Semantic Flow
Source Document (Adventures of Huckleberry
Finn)TopoChunker Output (Self-Contained Atom)
...And next time Jim told it he said they rode him down to New
Orleans; and, after that, every time he told it he spread it more
and more, till by-and-by he said they rode him all over the
world, and tired him most to death, and his back was all over
saddle-boils.
Jim was monstrous proud about it , and he got so he
wouldn’t hardly notice the other niggers. Niggers would come
miles to hear Jim tell about it, and he was more looked up to
than any nigger in that country. ... Niggers would come from
all around there and give Jim anything they had, just for a sight
ofthat five-center piece ; but they wouldn’t touch it, because
the devil had had his hands on it. Jim was most ruined for a
servant, because he got stuck up on account of having seen the
devil and been rode by witches.
Well, when Tom and me got to the edge of the hilltop we
looked away down into the village and could see three or four
lights twinkling...[Path]Adventures of Huckleberry Finn→Content→
Segment X
[Signature]Jim’s Witch Story Fame and Pride
[Context Supplement]
→it: Jim’s story about being bewitched by witches.
→five-center piece: The five cents Tom laid on the table.
[Text]
“Jim was monstrous proud about it, and he got so he
wouldn’t hardly notice the other niggers. Niggers would
come miles to hear Jim tell about it, and he was more
looked up to than any nigger in that country. ... Niggers
would come from all around there and give Jim anything
they had, just for a sight of that five-center piece; but
they wouldn’t touch it, because the devil had had his
hands on it. Jim was most ruined for a servant, because
he got stuck up on account of having seen the devil and
been rode by witches.”
(b) Structured Document (GovReport Dataset)→Routed via Path 1: Structural Rule
Source Document (GovReport / GAO-19-504) TopoChunker Output (Structured JSONL
Atom)
...NASA awarded firm-fixed-price contracts in 2014 to Boeing
and SpaceX, valued at up to $4.2 billion and $2.6 billion,
respectively, for the development of crew transportation
systems... The Commercial Crew Program manages two
processes in order to support the contractors’ uncrewed test
flight...
Contractors Are Making Progress on Vehicles, but
Certification Date Remains Unclear
Both contractors have made progress building and testing
hardware, including SpaceX’s uncrewed test flight. But
continued schedule delays and remaining work for the
contractors andthe program create continued uncertainty
about when either contractor will be certified to begin
conducting operational missions to the ISS.
The program has made progress reviewing the contractors’
certification paperwork, but contractor delays in submitting
evidence for NASA approval may compound a ’bow wave’ of
work, which creates uncertainty about when either contractor
will be certified...[Path]GovReport→GAO-19-504
[Signature]Commercial Crew Program Certification
Delays and Schedule Uncertainty
[Context Supplement]
→Both contractors: Boeing and SpaceX.
→The program: The Commercial Crew Program.
[Text]
“Contractors Are Making Progress on Vehicles, but
Certification Date Remains Unclear paragraphs: Both
contractors have made progress building and testing
hardware, including SpaceX’s uncrewed test flight. But
continued schedule delays and remaining work for the
contractors and the program create continued
uncertainty...”
Figure 4: Qualitative Examples of TopoChunker output resolving contextual fragmentation.(a)On the GutenQA
dataset, the lack of heading hierarchy triggers Path 2: Semantic Flow. TopoChunker actively resolves semantic
islands by explicitly mapping dangling pronouns to their ancestral entities in the Context Supplement .(b)On the
GovReport dataset, explicit hierarchies trigger Path 1: Structural Rule. Generic references are successfully resolved
to specific entities, ensuring chunks remain semantically self-contained for downstream RAG.

B End-to-End Workflow: From Raw PDF to Semantic Atoms
Input Document (PDF Content)
Chapter 1: General Provisions on Data Security Management
Article 1 [Data Classification and Grading Scope]Based on the impact of data leakage on the Company and
national security, data is categorized into three levels and five categories.
(1)Level 1 (Top Secret): Refers to data that, if leaked, would directly lead to the loss of the Company’s
core competitiveness or seriously threaten national security. The scope is strictly limited to: Core Technology:
Proprietary large model training weights (Checkpoints), unpatented chip architecture diagrams, and hardware
security module (HSM) Root Keys.
(2)Level 2 (Confidential): Refers to data that, if leaked, could cause significant damage to the Company’s
reputation or finances. The scope includes: User Privacy: Real-name user databases containing ID numbers,
facial recognition features, and bank card numbers. Business Secrets: Key Account (KA) lists in CRM systems,
exclusive supply chain pricing agreements, and quarterly detailed financial statements (unpublished). Source
Code: Application-layer code for non-core business systems and API interface documentation.
(3)Level 3 (Internal Use): Refers to data known only to internal employees, including administrative notices,
desensitized weekly operational reports, internal directories, and general technical documentation.
Article 2 [Access Control Strategies for Each Level]Unauthorized access to sensitive data is strictly prohibited.
The circulation and access of data at all levels must comply with the following mandatory regulations:
(1) For Level 1 Data: Implement a "Zero Trust" access strategy. It must be viewed via intranet terminals within a
physically isolated Clean Room. Any form of downloading, exporting, or transmission via the Internet is strictly
prohibited. Access requires dual approval (Double-Check) from the CTO and the Compliance Director.
(2) For Level 2 Data: Export is strictly prohibited in principle. If export is necessary, it must be approved by a
department head at L8 level or above, and the system must forcibly add an invisible Digital Watermark containing
the employee ID and timestamp.
(3) For Level 3 Data: Access and circulation are permitted on Company intranet office terminals, but uploading to
unauthorized public cloud storage platforms is prohibited.
(4) For Level 4 Data (Public/Desensitized): Implement a "Pre-release Review" system. Although such data is
permitted for external disclosure (e.g., financial report summaries, marketing materials), it must undergo content
compliance review by PR or Legal departments before release.
(5) For Audit and Monitoring: Implement full-link "Behavioral Profiling" monitoring. All cross-level data access
behaviors must retain immutable Audit Logs for a retention period of no less than 12 months. The system must
deploy UEBA policies to trigger real-time blocking.
Chapter 2: Organizational Guarantee and Legal Liability
Article 1 [Organizational Structure and Responsibilities]The Company establishes the Data Security Committee
(DSC) as the highest decision-making body, with the CEO serving as the group leader. Heads of business lines (BP)
are the primary persons responsible for data security in their respective departments and must sign the "Annual Data
Security Responsibility Letter."
Article 2 [Data Security Emergency Response]In the event of security incidents such as data leakage, tampering,
or loss, the discoverer shall report to the Security Operations Center (SOC) within 15 minutes. For major leakage
incidents involving Level 1 or Level 2 data, the SOC must activate the Level 1 Emergency Plan within 1 hour.
Article 3 [Personnel Training and Assessment]The HR Department, in conjunction with the Information Security
Department, shall organize company-wide data security awareness training quarterly. Personnel in core positions
involving contact with Level 1 or Level 2 data must pass the "Data Security Compliance Exam" annually.
Figure 5: The raw text content extracted from the input PDF document used in the evaluation case study. This
document features complex hierarchical structures (Chapters, Articles, Clauses) and domain-specific terminology
(e.g., "L8 level", "Zero Trust").

Intermediate Representation (Markdown Stream)
# Chapter 1: General Provisions on Data Security Management
## Article 1 [Data Classification and Grading Scope]Based on the impact of data leakage on the Company and
national security, data is categorized into three levels and five categories.
### (1) Level 1 (Top Secret):Refers to data that, if leaked, would directly lead to the loss of the Company’s core
competitiveness or seriously threaten national security. The scope is strictly limited to: Core Technology: Proprietary large
model training weights (Checkpoints), unpatented chip architecture diagrams, and hardware security module (HSM) Root
Keys.
### (2) Level 2 (Confidential):Refers to data that, if leaked, could cause significant damage to the Company’s
reputation or finances. The scope includes: User Privacy: Real-name user databases containing ID numbers, facial
recognition features, and bank card numbers. Business Secrets: Key Account (KA) lists in CRM systems, exclusive supply
chain pricing agreements, and quarterly detailed financial statements (unpublished)...
### (3) Level 3 (Internal Use):Refers to data known only to internal employees, including administrative notices,
desensitized weekly operational reports, internal directories, and general technical documentation.
## Article 2 [Access Control Strategies for Each Level]Unauthorized access to sensitive data is strictly prohibited. The
circulation and access of data at all levels must comply with the following mandatory regulations:
### (1) For Level 1 Data:Implement a "Zero Trust" access strategy. It must be viewed via intranet terminals within
a physically isolated Clean Room. Any form of downloading, exporting, or transmission via the Internet is strictly
prohibited...
### (2) For Level 2 Data:Export is strictly prohibited in principle. If export is necessary, it must be approved by a
department head at L8 level or above, and the system must forcibly add an invisible Digital Watermark containing the
employee ID and timestamp.
...
# Chapter 2: Organizational Guarantee and Legal Liability
## Article 1 [Organizational Structure and Responsibilities]The Company establishes the Data Security Committee
(DSC) as the highest decision-making body, with the CEO serving as the group leader. Heads of business lines (BP) are the
primary persons responsible for data security in their respective departments...
## Article 2 [Data Security Emergency Response]In the event of security incidents such as data leakage, tampering, or
loss, the discoverer shall report to the Security Operations Center (SOC) within 15 minutes. For major leakage incidents
involving Level 1 or Level 2 data, the SOC must activate the Level 1 Emergency Plan within 1 hour.
...
Figure 6: The standardized Structured Intermediate Representation (SIR) generated in Stage 2. The parser has
successfully mapped visual font cues to a logical Markdown stream (#, ##, ###), preserving the hierarchical skeleton
for the downstream Refiner Agent.

Full-Content Knowledge Base (JSONL Output)
[
{
"path": "Chapter 1: General Provisions > Article 1",
"signature": "Data Classification and Grading Scope",
"Context Supplement": {
"three levels": "Level 1 (Top Secret), Level 2 (Confidential), and Level 3 (Internal Use)",
"five categories": "Core Technology, User Privacy, Business Secrets, Source Code, and General Internal
Docs"
},
"text": "Article 1 [Data Classification and Grading Scope] Based on the impact of data leakage on the Company
and national security, data is categorized intothree levelsandfive categories. (1) Level 1 (Top Secret): Refers to
data that, if leaked, would directly lead to the loss of the Company’s core competitiveness. Scope limited to: Core
Technology: Proprietary large model training weights (Checkpoints), unpatented chip architecture diagrams, and
hardware security module (HSM) Root Keys... (2) Level 2 (Confidential): ... Business Secrets: Key Account (KA)
lists in CRM systems, exclusive supply chain pricing agreements..."
},
{
"path": "Chapter 1: General Provisions > Article 2",
"signature": "Access Control Strategies for Each Level",
"Context Supplement": {
"Level 1 Data": "Top Secret data (e.g., Core Technology, large model weights, Root Keys) as defined in
Article 1",
"Level 2 Data": "Confidential data (e.g., User Privacy, Business Secrets, Source Code) as defined in Article
1"
},
"text": "Article 2 [Access Control Strategies for Each Level] Unauthorized access to sensitive data is strictly
prohibited. The circulation and access of data at all levels must comply with the following mandatory regulations: (1)
ForLevel 1 Data: Implement a "Zero Trust"access strategy... (2) ForLevel 2 Data: Export is strictly prohibited in
principle. If export is necessary, it must be approved by a department head at L8 level or above, and the system must
forcibly add an invisible Digital Watermark containing the employee ID and timestamp."
},
{
"path": "Chapter 2: Organizational Guarantee and Legal Liability > Article 1",
"signature": "Organizational Structure and Responsibilities",
"Context Supplement": {
"data security": "The protection of Level 1, Level 2, and Level 3 categorized data across all business lines",
"Annual Data Security Responsibility Letter": "A mandatory internal accountability document linking
department heads to data compliance"
},
"text": "Chapter 2: Organizational Guarantee and Legal Liability. Article 1 [Organizational Structure and
Responsibilities] The Company establishes the Data Security Committee (DSC) as the highest decision-making body,
with the CEO serving as the group leader. Heads of business lines (BP) are the primary persons responsible fordata
securityin their respective departments and must sign the "Annual Data Security Responsibility Letter.""
}
]
Figure 7: The structured JSONL output generated by TopoChunker. TopoChunker moves beyond trivial dictionary
definitions to perform cross-chunk semantic resolution—such as automatically fetching the Article 1 definitions
forLevel 1 Data andLevel 2 Data when processing Article 2—significantly mitigating retrieval ambiguity for
downstream LLMs. Furthermore, because the source document features a large number of tables and complex
layouts, the Inspector Agent adaptively routes it through Path 3: Visual Topology, ensuring the structural integrity
of these non-textual elements.

C Agent System Prompts
Table 3 presents the core system prompts utilized by the dual-agent architecture. The Inspector Agent
executes the multi-criteria reasoning loop for adaptive routing, while the Refiner Agent enforces strict
verbatim preservation during semantic auditing and contextual disambiguation.
Table 3: System Prompts for the Inspector and Refiner Agents.
Part I: Inspector Agent
Agent Role:Strategic Cognitive Planner for document routing.
Core Task:You will be provided solely with a Document ID. You must autonomously invoke analytical tools to extract the
document’s structural fingerprint ( ⃗f) and determine the optimal routing pathway by heuristically balancing three dimensions:
computational cost (token expenditure), risk of information loss, and structural fidelity.
Step 1 (Active Probing via Tools):Using the provided Document ID, invoke the available lightweight API tools (e.g.,
analyze_metadata,probe_layout_complexity,test_rule_adherence) to perform sparse sampling. Do not attempt to process the
raw document text. Rely strictly on the JSON responses from these tools to assess visual element density, scanned status, and
textual hierarchy clarity.
Step 2 (Multi-Criteria Reasoning & Routing):Evaluate the tool outputs to balance the trade-offs and select one routing
path:
•Select VISUAL_TOPOLOGY (Path 3) for scanned or visually complex documents to prevent structural loss despite higher
computational cost.
• SelectSTRUCTURAL_RULE(Path 1) for digital documents with clear heading hierarchies to minimize cost.
• SelectSEMANTIC_FLOW(Path 2) for unstructured narratives to prioritize semantic continuity.
Output Format:Output the final decision strictly as a JSON object:
{
"selected_path": "VISUAL_TOPOLOGY" | "STRUCTURAL_RULE" | "SEMANTIC_FLOW",
"confidence_score": float,
"risk_score": float,
"reason": "string (concise justification based on tool observations)"
}
Part II: Refiner Agent
Agent Role:Expert Document Contextualization Agent.
Core Task:Execute semantic auditing and topological context disambiguation to transform raw document sections into
self-contained Semantic Atoms optimized for RAG.
Core Principle:Preserve the original text verbatim. You must not rewrite or paraphrase the text body. Your value is added
strictly through Metadata Headers and Contextual Disambiguation.
Strategy 1 (Title Generation):Create a concise (3-8 words) title capturing the main topic. Do not recycle the first sentence.
Maximize lexical discriminability.
Strategy 2 (Contextual Disambiguation):Scan for unresolved pronouns or isolated references. If the referent exists within
the chunk, do nothing. If missing, invoke the SIR_Query tool to retrieve ancestor nodes. If the definitive entity is found, create
a "Context Supplement" field.
Capacity Auditing & Operational Rules:If the system flags the current node as oversized ( is_oversized = True ), you
must invoke the Semantic_Slicer tool first to explicitly identify optimal sentence boundaries, and then process each resulting
segment independently. Do not merge them back.
Output Format:Wrap results in<final_chunks>tags as a JSON list of strings. Each string must follow this template:
[
"# {Semantic Signature}
Context Supplement: {Entity Mappings} // Omit if unnecessary
{Original Text Content Verbatim}"
]
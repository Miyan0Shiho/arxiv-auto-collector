# ORACAL: A Robust and Explainable Multimodal Framework for Smart Contract Vulnerability Detection with Causal Graph Enrichment

**Authors**: Tran Duong Minh Dai, Triet Huynh Minh Le, M. Ali Babar, Van-Hau Pham, Phan The Duy

**Published**: 2026-03-30 07:46:59

**PDF URL**: [https://arxiv.org/pdf/2603.28128v1](https://arxiv.org/pdf/2603.28128v1)

## Abstract
Although Graph Neural Networks (GNNs) have shown promise for smart contract vulnerability detection, they still face significant limitations. Homogeneous graph models fail to capture the interplay between control flow and data dependencies, while heterogeneous graph approaches often lack deep semantic understanding, leaving them susceptible to adversarial attacks. Moreover, most black-box models fail to provide explainable evidence, hindering trust in professional audits. To address these challenges, we propose ORACAL (Observable RAG-enhanced Analysis with CausAL reasoning), a heterogeneous multimodal graph learning framework that integrates Control Flow Graph (CFG), Data Flow Graph (DFG), and Call Graph (CG). ORACAL selectively enriches critical subgraphs with expert-level security context from Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), and employs a causal attention mechanism to disentangle true vulnerability indicators from spurious correlations. For transparency, the framework adopts PGExplainer to generate subgraph-level explanations identifying vulnerability triggering paths. Experiments on large-scale datasets demonstrate that ORACAL achieves state-of-the-art performance, outperforming MANDO-HGT, MTVHunter, GNN-SC, and SCVHunter by up to 39.6 percentage points, with a peak Macro F1 of 91.28% on the primary benchmark. ORACAL maintains strong generalization on out-of-distribution datasets with 91.8% on CGT Weakness and 77.1% on DAppScan. In explainability evaluation, PGExplainer achieves 32.51% Mean Intersection over Union (MIoU) against manually annotated vulnerability triggering paths. Under adversarial attacks, ORACAL limits performance degradation to approximately 2.35% F1 decrease with an Attack Success Rate (ASR) of only 3%, surpassing SCVHunter and MANDO-HGT which exhibit ASRs ranging from 10.91% to 18.73%.

## Full Text


<!-- PDF content starts -->

ORACAL: A Robust and Explainable Multimodal Framework for Smart Contract
Vulnerability Detection with Causal Graph Enrichment
Tran Duong Minh Daia,b, Triet Huynh Minh Lec, M. Ali Babarc, Van-Hau Phama,b, Phan The Duya,b,∗
aInformation Security Lab, University of Information Technology, Ho Chi Minh City, Vietnam
bVietnam National University, Ho Chi Minh City, Vietnam
cSchool of Computer Science and Information Technology, Adelaide University, Adelaide, Australia
Abstract
Although Graph Neural Networks (GNNs) have shown promise for smart contract vulnerability detection, they still face significant
limitations. Homogeneous graph models fail to capture the interplay between control flow and data dependencies, while heterogeneous
graph approaches often lack deep semantic understanding, leaving them susceptible to adversarial attacks. Moreover, most black-box
models fail to provide explainable evidence, hindering trust in professional audits. To address these challenges, we propose ORACAL
(Observable RAG-enhanced Analysis with CausAL reasoning), a heterogeneous multimodal graph learning framework that integrates
Control Flow Graph (CFG), Data Flow Graph (DFG), and Call Graph (CG). ORACAL selectively enriches critical subgraphs with
expert-level security context from Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), and employs a
causal attention mechanism to disentangle true vulnerability indicators from spurious correlations. For transparency, the framework
adopts PGExplainer to generate subgraph-level explanations identifying vulnerability triggering paths. Experiments on large-scale
datasets demonstrate that ORACAL achieves state-of-the-art performance, outperforming MANDO-HGT, MTVHunter, GNN-SC,
and SCVHunter by up to 39.6 percentage points, with a peak Macro F1 of 91.28% on the primary benchmark. ORACAL maintains
strong generalization on out-of-distribution datasets with 91.8% on CGT Weakness and 77.1% on DAppScan. In explainability
evaluation, PGExplainer achieves 32.51% Mean Intersection over Union (MIoU) against manually annotated vulnerability triggering
paths. Under adversarial attacks, ORACAL limits performance degradation to approximately 2.35% F1 decrease with an Attack
Success Rate (ASR) of only 3%, surpassing SCVHunter and MANDO-HGT which exhibit ASRs ranging from 10.91% to 18.73%.
Keywords:Smart Contract Vulnerability, Graph Learning, Multimodal, Vulnerability Detection, Explainable AI
1. Introduction
Smart contracts have become a foundational element of the
modern blockchain ecosystem, driving widespread adoption in
Decentralized Finance (DeFi) and Non-Fungible Tokens (NFTs).
Their ability to automatically execute business logic without
trusted intermediaries has fostered innovation; however, their
immutable nature presents a ”double-edged sword”. Once de-
ployed, code cannot be easily patched, meaning vulnerabilities
can lead to permanent and catastrophic financial losses.
In recent years, high profile incidents such as the Cetus Pro-
tocol exploit, in which an integer overflow allowed attackers
to mint spoof tokens and drain over $220 million in May 2025
[1], and the Zoth Protocol breaches involving combined logic
and governance failures [ 2], demonstrated that vulnerabilities
are evolving beyond simple logic errors toward complex eco-
nomic state manipulations. In early 2026, the TrueBit protocol
suffered a mathematical overflow flaw in its minting logic that
enabled unauthorized minting and extraction of ETH reserves,
∗Corresponding author
Email addresses:22520183@gm.uit.edu.vn(Tran Duong Minh Dai),
triet.h.le@adelaide.edu.au(Triet Huynh Minh Le),
ali.babar@adelaide.edu.au(M. Ali Babar),haupv@uit.edu.vn
(Van-Hau Pham),duypt@uit.edu.vn(Phan The Duy)resulting in an approximately $26.6 million loss [ 3]. Similarly,
DeFi liquidity protocols Aperture Finance and Swapnet were ex-
ploited due to insufficient input validation and arbitrary external
call vulnerabilities across multiple chains in January 2026 [ 4].
Reports indicate that DeFi exploits caused over $2.9 billion in
losses in 2025 alone [ 5], highlighting an urgent need for robust
automated security analysis in smart contract systems.
Traditional vulnerability detection methods, such as static
analysis tools like Slither or Mythril, rely heavily on predefined
rules and pattern matching. While effective for known vulner-
ability classes, they struggle with complex, novel logic errors
and lack deep semantic understanding [ 6,7]. Deep learning
approaches have emerged as a promising alternative, particularly
identifying vulnerabilities through graph representations of code
[8,9]. However, homogeneous graphs capture only limited as-
pects of the code, as they typically model all nodes and relations
uniformly, overlooking the diverse semantic roles of program
constructs [ 10]. Recent advances utilizing heterogeneous graphs
[11,12] have shown improved performance but often still treat
code semantics superficially. More critically, recent studies
have demonstrated that heterogeneous graph neural networks
are highly susceptible to adversarial attacks [ 13,14,15], where
minor structural perturbations can mislead the model. This vul-
nerability stems from the model’s reliance on shallow structural
Preprint submitted to Elsevier March 31, 2026arXiv:2603.28128v1  [cs.LG]  30 Mar 2026

features without a robust grounding in the code’s actual semantic
”intent” and security context, and it raises the broader question
of how to design vulnerability detectors that remain robust under
adversarially perturbed but semantically equivalent programs.
The rise of Large Language Models (LLMs) offers a new
frontier. Models like GPT-4 or CodeBERT can capture intri-
cate code semantics. However, current LLM-based vulnerability
detectors do not inherently resolve the adversarial robustness
issue identified above: they can be misled by carefully crafted
prompts or semantically equivalent but security-critical code
transformations, and still lack guarantees against adversarial ma-
nipulation of their inputs [ 16,17]. In other words, the adversarial
fragility observed in heterogeneous GNNs carries over to purely
LLM-based pipelines unless robustness is explicitly modeled.
Moreover, when used in isolation for vulnerability detection,
they suffer from two additional issues: a lack of domain-specific
grounding leading to “hallucinations” [ 18], and a “black-box”
nature that fails to provide explainable evidence for their predic-
tions [ 19]. In security audits, identifying a bug is not enough;
auditors need to knowwhyit is a bug andwherethe root cause
lies, as emphasized in both industrial auditing practice and em-
pirical studies of automated analyzers [20].
To address these gaps, we proposeORACAL(Observable
RAG enhanced Analysis with CausAL reasoning), a novel
framework for trustworthy smart contract vulnerability detection.
ORACAL integrates the structural rigor of heterogeneous graphs
with the rich semantic understanding of LLMs via Retrieval
Augmented Generation (RAG). Unlike existing methods that
rely solely on static graph features, ORACAL enriches the
most critical nodes in the graph with deep semantic context,
specifically “Operational Context” and “Security Analysis”,
generated by an LLM grounded in a curated knowledge
base of Solidity vulnerabilities. Here, “Operational Context”
summarizes the intended behavior of a function or contract,
its invariants, and environmental assumptions, encompassing
access control, token flow, and interaction patterns, while
“Security Analysis” captures potential vulnerability types,
exploit preconditions, and expected on-chain impacts at a level
suitable for human auditors.
Furthermore, we introduce a causal attention mechanism to
mitigate the “Clever Hans” effect [ 21], in which models may
rely on superficial artifacts, comprising node counts, common
control flow graph patterns, or frequent opcodes, rather than
actual security logic. This design ensures that the model focuses
on causal vulnerability patterns instead of spurious correlations
in the dataset and strengthens robustness against adversarial
perturbations. To enhance interpretability, ORACAL adopts
PGExplainer [ 22] to generate subgraph-level explanations that
identify the most influential nodes and edges contributing to
each vulnerability prediction, enabling auditors to quickly verify
the root cause of detected issues.
This paper makes the following key contributions:
•Heterogeneous Multimodal Graph Framework:We
propose a novel method to construct a heterogeneous
graph (combining CFG, DFG, and Call Graph) and enrich
its critical subgraphs using an LLM-based RAG pipeline.This injects expert-level security knowledge directly into
the graph structure.
•Causal Attention Learning:We design a dual-branch
graph neural network that explicitly disentangles causal
features (derived from RAG enrichment) from spurious
features (node contents), significantly improving general-
izability across different datasets.
•Explainability and Trustworthiness:We enhance trans-
parency by employing PGExplainer to provide subgraph-
level explanations for detected vulnerabilities, achieving
32.51% MIoU when evaluated against manually annotated
vulnerability triggering paths. Furthermore, we evaluate
robustness through adversarial attacks, demonstrating that
our enriched context significantly reduces the Attack Suc-
cess Rate (ASR) to approximately 3%, compared to up
to 19% for existing heterogeneous GNN detectors (specif-
ically, SCVHunter ASR 18.73%, MANDO-HGT ASR
12.41%, GNN-SC ASR 11.04%, and MTVHunter ASR
10.91%).
•Comprehensive Evaluation:We evaluate ORACAL on
large-scale benchmark datasets derived from real-world
Ethereum smart contracts (SoliAudit [ 23], CGTWeakness
[24]), along with two out-of-distribution datasets that
also serve as explainability benchmarks: the real-world
DeFi/NFT vulnerability test set DAppScan [ 25] and
the Line-Level Manually Annotated Vulnerabilities
dataset (LLMA V) [ 26]. Our results show ORACAL
achieves state-of-the-art performance with a Macro
F1 of 91.28% on the primary test set, improving over
prior heterogeneous GNN detectors ranging from
MANDO-HGT (82.63%), MTVHunter (78.10%), and
GNN-SC (70.74%) to SCVHunter (51.62%) by up to 39
percentage points in F1 [ 27,28], and maintains strong
generalization on out-of-distribution datasets (Macro F1
of 91.8% on CGTWeakness and 77.1% on DAppScan),
confirming its robustness in real-world scenarios.
Paper structure. Section 2 reviews existing approaches to
smart contract vulnerability detection and positions our work
relative to prior methods. Section 3 presents the ORACAL
framework in detail. Section 4 describes our experimental design
and evaluation results. Section 5 analyzes threats to the validity
of our study, and Section 6 concludes the paper and outlines
future research directions.
2. Related Work
This section reviews prior work on smart contract vulnerabil-
ity detection. As summarized in Table 1, we categorize existing
approaches into three groups: Rule-based Static Analysis, Homo-
geneous Graph Learning, and Heterogeneous Graph Learning.
Compared to these methods, our proposed ORACAL framework
introduces a Heterogeneous Multimodal Graph enriched with
dynamic RAG-based security knowledge and provides explicit
explainability via causal attention and subgraph-level explana-
tions, which are capabilities absent from all prior approaches.
2

Table 1: Comparison of Our Work with Related Smart Contract Vulnerability Detection Methods
Method Year Category Technique Semantic Explainability
Oyente [29] 2016 Static Analysis Symbolic Execution No Yes
Slither [30] 2018 Static Analysis Rule-based No Yes
Peculiar [31] 2019 Deep Learning Homogeneous Graph No No
GNN-SC [28] 2021 Deep Learning Homogeneous Graph No No
Mando-HGT [32] 2023 Deep Learning Heterogeneous Graph No No
SCVHUNTER [11] 2024 Deep Learning Heterogeneous Graph No No
MTVHunter [27] 2025 Deep Learning Heterogeneous Graph Static Knowledge No
ORACAL (Ours) 2026 Deep Learning Heterogeneous Multimodal Graph Dynamic Knowledge Yes
2.1. Static Analysis
As summarized in Table 1, static analysis tools exemplified
by Slither [ 30] and Oyente [ 29] analyze Solidity source code or
EVM bytecode using symbolic execution, control flow reason-
ing, and predefined vulnerability patterns. These tools provide
rule-level traceability and interpretable warnings, which make
them suitable for manual auditing workflows.
Nevertheless, their detection capability is inherently bounded
by handcrafted rules and predefined signatures. As a result, they
primarily identify known vulnerability patterns and struggle
to generalize to unseen or evolving attack strategies. Ghaleb
and Pattabiraman [ 33] report substantial false-positive and false-
negative rates across widely used smart contract analyzers. Their
findings indicate that rule-based engines heavily rely on prede-
fined vulnerability patterns, thereby limiting their reliability
when analyzing complex real-world contracts.
2.2. Graph Learning Approaches
Homogeneous Graph Models.To move beyond hand-
crafted rules, early deep learning approaches model smart con-
tracts as graphs and apply graph neural networks for vulner-
ability detection. Peculiar [ 31] constructs a critical data flow
graph that focuses specifically on security-sensitive operations
and dangerous nodes, aiming to reduce noise and emphasize
vulnerability-relevant dependencies. GNN-SC [ 28] formulates
smart contract vulnerability detection as a multi-label classifi-
cation problem over control flow graphs and employs a Graph
Convolutional Network to learn structural representations from
execution paths.
These homogeneous graph models improve generalization
compared to static analyzers by learning structural features di-
rectly from program representations. However, they encode a
single type of structural dependency, either data flow or con-
trol flow, within a unified graph schema. Such single-view
modeling limits the ability to jointly reason about interactions
across multiple semantic dimensions, which is often required
for detecting complex vulnerabilities involving state transitions,
cross-function interactions, or intertwined control and data de-
pendencies.
Heterogeneous Graph Models.To overcome the structural
constraints of homogeneous graphs, recent studies construct het-
erogeneous program graphs that integrate multiple semantic rela-
tions within a unified framework. MANDO-HGT [ 32] builds het-
erogeneous contract graphs (HCGs) from either source code orbytecode, incorporating control-flow and function-call relations,
and employs heterogeneous graph transformers with customized
meta-relations to model diverse node and edge types for both
contract-level and line-level vulnerability detection. SCVHunter
[11] designs a heterogeneous semantic graph based on intermedi-
ate representations and applies a heterogeneous graph attention
network to capture structural and semantic dependencies, while
allowing lightweight expert guidance to emphasize critical nodes.
These approaches demonstrate that multi-relational modeling
provides richer contextual representations than single-structure
graph encodings.
Despite these structural advances, heterogeneous graph mod-
els remain primarily syntax-driven. Node relationships are typi-
cally derived from static structural dependencies, without explic-
itly addressing noise interference or missing semantic informa-
tion at the bytecode level. MTVHunter [ 27] introduces a multi-
teacher framework to enhance bytecode vulnerability detection,
incorporating an instruction denoising teacher and a seman-
tic complementary teacher with neuron distillation to transfer
opcode-level knowledge. However, its semantic enhancement
remains largely static and task-specific, lacking mechanisms to
dynamically adapt to evolving vulnerability patterns or higher-
level reasoning requirements.
2.3. Explainability and Robustness Challenges
As highlighted in Table 1, a common failing across almost
all graph-based deep learning methods is the lack of explain-
ability. They operate as black boxes, providing a prediction
without offering the “why,” which is a critical requirement for
trust in security auditing. A comprehensive survey by Li et al.
[34] highlights that existing explanation techniques for graph
neural networks are still insufficient for delivering faithful and
human-interpretable justifications. In the context of vulnerabil-
ity detection, Chu et al. [ 35] propose a counterfactual explana-
tion framework to identify minimal structural changes that alter
model predictions, demonstrating that additional mechanisms
are required to interpret graph-based detectors. Similarly, Cao
et al. [ 36] introduce a causal explanation approach to improve
both interpretability and robustness of GNN-based vulnerability
detection systems. These studies indicate that explainability is
not inherently supported by standard graph neural architectures
and must be explicitly incorporated.
Robustness is another critical concern. Graph neural net-
works depend heavily on structural patterns learned from graph
3

topology. Prior studies have shown that graph-based models can
be sensitive to structural perturbations and spurious correlations
[37,14,15]. Minor modifications to nodes or edges may signif-
icantly alter predictions even when program semantics remain
unchanged. This structural dependency makes heterogeneous
graph models potentially vulnerable to adversarial manipulation,
raising concerns for deployment in obfuscated or adversarial
blockchain environments.
2.4. Summary
In summary, static analyzers provide interpretable yet rule-
constrained detection with limited generalization. Homoge-
neous graph models improve learning capacity but offer re-
stricted structural coverage, while heterogeneous graph models
enhance multi-relational representation yet remain syntax-driven
and lack intrinsic semantic reasoning. Graph-based deep learn-
ing approaches also face persistent challenges in explainability
and robustness. To bridge this gap between structural mod-
eling and deep semantic understanding, ORACAL integrates
Retrieval-Augmented Generation (RAG) with LLM-derived se-
curity knowledge into heterogeneous graph representations, and
employs a causal attention mechanism to distinguish genuine
vulnerability indicators from spurious correlations, aiming for
accurate, robust, and interpretable detection suitable for practical
smart contract auditing.
3. Methodology
We propose ORACAL (Observable RAG enhanced Analysis
with CausAL reasoning), a novel framework that synergizes Het-
erogeneous Graph Neural Networks with Retrieval-Augmented
Generation (RAG) and Causal Attention. The workflow consists
of four phases: (1) Graph Construction, (2) Critical Node Se-
lection, (3) RAG-based Semantic Enrichment, and (4) Causal
Disentanglement Learning. Figure 1 illustrates the overall archi-
tecture.
Given the Solidity source code of a contract, ORACAL
first extracts control-flow, data-flow, and call relationships from
the code. The pipeline then proceeds through four successive
phases:
1.Phase (1) – Graph Construction:Build a heterogeneous
contract graph from the extracted homogeneous graphs
(CFG, DFG, and call graph) and obtain initial node em-
beddings with GraphCodeBERT [38].
2.Phase (2) – Critical Node Selection:Compute an impor-
tance score for each node and extract a connected top- k
subgraph that summarizes the most security-critical exe-
cution region while preserving local context.
3.Phase (3) – RAG-based Semantic Enrichment:Trans-
late the selected subgraph into a textual description, query
a RAG pipeline over a security corpus, and encode the
returned natural-language annotations for nodes and edges
with GraphCodeBERT to obtain enriched features.
4.Phase (4) – Causal Attention Learning:Feed both orig-
inal and enriched node features into a causal-attention
GNN with dual branches (causal vs. spurious) to perform
vulnerability classification.The causal-attention GNN in Phase (4) is trained end-to-
end under a multi-task loss (Section 3.4.2) that (i) supervises
vulnerability prediction from the causal branch, (ii) regularizes
the spurious branch towards high-entropy, near-uniform out-
puts, and (iii) enforces a contrastive separation between the two
representations. This shared objective encourages ORACAL
to ground its decisions in semantically enriched, causally rele-
vant evidence while actively unlearning dataset-specific artifacts,
thereby improving robustness under distribution shift. Formally,
given the Solidity source code of a contract, ORACAL outputs a
multi-label vector ˆy∈{ 0,1}Cover Cvulnerability types (specif-
ically, ˆy=[1,0,1,1] indicates that three of four vulnerability
categories are detected for this contract).
3.1. Heterogeneous Graph Construction
Prior work on smart contract analysis has shown that model-
ing contracts as graphs that capture control and data dependen-
cies substantially improves vulnerability detection compared to
token-level or AST-only representations [ 39,40]. More recent
systems build unified or heterogeneous graphs that integrate
AST, CFG, and DFG information and report state-of-the-art per-
formance on large Solidity benchmarks [ 41,42]. Heterogeneous
graph transformers tailored to smart contracts (exemplified by
MANDO-HGT) similarly rely on combining call, control-flow,
and semantic relations to capture complex exploit patterns [ 32].
These results motivate ORACAL’s decision to explicitly con-
struct and fuse CFG, DFG, and CG into a single heterogeneous
contract graph.
The first phase involves extracting and constructing a het-
erogeneous graph from Solidity source code. We build on three
standard graph representations of code:
•Control Flow Graph (CFG):Nodes represent basic
blocks of execution; edges represent control flow paths
(jumps, branches). A CFG captures the possible execution
order and is effective for detecting logic errors and
unreachable code, but it does not explicitly model data
dependencies. We extract the CFG using EtherSolve
[43], which analyzes EVM bytecode to produce accurate
control flow structures, handling jumps and basic blocks
efficiently.
•Data Flow Graph (DFG):Nodes represent variables and
operations; edges track data usage, definitions, and influ-
ences. A DFG is crucial for detecting tainted data and
improper computations, notably unsafe arithmetic and
unchecked external calls, but it loses direct execution con-
text. We construct the DFG using the Solidity Compiler
(solc) [ 44] to generate the Abstract Syntax Tree (AST),
followed by a custom SolidityExtractor that traces
data dependencies (variable usage, assignments) and in-
fluences.
•Call Graph (CG):Nodes represent functions; edges rep-
resent function invocations (caller–callee). A CG captures
high-level interaction structure and is vital for reason-
ing about reentrancy and cross-function data flow. We
generate the CG using Slither [ 30], which identifies both
4

Figure 1: Overview of the ORACAL Framework. The pipeline proceeds from heterogeneous graph construction to identifying critical nodes, enriching them via RAG,
and finally training a causal attention-based GNN.
internal and external function calls to map the execution
hierarchy.
Node types:The resulting heterogeneous graph G=(V,E)
integrates three subgraph types:
•CFG Nodes:Represent basic blocks of EVM opcodes.
Features include the list of opcodes and basic block meta-
data.
•DFG Nodes:Represent variables, expressions, and opera-
tions. Features include variable names, types, and values.
•CG Nodes:Represent functions. Features include func-
tion names, visibility, and modifiers.
Edge types:We establishcross-graph edgesto capture
multidimensional relationships based on opcode semantics:
•CFG→CG:Linked via function call opcodes including
CALLandDELEGATECALL.
•CFG↔DFG:Linked via storage/memory operations.
SSTORE /MSTORE link CFG blocks to DFG variables
(Write), while SLOAD /MLOAD link DFG variables to CFG
blocks (Read).
•CG→CFG:Connects function entry points (CG) to their
corresponding usage in control flow (CFG).Node Initialization:We utilize GraphCodeBERT to gener-
ate initial embeddings for all nodes. GraphCodeBERT is chosen
for its pre-trained understanding of code structure and data flow,
providing a rich starting representation (768 dimensions) before
any GNN processing.
3.2. Critical Node Selection (Top-k Extraction)
Running RAG over every node in a large heterogeneous
contract graph is prohibitively expensive: the cost of LLM in-
ference grows roughly linearly with the number of input and
output tokens, and frontier models can differ in price by two
orders of magnitude [ 45,46]. Analyses of RAG pipelines further
highlight that hardware, vector search, and repeated LLM calls
together make end-to-end RAG noticeably more expensive than
standard neural inference, especially at scale [ 47,48]. To opti-
mize computational cost and focus the expensive RAG process
on the most relevant parts of the code, we therefore select the
top-kcriticalnodes. For example, when k=50, only the 50
highest-scoring nodes are retained for subsequent RAG-based
semantic enrichment.
This selection uses a hybrid importance score that combines
graph-theoretic metrics with learned attention weights. Prior
work on graph pooling and vital-node identification has shown
that ranking nodes by a combination of structural centrality statis-
tics, such as PageRank and k-core, and attention-based scores
leads to more expressive and efficient graph-level representations
than using either family of signals alone [ 49,50,51,52]. Our
5

design follows this line of research by integrating both learned
and hand-crafted measures into a single importance score.
Importance Metrics:
1.GAT Attention:Derived from a pre-trained GAT model
[53], highlighting nodes that receive high attention
weights during preliminary classification.
2.K-Core Decomposition:Identifies the core nodes of the
graph based on their degree-based subgraph membership
[54].
3.PageRank:Measures node influence and prestige based
on recursive connectivity within the graph [55].
4.Community & Centrality:Includes Louvain community
detection [ 56] for identifying functional modularity and
Betweenness Centrality [ 57] for measuring the extent to
which a node lies on the shortest paths between others.
These metrics are normalized and combined into a final
importance score:
Score(v i)=X
βj·norm(metric j(vi)) (1)
whereβweights sum to 1.
Connected Top- kSelection Algorithm.To ensure that
the selected nodes form a connected subgraph, we employ the
greedy selection strategy described in Algorithm 1. Given a
directed graph G=(V,E) with node-level importance scores,
the algorithm incrementally constructs a connected node set S
of size at mostk.
The algorithm consists of two stages. First, all nodes are
sorted in descending order of their importance scores and in-
serted into a priority queue Q, which determines the selection
order. Second, nodes are examined sequentially from Q. The
node with the highest importance score is selected to initialize
the set S. For each subsequent candidate node v, the algorithm
adds vtoSonly if vis reachable from at least one node already
inSin the graph G. This reachability condition is verified using
Breadth-First Search (BFS) restricted to the directed edges of G.
The selection process terminates when either knodes have
been added to Sor when no remaining nodes in Qsatisfy the con-
nectivity constraint. In the case of a disconnected graph, the al-
gorithm returns the largest connected subset of high-importance
nodes with cardinality |S|≤k . This design guarantees that the
resulting subgraph preserves structural connectivity while priori-
tizing semantically critical nodes, which is essential for effective
context aggregation in the subsequent RAG-based semantic en-
richment stage.
3.3. RAG-based Semantic Enrichment
Building upon the previously constructed heterogeneous
graphs, this phase further augments the representation with
external security knowledge through a Retrieval-Augmented
Generation (RAG) pipeline.
Traditional heterogeneous contract graphs mainly capture
structural and syntactic relations, encompassing control-flow,
data-flow, and call edges, which limits the amount of explicit
semantic knowledge available to the model. We extend this rep-
resentation byenrichingthe graph with security-domain knowl-
edge retrieved via RAG. As shown in Figure 2, panel (a) containsAlgorithm 1Connected Top-kNode Selection
Require: Directed graph G=(V,E), target number of nodes k,
importance scores Score(v) for allv∈V
Ensure:A connected node setSwith|S|≤k
1:Sort all nodes in descending order of Score(v)
2:Q←priority queue containing all nodes sorted by score
3:S←∅
4:while|S|<kandQ,∅do
5:v←dequeue the highest-scoring node fromQ
6:ifS=∅then
7:S←S∪{v}
8:else
9: ifvis reachable from at least one node in Svia BFS
onGthen
10:S←S∪{v}
11:end if
12:end if
13:end while
14:returnS
Figure 2: Comparison between (a) a standard heterogeneous graph and (b) our
enriched heterogeneous graph.
only raw code-derived information, whereas panel (b) illustrates
our enriched graph, in which nodes are augmented with seman-
tic explanations and additional semantic edge relations. This
produces a more informative input for downstream GNNs, en-
abling reasoning not only over structure but also over high-level
security intent and risk patterns.
3.3.1. RAG Architecture
Large Language Models (LLMs) can be effective at sum-
marizing code intent, but they may lack specialized, up-to-date
domain knowledge and can hallucinate plausible but incorrect
details. Retrieval-Augmented Generation (RAG) addresses this
by grounding generation in an external corpus: the model first
retrieves the most relevant passages and then conditions the final
output on those retrieved documents [ 58]. In our setting, this
corresponds to (i) embedding the query (node/edge context from
the selected subgraph), (ii) retrieving relevant Solidity/EVM/au-
6

dit materials, and (iii) generating structured, evidence-grounded
enrichments.
We leverage LangChain for orchestration, Milvus as the vec-
tor store, and Google Gemini 3 Flash (gemini-3-flash-preview)
[59] as the LLM. Gemini 3 Flash is a cost-efficient, low-latency
model designed for high-throughput generation while maintain-
ing quality comparable to Google Gemini 2.5 Pro in structured
reasoning and code-related tasks. Compared to many open-
source LLMs, which often require substantially more computa-
tional resources and inference time to achieve similar enrichment
quality, Gemini 3 Flash provides a favorable balance between
speed and output fidelity. In addition, Gemini 3 Flash supports
up to 65,536 output tokens per request, which is sufficient to
enrich 50–100 nodes and their edges in a single call. This large
context window enables batch-level enrichment without frag-
mentation, thereby reducing API overhead, minimizing latency,
and keeping operational costs manageable at scale.
•Vector Store:A domain-specific corpus including Solid-
ity documentation, official EVM specifications, vulner-
ability taxonomies, and real-world audit reports is con-
structed and segmented (size=1000, overlap=200). The
corpus is embedded using BAAI/bge-large-en-v1.5
[60], a high-performance dense retrieval model validated
on benchmarks such as MTEB [ 61]. We adopt BGE due
to its instruction-aware contrastive training, which im-
proves alignment between technical queries and structured
documentation. This is particularly important for smart
contract vulnerability analysis, where subtle semantic dif-
ferences in opcode behavior and execution flow can lead
to distinct security outcomes. The Solidity and EVM
specifications provide authoritative protocol semantics
[44,62], while the SWC Registry [ 63] and public audit
datasets such as SC-Bench [ 64] contribute standardized
vulnerability patterns and real exploit evidence . Together,
these sources ensure that retrieved context is both formally
grounded and empirically informed, thereby enhancing
reasoning reliability and reducing hallucination.
•Retrieval and Re-ranking:For each vulnerability-
relevant subgraph, we first retrieve the top-10 semantically
similar document chunks using dense vector similarity
search. Although dense retrieval effectively captures
global semantic relevance, it may not sufficiently
prioritize fine-grained security-critical cues. To improve
precision, we further apply bge-reranker-v2-m3
[65,66], which adopts a cross-encoder architecture to
jointly encode the query and candidate passage, enabling
token-level interaction and more accurate relevance
estimation. The top-3 re-ranked passages are then
selected to construct the final prompt context. This
two-stage retrieval strategy enhances evidence fidelity
and ensures that the language model is conditioned on
highly relevant and security-grounded information.
3.3.2. Prompt Engineering
To guide the LLM effectively, we construct a structured
prompt with three distinct sections, as illustrated in Figures 3, 4,and 5.
Figure 3 illustrates theCONTEXTpart of the prompt, where
relevant technical documents retrieved from the knowledge cor-
pus are provided. This section supplies the LLM with the neces-
sary definitions, specifications, and known vulnerability patterns
related to the specific opcodes or functions present in the sub-
graph, acting as an external knowledge base.
RAG Prompt Part 1: Context Integration
System Instruction:Based on the provided context, answer
the question clearly and concisely.
CONTEXT:[...Dynamically Retrieved Technical Docs,
EVM Specifications, Known Vulnerability Patterns...]
QUESTION:{question}
Figure 3: RAG Prompt Structure: Injecting Knowledge Context.
Figure 4 depicts the first part of theQUESTIONsection. It
begins with a detailed textual description of the subgraph struc-
ture, listing the critical nodes, their connections, and attributes.
Following this, it instructs the LLM to perform a ”Step 1: Com-
prehensive Subgraph Analysis” using a Chain-of-Thought ap-
proach. The model is asked to identify execution paths, vulnera-
bility hotspots (like storage writes or external calls), and security
patterns before generating individual node enrichments.
RAG Prompt Part 2: Subgraph Description
Role Definition:You are an expert in smart contract secu-
rity analysis and graph-based vulnerability detection.
Task:Analyze ONE important subgraph (top-K nodes) rep-
resenting critical execution paths relevant for vulnerability
detection.
Input Data (Contextualized Subgraph):
•Structure:{num nodes}critical nodes,{num edges}
edges.
•Important Nodes:List of nodes with ID, type, im-
portance score, and rank.
•Important Edges:Source→Target relationships.
•Connectivity:Explicit BFS reachability map show-
ing data flow.
Figure 4: RAG Prompt Structure: Subgraph Representation.
Figure 5 presents the second part of theQUESTIONsection,
specifically the instructions for ”Step 2: Enrich Nodes” and
”Step 3: Enrich Edges”. It strictly defines the JSON output
format and details the specific fields required for each node and
edge, ensuring the output is structured and machine-parsable for
integration back into the graph pipeline.
7

RAG Prompt Part 3: Analysis & Output Schema
Chain-of-Thought Instructions:
1.Holistic Analysis:Identify critical execution paths,
vulnerability hotspots (storage writes, external calls),
and flow patterns.
2.Enrich Nodes:Generate semantic meaning ,
operational context , and security analysis
(Max 20 words).
3.Enrich Edges:Generateedge relationship.
Required Output Format (JSON):
{
"enriched_nodes": {
"node_id": {
"semantic_meaning": "...",
"operational_context": "...",
"security_analysis": "..."
}
},
"enriched_edges": [
{
"edge_id": "...",
"source": "...",
"target": "...",
"edge_relationship": "..."
}
]
}
Figure 5: RAG Prompt Structure: Reasoning Steps and JSON Schema.
3.3.3. Enriched Output Structure
Figures 6a and 6b present examples of the structured JSON
output generated by the RAG system for nodes and edges, re-
spectively.
The output JSON contains specific fields to ensure traceabil-
ity and correct mapping back to the heterogeneous graph:
•Identity Fields:
–node id(represented by “cfg 1227”) and edge id
(represented by “cfg 1227 cfg1236 control flow”)
uniquely identify the elements in the original graph.
–source andtarget explicitly define the edge direc-
tionality, essential for maintaining the causal flow
during the enrichment integration.
•Semantic Fields:We specifically designed the output to
contain four distinct fields to balance semantic depth with
token efficiency:
–Semantic Meaning:Captures the ”What”, rep-
resenting the high-level intent of the code block
(namely, ”Token Transfer Logic”). This allows the
GNN to understand the functional purpose beyond
raw opcodes.–Operational Context:Captures the ”How”, refer-
ring to the node’s role in the execution state (specifi-
cally, ”Updates balance storage slot”). This provides
state-transition awareness.
–Security Analysis:Captures the ”Risk”, detailing
specific vulnerability indicators (particularly,
”Unchecked external call”). This directly injects
domain expert knowledge into the feature space.
–Edge Relationship:Captures the ”Structure”, es-
tablishing the logical connection between nodes (de-
fined as ”Control flow dependency”). This enriches
the graph topology with semantic reasoning.
These textual descriptions are encoded via GraphCodeBERT
and concatenated with the original embeddings to form the en-
riched feature set Xenriched∈R1536(768 original+768 enriched).
3.4. Causal Attention Learning
Neural models often overfit to dataset artifacts (spurious
correlations) that are predictive during training but not causally
related to vulnerabilities, including variable naming conventions
or functional style patterns. To address this issue, causal atten-
tion seeks to disentangle causally relevant features from spuri-
ous ones by introducing an intervention-style learning objective
and an attention mechanism that emphasizes true causal signals
[67,68,69], thereby improving robustness under distribution
shift and adversarially induced superficial changes.
Accordingly, to ensure trustworthy predictions, the model
must distinguish between true causal features (valid security
logic) and spurious correlations (biases). We implement the
CausalAttentionHeteroClassifier, as detailed in Figure 7.
3.4.1. Detailed Layer Architecture
The framework operates through a sequence of specialized
layers, each serving a distinct purpose in the disentanglement
process:
1. Dual Feature Encoders (Input Processing).Two par-
allel feed-forward networks (Linear →LayerNorm→ReLU
→Dropout) map the input feature space (768 dimensions) to
a hidden dimension (128). TheCausal Encoderexclusively
processes Xenriched (RAG-derived features containing ”Security
Analysis”), while theSpurious Encoderprocesses Xoriginal (raw
GraphCodeBERT embeddings containing syntax/tokens). This
physical separation allows the model to learn distinct representa-
tions for semantic logic versus structural patterns.
2. Node Attention Layer (Feature Selection).A Softmax-
based attention mechanism computes two scalar weights αcand
αsfor each node. This layer dynamically weighs the importance
of each branch. For example, if a node’s original embedding is
noisy (say, standard boilerplate code), the model learns to assign
a lowerαsand higherαc, effectively filtering out the noise before
it propagates.
3. Feature Fusion Layer (Combination).The weighted
features are fused via hcombined =Linear (Concat (αc·henrich,αs·
horig)). This creates a unified representation that prioritizes
8

Example of RAG-enriched Node Output.
"cfg_1227": {
"node_id": "cfg_1227",
"semantic_meaning": "Control flow node
indicating decision branch.",
"operational_context": "Reached from cfg_1207,
leads to cfg_1236 seq-flow.",
"security_analysis": "Potential reentrancy
point (state update post-loop).",
"enrichment_source": "rag"
}
(a) RAG-enriched Node OutputExample of RAG-enriched Edge Output.
{
"edge_id": "cfg_1227__cfg_1236__control_flow",
"source": "cfg_1227",
"target": "cfg_1236",
"edge_relationship": "Control flow representing
sequential execution.",
"relation": "control_flow",
"enrichment_source": "rag"
}
(b) RAG-enriched Edge Output
Figure 6: Examples of RAG-enriched outputs: (a) node-level enrichment and (b) edge-level enrichment.
the ”cleaned” causal signal while retaining necessary structural
context for graph propagation.
4. Heterogeneous GNN Layers (Propagation).Stacked
HeteroConv+GATConv layers (256 dimensions) propagate the
refined causal information across the graph while preserving
edge types. Since the input to the GNN is already “cleaned” by
the attention layer, the message passing spreads valid vulnerabil-
ity context, exemplified by “tainted flow”, rather than irrelevant
syntax patterns.
5. Graph Pooling (Aggregation).Mean pooling aggregates
node representations into a graph-level vector to verify the vul-
nerability status of the entire contract based on its most critical
components.
6. Dual Classifiers (Prediction).Two separate classifica-
tion heads yield logits zc(Causal Prediction) and zs(Spurious
Prediction). TheMain Predictionfor final auditing is taken ex-
clusively from zc, ensuring decisions are based on the enriched,
causal path.
7. Backdoor Adjustment Head (Do-Calculus).To ensure
predictions remain stable under different spurious contexts, we
simulate do-calculus by intervening on spurious features. We
pool node-level causal and spurious representations to graph-
level vectors HcandHs, then form H′
sby shuffling Hswithin the
batch (breaking the spurious–label link). An auxiliary classifier
MLP upredicts from the concatenation of causal and intervened
spurious features:
zu=MLP u(Hc∥H′
s).(2)
Training zuto match the true label y(viaLcausal ) encourages the
model to rely on stable causal features rather than spurious ones.
3.4.2. Multi-Task Loss Function
The following loss terms support causal attention learning by
disentangling causal (enriched) from spurious (original) features
and stabilizing predictions under intervention. Sui et al. [ 68]
decompose the graph into causal and trivial attended-graphs
via learned masks and train the causal branch to predict the
label while pushing the trivial branch to uniformity and apply-
ing backdoor adjustment at the representation level. We adoptthe same spirit for vulnerability detection with dual encoders
and node attention, and add a contrastive loss to keep the two
representations distinct. We minimize a joint loss function:
Ltotal=L sup+λ 1Luni f+λ 2Lcausal +λ 3Lcontra (3)
whereλ1,λ2, andλ3are hyperparameters that determine the
relative importance of each objective.
•L sup(Supervised Loss):The causal branch prediction zc
is trained to match the true label yvia binary cross-entropy
(implemented as BCEWithLogitsLoss ). For multi-label
targets, over the batch and labels:
Lsup=−EDh
y⊤logσ(z c)+(1−y)⊤log 1−σ(z c)i
(4)
whereσdenotes the sigmoid function. This ensures the
model learns accurate vulnerability classification from the
enriched, causal path.
•L uni f(Uniform Distribution Loss):To discourage re-
liance on superficial patterns (spurious features), we push
the spurious-branch prediction zstowards maximum un-
certainty [ 70]. For multi-label detection the target is 0.5
per label. We minimize the mean squared error between
σ(z s) and the uniform target:
Luni f=EDhσ(z s)−0.52
2i
(5)
By penalizing confident predictions in the spurious branch,
the model shifts its decision-making to the causal branch.
•L causal (Backdoor Adjustment/Do-Calculus):We sim-
ulate intervention on spurious features by forming graph-
level Hc,Hs, and H′
s(shuffled Hswithin the batch). The
auxiliary prediction zu=MLP u(Hc∥H′
s) (Step 7) is trained
to match y. Because H′
sis randomized, the model must
rely onH cto predict correctly [68]:
Lcausal =−EDh
y⊤logσ(z u)+(1−y)⊤log 1−σ(z u)i
(6)
9

Figure 7: Detailed Architecture of the Causal Attention HeteroClassifier. The
input is split into Causal ( henrich ) and Spurious ( horig) branches, processed by
dual encoders, fused via attention, and refined by GNN layers before final
classification.
•L contra (Contrastive Loss):This loss enforces orthog-
onality between the learned representations of the two
branches [ 71]. We maximize the Euclidean distance be-
tween the enriched representation henrich and the original
representationh orig, up to a marginm:
Lcontra =max(0,m−∥h enrich−h orig∥2) (7)
where m=0.5. This prevents ”feature leakage” where the
enriched encoder might lazily copy the original features.
It guarantees that henrich captures new, distinct semantic
information provided by the RAG process, rather than
redundant structural data.
This multi-task objective ensures robust generalization by an-
choring predictions in causal semantics while actively unlearning
reliance on spurious artifacts.
4. Experiments and Evaluation
4.1. Research Questions
To evaluate ORACAL comprehensively, we address three
Research Questions (RQs):
•RQ1:How do enrichment strategies and training
paradigms (Standard vs. Causal) affect ORACAL’s
detection performance and its generalization across
in-domain and out-of-distribution datasets?•RQ2:How effectively does ORACAL generate explana-
tions for its vulnerability predictions ?
•RQ3:How does ORACAL compare with state-of-the-art
GNN methods (Mando-HGT, SCVHunter, MTVHunter)
in terms of detection accuracy and robustness under ad-
versarial attacks?
4.2. Experimental Setup
This experiment is conducted in a Google Colab environment
configured with 1×NVIDIA A100 80 GB Tensor Core GPU and
an Intel Xeon CPU @ 2.20 GHz (6 physical cores, 12 threads),
with approximately 230 GB of system RAM.
4.2.1. Datasets
We utilize four datasets to evaluate our model. SoliAudit
and CGT Weakness serve as training and in-domain evaluation
sources, while DAppScan and LLMA V are reserved entirely for
out-of-distribution (OOD) evaluation and explainability assess-
ment.
•SoliAudit [ 23].SoliAudit is a multi-label smart contract
vulnerability dataset introduced alongside a hybrid se-
curity analysis framework combining machine learning
and fuzz testing. After rigorous preprocessing (filtering
compilation errors, graph extraction failures, and unsup-
ported versions), the dataset used in this study contains
10,655 valid samples and serves as the primary source for
learning fundamental vulnerability patterns.
•CGT Weakness [ 24].CGT (Consolidated Ground Truth)
is a dataset of 3,103 smart contracts, built by aggregat-
ing several previously published benchmarks including
CodeSmells, ContractFuzzer, SolidiFI, Zeus, etc. In this
dataset, smart contracts were manually labeled by the au-
thors of each original dataset; labels were then unified
through voting algorithms to select a consistent label for
each contract. After preprocessing and conversion to CFG,
DFG, and CG graph representations, 1,345 contracts were
successfully retained.
•DAppScan [25].This is a real-world dataset constructed
from 1,199 professional security audit reports, cov-
ering 682 decentralized applications (DApps). The
DAPPSCAN-SOURCE subset includes 947 labeled
Solidity source files. Due to complex dependency
structures common in large DApp projects, only 124
contracts were successfully compiled, processed, and
converted into graphs.
•LLMA V [ 26].The Line-Level Manually Annotated Vul-
nerabilities (LLMA V) dataset originates from the empir-
ical study by Salzano et al., which provides the largest
collection of Solidity smart contracts with manually ver-
ified, line-level vulnerability annotations. The original
dataset contains 2,081 annotated contracts covering seven
DASP vulnerability categories. Unlike SoliAudit and
CGT Weakness, which provide only contract-level labels,
10

LLMA V offers fine-grained statement-level ground truth,
making it uniquely suited for evaluating both detection
generalization and the quality of model-generated explana-
tions against manually annotated vulnerability triggering
paths.
Both DAppScan and LLMA V are excluded from training
entirely and serve a dual purpose in our evaluation: they function
as out-of-distribution benchmarks for assessing generalization
to unseen contract populations, and they provide line-level or
audit-level annotations that enable rigorous evaluation of the
explainability component.
Table 2 presents the detailed distribution of vulnerabilities
across all four datasets. In our multi-label classification context,
“Positive”refers to the count of contracts explicitly identified as
containing a specific vulnerability type, while“Negative”refers
to those free from that specific issue. It is important to note that
since a single smart contract can exhibit multiple vulnerabilities
simultaneously (considering a contract with both Reentrancy
and Arithmetic bugs), the sum of “Positive” samples across
categories may exceed the total number of unique contracts.
This distinction is crucial for understanding the class imbalance
challenges our model must address. Notably, LLMA V exhibits a
distinct distribution from the training sources: Arithmetic (472
positive) and Time Manipulation (415 positive) are the most
prevalent categories, while Front Running is extremely rare (7
positive), reflecting a distribution that differs substantially from
SoliAudit and thus provides a rigorous test of generalization.
4.2.2. Training and Testing Scenarios
To ensure consistency within our multimodal framework,
we apply the proposed RAG-based semantic enrichment to all
data prior to model training and evaluation. This guarantees
that every split is represented in a uniformly enriched semantic
space, enabling the model to capture contextual dependencies
across multiple vulnerability categories.
For experimental setup, we employ an 80:20 multilabel strat-
ified split for SoliAudit and CGTWeakness to preserve the joint
distribution of vulnerability labels. Specifically, 80% of each
dataset is merged to construct a unified training set, while the
remaining 20% portions are retained as in-domain hold-out test
sets. In contrast, both DAppScan and LLMA V are excluded from
training entirely and reserved for evaluation only. DAppScan
serves as a real-world out-of-distribution (OOD) benchmark
representing production-grade DeFi contracts, while LLMA V
provides an OOD benchmark with line-level annotations that
additionally enables explainability evaluation. To ensure strict
data separation and prevent data leakage, we perform contract
address deduplication between LLMA V and the training sources.
Of the original 2,081 contracts in the LLMA V dataset, 1,052
share overlapping addresses with contracts already present in
SoliAudit or CGTWeakness; these are removed, yielding 1,029
unique contracts for OOD evaluation. This configuration allows
us to assess both in-domain detection performance and cross-
dataset generalization under realistic deployment conditions,
while simultaneously validating the quality of model-generated
explanations.We train the model on four vulnerability classes, namely
Arithmetic, Denial of Service, Low Level Calls, and Time Ma-
nipulation, due to data imbalance considerations. This selection
is guided by the label distribution analysis in Table 2, where
these categories exhibit the most balanced positive to negative
ratios across all datasets. As a result, they provide sufficient
positive samples in both the training sources and the out of dis-
tribution test sets to support statistically meaningful evaluation.
In contrast, other vulnerability types including Reentrancy with
503 positive samples out of 10,655 in SoliAudit but only 8 in
DAppScan and 49 in LLMA V , Access Control with 864 positive
samples in SoliAudit and 5 in DAppScan, and Front Running
with only 7 positive samples in LLMA V , suffer from extreme
class imbalance. This imbalance would make per class F1 scores
unreliable due to high variance caused by limited sample sizes.
Therefore, focusing on these four classes during training ensures
that the reported results reflect actual model capability rather
than artifacts of sampling noise.
The detailed label-wise distribution across the resulting train-
ing and testing splits is summarized in Table 3.
4.2.3. Statistical Analysis
To compare the highest-performing configurations under the
standard and causal attention training paradigms, we evaluate
the macro F1 scores obtained from five independent runs with
different random seeds. This repeated-run setting accounts for
stochastic variability during model training.
Since performance metrics in deep learning are often non-
normally distributed and we conduct five different runs for each
training paradigm, we adopt non-parametric statistical methods
[72]. Specifically, we apply the Wilcoxon signed-rank test [ 73]
to assess whether significant differences exist between the paired
macro F1 scores of the two models. The test is conducted
using an exact two-sided p-value under the null hypothesis of
no difference. We consider p<0.05 statistically significant and
p<0.10 marginally significant.
In addition to significance testing, we report the Vargha–
Delaney ˆA12effect size [ 74] to quantify the magnitude of per-
formance differences. This statistic represents the probability
that a randomly selected macro F1 score from the causal model
exceeds one from the standard model. A value of ˆA12=0.5
indicates no difference, while ˆA12≥0.71 corresponds to a large
effect. Reporting both p-values and effect sizes provides a more
reliable interpretation, particularly with small sample sizes.
4.2.4. Performance Metrics
We formulate vulnerability detection asmulti-label classifi-
cation, where a single smart contract can simultaneously exhibit
multiple vulnerability types, encompassing both Reentrancy and
Timestamp Dependence, represented as a binary vector over C
classes [ 75]. This reflects the reality of smart contract security,
where vulnerabilities are not mutually exclusive.
We evaluate ORACAL across three dimensions: detection
performance (Accuracy, Precision, Recall, Macro F1), robust-
ness under adversarial attack, and explainability quality. For
robustness, we employ the evasionAttack Success Rate (ASR),
11

Table 2: Distribution of Vulnerabilities Across Datasets
Vulnerability TypeSoliAudit CGT Weakness DAppScan LLMA V
Pos. Neg. Pos. Neg. Pos. Neg. Pos. Neg.
Reentrancy 503 10,152 775 570 8 116 49 980
Access Control 864 9,791 778 567 5 119 63 966
Arithmetic 9,849 806 530 815 20 104 472 557
Low-Level Calls 3,109 7,546 801 544 3 121 68 961
Denial of Service 4,657 5,998 776 569 10 114 119 910
Front Running 1,609 9,046 595 750 5 119 7 1,022
Time Manipulation 3,420 7,235 845 500 7 117 415 614
Total Samples 10,655 1,345 124 1,029
Table 3: Label-wise Distribution of Smart Contract Vulnerabilities in Training
(SoliAudit+CGTWeakness) and Testing Sets. DAppScan and LLMA V are used
exclusively as OOD test datasets and explainability benchmarks.
Dataset Vulnerability Train Test Total
SoliAuditReentrancy 402 101 503
Access Control 691 173 864
Arithmetic 7,879 1,970 9,849
Low Level Calls 2,487 622 3,109
Denial of Services 3,726 931 4,657
Front Running 1,287 322 1,609
Time Manipulation 2,736 684 3,420
CGTWeaknessReentrancy 620 155 775
Access Control 622 156 778
Arithmetic 424 106 530
Low Level Calls 641 160 801
Denial of Services 621 155 776
Front Running 476 119 595
Time Manipulation 676 169 845
DAppScanReentrancy 0 8 8
Access Control 0 5 5
Arithmetic 0 20 20
Low Level Calls 0 3 3
Denial of Services 0 10 10
Front Running 0 5 5
Time Manipulation 0 7 7
LLMA VReentrancy 0 49 49
Access Control 0 63 63
Arithmetic 0 472 472
Low Level Calls 0 68 68
Denial of Services 0 119 119
Front Running 0 7 7
Time Manipulation 0 415 415
defined as the fraction of correctly predicted vulnerable sam-
ples misclassified as safe (1 →0 label flip) after perturbation.
For explainability, we adopt the Vulnerability Triggering Path
(VTP) framework from Cao et al. [ 36], using Mean Statement
Precision (MSP), Mean Statement Recall (MSR), and Mean In-
tersection over Union (MIoU) to quantify alignment between
model-generated explanations and human-annotated vulnerabil-
ity paths. All metric definitions are summarized in Table 4.Table 4: Summary of performance metrics used for evaluation.
Category Metric Description
DetectionAccuracy Jaccard score (intersection over union of
predicted and ground-truth label sets per
sample).
Precision The ratio of correctly predicted positive ob-
servations to the total predicted positive ob-
servations.
Recall The ratio of correctly predicted positive ob-
servations to all actual observations in the
class.
F1 Score The harmonic mean of Precision and Re-
call, providing a balanced measure of per-
formance.
RobustnessASR Attack Success Rate: The percentage of
adversarial examples that successfully fool
the model by flipping a vulnerable label to
safe (1→0).
ExplainabilityMSP Mean Statement Precision: The propor-
tion of explanatory statements correctly re-
lated to the identified vulnerability.
MSR Mean Statement Recall: The ability of the
explainer to cover all contextual statements
within the vulnerability-triggering path.
MIoU Mean Intersection over Union: The over-
all overlap between explanatory and ground-
truth statements.
4.2.5. Hyperparameters
Table 5 summarizes the hyperparameters for both model
training and the Retrieval-Augmented Generation (RAG) seman-
tic enrichment process. For training, we utilized the AdamW
optimizer with a learning rate of 2 ×10−4and a batch size of
64. The model architecture consists of 3 GNN layers with a
hidden dimension of 256, chosen to maximize representation
capacity while minimizing the risk of over-smoothing. The
causal loss methodology employs specific weighting factors
(λuni f=0.1,λcausal =0.5,λcontra =0.2) to balance standard
classification with causal feature disentanglement.
Regarding the semantic enrichment via RAG, we partitioned
contract source code into chunks of 1000 characters with an over-
lap of 200 to preserve contextual continuity. During retrieval,
we set k=10 to extract the most relevant snippets from the ex-
ternal security knowledge base, which were subsequently used
to enrich the top 50 high-importance nodes identified through
code analysis.
12

Table 5: Hyperparameters for Model Training and RAG Enrichment
Parameter Type Parameter Value
Training ParametersEpochs 50
Learning Rate 2e-4
Batch Size 64
Optimizer AdamW
Weight Decay 0.0001
Dropout Rate 0.2
Hidden Dimension 256
GNN Layers 3
Enriched Attn Weight 2.0
Loss Parametersλuni f 0.1
λcausal 0.5
λcontra 0.2
Margin 0.5
RAG ParametersChunk Size 1000
Chunk Overlap 200
Retrievalk10
Top-kEnrichment Nodes 50
Figure 8: Average code-level metrics per contract across four datasets. Bars rep-
resent: LOC (teal), Invocation (blue), StateVars (coral), CFComplexity (orange),
and ExtCalls (dark gray).
4.3. Exploratory Data Analysis (EDA)
To further characterize the graph corpus, we report average
code and graph metrics across all four data sources (Figure 8)
described in Section 4.2.1.
Figure 8 compares five metrics across the four datasets and il-
lustrates differences in both size and structural complexity.LOC
(Lines of Code)approximates the overall implementation size of
a contract.Invocationis the size of the invocation space, i.e., the
number of nodes in the call graph, corresponding to the number
of functions (or callable entry points) in the contract.State-
Vars (State Variables)is a proxy for state-space size: it counts
DFG nodes of type variable declaration , including both
contract-level state variables and local variables.CFComplexity
(Control Flow Complexity)is McCabe cyclomatic complex-
ity [76] of the control-flow graph, M=E−N +2 where Eand
Nare the number of edges and nodes in the CFG, capturing the
number of linearly independent execution paths and thus the
Figure 9: Average graph-scale statistics per contract across four datasets. Bars
represent: Node count (teal), Edge count (blue), Node enrichment length in
characters (coral), and Edge enrichment length in characters (tan).
branching complexity of the contract.ExtCallsis the number
of inter-function calls, i.e., edges in the call graph (caller–callee
relations), which reflects how intensively a contract interacts
with other functions and components.
As shown in Figure 8, DAppScan contains the largest and
most complex contracts (LOC 327.2, StateVars 99.7, ExtCalls
14.6), representative of production-grade DeFi code with dense
state spaces and frequent external interactions. SoliAudit fol-
lows with moderately high values (LOC 257.2, ExtCalls 11.1),
reflecting medium-complexity contracts typical of curated secu-
rity benchmarks. LLMA V presents a notable anomaly: despite
lower LOC (225.7) than SoliAudit, it exhibits higher CFCom-
plexity (52.2 vs. 47.7), suggesting denser branching logic per
line of code. CGT is the most compact (LOC 185.7) yet carries
the highest CFComplexity-to-LOC ratio, indicating algorithmi-
cally dense contracts with many independent execution paths.
Together, the four corpora span a spectrum from complex pro-
duction contracts (DAppScan) to compact but structurally dense
benchmarks (CGT), providing a rigorous and diverse evaluation
setting for graph-based vulnerability detection.
Figure 9 summarizes graph-scale statistics across the four
datasets. DAppScan produces the largest graphs (682.4 nodes,
565.3 edges), followed by SoliAudit (590.8 nodes, 481.5 edges),
CGT (544.6 nodes, 493.6 edges), and LLMA V (493.9 nodes,
456.7 edges). The node-enrichment text length is relatively
consistent across datasets, ranging from 201.4 (DAppScan) to
215.1 (CGT) characters, indicating that the RAG pipeline gener-
ates similarly concise semantic descriptions regardless of graph
complexity. Edge enrichment text is shorter and more uniform
(64.6–73.1 characters). These sizes are large enough to encode
meaningful semantic cues, comprising intent, data-flow roles, or
security-relevant hints, while remaining well within the token
limits of GraphCodeBERT.
4.4. Results and Analysis
4.4.1. RQ1: Ablation study on SoliAudit Dataset
Motivation.To understand the contribution of each archi-
tectural and enrichment component in ORACAL, we perform a
systematic ablation study. Each experiment isolates a specific
13

module or enrichment strategy while keeping the remaining
components unchanged. Examining individual components in
this controlled manner allows us to determine how each element
influences detection accuracy, robustness, and generalization
behavior. This analysis clarifies whether performance gains orig-
inate from causal training, retrieval-based enrichment, or their
interaction.
Method.We compare a baseline model (no enrichment)
against seven enrichment variants—Only edge enrichment, All
node+edge enrichment, All node enrichment, Only enrichment
text, Only Operational Context, Only Security Analysis, and
Only Semantic Meaning—under both Standard and Causal train-
ing, reporting detection performance and per-class F1 on the
SoliAudit test set. We further conduct a structural bias analysis
on misclassified samples and measure inference cost (ms/sam-
ple) to assess computational trade-offs. For generalization, we
select the top-performing checkpoint from each paradigm and
evaluate it on CGT, DAppScan, and LLMA V over 5 independent
runs. The detailed evaluation protocol is described alongside the
results below.
Scenario Definitions.Each scenario controls which feature
modalities are fed into the model. In all enriched scenarios, the
original GraphCodeBERT embeddings ( Xorig) are always present
as the graph backbone; the variants differ in whichadditional
enriched features are concatenated. Specifically:No enrichment
(baseline)uses only Xorig;Only enrichment textis the excep-
tion—it replaces Xorigentirely with only the concatenation of
all enriched text fields (no graph features);Only Operational
Context/Security Analysis/Semantic Meaningaugments Xorig
with one specific node-level field;All node enrichmentaugments
Xorigwith all three node-level fields;Only edge enrichmentaug-
ments Xorigwith the edge relationship field only; andAll node+
edge enrichmentaugments Xorigwith all four fields (three node
+one edge).
Ablation Study Results.Table 6 presents the quantitative
comparison of Standard and Causal training across the different
enrichment scenarios.
Table 6 reports the Macro F1 results under different enrich-
ment configurations and training paradigms. Across all com-
parable settings, Causal training consistently achieves higher
performance than the Standard paradigm.
The performance gap between Causal and Standard training
ranges from 2.61% to 4.09% in Macro F1, depending on the
enrichment configuration. The largest improvement is observed
in the “Only edge enrichment” setting, where Causal testing
achieves 91.28% compared to 87.19% under Standard training,
yielding a difference of 4.09 percentage points. Even in the base-
line configuration without enrichment, Causal training improves
performance from 86.94% to 89.55%, corresponding to a gain
of 2.61 percentage points.
Comparing enrichment strategies against their respective
baselines further reveals that edge enrichment provides the most
substantial benefit. For Causal training, performance increases
from 89.55% in the baseline to 91.28% with only edge enrich-
ment. Under Standard training, performance increases from
86.94% to 87.19%. In contrast, other enrichment types yield
smaller gains, indicating that structural semantic information em-bedded in edges plays a more critical role than additional node-
level descriptions. These results suggest that message passing
over semantically enriched edges contributes more effectively to
vulnerability detection than isolated contextual augmentation.
Per-class performance analysis.We further visualize the
performance details through class-wise F1-Score heatmaps for
both paradigms on the SoliAudit test set in Figure 10. The
heatmaps reveal the performance distribution across four key
vulnerability classes: Denial of Service, Time Manipulation,
Low-Level Calls, and Arithmetic. For instance, Causal training
significantly boosts the detection of Time Manipulation (im-
proving from∼0.85 in Standard to ∼0.94 in Causal) and Denial
of Service (improving from ∼0.85 to∼0.89). Both paradigms
perform exceptionally well on the Arithmetic class (0.96-0.97),
but the overall stability and robustness across complex vulnera-
bilities are visibly superior in the Causal setting. Expectedly, the
“Enrichment Only” scenario performs poorly in both, reaffirming
that semantic text alone is insufficient without the underlying
structural graph context.
Structural bias analysis.To better understand the strengths
and limitations of the models across different complexity pro-
files, we analyze the structural properties of misclassified sam-
ples (False Positives and False Negatives) of the causal attention
model. Using four graph-level metrics (CFComplexity, Stat-
eVars, ExtCalls, and Invocation) defined in Section 4.3, we
compute theMean Relative Bias:
FN avg−FP avg
FPavg(8)
where FN avgandFPavgdenote the average structural metric
values of False Negative and False Positive samples, respectively.
This metric captures the structural divergence between missed
vulnerabilities and false alarms, where a positive bias indicates
a tendency to miss vulnerabilities in complex contracts, while a
negative bias reflects over-prediction. The detailed results across
vulnerability types are summarized in Figure 11.
As shown in Figure 11, there is a clear relationship between
the model’s failure cases and the structural characteristics of
specific vulnerabilities. For instance, vulnerabilities heavily re-
liant on external interactions, notablyTime Manipulationand
Unchecked External Calls, generally exhibit a high positive
bias for EXTCALL and INVOCATION. This indicates that de-
tecting these vulnerabilities becomes disproportionately harder
(yielding more False Negatives) when the contract contains ex-
tensive external calls and large invocation spaces. Notably, for
Unchecked External Calls, theenrichment onlyscenario dis-
plays extreme positive bias (EXTCALL+2.03, INVOCATION
+1.44), proving that text-based semantic enrichment completely
fails to capture inter-contract dependencies without a structural
graph backbone. Conversely, forArithmeticvulnerabilities, we
observe a consistent negative bias across most scenarios for
CFCOMPLEXITY (up to -0.71) and STATEV AR (up to -0.73),
meaning the models tend to over-predict arithmetic bugs in con-
tracts with highly branched control flows and numerous state
variables.
Additionally, different enrichment strategies introduce
distinct structural trade-offs. Thebaselinemodel exhibits
14

Table 6: Ablation Study Comparing Standard and Causal Training on the SoliAudit Test Set. Highlighted rows indicate the models with the highest F1score (lighter
red for Standard and darker red for Causal).
Training Type Scenario Accuracy (%) F1-Macro (%) Precision (%) Recall (%)
StandardNo enrichment (baseline) 82.84 86.94 90.36 84.05
Only edge enrichment 82.85 87.19 89.94 84.79
All node+edge enrichment 82.62 86.93 89.03 85.11
All node enrichment 82.02 86.74 88.37 85.29
Only enrichment text 72.63 72.21 79.38 67.92
Only Operational Context 82.90 87.14 90.32 84.35
Only Security Analysis 82.55 86.91 87.59 86.43
Only Semantic Meaning 83.09 87.13 89.14 85.42
CausalNo enrichment (baseline) 85.84 89.55 91.24 88.10
Only edge enrichment 87.40 91.28 92.99 89.79
All node+edge enrichment 86.59 90.70 92.63 89.00
All node enrichment 86.24 90.48 91.56 89.50
Only enrichment text 74.94 74.75 79.54 72.65
Only Operational Context 86.50 90.71 92.75 88.94
Only Security Analysis 87.15 90.96 91.60 90.40
Only Semantic Meaning 86.44 90.64 92.40 89.08
significant positive bias in EXTCALL forTime Manipulation
(+2.08) andArithmetic(+0.39), highlighting its struggle without
semantic context. Node-only enrichments (sec analysis only,
allnode fields) show strong negative bias in CFCOMPLEXITY
and STATEV AR, confirming that dense textual features without
explicit structural edges can lead to over-prediction in intricate
contracts.
Notably, thesemantic meaningenrichment scenario presents
the most balanced structural profile, with relative biases closest
to zero across most metrics. Although theall edge fieldssce-
nario achieves the highest overall macro F1 score (Table 6), it
still retains moderate positive bias towards highly interactive
contracts (specifically,Time ManipulationEXTCALL+1.11).
In conclusion, this analysis demonstrates that misclassification
patterns are fundamentally tied to the structural properties of the
vulnerabilities themselves. While causal attention with semantic
edge enrichment significantly mitigates these weaknesses, stabi-
lizing detection for vulnerabilities that are heavily dependent on
external interactions within complex execution spaces remains a
persistent challenge for current graph-based models.
Loss analysis.In Figure 12, we illustrate the training and
testing loss trends across epochs to observe convergence differ-
ences between the Causal and Standard models. For the Causal
models, although they optimize a more complex multi-objective
loss function, the test loss curves show that the baseline scenario
plateaus at a higher loss value (around 6 ×10−1), whereas struc-
turally enriched scenarios, notably All Edge Fields and All Node
Fields, converge to lower test losses (around 5 ×10−1). This
indicates that semantic enrichment actively assists the causal
objective in discovering better, generalizable semantic-structural
representations. Conversely, in the Standard training, the base-
line model without enrichment achieves the lowest absolute test
loss, yet its actual classification F1-score on the test set is lower
than the enriched variants. This discrepancy highlights that stan-dard cross-entropy training on purely structural features tends
to quickly fit trivial subgraph patterns but struggles to general-
ize. In contrast, the Causal paradigm coupled with RAG-based
enrichment forces the network to learn robust causal features,
thereby preventing overfitting and enhancing overall detection
accuracy.
Inference Time Comparison.To assess the computational
trade-offof our enhancement strategies, we evaluate the infer-
ence time per sample (ms/sample) for each scenario, comparing
Causal and Standard models as illustrated in Figure 13. The
results highlight that while enrichment adds some overhead,
the models maintain highly efficient inference speeds overall.
Specifically, baseline scenarios without enrichment exhibit the
lowest processing times (1.40 ms for Causal and 1.33 ms for
Standard). Most node-only text enrichments, such as Opera-
tional Context (Causal: 1.40 ms, Standard: 1.33 ms) and Secu-
rity Analysis (Causal: 1.39 ms, Standard: 1.32 ms), incur almost
no additional delay compared to the baseline.
However, incorporating edge-based enrichment noticeably
increases inference latency. The highest times are recorded
for the “Only Edge Enrichment” (Causal: 1.90 ms, Standard:
1.80 ms) and “All Node+Edge Enrichment” (Causal: 1.90 ms,
Standard: 1.85 ms) variants. This indicates that processing the
structural edge attributes and their corresponding semantic de-
scriptions within the graph message-passing layers is the most
computationally demanding step. Despite this increase, the max-
imum inference time remains under 2 milliseconds per sample,
confirming that the trade-offis minor and the model remains
viable for rapid large-scale deployments.
Generalization Results.To evaluate how well our proposed
approaches adapt to different data environments, we select four
representative checkpoints and test them on the in-domain CGT
benchmark and the two out-of-distribution datasets, DAppScan
and LLMA V . Specifically, we evaluate two enrichment scenarios
15

Figure 10: F1-Score heatmaps comparing Causal and Standard models on the SoliAudit test set.
Figure 11: Summary of structural bias across scenarios.
under both training paradigms: (i) “ORACAL-edge”, corre-
sponding to theall edge fieldscenario that achieves the highest
F1 score on the SoliAudit test set (Table 6), and (ii) “ORACAL-
node-edge”, corresponding to theall node+edge fieldscenario
that represents our full proposed feature set combining both
node-level and edge-level enrichments. For each scenario, both
Causal and Standard checkpoints are evaluated, yielding four
configurations in total. We restrict this evaluation to these repre-
sentative checkpoints rather than all enrichment scenarios due
to the substantial computational cost of multi-run evaluation:
each 5-run evaluation on three datasets requires approximately
15 full inference passes with associated GPU time for graph con-
struction, embedding computation, and model forward passes.
Running all 8 enrichment scenarios under this protocol would
increase the total computation eightfold, exceeding the practi-
cal budget of our A100 GPU environment. To ensure robustperformance estimates and account for variance, we conduct
this evaluation over 5 independent runs with different random
seeds. Table 7 reports the resulting mean and standard devia-
tion for the different vulnerability classes. Table 7 presents the
Macro F1 scores on the CGT, DAppScan, and LLMA V test sets
across five independent runs for all four configurations. Two key
observations emerge from this comparison.
First, Causal training consistently outperforms Standard
training under both enrichment scenarios. On the CGT test
set, the ORACAL-edge Causal model attains a mean Macro F1
of 0.918 compared to 0.883 for ORACAL-edge Standard. The
Wilcoxon signed rank test (exact, two tailed) yields p=0.0625.
While this pvalue is slightly above the conventional 0.05 thresh-
old due to the small sample size ( n=5), every paired difference
is positive, indicating directional consistency. The Vargha De-
laney effect size reaches ˆA12=1.00, meaning that a randomly
selected run from the Causal model outperforms one from the
Standard model with probability 1.00. This pattern holds identi-
cally for ORACAL-node-edge (0.908 vs 0.874), confirming that
the advantage of causal disentanglement is not contingent on a
specific enrichment configuration.
On the DAppScan test set, the Causal model again shows
superiority across all five runs under both scenarios, achieving
mean Macro F1 scores of 0.771 (ORACAL-edge) and 0.759
(ORACAL-node-edge) compared to 0.709 and 0.700 for their
Standard counterparts. The Wilcoxon signed rank test yields
p=0.0625 in both cases, and the effect size remains ˆA12=1.00,
confirming that the Causal paradigm produces uniformly higher
performance on this real-world OOD benchmark.
On the LLMA V dataset, the Causal model achieves
mean Macro F1 scores of 0.813 (ORACAL-edge) and 0.802
(ORACAL-node-edge) compared to 0.757 and 0.748 for
Standard. This dataset represents a particularly informative
generalization test because LLMA V contracts were annotated by
a completely independent group of researchers using line-level
manual analysis, resulting in a label distribution that differs
16

Figure 12: Comparison of training and test losses across 50 epochs for both Causal and Standard training models on the SoliAudit test set.
Table 7: Generalization Performance (Macro F1, Mean ±Std Dev, 5 runs). ORACAL-edge: best SoliAudit F1 (all edge fields); ORACAL-node-edge: full feature set
(all node+edge fields). Causal/Standard refer to training paradigms. Red-highlighted rows indicate the best-detected vulnerability class per dataset.
Dataset Vulnerability Class ORACAL-edge Causal ORACAL-edge Standard ORACAL-node-edge Causal ORACAL-node-edge Standard
(Mean±Std Dev) (Mean±Std Dev) (Mean±Std Dev) (Mean±Std Dev)
CGT Arithmetic 0.982±0.004 0.967±0.004 0.976±0.005 0.959±0.006
LowLevelCalls 0.877±0.015 0.836±0.017 0.869±0.017 0.827±0.019
DenialOfService 0.882±0.013 0.844±0.018 0.874±0.015 0.835±0.020
TimeManipulation 0.921±0.010 0.883±0.019 0.912±0.012 0.874±0.021
Overall (Macro F1) 0.918±0.004 0.883±0.006 0.908±0.006 0.874±0.008
DAppScan Arithmetic 0.696±0.013 0.638±0.012 0.688±0.015 0.629±0.014
LowLevelCalls 0.833±0.017 0.759±0.018 0.824±0.019 0.750±0.020
DenialOfService 0.761±0.019 0.700±0.017 0.752±0.021 0.691±0.019
TimeManipulation 0.780±0.016 0.738±0.018 0.771±0.018 0.729±0.020
Overall (Macro F1) 0.771±0.011 0.709±0.012 0.759±0.013 0.700±0.014
LLMA V Arithmetic 0.862±0.012 0.819±0.015 0.854±0.014 0.810±0.017
LowLevelCalls 0.789±0.019 0.729±0.024 0.780±0.021 0.720±0.026
DenialOfService 0.773±0.022 0.718±0.021 0.764±0.024 0.709±0.023
TimeManipulation 0.817±0.013 0.762±0.016 0.808±0.015 0.753±0.018
Overall (Macro F1) 0.813±0.009 0.757±0.014 0.802±0.011 0.748±0.016
markedly from the training sources (specifically, Arithmetic
constitutes 45.9% of positive labels in LLMA V versus 40.6%
in SoliAudit). Across all four evaluated classes, the Causal
model consistently exceeds the Standard baseline by 4.3
to 6.0 percentage points under both enrichment scenarios,
demonstrating that the causal disentanglement mechanism
provides robust gains even when the test distribution deviates
substantially from the training data.
Second, ORACAL-edge slightly outperforms ORACAL-
node-edge across all datasets. The all edge field configuration
achieves 0.5–1.2 percentage points higher Macro F1 than the all
node+edge field configuration under both Causal and Standard
paradigms (witnessed in the 0.918 vs 0.908 on CGT Causal,0.771 vs 0.759 on DAppScan Causal). This marginal gap is
consistent with the ablation results in Table 6, where edge-only
enrichment (91.28% Causal) yields a slightly higher SoliAu-
dit F1 than the full node+edge enrichment (90.70% Causal).
The result suggests that while node-level enrichments provide
additional semantic context, the edge-level structural seman-
tics contribute more decisively to vulnerability detection, and
combining both modalities introduces a small amount of feature
redundancy that marginally dilutes the edge signal. Nevertheless,
the ORACAL-node-edge configuration remains competitive and
may offer advantages in explainability tasks where richer node-
level descriptions facilitate interpretation.
Taken together, the statistical analysis reveals that Causal
17

Figure 13: Inference time comparison (ms/sample) across Causal and Standard
models for various enrichment scenarios on the SoliAudit test set.
training provides stable, large, and fully consistent generaliza-
tion gains across all three benchmarks under both enrichment
configurations. The maximal effect sizes ˆA12=1.00 across
datasets, coupled with directional uniformity in all paired com-
parisons, indicate that the observed improvements are systematic
and reproducible rather than attributable to random variation,
despite the conservative nature of the test with limited runs.
Answer to RQ1
Causal training with edge enrichment yields the best per-
formance: 91.28% Macro F1 on SoliAudit (+4.09% over
Standard), with inference under 2 ms/sample. In cross-
dataset generalization, ORACAL-edge Causal achieves
0.918 (CGT), 0.771 (DAppScan), and 0.813 (LLMA V),
while ORACAL-node-edge Causal follows closely at
0.908, 0.759, and 0.802. Both consistently outperform
Standard counterparts ( ˆA12=1.00), confirming that
causal disentanglement provides systematic generaliza-
tion gains.
4.4.2. RQ2: Interpretability and Explainability Analysis
Motivation.While achieving state-of-the-art detection accu-
racy is essential, the ”black-box” nature of deep learning models
often obscures the underlying reasoning, which can hinder trust
in professional security audits. In the context of smart contract
security, a detector must not only identify a vulnerability but
also provide actionable evidence that allows human auditors to
quickly verify the root cause. This interpretability is crucial for
streamlining the auditing process and ensuring that the model is
learning meaningful security logic rather than exploiting spuri-
ous syntactic correlations.
Method.To enhance the transparency of our framework and
identify the most effective explanation approach, we conduct
a comparative study using three complementary explanation
methods: GNNExplainer [ 77], PGExplainer [ 22], and Atten-
tionExplainer. The rationale for evaluating multiple explainers
is twofold. First, different explainers operate under fundamen-
tally different paradigms, and their suitability for heterogeneous
graph structures is not known a priori. GNNExplainer optimizes
a per-instance soft mask over nodes and edges to identify the
most influential subgraph for each individual prediction. PG-
Explainer, in contrast, trains a parameterized neural networkTable 8: Explanation quality comparison using VTP metrics.Boldvalues
indicate the best-performing explainer for each dataset and metric.
Dataset Explainer MSP(%) MSR(%) MIoU(%)
LLMA V [26]GNNExplainer 35.86 39.72 27.96
PGExplainer 40.91 44.85 32.51
AttentionExplainer 27.94 31.62 19.17
DAppScan [25]GNNExplainer 33.74 37.91 26.05
PGExplainer 39.68 42.77 30.85
AttentionExplainer 25.81 29.47 17.08
to produce explanation masks that generalize across instances,
avoiding the need for per-sample optimization and thus offering
better scalability and consistency. AttentionExplainer directly
extracts attention weights from the Causal Attention mechanism
learned during training to rank node importance, offering ex-
planations aligned with the model’s internal decision process
without requiring additional backward passes. By comparing
these three approaches, we can determine which explanation
paradigm best captures the vulnerability semantics encoded by
ORACAL’s heterogeneous graph structure.
Second, raw importance scores from any single explainer
are not directly interpretable without a structured evaluation
protocol. To address this limitation, we adopt the Vulnerability
Triggering Path (VTP) evaluation framework using three met-
rics:Mean Statement Precision(MSP),Mean Statement Recall
(MSR), andMean Intersection over Union(MIoU), as defined
by Cao et al. [ 36]. These metrics quantify the overlap between
model-generated explanations and manually annotated vulner-
ability triggering paths, enabling a fine-grained assessment of
explanation quality.
Since the SoliAudit and CGTWeakness datasets do not pro-
vide line-level vulnerability annotations, we evaluate explanation
quality on the LLMA V dataset [ 26], which contains manually
verified vulnerability locations at the statement level. In addition,
we include the DAppScan dataset, which has been described in
Section 4.2.1, to assess the robustness of explanations in more
complex and diverse smart contract scenarios. For each predic-
tion, each explainer returns thetop-10 nodeswith the highest
importance scores. These nodes are then mapped back to the
corresponding source code line numbers via the node-to-line
mapping produced during graph construction, yielding a set of
predicted vulnerability-relevant statements. This predicted set is
compared against the ground-truth vulnerability triggering paths
annotated in the dataset to compute the VTP metrics (MSP, MSR,
MIoU). All experiments are conducted under the ORACAL-all
configuration, which provides the most comprehensive struc-
tural representation and achieves the strongest predictive perfor-
mance.
Quantitative Explainer Evaluation.Table 8 presents the
VTP evaluation results on both the LLMA V and DAppScan
datasets. The comparative study reveals a clear ranking among
the three explanation methods, with PGExplainer emerging as
the most effective approach for interpreting ORACAL’s predic-
tions.
PGExplainer consistently achieves the highest scores across
all metrics on both datasets. On the LLMA V dataset, PGEx-
18

plainer reaches 40.91% MSP, 44.85% MSR, and 32.51% MIoU,
outperforming GNNExplainer by 5.05, 5.13, and 4.55 percent-
age points, respectively. This advantage stems from PGEx-
plainer’s parameterized architecture, which learns a global ex-
planation model from the training data and produces explanation
masks that generalize across instances. In contrast, GNNEx-
plainer optimizes a separate mask for each prediction, which
limits its ability to capture cross-instance vulnerability patterns.
AttentionExplainer yields the weakest performance (19.17%
MIoU on LLMA V), indicating that while attention scores re-
flect the model’s learned focus, they do not directly optimize
for subgraph-level explanation quality. The attention mecha-
nism prioritizes features useful for classification rather than for
isolating complete vulnerability triggering paths, resulting in
lower alignment with ground-truth annotations compared to the
mask-based explainers.
Compared to LLMA V , all explainers obtain slightly lower
scores on DAppScan (demonstrated by PGExplainer achieving
30.85% MIoU versus 32.51% on LLMA V). This degradation is
expected, as DAppScan contains more complex and diverse exe-
cution patterns from production-grade DeFi contracts, making
it more challenging to precisely localize vulnerability trigger-
ing paths. Nevertheless, the relative ranking among explainers
remains consistent across both datasets, indicating the robust-
ness and generalizability of PGExplainer’s advantage. Based
on these findings, we recommend PGExplainer as the primary
explanation method for ORACAL, as it combines the strongest
quantitative alignment with manually annotated vulnerability
triggering paths and the practical benefit of instance-independent
explanation generation.
Qualitative Case Study.To complement the quantitative
evaluation, Figure 14 visualizes a line-level comparison
of three explainers against the ground truth for contract
0xc5B2508E878af367Ba4957BDBEb2bBc6DA5BB349.sol , a
true positive Unchecked Low Level Calls sample. The contract
consists of an oracle contract ( AmIOnTheFork ) and a splitter
contract ( EthSplit ). The split function (lines 7–19) routes
ETH or ETC to different addresses depending on a fork check,
using three low-level calls: ethAddress.call.value(msg.v
alue)() at line 11, fees.send(fee) at line 16, and
etcAddress.call.value(msg.value-fee)() at line 17.
None of these return values are checked, constituting the core
vulnerability. The ground truth additionally marks lines 22–23
(a fallback function using throw ), line 27 (external contract
instantiation), and line 28 (external address assignment) as
vulnerability-relevant.
Among the three explainers,PGExplainerachieves the clos-
est alignment with the ground truth. It correctly identifies all
three unchecked call sites (lines 11, 16, 17), the fee computation
(line 15), the fork-dependent branching logic (lines 9, 13, 14),
and the external contract references (lines 27, 28). Crucially,
PGExplainer produces very few false positives: it does not flag
boilerplate lines such as the pragma directive (line 1) or function
signatures that are not directly involved in the vulnerability.
GNNExplaineridentifies several critical lines (11, 15, 16,
17) but introduces additional false positives on structurally
prominent but semantically irrelevant nodes. For instance,it marks line 3 ( function forked() ) despite this being
a simple view function in the oracle contract that does not
participate in the unchecked call pattern. This behavior reflects
GNNExplainer’s per-instance optimization, which can overfit to
local structural salience.
AttentionExplainercaptures some vulnerability-relevant
lines (lines 11, 16, 17) but also assigns high importance to
the pragma directive (line 1) and the oracle function signature
(line 3), which are syntactically prominent but not causally re-
lated to the vulnerability. This pattern confirms that attention
weights, while useful for classification, do not reliably distin-
guish between structural importance and vulnerability-specific
relevance.
This case study demonstrates that PGExplainer’s parameter-
ized, globally trained explanation masks provide the most pre-
cise and actionable explanations for security auditors, correctly
isolating the unchecked low-level call pattern while minimizing
noise from structural artifacts.
Answer to RQ2
PGExplainer achieves the best explanation quality
(32.51% MIoU on LLMA V , 30.85% on DAppScan), pre-
cisely isolating vulnerability-triggering statements rang-
ing from unchecked call.value tosend with minimal
false positives. GNNExplainer offers competitive recall
but more false positives, while AttentionExplainer as-
signs importance to syntactically prominent but causally
irrelevant nodes. Overall, ORACAL provides reliable,
auditor-friendly explanations grounded in true vulnera-
bility semantics.
4.4.3. RQ3: Comparison with SOTA and Robustness under Ad-
versarial Attacks
Motivation.While high detection accuracy is fundamen-
tal, the practical utility of a vulnerability detector is limited if
it remains susceptible to adversarial evasion or fails to remain
competitive against established state-of-the-art architectures. In
real-world security auditing, malicious actors may intentionally
obfuscate contract logic or introduce semantic perturbations to
bypass automated tools. This necessitates a comprehensive eval-
uation of ORACAL’s performance relative to existing SOTA
models, alongside an assessment of its structural and textual ro-
bustness, to ensure that the framework maintains reliable defense
in adversarial environments.
Method.We first conduct a comparative analysis of detec-
tion performance against several SOTA graph-based models,
including GNN-SC [ 28], SCVHunter [ 11], MTVHunter [ 27],
and MANDO-HGT [ 32], as listed in Table 1 of Section 2. For
this comparison, we evaluate three variants of our proposed
framework: (i)ORACAL-base, which utilizes only the structural
graph modality; (ii)ORACAL-enrich, which relies exclusively
on the enriched semantic text modality; and (iii)ORACAL-all,
our complete multimodal architecture that fuses node/edge at-
tributes, enriched text, and graph topology.
Subsequently, we perform robustness tests by simulating re-
19

Figure 14: Line-level explainer comparison for an Unchecked Low Level Calls vulnerability (contract 0xc5B2508...5BB349.sol ). Columns show binary
importance scores (1=identified as important, 0=not) assigned by Ground Truth, GNNExplainer, PGExplainer, and AttentionExplainer. Red values indicate
disagreement with the ground truth.
alistic adversarial scenarios. In practice, an attacker may attempt
to fool the detector by subtly modifying the contract’s control- or
data-flow (graph structure) or by introducing adversarial noise
into the natural language descriptions (enriched text). Since
ORACAL leverages both graph and text modalities, we must
evaluate its robustness against both types of perturbations. Typ-
ically, adversarial attacks on graphs and text aim to minimize
the perturbation budget while maximizing the misclassification
rate. To this end, we choose the following representative attack
methods:
As illustrated in Figure 15, textual adversarial attacks of-
ten involve word-level perturbations, by replacing semantic in-
dicator keywords with their antonyms (substituting “Critical”
with “Safe”, “external” with “internal”, or “before” with “af-
ter”). Such modifications are designed to preserve the overall
grammatical structure while reversing the semantic meaning
to mislead the model’s textual modality. Similarly, Figure 16
demonstrates structural attacks, which involve the addition of
spurious edges or the removal of critical execution paths, no-
tably adding an edge from CG to DFG or removing a CFG-DFG
link. These structural alterations probe the model’s reliance on
specific graph topologies and its susceptibility to tampering in
contract control- and data-flow.
To comprehensively evaluate robustness against these per-
turbations, we select Representative attack methods:
•HSAttack [ 15]:For heterogeneous architectures
(ORACAL, SCVHunter, MANDO-HGT), we employ
the Metapath-free Structure Attack (HSAttack). HSA
partitions the graph into single-edge-type subgraphs
to identify the importance of specific relations without
requiring structural metapaths. It then iteratively adds or
removes high-impact edges to assess structural stability.
Figure 15: Illustration of textual adversarial perturbation via antonym substitu-
tion.
•CAMA [ 78]:For models using homogeneous graphs or
simpler representations (MTVHunter, GNN-SC), we uti-
lize the Class Activation Mapping-based Attack (CAMA).
CAMA localizes structural vulnerabilities by producing
node-level importance maps, enabling unnoticeable per-
turbations.
•SubAttack [ 79]:To evaluate the robustness of text-based
modalities (ORACAL-enrich, ORACAL-all), we use Sub-
Attack. This word-level attack identifies semantic indi-
cator keywords and performs antonym substitution while
ensuring the adversarial text remains readable.
All three methods are implemented as hard-label black-box
evasion attacks, simulating a realistic threat model where the
attacker has no access to the model’s internal parameters and
20

Figure 16: Structural adversarial perturbation involving edge addition and re-
moval.
strikes during the inference phase. In this context, we evaluate
robustness using theAttack Success Rate (ASR), as defined in
Table 4.
Robustness evaluation setting.Due to limited computa-
tional resources, we evaluate robustness using a single repre-
sentative setting for each attack type across all experiments.
For structural perturbations (HSAttack and CAMA), we set
theedge budget to 0.1, meaning the attacker can add or re-
move edges up to 10% of the total edge count in the original
graph. For textual perturbations (SubAttack), we restrict the
keyword budget to 10 words per node enrichment, ensuring
that the adversarial modifications remain unnoticeable to human
auditors while testing the model’s reliance on semantic cues.
The application of these attacks is model-dependent: CAMA
is applied to models targeting homogeneous representations
(GNN-SC and MTVHunter), as it was specifically designed
for such architectures. In contrast, HSAttack is utilized for
the remaining heterogeneous models, including SCVHunter,
MANDO-HGT, ORACAL-base, and the full ORACAL-all vari-
ant. Finally, adversarial text attacks (SubAttack) are exclusively
performed on models that incorporate enriched natural language
input—specifically ORACAL-enrich and ORACAL-all.
Detection performance comparison.Table 9 (Original
setting) summarizes the detection accuracy of ORACAL com-
pared to SOTA methods across the three datasets. ORACAL-all
consistently achieves the highest Macro F1 scores (90.48% on
SoliAudit, 90.83% on CGTWeakness, and 72.82% on DApp-
Scan), outperforming baseline models such as MANDO-HGT
(82.63%), MTVHunter (78.10%), and SCVHunter, which ex-
hibits lower performance on the real-world DAppScan dataset
(28.44%).
Modality-specific analysis reveals that while structural in-
formation (ORACAL-base) is effective for curated benchmarks,
semantic enrichment (ORACAL-enrich) provides superior invari-
ance on complex, out-of-distribution datasets like DAppScan.
By fusing both,ORACAL-allachieves 72.82% F1 on DAppScan,
a+10.82% lead over MANDO-HGT, demonstrating that multi-
modal integration is essential for capturing high-level logic that
generalizes to industrial-grade contracts.
This trend is further supported by the Receiver Operating
Characteristic (ROC) curves shown in Figure 20. On the So-
liAudit dataset, ORACAL-all achieves a high Area Under the
Curve (AUC) of 0.96, followed by ORACAL-base (0.95), whileSCVHunter exhibits a lower curve with an AUC of 0.73. A
similar trend is observed in CGTWeakness, where ORACAL-all
maintains an AUC of 0.96. A notable performance gap appears
in the real-world DAppScan dataset, where current SOTA meth-
ods exhibit limitations: SCVHunter’s AUC decreases to 0.39,
while ORACAL-all maintains resilience with an AUC of 0.76.
The slope of ORACAL’s curve near the y-axis across all datasets
demonstrates its ability to maintain a high True Positive Rate
(TPR) at low False Positive Rates (FPR), which is practical for
security tools seeking to minimize manual audit overhead.
Robustness result.The results of the evasion attacks are
summarized in Table 9. Quantitative analysis reveals that
ORACAL-all maintains stability compared to SOTA GNN
models. While SCVHunter exhibits a significant performance
degradation under HSAttack, with a Macro F1 decrease of up
to 18.58% and an Attack Success Rate (ASR) of 18.73% on
SoliAudit, ORACAL-all remains relatively stable. Specifically,
under the same HSAttack structural perturbation, ORACAL-
all’s F1 score decreases by only 2.35% on SoliAudit and 2.18%
on DAppScan, maintaining an ASR between 2.88%–3.21%.
The robustness of ORACAL-all is also profoundly evident
in the textual and combined attack scenarios. As highlighted in
Table 9, the model exhibits exceptional resilience against textual
adversarial perturbations (SubAttack), achieving the lowest ASR
(∼2.0%) and minimal F1 degradation (1.11%–1.84%) across all
datasets. Even in the ”Both” scenario, where both structural and
textual attacks are applied simultaneously to simulate a worst-
case evasion attempt, ORACAL-all preserves a high Macro F1
score of 85.91% on SoliAudit, with its performance degradation
(-4.57%) still being significantly lower than that of any single
structural attack on baseline SOTA models (exemplified by the
-9.69% decrease for MANDO-HGT).
This notable robustness can be attributed to two core mecha-
nisms in ORACAL’s architecture: (i)Multimodal Redundancy:
The fusion of diverse modalities (graph topology and semantic
enrichment) provides a ”fail-safe” mechanism. If an attacker cor-
rupts the graph structure, the model can still rely on the invariant
semantic cues in the enriched text, and vice versa. (ii)Causal
Attention Learning: By disentangling causal invariant features
from spurious correlations, ORACAL’s training paradigm filters
out adversarial noise that does not align with the true underlying
causes of vulnerabilities, thereby preventing the model from
being easily misled by superficial perturbations.
Answer to RQ3
ORACAL-all achieves the highest detection accuracy
(90.83% F1 on CGTWeakness, ROC-AUC up to 0.96)
and superior adversarial robustness compared to SOTA:
while SCVHunter degrades by up to 18.58% F1 under
HSAttack (ASR 18.73%), ORACAL-all limits degrada-
tion to 2.35% (ASR ∼3%). This resilience stems from
ORACAL’s multimodal redundancy and causal attention,
which filter out structural and semantic adversarial noise
without relying on spurious correlations.
21

Figure 17: ROC Curve - SoliAudit
 Figure 18: ROC Curve - CGTWeakness
 Figure 19: ROC Curve - DAppScan
Figure 20: ROC-AUC comparison across datasets.
5. Threats to Validity
This section discusses threats to internal, external, construct,
and conclusion validity, along with our mitigation strategies.
5.1. Internal Validity
•Data Preprocessing and Extraction:The reliance on
compilation and AST extraction may bias our datasets
toward well-structured contracts. We mitigate this by
following standard preprocessing pipelines and ensuring
our extraction scripts are deterministic for reproducibility.
•RAG Enrichment and LLM Stochasticity:Semantic
enrichment quality depends on corpus composition and
LLM stochasticity. To control for variability, we fix all
random seeds, use zero temperature for LLM inference,
construct the corpus from authoritative sources, and apply
a two-stage retrieval with re-ranking.
•Checkpoint Selection for Generalization:Evaluating
all configurations is computationally expensive, so gener-
alization evaluation uses only the top-performing check-
point from each paradigm. We mitigate this limitation
by selecting checkpoints based on primary benchmark
performance to represent practically relevant models.
5.2. External Validity
•Generalizability:Our evaluation focuses on Solidity
datasets covering DASP categories, which may not trans-
fer to other languages or novel vulnerabilities ranging
from flash loans to MEV . We address this by evaluat-
ing across structurally diverse datasets, including two in-
dependent out-of-distribution datasets (DAppScan and
LLMA V), demonstrating generalization across distinct
collection and labeling methodologies.
•Temporal Concept Drift:The model is trained on static
snapshots, risking performance degradation as the Solid-
ity ecosystem evolves and new vulnerabilities emerge.
We partially mitigate this through ORACAL’s modular
RAG corpus, which can be updated independently, and
its causal attention mechanism designed to learn invariant
semantics, improving resilience to distributional shifts.5.3. Construct Validity
•Label Reliability:Ground truth labels are derived from
heterogeneous sources, introducing potential noise. We
mitigate this by using consolidated datasets with multi-
tool voting and cross-referencing automated labels with
manual case studies to validate semantic consistency.
•Explainability Ground Truth:Explainability evalua-
tion relies on LLMA V’s manual line-level annotations,
involving subjective boundary decisions. We address this
by evaluating on both LLMA V and DAppScan, demon-
strating consistent explainer rankings across independent
datasets, reducing the likelihood of methodology artifacts.
•Adversarial Budget Scope:We evaluate robustness un-
der fixed perturbation budgets, while real-world adver-
saries may operate under different constraints. We ac-
knowledge this boundary condition and note that evalu-
ating a range of budgets would provide a more complete
robustness profile.
•Evaluation Metrics:Standard metrics may not capture all
aspects of practical auditing (specifically, Macro F1 treats
all classes equally regardless of severity). We supplement
aggregate metrics with per-class breakdowns, qualitative
case studies, and structural bias analyses.
5.4. Conclusion Validity
•Statistical Soundness:The out-of-distribution test sets
are relatively small, and Wilcoxon signed-rank tests ( n=
5) yield p-values constrained by a discrete distribution.
We complement p-values with Vargha-Delaney ˆA12effect
sizes, where consistently maximal values ( ˆA12=1.00)
provide strong evidence of systematic gains.
•Implicit Overfitting through Iterative Design:Re-
peated evaluation may inadvertently guide architectural
choices toward test-specific optimizations. We mitigate
this by maintaining strict train-test separation, reserv-
ing two independent OOD datasets (DAppScan and
LLMA V) unused during model selection, and reporting
performance across multiple independent benchmarks.
22

Table 9: Evasion attack performance comparison. Values in parentheses indicate the relative F1 decrease vs. the Original setting. The light-red row indicates the
method achieving the highest Original F1 score (ORACAL-all). The dark-red row indicates the attack scenario with the lowest F1 degradation and ASR (ORACAL-all
under SubAttack). “–” denotes not applicable (original, pre-attack setting; ASR is undefined).
Model MethodSoliAudit CGTWeakness DAppScan
F1(%) ASR(%) F1(%) ASR(%) F1(%) ASR(%)
GNN-SCOriginal 70.74 - 70.20 - 66.47 -
HSAttack 61.92(-8.82) 11.04 61.07(-9.13) 11.62 57.90(-8.57) 10.31
MTVHunterOriginal 78.10 - 78.73 - 58.94 -
HSAttack 69.65(-8.45) 10.91 69.82(-8.91) 11.47 51.92(-7.02) 9.73
SCVHunterOriginal 51.62 - 50.20 - 28.44 -
HSAttack 33.04(-18.58) 18.73 32.71(-17.49) 19.16 17.29(-11.15) 12.02
MANDO-HGTOriginal 82.63 - 83.77 - 62.00 -
HSAttack 72.94(-9.69) 12.41 73.15(-10.62) 13.07 54.47(-7.53) 10.11
ORACAL-baseOriginal 89.35 - 90.32 - 63.65 -
HSAttack 85.54(-3.81) 4.22 86.33(-3.99) 4.51 60.47(-3.18) 4.63
ORACAL-enrichOriginal 74.60 - 76.83 - 68.10 -
SubAttack 71.18(-3.42) 4.68 73.02(-3.81) 4.94 64.61(-3.49) 5.03
ORACAL-allOriginal 90.48 - 90.83 - 72.82 -
HSAttack 88.13(-2.35) 2.94 88.07(-2.76) 3.21 70.64(-2.18) 2.88
SubAttack 88.81(-1.67) 2.13 88.99(-1.84) 2.27 71.71(-1.11) 1.96
Both 85.91(-4.57) 5.71 85.36(-5.47) 6.04 68.61(-4.21) 5.31
6. Conclusion and Future Work
In this paper, we presented ORACAL, a novel framework
for smart contract vulnerability detection that bridges the gap
between graph-based structural analysis and LLM-based se-
mantic reasoning. By constructing a comprehensive hetero-
geneous graph and enriching critical nodes with “Operational
Context” and “Security Analysis” via a trusted RAG pipeline,
ORACAL achieves state-of-the-art detection performance, reach-
ing a Macro F1 of 91.28% on the SoliAudit test set (with 97.34%
per-class F1 on Arithmetic bugs). Our Causal Attention mech-
anism effectively disentangles true vulnerability signals from
spurious correlations, yielding consistent generalization gains
across diverse datasets: 91.8% Macro F1 on CGT, 77.1% on
DAppScan, and 81.3% on LLMA V , outperforming the Standard
paradigm by 3.5–6.2 percentage points across all benchmarks.
Beyond detection accuracy, ORACAL demonstrates strong
explainability and robustness. In explainability evaluation, PG-
Explainer achieves 32.51% MIoU on the line-level annotated LL-
MA V dataset and 30.85% on DAppScan, providing actionable,
auditor-friendly explanations that precisely isolate vulnerability-
triggering statements such as unchecked low-level calls. In
adversarial robustness testing, ORACAL-all maintains an At-
tack Success Rate (ASR) of approximately 2–3% under both
structural (HSAttack) and textual (SubAttack) perturbations,
compared to up to 18.73% ASR for existing methods such as
SCVHunter, confirming that the multimodal architecture and
causal training paradigm jointly provide effective defense against
adversarial evasion.Future work will focus on three directions: (1) extending
the framework toNode Classificationto pinpoint the exact lo-
cations of vulnerabilities within the code (line-level detection),
(2) employing established evaluation metrics [ 80] to systemati-
cally assess the faithfulness and relevance of RAG-generated se-
mantic descriptions, ensuring that the enriched content remains
grounded in technical truth, and (3) exploringIncremental
Learningtechniques [ 81] to efficiently integrate new samples
without full retraining. ORACAL represents a significant step to-
wards transparent, automated, and explainable security auditing
for the blockchain ecosystem.
Acknowledgement
This research is funded by Vietnam National University Ho
Chi Minh City (VNU-HCM) under grant number NCM2025-26-
01.
Phan The Duy was funded by the Postdoctoral Scholarship
Programme of Vingroup Innovation Foundation (VINIF), Vin-
University, code VINIF.2025.STS.20.
References
[1]SecurityWeek. 223 million stolen in cetus protocol hack. https://ww
w.securityweek.com/223-million-stolen-in-cetus-protoco
l-hack/, May 2025. Online.
[2]Halborn. Explained: The zoth hack. https://www.halborn.com/bl
og/post/explained-the-zoth-hack-march-2025 , March 2025.
Online.
23

[3]CoinDesk. Truebit token crashes 99.9% after usd 26.6m exploit drains
8,535 eth. https://www.coindesk.com/markets/2026/01/09/tr
uebit-token-tru-crashes-99-9-after-usd26-6m-exploit-d
rains-8-535-eth, Jan 2026. Online.
[4]BlockSec. $17m closed-source smart contract exploit: Arbitrary-call
vulnerability in swapnet and aperture finance. https://blocksec.com
/blog/17m-closed-source-smart-contract-exploit-arbitra
ry-call-swapnet-aperture, Jan 2026. Online.
[5]SlowMist. 2025 blockchain security and aml annual report. https:
//lazarus.day/media/post/files/2026/01/02/2025-Blockch
ain-Security-and-AML-Annual-ReportEN.pdf , December 2025.
Online.
[6]Niosha Hejazi and Arash Habibi Lashkari. A comprehensive survey of
smart contracts vulnerability detection tools: Techniques and method-
ologies.Journal of Network and Computer Applications, 237:104142,
2025.
[7]Gerardo Iuliano and Dario Di Nucci. Smart contract vulnerabilities, tools,
and benchmarks: An updated systematic literature review.Journal of
Systems and Software, page 112788, 2026.
[8]Michael Bresil, Pwc Prasad, Md Shohel Sayeed, and Umar Ali Bukar.
Deep learning-based vulnerability detection solutions in smart contracts:
A comparative and meta-analysis of existing approaches.IEEE Access,
2025.
[9]Joao Crisostomo, Fernando Bacao, and Victor Lobo. Machine learning
methods for detecting smart contracts vulnerabilities within ethereum
blockchain- a review.Expert Systems With Applications, 268:126353,
2025.
[10] Ben Wang, Yanxiang Tong, Shunhui Ji, Hai Dong, Xiapu Luo, and
Pengcheng Zhang. A review of learning-based smart contract vulner-
ability detection: A perspective on code representation.ACM Transactions
on Software Engineering and Methodology, 2025.
[11] Feng Luo, Ruijie Luo, Ting Chen, Ao Qiao, Zheyuan He, Shuwei Song,
Yu Jiang, and Sixing Li. Scvhunter: Smart contract vulnerability detection
based on heterogeneous graph attention network. InProceedings of the
IEEE/ACM 46th international conference on software engineering, pages
1–13, 2024.
[12] Zhaoyi Meng, Zexin Zhang, Wansen Wang, Jie Cui, and Hong Zhong.
Smartscope: Smart contract vulnerability detection via heterogeneous
graph embedding with local semantic enhancement.Expert Systems with
Applications, page 129857, 2025.
[13] He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, and Chunyan Miao.
Hgattack: Transferable heterogeneous graph adversarial attack. In2024
IEEE International Conference on Agents (ICA), pages 100–105. IEEE,
2024.
[14] Yuling Wang, Zihui Chen, Pengfei Jiao, and Xiao Wang. Heta: Relation-
wise heterogeneous graph foundation attack model.arXiv preprint
arXiv:2506.07428, 2025.
[15] Haoran Li, Jian Xu, Long Yin, Qiang Wang, Yongzhen Jiang, and Jingyi
Liu. Metapath-free adversarial attacks against heterogeneous graph neural
networks.Information Sciences, 713:122143, 2025.
[16] Zeyu Yang, Zhao Meng, Xiaochen Zheng, and Roger Wattenhofer. Assess-
ing adversarial robustness of large language models: An empirical study.
arXiv preprint arXiv:2405.02764, 2024.
[17] Muhammad Shahid Jabbar, Sadam Al-Azani, Abrar Alotaibi, and Moataz
Ahmed. Red teaming large language models: A comprehensive review and
critical analysis.Information Processing&Management, 62(6):104239,
2025.
[18] Zongwei Li, Xiaoqi Li, Wenkai Li, and Xin Wang. Scalm: Detecting bad
practices in smart contracts through llms. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 39, pages 470–477, 2025.
[19] Fahad Al Debeyan, Tracy Hall, and Lech Madeyski. Emerging results
in using explainable ai to improve software vulnerability prediction. In
Proceedings of the 33rd ACM International Conference on the Foundations
of Software Engineering, pages 561–565, 2025.
[20] Hanouf Al Ghanmi, Sabreen Ahmadjee, and Rami Bahsoon. Evaluating the
need for explanations in blockchain smart contracts to reconcile surprises.
ACM Transactions on Software Engineering and Methodology, 34(8):1–35,
2025.
[21] Sebastian Lapuschkin, Stephan W ¨aldchen, Alexander Binder, Gr ´egoire
Montavon, Wojciech Samek, and Klaus-Robert M ¨uller. Unmasking clever
hans predictors and assessing what machines really learn.Nature commu-nications, 10(1):1096, 2019.
[22] Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong,
Haifeng Chen, and Xiang Zhang. Parameterized explainer for graph neural
network.Advances in neural information processing systems, 33:19620–
19631, 2020.
[23] Jian-Wei Liao, Tsung-Ta Tsai, Chia-Kang He, and Chin-Wei Tien. Soliau-
dit: Smart contract vulnerability assessment based on machine learning
and fuzz testing. In2019 Sixth International Conference on Internet of
Things: Systems, Management and Security (IOTSMS), pages 458–465.
IEEE, 2019.
[24] Monika Di Angelo and Gernot Salzer. Consolidation of ground truth sets
for weakness detection in smart contracts. InInternational Conference
on Financial Cryptography and Data Security, pages 439–455. Springer,
2023.
[25] Zibin Zheng, Jianzhong Su, Jiachi Chen, David Lo, Zhijie Zhong, and
Mingxi Ye. Dappscan: building large-scale datasets for smart contract
weaknesses in dapp projects.IEEE Transactions on Software Engineering,
50(6):1360–1373, 2024.
[26] Francesco Salzano, Cosmo Kevin Antenucci, Simone Scalabrino, Giovanni
Rosa, Rocco Oliveto, and Remo Pareschi. An empirical analysis of vulner-
ability detection tools for solidity smart contracts using line level manually
annotated vulnerabilities.arXiv preprint arXiv:2505.15756, 2025.
[27] Guokai Sun, Yuan Zhuang, Shuo Zhang, Xiaoyu Feng, Zhenguang Liu,
and Liguo Zhang. Mtvhunter: Smart contracts vulnerability detection
based on multi-teacher knowledge translation. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 39, pages 15169–15176,
2025.
[28] Yoo-Young Cheong, Jihwan Shin, Taekyung Kim, Jinhyun Ahn, Dong-
Hyuk Im, et al. Gnn-based ethereum smart contract multi-label vulnerabil-
ity detection. In2024 International Conference on Information Networking
(ICOIN), pages 57–61. IEEE, 2024.
[29] Loi Luu, Duc-Hiep Chu, Hrishi Olickel, Prateek Saxena, and Aquinas
Hobor. Making smart contracts smarter. InProceedings of the 2016 ACM
SIGSAC conference on computer and communications security, pages
254–269, 2016.
[30] Josselin Feist, Gustavo Grieco, and Alex Groce. Slither: a static analysis
framework for smart contracts. In2019 IEEE/ACM 2nd International
Workshop on Emerging Trends in Software Engineering for Blockchain
(WETSEB), pages 8–15. IEEE, 2019.
[31] Hongjun Wu, Zhuo Zhang, Shangwen Wang, Yan Lei, Bo Lin, Yihao Qin,
Haoyu Zhang, and Xiaoguang Mao. Peculiar: Smart contract vulnerability
detection based on crucial data flow graph and pre-training techniques.
In2021 IEEE 32nd International Symposium on Software Reliability
Engineering (ISSRE), pages 378–389. IEEE, 2021.
[32] Hoang H Nguyen, Nhat-Minh Nguyen, Chunyao Xie, Zahra Ahmadi,
Daniel Kudendo, Thanh-Nam Doan, and Lingxiao Jiang. Mando-hgt: Het-
erogeneous graph transformers for smart contract vulnerability detection.
In2023 IEEE/ACM 20th International Conference on Mining Software
Repositories (MSR), pages 334–346. IEEE, 2023.
[33] Asem Ghaleb and Karthik Pattabiraman. How effective are smart con-
tract analysis tools? evaluating smart contract static analysis tools using
bug injection. InProceedings of the 29th ACM SIGSOFT international
symposium on software testing and analysis, pages 415–427, 2020.
[34] Xuyan Li, Jie Wang, and Zheng Yan. Can graph neural networks be
adequately explained? a survey.ACM Computing Surveys, 57(5):1–36,
2025.
[35] Zhaoyang Chu, Yao Wan, Qian Li, Yang Wu, Hongyu Zhang, Yulei Sui,
Guandong Xu, and Hai Jin. Graph neural networks for vulnerability
detection: A counterfactual explanation. InProceedings of the 33rd ACM
SIGSOFT International Symposium on Software Testing and Analysis,
pages 389–401, 2024.
[36] Sicong Cao, Xiaobing Sun, Xiaoxue Wu, David Lo, Lili Bo, Bin Li, and
Wei Liu. Coca: Improving and explaining graph neural network-based
vulnerability detection systems. InProceedings of the IEEE/ACM 46th
International Conference on Software Engineering, pages 1–13, 2024.
[37] Jiarong Xu, Junru Chen, Siqi You, Zhiqing Xiao, Yang Yang, and Jiangang
Lu. Robustness of deep learning models on graphs: A survey.AI Open,
2:69–78, 2021.
[38] Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie
Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, et al.
Graphcodebert: Pre-training code representations with data flow.arXiv
24

preprint arXiv:2009.08366, 2020.
[39] Yuan Zhuang, Zhenguang Liu, Peng Qian, Qi Liu, Xiang Wang, and
Qinming He. Smart contract vulnerability detection using graph neural
networks. InProceedings of the twenty-ninth international conference on
international joint conferences on artificial intelligence, pages 3283–3290,
2021.
[40] Zhenguang Liu, Peng Qian, Xiaoyang Wang, Yuan Zhuang, Lin Qiu, and
Xun Wang. Combining graph neural networks with expert knowledge for
smart contract vulnerability detection.IEEE Transactions on Knowledge
and Data Engineering, 35(2):1296–1310, 2021.
[41] Aria Seo, Young-Tak Kim, Ji Seok Yang, YangSun Lee, and Yunsik Son.
Software weakness detection in solidity smart contracts using control
and data flow analysis: A novel approach with graph neural networks.
Electronics (2079-9292), 13(16), 2024.
[42] Jingjie Xu, Ting Wang, Mingqi Lv, Tieming Chen, Tiantian Zhu, and
Baiyang Ji. Mvd-hg: multigranularity smart contract vulnerability de-
tection method based on heterogeneous graphs.Cybersecurity, 7(1):55,
2024.
[43] Filippo Contro, Marco Crosara, Mariano Ceccato, and Mila Dalla Preda.
Ethersolve: Computing an accurate control-flow graph from ethereum
bytecode. In2021 IEEE/ACM 29th International Conference on Program
Comprehension (ICPC), pages 127–137. IEEE, 2021.
[44] Solidity Documentation Contributors. Solidity documentation. https:
//docs.soliditylang.org/_/downloads/en/latest/pdf/ , 2026.
Accessed: February 15, 2026.
[45] JZ Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use
large language models while reducing cost and improving performance,
2024.arXiv preprint arXiv:2305.05176.
[46] Adrien Laurent. Llm api pricing comparison (2025): Openai, gemini,
claude. https://intuitionlabs.ai/articles/llm-api-prici
ng-comparison-2025, 2026. Accessed: 2026-02-15.
[47] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shufan Liu, Xuanzhe
Liu, and Xin Jin. Ragcache: Efficient knowledge caching for retrieval-
augmented generation.ACM Transactions on Computer Systems, 44(1):1–
27, 2025.
[48] Chuangtao Ma, Zeyu Zhang, Arijit Khan, Sebastian Schelter, and Paul
Groth. Cost-efficient rag for entity matching with llms: A blocking-based
exploration.arXiv preprint arXiv:2602.05708, 2026.
[49] Zimian Liu, Han Qiu, Wei Guo, Junhu Zhu, and Qingxian Wang. Nie-
gat: node importance evaluation method for inter-domain routing network
based on graph attention network.Journal of Computational Science,
65:101885, 2022.
[50] Yangding Li, Shaobin Fu, Yangyang Zeng, Hao Feng, Ruoyao Peng,
Jinghao Wang, and Shichao Zhang. Centrality-based relation aware hetero-
geneous graph neural network.Knowledge-Based Systems, 283:111174,
2024.
[51] Yue Peng, Jiwen Xia, Dafeng Liu, Miao Liu, Long Xiao, and Benyun
Shi. Unifying topological structure and self-attention mechanism for node
classification in directed networks.Scientific Reports, 15(1):805, 2025.
[52] Elizaveta Kovtun, Maksim Makarenko, Natalia Semenova, Alexey Zaytsev,
and Semen Budennyy. Pine: Pipeline for important node exploration in
attributed networks.arXiv preprint arXiv:2512.07244, 2025.
[53] Petar Veli ˇckovi ´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero,
Pietro Lio, and Yoshua Bengio. Graph attention networks.arXiv preprint
arXiv:1710.10903, 2017.
[54] Stephen B Seidman. Network structure and minimum degree.Social
networks, 5(3):269–287, 1983.
[55] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. The
pagerank citation ranking: Bringing order to the web. Technical report,
Stanford infolab, 1999.
[56] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne
Lefebvre. Fast unfolding of communities in large networks.Journal of
statistical mechanics: theory and experiment, 2008(10):P10008, 2008.
[57] Linton C Freeman. A set of measures of centrality based on betweenness.
Sociometry, pages 35–41, 1977.
[58] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems,
33:9459–9474, 2020.
[59] Google AI Developers. Gemini 3 developer guide. https://ai.google.dev/gemini-api/docs/gemini-3 , 2025. Accessed on February 18,
2026.
[60] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack:
Packaged resources to advance general chinese embedding, 2023.
[61] Niklas Muennighoff, Nouamane Tazi, Lo ¨ıc Magne, and Nils Reimers.
Mteb: Massive text embedding benchmark. InProceedings of the 17th
Conference of the European Chapter of the Association for Computational
Linguistics, pages 2014–2037, 2023.
[62] Gavin Wood et al. Ethereum: A secure decentralised generalised transac-
tion ledger.Ethereum project yellow paper, 151(2014):1–32, 2014.
[63] SWC Registry Contributors. Smart contract weakness classification (swc)
registry. https://swcregistry.io/ , 2026. Accessed: February 15,
2026.
[64] Shihao Xia, Mengting He, Linhai Song, and Yiying Zhang. Sc-bench:
A large-scale dataset for smart contract auditing. In2025 IEEE/ACM
International Workshop on Large Language Models for Code (LLM4Code),
pages 57–64. IEEE, 2025.
[65] Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao. Making large
language models a better foundation for dense retrieval, 2023.
[66] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and
Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-
granularity text embeddings through self-knowledge distillation, 2024.
[67] Xu Yang, Hanwang Zhang, Guojun Qi, and Jianfei Cai. Causal attention
for vision-language tasks. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 9847–9857, 2021.
[68] Yongduo Sui, Xiang Wang, Jiancan Wu, Min Lin, Xiangnan He, and Tat-
Seng Chua. Causal attention for interpretable and generalizable graph
classification. InProceedings of the 28th ACM SIGKDD conference on
knowledge discovery and data mining, pages 1696–1705, 2022.
[69] Shuailin Yang, Jiadong Ren, Jiazheng Li, and Dekai Zhang. Vuldiac:
Vulnerability detection and interpretation based on augmented cfg and
causal attention learning.Journal of Systems and Software, page 112595,
2025.
[70] Amrith Setlur, Benjamin Eysenbach, Virginia Smith, and Sergey Levine.
Maximizing entropy on adversarial examples can improve generalization.
InICLR 2022 Workshop on PAIR {\textasciicircum}2Struct: Privacy,
Accountability, Interpretability, Robustness, Reasoning on Structured Data,
2022.
[71] Michael Zhang, Nimit S Sohoni, Hongyang R Zhang, Chelsea Finn, and
Christopher R ´e. Correct-n-contrast: A contrastive approach for improving
robustness to spurious correlations.arXiv preprint arXiv:2203.01517,
2022.
[72] Tae Kyun Kim. T test as a parametric statistic.Korean journal of anesthe-
siology, 68(6):540–546, 2015.
[73] Robert F Woolson. Wilcoxon signed-rank test.Wiley encyclopedia of
clinical trials, pages 1–3, 2007.
[74] Andr ´as Vargha and Harold D Delaney. A critique and improvement of the
cl common language effect size statistics of mcgraw and wong.Journal of
Educational and Behavioral Statistics, 25(2):101–132, 2000.
[75] Mohammad S Sorower. A literature survey on algorithms for multi-label
learning.Oregon State University, Corvallis, 18(1):25, 2010.
[76] Thomas Mccabe. Cyclomatic complexity and the year 2000.IEEE Soft-
ware, 13(3):115–117, 1996.
[77] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure
Leskovec. Gnnexplainer: Generating explanations for graph neural net-
works.Advances in neural information processing systems, 32, 2019.
[78] Xin Wang, Heng Chang, Beini Xie, Tian Bian, Shiji Zhou, Daixin Wang,
Zhiqiang Zhang, and Wenwu Zhu. Revisiting adversarial attacks on graph
neural networks for graph classification.IEEE Transactions on Knowledge
and Data Engineering, 36(5):2166–2178, 2023.
[79] Chenqi Hua, Xiaojian Liu, Yi Zhu, Chaowei Zhang, Yun Li, Yunhao Yuan,
and Jipeng Qiang. Subattack: A word-level adversarial textual attack
method via antonym substitution.Engineering Applications of Artificial
Intelligence, 163:113159, 2026.
[80] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng
Liu. Evaluation of retrieval-augmented generation: A survey. InCCF
Conference on Big Data, pages 102–120. Springer, 2024.
[81] Justin Leo and Jugal Kalita. Survey of continuous deep learning meth-
ods and techniques used for incremental learning.Neurocomputing,
582:127545, 2024.
25
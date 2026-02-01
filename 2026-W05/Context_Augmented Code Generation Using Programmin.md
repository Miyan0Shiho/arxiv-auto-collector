# Context-Augmented Code Generation Using Programming Knowledge Graphs

**Authors**: Shahd Seddik, Fahd Seddik, Iman Saberi, Fatemeh Fard, Minh Hieu Huynh, Patanamon Thongtanunam

**Published**: 2026-01-28 17:58:30

**PDF URL**: [https://arxiv.org/pdf/2601.20810v1](https://arxiv.org/pdf/2601.20810v1)

## Abstract
Large Language Models (LLMs) excel at code generation but struggle with complex problems. Retrieval-Augmented Generation (RAG) mitigates this issue by integrating external knowledge, yet retrieval models often miss relevant context, and generation models hallucinate with irrelevant data. We propose Programming Knowledge Graph (PKG) for semantic representation and fine-grained retrieval of code and text. Our approach enhances retrieval precision through tree pruning and mitigates hallucinations via a re-ranking mechanism that integrates non-RAG solutions. Structuring external data into finer-grained nodes improves retrieval granularity. Evaluations on HumanEval and MBPP show up to 20% pass@1 accuracy gains and a 34% improvement over baselines on MBPP. Our findings demonstrate that our proposed PKG approach along with re-ranker effectively address complex problems while maintaining minimal negative impact on solutions that are already correct without RAG. The replication package is published at https://github.com/iamshahd/ProgrammingKnowledgeGraph

## Full Text


<!-- PDF content starts -->

Context-Augmented Code Generation Using Programming Knowledge Graphs
SHAHD SEDDIK, FAHD SEDDIK, IMAN SABERI, and FATEMEH FARD,University of British Columbia,
Canada
MINH HIEU HUYNH and PATANAMON THONGTANUNAM,University of Melbourne, Australia
Large Language Models (LLMs) excel at code generation but struggle with complex problems. Retrieval-Augmented Generation
(RAG) mitigates this issue by integrating external knowledge, yet retrieval models often miss relevant context, and generation
models hallucinate with irrelevant data. We propose Programming Knowledge Graph (PKG) for semantic representation and fine-
grained retrieval of code and text. Our approach enhances retrieval precision through tree pruning and mitigates hallucinations via
a re-ranking mechanism that integrates non-RAG solutions. Structuring external data into finer-grained nodes improves retrieval
granularity. Evaluations on HumanEval and MBPP show up to 20% pass@1 accuracy gains and a 34% improvement over baselines on
MBPP. Our findings demonstrate that our proposed PKG approach along with re-ranker effectively address complex problems while
maintaining minimal negative impact on solutions that are already correct without RAG. The replication package is published at
https://github.com/iamshahd/ProgrammingKnowledgeGraph.
CCS Concepts:•Computing methodologiesÑNatural language processing;•Software and its engineering;
Additional Key Words and Phrases: Programming Knowledge Graph, Retrieval Augmented Generation, LLM Code Generation
ACM Reference Format:
Shahd Seddik, Fahd Seddik, Iman Saberi, Fatemeh Fard, Minh Hieu Huynh, and Patanamon Thongtanunam. 2018. Context-Augmented
Code Generation Using Programming Knowledge Graphs.ACM Trans. Softw. Eng. Methodol.37, 4, Article 111 (August 2018), 33 pages.
https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Large Language Models (LLMs) have enabled natural-language-to-code generation on a wide range of programming tasks
[3,13,27]. However, functional correctness often depends on external programming knowledge that is not consistently
stored in model parameters, including API usage conventions, corner cases, and idiomatic patterns. Consequently,
Retrieval-Augmented Generation (RAG) has become a practical mechanism for grounding code generation in external
sources such as library documentation, tutorials, and code repositories [ 38,43]. Recent large-scale evaluations further
show that end-to-end gains depend jointly on retrieval quality and the generator’s ability to use retrieved context
effectively [35].
At the same time, code-oriented RAG remains brittle. Prior work has shown that retrieved context can be redundant,
partially relevant, or misleading, and that models may under-utilize or be distracted by long contexts [ 35]. More broadly,
analyses of context utilization caution that performance measured under simplified or synthetic assumptions can
Authors’ Contact Information: Shahd Seddik, shahd.seddik@ubc.ca; Fahd Seddik, fahd.seddik@ubc.ca; Iman Saberi, iman.saberi@ubc.ca; Fatemeh
Fard, fatemeh.fard@ubc.ca, University of British Columbia, Kelowna, Canada; Minh Hieu Huynh, minhhieuh@student.unimelb.edu.au; Patanamon
Thongtanunam, patanamon.t@unimelb.edu.au, University of Melbourne, Victoria, Australia.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2601.20810v1  [cs.SE]  28 Jan 2026

2 Seddik et al.
overestimate how effectively models use retrieved context under realistic retrieval distributions [ 11]. Recent RAG
pipelines for code have explored diverse retrieval strategies and knowledge bases [ 30], yet systematic evidence suggests
two recurring bottlenecks: accurately identifying helpful evidence for a given programming query, and operating under
limited context budgets where irrelevant retrieval can directly harm generation [35].
A central reason these bottlenecks persist is that actionable programming knowledge is heterogeneous. Developers
draw from code-centric artifacts (e.g., implementations, utilities, reusable patterns) as well as text-centric artifacts (e.g.,
tutorial narratives, API documentation, and Q&A explanations) [ 35]. This heterogeneity is not incidental; software
maintenance evidence indicates that code and natural-language documentation often co-evolve, implying that text
encodes complementary signals rather than redundant descriptions of code [ 24]. As a result, the representation used
for retrieval must cope with fundamentally different structures across artifacts.
This raises an under-specified design problem. For code generation, what should be retrieved, and in what representa-
tion, to preserve useful knowledge while minimizing noise? Empirical studies show that different retrieved information
types have qualitatively different values. Context-relevant information can help, while superficially similar snippets can
introduce noise and degrade results [ 8]. These findings motivate knowledge representations that are explicitly designed
to separate, organize, and filter heterogeneous knowledge rather than treating all data sources as flat nearest-neighbor
chunks [15].
Recent work supports the principle that structure can materially change retrieval behavior and downstream generation
[31]. For code-centric corpora, structure-aware chunking based on Abstract Syntax Trees (ASTs) yields more coherent
retrievable units and improves retrieval-augmented code generation relative to line-based heuristics [ 40]. For long or
redundant sources, tree-structured search improves over flat ranking [ 31]. In addition, work on granularity selection
shows that the size of retrieval units induces an inherent precision–recall trade-off that can substantially affect
downstream performance [ 42]. Collectively, these results motivate studying representations that (i) align retrieval units
with structure and (ii) provide explicit control over how context is assembled.
In this work, we presentProgramming Knowledge Graphs (PKGs), a novel representation for RAG, using two
stand-alone variants to isolate the role of knowledge modality and representation, being code-centric PKG and text-
centric PKG. Through PKG, we address the two challenges of accurately identifying and retrieving helpful documents
for code generation. Thecode-centric PKGis constructed by parsing source code into an AST-derived hierarchy and then
constructing a directed structure that mirrors syntactic containment. A function node expands to a sequence of block
nodes, each of which may expand to child block nodes (and so on), yielding a multi-level tree rooted at the function.
This construction aligns retrieval units with syntactic boundaries and preserves self-contained code regions, consistent
with evidence that AST-respecting segmentation produces more coherent retrieval units for retrieval-augmented code
generation [40].
A key advantage of this hierarchy is that it enables retrieval at multiple granularities. We instantiate two retrieval
settings:Func-PKG, which retrieves at the function level (coarser units), andBlock-PKG, which retrieves at the block
level (finer units). These settings operationalize an inherent granularity trade-off: coarser units tend to increase recall
but risk injecting irrelevant content, while finer units can improve precision but may omit surrounding context needed
for correct synthesis. Prior RAG work highlights that chunk granularity directly affects retrieval precision and recall and
that selecting an appropriate granularity can significantly change downstream performance [ 42]. We therefore evaluate
both granularity settings to quantify this precision–recall trade-off in code-centric PKG retrieval and to determine
when finer structure improves (or harms) end-to-end code generation.
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 3
Thetext-centricPKG targets tutorial/documentation-style knowledge, where usefulness often depends on internal
structure (titles, sections, explanations, examples) and on the ability to retrieve specific fields rather than entire passages
[15]. We adopt a structured JSON representation as an explicit schema over heterogeneous tutorial fields, then construct
a Directed Acyclic Graph (DAG) whose nodes correspond to JSON fields/sections and whose edges encode structural
relations (e.g., example-to-explanation linkage). The intent is twofold: (i) normalize diverse textual artifacts into a
parseable representation that supports field-level retrieval and assembly; and (ii) enable structure-aware traversal and
composition, following the general retrieval-and-structuring perspective that hierarchical organization can improve
controllability and reduce noise [15, 39].
Because structured expansion can still overgrow context windows, we incorporatetree pruningas an explicit control
mechanism before prompt construction. This design follows evidence that pruning irrelevant context can reduce
computational overhead and mitigate noise propagation in retrieval-augmented generation [ 5]. Finally, even with
improved context, different approaches excel at solving distinct types of problems [ 32]. We therefore complement
retrieval withpost-generation reranking(sample–then–select), where multiple candidate solutions from multiple methods
(e.g., RAG and non-RAG approaches) are generated and then ranked to select the best. Recent work on code generation
studies candidate reranking explicitly and reports improved success by selecting among diverse samples, supporting
reranking as a practical method applied to retrieval [ 2,29]. The reranker, therefore, can prioritize solutions generated
without relying on RAG-based content, reducing the influence of erroneous data when the retrieved content introduces
hallucinations into the output.
We compare NoRAG, sparse retrieval (e.g., BM25), dense retrieval, and PKG-based retrieval under both code-centric
and text-centric variants, and we analyze when and why performance changes via topic-level breakdowns, error-type
shifts, and cost trade-offs. We evaluate our method using HumanEval [ 3] and MBPP [ 1]. Our approach improves the
pass@1 accuracy across all baseline models on both the HumanEval [ 3] and MBPP [ 1] benchmarks by up to 20%
compared to the NoRAG method. In comparison to sparse and dense retrieval, our method demonstrates up to an 8%
increase in accuracy on HumanEval and up to a 34% improvement on MBPP. Error analysis on the MBPP dataset, which
contains more and complex problems, reveals that Assertion errors are reduced significantly, though Name errors
are introduced. Additionally, topic analysis on MBPP demonstrate the difficulty of solving some problems e.g., string
manipulation when using RAG based on PKG.
Our main contributions can be summarized as follows:
(1)A novel representation of code-centric and text-centric corpora in the form of a PKG for retrieval-
augmented code generation.We position PKGs as a structured knowledge representation for retrieving context
during code generation.
(2)Code-centric PKG construction with an explicit granularity comparison.We construct a code-centric
PKG by parsing code into an AST-derived hierarchy and materializing a containment structure (function Ñblock
Ñchild blockÑ¨¨¨ ). We evaluate retrieval at two granularities—Func-PKG(function-level) andBlock-PKG
(block-level).
(3)Text-centric PKG construction via structured JSON and DAG extraction.We construct a text-centric PKG
by converting tutorial/documentation-style text into structured JSON and extracting a DAG, enabling field-level
retrieval and structured context assembly.
(4)Structure-based context control through pruning.We incorporate pruning over retrieved graph structure
prior to prompt construction to control context growth and reduce inclusion of low-utility branches.
Manuscript submitted to ACM

4 Seddik et al.
(5)Post-generation reranking of solution candidates as an orthogonal selection mechanism.We apply
reranking over multiple generated candidate solutions to improve final correctness.
The rest of the paper is organized as follows. Section 2 discusses related works. Section 3 explains the methodology.
Sections 4 and 5 present the experimental setup and results, respectively. Section 6 provides a discussion of the findings,
error analysis, cost trade-off, and implications. Section 7 discusses the threats to the validity of this research. Finally,
Section 8 concludes the paper.
2 Related Work
In this section, we provide an overview of the related literature and position the differences of our work with previous
studies.
2.1 Retrieval Augmented Generation
RAG approaches are widely explored in general text generation [ 7,10,16,17]. RAG continues to evolve along three
mutually reinforcing directions: (i) structure- and graph-aware retrieval that exploits document- or event-level topology;
(ii) semantic pre-processing that produces more coherent retrieval units; and (iii) architectural optimizations that trade
off latency, memory and fidelity. Our approach is best characterized as structure-aware (graph-based) RAG, since it
explicitly represents code and tutorial content as a PKG and retrieves connected code blocks and path-value nodes from
that graph. It also incorporates semantic pre-processing (AST/JSON chunking) and an advanced re-ranking stage, but it
does not target architectural/decoding speedups.
Structure- and graph-aware retrieval.Recent work shows clear gains from retrieving connected, topology-aware
units rather than isolated passages. StructRAG builds document-level scholarly KGs and reports modest semantic-
accuracy gains alongside larger improvements in lexical diversity and readability (while flagging KG construction cost/s-
calability) [ 14]; KG2RAG augments seed retrieval with KG-guided expansion and KG-driven linearization (MST+DFS)
and can use either external KGs or LLM-extracted triplets, so its effectiveness depends on KG coverage and linking
quality. GRAG formalizestextual subgraphretrieval and fuses text+topology views for generation, showing strong
gains on graph-centric multi-hop tasks but limited cross-domain validation [ 12]. Domain variants (GeoGraphRAG,
EventRAG) demonstrate clear task-specific benefits (geospatial operators, temporal/event fusion) while illustrating
portability and modality limits [18, 37].
Semantic pre-processing / chunking.Methods that produce semantically coherent retrieval units improve down-
stream precision: SemRAG emphasizes semantic chunking and KG-style organization and documents the granularity
trade-off (precision vs. recall/coverage) [ 41]; KG2RAG and related pipelines supplement chunking with triplet extraction
or entity-linking to form richer contexts, but these steps add engineering cost and propagate KG/linking errors when
automated [46].
Architectural / decoding optimizations.A separate line targets latency and memory: REFRAG compresses and
selectively expands retrieved chunks during decoding to exploit block-sparse attention, reporting large improvements in
time-to-first-token ( «30ˆ) and competitive perplexity, though it requires additional encoder/projector/policy training
and careful validation for fidelity and compatibility with deployed LLMs [19].
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 5
2.2 RAG for Code Generation
RAG’s use in code-related tasks remains underexplored [ 34]. Previous studies, like [ 25], focused on smaller models like
CodeBERT and GraphCodeBERT for tasks like code summarization and generation, often fine-tuning the retriever. In
contrast, our approach applies RAG during inference without fine-tuning. DocPrompting [ 44] is designed to support
code generation across multiple programming languages and relies on retrieving information from a documentation
pool. It has also been tested on smaller-scale models. RepoCoder [ 38] is specifically designed for repository-level code
completion, leveraging an iterative retrieval-generation pipeline to refine the generated code. HippoRAG [ 9] is another
approach that introduces a neurobiologically inspired approach for integrating long-term memory into large language
models, primarily focusing on natural language tasks that require knowledge aggregation from multiple graph-based
sources. While both approaches leverage structured knowledge retrieval, our method is tailored to programming-related
tasks, emphasizing the seamless integration of documentation and code to improve the quality of generated outputs.
Recent literature highlights the efficacy of incorporating structural information into RAG frameworks for code. In the
context of code completion, RepoCoder [ 38] establishes a baseline for structure-aware RAG, utilizing a multi-granularity
framework that streamlines repository-level completion and achieves over 10% improvement over non-structural
baselines. Subsequent works have sought to capture more intricate dependencies through Knowledge Graphs (KGs). For
example, KGCompass [ 36] constructs a repository-aware KG to map issues to specific code entities, using entity path
tracing to narrow the search space for software repair. Similarly, Prometheus [ 4] unifies files, ASTs, and natural language
into a heterogeneous graph with typed edges, demonstrating effectiveness across seven programming languages. More
recent approaches integrate these structural representations with agentic workflows. CodexGraph [ 20] interfaces LLMs
with graph databases to support complex, multi-task operations, while LingmaAgent [ 22] employs a Monte Carlo tree
search strategy over a condensed KG to enable top-down repository exploration. Finally, RepoGraph [ 23] provides a
plug-in module, facilitating interactive navigation of repository-level structure in AI-driven engineering solutions.
Differences of our work with the existing literature.While [ 34] explores LLMs and Code-LLMs across data
sources, they note issues with retrievers and limited model context. Our method improves knowledge representation,
enabling more accurate retrieval and reducing hallucinations by prompting models with only relevant content. A
fundamental difference of PKG with [ 44] lies in the retrieval mechanism. Our method decomposes content into fine-
grained semantic nodes within the PKG, integrating both documentation and code. In contrast to RepoCoder [ 38],
our PKG-based approach is adaptable to a broader range of code-centric and text-centric datasets, making it more
versatile in various software development scenarios and can be a complement to RepoCoder. Our retrieval mechanism
prioritizes extracting the most relevant, fine-grained content from the knowledge base. Unlike RepoCoder’s iterative
retrieval strategy, which refines results over multiple retrieval cycles, our method ensures that only the most pertinent
information is retrieved from the outset, leading to more efficient and precise code generation.
Despite these advancements, the effective utilization of structural granularity (specifically the trade-off between
coarse-grained and fine-grained retrieval units) remains under-explored in current RAG frameworks. While approaches
like RepoCoder [ 38] and DocPrompting [ 44] leverage structure, they typically rely on iterative retrieval cycles or coarse
document pooling, which can introduce irrelevant context or obscure precise semantic boundaries. Furthermore, many
existing graph-based methods [ 20,22] focus on repository-wide navigation via complex agents rather than optimizing
the immediate representation of knowledge for generation. In contrast, our work introduces a novel inference-time
framework that creates distinct hierarchical representations for code (via AST-based containment) and documentation
(via JSON-based DAGs). Unlike prior methods that require fine-tuning or undefined granularity, our approach explicitly
Manuscript submitted to ACM

6 Seddik et al.
operationalizes the precision-recall trade-off through variable retrieval units (e.g.,Block-PKGvs.Func-PKG) and employs
structural pruning to actively minimize noise within the context window.
3 Methodology
Our method consists of three stages: (i) PKG construction, (ii) retrieval from the PKG to form an augmented prompt, and
(iii) candidate selection via reranking. Figure 1 summarizes the construction pipeline and Figure 2 summarizes retrieval.
3.1 Programming Knowledge Graph (PKG) Construction
Data sources.We build two PKGs from (1) a code-centric corpus derived from PythonAlpaca [ 26] and (2) a text-centric
corpus derived from a Python-focused subset of Tutorials [ 34] (Step 1 in Figure 1). Let Dcodedenote the code-centric
dataset andDtextdenote the text-centric dataset.
Graph schema.The PKG stores each artifact in multiple, consistent granularities so that retrieval can trade off context
breadth and specificity. Each code artifact is represented at (i) function granularity (the full implementation) and (ii)
block granularity (syntactic sub-structures). Each tutorial artifact is represented as a hierarchy, where leaves correspond
to primitive values addressed by their JSON paths.
We model the PKG as a typed, directed graph 𝐺“p𝑉,𝐸,𝜏,𝜙q where𝑉is a set of nodes, 𝐸Ď𝑉ˆ𝑉 is a set of
directed edges, 𝜏:𝑉ÑT assigns a node type, and 𝜙:𝑉ÑΣ˚maps each node to a textual payload (code or text)
that is embedded for retrieval (Section 3.1.3). Edges implement a refinement relation between granularities: a coarse
unit refines into finer units (function Ñblocks, JSON section Ñnested leaves). This refinement relation is acyclic
because syntactic nesting and JSON nesting are acyclic. A key property is that any retrieved fine-grained unit has a
well-defined projection back to its parent coarse unit, enabling the system to return either precise snippets or broader
context depending on the retrieval mode (Section 3.2).
For completeness, Appendix A enumerates the concrete node and edge types used in our implementation, including
identifier-to-content links for code and containment links for both code and JSON-derived structures.
3.1.1 Code-centric extraction: functions and block hierarchy.For each example in Dcode, we extract Python functions
from the code field(s) and parse them using a Python AST parser. For a function 𝐹, letBp𝐹q denote the set of syntactic
blocks within 𝐹(for example, if,for,while ,try,with , and function-local compound statements). Each block 𝑏PBp𝐹q
is represented by its textual payload𝜙p𝑏q.
For each function 𝐹, we create a subgraph 𝐺𝐹“p𝑉𝐹,𝐸𝐹,𝜏,𝜙q consisting of oneNamenode, oneImplnode, and one
Blocknode per extracted block:
𝑉𝐹“t𝑣𝐹
Name,𝑣𝐹
ImpluYt𝑣𝐹,𝑏
Block:𝑏PBp𝐹qu.(1)
Edges encode containment and nesting:
𝐸𝐹“tp𝑣𝐹
Name,𝑣𝐹
Implqu Y tp𝑣𝐹
Impl,𝑣𝐹,𝑏
Blockq:𝑏PBp𝐹qu
Y tp𝑣𝐹,𝑏𝑝
Block,𝑣𝐹,𝑏𝑐
Blockq:𝑏𝑐is a direct syntactic child of𝑏 𝑝u.(2)
Theparentedges form a directed acyclic graph (DAG) because syntactic nesting is acyclic. This step is shown in Step 3
of Figure 1.
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 7
3.1.2 Text-centric extraction: JSON path-value graph.For each document 𝑥PDtext, we produce a JSON representation
𝐽p𝑥q using a constrained prompting procedure and validate the output by JSON parsing. We model each JSON as a
rooted tree whose nodes are keys and whose edges follow JSON nesting.
A JSON value is either aprimitive(string, number, boolean), anobject(a finite map from keys to values), or anarray
(an ordered list of values). We view an object 𝐽as a rooted tree where: (i) therootconsists of the top-level keys in
the JSON file, and (ii) a key has children when its associated value is itself an object (or an array whose elements are
objects). Thus, nested JSON objects induce parent-child relationships between keys, and leaves correspond to keys
whose values are primitives.
For any key reachable in this tree, itspathis the sequence of keys from a top-level key to that key. For a key 𝑘,
we write its path as 𝑝p𝑘q“p𝑘 0,𝑘1,...,𝑘ℓq, where𝑘0is a top-level key and 𝑘ℓ“𝑘. We use a string form Joinp𝑝q (for
example,k0/k1/.../kℓ) as a unique identifier.
We createPathValuenodes only forleafpaths whose associated value is primitive. Let Leavesp𝐽q denote the set of
leaf keys in 𝐽, and let𝑣𝑎𝑙p𝑘q be the value associated with key 𝑘. For each𝑘PLeavesp𝐽q , we create a node 𝑣𝑘of type
PathValuewith payload:
𝜙p𝑣𝑘q“SerializepJoinp𝑝p𝑘qq,𝑣𝑎𝑙p𝑘q q,(3)
whereSerializeproduces a short textual form such aspath = ... ; value = ....
To preserve hierarchy, we materialize internal path nodes and add directed edges from each parent path to its child
path. Since JSON nesting is tree-structured, the resulting directed graph is acyclic. This step is illustrated in Step 3 of
Figure 1.
3.1.3 Embedding and storage.We embed each node payload for semantic retrieval. Let E:Σ˚ÑR𝑑be an embedding
model. Each node𝑣P𝑉stores an embedding vector:
z𝑣“Ep𝜙p𝑣qq.(4)
As shown in step 4 of Figure 1, we store 𝐺andtz𝑣u𝑣P𝑉in a graph database with a vector index to support approximate
nearest-neighbor search.
3.2 Retrieval from PKG
As in step 1 in Figure 2, given a user query 𝑞PΣ˚, we compute its embeddingz 𝑞“Ep𝑞q and retrieve relevant nodes
under three retrieval modes: block-wise (results will be labeledBlock-PKG), function-wise (results will be labeled
Func-PKG), and tutorial path-value (results will be labeledJSON-PKG). Let 𝑉𝑡“t𝑣P𝑉 :𝜏p𝑣q“𝑡u be the set of nodes
of type𝑡. For a retrieval mode 𝑡PtBlock,Impl,PathValueu , we retrieve the top node based on the following objective
(Steps 2 and 3 in Figure 2):
𝑣˚p𝑞q“arg max
𝑣P𝑉𝑡Simp𝑞,𝑣q.(5)
whereSimp¨,¨qrefers to the cosine similarity. In practice, we compute𝑣˚using a vector index.
Pruning of block subtrees.Given a retrieved code node 𝑣˚(either aImplnode for function-wise retrieval or aBlock
node for block-wise retrieval), we optionally remove irrelevant branches from its associated syntactic containment
DAG to better align the returned context with the query (Step 3 in Figure 2).
Let𝐺𝑣˚“p𝑉𝑣˚,𝐸𝑣˚qdenote the DAG induced by 𝑣˚and its descendantBlocknodes underparentedges. We
define abranchas a child-subtree rooted at a direct child of 𝑣˚in𝐺𝑣˚. Let Childp𝑣˚qbe the set of direct children of 𝑣˚
Manuscript submitted to ACM

8 Seddik et al.
in this DAG. For each 𝑢PChildp𝑣˚q, let𝑇𝑢p𝑣˚qbe the subtree rooted at 𝑢(including all descendants of 𝑢). We define
the pruned graph obtained by removing that branch as
𝐺´𝑢
𝑣˚“𝐺𝑣˚z𝑇𝑢p𝑣˚q,(6)
where𝐺z𝑇removes the nodes in𝑇and all incident edges from𝐺.
To select which branch to remove, we score each pruned graph by its cosine similarity to the query embedding.
Concretely, we serialize the remaining graph content into text 𝜙p𝐺´𝑢
𝑣˚q(by concatenating the code payloads of its
remaining nodes in a fixed order), embed it using the same embedder E, and compute its cosine similarity. The selected
pruned graph is
𝐺pruned“arg max
𝑢PChildp𝑣˚qSim´
𝑞,𝐺´𝑢
𝑣˚¯
.(7)
The retrieved context returned to the generator is then the serialized content𝜙p𝐺 prunedq.
Query augmentation.Let 𝑡PtBlock,Impl,PathValueu denote the active retrieval mode and let 𝑣˚p𝑞qbe the single
retrieved node from Equation 5. We convert the retrieval result into a textual context block and append it to the original
query (Step 4 in Figure 2).
For tutorial retrieval (𝑡“PathValue), the context is simply the payload of the retrieved node:
Cp𝑞q “𝜙`
𝑣˚p𝑞q˘
.(8)
For code retrieval (𝑡PtBlock,Implu), the contextCp𝑞qis giving by:
Cp𝑞q “𝜙p𝐺 prunedq(9)
where𝐺 pruned is defined by the pruning step applied to the induced DAG of𝑣˚p𝑞q.
Finally, we construct the augmented prompt by concatenating the user query with the retrieved context under a
deterministic template:
𝑞aug“Formatp𝑞,Cp𝑞qq,(10)
where Format inserts explicit delimiters and a brief instruction indicating that the model may use the provided context
as reference. The augmented prompt𝑞 augis then passed to the code generation model.
3.3 Solution Reranking
As shown by recent research [ 32], retrieval can both help and harm. We therefore select among multiple candidate
solutions generated via different methods.
Let𝑀be the code generation model. For each task, we generate one candidate under each condition (NoRAG, BM25,
PKG variants) using the same decoding settings. Let𝐶“t𝑐 1,...,𝑐𝑁ube the union of all candidates.
We apply two deterministic filters:
‚Syntactic validity:𝐴p𝑐q“1if𝑐parses as Python, else0.
‚Runtime sanity:𝑅p𝑐q“1if𝑐can be executed in a sandbox without raising exceptions, else0.
Let𝐶𝐴“t𝑐P𝐶:𝐴p𝑐q“1uand𝐶 𝑅“t𝑐P𝐶𝐴:𝑅p𝑐q“1u.
Among remaining candidates, we select using a similarity score between the query and candidate code:
𝑐˚“arg max
𝑐P𝐶𝑅Simp𝑞,𝑐q,(11)
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 9
where Simp¨,¨q is the cosine similarity, and the candidate embedding is computed as Ep𝑐q . We report oracle selection
separately by choosing the first candidate that passes benchmark tests, which provides an upper bound on attainable
gains from candidate diversity. Note that the reranker in Equation 11 is intentionally simple and model-agnostic. Its
role is to reduce retrieval-induced regressions by selecting candidates that remain aligned with the user query. We
evaluate this reranker against both non-reranked settings and an oracle upper bound in Section 5.
4 Experimental Setup
Retrieval Approaches:We utilized two retrieval methods based on a comparative analysis of various code retrieval
models, as described by [ 34]. For dense retrieval, we selected the Voyage-Code-2 model, recognized as one of the
top-performing dense retrievers for code. Embeddings were obtained through API calls to this model. For sparse
retrieval, we employed the BM25 algorithm, implemented using the 𝑟𝑎𝑛𝑘_𝑏𝑚 25Python library1, which exhibited the
strongest performance among sparse retrieval techniques.
Dataset and PKG Generation:We used the PythonAlpaca dataset [ 26] as a code-centric data source, which contains
143,000 general Python question-answer pairs. After preprocessing, we extracted 115,000 Python functions from the
dataset. This extraction enabled us to construct a PKG comprising 425,058 nodes and 434,518 relations.
We also performed experiments with the Tutorials dataset [ 34] as a text-centric data source, which contains 76,600
programming tutorial content. After converting them into json representations, pkg contains 288,583 path-value nodes
and 287,936 relations. The graphs were generated using Neo4J version 5.20.0, optimized for handling large-scale graphs
and supporting semantic search over the stored content.
Code Generation Models:We conducted our experiments on four well-known Code-LLMs: CodeLlama-7B
[28],CodeLlama-13B [ 28], StarCoder2-7B [ 21], and DeepSeek-Coder-7B [ 45]. In addition, we tested Llama3.1-8B [ 6],
a general-purpose LLM that has demonstrated strong performance on code generation tasks. All experiments were
conducted using a single A100 GPU.
Evaluation Metric:To evaluate the accuracy of generated code, we used the pass@1 metric [ 3]. Due to resource
constraints, we adopted a greedy decoding approach for the pass@1 evaluation, generating a single solution with a
temperature setting of 𝑡“0and a token limit of 512 p𝑚𝑎𝑥_𝑛𝑒𝑤_𝑡𝑜𝑘𝑒𝑛𝑠“ 512q. Due to the deterministic nature of the
scores, the results should be compared directly without requiring statistical tests.
Benchmarks:In this study, we aim to evaluate the general Python programming skills of both Code-LLMs and
LLMs. To achieve this, we have selected the HumanEval dataset [ 3] and the MBPP benchmark [ 1]. These datasets are
well-established in the literature and are widely used to assess both problem-solving and reasoning capabilities in
Python programming.
5 Results
In this section we carry out experiments to answer the following research questions.
RQ1: Does code-centric PKG improve code generation?
In this research question, we aim to explore the potential of leveraging graph-based retrieval-augmented methods on
code-centric data source to improve code generation task. Specifically, we examine how relevant code retrieved from a
PKG built on PythonAlpaca [26] can improve the performance of LLMs and Code-LLMs in generating accurate code.
1https://pypi.org/project/rank-bm25/
Manuscript submitted to ACM

10 Seddik et al.
Fig. 1. The overview of generating PKGManuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 11
Fig. 2. Overview of the retrieval process from PKG
Table 1. Performance of code-centric retrieval-augmented code generation on HumanEval, reported as pass@1. Red cells indicate
performance below NoRAG and green cells show scores above NoRAG, with color intensity reflecting significance. “Ideal Reranker”
serves as an oracle upper bound for our proposed re-ranking stage.
Model None BM25 VoyageEmb Func-BM25 Func-PKG Block-PKG Reranked Ideal Reranker
CodeLlama-7B 33% 21% 42% 33% 38% 40% 46% 56%
CodeLlama-13B 42% 34% 45% 43% 46% 47% 51% 63%
Llama3.1-8B 55% 34% 50% 54% 55% 61% 66% 75%
StarCoder2-7B 45% 41% 53% 57% 56% 59% 63% 72%
DeepSeek-Coder-7B 70% 44% 60% 62% 69% 68% 73% 83%
Avg. 49.0% 34.8% 50.0% 49.8% 52.8% 55.0% 59.8% 69.8%
Table 2. Performance of code-centric retrieval-augmented code generation on MBPP, reported as pass@1. Rows are grouped into
open-source and closed-source models. Red cells indicate accuracy below NoRAG, green cells indicate accuracy above, and color
intensity reflects significance. “Ideal Re-ranker” serves as the upper bound for the proposed re-ranker method.
Model None BM25 VoyageEmb Func-BM25 Func-PKG Block-PKG Reranked Ideal Reranker
CodeLlama-7B 38% 27% 32% 27% 44% 46% 58% 60%
CodeLlama-13B 44% 36% 26% 36% 40% 48% 55% 57%
Llama3.1-8B 43% 38% 41% 41% 46% 49% 63% 66%
StarCoder2-7B 46% 25% 17% 31% 29% 51% 62% 64%
DeepSeek-Coder-7B 56% 50% 45% 47% 50% 47% 65% 68%
Avg. 45.4% 35.2% 32.2% 36.4% 41.8% 48.2% 60.6% 63.0%
Our method retrieves relevant code from the PKG and integrates it into the generation process (See section B.5
in Appendix). We compare this approach to several baselines, detailed in Tables 1 and 2, for HumanEval and MBPP
benchmarks. The baselines include: 1) None: No retrieval-augmented generation, 2) BM25: Applied to the entire dataset
without pre-processing, 3) VoyageEmb: Embeddings from question-answer pairs for retrieval, 4) Func-BM25: BM25
applied to function-extracted data, 5) Func-PKG: Semantic search over function-related nodes, 6) Block-PKG: Granular
semantic search over code blocks for deeper context, 7) Reranked: Reranking of candidates from the retrieval methods
1-6, and 8) Ideal Reranker: Selecting the first solution that passes. This serves as an upper bound oracle simulating
perfect reranking conditions, as the correct solution is shown to the model in the retreival process.
Tables 1 and 2 show that the effect of retrieval augmentation depends critically on (i) the retrieval unit and (ii)
selection strategy. Naïve row-level retrieval is unstable: BM25 applied over unprocessed rows degrades performance for
all models on both HumanEval and MBPP. For example, average pass@1 drops by 14.2 points on HumanEval and 10.2
Manuscript submitted to ACM

12 Seddik et al.
points on MBPP relative to no retrieval. Dense retrieval over Q&A pairs (“VoyageEmb”) is similarly inconsistent; on
MBPP, dense retrieval underperforms the no-retrieval baseline by 13.2 points on average. These results indicate that
retrieval is not inherently beneficial; when the retrieved context is noisy or weakly aligned to the target task, it can
distract the model and reduce correctness.
In contrast, code-structured retrieval units improve robustness. When retrieval operates over extracted functions
(Func-PKG) or smaller code blocks (Block-PKG), performance improves on average. Block-level retrieval yields the best
average performance among non-reranked methods, consistent with the hypothesis that smaller units reduce irrelevant
context and increase the density of actionable signals. However, block-level retrieval is not uniformly beneficial:
DeepSeek-Coder-7B is negatively impacted by Block-PKG on both benchmarks. This aligns with observations from a
related study by [ 34], where it exhibited similar behavior. Based on these findings, we hypothesize that DeepSeek-Coder
may not be effectively utilizing additional contextual information during training.
The largest improvements arise when retrieval is combined with multi-candidate selection. The reranking stage
improves pass@1 over the best non-reranked method by approximately 4 points on HumanEval and 12 points on
MBPP across open models, suggesting that retrieval should be treated as a mechanism for generating diverse candidate
solutions, after which selection becomes the primary bottleneck. The oracle (“Ideal Reranker”) indicates additional
headroom, especially on HumanEval ( «10-point average gap), implying that the candidate sets frequently include
correct solutions that are not selected by the current reranker.
Answer to RQ1:Code-centric PKG improves code generation when retrieval units are code-structured and
when candidate selection is used; naïve retrieval can harm performance. Smaller retrieval units yield the best
average performance.
RQ2: Does text-centric PKG improve code generation?In this research question, we investigate the potential of
leveraging text-centric data to improve code generation. To be more specific, we built a PKG on the Tutorials dataset
[34], which was processed using the Gemma2-9B model [ 33], producing hierarchical JSON representations, enabling us
to generate a PKG as explained in Section 3.1. Our baselines include: 1) None: No retrieval-augmented generation, 2)
BM25: Applied to the entire dataset without pre-processing, 3) VoyageEmb: Embeddings from question-answer pairs
for retrieval, and 4) JSON-PKG: Our proposed PKG based on text-centric data.
Tables 3 and 4 show the evaluation results. Relative to no retrieval, JSON-PKG yields heterogeneous effects on
HumanEval. It improves pass@1 for CodeLlama-7B (+3 points), Llama3.1-8B (+8 points), and StarCoder2-7B (+16 points),
but degrades performance for CodeLlama-13B (-1 point) and DeepSeek-Coder-7B (-5 points). On MBPP, JSON-PKG
yields consistent but modest improvements across all evaluated models, ranging from +1 to +8 points, with the largest
gain for Llama3.1-8B (+8 points).
Baselines that retrieve from unstructured tutorial rows remain unstable. BM25 reduces the performance on HumanEval
and slightly on MBPP, while dense retrieval (VoyageEmb) improves the performance on HumanEval slightly but on
average reduces the performance of the models on MBPP. Overall, these results indicate that retrieval effectiveness
depends not only on the retriever, but also on the structure and relevance of the indexed artifacts.
Comparing JSON-PKG (text-centric) against Block-PKG (code-centric) from Tables 1 and 2, text-centric retrieval is,
on average, slightly weaker on HumanEval but competitive on MBPP. On HumanEval, JSON-PKG (pass@1=53.2%) trails
Block-PKG (pass@1=55.0%) by 1.8 points on average, but it outperforms Block-PKG for Llama3.1-8B and StarCoder2-7B.
On MBPP, JSON-PKG is approximately 0.6 points higher on average, largely because Block-PKG substantially degrades
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 13
DeepSeek-Coder-7B on MBPP while JSON-PKG yields a small improvement for the same model. This suggests that
tutorial-based retrieval can be beneficial when code-based retrieval surfaces misleading context.
The results support a precision-relevance tradeoff for text-centric retrieval. Tutorial content can provide high-level
scaffolding such as algorithm outlines, invariants, and edge-case reasoning that improves generation, which is consistent
with the large gains observed for StarCoder2-7B and Llama3.1-8B on HumanEval. However, unstructured tutorial text can
also introduce noise. Retrieved explanations may be topically related but not operationally aligned with the benchmark
specification, and this can bias the model toward an incorrect implementation pattern or away from benchmark-
specific constraints. This is consistent with the degradations observed for CodeLlama-13B and DeepSeek-Coder-7B on
HumanEval.
Taken together, the evidence indicates that text-centric PKG retrieval can improve code generation, but its benefits
are model- and benchmark-dependent. We hypothesize that general LLMs benefit more from text-centric data as
supplementary context than CLLMs. Additionally, while we tested retrieval methods like BM25 and VoyageEmb,
JSON-PKG outperformed them in Pass@1 accuracy across both benchmarks. However, when comparing Block-PKG
with JSON-PKG, code-centric data still offers greater benefits for code generation tasks, highlighting that code-focused
data remains more effective for these specific tasks.
Answer to RQ2.Text-centric PKG can improve code generation, but the effect is not uniform across models.
While general LLMs benefit more from text-centric data, Code-LLMs benefit more from code-centric data.
Table 3. The performance of PKG on HumanEval, using tutorials data, is reported as pass@1. Red cells indicate accuracy below
NoRAG, green cells above, with color intensity reflecting significance.
Model None BM25 VoyageEmb JSON-PKG
CodeLlama-7B 33% 28% 35% 36%
CodeLlama-13B 42% 29% 43% 41%
Llama3.1-8B 55% 47% 58% 63%
StarCoder2-7B 45% 40% 59% 61%
DeepSeek-Coder-7B 70% 60% 59% 65%
Avg. 49.0% 40.8% 50.8% 53.2%
Table 4. PKG performance on MBPP using tutorial data, measured by pass@1. Red indicates accuracy below NoRAG, green above,
with shading intensity showing significance.
Model None BM25 VoyageEmb JSON-PKG
CodeLlama-7B 38% 29% 30% 41%
CodeLlama-13B 44% 35% 36% 45%
Llama3.1-8B 43% 48% 51% 51%
StarCoder2-7B 46% 49% 38% 50%
DeepSeek-Coder-7B 56% 58% 51% 57%
Avg. 45.4% 43.8% 41.2% 48.8%
RQ3: Which knowledge representation method is most effective in optimizing context retrieval for code
generation tasks?
Manuscript submitted to ACM

14 Seddik et al.
To isolate the impact of representation, we compare three families of retrieval units: (i) row/Q&A representations,
(ii) function-level representations, and (iii) block-level representations, as shown in Tables 1 and 2. Across open models,
row/Q&A representations are the least reliable: both sparse and dense retrievers over these units often underperform
no retrieval, indicating that “retrieval quality” is not just a function of the retriever algorithm but also of the granularity
and cleanliness of the indexed artifacts.
Function-level representations mitigate some of this instability but do not dominate uniformly, implying that
simply extracting functions is insufficient when retrieved functions are long or only partially relevant. Block-level
representations generally provide stronger non-reranked performance and larger gains over function-level retrieval
on MBPP, but can still fail for certain models. This failure mode is consistent with a precision–recall tradeoff: higher
granularity increases the probability of retrieving highly relevant fragments, but also increases exposure to superficially
similar fragments that trigger incorrect solution templates.
Finally, reranking consistently dominates representation-only improvements, indicating that the most effective
system is not “the best single context, ” but rather a pipeline that uses representation to generate candidate diversity and
uses selection to control retrieval-induced error.
Answer to RQ3:Code-structured representations (functions/blocks) are generally more effective than row/Q&A,
but the decisive factor in these results is robust selection (reranking), not representation alone.
6 Discussions
This section synthesizes the empirical findings into actionable insights. We first explore the impact of PKG on closed-
source models (Section 6.1). Rather than focusing only on aggregate pass@1, we examine how retrieval changes behavior
across problem topics (Section 6.2) and error modes (Section 6.3), and what these changes imply for designing reliable
retrieval-augmented code generation systems. We then contextualize accuracy gains with preprocessing and storage
costs (Section 6.4). Section 6.5 discusses implications for researchers and practitioners regarding when PKG-based
retrieval is beneficial, when it can be harmful, and what system components most strongly determine robustness. Finally,
Section 6.6 discusses open gaps and limitations.
6.1 Extending to Closed-source Models
We explore the effect of PKG on closed-source models. As shown in Tables 5 and 6, for commercial models, the marginal
benefit of retrieval is substantially smaller and sometimes negative. Given their high no-retrieval baselines (e.g., 96.3
pass@1 on HumanEval for Claude Sonnet 4), retrieval likely provides limited novel information while still incurring
risks from irrelevant or contradictory context. This suggests that retrieval policies may need to be model-adaptive, with
conservative gating for powerful, closed-source models and more diverse candidates for smaller, open-source models.
Note that still even closed-source models can benefit from the Func-PKG and reranker proposed by our work. The most
significant improvements are seen for GPT-4o and GPT-4o-mini, where the results are improved by 2 and 2.8 pass@1
scores, respectively, when the reranking is applied.
6.2 Problem Topics Benefiting from RAG
We analyze topic-level performance on MBPP using DeepSeek-Coder-7B. We map 134 MBPP unique categories into 10
broader topics and report pass@1 per topic for four settings: NoRAG, BM25, PKG, and Re-ranked. Figure 3 shows three
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 15
Fig. 3. Comparison of different approaches across 10 topics using the MBPP benchmark on StarCoder2-7B
Table 5. Performance of closed-source models with code-centric retrieval-augmented code generation on HumanEval, reported as
pass@1. Red cells indicate performance below NoRAG and green cells show scores above NoRAG, with color intensity reflecting
significance. “Ideal Reranker” serves as an oracle upper bound for our proposed re-ranking stage.
Model None BM25 VoyageEmb Func-BM25 Func-PKG Block-PKG Reranked Ideal Reranker
Claude-3-Haiku 74.4% 72.0% 73.8% N/A 71.3% 67.7% 73.8% 87.8%
Claude-Sonnet-4 96.3% 95.1% 95.1% N/A 95.1% 94.5% 95.7% 100.0%
GPT-4o 88.4% 89.0% 89.6%N/A 89.0% 89.6% 89.6% 95.7%
GPT-4o-mini 87.2% 82.9% 85.4% N/A 86.0% 83.5% 87.8% 95.1%
Avg. 86.6% 84.8% 86.0% N/A 85.4% 83.8% 86.7% 94.7%
consistent patterns. First, PKG outperforms BM25 across all topics. The improvement is largest in topics where lexical
matching is least reliable, includingMathematics and Number TheoryandOptimization Techniques, where BM25 is low
Manuscript submitted to ACM

16 Seddik et al.
Table 6. Performance of closed-source models with code-centric retrieval-augmented code generation on MBPP, reported as pass@1.
Red cells indicate accuracy below NoRAG, green cells indicate accuracy above, and color intensity reflects significance. “Ideal
Re-ranker” serves as the upper bound for the proposed re-ranker method.
Model None BM25 VoyageEmb Func-BM25 Func-PKG Block-PKG Reranked Ideal Reranker
Claude-3-Haiku 67.2% 67.6% 66.0% N/A 67.6% 67.0% 67.6% 73.0%
Claude-Sonnet-4 83.6% 82.4% 82.2% N/A 83.2% 83.0% 82.2% 87.4%
GPT-4o 81.4% 80.6% 83.0% N/A 81.6% 81.4% 83.4% 86.0%
GPT-4o-mini 74.2% 74.0% 73.4% N/A 74.6% 73.8% 77.2% 79.2%
Avg. 76.6% 76.2% 76.2% N/A 76.8% 76.3% 77.6% 81.4%
while PKG is substantially higher. This indicates that topic-relevant retrieval requires more than surface-form overlap,
and that PKG retrieves context that better matches the semantic intent of the tasks.
Second, relative to NoRAG, PKG improves pass@1 in 7 out of 10 topics. Gains are most visible for topics such as
Mathematics and Number Theory,Optimization Techniques,Geometry and Trigonometry, andSorting and Searching. In
contrast, PKG underperforms NoRAG forString Manipulation and Text ProcessingandData Structures. For these two
topics, results indicate that retrieved context can be net distracting even when it is broadly relevant at the topic level.
Third, re-ranking changes the topic profile. Re-ranked achieves the highest pass@1 in most topics, with prominent
gains inString Manipulation and Text Processing,Data Structures,Conditional and Loop Structures, andData Conversion
and Representation. However, re-ranking is not uniformly beneficial. InMathematics and Number Theory,Optimization
Techniques, andAlgorithms, the Re-ranked curve falls below PKG, suggesting that the re-ranker can select suboptimal
candidates in these domains even when retrieval produces useful options.
The topic-level differences suggest that retrieval provides value through two mechanisms, and that their relative
importance varies by topic. In topics such as sorting, geometry, and control-flow patterns, correct solutions often reuse
recognizable program schemas. Retrieval can supply concrete implementation details (for example, standard loops,
boundary handling, or library usage), and re-ranking can exploit candidate diversity to select a correct instantiation.
This is consistent with the large separation between Re-ranked and NoRAG in these topics.
In contrast, string processing and data-structure tasks are often sensitive to small semantic details, including off-by-
one behavior, formatting constraints, and invariants that are easy to violate by adapting a superficially similar snippet.
If retrieved context encourages a near-match template, the model can become less faithful to the exact specification,
which is consistent with PKG underperforming NoRAG in these two topics. The fact that re-ranking strongly improves
these topics suggests that retrieval still produces some correct candidates, but selection becomes the bottleneck, the
system benefits when it can choose among multiple solutions and avoid the misleading template.
The re-ranker underperformance in mathematics, optimization, and algorithms relative to PKG indicates a different
failure mode. A plausible explanation is that candidate sets in these topics contain many solutions that are syntactically
well-formed and structurally plausible but mathematically incorrect, and the re-ranker scores correlate with surface
plausibility rather than semantic validity. This motivates adding topic-aware signals, such as lightweight symbolic checks,
constraint extraction, or re-ranker features that better reflect semantic correctness for mathematically constrained
tasks.
Figure 4 presents the pass@1 accuracy for each method—NoRAG, PKG, BM25, and the re-ranked approach—across
various programming topics. Similar to the performance observed with the StarCoder2-7B model, the re-ranker struggles
to correctly prioritize solutions in the ’Optimization Techniques,’ ’Mathematics,’ and ’Algorithm’ categories. However,
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 17
in other topic areas, the re-ranker demonstrates superior performance compared to the other methods. Notably, for
this model, PKG achieves higher accuracy across most topics, with the exception of ’String Manipulation’ and ’Data
Structures,’ where it is outperformed by other approaches.
Fig. 4. Comparison of different approaches across 10 topics using the MBPP benchmark on CodeLlama-7B.
Figure 5 illustrates the Pass@1 accuracy for each evaluation method: NoRAG, PKG, BM25, and the re-ranked approach,
across a range of programming topics. The performance trends observed with the DeepSeek-Coder-7B model are echoed
here. Specifically, the re-ranking method shows difficulty in accurately prioritizing solutions within the categories of
’Optimization Techniques,’ ’Mathematics,’ and ’Algorithms.’ Despite these challenges, the re-ranked approach excels in
other topic areas, demonstrating superior performance compared to the other methods.
Notably, the PKG method achieves higher accuracy across most topics evaluated. However, it does face competition
in the ’String Manipulation’ and ’Data Structures’ categories, where it is outperformed by NoRAG approach. We have
observed the same behaviour for the previous models.
Manuscript submitted to ACM

18 Seddik et al.
Fig. 5. Comparison of different approaches across 10 topics using the MBPP benchmark on DeepSeek-Coder-7B
In summary, on MBPP, PKG improves performance over BM25 for all topics and improves over NoRAG for most
topics, but it is not uniformly beneficial. Topics that rely on reusable program schemas benefit strongly from retrieval
and re-ranking, while topics that are sensitive to fine-grained semantic constraints can experience retrieval-induced
regressions unless the system includes robust candidate selection.
6.3 Error Types Reduced or Introduced by Applying RAG
Table 7 summarizes exception types observed on MBPP for three models with and without Block-PKG. MBPP is chosen
as it is a more complex dataset compared to HumanEval. The raw counts show that retrieval changes not only whether
a solution is correct, but also how it fails when it is incorrect. For StarCoder-7B, Block-PKG reduces AssertionErrors
from 198 to 147 ( ´51) and reduces SyntaxErrors from 2 to 0 ( ´2), but increases NameErrors from 51 to 64 ( `13) and
introduces IndentationErrors from 0 to 18 ( `18). For CodeLlama-7B, Block-PKG reduces NameErrors from 138 to 65
(´73) and reduces AssertionErrors from 180 to 162 ( ´18), but increases TypeErrors from 28 to 37 ( `9) and introduces
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 19
Table 7. Error Analysis on MBPP for different settings
Error Type StarCoder-7BStarCoder-7B
+ Block-PKGCodeLlama-7BCodeLlama-7B
+ Block-PKGDeepSeekCoder-7BDeepSeekCoder-7B
+ Block-PKG
# of AssertionErrors 198 147 180 162 135 146
# of NameErrors 51 64 138 65 64 78
# of TypeErrors 11 8 28 37 4 16
# of SyntaxErrors 2 0 0 1 0 0
# of IndentationErrors 0 18 0 0 0 0
# of Others 3 7 11 4 5 9
SyntaxErrors from 0 to 1 ( `1). For DeepSeekCoder-7B, Block-PKG increases AssertionErrors from 135 to 146 ( `11),
increases NameErrors from 64 to 78 ( `14), and increases TypeErrors from 4 to 16 ( `12). The total number of recorded
errors (summed across listed exception types) decreases for StarCoder-7B (265 to 244, ´21) and CodeLlama-7B (357 to
269,´88), but increases for DeepSeekCoder-7B (208 to 249,`41).
Two consistent mechanisms explain the observed shifts. First, retrieval can reduce semantic mismatch failures, which
often manifest as AssertionErrors and NameErrors. When retrieved context provides relevant function signatures,
variable naming conventions, or common implementation patterns, models may produce code that better matches
the target specification and avoids undefined identifiers. This is consistent with the large NameError reduction for
CodeLlama-7B and the AssertionError reduction for StarCoder-7B.
Second, retrieval can introduce formatting and compatibility failures. The appearance of IndentationErrors for
StarCoder-7B suggests that injected code blocks, or the model’s copying of retrieved code, can disrupt indentation
structure. The increase in TypeErrors for CodeLlama-7B and DeepSeekCoder-7B is consistent with models adopting
retrieved templates that assume different input types or data conventions than the benchmark requires.
The DeepSeekCoder-7B pattern is qualitatively different: error counts increase across major categories. This should
not be attributed to an inability to use context without additional evidence. A simpler explanation is that retrieval reduces
correctness for this model under Block-PKG (as also suggested by the aggregate MBPP results), thereby increasing the
number of failing executions and the opportunity for errors to occur.
On MBPP, retrieval can reduce failures consistent with semantic mismatch (notably AssertionErrors and NameErrors
for some models), but it can introduce new failure modes, particularly formatting errors (IndentationErrors) and type
mismatches (TypeErrors). The direction and magnitude of these effects are model-dependent.
As shown in Figure 6, different approaches excel at solving distinct types of problems, demonstrating the need for a
re-ranker. When the initial retrieved content introduces hallucinations into the output, the re-ranker can prioritize
solutions generated without relying on RAG-based content, reducing the influence of erroneous data.
6.4 Cost Trade-off
We compare PKG’s generation cost with a standard RAG setup using VoyageAI and BM25 retrieval on the PythonAlpaca
dataset [ 26] in Table 8. PKG and VoyageAI share identical encoding times since both use the same embedding model
(Voyage-Code2) and dataset. Unlike embedding-based RAG methods, PKG requires an additional hour of processing
but achieves a 9.4% higher accuracy on average. Neo4j’s semantic vector indexing enables efficient graph updates
with logarithmic complexity: 𝑂plog𝑁q for nodes and 𝑂plog𝑀q for relationships. Retrieval involves comparing query
embeddings with all nodes, resulting in 𝑂p𝑁¨𝑑q complexity, where 𝑑is the embedding dimension. In practice, queries
took about 3 seconds each.
Manuscript submitted to ACM

20 Seddik et al.
Fig. 6. This figure illustrates the impact of three approaches – our technique, Programming Knowledge Graph (Block-PKG), Func-
BM25, and NoRAG – on solving HumanEval problems using the DeepSeek-Coder-7B (left) and CodeLlama-7B (right) models.
Considering CodeLlama-7B, it shows that 16 problems were uniquely solved by the PKG, 12 problems by Func-BM25, and 27 problems
were solved by all three approaches.
Table 8. Time and storage usage for creating RAG data sources on PythonAlpaca [ 26]. Time is in minutes, and storage (last row) is in
megabytes (MB).
Step PKG VoyageAI BM25
Python Code Extraction 3 - -
Block Extraction 25 - -
Encoding 241 240 44
Neo4j Graph Generation 33 - -
Overall Time 301 241 44
Storage Usage (MB) 12,530 8,440 315
Although PKG requires more time for processing, computational inference costs is reduced, while performance is
increased. Table 9 presents the average token length of the additional context across different retrieval settings. As
shown in the table, Block-PKG has the lowest number of tokens compared to others. One of the key advantages of
retrieving finer-grained contextual information is the ability to provide the model with a reduced token budget, thereby
lowering computational inference costs and minimizing monetary expenses, particularly for proprietary models.
6.5 Implications for Researchers and Practitioners
Our results suggest that retrieval-augmented code generation is best viewed as a coupled system with three interacting
design choices: (i) theunit of retrieval(rows versus functions versus blocks versus path-values), (ii) theselection
mechanism(single-shot augmentation versus candidate selection), and (iii) thebudgeting policy(how much retrieval
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 21
Table 9. The average number of additional tokens in different approaches for the HumanEval benchmark.
RAG Method Avg. Tokens (CodeLlama) Avg. Tokens (DeepSeek)
Block-PKG 87 84
Func-PKG 188 182
BM-25 226 218
Voyage 349 339
and how many candidates are permitted). Below we summarize implications that follow directly from our findings and
analysis.
Implication 1: Treat retrieval as a high-variance intervention that requires safeguards.Across models and topics,
retrieval can both improve and degrade outcomes. Topic-level results show that even a semantically motivated retriever
(PKG) can underperform NoRAG in categories that are sensitive to specification details, such as string processing and
data-structure tasks. Error analysis further shows that retrieval can introduce new failure modes, including indentation
and type errors, that are absent or rare without retrieval. For practitioners, this implies that retrieval should not be
enabled unconditionally. A practical deployment strategy is to use retrieval gating based on simple signals, for example,
predicted relevance, topic classifier confidence, or a heuristic that detects when retrieved code is likely to conflict with
the query constraints, and to fall back to NoRAG when uncertainty is high.
Implication 2: Granularity is not a cosmetic choice; it changes the failure modes.Coarser units provide broader context
but increase the risk of irrelevant content, while finer units can increase precision but also increase exposure to
superficially similar templates. The topic analysis suggests that tasks with reusable schemas benefit from fine-grained
retrieval and candidate selection, whereas tasks with tight semantic constraints can be harmed by near-miss templates.
For researchers, this implies that reporting only overall pass@1 can hide important differences. Studies should report
performance stratified by topic or difficulty and should include a help-hurt analysis to characterize regressions induced
by retrieval at different granularities.
Implication 3: Selection is a first-class research problem, not an implementation detail.Candidate selection changes
both accuracy and error types. The topic-level results show that reranking can substantially improve categories that
otherwise regress under retrieval, indicating that many failures are selection failures rather than retrieval failures. At the
same time, reranking underperforms PKG in topics such as mathematics and algorithms, suggesting that similarity-based
selection can correlate with surface similarities rather than semantic validity. This motivates reranking objectives that
are closer to correctness, including execution-aware signals, constraint checks, or rerankers trained on counterexamples
where plausible solutions are incorrect. For practitioners, this implies that single-prompt augmentation is unlikely to be
robust without some form of candidate selection.
Implication 4: Error-mode shifts can guide targeted mitigation.Execution-trace errors provide actionable feedback
about what retrieval changes. For example, indentation errors point to formatting issues in how retrieved code is
inserted into the prompt, and type errors suggest mismatches between retrieved templates and the task specification.
These signals can be used to design defensive post-processing and formatting rules, such as canonical indentation
normalization, strict delimiter templates, and lightweight static checks before execution. For researchers, this implies
that evaluation should report not only accuracy but also how retrieval redistributes failure types, since two systems
with similar pass@1 may have very different reliability characteristics.
Manuscript submitted to ACM

22 Seddik et al.
Implication 5: Indexing and preprocessing costs are acceptable when amortized, but must be justified by downstream gains.
The cost analysis shows that PKG construction introduces additional preprocessing steps beyond embedding-based
retrieval, increasing both build time and storage footprint. However, these costs are amortized across many queries and
can be justified when the downstream accuracy improvements are large enough and when updates are incremental.
For practitioners, this implies that PKG-style retrieval is most appropriate in settings where (i) the corpus is relatively
stable or updated incrementally, and (ii) the system serves many queries per build. For researchers, this implies that
comparisons should include both accuracy and amortized costs, since build-time overhead is often negligible compared
to repeated inference-time improvements at scale.
For both researchers and practitioners, the main takeaway is that structured retrieval (such as PKG) improves the
quality of retrieved context relative to lexical baselines, but robust gains depend on careful selection and on mechanisms
that prevent retrieval-induced regressions. The most promising path forward is to couple fine-grained retrieval with
selection methods that are aligned with semantic correctness, while making retrieval conditional and budget-aware to
preserve robustness across topics and models.
6.6 Open Gaps and Directions
Our analyses surface several gaps that future work can build on. (1)Adaptive granularity:choosing function- vs. block-
level retrieval is currently static; developing query-adaptive policies could better match precision–recall needs across
problem types. (2)Robust text structuring:the text-centric pipeline depends on converting tutorials into structured JSON;
improving schema design and robustness of structuring under noisy or diverse prose remains open. (3)Selection beyond
retrieval:post-generation reranking improves outcomes but is decoupled from retrieval; integrating retrieval-time
uncertainty signals with candidate selection is an open design space. (4)Cost-aware retrieval:storage and preprocessing
costs can dominate end-to-end feasibility; more cost-effective graph construction, compression, and caching strategies
are needed. These directions follow directly from the trade-offs observed in our topic/error/cost analyses and aim to
make context retrieval for code generation both more reliable and more practical.
7 Threats to Validity
Internal threats relates to the selection of the datasets and models. We have selected widely used datasets for code and
tutorials, and models that are used in various code-intelligence studies. The embedder model was the top model on
HuggingFace at the time of running the experiments. As the models used are considered across all experiments, the
effects on the results are the same and therefore, mitigates any threats to the reliability of the results.
Though we have conducted experiments on no-RAG and RAG approaches and compared our work using open and
closed source models, they are also conducted on Python and English tutorials. Thus the results are limited to these
languages and more experiments are needed to evaluate the effects on other languages. We should also note that the
scope of the results currently is limited to the dataset used to build the knowledge graph. Therefore, if other domains
in Python or highly specific projects are considered, the results might not be directly applicable. In such cases, we
recommend that a new knowledge graph be developed based on the context of the project.
We anticipate minimal threats to the validity of the conducted tests. Though we used pass@1, we use a deterministic
approach and therefore there is no need to conduct statistical tests and the results are directly comparable. Similarly,
we provided various analysis on the topics and discussed the details from different angles, alleviating the conclusion
validity of the experiments and results.
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 23
8 Conclusion
We presented Programming Knowledge Graph, a novel approach for retrieval-augmented code generation, being both
code-centric and text-centric. Our results lead to key findings. Retrieval is not uniformly beneficial for code generation.
Structuring the retrieval space considering granularity improves reliability. Enhancing retrieval alone is insufficient and
candidate selection with reranking is a primary driver of gains. The effectiveness of RAG approaches vary by topic and
error mode; retrieval and reranking benefits are uneven across domains. These findings suggest that retrieval-augmented
code generation should be treated as a coupled problem of (i) constructing and indexing task-relevant artifacts at
appropriate granularity, and (ii) robustly selecting among candidates in the presence of retrieval noise. Practically, this
implies that future systems should emphasize retrieval gating, compute-parity evaluation, and rerankers that better
correlate with semantic correctness. Promising directions include learning rerankers with execution-aware signals,
incorporating lightweight specification checks for mathematically constrained topics, and developing adaptive retrieval
policies that account for model capacity and topic characteristics.
References
[1]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le,
et al. 2021. Program synthesis with large language models.arXiv preprint arXiv:2108.07732(2021).
[2]Zhuchen Cao, Sven Apel, Adish Singla, and Vera Demberg. 2025. Pragmatic Reasoning improves LLM Code Generation. arXiv:2502.15835 [cs.CL]
doi:10.48550/arXiv.2502.15835
[3]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, et al. 2021. Evaluating large language models trained on code.arXiv preprint arXiv:2107.03374(2021).
[4]Zimin Chen, Yue Pan, Siyu Lu, Jiayi Xu, Claire Le Goues, Martin Monperrus, and He Ye. 2025. Prometheus: Unified Knowledge Graphs for Issue
Resolution in Multilingual Codebases. arXiv:2507.19942 [cs.SE] https://arxiv.org/abs/2507.19942
[5]Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and Stéphane Clinchant. 2025. Provence: efficient and robust context pruning for
retrieval-augmented generation. arXiv:2501.16214 [cs.CL] doi:10.48550/arXiv.2501.16214 Accepted to ICLR 2025.
[6]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang,
Angela Fan, et al. 2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783(2024).
[7]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation
for large language models: A survey.arXiv preprint arXiv:2312.10997(2023).
[8]Wenchao Gu, Juntao Chen, Yanlin Wang, Tianyue Jiang, Xingzhe Li, Mingwei Liu, Xilin Liu, Yuchi Ma, and Zibin Zheng. 2025. What to Retrieve for
Effective Retrieval-Augmented Code Generation? An Empirical Study and Beyond. arXiv:2503.20589 [cs.SE] https://arxiv.org/abs/2503.20589
[9]Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. Hipporag: Neurobiologically inspired long-term memory for
large language models. InThe Thirty-eighth Annual Conference on Neural Information Processing Systems.
[10] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. InInternational
conference on machine learning. PMLR, 3929–3938.
[11] Lovisa Hagström, Sara Vera Marjanovic, Haeun Yu, Arnav Arora, Christina Lioma, Maria Maistro, Pepa Atanasova, and Isabelle Augenstein. 2025.
A Reality Check on Context Utilisation for Retrieval-Augmented Generation. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.).
Association for Computational Linguistics, Vienna, Austria, 19691–19730. doi:10.18653/v1/2025.acl-long.968
[12] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025. GRAG: Graph Retrieval-Augmented Generation.
arXiv:2405.16506 [cs.LG] https://arxiv.org/abs/2405.16506
[13] Dong Huang, Qingwen Bu, Jie M Zhang, Michael Luck, and Heming Cui. 2023. Agentcoder: Multi-agent-based code generation with iterative testing
and optimisation.arXiv preprint arXiv:2312.13010(2023).
[14] Runsong Jia, Bowen Zhang, Sergio José Rodríguez Méndez, and Pouya G Omran. 2025. StructRAG: Structure-Aware RAG Framework with Scholarly
Knowledge Graph for Diverse Question Answering. InCompanion Proceedings of the ACM on Web Conference 2025. 2567–2573.
[15] Pengcheng Jiang, Siru Ouyang, Yizhu Jiao, Ming Zhong, Runchu Tian, and Jiawei Han. 2025. A Survey on Retrieval And Structuring Augmented
Generation with Large Language Models. arXiv:2509.10697 [cs.CL] doi:10.48550/arXiv.2509.10697
[16] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active
Retrieval Augmented Generation. InProceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 7969–7992.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in Neural Information Processing Systems33
Manuscript submitted to ACM

24 Seddik et al.
(2020), 9459–9474.
[18] Jianyuan Liang, Shuyang Hou, Haoyue Jiao, Yaxian Qing, Anqi Zhao, Zhangxiao Shen, Longgang Xiang, and Huayi Wu. 2025. GeoGraphRAG: A
graph-based retrieval-augmented generation approach for empowering large language models in automated geospatial modeling.International
Journal of Applied Earth Observation and Geoinformation142 (2025), 104712.
[19] Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, and Vijai Mohan. 2025. Refrag: Rethinking rag based decoding.arXiv
preprint arXiv:2509.01092(2025).
[20] Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Qizhe Shieh, and Wenmeng Zhou. 2025. Codexgraph: Bridging
large language models and code repositories via code graph databases. InProceedings of the 2025 Conference of the Nations of the Americas Chapter of
the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 142–160.
[21] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang
Wei, et al. 2024. Starcoder 2 and the stack v2: The next generation.arXiv preprint arXiv:2402.19173(2024).
[22] Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. 2025. Alibaba LingmaAgent: Improving Automated Issue Resolution
via Comprehensive Repository Exploration. arXiv:2406.01422 [cs.SE] https://arxiv.org/abs/2406.01422
[23] Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, Jiawei Han, Hongming Zhang, and Dong Yu. 2025. RepoGraph:
Enhancing AI Software Engineering with Repository-level Code Graph. arXiv:2410.14684 [cs.SE] https://arxiv.org/abs/2410.14684
[24] Kunal Pai, Premkumar Devanbu, and Toufique Ahmed. 2025. CoDocBench: A Dataset for Code-Documentation Alignment in Software Maintenance.
arXiv:2502.00519 [cs.SE] doi:10.48550/arXiv.2502.00519 Accepted at the 2025 IEEE/ACM 22nd International Conference on Mining Software
Repositories (MSR) – Data and Tool Showcase Track.
[25] Md Rizwan Parvez, Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Retrieval Augmented Code Generation and
Summarization. InFindings of the Association for Computational Linguistics: EMNLP 2021. 2719–2734.
[26] Nicolas Mejia Petit. 2024. Tested-143k-Python-Alpaca. https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca
[27] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al .
2023. Code llama: Open foundation models for code.arXiv preprint arXiv:2308.12950(2023).
[28] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al .
2023. Code llama: Open foundation models for code.arXiv preprint arXiv:2308.12950(2023).
[29] Nikita Sorokin, Ivan Sedykh, and Valentin Malykh. 2025. Iterative Self-Training for Code Generation via Reinforced Re-Ranking.
arXiv:2504.09643 [cs.CL] doi:10.48550/arXiv.2504.09643 Published at ECIR 2025; related DOI: 10.1007/978-3-031-88714-7_21.
[30] Hongjin Su, Shuyang Jiang, Yuhang Lai, Haoyuan Wu, Boao Shi, Che Liu, Qian Liu, and Tao Yu. 2024. EVOR: Evolving Retrieval for Code
Generation. InFindings of the Association for Computational Linguistics: EMNLP 2024. Association for Computational Linguistics, 2538–2554.
https://aclanthology.org/2024.findings-emnlp.143/
[31] Hao Sun, Hengyi Cai, Yuchen Li, Xuanbo Fan, Xiaochi Wei, Shuaiqiang Wang, Yan Zhang, and Dawei Yin. 2025. Enhancing Retrieval-Augmented
Generation via Evidence Tree Search. InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics,
Vienna, Austria, 24116–24127. doi:10.18653/v1/2025.acl-long.1175
[32] Yicheng Tao, Yao Qin, and Yepang Liu. 2025. Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches.arXiv
e-prints(2025), arXiv–2510.
[33] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard,
Bobak Shahriari, Alexandre Ramé, et al .2024. Gemma 2: Improving open language models at a practical size.arXiv preprint arXiv:2408.00118(2024).
[34] Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. 2024. CodeRAG-Bench: Can Retrieval
Augment Code Generation?arXiv preprint arXiv:2406.14497(2024).
[35] Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F. Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. 2025. CodeRAG-Bench: Can Retrieval
Augment Code Generation?. InFindings of the Association for Computational Linguistics: NAACL 2025, Luis Chiruzzo, Alan Ritter, and Lu Wang
(Eds.). Association for Computational Linguistics, Albuquerque, New Mexico, 3199–3214. doi:10.18653/v1/2025.findings-naacl.176
[36] Boyang Yang, Jiadong Ren, Shunfu Jin, Yang Liu, Feng Liu, Bach Le, and Haoye Tian. 2025. Enhancing repository-level software repair via
repository-aware knowledge graphs. arXiv:2503.21710 [cs.SE] https://arxiv.org/abs/2503.21710
[37] Zairun Yang, Yilin Wang, Zhengyan Shi, Yuan Yao, Lei Liang, Keyan Ding, Emine Yilmaz, Huajun Chen, and Qiang Zhang. 2025. EventRAG:
Enhancing LLM Generation with Event Knowledge Graphs. InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). 16967–16979.
[38] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-
level code completion through iterative retrieval and generation.arXiv preprint arXiv:2303.12570(2023).
[39] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Junnan Dong, Yi
Chang, and Xiao Huang. 2025. A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models. arXiv:2501.13958 [cs.CL]
https://arxiv.org/abs/2501.13958
[40] Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, and Tongshuang Wu. 2025. cAST: Enhancing Code Retrieval-Augmented
Generation with Structural Chunking via Abstract Syntax Tree. InFindings of the Association for Computational Linguistics: EMNLP 2025, Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China,
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 25
8106–8116. doi:10.18653/v1/2025.findings-emnlp.430
[41] Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, and Shijing Chen. 2025. SemRAG: Semantic Knowledge-Augmented RAG for Improved
Question-Answering.arXiv preprint arXiv:2507.21110(2025).
[42] Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, and Zengchang Qin. 2025. Mix-of-Granularity: Optimize the Chunking Granularity for
Retrieval-Augmented Generation. InProceedings of the 31st International Conference on Computational Linguistics, Owen Rambow, Leo Wanner,
Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert (Eds.). Association for Computational Linguistics, Abu Dhabi,
UAE, 5756–5774. https://aclanthology.org/2025.coling-main.384/
[43] Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang, and Graham Neubig. 2022. DocPrompting: Generating Code by Retrieving the Docs. InThe
Eleventh International Conference on Learning Representations.
[44] Shuyan Zhou, Uri Alon, Frank F Xu, Zhiruo Wang, Zhengbao Jiang, and Graham Neubig. 2022. Docprompting: Generating code by retrieving the
docs.arXiv preprint arXiv: 2207.05987(2022).
[45] Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al .2024. DeepSeek-Coder-V2:
Breaking the Barrier of Closed-Source Models in Code Intelligence.arXiv preprint arXiv:2406.11931(2024).
[46] Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge graph-guided retrieval augmented generation.arXiv preprint
arXiv:2502.06864(2025).
A PKG Schema Inventory
This appendix lists the concrete node and edge types used to instantiate the abstract schema in Section 3.1.
A.1 Node types
We use the following node types:
‚Name: a function identifier (string).
‚Impl: a full function implementation (code).
‚Block: a syntactic code block span inside a function (code).
‚PathValue: a (path, value) leaf extracted from a JSON representation of tutorial content (text), following
Section 3.1.2.
A.2 Edge types
Edges capture identifier-to-content links and refinement (containment) relations:
‚has_impl:NameÑImpl.
‚has_block:ImplÑBlock.
‚parent:BlockÑBlock(syntactic block nesting).
‚json_child:PathValue ÑPathValue(JSON hierarchy when internal path nodes are materialized, see Sec-
tion 3.1.2).
B Examples and Prompts for PKG Approach
B.1 Challenges in Retrieving Information from PKG
This section discusses scenarios where the PKG fails to retrieve accurate or relevant information. One notable challenge
arises when the task requires domain-specific expertise. For example, if the task involves a specialized framework or
project-specific code, the PKG must be populated with relevant data from the corresponding domain or project. Failures
occur when queries target a graph that lacks such domain knowledge. Addressing this issue necessitates updating the
graph with appropriate domain-specific information.
Manuscript submitted to ACM

26 Seddik et al.
Through topic analysis, we identified that the PKG often struggles with certain problem categories, such as string
manipulation. Experimental observations indicate that this challenge stems from the limitations of both the embedder
model and the baseline model, which tend to prioritize semantic meaning over structural characteristics of strings.
Example Problem: Write a Python function to convert lowercase characters to uppercase and vice versa, transforming
inputs such as "Hello" into "hELLO" and "pYthon" into "PyTHON".
Challenges:
‚Embedding Model’s Semantic Bias:
In RAG, the embedder retrieves content primarily based on semantic meaning rather than formatting or structural
patterns. For example, it might interpret "Hello" as a greeting, ignoring the case transformation requirement.
‚LLM’s Tokenization and Semantic Prioritization:
LLMs tokenize text based on meaning rather than formatting. Consequently, tokens like "Hello" and "hello" are
often treated identically, making tasks involving case transformations particularly challenging.
In summary, both RAG retrieval and LLM tokenization emphasize semantic understanding over structural or
formatting details, complicating the handling of tasks like string manipulation. This limitation reduces the effectiveness
of PKG-based approaches for such problem categories.
B.2 CodeLLama7b
B.2.1 Prompts.The prompts we have used for CodeLlama7B model is provided in Code B.2.1:
1def codellama_prompt(problem,augmented_data=None):
2if augmented_data:
3prompt = f"""[INST]␣You␣are␣a␣python␣programmer.␣Solve␣the␣following␣problem:\n{problem}␣\n\
nThe␣following␣code␣might␣be␣helpful:\n{augmented_data}\nIf␣helper␣section␣is␣useful,␣
integrate␣their␣logic␣directly␣into␣the␣body␣of␣the␣main␣function,␣otherwise␣just␣ignore␣
them.␣You␣MUST␣write␣your␣solution␣between␣[PYTHON]␣and␣[/PYTHON].␣Your␣solution␣MUST␣be␣
executable.[/INST]"""
4return prompt
5else:
6prompt = f"""[INST]␣You␣are␣a␣python␣programmer.␣Solve␣the␣following␣problem:\n{problem}␣\n\
nPlease␣write␣the␣python␣solution␣inside␣[PYTHON]␣and␣[/PYTHON]␣tags.\n[/INST]"
7"""
8␣␣␣␣␣␣␣␣return␣prompt
B.2.2 Topic-Based Accuracy distribution.Figure 7 illustrates the distribution of MBPP problems on a two-dimensional
plot, where the embedding dimensions have been reduced to two for visualization purposes. The different problem
topics are represented by distinct shapes, while the correctness of the solutions is indicated by color. Problems that
were solved incorrectly are shown in orange, and those solved correctly are shown in green. The legend for each topic
separates the total number of correct solutions from the incorrect ones using a slash ("/"). Figure 8 shows the distribution
of correct and incorrect problems when we apply our approach.
B.3 StarCoder2-7B
B.3.1 Prompts.The prompts we have used for StarCoder2-7B model is provided in Code B.3.1:
1
2def starcoder_prompt(problem,augmented_data=None):
3if augmented_data:
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 27
Fig. 7. The distribution of MBPP solutions on each topic in NoRAG setting using CodeLLama7b.
Fig. 8. The distribution of MBPP solutions on each topic using our re-ranker using CodeLLama7b.
4prompt = f"""###␣Instruction
5␣␣␣␣␣␣␣␣You␣are␣a␣python␣programmer.␣Solve␣the␣following␣problem:\n{problem}␣\n\n␣The␣following␣code␣
might␣be␣helpful:\n{augmented_data}\n.␣If␣they␣are␣useful,␣integrate␣their␣logic␣directly␣into␣
the␣body␣of␣the␣main␣function,␣otherwise␣just␣ignore␣them.\n
6␣␣␣␣␣␣␣␣###␣Response
Manuscript submitted to ACM

28 Seddik et al.
7␣␣␣␣␣␣␣␣"""
8return prompt
9else:
10prompt = f"""###␣Instruction
11␣␣␣␣␣␣␣␣You␣are␣a␣python␣programmer.␣Solve␣the␣following␣problem:\n{problem}␣\n\n
12␣␣␣␣␣␣␣␣###␣Response
13␣␣␣␣␣␣␣␣"""
14return prompt
Fig. 9. The distribution of MBPP solutions on each topic without RAG using StarCoder2-7B.
B.3.2 Topic-Based Accuracy Distribution.Figure 9 presents the distribution of MBPP problems on a two-dimensional
plot, with the embedding dimensions reduced for visualization. Each problem topic is represented by a unique shape,
while solution correctness is color-coded. Problems incorrectly solved by StarCoder2-7B are highlighted in orange,
whereas correctly solved problems are shown in green. The legend for each topic indicates the total number of correct
versus incorrect solutions using a "correct/incorrect" format.
Additionally, Figure 10 visualizes the same distribution but reflects the accuracy after applying our proposed approach,
showcasing improvements in solution correctness across topics.
B.4 DeepSeek-Coder-7B
B.4.1 Prompts.The prompts we have used for DeepSeek-Coder-7B model is provided in Code B.4.1:
1
2def deepseek_prompt(problem,augmented_data=None):
3if augmented_data:
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 29
Fig. 10. The distribution of MBPP solutions on each topic using our proposed re-ranker using StarCoder2-7B.
4prompt = f"""[INST]␣You␣are␣a␣python␣programmer.␣Solve␣the␣following␣problem:\n{problem}␣\n\n␣
The␣following␣code␣might␣be␣helpful:\n{augmented_data}\n.If␣they␣are␣useful,␣integrate␣
their␣logic␣directly␣into␣the␣body␣of␣the␣main␣function,␣otherwise␣just␣ignore␣them.\n[/
INST]"""
5return prompt
6else:
7prompt = f"""[INST]␣You␣are␣a␣python␣programmer.␣␣Solve␣the␣following␣problem:␣\n␣{problem}␣\n
\n[/INST]"""
8return prompt
B.4.2 Topic-Based Accuracy Distribution.Figure 11 displays the distribution of problems from the MBPP dataset in a
two-dimensional plot, achieved by reducing the embedding dimensions for improved visualization. Each distinct shape
in the plot corresponds to a specific problem topic, while the correctness of the solutions is indicated by color coding.
Problems that were solved incorrectly are represented in orange, whereas those that were solved correctly are shown in
green. The legend accompanying each topic delineates the total number of correct solutions from the incorrect ones,
separated with a slash ("/").
In addition, Figure 12 presents a similar distribution of problems, highlighting the outcomes after applying our novel
approach. This figure further distinguishes between correct and incorrect solutions, allowing for a comparative analysis
of the effectiveness of our method.
B.5 Examples
In this section, we present two selected samples from the HumanEval benchmark. We provide the responses generated
by StarCoder-2-7B and DeepSeek-Coder-7B models. Each model’s output is displayed in two scenarios: first, without
Manuscript submitted to ACM

30 Seddik et al.
Fig. 11. The distribution of MBPP solutions on each topic in NoRAG setting using DeepSeek-Coder-7B.
Fig. 12. The distribution of MBPP solutions on each topic using our reranker using DeepSeek-Coder-7B.
using RAG, and second, utilizing our PKG approach. These examples illustrate how incorporating additional context
can enhance the models’ ability to solve complex problems more effectively.
HumanEval problem 159, solved by Starcoder2-7B without RAG (Failed):
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 31
1
2def eat(number, need, remaining):
3"""
4␣␣␣␣You're␣a␣hungry␣rabbit,␣and␣you␣already␣have␣eaten␣a␣certain␣number␣of␣carrots,
5␣␣␣␣but␣now␣you␣need␣to␣eat␣more␣carrots␣to␣complete␣the␣day's␣meals.
6␣␣␣␣you␣should␣return␣an␣array␣of␣[␣total␣number␣of␣eaten␣carrots␣after␣your␣meals,
7␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣the␣number␣of␣carrots␣left␣after␣your␣meals␣]
8␣␣␣␣if␣there␣are␣not␣enough␣remaining␣carrots,␣you␣will␣eat␣all␣remaining␣carrots,␣but␣will␣still␣be␣
hungry.
9
10␣␣␣␣Example:
11␣␣␣␣*␣eat(5,␣6,␣10)␣->␣[11,␣4]
12␣␣␣␣*␣eat(4,␣8,␣9)␣->␣[12,␣1]
13␣␣␣␣*␣eat(1,␣10,␣10)␣->␣[11,␣0]
14␣␣␣␣*␣eat(2,␣11,␣5)␣->␣[7,␣0]
15
16␣␣␣␣Variables:
17␣␣␣␣@number␣:␣integer
18␣␣␣␣␣␣␣␣the␣number␣of␣carrots␣that␣you␣have␣eaten.
19␣␣␣␣@need␣:␣integer
20␣␣␣␣␣␣␣␣the␣number␣of␣carrots␣that␣you␣need␣to␣eat.
21␣␣␣␣@remaining␣:␣integer
22␣␣␣␣␣␣␣␣the␣number␣of␣remaining␣carrots␣thet␣exist␣in␣stock
23
24␣␣␣␣Constrain:
25␣␣␣␣*␣0␣<=␣number␣<=␣1000
26␣␣␣␣*␣0␣<=␣need␣<=␣1000
27␣␣␣␣*␣0␣<=␣remaining␣<=␣1000
28
29␣␣␣␣Have␣fun␣:)
30␣␣␣␣"""
31return [number + need, remaining - need]
HumanEval problem 159, solved by Starcoder2-7B using PKG (Passed):
1
2### Instruction
3#You are a python expert programmer. Implement a python function to #solve the following problem:
4
5def eat(number, need, remaining):
6"""
7␣␣␣␣You're␣a␣hungry␣rabbit,␣and␣you␣already␣have␣eaten␣a␣certain␣number␣of␣carrots,
8␣␣␣␣but␣now␣you␣need␣to␣eat␣more␣carrots␣to␣complete␣the␣day's␣meals.
9␣␣␣␣you␣should␣return␣an␣array␣of␣[␣total␣number␣of␣eaten␣carrots␣after␣your␣meals,
10␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣the␣number␣of␣carrots␣left␣after␣your␣meals␣]
11␣␣␣␣if␣there␣are␣not␣enough␣remaining␣carrots,␣you␣will␣eat␣all␣remaining␣carrots,␣but␣will␣still␣be␣
hungry.
12
13␣␣␣␣Example:
14␣␣␣␣*␣eat(5,␣6,␣10)␣->␣[11,␣4]
15␣␣␣␣*␣eat(4,␣8,␣9)␣->␣[12,␣1]
16␣␣␣␣*␣eat(1,␣10,␣10)␣->␣[11,␣0]
17␣␣␣␣*␣eat(2,␣11,␣5)␣->␣[7,␣0]
18
19␣␣␣␣Variables:
20␣␣␣␣@number␣:␣integer
Manuscript submitted to ACM

32 Seddik et al.
21␣␣␣␣␣␣␣␣the␣number␣of␣carrots␣that␣you␣have␣eaten.
22␣␣␣␣@need␣:␣integer
23␣␣␣␣␣␣␣␣the␣number␣of␣carrots␣that␣you␣need␣to␣eat.
24␣␣␣␣@remaining␣:␣integer
25␣␣␣␣␣␣␣␣the␣number␣of␣remaining␣carrots␣thet␣exist␣in␣stock
26
27␣␣␣␣Constrain:
28␣␣␣␣*␣0␣<=␣number␣<=␣1000
29␣␣␣␣*␣0␣<=␣need␣<=␣1000
30␣␣␣␣*␣0␣<=␣remaining␣<=␣1000
31
32␣␣␣␣Have␣fun␣:)
33␣␣␣␣"""
34
35
36#The following helper code might be helpful:
37#helper code 1:
38postEatWeight = totalWeight - amountEaten
39combinedWeight = postEatWeight / 7
40largerBoxPostEat = combinedWeight * 4
41smallerBoxPostEat = combinedWeight * 3
42largerBox = largerBoxPostEat + amountEaten
43smallerBox = smallerBoxPostEat
44return (round(largerBox), round(smallerBox))
45#End of helper section.
46#Write your solution inside [PYTHON] and [/PYTHON] tags. The solution #should be executable and do NOT
call helper functions. If they are #useful, integrate their logic directly into the body of the
main #function, otherwise just ignore them.
47### Response
48#Here is the Python function to solve the problem:
49
50
51def eat(number, need, remaining):
52if remaining >= need:
53return [number + need, remaining - need]
54else:
55return [number + remaining, 0]
HumanEval problem 90, solved by DeepSeek-Coder2 without RAG (Failed):
1def next_smallest(lst):
2"""
3␣␣␣␣You␣are␣given␣a␣list␣of␣integers.
4␣␣␣␣Write␣a␣function␣next_smallest()␣that␣returns␣the␣2nd␣smallest␣element␣of␣the␣list.
5␣␣␣␣Return␣None␣if␣there␣is␣no␣such␣element.
6
7␣␣␣␣next_smallest([1,␣2,␣3,␣4,␣5])␣==␣2
8␣␣␣␣next_smallest([5,␣1,␣4,␣3,␣2])␣==␣2
9␣␣␣␣next_smallest([])␣==␣None
10␣␣␣␣next_smallest([1,␣1])␣==␣None
11␣␣␣␣"""
12if len(lst) < 2:
13return None
14else:
15lst.sort()
16return lst[1]
Manuscript submitted to ACM

Context-Augmented Code Generation Using Programming Knowledge Graphs 33
HumanEval problem 90, solved by DeepSeek-Coder2 using PKG (Passed):
1### Instruction:
2#You are an expert python programmer. Solve the following PROBLEM in #python:
3
4def next_smallest(lst):
5"""
6␣␣␣␣You␣are␣given␣a␣list␣of␣integers.
7␣␣␣␣Write␣a␣function␣next_smallest()␣that␣returns␣the␣2nd␣smallest␣element␣of␣the␣list.
8␣␣␣␣Return␣None␣if␣there␣is␣no␣such␣element.
9
10␣␣␣␣next_smallest([1,␣2,␣3,␣4,␣5])␣==␣2
11␣␣␣␣next_smallest([5,␣1,␣4,␣3,␣2])␣==␣2
12␣␣␣␣next_smallest([])␣==␣None
13␣␣␣␣next_smallest([1,␣1])␣==␣None
14␣␣␣␣"""
15
16#The following helper code might be helpful:
17#helper code 1:
18if len(lst) < 2:
19return None
20smallest = float('inf')
21next_smallest = float('inf')
22for i in range(0, len(lst)):
23if lst[i] < smallest:
24next_smallest = smallest
25smallest = lst[i]
26elif lst[i] > smallest and lst[i] < next_smallest:
27next_smallest = lst[i]
28return None if next_smallest == float('inf') else next_smallest
29
30#End of helper section.
31#Please write the python solution inside [PYTHON] and [/PYTHON] tags.
32### Response:
33
34def next_smallest(lst):
35if len(lst) < 2:
36return None
37smallest = second_smallest = float('inf')
38for element in lst:
39if element < smallest:
40second_smallest = smallest
41smallest = element
42elif smallest < element < second_smallest:
43second_smallest = element
44return None if second_smallest == float('inf') else second_smallest
Manuscript submitted to ACM
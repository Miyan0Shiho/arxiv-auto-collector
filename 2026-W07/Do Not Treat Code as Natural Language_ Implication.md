# Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond

**Authors**: Minh Le-Anh, Huyen Nguyen, Khanh An Tran, Nam Le Hai, Linh Ngo Van, Nghi D. Q. Bui, Bach Le

**Published**: 2026-02-12 07:44:00

**PDF URL**: [https://arxiv.org/pdf/2602.11671v1](https://arxiv.org/pdf/2602.11671v1)

## Abstract
Large language models for code (CodeLLMs) have demonstrated remarkable success in standalone code completion and generation, sometimes even surpassing human performance, yet their effectiveness diminishes in repository-level settings where cross-file dependencies and structural context are essential. Existing Retrieval-Augmented Generation (RAG) approaches often borrow strategies from NLP, relying on chunking-based indexing and similarity-based retrieval. Chunking results in the loss of coherence between code units and overlooks structural relationships, while similarity-driven methods frequently miss functionally relevant dependencies such as helper functions, classes, or global variables. To address these limitations, we present Hydra, a repository-level code generation framework that treats code as structured code rather than natural language. Our approach introduces (i) a structure-aware indexing strategy that represents repositories as hierarchical trees of functions, classes, and variables, preserving code structure and dependencies, (ii) a lightweight dependency-aware retriever (DAR) that explicitly identifies and retrieves the true dependencies required by a target function, and (iii) a hybrid retrieval mechanism that combines DAR with similarity-based retrieval to provide both essential building blocks and practical usage examples. Extensive experiments on the challenging DevEval and RepoExec benchmarks, both requiring function implementation from real-world repositories with complex large repository context, show that Hydra achieves state-of-the-art performance across open- and closed-source CodeLLMs. Notably, our method establishes a new state of the art in repository-level code generation, surpassing strongest baseline by over 5% in Pass@1 and even enabling smaller models to match or exceed the performance of much larger ones that rely on existing retrievers.

## Full Text


<!-- PDF content starts -->

Do Not Treat Code as Natural Language: Implications for
Repository-Level Code Generation and Beyond
MINH LE-ANH,FPT Software AI Center; Hanoi University of Science and Technology, Vietnam
HUYEN NGUYEN,Hanoi University of Science and Technology, Vietnam
AN KHANH TRAN,Hanoi University of Science and Technology, Vietnam
NAM LE HAI∗,Hanoi University of Science and Technology, Vietnam
LINH NGO VAN,Hanoi University of Science and Technology, Vietnam
NGHI D.Q. BUI,FPT Software AI Center, Vietnam
BACH LE,The University of Melbourne, Australia
Large language models for code (CodeLLMs) have demonstrated remarkable success in standalone code com-
pletion and generation, sometimes even surpassing human performance, yet their effectiveness diminishes in
repository-level settings where cross-file dependencies and structural context are essential. Existing Retrieval-
Augmented Generation (RAG) approaches often borrow strategies from NLP, relying on chunking-based
indexing and similarity-based retrieval. Chunking results in the loss of coherence between code units and
overlooks structural relationships, while similarity-driven methods frequently miss functionally relevant
dependencies such as helper functions, classes, or global variables. To address these limitations, we present
Hydra, a repository-level code generation framework that treats code as structured code rather than natural
language. Our approach introduces (i) a structure-aware indexing strategy that represents repositories as hier-
archical trees of functions, classes, and variables, preserving code structure and dependencies, (ii) a lightweight
dependency-aware retriever (DAR) that explicitly identifies and retrieves the true dependencies required by a
target function, and (iii) a hybrid retrieval mechanism that combines DAR with similarity-based retrieval to
provide both essential building blocks and practical usage examples. Extensive experiments on the challenging
DevEval and RepoExec benchmarks, both requiring function implementation from real-world repositories
with complex large repository context, show thatHydraachieves state-of-the-art performance across open-
and closed-source CodeLLMs. Notably, our method establishes a new state of the art in repository-level code
generation, surpassing strongest baseline by over 5% in Pass@1 and even enabling smaller models to match or
exceed the performance of much larger ones that rely on existing retrievers.
CCS Concepts:•Software and its engineering →Automatic programming;•Information systems →
Information retrieval;•Computing methodologies→Artificial intelligence.
Additional Key Words and Phrases: Repository-Level Code Generation, Code Generation, Retrieval-Augmented
Generation, Large Language Models
∗Nam Le Hai is the corresponding author
Authors’ Contact Information: Minh Le-Anh, minhla4@fpt.com, FPT Software AI Center; Hanoi University of Science and
Technology, Hanoi, Vietnam; Huyen Nguyen, huyen.nt235507@sis.hust.edu.vn, Hanoi University of Science and Technology,
Hanoi, Vietnam; An Khanh Tran, khanh.ta225447@sis.hust.edu.vn, Hanoi University of Science and Technology, Hanoi,
Vietnam; Nam Le Hai, namlh@soict.hust.edu.vn, Hanoi University of Science and Technology, Hanoi, Vietnam; Linh Ngo Van,
linhnv@soict.hust.edu.vn, Hanoi University of Science and Technology, Hanoi, Vietnam; Nghi D.Q. Bui, bdqnghi@gmail.com,
FPT Software AI Center, Hanoi, Vietnam; Bach Le, bach.le@unimelb.edu.au, The University of Melbourne, Melbourne,
Victoria, Australia.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM XXXX-XXXX/2026/2-ART
https://doi.org/10.1145/3797144
, Vol. 1, No. 1, Article . Publication date: February 2026.arXiv:2602.11671v1  [cs.SE]  12 Feb 2026

2 Minh et al.
ACM Reference Format:
Minh Le-Anh, Huyen Nguyen, An Khanh Tran, Nam Le Hai, Linh Ngo Van, Nghi D.Q. Bui, and Bach Le. 2026.
Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond. 1, 1
(February 2026), 24 pages. https://doi.org/10.1145/3797144
1 Introduction
Large language models for code (CodeLLMs) have achieved impressive results in code completion
and generation, but they typically operate within a single file or limited context [ 2,5,6,26,38,46].
In real-world software development, completing code often requires incorporating repository-level
context or cross-file context [ 8,25,27,28,49,52]. Retrieval-Augmented Generation (RAG) has
emerged as a promising approach to address this need [ 3,48,49,52]. In repository-level code
completion, a retrieval module fetches relevant code snippets from across the codebase to supply
the model with cross-file information beyond the current file. This paradigm has shown strong
empirical gains, bridging the gap in cross-file knowledge. Recent work has further improved RAG
by designing better retrieval mechanisms and structured prompts for code LMs [ 49]. However,
most existing approaches treat code as if it were natural language, directly borrowing techniques
from the NLP domain. This leads to several key challenges that we aim to overcome.
Limitations of Chunking-Based Indexing.Multiple prior code generation RAG-based systems
index a codebase by breaking files into sequential chunks of code text [ 32,47,49,52]. For example,
files are split into fixed-size snippets (e.g. 20-line blocks) with sliding windows , which are then used
to build a retrieval index. At query time, the model’s context is matched against these chunks using
similarity search. While this chunking strategy is common in long-document NLP tasks [ 11,13,43],
it is poorly aligned with the structural nature of code: it fragments logically coherent units (e.g.
splitting a function or class), includes irrelevant surrounding code context, and assumes that relevant
context must be contiguous in text. In reality, the most important information for code completion
often lies in structure code blocks such as function definitions, class implementations, or imported
modules that may reside in entirely different files. Unfortunately, most existing methods for repo-
level code generation that often chunk files into continuous blocks [ 47,49,52] fail to retrieve
context that fully reflects this behavior. Treating code as flat text chunks ignores its hierarchical
and dependency-driven structure, ultimately limiting retrieval effectiveness in repository-level
tasks.
Limitations of Similarity-Based Retrieval.Similarity-driven methods such as BM25, TF-IDF,
Jaccard similarity, or embedding-based cosine search have been widely applied in document and
code search [ 7,15,24,29,37], where the goal is to retrieve a semantically or lexically relevant
snippet given a natural language or code query. In that setting, ranking by surface similarity often
suffices, since the user primarily seeks related examples or references. However, in repository-level
code generation, the requirements are fundamentally different. Generated code must align with the
current state of the repository, respect existing dependencies, and follow the coding conventions
of the project. Pure similarity matching is suboptimal for this, as functionally relevant context
(e.g., a helper function definition, a data structure declaration, or an API implementation) may not
share obvious lexical overlap with the target code. As a result, models relying only on similarity
retrieval can miss crucial dependencies or introduce superficially related but irrelevant code. Recent
studies [ 25,27] underscore that: providing a model with the ground-truth dependency context (all
functions/classes that the code calls) yields significant gains in repository-level generation, whereas
missing these dependencies leads to errors or redundant reimplementation. Gu et al . [14] further
show that similarity-based retrieval “struggle to capture code semantics” and may even degrade
generation quality by introducing noisy context, sometimes reducing accuracy. These findings
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 3
underscore that while similarity retrieval is effective for code search, it might be insufficient for
the task of code generation in repository-level settings.
To bridge these gaps, we proposeHydra, a repository-level code generation framework that
explicitly exploits the structure and dependencies in a codebase. Specifically,Hydrais designed
to retrieve and utilize repository context in a way that mirrors how developers themselves would
navigate code, addressing the shortcomings identified above. Our contributions are summarized as
follows:
1○We present a method, Structure-aware indexing, to represent the entire repository as a hierar-
chical tree of code components (such as functions, classes, and global variables). Rather than
splitting files into arbitrary text chunks, we parse and index the code at the level of logical units.
This structured representation preserves critical relationships (for example, which functions
belong to a class, or which files import which modules) that would be overlooked if flat text
chunking is applied.
2○We introduce a retrieval mechanism centered oncall-graph dependencies, namely dependency-
aware retriever (DAR). Given a code generation query, the DAR automatically identifies
referenced symbols and retrieves their definitions or implementations from the repository.
To make this practical, we also propose a lightweight model with a fine-tuning strategy that
empowers efficient and accurate dependency retrieval without heavy computational overhead.
By explicitly modeling dependencies rather than relying on similarity alone, DAR fills a critical
gap left by prior work.
3○We proposeHydraRetriever, a hybrid retrieval strategy that combines DARwith simil arity-
based retrieval (BM25). While DAR ensures that the model has access to the correct building
blocks (functions, classes, variables), similarity-based retrieval complements this by providing
usage examples of how these dependencies are invoked in practice. This design yields a richer
and more reliable context, substantially improving repository-level code generation.
4○We conduct extensive experiments on two recent repository-level benchmarks, DevEval
[28] and RepoExec [ 27], focusing on function-level code generation in Python. Using both
open-source (Qwen2.5-Coder, 1.5B & 7B) and closed-source (GPT-4.1 mini) models, we show
thatHydrageneralizes across model families and sizes. It achieves state-of-the-art results,
surpassing the strongest baselines with close-sourced generator by more than 5%, and notably
allows a 1.5B model to match or surpass a 7B model with existing retriever or no retrieval,
effectively bridging a∼4×size gap.
2 Background & Related Work
2.1 Large Language Models for Code
The rise of large language models pretrained on code (CodeLLMs) has transformed code generation.
Open-source like StarCoder [ 36] and CodeLlama [ 41] provided multilingual support and achieved
state-of-the-art performance on benchmarks like MultiPL-E [ 4]. More recently, instruction-tuned
models such as DeepSeek-Coder[ 17] and Qwen-Coder[ 21] have improved controllability and
debugging, narrowing the gap with proprietary models. Closed-source models have pushed the
frontier further. OpenAI Codex [ 6] pioneered zero-shot code generation, later extended by GPT-4/4o
[1,23] with code interpreter functionalities. Anthropic’s Claude 3 and Google’s Gemini [ 44,45]
also demonstrate strong reasoning and multilingual coding ability, with enhanced support for
repository-level understanding and integration into developer workflows.
, Vol. 1, No. 1, Article . Publication date: February 2026.

4 Minh et al.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) [ 13,43,53] is a general framework that enhances generative
models by providing them with external, task-relevant context. Instead of relying solely on a model’s
parametric knowledge, RAG augments the input with information retrieved from a supporting
corpus. For repository-level code generation, this means treating the project’s entire codebase as
the external knowledge source [ 30,49,52], enabling the model to access cross-file context beyond
the current file.
Conceptually, RAG is composed of two modules: a retriever and a generator. The retriever is
responsible for locating relevant pieces of information given a query, which in this domain may be
an incomplete function or surrounding code context. It searches the codebase for snippets that are
most likely useful to complete the query. These retrieved elements are then combined with the
original prompt to form an enriched context. The generator, typically a large code model (Section
2.1), consumes this augmented input and produces the final output, such as completing a function
or generating a new implementation. We can formalize this interaction for a code generation task.
Given a prompt 𝑃and a knowledge corpus 𝐾representing the target repository, the final generated
code,𝐶 final, is produced by the function:
𝐶final=G(𝑃⊕R(𝑃,𝐾))(1)
whereR(·) is the retrieval function that returns a set of relevant code chunks or documents
from𝐾. The operator⊕denotes the augmentation (e.g. concatenate) process, where the retrieved
information is integrated with the original prompt 𝑃. The generator function G(·) then synthesizes
the final code based on this augmented input.
In most prior work, the retrieval component R(·) has been approached through either sparse
syntactic methods, such as BM25, which rely on lexical overlap, or dense semantic methods, which
leverage embedding similarity to capture semantic relatedness.
(1)Syntactic sparse retrieval using BM25 [40], which relies on lexical overlap between the
query𝑞and the code snippet𝑑.
BM25(𝑞,𝑑)=∑︁
𝑡∈𝑞IDF(𝑡)·𝑓(𝑡,𝑑)·(𝑘 1+1)
𝑓(𝑡,𝑑)+𝑘 1·
1−𝑏+𝑏·|𝑑|
avgdl (2)
where:
•𝑘 1is a saturation parameter controlling the impact of term frequency (typically1 .2≤𝑘 1≤
2.0),𝑏is the length-normalization parameter (0≤𝑏≤1),
•𝑎𝑣𝑔𝑑𝑙is the average document length across the collection,
(2)Semantic dense retrieval using embedding similarityis typically implemented with a
pretrained encoder (e.g., UniXCoder [ 15] or CodeBERT [ 12]), which maps both the query 𝑞
and candidate document 𝑑into vector representations. A similarity score (commonly cosine
similarity) is then computed between these vectors to rank and retrieve the most relevant
snippets.
cosine_sim(𝑞,𝑑)=𝑞·𝑑
∥𝑞∥·∥𝑑∥(3)
where∥·∥denotes the L2 norm (also called the Euclidean norm) of a vector.
2.3 Repository-Level Code Generation
Recent advances in Large Language Models (LLMs) have demonstrated remarkable progress in
standalone code generation, where the goal is to produce a self-contained function or snippet
given a natural language description or partial code [2, 6, 20, 37]. While impressive, such settings
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 5
often oversimplify real-world software development, where code rarely exists in isolation. In
practice, completing or generating code usually requires access to repository-level context-including
functions, classes, modules, and APIs defined across multiple files. This has drawn increasing
attention from both academia and industry to the task of repository-level code generation, which
better reflects practical development scenarios and has led to the introduction of several dedicated
benchmarks [ 9,10,25,27,28,34,50]. Formally, the task can be viewed as producing the final code
given a query and relevant information drawn from the entire repository. In this work, we restrict
our study to function-level code generation in Python within a repository-level context, reflecting
the practical design of most benchmarks [ 27,28,51,52], where the query typically includes a
function signature and, optionally, its docstring.
Previous methodologies have primarily adopted the RAG framework (Section 2.2) to incorporate
cross-file context for tackling this task. For instance, RepoCoder [ 52] employs an iterative RAG
mechanism, where code generated in a previous step is used to enrich the context for the subsequent
turn. RepoFormer [ 49] introduces a Self-selective RAG, which allows the model to trigger a special
token to decide whether retrieving external context is necessary. Meanwhile, RLCoder [ 47] utilizes
reinforcement learning to train its retriever encoder, RLRetriever, aiming to close the vector space
gap between a query and its most useful context. A3-CodeGen [ 33] extracts, fuses, and feeds
three types of repository information into the LLM: local-aware, global-aware, and third-party-
library information. R2C2-Coder [ 8] introduces R2C2-Enhance, a prompt assembling method which
retrieves from the constructed candidate pool for each completion cursor position. Another method,
GraphCoder [ 35], first builds a code context graph which contains control flow, data, and control
dependence between code statements. It then performs coarse-to-fine retrieval of context-similar
code snippets using this graph. RepoMinCoder [ 30] uses an additional round of screening and
ranking based on information loss to complement the original RAG process. RepoFuse [ 32] combines
analogy context and rationale context, then compresses them into prompts with restricted size.
While these techniques have demonstrated promising results, they also highlight several per-
sistent challenges in the field: non-adaptive chunking methods that lack sufficient semantic-level
understanding of code structure, and naive similarity measures incapable of capturing the functional
relationships among programming elements.
3 Approach
3.1 Overview
Figure 1 provides an overview of the proposed approach, which operates as follows:
(1) It begins with a task description (e.g., an incomplete function) and a repository as inputs. The
repository is parsed into fine-grained components (e.g., classes, functions, variables) using an
AST parser, which preserves both the overall tree structure of the repository and the complete
implementation of each component. The codebase is then indexed using our structure-aware
indexing approach.
(2)Next, retrieval is performed in two complementary ways: (i) the Dependency-aware retriever
predicts and extracts the relevant dependencies for the target function by traversing the corre-
sponding tree nodes, guided by the current file and its imported modules, and (ii) a similarity-
based retriever (BM25) provides the top-k most similar snippets based on lexical similarity.
These two sources are then merged to form a unified context that includes both dependency
definitions and representative usage examples.
(3)Finally, the enriched context is concatenated with the task description and passed into a
generator model (e.g., LLMs) to produce the completed code.
, Vol. 1, No. 1, Article . Publication date: February 2026.

6 Minh et al.
Task description
RepositoryIndexed codebase
Structure-
aware indexingRelevant Code Snippets
Generator 
Dependency-aware
Retriever
Similarity-based RetrieverHydra
Fig. 1. Overview of our workflow. The approach integrates structure-aware indexing, dependency-aware
retrieval, and hybrid context construction to support repository-level code generation.
3.2 Structure-aware Indexing
Unlike prior approaches that segment the codebase into arbitrary text chunks, our method preserves
the semantic and structural integrity of the repository. Instead of viewing code as flat text, we
treat it as a hierarchy of components derived from the program’s Abstract Syntax Tree (AST)
using Python’s ast1module. This allows us to represent the codebase as a graph of nodes, where
each node corresponds to a fine-grained unit of code, and edges capture the relationships among
them (e.g., imports or usage links). Specifically, we focus on three fundamental structures that are
ubiquitous across programming languages:
•Function nodes, representing individual function or method implementations.
•Class nodes, encapsulating class definitions and their member methods.
•Variable nodes, including global or module-level variables that influence program behavior.
Each node preserves the complete implementation content of the corresponding component,
rather than fragmenting it across multiple chunks. For example, a function node contains the entire
body of the function, ensuring that retrieval always yields a self-contained and executable unit.
Figure 2 contrasts our structure-aware indexing with conventional chunk-based indexing. In chunk-
ing, the repository is split into sequential blocks of code lines, which often fragment semantically
related components (e.g., splitting a function across chunks or grouping unrelated code together).
This fragmentation leads to noisy retrieval and loss of critical contextual information. By contrast,
our approach organizes the repository into nodes aligned with the program’s natural structure.
Dependencies across files (e.g., imports) are explicitly captured as edges between nodes, allowing
retrieval methods to navigate not only by textual similarity but also by structural relationships.
This design enables more accurate and contextually faithful retrieval for repository-level code
generation. It ensures that the model can access the precise functions, classes, or variables required to
complete a task, while maintaining their full definitions and preserving the hierarchical organization
of the codebase.
1https://docs.python.org/3/library/ast.html
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 7
#CHUNK 1
#CHUNK 2...content
random.shuffle(chars)
#convert the shuffled list to str
return ''.join(chars)Chunking-
based
Indexing
Repo
File 1
File 2
File 3
File m......
......content
def strip_html() -> str:
   ...
   return r.sub('', input_string)
Function nodes
Variable nodes
Class nodes
#CHUNK nRepoStructure-
Aware
Indexing
import connectioncontent
CAMEL_CASE_REPLACE_RE =
re.compile(r'([a-z]|[A-
Z]+)(?=[A-Z])')
Fig. 2. Comparison of chunk-based indexing and structure-aware indexing. Our approach preserves full
functions, classes, and variables as nodes with structural links, avoiding the fragmentation and noise of
traditional chunking.
3.3 Dependency-Aware Retrieval
Motivation.Previous methods commonly rely on NLP-inspired similarity-based retrieval, such
as BM25 or embedding models like UniXCoder, to support RAG for repository-level code generation.
While these methods can surface code snippets that are lexically or semantically similar, they are
poorly suited to capture the structural and functional relationships inherent in code. In practice,
the most relevant context for generation is often not the “most similar” snippet, but rather the
precise functions, classes, or variables that the target function depends on. Recent work, such as
RepoExec [ 27], highlights that without access to ground-truth dependencies, models might produce
low-quality code and accumulate technical debt [ 18,19,31,42]. However, RepoExec primarily
establishes a benchmark and evaluation setting; it does not address the practical challenge of
how to automatically retrieve these dependencies from a large codebase. Our empirical analysis
confirms this gap: when relying solely on similarity-driven retrieval methods such as BM25 or
UniXCoder,40–60% of ground-truth dependencies required by the target function are missing from the
retrieved context. This shortfall demonstrates that similarity-based retrieval alone fails to expose
the fine-grained dependencies essential for accurate repository-level code generation.
Solution Approach.To address this challenge, we introduce a lightweight dependency-aware
retriever (DAR) that is fine-tuned to detect relevant dependencies for a given query (i.e., a function
signature and its accompanying docstring, if available). Training this retriever requires a dataset of
triplets(𝑞,𝑑+,𝑑−), where𝑞denotes the query, 𝑑+is a set of true dependencies, and 𝑑−is the set
of irrelevant candidates. To construct such data, we mined Python repositories from GitHub and
processed them using thestructure-aware indexingprocedure described in Section 3.2. The full data
preparation pipeline is detailed in Section 3.3.1. With this dataset, we fine-tune an encoder, framing
dependency detection as abinary classification taskover query–candidate pairs. Specifically, the
model is trained to predict whether a pair (𝑞,𝑑) corresponds to a valid dependency, as described in
Section 3.3.2.
3.3.1 Training Dataset Construction.
, Vol. 1, No. 1, Article . Publication date: February 2026.

8 Minh et al.
Algorithm 1Training Data Construction
Require:Filtered set of Python repositories𝑅 𝑓 𝑖𝑙𝑡𝑒𝑟𝑒𝑑 .
Ensure:Training dataset𝐷with(𝑞,𝑑+,𝑑−)triplets.
1:𝐷←∅
2:for allrepository𝑟∈𝑅 𝑓 𝑖𝑙𝑡𝑒𝑟𝑒𝑑 do
3:F,C,V←ExtractAllCodeUnits(𝑟)⊲Function, Class, and Variable Pools
4:for allfunction𝑓 𝑖∈Fdo
5:𝑓𝑖𝑙𝑒𝑠 𝑟𝑒𝑙←GetFileOf(𝑓 𝑖)∪GetImportedFilesBy(𝑓 𝑖)
6:𝑆 𝑐𝑜𝑛𝑡𝑒𝑥𝑡←FilterUnitsByFiles(F,C,V,𝑓𝑖𝑙𝑒𝑠 𝑟𝑒𝑙)
7:𝑞 𝑖←GetSignature(𝑓 𝑖)+GetDocstring(𝑓 𝑖)⊲Create the query
8:𝑑+
𝑖←AnalyzeDependenciesAST(𝑓 𝑖,𝑆𝑐𝑜𝑛𝑡𝑒𝑥𝑡)⊲Positive set
9:𝑑−
𝑖←𝑆 𝑐𝑜𝑛𝑡𝑒𝑥𝑡\(𝑑+
𝑖∪{𝑓 𝑖})⊲Negative set
10:Append(𝑞 𝑖,𝑑+
𝑖,𝑑−
𝑖)to𝐷
11:end for
12:end for
13:return𝐷
Data Source.To construct a high-quality dataset for training our retriever, we collected Python
repositories from GitHub that met the following criteria: each repository had at least 30 forks, 10
stars, and a minimum of 5 Python files, resulting in 2,864 repositories. To avoid data contamination,
we excluded any repositories that overlapped with the evaluation datasets. From this curated
collection, we extracted all function, class, and variable blocks as in Section 3.2, ensuring that the
dataset preserved the structural granularity of source code.
Triplets Construction.The training data for our retriever is organized into triplets (𝑞,𝑑+,𝑑−).
The detailed process is presented in Algorithm 1. To create the query 𝑞, we extract function nodes set
F𝑡𝑎𝑟𝑔𝑒𝑡 from the crawled repositories and use their function signature together with the docstring
(if available) as the query. Since our objective is to retrieve the dependencies that the target function
directly calls, we restrict candidates to components appearing in the same file or in its imported
files. As illustrated in Figure 2, we traverse these files through the import connections to collect all
components (functions, classes, and variables). For each anchor function 𝑓𝑖∈F𝑡𝑎𝑟𝑔𝑒𝑡 , components
that are explicitly invoked by 𝑓𝑖are added to the positive set 𝑑+
𝑖, while all others serve as negative
samples𝑑−
𝑖, to form(𝑞 𝑖,𝑑+
𝑖,𝑑−
𝑖).
3.3.2 Retriever Fine-Tuning Strategy.Once the training dataset is constructed (Section 3.3.1), we
fine-tune an encoder (UniXCoder [ 15]) to detect the true dependencies 𝑑+for a given query 𝑞. We
considered two alternative training strategies.
(1)Contrastive learning:Following common practice in code search, one option is to optimize
a contrastive loss, encouraging embeddings of (𝑞,𝑑+)pairs to be closer than (𝑞,𝑑−)pairs. At
inference, retrieval is then performed via embedding similarity (e.g., cosine similarity). This
approach is computationally efficient for large-scale retrieval.
(2)Pairwise classification:Alternatively, we treat each pair (𝑞,𝑑) with𝑑∈𝑑+∪𝑑−as an
independent input and train the encoder to classify concatenated input 𝑝⊕𝑑 into {0,1}, where
1denotes a valid dependency and0otherwise.
While the contrastive approach can offer speed advantages during inference, in our setting
it proved unstable and failed to converge, as below discussion. We therefore adopt the pairwise
classification strategy, which yields more reliable training dynamics and more accurate dependency
detection. As a result, this process yielded a dataset of approximately564 ,740data samples. The
data is divided into train/validation/test splits with a ratio of 8:1:1.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 9
def snake_case_to_camel(...)
    ...
def camel_case_to_snake (...):
    ...
    if not is_string (input_string ):
        raise InvalidInputError (input_string )
    if not is_camel_case (input_string ):
        return input_string
   ...def is_string (...):
    ...
class InvalidInputError (...):
      ...
def is_snake_case (...):
    ...
def snake_case_to_camel (...):
    ...
    if not is_string (input_string ):
        raise InvalidInputError (input_string )
    if not is_snake_case (input_string , separator ):
        return input_string
    ...Query
Solution
DAR
BM25
Fig. 3. Example of retrieved results from DAR and BM25 for a query (incomplete function under generation)
in the RepoExec dataset.
Discussion.Our experiments with contrastive learning approach showed that the training
process failed to converge. We attribute this failure to the query-dependent nature of code depen-
dencies, which creates a contradictory optimization objective. During training, the same pair of
code units may be treated as both similar and dissimilar depending on the query context, creating
an ambiguous optimization landscape. For instance, function Func_A and function Func_B may
both be valid dependencies for a query, requiring their embeddings to be pulled closer. However, for
a different query, Func_A might be a true dependency, a positive sample, while Func_B is a negative
one, forcing the model to push their embedding apart. These contradictory signals result in con-
flicting gradient updates, destabilizing the training process. In contrast, the pairwise classification
strategy circumvents this issue by evaluating each (query,candidate) pair independently.
For pairwise classification strategy, due to the natural class imbalance where negative dependen-
cies vastly outnumber positive ones, to prevent model bias, we downsampled the negative class
exclusively within the training set to match the number of positive samples, creating a balanced 1:1
ratio. The validation and test sets were kept in their original imbalanced state to ensure a realistic
evaluation.
3.4HydraRetriever - A Hybrid Approach
We hypothesize that dependency-aware retrieval provides the model with the essential “materials”
needed to implement the target function. However, this retriever might overlook the signal of
howthese components should be used in practice (e.g., how to invoke a function or in which
context a global variable is typically applied). In contrast, similarity-based retrieval can supply
complementary information: by retrieving implementations that are lexically or semantically close
to the target function, their bodies often illustrate concrete usage patterns of similar dependencies.
Indeed, as illustrated in Figure 3, when generating the snake_case_to_camel function, DAR
successfully identifies the correct dependencies that must be called. At the same time, BM25 can
find the related function camel_case_to_snake , which demonstrates how such dependencies are
invoked in practice.
To combine these strengths, we designHydraRetriever, a hybrid retrieval strategy that
integrates dependency-aware retrieval (DAR) with similarity-based retrieval (i.e., BM25). By merging
these complementary contexts,Hydraprovides the model with both the necessary building blocks
and usage examples, yielding a more complete and effective context for code generation.
, Vol. 1, No. 1, Article . Publication date: February 2026.

10 Minh et al.
Inference process ofHydra:During inference, the target function is passed through both the
BM25 retriever and the DAR.
•For BM25, we apply the standard formulation (Equation 2), following common practice by fixing
𝑘1=1.5and𝑏=0.75. We then retrieve the top-5 most similar snippets.
•For DAR, rather than using only the hard classification labels described in Section 3.3.2, we
leverage the prediction probability 𝑝of class 1 to retain candidates, thereby reducing the risk of
discarding true dependencies. To achieve this, we introduce a threshold hyperparameter 𝑇; a
candidate dependency is retained if its predicted probability 𝑝>𝑇 . The threshold 𝑇is tuned on
the validation set following the strategy described below.
Our motivation in threshold selection is to capture as many ground-truth dependencies as possible
(recall 1), while keeping the model’s ability to detect false dependencies (recall 0) at an acceptable
level. To balance this trade-off, we purpose the use of the Balanced Recall Penalty (BRP) score to
select the optimal threshold. This metric’s core function is to apply a penalty to the divergence
between recall 1and recall 0, scaled by a coefficient 𝛼that reflects the class imbalance. The metric
is defined as:
BRP=recall 1−𝛼(recall 1−recall 0)2
The penalty coefficient is calculated based on the class distribution as
𝛼=1
⌊𝑁class 1+𝑁class 0
𝑁class 1⌋
We performed a grid search using BRP on the validation set, which was constructed in Section
3.3.2 with 46,141 negative instances (false dependencies) and 10,333 positive samples (true
dependencies). Due to the class imbalance, the value of𝛼is thus determined to be 0.2.
Table 1. BRP scores at different threshold values𝑇on the validation set.
Threshold0.15 0.200.250.30 0.35 0.40 0.45 0.50
BRP0.8829 0.88520.88610.8838 0.8800 0.8755 0.8660 0.8522
As shown in Table 1, a threshold of 0.25 yields the highest BRP score of 0.8861. Therefore, we
select𝑇=0.25for ourHydraRetriever during inference.
4 Experiment Setup
4.1 Research Questions and Setup
Our evaluation investigates the following research questions:
RQ1:How effective is Structure-Aware Indexing compared to Chunking-Based Indexing?
↩→Setup:For chunking-based indexing, we follow Zhang et al . [52] and segment each code file
into fixed-size chunks of 2048 tokens with a 50% overlap. In contrast, structure-aware indexing
is performed as described in Section 3.2. For both indexing strategies, we employ the same BM25
retriever and Qwen2.5-Coder generator, and evaluate them on function-level code generation
within a repository-level context.
RQ2:How do different retrieval approaches affect repository-level code generation performance?
↩→Setup:In this RQ, we first evaluate the effectiveness of our proposed dependency-aware
retriever (DAR) compared to prior similarity-based methods, including sparse retrieval with BM25
and dense retrieval with UniXCoder, in correctly retrieving the dependencies required to complete
a function. For fairness, both strategies are applied over structure-aware indexing and evaluated on
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 11
the test set described in Section 3.3.1, together with two considered benchmarks. We then extend
the comparison to assess how these three retrieval methodologies, together withHydra, affect
performance on the repository-level code generation task.
RQ3:How effective isHydracompared to existing state-of-the-art approaches for repository-level code
generation?
↩→Setup:We compareHydraagainst three state-of-the-art repository-level code generation
approaches—RepoCoder [ 52], RLCoder [ 47], and Repoformer [ 49] (Section 2.3); and further analyze
their testing logs to provide a deeper assessment of output quality. In addition, we conduct an
ablation study by removing functions, classes, and variables from Hydra’s search space to quantify
the importance of granular components.
RQ4:How does the computational cost of runningHydracompare to state-of-the-art approaches for
repository-level code generation?
↩→Setup:We compare the retrieval time ofHydraagainst the baselines considered in RQ3 to
demonstrate the practical efficiency of our approach. We report the mean, median, and maximum
retrieval times across benchmarks to capture not only average efficiency but also the stability of
each method under different query complexities.
4.2 Benchmarks, Generator Backbones and Metrics
Benchmarks.In our evaluation, we utilize two recent benchmarks designed for repository-level
code generation:
•RepoExec[ 27], a benchmark designed for the evaluation of repository-level code generation,
with a focus on executability, functional correctness, and dependency utilization. The bench-
mark includes 355 problems. The core task challenges a model to generate code by effectively
integrating provided code dependencies within a repository context. RepoExec provides an
executable environment and a mechanism to automatically generate high-coverage test cases
to robustly assess the functional correctness of the generated code.
•DevEval[ 28], a manually annotated benchmark created to closely align with real-world
coding practices. DevEval comprises 1,825 test samples from 117 Python repositories across
10 popular domains. The core task is repository-level code generation, where a model is
provided with a function signature, a detailed natural language requirement, and the full
context of the repository.
Generator Backbones.For the generator models, we use the open-source Qwen2.5-Coder [ 22]
with 1.5B and 7B parameters, as well as the closed-source GPT-4.1-mini [ 39]. This setup allows us to
evaluate the adaptability of our method across different model types and scales. During generation,
we employ nucleus sampling and generate 5 candidate outputs per query for all models.
Metrics.To evaluate the functional correctness of the generated code, we employ the widely-
adopted execution-based metric, Pass@k [ 6], with k∈ {1,3,5}. This metric assesses whether
the generated code can successfully pass a set of predefined unit tests. In addition, we leverage
the Dependency Invocation Rate (DIR) introduced by Le Hai et al . [27] to evaluate whether the
generated code correctly calls the intended dependencies. This metric is computed as the size of
the intersection between the set of dependencies invoked in the generated solution and those in
the ground truth, divided by the total number of ground-truth dependencies.
5 Evaluation
In this section, we present the empirical results of our investigation, along with an in-depth analysis
of our findings regarding the research questions.
, Vol. 1, No. 1, Article . Publication date: February 2026.

12 Minh et al.
class FileSystem (metaclass =abc.ABCMeta):
        ...
    def exists(self, path):
        ...
    def isdir(self, path):
        """
        Return ``True`` if the location at ``path`` is a directory. 
        If not, return ``False``.
        ...
        """
        raise NotImplementedError ("isdir() not implemented on {0} "\
              .format (self.__class__ .__name__ ))
    def listdir (self, path):
        """Return a list of files rooted in path.
        ...
        """
        
        raise  NotImplementedError ("listdir() not implemented on {0} "\
        .format (self.__class__ .__name__ ))
    def move(self, path, dest):
        ...
    def rename_dont_move (self, path, dest):
        ...def _list_existing (filesystem , glob, paths):
    ...
    if isinstance (paths, dict):
        for key, value_list in paths.items():
            for value in value_list :
                if filesystem .isdir(value):
                    existing_paths .update(filesystem .listdir(value))
                elif filesystem .isfile(value):
                    existing_paths .add(value)
    ....
AttributeError: 'LocalFileSystem'
object has no attribute 'isfile'
Missing ContextQwen2.5-Coder-1.5B-Instruct Solution
Available Context Fed Into GeneratorMissing Context
Ground-truth Callee
Retrieved
Chunk
Fig. 4. Example of contextual fragmentation under chunk-based retrieval. The retriever provides partial context
(isdir ,listdir ) but misses the ground-truth dependency exists . As a result, the model hallucinates an
isfilemethod, leading to anAttributeError.
Table 2. Results of chunking-based vs. structure-aware context indexing on RepoExec and DevEval benchmarks
with BM25 retriever, Qwen2.5-Coder-Instruct 1.5B and 7B versions as generator, reported in Pass@1/3/5.
Model Size Benchmark Indexing Strategy Pass@1 Pass@3 Pass@5
1.5BRepoExecChunking-based Indexing 9.13 14.96 17.75
Structure-Aware Indexing 12.39 16.70 18.02
DevEvalChunking-based Indexing 4.81 6.60 7.34
Structure-Aware Indexing 10.32 14.13 15.89
7BRepoExecChunking-based Indexing 14.03 19.32 21.69
Structure-Aware Indexing 20.79 27.61 30.14
DevEvalChunking-based Indexing 8.46 10.88 11.78
Structure-Aware Indexing 16.67 21.73 23.78
RQ1: How effective is Structure-Aware Indexing compared to Chunking-Based Indexing?
To address this RQ, we evaluate repository-level code generation under two alternative context
formulation strategies: chunking-base indexing and our proposed structure-aware context indexing.
The results are presented in Table 2.
Quantitative Findings.The results demonstrate a consistent and substantial advantage for
structure-aware context indexing. Across both model scales (1.5B and 7B) and benchmarks (Re-
poExec and DevEval), node-based context significantly outperforms chunk-based baselines. The
performance gap is most pronounced on the challenging DevEval benchmark-for example, with
QwenCoder-7B-Instruct, Pass@1 nearly doubles when using node extraction (16.67% vs. 8.46%). On
RepoExec, we also observe steady improvements of 3–6% in Pass@1, highlighting that this strategy
is robust across repositories with diverse structures. These findings validate our hypothesis that
preserving the syntactic and semantic integrity of code units yields a more coherent and complete
view of the codebase, thereby improving the likelihood of correct generation.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 13
Qualitative Analysis: A Case Study on Contextual Fragmentation.Quantitative metrics
alone cannot fully explain why chunking degrades retrieval quality. To illustrate the failure modes,
Figure 4 presents a case study on completing the _list_existing function, which relies on
theFileSystem object. The chunk-based retriever captures fragments containing the isdir and
listdir methods but fails to retrieve the critical exists method. Without this context, the model
generates filesystem.isfile(path) , a plausible but hallucinated method inferred by analogy
toisdir . The resulting AttributeError , illustrated in the traceback, is a direct consequence of
semantic fragmentation: the retriever supplies incomplete and misleading context, which forces
the model into an erroneous inference.
[RQ1] Summary:Unlike natural language, code cannot be reliably segmented into arbitrary
chunks without breaking logical boundaries. Chunk-based indexing often fragments depen-
dencies, yielding incomplete or misleading context that leads models to hallucinate calls and
produce incorrect code. Structure-aware context indexing instead preserves functions, classes,
and variables as coherent units, providing a more faithful representation of the repository. This
approach significantly improves retrieval quality and code generation, surpassing chunking-
based strategies by a wide margin, with around2×gains on DevEval.
RQ2: How do different retrieval approaches affect repository-level code generation
performance?
Table 3. Comparison of retrieval methods. The top section shows overall performance in terms of Recall,
Precision, and F1. The bottom section provides a fine-grained breakdown of Recall by different dependency
types, functions (FRecall), classes (CRecall), and variables (VRecall)
RetrieverRepoExec DevEval
Recall Precision F1 Recall Precision F1
Sparse Retriever 0.5378 0.0809 0.1406 0.5734 0.1243 0.2044
Dense Retriever 0.5483 0.0824 0.1432 0.5183 0.1124 0.1848
Dependency-aware Retriever0.9223 0.1086 0.1957 0.8884 0.1559 0.2652
FRecall CRecall VRecall FRecall CRecall VRecall
Sparse Retriever 0.6522 0.6667 0.3174 0.7216 0.6562 0.3227
Dense Retriever 0.6957 0.5196 0.3832 0.7557 0.5191 0.3506
Dependency-aware Retriever0.9420 0.8922 0.9162 0.8718 0.9194 0.8812
First, we evaluate the effectiveness of the proposed Dependency-Aware Retrieval (DAR) in com-
parison to sparse and dense retrieval methods for acquiring the dependencies required to implement
a target function. Experiments are conducted on our constructed test dataset (Sections 3.3.1 and
3.3.2) as well as on the RepoExec and DevEval benchmarks. We report Precision, Recall, and F1
for true dependency class. Since our primary objective is to ensure that true dependencies are not
missed, Recall on the positive class (relevant dependencies) serves as the main evaluation metric.
At the same time, retrieving irrelevant dependencies introduces noise, which reflects the model’s
ability to correctly filter candidates. Although such noise should be mitigated, it also provides
insight into how effectively the retriever selects useful materials for the generator.
Table 3 reports the performance of sparse, dense, and dependency-aware retrieval (DAR) in
detecting true dependencies. On RepoExec, both sparse and dense retrievers achieve moderate
recall (around 0.53–0.55), but their very low precision (<0.09) shows that many retrieved candidates
, Vol. 1, No. 1, Article . Publication date: February 2026.

14 Minh et al.
Table 4. Code generation results with different retrieval strategies on RepoExec and DevEval benchmarks
using Qwen2.5-Coder models (1.5B and 7B) as generator. Results are reported in Pass@1/3/5 and Dependency
Invocation Rate (DIR).
Model Size RetrieverRepoExec DevEval
Pass@1 Pass@3 Pass@5 DIR Pass@1 Pass@3 Pass@5
1.5BSparse Retriever 12.39 16.70 18.02 39.93 10.32 14.13 15.89
Dense Retriever 12.73 19.04 21.40 39.86 10.24 13.89 15.56
Dependency-aware Retriever 13.46 18.02 20.28 43.81 10.4314.58 16.22
HydraRetriever15.72 21.30 23.38 44.61 10.7114.50 16.05
7BSparse Retriever 20.79 27.61 30.14 49.67 16.67 21.73 23.78
Dense Retriever 19.89 26.45 29.58 46.38 15.93 20.92 22.79
Dependency-aware Retriever 21.97 29.89 32.96 52.24 16.84 21.93 23.95
HydraRetriever23.32 31.32 34.36 53.46 17.27 22.44 24.44
are spurious. In contrast, DAR reaches a recall of 0.92, substantially higher than either baseline,
confirming its ability to capture the majority of ground-truth dependencies. Although precision
remains modest, this trade-off is acceptable: for code generation, failing to retrieve true dependencies
is far more damaging than including some irrelevant context. On DevEval, the results are consistent.
Sparse and dense methods again show moderate recall (0.51–0.57) and low precision (0.11–0.12),
while DAR achieves a recall of 0.89 with the highest F1 among all methods. These gains underscore
the robustness of DAR across different benchmarks. Overall, the results validate our design choice
to prioritize recall of true dependencies, ensuring that the generator is supplied with the essential
building blocks for producing correct and executable code, even if some noise is introduced.
We further conduct a fine-grained analysis of retrieval performance across different dependency
types, namely functions, classes, and variables. As shown in Table 3, similarity-based retrievers
achieve their highest recall on function dependencies (0.65–0.75). This is understandable since
the task is framed as function-level completion and many pretrained models are optimized using
function-level code snippets [ 15,24,37,46]. These retrievers struggle more with class dependencies,
though their recall remains at a moderate level. In contrast, variables remain the most challenging
to retrieve for similarity-based approaches, with recall below0 .4. To our knowledge, variables
have received little explicit consideration in prior work, either during pretraining or fine-tuning of
retrieval and code generation models. Nevertheless, variables-particularly constants and configura-
tion values-are frequently leveraged in implementing target functions. Missing these dependencies
not only reduces generation accuracy but also risks overlooking the underlying human intent of
the code. By comparison, the proposed DAR achieves consistently high recall across all dependency
types, reaching 0.94/0.89/0.92 on RepoExec and 0.87/0.92/0.88 on DevEval for functions, classes,
and variables, respectively. This balanced coverage highlights the robustness of DAR: It not only
significantly improves performance on function type, where similarity-based methods already
perform best, but also substantially improves retrieval for classes and variables, which are critical
for repository-level code generation but often missed by similarity-driven approaches.
Additionally, we evaluate our method within the full RAG pipeline for repository-level code
generation, in order to facilitate a comprehensive assessment of its effectiveness. Table 4 reports
code generation results on RepoExec and DevEval using different retrieval strategies with Qwen2.5-
Coder models (1.5B and 7B). We can observe that DAR achieves consistently higher Pass@k
scores than similarity-based retrievers, indicating that providing true dependencies offers greater
value to the generator than merely retrieving lexically or semantically similar snippets. Besides,
it also achieves higher DIR compared to sparse and dense methods (e.g., 43.81 vs. 39.9 at 1.5B,
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 15
Fig. 5. Case study comparing the generated code byHydraand text-based retrieval on task_id 18 (RepoExec).
(a) Target Function Prompt
1def is_url(input_string: Any, allowed_schemes: Optional[List[str]] = None) -> bool:
2"""
3Check if a string is a valid URL.
4[...]
5"""
(b) Solution generated withHydra
1def is_url(input_string: Any, allowed_schemes: Optional[List[str]] = None) -> bool:
2[...]
3if not is_full_string(input_string):
4return False
5[...]
(c) Solution generated with BM25
1def is_url(input_string: Any, allowed_schemes: Optional[List[str]] = None) -> bool:
2[...]
3if not isinstance(input_string, str):
4raise InvalidInputError(input_string)
5[...]
and 52.24 vs. 46–49 at 7B), confirming its superior ability to recover true dependencies, thereby
reducing the risk of code smells and technical debt. However, the performance gap remains modest,
suggesting that similarity-based context still provides complementary value. Indeed, the hybrid
HydraRetriever consistently delivers the best overall performance. At the 1.5B scale,Hydra
raises RepoExec Pass@1 to 15.72, surpassing sparse (12.39), dense (12.73), and DAR (13.46) by more
than 2%. Similar improvements are observed in Pass@3, Pass@5, and DIR metrics. On the more
challenging DevEval benchmark,Hydraachieves Pass@1 scores of 10.71 and 17.27 with the 1.5B
and 7B models, respectively, outperforming all baselines. These results demonstrate that similarity-
based context complements dependency context by providing signals onhowdependencies are
used, thereby helping the generator produce code that is both functionally correct and aligned with
the current repository state, further strengthening our hypothesis and discussion in Section 3.4.
Case Study:We demonstrate Hydra’s robustness over text-based retrievers through a case study
in Figure 5. In RepoExec ( task_id 18), completing is_url requires validating that the input is a
proper string via the auxiliary function is_full_string .Hydra, using DAR, successfully retrieves
this dependency, allowing the LLM to reuse the existing API and produce a correct implementation.
In contrast, similarity-based retrievers miss this function, forcing the LLM to re-implement the logic
and often deviate from the intended behavior. This highlightsHydra’s advantage in retrieving
critical dependencies that text-based retrievers overlook.
, Vol. 1, No. 1, Article . Publication date: February 2026.

16 Minh et al.
[RQ2] Summary:Similarity-based retrieval methods, such as sparse retrieval or dense retrieval
with a general encoder, frequently overlook a substantial portion of required dependencies-
particularly variables and classes. In contrast, the proposed Dependency-Aware Retrieval (DAR)
demonstrates greater robustness in accurately identifying candidate dependencies.
Furthermore, DAR consistently achieves higher Pass@k and DIR than similarity-based retrievers,
showing that true dependencies provide greater value to the generator and help reduce risks such
as code smells and technical debt. Nonetheless, similarity-based context still adds complementary
usage signals, and the hybridHydraRetriever leverages both to achieve the strongest results.
RQ3: How effective isHydracompared to existing state-of-the-art approaches for
repository-level code generation?
Hydravs. Baselines.Table 5 comparesHydrawith several representative baselines, including
RepoCoder, RepoFormer, and RLCoder, as well as no-context generation. Across all model families
and sizes,Hydraconsistently delivers the best performance, establishing itself as a new state
of the art in repository-level code generation. Notably, while existing methods already improve
over the no-context baseline by supplying additional cross-file information, their reliance on
chunking or similarity-based retrieval limits the completeness and utility of the retrieved context.
In contrast,Hydra’s hybrid strategy-anchoring generation on true dependencies, augmenting
them with similarity-based usage signals, and leveraging structure-aware indexing-supplies the
model with both the essential building blocks and clear guidance on how to apply them effectively.
This advantage holds for both open-source and closed-source models, and across benchmarks,
underscoring the robustness and generalizability of our approach.
A particularly striking result is that, on both benchmarks, the 1.5B model equipped with our
Hydraretriever surpasses much larger baselines: it outperforms RepoCoder on RepoExec and
even exceeds RepoFormer on DevEval with a 7B model (15.72 vs. 14.82 and 10.71 vs. 10.41 in
Pass@1, respectively). This demonstrates that high-quality retrieval can narrow or even bridge the
performance gap between small and large models, highlighting the potential value of our approach
in resource-constrained settings.
Failure Modes of Generated Code.We further investigate the failure modes of generated code
and present the error statistics in Figure 6. Compared to baselines (No Context, RepoCoder, Repo-
Former, and RLCoder),Hydrasignificantly reduces critical errors such as NameError, TypeError,
and AttributeError. These error categories are particularly indicative of model hallucination, where
the generator fabricates dependencies that do not exist in the repository. The substantial reduction
demonstrates the effectiveness ofHydrain preserving code structure, retrieving true dependencies,
and providing accurate usage guidance, thereby helping the model avoid spurious or inconsistent
component calls.
At the same time, we observe a relative increase in AssertionError. Rather than signaling degraded
generation quality, this rise stems from richer test execution coverage: with correct dependencies
in place, the generated functions are more likely to run and reach assertion checks, exposing deeper
logical mismatches rather than failing prematurely due to missing symbols. This trade-off suggests
thatHydrashifts errors from structural failures toward more nuanced behavioral validation,
reflecting an important step toward generating executable and repository-consistent code.
Ablation study.To comprehend the individual contributions of each component type to Hydra’s
effectiveness, we conduct an ablation study by systematically removing function-level, class-level,
and variable-level retrieval from the search space. Table 6 presents the results across both bench-
marks using Qwen2.5-Coder-1.5B-Instruct and Qwen2.5-Coder-7B-Instruct. Here,“w/o-Function”,
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 17
Table 5. Comparison ofHydrawith prior retrieval-based approaches and no-context baselines. Results are
reported in Pass@1/3/5 using close sourced GPT-4.1 mini and open source Qwen2.5-Coder-1.5B and 7B
Instruct models.
Generator MethodRepoExec DevEval
Pass@1 Pass@3 Pass@5 Pass@1 Pass@3 Pass@5
GPT-4.1 miniNo Context 21.58 24.42 25.63 19.72 23.19 24.71
RepoCoder 22.20 26.08 27.89 17.48 23.15 25.70
RepoFormer 39.15 42.42 43.94 30.89 34.21 35.40
RLCoder 38.14 42.17 43.38 29.46 32.76 34.14
Hydra43.55 45.72 46.48 31.91 35.56 36.99
QwenCoder-1.5B-InstructNo Context 5.75 8.31 9.30 3.53 5.20 5.97
RepoCoder 7.15 11.72 14.37 4.54 8.08 9.81
RepoFormer 11.15 16.42 18.87 5.58 7.94 8.99
RLCoder 14.87 21.0423.949.34 12.90 14.47
Hydra15.72 21.3023.3810.71 14.50 16.05
QwenCoder-7B-InstructNo Context 13.30 17.04 18.03 7.10 9.16 10.03
RepoCoder 14.82 21.99 25.07 6.39 10.63 12.82
RepoFormer 17.69 25.04 28.45 10.41 13.68 14.90
RLCoder 20.17 23.69 27.61 13.00 17.67 19.61
Hydra23.32 31.32 34.36 17.27 22.44 24.44
Fig. 6. Distribution of error types in generated code for different repository-level code generation methods,
evaluated with Qwen2.5-Coder-1.5B-Instruct on RepoExec.
AssertionError NameError TypeError AttributeError SyntaxError ValueError0100200300400500600Error counts651
283
179
138
68 71585
489
217
163
5581528
493
221
172
75
54599
351
195
161
96
52538
364
152190
100
69Hydra (Ours)
No Context
RepoCoder
RepoFormer
RLCoder
“w/o-Class”, and“w/o-Variable”denote variants where function-level, class-level, and variable-level
components are respectively removed from the search space ofHydra. From Table 6, we observe
that removing any retrieval granularity consistently degrades performance across both bench-
marks and model scales. On RepoExec and DevEval, ablating function-level and class-level retrieval
leads to a pronounced performance drop, with Pass@1 decreasing by up to approximately 4%.
In contrast, ablating variable-level retrieval results in a comparatively smaller degradation, with
Pass@1 typically dropping by around 1-2%. These results indicate that function- and class-level
, Vol. 1, No. 1, Article . Publication date: February 2026.

18 Minh et al.
Table 6. Ablation study evaluating the contribution of different granular retrieval components in Hydra,
including function-level, class-level, and variable-level retrieval.
Benchmark MethodQwenCoder-1.5B-Instruct QwenCoder-7B-Instruct
Pass@1 Pass@3 Pass@5 Pass@1 Pass@3 Pass@5
RepoExecHydra15.72 21.30 23.38 23.32 31.32 34.36
w/o-Function13.12 15.69 16.62 17.97 24.54 27.32
w/o-Class11.94 17.18 19.15 20.73 27.15 30.14
w/o-Variable13.01 17.86 20.00 21.01 28.17 30.42
DevEvalHydra10.71 14.50 16.05 17.27 22.44 24.44
w/o-Function8.58 12.43 14.30 14.76 19.58 21.48
w/o-Class8.24 12.03 14.02 14.15 19.04 21.21
w/o-Variable9.21 13.35 15.29 15.72 20.07 21.81
retrieval provide more essential API and structural signals, while variable-level retrieval plays a
complementary role.
[RQ3] Summary:Hydraestablishes a new state of the art in repository-level code generation,
consistently surpassing baselines such as RepoCoder, RepoFormer, and RLCoder across models
and benchmarks. Its hybrid design, combining dependency-aware retrieval, similarity-based
usage signals, and structure-aware indexing-allows smaller models to surpass larger ones,
highlighting the value of high-quality retrieval. Error analysis further shows thatHydrareduces
hallucination-related failures while shifting errors toward deeper behavioral checks, leading to
more executable and repository-consistent code. Additionally, The ablation study shows that
Hydra benefits from all granular retrieval components, with function- and class-level retrieval
contributing most, while variable-level retrieval provides complementary gains that further
improve generation quality.
RQ4: How does the computational cost of runningHydracompare to state-of-the-art
approaches for repository-level code generation?
In this RQ, we evaluate the retrieval latency of different methods to assess their practicality for
repository-level code generation. Table 7 compares the retrieval latency of different repository-level
code generation methods across benchmarks. The results highlight clear differences in efficiency
stemming from how each method structures its retrieval process.
RepoCoder exhibits by far the highest latency, with maximum times extending to several tens
of seconds. This inefficiency stems from its iterative RAG design, where the retriever repeatedly
queries and refines the context through multiple interactions with the codebase, leading to high
computational cost. In contrast, RLCoder achieves relatively reasonable efficiency, with most
queries completing under one second. However, it still suffers from outliers, reflected in large
gaps between mean and median latencies, suggesting instability in retrieval time. RepoFormer
demonstrates the fastest median retrieval time across both benchmarks, benefiting from its selective
retrieval strategy, which invokes RAG only when the model’s generation is predicted to benefit
from additional context. However, this comes at the cost of occasionally very high maximum
latencies, reflecting the overhead when retrieval is triggered.
By comparison, our proposedHydraachieves consistently low latency while maintaining stable
performance across queries, owing to its context-aware retrieval design. Instead of exhaustively
scanning the entire repository,Hydraanchors retrieval on call-graph dependencies at the function
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 19
Table 7. Latency comparison (in milliseconds per example) of different repository-level code generation
methods. Results are reported in terms of minimum, maximum, mean, and median latency.
MethodRepoExec DevEval
Min Max Mean Median Min Max Mean Median
RLCoder 91.71 2927.08 644.92 347.16 91.491824.29894.40 821.43
RepoFormer 13.59 27286.85 1766.2818.4412.40 34336.94 4432.4118.86
RepoCoder 919.64 34072.90 7212.16 5201.46 566.52 24306.62 5096.50 4618.88
Hydra0.01 1836.32 362.95277.800.0121234.01432.86221.22
and file levels, restricting the search space to the current file and its imported modules (via import
connections in Figure 2). This is particularly effective in practice, as many target functions reside
in leaf files that do not import other modules, resulting in a very small candidate pool or even no
candidate dependencies. In contrast, prior works traverse the entire codebase without considering
dependency structure, resulting in unnecessary computation over large amounts of irrelevant
code, higher average latency, and frequent extreme outliers. Moreover,Hydraincorporates a
lightweight binary classification step to predict whether a candidate dependency is relevant to
the target function, further narrowing the retrieved context and reducing redundant processing.
From a complexity perspective,Hydraincurs a retrieval cost of 𝑂(𝑐) , where𝑐corresponds to the
size of the current file and its imported dependencies, whereas existing approaches exhibit an
𝑂(𝑛) cost with respect to the full repository size. Since 𝑛grows much faster than 𝑐in real-world
codebases, this gap becomes increasingly pronounced as repositories scale. Consequently,Hydra
delivers substantial retrieval speedups without sacrificing generation quality, making it well-suited
for practical repository-level code generation scenarios where both accuracy and low latency are
critical.
[RQ4] Summary:Hydradelivers both state-of-the-art accuracy and the lowest, most stable
latency by anchoring retrieval on dependencies and narrowing the search space to relevant files.
6 Threats to Validity
Our study, while demonstrating the effectiveness ofHydra, is subject to several potential threats
to internal and external validity.
6.1 Internal validity
Retriever Backbone Choice:A potential threat lies in the choice of backbone model for the
retriever. In this work, we only evaluate UniXCoder as the encoder for dependency detection.
While UniXCoder provides a strong baseline and stable convergence in our setting, alternative
backbones (e.g., CodeBERT [ 12], GraphCodeBERT [ 16], or newer code-specific encoders) might
lead to different retrieval effectiveness. As a result, our conclusions about retriever performance
may be partly tied to this specific backbone choice.
Data Contamination:For the Hydra retriever, we filter out any repositories that overlap with
the evaluation datasets during training data construction, preventing direct data leakage. Besides,
we computed the entropy of DAR’s predictions on RepoExec and DevEval. The relatively high
average entropy (0.3634 on RepoExec and 0.4035 on DevEval, with a maximum of 0.6) suggests
that DAR is not biased toward memorized patterns and is minimally affected by potential data
contamination. Regarding the risk of generator (LLMs) being pretrained on the evaluation data, we
, Vol. 1, No. 1, Article . Publication date: February 2026.

20 Minh et al.
empirically show that plain LLMs without Hydra’s retrieval module perform substantially worse
than Hydra across all settings. In particular, removing retrieval leads to performance drops of
over 10% on the QwenCoder variants and over 20% on GPT-4.1-mini. These results indicate that
Hydra’s performance gains primarily stem from its retrieval mechanism rather than memorization,
mitigating the potential data contamination.
6.2 External validity
Scope Limited to Python Benchmarks:Our experiments are restricted to Python and two
benchmarks, RepoExec and DevEval. While these benchmarks represent realistic and challenging
settings, results may not directly generalize to other programming languages, ecosystems, or
repository structures. The focus on function-level generation, while aligned with benchmark
design, also narrows the scope, leaving the effectiveness ofHydrafor higher-level tasks (e.g., class
or module generation) as future work.
Limited choice of LLMs:Our experiments involve two types of LLMs as the generator: Qwen2.5-
Coder and GPT-4.1 mini. Although our method is model-agnostic, evaluation on a broader range
of models lies beyond the scope of this work. We believe, however, that our selected models
are representative, covering both state-of-the-art open-source and closed-source LLMs for code
generation.
7 Conclusion
In this work, we introducedHydra, a framework for repository-level code generation that leverages
structure-aware indexing, dependency-aware retrieval, and a hybrid strategy combining depen-
dencies with similarity-based signals. Experiments on two challenging benchmarks, RepoExec
and DevEval, show thatHydraconsistently surpasses baselines, establishes a new state of the
art, and even enables small models to rival larger ones. Ablation studies further confirm the ef-
fectiveness of incorporating granular context at multiple levels, including functions, classes, and
variables, which contributes substantially to performance gains. Error analysis reveals thatHy-
drareduces hallucination-related failures and shifts errors toward deeper behavioral validation,
resulting in more executable and repository-consistent code. Moreover, our time complexity and
latency evaluation demonstrate thatHydraachieves practical efficiency by restricting retrieval
to dependency-localized context rather than repository-wide scanning. Overall, these findings
highlight the importance of moving beyond NLP-inspired chunking and similarity retrieval toward
dependency-centered approaches for practical and reliable repository-level code generation.
8 Data Availability
A replication package that contains the source code and datasets for this paper is available at
https://github.com/solis-team/Hydra
Acknowledgement
Nam Le Hai was funded by the PhD Scholarship Programme of Vingroup Innovation Foundation
(VINIF), VinUniversity, code VINIF.2025.TS68.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 21
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,
Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al .2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774
(2023).
[2]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang,
Carrie J. Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program Synthesis with Large Language Models.
arXiv:2108.07732 [cs.PL] https://arxiv.org/abs/2108.07732
[3]Bowen Cao, Deng Cai, Leyang Cui, Xuxin Cheng, Wei Bi, Yuexian Zou, and Shuming Shi. 2024. Retrieval is Accurate
Generation. InThe Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11,
2024. OpenReview.net. https://openreview.net/forum?id=oXYZJXDdo7
[4]Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho
Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, et al .2023. Multipl-e: A scalable and polyglot approach to
benchmarking neural code generation.IEEE Transactions on Software Engineering49, 7 (2023), 3675–3691.
[5]Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. [n. d.]. CodeT:
Code Generation with Generated Tests. InThe Eleventh International Conference on Learning Representations.
[6]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards,
Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf,
Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser,
Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert,
Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak,
Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan
Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie
Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021.
Evaluating Large Language Models Trained on Code. arXiv:2107.03374 [cs.LG] https://arxiv.org/abs/2107.03374
[7]Rodrigo Fernandes Gomes da Silva, Chanchal K Roy, Mohammad Masudur Rahman, Kevin A Schneider, Klérisson
Paixão, Carlos Eduardo de Carvalho Dantas, and Marcelo de Almeida Maia. 2020. CROKAGE: effective solution
recommendation for programming tasks by leveraging crowd knowledge.Empirical Software Engineering25, 6 (2020),
4707–4758.
[8]Ken Deng, Jiaheng Liu, He Zhu, Congnan Liu, Jingxin Li, Jiakai Wang, Peng Zhao, Chenchen Zhang, Yanan Wu,
Xueqiao Yin, Yuanxing Zhang, Wenbo Su, Bangyu Xiang, Tiezheng Ge, and Bo Zheng. 2024. R2C2-Coder: Enhanc-
ing and Benchmarking Real-world Repository-level Code Completion Abilities of Code Large Language Models.
arXiv:2406.01359 [cs.CL] https://arxiv.org/abs/2406.01359
[9]Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan,
Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. 2023. CrossCodeEval: A Diverse and Multilingual
Benchmark for Cross-File Code Completion. InAdvances in Neural Information Processing Systems 36: Annual Conference
on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh,
Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.). http://papers.nips.cc/paper_
files/paper/2023/hash/920f2dced7d32ab2ba2f1970bc306af6-Abstract-Datasets_and_Benchmarks.html
[10] Yangruibo Ding, Zijian Wang, Wasi U Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia,
Dan Roth, and Bing Xiang. 2024. CoCoMIC: Code Completion by Jointly Modeling In-file and Cross-file Context. In
Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024). 3433–3445.
[11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitan-
sky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[12] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu,
Daxin Jiang, and Ming Zhou. 2020. CodeBERT: A Pre-Trained Model for Programming and Natural Languages. In
Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020 (Findings
of ACL, Vol. EMNLP 2020), Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics,
1536–1547. doi:10.18653/V1/2020.FINDINGS-EMNLP.139
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen
Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[14] Wenchao Gu, Juntao Chen, Yanlin Wang, Tianyue Jiang, Xingzhe Li, Mingwei Liu, Xilin Liu, Yuchi Ma, and Zibin
Zheng. 2025. What to Retrieve for Effective Retrieval-Augmented Code Generation? An Empirical Study and Beyond.
arXiv preprint arXiv:2503.20589(2025).
, Vol. 1, No. 1, Article . Publication date: February 2026.

22 Minh et al.
[15] Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. UniXcoder: Unified Cross-Modal
Pre-training for Code Representation. InProceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, Smaranda Muresan, Preslav Nakov, and
Aline Villavicencio (Eds.). Association for Computational Linguistics, 7212–7225. doi:10.18653/V1/2022.ACL-LONG.499
[16] Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie LIU, Long Zhou, Nan Duan, Alexey Svyatkovskiy,
Shengyu Fu, et al .2021. GraphCodeBERT: Pre-training Code Representations with Data Flow. InInternational
Conference on Learning Representations.
[17] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli
Luo, Yingfei Xiong, and Wenfeng Liang. 2024. DeepSeek-Coder: When the Large Language Model Meets Programming
– The Rise of Code Intelligence. arXiv:2401.14196 [cs.CL] https://arxiv.org/abs/2401.14196
[18] Nam Le Hai, Anh M. T. Bui, Phuong T. Nguyen, Davide Di Ruscio, and Rick Kazman. 2025. Detection of Technical
Debt in Java Source Code. arXiv:2411.05457 [cs.SE] https://arxiv.org/abs/2411.05457
[19] Nam Le Hai and Nghi D. Q. Bui. 2024. Dopamin: Transformer-based Comment Classifiers through Domain Post-
Training and Multi-level Layer Aggregation. InProceedings of the Third ACM/IEEE International Workshop on NL-Based
Software Engineering(Lisbon, Portugal)(NLBSE ’24). Association for Computing Machinery, New York, NY, USA, 61–64.
doi:10.1145/3643787.3648044
[20] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir
Puranik, Horace He, Dawn Song, and Jacob Steinhardt. 2021. Measuring Coding Challenge Competence With APPS.
InProceedings of the NeurIPS Datasets and Benchmarks Track (Round 2), NeurIPS. Available in the NeurIPS Datasets
and Benchmarks Proceedings.
[21] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming
Lu, et al. 2024. Qwen2. 5-coder technical report.arXiv preprint arXiv:2409.12186(2024).
[22] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming
Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan,
Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. 2024. Qwen2.5-Coder Technical
Report. arXiv:2409.12186 [cs.CL] https://arxiv.org/abs/2409.12186
[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda,
Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card.arXiv preprint arXiv:2410.21276(2024).
[24] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Codesearchnet
challenge: Evaluating the state of semantic code search.arXiv preprint arXiv:1909.09436(2019).
[25] Naman Jain, Manish Shetty, Tianjun Zhang, King Han, Koushik Sen, and Ion Stoica. 2024. R2e: Turning any github
repository into a programming agent environment. InForty-first International Conference on Machine Learning.
[26] Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu-Hong Hoi. 2022. CodeRL: Mastering
Code Generation through Pretrained Models and Deep Reinforcement Learning. InNeurIPS.
[27] Nam Le Hai, Dung Manh Nguyen, and Nghi DQ Bui. 2025. On the Impacts of Contexts on Repository-Level Code
Generation. InFindings of the Association for Computational Linguistics: NAACL 2025. 1496–1524.
[28] Jia Li, Ge Li, Yunfei Zhao, Yongmin Li, Huanyu Liu, Hao Zhu, Lecheng Wang, Kaibo Liu, Zheng Fang, Lanshen Wang,
Jiazheng Ding, Xuanming Zhang, Yuqi Zhu, Yihong Dong, Zhi Jin, Binhua Li, Fei Huang, Yongbin Li, Bin Gu, and
Mengfei Yang. 2024. DevEval: A Manually-Annotated Code Generation Benchmark Aligned with Real-World Code
Repositories. InFindings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual
meeting, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational
Linguistics, 3603–3614. doi:10.18653/V1/2024.FINDINGS-ACL.214
[29] Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia, Hao Zhang, Xinyi Dai, Yasheng Wang, and Ruiming Tang. 2025. CoIR:
A Comprehensive Benchmark for Code Information Retrieval Models. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova,
and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 22074–22091.
doi:10.18653/v1/2025.acl-long.1072
[30] Yifan Li, Ensheng Shi, and Dewu Zheng. 2024. RepoMinCoder: Improving Repository-Level Code Generation Based
on Information Loss Screening.Proceedings of the 15th Asia-Pacific Symposium on Internetware (Internetware 2024)
(2024). doi:10.1145/3671016.3674819
[31] Yikun Li, Mohamed Soliman, and Paris Avgeriou. 2023. Automatic identification of self-admitted technical debt from
four different sources.Empirical Software Engineering28, 3 (2023), 65.
[32] Ming Liang, Xiaoheng Xie, Gehao Zhang, Xunjin Zheng, Peng Di, Hongwei Chen, Chengpeng Wang, and Gang
Fan. 2024. REPOFUSE: Repository-Level Code Completion with Fused Dual Context. arXiv:2402.14323 [cs.SE]
https://arxiv.org/abs/2402.14323
[33] Dianshu Liao, Shidong Pan, Xiaoyu Sun, Xiaoxue Ren, Qing Huang, Zhenchang Xing, Huan Jin, and Qinying Li. 2024.
$\mathbf{Aˆ{3}}$A3-CodGen: A Repository-Level Code Generation Framework for Code Reuse With Local-Aware,
, Vol. 1, No. 1, Article . Publication date: February 2026.

Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond 23
Global-Aware, and Third-Party-Library-Aware.IEEE Trans. Software Eng.50, 12 (2024), 3369–3384. doi:10.1109/TSE.
2024.3486195
[34] Tianyang Liu, Canwen Xu, and Julian J. McAuley. 2024. RepoBench: Benchmarking Repository-Level Code Auto-
Completion Systems. InThe Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria,
May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=pPjZIOuQuF
[35] Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, and Qianxiang Wang. 2024. Graph-
Coder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model.
arXiv:2406.07003 [cs.SE] https://arxiv.org/abs/2406.07003
[36] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang,
Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al .2024. Starcoder 2 and the stack v2: The next generation.arXiv preprint
arXiv:2402.19173(2024).
[37] Dung Nguyen, Le Nam, Anh Dau, Anh Nguyen, Khanh Nghiem, Jin Guo, and Nghi Bui. 2023. The Vault: A Compre-
hensive Multilingual Dataset for Advancing Code Understanding and Generation. InFindings of the Association for
Computational Linguistics: EMNLP 2023. 4763–4788.
[38] Dung Manh Nguyen, Thang Chau Phan, Nam Le Hai, Tien-Thong Doan, Nam V. Nguyen, Quang Pham, and Nghi
D. Q. Bui. 2025. CodeMMLU: A Multi-Task Benchmark for Assessing Code Understanding & Reasoning Capabilities of
CodeLLMs. InThe Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=
CahIEKCu5Q
[39] OpenAI. 2025. Introducing GPT-4.1 in the API. https://openai.com/index/gpt-4-1/. Accessed: 2025-05-20.
[40] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond.Foundations
and Trends in Information Retrieval3 (01 2009), 333–389. doi:10.1561/1500000019
[41] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu,
Romain Sauvestre, Tal Remez, et al .2023. Code llama: Open foundation models for code.arXiv preprint arXiv:2308.12950
(2023).
[42] Barbara Russo, Jorge Melegati, and Moritz Mock. 2025. Leveraging Multi-Task Learning to Improve the Detection of
SATD and Vulnerability. In2025 IEEE/ACM 33rd International Conference on Program Comprehension (ICPC). 01–12.
doi:10.1109/ICPC66645.2025.00017
[43] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. 2024. Raptor:
Recursive abstractive processing for tree-organized retrieval. InThe Twelfth International Conference on Learning
Representations.
[44] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al .2023. Gemini: a family of highly capable multimodal models.arXiv
preprint arXiv:2312.11805(2023).
[45] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent,
Zhufeng Pan, Shibo Wang, et al .2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of
context.arXiv preprint arXiv:2403.05530(2024).
[46] Yue Wang, Weishi Wang, Shafiq R. Joty, and Steven C. H. Hoi. 2021. CodeT5: Identifier-aware Unified Pre-trained
Encoder-Decoder Models for Code Understanding and Generation. InProceedings of the 2021 Conference on Em-
pirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11
November, 2021, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for
Computational Linguistics, 8696–8708. doi:10.18653/V1/2021.EMNLP-MAIN.685
[47] Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and Zibin Zheng. 2025. RLCoder:
Reinforcement Learning for Repository-Level Code Completion. In47th IEEE/ACM International Conference on Software
Engineering, ICSE 2025, Ottawa, ON, Canada, April 26 - May 6, 2025. IEEE, 1140–1152. doi:10.1109/ICSE55347.2025.00014
[48] Zora Zhiruo Wang, Akari Asai, Frank F Xu, Yiqing Xie, Graham Neubig, Daniel Fried, et al .2025. CodeRAG-Bench:
Can Retrieval Augment Code Generation?. InFindings of the Association for Computational Linguistics: NAACL 2025.
3199–3214.
[49] Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, and Xiaofei Ma. 2024. Repoformer: Selective
Retrieval for Repository-Level Code Completion. InForty-first International Conference on Machine Learning, ICML
2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. https://openreview.net/forum?id=moyG54Okrj
[50] Yiqing Xie, Alex Xie, Divyanshu Sheth, Pengfei Liu, Daniel Fried, and Carolyn Rose. 2024. Codebenchgen: Creating
scalable execution-based code generation benchmarks.arXiv preprint arXiv:2404.00566(2024).
[51] Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang, Ying Li, Qianxiang Wang, and Tao
Xie. 2024. Codereval: A benchmark of pragmatic code generation with generative pre-trained models. InProceedings
of the 46th IEEE/ACM International Conference on Software Engineering. 1–12.
[52] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen.
2023. RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation. InProceedings
, Vol. 1, No. 1, Article . Publication date: February 2026.

24 Minh et al.
of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-
10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, 2471–2484.
doi:10.18653/V1/2023.EMNLP-MAIN.151
[53] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao
Zhang, Jie Jiang, and Bin Cui. 2024. Retrieval-Augmented Generation for AI-Generated Content: A Survey.
arXiv:2402.19473 [cs.CV] https://arxiv.org/abs/2402.19473
, Vol. 1, No. 1, Article . Publication date: February 2026.
# Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation

**Authors**: Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren

**Published**: 2026-08-03 09:00:49

**PDF URL**: [https://arxiv.org/pdf/2608.01927v1](https://arxiv.org/pdf/2608.01927v1)

## Abstract
LLM-based repository-level code generation aims to generate code using the context available in a software repository, requiring LLMs to reason over complex code dependencies. Due to limited context windows and insufficient repository-specific understanding, LLMs typically rely on retrieval-augmented generation (RAG) to incorporate relevant code. Early RAG approaches primarily employ similarity-based retrieval, which often fails to retrieve code snippets that the target function depends on. Recent work introduces graph-based retrieval to model such dependencies, but typically relies on manually designed rules and static global graphs, leading to limited flexibility and high construction and maintenance costs. In contrast, human developers collect helpful context by implicitly constructing a partial dependency graph and iteratively inspecting along it. Inspired by this behavior, we propose DyRetriever, an efficient context retrieval method via partial dependency graphs. DyRetriever uses an LLM to first select a set of entry-point functions and then perform multi-hop reasoning along the code dependency graph. During multi-hop reasoning, it uses the LLM's semantic understanding to validate whether a function can help generate the target function, eliminating manually designed rules and enabling flexibility across scenarios. Instead of statically constructing a global dependency graph, DyRetriever builds a partial graph on demand and discards it after use, reducing construction and maintenance costs. We integrate DyRetriever with a similarity-based code retriever to build DyCoder and evaluate it on CoderEval and DevEval. Experimental results show that DyCoder achieves relative Pass@1 improvements of 25.63% and 59.73% on CoderEval and DevEval, respectively, compared with existing RAG-based methods, while being 7.4x faster than baselines based on static dependency graph construction.

## Full Text


<!-- PDF content starts -->

Effective and Efficient Context Retrieval via Partial Dependency
Graph for Repository-Level Code Generation
Zhongxin Liu
liu_zx@zju.edu.cn
College of Computer Science and
Technology and the State Key
Laboratory of Blockchain and Data
Security, Zhejiang University
Hangzhou, ChinaZhonghao Jiang
zhonghao.j@zju.edu.cn
College of Computer Science and
Technology and the State Key
Laboratory of Blockchain and Data
Security, Zhejiang University
Hangzhou, ChinaZhifan Ye
yezhifan@zju.edu.cn
College of Computer Science and
Technology and the State Key
Laboratory of Blockchain and Data
Security, Zhejiang University
Hangzhou, China
Haoye Wang∗
wanghaoye@hzcu.edu.cn
School of Computer and Computing
Science, Hangzhou City University
Hangzhou, ChinaJiakun Liu
jiakunliu@hit.edu.cn
Faculty of Computing, Harbin
Institute of Technology
Harbin, ChinaXiaoxue Ren
xxren@zju.edu.cn
School of Software Technology and
the State Key Laboratory of
Blockchain and Data Security,
Zhejiang University
Hangzhou, China
Abstract
LLM-based repository-level code generation aims to generate code
according to the available context within a software repository,
requiring LLMs to understand and reason over complex code de-
pendencies. Due to limited context windows and insufficient un-
derstanding of repository-specific code, LLMs typically rely on
retrieval-augmented generation (RAG) to incorporate relevant con-
textual code. Early RAG approaches primarily employ similarity-
based retrieval, which often fails to retrieve code snippets that the
target function actually depends on. Recent work introduces graph-
based retrieval to model such dependencies, but typically relies on
manually designed rules and static global graphs, leading to limited
flexibility and high construction and maintenance costs.
In contrast, human developers typically collect helpful context
by implicitly constructing a partial dependency graph and itera-
tively inspecting along it. Inspired by human behavior, we propose
DyRetriever, an effective and efficient context retrieval method
via partial dependency graphs.DyRetrieverleverages an LLM
to simulate how human developers collect helpful context by first
selecting a set of entry-point functions and then performing multi-
hop reasoning along the code dependency graph. During multi-hop
reasoning,DyRetrieverleverages the LLM’s semantic understand-
ing to explicitly validate whether a function can help to generate
the target function, thereby eliminating reliance on manually de-
signed rules and enabling flexibility across application scenarios.
Instead of statically constructing a global dependency graph,DyRe-
trieverbuilds a partial graph on demand and discards it after use,
reducing construction overhead and avoiding maintenance costs.
∗Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
ASE ’26, Munich, Germany
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2882-2/2026/10
https://doi.org/10.1145/3832783.3834350We integrateDyRetrieverwith a similarity-based code retriever
to build a new code generation approach,DyCoder, and evaluate it
on two widely used repository-level code generation benchmarks,
CoderEval and DevEval. Experimental results show thatDyCoder
achieves relative Pass@1 improvements of 25.63% and 59.73% on
CoderEval and DevEval, respectively, compared with existing RAG-
based methods, while being 7.4 ×faster than baselines based on
static dependency graph construction.
CCS Concepts
•Software and its engineering→Automatic programming.
Keywords
Repository-Level Code Generation, Partial Dependency Graph, Re-
trieval Augmented Generation
ACM Reference Format:
Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, and Xi-
aoxue Ren. 2026. Effective and Efficient Context Retrieval via Partial Depen-
dency Graph for Repository-Level Code Generation. InProceedings of the
41st IEEE/ACM International Conference on Automated Software Engineering
(ASE ’26), October 12–16, 2026, Munich, Germany.ACM, New York, NY, USA,
13 pages. https://doi.org/10.1145/3832783.3834350
1 Introduction
Code generation aims to automatically generate the code implemen-
tation corresponding to a given specification [ 16,44–46,49]. Large
language models (LLMs) [ 14,30,51], owing to their strong capabili-
ties in language understanding and generation, have demonstrated
remarkable performance in code generation [ 15,20,31,38,57]. Re-
cently, repository-level code generation, which aims to generate
a target function according to its signature1and available context
in the repository, has attracted widespread attention due to its
close alignment with real-world development scenarios [ 18,46].
1In this paper, we consider a function’s signature to include its corresponding docstring.
arXiv:2608.01927v1  [cs.SE]  3 Aug 2026

ASE ’26, October 12–16, 2026, Munich, Germany Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren
However, due to their limited context window and lack of under-
standing of specific code repositories, it is difficult for LLMs to
handle complex repository-level code generation tasks. To mitigate
this challenge, recent studies [ 17,33,50,54] typically adopt the
retrieval-augmented generation (RAG) paradigm, wherein the sig-
nature of the target function is used as queries to retrieve helpful
code snippets from the repository, and the retrieved code is then
provided to the LLM as the context for code generation.
The key challenge for RAG-based methods is retrieving contexts
that accurately guide the LLM to generate correct target functions
within a repository. Prior works [ 17,50,54] mainly retrieve similar
code snippets through sparse retrieval (e.g., BM25 [ 34,40]) or dense
retrieval (e.g., RepoCoder [ 54]) as the contexts, which have been
shown to be beneficial. However, similarity-based retrieval is often
insufficient for generating the target function [ 5,32,50], since it
may overlook code snippets that the target function is not similar to,
but actually depends on. Therefore, advanced methods incorporate
graph-based retrieval to supplement similarity-based retrieval to
capture semantic relationships, including dependencies [ 32] and
containment [ 33], that similarity alone cannot reveal. Specifically,
these methods construct code graphs, such as program dependency
graphs [ 32,33], of the whole repository using static analysis tools,
and design retrieval algorithms using manually crafted rules to
perform retrieval based on reachability on code graphs.
However,such designs often lack flexibility in capturing the diverse
and complex dependency patterns that arise in real-world repositories.
Existing studies have [ 13,29] demonstrated that repository-level
code generation in different application scenarios requires different
types of context, e.g., in-file code snippets, cross-file dependen-
cies, API, or symbol definitions. Nevertheless, existing graph-based
methods typically rely on fixed rules that are tailored to a specific
type of context, such as handcrafted similarity metrics [ 32] and
import-based traversal [ 33], making them difficult to adapt when
the required contextual types change across tasks. For example,
RepoScope [ 33] retrieves cross-file context by traversing the code
graph from entities imported by the target function, and thus can-
not obtain such context when relevant cross-file dependencies are
not explicitly imported. Moreover,the complete dependency graph of
the whole repository is both time-consuming to parse and difficult to
maintain. For example, it may take several hours to construct the de-
pendency graph for a large codebase with more than 100,000 lines of
code [ 7,21]. Such overhead is often unacceptable for practical code
generation, as developers typically expect to obtain code within a
short time frame [ 28,46,52]. In addition, developers continuously
modify and extend code within a repository. As development pro-
gresses, dependency relations between code elements also evolve,
requiring frequent reconstruction or updating of the dependency
graph and thus incurring additional maintenance costs.
In contrast to existing graph-based retrieval methods, human
developers typically do not rely on predefined retrieval rules to
locate relevant context. Instead, they flexibly inspect code based
on their understanding of the task and the repository, iteratively
judging whether a function is helpful for completing the target
function [ 10]. Moreover, they do not pre-construct or maintain a
complete repository-level dependency graph. They usually start
from a small number of functions that appear relevant as entrypoints and explore related functions on demand by following de-
pendency chains [ 23,24]. Through this process, developers implic-
itly construct a partial dependency graph during inspection, rather
than relying on a global one [2].
Inspired by how developers iteratively collect helpful contexts,
we proposeDyRetriever, an effective and efficient context re-
trieval method via partial dependency graphs.DyRetrieverre-
trieves functions that are useful for generating the target function
by simulating the behavior of human developers with the help of
an LLM. It first leverages an LLM to identify some functions as
entry points based on the signature of the target function and the
repository structure, mimicking human developers’ initial selection.
Starting from these entry points,DyRetrieverexplores related
functions through multi-hop reasoning, reflecting how developers
progressively decide whether additional dependencies are helpful.
However, such decisions are inherently dependent on developers’
understanding of a function and its relationship to the target func-
tion, which are hard to adequately model with pre-defined rules. We
therefore leverage an LLM, whose strong semantic understanding
of code enables it to determine at each hop whether exploring ad-
ditional dependencies is likely to be helpful. In addition, to support
such multi-hop reasoning,DyRetrieverrequires local dependency
information that can be obtained on demand. However, existing
static analysis tools focus on providing global dependency graphs
and do not support efficient construction of partial dependency
graphs [ 3,42]. Considering the callees of a function can usually be
inferred from the code and its imported modules, we further employ
the LLM to dynamically construct partial dependency graphs, en-
abling human-like on-demand construction of a partial dependency
graph during retrieval.
DyRetrievercan supplement similarity-based retrieval by
leveraging semantic information that is not captured by similarity.
By integratingDyRetrieverwith similarity-based code retrieval,
we further propose a new repository-level code generation ap-
proach namedDyCoder. We evaluateDyCoderon two popular
repository-level code generation benchmarks, CoderEval [ 53] and
DevEval [ 26], with three advanced LLMs, and perform compre-
hensive comparisons with state-of-the-art baselines. Experimental
results show that, compared with existing RAG methods,DyCoder
achieves an average relative improvement of 25.63% and 59.73%
in Pass@1 on CoderEval and DevEval, respectively. The results of
our ablation studies show that removing either theDyRetriever
or the similarity-based retrieval leads to a comparable average
performance drop of 14.80% and 15.85%, respectively, while remov-
ing multi-hop reasoning results in an average 6.55% performance
degradation. This demonstrates the complementarity between
the two types of retrieval methods and reveals the effectiveness
of multi-hop reasoning. The results of our efficiency evaluation
show that, compared with the static graph construction-based
method, RepoScope,DyCoderachieves a 7.4 times improvement
in efficiency without training cost, while attaining efficiency
comparable to similarity-based methods. Finally, we combine the
context collected byDyRetrieverwith different similarity-based
methods to verify the generalizability ofDyRetriever. The results
show thatDyRetrievercan achieve 7.5%-30.99% performance
improvement over existing similarity-based methods. This not only

Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation ASE ’26, October 12–16, 2026, Munich, Germany
def normpath (path):
    # type: (T ext) -> Text
    """Normalize a path.
    This function simplifies a path by collapsing back-references
    and removing duplicated separators.
    """
(......function body omitted......)Similarity Score: 0.592
Threshold: 0.792AlignCoder
    fs/path.py:: relativefrom
    fs/path.py:: dirname
    fs/path.py:: split
    fs/path.py:: parts
    fs/path.py:: abspathRepoScope
Incorrect Call Chain
RetrievePredictdef iteratepath (path):
    from .path import normpath
    _path = normpath (path)
    if _path == "/":
        return []
    components = _path. strip("/").split("/")
    return [component for component in components if component]
PASSEDOurs
def iteratepath (path):
    if not path:
        return []
    path = path. strip("/")
    if not path:
        return []
    components = [component for component in path.split("/") if component]
    return components
FAILEDStatic Build GraphTime Consuming
RepoScoperecurse
path
abspath joinparts
iterate
pathnorm
pathrelative
from
Target
FunctionFunction
Calldef iteratepath (path):
    # type: ( Text) -> List[ Text]
    """
    This f unction tak es a path as input and i terates o ver its
individual components. It r eturns a l ist of path components.
    Input-Output Ar guments
    :par am path: T ext. The path to i terate over. For example,
'/foo/bar/baz'.
    :return: List of T ext. A l ist of path components.
    """Target Function's Signature Dependency
Figure 1: A motivating example of“fs.path.iteratepath”in DevEval.
demonstrates the general applicability ofDyRetriever, but also
provides directions for future enhancements.
In summary, our main contributions are as follows:
•We proposeDyRetriever, a context retrieval method for
repository-level code generation, inspired by human behaviors
in collecting helpful code snippets.DyRetrieveruses LLMs’
semantic understanding to assess code snippets and dynam-
ically construct partial dependency graphs, achieving good
effectiveness and efficiency.
•We construct a new repository-level code generation approach,
DyCoder, by integratingDyRetrieverwith a similarity-based
retrieval method, and also demonstrate the generalizability of
DyRetrieveracross different similarity-based retrieval methods.
•Extensive and comprehensive experiments on CoderEval [ 53]
and DevEval [ 26] show thatDyCoderachieves relative improve-
ments of 25.63% and 59.73% in Pass@1, respectively, compared
with other RAG methods, and is 7.4 ×faster than the baseline
based on static graph construction.
2 Motivation
A key challenge of repository-level code generation lies in extract-
ing the contexts, which are helpful for the LLM to generate the
target function, from the repository. Existing methods typically re-
trieve code snippets based on text similarity or manually designed
retrieval rules on the code graphs. Figure 1 presents a motivating ex-
ample collected from the PyFilesystem [36] repository. This task
aims to implement the iteratepath function, which returns a list
of path components and depends on the normpath function defined
infs/path.py. However, neither the state-of-the-art similarity-based
method AlignCoder nor the state-of-the-art graph-based method
RepoScope successfully retrievesnormpathas context.
We further investigate why AlignCoder and RepoScope fail in
this case. The cosine similarity between the embeddings of the
target function’s signature and the function normpath is 0.592.
AlignCoder only retrieves the top-5 most similar code snippets
as context, leading to the similarity threshold for this case being
0.792. Thus, AlignCoder fails to retrieve normpath . This indicates
that relying solely on similarity-based retrieval may overlook useful
dependencies. On the other hand, RepoScope designs an import-
based retrieval rule such that only nodes reachable from entitiesimported by the target function within a certain distance in the
dependency graph can be retrieved. In this case, normpath is de-
fined in the same file as iteratepath and does not meet the rule
mentioned above. Thus, RepoScope also fails to retrieve normpath .
Although this issue could be mitigated by adding extra heuristics
(e.g., retrieving in-file functions), such case-by-case rules further
highlight the limited flexibility of manually designed rules.
In practice, human developers can easily locate normpath . For
example, one may initially select parts andjoin based on their
understanding of this repository since they may share partially
similar program behaviors (as illustrated by the dependency graph
in the top-right of Figure 1). When examining parts andjoin , the
developer can trace their callees and find that normpath is a basic op-
eration when processing paths and is used frequently by both parts
andjoin. Thus, the developer can successfully identifynormpath
as a relevant dependency for implementing iteratepath . This ob-
servation suggests that human developers identify helpful context
by first selecting some potential relevant functions as entry points
and then iteratively following their dependency relations to explore
additional relevant functions. Since it is widely believed that LLMs
operate similarly to humans, we may be able to collect context by
simulating human developers using LLMs.
Achieving such a process requires constructing a dependency
graph for the repository. Existing works use static analysis tools to
parse dependencies across the entire repository. For example, Re-
poScope uses PyType [11] and Tree-Sitter [47] to parse relations
such as calls, imports, and inherits to construct a static global graph.
However, statically constructing the global dependency graph of a
repository is time-consuming [ 7,21]. For instance, RepoScope takes
261.61s to build the graph corresponding to this case. In contrast,
humans dynamically explore relevant parts by expanding a partial
dependency graph. Inspired by the way humans analyze code, we
proposeDyRetriever, an effective and efficient context retrieval
method for repository-level code generation. It mimics human code
analysis by leveraging the semantic understanding capabilities of
LLMs and constructing and expanding partial dependency graphs.

ASE ’26, October 12–16, 2026, Munich, Germany Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren
Repository
Target Function
Parse
All Functions
Embed
Function
Index
RepositoryParse
Repo Structur eLLM
Related
FilesFunctions
In Files
Parse
VisitExpand Init
VisitExpand
Visit ......
Calculate Similarity
with Target Function
Similar
FunctionsLLM
Selected
Entry P oints
Expand
Visit
Related
Functions
Target
Function
+
+
Similarity
ScoresSelect
Top-k
Summary T op-k
Completed
Function
LLMDyRetriever: Graph-based Context Collection
Similarity-based Context RetrievalGeneration
Multi-hop Reasoning
with Partial Graph 
...Select Select Select Select
Candidate Node
Visited Node
To be Visi ted Node
Constuct gr aph 
Get code of the nodeEntry Points Selection
Figure 2: Overview ofDyCoder.
3 Approach
In this section, we introduceDyCoder, an LLM-driven repository-
level code generation framework. Different from existing graph-
based methods,DyCoderis a rule-free, autonomous framework
that flexibly collects context through partial graphs. As shown in
Figure 2,DyCodertakes the signature of a target function and a
repository as inputs and returns the completed code of the target
function. It consists of three phases, i.e., graph-based context col-
lection, similarity-based context retrieval, and generation. In the
graph-based context collection phase (Section 3.1), we introduce
DyRetrieverto perform effective and efficient retrieval.DyRe-
trieversimulates the way human developers explore a repository
when collecting contexts by reasoning over function dependencies.
Specifically,DyRetrieverfirst initializes one or more functions
as the entry points of the reasoning chain based on the provided
repository structure. It then dynamically expands the partial code
dependency graph and collects related functions as context accord-
ing to the LLM’s traversal path over the graph nodes, which avoids
the need to parse and store a global repository-level dependency
graph, thereby significantly reducing time and maintenance over-
head. In the similarity-based context retrieval phase (Section 3.2),
DyCoderfirst extracts all functions in the repository, and then
computes the cosine similarity of embeddings generated by a pre-
trained embedding model between each function and the signature
of the target function to retrieve similar functions as context. In the
generation phase (Section 3.3),DyCoderdeduplicates the context
retrieved in the previous two phases, and then generates the codeof the target function based on its signature and the deduplicated
functions through prompt engineering.
3.1DyRetriever: Graph-based Context
Collection
This phase takes the signature of a target function and the repos-
itory as input and returns a list of functions in the repository.
Specifically, it consists of two stages, i.e., entry points selection
and multi-hop reasoning with partial graphs. The entry points se-
lection stage (Section 3.1.1) aims to identify several functions from
the code repository as entry points for multi-hop reasoning and
partial graph construction. It simulates how human developers,
based on their understanding of the repository, identify several
potentially relevant code snippets. The multi-hop reasoning with
partial graphs stage (Section 3.1.2) aims to iteratively perform graph
expansion, node selection, and node visiting based on the entry
point. It simulates how human developers traverse dependency
graphs and iteratively assess whether each function contributes to
implementing the target function.
3.1.1 Entry Points Selection.This stage aims to establish the LLM’s
understanding of the code repository and, based on the understand-
ing, select several functions that may be relevant to the target
function. It reduces the search scope from the whole repository
to several files that may be related to the target function. It takes
the signature of a target function and the repository as input and
returns a list of functions as entry points. To establish the LLM’s
understanding of the code repository, we first parse the repository

Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation ASE ’26, October 12–16, 2026, Munich, Germany
into a tree-based structure following prior works [ 18] to pre-
vent exceeding the context window when letting the LLM read
the entire repository. The tree-based structure presents the
organizational structure of files in the repository according to the
folder containment hierarchy. Then, to identify the entry points,
we proceed from the file level to the function level to identify the
entry points. Specifically, based on the tree-based structure
and the signature of the target function, the LLM is prompted to
examine the overall layout of the codebase and return a list of file
paths that are related to the target function, denoted as the related
file list. To ensure that the subsequent reasoning process includes
both in-file context and cross-file context, we additionally verify
whether the LLM includes the file containing the target function. If
not, we explicitly add it to the related file list as the source of in-file
context, while the remaining files serve as the source of cross-file
context. Next, we use a parser to extract all functions contained in
the files from the related file list and construct a mapping, where
the file path is used as the key, and all the function names contained
in that file are used as the value. Finally, we provide the signature of
the target function and the mapping as context to the LLM, asking
it to identify several function names as entry points.
3.1.2 Multi-hop Reasoning with Partial Graphs.This stage aims to
perform multi-hop reasoning along the dependency graph based on
the entry points and examine, one by one, whether the functions
on the traversal path contribute to generating the target function.
Additionally, it dynamically constructs a function-level partial de-
pendency graph instead of using a static parser to construct the
entire graph of the repository to improve efficiency. It takes the
signature of the target function and the entry points as input and
finally returns the top-k related functions.
As shown in Algorithm 1, the process begins by initializing an
empty list to record the traversal trajectory (Line 1) and a set of
candidate functions using the entry points (Line 2). Thecandidate
functionsform a global search space that is dynamically updated
during exploration. We also set a maximum number of reasoning
steps (Line 3), which serves as a global exploration budget. The
traversal algorithm consists of three steps: select, visit, and expand.
•Select Step:Select one function from thecandidate functions
using the LLM (Line 5). The LLM is prompted to leverage the
current context (i.e., previously selected functions and the target
function signature) to identify a candidate function that best
aligns with the target function.
•Visit Step:Based on the function selected in the previous step,
we determine whether this function has been visited before (Line
6); if so, this visit is skipped (Line 7), otherwise we obtain its
source code (Line 9), update its visit status (Line 10-11), and
record the visit trajectory in order (Line 12).
•Expand Step:According to the function selected in the select
step, all import statements from the code file in which the func-
tion resides are extracted (Line 14). Then, using the source code
of this function obtained in the visit step and the import state-
ments as input, the LLM is prompted to identify the functions
it depends on (Line 15). To reduce hallucination during LLM
identification,DyRetrieveronly requires the LLM to recognize
two types of relations: call and lazy import. Because import state-
ments provide information about the source of APIs, the LLMis able to determine whether call or lazy import relations are
valid based on the code. This step expands the partial depen-
dency graph dynamically and retrieves the neighbor nodes of
the currently visited function.DyRetrieversubsequently adds
the functions identified by the LLM to thecandidate functionsto
enable expansion before the next traversal iteration (Line 16).
After performing a limited number of traversal iterations, the
multi-hop reasoning algorithm uses the trajectory collected during
traversal as context and prompts the LLM, based on the target
function, to determine the most related top-k functions from the
trajectory (Line 20-21) as the semantic-based context.
While the LLM can efficiently identify related functions from
the traversal trajectory, we observe two major issues when directly
prompting it to expand the dependency graph. First, hallucinations
of the LLM may fabricate non-existent functions. Second, APIs from
standard libraries may be mistakenly reported as cross-file APIs.
For example, the match API from the relibrary may be incorrectly
identified by the LLM as one of the function’s dependencies. Since
we focus only on cross-file API dependencies, such dependencies
should be discarded. To address these two issues, we post-process
the dependency graph constructed by the LLM. Specifically, for
each dependency function identified by the LLM,DyRetriever
searches the code repository for its corresponding implementation
by matching the function name. If no implementation is found, the
dependency is discarded. This step improves the reliability of the
constructed dependency graph and enhances the effectiveness of
DyRetriever.
3.2 Similarity-based Context Retrieval
This phase takes the signature of the target function and the repos-
itory as input, which aims to retrieve a list of similar functions that
are useful for generating the target function. Since existing stud-
ies have already demonstrated that function-level context is more
effective than code-slice-based context [ 12], we only retrieve simi-
lar functions. As prior studies [ 18,27,46] have shown that dense
retrieval outperforms sparse retrieval, we adopt embedding-based
dense retrieval in this phase. Specifically, a static parser is used to
extract all functions from the entire repository, embed them using
an embedding model, and construct an index using their embed-
dings (denoted asfunction embeddings). Then,DyCoderembeds
the signature of the target function using the same embedding
model and compute the cosine similarity between the obtained
embedding and the function embeddings in the repository. Finally,
DyCoderselects the top-k most similar functions in descending
order of similarity.
3.3 Generation
This phase takes the signature of a target function, the related
functions acquired by theDyRetriever, and the similar functions
acquired by the similarity-based context retrieval as input, and
returns the generated completed target function. Specifically, to
avoid redundant context from interfering with the LLM’s reasoning,
the retrieved relevant functions and similar functions are merged
into a single deduplicated set. The functions in this set are regarded
as the context of the target function. Then the LLM is prompted

ASE ’26, October 12–16, 2026, Munich, Germany Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren
Algorithm 1Multi-hop Reasoning with Partial Graphs
Input:tarFunc⊲Signature of the target function
Input:entries⊲List of entry point functions including file
names and function names
Input:maxHop⊲Maximum steps of multi-hop reasoning
Output:contextFunctions⊲Functions retrieved by
DyRetrieveras contexts
1:trajs = []⊲Record traversal path
2:candidates = entries⊲To be visited
3:hop = 0
4:whilehop<maxHopdo⊲Start Traversal
5:selectedFunc = LLM 𝑠𝑒𝑙𝑒𝑐𝑡 (tarFunc, candidates, trajs)⊲
Select step
6:ifisVisited(selectedFunc)then
7:continue
8:end if
9:funcContent = getCode(selectedFunc)⊲Visit step
10:setIsVisited(selectedFunc)
11:candidates = candidates - selectedFunc
12:trajs.append([selectedFunc, funcContent])⊲Save trajs
13:// Begin Expand Step
14:imports = matchExtract(selectedFunc)⊲Get all import
statements in the file
15:neighbors = LLM 𝑐𝑜𝑛𝑠𝑡𝑟𝑢𝑐𝑡 (imports, funcContent)⊲
Dynamically expand dependency graph
16:candidates = candidates∪neighbors
17:// End Expand Step
18:hop += 1
19:end while
20:contextFunctions = LLM 𝑒𝑥𝑡𝑟𝑎𝑐𝑡 (tarFunc, trajs)
21:returncontextFunctions
to generate the target function based on its signature and the pro-
vided context. Finally, to prevent the generated code from becoming
non-executable due to indentation errors, a post-processing stage
is applied to adjust the indentation of the completed function. The
post-processed result is then returned as the final completed imple-
mentation of the target function. The detailed prompt templates of
all phases can be found in our replication package.
4 Experiment Setup
4.1 Datasets & Metrics
To evaluate the performance ofDyCoderon repository-level func-
tion generation tasks, we employ two widely used benchmarks,
including CoderEval-Python [ 53] and DevEval [ 26], following prior
work [ 1,33]. These two benchmarks require LLMs to generate the
implementation of a target function in a repository according to
its signature, and provide a test suite for each task to assess the
correctness of the generated function.
•CoderEval-Python[ 53] is the Python subset of CoderEval con-
taining 230 instances. It is designed to assess code generation
capabilities across six levels of contextual dependency.
•DevEval[ 26] is a manually annotated benchmark designed to
evaluate the programming capabilities of large language models
within real-world code repositories. It contains 1,825 instances
extracted from 117 codebases. Since DevEval lacks a sandboxedevaluation environment, we find that 209 instances fail in our
environment when executing the provided test suites to evalu-
ate the ground-truth solutions. We provide detailed information
about these cases in the replication package. For consistency, we
treat all 209 instances as failures forDyCoderand all baselines,
and compute the final metrics over the full set of 1,825 instances.
To evaluate the effectiveness, we adopt the widely used metric
Pass@1[ 4,31] following prior work [ 33]. For efficiency evaluation,
we record theruntime (in seconds)of each method and count the
number of tokens (#Token) involved in LLM interactions.
4.2 Baselines
We mainly compareDyCoderwith open-source state-of-the-art
methods that focus on repository-level function generation, in-
cluding RLCoder [ 50], AlignCoder [ 17], and RepoScope [ 33]. We
additionally include direct prompting and a simple RAG approach
as baselines to highlight the effectiveness of the context collected
byDyCoder. The baselines are described as follows:
•Direct: It generates the function body directly from the function
signature without providing any context.
•SimpleRAG: It retrieves a set of similar code slices by computing
the cosine similarity between the embedding of the signature of
the target function and code slices in the repository, and uses
them as the context for generation. Following prior work [ 33,50],
the code slices are obtained by using a sliding window algo-
rithm [54] to segment the code files in the repository.
•RLCoderis a reinforcement learning-based framework for repo-
level code completion that optimizes retriever behavior through
perplexity feedback.
•AlignCoderconstructs an enhanced query by generating multi-
ple candidate completions to bridge the semantic gap between the
initial query and the target code, and uses reinforcement learn-
ing to train a retriever to learn how to leverage the inference
information contained in the enhanced query.
•RepoScopebuilds a Repository Structural Semantic Graph
(RSSG) and retrieves a four-view context that integrates struc-
tural and similarity-based information. It further introduces a
call-chain prediction method that leverages repository-level
structural semantics to identify the target function’s callees
more accurately.
4.3 Research Questions
•RQ1. Effectiveness: How effective isDyCoderin repository-
level function generation?
•RQ2. Efficiency: How efficient isDyCoderin repository-level
function generation?
•RQ3. Ablation: How do the key components ofDyCodercon-
tribute to its effectiveness?
•RQ4. Generalizability: Can the context collected byDyRe-
trieverbe effectively combined with different similarity-based
methods?
4.4 Implementation Details
We conduct experiments with Qwen3-Coder [ 51], DeepSeek-
v3.2 [ 30], and GPT-4o-mini [ 14]. They are from different families

Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation ASE ’26, October 12–16, 2026, Munich, Germany
Table 1: Performance comparison of different methods across
different LLMs and benchmarks.
MethodModelAvg.
Qwen3-Coder-30B DeepSeek-v3.2 GPT-4o-miniCoderEvalDirect 24.35↑103.2% 24.00↑112.7% 20.78↑94.2% 23.04↑103.83%
SimpleRAG 36.17↑36.8% 39.83↑28.1% 28.26↑42.8% 34.75↑35.14%
RLCoder 35.65↑38.8% 42.17↑21.0% 34.78↑16.0% 37.52↑25.16%
AlignCoder 38.17↑29.6% 38.26↑33.4% 34.78↑16.0% 37.07↑26.68%
RepoScope 42.96↑15.2% 43.22↑18.1% 36.00↑12.1% 40.72↑15.32%
DyCoder 49.48 51.04 40.35 46.96DevEvalDirect 12.90↑263.4% 13.91↑239.7% 5.18↑448.1% 10.66↑283.11%
SimpleRAG 27.34↑71.5% 31.87↑48.3% 15.37↑84.7% 24.86↑64.28%
RLCoder 23.97↑95.6% 27.45↑72.1% 15.90↑78.6% 22.44↑82.00%
AlignCoder 23.60↑98.6% 27.70↑70.6% 16.46↑72.5% 22.58↑80.87%
RepoScope 43.06↑8.9% 45.39↑4.1% 25.54↑11.2% 38.00↑7.47%
DyCoder 46.88 47.25 28.39 40.84
and cover both open-source and closed-source models. For Qwen3-
Coder, we employ Qwen3-Coder-30B-Instruct and deploy it locally
via vLLM [ 22] on 2 Nvidia A100 GPUs. For DeepSeek-v3.2 and
GPT-4o-mini, we access them through their official API services.
In the official implementation released by RepoScope [ 33], the
corresponding test suites of the target function are not masked from
the repository, which may lead to artificially inflated performance.
To prevent potential data leakage, we mask both the signature and
implementation of the target function in the repository and provide
only the signature to the LLM as the task specification in the prompt.
Furthermore, all methods are restricted from accessing non-code
files or test files.
To ensure a fair comparison, we use the Qwen3-Embedding-0.6B
model [ 56] via vLLM on 1 Nvidia A100 GPU, as the embedding
model for embedding-based similarity calculation in all the meth-
ods except RLCoder and AlignCoder, since they train their own
embedding models. We reuse the embedding models released by
RLCoder [ 50] and AlignCoder [ 17]. Since these two methods are
originally designed for repository-level code completion, we adapt
them to the CoderEval [ 53] and DevEval [ 26] benchmarks by re-
placing their prompt templates with those ofDyCoder.
ForDyCoder, we set the maximum hop of the multi-hop reason-
ing to 10, since preliminary experiments setting it in [1,20] show
that when the maximum hop exceeds 10,DyCoder’s performance
begins to plateau, while the token consumption increases by 19.1%.
The number of retrieved code snippets of all methods is controlled
consistently with RepoScope [ 33], i.e., providing the Top-10 similar
functions for the similarity-based context, and providing the Top-5
related functions for extra context.
When interacting with LLMs, we employ greedy sampling (tem-
perature = 0) to ensure the reproducibility of the LLM’s sampling
results. For measuring time cost, we record the execution time of
all the methods on Intel Xeon Platinum 8358P CPUs, using a serial
execution paradigm.
5 Experiment Results
5.1 RQ1: Effectiveness
To validate the effectiveness ofDyCoderin repository-level code
generation, we conduct experiments with three different LLMs and
# fs/path.py
def parts(path):
    # type: (Text) -> List[Text]
    """Split a path in to its component parts .""" 
    _path = normpath (path)
    components = _path. strip("/")
    ...
# fs/path.py
def relativefrom (base, path):
    """Return a path relative from a given base path.
    Insert backrefs as appropriate to reach the path from the base. """
    base_parts = list(iteratepath (base))
    path_parts = list(iteratepath (path))
    ...
# fs/path.py
def normpath (path):
    """Normalize a path. This function simplifies a path by collapsing
back-references and removing duplicated separators.
    """
Inconsistent semantic goals
with the target function
Target DependencyAlignCoder
RepoScope
DyRetrieverTraversal rules cannot apply
to the in-file function.
Figure 3: An example of the retrieved context and generated
code byDyCoderand baselines. The target function is the
same as the one in Figure 1.
compareDyCoderwith the baselines. Table 1 presents the experi-
mental results on CoderEval and DevEval, respectively. Experimen-
tal results show that on CoderEval and DevEval,DyCoderachieves
an improvement of 12.1%-42.8% and 4.1%-98.6%, respectively, com-
pared to the RAG-based baselines, which are significantly more
effective in repository-level code generation than the baselines.
This demonstrates that by simulating human developers’ behavior
of collecting contextual information,DyCodercan effectively re-
trieve useful context to support repository-level code generation.
Moreover, compared withsimilarity-based methods, i.e., SimpleRAG,
RLCoder, and AlignCoder,DyCoderachieves an average improve-
ment of 29.13% and 76.94% on CoderEval and DevEval, respectively.
Compared with thegraph-based method, RepoScope,DyCoderstill
achieves an improvement of 4.1%–18.1%, indicating that, unlike
rule-based retrieval algorithms, leveraging LLM to assess the use-
fulness of functions enablesDyCoderto more flexibly retrieve
truly helpful context. Despite this,DyCoderis also 7 ×faster than
RepoScope, making it an accurate and efficient approach.
The improvement in DevEval is greater than in CoderEval be-
cause, compared to CoderEval, the repositories in DevEval are
larger and contain more complex dependency relations [ 26], mak-
ing it more difficult for similarity-based methods to retrieve useful
context. This indicates that compared with similarity-based meth-
ods,DyCodercan effectively acquire useful context from complex
dependency relationships through multi-hop reasoning.
To further investigate whyDyCoderoutperforms other base-
lines, we manually inspect the experimental results and summarize
two main reasons. First, by simulating the behaviors of human
developers,DyRetrieverprovides additional useful context about
the target function compared with the similarity-based baseline.
For example, Figure 3 presents the context retrieved and the code
generated byDyCoderand two best-performing baseline methods
for the example given in Figure 1. AlignCoder [ 17] retrieves the
similar function parts . However, the semantic goals of the two
functions are inconsistent: parts encodes path representations,
while iteratepath requires pure path components. Reusing parts

ASE ’26, October 12–16, 2026, Munich, Germany Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren
Table 2: Efficiency analysis on CoderEval using Qwen3-Coder.
MethodTime Cost Inference CostTrain?
Indexing Retrieval Generation Total imp. #Token $Cost
Direct 0 0 9.65 9.65 0.07×3.29k 0.0016✗
SimpleRAG 100.87 17.36 9.27 127.50 0.86×3.65k 0.0016✗
RLCoder 103.09 45.23 4.71 153.03 1.03×2.81k 0.0010✓
AlignCoder 100.89 49.09 4.25 154.23 1.04×3.65k 0.0013✓
RepoScope 875.32 209.72 11.19 1096.23 7.40×5.11k 0.0020✗
DyCoder 85.62 41.09/50.01112.46 148.09 1.00× 13.09k 0.0039 ✗
141.09s forDyRetrieverand 50.01s for similarity-based retrieval.
results in incorrect behavior in iteratepath , which indicates that
similar functions are not necessarily relevant to the requirement
and may even mislead the generation. In contrast,DyCoderre-
trieves the full implementation of normpath , enabling the LLM to
obtain the complete context and examples necessary to generate
the target function, leading to a correct solution.
Second,DyCodercan more flexibly obtain useful context related
to the target function through dependency graphs. For example,
RepoScope’s call chain prediction algorithm is manually designed,
which relies on code entities imported by the target functions as
entry points to start traversing the code graph. However, the tar-
get functioniteratepathand its dependency functionnormpath
are in the same code file, leading to the lack of import statements.
Consequently, RepoScope cannot establish an effective traversal
entry point for this case and thus fails to retrieve normpath . Al-
though the call chain predicted by RepoScope includes the caller
function relativefrom ofiteratepath , this function does not
provide enough information about normpath , which leads the LLM
to use an incorrect method for the path normalization operation
during generation. This rule-free approach, which relies on an
LLM to assess helpfulness, enables flexible and accurate multi-hop
reasoning, allowingDyCoderto obtain the desired context.
RQ1 Summary:Compared to RAG-based baselines,DyCoder
achieves an average improvement of 25.63% on CoderEval and
59.73% on DevEval in Pass@1, demonstrating its effectiveness
in performing retrieval by simulating the behavior of humans.
5.2 RQ2: Efficiency
To evaluate the efficiency ofDyCoderin repository-level code
generation tasks, we compare its cost with that of other baselines
from two perspectives: time cost and inference cost. For time cost,
we measure the execution time of the indexing, retrieval, and gener-
ation phases for RAG-based methods. SinceDyRetrieverand the
similarity-based context retrieval phase ofDyCodercan execute
in parallel, we include the maximum of the two as part of the total
time. For inference cost, we obtain the token usage information and
compute the dollar cost according to the official pricing. We conduct
measurements on CoderEval using a locally deployed Qwen3-Coder
model (with a generation speed of about 400 tokens/s).
As shown in Table 2,DyCoderincurs a lower total time cost than
all baselines except Direct and SimpleRAG, and is 7.4 times faster
than the graph-based RepoScope. In theindexing phase,DyCoder
is faster than all RAG-based baselines. This is because, compared
with similarity-based baselines,DyCoderonly extracts the func-
tions in the repository via static analysis, avoiding finer-grainedslicing and embedding. Compared with graph-based baselines,Dy-
Coderdynamically constructs a partial dependency graph instead
of constructing a full repository-level code graph, reducing the
indexing time overhead by roughly an order of magnitude. More-
over,DyCoderconstructs partial dependency graphs using an LLM
and discards them after use, thereby avoiding the computational
overhead incurred by static-analysis-based graph construction.
In theretrieval phase,DyCoderincurs a time cost comparable to
that of similarity-based baselines. The time cost ofDyRetriever
is slightly higher than that of similarity-based retrieval, as partial
graph construction requires multiple rounds of LLM reasoning.
Nevertheless, as graph-based baselines (i.e., RepoScope) rely on
manually designed, high-complexity algorithms for call chain pre-
diction,DyCoderremains nearly four times faster in the retrieval
phase.
In thegeneration phase, the time cost depends on the length
of the retrieved context. Since bothDyCoderand RepoScope in-
corporate context from two perspectives, their generation time is
slightly longer than that of the other baselines. Overall,DyCoder
achieves time efficiency comparable to similarity-based baselines
while substantially outperforming graph-based baselines.
This efficiency gain is partly attributable to constructing only par-
tial dependency graphs, which reduces parsing overhead, and partly
to relying on LLMs for dependency graph construction, which
avoids the maintenance cost incurred when the codebase changes.
In real-world development scenarios, developers are typically not
limited to generating a single function, which causes dependencies
in certain parts of the repository to change frequently. Under such
conditions, constructing and maintaining a global graph index with
static tools requires repeatedly rebuilding parts of [ 41,43], or even
the entire [ 39] graph index when dependencies evolve. Since build-
ing such a graph index is a highly time-consuming operation, this
process slows down the developers’ workflow.
Table 2 also shows thatDyCoderconsumes 1.95-2.43 times more
tokens. We argue that this is often acceptable for developers for two
reasons. First, the absolute dollar cost remains low, as generating
each function costs only $0.0039 on average. Second,DyCoder
achieves superior effectiveness while maintaining a high genera-
tion speed. With a relatively low absolute dollar cost,DyCoder
outperforms graph-based baselines while operating at a speed close
to that of similarity-based baselines. Finally, as an out-of-the-box
method,DyCoderrequires no training, which aligns well with
developer practices in real application scenarios [37].
RQ2 Summary:DyCoderachieves a 7.4 ×speedup over the
graph-based baseline, performs almost on par with similarity-
based baselines, and incurs an average cost of only $0.0039 per
generated function, demonstrating its efficiency.
5.3 RQ3: Ablation
DyCoderconsists of two key context components, i.e., similarity-
based context retrieval andDyRetriever. WithinDyRetriever,
multi-hop reasoning with partial graphs serves as a core subcom-
ponent. To analyze their contributions toDyCoder’s effectiveness,
we first separately eliminate each of the key context components

Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation ASE ’26, October 12–16, 2026, Munich, Germany
Table 3: Ablation results with two representative models.
MethodQwen3-Coder-30B GPT-4o-mini
Pass@1 #Token Pass@1 #TokenCoderEvalDyCoder 49.48 13.09k 40.35 11.81k
w/o Similarity-based 42.87↓13.4% 11.61k↓11.3% 33.83↓16.2% 10.13k↓14.2%
w/oDyRetriever41.57↓16.0% 4.67k↓64.3% 34.00↓15.7% 4.60k↓61.0%
w/o Multi-hop Reasoning 46.52↓6.0% 6.81k↓47.9% 37.48↓7.1% 6.69k↓43.4%DevEvalDyCoder 46.88 112.41k 28.39 102.38k
w/o Similarity-based 35.71↓23.8% 100.09k↓11.0% 26.56↓6.5% 88.11k↓13.9%
w/oDyRetriever22.94↓51.1% 45.54k↓59.5% 15.55↓45.2% 37.38k↓63.5%
w/o Multi-hop Reasoning 42.40↓9.6% 69.31k↓38.3% 25.76↓9.3% 63.49k↓38.0%
to construct two main variants (w/o Similarity-basedandw/oDyRe-
triever). Then, we further construct a variant that removes only
the multi-hop reasoning with partial graphs (w/o Multi-hop Rea-
soning). This variant directly uses the entry points obtained in
Section 3.1.1 as the related functions retrieved byDyRetrieverto
generate the target function. We evaluate all variants on CoderEval
and DevEval using the open-source model Qwen3-Coder-30B and
the closed-source model GPT-4o-mini, asDyCoderachieves better
performance gains on CoderEval across both models.
As shown in Table 3,DyCoderoutperforms all three variants
in Pass@1 across CoderEval and DevEval. Specifically, removing
the similarity-based context retrieval leads to noticeable perfor-
mance drops on both datasets, with decreases of 13.4% and 16.2% on
CoderEval, and 23.8% and 6.5% on DevEval under Qwen3-Coder-30B
and GPT-4o-mini, respectively. Similarly, removingDyRetriever
results in even larger degradations, reaching up to 16.0% and 51.1%
on CoderEval and DevEval, respectively. These results demonstrate
that both types of context used inDyCodercontribute to generat-
ing the target function, and that the useful information they provide
exhibits a certain degree of complementarity.
In addition, we find that removing the multi-hop reasoning with
partial graphs leads to consistent performance drops across both
datasets (6.0%–7.1% on CoderEval and 9.3%–9.6% on DevEval). This
component simulates how human developers inspect code along
the dependency graph. Without it,DyCoderbecomes less capable
of identifying helpful functions that can help to generate the target
function, since multi-hop reasoning enables the LLM to flexibly
traverse the code dependency graph and assess whether a code snip-
pet is helpful to the target function. Although removing multi-hop
reasoning leads to a certain performance degradation forDyCoder,
its Pass@1 still outperforms the variant (i.e.,w/oDyRetriever)
that relies solely on similarity-based context. This is because the
w/o Multi-hop Reasoningvariant still preserves the simulation of
how human developers select entry points based on their under-
standing of the repository. Because this simulation provides code
snippets as context from a perspective different from similarity,
it can collect additional contextual information that is useful for
generating the target function. These results further support the
effectiveness ofDyCoderin retrieving context by simulating how
human developers search for relevant information.
RQ3 Summary:The similarity-based context retrieval and
DyRetrieverexhibit strong complementarity, and removing
either of them leads to a 6.5%-23.8% and 15.7%-51.1% perfor-
mance drop, respectively. Multi-hop reasoning with partialTable 4: Performance of different similarity-based method
combined withDyCoderon CoderEval.
MethodModel
Qwen3-Coder-30B GPT-4o-mini
BM25 41.71 34.87
+DyRetriever 48.70↑14.65% 37.48↑7.50%
RLCoder 35.65 34.78
+DyRetriever 43.91↑23.17% 37.91↑9.00%
AlignCoder 38.17 34.78
+DyRetriever 50.00↑30.99% 37.57↑8.00%
graphs plays a crucial role inDyRetriever, which contributes
6.0%-9.6% of the overall performance.
5.4 RQ4: Generalizability
To investigate whether the context collected byDyRetrievercan
generalize and be combined with various similarity-based baselines,
we conduct experiments using the same models on CoderEval. We
replace the similarity-based context retrieval inDyCoderwith
the context retrieved by RLCoder and AlignCoder to examineDy-
Coder’s generalizability to dense retrieval. Additionally, we in-
corporate context retrieved using BM25 to evaluateDyCoder’s
generalizability to sparse retrieval, since BM25 is one of the most
popular sparse retrieval algorithms [ 34] and has already demon-
strated strong performance in the code search domain [ 35]. We do
not combineDyRetrieverwith RepoScope, as RepoScope already
incorporates graph-based context.
Table 4 presents the Pass@1 scores of the two models on
CoderEval. For Qwen3-Coder-30B, combining similarity-based
baselines withDyRetrieveryields an improvement of 14.65%-
30.99%. For GPT-4o-mini, the improvement ranges from 7.5%
to 9.0%. These results indicate that the context collected by
DyRetrieverexhibits low coupling with the context retrieved
by both types of similarity-based methods and shows strong
complementarity with them. This benefit arises fromDyRetriever
’s ability to retrieve context useful for generating the target func-
tion by simulating how human developers retrieve information.
Through performing multi-hop reasoning along the dependency
graph,DyRetrieverexplicitly examines each function along the
traversal path to determine its usefulness. As this perspective
differs from similarity-based retrieval, the two retrieval methods
are complementary. Notably, the improvement ofDyRetrieveron
GPT-4o-mini is smaller than that on Qwen3-Coder-30B. This may
be due to inherent differences in the reasoning capabilities of differ-
ent models, and also suggests thatDyRetrieverexhibits stronger
effectiveness when applied to models with higher reasoning
capacity.
RQ4 Summary:DyRetrievercan be integrated with existing
similarity-based methods to achieve 7.5%-30.99% performance
improvement. This not only demonstrates the generalizability
ofDyRetrieverbut also highlights its advantages and opti-
mization potential as a plug-and-play module.

ASE ’26, October 12–16, 2026, Munich, Germany Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren
6 Discussion
6.1 Are Graphs Constructed by LLMs Accurate?
Constructing a dependency graph for a code repository is far
from trivial, especially for dynamically typed languages such as
Python [ 3]. Prior work [ 3,42] has shown that existing static analy-
sis tools cannot guarantee sufficient accuracy, and an inaccurate
dependency graph may limit the upper bound ofDyRetriever’s
performance. SinceDyRetrieverrelies on the LLM to identify
dependency relations between functions, and LLM hallucination is
unavoidable, it is necessary to ensure the quality of the dependency
graph constructed by the LLM. To investigate whether the depen-
dency relations identified by the LLM are sufficiently accurate for
DyRetrieverafter the post-processing, we manually inspected
the construction process in 60expandoperations and counted
the number of cases in which the constructed dependencies are
completely accurate. Manual inspection shows that for93.22%
of the functions, the LLM can accurately identify all of their
dependencies and construct the dependency graph.
Moreover, we also examine why the LLM is able to accurately
identify dependency relations and summarize two main reasons.
First,DyRetrieverrequires the LLM to identify only downstream
dependencies (i.e., the callee functions). Unlike identifying upstream
dependencies (i.e., the caller functions), which requires a global
analysis, identifying downstream dependencies only requires ex-
amining the callees of a given function. Since recognizing callees is
far easier for an LLM than tracing callers [ 48], the LLM can con-
struct the dependency graph with relatively high accuracy. Second,
identifying the dependencies of an individual function is relatively
easy, since the dependencies of a single function are few and fairly
explicit. For example, among the 60 cases we examined, only one
function had more than ten dependencies, and about 40% of the
functions had fewer than five dependencies. Moreover, their de-
pendencies can be found in either the same code file or the import
statements of the file. Thus, the LLM can easily infer its origins
from the import statements and thereby identify the dependencies.
Despite this, the LLM still cannot achieve 100% identification accu-
racy, which may occur when the given function is long (spanning
hundreds of lines), causing redundant context to interfere with the
LLM’s recognition, or when the function has many dependencies
(above 10), leading the LLM to identify only a subset of them. But
this accuracy is acceptable, since it is already comparable to the
accuracy (77.4%-99.2%) achieved by static analysis tools [42].
6.2 Threats to Validity
6.2.1 Internal Validity.Hyperparameter settings.The number
of retrieved code snippets affects the effectiveness ofDyCoder
compared with the baselines. To ensure a fair comparison, we align
the number of the two types of context retrieved byDyCoderwith
that retrieved by RepoScope [ 33]. The maximum number of hops in
the multi-hop reasoning may affect the effectiveness ofDyCoder.
To mitigate this threat, we conduct a preliminary exploration on
CoderEval [ 53] with hop counts ranging from 0 to 20, and select
the best-performing setting.
Baseline implementation.Since AlignCoder [ 17] and RL-
Coder [ 50] are originally proposed for repository-level code
completion, we adapt them to the code generation setting forcomparison by modifying their prompt template. To mitigate
performance variations introduced by this adaptation, we reuse
all released artifacts from the original papers, including their
embedding models, workflows, and related components. We follow
all hyperparameter settings released in the original papers to
ensure the consistency of reproduced results, to mitigate such
threats.
6.2.2 External Validity.Generalization to other programming
languages.We currently evaluate our method only on Python,
and the generalization to other programming languages may not
yet be fully validated. However, Python is one of the most popu-
lar and widely used languages in repository-level code generation
benchmarks [ 6,25,26,53,54], which can demonstrate the effec-
tiveness ofDyCoder. Moreover, as a dynamically typed language,
it is more challenging to identify dependencies in the application
scenarios. Statically typed languages such as Java provide clearer
and more easily extractable type relations, which may make depen-
dency identification easier for LLMs and thus mitigate this threat
to some extent. SinceDyCoderdoes not rely on static analysis to
obtain effective information, it can be readily generalized to other
programming languages.
7 Related Work
In recent years, repository-level code generation has attracted exten-
sive attention from researchers, as it more closely reflects real-world
production scenarios. Unlike simple code generation tasks, which
only need to generate single functions [ 8,9,31], target functions
of repository-level code generation typically exhibit complex con-
textual dependencies. A lot of benchmarks have been proposed to
evaluate the capability of LLMs on repository-level code generation
tasks, such as RepoEval [ 54], CoderEval [ 53], and SWE-bench [ 20].
Researchers have introduced retrieval-augmented generation (RAG)
techniques into code generation, aiming to provide effective con-
text for repository-level code generation. Specifically, RAG methods
first build an index for the code repository, then retrieve relevant
information based on the requirement description, and finally use
the retrieved content as context to prompt the LLM for generation.
Existing repository-level retrieval-augmented code generation
methods can generally be categorized intosimilarity-based,graph-
based, andagent-basedmethods.Similarity-basedmethods [ 17,54]
compute similarity scores using sparse [ 40] or dense retrieval algo-
rithms [ 56] and select the most similar code snippets as context. For
example, AlignCoder [ 17] first performs sparse retrieval using the
BM25 [ 40] algorithm to generate a code draft, and then employs re-
inforcement learning to train the embedding model AlignRetriever
for dense retrieval. RepoCoder [ 54] designs an iterative retrieval
mechanism that uses the partially generated code to retrieve similar
code snippets from the repository, and then leverages the retrieved
results as domain knowledge to assist the model in regenerating
the code. These methods rely solely on similarity and overlook the
program-structural semantics in code repositories, which may lead
to inferior performance of code generation.
Graph-basedmethods [ 5,19,32,33] model code dependency re-
lations by static analysis and design retrieval algorithms based on
manually designed rules to retrieve code snippets over the code

Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation ASE ’26, October 12–16, 2026, Munich, Germany
graph, thereby supplementing context from a structured perspec-
tive of the code. For example, GraphCoder [ 32] constructs a code
context graph (CCG) to model control flow and data dependence
at the line level and manually designs a similarity metric between
slices of CCG to perform retrieval and capture context in a coarse-
to-fine manner. DRACO [ 5] constructs a data-flow graph for the
repository and retrieves code snippets by computing similarity be-
tween module names through string matching based on import
statements, using depth-first search (DFS). RepoScope [ 33] builds a
repository structural semantic graph and designs a call chain pre-
diction algorithm that starts from entities imported by the target
function and assigns similarity scores to call chains via DFS to en-
able multi-view context retrieval. However, the retrieval algorithms
used by these methods [ 5,32,33] depend on manually designed
rules, such as differential similarity metrics, which make them ex-
hibit limited flexibility in real application scenarios. In addition,
these methods construct complex code dependency graphs through
static analysis, which is both time-consuming and difficult to main-
tain for large repositories with frequently changing dependencies.
Recently, agents have attracted increasing attention in the field
of code generation due to their strong reasoning and tool-calling
capabilities.
Agent-basedmethods [ 19,27,55] design a series of tools that
enable agents to access the code repository and retrieve context. For
example, CodeAgent [ 55] integrates five programming tools and
implements four agent strategies, enabling large language models
to autonomously determine retrieval targets. GraphCodeAgent [ 27]
enables the LLM to identify retrieval targets by establishing a map-
ping between the requirement graph and the structural-semantic
code graph. However, the retrieval tools equipped by these agents
still fall undersimilarity-basedorgraph-basedretrieval. Since the
limitations of these two categories of methods remain unresolved,
agent-basedmethods still exhibit limited performance.
Different from existing methods, we designDyRetrieverby
simulating human developers’ behaviors when collecting helpful
context. Specifically,DyRetrieverfirst uses an LLM to select some
entry point functions based on the structure of the repository, and
then designs a multi-hop reasoning algorithm to flexibly traverse
the code dependency graph and determine whether a function can
help to generate the target function. This breaks the dependence of
manually designed rules and makesDyRetrieveradaptively adjust
its search direction, resulting in better flexibility. When constructing
code dependency graphs,DyRetrieveruses a partial graph instead
of a global graph to reduce the time overhead. Since no tools can be
used to construct a partial graph dynamically, we design an LLM-
based partial dependency graph construction method to achieve it.
The partial dependency graph is discarded after use, requiring no
additional maintenance cost and allowing the method to adapt to
continuously evolving code repositories.
8 Conclusion and Future Work
We proposedDyRetriever, an effective and efficient context re-
trieval method via partial dependency graphs for repository-level
code generation. By simulating human developers’ behavior when
collecting helpful context,DyRetrieverleverages multi-hop rea-
soning over the code dependency graph to collect helpful contextflexibly, breaking the dependence of manually designed rules in
retrieval algorithms. In addition,DyRetrieveremploys an LLM-
driven code dependency graph construction method that dynami-
cally builds partial code dependency graphs during multi-hop rea-
soning and discards them after use to reduce the unnecessary time
overhead of pre-constructing a global dependency graph using
static analysis tools, which requires no maintenance. We integrate
DyRetrieverwith similarity-based retrieval and proposeDyCoder.
Extensive experiments on the widely used repository-level code gen-
eration benchmarks CoderEval and DevEval show thatDyCoder
outperforms baseline methods in both generation success rate and
speed, achieving up to a 7.4 ×speedup and a 98.6% improvement in
Pass@1.
In the future, we plan to extendDyCoderto more program-
ming languages and evaluate it in more complex code generation
scenarios (e.g., feature addition). We also intend to further investi-
gate LLM-based code dependency graph construction techniques to
achieve more accurate dependency graph generation. In addition,
we will explore leveraging the plug-and-play nature ofDyCoder
to strengthen agent-based code generation methods.
Data Availability
The replication package is available at https://doi.org/10.
5281/zenodo.19235859. Our official repository is available at
https://github.com/ZJU-CTAG/DyCoder.
Acknowledgments
This research is supported by the National Natural Science Foun-
dation of China (No. 92582107 and No. 62302430) and Zhejiang
Provincial Natural Science Foundation of China (No. LZ25F020003
and No. LQ24F020017).
References
[1]Zhangqian Bi, Yao Wan, Zheng Wang, Hongyu Zhang, Batu Guan, Fangxin Lu,
Zili Zhang, Yulei Sui, Hai Jin, and Xuanhua Shi. 2024. Iterative Refinement of
Project-Level Code Context for Precise Code Generation with Compiler Feedback.
InFindings of the Association for Computational Linguistics. 2336–2353.
[2]Leah Bidlake, Eric Aubanel, and Daniel Voyer. 2025. Investigating the Progression
of the Mental Models Formed by Programmers Learning Parallel Programming.
ACM Transactions on Computing Education25, 1, Article 5 (2025).
[3]Islem Bouzenia, Bajaj Piyush Krishan, and Michael Pradel. 2024. DyPyBench: A
benchmark of executable python software.Proceedings of the ACM on Software
Engineering1, FSE (2024), 338–358.
[4]Mark Chen. 2021. Evaluating large language models trained on code.arXiv
preprint arXiv:2107.03374(2021).
[5]Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflow-guided retrieval augmen-
tation for repository-level code completion. InProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
7957–7977.
[6]Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain,
Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth,
et al.2023. Crosscodeeval: A diverse and multilingual benchmark for cross-file
code completion.Advances in Neural Information Processing Systems36 (2023),
46701–46723.
[7]Gang Fan, Chengpeng Wang, Rongxin Wu, Xiao Xiao, Qingkai Shi, and Charles
Zhang. 2020. Escaping dependency hell: finding build dependency errors with the
unified dependency graph. InProceedings of the 29th ACM SIGSOFT International
Symposium on Software Testing and Analysis. 463–474.
[8]Lishui Fan, Zhongxin Liu, Haoye Wang, Lingfeng Bao, Xin Xia, and Shanping Li.
2025. FGit: Fault-Guided Fine-Tuning for Code Generation. In2025 40th IEEE/ACM
International Conference on Automated Software Engineering. 1338–1350.
[9]Lishui Fan, Yu Zhang, Mouxiang Chen, and Zhongxin Liu. 2026. ReCode: Rein-
forcing Code Generation with Reasoning-Process Rewards. InProceedings of the
64th Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 43896–43914.

ASE ’26, October 12–16, 2026, Munich, Germany Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren
[10] Scott D Fleming, Chris Scaffidi, David Piorkowski, Margaret Burnett, Rachel
Bellamy, Joseph Lawrance, and Irwin Kwan. 2013. An information foraging
theory perspective on tools for debugging, refactoring, and reuse tasks.ACM
Transactions on Software Engineering and Methodology22, 2 (2013), 1–41.
[11] Google. 2026. google/pytype: A static type analyzer for Python code. https:
//github.com/google/pytype. Accessed: 2026-01-30.
[12] Wenchao Gu, Juntao Chen, Yanlin Wang, Tianyue Jiang, Xingzhe Li, Mingwei
Liu, Xilin Liu, Yuchi Ma, and Zibin Zheng. 2025. What to Retrieve for Effective
Retrieval-Augmented Code Generation? An Empirical Study and Beyond.arXiv
preprint arXiv:2503.20589(2025).
[13] Zhanming Guan, Junlin Liu, Jierui Liu, Chao Peng, Dexin Liu, Ningyuan Sun,
Bo Jiang, Wenchao Li, Jie Liu, and Hang Zhu. 2024. ContextModule: Improving
Code Completion via Repository-level Contextual Information.arXiv preprint
arXiv:2412.08063(2024).
[14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card.arXiv preprint arXiv:2410.21276(2024).
[15] Naman Jain, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang,
Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Livecodebench:
Holistic and contamination free evaluation of large language models for code. In
International Conference on Learning Representations, Vol. 2025. 58791–58831.
[16] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. 2026. A
survey on large language models for code generation.ACM Transactions on
Software Engineering and Methodology35, 2 (2026), 1–72.
[17] Tianyue Jiang, Yanli Wang, Yanlin Wang, Daya Guo, Ensheng Shi, Yuchi Ma, Jiachi
Chen, and Zibin Zheng. 2025. AlignCoder: Aligning Retrieval with Target Intent
for Repository-Level Code Completion. In2025 IEEE/ACM 40th International
Conference on Automated Software Engineering. 971–982.
[18] Zhonghao Jiang, David Lo, and Zhongxin Liu. 2025. Agentic Software Issue Res-
olution with Large Language Models: A Survey.arXiv preprint arXiv:2512.22256
(2025).
[19] Zhonghao Jiang, Xiaoxue Ren, Meng Yan, Wei Jiang, Yong Li, and Zhongxin Liu.
2025. Issue Localization via LLM-Driven Iterative Code Graph Searching. In
2025 40th IEEE/ACM International Conference on Automated Software Engineering.
3034–3045.
[20] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve
real-world github issues?. InInternational Conference on Learning Representations,
Vol. 2024. 54107–54157.
[21] Mehdi Keshani, Georgios Gousios, and Sebastian Proksch. 2024. Frankenstein:
fast and lightweight call graph generation for software builds.Empirical Software
Engineering29, 1 (2024), 1.
[22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serving with pagedattention. In
Proceedings of the 29th symposium on operating systems principles. 611–626.
[23] Thomas D LaToza and Brad A Myers. 2010. Developers ask reachability ques-
tions. InProceedings of the 32Nd ACM/IEEE International Conference on Software
Engineering-Volume 1. 185–194.
[24] Joseph Lawrance, Christopher Bogart, Margaret Burnett, Rachel Bellamy, Kyle
Rector, and Scott D Fleming. 2010. How programmers debug, revisited: An infor-
mation foraging theory perspective.IEEE Transactions on Software Engineering
39, 2 (2010), 197–215.
[25] Jia Li, Ge Li, Xuanming Zhang, Yunfei Zhao, Yihong Dong, Zhi Jin, Binhua Li,
Fei Huang, and Yongbin Li. 2024. Evocodebench: An evolving code generation
benchmark with domain-specific evaluations.Advances in Neural Information
Processing Systems37 (2024), 57619–57641.
[26] Jia Li, Ge Li, Yunfei Zhao, Yongmin Li, Huanyu Liu, Hao Zhu, Lecheng Wang,
Kaibo Liu, Zheng Fang, Lanshen Wang, et al .2024. Deveval: A manually-
annotated code generation benchmark aligned with real-world code repositories.
InFindings of the Association for Computational Linguistics. 3603–3614.
[27] Jia Li, Xianjie Shi, Kechi Zhang, Ge Li, Zhi Jin, Lei Li, Huangzhao Zhang, Jia Li,
Fang Liu, Yuwei Zhang, Zhengwei Tao, Yihong Dong, Yuqi Zhu, and Chongyang
Tao. 2025. GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-
Augmented Repo-Level Code Generation.arXiv preprint arXiv:2504.10046(2025).
[28] Zongjie Li and Shuai Wang. 2025. Reasoning as a resource: Optimizing fast and
slow thinking in code generation models. InProceedings of the 1st ACM SIGPLAN
International Workshop on Language Models and Programming Languages. 1–5.
[29] Dianshu Liao, Shidong Pan, Xiaoyu Sun, Xiaoxue Ren, Qing Huang, Zhenchang
Xing, Huan Jin, and Qinying Li. 2024. A3-codgen: A repository-level code gener-
ation framework for code reuse with local-aware, global-aware, and third-party-
library-aware.IEEE Transactions on Software Engineering(2024).
[30] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu,
Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al .2025. DeepSeek-
V3.2: Pushing the Frontier of Open Large Language Models.arXiv preprint
arXiv:2512.02556(2025).
[31] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your
code generated by chatgpt really correct? rigorous evaluation of large languagemodels for code generation.Advances in Neural Information Processing Systems
36 (2023), 21558–21572.
[32] Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, and
Qianxiang Wang. 2024. Graphcoder: Enhancing repository-level code completion
via coarse-to-fine retrieval based on code context graph. InProceedings of the 39th
IEEE/ACM International Conference on Automated Software Engineering. 570–581.
[33] Yang Liu, Li Zhang, Fang Liu, Zhuohang Wang, Donglin Wei, Zhishuo Yang, Kechi
Zhang, Jia Li, and Lin Shi. 2025. Reposcope: Leveraging call chain-aware multi-
view context for repository-level code generation.arXiv preprint arXiv:2507.14791
(2025).
[34] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey Svy-
atkovskiy. 2022. ReACC: A Retrieval-Augmented Code Completion Framework.
InProceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 6227–6240.
[35] Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin
Li. 2025. Alibaba lingmaagent: Improving automated issue resolution via com-
prehensive repository exploration. InProceedings of the 33rd ACM International
Conference on the Foundations of Software Engineering. 238–249.
[36] Will McGugan, Martin Larralde, Giampaolo Cimino, Geoff Jukes, and contributors.
2026. PyFilesystem2: Python’s Filesystem Abstraction Layer. https://github.com/
PyFilesystem/pyfilesystem2. Accessed: 2025-12-06.
[37] Melissa Z Pan, Negar Arabzadeh, Riccardo Cogo, Yuxuan Zhu, Alexander Xiong,
Lakshya A Agrawal, Huanzhi Mao, Emma Shen, Sid Pallerla, Liana Patel, et al .
2025. Measuring agents in production.arXiv preprint arXiv:2512.04123(2025).
[38] Ruwei Pan and Hongyu Zhang. 2025. Modularization is Better: Effective Code
Generation with Modular Prompting.arXiv preprint arXiv:2503.12483(2025).
[39] Thomas Reps, Susan Horwitz, and Mooly Sagiv. 1995. Precise interprocedural
dataflow analysis via graph reachability. InProceedings of the 22nd ACM SIGPLAN-
SIGACT symposium on Principles of programming languages. 49–61.
[40] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval
3, 4 (2009), 333–389.
[41] Barbara G Ryder and Frank Tip. 2001. Change impact analysis for object-oriented
programs. InProceedings of the 2001 ACM SIGPLAN-SIGSOFT workshop on Program
analysis for software tools and engineering. 46–53.
[42] Vitalis Salis, Thodoris Sotiropoulos, Panos Louridas, Diomidis Spinellis, and
Dimitris Mitropoulos. 2021. Pycg: Practical call graph generation in python. In
2021 IEEE/ACM 43rd International Conference on Software Engineering. 1646–1657.
[43] Jordan Samhi, Marc Miltenberger, Marco Alecci, Steven Arzt, Tegawendé Bis-
syandé, and Jacques Klein. 2025. Do you have 5 min? Improving Call Graph
Analysis with Runtime Information. InProceedings of the 33rd ACM International
Conference on the Foundations of Software Engineering. 540–544.
[44] Weifeng Sun, Naiqi Huang, Meng Yan, Li Huang, Zhongxin Liu, Xiao Liu, and
David Lo. 2026. Cost-Effective Adversarial Attacks Against Code LLM with
Model Attention.IEEE Transactions on Software Engineering(2026).
[45] Weifeng Sun, Naiqi Huang, Meng Yan, Zhongxin Liu, Hongyan Li, Yan Lei, and
David Lo. 2026. On-the-Fly Generation-Quality Enhancement of Deep Code
Models via Model Collaboration.ACM Transactions on Software Engineering and
Methodology35, 6 (2026), 1–40.
[46] Yicheng Tao, Yao Qin, and Yepang Liu. 2025. Retrieval-Augmented Code Gen-
eration: A Survey with Focus on Repository-Level Approaches.arXiv preprint
arXiv:2510.04905(2025).
[47] Tree-sitter Contributors. 2026. py-tree-sitter Documentation. https://tree-sitter.
github.io/py-tree-sitter/. Accessed: 2026-01-30.
[48] Ashwin Prasad Shivarpatna Venkatesh, Rose Sunil, Samkutty Sabu, Amir M Mir,
Sofia Reis, and Eric Bodden. 2025. An empirical study of large language models
for type and call graph analysis in Python and JavaScript.Empirical Software
Engineering30, 6 (2025), 167.
[49] Jianxun Wang and Yixiang Chen. 2023. A review on code generation with llms:
Application and evaluation. In2023 IEEE International Conference on Medical
Artificial Intelligence. 284–289.
[50] Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma,
and Zibin Zheng. 2025. RLCoder: Reinforcement Learning for Repository-Level
Code Completion. In2025 IEEE/ACM 47th International Conference on Software
Engineering. 1140–1152.
[51] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[52] Zezhou Yang, Sirong Chen, Cuiyun Gao, Zhenhao Li, Xing Hu, Kui Liu, and Xin
Xia. 2025. An empirical study of retrieval-augmented code generation: Challenges
and opportunities.ACM Transactions on Software Engineering and Methodology
(2025).
[53] Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang,
Ying Li, Qianxiang Wang, and Tao Xie. 2024. Codereval: A benchmark of prag-
matic code generation with generative pre-trained models. InProceedings of the
46th IEEE/ACM International Conference on Software Engineering. 1–12.
[54] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi
Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code

Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation ASE ’26, October 12–16, 2026, Munich, Germany
completion through iterative retrieval and generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. 2471–2484.
[55] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. CodeAgent: Enhancing
Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level
Coding Challenges. InProceedings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers). 13643–13658.
[56] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al .2025. Qwen3 Embedding:
Advancing Text Embedding and Reranking Through Foundation Models.arXivpreprint arXiv:2506.05176(2025).
[57] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira
Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, et al .
2025. Bigcodebench: Benchmarking code generation with diverse function calls
and complex instructions. InInternational Conference on Learning Representations,
Vol. 2025. 66602–66656.
Received 2026-03-26; accepted 2026-06-18
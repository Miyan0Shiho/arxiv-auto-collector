# SaraCoder: Orchestrating Semantic and Structural Cues for Profit-Oriented Repository-Level Code Completion

**Authors**: Xiaohan Chen, Zhongying Pan, Quan Feng, Yu Tian, Shuqun Yang, Mengru Wang, Lina Gong, Yuxia Geng, Piji Li, Xiang Chen

**Published**: 2025-08-13 11:56:05

**PDF URL**: [http://arxiv.org/pdf/2508.10068v1](http://arxiv.org/pdf/2508.10068v1)

## Abstract
Retrieval-augmented generation (RAG) for repository-level code completion
commonly relies on superficial text similarity, leading to results plagued by
semantic misguidance, redundancy, and homogeneity, while also failing to
resolve external symbol ambiguity. To address these challenges, we introduce
Saracoder, a Hierarchical Feature-Optimized retrieval framework. Its core
Hierarchical Feature Optimization module systematically refines candidates by
distilling deep semantic relationships, pruning exact duplicates, assessing
structural similarity with a novel graph-based metric that weighs edits by
their topological importance, and reranking results to maximize both relevance
and diversity. Furthermore, an External-Aware Identifier Disambiguator module
accurately resolves cross-file symbol ambiguity via dependency analysis.
Extensive experiments on the challenging CrossCodeEval and RepoEval-Updated
benchmarks demonstrate that Saracoder significantly outperforms existing
baselines across multiple programming languages and models. Our work proves
that systematically refining retrieval results across multiple dimensions
provides a new paradigm for building more accurate and robust repository-level
code completion systems.

## Full Text


<!-- PDF content starts -->

SaraCoder: Orchestrating Semantic and Structural Cues for Profit-Oriented
Repository-Level Code Completion
Xiaohan Chen1, Zhongying Pan2, Quan Feng3, Yu Tian4, Shuqun Yang1,
Mengru Wang5, Lina Gong1, Yuxia Geng1, Piji Li1, Xiang Chen1*
1MIIT Key Laboratory of Pattern Analysis and Machine Intelligence,
College of Computer Science and Technology,
Nanjing University of Aeronautics and Astronautics
2Huaneng Information Technology Co., Ltd.3Hunan Vanguard Group Corporation Co., Ltd.
4Tsinghua University5Zhejiang University6PowerChina Huadong Engineering Co., Ltd.
{cxh030414, xiang chen}@nuaa.edu.cn
Abstract
Retrieval-augmented generation (RAG) for repository-level
code completion commonly relies on superficial text similar-
ity, leading to results plagued by semantic misguidance, re-
dundancy, and homogeneity, while also failing to resolve ex-
ternal symbol ambiguity. To address these challenges, we in-
troduce S ARACODER , a Hierarchical Feature-Optimized re-
trieval framework. Its core Hierarchical Feature Optimiza-
tion module systematically refines candidates by distilling
deep semantic relationships, pruning exact duplicates, assess-
ing structural similarity with a novel graph-based metric that
weighs edits by their topological importance, and rerank-
ing results to maximize both relevance and diversity. Fur-
thermore, an External-Aware Identifier Disambiguator mod-
ule accurately resolves cross-file symbol ambiguity via de-
pendency analysis. Extensive experiments on the challenging
CrossCodeEval and RepoEval-Updated benchmarks demon-
strate that S ARACODER significantly outperforms existing
baselines across multiple programming languages and mod-
els. Our work proves that systematically refining retrieval re-
sults across multiple dimensions provides a new paradigm for
building more accurate and robust repository-level code com-
pletion systems.
Introduction
Code Large Language Models (CLLMs) (Izadi, Gismondi,
and Gousios 2022; Li et al. 2022b; Allal et al. 2023)
built on the Transformer architecture and trained on mas-
sive code datasets, have significantly empowered modern
software development. They compress vast programming
knowledge into billions or trillions of parameters, leading to
successful deployments that streamline real-world develop-
ment and boost productivity. However, as software projects
grow in complexity, traditional Code LLMs struggle with
repository-level completion. Their limited context windows
cannot grasp long-range, cross-file dependencies, preventing
a deep understanding of project semantics and architecture.
This results in localized, often insufficient, suggestions. Di-
rectly processing entire repositories also causes inefficient
retrieval, high latency, and an inability to leverage crucial
higher-level software engineering knowledge.
*Corresponding author.To tackle these challenges, Retrieval-Augmented Genera-
tion (RAG) (Tang et al. 2023; Zan et al. 2022; Zhang et al.
2023) offers a compelling solution. RAG overcomes the lim-
itations of traditional models by incorporating an external
knowledge retrieval mechanism. Within this framework, an
efficient retriever locates relevant code snippets, API doc-
umentation, or type definitions from the repository in real
time. A generation model then integrates these retrieved re-
sults with the current context to create syntactically cor-
rect, semantically consistent, and standard-compliant code
suggestions. RAG is progressively becoming a founda-
tional technology for achieving accurate and trustworthy
repository-level intelligent code completion.
Repository-level code completion can be divided into two
key scenarios: In-File and Cross-File. In-File completion re-
lies solely on the current file’s context. It is a fundamen-
tal and high-frequency task in repository-level intelligence.
Cross-File completion involves dependencies on symbols
from other files. The true complexity of modern software
lies in these inter-module interactions. Cross-File comple-
tion is a crucial aspect of repository-level intelligence. How-
ever, some existing retrieval-augmented solutions rely too
heavily on similarity search without proper filtering, and
they often lack the crucial injection of symbolic relation-
ships. This strategy exposes significant flaws in two types
of scenarios, as shown in Figure 1: (1) In-File Scenarios:
Similarity-based retrieval exhibits two critical flaws in prac-
tice. (a)Misleading Superficial Similarity occurs when re-
trievers recall code fragments that share surface-level char-
acteristics but lack semantic relevance to the current con-
text. The approach suffers from (b)Redundant Retrieval
Homogenization , where duplicate or near-identical snippets
are repeatedly retrieved, unnecessarily consuming valuable
context window capacity that could be allocated for more
diverse and informative content. (2) Cross-File Scenarios:
Besides the limitations also observed in In-File scenarios,
(c)External Symbol Ambiguity is a more severe limita-
tion. Retrievers fail to capture critical dependencies such as
unreferenced classes, interface constraints, or architectural
rules, triggering error cascades including type inference fail-
ures, method signature mismatches, fabricated parameter-
s/attributes, and architecture violations.arXiv:2508.10068v1  [cs.SE]  13 Aug 2025

(a) Misleading Surface Similarity
Question：
positive_test_data = [-3, 0, 7, 2, -1, 4]
result = count_items(positive_test_data)
print(f"The number of positive numbers 
in the list is: {result}") 
def count_items(items):
    count = 0
    for item in items:
    // Complete the line 
Traditional retrieval case：
positive_test_data = [-3, 0, 7, 2, -1, 4]
result = count_items(positive_test_data)
print(f"The sum of numbers in the list is: 
{result}") 
def sum_items(items):
    count = 0  
    for item in items:
        count += item
    return count(b) Redundant Retrieval Homogenization (c) External Symbol Ambiguity
Context： // Complete the last line of the code 
Map<String, String> columnNames = ... ;  
for (String columnName : updateColumn) {  
    String fieldName = columnNames.get(columnName);  
    if (StringUtils.isNotEmpty(fieldName)) {    
CLM Answer： TableFieldUtil.setFieldValue(r, fieldName, columnName);  
What We Want：TableFieldUtil.getFieldMap(r.getClass());  //from TableTest.java
for (ConfigItem item : changedItems) {  
    String fieldName = configMap.get(item.key());  
    if (fieldName != null) {   
        TableFieldUtil.setFieldValue(configObj, 
fieldName, item.value());  
    }  
}  Retrieval case：package com.example.service; 
import com.yourcompany.util.TableFieldUtil; 
....
Map<String, String> columnNames = ... ;  
for (String columnName : updateColumn) {  
    String fieldName = columnNames.get(columnName);  
      if (StringUtils.isNotEmpty(fieldName)) { File to complete
Traditional retrieval prompt
No information about the class methods of 
TableFieldUtil in the retrieved code cases
Retrieval without 
Filtering 
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / countfrom calculate.py
from table.py
from average.py
Traditional 
Retrieval Queue... ...
... ...from calculate.py
Our Retrieval 
QueueMore computing resources
 Semantic equivalence
Retrieval with 
Filtering 
 Fewer computing resources
Obscure the real demands
Misuse of methods
 Fabricate method parameters
Misleading code generation
positive_test_data = [-3, 0, 7, -1, 4]
result = count_items(positive_test_data)
print(f"The number of positive numbers 
in the list is: {result}")
def count_items(items):
    count = 0
    for item in items:
        if item > 0:
            count += 1
    return countOur retrieval case：
 Accurate code generationOur retrieval prompt
Context：// Complete the last line of the code 
Map<String, String> columnNames = ... ;  
for (String columnName : updateColumn) {  
    String fieldName = columnNames.get(columnName);  
    if (StringUtils.isNotEmpty(fieldName)) {    
//from TableTest.java
for (ConfigItem item : changedItems) {  
    String fieldName = configMap.get(item.key()); 
    ... Retrieval case：
External Symbols:import com.yourcompany.util.TableFieldUtil;
//public class TableFieldUtil {
//public static Map<String, Field> 
getFieldMap(Class<?> clazz) {Information About  
TableFieldUtil
Correct use of methods Authentic method parameters
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / countdef calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count
Figure 1: The pitfalls of pure similarity retrieval and the highlights of S ARACODER . Pink boxes illustrate traditional retrieval
results based purely on surface similarity, while green boxes demonstrate results from our method S ARACODER .
To address these challenges, we introduce S ARA-
CODER , aSemantic- Aware Code Retrieval framework with
Redundancy Reduction and Ambiguity Suppression for
Repository-Level Code Completion. S ARACODER imple-
ments a hierarchical feature-optimized retrieval-augmented
code completion framework, that goes beyond surface-level
similarity by extracting deeper relationships such as control
and data flow from code snippets. To further refine retrieval
quality, we use semantic alignment distillation to capture
deep semantic relationships. This is paired with a graph-
based structural similarity metric that assesses a candidate’s
structural proximity to the target context, weighting edit op-
erations by their topological importance. To combat redun-
dant retriveal homogenization, we integrate deduplication
pruning and diversity-aware reranking, ensuring relevance
while maximizing diversity. Furthermore, to resolve exter-
nal symbol ambiguity of Cross-File scenarios in repository-
level code completion, we add an external-aware identifier
disambiguator that analyzes project-level dependencies. Our
key contributions are:
• We propose S ARACODER , a hierarchical and feature-
optimized retrieval-augmented code completion frame-
work. Designed for repository-level code completion,
SARACODER provides a comprehensive solution that pri-
oritizes high-quality, diverse, and streamlined suggestions.
• SARACODER seamlessly integrates semantic alignment,
redundancy pruning, topological proximity assessment,
and diversity-driven reranking, resolving misleading su-
perficial similarity and redundant retrieval homogeniza-
tion. It also incorporates an external-aware identifier dis-ambiguator that accurately resolves external symbol ambi-
guities by leveraging project-wide import dependencies.
• SARACODER ’s coordination of semantic and structural
signals enables strong performance even in large, resource-
constrained codebases. Its design allows it to complement
orthogonally other cross-file methods, delivering synergis-
tic improvements when used in combination.
Related Work
Current RAG methods for repo-level code completion
mainly rely on code similarity or cross-file dependencies.
Similar Code Snippet Retrieval for RAG
This approach enhances the quality of code LLM gen-
eration by retrieving semantically similar code snippets
and integrating them into prompts, mimicking the refer-
ence behaviors of programmers. CodeSearchNet (Husain
et al. 2020) pioneers large-scale code corpora construction,
providing retrieval-based completion references; CodeRe-
triever (Li et al. 2022a) integrates pretrained models like
CodeBERT (Feng et al. 2020) to enhance complex sce-
nario handling; ReACC(Lu et al. 2022) combines vector and
string-based retrieval to significantly optimize long-code
processing; GraphCoder (Liu et al. 2024) improves code
completion by using program dependencies for structured
representations, allowing coarse-to-fine retrieval for Python
and Java. However, like many other methods, its dependency
analysis does not fully grasp deep semantic relationships
in code. Additionally, most approaches rely too much on

surface-level textual similarity. This often results in redun-
dant retrieved content, wasting resources.
Cross-File Dependency Retrieval Augmentation
This method approaches code completion in complex repos-
itories by leveraging cross-file code context (e.g., depen-
dencies, dataflow, and subsequent similar code). Inspired
by Ding et al.’s (Ding et al. 2023) observation that sub-
sequent content of high-similarity snippets effectively in-
forms completion, it injects these snippets into prompts. CO-
COMIC (Ding et al. 2024) dynamically fuses the context of
this file with the cross-file entities retrieved by CCFinder
(compressed into [SUM] vectors) through a joint atten-
tion mechanism to achieve location-aware code completion.
DraCo (Cheng, Wu, and Hu 2024) extends this paradigm
through dataflow-guided retrieval, parsing private reposito-
ries into code entities and constructing a repository-specific
context graph reflecting dependencies. DraCo retrieves pre-
cise contextual knowledge from this graph to generate well-
structured prompts, overcoming cross-file information bar-
riers and repository-specific accuracy gaps. Current limita-
tions include Python-exclusive implementation with type-
sensitive dependencies lacking multilingual support.
Repository-Level Code Completion Evaluation
Although traditional code completion benchmarks (Chen
et al. 2021; Austin et al. 2021) focus primarily on local
file contexts and isolated code snippets, the inherent com-
plexity of modern software development characterized by
vast codebases and intricate in-file dependencies requires a
more sophisticated evaluation approach. To address this gap,
specialized benchmarks like RepoEval (Zhang et al. 2023),
CrossCodeEval (Ding et al. 2023), RepoBench (Liu, Xu, and
McAuley 2024), and ReccEval (Cheng, Wu, and Hu 2024)
allow for evaluation of repository-level code completion.
These establish standardized test scenarios spanning multi-
ple languages, project scales, and contextual complexity lev-
els to ensure rigorous and reproducible assessment. For ex-
ample, RepoEval targets In-File completion; CrossCodeEval
and ReccEval assess Cross-File completion requiring deep
contextual understanding.
Method
As shown in Figure 2, S ARACODER is a hierarchical
feature-optimized retrieval-enhanced code completion
framework , applicable to both In-File and Cross-File com-
pletion scenarios. Formally, given a code context Ccontext =
{x1, x2,···, xn}and its containing file path F, the task
aims to predict the next statement ˜y.
Database and Initial Candidate Construction
To better represent code logic, we introduce a multi-level
code context graph model that integrates control flow, data
dependency, and control dependency (Liu et al. 2024). This
structured representation offers enhanced generalization ca-
pabilities compared to serialization methods, enabling more
effective capture of task-relevant context and facilitating
easier adaptation to other languages. We utilize programslicing to generate precisely mapped, task-relevant sub-
graphs from source code on-demand, constructing struc-
tured codebases tailored to support specific analysis tasks.
When a code completion task request occurs, we extract the
Unfinished Code Context Ccontext and the Import Statements
Ifrom the code file . Ccontext is then used to retrieve an ini-
tial candidate set of topk×p1code snippets Cvia text
similarity from structured codebases.
Hierarchical Feature Optimization (HF OP)
Semantic Alignment Distillation (SAD) SAD addresses
Superficial Similarity Misguidance by leveraging the Graph-
codeBERT (Guo et al. 2021), a pretrained model specialized
in code understanding, to capture deep semantic relation-
ships between code snippets. First, the query code Qand
candidate set Care tokenized into subword sequences and
uniformly padded or truncated to a fixed length L= 512 .
Subsequently, during the feature encoding phase, a 768-
dimensional semantic vector vsis extracted for each code
units∈Q∪C, with vector space standardized through
L2 normalization. When code repositories lack sufficient
repetitive or relevant code, standard filtering methods are
too strict, often leading to zero-candidate scenarios . This
scarcity of reference material then hurts the accuracy of
large language models. To fix this “one-size-fits-all” prob-
lem, we introduce a new dynamic quantile threshold mecha-
nism. During the dynamic filtering phase, the cosine similar-
ity set S={cos(vQ, vc)|c∈C}is computed between the
query vector vQand all candidate vectors vc. An adaptive
threshold τ=quantile (S,0.75)is set at the 75th percentile,
outputting filtered results CSAD ={c|cos(vQ, vc)≥τ}.
To reduce redundant computation overhead and improve ef-
ficiency, a caching mechanism stores encoding results for
high-frequency code.
Redundancy-Aware Pruning (RAP) This module im-
plements lightweight hash-based deduplication via exact
text matching. Using the MD5 algorithm (Rivest 1992),
it generates 128-bit hash fingerprints ( single computation
≈0.02ms,memory footprint 32bytes/hash) to eliminate
verbatim duplicates from candidate set CSAD with minimal
computational cost, significantly reducing downstream over-
head. The module maintains a global hash set Hseento dy-
namically track processed sample fingerprints: for each can-
didate c∈CSAD, if its MD5 hash hc/∈Hseen,cis added
to deduplicated result set CRAP andHseenis updated. This
achieves real-time processing with O(N)time complexity.
After SAD processing, the number of code snippets requir-
ing MD5 hashing is limited and their structure is fixed by
syntactic and semantic constraints. The MD5 collision resis-
tance (theoretical probability ≈1.47×10−18) is sufficient
for strict sensitivity. Additionally, MD5’s superior speed and
lower memory footprint provide optimal cost-performance.
Topological Proximity Metric (TPM) At this layer, the
decaying subgraph edit distance (D-SED) is introduced to
1To ensure a high-quality final candidate set of topkresults,
we expand the initial candidate pool to topk×p, allowing more
candidates to participate in the Hierarchical Feature Optimization.

Repository 
Hierarchical Feature Optimization
 Retrieve  
code snippets 
similar to 
unfinished 
code contextSemantic Alignment 
DistillationRedundancy-Aware 
PruningTopological 
Proximity Metric Diversity-Aware 
Reranking
 External-Aware Identifier Disambiguator
Unfinished code 
and file path
（input） Unfinished 
Code Context
Import 
Statement 
Per-character MD5 hashingGraphcodeBERT 
semantic filterp×top_k
Subgraph Edit 
Distance(SED) 
Calculator
Top SED 
EliminationReserve
MMR Score Calculation
Static analysis of 
the import 
statement
 Enhanced 
prompts
Search in 
identifier 
symbol tableFunction
def preprocess_data(raw: list) -> list:
     """Clean and normalize raw data"""
     cleaned = [x.strip() for x in raw if x]
     mean = sum(cleaned) / len(cleaned)
     return [x/mean for x in cleaned]Class
class TextEncoder:
def __init__(self, vocab_size=5000):
def forward(self, inputs):
def Encoder(self, inputs):Prompt Generation
Similar Code 
Snippet
External 
Symbols 
Enhancement
Unfinished 
Code ContextPrompt 
Predicted Statement
(Output)
 Unfinished Code Processing Database Construction
Program slicingsInduced
graph slices
Structured 
codebases
Making a retrieval requestFigure 2: An illustration of S ARACODER framework. (1) Database Construction. This phase constructs a key-value codebase.
This involves using a slicing algorithm to create induced graph slices, which are then precisely mapped to source code snippets.
(2) Code Retrieval. This phase takes code context as input and retrieves similar code, then refines suggestions via Hierarchical
Feature Optimization. Concurrently, an External-Aware Identifier Disambiguator clarifies external symbols via dependency
analysis, delivering highly accurate candidates. (3) Code Generation. This phase generates prompts by integrating outputs
from code retrieval with the code completion context. These prompts are then fed into an LLM to predict completion statements.
measure the graph similarity between the query graph Gq
and the candidate graph Gc(Ranjan et al.; Zeng et al.).
D−SED ( Gq, Gc) =X
op=Oγl(op)·c(op) (1)
Editing operations Oare the set of operations to transform
GctoGq, include adding, deleting, and modifying nodes
and edges. Each operation op∈Ohas a cost c(op)and
a hop count l(op)from its “core node”. For simplicity, we
choose the node with the largest ID as core node. Oper-
ations closer to the core exert greater structural influence.
γ∈(0,1)is an attenuation factor that reduces the cost
weight for operations farther from the core node. After com-
puting D-SED scores for each candidate c∈CRAP, we
compute a composite score sas a weighted sum of text sim-
ilarity (calculated during initial candidate generation) and
structural similarity (D-SED scores). Subsequently, we gen-
erate QTPM = [(c,s), . . .], ordered in descending score s.
Diversity-Aware Reranking (DAR) This module imple-
ments a variability-aware ranking model based on the Max-
imal Marginal Relevance (MMR) (2017) algorithm to max-
imize result diversity while preserving relevance. It ad-
dresses homogeneity in traditional rankings through adver-
sarial similarity calculation and dynamic weight adjustment.
MMR = arg max
ci∈Sh
λ·Sim1(ci, q)−
(1−λ)·max
cj∈CfinalSim2(ci, cj)i(2)
Scontains items (c, s)∈QTPM that have not been se-
lected into Cfinal yet. Sim 1represents the relevance ( si=
Similar Code 
Snippet
External 
Symbols 
Enhancement
Unfinished 
Code Context
Predicted Statement embeddings = encoder.forward(processed)# Here are some relevant code fragments from other files of the repo:
# --------------------------------------------------
# The below code fragment can be found in:
# pipeline/integration.py
# --------------------------------------------------
# def run_pipeline(inputs, config):
#     """Run full data processing pipeline"""
#     cleaned = clean_inputs(inputs)
#     normalized = normalize(cleaned)
#     validated = validate(normalized)
#     if config['encode']:
#         encoded = encoder.encode(validated)
#         return encoded
#     return validated
# --------------------------------------------------
# Cross-file reference snippets:
# --------------------------------------------------
from utils.data_processor import preprocess_data
# def preprocess_data(raw: list) -> list:
#     """Clean and normalize raw data"""
# ...
#     return [x/mean for x in cleaned]
from utils.data_processor import DataValidator
# class DataValidator:
#     def check_integrity(self, data):
from models.transformer import TextEncoder
# class TextEncoder:
#     def __init__(self, vocab_size=5000):
#     def forward(self, inputs):
# --------------------------------------------------
# Based on above, complete the next statement of the following codes:
def run_pipeline(input_texts):
    processed = preprocess_data(input_texts)
    validator = DataValidator()
    if validator.check_integrity(processed):
        encoder = TextEncoder(vocab_size=10000)Prompt generated by Saracoder
The output of the large model after receiving the promptFrom 
InputGenerated 
from Code 
RetrievalFigure 3: Prompt template used in S ARACODER .
π2◦ιci(S)) of item cito query q. Sim 2denotes the maximum
cosine similarity between ciand any item cjin the selected
setCfinal .λis a trade-off parameter that balances the em-
phasis between relevance (Sim 1) and diversity (Sim 2).
External-Aware Identifier Disambiguator (EAID)
This module enhances knowledge through external identi-
fier augmentation. Firstly, file-level entity modeling parses
code per file F, extracting method entities Emethod (function-
s/class methods with identifier, alias, line range [lstart, lend],

parameter signature, scope) and class entities Eclass(class
definitions with identifier, alias, line range [lstart, lend], mem-
ber mappings) that built in the F. After that, it generates
a structured identifier symbol table STlib={identifier 7→
syntax features }, where identifier corresponds to either:
(1) the unique identifier of a method entity ∀e∈Emethod ,
or (2) the unique identifier of a class entity ∀c∈Eclass,
with the mapped syntax features containing all associated
attributes for that entity. Subsequently, the dependency reso-
lution mechanism processes all import statements ( I) within
the unfinshed file. For Intra-project Cross-module Refer-
ence, this phase retrieves complete entities ( Elib) from the
pre-built entity library ( STlib) by determining their corre-
sponding file paths ( p). These paths are constructed through
decomposition of module components derived from either
dotted names (e.g., my.module.MyClass ) or relative im-
ports (e.g., from .sub module import MyClass ),
which are subsequently joined using directory separators ( /)
and appended with the .py file extension. For standard and
third-party libraries, the system constructs a lightweight ref-
erence table Text={canonical name 7→alias}to efficiently
manage external dependencies without full entity resolution.
The enhanced prompts PE=I⊕Elib⊕Text.
Prompt Generation
Following code retrieval and external link resolution, S ARA-
CODER employs an external LLM to generate subsequent
statements. The final prompt Pfinalis constructed by concate-
nating three components: the external symbols enhance-
ment PEwhere entities are ordered by file import sequence
reflecting call probability decay—with function entities pop-
ulated with complete function bodies and class entities con-
taining variable tables and method definitions; the simi-
lar code snippets Cfinal containing code snippets strictly
sorted in ascending order of similarity and annotated with
source paths; the unfinished code context Ccontext . This ar-
chitecture follows Pfinal=Cfinal⊕PE⊕Ccontext . (Figure 3).
Experiments
Experimental Settings
To evaluate the performance of S ARACODER on repository-
level code completion, we have formulated the following
four research questions (RQs):
RQ1 Effectiveness in Cross-File Scenarios: How does
SARACODER perform when cross-context under-
standing is required, compared to other methods?
RQ2 Cost Analysis in Cross-File Scenarios: How does
SARACODER ’s resource consumption compare to
GraphCoder in Cross-File scenarios?
RQ3 Synergistic Gain Property: How does S ARACODER
perform when integrated orthogonally with other
cross-file methods?
RQ4 Advantage in In-File Scenarios: How does S ARA-
CODER perform on tasks without cross-context re-
quirements and what are its advantages?Datasets We primarily utilize two datasets here: Cross-
CodeEval and RepoEval-Updated. Table 1 shows details.
•CrossCodeEval (Ding et al. 2023): This benchmark eval-
uates code completion in complex Cross-File scenarios like
type inference and dependency analysis. It is ideal for as-
sessing performance that requires a deep understanding of
code across multiple files.
•RepoEval-Updated (Liu et al. 2024): Expanded from Re-
poEval (2023), this new version, includes repositories of
varying scales and duplication rates, offering a better way
to evaluate In-File completion performance.
We use CrossCodeEval to test models on code comple-
tion tasks that involve complex cross-file dependencies.
RepoEval-Updated will assess other aspects like basic syn-
tax, common API usage, and local context understanding.
CrossCodeEval RepoEval-Updated
Python Java Python Java
Total Repositories 471 239 10 8
Total Files 1368 745 3258 8260
Total Task cases 2665 2139 2000 1600
Applicable Scenarios Cross-File completion In-File completion
Table 1: CrossCodeEval vs. RepoEval-Updated comparison.
Evaluation Indicators Setting In this study, the following
several evaluation indicators are used to assess the effect of
code completion (Lu et al. 2021; Ding et al. 2023).
•Code Exact Match (EM): Proportion of generated code
exactly matching the ground truth. EM is given only for a
perfect semantic and syntactic match.
•Identifier Exact Match (ID EM): The percentage of
identifiers (variables, functions, etc.) perfectly matching
the ground code. A high ID EM means the model correctly
understands the context and chooses the right names.
•Identifier F1 Score (ID F1): A more nuanced evaluation
of identifier matching by combining precision and recall.
This integration offers a more comprehensive assessment
of identifier completion quality.
•Edit Similarity (ES): Similarity metric between generated
and ground-truth code based on edit distance. It tolerates
slight variations, requiring the completed code to be highly
similar in structure, syntax, and token order to the target.
Baseline Setting We employ the following five meth-
ods as controls to evaluate the effectiveness of retrieval-
augmented generation (RAG) in code completion: No RAG
(Zero-shot-only baseline), Shifted RAG (Target-context dy-
namic retrieval), Vanilla RAG (Exemplar-similarity fixed re-
trieval), Repocoder (Iterative-fragment integration (Zhang
et al. 2023)), Graphcoder (Structure-modeling CCG utiliza-
tion (Liu et al. 2024)).
Model Selection In this experiment, we select
Codegen2-7b ,Codegen25-7b (Nijkamp et al.
2023b,a), CodeLlama-7b-Instruct (Rozi `ere et al.
2024) and deepseek-coder-6.7b-instruct (Guo
et al. 2024) for code completion task inference.

Language MethodsCodegen2-7b Codegen25-7b deepseek-coder-6.7b-instruct CodeLlama-7b-Instruct
Code Match Identifier Match Code Match Identifier Match Code Match Identifier Match Code Match Identifier Match
EM ES EM F1 EM ES EM F1 EM ES EM F1 EM ES EM F1
PythonNo Rag 0.00 13.38 0.00 2.24 0.00 13.26 0.00 2.10 0.00 4.51 0.00 0.57 0.00 13.27 0.00 2.22
Shifted Rag 4.84 46.67 11.48 42.72 7.40 48.88 14.09 44.62 8.19 50.18 14.77 46.69 6.91 49.12 13.60 45.12
Vanilla Rag 9.48 50.97 17.15 47.81 12.39 53.92 23.61 53.14 13.00 54.00 26.63 55.77 11.45 52.93 19.23 50.49
Repocoder 12.47 54.08 21.57 51.89 16.62 56.80 25.73 57.85 17.11 58.11 26.71 56.46 15.14 56.28 24.56 54.22
Graphcoder 10.88 52.36 19.68 49.73 14.54 55.29 23.38 52.61 15.53 57.05 24.29 55.01 13.30 55.41 22.63 52.91
SARACODER 15.04 56.03 24.44 54.68 18.36 58.30 27.28 56.22 19.72 59.93 28.52 58.26 17.91 58.37 27.77 56.82
JavaNo Rag 1.03 21.79 0.64 16.86 1.50 21.77 24.78 24.78 6.40 35.79 10.42 32.90 0.93 20.83 1.96 16.17
Shifted Rag 6.08 46.09 12.11 43.76 5.89 38.00 10.23 36.44 5.84 36.19 11.64 35.23 6.73 43.75 12.71 41.68
Vanilla Rag 9.30 47.42 15.71 45.69 10.38 40.76 15.29 39.91 8.88 33.59 15.01 33.51 10.93 45.01 17.81 44.08
Repocoder 10.71 41.83 16.18 41.51 12.16 42.38 17.63 41.69 9.58 34.25 15.76 34.13 13.23 46.01 19.87 45.14
Graphcoder 8.13 45.18 14.35 43.32 8.42 36.77 12.85 35.84 7.39 32.34 12.76 32.05 8.51 40.57 14.96 39.57
SARACODER 11.73 46.69 18.37 45.47 11.40 39.33 16.22 38.97 11.92 34.55 17.72 34.99 12.95 42.71 19.54 42.33
Table 2: Performance comparison on the CrossCodeEval dataset. Numbers are shown in percentage ( %). The top results are
bolded , and the second best are underlined .
1 2 3 4 5 6 7 8 9 1078910
35.035.536.036.537.0
top_kEM(%)
Code Match (java)
ES(%)Saracoder(ES)Graphcoder(ES) Graphcoder(EM)
Saracoder(EM)
1 2 3 4 5 6 7 8 9 1011121314
33343536
top_kID_EM(%)
Identifier Match （java）
ID_F1(%)Saracoder(ID_F1)Graphcoder(ID_F1) Graphcoder(ID_EM)
Saracoder(ID_EM)
1 2 3 4 5 6 7 8 9 101214161820
4850525456
top_kEM(%)
Code Match (python)
ES(%)Saracoder(ES)Graphcoder(ES) Graphcoder(EM)
Saracoder(EM)
1 2 3 4 5 6 7 8 9 102022242628
4648505254
top_kID_EM(%)
Identifier Match (python)
ID_F1(%)Saracoder(ID_F1)Graphcoder(ID_F1) Graphcoder(ID_EM)
Saracoder(ID_EM)
Figure 4: Impact of top k on CrossCodeEval. (The two on the left are Java tasks, and the two on the right are Python tasks.)
Main Results
For RQ1: Dominant Cross-File Code Accuracy. Table 2
illustrates that S ARACODER surpasses the top-performing
Repocoder on the CrossCodeEval dataset, achieving an aver-
age improvement of 1.50in EM, 0.77in ES, 1.11in ID EM,
and0.61in ID F1. This indicates S ARACODER provides
more effective information and generates code with higher
semantic accuracy, better capturing intended functionality.
The enhanced ID EM further shows S ARACODER ’s supe-
rior ability to interpret context and select appropriate iden-
tifiers. These advancements effectively mitigate misleading
superficial similarity and external symbol ambiguity, lead-
ing to more reliable and contextually relevant code. For
Java code completion, S ARACODER shows better EM and
IDEM, with slightly lower ES and F1 scores. Because
of Java’s static typing and verbosity, even small structural
changes (e.g., bracket placement) can heavily affect ES/F1
scores, despite functional correctness.
For RQ2: Cost-Optimized Accuracy Advantage In
Cross-File. We experiment with code completion effi-
ciency using codegen25-7b and Graphcoder. Our goal is
to see how retrieving more similar cases ( topk) impacts
accuracy. Since using fewer topkcases saves input to-
kens2, this study shows the balance between resources and
accuracy. Our experiments demonstrate significant perfor-
mance saturation for both retrieval methods when topk
reaches 3-4, with no observable fluctuations upon increas-
ing to topk= 10 . SARACODER achieves comprehen-
2For relevant explanations, please refer to the Appendix D.2sive superiority in Python tasks (e.g., 9.4% EM improve-
ment ) while maintaining advantages in Java tasks despite
a marginal 0.1decrease in ES . Crucially, under resource-
constrained topk= 1 conditions: all Python metrics out-
perform the baseline; three Java metrics (EM/ES/ID EM)
show improvements; and Java ID F1 initially trails ( 35.22
vs.35.27) but ultimately surpasses the baseline at saturation
(35.84vs.35.77). Our method achieves performance break-
throughs at lower computational cost (stable at topk≈4)
by reducing redundant and homogeneous cases (Figure 4).
For RQ3: Synergistic Integration of S ARACODER
Achieves Enhanced Completion. We examine two
prominent methods that demonstrate exceptional perfor-
mance in Cross-File scenarios. (1) Repocoder (Zhang et al.
2023), distinct from the original, assumes that if code snip-
pets are similar, their subsequent content is also likely rele-
vant. In the next search round, it specifically gets the code
following those similar snippets (hereafter referred to as Re-
pocoder). (2) Draco (Cheng, Wu, and Hu 2024), analyzes
code to create entity dependency graphs, allowing detailed
background knowledge retrieval. It then uses this informa-
tion to create structured prompts. Currently, Draco only
works with Python. As shown in Table 3, adding our method
significantly boosts all four Python metrics (by 3.42to4.52)
compared to using Repocoder or Draco alone. For Java, our
method improves EM by 0.45 and ID EM by 0.33over Re-
pocoder, showing S ARACODER exhibits significant syner-
gistic gain property with existing cross-file methods.3
3You can find the causes of synergistic gains in Appendix D.3.

Language MethodsCodegen2-7b Codegen25-7b CodeLlama
Code Match Identifier Match Code Match Identifier Match Code Match Identifier Match
EM ES EM F1 EM ES EM F1 EM ES EM F1
PythonNo Rag 5.44 57.85 11.71 42.22 7.77 60.52 14.45 45.40 9.49 61.97 16.44 47.36
Shift Rag 4.87 58.36 11.64 42.91 7.44 60.20 14.17 44.78 6.95 60.35 13.75 45.36
Vanilla Rag 9.52 61.87 17.42 48.01 12.43 63.81 20.74 51.00 11.48 63.66 19.42 50.72
SARACODER 12.16 54.16 18.37 45.47 11.50 44.90 16.22 38.95 13.32 48.04 19.54 42.34
Repocoder 12.50 64.48 21.87 52.09 16.66 66.67 25.99 55.06 15.19 66.24 24.74 54.39
Repocoder + S ARACODER 15.94 +3.44 66.43 +1.95 25.73 +3.86 54.80 +2.71 19.49 +2.83 68.53 +1.86 28.90 +2.91 57.49 +2.43 19.19 +4.00 68.45 +2.21 29.05 +4.31 57.47 +3.08
Draco 20.06 66.33 29.13 56.53 22.93 68.70 32.45 59.34 23.50 68.56 32.57 59.49
Draco + S ARACODER 24.06 +4.00 69.40 +3.07 34.00 +4.87 61.12 +4.59 27.05 +4.12 71.86 +3.16 36.80 +4.35 63.48 +4.14 27.20 +3.7 72.01 +3.45 37.29 +1.72 64.35 +4.85
JavaNo Rag 0.00 25.92 0.05 17.48 0.00 25.46 0.05 17.61 0.00 25.17 0.00 17.23
Shift Rag 6.45 54.84 12.11 43.75 6.08 44.73 10.27 36.46 7.11 50.96 12.72 41.68
Vanilla Rag 9.68 55.71 15.71 45.71 10.47 47.09 15.29 39.93 11.31 51.48 17.81 44.09
SARACODER 12.16 54.16 18.37 45.47 11.50 44.90 16.22 38.95 13.32 48.04 19.54 42.34
Repocoder 11.22 56.89 17.72 47.41 10.85 47.93 16.18 41.50 13.60 52.17 19.87 45.14
Repocoder + S ARACODER 11.50 +0.28 56.09 -0.80 17.72 0.00 46.96 -0.45 11.27 +0.42 46.53 -1.40 16.41 +0.23 40.47 -1.03 14.26 +0.66 50.79 -1.38 20.62 +0.75 44.38 -0.76
Table 3: Performance benefits of S ARACODER when integrated orthogonally with other cross-file approaches ( %).
Language MethodsCodegen2-7b Codegen25-7b deepseek-coder-6.7b-instruct CodeLlama-7b-Instruct
Code Match Identifier Match Code Match Identifier Match Code Match Identifier Match Code Match Identifier Match
EM ES EM F1 EM ES EM F1 EM ES EM F1 EM ES EM F1
PythonNo Rag 17.40 32.54 23.75 30.21 19.55 34.48 25.75 32.16 11.50 30.33 15.30 22.39 17.35 33.05 23.55 30.53
Shifted Rag 32.70 59.22 40.10 55.66 36.45 61.96 43.20 58.31 20.90 42.95 26.50 38.88 33.90 60.28 41.50 56.39
Vanilla Rag 38.70 63.58 46.45 60.43 42.25 66.26 48.75 62.79 22.20 41.48 27.85 37.58 40.30 65.03 47.55 61.06
Repocoder 37.60 61.98 45.10 58.47 40.55 64.48 46.85 60.71 21.35 40.18 26.65 35.93 39.60 63.71 47.05 59.78
Graphcoder 42.40 65.73 49.45 62.07 44.65 67.59 51.00 63.82 28.50 44.63 33.35 42.63 43.90 67.26 51.15 63.51
SARACODER 42.60 65.92 50.15 62.61 44.50 67.79 51.10 63.84 28.25 46.91 33.45 42.95 45.00 68.27 52.00 63.97
JavaNo Rag 6.55 16.84 9.15 8.84 5.35 16.21 9.05 8.65 6.40 20.61 7.75 8.73 6.85 17.11 9.45 9.06
Shifted Rag 30.87 62.52 43.94 61.01 26.63 58.46 37.75 56.57 28.00 55.12 36.81 53.39 35.38 64.63 45.56 62.98
Vanilla Rag 33.50 63.82 45.44 62.08 32.00 61.52 41.56 59.48 21.13 46.51 31.19 45.15 38.56 66.46 48.06 64.90
Repocoder 30.13 60.01 42.31 57.10 28.75 57.73 37.88 54.46 22.19 46.93 32.06 44.26 35.38 62.33 44.44 59.63
Graphcoder 37.75 66.19 50.68 64.77 36.63 64.74 46.13 62.62 28.81 55.75 40.06 53.61 43.00 69.68 52.69 67.85
SARACODER 37.93 67.06 50.93 65.48 36.75 65.54 46.44 62.61 29.75 56.38 40.75 54.18 42.88 69.37 52.00 67.37
Table 4: Performance comparison on the RepoEval-Updated dataset. Numbers are shown in percentage ( %). The top results are
bolded , and the second best are underlined .
python java05101520EM(%)Saracoder
Saracoder - EAIDSaracoder - EAID - HF_OP
Saracoder - EAID - CCG
python java020406080ES(%)Saracoder
Saracoder - EAIDSaracoder - EAID - HF_OP
Saracoder - EAID - CCG
python java0102030ID_EM(%)Saracoder
Saracoder - EAIDSaracoder - EAID - HF_OP
Saracoder - EAID - CCG
python java0204060ID_F1(%)Saracoder
Saracoder - EAIDSaracoder - EAID - HF_OP
Saracoder - EAID - CCG
Figure 5: Ablation study. (Each three-data-point group rep-
resents CodeGen2-7B, CodeGen2.5-7B, and CodeLlama-
7B-Instruct models. Bar lengths show their average perfor-
mance, with I-shaped error bars indicating S.D.)
For RQ4: Enhanced In-File Accuracy and Resource
Efficiency. On the RepoEval-Updated dataset (Table 4),
SARACODER shows superior semantic and identifier accu-
racy (surpassing the top-performing Graphcoder: +0.547
EM,+0.737ES,+0.125IDEM, and +0.667F1) for both
Python and Java code completion. The cost analysis (Ap-
pendix D.1) further indicates S ARACODER generally per-
forms better and exhibits higher stability across most Python
metrics (excluding EM) and all Java metrics. This makes
it particularly effective for resource-constrained environ-ments, especially at lower topkvalues. However, S ARA-
CODER ’s gains over Graphcoder in code and identifier
matching are smaller here than on CrossCodeEval. This
is primarily because RepoEval-Updated projects contain a
higher prevalence of similar code snippets, resulting in re-
duced code diversity. Overall, the conclusions align with
those from the CrossCodeEval dataset.
Ablation Study
To understand the importance of each part of S ARACODER ,
we conduct ablation tests on the CrossCodeEval dataset
(Figure 5). “-EAID” indicates disabling External-Aware
Identifier Disambiguator , resulting in the loss of exter-
nal dependency integration capabilities; “-HF OP” denotes
removing Hierarchical Feature Optimization , canceling
the similar fragment screening mechanism; “-CCG” indi-
cates disabling the code context graph , so it lost the under-
standing of code structure. The ablation experiments demon-
strate that the complete S ARACODER achieves optimal per-
formance, with all components positively contributing to
repository-level completion. Notably, even without EAID,
SARACODER still outperforms Shift RAG and Vanilla RAG,
and even surpasses Repocoder in Python tasks4, proving that
HFOP screening substantially enhances case quality.
Conclusion and Outlook
This paper proposes S ARACODER , a profit-oriented,
repository-level code completion method. It integrates se-
4Detailed data can be found in Appendix, Table 4.

mantic topology with disambiguation to tackle superfi-
cial similarity distraction, retrieval redundancy and rigidity,
and external symbol ambiguity. The results of experiments
on CrossCodeEval and RepoEval-Updated datasets show
SARACODER ’s excellent accuracy and efficiency. However,
though S ARACODER balances accuracy with reduced re-
source use by pruning unnecessary code examples, more op-
timal methods need further investigation.
References
Allal, L.; Li, R.; Kocetkov, D.; Mou, C.; Akiki, C.; Fer-
randis, C.; Muennighoff, N.; Mishra, M.; Gu, A.; Dey, M.;
Umapathi, L.; Anderson, C.; Zi, Y .; Poirier, J.; Schoelkopf,
H.; Troshin, S.; Abulkhanov, D.; Romero, M.; Lappert, M.;
Toni, F.; R ´ıo, B.; Liu, Q.; Bose, S.; Bhattacharyya, U.; Zhuo,
T.; Yu, I.; Villegas, P.; Zocca, M.; Mangrulkar, S.; Lansky,
D.; Nguyen, H.; Contractor, D.; Villa, L.; Li, J.; Bahdanau,
D.; Jernite, Y .; Hughes, S.; Fried, D.; Guha, A.; Vries, H.;
and Werra, L. 2023. SantaCoder: don’t reach for the stars!
Workingpaper, arXiv.
Austin, J.; Odena, A.; Nye, M.; Bosma, M.; Michalewski,
H.; Dohan, D.; Jiang, E.; Cai, C.; Terry, M.; Le, Q.; and
Sutton, C. 2021. Program Synthesis with Large Language
Models. arXiv:2108.07732.
Carbonell, J.; and Goldstein, J. 2017. The Use of MMR,
Diversity-Based Reranking for Reordering Documents and
Producing Summaries. SIGIR Forum , 51(2): 209–210.
Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; de Oliveira Pinto,
H. P.; Kaplan, J.; Edwards, H.; Burda, Y .; Joseph, N.; Brock-
man, G.; Ray, A.; Puri, R.; Krueger, G.; Petrov, M.; Khlaaf,
H.; Sastry, G.; Mishkin, P.; Chan, B.; Gray, S.; Ryder, N.;
Pavlov, M.; Power, A.; Kaiser, L.; Bavarian, M.; Winter, C.;
Tillet, P.; Such, F. P.; Cummings, D.; Plappert, M.; Chantzis,
F.; Barnes, E.; Herbert-V oss, A.; Guss, W. H.; Nichol, A.;
Paino, A.; Tezak, N.; Tang, J.; Babuschkin, I.; Balaji, S.;
Jain, S.; Saunders, W.; Hesse, C.; Carr, A. N.; Leike, J.;
Achiam, J.; Misra, V .; Morikawa, E.; Radford, A.; Knight,
M.; Brundage, M.; Murati, M.; Mayer, K.; Welinder, P.; Mc-
Grew, B.; Amodei, D.; McCandlish, S.; Sutskever, I.; and
Zaremba, W. 2021. Evaluating Large Language Models
Trained on Code. arXiv:2107.03374.
Cheng, W.; Wu, Y .; and Hu, W. 2024. Dataflow-Guided Re-
trieval Augmentation for Repository-Level Code Comple-
tion. In Ku, L.-W.; Martins, A.; and Srikumar, V ., eds., Pro-
ceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , 7957–
7977. Bangkok, Thailand: Association for Computational
Linguistics.
Ding, Y .; Wang, Z.; Ahmad, W.; Ramanathan, M. K.; Nal-
lapati, R.; Bhatia, P.; Roth, D.; and Xiang, B. 2024. Co-
CoMIC: Code Completion by Jointly Modeling In-file and
Cross-file Context. In Calzolari, N.; Kan, M.-Y .; Hoste,
V .; Lenci, A.; Sakti, S.; and Xue, N., eds., Proceedings
of the 2024 Joint International Conference on Compu-
tational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024) , 3433–3445. Torino, Italia: ELRA
and ICCL.Ding, Y .; Wang, Z.; Ahmad, W. U.; Ding, H.; Tan, M.; Jain,
N.; Ramanathan, M. K.; Nallapati, R.; Bhatia, P.; Roth, D.;
and Xiang, B. 2023. CrossCodeEval: A Diverse and Mul-
tilingual Benchmark for Cross-File Code Completion. In
Thirty-seventh Conference on Neural Information Process-
ing Systems Datasets and Benchmarks Track .
Feng, Z.; Guo, D.; Tang, D.; Duan, N.; Feng, X.; Gong,
M.; Shou, L.; Qin, B.; Liu, T.; Jiang, D.; and Zhou, M.
2020. CodeBERT: A Pre-Trained Model for Programming
and Natural Languages. In Cohn, T.; He, Y .; and Liu, Y .,
eds., Findings of the Association for Computational Lin-
guistics: EMNLP 2020 , 1536–1547. Online: Association for
Computational Linguistics.
Guo, D.; Ren, S.; Lu, S.; Feng, Z.; Tang, D.; LIU, S.; Zhou,
L.; Duan, N.; Svyatkovskiy, A.; Fu, S.; Tufano, M.; Deng,
S. K.; Clement, C.; Drain, D.; Sundaresan, N.; Yin, J.; Jiang,
D.; and Zhou, M. 2021. GraphCodeBERT: Pre-training
Code Representations with Data Flow. In International Con-
ference on Learning Representations .
Guo, D.; Zhu, Q.; Yang, D.; Xie, Z.; Dong, K.; Zhang, W.;
Chen, G.; Bi, X.; Wu, Y .; Li, Y . K.; Luo, F.; Xiong, Y .; and
Liang, W. 2024. DeepSeek-Coder: When the Large Lan-
guage Model Meets Programming – The Rise of Code Intel-
ligence. arXiv:2401.14196.
Husain, H.; Wu, H.-H.; Gazit, T.; Allamanis, M.; and
Brockschmidt, M. 2020. CodeSearchNet Challenge: Evalu-
ating the State of Semantic Code Search. arXiv:1909.09436.
Izadi, M.; Gismondi, R.; and Gousios, G. 2022. CodeFill:
multi-token code completion by jointly learning from struc-
ture and naming sequences. In Proceedings of the 44th In-
ternational Conference on Software Engineering , ICSE ’22,
401–412. New York, NY , USA: Association for Computing
Machinery. ISBN 9781450392211.
Li, X.; Gong, Y .; Shen, Y .; Qiu, X.; Zhang, H.; Yao, B.; Qi,
W.; Jiang, D.; Chen, W.; and Duan, N. 2022a. CodeRe-
triever: A Large Scale Contrastive Pre-Training Method for
Code Search. In Goldberg, Y .; Kozareva, Z.; and Zhang,
Y ., eds., Proceedings of the 2022 Conference on Empiri-
cal Methods in Natural Language Processing , 2898–2910.
Abu Dhabi, United Arab Emirates: Association for Compu-
tational Linguistics.
Li, Y .; Choi, D.; Chung, J.; Kushman, N.; Schrittwieser, J.;
Leblond, R.; Eccles, T.; Keeling, J.; Gimeno, F.; Lago, A. D.;
Hubert, T.; Choy, P.; de Masson d’Autume, C.; Babuschkin,
I.; Chen, X.; Huang, P.-S.; Welbl, J.; Gowal, S.; Cherepanov,
A.; Molloy, J.; Mankowitz, D. J.; Robson, E. S.; Kohli, P.;
de Freitas, N.; Kavukcuoglu, K.; and Vinyals, O. 2022b.
Competition-level code generation with AlphaCode. Sci-
ence, 378(6624): 1092–1097.
Liu, T.; Xu, C.; and McAuley, J. 2024. RepoBench: Bench-
marking Repository-Level Code Auto-Completion Systems.
InThe Twelfth International Conference on Learning Rep-
resentations .
Liu, W.; Yu, A.; Zan, D.; Shen, B.; Zhang, W.; Zhao,
H.; Jin, Z.; and Wang, Q. 2024. GraphCoder: Enhanc-
ing Repository-Level Code Completion via Coarse-to-fine
Retrieval Based on Code Context Graph. In Proceedings

of the 39th IEEE/ACM International Conference on Auto-
mated Software Engineering , ASE ’24, 570–581. New York,
NY , USA: Association for Computing Machinery. ISBN
9798400712487.
Lu, S.; Duan, N.; Han, H.; Guo, D.; Hwang, S.-w.; and
Svyatkovskiy, A. 2022. ReACC: A Retrieval-Augmented
Code Completion Framework. In Muresan, S.; Nakov, P.;
and Villavicencio, A., eds., Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , 6227–6240. Dublin, Ireland: As-
sociation for Computational Linguistics.
Lu, S.; Guo, D.; Ren, S.; Huang, J.; Svyatkovskiy, A.;
Blanco, A.; Clement, C.; Drain, D.; Jiang, D.; Tang, D.;
Li, G.; Zhou, L.; Shou, L.; Zhou, L.; Tufano, M.; GONG,
M.; Zhou, M.; Duan, N.; Sundaresan, N.; Deng, S. K.; Fu,
S.; and LIU, S. 2021. CodeXGLUE: A Machine Learn-
ing Benchmark Dataset for Code Understanding and Gen-
eration. In Vanschoren, J.; and Yeung, S., eds., Proceed-
ings of the Neural Information Processing Systems Track on
Datasets and Benchmarks , volume 1.
Nijkamp, E.; Hayashi, H.; Xiong, C.; Savarese, S.; and
Zhou, Y . 2023a. CodeGen2: Lessons for Training LLMs on
Programming and Natural Languages. arXiv:2305.02309.
Nijkamp, E.; Pang, B.; Hayashi, H.; Tu, L.; Wang, H.; Zhou,
Y .; Savarese, S.; and Xiong, C. 2023b. CodeGen: An Open
Large Language Model for Code with Multi-Turn Program
Synthesis. arXiv:2203.13474.
Ranjan, R.; Grover, S.; Medya, S.; Chakaravarthy, V .; Sab-
harwal, Y .; and Ranu, S. 2022. GREED: A Neural Frame-
work for Learning Graph Distance Functions. In Oh, A. H.;
Agarwal, A.; Belgrave, D.; and Cho, K., eds., Advances in
Neural Information Processing Systems .
Rivest, R. 1992. RFC1321: The MD5 Message-Digest Al-
gorithm.
Rozi `ere, B.; Gehring, J.; Gloeckle, F.; Sootla, S.; Gat, I.; Tan,
X. E.; Adi, Y .; Liu, J.; Sauvestre, R.; Remez, T.; Rapin, J.;
Kozhevnikov, A.; Evtimov, I.; Bitton, J.; Bhatt, M.; Ferrer,
C. C.; Grattafiori, A.; Xiong, W.; D ´efossez, A.; Copet, J.;
Azhar, F.; Touvron, H.; Martin, L.; Usunier, N.; Scialom,
T.; and Synnaeve, G. 2024. Code Llama: Open Foundation
Models for Code. arXiv:2308.12950.
Tang, Z.; Ge, J.; Liu, S.; Zhu, T.; Xu, T.; Huang, L.;
and Luo, B. 2023. Domain Adaptive Code Completion
via Language Models and Decoupled Domain Databases.
arXiv:2308.09313.
Zan, D.; Chen, B.; Yang, D.; Lin, Z.; Kim, M.; Guan, B.;
Wang, Y .; Chen, W.; and Lou, J.-G. 2022. CERT: Continual
Pre-training on Sketches for Library-oriented Code Genera-
tion. In Raedt, L. D., ed., Proceedings of the Thirty-First
International Joint Conference on Artificial Intelligence,
IJCAI-22 , 2369–2375. International Joint Conferences on
Artificial Intelligence Organization. Main Track.
Zeng, Z.; Tung, A. K. H.; Wang, J.; Feng, J.; and Zhou, L.
2009. Comparing stars: on approximating graph edit dis-
tance. Proc. VLDB Endow. , 2(1): 25–36.
Zhang, F.; Chen, B.; Zhang, Y .; Keung, J.; Liu, J.; Zan,
D.; Mao, Y .; Lou, J.-G.; and Chen, W. 2023. RepoCoder:Repository-Level Code Completion Through Iterative Re-
trieval and Generation. In Bouamor, H.; Pino, J.; and Bali,
K., eds., Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing , 2471–2484. Sin-
gapore: Association for Computational Linguistics.
Reproducibility Checklist
Instructions for Authors:
This document outlines key aspects for assessing repro-
ducibility. Please provide your input by editing this .tex
file directly.
For each question (that applies), replace the “Type your
response here” text with your answer.
Example: If a question appears as
\question {Proofs of all novel claims
are included } {(yes/partial/no) }
Type your response here
you would change it to:
\question {Proofs of all novel claims
are included } {(yes/partial/no) }
yes
Please make sure to:
• Replace ONLY the “Type your response here” text and
nothing else.
• Use one of the options listed for that question (e.g., yes,
no,partial , orNA).
•Not modify any other part of the \question com-
mand or any other lines in this document.
You can \input this .tex file right before
\end{document }of your main file or compile it as
a stand-alone document. Check the instructions on your
conference’s website to see if you will be asked to provide
this checklist with your paper or separately.
The questions start here
1. General Paper Structure
1.1. Includes a conceptual outline and/or pseudocode de-
scription of AI methods introduced (yes/partial/no/NA)
yes
1.2. Clearly delineates statements that are opinions, hypoth-
esis, and speculation from objective facts and results
(yes/no) yes
1.3. Provides well-marked pedagogical references for less-
familiar readers to gain background necessary to repli-
cate the paper (yes/no) yes

2. Theoretical Contributions
2.1. Does this paper make theoretical contributions?
(yes/no) yes
If yes, please address the following points:
2.2. All assumptions and restrictions are stated clearly
and formally (yes/partial/no) yes
2.3. All novel claims are stated formally (e.g., in theorem
statements) (yes/partial/no) yes
2.4. Proofs of all novel claims are included (yes/par-
tial/no) yes
2.5. Proof sketches or intuitions are given for complex
and/or novel results (yes/partial/no) yes
2.6. Appropriate citations to theoretical tools used are
given (yes/partial/no) yes
2.7. All theoretical claims are demonstrated empirically
to hold (yes/partial/no/NA) yes
2.8. All experimental code used to eliminate or disprove
claims is included (yes/no/NA) yes
3. Dataset Usage
3.1. Does this paper rely on one or more datasets? (yes/no)
yes
If yes, please address the following points:
3.2. A motivation is given for why the experiments
are conducted on the selected datasets (yes/par-
tial/no/NA) yes
3.3. All novel datasets introduced in this paper are in-
cluded in a data appendix (yes/partial/no/NA) NA
3.4. All novel datasets introduced in this paper will be
made publicly available upon publication of the pa-
per with a license that allows free usage for research
purposes (yes/partial/no/NA) NA
3.5. All datasets drawn from the existing literature (po-
tentially including authors’ own previously pub-
lished work) are accompanied by appropriate cita-
tions (yes/no/NA) yes
3.6. All datasets drawn from the existing literature
(potentially including authors’ own previously
published work) are publicly available (yes/par-
tial/no/NA) yes
3.7. All datasets that are not publicly available are de-
scribed in detail, with explanation why publicly
available alternatives are not scientifically satisficing
(yes/partial/no/NA) NA
4. Computational Experiments4.1. Does this paper include computational experiments?
(yes/no) yes
If yes, please address the following points:
4.2. This paper states the number and range of values
tried per (hyper-) parameter during development of
the paper, along with the criterion used for selecting
the final parameter setting (yes/partial/no/NA) yes
4.3. Any code required for pre-processing data is in-
cluded in the appendix (yes/partial/no) yes
4.4. All source code required for conducting and analyz-
ing the experiments is included in a code appendix
(yes/partial/no) yes
4.5. All source code required for conducting and ana-
lyzing the experiments will be made publicly avail-
able upon publication of the paper with a license
that allows free usage for research purposes (yes/-
partial/no) yes
4.6. All source code implementing new methods have
comments detailing the implementation, with refer-
ences to the paper where each step comes from (yes/-
partial/no) yes
4.7. If an algorithm depends on randomness, then the
method used for setting seeds is described in a way
sufficient to allow replication of results (yes/par-
tial/no/NA) NA
4.8. This paper specifies the computing infrastructure
used for running experiments (hardware and soft-
ware), including GPU/CPU models; amount of
memory; operating system; names and versions of
relevant software libraries and frameworks (yes/par-
tial/no) no
4.9. This paper formally describes evaluation metrics
used and explains the motivation for choosing these
metrics (yes/partial/no) yes
4.10. This paper states the number of algorithm runs used
to compute each reported result (yes/no) yes
4.11. Analysis of experiments goes beyond single-
dimensional summaries of performance (e.g., aver-
age; median) to include measures of variation, con-
fidence, or other distributional information (yes/no)
yes
4.12. The significance of any improvement or decrease in
performance is judged using appropriate statistical
tests (e.g., Wilcoxon signed-rank) (yes/partial/no)
yes
4.13. This paper lists all final (hyper-)parameters used
for each model/algorithm in the paper’s experiments
(yes/partial/no/NA) partial
# Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding

**Authors**: Xinkui Zhao, Rongkai Liu, Yifan Zhang, Chen Zhi, Lufei Zhang, Guanjie Cheng, Yueshen Xu, Shuiguang Deng, Jianwei Yin

**Published**: 2025-12-04 07:37:59

**PDF URL**: [https://arxiv.org/pdf/2512.04538v1](https://arxiv.org/pdf/2512.04538v1)

## Abstract
As code completion task from function-level to repository-level, leveraging contextual information from large-scale codebases becomes a core challenge. However, existing retrieval-augmented generation (RAG) methods typically treat code as plain natural language, relying primarily on shallow semantic matching while overlooking structural semantics and code-specific dependencies. This limits their ability to capture control flow and underlying intent, ultimately constraining the quality of generated code. Therefore, we propose CoCo, a novel framework that enables code Completion by Comprehension of multi-granularity context from large-scale code repositories. CoCo employs static code analysis to extract structured context at the function, file, and project levels, capturing execution logic and semantic dependencies. It then adopts an graph-based multi-granularity context selection mechanism to filter out redundant information and remove noise. Consequently, the information is converted into natural language in a consistent manner, thereby functioning as explicit contextual prompts to guide subsequent code completion. Additionally, a structure-aware code re-ranker mechanism ensures alignment at both semantic and structural levels. Extensive experiments on CrossCodeEval and RepoEval benchmarks demonstrate that CoCo consistently surpasses state-of-the-art baselines, achieving up to 20.2% gains in EM. Moreover, the framework is model-agnostic and can be seamlessly integrated into existing methods, leading to significant performance.

## Full Text


<!-- PDF content starts -->

Completion by Comprehension: Guiding Code Generation with
Multi-Granularity Understanding
XINKUI ZHAO∗,Zhejiang University, China
RONGKAI LIU∗,Zhejiang University, China
YIFAN ZHANG†,Zhejiang University, China
CHEN ZHI,Zhejiang University, China
LUFEI ZHANG,State Key Laboratory of Mathematical Engineering and Advanced Computing, China
GUANJIE CHENG,Zhejiang University, China
YUESHEN XU,Xidian University, China
SHUIGUANG DENG,Zhejiang University, China
JIANWEI YIN,Zhejiang University, China
As code completion task from function-level to repository-level, leveraging contextual information from large-scale codebases becomes
a core challenge. However, existing retrieval-augmented generation (RAG) methods typically treat code as plain natural language,
relying primarily on shallow semantic matching while overlooking structural semantics and code-specific dependencies. This limits
their ability to capture control flow and underlying intent, ultimately constraining the quality of generated code. Therefore, we
proposeCoCo, a novel framework that enables codeCompletion byComprehension of multi-granularity context from large-scale
code repositories. CoCo employs static code analysis to extract structured context at the function, file, and project levels, capturing
execution logic and semantic dependencies. It then adopts an graph-based multi-granularity context selection mechanism to filter
out redundant information and remove noise. Consequently, the information is converted into natural language in a consistent
manner, thereby functioning as explicit contextual prompts to guide subsequent code completion. Additionally, a structure-aware
code re-ranker mechanism ensures alignment at both semantic and structural levels. Extensive experiments on CrossCodeEval and
RepoEval benchmarks demonstrate that CoCo consistently surpasses state-of-the-art baselines, achieving up to 20.2% gains in EM.
Moreover, the framework is model-agnostic and can be seamlessly integrated into existing methods, leading to significant performance.
Additional Key Words and Phrases: Repository-Level Code Completion, Large language model, Code comprehension
1 Introduction
Automatic code completion plays a fundamental role in modern software development [ 17,30,35,36], significantly
boosting developer productivity and code quality [ 29,46,55]. Traditionally, code completion approaches have focused on
generating code by leveraging static analysis [ 3,40], rule-based heuristics [ 49], or statistical language modeling [ 2,15,44].
While these methods perform well for simple scenarios—such as completing statements or standalone functions—they
fall short in real-world settings where code is organized in complex repositories with abundant cross-file dependencies,
customized APIs, and project-specific conventions [ 16,34]. The advent of large-scale pre-trained language models
∗Both authors contributed equally to this research.
†Corresponding author.
Authors’ Contact Information: Xinkui Zhao, zhaoxinkui@zju.edu.cn, Zhejiang University, Hangzhou, China; Rongkai Liu, 15274800780@163.com,
Zhejiang University, Hangzhou, China; Yifan Zhang, 12451018@zju.edu.cn, Zhejiang University, Hangzhou, China; Chen Zhi, zjuzhichen@zju.edu.cn,
Zhejiang University, Hangzhou, China; Lufei Zhang, zhanglf04@126.com, State Key Laboratory of Mathematical Engineering and Advanced Computing,
Zhengzhou, China; Guanjie Cheng, chengguanjie@zju.edu.cn, Zhejiang University, Hangzhou, China; Yueshen Xu, ysxu@xidian.edu.cn, Xidian University,
Xian, China; Shuiguang Deng, dengsg@zju.edu.cn, Zhejiang University, Hangzhou, China; Jianwei Yin, zjuyjw@cs.zju.edu.cn, Zhejiang University,
Hangzhou, China.
Manuscript submitted to ACM 1arXiv:2512.04538v1  [cs.SE]  4 Dec 2025

2 Xinkui Zhao et al.
for code (Code LLMs), such as GPT-4 [ 1], DeepSeekCoder [ 14,54], Qwen [ 18,48] and Yi [ 50], has greatly advanced
the capabilities of automated code generation. However, these models are inherently limited by their input context
window and cannot directly access the full scope of a repository, where critical information is often distributed across
multiple files [ 45]. This limitation becomes particularly challenging inrepository-level code completion, where accurate
code generation requires a holistic understanding of repository-wide dependencies, shared utilities, and inter-module
interactions. To address this challenge, theretrieval-augmented generation(RAG) paradigm has emerged as a promising
and widely adopted approach for the repository-level code completion task [ 27,51]. Typically, RAG methods operate
by first retrieving potentially relevant code snippets from a large codebase, and then concatenating these retrieved
fragments with the original prompt before feeding the combined context into a code generation model. This mechanism
enables the model to leverage external knowledge beyond the immediate file, thereby mitigating the limitations of
input length and local context.
While RAG-based approaches enhance code generation by retrieving semantically similar code examples, they still
face notable limitations. First, most existing retrievers—whether lexical-based or model-based—treat code purely as text
sequences and compute similarity based on surface-level token overlap, ignoring the inherent structural and semantic
characteristics of source code. As a result, retrieved examples often exhibit superficial lexical overlap, such as shared
variable names or common keywords, with the query, yet fail to preserve deeper structural correspondences. This
structural mismatch introduces irrelevant or even misleading code fragments, which may divert the generation model
from the intended logic, resulting in syntactically correct but semantically incorrect completions. Second, retrieval
results rarely capture fine-grained contextual dependencies within the same file or project-wide information scattered
across multiple modules. However, such information are often crucial for accurately completing complex code segments.
The absence of this information during generation inevitably limits the model’s performance in real-world scenarios.
To address these problems, we proposeCoCo, a novel framework that enables large language models to perform
repository-level code completion bycomprehension before completion. Inspired by the way human developers first
understand and analyze code context before writing new code, CoCo constructs a multi-granularity representation of
the unfinished code to guide LLM-based generation. Specifically, CoCo first applies static code analysis tools (such
as AST parsers) to extract rich contextual information at thefunction,file, andrepositorylevels. At the function level,
we analyze execution logic and control flow to capture the local semantics surrounding the unfinished code. At the
file level, we identify both explicit and potential dependencies between the unfinished code and other entities in the
same file. At the repository level, we model cross-file interactions and external module relationships. Through these
operations, we obtain a large volume of information related to the unfinished code. However, such information may be
excessive, with some being unhelpful or even detrimental to guiding subsequent LLM-based code generation—there
may also be redundancy or conflicts. To address this, CoCo incorporates an graph-based multi-granularity context
selection mechanism to sift through and retain only the most relevant information. The filtered information is then
transformed into concise and coherent natural language descriptions, making the underlying structural and semantic
context accessible to LLMs. Furthermore, CoCo introduces a structure-aware code re-ranker mechanism that refines
retrieved code examples, ensuring both semantic relevance and structural consistency with the completion target.
Finally, CoCo synthesizes the multi-granularity contextual information and the retrieved code examples into a unified
prompt, which is fed into the LLM for code completion. Unlike conventional RAG-based methods that focus only on
local code fragments, CoCo follows a “Comprehension first, then Completion” paradigm. This enables the LLM to
reason globally and locally about cross-file dependencies, shared utilities, and project-specific conventions, resulting
in more accurate and contextually consistent repository-level code completion. Furthermore, our framework can be
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 3
flexibly integrated with existing methods, further improving their performance by supplying high-level, structured
context.
Our main contrubution are:
•We propose CoCo, a novel comprehension-driven code completion framework.CoCo enables large
language models to perform repository-level code completion by deeply comprehending the context of unfinished
code, rather than relying solely on fragment-level retrieval.
•We design a multi-granularity analysis pipelinethat gathers and filters function-, file-, and repository-level
contextual information and transforms the resulting insights into LLM-friendly natural-language prompts.
•We introduce a structure-aware code re-ranker mechanismthat considers both semantic similarity and
structural consistency, ensuring that retrieved code exemplars are highly relevant and contextually aligned with
the unfinished code.
•We demonstrate the effectiveness and generalizability of CoCo through extensive experiments on
both CrossCodeEval and RepoEval.CoCo consistently outperforms state-of-the-art methods and integrates
flexibly with existing frameworks. Our code is publicly available. We make the code publicly available at the link.
2 Background
2.1 Retrieval-Augmented Generation Enhanced Repository Code Completion
RAG [ 12,22] is a general framework that enhances LLMs by leveraging external knowledge retrieved from a large
corpus. A typical RAG workflow consists of three main stages: (1)Indexing: The external corpus is first pre-processed
and encoded into a searchable index, often using dense or sparse vector representations; (2)Retrieval: Given an input
query or prompt, the retriever searches the index to select relevant documents or knowledge pieces; (3)Generation:
The retrieved information is then incorporated into the model’s context and the LLM generates the final output based
on this augmented context. By integrating relevant external knowledge, RAG enables LLMs to overcome the limitations
of their parametric memory and adapt more flexibly to downstream tasks.
In recent years, researchers have conducted a substantial amount of research utilizing RAG for code-related research
[4,7,23,26–28,31,32,51,53]. For repository-level code generation, directly providing the entire codebase as context to
LLMs is infeasible due to the massive size of repositories and the limited context window of LLMs [ 45]. Moreover, code
fragments to be generated often share similarities or semantic connections with existing code within the repository.
And traditional LLMs for code generation are often constrained by the closed world of their training data, making it
difficult to access up-to-date API usages, third-party library information, or relevant private knowledge within code
repositories. As a result, RAG has proven to be an effective solution in this domain. Typically, RAG methods selectively
retrieve relevant code snippets and inject them into the model’s context window. This targeted retrieval enables the
LLM to access both intra-file and cross-file information [ 9,24], thereby enhancing its ability to generate more accurate
and contextually appropriate code completions.
2.2 Limitations
Despite its promise, this approach is subject to several inherent limitations. Conventional RAG methods typically
assume that, for any code completion task, sufficiently similar and relevant code snippets can always be found within
the repository. In practice, however, this assumption frequently fails, giving rise to several fundamental challenges.
Manuscript submitted to ACM

4 Xinkui Zhao et al.
First, traditional retrieval methods, such as lexical-based approaches BM25 [ 20,37], generally treat code as plain
text and struggle to capture the underlying semantics and intent of code snippets. Such a semantic gap can cause the
retriever to select code that appears textually similar but is actually semantically irrelevant, resulting in ineffective or
even misleading augmentation for code generation. For example, as illustrated in Fig. 1, the retriever selects a lexically
similar variable input_id by mimicking applied_input_ids without considering the actual semantic meaning of
these variables.Second, retrieval noise and errors are difficult to eliminate entirely: the retriever may include irrelevant,
redundant, or misleading code snippets, and incorporating such noisy context into the prompt can confuse the generation
model and degrade completion quality.Third, most existing retrieval mechanisms inadequately model the complex
structural and cross-file dependencies present in real-world repositories, such as data flows, control flows, and inter-file
relationships. This structural context gap makes it challenging to provide the generation model with the most relevant
and informative context for accurate code completion. Another factor contributing to the failure in Fig. 1 is the neglect
of the grammatical and structural roles of variables and functions in the context. Variables like input_ids often serve
as consumed intermediate variables: they are used immediately after their definition and are seldom directly referenced
again in the remaining code. Due to the lack of understanding of code control flow and syntactic structure, the retriever
or generation model tends to select such already-consumed variables for completion, rather than prioritizing other
unused-but-defined variables that are more likely to be semantically appropriate in the current context.
# exllama/example_ws.py
...
def begin_stream(prompt: str, stop_conditions: list, max_new_tokens: int, 
gen_settings: ExLlamaGenerator.Settings):
    global model, cache, config, generator, tokenizer
    global stop_strings, stop_tokens, prompt_ids, held_text, max_stop_string, 
remaining_tokens
    global full_prompt, utilized_prompt, built_response
    # Tokenize prompt and limit length to allow prompt and (max) new tokens 
within max sequence length
    max_input_tokens = model.config.max_seq_len - max_new_tokens
    input_ids = cached_tokenize(prompt)
    input_ids = input_ids[:, -max_input_tokens:]
    prompt_ids = input_ids
    full_prompt = prompt
    utilized_prompt = tokenizer.decode(prompt_ids)[0]Query (Unfinished Code)
# exllama/alt_generator.py
Retrieved Chunk 1:
new_enc = self.tokenizer.encode(text, encode_special_characters)
self.tokenizer_cache[text] = new_enc
Retrieved Chunk 2:
self.remaining_tokens = max_new_tokens
input_ids = self.cached_tokenize(prompt, encode_special_characters)
applied_input_ids = input_ids[:, -max_input_tokens:]
if applied_input_ids.shape[0] < input_ids.shape[0]:
    self.sequence_str = self.tokenizer.decode(applied_input_ids)[0]
else:
    self.sequence_str = promptRetrieved Context
    
...
utilized_prompt = tokenizer.decode(input_ids)Generation with Retrieval Semantic Gap
Structural MismatchRetrieved 
Noise
Fig. 1. Limitations of traditional retrieval-based methods can introduce semantic and structural context gaps, leading to the generation
of incorrect code.
3 Methodology
3.1 Overview
In this section, we introduce CoCo, a framework for repository-level code completion that captures multi-granularity
context to guide LLMs in generating code that is aligned with both semantic intent and structural constraints. As
illustrated in Fig. 2, CoCo is composed of three modules: code comprehension, code retrieval, and code generation. In the
code comprehension stage, we use static analysis tools to extract background information relevant to the unfinished code,
including execution logic, in-file dependencies, and cross-file module references. To ensure the resulting information
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 5
is not overly redundant and does not negatively impact subsequent generation tasks, we employ a graph-based
multi-granularity context selection mechanism to sift through the data, retaining only the most critical and relevant
information. This information is then unified and structured into a context representation that improves the LLM’s
comprehension of the intended logic flow and developmental intent. In the code retrieval stage, we retrieve similar code
snippets from the code repository that are semantically similar to the unfinished code. To ensure that the retrieved
candidates are not only semantically relevant but also structurally consistent, we further apply a structure-aware code
re-rank mechanism to refine the results. Finally, in the code generation stage, the contextual information collected is
integrated into a single prompt. This prompt is then fed into the LLM to generate the continuation code that complies
with the given context.
Code Repository
Unfinished CodeCode ParserFunction-Level
File-Level
Project-Level
Retriever Candidate Codes
Structural-Aware Code Re-Ranker
Code ExamplePrompt
LLM
Generated CodeCode Comprehension  
Code Retrieval
Code Completion
Graph-based Multi-granularity
 Context Selection
Fig. 2. Oveiview of CoCo.
3.2 Code comprehension
In the task of repository-level code completion, the code repository itself contains rich structural and semantic
information that can inform the generation of unfinished code. To fully exploit such information, we employ static
analysis tools to construct multi-granularity contextual representations surrounding the unfinished code, enabling the
model to reason about potential control flows and developer intent, rather than relying solely on surface-level similar
code examples. Specifically, we utilize Tree-sitter and the abstract syntax tree (AST) module to dynamically zoom out
from the function level, progressively expanding the context to encompass the file and project levels. This process forms
a multi-granularity representation grounded in execution logic, semantic dependencies, and inter-module relations.
3.2.1 Function-Level Analysis.Within a function, the surrounding code typically provides the most relevant context
for the missing line, containing the majority of information needed for accurate code completion. This is because local
variables, control flows, and intermediate computations within the same function directly influence the semantics of the
target code. To effectively capture these local execution semantics, we introduce a function-level analysis mechanism
that constructs a control-flow-aware context preceding the target line. This enables the large language model to better
understand the semantic structure and reason about the logical pattern of the unfinished code. We begin by constructing
the AST of the source file and recursively traversing its nodes to locate the target function of the target line. Specifically,
Manuscript submitted to ACM

6 Xinkui Zhao et al.
    expect_tokens = in_tokens.shape[-1] + 
max_response_tokens
    max_tokens = config.max_seq_len - 
expect_tokens
    if generator.gen_num_tokens() >= max_tokens:
        generator.Location Identification# webui/session.py
from model import ExLlama, ExLlamaCache, 
ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
...
While True:
...
    # little extra so we don't end up 
rebuilding the cache on every line when up 
against the limit.
    expect_tokens = in_tokens.shape[-1] + 
max_response_tokens
    max_tokens = config.max_seq_len - 
expect_tokens
    if generator.gen_num_tokens() >= 
max_tokens:
        generator.gen_prune(max_tokens - 
extra_prune)CFG Construction
# Target Function and Its Analysis:
Assign: expect_tokens = in_tokens.shape[-1] + 
max_response_tokens
Assign: max_tokens = config.max_seq_len - 
expect_tokens
Condition Check: generator.gen_num_tokens() >= 
max_tokens
If True:
Incomplete Action: Call Function: 
generator.method_name
Fig. 3. Illustration of Function-Level Analysis. The yellow code block indicates unfinished code.
thefunction_definition node is identified as the target function if its start and end line numbers encompass the line
where the code is missing. Once the target function is located, we extract the code segment ranging from the beginning
of the function up to the target line as the local context for code generation. If the target line does not reside within
any function body—i.e., it appears at the top-level script scope—we instead extract the code before the target line as
local context. As illustrated in Fig. 3, after obtaining the local code context, we construct a control flow graph (CFG) to
model the control dependencies among statements. During this process, we identify key control structures within the
AST, such as conditionals ( if), loops ( for, while ), and termination statements ( return ). These statements are treated
as nodes in the graph. Directed edges are then added between nodes based on the program’s execution semantics to
reflect control flow transitions. This CFG-based structural representation makes the reachable paths prior to the target
line explicit, thereby providing the language model with a clearer execution context to guide code generation.
# mCLS/train-CLS.py
import logging
import torch
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
logger = logging.getLogger(__name__)
def string_compare(s,s1,s2):
...
def load_examples():
...
def evaluate():
dataset = load_examples(args...)
logger.warning("***** Running 
evaluation {} *****".format(prefix))
...
pred = string_compare(s,s1,s2)Location Identification
evaluate
├── load_examples
└── logger
evaluate
├──IGNORE_INDEX
└──string_compareInformation Extraction
Explicit Dependency Potential Dependency
*Defined and Used entities
load_examples,logger
*Defined but Unused entities：
string_compare,IGNORE_INDEX
Fig. 4. Illustration of File-Level Analysis. The yellow code block indicates unfinished code. The green code block indicates useful
information.
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 7
3.2.2 File-Level Analysis.To enhance the semantic coverage of the context, we also introduce a file-level analysis
mechanism to model the dependencies between the target code and other entities defined within the same source
file. Specifically, we first parse the abstract syntax tree of the source file and traverse upward from the target line to
collect all functions and variables defined before the target line, thereby constructing a static symbol definition set.
We then extract all identifiers used in the target function to form a symbol usage set. By computing the intersection
between the definition and usage sets, we extract the explicit dependency information (while filtering out function-
internal self-references such as local variables and recursive calls), which precisely captures the target function’s actual
reliance on other entities within the file. This information includes the dependent symbols’ names, types, locations, and
corresponding source code snippets. When the unfinished code is located outside any function body (i.e., in a script-level
context), we skip dependency matching and extract only symbols. In addition to explicit dependencies, we also identify
potential dependency information—symbols that are defined in the current file but have not yet been used prior to
the target line. This is accomplished by computing the set difference between the sets of defined and used symbols, a
process applied regardless of whether the missing code appears within a function or at the top level. Although these
symbols have not been referenced so far, they are often intended for use in subsequent code, reflecting the programmer’s
design intent. As illustrated by the green-highlighted block in the lower-right corner of Fig. 4, string_compare remains
unused in the existing code but is invoked in the incomplete statement, demonstrating the importance of potential
dependency information in facilitating accurate code completion.
# CoCo/src/datasets.py
import pandas as pd
import os
from parser_utils import parser_code
from data import process_data
def init_parser_py()：
...
def load_test_dataset(a):
path = os.path.join(a,'x/x')
dataframe = pd.read_parquet(path)
data = process_data(dataframe)
parser = init_parser_py()
res = parser_code(data, parser)
return res
if __name__ == "__main__":
print(load_test_dataset())Location Identification
Import resolution
Information Extraction
load_test_dataset
├── os
├── pd
└── process_data
load_test_dataset
└── parser_codeExplicit Dependency Potential Dependency
*Third-party/Standard libraries：
pandas，os
*Cross-file  definitions：
process_data，parser_code
Fig. 5. Illustration of Project-Level Analysis. The yellow code block indicates unfinished code. The green code block indicates useful
information.
3.2.3 Project-Level Analysis.Real-world code repositories frequently exhibit complex interactions across multiple
modules and external libraries. Accurately modeling these cross-file and cross-library dependencies is critical for
providing comprehensive context and enabling the generation model to produce semantically correct and contextually
appropriate code completions. To capture dependencies between the target code and external modules or libraries, we
introduce a project-level code analysis mechanism that enhances contextual completeness. This mechanism focuses on
two categories of critical information. The first is explicit cross-module dependencies, which refer to entities that are
both imported and actively referenced within the target function—either from other source files in the project or from
Manuscript submitted to ACM

8 Xinkui Zhao et al.
external libraries. The second is potential cross-module dependencies, which denote symbols that have been imported
into the current file but have not yet been used before the target line; these may be invoked in subsequent code and
thus are highly relevant for code completion.
We begin by parsing the AST of the target file to extract all import statements, categorizing them into cross-file
definitions (i.e., symbols defined in other source files within the same repository) and third-party or standard library
imports. To distinguish between these, we recursively scan all Python files in the project root to build a mapping from
module names to their corresponding physical file locations. If an imported module matches an entry in this mapping
(via full-name or suffix matching), it is classified as a cross-file definition; otherwise, it is treated as a third-party or
standard library. For cross-file definitions, we further parse the corresponding source files to extract the imported
function or class definitions, recording their names, types, source code locations, and full implementations.
Next, we extract the set of identifiers used within the target function and match them against the previously collected
imported definitions. All matched entities constitute the set of explicit cross-module dependencies, which may include
both cross-file symbols and external libraries. Dependencies are organized by their source: for cross-file definitions, we
record their source locations and code content in a cross-file dependency dictionary; for third-party or standard libraries,
only the module and symbol names are retained. As illustrated in Fig. 5, the collected dependencies include os(a
standard library), pd(a third-party library) and process_data (a cross-file definition), together forming the complete set
of explicit cross-module dependencies. If the unfinished code is not located within a function body, identifier matching
is omitted and only definitions are extracted.
Building on this, we further identify potential cross-module dependencies—imported symbols that have not yet been
used before the target line but are likely to appear in subsequent code. Such symbols often reflect the programmer’s
latent intent and are valuable for guiding the model in completing the unfinished code. Specifically, we compute the
set difference between all imported entities and those symbols already used (regardless of whether they occur within
a function body). The resulting items are treated as potential dependencies, for each of which we record the import
statement’s source code location, dependency type, and, if applicable, the symbol’s definition location and code snippet.
As shown in the green highlighted area in the lower-right of Fig. 5, although parse_code is not used before the target
line, it is invoked in the unfinished statement, demonstrating the importance of potential cross-module dependency
information in facilitating accurate code completion.
3.2.4 Graph-based Multi-granularity Context Selection.Although multi-granularity analysis provides a comprehensive
view of the unfinished code, directly feeding all extracted information into a large language model is impractical.
File-level and project-level analyses often generate substantial amounts of data. Incorporating all such information
inevitably exceeds the model’s token budget, while naïve truncation risks removing semantically crucial elements,
thereby damaging contextual integrity and resulting in generated code that no longer aligns with the surrounding
program structure. Meanwhile, the raw context frequently contains redundant or irrelevant elements. These noisy
components not only fail to support the generation process but may also mislead the model by diluting the semantic
signals associated with the unfinished code. To address these challenges, we introduce a graph-based context selection
mechanism that systematically models the semantic relations among multi-granularity elements and quantifies their
relative importance. The detailed algorithmic procedure is shown in Algorithm 1.
We first construct a heterogeneous semantic graph from the parsed multi-granularity context. The target function
containing the unfinished code is designated as the central node, while additional nodes represent symbols, type
definitions, calls, import statements, and cross-file entities extracted from file- and project-level analyses. Edges are
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 9
constructed strictly according to semantic relations present in the source code, such as function invocations, variable
usages, and inheritance relations, ensuring that the graph accurately reflects the program’s underlying structure. We
add directed edges from the central node to all other nodes so that the ranking process is explicitly centered on the
function under completion and importance propagation naturally originates from it.
We compute importance scores using Personalized PageRank, which differs from classical PageRank by allowing the
random walk to restart preferentially from a designated subset of nodes. Given a graph 𝐺=(𝑉,𝐸) , the importance
score for node𝑣 𝑖is defined as:
𝑆𝑐𝑜𝑟𝑒(𝑣𝑖)=𝛼∑︁
𝑣𝑗∈𝐼𝑛(𝑖)𝑆𝑐𝑜𝑟𝑒(𝑣𝑗)
𝑑𝑒𝑔(𝑣𝑗)+(1−𝛼)𝑝(𝑣 𝑖)(1)
where𝛼is the teleportation factor, 𝐼𝑛(𝑖) denotes the set of nodes with edges pointing to 𝑣𝑖,𝑆𝑐𝑜𝑟𝑒(𝑣𝑗)is the PageRank
score of node𝑣 𝑗in the previous iteration,𝑑𝑒𝑔(𝑣 𝑗)is its out-degree, and𝑝(·)is the personalization distribution.
In our setting, 𝑝(𝑣) assigns probability 1 to the central node and 0 to all other nodes, meaning that random-walk
restarts always return to the target function rather than choosing uniformly at random. This ensures that importance
diffuses outward from the missing-code function and decays with structural distance. After convergence, we rank
file-level and project-level nodes separately and select the top-k nodes from each level as the distilled context for code
generation.
Algorithm 1Graph-based Multi-granularity Context Selection and PPR
Require:File-level, Project-level parsed context and target function𝑓
Ensure:Distilled context (top-𝑘file/project-level nodes)
1:Construct heterogeneous semantic graph𝐺=(𝑉,𝐸)from parsed context
2:Compute node importance scores using Personalized PageRank on𝐺:
3:For all𝑣 𝑖∈𝑉, iteratively update until convergence:
4:Score(𝑣 𝑖)=𝛼Í
𝑣𝑗∈In(𝑖)Score(𝑣 𝑗)
deg(𝑣 𝑗)+(1−𝛼)𝑝(𝑣 𝑖)
5:𝑝(𝑣)= 
1if𝑣=𝑣 central
0otherwise
6:Split nodes into file-level set𝑉 fileand project-level set𝑉 proj
7:Sort both sets by their importance scores in descending order
8:Select top-𝑘nodes from each sorted set to form the distilled context
9:returndistilled context
3.3 Code Retrieval
In the code retrieval stage, we follow the common RAG paradigm to retrieve semantically similar code snippets from
the repository as examples to guide subsequent code generation. However, existing methods typically treat code as
plain text, relying solely on embedding-based semantic similarity, which may result in the retrieval of structurally
mismatched examples and thereby impair generation quality. To address this limitation, we introduce a Structure-Aware
Code Re-Ranker that enhances the structural consistency of retrieved exemplars. Specifically, after obtaining an initial
set of candidate snippets via RAG, we extract AST path representations from both the query and each candidate, and
Manuscript submitted to ACM

10 Xinkui Zhao et al.
compute a structure score based on the Jaccard similarity. The calculation formula is as follows.
𝑠𝑡𝑟𝑢𝑐𝑡𝑢𝑟𝑒_𝑠𝑐𝑜𝑟𝑒=𝑃𝑞𝑢𝑒𝑟𝑦∩𝑃𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒
𝑃𝑞𝑢𝑒𝑟𝑦∪𝑃𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒(2)
where𝑃𝑞𝑢𝑒𝑟𝑦 and𝑃𝑐𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒 denote the sets of AST paths extracted from the query and candidate snippets, respectively.
The final ranking score is computed as a weighted combination of the semantic and structural scores. Based on the final
score, we re-rank the candidate snippets and select the topk results as retrieval examples.
Target Function and Its Analysis:
Control Flow Graph:
...
File-Level Contexts — symbols defined in the same file as the target code:
file_info 1:
file_info2:
...
Project-Level Contexts— related entities from other files or modules:
pro_info 1:
pro_info2:
...
# Below are some similar code examples 
from the code repository 
code example 1:
code example 2:
...
code example n:
# Inference endpoint with "precise" 
preset settings
@app.route('/infer_precise', methods=['POST'])
def inferContextP():
    print(request.form)
    prompt = request.form.get('prompt')Multi-Granularity 
Contextual 
Information
Retrieved 
Code
Unfinished 
Code
Fig. 6. Prompt template provided to the code generation model, consisting of (1) multi-granularity contextual information that
captures control flow, file-level, and project-level code relationships; (2) retrieved similar code examples from the repository; and (3)
the unfinished code segment to be completed. * represents the optional context.
3.4 Code Generation
In this stage, we construct a unified prompt by integrating multi-granularity contextual information and similar code
examples, which serves as the direct input to the LLM for code generation. As shown in Fig. 6, the prompt enhances
existing prompting strategies by incorporating function-level, file-level, and project-level contextual information. It is
structured into three components: multi-granularity contextual infromation, similar code examples, and the target code
to be completed.
4 Experimental Setup
In this section, we will introduce the experimental environment, the evaluation metrics adopted, the compared methods,
and the analysis of the experimental results. And we aim to answer the following research questions through our
experimentation.
•RQ1.How effective is CoCo in repository-level code completion?
•RQ2.Does each component of CoCo contribute to its performance?
•RQ3.How well does CoCo generalize as a plug-in to other methods in terms of performance?
•RQ4.What is the process latency of integrating CoCo, and is it acceptable relative to its performance gains?
•RQ5.How do different levels of contextual granularity affect code completion performance?
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 11
4.1 Benchmark
To verify the effectiveness of CoCo, we select two widely used benchmarks for comparative experiments: CrossCodeE-
val [8] and RepoEval [ 52]. CrossCodeEval is a multilingual repository-level code generation benchmark, where the
generation of each sample need rely on cross-file information. It covers four programming languages: Python, Java,
TypeScript, and C#. We use the Python and Java subsets for the experiments. RepoEval is a multigranular repository-
level code generation benchmark, containing subsets at three levels: line, api, and function, which covers scenarios of
real-world repositories. We use the line-level and api-level subsets for the experiments, which consist of 1600 samples.
4.2 Baselines
To verify the effectiveness of CoCo, we select some commonly used and state-of-the-art (sota) methods to compare,
including RawRAG, RepoCoder, and RLcoder.
•RawRAG.It is a standard repository-level code generation method, consisting of two parts: similar code retrieval
and code generation. The process is as follows: retrieve code snippets similar to the code to be generated through
the retriever, then concatenate these snippets with the code to be generated into a prompt, and input it into a
LLM for code completion.
•RepoCoder.As a commonly used repository-level code generation method, it shares the same overall workflow
as RawRAG. However, it has improved the code retrieval process by adopting an iterative retrieval approach to
enhance the quality of similar code.
•RLCoder.It is a sota method in the field of repository-level code generation, which also focuses on optimizing
the code retrieval process. It trains the retriever using a reinforcement learning mechanism and eliminates
retrieval candidates that may have negative impacts through a stop signal mechanism.
4.3 Metrics
We use four widely adopted evaluation metrics to compare and analyze the experimental results: Exact Match (EM),
Edit Similarity (ES), Identifier Exact Match (ID.EM) and F1-score.
•EM measures the proportion of generated code that exactly matches the ground-truth code, quantifying overall
correctness, which calculation process is as follows.
𝐸𝑀=𝐽𝑢𝑑𝑔𝑒(𝑦 𝑖,𝑦∗
𝑖)(3)
where𝑦and𝑦∗represent the generated code and ground-truth code respectively, 𝐽𝑢𝑑𝑔𝑒() refers a function that
determines whether two codes are equal.
•ES evaluates the similarity between generated and ground-truth code, assessing statement matching granularity,
which calculation process is as follows.
𝐸𝑆=1−𝐿𝑒𝑣(𝑦𝑖,𝑦∗
𝑖)
𝑚𝑎𝑥(𝑙𝑒𝑛(𝑦 𝑖),𝑙𝑒𝑛(𝑦∗
𝑖))(4)
where𝐿𝑒𝑣() denotes the Levenshtein distance, which quantifies the edit distance between two codes, and 𝑙𝑒𝑛()
represents a function that calculates the string length.
•ID.EM measures the proportion of generated code identifiers that match the ground-truth code. Its calculation
process is similar to EM, except that the object is replaced by the set of identifiers.
Manuscript submitted to ACM

12 Xinkui Zhao et al.
•F1 comprehensively evaluates the model’s performance in predicting core identifiers in code through the harmonic
mean of precision (the proportion of correctly predicted identifiers among all predicted ones) and recall (the
proportion of truly existing identifiers that are correctly predicted).
4.4 Experiment Settings
All experiments are conducted on a machine with one Tesla A100 GPU, with 40 GB memory. In the experments, we
select three different LLMs as code generation backbone, including DeepSeekCoder-1B [ 14], Codellama-7B [ 38],Yi-
Coder-1.5B [ 50]. The selection of these models is motivated by our goal to comprehensively evaluate the effectiveness of
our framework across different model sizes and architectures. Notably, we place particular emphasis on the performance
of smaller-scale models, as their efficiency and deployment-friendliness are of great practical significance for real-world
applications and resource-constrained scenarios. Following the setup of other methods, the random seed is set to 123
and the max number of generated tokens is limited to 64 for all models. And we use the retriever of RLCoder [ 45] as
our retriever. To ensure fair and consistent evaluation, we directly reuse the publicly available implementations of
evaluation metrics from previous studies [8, 52], rather than reimplementing them.
Table 1. Performance comparison of different models and baselines. CL-7B denotes CodeLlama-7B, Yi-1.5B denotes Yi-Coder-1.5B,
and DSC-1B denotes DeepSeekCoder-1B.
MethodCrossCodeEval (Python) CrossCodeEval (Java) RepoEval (Line) RepoEval (API)
EM ES ID.EM F1 EM ES ID.EM F1 EM ES ID.EM F1 EM ES ID.EM F1
RawRAGCL-7B20.56 67.34 29.86 59.01 20.10 63.76 27.58 55.99 42.87 64.68 27.37 47.40 34.68 62.06 28.50 60.89
RepoCoderCL-7B24.24 69.73 34.26 61.11 22.21 63.83 30.48 56.79 41.88 64.28 26.75 47.29 37.25 63.57 30.06 62.34
RLCoderCL-7B27.17 71.68 38.19 64.44 23.79 64.81 32.58 58.12 48.00 69.01 29.12 49.87 37.68 64.06 30.62 62.56
CoCoCL-7B 31.23 74.87 41.95 67.89 26.21 66.47 35.17 62.29 49.93 69.88 30.71 50.75 41.21 67.37 33.14 65.93
RawRAGYi-1.5B18.20 66.72 27.39 55.93 15.52 63.14 24.03 54.49 38.75 61.26 24.94 45.02 32.50 60.10 27.06 60.19
RepoCoderYi-1.5B21.73 69.01 31.63 59.64 16.92 63.98 25.62 55.58 41.50 63.35 26.38 46.58 35.00 62.15 28.25 61.43
RLCoderYi-1.5B24.91 70.42 35.94 63.32 19.21 64.97 27.91 57.13 42.87 64.92 26.87 47.47 35.87 62.42 29.31 61.38
CoCoYi-1.5B 29.93 73.44 41.12 67.87 22.78 66.53 30.04 59.61 47.74 67.72 29.03 49.15 38.92 64.54 32.04 64.26
RawRAGDSC-1B16.36 65.73 24.58 53.93 15.66 58.68 22.58 50.13 37.31 59.43 23.75 43.29 32.00 58.30 26.25 57.43
RepoCoderDSC-1B19.55 67.74 28.29 56.97 18.23 60.15 25.48 52.21 39.44 61.55 25.38 45.07 34.31 60.13 28.63 59.41
RLCoderDSC-1B22.55 69.47 32.61 60.71 20.24 62.09 28.61 54.73 40.69 62.99 25.25 45.65 34.13 60.91 28.50 59.64
CoCoDSC-1B 26.53 71.97 36.92 64.87 23.52 65.37 32.30 58.62 44.27 65.91 27.28 47.36 37.73 63.84 30.72 62.69
5 Evaluation
5.1 RQ1: Effectiveness of Coco
Table 1 presents a comparison of the overall performance of CoCo and baselines across different LLMs and benchmarks
in the code completion task. It is evident that CoCo consistently outperforms all baselines on EM, ES, ID.EM, and
F1 across various backbone LLMs and benchmarks, highlighting its strong cross-model adaptability and universal
effectiveness. For example, with Yi-Coder-1.5B, CoCo improves the four metrics by 20.2%, 4.3%, 14,4%, and 7.2% on
CrossCodeEval (Python), and achieves similar gains on CrossCodeEval (Java), RepoEval (Line) and RepoEval (API),
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 13
demonstrating strong robustness across diverse code distributions. These results indicate that, beyond retrieving similar
examples, incorporating multi-granularity contextual information enables the LLM to better understand the code and
generate more accurate completions.
Answer to RQ1:CoCo consistently outperforms state-of-the-art methods across all backbone LLMs. On Cross-
CodeEval (Python), it achieves improvements of 20.2% in EM, 4.3% in ES, 14.4% in ID.EM, and 7.2% in F1, demon-
strating substantial gains across all key metrics.
Fig. 7. Performance comparison of ablation study.
5.2 RQ2: Impact of each component.
To assess the individual contributions of each component within CoCo framework, we conduct two ablation studies.
We first consider CoCo_w/o_cc , a variant that removes the Code Comprehension mechanism and relies solely on
semantically retrieved examples for generation. The second variant, CoCo_w/o_sm , disables the Structural-Aware Code
Re-Ranker, so no structure-level re-ranking is performed on the retrieved candidates. All experiments are conducted on
the CrossCodeEval (Python) benchmark using DeepSeekCoder-1B as the backbone model.
As shown in Fig. 7, CoCo_w/o_cc consistently outperforms RLCoder across all metrics, and CoCo further surpasses
CoCo_w/o_sm, demonstrating the effectiveness of the Structural-Aware Code Re-Ranker. While the performance
gain from re-ranking is relatively modest, it still contributes positively to the final results. This suggests that filtering
semantically retrieved code examples based on structural similarity remains beneficial. In our current design, we
employ Jaccard similarity to quantify structure alignment, which may be overly restrictive. Future work could explore
more flexible or semantically-aware structural similarity measures to improve effectiveness. Moreover, the strong
Manuscript submitted to ACM

14 Xinkui Zhao et al.
performance of CoCo_w/o_sm over RLCoder further confirms the importance of the Code Comprehension mechanism.
The finding suggests that incorporating multi-granular context to enhance the model’s understanding of the target
code plays a critical role in improving repository-level code generation.
Answer to RQ2:Each component contributes positively to repository-level code completion, with code compre-
hension having the greater impact.
5.3 RQ3: Generalizability of CoCo
While many existing repository-level code completion methods focus on enhancing the retriever, our proposed method,
CoCo, takes a different perspective: instead of modifying the retriever, it introduces structured, multi-granularity
contextual information to equip the large language model with a deeper comprehension of the unfinished code before
generation, thereby improving the overall accuracy and consistency of the output. This design enables CoCo to
function as a plug-and-play enhancement module that can be seamlessly integrated into existing methods without
interfering with their retrieval pipelines. To validate the generalizability of our method, we incorporate CoCo into three
representative methods—RawRAG, RepoCoder, and RLCoder—using DeepSeekCoder-1B as the backbone, and evaluate
their performance on two standard benchmarks: CrossCodeEval and RepoEval.
Table 2. Experimental results for various baselines when combined with CoCo.
MethodCrossCodeEval (Python) CrossCodeEval (Java) RepoEval (Line) RepoEval (API)
EM ES ID.EM F1 EM ES ID.EM F1 EM ES ID.EM F1 EM ES ID.EM F1
RawRAG16.36 65.73 24.58 53.93 15.66 58.68 22.58 50.13 37.31 59.43 23.75 43.29 32.00 58.30 26.25 57.43
RawRAG+CoCo 22.94 69.08 32.61 61.47 19.74 60.93 28.22 53.99 44.16 65.14 27.23 47.46 36.44 62.96 30.48 62.59
RepoCoder19.55 67.74 28.29 56.97 18.23 60.15 25.48 52.51 39.44 61.55 25.38 45.07 34.31 60.13 28.63 59.41
RepoCoder+CoCo 23.86 70.37 33.95 62.21 21.51 63.12 30.13 56.38 43.47 64.85 26.11 46.82 36.33 62.99 29.92 61.88
RLCoder22.55 69.47 32.61 60.71 20.24 62.09 28.61 54.73 40.69 62.99 25.25 45.65 34.13 60.91 28.50 59.64
CoCo 26.53 71.97 36.92 64.87 23.52 65.37 32.30 58.62 44.27 65.91 27.28 47.36 37.73 63.84 30.72 62.69
As shown in Table 2, integrating CoCo into different baseline methods consistently results in significant performance
gains across all benchmarks and evaluation metrics. This demonstrates the general applicability of CoCo and its
ability to enhance code generation quality regardless of the underlying generation backbone. Although RepoCoder and
RLCoder initially outperform RawRAG, the performance gap narrows substantially once CoCo is integrated. Notably,
in some cases, RawRAG+CoConot only closes the performance gap but even surpasses other methods, achieving the
best overall results. For example, on RepoEval (Line) benchmark, RawRAG+CoCoattains the highest F1 score among all
evaluated methods. These results suggest that multi-granularity contextual information contributes more substantially
to generation quality than retrieving semantically similar examples alone, reinforcing the necessity of thoroughly
modeling the underlying intent and contextual cues of the target code in repository-level generation tasks.
Answer to RQ3:CoCo demonstrates strong generalizability when integrated into existing code generation
methods, consistently enhancing performance across diverse settings. This highlights the importance of enriching
the model’s comprehension of the target code, rather than relying solely on retrieved examples for guidance.
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 15
5.4 RQ4: Time Efficiency of CoCo
Although the previous experimental results have demonstrated the significant effectiveness of CoCo in improving
code generation performance, its integration as a plug-in module inevitably introduces some computational overhead.
To evaluate its impact on inference efficiency, we conducted comparative experiments based on RLCoder on the
CrossCodeEval and RepoEval benchmarks.
As shown in Table 3, after integrating CoCo, the inference time increased by 132.4, 156.2, 46.0, and 72.9 on Cross-
CodeEval (Python), CrossCodeEval (Java), RepoEval (Line), and RepoEval (API), respectively, corresponding to 1.9%,
2.1%, 1.0%, and 1.6% increases compared to the original method. In all cases, the overhead remains below 5%, indicating
that the integration of CoCo introduces negligible latency. Given the consistent performance improvements it brings
across benchmarks, this small cost is well justified.
Table 3. Experimental Results of Time Efficiency.
Benchmark Base Time(s) Overhead(s)
CrossCodeEval (Python) 7122.9 132.4
CrossCodeEval (Java) 7288.2 156.2
RepoEval (Line) 4723.6 46.0
RepoEval (API) 4667.1 72.9
Answer to RQ4:Integrating CoCo leads to a slight increase in inference latency; however, the performance gains
it brings across various benchmarks justify this overhead, making the trade-off acceptable.
5.5 RQ5: Impact of Contextual Granularity on CoCo
To investigate how context granularity affects CoCo’s code generation performance, we conduct four ablation studies.
The CoCo_func variant supplies only function-level information, enriching the local execution logic of the target function.
The CoCo_file variant provides only file-level information to expose intra-file symbol relationships. The CoCo_pro
variant supplies only project-level information, capturing cross-file dependencies. Finally, CoCo_all combines all three
granularities to examine the effect of multi-granularity input. All experiments are performed on the CrossCodeEval
(Python) benchmark using DeepSeek-Coder-1B as the backbone model. For these ablation variants, no additional
context filtering is applied, allowing us to assess the impact of each granularity in its raw form.
As shown in Fig. 8, all four variants outperform the RLCoder baseline, suggesting that each granularity contributes
positively to code generation. However, the magnitude of improvement varies significantly. Project-level information
produces the largest gains, followed by file-level information, while function-level information yields the smallest
and only marginal improvement. This pattern is expected. Since function-level information mainly describes the
internal logic of the target function, and the LLM already has access to the function as part of its input, it provides
only incremental semantic reinforcement. In contrast, CrossCodeEval is specifically designed for repository-level code
generation, where correct output heavily depends on cross-file references. Thus, project-level information provides the
most influential external semantic cues, naturally resulting in the highest performance boost.
It is worth noting that CoCo_all does not significantly outperform CoCo_pro. This indicates that simply stacking
multi-granularity context does not necessarily yield substantial gains. While each level of contextual information
contributes positively to code generation, combining all levels simultaneously does not further improve performance
Manuscript submitted to ACM

16 Xinkui Zhao et al.
Fig. 8. Performance comparison of different contextual granularity.
and may even introduce redundancy or noise, making it difficult for the model to focus on critical information in
a long context. The superior performance of CoCo compared to CoCo_all supports this observation. By employing
a graph-based multi-granularity context selection mechanism, CoCo effectively filters out noise and highlights key
dependencies, enabling the LLM to fully leverage information across different granularities and achieve significantly
better results than any of the ablation variants. This further validates the effectiveness of the proposed Graph-based
Multi-granularity Context Selection.
Answer to RQ5:All granularities of contextual information positively contribute to CoCo’s code generation per-
formance, with project-level information exerting the strongest influence. And the Graph-based Multi-granularity
Context Selection effectively identifies and prioritizes key information, enabling LLMs to fully leverage multi-
granularity context.
5.6 Case Study
As illustrated in Fig. 9, this case study demonstrates the effectiveness of CoCo in enhancing code generation by
leveraging multi-granular contextual information. Given a piece of unfinished code, CoCo first performs static code
analysis to extract hierarchical contextual information at the function, file, and project levels. In this example, only
function-level and project-level information are informative, so file-level data is omitted for clarity. From the function-
level context, CoCo captures the local execution logic—specifically, the definition and usage of the variable subparser
andcmd_class . The project-level context reveals that the imported class AppsignalCLICommand contains a mfunction
named init_parser , whose input parameter types are related to the variable subparser . Moreover, the next line of the
unfinished code references cmd_class of type AppsignalCLICommand , suggesting that a function of this class is likely
to be invoked. Combining these cues, CoCo successfully generation code. In contrast, RAG-based methods rely solely
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 17
    
        cmd_class.init_parser(subparser)
# src/appsignal/cli/base.py
from __future__ import annotations
import sys
from argparse import ArgumentParser
from .command import AppsignalCLICommand
def _register_commands(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers()
    parser.set_defaults(cmd=None)
    cmd_class: type[AppsignalCLICommand]
    for name, cmd_class in COMMANDS.items():
        subparser = subparsers.add_parser(name=name, 
help=cmd_class.__doc__)
        subparser.set_defaults(cmd=cmd_class)
        cmd_class.>___________________Unfinished Code
CoCo Generation
# src/appsignal/cli/diagnose.py
        parser.add_argument(
            "--send-report",
            action="store_true",
            help="Send the report to AppSignal",
        )
        parser.add_argument(
            "--no-send-report",
            action="store_true",
            help="Do not send the report to AppSignal",
        )Bad Retrieved Code
class AppsignalCLICommand(ABC):
    args: Namespace
    @staticmethod
    def init_parser(parser: ArgumentParser) -> None:
        parser.add_argument(
           "--push-api-key",
           default=os.environ.get("APPSIGNAL_PUSH_API_KEY"),
           help="Push API Key",
        )
        parser.add_argument(
           "--application",
           default=os.environ.get("APPSIGNAL_APP_NAME"),
           help="Application name",
        ) Repo  Level Contextual 
Information
cmd_class.add_arguments(subparser)RAG Generation
# Target Function and Its Analysis:
Assign: subparsers = parser.add_subparsers()
  Call Function: parser.add_subparsers(())
  Call Function: parser.set_defaults((cmd=None))
  Assign: cmd_class = 
Loop Condition: name, cmd_class
  Loop Body:
  Call Function: COMMANDS.items(())
  Assign: subparser = subparsers.add_parser(name=name, 
help=cmd_class.__doc__)
  Call Function: subparsers.add_parser((name=name, 
help=cmd_class.__doc__))
  Call Function: subparser.set_defaults((cmd=cmd_class))Func Level Contextual 
InformationMulti-Granularity 
Contextual Information
Fig. 9. Case Study: Comparing CoCo with Baseline Methods
on surface-level semantic similarity. In this case, the frequent occurrence of the word parser in both the unfinished
code and candidate examples leads to the retrieval of a misleading example containing multiple add_argument() calls,
which subsequently biases the model toward generating incorrect code. While CoCo also retrieves the example, its
generation is further guided by the extracted multi-granular context, enabling it to ignore the misleading patterns and
produce the correct completion. This example highlights the advantage of parsering different levels of code granularity.
Unlike surface-level similarity alone, such enriched contextual comprehension allows CoCo to better infer intent and
generate accurate code completions.
6 Threats to Validity
One potential limitation of our approach lies in the acquisition of multi-granular contextual information. While such
information enables LLMs to better understand the target code and thereby improves generation quality, it is extracted
via static analysis tools using rule-based heuristics. However, these handcrafted rules inevitably overlook certain edge
cases or complex structures in practice. For instance, the control flow graph construction may miss important execution
paths in the presence of intricate logic, which can in turn affect the quality of downstream code generation. In future
work, exploring more robust and adaptive strategies for extracting code-related context could further enhance the
reliability of context construction.
Manuscript submitted to ACM

18 Xinkui Zhao et al.
Another potential limitation lies in the size and relevance of the extracted multi-granular context. In particular,
when the parsed information includes many entire function bodies, it may significantly inflate the input length, leading
to increased computational overhead and, in some cases, exceeding the model’s context window. Moreover, not all
parsed information is equally valuable. Redundant or noisy content may distract the model and negatively impact
generation performance. To mitigate this, future research could investigate LLM-based filtering mechanisms that
preselect salient contextual elements or summarize raw context to reduce noise and token overhead. A further limitation
of our work is its primary focus on the generation of substantive code entities—such as functions, variables, and
control structures—while largely overlooking non-entity code elements that are commonplace in real-world software
engineering. In practical repository-level code completion scenarios, developers frequently need to generate or complete
non-entity code components, including comments, annotations, and formatting symbols [ 10,43]. These elements, while
not directly contributing to program logic, are essential for code readability and maintainability. Future work could
extend our method to better support and assess the generation of these elements.
7 Related Work
7.1 Repository Level Code Completion
Repository-level code completion is highly significant as it better reflects the complexities and requirements of real-
world software development compared to traditional, function-level code completion. This technique is inherently
flexible and can be seamlessly integrated into modern IDEs and programming plugins [ 6], making advanced code
intelligence readily accessible in daily development workflows. Early efforts to enhance neural code generation models
focused on incorporating code structure information, such as leveraging ASTs and static analysis tools. For example,
AST-T5 [ 13] utilizes a T5 model enhanced by AST-guided segmentation and span corruption objectives to integrate
code structural information during pretraining. Similarly, [ 19] proposes enhancing a tree-structured LSTM decoder
by introducing AST-based attention over historical actions and applying multi-task learning for joint prediction of
current and future actions. While an increasing number of large language models (LLMs) have demonstrated strong
performance on general code completion tasks, they often struggle with repository-level code completion due to a lack
of repository-specific knowledge [ 8,25,42]. Repository-level code completion is attracting significant attention as a
key challenge for intelligent software development in real-world scenarios [ 9,24,33,39]. To address this limitation, it is
crucial to effectively inject repository knowledge into LLMs. For example, CoCoMIC [ 9] and RepoFusion [ 39] finetune
LLMs using both in-file and relevant cross-file contexts, thereby incorporating repository knowledge into the models.
However, such methods are generally limited to open-source models. To address the problem, some post-processing
frameworks based on pre-trained models [ 21,42] have been introduced. These frameworks adjust the output probability
of the next token prediction based on the token frequency statistics in the repository.
7.2 Retrieval-Augmented Code Completion
With the rise of Retrieval-Augmented Generation (RAG), it has become a common paradigm for repository-level code
completion. The core idea is to retrieve the top-k semantically similar code snippets from the repository based on the
target code, and use them as examples to guide large language models (LLMs) in generation [ 24,27,41,52]. However,
conventional RAG methods often treat code as plain text during retrieval, neglecting its inherent structural and semantic
characteristics. In order to overcome the limitations of conventional RAG methods, a growing number of approaches
have been suggested to improve the quality of generation by enhancing the retriever, such as incorporating structural
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 19
information or syntax-aware techniques. For example, DraCo [ 5] introduces a dataflow-guided retrieval strategy by
parsing repositories into fine-grained code entities and constructing a repository-specific context graph, enabling more
accurate and structurally aware retrieval than traditional text-based methods. GraphCoder [ 26] introduces a structured,
graph-based retrieval framework that leverages code context graphs to more effectively identify and weigh relevant
code snippets for completion tasks, substantially improving retrieval quality and efficiency. RRG [ 11] introduces a
code refactorer module between retrieval and generation, which compresses and restructures retrieved code into more
model-friendly and answer-relevant contexts, thereby improving code generation quality and reducing inference costs.
Repoformer [ 47] proposes a selective retrieval strategy, allowing the model to decide whether or not to retrieve external
context for a given input. This conditional retrieval mechanism reduces unnecessary overhead and improves robustness.
RLCoder [ 45] introduces a reinforcement learning-based retriever that autonomously learns to select and filter useful
context for code generation without supervision, leveraging feedback from code perplexity and a stop signal mechanism,
thereby achieving significant performance gains over state-of-the-art baselines.
8 CONCLUSION
In this paper, we propose CoCo, a novel framework for repository-level code completion. Unlike other methods that
primarily focus on improving RAG-based retrieval quality, CoCo takes a complementary perspective by enhancing the
model’s comprehension of unfinished code before generation. Specifically, it leverages static analysis tools to extract
multi-granularity contextual information—including function-level, file-level, and project-level semantics—capturing
both the potential execution logic and relevant dependencies. Experimental results demonstrate that CoCo consistently
outperforms state-of-the-art baselines across multiple benchmarks and exhibits strong generalizability, enabling seamless
integration into other code generation methods to further enhance performance.
9 Acknowledgments
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam
Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774(2023).
[2]Miltiadis Allamanis, Earl T Barr, Premkumar Devanbu, and Charles Sutton. 2018. A survey of machine learning for big code and naturalness.ACM
Computing Surveys (CSUR)51, 4 (2018), 1–37.
[3]Marcel Bruch, Martin Monperrus, and Mira Mezini. 2009. Learning from examples to improve code completion systems. InProceedings of the 7th
joint meeting of the European software engineering conference and the ACM SIGSOFT symposium on the foundations of software engineering. 213–222.
[4]Tuan-Dung Bui, Duc-Thieu Luu-Van, Thanh-Phat Nguyen, Thu-Trang Nguyen, Son Nguyen, and Hieu Dinh Vo. 2024. Rambo: Enhancing rag-based
repository-level method body completion.arXiv preprint arXiv:2409.15204(2024).
[5]Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflow-guided retrieval augmentation for repository-level code completion.arXiv preprint
arXiv:2405.19782(2024).
[6] GitHub Copilot. 2023. Github copilot.
[7]Yuxin Ding, Jing Cao, and Ningxin Huang. 2025. Retrieval-Enhanced Method Using Siamese Networks and Graph Kernel Functions for Code
Summarization. InInternational Conference on Neural Information Processing. Springer, 149–163.
[8]Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia,
Dan Roth, et al .2023. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion.Advances in Neural Information Processing
Systems36 (2023), 46701–46723.
[9]Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang.
2022. Cocomic: Code completion by jointly modeling in-file and cross-file context.arXiv preprint arXiv:2212.10007(2022).
[10] Sarah Fakhoury, Devjeet Roy, Adnan Hassan, and Vernera Arnaoudova. 2019. Improving source code readability: Theory and practice. In2019
IEEE/ACM 27th International Conference on Program Comprehension (ICPC). IEEE, 2–12.
[11] Xinyu Gao, Yun Xiong, Deze Wang, Zhenhan Guan, Zejian Shi, Haofen Wang, and Shanshan Li. 2024. Preference-Guided Refactored Tuning for
Retrieval Augmented Code Generation. InProceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering. 65–77.
Manuscript submitted to ACM

20 Xinkui Zhao et al.
[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A survey.arXiv preprint arXiv:2312.109972, 1 (2023).
[13] Linyuan Gong, Mostafa Elhoushi, and Alvin Cheung. 2024. Ast-t5: Structure-aware pretraining for code generation and understanding.arXiv
preprint arXiv:2401.03003(2024).
[14] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al .2024. DeepSeek-Coder:
When the Large Language Model Meets Programming–The Rise of Code Intelligence.arXiv preprint arXiv:2401.14196(2024).
[15] Vincent J Hellendoorn, Christian Bird, Earl T Barr, and Miltiadis Allamanis. 2018. Deep learning type inference. InProceedings of the 2018 26th acm
joint meeting on european software engineering conference and symposium on the foundations of software engineering. 152–162.
[16] Vincent J Hellendoorn and Premkumar Devanbu. 2017. Are deep neural networks the best choice for modeling source code?. InProceedings of the
2017 11th Joint meeting on foundations of software engineering. 763–773.
[17] Vincent J Hellendoorn, Sebastian Proksch, Harald C Gall, and Alberto Bacchelli. 2019. When code completion fails: A case study on real-world
completions. In2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE). IEEE, 960–970.
[18] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al .2024. Qwen2. 5-Coder
Technical Report.arXiv preprint arXiv:2409.12186(2024).
[19] Hui Jiang, Linfeng Song, Yubin Ge, Fandong Meng, Junfeng Yao, and Jinsong Su. 2021. An AST structure enhanced decoder for code generation.
IEEE/ACM Transactions on Audio, Speech, and Language Processing30 (2021), 468–476.
[20] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models
resolve real-world github issues?arXiv preprint arXiv:2310.06770(2023).
[21] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2019. Generalization through memorization: Nearest neighbor
language models.arXiv preprint arXiv:1911.00172(2019).
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in neural information processing systems33
(2020), 9459–9474.
[23] Jia Allen Li, Yongmin Li, Ge Li, Xing Hu, Xin Xia, and Zhi Jin. 2021. Editsum: A retrieve-and-edit framework for source code summarization. In2021
36th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 155–166.
[24] Dianshu Liao, Shidong Pan, Qing Huang, Xiaoxue Ren, Zhenchang Xing, Huan Jin, and Qinying Li. 2023. Context-aware code generation framework
for code repositories: Local, global, and third-party library awareness.CoRR(2023).
[25] Tianyang Liu, Canwen Xu, and Julian McAuley. 2023. Repobench: Benchmarking repository-level code auto-completion systems.arXiv preprint
arXiv:2306.03091(2023).
[26] Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, and Qianxiang Wang. 2024. Graphcoder: Enhancing repository-level
code completion via code context graph-based retrieval and language model.arXiv preprint arXiv:2406.07003(2024).
[27] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey Svyatkovskiy. 2022. Reacc: A retrieval-augmented code completion
framework.arXiv preprint arXiv:2203.07722(2022).
[28] Elijah Mansur, Johnson Chen, Muhammad Anas Raza, and Mohammad Wardat. 2024. RAGFix: Enhancing LLM Code Repair Using RAG and Stack
Overflow Posts. In2024 IEEE International Conference on Big Data (BigData). IEEE, 7491–7496.
[29] Mariana Mără s,oiu, Luke Church, and Alan Blackwell. 2015. An empirical investigation of code completion usage by professional software developers.
InProceedings of the 26th Annual Workshop of the Psychology of Programming Interest Group.
[30] Steve McConnell. 2004.Code complete. Pearson Education.
[31] Manisha Mukherjee and Vincent J Hellendoorn. 2025. Sosecure: Safer code generation with rag and stackoverflow discussions.arXiv preprint
arXiv:2503.13654(2025).
[32] Ahmet Okutan, Samuel Merten, Christoph C Michael, and Ben Ryjikov. 2024. Leveraging RAG-LLM to Translate C++ to Rust. In2024 International
Conference on Assured Autonomy (ICAA). IEEE, 102–105.
[33] Hengzhi Pei, Jinman Zhao, Leonard Lausen, Sheng Zha, and George Karypis. 2023. Better context makes better code language models: A case study
on function call argument completion. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 5230–5238.
[34] Veselin Raychev, Pavol Bielik, and Martin Vechev. 2016. Probabilistic model for code with decision trees.ACM SIGPLAN Notices51, 10 (2016),
731–747.
[35] Veselin Raychev, Martin Vechev, and Eran Yahav. 2014. Code completion with statistical language models. InProceedings of the 35th ACM SIGPLAN
conference on programming language design and implementation. 419–428.
[36] Romain Robbes and Michele Lanza. 2008. How program history can improve code completion. In2008 23rd IEEE/ACM International Conference on
Automated Software Engineering. IEEE, 317–326.
[37] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance framework: BM25 and beyond.Foundations and Trends®in Information
Retrieval3, 4 (2009), 333–389.
[38] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al .
2023. Code llama: Open foundation models for code.arXiv preprint arXiv:2308.12950(2023).
[39] Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, and Torsten Scholak. 2023. Repofusion: Training code models to understand
your repository.arXiv preprint arXiv:2306.10998(2023).
Manuscript submitted to ACM

Completion by Comprehension: Guiding Code Generation with Multi-Granularity Understanding 21
[40] Alexey Svyatkovskiy, Ying Zhao, Shengyu Fu, and Neel Sundaresan. 2019. Pythia: Ai-assisted code completion system. InProceedings of the 25th
ACM SIGKDD international conference on knowledge discovery & data mining. 2727–2735.
[41] Hanzhuo Tan, Qi Luo, Ling Jiang, Zizheng Zhan, Jing Li, Haotian Zhang, and Yuqun Zhang. 2024. Prompt-based code completion via multi-retrieval
augmented generation.ACM Transactions on Software Engineering and Methodology(2024).
[42] Ze Tang, Jidong Ge, Shangqing Liu, Tingwei Zhu, Tongtong Xu, Liguo Huang, and Bin Luo. 2023. Domain adaptive code completion via language
models and decoupled domain databases. In2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 421–433.
[43] Ted Tenny. 1988. Program readability: Procedures versus comments.IEEE Transactions on Software Engineering14, 9 (1988), 1271–1279.
[44] Zhaopeng Tu, Zhendong Su, and Premkumar Devanbu. 2014. On the localness of software. InProceedings of the 22nd ACM SIGSOFT international
symposium on foundations of software engineering. 269–280.
[45] Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and Zibin Zheng. 2024. Rlcoder: Reinforcement learning for repository-
level code completion.arXiv preprint arXiv:2407.19487(2024).
[46] Thomas Weber, Maximilian Brandmaier, Albrecht Schmidt, and Sven Mayer. 2024. Significant productivity gains through programming with large
language models.Proceedings of the ACM on Human-Computer Interaction8, EICS (2024), 1–29.
[47] Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, and Xiaofei Ma. 2024. Repoformer: Selective retrieval for repository-level
code completion.arXiv preprint arXiv:2403.10059(2024).
[48] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong,
Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He,
Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao,
Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren,
Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and
Zhihao Fan. 2024. Qwen2 Technical Report.arXiv preprint arXiv:2407.10671(2024).
[49] Yunwen Ye and Gerhard Fischer. 2002. Supporting reuse by delivering task-relevant and personalized information. InProceedings of the 24th
international conference on Software engineering. 513–523.
[50] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu, Jianqun Chen, et al .2024. Yi:
Open foundation models by 01. ai.arXiv preprint arXiv:2403.04652(2024).
[51] Chi Yu, Guang Yang, Xiang Chen, Ke Liu, and Yanlin Zhou. 2022. Bashexplainer: Retrieval-augmented bash code comment generation based on
fine-tuned codebert. In2022 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, 82–93.
[52] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-
level code completion through iterative retrieval and generation.arXiv preprint arXiv:2303.12570(2023).
[53] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonzalez. 2024. Raft: Adapting language model to
domain specific rag.arXiv preprint arXiv:2403.10131(2024).
[54] Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al .2024. Deepseek-coder-v2:
Breaking the barrier of closed-source models in code intelligence.arXiv preprint arXiv:2406.11931(2024).
[55] Albert Ziegler, Eirini Kalliamvakou, X Alice Li, Andrew Rice, Devon Rifkin, Shawn Simister, Ganesh Sittampalam, and Edward Aftandilian. 2022.
Productivity assessment of neural code completion. InProceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming.
21–29.
Manuscript submitted to ACM
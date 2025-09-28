# Enhancing LLM-based Fault Localization with a Functionality-Aware Retrieval-Augmented Generation Framework

**Authors**: Xinyu Shi, Zhenhao Li, An Ran Chen

**Published**: 2025-09-24 20:37:11

**PDF URL**: [http://arxiv.org/pdf/2509.20552v1](http://arxiv.org/pdf/2509.20552v1)

## Abstract
Fault localization (FL) is a critical but time-consuming task in software
debugging, aiming to identify faulty code elements. While recent advances in
large language models (LLMs) have shown promise for FL, they often struggle
with complex systems due to the lack of project-specific knowledge and the
difficulty of navigating large projects. To address these limitations, we
propose FaR-Loc, a novel framework that enhances method-level FL by integrating
LLMs with retrieval-augmented generation (RAG). FaR-Loc consists of three key
components: LLM Functionality Extraction, Semantic Dense Retrieval, and LLM
Re-ranking. First, given a failed test and its associated stack trace, the LLM
Functionality Extraction module generates a concise natural language
description that captures the failing behavior. Next, the Semantic Dense
Retrieval component leverages a pre-trained code-understanding encoder to embed
both the functionality description (natural language) and the covered methods
(code) into a shared semantic space, enabling the retrieval of methods with
similar functional behavior. Finally, the LLM Re-ranking module reorders the
retrieved methods based on their contextual relevance. Our experiments on the
widely used Defects4J benchmark show that FaR-Loc outperforms state-of-the-art
LLM-based baselines SoapFL and AutoFL, by 14.6% and 9.1% in Top-1 accuracy, by
19.2% and 22.1% in Top-5 accuracy, respectively. It also surpasses all
learning-based and spectrum-based baselines across all Top-N metrics without
requiring re-training. Furthermore, we find that pre-trained code embedding
models that incorporate code structure, such as UniXcoder, can significantly
improve fault localization performance by up to 49.0% in Top-1 accuracy.
Finally, we conduct a case study to illustrate the effectiveness of FaR-Loc and
to provide insights for its practical application.

## Full Text


<!-- PDF content starts -->

1
Enhancing LLM-based Fault Localization with a
Functionality-Aware Retrieval-Augmented
Generation Framework
Xinyu Shi , Zhenhao Li , and An Ran Chen
Abstract—Fault localization (FL) is a critical but time-
consuming task in software debugging, aiming to identify faulty
code elements. While recent advances in large language models
(LLMs) have shown promise for FL, they often struggle with
complex systems due to the lack of project-specific knowledge
and the difficulty of navigating large projects. To address
these limitations, we propose FaR-Loc, a novel framework that
enhances method-level FL by integrating LLMs with retrieval-
augmented generation (RAG). FaR-Loc consists of three key
components: LLM Functionality Extraction, Semantic Dense
Retrieval, and LLM Re-ranking. First, given a failed test and its
associated stack trace, the LLM Functionality Extraction module
generates a concise natural language description that captures the
failing behavior. Next, the Semantic Dense Retrieval component
leverages a pre-trained code-understanding encoder to embed
both the functionality description (natural language) and the
covered methods (code) into a shared semantic space, enabling the
retrieval of methods with similar functional behavior. Finally, the
LLM Re-ranking module reorders the retrieved methods based
on their contextual relevance. Our experiments on the widely used
Defects4J benchmark show that FaR-Loc outperforms state-of-
the-art LLM-based baselines SoapFL and AutoFL, by 14.6% and
9.1% in Top-1 accuracy, by 19.2% and 22.1% in Top-5 accuracy,
respectively. It also surpasses all learning-based and spectrum-
based baselines across all Top-N metrics without requiring re-
training. Furthermore, we find that pre-trained code embedding
models that incorporate code structure, such as UniXcoder, can
significantly improve fault localization performance by up to
49.0% in Top-1 accuracy. Finally, we conduct a case study to
illustrate the effectiveness of FaR-Loc and to provide insights
for its practical application.
Index Terms—Fault localization, large language model, debug-
ging
I. INTRODUCTION
Fault localization (FL) is a fundamental but time-consuming
task in software debugging, with the goal of identifying faulty
code elements. Research indicates that more than half of the
debugging time is spent on finding the fault location [1],
[2], [3]. To reduce manual effort, researchers have introduced
various automated FL techniques. Among these, spectrum-
based [4], [5], [6] and learning-based techniques [7], [8], [9]
have gained popularity due to their effectiveness.
Recent advancements in Large Language Models (LLMs)
have introduced new opportunities for improving FL. In par-
ticular, these models have been pre-trained on large corpus
Xinyu Shi and An Ran Chen are with the Department of Electrical and
Computer Engineering, University of Alberta, Edmonton, Canada (e-mail:
xshi12@ualberta.ca; anran6@ualberta.ca).
Zhenhao Li is with the School of Information Technology, York University,
Toronto, Canada (e-mail: lzhenhao@yorku.ca).of code and textual data, which provides them with strong
capability in code comprehension and reasoning [10], [11].
However, applying LLMs to fault localization presents several
key challenges. First, LLMs often operate with limited de-
bugging information, such as stack traces and test failures.
Without additional insights, the model may fail to capture
the full scope of the fault and tend to hallucinate [12],
[13]. Second, LLMs must navigate the entire codebase, where
identifying relevant context for the fault becomes difficult.
Efficient retrieval techniques are essential to prioritize the
most fault-relevant information, but current approaches often
struggle on large-scale codebases [14], [15]. Finally, reasoning
over complex code structures remains a significant challenge.
LLMs may lack the deep semantic understanding necessary
to track how faults propagate in complex systems, which can
degrade their localization accuracy [16], [17], [18].
Recent LLM-based fault localization methods explore vari-
ous strategies to support code navigation within large projects.
Some methods narrow down the fault location by letting LLMs
make step-wise decisions. One common strategy is hierarchi-
cal navigation [19], [20], [21], [17], [16], where the LLM
first identifies suspicious files or classes before narrowing
down to specific methods. Another approach provides a set
of external functions for LLMs to call [22], such as retrieving
a specific class or accessing comments. However, they are
prone to cascading errors: an early misstep in the retrieval
process (e.g., selecting the wrong file or class) can propagate
into subsequent errors, particularly when LLMs are required
to make decisions under limited contextual information. Other
approaches use LLM-generated code summaries at varying
granularities [23], [17], [16]. Although these summaries can
condense context and potentially facilitate localization, they
may introduce substantial computational overhead, increase
the likelihood of hallucinations and error accumulation, and
omit critical information like root cause of the fault. These
trade-offs underscore the need for more efficient FL techniques
that support robust code navigation while reducing dependence
on large-scale summaries and high-stakes LLM decision-
making.
Recently, retrieval-augmented approaches have been pro-
posed for various downstream software engineering tasks, such
as code generation [24], [25], test case generation [26], [27],
log analysis [28], [29], and automated program repair [30],
[31]. These approaches enhance the capabilities of LLMs by
incorporating task-relevant context through retrieval, all with-
out requiring model re-training. Despite their demonstratedarXiv:2509.20552v1  [cs.SE]  24 Sep 2025

2
success, retrieval-augmented techniques have not yet been
explored in the context of fault localization. In line with this
paradigm, our intuition is thatthe test failure information (e.g.,
stack traces) can be augmented as failing functionality, which
then serves as a retrieval query for identifying suspicious
methods and reasoning about them. This retrieval-augmented
formulation narrows the search space to a semantically and
functionally relevant subset of the codebase, thereby reducing
input complexity.
In this paper, we propose a novel framework, FaR-
Loc (Functionality-Aware faultlocalization Framework via
Retrieval-Augmented LLMs), which leverages a retrieval aug-
mented framework to improve performance in large-context
scenarios. FaR-Loc consists of three main components:LLM
Functionality Extraction,Semantic Dense Retrieval, and
LLM Re-ranking. Specifically, the LLM Functionality Ex-
traction module first augments the sparse debugging infor-
mation (i.e., stack traces) into a functionality query. This
helps to drastically reduce the irrelevant or noisy debugging
information. Then, the functionality query is passed to the
Semantic Dense Retrieval component, which performs the
retrieval process to extract only the most relevant methods
to the functionality query. This step helps reduce the large
search space to prioritize the most suspicious covered methods.
Finally, the LLM Re-ranking module re-ranks the identified
suspicious methods by reasoning around the failing function-
ality. This approach enables the LLM to reason about the
fault using a concise and focused query, rather than being
overwhelmed by excessive test failure information.
We evaluate the effectiveness of FaR-Loc on the widely
adopted Defects4J-V1.2.0 dataset [32], which includes 395
real-world faults across eight studied systems. The results
show that FaR-Loc consistently outperforms both the state-of-
the-art LLM-based and learning-based baselines. Specifically,
FaR-Loc outperforms state-of-the-art LLM-based baselines
SoapFL [17] and AutoFL [22], by 14.6% and 9.1% in Top-1
accuracy, by 19.2% and 22.1% in Top-5 accuracy, respectively.
It also surpasses all learning-based and spectrum-based base-
lines across all Top-N metrics. Our cost analysis also shows
that FaR-Loc is competitively efficient. We further conduct
an ablation study to assess the contribution of individual
component within FaR-Loc. Our results show that when inte-
grating with the re-ranking mechanism, FaR-Loc can filter out
the noisy candidates through semantic reasoning, achieving a
24.8% improvements in Top-5 accuracy. We also analyze the
impact of different LLMs and embedding models on FaR-
Loc’s performance to understand how the choice of model
affects each component. In addition, we conduct a case study
to illustrate the rationale behind FaR-Loc’s effectiveness. Fi-
nally, to assess the generalizability of our approach, we further
evaluate FaR-Loc on 226 additional faults from the Defects4J-
V2.0.0 dataset [33]. Across this extended evaluation, FaR-Loc
continues to outperform all baselines, which demonstrates its
robustness. The main contributions of this paper are:
•We propose a novel LLM-based fault localization frame-
work, FaR-Loc, that combines Large Language Model
(LLM) with Retrieval-Augmented Generation (RAG).
FaR-Loc retrieves suspicious methods by embedding boththe failing functionality and covered methods into a
shared semantic space. It then locates the most suspicious
methods by re-ranking the retrieved methods based on
their contextual relevance to the failing functionality.
•Our extensive evaluation shows that FaR-Loc outperforms
state-of-the-art LLM-based baselines SoapFL [17] and
AutoFL [22], by 14.6% and 9.1% in Top-1 accuracy,
by 19.2% and 22.1% in Top-5 accuracy, respectively. It
also surpasses all learning-based [7], [8] and spectrum-
based [5] baselines across all Top-N metrics.
•We observe that while the functionality queries generated
by different LLMs are largely consistent, their ability to
re-rank suspicious methods varies significantly.
•We observe that code embeddings vary in their ef-
fectiveness for FL. Embedding models that incorporate
Abstract Syntax Trees (ASTs) during pre-training, such
as UniXcoder [34], can boost Top-1 accuracy by up to
49%.
•We conduct a case study to evaluate the effectiveness
of FaR-Loc on complex systems. Our finding shows that
functionality queries provide additional semantic context,
semantic dense retrieval identifies relevant methods, and
LLM re-ranking further refines the candidate list.
II. BACKGROUND& RELATEDWORK
A. Fault Localization
Spectrum-Based and Learning-Based Fault Localization.
Prior to the emergence of LLMs, a variety of automated fault
localization techniques have been developed. Spectrum-Based
Fault Localization (SBFL) methods compute suspiciousness
scores based on the coverage spectra information of passing
and failing test cases [5], but their accuracy remains limited
in practice [35], [36], [37]. Learning-based Fault Localization
(LBFL) utilize machine learning algorithms to learn the fault
patterns from previous bugs [7], [38] or integrates multiple
metrics to rank potential fault locations [39], [8], [9], [40].
Although effective in many scenarios, LBFL often require
large amounts of labeled data and extensive training. They
typically depend on manually crafted features, risk overfitting
to specific projects, and struggle to generalize to previously
unseen systems. Moreover, they lack the ability to provide
explanations for the potential causes of faults.
LLM-based Fault Localization.Recent advancements in
Large Language Models (LLMs) have introduced new op-
portunities for improving fault localization. Early LLM-based
fault localization research primarily concentrated on con-
strained contexts, such as identifying buggy lines within
individual files or methods. Wu et al. [14] prompt ChatGPT-
3.5/4 with buggy methods and error logs for line-level fault
localization, but observe that performance deteriorate signifi-
cantly even when the context is merely expanded to the class
level. LLMAO [18] fine-tunes bidirectional adapters on LLMs
to identify buggy lines, but is limited to fixed-length code
segments within 128 lines and cannot scale to project-level
localization.
Recent LLM-based techniques employ different strategies
to navigate entire codebases, enabling fault localization at

3
the project level. Some approaches incorporate traditional
fault localization techniques or predefined helper functions
to support code navigation. FlexFL [13] relies on traditional
FL techniques to narrow the search space before invoking
LLMs, making its performance heavily dependent on the
quality of initial retrieval. AutoFL [22] equips LLMs with
function-calling to navigate the codebase via externally defined
functions. However, it is constrained by a fixed function call
budget, and at each step, the model must make decisions
based on limited information such as class and method sig-
natures. Other approaches adopt a coarse-to-fine localization
strategy, first narrowing down to files or modules and then
to specific functions, often leveraging LLMs to generate tex-
tual summaries of code for auxiliary guidance. CosFL [16]
clusters code into modules based on the call graph and uses
LLMs to generate summaries for each module and method,
enabling hierarchical localization from modules to methods.
SoapFL [17] employs a multi-agent architecture that performs
file-level localization followed by method-level localization,
and further utilizes LLMs to complete missing documentation
comments. However, when dealing with complex systems,
these approaches may suffer from error propagation across
stages—if the initial step (such as class or module localization)
is incorrect, subsequent efforts will be misguided. Moreover,
these methods rely heavily on LLM-generated textual sum-
maries or documentation, which may exhibit hallucinations or
omit error-relevant details.
Our proposed framework, FaR-Loc, effectively leverages
the functionality information of failing behaviors and employs
retrieval-augmented generation (RAG) techniques to signifi-
cantly reduce the candidate list presented to the LLM. This
design not only enables the LLM to deeply reason about the
failure context, but also mitigates the risk of hallucination
accumulation and error propagation introduced by repeated
and redundant LLM invocations.
B. Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) is a paradigm that
enhances LLMs by integrating external retrieval mecha-
nisms [41], [42], [43]. Unlike traditional retrieval approaches,
RAG dynamically incorporates information from external
sources such as documentation, codebases, or knowledge
bases. This architecture provides three advantages: First, it
substantially reduces hallucination by grounding model out-
puts in factual, retrievable information, thereby improving
response reliability. Second, it enables models to leverage
information beyond their pre-training cutoff, accessing spe-
cialized knowledge without requiring re-training. Finally, RAG
frameworks facilitate continuous knowledge updates through
simple modifications to the retrieval corpus, allowing sys-
tems to remain up-to-date with evolving information without
model redeployment. These capabilities make RAG particu-
larly valuable for domains with rapidly changing information
or specialized knowledge requirements. Although RAG has
been successfully applied to a variety of downstream software
engineering tasks, including code suggestion [44], [15], [45],
code optimization[46], and automated program repair [30],[31], to the best of our knowledge, its integration with LLM-
based fault localization has not yet been explored.
C. Code Embeddings
Code embeddings are dense vector representations of code
snippets that encode their semantic and structural informa-
tion [47]. Early techniques, such as bag-of-words and TF-IDF,
treat code as plain text and fail to capture its hierarchical and
syntactic nature. Later neural approaches like code2vec [48]
and code2seq [49] improve this by leveraging syntax paths
to model code structure, yet they remain limited in handling
long-range dependencies and generalizing across languages.
Transformer-based models have greatly expanded the ap-
plication of code embeddings by enabling richer, context-
aware representations of code. Models such as CodeBERT [50]
and GraphCodeBERT [51] jointly learn from source code
and natural language using masked language modeling and
structural objectives. Later architectures, including UniXcoder,
further advanced this paradigm by incorporating multiple code
modalities, while CodeT5 [52] and CodeT5+ [53] unified
understanding and generation tasks within a single framework.
A key innovation in modern embedding models is their
ability to align code and natural language within a shared
representation space, facilitating cross-modal tasks such as
retrieval and reasoning. In this work, we focus on the repre-
sentation learning capability of these models, specifically their
encoder-side embeddings, which are sufficient for downstream
applications like semantic code search and fault localization.
III. METHODOLOGY
We propose FaR-Loc, a LLM-based framework for method-
level fault localization. FaR-Loc leverages a retrieval-
augmented framework that retrieves code snippets seman-
tically similar to the failing functionality to help in fault
understanding. The core intuition behind FaR-Loc is that by
first understanding which functionality is failing, we can use
that insight to more precisely locate the corresponding faults
in the source code. At a high level, FaR-Loc takes failed
tests and its associated stack trace as inputs to generate the
failing functionality, which then serves as a search query
to find suspicious methods. Figure 1 shows an overview of
our framework, which comprises three components: (1)LLM
Functionality Extraction, (2)Semantic Dense Retrieval, and
(3)LLM Re-ranking. First, FaR-Loc identifies the failing
functionality through test code and stack trace. It leverages the
LLM’s reasoning capabilities to generate a natural language
description of what functionality is broken. Then, FaR-Loc
narrows down potential faulty methods by using the semantic
dense retrieval mechanism to identify code segments that
align with the failing functionality. The objective of this
retrieval process is to pinpoint the methods relevant to the
root cause of the fault. Lastly, we refine the overall ranking of
the suspicious methods list by leveraging an LLM-based re-
ranking mechanism. This step further improves the accuracy of
the ranked list by re-evaluating the relevance of each method
in the context of the failing functionality.

4
OutputFailing T est Input
Covered MethodsEncoder
 query embeddingsIndex embeddingsDense Space
LLMTest Code Stack Tracemethod code </>
 method code </>
Functionality QuerySuspicious list
method 1
method 2Method
codes
< / >
LLMFinal ranking
method 1
method 2
method 3
LLM Functionality Extraction Semantic Dense Retrieval LLM Re-ranking
Fig. 1. Overview of the FaR-Loc framework.
A. LLM Functionality Extraction
In this component, we utilize the LLM to analyze the failing
test information and generate a concise description of the un-
derlying functionality. Prior studies have shown that LLMs are
well-suited for tasks such as code context understanding [10]
and reasoning runtime behaviours [11]. Therefore, we leverage
the LLM’s advanced reasoning capabilities to generate a high-
level functionality description to center the debugging process
around the intent of failing code. The output of this component
is a natural language description, which is then employed as
an augmented query for the semantic dense retrieval stage.
The LLM Functionality Extraction component addresses
the challenge of limited debugging information. Although test
failure information (i.e., stack traces, test results) has been
shown to be useful in LLM-based FL [14], [54], LLMs can
still struggle to identify the fault due to the limited nature of
this information. For instance, stack traces typically contain
only a narrow snapshot of the program’s state at the time
of failure, primarily showing the call stack rather than the
full execution context. As such, they offer a structural view
of the failure but lack contextual and semantic details. To
mitigate this limitation, we generate a functionality query that
provides additional semantic context, clarifying the intended
behavior of the code. This not only improves the LLM’s
understanding of the failure but also filters out noise from
irrelevant information, such as third-party, built-in, or test
helper method calls commonly found in stack traces. As a
result, LLM can better focus on the core failing behavior when
locating faults.
To effectively guide the LLM in uncovering the underlying
functionality information behind failing tests, we carefully
design the prompt tailored for the fault localization task.
Motivated by recent research in LLM prompt engineering [55]
that shows role-playing can effectively stimulate the LLM’s
knowledge in specific domains, we assign LLM the role of a
code assistant. This prompt provides the LLM with the failing
test code snippets and stack traces, and encourages it to reason
about the functionality that is failing. The prompt is shown
below.Prompt for LLM Functionality Extraction
You are a code assistant helping to identify faulty program
behavior. One or more unit tests have failed due to the same
underlying functionality issue.
Given the following test failure information (including multiple
test codes, and stack traces), extract **only** the underlying
functional logic that failed. Your output should be a clean,
concise description of the shared functionality that failed to be
implemented correctly.
Requirements:
- Focus on what functionality failed, not how the tests failed.
- Include any relevant objects, inputs, and expected behavior if
available.
- The description should be precise and suitable for use as a
semantic query to retrieve code (in natural language).
- Avoid unrelated details.
Test name:{test name}
Test code:{test code}
Stack trace:{stack trace}
To ensure the LLM is provided with sufficient context, we
incorporate detailed failing test information, including both the
test code snippets and corresponding stack traces. Specifically,
we extract the fully qualified names of the failing tests from
the test framework and statically locate the associated code
snippets within the source files. In cases where multiple
failing tests share identical test code due to inheritance from a
common test class, we include only the parent test method in
the prompt to avoid redundancy. Additionally, the stack traces
from the failing tests are included to offer further contextual
insights into the failure, enabling the LLM to generate a more
precise and comprehensive functionality description.
B. Semantic Dense Retrieval
In this component, we leverage the failing functionality de-
scription and the failing test coverage as input. Specifically,
we use the functionality description (natural language) as the
query and the methods (code) covered by the failing tests
as the index to identify the most relevant methods. We first
determine the methods covered by the failing tests using
Cobertura [56], a dynamic Java coverage tool integrated into
the Defects4J dataset framework. Next, we extract the code

5
snippets of these covered methods and construct an index
for the retrieval process. The objective is to retrieve methods
that are semantically aligned with the failing functionality
description, thereby improving the likelihood of finding the
actual source of the bug.
To bridge the gap between natural language descrip-
tions and code representations, we leverage pre-trained code-
understanding encoders—such as CodeBERT, CodeT5+, and
UniXcoder—that embed both natural language and code into
a shared semantic space. These models are specifically trained
to capture cross-modal semantic relationships [50], [53], [34],
allowing the functionality description and method implemen-
tations to be meaningfully compared during retrieval. For
the small subset of method snippets that exceed the token
limit of the code embedding models, we divide each method
into smaller chunks that fit within the limit and compute
embeddings for each chunk. To obtain a single representa-
tion for the method, we aggregate these embeddings using
max pooling. This approach is motivated by the observation
that different segments of a method may contain important
semantic information (e.g., parameter lists, return statements),
and max pooling effectively preserves the most salient features
across all chunks. After encoding, the functionality description
is treated as a query, while the code snippets of covered
methods form the retrieval index. We utilize FAISS [57] to
construct an efficient index that supports scalable nearest-
neighbor search. Cosine similarity [58] is used to compute
semantic relevance between the query and indexed methods,
and the top-k most relevant candidates are retrieved.
Our Semantic Dense Retrieval module narrows the input
scope for the LLM by retrieving methods that are semantically
aligned with the failing behavior. Using a cross-modal encoder,
it effectively captures functional similarities between code and
natural language, addressing code and structural complexity.
Prior approaches[17], [59], [22] often depend on limited
contextual cues (e.g., file or class names) or generate syn-
thetic documentation via another LLM, which can introduce
hallucinations or overlook the root cause of the failure. In
contrast, by focusing on functionality, our approach reduces
context size while minimizing hallucination risk.
C. LLM Re-ranking
In this component, the LLM re-ranks the suspicious meth-
ods identified in the previous stage. It leverages the failing
functionality description produced by the LLM Functionality
Extraction module, together with the code snippets of the
retrieved candidate methods, to generate a final ranked list of
suspicious methods. By providing both the high-level function-
ality description and the detailed method implementations, the
LLM can more accurately understand the relationship between
the failing behavior and specific code, which is often essential
for method-level localization [17].
Prior approaches [22], [17] often require the LLM to make
decisions (e.g., invoke external function calls or perform
coarse-grained localization) while being overwhelmed by ex-
cessive failure information and lacking project-specific knowl-
edge. They typically involve multiple rounds of reasoning,increasing the risk of hallucinations and error propagation.
In contrast, our LLM Re-ranking module enables decision-
making over a filtered set of candidate methods that are
functionally relevant to the failure, significantly reducing input
complexity. This focused scope allows full code snippets to
be included within context limits. Moreover, our functionality
query helps the LLM better understand the failing behavior,
which further improves the localization accuracy.
The output of this module is a ranked list of faulty methods,
where the most likely buggy methods are positioned at the
top. To ensure clarity and consistency, the LLM is explicitly
instructed to rank the methods in descending order of their
likelihood of being faulty, with rank 1 assigned to the most
suspicious method.
We also prompt LLM to generate the results in a structured
JSON format, which includes the class name, method name,
and rank for each method. This format not only simplifies
integration with downstream fault localization tasks but also
enhances the interpretability and usability of the results. The
prompt used for this re-ranking step is carefully designed to
guide the LLM in performing an accurate and thorough evalua-
tion. It explicitly instructs the LLM to analyze each method’s
code snippet in the context of the failing functionality and
to produce a ranked list in the specified JSON format. The
prompt is shown below.
Prompt for LLM Re-ranking
You are given several suspicious methods retrieved via
embedding-based search. Your task is to carefully read each
code snippet and determine how likely each method causes the
bug described earlier. Then, **rank the methods** from most
likely buggy (rank 1) to least likely buggy, output is in json form.
Use this JSON output schema:
method ={’class’: str, ’method’:str, ’rank’: int}
return list[method]
class:{class name}
method:{method name}
code snippet:{method code}
IV. EXPERIMENTALSETTINGS
Benchmark.We conduct our experiments on the De-
fects4J [33] dataset, a widely used benchmark for evaluat-
ing fault localization techniques, which includes two stable
versions: Defects4J-v1.2.0 and Defects4J-v2.0.0. The dataset
consists of real-world Java projects with known bugs and
corresponding test cases.
In particular, we use Defects4J-v1.2.0 to compare FaR-Loc
against baseline approaches. This version contains 395 bugs.
During our evaluation, we exclude 14 bugs where the fault
does not reside within any method body, as these represent
edge cases outside the scope of method-level fault localization.
In addition, following prior studies [60], [17], we further
evaluate the generalizability of FaR-Loc’s performance using
226 additional bugs from Defects4J-v2.0.0 in Section VI-A.
Evaluation Metrics.To measure the effectiveness of FaR-Loc,
we employ the following metrics:

6
Top-N denotes the number of bugs for which the actual faulty
method appears within the top N positions in the list.
MAP (Mean Average Precision) captures the average preci-
sion across all bugs, providing a comprehensive measure of
how well FaR-Loc ranks all relevant methods in the list.
MRR (Mean Reciprocal Rank) quantifies the average rank of
the first relevant result, showing how well FaR-Loc ranks the
most relevant method in the list.
Configurations.In RQ1, we adopt OpenAI’s gpt-4.1-mini as
the LLM component to ensure fair comparison with other
baselines, together with Microsoft’s UniXcoder as the code
embedding model for semantic representation. For the subse-
quent analyses in RQ2, RQ3, and RQ4 we employ Google’s
gemini-2.0-flash as a more cost-efficient alternative, while
still preserving the ability to assess the effectiveness of our
framework. During retrieval, our framework obtains the top 40
results from Semantic Dense Retrieval and re-ranks them with
the LLM to produce the final top 10 suspicious methods. The
choice of 40 is empirically determined and further discussed
in the sensitivity analysis of section VII.
V. RESULTS
This section presents the results of our experiments by propos-
ing and answering three research questions (RQs).
A. RQ1: How does FaR-Loc compare to state-of-the-art ap-
proaches?
Motivation.The recent advancements of LLMs in fault local-
ization have attracted significant attention from the software
engineering community. Therefore, in this RQ, we compare
FaR-Loc with recent state-of-the-art approaches, both LLM-
based and non-LLM, to assess its performance in accurately
identifying faulty locations.
Approach.We evaluate the effectiveness of FaR-Loc against
state-of-the-art fault localization techniques from both LLM-
based and non-LLM approaches. A survey study [61] with
practitioners shows that most developers only inspect Top-5
elements during fault localization. Therefore, following prior
work on fault localization [17], [22], [60], [8], we evaluate the
effectiveness in terms of Top-1, Top-3, and Top-5 accuracy.
To ensure consistency, we employ gpt-4.1-mini as the base
LLM for FaR-Loc and all LLM-based baselines. The selected
baselines are as follows:
LLM-based : SoapFL [17], AutoFL [22]. SoapFL adopts a
multi-agent framework that performs fault localization in two
stages: file-level followed by method-level. AutoFL utilizes
the function-calling capabilities of LLMs to autonomously
navigate the codebase and identify buggy code.
Learning-based : GRACE [60], FLUCCS [8]. GRACE uses
Gated Graph Neural Networks to leverage detailed coverage
data and rank program entities. FLUCCS leverages learning-
to-rank techniques to combine SBFL scores with code and
change metrics.
Spectrum-based : Ochiai [62]. A classical SBFL approach that
ranks suspicious elements using the Ochiai formula based on
test failures and coverage data.In addition, we conduct the cost analysis of FaR-Loc both
in terms of API cost and time. For API cost, we calculate
the token usage with the per-million-token price, specifically
$0.15 for input tokens and $0.60 for output tokens [63].
Results.FaR-Loc outperforms the other two LLM-based
baselines SoapFL and AutoFL, by 14.6% and 9.1% in Top-1,
by 19.2% and 22.1% in Top-5, respectively.Table I compares
the results between FaR-Loc and the baseline techniques.
Overall, FaR-Loc consistently outperforms all baselines with
respect to all Top-N metrics. Compared to LLM-based tech-
niques, FaR-Loc correctly locates 228 faults at Top-1, which
represents a 14.6% and 9.1% improvement over SoapFL and
AutoFL, respectively. FaR-Loc also demonstrates better effec-
tiveness under the Top-3 and Top-5 metrics, achieving perfor-
mance improvements ranging from 13.6% to 22.1%. These
results highlight the effectiveness of FaR-Loc’s retrieval-
augmented architecture, which consistently outperforms both
SoapFL’s multi-agent approach and AutoFL’s function-calling
strategy.
To better understand these results, we further investigate the
performance of FaR-Loc on specific systems. We observe that
its performance improves most significantly on Closure, which
is also the most complex system in the benchmark. FaR-Loc
achieves 24.4%, 19.0%, and 26.9% improvements compared
to SoapFL, and 37.8%, 82.9%, and 107.3% compared to
AutoFL at Top-1, Top-3, and Top-5, respectively. We attribute
the effectiveness of FaR-Loc on Closure to its integrated
framework that combines focused functionality query gen-
eration with code embedding-based semantic retrieval. Prior
studies [17], [22] found that localizing faults in complex
systems like Closure is particularly challenging due to the
large search space. For example, the average number of
methods covered by failing tests in Closure is 634, compared
to only 16 to 347 methods in other systems. Since the code
coverage contains excessive information, it becomes difficult
to identify the relevant methods among the many covered
methods. To address this issue, FaR-Loc not only employs
the LLM Functionality Extraction component to generate a
concise query that captures the problematic functionality, but
also applies semantic retrieval over pre-trained code embed-
dings to effectively reduce the reasoning space for the LLM.
The natural language description of the fault helps FaR-Loc
focus on the most important information and avoids the risk
of overwhelming it with unnecessary context.
FaR-Loc also outperforms all learning-based and spectrum-
based methods across every Top-N metric.To further eval-
uate the effectiveness of FaR-Loc, we compare it against
non-LLM fault localization methods, including learning-based
(GRACE and FLUCCS) and spectrum-based (Ochiai) tech-
niques. As shown in Table I, FaR-Loc consistently outperforms
all learning-based and spectrum-based techniques across all
Top-N metrics and all studied systems. FaR-Loc identifies 228
faults at Top-1, exceeding GRACE, FLUCCS, and Ochiai by
36, 68, and 148 faults, respectively. Similarly, in terms of Top-
3 and Top-5, FaR-Loc achieves an improvement between 3.4%
and 73.3% over the other three baselines.
The results demonstrate that LLM-based techniques can
outperform learning-based techniques in zero-shot settings.

7
TABLE I
COMPARISON BETWEENFAR-LOC, LLM-BASED,LEARNING-BASED,AND SPECTRUM-BASED BASELINES ONDEFECTS4J-V1.2.0 (GPT-4.1-MINI)
Systems # BugsLLM-based (gpt-4.1-mini) Learning-based Spectrum-based
FaR-Loc SoapFL AutoFL GRACE FLUCCS Ochiai
Top1 Top3 Top5 Top1 Top3 Top5 Top1 Top3 Top5 Top1 Top3 Top5 Top1 Top3 Top5 Top1 Top3 Top5
Chart 25 17 21 22 18 23 23 21 23 23 14 20 22 15 19 16 6 14 15
Closure 130 51 75 85 41 63 67 37 41 41 47 70 81 42 66 77 14 30 38
Lang 61 54 58 60 45 51 51 52 58 58 42 54 57 40 53 55 24 44 50
Math 104 72 87 89 62 74 75 67 86 87 61 78 89 48 77 83 23 52 62
Mockito 36 19 24 28 20 23 23 17 22 22 17 24 26 7 19 22 7 14 18
Time 25 15 19 20 13 16 16 15 18 18 11 14 19 8 15 18 6 11 13
Overall 381 228 284 304 199 250 255 209 248 249 192 260 294 160 249 271 80 165 196
One key advantage of FaR-Loc is its ability to leverage
pre-trained knowledge to accurately identify faults without
requiring task-specific labeled data, using retrieval augmen-
tation. Since LLMs are trained on massive corpora of code
and natural language, they possess a deep understanding of
language, allowing them to generate contextually relevant and
semantically coherent links between suspicous code and fail-
ing functionality. In contrast, learning-based techniques rely
on large labeled datasets, but their performance often drops
on unseen systems. This advantage is particularly notable
because learning-based techniques can leverage rich but po-
tentially overfitted in-project knowledge. Our findings suggest
that retrieval-augmented frameworks can effectively enhance
general-purpose LLMs by focusing on relevant contextual
information, establishing a new research direction for LLM-
based fault localization approaches.
FaR-Loc demonstrates competitive cost-effectiveness and
efficiency.Specifically, FaR-Loc costs only $0.019 per bug
compared to SoapFL’s $0.055 and AutoFL’s $0.065, which
represents a cost reduction of approximately 65 to 70%. This
significantly lower cost is expected as FaR-Loc only focuses
on functionality-related context, which minimizes API calls
and number of tokens being processed. Beyond monetary
savings, FaR-Loc also achieves high computational efficiency.
FaR-Loc uses open-source models on a single NVIDIA 2080Ti
GPU with merely 728 MB of memory and computes embed-
dings for 100 methods in just 0.986 seconds. This efficiency
leads to faster overall execution, and FaR-Loc can successfully
localize 95% of bugs within 60 seconds.
RQ1 Takeaway
FaR-Loc outperforms LLM-based baselines SoapFL and
AutoFL, by 14.6% and 9.1% in Top-1, and by 19.2% and
22.1% in Top-5, respectively, while remaining comparably
cost-effective. It also outperforms all learning-based and
spectrum-based baselines across every Top-N metric.
B. RQ2: How do our design choices impact localization
effectiveness?
Motivation.In this RQ, we conduct an ablation study to
investigate the contribution of each component individually
and analyze its impact on the effectiveness of fault localization.
The findings may inspire and provide insights for future work
on partially adapting our design.Approach.We investigate the impact of key design choices
on the fault localization performance of FaR-Loc. For each
component, we conduct controlled experiments by removing
or altering it while keeping the others unchanged, so as
to isolate its contribution. This enables us to quantify the
importance of each component and examine whether our
framework contains unnecessary complexity or redundancy.
To this end, we construct the following variants:
w/o functionality query To evaluate the effectiveness of the
functionality query, we use the raw inputs (i.e., test code and
stack trace) directly as the retrieval query. In this setting, we
further examine the effectiveness of incorporating the stack
trace as textual context in query generation, and report that
the quality of the functionality query has a substantial impact
on fault localization performance.
w/o semantic dense retrieval To evaluate the role of semantic
dense retrieval, we remove the code embedding models and
instead employ the traditional approach based on keyword
matching BM25 algorithm [64] to retrieve potentially faulty
methods.
w/o LLM re-ranking To assess the effectiveness of LLM re-
ranking, we directly rely on the initial retrieval results from
semantic dense retrieval and skip the re-ranking step.
Results.LLM functionality extraction improves Top-5 ac-
curacy by 11.2%. Stack traces also provide valuable hints
for LLM-based fault localization.Using raw inputs (i.e., test
code and stack trace) directly as the retrieval query leads to
a noticeable performance degradation. Top-1 accuracy drops
from 216 to 200, Top-3 from 286 to 258, and Top-5 from 307
to 276. Correspondingly, MAP decreases from 0.619 to 0.552,
and MRR drops from 0.665 to 0.607. These results suggest that
raw failure data may introduce noise that negatively impacts
localization accuracy. Incorporating the failing functionality
as a semantic query helps the retriever better filter relevant
methods. Future studies should consider exploring beyond raw
failure data when designing fault localization techniques.
In addition, removing the stack trace information also leads
to a noticeable performance degradation: Top-1 accuracy drops
from 216 to 195, Top-3 from 286 to 252, and Top-5 from
307 to 282. These results show that the presence of stack
trace can significantly enhance the effectiveness of FaR-Loc by
providing additional fault-relevant context (e.g., code elements
related to the failing execution path). Our finding highlights
the usefulness of stack trace in fault localization, which is
consistent with prior studies [65], [66].

8
TABLE II
ABLATION STUDY OFFAR-LOC ONDEFECTS4J-V1.2.0 (GEMINI-2.0-FLASH)
Variants Top1 Top3 Top5 MAP MRR
FaR-Loc 216 286 307 0.619 0.665
w/o functionality query 200 258 276 0.552 0.607
w/o code embeddings (BM25) 186 240 259 0.523 0.566
w/o LLM re-ranking 135 214 246 0.465 0.488
Semantic dense retrieval improves Top-5 accuracy by 18.5%.
Replacing semantic dense retrieval with traditional keyword-
based retrieval (BM25) leads to a significant performance
drop. Specifically, Top-1 accuracy decreases from 216 to 186,
Top-3 from 286 to 240, and Top-5 from 307 to 259. Corre-
spondingly, MAP falls from 0.619 to 0.523, and MRR from
0.665 to 0.566. These results indicate that while keyword-
based retrieval can help identify some fault-relevant methods,
it is limited by its reliance on exact term matching and can-
not capture deeper semantic relationships between the query
and code components. In contrast, semantic dense retrieval
uses pre-trained code representations to match codes and the
intended functionality, rather than relying only on keyword
overlap, resulting in a more reliable starting candidate list for
re-ranking.
LLM re-ranking significantly improves Top-5 accuracy by
24.8%, playing a critical role in improving localization
effectively by filtering out noisy candidates through semantic
reasoning.When LLM re-ranking is disabled, the Top-1
accuracy drops from 216 to 135, Top-3 from 286 to 214, and
Top-5 from 307 to 246. Correspondingly, MAP decreases from
0.619 to 0.465, and MRR drops from 0.665 to 0.488. These
results demonstrate that while the semantic dense retrieval
component can help identify fault-relevant methods, the suspi-
cious list (i.e., generated based on the similarity between the
embeddings of covered methods and the failing functionality
query) does not always reflect the root cause of the faults. The
use of LLM-based re-ranking significantly improves the Top-5
accuracy by further filtering noisy candidates through deeper
semantic reasoning.
RQ2 Takeaway
While all design choices contribute to the overall ef-
fectiveness of FaR-Loc, LLM-based re-ranking plays the
most critical role by filtering noisy candidates through
semantic reasoning, which significantly improves the Top-
5 accuracy by 24.8%.
C. RQ3: What is the effectiveness of FaR-Loc with different
LLMs?
Motivation.Recent research [22] has shown that the choice
of LLM can significantly influence fault localization perfor-
mance. In this RQ, we systematically evaluate the effectiveness
and generalizability of FaR-Loc when integrated with different
LLMs. Specifically, we examine how different LLMs affect
functionality query generation quality and re-ranking accuracy.This analysis reveals how LLM selection impacts each com-
ponent of our framework.
Approach.To evaluate the impact of different LLMs on FaR-
Loc’s performance, we integrate three representative models:
gpt-4.1-mini, gemini-2.0-flash, and gpt-4o-mini. These models
come from different providers and vary in capability and cost.
We use this diversity to examine how LLM choice affects
functionality query generation and re-ranking accuracy. We
conduct our experiments on the Defects4J v1.2.0 benchmark,
with Top-N (N=1, 3, 5), MAP, and MRR metrics. We report
both intermediate retrieval and final re-ranking results to ana-
lyze the contribution of each LLM to the overall framework.
Results.Retrieval results are consistently stable across
LLMs, with differences ranging only from 1.4% to 3.7%.
Our experiments show that the choice of LLM for func-
tionality query generation has trivial impact on the retrieval
performance. As shown in Table III, the number of relevant
methods retrieved at Top-60 ranges from 347 (gpt-4.1-mini) to
352 (gemini-2.0-flash), with a mere 1.4% difference. Similar
consistency appears in the Top-20 and Top-40 results, where
differences remain under 4.0% (3.7% and 3.6%, respectively).
Despite using different LLMs, functionality queries generated
from the same failure information remain highly consistent.
We also invested how well the generated functionality
queries actually match their corresponding buggy methods by
measuring semantic similarity with UniXcoder. The results
show that the cosine similarity scores are consistent across
all LLMs: 0.866 for gpt-4.1-mini, 0.890 for gemini-2.0-flash,
and 0.860 for gpt-4o-mini. These results indicate that all tested
LLMs generate functionality queries that align well with the
target code, and thereby maintain high retrieval performance.
The reasoning ability of the LLM is crucial for effective re-
ranking.In contrast, the choice of LLM has a much greater
impact on re-ranking performance. As shown in Table III,
gpt-4.1-mini achieves the highest Top-1, Top-3, and Top-5
accuracies (228, 284, and 304, respectively), as well as the
best MAP (0.623) and MRR (0.679). By comparison, gpt-4o-
mini performs noticeably worse, with Top-1 accuracy dropping
to 180 (21.1% lower) and MAP to 0.520 (16.5% lower).
Overall, stronger LLMs improve re-ranking effectiveness by
up to 26.7% in Top-1 accuracy and 19.8% in MAP compared
to weaker models. These results suggest that while function-
ality query generation ensures strong retrieval, the reasoning
capabilities of the LLM are crucial for effective re-ranking
and higher fault localization accuracy. We also observe that
although gemini-2.0-flash performs best in retrieval results, its
re-ranking performance is relatively weaker than gpt-4.1-mini
in the Top-1 metric. Our findings suggest that future studies

9
TABLE III
FLPERFORMANCE WITH DIFFERENTLLMS ONDEFECTS4J-V1.2.0
LLMRetrieval Results Re-ranking Results
Top-20 Top-40 Top-60 Top-1 Top-3 Top-5 MAP MRR
gpt-4.1-mini 300 332 347 228 284 304 0.623 0.679
gemini-2.0-flash 309 341 352 216 286 307 0.619 0.665
gpt-4o-mini 298 329 348 180 243 261 0.520 0.566
should consider using different models for different parts of
the pipeline.
RQ3 Takeaway
FaR-Loc’s retrieval performance remains stable across
LLMs, but its re-ranking accuracy varies significantly.
Stronger LLMs improve Top-1 accuracy by up to 26.7%,
underscoring the importance of reasoning ability in FL.
D. RQ4: What is the effectiveeness of FaR-Loc with different
code embedding models?
Motivation.A prior study [47] finds that different code
embeddings can have different impact on the performance of
downstream software engineering tasks. Therefore, in this RQ,
we investigate the generalizability of FaR-Loc across different
embedding models.
Approach.The choice of embedding model plays an impor-
tant role in determining the quality of semantic matching be-
tween failing functionality descriptions and potentially faulty
code components. As such, it directly affects the initial re-
trieval precision and, consequently, the overall localization per-
formance. Therefore, we integrate three representative models
into our framework: CodeBERT, CodeT5+, and UniXcoder.
Each model employs different architectures and pre-training
approaches for code representation.
Results.UniXcoder significantly outperforms other embed-
ding models for fault localization across all Top-N accu-
racy.Table IV compares the fault localization performance
of FaR-Loc when using different code embedding models
for retrieval. Retrieval performance varies greatly across em-
bedding models, which directly determines the upper bound
of the final re-ranking results. Among the evaluated models,
UniXcoder yields the best results across all evaluation metrics.
Specifically, UniXcoder locates 216 faults at Top-1 accu-
racy, substantially outperforming CodeT5+ and CodeBERT by
11.3% and 49.0%, respectively. The MAP and MRR scores
further confirm this trend, with UniXcoder achieving 0.619
and 0.665 respectively, representing relative improvements
of 15.0% (MAP) and 15.2% (MRR) over CodeT5+, and
63.7% (MAP) and 58.3% (MRR) over CodeBERT. These
results suggest that UniXcoder provides more accurate and
semantically relevant code representations, making it better
suited for fault localization tasks.
To further understand the behavior of different embeddings,
we analyze the overlap of Top-1 results across UniXcoder,
CodeT5+, and CodeBERT. As shown in Figure 2, although
there is a intersection among the methods correctly identified
13 15
938
8 55
115
CodeBERT CodeT5+UniXcoderFig. 2. FaR-Loc’s overlap results between different embeddings.
by each embedding, each model also contributes uniquely
localized bugs. This observation highlights the complementary
nature of embedding models—while UniXcoder is overall su-
perior, CodeT5+ and CodeBERT can still successfully localize
some bugs that UniXcoder misses. This motivates future work
to consider ensemble or hybrid retrieval strategies that leverage
the strengths of multiple encoder architectures.
RQ4 Takeaway
The choice of embedding model greatly affects FaR-
Loc’s performance. Specifically, UniXcoder exhibits the
best performance. When integrating or applying FaR-Loc,
practitioners and future studies may consider ensemble or
hybrid retrieval strategies to benefit from multiple encoder
architectures.
VI. DISCUSSION
A. Generalizability of FaR-Loc on Additional Datasets
To evaluate generalizability, we follow prior studies [60],
[13] and assess FaR-Loc on 226 additional faults from the
Defects4J-v2.0.0 benchmark. Table V presents the effective-
ness of FaR-Loc, SoapFL, GRACE and Ochiai. FaR-Loc
outperforms all baseline methods across the Top-N metrics. In
particular, FaR-Loc localizes 122 bugs within Top-1, outper-
forming SoapFL (99), GRACE (85), and Ochiai (32). The Top-
1 improvement over GRACE (i.e., 43.5%) is even higher than
that observed on Defects4J-V1.2.0 (i.e., 18.8%). Moreover, the
improvements persist in the Top-3 and Top-5 metrics, where
FaR-Loc locates 159 and 167 faults, respectively. These results
suggest that FaR-Loc performs even better on newer and more
diverse studied systems, highlighting its generalizability across
different benchmarks.

10
TABLE IV
FLPERFORMANCE WITH DIFFERENT EMBEDDING MODELS ONDEFECTS4J-V1.2.0 (GEMINI-2.0-FLASH)
Embedding ModelRetrieval Results Re-ranking Results
Top-20 Top-40 Top-60 Top-1 Top-3 Top-5 MAP MRR
UniXcoder 309 341 352 216 286 307 0.619 0.665
CodeT5+ 218 271 290 194 237 258 0.539 0.577
CodeBERT 147 191 213 145 173 181 0.379 0.420
TABLE V
COMPARISON OFFAULTLOCALIZATIONTECHNIQUES ON
DEFECTS4J-V2.0.0 (GPT-4.1-MINI)
Project # Bugs Techniques Top1 Top3 Top5
Overall 214FaR-Loc 122 159 167
SoapFL 99 128 134
GRACE 85 119 140
Ochiai 32 74 93
B. Case Study and Implications
To demonstrate the practical implications of our approach,
we conduct a case study to showcase the effectiveness of
FaR-Loc in addressing two major challenges in fault local-
ization. First, LLM-based fault localization often struggles
to locate faults in large search space (i.e., the entire code
software system). Although prior studies [17], [13], [14] have
proposed methods for automatically identifying fault-relevant
methods using LLMs, their overall effectiveness often falls
short compared to learning-based approaches. For example,
SoapFL’s Top-1 metric on Closure, the largest project in
Defects4J, drops to 31.5%, significantly lower than its average
performance on other systems, 62.9%. Second, while many
LLM-based methods [16], [13], [22], [12] attempt to leverage
the failing test information (e.g., test code and stack trace),
they directly use this information as context to improve
localization performance. Yet, such information often contains
limited or irrelevant information for fault diagnosis (e.g.,
third-party method calls or code executed after the failure
point). Therefore, to investigate the effectiveness of FaR-Loc
in addressing theses challenges, we showcase a fault (i.e.,
Closure-112) that contains limited diagnostic information from
a large and complex system.
Figure 3 presents the details ofClosure-112, including the
test code, stack trace, buggy method and the functionality
query generated by FaR-Loc. By combining the LLM’s natural
language understanding capabilities with the code compre-
hension strength of embedding models, FaR-Loc successfully
localizes the buggy method at Top-1, effectively addressing
both identified challenges. In comparison, SoapFL locates the
fault beyond Top-5.
Effectiveness of Functionality Query.As illustrated in Fig-
ure 3, both the test code and stack trace provide limited
diagnostic information for localizing the buggy method. The
test code only invokes a helper function,testTypes, while
the stack trace presents only a partial execution path up to
the assertion failure, omitting intermediate method calls. FaR-
Loc addresseses this challenge by first reasoning about theTABLE VI
COMPARISON OFCODEEMBEDDINGMODELS
Model Year Architecture Pre-training Data
CodeBERT 2020 Encoder-only CodeSearchNet
UniXcoder 2022 Unified TransformerCodeSearchNet,
ASTs
CodeT5+ 2023 Encoder-decoder (T5)CodeSearchNet,
GitHub Code Dataset
failing functionality. It identifies that the issue lies in the
type checker, whichincorrectly infers or propagates types
in templated functions. More specifically, it determines that
there is a type mismatch when calling a templated function, as
highlighted in the functionality query shown in Figure 3. This
insight helps guide FaR-Loc’s semantic search for relevant
methods. Notably, keywords such asinfersandcallingdo not
appear in the test code or stack trace, yet are highly relevant
to the faulty methodinferTeamplatedTypesForCall
due to their semantic similarity. The functionality query is a
contributing factor to the effectiveness of FaR-Loc. Without it,
test code and stack trace alone contain too much noise, which
may hinder the accurate retrieval and localization of fault-
relevant methods. In the absence of the functionality query,
FaR-Loc places the faulty method outside the Top-5. With
this semantic clue, however, FaR-Loc re-rank it to Top-1.
Implication. Incorporating failing functionality queries en-
hances FL by generating additional semantic context that
goes beyond the information available in test code and stack
traces. Our results show that FaR-Loc achieves significant im-
provements when reasoning about failing functionality. Future
studies should explore the systematic integration of failing
functionality insights to further improve the effectiveness of
FL techniques.
Effectiveness of Embedding Models.We further evaluate
the impact of different code embedding models (CodeBERT,
UniXcoder, and CodeT5+) on Closure-112. While all three
models are well-established for code representation tasks,
their effectiveness varies considerably in our framework. In
particular, our framework with UniXcoder successfully ranks
the faulty method at Top-1, whereas those with CodeBERT
and CodeT5+ place it outside the Top-5. This performance gap
may be attributed to differences in their pre-training process.
Table VI summarizes the key characteristics of the three code
embedding models, including their architectural design and
pre-training corpora. Notably, our framework with UniXcoder
outperforms those with CodeT5+ and CodeBERT across all

11
  Buggy Method (ground truth):   
   private boolean  inferTemplatedTypesForCall (Node n, FunctionType fnType){
      - - -
      // Try to infer the template types
     Map<TemplateType, JSType> inferred = 
          inferTemplateTypesFromParameters(fnType, n);
      // Replace all template types. If we couldn't find a replacement, we
      // replace it with UNKNOWN.
      TemplateTypeReplacer replacer = new TemplateTypeReplacer(registry, inferred);
      - - -
      return replacer.madeChanges;
    }  Test Code:
  public void testTemplatized11() throws Exception {
    testTypes(
        "/** \n" +
        " * @template T\n" +
        " * @constructor\n" +
        " */\n" +
        "function C() {}\n" +
        "\n" +
        "/**\n" +
        " * @param {T|K} a\n" +
        " * @return {T}\n" +
        " * @template K\n" +
        " */\n" +
        "C.prototype.method = function (a) {};\n" +
        "\n" +
        // method returns "?"
        "/** @type {void} */ var x = new C().method(1);");
  }
Stack Trace:
 --- com.google.javascript.jscomp.TypeCheckTest::testTemplatized11
 junit.framework.AssertionFailedError: unexpected warnings(s):
 JSC_TYPE_MISMATCH. initializing variable
 found   : number
 required: undefined at [testcode] line 14 : 28
    at junit.framework.Assert.fail(Assert.java:57)
    at junit.framework.TestCase.fail(TestCase.java:227)
    at com.google.javascript.jscomp.TypeCheckTest.testTypes(TypeCheckTest.java:12407)
    at com.google.javascript.jscomp.TypeCheckTest.testTypes(TypeCheckTest.java:12381)
    at com.google.javascript.jscomp.TypeCheckTest.testTypes(TypeCheckTest.java:12317)
    at com.google.javascript.jscomp.TypeCheckTest.testTypes(TypeCheckTest.java:12313)Functionality Query:
The type checker incorrectly infers or propagates types  in templated 
functions or methods. Specifically, there is a type mismatch when  calling 
a templated function  with a parameter that has a different type than 
expected, or when the return type of a templated function is not correctly 
inferred during assignment.Semantic dense retrievalLLM re -ranking Top-1 hit
Top-20 hitComplex Closure  system
Limited  bug information
Fig. 3. Case Study of FaR-Loc on Closure-112.
localization metrics, likely due to its distinctive use of Abstract
Syntax Trees (ASTs) during pre-training [34]. The structural
knowledge encoded by ASTs enhances UniXcoder’s ability to
capture both code semantics and syntax structure, which is
particularly beneficial for fault localization tasks.
Interestingly, CodeT5+, despite being trained on a substan-
tially larger dataset with token counts reportedly 50 times
greater than CodeSearchNet [53], performs worse than UniX-
coder. This suggests that merely scaling up training data
without incorporating structural signals may be insufficient for
tasks that benefit from fine-grained code understanding.
Implication. The effectiveness of different code embedding
models in fault localization varies considerably. Models that
incorporate structural code information—such as Abstract
Syntax Trees (ASTs) in UniXcoder—consistently outperform
those that rely solely on large-scale pre-training data. This
suggests that future FL approaches should further exploit
structural code representations to enhance embedding quality.
Effectiveness of LLM Re-ranking.As demonstrated
in our earlier ablation study, LLM re-ranking plays a
crucial role in our framework. This is further evi-
denced in the Closure-112 case study. The buggy method
inferTemplatedTypesForCalldoes not always appear
at the top of the initial candidate list produced by semantic
dense retrieval. However, after applying LLM re-ranking the
correct method is consistently ranked at Top-1. We attribute
this to the fact that semantic dense retrieval may introduce
noise, resulting in irrelevant methods being included in the
candidate list. That been said, the LLM re-ranking module
can compensate for this by analyzing the code and performing
deeper reasoning, which allows FaR-Loc to accurately priori-tize the actual buggy method.
Implication. LLM re-ranking effectively mitigates the noise in-
troduced by dense retrieval, leveraging its code understanding
and reasoning capabilities to improve the accuracy of fault
localization. This suggests that future studies should further
explore specialized re-ranking strategies, or similar hybrid
designs that combine lightweight retrieval techniques with
reasoning-intensive LLMs.
VII. THREATS TOVALIDITY
Internal Validity.One potential threat to internal validity is
data leakage. Since the Defects4J dataset may be included
in the training data of LLMs, there is a risk that the models
could inadvertently access information about the defects under
study. To mitigate this risk, we ensure that our framework does
not expose any project names, identifiers, or human-provided
labels during the evaluation process.
External Validity.A threat to external validity concerns the
generalizability of our results. Our experiments are primarily
conducted on the Defects4J-v1.2.0 dataset, which may not
fully represent all real-world software projects. To address this
threat, following prior studies [17], [60], we further evaluate
FaR-Loc on 226 additional faults from Defects4J-v2.0.0, and
the consistent results across versions support the generaliz-
ability of our findings. Additionally, our current evaluation is
limited to Java projects. Future work is needed to assess the
applicability of our approach to other programming languages.
Sensitivity Analysis.We conduct a sensitivity analysis to
examine the impact of varying the number of retrieved results
provided to the LLM re-ranking component. Specifically, we
experiment with supplying the top 20, 40, and 60 retrieval

12
results. The performance remains similar, with only a 3.36%
variation across these settings. We use 40 retrieved results in
our approach, as it yields the best outcome.
VIII. CONCLUSION
In this paper, we propose FaR-Loc, a novel fault local-
ization framework that integrates LLMs with RAG. FaR-
Loc introduces three components: LLM Functionality Ex-
traction, Semantic Dense Retrieval, and LLM Re-Ranking,
which collectively advance the effectiveness of fault local-
ization. On the Defects4J-v1.2.0 benchmark, FaR-Loc sig-
nificantly outperforms state-of-the-art LLM-based, learning-
based and spectrum-based baselines, achieving consistent Top-
1 improvements. The API usage and runtime cost of FaR-
Loc are comparable to those of other LLM-based methods,
making it a practical choice for fault localization. Our ablation
study shows that each component contributes to the overall
performance of FaR-Loc. Further analysis reveals that the
choice of models plays a important role in its effectiveness.
While functionality queries generated by different LLMs are
largely consistent, their effectiveness in re-ranking suspicious
methods varies considerably. Similarly, code embedding mod-
els that incorporate Abstract Syntax Trees (ASTs), such as
UniXcoder, achieve greater improvements and highlight the
importance of structural representation in fault localization.
To the best of our knowledge, FaR-Loc is the first fault
localization approach that leverages RAG framework with
strong performance. Our findings suggest that integrating
LLMs with RAG and functionality-aware design can bridge
the gap between general-purpose language models and the
challenges of real-world fault localization.
REFERENCES
[1] M. B ¨ohme, E. O. Soremekun, S. Chattopadhyay, E. Ugherughe, and
A. Zeller, “Where is the bug and how is it fixed? an experiment
with practitioners,” inProceedings of the 2017 11th Joint Meeting on
Foundations of Software Engineering. Paderborn Germany: ACM, Aug.
2017, pp. 117–128.
[2] A. Alaboudi and T. D. LaToza, “An exploratory study of debugging
episodes,”arXiv preprint arXiv:2105.02162, 2021.
[3] A. R. Chen, T.-H. P. Chen, and S. Wang, “Demystifying the challenges
and benefits of analyzing user-reported logs in bug reports,”Empirical
Software Engineering, vol. 26, no. 1, pp. 1–30, 2021.
[4] R. Abreu, P. Zoeteweij, and A. J. Van Gemund, “An evaluation of sim-
ilarity coefficients for software fault localization,” in2006 12th Pacific
Rim International Symposium on Dependable Computing (PRDC’06).
IEEE, 2006, pp. 39–46.
[5] ——, “On the accuracy of spectrum-based fault localization,” inTesting:
Academic and industrial conference practice and research techniques-
MUTATION (TAICPART-MUTATION 2007). IEEE, 2007, pp. 89–98.
[6] W. E. Wong, V . Debroy, R. Gao, and Y . Li, “The dstar method for
effective software fault localization,”IEEE Transactions on Reliability,
vol. 63, no. 1, pp. 290–308, 2013.
[7] C.-P. Wong, Y . Xiong, H. Zhang, D. Hao, L. Zhang, and H. Mei,
“Boosting bug-report-oriented fault localization with segmentation and
stack-trace analysis,” in2014 IEEE international conference on software
maintenance and evolution. IEEE, 2014, pp. 181–190.
[8] J. Sohn and S. Yoo, “Fluccs: Using code and change metrics to
improve fault localization,” inProceedings of the 26th ACM SIGSOFT
International Symposium on Software Testing and Analysis, 2017, pp.
273–283.
[9] X. Li, W. Li, Y . Zhang, and L. Zhang, “Deepfl: Integrating multiple fault
diagnosis dimensions for deep fault localization,” inProceedings of the
28th ACM SIGSOFT International Symposium on Software Testing and
Analysis, 2019, pp. 169–180.[10] D. Nam, A. Macvean, V . Hellendoorn, B. Vasilescu, and B. Myers,
“Using an llm to help with code understanding,” inProceedings of
the IEEE/ACM 46th International Conference on Software Engineering,
2024, pp. 1–13.
[11] J. Chen, Z. Pan, X. Hu, Z. Li, G. Li, and X. Xia, “Reasoning runtime
behavior of a program with llm: How far are we?” inProceedings of
the IEEE/ACM 47th International Conference on Software Engineering,
2025.
[12] M. N. Rafi, D. J. Kim, T.-H. Chen, and S. Wang, “Enhancing fault
localization through ordered code analysis with llm agents and self-
reflection,”arXiv preprint arXiv:2409.13642, 2024.
[13] C. Xu, Z. Liu, X. Ren, G. Zhang, M. Liang, and D. Lo, “Flexfl: Flexible
and effective fault localization with open-source large language models,”
IEEE Transactions on Software Engineering, 2025.
[14] Y . Wu, Z. Li, J. M. Zhang, M. Papadakis, M. Harman, and Y . Liu, “Large
language models in fault localisation,”arXiv preprint arXiv:2308.15276,
2023.
[15] Z. Yang, S. Chen, C. Gao, Z. Li, X. Hu, K. Liu, and X. Xia, “An
empirical study of retrieval-augmented code generation: Challenges
and opportunities,”ACM Transactions on Software Engineering and
Methodology, 2025.
[16] Y . Qin, S. Wang, Y . Lei, Z. Zhang, B. Lin, X. Peng, L. Chen, and
X. Mao, “Fault localization from the semantic code search perspective,”
arXiv preprint arXiv:2411.17230, 2024.
[17] Y . Qin, S. Wang, Y . Lou, J. Dong, K. Wang, X. Li, and X. Mao,
“SoapFL: A Standard Operating Procedure for LLM-based Method-
Level Fault Localization,”IEEE Transactions on Software Engineering,
pp. 1–15, 2025.
[18] A. Z. H. Yang, C. Le Goues, R. Martins, and V . Hellendoorn, “Large
Language Models for Test-Free Fault Localization,” inProceedings of
the IEEE/ACM 46th International Conference on Software Engineering.
Lisbon Portugal: ACM, Feb. 2024, pp. 1–12.
[19] Z. Jiang, X. Ren, M. Yan, W. Jiang, Y . Li, and Z. Liu, “Cosil: Software
issue localization via llm-driven code repository graph searching,”arXiv
preprint arXiv:2503.22424, 2025.
[20] Z. Yu, H. Zhang, Y . Zhao, H. Huang, M. Yao, K. Ding, and J. Zhao,
“Orcaloca: An llm agent framework for software issue localization,”
arXiv preprint arXiv:2502.00350, 2025.
[21] Z. Chen, X. Tang, G. Deng, F. Wu, J. Wu, Z. Jiang, V . Prasanna,
A. Cohan, and X. Wang, “Locagent: Graph-guided llm agents for code
localization,”arXiv preprint arXiv:2503.09089, 2025.
[22] S. Kang, G. An, and S. Yoo, “A Quantitative and Qualitative Evaluation
of LLM-Based Explainable Fault Localization,”Proceedings of the ACM
on Software Engineering, vol. 1, no. FSE, pp. 1424–1446, Jul. 2024.
[23] I. Yeo, D. Ryu, and J. Baik, “Improving llm-based fault localization with
external memory and project context,”arXiv preprint arXiv:2506.03585,
2025.
[24] J. Li, Y . Li, G. Li, Z. Jin, Y . Hao, and X. Hu, “Skcoder: A sketch-
based approach for automatic code generation,” in2023 IEEE/ACM 45th
International Conference on Software Engineering (ICSE). IEEE, 2023,
pp. 2124–2135.
[25] S. Zhou, U. Alon, F. F. Xu, Z. Wang, Z. Jiang, and G. Neubig,
“Docprompting: Generating code by retrieving the docs,”arXiv preprint
arXiv: 2207.05987, 2022.
[26] J. Shin, R. Aleithan, H. Hemmati, and S. Wang, “Retrieval-augmented
test generation: How far are we?”arXiv preprint arXiv:2409.12682,
2024.
[27] C. Arora, T. Herda, and V . Homm, “Generating test scenarios from nl
requirements using retrieval-augmented llms: An industrial study,” in
2024 IEEE 32nd International Requirements Engineering Conference
(RE). IEEE, 2024, pp. 240–251.
[28] Z. Ma, D. J. Kim, and T.-H. Chen, “Librelog: Accurate and efficient
unsupervised log parsing using open-source large language models,”
arXiv preprint arXiv:2408.01585, 2024.
[29] W. Zhang, Q. Zhang, E. Yu, Y . Ren, Y . Meng, M. Qiu, and J. Wang,
“Lograg: Semi-supervised log-based anomaly detection with retrieval-
augmented generation,” in2024 IEEE International Conference on Web
Services (ICWS). IEEE, 2024, pp. 1100–1102.
[30] W. Wang, Y . Wang, S. Joty, and S. C. Hoi, “Rap-gen: Retrieval-
augmented patch generation with codet5 for automatic program repair,”
inProceedings of the 31st ACM Joint European Software Engineering
Conference and Symposium on the Foundations of Software Engineering,
2023, pp. 146–158.
[31] N. Nashid, M. Sintaha, and A. Mesbah, “Retrieval-based prompt se-
lection for code-related few-shot learning,” in2023 IEEE/ACM 45th
International Conference on Software Engineering (ICSE). IEEE, 2023,
pp. 2450–2462.

13
[32] R. Just, D. Jalali, and M. D. Ernst, “Defects4j: A database of existing
faults to enable controlled testing studies for java programs,” inPro-
ceedings of the 2014 International Symposium on Software Testing and
Analysis, 2014, pp. 437–440.
[33] “The defects4j dataset version 2.0.0,” https://github.com/rjust/defects4j,
2022, last accessed in May 2025.
[34] D. Guo, S. Lu, N. Duan, Y . Wang, M. Zhou, and J. Yin, “Unixcoder:
Unified cross-modal pre-training for code representation,”arXiv preprint
arXiv:2203.03850, 2022.
[35] S. Pearson, J. Campos, R. Just, G. Fraser, R. Abreu, M. D. Ernst,
D. Pang, and B. Keller, “Evaluating and improving fault localization,”
in2017 IEEE/ACM 39th International Conference on Software Engi-
neering (ICSE). IEEE, 2017, pp. 609–620.
[36] M. Wen, J. Chen, Y . Tian, R. Wu, D. Hao, S. Han, and S.-C. Cheung,
“Historical Spectrum Based Fault Localization,”IEEE Transactions on
Software Engineering, vol. 47, no. 11, pp. 2348–2368, 2019.
[37] P. S. Kochhar, X. Xia, D. Lo, and S. Li, “Practitioners’ expectations on
automated fault localization,” inProceedings of the 25th international
symposium on software testing and analysis, 2016, pp. 165–176.
[38] F. Niu, C. Li, K. Liu, X. Xia, and D. Lo, “When deep learning meets
information retrieval-based bug localization: A survey,”ACM Computing
Surveys, 2025.
[39] T.-D. B. Le, D. Lo, C. Le Goues, and L. Grunske, “A learning-to-rank
based fault localization approach using likely invariants,” inProceedings
of the 25th International Symposium on Software Testing and Analysis,
2016, pp. 177–188.
[40] P. Chakraborty, M. Alfadel, and M. Nagappan, “Rlocator: Reinforcement
learning for bug localization,”IEEE Transactions on Software Engineer-
ing, 2024.
[41] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
H. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,”arXiv preprint arXiv:2312.10997, vol. 2,
no. 1, 2023.
[42] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, “Retrieval
augmentation reduces hallucination in conversation,”arXiv preprint
arXiv:2104.07567, 2021.
[43] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language
models in retrieval-augmented generation,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 17 754–
17 762.
[44] J. Chen, X. Hu, Z. Li, C. Gao, X. Xia, and D. Lo, “Code search is all you
need? improving code suggestions with code search,” inProceedings of
the IEEE/ACM 46th International Conference on Software Engineering,
2024, pp. 1–13.
[45] Y . Xu, F. Lin, J. Yang, N. Tsantaliset al., “Mantra: Enhancing auto-
mated method-level refactoring with contextual rag and multi-agent llm
collaboration,”arXiv preprint arXiv:2503.14340, 2025.
[46] S. Gao, C. Gao, W. Gu, and M. Lyu, “Search-based llms for code
optimization,” in2025 IEEE/ACM 47th International Conference on
Software Engineering (ICSE). IEEE Computer Society, 2024, pp. 254–
266.
[47] Z. Ding, H. Li, W. Shang, and T.-H. P. Chen, “Can pre-trained code
embeddings improve model performance? revisiting the use of code
embeddings in software engineering tasks,”Empirical Software Engi-
neering, vol. 27, no. 3, p. 63, 2022.
[48] U. Alon, M. Zilberstein, O. Levy, and E. Yahav, “code2vec: Learning
distributed representations of code,”Proceedings of the ACM on Pro-
gramming Languages, pp. 1–29, 2019.
[49] U. Alon, S. Brody, O. Levy, and E. Yahav, “code2seq: Generating
sequences from structured representations of code,”arXiv preprint
arXiv:1808.01400, 2018.
[50] Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin,
T. Liu, D. Jianget al., “Codebert: A pre-trained model for programming
and natural languages,”arXiv preprint arXiv:2002.08155, 2020.
[51] D. Guo, S. Ren, S. Lu, Z. Feng, D. Tang, S. Liu, L. Zhou, N. Duan,
A. Svyatkovskiy, S. Fuet al., “Graphcodebert: Pre-training code repre-
sentations with data flow,”arXiv preprint arXiv:2009.08366, 2020.
[52] Y . Wang, W. Wang, S. Joty, and S. C. Hoi, “Codet5: Identifier-aware
unified pre-trained encoder-decoder models for code understanding and
generation,”arXiv preprint arXiv:2109.00859, 2021.
[53] Y . Wang, H. Le, A. D. Gotmare, N. D. Q. Bui, J. Li, and S. C. H. Hoi,
“CodeT5+: Open Code Large Language Models for Code Understanding
and Generation,” May 2023.
[54] S. Kang, J. Yoon, and S. Yoo, “Large Language Models are Few-shot
Testers: Exploring LLM-based General Bug Reproduction,” in2023
IEEE/ACM 45th International Conference on Software Engineering
(ICSE). Melbourne, Australia: IEEE, May 2023, pp. 2312–2323.[55] Y . Cheng, J. Chen, Q. Huang, Z. Xing, X. Xu, and Q. Lu, “Prompt
sapper: a llm-empowered production tool for building ai chains,”ACM
Transactions on Software Engineering and Methodology, pp. 1–24,
2024.
[56] “Cobertura,” https://cobertura.github.io/cobertura/, 2025, last accessed in
May 2025.
[57] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar ´e,
M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,”arXiv preprint
arXiv:2401.08281, 2024.
[58] G. Salton, A. Wong, and C.-S. Yang, “A vector space model for
automatic indexing,”Communications of the ACM, vol. 18, no. 11, pp.
613–620, 1975.
[59] Y . Zhang, H. Ruan, Z. Fan, and A. Roychoudhury, “Autocoderover:
Autonomous program improvement,” inProceedings of the 33rd ACM
SIGSOFT International Symposium on Software Testing and Analysis,
2024, pp. 1592–1604.
[60] Y . Lou, Q. Zhu, J. Dong, X. Li, Z. Sun, D. Hao, L. Zhang, and
L. Zhang, “Boosting coverage-based fault localization via graph-based
representation learning,” inProceedings of the 29th ACM Joint Meeting
on European Software Engineering Conference and Symposium on the
Foundations of Software Engineering, 2021, pp. 664–676.
[61] P. S. Kochhar, X. Xia, D. Lo, and S. Li, “Practitioners’ expectations on
automated fault localization,” inProceedings of the 25th International
Symposium on Software Testing and Analysis. Saarbr ¨ucken Germany:
ACM, Jul. 2016, pp. 165–176.
[62] R. Abreu, P. Zoeteweij, and A. J. van Gemund, “On the accuracy
of spectrum-based fault localization,” inTesting: Academic and In-
dustrial Conference Practice and Research Techniques - MUTATION
(TAICPART-MUTATION 2007), 2007, pp. 89–98.
[63] “Gpt-4o mini: advancing cost-efficient intelligence,” https://platform.
openai.com/docs/models/gpt-4o-mini, 2025, last accessed September
2025.
[64] S. Robertson, H. Zaragozaet al., “The probabilistic relevance frame-
work: Bm25 and beyond,”Foundations and Trends® in Information
Retrieval, vol. 3, no. 4, pp. 333–389, 2009.
[65] A. R. Chen, “An empirical study on leveraging logs for debugging
production failures,” in2019 IEEE/ACM 41st International Conference
on Software Engineering: Companion Proceedings (ICSE-Companion).
IEEE, 2019, pp. 126–128.
[66] L. B. S. Pacheco, A. R. Chen, J. Yanget al., “Leveraging stack traces for
spectrum-based fault localization in the absence of failing tests,”arXiv
preprint arXiv:2405.00565, 2024.
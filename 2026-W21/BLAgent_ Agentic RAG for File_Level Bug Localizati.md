# BLAgent: Agentic RAG for File-Level Bug Localization

**Authors**: Md Afif Al Mamun, Gias Uddin

**Published**: 2026-05-18 07:20:13

**PDF URL**: [https://arxiv.org/pdf/2605.17965v1](https://arxiv.org/pdf/2605.17965v1)

## Abstract
Bug localization remains a key bottleneck in downstream software maintenance tasks, including root cause analysis, triage, and automated program repair (APR), despite recent advances in large language model (LLM)-based repair systems. File-level bug localization is especially critical in hierarchical pipelines, where errors can propagate to downstream stages such as statement-level localization or patch generation. While Retrieval-Augmented Generation (RAG) offers a promising direction for grounding LLMs in repository context, existing RAG pipelines rely on static retrieval and lack the reasoning needed to identify faulty code accurately. In this work, we present BLAgent, a novel agentic RAG framework for file-level bug localization that integrates three key ideas: (i) code structure-aware repository encoding with path-augmented AST-based chunking, (ii) dual-perspective query transformation capturing both structural and behavioral signals, and (iii) two-phase agentic reranking combining symbolic inspection with evidence-grounded reasoning. Unlike prior graph-based or multi-hop agentic approaches, BLAgent performs bounded reasoning over a compact candidate set, balancing accuracy and cost. On SWE-bench Lite, BLAgent attains over 78% Top-1 accuracy with open-source models and over 86% with a closed-source model, while being over 18x cheaper than the strongest baseline using the same model. When integrated into an APR framework, it improves end-to-end repair success by over 20%.

## Full Text


<!-- PDF content starts -->

BLAgent: Agentic RAG for File-Level Bug Localization
MD AFIF AL MAMUN,University of Calgary, Canada
GIAS UDDIN,York University, Canada
Bug localization remains a key bottleneck in downstream software maintenance tasks, including root cause analysis, triage,
and automated program repair (APR), despite recent advances in large language model (LLM)-based repair systems. File-level
bug localization is especially critical in hierarchical pipelines, where errors can propagate to downstream stages such as
statement-level localization or patch generation. While Retrieval-Augmented Generation (RAG) offers a promising direction
for grounding LLMs in repository context, existing RAG pipelines rely on static retrieval and lack the reasoning needed
to identify faulty code accurately. In this work, we presentBLAgent, a novel agentic RAG framework for file-level bug
localization that integrates three key ideas: (i) code structure-aware repository encoding with path-augmented AST-based
chunking, (ii) dual-perspective query transformation capturing both structural and behavioral signals, and (iii) two-phase
agentic reranking combining symbolic inspection with evidence-grounded reasoning. Unlike prior graph-based or multi-hop
agentic approaches,BLAgentperforms bounded reasoning over a compact candidate set, balancing accuracy and cost. On
SWE-bench Lite,BLAgentattains over 78% Top-1 accuracy with open-source models and over 86% with a closed-source model,
while being over 18 ×cheaper than the strongest baseline using the same model. When integrated into an APR framework, it
improves end-to-end repair success by over 20%.
Replication Package. https://github.com/afifaniks/BLAgent
CCS Concepts:•Software and its engineering→Software maintenance tools.
ACM Reference Format:
Md Afif Al Mamun and Gias Uddin. 2026.BLAgent: Agentic RAG for File-Level Bug Localization. 1, 1 (May 2026), 45 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Bugs are inevitable in software systems. When a bug is logged, various tasks take place, such as triaging,
debugging, root cause analysis, and finally fixing. Across all these tasks, bug localization often remains the first
step [ 4,52,53]. Bug localization (BL) is the process of identifying faulty code regions within a software repository.
In large and evolving repositories, determining which files are most likely responsible for a reported issue can
require substantial effort [ 4,46]. Therefore, effective localization plays a broader role in software engineering by
reducing the search space for both human and automated maintenance workflows.
Traditional localization approaches often operate at multiple granularities, including file-level, function-level,
and statement-level localization [ 7,14]. However, recent empirical evidence reveals that file-level localization is
the most critical component in hierarchical bug localization pipelines [ 7]. The study demonstrated that removing
file-level localization from a multi-granularity localization framework caused a catastrophic 94% drop in Top-5
accuracy and a 96% reduction in Mean Average Precision (MAP) at the statement level. This finding suggests that
without accurate file-level localization, even sophisticated localization techniques may fail to identify buggy code.
Authors’ Contact Information: Md Afif Al Mamun, afif.mamun@ucalgary.ca, University of Calgary, Canada; Gias Uddin, guddin@yorku.ca,
York University, Canada.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that
copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page.
Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy
otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from
permissions@acm.org.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM XXXX-XXXX/2026/5-ART
https://doi.org/10.1145/nnnnnnn.nnnnnnn
, Vol. 1, No. 1, Article . Publication date: May 2026.arXiv:2605.17965v1  [cs.SE]  18 May 2026

2•Md Afif Al Mamun and Gias Uddin
Take SWE-bench [ 15] for an example, a widely used Automated Program Repair (APR) benchmark for large
language model (LLM)-based coding assistants. SWE-bench averages over 11,000 functions and 168,000 state-
ments—direct statement-level localization is computationally infeasible, particularly for LLM-based systems [ 7].
File-level localization, by contrast, reduces the search space by several orders of magnitude, enabling efficient
downstream localization and repair. Consequently, improving file-level accuracy offers the most impactful path
toward advancing end-to-end APR performance. Recent advances in LLMs have significantly improved APR capa-
bilities on real-world bugs [ 50,55,60]. However, these systems remain fundamentally constrained by localization
quality, as inaccurate file selection directly limits repair effectiveness [ 7,14]. While bug localization also supports
developer-centric activities such as manual debugging and root cause analysis, evaluating its effectiveness in such
settings often requires subjective or task-specific studies. We therefore adopt APR as a representative downstream
evaluation setting, where localization quality can be assessed objectively through patch correctness and issue
resolution.
diff --git a/django/db/models/fields/__init__.py
b/django/db/models/fields/__init__.py\nindex f5823d9..b422cfb
100644\n--- a/django/db/models/fields/__init__.py\n+++
b/django/db/models/fields/__init__.py\n@@ -1708,8 +1708,14 @@
class FilePathField(Field):\n         return str(value)\n \n     def
formfield(self, **kwar gs):\n+        # Resolve the path if a callable
was provided. This allows the\n+        # FilePathField to accept a
callable for ``path`` so that the\n+        # actual filesystem location
can be determined at runtime (e.g.,\n+        # based on settings on the
current machine) rather than at import\n+        # time.\n+       
resolved_path = self.path() if callable(self.path) else self.path\n       
 return super().formfield(**{\n-            'path': self.path,\n+           
'path': resolved_path,\n             'match': self.match,\n           
 'recursive': self.recursive,\n             'form_class':
forms.FilePathField,\nPatch Generation
Allow FilePathField path to accept a callable.
Description
I have a specia l case where I want to create a model containing the path to some local files on the server/dev mach ine. Seeing as the place where
these files are stored is dif ferent on dif ferent machines I have the following:
import os
from django.conf import settings
from django.db import models
class LocalFiles(models.Model):
        name = models.CharField(max_length=255)
        file = models.FilePathField(path=os.path.join(settings.LOCAL_FILE_DIR, 'example_dir'))
Now when running manage.py makemigrations it will resolve the path based on the machine it is being run on. Eg:
/home/<username>/server_files/example_dir
I had to manu ally change the migration to include the os.path.join() part to not break this when running the migration on production/other
machine.django__django-10924
Agentless Localization
.../db/models/fields/files.py
.../db/migrations/serializer .py
.../db/migrations/writer .py
Correct file not in
the list BLAgent Localization
.../db/models/fields/__init__.py
.../db/migrations/operations/fields.py
.../db/migrations/state.py
Agentless (APR) Framework
File
Localization
diff --git a/django/db/migrations/serializer .py
b/django/db/migrations/serializer .py\nindex ec5c5e5..4f95bb9
100644\n--- a/django/db/migrations/serializer .py\n+++
b/django/db/migrations/serializer .py\n@@ -320,6 +320,1 1 @@
def serializer_factory(value):\n         return
OperationSerializer(value)\n     if isinstance(value, type):\n       
 return TypeSerializer(value)\n+    # Functions (or any other
callables) should be serialized by their import path.\n+    # This
allows fields like FilePathField to accept a callable for the
`path`\n+    # ar gument, which is then correctly rendered in
migration files.\n+    if callable(value):\n+        return
TypeSerializer(value)\n     # Anything that knows how to
deconstruct itself.\n     if hasattr(value, 'deconstruct'):\n       
 return DeconstructableSerializer(value)\n
Incorr ect Corr ect
Correct file at
Top-1
Fig. 1. An example demonstrating how incorrect file localization may lead to incorrect patch generation.
Consider the motivating example illustrated in Figure 1, which demonstrates how incorrect file localization
directly leads to incorrect patch generation in an example ( django-10924 ) from the Django project in the SWE-
bench-Lite dataset. The bug report describes the need to allow FilePathField to accept a callable for its path
parameter, thereby enabling runtime resolution of file system locations. An existing APR system (Agentless [ 55])
incorrectly localizes the fault to files such as serializer.py . This misdirection causes the system to generate
a patch modifying the migration serializer to handle callable values but misses the core issue persisting in a
different file. In contrast, when the correct file __init__.py is identified through more precise localization, the
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•3
generated patch by the same APR resolves the callable path within the FilePathField.formfield() method,
addressing the root cause of the issue.
This example shows that even sophisticated repair systems fail when bug localization provides incorrect context,
highlighting the critical need for improved localization strategies that can bridge the semantic gap between
natural language bug reports and code-level implementations. While providing more candidate files might seem
to mitigate localization errors, this approach may face critical constraints. LLMs are constrained by limited
context windows, and processing more files individually incurs substantial computational overhead. Moreover,
LLMs are also sensitive to the order of the inputs, where their ability to extract relevant information deteriorates
when details appear in the middle/later of lengthy inputs [ 28,29]. For such reasons, APRs typically restrict
consideration to only the top-k(e.g., Top-3) files to balance context length and computational feasibility [ 17,55].
Retrieval-Augmented Generation (RAG) [ 22] offers a promising direction for addressing these localization
challenges. RAG enhances LLMs by grounding their predictions in external knowledge retrieved from code
repositories, thereby improving factuality and reducing hallucinations. Recent work has explored RAG for code-
related tasks such as code completion and bug localization, demonstrating that retrieval-based approaches can
effectively bridge the vocabulary gap between natural language bug descriptions and technical code artifacts
[15,47,66,69]. However, conventional RAG strategies for code often rely on holistic code-text embeddings and
naive text-based chunking, which may fail to capture the structural intricacies of code and result in suboptimal
retrieval quality [66].
Furthermore, purely embedding-based retrieval approaches, while efficient, lack the reasoning capabilities
necessary to assess the functional relevance and appropriateness of retrieved code. Recent advances in LLM-
based agents–where LLMs autonomously plan, reason, and act using external tools—present an opportunity to
enhance RAG systems with explicit code-level reasoning [ 11,13]. Agentic approaches have shown promise in
various software engineering tasks by combining iterative reasoning with tool usage, enabling more informed
decision-making compared to static retrieval methods [ 5,9,50,60]. Recent agentic localization methods often
employ graph-based repository traversal, which often incur prohibitive cost [ 9,14,50] or require extensive
LLM-finetuning [7, 9, 14].
Motivated by these limitations, we proposeBLAgent(B ug Localization using Agent ic RAG), a novel agentic
RAG framework for file-level bug localization that addresses the key limitations of typical RAG pipelines. First,
BLAgentencodes code repository using path-augmented, abstract syntax tree (AST)-aware chunking where
source files are split at syntactic boundaries (e.g., functions, classes, logical units) via AST analysis, and each chunk
is prepended with its relative file path. This preserves semantic integrity within chunks while creating a direct
alignment channel between path-like signals in bug reports (e.g., tracebacks, module references) and the code they
describe. Second, it applies dual-perspective query transformation, decomposing each bug report into a structural
query that highlights code entities such as function names, modules, and identifiers, and a behavioral query that
captures runtime symptoms and expected vs. observed behavior. Issuing both queries independently for retrieval
and merging their candidate sets ensures that the retrieval pool covers faults described by lexical reference and
by observable behavior. Third,BLAgentperforms two-phase agentic reranking: a ReAct-based reasoning agent
iteratively inspects structural skeletons of candidate files and assigns relevance scores, followed by a single-shot
evidence-anchored reranking step that expands only retriever-highlighted code regions within top-ranked files
for a final, implementation-level ranking decision. Unlike graph-based agentic approaches,BLAgentbounds
its reasoning to a compact, retrieval-filtered candidate set, avoiding the costly unbounded traversal of existing
approaches. We evaluateBLAgentthrough two research questions:
RQ1. How effective isBLAgentover baselines for bug localization?
BLAgentsignificantly outperforms both conventional RAG pipelines and state-of-the-art approaches —
including complex, graph-based methods — achieving an MRR of 0.900 and a Top-1 accuracy of 86.7% with a
, Vol. 1, No. 1, Article . Publication date: May 2026.

4•Md Afif Al Mamun and Gias Uddin
closed-source model, surpassing LocAgent [ 9] at 18 times lower API cost, and an MRR of 0.851 with a Top-1
accuracy of 78.6% using an open-source model.
RQ2. How does such a pipeline influence end-to-end Automated Program Repair (APR)?
When integrated intoAgentless[ 55], an established open-source APR framework,BLAgentimproves the
overall issue resolution rate by over 20%, demonstrating that more precise file-level localization directly translates
into measurable gains in end-to-end patch synthesis.
2 BLAgent Methodology
We introduce theBLAgentlocalization framework that precisely identifies and ranks faulty source files related
to a given bug report. Figure 2 provides an overview of the entire pipeline. The localization stage employs a
structured pipeline comprising repository encoding, query augmentation, file retrieval, and agentic reranking.
This design enables the proposed framework to systematically narrow down the search space and identify the
most relevant source files for a given bug report.
Transformed
QueriesBug Report (Query)
Transformation
Code
Chunks
Transformed
Queries
Retrieved Candidate Files and ChunksCandidate File
Retrieval
FileV iewingT ool
Ranked Files
   Ranked
  File PathsLLM
Agent
Bug
Report
Vector  DatabaseAgentic RAG Pipeline (BLAgent)Agentic Localization
Path-A ware 
Code Chunking
Code 
Repository
Fig. 2. Overall outline of the proposed localization approach.
2.1 Repository Encoding
Given a target repository, we begin with repository-level preprocessing and encoding. All source files (e.g., .py)
are extracted, while non-executable files (e.g., licenses, markdowns) are filtered out since they rarely contribute to
bug fixing. The selected source files are then segmented into smaller, semantically meaningful units (i.e., chunks)
for vector encoding and storage. Instead of relying on naive text-based chunking—where code is divided by
arbitrary character or line limits—we employ acode chunkingstrategy using CodeSplitter1. This approach
leverages the abstract syntax tree (AST) of the target language to produce chunks that align with syntactic and
semantic boundaries (e.g., functions, classes, or logical code blocks). Such segmentation aims to preserve the
structural and semantic integrity of the chunks.
Semantic Preservation.We definesemantic preservationas ensuring that no chunk cuts across an AST construct.
For example, splitting a function body mid-expression or separating a decorator from its decorated function.
Such fragmentation strips away local dependencies (e.g., parameter bindings, loop invariants, return contracts)
that the embedding model requires to produce a meaningful vector representation. As shown in Figure 3, naive
text splitting breaks the normalize() function mid-expression, merging incomplete fragments into a single
incoherent chunk, whereas AST-aware splitting retains each function as a self-contained logical unit. When
a function exceeds the maximum chunk size, the splitter recursively subdivides it at the finest available AST
1https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/code/#llama_index.core.node_parser.CodeSplitter
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•5
(a) Naive text-based chunking
1defcompute_average(nums):
2total =sum(nums)
3count =len(nums)
4returntotal / count
5
6defnormalize(nums):
7mean = compute_average(nums)
8range_val =max(nums) -min(nums)
9return[(n - mean) / range_val
10
11######SPLIT######
12
13forninnums]
14
15defscale(nums, factor):
16return[n * factorforninnums]
17. . .(b) Code-aware w/ file path chunking (AST-based)
1[PATH]src/core/utils/math.py
2[CODE]
3defcompute_average(nums):
4total =sum(nums)
5count =len(nums)
6returntotal / count
7
8######SPLIT######
9
10[PATH]src/core/utils/math.py
11[CODE]
12defnormalize(nums):
13mean = compute_average(nums)
14range_val =max(nums) -min(nums)
15return[(n - mean) / range_valfornin
nums]
16. . .
Fig. 3. Comparison of naive text-based versus AST-aware code splitting. The naive splitter (a) breaks the normalize()
function mid-expression, while the code-aware splitter (b) maintains full structural integrity.
boundaries—such as nested blocks, conditionals, or statement-level nodes—rather than at an arbitrary character
offset. Each resulting sub-chunk thus still represents a syntactically complete program fragment, preserving local
variable scopes and control-flow context within that segment.
File Path Augmentation.To further enhance contextual precision, each chunk is augmented with itsfile path
(e.g., src/utils/math.py ). This augmentation creates a direct alignment channel between bug reports and code
chunks. For example, bug reports frequently contain implicit or explicitpath-likesignals—module references
(e.g.,django.db.models.fields), traceback entries (e.g.,File "django/db/migrations/writer.py"), or package-qualified
identifiers (e.g.,sklearn.linear_model.LogisticRegressionCV)—that correspond structurally to file system paths.
Without path augmentation, these signals are present in the query embedding but absent from the chunk embed-
ding, resulting in a systematic vocabulary mismatch. Prepending the relative file path to each chunk closes this
gap: the embedding model jointly encodes content and location; therefore, path-bearing tokens in the bug report
directly reinforce similarity with the chunks they describe. This is also useful for identically named entities across
different project hierarchies (e.g., module/classA.method_x vs.module/classB.method_x ), where content-only
embeddings would otherwise assign near-identical similarity scores to structurally unrelated functions. Figure 3
illustrates the difference between naive text chunking and our code-aware, path-augmented chunking. The final
chunks are embedded using a pre-trained model to obtain dense vector representations, stored in a vector database
for efficient similarity-based retrieval. We adoptdense retrieval, as prior work demonstrates its superiority over
traditional sparse methods [37, 43].
However, we did not consider LLM-based semantic chunking, where a language model identifies topically
coherent boundaries and segments code accordingly. Recent work suggests that the advantages of semantic
chunking are highly task-dependent and frequently fail to justify the added computational cost [ 39], whereas
AST-based chunking yields self-contained, semantically coherent units that improve retrieval on code-related
tasks [ 66]. Given that our experimental setup spans the full SWE-bench-Lite benchmark—repositories averaging
, Vol. 1, No. 1, Article . Publication date: May 2026.

6•Md Afif Al Mamun and Gias Uddin
over 11,000 functions each—applying such chunking involving an LLM at indexing time would incur substantial
cost that is difficult to justify.
2.2 Query Transformation
You are an AI assistant that rewrites Python bug reports and technical descriptions into focused search queries to retrieve
source code most relevant to the root cause of the issue.
Focuson the code structure and error source. Include:
-Relevantfunctions, methods, or classes (e.g.,`__init__`,`from_pretrained`,`__repr__`)
- Module or package names (e.g.,`sklearn.linear_model`,`torch.nn.functional`)
- File or traceback context if available (e.g., error in`sklearn/utils/validation.py`)
- Probable reason (e.g., wrong type comparison, improper shape validation, missing null check)
-Avoidrestating full tracebacks or irrelevant details
- Do not include user-defined variable/class names, or specific instance details unless they are part of the core issue.
###Example
Bug:
With`sklearn.set_config(print_changed_only=True)`, printing`LogisticRegressionCV(Cs=np.array([0.1, 1]))`raises:
ValueError: The truth value of an array with more than one element is ambiguous.
Transformed:
In`sklearn.linear_model.LogisticRegressionCV`, enabling`print_changed_only=True`causes a ValueError when`Cs`is a
numpy array. The issue likely stems from a parameter comparison in`__repr__`or`__init__`using`!=`on arrays without`.
all()`or`.any()`. Suspected faulty logic in parameter diffing or config-aware repr code in`sklearn.utils.
_param_validation`or related helpers.
Outputonly the rewritten query text. No sections, headers, or markdowns. Be retrieval-friendly and concise, with no extra
text. Aim for 2-4 sentences.
Listing 1. Structural query transformation prompt (PT 0).
Directly using the raw text of a bug report for retrieval can lead to suboptimal results as lexical mismatches
between natural language descriptions and code identifiers, as well as extraneous narrative or contextual
information, can distort query embeddings and hinder retrieval of relevant code fragments [ 25,44,58]. In practice,
relevant code entities may be described either structurally (e.g., via class or method names) or behaviorally (e.g.,
through observed runtime behavior), but these aspects are not always explicitly stated in the surface text.
To address this challenge, we employ a query transformation approach inspired by retrieval-augmented
generation (RAG) methods (e.g., rewriting, decomposition, etc.) that improve document-query alignment [ 6,25,30].
The key idea is to reformulate a bug report into retrieval-oriented queries that emphasize complementary
perspectives of the fault, bridging the semantic gap between natural language descriptions and the underlying
code.
Specifically, we decompose every bug report into two complementary transformations: a structural transfor-
mation (𝑇0) and a behavioral transformation ( 𝑇1). We chose these two because bug reports typically contain both
(1) explicit or implicit references to program entities (e.g., function names, APIs, parameters), and (2) descriptions
of the erroneous behavior or execution context. The structural transformation ( 𝑇0) distills the report into its
code-related components like identifiers, modules, and relevant structural cues, making it suitable for retrieving
code elements that directly match these terms. In contrast, the behavioral transformation ( 𝑇1) reformulates the
report, capturing how the system is expected to behave versus how it actually behaves.
Structural Query Transformation (𝑇0).This transformation focuses on the static, syntactic aspects of the bug
report, extracting signals related to identifiable code entities—such as modules, functions, and helper routines.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•7
Make django.utils.text.slugify()  strip dashes and underscores
Description
         
(last modified by Elinaldo do Nascimento Monteiro)
         
Bug generation slug
Example:
from django.utils import text
text.slugify("___This is a test ---")
output: ___this-is-a-test-
Improvement after correction
from django.utils import text
text.slugify("___This is a test ---")
output: this-is-a-testdjango.utils.text.slugify implementation strip leading/trailing
hyphens and underscores; search for slugify function in
django.utils.text module, focusing on regex that replaces
non‑ alphanumeric characters and the final .strip("-_") call. Look
for handling of underscores in the pattern (e.g., r"[^\w\s-]") and
the post‑ processing step that trims hyphens/underscores from the
result. Also examine any helper functions used by slugify (e.g.,
_slugify) that may affect removal of consecutive dashes or
underscores.
`django.utils.text.slugify` returns strings with leading/trailing
hyphens/underscores (e.g., `"___This is a test ---"` →  `"___this-
is-a-test-"`) instead of stripping them, while the expected output is
`"this-is-a-test"`. Triggered by calling `slugify` on strings
containing leading/trailing underscores or dashes. Likely issue in
the slugify implementation’s post ‑ processing step that fails to trim
non‑ alphanumeric separators. Search for slugify handling of
leading/trailing characters in `django.utils.text`. Human-reported Bug
LLM(a) Structural Query Transformation  (T0)
(b) Behavioral Query Transformation (T1)
Behavioral Cue
Structural Cue
Redundant Information for RetrievalTransformed
Bug ReportPT0 (Listing 1) 
PT1 (Listing 2) 
Fig. 4. Query transformation of human-reported bug (Example Bug:django__django-12983).
We use prompt(𝑃𝑇 0)in Listing 1 to transform queries for structural augmentation. We provide one example to
the LLM to show how this can be done. In Listing 1, the example shows that when a specific configuration is set,
initializing LogisticRegressionCV(...) ends up in an error. The transformed query also adds more structural
cues like __init__ ,__repr__ since the issue happens during initialization. This is particularly beneficial for
dense retrieval where reasoning is not available and its complete dependency on the embedding similarity of
query and the documents.
Similarly, we provide one real example shown in Figure 4a, where the transformation focused on lexical
and structural components. The transformed query explicitly names the relevant function ( slugify ), module
(django.utils.text ), possible helper routine ( _slugify ), and even points to specific syntactic constructs such
as the regular expression and post-processing call ( .strip() ). This structural transformation of the original bug
guides retrieval toward locations in the repository where the implementation logic or post-processing behavior
may reside.
Behavioral Query Transformation (𝑇1).The behavioral query transformation specifically focuses on the
runtime behavior described in the bug report, capturing cues about faulty execution, unexpected outputs, or
missing functionality. As shown in Figure 4b, this transformation reframes the bug in terms of observed program
behavior and expected outcomes, allowing the retriever to better connect textual failure descriptions to the
underlying runtime logic.
For example, in the slugify case, the raw bug report describes an undesired output string produced by
the function (i.e., it fails to strip leading and trailing hyphens or underscores). The behavioral transformation
reformulates this report to explicitly contrast the observed behavior (“returns strings with trailing underscores or
dashes”) with the expected outcome (“should strip non-alphanumeric separators”). This reframing highlights the
symptom in a way that points the retriever toward code regions responsible for normalization or post-processing.
We use prompt(𝑃𝑇 1)(Listing 2) to elicit such transformations, guiding the LLM to emphasize runtime mismatches,
incorrect outputs, and failed handling of edge cases.
2.3 Candidate File Retrieval
Let𝑇0and𝑇1denote the two query transformations (structural and behavioral, respectively). For a transformed
query𝑞(𝑘)with𝑘∈0,1 , we embed the query using the same encoder applied to repository chunks. Each
repository file 𝑓is partitioned into chunks, and each chunk 𝑐𝑖,𝑗(the𝑗-th chunk of file 𝑓𝑖) has an embeddingc 𝑖,𝑗
produced by the same encoder. Similar code chunks are then retrieved from the database using nearest neighbour
, Vol. 1, No. 1, Article . Publication date: May 2026.

8•Md Afif Al Mamun and Gias Uddin
You are an AI assistant that rewrites Python bug reports and behavioral descriptions into focused search queries for
retrieving the most relevant source code responsible for the issue.
Focuson the user-observed behavior and triggering conditions. Include:
- The exact observed behavior (e.g., silent failure, incorrect output, crash)
-Expectedvs. actual behavior
- Triggers such as input parameters, environment, CLI flags, or API calls
- Probable module or component involved (if clearly inferable)
- Possible reason behind the behavior (e.g., missing fallback, incorrect state check)
-Avoidspeculative deep internals; keep it grounded in externally visible symptoms
###Example
Bug:
When training a model using`transformers.Trainer`with`fp16=True`on a 4GB GPU, training silently hangs. No traceback
or error message is shown; the training loop remains stuck after the first step.
Transformed:
`transformers.Trainer`hangs during training when`fp16=True`and VRAM is low (e.g., 4GB GPU). Likely module involved:`
accelerate`or mixed precision handling in`Trainer`.ExpectedOOM error or graceful fallback, but instead the training
loop freezes without exception, possibly due to unhandled CUDA error or silent failure in gradient scaler.
Outputonly the rewritten query text. No sections, headers, or markdowns. Be retrieval-friendly and concise, with no extra
text. Aim for 2-4 sentences.
Listing 2. Behavioral query transformation prompt (PT 1).
search [ 34]. As a code file typically contains multiple functions or logical units while a bug is most often localized
to a single unit, we aggregate chunk-level similarity scores for each file by taking the maximum over its chunks.
The file-level similarity under query transformation𝑇 𝑘is therefore
sim 𝑞(𝑘),𝑓𝑖=𝑆(𝑘)∗𝑖=max∗𝑗∈𝐶(𝑓 𝑖)𝑠(𝑘)
𝑖,𝑗,
where𝐶(𝑓 𝑖)is the index set of chunks in file 𝑓𝑖. Taking the maximum highlights the single region of a file that
best matches the query and avoids diluting strong local matches by averaging with unrelated code.
For each transformation(𝑇 𝑘)we retrieve a ranked list of files ordered by decreasing file-level similarity:
𝐹𝑇𝑘=
𝑓(𝑘)∗1,𝑓(𝑘)∗2,...,𝑓(𝑘)∗𝑛𝑘
,sim 𝑞(𝑘),𝑓(𝑘)∗𝑖≥sim 𝑞(𝑘),𝑓(𝑘)∗𝑖+1.
To form the agent’s candidate input we take the top-(m) files from each list, concatenate these two prefixes, and
remove duplicates while preserving first-occurrence order:
𝐹candidate =unique
𝐹𝑇0[:𝑚]⊕𝐹𝑇 1[:𝑚]
,
where(⊕)denotes concatenation, 𝐹𝑇𝑘[:𝑚]represents the first 𝑚files from the ranked list 𝐹𝑇𝑘, and unique(·)
eliminates repeated file entries while keeping the ordering induced by the concatenation. By construction each
file appears at most once in 𝐹candidate . We limit the number of candidates to 15. Although this concatenation order
places𝑇0candidates first by default, Section 3.5 shows that reversing the order produces negligible difference in
localization accuracy, confirming that the ordering is an implementation convenience rather than a structural
dependency.
Finally, the agent receives the candidate file set 𝐹candidate including the chunks in each candidate together with
the original problem statement for downstream reasoning and reranking.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•9
Candidate File List
(from retriever)
Input: Top-k candidate paths 
e.g., [
"astr opy/modeling/cor e.py",
"astr opy/modeling/separable.py",
"astr opy/modeling/convolution.py", ...
]Original Bug Report
Modeling's `separability_matrix` does not
compute separability correctly for nested...
The agent alternates between reasoning ("Thought")  and
tool use ("Action") , guided by file skeleton observations.
Action: R eadFileSkeleton("...separable.py")  
Observation:                                       
 - Function: separability_matrix(model)  
- Notes: Uses inputs/outputs attributes
Thought: Mentions `_separable`; may link to
CompoundModel logic 
(continues for all relevant files...)
Thought: I should inspect
`astropy/modeling/separable.py`
ReadFileSkeleton("../foo.py")
class Foo:
""documentation"
def bar(a, b):
""documentation"
...
The agent synthesizes all observations and
assigns relevance scores:
```json 
{ 
"astropy/modeling/separable.py": 10, 
"astropy/modeling/convolution.py": 8,
"astropy/modeling/core.py": 6,         
} 
 ``` 
ReAct Agent
Single Shot Inference on Top-M files
Build file context with Top-N chunks from the retriever:
```
def is_separable(): // Chunk k exists in this method
[Include full method body]
def separability_matrix(): // No chunk in this method
[Just include the method signature]
```FILE: astropy/modeling/separable.py
Final Reranked File-Paths
```json 
["astropy/modeling/separable.py", 
"astropy/modeling/core.py",
"astropy/modeling/convolution.py" ]        
 ``` 
Phase 1: Skeleton-Based Agent Scoring (SAS) Phase 2: Evidence-Anchored Reranking (EAR)(context includes all Top-M files...)
Fig. 5. Reranking of the candidate files with ReAct agent.
2.4 Agentic Reranking
We apply a two-phase agentic reranking step to the candidate file set 𝐹candidate obtained from the retrieval stage.
First, a reasoning agent inspects structural skeletons of candidate files and assigns relevance scores. Second,
a one-shot LLM inference, reranks the top-scored files using pruned, chunk-grounded file contexts to resolve
near-ties and verify implementation-level relevance.
Phase 1: Skeleton-Based Agent Scoring (SAS).We instantiate a ReAct-based [ 61] reasoning agent that
iteratively assesses the retrieved candidates. The agent operates through an explicit reasoning loop (Thought →
Action→Observation→Answer), where each iteration involves (i) forming a hypothesis about which files are
likely relevant, (ii) selectively inspecting file representations through the equipped tool ReadFileSkeleton , and
(iii) updating its internal assessment before producing a final scored output. This reasoning pattern allows the
agent to dynamically decide which files to examine more closely based on observed evidence, rather than relying
solely on static similarity scores.
To facilitate symbolic inspection, each code file is represented as a structuralskeletoncapturing its classes,
function signatures, and docstrings. Rather than processing full source code, the agent operates on these skeletal
representations to identify files most relevant to the reported bug. For file-level localization, the most informative
cues are typically structural and declarative—function names, signatures, and docstrings—rather than implemen-
tation details. This abstraction enables the agent to reason efficiently about code organization while staying
within a manageable context budget.
The agent is prompted to assign a relevance score in the range of 0–10 (not relevant to highly relevant) to
each candidate file (Listing 3), reflecting how likely it is to require modification to resolve the reported bug. By
iteratively inspecting file skeletons, it refines its confidence estimates and produces a final mapping from file
paths to scores.
Phase 2: Evidence-Anchored Reranking (EAR).Skeleton-based scoring provides an efficient, organization-
level view of each candidate file. To further refine this ordering with implementation-level evidence—particularly
when multiple candidate files implement similar interface or inherit from a common base class, making their
skeletons structurally indistinguishable at the signature level—we introduce a one-shot evidence-anchored
reranking step over the top- 𝑀Phase 1-scored files. We set 𝑀= 5to restrict EAR to a small set of highly plausible
candidates, enabling effective joint comparison while keeping the reasoning context bounded. This design choice
is further supported by our ablation study (Section 5.5), which shows diminishing returns when increasing the
number of candidates. Similarly, for each of the 𝑀files, we construct apruned file contextusing the retriever’s top
5 chunks as evidence anchors: we preserve global file structure (imports, class definitions, and method/function
, Vol. 1, No. 1, Article . Publication date: May 2026.

10•Md Afif Al Mamun and Gias Uddin
You are a powerful AI code assistant.
Bug Report:
{problem_statement}
You are provided with a list of candidate code files that may be relevant to this bug.
Yourtask:
1. Use the`ReadFileSkeleton`tool to inspect the structure of each file.
2. Assign a relevance score from 0 (not relevant) to 10 (definitely needs to be modified) to each file.
The score should reflect how likely the file is to require changes to fix the bug described.
3. YouMUSTview the file skeletons before assigning scores.
4. Construct a dictionary mapping file paths to their relevance scores.
5. YouMUSTRETURN scores for at least {num_files} files.
MUST NOTE:
- Multiple files can have the same score
- Should provide the complete file paths as they appear for the`ReadFileSkeleton`tool.
- The provided file paths are sorted by the chunk similarity scores, but AVOID making assumptions based on order.
STOP CONDITION:
- After you come to conclusion, you must immediately output the final answer in the required JSON format andSTOP.
- Do NOT write anotherThought: after receiving the finalObservationand the last thing in your output must be:
Final Answer
{{...}}
FORMAT INSTRUCTIONS:
You must respond using this ReAct format exactly:
Thought: I need to inspect the structure of fileA.py
Action: ReadFileSkeleton
Action Input: "fileA.py"
Observation: [skeleton output here]
Final Answer:
```json
{{
"fileB.py": 8,
"fileA.py": 7,
"fileC.py": 7
}}
Your candidate files:
{retrieved_file_paths}
```
Listing 3. Skeleton-Based Agent Scoring (SAS) Prompt.
signatures) while expanding only the methods that contain retrieved chunks to their full implementations; all
other methods are kept as signatures. This exposes implementation-level detail precisely where the retriever
considers it most relevant, without loading entire files. The LLM jointly compares these 𝑀pruned contexts
in a single inference pass using the prompt defined in Listing 4 and outputs the final ranked list of file paths,
improving ranking precision while controlling context length and inference cost.
This two-phase design is deliberate. Pruned-context reranking over all 15 candidates would significantly inflate
context size, increasing cost and risking mid-context attention degradation [ 28]; moreover, the ReAct agent
contributes an independent localization signal through hypothesis-driven structural inspection that no single
static inference pass can replicate. Phase 1 thus both reduces the candidate set to a tractable size and enriches the
ranking signal that Phase 2 refines.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•11
The following are the top {len(file_paths)} ranked files retrieved for a given bug report. Analyze and rerank them based on
their relevance to the problem statement. The idea is to find the actual file where a patch needs to be applied to fix the
bug.
Return a ranked list of file paths, ordered from most relevant to the least, based on their content and relevance to the
problem statement.
Example Output:
```json
{{
"ranked_files": [
"path/to/most_relevant_file.py",
"path/to/second_most_relevant_file.py",
...
"path/to/least_relevant_file.py"
]
}}
```
Do not include any explanations or additional text outside the JSON structure.
Problem Statement:
{problem_statement}
Possibly Relevant Files:
{aggregated_code_text}
Listing 4. Evidence-Anchored Reranking (EAR) Prompt.
3 Accuracy Assessment of Bug Localization in BLAgent (RQ1)
To evaluate how effectivelyBLAgentlocalizes bugs at the file level, we decompose RQ1 into the following
sub-questions:
•RQ1.1: CanBLAgentoutperform state-of-the-art baselines?
We investigate whether agentic reasoning in a RAG pipeline (i.e.,BLAgent) can improve file-level localiza-
tion beyond current state-of-the-art methods.
•RQ1.2: How do different retrieval and RAG configurations influence file-level localization accu-
racy?
We evaluate different chunking and retrieval configurations in a traditional RAG pipeline to identify the
most effective setup for file-level localization.
•RQ1.3: Can agentic reranking improve over traditional RAG?
We examine if agentic reranking can improve over traditional RAG pipelines in localization by using a
ReAct-based agent to rerank dense retrieval candidates.
•RQ1.4: Can query transformation further improve localization?
We investigate whether reformulating bug reports into structured, retrieval-oriented queries further
improves localization.
•RQ1.5: Does the choice of LLM variant affect the performance of the agent?
We analyze whether the underlying LLM influences the agent’s ability to accurately localize faults. Specifi-
cally, we examine how different LLM variants—varying in size and type (e.g., closed or open-source) impact
the overall effectiveness.
•RQ1.6: CanBLAgentbe extended to function level localization?
We assess whether the proposed architecture can be generalized to function-level localization.
, Vol. 1, No. 1, Article . Publication date: May 2026.

12•Md Afif Al Mamun and Gias Uddin
3.1 Experimental Setup
3.1.1 Localization Baselines.We compareBLAgentagainst recent dedicated bug localization frameworks and
localization components within APR systems. Among dedicated localization frameworks, LocAgent [ 9] adopts a
graph-guided LLM agent framework that represents code repositories as directed heterogeneous graphs encoding
structural elements and dependencies, enabling multi-hop reasoning to navigate and localize relevant code
entities efficiently. CoSIL [ 14] introduces an LLM-driven, function-level localization framework that performs a
two-phase code graph search — first exploring broadly at the file level through module call graphs, then refining
at the function level with pruning and reflection mechanisms to control search direction and context quality.
BugCerberus [ 7] adopts a hierarchical localization design powered by specialized LLMs operating at the file,
function, and statement levels, which progressively narrow down bug locations by leveraging intermediate
program representations and contextual cues
Among end-to-end APR frameworks, Agentless [ 55] combines both an embedding-based retrieval method
and an LLM-based approach that reasons over the project’s file tree to rank relevant files. AutoCodeRover [ 65]
is an agentic framework that utilizes multiple specialized agents alongside AST-based code search APIs and
spectrum-based fault localization to iteratively localize buggy code and propose fixes. We also compare the
localization accuracy with OpenHands [ 50], which uses its default CodeAct agent that localizes bugs by iteratively
issuing shell and code execution actions to explore the repository.
3.1.2 Evaluation Metrics.Following prior work [ 7,14,55], we employ standard information retrieval metrics to
assess localization performance.
Mean Reciprocal Rank (MRR).MRR measures the average of the reciprocal ranks of the first correctly localized
file for each bug report. Formally, let 𝑄denote the set of bug reports, and 𝑟𝑖the rank position of the first correctly
localized file for query𝑖. Then,
MRR=1
|𝑄||𝑄|∑︁
𝑖=11
𝑟𝑖.(1)
A higher MRR indicates that the model ranks the correct file closer to the top.
Top-𝑘Accuracy.Top- 𝑘Accuracy evaluates the proportion of bug reports for which at least one correct bug
location appears within the top- 𝑘ranked predictions. Let ⊮[·] denote the indicator function, which equals 1 if
the condition is true and 0 otherwise. Then,
Top-k=1
|𝑄||𝑄|∑︁
𝑖=1⊮[correct file∈Top-𝑘 𝑖].(2)
We report Top-{1,3,5,10}accuracy to evaluate the competitive methods.
3.1.3 Dataset.For our experiments, we use theSWE-bench-litedataset [ 15], which comprises 300 real-world
bug instances collected from open-source Python projects. Each entry in the dataset contains a detailed bug
report, the corresponding commit hash of the repository representing the HEAD of the repository before the
issue was fixed, ground truth patch, etc. SWE-bench-lite is specifically designed for evaluating bug localization
and program repair techniques, providing a diverse set of bugs that vary in complexity, type, and affected code
components.
3.1.4 Large Language Models.We use the GPT-OSS-120B2model from Ollama as the primary LLM for all agentic
reasoning and text-generating components in our framework and the downstream APR framework. The model
uses 128 mixture-of-experts (MoE) [ 45] and has a context length of 128,000. We particularly adopt this model
2https://openai.com/index/introducing-gpt-oss/
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•13
due to its openness and proven reasoning capability compared to other proprietary models like GPT-o4-mini, or
o3-mini on different benchmarks. Additionally, we also validate our localization results with SOTA proprietary
LLM—Claude-4.6-Sonnet (Claude-4.6).
3.1.5 Embedding Model.For text and code representation, we employ the nomic-ai/nomic-embed-text-v1
model3to generate high-quality embeddings used in retrieval and similarity-based ranking. This model was
selected for its strong empirical performance—comparable to proprietary alternatives such as OpenAI’s text-
embedding-3-small—and its large context window of 8192 tokens, enabling effective encoding of longer code
segments and detailed bug reports.
3.1.6 System Configuration.All experiments were conducted on a system equipped with 2 ×NVIDIA L40S 48GB
GPUs, an Intel Xeon Gold 6442Y×96 processor, and 250 GB RAM.
3.2 RQ1.1 CanBLAgentoutperform state-of-the-art baselines?
3.2.1 Approach.To assess the effectiveness of the proposed agentic RAG framework, we conduct a comparative
evaluation against several state-of-the-art bug localization approaches discussed in Section 3.1.1. We evaluate
BLAgentusing GPT-OSS-120B as the base LLM and nomic-embed-text-v1 as the embedding model. Rather
than reproducing all baseline results—which would incur prohibitive API costs—we report each baseline at its
best published configuration on SWE-bench-Lite. For systems primarily designed as end-to-end APR pipelines
that do not explicitly report file-level localization accuracy, we use results from published papers on the same
benchmark. AutoCodeRover results are taken from Chang et al. [ 7], and OpenHands results are taken from Chen
et al. [9].
To further disentangle the contribution of theBLAgentarchitecture from the choice of LLM, we conduct a
controlled comparison where bothBLAgentand LocAgent (the strongest Top- 𝑘baseline) are run under the same
model, Claude-4.6-Sonnet. Due to the substantial API cost of running full benchmark evaluations with proprietary
models, we restrict this controlled comparison only to these two systems. We allocate a fixed evaluation budget of
$300 for LocAgent using Claude-4.6-Sonnet to control evaluation cost. Under this constraint, LocAgent processes
182 of the 300 instances before exhausting the budget, and results are reported over this subset (see Section 5.1
for detailed cost analysis).
3.2.2 Results.Table 1 reports the overall file-level localization accuracy forBLAgent(Phase 1 + Phase 2)
compared to leading baselines on the SWE-bench-lite dataset. The results indicate a consistent advantage for
agentic RAG across all metrics.
Comparison against dedicated localization methods.BLAgent(GPT-OSS-120B) achieves MRR 0.851 with a
Top-1 accuracy of 78.6%, outperforming all baselines at their best published configurations. LocAgent (Claude-3.5)
is the closest competitor, achieving Top-1 77.7% and Top-3 91.9%, yetBLAgentsurpasses it on Top-1 and Top-3
while relying on an open-source model and without requiring any specialized fine-tuning, pre-built repository
dependency graphs, or static analysis infrastructure. These gains are particularly meaningful at lower 𝑘values,
as downstream repair systems often restrict consideration to only the top-1 or top-3 ranked files due to cost and
context window constraints [55].
Comparison against APR-based localization strategies.BLAgentconsistently outperforms the localization
components used within existing APR systems under standard ranked retrieval metrics. Agentless [ 55], combining
embedding-based retrieval and LLM reasoning, achieves Top-1 63.0% with GPT-4o. AutoCodeRover [ 65], which
performs localization through agentic code search using AST-based APIs and spectrum-based signals, achieves
55.7% Top-1 accuracy. Among APR systems, OpenHands [ 50] achieves the strongest file-level localization (76.2%
Top-1 accuracy with Claude-3.5), yet remains belowBLAgent(78.6%) using an open-source model. However,
3https://huggingface.co/nomic-ai/nomic-embed-text-v1
, Vol. 1, No. 1, Article . Publication date: May 2026.

14•Md Afif Al Mamun and Gias Uddin
Table 1. File-level localization accuracy of different methods.†indicates the approaches we reproduced. The baselines include
both dedicated localization methods (BugCerberus, CoSIL, LocAgent) and localization methods within end-to-end APR
frameworks (Agentless, AutoCodeRover, OpenHands).
Dataset Approach MRR Top-1 Top-3 Top-5 Top-10
SWE-bench-lite(a) Comparison against published baselines
BugCerberus 0.733 0.651 0.745 0.754 0.791
CoSIL (Qwen-2.5-32B) 0.701 0.613 0.780 0.837 –
LocAgent (Qwen-2.5-32B FT) – 0.759 0.905 0.927 –
LocAgent (Claude-3.5) – 0.777 0.919 0.941 –
Agentless (GPT-OSS)†0.719 0.623 0.817 0.850 0.857
Agentless (GPT-4o) 0.715 0.630 0.817 0.850 0.883
AutoCodeRover (GPT-4o) 0.650 0.557 0.698 0.741 0.754
Openhands (GPT-4o) – 0.609 0.719 0.737 –
Openhands (Claude-3.5) – 0.762 0.897 0.901 –
BLAgent(GPT-OSS)0.851 0.786 0.923 0.933 0.943
(b) Controlled comparison under the same LLM
LocAgent (Claude-4.6)†– 0.824 0.863 0.863 –
BLAgent(Claude-4.6) 0.900 0.867 0.930 0.947 0.953
we note that not all APR systems report localization under comparable evaluation protocols. In particular,
LingmaAgent [ 31] reports a file-localization accuracy of 67.7% based on a patch-level criterion—checking whether
the generated patch targets the correct file—rather than ranked retrieval metrics (Top- 𝑘, MRR). As such, this
result is not directly comparable. Nevertheless, even under a conservative interpretation treating this figure as
Top-1 accuracy, it remains below multiple baselines reported in Table 1, includingBLAgent.
Controlled comparison under the same LLM.When both systems are run under Claude-4.6-Sonnet (Claude-
4.6),BLAgentachieves an MRR of 0.900 and a Top-1 of 86.7%, Top-3 of 93.0%, and Top-5 of 94.7% across all
300 instances. The closest competitor, LocAgent (Claude-4.6), evaluated on 182 completed instances, achieves a
Top-1 of 82.4% — higher than its Claude-3.5 result (77.7%), suggesting that the stronger model improves Top-1
predictions. However, its Top-3 and Top-5 accuracy both drop to 86.3%, falling well below its own Claude-3.5
results (91.9% and 94.1%, respectively). For a fair comparison, we evaluated the localization results ofBLAgenton
the same 182 instances, and it demonstrated a 90.6% Top-1 accuracy. Furthermore, we inspected the localization
results of LocAgent and found that Claude-4.6 causes LocAgent to return fewer candidate files in most instances,
often only one or two, reflecting overconfidence in its top prediction at the expense of recall. In contrast,BLAgent
with Claude-4.6-Sonnet improves consistently across all thresholds, confirming that the architectural design —
bounded agentic reasoning over a structured candidate set — is the primary driver of its gains rather than the
underlying LLM.
Observation O1:BLAgentimproves localization performance across most baseline methods. Iterative
reasoning combined with contextual generation enables a deeper contextual understanding of bug reports.
3.2.3 Discussion.WhileBLAgentdemonstrates superior performance compared to state-of-the-art methods
across all Top-k, it is important to examine the cases where it fails to correctly localize faulty files. While using
Claude-4.6, we find that in all 14 failed cases, the correct file was not retrieved in the top 15 candidates by any of
the query transformation techniques.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•15
Failure case analysis.A systematic analysis of the 14 retrieval failures reveals three distinct patterns, summarized
in Table 2. Two cases involvefeature requests, where no existing code snippet (e.g., function) was modified by
the patch, and the fix resides in configuration or infrastructure code with no strong retrieval signal. Notably,
our baseline LocAgent explicitly excludes these same instances from its own evaluation, stating they do not
modify any existing function [ 9]. Five cases involvehidden dependencies, where retrieval correctly identifies the
primary symptom site but the fix resides in an imported utility or base class that the retrieved file depends on. The
remaining seven arevocabulary mismatchcases, where the bug report’s surface text strongly suggests one module
while the fix resides in a behaviorally connected but lexically distant one. To investigate the recall ceiling, we
extended retrieval up to Top-50 per query transformation for the 12 non-feature-request failures. The correct file
appears at ranks 18–50 in 8 instances, while the remaining 4 are absent from the top-50 of both transformations,
asserting that an embedding-level semantic gap that merely increasing 𝑘may not resolve. However, extending 𝑘
to 50 inflates the candidate pool ( 𝑇0+𝑇1) from 15 to∼63 unique files on average, incurring substantial cost and
context overflow in the agentic reranking pipeline.
Table 2. Taxonomy of the 14 retrieval failures where the correct file was absent from the Top-15 candidate pool.
Category Bug ID Description
Feature Requestdjango-11564,sympy-20590 No faulty file exists to localize — the patch adds new
logic to infrastructure or configuration code with no
semantic signal in the bug report (e.g., django-11564
adds SCRIPT_NAME support via conf/__init__.py
while the report describes template tag behavior).
Hidden Dependencydjango-14997 ,django-16400 ,
sympy-13146 ,sympy-18087 ,
sympy-21612Retrieval correctly identifies the symptom site but the
fix resides in an imported utility or base class. For exam-
ple,django-14997 retrieves sqlite3/schema.py (the
crash site) while the fix is in ddl_references.py which
it imports.
Vocabulary Mismatchdjango-11797 ,sympy-12236 ,
sympy-13031 ,sympy-13915 ,
sympy-20322 ,sympy-21379 ,
sympy-21627The bug reports strongly suggest one module while
the fix resides in a behaviorally connected but lex-
ically distant one. For example, sympy-20322 de-
scribes a ceiling inconsistency, directing retrieval to
integers.py as it has multiple ceiling and rounding
methods, while the fix is incore/mul.py.
Repository-level concentration.Ten of the 14 failures originate from the sympy repository, representing a
13.0% repository failure rate (10 out of 77 instances), compared to only 3.5% for django (4 out of 114 instances).
This concentration reflects a systematic mismatch between how sympy bug reports are written and how its
implementation is organized. In 9 of 10 failed sympy cases,BLAgentretrieves at least one file from the same
directory as the true patch, indicating that the relevant neighborhood is usually identified correctly. Bug reports
describe the user-visible mathematical symptom, which consistently points retrieval toward the observable
behavior rather than the underlying implementation site. For example, in sympy-21612 the report describes a
“LaTeX parsing” error, so retrieval surfaces parsing/latex/ andprinting/latex.py ; yet none of the dominant
report terms appear in the actual patch file printing/str.py . Insympy-18087 , the termstrigsimp,cos, andsqrt
are absent from the patch file core/exprtools.py , making it invisible to embedding-based retrieval. This is
compounded by sympy ’s mathematical naming conventions: files such as exprtools.py andoperations.py
implement low-level algebraic primitives that are silently invoked by the higher-level operations users interact
, Vol. 1, No. 1, Article . Publication date: May 2026.

16•Md Afif Al Mamun and Gias Uddin
with and are described mostly in abstract algebraic terms that bear no lexical resemblance to user-reported
symptoms. By contrast, django ’s domain-descriptive identifiers align naturally with the terms users write in bug
reports, giving the retriever a strong discriminative signal.
These observations highlight two critical insights: first, dense retrieval remains a crucial component of the
pipeline, as it determines the initial pool of files available for reranking; second, improving recall in the retrieval
stage may directly translate into improved localization accuracy.
Observation O2:The overall effectiveness of the agentic RAG pipeline is bounded by the recall of its dense
retriever. An agent’s reasoning and reranking are only as effective as the candidate pool it receives—if the
correct file is missing (e.g., beyond Top-15), subsequent localization becomes unattainable regardless of
reasoning capability.
3.3 RQ1.2 How do different retrieval and RAG configurations influence file-level localization
accuracy?
Chunking .py files
VectorDB
LLM
RerankerRanked
File Paths
Bug ReportRetrieve
Code ChunksBug Report
+
Code Chunks
Retrieval QueryCode 
Repository
Fig. 6. Basic RAG pipeline for file-level localization.
3.3.1 Approach.We design and evaluate RAG pipelines using different chunking strategy under two ranking
settings: (1) Direct Dense Retrieval (Query →Retrieval), where candidate files are ranked purely by embedding
similarity with no LLM involvement, and (2) LLM Reranking (Basic RAG: Query →Retrieval→Generation),
where the top retrieved chunks are passed as context to an LLM that reranks the candidate files based on their
relevance to the bug report. We extract all source code files (e.g., .pyfiles) from each repository and apply three
distinct chunking strategies:
•Text-based chunking, which splits files into fixed-size textual segments without code structure awareness;
•Code-aware chunking, which preserves syntactic boundaries (e.g., functions and classes) to retain structural
coherence; and
•Code-aware with file path context, which further embeds each chunk with its file path to provide hierarchical
grounding during retrieval (discussed in Section 2).
These chunks are then encoded into dense vector representations and stored in a vector database (chromadb4) for
efficient retrieval. Next, we assess the retrieval accuracy of each chunking strategy independently to measure how
effectively relevant code files can be identified given a bug report. Finally, we integrate the retrieval component
into a full RAG pipeline (Figure 6), where the retrieved code chunks serve as contextual input to an LLM. We
4https://www.trychroma.com/
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•17
provide Top-15 files to the LLM for ranking to keep the context length reasonable. The prompt used for the RAG
pipeline is provided in Listing 5. This allows us to find the suitable configuration (e.g., chunking strategy, whether
LLM integration is useful or not) that provides the best result.
You are an expert software debugging assistant.
Identify which retrieved files are most likely to contain the bug described below.
Bug Report:
{bug_report}
Code Context:
{Code Path 1}
{Code Chunk 1}
--
{Code Path 2}
{Code Chunk 2}
...
Instructions:
1. Carefully analyze the bug report and snippets.
2. Respond ONLY with a JSON array of file paths, ordered from most to least relevant.
3. If there is no path provided, return an empty JSON array.
4. DO NOT include any explanations or additional text.
5. Return 10 ranked file paths from the context.
6. DO NOT include paths that were not provided.
Example:
["src/core/utils.py", "src/main/model.py", "src/data/loader.py"]
Listing 5. LLM Prompt for the RAG pipeline.
Table 3. Localization accuracy of different retrieval configurations on SWE-bench-lite. The best setup is marked with★.
Chunking File Ranking MRR Top-1 Top-3 Top-5 Top-10
TextDense Retrieval 0.459 0.350 0.526 0.600 0.686
+LLM Reranking 0.690 0.643 0.740 0.746 0.750
Code-AwareDense Retrieval 0.473 0.363 0.530 0.593 0.703
+LLM Reranking 0.722 0.667 0.783 0.793 0.800
Code-Aware w/ File Path★Dense Retrieval 0.553 0.417 0.627 0.7130.873
+LLM Reranking 0.734 0.673 0.796 0.8200.823
3.3.2 Results.Table 3 presents the localization performance across different dense retrieval and RAG configura-
tions on the SWE-bench-lite dataset. The results reveal several important observations.
First,direct dense retrievalmethods show limited capability in accurately ranking faulty files in top posi-
tions. When standard text-based chunking is applied—using conventional text-splitters designed for natural
language—the performance is notably low. This confirms that code structure and semantics are poorly preserved
under text-oriented segmentation. In contrast,code-aware chunkingslightly improves the MRR and Top-1 accu-
racy, demonstrating that structural cues such as function boundaries and class definitions contribute to more
meaningful embeddings and, consequently, more accurate retrieval. However, we observed that when we embed
, Vol. 1, No. 1, Article . Publication date: May 2026.

18•Md Afif Al Mamun and Gias Uddin
file paths (e.g., module/src/file.py) along with each chunks, the retrieval performance increases significantly
(16.9% MRR improvement compared to straightforward Code-Aware Chunking).
Integrating these retrieval configurations into a RAG framework (i.e., when we use an LLM to rerank the dense
retrieval candidates) leads to even greater gains, with MRR improving by up to 53% across chunking strategies.
However, the best results are achieved with path-embedded, code-aware chunking. This suggests that while
dense retrieval alone may struggle to surface the faulty file in the top positions (e.g., Top-1 or Top-3), ensuring
that the correct file appears within the Top-10 or Top-15 candidates allows the LLM to leverage its contextual
reasoning capabilities to rerank the files more effectively. However, for larger values of k (e.g., Top-10), direct
dense retrieval with code-aware chunking and file path augmentation outperforms LLM reranking across all
chunking strategies
Observation O3:AST-based code-aware chunking outperforms standard text-based chunking for source
code. Notably, incorporating relative file paths into the chunks further enhances dense retrieval performance,
yielding up to a 20.4% improvement over traditional text chunking.
Observation O4:Dense retrieval alone is not sufficient for precise file-level localization. With a complete
RAG pipeline, localization accuracy improves substantially—yielding up to a 53% increase in MRR over
standalone dense retrieval.
3.3.3 Discussion.While LLM reranking in the RAG pipeline substantially improves ranking accuracy compared
to direct dense retrieval, the effectiveness of the underlying dense retrieval remains a critical factor. It determines
whether the relevant file appears within the candidate list at all—since an LLM or agent can only rerank files that
have already been retrieved. We observe that path-augmented code-aware chunking yields the most effective
retrieval results. For instance, this direct retrieval technique achieves a Top-10 accuracy of 87.3%, compared to
only 70.3% for standard code-aware chunking—a 24% improvement. This suggests that incorporating hierarchical
path information helps embeddings better capture repository structure and contextual dependencies, improving
semantic recall at the file level that is consistent with the design rationale in Section 2.1.
JSONField values are displayed as dict when readonly in the
admin.
For example, {"foo": "bar"} would be displayed as {'foo':
'bar'}, which is not valid JSON.
I belie ve the fix would be to add a special case in
django.contrib.admin.utils. display_for_field  to call the
prepare_value of the JSONField (not calling json.dumps
directly to take care of the InvalidJSONInput case).
Chunks[PATH] django/contrib/admin/utils .py
[CODE]
def display_for_field (value, field, empty_value_display):
    ...Path-Augmented Code Chunk
High Similarity Score
def display_for_field (value, field, empty_value_display):
    ...Naive Code Chunk
Low Similarity Score
Retrieverdjango__django-12308
Problem Statement
...
 File ".../python3.7/site-
packages/ mpl_toolkits/axes_grid1/axes_grid.py ", line 618, in
__init__
    self.set_label_mode(label_mode)
  File ".../python3.7/site-
packages/ mpl_toolkits/axes_grid1/axes_grid.py ", line 389, in
set_label_mode
    _tick_only(ax, bottom_on=False, left_on=False)
 ...matplotlib__matplotlib-26020
Problem Statement[PATH] lib/ mpl_toolkits/axes_grid1/axes_grid.py
[CODE]
def _tick_only(ax, bottom_on, left_on) :
    bottom_of f = not bottom_on
...Path-Augmented Code Chunk
High Similarity Score
def _tick_only(ax, bottom_on, left_on) :
    bottom_of f = not bottom_on
...Naive Code Chunk
Low Similarity Score
Case 1: Implicit Path Match
Case 2: Explicit Traceback Match
Fig. 7. Two cases illustrating how path-augmented code chunking improves retrieval similarity.
Impact of Path Augmentation.We further analyze how path augmentation improves localization by examining
cases where path-augmented chunks correctly surface the faulty file in the top-10 while naive code-aware chunks
fail. We identify two prevalent patterns by which bug reports carry path-like signals that augmented chunks
can exploit:implicit path matches, where package-qualified identifiers in the bug report structurally mirror the
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•19
repository path hierarchy, andexplicit traceback matches, where stack trace entries directly reference the file
path. In both cases, naive chunks—containing only the function body—lack these signals and rank substantially
lower, while path-augmented chunks align with query tokens and achieve higher similarity scores (Figure 7).
Case 1 (Implicit Path Match).Indjango__django-12308, the bug report referencesdjango.contrib.admin.utils.
Although not a verbatim file path, the dot-separated package hierarchy structurally mirrors the repository path
django/contrib/admin/utils.py. A path-augmented chunk prepending this path aligns with these tokens in the
query embedding, yielding a high similarity score and surfacing the correct file. The naive chunk, containing
only the function body, lacks this signal and ranks substantially lower.
Case 2 (Explicit Traceback Match).Inmatplotlib__matplotlib-26020, the bug report contains a direct stack
trace entry referencingmpl_toolkits/axes_grid1/axes_grid.py. The path-augmented chunk prepended with the
relative path of the code matches these tokens directly, producing a high similarity score. The naive chunk,
containing only the_tick_onlyfunction body, misses this signal entirely and ranks lower.
RAG vs. Direct Retrieval.While the traditional RAG pipeline performs well in Top-{1 −3} accuracy, its
performance declines for larger 𝑘. Specifically, for code-aware chunking with file paths, LLM-based reranking
achieves a Top-10 accuracy of 82.3%, lower than the 87.3% achieved by direct dense retrieval alone (Table 3). We
find that the LLM often returns fewer than ten ranked files despite explicit prompt instructions to return 10 files.
This behavior likely arises from the model’s tendency to prioritize high-confidence predictions and truncate
low-confidence candidates to conserve context length, leading to incomplete Top- 𝑘outputs. Overall, while a
basic RAG pipeline may significantly improve Top-1 or 3 positions, it may not be as useful for later positions.
Observation O5:Although RAG pipelines enhance retrieval accuracy, the effectiveness of dense retrieval
still remains a key determinant of overall performance.
3.4 RQ1.3 Can agentic reranking improve over traditional RAG?
3.4.1 Approach.In this RQ, we explore whether adding an agent-based reranker can improve over the traditional
RAG pipeline. To do this, we adopt the best chunking method ( ★) identified in 3.3 and compare three setups under
the same configuration–(1) a baseline setup without any reranking, where the dense retriever’s initial ranking is
used directly; (2) a traditional RAG pipeline that applies LLM-based contextual reasoning over the top-ranked
files to select the most relevant one; (3)BLAgentwith agentic scoring only (Phase 1), which uses a ReAct agent
to iteratively inspect file skeletons and assign relevance scores; and (4) the fullBLAgentpipeline (Phase 1 +
Phase 2), which further applies evidence-anchored reranking.
Table 4. Impact of agentic reranking compared to traditional RAG pipeline.
Method MRR Top-1 Top-3 Top-5 Top-10
No Reranking 0.553 0.417 0.627 0.713 0.873
RAG (LLM Reranking) 0.734 0.673 0.796 0.820 0.823
BLAgent(Phase 1. SAS) 0.769 0.685 0.839 0.886 0.896
BLAgent(+Phase 2. EAR)0.819 0.762 0.889 0.889 0.899
3.4.2 Results.Table 4 summarizes the comparative results. The traditional RAG pipeline already improves
substantially over direct dense retrieval (MRR: 0.734 vs. 0.553), confirming that contextual reasoning over
retrieved code snippets enhances relevance estimation. However, introducing agentic reranking yields further
, Vol. 1, No. 1, Article . Publication date: May 2026.

20•Md Afif Al Mamun and Gias Uddin
improvements across all metrics—raising MRR by 11.5%. This gain indicates that the agentic reranking can better
identify and promote truly relevant files that would otherwise remain buried in the candidate list.
3.4.3 Discussion.The results demonstrate that each phase of the reranking pipeline contributes meaningfully
and for distinct reasons. Basic RAG with LLM reranking already captures substantial gains through contextual
reasoning over retrieved candidates. Skeleton-based Agentic Scoring (Phase 1) extends this through iterative,
hypothesis-driven skeleton inspection, but produces marginal Top-1 gains (+1.8% over Basic RAG) when structural
signatures alone are insufficient to distinguish files that implement similar interfaces or inherit from a common
base class. Evidence-anchored reranking (Phase 2) further improves accuracy by constructing pruned file contexts
that expand only the method bodies the retriever considers most relevant, grounding the final ranking in
implementation-level evidence. The substantial Top-1 improvement from Phase 2 (+11.2% over Phase 1) confirms
that richer, targeted context yields better ranking decisions. Together, the two phases deliver a +11.5% MRR
gain over Basic RAG, with dense retrieval establishing the candidate pool, agentic scoring narrowing it through
structural reasoning, and evidence-anchored reranking refining the final ordering with implementation-level
precision.
Observation O6:Reranking significantly improves localization accuracy over dense retrieval. Agentic
reranking further enhances performance through iterative reasoning and structural inspection, while the
additional generation-based consolidation stage enables holistic comparison under an expanded but controlled
context. Together, these stages yield more consistent Top- 𝑘gains and more precise identification of faulty
files.
3.5 RQ1.4 Can query transformation further improve localization effectiveness?
3.5.1 Approach.In Section 3.3, we discussed that the overall performance of RAG pipelines heavily depends
on the quality of dense retrieval. While chunking strategies play a crucial role in preserving code semantics,
another key factor influencing retrieval effectiveness is the formulation of the input query used to search the
vector database. We investigate whether systematic transformations of bug reports can lead to more effective
retrieval. To this end, we experiment with two distinct query transformation strategies (see Section 2.2). We
apply both transformations on a bug report and then use the transformed reports (i.e., query) as input queries for
dense retrieval. The retrieved candidates are then passed to theBLAgentpipeline to evaluate their impact on
localization accuracy across both setups.
Table 5. Impact of query transformation in different settings.
Method Transformation Type MRR Top-1 Top-3 Top-5 Top-10
Dense RetrievalNone (Base Bug Report) 0.553 0.417 0.627 0.713 0.873
Structural (𝑇 0) 0.680 0.550 0.773 0.860 0.920
Behavioral (𝑇 1) 0.618 0.480 0.720 0.797 0.887
BLAgent(Phase 1)None (Base Bug Report) 0.769 0.685 0.839 0.886 0.896
Structural (𝑇 0) 0.785 0.696 0.856 0.903 0.930
Behavioral (𝑇 1) 0.783 0.696 0.860 0.900 0.910
Both(𝑇 0⊕𝑇1)0.795 0.710 0.860 0.903 0.943
3.5.2 Results.Table 5 summarizes the impact of query transformation on localization accuracy across both
dense retrieval andBLAgent. In the dense retrieval setup, both transformation strategies substantially improve
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•21
performance over using the base bug report as query (i.e., No Transformation), indicating that enriching the
input with syntactic or behavioral cues enhances embedding alignment with relevant code fragments. Among
the two, the syntactic transformation ( 𝑇0) proves more effective (22.9% MRR improvement compared to using
base bug report as query) during direct dense retrieval, suggesting that structural elements—such as module and
function identifiers—provide stronger retrieval signals than behavioral descriptions alone.
When these transformations are used withinBLAgent, smaller but consistent improvement is realized. Once
a relevant file is retrieved within the candidate pool (e.g., Top-15), the agent can typically recover it through
reasoning, regardless of its initial rank. Nevertheless, combining both transformations ( 𝑇0+𝑇1) enables the agent to
reason over candidates retrieved by complementary signals resulting in a consistent gain. These transformations
are particularly valuable when using smaller-context LLMs or when faster reasoning is desired by reducing the
number of candidates passed to the model. In such settings, the number of candidates passed to the model is
reduced, making it more dependent on the quality of the initial query to ensure that relevant files are retrieved.
Table 6. Sensitivity of BLAgent to candidate ordering retrieved by different query transformations.
Transformation Order MRR Top-1 Top-3 Top-5 Top-10
Structural First (𝑇 0⊕𝑇1) 0.795 0.710 0.860 0.903 0.943
Behavioral First (𝑇 1⊕𝑇0) 0.797 0.716 0.866 0.903 0.930
Sensitivity to Candidate Ordering.To assess whether the concatenation order of 𝑇0and𝑇1introduces positional
bias when combining candidates from both transformations in the agentic scoring stage, we evaluate two orderings:
(1)𝑇0-first (default), where candidates retrieved by the structural query are prepended before behavioral-query
candidates, and (2) 𝑇1-first, where this order is reversed. Table 6 shows that both orderings produce nearly
identical results across all metrics, with MRR differing by only 0.2% (0.795 vs. 0.797) and Top-1 by 0.8% (0.710
vs. 0.716). This robustness follows from two properties of the pipeline. First, the candidate pool is constructed
by appending only unique files from the second transformation that are not already retrieved by the first —
so both orderings yield largely overlapping candidate sets that differ only in which files occupy the marginal
positions. Second, the ReAct agent selectively decides which files to inspect based on its evolving reasoning
state, rather than processing the list sequentially. Together, these properties ensure that the 𝑇0-first default is an
implementation convenience rather than a structural dependency of the pipeline.
Observation O7:Query transformations, especially syntactic ( 𝑇0), significantly boost dense retrieval (22.9%
MRR gain). Using both transformations to select candidates for agentic reranking in theBLAgentpipeline
further improves performance across all Top-𝑘ranks.
3.5.3 Discussion.We further analyze how different transformations impact dense retrieval by examining the
overlap and exclusivity of cases where the correct file is ranked highly. Figure 8 illustrates the complementary
nature of the three approaches: the actual bug report, structural query transformation ( 𝑇0), and behavioral query
transformation (𝑇 1).
In the Top-1 scenario, both query transformations identify numerous faulty files that the base bug report alone
fails to find. Specifically, 𝑇0and𝑇1uniquely contribute to 29 and 12 Top-1 localizations, respectively, that were not
captured by the base report. Additionally, 73 instances are covered by at least one transformed query but missed
by the base, demonstrating that transformations are not merely overlapping but in many cases essential for recall.
When the window is extended to Top-10, while most of the cases are covered by at least one transformation, there
are still 28 cases combined where only query-transformed inputs succeed. However, behavioral transformation
becomes less effective on such ranks.
, Vol. 1, No. 1, Article . Publication date: May 2026.

22•Md Afif Al Mamun and Gias Uddin
1229
3210
913
91Top-1
1228
284
88
168Top-3
211
152
56
244Top-10
Actual Bug Report Query Transformation (T0)Query Transformation (T1)
Fig. 8. File-level localization in dense retrieval when the correct file appears in the Top-1,3,10 locations.
Complementarity.Across 300 instances, structural queries ( 𝑇0) retrieve the correct file within Top-10 in 92.0%
of cases and behavioral queries ( 𝑇1) in 88.7%, but theirunionachieves 94.3%—a gain neither type reaches alone. In
17 cases, only 𝑇0retrieves the correct file; in 7 cases, only 𝑇1does. This improvement results from a systematic
divergence in retrieval behavior: 𝑇0anchors the candidate set to implementation files by matching on identifiers,
traceback tokens, and module references, while 𝑇1broadens it toward observable entry points by aligning with
symptom descriptions and runtime behavior. Furthermore, 57.7% of instances exhibit high complementarity, with
at least two unique files in each query’s Top-5, providing the downstream agent with a broader and more diverse
candidate pool. Figure 9 illustrates this complementarity with two representative cases.
Color mapping fails with boolean data
...
File ~/code/seaborn/seaborn/_core/scales.py:351,
in ContinuousBase._setup(self, data, prop, axis)
    349 vmin, vmax = axis.convert_units((vmin,
vmax))
    350 a = forward(vmin)
--> 351 b = forward(vmax) - forward(vmin)
...seaborn._core.scales ContinuousBase._setup boolean
subtraction TypeError; handling of `color` ar gument with
bool array in Plot._setup_scales;
 conversion ... or categorical scale
in seaborn/_core/scales.py
Seaborn Plot raises a `T ypeError` when a boolean list is
passed to the `color` ar gument (e.g., `Plot(["a","b"], [1,2],
color=[T rue, False]).add(Bar())`). Expected: boolean values
should be treated as categorical colors or produce a clear
error ...seaborn/_cor e/scales.py ✓
seaborn/_core/properties.py
seaborn/_core/plot.py
seaborn/categorical.py
seaborn/rcmod.py
seaborn/palettes.pyActual Bug Report
seaborn-3190T0
T1Transformed Queries Files Retrieved
LaTeX printer inconsistent with pretty printer
The LaT eX printer should always give the same
output as the pretty printer
...
In [10]: latex(exp(-x)*log(x))
Out[10]: '\\frac{1}{e^{x}} \\log{\\left (x \\right )}
...sympy .printing.latex latex() inconsistent with
sympy .printing.pretty .pprint output for expressions like
exp(-x)*log(x) and 1/(x + y)/2; 
...
Likely issue in sympy .printing.latex._print_Mul or
_print_Pow 
...
LaTeX printer produces dif ferent formatting than 
the pretty printer for the same SymPy expression
...
Expected identical output unless LaT eX can improve it.
Triggered by calling `latex()` and `pprint()` on expressions
with assumptions or rational factors.
....../benchmarks/bench_solvers.py
sympy/printing/mathml.py
sympy/physics/quantum/qexpr .py
sympy/printing/pretty/pretty .py
sympy/printing/latex.py ✓
sympy/physics/vector/printing.py
sympy-1 1897T0
T1
(a)
(b)
Fig. 9. Example of different query transformations and retrieved files.
When Structural Queries Outperform Behavioral Ones.Structural queries are more effective when the bug
report contains an explicit class, method, or traceback references that uniquely identify the implementation site.
Inseaborn-3190(Figure 9 (a), patch:seaborn/_core/scales.py), the structural query encodes the precise traceback
location (ContinuousBase._setup) and the specific file path, retrieving the correct file at Rank 1. The behavioral
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•23
query, despite naming the correct module, disperses similarity mass across color-handling files (e.g.,categorical.py,
palettes.py) due to symptom-level tokens, pushingscales.pyoutside the Top-10. The structural query wins
because low-frequency, high-specificity tokens from the traceback uniquely ground the embedding toward the
implementation module.
When Behavioral Queries Outperform Structural Ones.Behavioral queries work best in cases where
structural cues are weak or misleading—specifically, when the bug manifests as a runtime output mismatch
rather than a named API failure. Insympy-11897(Figure 9 (b), patch:sympy/printing/latex.py), the structural
query targets_print_Muland_print_Pow, focusing on the mathematical equations provided in the bug report
and retrieves files responsible for such operations (e.g.,bench_solvers.py) which are actually unrelated to this
issue. The behavioral query, describing the observed output discrepancy between latex() andpprint() for the
same expression, retrieves the correct file at Rank 2 alongside closely related printing modules. The behavioral
framing succeeds because it aligns with the module’s functional responsibility (output formatting) rather than its
internal method names.
Observation O8:Query transformations provide complementary retrieval signals, substantially enhancing
recall over the base bug report. Many faulty files are uniquely retrieved by transformed queries (29 by 𝑇0, 12
by𝑇 1in Top-1 out of 300), demonstrating that transformations can be useful for dense retrieval.
3.6 RQ1.5 Does the choice of LLM variant affect the performance of the agent?
3.6.1 Approach.To evaluate the robustness ofBLAgentacross different LLM backbones, we integrate the
agentic framework with multiple large language models spanning open-source and closed-source families:
Qwen3-32B,GPT-OSS-120B, andClaude-4.6-Sonnet. All models operate on the same query-transformed retrieval
process and reasoning protocol with the same temperature (0.7), allowing us to isolate the effect of model scale
and architecture on localization performance. We assess file-level accuracy on the SWE-bench-Lite dataset and
compare metrics such as MRR and Top-k accuracy to determine whether model scale and type—open-source
versus closed-source—provide meaningful gains in the context of agentic reasoning.
Table 7. File-level localization accuracy ofBLAgentwhen equipped with different LLM backbones.
Dataset LLM #Parameters MRR Top-1 Top-3 Top-5 Top-10
SWE-bench-liteQwen3 (Phase 1. SAS) 32B 0.789 0.706 0.863 0.900 0.936
Qwen3 (+Phase 2. EAR) 32B 0.847 0.797 0.887 0.903 0.936
GPT-OSS (Phase 1) 120B 0.794 0.710 0.860 0.903 0.943
GPT-OSS (+Phase 2) 120B 0.851 0.786 0.923 0.933 0.943
Claude-4.6 (Phase 1) – 0.889 0.843 0.933 0.947 0.953
Claude-4.6 (+Phase 2) – 0.900 0.867 0.930 0.947 0.953
3.6.2 Results.Table 7 reports localization performance across all three LLM backbones. Among open-source
models, despite GPT-OSS-120B being nearly3 .8×larger than Qwen3-32B, Phase 1 MRR differs by only 0.6% (0.794
vs. 0.789), and Top- 𝑘accuracies are nearly identical across all thresholds. This indicates that model scale provides
only marginal gains within the agentic reasoning framework when operating at the skeleton-inspection level.
The addition of Phase 2 (evidence-anchored reranking) yields consistent improvements across all three models.
Notably, Qwen3-32B benefits most from Phase 2, with MRR improving by 7.3% and Top-1 by 12.8% (0.706 →0.797).
GPT-OSS-120B shows a similar pattern with a 7.1% MRR gain. However, the performance for both models in both
, Vol. 1, No. 1, Article . Publication date: May 2026.

24•Md Afif Al Mamun and Gias Uddin
stages remains almost identical. In contrast, Claude-4.6-Sonnet already achieves strong Phase 1 performance
(MRR 0.889, Top-1 0.843), and Phase 2 provides a marginal improvement. This model-dependent pattern suggests
that larger, more capable models produce better-separated relevance scores in Phase 1, reducing the marginal
contribution of implementation-level evidence in Phase 2, while smaller open-source models rely more heavily
on Phase 2 to resolve ambiguous rankings.
3.6.3 Discussion.These findings reveal a consistent two-tier pattern across model families. For open-source
models, agentic reasoning and retrieval coordination are the dominant performance drivers — the LLM’s role
is primarily to follow structured reasoning protocols over retrieved skeletal evidence rather than to draw on
parametric knowledge, which explains why Qwen3-32B and GPT-OSS-120B perform near-identically in Phase 1
despite a3.8×size difference. Phase 2 then contributes meaningfully for both, confirming that implementation-
level context bridges the gap that skeleton-level scoring leaves open.
For closed-source frontier models, Claude-4.6-Sonnet demonstrates that stronger reasoning capability translates
into more precise skeleton-based scoring, achieving Top-1 84.3% in Phase 1 alone — substantially above both
open-source models. Phase 2 still improves performance, but the gain is smaller implying that well-separated
Phase 1 scores leave less room for consolidation to refine. Practically, this presents a clear deployment trade-off:
open-source models with Phase 2 achieve competitive accuracy at lower cost, while frontier models with Phase 1
alone may suffice for a cost-effective solution.
Observation O9:BLAgentmaintains high and consistent performance across both open-source and
proprietary LLM backbones. For open-source models, medium-sized models achieve near-equivalent accuracy
to larger models, demonstrating that reasoning and retrieval coordination, rather than model scale, are the
primary determinants of localization success. For frontier models, Phase 1 alone achieves strong performance,
with Phase 2 providing additional but diminishing improvements.
3.7 RQ1.6 Can BLAgent be extended to function level localization?
3.7.1 Approach.WhileBLAgentis primarily designed around file-level localization motivated by the require-
ments of LLM-based APR pipelines, its core components, such as query transformation, AST-aware retrieval,
and agentic reranking, are not inherently file-level specific. To evaluate whether the architecture generalizes,
we extendBLAgentto function-level localization with asingle modification: we replace the final consolidated
reranking prompt (Listing 4) in Phase 2 (EAR) of our agentic reranking stage (Section 2.4) to instruct the LLM
to identify relevant Class::method entities rather than files (Listing 6), keeping all upstream components un-
changed. We report Top- 𝑘accuracy following prior work [ 7,14], where a prediction is counted as correct if any
predicted method name appears within the method part of any ground-truthfile::methodstring in the top-𝑘
predictions. To support a same-model comparison with the strongest baseline, we additionally evaluateBLAgent
under Claude-4.6 and compare with LocAgent under the same LLM, mirroring the controlled comparison setup
used in Section 3.2.
3.7.2 Results.BLAgentgeneralizes effectively to function-level localization with a single prompt change,
achieving state-of-the-art accuracy under both open-source and frontier LLM settings (Table 8). With the open-
source GPT-OSS model,BLAgentis already competitive with the best prior system, LocAgent (Claude-3.5).
Under the same-model comparison,BLAgent(Claude-4.6) establishes a new state of the art across all Top- 𝑘
thresholds, with the largest margin at Top-1, where it improves over LocAgent (Claude-4.6) by 88%. Interestingly,
LocAgent (Claude-4.6) remains competitive at Top-5 and Top-10 but suffers a sharp Top-1 drop relative to its
Claude-3.5 configuration, suggesting that the stronger model surfaces the correct function within its candidate
list but ranks it lower—a calibration issue rather than a recall loss.BLAgent, in contrast, improves consistently
across all thresholds when moving to Claude-4.6, similar to the trend observed at the file level (Section 3.2).
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•25
The following are the top {len(file_paths)} ranked files retrieved for a given bug report.
Your job is to analyze the python files and method to rerank them based on their relevance to the problem statement.
The idea is to find the actual python method where a patch needs to be applied to fix the bug.
You should return a ranked list of file paths and Class::method names, ordered from most relevant to least relevant,
based on their content and relevance to the problem statement. Think and analyze properly to return the Class::function/
method name.
If there is no class, just return the function name. If there is no function, just return the class name.
If there is both, return in Class::method format.
Do not return more than 3 methods per file. You should return at least 10 Class/method names across all files.
Example Output:
```json
{{
"ranked_files": [
{{"path/to/most_relevant_file.py": ["Class::most_relevant_method", "second_most_relevant_method"]}},
{{"path/to/second_most_relevant_file.py": ["second_most_relevant_method"]}},
...
{{"path/to/least_relevant_file.py": ["least_relevant_method"]}}
]
}}
```
Do not include any explanations or additional text outside the JSON structure.
Problem Statement:
{entry['problem_statement']}
Possibly Relevant Files:
{aggregated_code_text}
"""
Listing 6. Prompt for function-level localization in BLAgent.
Importantly, this result is achieved through the single-prompt modification described in Section 3.7—all
upstream pipeline components remain unchanged—meaning the function-level extension introduces minimal
architectural overhead beyond the file-level pipeline. The EAR consolidation step averages only ≈15,300 tokens
per instance in a single inference pass, similar to the cost reported for file-level localization inBLAgent(Section
5.1).BLAgent’s ability to use an open-source model without extensive fine-tuning further removes dependence
on proprietary APIs entirely, making the function-level extension practical for deployment at scale.
Observation O10:BLAgentachieves state-of-the-art function-level localization with a single prompt change
and at a fraction of the cost of purpose-built systems, demonstrating that its architecture generalizes beyond
file-level without structural modification.
3.7.3 Discussion.The strong function-level result is best understood as a direct consequence of BLAgent’s
file-level precision rather than the prompt modification alone. BLAgent retrieves the patch file within its Top-5
candidates for 90.7% of all instances, and once the correct file is present, the problem of identifying relevant
methods becomes a tightly constrained search. When the candidate file set is accurate and compact, function-
level—and by extension statement-level—prediction tasks benefit from a dramatically reduced search space, lower
context pressure on the LLM, and a cleaner signal-to-noise ratio in the retrieved evidence. The further gains
from GPT-OSS to Claude-4.6 (35.4% at Top-1) reinforce that, given a high-precision candidate set, function-level
accuracy scales primarily with the underlying LLM’s reasoning capability rather than with additional architectural
complexity.
, Vol. 1, No. 1, Article . Publication date: May 2026.

26•Md Afif Al Mamun and Gias Uddin
Table 8. Function-level localization accuracy ofBLAgentcompared to other approaches on SWE-bench lite.†indicates the
approaches we reproduced.
Method Top-1 Top-5 Top-10
Agentless (GPT-4o) 0.427 0.671 0.700
BugCerbarus 0.406 0.569 0.624
CoSIL (Qwen2.5-32B-FT) 0.430 0.580 –
Openhands (Claude-3.5) – 0.682 0.700
LocAgent (Claude-3.5) 0.554 0.784 0.832
LocAgent (Claude-4.6)†0.386 0.765 0.841
BLAgent(GPT-OSS) 0.536 0.781 0.810
BLAgent(Claude-4.6)0.726 0.861 0.872
This points to a natural integration opportunity. Rather than treating file-level and function-level localization
as competing tasks, a cascaded design—in which BLAgent’s ranked files serve as input to a graph-guided function-
level agent such as LocAgent or Agentless—would allow each component to operate in its strongest regime:
BLAgent contributes high-precision, low-cost file discovery, while the downstream agent focuses its traversal
within a pre-filtered, high-confidence candidate set. Such a cascade could outperform either system in isolation
while remaining cost-efficient, and is a promising direction for future hierarchical repair pipelines.
Observation O11:Precise file-level localization directly enables fine-grained localization by tightly con-
straining the function search space.
4 Impact Assessment of BLAgent Bug Location on Automated Bug Repair (RQ2)
To better understand howBLAgentlocalization influences downstream repair effectiveness, we break down RQ2
research question into the following sub-RQs:
•RQ2.1: Does localizing withBLAgentimprove program repair?
We investigate whether replacing the baseline localization module withBLAgentleads to a higher propor-
tion of correct repairs in Agentless, thereby assessing the causal impact of improved localization on overall
APR effectiveness.
•RQ2.2: At which stage does the repair process fail even with correct localization?
We decompose the end-to-end APR pipeline to identify where failures arise—across different levels of
localization (e.g., file, function, line) and patch generation—and analyze why these issues occur despite
accurately identifying the faulty file.
4.1 Experimental Setup
4.1.1 Agentless APR Framework.We adopt Agentless [ 55] as our downstream APR baseline due to several reasons.
First, it is widely used both in the industry (e.g., OpenAI) and academia [ 7,9,14] to create and compare bug repair
patches. Second, it is highly modular with a clear seperation between localization and repair components. Third,
Agentless has demonstrated competitive performance on the SWE-bench dataset, making it a representative
baseline for modern LLM-based APR systems. While Agentless itself is non-agentic, we use it purely as a
downstream APR framework to evaluate the impact of BLAgent’s enhanced file-level localization. Since the
original framework does not natively support Ollama-based LLMs or the embedding models used in our setup,
we extend it to interface with the Ollama runtime and the HuggingFace embedding library. All other components
of the Agentless pipeline, including patch generation, validation, and evaluation, remain unchanged to ensure
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•27
comparability. We use the same GPT-OSS-120B model as the base LLM and nomic-embed-text-v1 as the embedding
model (similar to Section 3.1.1).
Patch
GenerationPatch
GenerationLocalized
Files 
by BLAgentPatch
Validation
APR Framework
   Generated
   Patches
Fig. 10. Integration of BLAgent into another APR framework.
For this evaluation, we first execute Agentless using its native localization module as a baseline. Then, we
replace Agentless’ native file-localization approach withBLAgent. Given a bug report and a target repository, our
framework produces a ranked list of candidate files, assigning each file a relevance score via agentic reranking.
The top-3 files (matching Agentless’s default) are passed to the Agentless repair pipeline. From this stage onward,
the workflow follows the standard Agentless protocol: line-level localization is performed within the selected
files using targeted LLM prompting, after which 40 candidate patches are generated using an LLM through high-
temperature of 0.8 (default configuration). The final patch is then selected via majority voting, with regression
and reproduction tests employed to validate the correctness of the repair.
4.1.2 Evaluation Metrics.Following the evaluation protocol of SWE-bench [ 15], we assess the effectiveness of
program repair using theResolved Rate. This metric represents the proportion of bug reports for which the
generated patches successfully pass all the validation tests. Let 𝑁resolved denote the number of resolved bugs and
𝑁totalthe total number of bugs. Then,
Resolved Rate=𝑁resolved
𝑁total×100%.(3)
A higher resolved rate indicates a stronger end-to-end repair capability of the system.
4.2 RQ2.1 Does localizing withBLAgentimprove program repair?
4.2.1 Approach.We integrateBLAgentinto the Agentless framework [ 55] and evaluate it on the SWE-bench-Lite
dataset [ 15]. Similar to the original work, we conduct experiments under three configurations: (i)Majority Voting,
which generates 40 patch candidates and selects the final one via majority voting; (ii)+Regression Test, where
available regression tests are included; and (iii)+Reproduction Test, which further generates and incorporates
reproduction tests to select a generated patch.
Table 9. Resolution rate on SWE-bench-Lite dataset with different methods (Best result is reported for both methods across
3 runs).
APR Patch Selection Localization Method #Instances #Resolved Rate
AgentlessMajority VotingNative 300 83 27.6%
BLAgent300 102 (↑22.8%) 34.0%
+Regression TestNative 300 86 28.6%
BLAgent300 108 (↑25.0%) 36.0%
+Reproduction TestNative 300 96 32.0%
BLAgent300 115 (↑19.7%) 38.3%
, Vol. 1, No. 1, Article . Publication date: May 2026.

28•Md Afif Al Mamun and Gias Uddin
4.2.2 Results.Table 9 summarizes the results. IntegratingBLAgentconsistently improves the end-to-end repair
rate across all settings. We run the framework with both localization approaches 3 times and report the best
result for both method in Table 9. Under the majority voting setup, our approach resolves up to 102 bugs versus
83 by the baseline (comparable to the 79 originally reported byAgentlesswith GPT-4o), representing a relative
improvement of over 20%. When regression and reproduction tests are included, a similar performance gap
persists.
A closer examination of the generated patches reveals another important benefit. When using GPT-OSS:120B as
the base LLM, the baseline framework produced 38 empty patches—instances where no valid patch was ultimately
selected—compared to 14 when we used theBLAgentas the file-level localization method. Empty patches often
indicate that the LLM received insufficient or irrelevant context, either due to inaccurate localization or because
the generated patch could not be parsed as a valid fix. The reduction in such cases suggests that the proposed
method provides more semantically grounded input, allowing the LLM to reason more effectively about the
underlying fault and produce syntactically valid, contextually coherent patches.
We note that a subset of issues (e.g., 23 issues from matplotlib ) could not execute regression or reproduction
tests due to the reliance of Agentless on an obsolete SWE-bench 2.1 interface, which is now outdated. Attempts
to migrate to the latest version revealed compatibility issues requiring nontrivial engineering effort, which we
leave for future work. Therefore, the true repair potential is likely higher than reported, as these missing cases
could result in additional successful patches.
AlthoughBLAgentachieves a Top-1 accuracy close to 80%, the corresponding end-to-end resolution rate
remains around 38%. This disparity highlights a key insight: accurate file-level localization, while necessary, is
not sufficient for successful repair. Downstream steps—particularly line-level localization and semantic patch
generation in the APR remain major limiting factors.
24
55
31
86Agentless: Resolved Bugs Across Runs
67
168
206
73BLAgent: Resolved Bugs Across Runs
35 5 101
Agentless BLAgentTotal Bug Fixed
Run 1 Run 2 Run 3
Fig. 11. Overlap of repaired issues across multiple runs using different localization strategies.
Observation O12:IntegratingBLAgentintoAgentlessconsistently improves end-to-end repair rates
(over 20% gain) and reduces empty patches. While precise file localization substantially facilitates successful
repair, ultimate patch correctness also depends on subsequent stages within an APR.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•29
Observation O13:Integrating regression and reproduction tests for patch selection consistently enhances
repair performance, independent of the localization method, by guiding the APR system toward correct and
verifiable patches.
4.2.3 Discussion.WhileBLAgentconsistently improves repair performance, we next investigate the stability of
these improvements. Figure 11 compares repair overlaps across multiple runs using both the native andBLAgent
localizations. We observe some divergence between runs, with each resolving partially disjoint sets of issues.
This instability can largely be attributed to the stochasticity inherent in high-temperature decoding during
both patch and test generation, suggesting that overall resolved counts alone may obscure deeper behavioral
patterns of APR systems. To quantify whether the observed improvements are statistically meaningful, we apply
a paired Wilcoxon signed-rank test over per-bug solve frequencies across the three runs, confirming that the
observed improvement is statistically significant ( 𝑝=0.014) with𝑝<0.05. Across all 3 runs, using BLAgent as
the localization module leads to 136 unique total bug fixes (45.3%) in the SWE-bench-Lite dataset, where it solves
35 unique bugs, while the native localization method only solved 106 (35.3%).
Table 10. Unique repairs per method and frequency with which the opposite method failed to localize the correct file. Values
represent (incorrect localizations on the opposing method / total unique repairs on a method).
Loc Method Run 1 (Incorrect (I) / Total (T)) Run 2 (I / T) Run 3 (I / T)
BLAgent Loc. (vs. Agentless Loc.) 12 / 34 (35.3%) 11 / 24 (45.8%) 17 / 34 (50.0%)
Agentless Loc. (vs. BLAgent Loc.) 0 / 22 (0.0%) 1 / 18 (5.6%) 1 / 15 (6.7%)
Furthermore, we compare the results from multiple runs of the Agentless APR framework—once using its
native localization method and once usingBLAgentfor file-level localization. Specifically, we focus on the
uniquely repairedissues, cases successfully fixed when using one localization method but not the other. For
these issues, we examine whether the alternative method failed to correctly identify the faulty file. Table 10
summarizes this comparison by showing how often the other method failed to localize the correct file for these
uniquely repaired cases. When our approach is used during localization, and it exclusively repairs an issue,
the baseline localization method (Agentless) often fails to correctly localize the faulty file (e.g., 50% of such
cases in Run 3). In contrast, for issues uniquely repaired by Agentless when using its own localization method,
BLAgentlocalized the correct file in nearly all instances. This pattern shows that unique repairs resulting from
the BLAgent’s localization frequently arise from scenarios where the baseline’s localization was insufficient,
suggesting that improved file-level grounding directly enables successful patch synthesis. Conversely, the few
cases where Agentless succeeded despite our correct localization indicate failures in later APR stages—such as
patch generation—rather than in file-level localization itself.
Overall, these findings highlight that accurate localization is not a peripheral factor but a foundational
determinant of downstream repair success. While improved localization does not guarantee a correct patch, it
substantially increases the probability of success by ensuring the LLM operates on the correct fault context.
When the faulty file is missing from the retrieved candidates, even advanced reasoning cannot compensate.
Observation O14:Repair outcomes in the APR framework exhibit some stochastic variation across runs,
resolving partially disjoint issue sets. This instability shows the sensitivity of LLM-based APRs to sampling
randomness in patch and test generation.
, Vol. 1, No. 1, Article . Publication date: May 2026.

30•Md Afif Al Mamun and Gias Uddin
Observation O15:Most unique repairs obtained withBLAgentoccur when the baseline localization method
fails to localize the correct file, demonstrating a direct causal link between accurate localization and successful
repair. This finding emphasizes that localization quality is a decisive factor in end-to-end APR effectiveness,
not just a secondary component.
4.3 RQ2.2 At which stage does the repair process fail even with correct localization?
4.3.1 Approach.To better understand where and how the program repair framework fails despite high file-level
localization accuracy, we conduct a detailed error analysis across Agentless’s hierarchical pipeline. Specifically,
we aim to identify whether unresolved cases stem from failures in localization (at file, function, or line level) or
from the subsequent patch generation step. For this experiment, we analyze the artifacts from Run 2 (Section 4.2),
whereBLAgentwas used as the localization method and the lowest number of bugs (102) were resolved across
the three runs. Focusing on the worst-performing run provides a conservative view of the failure modes, ensuring
that the identified bottlenecks are not artifacts of a particularly favorable run.
We begin by selecting 177 unresolved bug instances—cases where Agentless failed to produce a correct
patch—and systematically trace their progression through each stage of the framework. Using Agentless’s default
configuration, the model considers the top- 𝑘ranked outputs from each previous stage as context for the next
(with𝑘=3in our experiments). For example, during function-level localization, the model only explores functions
from the top-3 retrieved files; for line- or statement-level localization, it only considers the top-3 functions, and
so on. This hierarchical dependency makes error propagation particularly important to diagnose.
For each stage, we compare model predictions with ground-truth locations extracted from the actual developer
patches. At the file level, localization is considered correct if the ground-truth file appears among the top-3
retrieved candidates. The same criterion is applied to function-level localization. However, evaluating line-level
accuracy is more nuanced, since small semantic differences can lead to valid fixes at slightly different line numbers
or equivalent statements. To address this, we employ a lightweight heuristic approximation to estimate line-level
localization accuracy.
Line-level Approximation.The approximation algorithm compares the first changed statement in the model-
generated patch with that in the ground-truth patch. It proceeds as follows:
(1)Extract the first code line (excluding comment and docstrings) that was added or removed in both patches.
(2)If both edits occur within the same file and their line numbers differ by at most 5 lines, line-level localization
is considered correct.
(3)Otherwise, compute the normalized sequence similarity between the two modified lines. We straightfor-
wardedly calculate the similarity usingSequenceMatcher5.
(4)If the edits belong to the same file, exhibit textual similarity above a threshold (similarity ≥0.6), and occur
within a small positional window (e.g., within 5–10 lines), we treat the result as an approximate match
at the line level. The threshold of 0.6 was determined by manually inspecting a small sample of cases,
confirming that it reliably captures genuine near-miss localizations.
This heuristic allows us to capture cases where the framework localizes the correct logical region but introduces
the modification slightly above or below the true location—an issue common in real repair settings. While this
approximation does not provide an exact measure of line-level accuracy, it offers a pragmatic way to distinguish
between near-miss localizations and genuine mislocalizations. Similar strategies have been adopted in prior
studies to approximate line-level localization accuracy [55].
4.3.2 Results.As shown in Figure 12a, approximately one-third of all instances (34%) were successfully resolved
by the system. However, a majority of the cases remain unresolved, including a smaller fraction (7.3%) of empty
5https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•31
102
(34.0%)
21
(7.0%)177
(59.0%)(a) Overall Repair Outcomes
Resolved
Empty Patches
Unresolved19
(10.7%)
79
(44.6%)47
(26.6%)32
(18.1%)(b) Reason of Failure at Different Stages
File-Level Loc Failures
Function-Level Loc Failures
Line-Level Loc Failures
Patch Failures
Fig. 12. Overall resolution and failure percentage at different levels of program repair stage.
patches (i.e., no patch was selected at the end), suggesting that the APR framework often fails to produce a
syntactically and logically valid modification. The unresolved cases (almost 60%) where a patch was generated
but could not fix the bug motivate a deeper look into the repair pipeline to understand where failures originate.
Out of the 177 unresolved cases, we find that 159 issues were localized correctly at the file-level (Figure 12b) with
BLAgent, however, failed in a later stage in the program repair process.
Failures Across Localization Stages.We decompose unresolved cases into four categories in Figure 12(b),
reflecting the specific stages at which the program repair pipeline fails. Most of these issues occurred in one of
the stages of the hierarchical localization pipeline.
File-level localization failuresconstitute 10.7% of the unresolved cases, indicating that a subset of bug reports
still leads the retrieval module toward incorrect source files. These failures predominantly occur when the
correct file does not appear among the Top-3 retrieved candidates. While expanding the retrieval depth could
mitigate some of these errors, it would also increase the computational overhead and complexity of the Agentless
framework, creating a trade-off between coverage and efficiency.
Expected Format:
sympy/core/expr.py
function: Expr.__eq__
Actual LLM Output:
sympy/core/expr.py: Expr.__eq__
sympy/core/sympify.py: sympify
sympy/core/sympify.py: _sympify
sympy/core/sympify.py: converter
Listing 7. Example of function-level localization failure due to parsing issue.
In our observation, the most frequent failures occur at thefunction level localization(44.6%), highlighting
persistent challenges in accurately ranking the correct function among the Top-3 candidates even when the
faulty file itself is correctly retrieved. A notable source of these errors arises from formatting inconsistencies in
LLM outputs that prevent the APR framework from correctly parsing the generated function names. For instance,
insympy__sympy-18057 , although the correct function name appeared in the LLM’s response, the framework
failed to interpret it due to a structural mismatch, leading to no valid function being selected (see Listing 7).
, Vol. 1, No. 1, Article . Publication date: May 2026.

32•Md Afif Al Mamun and Gias Uddin
Localization Log:
2025-10-17 00:58:42,983 - INFO - Creating Ollama API request
2025-10-17 00:59:56,302 - INFO - Ollama API response:```
src/flask/cli.py
line: 1021
line: 1022
...
line: 1034
```
Ground Truth Patch:
--- a/src/flask/cli.py
+++ b/src/flask/cli.py
@@ -9,7 +9,7 @@
...
@@ -989,49 +989,62 @@ def shell_command() -> None:
@click.option(
"--sort",
"-s",
...
Listing 8. Example of line-level localization failure.
The second most common failures occur at theline level localization(26.6%), where the model successfully
identifies the faulty function but struggles to locate the precise statement or edit region within it. These near-
miss cases indicate that while the model captures the correct semantic region, it lacks sufficient fine-grained
reasoning about program behavior and fix placement. The underlying causes of such line-level failures can be
diverse—ranging from intrinsic LLM limitations in reasoning about code execution to occasional hallucinations
[12] or incomplete code understanding when the LLM outputs completely unrelated line numbers. In our
investigation, we also observed cases such as pallets__flask-5063 , where the ground-truth patch modifies
multiple methods, but the LLM focused only on a single one (lines 1021–1034), resulting in a partial and ultimately
incorrect localization (see Listing 8).
Finally,patch-generation failures(18.1%) occur when localization is successful at all steps, but the code
synthesis stage either produces invalid or semantically incorrect edits. This points to the limitations of current
generation models in maintaining compilation consistency and applying contextually coherent changes, even
with accurate localization cues. Figure 13 provides a concrete example from sympy__sympy-22005 . Here, the
localization is accurate down to the line level, and the generated patch targets the same location as the ground-
truth patch. However, the generated patch (Figure 13a) adds a conditional check to restrict univariate polynomials
to systems with a single generator. While syntactically correct, it does not fully encode the intended semantics.
In contrast, the ground-truth patch (Figure 13b) raises a NotImplementedError for underdetermined systems,
enforcing the zero-dimensional constraint and preserving correct program behavior.
Overall, our evaluation indicates that while end-to-end APR failures arise at multiple stages, localization
constitutes a major bottleneck. File-level, function-level, and line-level localization collectively account for the
majority of unresolved cases. Notably, in our analysis, approximately 82% of unresolved issues could be traced to
failures at some stage of hierarchical localization. These findings highlights that precise bug-context localization
is a necessary prerequisite for effective LLM-based program repair, and that improvements in both fine-grained
localization and code synthesis are essential to substantially increase repair success.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•33
(a) Genrated Patch
1diff --git a/sympy/solvers/polysys.py b/sympy/
solvers/polysys.py
2index 66efb1a..8ab9101 100644
3--- a/sympy/solvers/polysys.py
4+++ b/sympy/solvers/polysys.py
5@@ -240,7 +240,9 @@ def solve_generic(polys, opt):
6univariate = list(filter(_is_univariate, basis
))
7- if len(univariate) == 1:
8+ # Only accept a single univariate polynomial
when the system has exactly one generator.
9+ # This prevents treating underdetermined (
infinite-solution) systems as zero-
dimensional.
10+ if len(univariate) == 1 and len(gens) == 1:
11f = univariate.pop()(b) Ground-truth Patch
1diff --git a/sympy/solvers/polysys.py b/sympy/
solvers/polysys.py
2--- a/sympy/solvers/polysys.py
3+++ b/sympy/solvers/polysys.py
4@@ -240,6 +240,12 @@ def _solve_reduced_system(
system, gens, entry=False):
5univariate = list(filter(_is_univariate, basis
))
6
7+ if len(basis) < len(gens):
8+ raise NotImplementedError(filldedent('''
only zero-dimensional systems supported (
finite number of solutions)'''))
9+
10if len(univariate) == 1:
Fig. 13. Comparison of (a) APR-generated incorrect patch, and (b) Ground-truth patch forsympy__sympy-22005.
Observation O16:Failures in hierarchical localization are the main source of unresolved APR cases, with
function-level (44.6%) and line-level (26.6%) errors being the most common. Accurate identification of the
faulty file, function, and line is therefore critical to enable successful patch generation.
Observation O17:Parsing errors in LLM outputs block correct function selection, propagating failures
downstream and limiting APR effectiveness.
Observation O18:Even when localization is correct, 18.1% of failures arise from patch-generation errors,
where the model produces invalid or semantically incorrect edits. This shows that while precise localization is
necessary, the success of APR also depends on the LLM’s ability to reason about program logic and generate
contextually correct patches.
4.3.3 Discussion.Given that most unresolved cases stem from function- or line-level localization errors, we
investigate whether these failures are purely localization-related and if providing correct statement-level context
could lead to successful repair.
We analyze the case ofdjango__django-11133, where file-level localization was correct but line-level local-
ization failed in one of the experimental runs. During the failed run, the APR system incorrectly targeted line
309 (Figure 14a) instead of the correct region near line 229 (Figure 14b). As a result, the generated patch was
syntactically valid but semantically irrelevant, and the repair failed.
To test whether precise localization alone could resolve the issue, we re-executed the repair process by replacing
the model’s predicted line number with the correct one. With accurate line-level context, the APR successfully
generated the correct patch (Figure 15), demonstrating that the previous failure was not due to deficiencies in
patch synthesis but rather to mislocalized edit positions. This case study highlights a key insight: deviations in
statement- or line-level localization can derail the repair process entirely.
, Vol. 1, No. 1, Article . Publication date: May 2026.

34•Md Afif Al Mamun and Gias Uddin
(a) Patch resulted from inaccurate localization
1--- a/django/http/response.py
2+++ b/django/http/response.py
3@@ -309,7 +309,8 @@ class HttpResponse(
HttpResponseBase):
4@content.setter
5def content(self, value):(b) Ground-truth Patch
1--- a/django/http/response.py
2+++ b/django/http/response.py
3@@ -229,7 +229,7 @@ def make_bytes(self, value):
4# Handle string types -- we can't rely on
force_bytes here...
Fig. 14. Example of failed line level localization (django__django-11133).
1diff --git a/django/http/response.py b/django/http/response.py
2index 5126111..fa61f0b 100644
3--- a/django/http/response.py
4+++ b/django/http/response.py
5@@ -226,14 +226,18 @@ class HttpResponseBase:
6# [SOME COMMENTS ARE TRUNCATED TO SAVE SPACE]
7+ if isinstance(value, (bytes, bytearray, memoryview)):
8+ return bytes(value)
9+
10- if isinstance(value, bytes):
11- return bytes(value)
12if isinstance(value, str):
13return bytes(value.encode(self.charset))
14- # Handle non-string types.
15+
16+ # Handle non-string, non-bytes-like types.
17return str(value).encode(self.charset)
18
19# These methods partially implement the file-like object interface.
Fig. 15. Generated patch with correct line level information.
Observation O19:Providing the correct statement-level context allows the APR system to generate a correct
patch, demonstrating that accurate localization is a critical enabler of effective reasoning and patch synthesis
in LLM-based APR pipelines.
5 Impact of Key Design Decisions in BLAgent
5.1 Cost Analysis
Table 11 reports the average token consumption and estimated API cost per bug instance in USD for BLAgent across
two model backends. For the Claude-4.6-Sonnet model, we calculate the cost using their official documentation6,
and for the GPT-OSS-120B model, we estimate the cost using a third-party provider7. With GPT-OSS-120B, the full
agentic reranking pipeline consumes 24,523 prompt tokens and 1,939 completion tokens per bug, corresponding
to an estimated cost of $0.0017 per instance. With Claude-4.6-Sonnet, the total consumption is 26,180 prompt
6https://platform.claude.com/docs/en/about-claude/pricing (Input $3/M Tokens; Output $15/M Tokens)
7https://novita.ai/models/model-detail/openai-gpt-oss-120b (Input $0.05/M Tokens; Output $0.25/M Tokens)
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•35
tokens and 582 completion tokens, at an estimated $0.09 per instance. Across all 300 SWE-bench-Lite instances,
the total localization cost amounts to less than $1 with GPT-OSS-120B and $27 with Claude-4.6-Sonnet.
Table 11. Per instance cost analysis of BLAgent using different LLMs.
Model Reranking Phase Prompt Tokens Completion Tokens Cost ($)
GPT-OSSPhase 1 8,550 1,866 0.0009
Phase 2 15,973 72 0.0008
Total24,523 1,9390.0017
Claude-4.6Phase 1 6,453 483 0.03
Phase 2 19,727 99 0.06
Total26,180 5820.09
Comparison with LocAgent.To contextualize BLAgent’s cost efficiency, we ran LocAgent under identical
conditions using Claude-4.6-Sonnet with a fixed budget of $300 for the 300-instances in the SWE-bench-Lite
benchmark. LocAgent exhausted the full $300 budget after processing only 182 instances, producing valid results
for 158 of them. The remaining instances either timed out or exceeded 800,000 tokens/minute per instance due to
unbounded graph traversal loops before the budget was depleted. Among the 158 completed instances, LocAgent
consumed an average of 282,720 prompt tokens and 2,634 completion tokens per bug, at an average cost of $0.89
per instance. However, this figure understates the true cost, as failed instances consumed disproportionately
more tokens—bringing the observed cost across all 182 attempted instances to $1.65 per instance ($300 ÷182). In
contrast, BLAgent completed all 300 instances at $0.09 per instance ($27 total), representing a cost reduction of
more than 18×relative to LocAgent’s observed rate.
Observation O20:BLAgent’s bounded agentic reasoning is substantially more cost-efficient than complex
graph-traversal-based approaches: BLAgent completes all 300 benchmark instances at $0.09 per bug with
Claude-4.6, while LocAgent’s effective cost is $1.65 per instance.
5.2 ReAct Agents for Iterative Reasoning
While the ReAct agent employed in this study demonstrated clear improvements in bug localization compared
to traditional RAG pipelines, its architecture introduces several inherent limitations. ReAct operates through
a strictly sequential reasoning loop (Thought →Action→Observation→Answer/Repeat), which constrains
both scalability and responsiveness. Each reasoning step appends additional traces to the prompt, causing rapid
context growth over multiple iterations. This iterative accumulation increases inference latency and memory
overhead. For instance, the traditional RAG pipeline required approximately 10–25 seconds per issue on our local
GPT-OSS-120B setup with Ollama, whereas the agentic RAG pipeline often required 15–40 seconds—particularly
when the agent performed more reasoning steps before reaching a final decision. However, such issues were not
noticed when using the Claude-4.6 model due to its fast inference.
To mitigate this overhead, our proposed pipeline incorporates two strategies. First, the agent operates on a
limited candidate set (e.g., Top-15 retrieved files) to reduce unnecessary reasoning scope. Second, it is explicitly
instructed to analyze detailed file contents (i.e., code skeletons) only when deemed necessary by its reasoning
policy. Nonetheless, the ReAct architecture remains inherently sequential and lacks parallelism. Consequently, the
agent’s decision-making becomes tightly coupled with the linear order of its reasoning: an incorrect hypothesis
formed in an early reasoning step—for example, prematurely concluding that a structurally similar file is the
fault location—can bias subsequent file inspections and persist uncorrected through the final scoring output.
, Vol. 1, No. 1, Article . Publication date: May 2026.

36•Md Afif Al Mamun and Gias Uddin
This strict sequentiality, while advantageous for interpretability and modular reasoning, imposes trade-offs in
scenarios requiring long candidate pool exploration or frequent context switching.
Observation O21:The sequential nature of ReAct reasoning constrains scalability and responsiveness,
causing early errors to propagate and increasing inference latency and memory usage.
5.3 Code Skeletons for Efficient Context Management
To support effective agentic reasoning, we employed abstract code skeletons—structural representations contain-
ing only class and function signatures—instead of full source code. Prior work has shown that using code skeletons
improves localization accuracy in LLM-based repair systems [ 55]. In our setting, however, this design choice is
primarily motivated by the context sensitivity of ReAct agents, whose internal state grows cumulatively across
reasoning turns. Specifically, each iteration in a ReAct agent’s cycle adds reasoning traces, action descriptions,
and intermediate results to the conversation history. This cumulative expansion rapidly depletes the available
context window and escalates computational cost, making ReAct agents expensive when many reasoning steps
are required or the input is large.
During our experiments, we observed that limiting the agent’s view to the code skeleton substantially improved
both reasoning efficiency and overall stability. When restricted to skeletons, the agent typically converged within
five reasoning steps, as it was guided to reach a conclusion as soon as sufficient evidence was gathered. This
behavior eliminates the need to sequentially rank all related files, allowing the agent to focus only on those
it considers most relevant. In contrast, when provided with full source code files—often spanning thousands
of tokens—the agent’s prompt grew rapidly, causing significant latency (sometimes exceeding ten minutes per
action) and frequent failures due to context overflow.
Consequently, supplying structural representations rather than full implementations mitigates prompt satura-
tion, reduces token redundancy, and allows deeper inspection of candidate files within a fixed computational
budget. These observations indicate that for ReAct-style agentic bug localization systems, the use of code skeletons
is not merely an optimization but a functional necessity for maintaining both efficiency and reliability.
Observation O22:Using code skeletons instead of full source files is essential for ReAct agents, reducing
context growth, improving reasoning efficiency, and ensuring stable localization outcomes.
5.4 Query Transformation for Retrieval in Agentic Pipeline
Bug localization often suffers from a severe lexical gap between bug reports and source code [ 36]. While neural
retrieval models (e.g., transformer-based encoders [ 40]) partially mitigate this issue through semantic embeddings,
they still struggle to capture the structural and behavioral cues essential for accurate localization. To address
this,BLAgentreformulates each bug report into two complementary queries. Thestructuralquery extracts
code-centric identifiers, while thebehavioralquery abstracts the described symptoms—an approach shown to
improve retrieval effectiveness [42] (Observation O7, O8).
The candidates retrieved by these queries provideBLAgentwith complementary, multi -perspective evidence
for file -level localization. Each query generates its own ranked list of candidate files, allowing the agent to
cross -validate results based on strucutal reasoning (i.e., viewing the code structure). Prior studies confirm that
exploiting structural identifiers like class and method names improves retrieval precision in bug localization
[41], and that bug reports that clearly articulate observed and expected behavior provides richer localization
cues [ 3]. However, while using multiple transformed queries is beneficial, expanding the candidate pool with
too many queries or a very large candidate pool can result in increased latency, agent reasoning overhead, and
token budget in the ReAct loop (Observations O21, O22). Hence, to balance coverage and efficiency, we limit
the transformations to two complementary queries, ensuring that we capture the most important cues without
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•37
overloading the agent. We then merge only the unique top-ranked files, limiting the candidate pool to 15 files
(see Section 2.4), and the agent subsequently ranks the top-10 files among this pool.
Observation O23:Query transformations provide complementary retrieval signals and improves dense
retrieval performance, however, limiting the number of transformations and the candidate pool is crucial for
efficient and stable agentic reasoning.
5.5 Effect of Context Size on EAR Performance
To validate the design choices underlying Phase 2, we ablate two key hyperparameters: the number of candidate
files passed to EAR (#Files) and the number of retriever-highlighted chunks used to construct each file’s pruned
context (#Chunks). Table 12 reports Top-1, Top-3, and prompt token counts across four configurations. Increasing
the number of files from 5 to 10 raises token usage by over 86% without Top-1 improvement, suggesting that
lower-ranked candidates carry little additional signal and only inflate the context size. Similarly, an increasing
number of chunks from 5 to 10 adds 26–27% token overhead with no noticeable accuracy gain, indicating that
the retriever already surfaces the most relevant method bodies within the top-5 chunks and further expansion
recovers no new evidence. Based on these insights, we find the Top-5 files and chunks optimal for Phase 2
reranking.
Table 12. Impact of number of files and chunks in Evidence-Anchored Reranking.
#Files #Chunks Top-1 Top-3 #Prompt Tokens
5 5 0.786 0.923 15,973
5 10 0.780 0.923 20,264
10 5 0.773 0.910 29,804
10 10 0.780 0.930 37,884
Overall, these results highlight that accurate file localization with a RAG pipeline depends less on how much
context is provided and more on how well that context is selected, a principle reflected across both reranking
phases and the ablation results.
Observation O24:Increasing context size raises cost without improving accuracy, indicating that effective
localization in EAR relies on narrowing the search space rather than expanding context.
6 Recommendations
Based on the observations(O ★)from our study of agentic RAG for bug localization and its impact on program
repair, we provide targeted recommendations(R ★)organized by three overarching themes: (1) Retrieval Quality
and Pipeline Architecture, (2) Hierarchical Localization and Fine-Grained Reasoning, and (3) Practical Utility.
Each recommendation is grounded in specific findings and paired with actionable implementation strategies.
6.1 Theme 1. Retrieval Quality and Pipeline Architecture
Recommendation (R1).Dense retrieval recall fundamentally bounds a RAG or agentic RAG pipeline—when
the correct file is absent from the candidates provided to the pipeline, no subsequent reasoning can compensate.
Instead of scaling to larger models, practitioners may focus on improving embedding quality and retrieval
mechanisms. Path-aware, code-structured chunking yields 20.4% improvement over text-based chunking, and
incorporating relative file paths adds another 16.9% gains to retrieval accuracy in our experiments. Similar
, Vol. 1, No. 1, Article . Publication date: May 2026.

38•Md Afif Al Mamun and Gias Uddin
approaches may be explored for code-aware embeddings that preserve syntactic boundaries and augment chunks
with hierarchical repository context (module/file paths) to enhance semantic recall at the file level.
Recommendation (R2).Retrieval pipelines should treat bug reports as multi-perspective queries rather than
single text inputs. Refining queries have been found to be useful in RAG pipelines [ 6,26]. Similarly, we also
showed how transformations that disentangle structural (syntactic) and behavioral (semantic) aspects of a bug
report allows the system to retrieve complementary code regions. This dual-channel formulation promotes
balanced recall and precision, ensuring robust localization even when individual query views are incomplete.
Similar approaches should be explored for better retrieval.
Recommendation (R3).While RAG pipelines improve over dense retrieval, agentic reranking allows for further
improvement in file-level localization. Such pipelines can generalize to function-level with minimal adaptation,
and cascading them with graph-guided function-level agents represents a promising direction toward fully
hierarchical, cost-efficient localization for LLM-based patch generation.
Recommendation (R4).We discussed in Section 3.6 that models having a high difference in size can still
yield competitive performance within an agentic pipeline when the right context is provided to the agent. This
observation enables a practical approach for cost-conscious organizations: adopt tiered LLM utilization where
medium-sized models (e.g., Qwen3-32B) handle localization through agentic reasoning, reserving larger or
proprietary models (e.g., GPT-4) only for computationally demanding downstream tasks like patch generation
where semantic complexity may justify the additional expense.
6.2 Theme 2. Hierarchical Localization and Fine-Grained Reasoning
Recommendation (R5).We demonstrated in Figure 12b, function and line-level localization comprise the
majority of residual failures—even when file-levelBLAgentsucceeds. To address this critical bottleneck, we
recommend extending reasoning methods to every hierarchical stage of localization. In particular, frameworks
should (1) implement multi-step, structured reasoning to iteratively assess and rank function and line candidates,
(2) deploy confidence-aware candidate expansion strategies—such as dynamically retrieving additional functions
or statements when top candidates have low discriminatory signal, and (3) integrate program structure-aware
validation checks that verify proposed localizations and edits maintain code correctness and semantic intent.
Pursuing these reasoning-based context-rich approaches at all bug localization stages will close the gap between
file-level success and end-to-end automated repair performance.
Recommendation (R6).APR frameworks should incorporate robust and tolerant output parsing strategies that
leverage fuzzy matching, or schema-driven validation to handle minor variations and formatting inconsistencies
in LLM-generated outputs. Such techniques may reduce false negatives from strict parsing failures and enhance
system reliability by enabling automatic corrections or fallback prompts. Empirical studies in LLM-based log
parsing and structured data extraction highlight the effectiveness of these approaches in improving parsing
robustness and downstream task performance [32].
Recommendation (R7).Given that manual provision of correct line-level localization can improve automated
repair success (see Figure 15), APR systems may implement iterative refinement mechanisms for line-level
localization. Specifically, these systems can dynamically generate, evaluate, and update candidate faulty lines
through multi-pass reasoning or feedback loops, enabling correction of initial mislocalizations.
6.3 Theme 3. Practical Utility
Recommendation (R8).To improve APR success rates amidst semantic patch generation errors despite correct
localization (Observation O18), we strongly recommend tightly integrating testing and semantic validation
within the patch selection pipeline. Specifically, regression and reproduction tests should be employed wherever
available to filter generated patches before acceptance. When test oracles are unavailable, lightweight semantic
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•39
Table 13. Summary of Findings, Recommendations, and Supporting Observations
Finding Recommendation Observation(s)
Theme 1: Retrieval Quality and Pipeline Architecture
Dense retrieval recall bounds pipeline performance R1: Prioritize retrieval enhancement O2, O3, O5
Query transformations provide complementary signals R2: Multi-signal query transformation O7, O8
RAG improves robustness across Top-k, agentic RAG
further improves across all Top-k ranksR3: Leverage agentic reranking O1, O4, O6, O7,
O10, O11
Model scale provides minimal gains in agentic pipelines R4: Smaller sized reasoning models should
be evaluatedO9
Theme 2: Hierarchical Localization and Fine-Grained Reasoning
Localization stages account for most of the failures R5: Extend agentic reasoning to function/-
line levelO16, O19
Function-level parsing errors propagate failures R6: Robust output parsing for functions O17
Line-level context insufficient; needs semantic reason-
ingR7: Context-preserving line-level localiza-
tionO19
Theme 3: Practical Utility
Semantic patch errors despite correct localization R8: Semantic patch validation & test inte-
grationO13, O18
Stochasticity causes unstable repair outcomes R9: Stochasticity-aware evaluation metrics O14
Modular integration improves APR effectiveness R10: Modularize localization components O12, O15
Agentic reasoning and large prompts may increase la-
tency and instabilityR11: Provide only necessary context and
limit reasoning iterationsO21, O22, O23
Agentic RAG is cost-efficient compared to unbounded
graph-guided approachesR12: Use agentic RAG to narrow down
search space and avoid unnecessary con-
text expansionO10, O20, O24
validations—such as early syntax compilation checks, static analysis for invariant preservation, and heuristic
differential testing against available reference implementations—should be incorporated.
Recommendation (R9).Given the inherent stochasticity of APR frameworks employing high-temperature
decoding and sampling, as observed in our study with notably divergent repair sets across runs (Observation
O14), evaluations should incorporate multiple independent execution runs to capture variability. Rather than
reporting only point estimates, practitioners are encouraged to aggregate results across multiple trials, analyze
overlaps among uniquely repaired issues, and provide nuanced metrics that reflect both mean effectiveness and
coverage diversity.
Recommendation (R10).The improvement in repair rates (Observation O12) when integratingBLAgentinto
Agentless demonstrates that modular localization components can enhance existing APR systems. Framework
developers may expose localization as a pluggable interface with standardized API contracts, allowing researchers
and practitioners to quickly swap improved localization modules without modifying downstream components.
This reduces integration friction and accelerates adoption of localization advances across the broader APR
ecosystem.
, Vol. 1, No. 1, Article . Publication date: May 2026.

40•Md Afif Al Mamun and Gias Uddin
Recommendation (R11).Agentic reasoning with ReAct architectures may face scalability challenges due to
cumulative context growth and sequential reasoning constraints (Observations O21, O22). To address this, if
practitionerswant toadopt agentic approach, they may (1) limit input context by providing lightweight structural
abstractions such as code skeletons when possible, (2) implement adaptive iteration controls to encourage early
convergence of reasoning, and (3) crucially incorporate explicit agent memory mechanisms [ 67] that persist and
selectively reuse past reasoning states, observations, and decisions. The integration of persistent memory allows
the agent to flexibly update its internal state without reiterating the entire reasoning history, mitigating prompt
saturation and error propagation.
Recommendation (R12).Unbounded graph-guided localization achieves strong accuracy but at substantial
inference cost. Agentic RAG instead constrains reasoning to a compact, retrieval-filtered candidate set, avoiding
unnecessary context expansion while achieving comparable accuracy. Practitioners may adopt agentic RAG
as a low-cost upstream filter and reserve graph-guided traversal for low-confidence cases, enabling scalable
hierarchical localization without sacrificing precision.
7 Threats to Validity
Internal Validity.Internal validity concerns whether the observed improvements can be confidently attributed to
the proposed agentic RAG framework rather than uncontrolled factors. To mitigate potential confounding effects,
we maintained consistent experimental conditions across all pipelines, including identical retrieval databases,
model configurations, and prompt templates. The ReAct agent’s reasoning cycles were constrained by a fixed
step limit and deterministic execution policy to ensure reproducibility. Nevertheless, inherent stochasticity in
LLM responses may have influenced intermediate reasoning behaviors. Future work could further reduce this
variance by incorporating repeated trials and statistical aggregation of results.
External Validity.Our evaluation relied primarily on the SWE-bench Lite benchmark, which, while repre-
sentative of Python-based GitHub projects and issues, may not fully reflect the diversity of industrial-scale
software development. Projects with highly domain-specific APIs, sparse documentation, or unconventional code
organization could produce different outcomes. Moreover, the agent’s effectiveness was assessed under specific
model and retrieval settings (e.g., medium-sized LLMs with static retrieval backends). Extending the evaluation to
other programming languages, larger repositories, and alternative retrieval or agent architectures would enhance
the generalizability of our findings. Furthermore, our evaluation uses APR as a representative downstream task.
While APR provides an objective and scalable setting, bug localization is also used in developer-centric tasks
such as debugging and root-cause analysis, where evaluation typically requires user studies. Thus, our findings
may not fully generalize to all usage scenarios. We leave broader evaluations as future work.
8 Related Work
Our work bridges two active research areas: bug localization and automated program repair, with a particular
emphasis on leveraging retrieval-augmented generation and agentic reasoning for repository-level bug localization.
We review representative work in these areas and position our contributions.
8.1 Bug Localization
Bug localization techniques aim to identify buggy code locations to assist debugging. These approaches can be
categorized into spectrum-based, information retrieval-based, and learning-based methods.
Spectrum-Based Fault Localization (SBFL).SBFL techniques [ 1,16,52] leverage program execution spec-
tra—coverage information collected from passing and failing test cases—to compute a suspiciousness score for
each program entity. Classic formulas such as Tarantula [16], Ochiai [1], and Dstar [51] quantify the likelihood
that a statement or function is faulty. To improve on these traditional methods, Zhang et al. [ 62] proposed PRFL,
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•41
which incorporates a PageRank-based reweighting mechanism to better capture the relative contribution of
individual tests, achieving notable accuracy gains over conventional SBFL formulations.
Information Retrieval (IR)-Based Localization.IR-based approaches [ 20,41,48,49,70] model localization
as a retrieval problem, ranking source files according to their textual similarity to bug reports. BugLocator [ 70]
pioneered this direction by introducing a revised Vector Space Model (rVSM) that integrates historical bug-
fix information to enhance ranking. Subsequent work, such as Pathidea [ 8], augmented IR-based retrieval
with execution path reconstruction from log data, demonstrating that runtime information can significantly
complement textual similarity in identifying faulty files.
Learning-Based Localization.More recently, deep learning methods have become increasingly prevalent [ 24,
35,68]. Zhang et al. [ 68] conducted a comprehensive empirical study comparing convolutional neural networks
(CNNs), recurrent neural networks (RNNs), and multilayer perceptrons (MLPs) for real-world bug localization,
finding CNN-based approaches most effective. Lam et al. [ 20] integrated deep neural representations with VSM
to mitigate lexical mismatch between bug reports and code artifacts.
The emergence of LLMs has further advanced this field by enabling models to jointly reason over natural
language and source code [ 2,7,14,38,54]. Recent studies have explored bothagentic[ 38] andagent less[ 7,15,55]
paradigms for repository-level bug localization, consistently outperforming traditional approaches. AutoFL [ 19]
is an early LLM-based approach that uses ChatGPT with tool calls to navigate Java projects and localize faults on
Defects4J [ 18], given failing tests and their execution outcomes. AgentFL [ 38] scales this setting to project-level
context by coordinating multiple LLM-based agents for fault comprehension, code navigation, and confirmation,
but assumes a test-driven Defects4J workflow. LocAgent [ 9] constructs a heterogeneous code graph over files,
classes, and functions, and uses an LLM agent to navigate it iteratively via BM25-based entity search, graph
traversal, and code retrieval, progressively narrowing the fault location from file to function level on SWE-bench-
lite and LocBench. While effective, its multi-step graph traversal incurs substantially higher cost per instance
than retrieval-based approaches according to our evaluation and literature [14].
To the best of our knowledge, no prior work has examined how RAG pipelines perform when coupled with
an agentic reasoning process forfile-level localization. Moreover, existing studies have not explored whether
transforming bug reports into retrieval-friendly queries can further improve ranking accuracy in agentic pipelines.
8.2 Automated Program Repair
Automated Program Repair (APR) techniques attempt to automatically generate patches to fix software bugs.
While we do not propose a new APR technique or improve upon the existing, one effective way to measure the
impact of an FL-technique is to see how the downstream APR gets impacted by a localization method. Different
approaches like template-based [ 27], heristic-based [ 21], or LLM-based [ 50,55,63] are utilized for automated
program repair. Recent works have shown that learning-based frameworks adopting LLMs that generate multiple
candidate patches for each bug outperforms all other traditional approaches including neural machine translation
(NMT) approaches [56].
LLM-based APR has evolved through four main paradigms–fine-tuning, prompting, procedural, and agentic
[59]. Fine-tuning approaches [ 33,64] adapt LLM weights using bug-fix data for task alignment, achieving
strong performance but demanding high computational cost. Prompting methods [ 10,57] rely on carefully
designed prompts to elicit repairs from pre-trained models without retraining, offering flexibility but limited
by prompt quality and context length. Procedural frameworks [ 15,55,69] decompose repair into explicit stages
such as localization, patch generation, and validation, enabling reproducibility with moderate overhead. Agentic
systems [ 5,50,60,65] instead delegate control to the LLM, allowing dynamic planning and reasoning for multi-file
bugs at the cost of higher latency. Despite the impressive success of LLM-based APR frameworks, their accuracy
in generating correct patches highly depends on correct FL [ 7,23]. While existing LLM-based APR research has
, Vol. 1, No. 1, Article . Publication date: May 2026.

42•Md Afif Al Mamun and Gias Uddin
primarily focused on enhancing patch generation mechanisms and prompt engineering strategies, our work
addresses the upstream bottleneck–file level localization, that may limit all these approaches.
9 Conclusion & Future Work
In this paper, we presentedBLAgent, a novel agentic retrieval-augmented generation (RAG) framework for
file-level bug localization that combines structure-aware retrieval, dual-perspective query transformation, and
bounded agentic reasoning. By explicitly modeling both structural and behavioral signals from bug reports
and grounding reasoning over retrieval-filtered code contexts,BLAgentachieves state-of-the-art localization
performance on SWE-bench Lite while maintaining substantially lower cost compared to existing graph-based or
heavily instrumented approaches. Our findings further demonstrate that improvements at the file level translate
directly into meaningful gains in downstream automated program repair. Furthermore, the proposed framework
generalizes across localization granularities, as our results show that the same agentic RAG design naturally
extends to finer-grained tasks such as function-level localization, highlighting its flexibility as a unified reasoning
framework over code.
Looking forward, a key direction is to evolve BLAgent into a unified, multi-granularity localization framework
that seamlessly integrates file-, function-, and line-level reasoning within a single pipeline. While the current
design naturally extends across these levels, further optimization is needed to enable precise fine-grained
localization without compromising efficiency. In particular, we envision a hybrid retrieval strategy where
dense retrieval is complemented with lightweight graph-based exploration in cases where semantic signals are
insufficient, enabling more robust candidate discovery in complex dependency structures. Coupled with tighter
integration between retrieval and reasoning, this would allow the agent to progressively refine localization from
coarse to fine granularity, moving toward an end-to-end localization system that directly supports debugging
and automated repair.
References
[1]R. Abreu, P. Zoeteweij, and A. J. Van Gemund. On the accuracy of spectrum-based fault localization. InTesting: Academic and industrial
conference practice and research techniques-MUTATION (TAICPART-MUTATION 2007), pages 89–98. IEEE, 2007.
[2]M. Asad, R. M. Yasir, A. Geramirad, and S. Malek. Leveraging large language model for information retrieval-based bug localization.
arXiv preprint arXiv:2508.00253, 2025.
[3]N. Bettenburg, S. Just, A. Schröter, C. Weiss, R. Premraj, and T. Zimmermann. What makes a good bug report? InProceedings of the 16th
ACM SIGSOFT International Symposium on Foundations of software engineering, pages 308–318, 2008.
[4]M. Böhme, E. O. Soremekun, S. Chattopadhyay, E. Ugherughe, and A. Zeller. Where is the bug and how is it fixed? an experiment with
practitioners. InProceedings of the 2017 11th joint meeting on foundations of software engineering, pages 117–128, 2017.
[5]I. Bouzenia, P. Devanbu, and M. Pradel. Repairagent: An autonomous, llm-based agent for program repair.arXiv preprint arXiv:2403.17134,
2024.
[6]C.-M. Chan, C. Xu, R. Yuan, H. Luo, W. Xue, Y. Guo, and J. Fu. Rq-rag: Learning to refine queries for retrieval augmented generation.
arXiv preprint arXiv:2404.00610, 2024.
[7]J. Chang, X. Zhou, L. Lulu, D. Lo, and B. Li. Bridging bug localization and issue fixing: A hierarchical localization framework leveraging
large language models.IEEE Transactions on Software Engineering, 2026.
[8]A. R. Chen, T.-H. Chen, and S. Wang. Pathidea: Improving information retrieval-based bug localization by re-constructing execution
paths using logs.IEEE Transactions on Software Engineering, 48(8):2905–2919, 2021.
[9]Z. Chen, R. Tang, G. Deng, F. Wu, J. Wu, Z. Jiang, V. Prasanna, A. Cohan, and X. Wang. Locagent: Graph-guided llm agents for code
localization. InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
8697–8727, 2025.
[10] Z. Fan, X. Gao, M. Mirchev, A. Roychoudhury, and S. H. Tan. Automated repair of programs from large language models. In2023
IEEE/ACM 45th International Conference on Software Engineering (ICSE), pages 1469–1481. IEEE, 2023.
[11] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang. Large language model based multi-agents: A survey
of progress and challenges.arXiv preprint arXiv:2402.01680, 2024.
[12] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions.ACM Transactions on Information Systems, 43(2):1–55, 2025.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•43
[13] X. Huang, W. Liu, X. Chen, X. Wang, H. Wang, D. Lian, Y. Wang, R. Tang, and E. Chen. Understanding the planning of llm agents: A
survey.arXiv preprint arXiv:2402.02716, 2024.
[14] Z. Jiang, X. Ren, M. Yan, W. Jiang, Y. Li, and Z. Liu. Cosil: Software issue localization via llm-driven code repository graph searching.
arXiv preprint arXiv:2503.22424, 2025.
[15] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan. SWE-bench: Can language models resolve real-world
github issues? InThe Twelfth International Conference on Learning Representations, 2024.
[16] J. A. Jones, M. J. Harrold, and J. Stasko. Visualization of test information to assist fault localization. InProceedings of the 24th international
conference on Software engineering, pages 467–477, 2002.
[17] H. Joshi, J. C. Sanchez, S. Gulwani, V. Le, G. Verbruggen, and I. Radiček. Repair is nearly generation: Multilingual program repair with
llms. InProceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 5131–5140, 2023.
[18] R. Just, D. Jalali, and M. D. Ernst. Defects4j: a database of existing faults to enable controlled testing studies for java programs. In
Proceedings of the 2014 International Symposium on Software Testing and Analysis, ISSTA 2014, page 437–440, New York, NY, USA, 2014.
Association for Computing Machinery.
[19] S. Kang, G. An, and S. Yoo. A quantitative and qualitative evaluation of llm-based explainable fault localization.Proceedings of the ACM
on Software Engineering, 1(FSE):1424–1446, 2024.
[20] A. N. Lam, A. T. Nguyen, H. A. Nguyen, and T. N. Nguyen. Bug localization with combination of deep learning and information retrieval.
In2017 IEEE/ACM 25th International Conference on Program Comprehension (ICPC), pages 218–229. IEEE, 2017.
[21] X. B. D. Le, D. Lo, and C. Le Goues. History driven program repair. In2016 IEEE 23rd international conference on software analysis,
evolution, and reengineering (SANER), volume 1, pages 213–224. IEEE, 2016.
[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[23] F. Li, J. Jiang, J. Sun, and H. Zhang. Hybrid automated program repair by combining large language models and program analysis.ACM
Transactions on Software Engineering and Methodology, 34(7):1–28, 2025.
[24] X. Li, W. Li, Y. Zhang, and L. Zhang. Deepfl: Integrating multiple fault diagnosis dimensions for deep fault localization. InProceedings of
the 28th ACM SIGSOFT international symposium on software testing and analysis, pages 169–180, 2019.
[25] Z. Li, J. Wang, Z. Jiang, H. Mao, Z. Chen, J. Du, Y. Zhang, F. Zhang, D. Zhang, and Y. Liu. Dmqr-rag: Diverse multi-query rewriting for
rag.arXiv preprint arXiv:2411.13154, 2024.
[26] K. Lin, K. Lo, J. E. Gonzalez, and D. Klein. Decomposing complex queries for tip-of-the-tongue retrieval.arXiv preprint arXiv:2305.15053,
2023.
[27] K. Liu, A. Koyuncu, D. Kim, and T. F. Bissyandé. Tbar: Revisiting template-based automated program repair. InProceedings of the 28th
ACM SIGSOFT international symposium on software testing and analysis, pages 31–42, 2019.
[28] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the middle: How language models use long
contexts.arXiv preprint arXiv:2307.03172, 2023.
[29] Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot
prompt order sensitivity.arXiv preprint arXiv:2104.08786, 2021.
[30] X. Ma, Y. Gong, P. He, N. Duan, et al. Query rewriting in retrieval-augmented large language models. InThe 2023 Conference on Empirical
Methods in Natural Language Processing, 2023.
[31] Y. Ma, Q. Yang, R. Cao, B. Li, F. Huang, and Y. Li. Alibaba lingmaagent: Improving automated issue resolution via comprehensive
repository exploration. InProceedings of the 33rd ACM International Conference on the Foundations of Software Engineering, pages
238–249, 2025.
[32] Z. Ma, A. R. Chen, D. J. Kim, T.-H. Chen, and S. Wang. Llmparser: An exploratory study on using large language models for log parsing.
InProceedings of the IEEE/ACM 46th International Conference on Software Engineering, pages 1–13, 2024.
[33] R. Macháček, A. Grishina, M. Hort, and L. Moonen. The impact of fine-tuning large language models on automated program repair.
arXiv preprint arXiv:2507.19909, 2025.
[34] Y. A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world
graphs.IEEE transactions on pattern analysis and machine intelligence, 42(4):824–836, 2018.
[35] X. Meng, X. Wang, H. Zhang, H. Sun, and X. Liu. Improving fault localization and program repair with deep semantic features and
transferred knowledge. InProceedings of the 44th International Conference on Software Engineering, pages 1169–1180, 2022.
[36] F. Niu, C. Li, K. Liu, X. Xia, and D. Lo. When deep learning meets information retrieval-based bug localization: A survey.ACM Computing
Surveys, 57(11):1–41, 2025.
[37] M. R. Parvez, W. U. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang. Retrieval augmented code generation and summarization.arXiv
preprint arXiv:2108.11601, 2021.
[38] Y. Qin, S. Wang, Y. Lou, J. Dong, K. Wang, X. Li, and X. Mao. Agentfl: Scaling llm-based fault localization to project-level context.arXiv
preprint arXiv:2403.16362, 2024.
, Vol. 1, No. 1, Article . Publication date: May 2026.

44•Md Afif Al Mamun and Gias Uddin
[39] R. Qu, R. Tu, and F. Bao. Is semantic chunking worth the computational cost? InFindings of the Association for Computational Linguistics:
NAACL 2025, pages 2155–2177, 2025.
[40] N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks.arXiv preprint arXiv:1908.10084, 2019.
[41] R. K. Saha, M. Lease, S. Khurshid, and D. E. Perry. Improving bug localization using structured information retrieval. In2013 28th
IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 345–355. IEEE, 2013.
[42] A. M. Samir and M. M. Rahman. Improved ir-based bug localization with intelligent relevance feedback.arXiv preprint arXiv:2501.10542,
2025.
[43] K. Sawarkar, A. Mangal, and S. R. Solanki. Blended rag: Improving rag (retriever-augmented generation) accuracy with semantic search
and hybrid query-based retrievers. In2024 IEEE 7th international conference on multimedia information processing and retrieval (MIPR),
pages 155–161. IEEE, 2024.
[44] S. Shao and T. Yu. Enhancing ir-based fault localization using large language models.arXiv preprint arXiv:2412.03754, 2024.
[45] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean. Outrageously large neural networks: The sparsely-gated
mixture-of-experts layer.arXiv preprint arXiv:1701.06538, 2017.
[46] Stripe. The developer coefficient: Software engineering efficiency and its $3 trillion impact on global GDP. https://stripe.com/files/
reports/the-developer-coefficient.pdf, Sept. 2018. Accessed: 2026-03-24.
[47] Y. Tao, Y. Qin, and Y. Liu. Retrieval-augmented code generation: A survey with focus on repository-level approaches.arXiv preprint
arXiv:2510.04905, 2025.
[48] Q. Wang, C. Parnin, and A. Orso. Evaluating the usefulness of ir-based fault localization techniques. InProceedings of the 2015
international symposium on software testing and analysis, pages 1–11, 2015.
[49] S. Wang and D. Lo. Version history, similar report, and structure: Putting them together for improved bug localization. InProceedings of
the 22nd international conference on program comprehension, pages 53–63, 2014.
[50] X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, et al. Openhands: An open platform for ai software
developers as generalist agents.arXiv preprint arXiv:2407.16741, 2024.
[51] W. E. Wong, V. Debroy, R. Gao, and Y. Li. The dstar method for effective software fault localization.IEEE Transactions on Reliability,
63(1):290–308, 2013.
[52] W. E. Wong, R. Gao, Y. Li, R. Abreu, and F. Wotawa. A survey on software fault localization.IEEE Transactions on Software Engineering,
42(8):707–740, 2016.
[53] W. E. Wong, R. Gao, Y. Li, R. Abreu, F. Wotawa, and D. Li. Software fault localization: An overview of research, techniques, and tools.
Handbook of Software Fault Localization: Foundations and Advances, pages 1–117, 2023.
[54] Y. Wu, Z. Li, J. M. Zhang, M. Papadakis, M. Harman, and Y. Liu. Large language models in fault localisation.arXiv preprint arXiv:2308.15276,
2023.
[55] C. S. Xia, Y. Deng, S. Dunn, and L. Zhang. Demystifying llm-based software engineering agents.Proc. ACM Softw. Eng., 2(FSE), June 2025.
[56] C. S. Xia, Y. Wei, and L. Zhang. Automated program repair in the era of large pre-trained language models. In2023 IEEE/ACM 45th
International Conference on Software Engineering (ICSE), pages 1482–1494. IEEE, 2023.
[57] C. S. Xia and L. Zhang. Conversational automated program repair.arXiv preprint arXiv:2301.13246, 2023.
[58] Y. Xiao, J. Keung, K. E. Bennin, and Q. Mi. Improving bug localization with word embedding and enhanced convolutional neural
networks.Information and Software Technology, 105:17–29, 2019.
[59] B. Yang, Z. Cai, F. Liu, B. Le, L. Zhang, T. F. Bissyandé, Y. Liu, and H. Tian. A survey of llm-based automated program repair: Taxonomies,
design paradigms, and applications.arXiv preprint arXiv:2506.23749, 2025.
[60] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press. Swe-agent: Agent-computer interfaces enable automated
software engineering.Advances in Neural Information Processing Systems, 37:50528–50652, 2024.
[61] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. In
The eleventh international conference on learning representations, 2022.
[62] M. Zhang, Y. Li, X. Li, L. Chen, Y. Zhang, L. Zhang, and S. Khurshid. An empirical study of boosting spectrum-based fault localization
via pagerank.IEEE Transactions on Software Engineering, 47(6):1089–1113, 2019.
[63] Q. Zhang, C. Fang, Y. Xie, Y. Ma, W. Sun, Y. Yang, and Z. Chen. A systematic literature review on large language models for automated
program repair.arXiv preprint arXiv:2405.01466, 2024.
[64] T. Zhang, T. Yu, T. Hashimoto, M. Lewis, W.-t. Yih, D. Fried, and S. Wang. Coder reviewer reranking for code generation. InInternational
Conference on Machine Learning, pages 41832–41846. PMLR, 2023.
[65] Y. Zhang, H. Ruan, Z. Fan, and A. Roychoudhury. Autocoderover: Autonomous program improvement. InProceedings of the 33rd ACM
SIGSOFT International Symposium on Software Testing and Analysis, pages 1592–1604, 2024.
[66] Y. Zhang, X. Zhao, Z. Z. Wang, C. Yang, J. Wei, and T. Wu. cast: Enhancing code retrieval-augmented generation with structural
chunking via abstract syntax tree.arXiv preprint arXiv:2506.15655, 2025.
[67] Z. Zhang, Q. Dai, X. Bo, C. Ma, R. Li, X. Chen, J. Zhu, Z. Dong, and J.-R. Wen. A survey on the memory mechanism of large language
model-based agents.ACM Transactions on Information Systems, 43(6):1–47, 2025.
, Vol. 1, No. 1, Article . Publication date: May 2026.

BLAgent: Agentic RAG for File-Level Bug Localization•45
[68] Z. Zhang, Y. Lei, X. Mao, M. Yan, L. Xu, and X. Zhang. A study of effectiveness of deep learning in locating real faults.Information and
Software Technology, 131:106486, 2021.
[69] Y. Zhao, S. Chen, J. Zhang, and Z. Li. Recode: Improving llm-based code repair with fine-grained retrieval-augmented generation.arXiv
preprint arXiv:2509.02330, 2025.
[70] J. Zhou, H. Zhang, and D. Lo. Where should the bugs be fixed? more accurate information retrieval-based bug localization based on bug
reports. In2012 34th International conference on software engineering (ICSE), pages 14–24. IEEE, 2012.
, Vol. 1, No. 1, Article . Publication date: May 2026.
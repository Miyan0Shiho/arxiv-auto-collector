# MEMCoder: Multi-dimensional Evolving Memory for Private-Library-Oriented Code Generation

**Authors**: Mofei Li, Taozhi Chen, Guowei Yang, Jia Li

**Published**: 2026-04-27 09:27:59

**PDF URL**: [https://arxiv.org/pdf/2604.24222v1](https://arxiv.org/pdf/2604.24222v1)

## Abstract
Large Language Models (LLMs) excel at general code generation, but their performance drops sharply in enterprise settings that rely on internal private libraries absent from public pre-training corpora. While Retrieval-Augmented Generation (RAG) offers a training-free alternative by providing static API documentation, we find that such documentation typically provides only isolated definitions, leaving a fundamental knowledge gap. Specifically, LLMs struggle with a task-level lack of coordination patterns between APIs and an API-level misunderstanding of parameter constraints and boundary conditions. To address this, we propose MEMCoder, a novel framework that enables LLMs to autonomously accumulate and evolve Usage Guidelines across these two dimensions. MEMCoder introduces a Multi-dimensional Evolving Memory that captures distilled lessons from the model's own problem-solving trajectories. During inference, MEMCoder employs a dual-source retrieval mechanism to inject both static documentation and relevant historical guidelines into the context. The framework operates in an automated closed loop by using objective execution feedback to reflect on successes and failures, resolve knowledge conflicts, and dynamically update memory. Extensive evaluations on the NdonnxEval and NumbaEval benchmarks demonstrate that MEMCoder substantially enhances existing RAG systems, yielding an average absolute pass@1 gain of 16.31%. Furthermore, MEMCoder exhibits vastly superior domain-specific adaptation compared to existing memory-based continual learning methods.

## Full Text


<!-- PDF content starts -->

MEMCoder: Multi-dimensional Evolving Memory
for Private-Library-Oriented Code Generation
Mofei Li*
College of AI
Tsinghua University
Beijing, China
Fitten Tech Co., Ltd.
Beijing, China
lmf25@mails.tsinghua.edu.cnTaozhi Chen*
College of AI
Tsinghua University
Beijing, China
chentaozhi313@gmail.comGuowei Yang
Fitten Tech Co., Ltd.
Beijing, China
gavin@eniacode.comJia Li†
College of AI
Tsinghua University
Beijing, China
jia_li@mail.tsinghua.edu.cn
Abstract—Large Language Models (LLMs) excel at general
code generation, but their performance drops sharply in enterprise
settings that rely on internal private libraries absent from public
pre-training corpora. While Retrieval-Augmented Generation
(RAG) offers a training-free alternative by providing static API
documentation, we find that such documentation typically provides
only isolated definitions, leaving a fundamental knowledge gap.
Specifically, LLMs struggle with a task-level lack of coordination
patterns between APIs and an API-level misunderstanding of
parameter constraints and boundary conditions. To address
this, we propose MEMCODER, a novel framework that
enables LLMs to autonomously accumulate and evolveUsage
Guidelinesacross these two dimensions. MEMCODERintroduces
a Multi-dimensional Evolving Memory that captures distilled
lessons from the model’s own problem-solving trajectories.
During inference, MEMCODERemploys a dual-source retrieval
mechanism to inject both static documentation and relevant
historical guidelines into the context. The framework operates in
an automated closed loop by using objective execution feedback
to reflect on successes and failures, resolve knowledge conflicts,
and dynamically update memory. Extensive evaluations on the
NdonnxEval and NumbaEval benchmarks demonstrate that
MEMCODERsubstantially enhances existing RAG systems,
yielding an average absolutepass@1gain of 16.31%. Furthermore,
MEMCODERexhibits vastly superior domain-specific adaptation
compared to existing memory-based continual learning methods.
I. INTRODUCTION
Large Language Models (LLMs) have demonstrated
exceptional proficiency in general code generation tasks [ 1–4].
However, their performance drops sharply in real-world enter-
prise environments that rely heavily on internal private libraries,
which are absent from public pre-training corpora [ 5,6]. This
gap defines the task ofPrivate-Library-Oriented Code Gener-
ation, which requires models to accurately coordinate project-
specific APIs to fulfill complex requirements. Since LLMs
typically lack prior knowledge of these libraries, they often
struggle to utilize them effectively, thereby significantly limiting
their practical efficacy in real-world software development.
To address the issue that private-library APIs are absent
from public pre-training corpora, the prevailing paradigm
leverages Retrieval-Augmented Generation (RAG) to provide
* Equal contribution.†Corresponding author.LLMs with static API documentation [ 5–8]. However, our
study in Section III reveals that the performance gains from
merely providing static documentation are remarkably limited.
Empirical results show that even in anOraclesetting where
the complete set of required documentation is perfectly
injected, the performance of advanced code models remains
unsatisfactory. For instance, in the NumbaEval benchmark [ 9–
11], providing full API specifications only improves the Pass@1
of Qwen2.5-Coder-7B-Instruct [ 12] from 26.15% to 27.70%,
an incremental gain of only 1.55%. Through qualitative
observation of these failure cases, we find that the root cause is
that static documentation typically provides only isolated API
definitions, and the model still does not know how to correctly
utilize these APIs in complex scenarios. Specifically, this gap
exists at two levels: atask-levellack of coordination patterns
between multiple APIs, and anAPI-levelmisunderstanding of
parameter constraints and boundary conditions. To this end, as
illustrated in Figure 1, we manually writeUsage Guidelines
covering both levels and inject them into the context, finding
that the model can correct previous invocation errors and
produce executable solutions. This contrast suggests that
relying solely on static API documentation is insufficient, and
incorporatingUsage Guidelinesthat can teach the model how
to use APIs into the retrieval source can further enhance the
performance of private-library-oriented code generation.
Given the critical value ofUsage Guidelines, a key
operational challenge arises in their acquisition. For large-scale
and rapidly evolving private libraries, relying on manual efforts
to craft theseUsage Guidelinesis not only labor-intensive but
also entirely unsustainable in real-world development scenarios.
Considering that existing research demonstrates LLMs can
self-reflect through test-time interactions with the environment
and continuously accumulate insights into the reasons behind
task success or failure [ 13], we naturally investigate in
Section III whether LLMs can leverage this capability to
automatically reflect on and generateUsage Guidelinesfrom
their own generated responses and execution feedback. Our
analysis (as illustrated in Figure 2) reveals that when provided
with their initially generated code and the corresponding
execution feedback, the models demonstrate a remarkablearXiv:2604.24222v1  [cs.SE]  27 Apr 2026

ability to reflect and accurately generate high-qualityUsage
Guidelines, which subsequently enable the models to correct
their previous errors and produce correct solutions. This finding
suggests that instead of relying on high-cost manual curation,
we can implement an automated closed loop of generation,
execution, feedback, and reflection to achieve the autonomous
accumulation ofUsage Guidelinesduring the inference process.
Based on the above observations and analysis, we propose
MEMCODER, a novel framework for private-library-oriented
code generation characterized by its training-free, plug-and-play,
and continual learning nature. To address the identified task-
level and API-level knowledge gaps, the core of MEMCODER
is aMulti-dimensional Evolving Memorythat stores and
maintainsUsage Guidelinesacross two dimensions—task-level
memory and API-level memory—supplemented by key auxil-
iary information such as code snippets and execution feedback.
During inference, MEMCODERemploys aGuideline-Driven
Code Generationstrategy, which injects relevant static
documentation along withUsage Guidelinesinto the context
through a dual-source retrieval mechanism. After code exe-
cution, MEMCODERutilizes execution feedback to refine and
update theUsage Guidelinesvia aFeedback-Driven Memory
Evolutionmechanism. We conducted extensive experiments
on private-library benchmarks, and the results validate the
effectiveness of MEMCODER. Integrating MEMCODERinto
strong static RAG baselines yields an average absolute Pass@1
gain of 16.31%. Moreover, under continual adaptation settings,
MEMCODERoutperforms existing memory-based continual
learning baselines without updating model parameters.
In summary, our contributions are as follows:
•We identify the limitations of static API documentation in
private-library-oriented code generation. To address this,
we introduce a Multi-dimensional Evolving Memory that
extends the retrieval source from static documentation
alone to documentation enriched with reusableUsage
Guidelines.
•We propose MEMCODER, a novel framework for private-
library-oriented code generation characterized by its
training-free, plug-and-play, and continual learning nature.
It organizesUsage Guidelinesinto task-level and API-level
memory, jointly retrieves documentation and memory dur-
ing inference, and continuously evolves memory through
a closed loop of execution, feedback, and reflection.
•We conduct extensive experiments on private-library
benchmarks. MEMCODERconsistently improves strong
static RAG baselines and outperforms existing memory-
based continual learning methods without updating model
parameters.
II. RELATEDWORK
Large Language Models for Code Generation.Large
language models (LLMs) have recently become a dominant
paradigm for code generation and software engineering
tasks [ 14–18]. Recent model families, including GPT [ 19,20],
LLaMA [ 21,22], Qwen [ 23,24], and DeepSeek [ 25,26],
have achieved strong performance on a broad range ofbenchmarks for code completion, function synthesis, and
program construction. These results suggest that modern code
LLMs can capture common programming patterns, language
syntax, and knowledge about widely used public libraries.
Despite this progress, the capability of code LLMs remains
fundamentally constrained by the knowledge available during
training [ 27]. After pretraining or instruction tuning, model
parameters provide only a static representation of library knowl-
edge. As a result, code LLMs often struggle on tasks that de-
pend on knowledge absent from public corpora, such as project-
internal APIs, proprietary libraries, or rapidly evolving software
ecosystems [ 6,28]. This limitation has motivated a growing line
of research on code generation under external knowledge sup-
port, especially in settings where successful generation depends
on access to private-library information at inference time.
Private-Library-Oriented Code Generation.Private-library-
oriented code generation studies how to generate executable
code for tasks that depend on proprietary or project-internal
libraries [ 6]. Compared with conventional code generation [ 29–
32], this setting is more challenging because the target APIs
are often absent from public training corpora, and successful
generation depends on accurate understanding and correct
use of library-specific functions. Existing work includes both
training-based and training-free solutions, but training-based
adaptation is often less practical in private-library settings
because supervision data is limited and library APIs evolve
over time [11, 33–35].
Existing training-free approaches consistently address
private-library code generation with retrieval-augmented
generation (RAG) [ 36]. Early methods such as APIFinder [ 6]
and DocPrompting [ 7] retrieve relevant APIs or library
documents and provide the retrieved information in the
prompt. Later work improves this pipeline by strengthening
API retrieval. For example, EpiGEN [ 8] and CAPIR [ 37]
decompose coding requirements into intermediate subtasks
or intents before retrieval to improve API matching and
recall, while ExploraCoder [ 5] further incorporates real-time
execution feedback within a single task to alleviate failures
caused by incomplete or ambiguous API documentation.
Existing methods still frame private-library code generation
mainly as a retrieval problem [ 6–8,37]. Retrieved
documentation is necessary, but often insufficient for
executable solutions, because correct private-library use
also depends on cross-API composition patterns, practical
constraints, and error-prone details that are not fully specified
in documentation. This gap motivates methods that leverage
reusable usage experience in addition to retrieved documents.
Memory-Based Continual Learning.A growing line of work
studies how language models can improve over time by accumu-
lating and reusing memory during inference, without repeated
offline retraining [ 13,38]. In these approaches, the model is
augmented with dynamically updated external memory, such
as retrieved context, reusable summaries, reflections, execution
trajectories, or other experience records [ 13,38]. Such methods
are especially appealing when the target knowledge is private,
evolving, or difficult to collect as supervised training data [ 11].

TABLE I
Pass@kONNdonnxEvalANDNumbaEval.
MethodNdonnxEval NumbaEval
pass@1 pass@3 pass@5 pass@1 pass@3 pass@5
Qwen2.5-Coder-7B-Instruct
Vanilla 14.20 23.57 27.78 26.15 42.89 50.08
Oracle 45.62 61.31 66.66 27.70 45.35 52.65
Gain(↑) +31.42 +37.74 +38.88 +1.55 +2.46 +2.57
Llama-3.1-8B-Instruct
Vanilla 6.75 14.13 18.31 9.57 21.37 28.90
Oracle 26.92 43.84 51.06 10.16 21.77 28.79
Gain(↑) +20.17 +29.71 +32.75 +0.59 +0.40 -0.11
DeepSeek-Coder-6.7B-Instruct
Vanilla 12.60 21.53 26.34 11.93 27.46 37.52
Oracle 43.73 59.66 65.50 13.53 29.47 38.26
Gain(↑) +31.13 +38.13 +39.16 +1.60 +2.01 +0.74
Recent work has explored several forms of memory-based
continual learning. For example, Dynamic Cheatsheet [ 38]
maintains a persistent adaptive memory of reusable strategies,
code snippets, and problem-solving insights, while and
ReMem [ 13] study how historical experience can be retrieved,
refined, and updated to support future inference. These
approaches show that inference-time accumulation and reuse
of experience can improve downstream performance without
modifying model parameters [13, 38].
Our work is related to this line of research, but differs in
both task setting and memory design. Existing methods are
mostly designed for general reasoning or agent tasks, rather
than private-library-oriented code generation. In addition, prior
methods typically store reusable experience in a single memory
space, without explicitly separating task-level reusable knowl-
edge from API-level reusable knowledge. In contrast, MEM-
CODERuses a Multi-dimensional Evolving Memory tailored
to private-library-oriented code generation and updates both
task-level and API-level experience from execution feedback.
III. MOTIVATION
Currently, private-library-oriented code generation primarily
relies on the RAG framework. Research efforts have largely
focused on optimizing retrieval algorithms to improve the
recall of API documentation, aiming to push performance
toward the theoretical upper bound of this paradigm. However,
little research has investigated the limitations of this upper
bound itself—specifically, whether relying solely on static API
documentation is sufficient to support reliable code generation.
To investigate the theoretical bottleneck, we design an Oracle
study. By directly injecting all required API specifications
(including signatures and descriptions), we completely elimi-
nate retrieval errors to observe the actual gains of static API
documentation. We compare this setting with a Vanilla setting
where no reference documentation is provided, and evaluate
three mainstream open-source models on the NdonnxEval and
NumbaEval benchmarks.
Empirical results in Table I reveal a stark reality: even
in an idealized state where retrieval errors are eliminated,
the performance of existing models remains far below the
Usage Guidelines:•Task-level: Align data, mask, and the constant-filled target to a shared broadcasted shape before masked selection.•API-level: 1.Usendx.full(shape, fill_value)to construct the constant-filled target array.2.Usendx.broadcast_arrays(...)to align the data input, mask, and target to one shared shape before selection.3.Inndx.where(cond, a,b), cond=True selects a, andcond=Falseselectsb.1: defmasked_fill_like(x, shape, fill_value, mask=None):2: x_arr = ndx.asarray(x)3:result = ndx.full(shape, fill_value)4:if mask is not None:5:mask_arr = ndx.asarray(mask)6: data_b, mask_b, target_b = ndx.broadcast_arrays(x_arr, mask_arr, target)7:result = ndx.where(mask_b, target_b, data_b)8:else:9:result = target10:     returnresultFailure under Oracle Docs OnlyInput: 
Output:
Failure:Goal: Convert input data to an array of a given shape filled with a specified constant value; if a mask is provided, masked positions should take the constant while unmasked positions keep the original values.Required API Knowledge (Oracle):•ndonnx.full(shape, fill_value, *, dtype=None, device=None) -> ArrayCreates a constant-filled target array with the requested shape.•ndonnx.broadcast_arrays(*arrays) -> list[Array]Aligns multiple arrays to a shared broadcasted shape.•ndonnx.where(cond, a, b) -> ArrayReturnsawherecondis true, andbotherwise.•…1: defmasked_fill_like(x, shape, fill_value, mask=None):2: x_array = ndx.asarray(x)3:result = ndx.full(shape, fill_value)4:if mask is not None:5:mask_array = ndx.asarray(mask)6:broadcasted_mask = ndx.broadcast_to(mask_array, x_array.shape)7:result = ndx.where(broadcasted_mask, x_array, result)8:returnresultValueError: target rank must be equal or greater than rank of 'self'Success with Usage GuidelinesInput: 
Output:
Success:Goal: Convert input data to an array of a given shape filled with a specified…Required API Knowledge (Oracle):•ndonnx.full…
Passes all test cases.Fig. 1. A representative failure case study on NdonnxEval. Qualitative analysis
reveals model failures at both the task-level and API-level . Injecting targeted
Usage Guidelines addressing these two levels successfully recovers the correct
invocation pattern.
requirements for practical deployment. The effects of injecting
static API documentation vary significantly across libraries. On
NdonnxEval, although the three models achieve a significant
average Pass@1 gain of 27.57%, the theoretical upper bound
remains constrained, with an average Pass@5 success rate of
only 61.07%. On the more complex NumbaEval benchmark,
the injection of static API documentation is nearly ineffective,
with an average Pass@1 improvement of only 1.25% across
the three models, and Llama-3.1-8B-Instruct even experiences
negative growth in Pass@5. This demonstrates that the primary
performance bottleneck is no longer retrieval quality, but rather
the model’s inability to effectively translate accurate API
information into correct and executable code. This phenomenon
suggests that static API documentation possesses significant
limitations when handling complex private-library logic.

♂pencil-altObservation 1:Even with perfect retrieval, the information
provided by static API documentation is inadequate to support
models in completing complex private-library code generation.
We therefore conduct a case study on the failure cases under
the Oracle setting to investigate why the model fails despite
having complete API specifications,. As shown in Figure 1,
we select the masked_fill_like task from NdonnxEval as
a representative example. This task requires first constructing
a constant target tensor of a specified shape. It then uses a
boolean mask to fill this constant where the mask is true,
while retaining the original input data elsewhere.
Through an in-depth analysis of numerous failure cases,
we find that the model’s errors are primarily manifested
at two levels, and the aforementioned representative case
exposes both of these typical failure modes. First is the API
coordination failure at thetask-level. The model fails to
use ndonnx.broadcast_arrays provided by the Oracle to
jointly align the input data, mask, and target tensor. Instead, it
extracts an internal function from the underlying source code
and erroneously performs a one-way ndonnx.broadcast_to
operation solely on the mask, disrupting the coordination work-
flow. Second is the misunderstanding of parameter semantics
and constraints at theAPI-level. Specifically, when calling
ndonnx.where(cond, a, b) , the model not only reverses
the constant branch and the original data branch that should be
selected when mask=True , but also ignores the strict implicit
constraint of this interface requiring the dimensions (rank) of
the input tensors to be aligned.
To address the failure modes at these two levels, we
summarize the missing coordination patterns and interface
usage details into specific Usage Guidelines and re-inject
them into the Oracle context for verification. Experiments
show that after adding these guidelines, the model successfully
corrects its previous invocation errors, recovers the correct
API coordination pattern, and ultimately generates executable
code that passes all test cases. This stark contrast proves
that supplementing explicit usage guidelines can effectively
compensate for the deficiencies of static documentation, thereby
improving the performance of private-library code generation.
♂pencil-altObservation 2:The model failure is due to the lack of task-level
collaboration and the misunderstanding of API-level constraints;
using the usage guide for injection can effectively correct errors
and improve performance.
Although manually-written usage guidelines can significantly
enhance performance, when dealing with large-scale and
rapidly evolving private libraries, relying on expert experience
for manual annotation is not only costly but also unsustainable
in actual production environments. Therefore, we further
explored whether the model has the ability to conduct
self-reflection based on environmental feedback during testing
and autonomously extract usage guidelines.
As shown in Figure 2, we reuse the masked_fill_like
failure case described previously and provided the initially
generated error code and corresponding execution feedback to
Input: 
Output:Goal: Convert input data to an array of a given shape filled with a specified…Failed Output:1: defmasked_fill_like(x, shape, fill_value, mask=None):2: x_array = ndx.asarray(x)3:…Runtime Error:ValueError: target rank must be equal or greater than rank of 'self'Reflection Prompts :L1: Summarize the missing task-level coordination pattern across multiple APIs from the failed code and feedback.L2:Distillthe API-level parameter-role semantics violated in the failed execution.Task-Level Guideline:Ensure that the data input, mask, and constant-filled target are aligned together usingndonnx.broadcast_arraysbefore performing any masked selection operations.API-Level Guideline:Whencondis true,ais selected; otherwise,bis selected.Fig. 2. A reflection case on NdonnxEval with Qwen2.5-Coder-7B-Instruct.
From failed code and execution feedback, the model derives task-level and
API-level Usage Guidelines.
Qwen2.5-Coder-7B for reflection. Experimental observations re-
vealed that the model demonstrated outstanding self-correction
and summarization capabilities: at the task level, it accurately
identified the alignment gaps in multi-API collaboration and
summarized that ndonnx.broadcast_arrays must be used
before the execution selection operation for unified alignment;
at the API level, it also precisely extracted the correct
execution semantics of the conditional branches and parameter
roles in ndonnx.where. This performance of learning from
feedback proves that we do not need to rely on costly manual
maintenance, but can achieve autonomous accumulation and
evolution of private library practical experience through the
automated closed loop of ”generation – execution – feedback –
reflection.”
♂pencil-altObservation 3:While manual curation is unsustainable, LLMs
can autonomously distill task-level and API-level knowledge
from feedback, enabling an automated closed-loop for experience
accumulation.
IV. METHODOLOGY
A. Overview
We propose MEMCODER, a novel framework for private-
library-oriented code generation featuring training-free,
plug-and-play, and continual learning characteristics. Formally,
letLdenote the target private library and Ddenote its static
API documentation set. To manage the aforementioned Usage
Guidelines, MEMCODERintroduces aMulti-dimensional
Evolving Memory M(Section IV-B ), which systematically
organizes and stores guidelines and their auxiliary code
information across two dimensions: the task-level ( MTask) and
the API-level ( MAPI). As illustrated in Figure 3, MEMCODER
operates in a closed-loop workflow consisting of forward
generation and backward evolution.
During the forward phase,Guideline-Driven Code Genera-
tion(Section IV-C ), MEMCODERsynthesizes the code solution
ctfor a given natural language requirement rt. Serving as a
flexible extension to the conventional RAG pipeline, it employs
a dual-source retrieval strategy: in addition to retrieving the
API documentation D, it retrieves relevant historical records

Multi-dimensional Evolving Memory1.Retrieve viaRequirementGuideline-Driven Code GenerationFeedback-Driven Memory Evolution
- API Name: ndonnx.asarray
Index Layer
2.Return Mem3.Retrievevia API docs
6.GenerateCode5.Inject Mem4.RetrieveAPI Mem
8.StoreTask Query9.GenerateMem7.UpdateTask-Level Mem10.UpdateAPI-Level Mem11.StoreCode Snipptes13.GenerateUsage Guideline14.Discard/Add/Deleted12.UpdateUsage Weights
CodeMEMCoderRetrieved APIsRequirement
Usage GuidelineMEMCoderExecution FeedbackMEMCoder- Requirement: Write a helper that computes ……
Index Layer- Used APIs: ndonnx.asarray, ndonnx.boradcast_arrays ……- Code: def masked_min_and_row_product(x, y, mask)…- Execution Feedback: Success - Usage Guideline: Use ndx.asarray for shape compatibility…Task-level Memory
API-level Memory
[-] Deleted (Weight: 3) Note: Directly using a sequence of Arrays in'asarray' causes errors; convert it to a single array first. ……[+] Active (Weight: 77) Note: Always check if the input object isalready an `Array` before copying to avoid unnecessary operations.
Content Layer 
- Usage Guideline:  {if isinstance(x,list):\n  x = ndx.asarray(x)……} ……- Code Snippet :     ·Qualname: ndonnx.asarray    ·Signature: (obj, ...) -> Array     ·Definition: def asarray(obj, ...):  ...- API Documentation: 
Content Layer Fig. 3. Overview of the MEMCODERframework.Middle:The Multi-dimensional Evolutionary Memory ( IV-B ) stores refined task-level and API-level
memories.Left:The Guideline-Driven Code Generation pipeline ( IV-C ) retrieves these memories along with API docs to guide code generation.Right:The
Feedback-Driven Memory Evolution module (IV-D) updates and optimizes the memory based on real-time execution feedback.
fromMTaskand matches corresponding Usage Guidelines from
MAPIfor the candidate APIs. By injecting the documenta-
tionDrt⊂ D and the retrieved multi-dimensional memory
Mrt⊂ M into the context, the LLM directly generates
the final code ct. This ensures that the model masters the
coordination strategies while understanding the API definitions.
In the backward phase following generation, namely
Feedback-Driven Memory Evolution(Section IV-D ), the system
obtains objective feedback ftby executing ctto continuously
update the memory M. By analyzing the execution trajectory,
the LLM Reflector extracts new Usage Guidelines and updates
the contents and weights of the existing memory M. Through
this closed-loop evolution, MEMCODERachieves continual
capability enhancement without parameter updates.
B. Multi-dimensional Evolving Memory
To systematically store and manage theUsage Guidelines
automatically generated by the LLM, we construct aMulti-
dimensional Evolving Memory M. As detailed in the middle
block of Figure 3, to balance efficient retrieval matching
with rich generative context, every memory entry in our
framework is organized into two components: anIndex Layer
designed for semantic search, and aContent Layerdesigned
for prompt injection. We instantiate this architecture across
two complementary dimensions: task-level and API-level.
(1) Task-Level Memory ( MTask).Task-level memory is
designed to capture cross-API orchestration strategies, teaching
the model how to coordinate multiple APIs to fulfill a specific
user requirement. Formally, a task-level memory entry is
defined as a tuple:
mTask=⟨r, c, f,A used, gTask⟩,(1)where the Index Layer consists of the historical natural
language requirement r, which is embedded into a dense vector
space for semantic similarity retrieval. The remaining elements
constitute the Content Layer, storing the essential execution
footprint and guidelines of this task: cis the generated code
solution; frepresents the runtime execution feedback (such as
a success signal or error traceback); Ausedis the specific set
of private APIs successfully invoked; and gTaskis a concise,
natural-language usage guideline distilled by the reflector,
providing high-level intent and cross-API collaboration
patterns that are often absent in static API documentation.
(2) API-Level Memory ( MAPI).While task-level memory
addresses orchestration, API-level memory is dedicated to
execution fidelity by capturing individual API usage constraints
and failure modes. Formally, an API-level memory entry is
defined as:
mAPI=⟨a,D a, Ca, Ga⟩,(2)
where the Index Layer is anchored by the unique API name
a. The Content Layer maintains an evolving profile of the
API, comprising: Da, which represents the static baseline API
documentation (including signature and source code); Ca, a
repository of concrete code snippets exclusively relevant to
this API that records successful invocations and failed attempts
along with their runtime error messages; and Ga, a dynamic
collection of extracted usage guidelines. Each guideline g∈G a
is associated with a dynamic weight w, enabling the framework
to effectively manage and refine the pool of available guidelines.
C. Guideline-Driven Code Generation
With the Mestablished, this section details the forward
pass of MEMCODER. As illustrated in Figure 3, the code
generation process is disentangled into two sequential stages:
dual-source API retrieval and context-augmented generation.

(1) Dual-Source API Retrieval.Existing RAG-based
methods typically use the natural language requirement rt(or
decomposed sub-requirements) to retrieve API documentation
D. However, this cross-modal matching between requirements
and code signatures often suffers from semantic gaps. As a
plug-and-play framework, MEMCODERaugments the existing
RAG pipeline with an additional retrieval path that leverages
historical task guidelines via task-to-task semantic matching.
Specifically, given the current requirement rt, we first retrieve
theK taskmost similar historical tasks fromM Task:
Mrt= Top-K task
m∈M Tasksim(r t, m.r),(3)
where sim(·,·) denotes the semantic similarity score. By
extracting the historically verified APIs ( m.A used) from these
retrieved entries, we identify a set of candidate APIs that have
successfully resolved similar requirements.
Concurrently, we follow the existing RAG-based method to
retrieve KdocAPIs directly from the documentation D, denoted
asAdoc. To ensure high recall, the final candidate API set Acand
is constructed by taking the union of the explicitly retrieved
documentation APIs ( Adoc) and the implicitly verified APIs
extracted from the retrieved tasks (S
m∈M rtm.A used).
(2) Context-Augmented Generation.To ensure the
LLM understands how to orchestrate the identified APIs,
MEMCODERfurther retrieves the corresponding API-level
memory entry m(a)
APIfor each a∈ A cand. We then construct a
structured context Ctby concatenating the multi-dimensional
memory components with the requirement:
Ct=h
m.c, m.f, m.g Task	
m;
m(a)
API	
a;rti
,(4)
where [·;·] denotes sequence concatenation, and m∈ M rt
represents each retrieved task-level memory entry. In this
unified context Ct, the API-level usage guidelines and
code snippets prevent common invocation errors, while
the task-level guidelines provide cross-API orchestration
strategies. Finally, the LLM takes Ctas input to synthesize
the code ct. By conditioning on both dimensions of evolving
memory, MEMCODEReffectively improves the accuracy of
private-library-oriented code generation.
D. Feedback-Driven Memory Evolution
In the backward phase following generation, MEMCODER
executes the generated code ctto obtain objective feedback
ft∈ {Success,Failure} . The primary objective of this evolution
phase is to build a generalizable and actionable knowledge
base by distilling the execution trajectory (ct, ft)into concise
Usage Guidelines. While raw code snippets and their execution
outcomes are concurrently archived into Cato serve as few-
shot demonstrations, the LLM Reflector focuses on evolving
the existing memory Mthrough two core mechanisms:
guideline refinement and dynamic weight updating.
(1) Guideline Extraction and Refinement.Upon receiving
ft, the reflector first updates the task-level memory MTask.
It identifies the APIs Actinvoked in the generated code ct,
filtering out any APIs not present in the official documentationD. Guided by ft, the reflector distills a task-level usage
guideline gTask. This reflection explicitly analyzes the
combinatorial relationships, data dependencies, and execution
order among the invoked APIs, identifying robust coordination
patterns or diagnosing conflicting usage that led to failure.
The new memory entrym Taskis then appended toM Task.
For API-level knowledge, MEMCODERintroduces a re-
finement protocol to maintain MAPI. For each invoked API
a∈ A ct, the reflector generates a candidate guideline ˆg
detailing technical nuances that contributed to the outcome.
To prevent memory bloat and resolve knowledge conflicts, the
reflector evaluates ˆgagainst the documentation Daand existing
guidelinesG avia three routing actions:
•Discard: If ˆgis redundant or already covered by Da, it is
discarded to maintain memory compactness.
•Delete: If ˆginvalidates an existing guideline in Ga(e.g.,
correcting an outdated workaround), the obsolete entry is
marked for deletion.
•Add: If ˆgrepresents a novel, verified insight, it is added to
Gawith an initialized weight.
(2) Dynamic Weight Updating.To handle the varying reli-
ability of distilled knowledge across different contexts, MEM-
CODERimplements a feedback-driven credit assignment mech-
anism for Ga. When a new usage guideline is first extracted and
added to the memory, it is assigned an initial positive weight
winit. During the retrieval phase, guidelines are prioritized and
selected based on their current weights. To evaluate the utility of
these insights, we track the specific subset of guidelines Gused⊆
Gathat were injected into the contextC tfor a given task.
Once the execution feedback ftis obtained, MEMCODER
dynamically adjusts the weight wtof each guideline g∈G used
to compute its updated weightw t+1:
wt+1=(
wt+ ∆w+,iff t=Success,
max(w min, wt−∆w−),iff t=Failure,(5)
where ∆w+and∆w−are the reward and penalty step sizes
respectively, and wmin>0 acts as a strict floor threshold for
weight degradation. Instead of physically erasing penalized
guidelines, they are retained in the memory with the lowest
retrieval priority bounded by wminto prevent the model from
repeating past mistakes.
V. EXPERIMENTALSETUP
To assess the effectiveness of MEMCODER, we conduct
a comprehensive study designed to address four key Research
Questions (RQs). This section details our experimental configu-
ration, including the formal definition of RQs, the benchmarks
utilized for evaluation, the selection of metrics, the competitive
baselines across different categories, the large language
models (LLMs) employed, and specific implementation details
regarding retrieval and memory management.
A. Research Questions
Our study aims to answer the following RQs.
RQ1: How effective is MEMCODERin improving private-
library-oriented code generation?This research question

evaluates the practical utility of MEMCODERas a plug-and-
play enhancement for existing RAG systems. By applying
MEMCODERto established baselines, we examine whether
augmenting static API documentation with our evolving
Usage Guidelinesleads to measurable performance gains. This
analysis allows us to verify if the autonomous accumulation
of task-level and API-level guidelines can successfully bridge
the knowledge gap that static documentation fails to address.
RQ2: How does MEMCODERperform in private-library-
oriented code generation compared to existing CL-based
methods?To evaluate MEMCODER’s evolving capability,
we adapt representative memory-based CL methods to the
private-library code generation task as baselines. By comparing
them across a continuous stream of tasks, we assess whether
our method provides superior adaptation and domain-specific
knowledge retention for evolving private libraries.
RQ3: What are the individual contributions of the core
components in MEMCODER?To dissect the inner workings
of MEMCODER, we conduct comprehensive ablation studies
to evaluate its multi-dimensional memory and feedback-driven
evolution. Specifically, we isolate the impact of task-level
and API-level memory, and compare our dynamic memory
refinement against a naive first-in-first-out accumulation
baseline (MEMCODER-Accum).
B. Benchmarks
We evaluate MEMCODERon two specialized benchmarks
designed for private-library-oriented code generation:
NdonnxEvalandNumbaEval[ 11]. Since real-world private
libraries are proprietary and inaccessible for public research,
we utilize these benchmarks tosimulateprivate-library
scenarios. These benchmarks target libraries released in 2024
and significantly developed throughout 2025, ensuring their
latest APIs are absent from the training corpora of the models
released before these dates to mitigate data contamination.
•NdonnxEval.This benchmark evaluates code generation
using the ndonnx library, an ONNX-based tensor library.
It consists of 169 manually curated programming instances.
On average, each instance requires the coordinated invoca-
tion of more than 4 distinct APIs and is rigorously verified
by over 9 unit tests to ensure functional correctness.
•NumbaEval.This benchmark focuses on code generation
with the numba-cuda library, which is used for high-
performance CUDA JIT compilation. It contains 187
manually curated instances involving complex algorithmic
requirements. Following the same rigorous construction
standards as NdonnxEval, each instance requires the coor-
dinated use of more than 4 distinct APIs on average and
is verified by over 9 unit tests to ensure high quality and
solvability.
C. Metrics
We use functional correctness and efficiency as our main
evaluation metrics. To reduce randomness and obtain more
reliable estimates, we compute the functional metrics using
standard unbiased estimators.•Pass@k ( k∈ {1,3,5} ).For each instance, we sample n≥k
candidate solutions (we use n= 10 ), execute the provided
test cases, and count the number of passing solutions c.
Following prior work [ 39,40], we compute Pass@k using
the unbiased estimator:
Pass@k=E instances"
1− n−c
k
 n
k#
.(6)
•Exec@k ( k∈ {1,3,5} ).As models often misuse private
APIs and trigger runtime failures, we report Exec@k to
measure basic executability. It is defined analogously to
Pass@k, except that a solution is counted as successful if
it runs to completion on the test inputs without raising any
runtime exceptions.
D. Baselines
We establish a diverse set of baselines to comprehensively
evaluate MEMCODERacross our research questions. These
baselines can be divided into two main categories: existing
retrieval-augmented generation (RAG) approaches designed
for private-library code generation, and representative
memory-based Continual Learning (CL) frameworks.
(1) RAG-based Methods:To evaluate the enhancement
capability of our framework, we select three representative RAG
pipelines as backbones, comparing their performance both with
and without the integration of MEMCODER.Naive RAG[ 6,7]
represents the standard RAG paradigm, which embeds the
comprehensive API documentation, retrieves relevant APIs
based on the task requirement, and directly injects their static
specifications into the prompt.EpiGen[ 8] introduces a task-
oriented retrieval strategy by utilizing an LLM to decompose
complex requirements into fine-grained subtasks, performing
independent API retrieval for each subtask to enhance the
relevance of the injected documentation.CAPIR[ 37] further
optimizes the retrieval precision by leveraging an LLM to rerank
and filter the retrieved candidate APIs, effectively reducing
context redundancy before final code generation. By integrating
MEMCODERinto these diverse backbones, we assess whether
our multi-dimensionalUsage Guidelinescan consistently
provide performance gains across different RAG architectures.
(2) CL-based Methods:To assess the evolving capability of
our approach over a continuous stream of tasks, we compare it
against two representative memory-based CL methods. Since
there are currently no CL frameworks specifically tailored
for evolving private libraries, we carefully adapt them to our
task scenario. To ensure a fair comparison, all CL baselines
are initially equipped with a Naive RAG module to retrieve
and inject the top- Krelevant APIs for each task.Dynamic
Cheatsheet (DC-RS)[ 38] introduces an adaptive external
memory to accumulate reusable problem-solving strategies
and code snippets across tasks. Specifically, it retrieves
relevant historical requirement-generation pairs and employs
an LLM curator to synthesize a task-specific cheat sheet prior
to generation, notably curating memory from its own past
outputs without requiring ground-truth execution feedback.
ReMem[ 13] serves as a task-level retrieval agent that stores

historical requirements, generated outputs, and execution feed-
back as unified memory entries. For a new task, it introduces
a dynamic “Think-Act-Refine” decision loop, empowering
the LLM to actively evaluate and reorganize the most similar
retrieved experiences before generating the final code. These
adapted baselines allow us to evaluate whether MEMCODER
provides superior adaptation and domain-specific knowledge
retention compared to existing continual learning paradigms.
E. Models
Given the strict confidentiality and data privacy requirements
inherent in private-library development, enterprise code
generation systems are typically deployed on-premise.
Therefore, we evaluate MEMCODERusing three widely
adopted, high-performance open-source large language models
(LLMs) that are highly suitable for local deployment:Qwen2.5-
Coder-7B-Instruct[ 12],Llama-3.1-8B-Instruct[ 21], and
DeepSeek-Coder-6.7B-Instruct[26].
These models represent the state-of-the-art in open-source
code intelligence at the 7B-8B parameter scale. Crucially,
the official knowledge cutoffs of these models precede the
release of the latest APIs evaluated inNdonnxEvaland
NumbaEval, thereby strictly preventing any potential data
leakage during their pre-training phases. For brevity, we
hereafter refer to them as Qwen2.5-Coder ,Llama-3.1 ,
andDeepSeek-Coder.
F . Implementation Details
We deploy all evaluated Large Language Models locally and
maintain strict consistency in inference hyperparameters. For
code generation in the main evaluations (RQ1 and RQ2), we set
the maximum generation length to 4096 tokens, the sampling
temperature to 0.7, and top_p to 0.95. For the ablation
study (RQ3), we evaluate Qwen-7B by reportingpass@5and
exec@5to provide a more robust assessment of the individual
contributions of our framework’s core components.
For all modules and baselines that involve semantic similarity
retrieval, we uniformly employ bge-base-en-v1.5 as the
core embedding model. Regarding the retrieval configurations,
any component relying on API documentation retrieval
(including the Naive RAG modules equipped in the CL
baselines) is set to retrieve the top-5 most relevant APIs. For
our proposed MEMCODER, memory retrieval is performed
at both the task and API levels. For each candidate API, the
framework retrieves 1 successful code snippet as a few-shot
demonstration, along with 3Usage Guidelinesthat encapsulate
reflections distilled from both successful and failed execution
feedback. In contrast, for the existing continual learning
baselines (DC-RS, ExpRAG, and ReMem), the retrieval
module is configured to fetch the top-3 most similar historical
interaction traces for each new requirement.
VI. EXPERIMENTALRESULTS
A. RQ1: Effectiveness of MEMCoder in Private-Library Code
Generation
This research question evaluates the practical effectiveness
of MEMCODERin enhancing existing RAG systems. Weintegrate MEMCODERinto three representative RAG
backbones (Naive RAG,EpiGen, andCAPIR) and measure
the performance growth across multiple LLMs.
Results.The results for all RAG-based approaches are reported
in Table II.
❶MEMCoder consistently yields substantial performance
gains across all evaluated RAG backbones.As shown in
Table II, regardless of the backbone or the underlying model,
the integration of MEMCODERleads to a significant leap
in correctness. For instance, on the NdonnxEval benchmark,
MEMCODERimproves thepass@1ofNaive RAG,EpiGen,
andCAPIRby 25.32%, 18.46%, and 19.23% respectively
using Qwen2.5-Coder . These universal gains demonstrate
that MEMCODERserves as a robust enhancement framework
for any documentation-based retrieval pipeline.
❷MEMCoder surpasses the benefits of complex retrieval
optimization strategies.Empirical comparisons show that
Naive RAGcombined with MEMCODERachieves superior
performance to standalone advanced RAG methods. For
example, on NumbaEval with Llama-3.1 ,Naive RAG
+MEMCODERachieves apass@1of 28.72%, whereas
EpiGenandCAPIRonly reach 7.86% and 5.72%, respectively.
This indicates that the task-level and API-level memory
stored in MEMCODERprovide more effective guidance for
code generation than the task decomposition or reranking
mechanisms used in existing pipelines.
Answer to RQ1:MEMCODERsignificantly enhances RAG sys-
tems, outperforming complex retrieval strategies by injecting task-
level and API-level memory to supplement static documentation.
B. RQ2: Effectiveness Compared to CL-based Methods
This research question evaluates the evolving capability of
MEMCODERacross a continuous task stream, comparing it
with representative memory-based Continual Learning (CL)
methods:DC-RSandReMem. All methods process tasks
sequentially and evolve their memory via execution feedback.
Results.Table III presents the performance across CL baselines.
❶MEMCODERdemonstrates superior knowledge retention
and adaptation in most scenarios.As shown in Table III,
MEMCODERconsistently achieves higher functional
correctness than existing CL baselines across the majority
of configurations. For Qwen2.5-Coder onNumbaEval ,
MEMCODERachieves apass@1of 33.74%, significantly
outperformingDC-RS(17.33%) andReMem(23.16%). On
Llama-3.1 , MEMCODERmaintains a decisive lead on both
benchmarks; notably on NumbaEval , itspass@1(28.72%)
is several times higher than that ofReMem(3.10%). This
indicates that our Multi-dimensional Evolving Memory
structure generalizes better than raw dialogue logs or simple
experience stacking when handling complex library logic.
❷By distilling guidelines from execution feedback,
MEMCODERachieves more robust evolving effects.The
stability of MEMCODERis particularly evident inexec@1. For
instance, on NumbaEval withLlama-3.1 , MEMCODER’s
exec@1reaches 63.21%, while all baseline methods remain

TABLE II
PERFORMANCE COMPARISON WITHRAG-BASED BASELINESIN TERMS OFpass@kANDexec@k(%)ONNdonnxEvalANDNumbaEval.
Model MethodNdonnxEval NumbaEval
pass@1 pass@3 pass@5 exec@1 exec@3 exec@5 pass@1 pass@3 pass@5 exec@1 exec@3 exec@5
Qwen2.5
-CoderNaive RAG 27.22 41.16 47.39 34.32 50.34 56.64 23.16 40.04 48.09 40.27 66.37 76.82
+MEMCODER 52.54 64.96 69.42 60.71 71.92 75.87 33.74 46.91 52.41 54.87 71.62 76.84
EpiGen 23.49 38.41 45.34 30.41 48.13 55.53 19.73 36.19 44.63 34.44 59.90 71.10
+MEMCODER 41.95 53.83 58.57 52.19 64.50 68.89 30.05 40.60 44.53 46.95 63.55 68.86
CAPIR 30.89 45.24 50.85 38.76 56.21 62.78 20.91 36.84 43.83 36.15 63.14 73.92
+MEMCODER 50.12 62.26 66.79 63.37 76.14 80.82 32.67 43.37 48.37 51.12 66.93 71.67
Llama-3.1Naive RAG 11.95 20.90 25.80 24.85 42.64 51.23 5.61 13.98 20.12 22.57 47.66 61.47
+MEMCODER 20.41 30.46 34.92 36.39 51.16 57.11 28.72 39.69 44.69 63.21 79.95 85.07
EpiGen 12.31 22.14 27.17 21.72 37.13 44.19 7.86 17.41 23.04 25.88 53.65 67.59
+MEMCODER 19.11 30.25 35.52 33.20 49.09 55.67 24.33 34.49 38.71 51.44 69.34 75.65
CAPIR 13.37 22.11 26.72 23.08 37.20 44.07 5.72 13.95 19.42 22.25 48.62 62.78
+MEMCODER 19.29 28.89 33.43 32.37 45.52 50.61 24.92 35.13 39.47 50.53 69.50 75.88
DeepSeek
-CoderNaive RAG 24.26 40.13 48.29 35.68 54.72 62.60 7.86 18.36 25.63 17.59 39.49 52.70
+MEMCODER 44.44 56.35 60.09 53.55 66.67 70.82 32.25 46.08 51.70 55.08 75.03 81.93
EpiGen 30.00 44.91 50.31 39.11 56.30 62.13 10.05 22.18 29.84 20.80 45.31 59.53
+MEMCODER 42.90 54.16 58.08 52.49 66.12 70.70 30.70 44.42 49.40 54.65 73.78 79.29
CAPIR 25.44 39.95 45.81 32.84 51.45 59.31 10.75 23.51 31.48 21.02 44.10 57.24
+MEMCODER 43.79 54.80 58.42 55.09 66.67 70.67 32.30 44.51 48.99 55.99 76.83 82.54
below 20.36%. This confirms that our framework effectively
guides the model to avoid repetitive pitfalls and correct
deep-seated invocation errors. In contrast, the static cheat
sheets inDC-RSor the simple retrieval mechanism inReMem
often fail to resolve complex library constraints due to the
lack of explicit reflections on execution failures, making it
difficult for the model to recover from erroneous states.
Answer to RQ2:MEMCODERoutperforms memory-based CL
methods by evolving its Multi-dimensional Evolving Memory,
enabling superior adaptation and execution correctness.
C. RQ3: Ablation Study
MEMCODERintroduces aMulti-dimensional Evolving Mem-
oryarchitecture and a feedback-driven evolution mechanism.
This RQ conducts comprehensive ablation studies to dissect
the individual contributions of these core components using
Qwen2.5-Coder.
Setting.We design three experimental variants to evaluate the
necessity of our memory layers and refinement protocol: (1)
MEMCODERw/o MTask: Disabling the task-level evolving
memory. This removes the retrieved historical code solutions
m.cand orchestration guidelines gTaskfrom the context, thereby
losing cross-API collaboration strategies. (2) MEMCODER
w/oMAPI: Removing the API-level evolving memory. This
omits the specific usage guidelinesG aand execution-derived
code snippets Cafor each candidate API, providing only
the static baseline documentation Da. (3) MEMCODER-
Accum: Replacing the feedback-driven evolution (the Reflector’s
distillation and weight updating) with a naive first-in-first-out
(FIFO) accumulation. All raw execution trajectories are directly
appended as Cawithout extracting high-density guidelines Ga
or resolving knowledge conflicts.Results.The results forpass@5andexec@5are detailed in
Table IV.
❶Both Task-level and API-level evolving memory are indis-
pensable for accurate code generation.Removing either mem-
ory dimension leads to distinct performance degradation. When
MTaskis removed (MEMCODERw/o MTask), the model loses
its ability to leverage historically verified orchestration patterns,
causingpass@5to drop to 58.66% on NdonnxEval and 46.34%
onNumbaEval . More critically, removing MAPI(MEMCODER
w/oMAPI) results in a catastrophic collapse in execution
fidelity. On NdonnxEval ,pass@5plummets from 69.42% to
38.12%, andexec@5drops to 45.44%. This confirms that static
documentation Dis insufficient for preventing execution errors,
highlighting the immense value of our extracted API-level
guidelinesG aand snippetsC ain ensuring execution fidelity.
❷Feedback-driven evolution is strictly superior to
naive accumulation.ComparingFullMEMCODERwith
MEMCODER-Accumreveals the necessity of our refinement
protocol. Under the naive accumulation strategy, the model
directly appends raw trajectories without distilling guidelines,
leading to severe knowledge collision. This causespass@5
onNdonnxEval to drop dramatically from 69.42% to 39.09%,
performing similarly to removing the API memory entirely.
In the more complex NumbaEval tasks, MEMCODER-Accum
achieves anexec@5of only 59.44%, whileFullMEMCODER
reaches 76.84%. These results demonstrate that distilling raw
feedback into structured guidelines is essential for maintaining
a noise-free and effective evolving memory.
Answer to RQ3:Multi-dimensional Evolving Memory ensures
orchestration and execution fidelity. Feedback-driven evolution
justifies its complexity by distilling trajectories into high-density
guidelines, vastly outperforming naive accumulation.

TABLE III
PERFORMANCE COMPARISON WITHCL-BASED BASELINESIN TERMS OFpass@kANDexec@k(%)ONNdonnxEvalANDNumbaEval
Model MethodNdonnxEval NumbaEval
pass@1 pass@3 pass@5 exec@1 exec@3 exec@5 pass@1 pass@3 pass@5 exec@1 exec@3 exec@5
Qwen2.5
-CoderReMem 24.67 37.33 42.44 32.07 48.30 55.01 23.16 40.12 48.48 44.5572.13 81.37
DC-RS 19.11 38.20 47.35 25.38 48.26 58.04 17.33 31.67 38.57 34.55 61.05 70.89
MEMCODER 52.54 64.96 69.42 60.71 71.92 75.87 33.74 46.91 52.41 54.87 71.62 76.84
Llama-3.1ReMem 8.17 16.23 20.83 20.36 36.95 44.72 3.10 8.34 12.66 19.79 45.14 59.98
DC-RS 7.87 17.51 22.99 14.62 31.97 41.63 1.93 5.02 7.49 11.28 26.21 35.71
MEMCODER 20.41 30.46 34.92 36.39 51.16 57.11 28.72 39.69 44.69 63.21 79.95 85.07
DeepSeek
-CoderReMem 28.11 47.83 55.74 42.4367.22 75.83 10.53 23.43 31.16 22.89 49.24 63.42
DC-RS 18.52 33.58 40.33 32.84 54.70 62.90 6.52 16.37 23.43 15.99 38.11 52.36
MEMCODER 44.44 56.35 60.09 53.55 66.67 70.82 32.25 46.08 51.70 55.08 75.03 81.93
TABLE IV
ABLATION STUDY OFMEMCODER ONNdonnxEvalANDNumbaEval
USINGQW E N2.5-CO D E R
SettingNdonnxEval NumbaEval
pass@5 exec@5 pass@5 exec@5
MEMCODERw/oM API 38.12 45.44 46.33 72.28
MEMCODERw/oM Task 58.66 67.79 46.34 66.09
MEMCODER-Accum 39.09 59.44 37.93 59.44
FullMEMCODER 69.42 75.87 52.41 76.84
VII. DISCUSSION
Threats to Validity.In this section, we summarize the identified
potential threats and limitations of our study as follows:
❶Computational Overhead and Inference Efficiency.A
potential challenge for MEMCODERis the additional com-
putational overhead introduced by the Guideline-Driven Code
Generation and Feedback-Driven Memory Evolution phases.
The framework requires injecting relevant memories into the
context during forward inference and performing guideline
extraction and refinement in the backward phase, which
inevitably leads to higher token consumption compared to
naive RAG. Taking the NumbaEval benchmark as an example,
MEMCODERconsumes an average of 17,111 additional total
tokens per task. Based on current mainstream LLM pricing, the
extra cost to process the entire NumbaEval benchmark (187
tasks) is approximately $0.46. These results demonstrate that
the overhead of MEMCODERis highly acceptable in practical
production environments.
❷Knowledge Conflict and Bloat in Long-term Memory.As
the task stream extends, the volume of evolving memory con-
tinues to grow, potentially posing threats to retrieval efficiency
and knowledge consistency. However, MEMCODERadopts
differentiated management strategies for different memory
dimensions. For task-level memory, continuous accumula-
tion is essential as it increases the probability of matching
historical tasks similar to new requirements. For API-level
memory, the static documentation does not accumulate over
time, and code snippets are managed by retaining only the
latest execution records to achieve logic coverage rather than
full injection. Furthermore,Usage Guidelinesare managed
through a specialized refinement protocol, includingDiscard,
Add, andDeleteactions. As demonstrated by the ablationstudy in Section VI-C , this evolving management mechanism
significantly outperforms MEMCODER-Accum. This proves
that MEMCODERcan effectively mitigate knowledge conflicts
during long-term accumulation while maintaining high knowl-
edge density, thereby ensuring the robustness of the system.
VIII. CONCLUSION
In this paper, we address the challenge ofPrivate-Library-
Oriented Code Generation, where LLMs struggle due to the
absence of proprietary APIs in their pre-training data. Our
motivation study reveals that even with perfect retrieval of static
API documentation, models frequently fail due to a lack of task-
level coordination patterns and API-level execution constraints.
To bridge this gap, we propose MEMCODER, a training-free
and plug-and-play framework that extends the retrieval source
from static documents to evolvingUsage Guidelines.
The core of MEMCODERis aMulti-dimensional Evolving
Memorythat systematically organizes task-level and API-level
Usage Guidelinesdistilled from runtime feedback. Through a
closed-loop mechanism of execution, reflection, and refinement,
MEMCODERautonomously accumulates these guidelines
to explicitly guide the model and prevent execution traps.
Extensive experiments on the NdonnxEval andNumbaEval
benchmarks demonstrate that MEMCODERconsistently
and significantly improves the performance of various RAG
backbones and outperforms existing memory-based continual
learning methods. By enabling the autonomous accumulation
and reuse of programming knowledge, MEMCODERprovides
a robust and evolving solution for code generation in private
software ecosystems.
REFERENCES
[1]J. Li, G. Li, Y . Li, and Z. Jin, “Structured chain-of-thought
prompting for code generation,”ACM Transactions on
Software Engineering and Methodology, vol. 34, no. 2,
pp. 1–23, 2025.
[2]S. Jiang, J. Li, H. Zong, H. Liu, H. Zhu, S. Hu, E. Li,
J. Ding, Y . Han, W. Ning,et al., “aixcoder-7b: A
lightweight and effective large language model for code
processing,” in2025 IEEE/ACM 47th International Con-
ference on Software Engineering: Software Engineering
in Practice (ICSE-SEIP), pp. 215–226, IEEE, 2025.

[3]C. Li, Y . Zhang, J. Li, L. Cai, and G. Li, “Beyond
autoregression: An empirical study of diffusion large
language models for code generation,”arXiv preprint
arXiv:2509.11252, 2025.
[4]L. Cai, Y . Ren, Y . Zhang, and J. Li, “Ai-driven self-
evolving software: A promising path toward software
automation,”arXiv preprint arXiv:2510.00591, 2025.
[5]Y . Wang, Y . Zhang, Z. Qin, C. Zhi, B. Li, F. Huang,
Y . Li, and S. Deng, “Exploracoder: Advancing code
generation for multiple unseen apis via planning and
chained exploration,” inProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 18124–18145, 2025.
[6]D. Zan, B. Chen, Z. Lin, B. Guan, W. Yongji, and J.-G.
Lou, “When language model meets private library,” in
Findings of the Association for Computational Linguistics:
EMNLP 2022, pp. 277–288, 2022.
[7]S. Zhou, U. Alon, F. F. Xu, Z. Jiang, and G. Neubig,
“Docprompting: Generating code by retrieving the docs,”
inThe Eleventh International Conference on Learning
Representations, 2022.
[8]S. Li, S. Li, H. Zhang, S. Li, K. Chen, J. Yuan, Y . Cao, and
L. Yang, “Epigen: An efficient multi-api code generation
framework under enterprise scenario,” inProceedings of
the 2024 Joint International Conference on Computational
Linguistics, Language Resources and Evaluation (LREC-
COLING 2024), pp. 6206–6215, 2024.
[9]QuantCo, “ndonnx (version 0.17.1).” https://pypi.org/pro
ject/ndonnx/0.17.1/, 2025.
[10] NVIDIA, “numba-cuda (version 0.27.0).” https://pypi.org
/project/numba-cuda/0.27.0/, 2026.
[11] Y . Zhang, C. Li, R. Chen, G. Yang, X. Jia, Y . Ren, and
J. Li, “To see is not to master: Teaching llms to use
private libraries for code generation,” 2026.
[12] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu,
J. Zhang, B. Yu, K. Dang, A. Yang, R. Men, F. Huang,
X. Ren, X. Ren, J. Zhou, and J. Lin, “Qwen2.5-coder
technical report,”CoRR, vol. abs/2409.12186, 2024.
[13] T. Wei, N. Sachdeva, B. Coleman, Z. He, Y . Bei, X. Ning,
M. Ai, Y . Li, J. He, E. H. Chi, C. Wang, S. Chen,
F. Pereira, W.-C. Kang, and D. Z. Cheng, “Evo-memory:
Benchmarking llm agent test-time learning with self-
evolving memory,” 2025.
[14] C. Qian, W. Liu, H. Liu, N. Chen, Y . Dang, J. Li, C. Yang,
W. Chen, Y . Su, X. Cong,et al., “Chatdev: Communicative
agents for software development,” inProceedings of the
62nd annual meeting of the association for computational
linguistics (volume 1: Long papers), pp. 15174–15186,
2024.
[15] L. Li, R. Wang, H. Song, Y . Mao, T. Zhang, Y . Wang,
J. Fan, Y . Zhang, J. Ye, C. Zhang,et al., “What papers
don’t tell you: Recovering tacit knowledge for automated
paper reproduction,”arXiv preprint arXiv:2603.01801,
2026.
[16] Y . Zhang, Y . Li, Y . Liu, J. Li, X. Jia, Z. Li, and G. Li,
“Lookahead-then-verify: Reliable constrained decodingfor diffusion llms under context-free grammars,”arXiv
preprint arXiv:2602.00612, 2026.
[17] J. Li, Y . Zhao, Y . Li, G. Li, and Z. Jin, “Acecoder:
An effective prompting technique specialized in code
generation,”ACM Transactions on Software Engineering
and Methodology, vol. 33, no. 8, pp. 1–26, 2024.
[18] L. Yang, Y . Liu, Y . Zhang, and J. Li, “Difftester: Acceler-
ating unit test generation for diffusion llms via repetitive
pattern,”arXiv preprint arXiv:2509.24975, 2025.
[19] A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh,
A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow,
A. Ananthram,et al., “Openai gpt-5 system card,”arXiv
preprint arXiv:2601.03267, 2025.
[20] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman,
A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes,
A. Radford,et al., “Gpt-4o system card,”arXiv preprint
arXiv:2410.21276, 2024.
[21] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Ka-
dian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten,
A. Vaughan,et al., “The llama 3 herd of models,”arXiv
preprint arXiv:2407.21783, 2024.
[22] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat,
X. E. Tan, Y . Adi, J. Liu, R. Sauvestre, T. Remez,et al.,
“Code llama: Open foundation models for code,”arXiv
preprint arXiv:2308.12950, 2023.
[23] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng,
B. Yu, C. Gao, C. Huang, C. Lv,et al., “Qwen3 technical
report,”arXiv preprint arXiv:2505.09388, 2025.
[24] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu,
J. Zhang, B. Yu, K. Lu,et al., “Qwen2. 5-coder technical
report,”arXiv preprint arXiv:2409.12186, 2024.
[25] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu,
C. Zhao, C. Deng, C. Zhang, C. Ruan,et al., “Deepseek-
v3 technical report,”arXiv preprint arXiv:2412.19437,
2024.
[26] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong,
W. Zhang, G. Chen, X. Bi, Y . Wu, Y . Li,et al.,
“Deepseek-coder: when the large language model meets
programming–the rise of code intelligence,”arXiv preprint
arXiv:2401.14196, 2024.
[27] C. Wang, Z. Chu, Z. Cheng, X. Yang, K. Qiu, Y . Wan,
Z. Zhao, X. Shi, and D. Chen, “Codesync: Synchronizing
large language models with dynamic code evolution at
scale,”arXiv preprint arXiv:2502.16645, 2025.
[28] G. Ou, Q. Zhang, S. Chen, A. Li, D. Xu, T. Luo, D. Dai,
C. Gao, L. Wang, J. Zhou, M. Liu, and Z. Zheng, “Unseen-
codebases-domain data synthesis and training based on
code graphs,” 2026.
[29] D. Zan, A. Yu, B. Shen, B. Chen, W. Li, Y . Gong, X. Chen,
Y . Yao, W. Luo, B. Guan,et al., “Diffcoder: Enhancing
large language model on api invocation via analogical
code exercises,”Proceedings of the ACM on Software
Engineering, vol. 1, no. FSE, pp. 406–426, 2024.
[30] X. Gu, M. Chen, Y . Lin, Y . Hu, H. Zhang, C. Wan, Z. Wei,
Y . Xu, and J. Wang, “On the effectiveness of large lan-
guage models in domain-specific code generation,”ACM

Transactions on Software Engineering and Methodology,
vol. 34, no. 3, pp. 1–22, 2025.
[31] J. Liu, Y . Zhang, D. Wang, Y . Li, and W. Dong,
“Think: Tackling api hallucinations in llms via injecting
knowledge,” in2025 IEEE International Conference on
Software Analysis, Evolution and Reengineering (SANER),
pp. 229–240, IEEE, 2025.
[32] M. Liu, T. Yang, Y . Lou, X. Du, Y . Wang, and X. Peng,
“Codegen4libs: A two-stage approach for library-oriented
code generation,” in2023 38th IEEE/ACM International
Conference on Automated Software Engineering (ASE),
pp. 434–445, IEEE, 2023.
[33] D. Zan, B. Chen, Y . Gong, J. Cao, F. Zhang, B. Wu,
B. Guan, Y . Yin, and Y . Wang, “Private-library-oriented
code generation with large language models,”Knowledge-
Based Systems, vol. 326, p. 113934, 2025.
[34] H. Li, L. Ding, M. Fang, and D. Tao, “Revisiting
catastrophic forgetting in large language model tuning,”
2024.
[35] Y . Luo, Z. Yang, F. Meng, Y . Li, J. Zhou, and Y . Zhang,
“An empirical study of catastrophic forgetting in large
language models during continual fine-tuning,” 2025.
[36] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel,
et al., “Retrieval-augmented generation for knowledge-
intensive nlp tasks,”Advances in neural information
processing systems, vol. 33, pp. 9459–9474, 2020.
[37] Z. Ma, S. An, B. Xie, and Z. Lin, “Compositional api
recommendation for library-oriented code generation,”
inProceedings of the 32nd IEEE/ACM International
Conference on Program Comprehension, pp. 87–98, 2024.
[38] M. Suzgun, M. Yuksekgonul, F. Bianchi, D. Jurafsky,
and J. Zou, “Dynamic cheatsheet: Test-time learning with
adaptive memory,” 2025.
[39] B. Chen, F. Zhang, A. Nguyen, D. Zan, Z. Lin, J.-G. Lou,
and W. Chen, “Codet: Code generation with generated
tests,”arXiv preprint arXiv:2207.10397, 2022.
[40] B. Athiwaratkun, S. K. Gouda, Z. Wang, X. Li, Y . Tian,
M. Tan, W. U. Ahmad, S. Wang, Q. Sun, M. Shang,et al.,
“Multi-lingual evaluation of code generation models,”
arXiv preprint arXiv:2210.14868, 2022.
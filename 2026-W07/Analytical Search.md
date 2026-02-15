# Analytical Search

**Authors**: Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, Qingyao Ai

**Published**: 2026-02-12 05:06:29

**PDF URL**: [https://arxiv.org/pdf/2602.11581v1](https://arxiv.org/pdf/2602.11581v1)

## Abstract
Analytical information needs, such as trend analysis and causal impact assessment, are prevalent across various domains including law, finance, science, and much more. However, existing information retrieval paradigms, whether based on relevance-oriented document ranking or retrieval-augmented generation (RAG) with large language models (LLMs), often struggle to meet the end-to-end requirements of such tasks at the corpus scale. They either emphasize information finding rather than end-to-end problem solving, or simply treat everything as naive question answering, offering limited control over reasoning, evidence usage, and verifiability. As a result, they struggle to support analytical queries that have diverse utility concepts and high accountability requirements.
  In this paper, we propose analytical search as a distinct and emerging search paradigm designed to fulfill these analytical information needs. Analytical search reframes search as an evidence-governed, process-oriented analytical workflow that explicitly models analytical intent, retrieves evidence for fusion, and produces verifiable conclusions through structured, multi-step inference. We position analytical search in contrast to existing paradigms, and present a unified system framework that integrates query understanding, recall-oriented retrieval, reasoning-aware fusion, and adaptive verification. We also discuss potential research directions for the construction of analytical search engines. In this way, we highlight the conceptual significance and practical importance of analytical search and call on efforts toward the next generation of search engines that support analytical information needs.

## Full Text


<!-- PDF content starts -->

Analytical Search
Yiteng Tu
DCST, Tsinghua University
Quan Cheng Laboratory
Beijing, China
tyt24@mails.tsinghua.edu.cnShuo Miao
DCST, Tsinghua University
Beijing, ChinaWeihang Su
DCST, Tsinghua University
Beijing, China
Yiqun Liu
DCST, Tsinghua University
Beijing, ChinaQingyao Ai*
Quan Cheng Laboratory
DCST, Tsinghua University
Beijing, China
aiqy@tsinghua.edu.cn
Abstract
Analytical information needs, such as trend analysis and causal
impact assessment, are prevalent across various domains including
law, finance, science, and much more. However, existing informa-
tion retrieval paradigms, whether based on relevance-oriented doc-
ument ranking or retrieval-augmented generation (RAG) with large
language models (LLMs), often struggle to meet the end-to-end
requirements of such tasks at the corpus scale. They either empha-
size information finding rather than end-to-end problem solving,
or simply treat everything as naive question answering, offering
limited control over reasoning, evidence usage, and verifiability.
As a result, they struggle to support analytical queries that have
diverse utility concepts and high accountability requirements.
In this paper, we proposeanalytical searchas a distinct and
emerging search paradigm designed to fulfill these analytical infor-
mation needs. Analytical search reframes search as an evidence-
governed, process-oriented analytical workflow that explicitly mod-
els analytical intent, retrieves evidence for fusion, and produces
verifiable conclusions through structured, multi-step inference. We
position analytical search in contrast to existing paradigms, and
present a unified system framework that integrates query under-
standing, recall-oriented retrieval, reasoning-aware fusion, and
adaptive verification. We also discuss potential research directions
for the construction of analytical search engines. In this way, we
highlight the conceptual significance and practical importance of
analytical search and call on efforts toward the next generation of
search engines that support analytical information needs.
CCS Concepts
•Computing methodologies →Artificial intelligence;•In-
formation systems→Retrieval tasks and goals.
Keywords
Analytical Search, Information Retrieval, Reasoning
1 Introduction
Information Retrieval (IR) has long been centered on a fundamental
goal: assisting users in accomplishing tasks by organizing, retriev-
ing, and utilizing information to satisfy their needs [ 2,28,39,49].
These needs span a broad spectrum, ranging from simple fact
lookup, complex exploratory and decision making process [ 29,42].Classical IR systems based on relevance ranking have been proven
to be highly effective for ad-hoc retrieval tasks in which the user
seeks documents that are topically relevant to a query [ 17,34,38,
53]. Even with the advent of conversational search [ 31,37] and
session search [ 5,6] systems where models can exploit the user
feedback or context information, the dominant interaction para-
digm remains largely unchanged: systems return a ranked list of
results, and users manually inspect, synthesize, and reason over the
retrieved information to complete their tasks [ 23]. This paradigm
is well crafted to support naiveinformation finding, where infor-
mation needs can be solved by examining one or a few relevant
documents that contain the answers to the user’s questions.
However, real-world information needs of users often go beyond
naive information finding [ 2,30,42,48]. In domains such as law,
finance, scientific research, and political analysis, users often pose
analytical, exploratory, and decision-oriented questions that require
synthesizing evidence across sources, performing comparisons or
aggregations, and drawing reasoned conclusions [ 12,25]. For in-
stance, queries such as "What was the total number of theft incidents
reported on public transit over the past year?" or "How did News A
influence Stock B?" may not be able to find answers from one or
a few documents, but requires collecting heterogeneous evidence,
aligning it along temporal or causal dimensions, and applying multi-
step reasoning to synthesize the final answer. We refer to these
types of needs asanalytical information needs. Historically,
to solve analytical problems, users must rely on a labor-intensive
workflow: issuing multiple search queries against search engines
or databases, manually filtering and validating retrieved materials,
and incrementally assembling evidence to support an analysis. This
process is not only time-consuming and cognitively demanding,
but also difficult to scale and reproduce.
Fortunately, recent advances in large language models (LLMs) [ 1,
3,10,14,50,52] have provided new opportunities for analytical in-
formation needs. The integration of LLMs with IR systems, through
paradigms such as retrieval-augmented generation (RAG) [ 4,11,
15,19,43,44], search agents [ 7,16,21,22], and tool-augmented
reasoning frameworks [ 20,27,33], has substantially expanded the
scope of tasks that modern IR systems can do. Similar to humans,
LLMs can understand both the deep semantics of text and filter doc-
uments based on the needs of input queries with complex reasoning
processes. They have the potential to largely replace human efforts
in analytical tasks, fully automate the labor-intensive workflowarXiv:2602.11581v1  [cs.IR]  12 Feb 2026

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, and Qingyao Ai*
that used to be necessary, and shift IR systems from an information
finding tool to a true problem-solving system.
Nevertheless, simply adapting LLMs or coupling them with re-
trieval components is insufficient for solving analytical information
needs in practice. Theoretically, analytical information needs could
be solved by using a strong LLM to examine all documents in a cor-
pus and synthesize results based on the needs of user queries. Empir-
ically, such a naive method is not feasible considering the extreme
computational cost of LLMs and the large scale of retrieval collec-
tions. In our preliminary experiment, employing Qwen-32B [ 3,52]
to identify instances of voluntary surrender in ten thousand le-
gal case documents requires over 3 hours on 2 A100 GPUs. This
inference cost is prohibitive for scaling to a corpus of millions of
documents. While equipping LLMs with existing search tools to con-
duct RAG [ 11,15,19] seems to be a straightforward solution, many
studies have shown that existing RAG paradigms are incapable in
such complex analytical tasks [ 12,25,46]. Their performance is
often limited by the small set of "top" documents, which are re-
trieved by models optimized for topical relevance but not analytical
value. More importantly, analytical information needs often involve
document filtering and analysis from multiple aspects that require
different types of querying and retrieval systems. Existing RAG
paradigms intentionally decouple the retrieval process from the
generation process, where LLMs don’t know how to effectively use
multiple types of retrieval tools, and retrieval systems don’t know
how to construct indexes and optimize the efficiency of the whole
analytical workflow.
In this paper, we argue that a new generation of search tech-
niques is needed for analytical information needs, which we refer
to asanalytical search. Analytical search moves beyond naive
relevance-oriented document retrieval [ 17,38] toward systematic
frameworks in which search is understood as a process of assem-
bling the whole picture rather than finding an isolated puzzle piece.
This means rethinking search engine design from multiple perspec-
tives, involving but not limited to: (1) Analytical query processing,
how to decompose analytical queries to actionable items, utilize
multiple types of retrieval tools, and adjust plans iteratively based
on preliminary results and feedback; (2) Evidence-oriented retrieval
pipeline, how to optimize recall in a multi-path retrieval workflow,
integrate the retrieved evidence while providing traceable and au-
ditable process signals, and conduct efficient evidence verification
with retrieval and generation models jointly; and (3) Dynamic and
reasoning-enhanced indexing, how to cache the reasoning process
of data analysis, adapt the index structure to online request distribu-
tions, and continually improve both the effectiveness and efficiency
of analytical search. To better illustrate the unique importance and
challenges of analytical search, we discuss and differentiate it from
related concepts such as RAG, deep research, agentic databases,
etc., and present an example system framework that captures the
core components and interactions required to support analytical
search. We outline open research challenges and future directions,
and hope to attract more community efforts to look into this critical
and relatively underexplored area of information retrieval.2 Scope, Objectives, and Positioning
2.1 Analytical Information Needs
A defining characteristic of analytical search lies in the nature of
the information needs it seeks to address.Analytical informa-
tion needsrefer to user queries that require systematic reasoning
over multiple pieces of evidence, including the explicit modeling
of implicit intent, the decomposition of complex problems, and
the synthesis and validation of conclusions rather than the simple
retrieval of isolated facts. They manifest in several recurring forms
that differ in their analytical depth, reasoning requirements, and
decision implications. Following the taxonomy widely used in data
analytics and statistics [8, 18, 41], we broadly group them into:
•Descriptive Analytical Needs. Descriptive analytics focuses
on understanding and summarizing what has happened, often
through aggregation, comparison, and pattern identification. It
aims to construct an interpretable representation of historical
data or evidence, frequently serving as the foundation for deeper
analysis. Descriptive queries may involve simple operations such
as counting, ranking, or filtering, but often extend to more com-
plex descriptive analyses, including temporal trends and multi-
faceted comparisons. Even when the underlying operations are
conceptually simple, there are still many challenges, such as
identifying appropriate data sources, aligning definitions, and
ensuring evidential consistency. Examples:
-How often did people get robbed on the bus or subway last year?
-Which cities experienced the highest growth in public transporta-
tion crime rates?
-What are the most frequently cited reasons for product returns
across different regions?
Instead of finding single facts, these queries often involve analysis
like structured aggregation across time, entities, and sources.
•Predictive Analytical Needs. Predictive analytics aims to pre-
dict future outcomes or latent relationships based on observed
evidence. Predictive queries require the system to move from
summarization to future inference. This often involves causal
reasoning, correlation analysis, or extrapolation from historical
patterns. Predictive analytical queries typically require identi-
fying relevant signals, assessing their reliability, and integrat-
ing evidence across heterogeneous sources. While full statistical
modeling may not always be needed, predictive analytical search
needs to reason about uncertainty, temporal dependency, and
potential confounding factors. Examples:
-Did the introduction of traffic cameras lead to a reduction in pedes-
trian accidents?
-How is News A likely to affect Stock B in the short term?
-Based on historical data, is the current trend in influenza cases
likely to continue?
These queries naturally incorporate causal analysis or correlation-
based reasoning, seek not only to describe surface phenomena,
but to infer underlying relationships and anticipate future states.
•Prescriptive Analytical Needs. Prescriptive analytics repre-
sents the most decision-oriented form of analytical reasoning.
Queries in this category aim to evaluate alternative actions, poli-
cies, or strategies, and to provide guidance on what should be
done under specific constraints or objectives. In analytical search,
prescriptive queries often build upon descriptive and predictive

Analytical Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
analyses, requiring integrated reasoning across evidence, out-
comes, and trade-offs. Such queries are especially prevalent in
domains such as policy analysis, business strategy, and opera-
tional decision-making. They demand not only accurate evidence
retrieval and inference, but also explicit modeling of objectives,
constraints, and evaluation criteria. Examples:
-Given recent market conditions, should an investor increase expo-
sure to Stock C?
-Which policy intervention would most effectively reduce theft in-
cidents on public transportation?
-Which COVID-19 mitigation strategy would best balance public
health outcomes and economic activity?
These queries require generating analytical insights to support
actionable conclusions, and the answers must be inherently con-
tingent, context-dependent, and justified through transparent
reasoning over evidence.
Together, these categories illustrate the breadth and complexity
of analytical information needs. More importantly, they expose a
set of fundamental challenges that arise when attempting to resolve
such needs in an automated or semi-automated manner. Unlike
conventional information-seeking tasks, the difficulty of analytical
information needs does not stem from any single component, but
from the compounded challenges distributed across the entire ana-
lytical workflow—from query understanding to evidence retrieval,
fusion, and validation:
•Implicit and Complex Analytical Intent. Analytical infor-
mation needs are frequently expressed in natural language that
underspecifies critical analytical assumptions, constraints, and
objectives. Users often omit temporal scopes, comparison base-
lines, evaluation criteria, or even the precise analytical goal. For
example, a query such as "Has policy A improved urban air
quality?" implicitly assumes a time window, a definition of im-
provement, and one or more baseline conditions. This creates
a fundamental challenge at the query understanding stage: sys-
tems must reconstruct latent analytical intent rather than merely
interpret surface-level query semantics. Failure to do so leads
to incomplete or misaligned downstream analysis. On the other
hand, analytical queries rarely correspond to a single retrievable
fact and require decomposition into multiple interdependent sub-
questions, such as identifying relevant entities, collecting longi-
tudinal evidence, performing comparisons, or estimating effects.
These sub-tasks are not independent: errors or omissions in early
decomposition can propagate through the analytical pipeline.
Designing systems that can reliably decompose complex analyti-
cal needs into executable and logically coherent sub-tasks also
remains a core challenge.
•Beyond Naive Ad-hoc Retrieval. At the evidence collecting
stage, analytical information needs impose substantially higher
demands than traditional ad-hoc search. Potentially useful evi-
dence is often sparse, distributed across heterogeneous sources,
and may be weakly aligned with the original query at the sur-
face level. Critical evidence may reside in structured databases,
statistical reports, policy documents, or unstructured narratives,
each requiring different retrieval strategies. Moreover, analyti-
cally important evidence may not be topically prominent, making
precision-oriented top- 𝑘retrieval insufficient. Ensuring sufficientrecall of analytically critical evidence, while managing noise and
scale, constitutes a central retrieval challenge.
•Reasoning-intensive Fusion. Even when relevant evidence is
successfully collected, it still requires fusion strategies that go
beyond simple aggregation or summarization. People or systems
must perform multi-step reasoning operations such as filtering,
temporal alignment, comparison across alternatives, causal attri-
bution, trend extrapolation, or trade-off analysis. These reasoning-
based fusion steps often depend on intermediate representations
and partial results, sometimes with the aid of external tools,
rather than a single pass over retrieved content. Designing fusion
mechanisms that are robust to incomplete or partially conflicting
evidence is also a major challenge.
•Rigorous Conclusion Verification. Analytical conclusions are
rarely self-evident from individual pieces of evidence. Instead,
they must be validated through consistency checking, cross-
source corroboration, or even sensitivity analysis. Conflicting
evidence, data quality issues, or ambiguous causal signals are
common in real-world analytical tasks. As a result, it imposes
strong requirements on verification: systems must not only gen-
erate conclusions, but also assess whether those conclusions are
sufficiently supported, identify potential weaknesses, and, when
necessary, trigger additional retrieval or revision.
2.2 Analytical Search
Analytical search is the search paradigm that aims to resolve analyt-
ical information needs by explicitly modeling reasoning demands,
retrieving and organizing evidence for analysis, and producing ver-
ifiable conclusions through structured, multi-step problem-solving.
Unlike traditional information retrieval systems that prioritize top-
ical relevance and ranked document lists, analytical search treats
search as an end-to-end analytical process. Its objective is not
merely to surface potentially relevant information, but to support
users in constructing justified conclusions through evidence-backed
reasoning. This shift gives rise to several characteristics:
•Conclusion-oriented. The primary goal of analytical search is
not to retrieve isolated facts, but to generate conclusions that
synthesize multiple pieces of evidence. Accordingly, an analytical
system is explicitly optimized for the correctness, completeness,
and justification of the conclusion, rather than the linguistic flu-
ency or stylistic quality of the response, unlike traditional QA
systems. While natural language generation remains an impor-
tant interface component, it is subordinated to the quality of
the underlying analysis. A concise, well-supported conclusion
is preferred over a verbose but weakly grounded answer. This
orientation reflects a shift from answer-centric evaluation toward
outcome-centric assessment, where the validity of conclusions
and the soundness of the supporting reasoning take precedence.
•Complex Relevance. In analytical search, relevance is no longer
determined primarily by surface-level lexical or semantic similar-
ity between queries and documents. Instead, relevance is defined
by a piece of information’s utility for reasoning and analysis. A
document may be topically related yet analytically irrelevant if
it does not contribute to answering a sub-question, supporting a
hypothesis, or constraining an inference. Conversely, evidence
that appears only weakly related at the surface level may be

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, and Qingyao Ai*
crucial for completing a reasoning chain. This notion of com-
plex relevance requires retrieval systems to assess documents
in terms of their logical role, evidential value, and compatibility
with downstream reasoning processes, fundamentally extending
the traditional IR relevance model.
•Evidence-governed. Finally, in analytical search, conclusions
produced by the system must be grounded in verifiable, traceable
evidence drawn from trusted data sources. Unlike generative
search systems that may prioritize fluent answer generation, an-
alytical search emphasizes evidential accountability: each claim,
comparison, or quantitative result should be supported by ex-
plicit evidence that can be inspected and validated. This property
is particularly critical in high-stakes domains such as policy anal-
ysis, finance, law, and scientific inquiry, where unsupported or
hallucinated outputs are unacceptable.
2.3 Differences with Existing Search Paradigms
Analytical search does not emerge in isolation; rather, it inter-
sects with several recent paradigms that combine retrieval, fusion,
and generation. However, despite superficial similarities, analyti-
cal search is conceptually and operationally distinct from existing
approaches. Here, we clarify these distinctions by contrasting ana-
lytical search with three closely related paradigms.
2.3.1 Analytical Search vs. Retrieval-Augmented Generation (RAG).
Retrieval-Augmented Generation (RAG) [ 4,11,15,19,43,44] was
originally proposed to improve the factual grounding of large lan-
guage model outputs by conditioning generation on retrieved doc-
uments [ 45,51]. In RAG-style systems, retrieval primarily serves as
supporting context for answer generation, mitigating hallucinations
and enhancing factual correctness. The dominant optimization tar-
get is the quality of the generated answer—typically measured by
fluency, relevance, or factual accuracy at the surface level.
Analytical search, on the other hand, departs from this framing
in several fundamental ways. It aims at optimizing problem-solving
correctness and traceability, prioritizing whether the final conclu-
sion is logically sound, evidentially supported, and reproducible.
Rather than treating retrieval as auxiliary input to a generator, an-
alytical search treats retrieval as evidence construction. Besides,
retrieved evidence is not merely prompts for generation but is ex-
plicitly selected, organized, and validated to support downstream
reasoning. It is expected to reason over retrieved evidence through
structured, multi-step processes such as decomposition, compari-
son, aggregation, and validation. In this paradigm, reasoning is a
first-class objective, not an implicit by-product of generation. As a
result, analytical search systems must explicitly model reasoning
demand and evidential roles—capabilities that are largely outside
the design scope of conventional RAG pipelines.
2.3.2 Analytical Search vs. Deep Research.
Deep research [ 13,22,54] systems are designed to support open-
ended exploration and knowledge accumulation, often aiming to
produce comprehensive reports, background summaries, or literature-
style syntheses. Their primary objective is breadth and coverage:
gathering diverse perspectives, accumulating information across
sources, and presenting coherent narratives that assist human un-
derstanding. Analytical search, in contrast, targets well-definedanalytical questions with the explicit goal of reaching a correct
and verifiable conclusion. Rather than maximizing coverage, an-
alytical search emphasizes sufficiency and relevance of evidence
with respect to a specific analytical objective. The task structure is
therefore markedly different. Deep research tasks are often loosely
structured and evolve dynamically during execution, with stop-
ping criteria that are implicit, heuristic, or user-driven. Analytical
search tasks, by contrast, are explicitly structured, with clear ana-
lytical goals, decomposed sub-tasks, and well-defined termination
conditions-such as satisfying evidential requirements or completing
a reasoning chain.
Differences also emerge in how reasoning and evidence are han-
dled. In deep research systems, reasoning is typically implicit and
narrative-driven, embedded within long-form synthesis that prior-
itizes coherence and readability. Evidence is often aggregated de-
scriptively rather than interrogated analytically. Analytical search
instead requires explicit, step-wise reasoning tightly coupled with
evidence selection, transformation, and validation. Each reason-
ing step is expected to serve a functional role in advancing the
analysis, rather than contributing to a general narrative. Accord-
ingly, the evaluation focus diverges. Deep research systems are
commonly assessed based on usefulness, coverage, and readability
of the synthesized output. Analytical search systems, by contrast,
are evaluated on the correctness of conclusions, the completeness
of critical evidence, and the traceability of the reasoning process
that leads from query to conclusion.
2.3.3 Analytical Search vs. Agentic Databases.
Agentic databases and database agents [ 12,46,47] represent an-
other related paradigm, emphasizing natural language interaction
with structured data systems. These systems typically operate over
well-defined, structured data and focus on translating user intent
into executable queries (e.g., SQL), optimizing query execution, and
returning precise results. Their strength lies in accurate query for-
mulation, efficient execution, and deterministic correctness within
a closed-world schema. Analytical search extends beyond this scope
in both data coverage and system objectives. Rather than operating
solely on structured databases, analytical search must function over
heterogeneous and evolving data environments, integrating struc-
tured records, semi-structured sources, unstructured text, and open
web data, while optimizing indexes within the online workflow.
It often requires combining quantitative results from databases
with qualitative evidence from documents, reports, or news articles,
exceeding the assumptions of most agentic database systems.
More fundamentally, analytical search reframes the role of search
itself. Instead of viewing search as query execution, analytical
search treats it as analysis orchestration. The system must coordi-
nate retrieval, fusion, validation, and possibly iterative refinement
across multiple data modalities and tools. Query execution is only
one component in a broader analytical pipeline whose ultimate
goal is not data access per se, but evidence-based problem solving.
3 Conceptual Framework
Based on the aforementioned characteristics and challenges, ana-
lytical search requires a system architecture that goes beyond tradi-
tional IR (+LLM) pipelines. Instead of treating search as a single-step
mapping from query to answer, analytical search should support

Analytical Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Natural 
Language 
Query
Database/ 
Corpus
Query Module
Intent & Constraint 
Extraction
Task Decomposition
Task PlanningQuery Reformulation
Structured 
Analysis PlanRetrieval Module
Unstructured 
TextStructured 
Data
Retrieval Index Library
Multi-channel Recall
Principle: Coverage > PrecisionSparse / Dense
RetrievalConditional 
Filtering
Evidence Standardization 
& Alignment
Candidate 
Evidence Set
(Minimize Redundancy)Fusion Module
Evidence Selection & 
Pruning
Evidence Aggregation 
& Fusion
Multi-step Reasoning
Logic
Traceable Conclusion
Temporal
 Quantitative
Evidence ConfidenceVerification Module
Process 
Consistency 
Check
Quantitative Results
Supporting Evidence 
UncertaintyAccepted?
Yes NoResult 
Confidence 
Check
Passed? Yes
Per-evidence LLM 
VerificationNo
Update Analytical PlanSupplement & Optimize Index based on Analytical Results
Figure 1: A conceptual framework for analytical search.
end-to-end analytical problem solving, in which intermediate task
structures, evidence states, and reasoning steps are explicit and
integral to system behavior. At a high level, analytical search can
be viewed as an analytical workflow that transforms a user’s natu-
ral language query into an explainable and verifiable conclusion.
This workflow proceeds through a sequence of stages:analytical
query→task modeling→evidence retrieval→fusion→verification
→conclusion(as shown in Figure 1). Each stage serves a distinct
module and produces explicit intermediate representations that can
be inspected, revised, or validated. By separating what needs to
be solved, what evidence is required, how evidence is fused, and
how conclusions are justified, this workflow enables transparency,
controllability, and robustness when addressing complex analyt-
ical information needs. In this section, we present a conceptual
framework that operationalizes this workflow, comprising four
core components: query modeling, evidence retrieval, reasoning-
intensive fusion, and adaptive verification, which collectively form
a coordinated system for evidence-grounded analytical problem-
solving. It distinguishes analytical search from conventional IR
systems that largely collapse analysis into relevance ranking.
3.1 Query Module
The query module serves as the system’s entry point and is respon-
sible for transforming complex, ambiguous, informal, or underspec-
ified natural language queries into explicit analytical (sub-)task
structures that the system can execute. Additionally, analytical
queries are often characterized by implicit intent, unspoken con-
straints, and non-expert terminology. Therefore, the query module
must go beyond keyword interpretation or intent classification and
instead construct a structured analytical plan that captures both
what the user is asking and what must be done to answer it. Its
core capabilities include:•Intent understanding and constraint extraction, identify-
ing analytical goals, target entities, temporal scopes, potential
constraints, comparison baselines, and evaluation criteria, even
when they are not explicitly stated.
•Task decomposition, breaking complex analytical queries into
logically coherent sub-tasks that can be addressed independently
and then recombined.
•Query reformulation, resolving ambiguity, normalizing col-
loquial or non-professional expressions, and making implicit
assumptions explicit so that downstream retrieval, fusion, and
reasoning are well-posed.
•Retrieval and tool planning, determining which data sources,
retrieval paradigms, or analytical tools are appropriate for each
sub-task and how they should be orchestrated.
The output of the query module is not simply rewritten queries,
but an executable analytical plan that guides subsequent stages.
3.2 Retrieval Module
In analytical search, retrieval plays a role that is fundamentally
different from that in traditional IR systems. Rather than function-
ing primarily as a mechanism for information access, the retrieval
module is responsible for task-conditioned evidence acquisition,
assembling the evidence necessary to support downstream fusion,
analysis, and reasoning. This module warrants explicit separation
because failures in analytical search are more often attributable to
missing, biased, or insufficient evidence than to shortcomings in
language generation. As a result, retrieval in analytical search prior-
itizes evidential completeness and analytical coverage over narrow
relevance ranking. Unlike conventional retrieval pipelines that em-
phasize precision-oriented top- 𝑘results and ranking sharpness,
analytical search adopts arecall-oriented perspective, ensuring
that all potentially relevant evidence needed for analysis is available
downstream. Moreover, retrieval is no longer confined to a single

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, and Qingyao Ai*
document type or data modality; instead, it operates over heteroge-
neous sources, encompassing unstructured text, semi-structured
documents, tables, statistical data, and structured records, all of
which may be jointly required to support a coherent analytical
conclusion. The retrieval module is also responsible for evidence
normalization and alignment, that is, harmonizing representations
across sources so that evidence can be meaningfully compared, ag-
gregated, or reasoned over. This module thus constructs an evidence
space tailored to the analytical task, rather than merely returning a
ranked list of documents.
3.3 Fusion Module
The fusion module constitutes the analytical core of the system,
operating over the retrieved evidence to perform structured, multi-
step inference that incrementally advances toward an analytical
conclusion. In analytical search, fusion is explicitly modeled as
a decision process with traceable intermediate states, where the
system autonomously selects among possible reasoning steps, in-
cluding analytical operations and utilizing external tools, based on
the evolving problem state and evidence context. These steps, such
as choosing which evidence to focus on, deciding when and how
to aggregate or filter information, or determining which analytical
operation to apply, are not pre-fixed stages but flexible reasoning
actions that are dynamically invoked as needed. Within this process,
the fusion module is responsible for identifying and aggregating
relevant evidence, executing logical, temporal, and quantitative
operations such as cross-period comparison, causal inference, trend
detection, and numerical computation through internal calculations
or by explicitly calling external tools such as coding and SQL. It also
ensures that all intermediate inferences and conclusions remain
firmly grounded in the available evidence. By making reasoning-
based fusion process explicit, stateful, and inspectable, this module
reduces the risk of hallucination and enables robust, explainable
analytical problem solving, clearly distinguishing analytical search
from purely generative approaches.
3.4 Verification Module
Verification is not an optional add-on in analytical search; it is
an intrinsic system component. Because analytical conclusions
often inform decisions or interpretations in high-stakes domains,
they must be auditable and trustworthy. The verification module
provides analytical accountability by examining both the reasoning
process and the resulting conclusions. Its core functions include:
•Process-level consistency checking, ensuring that reasoning
steps are logically coherent and that evidence is exploited appro-
priately.
•Result-level validation, confirming that conclusions are sup-
ported by sufficient and non-contradictory evidence.
•Adaptive control, triggering backtracking or additional retrieval
when evidence is insufficient or conflicting, while allowing veri-
fication to be skipped or minimized when confidence is already
high (e.g., when a query can be resolved by a simple, well-defined
database operation).
Through this adaptive verification mechanism, analytical search
balances rigor and efficiency, ensuring reliability without unneces-
sary computational overhead.3.5 An Illustrative End-to-End Example
To illustrate how the proposed modules interact in practice, consider
the query "Did the introduction of traffic cameras reduce pedestrian
accidents in City X over the past five years?" The query module
first reconstructs its implicit structure by identifying the interven-
tion variable (i.e., "camera deployment"), the outcome variable (i.e.,
"pedestrian accident frequency"), the temporal scope, and the need
for a before-and-after comparison with potential control factors. It
decomposes the task into sub-steps, including determining the cam-
era deployment timeline, retrieving structured accident statistics,
collecting relevant environment variables, and other contextual
information. The retrieval module then conducts recall-oriented,
multi-path evidence acquisition. For example, it may obtain struc-
tured data through executable database queries (e.g., Text-to-SQL
for yearly accident counts) and use sparse and dense retrieval meth-
ods to collect unstructured reports and policy documents. The
fusion module then performs temporal alignment and quantitative
aggregation, and invokes external analytical tools if needed (e.g.,
statistical testing or regression) to estimate whether accident rates
significantly declined after camera deployment. Finally, the verifi-
cation module checks the logical consistency, evidential sufficiency,
and cross-source agreements on the evidence, triggering additional
retrieval if necessary, and produces a traceable and confidence-
calibrated conclusion.
4Potential Research Directions and Challenges
Based on the conceptual framework introduced above, this section
outlines key research directions for realizing analytical search in
practice, together with the fundamental challenges that shape their
realization. We discuss a set of methodological perspectives and
their accompanying challenges and believe that addressing these
intertwined opportunities and constraints is central to building
robust, scalable, and accountable analytical search systems.
4.1 Reasoning as Sequential Decision Making
From a methodological perspective, analytical search can be natu-
rally formulated as a sequential decision-making problem, in which
the system incrementally constructs an analytical solution through
a series of interdependent reasoning actions. Rather than view-
ing query understanding, retrieval planning, evidence fusion, and
verification as loosely coupled stages optimized in isolation, ana-
lytical search treats them as a coordinated reasoning process that
spans the entire analytical workflow. At each step, the system must
decide how to interpret and decompose the query, from which
data source and in what manner the evidence is retrieved, how to
calculate and fuse the current evidence, and whether the current
conclusions require further validation or revision. Within this for-
mulation, analytical search can be viewed as an agent interacting
with an environment composed of heterogeneous data sources,
retrieval tools, intermediate evidence states, and evolving reason-
ing contexts. Actions correspond to analytical decisions such as
query decomposition and reformulation, retrieval planning and
control, evidence selection and aggregation, analytical inference,
and adaptive triggering of verification. The state captures the cur-
rent analytical context, including the task structure derived from

Analytical Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
the query, the evidence collected so far, intermediate reasoning re-
sults, and confidence estimates. This perspective explicitly models
analytical reasoning as a controlled, multi-step process rather than
an implicit by-product of text generation.
4.1.1 Structural Design.
Under the sequential decision-making view, analytical search is real-
ized through a coordinated chain of reasoning decisions distributed
across three core modules: thequery module, which initiates and
structures the analytical trajectory by interpreting intent and plan-
ning sub-tasks; thefusion module, which drives the progression
of analysis through stateful, multi-step reasoning over evidence;
and theverification module, which monitors, evaluates, and adap-
tively regulates the reasoning process to ensure correctness and
evidential sufficiency. Thequery moduleserves as the entry point
of analytical reasoning. Since analytical queries are often under-
specified, ambiguous, or expressed in non-expert language, making
direct retrieval ineffective, the query module has to perform active
reasoning over the query itself, transforming a natural language
request into an executable analytical plan, including identifying
the underlying analytical intent, extracting implicit constraints
and assumptions, decomposing the query into interdependent sub-
tasks, and reformulating these sub-tasks into well-posed retrieval or
analysis requests as discussed in §3.1. For example, quantitative ag-
gregation tasks may be routed to structured databases, while causal
or contextual analysis may require unstructured text collections or
external reports. Through this planning process, the query module
directly shapes the downstream evidence space and constrains the
reasoning paths available to the system. Thefusion module, in turn,
acts as the core reasoning agent that operates over the retrieved
evidence to advance the analysis toward a conclusion. Rather than
passively generating text conditioned on retrieved documents, the
fusion module performs a sequence of explicit reasoning actions,
such as selecting relevant evidence, aligning information across
time or entities, performing filtering, comparisons, or aggregations,
and drawing intermediate inferences. These actions are stateful and
interdependent: each reasoning step updates the analytical context
and informs subsequent decisions. Complex analytical queries typ-
ically require chaining multiple such steps, making long-horizon
planning and stability essential. To ensure correctness and robust-
ness, reasoning-based fusion in analytical search is tool-augmented
by design. The fusion module selectively invokes external tools,
such as SQL engines, code execution environments, or domain-
specific analytical functions, when precise computation, aggrega-
tion, or simulation is required. Tool invocation is also treated as
an explicit reasoning action rather than an opaque internal oper-
ation, allowing the system to offload deterministic computation
while maintaining control over the overall analytical logic. This
separation improves numerical accuracy, reproducibility, and in-
terpretability. Throughout this sequential reasoning process, the
verification moduleprovides adaptive analytical control. Verifica-
tion is not executed uniformly at every step; instead, it is triggered
when confidence is low, evidence is conflicting, or conclusions carry
high stakes. Verification actions may include consistency check-
ing across evidence sources, validation of intermediate results, or
backtracking to earlier stages to acquire additional evidence. Byintegrating verification into the reasoning loop, the system bal-
ances rigor and efficiency, avoiding both premature conclusions
and unnecessary computation.
4.1.2 Training Strategies and Challenges.
This sequential decision-making view enables end-to-end optimiza-
tion of the analytical search workflow. Reinforcement learning (RL)
and related optimization frameworks like GRPO [ 10,40] can be
applied to jointly optimize the aforementioned reasoning actions
of the three modules based on task-level rewards without an extra
reward model. Such reward signals can also be derived from mul-
tiple complementary criteria. For example, the most direct signal
comes from the correctness of the final analytical conclusion, re-
flecting whether the system ultimately resolves the analytical task
as intended. Beyond final outcomes, intermediate rewards can be
defined based on evidence quality, encouraging the selection of suf-
ficient, diverse, and non-contradictory evidence; reasoning stability,
favoring coherent and logically consistent reasoning paths; and
efficiency, penalizing unnecessary reasoning steps, excessive tool
invocations, or redundant verification when confidence is already
high. As a result, analytical search systems can learn to adapt their
reasoning strategies in an end-to-end manner across different ana-
lytical scenarios and metrics, determining when to explore broadly,
when to reason deeply, and when to terminate with a confident,
evidence-backed conclusion.
However, formulating analytical search as sequential decision
making also exposes a set of fundamental training challenges that
are intrinsic to this paradigm. Unlike conventional retrieval or gen-
eration tasks, as mentioned above, analytical reasoning policies are
learned under highly underspecified intent. As a result, training
signals must encourage the system to actively hypothesize, test, and
revise latent analytical intent through interaction with evidence,
rather than merely imitate fixed query–answer mappings. This
creates a tension between flexibility and overcommitment: models
must reason beyond what is explicitly stated, while avoiding spuri-
ous assumptions that lead the reasoning process astray. Designing
learning objectives and algorithms that support robust intent infer-
ence in such open-ended settings remains a central challenge for
sequential reasoning-based analytical search.
A closely related difficulty lies in the stability of long-horizon rea-
soning. The analytical workflow often requires extended decision
sequences that span query decomposition, multi-round retrieval,
tool invocation, evidence fusion, and verification. Errors introduced
early in the trajectory, such as incorrect task decomposition or bi-
ased retrieval decisions, can propagate and compound, ultimately
invalidating the conclusion. From a sequential decision-making
standpoint, this corresponds to the problem of maintaining policy
consistency and recoverability over long horizons in a partially ob-
servable and non-stationary environment. Therefore, an effective
training strategy should equip the agent with the ability to detect
and correct intermediate mistakes, revise earlier decisions when
new evidence emerges, and balance exploration and exploitation
across reasoning steps. Purely myopic or step-local optimization is
insufficient; instead, training must explicitly account for the cumu-
lative effects of early decisions on downstream outcomes.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, and Qingyao Ai*
Finally, it poses a distinctive challenge in learning from sparse
and delayed rewards. In many analytical tasks, meaningful super-
vision is only available at the end of the reasoning process, when
the correctness and defensibility of the ultimate conclusion can
be assessed. Intermediate reasoning steps, such as evidence selec-
tion, aggregation choices, or verification triggers, often lack explicit
ground-truth labels, making credit assignment inherently difficult.
Within the sequential decision-making framework, this necessitates
the design of reward structures and auxiliary learning signals that
align intermediate actions with long-term analytical objectives. Po-
tential approaches include incorporating evidence-quality rewards,
reasoning-consistency constraints, or efficiency-aware penalties
that discourage unnecessary retrieval, tool usage, or verification
when confidence is already high. Integrating such signals allows
analytical search systems to learn not only what conclusions to
reach, but how to reason efficiently and reliably toward them, re-
inforcing the view that training is inseparable from the system’s
sequential reasoning formulation.
4.2 Recall-Oriented Multi-Path Retrieval
A core methodological departure of analytical search from tradi-
tional IR systems lies in its recall-first orientation. Classical IR
systems are typically optimized for precision at top- 𝑘, under the
assumption that users can manually inspect a small set of highly
ranked results. Analytical search, however, shifts the burden of
analysis from the user to the system. As a consequence, missing
critical evidence is often more damaging than introducing moderate
amounts of noise. Analytical failures are therefore more likely to
arise from insufficient or biased evidence coverage than from imper-
fect ranking. This makesrecall, rather than precision, the primary
optimization objective in the retrieval stage of analytical search.
Besides, from an analytical perspective, relevance is not a binary
or surface-level notion. A document or data record that appears
only weakly relevant in isolation may become indispensable when
combined with other evidence in a reasoning chain. Analytical
queries frequently require triangulating facts across time, sources,
or modalities, and such triangulation is only possible if the relevant
evidence is present in the candidate pool. Consequently, analytical
search systems must prioritize evidential completeness, even at the
cost of admitting additional noise that can later be filtered through
reasoning and verification.
4.2.1 Retrieval Routing.
To support this objective, analytical search adopts a multi-path
retrieval strategy, in which different retrieval mechanisms are in-
voked depending on the structure of the query and the nature
of the underlying data. Rather than relying on a single retrieval
paradigm, the system dynamically routes sub-tasks through spe-
cialized retrieval paths that are best suited to their evidential re-
quirements. Forstructured data, analytical search emphasizes pre-
cise, executable retrieval. When the analytical task involves counts,
aggregations, or well-defined constraints, such as incident statis-
tics or financial indicators, the system first transforms natural lan-
guage sub-queries into structured representations through a Text-
to-SQL [ 9,24,26,36] model. These structured queries are then
executed directly against databases, enabling accurate retrieval of
quantitative evidence with well-defined semantics. This pathwayensures determinism and correctness for analytically critical op-
erations such as filtering, grouping, and aggregation, which are
difficult to approximate reliably through text-based retrieval alone.
On the other hand, forunstructured data, analytical search employs
sparse and dense retrieval methods to construct a broad candidate
evidence set. Sparse retrieval [ 34,38] is effective for capturing exact
matches, domain-specific terminology, and named entities, while
dense retrieval [ 17,53] provides semantic generalization across
paraphrases and contextual variations. Given the recall-first objec-
tive, these methods are used in a complementary manner to max-
imize coverage. However, since high recall inevitably introduces
noise, the system can incorporate a subsequent re-ranking [ 32,35]
stage that evaluates candidates not merely by topical relevance,
but by their reasoning value—that is, their potential contribution
to analytical reasoning, such as supporting comparisons, tempo-
ral alignment, or causal inference in the fusion module. Together,
these structured and unstructured retrieval paths form a unified,
recall-oriented retrieval framework tailored to analytical search. By
explicitly embracing recall as the dominant objective and routing
queries through modality-appropriate retrieval strategies, it estab-
lishes a robust evidential foundation upon which reliable reasoning
and verifiable conclusions can be built.
4.2.2 Challenges.
Despite its conceptual advantages, this retrieval approach also in-
troduces a set of non-trivial challenges. A primary difficulty lies
in the trade-off between evidential coverage and downstream ef-
ficiency. While maximizing recall is essential to avoid missing an-
alytically indispensable evidence, excessively low precision can
severely burden subsequent fusion and filtering stages. A noisy or
redundant evidence pool increases the difficulty of reasoning-based
fusion, amplifies the risk of spurious correlations, and leads to un-
necessary tool invocations and verification overhead. As a result,
recall-oriented retrieval cannot be treated as indiscriminate expan-
sion; it must still maintain a minimal level of analytical precision to
ensure that downstream reasoning remains tractable and stable. Be-
yond this precision–recall tension, multi-path retrieval introduces
additional coordination challenges: different retrieval paths often
return evidence with heterogeneous semantics, granularity, and
reliability. Aligning these results into a coherent evidence space,
resolving conflicts between sources, and avoiding systematic bias
toward any single retrieval path remain non-trivial problems. More-
over, analytical queries may dynamically shift retrieval priorities
as reasoning progresses, requiring retrieval policies that can adap-
tively rebalance paths rather than execute them in a fixed, one-shot
manner. Addressing these challenges is crucial for ensuring that
recall-oriented multi-path retrieval serves as an enabler, rather than
a bottleneck, for efficient and reliable analytical reasoning.
4.3 Dynamic and Task-Aware Index
Organization
Analytical search operates in settings where information needs are
open-ended, evolving, and often difficult to anticipate in advance. As
a result, the underlying index cannot be treated as a static structure
constructed once and queried indefinitely. Instead, analytical search
requires a dynamic and task-aware index organization in which
the index itself evolves in response to real analytical workloads.

Analytical Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
In this paradigm, the index embodies long-term analytical priors,
capturing accumulated knowledge about what kinds of information
and structures are repeatedly useful for analysis, while retrieval
mechanisms handle short-term task execution, selecting evidence
relevant to a specific query or sub-task. This distinction highlights
a fundamental shift from traditional IR systems, where the index is
largely fixed and query-independent. In analytical search, the index
functions as an evolving analytical substrate that is continuously
refined through interaction. Index evolution is driven by observed
query patterns, reasoning trajectories, and evidence usage, allowing
the system to progressively internalize structural regularities that
are difficult to define a priori.
For structured data, this need is particularly evident. In many real-
world databases, it is impractical to predefine all possible columns
or analytical dimensions that users may require. Analytical queries
frequently introduce new perspectives, such as derived metrics,
composite attributes, or domain-specific categorizations, that are
absent from the original schema. Therefore, the analytical search
system can benefit from query-driven index evolution, in which
user queries and analytical plans trigger the discovery and con-
struction of new columns or views in the background. These dy-
namically induced structures allow future queries to be answered
more directly and reliably, effectively transforming latent analytical
concepts into explicit, indexed representations. A similar dynamic
also applies to unstructured text collections. Not all documents
contribute equally to analytical reasoning, and their importance
often becomes apparent only through repeated use in analytical
workflows. By observing interaction histories, such as which docu-
ments are frequently retrieved, selected during reasoning, or cited
in validated conclusions, analytical search systems can expand and
reorganize textual indexes to emphasize high-value content. Core,
representative, or frequently referenced documents can be enriched
with additional index entries, semantic annotations, or alternative
representations, making them more readily accessible to retrieval
components in subsequent analyses.
Collectively, these mechanisms redefine the index as more than a
passive data structure. In analytical search, the index becomes a liv-
ing representation of accumulated analytical experience, shaped by
real tasks rather than predefined assumptions. By enabling query-
driven index evolution, analytical search systems establish a feed-
back loop in which past analytical activity informs future efficiency,
robustness, and coverage—laying the groundwork for scalable, long-
term analytical intelligence. However, on the other hand, such an
indexing strategy also introduces practical challenges for analyti-
cal search. Aggressively evolving indexes in response to observed
analytical workloads can improve efficiency, but may also risk over-
fitting to historical query distributions, leading to brittle behavior
when analytical intents shift. Moreover, enriching indexes with de-
rived structures, annotations, or analytical views increases storage
and maintenance costs, and may blur the boundary between index-
ing and reasoning if not carefully controlled. Therefore, balancing
adaptability with stability, ensuring that index evolution captures
durable analytical value rather than transient patterns, remains a
key challenge for deploying analytical search at scale.4.4 Evaluation Principles
Evaluating analytical search systems requires moving beyond tradi-
tional relevance-based metrics that dominate classical IR evaluation.
Because analytical search is designed to solve complex analytical
problems rather than merely retrieve topically relevant documents,
its evaluation must reflect analytical contribution, reasoning quality,
and system efficiency. This necessitates a multidimensional evalua-
tion framework that captures both outcomes and processes. Here,
we provide five important evaluation dimensions for consideration:
•Correctness of the Conclusion. The primary evaluation crite-
rion in analytical search is whether the system reaches a correct
and reasonable conclusion. Unlike factoid question answering,
conclusions may involve aggregation, comparison, or inference,
and correctness must be assessed with respect to task-specific
ground truth, expert judgment, or validated analytical outcomes.
•Recall of Critical Evidence. Analytical search systems must
be evaluated on their ability to retrieve and utilize critical evi-
dence required for sound reasoning. Missing key evidence can
invalidate an otherwise plausible conclusion. Evaluation should
therefore emphasize evidential recall and coverage, particularly
for evidence that plays an essential role in the reasoning chain.
•Logical Consistency. Beyond the final conclusion, the inter-
nal reasoning process should be logically coherent. Intermediate
steps should follow valid inferential patterns, avoid contradic-
tions, and maintain consistency across evidence sources and an-
alytical assumptions. Logical inconsistency, even when leading
to a correct answer by chance, should be penalized.
•Traceability and Explainability. Analytical conclusions should
be traceable back to explicit evidence and reasoning steps. Evalu-
ation should assess whether the system can provide transparent
explanations that expose how evidence was selected, transformed,
and combined to support the final conclusion. This criterion is
especially important for trust, auditability, and error analysis.
•Efficiency. Finally, analytical search systems must be evaluated
on efficiency, including the number of reasoning steps, tool in-
vocations, retrieval rounds, and verification actions required to
reach the conclusion. Efficient analytical reasoning reflects not
only computational performance but also the system’s ability to
adaptively control depth and avoid unnecessary operations.
While these principles outline what should be evaluated, a core
difficulty is the absence of a single gold-standard reasoning path.
For many analytical tasks, multiple reasoning trajectories differ-
ing in decomposition strategies, evidence ordering, or analytical
operations may all lead to defensible conclusions. Consequently,
evaluation frameworks that assume a fixed reference process or a
unique correct output are ill-suited for analytical search. Systems
must be assessed in a manner that tolerates procedural diversity
while still enforcing constraints on evidential sufficiency, logical co-
herence, and analytical validity. Another inherent challenge arises
from the multiplicity and conditionality of valid outcomes. Since
analytical conclusions are often contingent on assumptions, scope
choices, or evidential framing that may not be uniquely determined
by the query alone. Different, yet reasonable, analytical interpreta-
tions can therefore yield different conclusions without any being
strictly incorrect. This characteristic undermines evaluation proto-
cols based purely on answer matching or surface-level correctness.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, and Qingyao Ai*
Therefore, evaluation must shift toward assessing whether a con-
clusion is justified given the evidence and assumptions made, and
whether those assumptions are explicitly surfaced and internally
consistent, further reinforcing the importance of evidence-level
and process-level evaluation, rather than treating the final answer
as an isolated artifact. In addition, human-in-the-loop evaluation
incorporating expert judgment may be unavoidable for complex an-
alytical tasks, especially in high-stakes domains where correctness,
accountability, and interpretability outweigh scalability concerns.
Together, these considerations position evaluation not merely as
a benchmarking exercise but as a significant component of ana-
lytical search system design, shaping how systems reason, justify
conclusions, and expose uncertainty.
5 Conclusions
In this paper, we introduce a novel search paradigm,analytical
search, which represents a distinct and increasingly important
information-seeking paradigm that cannot be adequately addressed
by traditional relevance-oriented IR systems or by straightforward
combinations of retrieval and generation. By formalizing analyti-
cal information needs, articulating a new analytical search para-
digm, and presenting a conceptual system framework together with
research directions and challenges, we highlight how analytical
search shifts the focus from naive information finding to evidence-
grounded problem solving, from answer fluency to conclusion cor-
rectness, and from static retrieval to reasoning-aware analytical
workflows. More broadly, analytical search serves as a unifying
framework across information retrieval, natural language process-
ing, and database systems, bringing together retrieval, reasoning,
structured querying, and verification under a common analytical
objective. As complex analytical information needs continue to pro-
liferate across various domains, addressing analytical queries and
building analytical search systems are both foundational challenges
and critical opportunities for the IR community. We therefore call
on the community to recognize analytical search as a first-class
research problem and to invest in its systematic study.
Acknowledgments
This work is supported by the Research Project of Quan Cheng
Laboratory, China (Grant No. QCL20250105)
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774
(2023).
[2]Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong
Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, et al .2023. Information retrieval
meets large language models: a strategic report from chinese ir community.AI
open4 (2023), 80–90.
[3]Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
Wenbin Ge, Yu Han, Fei Huang, et al .2023. Qwen technical report.arXiv preprint
arXiv:2309.16609(2023).
[4]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bog-
dan Damoc, Aidan Clark, et al .2022. Improving language models by retrieving
from trillions of tokens. InInternational conference on machine learning. PMLR,
2206–2240.
[5]Jia Chen, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2019. TianGong-
ST: A new dataset with large-scale refined real-world web search sessions. InProceedings of the 28th ACM International Conference on Information and Knowl-
edge Management. 2485–2488.
[6]Jia Chen, Weihao Wu, Jiaxin Mao, Beining Wang, Fan Zhang, and Yiqun Liu. 2022.
Overview of the NTCIR-16 session search (SS) task.Proceedings of NTCIR-16. to
appear(2022).
[7]Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng
Zhu, Haofen Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al .2025. Learn-
ing to reason with search for llms via reinforcement learning.arXiv preprint
arXiv:2503.19470(2025).
[8]Christo El Morr and Hossam Ali-Hassan. 2019. Descriptive, predictive, and
prescriptive analytics. InAnalytics in healthcare: a practical introduction. Springer,
31–55.
[9]Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and Jin-
gren Zhou. 2023. Text-to-sql empowered by large language models: A benchmark
evaluation.arXiv preprint arXiv:2308.15363(2023).
[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948(2025).
[11] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.
Retrieval augmented language model pre-training. InInternational conference on
machine learning. PMLR, 3929–3938.
[12] Chuxuan Hu, Maxwell Yang, James Weiland, Yeji Lim, Suhas Palawala, and Daniel
Kang. 2025. Drama: Unifying Data Retrieval and Analysis for Open-Domain
Analytic Queries.Proceedings of the ACM on Management of Data3, 6 (2025),
1–28.
[13] Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng
Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, et al .2025. Deep
research agents: A systematic examination and roadmap.arXiv preprint
arXiv:2506.18096(2025).
[14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card.arXiv preprint arXiv:2410.21276(2024).
[15] Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with
generative models for open domain question answering. InProceedings of the 16th
conference of the european chapter of the association for computational linguistics:
main volume. 874–880.
[16] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-r1: Training llms to reason
and leverage search engines with reinforcement learning.arXiv preprint
arXiv:2503.09516(2025).
[17] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for
Open-Domain Question Answering.. InEMNLP (1). 6769–6781.
[18] Katerina Lepenioti, Alexandros Bousdekis, Dimitris Apostolou, and Gregoris
Mentzas. 2020. Prescriptive analytics: Literature review and research challenges.
International Journal of Information Management50 (2020), 57–70.
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[20] Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu,
Zhoujun Li, Fei Huang, and Yongbin Li. 2023. Api-bank: A comprehensive
benchmark for tool-augmented llms.arXiv preprint arXiv:2304.08244(2023).
[21] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic search-enhanced large
reasoning models.arXiv preprint arXiv:2501.05366(2025).
[22] Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen,
Yutao Zhu, and Zhicheng Dou. 2025. Webthinker: Empowering large reasoning
models with deep research capability.arXiv preprint arXiv:2504.21776(2025).
[23] Jiqun Liu. 2021. Deconstructing search tasks in interactive information retrieval:
A systematic review of task dimensions and predictors.Information Processing &
Management58, 3 (2021), 102522.
[24] Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuxin Zhang, Ju Fan,
Guoliang Li, Nan Tang, and Yuyu Luo. 2025. A survey of text-to-sql in the era of
llms: Where are we, and where are we going?IEEE Transactions on Knowledge
and Data Engineering(2025).
[25] Qi Luo, Xiaonan Li, Tingshuo Fan, Xinchi Chen, and Xipeng Qiu. 2025. To-
wards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level
Reasoning.arXiv preprint arXiv:2510.26205(2025).
[26] Yuyu Luo, Guoliang Li, Ju Fan, Chengliang Chai, and Nan Tang. 2025. Natural
language to sql: State of the art and open problems.Proceedings of the VLDB
Endowment18, 12 (2025), 5466–5471.
[27] Yubo Ma, Zhibin Gou, Junheng Hao, Ruochen Xu, Shuohang Wang, Liang-
ming Pan, Yujiu Yang, Yixin Cao, Aixin Sun, Hany Awadalla, et al .2024. Scia-
gent: Tool-augmented language models for scientific reasoning.arXiv preprint
arXiv:2402.11451(2024).

Analytical Search Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[28] Christopher D Manning. 2008.Introduction to information retrieval. Syngress
Publishing.
[29] Gary Marchionini. 2006. Exploratory search: from finding to understanding.
Commun. ACM49, 4 (2006), 41–46.
[30] Donald Metzler, Yi Tay, Dara Bahri, and Marc Najork. 2021. Rethinking search:
making domain experts out of dilettantes. InAcm sigir forum, Vol. 55. ACM New
York, NY, USA, 1–27.
[31] Fengran Mo, Kelong Mao, Ziliang Zhao, Hongjin Qian, Haonan Chen, Yiruo
Cheng, Xiaoxi Li, Yutao Zhu, Zhicheng Dou, and Jian-Yun Nie. 2025. A survey
of conversational search.ACM Transactions on Information Systems43, 6 (2025),
1–50.
[32] Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019. Multi-stage
document ranking with BERT.arXiv preprint arXiv:1910.14424(2019).
[33] Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022. Talm: Tool augmented language
models.arXiv preprint arXiv:2205.12255(2022).
[34] Jay M Ponte and W Bruce Croft. 2017. A language modeling approach to in-
formation retrieval. InACM SIGIR Forum, Vol. 51. ACM New York, NY, USA,
202–208.
[35] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. Rankvicuna:
Zero-shot listwise document reranking with open-source large language models.
arXiv preprint arXiv:2309.15088(2023).
[36] Bowen Qin, Binyuan Hui, Lihan Wang, Min Yang, Jinyang Li, Binhua Li, Ruiying
Geng, Rongyu Cao, Jian Sun, Luo Si, et al .2022. A survey on text-to-sql parsing:
Concepts, methods, and future directions.arXiv preprint arXiv:2208.13629(2022).
[37] Filip Radlinski and Nick Craswell. 2017. A theoretical framework for conversa-
tional search. InProceedings of the 2017 conference on conference human informa-
tion interaction and retrieval. 117–126.
[38] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond.Foundations and Trends®in Information Retrieval
3, 4 (2009), 333–389.
[39] Chirag Shah and Ryen W White. 2025. From To-Do to Ta-Da: Transforming
Task-Focused IR with Generative AI. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
3911–3921.
[40] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al .2024. Deepseekmath: Pushing
the limits of mathematical reasoning in open language models.arXiv preprint
arXiv:2402.03300(2024).
[41] Ashish K Sharma, Durgesh M Sharma, Neha Purohit, Saroja Kumar Rout, and
Sangita A Sharma. 2022. Analytics techniques: descriptive analytics, predictive
analytics, and prescriptive analytics. InDecision intelligence analytics and the
implementation of strategic business management. Springer, 1–14.
[42] Georg Singer, Ulrich Norbisrath, and Dirk Lewandowski. 2013. Ordinary search
engine users carrying out complex search tasks.Journal of Information Science
39, 3 (2013), 346–358.
[43] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
dynamic retrieval augmented generation based on the information needs of large
language models.arXiv preprint arXiv:2403.10081(2024).
[44] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented
generation. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 1240–1250.
[45] Weihang Su, Changyue Wang, Qingyao Ai, Yiran Hu, Zhijing Wu, Yujia Zhou,
and Yiqun Liu. 2024. Unsupervised real-time hallucination detection based on the
internal states of large language models.arXiv preprint arXiv:2403.06448(2024).
[46] Ji Sun, Guoliang Li, Peiyao Zhou, Yihui Ma, Jingzhe Xu, and Yuan Li. 2025.
Agenticdata: An agentic data analytics system for heterogeneous data.arXiv
preprint arXiv:2508.05002(2025).
[47] Zirui Tang, Weizheng Wang, Zihang Zhou, Yang Jiao, Bangrui Xu, Boyu Niu,
Dayou Zhou, Xuanhe Zhou, Guoliang Li, Yeye He, et al .2025. Llm/agent-as-data-
analyst: A survey.arXiv preprint arXiv:2509.23988(2025).
[48] Zhiwen Tang and Grace Hui Yang. 2022. A re-classification of information seeking
tasks and their computational solutions.ACM Transactions on Information Systems
(TOIS)40, 4 (2022), 1–32.
[49] Robert S Taylor. 1962. The process of asking questions.American documentation
13, 4 (1962), 391–396.
[50] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models.arXiv
preprint arXiv:2302.13971(2023).
[51] Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and Qingyao Ai. 2025. RbFT: Ro-
bust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects.
arXiv preprint arXiv:2501.18365(2025).
[52] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[53] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Rep-
bert: Contextualized text embeddings for first-stage retrieval.arXiv preprintarXiv:2006.15498(2020).
[54] Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Huifeng
Guo, Yong Liu, and Xiangyu Zhao. 2025. Deep research: A survey of autonomous
research agents.arXiv preprint arXiv:2508.12752(2025).
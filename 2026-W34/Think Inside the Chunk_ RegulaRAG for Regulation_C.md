# Think Inside the Chunk: RegulaRAG for Regulation-Compliant Scenario Generation using LLMs: A Case Study of UN Regulation No. 152

**Authors**: Vahid Zolfaghari, Nenad Petrovic, AndrÉ Schamschurko, Alois Knoll

**Published**: 2026-08-17 10:46:33

**PDF URL**: [https://arxiv.org/pdf/2608.16394v1](https://arxiv.org/pdf/2608.16394v1)

## Abstract
Generating regulation-compliant test scenarios is essential for validating safety-critical automotive systems, yet Large Language Models (LLMs) struggle to ground outputs in long, hierarchical standards. We present RegulaRAG, a Retrieval-Augmented Generation (RAG) pipeline that couples SmartChunking, reference-aware enrichment of paragraphs and tables via graph traversal, with Smart Retrieve & Rerank over these enriched units. To test our system, we evaluate on a manually curated dataset covering all scenarios in UN Regulation No. 152 (AEBS). Our study comprises: (i) a three-step progressive search that identifies near-optimal retrieval parameters without exhaustive grid search; (ii) head-to-head comparisons against five baseline RAG systems; and (iii) a robustness stress test that scales the source corpus with distractor content. Outputs are evaluated using a customized penalized scoring metric. Across all experiments, RegulaRAG achieves the highest average Meta-Score (82.99), outperforming the next-best system by 43% (NoRAG: 57.94), while operating at 14k-25k tokens per query versus up to 500k for graphcentric baselines. It maintains strong performance, remaining stable even as the number of regulatory sources grows, whereas competing RAG systems degrade sharply in both quality and robustness.

## Full Text


<!-- PDF content starts -->

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1521
Think Inside the Chunk: RegulaRAG for
Regulation-Compliant Scenario Generation using
LLMs: A Case Study of UN Regulation No. 152
Vahid Zolfaghari,IEEE, Nenad Petrovic, Andr ´e Schamschurko, and Alois Knoll,Fellow, IEEE
Abstract—Generating regulation-compliant test scenarios is
essential for validating safety-critical automotive systems, yet
Large Language Models (LLMs) struggle to ground outputs
in long, hierarchical standards. We present RegulaRAG, a
Retrieval-Augmented Generation (RAG) pipeline that couples
SmartChunking, reference-aware enrichment of paragraphs and
tables via graph traversal, with Smart Retrieve & Rerank over
these enriched units. To test our system, we evaluate on a
manually curated dataset covering all scenarios in UN Regulation
No. 152 (AEBS). Our study comprises: (i) a three-step progressive
search that identifies near-optimal retrieval parameters without
exhaustive grid search; (ii) head-to-head comparisons against
five baseline RAG systems; and (iii) a robustness stress test
that scales the source corpus with distractor content. Outputs
are evaluated using a customized penalized scoring metric.
Across all experiments, RegulaRAG achieves the highest average
Meta-Score (82.99), outperforming the next-best system by 43%
(NoRAG: 57.94), while operating at 14k–25k tokens per query
versus up to 500k for graph-centric baselines. It maintains strong
performance, remaining stable even as the number of regulatory
sources grows, whereas competing RAG systems degrade sharply
in both quality and robustness.
Index Terms—Automotive software, Large Language Models,
Retrieval Augmented Generation, Standard compliance
I. INTRODUCTION
LARGE Language Models (LLMs) such as GPT-4o
demonstrate impressive capabilities in generating human-
like responses across various domains. However, LLMs face
challenges such as hallucinations, outdated knowledge, and
difficulty in recalling rare or highly specific information
[1], [2]. To overcome these limitations, Retrieval-Augmented
Generation (RAG) has emerged as a promising approach by
integrating external documents during inference to enhance
accuracy, traceability, and grounding [3].
In the context of automotive software development, par-
ticularly for Advanced Driver Assistance Systems (ADAS)
and Autonomous Driving Systems (ADS), generating precise,
regulation-compliant test scenarios is critical. Traditional man-
ual processes for interpreting and validating against regulatory
standards like UN Regulation No. 152 are time-consuming,
error-prone, and costly. LLMs have potential to automate
V . Zolfaghari, N. Petrovic, A. Schamschurko, and A. Knoll are with the
Chair of Robotics, Artificial Intelligence (AI) and Embedded Systems, Tech-
nical University of Munich, Munich, Germany (e-mail: v.zolfaghari@tum.de;
nenad.petrovic@tum.de; andre.schamschurko@tum.de; k@tum.de).
Corresponding author: Vahid Zolfaghari (e-mail: v.zolfaghari@tum.de).
This research was funded by the Federal Ministry of Research, Technology
and Space of Germany (BMFTR) as part of the CeCaS project, FKZ:
16ME0800K.this effort, but without RAG, they often struggle to identify
the correct numerical parameters, differentiate test conditions
(e.g., laden vs. unladen), and manage long documents ex-
ceeding context limits. Furthermore, directly inputting the full
regulation to an LLM results in high token usage and asso-
ciated costs. Although recent large language models (LLMs)
such as Claude provide extended context windows of up to
100k tokens [4], simply feeding full-length automotive docu-
ments (e.g., requirements or specifications) into the prompt is
rarely feasible due to confidentiality and intellectual property
concerns. Moreover, long-context models often struggle with
effective utilization of all input tokens, suffering from issues
such as context dilution or the so-called ”lost in the middle”
effect [5].
To address these challenges, we introduceRegulaRAG, a
smart two-stage Retrieval-Augmented Generation pipeline tai-
lored for regulation-compliant scenario generation. Regu-
laRAG employs a novelSmartChunkingstrategy to preprocess
PDF-based standards, identify hierarchical paragraph struc-
tures and nested references, and enrich document chunks via
a graph-based traversal. This ensures that retrieved content
includes all semantically linked paragraphs and tables, while
maintaining a compact token footprint. OurSmart Retrieve
and Rerankmodule performs query-aware retrieval over these
reference-enriched chunks, allowing the LLM to generate
accurate, traceable, and standards-compliant scenarios. This
method outperforms traditional retrieval pipelines that operate
on uninformed or semantically shallow chunks. An example
of such traditional methods is provided in [6], where a
simple Retrieve-and-Rerank strategy was applied for question
answering over automotive documents to support hardware
and software design workflows. The core methodological
novelty of RegulaRAG lies in two algorithmic contributions:
(i) reference-aware chunk enrichment via BFS graph traversal
over the regulation’s explicit cross-reference structure, and
(ii) retrieval and reranking against these enriched representa-
tions to surface dispersed regulatory evidence in a single step.
Bounded BFS depth, chunk deduplication, and metadata com-
pression areengineering optimizationsthat improve practical
scalability on long regulations without altering the underlying
algorithm; they are reported as such in Section III.
The remainder of this paper is structured as follows: Section
II reviews related work on LLMs for compliance and scenario
generation. Section III presents the RegulaRAG system and the
dataset. Section IV describes the experimental setup. Section V
presents the experiments and results. Section VI concludes the
arXiv:2608.16394v1  [cs.AI]  17 Aug 2026

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1522
paper.
II. LITERATUREREVIEW
The challenge of processing complex PDF documents, a
common format for technical specifications in the automotive
industry, has been addressed by several researchers. For ex-
ample, [7] propose a method that addresses key challenges
in automotive document processing, including multi-column
layouts and technical specifications. Following is the main
challenges in processing these documents:
1)Tables:Automotive standards often contain critical ta-
bles with specifications, test results, or compliance data.
These may span multiple pages in the PDF and get split
into separate tables after conversion, while in fact they
must be interpreted as one [7]. An example of these split
tables can be seen in Fig. 5.
2)Domain-specific terminology:Regulations use highly
technical and legal jargon, abbreviations, and region-
specific terms. Since RAG relies on similarity between
queries and chunks, such variation often causes it to miss
relevant content.
3)Lexical and syntactic variation:The same require-
ment can appear in different formulations, typically as
long, complex sentences with passive voice and nested
clauses. This makes retrieval and interpretation challeng-
ing [8] .
4)Nested and interlinked references:Requirements are
frequently split across multiple clauses and annexes that
point to one another. An example of these interlinked
references is illustrated in Figure 1. Standard RAG sys-
tems often retrieve isolated chunks, missing the linked
context.
Recent research has increasingly focused on leveraging
Large Language Models (LLMs) for compliance verification
across regulated domains. Sun et al. [9] propose a compli-
ance checking framework that integrates Retrieval-Augmented
Generation (RAG) with an “eventic graph” to align busi-
ness process descriptions with regulatory rules. Their method
demonstrates the benefits of combining structured knowledge
with retrieval-based augmentation, but it primarily produces
compliance verdicts rather than executable artifacts. Bolton
et al. [10] introduce DRAFT, a document-retrieval-augmented
fine-tuning approach for safety-critical software assessments.
By fine-tuning LLMs with both standards and distractors, their
method improves factual correctness and evidence handling.
However, the output remains focused on assessment text and
explanations rather than concrete test cases. LLMs and RAG
are also used for detecting inconsistencies, contradictions, and
conflicts in regulatory documents. Kumar and Roussinov [11]
explore the use of GPT-4 for these purposes. While effective
for semantic analysis at the document level, the method
does not extend toward structured instantiation of compliance
requirements. A complementary line of work seeks to translate
regulatory clauses into executable rules. Li et al. [12] present
Compliance-to-Code, which maps financial regulations into
programmatic logic to automate compliance checks. Althoughthis bridges textual requirements and computational enforce-
ment, it remains abstract and disconnected from domain-
specific simulation semantics.
Agarwal et al. [13] propose a multi-agent knowledge graph
framework for regulatory Quesion and Answer, where ex-
tracted triplets are combined with RAG-based retrieval to
improve factual correctness and traceability. This enhances
transparency in compliance reasoning but is not tailored to
generating domain-grounded artifacts such as test scenarios.
Compared to these approaches, our work targets a differ-
ent level of compliance verification. Rather than producing
static classifications, narrative explanations, or direct code
translations,RegulaRAGoperationalizes regulatory text into
simulation-ready test scenarios. By combining SmartChunk-
ing, reference-aware enrichment, and selective retrieval, our
method dynamically reconstructs scenario definitions from
regulatory clauses, tables, and cross-references in a traceable
manner.
A more recent line of work targetsstructure-awareretrieval
over complex documents. Xu et al. [14] propose RDR2, a
Retrieve–Document Route–Read pipeline in which an LLM-
based router navigates a document structure tree at inference
time to select or expand the most relevant heading nodes
before generation. RDR2 shows strong gains on general QA
benchmarks (TriviaQA, HotpotQA, ASQA), but its structure
prior is a heading hierarchy and does not address table
reconstruction across page boundaries, typed legal cross-
references, or normative dependencies between provisions.
Yu et al. [15] propose TableRAG, which addresses the loss
of table integrity in standard chunking by storing tables
in a relational database and enabling SQL-driven retrieval
alongside text-based search. TableRAG is the closest prior
work on the table-preservation problem, yet it targets docu-
ment QA over Wikipedia-style sources and does not handle
regulatory cross-references or downstream scenario synthesis.
Closer to the normative domain, Oliveira et al. [16] build a
knowledge graph over Portuguese legal resolutions with typed
nodes (Document, Article, Paragraph) and edges that explicitly
model amendment and revocation relations (Contain, Modify,
Revoke). Their system expands retrieved seed nodes to graph
neighbours but discards table and image content and does
not produce structured scenario artefacts. RegulaRAG differs
from all three in targeting regulation-specificscenario gener-
ation: it reconstructs fragmented PDF tables via a rule-based
heuristic, performs BFS reference-closure enrichment over
the regulation’s explicit cross-reference graph, and produces
compliance-grounded scenario text rather than short factual
answers.
Unlike knowledge-graph-based approaches, RegulaRAG
does not perform general entity recognition or relation ex-
traction. This is a deliberate design choice grounded in the
properties of regulatory text: legal language in standards such
as UN Regulation No. 152 relies on heavily nested clause
structures, conditional syntax, and domain-specific cross-
references that make reliable open-domain entity detection and
edge extraction challenging. Rather than building a general-
purpose KG, RegulaRAG exploits the regulation’sexplicit
cross-reference structure (numbered paragraph headings and

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1523
Fig. 1.Example of a reference and table chain inUN Regulation No. 152.
direct table pointers already embedded in the document) as a
lightweight, structure-preserving proxy for knowledge linkage.
This avoids the LLM-heavy NER, triple extraction, and graph-
expansion passes that drive the high token costs observed in
KG-based pipelines (see Section V-B for a direct comparison).
While we evaluate RegulaRAG on UN Regulation No. 152
[17]—which specifies the performance, testing procedures,
and safety requirements for Automated Emergency Braking
Systems (AEBS) in passenger vehicles—the pipeline is not
specific to AEBS or to Regulation 152. In principle, the same
workflow can be applied to other UN regulations that share
similar structural properties (hierarchical paragraphs, annexes,
and interlinked references) and can potentially be adapted
to other languages with very little modification. Applying
RegulaRAG to a different UN regulation primarily requires
updating the prompts, as the core algorithms for chunk en-
richment, de-duplication, and retrieval remain universal. This
shifts compliance validation from purely textual analysis to the
generation of executable test scenarios, bridging the gap be-
tween regulatory interpretation and practical system validation
in safety-critical testing workflows.
In the technical standards domain, CompliAT [18] presents
a structured approach for compliance checking of Assistive
Technology (AT) products against relevant standards. It fo-
cuses on verifying terminological consistency with standard-
defined terms, correctly classifying products, and validating
product specifications against applicable requirements. How-
ever, unlike these works, which primarily aim at classification
or static compliance checks, our work dynamically generates
test scenarios based on regulatory documents to validate
system behavior through simulation-based execution.
A parallel line of research applies LLMs to the au-
tomated generation of driving scenarios for ADS testing.
Chat2Scenario [19] extracts concrete simulation scenarios
from naturalistic datasets using GPT-4, while TARGET [20]
translates textual traffic rules into a formal DSL for test gen-
eration. LeGEND [21] derives scenarios from accident reports
to enhance diversity and realism, and LEADE [22] lever-ages multimodal prompting on traffic videos to reconstruct
functional scenarios where ADS may fail. OmniTester [23]
combines multimodal LLMs with user prompts to produce
diverse and critical scenarios. While these works focus on
creating challenging or diverse test cases to stress ADS in
simulation, our objective differs: rather than inventing novel
or corner-case scenarios, RegulaRAG systematically extracts
regulation-compliantscenarios directly from formal standards
such as UN Regulation No. 152, thereby prioritizing com-
pleteness, traceability, and compliance over diversity alone.
Because our method generates regulation-faithful scenarios
in natural language, they can serve as inputs to downstream
code-generation systems that set up simulations; for example,
Lebioda et al. [24] show that LLMs can translate abstract
requirements from automotive regulations into executable
CARLA configuration code.
In summary, while all these works demonstrate the power
of LLMs in test scenario generation, our approach is unique in
targeting the structured extraction of test scenarios from regu-
latory standards for compliance verification—bridging the gap
between scenario generation and formal safety requirements in
a traceable and automated manner.
III. SYSTEMDESCRIPTION
The main goal of the RegulaRAG pipeline is to extract and
structure the most relevant parts of lengthy regulation docu-
ments so that Large Language Models (LLMs) can generate
regulation-compliant test scenarios. An overview of the full
RegulaRAG pipeline can be seen in Figure 2. The pipeline
operates in three phases:
a) Phase 1: Extraction.:We use the Mineru tool [25] to
convert the regulation document from PDF into Markdown for-
mat and then normalize the extracted text to make it machine-
usable. Concretely, we convert legal paragraph numbers (e.g.,
5.2.1.4) into canonical Markdown headers, repair extraction
artifacts (line breaks, hyphenation), and standardize tables
while preserving the original numbering as stable IDs. The
resulting structure keeps chunk boundaries aligned with the

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1524
regulation’s paragraph hierarchy, makes cross-references de-
tectable for reference-aware enrichment, and provides cleaner
inputs for retrieval and re-ranking.
b) Phase 2: Chunking (SmartChunking).:Given the
structured Markdown produced in Phase 1, SmartChunking
first segments the text into semantically coherent base chunks
using LangChain’sRecursiveCharacterTextSplitter1, then
builds a reference-closure for each chunk by resolving cross-
referenced headings and tables via BFS, so every chunk is self-
contained with respect to the regulation’s internal dependen-
cies. We separate the core algorithmic steps from engineering
optimizations below.
Following is the core algorithm for SmartChunking:
1)Detect references (heading and table references).For
each chunk, we record three types of metadata using
rule-based patterns over the normalized text:
a)Defined Headings: Headings explicitly defined
within the chunk using Markdown syntax (lines
starting with#).
b)Referenced Headings: Identifiers of headings
(e.g., “2.3”) mentioned in the chunk but defined
elsewhere. These are resolved by locating the
defining chunk and linking it.
c)Referenced Tables: References to HTML-
formatted tables identified using heuristics (e.g.,
“the following table”). Chunks mentioning such
phrases are linked to the chunk containing the
actual table.
Heading references are resolved by locating the chunk
that defines the target heading ID. For table references,
we apply arule-based heuristic: any paragraph con-
taining a trigger phrase such as “the following table”
is treated as an explicit pointer to subsequent table
content. We then collect the consecutive<table>
HTML blocks that immediately follow the referencing
paragraph in Mineru’s Markdown/HTML output and
group them as a single logical table entry. This grouping
reconstructs PDF tables that were split across pages
and converted into multiple adjacent HTML fragments.
Each resolved reference is mapped to its target stable
regulation ID.
2)Build a document reference graph.We construct a
directed graph where nodes represent heading/table IDs
(and their defining chunks), and edges represent explicit
references from one node to another. This graph encodes
the regulation’s cross-reference structure.
3)Compute transitive reference closure via BFS.Af-
ter metadata extraction, we expand each chunk’s con-
text by traversing the reference graph via Breadth-
First Search (BFS), recursively collecting all referenced
headings/tables including nested chains (e.g., a chunk
referencing 6.6, which in turn references 6.3.2, as de-
picted in Figure 1). This produces a reference-closure set
of evidence associated with the base chunk. During this
stage, we apply inter-chunk deduplication to eliminate
1https://python.langchain.com/api reference/text splitters/character/
langchain text splitters.character.RecursiveCharacterTextSplitter.htmlredundant content. For instance, if a chunk references
two subparagraphs defined within the same chunk, we
ensure that the defining chunk is not added twice. This
deduplication ensures efficient context expansion and
prevents excessive token usage.
4)Create enriched chunk representations (base + clo-
sure).We form anenriched representationfor each
chunk by concatenating the base chunk with the text
of its reference-closure items (headings/tables). These
enriched representations are used as retrieval units so
that evidence dispersed across paragraphs and annexes
can be retrieved together.
To keep SmartChunking practical on long regulations, we
apply several Engineering optimizations that do not change
the underlying algorithm: (i)bounded BFS(we cap BFS
expansion to a maximum of 50 nodes per chunk, i.e.,
max_expand= 50, to prevent pathological expansion in
densely cross-referenced regulations); (ii)de-duplication(we
remove repeated referenced items both within a chunk’s
closure and across overlapping closures to avoid redundant
context); and (iii)metadata compression(we store references
as integer IDs/pointers instead of copying full referenced text
during preprocessing, materializing text only when assembling
retrieval contexts). We report these optimizations explicitly as
implementation choices aimed at improving runtime and token
efficiency.
c) Phase 3: Retrieval and Generation.:Smart Retrieve
and Rerank differs from standard top-kretrieval in that similar-
ity is computed between the query and each chunk’senriched
representation (base text concatenated with its reference-
closure), so that a query for a test condition can match a chunk
through its referenced table or sub-paragraph even when the
base text alone would score poorly; the retrieved base chunks
and their closure items are then assembled into the final LLM
context. In the third phase, we embed all enriched chunk repre-
sentations withsentence-transformers/all-mpnet-base-v22
from the sentence-transformers library (v3.4.1), which maps
each sentence or paragraph to a 768-dimensional dense vector.
This model is based on MPNet and is widely used for semantic
search and dense retrieval due to its strong performance on
sentence-level similarity benchmarks. We selected this model
for retrieval because it provides robust semantic matching
performance while maintaining moderate computational cost,
which is important for large regulatory corpora.
To compute similarity between the query and each enriched
chunk, we use the cosine similarity function from the scikit-
learn library.3Based on these scores, we retrieve the top-k
most relevant chunks.
After ranking enriched chunks and selecting the top-k
candidates, we construct the final context provided to the LLM
through a structured assembly procedure (see Fig. 2). This step
consists of three stages:
1)Reference Expansion.For each selected base chunk,
we retrieve its reference-closure set (headings and ta-
2https://huggingface.co/sentence-transformers/all-mpnet-base-v2
3https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.
cosine similarity.html

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1525
bles obtained via BFS in Phase 2). This ensures that
all semantically linked regulatory elements required to
instantiate a scenario are included.
2)Intra- and Inter-Chunk De-duplication.Because mul-
tiple selected chunks may reference the same paragraph
or table, we remove duplicate entries using their stable
regulation IDs. This guarantees that each referenced
element appears exactly once in the final context and
avoids unnecessary token overhead.
3)Canonical Ordering.The resulting set of base chunks
and referenced items is sorted according to the regu-
lation’s original hierarchical numbering (e.g., 5.2.1≺
5.2.1.1≺5.2.2). Tables inherit the position of their
defining paragraph ID. This ordering reconstructs the
logical flow of the regulation rather than the arbitrary
similarity-based retrieval order. The sorted chunks are
then concatenated in this canonical order to form the
final context window returned to the LLM.
This ordering step is crucial because similarity-based retrieval
alone may return relevant fragments in a non-coherent order.
By restoring canonical regulation order and merging refer-
enced tables with their defining clauses, RegulaRAG provides
the LLM with a logically structured evidence block, reducing
hallucinations and improving numerical consistency in gener-
ated scenarios.
Algorithm 1 formalises the complete context assembly
procedure.
Algorithm 1:Reference-Aware Context Assembly
(Smart Retrieve & Rerank)
Input:Queryq; enriched chunk setC Ewhere each chunkc
carries base text and reference-closureR(c); retrieval
encoderenc; retrieval breadthk.
Output:Ordered, de-duplicated context windowW.
Eq←enc(q);// embed query
foreachenriched chunkc∈ C Edo
Ec←enc(c.enriched text);
sc←cosine similarity(E q, Ec);
T ←argtopk{sc:c∈ C E};// select top-k
chunks
// Step 1 | Reference expansion
Texp← T;
foreachbase chunkc∈ Tdo
Texp← T exp∪R(c)
// Step 2 | De-duplication
Tuniq←deduplicate(T exp);
// Step 3 | Canonical ordering ;
W←sort(T uniq,key = regulation id);
returnW;
A. Dataset description
To evaluate the capabilities of Large Language Models
(LLMs) in generating test scenarios from regulatory standards,
we constructed a dataset derived from UN Regulation No. 152.
This dataset consists of manually extracted and categorized test
scenarios relevant to Advanced Emergency Braking Systems(AEBS). The dataset is summarized in table I. We release the
dataset publicly on Hugging Face atvahidzolf/un_152.
TABLE I
SUMMARY OFTESTSCENARIOS BYCATEGORY ANDCONDITION
Category Condition Num Scenarios
CtoStCunladen 12
laden 12
CtoMoCunladen 8
laden 7
CtoPMassRun 10
MaxMass 10
The scenarios are divided into three primary categories, each
representing a distinct type of test situation:
•Car-to-Stationary-Car (CtoStC): These scenarios assess
the AEBS functionality when a vehicle approaches a
stationary car. The tests include variations in vehicle load
(laden/unladen) and different approach speeds (e.g., 20
km/h, 42 km/h, 60 km/h).
•Car-to-Moving-Car (CtoMoC): This category focuses on
test cases where a vehicle interacts with another moving
vehicle, considering relative speeds and conditions spec-
ified in the standard.
•Car-to-Pedestrian (CtoP): These scenarios examine the
effectiveness of AEBS in detecting and mitigating col-
lisions with pedestrians crossing the road at predefined
speeds and trajectories.
Each scenario entry in the dataset contains a unique identi-
fier, test title, and a detailed textual description outlining the
specific test conditions.
a) Dataset construction and verification.:To construct
the ground-truth dataset, we systematically reviewed UN Reg-
ulation No. 152 and identified the three scenario categories
defined in the standard: CtoStC, CtoMoC, and CtoP. For each
category, we first extracted the general test conditions, which
are distributed across multiple sections of the regulation. For
example, the CtoStC conditions include the requirement from
Section 5.2.1.4 that “the subject vehicle shall approach the
lead vehicle in a straight line for at least 2 s before the
functional part of the test commences.” We then combined
these general conditions with the scenario-specific numerical
requirements for each test case. Based on this combined
information, we created a template scenario for each category,
then instantiated each template by varying the relevant pa-
rameters according to the values specified in the regulation.
For CtoStC, for instance, we varied approach speed using the
values listed in the “Maximum Relative Impact Speed (km/h)
for M1 vehicles” table and repeated the process separately
for laden and unladen vehicle conditions; the corresponding
post-condition requirements were filled from the applicable
regulation values for the stationary target case. The same
procedure was followed for CtoMoC and CtoP. After the
initial extraction and construction, all resulting scenarios were
reviewed for consistency with the regulation text and cross-
checked against the original source sections and tables to
confirm accuracy.

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1526
Fig. 2.RegulaRAG pipeline
B. Evaluation system
To evaluate the performance of different Large Language
Models (LLMs), we employ a structured scoring system based
on the dataset we have prepared. The generated test scenarios
from each LLM are compared against our manually curated
ground truth dataset to assess their accuracy and completeness.
The scoring process begins with the RAG system, which
retrieves relevant context from UN Regulation No. 152 to
generate test scenarios based on an initial prompt. The output
is then directly compared to the manually constructed ground
truth scenarios.
For evaluation of generated scenarios, we use a different em-
bedding model, namelysentence-transformers/all-MiniLM-
L6-v24from the same sentence-transformers library (v3.4.1),
which produces 384-dimensional embeddings. This model
differs from the retrieval model and is computationally lighter,
making it well-suited for large-scale pairwise similarity scor-
ing between generated and ground-truth scenarios. We use this
model to create vector embeddings for both the generated and
ground-truth scenarios.
The similarity measurement is conducted using cosine sim-
ilarity. We chose it as it is widely adopted in semantic text
comparison tasks due to its efficiency, scale-invariance, and
robustness to variations in text length [26].
One key challenge in LLM-generated scenarios is the in-
correct assignment of numerical values from the regulation
text. While LLMs tend to generate the general structure and
wording of the test scenarios correctly, they often fail to
extract and apply numerical values accurately. For example, a
scenario requires the subject vehicle to travel at 20 km/h (with
a tolerance of +0/-2 km/h) before braking. This speed value
4https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2should be derived from the regulation statement:”Tests shall
be conducted with a vehicle travelling at 20, 42, and 60 km/h
(with a tolerance of +0/-2 km/h). ”However, LLMs sometimes
generate incorrect speed values that are not supported by
the regulation text. Similarly, the post-condition requirement,
which dictates that the ”AEBS must decrease the vehicle speed
to 0 km/h”, should be extracted from the Maximum Relative
Impact Speed (km/h) for M1 vehicles table (see Figure 5).
LLMs often fail to distinguish between laden and unladen
conditions, leading to incorrect post-condition values in the
generated scenarios.
However, to ensure a more nuanced evaluation, we incor-
porate a penalization mechanism that accounts for critical
differences between the generated and reference scenarios.
This penalization ensures that minor lexical variations are
tolerated while significant discrepancies, such as omitted or
incorrect details, are appropriately reflected in the final eval-
uation. A related scoring challenge is that cosine similarity
scores tend to be very high, often close to 1.0, due to the
fact that the LLM correctly reproduces the scenario template
but fails to assign proper numerical values. To address this, we
implemented a regex-based matching approach that extracts all
numerical values along with the terms laden and unladen from
both the generated and ground truth scenarios. The extracted
values are then compared using string matching. While cosine
similarity might yield a score around 0.95, we introduce a
penalty weightλ= 0.2applied once per discrepancy identified
in the numerical values or laden/unladen conditions. This
adjustment ensures that LLMs producing incorrect numerical
values receive a final score below 0.9 (θin Algorithm 2
), which we classify as not equivalent to the ground truth.
This refined approach enhances the robustness of our scoring

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1527
mechanism, ensuring that the most critical numerical details
are accurately assessed and penalized when incorrect.
To make scenario matching sensitive to critical compliance
attributes that cosine similarity alone often misses (notably
numeric parameters and loading conditions), we extract two
structured key sets from each scenario text: (i) numeric values
and (ii) load/condition terms. Formally, we define two regular-
expression families:
Numeric pattern setR #.We extract signed integers and
decimals (supporting both decimal dots and commas) using:
R#:(?<!\w)[+\-]?\d+(?:[.,]\d+)?.
Prior to extraction, we normalize the text by (a) converting
decimal commas to decimal dots (e.g.,0,2→0.2), and (b)
splitting fused unit strings (e.g., “2s”→“2 s”).
Load/condition term setR ℓ.We extract regulation-relevant
categorical flags as case-insensitive keywords and de-duplicate
them:
Rℓ:\b(?:unladen|laden|MassRun|MaxMass|
Maximum\s+mass|Mass\s+in\s+running
\s+order|stationary|moving|stopped)\b
Given a scenario textx, the extraction function returns
K(x) = 
N(x), L(x)
, whereN(x)is the list of numbers
extracted byR #after normalization/masking, andL(x)is the
set of categorical terms extracted byR ℓ. We then compute
a discrepancy score∆between two scenarios by combining:
(i) a tolerance-based one-to-one matching F1 over numeric
lists (with absolute toleranceε absand relative toleranceε rel),
and (ii) Jaccard distance over the categorical term sets. This
discrepancy is converted into a penalized score. Concretely,
for a candidate pair of generated scenariohand ground-truth
scenariog:
s(h, g) =s cos(Eh, Eg)−λ∆ 
K(g), K(h)
(1)
wheres cosis the cosine similarity between sentence embed-
dingsE handE g,λis the penalty weight (defaultλ=
0.2), and∆is the combined numeric/categorical discrepancy
defined above. The complete matching and F1 computation
procedure is formalised in Algorithm 2.
a) Metric generalizability.:The evaluation framework
comprises two layers with different degrees of domain
specificity. The penalized F1 metric (Eq. 1) is structurally
regulation-agnostic: the cosine-similarity backbone, tolerance-
based numeric matching, and Jaccard-distance categorical
comparison can in principle be applied to any domain that
produces structured textual artefacts with verifiable numer-
ical parameters (e.g., pharmaceutical dosage specifications,
financial compliance documents, or aviation checklists). The
domain-specific elements are confined to the two regex fam-
ilies:R #captures generic numeric expressions and requires
minimal adaptation across domains, whereasR ℓencodes UN-
152-specific load-condition flags (laden,unladen,MassRun,
etc.) that must be redefined to match the categorical vocabulary
of another standard. The penalty weightλis a tunable hy-
perparameter, not a regulation-specific constant. Similarly, theAlgorithm 2:Penalized Scoring of Generated Scenar-
ios
Input:Ground-truth CSVG; Generated CSVH; threshold
θ; penalty weightλ(defaultλ= 0.2); regex setsR #
(numbers),R ℓ(load terms).
Output:Precision, Recall, F1; counts TP/FP/FN.
Load all rows fromGandH; keep Unique ID and text;
InitializeTP←0,FP←0,FN←0;
Mark allh∈ Has unmatched;
foreachground-truth scenariog∈ Gdo
Eg←encode(g.text);
Kg←extract(g.text;R #∪ Rℓ);// numbers +
load flags
s⋆← −∞,h⋆←∅;
foreachgenerated scenarioh∈ Hdo
Eh←encode(h.text);
Kh←extract(h.text;R #∪ Rℓ);
scos←cosine similarity(E h, Eg);
∆←diff(K g, Kh);// count
discrepancies
s←s cos−λ∆;// apply penalty:λper
discrepancy
ifs > s⋆then
s⋆←s;
h⋆←h;
ifs⋆≥θthen
Markh⋆as matched;
TP←TP + 1;
else
FN←FN + 1;
FP← |{h∈ H:hmatched}| −TP;
Precision←TP
TP+FP;
Recall←TP
TP+FN;
F1←2·Precision·Recall
Precision+Recall;
Meta-Score ( ¯F1−σ) is fully domain-agnostic: it summarises
mean performance and cross-condition stability using any
compatible base metric and is applicable whenever a system
is evaluated across multiple structured task categories. In
summary, adapting the metric to a new regulatory domain
requires (i) replacingR ℓwith domain-relevant categorical
keywords, and (ii) re-tuningλand the matching thresholdθ
on a representative sample; the rest of the framework transfers
without modification.
b) Compliance-oriented metrics.:Automotive regula-
tions do not define a universal compliance-accuracy metric;
compliance is usually assessed through regulation-specific
pass/fail criteria, tolerances, and traceability. In our evaluation,
these aspects are encoded in the manually constructed ground-
truth scenarios from UN Regulation No. 152. Thus, the
reported F1-score serves as a proxy for scenario-level com-
pliance, while the penalized scoring explicitly checks critical
numerical values (R #) and loading conditions (R ℓ). Dedicated
metrics such as Requirement Coverage, Parameter Accuracy,
and Traceability Coverage would provide additional insight
and are left as future work, informed by recent advances in
LLM-based requirements traceability [27]–[29].

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1528
IV. EXPERIMENTALSETUP
A. Language Models
We report results with three representative LLMs: (i)
gpt-4o-2024-11-20accessed via the OpenAI API5; (ii)
Llama 3.3 70B (“llama-3.3-70b-versatile”)served through
the Groq API6; and (iii)DeepSeek-chat(API model iden-
tifier:deepseek-chat) invoked via the vendor’s direct
API7. RegulaRAG operates upstream of generation: it de-
termines which parts of the regulation reach the prompt,
and the language model is an interchangeable component of
the pipeline rather than part of the contribution. The three
models were therefore selected to span distinct deployment
settings — a proprietary API model, an openly available
model, and a low-cost API model — rather than to identify a
best-performing generator, and pinned model snapshots (e.g.,
gpt-4o-2024-11-20) were preferred so that runs remain
reproducible. Because every RAG system is evaluated under
the same generator with identical prompts, scenario defini-
tions, and decoding settings (Section V-A2), the comparison
isolates the retrieval strategy, and a different or newer gen-
erator can be substituted without any change to the pipeline.
DeepSeek offers both areasoningmodel (DeepSeek-reasoner)
and achatmodel; We adopt the chat variant because, in our
setting, it achieves competitive accuracy while being over 12×
cheaper than the reasoning model ($0.14 vs. $1.74 per million
input tokens on a cache miss), making it the practical choice
for large-scale pipeline execution. Note that the DeepSeek
API does not expose a pinned checkpoint version for the
deepseek-chatendpoint; the underlying model may be
updated by the provider without a version change in the model
string. All models are run with low temperature (τ= 0.1) to
reduce output variance; decoding settings are held fixed across
methods. Retrieval hyperparameters (top kandchunk size)
are shared across all LLMs and determined via grid search, as
described in Section V-A1.
B. Evaluation Protocol
We report Precision, Recall, and F1 at the scenario level,
using F1 as the primary metric. Each failing run is au-
tomatically retried up to two additional times and marked
okupon the first successful completion. For corpus-scaling
experiments (Section V-B), we additionally record end-to-end
runtime. Scenario matching uses a cosine similarity threshold
ofθ= 0.9: a generated scenario is counted as a true positive
only if its penalized similarity to the best-matching ground-
truth scenario meets or exceeds this value. Numeric values
within scenarios are compared using an absolute tolerance
εabs= 0.05and a relative toleranceε rel= 0.05(5%). Due
to resource constraints, each (RAG system, LLM, scenario-
type) configuration was evaluated in a single run; trial-level
dispersion statistics across repeated executions are therefore
not reported. This limitation is explicitly acknowledged in
Section VI.
5https://platform.openai.com/docs/api-reference
6https://console.groq.com/docs/models
7https://api-docs.deepseek.com/User Prompt for Car-to-Moving-Car (Cto-
MoC)
You are an autonomous driving engineer spe-
cializing in AEBS testing for M1 category ve-
hicles. You must create test scenarios following
UN Regulation No. 152 exactly as specified,
with particular attention to: - Vehicle position-
ing and approach conditions - Precise speed
requirements including tolerances - Time To Col-
lision (TTC) specifications - Loading conditions
(laden/unladen) - Clear post-condition require-
ments
Fig. 3. System prompt for Car-to-Moving-Car (CtoMoC)
C. Compute and Reproducibility
Experiments were executed on a Linux workstation with
an NVIDIA GeForce GTX 1080 Ti (11 GB GDDR5X), an
Intel®Core™ i7-3770 CPU @ 3.40 GHz, and 32 GB RAM.
The software stack uses Python 3.10.12. Random seeds (where
applicable), prompt templates, and decoding parameters are
fixed across conditions. The ground-truth scenario dataset is
publicly available on Hugging Face (vahidzolf/un_152;
see Section III-A). Code, prompt templates, and evaluation
CSVs necessary to reproduce the results are publicly available
in the project repository8; API keys are required for commer-
cial models.
Two distinct sentence-transformer models from the
sentence-transformerslibrary (v3.4.1) are used at separate
pipeline stages:
•Retrieval:all-mpnet-base-v2(768-dim) — embeds en-
riched chunks and queries for similarity-based top-k
selection in Phase 3.
•Evaluation:all-MiniLM-L6-v2(384-dim) — computes
pairwise cosine similarity between generated and ground-
truth scenarios in the penalized scoring metric.
D. Prompting System
To obtain consistent outputs from all language models
(LLMs), we used identical prompts across all models without
applying any advanced prompting techniques. A single system
prompt was used to describe the general task, and three user
prompts were designed, each targeting a specific type of
scenario: Car-to-Stationary-Car, Car-to-Moving-Car, and Car-
to-Pedestrian.
These user prompts requested the generation of scenarios
according to the UN Regulation No. 152, each including a
single example to act as a structural template. You can find
the complete user prompts in Appendix A. We deliberately
avoided including instructions on how to derive the scenarios
in prompts, to keep the LLMs task-focused and to support
automated evaluation against a ground truth dataset. The
example format ensured that all outputs followed a consistent
structure, enabling effective comparison and scoring.
8https://gitlab.lrz.de/vahidev/retrieval-augmented-generation

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 1529
V. RESULTS ANDEXPERIMENTS
A. Experiment Design
To thoroughly evaluate our system, a staged experimental
design was adapted. The search space is large because multiple
pipeline components interact; therefore, we decompose the
problem into controllable factors and explore them system-
atically rather than exhaustively. Accordingly, the following
evaluation dimensions was defined, each capturing a distinct
aspect of the RAG pipeline that must be scrutinized to under-
stand performance, robustness, and scalability:
1)RAG systemWe evaluated six RAG systems:Regu-
laRAG(our method),R&R+RCS,NoRAG,OpenAI-
RAG,HippoRAG, andHybrid.
2)Scenario familyCar-to-Stationary-Car (CtoStC),
Car-to-Moving-Car (CtoMoC), and Car-to-Pedestrian
(CtoP). Section V-A2 reports the corresponding results
of this and previous item.
3)Retrieval breadth(top k)top k∈
{10,15,20,25,30}.
4)Chunking granularity(chunk size)
To assess the impact of chunk granularity on RAG
performance, the chunk size varied across a range of
values. Section V-A1 reports the corresponding results.
5)Source size (corpus scale)To assess robustness
under semantically similar distractors, we progressively
expand the retrieval corpus using eight UN regulations in
the automotive software domain, including UN Regula-
tion No. 152. This setup enables a systematic evaluation
of the scalability, strengths, and weaknesses of our
method relative to the strongest competing RAG base-
line. section V-B discusses the result of this experiment.
6)Language model (LLM)GPT-4o was used as
a widely adopted commercial model, and include
DeepSeek-Chat and Llama 3.3 to probe cost/latency and
open-weight generalization.
An exhaustive grid over all factors would be computa-
tionally prohibitive and confounds effects across dimensions.
Instead, we adopt a progressive design that (i) first identifies
stable hyperparameters for retrieval by performing grid search,
(ii) then compares RAG systems under matched conditions (to
isolate retrieval effects), and (iii) finally stresses the systems
with larger corpora (to assess scaling and distractor robust-
ness). This ordering controls nuisance variability and yields
conclusions that are both fair and reproducible. Following is
the sequence of experiments:
1) Hyperparameter selection via grid search:We estimate
robust retrieval hyperparameters by performing a full grid
sweep on the CtoStC scenario using UN-152 as the sole
source. We evaluate:
topk∈ {10,15,20,25,30}
chunk size∈ {800,1000,1400,1600,2000,3000,4000}for
our RegulaRAG method when using GPT-4o for the final
scenario generation. Figure 4 presents F1 as a heatmap
over the(chunk size, top k)grid. This pattern is consistent
with RAGGED [30], which finds that readers exhibiting an
improve-then-plateau response to retrieval depth — a trend
GPT-4o follows alongside GPT-3.5-turbo — incur substan-tially smaller performance loss when operating away from
their individually optimaltop kthan noise-sensitive readers
(e.g., LLaMA, Claude); this may partly explain whytop k=30
remained effective across the eight-regulation scalability ex-
periment (Section V-B). The resulting response surface shows
broad plateaus with a few sharp maxima. This plot reveals that,
despite a few outliers in the heatmap, the overall performance
trend stabilizes around(top k=30, chunk size=2000). At
this configuration the F1-score reaches its peak before grad-
ually degrading for larger values, indicating that it represents
a near-optimal balance between retrieval breadth and chunk
granularity. To ensure a fair comparison, all retrieval-based
baselines (R&R+RCS, OpenAI-RAG, HippoRAG, and Hy-
brid) were subsequently evaluated using the sametop k=30
value identified here.
2) RAG system comparison.:With(top k∗, chunk size∗)
fixed, we evaluated six Retrieval-Augmented Generation
(RAG) pipelines across all three scenario families (CtoStC,
CtoMoC, and CtoP) on UN–152:
•RegulaRAG— our method: SmartChunking with
reference-aware enrichment + smart retrieval: Smart Re-
trieve and Rerank which is integrated into our Regu-
laRAG pipeline. It compares the enriched chunks (in-
cluding referenced content) to the query, then gathers the
corresponding original chunks and appends the reference
chunks after deduplication.
•R&R+RCS (rr rcs)— baseline: RecursiveCharacter-
Splitter (RCS) + simple retrieval: this method uses theall-
mpnet-base-v29model, the same model as the one we
used for our regulaRAG method, to embed the chunks and
queries. The standard version performs retrieval directly
over the original chunks and returns the top-k chunks
based on similarity.
•OpenAI-RAG (openai rag)— baseline: RCS + OpenAI
embeddings + Chroma: In this method, we use OpenAI’s
text-embedding-ada-002model10to generate chunk
embeddings, which are stored in a Chroma vector
database. Retrieval is performed using LangChain’s vec-
tor store-backed retriever with the
similarity score threshold search type and parameters
search kwargs={”k”: 30, ”score threshold”: 0.1}.
•HippoRAG— baseline: As a state-of-the-art knowledge
graph-based RAG method, HippoRAG [31] is included
to benchmark our retrieval pipeline against recent graph-
based systems. It has shown competitive performance in
recent benchmarks.
•NoRAG— reference baseline: Direct LLM generation
without any retrieval context. The model receives only the
user prompt and scenario-format example, with no regu-
latory document chunks provided. Included to measure
the scenario-generation capability embedded in model
weights alone, quantifying the marginal benefit of the
retrieval pipeline.
•Hybrid— baseline: A hybrid retrieval pipeline that
combines sparse keyword-based (BM25) retrieval with
9https://huggingface.co/sentence-transformers/all-mpnet-base-v2
10https://platform.openai.com/docs/guides/embeddings

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15210
Fig. 4.Hyperparameter Selection
dense semantic (embedding-based) retrieval over RCS-
generated chunks, merging the two ranked lists before
context assembly.
This design isolates the contribution of the chunking and
retrieval strategy by keeping the scenario definitions, prompts,
and model-specific decoding settings constant. We report F1
score as the primary metric and track precision/recall for
diagnostic completeness. The goal is to establish whether
RegulaRAG’s reference-aware enrichment translates into con-
sistent gains across distinct task conditions. The results are
presented in Table II.↑arrows are for metrics where higher
is better and↓arrows are for metrics where lower is better.
To capture both the average performance and the stability
of each RAG system, we introduce aMeta-Score, computed
from the F1-scores of the three evaluated LLMs (GPT-4o,
DeepSeek-chat, and LLaMA-3.3). While Algorithm 2 de-
scribed the penalized evaluation of individual scenarios which
yields the F1-score, our goal here is to obtain a metric that
also reflects the variance of LLM performance and summa-
rizes how consistently a given RAG–LLM pipeline behaves
across different scenario categories. The Meta-Score therefore
provides a more holistic indicator of system-level robustness.
Let¯F1denote the mean of the three F1 values for 3 types
of scenarios for a given RAG system, and letσdenote their
standard deviation. The Meta-Score is then defined as:
MetaScore= ¯F1−σ.
We propose the Meta-Score as a robustness-adjusted aggrega-
tion designed for this study; it is not a standard benchmark
metric in RAG evaluation. Its formulation is inspired by the
classical mean–variance tradeoff principle [32] and the one-
standard-error model-selection heuristic of [33], both of which
penalize high-variance estimators by combining mean perfor-
mance with a dispersion penalty. We employ the coefficient of
variation (CV) as a normalized measure of relative variability,a well-established statistical tool for comparing dispersion
across scales.
This formulation rewards RAG systems with high average
F1 performance while penalizing those with high variability
across the three LLMs, thereby reflecting both accuracy and
robustness. Negative values in Table II arise naturally from the
definition of the Meta-Score.When a RAG system produces
highly inconsistent results across the three scenario categories
like scoring near 0 on one task while performing moderately
on others, the variance term becomes large and dominates the
average performance, yielding a negative Meta-Score.
Table II reveals three clear patterns. First,RegulaRAGcon-
sistently achieves the strongest overall performance and sta-
bility across all language models. Across GPT-4o, DeepSeek-
chat, and LLaMA-3.3, RegulaRAG attains the highest average
Meta-Score of82.99, clearly outperforming all competing
methods. In comparison, the second-best system (NoRAG)
achieves an average Meta-Score of 57.94, followed by Hybrid
(56.71) and HippoRAG (55.75).
This corresponds to an improvement of approximately43%.
At the individual model level, RegulaRAG maintains consis-
tently high Meta-Scores across all LLMs (80.99 for GPT-
4o, 87.90 for DeepSeek-chat, and 80.08 for LLaMA-3.3),
whereas competing methods exhibit substantial variability and,
in some cases, severe degradation (e.g., negative Meta-Score
for DeepSeek-chat under R&R+RCS).
The baseline comparison also provides partial ablation evi-
dence for RegulaRAG’s core components . R&R+RCS shares
the same embedding model (all-mpnet-base-v2) and the
same RecursiveCharacterSplitter chunking as RegulaRAG but
omits reference-aware BFS enrichment and smart reranking;
the large Meta-Score gap between RegulaRAG and R&R+RCS
(82.99 vs. 8.88, averaged across LLMs) therefore isolates the
joint contribution of those two components. NoRAG, which
supplies the LLM with no retrieval context at all, establishes

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15211
a lower bound on scenario-generation capability embedded
in model weights alone (average Meta-Score 57.94); com-
paring it with RegulaRAG quantifies the marginal value of
the entire retrieval pipeline. A fully factorial component-level
ablation—isolating BFS reference-closure depth, deduplica-
tion, and canonical reordering individually—remains as future
work.
A manual inspection of retrieved contexts shows that the
presence and correct ordering of the necessary table chunks
is the primary determinant of high F1. Simply retrieving
a relevant table slice is insufficient when the original PDF
table is split across pages (Fig. 5) and subsequently appears
as multiple HTML tables (Fig. 6). To address this, Regu-
laRAG uses a rule-based heuristic for table detection and
reconstruction. Specifically, when a paragraph contains cues
such as “the following table,” we treat this as an explicit
pointer to subsequent table content. In the Markdown/HTML
output produced by Mineru, tables are marked by<table>
tags. We therefore collect the consecutive<table>blocks
that immediately follow the referring paragraph and attach
them to that paragraph as referenced table metadata. This
heuristic is particularly useful for PDF tables that are sliced
across pages and converted into multiple adjacent HTML table
fragments: by grouping consecutive table tags and preserving
their original order, the system reconstructs the logical table
content before retrieval and prompt assembly.
RegulaRAG’s reference-aware chunk reordering ensures these
fragments are placed adjacently and in the correct logical
order, improving recall and reducing hallucinated numerical
substitutions.
Second, the three scenario categories differ markedly in
difficulty.CtoPremains the easiest because it is anchored
in a self-contained regulatory section (section 5.2.2. Car to
pedestrian scenario in [17] ) with a dedicated table. In contrast,
CtoStCandCtoMoCrequire integrating information dispersed
across multiple paragraphs and shared tables, making them
more sensitive to retrieval errors.
Third, the language models respond differently to retrieval
conditions. DeepSeek-chat performs competitively under Reg-
ulaRAG but becomes highly unstable under weaker retrieval
methods, as reflected by large variance and several negative
Meta-Scores for R&R+RCS and OpenAI-RAG. LLaMA-3.3
shows consistent improvement under RegulaRAG but struggles
substantially with baselines, especially in scenarios requiring
precise table interpretation and paragraph-level aggregation.
GPT-4o displays the most stable behavior under RegulaRAG,
yet—even for GPT-4o—the baseline RAG systems exhibit
large variability across tasks.
Another reason that we got better results comparing to
the baseline methods is that, thanks to our mechanism for
untangling the chain of references distributed across different
chunks, the input query is then compared against the entire
chain of referenced content along with the chunk’s original
text, thereby increasing the likelihood of selecting the most
relevant chunks. For example, when the input query pertains
to a Car-to-Pedestrian scenario in UN Regulation No. 152, a
standard chunking and retrieval method may select paragraph
5.2.2, but miss the “Maximum Impact Speed (km/h) for M1vehicles” table, which contains the required speed values. This
happens because the table shares no explicit keywords or
semantic similarity with the query. In contrast, our method
detects that paragraph 5.2.2 contains phrases like “the fol-
lowing table,” prompting a search for and attachment of that
table to the chunk. This increases the likelihood of retrieving
the chunk and provides the LLM with all necessary data to
generate complete scenario variants.
a) Qualitative example.:To illustrate the end-to-end
pipeline, consider a CtoStC scenario from UN Regulation
No. 152. Figure 1 shows how the relevant conditions are
distributed across cross-linked sections: the test speed values
reside in a table, and the approach constraint is stated in other
Section. Figure 5 shows how the speed table is split across
two PDF pages and emerges as separate HTML fragments
after conversion. RegulaRAG’s SmartChunking detects the
“the following table” cue, attaches the consecutive HTML
table fragments, and add it as metadata to the original chunk.
The Smart Retrieve and Rerank module then retrieves the
enriched chunk,and with this assembled context, the LLM
generates the scenario. Table III shows the expected output
(ground-truth entryTest_CtoStC_unladen_10from the
published dataset, Section III-A). Without reference-aware
enrichment the speed table is not retrieved, and the LLM either
omits the speed value or hallucinates one which is the primary
failure mode described in Section V-C.
TABLE III
GROUND-TRUTH SCENARIOTE S T_CT OSTC_U N L A D E N_10—EXPECTED
LLMOUTPUT FOR ACTOSTCTEST AT10KM/H(UNLADEN).
Two vehicles of Category M1 AA saloon shall be positioned.
The subject vehicle that performs the braking and the lead
vehicle.
Both vehicles should face in the same direction of travel.
The subject vehicle shall approach the lead vehicle in a
straight line for at least 2 s before the functional part of
the test commences.
The subject vehicle should travel at the speed of 10 km/h
(with a tolerance of+0/−2 km/h) when the vehicle brakes.
The functional part of the test shall start at a distance
corresponding to a Time To Collision (TTC) of at least
4 seconds from the target.
The subject vehicle is unladen.
The lead vehicle is stationary.
Post-condition requirements:
When the system is activated, the AEBS shall decrease the
speed to 0 km/h.
B. input document scalability
To assess robustness under growing evidence, we gradually
expand the retrieval corpus from a single regulation (152)
to bundles containing eight documents. The evaluation pool
contains eight UN regulations of varying length and structural
complexity:
•No. 10: Electromagnetic compatibility [34]
•No. 79: Steering equipment [35]
•No. 130: Lane Departure Warning System (LDWS) [36]
•No. 140: Electronic Stability Control (ESC) [37]
•No. 152: AEBS (primary target corpus) [38]

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15212
TABLE II
PERFORMANCE, STABILITY,ANDMETA-SCOREACROSSRAG SYSTEMS ANDMODELS
RAG System Model CtoStC (↑) CtoMoC (↑) CtoP (↑) Mean (↑) stdev (↓) Meta-Score (↑) Avg. 3 LLMs (↑)
RegulaRAGGPT-4o 76.92 96.55 100.00 91.16 10.16 80.99 82.99
DeepSeek-chat 85.71 96.55 97.44 93.23 5.33 87.90
Llama-3.3 76.92 96.55 91.89 88.45 8.37 80.08
R&R+RCSGPT-4o 8.00 88.89 91.89 62.93 38.86 24.07 8.88
DeepSeek-chat 0.00 0.00 97.44 32.48 45.93 -13.45
Llama-3.3 76.92 0.00 91.89 56.27 40.26 16.01
NoRAGGPT-4o 22.22 96.55 100.00 72.92 35.88 37.04 57.94
DeepSeek-chat 73.68 96.55 97.44 89.22 11.00 78.23
Llama-3.3 50.00 96.55 91.89 79.48 20.93 58.55
HippoRAGGPT-4o 62.86 96.55 97.44 85.62 16.10 69.52 55.75
DeepSeek-chat 90.91 80.00 97.44 89.45 7.19 82.26
Llama-3.3 73.68 0.00 91.89 55.19 39.73 15.46
OpenAI-RAGGPT-4o 62.86 94.00 99.15 85.33 16.03 69.30 43.53
DeepSeek-chat 30.30 0.00 97.44 42.58 40.72 1.86
Llama-3.3 75.79 55.37 91.89 74.35 14.94 59.41
HybridGPT-4o 54.55 96.55 97.44 82.85 20.01 62.83 56.71
DeepSeek-chat 88.37 96.55 97.44 94.12 4.08 90.04
Llama-3.3 85.71 0.00 91.89 59.20 41.94 17.26
Fig. 5.Example of split table
Fig. 6.Converted table (HTML) after PDF split
•No. 155: Cybersecurity and CSMS [39]
•No. 156: Software update and SUMS [40]
•No. 13-H: Braking (passenger cars) [41]We fixLLM∗=GPT-4o,top∗
k= 30, andchunk size∗=
3000, varying only the corpus size. This experiment isolates
the effects of semantic clutter and cross-document interactions

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15213
on retrieval stability.
Alongside REGULARAG, we evaluate HIPPORAG, a
knowledge-graph-driven system that performs LLM-based
NER, triple extraction, and graph-structured retrieval. Compar-
ing them under identical scaling conditions highlights contrast-
ing computational behaviors. We include HippoRAG because
it trails RegulaRAG across all categories in Table II.
Figures 7, 8, and 9 report F1-score, end-to-end runtime, and
OpenAI token usage as the corpus grows.
a) Runtime stability.:Across all corpus sizes, REGU-
LARAG maintains a stable runtime (47–59 s), indicating that
its enrichment and reranking pipeline scales gracefully. The
enrichment module has been optimized by removing deep
copies, avoiding storage of enriched text, compressing meta-
data into integer references, and bounding BFS expansion. As
shown in Figure 8, these optimizations enable RegulaRAG to
maintain stable runtime even as the corpus grows. In contrast,
HIPPORAG runtime grows steeply, exceeding 200 s on seven
or more regulations, driven by its LLM-heavy OpenIE and
graph-construction passes.
b) Token usage scaling.:The two models consume sub-
stantially different amounts of tokens. REGULARAG remains
compact, with token usage fluctuating slightly around 14k–25k
tokens. By contrast, HIPPORAG grows near-exponentially,
from 34k tokens on a single document to nearly 500k tokens
on the full set. This growth is driven by multiple LLM-
dependent stages—NER, triple extraction, graph expansion,
reranking, and auxiliary calls.
c) Retrieval quality.:HIPPORAG delivers consistently
strong accuracy (97–100% F1), reflecting the benefits of
explicit graph-structured reasoning. REGULARAG, optimized
for paragraph-level reference reconstruction, achieves high
accuracy on smaller inputs but declines as the corpus becomes
increasingly heterogeneous (down to 82.35%). This repre-
sents a natural trade-off: RegulaRAG prioritizes computational
efficiency and highly targeted retrieval, whereas HippoRAG
prioritizes robustness and semantic coverage, at the cost of
substantially higher token usage and runtime.
Fig. 7. F1-score scaling of REGULARAG and HIPPORAG as corpus size
increases.
Fig. 8. Runtime scaling for REGULARAG and HIPPORAG.
Fig. 9. Token usage comparison across increasing corpus sizes.
C. Failure Mode Analysis
Despite RegulaRAG’s strong overall performance, manual
inspection of generated scenarios reveals three recurring fail-
ure patterns that affect all evaluated RAG systems to varying
degrees.
(1) Incorrect numerical value assignment.LLMs tend to
reproduce the structural template of a scenario correctly but
fail to extract and apply numerical values accurately from the
regulation text. For example, a CtoStC scenario requires the
subject vehicle to travel at 20 km/h (tolerance+0/−2 km/h)
before braking, derived from the regulation statement “Tests
shall be conducted with a vehicle travelling at 20, 42, and
60 km/h.” LLMs sometimes substitute values not present in
the regulation, particularly for speeds that must be read from
the Maximum Relative Impact Speed table (Fig. 5).
(2) Loading condition confusion.LLMs frequently fail to
distinguish laden from unladen conditions, generating incor-
rect post-condition values. The correct post-condition (e.g.,
the AEBS must reduce vehicle speed to 0 km/h) depends
on loading state and must be read from the regulation table;
confusing the two conditions yields numerically incorrect
scenarios despite plausible wording.
(3) Cosine similarity inflation.Because LLMs reproduce
the scenario template structure faithfully, cosine similarity
between generated and ground-truth scenarios is often close
to 1.0 even when numerical content is wrong. This inflates

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15214
raw similarity scores and motivates the penalized metric (Al-
gorithm 2), which explicitly penalises numeric and categorical
discrepancies to surface these otherwise hidden errors.
RegulaRAG directly addresses failure modes (1) and (2) by
ensuring the relevant table chunk is present in the retrieved
context through reference-aware enrichment, thereby provid-
ing the LLM with the correct numerical grounding. Exact
numeric hallucination that occurs even when the correct table
is supplied—where the LLM misreads or ignores explicit table
values—is a well-known problem across regulated-domain
LLM applications and remains outside the scope of this
work; mitigating it through prompting strategies or constrained
decoding is a planned direction for future research.
VI. CONCLUSION ANDFUTUREWORK
In this paper, we introduced RegulaRAG, a two-stage
Retrieval-Augmented Generation (RAG) pipeline designed to
generate regulation-compliant test scenarios from complex
technical standards such as UN Regulation No. 152. By
integrating a SmartChunking strategy with reference-aware
enrichment and a semantic retrieval mechanism, RegulaRAG
offers a favorable balance between accuracy, runtime, and
token usage. Across multiple scenario types and language
models, RegulaRAG achieves the highest average Meta-Score
(82.99, 43% above the next-best baseline NoRAG at 57.94),
indicating strong and stable performance while operating at
substantially lower context cost (14k–25k tokens per query)
than graph-centric alternatives such as HippoRAG (up to 500k
tokens on an eight-document corpus).
Future work will focus on further improving RegulaRAG,
including refinements to chunk enrichment, scoring, and re-
trieval ranking. Finally, we aim to generalize the pipeline
to additional regulatory documents and domains, thereby
broadening its applicability in compliance-critical settings. We
further envision integrating the generated scenarios directly
with automotive simulation environments (e.g., CARLA, Car-
Maker) and real testbench infrastructure.
Limitations
Several limitations of the present work merit acknowl-
edgment. First, RegulaRAG was developed and evaluated
exclusively on UN Regulation No. 152. The core pipeline
— SmartChunking, BFS reference-closure enrichment, and
semantic retrieval — is regulation-agnostic and in principle
applicable to any domain whose standards share similar struc-
tural properties: hierarchical numbered paragraphs, explicit
cross-references, and tabular specifications. Domains such as
pharmaceutical (e.g., ICH guidelines), aviation (e.g., DO-
178C), or financial compliance standards share these structural
characteristics and are plausible candidates for adaptation .
Two bounded adaptation steps are required: (i) the categori-
cal regex familyR ℓ, which encodes UN-152 load-condition
keywords (laden,unladen,MassRun), must be replaced with
domain-relevant vocabulary; the cosine-similarity backbone,
numeric matching, penalty weightλ, and Meta-Score formula
are domain-agnostic and transfer without modification (see
Section III-B for a full breakdown); and (ii) the retrievalhyperparameters(top k=30, chunk size=2000)were tuned
via grid search on the CtoStC scenario family using UN-152
alone; regulations with substantially different document den-
sity, cross-reference depth, or chunk length distributions may
require re-tuning of these parameters before deployment. This
transfer risk may be partially bounded fortop kspecifically:
prior work on reader noise-sensitivity [30] suggests readers
with an improve-then-plateau retrieval-depth response — a
pattern GPT-4o follows alongside GPT-3.5-turbo — degrade
less whentop kis not re-tuned for a new corpus than noise-
sensitive readers; this bound does not extend tochunk size,
which is not examined in that work, and re-tuning both
parameters remains advisable. Second, the same subject-matter
experts contributed to both dataset construction and qualitative
evaluation, which introduces a potential consistency bias;
role separation or an independent validation cohort would
strengthen future evaluations. Third, RegulaRAG performs
reference-aware chunk enrichment and reordering but does not
construct or reason over a full knowledge graph (see Section II
for a detailed discussion): the legal language of regulatory
standards — nested clauses, conditional syntax, and domain-
specific terminology — makes reliable general-purpose entity
and relation extraction challenging, and KG-based pipelines
incur substantially higher token and runtime costs as corpus
size grows (Section V-B). Systems that build explicit entity–
relation graphs may capture deeper cross-reference semantics
at the cost of this additional complexity.
Finally, owing to resource constraints, each configuration
was evaluated in a single run without repeated-trial statistics.
While the baseline comparison provides partial ablation evi-
dence—R&R+RCS isolates the contribution of BFS reference-
aware enrichment and smart reranking, and NoRAG isolates
the entire retrieval pipeline—a component-level ablation iso-
lating BFS reference-closure depth, deduplication, and canon-
ical chunk reordering individually was not conducted; these
remain open empirical questions for future work.
APPENDIXA

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15215
USERPROMPTS FORSCENARIOGENERATION
In addition to the system prompt provided in Listing IV-D,
the language model receives a scenario-specific user prompt.
These prompts instruct the model to generate AEBS test
scenarios in a structured format that mirrors the examples
defined in UN Regulation No. 152. The three user prompts
corresponding to the Car-to-Stationary-Car (CtoStC), Car-to-
Moving-Car (CtoMoC), and Car-to-Pedestrian (CtoP) scenario
families are listed below.
User Prompt for Car-to-Stationary-Car (CtoStC)
CONTEXT:{rag_context}
Based on the context provided (from UN Regulation No.
152), generate all test scenarios related to Car to stationary
Car scenario (CtoStC) tests for M1 vehicles. The technical
service is strictly required to test any speed, therefore a
complete list of all test cases (with the speed range defined
in regulation tables) must be produced. The system shall be
active at least within the vehicle speed range specified by
Maximum Relative Impact Speed (km/h) for M1 vehicles.
Follow this exact format for every test scenario: 1-
”Write the scenario title (e.g., Test_CtoStC_unladen_42)
as the first line.” 2- ”Immediately after the title, write the
full scenario description.” 3- ”After each complete test
case, add exactly one separator line: ———”
Formatting rules: - Use the provided example structure
strictly. - Do not add explanations, notes, or extra text. -
Only output the requested test scenarios and separators. -
Do not include index numbers or any other identifiers.
Example:{example_test_case}
[Start generation now.]
User Prompt for Car-to-Moving-Car (CtoMoC)
CONTEXT:{rag_context}
Based on the context provided (from UN Regulation No.
152), generate all test scenarios related to Car to moving
Car scenario (CtoMoC) tests for M1 vehicles. The technical
service is strictly required to test any speed, therefore all
speeds defined in the regulation tables must be included.
The system shall be active at least within the vehicle speed
range specified by the Maximum Relative Impact Speed
(km/h) for M1 vehicles.
Follow this exact format for every test scenario: 1-
”Write the scenario title (e.g., Test_CtoMoC_unladen_10)
as the first line.” 2- ”Immediately after the title, write the
full scenario description.” 3- ”After each complete test
case, add exactly one separator line: ———”
Formatting rules: - Use the provided example structure
strictly. - Do not add explanations, notes, or extra text. -
Only output the requested test scenarios and separators. -
Do not include index numbers or any other identifiers.
EXAMPLE:{example_test_case}
[Start generation now.]
User Prompt for Car-to-Pedestrian (CtoP)CONTEXT:{rag_context}
Based on the context provided (from UN Regulation No.
152), generate all test scenarios related to Car to pedestrian
scenario (CtoP) tests for M1 vehicles. The technical service
is strictly required to test any speed; therefore, disregard
speeds mentioned in the context and generate the complete
set of speeds defined in the regulation tables. The system
shall be active at least within the vehicle speed range
specified by the Maximum Impact Speed (km/h) for M1
vehicles.
Follow this exact format for every test scenario: 1-
”Write the scenario title (e.g., Test_CtoP_MassRun_20)
as the first line.” 2- ”Immediately after the title, write the
full scenario description, using the exact wording of the
provided example, adapting only the relevant parameters.”
3- ”After each complete test case, add exactly one separator
line: ———”
Formatting rules: - Use the provided example structure
strictly. - Do not add explanations, notes, or extra text. -
Only output the requested test scenarios and separators. -
Do not include index numbers or any other identifiers.
Example:{example_test_case}
[Start generation now.]
ACKNOWLEDGMENT
This research was funded by the Federal Ministry of Re-
search, Technology and Space (BMFTR) as part of the CeCaS
project, FKZ: 16ME0800K.
REFERENCES
[1] N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large
Language Models Struggle to Learn Long-Tail Knowledge,” https:
//arxiv.org/abs/2211.08411, Jul. 2023, arXiv preprint arXiv:2211.08411.
[2] A. Mallen, A. Asai, V . Zhong, R. Das, D. Khashabi, and H. Hajishirzi,
“When Not to Trust Language Models: Investigating Effectiveness of
Parametric and Non-Parametric Memories,” Jul. 2023.
[3] K. Lee, M.-W. Chang, and K. Toutanova, “Latent Retrieval for Weakly
Supervised Open Domain Question Answering,” https://arxiv.org/abs/
1906.00300, Jun. 2019, arXiv preprint arXiv:1906.00300.
[4] Anthropic, “Introducing 100K context windows,” https://www.anthropic.
com/news/100k-context-windows, 2023, accessed: Sep. 28, 2025.
[5] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
and P. Liang, “Lost in the Middle: How Language Models Use Long
Contexts,” Nov. 2023.
[6] V . Zolfaghari, N. Petrovic, F. Pan, K. Lebioda, and A. Knoll, “Adopting
RAG for LLM-Aided Future Vehicle Design,” https://arxiv.org/abs/2411.
09590, Nov. 2024, arXiv preprint arXiv:2411.09590.
[7] F. Liu, Z. Kang, and X. Han, “Optimizing RAG techniques for
automotive industry PDF chatbots: A case study with locally deployed
Ollama models,” arXiv preprint arXiv:2408.05933, 2024. [Online].
Available: https://arxiv.org/abs/2408.05933
[8] F. Niu, R. Pan, L. C. Briand, H. Hu, and K. Koravadi, “TVR: Au-
tomotive System Requirement Traceability Validation and Recovery
Through Retrieval-Augmented Generation,” http://arxiv.org/abs/2504.
15427, 2025, arXiv preprint arXiv:2504.15427.
[9] J. Sun, Z. Luo, and Y . Li, “A Compliance Checking Framework
Based on Retrieval Augmented Generation,” inProceedings of the 31st
International Conference on Computational Linguistics. Abu Dhabi,
UAE: Association for Computational Linguistics, Jan. 2025, pp. 2603–
2615. [Online]. Available: https://aclanthology.org/2025.coling-main.
178/
[10] R. Bolton, M. Sheikhfathollahi, S. Parkinson, V . Vulovic, G. Bamford,
D. Basher, and H. Parkinson, “Document Retrieval Augmented Fine-
Tuning (DRAFT) for safety-critical software assessments,” http://arxiv.
org/abs/2505.01307, 2025, arXiv preprint arXiv:2505.01307.

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15216
[11] B. Kumar and D. Roussinov, “NLP-based Regulatory Compliance –
Using GPT 4.0 to Decode Regulatory Documents,” http://arxiv.org/abs/
2412.20602, 2024, arXiv preprint arXiv:2412.20602.
[12] S. Li, J. Chen, R. Yao, X. Hu, P. Zhou, W. Qiu, S. Zhang, C. Dong,
Z. Li, Q. Xie, and Z. Yuan, “Compliance-to-Code: Enhancing Financial
Compliance Checking via Code Generation,” http://arxiv.org/abs/2505.
19804, 2025, arXiv preprint arXiv:2505.19804.
[13] B. Agarwal, H. S. Jomraj, S. Kaplunov, J. Krolick, and V . Ro-
jkova, “RAGulating Compliance: A Multi-Agent Knowledge Graph for
Regulatory QA,” http://arxiv.org/abs/2508.09893, 2025, arXiv preprint
arXiv:2508.09893.
[14] L. Xu, C. Feng, K. Zhang, L. Zhengyong, W. Xu, and F. Meng,
“Equipping retrieval-augmented large language models with document
structure awareness,” inFindings of the Association for Computational
Linguistics: EMNLP 2025. Association for Computational Linguistics,
2025, pp. 24 608–24 631.
[15] X. Yu, P. Jian, and C. Chen, “TableRAG: A retrieval augmented genera-
tion framework for heterogeneous document reasoning,” inProceedings
of the 2025 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics, 2025, pp.
14 063–14 082.
[16] V . T. d. Oliveira, D. O. d. Silva, M. d. A. Souza, M. R. Lima,
S. S. T. d. Oliveira, and T. C. Rosa, “Retrieval-augmented generation
and knowledge graphs in Portuguese-language legal documents,” in
Proceedings of the 17th International Conference on Computational
Processing of Portuguese (PROPOR 2026), vol. 1, 2026, pp. 1–10.
[17] Publications Office of the European Union, “UN regulation no 152
— uniform provisions concerning the approval of motor vehicles
with regard to the advanced emergency braking system (AEBS)
for M1 and N1 vehicles,” https://op.europa.eu/en/publication-detail/
-/publication/fc2d3589-1a7c-11eb-b57e-01aa75ed71a1, Oct. 2020, ac-
cessed: Oct. 1, 2024.
[18] C. Arora, J. Grundy, L. Puli, and N. Layton, “Towards Standards-
Compliant Assistive Technology Product Specifications via LLMs,” Apr.
2024.
[19] Y . Zhao, W. Xiao, T. Mihalj, J. Hu, and A. Eichberger, “Chat2Scenario:
Scenario Extraction From Dataset Through Utilization of Large Lan-
guage Model,” in2024 IEEE Intelligent Vehicles Symposium (IV), Jun.
2024, pp. 559–566.
[20] Y . Deng, J. Yao, Z. Tu, X. Zheng, M. Zhang, and T. Zhang, “TAR-
GET: Automated Scenario Generation from Traffic Rules for Testing
Autonomous Vehicles,” Oct. 2023.
[21] S. Tang, Z. Zhang, J. Zhou, L. Lei, Y . Zhou, and Y . Xue, “LeGEND:
A Top-Down Approach to Scenario Generation of Autonomous Driving
Systems Assisted by Large Language Models,” Sep. 2024.
[22] H. Tian, X. Han, Y . Zhou, G. Wu, A. Guo, M. Cheng, S. Li, J. Wei,
and T. Zhang, “LMM-enhanced Safety-Critical Scenario Generation
for Autonomous Driving System Testing From Non-Accident Traffic
Videos,” Jan. 2025.
[23] Q. Lu, X. Wang, Y . Jiang, G. Zhao, M. Ma, and S. Feng, “Multimodal
Large Language Model Driven Scenario Testing for Autonomous Vehi-
cles,” Sep. 2024.
[24] K. Lebioda, N. Petrovic, F. Pan, V . Zolfaghari, A. Schamschurko,
and A. Knoll, “Are requirements really all you need? using LLMs to
generate configuration code: A case study in automotive simulations,”
IEEE Access, vol. 13, pp. 145 115–145 126, 2025. [Online]. Available:
https://ieeexplore.ieee.org/abstract/document/11122468
[25] B. Wang, C. Xu, X. Zhao, L. Ouyang, F. Wu, Z. Zhao, R. Xu, K. Liu,
Y . Qu, F. Shang, B. Zhang, L. Wei, Z. Sui, W. Li, B. Shi, Y . Qiao,
D. Lin, and C. He, “MinerU: An Open-Source Solution for Precise
Document Content Extraction,” http://arxiv.org/abs/2409.18839, 2024,
arXiv preprint arXiv:2409.18839.
[26] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using Siamese BERT-networks,” inProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing (EMNLP).
Association for Computational Linguistics, 2019, pp. 3982–3992.
[Online]. Available: https://arxiv.org/abs/1908.10084
[27] N. Alturayeif, I. Ahmad, and J. Hassine, “TraceLLM: Leveraging large
language models with prompt engineering for enhanced requirements
traceability,”Requirements Engineering, 2026, arXiv:2602.01253.
[28] R. Etezadi, S. Abualhaija, C. Arora, and L. Briand, “Classifier or prompt:
A case study on legal requirements traceability,”Empirical Software
Engineering, 2026, arXiv:2502.04916.
[29] O. Folorunsho and H. Reza, “AI-driven test case generation from natural
language requirements: A survey of techniques and research gaps,” https:
//arxiv.org/abs/2606.06563, 2026.[30] J. Hsia, A. Shaikh, Z. Wang, and G. Neubig, “RAGGED: To-
wards Informed Design of Scalable and Stable RAG Systems,”
https://arxiv.org/abs/2403.09040, 2025, proceedings of the 42nd Inter-
national Conference on Machine Learning (ICML), PMLR 267.
[31] B. J. Guti ´errez, Y . Shu, W. Qi, S. Zhou, and Y . Su, “From RAG to Mem-
ory: Non-Parametric Continual Learning for Large Language Models,”
http://arxiv.org/abs/2502.14802, 2025, arXiv preprint arXiv:2502.14802.
[32] H. Markowitz, “Portfolio selection,”The Journal of Finance, vol. 7,
no. 1, pp. 77–91, 1952.
[33] T. Hastie, R. Tibshirani, and J. Friedman,The Elements of Statistical
Learning: Data Mining, Inference, and Prediction, 2nd ed. Springer,
2009.
[34] “Regulation no 10 of the economic commission for europe of the united
nations (UN/ECE) — uniform provisions concerning the approval of
vehicles with regard to electromagnetic compatibility,” https://eur-lex.
europa.eu/eli/reg/2012/10/oj/eng, United Nations Economic Commission
for Europe, 2012, accessed: Sep. 29, 2025.
[35] “Regulation no 79 of the economic commission for europe of the united
nations (UN/ECE) — uniform provisions concerning the approval of
vehicles with regard to steering equipment,” https://eur-lex.europa.eu/
eli/reg/2008/79(2)/oj/eng, United Nations Economic Commission for
Europe, 2008, accessed: Sep. 29, 2025.
[36] “Regulation no 130 of the economic commission for europe of the
united nations (UN/ECE) — uniform provisions concerning the approval
of motor vehicles with regard to the lane departure warning system
(LDWS),” https://eur-lex.europa.eu/eli/reg/2014/130/oj/eng, United Na-
tions Economic Commission for Europe, 2014, accessed: Sep. 29, 2025.
[37] “Regulation no 140 of the economic commission for europe of the united
nations (UN/ECE) — uniform provisions concerning the approval of
passenger cars with regard to electronic stability control (ESC) systems,”
https://eur-lex.europa.eu/eli/reg/2018/1592/oj/eng, United Nations Eco-
nomic Commission for Europe, 2018, accessed: Sep. 29, 2025.
[38] “UN regulation no 152 — uniform provisions concerning the approval
of motor vehicles with regard to the advanced emergency braking system
(AEBS) for M1 and N1 vehicles,” https://eur-lex.europa.eu/eli/reg/2020/
1597/oj/eng, United Nations Economic Commission for Europe, 2020,
accessed: Sep. 29, 2025.
[39] “UN regulation no 155 — uniform provisions concerning the approval
of vehicles with regards to cybersecurity and cybersecurity manage-
ment system,” https://eur-lex.europa.eu/eli/reg/2021/387/oj/eng, United
Nations Economic Commission for Europe, 2021, accessed: Sep. 29,
2025.
[40] “UN regulation no 156 — uniform provisions concerning the approval
of vehicles with regards to software update and software updates
management system,” https://eur-lex.europa.eu/eli/reg/2021/388/oj/eng,
United Nations Economic Commission for Europe, 2021, accessed:
Sep. 29, 2025.
[41] “UN regulation no 13-h — uniform provisions concerning the approval
of passenger cars with regard to braking,” https://eur-lex.europa.eu/eli/
reg/2023/401/oj/eng, United Nations Economic Commission for Europe,
2023, accessed: Sep. 29, 2025.
V AHID ZOLFAGHARIreceived the M.Sc. degree
in computer science from the Amirkabir University
of Technology. He is currently pursuing the Ph.D.
degree with the Technical University of Munich. He
is part of the Central Car Server (CeCaS) Project.
Previously, he was a Senior Test Engineer in network
security and later a Research Assistant in Italy,
contributing to projects in IoT security, robotic sys-
tems, and neurorobotics. He has co-authored several
publications on generative AI and automotive system
design. His research focuses on the application of
large language models and retrieval-augmented generation systems for auto-
motive software engineering.

ZOLFAGHARIet al.: THINK INSIDE THE CHUNK: REGULARAG FOR REGULATION-COMPLIANT SCENARIO GENERATION USING LLMS: A CASE STUDY OF UN REGULATION NO. 15217
Nenad Petrovicwas born in Pirot, Serbia, in 1992.
He received the bachelor’ degree in computer sci-
ence and informatics from the Facultyof Electronic
Engineering, University of Nis, Nis, Serbia, in 2015,
the Laurea Magistrale degree in computer engineer-
ing from the Politecnico di Milano, Milan, Italy,
and the Ph.D. degree in computing and electrical
engineering from the Faculty of Electronic Engineer-
ing. He was a Teaching Assistant. He is currently a
Postdoctoral Researcher and a Scientific Coordina-
tor/Supervisor with the Chair of Robotics, Artificial
Intelligence and Real-Time Systems, Technical University of Munich (TUM),
focused on automotive-related projects in area of GenAI-driven software
development. He is the author or co-author of more than 200 scientific
publications. His main areas of interests include model-driven software
engineering, semantic technology, the Internet of Things (IoT), and generative
artificial intelligence (GenAI).
ANDR ´E SCHAMSCHURKOreceived the
Dipl.Inf. degree in computer science from TU
Dresden, in 2023. He is currently pursuing the
Ph.D. degree with the Chair of Robotics, Artificial
Intelligence and Real-Time Systems, Technical
University of Munich. The primary focus of his
research is on natural language processing and
generative AI models, with a particular interests in
reducing their hallucinations.
ALOIS KNOLL(Fellow, IEEE) received the
M.Sc. degree in electrical/communications engineer-
ing from the University of Stuttgart, Stuttgart, Ger-
many, in 1985, and the Ph.D. degree (summa cum
laude) in computer science from the Technical Uni-
versity of Berlin (TU Berlin), Berlin, Germany, in
1988. He was on the Faculty of the Computer Sci-
ence Department, TU Berlin, until 1993. He joined
the University of Bielefeld, Bielefeld, Germany, as
a Full Professor, where he was the Director of the
Technical Informatics Research Group, until 2001.
Since 2001, he has been a Professor with the Department of Informatics,
Technical University of Munich (TUM), Munich, Germany. He was on the
board of directors of the Central Institute of Medical Technology at TUM
(IMETUM). From 2004 to 2006, he was the Executive Director of the
Institute of Computer Science, TUM. From 2007 to 2009, he was a member
of the EU’s highest advisory board on information technology, ISTAG, the
Information Society Technology Advisory Group, and its subgroup on future
and emerging technologies (FET). In this capacity, he was actively involved in
developing the concept of EU’s FET flagship projects. His research interests
include cognitive, medical, and sensor-based robotics; multi-agent systems;
data fusion; adaptive systems; multimedia information retrieval; modeldriven
development of embedded systems, with applications to automotive software
and electric transportation; and simulation systems for robotics and traffic.
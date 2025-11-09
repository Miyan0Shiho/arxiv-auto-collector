# DeepSpecs: Expert-Level Questions Answering in 5G

**Authors**: Aman Ganapathy Manvattira, Yifei Xu, Ziyue Dang, Songwu Lu

**Published**: 2025-11-03 07:39:22

**PDF URL**: [http://arxiv.org/pdf/2511.01305v1](http://arxiv.org/pdf/2511.01305v1)

## Abstract
5G technology enables mobile Internet access for billions of users. Answering
expert-level questions about 5G specifications requires navigating thousands of
pages of cross-referenced standards that evolve across releases. Existing
retrieval-augmented generation (RAG) frameworks, including telecom-specific
approaches, rely on semantic similarity and cannot reliably resolve
cross-references or reason about specification evolution. We present DeepSpecs,
a RAG system enhanced by structural and temporal reasoning via three
metadata-rich databases: SpecDB (clause-aligned specification text), ChangeDB
(line-level version diffs), and TDocDB (standardization meeting documents).
DeepSpecs explicitly resolves cross-references by recursively retrieving
referenced clauses through metadata lookup, and traces specification evolution
by mining changes and linking them to Change Requests that document design
rationale. We curate two 5G QA datasets: 573 expert-annotated real-world
questions from practitioner forums and educational resources, and 350
evolution-focused questions derived from approved Change Requests. Across
multiple LLM backends, DeepSpecs outperforms base models and state-of-the-art
telecom RAG systems; ablations confirm that explicit cross-reference resolution
and evolution-aware retrieval substantially improve answer quality,
underscoring the value of modeling the structural and temporal properties of 5G
standards.

## Full Text


<!-- PDF content starts -->

DeepSpecs: Expert-Level Questions Answering in 5G
Aman Ganapathy Manvattira*, Yifei Xu*, Ziyue Dang, Songwu Lu
University of California, Los Angeles
amanvatt02@g.ucla.edu, {yxu, ziyue.dang, slu}@cs.ucla.edu
Abstract
5G technology enables mobile Internet access
for billions of users. Answering expert-level
questions about 5G specifications requires nav-
igating thousands of pages of cross-referenced
standards that evolve across releases. Existing
retrieval-augmented generation (RAG) frame-
works, including telecom-specific approaches,
rely on semantic similarity and cannot reli-
ably resolve cross-references or reason about
specification evolution. We present DEEP-
SPECS, a RAG system enhanced by structural
and temporal reasoning via three metadata-rich
databases: SpecDB (clause-aligned specifica-
tion text), ChangeDB (line-level version diffs),
and TDocDB (standardization meeting docu-
ments). DEEPSPECSexplicitly resolves cross-
references by recursively retrieving referenced
clauses through metadata lookup, and traces
specification evolution by mining changes and
linking them to Change Requests that doc-
ument design rationale. We curate two 5G
QA datasets: 573 expert-annotated real-world
questions from practitioner forums and edu-
cational resources, and 350 evolution-focused
questions derived from approved Change Re-
quests. Across multiple LLM backends, DEEP-
SPECSoutperforms base models and state-of-
the-art telecom RAG systems; ablations con-
firm that explicit cross-reference resolution and
evolution-aware retrieval substantially improve
answer quality, underscoring the value of mod-
eling the structural and temporal properties of
5G standards.
1 Introduction
5G technology enables mobile and wireless Inter-
net access for billions of users globally. At the foun-
dation of this ecosystem lies the 3GPP standard,
a continuously evolving set of specifications that
define every technical detail of 5G systems. These
documents span thousands of pages per release and
*Equal contribution.cover diverse aspects such as architecture, proto-
cols, and procedures. They are vital to the entire
telecom workflow, including design, implementa-
tion, testing, and operations (3rd Generation Part-
nership Project (3GPP), 2024a). Practitioners such
as system architects, field engineers, and vendor de-
velopers frequently need to answer complex techni-
cal questions grounded in these specifications. For
example, they may ask “What are the conditions
required to support Feature X?” or “When and why
was Feature Y introduced?” Answering such ques-
tions typically involves navigating multiple cross-
referred documents, tracing changes across differ-
ent releases, and understanding the rationale behind
design decisions. This process is time-consuming
and requires substantial domain expertise (Chen
et al., 2022; Rodriguez Y, 2025).
Recent advances in large language models
(LLMs) and retrieval-augmented generation (RAG)
have emerged as promising tools for automating
technical question answering (Lewis et al., 2020;
Guu et al., 2020; Izacard et al., 2023). Recent ef-
forts have adapted these approaches to the telecom
domain. Systems such as Tele-LLMs (Maatouk
et al., 2024), Telco-RAG (Bornea et al., 2024),
and Chat3GPP (Huang et al., 2025) incorporate
domain-specific retrieval or fine-tuning on 3GPP
specifications. However, these systems largely fol-
low vanilla RAG pipelines without considering
the unique inter-document relationships that span
across the 3GPP corpus. Two core challenges re-
main when applying these methods to 5G speci-
fications.(1) Dense cross-references:5G specifi-
cations are designed for modularity and precision.
Instead of repeating content, for readability they
refer extensively to other clauses or documents
via specification and clause IDs (Qualcomm, 2017;
3rd Generation Partnership Project (3GPP), 2024b).
As a result, critical definitions or conditions are of-
ten located outside of semantically similar text, re-
quiring accurate reference resolution beyond what
1arXiv:2511.01305v1  [cs.CL]  3 Nov 2025

standard semantic retrieval can provide.(2) Im-
plicit reasoning behind the evolution of specifica-
tions:As 5G specifications evolve incrementally
over many releases, each version defines only what
the intended behavior is at that point in time. De-
sign rationale and historical context are excluded
by intent, as specifications aim to specify "what
is" rather than "why it changed." This makes it
challenging to trace when and how a feature was
introduced, modified, or deprecated without con-
sulting external sources such as version changes
and meeting discussions (Baron and Gupta, 2018;
Chen et al., 2022).
To address these gaps, we presentDEEPSPECS,
a QA system tailored to the 5G domain. DEEP-
SPECSextends the RAG paradigm with structural
and temporal understanding of 3GPP specifications.
At its core, DEEPSPECSintegrates three domain-
specific, metadata-rich databases:SpecDB, which
stores aligned and structured specification contents;
ChangeDB, which tracks line-level changes and
associated metadata across releases; andTDocDB,
which organizes 3GPP meeting documents (TDocs)
to support reasoning about design-phase intents.
Building on this foundation, DEEPSPECSprovides
two clause-level capabilities that mimic how a
human expert navigates the standards: (i)cross-
reference resolutionvia rule-based reference ex-
traction and hybrid retrieval, enabling fine-grained
navigation within and across documents; and (ii)
specification-evolution reasoningby mining clause
changes and linking them to corresponding discus-
sions through 3GPP-specific metadata, allowing
the system to recall when a feature was introduced,
how it evolved, and why.
To evaluate the effectiveness of our method, we
curate a real-world dataset of 573 question–answer
pairs spanning over 3 years from public telecom
forums and blog posts, annotated by three tele-
com experts. The dataset covers practitioner-facing
questions across RAN, core, protocol features,
and evolution-related queries, providing a realis-
tic testbed for telecom QA. DEEPSPECSdemon-
strates consistent gains over strong LLM baselines
and a state-of-the-art telecom-specific RAG sys-
tem. We further design targeted microbenchmarks
for cross-reference resolution and specification-
evolution reasoning, and validate the effectiveness
of DEEPSPECSon both.
The source code for DEEPSPECSis currently
available upon email request.Contributions.
•We introduce DEEPSPECS, a novel frame-
work for 5G QA that enables clause-level
cross-reference resolution and specification-
evolution reasoning, supporting expert-level
question answering over technical standards.
•We present, to our knowledge, the first expert-
annotated 5G QA dataset, comprising 573
question–answer pairs collected from real-
world practitioner forums and blog posts.
•Through automated and human evaluation, we
show that DEEPSPECSoutperforms strong
LLM baselines combined with state-of-the-
art retrieval, and demonstrate the benefit of
cross-reference resolution and specification-
evolution reasoning.
2 Related Work
RAG Approach for Advanced QA: Standard
RAG systems like Atlas (Izacard et al., 2023) and
IRCoT (Trivedi et al., 2023) improve factual con-
sistency through retrieval. However, they struggle
with technical specifications like 5G, which require
precise resolution of intra- and inter-document
references. Standard RAG retrieves isolated text
chunks without resolving cross-references or track-
ing specification evolution across versions. They
also lack support for surfacing the rationale behind
changes, limiting their utility in domains where
accurate interpretation of evolving, interconnected
documents is critical. Recent RAG research ad-
dresses some gaps: GraphRAG (Edge et al., 2024)
models text as knowledge graphs; agentic frame-
works like Legal Document RAG (Zhang et al.,
2025a) and FinSage (Wang et al., 2025) employ
multi-agent traversal; Temporal RAG systems such
asE2RAG (Zhang et al., 2025b) and ConQRet
(Dhole et al., 2025) support fine-grained reason-
ing over evolving corpora. However, these have
yet to address the technical and versioned com-
plexity of telecom standards. Our system extends
this frontier by integrating structural, temporal, and
argumentative retrieval to enable evolution-aware,
cross-referenced question answering for 5G speci-
fications.
Domain-Specific Telecom QA Systems: Recent
advances in LLMs have enabled their use in
telecommunications to improve access to the com-
plex LTE/5G standard specifications. Tele-LLMs
2

(Maatouk et al., 2024) are domain-specialized mod-
els trained on 3GPP specifications, achieving im-
provements over general LLMs. RAG-based sys-
tems such as Chat3GPP (Huang et al., 2025),
Telco-RAG (Bornea et al., 2024), and Telco-oRAG
(Bornea et al., 2025) employ hybrid dense-sparse
retrieval, neural routing and query reformulation
for 3GPP document collections. While these sys-
tems improve upon basic semantic search, they re-
main fundamentally vanilla RAG approaches that
retrieve text chunks based on semantic similarity
without explicit support for cross-reference resolu-
tion or specification evolution tracking. Comple-
mentary to these systems, TSpec-LLM (Nikbakht
et al., 2024) provides a comprehensive 3GPP
dataset and shows improved QA performance using
naive RAG, but its evaluation is limited to multiple-
choice questions and lacks support for reference
resolution or specification evolution. In contrast,
our system extends this line of work by incorpo-
rating structural, temporal, and argumentative re-
trieval tailored to 3GPP specifications. It explicitly
resolves inter-document references and tracks spec-
ification evolution over time, which are capabili-
ties essential for answering expert-level telecom
questions. We also develop new datasets targeting
reference resolution and specification evolution, en-
abling evaluation beyond multiple-choice formats.
3 Method
3.1 Goal
Our goal is to build a system that answers expert-
level questions over 5G specifications by producing
accurate and helpful responses. This requires struc-
tural and temporal understanding beyond surface
semantics, in particular: (i)cross-reference resolu-
tion, which follows spec IDs and clause numbers
to integrate information scattered across modular,
non-redundant documents; and (ii)specification-
evolution reasoning, which traces when a feature
was introduced, how it changed across releases,
and why. In line with practitioner practice, our
goal emphasizes not only factual accuracy but also
explanation helpfulness, reflecting how experts con-
textualize and justify their answers.
3.2 Overview
DEEPSPECSanswers expert-level questions over
5G specifications by extending a RAG frame-
work with structural and temporal reasoning. Fol-
lowing human expert practice, it supports clause-level cross-reference resolution and specification-
evolution reasoning.
DEEPSPECSoperates in four stages:(1)
Database construction, where specification con-
tent and TDocs are indexed into three databases:
SpecDB, serving as the primary context provider
enriched with metadata for accurate reference re-
trieval;ChangeDBandTDocDB, which capture the
temporal evolution of specifications and support
reasoning over changes and design decisions;(2)
Cross-reference resolution, which extracts specifi-
cation and clause IDs using rule-based patterns and
retrieves the corresponding clauses from filtered
chunks with metadata;(3) Specification-evolution
reasoning, which maps the queried feature to rel-
evant entries in ChangeDB and uses the associ-
ated metadata to retrieve design-phase reasoning
from TDocDB; and(4) Answer generation, which
produces the final response. Each component is
described in detail below.
3.3 Database Construction
As illustrated in Figure 1, we construct three struc-
tured databases that serve as the core retrieval
sources for DEEPSPECS.
SpecDB:This database stores the core content
of 3GPP specification documents. We download
official DOCX-formatted specifications from the
3GPP official archive1. We chunk documents by
atomic clauses (e.g., “7.4.1.1.2 Mapping to phys-
ical resource”) and annotated with metadata in-
cluding specification ID, clause ID, version num-
ber, and timestamp, which are extracted based
on structural patterns followed consistently across
3GPP specs (OpenAirInterface, 2022). This clause-
aligned indexing enables fine-grained retrieval and
also serves as the basis for constructing ChangeDB.
We employ an LLM for metadata extraction.
ChangeDB:This database records line-level dif-
ferences between adjacent versions of each specifi-
cation clause. Using the clause-aligned chunks de-
rived from SpecDB, we sort all versions of a given
clause by timestamp and skip missing versions for
robustness consideration. We then compute diffs
between each pair of adjacent versions and extract
added, removed, or modified lines. Each change en-
try is annotated with metadata from both versions,
including timestamps, and is indexed for temporal
reasoning. We also treat the first observed version
of a clause as its initial addition, allowing us to
1https://www.3gpp.org/ftp/
3

Specs
7.4.1.1.2 Mapping to
physical resources
The UE shall assume the
PDSCH DM-RS being
mapped to ...spec_id: 38.211
version: 18.7.0
date: 2025-06
cls_id: 7.4.1.1.2Chunk k
(Spec I, Clause j)V0 V1 Vlatest ...
+ For PDSCH mapping type A - the
case dmrs-AdditionalPosition  equals
to 'pos3' is only supported when ...
- No Deletionspec_id: 38.211 cls_id: 7.4.1.1.2
version_a: 15.8.0 date_a: 2020-01
version_b: 16.2.0 date_b: 2020-07TDocs
In the NR, the use of
baseline front-loaded RSs
is supported due to its
ability for achieving low
latency ... spec_id: 38.211
date: 2017-02Sort by
VersionLine-Level
ChangesClause-Aligned
Chunking &
Metadata ExtractionTDoc-Specific Chunking &
Metadata Extraction
SpecDB ChangeDB TDocDBFigure 1: Database construction of DEEPSPECS. The system builds three vector databases from 5G specs and
TDocs, each embedded with rich metadata: SpecDB provides the backbone context; ChangeDB tracks the temporal
evolution of specs; TDocDB stores the reasoning behind each change.
track the introduction of new features.
TDocDB:This database stores 3GPP meeting docu-
ments (TDocs), which capture design-phase discus-
sions explaining feature additions, protocol trade-
offs, and specification ambiguities. Although such
content is essential for understanding the rationale
behind changes, it is omitted from the formal speci-
fications for reasons of brevity and standardization
(ShareTechnote; Baron and Gupta, 2018). In this
work, we focus onChange Requests(CRs), which
document the reasoning behind each proposed mod-
ification. We collect DOCX-formatted CRs from
the official archive. These CRs follow a consistent
structure. To process them, we employ an LLM
extractor for CR-specific chunking and metadata
extraction. In our parsing strategy for change re-
quests (CRs), we extract content from three key
sections: the summary of changes, the reasons for
the change, and the consequences of non-approval.
Each individual change listed in the summary is
segmented into a separate chunk, within which it is
then paired with its corresponding rationale and the
potential consequences if the change is not adopted.
All three databases are indexed for both exact
metadata filtering and dense semantic retrieval us-
ing OpenAI’s text-embedding-3-large embed-
dings (OpenAI, 2024b). We use these indices for
hybrid retrieval during following reference resolu-
tion and evolution reasoning stages.
3.4 Cross-Reference Resolution
5G specifications are intentionally modular. To re-
duce redundancy and maintain formality, clauses
frequently cite other clauses within the same docu-
ment or across different documents. For instance,
a clause in TS 38.211 may define a feature but de-fer key operational conditions to TS 38.214. As
a result, answering a single technical query often
requires chaining together multiple cross-referred
clauses (OpenAirInterface, 2022).
To support this, DEEPSPECSimplements clause-
level cross-reference resolution, as shown in Fig-
ure 2. The resolution process consists of the fol-
lowing steps:
(1) Initial Retrieval:Given a user query, we first
retrieve top- k1ranked passages from SpecDB ac-
cording to semantic similarity. Specifically, we
employ HyDE (Gao et al., 2023) in this phase since
practical user queries in the 5G domain often lack
sufficient context for effective zero-shot retrieval.
(2) Reference Extraction:From the initially re-
trieved passages, we extract all cited specification
and clause ID pairs <spec_id, cls_id> . This
is done using regular expressions based on cita-
tion patterns commonly found in 3GPP documents,
such as “See clause 5.1.6.4 of TS 38.214.”
(3) Reference Retrieval:Each pair of extracted
<spec_id, cls_id> is used to locate the corre-
sponding clause in SpecDB by matching its spec
ID and clause ID through metadata lookup. If the
retrieved chunk contains additional references, the
system recursively resolves them to build a deeper
context chain. To avoid over-expansion, we use
a configurable maximum recursion depth. This
process enables the system to construct a retrieval
trace that reflects the true dependency structure of
the initially retrieved passage.
(4) Reference Ranking:All referenced chunks,
including those obtained recursively, are re-ranked
according to their semantic relevance to the original
question. The top- k2chunks are returned for final
answer generation.
4

Metadata Filter:
- spec_id match
- [optional filter rules]QueryExtract
RefsInitial
RetrievalRetrieve
Refs
Extract Refs (Recursively)
AnswerLLM
GenerateCross-Reference Resolution
Change:
++++++++
- - - - - - - -Retrieve
ChangesWe consider adding the
support of ... due to ...Retrieve
TDocs
Specification-Evolution Reasoning... as described in clause
5.1.6.4 of [6, TS 38.214] .spec_id: 38.214
cls_id:  5.1.6.45.1.6.4 SRS reception
procedure for CLI ...
Extract Keywords
metadataSpecDB
ChangeDB TDocDBFigure 2: The retrieval process of DEEPSPECS. The system leverages the metadata associated with the chunks in
each DB to resolve cross-references and trace changes of specs. This allows DEEPSPECSto provide informative
context with structural and temporal understanding of 5G specs for question answering.
3.5 Specification-Evolution Reasoning
Many expert-level queries require understanding
how a feature has changed across specification ver-
sions and why certain changes were introduced.
While 3GPP specifications describe the current in-
tended behavior with precision, they omit the ra-
tionale behind changes for conciseness and neu-
trality (Baron and Gupta, 2018). This rationale
is often recorded instead in 3GPP meeting docu-
ments (TDocs), which capture discussions, debates,
and design motivations during the standardization
process.
As illustrated in Figure 2, DEEPSPECSleverages
the TDocs and traces the evolution of a feature with
the following steps:
(1) Direct Extraction:DEEPSPECSfirst attempts
to extract explicit mentions of specification IDs
from the user query (practically, users often name
a spec directly). The extracted spec IDs feed a
later metadata-based filter. We employ an LLM
extractor for this step.
(2) Change Retrieval:If no specification
is mentioned explicitly, DEEPSPECSqueries
CHANGEDB using dense semantic retrieval (along
with HyDE) to locate the most relevant change
entry. This step discovers additions, removals, or
modifications related to the queried feature.
(3) Metadata-Based Filtering:Using the meta-
data (e.g., date, spec ID) of the identified change,
DEEPSPECSmaps the query or retrieved change (if
any) to a narrowed set of candidate TDocs. While
our current implementation matches by spec ID
only, which is sufficiently accurate for CRs, the
DEEPSPECSframework also supports filtering withadditional metadata (e.g., working group, feature
tags) and rules when other TDoc types are included.
(4) TDoc Ranking:All filtered TDocs from
TDOCDB are re-ranked according to their semantic
relevance to the HyDE aligned hypothetical docu-
ment generated from the original query. The top- k3
chunks are returned for final answer generation.
This hybrid strategy, supported by ChangeDB,
not only grounds the query-to-TDoc mapping in
explicit specification metadata but also enhances
the interpretability and transparency of retrieval
compared to purely semantic methods, which is
especially valuable for practitioners.
3.6 Answer Generation
After retrieving relevant specification clauses,
resolving cross-references, and identifying sup-
porting TDocs, DEEPSPECSassembles a gen-
eration prompt for a general-purpose language
model to synthesize the final answer. From the
stage of initial retrieval, reference resolution, and
specification-evolution reasoning, it selects the top-
k1,k2, andk3chunks, respectively, ranked by se-
mantic similarity.
4 Datasets
There are very few established benchmarks on tele-
com QA and most emphasize surface-level fac-
tual queries, rely on synthetic QA collections, and
often restrict evaluation to multiple-choice prob-
lems (Maatouk et al., 2024; Huang et al., 2025;
Bornea et al., 2024, 2025; Nikbakht et al., 2024).
We address these gaps by constructing two new
datasets that capture (i) real-world practitioner in-
5

formation needs and (ii) reasoning behind the evo-
lutions of specifications.
4.1 Real-World QA Dataset
To evaluate system performance in realistic pro-
fessional settings, we curate a dataset of ques-
tion–answer pairs that reflect genuine practitioner
information needs. The data are drawn from pub-
licly available forums and educational resources
where telecommunications professionals actively
exchange knowledge on 5G, ensuring alignment
with engineering and instructional use cases.
Data Collection.We collect QA data from two
widely used sources spanning a three-year pe-
riod from August 2022 to August 2025.tele-
comHall(telecomHall) is a long-standing global
telecommunications community (established in
1999) where practitioners discuss technical chal-
lenges. We retain only explicit question–answer
exchanges that are technically accurate, complete,
and well-formed.ShareTechnote(Ryu) is an educa-
tional resource with extensive technical documen-
tation on cellular communications, with a strong
emphasis on 5G. From 236 webpages, we formu-
late precise questions and extract answers directly
from the material. The questions are categorized
into five technical domains (detailed taxonomy in
Appendix B).
Human Annotation.Three volunteers with ad-
vanced backgrounds in mobile networks and cellu-
lar systems review each QA pair to validate techni-
cal correctness, clarity, and relevance. The curation,
data collection, and verification process require ap-
proximately 120 hours of annotation effort across
a 3-month collection period.
4.2 CR-Focused QA Dataset
To support targeted evaluation of DEEPSPECS
specification-evolution reasoning, we construct an-
other QA dataset of derived from real 3GPP speci-
fication changes with rich supervision signals.
Data Collection.We collect approved Change
Requests from 3GPP specifications spanning Re-
leases 17 and 18. The collection yield 997 CR
documents, with the majority addressing physi-
cal layer procedures, channels, and multiplexing,
along with dual connectivity enhancements. Each
approved CR is retrieved with its associated TSG
documentation package containing the completeSource # Avg. Q Len Avg. A Len
Real-world QA Dataset
telecomHall 102 46.9 tokens 126.9 tokens
ShareTechnote 471 30.3 tokens 102.7 tokens
Total573 33.2 tokens 107.0 tokens
CR-focused QA Dataset
3GPP Change Requests 350 45.4 tokens 287.7 tokens
Table 1: Statistics of the QA datasets, including real-
world sources and CR-based pairs.
change proposal, rationale, technical details, and
impact analysis.
QA Generation.We use a combination of LLM
and human verification to generate the QA pairs.
We first extract three key fields from each CR:
summary of change, reason for change, and conse-
quences if not approved. To ensure quality, we filter
out trivial CRs where all three fields contain only
brief descriptions (fewer than 200 words each), typ-
ically representing minor editorial corrections lack-
ing sufficient technical depth. For substantive CRs,
we employ an LLM to generate question-answer
pairs following a structured prompt (detailed in
Appendix A.3). All generated QA pairs are manu-
ally reviewed to verify technical accuracy, question
practicality, and answer quality.
4.3 Dataset Characteristics.
Table 1 summarizes statistics of the final collected
real-world and CR-focused QA datasets.
5 Experiments
We evaluate DEEPSPECSon its ability to an-
swer expert-level questions about 5G specifications.
Specifically, we ask:
1.How well does DEEPSPECSanswer practical
questions encountered by practitioners, com-
pared to the state-of-the-art baselines?
2.How does semantic retrieval differ from cross-
reference resolution performed by experts?
3.How well does DEEPSPECSperform
specification-evolution reasoning on tasks
targeting evolution-related reasoning?
5.1 Experimental Setup
5.1.1 Retrieval Databases
For SPECDB, we collect all Release 17 and 18
3GPP technical specifications, following prior
6

work (Huang et al., 2025). In total, we obtain
2137 specifications as input for database construc-
tion. We download Change Requests from 3GPP
Releases 17 and 18 that satisfy "approved" status
requirements, yielding 997 TDocs in total.
5.1.2 Baselines
We compare DEEPSPECSagainst two categories of
baselines:(1) Base Model:Direct prompting of
the base LLMs without any retrieval context. For
consistency, we use the same answer-generation
prompt, but replace the context section with “ NO
CONTEXT .”(2) Chat3GPP:A state-of-the-art RAG
system tailored for telecom-domain question an-
swering (Huang et al., 2025) employs hybrid re-
trieval, specialized chunking, and efficient index-
ing, and demonstrates superior performance over
existing methods. Nevertheless, Chat3GPP still
largely follows the vanilla RAG paradigm. We use
it as our primary baseline to demonstrate the limita-
tions of standard RAG approaches on expert-level
5G questions.
For both baselines, we test a collectin
of generation backends, including GPT-
4o/4.1/4.1 mini (OpenAI, 2024a, 2025),
Qwen3-4/8/14/32B (Yang et al., 2025), and
Claude 3.5 Haiku (Anthropic, 2024), all using their
default or recommended hyperparameter settings.
5.1.3 Evaluation Metrics
Pairwise Win Rate.We employ an LLM evalu-
ator validated with human evaluation to compute
head-to-head win rates between system outputs.
The evaluator is given the question, a gold answer,
and two candidate answers, and is instructed to
select the preferred one based on accuracy and
helpfulness. The evaluation prompt is provided
in Appendix A.1. To mitigate positional bias, eval-
uation is repeated with reversed answer order. We
validate the alignment with human evaluators by
asking domain experts to re-judge a subset of 144
pairs. The LLM-based judgments are aligned with
human assessments in 92.4% of cases, with most
disagreements occurring in close comparisons.
Rubric-Based Scoring.To complement pairwise
win rates with absolute quality scores, we design
question-specific rubrics. For each QA pair, rubrics
are first generated with an LLM, then curated and
validated by domain experts. During evaluation,
an LLM applies these rubrics to score each candi-
date answer along multiple dimensions. The rubricdesign and evaluation prompts are provided in Ap-
pendix A.2.
We provide more implementation details for our
experiments in Appendix D.
5.2 Results on Real-World QA
Table 2 summarizes results across multiple model
families. For DEEPSPECS, we fix retrieval param-
eters ( k1=4,k2=3,k3=3), and have Chat3GPP
return the top-10 chunks to ensure that both DEEP-
SPECSand Chat3GPP return the same amount
of context. We report both win rates and output
lengths to ensure that observed improvements are
not confounded by verbosity bias.
Retrieval consistently helps, with DEEPSPECS
providing the strongest gains.Both Chat3GPP
and DEEPSPECSimprove over base models, un-
derscoring the importance of retrieval augmenta-
tion for navigating 3GPP specifications. While
Chat3GPP offers decent improvements, it remains
limited by its inability to resolve cross-references
or track evolutions of specifications. In contrast,
DEEPSPECSleverages structural- and evolution-
aware retrieval to deliver consistent gains across
generator models and question categories, show-
ing robustness regardless of model size or out-
put length. Rubric-based evaluation provides fur-
ther confirmation of these findings. With GPT-
4.1, DEEPSPECSachieves a mean score improve-
ment of 0.62 points on a 5-point scale (p<0.0001,
Wilcoxon signed-rank test, dz=0.41), while against
GPT-4.1-mini, the improvement is 0.60 points
(p<0.0001, dz=0.40). Detailed results on per-
category win rates and rubric-based scoring are
provided in Appendix B.
Gains are larger on certain weaker models, but
remain meaningful for stronger ones.The re-
sults indicate that even high-capacity models ben-
efit from explicit grounding in structural and tem-
poral aspects of the specifications. Given the rapid
evolution of 3GPP standards, such grounding be-
comes especially valuable for maintaining accurate
and helpful QA along the time.
5.3 Evaluating Cross-Reference Resolution
We quantify the gap between semantic retrieval
and cross-reference resolution using a microbench-
mark over 10 specifications (TS 38.181–TS 38.304
from release 18). In total, we sample 547 chunks
and extract 903 cross-references. These are filtered
with an LLM helpfulness judge (Appendix A.4)
7

Table 2: Comparative performance on the real-world
QA dataset. Both Chat3GPP and DEEPSPECSreturn
the top-10 chunks. For Base GPT-4o, the win rate is
fixed at 50%.
Model MethodLength
(tokens)Win Rate
vs. GPT-4o (%)
GPT-4oBase Model 295 (50)
Chat3GPP 295 62.5
DEEPSPECS30168.1
Qwen3-4BBase Model 307 34.4
Chat3GPP 427 67.3
DEEPSPECS41571.2
Qwen3-8BBase Model 346 50.8
Chat3GPP 490 74.4
DEEPSPECS48379.8
Qwen3-14BBase Model 331 64.7
Chat3GPP 459 78.9
DEEPSPECS46083.1
Qwen3-32BBase Model 373 73.6
Chat3GPP 499 82.9
DEEPSPECS49788.0
Claude 3.5 HaikuBase Model 268 48.6
Chat3GPP 278 56.6
DEEPSPECS27859.4
GPT-4.1 miniBase Model 395 82.0
Chat3GPP 424 87.8
DEEPSPECS44290.1
GPT-4.1Base Model 427 87.2
Chat3GPP 494 93.7
DEEPSPECS51195.4
Table 3: Cross-reference resolution microbenchmark
performance. Precision, recall, and f1 are calculated on
the filtered set of "helpful" references.
Method Precision Recall F1
Chat3GPP 0.0709 0.2334 0.1031
DEEPSPECS0.1876 0.7055 0.2837
and validated by experts to form a set of helpful
references, on which we calculate the precision,
recall, and f1 score. We compare the retrievals of
(a) Chat3GPP and (b) DEEPSPECS’s standalone
cross-reference resolver to this set of helpful refer-
ences. To ensure a fair comparison, we fix retrieval
parameters ( k1=3,k2=2,k3=0) for DEEPSPECS
and have Chat3GPP return the top-5 chunks. Ta-
ble 3 shows substantially higher precision, recall
and F1 for DEEPSPECS, indicating the limitations
of semantic-only retrieval on interlinked specs and
the effectiveness of DEEPSPECS’s cross-reference
resolution method.
5.4 Evaluating Spec-Evolution Reasoning
We isolate the contribution of evolution reasoning
using a CR-focused QA dataset. To control forTable 4: Comparative Performance on the CR-Focused
QA Dataset. Even with reference resolution disabled,
DEEPSPECSoutperforms the base model and Chat3GPP
at comparable output lengths.
Model MethodLength
(tokens)Win Rate
vs. GPT-4o (%)
GPT-4oBase Model 273 (50)
Chat3GPP 265 59.3
DEEPSPECS26977.0
confounds, we disable reference resolution and fix
retrieval parameters ( k1=5,k2=0,k3=5), ensur-
ing DEEPSPECSand Chat3GPP retrieve the same
number of chunks. Table 4 reports GPT-4o results.
Relative to both the base model and Chat3GPP,
DEEPSPECS(with reference resolution disabled)
still achieves significantly higher win rates while
producing outputs of comparable length. Results
for additional models are provided in Appendix C
and show the same trend. Rubric-based evalua-
tion reinforces these findings. On this CR-focused
subset, DEEPSPECSachieves a mean quality score
of 3.05, compared to 2.48 for Chat3GPP and 2.26
for the base model (all differences significant at
p<0.0001, detailed scores in Appendix C).
6 Conclusion
We introduce DEEPSPECS, a QA framework in-
fused with the structural and temporal under-
standing of 3GPP documents. DEEPSPECSper-
forms clause-level cross-reference resolution and
specification-evolution reasoning, which are ca-
pabilities missing in standard RAG but essential
for accurate and helpful question answering in 5G.
Evaluations on real-world questions and targeted
microbenchmarks demonstrate consistent gains
over strong LLM and RAG baselines, with both
structural and temporal components contributing
to performance.
Limitations
Coverage of Specifications and TDocs.Our
study focuses on English NR (3GPP) specifica-
tions, primarily Releases 17–18, and question types
skewed toward RAN/PHY procedures. For TDocs,
we currently include only CR-type documents. We
have not evaluated on other TDoc types, earlier
or future releases, or multilingual corpora, so our
claims should not be over-generalized.
8

Structural Assumptions on Documents.The
method assumes consistent clause numbering,
machine-parseable cross-references, and semi-
templated Change Requests (CRs). While this
holds for most 3GPP documents, exceptions do oc-
cur (e.g., atypical numbering, unusual references,
or editorial inconsistencies). Such cases can re-
duce the reliability of linking and reasoning across
documents.
Dependence on LLM-Based Processing.Our
pipeline makes extensive use of large language
models for chunking, extraction, and metadata han-
dling. This introduces potential errors and com-
putational or API calling cost. However, most of
the processing is one-time or incremental with new
releases, and we found that budget-oriented models
are sufficient for these tasks. Still, downstream ac-
curacy ultimately depends on model quality, which
may vary with version updates.
Evaluation.(1) Our annotated datasets are lim-
ited in size due to the need for careful expert cu-
ration and limited number of high-quality data
sources. This constraint may affect statistical sig-
nificance, and performance estimates should be
interpreted with caution. (2) For our evaluation,
some evaluation metrics rely on LLM-based judges
or scorers. Although we include human validation,
LLM evaluators can reward fluency or plausible
reasoning over our desired factors, and prompt bias
may persist. We view LLM-based evaluation as a
useful complement rather than a full substitute for
human evaluation.
Potential Risks.Our system generates fluent,
citation-backed outputs, but answers may still be in-
complete or misleading if context is missed. Over-
reliance on such outputs without cross-checking
against the specifications could encourage misin-
terpretation or non-compliant implementations. As
our work is based on public 3GPP standards, direct
risks related to privacy, security, or fairness are min-
imal. We encourage deployment with safeguards
such as citation enforcement, uncertainty reporting,
and access controls.
References
3rd Generation Partnership Project (3GPP). 2024a. 5g
system overview. Accessed: 2025-07-28.3rd Generation Partnership Project (3GPP). 2024b.
Specifications and technologies. Accessed: 2025-
07-28.
Anthropic. 2024. Claude 3.5 haiku. https://www.
anthropic.com/claude/haiku . Accessed: 2025-
10-06.
Justus Baron and Kirti Gupta. 2018. Unpacking 3gpp
standards.Journal of Economics & Management
Strategy, 27(3):433–461.
Andrei-Laurentiu Bornea, Fadhel Ayed, Antonio
De Domenico, Nicola Piovesan, and Ali Maatouk.
2024. Telco-rag: Navigating the challenges of
retrieval augmented language models for telecom-
munications. InGLOBECOM 2024-2024 IEEE
Global Communications Conference, pages 2359–
2364. IEEE.
Andrei-Laurentiu Bornea, Fadhel Ayed, Antonio
De Domenico, Nicola Piovesan, Tareq Si Salem,
and Ali Maatouk. 2025. Telco-orag: Optimizing
retrieval-augmented generation for telecom queries
via hybrid retrieval and neural routing.arXiv preprint
arXiv:2505.11856.
Yi Chen, Di Tang, Yepeng Yao, Mingming Zha, Xi-
aoFeng Wang, Xiaozhong Liu, Haixu Tang, and
Dongfang Zhao. 2022. Seeing the forest for the
trees: Understanding security hazards in the {3GPP }
ecosystem through intelligent analysis on change
requests. In31st USENIX Security Symposium
(USENIX Security 22), pages 17–34.
Kaustubh Dhole, Kai Shu, and Eugene Agichtein. 2025.
ConQRet: A new benchmark for fine-grained au-
tomatic evaluation of retrieval augmented computa-
tional argumentation. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 5687–5713, Albuquerque, New Mexico.
Association for Computational Linguistics.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Long Huang, Ming Zhao, Limin Xiao, Xiujun Zhang,
and Jungang Hu. 2025. Chat3gpp: An open-source
retrieval-augmented generation framework for 3gpp
9

documents. In2025 IEEE International Conference
on Communications Workshops (ICC Workshops),
pages 492–497.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Ali Maatouk, Kenny Chirino Ampudia, Rex Ying, and
Leandros Tassiulas. 2024. Tele-llms: A series of
specialized large language models for telecommuni-
cations.arXiv preprint arXiv:2409.05314.
Rasoul Nikbakht, Mohamed Benzaghta, and Giovanni
Geraci. 2024. Tspec-llm: An open-source dataset
for llm understanding of 3gpp specifications.arXiv
preprint arXiv:2406.01768.
OpenAI. 2024a. Gpt-4o system card. https://
openai.com/index/gpt-4o-system-card/ . Ac-
cessed: 2025-10-06.
OpenAI. 2024b. Openai text-embedding-
3-large. https://openai.com/index/
new-embedding-models-and-api-updates/.
OpenAI. 2025. Gpt-4.1 (and variants: mini,
nano). https://openai.com/index/gpt-4-1/ .
Accessed: 2025-10-06.
OpenAirInterface. 2022. Walkthrough 3gpp specifica-
tions. Accessed: 2025-07-28.
Qualcomm. 2017. Understanding 3gpp: Starting with
the basics. Accessed: 2025-07-28.
Felipe A Rodriguez Y . 2025. Technical language pro-
cessing for telecommunications specifications.Ap-
plied AI Letters, 6(2):e111.
Jaeku Ryu. Sharetechnote. https://www.
sharetechnote.com. Accessed: 2025-10-04.
ShareTechnote. 5g frame structure and related discus-
sions: Background context and rationale. Accessed:
2025-07-28.
telecomHall. telecomhall forum. https://www.
telecomhall.net. Accessed: 2025-10-04.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
10014–10037.Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung
Sum Thomas Kwok, Muzhi Li, Zhuhong Li, Hailin
He, Yuchen Hua, Peng Lu, Suyuchen Wang, and 1
others. 2025. Finsage: A multi-aspect rag system for
financial filings question answering.arXiv preprint
arXiv:2504.14493.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Wutong Zhang, Hefeng Zhou, Qiang Zhou, Yunshen
Li, Yuxin Liu, Jiong Lou, Chentao Wu, and Jie Li.
2025a. Towards comprehensive legal document anal-
ysis: A multi-round rag approach. InProceedings
of the 2025 International Conference on Multimedia
Retrieval, pages 1840–1848.
Ze Yu Zhang, Zitao Li, Yaliang Li, Bolin Ding,
and Bryan Kian Hsiang Low. 2025b. Respecting
temporal-causal consistency: Entity-event knowl-
edge graphs for retrieval-augmented generation.
arXiv preprint arXiv:2506.05939.
A Prompt Templates
A.1 Pairwise Win Rate
We use the following prompt for the pairwise eval-
uator:
# Task
You are an expert in the 5G domain. You will be
provided with a 5G-related question, an
expert-written reference answer, and two
candidate answers to evaluate. Your task is to
determine which candidate answer is better
based on the reference answer and the following
criteria:
- **Answer Accuracy**: How factually correct is
each candidate answer compared to the reference
answer? Are there any inaccuracies, omissions,
or misleading statements?
- **Explanation Helpfulness**: How well does
each candidate answer explain the reasoning
behind itself? Does it provide useful context,
background information, or insights that
enhance understanding of the answer?
# Question
{question}
# Reference Answer
{ground_truth}
# Candidate Answers
- Answer A: {answer_a}
- Answer B: {answer_b}
# Instructions
Provide brief reasoning based on the criteria
above, followed by your final decision in the
format below:
```
Reasoning: <Your brief reasoning>
Judgment: <"Answer A" or "Answer B">
```
10

A.2 Rubric-based scoring
We use the following prompt to generate the initial
question-specific for each QA pairs:
You are an expert in 5G telecommunications and
assessment design. Create a detailed scoring
rubric for the following question based on the
ground truth answer provided.
Question: {question}
Ground Truth Answer: {ground_truth}
Create a scoring rubric with a total of 5
points. Break down the points based on:
1. Core answer/concept (typically 2-3 points)
2. Key details/explanation (typically 1-2
points)
3. Technical accuracy (typically 0.5-1 point)
4. Completeness (typically 0.5-1 point)
Return ONLY a JSON object with this structure:
{
"total_points": 5,
"criteria": [
{"description": "criterion description",
"points": X},
{"description": "criterion description",
"points": Y}
]
}
Make the rubric specific to this question. Do
not include any other text outside the JSON.
After human curation and validation, we use the
following prompt to score each answer with the
question-specific rubrics:
You are an expert grader for 5G
telecommunications questions. Grade the
predicted answer against the ground truth using
the provided rubric.
Question: {question}
Ground Truth Answer: {ground_truth}
Predicted Answer: {predicted_answer}
Scoring Rubric (Total: {total_points} points):
{criteria_text}
Carefully evaluate the predicted answer against
each criterion in the rubric. Award points
based on:
- Factual correctness compared to ground truth
- Completeness of the answer
- Technical accuracy of terminology
- Whether key points from ground truth are
covered
Return ONLY a JSON object with this exact
structure:
{
"score": X.X,
"grading_reasoning": "Detailed explanation of
the score, breaking down points
awarded/deducted for each criterion"}
The score should be a number between 0 and
{total_points}, and can include decimals (e.g.,
3.5, 4.0).
Be strict but fair. If the predicted answer
contradicts the ground truth or misses key
points, deduct points accordingly.
Do not include any text outside the JSON object.
A.3 CR QA Generation Prompt
We use the following prompt to generate question-
answer pairs from approved Change Request docu-
ments. The prompt instructs the language model to
create questions in one of two formats depending
on the clarity of before-and-after states in the spec-
ification text, and to skip CRs that lack sufficient
technical detail for meaningful QA generation.
You are analyzing a 3GPP Change Request (CR)
document. Based on the information provided,
generate ONE question-answer pair.
CR Information:
- Specification: {spec_number}
- Summary of Change: {summary}
- Reason for Change: {reason}
- Consequences if Not Approved: {consequence}
- Clauses Affected: {clauses_affected}
Decision rule BEFORE producing output:
- If AND ONLY IF the Summary of Change, Reason
for Change, AND Consequences are all very short
one-liners (i.e., each is so brief that you
cannot identify a meaningful technical change
and rationale), then DO NOT create a Q&A.
Instead, return:
{
"skip": true,
"reason": "why the three fields are too
simple to form a meaningful Q&A"
}
Otherwise, create exactly ONE Q&A with the
following instructions.
Instructions for Q&A:
1) Decide if you can identify clear "before"
and "after" states in the spec text from the
provided info.
2) Generate a question in ONE of these formats:
Format A (if before AND after states are
clear; do NOT mention the spec number in the
question):
- Example templates:
* "Why was [specific parameter/behavior X]
changed to [specific parameter/behavior Y]?"
* "What is the reason for changing
[feature/requirement X] to
[feature/requirement Y]?"
* "I observed that [X] was modified to [Y].
What is the rationale for this change?"
Format B (if before OR after is unclear;
explicitly mention the specification number):
11

- Example templates:
* "What is the reason for the change to
[specific aspect] in specification
{spec_number}?"
* "Why was [feature/behavior] modified in
3GPP TS {spec_number}?"
* "I observed a change regarding [topic] in
specification {spec_number}. What is the
reason?"
3) The answer should:
- Explain the reason for the change,
- Include the consequences if the change is
not accepted,
- Be natural and technical (no copy-paste;
use proper terminology).
Output format (strict JSON):
- If skipping:
{
"skip": true,
"reason": "short explanation"
}
- If generating Q&A:
{
"question": "your generated question",
"answer": "your generated answer",
"format_used": "A" or "B"
}
All generated question-answer pairs undergo
manual review to verify technical accuracy, en-
sure natural phrasing, and confirm that answers
adequately explain both the rationale for the speci-
fication change and the potential consequences if
the change were not approved.
A.4 Reference Helpfulness Filter
We use the following prompt to filter the helpful ref-
erences for the microbenchmark on cross-reference
resolution:
Compare the original chunk to the reference
chunk.
original chunk: {org_chunk}
reference chunk: {ref_chunk}
Instructions:
1. See if the reference chunk provides
information that would be useful to understand
what the original chunk is talking about.
2. Return'helpful'if the reference chunk's
content will provide necessary detail to fully
understand what the original chunk's content is
talking about.
3. Return'unhelpful'if the reference chunk's
content is not necessary to fully understand
what the original chunk is talking about.
4. Respond only with'helpful'or'unhelpful'
B Additional Results on Real-World QA
B.1 Dataset Categorization
The 573 questions in our real-world QA dataset are
organized into five technical categories that reflectthe organizational structure of 5G standardization
and engineering practice:
•Category 1: Air Interface & Physical Layer
(186 questions, 32.5%) – Physical channel
structures, modulation schemes, MIMO con-
figurations, beamforming, and radio resource
allocation in the NR air interface.
•Category 2: Protocol Stack & Resource
Management(174 questions, 30.4%) – Layer
2/3 protocol operations (MAC, RLC, PDCP,
SDAP), quality-of-service mechanisms, nu-
merology, and resource mapping procedures.
•Category 3: Core Network & Session Man-
agement(114 questions, 19.9%) – 5G Core
architecture, session establishment and mobil-
ity procedures, network function interactions,
and signaling flows.
•Category 4: Network Architecture & De-
ployment(66 questions, 11.5%) – Network
slicing, multi-access architectures, deploy-
ment scenarios (NSA/SA), interworking with
legacy systems, and infrastructure planning.
•Category 5: Operations, Optimization &
Troubleshooting(33 questions, 5.8%) – Per-
formance optimization strategies, diagnostic
procedures, measurement reporting, and oper-
ational best practices.
The distribution reflects the relative emphasis in
5G technical documentation and practitioner dis-
cussions, with lower-layer protocol mechanics and
physical-layer operations receiving the most atten-
tion. Each question is assigned to exactly one cate-
gory by the annotation team based on its primary
technical focus.
B.2 Additional Evaluation
Table 5 presents additional rubrics-scoring results.
Table 8 presents the Per-category win rates against
the GPT-4o baseline across five 5G technical do-
mains, demonstrating the performance of base
models, Chat3GPP, and DEEPSPECSfor different
model families and sizes.
C Additional Results on CR-focused QA
Table 6 presents additional pairwise win rate results.
Table 7 presents additional rubric-based scoring
results.
12

Table 5: Rubric-based scoring results comparing DEEP-
SPECSagainst base models. Scores range from 0 to
5. All differences significant at p<0.0001 (Wilcoxon
signed-rank test).
Metric GPT-4.1 GPT-4.1-mini
Baseline Mean 2.67 2.46
DEEPSPECSMean 3.29 3.06
Improvement +0.62*** +0.60***
Effect Size (dz) 0.41 0.40
Table 6: Additional results on the CR-focused QA
dataset.
Model MethodLength
(tokens)Win Rate
vs. GPT-4o (%)
Qwen3-4BBase Model 283 68.6
Chat3GPP 372 81.4
DEEPSPECS34188.0
Qwen3-8BBase Model 312 73.3
Chat3GPP 466 85.7
DEEPSPECS42792.6
Qwen3-14BBase Model 295 81.1
Chat3GPP 481 91.4
DEEPSPECS44393.6
Qwen3-32BBase Model 353 89.9
Chat3GPP 549 93.7
DEEPSPECS50695.7
D Implementation Details
D.1 Inference Hyperparameters
We adopt default or recommended settings for in-
ference across APIs and locally deployed models:
•OpenAI models (GPT-4o, GPT-4.1):tem-
perature = 1.0, top_p = 1.0
•Claude 3.5 Haiku:temperature = 1.0, top_p
= 1.0
•Qwen3 models:temperature = 0.6, top_p =
0.95, top_k = 20, with thinking mode enabled
D.2 Computational Infrastructure
Hardware and API Access.Experiments with
GPT and Claude models were conducted via of-
ficial APIs. For Qwen3 models, inference was
performed locally using vLLM on a single NVIDIA
A100 80GB GPU.
D.3 Other Experimental Settings
Unless otherwise specified:
•DEEPSPECScross-reference resolution, the
recursion depth is fixed at 2.Table 7: Rubric-based scoring results on CR-focused
QA using GPT-4o as the base model. Scores range from
0 to 5. All pairwise differences significant at p<0.0001
(Wilcoxon signed-rank test).
Metric Score
Base Model Mean 2.26
Chat3GPP Mean 2.48
DEEPSPECSMean 3.05
DEEPSPECSvs. Base +0.79 (0.68) ***
DEEPSPECSvs. Chat3GPP +0.58 (0.51) ***
Chat3GPP vs. Base +0.21 (0.26) ***
Effect sizes (dz) shown in parentheses.
•All embeddings are obtained using
text-embedding-3-large.
•GPT-4o serves as the default assistant, extrac-
tor, generator, and evaluator model.
13

Table 8: Per-category performance on the real-world QA dataset (Win Rate vs. GPT-4o, %).
Model MethodAir
InterfaceProtocol
StackCore
NetworkNetwork
Arch.Operations &
TroubleshootingOverall
GPT-4oBase Model (50) (50) (50) (50) (50) (50)
Chat3GPP 54.8 67.8 64.5 62.1 69.7 62.4
DEEPSPECS60.5 72.1 72.4 66.7 77.3 68.1
Qwen3-4BBase Model 33.6 30.7 38.6 39.4 33.3 34.4
Chat3GPP 59.9 70.4 71.9 72.0 66.7 67.3
DEEPSPECS62.9 76.1 74.6 75.0 72.7 71.2
Qwen3-8BBase Model 47.6 50.6 50.4 56.1 60.6 50.8
Chat3GPP 72.0 78.4 71.9 74.2 75.8 74.4
DEEPSPECS73.4 83.6 80.3 86.4 80.3 79.8
Qwen3-14BBase Model 56.5 61.5 65.4 65.2 63.6 61.2
Chat3GPP 73.985.979.8 75.8 77.3 79.1
DEEPSPECS75.885.682.0 87.9 81.8 81.8
Qwen3-32BBase Model 67.7 77.0 71.9 78.0 86.4 73.6
Chat3GPP 79.6 87.4 84.2 78.0 83.3 82.9
DEEPSPECS82.3 92.5 86.8 92.4 92.4 88.0
Claude 3.5
HaikuBase Model 47.8 50.9 53.1 41.7 39.4 48.6
Chat3GPP 51.6 62.4 57.5 53.0 59.1 56.6
DEEPSPECS53.0 64.4 61.8 58.3 63.6 59.4
GPT-4.1 miniBase Model 86.3 81.3 81.1 75.0 78.8 82.0
Chat3GPP 84.4 89.189.988.6 90.9 87.8
DEEPSPECS87.6 91.489.592.4 93.9 90.1
GPT-4.1Base Model 89.2 87.6 83.8 84.8 89.4 87.2
Chat3GPP 92.5 93.4 94.397.792.4 93.7
DEEPSPECS93.8 95.4 96.1 97.7 97.0 95.4
14
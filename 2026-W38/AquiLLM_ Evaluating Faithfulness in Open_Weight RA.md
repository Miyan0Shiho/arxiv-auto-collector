# AquiLLM: Evaluating Faithfulness in Open-Weight RAG-LLM Systems for Scientific Research

**Authors**: Bernie Boscoe, Srinath Saikrishnan, Vikram Seenivasan, Jack Stark, Andrew Lizarraga, Morgan Himes, Jonathan Soriano, PJ Allen, Tuan Do

**Published**: 2026-09-15 02:09:45

**PDF URL**: [https://arxiv.org/pdf/2609.16519v2](https://arxiv.org/pdf/2609.16519v2)

## Abstract
Scientific research increasingly relies on large, heterogeneous data sources, motivating interest in retrieval-augmented generation (RAG) systems that provide natural language access to scientific knowledge and research workflows. Researchers are exploring the viability of these systems as natural language interfaces for document search and for generating analysis code and pipeline components. At the same time, concerns about data privacy and control over research infrastructure have motivated interest in open-weight models and open-source deployments hosted within research institutions. In astronomy, this development follows a long history of computational infrastructure development, from archival databases and Structured Query Language (SQL)-based systems to large language model (LLM)-assisted research tools. This paper presents a domain-expert evaluation of faithfulness for AquiLLM, an open-weight, offline RAG-LLM platform designed to support scientific research groups in the use and preservation of tacit and formal knowledge. We define faithfulness as the extent to which generated responses remain grounded in retrieved scientific context without unsupported claims or omissions. We report results from an astronomy case study evaluating AquiLLM across retrieval and scientific analysis tasks. AquiLLM performs most reliably on explicit retrieval-oriented questions grounded in the RAG collection, while faithfulness degrades for queries requiring synthesis or ambiguity resolution. These results highlight both the promise and limitations of open-weight RAG-LLM systems for scientific research and demonstrate the importance of domain-expert evaluation beyond standard benchmark leaderboards.

## Full Text


<!-- PDF content starts -->

AquiLLM: Evaluating Faithfulness in Open-Weight
RAG-LLM Systems for Scientific Research
Bernie Boscoe∗, Srinath Saikrishnan§, Vikram Seenivasan†, Jack Stark†, Andrew Lizarraga‡,
Morgan Himes†, Jonathan Soriano†, PJ Allen∗, Tuan Do†
∗Dept. of Computer Science, Southern Oregon University, USA
†Dept. of Physics & Astronomy, University of California, Los Angeles, USA
‡Dept. of Statistics, University of California, Los Angeles, USA
§Dept. of Computer Science, University of California, Los Angeles, USA
boscoeb@sou.edu, srinathsai22@ucla.edu, vikrams25@ucla.edu,
jstark@astro.ucla.edu, andrewlizarraga@ucla.edu, jsoriano@astro.ucla.edu,
morganhimes@ucla.edu, pricep@sou.edu, tdo@astro.ucla.edu
Abstract—Scientific research increasingly relies on large,
heterogeneous data sources, motivating interest in retrieval-
augmented generation (RAG) systems that provide natural lan-
guage access to scientific knowledge and research workflows.
Researchers are exploring the viability of these systems as natural
language interfaces for document search and for generating anal-
ysis code and pipeline components. At the same time, concerns
about data privacy and control over research infrastructure
have motivated interest in open-weight models and open-source
deployments hosted within research institutions.
In astronomy, this development follows a long history
of computational infrastructure development, from archival
databases and Structured Query Language (SQL)-based systems
to large language model (LLM)-assisted research tools. This
paper presents a domain-expert evaluation of faithfulness for
AquiLLM, an open-weight, offline RAG-LLM platform designed
to support scientific research groups in the use and preservation
of tacit and formal knowledge.
We define faithfulness as the extent to which generated
responses remain grounded in retrieved scientific context without
unsupported claims or omissions. We report results from an
astronomy case study evaluating AquiLLM across retrieval and
scientific analysis tasks. AquiLLM performs most reliably on
explicit retrieval-oriented questions grounded in the RAG collec-
tion, while faithfulness degrades for queries requiring synthesis
or ambiguity resolution. These results highlight both the promise
and limitations of open-weight RAG-LLM systems for scientific
research and demonstrate the importance of domain-expert
evaluation beyond standard benchmark leaderboards.
Index Terms—retrieval-augmented generation, large language
models, scientific workflows, natural language interfaces, faith-
fulness evaluation, astronomy, open-weight models
I. INTRODUCTION
As scientific data systems grow in scale and complexity,
natural language interfaces can lower barriers for researchers
seeking to access increasingly heterogeneous forms of knowl-
edge [1]. Modern scientific workflows require navigating data
and code distributed across large, often non-interoperable
platforms. Research knowledge also exists in less formal,
tacit forms, including unpublished manuscripts, meeting notes,
emails, software documentation, and Jupyter notebooks. The
fragmentation of these sources creates substantial cognitive
and technical overhead for researchers.Astronomy has long been a leader in developing computa-
tional data infrastructure and large-scale scientific archives in
response to the growing scale and complexity of observational
data. Projects such as the Sloan Digital Sky Survey trans-
formed access to astronomical data through searchable repos-
itories, database-driven architectures, and SQL-based query
interfaces developed through collaborations working at the
intersection of astronomy and computer science [2]. These
systems helped establish a broader paradigm for data-intensive
science in which large observational datasets could be explored
remotely through web portals and structured database queries.
Although such SQL-based systems significantly democratized
access to scientific data, they still require familiarity with
database schemas and query syntax.
Natural language interfaces offer a way to reduce this
complexity by allowing researchers to interact with scientific
systems conversationally without requiring specialized query
languages or software expertise [3], [4]. Retrieval-augmented
large language model (RAG-LLM) systems are particularly
relevant in this transition from structured SQL query systems
toward natural-language scientific interfaces [5]. Recent ad-
vances in LLMs, RAGs, and accessible computational infras-
tructure have accelerated interest across scientific domains [6],
[7]. Despite this momentum, many research groups remain
cautious about relying on commercial AI platforms for work-
flows involving unpublished data and proprietary analysis.
Concerns surrounding privacy, reproducibility, and cost have
motivated growing interest in open-weight models and locally
deployed RAG-LLM systems.
AquiLLM was developed as an open-source open-weight
platform designed to help research groups collectively preserve
and interact with both formal and tacit knowledge [8], [9]. We
useopen-weightto refer to models whose trained parameters
are publicly available, while the underlying training data, code,
and methodology remain unavailable. These models can still
be locally deployed and fine-tuned. The AquiLLM system
allows researchers to create personal or group collections and
query them using natural language to support information
retrieval and onboarding of new group members. Additionally,
arXiv:2609.16519v2  [cs.AI]  16 Sep 2026

AquiLLM is modular, allowing research groups to adapt mod-
els to domain-specific workflows and retrieval over specialized
research corpora.
Open-weight language models are proliferating rapidly, yet
they are still evaluated through benchmark leaderboards and
general-purpose language tasks [10]. While these benchmarks
provide useful comparative metrics, performance on stan-
dardized evaluations does not necessarily reflect reliability
in scientific contexts. Scientific applications require systems
to accurately reflect source material and avoid unsupported
inference. In this work, we definefaithfulnessas the extent
to which generated responses remain grounded in retrieved
scientific context without fabrication or unsupported inference.
Although human evaluation remains the gold standard for
assessing faithfulness, such evaluations are difficult and costly
to conduct at scale [11].
We conduct a domain-expert evaluation of faithfulness in
which astronomers assess AquiLLM responses across cate-
gories of scientific queries created from their research group’s
own data and documentation. The study includes questions
involving both factual retrieval and comparative scientific
analysis. Through this evaluation, we examine how faith-
fulness varies across different forms of scientific interaction
and identify limitations that are not captured by standard
benchmark evaluations. This work establishes a baseline for
understanding the limitations of open-weight offline RAG-
LLM systems in domain science and highlights the importance
of domain-expert evaluation for improving trustworthy RAG-
LLM systems.
This paper makes three contributions. First, we present
AquiLLM, a deployed, modular, open-weight RAG-LLM sys-
tem (Section III) that research groups can adopt directly
for local, privacy-preserving access to their own scientific
knowledge. Second, we introduce a domain-expert evalua-
tion methodology built around five query categories spanning
varying levels of retrieval complexity and reasoning (Sec-
tion IV-D), which other research groups can reuse to evaluate
faithfulness in their own scientific RAG deployments. Third,
we report empirical findings from applying this methodology
to an astronomy research group, identifying six recurring fail-
ure modes and showing that faithfulness degrades specifically
for cross-source synthesis and comparative reasoning rather
than uniformly across query types (Section V). Because this
is an initial, single-group, single-domain study, we treat these
contributions as a baseline evaluation rather than a general
benchmark; Section VII details the resulting scope constraints,
including the absence of a controlled comparison against an
alternative RAG system.
II. RELATEDWORK
Early work on natural language interfaces to databases
explored how users could query conversationally rather than
through formal query languages such as SQL [1]. Natural
language processing (NLP) evolved alongside the growth
of cyberinfrastructure platforms designed to support data-
intensive research workflows [12]. Research software engi-neers, whether identified by this title or otherwise, have played
a central role in the building and maintenance of these systems
to aid computational research [13]. Before the research soft-
ware engineer (RSE) role became more formalized, domain
scientists often relied on graduate students, collaborations
with computer scientists, or researchers working across both
disciplinary and computational domains to develop scientific
software and infrastructure.
Contemporary LLM interfaces emerged from decades of
scientific cyberinfrastructure development, extending earlier
database systems into conversational knowledge interfaces
[14]. Recent RAG architectures combine LLMs with retrieval
systems [15], using vector databases and semantic search
methods to ground responses in domain-specific document
collections. In scientific settings, these systems have been
explored for literature search, scientific question answering,
and interaction with technical documentation and code [6],
[16], [17]. Many domain-specific RAG systems focus on
grounded generation to reduce hallucinations and improve the
reliability of generated responses [18]. AquiLLM builds on
these approaches while emphasizing the preservation of tacit
research knowledge and research-group context [19].
Much of the recent development in LLM interfaces has
occurred through commercial AI platforms and hosted ser-
vices. In contrast, open-weight models and locally devel-
oped RAG-LLM systems give research groups direct control
over how their scientific knowledge systems are configured,
audited, and maintained, without depending on an external
provider’s infrastructure or terms of service. Benchmarking in
other scientific domains has similarly found that open-source
models can match closed-source performance while offering
greater transparency, reproducibility, cost-effectiveness, and
data privacy [20]. AquiLLM builds on this perspective through
a modular, locally-controlled, open-source approach to a RAG-
LLM system.
Central to these systems is response reliability. LLMs may
generate plausible but fabricated or misleading information
[21]. Prior work has explored concepts including correct-
ness, factuality, faithfulness, and groundedness for evaluating
generated responses [22], [23]. In scientific contexts, these
concerns are important because research workflows depend
on accurate interpretation of source material and trustworthy
communication of technical knowledge.
Scientific applications require higher standards of relia-
bility than general-purpose conversational systems. Plausible
but unsupported claims or incorrect synthesis across docu-
ments undermines trust in LLM-enabled workflows. Retrieval-
augmented systems aim to reduce these risks by grounding
responses in retrieved evidence, though questions remain
regarding how faithfully such systems represent scientific
source material in practice [24], [25]. Benchmark leaderboards
provide useful comparative metrics but do not adequately
capture reliability in domain-specific scientific settings [10].
Evaluating specialized scientific knowledge requires contex-
tual understanding and familiarity with research workflows
that automated metrics may fail to detect [18]. As a result,

Fig. 1. High-level AquiLLM architecture. Background workers handle document ingestion and indexing, while runtime requests pass through a prompt
orchestration layer coordinating retrieval, memory, and tool use. Local inference is served through vLLM using separate models for chat, embedding, reranking,
and multimodal processing.
human evaluation remains an important approach for assessing
faithfulness and reliability in scientific systems. However,
expert annotation is difficult and costly. These challenges
motivate contextual evaluation frameworks in which domain
experts assess responses grounded in their own research-
group knowledge collections, rather than relying solely on
standardized benchmark tasks developed by model builders.
III. AQUILLM SYSTEMOVERVIEW
AquiLLM is an open-source browser-based RAG-LLM tool
designed to support research groups through natural language
interaction with scientific knowledge collections [8]. Users can
create personal or shared collections containing both published
and private data, then query these materials conversationally.
The system supports information retrieval while preserving
tacit knowledge embedded in scientific workflows and research
collaboration. Originally developed for astronomy and envi-
ronmental science research groups, AquiLLM is designed as
a general framework for research-group knowledge systems.
AquiLLM source code and deployment configuration are
openly available through the GitHub project repository, allow-
ing research groups to build and deploy their own AquiLLM
instances. Persistent public deployments remain constrained
by the computational resources required to support locally
hosted open-weight models and multimodal retrieval work-
loads. Our current research deployments operate on dedicatedGPU infrastructure to support interactive workflows and con-
current users. We periodically provide live demonstrations
while exploring more computationally accessible deployment
configurations for research groups.
A. Astronomy Case Study
For this study, we evaluated an astronomy-specific deploy-
ment of AquiLLM built around astronomy and machine learn-
ing research collections. Participating astronomers prepared a
dedicated collection consisting of survey documentation and
publications used in their research. For this initial study, we
focused on formal scientific artifacts, including unpublished
manuscripts and technical documents, rather than broader tacit
workflow and collaborative research materials. This allowed us
to establish a baseline evaluation of faithfulness before incor-
porating more complex research-group knowledge collections.
B. System Architecture
AquiLLM uses a modular RAG architecture organized
around ingestion, retrieval, orchestration, and inference com-
ponents. Documents are uploaded through a browser-based
interface, processed asynchronously, chunked, and indexed
for semantic retrieval, shown in Fig. 1. The system supports
heterogeneous scientific materials including PDFs, figures,
and LaTeX-based mathematical notation. During query-time
execution, retrieved document chunks are combined with user

prompts through an orchestration layer that grounds responses
in domain-specific collections.
The architecture supports local deployment using open-
weight models rather than commercial AI APIs. AquiLLM
separates generation, embedding, reranking, and OCR into
independent components, allowing retrieval and multimodal
workloads to operate separately from the primary conver-
sational model. This modular design improves extensibility
across research domains while allowing research groups to
modify retrieval pipelines, models, and document collections
without changing the overall framework. The astronomy de-
ployment evaluated in this study was hosted on Jetstream2
using dedicated H100 GPU resources. Recent development
has focused on local embeddings, reranking, vLLM-based
inference, and memory-layer support for multimodal retrieval
and private deployment workflows.
C. Model Selection and Configuration
The astronomy deployment evaluated in this study used
locally hosted Qwen-based open-weight models integrated
through the AquiLLM inference pipeline. The primary con-
versational model wasQwen3.5-27B-Claude-4.6-OS
-Auto-Variable-Thinking[26], an independent com-
munity fine-tune of the official Qwen3.5-27B base model [27],
selected for stronger reasoning behavior and more consistent
tool usage than the base model in our own informal testing.
This fine-tune was produced by a third-party contributor (not
Alibaba, the Qwen family’s publisher) by further training
the base model on a ”Claude-4.6-OS” dataset of Claude-
derived examples, intended to introduce Claude-style variable-
length reasoning (”thinking”) behavior into the base model. Its
behavior is therefore not directly covered by Alibaba’s own
published Qwen benchmarks [27], and, to our knowledge, no
independent technical report or benchmark evaluation of this
specific fine-tune exists; our selection was based on informal
comparison against the base Qwen3.5-27B model on our own
workflows rather than a published evaluation. Both the base
model and this fine-tune are released under a permissive
Apache 2.0 license and support local deployment and further
fine-tuning.
The deployment additionally used separate Qwen-based
models for embedding, reranking, and multimodal processing,
allowing retrieval and document-processing workloads to op-
erate independently from the conversational model. This mod-
ular approach improved flexibility while supporting heteroge-
neous scientific workflows and multimodal retrieval tasks.
Model selection was guided by reasoning quality, retrieval
performance, computational cost, and feasibility within lo-
cally managed infrastructure. The selected configuration fit
within the available H100 GPU resources on Jetstream2 while
supporting retrieval-augmented prompting and concurrent in-
teractive usage. Larger models were considered, but many
exceeded practical GPU memory limits or introduced latency
that reduced usability. The deployment therefore represents
a balance between model capability and deployability within
institutional computing environments. Future work will ex-TABLE I
RETRIEVAL,CHUNKING,AND RERANKING CONFIGURATION USED DURING
THE EVALUATION DEPLOYMENT.
Setting Value
Embedding model Qwen3-VL-Embedding-2B (1024-dim)
Reranker model Qwen3-VL-Reranker-2B
Chunk size 2048 characters
Chunk overlap 384 characters
Vector retrieval top-k12
Lexical (trigram) retrieval top-k12
Candidate fan-out multiplier 3×
plore domain-specific fine-tuning using astronomy corpora and
evaluation feedback collected through AquiLLM.
D. Retrieval and Indexing Configuration
To support reproducibility, Table I reports the retrieval,
chunking, and reranking configuration used during the eval-
uation deployment. Documents are split into overlapping
character-based chunks, embedded, and indexed for retrieval.
At query time, AquiLLM performs a hybrid search that pools
candidates from dense vector similarity and lexical (trigram)
matching, expands the candidate pool by a fixed multiplier
beyond the requested top-k, and reranks the pooled candidates
before passing the top results to the generation model. These
values correspond to the project’s default configuration at
the time of the study (source-controlled configuration as of
March 31, 2026, immediately preceding the evaluation period)
and are exposed as environment variables so that individual
deployments can retune them for their own corpora [8].
Generated responses are additionally subject to citation
enforcement: the generation pipeline requires cited references
to map only to chunks actually returned by the retrieval tools,
with a fail-safe fallback that produces a cited response if the
model does not otherwise comply. In the AquiLLM interface,
citations in generated responses are clickable and open a
slide-out panel showing the originating document with the
cited passage highlighted, allowing users to directly verify a
claim against its source material. This citation-enforcement
mechanism was implemented as of March 31, 2026, the
same source-controlled snapshot reported in Table I, and was
therefore active throughout the evaluation.
IV. STUDYDESIGN ANDMETHODS
A. Participants
The study involved five participants from the same astron-
omy research group with expertise in observational astron-
omy and machine learning workflows. Participants regularly
collaborated on publications, datasets, and research software
associated with the group’s astronomy projects and had prior
familiarity with AquiLLM for approximately one year.
B. Corpus and Collection Construction
Participants constructed a dedicated AquiLLM collection
consisting of 31 scientific documents and technical references
associated with the group’s astronomy and machine learning
workflows. The collection included survey documentation and

instrumentation papers related to the Hyper Suprime-Cam
Survey (HSC), the Euclid Mission, and the Rubin Observa-
tory Legacy Survey of Space and Time (LSST), along with
the group’s own publications, unpublished work, and techni-
cal references connected to the group’s GalaxiesML dataset
research efforts. The 31-document collection comprised 15
peer-reviewed journal articles, 4 conference papers, and 12
additional technical resources (preprints, mission and survey
documentation, and dataset descriptions), listed in full in
the accompanying test-collection bibliography. We did not
separately measure the corpus’s indexed storage footprint (e.g.,
chunk or token count); Section VI-C discusses how faithful-
ness and retrieval behavior may change for substantially larger
collections, which this study does not evaluate.
The corpus combined straightforward factual materials with
more complex research-oriented content. Evaluation tasks in-
cluded both direct retrieval questions and more challenging
interpretation and comparative analysis tasks involving redshift
prediction models, emission-line reconstruction, and machine
learning performance metrics. The collection was uploaded
into AquiLLM for shared use by participants.
C. Evaluation Procedure
Participants contributed an initial pool of candidate domain-
specific questions spanning the major projects in their research
group (survey and instrumentation documentation, the group’s
GalaxiesML dataset and MMAE model work, and general
machine-learning/statistics onboarding topics), together with
draft reference answers grounded in their own research activi-
ties, datasets, workflows, and publications. From this pool, the
research team selected ten representative questions (Table V
in the Appendix) for the primary evaluation. Questions were
selected to keep the rating workload manageable for a five-
participant panel while still spanning all five query categories
(Section IV-D) and each of the group’s major project areas,
rather than being sampled at random or exhaustively evaluating
the full candidate pool. Participants then collaboratively es-
tablished consensus reference answers for these ten questions;
reference answers were discussed and finalized as a group dur-
ing the research group’s regular astronomy research meetings,
rather than by a single arbiter.
To account for variability in generation behavior, each
participant independently submitted each of the ten questions
to AquiLLM up to three times through their own chat session,
generating and then rating their own responses through the
built-in evaluation form; no two participants rated the same
generated response text. This yielded up to 150 independent
generation-and-rating instances (10 questions×5 participants
×3 submissions); after removing incomplete submissions,
the final evaluation dataset contained 141 rated responses
(Table II), consistent with the per-question counts of 12–15
completed ratings reported in Section V-B. Participants re-
viewed generated responses and assessed faithfulness relative
to the retrieved scientific context and their domain expertise.
Responses were evaluated using a five-point scale measuring
factual accuracy, grounding, preservation of scientific con-TABLE II
DISTRIBUTION OF QUERY CATEGORIES IN THE EVALUATION DATASET.
Query Category Count
Factual Retrieval 60
Comparative Scientific Analysis 39
Scientific Interpretation 15
Metadata Lookup 14
Onboarding / Conceptual Explanation 13
text, and avoidance of unsupported claims. Participants also
provided qualitative written feedback describing strengths,
ambiguities, omissions, or inaccuracies.
Participants interacted with AquiLLM through the browser-
based interface used during normal workflows. Ratings and
qualitative comments were submitted through a built-in eval-
uation form, anonymized using participant identifiers, and
exported for analysis. AquiLLM had no access to the public
internet via external tool calls during this study. This reflects
the privacy-sensitive scientific environments in which the
system may be deployed, where external web access may
be undesirable or prohibited. Restricting internet access also
ensured that generated responses were grounded exclusively
in the curated test collection used for evaluation, rather than
external web content or SEO-optimized sources of varying
reliability.
D. Query Design
Evaluation questions were grouped into five query cate-
gories: factual retrieval, metadata lookup, onboarding and con-
ceptual explanation, scientific interpretation, and comparative
scientific analysis. These categories captured varying levels of
retrieval complexity, synthesis, and domain reasoning within
astronomy workflows.
Factual retrieval questions involved direct lookup of scien-
tific information from retrieved documents. Metadata lookup
questions focused on survey structure, observational channels,
release information, and dataset characteristics. Onboarding
and conceptual explanation questions evaluated whether the
system could explain terminology or concepts useful for
newcomers. Scientific interpretation questions required under-
standing scientific outputs and model behavior, while compar-
ative scientific analysis questions involved comparing models,
datasets, or performance metrics across contexts.
The final evaluation dataset consisted of 141 annotated
responses distributed across the five query categories shown
in Table II.
E. Evaluation Criteria
In this study, faithfulness was defined as the extent to which
generated responses remained grounded in scientific context
without fabrication or unsupported inference. Participants eval-
uated AquiLLM responses against collaboratively established
reference answers for the selected evaluation questions.
Evaluation emphasized consistency with reference answers
and an absence of hallucinated claims. Responses were evalu-
ated using a five-point Likert-style scale ranging from low to

high faithfulness. The evaluation form presented this scale as
a plain 1–5 rating control without predefined verbal anchors
for each point (e.g., no fixed text such as “fully faithful” or
“not faithful” attached to a specific value); interpretation of
the scale instead relied on participants’ shared, group-level un-
derstanding of faithfulness developed through approximately
one year of prior collaborative use of AquiLLM. Each rating
was accompanied by required qualitative written feedback
describing strengths, ambiguities, omissions, or inaccuracies,
which we used to recover the specific reasoning behind low
or high scores (Section V-C). We treat the absence of formal
verbal anchors as a limitation of the current evaluation instru-
ment (Section VII). Participants generated and rated their own
responses independently through their own AquiLLM chat
session (Section IV-C): the evaluation interface did not display
other participants’ ratings or generated responses, so raters
could not see or be influenced by one another’s scores, and no
two participants ever rated the same generated response text.
Individual ratings were not reconciled into a single consensus
value, unlike the reference answers, which were discussed as a
group; the dataset retains each participant’s rating separately,
which is what enables the descriptive consistency analysis
in Section V-B, though that analysis cannot separate rater
subjectivity from genuine response-to-response variation for
the reason just described. Because evaluators were members
of the same research group and collaboratively established
reference answers, the study emphasized qualitative agreement
and contextual scientific interpretation rather than formal inter-
rater reliability metrics; Section V-B nonetheless reports a
descriptive analysis of rating consistency across participants
for each question, computed from the existing multi-rater data.
V. RESULTS
Across 141 cleaned responses (140 non-missing ratings),
AquiLLM demonstrated generally strong faithfulness perfor-
mance, with a mean rating of 3.80 (median = 4.0, SD =
1.24). Most responses were evaluated positively, with 68.6%
receiving ratings of 4–5, while 17.9% received ratings of 1–2.
The system performed most reliably on retrieval-oriented
questions grounded in single scientific documents, while per-
formance became less stable for synthesis and comparative
analysis tasks requiring integration across multiple sources.
Qualitative comments described AquiLLM as useful and scien-
tifically helpful, though participants also identified omissions
and unsupported claims in more complex query settings.
A. Performance Across Query Types
As shown in Table III, query-type variation was the
strongest signal in the evaluation. The most robust contrast,
drawn from the two categories built from multiple underlying
questions, is between Factual Retrieval (4 questions,n=60,
mean 4.17) and Comparative Scientific Analysis (3 questions,
n=39, mean 2.97), indicating that cross-source synthesis and
comparison remain the primary failure surface. The remaining
three categories, Scientific Interpretation, Metadata Lookup,
and Onboarding/Conceptual Explanation, are each representedby a single underlying question in this study (Table V), so their
category-level means are single-question results rather than
generalizable category effects. In particular, the high mean
for Scientific Interpretation (4.73) reflects performance on one
specific question about MMAE emission-line reconstruction
and should not be read as evidence that reasoning-oriented
questions generally outperform retrieval-oriented ones; the ab-
stract’s summary claim that AquiLLM performs most reliably
on retrieval-oriented questions rests on the multi-question Fac-
tual Retrieval/Comparative Scientific Analysis contrast, not on
this single-question result. Onboarding and metadata questions
were generally helpful but showed more moderate reliability,
with errors more often related to precision and scope than
outright hallucination.
B. Rating Consistency Across Independent Attempts
Because each participant independently generated and rated
their own responses rather than rating a shared set of fixed
responses (Section IV-C), no two ratings in this dataset cor-
respond to the same generated response text. As a result, the
spread of ratings for a given question reflects a mixture of
two sources that cannot be separated with this data: genuine
variation in how faithfully AquiLLM answered that question
across repeated, independent attempts, and differences in how
strictly individual participants rated. This second source is
itself an expected one: although human evaluation is generally
considered the gold standard for assessing faithfulness, it is
also known to be prone to subjective variation between eval-
uators [11]. We therefore report the following as a descriptive
measure of overall rating consistency, computed directly from
the released evaluation dataset, rather than a formal inter-
rater reliability statistic, which would require multiple raters
to score the same fixed response (Section IV-E).
Across the ten evaluation questions (12–15 completed rat-
ings per question, from up to five participants each), the mean
within-question standard deviation of ratings was 0.90 (range
0.35–1.46 across questions), and 77.8% of individual ratings
fell within one point of that question’s mean rating.
Consistency was highest for the three highest-scoring ques-
tions (Q6–Q8 in Table V), which per their own classification
span Scientific Interpretation, Factual Retrieval, and a mixed
retrieval/reasoning task rather than being uniformly retrieval-
oriented, where 100% of ratings fell within one point of the
question mean, and lowest for two Comparative Scientific
Analysis questions (Q4 and Q9), where only 42.9% and 46.2%
of ratings, respectively, fell within one point of the mean.
Because response variation and rater variation are completely
confounded in this dataset, as described above, we cannot
attribute this pattern to either source specifically: it may reflect
AquiLLM’s own output being less stable for comparative-
analysis questions from one independent attempt to the next,
participants disagreeing more about how to score genuinely
ambiguous synthesis attempts, or some mixture of both, and
this analysis provides no way to distinguish between these
possibilities. We note only that lower consistency co-occurs
with the same category, Comparative Scientific Analysis,

TABLE III
PERFORMANCE ACROSS THE FIVE QUERY TYPES. FACTUALRETRIEVAL ANDCOMPARATIVESCIENTIFICANALYSIS AGGREGATE MULTIPLE QUESTIONS
EACH(4AND3,RESPECTIVELY);THE REMAINING THREE CATEGORIES ARE EACH A SINGLE QUESTION(TABLEV)REPEATED ACROSS PARTICIPANTS
AND SUBMISSIONS,AND THEIR MEANS SHOULD BE READ ACCORDINGLY.
Query Type n Mean Observed Strengths Common Failure Modes
Factual Retrieval (4 questions) 60 4.17 Accurate grounded responses;
strong direct evidence retrieval;
generally high consistencyOccasional omission; missing
qualifiers/details
Comparative Scientific Analysis (3 questions) 39 2.97 Attempts cross-result comparison
and tradeoff framingInconsistent grounding across
sources; selective omission;
incomplete synthesis
Scientific Interpretation (1 question) 15 4.73 Helpful synthesis and
interpretation when context is
clear; strong explanatory
coherenceUnsupported inference in edge
cases; occasional overreach
beyond cited evidence
Metadata Lookup (1 question) 14 3.64 Useful schema/channel/parameter
lookup support; good orientation
to dataset structurePartial precision; assumptions not
always explicit
Onboarding / Conceptual Explanation (1 question) 13 3.62 Useful onboarding support for
foundational concepts; accessible
explanationsOvergeneralization; simplified but
under-specified answers
where mean faithfulness was also lowest (Section V), without
claiming a specific causal explanation for that co-occurrence.
We report this analysis as an exploratory substitute for a formal
reliability metric, not a replacement for one (Section VII).
C. Failure Modes
To better understand why some responses received low
faithfulness ratings, we categorized evaluator-identified errors
into six recurring failure modes.
1) Hallucinated Scientific Details:The model occasionally
introduced unsupported numeric or technical claims while
sounding scientifically plausible. In one comparison between
two redshift prediction models, the system reported a 2.4x bias
gap when the source-supported value was 1.2x.
2) Unsupported Synthesis:When evidence was fragmented
across documents, the model sometimes inferred conclu-
sions without sufficient grounding. In one case involving
c_model_magcomputation, evaluators reported that the sys-
tem initially stated it could not locate the method, but then
generated a profile-fitting explanation not supported by the
retrieved materials.
3) Retrieval Mismatch:Some failures reflected incorrect
retrieval alignment or incomplete anchoring to the requested
source context. Evaluators flagged responses that referenced
unspecified tables or documents, or stated that information
could not be found even though it was present in the supplied
collection.
4) Subtle Factual Drift:Not all failures were overt hallu-
cinations; several responses were nearly correct but contained
shifted values, missing qualifiers, or scope drift. One evaluator
noted that a response correctly identified the scientific trend
under discussion but reported the wrong scatter value and
omitted important conditional details.
5) Overconfident Interpretation:The model occasionally
presented uncertain or weakly supported claims with exces-sive confidence. In conceptual explanation tasks, such as
discussions of z-band naming conventions, evaluators observed
assertive interpretations that were not strongly supported by
the retrieved scientific context.
6) Failure Under Ambiguity:For underspecified prompts,
the system did not always request clarification before an-
swering. Evaluators noted that some survey-channel questions
were answered for a specific astronomy survey without first
confirming which survey the user intended.
Overall, the dominant risk observed in the evaluation was
not random error, but credible-sounding overreach, particularly
in synthesis and comparative analysis tasks requiring inte-
gration across multiple sources. This pattern aligns with the
lower average faithfulness ratings observed for Comparative
Scientific Analysis questions.
D. Expert Perspectives
Participants found AquiLLM most useful for direct
retrieval-oriented tasks involving numeric values, survey meta-
data, or information grounded within a single document.
Several described the system as effective for quickly locating
scientific details and navigating technical materials.
Trust decreased for tasks requiring reasoning across mul-
tiple papers or fragmented scientific context. Participants ob-
served greater variability and lower reliability for synthesis-
oriented questions, often producing plausible but only partially
grounded responses. One participant summarized this behavior
as, “when asked to reason across papers it fell apart.”
Participants also emphasized the importance of query speci-
ficity. Broad or underspecified prompts often caused retrieval
of excessive or weakly related information spanning multiple
astronomy surveys rather than the intended dataset.
Another recurring issue was variability across repeated
queries. Repeated submissions of the same question occa-
sionally produced inconsistent retrieval behavior or conflicting

answers. These inconsistencies raised questions about retrieval
persistence, memory effects, and tool-use stability.
E. Sources of Failure
Observed failures reflected interactions between retrieval
quality, orchestration behavior, model reasoning, and imple-
mentation limitations. Some errors resulted from incomplete
retrieval or insufficient ranking of relevant scientific context,
particularly for synthesis-oriented questions spanning multiple
documents.
Other inconsistencies appeared related to tool-calling and
orchestration behavior. Participants observed variability in
whether retrieval tools were invoked across repeated gener-
ations of the same query, including cases where responses
were generated without visible retrieval after earlier successful
retrieval attempts.
Several issues also appeared to reflect implementation-
level limitations rather than failures of the underlying lan-
guage model itself, including inconsistent retrieval visibility,
repeated-query variability, and poor handling of ambiguous
prompts.
Overall, the evaluation suggests that improving faithfulness
in systems such as AquiLLM will require advances in retrieval
quality, tool-use consistency, ambiguity handling, and system-
level robustness rather than foundation models alone.
VI. DISCUSSION
A. What Faithfulness Means in Scientific Contexts
The results of this study suggest that faithfulness in of-
fline scientific RAG-LLM systems differs substantially from
general-purpose notions of chatbot helpfulness or conversa-
tional fluency. In scientific workflows, responses must remain
grounded in source material and avoid unsupported inference
that could affect interpretation or reproducibility. As a result,
scientific trust depends not only on whether a response is
correct, but also on whether researchers can understand how
and why the response was generated from the underlying
evidence. This becomes particularly important in scientific
domains where measurements and analysis methods differ
across research groups investigating similar phenomena. One
long-term goal of AquiLLM is to support this kind of contex-
tual scientific reasoning by distinguishing between competing
methodological interpretations and explicitly communicating
these differences within generated responses.
Participants in the study frequently evaluated responses
in terms of traceability and provenance rather than stylistic
quality alone. Responses that accurately reflected retrieved
documents and consistent with known references were gen-
erally viewed as trustworthy, even when incomplete. In con-
trast, plausible-sounding synthesis without clear grounding
often reduced confidence, particularly in comparative analysis
tasks involving multiple papers or datasets. Evaluators also
emphasized the importance of uncertainty awareness, noting
that overly confident explanations could become misleading
when evidence was ambiguous or only partially retrieved.These findings suggest that scientific RAG-LLM systems re-
quire stronger support for transparent retrieval behavior, source
attribution, ambiguity handling, and calibrated uncertainty,
than is typically emphasized in consumer conversational AI
systems. More broadly, the evaluation highlights that scientific
faithfulness is not solely a property of the language model
itself, but emerges from the interactions between retrieval
systems, orchestration pipelines, domain context, and user
expectations surrounding evidence and reproducibility.
B. Implications for Scientific Cyberinfrastructure
Systems such as AquiLLM can be viewed as part of a
longer evolution in scientific cyberinfrastructure, particularly
within astronomy. Earlier platforms such as the Sloan Digital
Sky Survey (SDSS), CasJobs, and web-based archive systems
expanded scientific access through searchable databases and
SQL-driven interfaces. RAG-LLM systems represent a poten-
tial new interface layer by enabling conversational interaction
with publications, datasets, technical documentation, and col-
laborative research knowledge.
However, the results of this study suggest that important
reliability constraints remain unresolved. While AquiLLM per-
formed relatively well for retrieval-oriented tasks, synthesis-
heavy and comparative reasoning questions exposed limita-
tions in grounding, retrieval consistency, and tool orchestra-
tion. These findings suggest that future scientific NLP-based
infrastructure will require advances not only in foundation
models, but also in retrieval pipelines, provenance tracking,
and transparent reasoning workflows.
C. Scalability and Generalization Potential
This study evaluated AquiLLM against a modest, curated
collection of 31 documents; we have not yet systematically
measured how retrieval quality, latency, or faithfulness change
as collections grow toward the scale of active research-group
archives (hundreds to thousands of documents) or shared
institutional repositories (potentially hundreds of thousands).
AquiLLM’s retrieval layer combines dense vector search with
lexical matching over indexed chunks (Section III-D), and its
modular separation of ingestion, embedding, reranking, and
generation (Fig. 1) is intended to let each component be scaled
or replaced independently as corpus size grows. These are ar-
chitectural design goals rather than measured outcomes, how-
ever, and larger or noisier collections may increase retrieval
ambiguity in ways that compound the comparative-analysis
failure modes already observed at small scale (Section V-C);
we did not evaluate this directly.
Generalization beyond this single astronomy research group
raises a related but distinct question. AquiLLM’s modular
design lets each research group configure its own collection,
retrieval settings, and (within hardware constraints) underlying
models, and we expect the query-category evaluation method-
ology used in this study (Section IV-D) to transfer directly
to other domains. Faithfulness thresholds, common failure
modes, and appropriate evaluation panels are nonetheless
likely to be domain-specific: a collection of numerically dense

survey papers, as used here, differs in structure and ambiguity
from collections dominated by qualitative methods, code, or
heterogeneous file types. We therefore treat this study as estab-
lishing a reusable evaluation methodology rather than a result
that generalizes numerically to other domains without re-
evaluation; Section VII-A outlines planned evaluation across
additional research groups and domains to test this directly.
D. Open-Weight LLMs for Research Communities
The study additionally highlights the potential importance
of open-weight RAG-LLM systems for research communi-
ties. Locally deployable systems provide greater institutional
control over data, infrastructure, reproducibility, and long-
term preservation compared to externally managed commercial
AI platforms. Although frontier commercial systems may
outperform smaller open-weight models, research groups may
still prefer open systems that can be modified, and preserved
within research computing environments.
E. Lessons for RSE Teams
For research software engineers evaluating or deploying
similar offline RAG-LLM systems, three practical lessons
emerged from this study. First, retrieval-oriented and synthesis-
oriented queries behave differently enough that they warrant
separate evaluation and, likely, separate design attention; a
single aggregate faithfulness score can mask large differences
between query types (Table III). Second, requiring a short free-
text justification alongside each numeric rating, even without a
formalized rubric or predefined scale anchors, recovered most
of the diagnostic value in this study (Section V-C) and is
a low-cost addition to any in-house evaluation form. Third,
several of the observed failures traced back to retrieval and
orchestration behavior rather than to the underlying language
model (Section V-E), suggesting that RSE teams building
similar systems should budget evaluation and debugging effort
for the retrieval/orchestration layer specifically, not only for
model selection.
F . Human Expertise Remains Essential
The evaluation demonstrates that domain experts remain es-
sential for assessing scientific faithfulness in non-deterministic
systems. Participants identified subtle inaccuracies that would
likely remain invisible in benchmark evaluations. These find-
ings suggest that benchmark performance alone is insufficient
and expert-centered evaluation should remain central to future
scientific RAG-LLM research.
VII. LIMITATIONS
This study has several limitations. First, the evaluation
involved five astronomers from a single research group, which
may limit broader generalization. Second, the study focused on
a single scientific domain and a specific astronomy deployment
of AquiLLM; other disciplines may exhibit different retrieval
practices, terminology, and expectations surrounding scientific
faithfulness (Section VI-C). Third, the evaluation examined a
single open-weight model configuration within the AquiLLMarchitecture. Because the open-weight ecosystem is evolving
rapidly, newer models and tools may produce substantially
different behaviors and failure characteristics. Although pre-
liminary internal testing with commercially hosted models
connected to AquiLLM suggested similar grounding and syn-
thesis limitations, these configurations were not systematically
evaluated in the present study.
Fourth, this study does not include a controlled comparison
against an alternative RAG system or a component-level abla-
tion isolating the contribution of reranking, retrieval fan-out,
or orchestration. A direct comparison against a commercially
hosted RAG system is complicated by the same privacy moti-
vation that underlies AquiLLM’s design: the evaluation corpus
includes the research group’s unpublished manuscripts and in-
ternal technical references (Section IV-B), and transmitting this
material to an external commercial API would undermine the
offline, privacy-preserving deployment model that this paper
argues for. Restricting such a comparison to only the corpus’s
publicly available documents would avoid this problem but
would no longer test the same corpus used throughout this
study, weakening rather than strengthening the comparison.
We therefore view two next steps as more coherent than a
commercial-system comparison: (1) a component-level abla-
tion conducted entirely within AquiLLM’s own offline de-
ployment, varying reranking and cross-document orchestration
behavior while holding the retrieval and generation models
fixed, targeted at the Comparative Scientific Analysis category
where faithfulness failures and rating inconsistency both con-
centrated (Sections V and V-B); and (2) a comparison against
another locally-deployable open-source RAG framework on
the same corpus, which avoids the data-exposure problem of
a commercial API. We plan to pursue both as follow-ups to
this study.
Fifth, the five-point faithfulness scale used in the built-
in evaluation form did not include predefined verbal anchors
(Section IV-E). Moreover, because each participant generated
and rated their own responses rather than rating a shared set
of fixed responses (Section IV-C), formal inter-rater reliability
statistics are not computable from this dataset, not merely
uncomputed: no two ratings correspond to the same response
text, so any measured spread in ratings necessarily mixes rater
subjectivity with genuine generation-to-generation variability
that cannot be separated post hoc. The descriptive analysis in
Section V-B is intended as an initial, exploratory substitute,
not a replacement for a validated rubric and a study design
that would support a true reliability measure, such as multiple
raters independently scoring the same fixed response.
Data and Code Availability
AquiLLM’s source code and deployment configuration are
publicly available athttps://github.com/AquiLLM
[8]. We do not publicly release the evaluation dataset (ques-
tions, reference answers, ratings, and qualitative comments)
at this time. This question set and its reference answers are
intended to remain a held-out evaluation benchmark across
the follow-up studies described in Section VII-A; publishing

them openly now risks their incorporation into future model
training corpora, which would contaminate the benchmark
and undermine its usefulness for evaluating later open-weight
models on the same questions. The dataset is available directly
from the authors to individual researchers for reproducibility
purposes, on the condition that it not be published or used,
directly or indirectly, to train a model; we plan to release
it publicly once the planned follow-up evaluation rounds are
complete.
A. Future Work
Our nearest-term priority is evaluation, not new system fea-
tures. AquiLLM has since undergone a series of architectural
updates, including local embedding and reranking, multimodal
capabilities, and expanded memory support, described in a
companion paper [28]. We plan to rerun this faithfulness study,
using the same query categories and evaluation methodology
(Section IV-D), against the updated system to test directly
whether these changes reduce the synthesis and comparative-
analysis failure modes identified in Section V-C. This re-
run will specifically incorporate knowledge-graph-enhanced
retrieval, extending AquiLLM’s existing hybrid vector/lexical
search (Section III-D) with a graph of relationships between
scientific concepts, entities, and their supporting evidence,
since Comparative Scientific Analysis, the primary failure
surface identified in Section V, centers on cross-source syn-
thesis and reasoning that a graph-structured representation is
intended to support more directly than chunk-level retrieval
alone. This rerun will be conducted alongside a broader com-
ponent and configuration comparison, extending the ablation
and open-source RAG comparison described in Section VII
to measure how individual components, including retrieval,
reranking, knowledge graphs, and memory, each contribute
to performance, and we plan to extend the evaluation to
additional research groups and domains, as well as to the tacit
and informal knowledge artifacts (e.g., meeting notes, internal
documentation) that [28] identifies as not yet evaluated for
faithfulness.
Beyond this evaluation program, planned system devel-
opment includes refining domain-specific knowledge-graph
schemas and interactive graph exploration tools; extending
provenance and citation tracing to graph relationships as
well as individual passages; improving query interpretation,
multi-part question handling, and ambiguity handling; more
reliable tool orchestration and more robust multimodal docu-
ment ingestion; continued latency and scalability work across
retrieval, reranking, graph extraction, and local inference;
domain-specific model fine-tuning; and methods for identify-
ing when answers may be unreliable, ambiguous, or insuf-
ficiently supported by retrieved evidence. Ongoing work on
prompt compression, additional multimodal embeddings, and
longer-horizon memory planning is described in [28].
VIII. CONCLUSION
Natural language interfaces are likely to become an im-
portant component of scientific cyberinfrastructure. Systemssuch as AquiLLM suggest that open-weight, offline retrieval-
augmented language models can support scientific workflows
through conversational interaction. In particular, retrieval-
oriented tasks demonstrated promising levels of faithfulness
and usability within the astronomy evaluation environment.
Additionally, the study demonstrates that scientific de-
ployment of RAG-LLM systems requires rigorous evaluation
within real research settings. Although AquiLLM performed
well on retrieval-oriented tasks, the evaluation showed that
maintaining faithfulness becomes substantially more difficult
when systems must synthesize information across documents
or operate under ambiguity. These findings highlight the
importance of expert-centered evaluation for understanding the
limitations of scientific RAG-LLM systems beyond standard
benchmark performance.
ACKNOWLEDGMENT
This work used resources provided through ACCESS-CI
allocation CIS260251, “AquiLLM: A RAG-LLM tool for
research groups,” and NAIRR Pilot allocations NAIRR240359
and NAIRR240464. Computational resources were provided
in part by Jetstream2. This material is also based upon work
supported by the National Science Foundation under Grant
No. 2448094 and by the Alfred P. Sloan Foundation.
REFERENCES
[1] I. Androutsopoulos, G. D. Ritchie, and P. Thanisch, “Natural Language
Interfaces to Databases - An Introduction,” Mar. 1995, arXiv:cmp-
lg/9503016. [Online]. Available: http://arxiv.org/abs/cmp-lg/9503016
[2] A. S. Szalay, P. Z. Kunszt, A. Thakar, J. Gray, D. Slutz, and R. J.
Brunner, “Designing and mining multi-terabyte astronomy archives: the
Sloan Digital Sky Survey.” ACM Press, 2000, pp. 451–462. [Online].
Available: http://portal.acm.org/citation.cfm?doid=342009.335439
[3] O. Zaoui Seghroucheni, M. LAZAAR, and M. Al Achhab, “Using AI
and NLP for Tacit Knowledge Conversion in Knowledge Management
Systems: A Comparative Analysis,”Technologies, vol. 13, p. 87, Feb.
2025.
[4] H. Collins,Tacit and explicit knowledge. Chicago, Ill: University of
Chicago Press, 2013.
[5] A. J. G. Hey and Microsoft Research, Eds.,The fourth paradigm: data
intensive scientific discovery, 2nd ed. Redmond, Wash: Microsoft
Research, 2009.
[6] M. D. Skarlinski, S. Cox, J. M. Laurent, J. D. Braza, M. Hinks,
M. J. Hammerling, M. Ponnapati, S. G. Rodriques, and A. D.
White, “Language agents achieve superhuman synthesis of scientific
knowledge,” Sep. 2024, arXiv:2409.13740 [cs]. [Online]. Available:
http://arxiv.org/abs/2409.13740
[7] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and
D. Kiela, “Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks,” Apr. 2021, arXiv:2005.11401 [cs]. [Online]. Available:
http://arxiv.org/abs/2005.11401
[8] B. Boscoe, T. Do, and et al., “AquiLLM,” 2026. [Online]. Available:
https://github.com/AquiLLM
[9] C. Campbell, B. Boscoe, and T. Do, “AquiLLM: a RAG Tool
for Capturing Tacit Knowledge in Research Groups.” Philadelphia,
PA: arXiv, Oct. 2025, arXiv:2508.05648 [cs]. [Online]. Available:
http://arxiv.org/abs/2508.05648
[10] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga,
Y . Zhang, D. Narayanan, Y . Wu, A. Kumar, B. Newman, B. Yuan,
B. Yan, C. Zhang, C. Cosgrove, C. D. Manning, C. R ´e, D. Acosta-
Navas, D. A. Hudson, E. Zelikman, E. Durmus, F. Ladhak, F. Rong,
H. Ren, H. Yao, J. Wang, K. Santhanam, L. Orr, L. Zheng,
M. Yuksekgonul, M. Suzgun, N. Kim, N. Guha, N. Chatterji,
O. Khattab, P. Henderson, Q. Huang, R. Chi, S. M. Xie, S. Santurkar,
S. Ganguli, T. Hashimoto, T. Icard, T. Zhang, V . Chaudhary, W. Wang,

X. Li, Y . Mai, Y . Zhang, and Y . Koreeda, “Holistic Evaluation
of Language Models,” Oct. 2023, arXiv:2211.09110 [cs]. [Online].
Available: http://arxiv.org/abs/2211.09110
[11] B. Malin, T. Kalganova, and N. Boulgouris, “A review of faithfulness
metrics for hallucination assessment in Large Language Models,”IEEE
Journal of Selected Topics in Signal Processing, vol. 19, no. 7,
pp. 1362–1375, Oct. 2025, arXiv:2501.00269 [cs]. [Online]. Available:
http://arxiv.org/abs/2501.00269
[12] J. Gray and A. S. Szalay, “The World Wide Telescope: An Archetype
for Online Science,” Mar. 2004, arXiv:cs/0403018. [Online]. Available:
http://arxiv.org/abs/cs/0403018
[13] A. Szalay, J. Gray, A. Thakar, P. Z. Kunszt, T. Malik, J. Raddick,
C. Stoughton, and J. vandenBerg, “The SDSS SkyServer, Public Access
to the Sloan Digital Sky Server Data,” Nov. 2001, arXiv:cs/0111015.
[Online]. Available: http://arxiv.org/abs/cs/0111015
[14] Z. Luo, Z. Yang, Z. Xu, W. Yang, and X. Du,
“LLM4SR: A Survey on Large Language Models for Scientific
Research,” Jan. 2025, arXiv:2501.04306 [cs]. [Online]. Available:
http://arxiv.org/abs/2501.04306
[15] P. Ersoy and M. Ers ¸ahın, “A Comparative Evaluation of RAG Archi-
tectures for Cross-Domain LLM Applications: Design, Implementation,
and Assessment,”IEEE Access, vol. 13, pp. 194 185–194 196, 2025.
[Online]. Available: https://ieeexplore.ieee.org/document/11245479/
[16] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang,
“Retrieval Augmented Language Model Pre-Training,” inProceedings
of the 37th International Conference on Machine Learning.
PMLR, Nov. 2020, pp. 3929–3938. [Online]. Available:
https://proceedings.mlr.press/v119/guu20a.html
[17] T. D. Nguyen, Y .-S. Ting, I. Ciuc ˘a, C. O’Neill, Z.-C. Sun, M. Jabło ´nska,
S. Kruk, E. Perkowski, J. Miller, J. Li, J. Peek, K. Iyer, T. R ´o˙za´nski,
P. Khetarpal, S. Zaman, D. Brodrick, S. J. R. M ´endez, T. Bui,
A. Goodman, A. Accomazzi, J. Naiman, J. Cranney, K. Schawinski, and
UniverseTBD, “AstroLLaMA: Towards Specialized Foundation Models
in Astronomy,” Sep. 2023, arXiv:2309.06126 [astro-ph]. [Online].
Available: http://arxiv.org/abs/2309.06126
[18] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang,
A. Madotto, and P. Fung, “Survey of Hallucination in Natural Language
Generation,”ACM Comput. Surv., vol. 55, no. 12, pp. 248:1–248:38,
Mar. 2023. [Online]. Available: https://dl.acm.org/doi/10.1145/3571730
[19] M. Polanyi,The Tacit Dimension, ser. An-
chor books. Anchor Books, 1967. [Online]. Available:
https://books.google.com/books?id=jwLXAAAAMAAJ
[20] F. Yang, W. Chen, and J. D. Evans, “Large language
models in materials science and the need for open-source
approaches,” Nov. 2025, arXiv:2511.10673 [cs]. [Online]. Available:
http://arxiv.org/abs/2511.10673
[21] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, “Retrieval
Augmentation Reduces Hallucination in Conversation,” inFindings of the
Association for Computational Linguistics: EMNLP 2021, M.-F. Moens,
X. Huang, L. Specia, and S. W.-t. Yih, Eds. Punta Cana, Dominican
Republic: Association for Computational Linguistics, Nov. 2021, pp.
3784–3803. [Online]. Available: https://aclanthology.org/2021.findings-
emnlp.320/
[22] S. A. Akbar, M. M. Hossain, T. Wood, S.-C. Chin, E. M.
Salinas, V . Alvarez, and E. Cornejo, “HalluMeasure: Fine-grained
Hallucination Measurement Using Chain-of-Thought Reasoning,” in
Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing. Miami, Florida, USA: Association for
Computational Linguistics, 2024, pp. 15 020–15 037. [Online]. Available:
https://aclanthology.org/2024.emnlp-main.837
[23] E. Khan, L. Rodriguez, and M. Queudot, “Reason and
Verify: A Framework for Faithful Retrieval-Augmented Gener-
ation,” Mar. 2026, arXiv:2603.10143 [cs]. [Online]. Available:
http://arxiv.org/abs/2603.10143
[24] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, “On Faithfulness
and Factuality in Abstractive Summarization,” inProceedings of the
58th Annual Meeting of the Association for Computational Linguistics,
D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online:
Association for Computational Linguistics, Jul. 2020, pp. 1906–1919.
[Online]. Available: https://aclanthology.org/2020.acl-main.173/
[25] M. S. Tamber, F. S. Bao, C. Xu, G. Luo, S. Kazi, M. Bae, M. Li,
O. Mendelevitch, R. Qu, and J. Lin, “Benchmarking LLM Faithfulness
in RAG with Evolving Leaderboards,” Nov. 2025, arXiv:2505.04847
[cs]. [Online]. Available: http://arxiv.org/abs/2505.04847[26] DavidAU, “Qwen3.5-27B-Claude-4.6-OS-Auto-Variable-Thinking,”
2026, community model card; independent fine-tune, not an
official Qwen (Alibaba) release. Accessed September 2026.
[Online]. Available: https://huggingface.co/DavidAU/Qwen3.5-27B-
Claude-4.6-OS-Auto-Variable-Thinking
[27] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu,
C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang,
F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang,
J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang,
L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu,
R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren,
X. Wang, X. Zhang, X. Ren, Y . Fan, Y . Su, Y . Zhang, Y . Zhang,
Y . Wan, Y . Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu,
“Qwen3 Technical Report,” May 2025, arXiv:2505.09388 [cs]. [Online].
Available: http://arxiv.org/abs/2505.09388
[28] J. Stark, S. Saikrishnan, V . Seenivasan, B. Boscoe, A. Lizarraga,
and T. Do, “AquiLLM: An Architecture for Supporting
Tacit Knowledge Capture in Research Groups,” Aug. 2026,
arXiv:2608.08883 [cs]. Presented at NGEN-AI 2026. [Online]. Available:
http://arxiv.org/abs/2608.08883

APPENDIXA
DEPLOYMENTCOST ANDHARDWAREREQUIREMENTS
The astronomy deployment evaluated in this study ran on a single NVIDIA H100 GPU (80 GB), allocated through Jetstream2
via an ACCESS-CI allocation using theg5.xlinstance flavor (20 vCPUs, 240 GB system RAM, 1×H100 GPU with 80 GB
GPU memory), which Jetstream2 rates at 128 Service Units (SUs) per hour against the allocation. All inference components
used in the study, the primary chat model, embedding, reranking, OCR, and transcription, were served as separate vLLM
processes on this single GPU using memory-utilization partitioning (Section III-D) to fit within its available memory, rather
than requiring one GPU per component.
The SU-based rate above reflects usage against a grant-funded academic allocation rather than a direct cash payment by the
research group. For research groups without access to an academic HPC allocation, Table IV instead gives an approximate
commercial-cloud equivalent cost for renting H100 GPU capacity, based on rates checked directly against provider pricing
pages in September 20261. Specialist GPU cloud providers (e.g., RunPod) listed on-demand H100 rates around $2–3.50 per
GPU-hour; hyperscaler providers (Azure, AWS) listed single-H100 or per-GPU-equivalent rates around $7–11 per GPU-hour.
These configurations are not strictly comparable to each other or to the deployment evaluated in this study: Azure’s single-GPU
instance bundles roughly twice the vCPUs and system RAM of theg5.xlflavor used here, and the AWS figure requires
committing to an entire 8-GPU node regardless of whether a research group needs that capacity. All figures also cover GPU
compute only and exclude storage, networking, and engineering or maintenance time.
For a small research group (e.g., approximately seven users, comparable in size to the participant panel in this study plus its
broader research group), usage is likely bursty rather than continuous: interactive queries occur intermittently, while document
ingestion and indexing run asynchronously in the background. Table IV therefore reports two usage scenarios: continuous
(24/7) operation, which keeps the deployment immediately available and avoids model reload latency, and business-hours-only
operation (approximately 8 hours per weekday), which reduces cost at the expense of availability outside that window.
TABLE IV
APPROXIMATE COMMERCIAL-CLOUD COST TO RUN A SINGLEH100 GPUDEPLOYMENT,USING ON-DEMAND RATES CHECKED DIRECTLY AGAINST
PROVIDER PRICING PAGES INSEPTEMBER2026 (SEE TEXT FOR PROVIDER-SPECIFIC CAVEATS).
Specialist cloud Hyperscaler
(˜$2–3.50/hr) (˜$7–11/hr)
Continuous (24/7) $1,440–2,520/mo $5,040–7,920/mo
Business hours only*$350–620/mo $1,230–1,940/mo
*Approximately 8 hours/weekday, 22 weekdays/month (≈176 hours).
These figures are indicative rather than precise: actual GPU pricing fluctuates, and the appropriate configuration depends
on a research group’s size, query volume, and availability requirements. They nonetheless suggest that a single-GPU, multi-
component deployment of the kind evaluated in this study is within reach of a specialist-cloud budget of roughly several hundred
to a few thousand dollars per month for a small research group, likely less expensive than the hyperscaler configurations in
Table IV, though the two are not directly comparable given the differences in bundled resources and minimum commitment
described above. The deployment evaluated in this study itself was run without direct cloud-compute charges to the research
group, against a grant-funded ACCESS-CI allocation instead; that allocation still represents a real, finite compute resource
rather than a cost-free one.
1Specialist: RunPod pricing page (runpod.io/pricing), H100 PCIe/SXM on-demand, $1.99–3.49/hr as of September 2026. Hyperscaler: Microsoft
Azure’s single-GPUNC40ads_H100_v5instance, $6.98/hr on-demand (Azure pricing, September 2026); Amazon Web Services does not sell a standalone
single-H100 instance, itsp5.48xlargeinstance bundles 8 H100 GPUs at $55.04/hr on-demand ($6.88/hr per GPU-equivalent), so the AWS figure is a
computed per-GPU rate from an 8-GPU-minimum product rather than a directly purchasable single-GPU offering. Rates fluctuate and should be treated as
indicative rather than precise.

APPENDIXB
EVALUATIONQUESTIONS
TABLE V
METADATA AND CLASSIFICATION OF EVALUATION QUESTIONS.
Question Query Type Reasoning Synthesis Ambiguity Retrieval vs
Reasoning
What are the channels of the galaxy data in this survey? Metadata Lookup Low No Yes Retrieval
Why is there a z-channel in photometric images and why is the redshift also
denoted by z?Onboarding /
Conceptual
ExplanationMedium Yes No Mixed
How are the c model mag magnitudes calculated? Factual Retrieval Medium Yes No Mixed
What are the spectroscopic survey changes from the second public data release
to the third public data release of HSC?Comparative
Scientific AnalysisHigh Yes No Reasoning
How does the MMAE model’s redshift prediction performance compare to
CNN-based models in terms of scatter?Factual Retrieval Medium No No Retrieval
How do the shapes of emission lines reconstructed by the MMAE compare to
those of the true lines?Scientific
InterpretationHigh Yes No Reasoning
What is the maximum redshift of GalaxiesML-Spectra? Factual Retrieval Low No No Retrieval
At approximately what percentage of the GalaxiesML dataset do the LoRA
model’s performance metrics saturate?Factual Retrieval Medium Yes Yes Mixed
What is a drawback of using CNN-LoRA compared to CNN-Combo? Comparative
Scientific AnalysisHigh Yes No Reasoning
How does CNN-Base-Rev perform when compared to CNN-LoRA and
CNN-LoRA-Rev on the Combo dataset?Comparative
Scientific AnalysisHigh Yes No Reasoning

TESTCOLLECTIONPAPERS
[1] E. Collaboration, Y . Mellier, Abdurro’ufet al., “Euclid. I. Overview of
the Euclid mission,” May 2024, arXiv:2405.13491 [astro-ph]. [Online].
Available: http://arxiv.org/abs/2405.13491
[2] G. Desprez, S. Paltani, J. Couponet al., “Euclid preparation
- X. The Euclid photometric-redshift challenge,”Astronomy
& Astrophysics, vol. 644, p. A31, Dec. 2020. [On-
line]. Available: https://www.aanda.org/articles/aa/abs/2020/12/aa39403-
20/aa39403-20.html
[3] H. Aihara, N. Arimoto, R. Armstronget al., “The Hyper Suprime-
Cam SSP Survey: Overview and Survey Design,”Publications
of the Astronomical Society of Japan, vol. 70, no. SP1,
p. S4, Jan. 2018, arXiv:1704.05858 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/1704.05858
[4] M. Tanaka, J. Coupon, B.-C. Hsiehet al., “Photometric
redshifts for Hyper Suprime-Cam Subaru Strategic Program Data
Release 1,”Publications of the Astronomical Society of Japan,
vol. 70, no. SP1, p. S9, Jan. 2018. [Online]. Available:
https://academic.oup.com/pasj/article/doi/10.1093/pasj/psx077/4494086
[5] J. Singal, G. Silverman, E. Joneset al., “Machine Learning
Classification to Identify Catastrophic Outlier Photometric Redshift
Estimates,”The Astrophysical Journal, vol. 928, no. 1, p. 6, Mar. 2022.
[Online]. Available: https://dx.doi.org/10.3847/1538-4357/ac53b5
[6] A. A. Khostovan, J. S. Kartaltepe, M. Salvatoet al., “COSMOS
Spectroscopic Redshift Compilation (First Data Release): 165k
Redshifts Encompassing Two Decades of Spectroscopy,” Feb.
2025, aDS Bibcode: 2025arXiv250300120K. [Online]. Available:
https://ui.adsabs.harvard.edu/abs/2025arXiv250300120K
[7] R. Beck, C.-A. Lin, E. E. O. Ishidaet al., “On the realistic validation
of photometric redshifts,”Monthly Notices of the Royal Astronomical
Society, vol. 468, no. 4, pp. 4323–4339, Jul. 2017. [Online]. Available:
https://doi.org/10.1093/mnras/stx687
[8] E. Jones, T. Do, B. Boscoeet al., “Improving Photometric
Redshift Estimation for Cosmology with LSST Using Bayesian Neural
Networks,”The Astrophysical Journal, vol. 964, no. 2, p. 130, Apr. 2024.
[Online]. Available: https://iopscience.iop.org/article/10.3847/1538-
4357/ad2070
[9] B. Boscoe, T. Do, E. Joneset al., “Elements of effective machine
learning datasets in astronomy,” Nov. 2022, arXiv:2211.14401 [astro-
ph]. [Online]. Available: http://arxiv.org/abs/2211.14401
[10] E. Jones, T. Do, B. Boscoeet al., “Photometric Redshifts for
Cosmology: Improving Accuracy and Uncertainty Estimates Using
Bayesian Neural Networks,” Feb. 2022, arXiv:2202.07121 [astro-ph].
[Online]. Available: http://arxiv.org/abs/2202.07121
[11] DESI Collaboration, A. G. Adame, J. Aguilaret al., “The Early
Data Release of the Dark Energy Spectroscopic Instrument,”The
Astronomical Journal, vol. 168, no. 2, p. 58, Aug. 2024. [Online].
Available: https://iopscience.iop.org/article/10.3847/1538-3881/ad3217
[12] R. Wu, H. Wang, H.-T. Chenet al., “Deep Multimodal Learning
with Missing Modality: A Survey,” Oct. 2024, arXiv:2409.07825 [cs].
[Online]. Available: http://arxiv.org/abs/2409.07825
[13] J. A. Newman and D. Gruen, “Photometric Redshifts for Next-
Generation Surveys,”Annual Review of Astronomy and Astrophysics,
vol. 60, no. 1, pp. 363–414, Aug. 2022, arXiv:2206.13633 [astro-ph].
[Online]. Available: http://arxiv.org/abs/2206.13633
[14] H. Aihara, Y . AlSayyad, M. Andoet al., “Second Data Release of
the Hyper Suprime-Cam Subaru Strategic Program,”Publications
of the Astronomical Society of Japan, vol. 71, no. 6, p.
114, Dec. 2019, arXiv:1905.12221 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/1905.12221
[15] T. Do, B. Boscoe, E. Joneset al., “GalaxiesML: a dataset of galaxy
images, photometry, redshifts, and structural parameters for machine
learning,” Sep. 2024, arXiv:2410.00271 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/2410.00271
[16] A. Lizarraga, E. Hanchen Jiang, J. Nowacket al., “Understanding
Galaxy Morphology Evolution Through Cosmic Time via Redshift
Conditioned Diffusion Models.” Vacouver, B.C.: arXiv, Dec.
2024, aDS Bibcode: 2024arXiv241118440L. [Online]. Available:
https://ui.adsabs.harvard.edu/abs/2024arXiv241118440L
[17] R. Laureijs, J. Amiaux, S. Arduiniet al., “Euclid Definition Study
Report,” Oct. 2011, aDS Bibcode: 2011arXiv1110.3193L. [Online].
Available: https://ui.adsabs.harvard.edu/abs/2011arXiv1110.3193L[18] M. Lieu, “A Comprehensive Guide to Interpretable AI-Powered
Discoveries in Astronomy,”Universe, vol. 11, no. 6, p. 187, Jun. 2025.
[Online]. Available: https://www.mdpi.com/2218-1997/11/6/187
[19] V . Seenivasan, S. Saikrishnan, A. Lizarragaet al., “Combining datasets
with different ground truths using Low-Rank Adaptation to generalize
image-based CNN models for photometric redshift prediction,” San
Diego, California, Dec. 2025.
[20] M. Himes, S. Krishnamurthy, A. Lizarragaet al., “Multi-Modal Masked
Autoencoders for Learning Image-Spectrum Associations for Galaxy
Evolution and Cosmology,” San Diego, California, Dec. 2025. [Online].
Available: http://arxiv.org/abs/2510.22527
[21] M. Abdul Karim, J. Aguilar, S. Ahlenet al., “DESI DR2 results. II.
Measurements of baryon acoustic oscillations and cosmological
constraints,”Physical Review D, vol. 112, p. 083515, Oct.
2025, aDS Bibcode: 2025PhRvD.112h3515A. [Online]. Available:
https://ui.adsabs.harvard.edu/abs/2025PhRvD.112h3515A
[22] R. Beck, L. Dobos, T. Budav ´ariet al., “Photometric redshifts for the
SDSS Data Release 12,”Monthly Notices of the Royal Astronomical
Society, vol. 460, no. 2, pp. 1371–1381, Aug. 2016. [Online]. Available:
https://doi.org/10.1093/mnras/stw1009
[23] J. Soriano, S. Saikrishnan, V . Seenivasanet al., “Using different
sources of ground truths and transfer learning to improve the
generalization of photometric redshift estimation.” Vancouver, B.C.:
arXiv, Dec. 2024, arXiv:2411.18054 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/2411.18054
[24] E. Jones, T. Do, Y . Q. Liet al., “Redshift Prediction with Images
for Cosmology Using a Bayesian Convolutional Neural Network with
Conformal Predictions,”The Astrophysical Journal, vol. 974, no. 2,
p. 159, Oct. 2024. [Online]. Available: https://doi.org/10.3847/1538-
4357/ad6d5a
[25] D. Collaboration, M. Abdul-Karim, A. G. Adameet al.,
“Data Release 1 of the Dark Energy Spectroscopic Instrument,”
Mar. 2025, arXiv:2503.14745 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/2503.14745
[26] Y . Q. Li, T. Do, E. Joneset al., “Using Galaxy Evolution
as Source of Physics-Based Ground Truth for Generative
Models,” Jul. 2024, arXiv:2407.07229 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/2407.07229
[27] J. Soriano, T. Do, S. Saikrishnanet al., “Improving Generalization and
Uncertainty Quantification of Photometric Redshift Models,” Jan. 2026.
[Online]. Available: https://arxiv.org/abs/2601.17222v1
[28] M. Himes, “Multi-Modal Masked Autoencoders for Image and Spectra
Reconstruction (WIP).”
[29] H. Aihara, Y . AlSayyad, M. Andoet al., “Third Data Release
of the Hyper Suprime-Cam Subaru Strategic Program,”Publications
of the Astronomical Society of Japan, vol. 74, no. 2, pp. 247–
272, Apr. 2022, arXiv:2108.13045 [astro-ph]. [Online]. Available:
http://arxiv.org/abs/2108.13045
[30] C. Campbell, B. Boscoe, and T. Do, “AquiLLM: a
RAG Tool for Capturing Tacit Knowledge in Research
Groups,” Jul. 2025, arXiv:2508.05648 [cs]. [Online]. Available:
http://arxiv.org/abs/2508.05648
[31] ˇZ. Ivezi ´c, S. M. Kahn, J. A. Tysonet al., “LSST: from Science Drivers
to Reference Design and Anticipated Data Products,”The Astrophysical
Journal, vol. 873, no. 2, p. 111, Mar. 2019, arXiv:0805.2366 [astro-ph].
[Online]. Available: http://arxiv.org/abs/0805.2366
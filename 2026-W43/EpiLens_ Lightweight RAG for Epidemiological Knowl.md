# EpiLens: Lightweight RAG for Epidemiological Knowledge Acquisition

**Authors**: Vincenzo Perri, Ciro Cattuto, Daniela Paolotti

**Published**: 2026-10-19 16:58:53

**PDF URL**: [https://arxiv.org/pdf/2610.26730v1](https://arxiv.org/pdf/2610.26730v1)

## Abstract
To prepare for future outbreaks, researchers need to know what studies have been done and what data they use. Mapping a large literature by hand is slow, while large language models alone can misstate or conflate sources. We present EpiLens, a lightweight, open-source retrieval-augmented generation package that applies a fixed retrieval template and output schema to every paper in a corpus and links outputs to source passages. We build a corpus of 2,297 papers from the sources of a multi-pathogen parameter study and evaluate EpiLens's paper-classification workflow against 214 manually labelled papers. Without task-specific training data, the classifier matches or exceeds the tested supervised and zero-shot baselines across four tasks. The study's early extraction sheets cite published sources for 327 estimates. We link each estimate to a cited document or to corpus studies included by a review. Then, we use the classifier's labels to compare the papers associated with different parameters and pathogens. EpiLens labels more papers linked to reproduction numbers than fatality risks as model-based (54% versus 22%, compared with 62% versus 24% in the manual labels). The gap is consistent with the frequent use of transmission models to estimate reproduction numbers and persists within each of five pathogens compared. Beyond the contrast between parameters, the model-based share of papers linked to incubation-period estimates differs by pathogen: 55% for SARS-CoV-2 versus 6% for mpox.

## Full Text


<!-- PDF content starts -->

EpiLens: Lightweight RAG for Epidemiological
Knowledge Acquisition
Vincenzo Perri1*, Ciro Cattuto1and Daniela Paolotti1
1*ISI Foundation, Turin, Italy.
*Corresponding author(s). E-mail(s): vincenzo.perri@isi.it;
Contributing authors: ciro.cattuto@isi.it; daniela.paolotti@isi.it;
Abstract
To prepare for future outbreaks, researchers need to know what studies have
been done and what data they use. Mapping a large literature by hand is slow,
while large language models alone can misstate or conflate sources. We present
EpiLens, a lightweight, open-source retrieval-augmented generation package that
applies a fixed retrieval template and output schema to every paper in a corpus
and links outputs to source passages. We build a corpus of 2,297 papers from
the sources of a multi-pathogen parameter study and evaluate EpiLens’s paper-
classification workflow against 214 manually labelled papers. Without task-specific
training data, the classifier matches or exceeds the tested supervised and zero-
shot baselines across four tasks. The study’s early extraction sheets cite published
sources for 327 estimates. We link each estimate to a cited document or to corpus
studies included by a review. Then, we use the classifier’s labels to compare the
papers associated with different parameters and pathogens. EpiLens labels more
papers linked to reproduction numbers than fatality risks as model-based (54%
versus 22%, compared with 62% versus 24% in the manual labels). The gap is
consistent with the frequent use of transmission models to estimate reproduction
numbers and persists within each of five pathogens compared. Beyond the contrast
between parameters, the model-based share of papers linked to incubation-period
estimates differs by pathogen: 55% for SARS-CoV-2 versus 6% for mpox.
Keywords:Retrieval-augmented generation, Large language models, Digital
epidemiology, Knowledge acquisition, Literature analysis, Zero-shot classification,
Meta-research
1
arXiv:2610.26730v1  [cs.CL]  19 Oct 2026

1 Introduction
If a new pandemic started tomorrow, we would again face basic questions. Which
data streams are available and where? How have they been used in past outbreaks?
Which analyses depended on non-traditional data such as mobility traces, economic
signals, or social media activity, and under what conditions were those data shared?
What kinds of papers lie behind the parameter estimates we would reach for, such as
reproduction numbers or fatality risks, and how often are they model-based or present
simulated data? Recent work in digital epidemiology and non-traditional data has
shown both the potential and the limitations of these sources for crisis response [ 1–3].
Yet, answering such questions still requires the manual screening of hundreds (if not
thousands) of papers.
At the same time, large language models (LLMs) have advanced to the point where
they can support a wide range of knowledge-intensive tasks, including search-like
question answering, summarisation, and information extraction across scientific and
technical domains [ 4–6]. Nevertheless, these models remain prone to hallucination: they
produce fluent but non-factual statements or conflate sources, which undermines their
reliability as stand-alone knowledge repositories [ 7]. Retrieval-augmented generation
(RAG) addresses this issue by coupling LLMs with explicit document retrieval, so
that generation is conditioned on retrieved evidence and not just on the model’s
parameters [ 8,9]. For epidemiology, where provenance, transparency, and reviewability
are essential, such architectures provide a natural way to exploit the strengths of LLMs
while maintaining stricter epistemic standards.
This paper’s aim is not to automate epidemiological judgment. Instead, we focus on
reallocating human effort. RAG-based tooling can handle the mechanical processing of
research literature at scale (search, extraction, structured labelling), while researchers
focus on intentionality, domain expertise, and interpretation. This assistance can be
particularly relevant during emergencies, when time pressure makes it difficult to
maintain both breadth and depth in literature surveillance. Figure 1 summarises this
division of labour as a design principle.
We argue that epidemiology can benefit fromAI-assisted knowledge acquisition,
and we develop EpiLens, an open-source RAG package designed to support this shift.
Prior work already establishes the strengths of retrieval-augmented generation (RAG)
in grounding model outputs, as well as its vulnerabilities to low-quality corpora,
retrieval errors, and context misuse [ 8–10]. Therefore, we do not seek to re-establish
these findings, but focus on the design of workflows that apply retrieval-augmented
generation systematically across a corpus. EpiLens providesstructured workflows:
reusable analyses that fix the retrieval template and constrain the model’s output
with an explicit schema, then apply that same template identically to every paper in
a corpus. Constraining the task in this way is what makes the analysis scalable and
trustworthy: every paper is interrogated the same way, the output is validated against
the schema, and the result is machine-readable rather than free-form. We then use
a multi-pathogen corpus as a case study to show what becomes possible when such
workflows are applied to questions of data use, availability, and geographic coverage,
and to the source literature associated with compiled parameter estimates.
2

ὄ1
EpiLens/file-alt
Corpus/user-graduate
Researcher
Outbreak literature Extraction Synthesis
Judgment &
interpretationdata type
accessibility
geography
paper typeEpidemiological
Knowledge
Fig. 1 Division of labour between EpiLens and the researcher.Large-scale, unstructured
outbreak literature (left) is systematically processed by EpiLens to resolve a corpus into distinct,
structured dimensions. The dimensions shown (data type, accessibility, geography, and paper type) are
theexampleswe carry through this paper. They are not fixed by the system, which the user configures
for their own questions. By handling the mechanical extraction and provenance of these variables, the
system provides filtered evidence to the researcher (right). This separation lets the researcher move
from labour-intensive sorting of documents to higher-level work: defining label vocabularies, auditing
edge cases, and interpreting the resulting patterns.
1.1 Contributions
The contributions of this paper are:
•A Python package for structured, reusable analysis workflows over
epidemiological corpora.EpiLens turns an analyst-defined retrieval template
and output schema into a workflow that applies consistently across papers and
adapts to new tasks. It supports local or remote inference and links each output
to its supporting passages.
•A reliable, reusable worked example: the Paper Classifier.We develop
and evaluate a classification workflow across four label axes with different evidence
demands, showing the accuracy attainable without task-specific training labels
and where body-text retrieval is most useful.
•A corpus-scale case study in digital epidemiology.We provide a multi-
axis description of the review-seeded corpus of 2,297 papers and a source-level
comparison of the papers associated with parameter estimates from the seed study
by Ward et al. [11], by parameter and pathogen.
Section 2 positions the work, and Section 3 describes EpiLens, its two workflows,
and the evaluation setup. Section 4 evaluates the Paper Classifieron a manually labelled
sample, applies it to the full corpus, and links the corpus papers to the parameter
estimates of the seed study. Section 5 discusses how retrieval-grounded workflows
might fit into the knowledge-acquisition toolkit of digital epidemiology, in light of
our results and of recent work applying RAG in general scientific settings [ 5] and in
healthcare [ 12], and sets out the limitations. The appendices hold the detail behind
each step: system components, an extraction workflow, and inference cost (Section A),
corpus construction (Section B), the labelling protocol (Section C), full metrics, the
fully local deployment and error diagnostics (Section D), the baselines and statistical
tests (Section E), and the linkage of estimates to corpus papers (Section F).
3

2 Background
This work sits at the intersection of digital epidemiology, retrieval-augmented gener-
ation, and automated literature processing. From digital epidemiology it takes the
motivation to understand how diverse data sources and methods have been used across
outbreaks and to turn that understanding into a component of preparedness. From
RAG it takes the architectural pattern of grounding LLM outputs in retrieved evidence
and the associated evaluation concerns. From literature processing tools it inherits an
emphasis on machine-readable and extensible corpora and references. By combining
these elements in an open, lightweight package focused on epidemiological corpora,
EpiLens aims to make AI-assisted knowledge acquisition a part of digital epidemiology.
2.1 Digital Epidemiology and Data-Centric Preparedness
Digital epidemiology emerged as a response to the increasing availability of digital data
describing human behaviour, mobility, and health, and to the need for more timely
and granular information on disease dynamics. Early work framed digital epidemiology
as the use of digitally collected data and tools to study patterns of health and
disease, emphasising both its potential and the technical and computational challenges
it introduces [ 13]. Subsequent reviews have documented how digital epidemiology
expanded in scope, driven by mobile devices, online platforms, and novel sensing
technologies, and how it began to complement traditional surveillance by filling gaps
in timeliness or spatial coverage [ 3,14]. The COVID–19 pandemic accelerated the use
of varied data sources in epidemiology, including structured and unstructured data
collected inside and outside health systems [15].
Alongside this expansion came a need to better understand not only which data
exist but also how they have been used across pathogens, regions, and modelling
tasks. Work on non-traditional data during COVID–19 illustrates this point. The
Data4COVID19 Review [ 1] and related efforts [ 2] catalogued the ways in which mobility,
financial transactions, social media activity, and other unconventional sources were
used to support decision making, while also highlighting difficulties around integration,
governance, and sustainability. More broadly, recent assessments argue that digital
epidemiology will only fulfil its promise if the biases and limitations of new data sources
are explicitly recognised and if methods and data practices are adapted accordingly
[15]. In this context, being able to quickly map which data have been used for which
analyses, under which assumptions, becomes a component of preparedness in its own
right.
2.2 Large Language Models and Retrieval-Augmented
Generation
Large language models can summarise, answer questions, and extract information at
scale [16], but remain prone to hallucinating plausible but incorrect statements when
used as stand-alone knowledge sources [ 7]. Grounding conclusions in verifiable evidence
is a basic requirement across the empirical sciences. In epidemiology and public health,
the cost of failing to do so is especially high. A fabricated or misattributed finding
4

can propagate into surveillance summaries, resource allocation, and clinical or policy
decisions. During an outbreak, these decisions are made under time pressure, with little
opportunity to catch an error before it has consequences. An unverifiable model output
is therefore not merely an inconvenience in this setting but a direct risk. Retrieval-
augmented generation (RAG) addresses this by conditioning generation on passages
retrieved from an external corpus rather than on parametric memory alone [8].
RAG has also been used to work with scientific literature [ 5]. Recent surveys of
RAG architectures describe how retrieval and generation are combined for knowledge-
intensive tasks such as open-domain question answering and summarisation, and they
analyse common design choices and trade-offs in these workflows [ 9,10]. PaperQA2
and OpenScholar use RAG to synthesise cited answers to questions over scientific
literature [ 17,18]. Our focus is repeatable corpus analysis: EpiLens applies the same
retrieval template and output schema to every paper, producing comparable records for
aggregation and inspection. Ease of use is also a design goal: researchers can start from
a predefined workflow and adapt its retrieval template and output schema without
assembling the underlying RAG pipeline.
RAG has also begun to be adopted in healthcare and biomedicine, where factual
accuracy and provenance are critical. Surveys document applications in medical
question answering, guideline querying, and electronic health record summarisation [ 12,
19,20]. The challenge is to adapt these ideas to the specific needs of epidemiology, where
questions often concern data availability, study design, and modelling assumptions
rather than only clinical recommendations.
2.3 Approaches to classifying a literature at scale
Classification is our main example of an EpiLens workflow. In this context, it tackles
the need to organise a body of literature too large to screen by hand into analytically
useful categories, without turning each new research question into a new annotation
project. Methods for this differ along two requirements that are easy to conflate: i)
whether the category scheme is specified by the analyst or discovered from the corpus,
and ii) whether the method needs manually labelled training examples before it can
be applied. The Paper Classifier provides an analyst-specified scheme, revisable as
text, applied withnotask-specific training labels, and grounded in full-document
evidence through retrieval. Supervised classifiers share the category control but pay for
it in labels. Off-the-shelf zero-shot classifiers likewise apply analyst-specified categories
without task-specific training labels; the NLI and prototype baselines tested here score
the supplied title and abstract without retrieving passages from the paper’s body.
Topic models need neither but cannot be told the scheme. Recent general-purpose
decision models such as Jev [ 21,22] extend the zero-shot family: given analyst-written
options and their descriptions, they return a probability for each without task-specific
training. They decide from the text they are given, so they complement evidence
retrieval rather than replace it. We lay out this design space, instantiate each family
with standard methods, and summarise the head-to-head comparison in Section 4.2.3,
developing it in full in Section E.
5

3 Methods
This section describes the system design of EpiLens, our Retrieval-Augmented Genera-
tion (RAG) package for epidemiological literature analysis. The system is intended
to offload mechanical text-processing work (retrieval, evidence filtering, extraction,
and coarse categorisation) while keeping domain judgment and interpretation with
the researcher. EpiLens’s orchestration supports a family of structured workflows in
which the user defines the label vocabulary or extraction target and the shape of the
output. In this paper, we develop one such workflow, the Paper Classifier, as a worked
example and as the target of our quantitative evaluation.
3.1 System overview
EpiLens implements a standard three-stage RAG pipeline: ingestion and indexing,
retrieval, and generation. Ingestion parses papers, segments them into chunks, embeds
them, and stores them in a searchable index. Retrieval selects the most relevant chunks
for a query or template, optionally filtered by paper metadata. Generation prompts an
LLM to produce an output that is grounded in the retrieved evidence. We accompany
every output with structured provenance that links each extracted fact or label back to
the source paper and the specific text snippets used as evidence. EpiLens is local-first
in its orchestration. In the evaluated configuration, parsing, vector storage, retrieval,
and task logic run under the analyst’s control, while embedding and generation can
run locally or through external APIs. The package also permits a remotely hosted
vector database. This local layer governs what is sent to the model by selecting and
structuring relevant chunks, and by enforcing schemas before and after generation.
In this sense, we treat the LLM as a stateless computation step: it receives a locally
assembled prompt and returns a structured output, with no ability to query the corpus
or influence what context it receives.
Figure 2 illustrates the ingestion (indexing) and interrogation (querying) branches
of this pipeline. Additional implementation detail, including component interfaces and
backend choices, is provided in Section A.
EpiLens is deliberately lightweight, by which we mean three things. First, it
depends only on standard, swappable libraries rather than on a general-purpose RAG
framework. Document parsing uses unstructured [23] and pypdf, or GROBID [ 24]
for richer section structure. Embeddings use a sentence-transformers model [25] or
a provider embedding API (we use Gemini embeddings in our experiments). Vector
storage uses a local file index, FAISS [ 26], or a Qdrant server [ 27]. Schema definition
and validation use pydantic [28], and generation uses a provider API (OpenAI,
Google, Anthropic, OpenRouter) or a local model. Any layer can be swapped without
touching the others. Second, the core orchestration can run on commodity hardware
such as a laptop. External embedding, generation, or vector-database services are
optional deployment choices rather than architectural requirements. Setup is minimal:
local-first defaults run out of the box, with optional upgrades (GROBID parsing, a
Qdrant server, a Streamlit UI) for users who want to scale up. Third, cost is bounded:
because each prompt carries a fixed retrieval budget regardless of document length,
per-paper token usage is nearly flat in the length of the paper, which keeps corpus-scale
6

/file-alt
Corpus
(papers)
♂question-circle
Query/da◎abase
Vector DB
(index)/file-alt
Relevant
chunks♂search
TemplatesὌB
Instructions
(& schemas)♂robot
LLM
♂reply
Structured
output
+ provenanceembeddingindexing
embedding
searchtop-kLocal indexing, retrieval, and task logicLLM backend
(local or remote)
Fig. 2 RAG workflow used in EpiLens.Dashed arrows indicate one-time ingestion and indexing
of the corpus. Solid arrows indicate per-task interrogation (query embedding, search, and generation)
with structured outputs linked to source chunks.
application affordable (Section A.3). General RAG toolkits such as LangChain [ 29],
LlamaIndex [ 30], and Haystack [ 31] expose low-level primitives for assembling arbitrary
pipelines and maximise flexibility. EpiLens offers higher-level building blocks, from
ad-hoc retrieval-augmented question answering to thestructured workflows, trading
breadth of customisation for out-of-the-box usability.
3.2Structured workflows for systematic corpus characterisation
Beyond answering one-off questions, EpiLens supportsstructured workflows: automated
analyses that produce structured, machine-readable outputs rather than free-text
answers. Running RAG over each document of a corpus raises two problems that one-
off question answering does not. First, the analysis must beidenticalacross papers so
that the per-paper results are comparable and can be aggregated. Second, the outputs
must bestructured and validatedso that results can be computed across thousands of
papers. Our workflow resolves both through two artefacts that the user writes once: a
templateand aschema.
Thetemplatefixes how every paper is interrogated, through two parts with distinct
jobs. Itsretrieval queriesfixwhat evidence is gathered: a fixed set of search phrases
(or per-category exemplars) is run against each paper, so the same kind of passage is
surfaced from every document rather than left to whatever a one-off query happens to
return. Itsprompt instructionsfixwhat is done with that evidence: the task to perform
and the criteria for performing it, such as the operational definition of each label, held
identical across papers so that the same judgement is applied everywhere. Together they
guarantee that every paper is asked the same question over the same kind of evidence.
Theschemafixes theshape of the answer: the allowed fields, their types, and
their permitted values, expressed as a constrained JSON structure. It also acts as
a validation gate, since every generated output is checked against it and rejected
if it does not conform. This is what makes the results usable at scale: each one is
guaranteed to be well-formed structured data ready for aggregation, not prose to be
re-read and re-parsed by hand. As shown in Figure 2, the template governs the query
7

and instruction components of the interrogation branch and the schema governs the
structure of what the generator returns, turning an interactive RAG session into a
pipeline that can run unattended over an entire corpus.
The workflows reuse the same ingestion, retriever, generator, validation, and
provenance components, but configure them differently for their tasks. The Paper
Classifier performs category-specific evidence aggregation and returns labels from
a fixed vocabulary. The Precision Miner pools evidence from flat queries, can filter
by section, and returns a variable-length list of items. We evaluate the classifier
quantitatively and show the miner’s output on four corpus papers in Section A.2.
3.2.1 Paper Classifier
In this study, we configure the Paper Classifier to label each paper by data type, data
accessibility, geographical coverage, and paper type (Section C). The difficulty is that
the category scheme is a moving target: it is specific to the question and changes as the
question evolves, with new pathogens, a shifting geographic focus, or a revised coding
scheme. This is the setting that the usual tools handle poorly (Section 2.3): supervised
classifiers need hundreds of labelled examples per scheme version, and re-labelling
whenever it changes, while unsupervised methods cannot be told which scheme to
recover and instead surface what the vocabulary signals rather than what the question
requires. A RAG-based classifier sidesteps both: the analyst specifies each category in
natural language and revises it by editing text, with no task-specific training labels,
and retrieval grounds each decision in the paper’s own evidence. The trade-off becomes
an engineering one (retrieval quality, prompt design, model choice) rather than a data-
availability one. Mechanically, the classifier retrieves evidence for each candidate label
using exemplar-guided queries and asks the LLM to discriminate among the competing
categories, producing a standardised classification that supports a bird’s-eye view of a
literature. Component and workflow details are provided in Section A.
3.2.2 Precision Miner
Some questions need the concrete items themselves rather than a label: the datasets
and repositories a field uses, or the sources behind its parameter values. The Precision
Miner is theextractionworkflow for such questions: it pools evidence from several
retrieval queries per paper and returns a variable-length list of items, each with a
name, an optional URL or identifier, a short explanation, and the source sentence as
provenance. Its output is a per-paper worklist to verify rather than a distribution to
tally. Section A.2 describes it and shows its output on four corpus papers.
3.3 Evaluation setup
We evaluate the Paper Classifier on four axes:data type,data accessibility,geographical
coverage, andpaper type(Section C). The first three are set-based (multi-label), whereas
paper typeis a single primary label. Retrieval has a fixed budget of k= 15chunks
per label axis: for each axis, the system embeds every template paragraph, retrieves
itsknearest chunks from the target paper, pools these candidates across paragraphs,
8

removes duplicates, and keeps the khighest-scoring unique chunks as the evidence
context passed to the generator with the task instructions.
We compare four generator backends, with configurations that differ by model,
serving mode (local or remote), temperature, and, for the local generator, retrieval
index. Three models are accessed through remotely served inference: Google’s pro-
prietary gemini-2.5-flash andgemini-2.5-pro , each evaluated at T=0 and T=1,
and NVIDIA’s open-weight Nemotron (nemotron-3-super-120b-a12b , a mixture-of-
experts model with about12B active parameters, served free through OpenRouter),
evaluated at T=0 only. We self-host the open-weight qwen2.5:32b-instruct on our
own hardware and evaluate it at T= 0andT= 1. For each configuration, we repeat
the classification task R= 5times, holding the prompts and post-processing fixed
throughout. All remotely served configurations retrieve with gemini-embedding-001 .
For the self-hosted Qwen configurations, we evaluate two retrieval settings. In the
partially localsetting, generation runs on our hardware while retrieval uses the same
Gemini-embedded index as the remotely served configurations, improving comparabil-
ity across models. In thefully localsetting, both generation and retrieval are local,
with the corpus embedded on-device usingBAAI/bge-m3.
4 Results
After describing the corpus (Section 4.1), we evaluate the Paper Classifier on a manually
labelled sample (Section 4.2) using the settings of Section 3.3. We report accuracy and
run-to-run stability, test the contribution of retrieval by ablation, and compare the
classifier with external baselines. We then use the classifier to describe the full corpus
and the papers associated with the seed study’s parameter estimates (Sections 4.3
and 4.4). Section D.4 additionally compares the error sets of gemini-2.5-pro and
gemini-2.5-flash.
4.1 Case-study corpus
We evaluate EpiLens on a purpose-built corpus of epidemiological literature. The
corpus was seeded from the sources of a recent review of epidemiological parameters
for pathogens with pandemic potential [ 11], which we refer to as theseed study. Its
authors shared the data extraction sheets of an early version of the study (close to its
first preprint) with us [32].
Once identifiers are resolved, the sheets name138published sources: reviews,
studies, and other documents (Section F). We retained the directly cited studies and
other documents, then added the retrieved studies included in the source reviews.
The final corpus contains 2,297 papers and excludes the source reviews themselves. A
separate, semi-automated pipeline identified included studies with LLM assistance,
resolved bibliographic records and DOIs, retrieved documents, and ingested them.
Its archived records show what entered the corpus and what remained unresolved
or unretrieved, although reconstructing the exact document collection still depends
on access to the documents (Section B). We retain the links between corpus papers
and the seed study’s sources to define the labelled-sample strata (Section 4.2) and to
analyse the papers associated with its parameter estimates (Section 4.4).
9

4.2 Classification on the manually labelled sample
We evaluate the classifier on214papers sampled from the 2,297-paper corpus. The
sampling groups are the source reviews that contributed papers to the corpus and,
as one additional group, the corpus documents cited directly by the seed study
(Section 4.1). From each group of npapers, we randomly select max(⌊0.1n⌋,1)papers.
The one-paper minimum samples the smallest reviews at a higher rate than larger
ones. Only14sampled papers come from reviews with fewer than ten papers.
One annotator labelled each sampled paper on all four axes using a written protocol
that defines the labels and sets rules for ambiguous cases (Section C). We discuss
the limitation of relying on a single annotator in Section 5.2. The annotations were
completed before the annotator saw any model output. The classifier prompts use the
same label definitions, aligning the instructions given to the model with those followed
by the annotator.
We specified the classifier prompts, retrieval templates, and configuration from
the protocol before evaluating on the214papers and made no changes in response
to results on this set. The sample therefore serves as a held-out evaluation set for
EpiLens. Where a baseline required tuning of NLI wording, topic count, or supervised
model configuration, we report the strongest setting observed on these same papers
(Sections E.2.1, E.2.2 and E.5), which favours the baselines.
We report the unweighted mean per-paper Jaccard similarity ( J) on each axis, with
a95%bootstrap interval over papers. For each paper, Jis the number of labels shared
by the prediction and manual annotation divided by the number in either set. For
paper type, which has one label per paper, this is ordinary accuracy. We report micro-
and macro-averaged precision, recall, F 1, and subset accuracy in Section D. Because
papers from the smallest reviews were sampled at a higher rate, we checked whether
unequal sampling affects the reported averages. Weighting each paper by the inverse
of its selection probability changes the mean Jof EpiLens with gemini-2.5-flash at
T=0by at most0.005on any axis (effective sample size207).
4.2.1 Performance and stability of EpiLens
Table 1 reports the agreement of the served backends with the manual labels.
The served models perform well ongeographyandpaper type: at T= 0, all three
reachJ≥0.84, against majority baselines of0 .353and0.593, and both Gemini models
exceed0.87.Data typescores are also high (0 .950for Pro and0 .916for Flash at
T=0) but should be read against a majority baseline of0 .843. The manual sample
has too few examples of several data-type labels to establish their individual accuracy
(Section D.2).Data accessibilityis the most difficult axis: at T= 0, Pro reaches0 .802,
against0.670for Flash,0 .660for Nemotron, and a majority baseline of0 .472, and its
Pro–Flash gap (0 .132) is the largest of any axis, so this task benefits most from the
stronger model. Temperature has little systematic effect: moving from T= 0toT= 1
changes mean Jaccard by less than0 .03in every case, in no consistent direction. The
open-weight Nemotron-120B is competitive with Flash, within0 .04on every axis. Pro
has the highest mean of the served models on all four axes.
10

Table 1 Accuracy of the served backends.
Floor gemini-2.5-pro gemini-2.5-flash Nemotron
TaskMajority T=0T=1T=0T=1T=0
Geography0.353 0.928 0.89–0.95 0.925 0.89–0.95 0.875 0.83–0.91 0.871 0.83–0.90 0.886 0.84–0.92
Data access.0.472 0.802 0.76–0.84 0.775 0.73–0.81 0.670 0.62–0.72 0.684 0.65–0.72 0.660 0.62–0.70
Data type0.843 0.950 0.92–0.97 0.941 0.91–0.96 0.916 0.88–0.94 0.913 0.88–0.94 0.877 0.84–0.91
Paper type0.593 0.903 0.86–0.93 0.901 0.86–0.93 0.877 0.83–0.91 0.880 0.84–0.91 0.848 0.80–0.89
Mean Jaccard similarity on the214-paper evaluation set over R=5runs per configuration. The
subscript is a95%bootstrap interval over papers (BCa, B=10 000), computed after averaging each
paper’s score over the runs: it shows how precisely these papers determine the score. Each interval
describes one configuration; methods are compared with the paired tests of Section E.4. Run-to-run
standard deviations are in Table D1. TheFloorcolumn (set off on the left) is the majority-class
baseline, a single deterministic value included as a reference point for reading the scores.Nemotron
is an open-weight backend (NVIDIA Nemotron-120B) run through the identical pipeline at T=0.
Paper type is a single-label task. The remaining tasks are multi-label.
The self-hosted qwen2.5:32b-instruct configurations also score above the major-
ity baseline on every axis, though only slightly ondata type(0 .861–0.879against0 .843).
They come close to Flash ongeography(0 .869against0 .875when fully local), and lose
most ondata accessibility(0 .539against0 .670). AtT=0, moving the embedding step
on-device changesJby at most0.024on any axis (Section D.3).
Mean agreement with the manual labels does not show whether repeated runs
assign each paper the same labels. We measurestabilityas the share of papers whose
complete label set is identical across all five runs.
Temperature affects paper-level stability much more than mean J(Table 2). At
T=0, the Gemini models are stable on87–95%of papers ongeography,data type, and
paper type, but only63 .6–66.8%ondata accessibility, which is also the least stable
axis for Nemotron. Both local Qwen configurations return identical labels on every
axis atT=0. AtT=1, however, stability on data accessibility falls to32 .2%for Flash
and below21%for Qwen. Mean Jnevertheless varies little between runs (standard
deviation at most0 .027in any configuration; Tables D1 and D5): within a run, papers
that gain and lose agreement largely offset each other, and a change often alters only
part of a label set.
11

Table 2 Label stability across repeated runs.
Qwen2.5-32B(local)Gemini pro Gemini flash Nemotron
fully local partially local
TaskT=0T=1T=0T=1 T=0T=1T=0T=1T=0
Geography 100.0 80.4 100.0 78.5 94.9 93.0 92.1 81.3 86.9
Data access. 100.0 17.8 100.0 20.6 63.6 51.4 66.8 32.2 44.9
Data type 100.0 79.4 100.0 83.6 89.3 90.2 94.9 89.3 86.9
Paper type 100.0 87.4 100.0 86.9 87.9 86.0 89.7 76.2 82.7
Percentage of papers whose predicted label set was identical across five repeated runs
under each configuration. A prediction is considered stable if the full label set is unchanged
across runs. TheQwen2.5-32Bcolumns are the self-hosted backend of Section D.3:fully
localembeds the index locally with BAAI/bge-m3 , whilepartially localreuses the Gemini-
embedded index. Both are perfectly stable at T=0, returning an identical label set for every
paper across all five repeats.Gemini proandGemini flashare gemini-2.5-pro and
gemini-2.5-flash .Nemotronis the open-weight backend (NVIDIA Nemotron-120B) at
T=0. The majority floor is omitted here because it is deterministic.
Notice all three served backends change some labels even at T=0. Variation under
nominally deterministic settings has also been observed elsewhere [ 33]. Changes in
batching and floating-point reductions can alter outputs under greedy decoding [ 34–
36]. We did not vary batching or numerical precision while holding the model fixed, so
these mechanisms remain possible explanations for the served systems’ label changes.
4.2.2 Ablation of body-text access and selection
Aggregate performance does not reveal which components of the workflow are driving
its gains. Therefore, we compare conditions that, within each generator, use the same
model, prompt, and post-processing and differ only in the evidence supplied:
1.No body text( metadata_llm ): the model receives only the paper’s title and
abstract.
2.Random body retrieval( random_chunk_llm ): the prompt additionally receives
kbody-text chunks sampled uniformly at random from the paper.
3.Targeted retrieval(full EpiLens): the same number of body-text chunks is selected
by semantic similarity to the template paragraphs.
4.Whole body( full_body_llm ): the prompt receives every body-text chunk of the
paper, in document order, with no selection.
The contrast from step 1 to step 2 measures the contribution of access to body text,
while step 2 to step 3 measures the additional contribution of semantic targeting at a
fixed evidence budget. Step 4 tests whether selecting passages improves on supplying
the whole parsed body.
The ablation shows the same qualitative decomposition for both generators
(Figure 3). The largest and most consistent gain comes from access to body text, even
when the passages are selected at random. For gemini-2.5-flash , moving from title
12

0.00.20.40.60.81.0
0.710.83 0.82
0.240.87 0.89 0.90
0.500.88 0.870.92
0.670.890.850.90
0.63Served backend    gemini-2.5-flash
Paper type Geography Data type Data access.0.00.20.40.60.81.0
0.660.81
0.74
0.210.810.87 0.86
0.410.790.87 0.86
0.540.820.85 0.86
0.41Self-hosted backend    qwen2.5:32b-instruct
Jaccard similarity J1. No body text 2. Random body 3. T argeted retrieval 4. Whole bodyFig. 3 Ablation results for two generators.The upper panel uses the served gemini-2.5-flash
model and the lower panel the self-hosted qwen2.5:32b-instruct , both at T=0. Within each panel,
the model, prompt, and post-processing are fixed. The four conditions are title and abstract only, title
and abstract plus k=15randomly selected body-text chunks, title and abstract plus k=15semantically
retrieved chunks, and title, abstract, and the whole parsed body. Comparing conditions 1 and 2 shows
the effect of adding body text. Comparing conditions 2 and 3 shows the effect of targeted selection
with the same number of chunks.
and abstract alone to random body-text retrieval increases Jby+0.06ongeogra-
phy, +0.26ondata accessibility, +0.09ondata type, and +0.16onpaper type. The
corresponding gains for qwen2.5:32b-instruct are+0.06,+0.20,+0.12, and +0.15.
Papers in this corpus have33chunks on average (Section A.3), so the15randomly
selected chunks already expose the model to much of the body. The starting point
nevertheless differs by task. With Flash, title and abstract alone already give J=0.826
ongeography,0 .816ondata type, and0 .713onpaper type, but only0 .241ondata
accessibility, indicating that the latter depends much more strongly on information
beyond the abstract.
Semantic targeting contributes additional value more selectively. Its largest effect
is ondata accessibility, where replacing random chunks with targeted ones raises Jby
+0.17for Flash (0 .504→0.670) and by +0.13for Qwen (0 .412→0.539). On the other
three axes, the effect is small or absent. In the paired paper-level resampling check of
Section E.4, the gain from adding random body-text chunks to title and abstract meets
the within-task Holm-adjusted threshold on every axis for both generators. The gain
from replacing random chunks with targeted ones meets it only on data accessibility.
13

Under the same check, the whole-body condition shows no clear accuracy advantage
over the15targeted chunks on any axis, despite using about twice the input tokens
(Figure 3 and Section A.3).
Ondata accessibility, Qwen falls from J=0.539with targeted chunks to0 .410with
the whole body (pointwise95%BCa interval for the difference,0 .05–0.20). This is the
only one of the eight targeted-versus-whole-body comparisons that meets the within-
task Holm-adjusted threshold. The result suggests that selecting passages can help on
this task even when the whole parsed body fits in the model’s context window.
4.2.3 Comparison with classification baselines
We compare EpiLens with alternative workflows that a practitioner might use to
organise a literature corpus. Standard supervised classifiers pair a paper representation
with a classifier head trained on labelled examples. Here, we use TF-IDF+linear SVM
for a lexical setup and frozen ModernBERT with an MLP head for a long-context
setup. Zero-shot NLI instead uses a pretrained language encoder to score how strongly
a paper supports each verbalised category, without task-specific fitting. Guided and
semi-supervised BERTopic discover corpus-specific clusters rather than applying a
fixed taxonomy, so their topics require subsequent interpretation and alignment. SetFit
provides a distinct supervised route, contrastively fine-tuning an MPNet encoder on
pairs of labelled examples before fitting a classification head. We also include two
prototype approaches, which encode each category description (or, for geography, its
name) in sparse TF-IDF or dense SPECTER space and assign papers by similarity,
with label-free and threshold-calibrated variants. Finally, as in the preceding results,
a majority prior that always predicts the most prevalent label set provides a no-
text reference floor. The abstract-only metadata_llm row is retained only as the
scope-matched EpiLens condition analysed in the ablation above.
The methods also differ in practical text scope, recorded byBody?in Table 3.
The prototype, NLI, and SetFit baselines use title and abstract, whereas BERTopic,
ModernBERT, and full EpiLens use parsed body text. The broader supervised grid
tests both scopes wherever possible. Body text is not uniformly beneficial: some
representation–head combinations decline slightly, perhaps because additional text
dilutes task-relevant cues already concentrated in the abstract, although the best
full-body configuration raises the supervised ceiling on every axis.
Creating a sufficiently broad labelled set is a substantial additional step for a large
or diverse corpus, and a changed coding scheme generally requires renewed labelling
and model fitting. All rows are scored on the same papers with the same metric. Where
a baseline needs tuning, we report its strongest setting, which favours the baselines
because EpiLens was fixed before evaluation (Section 4.2). The notes to Table 3 define
its columns and its upper-bound rows, and Section E gives the configurations, tuning,
complete supervised grid, and statistical comparisons.
Against this deliberately favourable comparison, full EpiLens with Flash has the
highest point estimate on every axis, including the retrospective supervised ceiling,
without task-specific training labels. In a paired paper-level resampling check, the
higher mean Jof Flash than TF-IDF+SVM, SetFit, and ModernBERT meets the
14

Table 3 Performance and requirements of the baseline approaches.
Method Requirements Jaccard by task
Body?Task
sup.Paper
typeGeo-
graphyData
typeData
access.
Reference
Majority prior (floor) No Weak 0.593 0.353 0.843 0.472
Zero-shot (no training labels)
Prototype (TF-IDF) No None 0.407 0.283 0.315 0.217
Prototype (SPECTER) No None 0.126 0.165 0.515 0.360
Zero-shot NLI (DeBERTa) No None 0.565 0.709 0.018 0.354
Topic models (corpus-derived, label-guided)
Guided BERTopic Yes Weak 0.621 0.576 0.843 0.472
Semi-supervised BERTopic Yes Weak 0.626 0.581 0.843 0.477
Supervised (title + abstract)
TF-IDF + Linear SVM No Manual 0.713 0.592 0.854 0.533
SetFit (MPNet) No Manual 0.734 0.599 0.864 0.574
Prototype (TF-IDF) No Manual 0.407 0.339 0.315 0.247
Prototype (SPECTER) No Manual 0.126 0.179 0.672 0.363
Supervised (full body)
ModernBERT (MLP head) Yes Manual 0.785 0.652 0.864 0.577
Best supervised(ceiling) Yes Manual 0.785 0.771 0.898 0.619
LLM (abstract-only ablation and full EpiLens)
Abstract only
(metadata_llm)No None 0.713 0.826 0.816 0.241
EpiLens(gemini-2.5-flash) Yes None0.877
0.83–0.910.875
0.83–0.910.916
0.88–0.940.670
0.62–0.72
EpiLens(gemini-2.5-pro) Yes None0.903
0.86–0.930.928
0.89–0.950.950
0.92–0.970.802
0.76–0.84
Sample-averaged Jaccard similarity Jon the fixed214-paper evaluation set.Body?indicates
whether parsed body text is required.Nogenerally means title and abstract, while the
majority floor uses no text.Task sup.records the task-specific input used to construct
or fit the method:Noneuses no labelled papers,Weakuses a class prior, seeds, or partial
paper labels, andManualuses a fully manually labelled training set. All scores use the same
manual labels. Prototype rows contrast uncalibrated similarity with threshold calibration.
BERTopic uses the best topic count and oracle cluster-to-label alignment.Best supervised
is the per-task maximum over the supervised grid. Both are favourable descriptive upper
bounds rather than single deployable configurations. For EpiLens, the mean is over R=5runs
atT=0, with its95%bootstrap interval over papers beneath it, as in Table 1. Other rows are
deterministic evaluations or cross-validated estimates. Scores are rounded to three decimals.
Paired tests are reported in Table E1. Full method specifications are given in Section E.
within-task Holm-adjusted threshold on all four axes (Section E.4 and Table E1). Pro
scores higher still.
4.3 Corpus-level analysis
As mentioned in Section 4.1, the corpus was assembled from sources cited for epidemi-
ological parameter estimates across several pathogens. Having evaluated the classifier
15

on a manually labelled sample, we turn to the literature those sources brought together.
The overall distribution of labels across 2,297 papers provides a reference for com-
paring the papers associated with different parameters and pathogens. Both analyses
usegemini-2.5-flash atT=0, chosen for its lower cost and strong agreement with
manual labels on paper type, geography, and data type (Section 4.2.1). We report
corpus shares only for those three axes.
Across the full corpus, the model labels57 .7%of papers as empirical and25 .0%
as inference, with46 .1%covering Asia and37 .9%Africa. It labels96 .3%as using
traditional data and18 .5%as presenting simulated data. Detailed distributions and a
fully local sensitivity check appear in Figure D1 and Section D.3.4.
4.4 Papers associated with parameter estimates
We now turn from the full corpus to the papers associated with the seed study’s
parameter estimates. We make these comparisons using the source links retained
during corpus construction (Section 4.1). The privately shared early extraction sheets
contain346estimate rows; our source-link audit finds that327cite a published source
(Section F). The other19do not enter this analysis. For each of the327, we link the
estimate to its cited corpus document or, if it cites a review, to the retrieved studies
included in that review. We refer to these as thepapers behindan estimate. When
an estimate cites a review, the link does not identify which of the retrieved included
studies informed the particular value. We describe those studies as a group associated
with the estimate, without attributing the value to any one study (Section F). Of those
estimates,180cite reviews and147cite single studies or other documents (Figure 4).
0 20 40 60 80 100
Number of estimatesReproduction number
Fatality risk
Serial interval
Incubation period
Dispersion k
Infectious period
Latent period68
83
44
53
17
35
27Source
Expanded review
Direct source
Fig. 4 Sources of parameter estimates.Estimates of each parameter, by source: a review whose
included studies are in our corpus, or a single study or other document in our corpus. Parameters are
ordered by the share of their estimates taken from reviews.
4.4.1 Parameter
Figure 5 compares the profiles of papers associated with each parameter. Its first row
shows all 2,297 corpus papers for reference. The profiles use the same paper-type,
data-type, and geographic labels as the corpus overview. For this comparison, we
group papers labelled inference or forecast undermodel-based, including statistical
16

distribution fitting and transmission modelling, and use the synthetic-data label for
simulated data(Section C). The model-based and simulated-data shares are tabulated
in Table F1.
0 50 100%All papers
Serial interval
Dispersion k
Reproduction number
Latent period
Infectious period
Incubation period
Fatality risk58 25 n=2297
28 12 51 n=539
26 1712 45 n=137
28 14 10 44 n=629
32 21 19 28 n=140
45 17 14 23 n=186
56 29 n=845
65 19 n=1549a  Paper type
Empirical
Methodological
Synthesis or commentaryOther or unclear
Forecast
InferenceTraditional Simulated Non-traditional No data
96 19 5 2
97 38 7 1
100 55 5 0
97 48 8 1
96 51 5 0
97 42 4 1
97 20 7 2
96 13 5 2b  Data type
Africa Asia Europe North America South America Oceania No location
38 46 26 19 10 8 6
29 48 24 23 9 9 5
77 22 911 3 4 2
38 47 26 25 15 11 2
89 614 13 2 3 3
74 16 27 25 11 12 3
29 52 26 19 810 7
46 46 24 13 7 6 4c  Geography
100% = 2297
10%
2%
12%
1%
3%
19%
53%d  Share of papers
Fig. 5 Profiles of associated papers by parameter.The first row describes all 2,297 corpus
papers, and each other row the papers behind the estimates of one parameter, as labelled by EpiLens
(model outputs, not estimates). Rows are ordered by the share of model-based papers, and nis the
number of papers. (a) Percentage of the papers with each paper type. Each paper has one, and model-
based papers are those labelled forecast or inference. (b, c) Percentage of the papers with each data
type and each continent. A paper can have several. “No data” means no empirical data, and “no
location” means irrelevant or unclear. (d) Fractional allocation of the corpus: each paper has total
weight one, divided equally among the pathogen–parameter pairs whose estimates it lies behind, so
that the row weights sum to100%.
We focus first on reproduction numbers and fatality risks, the parameters with
the most estimates (68and83). The model labels54%of the629papers behind
reproduction numbers as model-based, against22%of the1 ,549behind fatality
risks. These shares cover all associated papers but depend on model-assigned labels.
We therefore check the contrast on the manually labelled sample, where62%of61
reproduction-number papers and24%of144fatality-risk papers are model-based. A
higher model-based share for reproduction numbers is plausible given the role of model
fitting in their estimation.
The papers linked to reproduction-number estimates and fatality-risk estimates
cover different mixes of pathogens, which could contribute to the overall gap. We
therefore compare their model-based shares within pathogens. In each of the five
with at least20papers behind both parameters, the reproduction-number papers still
have a higher model-based share, although the gap narrows to4–25percentage points
(Table F2). The direction also holds under the fully local model and when any one
17

source review is omitted (Section F). The simulated-data label shows a related contrast
(48%versus13%), although it often overlaps with the model-based label.
4.4.2 Pathogen
We now compare the corpus papers linked to the seed study’s cited sources across
pathogens. Figure 6 shows their paper-type, data-type, and geographic profiles, pooling
papers across parameters for each pathogen.
0 50 100%All papers
SARS-CoV-2
Influenza A
MERS-CoV
SARS-CoV-1
Zika
Ebola
Marburg
Lassa
Mpox
CCHF
RVF
Nipah58 25 n=2297
36 48 n=590
30 15 45 n=199
47 48 n=90
41 12 33 n=64
56 11 27 n=146
58 11 19 n=262
69 17 n=59
72 10n=125
80 n=153
73 11 n=497
86 n=214
84 10 n=112a  Paper type
Empirical
Methodological
Synthesis or commentaryOther or unclear
Forecast
InferenceTraditional Simulated Non-traditional No data
96 19 5 2
97 28 10 2
97 39 4 1
100 33 10 0
94 44 3 0
97 33 3 1
96 25 5 1
95 7 3 3
94 10 2 1
99 8 6 1
93 1 2 4
99 2 4 1
99 4 8 1b  Data type
Africa Asia Europe North America South America Oceania No location
38 46 26 19 10 8 6
12 70 26 18 11 10 2
12 41 29 40 11 12 8
14 97 18 9 3 3 2
959 19 28 911 17
21 34 25 42 42 23 10
69 23 12 14 6 2 3
92 217 10 0 0 7
92 123 9 2 2 6
58 18 50 33 14 9 5
40 61 33 9 7 5 4
93 911 7 5 3 1
647 10 21 219 21c  Geography
100% = 2297
25%
9%
3%
2%
6%
9%
1%
5%
7%
19%
9%
5%d  Share of papers
Fig. 6 Papers linked to cited sources by pathogen.The first row describes all 2,297 corpus
papers. Each other row describes papers linked to sources cited for that pathogen across all parameters
in the seed study. The panels show the same measures as in Figure 5. Pathogens with at least10
associated papers are shown. Influenza A subtypes are merged here and listed separately in Table F1.
CCHF: Crimean–Congo haemorrhagic fever. RVF: Rift Valley fever.
In the model’s labels, the SARS-CoV-2 and influenza A source sets contain larger
shares of inference papers and papers using simulated data than are found in the
corpus as a whole. The mpox, RVF, and Nipah sets are predominantly empirical. In
the geography panel, nearly all papers linked to MERS-CoV cover Asia, and nearly all
those linked to Marburg and RVF cover Africa.
18

We also compare pathogens within a single parameter. Among papers linked to
incubation-period estimates,55%of the334behind SARS-CoV-2 are labelled model-
based, compared with6%of the115behind mpox (Table F2). The manual labels
show the same ordering (61%of31versus9%of11). These papers come from six
and five source reviews or directly cited documents, respectively, so the manual check
supports the direction more than a precise magnitude. Excluding papers also linked to
reproduction numbers or fatality risks leaves a similar contrast (54%of201versus6%
of70). It is not driven by one review (Section F). Fatality-risk papers also differ by
pathogen:52%of those linked to SARS-CoV-2 are model-based, against5–26%for
CCHF, Ebola, Lassa, Nipah, and RVF (Table F2).
5 Discussion
In this work we present EpiLens, a Retrieval-Augmented Generation (RAG) package
for organising epidemiological literature at a scale that manual review cannot reach. It
treats the generator model as a linguistic engine over retrieved passages and retains
the intermediate artefacts behind each output (the retrieved passages, the prompt,
and the schema-validated label or extraction with a back-link to its source chunks), so
that any aggregate result can be traced to the evidence that produced it rather than
discarded once the answer is returned.
Our evaluation shows both the effectiveness of the retrieval-augmented workflow
and how its performance depends on the target analytical task. In the broad corpus-
organising axes ofgeography,data type, andpaper type, the full RAG workflow achieves
strong agreement with the manual annotations and matches or exceeds the tested
supervised classifiers without requiring task-specific training labels (Section E). Perfor-
mance is lower ondata accessibility, which emerges consistently as the most difficult
of the four axes, although it remains substantially above the majority baseline and
improves with the stronger served model.
The ablation analysis clarifies where these gains come from (Section 4.2.2). Across
both the served gemini-2.5-flash model and the self-hosted qwen2.5:32b-instruct ,
access to body text provides the largest and most consistent improvement over title
and abstract only classification. Semantic targeting adds a further, task-dependent
gain: it matters most fordata accessibilityand little forgeography. This pattern is
consistent with access-related evidence being comparatively sparse or localised and
place-related cues being more widespread through a paper. Supplying the whole body
showed no clear accuracy advantage over selected passages despite about twice the
input tokens. Ondata accessibility, it reduced accuracy for the self-hosted generator.
The pattern recurs across two generators, while the overall performance level remains
model-dependent.
More broadly, the workflow fills a different role from standard supervised classifi-
cation. An analyst can define or revise a coding scheme in natural language, apply
it directly to full document evidence, and retain provenance for each classification.
This makes the approach particularly useful when taxonomies are evolving, labelled
examples are limited, or the relevant evidence is not consistently available in titles
and abstracts. It also makes the workflow complementary to existing approaches: an
19

annotation-free pass can map a corpus, identify difficult cases for expert inspection,
or provide candidate labels for subsequent supervised modelling or manual curation.
Once an axis has been validated on a labelled sample, the same specification can then
be applied across the full corpus to produce structured distributions while preserving
a path back to the papers and passages that support them. Applied to our corpus, the
labels describe both the literature as a whole and the paper sets linked to the seed
study’s parameter estimates (Sections 4.3 and 4.4). In this setting, domain expertise
shifts from manually coding every document toward defining operational categories,
validating their reliability, and auditing the evidence behind decisions.
5.1 Implications for Digital Epidemiology and Preparedness
Pandemic preparedness depends in part on having a transparent, up-to-date map of
what has been studied, on which data, and in which settings. EpiLens helps maintain
such a map, while retaining the link between high-level summaries and the underlying
documents that support them. However, a concern in digital epidemiology is not
only whether data exist, but how they have been used across pathogens, regions,
and modelling tasks. The corpus-level analyses outlined in Section 4.3 begin to make
this usage legible in a structured, updateable form. The four dimensions describe
complementary features of the literature. Data type records the kinds of evidence
used. Data accessibility records stated access routes or whether printed values permit
limited reconstruction. Geographical coverage locates the evidence, and paper type
records the principal contribution. Together they support retrospective assessment
and prospective planning by making the corpus’s mix of study types, data sources,
and geographic settings explicit. Aggregate agreement with manual labels on three
of these four dimensions (Section 4.2.1) suggests that such a characterisation can be
obtained with limited manual coding and updated as new studies appear. The fourth
dimension,data accessibility, is both the hardest for the workflow and arguably the
most directly preparedness-relevant: knowing whether the data behind past analyses
was made available and reusable is a prerequisite for any rapid re-analysis when a new
threat emerges. Because this axis has lower agreement and stability, improving it is a
priority. We examine why it is hard, and how to address it, in Section 5.2.
5.2 Limitations and Future Work
RAG reduces unsupported generations by conditioning outputs on retrieved passages,
but errors can still occur when relevant evidence is missed, incomplete, or incorrectly
interpreted.
Evaluation scope.
The present evaluation assumes that the coding scheme can be specified independently
of the corpus being evaluated. Our four axes were defined in advance using explicit
operational criteria and externally anchored distinctions, including the United Nations
M49 standard for geography [ 37] and the traditional/non-traditional data distinction
from previous work [ 1]. This makes it possible to evaluate the resulting fixed scheme
directly on the annotated sample. Other applications may instead derive or refine
20

their categories after inspecting the corpus or preliminary outputs. In that setting,
the workflow’s ability to revise categories in natural language remains useful, but the
evaluation design should separate a development set used to refine the scheme from a
held-out set used only for final performance estimation.
Corpus and annotation.
Some limitations arise from the case-study corpus rather than from EpiLens itself. The
corpus was assembled through a semi-automated pipeline (Sections B and 4.1). This
process is explicitly best-effort and logs exceptions such as inaccessible PDFs, incom-
plete references, and non-machine-readable supplements, but downstream retrieval
and classification can only operate on material that has been successfully identified,
parsed, and indexed. At the same time, EpiLens can provide useful downstream warn-
ing signals, for example when an unexpected paper-type label suggests that a spurious
document has entered the corpus. A further limitation is that the evaluation labels
were produced by a single annotator following the explicit coding protocol in Section C.
Operational definitions and tie-breaking rules reduce ambiguity, but they do not allow
us to report inter-annotator agreement. A multi-annotator evaluation with adjudicated
labels is therefore an important next step.
Tracing estimates to papers.
Our analysis links estimates to their cited sources. For an estimate citing a review,
the associated set contains its retrieved included studies, which may cover related
infections. The link does not identify which studies contributed to the reported value.
Parameter-specific study lists, where reviews provide them, would allow more precise
links, as theepireviewdata illustrate [38].
The corpus contains61%of the reported inclusions in expanded source reviews
with available counts, with coverage varying across reviews (Section F). Restricting the
reproduction-number–fatality-risk comparison to better-covered reviews preserves its
direction but changes the individual label shares. The profiles also use model-assigned
labels. The manual labels show the same ordering for the main contrasts, while the
small labelled sets behind some pathogen–parameter comparisons limit how precisely
their differences can be assessed.
Evidence retrieval and verification.
The ablation points to evidence selection as one important target for improvement.
Relevant statements may be missed because chunking separates them from their
context, the retrieval query does not match the terminology used in the paper, or
ranking places the decisive passage below the cut-off. This is most visible fordata
accessibility, which gains the most from targeted retrieval and also benefits most from
the stronger generator (Section 4.2.2). Improving this axis therefore calls for attention
to both evidence selection and subsequent interpretation. Promising directions include
hybrid lexical and dense retrieval, section-aware retrieval targeting locations such as
Methods and Data Availability, stronger reranking, and direct measurement of retrieval
recall separately from downstream classification performance. The accessibility axis
also illustrates an important scope distinction: our protocol evaluates what the paper
21

reportsabout access to its data, not whether the stated access mechanism still works.
A natural extension is therefore to add a verification stage that checks resolvable links,
accession identifiers, repositories, or request mechanisms after the reported access
condition has been extracted.
Reliability and repeatability.
Repeated runs provide an additional signal for identifying difficult cases. Most configu-
rations are highly stable on the three stronger axes, while disagreements concentrate
more heavily ondata accessibility. These disagreements can be surfaced automatically
for inspection and could support future consensus or abstention strategies, provided
they are not interpreted as calibrated confidence without further validation. Agreement
across models provides a complementary signal: on three of the four tasks, the stronger
model’s errors largely fall within the weaker model’s error set (76–80%containment at
T=0, Section D.4), identifying a shared subset of difficult papers.
Extensions.
Finally, a package like EpiLens benefits from the continuing improvements in language
models and retrieval components. Stronger locally runnable models may narrow
the remaining gap to the served models, while task-specific fine-tuning of smaller
models could provide an attractive option when labelled examples are available and a
classificationschemehasstabilised.Betterembeddingmodels,rerankers,andstructured
decoding methods can likewise be incorporated. More agentic approaches could also
be useful when kept deliberately constrained, for example by triggering a second
retrieval step, checking an external identifier, or requesting additional evidence only for
papers whose initial classification is ambiguous or unstable. Such selective escalation
is preferable here to long autonomous agent chains, since it preserves the predictable
cost, provenance, and inspectability that motivate the framework in the first place.
5.3 Ethical Considerations
Epidemiological evidence informs surveillance, resource allocation, and public-health
policy. Errors in how it is organised or interpreted may therefore have serious con-
sequences, particularly during emergencies. We therefore state the intended role of
EpiLens, the risks associated with its use, and the safeguards needed to mitigate them.
As with any analytical tool used in a consequential setting, its outputs should be
interpreted within clearly defined conditions of use and by people able to assess them.
EpiLensactsasafilterthroughwhichevidenceisviewed.Itsclassificationscanaffect
which studies are surfaced, inspected, prioritised, or treated as absent, so systematic
classification errors could distort the resulting map of the literature. Their practical
consequences depend on how the labels are used. When labels only prioritise papers
for expert review, false positives mainly add verification work.
When labels are used to exclude papers or stop screening, false negatives can alter
the apparent evidence base by removing relevant studies from view. Work on ML-
assisted screening therefore treats the risk of missed studies as something that must
be assessed and communicated [ 39]. Such omissions are harder to detect and correct
during outbreaks, when decisions are urgent, scientific information is uncertain and
22

changing,andtimeandresourcesarelimited.Iftheyarealsounevenlydistributedacross
regions or languages, the resulting map may reinforce existing under-representation.
WHO raises a related distributional concern for scientific research supported by large
multimodal models [40].
Accordingly, EpiLens should support evidence triage and corpus mapping rather
than replace systematic screening, critical appraisal, or expert judgement. In this study,
the manually labelled evaluation and repeated runs quantify performance and stability
for the case-study corpus, while provenance supports inspection of individual outputs.
Applications to a new corpus or coding scheme should therefore be validated on relevant
labelled examples, with closer review of unstable outputs and of performance across
relevant corpus strata where sample sizes permit. Domain experts should verify outputs
before they inform consequential research or preparedness decisions, and the level of
review should reflect the intended downstream use. Performance should be reassessed
after material changes to the corpus, taxonomy, retrieval pipeline, or underlying model.
6 Conclusions
We presented EpiLens, a retrieval-augmented generation package that applies analyst-
defined questions identically across a corpus of epidemiological papers, returns
structured, validated outputs, and keeps a path from each output back to its evidence.
Without task-specific training labels, the Paper Classifier matches or exceeds the super-
vised and zero-shot approaches tested for geography, data type, paper type, and data
accessibility. Agreement with manual labels is strongest for the first three, while data
accessibility remains more difficult. The ablations show that finding relevant evidence
in full texts is an important part of this performance.
We also applied the classifier to 2,297 outbreak-related papers assembled from the
cited sources of a multi-pathogen study of epidemiological parameters. The labels
describe this corpus as a whole. Linking them to the study’s cited sources shows that
the kinds of papers associated with its parameter estimates differ across parameters
and pathogens.
More broadly, we argue for treating tools such as EpiLens as part of epidemiology’s
analytical infrastructure. Their role is not to replace expert review, but to shift expert
effort from manually processing every paper toward defining meaningful questions,
validating how reliably they can be answered, and interrogating the evidence behind the
resulting map of the literature. Because the analytical scheme, retrieval components,
and language model remain separable, the same package can incorporate improvements
in models and retrieval without sacrificing provenance or inspectability. As corpora
grow and models improve, such tools can help ensure that the knowledge accumulated
across past epidemics remains accessible and actionable when the next threat emerges.
Abbreviations.AI: artificial intelligence. API: application programming inter-
face. BCa: bias-corrected and accelerated. CCHF: Crimean–Congo haemorrhagic
fever. COVID-19: coronavirus disease 2019. DOI: digital object identifier. JSON:
JavaScript Object Notation. LDA: latent Dirichlet allocation. LLM: large language
model. LSA: latent semantic analysis. MERS(-CoV): Middle East respiratory syn-
drome (coronavirus). MLP: multilayer perceptron. NLI: natural language inference.
23

NMF: non-negative matrix factorisation. NTD: non-traditional data. PCR: polymerase
chain reaction. PDF: Portable Document Format. PLSA: probabilistic latent semantic
analysis. RAG: retrieval-augmented generation. RVF: Rift Valley fever. SARS: severe
acute respiratory syndrome. SARS-CoV-1, SARS-CoV-2: severe acute respiratory syn-
drome coronavirus 1 and 2. SD: standard deviation. SVM: support vector machine.
TEI: Text Encoding Initiative. TF-IDF: term frequency–inverse document frequency.
URL: uniform resource locator. WHO: World Health Organization. XML: Extensible
Markup Language.
Declarations
Availability of data and materials.EpiLens is available on GitHub [ 41] and PyPI.
The datasets supporting the conclusions of this article are available in the Zenodo
repository at https://doi.org/10.5281/zenodo.22747111. The reproduction package [ 42]
contains the corpus bibliography, the214-paper manually labelled evaluation set, per-
paper and aggregate outputs, and a per-estimate file recording pathogen, parameter,
source author, year and DOI, and links to corpus papers. Its scripts regenerate the
reported tables and figures from the released inputs.
The article PDFs and full-text index cannot be redistributed because of publisher
copyright. The seed study’s extraction sheets [ 32] were shared privately and are not
redistributed. The released source links contain no parameter values.
Competing interests.The authors declare that they have no competing interests.
Funding.This project has received funding from the European Union’s Horizon
Europe research and innovation programme under grant agreement No 101095619
(ESCAPE).
Authors’ contributions.VP designed and implemented EpiLens, built the corpus,
produced the manual labels, ran the experiments, and wrote the manuscript. CC and
DP supervised the study. All authors approved the manuscript.
Acknowledgements.We thank Jack Ward and W. John Edmunds for sharing the
seed study’s early data extraction sheets and approving release of the derived source
links.
Ethics approval and consent to participate.Not applicable.
Consent for publication.Not applicable.
Appendix A System and workflow details
This appendix adds component-level detail to the system summarised in Section 3.
Here we specify the components and backends used at each stage of that pipeline.
A.1 Pipeline components and backends
Each stage of the pipeline is handled by distinct, interchangeable components, which
a researcher can swap to suit their setup.
24

A.1.1 Phase 1: Ingestion and Indexing
This phase transforms a collection of scientific papers into a searchable index. It
involves parsing, chunking, embedding, and storing the content.
Document Parsing.
The workflow begins by parsing PDF documents into structured text. EpiLens uses
GROBID[24] for this task, which is optimized for academic papers and can extract not
only plain text but also rich metadata (title, authors, journal), structured sections
(abstract, methods, results), and a bibliography. For non-PDF files or as a fallback,
theUnstructured [23] library is used. The parsed metadata and structured content
are stored in MongoDB or in memory.
Text Chunking.
To create focused, contextually relevant units for retrieval, the parsed text sections
are segmented into smaller chunks. EpiLens provides many of the standard strategies,
including splitting by paragraph, sentence, or fixed-size overlapping windows. When
not specified otherwise, in our experiments we split by paragraph.
Embedding.
Each text chunk is mapped to a high-dimensional vector using an embedding model.
EpiLens is model-agnostic and supports a range of providers, including open-source
sentence-transformer models from Hugging Face, as well as API-based models from
OpenAI,Google(Gemini),andlocalmodelsservedviaOllama.Thisallowsforflexibility
in balancing performance, computational cost, and privacy.
Vector Storage.
The generated embeddings and their associated metadata (e.g., paper ID, section
title, original text) are stored in a vector database. This database provides the core
functionality for efficient similarity search. EpiLens includes swappable backends for
Qdrant,FAISS, and a simple file-based storage system, enabling deployment in various
environments from local machines to server-based setups.
A.1.2 Phase 2: Retrieval
The retrieval phase calls the vector database to find the most relevant indexed text
chunks in response to a user query.
Query Processing.
EpiLens supports multiple retrieval modes. The primary mode issemantic search,
where a user query is embedded using the same model as the document chunks, and the
vector database is searched for the "nearest" chunk embeddings. For the experiments
in the main text we used Qdrant. Akeyword searchis also available for simple term
matching.
25

Similarity Search and Filtering.
The core retrieval operation is a nearest-neighbor search performed by the vector
database. Results can be filtered by metadata, for instance by restricting the search
to a specific paper ID or a set of paper sections (e.g.,methods). For the experiments
presented in the main text we used cosine similarity as distance metric. The stan-
dard RAG approach uses the user query as reference for finding similar chunks. We
additionally use a template-based approach (see Section A.2).
A.1.3 Phase 3: Generation
In the final phase, an LLM synthesizes a coherent answer from the retrieved evidence.
Prompt Engineering.
The retrieved text chunks are compiled into a carefully structured prompt, along with
the original user query. The prompt explicitly instructs the LLM to formulate its
answer strictly based on the provided evidence and to refrain from using its internal
knowledge. For the workflows discussed in Section A.2, we provide a schema to guide
the model toward generating a JSON output with specific characteristics.
Answer Synthesis.
EpiLens’s Generator component sends the prompt to a local or remote LLM through
a provider client. This allows users to select a model that fits their requirements for
performance, privacy, and cost.
Provenance and Inspectability.
A core principle of EpiLens is to ensure transparency. Every generated answer is
accompanied by a list of evidence items, each tracing back to a specific text chunk
from a source paper and containing the original snippet, the paper ID, and the section.
This makes it possible to inspect the path from query to answer and check the output
against the cited source passages.
A.2 High-Level Analytical Workflows in EpiLens
The structured-workflow contract (a fixed retrievaltemplateplus a validating JSON
schema, implemented with Pydantic [ 28] and applied identically across a corpus) is
defined in Section 3. Here we record the implementation specifics of the two instantia-
tions used in this paper, thePrecision Miner(extraction) and thePaper Classifier
(classification). They reuse the same ingestion, retriever, generator, validation, and
provenance components, but use workflow-specific evidence aggregation, prompts,
schemas, and parsing.
A.2.1 The Precision Miner for Structured Extraction
The Precision Miner is the extraction instantiation of the pattern. It finds and lists
specific, concrete pieces of information in a paper, such as the data sources used, links
to supplementary materials, or key bibliographic references.
26

Methodology.
The miner’s implementation of the pattern is geared towards maximising the discovery
of relevant facts within a document.
•Retrieval phase (multi-query aggregation):The miner is configured with
several queries designed to find relevant passages (for example, “What datasets
were used in this study?” or “Where can supplementary data be found?”). It
searches each paper separately for every query, can restrict candidates by section,
and combines the results into one deduplicated, score-ranked evidence pool.
The multi-query strategy is intended to broaden candidate coverage without
guaranteeing recall.
•Generation phase (the LLM as an extractor):The consolidated evidence
pool is formatted into a prompt and sent to the LLM, which is given the role of
an information extractor: it scans the snippets, identifies every instance of the
target information (for example, dataset names and URLs), and organises them
into a list. The JSON schema requires an object containing this list, where each
item has fields such as name,url, and explanation , with the source sentence
retained as provenance.
Worked outputs.
We ran the find_data_sources variant of the miner over the corpus index on four
corpus papers, one per pathogen family, to show what an extraction workflow returns
rather than to benchmark its accuracy. Table A1 lists the result: for each paper, a
variable-length list of concrete data sources, each with a link or accession where the
paper supplies one. The recovered sources are heterogeneous, from public code and data
repositories (GitHub, Zenodo, GenBank) to surveillance systems, reference datasets
(WHO, NOAA, a national soil survey), and the cited studies behind model parameters.
Other variants ship with the package to find supplementary materials or key references.
The table is limited to paper identifiers, source names, and links or accessions: the
miner also returns source sentences and model explanations for auditing a run, but
these can reproduce publisher text and are not redistributed.
A.2.2 The Paper Classifier for Automated Categorization
The Paper Classifier instantiates this pattern for classification. It is designed to
automatically assign categorical labels to an entire paper along predefined axes, such
as study design, data accessibility, or geographic focus.
Methodology.
The classifier’s implementation is tailored to gather comparative evidence and facilitate
a judgment.
•Retrieval Phase (Evidence Gathering by Category):Unlike the miner, the
classifier’s templates areexemplars: canonical text snippets that represent each
of the possible categories in a classification scheme (e.g., a sentence describing
27

Table A1 A second structured workflow composed with the classifier: the data-source
Precision Miner.
Paper Data sources extracted (name and link or accession where given)
Lauer et al. [43]
COVID-19
incubation•News and public health reports
•Johns Hopkins Center for Systems Science and Engineering
•HopkinsIDD/ncov_incubation GitHub repository (https://github.com/
HopkinsIDD/ncov_incubation)
•Zenodo record 3692048 (https://zenodo.org/record/3692048)
Althaus [44]
EbolaR 0•World Health Organisation (WHO)
•Chowell et al. (study)
Nguku et al. [45]
Rift Valley fever,
Kenya•International Livestock Research Institute (ILRI) GIS site (http://www.ilri.
org/gis)
•Kenya Soil Survey, Report E1, 1982
•LANDSAT images obtained in 1987 by JICA for the Kenya National Water
Master Plan
•National Oceanic and Atmospheric Administration’s (NOAA) Surface
Data Global Summary of the Day file (http://www.cdo.ncdc.noaa.gov/cdo/
dataproduct)
•1999 Kenya Census
•Genbank
Nakoune et al. [46]
Monkeypox, CAR•DDBJ/EMBL/GenBank database
•KP849469, A27L gene (reference sequence)
•Standardized tools for data collection (Ministry of Health and World Health
Organization)
•Consultation registers, health records, and hospitalization forms
•Institut Pasteur de Bangui (IPB) reports of suspected cases
Rights-safe output of the find_data_sources miner on four corpus papers, one per pathogen family.
The released summary contains identifiers, data-source names, and links or accessions. It deliberately
omits source sentences, retrieved text, prompts, and model traces.
open data access is an exemplar for the ’Open Data’ category). Each exemplar
retrieves its nearest chunks from the paper, and each distinct chunk is assigned
to the category whose exemplar it matches best. The khighest-scoring chunks
across all categories are passed to the model grouped by category, as a structured
“dossier” of evidence for the competing labels. A label whose exemplars match
only weakly may receive no chunk.
•Generation Phase (The LLM as a Classifier):This comparative dossier is
presented to the LLM. Its task is not to extract items, but to weigh the evidence
for the competing categories and make a final, reasoned decision about which
label (or labels) best applies to the paper as a whole. The JSON schema requires
one or more valid labels and a textual justification. A confidence score is optional.
Schema validation makes the output structured, while the retained passages make
the decision inspectable.
28

Table A2 Mean token usage per classification case in the
cost proxy.
Output (billable)
Axis Input Reasoning Visible Total
Paper type4172 1552 151 5875
Geography3356 913 169 4437
Data type3213 926 123 4261
Data accessibility3568 1514 155 5237
All axes (mean)3577 1226 149 4952
Measured on the random-chunk ablation with gemini-2.5-flash ,
T=0, and k=15. The targeted workflow uses the same prompts,
schemas, generator, retrieval budget, and repair path but different
evidence selection. Totals include repair invocations.
A.3 Token usage and projected inference cost
This section estimates the token cost of running the EpiLens Paper Classifier over a
corpus. A classification pass has two components: (i) aone-timeembedding stage that
indexes the corpus for retrieval, and (ii) aper-callgeneration stage that classifies each
paper on each axis. We report token usage rather than monetary cost, allowing readers
to convert these figures using the pricing model of their chosen provider. Embedding
costs were estimated from corpus statistics, as described below.
Generation: token usage per call.
Table A2 reports the mean token usage of one EpiLens classification call (one paper,
one axis). Input is dominated by the fifteen retrieved chunks (mean760characters,
approximately190tokens each) together with the task instructions. The visible JSON
output is small (about150tokens), but gemini-2.5-flash is a reasoning model, so
most billable output consists of internal reasoning tokens (about1 ,200tokens). Overall,
a call averages approximately3 ,600input and1 ,400output tokens. About7%of cases
require a second model invocation to repair a non-conforming output. The token counts
reported here already include this overhead. The whole-body ablation used7498input
tokens per call, against3577for the fixed-chunk proxy, a factor of2 .1(Section 4.2.2).
Generation at corpus scale.
EpiLens performs one classification case per paper for each of the four axes, with an
additional invocation when repair is needed. Averaged across all axes, classifying a
single paper requires approximately20 ,000tokens (roughly14 ,000input and5 ,500
output tokens). Scaling the proxy per-case averages to the full corpus of 2,297 papers
(9188classification cases) gives approximately33M input and13M output tokens, or
about46M generation tokens in total (Table A3). The214-paper evaluation corpus
corresponds to approximately4 .3M generation tokens. Alternative backends, particu-
larly non-reasoning models, would substantially reduce output-token usage but may
exhibit different classification accuracy.
29

Table A3 Projected token usage of one EpiLens classification pass over the
four axes.
Scope Cases Gen. in Gen. out Embed (1×)
Evaluation sample (214 papers)856 3.1M1.2M1.3M
Full corpus (2297 papers)9188 32.9M12.6M14.5M
Projected from the ablation-based per-case proxy in Table A2. Generation is incurred
per pass. Corpus embedding is a one-time estimate.
Retrieval: one-time embedding cost.
Generation is only one component of the overall token budget. Retrieval additionally
requires embedding the corpus, although this cost is incurred only once when con-
structing the index. Unlike generation, embedding usage was not metered directly.
Instead, we estimate it from corpus statistics. Using the observed mean of33retained
chunks of about190tokens each gives approximately6300embedded tokens per paper,
or roughly14M tokens for the complete corpus. This is an approximation because
chunking can overlap text and can omit short fragments. The cost is amortised across
subsequent workflows that reuse the same index. The queries, a fixed set of short
exemplar paragraphs (one for each candidate label), are embedded once and reused
for every paper, so their cost is negligible. Overall, corpus indexing contributes an esti-
mated one-time cost of about14M embedding tokens, compared with approximately
46M generation tokens for each complete classification pass (Table A3).
Appendix B Corpus construction pipeline details
This appendix documents the four-stage corpus construction pipeline summarised in
Section 4.1. Exact reconstruction of the document collection remains access-dependent.
Figure B1 illustrates the pipeline.
B.1 Stage 1: LLM-Assisted Review Expansion
The first stage identifies the papers reported as included evidence by each seed source
selected for review expansion. Parsing the complete reference section is insufficient
because a review can cite many papers that were not included in its analysis. We
therefore target the subset of references that are explicitly described as included studies
in the review text, tables, or appendices.
Methodology.
Stage 1 uses LLM-assisted reference extraction followed by rule-based matching to
identify included evidence for each source review. The extraction and matching code
is maintained separately from EpiLens. For each source review selected for expansion,
we execute the following steps.
1.Candidate extraction.An LLM reads the review PDF and extracts spans
that enumerate included studies. This includes narrative “study characteristics”
30

Review-expansion route
Source reviews selected for expansion
Stage 1: LLM-assisted
review expansion
Extract citations of included
papers from source reviews
Stage 2: Metadata
backfilling & DOI resolution
Resolve missing DOIs via bibliographic
APIs to enable reliable retrieval
Stage 3: Semi-automated
document retrieval
Best-effort PDF retrieval via
open-access and publisher sources
Stage 4: Corpus
ingestion & structuring
Parse with GROBID [ 24] and
store for indexing and retrieval
Structured corpus
2,297 records ready for
embedding, indexing, and RAGDirect-source route
Directly cited
source documentsSeed set
138 published sources named in the sheetsExtraction sheets of the seed study (early version)
346 parameter estimates and their sources
Logged exceptions
•Unmatched
references
•Supplement records
•Inaccessible PDFs
Fig. B1 Corpus construction pipeline.Source reviews are expanded to included papers, after
which identifiers are resolved, documents are retrieved, and records are ingested. Directly cited source
documents bypass review expansion and join the flow at identifier resolution. Unmatched references,
supplement-only cases, and inaccessible documents are retained in the audit records.
descriptions and tables listing included trials or observational studies. We use
Gemini’s document understanding [ 47,48], prompting the model to output the
included studies as full bibliographic citations exactly as reported in the review’s
reference list. When the included studies are listed in a supplementary file rather
thanthemaintext,themodelreturnsthesupplement’slabel(e.g.,“Supplementary
Table S1”) and URL instead, and the supplement is processed case by case.
31

2.Bibliographic parsing and normalization.In parallel, the review PDF is
processed with GROBID to extract structured bibliographic records and reference
strings [24].
3.Candidate–bibliography matching.We index bibliography entries by the
normalised surname of the first author and link each LLM-extracted candidate to
one entry. A unique exact surname match is selected directly. When several entries
share that surname, the implementation compares the normalised candidate
with each entry’s title plus raw reference string using RapidFuzz’s fuzz.ratio
similarity [ 49]. A score of at least65is recorded as title-based disambiguation.
Below that threshold, the highest-scoring same-surname entry is retained and
explicitly marked as an ambiguous best guess. If there is no exact surname token,
surnames are compared fuzzily and accepted at a threshold of70. Candidates
with no accepted surname match are marked unmatched and excluded from
downstream stages. This constrains candidates to bibliography entries parsed
independently from the source PDF and records the match score and reason for
audit.
4.Storage.We store the extracted candidate text, the matched bibliographic record,
and the matching confidence in a relational SQLite database. The database also
records run metadata and versioned pipeline settings to support reproducibility
and error analysis.
B.2 Stage 2: Metadata Backfilling and DOI Resolution
Matched references can lack a DOI or carry incomplete metadata. Stage 2 retains a
DOI parsed by GROBID when present and otherwise attempts to resolve one through
Crossref [50]. Directly cited source documents enter the pipeline at this stage.
Methodology.
For each matched reference without a DOI, the implementation:
1.Bibliographic query.Queries Crossref with the title, up to two author strings,
journal, and, where available, a publication-year filter.
2.Candidate scoring.Ranks the returned records by title similarity using
RapidFuzz token_set_ratio , dividing its score by100to place it on a0–1scale.
3.Database update.Writes the DOI of the best candidate when title similarity is
at least0.85. Otherwise it retains the unresolved status or query error.
B.3 Stage 3: Semi-Automated Document Retrieval
Stage 3 attempted to obtain the documents identified in Stage 2. The collection that
can be recovered depends on access to those documents at the time of retrieval.
Methodology.
We ran a DOI-driven PDF harvester over the resolved identifiers. It searched online
for full-text copies of each paper, downloaded candidate PDFs, and logged successes
and failures. For unresolved cases, a fallback script searched DOI and PubMed landing
32

pages for PDF links. We then examined failures and suspect files manually, recovered
papers where possible, and checked that the retained files were valid PDFs. The
archived download records also identify documents already obtained through another
review. The released corpus manifest lists the papers retained for analysis [ 42], with
identifiers that readers can use to attempt retrieval under their own access conditions.
Files of directly cited sources were named by year, first author, and parameter.
B.4 Stage 4: Corpus Ingestion and Structuring
Stage 4 converts the retrieved documents into the structured representation used by
EpiLens for downstream indexing and RAG workflows.
Methodology.
We process each document as follows.
1.Deep parsing.Each PDF is parsed with GROBID into structured TEI XML,
including section boundaries, metadata, and a parsed bibliography [24].
2.Corpus database population.We insert the parsed representation into the
corpus database (MongoDB) as the source of truth for retrieval and provenance.
The frozen manifest contains 2,297 corpus records used by every downstream analysis.
The resulting records are then available for embedding, vector indexing, and query-
dependent retrieval.
Appendix C Labeling protocol
This appendix defines four classification axes: the geography, accessibility, and type of
the data used, and the paper’s main research contribution. The first three describe
thedatasetsa paper uses. The fourth describes thepaperas a whole. Making these
principles explicit is essential both for defining the labels used as ground truth and
for ensuring consistency between these criteria and the instructions provided in the
LLM prompts, which would otherwise default to unknown and potentially inconsistent
standards. Specifically, some documents may plausibly fall into more than one category,
dependingoninterpretation,ormaybesensitivetowhereboundariesaredrawnbetween
categories in the context of our specific classification task. Potentially ambiguous cases
include unclear data availability statements (e.g., claims that data are in the public
domain, which may apply only under specific emergency conditions), the appropriate
geographic labelling of analyses of contagion events originating in one country but
detected in another, and the classification of paper types consistently with the practices
of the relevant scientific community (in this case, epidemiology).
For the three data axes (geography, data accessibility, and data type), classification
applies to the individual datasets within a paper, not to the paper itself. Each dataset
used to obtain results via the described methodology is assigned a label. This excludes
datasets that, e.g., are only mentioned in a background section. The final output
for each axis is the union of these per-dataset labels, so duplicates are automatically
collapsed: if three datasets are assigned labels A,A, andB, the resulting set is {A,B}.
33

The fourth axis, paper type, is instead assigned at the level of the whole paper, as
described in its own section below.
C.1 Label set
We assign labels along four axes, the first three abbreviated GEO, DAVAIL, and DTYPE
below:
1.Geography.A set-valued label capturing the continents associated with the epi-
demiological evidence used in the paper, or whether this information is irrelevant,
e.g., in the case of laboratory settings.
2.Data accessibility.A set-valued label recording the access mechanisms stated
for the data used in the paper, or whether printed values permit limited in-paper
reconstruction.
3.Data type.A multi-valued label capturing whether the contribution uses data
andthenatureofthedatausedforthecontribution,i.e.,whetheritusestraditional
data, non-traditional data [1], or synthetic data.
4.Paper type.One primary label used in evaluation, with optional secondary
labels retained in the stored output, capturing the paper’s research-contribution
type with respect to epidemiological parameter estimation.
Labels are assigned exclusively based on information provided within the document,
disregarding external knowledge or verification. For instance, if a paper states that
its data are available in an online repository, we use thatstatementregardless of
whether the repository is still accessible. This is motivated by the fact that we focus on
extraction fidelity, not on, for example, verifying data accessibility. The only exception
is the mapping of geographic locations to continents. Here we allow the model to map
a location to its geographic label using its internal knowledge, e.g., mapping a nation
to its continent, providing guidance only for the ambiguous cases discussed below.
Geography, data type, and paper type include an UNCLEAR fallback for insufficient
evidence, distinct fromIRRELEVANT(geography) andOTHER(paper type).
C.2 Geographic labelling
This pipeline is intended as a first-stage, high-recall filter for identifying subsets of
papers that merit subsequent, more specialized analyses. Accordingly, we adopt an
intentionally inclusive definition of geographic information: we include all locations
that areexplicitly tiedto the epidemiological evidence used in the paper, even when
multiple plausible “geographic perspectives” coexist (e.g., location of exposure vs.
location of detection/ascertainment). For instance, when infections are acquired in
one country but cases are detected, diagnosed, or reported in another, we includeboth
locations in the multi-label GEO output. We do not include a location mentioned only
incidentally: if a paper analyses an outbreak in continent A and merely notes similar
patterns in continent B, the correct label is A alone. More granular labelling (e.g.,
separating exposure from ascertainment into distinct labels) is feasible, but is beyond
our current aim.
34

There are, however, cases in which location information is considered irrelevant. If
the analysis is conducted entirely in controlled laboratory settings such that geographic
or ecological variation is not meaningful (e.g., laboratory-bred animals or strictly
controlled experimental conditions), the GEO axis is marked as IRRELEVANT . The same
label applies when the paper relies on purely synthetic or otherwise non-geographic
data that is not meaningfully tied to any real-world location. By contrast, wildlife,
farm, market, household, and other field-like settings are treated as geographically
meaningful and are labeled using the corresponding locations stated by the authors.
Consistent with our general principle of relying only on stated information, we
do not infer geographic origin from strain identifiers, sample naming conventions, or
external domain knowledge unless the paper explicitly states the geographic origin in
the text.
The continent label set is Africa, Asia, Europe, North America, South America,
and Oceania (together with IRRELEVANT andUNCLEAR). Some countries straddle two
continents and are intrinsically ambiguous under informal conventions. For these
transcontinental cases we fix a single, reproducible country-to-continent assignment
using the United Nations Statistics Division M49 standard (“Standard Country or
Area Codes for Statistical Use”, Series M, No. 49) [ 37], rather than ad hoc geopolitical
or cultural criteria. For transcontinental countries where the pathogens in scope,
those with pandemic potential, have been reported, the annotation prompt uses fixed
assignments: Russia to Europe, Egypt to Africa, and Turkey, Kazakhstan, Azerbaijan,
Georgia, Armenia, and Cyprus to Asia.
Other sources of ambiguity, such as no-longer-existing political entities (e.g., USSR,
Yugoslavia) or politically disputed/recognition-dependent territories (e.g., Kosovo), are
largely avoided in this setting because our evaluation targets continent-level labels. By
collapsing geographic mentions onto continents, we anchor the GEO axis to relatively
stable, predominantly geographic reference entities, reducing sensitivity to changes in
statehood, naming, or recognition. Finally, for papers mentioning a global scope, we
list all continents.
C.3 Data accessibility labelling
DAVAIL records what the paper states about access to each dataset and whether the
paper prints enough values to meet the REPORTED re-aggregation rule. It measures
reported accessibility and in-paper reconstructability, not verified access or general
dataset reusability.
OPEN.The data are publicly available via a persistent identifier (e.g., DOI, accession
number), a stable link to a repository, or by being fully released within the
paper package (e.g., full line-lists or complete contingency tables) in a form
sufficient for a third party to reconstruct the dataset for new analyses. Mentions
of supplementary material only qualify if they contain the specific, extractable
data used for the work (the link/identifier must point to the underlying data
supporting the results, not just be a “supplementary information exists” pointer).
35

REPORTED. The dataset is not fully released, but the paper provides granular,
extractable data values (e.g., detailed frequency tables, stratified counts, or par-
tial time-series) that allow a third party to independently recompute at least one
primary summary statistic used in the analysis (e.g., a frequency, proportion, or
rate)from the reported values alone, without requiring access to unreported data
or relying on the authors’ interpretive conclusions. This label applies based on
the richness of the in-paper reporting, regardless of whether the data source is
internal or a third party. It does not imply access to the underlying dataset or
the ability to reproduce the full analysis.
REFERENCED. The data are attributed to aspecifically identifiable third-party source
(e.g., public health authorities, institutional databases, emergency dashboards,
named public sources, other papers), but the paper provides neither a stable
retrieval path for the source data nor sufficient detail to re-aggregate the data
or re-use them as inputs for other analyses without relying on the author’s
interpretation (i.e., fails the REPORTED re-aggregation criterion). This data-thin
reporting (limited to derived outputs like effect sizes, coefficients, or fitted model
curves) prevents independent verification. Vague attributions (e.g., "government
data") do not qualify (lean towardsNOT_STATED).
AVAILABLE_UPON_REQUEST. The paper explicitly provides a request-based mechanism,
such as contacting the author, a data access committee, or a specific institutional
process.
CLOSED. The paper explicitly states that the data cannot be shared or are restricted
due to legal, ethical, or licensing constraints, without providing an actionable
access mechanism.
NOT_STATED. The paper provides no data access statement, no repository link, no
request mechanism, and no identifiable third-party attribution, while also failing
to provide extractable values sufficient for REPORTED . Statements that data were
collected/generated, without an access mechanism and without re-aggregatable
values, should be labeledNOT_STATED.
Precedence is applied within each dataset, after which the paper-level output is the
union of the per-dataset labels. For one dataset, use OPENwhen a qualifying public
route is stated. Otherwise use AVAILABLE_UPON_REQUEST when an actionable request
route exists, and CLOSEDfor an explicit restriction without such a route. If none
applies, distinguish REPORTED from REFERENCED by the re-aggregation criterion, and
useNOT_STATEDwhen neither applies.
C.3.1 Operational scope
The protocol is descriptive rather than a compliance assessment: it classifies what
a paper states or prints, using categories adapted to the heterogeneous availability
statements encountered across journals and communities. Two rules make the categories
operational for both annotation and prompting: OPENrequires a stable retrieval route
or a reconstructable in-paper release, and REPORTED requires enough printed values to
recompute at least one primary summary statistic.
36

C.4 Data type labelling
TheDTYPElabels identify the nature of the data used to support a paper’s contribution.
DTYPEis set-valued because a paper can use several data types. The operational
NO_EMPIRICAL_DATA output is exclusive and is used only when the contribution relies
on neither empirical nor simulated data. The traditional, health, mobility, economic,
and sentiment categories build on the #Data4COVID19 Review [ 1].SYNTHETIC and
NO_EMPIRICAL_DATA are extensions introduced for this corpus. In a nutshell, NTD
refers to information that is digitally captured, mediated, or observed through new
instrumentation mechanisms, which is often privately held and repurposed for uses
unrelated to its initial collection.
TRADITIONAL when a paper relies on standard epidemiological and public-health data
sources collected through established surveillance infrastructure, regardless of
digital vs analog collection methods. Examples include daily or weekly case counts
from a national surveillance system, hospital admissions and deaths, and PCR-
or antigen-confirmed test line lists.
NON_TRADITIONAL_HEALTH novel or non-standard health data sources that emerged
outside traditional public-health surveillance systems, such as: (i) digital patient
data from large-scale electronic-health-record repositories, (ii) platform-facilitated
self-reports and symptom proxies collected through apps or web platforms, (iii)
wastewater surveillance as a community-level proxy for spread, and (iv) wearable
or biometric signals.
NON_TRADITIONAL_MOBILITY when the paper uses mobility data, understood as data
describing the location of a device relative to people/landmarks in the physical
world or to other devices. This includes mobility signals derived from telecom-
munications (e.g., call detail records), location data from software development
kits embedded in smartphone apps, GPS and vehicle traces, and proximity tech-
nologies (e.g., Bluetooth), when used as epidemiological proxies (e.g., population
movement patterns, contact opportunities).
NON_TRADITIONAL_SENTIMENT when the paper uses NTD intended to capture how
individuals or groups perceive and interpret epidemic-related developments and
interventions. This category includes social-media, crowdsourced, app-based, and
online-panel data used to track behaviour or perceptions rather than clinical
infections. Conventional epidemiological surveys remainTRADITIONAL.
NON_TRADITIONAL_ECONOMIC when a paper uses digitally captured or platform-
mediated data describing the economic activity of individuals, groups, or
organisations. This category includes credit and debit card transaction data, open
procurement/contracting data, and supply chain data.
SYNTHETIC when simulated data is presented as evidence and support for claims. We
donotassign SYNTHETIC when simulations are used solely to illustrate model fit
or calibration. Examples include simulation-based evaluation of policy scenarios,
model-based counterfactual projections, and sensitivity analyses where simulated
outcomes support the main claim (versus a purely illustrative fit plot, for which
we do not assign the label). The distinction is based on whether what is produced
there is considered data at all, rather than on whether it is synthetic.
37

NO_EMPIRICAL_DATA when the paper relies on neither an empirical dataset nor simu-
lateddataasevidence.Examplesincludeaconceptualcommentaryoncontainment
strategies without data and a theoretical derivation of a model without fitting or
empirical evaluation.
C.5 Paper type labelling
PAPER_TYPE classifies each document by the kind of scientific contribution it makes
to epidemiological parameter estimation. Epidemiological research often integrates
multiple tasks (collecting data, inferring parameters, and making forecasts) within a
single workflow. While usually focused around a main contribution, papers combine
different types of contributions. We address this by distinguishing between a paper’s
primary contribution(the main scientific claim or deliverable) and anysecondary
contributions(supporting tasks that enable or accompany the primary contribution).
We therefore assign one primary label and may assign additional secondary labels as
needed. The same label set is used for both primary and secondary contributions.
EMPIRICAL. The contribution is based onoriginal empirical data collection, where
claims rely on direct measurement from a clearly defined study population or sam-
ple. Examples: observational cohort studies, case–control studies, cross-sectional
surveys, randomized trials, surveillance reports, or data papers describing new
datasets.
INFERENCE. The contribution involvesestimating parameters or reconstructing unob-
served epidemic processesby fitting explicit statistical or mechanistic models to
observed data. Results depend on model structure and assumptions, not on direct
measurement alone. Examples: papers estimating R0or generation time from case
data, reconstructing transmission chains, performing phylodynamic inference, or
estimating seroconversion rates from serological surveys.
FORECAST. The contribution isprediction or scenario analysis: forward projections of
epidemic trajectories, comparisons of intervention scenarios, or counterfactual
simulations. The outputs are projected future outcomes or policy comparisons
under stated assumptions. Examples: epidemic projections for policy planning,
intervention scenario comparisons, or real-time forecasting submissions.
SYNTHESIS. The contribution is asystematic synthesis of existing evidenceusing
predefined search, selection, and appraisal criteria (e.g., meta-analyses or
PRISMA-compliant systematic reviews). Examples: quantitative meta-analyses
pooling estimates across studies, or systematic reviews synthesizing evidence on
transmission parameters or intervention effectiveness.
METHODOLOGICAL. The contribution is thedevelopment, evaluation, or comparison of
methods: new estimators, inference frameworks, identifiability analyses, software
tools, or theoretical results about epidemic processes. Examples: papers introduc-
ing novel inference algorithms, proving identifiability results, developing software
packages, or deriving analytical results on epidemic thresholds.
COMMENTARY. The contribution isdiscussion or interpretationwithout presenting new
empirical estimates, model-based inference, or systematic evidence synthesis.
38

Examples include narrative literature reviews, perspective pieces on research
priorities, and policy commentaries.
We use OTHERwhen a paper does not fit any of the categories above. In our analysis
we focus on the primary contribution, making this a single-label classification task.
Secondary contributions are recorded but not used in the evaluation.
Appendix D Supplementary results and diagnostics
This appendix provides the detailed quantitative results that support Section 4. All
metrics are computed on the fixed manually annotated evaluation set of214papers.
Unless stated otherwise, values are mean±SD acrossR=5repeated runs.
D.1 Aggregate metrics for served configurations
TableD1reportsthefullsetofagreementmetricsforeachtaskandservedconfiguration,
including the open-weight Nemotron-120B model atT=0alongside the two Gemini
models. The local Qwen configurations are reported separately in Section D.3. These
results complement the Jaccard and stability summaries in Tables 1 and 2. For the
single-label paper-type task, micro-averaged scores coincide with subset accuracy. The
macro-averaged scores are lower than the corresponding micro-averaged scores, showing
that aggregate performance is sensitive to uneven class support. Table D2 shows which
labels account for the difference.
Table D1:Aggregate agreement metrics by task and configuration.
gemini-2.5-pro gemini-2.5-flash Nemotron
MetricT=0.0T=1.0T=0.0T=1.0T=0.0
Geography
Jaccard0.9283 ±0.0044 0.9248±0.0049 0.8748±0.0069 0.8709±0.0044 0.886±0.011
Micro precision0.8816 ±0.0091 0.8845±0.0082 0.794±0.011 0.780±0.016 0.816±0.014
Micro recall0.9457 ±0.0027 0.9434±0.0021 0.9279±0.0044 0.9240±0.0044 0.930±0.012
Micro F 1 0.9125±0.0058 0.9130±0.0051 0.8559±0.0073 0.8461±0.0099 0.870±0.012
Macro precision0.810 ±0.013 0.812±0.014 0.715±0.013 0.688±0.024 0.754±0.096
Macro recall0.8742 ±0.0064 0.8747±0.0052 0.8013±0.0099 0.792±0.010 0.782±0.033
Macro F 1 0.8383±0.0089 0.839±0.010 0.7430±0.0097 0.724±0.017 0.717±0.046
Subset accuracy0.9028 ±0.0069 0.8981±0.0069 0.8206±0.0097 0.8196±0.0026 0.854±0.014
Data Accessibility
Jaccard0.802 ±0.023 0.775±0.013 0.670±0.022 0.684±0.016 0.660±0.014
Micro precision0.834 ±0.019 0.810±0.013 0.687±0.018 0.697±0.013 0.757±0.016
Micro recall0.819 ±0.021 0.7951±0.0093 0.745±0.020 0.763±0.024 0.646±0.011
Micro F 1 0.826±0.020 0.802±0.011 0.715±0.019 0.728±0.016 0.697±0.013
Macro precision0.775 ±0.030 0.743±0.021 0.653±0.013 0.633±0.016 0.746±0.070
Macro recall0.803 ±0.013 0.7783±0.0100 0.7875±0.0099 0.773±0.037 0.574±0.037
Macro F 1 0.779±0.020 0.7508±0.0100 0.698±0.013 0.678±0.012 0.622±0.043
Subset accuracy0.709 ±0.027 0.679±0.013 0.523±0.022 0.526±0.019 0.532±0.012
Data Type
Jaccard0.950 ±0.018 0.9414±0.0054 0.9158±0.0063 0.9132±0.0091 0.8767±0.0086
(continued on next page)
39

gemini-2.5-pro gemini-2.5-flash Nemotron
MetricT=0.0T=1.0T=0.0T=1.0T=0.0
Micro precision0.954 ±0.013 0.9458±0.0046 0.9081±0.0065 0.9109±0.0094 0.933±0.013
Micro recall0.961 ±0.018 0.9547±0.0071 0.9492±0.0048 0.9414±0.0100 0.8422±0.0059
Micro F 1 0.958±0.015 0.9502±0.0048 0.9282±0.0054 0.9259±0.0073 0.8851±0.0089
Macro precision0.719 ±0.062 0.726±0.078 0.698±0.018 0.65±0.11 0.70±0.12
Macro recall0.72 ±0.14 0.682±0.071 0.721±0.035 0.616±0.089 0.475±0.089
Macro F 1 0.705±0.097 0.684±0.065 0.652±0.033 0.592±0.081 0.50±0.10
Subset accuracy0.924 ±0.027 0.9140±0.0078 0.8738±0.0081 0.870±0.013 0.8019±0.0097
Paper Type
Jaccard0.9028 ±0.0084 0.901±0.011 0.8766±0.0097 0.880±0.019 0.848±0.011
Micro precision0.9028 ±0.0084 0.901±0.011 0.8766±0.0097 0.880±0.019 0.848±0.011
Micro recall0.9028 ±0.0084 0.901±0.011 0.8766±0.0097 0.880±0.019 0.848±0.011
Micro F 1 0.9028±0.0084 0.901±0.011 0.8766±0.0097 0.880±0.019 0.848±0.011
Macro precision0.926 ±0.013 0.900±0.025 0.679±0.025 0.740±0.054 0.71±0.11
Macro recall0.922 ±0.015 0.902±0.042 0.702±0.018 0.747±0.090 0.600±0.077
Macro F 1 0.920±0.013 0.890±0.020 0.677±0.019 0.732±0.068 0.631±0.086
Subset accuracy0.9028 ±0.0084 0.901±0.011 0.8766±0.0097 0.880±0.019 0.848±0.011
Computed on the fixed evaluation set (214 papers). Values are mean with the standard deviation as a
subscript, across R=5repeated runs. The open-weightNemotronbackend (NVIDIA Nemotron-120B)
is reported atT=0alongside the two Gemini backends.
D.2 Per-label F 1
Table D2 gives the F 1of each label at T=0, with the number of papers that carry it. On
data type, the Jaccard similarity is dominated byTraditional(200of the214papers),
which every backend labels with an F 1of at least0 .955, whereasSyntheticranges from
0.437(Nemotron) to0 .945(Pro), and the non-traditional categories, carried by one
to three papers each, cannot be assessed reliably. On data accessibility, no backend
exceeds0.58onNot stated.
D.3 A fully local deployment
The Nemotron results show that the workflow does not depend on a proprietary model,
but Nemotron is still accessed through a third-party serving API. We therefore test a
stronger deployment claim: whether the classification workflow can run entirely on
hardware controlled by the analyst once the corpus has been assembled. We run the
Paper Classifier with qwen2.5:32b-instruct , served locally through Ollama [ 51], in
two configurations.
In thepartially localconfiguration, generation runs locally while retrieval reuses
the Gemini-embedded index used by the served models. This holds the retrieval index
fixed. Differences from the served configuration combine the generator, provider, and
serving-stack effects. In thefully localconfiguration, the same generator instead uses an
index built locally with BAAI/bge-m3 , so parsing, embedding, retrieval, and generation
in the classification workflow require no remote service. Because the two indices use
different encoders, they can retrieve different evidence, so we treat their comparison
primarily as a comparison between deployment configurations rather than as a direct
measurement of embedding quality.
40

Table D2 Per-label F 1of EpiLens against the manual labels.
LabelnPro Flash Nemotron Qwen (local)
Geography
Asia 850.9714 ±0.0039 0.9382±0.0043 0.9434±0.0057 0.927±0.000
Africa 670.9592 ±0.0082 0.9362±0.0047 0.9460±0.0080 0.913±0.000
Europe 420.8825 ±0.0060 0.8049±0.0092 0.876±0.012 0.812±0.000
North America 240.7783 ±0.0083 0.683±0.024 0.709±0.033 0.731±0.000
Irrelevant 150.864 ±0.026 0.636±0.065 0.30±0.11 0.640±0.000
Oceania 110.810 ±0.040 0.853±0.055 0.819±0.045 0.692±0.000
South America 110.870 ±0.000 0.758±0.025 0.837±0.031 0.741±0.000
Unclear 30.571 ±0.000 0.333±0.000 0.30±0.27 0.000±0.000
Data accessibility
Reported 1270.8930 ±0.0087 0.777±0.023 0.807±0.013 0.735±0.000
Referenced 740.805 ±0.035 0.713±0.026 0.594±0.019 0.568±0.000
Open 480.838 ±0.032 0.760±0.013 0.641±0.021 0.623±0.000
Not stated 190.574 ±0.028 0.411±0.023 0.463±0.031 0.176±0.000
Upon request 100.835 ±0.046 0.691±0.032 0.880±0.021 0.318±0.000
Closed 50.727 ±0.000 0.833±0.000 0.35±0.27 0.167±0.000
Data type
Traditional 2000.9852 ±0.0055 0.9732±0.0024 0.9717±0.0033 0.955±0.000
Synthetic 370.945 ±0.026 0.9134±0.0071 0.437±0.020 0.689±0.000
No empirical data 110.73 ±0.10 0.37±0.14 0.304±0.093 0.154±0.000
Non-trad. mobility 30.66 ±0.20 0.462±0.076 0.521±0.070 0.000±0.000
Non-trad. health 20.52 ±0.16 0.400±0.000 0.53±0.30 0.000±0.000
Non-trad. sentiment 20.16 ±0.36 0.440±0.055 0.53±0.13 0.571±0.000
Non-trad. economic 10.93 ±0.15 1.000±0.000 0.20±0.45 0.000±0.000
Paper type
Empirical 1270.9247 ±0.0071 0.9299±0.0093 0.9260±0.0092 0.891±0.000
Inference 560.8521 ±0.0088 0.851±0.011 0.807±0.011 0.673±0.000
Forecast 120.921 ±0.028 0.829±0.035 0.554±0.057 0.500±0.000
Methodological 110.897 ±0.023 0.666±0.045 0.40±0.16 0.400±0.000
Commentary 60.927 ±0.041 0.625±0.057 0.71±0.14 0.727±0.000
Synthesis 21.000 ±0.000 0.840±0.089 0.73±0.19 0.500±0.000
F1of each label against the manual labels on the214evaluation papers at T=0:
the mean over five runs, with the standard deviation as a subscript. nis the num-
ber of papers that carry the label in the manual labels. Labels that never occur
there are not shown.ProandFlashare gemini-2.5-pro and gemini-2.5-flash ,
Nemotronis nemotron-3-super-120b-a12b , andQwen (local)is the fully local
qwen2.5:32b-instruct configuration, whose repeats are identical. For labels carried by
few papers, F 1rests on a handful of decisions.
All other components are held fixed: the same GROBID parses, dense retrieval
with workflow_top_k= 15, prompts, schema-constrained decoding, and R=5repeats
over the same214papers. No prompt reached Qwen’s context window of32 ,768tokens:
the recorded prompts are at most12 ,090tokens long on the evaluation set,18 ,464
in the corpus-scale run of Section D.3.4, and22 ,771in the whole-body condition of
Section 4.2.2. At32B parameters, Qwen is also smaller in total size than Nemotron
(120B), although Nemotron, a mixture-of-experts model, activates only about12B
41

parameters per token. Table D3 compares its Jaccard scores at T=0with the served
models, Table D4 addsT=1, and Table D5 reports the full metric profile.
Table D3 A fully local, fully open-weight deployment of the same workflow.
FloorQwen2.5-32B(self-hosted)Served backends
TaskMajority fully local partially local Nemotronflash pro
Geography0.353 0.869±0.000 0.854±0.000 0.886±0.011 0.8748±0.0069 0.9283±0.0044
Data access.0.472 0.539±0.000 0.563±0.000 0.660±0.014 0.670±0.022 0.802±0.023
Data type0.843 0.861±0.000 0.875±0.000 0.8767±0.0086 0.9158±0.0063 0.950±0.018
Paper type0.593 0.794±0.000 0.785±0.000 0.848±0.011 0.8766±0.0097 0.9028±0.0084
Sample-averaged Jaccard on the214-paper evaluation set, as mean with the standard deviation
across R=5repeated runs shown as a subscript.Every column is at T=0. Table D4 and
Table D5 report the same configurations at T=1as well. BothQwen2.5-32Bcolumns generate with
qwen2.5:32b-instruct run locally through Ollama on our own hardware, and differ only in how the
retrieval index is embedded:fully localembeds with BAAI/bge-m3 on the same machine, so no stage
of the pipeline contacts a remote service, whereaspartially localreuses the Gemini-embedded index
shared with the served-backend runs. Everything else is held fixed across all columns (dense retrieval,
workflow_top_k= 15, grobid parses, identical prompts, schema-constrained decoding). TheFloor
column is the deterministic majority-class baseline.Nemotron(NVIDIA Nemotron-120B) and the
two Gemini backends, flashand pro, are reproduced from Table 1 for reference. Both self-hosted
configurations returnidenticalpredictions on all four axes across all five repeats, so their SD is
exactly zero (Section D.3).
Table D4 Accuracy of the self-hosted configurations.
Floor Qwen2.5-32B(open weight, self-hosted generation)
fully local(bge-m3)partially local(Gemini index)
TaskMajority T=0T=1 T=0T=1
Geography0.353 0.869±0.000 0.869±0.014 0.854±0.000 0.8677±0.0062
Data access.0.472 0.539±0.000 0.516±0.027 0.563±0.000 0.542±0.021
Data type0.843 0.861±0.000 0.8661±0.0061 0.875±0.000 0.8785±0.0059
Paper type0.593 0.794±0.000 0.775±0.013 0.785±0.000 0.779±0.012
Sample-averaged Jaccard as mean with the standard deviation across R=5repeated
runs shown as a subscript, on the214-paper evaluation set. Both blocks generate with
qwen2.5:32b-instruct run locally through Ollama on our own hardware.Fully local
additionally embeds the retrieval index locally with BAAI/bge-m3 , so no stage of the
pipeline contacts a remote service.Partially localreuses the Gemini-embedded index
built for the served-backend runs, and so remains partially remote. TheFloorcolumn
is the deterministic majority-class baseline. All T=0standard deviations are exactly
zero: the five repeats return identical label sets for every paper (Section D.3). Table 1
reports the served backends on the same tasks.
42

D.3.1 Task-level performance
The effect of moving to the fully local stack is strongly task dependent rather than
uniform. Performance remains close to Flash ongeography(0 .869versus0 .875) and
remains high ondata type(0 .861versus0 .916), although the majority baseline there
is already0 .843. The largest gap occurs ondata accessibility, where the fully local
configuration reaches0 .539, compared with0 .670for Flash and0 .802for Pro. This is
consistent with the earlier results identifying accessibility as the axis that places the
greatest demands on both locating and interpreting evidence.
The comparison with the supervised grid is similarly differentiated (Table 3 and
Section E.5). Without using training labels, the fully local configuration exceeds the
grid’s best per-task score ongeography(0 .869versus0 .771), matches it onpaper type
(0.794versus0 .785), and remains below it ondata type(0 .861versus0 .898) anddata
accessibility(0 .539versus0 .619). These maxima are an oracle ceiling assembled from
different supervised configurations by task, not one deployable model. Thus, a mid-
sized local model retains much of the classification performance of the served workflow,
with the largest cost concentrated on the axis that is already the most difficult in the
main evaluation.
D.3.2 Effect of local embedding
Moving the remaining embedding step on-device has only a small effect on downstream
classification in this experiment. Across the four axes, the partially and fully local
configurations differ in Jaccard by at most0 .024, and the direction is not consistently
negative. The fully local configuration scores higher ongeography(0 .869versus0 .854)
andpaper type(0 .794versus0 .785), and lower ondata type(0 .861versus0 .875) and
data accessibility(0 .539versus0 .563). In these experiments, replacing the Gemini-
embedded index with a locally constructed one therefore contributes little to the
downstream performance difference between the local and served configurations. This
does not establish that the two embedding models have equivalent retrieval quality,
but it shows that removing the remote embedding dependency has little effect on the
classification metrics considered here.
D.3.3 Run-to-run repeatability
Both self-hosted configurations were exactly repeatable at T=0in our experiments.
Across five independent executions, they returned identical label sets for every paper
on all four axes, giving exactly zero standard deviation in Tables D3 and D5. For
the fully local configuration, the repeats also agree on the generated text, retrieved
evidence, and recorded prompt and completion token counts.
This differs from the API-served configurations, all of which retain non-zero run-
to-run variation at T=0(Table D1 and Section D.4). Exact repeatability disappears
when the temperature is raised to T=1, without a consistent gain in accuracy: data
type improves slightly and geography is essentially unchanged, while paper type and
data accessibility decline. The repeatability observed at T=0should nevertheless
be understood as a property of the specific deployment tested here rather than as
a general guarantee of self-hosting. Kernel choice, numerical precision, parallelism,
43

and tensor reductions can all introduce variation even under greedy decoding. What
the experiment establishes is that controlling sampling and serving end to end was
sufficient to obtain exact repeatability on this hardware and software stack, which is
useful for an audit trail that may need to be regenerated later.
D.3.4 Corpus-scale classification
Performance on the214-paper evaluation set establishes what the local deployment
can recover where manual labels are available. A separate deployment question is how
its aggregate label distributions compare when the analysis is scaled to the complete
corpus. We therefore reran the three axes used for corpus-scale characterisation over
all 2,297 papers with the fully local configuration at T=0. Once the corpus had been
assembled, no stage of this classification pass contacted a remote service.
Figures D1 and D2 compare the served and fully local distributions. The served
distributions come from one corpus run. Across five runs on the214-paper evaluation
set, the largest spread in any label’s share was2 .3percentage points (1 .4for paper
type), although about one paper in ten changed its paper-type label.
0 50OtherUnclearSynthesisCommentaryForecastMethodologicalInferenceEmpirical
0.5%  (n=11)1.1%  (n=25)1.9%  (n=44)2.8%  (n=65)3.9%  (n=90)7.1%  (n=162)25.0%  (n=574)57.7%  (n=1326)Paper type
0 50UnclearIrrelevantOceaniaSouth AmericaNorth AmericaEuropeAfricaAsia
2.2%  (n=51)3.3%  (n=76)8.4%  (n=192)9.9%  (n=228)18.9%  (n=433)26.3%  (n=604)37.9%  (n=870)46.1%  (n=1059)Geographical scope
0 100EconomicSentimentUnclearHealthNo dataMobilitySyntheticTraditional
0.5%  (n=12)0.7%  (n=16)1.2%  (n=27)1.7%  (n=38)1.8%  (n=41)3.4%  (n=78)18.5%  (n=425)96.3%  (n=2212)Data type
Share of papers (%)
Fig. D1 Label distributions across all 2,297 corpus papers.Panels show paper type, geography,
and data type as labelled by gemini-2.5-flash atT=0. Bar lengths give percentages and the adjacent
numbers give paper counts. Paper type is single-label. Geography and data type are multi-label, so
their percentages can sum to more than100%. Mobility, health, sentiment, and economic data are the
non-traditional categories.
The two deployments produce similar aggregate distributions forgeographyand
data type. This comparison of marginals does not establish paper-level agreement.
Every regional share differs by at most7percentage points and most differ by less
than3, while the dominantTraditionalshare differs by less than2points (96 .3%
served versus94 .5%local). The main divergence ispaper type: the served model assigns
Empiricalto57.7%of the corpus, compared with69.4%under the local model.
There are no manual labels for the full corpus, so this disagreement cannot be
resolved directly at corpus scale. The labelled evaluation sample nevertheless provides
an informative diagnostic because it is drawn from the same corpus. On that sample,
the manual-label share ofEmpiricalpapers is59 .3%, compared with57 .5%for the
served model and68 .7%for the local model. The corpus-scale difference therefore
has the same direction and approximately the same magnitude as the difference
44

Table D5 Full agreement-metric profile of the self-hosted
open-weight backend.
Qwen2.5-32B(open weight, self-hosted generation)
fully local(bge-m3)partially local(Gemini index)
MetricT=0.0T=1.0T=0.0T=1.0
Geography
Jaccard0.869 ±0.000 0.869±0.014 0.854±0.000 0.8677±0.0062
Micro precision0.804 ±0.000 0.8222±0.0075 0.785±0.000 0.8214±0.0066
Micro recall0.891 ±0.000 0.889±0.014 0.876±0.000 0.881±0.013
Micro F 1 0.846±0.000 0.8543±0.0086 0.828±0.000 0.8500±0.0096
Macro precision0.651 ±0.000 0.6762±0.0089 0.628±0.000 0.678±0.023
Macro recall0.736 ±0.000 0.731±0.026 0.708±0.000 0.714±0.023
Macro F 1 0.682±0.000 0.694±0.016 0.657±0.000 0.688±0.022
Subset accuracy0.827 ±0.000 0.833±0.013 0.813±0.000 0.8299±0.0026
Data access.
Jaccard0.539 ±0.000 0.516±0.027 0.563±0.000 0.542±0.021
Micro precision0.574 ±0.000 0.552±0.023 0.594±0.000 0.582±0.021
Micro recall0.629 ±0.000 0.611±0.019 0.661±0.000 0.627±0.017
Micro F 1 0.600±0.000 0.580±0.020 0.625±0.000 0.603±0.018
Macro precision0.435 ±0.000 0.438±0.016 0.469±0.000 0.472±0.032
Macro recall0.491 ±0.000 0.512±0.011 0.495±0.000 0.498±0.032
Macro F 1 0.431±0.000 0.442±0.016 0.453±0.000 0.457±0.025
Subset accuracy0.364 ±0.000 0.328±0.044 0.383±0.000 0.355±0.029
Data type
Jaccard0.861 ±0.000 0.8661±0.0061 0.875±0.000 0.8785±0.0059
Micro precision0.897 ±0.000 0.8971±0.0086 0.910±0.000 0.9125±0.0024
Micro recall0.848 ±0.000 0.8578±0.0081 0.867±0.000 0.871±0.011
Micro F 1 0.871±0.000 0.8770±0.0070 0.888±0.000 0.8913±0.0050
Macro precision0.389 ±0.000 0.479±0.034 0.501±0.000 0.404±0.045
Macro recall0.375 ±0.000 0.464±0.060 0.429±0.000 0.401±0.037
Macro F 1 0.338±0.000 0.390±0.035 0.434±0.000 0.380±0.025
Subset accuracy0.790 ±0.000 0.7925±0.0085 0.813±0.000 0.813±0.011
Paper type
Jaccard0.794 ±0.000 0.775±0.013 0.785±0.000 0.779±0.012
Micro precision0.794 ±0.000 0.775±0.013 0.785±0.000 0.779±0.012
Micro recall0.794 ±0.000 0.775±0.013 0.785±0.000 0.779±0.012
Micro F 1 0.794±0.000 0.775±0.013 0.785±0.000 0.779±0.012
Macro precision0.740 ±0.000 0.631±0.033 0.686±0.000 0.628±0.031
Macro recall0.643 ±0.000 0.514±0.055 0.594±0.000 0.501±0.017
Macro F 1 0.615±0.000 0.520±0.043 0.595±0.000 0.531±0.016
Subset accuracy0.794 ±0.000 0.775±0.013 0.785±0.000 0.779±0.012
Mean with the standard deviation across R=5repeated runs shown as a sub-
script, on the fixed214-paper evaluation set. All four columns generate with
qwen2.5:32b-instruct run locally through Ollama under the identical workflow,
and use schema-constrained decoding. Thefully localpair embeds the retrieval
index with BAAI/bge-m3 on the same machine, so the pipeline makes no remote
call at any stage. Thepartially localpair reuses the Gemini-embedded index
shared with the served-backend runs. Within each pair the columns differ only
in sampling temperature. For the single-label paper-type task, micro-averaged
scores coincide with subset accuracy. All T=0standard deviations are exactly
zero: the five repeats return identical label sets for every paper.
45

0 50OtherUnclearForecastCommentaryMethodologicalSynthesisInferenceEmpirical
0.0%  (n=1)0.9%  (n=21)2.0%  (n=46)2.2%  (n=50)3.2%  (n=73)3.4%  (n=79)18.9%  (n=433)69.4%  (n=1594)Paper type
0 50UnclearIrrelevantOceaniaSouth AmericaNorth AmericaEuropeAfricaAsia
1.2%  (n=28)4.1%  (n=94)5.4%  (n=125)7.4%  (n=169)12.0%  (n=275)24.3%  (n=558)33.8%  (n=777)44.2%  (n=1015)Geographical scope
0 100EconomicNo dataUnclearMobilitySentimentHealthSyntheticTraditional
0.7%  (n=16)0.8%  (n=19)1.1%  (n=25)1.2%  (n=28)1.7%  (n=40)2.0%  (n=45)11.0%  (n=253)94.5%  (n=2170)Data type
Share of papers (%)Fig. D2 Corpus-scale label distributions from the fully local deployment.The figure
reports the same three axes and the same 2,297 papers as Figure D1, produced on local hardware
with qwen2.5:32b-instruct and a BAAI/bge-m3 index at T=0. Paper type is single-label, whereas
geography and data type are multi-label, so their percentages can sum to more than100%.
observed where manual labels are available. This is consistent with the corpus-scale
shift reflecting the local model’s lower paper-type accuracy rather than a substantive
difference in the underlying corpus.
The local model predicts somewhat fewer geographic labels than the served model
at corpus scale. On the labelled sample, some of those lower frequencies are closer to
the shares in the manually labelled sample. Therefore, the difference between the two
corpus-scale distributions should not automatically be interpreted as an error of the
local model.
The labelled sample therefore helps interpret the direction of aggregate differences,
but only the evaluation-set accuracy metrics support model comparisons at paper level.
D.4 Stability and error diagnostics
These diagnostics complement the aggregate performance summaries in Section 4 and
use the same fixed evaluation set of214papers. Aggregate scores alone do not show
whether different models fail on the same papers.
For a fixed task and temperature, we define theever-wrong set Efor a model as
the papers for which at least one of the R=5runs differs from the complete manual
label set. This diagnostic is conditional on five repeats: with more repeats, a paper
has more opportunities to enter E. We then quantify the overlap and containment of
the ever-wrong sets forgemini-2.5-proandgemini-2.5-flash.
TablesD6andD7showthattheever-wrongsetsoverlapsubstantially.Forgeography,
data type, anddata accessibility, most papers that are ever misclassified by Pro also
appear in Flash’s ever-wrong set. The two models therefore tend to share a subset
of difficult papers on these axes. Containment is substantially lower forpaper type,
indicating that disagreement on this task is more model specific. These patterns
motivate using persistent errors or cross-model disagreement as additional signals
for identifying papers that merit targeted retrieval, prompt refinement, or manual
inspection.
46

Table D6 Containment of ever-wrong sets atT=0.0.
Task|E pro| |Eflash| |E∩| |E∩|/|Epro| |E∩|/|Eflash|
Geography 25 45 20 80.0% 44.4%
Data type 25 32 20 80.0% 62.5%
Data accessibility 95 123 72 75.8% 58.5%
Paper type 34 37 15 44.1% 40.5%
For each task, Eis the set of papers that are incorrectly labeled in at
least one of the R=5runs. For multi-label tasks, incorrect means that
the predicted label set is not an exact match to the manual label set.
|Epro|and|Eflash|report ever-wrong set sizes for the two backends. |E∩|
reports overlap. The last two columns divide it by the corresponding
backend’s ever-wrong set size.
Table D7 Containment of ever-wrong sets atT=1.0.
Task|E pro| |Eflash| |E∩| |E∩|/|Epro| |E∩|/|Eflash|
Geography 28 59 25 89.3% 42.4%
Data type 30 40 26 86.7% 65.0%
Data accessibility 116 163 106 91.4% 65.0%
Paper type 37 54 23 62.2% 42.6%
Definitions as in Table D6.
Appendix E Comparison baselines
This appendix details the methods, configurations, and statistical comparisons behind
the baseline results in Section 4.2.3 and Table 3. The families differ fundamentally in
what they require and what they measure, so they are read here not as a homogeneous
leaderboard but as a map of what each kind of method can and cannot do on this task.
The bottom line is this: across the four axes, EpiLens matches or exceeds the
tested baselines while requiring no labelled training data. The advantage is largest
where the relevant evidence lies outside the title and abstract, indicating that the
main contribution is not simply model choice but access to body-text evidence, with
targeted retrieval mattering most for data accessibility (Section 4.2.2). On geography,
whose signal is largely in the title and abstract, the advantage over the abstract-only
LLM is small, although the margins over the supervised baselines are the largest of
any axis (Table E1).
E.1 The design space and the methods we compare
Classifying a literature into analytically useful categories admits methods that differ
along two requirements that are easy to conflate (Figure E1): whether the category
scheme isspecified by the analystordiscovered from the corpus, and whether the
method needsmanually labelled training examplesbefore it can be applied.
47

Topic models
LDA, NMF, BERTopic,
guided BERTopic
Infer themes. Seeds may
steer discovery, but topics
are interpreted after the fact.Zero-shot
classifiers
EpiLens, zero-
shot NLI, prototype
No task-specific training set.
Semi-supervised
topic models
semi-supervised BERTopic
Training-paper labels
steer discovery. Topics
remain corpus-derived.Supervised
classifiers
TF-IDF+SVM, Set-
Fit, ModernBERT
Learned from labelled papers.Corpus-inferred
categoriesAnalyst-specified
categories
No training
labels
Labelled
papers
requiredORIGIN OF CATEGORIESLABELING EFFORT
Fig. E1 The design space of methods for classifying scientific literature.The map separates
two choices: where the categories come from, and whether a labelled training set is required before
the method can be used. EpiLens occupies the analyst-specified, annotation-free cell: categories are
written as natural-language descriptions and applied without task-specific training labels, and retrieval
then grounds those decisions in paper evidence.
Unsupervised and guided topic models(latent semantic analysis (LSA) [ 52], proba-
bilistic latent semantic analysis (PLSA) [ 53], latent Dirichlet allocation (LDA) [ 54],
non-negative matrix factorisation (NMF) [ 55], standard BERTopic [ 56], and seed-
guided variants) need no labelled papers but infer themes from the corpus rather than
applying a predefined scheme. They answer what structure a literature happens to
exhibit, not how its papers fall under a fixed scheme. Semi-supervised topic-model
variants occupy a different position: they can use labelled papers to steer topic dis-
covery, but their output remains corpus-inferred topics rather than direct predictions
under the analyst’s coding scheme.
Supervised classifierslearn an analyst-specified scheme from labelled papers by
pairing arepresentationof the text with a trainedclassifier head. Rather than designate
a single representative, we name three points in that space.TF-IDF+SVM, a term-
frequency–inverse-document-frequency representation [ 57] (each word weighted by how
distinctive it is to a document) fed to a linear support vector machine [ 58], is the
classic lexical anchor.SetFit[ 59], which fine-tunes a sentence-embedding model on
the labelled examples, is the fine-tuned neural representative.ModernBERT[ 60], an
8192-token long-context encoder applied to the parsed body text (overflow handled by
window pooling), is the long-context body-text representative. Acalibrated prototype,
which tunes only a single decision threshold, is the most minimal use of labels. The
full representation ×text-scope×classifier-head grid behind these named points is
swept in Section E.5.
48

Off-the-shelf zero-shot classifiersapply analyst-specified category descriptions
without task-specific training labels. In the NLI formulation, the input text is treated
as a premise and each candidate label is verbalised as a hypothesis. The model then
scores whether the premise entails the hypothesis [ 61]. In our runs, the premise is
the paper title and abstract. This aligns with the standard usage mode for zero-
shot NLI classifiers. We do not use the full-paper as premise because standard NLI
models have a finite input window for scoring premise-hypothesis pairs which scientific
papers often exceed. We also evaluate aprototype similarityapproach for zero-shot
classification, which represents each label by a singleprototypevector (the embedding
of its description) and assigns the labels whose prototypes are closest to the paper’s
own embedding.
EpiLens’s Paper Classifier occupies the position the others leave awkward: analyst-
specified categories, revisable as text, applied with no training labels, and grounded
in full-document evidence through retrieval. Amajority prior(always predicting the
most common label set) provides a reference floor. Figure E1 maps these families and
EpiLens’s place among them. The next section records how each method is configured
and the role it plays. We then position them by what each requires against the accuracy
it reaches.
E.2 Configuration and role of each method
Having placed each family in the design space (Section E.1), we now give the config-
uration and role of each method, grouped by family, so the comparisons that follow
can be read against a fully specified set of baselines. Themajority priorsits outside
these families as a reference floor rather than a method: it always predicts the most
frequent label set (estimated leave-one-out to avoid leakage from the target paper’s
own label), takes no text as input, and marks how much apparent performance follows
from label skew alone.
E.2.1 Zero-shot baselines
Zero-shot classifier baselines assign papers to an analyst-specified label set without
training on manually annotated examples from the task. Here we use the term for two
retrieval-free, annotation-free classifiers: a prototype-similarity classifier whose label
representations are built from the coding protocol, and an off-the-shelf NLI classifier
that scores verbalised label hypotheses against the paper metadata. Both operate on
title and abstract only. Calibrated variants that tune thresholds on manual labels are
reported with the supervised baselines.
Prototype similarity.
Each label is represented by aprototypevector: the embedding of its definition text
where the coding protocol provides one (paper type, data type, data accessibility), and
of the label name otherwise (geography). At inference the paper’s title and abstract are
embedded in the same space and scored by cosine similarity to each prototype. We use
two representations: a sparse TF-IDF space and a dense one from allenai-specter ,
a sentence transformer trained on scientific-paper pairs. The choice of representation
49

matters for what “similarity” means. With TF-IDF the label prototype is essentially
its own words, so the method reduces to lexical overlap between the abstract and
the label text. For geography, where the prototype is just the continent name, this
is close to detecting whether the abstract literally mentions that region. The dense
SPECTER prototype relaxes this to semantic similarity, which is why it helps on
the axes with descriptive definitions more than on the name-only geography axis.
Assignment uses a relative rule rather than a single absolute cutoff: a label is kept
if its similarity is at least a fraction τrel=0.92of the top-scoring label’s similarity,
above an absolute floor τ≥0.03, and among the top three. Theuncalibratedregime
uses these defaults and needs no labels, so it is a genuinely annotation-free zero-shot
baseline. Thecalibratedregime instead selects the absolute floor τper fold by 5-fold
cross-validation (maximising mean Jaccard on the other folds, over a grid of score
quantiles that includes the default), holding the relative rule fixed. This uses labels,
so it counts as a minimal supervised method reported with the supervised baselines.
Because the default is among the candidates, calibration cannot underperform it on the
selection folds, though held-out performance may still be lower. Calibration recovers
part of the gap on the multi-label tasks (Table 3).Paper type, single-label and decided
by arg-max, has no threshold to tune. Even calibrated, the prototype stays below the
discriminative supervised classifiers, because a single global acceptance rule cannot
capture per-label decision boundaries.
Zero-shot NLI entailment.
Following the standard zero-shot text-classification recipe [ 61], each candidate label is
framed as a natural-language-inference hypothesis (“This paper is about {label}.”),
and an NLI-fine-tuned DeBERTa-v3-base checkpoint ( MoritzLaurer/DeBERTa-v3-
base-mnli-fever-anli [62,63]) scores the entailment of each hypothesis against the
paper’s title and abstract. Labels clearing an entailment threshold are kept (the highest-
scoring label for the single-labelpaper typetask). This is the most direct off-the-shelf
occupant of EpiLens’s design-space quadrant (analyst-specified yet annotation-free),
but, as we run it, it reads only the title and abstract and reasons one label at a time.
Its behaviour is highly uneven across axes. We analyse why in Section E.3.1. This
unevenness does not appear to be caused by an obviously weak hypothesis template.
We tested a task-specific template (“This paper uses label data”), but it did not recover
Traditionaland increased over-prediction of non-traditional labels. Replacing label
names with one-sentencedefinitionsfrom the coding protocol (Section C) also lowered
scores on most tasks, suggesting that the NLI model is sensitive to hypothesis wording
and does not reliably apply the full coding protocol from metadata alone. We therefore
report the stronger label-name variant.
E.2.2 Guided and semi-supervised topic-model baselines
Guided and semi-supervised BERTopic.
BERTopic represents documents with transformer embeddings, clusters them in a
reduced space, and labels each cluster with class-based TF–IDF descriptors [ 56]. Its
native output is a set of corpus-inferred topics, not assignments to an analyst-defined
taxonomy. In its plain, unlabelled form it is therefore a pure corpus-structure probe:
50

it asks how far each axis separates from corpus-level vocabulary structure with no
label information at all. In that form BERTopic is one of the topic models in the
corpus-structure audit (Section E.6), which is where we report that recoverability
question.
The guided and semi-supervised variants differ in the kind of analyst input they
consume. Theguidedvariant supplies seed terms drawn from each label definition,
biasing topic discovery toward the target concepts while leaving the discovered topics
free to merge, split, or ignore the labels. It therefore uses analyst guidance but not
labelled training papers. Thesemi-supervisedvariant instead passes the manual labels
of the training papers, using partial document-level supervision to steer the clustering.
For this reason the two variants occupy different rows of Figure E1. Guided
BERTopic remains annotation-free with respect to paper labels and belongs with the
topic-model family: the analyst supplies seeds, but the emitted topics remain corpus-
inferred and must be interpreted or aligned after the fact. Semi-supervised BERTopic
belongs in the labelled-papers row, because it consumes manual labels for training
papers, but it still remains in the corpus-inferred column rather than the analyst-
specified classifier column: labels steer the clustering, whereas the output is still a set
of topics that need not correspond one-to-one to the coding scheme. The contrast with
supervised classifiers is therefore not whether labels can influence the method, but
how they enter: a supervised head learns a decision boundary that maps text directly
onto the fixed label set, whereas guided or semi-supervised topic modelling uses seeds
or labels to bias topic discovery.
These topic-model variants answer a different question from EpiLens: not whether
a fixed coding protocol can be applied directly to each paper, but whether the label
axes align with separable corpus structure once discovery is nudged toward the target
concepts. We report these guided and semi-supervised topic-model variants in Table 3.
To avoid tying the comparison to a single topic granularity, we sweep the requested
number of topics over K∈{ 0.5,1,2,4}×|L|and report the best-aligned configuration.
E.2.3 Supervised baselines
Supervisedbaselinesallfollowonerecipe:pairarepresentationofthetextwithatrained
classifier head, fit on the manual labels under cross-validation. Within that recipe they
differ in one methodologically important way,where the label signal enters. In most
of them the representation is fixed independently of the labels, either unsupervised
(the sparse TF-IDF space) or a frozen pretrained encoder whose weights are never
updated, and the labels train only the head on top: there is no gradient path from
the head’s loss back into the representation, which reaches the head as a fixed feature
matrix. SetFit is the sole exception, letting the labels also reshape the representation
by contrastively fine-tuning the encoder before a head is fit on the adapted embeddings.
Even there the encoder and head are fit in separate stages, not by one joint objective,
so no baseline here is end-to-end fine-tuned in the sense of gradients flowing through
encoder and classifier together. Jointly training and tuning a full encoder and classifier
across four tasks would be disproportionate to an evaluation with only214manually
labelled papers and would also carry a high risk of overfitting. We surface three
named points, and Section E.5 sweeps the full representation ×scope×head grid
51

behind them:TF-IDF+SVM(classic lexical, head-only supervision),SetFit(the
one representation-adapting encoder), andModernBERT(frozen long-context body-
text encoder). All are trained and evaluated by cross-validation over the N=214
papers (using iterative multi-label stratification for the set-valued axes where feasible,
to balance label prevalence across folds), on the title+abstract scope shared with
the other abstract-scoped methods.ModernBERTis the exception, applied to the
parsed body text. We additionally report TF-IDF under leave-one-out as a robustness
variant. Together they establish how far label-supervised text classification reaches, as
a reference for the contribution of retrieval-grounded reasoning.
TF-IDF+SVM.
The classical anchor and simplest instance of the recipe, and a pure case of head-
only supervision: a sparse TF-IDF representation of the title and abstract (each term
weighted by how distinctive it is to a document), built from corpus term statistics
without ever seeing the labels, is fed to a linear support vector machine that is the
only trained component, under stratified 5-fold cross-validation with leave-one-out as
a robustness variant (Table 3). It is the strongest non-neural reference and, on the
lexically regular axes, the closest baseline to EpiLens.
SetFit.
SetFit [59] is a supervised, label-efficient text-classification method that first fine-tunes
a sentence-transformer on pairs of labelled examples with a contrastive objective, then
trains a classification head on the resulting embeddings. It is thus the one supervised
baseline here in which the labels reach the representation itself, not only the head,
though the contrastive fine-tuning and the head are still fit in separate stages rather
thanbyasinglejointobjective.Weuse sentence-transformers/all-mpnet-base-v2
as the base encoder and evaluate it under stratified 5-fold cross-validation across the
N=214evaluation papers. We do not use leave-one-out here, since it would need a
separate fine-tuning run for each held-out paper. To cap the number of contrastive
pairs drawn from our relatively large cross-validation folds, we set num_iterations= 2
instead of the default sampling strategy. Text scope is restricted to title and abstract.
This is also the natural scope for the chosen sentence-transformer: all-mpnet-base-v2
is intended for sentence- and short-paragraph embeddings, and its default sentence-
transformers configuration truncates inputs longer than a few hundred word pieces.
Applying it to the full body would therefore require either truncation or a separate
chunk-level pooling rule to aggregate window embeddings into a paper-level prediction.
ModernBERT (body text, long-context encoder).
ModernBERT [ 60] is a bidirectional encoder trained with a native8192-token sequence
length, substantially longer than the few-hundred-token windows of the other dense
encoders. We apply it as afrozenfeature extractor under the same stratified 5-fold
protocol as the other supervised baselines, with its weights never updated (as for the
other dense encoders here): each body window is encoded with the encoder held fixed,
token embeddings are mean-pooled to obtain a window representation, and these
fixed representations are passed to a separately trained MLP classification head. For
52

papers whose parsed body fits within the8192-token window, this gives a single body-
level representation. For longer papers, we encode multiple windows and mean-pool
their representations rather than truncating the body (Section E.5). ModernBERT
is therefore the supervised baseline with the widest body-text scope, testing how far
a frozen long-context encoder’s features, with only the head trained, can go without
label-targeted retrieval.
E.2.4 RAG ablation baselines
Three variants of the same workflow (metadata only, random chunks, and the whole
body) join full EpiLens in the four-condition ablation of Section 4.2.2.
Metadata-only LLM (metadata_llm).
Runs the same language model with the same classification prompts as EpiLens, but
replaces retrieved full-text evidence with only the paper’s title and abstract. This
directly isolates the contribution of retrieval: any performance gap between this baseline
and the full system is attributable to the retrieved body-text passages. It is given the
raw parsed metadata (title and abstract) and never any body text, keeping it a clean
ablation of retrieval. For the32papers with no machine-readable abstract it therefore
sees the title alone, exactly as the other abstract-scoped baselines do. The paper is
not skipped, so all214are scored on the same set as the full-text system.
Random-chunk LLM (random_chunk_llm).
Runs the same model and prompts as the full system, but selects kbody-text passages
byuniform random samplingfrom the per-paper document pool in the vector index
rather than by semantic similarity to the template paragraphs. This holds body-text
access fixed while ablating retrieval quality, thereby isolating the contribution of
targetedretrieval from the contribution ofbody-text accessper se.
Whole-body LLM (full_body_llm).
Runs the same model and prompts, with the same per-paper chunk pool, but passes
every body chunk of the paper in document order instead of kof them, so the prompt
holds the whole parsed body (the longest prompts have22451and22771tokens for
the two generators, within their context windows). It tests whether selecting passages
is needed at all. We ran this condition at T=0five times with gemini-2.5-flash and
once with the fully local qwen2.5:32b-instruct , whose full-workflow outputs were
identical across five repeats (Section D.3).
E.3 Method requirements and performance trade-offs
This comparison is not a leaderboard, but a map of what each approach requires in
order to reach a given level of agreement. The relevant trade-off is between annotation
effort, text scope, and task flexibility: EpiLens uses analyst-specified categories that
can be edited as text, requires no task-specific training labels, and grounds each
decision in retrieved body-text evidence. The baseline scores therefore measure not
53

only modelling accuracy, but also what is lost or gained when one removes retrieval,
restricts the text scope, or introduces labelled supervision.
Table3collectsthehead-to-headresults.Thenexttwosubsectionscomparemethods
first at title-and-abstract scope and then with access to body text.
To separate model effects from evidence access, we use metadata_llm as the pivot
comparison. This baseline keeps EpiLens’s generator and task instructions but restricts
the evidence to paper metadata, i.e. the title and abstract. It therefore answers two
questions in sequence: first, at title-and-abstract scope, what does an instruction-
following LLM buy relative to classical, neural, and zero-shot baselines? Second, what
does targeted body-text retrieval add on top?
E.3.1 Comparison at title-and-abstract scope
At title-and-abstract scope, metadata_llm is the cleanest comparison because it
uses the same instruction-following backend as EpiLens without targeted body-text
retrieval. Its performance (0 .713paper type,0 .826geography,0 .816data type,0 .241
data accessibility) shows that the benefit of the LLM at fixed evidence scope is
task-dependent rather than uniform.
Where the relevant signal is usually present in the abstract, the metadata-only
LLM matches or exceeds the standard abstract-scoped baselines. This is clearest for
geography(0 .826vs. TF-IDF+SVM0 .592, SetFit0 .599, NLI0 .709), and roughly
tied forpaper type(0 .713vs.0.713for TF-IDF+SVM and0 .734for SetFit). But
the labelled baselines are not dominated everywhere. Ondata type, the bag-of-words
and embedding classifiers perform better (0 .854and0.864vs.0.816), consistent
with a lexically regular task dominated by the near-universalTraditionallabel. The
clearest failure isdata accessibility: metadata_llm reaches only0 .241, far below the
supervised abstract baselines (0 .533–0.574), because the decisive access statement is
rarely available in the abstract. This marks the limit of metadata-only reasoning and
identifies where retrieval should matter most.
Whencomparedagainstthestrongesttunedabstract-scopesupervisedconfiguration
rather than the standard representative rows (Table E2), the metadata-only LLM’s
advantage is narrower: it remains ahead only ongeography(0 .826vs.0.750), while
tuned encoders overtake it elsewhere. Thus, EpiLens’s consistent advantage is not
simply instruction-following at fixed scope: it comes from adding body-text evidence.
The weakest baseline scores should be read as diagnostics of default method–task
mismatch, not as ceilings on the corresponding model families. Zero-shot NLI ondata
typeis the clearest case ( J=0.018): it almost never predicts the dominantTraditional
class. This appears to reflect a lexical-anchoring asymmetry rather than a residual-
category effect: in our protocolTraditionalis defined positively (standard surveillance
case counts, hospital admissions, PCR/antigen-confirmed test line lists), but it is
a broad, heterogeneous category that papers signal only implicitly, since a paper
reporting national surveillance counts rarely states in so many words that it uses
“traditional” data. The non-traditional categories, by contrast, each name a specific
kind of source (mobility, health, sentiment, economic) that papers typically do state
explicitly, giving an entailment hypothesis built from the label name a much stronger
lexical anchor to key on thanTraditionaloffers.
54

Prototype similarity shows a different failure mode: its uncalibrated version is a
minimal annotation-free semantic-matching baseline, while threshold calibration uses
labels to recover part of the gap. For example, calibration improves SPECTER ondata
typefrom0 .52to0.67, and TF-IDF ondata accessibilityfrom0 .22to0.25(Table 3).
We therefore treat the calibrated prototype as a minimal supervised method and the
uncalibrated prototype as a zero-shot diagnostic.
E.3.2 Comparison with body-text access
Moving from metadata_llm to full EpiLens holds the generator and task instructions
fixed while replacing metadata-only evidence with targeted body-text retrieval. The
resulting difference estimates the contribution of evidence access and evidence selection,
and it tracks where the discriminative signal lies. Ongeography, where the abstract
already carries much of the signal, retrieval adds little: EpiLens (0 .875) is only slightly
ahead of the metadata-only LLM (0 .826), and the paired difference does not meet the
adjusted threshold (Section E.4). This is the only comparison in which EpiLens does
not clearly lead. On the other three axes, EpiLens is ahead of every tested abstract-
scoped baseline under the paired tests reported in Table E1. The margin is largest
where decisive evidence is distributed through the body or concentrated in a local
statement: paper type improves from0 .713to0.877, and data accessibility from0 .241
to0.670. Ondata type, the margin is narrower (0 .916vs. TF-IDF+SVM0 .854, Holm-
adjustedp=0.0008), and full-document TF-IDF recovers much of the difference (0 .881,
Table E2), suggesting that this axis is driven more by broad lexical evidence than by
label-targeted retrieval.
This body-text access is not an information advantage withheld from the baselines.
Title and abstract are the natural scope for short-window encoders and off-the-shelf
zero-shot classifiers, but we also evaluate supervised methods with body-text access
in the configuration grid: full-document vectorisation for TF-IDF, chunk-and-pool
variants for short-window encoders, and ModernBERT ’s8192-token long-context window
for the dense long-context setting (Table E2). The distinction is therefore not whether
baselines are allowed to read the body, but how it is used: the language model applies
the coding protocol to body-text passages, and selecting those passages by the labels
adds most for data accessibility. The component ablation (Section 4.2.2) decomposes
this retrieval contribution directly, while the supervised grid and saturation analysis
(Section E.5) make an under-powered baseline or small-label artefact unlikely as the
main explanation.
E.4 Statistical methodology
The baseline comparison in Section 4.2.3 uses the mean per-paper Jaccard score defined
in Section 4.2 on the same214manually labelled papers. We use paired resampling
to check how sensitive the observed baseline differences and the ablation differences
in Section 4.2.2 are to the papers evaluated. For EpiLens with gemini-2.5-flash ,
each paper’s score is averaged over five runs. The supervised baselines use out-of-
fold predictions: within each configuration, each paper is predicted by a model fitted
without that paper.
55

For each comparison, we resample the per-paper Jaccard differences with replace-
ment (B=10 000) and calculate a bias-corrected and accelerated (BCa) interval for their
mean. This keeps both methods’ scores paired on each paper; BCa adjusts the interval
for bias and skewness [ 64]. We report pointwise95%intervals; each two-sided p-value
is the smallest significance level at which the corresponding BCa interval excludes zero.
We apply Holm–Bonferroni correction within each task, separately for the baseline
comparisons and each generator’s ablation comparisons. Holm’s step-down adjustment
limits the chance of at least one false rejection within each comparison family [ 65].
Tables 1 and 3 give the same BCa interval for one configuration’s mean J, resampling
papers in the same way. This resampling holds the manual labels, model predictions,
and selected configurations fixed. It does not reproduce the review-stratified sampling
design or model fitting, so the intervals describe sensitivity to these papers rather than
uncertainty about performance across corpora.
In Table E1, all twelve gaps to the three named supervised baselines meet the
adjusted threshold. This supports the observed lead over those baselines under resam-
pling of the labelled papers. The smaller geography gap against the abstract-only
metadata_llm (∆J=+ 0.049) does not meet the threshold, leaving the extra gain from
the full workflow over metadata alone uncertain on that axis.
Table E1 Paired comparisons of EpiLens (gemini-2.5-flash,T=0) with
selected baselines.
Baseline Paper type Geography Data type Data access.
Majority prior (floor)+0.283∗+0.521∗+0.073∗+0.198∗
Zero-shot NLI+0.311∗+0.166∗+0.898∗+0.315∗
TF-IDF + SVM+0.164∗+0.283∗+0.062∗+0.137∗
SetFit+0.143∗+0.275∗+0.052∗+0.096∗
ModernBERT (full body)+0.092∗+0.222∗+0.051∗+0.093∗
metadata_llm+0.164∗+0.049 +0.100∗+0.429∗
Each cell is the mean per-paper Jaccard difference (EpiLens minus the baseline) on the
N=214evaluation papers. A marker (∗) denotes a Holm-adjusted p-value below0 .05
across eight comparisons per task. Six are shown here. The full-body and leave-one-out
TF-IDF variants also enter the correction. Tests use B=10 000paired BCa resamples
(Section E.4). They hold the manual labels, predictions, and configurations fixed. The
topic-model andBest supervisedrows of Table 3 are descriptive because their alignment
or configuration was selected using the evaluation papers. Prototype rows are reported
as diagnostics and are not tested here.
E.5 Comprehensive supervised configuration grid and
saturation
To position EpiLens against a wider range of supervised approaches than the named
representatives of Section E.3, Table E2 crosses three choices. The representations
are sparse TF-IDF, frozen all-mpnet-base-v2 [25,66], frozen SPECTER [67], and
the frozen 8192-token long-context ModernBERT-base . Text scope is either title and
56

abstract or full body, ingested by chunk-and-pool for the short-window encoders and
within ModernBERT ’s8192-token window, with overflow mean-pooled across windows.
The classifier heads are linear SVM, logistic regression, gradient-boosted trees, and
MLP. Each combination follows the same stratified 5-fold protocol over the same
N=214papers, with leave-one-out additionally reported for the linear heads. The
MPNet embedding uses the sentence-transformers/all-mpnet-base-v2 check-
point [68]. Every cell is head-only supervision (Section E.2.3): the representation is
fixed (unsupervised TF-IDF or a frozen encoder) and only the head is trained, so the
grid isolates representation family, text scope, and head choice without the confound
of end-to-end fine-tuning. The heads use fixed settings, not tuned in an inner cross-
validation loop, and all configurations share one fold partition. TheBest supervisedrow
offsets this in the baselines’ favour: it picks the best configuration for each axis on the
evaluation papers themselves, which overstates what a configuration chosen in advance
would reach. Full-body access and dense encoders both raise the best score reached
in the grid, and non-linear heads edge the linear ones on dense representations, but
nocell reaches EpiLens on any axis (Table E2). The long-context ModernBERT , whose
8192-token window holds the full body of182of the214papers in a single window
(the32longer papers, up to ∼16k tokens, are mean-pooled across a few windows, as
the short-window encoders always are), lands in the same band, suggesting that short
input windows alone do not explain the gap to EpiLens.
One possible concern is that the supervised baselines have too few training examples.
Within the observed range, every learning curve in Figure E2 remains below the
EpiLens reference and most flatten before the full sample. The 5-fold and leave-one-out
scores in Table E2 also differ by only a few points. This does not establish a ceiling for
supervised learning, but it suggests that modest increases in labels alone would not
close the observed gap under the tested configurations.
E.6 Corpus-structure audit
The topic models of Section E.1 are run not as classifiers but as acorpus-structure
audit: a check of how separable each axis’s categories are by vocabulary alone, which
bounds how much of the task a purely lexical, retrieval-free method could solve in
principle. Table E3 reports the best Jeach model reaches across the Ksweep under
an oracle cluster-to-label alignment, anupper boundon that structural signal. Even so,
topic models only partly recover the lexically regular axes (paper type,data type) and
sit near the majority floor ondata accessibility: the categories that demand reading
and reasoning are exactly the ones vocabulary structure cannot separate, which is
where retrieval-grounded reading is decisive.
57

Table E2 Comprehensive supervised configuration grid.
Configuration Paper type Geog. Data type Data access.
TF-IDF (title+abstract)
Linear SVM 0.713/0.713 0.592/0.607 0.854/0.871 0.533/0.538
Logistic Reg. 0.706/0.696 0.593/0.575 0.881/0.871 0.528/0.514
GB trees 0.595/– 0.453/– 0.814/– 0.467/–
MLP 0.696/– 0.599/– 0.856/– 0.558/–
TF-IDF (full body)
Linear SVM 0.724/0.741 0.615/0.627 0.881/0.890 0.590/0.583
Logistic Reg. 0.694/0.712 0.600/0.589 0.862/0.850 0.555/0.566
GB trees 0.764/– 0.596/– 0.883/– 0.586/–
MLP 0.748/– 0.651/– 0.888/– 0.619/–
Frozen mpnet (title+abstract)
Linear SVM 0.672/0.693 0.703/0.686 0.822/0.817 0.461/0.453
Logistic Reg. 0.661/0.681 0.685/0.683 0.770/0.783 0.459/0.464
GB trees 0.738/– 0.694/– 0.875/– 0.593/–
MLP 0.780/– 0.750/– 0.873/– 0.588/–
Frozen mpnet (full body)
Linear SVM 0.661/0.676 0.736/0.726 0.836/0.815 0.539/0.475
Logistic Reg. 0.641/0.664 0.752/0.734 0.787/0.807 0.521/0.489
GB trees 0.750/– 0.724/– 0.898/– 0.612/–
MLP 0.757/– 0.771/– 0.883/– 0.577/–
Frozen SPECTER (title+abstract)
Linear SVM 0.686/0.719 0.551/0.545 0.822/0.854 0.494/0.507
Logistic Reg. 0.667/0.681 0.526/0.543 0.787/0.815 0.469/0.488
GB trees 0.720/– 0.555/– 0.861/– 0.574/–
MLP 0.742/– 0.633/– 0.881/– 0.558/–
Frozen SPECTER (full body)
Linear SVM 0.663/0.720 0.622/0.628 0.839/0.846 0.533/0.530
Logistic Reg. 0.663/0.716 0.619/0.631 0.804/0.835 0.512/0.525
GB trees 0.778/– 0.627/– 0.887/– 0.593/–
MLP 0.741/– 0.683/– 0.893/– 0.588/–
ModernBERT (8k) (title+abstract)
Linear SVM 0.726/0.693 0.507/0.543 0.826/0.834 0.482/0.511
Logistic Reg. 0.688/0.678 0.494/0.538 0.782/0.827 0.482/0.504
GB trees 0.720/– 0.485/– 0.863/– 0.565/–
MLP 0.731/– 0.581/– 0.878/– 0.583/–
ModernBERT (8k) (full body)
Linear SVM 0.702/0.706 0.591/0.605 0.829/0.875 0.495/0.524
Logistic Reg. 0.700/0.718 0.595/0.627 0.822/0.840 0.472/0.504
GB trees 0.734/– 0.534/– 0.871/– 0.572/–
MLP 0.785/– 0.652/– 0.864/– 0.577/–
Sample-averaged Jaccard for everyrepresentation ×text scope×classifier head
combination, all at stratified 5-fold CV over the same N=214papers. Each cell
is shown as “5-fold/LOO”. The gradient-boosted-tree and MLP heads show “–”
for the leave-one-out score, because a LOO sweep would refit the head once
per held-out paper, which is prohibitively expensive for these non-linear heads,
so only the 5-fold score is computed for them. The grid positions EpiLens
(Table 3) against the full space of reasonable supervised approaches rather than
a single representative. The small 5-fold-to-LOO gaps and the saturation curves
(Figure E2) show the baselines are near data-saturated at this sample size.
58

0.20.40.60.8TF-IDF
Jaccard J
Paper type
 Geog.
 Data type
 Data access.
0.20.40.60.8Frozen mpnet
Jaccard J
0.20.40.60.8Frozen SPECTER
Jaccard J
25 75 125 175
# labelled papers0.20.40.60.8ModernBERT (8k)
Jaccard J
25 75 125 175
# labelled papers
25 75 125 175
# labelled papers
25 75 125 175
# labelled papers
Linear SVM Logistic Reg. GB trees MLP title+abstract full body EpiLensFig. E2 Saturation of the supervised configuration grid.Learning curves (held-out Jaccard
vs. number of labelled training papers) grouped by representation (rows) and task (columns). Within
each panel, colour encodes the classifier head (linear SVM, logistic regression, gradient-boosted trees,
MLP) and line style the text scope (solid = title+abstract, dashed = full body). The black dotted
line is EpiLens (flash, T=0). Every curve remains below EpiLens within the observed training range,
and most curves flatten before the full sample, suggesting that the gap is unlikely to be closed merely
by small increases in labelled data.
59

Table E3 Corpus-structure audit: how separable each axis is by vocabulary alone.
Method Paper type Geography Data type Data access.
Reference
Majority prior (floor) 0.593 0.353 0.843 0.472
Unsupervised topic models (bestKacross sweep, oracle label alignment)
LSA 0.593 0.501 0.843 0.489
LDA 0.757 0.489 0.880 0.484
NMF 0.682 0.576 0.874 0.516
PLSA 0.748 0.539 0.875 0.562
BERTopic 0.626 0.581 0.843 0.477
Reference
EpiLens (gemini-2.5-flash,T=0) 0.877 0.875 0.916 0.670
Sample-averaged Jaccard Jfor unsupervised topic models on the same214-paper evaluation set,
reported as the best Jacross the Ksweep ( K∈{ 0.5,1,2,4}×|L|) under an oracle cluster-to-label
alignment, in their original full-text, leave-one-out configuration. These arenotclassifiers (they
discover themes rather than apply the analyst’s scheme), and the best- K, oracle-aligned figures
are anupper boundon what vocabulary structure alone recovers. The majority floor and EpiLens
are shown for reference. Topic models partly recover the lexically regular axes (paper type,data
type) but sit near the floor ondata accessibility, the axis whose signal is a single buried statement.
60

Appendix F Linking estimates to corpus papers
As described in Section 4.1, the corpus was assembled from sources named in the seed
study’s early extraction sheets [ 32]. Here we describe how the estimates in those sheets
were linked to corpus papers for the comparisons in Section 4.4, and check how review
coverage and classification choices affect the main contrasts.
F.1 Scope and inputs
We use the data extraction sheets shared by the authors of the seed study. The sheets
are not redistributed. They contain346estimate rows and, after identifiers are resolved,
138distinct published sources. The sheets are a working version and do not exactly
match either the first preprint or the published article. The first preprint reports342
estimates [ 32]. Five sources behind eight sheet rows do not appear in its references.
The published article [ 11] reports a revised set of302estimates from154articles. All
counts below refer to the early sheets.
We group the parameters into seven categories: reproduction number (basic, effec-
tive, or controlled), serial interval, incubation period, latent period, infectious period
(including effective infectious periods), dispersion parameter k, and fatality risk (case
or infection). These groups combine related quantities for describing their cited papers.
We call a review cited by an estimate asource review. Anexpanded source review
is one whose included studies are linked to corpus records, whether through corpus
construction or a published study list.
F.2 Matching cited sources
Nineteen sheet rows have no published source to link, mostly because they relate to
the seed study’s own incubation-period calculations. We link the other327to their
cited sources, as illustrated in Figure F1. For a directly cited study or other document,
we analyse that source itself. A source review links to its included studies present in
the corpus.
The corpus-building records group included studies under the source review’s first
author and year. We use these identifiers to connect a sheet citation to the appropriate
group and check the DOI for reviews sharing an author and year. We match individual
documents chiefly by DOI, with author and year or verified manual matching where
needed. Where an identifier is malformed, takes a different form, or identifies another
document, we apply a declared replacement checked against Crossref [50].
F.3 Associated paper sets and labels
The corpus records links between its papers and the sources named in the sheets. We
use those links to define the paper set associated with each estimate row r, denotedBr.
For a directly cited study or other document, Brcontains that document. For a source
review,Brcontains its included studies that are present in the corpus (Section B).
These are studies the review identifies as evidence for its analysis, rather than every
item in its bibliography. The sheets cite the review for a particular estimate without
naming the contributing studies. For example, a review reporting both serial-interval
61

Extraction sheets of the
seed study (early version)
346 parameter estimates
327 estimates citing
a published sourceNo published
source
19 estimates,
left out
Source review
180 estimates
from 62 reviewsSingle study or
other document
147 estimates
from 76 documents
its included studies in the corpus the cited document itself
Papers behind the 327 estimates
all 2,297 corpus papers, described with EpiLens labels
Fig. F1 Linking parameter estimates to corpus papers.The327sheet rows citing a published
source divide into reviews and single studies or other documents. A review principally links to its
retrieved included studies in the corpus, while a directly cited document is itself analysed. The review
and document counts are distinct sources after identifiers are resolved.
and incubation-period estimates may use different subsets of its included studies for
each. Thus Brreflects the review’s included evidence, which may be broader than
the studies used for one value. Then, a paper can be included in several reviews
even if it was first retrieved for just one. We retain every documented membership.
For each parameter, pathogen, or combination of the two, we take the union of the
corresponding Brsets. Each paper contributes once to a group’s paper count and label
shares, regardless of how many estimates link to it. The same paper may appear in
several groups, so their paper counts are not additive.
TableF1tabulatesmodel-basedandsimulated-datasharesforthepapersetsplotted
in Figures 5 and 6. It also lists the influenza A subtypes pooled in the pathogen figure.
These percentages use the frozen gemini-2.5-flash labels atT=0over all associated
corpus papers, so they are model outputs without sampling intervals. The figures do
not separate pathogens within each parameter. Table F2 provides that breakdown for
the comparisons discussed in Section 4.4.
62

Table F1 Labels of the papers behind the estimates, by
parameter and by pathogen.
PapersModel-based
(%)Simulated
data (%)
By parameter
Reproduction number629 54 48
Fatality risk1549 22 13
Serial interval539 58 38
Incubation period845 33 20
Dispersionk137 57 55
Infectious period186 37 42
Latent period140 46 51
By pathogen
SARS-CoV-2590 54 28
Influenza
A/H1N1109 72 65
A/H1N1pdm09121 68 55
A/H2N2127 60 53
A/H3N2121 68 56
A/H1N1 and A/H3N21— —
A/H5N1146 58 47
SARS and MERS
MERS-CoV90 50 33
SARS-CoV-164 39 44
Ebola and Marburg
Ebola virus262 27 25
Marburg virus59 19 7
Bunyavirales
CCHF virus497 10 1
Lassa virus125 12 10
RVF virus214 7 2
Mpox virus153 10 8
Nipah virus112 4 4
Zika virus146 31 33
All2297 29 19
Each row counts distinct corpus records associated with estimates
of one parameter or pathogen. A record associated with several
parameters or pathogens counts for each, so the rows add up to more
thanAll.Model-basedandSimulateddata:percentageofthesepapers
to which EpiLens assigns the label with gemini-2.5-flash (model
outputs, not estimates). Model-based means paper type INFERENCE or
FORECAST ,and simulateddatameansdatatype SYNTHETIC .Shares are
not shown for rows with fewer than 10 papers. Section 4.4 compares
these labels with the manual labels for the comparisons it draws.
Parameters are ordered as in Figure 4. Bold headings name the
pathogen groups in the early extraction sheets shared by the authors
of Ward et al. [32]that cover several pathogens. A/H1N1: estimates
not specific to the 2009 pandemic virus. A/H1N1pdm09: the 2009
pandemic virus. A/H1N1 and A/H3N2: estimates reported for both
subtypes together. CCHF: Crimean–Congo haemorrhagic fever. RVF:
Rift Valley fever. Fatality risk: case or infection fatality risk.
63

Table F2 Model-based shares in the pathogen–parameter
comparisons.
Pathogen Parameter PapersModel-based
(%)
Reproduction number and fatality risk
SARS-CoV-2 Reproduction number204 62
Fatality risk335 52
MERS-CoV Reproduction number55 40
Fatality risk67 36
Ebola virus Reproduction number117 47
Fatality risk237 26
Lassa virus Reproduction number22 36
Fatality risk124 11
Mpox virus Reproduction number61 11
Fatality risk55 7
Other fatality-risk comparisons
CCHF virus Fatality risk497 10
Nipah virus Fatality risk111 5
RVF virus Fatality risk185 8
Incubation period
SARS-CoV-2 Incubation period334 55
Mpox virus Incubation period115 6
Each row counts distinct corpus papers associated with estimates of
the named parameter for that pathogen. A paper can appear in sev-
eral rows. Model-based is the share labelled INFERENCE orFORECAST
by EpiLens with gemini-2.5-flash atT=0(model outputs, not
parameter estimates). Rows show the comparisons discussed in
Section 4.4. The full breakdown is in the reproduction package [ 42].
CCHF: Crimean–Congo haemorrhagic fever. RVF: Rift Valley fever.
F.4 Manual-label check
The shares reported for all associated papers in Section 4.4 depend on model-assigned
labels. We therefore check whether the contrast also appears under manual labels
among the associated papers in the214-paper evaluation sample. The manual labels
show the same ordering, with a38-percentage-point difference in model-based shares.
We examine how this difference varies across the cited sources represented in that
sample.
ThepairedbootstrapinSection E.4resamplesindividualpapersto assess differences
in classifier accuracy. Here, several labelled papers can come from one cited review, so
we resample source reviews and directly cited documents as groups. This asks whether
the manual-label contrast depends heavily on a few cited sources.
The resampling pool contains45sources represented by at least one labelled paper
behind either parameter. In each of4000draws, we select45sources with replacement,
keep each selected source’s labelled papers together, and recalculate the difference
between the two manual-label shares. A source shared by both parameters contributes
to both shares in the same draw. Within each parameter set, a paper linked to m
sources receives weight1 /mthrough each source, so that it counts once in the observed
64

share. The central95%of resampled differences is19–57percentage points, preserving
the direction under this source resampling. This check concerns the manually labelled
subset: the papers labelled within each review stay fixed, and the interval does not
estimate what manual labels would show across the full associated paper sets.
F.5 Sensitivity of the paper profiles
We now check whether the model-assigned reproduction-number–fatality-risk contrast
across all associated papers changes with review coverage, model choice, or the removal
of individual source reviews.
Review coverage.
Incomplete matching of a review’s included studies could change the paper profiles.
We linked corpus papers to62cited reviews, of which60report how many studies they
included. The other two are an analytic perspective and a narrative review without a
defined set of included studies. Counting a paper once for each review that includes
it, corpus records match2706of4467reported inclusions (61%). These are shares of
reported review inclusions, not shares of the corpus. Matching varies from14to100%
across reviews and is47%for those associated with reproduction numbers, against
56%for fatality risks.
Removing reviews below the median matching rate, while retaining directly cited
documents and reviews without reported totals, changes the model-based difference
from32to39percentage points. Hence, the direction persists.
Model choice and individual reviews.
Under the fully local qwen2.5:32b-instruct configuration, the model-based shares
are53%for reproduction numbers and15%for fatality risks, compared with54%and
22%under Flash. The simulated-data difference is smaller with the local configuration
(31%versus8%, compared with48%versus13%under Flash).
With the Flash labels, omitting each of the42source reviews in turn leaves the
model-based difference at28–36percentage points. The model-based ordering therefore
holds under both configurations and after the removal of any one review.
Incubation periods across pathogens.
In Section 4.4,55%of the334SARS-CoV-2 papers associated with incubation-period
estimates are labelled model-based, compared with6%of the115mpox papers. We
check whether this difference depends on one source review or on papers also linked
to other parameters. The manual comparison covers only31and11papers, so these
checks use the full model-assigned sets.
The model-based share is at least24%in each of seven SARS-CoV-2 source
reviews and at most11%in each of four mpox source reviews. The ordering therefore
appears across the cited reviews, rather than arising only when their papers are pooled.
Excluding papers also linked to reproduction-number or fatality-risk estimates for the
same pathogen leaves shares of54%of201SARS-CoV-2 papers and6%of70mpox
papers. These shares are close to those in the full associated sets (55%and6%), so
papers shared with those other parameter sets do not account for the difference.
65

References
[1]Chafetz H, Zahuranec AJ, Marcucci S, et al (2022) The #Data4COVID19 Review:
Assessing the Use of Non-Traditional Data During a Pandemic Crisis. https:
//review.data4covid19.org, The GovLab, New York University. Accessed 20 March
2025
[2]Bolt K, Gil-González D, Oliver N (2024) Unconventional data, unprecedented
insights: leveraging non-traditional data during a pandemic. Frontiers in Public
Health 12:1350743
[3]Fallatah DI, Adekola HA (2024) Digital epidemiology: harnessing big data for
early detection and monitoring of viral outbreaks. Infection prevention in practice
6(3):100382
[4]ChenQ,YangM,QinL,etal(2025)AI4Research:Asurveyofartificialintelligence
for scientific research. arXiv:2507.01903
[5]Gokdemir O, Siebenschuh C, Brace A, et al (2025) HiPerRAG: High-performance
retrieval augmented generation for scientific insights. In: Proceedings of the
Platform for Advanced Scientific Computing Conference, pp 1–13
[6]Annepaka Y, Pakray P (2025) Large language models: a survey of their devel-
opment, capabilities, and applications. Knowledge and Information Systems
67(3):2967–3022
[7]Huang L, Yu W, Ma W, et al (2025) A survey on hallucination in large language
models: Principles, taxonomy, challenges, and open questions. ACM Transactions
on Information Systems 43(2):1–55
[8]Lewis P, Perez E, Piktus A, et al (2020) Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33:9459–9474
[9]Arslan M, Ghanem H, Munawar S, et al (2024) A survey on RAG with LLMs.
Procedia computer science 246:3781–3790
[10]Fan W, Ding Y, Ning L, et al (2024) A survey on rag meeting llms: Towards
retrieval-augmented large language models. In: Proceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining, pp 6491–6501
[11]Ward J, Gressani O, Kim S, et al (2026) The epidemiology of pathogens with
pandemic potential: a review of key parameters and clustering analysis. Epidemics
54:100882. https://doi.org/10.1016/j.epidem.2025.100882
[12]Abo El-Enen M, Saad S, Nazmy T (2025) A survey on retrieval-augmentation
generation (RAG) models for healthcare applications. Neural Computing and
66

Applications 37(33):28191–28267
[13]Salathé M, Bengtsson L, Bodnar TJ, et al (2012) Digital epidemiology. PLoS
Computational Biology 8(7)
[14]Park HA, Jung H, On J, et al (2018) Digital epidemiology: use of digital data
collected for non-epidemiological purposes in epidemiological studies. Healthcare
informatics research 24(4):253–262
[15]Mesquita S, Perfeito L, Paolotti D, et al (2025) Epidemiological methods in
transition: Minimizing biases in classical and digital approaches. PLOS Digital
Health 4(1):e0000670. https://doi.org/10.1371/journal.pdig.0000670
[16]Zhao WX, Zhou K, Li J, et al (2023) A survey of large language models.
arXiv:2303.18223
[17]Skarlinski MD, Cox S, Laurent JM, et al (2024) Language agents achieve
superhuman synthesis of scientific knowledge. arXiv:2409.13740
[18]Asai A, He J, Shao R, et al (2024) OpenScholar: Synthesizing scientific literature
with retrieval-augmented LMs. arXiv:2411.14199
[19]Gargari OK, Habibi G (2025) Enhancing medical AI with retrieval-augmented
generation: A mini narrative review. Digital health 11:20552076251337177
[20]Liu S, McCoy AB, Wright A (2025) Improving large language model applications
in biomedicine with retrieval-augmented generation: a systematic review, meta-
analysis, and clinical development guidelines. Journal of the American Medical
Informatics Association 32(4):605–615
[21]TypeSafe AI (2026) Introducing System One models & Jev. https://typesafe.ai/
blog/introducing-system-one-models-and-jev, archived at https://web.archive.
org/web/20260930032358/https://typesafe.ai/blog/introducing-system-one-
models-and-jev. Accessed 2 October 2026
[22]Deußer T, Sparrenberg L, Sifa R (2026) Evaluating and benchmarking the System
One model Jev. arXiv:2609.37647
[23](2026) Unstructured: Open-source pre-processing tools for unstructured data.
https://github.com/Unstructured-IO/unstructured, accessed 5 October 2026
[24](2008–2025) GROBID. https://github.com/grobidOrg/grobid, Software Her-
itage ID swh:1:dir:dab86b296e3c3216e2241968f0d63b68e8209d3c. Accessed 15
September 2026
[25]Reimers N, Gurevych I (2019) Sentence-bert: Sentence embeddings using siamese
bert-networks. In: Proceedings of the 2019 conference on empirical methods in
natural language processing and the 9th international joint conference on natural
67

language processing (EMNLP-IJCNLP), pp 3982–3992
[26]Johnson J, Douze M, Jégou H (2019) Billion-scale similarity search with GPUs.
IEEE Transactions on Big Data 7(3):535–547
[27](2020) Qdrant: Vector search engine. https://github.com/qdrant/qdrant, GitHub
repository. Accessed 15 September 2026
[28] Colvin S, Jolibois E, Ramezani H, et al (2024) Pydantic. Zenodo
[29](2022) LangChain. https://github.com/langchain-ai/langchain, GitHub repository.
Accessed 15 September 2026
[30](2022) LlamaIndex. https://github.com/run-llama/llama_index, GitHub
repository. Accessed 15 September 2026
[31]deepset (2019) Haystack: An open-source LLM framework. https://github.com/
deepset-ai/haystack, GitHub repository. Accessed 15 September 2026
[32]Ward J, Gressani O, Kim S, et al (2025) The epidemiology of pathogens with
pandemic potential: a review of key parameters and clustering analysis. medRxiv
preprint, version 1, https://doi.org/10.1101/2025.03.13.25323659, version 1:
https://www.medrxiv.org/content/10.1101/2025.03.13.25323659v1
[33]Atil B, Aykent S, Chittams A, et al (2024) Non-determinism of “deterministic”
LLM settings. arXiv:2408.04667
[34]PyTorch Documentation (2026) Numerical accuracy. https://docs.pytorch.org/
docs/main/notes/numerical_accuracy.html, accessed 29 September 2026
[35]Yuan J, Li H, Ding X, et al (2025) Understanding and mitigating numerical
sources of nondeterminism in LLM inference. arXiv:2506.09501
[36]Gond R, Kamath AK, Ramjee R, et al (2026) LLM-42: Enabling determinism in
LLM inference with verified speculation. arXiv:2601.17768
[37]United Nations Statistics Division (2026) Standard country or area codes for
statistical use (M49). https://unstats.un.org/unsd/methodology/m49/overview/,
accessed 5 October 2026
[38]Naidoo T, Nash R, Morgenstern C, et al (2025) epireview: Tools to update and
summarise the latest pathogen data from the Pathogen Epidemiology Review
Group (PERG). https://github.com/mrc-ide/epireview, r package. Accessed 23
September 2026
[39]Callaghan M, Müller-Hansen F, Bond M, et al (2024) Computer-assisted screening
in systematic evidence synthesis requires robust and well-evaluated stopping
criteria. Systematic Reviews 13(1):284
68

[40]Organization WH (2024) Ethics and governance of artificial intelligence for health:
large multi-modal models. WHO guidance. World Health Organization
[41]Perri V (2026) EpiLens: Local-first, evidence-grounded RAG workflows for scien-
tific literature analysis. https://github.com/VinsRR/EpiLens, Python package,
version 0.1.0, https://pypi.org/project/epilens/. Accessed 14 September 2026
[42]Perri V, Cattuto C, Paolotti D (2026) EpiLens: Lightweight RAG for epidemi-
ological knowledge acquisition. Results, code, and corpus references. Zenodo,
https://doi.org/10.5281/zenodo.22747111
[43]Lauer SA, Grantz KH, Bi Q, et al (2020) The incubation period of coronavirus
disease 2019 (COVID-19) from publicly reported confirmed cases: estimation and
application. Annals of internal medicine 172(9):577–582. https://doi.org/10.7326/
M20-0504
[44] Althaus CL (2014) Estimating the reproduction number of Ebola virus (EBOV)
during the 2014 outbreak in West Africa. PLoS currents 6. https://doi.org/10.
1371/currents.outbreaks.91afb5e0f279e7f29e7056095255b288
[45]Nguku PM, Sharif S, Mutonga D, et al (2010) An investigation of a major outbreak
ofRiftValleyfeverinKenya:2006–2007.TheAmericanjournaloftropicalmedicine
and hygiene 83(2 Suppl):5–13. https://doi.org/10.4269/ajtmh.2010.09-0288
[46]Nakoune E, Lampaert E, Ndjapou SG, et al (2017) A nosocomial outbreak of
human monkeypox in the Central African Republic. Open Forum Infectious
Diseases 4(4):ofx168. https://doi.org/10.1093/ofid/ofx168
[47](2026) Gemini API: Files API documentation. https://ai.google.dev/gemini-api/
docs/files, Accessed 10 February 2026
[48](2026) Gemini API: Document processing. https://ai.google.dev/gemini-api/docs/
document-processing, Accessed 10 February 2026
[49](2020) RapidFuzz: Rapid fuzzy string matching in Python and C++. URL
https://github.com/rapidfuzz/RapidFuzz, Accessed 5 March 2026
[50] Hendricks G, Tkaczyk D, Lin J, et al (2020) Crossref: The sustainable source of
community-owned scholarly metadata. Quantitative Science Studies 1(1):414–427
[51] (2023) Ollama. https://ollama.com, Accessed 15 September 2026
[52]Deerwester S, Dumais ST, Furnas GW, et al (1990) Indexing by latent semantic
analysis. Journal of the American society for information science 41(6):391–407
[53]Hofmann T (1999) Probabilistic latent semantic indexing. In: Proceedings of the
22nd annual international ACM SIGIR conference on Research and development
in information retrieval, pp 50–57
69

[54]Blei DM, Ng AY, Jordan MI (2003) Latent Dirichlet allocation. Journal of Machine
Learning Research 3:993–1022
[55]Lee DD, Seung HS (1999) Learning the parts of objects by non-negative matrix
factorization. nature 401(6755):788–791
[56]Grootendorst M (2022) BERTopic: Neural topic modeling with a class-based
TF-IDF procedure. arXiv:2203.05794
[57]Sparck Jones K (1972) A statistical interpretation of term specificity and its
application in retrieval. Journal of documentation 28(1):11–21
[58]Cortes C, Vapnik V (1995) Support-vector networks. Machine learning 20(3):273–
297
[59]Tunstall L, Reimers N, Jo UES, et al (2022) Efficient few-shot learning without
prompts. arXiv:2209.11055
[60]Warner B, Chaffin A, Clavié B, et al (2025) Smarter, better, faster, longer: A
modernbidirectionalencoderforfast,memoryefficient,andlongcontextfinetuning
and inference. In: Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp 2526–2547
[61]Yin W, Hay J, Roth D (2019) Benchmarking zero-shot text classification: Datasets,
evaluation and entailment approach. In: Proceedings of the 2019 conference on
empirical methods in natural language processing and the 9th international joint
conference on natural language processing (EMNLP-IJCNLP), pp 3914–3923
[62]He P, Gao J, Chen W (2023) DeBERTaV3: Improving DeBERTa using ELECTRA-
style pre-training with gradient-disentangled embedding sharing. In: The Eleventh
International Conference on Learning Representations
[63]Laurer M (2026) DeBERTa-v3-base-mnli-fever-anli: Model card. Hugging Face,
https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, accessed
5 October 2026
[64]Efron B (1987) Better Bootstrap Confidence Intervals. Journal of the American
Statistical Association 82(397):171–185. https://doi.org/10.1080/01621459.1987.
10478410
[65]Holm S (1979) A Simple Sequentially Rejective Multiple Test Procedure.
Scandinavian Journal of Statistics 6(2):65–70. https://doi.org/10.2307/4615733
[66]Song K, Tan X, Qin T, et al (2020) Mpnet: Masked and permuted pre-training
for language understanding. Advances in neural information processing systems
33:16857–16867
70

[67]Cohan A, Feldman S, Beltagy I, et al (2020) Specter: Document-level representa-
tion learning using citation-informed transformers. In: Proceedings of the 58th
annual meeting of the association for computational linguistics, pp 2270–2282
[68]Sentence Transformers (2026) all-mpnet-base-v2: Model card. Hugging Face,
https://huggingface.co/sentence-transformers/all-mpnet-base-v2, accessed 5
October 2026
71
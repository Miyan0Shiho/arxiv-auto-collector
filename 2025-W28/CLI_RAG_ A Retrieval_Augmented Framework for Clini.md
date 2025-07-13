# CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs

**Authors**: Garapati Keerthana, Manik Gupta

**Published**: 2025-07-09 10:13:38

**PDF URL**: [http://arxiv.org/pdf/2507.06715v1](http://arxiv.org/pdf/2507.06715v1)

## Abstract
Large language models (LLMs), including zero-shot and few-shot paradigms,
have shown promising capabilities in clinical text generation. However,
real-world applications face two key challenges: (1) patient data is highly
unstructured, heterogeneous, and scattered across multiple note types and (2)
clinical notes are often long and semantically dense, making naive prompting
infeasible due to context length constraints and the risk of omitting
clinically relevant information.
  We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a
domain-specific framework for structured and clinically grounded text
generation using LLMs. It incorporates a novel hierarchical chunking strategy
that respects clinical document structure and introduces a task-specific
dual-stage retrieval mechanism. The global stage identifies relevant note types
using evidence-based queries, while the local stage extracts high-value content
within those notes creating relevance at both document and section levels.
  We apply the system to generate structured progress notes for individual
hospital visits using 15 clinical note types from the MIMIC-III dataset.
Experiments show that it preserves temporal and semantic alignment across
visits, achieving an average alignment score of 87.7%, surpassing the 80.7%
baseline from real clinician-authored notes. The generated outputs also
demonstrate high consistency across LLMs, reinforcing deterministic behavior
essential for reproducibility, reliability, and clinical trust.

## Full Text


<!-- PDF content starts -->

CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured
and Context Aware Text Generation with LLMs
Garapati Keerthana1, Manik Gupta1
1Birla Institute of Technology and Science, Pilani, Hyderabad, India
{p20240505,manik}@hyderabad.bits-pilani.ac.in
Abstract
Large language models (LLMs), including
zero-shot and few-shot paradigms, have shown
promising capabilities in clinical text genera-
tion. However, real-world applications face two
key challenges: (1) patient data is highly un-
structured, heterogeneous, and scattered across
multiple note types; and (2) clinical notes
are often long and semantically dense, mak-
ing naive prompting infeasible due to context
length constraints and the risk of omitting clin-
ically relevant information.
We introduce CLI-RAG (Clinically Informed
Retrieval- Augmented Generation), a domain-
specific framework for structured and clinically
grounded text generation using LLMs. It incor-
porates a novel hierarchical chunking strategy
that respects clinical document structure and
introduces a task-specific dual-stage retrieval
mechanism. The global stage identifies rele-
vant note types using evidence-based queries,
while the local stage extracts high-value con-
tent within those notes creating relevance at
both document and section levels.
We apply the system to generate structured
progress notes for individual hospital visits us-
ing 15 clinical note types from the MIMIC-III
dataset. Experiments show that it preserves
temporal and semantic alignment across vis-
its, achieving an average alignment score of
87.7%, surpassing the 80.7% baseline from real
clinician-authored notes. The generated out-
puts also demonstrate high consistency across
LLMs, reinforcing deterministic behavior es-
sential for reproducibility, reliability, and clini-
cal trust.
1 Introduction
Large language models (LLMs) have shown re-
markable success in natural language processing
(OpenAI, 2023; Yang et al., 2024), including clini-
cal applications such as summarization (Agrawal
et al., 2022; Wang et al., 2024), medical Q&A
(Singhal et al., 2023), and decision support (Fanget al., 2023). However, deploying LLMs in real-
world clinical settings remains challenging due
to the unstructured, fragmented, and semantically
dense nature of electronic health records (EHRs)
(Rule et al., 2021; Kuhn et al., 2015; Meystre et al.,
2008; Wang et al., 2017). Clinical documenta-
tion spans multiple heterogeneous note types, e.g.
nursing, radiology, consultations, each with varied
structure and granularity, often containing redun-
dant or incomplete content (Percha, 2021; Markel,
2010; Apathy et al., 2022).
Progress notes, typically structured in the SOAP
format (Subjective, Objective, Assessment, Plan),
are essential for ongoing care but are frequently
missing in real-world datasets appearing in only
8.56% of visits in MIMIC-III (Johnson et al.,
2016). Reconstructing such notes from fragmented
sources requires reasoning over multiple docu-
ments, temporal alignment, and adherence to clin-
ical structure. Off-the-shelf retrieval-augmented
generation (RAG) approaches (Lewis et al., 2020;
Izacard et al., 2022) are ill-suited for this task, as
they operate on flat corpora and ignore clinical
semantics, note provenance, or task-specific rele-
vance.
We introduce CLI-RAG (Clinically Informed
Retrieval- Augmented Generation), a structured
generation framework designed to synthesize
SOAP-format progress notes by composing evi-
dence from diverse EHR notes. Our method ad-
dresses two key research questions: 1. What infor-
mation is most relevant to progress note generation,
regardless of its source? 2. Which note types con-
sistently contribute towards this information?
To answer these, our system proposes a dual-
stage retrieval strategy. A global retrieval step
leverages task-specific clinical queries to extract
relevant content across all note types, followed by a
local retrieval phase that drills down into note-type-
specific content using tailored sub-queries. The
pipeline includes clinically structured preprocess-arXiv:2507.06715v1  [cs.CL]  9 Jul 2025

ing, hierarchical chunking, metadata-guided em-
beddings, and temporally conditioned prompting
to simulate real-world documentation workflows.
We evaluate our system on 1,108 patient visits
from MIMIC-III dataset, measuring lexical, seman-
tic, structural, and temporal fidelity. Our system
achieves a temporal alignment score of 87.7%, out-
performing clinician-authored notes 80.7%. These
findings highlight the effectiveness of clinically in-
formed retrieval in generating coherent, faithful
progress notes, with implications for summariza-
tion, documentation support, and synthetic EHR
generation.
2 Methodology
We present a structured retrieval-augmented gen-
eration framework tailored for synthesizing clini-
cal progress notes from heterogeneous electronic
health record (EHR) sources. The system is eval-
uated on the task of generating detailed SOAP-
format notes by composing evidence from 15 di-
verse note types in the MIMIC-III dataset (John-
son et al., 2016). Unlike generic RAG pipelines
that treat documents uniformly, our approach mod-
els the hierarchical structure and clinical seman-
tics inherent in multi-source clinical documenta-
tion to produce temporally coherent and clinically
grounded outputs.
2.1 Overview of CLI-RAG
Our framework is centered around a dual-stage,
clinically informed retrieval architecture that ad-
dresses two core challenges in clinical NLP: (1)
fragmented information across heterogeneous note
types, and (2) long, unstructured text that often ex-
ceeds the context window of large language models
(LLMs). To address this, we employ a global re-
trieval step to identify high-value note types using
predefined clinical questions, followed by local
retrieval to extract fine-grained evidence from spe-
cific sections within those notes. Notes are prepro-
cessed and segmented into hierarchical chunks that
preserve document structure, encoded using the all-
mpnet-base-v2 sentence transformer (Reimers and
Gurevych, 2019), and indexed with ChromaDB1
for efficient semantic retrieval. At inference, re-
trieved chunks are de-duplicated, reranked, and
assembled into prompts with structured metadata.
For longitudinal consistency, summaries of prior
visits are optionally incorporated. This modular
1https://www.trychroma.com/pipeline enables the generation of clinically faith-
ful, temporally coherent progress notes. An end-to-
end architecture and flow is shown in Figure 1.
2.2 Preprocessing and Hierarchical Chunking
Raw clinical notes in electronic health records
(EHRs) are rife with inconsistencies: they include
de-identification artifacts, UI-generated noise, vari-
able casing, non-standard bullets, and redundant
line breaks. Before any information retrieval or
language generation step, we perform a robust pre-
processing pipeline aimed at denoising, normaliz-
ing, and structurally realigning clinical text into a
form conducive to semantic understanding. The
preprocessing module executes a cascade of trans-
formations: it replaces Unicode symbols, removes
[**DEID**] tokens while preserving their content
where safe, strips JavaScript artifacts from the user
interface, normalizes whitespace and line breaks,
and collapses common bulleting styles into canoni-
cal formats. Additionally, numerical blocks such
as vitals or lab values are heuristically converted
into key-value format (e.g., WBC→WBC: 17.5 ) to
preserve measurement semantics across lines. Fi-
nally, all-caps section headers are normalized to
title case and missing colons are inserted when
necessary. This stage ensures that textual content,
often spanning multiple note types and documenta-
tion systems, becomes structurally comparable.
Beyond text normalization, we introduce a novel
hierarchical chunking strategy grounded in clin-
ical structure. Instead of naively chunking based
on fixed-length windows, our system respects the
logical organization of clinical documents. Each
note is first segmented at the granularity of high-
level clinical section headers such as History of
Present Illness ,Assessment , orImpression
which reflect discrete semantic zones of documen-
tation. These header-segmented blocks are then
recursively divided into smaller sub-chunks based
on character length thresholds with fixed overlaps,
ensuring downstream retrievability within embed-
ding limits, while maintaining contextual continu-
ity. Unlike flat chunking, this approach preserves
intra-document cohesion, enables section-aware re-
trieval, and dramatically reduces boundary-related
semantic leakage during encoding. For instance,
two separate chunks from an Assessment and
Plan section will be co-located in retrieval space,
while still being individually embeddable. Each
chunk is then indexed along with metadata iden-

Admission 
Notes
Nursing Shift 
NotesNursing 
Other NotesNutrition 
NotesPharmacy 
Notes
ECG Reports Echo ReportsRadiology 
Reports
Miscellaneo
us NotesEvent NotesConsult 
NotesProcedure 
Notes
Transfer 
NotesDischarge 
PlanningDischarge 
SummaryAdmission Phase
Ongoing Hospital StayDischarge Phase
SOAP PROGRESS 
NOTES
Subjective Objective 
Assessment PlanClinical Note
Preprocessing
Hierarchical 
ChunkingGlobal Retrieval
Local RetrievalChroma DB 
Vector StoreContributing 
note typesCLI-RAG System
Prompt 
Construction
LLM CLI-RAG: Clinically 
Informed
RAG15 Clinical Notes 
Types
per visitQueryFigure 1: Overview of the CLI-RAG framework: Fifteen structured clinical note types from each hospital visit
serve as evidence sources. These are preprocessed, hierarchically chunked, and passed through a dual-stage retrieval
pipeline before constructing prompts for large language models to generate structured progress notes.
tifying its patient ID, hospital admission ID, visit
date, originating note type, clinical section, and
local chunk identifier that tracks the count within
the clinical section as this can be further used to or-
der the chunks to maintain original clinical context
ordering during retrieval and this forms the founda-
tional unit of knowledge for downstream retrieval.
Given the variability in clinician documentation
styles, not all segments of a note will cleanly match
a known section. To accommodate this, we imple-
mented a fallback mechanism where unmatched
spans are tagged under an Unlabeled section dur-
ing the chunking stage. These Unlabeled chunks
preserve full contextual information and are treated
as first-class entities, assigned their own chunk_id
for ordering. During retrieval and prompt construc-
tion, any Unlabeled content deemed relevant is
explicitly included and annotated as “More Infor-
mation” within the final prompt structure, signaling
to the language model that these passages may con-
tain clinically useful but unstructured context. This
design choice ensures that no potentially valuable
information is discarded purely due to formatting
inconsistency, and that the model retains access
to all relevant cues needed for generation, even
when embedded in free-text form. This structurally
aligned pre-processing pipeline shown in Figure 3
enables high-fidelity semantic search by mitigatingthe brittleness of syntactic variance across clini-
cal notes. More importantly, it primes the system
for retrieval and generation strategies that are not
merely lexical but clinically aware.
2.3 Multi-Stage Retrieval: Global and Local
Contextualization
To accurately reconstruct clinically grounded
progress notes from fragmented EHR narratives,
our sytem employ a dual-stage retrieval architec-
ture that integrates both global and local contextual
signals. This design reflects a central hypothesis:
the information necessary to generate comprehen-
sive progress notes is scattered across multiple note
types and sections, and must first be surfaced and
contextualized in a structured manner.
Global Retrieval. The first stage in the retrieval
pipeline operates at the visit level and is tasked with
identifying globally relevant clinical information
regardless of where it appears in the patient record.
Given a predefined set of task-driven global clinical
questions (e.g., “What symptoms did the patient
report?” ,“What procedures were performed?” ),
the system performs dense retrieval over all avail-
able notes for a patient visit using sentence-level
embeddings. For each question, a semantic query
embedding is generated and a similarity search is

performed over a vector index built from all chun-
ked clinical documentation corresponding to the
same patient_id andvisit_date . Crucially, no
constraints are placed on the note_type dimension
at this stage allowing the system to retrieve across
all 15 clinical document categories. This wide fil-
tering enables maximal coverage and provides an
unbiased scan of what information is available and
from where.
Retrieved chunks are then passed through a dedu-
plication module, which filters out highly similar
spans using pairwise cosine similarity thresholds
to avoid redundant inclusion of semantically du-
plicated information across note types (e.g., simi-
lar text copied between a nursing and physician
note). To improve retrieval fidelity, we use a
hybrid scoring approach that linearly combines
BM25 lexical relevance and embedding-based co-
sine similarity. This ensures that retrieved evi-
dence is both semantically aligned and lexically
grounded. Finally, symptom-focused reranking pri-
oritizes chunks likely to contain clinically salient
observations using hand-crafted symptom keyword
heuristics.
This global retrieval phase answers two key re-
search questions: (1) What pieces of clinical infor-
mation are most relevant for downstream progress
note generation, regardless of source note type?
and (2) Which note types contribute most fre-
quently to these task-relevant evidence spans? The
system logs metadata about retrieved evidence
chunks, including their note type and section, en-
abling precise tracking of which document types
consistently support specific clinical tasks. This
allows us to both generate structured notes and in-
trospectively audit which sources contribute mean-
ingfully to clinical synthesis.
Local Retrieval. Having identified the most in-
formative note types through global retrieval, our
system proceeds to a second stage: fine-grained
retrieval within individual note types. For each
contributing note type, a curated set of local ques-
tions is queried (e.g., for progress_notes , we ask
“What is today’s clinical assessment?” ,“Were ABG
results reported?” ). This ensures that task-relevant
content is retrieved not only across notes but within
the structural logic of individual note types.
Unlike the global phase, local retrieval restricts
the search space not just by patient_id and
visit_date , but also by note_type . This addi-
tional constraint allows the system to isolate seman-tically dense evidence from specific clinical roles
such as radiologist interpretations, ICU nurse logs,
or attending physician assessments thus enabling
structured extraction tuned to the documentation
practices of each source.
Local retrieval is performed using the same hy-
brid scoring, deduplication, and reranking strategy
as the global phase, but scoped only to documents
within the specified note type. This constrains the
search space and enables highly targeted extraction
of contextual content. Importantly, by posing note-
type-specific questions, we preserve the original
structural intent behind each document (e.g., distin-
guishing subjective narrative in nursing notes from
diagnostic impressions in radiology reports), which
enhances the faithfulness of generation.
Retrieval Oriented Metadata. All re-
trieved chunks whether from global or lo-
cal retrieval are embedded with rich meta-
data, including note_type ,header_name ,
chunk_id_in_header , and temporal tags. These
are used both for reranking and for organizing
content during prompt construction. In cases
where a chunk does not match any known clinical
section from our exhaustive header set, it is
labeled as Unlabeled and later surfaced as “More
Information” during prompt formatting. This
design ensures that valuable but structurally
ambiguous content is not discarded, but rather
explicitly marked for LLM consumption.
Clinical Semantics via Retrieval. This dual-
stage retrieval strategy provides a principled mech-
anism to structure evidence-driven prompting. The
global stage captures macro-level salience by iden-
tifying relevant note types and tasks, while the local
stage enables fine-grained contextualization. To-
gether, they simulate a clinical reasoning workflow:
identifying where to look, and then extracting what
to say. Retrieval thus becomes not only a retrieval
operation, but a semantic alignment step that trans-
lates raw EHR sprawl into task-aligned, evidence-
rich scaffolds for generation. The entire pipeline is
depicted in Appendix 2
Prompt Construction and LLM-Based Gen-
eration. Once relevant context has been re-
trieved through global and local passes, we con-
struct an inference prompt for downstream gen-
eration of structured clinical progress notes. Re-
trieved chunks are sorted by note_type ,section
header , and chunk ID , preserving original doc-

umentation structure and temporal order. Each
chunk includes standardized metadata brackets
(e.g., [progress_notes | Assessment | chunk
2]) to anchor the source and enable interpretability.
Chunks labeled as Unlabeled during preprocess-
ing are explicitly surfaced under a separate section
titled More Information , ensuring they remain ac-
cessible during generation despite lacking section
headers.
A task-specific prompt template guides the LLM
to generate notes in the canonical SOAP format
(Subjective ,Objective ,Assessment ,Plan ), while
enforcing factual grounding. The instruction em-
phasizes clinical fidelity, discourages hallucination,
and encourages structured reasoning using only the
extracted content. When generating longitudinal
notes (for visits beyond the first), the prior note is
summarized and included to help the model dis-
tinguish new findings and reason about temporal
evolution. This modular prompt structure, coupled
with chunk-level ordering and scoped summariza-
tion, allows the model to produce detailed, coher-
ent, and reproducible progress notes aligned with
clinical expectations.
3 Experimental Setup and Evaluation
We evaluate our system across multiple axes lex-
ical fidelity, semantic alignment, structural adher-
ence, and temporal consistency using two open-
source LLMs: LLaMA-3 70B andMistral-7B .
Both models generate SOAP-style progress notes
grounded in multi-note evidence per hospital visit.
We benchmark these outputs against clinician au-
thored progress notes from the MIMIC-III dataset
(Johnson et al., 2016).
3.1 Cohort and Setup
We curated 56 patients with 10–57 visits each, to-
taling 1,108 hospital encounters. For each visit, a
progress note was generated using contemporane-
ous documentation (excluding progress notes). For
longitudinal realism, prior visit summaries were
included for all visits beyond the first.
3.2 Evaluation Dimensions
Lexical Similarity. We compute BLEU and
ROUGE (1/2/L) to capture token-level and n-gram
overlap. As expected in paraphrastic clinical text,
scores are low across both models.
Semantic Alignment. Cosine similarity between
embeddings (from all-mpnet-base-v2 ) mea-sures meaning preservation across generated and
gold notes.
Structural Completeness. Each note is checked
for the canonical SOAP sections. Both models
consistently yield full coverage of Subjective, Ob-
jective, Assessment, and Plan.
Length Control. Length ratio (generated to gold)
evaluates verbosity alignment. Both models pro-
duce slightly longer but controlled outputs.
Temporal Coherence. For longitudinal align-
ment, cosine similarity between adjacent notes
within a patient trajectory is averaged. Generated
notes consistently show stronger temporal consis-
tency than clinician-authored baselines.
LLM Preference Voting. In a blinded quality
check using Mistral-7B, generated notes were pre-
ferred over clinician-authored notes in 96% of
cases across dimensions of structure, fluency, and
completeness.
Metric LLaMA-3 70B Mistral-7B
BLEU 0.0116 0.0004
ROUGE-1 0.2738 0.1783
ROUGE-2 0.0743 0.0312
ROUGE-L 0.1102 0.0676
Semantic Similarity 0.7398 0.7511
SOAP Sections Present 4.00 4.00
Length Ratio 1.1975 N/A
Temporal Consistency (Gold) 0.807 0.9126
Temporal Consistency (Generated) 0.877 0.8248
Alignment Ratio 1.089 0.9038
LLM V oting Preference 96% 93.5%
Table 1: Evaluation metrics across LLaMA-3 70B and
Mistral-7B on 56-patient cohort. Semantic similarity
and structural adherence are consistently strong; tempo-
ral coherence exceeds real notes.
3.3 Interpretation
Despite low lexical overlap a known artifact in clin-
ical generation semantic similarity remains high
across both models, with SOAP section coverage
perfectly retained. LLaMA-3 showed higher tem-
poral alignment (0.877) and richer lexical diversity
(higher ROUGE), whereas Mistral-7B produced
more concise notes with stronger gold alignment
(0.9126 vs. 0.807). Importantly, both models out-
performed real notes in longitudinal coherence.
These results highlight the robustness and trans-
ferability of our retrieval-augmented approach
across LLMs. Regardless of model size or de-
coding variance, the system produces notes that
are clinically structured, semantically faithful, and
temporally aligned.

4 Related Work
Clinical Note Generation. Large language mod-
els (LLMs) have demonstrated strong capabilities
in clinical text generation, including discharge
summaries (Wang et al., 2024; Williams et al.,
2024; Ellershaw et al., 2024), SOAP-format notes
(Soni and Demner-Fushman, 2024), and dialogue-
informed documentation (Biswas and Talukdar,
2024). However, most methods operate over single-
note or single-visit inputs without integrating het-
erogeneous sources or modeling longitudinal clin-
ical reasoning. Encoder–decoder approaches like
ClinicalT5 (Lu et al., 2022) and MedSum (Tang
et al., 2023) target isolated summarization tasks
but lack mechanisms for cross-visit synthesis or
structure-aware generation.
Retrieval-Augmented Generation in Health-
care. Retrieval-Augmented Generation (RAG)
techniques have shown utility in open-domain QA
and summarization by injecting external evidence
(Lewis et al., 2020; Izacard et al., 2022; Guu et al.,
2020). Clinical RAG applications have largely fo-
cused on static resource grounding (Sohn et al.,
2025; Van Veen et al., 2024) or document-level re-
trieval from knowledge bases. Few systems explore
context-aware retrieval from longitudinal EHRs,
and even fewer account for clinical sectioning, doc-
ument metadata, or temporal relevance. Our system
introduces a domain-adapted RAG architecture that
bridges these gaps.
Longitudinal and Multi-source Modeling.
Modeling patient trajectories over time is critical
in clinical NLP but remains underexplored in
generative settings. Prior efforts in structured
disease modeling (Dieng et al., 2019; Isonuma
et al., 2020), patient timeline summarization (Jain
et al., 2022), and temporal QA (Shimizu et al.,
2024) provide valuable insights, but most do not
synthesize visit-specific notes across multiple
modalities. Our approach conditions generation
on temporally evolving patient context, explicitly
incorporating past visit summaries and diverse
note types to improve coherence.
Evaluation of Clinical Generation. Standard
lexical metrics like BLEU and ROUGE remain
common (Lin, 2004), though they often fail to
capture semantic fidelity in paraphrased clinical
outputs. Embedding-based similarity (Reimers
and Gurevych, 2019), structure-level checks (e.g.,SOAP adherence), and temporal alignment across
visits offer more robust evaluation. Few prior
works jointly assess these dimensions; our evalua-
tion protocol is designed to reflect semantic, struc-
tural, and longitudinal fidelity essential for real-
world deployment.
5 Discussion
This work demonstrates how structured retrieval
and longitudinal conditioning can be leveraged to
address fragmentation in electronic health records.
The proposed dual-stage retrieval pipeline com-
bining global question-driven relevance with note-
type-specific local refinement enables the system to
extract clinically meaningful context and generate
complete, structured progress notes. Incorporat-
ing summaries of prior visits facilitates temporal
consistency, reflected in higher alignment scores
than real clinician-authored notes. This suggests
the model not only synthesizes per-visit informa-
tion faithfully but also maintains narrative coher-
ence over time. Beyond generation, this framework
has practical implications for both retrospective
data curation and real-time clinical support back-
filling documentation gaps, augmenting training
data for clinical NLP tasks, or assisting clinicians
in note drafting. Future extensions should explore
factuality calibration, incorporation of multimodal
evidence, and broader generalization to underrepre-
sented specialties. Embedding human-in-the-loop
evaluation will be essential to ensure safety, trust-
worthiness, and integration into real-world work-
flows.
6 Conclusion
We introduce a clinically informed, retrieval-
augmented generation framework for synthesiz-
ing structured progress notes from heterogeneous
EHR sources. By combining hierarchical chunk-
ing, dual-stage retrieval, and longitudinal prompt
conditioning, the system produces outputs that are
semantically aligned, structurally complete, and
temporally consistent.
Experiments on MIMIC-III demonstrate strong
alignment with clinician-authored documentation
and improved coherence across hospital visits.
These results point to the framework’s potential
for enhancing clinical documentation, supporting
downstream reasoning tasks, and enabling the con-
struction of synthetic, high-fidelity longitudinal
EHR narratives.

References
Monica Agrawal, Stefan Hegselmann, Hunter Lang,
Yoon Kim, and David Sontag. 2022. Large language
models are few-shot clinical information extractors.
InProceedings of the 2022 Conference on Empiri-
cal Methods in Natural Language Processing , pages
1998–2022, Abu Dhabi, United Arab Emirates. Asso-
ciation for Computational Linguistics.
Nate C Apathy, Allison J Hare, Sarah Fendrich, and
Dori A Cross. 2022. Early changes in billing and
notes after evaluation and management guideline
change. Annals of internal medicine , 175(4):499–
504.
Anjanava Biswas and Wrick Talukdar. 2024. Intelli-
gent clinical documentation: Harnessing generative
ai for patient-centric clinical note generation. Inter-
national Journal of Innovative Science and Research
Technology (IJISRT) , page 994–1008.
Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei.
2019. The dynamic embedded topic model.
Simon Ellershaw, Christopher Tomlinson, Oliver E
Burton, Thomas Frost, John Gerrard Hanrahan,
Danyal Zaman Khan, Hugo Layard Horsfall, Mol-
lie Little, Evaleen Malgapo, Joachim Starup-Hansen,
Jack Ross, Martinique Vella-Baldacchino, Kawsar
Noor, Anoop D. Shah, and Richard Dobson. 2024.
Automated generation of hospital discharge sum-
maries using clinical guidelines and large language
models. In AAAI 2024 Spring Symposium on Clinical
Foundation Models .
Changchang Fang, Yuting Wu, Wanying Fu, Jitao Ling,
Yue Wang, Xiaolin Liu, Yuan Jiang, Yifan Wu, Yix-
uan Chen, Jing Zhou, Zhichen Zhu, Zhiwei Yan, Peng
Yu, and Xiao Liu. 2023. How does chatgpt-4 preform
on non-english national medical licensing examina-
tion? an evaluation in chinese language. PLOS Digi-
tal Health , 2:e0000397.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. Realm: retrieval-
augmented language model pre-training. In Proceed-
ings of the 37th International Conference on Machine
Learning , ICML’20. JMLR.org.
Masaru Isonuma, Junichiro Mori, Danushka Bollegala,
and Ichiro Sakata. 2020. Tree-Structured Neural
Topic Model. In Proceedings of the 58th Annual
Meeting of the Association for Computational Lin-
guistics , pages 800–806, Online. Association for
Computational Linguistics.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2022. Atlas: Few-shot learning with retrieval
augmented language models.
Raghav Jain, Anubhav Jangra, Sriparna Saha, and Adam
Jatowt. 2022. A survey on medical document sum-
marization.Alistair E. W. Johnson, Tom J. Pollard, Lu Shen, Li-
Wei H. Lehman, Mengling Feng, Mohammad Ghas-
semi, Benjamin Moody, Peter Szolovits, Leo An-
thony Celi, and Roger G. Mark. 2016. Mimic-iii,
a freely accessible critical care database. Scientific
Data , 3(1):160035.
Thomson Kuhn, Peter Basch, Michael Barr, Thomas
Yackel, and Medical Informatics Committee of the
American College of Physicians*. 2015. Clinical
documentation in the 21st century: executive sum-
mary of a policy position paper from the american
college of physicians. Annals of internal medicine ,
162(4):301–303.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Proceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems , NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. In Text Summariza-
tion Branches Out , pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Qiuhao Lu, Dejing Dou, and Thien Nguyen. 2022. Clin-
icalT5: A generative language model for clinical
text. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2022 , pages 5436–5443,
Abu Dhabi, United Arab Emirates. Association for
Computational Linguistics.
Arie Markel. 2010. Copy and paste of electronic health
records: a modern medical illness. The American
journal of medicine , 123(5):e9.
S. M. Meystre, G. K. Savova, K. C. Kipper-Schuler,
and J. F. Hurdle. 2008. Extracting information from
textual documents in the electronic health record:
a review of recent research. Yearbook of Medical
Informatics , pages 128–144. Systematic review.
OpenAI. 2023. GPT-4 technical report. CoRR ,
abs/2303.08774.
Bethany Percha. 2021. Modern clinical text mining:
A guide and review. Annual Review of Biomedical
Data Science , 4(V olume 4, 2021):165–187.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. In Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP) , pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
Adam Rule, Steven Bedrick, Michael F Chiang, and
Michelle R Hribar. 2021. Length and redundancy
of outpatient progress notes across a decade at an

academic medical center. JAMA Network Open ,
4(7):e2115334–e2115334.
Seiji Shimizu, Lis Pereira, Shuntaro Yada, and Eiji Ara-
maki. 2024. QA-based event start-points ordering
for clinical temporal relation annotation. In Pro-
ceedings of the 2024 Joint International Conference
on Computational Linguistics, Language Resources
and Evaluation (LREC-COLING 2024) , pages 13371–
13381, Torino, Italia. ELRA and ICCL.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara
Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen
Pfohl, Perry Payne, Martin Seneviratne, Paul Gam-
ble, Chris Kelly, Abubakr Babiker, Nathanael Schärli,
Aakanksha Chowdhery, Philip Mansfield, Dina
Demner-Fushman, Blaise Agüera y Arcas, Dale Web-
ster, Greg S. Corrado, Yossi Matias, Katherine Chou,
Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Ra-
jkomar, Joelle Barral, Christopher Semturs, Alan
Karthikesalingam, and Vivek Natarajan. 2023. Large
language models encode clinical knowledge. Nature ,
620(7972):172–180. Epub 2023 Jul 12.
Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon
Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim,
and Jaewoo Kang. 2025. Rationale-guided retrieval
augmented generation for medical question answer-
ing. In Proceedings of the 2025 Conference of the
Nations of the Americas Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers) , pages 12739–
12753, Albuquerque, New Mexico. Association for
Computational Linguistics.
Sarvesh Soni and Dina Demner-Fushman. 2024. To-
ward relieving clinician burden by automatically gen-
erating progress notes using interim hospital data.
Xiangru Tang, Andrew Tran, Jeffrey Tan, and Mark
Gerstein. 2023. GersteinLab at MEDIQA-chat 2023:
Clinical note summarization from doctor-patient con-
versations through fine-tuning and in-context learn-
ing. In Proceedings of the 5th Clinical Natural
Language Processing Workshop , pages 546–554,
Toronto, Canada. Association for Computational Lin-
guistics.
Dave Van Veen, Cara Van Uden, Louis Blanke-
meier, Jean-Benoit Delbrouck, Asad Aali, Christian
Bluethgen, Anuj Pareek, Malgorzata Polacin, Ed-
uardo Pontes Reis, Anna Seehofnerová, Nidhi Ro-
hatgi, Poonam Hosamani, William Collins, Neera
Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Ga-
tidis, John Pauly, and Akshay S. Chaudhari. 2024.
Adapted large language models can outperform med-
ical experts in clinical text summarization. Nature
Medicine , 30(4):1134–1142.
Michael D Wang, Raman Khanna, and Nader Najafi.
2017. Characterizing the source of text in elec-
tronic health record progress notes. JAMA internal
medicine , 177(8):1212–1213.Sheng Wang, Zihao Zhao, Xi Ouyang, Tianming Liu,
Qian Wang, and Dinggang Shen. 2024. Interactive
computer-aided diagnosis on medical image using
large language models. Communications Engineer-
ing, 3(1):133.
Christopher Y .K. Williams, Jaskaran Bains, Tianyu
Tang, Kishan Patel, Alexa N. Lucas, Fiona Chen,
Brenda Y . Miao, Atul J. Butte, and Aaron E. Korn-
blith. 2024. Evaluating large language models for
drafting emergency department discharge summaries.
medRxiv .
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, et al. 2024. Qwen2
technical report. arXiv preprint arXiv:2407.10671 .

A Ethics Statement
Our work introduces CLI-RAG, a clinically in-
formed retrieval-augmented generation system
aimed at generating structured and contextually
grounded patient progress notes. While our con-
tributions are technical, we acknowledge the high-
stakes nature of clinical applications. To mitigate
ethical risks:
•Data Privacy: All data used is de-identified
and complies with HIPAA and related privacy
standards. Our experiments are conducted on
publicly available datasets (e.g., MIMIC-III),
which are approved for research use.
•Clinical Safety: CLI-RAG is designed as a
research prototype and is not intended for
deployment in real-time clinical decision-
making without rigorous validation and over-
sight. It must not be used as a substitute for
expert clinical judgment.
•Bias and Fairness: We recognize that pre-
trained language models may inherit biases
present in clinical notes, documentation styles,
or healthcare systems. We take care to surface
relevant content transparently and plan future
work to evaluate representation fairness across
subpopulations.
•Transparency: CLI-RAG prioritizes inter-
pretability through structured chunk retrieval
and evidence attribution. All outputs are trace-
able to their source chunks to support clinician
review.
We commit to continuing responsible AI prac-
tices, including collaborating with clinicians, incor-
porating human feedback, and aligning with health-
care regulatory norms before any deployment.
B Broader Impact Statement
CLI-RAG demonstrates the promise of retrieval-
augmented generation systems in clinical nat-
ural language processing (NLP), particularly
for assisting clinicians in generating structured,
evidence-grounded summaries from complex med-
ical records. We introduce a modular, interpretable,
and clinically grounded RAG system that has the
potential to improve clinical documentation, health-
care research, and health informatics. The system is
designed to reduce the cognitive and administrativeburden placed on healthcare providers by synthe-
sizing raw clinical notes into coherent, structured
progress notes. In environments where physicians
spend significant time on documentation often at
the cost of patient interaction such systems can
serve as productivity aids, improving both clinician
efficiency and documentation quality.
Industrial Applications: In industrial and com-
mercial healthcare settings, CLI-RAG could be in-
tegrated into electronic health record (EHR) plat-
forms to assist with real-time note summariza-
tion, longitudinal chart review, and audit-ready
report generation. Additionally, its architecture
lends itself to medical billing support and clinical
coding, where structured output aligned with re-
trieved evidence can enhance compliance and trans-
parency. This is particularly valuable in hospitals
and telemedicine platforms facing documentation
backlogs or resource constraints.
•Clinical Documentation Support: CLI-
RAG can enhance electronic health record
(EHR) systems by suggesting high-quality
progress notes grounded in retrieved content,
reducing clinician burnout and improving doc-
umentation quality.
•Clinical Audit and Coding: Automated sum-
marization aligned with structured evidence
may support downstream tasks like medical
coding, insurance audits, or billing documen-
tation.
•Clinical Research Workflows: Researchers
can use CLI-RAG to rapidly extract relevant
information across longitudinal records, en-
abling cohort construction, phenotype extrac-
tion, and retrospective studies.
Research Contributions: For clinical re-
searchers, CLI-RAG presents a tool for retrospec-
tive data extraction and structured cohort analysis.
By automating the summarization of complex pa-
tient trajectories, researchers can focus on higher-
level tasks such as identifying phenotypes, evalu-
ating treatment responses, and constructing case
studies. The system’s use of fine-grained chunk
retrieval allows researchers to validate summaries
against traceable source evidence, supporting re-
producibility and methodological rigor.
•Structured Clinical RAG Paradigms: CLI-
RAG provides a blueprint for structured, inter-
pretable RAG in high-stakes domains, bridg-

ing retrieval-based QA and summarization in
a modular pipeline.
•Evaluation Frameworks: By enabling
chunk-level traceability, CLI-RAG lays the
foundation for more nuanced intrinsic and ex-
trinsic evaluations of clinical language mod-
els.
From an NLP perspective, CLI-RAG advances the
field by demonstrating how retrieval-augmented
generation can be tightly coupled with domain-
specific structure and explainability. It bridges tra-
ditionally separate tasks such as information re-
trieval, clinical summarization, and question an-
swering. The framework also sets the stage for new
evaluation paradigms, emphasizing interpretability
and attribution over black-box generation.
Limitations and Responsible Use: While our
system shows promise, it must be rigorously val-
idated across diverse clinical environments. Fur-
ther research is needed to assess generalizability,
temporal robustness, and human-AI collaboration
in clinical settings. We envision CLI-RAG as a
step toward safer, more explainable AI systems
for healthcare. We view CLI-RAG not as a final
product, but as a foundational system that invites in-
terdisciplinary development, robust clinical testing,
and ethical guardrails for future healthcare NLP
applications.

Raw Multi -Note Input 
(15 Note types)
Diverse structures, variable 
length, highly -fragmented 
clinical narrativesPreprocessing + 
Hierarchical Chunking
Generic Clinical 
T ext Cleaning
Note Specific 
Cleaning
Section Header 
based Chunking
Semantic 
Chunking per 
SectionCleaned Notes
[note type | section 
header | chunk number | 
note chunk]Vector Store
Chroma DBEmbedded 
using
sentence 
transformerLocal Retrieval Global Retrieval
globally 
relevant clinical 
chunks across 
all notes
Note Specific clinical questions
`PROGRESS NOTE: What is today’s 
clinical assessment?`Semantic MatchingDeduplication on 
retrieved chunks across 
different note typesHybrid based 
Reranking using 
BM25 + Cosine 
SimilarityContributing
Note Typeslocal relevant note 
type specific 
chunks
Semantic MatchingHybrid based 
Reranking using 
BM25 + Cosine 
Similarity
T ask driven global clinical questions
`What symptoms did the patient report?`Deduplication on 
retrieved chunks
Figure 2: CLI-RAG Architecture: End-to-end flow diagram showing how structured clinical note types are processed
including cleaning, chunking, global and local retrieval, and final LLM generation.
Chief Complaint: Hypoxia
80 y/o F with a history of COPD on 
home oxygen, lung cancer, 
hypertension, diabetes, morbid 
obesity who presented today for 
ERCP...
Post ERCP she was extubated and 
became tachypneic with oxygen 
saturation dropping into 80s, pink 
frothy sputum...
Labs: ECG ischemic changes, CXR 
pleural effusion and cardiomegaly...
Disposition: ICU.
Raw Text from Admission Notes
Chief Complaint: Hypoxia
HPI: 80 y/o F with COPD, lung cancer, 
morbid obesity. Post ERCP: 
tachypnea, frothy sputum. ECG: 
ischemia. CXR: effusion. ASA, 
furosemide given. BP 90s.
Medications: Lipitor, Citalopram, 
Lasix...
Vitals: HR: 86, BP: 104/58, SpO2: 
94
\
%
Assessment: COPD, Hypoxia, 
Diabetes, ICU monitoring, antibiotics, 
ventilator weaning
Cleaned Text after 
preprocessing
[admission_notes | Chief Complaint | 
Chunk 1] Hypoxia
[admission_notes | HPI | Chunk 1] 80 
y/o F ... Post ERCP deterioration. 
ECG/CXR findings.
[admission_notes | Vitals | Chunk 1] 
HR: 86, BP: 104/58, SpO2: 94%
[admission_notes | Assessment And 
Plan | Chunk 1] # COPD, # Hypoxia, # 
Diabetes
Plan: Cultures, antibiotics, ventilation
Support
After Hierarchical Chunking
Figure 3: Transformation of a raw free-text note into structured format: Preprocessing eliminates noise, and
hierarchical chunking extracts meaningful sections for downstream retrieval.

Note Type: Admission Notes
Text: 80 y/o F with a history of COPD on home oxygen, 
lung cancer, hypertension, diabetes, morbid obesity 
who presented today for ERCP for RUQ abdominal pain 
of 2 months duration with u/s showing evidence of CBD 
dilation…..
Note Type: Radiology Report
Text: [**2103
 -
6
-
18**] 8:21 AM ERCP BILIARY ONLY BY GI 
UNIT Clip # [**Clip Number (Radiology) 36334**]
Reason: Please review ERCP images done [**6
 -
18**]
Admitting Diagnosis: BILE DUCT 
OBSTRUCTION
 \
ERCP…..
Note Type: Event Notes
Text: 80 yo female patient with history of RUQ pain now 
sent to ERCP for evaluation.
  
She had procedure 
without event (sphincterotomy and sludge drained from 
the biliary tree) but following extubation had evolution 
of significant hypoxemia and ….
Question Q: What symptoms did the patient report today?
CLI
-
RAG
Global 
Retrieval[admission_notes | HPI | chunk 1] 80 y/o F with a history of COPD on home oxygen, 
lung cancer, hypertension, diabetes, morbid obesity who presented today for ERCP for 
RUQ abdominal pain of 2 months duration with u/s showing evidence of CBD dilation. She 
was electively intubated given her history of severe COPD. Post ERCP she was extubated 
and was on her way to the recovery room when she became tachypnoeic and oxygen 
saturation dropped into 80s, she also was having a good amount of pink frothy sputum, of 
note she only received 800cc of fluid during the ERCP. An ECG was obtained which was 
initially concerning for ischemic changes. A CXR showed left pleural effusion and 
cardiomegaly. She was re -intubated given her oxygen saturations, no blood gas was 
drawn prior to re -intubation. She was given ASA, furosemide and labs were sent including 
cardiac enzymes. Of note she was placed on propofol and her BP was in the 90s. An 
arterial line was also placed. She was transiently place on neo synephrine which was 
stopped upon arrival
[radiology_reports | Final Report | chunk 1] HISTORY: 80 -year -old female 
with recurrent brain upper quadrant pain and dilated CBD.  No prior studies 
available for comparison.  ERCP: Eight spot fluoroscopic images were obtained 
without a radiologist present and submitted for review. Images demonstrate 
cannulation of the pancreatic duct and common bile duct, with contrast 
injection. Pancreatic duct is normal in caliber. However, the CBD is diffusely 
dilated, without a focal filling defect identified.  IMPRESSION: Diffuse dilatation 
of the CBD, without a focal filling defect.  2103 -6-18 2:35 PM CHEST (PORTABLE 
AP) Clip # Clip Number (Radiology) 36988 Reason: check ettube, r/o pulm 
edema  Hospital 2 MEDICAL CONDITION: 80 year old woman with ,
[event_notes | Unlabelled | chunk 1] Clinician: Attending 80 yo female patient with 
history of RUQ pain now sent to ERCP for evaluation. She had procedure without event 
(sphincterotomy and sludge drained from the biliary tree) but following extubation had 
evolution of significant hypoxemia and required emergent re -intubation. She was intubated 
for procedure given history of significant pulmonary disease. Of note. Pt with history of 
COPD, obesity and lung cancer In the setting of post procedure extubation Patient with 
hypoxemia on usual FIO2 (85 \%) -She had pleural effusion noted -She had ECG --Anterior 
injury with loss of Q -waves and non -specific T -wave inversions -With persistent tachypnea 
patient re -intubated for recurrent hypoxemia. The source of the hypoxemia acutely post 
operatively is most likely a combination of background lung disease with severe COPD and 
she may well have had some atelectasis in the setting of intubation and obesity. Post 
operatively she had tachpnea which in the setting of chronic obstructive Clinical Notes NFigure 4: Example of global retrieval in CLI-RAG: Given a clinical question about symptoms, relevant chunks are
retrieved from multiple note types (admission, event, radiology) across a patient visit.
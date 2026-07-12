# Conversational Retrieval and On-the-Fly Knowledge Modeling of Historical Penitentiary Repression Records

**Authors**: Paula Font Solà, Adrià Molina Rodríguez, Josep Lladós

**Published**: 2026-07-09 13:18:52

**PDF URL**: [https://arxiv.org/pdf/2607.08459v1](https://arxiv.org/pdf/2607.08459v1)

## Abstract
Recent developments in digital libraries increasingly favor conversational and natural language access to information through Retrieval-Augmented Generation (RAG). Although these approaches are effective for extractive tasks grounded in individual records, they remain limited in their ability to interpret document collections holistically and to incorporate expert knowledge dynamically. In this article, we present a document analysis system designed for the management of historical digital libraries that supports on-the-fly knowledge modeling. The system is equipped with the capability to store facts produced either by expert archivists or derived from document retrieval processes within a graph-based structure. Through continuous professional interaction, the system can retrieve information not only from primary sources such as documents, but also from previously modeled knowledge, with the graph-based index acting as a memory for the language model to access. This enables increasingly complex queries involving long-term dependencies across documents, link discovery, and the integration of expert knowledge that may not be explicitly present in the original sources. As a result, the proposed approach facilitates the generation of richer and more comprehensive information.

## Full Text


<!-- PDF content starts -->

Conversational Retrieval and On-the-Fly
Knowledge Modeling of Historical Penitentiary
Repression Records
Paula Font Solà, Adrià Molina RodriguezB, Josep Lladós Canet
Computer Vision Center, Universitat Autònoma de Barcelona
paula.fonts@autonoma.cat, {amolina, josep}@cvc.uab.cat
Abstract.Recent developments in digital libraries increasingly favor
conversationalandnaturallanguageaccesstoinformationthroughRetrieval-
Augmented Generation (RAG). Although these approaches are effective
for extractive tasks grounded in individual records, they remain lim-
ited in their ability to interpret document collections holistically and
to incorporate expert knowledge dynamically. In this article, we present
a document analysis system designed for the management of historical
digital libraries that supports on-the-fly knowledge modeling. The sys-
tem is equipped with the capability to store facts produced either by
expert archivists or derived from document retrieval processes within a
graph-based structure. Through continuous professional interaction, the
system can retrieve information not only from primary sources such as
documents,butalsofrompreviouslymodeledknowledge,withthegraph-
based index acting as a memory for the language model to access. This
enables increasingly complex queries involving long-term dependencies
across documents, link discovery, and the integration of expert knowl-
edge that may not be explicitly present in the original sources. As a
result, the proposed approach facilitates the generation of richer and
more comprehensive information.
Keywords:Document Analysis Systems·Historical Documents
1 Introduction
In recent years, historical document analysis systems have shifted from indices
and enumerations of content to increasingly conversational and adaptive inter-
faces. This evolution is not merely a superficial redesign of how users access
archives and portals; it represents a deeper transformation in how archival data
is organized, interpreted, and communicated. The advent of Generative AI has
accelerated this transition, transforming both how users interact with systems,
using natural language queries and conversational exploration, and how results
are synthesized and delivered. Instead of static document lists, modern systems
generate summaries and reveal relationships in archival collections. Although it
may be tempting to attribute this transformation solely to the rapid develop-
ment of Large Language Models (LLMs) and Retrieval-Augmented Generation
arXiv:2607.08459v1  [cs.IR]  9 Jul 2026

2 P. Font et.al
(RAG), it is important to recognize that archival science had already begun
moving beyond document enumeration toward more holistic paradigms. The In-
ternational Council on Archives’s Records in Contexts (RiC) framework [17]
exemplifies this shift, emphasizing relationships, provenance, and contextual in-
terdependencies across collections rather than isolated descriptions of individual
records.
From the perspective of Document Analysis Systems, however, sustaining
such a rich and interconnected framework raises practical challenges. The con-
struction and maintenance of contextual knowledge representations (e.g. knowl-
edge graphs) have traditionally relied on the manual labor of experienced his-
torians and archivists. These professionals are expected not only to identify the
sourceandcontextofagivendocument,butalsotointerpretitscontents,resolve
ambiguities, trace links across collections, and determine the role each record
playswithinabroaderhistoricalnarrative.Whilethisworkisessential,itistime-
consuming, difficult to scale, and inevitably constrained by available resources.
Thischallengebecomesparticularlycriticalinlargehistoricalarchivescontaining
heterogeneous, partially digitized, and often degraded materials, where access to
personalized, well-focused, and detailed information is increasingly the main ex-
pectation of researchers and the general public alike. Users no longer seek merely
to retrieve documents; they seek explanations, connections, and synthesized in-
sights that cut across fonds, time periods, and institutional boundaries. Meeting
theseexpectationsrequiressystemscapablenotonlyofretrieval,butalsoofmod-
eling and reasoning over knowledge at the collection level. Recent GraphRAG
approaches attempt to address multi-hop reasoning and long-term dependencies
by integrating graph-based memory structures with language models. However,
many of these systems implicitly assume that documents are natively digital,
cleanly structured, or at least sufficiently well-processed for LLMs to operate
robustly over them. In the context of historical archives, this assumption rarely
holds. Documents are frequently handwritten or degraded, transcriptions are
noisy, and metadata may be incomplete or inconsistent. As a result, the reliabil-
ity of downstream reasoning critically depends on the quality and uncertainty
management of the transcription and extraction stages.
In this work, we address these limitations through two main contributions.
First, we introduce a methodology for the on-the-fly creation and continuous re-
finement of a graph-based knowledge structure during conversational interaction
(see Figure 1). The system stores facts derived either from document retrieval
or from expert input into a persistent graph that acts as structured memory
for the language model. This graph operates at both a local (conversation-level)
andglobal(collection-level)scale,enablingtheaccumulationofknowledgeacross
sessions and supporting increasingly complex queries involving multi-document
reasoning, link discovery, and contextual interpretation. Second, we propose a
transcription enhancement pipeline based on an ensemble of OCR and LLM cor-
rections. Rather than relying on a single automatic transcription, we combine
multiple hypotheses and exploit language model completion to reconstruct seg-
ments with high uncertainty. By explicitly detecting fragments that can be plau-

On-the-Fly Knowledge Modeling 3
Uses LLM
Entire 
Document
[term1, 
term2, ...]User 
QueryText
ChunksEmbedding Model
Selection of
key terms
Final Answer
GenerationEmbedding SpaceRetrieval
vectorizeindexing
searching
promptNatural 
Language
ResponseTop-K
ChunksRe-ranked
Chunks
Cross-Encoder
Uses LLMSpecific Graph
Global Graph
Natural 
Language
ResponseTriplet Extraction
Node classificationMatch to 
Existing Nodes
Node decomposition 
using NERTriplet Filtering 
and Refinement
Fig.1:Graph-basedknowledgemodelingframework.Extractedtriplesbuildlocal
graphs for conversation context and a global graph linking salient entities across
documents.
sibly reconstructed and flagging those that cannot, the system reduces the risk
of propagating hallucinated or low-confidence content into the knowledge layer.
By combining uncertainty-aware transcription with dynamic graph construc-
tion, our approach bridges the gap between document-centered RAG pipelines
and knowledge-centered archival frameworks. In doing so, it provides a scalable
and interactive pathway toward holistic historical document analysis, aligned
both with contemporary advances in language modeling and with long-standing
principles in archival science. For the experimental validation we use a cor-
pus of court-martial sentencing documents from the Spanish Civil War period
(1937–1940),
2 Related Work
Researchers have long worked on historical document analysis, developing many
methods over time. Early efforts focused on Optical Character Recognition
(OCR) and Handwritten Text Recognition (HTR), which formed the core of
processing pipelines. Still, preprocessing steps like binarization, layout analysis,
and text line detection were essential before applying OCR/HTR. The output
(plain transcribed text) then required interpretation and information extraction,
prompting the use of Named Entity Recognition (NER) and broader Natural
Language Processing (NLP) techniques [21].
NER [7] can be considered one of the first and most crucial processing steps
for information extraction. It identifies key entities that serve as referential an-
chors,guidingtextinterpretation.Yetthisapproachhasitslimitations,namelya

4 P. Font et.al
lack of flexibility: it offers little margin for information retrieval as well as a lim-
ited context understanding. The immediate solution was using the NER output
as a starting point for Information Extraction (IE) through hand-crafted rule-
based and ontology-driven approaches, for tasks such as entity linking and event
extraction. With the rise of Transformers, models such as BERT [5] have been
employed to learn semantic patterns, enabling deep contextual understanding
and relation extraction. NLP has also greatly benefited from LLMs, especially in
retrieval tasks. While models like BERT generate structured outputs, they often
lack accessibility. In contrast, LLMs offer a more effective solution for conversa-
tional retrieval due to their ability to produce coherent, context-aware responses
in natural language. Despite their popularity, LLMs still face challenges, most
notably,theissueof’hallucinations’andalackofin-domainorspecializedknowl-
edge. To address these limitations, Retrieval-Augmented Generation (RAG) was
introduced. RAG enhances LLMs by retrieving specific, relevant information to
use as contextual input for a query. This allows the LLM [14] to ground its
responses in external knowledge, thereby reducing hallucinations and incorpo-
rating domain-specific insights. Advancements in RAG include the use of agents
to optimize different components of the system [11], and the development of hy-
brid retrieval strategies [19] such as sentence-window retrieval and parent-child
chunking methods. The emerging challenge lies in effectively retrieving and in-
terpreting complex events defined by who did what to whom, when, and where.
Addressing such queries requires a deeper understanding and grounding answers
in structured knowledge. To this end, GraphRAG [6] has been proposed: a fusion
of RAG with Knowledge Graphs (KGs), enabling the system to retrieve informa-
tion from structured sources and present it as contextual input for generation.
The state of the art points toward multi-stage pipelines for historical doc-
ument processing combined with hybrid RAG/Knowledge Graph systems for
information retrieval. This article contributes by integrating these cutting-edge
approaches into a single application, providing a novel tool for exploring histor-
ically significant archival domains through modern AI techniques.
3 Method
The main objective of our methodology is to transform unstructured and often
degraded historical documents into a structured,queryableknowledge base that
supports both local document analysis and global information synthesis. In con-
trast to standard RAG architectures, we propose a threefold pipeline which, in
conjunction, provides to the system the capacity to accumulate facts within a
knowledge graph representation model.
The implementation of our complete system is organized into three primary
components: the text extraction pipeline (Section 3.1), an advanced RAG-based
retrieval system (Section 3.2), and a system for the dynamic generation of a
Knowledge Graph and its subsequent retrieval (Section 3.3).

On-the-Fly Knowledge Modeling 5
Original ImageBinarized ImageOriginal Image Fragments
Binarized Image Fragmentsx8Initial
Transcription
Binarization
Layout
AnalysisPreprocessingTesseract OCR
layout-aware
layout-unaware
x10Final
Transcription
x8Initial
Transcription Character-Level
HarmonizationWord-Level
HarmonizationVersion 
FilteringRefinement
LLM
Fig.2: Two-stage transcription pipeline:Preprocessinggenerates eight OCR
candidatesviabinarizationandlayoutanalysis;Refinementfilters,harmonizes,
and applies LLM-based correction.
3.1 The Text Extraction Pipeline
Ourmulti-stagepipeline,illustratedinFigure2,transcribeshistoricaldocuments
with the highest possible fidelity. It begins with pre-processing to generate mul-
tiple OCR candidates, followed by a sophisticated two-step process for consensus
and refinement.
Initial analysis of the document corpus highlighted two significant obstacles
requiringarobustdocumentanalysispipeline.Firstly,widespreadink-fadingand
low document contrast frequently caused standard OCR engines to fail, yield-
ing incomplete or empty text. Secondly, the documents exhibit a variable lay-
out, intermittently adopting a two-column format that confounds conventional
reading-order algorithms. This structural variability was critical to overcome, as
failing to isolate these columns (which contain key subject names) would disrupt
the reading order.
To address these challenges, we employ a multi-candidate strategy detailed
in Figure 2. We generate four base images using distinct binarization techniques
(original, global threshold, adaptive threshold, andHi-SAM[25]). These inputs
feed two parallel processing streams: aDirect Paththat extracts text from
the raw binaries, and aLayout-Aware Paththat utilizes Surya [20] to enforce
correct reading order. This yields an ensemble of eight diverse text candidates
per document.
Thisdual-pathapproachisdesignedformaximumrobustness,withthedirect
path serving as a crucial failsafe against potential errors in the layout analysis it-
self. By generating this diverse ensemble of eight candidates, we ensure complete
capture of all information from the original text.

6 P. Font et.al
Given the complexity of these inputs, the choice of OCR engine was critical.
We selected Tesseract after our preliminary analysis showed that it effectively
handles the high noise and variable alignment where other solutions failed [21].
Final Version GenerationThe raw transcriptions from the OCR stage, while
valuable, contain noise and inconsistencies. To produce a single, high-fidelity
text, we employ a multi-stage harmonization and refinement process, as shown
in the Refinement module of Figure 2.
First, the raw OCR outputs are consolidated. We begin by filtering the eight
raw transcriptions, discarding outliers based on heuristics such as word count
and line length distributions.
The layout-segmented transcriptions are separated from the others and gen-
erated a consensus text within each group using a character-level voting system
inspired by [12]. The system performs a Multiple Sequence Alignment (MSA) to
align the texts character by character. At each aligned position, a majority vote
determines the final character. This approach is highly effective for correcting
common OCR errors, particularly single-character substitutions.
Next, the two baseline transcriptions are passed to an LLM for advanced
error correction. Based on extensive evaluation, the ‘microsoft/phi-4‘ model [1]
was selected. To mitigate hallucinations and ensure consistency, each baseline is
refined multiple times by the LLM, generating a set of 10 high-quality candidate
texts.
Finally, all LLM-refined candidates are reconciled using a word-based voting
system (similar to the previous approach but operating on entire words). This
step is crucial for preserving the semantic, word-level corrections introduced by
the LLM. The final consensus yields a single, highly accurate transcription that
best represents the most probable content of the original document.
3.2 The Advanced RAG
Once high-fidelity transcriptions are secured, we build a RAG system to answer
user queries grounded in the historical documents, following a three-stage design
(Figure 1).
Offline Indexing.Document transcriptions are split into semantically coher-
ent chunks following legal section headers likeRESULTANDO(resulting, referring
to different facts) orCONSIDERANDO(considering, legal considerations or assess-
ments of legal arguments). Then the snippets of text are further segmented into
overlapping 50–100 word passages. Each chunk is encoded into a dense vector
usingmsmarco-bert-base-dot-v5from SentenceTransformers [22], selected for
its strong asymmetric semantic search performance.
Online Retrieval.At query time, aphi-4-powered agent extracts named en-
tities and reformulates the query into precise search terms. The refined query
is encoded and matched against all chunk embeddings via dot-product similar-
ity, retrieving the top-20 candidates. These are re-ranked by a cross-encoder
(ms-marco-MiniLM-L6-v2), which jointly scores query–chunk pairs for precise
relevance filtering. The top-ranked chunks then vote for a source document, and
the most frequently cited document is selected as context.

On-the-Fly Knowledge Modeling 7
Answer Generation.A prompt combining the original query and retrieved
context instructs the model to rely exclusively on the provided evidence, return-
ing a negative response if context is insufficient to prevent hallucinations.
3.3 Dynamic Knowledge Graph
While the RAG system described in Section 3.2 is effective at retrieving and
summarizing information from individual documents, it is limited in its ability
toanswercomplex,relationalqueriesthatrequiresynthesizingfactsacrossmulti-
ple documents. To overcome this limitation, we introduce a dynamic Knowledge
Graph (KG) that captures and connects entities and their relationships. The
graph is not pre-built but rather dynamically populated based on user interac-
tions. A key novelty of our approach is that the KG acts as a persistent, evolving
system memory. Unlike standard RAG systems that process each query in iso-
lation, our graph allows the system to ‘remember’ and accumulate knowledge
across multiple user interactions and documents, creating a structured founda-
tion for long-term information synthesis.
Graph ConstructionThe KG is populated by converting the natural language
answers generated by the RAG system into structured (subject, predicate, ob-
ject) triplets. The core extraction is performed by an LLM agent. We use a
dedicated prompt that instructs the model to identify concise, meaningful rela-
tionships within the text and return them in a structured list format.
The raw triplets from the LLM are noisy and require significant processing,
as illustrated in Figure 1. Predicates that are too long are discarded to ensure
relations are meaningful. To prevent duplicate nodes from minor text variations
(like capitalization or accents), we find string similarities using normalized Lev-
enshtein distance. New entities are matched against existing nodes, and if a
similar enough one is found, the existing node is reused.
Nodes that are overly long or contain conjunctions (like "Juan y Sara") are
decomposed.WeidentifyconstituentnamedentitiesusingSpaCy’sNER[13]and
create a chain of new triplets that explicitly link these components, preserving
their original relationships. Before adding the triplets to the graph, we check
for duplicates using the string distance method described above, preventing the
accumulation of redundant or poorly structured nodes.
This multi-step refinement process transforms raw LLM output into a clean,
normalized, and well-structured set of triplets ready for integration into the
graph.
Local and Global GraphsAkeydesignchoiceinoursystemisthemaintenance
oftwodistinctbutinterconnectedknowledgegraphsasdepictedinthefinalstage
of our creation pipeline (Figure 1).
We maintain two complementary graph structures. Alocal graphis built
per conversation, capturing knowledge extracted from a single document and
serving as a structured, queryable representation of its content. Aglobal graph
connects all conversations and documents, enabling cross-document synthesis.
The key challenge is merging knowledge across sources without conflating
document-specific entities (e.g., the node“delito”must represent distinct crimes

8 P. Font et.al
Uses LLM
prompt
Final Answer
Generation
Natural 
Language
ResponseQuery 
Key TermsExisting
Node Labels
Single-Entity Queries
Multi-Entity QueriesAdaptive 
Search
Relevant
Triplets
User 
Query
Node
Identification
Fig.3: Graph retrieval pipeline: query nodes are identified, relevant triplets re-
trieved via adaptive search, and used as context for answer generation.
acrossdocuments).Toaddressthis,onlypeople,places,anddates—identified
via NER and semantic rules — are unified globally; all other entities are tagged
withadocumentidentifier.Thispreservescontextualspecificityandsourcetrace-
ability while allowing the global graph to link shared entities across documents.
Graph-Based Retrieval and Question AnsweringWhenauserasksaquestion,the
system can leverage the Global Graph to find answers that may not be present
in any single document. The process for querying the graph and generating a
response is detailed in Figure 3.
First, we identify the key aspects of the query (using the same ones identified
in the RAG phase) and determine which of these aspects are represented in the
Global Graph.
Retrieval then begins by linking the query’s key entities to their correspond-
ing nodes in the graph through string similarity matching, using the same
method applied during graph construction. Based on the number of linked en-
tities identified in the query, the system employs one of two traversal strate-
gies (Figure 3): forsingle-entity queries(e.g.,“What is known about Juan
Pérez?”), it retrieves the local neighborhood up to three hops or until a globally
relevant entity is found; formulti-entity queries(e.g.,“What is the connection
between Juan Pérez and Rafael Prieto?”), it identifies all simple paths between
the corresponding nodes.
To ensure faithful reasoning over the retrieved graph context, we adopt a
similar guardrail philosophy as in the RAG component. The LLM is prompted
using a graph-specific reasoning template that constrains it to generate answers
strictly based on the retrieved triplets. This prompt guides the model to produce
a structured JSON response that includes a boolean flag indicating whether a
sufficient answer was found, effectively mitigating hallucinations when the graph
lacks sufficient evidence.

On-the-Fly Knowledge Modeling 9
(a)
 (b)
 (c)
Fig.4: Representative dataset samples: (a) standardized single-column layout
with high contrast; (b) complex layout with varying ink, bleed-through, and low
contrast; (c) second page of a multi-page document.
This graph-based reasoning capability transforms the system from a doc-
ument retriever into a genuine knowledge exploration tool. To further aid ex-
ploration, the system can compute and display global graph analytics, such as
node centrality measures, providing users with high-level insights into the entire
document collection.
4 Experimental Setup
4.1 Datasets and Tasks
Our work focuses on a corpus of 130 images derived from 65 court-martial sen-
tencing documents from the Spanish Civil War period (1937–1940) [18]. The
corpus is of enormous importance for archivists and other heritage professionals
because of its significant value in documenting the victims of the civil war and
supporting the restoration of collective memory. In this context, it becomes in-
creasingly important to modelwho,where, andwhywithin the knowledge graph
to ensure structured, meaningful, and ethically responsible representation of the
past. We use the sentencing pages, typically 1–2 per case, as they summarize
key information: details about the accused, individuals involved in the trial, a
summary of events, and the final conviction.
These historical records present significant text extraction challenges, in-
cluding faded ink, paper degradation, inconsistent typewriter fonts, and variable
layouts with both typewritten and handwritten elements [2,12]. As shown in Fig-
ure 4, documents (a-c) illustrate different examples from the working dataset.
Document (a) corresponds to a simple sentence with clear text and a straightfor-
ward page layout. Even so, handwritten elements, such as the final signatures,

10 P. Font et.al
are already visible. Documents (b,c) exemplify the primary technical hurdles of
the corpus, specifically multi-page structural complexity and non-uniform back-
ground noise. These samples suffer from ink bleed-through (show-through ar-
tifacts) and low stroke-to-background contrast (faded ink), which significantly
impede binarization and line segmentation algorithms.
4.2 Metrics and evaluation
To validate our system’s performance, we conducted a multi-faceted evaluation.
Since there is no existing ground truth dataset for this historical archive, a
common challenge in digital humanities, our strategy combines component-wise
analysis, qualitative demonstrations, and human-centered studies.
Our evaluation protocol assesses the transcription pipeline first, followed by
an automated test of the RAG system, a human evaluation of the retrieval
and generation modules, and finally, a visualization of the knowledge graph and
conversational interface.
Evaluation of the Text Extraction PipelineWeevaluatedthemulti-stagetran-
scription pipeline using both qualitative and quantitative methods to demon-
strate its superiority over a baseline OCR approach.
Qualitative AnalysisWe visually inspected the output to assess how the
complete pipeline improves upon the direct application of an OCR engine. This
analysis showcases both the significant improvements in layout detection and
text fidelity, as well as the limitations that persist in the final output.
Quantitative AnalysisTo quantify the impact of our pipeline, we manually
transcribed a representative set of five multi-page documents, including several
with severe degradation, to establish a ground-truth (GT) dataset. We compared
this generated GT with the output of both the direct OCR application and our
complete pipeline.
We employed Word Error Rate (WER) and Character Error Rate (CER)
as our primary metrics. These measure the number of substitutions, insertions,
and deletions required to transform the generated text into the ground-truth
reference, normalized by the length of the reference. WER operates at the word
level, while CER operates at the character level. For both, lower scores indicate
better performance.
Automated Evaluation of Advanced RAGWe conducted an automated evalu-
ation to quantitatively measure the RAG system’s performance on fact-retrieval
tasks. For this, we used an existing focused ground-truth (GT) dataset contain-
ing person-specific attributes, from which we constructed an initial test set of
222 question-answer pairs.
These pairs follow a structured generation pattern based on entity-attribute
tuples.Forexample,givenagroundtruthentryof{Entity: Juan, Attribute:
Age, Value: 22}, the corresponding query is “How old is Juan?”. We will use
both the concise extraction (“22”) and natural language variations (“Juan is 22
years old”) as valid targets.
Our evaluation follows a multi-stage funnel to isolate different system com-
ponents. From the initial 222 queries, our OCR pipeline successfully transcribed

On-the-Fly Knowledge Modeling 11
the correct source document in 175 cases (a 78.8% recall). Of these, our RAG
agent, which is designed to abstain if context is insufficient, generated a substan-
tive answer for 125 cases. This final set of 125 high-potential responses forms
the basis for our in-depth quality and component-level analysis.
To assess accuracy, our primary metric is aCustom Exact Match. This
metric employs domain-aware normalization rules to validate if the core factual
information was correctly extracted regardless of formatting. For instance, in the
age example above, the metric normalizes "twenty-two" to match the ground
truth "22", ensuring robust evaluation against linguistic variations.
Tofurtherdiagnosesystembehavior,weemployedtheRAGASframework
[8,26], which uses an LLM as an automated evaluator across three metrics:Con-
text Precision(signal-to-noise ratio of retrieved items),Context Recall(coverage
of necessary retrieval information), andFaithfulness(factual consistency of gen-
erated answers against retrieved context).Human Evaluation: Evaluating the
Efficacy of Dynamic Graphs in Complex Information Synthesis
Evaluating systems such as RAG or LLMs in question-answering scenarios is
challenging without a large ground-truth dataset. For the retrieval component,
common metrics includerelevance(how well the retrieved context matches
the query) andaccuracy(how correct the selected documents are among all
candidates). For the generation component, metrics typically includerelevance
(how well the answer addresses the question),faithfulness(how accurately the
answer reflects the provided context), andcorrectness(how close the answer
is to an ideal response) [26]. To evaluate our architecture, we conducted a hu-
man study with 8 non-expert participants, yielding 64 annotated query-response
pairs. We designed two query types:Simple Queries (SQ), seeking facts about
a single entity (e.g.,“What was the sentence for Juan Pérez?”), andComplex
Queries (CQ), requiring synthesis across multiple documents (e.g.,“Which in-
dividuals fled to France?”).
Participants were presented with answers generated by both the RAG and
the Knowledge Graph systems in a blind comparison. First, they were asked
to identify whether the RAG system, the Graph system, or both successfully
provided an answer. Subsequently, they rated each generated response on a 1–5
Likert scale across four key dimensions:Overall Quality,Relevance,Faith-
fulness, andAnswer Depth.
4.3 Implementation Details
OCR Engine SelectionTo determine the optimal recognition engine for this his-
torical corpus, we qualitatively evaluated seven open-source solutions. Tesseract
[23] performed well on typewritten and binarized documents but struggled with
complex layouts. Ocular [3], designed for historical printed or handwritten text
using probabilistic models, yielded very poor results on typewritten material and
was discarded early. OCRopus [4] worked partially on some images but proved
insufficiently robust, failing frequently on uppercase or light text. Calamari [24]
was similarly discarded after initial testing. Kraken [15], a fork of OCRopus

12 P. Font et.al
Fig.5: WER comparison across models and prompts.
specialized in historical manuscripts with advanced layout analysis, achieved ex-
cellent handwritten text recognition but at the cost of slow processing. Ocrad
[10],alightweightnon-neuralengine,producedunacceptableresultsonourdocu-
ments. Finally, TrOCR [16], a transformer-based encoder-decoder model, showed
promise architecturally but performed poorly on this specific historical script
without dedicated fine-tuning. Based on this evaluation, Tesseract was selected
as the primary engine given its balance of accuracy, robustness to preprocessing,
and practical deployability for typewritten court-martial documents.
LLM Selection and OptimizationThe harmonization and refinement stages
relyonaLargeLanguageModeltocorrectOCRerrorswithoutalteringsemantic
meaning. The optimal LLM refinement setup was found via controlled experi-
ments. A small ground truth (GT) dataset was created by manually correcting
document transcriptions. Using this GT, various models and prompts were eval-
uated based on Word Error Rate (WER). Initial tests showed ‘microsoft/phi-4‘
as the best model. Further, prompt engineering optimized its performance. As
Figure 5 shows, “phi-4” with Prompt 6 achieved the lowest WER and was chosen.
4.4 Prompt Engineering and Guardrails
A critical challenge in applying RAG to historical data is preventing the genera-
tionofplausiblebutincorrectfacts("hallucinations").Toaddressthis,weimple-
mented strictstructural prompting. Instead of free-text responses, all LLM
agents in the system (Query Analysis, RAG Generation, and Graph Extraction)
are constrained to output valid JSON. The most critical prompt is the RAG An-
swer Generator, which includes a mandatory boolean flag (found_information)
forcing the model to explicitly evaluate if the context supports the answer. We
also use a prompt to generate the graph triplets (see Figure 6), which ensures the
long sentences are transformed into relevant nodes with significant connections
between them.

On-the-Fly Knowledge Modeling 13
Knowledge Graph Triplet Extraction Prompt
— SYSTEM MESSAGE —
You are an agent receiving a text fragment in the following format:
BOS | [content] | EOS
Your job is to extract multiple concise triplets (subject, predicate and object)
for the construction of a knowledge graph.
–Use name entities or key concepts like subjects and objects.
–Keep the nodes short (1 to 3 words), avoid long sentences.
–Extract all the triplets related to the same sentence, using the context.
–Make sure all the triplets are connected to each other.
–The text is in Spanish, answer always in Spanish.
Only return a list of triplets with this format:
[(’s’, ’p’, ’o’), (’s’, ’p’, ’o’), ...], without explanations nor adi-
tional text.
— USER MESSAGE —
Convert this sentence into triplets:BOS | [RAG answer text here] | EOS.
Only return the list.
Fig.6: Prompt for extracting(subject, predicate, object)triplets from
RAG answers, structured as asystem message(rules and output format) and
auser message(input text).
5 Results
5.1 Evaluation of the Text Extraction Pipeline
Qualitative Analysis.Figure 7 shows two qualitative comparisons for representa-
tivedocumentsnippets.Thebaselinetranscription,producedbyapplyingTesser-
act directly to the unprocessed image, contains numerous errors typical of his-
torical documents, such as character substitutions and missed words due to low
contrast. In some cases, it completely fails to recognize any text. In contrast, the
output from our full pipeline corrects these issues, producing text that is signif-
icantly more faithful to the original and layout-aware, although the refinement
with the LLM sometimes adds words that do not change the overall meaning
but do alter the original transcription.
Quantitative Analysis.The difficulty of this corpus is evident in the baseline
performance of a standard Tesseract-based OCR, which produces a nearly unus-
able WER of 72.8%. In contrast, as shown in Table 1, our full pipeline reduces
the WER to 33.5%, representing a 54% relative error reduction. The CER sees a
similar improvement, dropping from 46.7% to 22.4%. This highlights the critical
role of our multi-stage harmonization and LLM-based refinement in achieving
high-fidelity transcriptions from difficult historical sources where standard meth-
ods fail. However, the WER remains high, partly due to the LLM introducing or
substituting words that preserve meaning but alter the original text. It nonethe-
less indicates that there is still room for improvement.

14 P. Font et.al
CONSEJO DE GUERRA 
PERMANENPE  En la Plaza de Pamplona 
a veintiuno
SENTENCIA  de febrero. de mil 
novecientos treinta y
PRESIDENTE  move  del Tercer Año 
Triunfal,
Coronel de Artilleria:
D, Manuel Suarez y Sanchez, El Consejo 
de Guerra Permanente de
direct transcriptionCONSEJO DE GUERRA PERMANENTE 
SENTENCIA
En la Plaza de Pamplona, a veintiuno de 
febrero de mil novecientos treinta y nueve, 
noveno  del Tercer Año Triunfal, 
Presidente: Coronel de Artillería: D. 
Manuel Suárez y Sánchez. 
El Consejo de Guerra Permanente de
our transcriptionsection to transcribe
direct transcriptiondente, y como demandado,
*-> . EI g 23 q. adante, y como demandado otra, el 
procesado  Benigno Zapata, sin segundo 
apellido, hijo de padres desconocidos,
our transcriptionsection to transcribe
Fig.7: Two qualitative examples illustrating the improvement from raw OCR
output to the final refined text after the complete harmonization and LLM re-
finement process. Red indicates transcription errors; orange marks layout issues
or hallucinated words introduced by the LLM.
5.2 Automated Evaluation of Advanced RAG
On the filtered set of 125 questions, we first evaluated end-to-end task suc-
cess. The Custom Exact metric achieved a high score of 0.824, demonstrating
the system’s effectiveness. The results of the RAGAS framework provide a more
granular view of component performance. The system achieved a very high faith-
fulness score of 0.886, indicating that the generator is highly reliable and rarely
hallucinates.Thecontextprecisionscoreof0.888furtherconfirmsthatthemodel
is adept at locating relevant information within the full-document context. Cru-
cially, even within this “successful” cohort, the RAGAS context recall metric
scored 0.776. This reveals a key insight: while our pipeline identified the cor-
rect source document, RAGAS’s stricter semantic analysis determined that in
22.4% of these cases, the context was still insufficient to perfectly formulate the
ground-truth answer, often due to subtle OCR errors or the specific informa-
tion not being present on the document. This highlights the distinction between
simple document retrieval and true contextual sufficiency (see Table 2).
5.3 Human Evaluation: Evaluating the Efficacy of Dynamic Graphs
in Complex Information Synthesis
First, we analyzed which system was capable of answering each query type.
Table 3 shows that the systems performed their intended roles. The RAG sys-
Table 1: Transcription er-
ror rates
Method WER↓CER↓
Baseline (Tesseract) 72.8% 46.7%
Our Pipeline (Full) 33.5% 22.4%Table 2: Evaluation met-
rics on the filtered set
Metric Score
Custom Exact Match 0.824
Faithfulness (Generation) 0.886
Context Precision (Retrieval) 0.888
Context Recall (Retrieval) 0.776Table 3: System answer-
ing rate by query type
Query Type RAG Graph
Simple (N=42) 83.33% 6.25%
Complex (N=14) 68.75% 87.5%

On-the-Fly Knowledge Modeling 15
tem handled 83% of Simple Queries, while the Graph system was nearly silent.
Conversely, for Complex Queries, the Graph system activated 87% of the time,
confirming its role as the specialist for relational reasoning. Table 4 presents
our core human evaluation results, comparing system performance across both
query types. It includes only interactions where the system provided an answer,
excluding cases with no response. In addition to the primary quality metrics,
we analyzed two specific behaviors of the RAG system based on user feedback:
its handling of its own limitations and its robustness to name variations. Ta-
ble 5 details the frequency of different outcomes for the 51 queries where the
RAG system provided an answer. The results show that the system is generally
robust. In over 92% of cases, it either provided a complete answer or correctly
identified and stated its own limitations. Similarly, it successfully handled name
variations (or cases where none were needed) over 96% of the time, with only a
small fraction of failures due to entity confusion.
Table 4: Detailed performance comparison with query types.
Simple Queries (SQ) Complex Queries (CQ)
Metric System Mean Std. Dev. Mean Std. Dev.
Overall Quality RAG4.501.18 4.64 0.81
Graph 3.67 2.314.710.83
Relevance RAG4.670.99 3.73 1.42
Graph 3.67 1.534.860.53
Faithfulness RAG 4.82 0.68 4.36 1.43
Graph5.000.004.641.08
Answer Depth RAG 4.61 0.82 4.45 0.82
Graph4.670.584.640.84
For Simple Queries, the RAG system significantly outperforms the Graph in
both Overall Quality and Relevance. While scores for Faithfulness and Answer
Depth are comparable, RAG’s superior relevance confirms its role as the effec-
tive baseline for single-document questions. For Complex Queries, the results
highlight the critical role of the Knowledge Graph. The most striking result is
the reversal in the Relevance metric, where the Graph system is clearly supe-
rior.Furthermore, the Graph demonstrates stronger performance in Faithfulness
and Answer Depth. These results provide strong evidence that while RAG can
generate fluent answers, the Graph is essential for retrieving the correct cross-
document context needed for genuine synthesis, confirming the complementary
strengthsofourhybridarchitecture.Insummary,thehumanevaluationconfirms
the specialized roles of our architecture. Its overall effectiveness is reflected in
the final user satisfaction score, which averaged 4.12 out of 5, indicating a highly
positive user experience even when accounting for system failures.

16 P. Font et.al
Table 5: Detailed Behavior Analysis of the RAG System
Category Metric (Observed Outcome) Frequency
Limitations HandlingComplete (No limitations apply) 0.745
Stated limitations clearly 0.177
Failed to state limitations 0.078
Name Variation HandlingHandled correctly (No variation) 0.922
Handled correctly (Variation present) 0.039
Failed (Entity confusion) 0.039
6 Conclusions
We have validated a system for automatic knowledge modeling via conversa-
tional interfaces, comparing our graph-based approach against standard RAG
pipelines.AmongsevenOCRenginesevaluated,Tesseractofferedthebesttrade-
off between efficiency and performance. An ensemble-based correction method-
ology reduces hallucinations and improves robustness in downstream knowledge
extraction. Evaluation shows that GraphRAG and standard RAG perform com-
parably on simple extractive queries, but GraphRAG demonstrates clear supe-
riority on complex, multi-hop reasoning across all metrics. Beyond quantitative
gains, the key contribution is enabling on-the-fly knowledge modeling in histor-
ical digital libraries: expert archivists can iteratively refine domain knowledge
through conversation, transforming the interface from a retrieval tool into an
evolving knowledge management environment. By unifying primary sources with
curated and inferred facts in a structured graph index, the system supports long-
term dependencies, cross-record linking, and contextual expertise not explicitly
encoded in documents. Future work will address scalability, more sophisticated
graph reasoning, and deeper human-model collaboration workflows.
Acknowledgements
This work has been partially supported by the Spanish project PID2024-157778OB-
I00, Ministerio de Ciencia e Innovación, the Departament de Cultura of the Generalitat
de Catalunya, and the CERCA Program. Adrià Molina is funded with the PRE2022-
101575 grant provided by MCIN / AEI / 10.13039 / 501100011033 and by ERDF/EU.
This work was partially included in Paula’s Bachelor’s Thesis and can be found in [9].
References
1. Abdin, M., et al.: Phi-4 technical report (2024)
2. Antonacopoulos, A., Karatzas, D.: Document image analysis for world war ii per-
sonal records. In: First International Workshop on Document Image Analysis for
Libraries, 2004. Proceedings.
3. Berg-Kirkpatrick, T., Durrett, G., Klein, D.: Unsupervised transcription of histor-
ical documents (2013)

On-the-Fly Knowledge Modeling 17
4. Breuel, T.: The ocropus open source ocr system. p. 68150 (01 2008)
5. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidi-
rectional transformers for language understanding (2019)
6. Edge, D., et al.: From local to global: A graph rag approach to query-focused
summarization (2025)
7. Ehrmann, et al.: Named entity recognition and classification in historical docu-
ments: A survey (2023)
8. Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: Ragas: Automated evaluation
of retrieval augmented generation (2025)
9. Font Solá, P.: Cerca conversacional i estructuració del coneixement de documents
històrics,https://ddd.uab.cat/record/317580
10. Free Software Foundation: Ocrad: The gnu ocr engine (2023)
11. Gao, Y., et al.: Retrieval-augmented generation for large language models (2024)
12. Grădinaru, M., et al.: Complete ocr solution for image analysis of world war 2
documents. In: 2022 21st RoEduNet Conference (2022)
13. Honnibal, M., Montani, I., Van Landeghem, S., Boyd, A., et al.: spacy: Industrial-
strength natural language processing in python (2020)
14. Huang, L., et al.: A survey on hallucination in large language models: Principles,
taxonomy, challenges, and open questions43(2) (2025)
15. Kiessling, B.: Kraken – a universal text recognizer for the humanities. In: Proceed-
ings of the Digital Humanities 2019 Conference. Utrecht, The Netherlands (2019)
16. Li, M., et al.: Trocr: Transformer-based optical character recognition with pre-
trained models. In: Proceedings of the AAAI conference on artificial intelligence
(2023)
17. Llanes-Padrón, D., Pastor-Sánchez, J.A.: Records in contexts: the road of archives
to semantic interoperability. Program51(4), 387–405 (2017)
18. Ministerio de Cultura y Deporte: PARES: Portal de Archivos Españoles.https:
//pares.cultura.gob.es/(2024)
19. Omrani, P., Hosseini, A., Hooshanfar, K., Ebrahimian, Z., Toosi, R., Ali Akhaee,
M.: Hybrid retrieval-augmented generation approach for llms query response en-
hancement. In: 10th International Conference on Web Research (2024)
20. Paruchuri, V., Datalab Team: Surya: A lightweight framework for analyzing doc-
uments and PDFs at scale (May 2025)
21. Philips, J.P., Tabrizi, N.: Historical document processing: Historical document pro-
cessing: A survey of techniques, tools, and trends (2020)
22. Reimers, N., Gurevych, I.: Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. In: Inui, K., Jiang, J., Ng, V., Wan, X. (eds.) Proceedings of
the(EMNLP-IJCNLP).pp.3982–3992.AssociationforComputationalLinguistics,
Hong Kong, China (Nov 2019)
23. Smith, R.: An overview of the tesseract ocr engine. In: Ninth International Con-
ference on Document Analysis and Recognition (ICDAR 2007) (2007)
24. Wick, C., et al.: Calamari - A High-Performance Tensorflow-based Deep Learning
Package for Optical Character Recognition. Digital Humanities Quarterly (2020)
25. Ye, M., Zhang, J., Liu, J., Liu, C., Yin, B., Liu, C., Du, B., Tao, D.: Hi-sam:
Marrying segment anything model for hierarchical text segmentation (2024)
26. Yu, H., et al.: Evaluation of Retrieval-Augmented Generation: A Survey. Springer
Nature Singapore (2025)
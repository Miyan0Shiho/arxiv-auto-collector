# BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian Personal Income Tax Law, including case law

**Authors**: Juvenal Domingos Júnior, Augusto Faria, E. Seiti de Oliveira, Erick de Brito, Matheus Teotonio, Andre Assumpção, Diedre Carmo, Roberto Lotufo, Jayr Pereira

**Published**: 2025-05-21 18:11:41

**PDF URL**: [http://arxiv.org/pdf/2505.15916v1](http://arxiv.org/pdf/2505.15916v1)

## Abstract
This paper presents BR-TaxQA-R, a novel dataset designed to support question
answering with references in the context of Brazilian personal income tax law.
The dataset contains 715 questions from the 2024 official Q\&A document
published by Brazil's Internal Revenue Service, enriched with statutory norms
and administrative rulings from the Conselho Administrativo de Recursos Fiscais
(CARF). We implement a Retrieval-Augmented Generation (RAG) pipeline using
OpenAI embeddings for searching and GPT-4o-mini for answer generation. We
compare different text segmentation strategies and benchmark our system against
commercial tools such as ChatGPT and Perplexity.ai using RAGAS-based metrics.
Results show that our custom RAG pipeline outperforms commercial systems in
Response Relevancy, indicating stronger alignment with user queries, while
commercial models achieve higher scores in Factual Correctness and fluency.
These findings highlight a trade-off between legally grounded generation and
linguistic fluency. Crucially, we argue that human expert evaluation remains
essential to ensure the legal validity of AI-generated answers in high-stakes
domains such as taxation. BR-TaxQA-R is publicly available at
https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R.

## Full Text


<!-- PDF content starts -->

arXiv:2505.15916v1  [cs.CL]  21 May 2025BR-TaxQA-R: A Dataset for Question Answering
with References for Brazilian Personal Income
Tax Law, including case law
Juvenal Domingos Júnior1, Augusto Faria1, E. Seiti de Oliveira1, Erick de
Brito2, Matheus Teotonio2, Andre Assumpção3, Diedre Carmo1, Roberto
Lotufo1, and Jayr Pereira1,2
1Universidade Estadual de Campinas (UNICAMP), Campinas–SP, Brazil
2Universidade Federal do Cariri (UFCA), Juazeiro do Norte–CE, Brazil
3National Center for State Courts (NCSC), Williamsburg, Virginia, United States
jayr.pereira@ufca.edu.br
Abstract. This paper presents BR-TaxQA-R , a novel dataset de-
signed to support question answering with references in the context
of Brazilian personal income tax law. The dataset contains 715 ques-
tions from the 2024 official Q&A document published by Brazil’s Inter-
nal Revenue Service, enriched with statutory norms and administrative
rulings from the Conselho Administrativo de Recursos Fiscais (CARF).
We implement a Retrieval-Augmented Generation (RAG) pipeline using
OpenAI embeddings for searching and GPT-4o-mini for answer genera-
tion. We compare different text segmentation strategies and benchmark
our system against commercial tools such as ChatGPT and Perplex-
ity.ai using RAGAS-based metrics. Results show that our custom RAG
pipeline outperforms commercial systems in Response Relevancy , indi-
cating stronger alignment with user queries, while commercial models
achieve higher scores in Factual Correctness andfluency. These findings
highlight a trade-off between legally grounded generation and linguis-
tic fluency. Crucially, we argue that human expert evaluation remains
essential to ensure the legal validity of AI-generated answers in high-
stakes domains such as taxation. BR-TaxQA-R is publicly available at
https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R .
Keywords: Retrieval-Augmented Generation ·Legal NLP ·Brazilian
Tax Law ·Question Answering ·CARF Rulings
1 Introduction
Alongstandingchallengemanyindependentjudiciariesandadministrativecourts
faceisthat,asthepopulationgrows,sodoesthenumberofcases,placingincreas-
ing pressure on courts and often exceeding their capacity [5]. The integration of
Artificial Intelligence (AI), particularly through Natural Language Processing
(NLP) techniques, offers the potential to significantly enhance judicial efficiency
and effectiveness. This vision is exemplified by the “Smart Courts” initiative

2 Domingos Júnior et al.
outlined in China’s Artificial Intelligence Development Plan, published by the
State Council [20]. AI has been applied to various legal domain tasks, rang-
ing from Named Entity Recognition to Ruling Prediction with the overarching
goal of improving judicial productivity [17]. Developments in other judicial sys-
tems suggest a growing interest in understanding how such technologies might
contribute to improved access and operational capacity. For instance, the 2023
Year-End Report on the Federal Judiciary in the United States highlights both
the opportunities and limitations of AI in the courtroom, noting its potential
to assist litigants with limited resources and to streamline certain processes,
while also cautioning against overreliance due to risks such as hallucinated con-
tent, data privacy concerns, and the challenges of replicating nuanced human
judgment4.
The successful application of NLP techniques in the legal domain relies on
the availability of specialized datasets. There is an ongoing effort to narrow the
resource gap for training and evaluating NLP systems in Brazilian Portuguese,
considering the variety of legal tasks involved. For example, [17] developed a
large corpus for the Brazilian legal domain, proposing a methodology to extract
and preprocess legal documents from the judiciary, legislative, and executive
branchesofgovernment.Thiscorpusisaimedatbeingusedforpretrainingtasks,
but still requires further processing for downstream applications. [16] proposed
semantic similarity datasets based on published decisions from two appeals bod-
ies of Brazilian Federal and Administrative Courts, creating a unique resource
for jurisprudence and case law research. [19] constructed a human-annotated,
relevance feedback dataset for legal information retrieval based on legislative
documents from the Brazilian Congress. Although focused on a specific sce-
nario, relevance feedback datasets are a crucial step toward developing robust
legal question answering systems. Finally, [13] leveraged the curated Tax Law
Question Answering (QA) manual for corporate entities, published by Brazil’s
Internal Revenue Service (i.e. Receita Federal do Brasil or RFB), to create a QA
dataset that includes legal document references supporting the answers. This
dataset was used to evaluate LLMs’ ability to generate answers when provided
withthegoldpassageascontext,enablingtheanalysisofbothanswercorrectness
and faithfulness to the supporting references.
In this paper, we propose BR-TaxQA-R, a dataset for Brazilian Tax Law
Question Answering with supporting references focused on personal income tax
law. BR-TaxQA-R extends the work of [13] by enabling the evaluation of the
complete QA pipeline, incorporating all legal document references, both ex-
plicitly and implicitly cited in the answers. As additional supporting context,
BR-TaxQA-R includes a curated set of rulings from CARF, the administrative
appeals court handling federal tax disputes in Brazil. These rulings compose the
case law portion of the dataset, providing real-world interpretations and applica-
tionsofPersonalIncomeTaxregulations.BR-TaxQA-Renablestheevaluationof
complete Retrieval-Augmented Generation (RAG) pipelines, encompassing both
the information retrieval and answer generation stages, and provides a baseline
4https://www.supremecourt.gov/publicinfo/year-end/2023year-endreport.pdf

BR-TaxQA-R 3
for future research. Our evaluation indicates that simple sliding-window segmen-
tation achieves good results, and that incorporating relevant jurisprudence fur-
ther improves performance. Although closed-source commercial tools employing
LLM-based search pipelines achieve superior performance, retrieval in the legal
domain remains a challenging task.
The key contributions of this paper are:
–We introduce BR-TaxQA-R, a novel dataset for tax-related QA in Brazilian
Portuguese, combining statutory and case law.
–We implement and benchmark a legal-domain-specific RAG pipeline using
hierarchical segmentation and legal prompting.
–WeevaluatethesystemagainstcommercialLLMtoolsanddiscussthetrade-
offs between legal traceability and linguistic fluency.
The remainder of this paper is organized as follows. Section 2 details the
methodology adopted to create the dataset, including the parsing of original
questions and answers (2.1), the acquisition of supporting legal documents (2.2),
and the construction of the additional jurisprudence set (2.3). Section 3 presents
thepublishedformatandstatisticsoftheBR-TaxQA-Rdataset.Sections4and5
describe the experiments conducted to evaluate the dataset and discuss the re-
sults. Finally, Section 6 concludes the paper.
2 Dataset Acquisition Methodology
This study aims to develop a dataset that can be used to train and evaluate
a Retrieval-Augmented Generation (RAG) system [6] for answering questions
relatedtoBrazilianpersonalincometaxlaw.Thedatasetconstructionfolloweda
three-step methodology: (1) extraction of questions and answers from the official
2024 “Questions and Answers” document published by the RFB (cf. Section
2.1), (2) collection and processing of tax regulations cited as references in the
answers (cf. Section 2.2), and (3) retrieval of relevant administrative rulings from
CARF to provide jurisprudential support (cf. Section 2.3). These components
were combined to create a legally grounded and contextually rich dataset aligned
with real-world tax guidance.
2.1 Questions extraction
The first step in the dataset acquisition process involved extracting the questions
and answers from the official document “Questions and Answers” published by
RFB for the year 2024 [14]. That document is available in PDF format, and
we applied a combination of automated tools and manual verification to ensure
accurate extraction.
Our approach was to extract as much information as possible from the doc-
ument, preserving the original to allow further error correction and processing.

4 Domingos Júnior et al.
In addition to the question and its answer, all legal document references pro-
vided to support the answers are relevant, and identifying and processing them
represented most of this extraction work.
After extracting the text information from the PDF using a Python Library5,
the document text was processed in two stages. The first stage consisted of
splitting the text into the following parts: question, answer, legal documents
supporting the answer, and links to other questions. The second stage consisted
of processing the answer body to extract additional legal document references
supporting the answer and additional links to other questions.
Since throughout the different questions the same legal document was ref-
erenced using different notation — abbreviations or acronyms — we applied a
semi-automated document deduplication strategy using LLM support: we clus-
tered the documents by their name’s initial letter and passed the list to the LLM
instructing for identifying and removing document’s duplication, whenever the
same part (e.g. article, paragraph, clauses) was referred to. The final list was
manually verified to fix the remaining duplicates.
2.2 Tax regulations
Tax regulations were obtained through a curated selection, defined by the ref-
erences listed in the processed “Questions and Answers” 2024 document. The
original documents were retrieved as PDF files or HTML pages, ensuring the
most up-to-date versions were selected. For PDF files, an automated download
was performed, followed by text extraction using Python libraries and stored as
text files.
In the case of HTML documents, web scraping techniques were applied to
parse and clean the page content, removing the amended or revoked parts, re-
spectively indicated by the <strike> (strikethrough text) and <del>(removed
text) tags. The resulting text was saved as plain text files, named and organized
according to the regulation identifiers.
2.3 Case law collection
Administrativerulings(caselaw)werecollectedthroughautomatedwebscraping
from the official repository of the Brazilian Ministry of Finance6, the federal
agency that houses CARF, and converted into plain text. Only 2023 rulings were
processed, with previous years potentially added in a future dataset release.
To ensure that the inclusion of case law would genuinely enhance the rel-
evance and contextual alignment of answers within BR-TaxQA-R, we adopted
two primary qualitative criteria for selecting rulings through web scraping. First,
we based the selection on the presence of keywords directly extracted from the
questions in the “Questions and Answers” document. This constraint helped en-
sure that the retrieved decisions addressed legal issues analogous to those in the
official guidance, thus avoiding the inclusion of unrelated jurisprudence.
5https://pymupdf.readthedocs.io
6https://acordaos.economia.gov.br/solr/acordaos2/browse/

BR-TaxQA-R 5
Second, we applied a temporal filter to guarantee that the rulings reflected
current legal interpretations. Only rulings published within one year of the 2024
“Questions and Answers” edition were considered. This time-bounded selection
criterionaimedtomitigatetheriskofreferencingoutdatedprecedentsthatmight
no longer align with current tax practices or administrative guidance. This aligns
with retrospective studies, in which closed cases are indexed by the “date of
death” and analyzed for their legal characteristics[9]. Likewise, selecting relevant
rulings relies on subjective representations of legal concepts, which must be
explicitly described and theoretically justified.
3 BR-TaxQA-R
We named the dataset BR-TaxQA-R, which stands for Brazilian Tax Question
Answering with References. The dataset is composed of three main components:
thequestions set , thesources set , and thecase law set .
3.1 Questions set
The question set contains 715 questions and answers extracted from the official
document published by RFB. 117 ( ∼16%) out of the 715 questions do not ref-
erence any external documents, since their answers are not directly defined in
a legal document; those questions were kept in the dataset for completion. The
answers to several questions reference other questions within the document, and
those links were captured. The question set was structured to hold the original
data as much as possible, along with the scraped information:
– question_number : The question number, starting with 1, as referred to
in the original document.
– question_summary : A very brief description of the question subject, ex-
tracted from the original document.
– question_text : The question itself, as originally posed.
– answer : Answer, as extracted from the original document. It is a list of
strings, respecting the PDF formatting. It contains all the information pro-
vided after the question_text and before a link to the document index,
provided at the end of all questions.
– answer_cleaned : The answer field after removing all explicit external ref-
erences — the legal documents captured in the sources set — and all explicit
inter-question references. External references were provided in the original
document: explicitly, through grayed boxes, and implicitly, embedded in the
answer text.
– references : The list of external references explicitly provided.
– linked_questions : List of other questions linked in the provided answer.
– formatted_references : The explicit external references, LLM-processed
to separate the document title, articles, sections, paragraphs, and other spe-
cific parts mentioned.

6 Domingos Júnior et al.
– embedded_references : External references are implicitly provided, em-
bedded in the answer text.
– formatted_embedded_references :TheimplicitexternalreferencesLLM-
processed to separate the specific information mentioned, similar to the for-
matted_references field.
– all_formated_references : Merge of formatted_references and format-
ted_embedded_referencesfields,combiningtheinformationofthelegaldoc-
uments, and including the name of the text file (the file sub-field) containing
each particular legal document has been captured in the dataset.
3.2 Sources and case law sets
The sources and case law sets compose a corpus supporting the answers provided
to the questions set. The sources set contains all the legal documents listed as
official sources for the answers provided in the original “Questions and Answers”
document, and corresponds to the minimal legal documents set required for a
RAG system to properly answer all the questions. The case law set contains
CARF administrative rulings on the topics covered by the questions, which can
potentially offer concrete examples on the concepts covered by the legal docu-
ments, and provide assistance to assertive answers by the same RAG system.
Both the sources and case law sets have the following format:
– filename : The legal document scraped data filename, as referred to within
the all_formated_references field in the questions set.
– filedata : The scraped legal document information, extracted as text data.
3.3 Dataset statistics
Although the dataset is relatively small, the legal domain introduces significant
complexity when selecting relevant segments from the supporting documents for
answering questions. The case law documents can improve answer quality, but
they also increase the overall pipeline complexity, as they are numerous and vary
significantly in structure. Table 1 summarizes the dataset size.
Tables 2 and 3 present statistics on the number of links found in the answers
to other questions and external documents. While there is one question with
20 links to others, most answers do not reference other questions, suggesting
they are mainly independent. The number of external links per question is more
heterogeneous, which can be interpreted as an indicator of complexity: it is
reasonabletoassumethatthe25questionswithmorethan10externalreferences
are more challenging for a Q&A system to answer correctly. Among the 478
external documents, only 10 account for over half of all references. Given the
significantvariationinlength,thesedocumentsalsopresentadditionalchallenges
for the information retrieval stage.

BR-TaxQA-R 7
Table 1. BR-TaxQA-R size statistics.
Questions Answers Source
documentsCase-Law
documents
count 715 715 478 7204
min words 3 6 24 425
max words 74 3118 165830 75584
mean words 19.11 143.36 4546.46 3171.70
median words 17.00 81.00 649.00 1983.50
Table 2. BR-TaxQA-R question links statistics.
Links to
other
questionsExplicit
external
linksImplicit
external
linksTotal
external
links
without 478 (66.85%) 151 (21.12%) 524 (73.29%) 117 (16.36%)
>2 and <10 51 (7.13%) 248 (34.69%) 55 (7.69%) 287 (40.14%)
≥10 4 (0.56%) 5 (0.70%) 3 (0.42%) 25 (3.50%)
minimum 0 0 0 0
maximum 20 18 16 21
mean 0.69 2.07 0.63 2.70
median 0.00 2.00 0.00 2.0
question
max129 442 and 560 177 177, 442, 560
4 Experiments
This section describes the experiments conducted using BR-TaxQA-R. The ex-
periments were designed to evaluate the performance of a custom Retrieval-
Augmented Generation (RAG) system, which was implemented using the BR-
TaxQA-R dataset source files as the knowledge base. We also assess the perfor-
mance of two commercial tools, ChatGPT and Perplexity.ai, using the same set
of questions from the BR-TaxQA-R dataset. The experiments aim to establish a
baseline for the RAG system’s performance in answering tax-related questions,
comparing different segmentation strategies, and evaluating the results against
commercial tools. In the next subsections, we describe the custom RAG system,
Table 3. 10 most referred source documents by BR-TaxQA-R questions.
Reference
countDocument Word
count
284 Decreto n º9.580, de 22 de novembro de 2018 165830
123 Instrução Normativa RFB n º1500, de 29 de outubro de 2014 28269
68 Lei n º9.250, de 26 de dezembro de 1995 6452
63 Instrução Normativa SRF n º83, de 11 de outubro de 2001 3177
52 Instrução Normativa SRF n º208, de 27 de setembro de 2002 10118
49 Lei n º7.713, de 22 de dezembro de 1988 6636
45 Instrução Normativa RFB n º2178, de 05 de março de 2024 3997
44 Instrução Normativa SRF n º84, de 11 de outubro de 2001 4539
37 Instrução Normativa RFB n º1585, de 31 de agosto de 2015 26504
22 Instrução Normativa SRF n º81, de 11 de outubro de 2001 4880
Total 787 53.32% (1476)

8 Domingos Júnior et al.
the commercial tools used for comparison, and the evaluation metrics employed
to assess the performance of the generated answers.
4.1 Custom RAG system
We implemented a custom RAG system using the BR-TaxQA-R dataset as the
knowledge base. The system is designed to efficiently retrieve relevant infor-
mation from the sources and case law sets, generate accurate answers to user
queries, and provide explicit references to the legal documents used in the an-
swer generation process. Following the principles outlined in [6], the RAG system
is structured into three main components: data preparation, indexing, and an-
swer generation, ensuring that retrieved content is seamlessly integrated into the
response generation pipeline.
Data Preparation The data preparation includes the document segmentation
andindexingfortheInformationRetrievalRAGstage.Weconsidered2dataseg-
mentation approaches for the sources andcase law datasets, each one making
increasing usage of the documents’ internal structure:
– Sliding-window , considering 2048-token windows and 1024-token stride,
producing regular-sized overlapping segments.
– Langchain Recursive Character Text Splitter7, which is recommended
for generic text, splitting the text recursively according to a given separators
list, until the resulting segments are small enough. In its default configura-
tion, the provided separators try to keep paragraph contents in a single
chunk, using very little information about the text’s internal structure. We
provided a customized separators list, including the documents containing
statutory law hierarchy with many splitting points; the expected effect is
that the recursive splitter would break the segments at those separators oc-
currences as much as possible, resulting in segments with meaningful bound-
aries according to the documents’ original internal hierarchical structure. We
considered chunks up to 1000-character length, with at most 100-character
overlap.
Indexing We adopted the dense passage retrieval approach [4]: once the doc-
uments were segmented, we indexed them using the text-embedding-3-small
commercial model offered through OpenAI API8. This model is designed to
generate dense vector representations of text, which can be used for similarity-
based retrieval. The embeddings were generated for each of the two segmenta-
tion strategies and saved using FAISS (Facebook AI Similarity Search) [1] to
enable efficient similarity-based retrieval. For each segmentation variant, a sep-
arate FAISS index was created. The FAISS IndexFlatL2 type was employed,
7https://python.langchain.com/docs/how_to/recursive_text_splitter/
8https://openai.com/index/new-embedding-models-and-api-updates/

BR-TaxQA-R 9
which computes the L2 (Euclidean) distance for nearest neighbor searches. This
step was repeated for each segmentation approach applied to the BR-TaxQA-
R dataset. For case law documents, only the sliding-window segmentation was
considered, due to the lack of hierarchical structure.
We implemented a retrieval function that uses the FAISS index to retrieve
relevant chunks based on user queries. The retrieval process involves embedding
the user query using the same text-embedding-3-small model and querying
the FAISS index to find the top kmost similar document segments.
Answer Generation The context retrieved by FAISS is fed into a prompt-
based RAG system powered by OpenAI’s gpt-4o-mini . The prompt is metic-
ulously crafted to emulate the behavior of a virtual assistant specializing in
Brazilian tax law. To enhance interpretability and improve legal reasoning, the
system employs Chain-of-Thought (CoT) prompting, guiding the model to artic-
ulate intermediate reasoning steps before producing the final answer. This helps
ensure that conclusions are logically grounded in the retrieved legal text. The
final version of the prompt was designed to ensure that responses:
–Are derived solely from the provided context;
–Contain no direct references to the context or user interaction;
–Include citations of applicable legal sources (norms and articles only) at the
end of the response in a structured list;
–Avoid mid-response citations or references to document structure such as
paragraphs or subitems;
–Follow naming conventions (e.g., “Decreto N º9.580” instead of “RIR/2018”).
If not enough information is found in the retrieved context, the model is
instructed to return a fallback answer indicating the system is still learning.
This structured response format, consisting of a generated answer and a list of
cited legal sources, enables consistent and automated evaluation of the RAG
pipeline across different segmentation strategies.
4.2 Commercial Tools
In addition to evaluating our proposed RAG system, an assessment was con-
ducted using prominent commercial Large Language Models (LLMs) equipped
with integrated web search or deep search capabilities. The tools examined in-
cluded:
– ChatGPT (utilizing GPT-4o and GPT-4o mini models), with its search
integration.
– Perplexity AI , employing its Deep Research feature9.
– Grok 3 , leveraging its DeepSearch functionality.
9https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research

10 Domingos Júnior et al.
Theprimaryobjectiveofthiscomparativeanalysiswastobenchmarktheper-
formance of these state-of-the-art commercial systems. We aimed to determine
whether their search-augmented responses could effectively approximate the ac-
curacy and completeness of the ground-truth answers. This evaluation sought to
understand the capabilities and limitations of readily available, market-leading
generative AI tools in retrieving factual information and providing valid, trace-
able sources for complex Brazilian tax inquiries, using the same question set
applied to our custom RAG model.
4.3 Evaluation Metrics
To quantitatively assess the performance of the responses generated by the com-
mercial tools detailed in the previous section, a dedicated evaluation framework
was employed, comparing its outputs against the established ground truth. This
framework is built upon the RAGAS (Retrieval-Augmented Generation Assess-
ment) library [2], a specialized Python package suited for evaluating generated
text against reference text.
A custom Python script automated this evaluation. It processed a dataset
structured in JSON format, containing the original question ( question_text ),
the ground truth answer ( answer), and the candidate response generated by
the commercial tool ( candidate ). The RAGAS library, within this script, uti-
lized Langchain wrappers to interface with specified LLMs (e.g., gpt-4o-mini )
and embedding models (e.g., text-embedding-3-small ) for calculating certain
evaluation metrics, as described below:
– Response Relevancy : Measures how relevant the answer is to the original
question by using an LLM to generate alternative questions from the an-
swer’s content, then calculating the average cosine similarity between their
embeddings and the original question. Higher scores reflect better alignment
with the query intent, penalizing incomplete or redundant answers.
– Factual Correctness : Refers to the degree to which the candidate answer
aligns with verified knowledge or ground truth claims, as if how much it was
entailed by them. Factuality is measured by comparing model completions
againsttrusteddatasetsorexternalknowledgebases.Themetriciscomputed
by matching generated answers to reference answers and checking how much
data as statements remained, was lost, and created.
– SemanticSimilarity :Semanticsimilaritywillmeasurehowcloselyamodel’s
outputmatchesatextusedasreference.Itwillhelpdetectwhetherresponses
preserve intended information. The metric is typically computed using sen-
tence embeddings, and Higher scores indicate closer semantic alignment be-
tween the candidate and the reference. Factual Correctness may use it to
match referred claims.
– BLEU Score : A precision-based metric that evaluates how many character
sequences in particular order from the candidate answer can be overlapped
to the reference texts. Scores range from 0 to 1, where higher scores indicate

BR-TaxQA-R 11
better overlap. BLEU is sensitive to exact word matches and word order, as
it was firstly formulated to evaluate translated text [11].
– ROUGE Score : A set of metrics that compares model output to refer-
ence summaries based on sensitivity (which is the fraction of correctly se-
lected data from all relevant entailments) of n-grams, longest common sub-
sequences, and skip-bigram matches. ROUGE-L is a widely known variant
that captures fluency and structure through sequence alignment [7].
5 Results
We evaluated the performance of our custom RAG system using multiple seg-
mentation strategies and compared it against leading commercial tools. Table 4
presents a summary of the results across all evaluation metrics.
Table 4. Evaluation metrics for answer generation across different systems. All the
Custom RAG used OpenAI’s gpt-4o-mini for generating responses.
Method Resp.
RelevancyFactual
Corr.Semantic
SimilarityBLEU ROUGE-L
Custom RAG Systems
Recursive segmentation 0.791 0.286 0.765 0.185 0.241
Sliding-window segmentation 0.819 0.313 0.766 0.178 0.248
Recursive segmentation + case law 0.811 0.296 0.763 0.175 0.241
Sliding-window segmentation + case law 0.829 0.327 0.768 0.190 0.248
Only case law w/ sliding-window 0.818 0.209 0.744 0.149 0.207
Commercial Tools
ChatGPT + Search tool 0.738 0.389 0.793 0.158 0.251
Perplexity.ai + Deep Research 0.665 0.469 0.757 0.075 0.106
Grok 3 + DeepSearch 0.509 0.454 0.745 0.099 0.089
Among the custom RAG configurations, the sliding-window segmenta-
tion with case law achieved the highest score in Response Relevancy (0.829),
outperforming all other systems, including commercial tools. This suggests that
retrieving overlapping segments enriched with administrative rulings contributes
positively to aligning model outputs with user intent.
In terms of Factual Correctness , commercial models performed better: Per-
plexity.ai reached the highest score (0.469), followed by ChatGPT (0.389).
Despite using grounded references, our RAG system was outperformed in this
dimension, likely due to the broader training data and advanced retrieval mech-
anisms available to commercial tools.
Thesliding-window + case law configuration also performed competi-
tively in BLEU(0.190) and ROUGE-L (0.248), indicating that its generated an-
swers were structurally and lexically similar to the gold standard. Nevertheless,
ChatGPT led in Semantic Similarity (0.793) and attained the best ROUGE-L
score (0.251), reflecting superior fluency and semantic alignment.

12 Domingos Júnior et al.
Thecase law only configuration yielded an interesting result: although it
performed worse across most metrics, it achieved the second-highest Response
Relevancy score. This outcome suggests that relying solely on jurisprudential
content (without normative references) limits the model’s ability to generate
precise and legally grounded responses: the answers might be relevant, but not
easily verifiable against the corresponding legislation.
Although commercial tools achieved better results, there remains room for
improvement, as demonstrated in the literature where RAGAS Factual Correct-
nessandSemantic Similarity metrics have been shown to align well with human
evaluations [15], [10]. The verified performance on these metrics reinforces the
understanding that retrieval remains a challenging task in the legal domain [8],
[12], [3]. Furthermore, the application of jurisprudence can help bridge the gap
between the abstract concepts encapsulated in legal statutes and regulations and
the real-world facts described in user questions [21], [18].
Overall, our results reveal a trade-off between legal traceability and linguistic
fluency:
–Our domain-specific RAG pipeline excels in relevance andcontextual preci-
sion;
–Commercial tools generate more fluent and complete answers, but often lack
explicit grounding in legal sources.
Fig. 1.Illustration of the trade-off between contextual precision and linguistic fluency.

BR-TaxQA-R 13
Figure 1 illustrates the trade-off between contextual precision and fluency.
We consider the following example: “Who can opt for the standard deduction
in the annual tax filing declaration?”. This case demonstrates the contrasting
behavior of the two models:
– CustomRAG :Theresponsefromthedomain-specificRAGmodelprovides
a concise and legally precise description, directly reflecting the relevant tax
regulations. It lists the exact criteria required for opting for the simplified
deduction, aligning closely with the formal language and structure typical of
regulatory documents. However, this approach tends to prioritize accuracy
over readability, resulting in a less conversational tone.
– ChatGPT : In contrast, the ChatGPT response adopts a more conversa-
tional style, using natural language that is generally easier to read. It cap-
tures the main points effectively but lacks the precise legal references found
in the RAG response. This broader, more accessible phrasing can be advan-
tageous for non-specialist audiences but risks omitting critical legal nuances.
This example highlights the broader trend observed in our experiments:
domain-specific models excel in precise, contextually accurate responses, while
general-purpose commercial tools like ChatGPT often favor fluency and com-
prehensiveness at the expense of explicit legal traceability.
These findings underscore the importance of human expert evaluation in le-
gal question answering. High scores in metrics such as semantic similarity or
ROUGE do not guarantee legal adequacy. In several cases, fluent answers gen-
erated by commercial tools were factually incorrect or unsupported by authori-
tative legal documents–a critical issue in high-stakes contexts like tax guidance.
6 Conclusions
This study introduced BR-TaxQA-R, a novel dataset designed to support the
development and evaluation of Retrieval-Augmented Generation (RAG) systems
in the domain of Brazilian personal income tax law. By combining statutory
documents, administrative rulings (CARF decisions), and an extensive set of
official questions and answers published by RFB, the dataset provides a valuable
resource for both academic research and applied legal NLP.
Our experiments demonstrated that a custom RAG pipeline, carefully tai-
lored to the legal domain through legal-specific prompting and employing simple
sliding-window segmentation over the legal corpus, achieved strong performance
in terms of Response Relevancy , particularly when jurisprudence on the ques-
tion topics was available. However, commercial systems such as ChatGPT, which
benefit from broader training data and advanced retrieval mechanisms, outper-
formed our model in Factual Correctness and fluency.
These findings suggest that while specialized systems can be more focused
and legally grounded, they may still fall short in naturalness and completeness
compared to state-of-the-art general-purpose tools. Importantly, the evaluation

14 Domingos Júnior et al.
results also emphasize the need for human assessment in legal QA tasks. Metrics
such as semantic similarity or BLEU/ROUGE alone are insufficient to guar-
antee that an answer is legally valid or practically useful. In our case, some
high-scoring answers from the RAG system lacked critical legal nuance, while
ChatGPT occasionally provided fluent but ungrounded content. Thus, expert
evaluation remains essential to ensure the legal accuracy and trustworthiness of
AI-generated responses.
Future work includes incorporating multi-year CARF decisions and improv-
ing response calibration mechanisms. We also aim to refine human-in-the-loop
evaluation protocols to better capture legal adequacy, traceability, and user trust
in automated systems.
References
1. Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E., Lomeli,
M., Hosseini, L., Jégou, H.: The faiss library (2025), https://arxiv.org/abs/
2401.08281
2. Es, S., James, J., Espinosa Anke, L., Schockaert, S.: RAGAs: Automated eval-
uation of retrieval augmented generation. In: Aletras, N., De Clercq, O. (eds.)
Proceedings of the 18th Conference of the European Chapter of the Associa-
tion for Computational Linguistics: System Demonstrations. pp. 150–158. As-
sociation for Computational Linguistics, St. Julians, Malta (Mar 2024), https:
//aclanthology.org/2024.eacl-demo.16/
3. Feng, Y., Li, C., Ng, V.: Legal case retrieval: A survey of the state of the art.
In: Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). pp. 6472–6485 (2024)
4. Karpukhin, V., Oguz, B., Min, S., Lewis, P.S., Wu, L., Edunov, S., Chen, D., Yih,
W.t.: Dense passage retrieval for open-domain question answering. In: EMNLP
(1). pp. 6769–6781 (2020)
5. Lai, J., Gan, W., Wu, J., Qi, Z., Yu, P.S.: Large language models in law: A survey.
AI Open (2024)
6. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks. In: Proceedings of the 34th Interna-
tional Conference on Neural Information Processing Systems. NIPS ’20, Curran
Associates Inc., Red Hook, NY, USA (2020)
7. Lin, C.Y.: ROUGE: A package for automatic evaluation of summaries. In: Text
Summarization Branches Out. pp. 74–81. Association for Computational Linguis-
tics, Barcelona, Spain (Jul 2004), https://aclanthology.org/W04-1013/
8. Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C.D., Ho, D.E.:
Hallucination-free? assessing the reliability of leading ai legal research tools. URL
https://arxiv. org/abs/2405.20362 (2024)
9. Okamoto, R.F., Trecenti, J.: Metodologia de pesquisa jurimétrica. Associação
Brasileira de Jurimetria (2022), https://livro.abj.org.br/ , acesso em: 6 maio
2025
10. Oro, E., Granata, F.M., Lanza, A., Bachir, A., De Grandis, L., Ruffolo, M.: Eval-
uating retrieval-augmented generation for question answering with large language
models. In: CEUR Workshop Proceedings. vol. 3762, pp. 129–134 (2024)

BR-TaxQA-R 15
11. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: a method for automatic
evaluation of machine translation. In: Proceedings of the 40th annual meeting of
the Association for Computational Linguistics. pp. 311–318 (2002)
12. Paul, S., Bhatt, R., Goyal, P., Ghosh, S.: Legal statute identification: A case study
using state-of-the-art datasets and methods. In: Proceedings of the 47th Inter-
national ACM SIGIR Conference on Research and Development in Information
Retrieval. pp. 2231–2240 (2024)
13. Presa, J.P.C., Camilo Junior, C.G., Oliveira, S.S.T.d.: Evaluating large language
models for tax law reasoning. In: Brazilian Conference on Intelligent Systems. pp.
460–474. Springer (2024)
14. Receita Federal do Brasil: Perguntas e Respostas IRPF 2024. https:
//www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/
perguntas-e-respostas/dirpf/pr-irpf-2024.pdf/view (2024), accessed:
2025-05-10
15. Roychowdhury, S., Soman, S., Ranjani, H., Gunda, N., Chhabra, V., Bala, S.K.:
Evaluation of rag metrics for question answering in the telecom domain. arXiv
preprint arXiv:2407.12873 (2024)
16. da Silva Junior, D., dos Santos Corval, P.R., de Oliveira, D., Paes, A.: Datasets
for portuguese legal semantic textual similarity. Journal of Information and Data
Management 15(1), 206–215 (2024)
17. Siqueira, F.A., Vitório, D., Souza, E., Santos, J.A., Albuquerque, H.O., Dias, M.S.,
Silva, N.F., de Carvalho, A.C., Oliveira, A.L., Bastos-Filho, C.: Ulysses tesemõ: a
new large corpus for brazilian legal and governmental domain. Language Resources
and Evaluation pp. 1–20 (2024)
18. Su, W., Hu, Y., Xie, A., Ai, Q., Que, Z., Zheng, N., Liu, Y., Shen, W., Liu,
Y.: Stard: A chinese statute retrieval dataset with real queries issued by non-
professionals. arXiv preprint arXiv:2406.15313 (2024)
19. Vitório, D., Souza, E., Martins, L., da Silva, N.F., de Carvalho, A.C.P.d.L.,
Oliveira, A.L., de Andrade, F.E.: Building a relevance feedback corpus for legal
information retrieval in the real-case scenario of the brazilian chamber of deputies.
Language Resources and Evaluation pp. 1–21 (2024)
20. Wang, N., Tian, M.Y.: “intelligent justice”: human-centered considerations in
china’s legal ai transformation. AI and Ethics 3(2), 349–354 (2023)
21. Xiao, C., Liu, Z., Lin, Y., Sun, M.: Legal knowledge representation learning. In:
Representation Learning for Natural Language Processing, pp. 401–432. Springer
Nature Singapore Singapore (2023)
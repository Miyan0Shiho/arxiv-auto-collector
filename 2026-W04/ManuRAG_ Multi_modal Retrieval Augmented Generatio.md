# ManuRAG: Multi-modal Retrieval Augmented Generation for Manufacturing Question Answering

**Authors**: Yunqing Li, Zihan Dong, Farhad Ameri, Jianbang Zhang

**Published**: 2026-01-21 19:59:27

**PDF URL**: [https://arxiv.org/pdf/2601.15434v1](https://arxiv.org/pdf/2601.15434v1)

## Abstract
The evolution of digital manufacturing requires intelligent Question Answering (QA) systems that can seamlessly integrate and analyze complex multi-modal data, such as text, images, formulas, and tables. Conventional Retrieval Augmented Generation (RAG) methods often fall short in handling this complexity, resulting in subpar performance. We introduce ManuRAG, an innovative multi-modal RAG framework designed for manufacturing QA, incorporating specialized techniques to improve answer accuracy, reliability, and interpretability. To benchmark performance, we evaluate ManuRAG on three datasets comprising a total of 1,515 QA pairs, corresponding to mathematical, multiple-choice, and review-based questions in manufacturing principles and practices. Experimental results show that ManuRAG consistently outperforms existing methods across all evaluated datasets. Furthermore, ManuRAG's adaptable design makes it applicable to other domains, including law, healthcare, and finance, positioning it as a versatile tool for domain-specific QA.

## Full Text


<!-- PDF content starts -->

ManuRAG: Multi-modal Retrieval Augmented
Generation for Manufacturing Question Answering
Yunqing Li2†, Zihan Dong3†, Farhad Ameri1*, Jianbang Zhang2
1School of Manufacturing Systems and Networks, Arizona State
University, Mesa, AZ, 85212, USA .
2AI Technology Center, Lenovo, Morrisville, NC, 27560, USA .
3Computer Science Department, Georgia Institute of Technology,
Atlanta, GA, 30332, USA .
*Corresponding author(s). E-mail(s): Farhad.Ameri@asu.edu;
Contributing authors: yli59@lenovo.com; puma122707@gmail.com;
zhangjb2@lenovo.com;
†These authors contributed equally to this work.
Abstract
The evolution of digital manufacturing requires intelligent Question Answering
(QA) systems that can seamlessly integrate and analyze complex multi-modal
data, such as text, images, formulas, and tables. Conventional Retrieval Aug-
mented Generation (RAG) methods often fall short in handling this complexity,
resulting in subpar performance. We introduce ManuRAG, an innovative multi-
modal RAG framework designed for manufacturing QA, incorporating specialized
techniques to improve answer accuracy, reliability, and interpretability. To bench-
mark performance, we evaluate ManuRAG on three datasets comprising a total
of 1,515 QA pairs, corresponding to mathematical, multiple-choice, and review-
based questions in manufacturing principles and practices. Experimental results
show that ManuRAG 4consistently outperforms existing methods across all eval-
uated datasets. Furthermore, ManuRAG’s adaptable design makes it applicable
to other domains, including law, healthcare, and finance, positioning it as a
versatile tool for domain-specific QA.
Keywords:Retrieval augmented generation, Multi-modality, Manufacturing question
answering, Large language models
1arXiv:2601.15434v1  [cs.CE]  21 Jan 2026

1 Introduction
1.1 Background and Motivation
As Industry 4.0 continues to evolve and expand, QA systems play a crucial role in
leveraging big data in manufacturing. A QA system is a type of information retrieval
system designed to automatically answer questions posed by users in natural language.
These systems use a combination of natural language processing, machine learning, and
often knowledge bases or large datasets to understand the question and generate an
appropriate answer. By facilitating data-driven decision-making, QA systems provide
engineers, operators, and managers with accurate and relevant insights across diverse
data formats. This accessibility is important for intelligent manufacturing, where data
from sensors, machines, and design files must be integrated seamlessly to enable effi-
cient and accurate production. Integrating information retrieval with systems that
prioritize connectivity, automation, and secure collaboration enables manufacturers to
streamline the entire value creation process—from ideation and production to sustain-
ment and adaptation. These advanced capabilities promote self-organizing, intelligent
production systems and ensure reliable, secure communication across the manufactur-
ing value chain. Additionally, QA systems are crucial for achieving interoperability,
enabling seamless interaction between legacy and next-generation equipment and plat-
forms, and for evaluating the impact of external factors across a product’s entire
life cycle. Potential users of manufacturing QA systems include mechanical design
engineers calculating precise tolerances, production engineers optimizing assembly
processes, machine operators implementing temperature adjustments, quality assur-
ance teams verifying dimensional accuracy, and research teams exploring advanced
materials and techniques for manufacturing efficiency.
1.2 Challenges
The complexity of manufacturing data presents a unique challenge for effective infor-
mation retrieval and QA. Manufacturing documents span a wide array of data types,
including text, images, formulas, Computer-aided design (CAD) models, tables, sensor
data, and videos [1–3]. Each format holds critical information necessary for under-
standing and optimizing manufacturing processes, designs, and operational insights.
For example, text data may contain operational guidelines, maintenance logs [4], or
compliance records that require natural language processing for efficient retrieval.
Images, like defect photos or assembly line snapshots, demand computer vision tech-
niques to identify patterns or detect quality issues. Formulas and equations, embedded
in documentation, need precise recognition and interpretation to enable accurate repli-
cation and troubleshooting of complex processes. Together, these varied data formats
underscore the important role of advanced information extraction and processing
techniques for efficient, data-driven manufacturing insights.
The adoption of Manufacturing QA systems in industry remains in its early stages,
driven by the growing complexity of manufacturing processes and the increasing need
for advanced data management solutions. GPT-powered chatbots [5], such as those
offered by Streebo [6], represent some of the closest analogs. However, these tools
2

primarily focus on conversational interactions, lacking the robust multi-modal retrieval
and domain-specific reasoning capabilities required for comprehensive Manufacturing
QA applications.
Traditional information retrieval methods for Manufacturing QA are typically lim-
ited to either text-based or image-based data retrieval, often failing to capture the full
context and complexities of multi-modal information. This limitation can lead to ineffi-
ciencies in production workflows and hinder effective decision-making. For instance, in
design and development departments—especially within complex industries like auto-
motive manufacturing—retrieving relevant information across various data formats
presents considerable challenges. An engineer, for example, might need to answer a
multiple-choice question identifying which vehicle model employs a five-link suspen-
sion system. The relevant evidence may be distributed across heterogeneous document
types: legacy models may be documented primarily in textual specifications, more
recent models may embed suspension details within PDF files containing technical
drawings, and newer designs may rely on CAD models, exploded views, or annotated
images. Manually sifting through these formats is time-intensive, increases the risk of
missing key details, and becomes even more difficult with extensive documentation.
This example highlights a core issue: the pressing need for a robust retrieval system
that can seamlessly integrate and interpret text, visual data, and structured tables in
a way that meets the complex needs of design and manufacturing.
Several factors limit today’s information retrieval systems for QA tasks in the
manufacturing domain. Manufacturing data is diverse, complex, and technical, requir-
ing specialized processing for each type [7–9]. Textual data involves domain-specific
terminology, while visual data like figures, tables, and slides demand advanced inter-
pretation tools—capabilities lacking in many current systems. Another challenge is
transforming text and image data into a unified format computers can process, with
minimal information loss [10, 11]. These issues lead to misinterpretation or loss of crit-
ical details, hindering efficient data retrieval and integration. Advanced multi-modal
retrieval approaches are crucial to enable faster, accurate decision-making in modern
manufacturing.
1.3 Objectives
To overcome these limitations, we introduce a multi-modal RAG system tailored for
Manufacturing QA. This system seamlessly integrates and processes various data
types—such as images, formulas, tables, and text—to deliver a deeper and more con-
textually accurate understanding of manufacturing documents. By bridging the gap
between diverse data formats and complex manufacturing queries, multi-modal RAG
empowers users with actionable insights that are both reliable and comprehensive. It
enables more informed decision-making, streamlines workflows, improves operational
efficiency, and minimizes production errors, ultimately enhancing productivity and
driving innovation in manufacturing systems.
This paper leverages multi-modal data integration and retrieval to enhance
Manufacturing QA and decision support. The main contributions of this paper include:
3

1. Develops a complete multi-modal QA pipeline for the manufacturing domain,
from extracting information in text, images, formulas, and tables to generating
answers, to support product design, bottleneck identification, and data-driven
decision-making.
2. Compares standalone large language models (LLMs), RAG-augmented LLMs,
and multi-modal RAG-augmented LLMs, evaluating their strengths and limita-
tions across different question types, including mathematical, multiple-choice, and
review-based formats.
3. Introduces and compares indexing and retrieval strategies for multi-modal data,
examining how pre- and post-embedding alignment of textual and visual content
impacts QA accuracy and interpretability.
The remainder of the paper is organized as follows: in Section 2, we review
related work. Section 3 introduces how we construct ManuRAG. In Section 4, we
present experiments demonstrating the effectiveness of our approach. Finally, Section 5
discusses the limitations of our method and suggests directions for future work.
2 Related Work
2.1 Question Answering Systems
QA systems have evolved notably, leveraging a range of techniques to retrieve and
respond to queries [12]. Traditional expert systems, built on domain-specific exper-
tise, have long been used in QA tasks but often lack scalability and adaptability for
complex, data-rich environments. Large language models (LLMs) such as Generative
Pre-trained Transformer (GPT) [13] and Transformer-based models [14], including
Bidirectional Encoder Representations from Transformers (BERT), have introduced
major advancements in QA by using deep learning to interpret queries and retrieve
relevant information from extensive datasets. These models excel in understanding
context and language, making them effective in general-purpose QA. Multi-modal
models have further advanced the field, enabling visual question answering by combin-
ing image and text processing [15] and supporting QA with the integration of tabular
and textual data for contextually rich responses [16]. However, none of these models
are suitable for the manufacturing domain due to the complexity and technical nature
of manufacturing data. Manufacturing QA requires precise extraction and robust inte-
gration of diverse, multi-modal data types which presents a unique challenge beyond
the capabilities of existing QA models.
2.2 Information Extraction
Information extraction has evolved to handle diverse formats in structured and
unstructured data, including text, images, tables, and multimedia. Multimedia doc-
uments, such as PDFs, pose unique challenges because they integrate text, images,
tables, and sometimes audio or video. Advanced data extraction approaches use lay-
out detection models [17, 18] to identify content regions. Formula recognition relies on
deep learning methods [19–21]. Optical Character Recognition (OCR) tools [22, 23]
are essential for accurate text extraction across fonts and languages, providing detailed
4

information extraction capabilities for multimedia content. Data extraction in man-
ufacturing involves extracting information from text, images, and other data types
to support decision making. For text, Li et al. [24] developed methods to extract
knowledge from manufacturer websites, streamlining service discovery. For images,
Bhandarkar and Nagi [25] convert geometric data into higher-level manufacturing
insights. Despite the prevalence of multimedia in manufacturing, few studies have
focused on extracting information from these diverse sources, highlighting a research
gap.
2.3 Prompt Engineering and Retrieval Augmented Generation
While LLMs excel at general QA, they struggle in manufacturing contexts because
they often lack domain knowledge, cannot handle specialized terminology or interpret
technical data (e.g., CAD files, sensor readings, process parameters), and may produce
imprecise or hallucinated answers when fine-grained accuracy is required [26]. Manu-
facturing queries require precise, domain-specific knowledge that LLMs are prone to
producing plausible but incorrect answers (hallucinations) [27], cannot reliably pro-
vide without specialized grounding, making them insufficient for manufacturing QA
without further refinement.
Prompt engineering involves crafting prompts to guide LLMs toward accurate, con-
textually relevant responses. Techniques like Chain of Thought (CoT) [28] and Tree of
Thought [29], enhance this process by promoting multi-step reasoning, enabling LLMs
to handle complex queries more effectively. In manufacturing QA, prompt engineering
integrates industry-specific language and technical constraints into tailored prompts,
enabling LLMs to produce precise, domain-relevant answers and effectively manage
specialized terminology and complexity [30, 31].
Another powerful approach is RAG, which grounds LLM outputs in external
knowledge by retrieving relevant documents or databases from real-world manu-
facturing data. Tools like Llama-Index [32] and LangChain [33] enable efficient
text-based retrieval, while GraphRAG [34] introduces graph-based retrieval for rea-
soning over interconnected data. RAG methods, such as text-embedding RAG and
graph-embedding RAG, match queries with documents or capture relationships within
complex, interconnected information sources. In manufacturing, RAG has been applied
to retrieve domain-specific terms [35], integrate ontological knowledge graphs for engi-
neering analysis, and support materials design [36]. Multi-modal RAG—an emerging
extension that integrates text, images, and other data types to provide richer context
for complex questions [37]—has shown promise in healthcare but has not yet been
tailored for the manufacturing domain.
Multi-modal large language models (MLLMs) [38, 39] have substantially advanced
the integration of text and visual data. While these MLLMs excel in understanding
multi-modal inputs, they are constrained by input token limits, which pose challenges
when processing massive amounts of information.
Multi-modal RAG overcomes this limitation by indexing both textual and visual
content and fetching the most relevant context at inference time, thus enabling effi-
cient access to extensive external knowledge sources. Whereas standard RAG operates
5

solely on text, multi-modal RAG fuses image and document embeddings, typically fol-
lowing one of two strategies: (1) converting images into descriptive text for seamless
integration into text-based workflows, and (2) aligning text and image data in a shared
embedding space using multi-modal embeddings like Contrastive Language-Image Pre-
Training (CLIP) [40] and Jina-CLIP [41]. These methods enable models to retrieve
and reason across modalities, enhancing their performance on tasks that require both
visual and textual comprehension. While multi-modal RAG has shown applications
in domains such as healthcare, its potential remains underexplored in manufacturing.
The manufacturing domain demands tailored solutions capable of handling intricate
schematics, technical diagrams, and process-specific content with high precision and
contextual relevance.
3 Method
Traditional RAG systems are primarily designed to handle text inputs, often overlook-
ing the multi-modal nature of information present in real-world data. To overcome
this limitation, we propose ManuRAG, an extension of RAG that incorporates multi-
modality to process diverse data types from manufacturing documents. The MLLM
employed in ManuRAG accepts both plain text and images as inputs, enabling it
to process.pdffiles (PDFs) through a pipeline of object detection and text recogni-
tion models. This process effectively categorizes documents into distinct parts such as
figures, images, formulas, and textual content.
The ManuRAG framework comprises four key stages: multi-modal data extrac-
tion, indexing, retrieval, and answer generation. A state-of-the-art PDF extraction
pipeline—integrating advanced vision and segmentation models—helps robust figure
and content recognition [42]. Details of the multi-modal data extraction process are
presented in Section 3.1. Indexing mechanisms are discussed in Section 3.1.4, retrieval
strategies in Section 3.3, and answer generation approaches in Section 3.4.
Additionally, we introduce four variations of the ManuRAG framework. These
variations explore different strategies for integrating multi-modal data. The details of
these variations are presented in Section 3.5.
3.1 Multi-modal Data Extraction
Using original manufacturing files directly in RAG presents notable challenges. These
files often contain complex structures, including images, tables, formulas, and non-
standard text layouts, which make direct use infeasible for efficient information
retrieval. Converting PDFs to plain text frequently results in the loss of crucial
contextual and structural information, yielding low-quality data that undermines
the performance of manufacturing QA. To address these challenges, we focus on
extracting and organizing multi-modal data—text, images, tables, and formulas—from
manufacturing PDFs to create a robust input for RAG systems.
The primary goal of this section is to construct high-quality data representa-
tions that accurately capture fundamental manufacturing knowledge while minimizing
noise. This process begins with extracting and processing information from manu-
facturing PDF files to create separate text and image representations, ensuring the
6

preservation of critical data. Using the pdf-extract-kit framework [42], we employed
state-of-the-art methodologies for multi-modal data extraction. This includes layout
detection to maintain the document’s structure, formula extraction to keep mathe-
matical content, and image and table extraction to incorporate visual data. OCR was
applied to digitize textual content from non-selectable regions. These multi-modal
representations are then embedded and stored in a unified database that preserves
relationships between different modalities, enabling seamless integration and effective
use in downstream tasks.
3.1.1 Layout Detection
Accurate layout detection is crucial for maintaining the structural integrity of manu-
facturing PDFs and extracting relevant content effectively. This process entails parsing
documents into distinct elements while filtering out irrelevant contents, such as page
numbers or repeated headers, that could negatively impact QA performance. To this
end, we adopt the trained LayoutLMv3 base model [18], which was pre-trained on 11
million pages from the IIT-CDIP [43] corpora using AdamW (β 1= 0.9, β 2= 0.98), a
base learning rate of 1×10−4with linear warmup, and a weight decay of 1×10−2.
To detect regions, we apply the pretrained LayoutLMv3 token-classification head to
assign each token (or image patch) one of the layout labels (image, table, title, or text),
using the highest-probability prediction for each region. By isolating these regions
at the token level, our pipeline ensures that only semantically meaningful content is
indexed and passed to the RAG framework.
3.1.2 Formula Extraction
Our formula extraction pipeline comprises two sequential stages. First, we employ
a fine-tuned YOLOv8 [44] model with an input resolution of 1280 pixels, a confi-
dence threshold of 0.25, and an intersection over union threshold of 0.45 to accurately
locate both inline and display-style formulas. Second, each detected region is passed
to the UniMERNet model [45] to convert formula images directly into L ATEX source
code. This unified detection-and-recognition approach—illustrated by examples such
asE=σ
ϵfor Young’s modulus andJ=−D∂C
∂xfor Fick’s first law—ensures spa-
tial accuracy, minimizes recognition errors, and preserves the semantic fidelity of
mathematical expressions.
3.1.3 Image and Table Extraction
Our table extraction pipeline follows a two-stage process. First, we detect and seg-
ment table regions using PaddleOCR [46], which identifies cell boundaries and overall
table layout within page crops. Second, each segmented table image is processed by
StructEqTable [47] to convert detected tables into clean L ATEX, HTML, or Markdown
code. This streamlined approach preserves complex cell relationships, formatting, and
numerical precision, enabling our QA system to leverage richly structured tabular data
in downstream retrieval and reasoning.
7

3.1.4 Optical Character Recognition
After removing non-textual elements, the remaining page regions are processed by Pad-
dleOCR, which outputs Markdown-formatted text annotated with bounding-box and
element-type metadata. Each extracted text segment and visual element is assigned
a 128-bit Universally Unique Identifier (UUID), enabling consistent cross-referencing
and seamless integration within our QA pipeline.
3.2 Indexing
Indexing plays a crucial role in RAG, as it helps efficient and accurate retrieval of
relevant information during query resolution. In ManuRAG, indexing involves struc-
turing and storing multi-modal data, including both textual and visual information, to
enable seamless integration into subsequent retrieval and generation tasks. By project-
ing data into a high-dimensional embedding space, the system ensures compatibility
with similarity-based searches, which are important for effective information retrieval.
3.2.1 Text Vector Database
Textual data, including paragraphs, formulas and links to the images, is pre-processed
dividing it into manageable chunks to preserve contextual relevance and coherence.
This segmentation process is governed by two key parameters: the chunk size, which
defines the maximum length of each segment, and the chunk overlap, which determines
the degree of overlap between consecutive chunks to maintain continuity of context.
Given a textual inputText, the division into overlapping chunks can be expressed as:
Chunks =
Ci:Ci=Text[i:i+S], i= 0, S−O,2(S−O), . . .	
(1)
whereSrepresents the chunk size,Odenotes the chunk overlap,C iis thei-th
chunk, and|Text|is the total length of the text.
We model each chunkC ias
Ci= 
Text i,Imgi
,Imgi={Imgi1, . . . ,Imgibi}.(2)
Here,Text iis the full textual content of thei-th chunk, andImgiis the set of all
images referenced in that chunk (withImgijdenoting thej-th image).
The chunks are projected into the embedding space using a LLM-based text
embedding model Embed Text. The embedding for each chunkC iis represented as:
vCi= Embed Text(Ci) (3)
These embeddings, along with their corresponding text chunks, are stored in a text
vector database, enabling efficient retrieval during the query resolution process.
3.2.2 Image Vector Database
For visual data—including extracted figures and tables—a multi-modal embedding
model is employed to project all content into the same space as the text embeddings.
This model handles bothTextandImgmodalities, enabling cross-modal search (for
8

example, text-to-image and image-to-text retrieval). To align with CLIP’s native train-
ing resolution, each visual elementImgjis first resized so that its shorter side measures
224 pixels and then center-cropped to a 224×224 pixel square. It is then encoded
into an embedding vectorv Imgjvia the multi-modal embedding function Embed MM:
vImgj= Embed MM 
Imgj
(4)
The resulting image embeddings, along with their associated image data, are stored
in an image vector database. This database is designed to support the retrieval of
visual information and to complement the textual data during queries.
Although the multi-modal embedding model is capable of processing both text
and images, its understanding of complex and lengthy textual content is not as good
as that of most LLM-based text embedding models. The text embedding model,
which is trained specifically on extensive textual datasets, provides a deeper under-
standing of intricate text structures and semantics. To capitalize on the strengths of
both approaches, some variations of ManuRAG maintain separate vector databases
for text and images. The details are explained in Section 3.5. This dual-database
approach enables ManuRAG to effectively leverage the specialized capabilities of the
text embedding model and the flexibility of the multi-modal embedding model.
3.3 Retrieval
Retrieval is a key part of RAG system, enabling the system to identify and extract the
most relevant information from the indexed data. In ManuRAG, retrieval serves the
critical function of narrowing down the multi-modal data—text and images—stored
in vector databases, so that only the most pertinent chunks are provided as context
to the language model for producing answers to manufacturing-related questions. By
incorporating retrieval, the system ensures efficiency and accuracy in handling large
and diverse datasets.
The retrieval process is based on calculating the similarity between the embedding
of an input manufacturing query and the embeddings of stored data. The similarity
metric used is cosine similarity, defined as:
Sim(q, v) =q·v
∥q∥∥v∥(5)
whereqis the embedding of the input manufacturing questionQ, andvis the
embedding vector of each stored item (text chunk or image). Cosine similarity mea-
sures the angular similarity between the query embedding and stored embeddings,
providing a reliable estimate of their relevance.
3.3.1 Retrieval from the Text Vector Database
For textual data, the system calculates cosine similarity between the query embedding
and each indexed chunk embedding. Using Eq. 5, the top-Ktext chunks are retrieved
as
Top@Ktext= arg topK
{Ci} 
Sim(q 1, vCi)
,(6)
9

whereq 1is the query embedding in text space, andv Ciis the embedding of chunk
Ci. Once theseKchunksC i1, . . . , C iKare selected, the set of all images they reference
is
K[
r=1Imgir=K[
r=1{Imgir1,Imgir2, . . .}.(7)
3.3.2 Retrieval from the Image Vector Database
For visual data, the input query is encoded into the same embedding space as the image
embeddings using a multi-modal embedding model. Cosine similarity is then calculated
between the query embedding and the image embeddings stored in the image vector
database. The topkimages with the highest similarity scores are retrieved:
Top@KImg= arg topK
{Imgj} 
Sim(q 2, vImgj)
.(8)
Here,q 2is the query embedding in the multi-modal space, andv Imgjis the embed-
ding of imageImgj, enabling direct retrieval of relevant visual data via embedding
similarity.
3.4 Answer Generation
Answer generation is the final stage of the RAG pipeline, where the system synthe-
sizes a response to the input query by leveraging the relevant data retrieved during the
retrieval phase. This step involves integrating the retrieved textual and visual infor-
mation with the manufacturing question and processing it through an LLM or MLLM
to produce an accurate and contextually appropriate answer.
3.4.1 Answer Generation Using Text-Only Retrieval Data
In the first strategy, the system combines the retrieved textual data, the manufacturing
question, and the domain-specific prompt as input to a large language model (LLM)
optimized for text-based processing. This approach excludes any retrieved visual data,
focusing solely on textual context for answer synthesis. The process can be expressed
as
ALLM= LLM(concat(P, Q, Top@K text)),(9)
wherePis the specially designed prompt for questions in the manufacturing
domain,Qrepresents the input query. This strategy is efficient and suitable for cases
where the retrieved textual data alone provides sufficient context to answer the query.
However, it does not account for visual information, which may limit its effectiveness
for multi-modal queries.
3.4.2 Answer Generation Using Multi-modal Retrieval Data
In the second strategy, the system combines the textual data, the visual data and
the manufacturing question as input to an MLLM. This approach incorporates both
10

text and images, enabling the model to consider visual context alongside textual
information for answer generation. The generated answer can be formulated as follows:
AMLLM = MLLM(concat(P, Q, Top@K text, Top@K Img,K[
r=1Imgir)).(10)
This strategy generates answers by using relevant images from two sources: the
images referenced in the top-Ktext chunks,SK
r=1Imgir, and the images retrieved
directly from the image vector database, Top@KImg. Note that either of these image
sets may be empty if no visual data is linked, indexed, or successfully retrieved.
We include our designed prompts for different types of manufacturing questions in
Appendix A, and further describe the differences among the ManuRAG variations in
Section 3.5.
3.5 ManuRAG Variations
We propose four variations of the ManuRAG framework to investigate the optimal
approach for aligning textual and visual data in manufacturing QA. Specifically, the
framework explores different strategies for integrating text and image data—whether
alignment should occur before embedding, within the embedding space, or after
embedding. Figure 1 highlights the workflows of each variation, emphasizing the dif-
ferences in multi-modal data handling, indexing strategies, and answer generation
methods.
ManuRAG 1processes both textual and visual data by embedding textual infor-
mation into a text vector database and visual data into an image vector database.
Answer generation is performed using an MLLM.
ManuRAG 3extends ManuRAG 1by prioritizing the retrieval of visual data linked
within the retrieved text chunks. It uses both text and image vector databases for
indexing but focuses on retrieving images referenced in the text.
ManuRAG 2simplifies ManuRAG 3by eliminating the image vector database. It
operates exclusively on textual data, storing embeddings in a text vector database,
and generates answers using a LLM.
ManuRAG 4introduces a unique step in multi-modal data processing. After iden-
tifying visual data referenced in the text, the visual content (for example, figures and
tables) is transformed into descriptive text using an MLLM. This transformed text
is then indexed in the text vector database and the answers are generated using an
LLM (or optionally an MLLM). This variation aligns textual and visual data before
embedding by converting visual data into a text-based representation. However, this
transformation may result in some information loss, especially when the original visual
context is critical.
4 Experiments
In this section, we present experiments to evaluate the effectiveness of the proposed
methods for Manufacturing QA using real-world datasets to ensure practical relevance.
The study focuses on evaluating how different RAG systems improve the performance
11

Fig. 1Four Types of ManuRAG Frameworks
of Manufacturing QA and examining the influence of context precision and context
recall on the effectiveness of these systems in retrieving relevant information. The
experimental settings, including datasets, baselines, and configurations, are detailed
in Section 4.1, while the evaluation metrics are introduced in Section 4.2. The general
performance of the models and the context-retrieval-related performance are discussed
in Sections 4.3 and 4.4.
12

4.1 Settings
4.1.1 Baselines and Configurations
We evaluate multiple variations of ManuRAG against alternative QA solutions to
demonstrate the performance improvements and unique advantages offered by our
approach. All models in the comparison leverage the same underlying MLLM, GPT-
4O, ensuring a consistent foundation for multi-modal capabilities. Our evaluation
encompasses the baseline GPT-4O, both in its standard configuration and with
manufacturing-specific prompts, as well as various RAG models, including traditional
and multi-modal RAG approaches.
All RAG models employ Ada v2 for text indexing and retrieval, CLIP for image
indexing and retrieval when an image vector store is used, and GPT-4O for generat-
ing final answers. Both the ManuRAG variants and the RAG methods are built using
the LlamaIndex framework [32]. The source documents for RAG context retrieval
are derived from the same PDF version of a textbook,Fundamentals of Modern
Manufacturing[48], which is a widely recognized textbook covering engineering
materials, manufacturing processes, and electronics technologies.
All experiments are conducted in Google Colab withk= 1,S= 600, andO= 100,
following Edge et al. [49] and Lewis et al. [50]. Except for the baseline models, all
other models use a unified prompt configuration, as detailed in Appendix A. Below
are additional details about the benchmark models:
•GPT-4O: Direct QA using GPT-4O without any specialized prompts.
•GPT-4O (CoT): Direct QA enhanced with Chain-of-Thought (CoT) prompting
to enable step-by-step reasoning with GPT-4O.
•RAG: RAG applied to raw text directly extracted from the files, excluding any
visual data.
•RAG hq: A high-quality RAG framework utilizing structured text extracted using
the method outlined in Section 3.1.
•ManuRAG 1−3: Variants of multi-modal RAG, as described in Section 3.5.
•ManuRAG 4: A variant of multi-modal RAG introduced in Section 3.5, incor-
porating GPT-4O to transform image data into text for enriched contextual
understanding.
4.1.2 Evaluation Datasets
We evaluate our benchmarks using three datasets consisting of 1,515 question-answer
pairs derived from the official solutions of Fundamentals of Modern Manufacturing [48].
Notably, this dataset is excluded from the content used for RAG retrieval. The datasets
are organized as follows:
•MathQ: 309 mathematical QA pairs requiring factual accuracy and semantic
understanding, often multi-modal with formulas, tables, and figures. Preprocessing
excludes excessively long questions/answers (over 600 tokens) or trims sub-questions
for LLM compatibility.
13

•MCQ: 471 multiple-choice questions (MCQs) evaluating the ability to select the
correct option—some requiring multi-modal analysis. Answers are formatted as
distinct choices (a–d) and include explanations to ensure precise evaluation.
•RQ: 735 review QA pairs focusing on comprehension and reasoning over textual
content to evaluate conceptual understanding.
Examples of each type of datasets are provided in Appendix B.
4.2 Evaluation Metrics
We adopt five evaluation metrics from RAGAS [51] with GPT-4 to comprehensively
evaluate the performance of QA models which are Factual Correctness, Semantic
Similarity, Recall-Oriented Understudy for Gisting Evaluation (ROUGE), Context
Precision and Context Recall. The metrics and their formulas are as follows.
4.2.1 Factual Correctness
Factual correctness measures the F1 score of answers by comparing them to the ground
truth and is applied across all evaluated models. This metric evaluates the degree to
which the response aligns with the reference, with scores ranging from zero to one,
where higher values indicate better performance. GPT-4 is employed to decompose
both the response and the reference into distinct claims, with natural language infer-
ence used to evaluate the factual overlap between them. By balancing precision and
recall through the F1 score, this metric provides a comprehensive measure of the
accuracy of the generated answers relative to the reference.
True Positive (TP) represents the number of claims in the answer that are present
in the reference. False Positive (FP) indicates the number of claims in the response
that are not present in the reference, while False Negative (FN) corresponds to the
number of claims in the reference that are not present in the answer. Using these
values, the precision, recall, and F1 score are calculated as follows:
Precision =TP
TP + FP(1)
Recall =TP
TP + FN(2)
Factual Correctness (F1 Score) =2×Precision×Recall
Precision + Recall(3)
4.2.2 Semantic Similarity
Semantic similarity evaluates how semantically close the generated answer is to the
reference answer, which is is applied across all evaluated models. It is measured using
a cross-encoder model that computes a semantic similarity score ranging from zero
to one. A higher score indicates a better semantic alignment between the generated
answer and the ground truth.
14

4.2.3 Recall-Oriented Understudy for Gisting Evaluation
ROUGE measures the lexical overlap between generated answers and reference
answers, and is used here to evaluate text-based responses for MCQs. It measures
the overlap ofn-grams (n= 1) using the F1 score to quantify similarity between the
response and the reference. The formula for ROUGE is defined as follows:
Precision n=Number of overlappingn-grams
Total number ofn-grams in the generated answer(11)
Recall n=Number of overlappingn-grams
Total number ofn-grams in the reference(12)
ROUGE n=2×Precision n×Recall n
Precision n+ Recall n(13)
This metric provides a straightforward and effective way to test lexical similarity
in MCQ dataset, where exact matches of words or phrases are crucial for evaluation.
4.2.4 Context Precision
Context precision measures the relevance of retrieved context chunks to the query.
It evaluates the proportion of relevant chunks among the retrieved contexts in RAG-
based models, including RAG 1, RAG 2, and ManuRAG 1to ManuRAG 4. This metric
incorporates a weighted precision approach, wherev krepresents the weight assigned
to each chunk at rankk. The precision at rankk(top k= 3) is computed as:
Precision@k =TP@k
TP@k + FP@k(1)
Using these precision values, the context precision for the topKresults is calculated
as:
Context Precision@K =PK
k=1(Precision@k×v k)
Total number of relevant items in Top@K(2)
4.2.5 Context Recall
Context recall measures the percentage of relevant context retrieved compared to the
total relevant context available among the retrieved contexts in RAG-based models.
Ground Truth (GT) refers to the set of reference claims or facts that are correct and
are used as a standard for evaluation. This metric calculates how many of the claims
in the ground truth are attributable to the retrieved contexts. The formula for context
recall is given as:
Context Recall =|GT attributed to retrieved contexts|
|Total number of claims in GT|(1)
15

Table 1Comparison of Different Metrics Across Methods for Manufacturing
QA (Highest performance for each dataset is highlighted inbold). We
compare six benchmark models; the method that integrates images and text
before indexing (ManuRAG 4) outperform those that treat modalities
separately, achieving the optimal factual-correctness scores across all tasks.
Category Factual Correctness (%) Semantic Similarity (%)
MathQ
GPT-4O 49.91 87.03
GPT-4O(CoT) 56.22 86.85
RAG 59.51 87.66
RAG hq 59.4887.90
ManuRAG 1 19.14 77.03
ManuRAG 2 56.69 87.32
ManuRAG 3 55.31 87.32
ManuRAG 4 62.0387.82
MCQ
GPT-4O 6.99 77.60
GPT-4O(CoT) 78.58 94.79
RAG 85.22 96.52
RAG hq 86.38 96.62
ManuRAG 1 86.82 95.27
ManuRAG 2 60.54 91.20
ManuRAG 3 61.62 90.15
ManuRAG 4 87.61 96.82
RQ
GPT-4O 38.01 89.93
GPT-4O(CoT) 40.47 89.92
RAG 45.51 91.08
RAG hq 45.64 91.12
ManuRAG 1 45.34 90.97
ManuRAG 2 40.5391.15
ManuRAG 3 39.10 91.13
ManuRAG 4 45.7591.00
4.3 Overall Performance
The performance of Manufacturing QA models, shown in Table 1, demonstrates that
ManuRAG 4consistently achieves the best performance across all three datasets. It
achieves the highest factual correctness in MathQ, outperforming GPT-4O and GPT-
4O with CoT, indicating its strength in handling mathematical reasoning. For MCQ
tasks, it also scores highest in factual correctness and ROUGE (Table 2). In RQ tasks,
while the differences across models are smaller due to the simpler, text-based nature
of the questions, ManuRAG 4still achieves the highest factual correctness, confirming
its overall effectiveness.
16

ManuRAG 4’s broader superiority can be attributed to two key mechanisms. First,
its position-aware, unified indexing converts each image into descriptive text and
embeds it at its original location alongside surrounding paragraphs, thereby preserv-
ing the natural document flow; this co-location of visual descriptions and adjacent
text is particularly well suited to domain-rich documents (for instance, textbooks or
technical papers) and enables more effective matching during retrieval. Second, by
removing the separate image index, ManuRAG 4minimizes the retrieval of irrelevant
diagrams or icons—reducing “visual noise” and sharpening both context precision and
factual correctness. Together, these advantages allow ManuRAG 4to strike a more
effective balance between relevance and coverage than variants that either maintain
independent image stores or forgo positional integration of visual content.
While ManuRAG 4consistently delivers the highest results, other configurations,
such as ManuRAG 1and ManuRAG 2, show strengths in specific tasks. For example,
ManuRAG 1performs well on MCQ and RQ tasks by effectively leveraging image
retrieval through links embedded in the textual context but performs poorly on
MathQ, indicating limitations in its ability to handle mathematical reasoning with
this retrieval method. The low performance of ManuRAG 3suggests that retrieving
more images does not necessarily lead to better answers for Manufacturing QA. We
include QA example comparisons across different models in Appendix B.
Across all datasets, the inclusion of CoT reasoning improves GPT-4O’s perfor-
mance, showcasing its effectiveness in enhancing logical reasoning and precision for
QA tasks. RAG hqdemonstrates better performance than RAG, highlighting the
effectiveness of the multi-modal data extraction method.
Semantic similarity scores are generally high across all models and datasets, reflect-
ing alignment with reference answers. However, semantic similarity alone does not
guarantee factual correctness, particularly in MathQ and MCQ, where precise reason-
ing and retrieval are critical. While semantic similarity remains useful for assessing
general alignment, factual correctness is a more reliable indicator of performance in
Manufacturing QA tasks.
4.4 Context-based Performance
While factual correctness evaluates the alignment of generated answers with GT, it
does not evaluate the quality of retrieved context, which is critical in RAG-based
models. Context precision and recall provide additional insights by measuring the
relevance and coverage of the retrieved context, offering a deeper understanding of
how well the model supports its generated answers. This section evaluates both the
quality of the generated answers and the completeness of the retrieved context, as
shown in Table 3.
By aligning text and image data before embedding, ManuRAG 4achieves superior
integration and strong performance across tasks. It demonstrates robust perfor-
mance across datasets, achieving the highest context precision for MathQ and MCQ,
high recall for MCQ and RQ, and effectively balancing relevance and coverage for
retrieval-augmented tasks. RAG hqexcels in context precision for RQ, emphasizing the
importance of high-quality indexed textual data, while standard RAG, with shorter
indexed text, achieves the highest recall for MathQ and competitive recall for MCQ
17

Table 2ROUGE Scores for MCQs
Across Different Models. The
highest scores are achieved by
ManuRAG 4, and RAG hq,
suggesting that joint text–image
embedding or high-quality text
indexing is associated with more
faithful multiple-choice answer
generation.
Model ROUGE (%)
GPT-4O 10.35
GPT-4O(CoT) 77.61
RAG 86.10
RAG hq 86.71
ManuRAG 1 81.19
ManuRAG 2 63.13
ManuRAG 3 58.88
ManuRAG 4 86.77
and RQ, highlighting a trade-off between precision and recall. In contrast, ManuRAG 1,
ManuRAG 2, and ManuRAG 3show competitive precision for RQ but struggle with
MathQ precision and lower recall for MathQ and MCQ, reflecting the limitations of
separate vector stores for text and image data.
Across our evaluated benchmarks, ManuRAG 4and RAG hqconsistently rank
among the top performers in context precision and factual correctness. Although
converting images into text inevitably results in some loss of visual information,
ManuRAG 4nevertheless surpasses the other variants across different manufacturing
question types. This observation indicates that—even when certain visual cues (for
example, precise axis labels or schematic annotations) are simplified—the joint integra-
tion of textual and visual content prior to retrieval is more effective than maintaining
separate image indices. When images are indexed independently, they lose their con-
textual linkage with the surrounding text; in manufacturing documents, many images
and accompanying paragraphs often convey critical information only when considered
together. Furthermore, independent retrieval of images can introduce “visual noise”
in the form of irrelevant diagrams or icons, which may undermine the benefits of
preserving complete visual data.
5 Conclusion and Future Work
Advances in digital manufacturing demand intelligent QA systems capable of integrat-
ing and analyzing complex multi-modal data. Traditional RAG methods often struggle
to process such data effectively. ManuRAG tackles these challenges by presenting the
multi-modal RAG framework, specifically designed for Manufacturing QA. Experi-
mental results demonstrate that ManuRAG 4outperforms existing methods, offering
valuable insights into selecting appropriate RAG strategies and aligning image and
text data for diverse manufacturing scenarios. Beyond manufacturing, ManuRAG’s
18

Table 3Context Recall and Precision for MathQ, MCQ, and RQ. Methods that integrate text
and image embeddings prior to retrieval are associated with higher context precision and recall,
demonstrating the benefit of preserving co-located semantics; text-only retrieval also performs
strongly when high-quality textual context is available.
Context Precision (%) Context Recall (%)
Category MCQ MathQ RQ MCQ MathQ RQ
RAG 81.95 92.56 92.24 33.3967.8381.23
RAG hq 85.35 93.2093.7434.61 57.00 81.27
ManuRAG 1 79.41 33.66 92.79 33.36 30.95 81.47
ManuRAG 2 80.89 33.33 92.52 32.01 22.98 80.63
ManuRAG 3 79.41 33.33 92.65 33.44 20.47 80.86
ManuRAG 4 86.84 93.5391.8437.8862.2681.56
adaptable design positions it as a versatile tool for domain-specific QA in areas such
as law, healthcare, and finance.
Experimental results show that effective context alignment between textual and
visual data improves manufacturing QA performance. In contexts such as textbooks,
images and accompanying text often share semantic content; aligning these modal-
ities before embedding preserves their co-location and produces more coherent and
relevant representations. Simply increasing the number of retrieved images does not
guarantee better performance; retrieval strategies should prioritize the relevance and
integration quality of images. For mathematical reasoning, robust multi-modal align-
ment significantly enhances performance. For tasks heavily reliant on text, simpler
retrieval methods can achieve competitive results if high-quality textual contexts are
well-indexed.
However, even ManuRAG 4, the top-performing configuration, still falls short of
perfect accuracy, particularly on queries that require precise interpretation of techni-
cal drawings and process schematics—showing that domain-specific QA remains an
open challenge and highlighting the usefulness of our benchmark in exposing these
gaps. These limitations point to three improvement paths: fine-tune, adapt, or replace
existing multi-modal encoders—such as Jina-CLIP [41] on manufacturing-specific cor-
pora to enhance visual–text alignment and achieve more accurate, domain-tailored
image or text retrieval; integrate advanced retrieval designs from recent RAG frame-
works [52, 53] to balance precision and recall; and extend the system to support
richer data types and document formats—such as CAD models [54], instructional
videos [55], enterprise work orders, maintenance manuals, and time-series sensor
logs—by developing specialized embedding strategies [56] to ensure robustness across
diverse manufacturing information sources.
Appendix A Prompt Design
The following shows the prompt designed for MathQ:
19

MathQ Prompt
As a manufacturing expert, solve the math question step by step using the
provided context and your knowledge of manufacturing calculations. Make sure
to clearly explain your reasoning:
{context str}
Question:
Query:{query str}
Answer:
The following presents the prompt designed for MCQ:
MCQ Prompt
As a manufacturing expert, answer the following multi-choice questions based
on the provided context step by step. Select the correct letter choices only.
Answer format:
“Explanation”: “Your explanation here.”, “YourChoice”: “Answer. (d), (e),
and (g).”
{context str}
Answer the following question:
Query:{query str}
Answer:
The following shows the prompt designed for RQ:
RQ Prompt
You are an expert in manufacturing major. You need to answer the following
short response question based on the context and your knowledge of manu-
facturing processes and materials. Please provide a detailed explanation and
justify your answer. Be concise, clear, and accurate.
{context str}
Answer the following question:
Query:{query str}
Answer:
Appendix B QA Examples
This section presents QA examples of different types and their answers generated by
three different models, specifically comparing GPT-4O (CoT), traditional RAG, and
the best-performing ManuRAG variant.
20

B.1 MathQ Example
The following is an example of QA pairs from MathQ. ManuRAG 4generates correct
answers, whereas RAG and GPT-4O (CoT) fail. GPT-4O (CoT) cannot reference
Table 4.1 or the associated formulas from the manufacturing documents, limiting its
accuracy. While RAG retrieves relevant formulas from the text data, it is constrained
by its inability to access visual information, such as the data in Table 4.1 (from the
textbook).
MathQ Problem
The starting diameter(D) of a shaft is 25.00 mm. This shaft is to be inserted
into a hole in an expansion fit assembly operation. To be readily inserted, the
shaft must be reduced in diameter by cooling. Determine the temperature to
which the shaft must be reduced from room temperature(T) 20℃to reduce its
diameter to 24.98 mm. Refer to Table 4.1.
Correct Answer
For steel,α= 12×10−6mm/mm/℃according to Table 4.1. Revise Eq. (4.1)
to:
D2−D 1=αD 1(T2−T1)
Substitute the known values:
24.98−25.00 = 12×10−6·25.00·(T 2−20)
Simplify:
−0.02 = 300×10−6·(T2−20)
−0.02 = 0.0003(T 2−20)
−0.02 = 0.0003T 2−0.006
−0.02 + 0.006 = 0.0003T 2
−0.014 = 0.0003T 2
T2=−0.014
0.0003
T2=−46.67◦C
21

GPT-4O (CoT)
To solve this problem, we need...
Calculation:Step 1: Understand the Concept of Thermal Contraction...
Step 4: Use the Provided Table to Find the Coefficient of Linear Expansion...
Since specific data (Table 4.1) is mentioned but not provided here , let’s
assume...
Step 6: Calculate the Final Temperature...
Using the assumed coefficient of linear expansion, the calculated temperature
is:
T2=−52.73◦C
Conclusion:The shaft must be cooled to approximately−52.73◦Cto achieve
the desired diameter reduction from 25.00 mm to 24.98 mm.
RAG
To solve this problem, we need...
CalculationThe formula for the change in diameter due to thermal expansion
or contraction is given by:
D2−D 1=α·D 1·(T2−T1)
First, calculate the change in diameter... Using the coefficient of linear thermal
expansion (α), the temperature is calculated as:
T2=−48.39◦C
Conclusion:The shaft must be cooled to approximately−48.39◦Cto achieve
the desired diameter reduction for the expansion fit assembly operation.
22

ManuRAG 4
To solve this problem, we need...
The change in diameter due to thermal contraction is given by:
D2−D 1=α·D 1·(T2−T1)
Calculation:1. Calculate the change in diameter: . . . 2. Substitute the values
into the formula: . . . 3. Calculate the temperature change:
T2=−0.02
0.0003+ 20
T2=−46.67◦C
Conclusion:The shaft must be cooled to approximately−46.67◦Cto reduce
its diameter from 25.00 mm to 24.98 mm for the expansion fit assembly
operation.
B.2 MCQ Example
This question represents a MCQ problem for the manufacturing materials. GPT-
4O (CoT) fails to generate a completely correct answer, whereas both RAG and
ManuRAG 4provide accurate responses. RAG and ManuRAG4 achieves the cor-
rect result by effectively utilizing retrieved textual context from the manufacturing
textbook.
MCQ Problem
Which of the following materials are used as fibers in fiber-reinforced plas-
tics (FRPs) (select more than one): (a) aluminum oxide, (b) boron, (c)
carbon/graphite, (d) epoxy, (e) Kevlar 49, (f) S-glass, and (g) unsaturated
polyester.
Correct Answer
The correct materials used as fibers in FRPs are:(a),(b),(c),(e), and(f) .
23

GPT-4O (COT)
Explanation:Fiber-reinforced plastics (FRPs) are composite materials made
of a polymer matrix reinforced with fibers to enhance their mechanical proper-
ties. In the given options: -(a)Aluminum oxide is typically not used as a fiber
in FRPs; rather, it’s more common as a ceramic or in powder form for rein-
forcement. -(b)Boron is indeed used as a fiber in certain high-performance
FRPs. -(c)Carbon/graphite is one of the most common and effective fiber
reinforcements. -(d)Epoxy is generally a type of polymer matrix in FRPs,
not a fiber. -(e)Kevlar 49 is an aramid fiber known for high strength and is
used in FRPs. -(f)S-glass is a specialized type of glass fiber used in FRPs for
higher strength. -(g)Unsaturated polyester is also a type of polymer matrix
and not used as a fiber.
Your Choice: (b),(c),(e), and(f) .
RAG
Explanation:Fiber-reinforced plastics (FRPs) are composite mate-
rials that consist of a polymer matrix reinforced with fibers.
According to the provided context , the principal fiber materials used in FRPs
include glass (such as E-glass and S-glass), carbon, and Kevlar 49. Addition-
ally, less common fibers mentioned are boron, silicon carbide (SiC), aluminum
oxide, and steel. Epoxy and unsaturated polyester are not fibers; they are
types of polymer matrices used in FRPs.
Your Choice: (a),(b),(c),(e), and(f) .
ManuRAG 4
Explanation:The question asks for materials used as fibers in fiber-reinforced
plastics (FRPs). Based on the provided context , the materials used as fibers in
FRPs include: -(a)Aluminum oxide (Al 2O3) is mentioned as a less common
fiber material in FRPs. -(b)Boron is mentioned as a fiber used in advanced
composites. -(c)Carbon/graphite is a principal fiber material in FRPs. -(e)
Kevlar 49 is listed as a principal fiber material in FRPs. -(f)S-glass is a type
of glass fiber used in FRPs.
Epoxy and unsaturated polyester are not fibers; they are matrix materials used
in FRPs.
Your Choice: (a),(b),(c),(e), and(f) .
B.3 RQ Example
The following is an example of review question in the dataset. GPT-4O (CoT) provides
a comprehensive list but includes less relevant variables such as pressure and barriers,
leading to less focus. RAG identifies the key variables accurately, leveraging retrieved
24

context but lacks consistency in prioritizing variables. ManuRAG 4achieves the most
concise and correct explanation by effectively retrieving and integrating context from
relevant sources.
RQ Problem
What are the important variables that affect mass diffusion?
Correct Answer
According to Fick’s first law, mass diffusion depends on the following variables:
•Diffusion Coefficient: The diffusion coefficient rises rapidly with temper-
ature, so temperature is an important variable.
•Concentration Gradient (dc/dx): The difference in concentration of
diffusing species across a distance.
•Contact Area (A): The area through which diffusion occurs.
•Time: The duration of the diffusion process.
GPT-4O (CoT)
Explanation:Mass diffusion is a critical process in manufacturing and
materials science. The variables that affect diffusion include:
1.Temperature: Temperature is perhaps the most influential factor affect-
ing diffusion. Higher temperatures increase atomic motion, accelerating
diffusion.
2.Concentration Gradient: The concentration gradient, or the difference
in concentration of diffusing species between two points, directly affects
diffusion rates.
3.Diffusing Species: The size, mass, and chemical nature of the particles
being diffused influence diffusion rates.
4.Medium or Matrix Structure: The structure of the material through
which diffusion occurs is critical.
5.Pressure: In gases and, to some extent, in liquids, pressure can influence
diffusion rates.
6.Time: Over longer periods, the extent of diffusion naturally increases as
particles spread through the medium.
7.Presence of Barriers or Interfaces: Physical barriers or interfaces, such
as coatings or phase boundaries, can alter the pathway and rate of diffusion.
25

RAG
Explanation:Mass diffusion in solids, particularly metals, is influenced by
several key variables:
1.Temperature: A critical factor affecting diffusion, as higher temperatures
increase atomic mobility.
2.Concentration Gradient (dc/dx): Represents the difference in concen-
tration of diffusing species across a distance.
3.Diffusion Coefficient: A material-specific parameter that quantifies how
easily atoms can move through a material.
4.Crystal Structure and Defects: Vacancies and other imperfections in
the crystal structure facilitate atomic motion.
5.Material Properties: Atomic size, bonding, and lattice structure influence
diffusion behavior.
6.Surface Area (A): The area of the boundary through which diffusion
occurs.
ManuRAG 4
Explanation:The important variables that affect mass diffusion include:
1.Temperature: A critical factor in diffusion. Higher temperatures increase
thermal agitation, allowing atoms to move more freely.
2.Concentration Gradient (dc/dx): The driving force for diffusion, repre-
senting the difference in concentration of atoms across a distance.
3.Diffusion Coefficient: A material-specific parameter that quantifies the
ease with which atoms can move through a material.
4.Time: The duration over which diffusion occurs affects the extent of
diffusion.
5.Area of the Boundary (A): The area through which diffusion occurs
impacts the rate of mass transfer.
References
[1] Tao, F., Qi, Q., Liu, A., Kusiak, A.: Data-driven smart manufacturing. Journal
of Manufacturing Systems48, 157–169 (2018)
[2] Letaief, M.B., Tlija, M., Louhichi, B.: An approach of cad/cam data reuse for
manufacturing cost estimation. International Journal of Computer Integrated
Manufacturing33(12), 1208–1226 (2020)
[3] He, C., Liu, M., Zhang, Y., Wang, Z., Hsiang, S.M., Chen, G., Li, W., Dai,
G.: Space–time–workforce visualization and conditional capacity synthesis in
uncertainty. Journal of Management in Engineering39(2), 04022071 (2023)
26

[4] Karray, M.H., Ameri, F., Hodkiewicz, M., Louge, T.: Romain: Towards a bfo
compliant reference ontology for industrial maintenance. Applied Ontology14(2),
155–177 (2019)
[5] OpenAI: ChatGPT (AutoCAD Master). https://openai.com/chatgpt. AI-
powered assistant for AutoCAD and design-related guidance. Accessed January
26, 2025. (2025)
[6] Streebo: ChatGPT in Manufacturing: Use Cases. https://www.streebo.com/
chatgpt-manufacturing-use-cases. Accessed: 2025-01-26
[7] Starly, B., Koprov, P., Bharadwaj, A., Batchelder, T., Breitenbach, B.: “unreal”
factories: Next generation of digital twins of machines and factories in the indus-
trial metaverse. Manufacturing Letters37, 50–52 (2023) https://doi.org/10.1016/
j.mfglet.2023.07.021
[8] Li, Y., Liu, X., Starly, B.: Manufacturing service capability prediction with graph
neural networks. Journal of Manufacturing Systems74, 291–301 (2024)
[9] Bharadwaj, A.G., Starly, B.: Knowledge graph construction for product designs
from large cad model repositories. Advanced Engineering Informatics53, 101680
(2022)
[10] Moro, G., Salvatori, S., Frisoni, G.: Efficient text-image semantic search: A
multi-modal vision-language approach for fashion retrieval. Neurocomputing538,
126196 (2023) https://doi.org/10.1016/j.neucom.2023.03.057
[11] Li, Y., Tang, Z., Zhuang, J., Yang, Z., Ameri, F., Zhang, J.: C-mag: Cascade
multimodal attributed graphs for supply chain link prediction. arXiv preprint
arXiv:2508.08071 (2025)
[12] Soares, M.A.C., Parreiras, F.S.: A literature review on question answering tech-
niques, paradigms and systems. Journal of King Saud University-Computer and
Information Sciences32(6), 635–646 (2020)
[13] Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., Tang, J.: Gpt
understands, too. AI Open5, 208–215 (2024)
[14] Vaswani, A.: Attention is all you need. Advances in Neural Information Processing
Systems (2017)
[15] Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh,
D., Rohrbach, M.: Towards vqa models that can read. In: Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
8317–8326 (2019)
[16] Zhang, Z., Li, X., Gao, Y., Lou, J.-G.: Crt-qa: A dataset of complex reasoning
27

question answering over tabular data. In: Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pp. 2131–2153 (2023)
[17] Soto, C., Yoo, S.: Visual detection with context for document layout analysis.
In: Proceedings of the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pp. 3464–3470 (2019)
[18] Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F.: Layoutlmv3: Pre-training for docu-
ment ai with unified text and image masking. In: Proceedings of the 30th ACM
International Conference on Multimedia, pp. 4083–4091 (2022)
[19] Gao, L., Yi, X., Liao, Y., Jiang, Z., Yan, Z., Tang, Z.: A deep learning-based
formula detection method for pdf documents. In: 2017 14th IAPR International
Conference on Document Analysis and Recognition (ICDAR), vol. 1, pp. 553–558
(2017). IEEE
[20] Zhou, M., Cai, M., Li, G., Li, M.: An end-to-end formula recognition method
integrated attention mechanism. Mathematics11(1), 177 (2022)
[21] Terven, J., C´ ordova-Esparza, D.-M., Romero-Gonz´ alez, J.-A.: A comprehensive
review of yolo architectures in computer vision: From yolov1 to yolov8 and yolo-
nas. Machine Learning and Knowledge Extraction5(4), 1680–1716 (2023)
[22] Ma, Y., Yu, D., Wu, T., Wang, H.: Paddlepaddle: An open-source deep learning
platform from industrial practice. Frontiers of Data and Domputing1(1), 105–115
(2019)
[23] Thorat, C., Bhat, A., Sawant, P., Bartakke, I., Shirsath, S.: A detailed review
on text extraction using optical character recognition. ICT Analysis and Appli-
cations, 719–728 (2022)
[24] Li, Y., Starly, B.: Building a knowledge graph to enrich chatgpt responses in
manufacturing service discovery. Journal of Industrial Information Integration
40, 100612 (2024)
[25] Bhandarkar, M.P., Nagi, R.: Step-based feature extraction from step geometry
for agile manufacturing. Computers in industry41(1), 3–24 (2000)
[26] Xue, Z., Li, L., Tian, S., Chen, X., Li, P., Chen, L., Jiang, T., Zhang, M.:
Domain knowledge is all you need: A field deployment of llm-powered test case
generation in fintech domain. In: Proceedings of the 2024 IEEE/ACM 46th
International Conference on Software Engineering: Companion Proceedings, pp.
314–315 (2024)
[27] Martino, A., Iannelli, M., Truong, C.: Knowledge injection to counter large lan-
guage model (llm) hallucination. In: European Semantic Web Conference, pp.
28

182–185 (2023). Springer
[28] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou,
D.,et al.: Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems35, 24824–24837 (2022)
[29] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., Narasimhan, K.: Tree
of thoughts: Deliberate problem solving with large language models. Advances in
Neural Information Processing Systems36(2024)
[30] Xia, L., Li, C., Zhang, C., Liu, S., Zheng, P.: Leveraging error-assisted fine-tuning
large language models for manufacturing excellence. Robotics and Computer-
Integrated Manufacturing88, 102728 (2024)
[31] Valentine, R.,et al.: Prompt engineering on the engineering design process. In:
DS 131: Proceedings of the International Conference on Engineering and Product
Design Education (E&PDE 2024), pp. 449–454 (2024)
[32] Contributors, L.: LlamaIndex: Interface between LLMs and external data sources.
Accessed: 2024-10-09 (2024). https://www.llamaindex.ai/
[33] Contributors, L.: LangChain: Building applications with large language models.
Accessed: 2024-10-09 (2024). https://www.langchain.com/
[34] Contributors, G.: GraphRAG: Graph-based Retrieval-Augmented Gener-
ation Framework. Accessed: 2024-10-09 (2024). https://www.github.com/
facebookresearch/graphrag
[35] Bei, Y., Fang, Z., Mao, S., Yu, S., Jiang, Y., Tong, Y., Cai, W.: Manufactur-
ing domain qa with integrated term enhanced rag. In: 2024 International Joint
Conference on Neural Networks (IJCNN), pp. 1–8 (2024). IEEE
[36] Buehler, M.J.: Generative retrieval-augmented ontologic graph and multiagent
strategies for interpretive large language model-based materials design. ACS
Engineering Au4(2), 241–277 (2024)
[37] Sun, T., Somalwar, A., Chan, H.: Multimodal retrieval augmented generation
evaluation benchmark. In: 2024 IEEE 99th Vehicular Technology Conference
(VTC2024-Spring), pp. 1–5 (2024). IEEE
[38] Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K.,
Mensch, A., Millican, K., Reynolds, M.,et al.: Flamingo: a visual language model
for few-shot learning. Advances in neural information processing systems35,
23716–23736 (2022)
[39] Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-
training with frozen image encoders and large language models. In: International
29

Conference on Machine Learning, pp. 19730–19742 (2023). PMLR
[40] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J.,et al.: Learning transferable visual models from
natural language supervision. In: International Conference on Machine Learning,
pp. 8748–8763 (2021). PMLR
[41] Xiao, H., Mastrapas, G., Wang, B.: Jina clip: Your clip model is also your text
retriever. In: Multi-modal Foundation Model Meets Embodied AI Workshop@
ICML2024 (2024)
[42] OpenDataLab: PDF-Extract-Kit: A Comprehensive Toolkit for High-Quality
PDF Content Extraction. https://github.com/opendatalab/PDF-Extract-Kit
(2024)
[43] Lewis, D., Agam, G., Argamon, S., Frieder, O., Grossman, D., Heard, J.: Building
a test collection for complex document information processing. In: Proceed-
ings of the 29th Annual International ACM SIGIR Conference on Research and
Development in Information Retrieval, pp. 665–666 (2006)
[44] Hussain, M.: Yolo-v1 to yolo-v8, the rise of yolo and its complementary nature
toward digital manufacturing and industrial defect detection. Machines11(7),
677 (2023)
[45] Zhang, W., Li, F., Chen, Y., Sun, X., Zhou, M.: UniMERNet: A Uni-
fied Detection-and-Recognition Network for Mathematical Expressions. https:
//github.com/opendatalab/UniMERNet. Accessed: 2025-06-11 (2024)
[46] Authors, P.: PaddleOCR, Awesome multilingual OCR toolkits based on Pad-
dlePaddle. https://github.com/PaddlePaddle/PaddleOCR (2020)
[47] Innovator, A.: StructEqTable-Deploy: High-Fidelity Table Recognition Toolkit.
https://github.com/Alpha-Innovator/StructEqTable-Deploy. Accessed: 2025-06-
11 (2023)
[48] Groover, M.P.: Fundamentals of Modern Manufacturing: Materials, Processes,
and Systems. John Wiley & Sons, Hoboken, NJ, USA (2010)
[49] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R.O., Larson, J.: From local to global: A graph
rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024)
[50] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H.,
Lewis, M., Yih, W.-t., Rockt¨ aschel, T.,et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
30

[51] Es, S., James, J., Espinosa Anke, L., Schockaert, S.: RAGAs: Automated
evaluation of retrieval augmented generation. In: Aletras, N., De Clercq, O.
(eds.) Proceedings of the 18th Conference of the European Chapter of the
Association for Computational Linguistics: System Demonstrations, pp. 150–
158. Association for Computational Linguistics, St. Julians, Malta (2024).
https://aclanthology.org/2024.eacl-demo.16/
[52] Li, Y., Ko, H., Ameri, F.: Integrating graph retrieval-augmented generation with
large language models for supplier discovery. Journal of Computing and Infor-
mation Science in Engineering25(2), 021010 (2025) https://doi.org/10.1115/
1.4067389 https://asmedigitalcollection.asme.org/computingengineering/article-
pdf/25/2/021010/7422522/jcise 252021010.pdf
[53] Sharma, K., Kumar, P., Li, Y.: Og-rag: Ontology-grounded retrieval-augmented
generation for large language models. In: Proceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pp. 32950–32969 (2025)
[54] Wu, S., Khasahmadi, A., Katz, M., Jayaraman, P.K., Pu, Y., Willis, K., Liu, B.:
Cad-llm: Large language model for cad generation. In: Proceedings of the Neural
Information Processing Systems Conference. neurIPS (2023)
[55] Huang, B., Wang, X., Chen, H., Song, Z., Zhu, W.: Vtimellm: Empower llm to
grasp video moments. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 14271–14280 (2024)
[56] Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., Xu, B.: Ts2vec:
Towards universal representation of time series. In: Proceedings of the AAAI
Conference on Artificial Intelligence, vol. 36, pp. 8980–8987 (2022)
31
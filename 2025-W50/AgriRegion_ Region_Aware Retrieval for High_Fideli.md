# AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice

**Authors**: Mesafint Fanuel, Mahmoud Nabil Mahmoud, Crystal Cook Marshal, Vishal Lakhotia, Biswanath Dari, Kaushik Roy, Shaohu Zhang

**Published**: 2025-12-10 22:06:41

**PDF URL**: [https://arxiv.org/pdf/2512.10114v1](https://arxiv.org/pdf/2512.10114v1)

## Abstract
Large Language Models (LLMs) have demonstrated significant potential in democratizing access to information. However, in the domain of agriculture, general-purpose models frequently suffer from contextual hallucination, which provides non-factual advice or answers are scientifically sound in one region but disastrous in another due to variations in soil, climate, and local regulations. We introduce AgriRegion, a Retrieval-Augmented Generation (RAG) framework designed specifically for high-fidelity, region-aware agricultural advisory. Unlike standard RAG approaches that rely solely on semantic similarity, AgriRegion incorporates a geospatial metadata injection layer and a region-prioritized re-ranking mechanism. By restricting the knowledge base to verified local agricultural extension services and enforcing geo-spatial constraints during retrieval, AgriRegion ensures that the advice regarding planting schedules, pest control, and fertilization is locally accurate. We create a novel benchmark dataset, AgriRegion-Eval, which comprises 160 domain-specific questions across 12 agricultural subfields. Experiments demonstrate that AgriRegion reduces hallucinations by 10-20% compared to state-of-the-art LLMs systems and significantly improves trust scores according to a comprehensive evaluation.

## Full Text


<!-- PDF content starts -->

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice
MESAFINT FANUEL,North Carolina A&T State University, USA
MAHMOUD NABIL MAHMOUD,The University of Alabama, USA
CRYSTAL COOK MARSHALL,North Carolina Agricultural and Technical State University, USA
VISHAL LAKHOTIA,Amazon AWS, USA
BISWANATH DARI,North Carolina Agricultural and Technical State University, USA
KAUSHIK ROY,North Carolina Agricultural and Technical State University, USA
SHAOHU ZHANG,North Carolina Agricultural and Technical State University, USA
Large Language Models (LLMs) have demonstrated significant potential in democratizing access to information. However, in the
domain of agriculture, general-purpose models frequently suffer from "contextual hallucination", which provides non-factual advice
or answers are scientifically sound in one region but disastrous in another due to variations in soil, climate, and local regulations.
We introduceAgriRegion, a Retrieval-Augmented Generation (RAG) framework designed specifically for high-fidelity, region-aware
agricultural advisory. Unlike standard RAG approaches that rely solely on semantic similarity,AgriRegionincorporates a geospatial
metadata injection layer and a region-prioritized re-ranking mechanism. By restricting the knowledge base to verified local agricultural
extension services and enforcing geo-spatial constraints during retrieval,AgriRegionensures that the advice regarding planting
schedules, pest control, and fertilization is locally accurate. We create a novel benchmark dataset,AgriRegion-Eval, which comprises
160 domain-specific questions across 12 agricultural subfields. Experiments demonstrate thatAgriRegionreduces hallucinations by
10-20% compared to state-of-the-art LLMs systems and significantly improves trust scores according to a comprehensive evaluation.
CCS Concepts:•Computing methodologies→Machine learning;•Applied computing→Agriculture.
Additional Key Words and Phrases: intelligent systems, AI, retrieval-augmented generation, agriculture
ACM Reference Format:
Mesafint Fanuel, Mahmoud Nabil Mahmoud, Crystal Cook Marshall, Vishal Lakhotia, Biswanath Dari, Kaushik Roy, and Shaohu
Zhang. 2025. AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice. 1, 1 (December 2025), 15 pages. https:
//doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Large Language Models (LLMs) such as ChatGPT [ 26], Deepseek [ 10], and Gemini [ 14] have shown promising capabilities
in image understanding and interpreting, text summarization, question answering (QA), and dialog systems [ 8,33,35].
Authors’ Contact Information: Mesafint Fanuel, North Carolina A&T State University, Greensboro, NC, USA, mfanuel@ncat.edu; Mahmoud Nabil
Mahmoud, The University of Alabama, Tuscaloosa, AL, USA, mmahmoud1@ua.edu; Crystal Cook Marshall, North Carolina Agricultural and Technical
State University, Greensboro, NC, USA, cacookmarshall@ncat.edu; Vishal Lakhotia, Amazon AWS, USA, lakhov@amazon.com; Biswanath Dari, North
Carolina Agricultural and Technical State University, Greensboro, NC, USA, bdari@ncat.edu; Kaushik Roy, North Carolina Agricultural and Technical
State University, Greensboro, NC, USA, kroy@ncat.edu; Shaohu Zhang, North Carolina Agricultural and Technical State University, Greensboro, NC,
USA, szhang1@ncat.edu.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1arXiv:2512.10114v1  [cs.AI]  10 Dec 2025

2 Mesafint Fanuel et al.
Despite their remarkable success, LLMs face challenges in domain-specific or knowledge-intensive tasks [ 20]. They
often struggle to provide accurate and relevant responses to niche or complex queries, particularly when they are
faced with questions requiring specialized knowledge, or when asked to generate content that requires up-to-date
information in region.
A promising solution to these challenges is Retrieval-Augmented Generation (RAG), which involves integrating
parametric and non-parametric memory components. This method combines the capabilities of LLMs with an external
information retrieval system, allowing the model to dynamically search and incorporate information from extensive
databases or document collections [ 15,20]. By leveraging external knowledge beyond the model’s pre-trained dataset,
this approach improves the model’s ability to produce accurate and contextually appropriate for domain-specific
responses. RAG enhances both the factual accuracy and relevance of model outputs, while reducing the risk of model
hallucination, where LLMs might produce seemingly plausible but incorrect or fabricated information. Additionally,
this method is highly effective for knowledge-intensive tasks, with document corpora functioning as a domain-specific
knowledge reservoir.
Agricultural question answer systems powered by LLMs can help farmers, researchers, and practitioners by providing
answers to a wide variety of questions from crop management and pest control to sustainable farming practices. For
instance, a RAG-based system can provide accurate, context-aware answers to farmer queries about crop diseases,
irrigation schedules, or soil management practices. The result is improved precision, explain-ability, and trust in AI-
driven advisory systems for agriculture [ 20]. By integrating specialized agricultural datasets into the retrieval process,
RAG models can offer contextually aware guidance that is tailored to specific agricultural challenges in real-time. The
result is a system that not only answers questions, but also supports decision-making, improves productivity, and
ultimately contributes to more sustainable agricultural practices.
However, several key challenges hinder the effective application of LLMs in the agriculture domain. Firstly, agricultural
science relies heavily on specialized terminology and complex concepts such as soil nutrient cycles and integrated
pest management. The parametric memory of generic LLMs would have a poor understanding of such a complex
knowledge base. Fetching the information from a text corpora might not on its own provide the LLM with enough
understanding. Consequently, LLM might not utilize factual agricultural concepts. In addition, accurately understanding
context-specific information, such as regional farming methods or environmental conditions, is challenging. This
is particularly difficult due to the high variability in agricultural practices in different regions and climates. Hence,
regional adaptation of an agricultural LLM is needed to bridge this gap. Another major obstacle is the inconsistency and
limited availability of high-quality agriculturally annotated data. Information in this field comes from diverse sources,
including textbooks, scientific papers, field reports, and sensor data, which vary greatly in format, depth, and reliability.
Additionally, LLM errors can have serious consequences in agriculture applications, from poor crop management to
resource waste. This creates skepticism among farmers when AI advice seems generic or impractical.
Current approaches [ 39,40] to answering agricultural questions, while valuable, often do not address these challenges.
Existing systems either rely heavily on generic LLMs without sufficient domain adaptation, or employ simple retrieval
mechanisms that lack the sophistication to handle complex agricultural queries. Moreover, few systems explicitly
account for regional variations in agricultural practices, which can lead to generic or contextually inappropriate
recommendations. To address these challenges, we proposeAgriRegion, a system that leveragesRegion-Aware Retrieval,
which builds a dynamic index of verifiedAgricultural extension documents, tagged with geospatial metadata. When a
query is received, the system does not just look for semantic relevance; it filters and re-ranks evidence based on the
user’s geolocation, ensuring actionable high-fidelity advice. Our work makes the following key contributions:
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 3
•We propose a novel region-aware RAG framework designed specifically for agriculture that moves beyond
standard semantic search. By integrating Region-Aware Retrieval, our system addresses the limitations of generic
LLMs and simple retrieval mechanisms that fail to handle the complexity of domain-specific agricultural queries.
•We introduce a mechanism for constructing a dynamic index of verified agricultural extension documents
enriched with geospatial metadata. This allows the retrieval pipeline to filter and re-rank evidence based on the
user’s geolocation, ensuring that the retrieved context is spatially aligned with the user’s environment.
•We mitigate the issue of generic or contextually inappropriate recommendations by explicitly enforcing regional
constraints during the retrieval process. This ensures that the generated advice is not only semantically relevant,
but also practically actionable and accurate for the specific local practices and conditions of the farmers.
The remainder of this paper proceeds as follows. Section 2 provides background and describes related work. In Section
3, we present the detailed design ofAgriRegion. Section 4 presents the comprehensive evaluation of our proposed
framework. Finally, we conclude in Section 5.
2 Related Work
The advancement of LLMs has spurred considerable research into their applications, particularly in the areas of question
answering, text generation, and domain-specific applications. This section explores related work with a focus on LLMs
and RAG applications in the agricultural domain.
2.1 Foundation Models and LLMs in Agriculture
Foundation models are large-scale models trained on vast, broad data that can be adapted to a wide array of downstream
tasks. In agriculture, this typically involves taking a general-purpose LLM (e.g., GPT or LLaMA) and fine-tuning it on
domain-specific corpora, such as agronomic textbooks, research papers, and extension manuals. This specialization
is crucial for tasks requiring deep expert knowledge, such as pest management, where general models may lack the
necessary specificity [ 41,44]. Previous studies, including [ 37], have highlighted the shortcomings of GPT-style models
in addressing agricultural extension questions, emphasizing the need for human-in-the-loop refinement. AGXQA [ 19]
advances this direction by employing fine-tuned models for agricultural question answering, supported by human-
preference assessments. AgriBench [ 45] and AgMMU [ 13] broaden evaluation to multimodal reasoning over visual and
textual content. AgEval [ 3] focuses on plant stress phenotyping, offering 12 tasks for zero- and few-shot evaluation. To
provide a comprehensive agriculture question dataset, we create a novel benchmark dataset,AgriRegion-Eval, which
comprises 160 domain-specific questions across 12 agricultural subfields including Agronomy, Soil, Pathology, Weeds,
Irrigation, Horticulture, Postharvest, Animal, Aquaculture, Food Safety, Economics, and Extension.
2.2 Retrieval-Augmented Generation
A primary limitation of pre-trained LLMs is that their knowledge is "frozen" at the time of training, leading to out-of-date
information and a propensity for "hallucination." RAG addresses this by augmenting the model’s generation process
with real-time information retrieval. Before generating a response, the system retrieves relevant documents from an
external, up-to-date knowledge base (e.g., a vector database of pest advisories) and provides this context to the LLM.
This methodology is particularly critical in agriculture, where advice must be accurate, timely, and often localized. [ 38]
provide a dedicated survey on RAG for agricultural decision-making, outlining its opportunities and unsolved problems.
Manuscript submitted to ACM

4 Mesafint Fanuel et al.
Other works have provided direct case studies, comparing RAG pipelines against fine-tuning and demonstrating RAG’s
utility in AI-powered optimization chatbots [4, 5].
AgriGPT [ 40] is an open-source language model ecosystem specifically designed for agricultural applications. It was
developed to address the lack of domain-specific content, adequate evaluation frameworks, and reliable reasoning capa-
bilities in general-purpose AI models for the agriculture sector. The AgriGPT ecosystem includes a modular framework
with several components including an Agri-342K Dataset, Tri-RAG Framework, and AgriBench-13K Benchmark Suit. Its
continuing work AgriGPT-VL [ 39] based on Qwen2.5-VL is a multimodal extension focusing on vision-language tasks
like identifying crop diseases from images.AgriRegionintegrates region-aware retrieval to addresses the limitations
of generic LLMs and simple retrieval mechanisms that fail to handle the complexity of domain-specific agricultural
queries.
3 System Design
AgriRegionis designed to create a region-aware agriculture AI agent that can answer specific real-world agricutural
questions.AgriRegionleverages structured and unstructured agricultural knowledge through semantic retrieval and
transformer-based generation [ 15,20]. As shown in the Figure 1, the system’s foundation is itsSeed Knowledge,
which consists of diverse data sources such as publications, reports, geo-labeled data, and datasets. This curated data is
then processed and stored in aVector Databaseusing RAG. This entire process is referred to as knowledge grounding,
which ensures that the AI’s answers are based on factual data. This Agent has two key capabilities withAgriculture
Domainknowledge (e.g., plant pathology, soil, irrigation, and agronomy) and a set ofSkills(e.g., context Reasoning,
multi-modality, and summarization). When a user asks a specific regional question (e.g., about fungicides for peanut
leaf spot in North Carolina), the LLM pulls information from the vector database and applies its domain knowledge and
skills to create a relevant and accurate response. Figure 2 provides the details of the pipeline, which include document
fragmentation, embedding generation, vector indexing, and retrieval-augmented generation at inference time.
3.1 Knowledge Retrieval
The retrieval module is the integration of Ada embeddings, chunk-level indexing, and Chroma’s efficient retrieval
architecture [ 15]. This module ensures that the generation model operates on the most semantically aligned and
contextually grounded content available, reducing hallucinations and increasing factual relevance.
3.1.1 Embedding and Vector Indexing.Following document segmentation, each chunk is converted into a high-
dimensional vector representation using OpenAI’s Ada v2 embedding model [ 15]. This model encodes natural language
into 1,536-dimensional dense vectors, enabling semantic similarity search that goes beyond surface-level lexical match-
ing. Ada was selected for its robust zero-shot performance and strong generalization in scientific and technical domains,
including agriculture, without requiring additional fine-tuning. The model is particularly effective in capturing nuanced
paraphrasing and terminology variants-an essential feature when handling heterogeneous sources such as textbook,
academic papers, and extension manuals.
Each chunk is embedded with the prompt prefix “text: ” to align with Ada’s training format. Embedding is performed
in batches using OpenAI’s API, and each vector is associated with its chunk metadata, including document ID, source
type, section heading, publication year, and any domain-specific tags (e.g., “drip irrigation,” “nematodes,” “clay soils”).
To enable fast, filterable semantic search, all embeddings are stored in Chroma DB, a lightweight, developer-friendly
vector database optimized for real-time retrieval applications. Chroma supports approximate nearest neighbor (ANN)
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 5
Seed 
Knowledge
Dataset
Geo-labeled Data
Report
……Knowledge ElicitationSkills Agriculture Domain
RAGGenerated 
KnowledgeInstructionsContext Reasoning Region Aware
Multi -ModalityAI Agent
Soil Irrigation Crop
Agronomy Plant Pathology……
Technology
PublicationData Curation Knowledge GroundingSummarization
UsersWhen should 
fungicides start 
for peanut leaf 
spot in humid 
NC summer?
Vector 
Database
Fig. 1. Overview ofAgriRegion.
Data Sources
(PDFs, Pages, Report)
Data Ingestion
Chunk & Normalize
Retriever
Ranker
FeedbackVector Index
Embedding
Prompt
Fig. 2. The System Pipeline of RAG.
search using HNSW (Hierarchical Navigable Small World) indexing, which allows scalable and low-latency retrieval
over millions of embeddings. Its support for metadata filtering enables powerful query constraints, such as retrieving
only “textbook excerpts on nitrogen application published after 2015” or “journal articles discussing salinity in maize.”
Manuscript submitted to ACM

6 Mesafint Fanuel et al.
Chroma’s modular design supports both in-memory prototyping and persistent storage, making it ideal for local
testing and cloud deployment alike. We maintain separate collections for different document types (e.g., textbooks,
journals, manuals), and define filters at query time based on the user’s intent or the model’s inferred need. Chroma’s
built-in support for cosine similarity scoring aligns well with Ada’s vector space, ensuring accurate top-k selection.
The system also includes a background daemon that monitors corpus updates and triggers automatic re-embedding
and re-indexing of new or modified documents. This ensures that retrieval performance remains stable over time,
even as the corpus evolves with new literature, policies, and research outcomes [ 18]. The combined use of Ada and
Chroma allows for rapid and semantically accurate retrieval, supporting real-time RAG generation grounded in relevant
agricultural knowledge.
3.1.2 Spatial-Semantic Retrieval.When a user submits a query, it is embedded using the same Ada v2 model. The
resulting query vector is used to retrieve the top-k most semantically similar chunks, where 𝑘is typically set to 5 or 10,
depending on the verbosity of the expected response. Retrieval is based on weighted cosine similarity 𝑆𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐 and a
distance decay function𝑆 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 [42].
𝑆𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 =1
1+𝑑(𝑔 𝑢𝑠𝑒𝑟,𝑔𝑑𝑜𝑐)(1)
𝑆𝑓 𝑖𝑛𝑎𝑙=(1−𝛼)𝑆 𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐+𝛼𝑆 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 (2)
Where𝑑(𝑔 𝑢𝑠𝑒𝑟,𝑔𝑑𝑜𝑐)is the normalized geodesic distance. If the user is within the document’s target region (e.g., a user
in Greensboro, NC, retrieving an NC Extension document), 𝑑≈0and𝑆𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒→1. As the user moves away (e.g., a
user in Virginia retrieving an NC doc), 𝑑increases, and 𝑆𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 decays.𝛼is a hyperparameter that controls the weight
of the locality (i.e., 𝛼= 0.5). Top-k results are re-ranked based on 𝑆𝑓 𝑖𝑛𝑎𝑙. For example, a query like “What is the impact
of salinity on maize yield under drip irrigation?” may retrieve five chunks: two from recent journal studies, one from
Principles of Agronomy for Sustainable Agriculture[ 29], and two from extension bulletins. These chunks are selected
to be both semantically and geographically relevant. The spatical-semantic score ensures that "local" knowledge is
maximally scored, while knowledge from ecologically similar but spatially distant regions is penalized but not strictly
eliminated.
The retrieved top-k chunks are then passed directly to the generation module in ranked order, maintaining all
metadata. This enables downstream logic to inform the language model of the source, document type, region, and even
temporal relevance (e.g., preferring recent studies when available).
3.2 Knowledge Generation
The text generation component of our RAG architecture is responsible for synthesizing a final answer based on the
retrieved evidence and user query. We employ a fine-tuned version of the LLaMA 3 13B model, a state-of-the-art
decoder-only transformer trained by Meta AI for high-quality language understanding and instruction-following [ 36].
LLaMA 3 has demonstrated superior performance over previous models such as GPT-3 and FLAN-T5 in benchmarks
across question answering, summarization, and reasoning tasks.
To align the model with agricultural applications, we fine-tuned LLaMA 3 on a curated dataset comprising instruction-
style examples drawn from agricultural QA corpora, university extension manuals, and textbook summaries. This
instruction-tuning step teaches the model to generate accurate, domain-specific answers while following the desired
format and tone expected by farmers, agricultural advisors, and researchers. Each prompt sent to the model includes:
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 7
•A role instruction defining the assistant’s function (e.g., “You are an agricultural assistant helping with evidence-
based answers.”),
•The top-𝑘retrieved chunks, ranked by spatial-semantic relevance and formatted with headings and source
metadata,
•The user question as the final input in the prompt structure.
The model processes this composite prompt and generates a single, fluent, and informative response grounded in the
retrieved documents. Unlike traditional generative models that hallucinate unsupported content, our RAG setup reduces
hallucination risk by grounding the model in verifiable context [ 15,20]. The fine-tuning dataset was created with
high-quality human annotations and supplemented with synthetic examples derived from agricultural textbooks such as
The Nature and Properties of Soils[ 7] andPrinciples of Agronomy for Sustainable Agriculture[ 29]. To ensure reliability, the
fine-tuning process was monitored using evaluation metrics such as BLEU and answer factuality against human-labeled
validation sets. Thus, the combination of RAG-based grounding and domain-specific instruction tuning enables the
model to provide precise, contextually appropriate, and explainable answers across a wide range of agricultural domains,
including soil fertility, pest control, irrigation, crop disease, and sustainability practices.
4 Evaluation
4.1 Agricultural Corpus Construction
We focus on North Carolina (NC) as the study area. We collected a data set of more than 70,000 documents from trusted
sources, including the Scopus bibliographic database, textbooks, and agricultural report in NC. To ensure a focused
and relevant corpus aligned with the needs of agricultural research and practice, we filtered the data using the Scopus
subject area classification system. Specifically, we included only documents categorized under the Agricultural and
Biological Sciences (AGRI) top-level subject codes. Within the AGRI subject area, Scopus further categorizes the content
into a set of discipline-specific subfields, each representing a major domain within agricultural science, as listed in Table
1. To provide structured peer-reviewed foundational knowledge in soil science, agronomy, and plant protection, the
database integrates four textbooks adopted worldwide, including The Nature and Properties of Soils [ 7],Soil Fertility
and Fertilizers[16],Principles of Agronomy for Sustainable Agriculture[29], andPlant Pathology[1].
In addition, the most critical component of theAgriRegioncorpus is the integration literature from local area.
For example, document from the extensive repository of the North Carolina Cooperative Extension [ 24] Extension
documents represent the "last mile" of agricultural knowledge—the translation of complex research into actionable
advice for farmers. Examples of ingested documents include:
•Carolina Lawns: A Guide to Maintaining Quality Turf in the Landscape [ 23]: Specific to the transition zone
climates of NC.
•Integrated Pest Management Publications and Factsheets [ 12]: It covers a wide variety of topics: insects, weeds,
diseases; crops from cotton to strawberries; turf; ornamental plants; pesticide information; equipment; or-
ganic/ecological production; and even public health pests.
•Vegetable Gardening: A Beginner’s Guide [ 6]: It provides the "spatiotemporal" specificity that generic LLMs lack
and contains exact dates, chemical trade names, and regulatory warnings applicable to the user’s location.
Manuscript submitted to ACM

8 Mesafint Fanuel et al.
Table 1. Subdomains within AGRI Subject Area Selected from Scopus
AGRI Subdomain Description
Crop Science and Agronomy Studies on crop physiology, genotype-environment interactions,
cropping systems, and yield response.
Soil Science Research on soil fertility, structure, carbon content, erosion,
salinity, and microbial processes.
Plant Pathology & Entomology Investigations into plant diseases, pest dynamics, pesticide use,
and biological control methods.
Animal Science Topics related to livestock nutrition, breeding, health, and
integrated crop-livestock systems.
Irrigation & Water Management Efficient water use, irrigation technologies, water stress mitigation,
and climate response.
Postharvest & Food Systems Storage, preservation, value addition, and food safety in the supply chain.
Horticulture Cultivation of fruits, vegetables, ornamentals, and greenhouse systems.
Agricultural Biotechnology Genetic engineering, transgenic crops, molecular breeding.
Agroecology & Sustainability Ecological farming, sustainable land use, biodiversity conservation.
4.2 Document Chunking and Segmentation
To prepare documents for embedding and semantic retrieval, we implemented a dedicated fragmentation and segmenta-
tion pipeline that transforms unstructured long-form texts into coherent retrievable units optimized for vector-based
search [ 20]. While traditional documents such as scientific papers and manuals are typically structured hierarchi-
cally (e.g., Introduction, Methods, Results), direct retrieval on full-length articles is computationally inefficient and
semantically diffuse. To address this, all content in our corpus—journal articles, extension bulletins, and textbooks—is
segmented into overlapping chunks of 300 tokens with a stride of 50 tokens between chunks.
Our system is designed to retain contextually meaningful units by detecting paragraph breaks, section headings, and
typographic cues such as bullet points or enumerated lists. Each chunk is tagged with metadata that include the original
section heading, the source document (journal, textbook, or manual), the publication year, and a persistent document
ID for citation alignment. In addition, headings are added as soft prompts to each section (e.g., “Heading: Soil Nitrogen
Dynamics”) to give downstream LLMs a more contextual foundation. For example, a research paper published inField
Crops Researchmight analyze drought stress in maize using methodology sourced from canonical textbooks, such as
Principles of Agronomy for Sustainable Agriculture[ 29] orSoil Fertility and Fertilizers[ 16]. In such cases, our chunking
system ensures that methodologically related excerpts from both textbooks and document are indexed in the same
semantic space. In total, The corpus has over 4 million overlapping textual chunks, each representing a semantically
atomic unit of information suitable for similarity search, grounding, and large language model generation.
Each model answers all 160 questions independently. To evaluate BLEU [ 27] and ROUGE [ 21], we adopt the approach
of normalizing the prompts [ 22], which lowercases the prompts, strips punctuation, and removes stopwords. For
semantic metrics, we use BERTScore (F1) [ 43] with domain-sensitive settings. We report RAGA-Precision to qualify
retrieval grounding. To mitigate sampling noise, we performed three rounds of experiments with different random
seeds and reported the mean [31].
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 9
4.3 AgriRegion-Eval Question Dataset
We evaluate the proposedAgriRegionagainst non-retrieval LLM baselines:GPT-4-Turbo[ 25],Claude 3.5 Sonnet[ 2],
Gemini 1.5 Pro[ 14], andMistral-7B-Instruct[ 17]. We create a benchmark dataset,AgriRegion-Eval, which comprises
160 domain-specific questions across 12 subfields (e.g., Agronomy, Soil, Pathology, Weeds, Irrigation, Horticulture,
Postharvest, Animal, Aquaculture, Food Safety, Economics, Extension). All systems produce short factual answers with
a fixed temperature at𝑇=0.2.
4.4 Evaluation Metrics
We report Exact Match (EM), F1, BLEU-4, ROUGE-L, BERTScore (F1), and RAGA-Precision(Retrieval-Augmented
Generation Assessment Suite) [ 11] to quantify the factual grounding and retrieval effectiveness. EM and F1 reflect
surface-form fidelity to reference answers [ 28], while BLEU-4 and ROUGE-L capture 𝑛-gram overlap and longest common
subsequence similarity, respectively. BERTScore measures semantic alignment using contextual embeddings [ 43]. RAGA
framework reports four sub-metrics [11]:
•Context Precision (P c)– the proportion of retrieved passages that are relevant to the question, reflecting
retrieval accuracy.
𝑃𝑐=|𝐶rel∩𝐶 retrieved|
|𝐶retrieved|
•Context Recall (R c)– the proportion of reference facts that appear in the retrieved context, reflecting retrieval
coverage.
𝑅𝑐=|𝐶rel∩𝐶 retrieved|
|𝐶rel|
•Faithfulness (F)– the fraction of generated statements that are directly supported by the retrieved evidence,
capturing grounding and the absence of hallucination [34].
𝐹=supported claims
total claims in answer
•Answer Relevance (R a)– the semantic similarity between the question and the generated answer, measuring
topical alignment [30].
All sub-metrics are computed on a [0, 1] scale using sentence-level embeddings and cosine similarity (we use
all-mpnet-base-v2[ 32] for our experiments). The overall RAGAS score is reported as the unweighted mean of the four
sub-scores:
RAGAS=1
4(𝑃𝑐+𝑅 𝑐+𝐹+𝑅 𝑎)
4.5 Overall Performance
To retrieve answers, we apply the structured prompt shown in Figure 3. Across 160 agricultural questions,AgriRe-
gionconsistently outperforms all non-RAG baselines on both lexical and semantic metrics. Table 2 summarizes the
results. Compared with the next-best model GPT-4-Turbo [ 25] and other LLMs,AgriRegionachieves notably 10-20%
higher EM, F1, BLEU-4, ROUGE-L, and BERTScore, indicating stronger semantic alignment. Its high RAGA-Precision
further shows that retrieved domain evidence is effectively incorporated into generated answers. Paired bootstrap
tests confirm that gains in EM, F1, and BERTScore are statistically significant ( 𝑝< 0.01). We also compute Cliff’s 𝛿[9],
obtaining medium-to-large effect sizes on EM ( 𝛿= 0.41) and F1 ( 𝛿= 0.47). Overall, these results demonstrate that
region-specific retrieval substantially enhances answer precision while reducing hallucinations.
Manuscript submitted to ACM

10 Mesafint Fanuel et al.
Prompt for Retrieval
System Instruction:
You are an agricultural expert specializing in North Carolina production systems. Base your answer strictly
on the retrieved passages and assume the user is farming in North Carolina. Adjust any ranges, timings, or
recommendations to North Carolina conditions if the evidence supports it.
Request:{question}
Retrieved Passages:{advice}
Fig. 3. Prompt template used for answer retrieval.
Table 2. Automatic evaluation across 160 agricultural questions.Higher is better.RAGA-Precision is only applicable to RAG systems.
Model EM F1 BLEU-4 ROUGE-L BERTScore RAGA-Precision
AgriRegion(ours)0.76 0.82 0.65 0.72 0.90 0.86
GPT-4-Turbo 0.64 0.70 0.55 0.61 0.82 —
Claude 3.5 Sonnet 0.60 0.66 0.51 0.57 0.80 —
Gemini 1.5 Pro 0.56 0.61 0.46 0.52 0.77 —
Mistral-7B-Instruct 0.49 0.55 0.39 0.45 0.71 —
Figure 4 contrasts models across six metrics. The expanded area forAgriRegionindicates balanced improvements
in both lexical and semantic dimensions. The qualitative examples in Table 3 further show how retrieval helps:
AgriRegionconsistently provides North Carolina–specific details, such as local timing windows, soil characteristics,
realistic nutrient rates, and storage conditions. In contrast, GPT-4-Turbo often gives correct but generic responses that
lack local agronomic nuance. By grounding answers in retrieved extension publications,AgriRegionreduces omissions,
avoids overgeneralization, and offers more actionable, region-tailored recommendations—resulting in higher accuracy
and lower hallucination rates.
4.6 Ablation Study
To isolate the retrieval effects, we vary the number of retrieved passages (top- 𝑘) and disable retrieval. As shown in Table
4, the ablation study demonstrates that retrieval is a key contributor toAgriRegion’s performance. Removing the retrieval
entirely (“No Retrieval”) results in a large drop in F1 (–0.15) and similar declines in EM and BERTScore. This confirms
that the model relies heavily on retrieved evidence to produce accurate and grounded answers. Varying the number of
retrieved passages shows that using too few documents (Top- 𝑘=2) slightly reduces answer completeness, especially for
questions requiring multiple facts. Increasing the retrieval depth (Top- 𝑘=8) yields modest improvements but does not
exceed the standard configuration (Top- 𝑘=5), indicating diminishing returns beyond the default setting. The “Random
Docs” condition produces the worst performance, highlighting that not just retrieval, but relevant retrieval, is crucial
for effective grounding.
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 11
EM
F1
BLEU-4
ROUGE-LBERTScoreRAGA-Precision
0.20.40.60.81.0AgriRegion
GPT-4-Turbo
Claude 3.5 Sonnet
Gemini 1.5 Pro
Mistral-7B-Instruct
Fig. 4.AgriRegionwith the largest closed area reflects strong performance across all metrics.
4.7 Domain-wise Effects.
We further analyze retrieval benefits across the 12 agricultural subdomains. The largest gains appear inSoil( +0.19F1),
Pathology(+0.17F1), andIrrigation( +0.21F1) when compared to the strongest non-RAG baseline (GPT-4-Turbo).
These domains frequently require threshold-based, rate-specific, or conditional agronomic recommendations, which
benefit substantially from retrieval grounding. More narrative domains such asEconomicsandExtensionshow
smaller but consistent improvements (+0.07F1).
We compute cosine similarity between the generated answer and the centroid of retrieved passages per subdomain
using SBERT embeddings. The confusion matrix for theAgriRegionmodel shows a considerably higher diagonal
intensity. By contrast, GPT-4-Turbo exhibits lighter diagonals. This suggests a mild semantic drift from the evidentiary
context.
4.8 Robustness Checks
We repeat the evaluation with rephrased prompts (retranslation via en →de→en) and with shuffled option orders for
questions that admit list-like answers. Performance deltas remain within ±0.02for EM and F1, indicating robustness
to minor prompt variations. We also test with and without citation-style prompts; grounding metrics improve the
BERTScore0.03when explicit citation prompts are used.
5 Conclusion and Future Work
This paper introducesAgriRegion, a retrieval-augmented generation framework designed specifically for agricultural
question answering and domain knowledge synthesis. By coupling dense retrieval over an expert-curated agricultural
corpus with large language model generation, the system bridges the gap between linguistic fluency and factual
reliability in agriculture-focused reasoning tasks.
Manuscript submitted to ACM

12 Mesafint Fanuel et al.
Table 3. Examples of Reference Answers and Model Outputs
Question ReferenceAgriRegionGPT-4-Turbo
What is the opti-
mal soil pH range
for field corn in
the Southeast?Target pH 6.0–
6.5; apply lime if
buffer pH indi-
cates need.The optimal soil pH range for field
corn in the Southeast, including
North Carolina, is 6.0 to 6.5. Main-
taining pH in this range maxi-
mizes nutrient availability and crop
yield—apply lime if your soil test
and buffer pH indicate a need to
raise pH.The optimal soil pH range for field
corn in the Southeast is 6.0 to 6.5. If
the soil pH falls below this range, ap-
plying lime is recommended based
on buffer pH tests.
How should
irrigation be
shut down
pre-harvest for
peanuts?Reduce irrigation
as pegs mature;
avoid excessive
late-season water
to limit disease.In North Carolina and the South-
east, begin reducing irrigation 2–3
weeks before digging to encourage
pod drying and minimize disease
risk. Avoid excess late water that
delays harvest or increases pod rot.Irrigation should be gradually re-
duced 2–3 weeks before harvest to
prevent excessive moisture and re-
duce disease risk such as pod rot.
Best cultural
practice to re-
duce tomato
early blight in
humid climates?Use crop rotation,
stake/mulch, drip
irrigation, and re-
sistant cultivars.Rotate crops for 2+ years, stake
plants, mulch to reduce soil splash,
use drip irrigation, and select resis-
tant cultivars adapted to the South-
east.Use crop rotation, resistant culti-
vars, staking and mulching to im-
prove airflow, and drip irrigation to
reduce foliage wetness.
Threshold for
soybean defolia-
tors at R stages?Typically∼15–
20% defoliation
at R1–R6 justifies
treatment.In the Southeast, economic thresh-
old is 15–20% defoliation during R1–
R6; scout frequently as pest pres-
sure varies with rainfall and soil
conditions.Treatment is justified when defolia-
tion reaches 15–20% during R1–R6;
regular scouting is essential to time
interventions.
What causes
chilling injury in
cucumbers?Storage below
∼50°F causes
pitting and
water-soaked
spots.Chilling injury occurs when cucum-
bers are stored below ∼50°F, causing
pitting and water-soaked lesions;
maintain 50–55°F in NC storage con-
ditions.Chilling injury results from stor-
ing cucumbers below ∼50°F (10°C),
leading to pitting and water-soaked
spots; maintain 50–55°F to prevent
injury.
Best temp for
leafy greens
storage?Near 32–36°F at
high RH to main-
tain turgor.Store leafy greens at 32–36°F with
95%+ RH; rapid cooling is particu-
larly important in NC’s warm, hu-
mid climate.The best storage temperature for
leafy greens is 32–36°F with high
humidity to preserve freshness and
minimize water loss.
The extensive evaluation across 160 questions demonstrated substantial and statistically significant improvements in
both lexical and semantic metrics compared to non-retrieval baselines.AgriRegionachieved gains in the F10 .12and
BERTScore0 .08over GPT-4-Turbo, while also receiving the highest human ratings for truthfulness and completeness.
Ablation experiments confirmed that these gains arise directly from evidence retrieval rather than model size or
decoding hyperparameters. Visual analyses further revealed stronger semantic alignment between generated answers
and retrieved documents, supporting the interpretability and traceability of the model’s reasoning.
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 13
Table 4. Ablation study
VariantTop-𝑘EM F1 BERTS RAGA-P
AgriRegion50.76 0.82 0.90 0.86
No RAG — 0.62 0.67 0.80 —
Top-𝑘=22 0.72 0.77 0.88 0.82
Top-𝑘=88 0.74 0.79 0.89 0.84
Random Docs 5 0.56 0.62 0.76 0.50
Table 5. Domain-wise F1 gains of Ag-RAG over GPT-4-Turbo.
Subdomain GPT-4-TurboAgriRegionΔGain
Soil Science 0.63 0.82+0.19
Plant Pathology 0.61 0.78+0.17
Irrigation 0.58 0.79+0.21
Weeds 0.66 0.77 +0.11
Horticulture 0.67 0.78 +0.11
Agronomy 0.70 0.81 +0.11
Extension 0.72 0.79 +0.07
Economics 0.73 0.80 +0.07
Fig. 5.AgriRegioninter-domain similarity.
 Fig. 6. GPT-4 inter-domain similarity.
Fig. 7. Cosine similarity heatmaps across 12 agricultural subfields. Higher diagonal values indicate stronger domain consistency.
From a broader perspective, the results validate retrieval augmentation as a practical and computationally efficient
pathway to domain adaptation—particularly in fields like agriculture, where structured expertise is spread across
heterogeneous sources. By grounding responses in verifiable domain documents,AgriRegionenhances factual accuracy
and user trust, paving the way for transparent and explainable AI systems for agriculture.
Several research extensions remain open:
Manuscript submitted to ACM

14 Mesafint Fanuel et al.
•Retrieval Optimization:Explore cross-encoder reranking and hybrid dense–sparse retrieval to improve context
precision and reduce redundancy.
•Adaptive Context Length:Dynamically adjust top- 𝑘retrieval and context window based on query complexity
and document entropy.
•Multimodal Integration:Extend the RAG pipeline to incorporate images (crop disease diagnostics), geospatial
features, and tabular datasets for end-to-end decision support.
•User Interaction Layer:Develop an interactive agent prototype—the AI Agronomist—capable of dialogic
reasoning, source citation, and on-device retrieval for field deployment.
In conclusion, this work establishes a technical basis for agricultural AI systems grounded in evidence. The framework
AgriRegionshows that spatial-semantic retrieval can achieve strong factual accuracy, clear domain interpretability, and
scalable flexibility, pointing to a promising path for the next generation of agricultural intelligence systems.
References
[1] George N Agrios. 2005. Plant Pathology.Elsevier Academic Press(2005).
[2] Anthropic. 2024.Claude 3.5 Sonnet Model Card. https://www.anthropic.com/news/claude-3-5-sonnet
[3]Muhammad Arbab Arshad, Talukder Zaki Jubery, Tirtho Roy, Rim Nassiri, Asheesh K Singh, Arti Singh, Chinmay Hegde, Baskar Ganapathysubra-
manian, Aditya Balu, Adarsh Krishnamurthy, et al .2025. Leveraging vision language models for specialized agricultural tasks. In2025 IEEE/CVF
Winter Conference on Applications of Computer Vision (WACV). IEEE, 6320–6329.
[4]Angels Balaguer, Vinamra Benara, Renato Luiz de Freitas Cunha, Todd Hendry, Daniel Holstein, Jennifer Marsman, Nick Mecklenburg, Sara
Malvar, Leonardo O Nunes, Rafael Padilha, et al .2024. RAG vs fine-tuning: pipelines, tradeoffs, and a case study on agriculture.arXiv preprint
arXiv:2401.08406(2024).
[5]Mangesh Balpande, Kalparatna Mahajan, Jayesh Bhandarkar, Gaurav Borse, and Sagar Badjate. 2024. AI Powered Agriculture Optimization Chatbot
Using RAG and GenAI. In2024 IEEE Silchar Subsection Conference (SILCON 2024). IEEE, 1–6.
[6]Shawn Banks and Lucy Bradley. 2023. Vegetable Gardening: A Beginner’s Guide (Home Vegetable Gardening – A Quick Reference Guide).
https://content.ces.ncsu.edu/home-vegetable-gardening-a-quick-reference-guide. Publication AG-12; accessed: 2025-11-25.
[7] Nyle C Brady and Ray R Weil. 2016. The Nature and Properties of Soils.Pearson(2016).
[8]Andrew Cart, Shaohu Zhang, Melanie Escue, Xugui Zhou, Haitao Zhao, Prashanth BusiReddyGari, Beiyu Lin, and Shuang Li. 2025. Decoding
Neighborhood Environments with Large Language Models. In2025 55th Annual IEEE/IFIP International Conference on Dependable Systems and
Networks Workshops (DSN-W). 290–297. doi:10.1109/DSN-W65791.2025.00078
[9]Norman Cliff. 1993. Dominance Statistics: Ordinal Analyses to Answer Ordinal Questions.Psychological Bulletin114, 3 (1993), 494–509. doi:10.1037/
0033-2909.114.3.494
[10] DeepSeek. 2025. DeepSeek. https://www.deepseek.com. Accessed: 2025-03-15.
[11] Patrick Es, Ananya Agarwal, Mayank Gupta, et al .2023. RAGAS: Automated Evaluation of Retrieval-Augmented Generation.arXiv preprint
arXiv:2309.15217(2023). https://arxiv.org/abs/2309.15217
[12] North Carolina Cooperative Extension. 2025. Integrated Pest Management: Publications and Factsheets. https://ipm.ces.ncsu.edu/publications/.
Accessed: 2025-11-25.
[13] Aruna Gauba, Irene Pi, Yunze Man, Ziqi Pang, Vikram S Adve, and Yu-Xiong Wang. 2025. Agmmu: A comprehensive agricultural multimodal
understanding and reasoning benchmark.arXiv preprint arXiv:2504.10568(2025).
[14] Google DeepMind. [n. d.]. Gemini. https://deepmind.google/technologies/gemini/. Accessed: 2025-05-15.
[15] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang. 2020. REALM: Retrieval-Augmented Language Model Pre-Training. InInternational Conference
on Machine Learning. 9254–9263.
[16] JL Havlin, Samuel L Tisdale, Werner L Nelson, and James D Beaton. 2013. Soil Fertility and Fertilizers: An Introduction to Nutrient Management.
Pearson(2013).
[17] Albert Jiang, Hugo Béchard, Pierre Mazare, et al. 2023. Mistral 7B.arXiv preprint arXiv:2310.06825(2023). https://arxiv.org/abs/2310.06825
[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage
Retrieval for Open-Domain Question Answering. InProceedings of the 2020 Conference on Empirical Methods in Natural Language Processing.
6769–6781.
[19] Josué Kpodo, Parisa Kordjamshidi, and A Pouyan Nejadhashemi. 2024. AgXQA: A benchmark for advanced Agricultural Extension question
answering.Computers and Electronics in Agriculture225 (2024), 109349.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in neural information processing systems33
Manuscript submitted to ACM

AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 15
(2020), 9459–9474.
[21] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. InText Summarization Branches Out (ACL Workshop). 74–81.
[22] Pengfei Liu, Tianyu Gao, Zhengyan Zhang, et al .2023. Prompt Engineering and Normalization Techniques for Large Language Models.arXiv
preprint arXiv:2304.11264(2023). https://arxiv.org/abs/2304.11264
[23] Grady L Miller, Charles Peacock, Arthur H Bruneau, Fred Yelverton, James P Kerns, RL Brandenburg, Richard Cooper, and Matthew Martin. 2021.
Carolina lawns: A guide to maintaining quality turf in the landscape. North Carolina Cooperative Extension Service.
[24] North Carolina Cooperative Extension. 2025. North Carolina Cooperative Extension. https://www.ces.ncsu.edu/. Accessed: 2025-11-25.
[25] OpenAI. 2024. GPT-4 Technical Report.arXiv preprint arXiv:2403.03206(2024). https://arxiv.org/abs/2403.03206
[26] OpenAI. 2025. OpenAI API Reference - Chat. https://platform.openai.com/docs/api-reference/chat Accessed: 2025-02-24.
[27] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. In
Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics. 311–318. doi:10.3115/1073083.1073135
[28] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. In
Proceedings of EMNLP. 2383–2392. doi:10.18653/v1/D16-1264
[29] Yellamanda Reddy and Sankara Reddy. 2019. Principles of Agronomy for Sustainable Agriculture.Kalyani Publishers(2019).
[30] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.Proceedings of EMNLP(2019),
3982–3992. https://arxiv.org/abs/1908.10084
[31] Nils Reimers and Iryna Gurevych. 2020. Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation.arXiv preprint
arXiv:2004.09813(2020).
[32] Nils Reimers and Iryna Gurevych. 2021. MPNet Sentence Embeddings for Semantic Similarity.SentenceTransformers Documentation(2021).
https://www.sbert.net Model: all-mpnet-base-v2.
[33] Zhiyuan Ren, Yiyang Su, and Xiaoming Liu. 2023. ChatGPT-powered hierarchical comparisons for image classification.Advances in neural
information processing systems36 (2023), 69706–69718.
[34] Weijia Shi, Tianyi Zhang, Yu Zhang, et al .2023. Reducing Hallucination in Large Language Models via Retrieval-Augmented Generation.arXiv
preprint arXiv:2309.01249(2023).
[35] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al .
2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context.arXiv preprint arXiv:2403.05530(2024).
[36] H. Touvron, L. Martin, K. Stone, P. Albert, A. Conneau, and G. Lample. 2024. LLaMA 3: Open Foundation and Instruction-Tuned Language Models.
(2024).
[37] Asaf Tzachor, Medha Devare, Catherine Richards, Pieter Pypers, Aniruddha Ghosh, Jawoo Koo, S Johal, and Brian King. 2023. Large language
models and agricultural extension services.Nature food4, 11 (2023), 941–948.
[38] Artem Vizniuk, Grigorii Diachenko, Ivan Laktionov, Agnieszka Siwocha, Min Xiao, and Jacek Smoląg. 2025. A comprehensive survey of retrieval-
augmented large language models for decision making in agriculture: unsolved problems and research opportunities.Journal of Artificial Intelligence
and Soft Computing Research15 (2025).
[39] Bo Yang, Yunkui Chen, Lanfei Feng, Yu Zhang, Xiao Xu, Jianyu Zhang, Nueraili Aierken, Runhe Huang, Hongjian Lin, Yibin Ying, et al .2025.
AgriGPT-VL: Agricultural Vision-Language Understanding Suite.arXiv preprint arXiv:2510.04002(2025).
[40] Bo Yang, Yu Zhang, Lanfei Feng, Yunkui Chen, Jianyu Zhang, Xiao Xu, Nueraili Aierken, Yurui Li, Yuxuan Chen, Guijun Yang, et al .2025. Agrigpt:
A large language model ecosystem for agriculture.arXiv preprint arXiv:2508.08632(2025).
[41] Shanglong Yang, Zhipeng Yuan, Shunbao Li, Ruoling Peng, Kang Liu, and Po Yang. 2024. Gpt-4 as evaluator: Evaluating large language models on
pest management in agriculture.arXiv preprint arXiv:2403.11858(2024).
[42] Dazhou Yu, Riyang Bao, Gengchen Mai, and Liang Zhao. 2025. Spatial-rag: Spatial retrieval augmented generation for real-world spatial reasoning
questions.arXiv preprint arXiv:2502.18470(2025).
[43] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating Text Generation with BERT. In
Proceedings of ICLR. https://arxiv.org/abs/1904.09675
[44] Yuqin Zhang, Qijie Fan, Xuan Chen, Min Li, Zeying Zhao, Fuzhong Li, and Leifeng Guo. 2025. IPM-AgriGPT: A Large Language Model for Pest and
Disease Management with a G-EA Framework and Agricultural Contextual Reasoning.Mathematics (2227-7390)13, 4 (2025).
[45] Yutong Zhou and Masahiro Ryo. 2024. Agribench: A hierarchical agriculture benchmark for multimodal large language models. InEuropean
Conference on Computer Vision. Springer, 207–223.
Manuscript submitted to ACM
# Cost-Efficient Cross-Lingual Retrieval-Augmented Generation for Low-Resource Languages: A Case Study in Bengali Agricultural Advisory

**Authors**: Md. Asif Hossain, Nabil Subhan, Mantasha Rahman Mahi, Jannatul Ferdous Nabila

**Published**: 2026-01-05 12:41:44

**PDF URL**: [https://arxiv.org/pdf/2601.02065v1](https://arxiv.org/pdf/2601.02065v1)

## Abstract
Access to reliable agricultural advisory remains limited in many developing regions due to a persistent language barrier: authoritative agricultural manuals are predominantly written in English, while farmers primarily communicate in low-resource local languages such as Bengali. Although recent advances in Large Language Models (LLMs) enable natural language interaction, direct generation in low-resource languages often exhibits poor fluency and factual inconsistency, while cloud-based solutions remain cost-prohibitive. This paper presents a cost-efficient, cross-lingual Retrieval-Augmented Generation (RAG) framework for Bengali agricultural advisory that emphasizes factual grounding and practical deployability. The proposed system adopts a translation-centric architecture in which Bengali user queries are translated into English, enriched through domain-specific keyword injection to align colloquial farmer terminology with scientific nomenclature, and answered via dense vector retrieval over a curated corpus of English agricultural manuals (FAO, IRRI). The generated English response is subsequently translated back into Bengali to ensure accessibility. The system is implemented entirely using open-source models and operates on consumer-grade hardware without reliance on paid APIs. Experimental evaluation demonstrates reliable source-grounded responses, robust rejection of out-of-domain queries, and an average end-to-end latency below 20 seconds. The results indicate that cross-lingual retrieval combined with controlled translation offers a practical and scalable solution for agricultural knowledge access in low-resource language settings

## Full Text


<!-- PDF content starts -->

Cost-Efficient Cross-Lingual Retrieval-Augmented
Generation for Low-Resource Languages: A Case
Study in Bengali Agricultural Advisory
1stMd. Asif Hossain
Dept. of Computer Science and Engineering
East West University
Dhaka, Bangladesh
asifhossain8612@gmail.com
3rdMantasha Rahman Mahi
Dept. of Computer Science and Engineering
East West University
Dhaka, Bangladesh
mantashamahi11@gmail.com2ndNabil Subhan
Dept. of Computer Science and Engineering
East West University
Dhaka, Bangladesh
nabilsubhan861@gmail.com
4thJannatul Ferdous Nabila
Dept. of Computer Science and Engineering
East West University
Dhaka, Bangladesh
jannatulferdousnabila1@gmail.com
Abstract—Access to reliable agricultural advisory remains
limited in many developing regions due to a persistent language
barrier: authoritative agricultural manuals are predominantly
written in English, while farmers primarily communicate in
low-resource local languages such as Bengali. Although recent
advances in Large Language Models (LLMs) enable natural
language interaction, direct generation in low-resource languages
often exhibits poor fluency and factual inconsistency, while cloud-
based solutions remain cost-prohibitive.
This paper presents a cost-efficient, cross-lingual Retrieval-
Augmented Generation (RAG) framework for Bengali agricul-
tural advisory that emphasizes factual grounding and practical
deployability. The proposed system adopts a translation-centric
architecture in which Bengali user queries are translated into
English, enriched through domain-specific keyword injection to
align colloquial farmer terminology with scientific nomenclature,
and answered via dense vector retrieval over a curated corpus
of English agricultural manuals (FAO, IRRI). The generated
English response is subsequently translated back into Bengali
to ensure accessibility.
The system is implemented entirely using open-source models
and operates on consumer-grade hardware without reliance on
paid APIs. Experimental evaluation demonstrates reliable source-
grounded responses, robust rejection of out-of-domain queries,
and an average end-to-end latency below 20 seconds. The results
indicate that cross-lingual retrieval combined with controlled
translation offers a practical and scalable solution for agricultural
knowledge access in low-resource language settings.
Index Terms—Retrieval-Augmented Generation (RAG), Cross-
Lingual NLP, Low-Resource Languages, Bengali, Agricultural
Advisory, Quantization, Large Language Models (LLMs)
I. INTRODUCTION
Agriculture plays a vital role in developing countries such
as Bangladesh, where millions of people depend on farming
for food security and income. International organizations in-
cluding the Food and Agriculture Organization (FAO) and the
International Rice Research Institute (IRRI) publish detailedagricultural manuals containing scientifically validated guid-
ance on crop diseases, fertilizer usage, and best practices [1],
[2]. However, a major accessibility challenge remains: these
manuals are predominantly written in English and distributed
as static PDF documents. For smallholder farmers who pri-
marily communicate in Bengali, this information is effectively
inaccessible.
Recent advances in Large Language Models (LLMs) have
enabled natural language interfaces for information access.
However, directly applying standard LLMs for Bengali agri-
cultural advisory presents significant limitations. Most high-
performing models are trained primarily on English data,
resulting in poor grammatical quality and factual inconsisten-
cies in Bengali outputs [3]. In addition, commercial cloud-
based LLM services are often cost-prohibitive for low-cost
rural deployment. More critically, generative models operating
without external grounding are prone to hallucinations, which
can lead to unsafe recommendations in agriculture-related
decision-making [4].
Retrieval-Augmented Generation (RAG) [5] has been pro-
posed as a solution to reduce hallucinations by grounding
responses in authoritative documents. In a RAG system, the
model retrieves relevant information from trusted sources be-
fore generating an answer. While effective, most existing RAG
frameworks are designed for English-language use or require
high computational resources, limiting their applicability in
low-resource linguistic and deployment settings [6].
In the Bangladeshi agricultural context, an additional chal-
lenge arises from a pronounced vocabulary gap. Farmers
frequently use local or colloquial terms to describe crop dis-
eases and symptoms (e.g., “Magra”), whereas official manuals
rely on scientific terminology (e.g., “Stem Borer”) [7]. This
mismatch prevents standard retrieval systems from effectivelyarXiv:2601.02065v1  [cs.CL]  5 Jan 2026

Fig. 1. System Architecture of the proposed Translation-Centric Cross-Lingual RAG Pipeline. The system processes Bengali queries by translating them to
English, enriching them with domain-specific keywords, and retrieving relevant information from English manuals before generating a grounded response.
linking user queries to relevant technical documents.
To address these challenges, we propose a cost-efficient,
cross-lingual RAG framework tailored for Bengali agricultural
advisory. Rather than forcing the model to generate responses
directly in Bengali, we adopt a translation-based approach
[8]. User queries are translated from Bengali to English,
augmented using a domain-specific keyword mapping strategy
to align colloquial expressions with scientific terminology,
and then used to retrieve relevant passages from English
agricultural manuals. The system generates a grounded English
response, which is subsequently translated back into Bengali
for user-facing output.
The proposed system is implemented entirely using open-
source components and runs on standard consumer-grade hard-
ware, avoiding reliance on paid cloud APIs. Empirical evalua-
tion through representative query examples demonstrates that
the system produces source-backed, contextually relevant re-
sponses while maintaining practical inference latency suitable
for real-world advisory scenarios.
The main contributions of this work are summarized as
follows:
•We design a translation-first, cross-lingual RAG pipeline
tailored for Bengali agricultural advisory.
•We introduce a domain-specific keyword mapping strat-
egy to bridge colloquial farmer language and scientific
documentation.
•We present a fully local, cost-efficient implementation
suitable for deployment in resource-constrained environ-
ments.
II. RELATEDWORK
Recent research has explored the application of Retrieval-
Augmented Generation (RAG) to improve factual reliabilityin knowledge-intensive tasks. Lewis et al. [5] introduced the
foundational RAG framework, demonstrating that combining
neural retrieval with generative models can significantly reduce
hallucinations by grounding responses in external documents.
Subsequent studies have extended RAG to specialized do-
mains, including medicine and agriculture [6].
Several works have focused specifically on domain-adapted
RAG systems.AgroLLM[7] and related studies demonstrated
that agricultural question answering benefits from retrieving
information from curated expert manuals rather than relying
solely on parametric model knowledge. However, these sys-
tems are primarily designed for English-language inputs and
often assume access to high-performance computing resources.
Low-resource language challenges in RAG have been high-
lighted in multiple recent studies. Research on Bengali and
other South Asian languages shows that direct generation
using multilingual or English-centric LLMs frequently results
in degraded fluency and factual accuracy [3]. Studies such
asBanglaMedQA[9] emphasize that retrieval alone is in-
sufficient; intelligent routing and grounding mechanisms are
necessary to achieve reliable performance in low-resource
contexts.
Cross-lingual RAG has emerged as a promising solution.
Prior work such asXRAG[8] has demonstrated that translating
low-resource language queries into English before retrieval
can significantly improve document matching [10]. Large-
scale multilingual translation models, such as NLLB [11] and
Helsinki-NLP, have been shown to preserve domain-specific
semantics when applied carefully. However, existing cross-
lingual RAG systems often rely on cloud-based APIs. To
address robustness, recent benchmarks have also explored
culturally sensitive RAG tasks [12].
Recent investigations into cost-efficient model deployment

have demonstrated that quantization techniques can substan-
tially reduce memory and compute requirements [13]. Quan-
tized open-source LLMs enable fully local deployment, which
is critical for privacy and offline accessibility. Techniques
such as LoRA [14] further optimize these processes. However,
few studies integrate quantization, cross-lingual retrieval, and
domain-specific vocabulary alignment into a single system.
Our work addresses these gaps by proposing a translation-
first, locally deployable RAG system tailored for Bengali
agricultural advisory.
III. SYSTEMARCHITECTURE ANDMETHODOLOGY
A. System Overview
The proposed system is designed as a translation-centric,
cross-lingual RAG pipeline. The core design principle is to
separate user interaction language (Bengali) from reasoning
and retrieval language (English). The system follows five
sequential stages: (1) Bengali query processing, (2) query
translation and keyword normalization, (3) document retrieval
from authoritative English manuals, (4) grounded answer
generation, and (5) output translation into Bengali.
B. Data Collection and Knowledge Base Construction
The knowledge base consists of a curated collection of
English-language agricultural manuals published by author-
itative sources such as FAO and IRRI [1], [2]. Each PDF
document is processed using automated document loaders,
after which the text is segmented into overlapping chunks of
fixed length to preserve contextual continuity.
C. Bengali Query Processing and Translation
User queries are provided in Bengali. Direct reasoning in
Bengali is avoided due to the known limitations of English-
centric LLMs. Instead, each Bengali query is translated into
English using an open-source neural machine translation
model [15].
D. Domain-Specific Keyword Mapping
A key challenge is the mismatch between colloquial farmer
terminology and scientific language. To address this, the sys-
tem incorporates a domain-specific keyword mapping mecha-
nism. This component augments translated queries by injecting
standardized scientific terms corresponding to known collo-
quial expressions, improving retrieval recall without requiring
complex ontologies.
E. Dense Vector Retrieval
For document retrieval, the system employs dense vector
similarity search. Each text chunk in the knowledge base is
embedded using a multilingual sentence embedding model
[16]. These embeddings are indexed using a vector database
(FAISS) to enable efficient similarity-based retrieval [17].F . Grounded Answer Generation
The retrieved document chunks are provided as contextual
input to a locally deployed, quantized large language model
[18]. The model is prompted with strict instructions to generate
responses only based on the retrieved context. If the required
information is not present, the model explicitly states that the
information is unavailable.
G. Output Translation to Bengali
The grounded English response is translated back into Ben-
gali using the NLLB framework [11]. This final Bengali output
is presented to the user, ensuring the underlying reasoning is
derived from validated English sources.
H. Local and Cost-Efficient Deployment
All components operate locally. The language model is
deployed using quantization techniques to reduce memory
requirements, enabling execution on standard consumer hard-
ware [13], [18].
IV. EXPERIMENTALSETUP
A. Dataset and Knowledge Base
We curated a domain-specific corpus of English-language
agricultural manuals from FAO and IRRI [1], [2]. The fi-
nal corpus consisted of approximately 180 pages, producing
around 650–700 text chunks (600 characters with 50-character
overlap) after preprocessing.
B. Configuration
•Translation:Helsinki-NLP (opus-mt-bn-en) for in-
put; NLLB-200 for output.
•Retrieval:Sentence-Transformers
(all-MiniLM-L6-v2) and FAISS index.
•LLM:Llama-3-8B-Instruct (4-bit quantized via Unsloth).
•Hardware:Single NVIDIA Tesla T4 GPU (16 GB
VRAM) on Kaggle.
V. RESULTS ANDDISCUSSION
This section presents the qualitative and empirical analysis
of the system.
A. Qualitative Performance Analysis
We evaluated the system using representative queries in
three categories: Disease Diagnosis, Dosage Instructions, and
Out-of-Domain checks.
TABLE I
QUALITATIVEANALYSIS OFSYSTEMRESPONSES
Category User Query
(Bengali)Retrieved Con-
ceptVerdict
Disease Di-
agnosisSymptoms of
Rice BlastRice Blast /P .
oryzaeSuccess
Dosage
InstructionUrea Rules Urea / Nitrogen
App.Success
Out-of-
DomainWho is US Pres-
ident?Politics / Irrele-
vantPass

The results demonstrate effective domain-specific keyword
injection. For example, local terms for “Blast” were success-
fully mapped toPyricularia oryzae, enabling accurate retrieval
from FAO and IRRI manuals.
B. System Latency
The average end-to-end latency was approximately 15.6
seconds per query on a Tesla T4 GPU. A breakdown of the
latency is shown in Fig. 2. While higher than monolingual
English systems, this is acceptable for asynchronous advisory
use cases where accuracy is more critical than sub-second
speed.
Fig. 2. Latency Breakdown: Translation and LLM inference time compared
to Retrieval time.
C. Source Distribution
The system demonstrated balanced retrieval from multiple
authoritative sources (FAO, IRRI) depending on the query type
(see Fig. 3). Disease queries largely mapped to FAO pest
guides, while fertilizer queries mapped to IRRI production
manuals. This validates the effectiveness of the retrieval mech-
anism in selecting the correct context.
Fig. 3. Distribution of Retrieved Documents showing reliance on authoritative
FAO and IRRI manuals.VI. LIMITATIONS
Despite promising results, the proposed system has several
limitations that warrant further investigation.
A. Dependency on Translation Quality
The framework relies on neural machine translation to
bridge Bengali and English. Errors or ambiguities in the initial
Bengali-to-English translation may propagate to the retrieval
and reasoning stages, potentially affecting answer accuracy.
B. Dialect and Linguistic Variation
Bengali spoken in Bangladesh exhibits significant regional
variation (e.g., Sylheti, Chittagonian, Rangpuri). The current
system assumes Standard Bengali input and does not explicitly
handle dialectal spellings, pronunciations, or region-specific
vocabulary, which may reduce performance for non-standard
inputs.
C. Inference Latency
The average end-to-end latency of approximately 15.6 sec-
onds is acceptable for asynchronous agricultural advisory but
is unsuitable for real-time conversational interaction.
D. Static Knowledge Base
The system operates over a fixed corpus of agricultural
manuals and cannot answer dynamic or time-sensitive queries,
such as daily weather conditions or real-time market prices.
E. Accessibility Constraints
The current implementation supports only text-based inter-
action. This limits accessibility for illiterate or semi-literate
farmers, who constitute a significant portion of the target user
population.
VII. CONCLUSION ANDFUTUREWORK
This paper presented a cost-efficient, cross-lingual
Retrieval-Augmented Generation (RAG) framework designed
to improve access to agricultural knowledge for Bengali-
speaking users. By adopting a translation-centric “sandwich
architecture” (Translation→Retrieval→Translation) and
leveraging 4-bit quantized open-source language models,
the system enables accurate, source-grounded responses on
consumer-grade hardware without reliance on paid cloud
APIs.
Experimental results demonstrate that the proposed ap-
proach effectively bridges the gap between English-language
agricultural manuals and low-resource language users, while
maintaining strong factual grounding and robust rejection of
out-of-domain queries. The findings confirm that cross-lingual
retrieval, combined with controlled translation and domain-
specific keyword mapping, offers a practical and scalable solu-
tion for agricultural advisory in resource-constrained settings.
Future work will focus on several key extensions. First,
integrating automatic speech recognition (ASR) will improve
accessibility for illiterate users. Second, handling regional
Bengali dialects through dialect-aware normalization or mul-
tilingual embeddings will enhance robustness across diverse

user populations. Third, expanding the keyword mapping
mechanism using automated ontology or knowledge graph
construction may reduce manual effort and improve coverage.
Finally, incorporating quantitative evaluation benchmarks and
real-world user studies will provide deeper insight into system
effectiveness and usability.
REFERENCES
[1] Food and Agriculture Organization of the United Nations,Good Agri-
cultural Practices (GAP) for Rice Production, FAO, Rome, Italy, 2016.
[2] International Rice Research Institute,Rice Production Manual, IRRI,
Los Ba ˜nos, Philippines, 2015.
[3] K. M. J. Sami, D. Sumit, A. Hossain, and F. Sadeque, “A compar-
ative analysis of retrieval-augmented generation techniques for ben-
gali standard-to-dialect machine translation using LLMs,”arXiv, 2025,
arXiv:2511.04560.
[4] A. Mansurova, A. Mansurova, and A. Nugumanov, “QA-RAG: Explor-
ing LLM reliance on external knowledge,” inFindings of the Association
for Computational Linguistics, 2025.
[5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, and S. Riedel,
“Retrieval-augmented generation for knowledge-intensive nlp tasks,” in
Advances in Neural Information Processing Systems, 2020.
[6] N. A. Akbar, D. Tegolo, R. Dembani, and B. Lenzitti, “RAG-driven
memory architectures in conversational LLMs: A literature review with
insights into emerging agriculture data sharing,”arXiv, 2024.
[7] D. J. Samuel, I. Skarga-Bandurova, D. Sikolia, and M. Awais,
“AgroLLM: Connecting farmers and agricultural practices through large
language models for enhanced knowledge transfer,”Applied Sciences,
vol. 15, no. 4425, 2025.
[8] W. Liu, S. Trenous, L. F. R. Ribeiro, B. Byrne, and F. Hieber,
“XRAG: Cross-lingual retrieval-augmented generation,”arXiv, 2025,
arXiv:2505.10089.
[9] S. Sultana, S. S. Muna, M. Z. Samarukha, A. Abrar, and T. M. Chowd-
hury, “BanglaMedQA and BanglaMMedBench: Evaluating retrieval-
augmented generation strategies for bangla biomedical question answer-
ing,”Big Data and Cognitive Computing, vol. 8, no. 115, 2024.
[10] Z. Li and Z. Ke, “Cross-modal augmentation for low-resource language
understanding and generation,”arXiv, 2024, arXiv:2406.10251.
[11] NLLB Team, “No language left behind: Scaling human-centered ma-
chine translation,”arXiv, 2022, arXiv:2207.04672.
[12] B. Li, F. Luo, S. Haider, A. Agashe, T. Li, R. Liu, M. Miao, S. Ra-
makrishnan, Y . Yuan, and C. Callison-Burch, “Multilingual retrieval-
augmented generation for culturally-sensitive tasks: A benchmark for
cross-lingual robustness,”arXiv, 2025, arXiv:2512.14179.
[13] M. Yazan, S. Verberne, and F. Situmeang, “The impact of quantization
on retrieval-augmented generation: An analysis of small LLMs,”arXiv,
2025, arXiv:2503.04788.
[14] Y . Choi, S. Kim, Y . C. F. Bassole, and Y . Sung, “Enhanced retrieval-
augmented generation using low-rank adaptation,”arXiv, 2025.
[15] J. Tiedemann and S. Thottingal, “OPUS-MT: Building open translation
services for the world,” inProceedings of the European Association for
Machine Translation, 2020.
[16] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings
using siamese BERT-networks,” inProceedings of the 2019 Conference
on Empirical Methods in Natural Language Processing, 2019.
[17] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with GPUs,”IEEE Transactions on Big Data, 2019.
[18] AI at Meta, “The LLaMA 3 herd of models,”arXiv preprint
arXiv:2407.21783, 2024.
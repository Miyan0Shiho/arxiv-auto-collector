# KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers

**Authors**: Mohd Ruhul Ameen, Akif Islam, Farjana Aktar, M. Saifuzzaman Rafat

**Published**: 2025-10-21 07:24:55

**PDF URL**: [http://arxiv.org/pdf/2510.18355v1](http://arxiv.org/pdf/2510.18355v1)

## Abstract
In Bangladesh, many farmers continue to face challenges in accessing timely,
expert-level agricultural guidance. This paper presents KrishokBondhu, a
voice-enabled, call-centre-integrated advisory platform built on a
Retrieval-Augmented Generation (RAG) framework, designed specifically for
Bengali-speaking farmers. The system aggregates authoritative agricultural
handbooks, extension manuals, and NGO publications; applies Optical Character
Recognition (OCR) and document-parsing pipelines to digitize and structure the
content; and indexes this corpus in a vector database for efficient semantic
retrieval. Through a simple phone-based interface, farmers can call the system
to receive real-time, context-aware advice: speech-to-text converts the Bengali
query, the RAG module retrieves relevant content, a large language model (Gemma
3-4B) generates a context-grounded response, and text-to-speech delivers the
answer in natural spoken Bengali. In a pilot evaluation, KrishokBondhu produced
high-quality responses for 72.7% of diverse agricultural queries covering crop
management, disease control, and cultivation practices. Compared to the
KisanQRS benchmark, the system achieved a composite score of 4.53 (vs. 3.13) on
a 5-point scale, a 44.7% improvement, with especially large gains in contextual
richness (+367%) and completeness (+100.4%), while maintaining comparable
relevance and technical specificity. Semantic similarity analysis further
revealed a strong correlation between retrieved context and answer quality,
emphasizing the importance of grounding generative responses in curated
documentation. KrishokBondhu demonstrates the feasibility of integrating
call-centre accessibility, multilingual voice interaction, and modern RAG
techniques to deliver expert-level agricultural guidance to remote Bangladeshi
farmers, paving the way toward a fully AI-driven agricultural advisory
ecosystem.

## Full Text


<!-- PDF content starts -->

KrishokBondhu: A Retrieval-Augmented
V oice-Based Agricultural Advisory Call Center for
Bengali Farmers
Mohd Ruhul Ameen
Marshall University
Huntington, WV , USA
ameen@marshall.eduAkif Islam
University of Rajshahi
Rajshahi, Bangladesh
iamakifislam@gmail.comFarjana Aktar
University of Rajshahi
Rajshahi, Bangladesh
farjana.aktar.cseru@gmail.comM. Saifuzzaman Rafat
University of Rajshahi
Rajshahi, Bangladesh
saif.rafat@gmail.com
Abstract—In Bangladesh, many farmers continue to face chal-
lenges in accessing timely, expert-level agricultural guidance. This
paper presents KrishokBondhu, a voice-enabled, call-centre–
integrated advisory platform built on a Retrieval-Augmented
Generation (RAG) framework, designed specifically for Bengali-
speaking farmers. The system aggregates authoritative agricul-
tural handbooks, extension manuals, and NGO publications;
applies Optical Character Recognition (OCR) and document-
parsing pipelines to digitize and structure the content; and
indexes this corpus in a vector database for efficient semantic
retrieval. Through a simple phone-based interface, farmers can
call the system to receive real-time, context-aware advice: speech-
to-text converts the Bengali query, the RAG module retrieves
relevant content, a large language model (Gemma 3-4B) gen-
erates a context-grounded response, and text-to-speech delivers
the answer in natural spoken Bengali. In a pilot evaluation,
KrishokBondhu produced high-quality responses for 72.7% of
diverse agricultural queries covering crop management, disease
control, and cultivation practices. Compared to the KisanQRS
benchmark, the system achieved a composite score of 4.53 (vs.
3.13) on a 5-point scale—a 44.7% improvement—with especially
large gains in contextual richness (+367%) and completeness
(+100.4%), while maintaining comparable relevance and tech-
nical specificity. Semantic-similarity analysis further revealed a
strong correlation between retrieved context and answer quality,
emphasizing the importance of grounding generative responses
in curated documentation. KrishokBondhu demonstrates the fea-
sibility of integrating call-centre accessibility, multilingual voice
interaction, and modern RAG techniques to deliver expert-level
agricultural guidance to remote Bangladeshi farmers, paving the
way toward a fully AI-driven agricultural advisory ecosystem.
Index Terms—Retrieval-Augmented Generation, Agricultural
Advisory System, Bengali NLP, Voice Interface, Large Language
Models, Knowledge Dissemination
I. INTRODUCTION
Agriculture continues to play a pivotal role in Bangladesh’s
economy. As reported in the 2022 Labour Force Survey, about
45.33% of employment was in the agricultural sector, a rise
from 40.6% in 2016–17 [1]. Yet, more recent estimates based
on ILO-modeled data suggest that the share has declined to
around 35.27% [2], reflecting shifts in the labor structure.
Despite its central importance, many farmers still lack reliable
and timely guidance on crucial issues such as crop diseases,
pest outbreaks, optimal cultivation techniques, and efficientresource use. Traditional extension services, though invaluable,
are stretched too thin to provide real-time, on-demand support
to all communities, especially in remote or underserved areas.
Another barrier is language: much of the agricultural knowl-
edge base is written in English or technical Bengali, making
it less accessible to farmers who may not be comfortable
with formal or specialized terminology. At the same time,
advances in large language models (LLMs) and retrieval-
augmented generation (RAG) create new opportunities to
bring agricultural knowledge directly to farmers [3], [4]. But
using a general-purpose LLM without domain grounding often
yields vague or incorrect advice, because it may ignore local
context—such as particular crop varieties, region-specific pest
cycles, or soil conditions. Systems like AgAsk demonstrate
how combining retrieval from scientific documents with con-
versational interfaces can yield more accurate, contextually
relevant answers in the agriculture domain [4]. Yet in the
Bengali context, building a system that understands users’
spoken or written queries in natural language and delivers
culturally appropriate guidance involves dealing with mor-
phological complexity, dialect variation, and limited linguistic
resources.
In this paper, we present KrishokBondhu, an agricultural
advisory system that combines retrieval-augmented generation
(RAG) with a spoken interface to offer farmers real-time,
context-aware guidance in Bengali. The name KrishokBondhu
translates to “farmer’s friend,” and reflects our mission to
bring expert agricultural knowledge within reach. Our system
is built to tackle three core challenges: anchoring advice
in verified agricultural sources to reduce hallucinations or
vague outputs; allowing users to interact via speech so literacy
does not become a barrier; and adapting recommendations to
Bangladesh’s unique cropping systems, climate, and farmer
practices. We make three primary contributions:
1) A document processing pipeline that collects, OCRs,
cleans, segments, and indexes Bengali agricultural texts
into a vector retrieval store.
2) A full implementation of KrishokBondhu using RAG,
LanceDB, the Gemma 3-4B model, and a speech inter-
face to enable natural query and response in Bengali.arXiv:2510.18355v1  [cs.CL]  21 Oct 2025

3) An evaluation framework built around realistic farmer
queries and a comparative benchmark with Kishan QRS,
demonstrating measurable improvements in quality, con-
textual richness, and accuracy.
II. RELATEDWORK
Agricultural advisory systems have progressed from early
rule-based and vocabulary-driven approaches toward neural
and multimodal architectures. FAO’s AGROVOC, a multi-
lingual agricultural thesaurus, has long supported indexing
and cross-lingual retrieval [5], but its controlled-vocabulary
approach struggles with expressive, free-form farmer queries
lacking flexibility or semantic inference.
More recently, neural QA systems tailored for agriculture
have emerged. The KisanQRS system trains deep models over
Kisan Call Centre logs to map farmer queries to responses [6].
However, dataset access is limited, and such systems often lack
grounding in authoritative texts or visual evidence. Systems
like AgroLLM extend this by integrating RAG to improve
the relevance and correctness of responses using agricultural
databases [7].
To overcome hallucination and data sparsity, Retrieval-
Augmented Generation (RAG) has become a dominant
paradigm. By retrieving supporting context before generation,
RAG grounds outputs in factual sources [3], [8], [9]. While
RAG is already adopted in legal, medical, and code domains,
its application in agriculture—especially in low-resource lan-
guage contexts—remains underexplored.
V oice-based advisory approaches have been studied to
overcome literacy barriers. Surveys of voice assistants in
agriculture summarize their promise and challenges in rural
deployment [10]. In greenhouse trials, voice messaging sys-
tems combined with human-sensor inputs have been used to
build agricultural knowledge over time [11]. However, most
such systems rely on recording or message playback rather
than interactive, context-aware conversational responses [12]
Multimodal AI is another emerging direction. AgriDoctor
fuses image, text, and knowledge retrieval to build a multi-
modal assistant for crop disease diagnosis and domain-aware
QA [13]. Similarly, spatial-vision systems using Earth Obser-
vation and retrieval-augmented methods enable conversational
assessments of agricultural plots [14]. These systems point
to the future of combining multiple modalities in agricultural
advisory.
Nevertheless, gaps remain: (i) few systems fully integrate
speech input, multimodal grounding, and localized domain
knowledge; (ii) many are not evaluated in low-resource or
local-language settings; (iii) adaptation to regional cropping
systems and dialects is rare. Our work addresses these gaps by
offering a voice-enabled RAG system tailored to Bangladeshi
agriculture, combining text and voice modalities, and evaluat-
ing on domain-specific queries in Bengali.III. SYSTEMARCHITECTURE ANDMETHODOLOGY
A. Data Collection and Source Curation
KrishokBondhu’s knowledge base is constructed from au-
thoritative agricultural documentation published by govern-
mental and non-governmental organizations in Bangladesh.
Key sources include theKrishi Projukti Hatboi[15],
Bangladesh Agricultural Research Council (BARC) hand-
books, Department of Agricultural Extension (DAE) field
manuals [16], Bengali agricultural science textbooks, and
sector-specific publications from the WorldFish Digital Repos-
itory [17]. The curated corpus comprises approximately 2,500
pages encompassing major crops (rice, wheat, jute, vegetables,
pulses, and oilseeds), as well as livestock management, fish-
eries, and integrated farming practices. Source materials vary
from well-structured digital PDFs to low-quality scanned im-
ages, necessitating a robust and adaptive document processing
pipeline capable of handling OCR-based text extraction, error
correction, and content normalization.
TABLE I
KEYSOURCES FORAGRICULTURALKNOWLEDGEBASE
Source Summary
Krishi Projukti Haatboi Core Bengali handbook on crops
and cultivation practices.
BARC Handbook National reference on modern crop
varieties, irrigation, and pest con-
trol.
DAE Field Manuals Practical guides for pest, soil, and
irrigation management.
Agriculture Textbooks Basic school-level materials on
crops, soil, and pests.
Agricultural Science V ol. 1 Advanced text on plant physiology
and crop science.
Farmers’ Guidebook Covers aquaculture, horticulture,
and integrated farming.
BARC Bulletins Short crop-based manuals with re-
cent updates.
West Bengal Agro Books Regional references aligned with
similar climate and soil.
B. Document Processing and OCR Pipeline
All collected documents were processed through a Bengali-
capable Optical Character Recognition (OCR) and cleaning
pipeline to ensure accurate text extraction and normalization.
The workflow included image enhancement, skew correction,
and noise reduction prior to OCR to improve recognition
accuracy. Post-processing corrected common Bengali character
errors using rule-based and contextual validation methods.
Extracted text then underwent normalization to handle in-
consistent Unicode encoding, mixed scripts, and formatting
artifacts. Finally, the cleaned text was segmented into se-
mantically coherent sections (150–300 tokens), each enriched
with metadata such as source, topic, and structural position.
The standardized output was stored in Markdown format
to facilitate efficient vectorization and retrieval in the RAG
system.

C. Vector Database and Retrieval System
The processed and segmented corpus was transformed
into dense vector embeddings using a sentence-transformer
model [18] optimized for semantic similarity. These embed-
dings capture contextual meaning between user queries and
document content, enabling retrieval based on semantics rather
than exact keywords. LanceDB [19] was employed as the
vector storage backend for its efficiency and Python integra-
tion. Each text segment, along with metadata, was indexed for
fast approximate nearest-neighbor (ANN) search using cosine
similarity. During query time, user questions are embedded
with the same model, and the top-ksemantically relevant
segments are retrieved to form the contextual input for genera-
tion. A lightweight re-ranking layer further improves relevance
by considering metadata and keyword overlap, combining the
precision of lexical matching with the contextual strength of
embeddings.
D. Language Model Integration and Response Generation
Gemma 3-4B [20], deployed through LM Studio [21],
provides the generative component. The 4-billion parameter
model balances response quality with computational require-
ments for local deployment while maintaining strong Bengali
language performance.
Prompt engineering optimizes factual accuracy, relevance,
and appropriate farmer interaction tone. Retrieved segments
are incorporated as context, with instructions to generate
responses based primarily on provided context, cite spe-
cific practices, and acknowledge uncertainty when informa-
tion is insufficient. The prompt specifies accessible language,
practical focus, and culturally appropriate communication.
Conservative temperature and sampling parameters prioritize
consistency and accuracy. Post-processing verifies coherence,
checks for hallucinations against retrieved context, and formats
responses for voice delivery.
Fig. 1. Document Processing and OCR Pipeline
Fig. 2. Vector Database and Retrieval Workflow
E. Voice Interface Integration with VAPI
The V API [22] module enables seamless voice interac-
tion, extending KrishokBondhu’s accessibility to farmers with
limited literacy. It performs Bengali speech-to-text (STT)
for incoming queries and text-to-speech (TTS) synthesis for
system responses. Operating in a client–server configuration,
V API handles audio capture, transcription, and synthesis, while
the RAG server performs embedding, retrieval, and generation.
The complete workflow begins with the farmer’s spoken query,
which is transcribed into text, processed by the RAG engine,
and converted back into natural Bengali speech. To miti-
gate recognition errors—particularly with agricultural terms—
the system applies fuzzy matching, ambiguity prompts, and
contextual dialogue tracking, ensuring coherent multi-turn
conversations.
F . Evaluation Methodology
Due to the absence of a standard Bengali agricultural QA
benchmark, KrishokBondhu was evaluated through a compar-
Fig. 3. V oice Interface Integration Workflow with V API

ative analysis with published examples from Kishan QRS [6].
A representative test set of Bengali farmer-style questions was
manually curated from agricultural handbooks and extension
guides, covering major topics such as crop diseases, pest
control, fertilizer use, irrigation, and variety selection. Each
system’s responses were assessed by agricultural experts using
three quality levels—high, moderate, and poor—based on
accuracy, relevance, and practical usefulness. For Kishan QRS
comparison, a 5-point scoring framework was applied across
relevance, completeness, actionability, contextual richness, and
specificity.
To complement human evaluation, semantic similarity be-
tween questions, retrieved segments, and generated responses
was computed using cosine similarity. This quantified both
retrieval precision and generation faithfulness, helping identify
gaps and potential hallucinations in responses. Overall, the
evaluation captures factual correctness, contextual grounding,
and practical utility for Bengali-speaking farmers.
IV. RESULTS ANDDISCUSSION
A. Overall System Performance
As shown in Figure 4, Krishokbondhu delivered high-
quality responses for 72.7% of test queries, indicating strong
reliability in addressing farmers’ information needs. Moderate-
quality answers (9.1%) typically arose when relevant context
was partially retrieved or when queries required synthesizing
dispersed information. Poor responses (18.2%) were mostly
linked to under-represented topics or recent developments
not yet present in the knowledge base. Table II presents
representative responses from Krishokbondhu across diverse
agricultural queries.
Fig. 4. Distribution of response quality categories across test questions,
showing strong performance with 72.7% high-quality responses.
B. Comparative Analysis with Kishan QRS
To contextualize Krishokbondhu’s performance, we con-
ducted a comparative analysis with responses published in
the Kishan QRS paper. Table II presents matched query-
response pairs from both systems, highlighting substantial
differences in response characteristics. While Kishan QRS
provides concise, technically focused answers averaging 87
characters, Krishokbondhu generates comprehensive responses
averaging 692 characters, representing a 7.9-fold increase in
detail.Quantitative evaluation across five criteria reveals Krishok-
bondhu’s superior performance, as presented in Table III.
The system achieved an overall score of 4.53 out of 5.00,
representing a 44.7% improvement over Kishan QRS’s score
of 3.13. The most substantial gains appear in contextual
richness (4.67 vs 1.00, +367%) and completeness (4.67 vs
2.33, +100.4%), reflecting Krishokbondhu’s comprehensive
approach to agricultural advisory.
Figure 2 visualizes the comparative performance across
evaluation criteria, highlighting Krishokbondhu’s particular
strengths in providing contextual understanding and compre-
hensive coverage. Both systems maintain equivalent relevance
to queries and similar levels of specific technical details, but
Krishokbondhu significantly enhances actionability through
step-by-step guidance and completeness through multi-faceted
responses.
Fig. 5. Radar chart comparing Krishokbondhu and Kishan QRS across
five evaluation criteria, showing Krishokbondhu’s superior performance in
contextual richness and completeness.
Information coverage analysis (Table IV) reveals distinct
differences in response philosophy. Krishokbondhu consis-
tently provides cause explanations (100% of responses), pre-
vention measures (100%), and expert referrals (100%), fea-
tures largely absent from Kishan QRS responses. This compre-
hensive approach addresses a critical gap in agricultural exten-
sion: farmers often lack understanding of underlying causes,
leading to improper implementation of recommendations.
Figure 3 illustrates the percentage improvements across
evaluation criteria, with contextual richness showing the most
dramatic enhancement at 367%. These improvements validate
several design decisions in Krishokbondhu, particularly the
RAG architecture’s effectiveness in grounding responses in
comprehensive agricultural handbooks and the system’s op-
timization for voice-based interaction where detailed explana-
tions enhance farmer understanding.
C. Semantic Similarity and Response Quality Correlation
Figure 7 illustrates a clear positive correlation between
semantic similarity and response quality. Most high-quality
responses had similarity scores above 0.85, indicating strong
alignment between user queries and retrieved content. This

TABLE II
SAMPLERESPONSES FROMKRISHOKBONDHUSYSTEM(BENGALI)
Fig. 6. Percentage improvement of Krishokbondhu over Kishan QRS base-
line across evaluation criteria, demonstrating substantial gains in contextual
richness and completeness.
Fig. 7. Relationship between semantic similarity and overall quality scores,
showing positive correlation between retrieval quality and response assess-
ment.TABLE III
COMPARISON BETWEENKRISHOKBONDHU ANDKISHANQRS
Krishokbondhu (Bengali) Kishan QRS (English)
Retrieval-Augmented Generation
(RAG) pipeline using Gemma-3-4B
and LanceDB for context-grounded
Bengali answers.Rule-based and deep-learning mod-
els trained on historical Kisan Call
Centre logs.
Integrates OCR, ASR, and TTS
modules for Bengali speech-based
interaction.Text-only interface; accepts English
or transliterated Hindi input.
Knowledge base built from BARC’s
Krishi Projukti Hatboi, DAE manu-
als, WorldFish repository, and text-
books.Proprietary dataset from Kisan Call
Centre logs; not publicly available
or extensible.
Generates detailed responses with
explanations, preventive actions,
and expert referral guidance.Produces short prescription-style
answers focused on chemical
dosage.
Retrieval layer expandable through
new document ingestion.Static dataset; cannot adapt or up-
date post-training.
Supports voice and text I/O for low-
literacy farmers via mobile IVR.Operated by call-centre agents;
farmers interact indirectly.
Evaluated on factuality, relevance,
and fluency metrics.Evaluated on text-mapping accuracy
only.
suggests that effective retrieval is a strong predictor of answer
quality.
Interestingly, a few lower-similarity cases still produced
good answers—often when information was scattered across
multiple segments or required broader reasoning. On the other
hand, some high-similarity matches led to weaker responses,
underscoring that retrieval precision, not similarity alone, is
critical for consistent quality.
V. CONCLUSION
This paper presented Krishokbondhu, a voice-based agri-
cultural advisory system built on a RAG architecture tailored

TABLE IV
EVALUATIONMETRICCOMPARISON
Metric Krishokbondhu Kishan QRS % Gain
Relevance 5.00 5.00 +0.0%
Completeness 4.67 2.33 +100%
Actionability 4.33 3.33 +30%
Contextual Info 4.67 1.00 +367%
Specific Detail 4.00 4.00 +0.0%
Avg. Score 4.53 3.13 +44.7%
Resp. Length 692 chars 87 chars 7.9×
TABLE V
INFORMATIONCOVERAGEANALYSIS
Information Feature Krishokbondhu Kishan QRS
Cause Explanation 3/3 (100%) 0/3 (0%)
Immediate Actions 3/3 (100%) 3/3 (100%)
Prevention Measures 3/3 (100%) 0/3 (0%)
Specific Dosages 2/3 (67%) 3/3 (100%)
Variety Recommendations 2/3 (67%) 0/3 (0%)
Expert Referral 3/3 (100%) 1/3 (33%)
Average Coverage 83.3% 38.9%
for Bengali-speaking farmers. By integrating authoritative doc-
umentation, OCR-based digitization, semantic retrieval (via
LanceDB), and the Gemma 3–4B language model, the system
delivers context-aware, voice-enabled responses with 72.7%
high-quality output across diverse agricultural queries. Com-
pared to the Kishan QRS baseline, Krishokbondhu achieved a
44.7% improvement in overall quality, with substantial gains
in contextual richness and completeness.
Despite these promising results, the current evaluation has
limitations. Due to unavailability of public datasets, only a few
matched queries from Kishan QRS were used, and manual
assessments may introduce subjectivity. Broader evaluations,
including automatic metrics aligned with domain-expert judg-
ments, are needed. The knowledge base, while extensive, still
lacks coverage of newly released crop varieties and evolving
practices. Future work will focus on expanding content cover-
age, integrating real-time data (e.g., weather, market prices),
and optimizing response length for voice delivery.
Further improvements may involve domain-specific fine-
tuning of LLMs, incorporating structured knowledge graphs
alongside vector retrieval, and introducing adaptive response
modes (e.g., quick-reference vs. detailed guidance) based
on farmer profiles. Krishokbondhu marks an important step
toward democratizing expert agricultural advice and highlights
the potential of retrieval-augmented NLP systems to enhance
knowledge access in low-resource rural settings.
REFERENCES
[1] Bangladesh Bureau of Statistics, “Labour force survey 2022,”
Bangladesh Bureau of Statistics, Ministry of Planning,
Dhaka, Bangladesh, Tech. Rep., 2023, final Report.
[Online]. Available: https://bbs.portal.gov.bd/sites/default/files/
files/bbs.portal.gov.bd/page/b343a8b4_956b_45ca_872f_4cf9b2f1a6e0/
2023-10-25-07-38-4304abd7a3f3d8799fbcb59ff91007b1.pdf[2] “Employment in agriculture (% of total employment) — bangladesh,”
World Bank / ILO modeled estimate, 2025, modeled ILO estimates for
agricultural employment share.
[3] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive nlp tasks,” in
Advances in Neural Information Processing Systems, vol. 33, 2020, pp.
9459–9474.
[4] B. Koopman, A. Mourad, H. Li, A. van der Vegt, S. Zhuang, S. Gibson,
Y . Dang, D. Lawrence, and G. Zuccon, “Agask: An agent to help answer
farmer’s questions from scientific documents,” inarXiv preprint, 2022,
demonstrates retrieval-based QA in agriculture.
[5] C. Caracciolo, A. Stellato, M. Ježi ´c, J. Johansen, Y . Jaques, and J. Keizer,
“Agrovoc: The linked data concept hub for food and agriculture,”
Computers and Electronics in Agriculture, vol. 196, p. 106909, 2022.
[6] A. Kakkar, V . Gupta, and R. Prasad, “Kishan qrs: Question-answering
system for agricultural domain in indian languages,”arXiv preprint
arXiv:2105.13509, 2021.
[7] D. J. Samuel, I. Skarga-Bandurova, D. Sikolia, and M. Awais, “Agrollm:
Connecting farmers and agricultural practices through large language
models for enhanced knowledge transfer and practical application,” in
arXiv preprint, 2025.
[8] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval
augmented language model pre-training,” inProceedings of the 37th
International Conference on Machine Learning. PMLR, 2020, pp.
3929–3938.
[9] G. Izacard and E. Grave, “Leveraging passage retrieval with gener-
ative models for open domain question answering,”arXiv preprint
arXiv:2007.01282, 2021.
[10] S. K. S, Aparna, B. Jain, K. N. Hegde, and V . Gaurav, “A survey on
smart voice assistant for farmers,”International Journal for Research
Trends and Innovation (IJRTI), vol. 9, no. 1, 2024.
[11] N. Uchihira and M. Yoshida, “Agricultural knowledge management
using smart voice messaging systems: Combination of physical and
human sensors,”arXiv preprint arXiv:2008.03711, 2020.
[12] P. Lokhandeet al., “Farmer’s assistant using ai voice bot,” inInterna-
tional Journal of Advanced Research in Science, Communication and
Technology (IJARSCT), 2023.
[13] M. Zhang, Z. Xu, P. Wang, R. Li, L. Wang, Q. Liu, J. Xu, X. Zhang, and
S. Wu, “Agridoctor: A multimodal intelligent assistant for agriculture,”
arXiv preprint arXiv:2509.17044, 2025.
[14] J. Cañada, R. Alonso, J. Molleda, and F. Díez, “A multimodal con-
versational assistant for the characterization of agricultural plots from
geospatial open data,”arXiv preprint arXiv:2509.17544, 2025.
[15] Bangladesh Agricultural Research Council, “Krishi projukti hatboi
(handbook of agricultural technologies),” Bangladesh Agricultural Re-
search Council, Dhaka, Bangladesh, Tech. Rep., 2022.
[16] Department of Agricultural Extension, “Agricultural extension manual
for bangladesh,” Ministry of Agriculture, Government of Bangladesh,
Dhaka, Bangladesh, Tech. Rep., 2021.
[17] WorldFish, “Worldfish digital repository: Bangladesh aquaculture and
fisheries resources,” Digital Archive, 2022, collection of publications
on aquaculture, integrated farming, and fisheries management in
Bangladesh. [Online]. Available: https://digitalarchive.worldfishcenter.
org/
[18] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings
using siamese bert-networks,” inProceedings of the Conference on
Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, 2019, pp. 3982–3992.
[19] LanceDB, “Lancedb: Developer-friendly serverless vector database
for ai applications,” Software and Documentation, 2024, open-source
vector database built on Lance format for RAG and multimodal AI.
[Online]. Available: https://lancedb.com/
[20] Gemma Team, “Gemma: Open models based on gemini research
and technology,”arXiv preprint arXiv:2403.08295, 2024. [Online].
Available: https://arxiv.org/abs/2403.08295
[21] LM Studio, “Lm studio: Discover, download, and run local llms,”
Desktop Application, 2024, cross-platform application for running large
language models locally. [Online]. Available: https://lmstudio.ai/
[22] Vapi AI, “Vapi: V oice ai platform for developers,” V oice API Platform,
2024, developer platform for building and deploying conversational
voice AI agents. [Online]. Available: https://vapi.ai/
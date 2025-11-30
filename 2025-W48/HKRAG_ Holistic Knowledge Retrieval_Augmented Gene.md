# HKRAG: Holistic Knowledge Retrieval-Augmented Generation Over Visually-Rich Documents

**Authors**: Anyang Tong, Xiang Niu, ZhiPing Liu, Chang Tian, Yanyan Wei, Zenglin Shi, Meng Wang

**Published**: 2025-11-25 11:59:52

**PDF URL**: [https://arxiv.org/pdf/2511.20227v1](https://arxiv.org/pdf/2511.20227v1)

## Abstract
Existing multimodal Retrieval-Augmented Generation (RAG) methods for visually rich documents (VRD) are often biased towards retrieving salient knowledge(e.g., prominent text and visual elements), while largely neglecting the critical fine-print knowledge(e.g., small text, contextual details). This limitation leads to incomplete retrieval and compromises the generator's ability to produce accurate and comprehensive answers. To bridge this gap, we propose HKRAG, a new holistic RAG framework designed to explicitly capture and integrate both knowledge types. Our framework features two key components: (1) a Hybrid Masking-based Holistic Retriever that employs explicit masking strategies to separately model salient and fine-print knowledge, ensuring a query-relevant holistic information retrieval; and (2) an Uncertainty-guided Agentic Generator that dynamically assesses the uncertainty of initial answers and actively decides how to integrate the two distinct knowledge streams for optimal response generation. Extensive experiments on open-domain visual question answering benchmarks show that HKRAG consistently outperforms existing methods in both zero-shot and supervised settings, demonstrating the critical importance of holistic knowledge retrieval for VRD understanding.

## Full Text


<!-- PDF content starts -->

HKRAG: Holistic Knowledge Retrieval-Augmented Generation
Over Visually-Rich Documents
Anyang Tong1*Xiang Niu1*Zhiping Liu1Chang Tian2Yanyan Wei1Zenglin Shi1†Meng Wang1
1Hefei University of Technology2KU Leuven
Abstract
Existing multimodal Retrieval-Augmented Generation
(RAG) methods for visually rich documents (VRD) are
often biased towards retrieving salient knowledge(e.g.,
prominent text and visual elements), while largely ne-
glecting the critical fine-print knowledge(e.g., small text,
contextual details). This limitation leads to incomplete
retrieval and compromises the generator’s ability to pro-
duce accurate and comprehensive answers. To bridge this
gap, we propose HKRAG, a new holistic RAG framework
designed to explicitly capture and integrate both knowledge
types. Our framework features two key components: (1)
a Hybrid Masking-based Holistic Retriever that employs
explicit masking strategies to separately model salient and
fine-print knowledge, ensuring a query-relevant holistic
information retrieval; and (2) an Uncertainty-guided Agen-
tic Generator that dynamically assesses the uncertainty
of initial answers and actively decides how to integrate
the two distinct knowledge streams for optimal response
generation. Extensive experiments on open-domain visual
question answering benchmarks show that HKRAG consis-
tently outperforms existing methods in both zero-shot and
supervised settings, demonstrating the critical importance
of holistic knowledge retrieval for VRD understanding.
1. Introduction
Retrieval-augmented generation (RAG) has emerged as a
leading paradigm to enhance the factual accuracy of large
language models (LLMs) by grounding them in external
knowledge sources [5, 7, 31]. Traditionally, RAG frame-
works operate under the assumption of a purely textual cor-
pus. However, a vast amount of real-world information is
encapsulated within visually rich documents (VRDs), such
as reports, invoices, and web pages, where meaning is con-
veyed through an intricate interplay of textual and visual
elements (e.g., layout, typography, images) [14, 20, 28, 29,
*These authors contributed equally.
†Corresponding author: zenglin.shi@hfut.edu.cn
Visual-rich Documents 
Salient Knowledge
Fine-print Knowledge
Given query: How many points did Kobe Bryant score on November 13, 2012?
Only Fine-print Knowledge
Holistic Knowledge
OnlySalientKnowledge
26496 points
21 points
28 points
Figure 1. We demonstrate that both salient knowledge and fine-
print knowledge are critical for retrieval and generation. Only by
jointly leveraging both can we produce reliable answers and effec-
tively mitigate hallucinations.
32]. Applying text-centric RAG to these multimodal doc-
uments creates a significant modality gap, leading to sub-
stantial information loss and unreliable generation.
Recognizing this limitation, several studies [8, 19, 39]
have explored multimodal RAG for document visual ques-
tion answering (DocumentVQA), typically leveraging large
vision-language models (LVLMs) [5, 31]. These methods
aim to establish a query-to-document alignment by project-
ing both queries and document images into a shared embed-
ding space. While a step forward, their retrieval mechanism
remains fundamentally biased. They are primarily adept at
capturing salient knowledge,i.e., information that is visu-
ally prominent through large fonts, central placement, or
high contrast. Consequently, they often fail to retrieve fine-
print knowledge: critical details embedded in small text,
footnotes, or dense contextual passages. This bias results
in incomplete retrieval, and when the retrieved, saliency-
1arXiv:2511.20227v1  [cs.IR]  25 Nov 2025

biased context is passed to an LVLM, the generator lacks
the necessary context to produce fully accurate answers, es-
pecially for queries hinging on nuanced details.
We argue that a comprehensive understanding of VRDs
demands holistic knowledge, which necessitates the inte-
gration of both salient and fine-print knowledge. The for-
mer provides the global context and main points, while the
latter contains essential qualifications, conditions, and pre-
cise details. However, achieving this is profoundly chal-
lenging for two reasons. First, from a retrieval perspec-
tive, simultaneously identifying sparse fine-print elements
and broad salient features within a complex document is
difficult, as standard similarity search tends to be domi-
nated by salient signals. Second, in generation, simply
concatenating all retrieved information is suboptimal; the
generator must dynamically assess which pieces of knowl-
edge—salient or fine-print—are most relevant to the query
and integrate them coherently.
To address these challenges, we propose HKRAG, a
novel RAG framework designed for holistic knowledge re-
trieval and generation over VRDs. Our approach consists of
two key components. First, we introduce aHybrid Masking-
based Holistic Retriever. It employs explicit masking ap-
proaches to separately model and enhance the embeddings
for salient and fine-print content within a document, en-
suring a query-relevant retrieval process that captures both
knowledge types. Second, we develop anUncertainty-
guided Agentic Generator. This module does not passively
process all retrieved documents. Instead, it first dynami-
cally selects a minimal sufficient set of documents. Then,
based on the uncertainty of the initial answers, it adaptively
decides how to fuse the complementary information from
the salient and fine-print knowledge sources to arrive at a
final, well-grounded response.
To summarize, our contributions are as follows:
•We identify and formalize a critical limitation in exist-
ing multimodal RAG methods for VRDs: their bias to-
wards salient knowledge and their failure to retrieve fine-
print knowledge, which leads to theinadequate holistic-
knowledge problem.
•We propose the HKRAG framework, which features (1) a
novel hybrid masking-based retriever for balanced knowl-
edge retrieval and (2) an uncertainty-aware agentic gener-
ator for dynamic knowledge integration.
•Extensive experiments on open-domain DocumentVQA
benchmarks demonstrate that HKRAG consistently out-
performs the existing methods in both zero-shot and su-
pervised settings, validating its effectiveness and general-
izability.
2. Related work
Retrieval-augmented generation.RAG aims to construct
and retrieve external knowledge bases to reduce the genera-tion of hallucinatory content and enhance model credibility
across diverse tasks [5–7, 9, 31]. Traditional RAG methods
have achieved remarkable success in natural language pro-
cessing tasks and have been effectively extended to diverse
domains such as images, videos, and audio [18, 26]. How-
ever, most approaches tend to originate from an academic
research perspective, relying on knowledge retrieval from
clean corpora [2, 23, 37]. In this paper, we focus on retriev-
ing visually rich documents from real-world open-domain
scenarios, where answers can only be derived from the doc-
uments [14, 20, 28, 29, 32]. Additionally, the content within
these documents is presented in diverse formats, encom-
passing both prominent and nuanced knowledge that often
requires integration to correctly respond to queries. Con-
sequently, establishing an effective RAG pipeline for these
documents remains a challenge.
RAG for document visual question answering.Vi-
sual documents are widely used in real-life scenarios and
present content in diverse formats [14, 20, 29]. To re-
trieve query-relevant knowledge from visual documents,
traditional RAG methods [11, 15, 16, 27, 33, 34] typically
rely on text detection techniques (OCR) to scan textual in-
formation within documents, severely neglecting the vast
amount of visual information present. Visual large language
models (LVLMs) [5, 31, 36], which combine visual en-
coders with large language models, have gained widespread
attention for their image understanding capabilities. Ex-
isting RAG methods,e.g., VisRAG [39], DSE [19], Col-
Pali [8], and VDocRAG [30] for visual documents employ
LVLM encoding for both queries and documents, construct-
ing image-based knowledge bases and retrieving knowledge
by calculating similarity between external knowledge and
queries [8, 19, 30, 39]. These methods classically lever-
age query-document alignment to enhance retrieval perfor-
mance. However, we pinpoint that simultaneously identi-
fying sparse fine-print elements and broad salient features
within a complex document is difficult, as standard similar-
ity search tends to be dominated by salient signals. Second,
in generation, simply concatenating all retrieved informa-
tion is suboptimal; the generator must dynamically assess
which pieces of knowledge—salient or fine-print—are most
relevant to the query and integrate them coherently.
3. Method
3.1. Problem Formulation
In the open-domain document visual question answering
(DocumentVQA) task, the retrieverRfilters query-relevant
documents from external visually rich document set and
supplies them to the generatorGto improve answer re-
liability. BothRandGare built upon a large vision-
language model (LVLM) with a dual-encoder architecture,
encoding queries and documents independently. Specifi-
2

(b) Uncertainty-Guided Agentic Generator
VLMinitial answer: 3.6%
I’m inconfidenceinitial answer: 3.4%
I’m confidenceJudger:uncertainty-based intelligent routing
Query:WhatisthedifferencebetweentheLamResearchof2019and2020?
Buffer pool𝒟!
forin L:Can it resolve the query?
Re-ranked VRDPruner:dynamic document subset selection
Buffer pool𝒟!
HQP
LQPFinal answer
holistic info:TheLamResearch'smarketsharedecreasedfrom14.2%in2019to10.8%in2020.
Summarizer:faithful answer generationcorrect?
Decoupler
Decoupler:iterativefine-printknowledgemining
f. suggestion
e. decoupling?b. focus on data in2019 and 2020c. rethink ×Na. salient infoTheimageshowsabarchartofmarketsharebycompanyfrom2018to2020d. fine-print infoTheLamResearchof2019is14.2%andtheLamResearchof2020is10.8%Anchor info
ProjectorImageEncoderQuery:WhatisthedifferencebetweentheLamResearchof2019and2020?
...
(a) Hybrid Masking-based Holistic Retriever
LLM
LLM
sharedHybrid MaskM!
CorrelationMatrixSparseMaskLossSparse Alignment
DenseMaskLossDense Alignment
Figure 2. The proposed HKRAG includes (a) Hybrid Masking-based Holistic Retriever and (b) Uncertainty-Guided Agentic Generator.
cally, given a textual queryqand a collection ofNvisual
documents represented as imagesD={d 1, d2, ..., d N},
the retriever is defined as a functionR: (q,D)→ D R,
which takesqandDas inputs, and returns a ranked subset
DR={d r1, dr2, ..., d rk} ⊂ Dbased on a search algorithm.
DRoften consists of the fixed top-kdocuments most rele-
vant toq. For the queryqand retrieved resultsD R, the gen-
erator can be defined as a functionG: (q,D R)→a, which
takesqandD Ras inputs and concatenates their features to
feed into LLM for generating the answera.
A fundamental limitation of existing multimodal RAG
methods in this setting is their inherent bias towards salient
knowledge. These methods lack the capability to recognize
and retrieve fine-print knowledge. This results in a failure to
capture the holistic knowledge necessary for comprehensive
document understanding. Consequently, the generator of-
ten produces answers based on an incomplete context, lead-
ing to factual inaccuracies and hallucinations, especially for
queries that require reasoning over subtle details. To over-
come this limitation, we propose HKRAG, a framework
designed to explicitly model and integrate both knowledge
types. It introduces (1) a hybrid masking-based holistic re-
triever for balanced retrieval of salient and fine-print knowl-
edge, and (2) an uncertainty-guided agentic generator for
dynamic, iterative integration of the retrieved knowledge to
produce faithful answers.
3.2. Hybrid Masking-based Holistic Retriever
To retrieve the query-relevant holistic knowledge from the
document set, we first obtain the embeddings of queries and
documents. Specifically, each document image undergoes
scaling and dynamic cropping processes, which decom-
poses complex global information into simpler local ones.
The resulting document image patches are processed by avisual encoder and subsequently projected into document
features through a two-layer MLP. Next, both the query and
visual document features are fed into a LLM, from which
we extract their final-layer embeddings(v q,vd), where doc-
ument embedding v dcarries substantial information and
shares the same dimensions as query embedding v q.
❑Hybrid embedding-masking.We then design a hy-
brid embedding-masking approach that contains a dense
mask to localize salient knowledge within query-relevant
documents, along with a sparse mask to pinpoint the fine-
print knowledge. We begin by normalizing embeddings
(vq,vd)to obtain vn
qand vn
d, then compute their correlation
as Cq,d=|vn
q⊙vn
d|. To accommodate document features
with varying visual information densities and ensure accu-
rate masking, we further scale the correlation values to the
range(0,1), defined by Cn
q,d:
Cn
q,d=
1 + exp
−Cq,d−µ Cq,d
σCq,d+ϵ−1
,(1)
whereϵis a negligible value,µ Cq,dandσ Cq,ddenote the
mean and variance of C q,d, respectively. Notably, the mean
µCn
q,dand varianceσ Cn
q,drepresent the average level and the
fluctuation degree of correlation Cn
q,dbetween queries and
documents, respectively. By adjusting their magnitudes, we
define the correlation-based hybrid mask M Has follows:
MH= (1(Cn
q,d>(µ Cn
q,d−α·σ Cn
q,d))+
1(Cn
q,d>(µ Cn
q,d+α·σ Cn
q,d)))/2,(2)
whereαrepresents the hybrid scaling parameter. The hy-
brid mask leverages dense masking downward to preserve
low-correlation detail features, thereby maintaining em-
bedding diversity and avoiding dominant biases. Simul-
taneously, it employs sparse masking upward to amplify
3

highly correlated salient contexts, enhancing discernible fo-
cal points while suppressing irrelevant noise.
❑Bidirectional retriever-tuning.Our goal is not only to
bridge the modality gap between textual queries and visual
documents, but also to leverage the designed hybrid mask to
effectively extract query-relevant holistic knowledge within
rich visual documents. Specifically, we first employ In-
foNCE [24] as a manner to bridge the modality gap. Pos-
itive pairs are formed from query-document pairs(q, d+),
while in-batch negatives are used to compute the contrastive
lossL IN(vq,vd+)as follows:
LIN(vq,vd+) =−logexp(sim(v q,vd+)/τ)PB
i=1exp(sim(v q,vdi)/τ),(3)
where sim(·,·)denotes the cosine similarity function.τis
a temperature scaling parameter, andBrepresents the batch
size. To achieve more accurate query-to-document align-
ment, we employ the hybrid mask to compute the dense
matching lossL DIN:
LDIN=1
2(LIN(vq,vd+·M H) +L IN(vd+·M H,vq)).(4)
We enhance the alignment between queries and documents
in the embedding space, which facilitates the extraction of
salient knowledge. To further exploit fine-print knowledge,
we utilizeNpartial masks to compute the sparse matching
lossL SINas follows:
LSIN=1
NNX
i=1LIN(vq,vd+·Mi
H),(5)
where Mi
Hand Mi+1
Hhave the same dimensions, Mi
H∩
Mi+1
H= 0, and M H=Si=1
NMi
H. The mask partitioning po-
sitions are randomized to ensure alignment of details from
diverse regions. We employ the lossL=L DIN+β· L SIN
to fine-tune the retriever, whereβadjusts the contribution
ratio betweenL DINandL SIN. The retriever learns to map
queries and relevant documents into closer proximity within
the embedding space, effectively amplifying its sensitivity
to query-relevant holistic knowledge.
3.3. Uncertainty-Guided Agentic Generator
Merely concatenating all retrieved information can over-
whelm the generator and introduce noise. To enable a more
intelligent integration of the retrieved holistic knowledge
(encompassing both salient and fine-print knowledge), we
introduce an uncertainty-based criterion to classify query-
document pairs. Specifically, we define low-uncertainty
query-document pair (LQP) and high-uncertainty query-
document pair (HQP) below.
Definition 1(Low-uncertainty Query-document Pair, LQP).
Given a queryqand retrieved documentsD R, we feed theminto LVLMΘto obtain an initial answerˆa= Θ(q,D R)and
its token sequenceˆw={ˆw 1,ˆw2, ...,ˆw L}. We further calcu-
late the average entropyH( ˆw) =−1
LPL
i( ˆwi·log( ˆw i))of
ˆwand normalize it to obtainH′( ˆw)∈(0,1). WhenH′( ˆw)
is lower than the uncertainty thresholdh, we define this pair
as a low-uncertainty query-document pair.
Definition 2(High-uncertainty query-document pair,HQP).
Given a queryqand retrieved documentsD R, we compute
H′( ˆw)according to Definition 1. WhenH′( ˆw)is higher
than the uncertainty thresholdhat timet, we define this
pair as a high-uncertainty query-document pair at timet.
An LQP indicates that the LVLM can derive sufficient
query-relevant knowledge from the provided documents to
produce a confident response, no need of further process-
ing. In contrast, an HQP suggests that the initial under-
standing is inadequate, often due to the need to reconcile
subtle relationships between salient and fine-print knowl-
edge. For HQP, we introduce iterative, time-aware reason-
ing to facilitate deeper knowledge extraction via test-time
computing. Building on this classification, we design an
uncertainty-guided agentic generator composed of four spe-
cialized agents:Pruner,Judger,Decoupler, andSumma-
rizer. The overall architecture and interaction flow of these
agents are illustrated in Figure 2.
❑Pruner: dynamic document subset selection.Existing
methods often employ fixed top-kdocument rankings em-
pirically for generation. While higherkvalues increase the
probability of retrieving query-relevant knowledge, they si-
multaneously introduce substantial irrelevant noise, particu-
larly when processing visually rich documents, where abun-
dant prominent information obscures query-relevant details.
To address this issue, we introduce a pruner that evaluates
total query-relevant knowledgeI(q, n)of top-nretrieved
documents based on LVLMΘ, defined as
I(q, n) :=Pn
iΘ(q, d i), di∈ DP (6)
whered iis stored in a dynamic buffer poolD Pwith capac-
ityk(n < k). WhenI(q, n)can answer the queryq, the
evaluation process terminates; otherwise, the buffer retains
the top-kdocuments. Finally,I(q, n)is next passed to the
Judger.
❑Judger: uncertainty-based intelligent routing.Be-
yond ensuring that sufficient knowledgeI(q, n)is re-
trieved, assessing the reliability of the generated answer is
more important. To address this, we introduce aJudger
that evaluates whether the generator can confidently an-
swer a given query. Specifically, theJudgercategorizes
each query–document pair according to Equations 1 and 2.
Queries with low uncertainty are deemed straightforward
and sent directly to the final agent, the Summarizer, for effi-
cient processing. In contrast, queries with high uncertainty,
4

Datasets Documents #Images #Train #Test
ChartQA[20] Chart 20,882 – 150
OpenWikiTable[14] Table 1,257 4,261 –
SlideVQA‡[29] Slide 52,380 – 760
VisualMRC[28] Webpage 10,229 6,126 –
InfoVQA[22] Infographic 5,485 9,592 1,048
DocVQA[21] Industry 12,767 6,382 –
DUDE[32] Open 27,955 2,135 496
MHDocVQA‡ Open 28,550 9,470 –
Table 1. We evaluate our models under both zero-shot and super-
vised settings. Thezero-shotevaluation measures their ability to
generalize to unseen datasets, whereas thesupervisedevaluation
quantifies performance when training data is provided.‡denotes
datasets requiring multi-hop reasoning.
indicating a potential need to reconcile subtle relationships
between salient and fine-print knowledge, are routed to the
Decouplerfor deeper analysis. Notably, the initial answer
produced during uncertainty measurement is neither used as
model input nor reflected upon, preventing historical out-
puts from biasing subsequent decisions.
❑Decoupler: iterative fine-print knowledge mining.We
further introduce aDecouplerto deal with the high-entropy
queries passed from theJudger. The query-relevant salient
knowledge is denoted asI h(q, n). This information often
shares the semantic similarity with the queryqbut fails
to accurately answer it. Therefore, we perform iterative,
fine-grained reasoning to mine overlooked fine-print knowl-
edgeI d(q, n, t)by anchoring on the readily available salient
knowledgeI h(q, n)at time stept:
Id(q, n, t) = Θ(I h(q, n),I d(q, n, t−1)).(7)
To prevent the model from conflating two knowledge types
and thereby stimulating the discovery of crucial details that
are initially missed, we explicitly decouple them:
I′
d(q, n, t) = Θ(I h(q, n),I d(q, n, t)).(8)
When the explored fine-print knowledgeI′
d(q, n, t)can an-
swer the query or reach the maximum iteration count; we
feed both the decoupled salient information and detail in-
formation to theSummarizer.
❑Summarizer: faithful answer generation.
Finally, theSummarizerserves as the synthesis point for
all information flows. It receives inputs from two paths:
direct, low-uncertainty cases from theJudger, and compre-
hensively reasoned information from theDecoupler. For
the former, it performs a final verification. For the latter, it
refines and integrates the global context and fine details to
generate the final, faithful answer. Through this structured
collaboration, our agentic generator dynamically adapts itsreasoning depth, ensuring both efficient processing of sim-
ple queries and comprehensive understanding of complex
ones.
4. Experiments
Data sources and settings.As the first collection of open-
domain visually rich documents, OpenDocVQA integrates
multiple DocumentVQA datasets that span a broad range
of real-world document types, as summarized in Table 1.
Following the protocol in [30], we evaluate model perfor-
mance on these public datasets under both supervised and
zero-shot settings. To more faithfully reflect real-world us-
age scenarios, we further conduct evaluations under two
retrieval configurations: single-pool and all-pool. In the
single-pool setting, retrieval is restricted to the document
pool corresponding to each individual dataset. In contrast,
the all-pool setting requires models to retrieve from a uni-
fied, large-scale corpus drawn from diverse domains, pro-
viding a more realistic and challenging assessment of re-
trieval robustness and cross-domain generalization.
Implementation details.For fair comparison, we ini-
tialize our retriever with Phi3V [1], a state-of-the-art LVLM
trained on high-resolution and multi-image datasets, consis-
tent with prior works [19, 30]. We fine-tune the retriever us-
ing LoRA [10] under two configurations—with and without
pre-training, while keeping the generator settings identical
to those in [30]. In HKRAG, the retriever and generator
use independent parameter sets, and all other components
remain frozen during fine-tuning. Training is conducted
for one epoch using eight NVIDIA RTX 4090 GPUs, with
the AdamW optimizer [17], FlashAttention [3] acceleration,
and a batch size of 64. The temperature parameterτis set
to 0.01, submask numberNis set to 2, and the Judger’s
uncertainty threshold is fixed at 0.8.
Unified evaluation metrics.We evaluate retrieval per-
formance using the nDCG@5 metric, a widely adopted
standard in information retrieval [8, 13]. For generation
evaluation, we follow [35] and uniformly report accuracy.
Specifically, we use the powerful Qwen-plus model to as-
sess the quality of each prediction against its ground-truth
answer, assigning a score on a 1–5 scale. A score of 4 indi-
cates a generally correct but less concise response, while a
score of 5 corresponds to a fully correct answer. Predictions
receiving either score are treated as correct, and all others
are counted as incorrect.
4.1. Retrieval Results
We evaluate HKRAG-Retriever against two categories of
retrievers. The first category comprises off-the-shelf text re-
trieval models and image retrieval models, including BM25
[27], Contriver [11], E5 [33], GTE [16], E5-Mistral [34],
NV-Embed-v2 [15], CLIP [25], DSE [19], and VisRAG-Ret
5

Model Initialization Docs Scale #PT #FTChartQA SlideVQA InfoVQA DUDE
Single All Single All Single All Single All
Off-the-shelf
BM25 [27] – Text 0 0 0 54.8 15.6 40.7 38.7 50.2 31.3 57.2 47.5
Contriever [11] BERT [4] Text 110M 1B 500K 66.9 59.3 50.8 46.5 42.5 21.0 40.6 29.7
E5 [33] BERT [4] Text 110M 270M 1M 74.9 66.3 53.6 49.6 49.2 26.9 45.0 38.9
GTE [16] BERT [4] Text 110M 788M 3M 72.8 64.7 55.4 49.1 51.3 32.5 42.4 36.0
E5-Mistral [34] Mistral [12] Text 7.1B 0 1.85M 72.3 70.0 63.8 57.6 60.3 33.9 52.2 45.2
NV-Embed-v2 [15] Mistral [12] Text 7.9B 0 2.46M 75.3 70.7 61.7 58.1 56.5 34.2 43.0 38.6
CLIP [25] Scratch Image 428M 400M 0 54.6 38.6 38.1 29.7 45.3 20.6 23.2 17.6
DSE [19] Phi3V [1] Image 4.2B 0 5.61M 72.7 68.5 73.0 67.2 67.4 49.6 55.5 47.7
VisRAG-Ret [39] MiniCPM-V [38] Image 3.4B 0 240K 87.2* 75.5* 74.3* 68.4* 71.9* 51.7* 56.4 44.5
Trained on OpenDocVQA
Phi3 [1] Phi3V [1] Text 4B 0 41K 72.5 65.3 53.3 48.4 53.2* 33.0* 40.5* 32.0*
VDocRetriever†[30] Phi3V [1] Image 4.2B 0 41K 80.8 71.2 66.0 60.4 63.9* 48.3* 44.1* 35.6*
VDocRetriever†[30] Phi3V [1] Image 4.2B 500K 41K 84.4 76.3 75.3 69.9 70.4* 53.9* 55.6* 47.9*
HKRAG-Retriever Phi3V [1] Image 4.2B 0 41K 83.7 77.0 69.5 62.4 65.8* 52.1* 45.4* 40.6*
HKRAG-Retriever Phi3V [1] Image 4.2B 500K 41K 87.6 82.1 76.2 72.5 72.4*57.8* 58.9*51.7*
Table 2. Comparison of HKRAG’s retrieval performance against embedding-based models and existing approaches in both single-pool
and all-pool settings across four open-domain DocumentVQA benchmarks. FT and PT indicate finetuning and pretraining, respectively. *
indicates performance on test data for which corresponding training samples are available, and†denotes reproduced results.
Generator Retriever DocsChartQA SlideVQA InfoVQA DUDE
Single All Single All Single All Single All
Phi3 [1] – – 12.7 12.7 23.7 23.7 18.1* 18.1* 8.9* 8.9*
Phi3 [1] Phi3 [1] Text 17.7 17.5 32.4 32.1 21.0* 20.3* 15.4* 13.7*
VDocGenerator [30] VDocRetriever [30] Image 54.0 51.3 52.2 53.9 39.4* 36.2* 37.3* 36.9*
HKRAG-Generator HKRAG-Retriever Image 60.7 58.0 56.5 56.6 42.7*39.6* 45.2*42.7*
Phi3 [1] Gold Text 23.2 23.2 33.4 33.4 23.7* 23.7* 21.5* 21.5*
VDocGenerator [30] Gold Image 77.3 77.3 72.1 72.1 54.1* 54.1* 64.5* 64.5*
HKRAG-Generator Gold Image 78.7 78.7 74.0 74.0 54.7*54.7* 68.8*68.8*
Table 3. DocumentVQA results. All models are fine-tuned on OpenDocVQA. The results marked with * denote performance on unseen
test samples, and the other results represent zero-shot performance. The performance gain in green is compared to the text-based RAG that
has the same base LLM. Gold knows the ground-truth documents. Models answer the question based on the top three retrieval results.
[39]. The second category consists of models fine-tuned on
the OpenDocVQA dataset.
Table 2 showcases the strong performance of HKRAG
across four open-domain DocumentVQA benchmarks un-
der both supervised and zero-shot settings. In the zero-shot
setting on ChartQA and SlideVQA, HKRAG achieves new
state-of-the-art results, with particularly notable gains in
the challenging all-pool scenario (82.1% on ChartQA and
72.5% on SlideVQA). On the supervised benchmarks In-
foVQA and DUDE, HKRAG also establishes new SOTA
performance across all evaluation metrics, demonstrating
excellent transferability and robustness. These results con-
firm that HKRAG effectively captures query-relevant holis-
tic knowledge from documents, substantially enhancing
retrieval quality. Even when compared with fine-tunedvision–language models such as DSE, VisRAG-Ret, and
VDocRAG, HKRAG consistently delivers superior perfor-
mance. This advantage primarily stems from our efficient
fine-tuning strategy, which enables precise localization of
both salient knowledge and fine-grained knowledge.
4.2. Retrieval-Augmented Generation Results
As shown in Table 3, HKRAG-Generator consistently out-
performs all competing methods across the four bench-
marks. Under the All setting, it delivers clear perfor-
mance improvements over VDocRAG—raising accuracy
on ChartQA by 6.7%, on SlideVQA by 3.5%, on InfoVQA
by 3.1%, and on DUDE by 6.4%. These gains stem
from the complementary strengths of HKRAG-Retriever
and HKRAG-Generator, which jointly enhance both docu-
ment selection and holistic reasoning over visually rich con-
6

MethodChartQA InfoVQA
Single All Single All
Baseline 50.7 44.7 31.5* 29.6*
HKRAG w/o Gen 51.3 49.3 31.9* 31.2*
HKRAG w/o Ret 57.3 52.7 41.9* 38.7*
HKRAG 60.7 58.0 42.7*39.6*
Table 4. Ablation study of Retriever (Ret) and Generator (Gen)
in HKRAG. The baseline employs Eq. 3 for retriever fine-tuning,
then directly inputs the generator.
MethodChartQA InfoVQA
Single All Single All
Baseline 84.4 76.3 70.4* 53.9*
Ret w/oL SIN 84.9 77.9 71.3* 55.2*
Ret w/oL DIN 85.7 79.4 70.7* 54.6*
HKRAG-Ret 87.6 82.1 72.4*57.8*
Table 5. Ablation study of different components in HKRAG-Ret.
MethodChartQA InfoVQA
Single All Single All
Baseline 51.3 49.3 31.9* 31.2*
Gen w/o pruner 56.0 54.7 40.4* 35.3*
Gen w/o decoupler 59.0 56.3 40.5* 38.0*
HKRAG-Gen 60.7 58.0 42.7*39.6*
Table 6. Ablation study of different components in HKRAG-Gen.
tent. Even in the Gold setting, HKRAG-Generator achieves
up to 4.3% higher accuracy than VDocGenerator, show-
ing that its benefits are not merely due to improved re-
trieval but rather from more effective holistic-knowledge
reasoning. Overall, the results demonstrate that HKRAG-
Generator substantially alleviates the holistic-knowledge
gap in open-domain visually rich document understanding,
enabling more accurate and dependable answer generation.
4.3. Analysis
Effect of each component in HKRAG.Table 4 reports
the ablation results of the hybrid masking–based holistic
retriever and the uncertainty-guided agentic generator on
ChartQA and InfoVQA. Removing either component re-
sults in clear performance drops, demonstrating that the
two modules play complementary roles. The bidirectional
retriever-tuning alone enhances retrieval accuracy by im-
proving query–document alignment, whereas the genera-
tor alone yields a more substantial improvement in answer
quality by adaptively selecting and reasoning over the in-
formative documents. When both modules are enabled,
HKRAG attains the highest performance across all settings,
indicating that holistic retrieval and adaptive reasoning syn-
Retrieval and Generation Flow2025/11/13 22:21127.0.0.1:53382
127.0.0.1:533821/1Retrieval and Generation Flow2025/11/13 22:16127.0.0.1:52838
127.0.0.1:528381/1
DUDEAll Queries:496CorrectRetrieval:293CorrectGeneration:182IncorrectRetrieval:203IncorrectGeneration:11140.9%59.1%36.7%22.4%(a) VdocRAGDUDEAll Queries:496CorrectRetrieval:304CorrectGeneration:225IncorrectRetrieval:192IncorrectGeneration:7938.7%61.3%45.4%15.9%(b) HKRAGFigure 3. Performance of (a) VDocRAG and (b) HKRAG on
DUDE. We present the distribution of queries across two cate-
gories: those that retrieved the correct document in the top-3 po-
sition (“correct retrieval”), and those that provided the correct an-
swer given the top-3 retrieved documents (“correct generation”).
020406080
100203040
qwen2.5-vl-3bqwen2.5-vl-7bqwen2.5-vl-32b68.071.381.335.362.775.3
1.82.25.511.715.428.3Accuracy (%)
Time  (s)
Figure 4. Comparison of VidoRAG [35] and our HKRAG across
different sizes within the same series. The shaded area represents
the gap between VidoRAG and HKRAG.
ergistically contribute to a more comprehensive understand-
ing of visually rich documents.
How does our retriever gain benefits?Table 5 presents
the ablation results for the individual components of the
Retriever. Removing eitherLSIN orLDIN leads to clear
performance degradation, highlighting their complemen-
tary roles. Specifically,LSIN reinforces fine-print knowl-
edge consistency, whileLDIN strengthens salient knowl-
edge consistency—both essential for reliable retrieval. The
full HKRAG-Ret achieves the highest accuracy across all
datasets and settings, demonstrating that jointly optimizing
these objectives strikes an effective balance between fine-
grained alignment and cross-domain generalization.
How does our generator gain benefits?In Table 6, re-
moving the dynamic buffer (Gen w/o pruner), which selects
minimal sufficient knowledge documents instead of using
fixed-K inputs, causes noticeable performance drops, par-
ticularly on InfoVQA, demonstrating its crucial role in effi-
cient knowledge retrieval. Without the decoupler that han-
dles HQP through reasoning disentanglement, performance
degrades to 56.3 (ChartQA All) and 38.0 (InfoVQA All),
7

Query1：Havewineimportamountsandvolumebeentrendingupordownduringtheperiodfrom1995totheyearinwhichItaly’syearlygrowthinsparklingvolumewasnegative?Reference document: Two slides; Answer: UpVdocRetriever: Top-3
Similar but helpless
Query2：InwhatconventioncenterwastheGlobalMobileGameCongressheldinMarch2014?Reference document: appotaupdates; Answer: CHINA NATIONAL
VdocRetriever: Top-3Can’t get detail
HKRAGRetriever: Top-1
Understand the holistic information in the document
HKRAGRetriever: Top-2
Accurate cross-documentresponse to multi-hop query
1
2
3
12
3
Query1：WhathappensbeforeIrritationofparietalperitoneum?Reference document: Time Course; Answer: Appendiceal distension 
Fixed K=3 documents
HallucinationGeneration:AppendicealobstructionDetailslostQuery2：WhatisthedifferencebetweenthesalesofmaximumsalesofcasualbagsandminimumsalesofTravelbags?Reference document: Sale value chart; Answer: 3594
Fixed K=3 documents
HallucinationGeneration:2382.45Detailslost
Theimageshowstheprogression，includingappendicealdistension.ReliableGeneration:Appendicealdistension.
Ranked DocumentsThisdocumentcontainsquery-relevantdetails.
Themaximumsalesofcasualbagsis12721andtheminimumsalesoftravelbagsis9127.ReliableGeneration:3594.Thisdocumentcontainsquery-relevantdetails.Ranked Documents
Dynamic SelectionDynamic Selection
VdocGeneratorHKRAGGeneratorFigure 5. Qualitative results of our HKRAG compared to state-of-the-art VDocRAG [30] on open-domain visually rich documents.
indicating its importance in resolving ambiguous cases.
HKRAG-Gen achieves the best results, validating that both
components synergistically enhance knowledge selection
and reasoning capability.
How accuracy and time efficiency vary under differ-
ent LVLM?In Figure 4, we demonstrate the accuracy and
time efficiency of VidoRAG and HKRAG across different
Qianwen 2.5 visual-language models, including 3b, 7b, and
32b. Through comparison, we observe that HKRAG consis-
tently achieves higher accuracy than VidoRAG under iden-
tical settings while requiring less computational time. This
indicates our generator efficiently and accurately infers an-
swers, owing to our uncertainty-guided reasoning process
that extracts holistic information based on uncertainty min-
ing within query-document pairs.
Qualitative results.We present a qualitative compar-
ison with VDocRAG in Figure 5. During the retrieval
phase (top), VDocRAG only includes documents semanti-
cally similar but irrelevant to Query 1 in its top-3 retrieval
results. In contrast, our retriever accurately identifies two
documents across the retrieval set that can answer the query,
effectively addressing the multi-hop problem. During the
generation phase (below), even when VDocRAG accurately
retrieves documents containing answers, it generates hal-
lucinations due to the difficulty of precisely inferring the
computational relationship between maximum and mini-mum values from extensive visual information. Notably,
our generator efficiently understands logical relationships
within instances by dynamically selecting query-relevant
documents amidst complex visual information.
5. Conclusion
In this paper, we identify a critical limitation in exist-
ing multimodal RAG methods for open-domain Docu-
mentVQA task: their inherent bias toward salient knowl-
edge and consistent oversight of fine-print knowledge,
which we term theinadequate holistic knowledge problem.
To address this, we propose HKRAG, a new multimodal
RAG framework that enables holistic retrieval and dynamic
integration of both knowledge types. Central to HKRAG
are two new components: a hybrid masking-based holistic
retriever, which explicitly enhances the model’s sensitivity
to both salient and fine-print content through structured fea-
ture masking, and an uncertainty-guided agentic generator,
which dynamically routes queries based on initial-answer
confidence and performs iterative reasoning when needed.
Extensive experiments on multiple DocumentVQA bench-
marks, including zero-shot (ChartQA, SlideVQA) and su-
pervised (InfoVQA, DUDE) settings, demonstrate that
HKRAG consistently outperforms the compared baselines,
confirming its robustness and generalizability in achieving
truly holistic document understanding.
8

References
[1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadal-
lah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, and
et al. Arash Bakhtiari. Phi-3 technical report: A highly ca-
pable language model locally on your phone.arXiv preprint
arXiv: 2404.14219, 2024. 5, 6
[2] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano,
Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola
Tonellotto, and Fabrizio Silvestri. The power of noise: Re-
defining retrieval for rag systems. InProceedings of the 47th
International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, pages 719–729, 2024.
2
[3] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christo-
pher R ´e. Flashattention: Fast and memory-efficient exact at-
tention with io-awareness.NeurIPS, 35:16344–16359, 2022.
5
[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. BERT: pre-training of deep bidirectional trans-
formers for language understanding. InNAACL-HLT, pages
4171–4186, 2019. 6
[5] Guanting Dong, Jiajie Jin, Xiaoxi Li, Yutao Zhu, Zhicheng
Dou, and Ji-Rong Wen. Rag-critic: Leveraging automated
critic-guided agentic workflow for retrieval augmented gen-
eration. InACL, pages 3551–3578, 2025. 1, 2
[6] Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang.
Minirag: Towards extremely simple retrieval-augmented
generation.arXiv preprint arXiv:2501.06713, 2025.
[7] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A sur-
vey on rag meeting llms: Towards retrieval-augmented large
language models. InSIGKDD, pages 6491–6501, 2024. 1, 2
[8] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, CELINE HUDELOT, and Pierre Colombo.
Colpali: Efficient document retrieval with vision language
models. InICLR, 2024. 1, 2, 5
[9] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. Lightrag: Simple and fast retrieval-augmented gen-
eration.arXiv preprint arXiv:2410.05779, 2024. 2
[10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models.ICLR,
1(2):3, 2022. 5
[11] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard
Grave. Unsupervised dense information retrieval with con-
trastive learning.Trans. Mach. Learn. Res., 2022, 2022. 2,
5, 6
[12] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch,
Chris Bamford, Devendra Singh Chaplot, Diego de las
Casas, Florian Bressand, Gianna Lengyel, Guillaume Lam-
ple, Lucile Saulnier, L ´elio Renard Lavaud, Marie-Anne
Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,
Thomas Wang, Timoth ´ee Lacroix, and William El Sayed.
Mistral 7b.arXiv preprint arXiv: 2310.06825, 2023. 6
[13] Ehsan Kamalloo, Nandan Thakur, Carlos Lassance,
Xueguang Ma, Jheng-Hong Yang, and Jimmy Lin. Re-sources for brewing beir: Reproducible reference models and
statistical analyses. InSIGIR, pages 1431–1440, 2024. 5
[14] Sunjun Kweon, Yeonsu Kwon, Seonhee Cho, Yohan Jo, and
Edward Choi. Open-wikitable: Dataset for open domain
question answering with complex reasoning over table.arXiv
preprint arXiv:2305.07288, 2023. 1, 2, 5
[15] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman,
Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-
embed: Improved techniques for training llms as generalist
embedding models. InICLR, 2025. 2, 5, 6
[16] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. Towards general text
embeddings with multi-stage contrastive learning.arXiv
preprint arXiv: 2308.03281, 2023. 2, 5, 6
[17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization.arXiv preprint arXiv:1711.05101, 2017. 5
[18] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li,
Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo,
and Rongrong Ji. Video-rag: Visually-aligned retrieval-
augmented long video comprehension.arXiv preprint
arXiv:2411.13093, 2024. 2
[19] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen,
and Jimmy Lin. Unifying multimodal retrieval via document
screenshot embedding. InEMNLP, pages 6492–6505, 2024.
1, 2, 5, 6
[20] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. Chartqa: A benchmark for question an-
swering about charts with visual and logical reasoning. In
ACL Findings, pages 2263–2279, 2022. 1, 2, 5
[21] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar.
Docvqa: A dataset for vqa on document images. InWACV,
pages 2200–2209, 2021. 5
[22] Minesh Mathew, Viraj Bagal, Rub `en Tito, Dimosthenis
Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa.
InWACV, pages 1697–1706, 2022. 5
[23] Marjorie A Oettinger, David G Schatz, Carolyn Gorka, and
David Baltimore. Rag-1 and rag-2, adjacent genes that syner-
gistically activate v (d) j recombination.Science, 248(4962):
1517–1523, 1990. 2
[24] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Repre-
sentation learning with contrastive predictive coding.arXiv
preprint arXiv:1807.03748, 2018. 4
[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual
models from natural language supervision. InICML, pages
8748–8763, 2021. 5, 6
[26] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei
Yin, and Chao Huang. Videorag: Retrieval-augmented gen-
eration with extreme long-context videos.arXiv preprint
arXiv:2502.01549, 2025. 2
[27] Stephen E. Robertson and Hugo Zaragoza. The probabilistic
relevance framework: BM25 and beyond.Found. Trends Inf.
Retr., 3(4):333–389, 2009. 2, 5, 6
[28] Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. Vi-
sualmrc: Machine reading comprehension on document im-
9

ages. InProceedings of the AAAI Conference on Artificial
Intelligence, pages 13878–13888, 2021. 1, 2, 5
[29] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku
Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A
dataset for document visual question answering on multiple
images. InProceedings of the AAAI Conference on Artificial
Intelligence, pages 13636–13645, 2023. 1, 2, 5
[30] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida,
Kuniko Saito, and Jun Suzuki. Vdocrag: Retrieval-
augmented generation over visually-rich documents. In
CVPR, pages 24827–24837, 2025. 2, 5, 6, 8
[31] Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Ri-
jke, Yixing Fan, and Xueqi Cheng. Boosting retrieval-
augmented generation with generation-augmented retrieval:
A co-training approach. InSIGIR, pages 2441–2451, 2025.
1, 2
[32] Jordy Van Landeghem, Rub `en Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Ju-
rkiewicz, Micka ¨el Coustaty, Bertrand Anckaert, Ernest Val-
veny, et al. Document understanding dataset and evaluation
(dude). InICCV, pages 19528–19540, 2023. 1, 2, 5
[33] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao,
Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu
Wei. Text embeddings by weakly-supervised contrastive pre-
training.arXiv preprint arXiv: 2212.03533, 2024. 2, 5, 6
[34] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Ran-
gan Majumder, and Furu Wei. Improving text embeddings
with large language models. InACL, pages 11897–11916,
2024. 2, 5, 6
[35] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shi-
hang Wang, Pengjun Xie, and Feng Zhao. Vidorag: Visual
document retrieval-augmented generation via dynamic itera-
tive reasoning agents.EMNLP, 2025. 5, 7
[36] Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu,
Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping
Luo. Lvlm-ehub: A comprehensive evaluation benchmark
for large vision-language models.IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 2024. 2
[37] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze D Gui, Ziran W
Jiang, Ziyu Jiang, et al. Crag-comprehensive rag benchmark.
NeurIPS, 37:10470–10490, 2024. 2
[38] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui,
Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui
He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye
Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han,
Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun.
Minicpm-v: A GPT-4V level MLLM on your phone.arXiv
preprint arXiv: 2408.01800, 2024. 6
[39] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran,
Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan
Liu, et al. Visrag: Vision-based retrieval-augmented gener-
ation on multi-modality documents. InICLR, 2025. 1, 2,
6
10
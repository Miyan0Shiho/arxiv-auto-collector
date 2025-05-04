# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities

**Authors**: Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang

**Published**: 2025-04-29 13:18:58

**PDF URL**: [http://arxiv.org/pdf/2504.20734v1](http://arxiv.org/pdf/2504.20734v1)

## Abstract
Retrieval-Augmented Generation (RAG) has shown substantial promise in
improving factual accuracy by grounding model responses with external knowledge
relevant to queries. However, most existing RAG approaches are limited to a
text-only corpus, and while recent efforts have extended RAG to other
modalities such as images and videos, they typically operate over a single
modality-specific corpus. In contrast, real-world queries vary widely in the
type of knowledge they require, which a single type of knowledge source cannot
address. To address this, we introduce UniversalRAG, a novel RAG framework
designed to retrieve and integrate knowledge from heterogeneous sources with
diverse modalities and granularities. Specifically, motivated by the
observation that forcing all modalities into a unified representation space
derived from a single combined corpus causes a modality gap, where the
retrieval tends to favor items from the same modality as the query, we propose
a modality-aware routing mechanism that dynamically identifies the most
appropriate modality-specific corpus and performs targeted retrieval within it.
Also, beyond modality, we organize each modality into multiple granularity
levels, enabling fine-tuned retrieval tailored to the complexity and scope of
the query. We validate UniversalRAG on 8 benchmarks spanning multiple
modalities, showing its superiority over modality-specific and unified
baselines.

## Full Text


<!-- PDF content starts -->

UniversalRAG: Retrieval-Augmented Generation over
Multiple Corpora with Diverse Modalities and Granularities
Woongyeong Yeo1∗Kangsan Kim1∗Soyeong Jeong1Jinheon Baek1Sung Ju Hwang1,2
KAIST1DeepAuto.ai2
{wgcyeo, kksan07, starsuzi, jinheon.baek, sungju.hwang}@kaist.ac.kr
Abstract
Retrieval-Augmented Generation (RAG) has
shown substantial promise in improving factual
accuracy by grounding model responses with
external knowledge relevant to queries. How-
ever, most existing RAG approaches are limited
to a text-only corpus, and while recent efforts
have extended RAG to other modalities such as
images and videos, they typically operate over
a single modality-specific corpus. In contrast,
real-world queries vary widely in the type of
knowledge they require, which a single type of
knowledge source cannot address. To address
this, we introduce UniversalRAG, a novel RAG
framework designed to retrieve and integrate
knowledge from heterogeneous sources with
diverse modalities and granularities. Specifi-
cally, motivated by the observation that forc-
ing all modalities into a unified representation
space derived from a single combined corpus
causes a modality gap, where the retrieval tends
to favor items from the same modality as the
query, we propose a modality-aware routing
mechanism that dynamically identifies the most
appropriate modality-specific corpus and per-
forms targeted retrieval within it. Also, be-
yond modality, we organize each modality into
multiple granularity levels, enabling fine-tuned
retrieval tailored to the complexity and scope
of the query. We validate UniversalRAG on
8 benchmarks spanning multiple modalities,
showing its superiority over modality-specific
and unified baselines. Our project page is at
https://universalrag.github.io .
1 Introduction
Recently, we have witnessed the remarkable per-
formance of Large Language Models (LLMs) in
various tasks, such as question answering (OpenAI
et al., 2024; Anil et al., 2023), and their widespread
adoption in various services, such as ChatGPT, to
empower users in everyday life. Yet, LLMs often
*Equal contributiongenerate factually incorrect or misleading informa-
tion, especially on topics they were less or not ex-
posed to during training (e.g., recent events) (Zhang
et al., 2023; Huang et al., 2025). To address this
issue, Retrieval-Augmented Generation (RAG) has
emerged as a promising approach, which allows
the model responses to be grounded in the query-
relevant knowledge retrieved from external knowl-
edge sources, enhancing factual accuracy (Lewis
et al., 2020; Gao et al., 2024; Chen et al., 2024a).
However, despite its effectiveness, existing RAG
approaches are typically designed for a single cor-
pus and modality, limiting their ability to address
user queries that demand different types of knowl-
edge sources. In practice, as illustrated in Figure 1,
user queries vary widely in the type of knowledge
that they require: some are best answered using
text (e.g., surface-level facts and definitions), oth-
ers demand visual understanding from images (e.g.,
spatial relations of objects), and yet others require
temporal reasoning supported by videos (e.g., step-
by-step instructions with dynamic scenes). On the
contrary, the field of RAG primarily originates with
a focus on the textual corpus (Lewis et al., 2020;
Jiang et al., 2023; Yan et al., 2024), and although re-
cent efforts have expanded it to modalities beyond
text (such as images and videos) (Abootorabi et al.,
2025; Riedler and Langer, 2024; Jeong et al., 2025),
existing RAG methods individually are typically
modality- and corpus-specific; therefore, they may
be suboptimal to serve as a universal, one-for-all
framework that can flexibly handle the wide range
of queries, whose knowledge requirements vary.
In this work, we present UniversalRAG, a novel
RAG framework that brings together knowledge
distributed across multiple modality-specific cor-
pora, including text, image, and video sources, and
leverages them to generate grounded responses to
queries in a universal workflow. To operational-
ize this, the most straightforward approach might
be to aggregate all entries from the collected, het-
1arXiv:2504.20734v1  [cs.CL]  29 Apr 2025

(A)RAG with Single Modality 
(C) UniversalRAG (Ours)
Simple Query: 
What is the capital of France?
Text RAG Query: Where was the CEO of Meta born?
Image RAG Query: What does Burj Khalifa look like?
Video RAG Query: How can I replace a bike wheel?
Router
(B)RAG with Single Corpus 
Text Corpus
Image Corpus
Video CorpusNo RetrievalText+Image+Video CorpusRetrieve
Image CorpusRetrieve
 Generate
Text RAG Query: 
Where was the CEO of Meta born?
Generate
Bicycle wheels 
are typically 
designed …Video RAG Query: 
How can I replace a bike wheel?Figure 1: Illustration of (a, b) limitations of existing RAG
methods and (c) the proposed RAG framework, UniversalRAG.
Text
CorpusImage
CorpusVideo
CorpusEmbedding Space
Visualization
Query
Text
Image
VideoFigure 2: t-SNE visualization
of unified embedding space.
0.300.350.40Avg ScoreInternVL-2.5Naïve
Paragraph
DocumentImage
Clip
VideoUnified
UniversalRAG
(Ours)
0.350.400.45Avg ScoreQwen2.5-VLFigure 3: Average scores of the
baselines and UniversalRAG.
erogeneous knowledge corpora, and embed them
into a unified representation space using a multi-
modal encoder (which is typically trained to align
inputs from different modalities if they are semanti-
cally similar). However, despite such alignment ef-
forts, we find that this strategy suffers from modal-
ity gaps, which is the tendency to cluster inputs
based on their modality rather than their semantic
meaning (visualized in Figure 2), a phenomenon
observed similarly in prior work under different
settings (Zhang et al., 2025; Wei et al., 2024). As a
result, retrieval becomes biased toward knowledge
sources that share the same modality as the query,
overlooking relevant content from other modalities.
To address this challenge, instead of relying on a
unified embedding space that forces all modalities
into the shared representation, we take a different
direction: introducing a modality-aware routing
strategy. Specifically, UniversalRAG dynamically
determines the most suitable knowledge source to
retrieve from, based on the modality requirement
of the given query, then routes the retrieval process
to the corresponding modality-specific corpus. It
is worth noting that this strategy not only sidesteps
modality gaps by avoiding direct cross-modal com-
parisons, but also enables seamless integration of
new modalities by extending the routing logic with-
out modifying existing modality-specific retrievers.
Beyond modality, another important dimension
is data granularity (the size or unit of each entry in
the corpus), which plays a critical role in both re-
trieval precision and generation quality (Chen et al.,
2024b; Zhong et al., 2025), since different queries
benefit from different levels of granularity even
within the same modality. This is because overly
fine-grained entries can dilute context, while overly
coarse entries may bundle unrelated information.
For example, a complex analytical question mayrequire long-form documents or full-length videos
to capture sufficient context, while a simple fac-
toid question might be best served with a single
paragraph or short video clip.
To accommodate this aspect, we further break
down each modality into multiple granularity lev-
els, organizing them into distinct corpora: textual
documents are additionally segmented into para-
graphs and stored in a paragraph-level corpus, and
similarly, full-length videos are divided into short
clips and stored, while images are kept intact since
they are inherently piecemeal. Overall, with these
modality- and granularity-aware corpora (including
paragraphs, documents, images, clips, and videos)
in place, as well as an additional no-retrieval op-
tion to efficiently handle straightforward queries
(that require no external knowledge), our Univer-
salRAG dynamically routes each query to the most
relevant knowledge source, ultimately supporting
the diverse information needs of real-world users.
We validate UniversalRAG on 8 benchmarks
with different modalities (Hendrycks et al., 2021;
Rajpurkar et al., 2016; Kwiatkowski et al., 2019;
Yang et al., 2018; Chang et al., 2022; Wang et al.,
2024a; Jeong et al., 2025). UniversalRAG outper-
forms all baselines in an average score, indicating
robust performance across diverse queries. We also
investigate the efficacy of multi-modal and multi-
granular corpora with experimental results.
2 Method
In this section, we present UniversalRAG, a novel
RAG framework that retrieves knowledge from di-
verse corpora spanning multiple modalities and
granularities, conditioned on the given query.
2

2.1 Preliminaries
We begin with preliminaries, introducing LVLMs
and RAG with formal descriptions.
Large Vision Language Models In order to ex-
tend the powerful capabilities of Large Language
Models (LLMs) beyond text and support the un-
derstanding of visual inputs such as images and
videos, Large Vision-Language Models (LVLMs)
have recently been introduced by incorporating vi-
sual encoders into LLMs, enabling them to pro-
cess both textual and visual inputs such as images
and videos. Formally, an LVLM takes as input
a sequence x= [x1, x2, . . . , x n], which may in-
clude both text and visual tokens, and generates
a sequence of output tokens y= [y1, y2, . . . , y m],
denoted as: y=LVLM(x). Nevertheless, despite
their multimodal capacity, LVLMs are still limited
to parametric knowledge and often struggle with
queries requiring detailed or grounded information
beyond what was encoded during pretraining.
Retrieval-Augmented Generation To address
the aforementioned limitations of parametric-only
models, Retrieval-Augmented Generation (RAG)
retrieves query-relevant information from a large
external corpus and incorporates it into the gen-
eration process. Specifically, in the retrieval step,
aRetriever selects the relevant context cfrom
a corpus C, formalized as c=Retriever (q;C),
where c∈ C. In the subsequent generation step, an
LVLM generates a response aconditioned on both
the input query and the retrieved context, denoted
asa=LVLM(q,c). However, most existing RAG
approaches are restricted to retrieving from a single
corpus within a single modality (e.g., image-only),
limiting their ability to handle diverse real-world
queries that often require multimodal information.
Modality Gap in Unified Retrieval Given that
external knowledge in real-world scenarios often
spans multiple modalities—such as text, images,
and videos—we define three modality-specific cor-
pora: the text corpus Ctext={t1, . . . , tn}, the
image corpus Cimage ={i1, . . . , im}, and the
video corpus Cvideo ={v1, . . . , vk}. A com-
mon approach to handling such heterogeneous
data is to unify all items into a shared embed-
ding space using a multimodal encoder, resulting
in a unified corpus Cunified =Ctext∪ C image∪
Cvideo, where each item is represented as a vec-
tor in the shared space (Zhang et al., 2025; Wei
et al., 2024), and retrieval is then performed asc=Retriever (q;Cunified ). However, our experi-
ments reveal a clear modality gap in such unified
spaces—as shown in Figure 2—where queries, be-
ing inherently textual, tend to align more closely
with text corpus items regardless of the actual
modality required. As a result, even when a query
demands visual or temporal understanding, the re-
triever returns text-based content, leading to sub-
optimal or irrelevant responses. This observation
highlights a fundamental limitation of unified re-
trieval strategies and motivates the need to maintain
separate feature spaces for different modalities.
2.2 UniversalRAG
We now turn to introduce UniversalRAG, a novel
framework that dynamically identifies and routes
queries to the most appropriate modality and gran-
ularity of knowledge for retrieval.
Modality-Aware Retrieval To address the
modality gap in retrieval, we maintain separate
embedding spaces for each modality, organizing
the overall corpus into three distinct sub-corpora:
Ctext,Cimage, andCvideo, where each consists of
modality-specific vector representations. We then
introduce a routing module, Router , which dynam-
ically selects the most appropriate modality for
each query. Specifically, given a query q,Router
predicts the query-relevant modality r∈{‘Text’,
‘Image’, ‘Video’}, formalized as r=Router (q).
Once the modality ris determined, a modality-
specific Retriever selects the relevant item cfrom
the corresponding corpus Cr, and the LVLM gener-
ates the final response based on the query and the
retrieved content. However, while this design miti-
gates the modality gap, separating corpora solely
by modality may still fall short, as different queries
can require varying levels of granularity—even
within the same modality.
Granularity-Aware Retrieval To flexibly ac-
commodate the varying information needs of dif-
ferent queries, we extend UniversalRAG to operate
across multiple levels of granularity within each
modality, constructing two corpus levels—fine-
grained and coarse-grained—for both the text and
video modalities. Specifically, while the text cor-
pus is initially organized at the paragraph level,
where each item typically contains knowledge
about a single entity, some complex queries re-
quire reasoning across multiple paragraphs. To
address this, we construct a document-level cor-
pusCdocument ={d1, . . . , dl}, where each dis the
3

vector representation of a document obtained by
concatenating multiple paragraphs and encoding
the resulting text. On the other hand, the original
video corpus consists of full-length videos, which
can often exceed an hour in duration, making it
inefficient to retrieve an entire video when certain
questions can be answered with only a short clip.
Therefore, we segment each full-length video into
multiple fixed-duration clips, constructing a clip-
level corpus Cclip={k1, . . . , kp}, where each k
denotes the representation of trimmed video clip
extracted from the original full-length videos. Note
that since images are inherently fine-grained, we
do not perform additional segmentation for the im-
age corpus and maintain it as is. To this end, the
routing decision rmade by Router falls into one of
six categories: {‘None’, ‘Paragraph’, ‘Document’,
‘Image’, ‘Clip’, ‘Video’}, and the retrieval process
is formalized as follows:
c=

None ifr=‘None’
Retriever (q;Ctext) ifr=‘Paragraph’
Retriever (q;Cdocument )ifr=‘Document’
Retriever (q;Cimage) ifr=‘Image’
Retriever (q;Cclip) ifr=‘Clip’
Retriever (q;Cvideo) ifr=‘Video’
Finally, the LVLM generates the final response a
conditioned on the retrieved content c, which re-
flects the most suitable modality and granularity
determined for the given query q. Furthermore, if
no retrieval is required (i.e., c=None ), the LVLM
directly generates the response based solely on q
without any additional context.
2.3 Router Design within UniversalRAG
Here, we explore two designs for the router, which
is responsible for dynamically selecting the re-
trieval modality and granularity based on the query.
Training-free Router The training-free router
utilizes the inherent knowledge and reasoning abil-
ities of a pretrained LLM to classify queries into
appropriate retrieval types without requiring addi-
tional training. Specifically, given a query q, the
LLM is prompted with a detailed instruction de-
scribing the routing task, accompanied by several
in-context examples, and predicts the most suitable
retrieval type from a set of six predefined options.
Trained Router We further explore training the
router module to enable more accurate routing deci-
sions. However, one key challenge in this strategyis the absence of ground-truth query-label pairs
for optimal corpus selection. To address this, we
construct a training dataset for the router by lever-
aging the modality-specific inductive biases of ex-
isting benchmarks—that is, we assume that each
benchmark is primarily associated with a partic-
ular modality and retrieval granularity. Specif-
ically, for text QA benchmarks, queries from
datasets intended to be answered solely based on
the model’s parametric knowledge are labeled as
‘None’, queries from single-hop RAG benchmarks
as ‘Paragraph’, and those from multi-hop RAG
benchmarks as ‘Document’. Similarly, queries
from image-based RAG benchmarks are labeled
as ‘Image’. For video QA benchmarks, queries
that focus on localized events or specific moments
within a video—such as identifying an action at a
particular timestamp—are labeled as ‘Clip’, while
those requiring comprehension of the full storyline
or broader temporal context are labeled as ‘Video’.
Using this constructed dataset, we train the router
to predict the appropriate retrieval type for a given
query at inference time.
3 Experimental Setup
In this section, we explain the experimental setup,
including datasets, models, evaluation metrics, and
implementation details.
3.1 Datasets
To evaluate the performance of our framework
across diverse modalities, we compile a comprehen-
sive QA benchmark covering six distinct retrieval
settings: no-retrieval, paragraph, document, image,
clip, and video.
QA Datasets For the no-retrieval setting, we uti-
lizeMMLU (Hendrycks et al., 2021), which evalu-
ates a model’s knowledge without requiring exter-
nal sources. For the text retrieval settings, we incor-
porate three benchmarks: SQuAD (Rajpurkar et al.,
2016) and Natural Questions (NQ) (Kwiatkowski
et al., 2019) serve as single-hop RAG benchmarks,
where the retrieval units are paragraphs, while Hot-
potQA (Yang et al., 2018) serves as a multi-hop
RAG benchmark, where the retrieval units are doc-
uments. For the image retrieval setting, we use
a subset of WebQA (Chang et al., 2022), con-
sisting of queries that require grounding in exter-
nal images. Finally, for the video retrieval set-
ting, we use queries from LVBench (Wang et al.,
2024a), VideoRAG-Wiki (Jeong et al., 2025), and
4

Table 1: Results of diverse RAG variants, including UniversalRAG and baselines, on modality-specific benchmarks. Our
methodology, UniversalRAG, represented by the colored cells, includes trained approaches for DistilBERT and T5-Large, while
GPT-4o operates in a train-free manner. Bold indicates the best performance for each metric; underline indicates the second-best
among UniversalRAG approaches. R-L and BERT refer to ROUGE-L and BERTScore, respectively.
Text Image Video
Avg. MMLU SQuAD NQ HotpotQA WebQA LVBench VideoRAG-Wiki VideoRAG-Synth
Models Acc EM F1 EM F1 EM F1 R-L BERT Acc R-L BERT R-L BERTInternVL-2.5-8BNaïve 64.50 7.82 16.86 24.71 38.11 12.92 20.87 40.63 90.30 28.60 15.74 84.20 14.93 85.73 33.76
Paragraph 64.50 20.62 30.97 35.14 47.89 14.45 23.05 35.72 89.13 29.19 14.82 84.08 19.15 86.53 35.59
Document 51.50 6.33 13.72 23.57 32.66 19.71 28.49 28.92 87.45 28.80 13.28 83.75 18.51 86.12 30.24
Image 54.50 7.41 15.74 23.57 32.96 13.11 20.18 46.50 91.32 31.64 17.26 83.79 20.72 87.02 33.43
Clip 53.50 4.58 12.52 13.86 21.82 9.38 16.51 39.53 90.27 35.36 18.76 86.38 27.37 89.34 31.55
Video 59.50 3.77 11.55 14.43 22.98 9.95 16.95 40.08 90.51 33.59 19.23 86.35 28.23 89.45 32.47
Unified 59.00 4.72 12.81 17.00 27.87 9.67 17.08 41.71 90.27 27.23 15.87 83.96 19.03 86.46 31.15
Random 55.50 7.68 16.20 22.71 32.79 12.82 20.37 38.37 89.71 31.15 16.55 84.79 21.02 87.37 32.27
GPT-4o 61.50 18.33 28.09 33.43 46.28 17.80 26.10 45.39 91.10 33.01 14.65 84.11 19.68 86.83 37.56
DistilBERT 62.00 19.14 29.71 33.57 46.45 19.43 28.35 46.40 91.29 35.16 19.23 86.35 28.15 89.44 39.60
T5-Large 63.00 20.49 30.87 35.00 47.78 18.09 26.90 45.47 91.09 34.28 19.18 86.32 27.71 89.33 39.36
Oracle 64.50 20.62 30.97 35.14 47.89 19.71 28.49 46.50 91.32 35.65 18.79 86.38 27.45 89.35 40.34Qwen2.5-VL-7BNaïve 73.00 10.78 19.85 17.29 25.71 18.47 25.47 61.26 94.39 29.38 14.26 83.04 10.52 84.34 38.48
Paragraph 72.00 23.58 34.25 38.43 49.37 19.04 26.54 53.42 92.65 27.13 14.88 83.30 12.62 84.93 39.94
Document 66.50 8.76 15.13 23.14 31.02 20.96 28.78 54.37 92.71 27.23 14.78 83.33 11.39 84.50 36.39
Image 68.50 11.19 18.30 16.14 23.14 16.94 23.01 64.39 94.73 30.17 16.17 83.62 13.35 85.10 37.97
Clip 68.50 10.65 17.66 15.14 22.69 16.46 22.86 62.78 94.38 33.50 18.39 85.04 20.53 87.75 38.81
Video 70.00 11.05 18.07 14.00 21.42 17.42 23.74 63.89 94.54 32.81 19.34 85.64 23.31 88.52 39.36
Unified 71.50 7.95 15.06 12.29 19.81 14.35 21.11 55.64 93.07 30.14 15.00 82.74 11.38 84.16 35.87
Random 72.00 12.67 19.49 20.86 29.23 18.47 25.09 59.67 93.85 28.80 15.96 83.91 15.63 86.01 38.50
GPT-4o 71.50 21.70 30.62 36.57 48.11 20.19 28.00 63.58 94.58 32.42 14.87 83.29 12.69 85.01 42.61
DistilBERT 73.50 21.83 32.52 37.57 48.27 20.96 28.87 64.20 94.70 33.01 19.34 85.64 23.18 88.46 44.34
T5-Large 72.50 23.58 34.12 38.29 49.22 19.52 27.32 63.53 94.55 33.01 19.34 85.62 22.85 88.38 44.01
Oracle 73.00 23.58 34.25 38.43 49.37 20.96 28.78 64.39 94.73 33.20 18.43 85.05 20.70 87.80 44.35Phi-3.5-Vision-InstructNaïve 61.00 9.30 18.32 10.43 18.49 14.26 21.01 54.01 93.01 29.58 15.94 83.64 34.58 90.66 35.16
Paragraph 58.50 22.24 33.38 34.86 46.07 17.03 24.82 59.90 93.65 28.21 17.31 85.02 32.11 89.94 39.61
Document 52.50 6.47 12.95 16.43 24.80 17.80 25.86 57.46 93.18 29.09 14.05 84.18 33.27 90.18 34.95
Image 55.50 8.36 15.20 9.86 15.73 13.68 18.70 63.25 94.13 31.15 15.16 85.02 34.18 90.32 35.24
Clip 54.00 7.68 13.38 11.43 16.48 13.40 18.73 60.22 93.60 35.06 19.50 86.04 36.34 90.97 35.62
Video 53.00 8.09 14.05 9.29 15.09 13.11 17.91 59.90 93.50 32.13 19.33 86.14 36.71 90.95 34.56
Unified 55.00 6.47 14.63 5.86 13.48 11.87 18.46 51.05 92.67 28.50 18.09 84.76 35.78 90.82 32.47
Random 55.50 9.57 16.67 14.00 21.72 15.12 21.15 58.84 93.48 29.77 16.94 85.02 33.88 90.31 35.28
GPT-4o 57.50 20.35 30.18 32.86 44.19 16.84 25.09 62.88 94.11 32.62 16.79 84.95 32.01 89.93 40.48
DistilBERT 57.00 20.62 31.90 33.71 44.87 18.18 26.30 63.39 94.14 34.87 19.33 86.14 36.48 90.91 41.78
T5-Large 58.50 22.37 33.36 34.71 45.94 17.61 25.98 62.69 94.04 34.97 19.33 86.10 36.31 90.87 42.08
Oracle 61.00 22.24 33.38 34.86 46.07 17.80 25.86 63.25 94.13 34.57 19.53 86.04 36.20 90.97 42.46
VideoRAG-Synth (Jeong et al., 2025). Among
them, queries targeting short or localized segments
are categorized as clip-level queries, whereas those
requiring an understanding of the long or entire
video are treated as video-level queries.
Retrieval Corpus To support retrieval across
modalities and granularities, we construct a re-
trieval corpus specific to each modality and gran-
ularity. For paragraph-level retrieval, we use a
Wikipedia paragraph corpus derived from SQuAD
and Natural Questions (Karpukhin et al., 2020). In
the case of document-level retrieval, we follow the
construction method from LongRAG (Jiang et al.,
2024) to build a corpus of aggregated Wikipedia
articles. Regarding image retrieval, we use a re-
trieval corpus consisting of images from the We-
bQA dataset. For video-related retrieval, we de-
fine two separate corpora: the video retrieval cor-
pus consists of full-length YouTube videos from
LVBench and VideoRAG, whereas the clip-level
retrieval corpus comprises trimmed segments ex-
tracted from the same videos. Further details on
dataset construction are provided in Appendix A.
3.2 Models
We compare UniversalRAG against eight differ-
ent baselines as follows: 1) Naïve answers queries
without retrieving external knowledge. 2) Para-graph ,3) Document ,4) Image ,5) Clip , and 6)
Video retrieve information only from their respec-
tive modality-specific corpora. 7) Unified retrieves
the information over a single unified embedding
space of multimodal encoder, InternVideo2 (Wang
et al., 2024b), for all data in different corpora, sim-
ilar to (Zhang et al., 2025; Wei et al., 2024). 8)
Random randomly selects one modality-specific
corpus for retrieval. We also implement three vari-
ants of UniversalRAG, varying in their retriever
components. 9) GPT-4o adopts GPT-4o (OpenAI
et al., 2024) as a training-free router. 10) Distil-
BERT and11) T5-Large use DistilBERT (Sanh
et al., 2019) and T5-Large (Raffel et al., 2020),
respectively, trained on the routing dataset. 12)
Oracle is our ideal setting, in which each query
is routed to the most appropriate modality-specific
corpus, simulating perfect routing.
3.3 Evaluation Metrics
We evaluate the performance of UniversalRAG and
the baselines with the following metrics. For bench-
marks with multiple choice questions, we use Top-
1 Accuracy (Acc) , which shows how many ques-
tions get correct answers. For benchmarks whose
answers are shorter than a few words, we use Ex-
act Match (EM) , which checks whether the pre-
dicted response exactly matches the ground truth,
5

Table 2: Router accuracy and generation performance across
retrieval methods on in- and out-of-domain dataset.
In-Domain Out-Domain
Models Router Acc Avg Score Router Acc Avg Score
Random 16.67 32.27 16.67 29.99
Unified 16.67 31.15 16.67 28.92
GPT-4o 57.23 37.56 69.49 36.85
DistilBERT 66.42 39.60 39.62 32.58
T5-Large 59.99 39.36 47.47 35.27
Ensemble 63.99 39.43 61.55 35.22
No Pa Do Im Cl ViNo
Pa
Do
Im
Cl
Vi0.6 0.2 0.2 0.0 0.0 0.0
0.1 0.8 0.2 0.0 0.0 0.0
0.0 0.5 0.5 0.0 0.0 0.0
0.0 0.1 0.0 0.9 0.0 0.0
0.0 0.4 0.1 0.0 0.4 0.1
0.0 0.1 0.1 0.0 0.3 0.3GPT-4o on In-Domain
No Pa Do Im Cl ViNo
Pa
Do
Im
Cl
Vi0.6 0.2 0.1 0.0 0.0 0.1
0.0 0.9 0.0 0.0 0.0 0.0
0.0 0.1 0.8 0.0 0.0 0.0
0.0 0.0 0.0 1.0 0.0 0.0
0.0 0.0 0.0 0.0 0.5 0.5
0.0 0.0 0.0 0.0 0.8 0.1DistilBERT on In-DomainNo Pa Do Im Cl ViNo
Pa
Do
Im
Cl
Vi0.4 0.4 0.2 0.0 0.0 0.0
0.1 0.8 0.1 0.0 0.0 0.0
0.0 0.0 1.0 0.0 0.0 0.0
0.0 0.1 0.0 0.9 0.0 0.0
0.0 0.0 0.0 0.0 0.9 0.1
0.0 0.0 0.0 0.0 0.9 0.1GPT-4o on Out-Domain
No Pa Do Im Cl ViNo
Pa
Do
Im
Cl
Vi0.0 0.7 0.2 0.0 0.0 0.1
0.0 0.4 0.6 0.1 0.0 0.0
0.4 0.2 0.0 0.0 0.3 0.1
0.0 0.0 0.0 1.0 0.0 0.0
0.0 0.0 0.0 0.0 1.0 0.0
0.0 0.0 0.0 0.0 1.0 0.0DistilBERT on Out-Domain
Figure 4: Confusion matrices of router predictions using
different models on in- and out-of-domain queries.
andF1 Score (F1) , which measures the word-level
overlap between the response and the reference an-
swer. Lastly, for benchmarks whose answers are
longer than a sentence, we use ROUGE-L , which
captures the longest matching sequences between
predicted and ground truth answers (Lin, 2004),
andBERTScore , which measures the semantic
similarity between response and annotation using
contextual embeddings (Zhang et al., 2020).
3.4 Implementation Details
To effectively retrieve information from differ-
ent modalities, we leverage modality-specific en-
coders: bge-large-en-v1.5 (Xiao et al., 2024) as
the text encoder, and InternVideo2 (Wang et al.,
2024b) as the vision encoder. For response gen-
eration, we use a variety of LVLMs, including
InternVL2.5-8B (Chen et al., 2025), Qwen2.5-VL-
7B-Instruct (Bai et al., 2025), and Phi-3.5-Vision-
Instruct (Abdin et al., 2024). For the router module,
the trainable routers are trained for 5 epochs with a
learning rate of 2e-5, selecting the best checkpoint
based on validation accuracy. In the training-free
setting, GPT-4o (OpenAI et al., 2024) is instanti-
ated through a prompt as shown in Figure 6. Fur-
ther details are provided in Appendix B.
4 Experimental Results and Analyses
We now present our results and in-depth analyses.
4.1 Main Results
Here, we present the overall results across diverse
retrieval scenarios spanning multiple modalitiesTable 3: Effect of granularity on the performance of three
models across two benchmarks. Gn denotes Granularity.
HotpotQA LVBench
Models Gn EM F1 Acc
GPT-4o✗ 14.26 22.95 32.32
✓ 17.80 26.10 33.01
DistilBERT✗ 14.55 23.08 33.20
✓ 19.43 28.35 35.16
T5-Large✗ 14.35 23.03 33.20
✓ 18.09 26.90 34.28
and levels of granularity.
Overall Results First of all, Figure 3 illustrates
the average scores of UniversalRAG and baseline
models across eight multimodal benchmarks, and
a detailed breakdown of the results is provided in
Table 1. UniversalRAG consistently outperforms
all baselines in terms of average score, demonstrat-
ing the effectiveness of leveraging multiple modali-
ties through adaptive corpus selection. In contrast
to single-modality corpora that provide limited in-
formation, UniversalRAG dynamically selects the
most relevant modality for each query, enabling
more accurate retrieval and generation.
Interestingly, UniversalRAG significantly out-
performs the Unified baseline, highlighting the ef-
fectiveness of our routing strategy in a realistic,
multi-modal setting. Specifically, the Unified base-
line struggles due to a modality gap in its unified
embedding space, often defaulting to retrieving
only textual data and consequently suffering in per-
formance. UniversalRAG mitigates this issue by us-
ing a router to select a single modality-specific cor-
pus for retrieval, effectively addressing the modal-
ity gap. Given the inherent challenge of construct-
ing a unified embedding space across modalities
without a modality gap, our router-based strategy
offers a promising direction to tackle this issue.
Effectiveness of Router Among the Universal-
RAG models, trained router models achieve better
results compared to the training-free router model
across all experiments with different LVLMs. This
improvement is due to the trained routers being
explicitly optimized for the routing task during
training, leading to superior routing performance.
As a result, UniversalRAG models with trained
routers are better at identifying the most optimal
data source and generating more accurate answers.
Nevertheless, the training-free router still outper-
forms other baseline methods, including the ran-
dom router, demonstrating that zero-shot routing
remains effective within our framework.
6

Table 4: Router accuracy with varying router model size.
Models # params Router Acc
T5-Small 60M 51.16
T5-Base 220M 63.65
T5-Large 770M 59.99
T5-XL 3B 67.50
To further understand the impact of routing on
overall system performance, we analyze the ac-
curacy of each router model and the correspond-
ing overall score. Figure 4 illustrates the confu-
sion matrices for zero-shot and trained router mod-
els. While both routers generally succeed in direct-
ing inputs to the appropriate modality, the trained
router demonstrates superior accuracy compared
to the training-free model. Notably, for the clip
and video modalities, there are a few misrouted
queries, primarily due to ambiguity in separating
two different granularities. Nevertheless, the in-
puts are still correctly routed to the video modality,
highlighting the robustness of the routing mech-
anism. As seen in Table 2, our routing methods
significantly outperform both random and unified
baselines in terms of routing accuracy. This im-
provement in accuracy directly translates to better
overall performance, demonstrating a strong cor-
relation between accurate routing and end-to-end
effectiveness. These results underscore the impor-
tance of correctly routing queries to the appropriate
modality corpus, demonstrating the necessity of a
reliable router for multimodal RAG scenario.
Effectiveness of Multigranularity To further in-
vestigate the effectiveness of incorporating mul-
tiple levels of granularity, we evaluate Universal-
RAG under both coarse- and fine-grained retrieval
settings. In the no-granularity (coarse) setting, a
router classifies queries into four broad modalities:
none, text, image, or video. In the granular (fine-
grained) setting, we further subdivide modalities
for more precise retrieval: text is split into para-
graph and document levels, while video is divided
into clip and full video. For benchmarking, we use
HotpotQA to evaluate document-level reasoning
across multiple entities, and LVBench for clip-level
tasks, as its questions are typically answerable us-
ing short video segments. As shown in Table 3,
UniversalRAG with granularity consistently out-
performs the model without granularity on both
benchmarks across all router models. This high-
lights that supporting different levels of granularity
in text and video corpora improves the performance
of UniversalRAG by enabling the model to retrieve
an appropriate amount of information tailored toeach query. In contrast, models without granular-
ity control apply the same level of granularity to
all queries, which may result in either insufficient
or excessive information retrieval. Therefore, sup-
porting multiple levels of granularity is crucial for
adaptively handling a wide range of user queries.
4.2 Analyses and Discussion
Here, we provide a detailed analysis of the perfor-
mance improvements.
Results on Out-of-Domain Datasets To inves-
tigate the generalizability of our approach, we
evaluate UniversalRAG on five unseen datasets,
with detailed descriptions of each benchmark
provided in Appendix A.2. As shown in Ta-
ble 2, GPT-4o achieves the highest routing ac-
curacy, even surpassing its in-domain perfor-
mance—demonstrating strong generalization ca-
pabilities. However, trained routers underperform
on out-of-domain data, demonstrating routers are
overfitted to the training data, mainly due to the in-
sufficient diversity of queries in training data. Fig-
ure 4 further highlights the performance trade-off
between in-domain and out-domain datasets. Bene-
fiting from its robust routing, GPT-4o also achieves
the highest average QA score, outperforming both
trained routers and baseline models.
As a solution to the performance trade-off be-
tween two settings, we introduce an ensemble
router using both trained and train-free routers.
Specifically, a routing result from the trained router
is selected if its confidence score is high enough;
otherwise, a response from the train-free router
is leveraged. This strategy enables exploiting the
trained router for queries that have characteristics
similar to in-domain dataset, while relying on gen-
eralized routing ability of the train-free router for
unfamiliar or out-of-domain queries. As shown in
Table 2, UniversalRAG with the ensemble router
demonstrates better performance in both the in- and
out-of-domain benchmarks.
Analysis on Router Size To assess the impact of
router size on routing accuracy, we evaluate Uni-
versalRAG with trained routers of varying model
sizes. Specifically, we train four variants of the T5
model with different parameter counts and measure
router accuracy using InternVL2.5 as the genera-
tor. As shown in Table 4, router accuracy varies
substantially with model size, indicating that larger
models are more effective at making accurate rout-
ing decisions across modalities and granularities.
7

1B 2B 4B 8B
Model Size25303540 Average Score
Naïve
ParagraphDocument
ImageClip
VideoUnified
GPT-4oDistilBERT
T5-LargeFigure 5: Generation performance with varying generation
model (InternVL2.5) size.
Analysis with Different Model Sizes To see
how the performance of UniversalRAG scales with
LVLM size, we evaluate our models and baselines
with different sizes of InternVL2.5 models, as re-
ported in Figure 5. Across all model sizes, Uni-
versalRAG scores consistently increase and outper-
form other baselines. This indicates the scalability
of UniversalRAG and implies that its performance
could be enhanced by employing larger LVLMs.
Case Study We present the case studies of Uni-
versalRAG in Appendix D.
5 Related Work
Large Vision Language Models Building on the
powerful performance of Large Language Models
(LLMs), researchers have made efforts to enable
LLMs to understand visual information. Liu et al.
(2023) pioneered Large Vision Language Models
(LVLMs) by employing a CLIP-based (Radford
et al., 2021) image encoder that allows the language
model to understand the input image within its tex-
tual feature space. Following this, various image
understanding language models have been intro-
duced, each using different vision encoders over
LLMs (Bai et al., 2023; Chen et al., 2024c; Liu
et al., 2024). As image understanding performance
has become robust, several studies have extended
these methods to video data, which can be viewed
as a sequence of image frames (Li et al., 2024a;
Chen et al., 2025; Bai et al., 2025). Thanks to larger
training datasets and improved model structures,
current LVLMs show strong image and video un-
derstanding abilities, as demonstrated by multiple
benchmark evaluations (Yue et al., 2024; Mathew
et al., 2021; Li et al., 2024b; Fu et al., 2024). How-
ever, standalone LVLMs often suffer from halluci-
nation mainly due to the limited knowledge bound-
ary inherited from their base language models.Retrieval-Augmented Generation Retrieval-
Augmented Generation (RAG) can address the
aforementioned challenges by incorporating exter-
nal knowledge when generating answers; however,
conventional RAG approaches rely solely on text
data, while recent studies have begun to explore
RAG over diverse multimodal corpora, highlight-
ing its significant potential beyond text-only set-
tings. Specifically, image-based RAG (Chen et al.,
2022; Riedler and Langer, 2024) was the first at-
tempt at multimodal RAG, which retrieves and
uses visual information to answer queries. Further-
more, Jeong et al. (2025) recently extends RAG
to video, capturing both visual and temporal ele-
ments for process-related questions. Despite these
advances, most existing methods only consider a
single modality corpus, which is impractical given
that real-world queries could require information
from any modality. Therefore, it is crucial to lever-
age all available data to generate the best possi-
ble answer, rather than restricting the model to
a limited modality. More recent approaches (Cui
et al., 2024; Liu et al., 2025a) support retrieval from
multimodal corpora, but typically retrieve from all
available modalities and decide what to use only
after retrieval—or even after generation—which is
inefficient and fails to adapt retrieval to the specific
needs of the query.
Handling diverse queries requires an RAG ap-
proach that adapts to the specific context and query,
instead of using a single fixed method. One promis-
ing approach is to route queries according to their
predefined complexity levels (Jeong et al., 2024;
Tang et al., 2025; Islam et al., 2024), categorizing
them as requiring no retrieval, single-step retrieval,
or multi-step retrieval, to balance performance and
latency. Another strategy leverages model con-
fidence (Ding et al., 2024; Yao et al., 2024), re-
trieving external information only when the model
confidence is low, therefore efficiently allocating
resources to challenging queries. Although adap-
tive retrieval has become central to RAG, existing
benchmarks (Zhang et al., 2024; Li et al., 2024c)
primarily evaluate text-only systems, leaving open
the question of how to adapt retrieval across multi-
ple modalities. In real-world scenarios, queries ben-
efit from different data types, making it essential to
identify the most suitable modality for retrieval in
a mixed-modality corpus.
Retrieval Granularity The size of indexing a
corpus, retrieval granularity, is a key design choice
8

in retrieval, as it significantly impacts both the
performance and efficiency of RAG. Chen et al.
(2024b) discovered that retrieval from a corpus
indexed in propositions outperforms sentence- or
passage-level retrieval performance. Recent stud-
ies (Liu et al., 2025b; Zhong et al., 2025) also
showed that considering multiple granularities
achieves better retrieval performance. Likewise,
granularity-aware text-to-video retrieval was stud-
ied to find not just a full video but a specific clip
related to the query from a video corpus (Chen
et al., 2023). Therefore, in multimodal corpora, it
is not sufficient to select the appropriate modality
alone; the system should also identify the optimal
level of granularity for retrieval.
6 Conclusion
In this paper, we propose UniversalRAG, a novel
RAG framework designed to retrieve from corpora
with diverse modalities and granularities. Through
a modality- and granularity-aware routing mecha-
nism, UniversalRAG dynamically selects the most
suitable knowledge source for each query, effec-
tively addressing the limitations posed by modality
gaps and fixed-granularity retrieval. Extensive eval-
uations across 8 benchmarks demonstrate that Uni-
versalRAG consistently outperforms both modality-
specific and unified baselines, showcasing robust
performance across diverse modalities. Further-
more, our analyses highlight the importance of fine-
grained retrieval and the complementary strengths
of train-free and trained routers. These findings
demonstrate the potential of UniversalRAG as an
adaptive solution for grounding LVLMs with het-
erogeneous external knowledge, opening new di-
rections for more reliable multimodal reasoning
and modality-aware information integration.
References
Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach,
Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat
Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck,
Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav
Chaudhary, Dong Chen, Dongdong Chen, and 110
others. 2024. Phi-3 technical report: A highly capa-
ble language model locally on your phone. Preprint ,
arXiv:2404.14219.
Mohammad Mahdi Abootorabi, Amirhosein Zobeiri,
Mahdi Dehghani, Mohammadali Mohammadkhani,
Bardia Mohammadi, Omid Ghahroodi, Mahdieh So-
leymani Baghshah, and Ehsaneddin Asgari. 2025.Ask in any modality: A comprehensive sur-
vey on multimodal retrieval-augmented generation.
Preprint , arXiv:2502.08826.
Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-
Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan
Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Mil-
lican, David Silver, Slav Petrov, Melvin Johnson,
Ioannis Antonoglou, Julian Schrittwieser, Amelia
Glaese, Jilin Chen, Emily Pitler, Timothy P. Lilli-
crap, and 33 others. 2023. Gemini: A family of
highly capable multimodal models. arXiv preprint
arXiv:2312.11805 .
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang,
Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou,
and Jingren Zhou. 2023. Qwen-vl: A versatile vision-
language model for understanding, localization, text
reading, and beyond. Preprint , arXiv:2308.12966.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shi-
jie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu,
Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei
Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 oth-
ers. 2025. Qwen2.5-vl technical report. Preprint ,
arXiv:2502.13923.
Valeriia Bolotova-Baranova, Vladislav Blinov, Sofya
Filippova, Falk Scholer, and Mark Sanderson. 2023.
WikiHowQA: A comprehensive benchmark for multi-
document non-factoid question answering. In Pro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 5291–5314, Toronto, Canada. Associ-
ation for Computational Linguistics.
Yingshan Chang, Guihong Cao, Mridu Narang, Jian-
feng Gao, Hisami Suzuki, and Yonatan Bisk. 2022.
Webqa: Multihop and multimodal QA. In IEEE/CVF
Conference on Computer Vision and Pattern Recog-
nition, CVPR 2022, New Orleans, LA, USA, June
18-24, 2022 , pages 16474–16483. IEEE.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024a. Benchmarking large language models in
retrieval-augmented generation. In Thirty-Eighth
AAAI Conference on Artificial Intelligence, AAAI
2024, Thirty-Sixth Conference on Innovative Applica-
tions of Artificial Intelligence, IAAI 2024, Fourteenth
Symposium on Educational Advances in Artificial
Intelligence, EAAI 2014, February 20-27, 2024, Van-
couver, Canada , pages 17754–17762. AAAI Press.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024b. Dense X retrieval: What retrieval
granularity should we use? In Proceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024 , pages 15159–15177.
Association for Computational Linguistics.
Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and
William W. Cohen. 2022. Murag: Multimodal
9

retrieval-augmented generator for open question an-
swering over images and text. In Proceedings of
the 2022 Conference on Empirical Methods in Natu-
ral Language Processing, EMNLP 2022, Abu Dhabi,
United Arab Emirates, December 7-11, 2022 , pages
5558–5570. Association for Computational Linguis-
tics.
Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu,
Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye,
Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang,
Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo,
Jiahao Wang, Tan Jiang, Bo Wang, and 23 others.
2025. Expanding performance boundaries of open-
source multimodal models with model, data, and
test-time scaling. Preprint , arXiv:2412.05271.
Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo
Chen, Sen Xing, Muyan Zhong, Qinglong Zhang,
Xizhou Zhu, Lewei Lu, and 1 others. 2024c. Internvl:
Scaling up vision foundation models and aligning
for generic visual-linguistic tasks. In Proceedings of
the IEEE/CVF conference on computer vision and
pattern recognition , pages 24185–24198.
Zhiguo Chen, Xun Jiang, Xing Xu, Zuo Cao, Yijun
Mo, and Heng Tao Shen. 2023. Joint searching and
grounding: Multi-granularity video content retrieval.
InProceedings of the 31st ACM International Confer-
ence on Multimedia, MM 2023, Ottawa, ON, Canada,
29 October 2023- 3 November 2023 , pages 975–983.
ACM.
Wanqing Cui, Keping Bi, Jiafeng Guo, and Xueqi
Cheng. 2024. MORE: Multi-mOdal REtrieval aug-
mented generative commonsense reasoning. In Find-
ings of the Association for Computational Linguistics:
ACL 2024 , pages 1178–1192, Bangkok, Thailand. As-
sociation for Computational Linguistics.
Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen,
and Xueqi Cheng. 2024. Retrieve only when it
needs: Adaptive retrieval augmentation for hallucina-
tion mitigation in large language models. Preprint ,
arXiv:2402.10612.
Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li,
Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu
Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen,
Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu,
Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing
Sun. 2024. Video-mme: The first-ever comprehen-
sive evaluation benchmark of multi-modal llms in
video analysis. Preprint , arXiv:2405.21075.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Song, and Jacob Stein-
hardt. 2021. Measuring massive multitask language
understanding. In 9th International Conference onLearning Representations, ICLR 2021, Virtual Event,
Austria, May 3-7, 2021 .
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Trans. Inf. Syst. , 43(2).
Shayekh Bin Islam, Md Asib Rahman, K. S. M. Toza-
mmel Hossain, Enamul Hoque, Shafiq Joty, and
Md. Rizwan Parvez. 2024. Open-rag: Enhanced re-
trieval augmented reasoning with open-source large
language models. In Findings of the Association for
Computational Linguistics: EMNLP 2024, Miami,
Florida, USA, November 12-16, 2024 , pages 14231–
14244. Association for Computational Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. In Proceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024 , pages 7036–7050. Association for Com-
putational Linguistics.
Soyeong Jeong, Kangsan Kim, Jinheon Baek, and
Sung Ju Hwang. 2025. Videorag: Retrieval-
augmented generation over video corpus. Preprint ,
arXiv:2501.05874.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023, Singapore, Decem-
ber 6-10, 2023 , pages 7969–7992. Association for
Computational Linguistics.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024.
Longrag: Enhancing retrieval-augmented generation
with long-context llms. Preprint , arXiv:2406.15319.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781,
Online. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
10

Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral Questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Advances in Neural Infor-
mation Processing Systems , volume 33, pages 9459–
9474.
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng
Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang,
Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a.
Llava-onevision: Easy visual task transfer. Preprint ,
arXiv:2408.03326.
Kuan Li, Liwen Zhang, Yong Jiang, Pengjun Xie,
Fei Huang, Shuai Wang, and Minhao Cheng. 2025.
LaRA: Benchmarking retrieval-augmented genera-
tion and long-context llms – no silver bullet for lc or
rag routing. Preprint , arXiv:2502.09977.
Kunchang Li, Yali Wang, Yinan He, Yizhuo Li,
Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo
Chen, Ping Lou, Limin Wang, and Yu Qiao. 2024b.
Mvbench: A comprehensive multi-modal video un-
derstanding benchmark. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, CVPR
2024, Seattle, WA, USA, June 16-22, 2024 , pages
22195–22206. IEEE.
Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang,
Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao
Zheng, Pengjun Xie, Philip S. Yu, Fei Huang, and
Jingren Zhou. 2024c. Benchmarking multimodal
retrieval augmented generation with dynamic vqa
dataset and self-adaptive planning agent. Preprint ,
arXiv:2411.02937.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. In Text Summariza-
tion Branches Out , pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Stephanie Lin, Jacob Hilton, and Owain Evans. 2022.
TruthfulQA: Measuring how models mimic human
falsehoods. In Proceedings of the 60th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 3214–3252, Dublin,
Ireland. Association for Computational Linguistics.
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae
Lee. 2024. Improved baselines with visual instruc-
tion tuning. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, CVPR 2024, Seat-
tle, WA, USA, June 16-22, 2024 , pages 26286–26296.
IEEE.Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning. In Advances in
Neural Information Processing Systems 36: Annual
Conference on Neural Information Processing Sys-
tems 2023, NeurIPS 2023, New Orleans, LA, USA,
December 10 - 16, 2023 .
Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan
Meng, Ding Wang, and Jun Ma. 2025a. Hm-rag:
Hierarchical multi-agent multimodal retrieval aug-
mented generation. Preprint , arXiv:2504.12330.
Zuhong Liu, Charles-Elie Simon, and Fabien Cas-
pani. 2025b. Passage segmentation of docu-
ments for extractive question answering. Preprint ,
arXiv:2501.09940.
Minesh Mathew, Dimosthenis Karatzas, and C. V . Jawa-
har. 2021. Docvqa: A dataset for VQA on document
images. In IEEE Winter Conference on Applications
of Computer Vision, WACV 2021, Waikoloa, HI, USA,
January 3-8, 2021 , pages 2199–2208. IEEE.
Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac,
Makarand Tapaswi, Ivan Laptev, and Josef Sivic.
2019. Howto100m: Learning a text-video embed-
ding by watching hundred million narrated video
clips. In Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV) .
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o
system card. Preprint , arXiv:2410.21276.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021. Learn-
ing transferable visual models from natural language
supervision. In Proceedings of the 38th International
Conference on Machine Learning, ICML 2021, 18-24
July 2021, Virtual Event , volume 139 of Proceedings
of Machine Learning Research , pages 8748–8763.
PMLR.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J. Liu. 2020. Exploring the limits
of transfer learning with a unified text-to-text trans-
former. J. Mach. Learn. Res. , 21:140:1–140:67.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. In Proceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing , pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
Ruchit Rawal, Khalid Saifullah, Miquel Farré, Ronen
Basri, David Jacobs, Gowthami Somepalli, and Tom
Goldstein. 2024. CinePile: A long video ques-
tion answering dataset and benchmark. Preprint ,
arXiv:2405.08813.
11

Monica Riedler and Stefan Langer. 2024. Beyond text:
Optimizing rag with multimodal inputs for industrial
applications. Preprint , arXiv:2410.21943.
Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. 2019. DistilBERT, a distilled version
of BERT: smaller, faster, cheaper and lighter. In
NeurIPS 2019 EMC2Workshop .
Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and
Sihong Xie. 2025. MBA-RAG: a bandit approach
for adaptive retrieval-augmented generation through
question complexity. In Proceedings of the 31st In-
ternational Conference on Computational Linguis-
tics, COLING 2025, Abu Dhabi, UAE, January 19-24,
2025 , pages 3248–3254. Association for Computa-
tional Linguistics.
Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng,
Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang,
Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang.
2024a. Lvbench: An extreme long video understand-
ing benchmark. Preprint , arXiv:2406.08035.
Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yi-
nan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun
Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Ji-
lan Xu, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali
Wang, and Limin Wang. 2024b. Internvideo2: Scal-
ing foundation models for multimodal video under-
standing. In Computer Vision - ECCV 2024 - 18th
European Conference, Milan, Italy, September 29-
October 4, 2024, Proceedings, Part LXXXV , volume
15143 of Lecture Notes in Computer Science , pages
396–416. Springer.
Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu,
Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen.
2024. Uniir: Training and benchmarking univer-
sal multimodal information retrievers. In Computer
Vision - ECCV 2024 - 18th European Conference,
Milan, Italy, September 29-October 4, 2024, Pro-
ceedings, Part LXXXVII , volume 15145 of Lecture
Notes in Computer Science , pages 387–404. Springer.
Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, and
Wenya Wang. 2025. Visual-rag: Benchmark-
ing text-to-image retrieval augmented generation
for visual knowledge intensive queries. Preprint ,
arXiv:2502.16636.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , SIGIR ’24, page 641–649, New
York, NY , USA. Association for Computing Machin-
ery.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
Preprint , arXiv:2401.15884.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao,
Linmei Hu, Weichuan Liu, Lei Hou, and Juanzi Li.
2024. Seakr: Self-aware knowledge retrieval for
adaptive retrieval augmented generation. Preprint ,
arXiv:2406.19215.
Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang,
Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu
Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao
Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan
Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and
3 others. 2024. MMMU: A massive multi-discipline
multimodal understanding and reasoning benchmark
for expert AGI. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, CVPR 2024,
Seattle, WA, USA, June 16-22, 2024 , pages 9556–
9567. IEEE.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. BERTScore:
Evaluating text generation with BERT. In 8th Inter-
national Conference on Learning Representations,
ICLR 2020, Addis Ababa, Ethiopia, April 26-30,
2020 .
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi
Dai, Dingkun Long, Pengjun Xie, Meishan Zhang,
Wenjie Li, and Min Zhang. 2025. GME: Improving
universal multimodal retrieval by multimodal llms.
Preprint , arXiv:2412.16855.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei
Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song
in the ai ocean: A survey on hallucination in large
language models. Preprint , arXiv:2309.01219.
Zihan Zhang, Meng Fang, and Ling Chen. 2024. Re-
trievalqa: Assessing adaptive retrieval-augmented
generation for short-form open-domain question an-
swering. In Findings of the Association for Compu-
tational Linguistics, ACL 2024, Bangkok, Thailand
and virtual meeting, August 11-16, 2024 , pages 6963–
6975. Association for Computational Linguistics.
Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang,
and Zengchang Qin. 2025. Mix-of-granularity:
Optimize the chunking granularity for retrieval-
augmented generation. In Proceedings of the 31st
International Conference on Computational Linguis-
tics, COLING 2025, Abu Dhabi, UAE, January 19-24,
2025 , pages 5756–5774. Association for Computa-
tional Linguistics.
12

A Additional Details on Dataset
Table 5 provides an overview of all datasets and
their corresponding data corpora used in our exper-
iments, including the target modality type as well
as the size of the queries and corpora. We divide
each dataset into a 3:7 ratio for training and testing.
A detailed explanation of each dataset is provided
below.
A.1 In-Domain Dataset
MMLU As a dataset comprising queries that can
be answered without the need for retrieval, we
use MMLU (Hendrycks et al., 2021), a bench-
mark that spans a wide range of tasks, including
problem-solving abilities (e.g., elementary math-
ematics, computer science) and world knowledge
(e.g., law, world religions). Specifically, we use
questions from all tasks in the development split.
SQuAD SQuAD v1.1 (Rajpurkar et al., 2016) is
a benchmark dataset consisting of questions gener-
ated by crowdworkers based on a set of Wikipedia
articles. Each question is answerable given the ap-
propriate context paragraph. From the dataset’s
100,000+ QA pairs, we randomly sample 1,060
pairs of dev split. For context retrieval, we utilize
the full provided Wikipedia corpus, segmenting
each article into paragraphs of at most 100 words.
Natural Questions (NQ) We also use Natural
Questions (Kwiatkowski et al., 2019), a question
answering dataset consisting of real user queries
issued to the Google search engine, with answers
annotated based on supporting Wikipedia articles.
We randomly sample 1,000 QA pairs of dev split,
and formulate the text corpus in same setting as
SQuAD, segmenting the Wikipedia corpus into
paragraphs of at most 100 words.
HotpotQA HotpotQA (Yang et al., 2018) is a
Wikipedia-based QA benchmark, but contains com-
plex queries that are annotated to reason over mul-
tiple articles. We utilize 1,492 randomly sampled
QA pairs of test split. As it requires multi-hop rea-
soning over multiple documents, we formulate the
text corpus by grouping multiple related documents
following LongRAG (Jiang et al., 2024), that can
be longer than 4K tokens.
WebQA WebQA (Chang et al., 2022) is a bench-
mark designed to evaluate the ability of LLMs to
reason over multiple sources of information, in-
cluding both text and images, in an open-domainsetting. As the dataset is originally constructed
with question-specific retrieval sources that com-
bine text and images, we extract a subset of ques-
tions that require only a single image for retrieval.
We then further filter these using GPT-4o with the
prompt shown in Figure 7 to make sure questions
are not grounded to certain image, resulting in a
final set of 2,000 QA pairs.
LVBench LVBench (Wang et al., 2024a) is a
benchmark developed for long video understand-
ing, featuring questions generated by annotators
based on YouTube videos with an average duration
of over one hour. Since the benchmark was origi-
nally designed for non-RAG tasks, we rephrased
the original text-video interleaved queries into a
text-only format to align with our experimental
setup using GPT-4o, with video metadata and a
prompt (Figure 8). Each query is associated with
a specific video and a corresponding time range.
Notably, the majority of queries are annotated
with timestamps spanning less than five minutes,
thereby focusing on short segments within the
longer videos. For training, we use these short-
timestamp queries as a clip-level dataset.
VideoRAG We also utilize VideoRAG-Wiki and
VideoRAG-Synth benchmarks, introduced in Vide-
oRAG (Jeong et al., 2025), which are designed to
evaluate RAG over video corpus. These bench-
marks are built on the HowTo100M (Miech et al.,
2019) corpus—a large-scale collection of instruc-
tional YouTube videos—with queries sourced from
WikiHowQA (Bolotova-Baranova et al., 2023) and
synthetically generated QA pairs based on the
videos. Since they lack timestamp annotations,
we employ GPT-4o to identify video-level queries
that are better answered through full video retrieval
rather than short segments from the ground-truth
video, which are then used as a video-level dataset
for training the router.
A.2 Out-of-Domain Dataset
Unlike the in-domain datasets, the out-of-domain
datasets are used solely for evaluation to assess
the generalizability of our routing approach, and
consist only of test splits.
TruthfulQA TruthfulQA (Lin et al., 2022) in-
cludes general knowledge questions designed to
test whether LLMs can avoid common false belief
or misconception, on diverse categories, includ-
ing health, law, and politics. We use the multiple-
13

Table 5: Dataset Summary. Avg. corpus length is the mean token count for text corpora and the mean duration for video corpora.
Dataset Gold Retrieval # Queries Corpus Size Avg. Corpus Length
In-Domain Datasets
MMLU None 285 - -
SQuAD Paragraph 1,060 1.19M 100 tokens
Natural Questions Paragraph 1,000 850k 100 tokens
HotpotQA Document 1,492 509k 693 tokens
WebQA Image 2,000 20k -
LVBench Clip/Video 1,376 94 3,941s
VideoRAG-Wiki Clip/Video 3749k 378sVideoRAG-Synth Clip/Video 374
Out-of-Domain Datasets
TruthfulQA None 790 - -
TriviaQA Paragraph 661 661k 100 tokens
LaRA Document 112 34 28k tokens
Visual-RAG Image 374 2k -
CinePile Clip/Video 1,440 144 158s
choice version of the dataset, which includes only
a single correct answer per question.
TriviaQA TriviaQA (Joshi et al., 2017) is a read-
ing comprehension dataset consisting of trivia ques-
tions paired with evidence texts sourced from
Wikipedia and the web. To distinguish between
queries that require text retrieval and those that do
not, we categorize each query based on whether
GPT-4o can produce an exact-match answer with-
out access to external text. We randomly sample
QA pairs from the dev split. Following the pre-
processing strategies used in SQuAD and NQ, all
supporting evidence documents are segmented into
paragraphs of no more than 100 words.
LaRA We also utilize LaRA (Li et al., 2025),
which is designed for understanding long-context
documents such as academic papers and novels.
For our use case, we focus on a subset of these
documents, specifically excluding queries on the
‘comparison’ task, as our goal is RAG, not reading
comprehension. Additionally, we slightly reformat
the remaining queries to align with a general QA
format. Given the length of the source material,
each document is treated as a single entry in the
document-level corpus.
Visual-RAG Visual-RAG (Wu et al., 2025) is
a question-answering benchmark designed for vi-
sual knowledge-intensive questions, specifically
tailored for text-to-image retrieval tasks. We utilize
the full set of provided queries but sample five im-
ages per category to construct the image retrieval
pool, ensuring efficient text-to-image retrieval.
CinePile CinePile (Rawal et al., 2024) is a long-
video question-answering benchmark that featuresquestions based on movie clips from YouTube.
Since the benchmark was originally designed for
video understanding tasks rather than RAG, we re-
formulate each query using the same procedure as
LVBench. For each of the 144 available videos,
we randomly select 10 questions from the test
split. Since CinePile does not provide granular-
ity annotations, we classify the questions into two
categories—clip-level and full-video-level gran-
ularity—using GPT-4o, following the same ap-
proach used in VideoRAG.
B Additional Implementation Details
To effectively leverage both visual and textual in-
formation for visual element retrieval, we employ
an ensemble approach that combines visual and tex-
tual similarity scores with a weighting ratio of 0.8
for visual information. The textual information con-
sists of image captions for images and scripts for
videos. During the generation stage, we use only
the top-1 retrieved result, selected based on the co-
sine similarity of the corresponding embeddings.
Moreover, we uniformly sample 32 frames per
video for both the retrieval and generation stages.
Trainable routers are trained for 5 epochs with a
learning rate of 2e-5, with the best-performing state
selected based on validation accuracy.
C Additional Experimental Results
C.1 Routing Results per Dataset
We present routing results of three routers for each
dataset in Table 6. On in-domain datasets, GPT-4o
often struggles to distinguish between Paragraph
and Document RAG queries, and misroutes Vide-
oRAG queries to textual corpus. Meanwhile, two
14

Table 6: Routing results across in-domain and out-domain dataset.
In-domain Dataset Out-domain Dataset
Text Image Video Text Image Video
MMLU SQuAD NQ HotpotQA WebQA LVBench VRAG-Wiki VRAG-Synth TruthfulQA TriviaQA LaRA Vis-RAG CinePile
Models 200 742 700 1045 1392 829 374 374 790 661 112 374 1440gpt-4oNone 117 44 57 38 25 3 8 27 374 121 0 0 0
Paragraph 39 512 588 505 102 17 304 271 205 509 4 18 0
Document 44 185 39 502 44 34 52 53 206 27 108 0 6
Image 0 1 5 0 1210 44 1 9 2 4 0 356 1
Clip 0 0 6 0 0 622 0 0 3 0 0 0 1354
Video 0 0 5 0 11 109 9 14 0 0 0 0 79DistilBERTNone 120 2 1 1 0 0 0 0 2 1 42 0 0
Paragraph 49 679 669 150 30 1 0 6 629 274 21 0 1
Document 11 32 12 866 12 3 0 0 53 338 2 2 1
Image 0 6 11 17 1351 7 0 0 12 32 4 371 1
Clip 5 0 1 8 5 818 0 2 3 7 34 0 1436
Video 15 23 6 3 2 0 374 366 91 9 9 1 1T5-LargeNone 110 4 1 0 1 0 1 1 21 4 30 0 0
Paragraph 79 731 698 461 145 13 5 15 709 558 74 0 13
Document 2 3 0 571 6 15 0 0 35 94 0 0 2
Image 0 1 0 9 1234 15 0 0 1 2 0 374 4
Clip 0 0 1 2 13 784 0 3 2 1 1 0 1420
Video 9 3 0 2 1 2 368 355 22 2 7 0 1
Table 7: Detailed results of UniversalRAG and baselines on
out-domain dataset.
Text Image Video
Avg. TruthfulQA TriviaQA LaRA Vis-RAG Cinepile
Models Acc EM F1 R-L BERT R-L BERT Acc
Naïve 64.68 49.47 57.92 23.15 87.62 6.24 80.98 30.76 33.88
Paragraph 58.73 54.61 65.14 20.23 86.48 4.74 80.77 30.07 33.88
Document 28.73 39.94 44.73 25.18 86.83 4.34 81.14 32.64 26.68
Image 57.85 45.23 52.50 21.40 87.09 7.31 82.32 34.03 33.35
Clip 51.01 31.62 42.40 19.64 87.50 6.92 81.32 35.63 29.59
Video 47.34 33.59 43.82 19.89 87.19 70.4 81.42 37.43 29.47
Unified 52.15 35.70 45.01 21.28 86.83 4.31 80.47 30.76 28.92
Random 51.27 42.66 51.51 21.81 87.27 5.81 81.09 32.43 29.99
GPT-4o 55.19 54.01 64.05 24.93 88.42 7.17 82.26 35.56 36.85
DistilBERT 57.85 42.51 51.01 20.96 87.26 7.34 82.32 35.63 32.58
T5-Large 57.85 50.08 60.16 20.63 86.73 7.31 82.32 35.49 35.27
Oracle 64.68 55.52 64.85 25.18 86.83 7.31 82.32 37.71 38.26
trained routers show strong classification perfor-
mance across all in-domain datasets. In out-of-
domain datasets, GPT-4o generalizes well for most
datasets, except for image-based RAG queries. In
contrast, trained routers fail to classify the appro-
priate granularity needed for each query. This is
mainly due to the limited diversity of training data,
which causes overfitting to seen examples.
C.2 Detailed Results on Out-domain Dataset
QA evaluation results of UniversalRAG models
and the baselines for each out-of-domain dataset
are shown in Table 7. UniversalRAG models out-
perform the baselines in general. GPT-4o demon-
strates robust performance across all datasets, pri-
marily due to the outstanding generalization capa-
bility of the router on unseen queries, as discussed
in Section C.1. However, trained routers show de-
graded performance compared to the results on
in-domain datasets, since their routers often mis-
classify unseen queries.
D Qualitative Results
We present case studies to demonstrate the effec-
tiveness of UniversalRAG. Table 8 compares the
results of various RAG approaches, including tra-
ditional single-modality methods and Universal-RAG, on queries from the WebQA dataset. Tra-
ditional approaches such as TextRAG and Vide-
oRAG fail to generate accurate answers—TextRAG
retrieves passages lacking relevant visual details,
while VideoRAG is better suited for temporal rea-
soning tasks. In contrast, UniversalRAG correctly
routes the query to the image modality, recogniz-
ing that visual information about color is necessary,
and successfully generates the correct response.
This highlights the advantage of modality-aware
routing in leveraging the appropriate data from the
correct modality corpus, demonstrating Universal-
RAG’s ability to adaptively select the most infor-
mative modality for accurate answer generation.
In addition to modality routing, we observe
that UniversalRAG also benefits from retrieving
information at the appropriate granularity. Ta-
ble 9 shows results from HotpotQA, where the
query requires complex reasoning over multiple
text sources. While paragraph-level granularity
fails to provide sufficient context for reasoning,
UniversalRAG routes the query to the document-
level corpus, to retrieve all the textual informa-
tion necessary for accurate reasoning. Similarly,
for video queries, Table 10 shows results from
LVBench, on the query that requires only a short
segment of the full long video to answer. While
full-video-level retrieval includes irrelevant content
and uniformly sampled 32 frames fail to capture the
necessary information, clip-level retrieval focuses
on smaller, more relevant segments of the video to
ensure that only the most pertinent visual details
are considered, leading to more accurate answer.
15

Comparison of UniversalRAG to Single-modality RAG
Question What color balloons were displayed on the USNS Carl Brashear during its ship launching
ceremony in San Diego?
Ground Truth The balloons were red, white, and blue that were displayed on the USNS Carl Brashear
during the ship launching ceremony in San Diego.
No Retrieval The USNS Carl Brashear’s ship launching ceremony in San Diego likely featured a diverse
range of balloons, but the specific color or colors of the balloons during the ceremony are
not universally documented. They may not have been a focus of the event, and such details
can vary or not be available in the public record. Therefore, the exact answer to the colors
of the balloons on the USNS Carl Brashear may not be specified or may be unknown.
TextRAG Retrieved: Randall Dark Randall P. Dark is a Canadian writer, director, and producer
recognized as an early proponent of HDTV . In 1995, Dark and his company, HD Vision,
were involved with the United Nations 50th Anniversary where they shot, in HDTV ,
President Bill Clinton’s speech in San Francisco. In 1999, Dark and HD Vision produced
a documentary for PBS about the Albuquerque Balloon Festival. This program aired in
HDTV in June 1999. Much of Dark’s work involves sweeping nature cinematography
including Trinity River Raft Ride featuring Eddie Bracken. One of the 3D projects Dark
directed in 2010 was "3 Cities in
Response: The provided documents do not contain any information about the color of the
balloons displayed on the USNS Carl Brashear during its ship launching ceremony in San
Diego.
VideoRAG Retrieved:
Response: Green, blue, and yellow.
UniversalRAG (Ours) Routed to: Image
Retrieved:
Response: The USNS Carl Brashear was adorned with red, white, and blue balloons
during its launch ceremony in San Diego.
Table 8: Case study comparing RAG on single modality vs. UniversalRAG (Ours). Traditional RAG models are
limited by modality-specific retrieval, often missing relevant knowledge from other modalities. UniversalRAG
(Ours) dynamically routes each query to the most appropriate modality, enabling more accurate retrieval and
response generation.
16

Comparison of Using Different Text Granularity
Question George Reed was, along with which Drake University player, one of the greatest running
backs in CFL history?
Ground Truth Johnny D. Bright
ParagraphRAG Retrieved: Blue Bombers. He remained in Winnipeg for two seasons (1986-87), but only
played in nine games. He finished his career with 4,586 rushing yards, 275 receptions
for 2,861 yards and 48 total touchdowns. His son Corey was a sprinter on Clemson’s
men’s track & field team. Lester Brown (Canadian football) Lester Brown (born January 5,
1957) is a former Canadian Football League running back for the Saskatchewan Roughrid-
ers, Montreal Concordes, Toronto Argonauts, Ottawa Rough Riders and Winnipeg Blue
Bombers. He played college football at Clemson University. Brown attended Myrtle
Beach High School, before accepting a football scholarship from Clemson University.
Response: Lester Brown
UniversalRAG (Ours) Routed to: Document
Retrieved:
George Reed (Canadian football) George Robert Reed, CM, SOM (born October 2, 1939),
is a former American college football and Canadian Football League player. Reed, along
with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being
the greatest running back in CFL history. In November 2006, Reed was voted one of the
CFL’s Top 50 players (#2) of the league’s modern era by Canadian sports network.
...
Johnny Bright Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional
Canadian football player in the Canadian Football League. He played college football at
Drake University. He is a member of the Canadian Football Hall of Fame, the National
Football Foundation’s College Football Hall of Fame, the Missouri Valley Conference
Hall of Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame,
and the "Des Moines Register’s" Iowa Sports Hall of Fame.
Response: Johnny Bright
Table 9: Case study comparing different levels of text granularity. The user query requires complex retrieval
involving multiple entities. ParagraphRAG retrieves limited context centered around a single entity, leading to an
incorrect answer. UniversalRAG (Ours) routes the query to the document corpus and retrieves richer document-level
information, allowing it to capture both relevant entities and generate the correct response.
17

Comparison of Using Different Video Granularity
Question Who finishes first in the Men’s 100M Round 1 Heat 5 during the London 2012 Olympics,
featuring Usain Bolt and Yohan Blake?
(A) Su BingTian
(B) Usain Bolt
(C) Asafa Powell
(D) Tyson Gay
Groud Truth C
VideoRAG Retrieved:
(Timestamp Range: 00:00~38:26)
Response: B
UniversalRAG (Ours) Routed to: Clip
Retrieved:
(Timestamp Range: 25:57~29:22)
Response: C
Table 10: Case study comparing different levels of video granularity. The user query requires only a segment
of the video to determine the answer. VideoRAG retrieves a broad range of frames across the video, which
includes irrelevant content and leads to an incorrect answer. UniversalRAG (Ours) routes the query to the clip-level
granularity, retrieving more focused and relevant visual information, enabling it to generate the correct response.
18

Classify the following query into one of six categories: [No, Paragraph, Document, Image,
Clip, Video] , based on whether it requires retrieval-augmented generation (RAG) and the most
appropriate modality. Consider:
•No: The query can be answered directly with common knowledge, reasoning, or computation
without external data.
•Paragraph : The query requires retrieving factual descriptions, straightforward explanations,
or concise summaries from a single source.
•Document : The query requires multi-hop reasoning, combining information from multiple
sources or documents to form a complete answer.
•Image : The query focuses on visual aspects like appearances, structures, or spatial relation-
ships.
•Clip: The query targets a short, specific moment or event within a video, without needing full
context.
•Video : The query requires understanding dynamic events, motion, or sequences over time in
a video.
Examples:
• "What is the capital of France?" →No
• "What is the birth date of Alan Turing?" →Paragraph
•"Which academic discipline do computer scientist Alan Turing and mathematician John von
Neumann have in common?" →Document
• "Describe the appearance of a blue whale." →Image
• "Describe the moment Messi scored his goal in the 2022 World Cup final." →Clip
• "Explain how Messi scored his goal in the 2022 World Cup final." →Video
• "Solve 12 × 8." →No
• "Who played a key role in the development of the iPhone?" →Paragraph
•"Which Harvard University graduate played a key role in the development of the iPhone?" →
Document
• "Describe the structure of the Eiffel Tower." →Image
• "Describe the moment Darth Vader reveals he is Luke’s father in Star Wars." →Clip
• "Analyze the sequence of events leading to the fall of the Empire in Star Wars." →Video
Classify the following query: { query }
Provide only the category.
Figure 6: Prompt for query routing in a train-free manner
19

Evaluate whether the query can be answered using general knowledge about the image’s subject
rather than relying solely on details unique to the provided image, and verify that the answer is
obtainable from the image and the query.
• Respond "yes" if:
1. The query can be fully answered using general knowledge about the subject.
2.The answer can be derived solely from the image and the query, without needing image-
specific details.
• Respond "no" if either condition is not met.
Example 1:
• Image: A portrait of Donald Trump
• Query: What is the color of Trump’s hair?
• Answer: White
• Response: "yes"
Example 2:
• Image: A close-up photo of a light bulb
• Query: What is the color of the light bulb in this image?
• Answer: Yellow
• Response: "no"
Figure 7: Prompt to filter queries for WebQA
You will receive a query from a video QA dataset and the title of the corresponding video on
YouTube. I want you to paraphrase the query by replacing "in the video?", "of the video",
or similar phrases with references to the video content naturally. The output should sound
as if a human is asking ChatGPT, and should not explicitly mention the exact name of the
video or even parts of the title. However, the rephrased query should contain enough implicit
information about the video to allow the model to identify it. Try to reduce the chance
of the model getting confused between multiple possible video candidates. If there could
be multiple video matches for a given query, try to include more information in the rephrased query.
Example 1:
• Query: What year appears in the opening caption of the video?
• Video Title: Blue Eye Samurai | Hammerscale | Full Episode | Netflix
• Upload Date: 2023-11-05
• Channel Name: Netflix
•Rephrased Output: What year appears in the opening caption of the Blue Eye Samurai episode
on Netflix?
Example 2:
•Query: After the vlogger sees a dog with an advertisement from the company named Smitten,
camera changes to the scene with ___.
• Video Title: My ICELAND Experience | Ultimate Travel Vlog
• Upload Date: 2022-10-26
• Channel Name: Kallmekris
•Rephrased Output: After spotting a dog with a Smitten advertisement, what scene does the
camera transition to in Kallmekris’s Iceland travel vlog from 2022?
Figure 8: Prompt to rephrase queries using video metadata for LVBench and CinePile
20
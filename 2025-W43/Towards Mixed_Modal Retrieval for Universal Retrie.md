# Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation

**Authors**: Chenghao Zhang, Guanting Dong, Xinyu Yang, Zhicheng Dou

**Published**: 2025-10-20 09:56:43

**PDF URL**: [http://arxiv.org/pdf/2510.17354v1](http://arxiv.org/pdf/2510.17354v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for
enhancing large language models (LLMs) by retrieving relevant documents from an
external corpus. However, existing RAG systems primarily focus on unimodal text
documents, and often fall short in real-world scenarios where both queries and
documents may contain mixed modalities (such as text and images). In this
paper, we address the challenge of Universal Retrieval-Augmented Generation
(URAG), which involves retrieving and reasoning over mixed-modal information to
improve vision-language generation. To this end, we propose Nyx, a unified
mixed-modal to mixed-modal retriever tailored for URAG scenarios. To mitigate
the scarcity of realistic mixed-modal data, we introduce a four-stage automated
pipeline for generation and filtering, leveraging web documents to construct
NyxQA, a dataset comprising diverse mixed-modal question-answer pairs that
better reflect real-world information needs. Building on this high-quality
dataset, we adopt a two-stage training framework for Nyx: we first perform
pre-training on NyxQA along with a variety of open-source retrieval datasets,
followed by supervised fine-tuning using feedback from downstream
vision-language models (VLMs) to align retrieval outputs with generative
preferences. Experimental results demonstrate that Nyx not only performs
competitively on standard text-only RAG benchmarks, but also excels in the more
general and realistic URAG setting, significantly improving generation quality
in vision-language tasks.

## Full Text


<!-- PDF content starts -->

Towards Mixed-Modal Retrieval for Universal
Retrieval-Augmented Generation
Chenghao Zhang
Guanting Dong
Renmin University of China
Beijing, China
chenghao-zhang@outlook.comXinyu Yang
Zhicheng Dou
Renmin University of China
Beijing, China
dou@ruc.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has emerged as a pow-
erful paradigm for enhancing large language models (LLMs) by
retrieving relevant documents from an external corpus. However,
existing RAG systems primarily focus on unimodal text documents,
and often fall short in real-world scenarios where both queries and
documents may contain mixed modalities (such as text and im-
ages). In this paper, we address the challenge of Universal Retrieval-
Augmented Generation (URAG), which involves retrieving and rea-
soning over mixed-modal information to improve vision-language
generation. To this end, we proposeNyx, a unified mixed-modal
to mixed-modal retriever tailored for URAG scenarios. To miti-
gate the scarcity of realistic mixed-modal data, we introduce a
four-stage automated pipeline for data generation and filtering,
leveraging web documents to constructNyxQA, a dataset compris-
ing diverse mixed-modal question-answer pairs that better reflect
real-world information needs. Building on this high-quality dataset,
we adopt a two-stage training framework forNyx: we first per-
form pre-training onNyxQAalong with a variety of open-source
retrieval datasets, followed by supervised fine-tuning using feed-
back from downstream vision-language models (VLMs) to align
retrieval outputs with generative preferences. Experimental re-
sults demonstrate thatNyxnot only performs competitively on
standard text-only RAG benchmarks, but also excels in the more
general and realistic URAG setting, significantly improving gen-
eration quality in vision-language tasks. Our code is released at
https://github.com/SnowNation101/Nyx
CCS Concepts
•Information systems →Web search engines;Multimedia
and multimodal retrieval;Web mining;Question answering;
Retrieval models and ranking.
Keywords
Multimodal Retrieval-Augmented Generation, Vision-Language
Model, Multimodal Embedding, Contrastive Learning
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
WWW ’26, Dubai, UAE
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX
Text
Image
She is crowned with an orb of da-rkness and trails a veil ofnocturnal mists. The figure is labelled NUKS on the vase. Image-TextPairInterleavedImage-TextIn Greekmyth-ology, Nyx is the goddess and personification of the night…
“Mixed-Modal”Figure 1: An illustration of the input patterns of “mixed-
modal” content in the URAG scenario.
1 Introduction
Large language models (LLMs) have shown remarkable capabilities
in text comprehension and generation [ 8,15,37,38,48]. To extend
their capabilities to multimodal understanding, vision-language
models (VLMs) incorporate visual encoders to process text and
image inputs [ 1,57]. However, like LLMs, VLMs often struggle
with queries needing up-to-date or external knowledge. Retrieval-
Augmented Generation (RAG) addresses this by retrieving doc-
uments from an external corpus to complement internal knowl-
edge [ 10,14,31]. Building on this, Multimodal RAG (MRAG) extends
the paradigm to settings where both queries and documents may
contain text, images, or both [5, 50].
Current MRAG methods fall broadly into two categories:(1)
The divide-and-conquer approach, which utilize text queries
for text documents and visual queries for images;(2) The cross-
modal retrieval, which uses text queries to retrieve visual content.
However, both paradigms suffer from notable limitations. They
often overlook the spatial and logical relationships between images
and text within a document, making it difficult to capture fine-
grained interactions crucial for downstream reasoning.
However, web documents in the real world are often far more
complex and diverse. As illustrated in Figure 1, they may include
pure text, individual images, paired image-text content, or arbitrar-
ily interleaved sequences of text and images. We refer to this broad
spectrum of formats asmixed-modalcontent, where the interplay
between modalities plays a critical role in conveying meaning.
While recent efforts, such as VLM2Vec [ 24], have introduced
unified multimodal embedding models, these approaches mainly
focus on embedding pure text, individual images, or neatly alignedarXiv:2510.17354v1  [cs.CL]  20 Oct 2025

WWW ’26, April 13–17, 2026, Dubai, UAE Zhang et al.
text-image pairs. Consequently, they face challenges in handling
complex multimodal structures, such as interwoven or densely
intertwined text and images. Furthermore, their application within
the MRAG framework is still largely unexplored.
This gap becomes even more pronounced in the context of
Universal Retrieval-Augmented Generation (URAG), as both
queries and documents can be of arbitrary mixed modalities. Unlike
traditional settings with purely textual queries, URAG introduces
dual challenges: understanding heterogeneous inputs and retrieving
from equally diverse corpora. An effective retriever needs to be
capable of encoding various content types and matching them
with complex document structures. This imposes new technical
requirements on representation learning, matching precision, and
alignment with downstream VLMs, highlighting the need for a truly
universal, flexible, and vision-language-aware retrieval paradigm.
To address these challenges, we proposeNyx, a unified retriever
designed for mixed-modal-to-mixed-modal retrieval in URAG sce-
narios. To mitigate data scarcity of realistic URAG training data,
we first introduce a four-stage automatic pipeline to buildNyxQA,
a new dataset tailored for URAG.NyxQAconsists of three com-
ponents: (1) a large-scale mixed-modal multiple-choice question
answering (QA) dataset, (2) a corresponding mixed-modal docu-
ment corpus, and (3) a pretraining dataset for contrastive learning.
Our construction process begins with sampling naturally inter-
leaved image-text documents from the web to form the corpus. We
then employ a powerful VLM to generate QA pairs conditioned on
these documents. To ensure high data quality, we apply a multi-
step post-processing procedure, yielding a clean and diverse QA set.
Finally, based on these QA pairs, we mine hard negatives from the
corpus to form the pretraining triplets used for contrastive learning.
Unlike existing multimodal datasets limited to specific modality
combinations,NyxQAsupports retrieval and generation involving
arbitrarily structured text, images, and their interleaved formats.
Building upon this dataset, we adopt a two-stage training frame-
work to developNyxfrom a pretrained VLM. In the first stage, we
pretrain the retriever onNyxQAand several public contrastive
learning datasets to establish general-purpose multimodal retrieval
capabilities. To balance retrieval effectiveness and efficiency, we
incorporate Matryoshka Representation Learning (MRL) [ 28], re-
sulting in a compact yet expressive encoder, termedNyx-pretrained.
In the second stage, we perform feedback-driven fine-tuning, align-
ing the retriever with the generative preferences of downstream
VLMs. This yields the final version of our retriever,Nyx.
Extensive experiments demonstrate thatNyxconsistently en-
hances retrieval accuracy and downstream reasoning performance
in challenging mixed-modal scenarios, showcasing its strong suit-
ability for URAG tasks.
In conclusion, our contributions are as follows:
•We pioneer the exploration of the Universal Retrieval Aug-
mented Generation (URAG) problem, addressing scenarios
where both queries and documents consist of arbitrarily in-
terleaved image-text content.
•We introduce a dataset specifically designed for real-world
URAG applications, created through a comprehensive four-
step web-based multimodal data synthesis pipeline. This
dataset offers a rich variety of interleaved content formats,serving as an effective benchmark for practical multimodal
retrieval tasks.
•We propose a two-stage training paradigm to developNyx,
a unified retriever optimized for URAG. The first stage in-
volves contrastive pretraining using MRL on both public and
synthetic datasets, resulting inNyx-pretrained. Then we
utilize feedback from VLMs to refine the retriever through
targeted supervision, culminating in our final model,Nyx.
2 Related Work
Multimodal Retrieval-Augmented Generation (MRAG).ex-
tends the traditional RAG framework to multimodal settings by
retrieving text, images, or image-text pairs from an external corpus
to support Vision-Language Models (VLMs) in generating textual
responses [ 5]. Current methods use various retrieval strategies:
dual-path strategies retrieve text with text queries and images with
image queries [ 13,40]; cross-modal retrieval techniques [ 6,50]; and
treating multimodal documents as images for retrieval [42].
In real-world applications, queries and corpora often contain
mixed-modalinputs—combinations of text and images. New multi-
modal deep search paradigms introduce iterative retrieval [ 18,33,
34,47], where intermediate queries may also be mixed-modal. How-
ever, a unified retrieval framework for URAG scenarios remains
undeveloped.
Multimodal Embedding Retrievers.focus on retrieving rele-
vant multimodal documents by encoding both queries and docu-
ments into a shared embedding space. In text-only contexts, embedding-
based retrievers have shown strong performance across various
tasks and languages [ 3,11,21,26,45]. Extending this to multi-
modal scenarios, cross-modal retrievers like CLIP [ 22,27,39,51]
and vision-language models such as BLIP-2 [ 17,32] encode text
and images into a unified space, enabling retrieval tasks such as
image-to-text, text-to-image, and image-to-image.
Recent advancements [ 23,24,46,55] have leveraged VLMs as
general-purpose encoders for text, images, and image-text pairs.
Other studies have focused on using synthetic data [2, 54, 56] and
improving contrastive learning objectives [ 29,43] to enhance em-
bedding quality. Building on this, MME [ 53] utilized synthetic data
to improve performance on interleaved text-image retrieval in the
wikiHow task. However, these methods lack support for text-to-text
and general interleaved text-image retrieval in URAG scenarios.
Moreover, most retrievers are trained independently of downstream
VLMs, resulting in suboptimal alignment. Therefore, this paper
proposes a unified retriever,Nyx, which builds a bridge for mixed-
modal to mixed-modal retrieval, leading to better alignment with
VLM generation.
3 Methodology
3.1 Problem Formulation: URAG
This work addresses the task ofUniversal Retrieval-Augmented
Generation (URAG), which aims to generate high-quality textual
responses to mixed-modal queries by retrieving and leveraging
relevant information from a mixed-modal corpus. Amixed-modal
content𝑥is defined as an ordered sequence of elements, where

Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, UAE
Step1:WebDocSampling
Question1Answer1AnswernQuestionn
Real-WorldWebPages…
…Sampling&Chunking
GoldenDocStep4:HardNegativeMiningQuestion:What is the title of the track mentioned in this document?Answer: Blackburn explains the value by giving potential … and the impact of limited flat …housing development.Answer is overly long and difficult to evaluate.Question directly referring to a specific document.
Question:Accordingto<image_76>wherecanIfindthebook?The image pointer points to a non-existent image.LowQualitySamples
QuestionNegativeDocNegativeDocNegativeDocNegativeDocMixed-ModalCorpus
A.Creative and unconventionalD.Casual and relaxedC.PersistentandauthenticB.Modern and urban
Choices:GoldenDoc
Step3:PostProcessing
Step2:QAGenerationCollectthenon-goldendocsfromtheretrieveddocuments
NyxPretrainingDataNyxQA
Mixed-ModalCorpus…
Accordingtothephotos
what does Shirley's attire suggest about her personality?Question
andThePostProcessingPipelineRawQ&AErrorFilteringQARefinementOptionGenerationNyxQAStep3
Figure 2: The proposed four-step automatedNyxQAconstruction pipeline.
each element can be either a textual segment or an image. Formally,
it can be represented as:𝑥∈{𝑎 1𝑎2...𝑎𝑛|𝑎𝑖∈{T,I}}.
To effectively accomplish this task, we presuppose access to
amixed-modal corpus C. A retriever is required to retrieve a
relevant subset ofdocuments R(𝑞)⊆C , conditioned on a mixed-
modalquestion 𝑞. The retrieved content then serves to guide the
generation of the ultimate textual response. Formally, the objective
is to learn a retrieval function Rand a generation model Gsuch
that:𝑦=G(𝑝,𝑞,R(𝑞)) , where𝑝denotes a textual prompt, 𝑞is
the mixed-modal query, and R(𝑞) represents the set of retrieved
documents. The output 𝑦is anatural-language textual response.
The retrieval function Rselects the top- 𝐾most relevant entries
from the corpusCbased on their relevance to the query𝑞:
R(𝑞)=TopK𝑑∈Csim(𝑞,𝑑),
where𝑑denotes a document in the mixed-modal corpus and sim(·,·)
is a similarity-based relevance function defined within the joint
vision-language embedding space.
The generation model G, typically instantiated as a VLM, pro-
cesses both the query 𝑞and the retrieved documents R(𝑞) to yield
a coherent and factually grounded textual output𝑦.
3.2NyxQA: A Dataset for URAG
To simulate a realistic web environment, we introduceNyxQA, a
large-scale mixed-modal dataset designed for the URAG setting. Our
dataset comprises three components: (1) a high-quality multiple-
choice QA datasetDNyxQA with mixed-modal questions, (2) a corre-
sponding mixed-modal document corpus Cmix, and (3) a contrastive
pretraining setDpretrain containing positive and hard negative ex-
amples for retriever training.
The construction ofNyxQAfollows a four-stage pipeline. First,
we sample and segment web documents to create a diverse mixed-
modal corpus. Next, we utilize a VLM to generate QA pairs fromthese document segments. This is followed by a post-processing
pipeline to filter errors, refine answers, and format multiple-choice
options. Finally, we employ hard negative mining using an existing
retriever to produce high-quality contrastive training triplets for
pretraining theNyxmodel.
Web Document Sampling.To obtain naturally occurring mixed-
modal documents, we sample from OBELICS [ 30], a large-scale
dataset of web pages featuring interleaved text and images that
reflect real-world multimodal distributions. Following standard
practices in text-only RAG [ 25], each document is segmented into
smallerchunks{𝑑𝑖}𝑁
𝑖=1, where each 𝑑𝑖contains up to 200 textual
tokens (excluding image tokens from the count). This segmen-
tation maintains semantic coherence and prevents length imbal-
ance caused by densely illustrated documents. The resulting set of
chunks forms our mixed-modal corpus Cmix={𝑑𝑖}𝑁
𝑖=1, comprising
46,741 segments in total. We then perform stratified sampling of
10,000 chunks from Cmixas the basis for QA pair generation, while
preserving the original modality distribution.
QA Pair Generation.For each sampled chunk 𝑑𝑖, whether text-
only or containing images, we use a VLM to generate up to five
context-independent raw QA pairs (𝑞raw
𝑖𝑗,𝑎raw
𝑖𝑗), ensuring that each
question can be answered solely based on its associated chunk.
For chunks with images, we specifically prompt the VLM to create
questions that reference the visual content. Since the model outputs
text only, we use special tags such as <image k> to denote the 𝑘-th
image within the chunk. This process produces the raw QA dataset
Draw={(𝑑𝑖,𝑞raw
𝑖𝑗,𝑎raw
𝑖𝑗)}, which contains diverse samples ranging
from pure text to multi-image questions, thereby enriching the
modality diversity ofNyxQA.
Post-Processing.The initial set Drawof generated QA pairs is of
suboptimal quality, containing various errors that could adversely

WWW ’26, April 13–17, 2026, Dubai, UAE Zhang et al.
Top-4Doc
(a)ModelArchitecture
(b)ContrastiveLearningwithMRL(c)VLM-GuidedFeedbackLargeLanguageModelVisionEncoder
…he was furious, and would have hurled him into the sea, had he not fled to the protection of Nyx, as Zeus……In the works of Greekpoets,Thetisis a Nereid who attracts the attention of bothZeusandPoseidon,…
……………<EOS>Embedding
MatryoshkaRepresentationQueryEmbeddingPos.DocumentEmbeddingNeg.DocumentEmbeddingpullpush
Top-1
Top-K…
Top-3
VLM(Generator)SlidingWindow(Length=L)
RetrievedDocsInputContextPromptQuestionAnswersLdocumentsinthewindowAnswer1Answer2
AnswerK-L+1Answer3Answer4…
Oneinstanceofthedownstream preference datasetQuestionQueryPositiveDocumentNegativeDocuments
…
Generates answers for each input contextJudgetheanswers
Top-1DocTop-3DocTop-KDoc…Top-2DocTop-5DocTop-2
Figure 3: Overview of theNyxarchitecture and its training paradigm.
affect subsequent training and evaluation. Therefore, we perform a
three-stage post-processing procedure on the raw data to produce
the finalNyxQAdataset.
•Error Filtering.Questions with explicit contextual refer-
ences (e.g., phrases like “in this document”) are removed
using rule-based filters. In addition, we ensure image–text
consistency by verifying that the image tags mentioned in
the generated question correspond to actual images present
in the chunk𝑑 𝑖.
•QA Refinement.We further refine the filtered QA pairs
using a VLM to enhance clarity and completeness. Each re-
tained pair(𝑞raw
𝑖𝑗,𝑎raw
𝑖𝑗)is compressed to its essential content,
eliminating redundancy while preserving factual accuracy.
This process yields concise, self-contained questions and
answers that align closely with the corresponding gold doc-
ument𝑑+
𝑖, resulting in the refined set(𝑑+
𝑖,𝑞𝑖𝑗,𝑎+
𝑖𝑗).
•Option Generation.For each refined QA, an LLM gener-
ates three semantically plausible distractors {𝑎−
𝑖𝑗}for the
question𝑞𝑖𝑗. After shuffling the distractors with the correct
answer, we finalize each sample with the question, four op-
tions, and the gold document, forming our multiple-choice
datasetD NyxQA ={(𝑞𝑖𝑗,{𝑎+
𝑖𝑗,𝑎−
𝑖𝑗},𝑑+
𝑖)}.
Hard Negative Mining.To enhance retriever pretraining, we
construct contrastive triplets using DNyxQA . Each question 𝑞𝑖𝑗serves
as a query, with its corresponding gold document 𝑑+
𝑖designated as
the positive sample. We then employ mmE5 [ 2] to retrieve the top-
10 relevant documents from the mixed-modal corpus Cmix. From
these, we select five documents that differ from 𝑑+
𝑖as hard nega-
tives{𝑑−
𝑖𝑗}, prioritizing the highest-ranked candidates. This yields
the pretraining dataset Dpretrain ={(𝑞𝑖𝑗,𝑑+
𝑖,{𝑑−
𝑖𝑗})}, a contrastive
training set specifically designed for mixed-modal retrieval.3.3Nyx: Training Paradigm
Overview.Our goal is to build a unified retriever capable of
handling mixed-modal queries and documents across diverse real-
world scenarios. To this end, we begin by pretrainingNyxon a
large-scale corpus that includes both public and synthetic datasets
spanning various modality configurations. This initialization equips
the retriever with general-purpose retrieval capabilities across text-
only, image-only, and multimodal pairs.
However, generic pretraining may not fully align with the spe-
cific information needs of downstream VLMs during generation.
Therefore, in the second stage, we fine-tuneNyx-pretrained through
a feedback-driven learning process, leveraging VLM responses to
construct high-quality examples that reflect the actual relevance
signals needed for multimodal generation.
Throughout both stages, we employ contrastive learning with
Ma Matryoshka Representation Learning [ 28] to ensure scalable and
efficient embedding quality under varying dimensional constraints.
We detail our training objective and the two-stage procedure below.
Training Objective.We build our retriever on top of a pre-
trained VLM, Qwen-2.5-VL-3B-Instruct [ 1], as the backbone en-
coder. Given an input sequence, we use the hidden representation
of the final<EOS>token as the global embedding for retrieval.
Following established practices in embedding model training,
we construct each training instance as a triplet {𝑞,𝑑+,{𝑑−
𝑛}𝑁
𝑛=1},
where𝑞is a query,𝑑+is a positive document, and {𝑑−
𝑛}are𝑁
negative documents. An instruction string is prepended to each
query before encoding. Both queries and documents may come
from mixed modalities (e.g., text, image, or interleaved image-text),
allowingNyxto operate in a unified embedding space.
To learn discriminative representations, we adopt the InfoNCE
loss for contrastive learning. Leth 𝑞∈R𝑑,h+∈R𝑑, and{h−
𝑛}𝑁
𝑛=1⊂
R𝑑denote the embeddings of the query, the positive document,
and the negative documents, respectively. We apply Matryoshka
Representation Learning (MRL) [ 28], which encourages the full

Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, UAE
embeddingh∈R𝑑to remain informative even when truncated
to lower-dimensional subspaces. This enables flexible trade-offs
between retrieval performance and memory efficiency.
Specifically, for a set of target dimensions {𝑑1,𝑑2,...,𝑑𝐾}, where
𝑑𝑘<𝑑, we truncate each embedding to its first 𝑑𝑘dimensions,
denoted ash(𝑑𝑘)∈R𝑑𝑘. For each𝑑𝑘, we compute an InfoNCE loss
as:
L(𝑑𝑘)
Info=−log𝜙(h(𝑑𝑘)
𝑞,h+(𝑑𝑘))
𝜙(h(𝑑𝑘)
𝑞,h+(𝑑𝑘))+Í𝑁
𝑛=1𝜙(h(𝑑𝑘)
𝑞,h−(𝑑𝑘)
𝑛),(1)
where𝜙(a,b)=exp(sim(a,b)/𝜏), and sim(·,·) denotes cosine sim-
ilarity with temperature hyperparameter 𝜏>0. Here,h+(𝑑𝑘)and
h−(𝑑𝑘)
𝑛 correspond to the positive and the 𝑛-th negative sample
document embeddings, respectively.
The final training objective aggregates the InfoNCE losses over
all truncated dimensions as a weighted sum:
LMRL=𝐾∑︁
𝑘=1𝑤𝑘·L(𝑑𝑘)
Info,where𝐾∑︁
𝑘=1𝑤𝑘=1,(2)
with𝑤𝑘denoting the weight for the 𝑘-th dimension. This objective
encourages each embedding prefix to preserve semantic integrity
under varying retrieval constraints.
Stage 1: Pretraining with Mixed-Modal Data.In the first
stage, we pretrainNyxas a general-purpose retriever using a
large-scale corpus constructed from both public and synthetic data
sources. Following mmE5 [ 2], we include MMEB [ 24] and synthetic
triplets from the mmE5 pipeline. To ensure the model’s ability to
handle genuinely mixed-modal scenarios, we further integrate our
proposedNyxQAdataset, which supports retrieval across diverse
modality combinations.
Since real-world retrieval tasks still predominantly involve tex-
tual inputs, we further enhance the retriever’s text understanding
ability by introducing additional text-only datasets. Specifically, we
use the training sets from HotpotQA [ 49], 2WikiMultiHopQA [ 19],
and MuSiQue [ 44]. For each query, we retrieve the top- 𝐾documents
from the full Wikipedia corpus using E5-v2 [ 45], and treat the top-1
document as the positive sample, while selecting negative samples
from documents ranked beyond top-10.
All the datasets described above are combined and jointly used
to train the initial retriever, resulting in theNyx-pretrained model.
Stage 2: Supervised Fine-tuning with VLM-Guided Feedback.
WhileNyx-pretrained demonstrates strong retrieval performance,
it is not explicitly optimized for supporting downstream generation
by VLMs. To bridge this gap, we introduce a fine-tuning stage that
leverages feedback from a VLM to align the retriever with the actual
information needs during generation.
Given a dataset 𝐷={(𝑞𝑖,𝑎𝑖)}of queries and their corresponding
answers, along with a retrieval corpus C, we proceed as follows.
For each query 𝑞𝑖, we first useNyx-pretrained to retrieve the top- 𝐾
candidate documents {𝑑1,𝑑2,...,𝑑𝐾}. Then, using a sliding win-
dow of length 𝐿, we construct a sequence of candidate contexts
by grouping contiguous subsets of the retrieved documents. Each
context window is concatenated with the query and fed into the
VLM to generate an answer.We select the first context window that either yields a correct
answer or exceeds a pre-defined generation metric threshold (e.g.
EM, F1). The first document in this window is treated as the positive
sample𝑑+, and the remaining 𝐾−1documents are used as negative
samples{𝑑−
𝑛}. If no window meets the quality threshold, the entire
instance is discarded from the feedback dataset.
By applying this procedure to all queries in 𝐷, we construct
adownstream preference datasetfrom the feedback 𝐷pref=
{(𝑞𝑖,𝑑+
𝑖,{𝑑−
𝑖,𝑛})}, which reflects the actual preferences of the VLM
in real generation scenarios. We then fine-tuneNyx-pretrained
on this dataset using the same contrastive learning framework
described earlier, thus obtaining the final retrieverNyx.
4 Main Experiments
Our main experiments consist of two parts, where we first evaluate
the generation performance in URAG scenarios and then examine
the embedding performance, given that the model is inherently an
embedding model. All experiments were conducted on a single node
equipped with 8×NVIDIA A800-SXM4-80GB GPUs. For efficient
training, we applied LoRA [ 20] with a rank of 8. The per-device
batch size was set to 20 with 4 gradient accumulation steps, and the
temperature parameter 𝜏in the InfoNCE loss was fixed at 0.02. To
avoid memory overflow when processing multi-image samples, the
maximum visual input resolution was limited to400 ×28×28pixels.
Additional implementation details can be found in the appendix.
4.1 Experimental Setup
Datasets and Metrics.We evaluate RAG pipelines incorporat-
ing our retriever across two categories: (1) text-only datasets and
(2) multimodal datasets, including MRAG and URAG.
For text-only RAG, followingReCall[ 4], we evaluate onHot-
potQA[ 49],MuSiQue[ 44], andBamboogle[ 36]. For each ques-
tion, we use E5-v2 to retrieve the top 20 documents from Wikipedia,
merge and deduplicate them to create a task-specific corpus, and
randomly sample up to 250 questions per dataset.
In the case of MRAG and URAG, we utilizeMultimodalQA
(MMQA) [ 41],ScienceQA(SciQA) [ 35], andNyxQA. MMQA re-
quires retrieval from a corpus containing text, tables, and images.
Since the tables in the MMQA corpus are originally stored in JSON
format, we employ the Python tabulate library to convert them
into a more comprehensible text table format. SciQA presents image-
text questions, and we construct its corpus using associated lec-
tures and QA examples.NyxQA, constructed from authentic web
pages, covers a broader spectrum of input and document modalities,
thereby facilitating more realistic and comprehensive evaluation of
URAG in real-world web environments.
During pretraining, we use the training sets of 2WikiMulti-
HopQA, HotpotQA, MuSiQue, andNyxQA, treating Bamboogle,
MMQA, and SciQA as out-of-domain (OOD) evaluation sets. For
feedback-based fine-tuning, feedback is collected from HotpotQA,
MuSiQue, MMQA, SciQA, andNyxQA, with Bamboogle serving
as the OOD benchmark.
For multiple-choice datasets, we reportaccuracy(Acc), while
for open-ended QA, we adoptexact match(EM) andF1 score(F1),
reflecting both strict correctness and token-level overlap between
predictions and references.

WWW ’26, April 13–17, 2026, Dubai, UAE Zhang et al.
Table 1: The overall results on the six RAG datasets. To ensure consistent evaluation, the top document retrieved by each
retriever was combined with the corresponding question, then input into Qwen2.5-VL-7B for answer generation. The exception
is SciQA, where the retrieval content consists of one lecture and two example-based retrieval results to suit the dataset’s
structure. This setup isolates the effect of the retrievers, facilitating a controlled comparison of retrieval performance. The best
results are highlighted in bold, and the second-best results are underlined.
MethodHotpotQA Bamboogle MuSiQue SciQA MMQA NyxQAAvg.
EM F1 EM F1 EM F1 Acc EM F1 Acc
Direct Answer
InternVL3 (8B) [57] 16.40 22.88 9.60 15.49 3.60 8.16 78.87 20.07 23.99 53.33 25.31
Qwen2.5-VL (7B) [1] 12.40 18.36 6.40 11.50 3.29 7.32 77.98 20.73 24.39 50.17 24.38
Text RAG
E5-v2 (109M) [45] 14.40 19.18 7.20 12.80 2.40 6.79 – – – – –
Vision-Language RAG
CLIP (150M) [39] 14.00 21.12 6.40 11.64 3.20 6.74 73.07 18.03 20.67 61.50 23.64
VLM2Vec (4B) [24] 14.40 22.08 10.40 16.95 3.60 10.12 79.56 19.91 23.34 56.50 25.69
VisRAG-Ret (3B) [50] 12.08 19.84 8.80 16.05 3.60 8.29 80.45 18.84 21.55 64.33 25.38
mmE5 (11B) [2] 17.60 24.30 13.60 18.69 5.20 9.70 81.40 34.00 38.50 66.83 30.98
Ours
Nyx-pretrained (3B) 22.00 31.38 16.00 22.87 5.60 11.00 81.33 31.75 35.97 74.83 33.27
Nyx(3B)24.40 33.19 16.80 25.93 7.20 12.80 81.75 39.66 44.50 81.83 36.46
Baseline Models.For the text-only retriever, we use E5-v2 [ 45]
as the unimodal RAG baseline, since it serves as the backbone model
for constructing our text-only retrieval datasets and is also one of
the most widely used retrievers in text-based RAG systems [ 25].
For multimodal retrievers, we use well-supervised fine-tuned em-
bedding models CLIP [ 39], VLM2Vec [ 24] and mmE5 [ 2], as well as
a retriever for visual document retrieval for RAG, VisRAG-Ret [ 50].
We also report the direct answering results of InternVL3-8B [ 57]
and Qwen2.5-VL-7B as baselines for comparison.
4.2 Results on Generation Performance
Our generation performance results are presented in Table 1. Over-
all,Nyxconsistently outperforms all baselines, clearly demonstrat-
ing its superiority. We further highlight the following insights:
Performance in Text-Only RAG.Despite the powerful 11 bil-
lion parameter VLM backbone of mmE5, our 3 billion parameter
Nyx-pretrained model still outperforms mmE5 on HotpotQA, Bam-
boogle, and MuSiQue, with performance gains of 9% and 6% on Hot-
potQA and Bamboogle, respectively. This result shows the strength
of targeted training. Moreover,Nyxsubstantially surpasses the
text-only retriever E5 that is commonly used in RAG frameworks,
further demonstrating its effectiveness in unimodal retrieval.
Multimodal RAG Performance.In multimodal tasks,Nyx-
pretrained performs competitively on MMQA andNyxQA, though
it trails mmE5 slightly on SciQA. This may be attributed toNyx’s
smaller parameter count and its broader training coverage, which
includes interleaved and text-only examples. Nevertheless, its ro-
bust performance across different input types highlights the ben-
efit of mixed-modal training. After incorporating feedback from
downstream VLMs,Nyxachieves the best performance across all
multimodal benchmarks, with great results on MMQA (F1: 35.97%→44.50%) andNyxQA(Accuracy: 74.83% →81.83%). On SciQA,
the gain is modest, possibly due to the limited informativeness of
the provided lecture corpus. Nonetheless, fine-tuning with feedback
still leads to alignment with the VLM’s preferences.
A McNemar’s test was conducted onNyxQAto assess the per-
formance differences among mmE5,Nyx-pretrained, andNyxas
retrievers. The comparison between mmE5 andNyx-pretrained
yielded a test statistic of 19.0631 ( 𝜒2, 1 degree of freedom) with
a p-value of 0.0000. Furthermore, the comparison betweenNyx-
pretrained andNyxresulted in a test statistic of 15.7538 and a
p-value of 0.0001. These results provide strong evidence that the
retrieval performance differs significantly across the methods.
Beyond Gold Documents: Learning from Preference.An in-
teresting observation arises fromNyxQA, where each question
is originally paired with a generation-originated “golden” docu-
ment. Although semantically relevant, these gold documents do
not always lead to correct answers during inference. Our feedback
analysis shows that documents preferred by the VLM may differ
from the labelled positives. Incorporating this preference signal
during fine-tuning leads to a 7-point accuracy gain onNyxQA.
This suggests the importance of further aligning retrieval models
with downstream generative utility in URAG systems.
4.3 Embedding Capability Analysis
Although aligning mixed-modal retriever with downstream models
can enhance the generative quality of the final VLM, the capability
of the retriever itself is also of central importance. We evaluate the
embedding ability of our models on the MMEB benchmark [ 24],
and the results are reported in Table 2. In the table, models from
CLIP to MMRet are evaluated in the zero-shot setting, whereas the
remaining models are trained with MMEB-labelled data. In particu-
lar,mmE5-Qwen2.5-3Bis trained on Qwen-2.5-VL-3B-Instruct, with

Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, UAE
Table 2: Performance comparison on the MMEB benchmark,
which includes 36 tasks spanning four categories: classifi-
cation (Class.), visual question answering (VQA), retrieval
(Retr.), and visual grounding (Ground.).
ModelsPer Meta-Task ScoreOverall
Class. VQA Retr. Ground.
CLIP [39] 42.8 9.1 53.0 51.8 37.8
BLIP2 [32] 27.0 4.2 33.9 47.0 25.2
OpenCLIP [7] 47.8 10.9 52.3 53.3 39.7
E5-V [23] 21.8 4.9 11.5 19.0 13.3
MagicLens [52] 38.8 8.3 35.4 26.0 27.8
MMRet [56] 47.2 18.4 56.5 62.2 44.0
VLM2Vec [24] 52.8 50.3 57.8 72.3 55.9
mmE5 [2] 67.6 62.8 70.9 89.7 69.8
mmE5-Qwen-3B 56.6 56.0 59.4 71.5 59.0
Nyx-pretrained 55.2 53.7 58.4 70.5 57.5
Nyx57.9 57.5 61.8 75.7 61.1
its training data consisting of MMEB-labelled data together with
the retrieval and VQA subsets of the mmE5 synthetic data, serving
as the ablation setting.
Compared to mmE5,mmE5-Qwen2.5-3Bperforms worse across
all capabilities, which can be attributed to its smaller backbone
size and the exclusion of the classification subset from the mmE5
synthetic data (to maintain alignment withNyx-pretrained and
Nyxsettings). Nevertheless, it still surpasses other baseline mod-
els. When OOD pure text data andNyxQAmixed-modal data
are included, the mismatch with the MMEB evaluation pattern
results in a 1.5% overall performance drop. However, after fine-
tuning with feedback from a VLM on OOD datasets with entirely
different tasks,NyxoutperformsmmE5-Qwen2.5-3Bacross all ca-
pabilities, achieving a 2.1% overall improvement. These findings
further demonstrate that incorporating VLM feedback not only
improves the performance of URAG systems but also enhances the
embedding capability of dense retrievers themselves.
5 Quantitative Analysis
5.1 Impact of Data Scale on URAG Performance
The scalability of training data is crucial for building effective re-
trievers. Prior studies have shown an approximately logarithmic-
linear relationship between the volume of training data and the
quality of retrieval model embeddings [ 2,16]. In this section, we
further examine how the scale of training data affectsNyx’s per-
formance in the URAG setting.
As illustrated in Figure 4, the performance trend closely follows
a logarithmic-linear curve, consistent with previous findings. The
steady improvement of URAG performance with increasing data
scale further confirms the high quality and diversity of our training
data. This indicates that enhancements in the retriever’s indepen-
dent capabilities translate proportionally into gains in end-to-end
URAG performance. Thus, increasing training data is expected to
predictably enhance URAG scenario generalization.
log(2.88k) log(5.76k) log(14.40k) log(57.60k) log(144.00k) log(288.00k) log(576.00k) log(1241.86k)
# Training Samples0.500.550.600.650.700.750.800.85Accuracy
y = 0.1204 * log(x) + 0.1041Trend
Linear Fit
PerformanceFigure 4: Impact of training data scale onNyxQAaccuracy
when trainingNyxwith varying sample sizes.
012 4 8 16
# In-Context Documents0.50.60.70.8Accuracy
( a )
RAG with mmE5
RAG with Nyx-pretrained
RAG with Nyx
2 B 8 B 14 B 38 B 78 B
Generator VLM0.20.30.40.50.60.70.80.9Accuracy( b )
Direct Answer
RAG with Nyx-pretrained
RAG with Nyx
Figure 5: Impact of (a) the number of in-context documents
and (b) feedback-based retriever fine-tuning on downstream
generation performance. Results are shown onNyxQAusing
InternVL3 models of varying sizes, respectively.
5.2 Effect of Retrieved Document Count
To examine how the number of retrieved documents influences gen-
eration quality, we vary the number of documents fed into Qwen2.5-
VL-7B from 0 to 16, evaluating the URAG results ofmmE5,Nyx-
pretrained, andNyx. As shown in Figure 5(a), adding more doc-
uments consistently improves all retrievers, though the gains di-
minish as the count increases.Nyxconsistently outperforms both
Nyx-pretrained and mmE5, demonstrating robust performance
even with fewer documents and confirming the effectiveness of
feedback-based fine-tuning in producing informative retrievals.
Overall, the results highlight the critical importance of high-quality
top-ranked retrieval for efficient generation.
5.3 Generalization Across Generators
WhileNyxis fine-tuned with supervision from Qwen2.5-VL-7B, we
also examine whether such supervision generalizes to other VLMs.
To assess this, we evaluate its performance across InternVL3 models
of varying sizes used as generators. As shown in Figure 5 (b),Nyx
consistently outperforms the direct-answer baseline across all In-
ternVL3 variants, indicating that supervision from Qwen2.5-VL-7B
transfers effectively across different generator architectures. Inte-
gratingNyxyields further improvements, particularly for InternVL3-
2B and InternVL3-14B, with absolute gains exceeding 0.2 points.
However, the degree of improvement varies with generator size,
indicating different alignment preferences among models. Addi-
tionally, since performance does not increase monotonically with

WWW ’26, April 13–17, 2026, Dubai, UAE Zhang et al.
GoldenGoldenGoldenGoldenGolden
29.8%29.8%29.8%29.8%29.8%
NonGoldenNonGoldenNonGoldenNonGoldenNonGolden
70.2%70.2%70.2%70.2%70.2%CorrectCorrectCorrectCorrectCorrect
66.8%66.8%66.8%66.8%66.8%
IncorrectIncorrectIncorrectIncorrectIncorrect
33.2%33.2%33.2%33.2%33.2%GoldenGoldenGoldenGoldenGolden
51.8%51.8%51.8%51.8%51.8%
NonGoldenNonGoldenNonGoldenNonGoldenNonGolden
48.2%48.2%48.2%48.2%48.2%CorrectCorrectCorrectCorrectCorrect
74.8%74.8%74.8%74.8%74.8%
IncorrectIncorrectIncorrectIncorrectIncorrect
25.2%25.2%25.2%25.2%25.2%GoldenGoldenGoldenGoldenGolden
66.5%66.5%66.5%66.5%66.5%
NonGoldenNonGoldenNonGoldenNonGoldenNonGolden
33.5%33.5%33.5%33.5%33.5%CorrectCorrectCorrectCorrectCorrect
81.8%81.8%81.8%81.8%81.8%
IncorrectIncorrectIncorrectIncorrectIncorrect
18.2%18.2%18.2%18.2%18.2%mmE5 Nyx-Pretrained Nyx
Figure 6: Comparison of retrieval and answer correctness distributions onNyxQAfor mmE5,Nyx-pretrained, andNyx.
Cursed (2005 film)
clawsOn television, she has played Vi in the series "Buffy the Vampire Slayer" (2003) and Dr. Holly Marten in "Eureka" (2011), and had a recurring role as Charlie Bradbury on "Supernatural" (2012–2015, 2018). She has also acted in movies such as "Bring It On Again" (2004), as well as the Internet musical "Dr. Horrible\'s Sing-Along Blog" (2008). In April 2017, she began appearing as Kinga Forrester in "Mystery Science Theater 3000".cameraFelicia Day
microphoneQuestion: What object is in front of Felicia Day's face?
mmE5
Nyx
Nyx-Pretrained
Qwen-2.5-VL-7B-Instruct
Felicia Day
Figure 7: Case study on MMQA. The top-1 retrieved docu-
ments by mmE5,Nyx-pretrained, andNyxare shown to-
gether with the corresponding answers produced by VLM.
model size, this suggests that generator size is not a reliable predic-
tor of RAG pipeline performance. Effective alignment is crucial in
bridging semantic gaps across VLMs.
5.4 Effect of MRL
In real-world retrieval systems, reducing embedding dimensions
can significantly decrease memory usage and speed up retrieval.
To adapt to varying resource budgets, we incorporate MRL into our
contrastive training framework. MRL ensures the model maintains
meaningful representations across reduced dimensions. We train
the model to generate effective embeddings at four target dimen-
sions: 2048, 1024, 512, and 256, with weights of [1.0, 1.0, 0.2, 0.2]
(before normalization), where 2048 is the default VLM output.
As shown in Table 3, the 1024-dimensional variant achieves
accuracy comparable to the 2048 one while halving storage, and
even the 512- and 256-dimensional versions maintain strong per-
formance. These results highlight MRL’s ability to provide efficient,
resource-aware retrieval with graceful performance degradation
under limited memory or latency budgets.Table 3: Performance ofNyxonNyxQAunder different out-
put dimensions
SettingOutput Embedding Dimension
2048-dim 1024-dim 512-dim 256-dim
Weight1.0000 1.0000 0.2000 0.2000
Accuracy0.8183 0.8100 0.7800 0.7467
5.5 Impact of Retrieved Docs on Generation
To investigate the impact of retrieved documents on generation,
we begin with a case study on MMQA. As shown in Figure 7, we
examines how retrieval influences the produced answers. Unlike
MMEB, which assumes modality-specific similarity computation,
real-world retrieval entails cross-modal relevance estimation across
multimodal documents. In this example, mmE5 retrieves an image-
text pair focused solely on “face,” missing the query subject;Nyx-
pretrained correctly identifies “Felicia Day” but provides the textual
evidence that fails to support the answer; in contrast,Nyxretrieves
the correct entity along with both the proper title and visual infor-
mation, directly grounding the generated response.
Building on these qualitative observations, we further perform
a quantitative analysis on theNyxQAdataset to study the rela-
tionship between retrieval correctness and answer correctness. The
retrieval quality is visualized in Figure 6 using Sankey diagrams. The
results reveal two key trends: (1) higher proportions of golden docu-
ments lead to higher answer accuracy; and (2) even with non-golden
documents, nearly half of the answers remain correct, demonstrat-
ing the robustness of VLMs. These findings suggest that improving
retrievers is crucial not only for ensuring faithful grounding but
also for mitigating noise from irrelevant evidence. Future improve-
ments may arise from modelling VLM preferences on non-golden
evidence, which can sometimes diverge from human intuition.
6 Conclusion
To enable Universal Retrieval-Augmented Generation (URAG) over
arbitrarily mixed-modal questions and corpora, we constructed
NyxQA, the first large-scale and comprehensive dataset that faith-
fully reflected real-world URAG scenarios, where text, images, and
their interleaved combinations naturally coexisted. Building on
this foundation, we introducedNyx, a unified multimodal retriever

Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, UAE
explicitly optimized for such settings.Nyxwas initially pretrained
via contrastive learning with Matryoshka Representation Learn-
ing on a diverse mixture of public and synthetic data, and was
subsequently fine-tuned using feedback from a downstream vision-
language generator, thereby better aligning retrieval relevance with
generation utility. Extensive experiments demonstrated that this
simple yet effective pipeline achieved consistent and substantial
improvements over both unimodal and multimodal baselines across
all modality combinations, underscoring the promise of unified
mixed-modal retrieval for next-generation URAG systems.
References
[1]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang,
Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan
Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu,
Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang
Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report.CoRRabs/2502.13923
(2025). arXiv:2502.13923 doi:10.48550/ARXIV.2502.13923
[2]Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and
Zhicheng Dou. 2025. mmE5: Improving Multimodal Multilingual Embeddings
via High-quality Synthetic Data.CoRRabs/2502.08468 (2025). arXiv:2502.08468
doi:10.48550/ARXIV.2502.08468
[3]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation.CoRRabs/2402.03216
(2024). arXiv:2402.03216 doi:10.48550/ARXIV.2402.03216
[4]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen
Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and
Weipeng Chen. 2025. ReSearch: Learning to Reason with Search for LLMs
via Reinforcement Learning.CoRRabs/2503.19470 (2025). arXiv:2503.19470
doi:10.48550/ARXIV.2503.19470
[5]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen. 2022.
MuRAG: Multimodal Retrieval-Augmented Generator for Open Question An-
swering over Images and Text. InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab
Emirates, December 7-11, 2022, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang
(Eds.). Association for Computational Linguistics, 5558–5570. doi:10.18653/V1/
2022.EMNLP-MAIN.375
[6]Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. 2024. MLLM Is
a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation
via Knowledge-enhanced Reranking and Noise-injected Training.CoRR
abs/2407.21439 (2024). arXiv:2407.21439 doi:10.48550/ARXIV.2407.21439
[7] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel
Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev.
2023. Reproducible Scaling Laws for Contrastive Language-Image Learning. In
IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023,
Vancouver, BC, Canada, June 17-24, 2023. IEEE, 2818–2829. doi:10.1109/CVPR52729.
2023.00276
[8]DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu,
Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun
Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian
Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi
Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie
Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin
Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong
Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang,
Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang,
Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi
Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi
Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu,
Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuip-
ing Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun,
W. L. Xiao, and Wangding Zeng. 2024. DeepSeek-V3 Technical Report.CoRR
abs/2412.19437 (2024). arXiv:2412.19437 doi:10.48550/ARXIV.2412.19437
[9]Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie
Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao
Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025. Agentic Entropy-Balanced Policy
Optimization. arXiv:2510.14545 [cs.LG] https://arxiv.org/abs/2510.14545
[10] Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu
Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. 2025. Tool-Star: Empow-
ering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning.CoRRabs/2505.16410 (2025). arXiv:2505.16410 doi:10.48550/ARXIV.2505.16410
[11] Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang
Zhou, and Jingren Zhou. 2025. Self-play with Execution Feedback: Improving
Instruction-following Capabilities of Large Language Models. InThe Thirteenth
International Conference on Learning Representations, ICLR 2025, Singapore, April
24-28, 2025. OpenReview.net. https://openreview.net/forum?id=cRR0oDFEBC
[12] Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan
Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui
Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025. Agentic Reinforced
Policy Optimization.CoRRabs/2507.19849 (2025). arXiv:2507.19849 doi:10.48550/
ARXIV.2507.19849
[13] Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou,
and Ji-Rong Wen. 2025. Progressive Multimodal Reasoning via Active Retrieval.
InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1,
2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
Pilehvar (Eds.). Association for Computational Linguistics, 3579–3602. https:
//aclanthology.org/2025.acl-long.180/
[14] Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Ji-Rong Wen, and
Zhicheng Dou. 2025. Understand What LLM Needs: Dual Preference Alignment
for Retrieval-Augmented Generation. InProceedings of the ACM on Web Con-
ference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025- 2 May 2025,
Guodong Long, Michale Blumestein, Yi Chang, Liane Lewin-Eytan, Zi Helen
Huang, and Elad Yom-Tov (Eds.). ACM, 4206–4225. doi:10.1145/3696410.3714717
[15] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ah-
mad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sra-
vankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien
Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh
Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra,
Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne
Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song,
Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Maha-
jan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab
AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic,
Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme
Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah
Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M.
Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana
Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer
Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen
Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca,
Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya
Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. 2024.
The Llama 3 Herd of Models.CoRRabs/2407.21783 (2024). arXiv:2407.21783
doi:10.48550/ARXIV.2407.21783
[16] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun
Liu. 2024. Scaling Laws For Dense Retrieval. InProceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(Washington DC, USA)(SIGIR ’24). Association for Computing Machinery, New
York, NY, USA, 1339–1349. doi:10.1145/3626772.3657743
[17] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline
Hudelot, and Pierre Colombo. 2025. ColPali: Efficient Document Retrieval with
Vision Language Models. InThe Thirteenth International Conference on Learning
Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. https:
//openreview.net/forum?id=ogjBpZ8uSi
[18] Xinyu Geng, Peng Xia, Zhen Zhang, Xinyu Wang, Qiuchen Wang, Ruixue Ding,
Chenxi Wang, Jialong Wu, Yida Zhao, Kuan Li, Yong Jiang, Pengjun Xie, Fei
Huang, and Jingren Zhou. 2025. WebWatcher: Breaking New Frontier of Vision-
Language Deep Research Agent.CoRRabs/2508.05748 (2025). arXiv:2508.05748
doi:10.48550/ARXIV.2508.05748
[19] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020, Do-
nia Scott, Núria Bel, and Chengqing Zong (Eds.). International Committee on
Computational Linguistics, 6609–6625. doi:10.18653/V1/2020.COLING-MAIN.580
[20] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of
Large Language Models. InThe Tenth International Conference on Learning
Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
https://openreview.net/forum?id=nZeVKeeFYf9
[21] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning.Trans. Mach. Learn. Res.2022 (2022).
https://openreview.net/forum?id=jKN1pXi7b0
[22] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham,
Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling Up

WWW ’26, April 13–17, 2026, Dubai, UAE Zhang et al.
Visual and Vision-Language Representation Learning With Noisy Text Super-
vision. InProceedings of the 38th International Conference on Machine Learn-
ing, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings of Machine Learning
Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 4904–4916.
http://proceedings.mlr.press/v139/jia21b.html
[23] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng
Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024. E5-V: Universal Em-
beddings with Multimodal Large Language Models.CoRRabs/2407.12580 (2024).
arXiv:2407.12580 doi:10.48550/ARXIV.2407.12580
[24] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu
Chen. 2025. VLM2Vec: Training Vision-Language Models for Massive Multi-
modal Embedding Tasks. InThe Thirteenth International Conference on Learn-
ing Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.
https://openreview.net/forum?id=TE0KOzWYAF
[25] Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao
Zhang, Tong Zhao, Zhao Yang, and Ji-Rong Wen. 2025. FlashRAG: A Modular
Toolkit for Efficient Retrieval-Augmented Generation Research. InCompanion
Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Aus-
tralia, 28 April 2025 - 2 May 2025, Guodong Long, Michale Blumestein, Yi Chang,
Liane Lewin-Eytan, Zi Helen Huang, and Elad Yom-Tov (Eds.). ACM, 737–740.
doi:10.1145/3701716.3715313
[26] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20,
2020, Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for
Computational Linguistics, 6769–6781. doi:10.18653/V1/2020.EMNLP-MAIN.550
[27] Andreas Koukounas, Georgios Mastrapas, Michael Günther, Bo Wang, Scott
Martens, Isabelle Mohr, Saba Sturua, Mohammad Kalim Akram, Joan Fontanals
Martínez, Saahil Ognawala, Susana Guzman, Maximilian Werk, Nan Wang, and
Han Xiao. 2024. Jina CLIP: Your CLIP Model Is Also Your Text Retriever.CoRR
abs/2405.20204 (2024). arXiv:2405.20204 doi:10.48550/ARXIV.2405.20204
[28] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya
Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M.
Kakade, Prateek Jain, and Ali Farhadi. 2022. Matryoshka Representation Learning.
InAdvances in Neural Information Processing Systems 35: Annual Conference on
Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA,
November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle
Belgrave, K. Cho, and A. Oh (Eds.). http://papers.nips.cc/paper_files/paper/2022/
hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html
[29] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. 2025. LLaVE:
Large Language and Vision Embedding Models with Hardness-Weighted Con-
trastive Learning.CoRRabs/2503.04812 (2025). arXiv:2503.04812 doi:10.48550/
ARXIV.2503.04812
[30] Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh,
Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush,
Douwe Kiela, Matthieu Cord, and Victor Sanh. 2023. OBELICS: An Open Web-
Scale Filtered Dataset of Interleaved Image-Text Documents. InAdvances in
Neural Information Processing Systems 36: Annual Conference on Neural Informa-
tion Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 -
16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt,
and Sergey Levine (Eds.). http://papers.nips.cc/paper_files/paper/2023/hash/
e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html
[31] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural In-
formation Processing Systems 33: Annual Conference on Neural Information
Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, Hugo
Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and
Hsuan-Tien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html
[32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. 2023. BLIP-2:
Bootstrapping Language-Image Pre-training with Frozen Image Encoders and
Large Language Models. InInternational Conference on Machine Learning, ICML
2023, 23-29 July 2023, Honolulu, Hawaii, USA (Proceedings of Machine Learning
Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara
Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 19730–19742. https:
//proceedings.mlr.press/v202/li23q.html
[33] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models.CoRRabs/2501.05366 (2025). arXiv:2501.05366 doi:10.48550/
ARXIV.2501.05366
[34] Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu,
Ji-Rong Wen, and Zhicheng Dou. 2025. WebThinker: Empowering Large Rea-
soning Models with Deep Research Capability.CoRRabs/2504.21776 (2025).
arXiv:2504.21776 doi:10.48550/ARXIV.2504.21776[35] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun
Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to Explain:
Multimodal Reasoning via Thought Chains for Science Question Answering.
InAdvances in Neural Information Processing Systems 35: Annual Conference on
Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA,
November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle
Belgrave, K. Cho, and A. Oh (Eds.). http://papers.nips.cc/paper_files/paper/2022/
hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html
[36] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. InFindings of the Association for Computational Linguistics: EMNLP 2023,
Singapore, December 6-10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.).
Association for Computational Linguistics, 5687–5711. doi:10.18653/V1/2023.
FINDINGS-EMNLP.378
[37] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song,
Zhuoma Gongque, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, Runfeng Qiao, Yifan
Zhang, Xiao Zong, Yida Xu, Muxi Diao, Zhimin Bao, Chen Li, and Honggang
Zhang. 2024. We-Math: Does Your Large Multimodal Model Achieve Human-
like Mathematical Reasoning?CoRRabs/2407.01284 (2024). arXiv:2407.01284
doi:10.48550/ARXIV.2407.01284
[38] Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan,
Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, et al .2025. We-Math 2.0:
A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning.
arXiv preprint arXiv:2508.10433(2025).
[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models
From Natural Language Supervision. InProceedings of the 38th International
Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Pro-
ceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang
(Eds.). PMLR, 8748–8763. http://proceedings.mlr.press/v139/radford21a.html
[40] Monica Riedler and Stefan Langer. 2024. Beyond Text: Optimizing RAG with
Multimodal Inputs for Industrial Applications.CoRRabs/2410.21943 (2024).
arXiv:2410.21943 doi:10.48550/ARXIV.2410.21943
[41] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai,
Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. 2021. MultiModalQA:
complex question answering over text, tables and images. In9th International
Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May
3-7, 2021. OpenReview.net. https://openreview.net/forum?id=ee6W5UgQLa
[42] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and
Jun Suzuki. 2025. VDocRAG: Retrieval-Augmented Generation over Visually-Rich
Documents. InCVPR.
[43] Raghuveer Thirukovalluru, Rui Meng, Ye Liu, Karthikeyan K, Mingyi Su, Ping Nie,
Semih Yavuz, Yingbo Zhou, Wenhu Chen, and Bhuwan Dhingra. 2025. Breaking
the Batch Barrier (B3) of Contrastive Learning via Smart Batch Mining.CoRR
abs/2505.11293 (2025). arXiv:2505.11293 doi:10.48550/ARXIV.2505.11293
[44] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554. doi:10.1162/TACL_A_00475
[45] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised
Contrastive Pre-training.CoRRabs/2212.03533 (2022). arXiv:2212.03533 doi:10.
48550/ARXIV.2212.03533
[46] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter,
and Wenhu Chen. 2024. UniIR: Training and Benchmarking Universal Multi-
modal Information Retrievers. InComputer Vision - ECCV 2024 - 18th European
Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXVII
(Lecture Notes in Computer Science, Vol. 15145), Ales Leonardis, Elisa Ricci, Stefan
Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 387–404.
doi:10.1007/978-3-031-73021-4_23
[47] Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, and Ziwei
Liu. 2025. MMSearch-R1: Incentivizing LMMs to Search.CoRRabs/2506.20670
(2025). arXiv:2506.20670 doi:10.48550/ARXIV.2506.20670
[48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng
Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou,
Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao
Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin,
Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su,
Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui,
Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report.
CoRRabs/2505.09388 (2025). arXiv:2505.09388 doi:10.48550/ARXIV.2505.09388
[49] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium,

Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, UAE
October 31 - November 4, 2018, Ellen Riloff, David Chiang, Julia Hockenmaier,
and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, 2369–2380.
doi:10.18653/V1/D18-1259
[50] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zheng-
hao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. VisRAG:
Vision-based Retrieval-augmented Generation on Multi-modality Documents. In
The Thirteenth International Conference on Learning Representations, ICLR 2025,
Singapore, April 24-28, 2025. OpenReview.net. https://openreview.net/forum?id=
zG459X3Xge
[51] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao,
Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu,
Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen
Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and Pengchuan Zhang. 2021.
Florence: A New Foundation Model for Computer Vision.CoRRabs/2111.11432
(2021). arXiv:2111.11432 https://arxiv.org/abs/2111.11432
[52] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu
Su, and Ming-Wei Chang. 2024. MagicLens: Self-Supervised Image Retrieval
with Open-Ended Instructions. InForty-first International Conference on Machine
Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.
[53] Xin Zhang, Ziqi Dai, Yongqi Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie,
Meishan Zhang, Jun Yu, Wenjie Li, and Min Zhang. 2025. Towards Text-Image
Interleaved Retrieval. InProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria,
July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and
Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics,
4254–4269. https://aclanthology.org/2025.acl-long.214/
[54] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long,
Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. GME: Improving
Universal Multimodal Retrieval by Multimodal LLMs.CoRRabs/2412.16855 (2024).
arXiv:2412.16855 doi:10.48550/ARXIV.2412.16855
[55] Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. 2024. VISTA:
Visualized Text Embedding For Universal Multi-Modal Retrieval. InProceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics,
3185–3200. doi:10.18653/V1/2024.ACL-LONG.175
[56] Junjie Zhou, Yongping Xiong, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo
Zhao, Chen Jason Zhang, and Defu Lian. 2025. MegaPairs: Massive Data Synthesis
for Universal Multimodal Retrieval. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025,
Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational
Linguistics, 19076–19095. https://aclanthology.org/2025.acl-long.935/
[57] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu,
Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui
Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang,
Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang,
Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao,
Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun
Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou,
Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai
Wang. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for
Open-Source Multimodal Models.CoRRabs/2504.10479 (2025). arXiv:2504.10479
https://doi.org/10.48550/arXiv.2504.10479

WWW ’26, April 13–17, 2026, Dubai, UAE Zhang et al.
Appendix
A Training Details
We train Nyx based on the Qwen2.5-VL-3B model using a single
node equipped with 8 ×NVIDIA A800-SXM4-80GB GPUs. To enable
efficient fine-tuning, we apply Low-Rank Adaptation (LoRA) [ 20]
with a rank of 8. Each GPU processes a batch of 20 samples, and we
accumulate gradients over 4 steps, resulting in an effective batch
size of 640. To prevent memory overflow when processing multi-
image inputs, we cap the visual input resolution at 400 ×28×28
pixels.
We use DeepSpeed with bf16 mixed-precision training and en-
able gradient checkpointing for memory efficiency. The optimizer
is AdamW, combined with a linear learning rate scheduler. The
base learning rate is set to 1e-5, with a warmup ratio of 0.05, and a
maximum gradient norm of 5.0. The contrastive loss function uses
a temperature of 0.02 and a negative sampling ratio of 1.
For efficient memory optimization, we employ DeepSpeed’s
ZeRO optimization at stage 2, which partitions model states across
devices to significantly reduce memory consumption. This allows
us to train larger models without overflow. ZeRO stage 2 also en-
ables communication optimizations such as allgather and reduce-
scatter, improving parallelism and computational efficiency. These
configurations, combined with gradient checkpointing and mixed-
precision training, work together to maximize both performance
and memory efficiency during training.
B Baseline Retriever Models
In this section, we present the baseline retriever models employed
in our main experiments. These models capture several recent
advances in text and multimodal retrieval, and provide strong base-
lines for assessing the effectiveness of our proposed method.
E5[45] is a series of cutting-edge text embeddings that perform
exceptionally well across a variety of tasks. The model is trained us-
ing a contrastive approach, with weak supervision signals derived
from a carefully curated large-scale text pair dataset. It demon-
strates strong performance both in zero-shot scenarios and after
fine-tuning.
CLIP[ 39] is a powerful multimodal model developed by OpenAI
that learns visual and textual representations jointly. It is trained
using a large dataset of image-text pairs in a contrastive manner,
enabling it to understand images and texts in a shared embedding
space. This approach allows CLIP to perform a variety of tasks, such
as zero-shot image classification, image search, and text-to-image
retrieval, without task-specific fine-tuning.
VLM2Vec[ 24] is a versatile multimodal embedding model designed
to convert any state-of-the-art VLM into a unified embedding space.
It employs a contrastive training framework on the Massive Multi-
modal Embedding Benchmark (MMEB), which encompasses four
meta-tasks—classification, visual question answering, multimodal
retrieval, and visual grounding—across 36 datasets. Unlike models
such as CLIP or BLIP, which process text and images independently,
VLM2Vec integrates both modalities based on task-specific instruc-
tions to produce fixed-dimensional vector representations.
mmE5[ 2] is a multimodal multilingual embedding model that en-
hances performance by leveraging high-quality synthetic datasets.These datasets encompass a wide range of tasks, modality combi-
nations, and languages, and are generated using a deep thinking
process within a single pass of a VLM. The synthetic data incorpo-
rates real-world images with accurate and relevant texts, ensuring
fidelity through self-evaluation and refinement. The model’s effec-
tiveness underscores the potential of high-quality synthetic data in
improving multimodal multilingual embeddings.
VisRAG-Ret[ 50] is a VLM-based retriever component of the Vis-
RAG framework, designed to enhance retrieval-augmented gener-
ation by directly processing document images. Unlike traditional
text-based RAG systems that rely on parsed text, VisRAG-Ret uti-
lizes a VLM to embed document images, preserving the visual layout
and content. It employs a bi-encoder architecture, mapping both
the query and document images into a shared embedding space,
facilitating efficient retrieval.
C Dataset Details
In this section, we describe the datasets used in our experiments,
covering both text-only and multimodal benchmarks. The text-
only datasets are employed to evaluate retrieval and reasoning in
purely linguistic settings, while the multimodal ones are used to as-
sess cross-modal understanding and MRAG performance. Together,
these datasets provide a comprehensive evaluation framework for
analysing the effectiveness and generalization of our proposed
method.
HotPotQA[ 49] is a popular dataset for multi-hop question answer-
ing, comprising questions that require synthesizing information
across multiple Wikipedia articles. The dataset includes complex
query types, such as comparison and bridge questions. It contains
90,447 training samples, and we follow ARPO [ 9,12] use a held-out
validation set with 250 examples for evaluation.
2WikiMultihopQA[ 19] is a large-scale dataset aimed at multi-hop
reasoning, constructed by combining structured knowledge from
Wikidata with unstructured passages from Wikipedia. It features
diverse question formulations and annotated reasoning chains to
facilitate explainable multi-step QA. The dataset includes 15,000
training samples, and our experiments use the test set consisting
of 250 examples.
Bamboogle[ 36] consists of manually curated multi-hop questions
designed to test compositional reasoning. Some questions demand
up to four inference steps, presenting a significant challenge in
integrating information across multiple supporting facts. It provides
only a test set, which we use for evaluation and which contains 125
examples.
MuSiQue[ 44] focuses on sequential multi-hop inference, where
each reasoning step depends on the output of the previous one.
This dependency-based structure increases the difficulty of the task.
The dataset comprises 19,938 training examples, and we use its
development set with 250 held-out samples for evaluation.
For the four text-only datasets above, we construct two separate
Wikipedia-derived corpora—one for training and one for evaluation.
To build the training corpus, we aggregate all training questions
and retrieve their top-20 relevant Wikipedia passages using the
E5 retriever over the full Wikipedia dump. The retrieved passages
are then deduplicated to form the final training corpus. Similarly,

Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation WWW ’26, April 13–17, 2026, Dubai, UAE
the evaluation corpus is constructed by collecting all test ques-
tions and retrieving their top-20 Wikipedia passages, followed by
deduplication. The training corpus is used during the feedback col-
lection stage, whereNyx-pretrained retrieves relevant passages
to construct the “downstream VLM preference dataset” for fine-
tuning. The evaluation corpus is used for benchmarking on the four
text-only datasets during the final testing stage.
MultimodalQA[41] is a challenging question-answering dataset
that necessitates joint reasoning across text, tables, and images. It
includes 23,817 training examples and 2,411 testing examples. We
combined text, tables, and images to create a large mixed-modal
corpus containing 285,370 instances for this MMQA task.
ScienceQA[ 35] is a large-scale multimodal science question dataset
that annotates answers with detailed lectures and explanations.
Each question is accompanied by context, either in the form of
natural language or an image. For this dataset, we constructed two
corpora: one consists of all the lectures appearing in the dataset,
with duplicates removed to form the lecture corpus; the other con-
tains the question-answer pairs from the training set, forming the
example QA corpus. During testing, we retrieve one lecture and
two example QAs to serve as external support information. The
training set contains a total of 12,726 examples, while the testing
set has 4,241 examples.
D Details for Raw QA Generation
In this section, we provide additional details on the raw QA genera-
tion process described in Subsection 3.2. Specifically, for a randomly
sampled subset of 10,000 document instances from Cmix, we em-
ploy two generation strategies depending on whether a document
contains images. Using the InternVL3-78B model, we instruct it to
produce up to five context-independent question–answer pairs for
each document.
Prompt for Text-Only Documents.The following prompt is used
when the input document contains only textual content:
Instructions for text QA Pair Generation
Given a text, please analyze the content of the text and raise
no more than five questions along with their corresponding
answers.
Requirements:
1. The question must be independent of the context, that is, it
cannot rely on background information that is not mentioned.
2. The questions raised can be answered in concise language.
Example:
Text:They broke the law, but it’s not a felony. It’s an act
of love. It’s an act of commitment to your family. I honestly
think that that is a different kind of crime that there should
be a price paid, but it shouldn’t rile people up that people are
actually coming to this country to provide for their families.
21 thoughts on “Unethical Quote of the Month: Jeb Bush”
Incorrect question:What does the speaker think about this
crime? (Without specifying who the "speaker" is)Correct question:What type of crime does Jeb Bush describe
as being committed by people coming to the country to provide
for their families?
Answer:Jeb Bush describes it as an act of love and commit-
ment to family, not a felony.
Output format:[Q1:... ,A1:... ], [Q2:... ,A2:... ], ...
Prompt for Multimodal QA.When a document contains both
text and images, the model is guided with the following prompt.
Instructions for multimodal QA Pair Generation
You are given a document containing text and images, please
analyze the content and raise no more than five questions
along with their corresponding answers.
Requirements:
1. The question must be independent of the context, that is, it
cannot rely on background information that is not mentioned.
2. You can ask questions about the images in the document,
but you need to clearly indicate them like:“Based on the im-
age,<image2> , ...” or “Considering both images, <image1> and
<image3>, ...” etc.
3. The questions raised can be answered in concise language.
Example:
Document: <|image|> The statement by Jeb Bush has its
sunny side, I suppose: with any luck, it should ensure that
we don’t have a Bush-Clinton contest in 2016. Maybe that
was Jeb’s intent. Otherwise, his comments are irresponsible
attacks on the rule of law, common sense, fairness and national
sovereignty.
There are means by which we can control our border better
than we have. And there should be penalties for breaking the
law. But the way I look at this — and I’m going to say this, and
it’ll be on tape and so be it. The way I look at this is someone
who comes to our country because they couldn’t come legally,
they come to our country because their families — the dad
who loved their children — was worried that their children
didn’t have food on the table. And they wanted to make sure
their family was intact, and they crossed the border because
they had no other means to work to be able to provide for
their family.
Incorrect question:Considering both the text and <image1> ,
what might be the context of Jeb Bush’s speech? (The question
cannot be answered without context)
Correct question:In the image, <image1> , what might be
the context of Jeb Bush’s speech?
Incorrect question:What is the main concern expressed
about Jeb Bush’s comments? (Without specifying what the
"comments" is)
Correct question:What is the main concern expressed about
Jeb Bush’s comments “someone who comes to our country
because they couldn’t come legally, they come to our country
because their families”?
Output format:[Q1:... ,A1:... ], [Q2:... ,A2:... ], ...
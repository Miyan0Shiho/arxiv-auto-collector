# Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation

**Authors**: Ruofan Hu, Shengyang Xu, Minjie Hong, Xiaoda Yang, Sashuai Zhou, Ke Lei, Tao Jin, Zhou Zhao

**Published**: 2026-08-31 02:30:11

**PDF URL**: [https://arxiv.org/pdf/2608.30163v1](https://arxiv.org/pdf/2608.30163v1)

## Abstract
Real-world knowledge resides in multimodal documents, necessitating retrieval-augmented generation (RAG) for accurate question answering. However, existing multimodal RAG models are primarily designed for single-image or closed-document settings and exhibit limited accuracy in realistic multi-image scenarios. Moreover, processing numerous retrieved images incurs substantial computational overhead from irrelevant visual tokens. To address these challenges, we introduce DocLongRAG, a large-scale dataset of 343K question--answer pairs, each associated with an average of 37.4 retrieved images to reflect authentic RAG workflows. Building on this dataset, we propose Doc-REFRAG, a question-guided framework that compresses visual tokens into coarse chunks and selectively expands question-relevant ones via a lightweight RL-based selector. Experiments on six benchmarks show that Doc-REFRAG outperforms eleven strong baselines, achieving state-of-the-art accuracy with significantly lower inference latency. Our resources are available at https://github.com/Collab-Gen/Doc-REFRAG.

## Full Text


<!-- PDF content starts -->

Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented
Generation
Ruofan Hu*, Shengyang Xu*, Minjie Hong, Xiaoda Yang,
Sashuai Zhou, Ke Lei, Tao Jin, Zhou Zhao†
Zhejiang University
ruofanhu@zju.edu.cn
Abstract
Real-world knowledge resides in multimodal
documents, necessitating retrieval-augmented
generation (RAG) for accurate question an-
swering. However, existing multimodal RAG
models are primarily designed for single-image
or closed-document settings and exhibit lim-
ited accuracy in realistic multi-image scenar-
ios. Moreover, processing numerous retrieved
images incurs substantial computational over-
head from irrelevant visual tokens. To ad-
dress these challenges, we introduce DocLon-
gRAG, a large-scale dataset of 343K question–
answer pairs, each associated with an aver-
age of 37.4 retrieved images to reflect authen-
tic RAG workflows. Building on this dataset,
we propose Doc-REFRAG, a question-guided
framework that compresses visual tokens into
coarse chunks and selectively expands question-
relevant ones via a lightweight RL-based se-
lector. Experiments on six benchmarks show
that Doc-REFRAG outperforms eleven strong
baselines, achieving state-of-the-art accuracy
with significantly lower inference latency. Our
resources are available at https://github.
com/Collab-Gen/Doc-REFRAG.
1 Introduction
Multimodal large language models (MLLMs) have
rapidly developed, achieving remarkable perfor-
mance in various visual understanding tasks (Lu
et al., 2024; Bai et al., 2025; Chen et al., 2024b;
Yao et al., 2024; Li et al., 2024a). Despite these
advances, current MLLMs remain limited by static
parametric knowledge. To address this limita-
tion, RAG enhances MLLMs with retrieved evi-
dence from external corpora, thereby expanding
factual coverage and mitigating hallucination (Ar-
slan et al., 2024). Crucially, a significant portion of
real-world knowledge resides in visually rich docu-
ments, which often contain textual passages, charts,
*Equal contribution.
†Corresponding author.
InternVL3.5 -8B
VDocGenerator -3BQwen3VL -8B
LLAVA -Mini -7BDocopilot -8BDocOwl2 -7BDoc-REFRAG -7B
5001,000
2,000Diameter -# Vis. TokensLLAVA -OV-8B
EPIC-7BGeneral MLLMs
Document -aware MLLMs
MLLMs with compression
Doc-REFRAGFigure 1: Doc-REFRAG achieves the best accuracy–
efficiency trade-off among MLLMs.
and figures. However, existing MLLMs trained on
individual images struggle with multi-image rea-
soning (Ma et al., 2024; Dong et al., 2025a; Tito
et al., 2023). Moreover, substantial visual tokens
from retrieved images impose significant computa-
tional overhead (Ye et al., 2023; Liu et al., 2024a).
To handle multi-image inputs, recent works con-
struct long-document training corpora (Van Lan-
deghem et al., 2023; Hu et al., 2024a; Tanaka et al.,
2025; Duan et al., 2025) and apply curriculum
learning to extend capacity from single to multi-
image layouts (Hu et al., 2024b; Duan et al., 2025).
However, these works generally assume a closed-
domain setting where all images belong to a sin-
gle, coherent document. RAG, however, retrieves
images across diverse documents that are hetero-
geneous, redundant, and contradictory. As shown
in (Dong et al., 2025b; Cuconasu et al., 2024), this
distribution shift severely degrades MLLM gener-
alization to the fragmented and noisy images en-
countered during RAG inference.
Orthogonal to the reasoning challenge, existing
compression methods fall into two types. Training-
free approaches typically rely on either token im-
portance (Chen et al., 2024a; He et al., 2024) or re-
dundancy estimation (Tan et al., 2025; Wang et al.,
2025) to prune visual tokens. In contrast, training-
aware methods introduce a linear projector to refine
arXiv:2608.30163v1  [cs.IR]  31 Aug 2026

the limited coarse tokens with fine-grained visual
details (Li et al., 2025; Hu et al., 2024b). Both cat-
egories prioritize visually complex patches, assum-
ing such regions are more informative for down-
stream tasks. This assumption fails in RAG sce-
narios, where visually complex patches are often
question-irrelevant. Consequently, processing mul-
tiple retrieved images still incurs numerous redun-
dant visual tokens and substantial prefill latency (Li
et al., 2024b).
To address these problems, we introduce Doc-
LongRAG, a new multimodal document dataset.
Compared to existing work (Hu et al., 2024a;
Tanaka et al., 2025; Duan et al., 2025; Luo et al.,
2025), this dataset has the following features. (1)
High-Quality Retrieval Contexts: semantically
similar images are explicitly interleaved with hard
negative distractors to simulate real-world RAG
inputs. (2)Long-Range Reasoning: each question
is associated with an average of 37.4 images, en-
abling training on substantially longer sequences.
(3)Broad Format Coverage: 343,474 question–
answer pairs grounded in 3.3M document images
spanning PDFs, slides, and posters, supporting ro-
bust generalization across document formats.
Building on DocLongRAG, we propose Doc-
REFRAG, a framework for efficient multimodal
RAG decoding. Unlike pruning-based methods,
Doc-REFRAG adopts a “compress-and-expand”
strategy: visual tokens are first compressed into
non-overlapping chunks. A lightweight question-
conditioned selector then re-expands only the most
relevant chunks into full-resolution tokens, recover-
ing fine-grained semantics precisely where needed
without the context loss of static pruning.
Integrating mixed sequences of chunked and
original tokens requires a specialized training strat-
egy. Doc-REFRAG adopts a three-stage frame-
work: single-image reconstruction for chunk se-
mantic fidelity, multi-image continual pretraining
for mixed-sequence processing, and RL-based se-
lector training using accuracy as reward. As shown
in Fig. 1, this cuts response time while improving
accuracy over prior work.
The contributions are summarized as follows:
•We introduce DocLongRAG, a large-scale multi-
modal dataset reflecting the noisy, long-context
nature of real-world retrieval.
•We propose Doc-REFRAG, which rethinks vi-
sual token compression as dynamic, question-
guided selective expansion, via a three-stage cur-riculum with an RL-based selector.
•Experiments on six benchmarks show that our
method outperforms eleven existing approaches.
2 Related Work
2.1 Document Understanding Models in RAG
RAG enhances MLLMs by retrieving external
knowledge. In multimodal document RAG, exist-
ing research focuses on retriever design (Yu et al.,
2024; Yan et al., 2025; Zhang et al., 2025a; Hu
et al., 2025; Yang et al., 2025; Hu et al., 2026)
while employing general-purpose MLLMs as gen-
erators. Early VDU MLLMs (Appalaraju et al.,
2024; Wang et al., 2023) and subsequent multi-
image variants (Feng et al., 2024; Duan et al., 2025;
Liu et al., 2025) primarily target coherent, single-
source documents. In contrast, real-world RAG
involves retrieving fragmented and contradictory
images from diverse sources, creating a distribution
shift that current models struggle to handle. While
some works reason over massive document piles
(Chen et al., 2025), they overlook the efficiency
required for interactive generation.
2.2 Visual Token Compression
Multi-image inputs cause a token explosion that
degrades inference efficiency, motivating visual to-
ken compression. These methods fall into training-
free and training-aware approaches. Training-free
methods typically adopt importance-based strate-
gies (Chen et al., 2024a; Zhang et al., 2024c), prun-
ing tokens by attention scores, and redundancy-
based approaches (Jiang et al., 2025; Wen et al.,
2025a), merging similar tokens by feature simi-
larity. Training-aware methods adopt lightweight
component replacements (Dai et al., 2023; Dong
et al., 2024; Li et al., 2024c). Recent examples
include MQT-LLaV A (Hu et al., 2024c), which
uses a dynamic Q-Former, and TokenPacker (Li
et al., 2025) and DocOwl2 (Hu et al., 2024b), which
use coarse-to-fine condensation. However, these
methods prioritize visually salient regions (Zhang
et al., 2025b), misaligning with question relevance
in RAG. A structured comparison across six design
dimensions is in App. A.
3 DocLongRAG
Conventional VDU datasets focus on individual or
coherent images, offering limited context. Con-
versely, RAG requires reasoning across long, noisy,

Single DocumentDocument
ExtensionImage 1
Image 2
…
Image NImage 1
Image 2
Image NHard Negative
Mining
Hard Negative
Mining
Hard Negative
MiningNeg 1.1 Neg 1.K …
Neg 2.1 Neg 2.K …
Neg N.1 Neg N.K ……Figure 2: Process of creating DocLongRAG.Doc-750k OpenDocVQA DocLongRAG
Context-Dependent✗ ✓ ✓
Online-Independent✗ ✓ ✓
Answerable✓ ✗ ✓
Ethically Reviewed✗ ✗ ✓
Retrieval-Noisy✗ ✗ ✓
#QA Pairs 758,000 43,474 343,474
#Images (Pages) 3,100,000 206,267 3,306,267
#Avg. Images per Question 4.1 2.1 37.4
Table 1: Comparison of related datasets.
and conflicting retrieved inputs. DocLongRAG
simulates the scale and noise of practical retrieval,
engineered for training stability.
3.1 Dataset Collection and Filtering
To support RAG training, we curate a diverse VDU
corpus integrating Doc-750K (Duan et al., 2025)
and OpenDocVQA (Tanaka et al., 2025), which ag-
gregate English data from 15 open-source datasets
(see App. B) and scientific documents. To address
instances conflicting with RAG requirements or
ethical guidelines, we filter four sample categories:
•Context-independent questions: Factoid ques-
tions answerable from parametric knowledge
alone, without any retrieved document image
(e.g., “Who won the Turing Award in 2018?”).
•Online search questions: Questions requiring
external web resources beyond the document con-
text (e.g., “What is the author’s Google Scholar
citation count?”), violating the assumption that
all information resides in the documents.
•Unanswerable questions: Questions lacking fac-
tual grounding in the documents, excluded as
they do not support document-based reasoning.
•Ethically problematic content: Questions with
personally identifiable information, offensive lan-
guage, harmful instructions, or sensitive imagery,
violating ethical standards.
This filtering reduces the pool by 18.2%, yield-
ing a curated corpus Cenforcing retrieval depen-
dence. The corpus retains diverse formats (PDFs,
web screenshots, slides) and answer types (extrac-
tive, abstractive), providing a challenging training
foundation. This filtering applies only to the train-
ing corpus, while the evaluation benchmarks retain
unanswerable and context-independent questions.
3.2 Negative Document Extension
We construct DocLongRAG to train VDU models
on multiple document images under realistic re-
trieval noise. Inspired by (Gao et al., 2025), weinterleave relevant images with hard-negative dis-
tractors fromCto form extended input sequences.
Specifically, each instance in Ccomprises
a question q, relevant document images D=
{d1, . . . , d N}, and an answer a. As shown in
Fig. 2, for each image dj∈D , we retrieve the
top-kmost similar images from C\ D using the
document-aware retriever (Günther et al., 2025).
However, due to the semantic richness of document
content, some images ∈C\ D may still contain
answer-supporting evidence. To avoid mislabel-
ing such ambiguous candidates as negatives, we
manually evaluate 500 randomly sampled queries
and find that on average the top 2.47 retrieved
images remain relevant (App. C). We therefore
exclude the top-3 results and define the next m
images (ranked 4 onward) as hard negatives for
dj, where mis uniformly sampled from [20,30] .
We construct the final sequence by interleaving
each relevant image djwith its hard negatives as
l=
{dj, n1
j, . . . , nm
j}N
j=1
. This yields a long,
redundant, and sometimes conflicting multimodal
context, closely mirroring real-world RAG. Apply-
ing this to all instances in Cyields the DocLon-
gRAG corpus. We emphasize that such constructed
noise is used only for training, while evaluation
relies on the noise produced by a real retriever over
real benchmark documents (Sec. 5).
Through ablation studies on the placement posi-
tion of relevant images (i.e., the meta-chunk) within
the hard-negative sequence (App. D), we show that
placing relevant document images before hard neg-
atives consistently improves model performance
during training. This strategic ordering helps estab-
lish longer dependencies, enhancing the effective-
ness of DocLongRAG.
3.3 Comparison with Related Datasets
Tab. 1 presents a statistical comparison of DocLon-
gRAG with existing VDU datasets. DocLongRAG
differs from prior work along three key dimensions,
establishing a more challenging and realistic setting
for training. First, it aggregates instances from mul-

tiple source datasets, covering diverse document
image formats and question types. Second, while
Doc-750K contains context-independent questions
and OpenDocVQA comprises unanswerable ones,
DocLongRAG requires all answers to be grounded
in the provided document images, enforcing a
retrieval-dependent setting. Third, whereas Open-
DocVQA grounds questions in individual images
and Doc-750K in coherent document images, Doc-
LongRAG associates each question with an average
of 37.4 images, intentionally curated to include re-
dundancy, noise, and conflicting evidence.
4 Doc-REFRAG
4.1 Model Architecture
The overall architecture of Doc-REFRAG is illus-
trated in Fig. 3, consisting of four key components:
(1) a document-tailored vision encoder built upon
DocOwl2, (2) a two-layer MLP projector for chunk
embedding, (3) a lightweight RL-based selector for
dynamic token restoration, and (4) a decoder-only
language model. A consolidated overview of all
components, training stages, and inference-time
operations is provided in App. E.
Given a question consisting of qtokens, denoted
ast= [t 1, . . . , t q], each retrieved document image
is encoded by the vision encoder, which maps the
input image to a compact sequence of m= 324
visual tokens. Concatenating these visual tokens
from all nretrieved images yields the initial visual
token sequence in the inputv= [v 1, . . . , v n×m].
In RAG, only a small subset of regions is seman-
tically relevant to the question, rendering most vi-
sual tokens redundant. Existing compression meth-
ods, however, are typically question-agnostic: they
prioritize visually dense regions (e.g., text-heavy
blocks) and discard sparser ones (e.g., blank areas),
yet visual density correlates poorly with semantic
relevance.
To address this mismatch, we propose a question-
guided compression strategy that prioritizes seman-
tic relevance over visual appearance. Specifically,
Doc-REFRAG first partitions the visual token se-
quence of each retrieved image into L=m/k
non-overlapping chunks, which serve as units for
coarse-level semantic modeling. For the i-th image,
we define its chunk group G(i)={C1
i, . . . , CL
i},
where each chunk Cj
iconsists of ktokens from vi.
This chunking preserves the 2D arrangement of the
page. We chunk layout-aware tokens rather than
raw patches: each token from the encoder summa-rizes a fixed region of the page and has aggregated
global context through self-attention. Since consec-
utive tokens in raster order are spatially adjacent,
chunking them keeps each group localized on the
page rather than scattered across arbitrary posi-
tions. Expansion then restores the original tokens
of exactly the same region, reinstating fine-grained
spatial detail where the selector fires. In this way,
two-dimensional reasoning is inherited from the
encoder, while our compression reuses rather than
rebuilds this spatial structure.
A two-layer MLP projector ϕthen projects
each chunk into a single d-dimensional embedding
cj
i=ϕ(Cj
i)∈Rd, where ddenotes the hidden
dimension of the decoder’s input embedding space,
preserving semantic content while reducing length.
The chunk embeddings from all retrieved images
are concatenated with question tokens to form the
input sequence:
z= [t 1, . . . , t q,c1
1, . . . ,cL
n]∈R(q+n×L)×d,
which is fed into a decoder-only foundation model
to generate the answery∼p θ(y|z).
To retain fine-grained details that coarse embed-
dings may obscure, a lightweight RL-based selec-
tor dynamically identifies critical chunks and ex-
pands them back to their original kconstituent
tokens during decoding. Let δj
i∈ {0,1} denote
whether the j-th chunk of the i-th image should
be expanded. The final input sequence becomes
z′= [t 1, . . . , t q,˜c1
1, . . . , ˜cL
n], where ˜cj
i=Cj
i∈
Rk×difδj
i= 1and˜cj
i=cj
i∈Rdotherwise. This
mechanism balances computational efficiency and
semantic fidelity based on the question.
4.2 Model Training
To align the encoder and decoder with z′, we adopt
a three-stage protocol: Single-Image Reconstruc-
tion, Multi-image Continual Pretraining (CPT), and
RL-Based Selector Training.
Single-Image Reconstruction.We freeze the de-
coder and train the encoder and MLP projector on
DocStruct4M (Hu et al., 2024a) with two objec-
tives: (1) aligning chunk embeddings with the de-
coder’s input embedding space, and (2) preserving
original visual semantics. We formulate two com-
plementary tasks: text parsing, where the model
recognizes all textual content in a document image,
and text localization, where it extracts text from
a spatially specified region. The encoder is then
frozen for subsequent stages.

How much of online GMV in 
2016 is projected to be 
influenced by women?
Document Corpus
Vision Token EmbeddingChunk EmbeddingDecoder-only Language Model Lora
Image Embeddings from 
Encoder
Retrieve
Top K
TrainableLightweight RL-based Selector
      Projector       Women influence 35% of online GMV in 2016.
Frozen Vision Token Chunk
Patch-level Visual Encoding
Figure 3: Architecture of Doc-REFRAG. Retrieved document images are encoded patch-level into chunk embed-
dings. A query-aware RL-based selector identifies informative chunks, which are expanded to fine-grained tokens
before being fed with the query into a decoder-only language model.
Multi-image Continual Pretraining.In the CPT
stage, we employ curriculum learning to train the
decoder on increasingly complex datasets, with two
objectives: (1) scaling from single- to multi-image
settings, and (2) enabling effective processing of
hybrid sequences that interleave chunk and visual
token embeddings.
Chunk expansion is supervised by pseudo-labels
δj
iderived from ColQwen (Faysse et al., 2024)
question-to-patch similarity scores. Since our en-
coder and ColQwen share the same normalized
image coordinate convention, a deterministic geo-
metric mapping from each coarse chunk to the fine
patches it subsumes is established (App. F; robust-
ness in App. G). At inference time, δj
iis predicted
by the RL-based selector.
Building upon this pseudo-labeling strategy, we
begin with OpenDocVQA (Tanaka et al., 2025), a
single-image VDU dataset: for each question, we
expand the top 20% of chunks ranked by ColQwen
similarity scores, enabling the decoder to fuse fine-
grained visual details with coarse chunk summaries
within a single image.
Next, we train on Doc-750K, a multi-image
VDU dataset comprising documents split across
multiple images, applying the same 20% expansion
per image. The resulting longer hybrid sequences
challenge the decoder to perform cross-image rea-
soning under extended context.
Finally, we train on DocLongRAG, which simu-
lates realistic RAG conditions. Unlike Doc-750K,
DocLongRAG comprises images from heteroge-
neous sources exhibiting redundancy, semantic
overlap, and contradiction. We reduce expansion
to 10% and additionally activate randomly selectedirrelevant chunks, introducing a dual challenge:
noisy heterogeneous context combined with sparse
and imperfect token expansion, thereby better align-
ing the model with practical RAG deployment.
RL-Based Selector Training.The RL-based se-
lector is a lightweight two-layer transformer that
computes selection logits over all chunk embed-
dings. Given the question tand chunk sequence
{cj
i}, the policy πθsamples binary decisions δj
i∈
{0,1} to determine whether each chunk should be
expanded to its original ktokens. The policy is
constrained to select exactly T=⌊p·n·L⌋ chunks
for expansion, where p∈(0,1] controls the budget.
We train a dedicated selector for each value of k
and demonstrate the influence in App. H. In our ex-
periments, pis uniformly sampled from [0.1,0.3]
during training and fixed at 0.2 during inference,
which is identified as optimal in App. I.
We train the selector on DocLongRAG using a
GRPO-style policy optimization framework with
clipped PPO updates. The reward signal is the an-
swer accuracy of the frozen decoder when process-
ing the input sequence {δj
i}constructed from the
selector’s choices. As shown in Sec. 5.2, this yields
lower inference latency than ColQwen with com-
petitive accuracy. Full training details, including
trajectory sampling and advantage normalization,
are provided in App. J.
5 Experiments
5.1 Experimental Setup
Dataset & Benchmarks.For training, we
perform single-image reconstruction with Doc-
Struct4M, followed by multi-image CPT on Open-

Model Size TokenVChartQA SlideVQA InfoVQA DUDE Vidore MMDocIR #Acc. #TTFT #TTIT
General MLLMs
InternVL3.5 8B∼2,304 56.8 53.1 48.1 45.8 42.1 51.5 49.5 2.78 0.92
LLaV A-OneVision 8B∼1,920 52.1 50.9 46.3 47.2 42.7 49.0 48.0 4.030.05
MiniCPM-V-2.6 8B∼576 54.8 52.3 48.5 46.4 41.1 48.8 48.6 5.80 0.12
Qwen3-VL 8B∼512 58.2 59.2 55.7 52.5 43.5 52.4 51.3 5.910.05
MQT-LLaV A†7B 256 35.7 39.3 33.4 40.5 37.4 34.2 36.7 3.430.05
EPIC†7B 256 39.7 41.1 42.7 32.4 33.3 38.2 37.9 3.42 0.06
LLaV A-Mini†7B 144 35.4 38.5 44.6 37.5 39.1 40.2 39.2 2.67 0.06
Document-aware MLLMs
Docopilot 8B∼2304 58.7 56.6 49.4 48.6 45.7 46.5 50.91.812.25
VDocGenerator 3B∼1920 53.3 42.8 43.2 37.9 34.6 36.8 41.4 9.20 6.40
DocOwl2†7B∼324 57.5 53.2 45.2 44.7 47.5 45.2 48.9 3.94 0.07
TokenPacker†7B 144 46.4 39.9 41.3 33.5 36.7 35.9 37.5 4.430.05
Ours
Doc-REFRAG 7B ∼150 61.2 60.8 58.7 53.2 57.3 54.9 57.5 2.57 0.07
Table 2: Comparison with general MLLMs and document-aware MLLMs on six benchmarks underRAG settings.
The ‘†’ indicates models that use visual token compression. ‘TokenV’ denotes the average number of visual tokens
per image. ‘Bold’ indicates SOTA performance within the group and ‘ Underline ’ indicates second-best performance
within the group. Accuracy (%↑), TTFT, and TTIT are reported in seconds (↓). # denotes averaged metrics.
DocVQA, Doc-750K, and DocLongRAG. The se-
lector is also trained on DocLongRAG. All test-
benchmark images are excluded to prevent data
leakage. Benchmarks include: ChartQA (Masry
et al., 2022), SlideVQA (Tanaka et al., 2023), In-
foVQA (Mathew et al., 2022), DUDE (Van Lan-
deghem et al., 2023), ViDoRe-v2 (Macé et al.,
2025), and MMDocIR (Dong et al., 2025a).
Baselines.We compare Doc-REFRAG with
MLLMs that support multi-image input, which
excludes methods designed for single-image set-
tings. First, we include general MLLMs: Qwen3-
VL (Bai et al., 2025), MiniCPM-V-2.6 (Yao et al.,
2024), LLaV A-OneVision (Li et al., 2024a), and
InternVL3.5 (Chen et al., 2024b). Second, we
evaluate visual token compression variants: MQA-
LLaV A (Hu et al., 2024c), EPIC (Wen et al.,
2025b), and LLaV A-Mini (Zhang et al., 2025b). Fi-
nally, we include models designed for document un-
derstanding: VDocGenerator (Tanaka et al., 2025),
Docopilot (Duan et al., 2025), TokenPacker (Li
et al., 2025), and DocOwl2 (Hu et al., 2024b).
Implementation details.Doc-REFRAG is built
upon DocOwl2, adopting its encoder and decoder
as the backbone. In single-image reconstruction,
we fine-tune the linear vision-to-text alignment
module in the encoder, along with a two-layer MLP
projector (hidden dimension 4096, matching the
decoder’s output dimension), using a peak learn-
ing rate of 2×10−4. In multi-image CPT, we
employ LoRA (rank = 64) (Hu et al., 2022) to fine-
tune the decoder using AdamW ( 1×10−5learn-
ing rate, 4% linear warmup, cosine decay) with a
batch size of 64. Both stages use cross-entropy lossover predicted tokens. The RL-based selector is
trained with GRPO (clip ratio = 0.2). All stages
train for 2 epochs. Experiments are conducted on
8×NVIDIA RTX 3090 GPUs. Per-stage datasets
and trainable parameters are detailed in App. K.
5.2 Main Result
We evaluate Doc-REFRAG against eleven state-of-
the-art MLLMs across six VDU benchmarks. To as-
sess model robustness under realistic RAG settings,
weretrieve the top-20 imagesfor each question
from its source benchmark with the state-of-the-
art retriever Jina-Embeddings-V4 (Günther et al.,
2025), and provide them as input to all MLLMs.
These top-20 candidates cover 93.4% of the gold
images on average across the six benchmarks (Re-
call@20), indicating that retrieval recall is not the
bottleneck in our evaluation. We present quanti-
tative results in this section and defer qualitative
examples to App. L.
Quality analysis.As shown in Tab. 2, Doc-
REFRAG with k= 3 outperforms both general
MLLMs and specialized long-document MLLMs.
This gain stems from two design aspects. First,
these MLLMs are typically trained on short, co-
herent documents. This diverges from the noisy
retrieval contexts in evaluation, which include re-
dundant, conflicting, or irrelevant content that im-
pairs reasoning. In contrast, Doc-REFRAG trains
on DocLongRAG, which is explicitly designed to
mimic such noisy retrieval contexts. Second, most
MLLMs degrade with input length due to attention
saturation, while Doc-REFRAG’s question-guided
compression dynamically reduces tokens while pre-

serving task-critical information.
Efficiency Analysis.We measure TTFT and
TTIT latency under a batch size of 1 with 20
retrieved images per question (accuracy–latency
trade-offs with more retrieved images are analyzed
in App. M). Within Doc-REFRAG, TTFT drops
from 3.24 s ( k=2, 194 tokens) to 1.95 s ( k=6, 108
tokens), with k=3 chosen as the default at 2.57 s
for the best accuracy–efficiency trade-off (Tab. 2).
Compared to Docopilot, the most accurate base-
line, Doc-REFRAG incurs a slight TTFT increase
but reduces TTIT by 2.18 s, accelerating end-to-
end generation for long outputs. When compared
to token-compressed MLLMs under a comparable
visual token budget, Doc-REFRAG matches their
TTFT while improving accuracy by over 18%.
Selector Analysis.Doc-REFRAG uses an RL
selector to decide which compressed chunks to ex-
pand at decoding. We compare three strategies: (1)
Random, uniformly sampling chunks; (2)ColQwen,
selecting chunks with the highest question-chunk
similarity, mirroring training; (3)Perplexity-based,
a heuristic ranking chunks by decoder perplexity
given the question and chunk embeddings, with
lower values indicating higher relevance.
As shown in Tab. 3, our RL-based selector out-
performs random (38.8%) and perplexity-based
(43.7%) strategies, achieving 57.5% average accu-
racy. In comparison, ColQwen, a dedicated dense
retriever for question-document relevance, achieves
higher accuracy on InfoVQA, possibly because In-
foVQA is part of its training data (Faysse et al.,
2024). However, ColQwen computes similarity
scores over patches and aggregates them to infer
chunk relevance, incurring a latency of 9.3 s. By
contrast, our selector identifies relevant chunks in
a single forward pass at 3.7 s.
5.3 Analysis
Can the chunk embedding handle dense text?
Chunk embeddings risk losing textual details criti-
cal for RAG. We design a synthetic benchmark of
A4 document images (595 ×842 pixels), synthe-
sizing English passages via PyMuPDF with font
sizes 10–20 and 500–6,000 characters per image.
We evaluate all models by their ability to recover
original text from images using Average Normal-
ized Levenshtein Similarity (ANLS). As shown in
Fig. 4, Doc-REFRAG (150 tokens/image) achieves
ANLS above 95% for documents up to 4,000 char-
acters, matching DeepSeek-OCR (256 tokens) (Weiet al., 2025) and DocOwl2 (324 tokens) despite its
compact representation. Even at 6,000 characters,
performance degrades only marginally, remaining
within a 5% absolute gap from the baselines. This
demonstrates that chunk embeddings preserve tex-
tual semantics even under high lexical density.
Can our training tasks be beneficial?Tab. 4
confirms that our training tasks are essential. Re-
moving single-image reconstruction drops accu-
racy from 55.1% to 19.7%, showing that chunk
embeddings lose semantic fidelity without recon-
struction supervision. Replacing curriculum learn-
ing with randomly shuffled multi-image sequences
reduces accuracy to 24.2%, showing that mixed-
token handling requires progressive exposure dur-
ing training. These results validate that our training
strategy enables effective compression while pre-
serving textual alignment.
Can our dataset improve the performance?
Conventional VDU datasets often lack the noise
and redundancy characteristic of multi-image re-
trieval. Excluding DocLongRAG from training
reduces Doc-REFRAG’s average accuracy from
55.1% to 48.9%, underscoring its critical role in
robust RAG adaptation. To assess generalizability,
we fine-tune InternVL3.5, MiniCPM-V-2.6, and
DocOwl2 on DocLongRAG, yielding accuracy im-
provements of 3.9%, 4.3%, and 3.9%, respectively.
This consistent cross-model gain validates DocLon-
gRAG as the first long-context, challenging dataset
for multimodal document RAG. A 2×2 factorial de-
composition of architecture and data contributions
is provided in App. N. To validate that gains stem
from the construction pipeline rather than mere
data volume, we compare against a naive random
baseline in App. O.
Can Doc-REFRAG help improve the reranker
performance?Document reranking is com-
1000 2000 3000 4000 5000 6000
Character Number859095100ANLS (%)
ANLS Score vs Character Number
mPLUG-DocOwl2
Doc-REFRAG
Deepseek-OCR
Figure 4: Parsing performance on A4 document image.

Strategy ChartQA SlideVQA InfoVQA DUDE Vidore MMDocIR Average Latency
Random 39.4 42.5 48.3 33.2 38.6 30.8 38.8 -
Perplexity 46.5 48.3 47.2 39.6 40.4 40.3 43.7 20.7
ColQwen 56.6 54.0 59.549.8 53.5 52.3 53.6 9.3
RL (Ours) 61.2 60.8 58.7 53.2 57.3 54.9 57.5 3.7
Table 3: Performance of different selection strategies for Doc-REFRAG. Accuracy (% ↑) and Latency in seconds ( ↓)
are measured on 8×RTX 3090 GPUs.
Variant Vidore MMDocIR DUDE Average
Doc-REFRAG (Full) 53.2 57.3 54.9 55.1
w/o Single-image Recon. 20.2 16.5 22.5 19.7
w/o Curriculum Learning 24.5 22.8 25.4 24.2
w/o Both 7.2 8.5 8.4 8.0
Table 4: Ablation study of training tasks. Results are
reported as accuracy (%↑) on three key benchmarks.
Model DUDE Vidore MMDocIR Average
InternVL3.5 (orig.) 45.8 42.1 51.5 46.5
InternVL3.5 (ft.) 47.1 50.4 53.8 50.4
MiniCPM-V-2.6 (orig.) 46.4 41.1 48.8 45.4
MiniCPM-V-2.6 (ft.) 51.0 45.3 52.8 49.7
DocOwl2 (orig.) 44.7 47.5 45.2 45.8
DocOwl2 (ft.) 48.4 51.6 49.1 49.7
Doc-REFRAG (w/o DocLongRAG) 48.5 43.7 54.6 48.9
Doc-REFRAG (Full) 53.2 57.3 54.9 55.1
Table 5: Ablation study of DocLongRAG. General
and document-aware MLLMs are compared in origi-
nal (orig.) and fine-tuned (ft.) versions on DocLon-
gRAG. “Doc-REFRAG (w/o DocLongRAG)” denotes
our model trained without the DocLongRAG corpus.
Results are reported as accuracy (%↑).
Model MMDocIR Vidore DUDE Latency
Jina-m0 (Günther et al., 2025) 81.1 84.6 86.2 13.6
MM-R5 (Xu et al., 2025) 82.3 81.9 81.3 49.5
MonoQwen2 (Chaffin and Lac, 2024) 81.4 82.1 82.4 12.5
Doc-REFRAG 84.2 85.8 87.7 7.3
Table 6: Performance comparison across rerankers. Re-
sults are reported as Recall@10 ( ↑). Latency is mea-
sured in seconds (↓).
monly formulated as binary classification, where
MLLMs sequentially assess the relevance of each
candidate image to a given question via the logit of
the “Yes” token. However, processing each image
incurs high inference latency, prohibitive across
many candidates.
Doc-REFRAG reduces per-document cost
through efficient visual encoding. By globally com-
pressing visual tokens and dynamically restoring
question-relevant details, it enables faster relevance
scoring without sacrificing accuracy. As shown in
Tab. 6, Doc-REFRAG achieves higher reranking
recall than dedicated rerankers on MMDocIR, yet
incurs much lower per-document latency.Pipeline Strategy Selection Method Latency Acc.
Direct Retrieval Top-5 (No Reranker) 4.246.6
Conventional Reranking MonoQwen2 (Top-20→5) 15.4 53.2
Doc-REFRAG Selective Encoding (Top-20) 6.1 57.5
Table 7: Comparison of different RAG pipelines. Re-
sults are reported as accuracy (% ↑) averaged across six
benchmarks, and total end-to-end time in seconds ( ↓),
including retrieval, (re-)ranking, and generation.
Can Doc-REFRAG surpass the reranker
pipeline?We compare Doc-REFRAG against
two standard retrieval-then-rerank baselines: (1)
Direct Retrieval, which feeds the top-5 candidates
from the retriever Jina-Embeddings-V4 directly
into Qwen3-VL-8B, and (2)Reranking Pipeline,
which uses MonoQwen2 (Chaffin and Lac, 2024)
to rerank top-20 candidates to 5 before answering.
As shown in Tab. 7, the Direct Retrieval baseline
suffers from evidence loss due to the performance
ceiling of dual-tower retrievers, where limited top-
kcandidates fail to encapsulate sufficient informa-
tion. The Reranking Pipeline mitigates this loss, im-
proving accuracy to 53.2%, but incurs 15.4s latency
from the LLM-based reranking stage ( ∼0.63s/img).
In contrast, Doc-REFRAG processes all top-20 can-
didates via question-guided compression, achiev-
ing 57.5% accuracy in only 6.1s, resolving the
trade-off between retrieval recall and inference effi-
ciency in multimodal RAG.
6 Conclusion
We introduce Doc-REFRAG, an efficient decod-
ing framework for multimodal document RAG.
It employs question-guided compression and a
three-stage training paradigm to compress visual
tokens into chunks, selectively expanded via an RL-
based selector. Experiments across six benchmarks
show Doc-REFRAG outperforms eleven baselines,
achieving state-of-the-art accuracy with lower la-
tency. We further validate our training methodol-
ogy, the role of DocLongRAG, and its transferabil-
ity to document reranking. Our approach enables
efficient large-context inference in RAG, especially
in resource-constrained deployments.

Limitations
While Doc-REFRAG advances efficient multi-
modal document RAG, three limitations remain.
First, the chunk size kis fixed as a training hy-
perparameter and cannot adapt dynamically at in-
ference time. Although k= 3 achieves the best
overall accuracy–efficiency trade-off (App. H), per-
formance degrades by approximately 5% ANLS
on documents exceeding 4,000 characters, indicat-
ing that adaptive granularity could further improve
coverage in high-density layouts. Adapting kat
inference time, however, is not free. Each selector
is trained at a fixed k, so supporting multiple chunk
sizes requires an additional routing step that dis-
patches each input to the matching selector, whose
latency works against the efficiency goal of the
method. We leave a single selector that generalizes
across chunk sizes to future work. Second, the RL-
based selector is trained with answer accuracy as
the sole reward signal, providing no fine-grained
attribution of which individual chunks contributed
to the correct answer. As shown in App. L, 64%
of remaining errors concentrate on dense or struc-
turally complex text regions, suggesting that richer
reward shaping or chunk-level supervision could
improve selector precision in these cases. Third,
while our evaluation uses a real retriever over real
documents, the training noise in DocLongRAG is
constructed, and no single construction pipeline
covers every deployment scenario. DocLongRAG
is also limited to English documents. Although the
construction pipeline is language-agnostic, dense
CJK text packs more information per patch and
may favor a smaller chunk size. Finally, due to
the high computational cost of training large mul-
timodal models, we report results from a single
training run, which is standard practice in the mul-
timodal document understanding community. All
baseline comparisons follow the same single-run
protocol.
Ethics Statement
DocLongRAG is constructed from publicly avail-
able datasets and document corpora. During
dataset construction, we apply a four-stage filtering
pipeline that removes context-independent ques-
tions, online-search-dependent queries, unanswer-
able instances, and ethically problematic content
including personally identifiable information, of-
fensive language, and sensitive imagery (Sec. 3).
The synthetic document images used in our dense-text evaluation are generated programmatically and
contain no real user data.
Doc-REFRAG is designed as a document un-
derstanding tool and does not store or transmit
user documents beyond the scope of inference.
Although grounding responses in retrieved docu-
ments reduces hallucination risk relative to purely
parametric models, the system can still produce
errors in documents with dense, overlapping, or
contradictory evidence. Deployment in high-stakes
decision-making contexts requires human oversight
and should not rely solely on model outputs.
Acknowledgments
This work was supported by the National Natu-
ral Science Foundation of China under Grant No.
U24A20326. We thank the anonymous reviewers
and action editors for their constructive feedback.
References
Srikar Appalaraju, Peng Tang, Qi Dong, Nishant
Sankaran, Yichu Zhou, and R Manmatha. 2024. Doc-
formerv2: Local features for document understand-
ing. InProceedings of the AAAI conference on artifi-
cial intelligence, volume 38, pages 709–718.
Muhammad Arslan, Hussam Ghanem, Saba Munawar,
and Christophe Cruz. 2024. A survey on rag with
llms.Procedia computer science, 246:3781–3790.
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, and 1 others.
2025. Qwen3-vl technical report.arXiv preprint
arXiv:2511.21631.
Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou,
Jie Tang, Yuxiao Dong, and Juanzi Li. 2024. Lon-
galign: A recipe for long context alignment of large
language models.arXiv preprint arXiv:2401.18058.
Antoine Chaffin and Aurélien Lac. 2024. Monoqwen:
Visual document reranking.
Jun Chen, Dannong Xu, Junjie Fei, Chun-Mei Feng, and
Mohamed Elhoseiny. 2025. Document haystacks:
Vision-language reasoning over piles of 1000+ doc-
uments. InProceedings of the Computer Vision
and Pattern Recognition Conference, pages 24817–
24826.
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Jun-
yang Lin, Chang Zhou, and Baobao Chang. 2024a.
An image is worth 1/2 tokens after layer 2: Plug-and-
play inference acceleration for large vision-language
models. InEuropean Conference on Computer Vi-
sion, pages 19–35. Springer.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai,
Zhijian Liu, Song Han, and Jiaya Jia. 2023. Longlora:
Efficient fine-tuning of long-context large language
models.arXiv preprint arXiv:2309.12307.
Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo
Chen, Sen Xing, Muyan Zhong, Qinglong Zhang,
Xizhou Zhu, Lewei Lu, and 1 others. 2024b. Internvl:
Scaling up vision foundation models and aligning
for generic visual-linguistic tasks. InProceedings of
the IEEE/CVF conference on computer vision and
pattern recognition, pages 24185–24198.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 719–729.
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong,
Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N
Fung, and Steven Hoi. 2023. Instructblip: Towards
general-purpose vision-language models with instruc-
tion tuning.Advances in neural information process-
ing systems, 36:49250–49267.
Kuicai Dong, Yujing Chang, Xin Deik Goh, Dexun
Li, Ruiming Tang, and Yong Liu. 2025a. Mmdocir:
Benchmarking multi-modal retrieval for long docu-
ments.arXiv preprint arXiv:2501.08828.
Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng
Wang, Ruiming Tang, and Yong Liu. 2025b. Bench-
marking retrieval-augmented multimodal generation
for document question answering.arXiv preprint
arXiv:2505.16470.
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang
Cao, Bin Wang, Linke Ouyang, Songyang Zhang,
Haodong Duan, Wenwei Zhang, Yining Li, and 1
others. 2024. Internlm-xcomposer2-4khd: A pioneer-
ing large vision-language model handling resolutions
from 336 pixels to 4k hd.Advances in Neural Infor-
mation Processing Systems, 37:42566–42592.
Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang,
Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou,
Tong Lu, Hongsheng Li, and 1 others. 2025. Do-
copilot: Improving multimodal models for document-
level understanding. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages
4026–4037.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Om-
rani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. 2024. Colpali: Efficient document re-
trieval with vision language models.arXiv preprint
arXiv:2407.01449.
Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang
Zhou, Houqiang Li, and Can Huang. 2024. Docpe-
dia: Unleashing the power of large multimodal model
in the frequency domain for versatile document un-
derstanding.Science China Information Sciences,
67(12):220106.Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang,
and Songlin Hu. 2025. Nextlong: Toward effective
long-context training without long documents.arXiv
preprint arXiv:2501.12766.
Michael Günther, Saba Sturua, Mohammad Kalim
Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang,
Sedigheh Eslami, Scott Martens, Maximilian Werk,
Nan Wang, and 1 others. 2025. jina-embeddings-v4:
Universal embeddings for multimodal multilingual
retrieval.arXiv preprint arXiv:2506.18902.
Yefei He, Feng Chen, Jing Liu, Wenqi Shao, Hong Zhou,
Kaipeng Zhang, and Bohan Zhuang. 2024. Zipvl:
Efficient large vision-language models with dynamic
token sparsification and kv cache compression.arXiv
preprint arXiv:2410.08584.
Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang
Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei
Huang, and 1 others. 2024a. mplug-docowl 1.5: Uni-
fied structure learning for ocr-free document under-
standing.arXiv preprint arXiv:2403.12895.
Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming
Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou.
2024b. mplug-docowl2: High-resolution compress-
ing for ocr-free multi-page document understanding.
arXiv preprint arXiv:2409.03420.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. LoRA: Low-rank adaptation of
large language models. InInternational Conference
on Learning Representations.
Ruofan Hu, Yan Xia, Minjie Hong, Jieming Zhu,
Bo Chen, Xiaoda Yang, Minghui Fang, and Tao Jin.
2025. Vela: scalable embeddings with voice large
language models for multimodal retrieval.arXiv
preprint arXiv:2506.14445.
Ruofan Hu, Menghui Zhu, Jieming Zhu, Bo Chen,
Shengyang Xu, Minjie Hong, Xiaoda Yang, Sashuai
Zhou, Li Tang, Tao Jin, and 1 others. 2026. Docre-
triever: A plug-and-play framework for multimodal
document retrieval with comprehensive benchmark.
InProceedings of the 32nd ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining V . 2,
pages 1780–1791.
Wenbo Hu, Zi-Yi Dou, Liunian Li, Amita Kamath,
Nanyun Peng, and Kai-Wei Chang. 2024c. Ma-
tryoshka query transformer for large vision-language
models.Advances in Neural Information Processing
Systems, 37:50168–50188.
Kai Huang, Hao Zou, Ye Xi, BoChen Wang, Zhen Xie,
and Liang Yu. 2024.IVTP: Instruction-Guided Vi-
sual Token Pruning for Large Vision-Language Mod-
els, page 214–230. Springer Nature Switzerland.
Yutao Jiang, Qiong Wu, Wenhao Lin, Wei Yu, and Yiyi
Zhou. 2025. What kind of visual tokens do we need?
training-free visual token pruning for multi-modal
large language models from the perspective of graph.

InProceedings of the AAAI Conference on Artificial
Intelligence, volume 39, pages 4075–4083.
Samir Khaki, Junxian Guo, Jiaming Tang, Shang Yang,
Yukang Chen, Konstantinos N. Plataniotis, Yao Lu,
Song Han, and Zhijian Liu. 2025. Sparsevila: Decou-
pling visual sparsity for efficient vlm inference. In
2025 IEEE/CVF International Conference on Com-
puter Vision (ICCV), page 23784–23794. IEEE.
Hugo Laurençon, Andrés Marafioti, Victor Sanh, and
Léo Tronchon. 2024. Building and better understand-
ing vision-language models: insights and future di-
rections.arXiv preprint arXiv:2408.12637.
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng
Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang,
Yanwei Li, Ziwei Liu, and 1 others. 2024a. Llava-
onevision: Easy visual task transfer.arXiv preprint
arXiv:2408.03326.
Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang,
Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong,
Qing Li, and Lei Chen. 2024b. A survey on large
language model acceleration based on kv cache man-
agement.arXiv preprint arXiv:2412.19442.
Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song
Wang, Jie Qin, Jianke Zhu, and Lei Zhang. 2025. To-
kenpacker: Efficient visual projector for multimodal
llm.International Journal of Computer Vision, pages
1–19.
Yanwei Li, Chengyao Wang, and Jiaya Jia. 2024c.
Llama-vid: An image is worth 2 tokens in large lan-
guage models. InEuropean Conference on Computer
Vision, pages 323–340. Springer.
Ronghui Liu, Hao Ren, Haojie Ren, Wu Rui, Wei
Cui, Xiaojun Liang, Chunhua Yang, and Weihua
Gui. 2025. Knowledge enhanced industrial question-
answering using large language models.Engineer-
ing.
Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li,
Zhiyin Ma, Shuo Zhang, and Xiang Bai. 2024a.
Textmonkey: An ocr-free large multimodal model
for understanding document.arXiv preprint
arXiv:2403.04473.
Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi
Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong,
Yu Qiao, Dahua Lin, and 1 others. 2024b. Mmdu: A
multi-turn multi-image dialog understanding bench-
mark and instruction-tuning dataset for lvlms.Ad-
vances in Neural Information Processing Systems,
37:8698–8733.
Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai
Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhu-
oshu Li, Hao Yang, and 1 others. 2024. Deepseek-vl:
towards real-world vision-language understanding.
arXiv preprint arXiv:2403.05525.Qi Luo, Xiaonan Li, Tingshuo Fan, Xinchi Chen, and
Xipeng Qiu. 2025. Towards global retrieval aug-
mented generation: A benchmark for corpus-level
reasoning.arXiv preprint arXiv:2510.26205.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, and 1 others. 2024. Mmlongbench-doc:
Benchmarking long-context document understanding
with visualizations.Advances in Neural Information
Processing Systems, 37:95963–96010.
Quentin Macé, António Loison, and Manuel Faysse.
2025. Vidore benchmark v2: Raising the bar for
visual retrieval.arXiv preprint arXiv:2505.17166.
Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. 2022. Chartqa: A benchmark
for question answering about charts with visual and
logical reasoning.arXiv preprint arXiv:2203.10244.
Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthe-
nis Karatzas, Ernest Valveny, and CV Jawahar. 2022.
Infographicvqa. InProceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vi-
sion, pages 1697–1706.
Minesh Mathew, Dimosthenis Karatzas, and CV Jawa-
har. 2021. Docvqa: A dataset for vqa on document
images. InProceedings of the IEEE/CVF winter con-
ference on applications of computer vision, pages
2200–2209.
Xudong Tan, Peng Ye, Chongjun Tu, Jianjian Cao,
Yaoxin Yang, Lin Zhang, Dongzhan Zhou, and Tao
Chen. 2025. Tokencarve: Information-preserving vi-
sual token compression in multimodal large language
models.arXiv preprint arXiv:2503.10501.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 24827–24837.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku
Hasegawa, Itsumi Saito, and Kuniko Saito. 2023.
Slidevqa: A dataset for document visual question
answering on multiple images. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 37, pages 13636–13645.
Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. 2021.
Visualmrc: Machine reading comprehension on docu-
ment images. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 35, pages 13878–
13888.
Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny.
2023. Hierarchical multimodal transformers for mul-
tipage docvqa.Pattern Recognition, 144:109834.
Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski,

Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anck-
aert, Ernest Valveny, and 1 others. 2023. Document
understanding dataset and evaluation (dude). InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision, pages 19528–19540.
Dongsheng Wang, Natraj Raman, Mathieu Sibue,
Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong
Pei, Armineh Nourbakhsh, and Xiaomo Liu. 2023.
Docllm: A layout-aware generative language model
for multimodal document understanding.arXiv
preprint arXiv:2401.00908.
Haicheng Wang, Zhemeng Yu, Gabriele Spadaro, Chen
Ju, Victor Quétu, Shuai Xiao, and Enzo Tartaglione.
2025. Folder: Accelerating multi-modal large lan-
guage models with enhanced performance.arXiv
preprint arXiv:2501.02430.
Haoran Wei, Yaofeng Sun, and Yukun Li. 2025.
Deepseek-ocr: Contexts optical compression.arXiv
preprint arXiv:2510.18234.
Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, and
Linfeng Zhang. 2025a. Token pruning in multimodal
large language models: Are we solving the right
problem?arXiv preprint arXiv:2502.11501.
Zichen Wen, Shaobo Wang, Yufa Zhou, Junyuan Zhang,
Qintong Zhang, Yifeng Gao, Zhaorun Chen, Bin
Wang, Weijia Li, Conghui He, and 1 others. 2025b.
Efficient multi-modal large language models via
progressive consistency distillation.arXiv preprint
arXiv:2510.00515.
Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Si-
hang Li, Zhifeng Gao, Renxin Zhong, and Hengxing
Cai. 2025. Mm-r5: Multimodal reasoning-enhanced
reranker via reinforcement learning for document
retrieval.arXiv preprint arXiv:2506.12364.
Yibo Yan, Guangwei Xu, Xin Zou, Shuliang Liu, James
Kwok, and Xuming Hu. 2025. Docpruner: A storage-
efficient framework for multi-vector visual document
retrieval via adaptive patch-level embedding pruning.
arXiv preprint arXiv:2509.23883.
Jianxin Yang. 2023. Longqlora: Efficient and effective
method to extend context length of large language
models.arXiv preprint arXiv:2311.04879.
Xiaoda Yang, Xize Cheng, Minghui Fang, Hongshun
Qiu, Yuhang Ma, JunYu Lu, Jiaqi Duan, Sihang Cai,
Zehan Wang, Ruofan Hu, and 1 others. 2025. Multi-
modal conditional retrieval with high controllability.
InProceedings of the 31st ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining V . 2,
pages 3577–3585.
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo
Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin
Zhao, Zhihui He, and 1 others. 2024. Minicpm-v:
A gpt-4v level mllm on your phone.arXiv preprint
arXiv:2408.01800.Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming
Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chen-
liang Li, Junfeng Tian, and 1 others. 2023. mplug-
docowl: Modularized multimodal large language
model for document understanding.arXiv preprint
arXiv:2307.02499.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and 1 others. 2024. Vis-
rag: Vision-based retrieval-augmented generation
on multi-modality documents.arXiv preprint
arXiv:2410.10594.
Chenghao Zhang, Guanting Dong, Xinyu Yang, and
Zhicheng Dou. 2025a. Towards mixed-modal re-
trieval for universal retrieval-augmented generation.
arXiv preprint arXiv:2510.17354.
Jiajie Zhang, Yushi Bai, Xin Lv, Wanjun Gu, Danqing
Liu, Minhao Zou, Shulin Cao, Lei Hou, Yuxiao Dong,
Ling Feng, and 1 others. 2024a. Longcite: Enabling
llms to generate fine-grained citations in long-context
qa.arXiv preprint arXiv:2409.02897.
Jiajie Zhang, Zhongni Hou, Xin Lv, Shulin Cao, Zhenyu
Hou, Yilin Niu, Lei Hou, Yuxiao Dong, Ling Feng,
and Juanzi Li. 2024b. Longreward: Improving
long-context large language models with ai feedback.
arXiv preprint arXiv:2410.21252.
Liang Zhang, Anwen Hu, Jing Zhang, Shuo Hu, and Qin
Jin. 2023. Mpmqa: multimodal question answering
on product manuals. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 37, pages
13958–13966.
Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng.
2025b. Llava-mini: Efficient image and video large
multimodal models with one vision token.arXiv
preprint arXiv:2501.03895.
Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao
Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy,
Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and 1
others. 2024c. Sparsevlm: Visual token sparsification
for efficient vision-language model inference.arXiv
preprint arXiv:2410.04417.

A Differentiated Contributions Over
Related Compression Methods
We provide a structured comparison of Doc-
REFRAG against the most closely related visual
token compression methods across six dimensions
in Tab. 8. The comparison covers methods that
appear as baselines in our main evaluation (Tab. 2),
DocOwl2 (our base model), and three additional
query-conditioned compression methods (Sparse-
VILA, IVTP, LLaMA-VID) that are most architec-
turally similar to ours.
We elaborate on the dimensions where Doc-
REFRAG introduces substantive novelty.
Query-conditioned compression.Methods such
as SparseVILA, IVTP, and LLaMA-VID condi-
tion token selection or compression on the input
query. However, their query-conditioning mech-
anisms are heuristic: SparseVILA scores visual
tokens against the query using attention at decode
time (training-free); IVTP aggregates text instruc-
tions into a CLS token to guide pruning via at-
tention rollout; LLaMA-VID generates a query-
dependent context token alongside fixed content
tokens. None of these methodslearna selection
policy through end-to-end optimization against a
downstream task reward. Doc-REFRAG trains its
selector via GRPO, directly optimizing chunk se-
lection for answer accuracy—making the selection
policy a first-class learned component rather than a
post-hoc heuristic.
Dynamic budget with RL training.Prior meth-
ods either fix the compression ratio at design
time (IVTP, EPIC, LLaV A-Mini) or apply query-
independent dynamic sparsity (SparseVILA). Doc-
REFRAG adopts a dynamic budget p∈[0.1,0.3]
sampled during training, enabling the model to
learn budget-robust selection policies via GRPO
with clipped PPO updates, using the frozen de-
coder’s answer accuracy as the reward signal—a
training paradigm not previously applied to visual
token compression.
RAG noise robustness.All existing compression
methods are developed and evaluated on coherent
single-source documents or video frames. Doc-
REFRAG is trained on DocLongRAG, which ex-
plicitly simulates realistic retrieval noise: retrieved
images are heterogeneous, may be redundant or
contradictory, and include hard negatives inter-
leaved with relevant content. This distributional
alignment between training and deployment is akey factor behind Doc-REFRAG’s performance ad-
vantage in RAG settings (Tab. 2).
Learned restoration vs. heuristic retrieval vs.
condensation.SparseVILA retains the full KV
cache at prefill and selectively retrieves tokens per
decoding step—a form of training-free heuristic
restoration, not a learned policy. TokenPacker and
DocOwl2 implement coarse-to-finecondensation:
progressively merging tokens into compact rep-
resentations (irreversible and question-agnostic).
Doc-REFRAG instead implementslearned restora-
tion: it first compresses all tokens into chunk
embeddings via an MLP projector, then an RL-
trained selector dynamically identifies and restores
original fine-grained tokens for question-relevant
chunks. The restoration decision is learned end-
to-end against task reward, not computed by a
heuristic. This makes Doc-REFRAG’s approach
fundamentally distinct from both the condensa-
tion paradigm (TokenPacker, DocOwl2) and the
training-free retrieval paradigm (SparseVILA).
B Source Datasets in DocLongRAG
Corpus
DocLongRAG aggregates English data from the
following 15 open-source datasets: (Van Lan-
deghem et al., 2023; Masry et al., 2022; Mathew
et al., 2021; Ye et al., 2023; Mathew et al., 2022;
Tanaka et al., 2021, 2023; Zhang et al., 2023; Lau-
rençon et al., 2024; Liu et al., 2024b; Chen et al.,
2023; Yang, 2023; Zhang et al., 2024a; Bai et al.,
2024; Zhang et al., 2024b).
CEvaluation of Hard Negative Relevance
Limitations about the annotation.Here, we
present an example from the benchmarks to il-
lustrate their limitations regarding the relevance
annotation, which emphasizes the importance of
negative samples exclusion.
When the query is [From this report, which
subgroup among Hispanics has gained most
confidence from 2008 to 2015?] , the origi-
nal benchmark MMDocIR designates Fig. 5a as
the exclusive retrieval target, as it provides a bar
chart comparing financial optimism across various
Latino subgroups from 2008 to 2015. The 08-15
column highlights the percentage-point increase in
optimism for each group, with the “Some college
or more” subgroup exhibiting the largest change
(+20 percentage points), reflecting the greatest gain
in confidence during this period.

Method Query-conditioned Dynamic Budget RL-trained Selector RAG Noise Robust. Learned Restoration Multi-image RAG
MQT-LLaV A (Hu et al., 2024c)✗ ✓ ✗ ✗ ✗ ✗
EPIC (Wen et al., 2025b)✗ ✗ ✗ ✗ ✗ ✗
LLaV A-Mini (Zhang et al., 2025b)✗ ✗ ✗ ✗ ✗ ✗
TokenPacker (Li et al., 2025)✗ ✗ ✗ ✗ ✗ ✗
DocOwl2 (Hu et al., 2024b)✗ ✗ ✗ ✗ ✗ ✓
SparseVILA (Khaki et al., 2025)✓ ✗ ✗ ✗ ✗ ✗
IVTP (Huang et al., 2024)✓ ✗ ✗ ✗ ✗ ✗
LLaMA-VID (Li et al., 2024c)✓ ✓ ✗ ✗ ✗ ✗
Doc-REFRAG (Ours) ✓ ✓ ✓ ✓ ✓ ✓
Table 8: Comparison of Doc-REFRAG with related visual token compression methods across six design dimensions.
Query-conditioned: selection conditioned on the input question.Learned Restoration: selective restoration of
compressed tokens via a learned policy (as opposed to training-free heuristic retrieval).
However, Fig. 5b also constitutes a valid retrieval
target. By explicitly referencing 2008 and noting
that this year was “7 years ago,” the document con-
firms that the data summarizes trends from 2008
to 2015. Furthermore, it specifies that economic
optimism among Latinos with some college educa-
tion increased by +20 percentage points—roughly
twice the growth observed among those with only
a high school diploma (+9) or less education (+11).
This reinforces that the “Some college or more”
subgroup experienced the most substantial rise in
confidence. Thus, the pair from the raw benchmark
overlooks an equally relevant candidate, which
hinders the fair evaluation for retrieval systems.
Human validation about the hard negative rel-
evance.To rigorously validate the semantic rel-
evance of non-annotated images and inform the
design of hard negative sampling in DocLongRAG,
we conduct a controlled human evaluation study.
We randomly select 500 questions from the cu-
rated corpus C, each associated with a set of doc-
ument images retrieved by a document-aware re-
triever. For each query, we examine the top-20
retrieved images that do not belong to the ground-
truth relevant set D. Specifically, we analyze im-
ages at retrieval ranks 1 through 20 to quantify how
semantic relevance decays with decreasing rank.
Two expert annotators participate in the evalua-
tion. Given a query q, its ground-truth answer a,
and a candidate image d, each annotator rates the
degree to which dcontains content that supports
answering q. Relevance is assessed on a 5-point
Likert scale defined as follows:
•5: The image contains explicit and sufficient
evidence to answer the question.
•4: The image provides partial or indirect evi-
dence that strongly supports the answer.•3: The image is topically related but lacks
direct evidential value.
•2: The image shares superficial similarity
(e.g., same document type or layout) but is
not helpful for answering.
•1: The image is clearly irrelevant.
Annotators work independently, and inter-
annotator agreement is measured using Krippen-
dorff’s alpha, yielding a value of 0.78, which indi-
cates substantial agreement.
The average relevance scores exhibit a clear de-
cay trend across retrieval ranks: images at rank 1, 2,
and 3 receive average scores of 4.62, 4.31, and 3.94,
respectively, indicating that they frequently contain
meaningful supporting evidence. At rank 4 and
onward, the average score remains lower than 3.27,
suggesting limited relevance. This pattern confirms
that images beyond rank 3 are substantially less
likely to provide evidential support for the correct
answer. Even so, the relevance at rank 4 and be-
yond is not zero, so a small fraction of hard nega-
tives may still contain answer evidence. This does
not corrupt the training objective: the annotated
relevant image is always retained in the sequence
by construction, so the supervision remains well-
defined, and such cases act as additional retrieval
noise rather than false supervision.
D Placing Meta-Chunk at Different
Positions
We explore three different strategies for combining
meta-chunk and hard negatives, which are repre-
sented by the following descriptions:
1.Head: Placing the meta-chunk at the begin-
ning of the retrieved hard negatives.
2.Tail: Placing the meta-chunk at the end of the
retrieved hard negatives.

(a) Ground Truth for Query
 (b) Another Proper Page for Query
Figure 5: Example for Benchmark.
3.Random: Randomly inserting the meta-
chunk within the retrieved hard negatives.
Tab. 9 shows that placing the meta-chunk at the
beginning (Head) yields better performance. Plac-
ing relevant images before hard negatives estab-
lishes longer dependencies between the question-
relevant content and the decoder, which enhances
overall reasoning effectiveness.
Training-time ordering and inference-time ro-
bustness.We investigate whether the head-
position advantage (Tab. 9) introduces a positional
bias at inference time, causing the decoder to ex-
ploit token position rather than semantic content.
We emphasize that the meta-chunk ordering
strategy studied here is atraining-timedesign
choice for CPT, while at inference time the RL-
based selector determines which chunks are ex-
panded based purely oncontent relevanceto the
question—not position. Specifically, the selector
operates on chunk embeddings from all retrieved
images jointly and assigns expansion decisions
δj
i∈ {0,1} via a learned policy πθtrained with
GRPO rewards on answer accuracy. The input to
the selector is a set of chunk-level semantic embed-
dings; the reward signal is answer accuracy, whichis invariant to the physical ordering of images in
the context. As a consequence, the training objec-
tive provides no gradient signal that encourages
position-dependent selection behavior.
This design ensures that the head-position ben-
efit observed in Tab. 9 stems from improved long-
range dependency modeling during training—not
from a spurious positional shortcut exploited at in-
ference. The meta-chunk always appears at the
head of thetrainingsequence, but at inference time
the selector re-evaluates every chunk independently
and can assign high expansion probability to any
image regardless of its position in the retrieved list.
E System Components: Architecture,
Training, and Inference
To clarify the roles of Doc-REFRAG’s components
across three conceptually distinct phases, we pro-
vide a consolidated overview in Tab. 10. Each
component is associated with a specific phase (Ar-
chitecture, Training, or Inference) and a brief de-
scription of its role, with exact hyperparameters for
cross-referencing with Secs. 3–5.

Model ChartQA SlideVQA InfoVQA DUDE Vidore MMDocIR #Acc.
Head 61.2 60.8 58.7 53.2 57.3 54.9 57.5
Tail 50.1 52.6 50.7 40.8 40.7 44.9 46.6
Random 58.5 54.1 53.5 45.4 42.8 53.0 51.2
Table 9: Performance Comparison of Different Insertion Strategies.
Component Phase Description Details
Vision Encoder Architecture Encodes each image into 324 visual tokens Inherited from DocOwl2; frozen after Stage 1
MLP Projector (ϕ) Architecture Mapskvisual tokens to one chunk embeddingcj
i∈Rd2-layer MLP, hidden dim 4096; trained in Stage 1
Decoder (LLM) Architecture Autoregressively generates the answer from the reconstructed token sequence 7B decoder with LoRA fine-tuned in Stage 2
RL Selector (π θ) Architecture Predicts binary expansion decisionsδj
ifor each chunk 2-layer Transformer, hidden dim 256, 8 heads;<1M params
Stage 1: Single-Image Reconstruction Training Aligns MLP projector with decoder embedding space DocStruct4M; decoder frozen; lr2×10−4
Stage 2: Multi-image CPT Training Scales to multi-image hybrid sequences with pseudo-label expansion OpenDocVQA + Doc-750K + DocLongRAG; LoRA rank 64; lr1×10−5
Stage 3: RL Selector Training Training Trains selector to maximize answer accuracy via GRPO DocLongRAG; GRPO + clipped PPO (clip 0.2); reward = decoder accuracy
Chunk Encoding Inference Compresses each image toL=m/kchunk embeddingsm= 324tokens/image,k= 3,L= 108chunks/image
Selector Forward Pass Inference Single forward pass to predict which chunks to expand; budgetp= 0.2Expands topT=⌊0.2·n·L⌋chunks; latency 3.7 s vs. 9.3 s (ColQwen)
Chunk Expansion Inference Selected chunks restored to originalkvisual tokens before decoding Enables fine-grained detail recovery for question-relevant regions
Table 10: System overview of Doc-REFRAG, organized by architectural components, training stages, and inference-
time operations.
F Geometric Mapping Between Coarse
and Fine Patch Grids
To utilize question-to-patch similarity scores from
ColQwen as pseudo-labels in CPT, we establish
a deterministic geometric mapping between the
coarse spatial grid used by our encoder and the
fine-grained patch grid employed by ColQwen.
Both models discretize the input images into
spatial patches using normalized image coordi-
nates, where each pixel location is represented as
(x, y)∈[0,1]2. LetGcdenote the coarse grid used
by CPT, with patch size sc=1
Hc×1
Wc, where Hc
andWcare the number of horizontal and vertical
divisions. Similarly, let Gfdenote the fine grid
used by ColQwen, with patch size sf=1
Hf×1
Wf,
whereH f> H candW f> W c.
For any coarse patch pc∈G c, its spatial extent
is defined by the bounding box:
pc= [i·s c,(i+ 1)·s c]×[j·s c,(j+ 1)·s c],
where(i, j)indexes the patch in the coarse grid.
Each such coarse patch pccovers a subset of fine
patches in Gf. Specifically, the set of fine patches
contained withinp cis given by:
F(p c) ={p f∈G f|bbox(p f)⊆bbox(p c)}.
Due to the alignment of coordinate systems and
uniform sampling, this containment relationship
is fully deterministic. We compute F(p c)by iter-
ating over all fine patches and checking whether
their bounding boxes lie entirely within pc. In prac-
tice, this can be implemented via integer arithmeticbased on the ratio r=Hf
Hc=Wf
Wc, which must be
an integer for perfect alignment.
This mapping allows us to aggregate the sim-
ilarity scores from the fine patches in F(p c)to
generate a pseudo-label δj
ifor the importance of
coarse chunk iin question j. The aggregation is
performed as:
δj
i=1
|F(p c)|X
pf∈F(p c)sim(q j, pf),
where sim(q, p f)denotes the question-to-patch
similarity score provided by ColQwen.
This deterministic and spatially aligned mapping
ensures that the pseudo-labels are geometrically
consistent and preserve local structure, enabling ef-
fective supervision of chunk selection during train-
ing.
G Robustness Analysis of Pseudo-Labels
Since ground-truth chunk-level relevance labels are
unavailable in our setting, we conduct a human
evaluation study to assess the validity of the binary
pseudo-labelsδj
i∈ {0,1}generated in App. F.
We randomly sample 200 question–document
pairs from DocLongRAG. For each document, we
compute δj
ifrom the aggregated ColQwen similar-
ity scores. Two expert annotators, familiar with
document understanding tasks, independently la-
bel each chunk as relevant or irrelevant based on
whether it contains content that directly or indi-
rectly supports answering the given question. The
inter-annotator agreement (Cohen’s κ) is 0.79, in-
dicating substantial agreement.

We compare the pseudo-labels against the
human-annotated labels. The pseudo-labels
achieve an accuracy of 86.4%, with a precision
of 84.1%, recall of 80.3%, and F1 score of 0.82.
These results demonstrate that the pseudo-labels
align well with human judgments of chunk rele-
vance, providing strong empirical justification for
their use as supervision signals in training.
H Analysis of Chunk Sizek
In this section, we provide a detailed comparison of
the performance and efficiency of Doc-REFRAG
under different chunk sizes k∈ {2,3,6} . As dis-
cussed in the main paper, the hyperparameter k
determines the granularity of visual token compres-
sion, where the original visual tokens of each image
are partitioned into non-overlapping chunks of size
k.
Tab. 11 presents the quantitative results across
six document-level benchmarks. Our experiments
reveal that k= 3 achieves the optimal balance
between semantic fidelity and computational effi-
ciency, yielding the highest average accuracy of
57.5%. Specifically, compared to k= 2 , thek= 3
configuration improves average accuracy by 2.0%
while maintaining comparable inference latency.
This suggests that a slightly coarser compression
(k= 3 ) effectively filters out redundant visual
noise without sacrificing critical details, whereas
k= 2 may retain excessive irrelevant tokens that
hinder the selector’s decision-making.
On the other hand, increasing the chunk size to
k= 6 significantly reduces the number of visual
tokens (from ∼194 to ∼108 per image) and low-
ers the Time-To-First-Token (TTFT) latency from
2.57s to 1.95s. However, this comes at the cost of
a noticeable performance drop, with average accu-
racy decreasing by 5.2% compared to k= 3 . The
degradation is particularly evident in data-intensive
tasks like ChartQA and InfoVQA, indicating that
overly aggressive compression ( k= 6 ) compro-
mises the preservation of fine-grained textual and
structural details essential for complex reasoning.
Based on these findings, we adopt k= 3 as the
default setting for Doc-REFRAG in our main evalu-
ations, as it delivers state-of-the-art accuracy while
ensuring efficient inference suitable for practical
RAG applications.I Optimal Chunk Expansion Budget
The budget parameter p∈(0,1] controls the num-
ber of chunks selected for expansion, defined as
T=⌊p·n·L⌋ , where n·L is the total number
of chunks. During training, pis uniformly sam-
pled from the interval [0.1,0.3] to encourage ex-
ploration and prevent overfitting to a fixed budget.
To determine the optimal value of pduring infer-
ence, we conduct a hyperparameter search on the
six benchmarks.
Specifically, we primarily fix the pto values in
the list {0.1,0.15,0.2,0.25,0.3} and measure the
answer accuracy of Doc-REFRAG when process-
ing the expanded input sequence generated by the
selector. As shown in Tab. 12, p= 0.2 achieves the
highest accuracy (57.5%) while maintaining low
inference latency. Values below 0.2 provide insuf-
ficient context, compromising accuracy, whereas
values above 0.2 incur higher computational costs
without yielding significant improvements. This
demonstrates that p= 0.2 achieves an optimal
balance between performance and efficiency.
Comparison with ColQwen across the full bud-
get range.Tab. 12 also enables a direct accuracy–
latency comparison against ColQwen, which serves
as an upper-bound selector oracle in our frame-
work (Sec. 5.2). ColQwen achieves 53.6% average
accuracy at a latency of 9.3 s. Our RL-based se-
lector at p= 0.2 surpasses ColQwen with 57.5%
accuracy while requiring only 3.7 s of selection
latency—a 2.5× reduction. At p= 0.25 the se-
lector reaches 56.4% in 2.62 s, and at p= 0.3 it
reaches 53.7% in 2.84 s, all remaining well below
ColQwen’s latency budget. Conversely, even at
p= 0.1 (2.19 s, 51.6%), the selector is substan-
tially faster than ColQwen while sacrificing only
2.0 percentage points of accuracy. Across the entire
search range p∈ {0.1, . . . ,0.3} , the RL selector
consistently occupies a more favorable accuracy–
latency operating point than ColQwen, confirming
that replacing the dense retriever with a lightweight
learned selector does not require trading accuracy
for speed.
J Training Procedure for the Selector
We train the chunk selection policy using the GRPO
framework. The full training procedure is outlined
below.
Policy Network Architecture.The selector is
a lightweight two-layer transformer encoder with

Model Size TokenVChartQA SlideVQA InfoVQA DUDE Vidore MMDocIR #Acc. #TTFT #TTIT
Doc-REFRAG k=2 7B∼194 58.6 56.2 56.4 51.9 55.1 53.7 55.5 3.24 0.07
Doc-REFRAG k=3 7B ∼150 61.2 60.8 58.7 53.2 57.3 54.9 57.5 2.57 0.07
Doc-REFRAG k=6 7B∼108 53.2 54.8 53.9 48.5 52.4 52.2 52.3 1.95 0.07
Table 11: Comparison of Doc-REFRAG performance with different chunk sizes k.k= 3 achieves the best
accuracy–efficiency trade-off. TTFT denotes Time-To-First-Token.
p Accuracy (%) TTFT (s) TTIT (s)
0.1 51.6 2.19 0.07
0.15 54.2 2.34 0.07
0.2 57.5 2.57 0.07
0.25 56.4 2.62 0.07
0.3 53.7 2.84 0.07
Table 12: Answer accuracy under different chunk ex-
pansion budgetsp.
hidden dimension 256 and 8 attention heads. Given
a question embedding tand a sequence of chunk
embeddings cj
i, the policy network computes logits
over each chunk:
δj
i=W o·MLP
Transformer(t,cj
i)
,
where Wois a learnable output weight matrix. The
policy πθsamples binary decisions δj
i∈ {0,1}
independently for each chunk.
Group-Wise Relative Advantage Estimation.
For each question-document pair, we sample G=
8chunk selection policies {δ(1), δ(2), . . . , δ(G)}
from πθ. For each policy, we reconstruct the in-
put sequence and compute the reward r(g)using
the fixed decoder from the CPT stage. The relative
advantage is defined as:
Ai=r(i)−1
GGX
g=1r(g),
which measures the performance of the i-th policy
relative to the group mean.
Policy Update with Clip and KL Regularization.
The policy is updated using the following objective:
L=E δ∼πθh
min
ρ(δ)A(δ),
clip(ρ(δ),1−ϵ,1+ϵ)A(δ)i
−β·D KL 
πθ∥πref
,(1)
where ρ(δ) =π θ(δ)/π ref(δ)is the importance sam-
pling ratio, ϵ= 0.2 ,β= 0.1 , and πrefis the refer-
ence model initialized with SFT weights.K Training Cost Analysis
The four components are not trained separately at
full cost: only Stage 2 carries a non-trivial cost,
while Stages 1 and 3 are lightweight. We provide a
stage-by-stage breakdown of training configuration
across the three stages in Tab. 13. All experiments
are conducted on 8 ×NVIDIA RTX 3090 GPUs
(24 GB VRAM each). The base model is DocOwl2
(7B parameters).
Stage Dataset Trainable Parameters Epochs
Stage 1: Single-Image Recon. DocStruct4M Encoder align. + MLP projector 2
Stage 2: Multi-image CPT OpenDocVQA + Doc-750K + DocLongRAG LoRA (rank 64) on 7B decoder 2
Stage 3: RL Selector DocLongRAG Selector (<1M params) 2
Table 13: Training cost summary for Doc-REFRAG,
broken down by stage, dataset, and trainable parameters.
Stage 1is computationally lightweight: the de-
coder is fully frozen and only the linear alignment
module in the vision encoder and the two-layer
MLP projector are updated. The small number
of trainable parameters and single-image inputs
(DocStruct4M (Hu et al., 2024a)) make this stage
converge quickly.
Stage 2constitutes the dominant cost. Training
on the combined corpus of OpenDocVQA (Tanaka
et al., 2025), Doc-750K, and DocLongRAG—
totaling over 1.1M question–image pairs—requires
processing substantially longer multi-image hybrid
sequences. Despite using LoRA (rank=64) to re-
duce the number of trainable parameters in the 7B
decoder, the dataset scale and extended context
lengths make this stage the bottleneck of the over-
all pipeline.
Stage 3is again lightweight because the selector
itself is a tiny two-layer transformer ( <1M param-
eters); the 7B decoder is frozen and used only for
reward evaluation on DocLongRAG with GRPO
group size G= 8 . The modest parameter count and
single-dataset scope ensure this stage adds minimal
overhead relative to Stage 2.
Overall, the use of LoRA in Stage 2 and a
lightweight selector in Stage 3 keeps the total train-

ing footprint manageable on commodity RTX 3090
hardware, making Doc-REFRAG accessible with-
out specialized data-center resources.
L Qualitative Analysis
To better understand the failure modes of Doc-
REFRAG, we conduct a qualitative analysis on
100 incorrectly answered questions, categorizing
the source of errors based on the type of informa-
tion required for correct inference. As shown in
Fig. 6, the majority of errors (64%) stem from tex-
tual content that is dense or semantically complex,
where critical details are embedded across multiple
paragraphs and require fine-grained reasoning.
In contrast, 22% of failures involve figures,
charts, or diagrams—visual elements that may not
be fully reconstructed due to selective chunking,
leading to incomplete interpretation. Another 14%
arise from structured data such as tables or lists,
which often contain precise numerical or relational
information that can be lost during compression if
key rows or columns are pruned.
These results suggest that Doc-REFRAG han-
dles unstructured textual content effectively, with
the majority of failures (64%) concentrated in
the most challenging regime: dense or semanti-
cally complex text requiring fine-grained multi-
paragraph reasoning. Visual and structured content
together account for the remaining 36% of errors,
reflecting the inherent trade-off between compres-
sion efficiency and fidelity when selective chunking
omits visually or numerically critical regions.
Figure/Chart/
Diagram
22%
Table/List
14%Text
64%
Figure 6: Qualitative analysis of the modalities that
predominantly contribute to incorrect answers in Doc-
REFRAG.M Empirical Efficiency Validation
To quantitatively assess the efficiency gains of Doc-
REFRAG, we define the acceleration ratio as:
Acceleration Ratio=Latencybaseline
Latencyours,
where latency is measured in terms of time-to-first-
token (TTFT) and time-to-inter-token (TTIT) under
a batch size of 1, with Nretrieved images condi-
tioned on each question.
As shown in Fig. 7, the TTFT acceleration ra-
tio increases monotonically with N, peaking at
2.46×when N= 50 . This reflects the effective-
ness of our question-guided chunk selection in re-
ducing visual tokens, thereby accelerating initial
attention computation. In contrast, TTIT accelera-
tion grows gradually (max 1.12 ×), indicating that
post-initial-token processing still involves selected
chunks. Moreover, Doc-REFRAG achieves an av-
erage accuracy of 50.2% with 50 retrieved images,
substantially outperforming DocOwl2, which at-
tains only 37.2%.
5 15 30 50
Number of Images0.961.461.962.46Acceleration Ratio
Acceleration Ratio: TTFT vs. TTIT
TTFT
TTIT
Baseline
Figure 7: Acceleration ratio of Doc-REFRAG versus
DocOwl2 in terms of TTFT (blue) and TTIT (green), as
a function of the number of retrieved images. The red
dashed line indicates the baseline (DocOwl2 Latency =
1×).
N Disentangling Architectural and Data
Contributions
We next analyze the independent and joint contri-
butions of architectural design and training data to
Doc-REFRAG’s overall performance. To this end,
we construct a 2×2 factorial analysis using results
already reported in Tab. 5, varying both architec-
ture (DocOwl2 vs. Doc-REFRAG) and training
data (without vs. with DocLongRAG).
Tab. 14 reveals three complementary sources of
improvement.

w/o DocLongRAG w/ DocLongRAG
DocOwl2 (base architecture) 45.8 49.7
Doc-REFRAG (our architecture) 48.9 55.1
Data gain (∆ data)49.7−45.8 = +3.9%
Architecture gain (∆ arch)48.9−45.8 = +3.1%
Synergy (∆ syn)55.1−49.7−3.1 = +2.3%
Total gain55.1−45.8 = +9.3%
Table 14: 2×2 factorial decomposition of average accu-
racy (%) across DUDE, Vidore, and MMDocIR. Base:
DocOwl2 (orig.). ∆data: gain from DocLongRAG
alone. ∆arch: gain from Doc-REFRAG architecture
alone.∆ syn: additional gain from their interaction.
Data contribution ( +3.9% ).Fine-tuning the
original DocOwl2 on DocLongRAG alone raises
average accuracy from 45.8% to 49.7%. This is
consistent with the finding in Sec. 5.3 that DocLon-
gRAG provides a broadly beneficial training signal
for models across different architectures.
Architecture contribution ( +3.1% ).Even with-
out DocLongRAG, replacing DocOwl2’s encoding
strategy with Doc-REFRAG’s chunk compression
and RL-based selector raises accuracy from 45.8%
to 48.9%. This demonstrates that the architectural
design itself provides meaningful gains that are
independent of the new training data.
Synergistic interaction ( +2.3% ).The full sys-
tem (Doc-REFRAG trained on DocLongRAG)
achieves 55.1%, exceeding the sum of the two indi-
vidual effects by 2.3 percentage points. This super-
additive interaction arises because Doc-REFRAG’s
question-guided compression makes the model bet-
ter able to exploit the noisy-retrieval training signal
present in DocLongRAG: the RL selector learns
to identify relevant chunks from heterogeneous re-
trieved images, a skill that DocLongRAG specifi-
cally exercises.
In summary, both contributions are indepen-
dently meaningful and mutually reinforcing. The
architectural gains cannot be explained by data
alone: even when DocOwl2 is fine-tuned on Do-
cLongRAG (49.7%), it remains 5.4 percentage
points below Doc-REFRAG Full (55.1%), demon-
strating the independent value of our architectural
design. Conversely, the data gains cannot substitute
for the architecture: Doc-REFRAG without Doc-
LongRAG (48.9%) still lags behind the full system
(55.1%), confirming that the dataset and the archi-
tecture are complementary rather than redundant.O Validating the DocLongRAG
Construction Pipeline
We investigate whether the performance improve-
ments attributed to DocLongRAG stem from the
carefully engineered construction pipeline (hard
negative mining and quality filtering) or simply
from exposing the model to a larger number of
images during training. To isolate these factors,
we design a naive random baseline that uses the
identical model architecture but replaces the cu-
rated DocLongRAG training data with randomly
assembled training instances.
Specifically, for this baseline—denotedDoc-
REFRAG (naive random)—we draw questions at
random from Doc-750K and OpenDocVQA, and
pair each question with approximately 37 randomly
sampled images (matching the average input length
in DocLongRAG), without performing any hard
negative mining or quality filtering. All other train-
ing hyperparameters and architectural choices are
held constant.
Model DUDE Vidore MMDocIR Average
DocOwl2 (orig.) 44.7 47.5 45.2 45.8
Doc-REFRAG (naive random) 34.2 36.1 35.8 35.4
Doc-REFRAG (Full) 53.2 57.3 54.9 55.1
Table 15: Comparison of Doc-REFRAG trained on ran-
domly assembled data (naive random) versus the full
DocLongRAG pipeline. The naive random baseline
uses the same architecture but pairs each question with
∼37 randomly selected images, without hard negative
mining or quality filtering. Results are reported as aver-
age accuracy (% ↑) on DUDE, Vidore, and MMDocIR.
See also Tab. 5.
Tab. 15 yields three clear findings. First, Doc-
REFRAG (naive random) achieves only 35.4% av-
erage accuracy—10.4 percentage pointsbelowDo-
cOwl2 (orig.) at 45.8%—demonstrating that sim-
ply exposing the model to more images is not
only unhelpful but actively harmful: random image
pairings introduce retrieval noise that disrupts the
model’s reasoning without providing a useful train-
ing signal. Second, the full DocLongRAG pipeline
raises average accuracy from the DocOwl2 base-
line of 45.8% to 55.1%, a gain of 9.3 percentage
points, confirming that the hard negative mining
and quality filtering steps are the true source of im-
provement. Third, the gap between Doc-REFRAG
(Full) and Doc-REFRAG (naive random) is 19.7
percentage points (55.1% −35.4%), clearly isolat-
ing the contribution of thepipeline designfrom the

contribution of raw image quantity. Together, these
results validate that the DocLongRAG construction
pipeline itself, not mere data volume, is responsible
for the observed performance gains.
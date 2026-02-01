# Pixel-Grounded Retrieval for Knowledgeable Large Multimodal Models

**Authors**: Jeonghwan Kim, Renjie Tao, Sanat Sharma, Jiaqi Wang, Kai Sun, Zhaojiang Lin, Seungwhan Moon, Lambert Mathias, Anuj Kumar, Heng Ji, Xin Luna Dong

**Published**: 2026-01-27 00:46:08

**PDF URL**: [https://arxiv.org/pdf/2601.19060v1](https://arxiv.org/pdf/2601.19060v1)

## Abstract
Visual Question Answering (VQA) often requires coupling fine-grained perception with factual knowledge beyond the input image. Prior multimodal Retrieval-Augmented Generation (MM-RAG) systems improve factual grounding but lack an internal policy for when and how to retrieve. We propose PixSearch, the first end-to-end Segmenting Large Multimodal Model (LMM) that unifies region-level perception and retrieval-augmented reasoning. During encoding, PixSearch emits <search> tokens to trigger retrieval, selects query modalities (text, image, or region), and generates pixel-level masks that directly serve as visual queries, eliminating the reliance on modular pipelines (detectors, segmenters, captioners, etc.). A two-stage supervised fine-tuning regimen with search-interleaved supervision teaches retrieval timing and query selection while preserving segmentation ability. On egocentric and entity-centric VQA benchmarks, PixSearch substantially improves factual consistency and generalization, yielding a 19.7% relative gain in accuracy on CRAG-MM compared to whole image retrieval, while retaining competitive reasoning performance on various VQA and text-only QA tasks.

## Full Text


<!-- PDF content starts -->

Pixel-Grounded Retrieval for Knowledgeable Large
Multimodal Models
Jeonghwan Kim1,2,∗,Renjie Tao1,Sanat Sharma1,Jiaqi Wang1,Kai Sun1,Zhaojiang Lin1,Seungwhan
Moon1,Lambert Mathias1,Anuj Kumar1,Heng Ji2,Xin Luna Dong1
1Meta Reality Labs,2University of Illinois Urbana-Champaign
∗Work done at Meta
Visual Question Answering (VQA) often requires coupling fine-grained perception with factual
knowledge beyond the input image. Prior multimodal Retrieval-Augmented Generation (MM-RAG)
systems improve factual grounding but lack an internal policy forwhenandhowto retrieve. We propose
PixSearch, the first end-to-end Segmenting Large Multimodal Model (LMM) that unifies region-
level perception and retrieval-augmented reasoning. During encoding,PixSearchemits <search>
tokens to trigger retrieval, selects query modalities (text, image, or region), and generates pixel-level
masks that directly serve as visual queries, eliminating the reliance on modular pipelines (detectors,
segmenters, captioners, etc.). A two-stage supervised fine-tuning regimen with search-interleaved
supervision teaches retrieval timing and query selection while preserving segmentation ability. On
egocentric and entity-centric VQA benchmarks,PixSearchsubstantially improves factual consistency
and generalization, yielding a 19.7% relative gain in accuracy on CRAG-MM compared to whole image
retrieval, while retaining competitive reasoning performance on various VQA and text-only QA tasks.
Date:January 28, 2026
Correspondence:First Author atjk100@illinois.edu, jeonghkim@meta.com
Code:https://github.com/wjdghks950/PixSearch
1 Introduction
Entity-centric visual question answering (VQA) sits at the nexus of perception and reasoning: it demands
recognizing specific entities in an image, leveraging factual knowledge about those entities, and when needed,
composing related evidence to answer the question. VQA onegocentricimages from wearable devices such
as smart glasses is even harder: as illustrated in Figure 1, wide-angle viewpoints render entities small, and
the entities themselves are often long-tail or niche, making them unlikely to be reliably covered by an LLM’s
internal knowledge.
Multimodal Retrieval-Augmented Generation (MM-RAG)has strengthened factual grounding in VQA (Marino
et al., 2021; Lin et al., 2022; Jian et al., 2024), but two limitations persist. First, most MM-RAG systems
either retrieve with the full image (Shah et al., 2019; Marino et al., 2021; Yang et al., 2023; Yan and Xie,
2024; Yu et al.; Ha et al., 2025; Sidhu et al., 2025), or use text-only queries that simply paraphrase the
image (Narasimhan and Schwing, 2018; Gardères et al., 2020; Gao et al., 2022; Salaberria et al., 2023).
Full-image retrieval pulls in distracting background, while text-only cues (e.g., “car”) lack the specificity needed
for fine-grained entity grounding and enrichment. Second, MM-RAG pipelines are often modular—detectors,
segmenters, captioners, etc.—to form queries, thus can introduce cross-modal translation errors, struggle with
composing multiple queries, and add latency when retrieval isnottruly necessary.
We presentPixSearch, the first end-to-end framework for retrieval-augmented reasoning. During generation,
PixSearch(i) learnswhento retrieve by emitting <search> tokens, (ii) decideshowto retrieve by routing
among text, whole-image, and region-level queries via token outputs, and (iii) grounds answers in the retrieved
evidence, supporting multi-step search. Built on segmenting LMMs (Large Multi-modal Models with segmenta-
tion capabilities),PixSearchnatively produces segmentation masks without external detection/segmentation
APIs, and uses these masks directly as retrieval queries. This yields pixel-level, context-aware grounding that
surpasses modular, text- or tool-driven pipelines.
1arXiv:2601.19060v1  [cs.CV]  27 Jan 2026

Search Index
User Query:Is this car suitable for sea0ng more than seven people?
Tool-Reliant ApproachesCarPixSearch
SAMRPNVSLLaVA-13BPixSearch-13B(Whole img.)Segmen0ng LMM directly grounds the en0ty speciﬁc pixels and use them for retrievalAccuracy 
⬆
Hallucina0on 
⬇PixSearch-13B(Full)
LLaVA-13BPixSearch-13B(Whole img.)PixSearch-13B(Full)Figure 1Egocentric images from wearables devices often render entities smaller than it appears because of wide-angle
cameras. MM-RAG methods that rely on full-image search or caption-only queries can introduce retrieval noises and
degrade QA quality.PixSearch, an end-to-end segmenting LMM, learns when to issue a query, how to route among
text, whole-image, and region-level queries, and how to reason over retrieved evidence for answer generation. Our
work also compares against pipeline, tool-based approaches. On CRAG-MM (Wang et al., 2025),PixSearch(full)
improves accuracy by 26% and reduces hallucination by 39%.
Integrating the aforementioned capabilities, nonetheless, is non-trivial. It either requires reinforcement learning
(RL)-based tuning as in previous work (Jin et al., 2025) or requires a supervised finetuning (SFT) dataset to
teach the model such behaviors. Nonetheless, the field currently lacks such data that interleaves retrieval
triggering, query type assignment and reasoning into a single model output sequence. To this end, we propose
an effective two-stage supervised finetuning strategy, together with a training data construction pipeline. We
leverage diverse VQA datasets (Singh et al., 2019; Chen et al.; Hu et al., 2023; Wang et al., 2025; Chang et al.,
2022; Marino et al., 2019; Schwenk et al., 2022) to teach the model to trigger retrieval only when needed and
to select appropriate query types, enabling effective multimodal RAG for entity-centric VQA while preserving
segmentation performance.
In summary, our paper makes the following three contributions.
1.Framework:We introducePixSearch, the first end-to-end segmenting LMM that autonomously triggers
retrieval and performs region-level grounding.
2.Training: We devise a two-stage supervised fine-tuning regimen and a multimodal training dataset
that teaches when to retrieve and how to form text, image, or region queries via search-interleaved
trajectories, and meanwhile preserving segmentation quality.
3.Experiments: Through comprehensive experimental results, we demonstrate thatPixSearchsub-
stantially improves factual consistency and generalization across egocentric and entity-centric VQA
benchmarks, achieving 19.7% accuracy improvement in CRAG-MM (Wang et al., 2025), and in particular
24.3% improvement on egocentric images (Figure 1).
2 Related Work
2.1 Knowledge-based Visual Question Answering
Prior research in knowledge-based visual question answering (KB-VQA) has recognized the need to go
beyond parametric model knowledge by incorporating external retrieval modules. Previous work such as
KRISP (Marino et al., 2021) combined implicit knowledge from vision-language transformers with explicit
symbolic knowledge retrieved for detected objects. Similarly, KAT (Gui et al., 2022) leveraged retrieved
knowledge snippets aligned to object regions for VQA, and MAVEx (Wu et al., 2022) proposed answer-
2

conditioned retrieval to validate candidate answers with external evidence. These systems demonstrated
that combining visual inputs with external knowledge improves factual grounding, but they largely rely on
text-based representationsof detected entities, e.g., object tags or noun phrases, as the query. More recent
approaches (Lin et al., 2022) incorporated object detector outputs as region-specific queries, or leveraged
LLMs to extract referring expressions for visual entity grounding (Jian et al., 2024), showing that retrieving
knowledge about individual detected objects significantly outperforms whole-image retrieval. While these
approaches highlight the promise of region-centric retrieval, they nonetheless depend on external object
detectors, separate segmentation modules, or hand-engineered pipelines.
2.2 Multimodal Retrieval-Augmented Generation.
In parallel, retrieval-augmented generation (RAG) has been extended to multimodal domains. EchoSight
(Yan and Xie, 2024) combined visual search with textual retrieval to answer encyclopedic VQA, while VisRAG
(Yu et al.) introduced document-level multimodal retrieval for visually rich pages. RA-CM3 (Yasunaga
et al., 2023) unified text and image retrieval for generation, and frameworks like UniIR/M-BEIR (Wei
et al., 2024) benchmarked multimodal retrieval performance. While these works demonstrate the benefits of
grounding multimodal models in external knowledge, their queries are typically either text-only or whole-image
embeddings, limiting their precision when a specific entity in the scene is most relevant.
2.3 Search Trigger and Interleaving External Knowledge
Another open challenge is teaching models not just touseretrieval, but to decidewhenretrieval is needed and
howto query. Self-RAG (Asai et al.) introduced reflection tokens for adaptive retrieval, Toolformer (Schick
et al., 2023) trained LLMs to call external APIs at the right time, and RePlug (Shi et al., 2024) optimized
retrievers based on LM likelihood. More recent line of work, Search-R1 (Jin et al., 2025) uses PPO/GRPO to
enable LLMs to interleave search with auto-regressive reasoning. These methods show that adaptive retrieval
policies can reduce unnecessary latency and hallucinations. However, they have been explored primarily in
text-only settings, and have not been extended to multimodal region-level retrieval.
2.4 Segmenting Large Multimodal Models.
Our solution is built upon Segmenting LLMs, which can output segmentation masks alongside text. Notable
examples include LISA (Lai et al., 2024), PixelLM (Ren et al., 2024b), GLaMM (Rasheed et al., 2024), Osprey
(Yuan et al., 2024), and PLUM (Blume et al., 2025). These models integrate segmentation into language
generation by predicting mask tokens or embeddings inline, enabling fine-grained visual grounding. For
instance, GLaMM was trained on millions of region-grounded annotations to generate segmentation masks in
a conversational setting, while PLUM introduced span-based tagging and a mask feedback loop to iteratively
refine object selection. These models show that segmentation can be natively integrated into the reasoning
process of LMMs. However, they are trained to answer questions without external retrieval, relying solely on
their internal knowledge.
3 Overview
3.1 Problem Definition
We study the Visual Question Answering (VQA) problem, which takes an image Iand a question Qregarding
the image as input, and outputs an answer Ato the question. We assume an external knowledge repository
to facilitate question answering. A good answer shall be relevant and helpful, and meanwhile consistent with
knowledge present in the repository.
We assume the knowledge repository is accessible through a retrieval API search_api (S, k), which returns the
top-krelevant text chunks on search query S. The query can be in three forms: (1) full image, where the API
returns information about the image; (2) masked region, normally for a particular entity and the API returns
information about the entity; and (3) text span, where the API returns search results for the text query.
3

PixSearchQ: “Is this car suitable for sea0ng more than 7 people?”
“The car<search> <region> </search><informa5on> Subaru WRX STI | … </informa5on> in the image appears to be a Subaru WRX STI <search> What is the the sea8ng capacity of Subaru WRX STI? </search><informa5on> Subaru WRX STI runs 28 mpg on … | … </informa5on> which is a high-performance variant of the Subaru Impreza, cannot seat more than 7 people since its maximum sea5ng capacity is 5.”Subaru WRX STI: a high-performance variant of …Mask Decoder
🔥Prompt Encoder𝒉𝒄𝒂𝒓𝑳%𝟏
Subaru Impreza: … is a compact hatchback ……Subaru WRX STI runs 28 mpg on ……
🔥
🔥
Retrieved KnowledgeSearch Trigger & QuerySpan Extracted for Mask
What is the the sea@ng capacity of Subaru WRX STI? Local Search IndexLocal Search IndexFigure 2 Overview of the proposed PixSearch framework.The model learns to decidewhenretrieval is needed,how
to query (text, whole image, or segmented region), and grounds its answers in retrieved evidence while preserving
mask-generation capabilities.
An effective MM-RAG solution needs to make the following decisions: 1) whether to issue (a single or multiple)
search queries; 2) the modality of each query—full image, masked region, or text span; 3) the specific image
region or text span to query; 4) the final answer based on the retrieval evidence. Existing pipeline-based
methods separate these abilities, introducing translation errors and instability. We next describe a uniform
framework that resolves all four through an interleaved search-and-generation decoding process.
3.2 PixSearch Framework
Figure 2 depicts thePixSearchframework.PixSearchconducts search-interleaved decoding, a retrieval-
augmented generation process that enables the model to decidewhento retrieve andhowto ground retrieved
evidence in its multimodal reasoning trajectory. At each decoding step t, the model autoregressively predicts
the next token xtbased on the image mand the previously generated tokens, until an end-of-sequence ( </s>)
token is reached:
xt∼pθ(xt|x<t, Q, I).(1)
An output token can be a special control token <search> , at which point the model temporarily halts textual
decoding to initiate a retrieval subroutine, which proceeds in three steps. First, the subroutine generates a
payloadstring in {<image>,<region>,<text>1}to describe the retrieval modality. We then generate the
search query for different modalities. For theimagemode, the whole input image Iserves as the query. For
theregionmode, the model invokes its aligned mask decoder to predict a binary mask ˆM=fθ(I, x<t), from
which a cropped visual query is extracted. For thetextmode, the model generates the textual query during
decoding.
S=

I,ifpayload=<image>
crop(I, ˆM),ifpayload=<region>
Text,ifpayload=Text(2)
x1:t←[x 1:t−1,<search>, S,</search>].(3)
Finally, the subroutine obtains the retrieved evidence:
E=search_api(S, k)(4)
where search_api returns a textual knowledge. This retrieved content is then injected back into the generation stream
as an<information>block (abbreviated as<info>hereafter for brevity):
x1:t+1←[x 1:t,<info>,E,</info>](5)
1The model directly generates the textual query instead of<text>
4

allowing the model to continue decoding while conditioning on the newly appended evidence.
Through generating multiple <query> blocks and populating back <information> evidence blocks, this decoding
strategy results in a dynamic reasoning trajectory that alternates between internal generation and external retrieval,
enabling the model to ground answers in factual evidence whereas maintaining fine-grained visual reasoning through
region-level queries.
4 PixSearch: Region-level Retrieval for LMMs
4.1 Overview of model training strategy
At the core ofPixSearchis a segmentation-capable Large Multimodal Models (i.e.,segmenting LMMs(Rasheed et al.,
2024; Lai et al., 2024; Ren et al., 2024b; Wang et al., 2024; Blume et al., 2025)), with two key capabilities required by
the framework: segmentation (to facilitate mask generation fθ) and decoding (Eq. 1-5). This design exploits the rich
textual semantics for pixel-level grounding, encompassing regular open-vocabulary segmentation to referring expression
segmentation, while avoiding specialized region-proposal networks (Ren et al., 2024a) or external segmentation models
(Kirillov et al., 2023) (§ 4.2).
Training an end-to-end model to perform segmentation and retrieval-augmented generation jointly is challenging:
optimization easily collapses, degrading segmentation accuracy or failing to learn the retrieval control (§5.3). We
address this with a two-stage training framework: Stage 1 preserves segmentation and visual grounding, and Stage 2
teaches the model when to retrieve, how to form queries, and how to attend to retrieved external knowledge in the
search-interleaved reasoning. This design enables stable optimization and yields an LMM that can dynamically balance
perception and knowledge reasoning. (Section 4.3)
4.2 PixSearch Model
Loss function.We extend the training objectives of segmenting LMM, which allowsPixSearchto maintain linguistic
coherence while achieving interpretable, text-conditioned segmentation (Blume et al., 2025):
L=L LM+λ1Lspan|{z }
sequence loss+λ2Lseg+λ3LBCE| {z }
segmentation loss+λ 4LKL|{z}
regularization.(6)
We next describe this formulation in detail. In thesequence supervision, LLMstands for the next-token cross-entropy
for the decoder, which we will describe in detail in Equation 10. The other term Lspanis a span-tagging loss for
grounding text to image regions. Specifically, let hL
i∈Rdbe the final-layer embedding of token i. A span extractor
applies bidirectional self-attention to predict BIO (B,I,O) tags for each token (Ramshaw and Marcus, 1999). We
train this tagger with cross-entropy Lspan; at inference, contiguous B→Ichains are merged into spans corresponding
to the referred object in the image.
Each resulting span S={(is, js)}N+
s=1is then projected to a set of “mask queries” qk=g(hL
k)∈Rmvia a learned
projection head g(·). The projected mask queries are fed into a mask decoder that predicts segmentation masks ˆMi.
The segmentation loss combines Focal-Tversky Abraham and Khan (2019) ( Lseg) and binary cross-entropy loss ( LBCE)
for mask prediction:
Lseg=1
N+X
yi̸=OLFT(Mi,ˆMi).(7)
Finally, to preserve alignment with the pretrained language space, we apply a Gaussian KL constraint, where tL
is:js
denotes frozen teacher embeddings.
LKL=1
N+N+X
s=1∥hL
is:js−tL
is:js∥2
2
2σ2.(8)
Information Token Masking.We modify the computation of LLMto decouple externally retrieved evidence from
direct optimization, allowing the model toconsumethe retrieved information as context while learning toreasonover
it rather than memorize or regurgitate it. For this purpose, we apply an information token masking scheme (Jin et al.,
2025): we define a binary maskm ∈ {0,1}Lover the input sequence of length L, masking out each token xibelonging
to aninformationspan from loss computation and gradient updates.
5

mi=(
0,if<info>⪯x i⪯</info>;
1,otherwise.(9)
We compute the masked language modeling loss as follows. Letx= ( x1, . . . , x L)denote the tokenized input sequence
andy= (y 1, . . . , y L)the target sequence.
LLM=−1P
imiLX
i=1mi·logp θ(yi|y<i, I),(10)
where pθdenotes the model’s conditional probability of predicting token yigiven previous context y<iand image
I. The maskmensures that gradients do not propagate through any tokens corresponding to retrieved information
segments, effectively detaching the retrieved payload from the autoregressive teacher forcing loop.
During batch collation, we computemdynamically using token offset mappings provided by the tokenizer to locate
character spans of <info>...</info> within each assistant response. Formally, for each conversation cwith assistant
textT c, we identify character-level spans
Sc={(s k, ek)}Kc
k=1,whereT c[sk:ek]∈[<info>,</info>].
Given the tokenizer offset mapΩ c={(s i, ei)}L
i=1, a tokeniis masked if and only if
∃(sk, ek)∈ S csuch that(s i< ek)∧(e i> sk).
The resulting per-conversation mask tensorm cis concatenated across all conversations to form the final batch-level
mask tensorM∈ {0,1}B×Lused to gate loss terms.
1.For <region> query types, sample the mask through the mask sampling pipeline2.For <text> query types, generate mul8ple query candidates and choose the query that corresponds to the answer that follows the respec8ve <search> callNeeds retrieval or not[Question] Is this car suitable for seating more than 7 people?Filter ques8ons based on ques8on typesBinary Ga)ngQues)on Filter
[Question]Is this car suitable for seating more than 7 people?LLMs[Need Retrieval] True (Or False)OCR ReadFactoid/KBMul3-hopFine-grained…[Question]Is this car suitable for seating more than 7 people?Each atomic ques8on forms an atomic reasoning step & Choose which sub-ques8on requires retrieval      or notQues)on Decomposi)on[Question]Is this car suitable for seating more than 7 people?Sub-Question #1: What is the model of this car?Sub-Ques3on #2: Can this car model seat more than 7 people?
Interleaving <search> tokens[Ques6on] Is this … 7 people?Output Response: the car shown in the image<search>is a Subaru WRX. This car <search>is a compact car with a total passenger capacity of 5 people, so it is not suitable.ICL samples guided the model to interleave <search> tokens into the model reasoning trajectory & response
LLMs[Ground-Truth Ans.] No …Sub-Question #2Sub-Ques3on #1[Sub-Ques6ons]
Per <search> call, assign the most suitable query type among <region>, <text> and <img>Query Type Assignment
<region><text>Output Response: the car shown in the image<search>is a Subaru WRX. This car <search>is a compact car with a total passenger capacity of 5 people, so it is not suitable.Query ConstructionOutput Response: the car shown in the image<search> <region> </search>is a Subaru WRX. This car is a Subaru WRX. This car <search><text> </search>is a compact car with a total passenger capacity of 5 …<text> 
LLMsOutput Response: the car shown in the image<search> <region> </search>is a Subaru WRX. This car is a Subaru WRX. This car <search>What is the seating capacity of Subaru WRX? </search>is a compact car with a total passenger capacity of 5 
LLMsRegion Extrac6on Pipeline<region> CarSubaru…
Step (i): Ques9on Selec9onStep (ii): Question Decomposition & Response Generation
Step (iii): Query Construc9on & Type AssignmentGround DINO2SAM
Figure 3 Data construction pipeline for Stage-2 training.
4.3 Two-stage Training
Model Initialization.Building upon the prior line of segmenting LMMs, we employ LLaVA (Liu et al., 2023) as our
multi-modal LLM backbone. We then initialize the parameters of our model with those of PLUM (Blume et al., 2025)
since it provides the state-of-the-art performance relative to existing segmenting LMMs in terms of visual reasoning
and provides a text-aligned mask decoder in tandem.
Stage 1: Mask Generation.The mask segmentation performance can suffer from a catastrophic forgetting issue
(§5.3), requiring us to construct a training dataset that enablesPixSearchto retain its visual reasoning and mask
6

generation capabilities. We include the following datasets into the mixture for mask generation to enable segmentation
on the nuanced language: ADE20k (Zhou et al., 2017), Pascal Parts (Chen et al., 2014), PartImageNet (He et al.,
2022), PACO-LVIS (Ramanathan et al., 2023), COCO-Stuff (Caesar et al., 2018), along with RefCOCO variants
(Kazemzadeh et al., 2014). Our training mixture also employs a visual instruction tuning dataset from LLaVA (Liu
et al., 2023), which consists of 665k textual responses and captions given an image, enabling our segmenting LMM to
retain its general visual understanding and reasoning ability.
Stage 2: Search-Interleaved Reasoning.Training in Stage 2 aims to teach our model when to trigger search and
when to construct a visually-grounded multimodal query (i.e., <region> ,<image> ). Consider the example question
“What is the conservation status of this animal?”, generating a textual caption of the long-tail animal directly for
retrieval, instead of composing a multi-modal search query, could lead to hallucination (Kim and Ji, 2024).
Figure 3 depicts the process of training data generation for Stage 2. We start with samples from the following datasets:
TextVQA (Singh et al., 2019), InfoSeek (Chen et al.), OVEN (Hu et al., 2023), CRAG-MM (Wang et al., 2025),
WebQA (Chang et al., 2022), OKVQA (Marino et al., 2019) and A-OKVQA (Schwenk et al., 2022). For each sample,
we construct training data in three steps.
(i)Question Selection:We prompt a proprietary LMM2with in-context learning to determine if the question can
benefit from external knowledge (e.g.,Where is this plant native to?) by classifying the question into 10 pre-defined
VQA question types (refer to supplementary for detail), and retain only the questions that belong to Multi-hop
External Knowledge Reasoning, Fine-grained Entity Identification, Factoid/KB Questions, where retrieval is mostly
needed to identify entities present in the image and require external knowledge associated with the entities to correctly
answer the questions.
(ii)Question Decomposition & Response Generation:We decompose questions into multiple atomic sub-questions
and determine independently per sub-question whether retrieval is needed. Then, we feed the original question,
sub-questions, and ground-truth answers into the prompt, and instruct LMMs to generate a <search> -interleaved
reasoning trajectory (i.e., response). Here, we generate N(N= 5) such trajectories and feed it back to the model for
self-refinement loop (Madaan et al., 2023), selecting the best response where the <search> token was appropriately
placed in the reasoning trajectory when search is deemed necessary.
(iii)Query Construction & Type Assignment:Guided by the in-context learned (ICL) samples, the model assigns
a pseudo-gold query type to each <search> token that appears in the final response. We assign <region> when
the question refers to an entity present in the input image, and assign <image> if the question requires a holistic
understanding of the whole image. For text queries, we prompt LLMs to generate based on the immediately preceding
text before the corresponding<search>token.
5 Experiments
Through comprehensive experiments, we answer the following research questions.
RQ1:IsPixSearcheffective on question answering across factual VQA, general VQA, and text-only QA?
RQ2:How much do region-level retrieval and multi-step search contribute toPixSearch’s performance?
RQ3:DoesPixSearchpreserve the segmentation capability?
5.1 Experiment setup
Dataset.We experimented with four VQA benchmarks CRAG-MM (Wang et al., 2025), TextVQA (Singh et al., 2019),
InfoSeek (Chen et al.), OVEN (Hu et al., 2023) and four text-only QA benchmarks (HotpotQA (Yang et al., 2018),
NQ (Kwiatkowski et al., 2019; Joshi et al., 2017), PopQA (Mallen et al., 2023), MuSiQue (Trivedi et al., 2022)); see the
Appendix for details. We constructed our own search API using 6M Wikipedia documents3and their corresponding
images, with DINOv3 (Siméoni et al., 2025) as our image encoder and MPNet4as the text encoder. Each embedding
in the image index is linked to a corresponding Wikipedia document such that an image-to-image retrieval returns the
corresponding document. For CRAG-MM (Wang et al., 2025), we use the search API provided by the benchmark5.
2we usegpt-4.1in this work
3Derived from: Wikidump 2022/10/01
4We use thesentence-transformers/all-mpnet-base-v2
5https://www.piwheels.org/project/cragmm-search-pipeline/
7

Table 1PixSearch obtains the highest accuracy and lowest hallucination, and thus highest truthfulness. All metrics in
% and Truthfulness ∈[−100,100]. For GroundedSAM (Ren et al., 2024a) + LLaVA-13B (Liu et al., 2023), we use
gpt-4.1to extract the key entities for mask generation.
ModelsCRAG-MM (Overall) CRAG-MM (Egocentric) CRAG-MM (Non-Egocentric)
Truth.↑Acc.↑Miss.↓Hallu.↓ Truth.↑Acc.↑Miss.↓Hallu.↓ Truth.↑Acc.↑Miss.↓Hallu.↓
Llama-3.2-11B-Vision(Grattafiori et al., 2024)
Llama-3.2-11B No Search -16.9 24.4 34.4 41.3 -22.5 21.0 35.5 43.5 0.1 34.2 31.7 34.1
Llama-3.2-11B Whole Image -8.6 35.3 20.3 43.9 -17.0 31.0 21.0 48.0 11.5 46.6 18.3 35.1
Pipeline Method for Region-Level MM-RAG
GroundedSAM+LLaVA-13B -30.4 23.4 22.8 53.8 -41.7 18.1 22.0 59.8 -23.6 25.7 25.0 49.3
LLaVA-13B Backbone(Liu et al., 2023)
LLaVA-13B -69.0 10.310.479.3 -72.8 8.111.080.9 -59.1 16.18.775.2
PLUM-13B -35.2 25.3 14.5 60.5 -43.2 20.5 15.8 63.7 -10.1 39.4 11.0 49.5
PixSearch-13B No Search -40.4 19.8 20.0 60.2 -47.8 15.7 20.8 63.5 -18.8 31.7 17.8 50.5
PixSearch-13B Text Query -44.5 18.2 19.1 62.7 -52.5 12.2 20.1 67.7 -14.8 34.4 16.4 49.2
PixSearch-13B Whole Image -19.0 31.5 18.1 50.5 -27.6 26.7 19.0 54.3 5.1 44.8 15.5 39.7
PixSearch-13B Full -3.0 37.721.640.7 -11.7 33.022.344.7 20.5 45.918.725.4
Evaluation.Following previous works, we evaluatedrelaxed Exact Match (EM)andF1 score(Kim and Ji, 2024).
For CRAG-MM, we use theTruthfulnessscore proposed by the benchmark (Wang et al., 2025), where each answer is
classified intoaccurate(score=1),missing(score=0) andhallucination(score=-1), and truthfulness is the average score
τ∈[−1.0,1.0]. In Table 1, we multiply Truthfulness by 100 for consistency with rest of the other metrics on the table.
Implementations.We comparedPixSearchwith a few baselines, including LLM-only solutions (LLaVA-13B (Liu
et al., 2023), PLUM-13B (Blume et al., 2025), Llama-3.2-11B Grattafiori et al. (2024)), baseline MM-RAG solutions
using the whole image for search. We also separately implement GroundedSAM (Ren et al., 2024a) + LLaVA-13B
baseline so we can test against the pipeline baseline; older pipeline methods such as REVIVE (Lin et al., 2022)
rely on weaker grounding models such as GLIP (Li et al., 2022) and FiD (Izacard and Grave, 2021). We detail
the implementation of the pipeline in Appendix. We also compared with ablated version of ourPixSearchmodel,
including (i) No Search: the finetunedPixSearchmodel without search triggering; (ii) Text Question: always using
the text question as the search query, wherein the retrieval results are prepended to the input question for output
generation; (iii) Whole Image: using the input image as the search query instead.
5.2 Overall performance (RQ1, RQ2)
Table 1 comparesPixSearch-13B Fullagainst VQA and MM-RAG baselines and PixSearch ablations. First, PixSearch
achieves the highest accuracy and lowest hallucination, and thus highest truthfulness, across ego-centric and normal
images. Second, whereas the small-sized models in general has low qualities, e.g., LLaVA and PLUM substantially
fall behind Llama-3.2 of smaller size,PixSearch Fullbeats Llama-3.2 for both the LLM-only solution and the
straightforward MM-RAG solution, which retrieves external knowledge using the whole image as a query. Third,
PixSearch Text Query gives similar and even slightly worse results toPixSearch No Search, validating that text-only
search is ineffective for MM-RAG, especially for ego-centric images. Fourth,PixSearch Whole Image is a strong ablation
and significantly improves over no-search; however,PixSearchthat can invoke multiple searches and can search only
a masked region still considerably outperforms searching the whole image. Furthermore, compared to the pipelined
approach of using GroundedSAM (Ren et al., 2024a) to extract image regions and a comparable LLaVA backbone to
reason upon them, ourPixSearch Fullexhibits substantially better performance, suggesting that interleaved text and
mask-based extraction reasoning is beneficial for enhancing the factuality of LMMs. Finally, we observe a big quality
gap between ego-centric images and non-egocentric images, illustrating the challenges faced by VQA on ego-centric
images when we apply a smaller model. Table 8 provides two examples illustrating how PixSearch works, compared
against the baseline backbone models.
8

Table 2Semantic Segmentation performance in mIoU. Percentage change (∆%) computed relative to PLUM. We
denote the sampling ratio for Stage-1 and Stage-2 training mixtures as (Stage-1:Stage-2) in the PixSearch rows.
Models ADE20K COCOStuff
MaskFormer (Cheng et al., 2021) 52.70 47.02
PLUM (Blume et al., 2025) 55.08 49.97
PixSearch(1:9) 9.85 (–82.1%) 11.07 (–77.8%)
PixSearch(5:9) 47.19 (–14.3%) 42.55 (–14.9%)
PixSearch(7:9)55.98(+1.6%)49.81(–0.3%)
Table 3Referring Expression Segmentation performance in gIoU.PixSearchretains competitive segmentation
performance across referring expression segmentation. We denote the sampling ratio for Stage-1 and Stage-2 training
mixtures as (Stage-1:Stage-2) in thePixSearchrows.
ModelsExtra
Segmentation DataRefCOCO
(val)RefCOCO
(testA)RefCOCO
(testB)RefCOCOg
(val(U))RefCOCOg
(test(U))RefCOCO+
(val)RefCOCO+
(testA)
LISA-7B (Lai et al., 2024)✗74.1 76.5 71.1 66.4 68.5 62.4 67.4
GLaMM (Rasheed et al., 2024)✓79.5 83.2 76.9 74.2 74.9 72.6 78.7
PixSearch(1:9)✗15.3 13.4 11.2 9.7 10.2 13.6 12.3
PixSearch(7:9)✗78.9 79.3 72.7 69.8 70.9 65.9 69.9
5.3 Mask Segmentation (RQ3)
For semantic segmentation, we evaluate our models on ADE20K (Zhou et al., 2019) and COCOStuff (Caesar et al.,
2018). Table 2 shows that PixSearch obtains similar segmentation quality compared to state-of-the-art semantic
segmentation model, MaskFormer, on the two benchmarks. Additionally, Table 3 shows that PixSearch performs
comparably against strong segmenting LMM baselines like GLaMM (Rasheed et al., 2024), outperforming other
variants such as LISA by a notable margin. We note that the sampling ratio of the Stage-1 and Stage-2 training
mixtures during the proposed curriculum learning framework has a substantial effect on the model’s mask prediction
performance. In this study, setting the sampling ratio between Stage-1 and Stage-2 training samples to 7:9 gave
the best segmentation performance. The results suggest that PixSearch’s mask decoder component can serve as a
standalone mask generator that enables the pixel-level grounding for both noun-phrases and referring expressions.
5.4 Robustness & Transferability (RQ1)
VQA on additional benchmarks.Table 5 compared PixSearch with other methods on other VQA benchmarks,
ranging from common-sense reasoning questions (TextVQA Abraham and Khan (2019)) to entity-centric knowledge-
dependent questions (InfoSeek (Chen et al.), OVEN (Hu et al., 2023)). On TextVQA, where search is not needed,
PixSearch slightly outperforms its segmenting LMM backbone (PLUM-13B, also trained on top of Llava-13B with the
same instruction-tuning dataset). On InfoSeek and OVEN, which can significantly benefit from external knowledge,
PixSearch outperforms its non-RAG backbone; since these two benchmarks mainly contain images focused on the
queried entities, whole image search is adequate and PixSearch does not regress its counterpart with whole image
search (PixSearch Whole Image ).
Textual QA.We further test PixSearch on out-of-domain text-only QA benchmarks, which require textual reasoning
and benefit from external knowledge grounding. Table 4 demonstrates PixSearch’s strong performance across multi-
hop reasoning (HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022)), open-domain question answering
(NQ (Kwiatkowski et al., 2019; Joshi et al., 2017)), and entity-centric question answering (PopQA (Mallen et al., 2023)).
The results suggest that our model, despite not being trained on text-only QA data, retains its textual reasoning
capability.
5.5 Search Pattern Analysis (RQ2)
In CRAG-MM dataset, we can divide the images into two buckets: egocentric and non-egocentric. We divide the
search pattern analysis into two subgroups: (i) search call frequency: the number of <search>tokens per sample; (ii)
proportion of <region>, <img>, textual query calls generated during the search-interleaved reasoning. At the top of
9

Table 4 Text-only QA Performance Evaluation (EM and F1)across open-domain QA benchmarks. Bold indicates the best in
each column.
ModelsHotpotQA NQ PopQA MuSiQue
EM F1 EM F1 EM F1 EM F1
LLaVA-13B 8.33 10.25 10.18 11.06 4.70 7.98 5.70 6.35
PLUM-13B 15.30 18.80 15.00 18.35 7.40 9.58 6.60 8.43
PixSearch-13B No Search 23.50 26.79 30.05 32.90 17.20 17.66 9.80 12.69
PixSearch-13B Full 26.20 27.79 30.90 33.38 31.60 32.18 11.50 12.75
Table 5 Performance on additional VQA benchmarks.Scores are reported in %.
Models TextVQA InfoSeek OVEN
EM F1 EM F1 EM F1
LLaVA-13B Backbone
LLaVA-13B 22.84 26.15 13.94 14.98 2.20 3.12
PLUM-13B 30.11 32.62 15.60 16.54 2.64 3.81
PixSearch-13B No Search 32.7634.55 18.34 19.40 3.82 5.45
PixSearch-13B Question 30.87 32.90 20.73 22.30 5.27 6.83
PixSearch-13B Whole Image 32.30 34.13 24.28 26.05 15.3117.08
PixSearch-13B Full 32.4435.10 24.55 26.62 15.8817.00
Egocentric Non-Egocentric02468# of <search> tokensDistribution of <search> Call Frequency per Sample
0.00 0.05 0.10 0.15 0.20 0.25 0.30
Average Count per Sampleregion
text
imgQuery TypeAverage Query Type Counts per Subset
Subset
Egocentric
Non-Egocentric
Figure 4Search Behavior Plot from thePixSearch Interleaved outputs for CRAG-MM evaluation set.
Figure 4, we evidence that while egocentric images trigger mainly 3 to 4 search calls (bottom sample of Table 8), whereas
non-egocentric images generally trigger 2 to 3 search calls. The pattern suggests that egocentric images in CRAG-MM
typically demand more than just identifying the target entity in question, but rather requires additional information
and reasoning to answer the questions. In Figure 4, we can infer that both the egocentric and non-egocentric images
assign heavy probability mass to <region>tokens. Moreover, textual queries also take up around 24% of the instances
for the egocentric case, which is mainly because of follow-up searches for the same question are often text queries, as
qualitatively attested by cases in Table 8.
5.6 Ablation Studies
Removal of Question TypesTable 6 evaluates the contribution of different search query modalities by restricting
the types of queriesPixSearchcan issue at inference time. Among single-modality settings,PixSearch Only Region
performs closest to the full model, especially on egocentric images, confirming that pixel-grounded region retrieval is
the primary driver of performance in visually cluttered, entity-centric scenes. In contrast,PixSearch Only Textshows
the largest degradation, with substantially lower accuracy and higher hallucination, indicating that text-only queries
are often insufficient for precise entity retrieval.
Whole-image retrieval (PixSearch Only Image) performs better than text-only but remains clearly inferior to region-based
search, highlighting the limitations of coarse visual queries. Removing text queries (PixSearch No Text) results in
only a small drop relative to the full model, suggesting that textual queries mainly serve as complementary follow-up
searches. Similarly, removing whole-image queries (PixSearch No Image) causes a modest degradation, particularly on
10

Table 6 Ablation study on question type removal.We restrictPixSearchfrom issuing specific types of search queries
during decoding and measure the performance degradation. All metrics are reported in %.
ModelsCRAG-MM (Overall) CRAG-MM (Egocentric) CRAG-MM (Non-Egocentric)
Truth.↑Acc.↑Miss.↓Hallu.↓ Truth.↑Acc.↑Miss.↓Hallu.↓ Truth.↑Acc.↑Miss.↓Hallu.↓
Question Type Removal (Inference-Time Constraint)
PixSearch-13B Full -3.0 37.721.640.7 -11.7 33.022.344.7 20.5 45.918.725.4
PixSearch-13B No Region -14.5 31.9 21.7 46.4 -23.0 27.5 22.0 50.5 8.0 43.5 21.0 35.5
PixSearch-13B No Text -6.3 35.9 21.9 42.2 -11.9 32.8 22.5 44.7 8.5 44.0 20.5 35.5
PixSearch-13B No Image -7.5 34.7 23.1 42.3 -13.0 32.0 23.0 45.0 7.0 42.0 23.0 35.0
PixSearch-13B Only Region -6.4 35.9 21.9 42.3 -12.2 32.6 22.6 44.8 9.0 44.5 20.0 35.5
PixSearch-13B Only Text -44.4 18.3 19.1 62.6 -55.5 12.2 20.1 67.7 -14.8 34.3 16.5 49.2
PixSearch-13B Only Image -18.7 31.7 18.0 50.3 -27.6 26.7 19.0 54.3 5.1 44.8 15.5 39.7
Table 7 Ablation study on the number of search tokens.We limit the maximum number of <search>calls allowed during
decoding and evaluate how multi-step search contributes to performance. All metrics are reported in %.
ModelsCRAG-MM (Overall) CRAG-MM (Egocentric) CRAG-MM (Non-Egocentric)
Truth.↑Acc.↑Miss.↓Hallu.↓ Truth.↑Acc.↑Miss.↓Hallu.↓ Truth.↑Acc.↑Miss.↓Hallu.↓
Search Budget Constraint
PixSearch-13B No Search(B=0) -40.4 19.8 20.0 60.2 -47.8 15.7 20.8 63.5 -18.8 31.7 17.8 50.5
PixSearch-13B Search≤1 -7.9 32.1 21.8 43.3 -16.5 28.0 22.9 49.1 15.0 43.0 18.9 28.0
PixSearch-13B Search≤2 -3.5 35.8 21.6 39.9 -12.4 32.0 22.8 45.2 20.0 46.0 18.4 26.0
PixSearch-13B Search≤3 -3.0 36.4 21.5 39.5 -12.0 32.7 22.6 44.7 20.8 46.2 18.5 25.7
PixSearch-13B Search≤4 -2.9 36.5 21.4 39.4 -11.8 32.9 22.5 44.6 20.6 46.0 18.6 25.6
PixSearch-13B Full(unbounded) -3.0 37.721.640.7 -11.7 33.022.344.7 20.5 45.918.725.4
non-egocentric images, where global context can be helpful.
Variations in the Number of Search Tokens.Table 7 studies how limiting the number of allowed search calls
affects performance. Allowing even a single search substantially improves over the no-search baseline, demonstrating
the importance of external knowledge retrieval. Performance continues to improve as the search budget increases, with
most gains realized within two to three search calls, particularly for egocentric images that require iterative entity
identification and reasoning.
Beyond three to four searches, performance saturates and closely matches the unbounded full model, with truthfulness
differing by only a small margin. While tighter search budgets slightly increase the missing rate, hallucination remains
relatively stable once minimal retrieval is enabled, indicating that additional searches primarily improve answer
completeness rather than merely reducing errors.
6 Conclusion
In this work, we introducedPixSearch, an end-to-end Large Multimodal Model that unifies region-level perception
and retrieval-augmented reasoning within a single framework. Unlike prior pipeline-based, tool-based or API-dependent
approaches, PixSearch learns to autonomously decide when retrieval is needed and how to formulate modality-aware
queries, i.e., region-based crop, whole-image, or textual, while retaining its pixel-level grounding capabilities via mask
segmentation that is a part of the proposed model. Through a two-stage training framework and the construction of a
search-interleaved reasoning dataset,PixSearchintegrates segmentation and retrieval abilities without sacrificing
visual understanding or mask prediction quality. Our experiments demonstrate thatPixSearchachieves competitive
segmentation performance across segmentation benchmarks and substantially outperforms prior LMMs and retrieval
augmented baselines on a wide range of visual and textual question answering tasks.PixSearchlays a foundation for
more factual, pixel-grounded multimodal understanding of LMM agents.
11

Table 8 Qualitative comparison between baseline and PixSearch-Interleaved outputs.Each example shows the input image
(left), the corresponding question and model responses (middle), and the pixel-level grounding mask generated by the
model (right). The payloads between the <information></information>are truncated to show entity names / titles
only for brevity.
Input Image Question & Model OutputsPixSearch Mask
Overlay
Q:What is the origin of this food item?
PLUM-13B:It looks like a pasta with toppings and sauce on top of it.
PixSearch-13B Full:The origin of the food item shown in the image
<search><region></search><information>Carne asada fries | Bacon
Cheeese Fries | Restaurant | ... </information>is bacon cheese fries. The
food item in the image<search>what is the origin of cheese bread or cheese
fries? </search><information>Southwest US cheese fries | McDonald’s
bacon cheese fries 2004 | ... </information>is a type of dish that originated
in the United States.
GT:Bacon cheese fries are from the United States, but the exact origin of
the dish is not known.
Q:Did this vehicle appear in the movie Pretty Woman back in the 90s?
LLaVA-13B:The white hatchback did not appear in the movie Pretty Woman.
PixSearch-13B FullThe vehicle shown in the image is a white SUV. This ve-
hicle<search><region></search><information>Lincoln MKS 2009-2016
| V6 engine 3.7L 3.5L | Toyota Rav4 2014 | Toyota RAV4 | Chicago As-
sembly | all-wheel drive </information>is a Toyota RAV4. The Pretty
Woman was released<search>when was the Pretty Woman released?
</search><information>Pretty Woman | Notting Hill | Movies in 90s </in-
formation>in 1989. The Toyota RAV4 was launched<search>when did
the Toyota RAV4 launch? </search><information>Toyota RAV4 | Toyota
| Toyota Motor Corporation | Japan Car of the Year | gasoline engine </in-
formation>in 1994. Therefore, this vehicle is not the vehicle that appears
in the movie Pretty Woman.
GT:No, because the first-generation Toyota RAV 4 was launched in 1994,
four years after the release date of the movie Pretty Woman on March 23,
1990.
12

References
Nabila Abraham and Naimul Mefraz Khan. A novel focal tversky loss function with improved attention u-net for
lesion segmentation. In2019 IEEE 16th international symposium on biomedical imaging (ISBI 2019), pages 683–687.
IEEE, 2019.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate,
and critique through self-reflection. InThe Twelfth International Conference on Learning Representations.
Ansel Blume, Jeonghwan Kim, Hyeonjeong Ha, Elen Chatikyan, Xiaomeng Jin, Khanh Duy Nguyen, Nanyun
Peng, Kai-Wei Chang, Derek Hoiem, and Heng Ji. PARTONOMY: Large multimodal models with part-level
visual understanding. InThe Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.
https://openreview.net/forum?id=yjLew3Nd7z.
Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. InComputer vision
and pattern recognition (CVPR), 2018 IEEE conference on. IEEE, 2018.
Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop
and multimodal qa. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
16495–16504, 2022.
Xianjie Chen, Roozbeh Mottaghi, Xiaobai Liu, Sanja Fidler, Raquel Urtasun, and Alan Yuille. Detect what you can:
Detecting and representing objects using holistic models and body parts. InProceedings of the IEEE conference on
computer vision and pattern recognition, pages 1971–1978, 2014.
Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can
pre-trained vision and language models answer visual information-seeking questions? InThe 2023 Conference on
Empirical Methods in Natural Language Processing.
Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic
segmentation.Advances in neural information processing systems, 34:17864–17875, 2021.
Feng Gao, Qing Ping, Govind Thattai, Aishwarya Reganti, Ying Nian Wu, and Prem Natarajan. Transform-retrieve-
generate: Natural language-centric outside-knowledge visual question answering. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 5067–5077, 2022.
François Gardères, Maryam Ziaeefard, Baptiste Abeloos, and Freddy Lecue. Conceptbert: Concept-aware representation
for visual question answering. InFindings of the Association for Computational Linguistics: EMNLP 2020, pages
489–498, 2020.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783, 2024.
Liangke Gui, Borui Wang, Qiuyuan Huang, Alexander Hauptmann, Yonatan Bisk, and Jianfeng Gao. Kat: A knowledge
augmented transformer for vision-and-language. InNAACL, 2022.
Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang,
Daniel Kang, and Heng Ji. Mm-poisonrag: Disrupting multimodal rag with local and global poisoning attacks.
arXiv preprint arXiv:2502.17832, 2025.
Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang,
Qihang Yu, and Alan Yuille. Partimagenet: A large, high-quality dataset of parts. InEuropean Conference on
Computer Vision, pages 128–145. Springer, 2022.
Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei
Chang. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. InProceedings of
the IEEE/CVF International Conference on Computer Vision, pages 12065–12075, 2023.
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question
answering. InProceedings of the 16th conference of the european chapter of the association for computational
linguistics: main volume, pages 874–880, 2021.
Pu Jian, Donglei Yu, and Jiajun Zhang. Large language models know what is key visual entity: An llm-assisted
multimodal retrieval for vqa. InProceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing, pages 10939–10956, 2024.
13

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han.
Search-r1: Training llms to reason and leverage search engines with reinforcement learning.Conference on Language
Modeling, 2025.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised
challenge dataset for reading comprehension. InProceedings of the 55th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.
Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in
photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors,Proceedings of the
2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, Doha, Qatar,
October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1086. https://aclanthology.org/
D14-1086/.
Jeonghwan Kim and Heng Ji. Finer: Investigating and enhancing fine-grained visual concept recognition in large
vision language models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors,Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing, pages 6187–6207, Miami, Florida, USA, November
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.356. https://aclanthology.org/
2024.emnlp-main.356/.
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer
Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. InProceedings of the IEEE/CVF International
Conference on Computer Vision, pages 4015–4026, 2023.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-
Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for
question answering research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi:
10.1162/tacl_a_00276.https://aclanthology.org/Q19-1026/.
Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation
via large language model. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 9579–9589, 2024.
Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan,
Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 10965–10975, 2022.
Yuanze Lin, Yujia Xie, Dongdong Chen, Yichong Xu, Chenguang Zhu, and Lu Yuan. Revive: Regional visual
representation matters in knowledge-based visual question answering.Advances in neural information processing
systems, 35:10560–10571, 2022.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri,
Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback.Advances in Neural
Information Processing Systems, 36:46534–46594, 2023.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust
language models: Investigating effectiveness of parametric and non-parametric memories. InProceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, 2023.
Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering
benchmark requiring external knowledge. InProceedings of the IEEE/cvf conference on computer vision and pattern
recognition, pages 3195–3204, 2019.
Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. Krisp: Integrating implicit and
symbolic knowledge for open-domain knowledge-based vqa. InCVPR, pages 14111–14121, 2021.
Medhini Narasimhan and Alexander G Schwing. Straight to the facts: Learning knowledge base retrieval for factual
visual question answering. InProceedings of the European conference on computer vision (ECCV), pages 451–468,
2018.
Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez,
14

Rama Kovvuri, Abhishek Kadian, et al. Paco: Parts and attributes of common objects. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7141–7151, 2023.
Lance A Ramshaw and Mitchell P Marcus. Text chunking using transformation-based learning. InNatural language
processing using very large corpora, pages 157–176. Springer, 1999.
Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M
Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.
Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng
Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks.arXiv preprint arXiv:2401.14159,
2024a.
Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel
reasoning with large multimodal model. InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 26374–26383, 2024b.
Ander Salaberria, Gorka Azkune, Oier Lopez de Lacalle, Aitor Soroa, and Eneko Agirre. Image captioning for effective
use of language models in knowledge-based visual question answering.Expert Systems with Applications, 212:118669,
2023. ISSN 0957-4174. doi: https://doi.org/10.1016/j.eswa.2022.118669. https://www.sciencedirect.com/science/
article/pii/S0957417422017055.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer,
Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools.Advances
in Neural Information Processing Systems, 36:68539–68551, 2023.
Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A
benchmark for visual question answering using world knowledge. InEuropean conference on computer vision, pages
146–162. Springer, 2022.
Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. Kvqa: Knowledge-aware visual
question answering.Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):8876–8884, Jul. 2019. doi:
10.1609/aaai.v33i01.33018876.https://ojs.aaai.org/index.php/AAAI/article/view/4915.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau
Yih. Replug: Retrieval-augmented black-box language models. InProceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), pages 8364–8377, 2024.
Mankeerat Sidhu, Hetarth Chopra, Ansel Blume, Jeonghwan Kim, Revanth Gangi Reddy, and Heng Ji. Search and
detect: Training-free long tail object detection via web-image retrieval. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 15129–15138, 2025.
Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc
Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3.arXiv preprint arXiv:2508.10104, 2025.
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus
Rohrbach. Towards vqa models that can read. InProceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 8317–8326, 2019.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via
single-hop question composition.Transactions of the Association for Computational Linguistics, 10:539–554, 2022.
Jiaqi Wang, Xiao Yang, Kai Sun, Parth Suresh, Sanat Sharma, Adam Czyzewski, Derek Andersen, Surya Appini,
Arkav Banerjee, Sajal Choudhary, et al. Crag-mm: Multi-modal multi-turn comprehensive rag benchmark.arXiv
preprint arXiv:2510.26160, 2025.
XuDong Wang, Shaolun Zhang, Shufan Li, Konstantinos Kallidromitis, Kehan Li, Yusuke Kato, Kazuki Kozuka, and
Trevor Darrell. Segllm: Multi-round reasoning segmentation.arXiv preprint arXiv:2410.18923, 2024.
Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training
and benchmarking universal multimodal information retrievers. InEuropean Conference on Computer Vision, pages
387–404, 2024.
Jialin Wu, Jiasen Lu, Ashish Sabharwal, and Roozbeh Mottaghi. Multi-modal answer validation for knowledge-based
vqa (mavex). InAAAI, pages 2725–2733, 2022.
15

Yibin Yan and Weidi Xie. Echosight: Advancing visual-language models with wiki knowledge. InFindings of the
Association for Computational Linguistics: EMNLP 2024, pages 1538–1551, 2024.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D
Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.
Zhuolin Yang, Wei Ping, Zihan Liu, Vijay Korthikanti, Weili Nie, De-An Huang, Linxi Fan, Zhiding Yu, Shiyi Lan,
Bo Li, et al. Re-vilm: Retrieval-augmented visual language model for zero and few-shot image captioning. In
Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11844–11857, 2023.
Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. Retrieval-augmented multimodal language modeling. InProceedings of the 40th
International Conference on Machine Learning, pages 39755–39769, 2023.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan
Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. InThe Thirteenth
International Conference on Learning Representations.
Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel
understanding with visual instruction tuning. InProceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 28202–28211, June 2024.
Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through
ade20k dataset. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.
Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic
understanding of scenes through the ade20k dataset.International Journal of Computer Vision, 127(3):302–321,
2019.
16

Appendix
A Hyperparameters and Compute Details
In Table 9, we detail the hyperparameter settings forPixSearchand our backbone model, PLUM (Blume et al.,
2025).
Hyperparameter PLUM PixSearch
Backbone
Language model LLaVA-13B LLaVA-13B (PLUM init.)
Vision tower CLIP ViT-L/14 CLIP ViT-L/14
Mask decoder SAM ViT-H SAM ViT-H
Training schedule
Input resolution1024210242
Max text length 512 512
Precision bf16 bf16
Epochs 25 + 4 Stage-1: 20, Stage-2: 6
Batch size 6 6
Grad. accumulation 10 10
Optimizer
Optimizer AdamW AdamW
LR3×10−42×10−4(S1),1×10−4(S2)
Betas (0.9, 0.95) (0.9, 0.95)
Weight decay 0 0
Loss weights
λCE 1.0 1.0
λseg 8.0 8.0
λBCE 2.0 2.0
λKL 0.1 0.1
λcls 2.0 2.0
Modules
BIO span tagger✓ ✓
Bidirectional encoder 2048 2048
Feedback Loop✓ ✓
Trainable SAM components decoder+prompt enc. decoder+prompt enc.
LoRA on LM (q,v)r= 8r= 8
Table 9 Hyperparameters for PLUM and PixSearch.
B Detailed Explanation of the Decode-with-Retrieval Algorithm
Algorithm 1 describes the search–interleaved decoding mechanism used by PixSearch. Below is a detailed walkthrough.
Autoregressive decoding with retrieval control.At each decoding step, the model autoregressively predicts the next
token. If the token is a normal language token, decoding continues normally. When the model emits the special token
<search>, it signals that retrieval is needed.
Stack-based parsing of retrieval spans.Each retrieval request is enclosed within <search> ...</search> . To correctly
pair them (especially when multiple retrieval calls occur in a single answer), a stack is maintained. The index of
each<search>token is pushed on the stack; when the model later emits</search>, that interval defines a payload
containing the retrieval modality and/or textual query.
Determining retrieval modality.Inside the<search>block, the model emits one of:
•<image>: retrieve using the entire image.
•<region> : call the mask decoder to predict a segmentation mask for the referred entity; crop the image using
the mask.
•<text>: use the generated textual span as a query.
1

Algorithm 1Decode with Retrieval
Require:
M: multimodal model,I: input image
search_api(q, k): retrieval function
Ensure:
Generated sequence augmented with retrieved info
1:gen←prompt_with_image(I),stack←[ ]
2:whilenot EOS and steps remainingdo
3:tok←M.generate_next(gen);gen←gen+tok
4:ifends with “<search>”thenpush index ontostack
5:else ifends with “</search>” andstacknot emptythen
6:payload←slice(gen,pop(stack))
7:mode←parse_payload(payload)
8:ifmode= “<region>”then
9:mask←predict_mask(last_entity(gen), I);query←crop(I,mask)
10:else ifmode= “<img>”then
11:query←I
12:else
13:query←payload_text
14:end if
15:info←format(search_api(query, k))
16:gen←append(gen,<information>+info+</information>)
17:end if
18:end while
19:returngen
Executing retrieval.The system then calls search_api(query, k) , where kis the number of returned documents.
Retrieved snippets are formatted and injected back into the generation sequence using <information> ...</informa-
tion>.
Search-interleaved reasoning.PixSearch may perform multiple retrievals throughout a single answer. Retrieved
evidence stays in the context, enabling multihop reasoning grounded in both the image and external knowledge.
Termination.Decoding continues until an end-of-sequence token is reached.
C Explanation of the Ten Visual Question Types
Table 10 lists the ten question categories used for Stage-2 SFT construction. Below we provide expanded definitions.
OCR Read.Questions requiring verbatim transcription of scene text.
OCR + Visual Reasoning.Requires interpreting the meaning of text in context (e.g., a scoreboard).
Multi-hop External Knowledge Reasoning.Requires chaining two or more knowledge lookups (e.g., identify entity→
retrieve its founding date).
Fine-grained Entity Identification.Entity-level identification often requiring region-level cropping.
Visual Reasoning – Attribute.Recognition of visible attributes (color, shape, texture).
Visual Reasoning – Counting.Counting objects in the image.
Visual Reasoning – Binary.Yes/no visual questions.
Social Commonsense Reasoning.Inferring human motivations or social context.
2

Table 10 Overview of the ten visual question types used in Stage-2.
Question Type Example Question
OCR Read“What does it say near the tail of the plane?”
OCR + Visual Reasoning“Which team is winning the game?”
Multi-hop External Knowledge Reasoning“When was the soft-drink company shown first created?”
Fine-grained Entity Identification“What class of animal is this creature?”
Visual Reasoning – Attribute“What is the color of the car in the background?”
Visual Reasoning – Counting“How many cars are there in the image?”
Visual Reasoning – Binary“Is the man in the image wearing a hat?”
Social Commonsense Reasoning“Why might the seated man have trouble getting around?”
Physical Commonsense Reasoning“What could block the washer’s door?”
Factoid / KB Questions“Hot dogs were invented in which country?”
Physical Commonsense Reasoning.Inferring physical affordances, constraints, and outcomes.
Factoid / KB Questions.Open-domain factual questions referencing entities in or implied by the image.
D In-Context Learning Samples
All examples are displayed in Tables 11, 12, 13, 14
D.1 Question Selection ICL Examples
To teach the modelwhenretrieval is needed, we constructed Question Selection examples that expose the model to
a diverse set of visual questions drawn from CRAG-MM, OK-VQA, and InfoSeek. Each example pairs a question
with an image and a binary label indicating whether external knowledge is required. The key design principle is that
retrieval is only beneficial when the image alone cannot resolve the question.
Thus, the examples include: (1) fine-grained or long-tail entity identification tasks (e.g., identifying car models, drink
brands, or rare animals), which require region-level or whole-image search; (2) multi-step factual or encyclopedic
queries (e.g., historical dates, object origins), where knowledge beyond the image is essential; and (3) questions solvable
purely from visual inspection (e.g., “Translate this”, “What is the couch made of?”, “What grade is the child in?”),
where retrieval would be unnecessary or potentially harmful.
By contrasting retrieval and no-retrieval cases with high visual similarity, the model learns a robust policy for deciding
whento trigger<search>calls.
D.2 Question Decomposition ICL Examples
Question Decomposition examples teach the model to break down complex questions into a sequence ofatomic,
retrieval-ready sub-questions. The rationale is that many visual knowledge queries involve implicit multi-hop reasoning
(e.g., identify the entity in the image, then query its properties). To capture this, the examples label questions as
either decomposable or non-decomposable, and provide the exact sub-questions that should be produced.
Decomposable cases typically involve: (1) entity grounding followed by factual lookup (e.g., identify the king of Spain
→find when he became king); (2) place or object recognition followed by knowledge retrieval (e.g., identify the arena
→retrieve capacity); or (3) multi-hop knowledge chains (e.g., identify the farm→locate it).
Non-decomposable examples demonstrate when a single visual or commonsense step suffices (e.g., “Translate this”,
“What kind of sculpture is this?”). This contrastive supervision helps the model learn when multi-hop decomposition is
beneficial and when it is unnecessary.
D.3 Response Generation ICL Examples
Response Generation samples demonstrate how the model should integrate external knowledge into coherent, grounded
answers by interleaving <search> tokens with natural language output. Each example provides: (1) the question, (2)
3

Table 11 Question Selection Examples.We show representative questions, whether retrieval was needed or not, and the
associated image filenames. TheImage Filepaths were truncated to have only the prefix of the image path for brevity.
Image File Question Retrieval
cragmm/4ec6f8ae.png How many hybrid variations of this car were there in 2024? Yes
cragmm/08629717.png Is that drink good for my gut health? Yes
cragmm/fb2fed47.png How many arms does this statue typically have? Yes
cragmm/1c613a06.png Translate this. No
cragmm/569a3617.png Which station has more tracks, this one or Penn Station? Yes
cragmm/a97e2470.png Where was the designer who developed this car originally from? Yes
cragmm/4f81b083.png What is the seating capacity of the car with the open trunk? Yes
cragmm/b797333f.png What does the word “skrzela” translate to in English? No
okvqa/3575845.png What is the couch made of? No
cragmm/d253fc27.png In what year did the president for whom this bridge is named win the
Battle of Trenton?Yes
okvqa/4597935.png Where is the farm depicted on the sign located? Yes
okvqa/3742825.png What brand of car is this? Yes
okvqa/3182455.png The fabric on that couch was very popular in the eighties — what was
it called?Yes
okvqa/1981195.png Why might the man be kicking up sand? No
okvqa/1217825.png What holiday might they be celebrating? Yes
okvqa/3778685.png With what religious tradition is the creature portrayed here associated? Yes
the model’s expected search-interleaved reasoning trajectory with properly placed <search> and</search> markers,
and (3) the corresponding ground-truth answer.
The central rationale is to teach the model not only to ask for external knowledge, but to do so at the correct semantic
point within the reasoning process. For instance, the model must first identify “King Felipe VI” before issuing a second
retrieval about his coronation date.
These examples also illustrate how retrieved facts are integrated back into the narrative, enabling PixaR to produce
faithful, factual answers without hallucinating details or over-triggering retrieval. The contrast between the final
grounded answer and the intermediate reasoning highlights how to combine multiple <search> calls into a single,
well-structured response.
D.4 Query-Type Assignment ICL Examples
To teach the modelhowto choose the correct retrieval modality, Query-Type Assignment examples pair each search-
interleaved response with: (1) the modality chosen for each <search> call (TEXT, IMAGE, or REGION), and (2) the
exact query string or region reference used.
The rationale behind these examples is rooted in the observation that different question types require different forms
of evidence:
•REGIONqueries are necessary when a specific entity must be identified or disambiguated (e.g., car brand, plant
type, or farm name).
•IMAGEqueries are appropriate when holistic scene recognition is required (e.g., identifying Madison Square
Garden from a full stadium view).
•TEXTqueries are used for abstract facts requiring no visual input (e.g., “When did Washington win the Battle of
Trenton?”).
By aligning each retrieval call with its intended modality, these examples teach the model to properly route retrieval
operations and to generate the correct query based on the context of the ongoing reasoning process. This ensures that
PixaR issues retrieval in a controlled, modality-aware manner that improves factual precision and reduces retrieval
noise.
4

Table 12 Question Decomposition Examples.Each example lists the original question, whether it was decomposed, and
the resulting sub-questions.
Image File Original Question Sub-Questions
cragmm/4dcc84dc.png When did the king of that country become king? 1) Who is the king of Spain?
2) When did the king of Spain be-
come king?
cragmm/da33192e.png What’s the capacity of this arena? 1) What is this place?
2)What’sthecapacityofthisarena?
cragmm/f73ab93c.png Where was the first sign accompanying this erected? 1) Where was the first pedestrian
crossing signal erected?
cragmm/878088c7.png What’s the ideal temperature for this plant? 1) What is this plant?
2) What is the ideal temperature for
this plant?
cragmm/b03b7dd6.png How long can I use it without turning it off? 1) What is the model name of this
generator?
2)HowlongcanIusethisgenerator?
cragmm/ec87776d.png Translate this into English. 1) Translate this into English.
cragmm/c0a60302.png What kind of sculpture is this? 1) What kind of sculpture is this?
cragmm/d253fc27.png In what year did the president for whom this bridge
is named win the Battle of Trenton?1) What is the name of the bridge?
2) What year did the president win
the Battle of Trenton?
okvqa/804725.png What grade is the child in? 1) What grade is the child in?
okvqa/3575845.png What is the couch made of? 1) What is the model name of this
couch?
okvqa/4830705.png Which historical group wore that clothing accessory? 1)Whichhistoricalgroupworespurs
as a clothing accessory?
okvqa/4597935.png Where is the farm depicted on the sign located? 1) What is the farm name?
2) Where is the farm located?
okvqa/5246795.png What is the purpose of the purple vehicle depicted? 1) What is the purpose of the purple
vehicle depicted?
okvqa/3742825.png What brand of car is this? 1) What brand of car is this?
infoseek/04308592.JPEG What country does this drink belong to? 1) Where was Louis Jadot made in?
infoseek/04123114.JPEG What is the basionym of this plant? 1) What is this plant?
2) What is the basionym of this
plant?
5

Table 13 Response Generation Examples.Each example includes the question, the model-generated answer (which
includes<search>calls), and the ground-truth answer.
Image File Question + Model Answer Ground-Truth Answer
cragmm/4dcc84dc.png Q: When did the king of that country become king?
A: The paper says “... independence from Spain.”
Since Mexico has no king, this refers to Spain. The
king of Spain <search> is King Felipe VI. Felipe VI
came to the throne<search>on June 19, 2014.June 19, 2014
cragmm/da33192e.png Q: What’s the capacity of this arena?
A: The place shown <search> is Madison Square Gar-
den. It can house <search> between 18,006 and 20,789
people.18,006–20,789 depending on event
cragmm/f73ab93c.png Q:Wherewasthefirstsignaccompanyingthiserected?
A: The first pedestrian crossing signal was erected
in<search> Bridge Street, Westminster, London, in
December 1868.Bridge Street, Westminster, London (1868)
cragmm/878088c7.png Q: What’s the ideal temperature for this plant?
A: The plant <search> appears to be Ravenea rivularis.
Ravenea rivularis prefers<search>65–85°F.Prefers 65–85°F
cragmm/b03b7dd6.png Q: How long can I use it without turning it off?
A: The generator appears to be <search> a Firman Tri
FuelGenerator. Itsusagetimeis <search> amaximum
of 12 hours with a 2–3 hour recommended break.12 hours, with 2–3 hour breaks
cragmm/ec87776d.png Q: Translate this into English.
A: It translates to: “What mixer is used in a Paloma?”“What mixer is used in a Paloma?”
cragmm/c0a60302.png Q: What kind of sculpture is this?
A: This is a snowman with red hat and carrot nose.Snowman balloon figure with red hat and carrot nose
cragmm/d253fc27.png Q: In what year did the president for whom this bridge
is named win the Battle of Trenton?
A: The bridge <search> is the George Washington
Bridge. George Washington won the Battle of Trenton
<search>on December 26, 1776.1776
okvqa/804725.png Q: What grade is the child in?
A: The child appears around 5 years old, which corre-
sponds to kindergarten.Kindergarten
okvqa/3575845.png Q: What is the couch made of?
A: The couch <search> appears to be a sofa made of
fabric, cotton, wood, foam, and leather.Fabric / cotton / wood / foam
okvqa/4830705.png Q: Which historical group wore that clothing acces-
sory?
A: Spurs were worn by<search>cowboys.Cowboys
okvqa/4597935.png Q: Where is the farm depicted on the sign located?
A: The farm <search> is Victoria Garden Centre Farm,
located<search>in the Midwest US.Midwest US
okvqa/5246795.png Q: What is the purpose of the purple vehicle depicted?
A: It is a tourist bus used to transport people.Tourist transport / mass transit
okvqa/3742825.png Q: What brand of car is this?
A: The brand<search>is Volkswagen.Volkswagen
infoseek/04308592.JPEG Q: What country does this drink belong to?
A: The drink is Louis Jadot, which was made in
<search>France.France
infoseek/04123114.JPEG Q: What is the basionym of this plant?
A: The plant <search> appears to be a type of tatsoi
mustard. Its basionym is <search> Cupressus lawso-
niana.Brassica rapa / Brassica rapa var. rosularis
6

Table 14 Query-Type Assignment Examples.Each example includes the original question, the model’s <search> -
interleaved response, and the corresponding query types and queries.
Image File Question +<search>-Interleaved Response Query Types Queries
cragmm/4dcc84dc.png Q: When did the king of that country become king?
A: The king of Spain <search> is King Felipe VI. </search> He came
to the throne<search>on June 19, 2014.</search>TEXT, TEXT 1) Who is the king of
Spain?
2) When did King Felipe
VI become king?
cragmm/da33192e.png Q: What’s the capacity of this arena?
A: The place shown <search> is Madison Square Garden.
</search> It can house <search> between 18,006 and 20,789 peo-
ple.</search>IMAGE, TEXT 1) image
2) What is the capacity of
this arena?
cragmm/f73ab93c.png Q: Where was the first sign accompanying this erected?
A: The first pedestrian crossing signal was erected <search> in
Bridge Street, Westminster, London.</search>TEXT Where was the first
pedestrian crossing signal
erected?
cragmm/878088c7.png Q: What’s the ideal temperature for this plant?
A: The plant <search> appears to be Ravenea rivularis. </search> It
prefers<search>65–85°F.</search>REGION, TEXT 1) plant
2) Ideal temperature for
Ravenea rivularis
cragmm/b03b7dd6.png Q: How long can I use it without turning it off?
A: The generator <search> appears to be a Firman Tri Fuel Gener-
ator. </search> Its usage time is <search> a maximum of 12 hours
with breaks.</search>REGION, TEXT 1) generator
2) Use time for Firman Tri
Fuel Generator
cragmm/ec87776d.png Q: Translate this into English.
A: The sentence translates to: “What mixer is used in a Paloma?”— —
cragmm/c0a60302.png Q: What kind of sculpture is this?
A: This is a snowman with red hat and carrot nose.— —
cragmm/d253fc27.png Q: In what year did the president for whom this bridge is named
win the Battle of Trenton?
A: The bridge <search> is the George Washington Bridge.
</search> Washington won the battle <search> on December 26,
1776.</search>REGION, TEXT 1) bridge
2) Battle of Trenton date
okvqa/804725.png Q: What grade is the child in?
A: The child appears around 5 years old, corresponding to kinder-
garten.— —
okvqa/3575845.png Q: What is the couch made of?
A: The couch <search> appears to be a sofa made of fabric, cotton,
wood, foam, and leather.</search>REGION couch
okvqa/4830705.png Q: Which historical group wore that clothing accessory?
A: Spurs were commonly worn by<search>cowboys.</search>TEXT Whathistoricalgroupused
to wear spurs?
okvqa/4597935.png Q: Where is the farm depicted on the sign located?
A: The farm <search> is Victoria Garden Centre Farm. </search> It
is located<search>in the Midwest US.</search>IMAGE, TEXT 1) image
2) Where is Victoria Gar-
den Centre Farm located?
okvqa/5246795.png Q: What is the purpose of the purple vehicle depicted?
A: It is a tourist bus used to transport people for sightseeing.— —
okvqa/3742825.png Q: What brand of car is this?
A: The car<search>is Volkswagen.</search>REGION car
infoseek/04308592.JPEG Q: What country does this drink belong to?
A: The drink is Louis Jadot, made in<search>France.</search>TEXT Where was Louis Jadot
made?
infoseek/04123114.JPEG Q: What is the basionym of this plant?
A: The plant<search>appears to be tatsoi mustard.</search>Its
basionym is<search>Cupressus lawsoniana.</search>REGION, TEXT 1) plant
2) Basionym of tatsoi mus-
tard
7
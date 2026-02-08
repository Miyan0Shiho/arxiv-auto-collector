# Nemotron ColEmbed V2: Top-Performing Late Interaction embedding models for Visual Document Retrieval

**Authors**: Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu, Oliver Holworthy, Benedikt Schifferer, Zhiding Yu, Yauhen Babakhin, Radek Osmulski, Jiarui Cai, Ryan Chesler, Bo Liu, Even Oldridge

**Published**: 2026-02-03 20:26:44

**PDF URL**: [https://arxiv.org/pdf/2602.03992v1](https://arxiv.org/pdf/2602.03992v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems have been popular for generative applications, powering language models by injecting external knowledge. Companies have been trying to leverage their large catalog of documents (e.g. PDFs, presentation slides) in such RAG pipelines, whose first step is the retrieval component. Dense retrieval has been a popular approach, where embedding models are used to generate a dense representation of the user query that is closer to relevant content embeddings. More recently, VLM-based embedding models have become popular for visual document retrieval, as they preserve visual information and simplify the indexing pipeline compared to OCR text extraction.
  Motivated by the growing demand for visual document retrieval, we introduce Nemotron ColEmbed V2, a family of models that achieve state-of-the-art performance on the ViDoRe benchmarks. We release three variants - with 3B, 4B, and 8B parameters - based on pre-trained VLMs: NVIDIA Eagle 2 with Llama 3.2 3B backbone, Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct, respectively. The 8B model ranks first on the ViDoRe V3 leaderboard as of February 03, 2026, achieving an average NDCG@10 of 63.42.
  We describe the main techniques used across data processing, training, and post-training - such as cluster-based sampling, hard-negative mining, bidirectional attention, late interaction, and model merging - that helped us build our top-performing models. We also discuss compute and storage engineering challenges posed by the late interaction mechanism and present experiments on how to balance accuracy and storage with lower dimension embeddings.

## Full Text


<!-- PDF content starts -->

Nemotron ColEmbed V2: Top-Performing Late Interaction
embedding models for Visual Document Retrieval
Gabriel de Souza P. Moreira1,†, Ronay Ak1,†, Mengyao Xu1, Oliver Holworthy1,
Benedikt Schifferer*, Zhiding Yu1, Yauhen Babakhin1, Radek Osmulski1, Jiarui Cai1,
Ryan Chesler1, Bo Liu1and Even Oldridge1
1NVIDIA
Abstract
Retrieval-Augmented Generation (RAG) systems have been popular for generative applications, powering language
models by injecting external knowledge. Companies have been trying to leverage their large catalog of documents
(e.g. PDFs, presentation slides) in such RAG pipelines, whose first step is the retrieval component. Dense retrieval
has been a popular approach, where embedding models are used to generate a dense representation of the user
query that is closer to relevant content embeddings. More recently, VLM-based embedding models have become
popular for visual document retrieval, as they preserve visual information and simplify the indexing pipeline
compared to OCR text extraction.
Motivated by the growing demand for visual document retrieval, we introduce Nemotron ColEmbed V2,
a family of models that achieve state-of-the-art performance on the ViDoRe benchmarks. We release three
variants—with 3B, 4B, and 8B parameters—based on pre-trained VLMs: NVIDIA Eagle 2 with Llama 3.2 3B
backbone, Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct, respectively. The 8B model ranks first on the
ViDoRe V3 leaderboard as of February 03, 2026, achieving an average NDCG@10 of 63.42.
We describe the main techniques used across data processing, training, and post-training—such as cluster-
based sampling, hard-negative mining, bidirectional attention, late interaction, and model merging—that helped
us build our top-performing models. We also discuss compute and storage engineering challenges posed by
the late interaction mechanism and present experiments on how to balance accuracy and storage with lower
dimension embeddings.
Keywords
Visual Document Retrieval, Late interaction, Visual Language Model, RAG, Dense Retrieval, ViDoRe
1. Introduction
Retrieval-Augmented Generation (RAG) has become a widely adopted paradigm for enhancing language
models generation with external knowledge, enabling them to retrieve and reason on relevant content
from large-scale corpora. Numerous high-performing text retrieval models including NV-Embed [ 1],
NV-Retriever [ 2], Qwen3-Embedding [ 3], and e5-mistral [ 4] have been proposed, and evaluated on
benchmarks such as MTEB [ 5,6] text benchmarks, that assume clean and well-formatted textual inputs.
In contrast, real-world use cases typically involve documents stored in formats like PDFs, PowerPoint
slides, or Word documents, requiring preprocessing pipelines to extract textual content (e.g. text
parsing, OCR). This process often results in the loss of critical visual information for modalities like
tables, charts, and infographics. To address those limitations, Visual Document Retrieval (VDR) [ 7] has
been proposed to retrieve document pages directly from their image, with no need for text extraction,
further preserving visual information and simplifying the indexing and search pipelines from document
retrieval systems. If page text is easily available, both page image and text can be used for better page
multimodal representation and retrieval.
Recent Vision-Language models (VLMs) aim to bridge the gap between text and image understanding
by learning joint representations across modalities. Models such as Qwen-VL [ 8], LLaMA-3.1-Nemotron-
Nano-VL [ 9], PaliGemma 2 [ 10], NVIDIA’s Eagle 2 [ 11,12] and Nemotron Nano V2 VL [ 13] have
*Work done while working at Nvidia.
†These authors contributed equally.
/envel⌢pe-⌢pengmoreira@nvidia.com (G. d. S. P. Moreira); ronaya@nvidia.com (R. Ak)
©2022 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).

demonstrated strong performance across a range of vision-language tasks. VLMs use image encoders
like CLIP [ 14], SigLIP [ 15] and C-RADIO [ 16] to extract image features and project them to the LLM
space. VLM decoder models have been adapted as contrastive embedding models for visual document
retrieval, like Jina CLIP [17] and Nomic Embed Vision [18].
ColBERT [ 19] demonstrated that a multi-vector late interaction approach could boost retrieval
accuracy, for allowing deeper interaction between query and textual context tokens, compared to the
pooling approaches (e.g., average, last) that compress the representation to a single-vector embedding.
ColPali [ 7] VLM embedding model leveraged late interaction between text-image tokens for improved
retrieval accuracy.
In order to evaluate visual document retrieval models, several benchmarks were introduced. The
most popular ones belong to the ViDoRe, which was released in three versions: V1 [ 7], V2 [ 20] and
V3 [21]. The latest Vidore V3 [ 21] is an important expansion for evaluating VDR on complex real-world
scenarios, including multi-type and multi-language queries across ten professional domains.
In this paper, we introduce the Nemotron ColEmbed V2, a family of state-of-the-art embedding
models for visual document retrieval. Our best-performing modelnemotron-colembed-vl-8b-v2achieves
an NDCG@10 of 63.42 on the Vidore V3 benchmark (+3% to the second place), ranking first on that
benchmark as of Feb. 03, 2026.
We initialized our 3B model from NVIDIA’s Eagle 2 vision-language model [ 11,12] with Llama 3.2 3B
LLM backbone and initialize our 4B and 8B models from Qwen3-VL models [ 22]. We also replaced the
original causal attention with bidirectional attention, and fine-tuned the models through contrastive
training with late interaction mechanism on curated datasets for visual document retrieval. Our training
datasets contain both text-only and text-image examples for contrastive learning, and we apply hard
negative mining following the methods proposed in NV-Retriever [ 2] to improve retrieval accuracy.
Finally, we use model merging to provide ensemble-level accuracy in a single model.
The main contributions of this paper are:
•We release three state-of-the-art models for visual document retrieval:llama-nemotron-colembed-
vl-3b-v21,nemotron-colembed-vl-4b-v22, andnemotron-colembed-vl-8b-v23. The 8B model achieves
top-1 performance in the MTEB ViDoRe V3 leaderboard, while the 4B and 3B models are among
top-6, outperforming the models of the same size in the leaderboard on NDCG@10.
•We describe the techniques that helped boost our model accuracy to the top of ViDoRe leader-
boards regarding data preprocessing (clustering-based sampling, hard-negative mining, cross-
lingual translation), two-stage training, late interaction, and model merging.
•Finally, we discuss some performance trade-offs of using late interaction mechanism, as its higher
accuracy comes at the price of increasing indexing embeddings storage and higher compute at
serving compared to simpler vector pooling. We provide an ablation on the reduced embedding
sizes for the late interaction.
2. Background
2.1. Visual Document Retrieval
Visual document retrieval has transitioned from rigid pipelines to integrated multimodal frameworks.
Early systems [ 23,24] relied on OCR-centric pipelines, where text was first extracted and then processed
by layout-aware encoders that fused content with 2D positions. Although effective for document
understanding, these early designs were computationally prohibitive for first-stage retrieval and were
largely restricted to reranking small candidate sets. In parallel, early CLIP-style bi-encoders [ 25]
offered global multimodal representations, but frequently struggled with cluttered pages where relevant
information was sparse or layout-dependent.
1https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v2
2https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2
3https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2

The rapid advancement of Large VLMs has blurred the boundaries between OCR, layout detection,
and visual understanding. Modern VLMs, trained on web-scale data, implicitly recognize rendered
text and layout structures directly from pixels, eliminating the need for explicit OCR or handcrafted
features. This shift has catalyzed a new generation of visual document retrieval models that combine
VLM backbones with ColBERT-style late interaction [ 19]. Notable implementations include ColPali
[7] (PaliGemma-based [ 26]), ColQwen [ 7] (Qwen2-based [ 27]), Jina Embedding Models [ 28] (Qwen2.5-
based [ 29]), and NVIDIA’s Nemoretriever Colembed [ 30] (Eagle2-based [ 11]). These models project
document images into a sequence of dense patch-level embeddings, enabling fine-grained, multi-vector
matching against text queries. Additionally, current research in this space also actively exploring various
optimization strategies to refine performance and efficiency. These include knowledge distillation from
high-capacity teachers [ 31], quantization for reduced memory footprints [ 31], and model merging
[3, 32, 33, 34] to ensemble the weights of diverse pre-trained models.
Our work builds upon this emerging paradigm of VLM-based late interaction. We leverage pretrained
VLM models as backbones for representation extraction and employ a late interaction mechanism
specifically tailored for visually rich documents.
2.2. Dense Retrieval
Dense retrieval methods for both text and visual documents are often categorized by how and when the
query and document interact. Existing approaches follow three primary paradigms: (1)bi-encoders:
These models independently encode queries and documents into single global vectors. Relevance is
determined by a simple similarity function, usually cosine similarity, as shown in Figure 1a. While
highly efficient and common in early text-only retrieval [ 35] or CLIP/SigLIP-style VLMs [ 14,15],
the single-vector representation bottleneck is often limited to capture fine-grained lexical, visual, or
layout-dependent cues. (2)Cross-encoders: These architectures jointly encode queries and candidate
documents, allowing for dense early interaction via self- and cross-attention over all tokens or patches.
They can model intricate token–level alignments and thus excel as strong rerankers. However, it does
not support pre-computing/indexing document embeddings, and the quadratic computational cost
renders them impractical for first-stage retrievers across large-scale collections. (3)Late interaction
models: Bridging the gap between the two, late interaction retains some expressiveness from cross-
encoders, but still allows pre-computing documents’ multi-vector embeddings, as further discussed in
the next section.
2.3. Late Interaction
The late interaction mechanism introduced by ColBERT [ 19] enables fine-grained interactions between
query and document tokens. As shown in Figure 1b, for a query, each token embedding interacts with
all document token embeddings using aMaxSimoperator, which selects the maximum similarity per
query token and sums these scores to produce the final relevance score. This requires storing all token
embeddings of the document corpus (text or images). At inference time, query token embeddings
are computed and interact with the stored document embeddings throughMaxSimop. We adopt this
mechanism in our models to enable fine-grained retrieval. While this approach offers the expressiveness
of token-level matching, compared to simpler pooling methods such as average or last-token pooling, as
shown in Figure 1a, the late-interaction method introduces latency and storage overhead that may need
to be assessed, as they become a concern for real-world applications. To ensure scalability, we further
investigate dimensionality reduction techniques, refining the trade-off between dense representation
quality and storage overhead, as discussed in Section 5.
2.4. Contrastive Learning
Dense retrieval models are typically trained with contrastive learning to maximize the embedding
similarity between the query and positive passage, while minimizing the similarity between the query

LLM LLM
Query CorpusS
Pooling(a) Bi-encoder architecture with Pooling
LLM LLM
Query CorpusS
MaxSim MaxSim MaxSim (b) Late-interaction architecture
Figure 1:Illustration of the bi-encoder and late-interaction architectures.
and negative corpus. The InfoNCE contrastive loss [ 36] is a popular choice to train the model to
distinguish between positive and negative pairs in a shared embedding space,
ℒ(𝑞, 𝑑+, 𝐷𝑁) =−logexp(sim(𝑞, 𝑑+)/𝜏)∑︀
𝑑𝑖∈{𝑑+}∪𝐷𝑁exp(sim(𝑞, 𝑑 𝑖)/𝜏),(1)
where 𝑞is the embedding of a query, and 𝑑+are embeddings positive documents. 𝐷𝑁denotes the set
of negative passages. 𝜏is the temperature parameter. 𝑠𝑖𝑚(·) represents a similarity function like cosine
similarity, dot product or late interaction similarity based on theMaxSimoperation.
3. Nemotron ColEmbed V2: a Family of Multimodal Late Interaction
Models for Visual Document Retrieval
In this section, we describe the Nemotron ColEmbed V2 models’ architectures and the main methods
we used to those top performing models, listed in Table 1.
Table 1
The Nemotron ColEmbed V2 family
Model (Huggingface ID) # Parameters (B)1Embedding Dimension
nvidia/llama-nemotron-colembed-3b-v2 3.99 3072
nvidia/nemotron-colembed-vl-4b-v2 4.43 2560
nvidia/nemotron-colembed-vl-8b-v2 8.14 4096
3.1. llama-nemotron-colembed-vl-3b-v2 Architecture
Ourllama-nemotron-colembed-vl-3b-v2late-interaction VLM embedding model is based on NVIDIA
Eagle 2 vision-language model [ 11,12], extending its first version [ 30]. These models adopt dynamic
image tiling to support inputs of varying resolutions, and employ a carefully curated data strategy that
improves multimodal learning. These design choices enable Eagle 2 models to achieve state-of-the-art
results on several multimodal benchmarks, providing a solid foundation for retrieval tasks. We initialize
our model from an internal pre-trained Eagle 2 VLM that uses SigLip 2 [ 15] as the image encoder and
Llama 3.2 3B [37] as the LLM backbone.
1# of parameters not considering the embedding weights

Regarding the dynamic tiling mechanism, the max_input_tiles parameter is used to control the
number of tiles produced from each image. Each image tile generates 256 visual tokens. For training,
we set max_input_tiles = 2 (including an additional thumbnail tile from the full page) to maintain
memory efficiency, as increasing it to 4 did not yield performance gains. During inference, we set
max_input_tiles = 8to allow for finer visual granularity.
Figure 2 illustrates the dynamic image tiling, image encoding into visual tokens, and the late interac-
tion mechanism.
Black Myth: W ukong Is Loaded
With RTX T echnology . As part of  
our collaboration...
Dynamic Image T ilingVision EncoderEmbedding Layer
LLM LLMMaxSim(                ,                 )             
... ... ...MaxSim(                ,                 )             
MaxSim(                ,                 )             Which GPU is recommended for W ukong?+
+
Late-interaction Score
Figure 2:llama-nemotron-colembed-vl-3b-v2architecture with dynamic image tiling and late interaction scoring
mechanisms[30].
3.2. nemotron-colembed-vl-4/8b-v2 Architectures
Ournemotron-colembed-vl-4b-v2andnemotron-colembed-vl-8b-v2late-interaction embeddding models
are based on Qwen3-VL 4B and 8B VLMs[ 22], which support multimodal inputs and long context.
Similarly to Eagle 2 used byllama-nemotron-colembed-vl-3b-v2, Qwen3-VL adopts a three-module
architecture, comprising SigLIP-2 vision encoder[ 15], a two-layer MLP-based vision–language merger,
and Qwen3 LLMs [38].
Their vision encoder is designed to handle dynamic, native-resolution images, mapping them to
visual token sequences of variable length. To enhance perceptual capability and preserve rich visual
details, Qwen3-VL extends the DeepStack mechanism by injecting visual tokens from intermediate
layers of the vision encoder into multiple layers of the LLM [22].
3.3. Key Methods for Better Performance
3.3.1. Modifying LLM decoder causal attention to bi-direction attention for encoders
LLMs and VLMs are decoder models and use causal (uni-directional) attention. It means during training,
when predicting a token, the model is prevented to access the following tokens (on the right) to avoid
leaking information that would not be available during inference.
When adapting decoder LLMs as embedding models (encoders), a common practice involves transi-
tioning the uni-directional attention to bi-directional attention. This modification enables Transformer
layers to attend to the full context of a sequence, allowing each token to integrate information from
both preceding and succeeding tokens. Such global representation has been shown to significantly
enhance retrieval accuracy for LLM-based embedding models [ 39,2,40]. Consistent with these find-
ings, we implement this technique in Nemotron ColEmbed V2 models, where we observed substantial
performance improvement when adapting Eagle 2 and Qwen3-VL architectures.
3.3.2. Hard-Negative Mining
Embedding models for retrieval are primarily trained with contrastive learning, a paradigm requiring
triplets composed of a query, positive examples, and negative examples. Extensive research on informa-

tion retrieval indicates the efficiency of contrastive learning hinges on the availability of hard-negatives,
i.e., false examples that exhibit high semantic or lexical similarity to the query. The hard-negatives can
be mined from the corpus using external sparse or dense "teacher" retrieval models.
For Nemotron ColEmbed V2 models, we used an internal Llama-Eagle 3B VLM embedding model for
mining from the corpus the top-k most similar page images to the queries.
We leverage thetop-k with percentage to positive thresholdmethod from NV-Retriever [ 2]. It filters
the potential hard-negatives by limiting their maximum similarity scores to a percentage of the positive
sample similarity score, creating a margin that reduces the number of false negatives (e.g. the ones
that should be actually positives). We set the threshold as 0.95, meaning we select the 𝐾most relevant
negative samples whose similarity to the query is less than 95% of the query–positive similarity score.
This encourages the model to learn from challenging negatives, while removing potential false
negatives that have high similarity scores.
3.3.3. Cluster-based Data Sampling
Public training datasets typically exhibit imbalance, with disproportionate numbers of samples across
different domains. Training on such skewed data could lead to overfitting to specific domains, which
compromises the model’s ability to generalize to others, especially the underrepresented ones.
To mitigate this, Nemotron-CLIMB[ 41] proposed a framework for optimizing the blend of LLM
training data by partitioning the the corpus into distinct clusters and sampling a curated percentage of
training examples from each.
For Nemotron ColEmbed V2 models, we adapted their method, clustering the positive contexts, then
sampling some positive samples from each cluster together with the associated queries and negatives.
For clustering, we generate embeddings from document page images using an internal Llama-Eagle
VLM embedding model, resulting in 3072-dim vectors. We then apply PCA to reduce embeddings
dimension to 50, followed by a K-Means clustering approach utilizing gap statistics [ 42] to choose the
most representativekclusters. To ensure diversity and balance the domains of our training blend, we
perform uniform sampling the 14 discovered clusters.
3.3.4. Cross-lingual Translation
There has been growing interest from the community in multi-lingual and cross-lingual retrieval, the
latter for cases where the query and corpus languages are different. This is particularly challenging
for visual document retrieval, as document pages are represented as images to the model. Vidore
V3 exemplifies this interest, as the corpus of PDF documents is in English or French and queries are
translated into six languages.
To enable Nemotron ColEmbed V2 models supporting cross-lingual retrieval, we have augmented
our training data by usingQwen3-235B-A22model to translate sampled queries from each discovered
cluster into other languages.
3.3.5. Two-stage Training
Thellama-nemotron-colembed-vl-3b-v2model is trained in two stages [ 30]. In the first stage, the model
is trained on a textual corpus consisting of query-positives-negatives triplets. This stage is designed to
establish a robust foundation for semantic similarity within the textual embedding space.
In the second stage, we fine-tune the model on an image-retrieval corpus. This stage facilitates
cross-modal alignment by grounding visual features in the textual representation space.
Fornemotron-colembed-vl-4b-v2andnemotron-colembed-vl-8b-v2, due to Qwen3-VL’s strong cross-
modal pre-training, we perform a single-stage contrastive learning training with image corpus.

3.3.6. Model Merging
Model merging or model souping is a technique that combines weights of multiple models, typically
those sharing the same architecture but trained with different data or hyperparameters [ 43,44]. It
is observed that the efficacy of this approach increases with the diversity of the constituent models’
weights. Recently, this approach has been popularized for improving the robustness and generalization
of embedding models, such as Qwen3-Embedding [ 3], Gemini Embedding [ 32], EmbeddingGemma [ 33],
and Llama-Embed-Nemotron-8B [34].
For Nemotron ColEmbed V2 models, we employed a simple weighted average of model weights for
merging. The 3B model is an ensemble of 8 individual models, and the 4B and 8B are merged from 4
individual models each. The individual models are trained with variations in the training blend and in
some hyperparameters.
By comparing the ensembled model with the best individual model used in the ensemble, we have
noticed that the accuracy gains might scale with the model size. We observed an improvement of
0.8% forllama-nemotron-colembed-vl-3b-v2, 1.0% for thenemotron-colembed-vl-4b-v2, and 1.5% for the
nemotron-colembed-vl-8b-v2model.
4. Results
In this section we demonstrate the effectiveness of Nemotron ColEmbed V2 models on different visual
document retrieval benchmarks, withnemotron-colembed-vl-8b-v2being the top-performer in all of
them.
4.1. Vidore V3 Leaderboard
The Vidore V3 [ 21] is a recent benchmark that emphasizes evaluation of vision document retrieval for
complex enterprise and real-world scenarios, including multi-type and multi-language queries across
ten professional domains.
To ensure submitted models do not overfit to the leaderboard, Vidore V3 has created two sets of
tasks/datasets: eight public and two private, for which test datasets were not released publicly. The
MTEB maintainers have established a process in which they run themselves the submitted models
evaluation on the private tasks and report in the leaderboard.
Table 2 presents the retrieval accuracy (NDCG@10) of the top models in MTEB Vidore V3 leaderboard,
as of Feb. 03, 2026. We can see that ournemotron-colembed-vl-8b-v2model places 1st in the leaderboard,
with NDCG@10 of 63.42, with +3% improvement to the second place. The leaderboard also presents
results fornemotron-colembed-vl-4b-v2andllama-nemotron-colembed-vl-3b-v2, which demonstrate the
highest Avg. NDCG@10 over the other models with the same size.
Table 2
Vidore V3 leaderboard as of Feb. 03, 2026. Our Nemotron ColEmbed V2 models are highlighted in gray. Retrieval
accuracy scores are NDCG@10.
Public tasks Private tasks
Rank Model Avg CompSci Energy FinanceEn FinanceFr HR Industrial Pharma Physics Nuclear Telecom
1 nemotron-colembed-vl-8b-v2 63.42 79.30 69.82 67.29 51.54 66.32 56.03 67.19 50.84 53.84 72.00
2 tomoro-colqwen3-embed-8b61.59 75.35 68.41 65.08 49.10 63.98 54.41 66.36 50.13 52.65 70.46
3 nemotron-colembed-vl-4b-v2 61.54 78.56 67.48 65.02 49.01 62.39 53.91 66.10 48.86 52.78 71.30
4 Ops-Colqwen3-4B61.17 77.74 66.49 65.71 48.81 61.81 53.99 66.42 49.14 52.23 69.33
5 tomoro-colqwen3-embed-4b60.20 75.44 66.43 63.84 46.83 60.09 53.58 65.74 49.32 51.23 69.44
6 llama-nemotron-colembed-vl-3b-v2 59.79 77.09 64.88 64.23 44.41 62.28 51.71 66.04 46.93 50.65 69.68
7 jina-embeddings-v457.52 71.81 63.50 59.30 46.10 59.53 50.38 63.09 46.63 50.02 64.81
8 colnomic-embed-multimodal-7b57.33 76.20 63.58 56.57 45.46 58.67 50.13 62.26 48.25 45.02 67.16
9 llama-nemoretriever-colembed-3b-v157.26 75.16 62.07 60.88 43.77 58.69 47.09 63.74 45.13 49.15 64.74
4.2. Vidore V1&V2 Leaderboard
Table 3 presents results for the MTEB Vidore V1&V2 leaderboard as of Feb. 03, 2026.

The leaderboard integrates the two benchmarks because Vidore V1 provided public in-domain training
data, and many models overfit to it. For Vidore V2, no in-domain training data was provided, as well
for Vidore V3.
Ournemotron-colembed-vl-8b-v2model places second, close to the leading model. Thellama-nemotron-
colembed-vl-3b-v2andnemotron-colembed-vl-4b-v2are also among the top-4.
Table 3
Vidore V1&V2 leaderboard as of Feb. 03, 2026. Our Nemotron ColEmbed V2 models are highlighted in gray.
Retrieval accuracy scores are NDCG@5.
Rank Model Avg.Vidore V1 Vidore V2
ArxivQA DocVQA InfoVQA Shift Project AI Energy Gov. Reports Healthcare TabFQuad TAT-DQA MIT Biomed. ESG Restau. En ESG Restau. Multi Econ. Macro
1Ops-Colqwen3-4B84.87 91.78 66.45 94.02 90.84 99.63 97.26 98.02 99.63 93.55 82.38 65.53 78.61 66.05 64.45
2 nemotron-colembed-vl-8b-v2 84.80 93.08 68.05 94.56 93.30 100.00 97.89 98.89 99.63 97.74 83.37 66.16 73.15 60.56 60.76
3 nemotron-colembed-vl-4b-v2 83.87 92.03 67.39 93.31 92.26 99.26 96.19 98.02 98.52 98.05 81.19 64.32 71.43 61.48 60.75
4 llama-nemotron-colembed-vl-3b-v2 83.64 90.40 67.17 94.68 92.00 100.00 98.02 97.95 98.89 97.25 81.04 63.19 73.11 58.64 58.59
5tomoro-colqwen3-embed-8b83.52 91.15 66.37 94.48 87.89 99.26 96.71 97.58 99.06 94.23 80.92 65.47 75.98 60.71 59.46
6EvoQwen2.5-VL-Retriever-7B-v183.41 91.49 65.07 94.11 88.80 99.63 96.63 96.29 98.89 93.63 82.26 65.20 76.98 59.67 59.13
7tomoro-colqwen3-embed-4b83.18 90.58 66.30 94.31 87.39 99.26 96.91 97.17 99.63 94.33 79.87 65.38 74.65 62.44 56.30
8llama-nemoretriever-colembed-3b-v183.10 88.35 66.21 94.92 90.70 99.63 96.63 97.82 99.26 95.94 80.57 62.70 75.38 57.38 57.84
9SauerkrautLM-ColQwen3-8b-v0.182.91 93.80 64.69 94.51 90.41 98.65 96.52 96.79 99.26 92.18 84.04 63.26 70.77 57.85 57.98
10EvoQwen2.5-VL-Retriever-3B-v182.76 90.46 63.67 92.22 88.60 100.00 97.63 98.89 99.26 93.99 82.00 63.63 67.11 59.05 62.19
4.3. MIRACL-Vision benchmark
MIRACL-Vision is a large multi-lingual VDR benchmark [ 45], covering many popular and under-
resourced languages. It is based on MIRACL[ 46] text multilingual benchmark and the corpus is composed
by Wikipedia passages. MIRACL-VISION corpus, instead, is composed of screenshot images extracted
from Wikipedia pages.
Table 4 shows retrieval accuracy (NDCG@10) on MIRACL-Vision multilingual benchmark. It can
be observed how models’ accuracy on visual document retrieval vary for popular vs. under-resourced
languages (e.g. Telugu, Yoruba).
Nemotron ColEmbed V2 models perform better among the analyzed models, because of the pre-
training of the backbone LLMs and also our augmented training blend containing cross-lingual examples.
Thenemotron-colembed-vl-8b-v2provides the highest score for most of the languages.
Table 4
MIRACL-Vision results (NDCG@10) on multi-lingual visual document retrieval. Nemotron ColEmbed V2 models
are in the last three columns, results from other models obtained from MIRACL-Vision paper[45].
Language dse-
qwen2-
2b-mrl-v1gme-
Qwen2-
VL-2B-
Instructvdr-2b-
multi-
v1colqwen2-
v1.0llama-
nemoretriever-
colembed-
3b-v1nemoretriever-
colembed-
3b-v2nemotron-
colembed-
vl-4b-v2nemotron-
colembed-
vl-8b-v2
Arabic 0.3893 0.4888 0.4379 0.4129 0.4247 0.5250 0.60280.7863
Bengali 0.2352 0.3755 0.2473 0.2888 0.4878 0.5391 0.51560.6160
Chinese 0.5962 0.6314 0.5963 0.4926 0.4355 0.4878 0.66970.7204
English 0.6605 0.6784 0.6784 0.6417 0.7363 0.7397 0.72460.7480
Farsi 0.2250 0.3085 0.2398 0.2616 0.3109 0.3570 0.42660.5289
Finnish 0.4162 0.6863 0.5283 0.6604 0.8513 0.8541 0.83980.8726
French 0.7160 0.6851 0.7194 0.6876 0.7988 0.7943 0.79430.8171
German 0.6267 0.6345 0.6205 0.5995 0.6831 0.6924 0.71000.7233
Hindi 0.1740 0.3127 0.2058 0.2209 0.4867 0.5319 0.53380.5902
Indonesian 0.4866 0.5416 0.5254 0.5320 0.6428 0.6550 0.64800.6680
Japanese 0.6232 0.7305 0.6553 0.6970 0.7260 0.7493 0.83260.8690
Korean 0.4446 0.6202 0.4952 0.4419 0.5158 0.5394 0.61360.7316
Russian 0.6505 0.7202 0.6995 0.6811 0.7670 0.7920 0.78790.8399
Spanish 0.5927 0.6277 0.6274 0.6224 0.7109 0.72360.7033 0.7089
Swahili 0.4156 0.5348 0.4509 0.49310.7767 0.7495 0.6886 0.7422
Telugu 0.0274 0.0893 0.0318 0.0264 0.1669 0.23250.1579 0.1899
Thai 0.2692 0.3563 0.3177 0.2389 0.4035 0.4727 0.59280.6699
Yoruba 0.4178 0.4884 0.4577 0.5120 0.5888 0.59430.4469 0.5252
Average 0.4426 0.5283 0.4741 0.4728 0.5841 0.6127 0.6272 0.6860

5. Late-interaction Deployment Challenges
Leaderboards and benchmarks typically evaluate performance based on accuracy metrics. Some also
include proxy indicators of computational efficiency, such as model size or embedding dimensionality.
Ultimately, rankings are determined by accuracy, which may not reflect the broader needs of real-world
applications. No solution fits all use-cases. In this section, we discuss trade-offs of late interaction in
the context of production deployment.
5.1. Retrieval Systems Considerations
Deploying a production system involves balancing accuracy, latency/throughput, and cost. Typical
retrieval system requirements involve the following aspects:
•Model size:All embeddings of documents are generated by retrieval model. This step can be
performed in batches, with support for continuous updates as new documents arrive. Throughput
and cost are key considerations, and the overall retrieval performance is primarily influenced by
the size of the model.
•Storage:The embeddings storage requirements are primarily determined by the embedding
dimension, numeric precision and number of vectors per document.
•Serving:Latency measures how quickly documents can be retrieved in response to a user query.
Since queries are typically short (around 50–100 tokens), the size of the embedding model plays a
smaller role in this stage. Incorporating a reranker in the retrieval pipeline, such as a cross-encoder,
can improve accuracy, but at the cost of increasing the latency to serve another model.
We discuss the trade-offs between those aspects in the next section.
5.2. Retrieval Pipelines Trade-offs
The late-interaction paradigm [ 19] has demonstrated significant performance improvements in retrieval
tasks by preserving fine-grained token-level interactions between queries and documents. Unlike
traditional pooling strategies that compress entire sequences into single vectors, late-interaction models
leverage all token-level representations. However, this approach introduces a fundamental trade-off
between accuracy and storage cost, as each document requires multiple token embeddings, leading to
significantly increased storage requirements.
Table 5 summarizes these trade-offs aspects for different retrieval approaches, reporting requirements
in GigaBytes (GB) for storing the embeddings for one million page images. The storage footprint of late-
interaction models depends on three key factors: token/embedding count (sequence length), embedding
dimension, and numerical precision (e.g., float32, float16, int8). As can be observed in the table, late
interaction models require much more storage for indexing document multi-vector embeddings.
Table 5
Comparison of different VLM models in terms of model size, embedding dimension, storage requirements and
retrieval accuracy (Vidore V3 NDCG@10). Last line is a retrieval pipeline in which an embedding model retrieves
the top 50 page images, subsequently reranked by a cross-encoder.
Model Params (B)Embed.
dimAvg. tokens/
embeddings
per image# floating
points
per imageStorage for 1M
fp16 embed.(GB)Vidore V3
NDCG@10
nemotron-colembed-vl-8b-v2 8.14 4096 773 3166208 5897.5 63.54
nemotron-colembed-vl-4b-v2 4.43 2560 773 1978880 3686.0 61.42
llama-nemotron-colembed-vl-3b-v2 3.99 3072 2304 7077888 13183.6 59.70
llama-nemoretriever-colembed-1b-v1 2.15 2048 2304 4718592 8789.1 55.48
llama-nemotron-embed-vl-1b-v2 1.41 2048 1 2048 3.8 48.69
llama-nemotron-embed-vl-1b-v2 w/
llama-nemotron-rerank-vl-1b-v21.41 + 1.41 2048 1 2048 3.8 54.41

The number of tokens per image (sequence length) is determined by the VLM image tiling/resizing
logic and its image encoder. For example, Qwen-3 VL (backbone models from our 8B and 4B embedding
models) generates on average 773 visual embeddings for Vidore V3 pages, while Eagle 2 generates 2304
visual embeddings for Vidore V3 page images (with resolution of about 1654x2339 pixels).
Late interaction models require orders of magnitude more storage. For instance, thellama-
nemoretriever-colembed-1b-v1late interaction model and thellama-nemotron-embed-vl-1b-v2single-
vector model. Both use Llama 3.2 1B as LLM backbone. While the single-vector model requires 3.8 GB
for embedding storage, the late-interaction model requires 8,789.1 GB (2312x). If we add thellama-
nemotron-rerank-vl-1b-v2cross-encoder model to the retrieval pipeline, reranking the top-50 page
images retrieved byllama-nemotron-embed-vl-1b-v2, the NDCG@10 boosts from 48.69 to 54.40, which
is close to the 55.48 accuracy from the late interactionllama-nemoretriever-colembed-1b-v1for a small
fraction of storage requirements4.
Latency is another important challenge for Late-interaction models. During inference, it requires
calculation between a query and the multi-vectors of all page images in the corpus. Late interaction
requires specialized vector database support to the MaxSim operation, and might introduce latency
overhead. The alternative pipeline composed of a single-vector embedding followed by a cross-encoder
provides much lower latency, because the cross-encoder performs early-interaction between query and
document tokens only for the top-k documents retrieved by the embedding model, and for the whole
corpus as in the late interaction approach.
Retrieval pipeline design that should be carefully aligned with the specific use case. For example,
[47] results demonstrate that in scenarios where corpus is large and number of queries is moderate, a
smaller less accurate embedding model combined with a reranker (to improve accuracy) can be more
cost-efficient than a larger embedding model.
Ultimately, the choice between late-interaction and bi-encoder paradigms depends on specific use
case requirements and system constraints.
5.3. Ablation on Embedding-size Reduction
Several techniques can minimize storage requirements for both paradigms, like reducing embedding
dimensions. Linear projection layers can be used to downsize the embeddings output by the LLM back-
bone. Matryoshka Representation Learning [ 48] allows having a single model that outputs embeddings
that can be sliced/pruned to multiple smaller dimensions.
Following the approach used in vidore/colqwen2-v1.0 models [ 7], we applied a linear projection layer
to reduce the output dimension to 512 and 128. To minimize accuracy variation in the ablation, for each
architecture and dim size, we train four models with different seeds (that sample different portions of
our data blend for training), and report the average NDCG@10 across these four models.
We can observe the ablation results in Table 6. For thenemotron-colembed-vl-8b-v2, projecting
embeddings to 512-dim reduces storage requirements by 87.5%, while keeping 96.02% of retrieval
accuracy. With 128-dim embeddings, it requires only 3% of storage, keeping 95.36% of the accuracy. We
see a similar trend fornemotron-colembed-vl-4b-v2model. However, even with 128-dim, the storage
requirement of 184.3 GB for 1M pages may still be too high for production environments handling large
document corpora.
Decreasing embeddings numerical precision, i.e., to float16 or int8, is an alternative to reducing
storage footprint. Many vector databases already support storage and retrieval of lower-precision
embeddings, some of them offering post-training quantization for precision reduction.
Additionally, binary quantization reduces precision to 1-bit per element, potentially reducing storage
by 16x. However, our experience with bi-encoders indicates that binary quantization performs poorly
when the embedding dimensionality is too small, and these techniques require further testing with
late-interaction embedding size of 128. AnswerAI’s late-pooling approach [ 49] can reduce token vectors
4Although both models use Llama 3.2 1B as LLM backbone, this comparison requires some remarks as late interaction is not the
only factor we change here: (1) those models are trained on different training blends, (2)llama-nemoretriever-colembed-1b-v1
uses a larger image encoder (SigLIP2 1B) thanllama-nemotron-embed-vl-1b-v2(SigLIP2 400M).

Table 6
Ablation study on reducing the embedding sizes of late interaction models. For these models, we don’t apply
model merging; instead, we report the average NDCG@10 across four models trained with different seeds.
Model Params (B)Embed.
dimAvg. tokens/
embeddings
per image# floating
points
per imageStorage for 1M
fp16 embed.(GB)% storageVidore V3
NDCG@10% NDCG@10
nemotron-colembed-vl-8b-v2 8.144096 773 3166208 5897.5 100% 62.29 100.00%
512 773 395776 737.2 13% 59.81 96.02%
128 773 98944 184.3 3% 59.40 95.36%
nemotron-colembed-vl-4b-v2 4.432560 773 1978880 3686.0 100% 60.42 100.00%
512 773 395776 737.2 20% 59.29 98.13%
128 773 98944 184.3 5% 58.47 96.77%
by factors of 3-5, while MUVERA [ 50] proposes converting multi-vector embeddings into single Fixed
Dimensional Encodings (FDEs) whose inner product approximates multi-vector similarity, enabling the
use of standard single-vector retrieval with smaller total embedding size.
6. Conclusion
In this paper, we introduce the Nemotron ColEmbed V2 family of late-interaction models for visual
document retrieval. We demonstrate their top-performance on ViDoRe benchmarks and multi-lingual
capabilities on MIRACL-Vision benchmark. We describe the main methods that boosted the accuracy
of our late-interaction models, like changing VLMs backbones to use bi-directional attention, using
positive-aware hard-negative mining, cluster-based data sampling, cross-lingual translation, and model
merging. Finally, we discuss the deployment challenges of late-interactions models and highlight
key considerations for real-world deployment. We present numbers that illustrate the trade-offs
between accuracy and storage requirements and provide an ablation on reducing embedding sizes, thus
storage requirements. Our release of Nemotron ColEmbed V2 late-interaction models provides a strong
foundation for future research and practical applications in vision document retrieval.
References
[1]C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, W. Ping, Nv-embed: Improved
techniques for training llms as generalist embedding models, arXiv preprint arXiv:2405.17428
(2024).
[2]G. d. S. P. Moreira, R. Osmulski, M. Xu, R. Ak, B. Schifferer, E. Oldridge, Nv-retriever: Improving
text embedding models with effective hard-negative mining, arXiv preprint arXiv:2407.15831
(2024).
[3]Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou,
Qwen3 embedding: Advancing text embedding and reranking through foundation models, arXiv
preprint arXiv:2506.05176 (2025).
[4]L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, F. Wei, Improving text embeddings with large
language models, arXiv preprint arXiv:2401.00368 (2023).
[5]N. Muennighoff, N. Tazi, L. Magne, N. Reimers, Mteb: Massive text embedding benchmark, arXiv
preprint arXiv:2210.07316 (2022).
[6]I. Chung, I. Kerboua, M. Kardos, R. Solomatin, K. Enevoldsen, Maintaining mteb: Towards long
term usability and reproducibility of embedding benchmarks, arXiv preprint arXiv:2506.21182
(2025).
[7]M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, P. Colombo, Colpali: Efficient
document retrieval with vision language models, 2024. URL: https://arxiv.org/abs/2407.01449.
arXiv:2407.01449.
[8]P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al., Qwen2-vl:

Enhancing vision-language model’s perception of the world at any resolution, arXiv preprint
arXiv:2409.12191 (2024).
[9]A. Bercovich, I. Levy, I. Golan, M. Dabbah, R. El-Yaniv, O. Puny, I. Galil, Z. Moshe, T. Ronen,
N. Nabwani, et al., Llama-nemotron: Efficient reasoning models, arXiv preprint arXiv:2505.00949
(2025).
[10] A. Steiner, A. S. Pinto, M. Tschannen, D. Keysers, X. Wang, Y. Bitton, A. Gritsenko, M. Minderer,
A. Sherbondy, S. Long, et al., Paligemma 2: A family of versatile vlms for transfer, arXiv preprint
arXiv:2412.03555 (2024).
[11] Z. Li, G. Chen, S. Liu, S. Wang, V. VS, Y. Ji, S. Lan, H. Zhang, Y. Zhao, S. Radhakrishnan, et al.,
Eagle 2: Building post-training data strategies from scratch for frontier vision-language models,
arXiv preprint arXiv:2501.14818 (2025).
[12] G. Chen, Z. Li, S. Wang, J. Jiang, Y. Liu, L. Lu, D.-A. Huang, W. Byeon, M. Le, T. Rintamaki, et al.,
Eagle 2.5: Boosting long-context post-training for frontier vision-language models, arXiv preprint
arXiv:2504.15271 (2025).
[13] A. S. Deshmukh, K. Chumachenko, T. Rintamaki, M. Le, T. Poon, D. M. Taheri, I. Karmanov, G. Liu,
J. Seppanen, G. Chen, et al., Nvidia nemotron nano v2 vl, arXiv preprint arXiv:2511.03929 (2025).
[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin,
J. Clark, et al., Learning transferable visual models from natural language supervision, in:
International conference on machine learning, PmLR, 2021, pp. 8748–8763.
[15] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans,
L. Beyer, Y. Xia, B. Mustafa, et al., Siglip 2: Multilingual vision-language encoders with improved
semantic understanding, localization, and dense features, arXiv preprint arXiv:2502.14786 (2025).
[16] M. Ranzinger, G. Heinrich, J. Kautz, P. Molchanov, Am-radio: Agglomerative vision foundation
model reduce all domains into one, in: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2024, pp. 12490–12500.
[17] H. Xiao, G. Mastrapas, B. Wang, Jina clip: Your clip model is also your text retriever, in: Multi-modal
Foundation Model meets Embodied AI Workshop@ ICML2024, 2024.
[18] Z. Nussbaum, B. Duderstadt, A. Mulyar, Nomic embed vision: Expanding the latent space, arXiv
preprint arXiv:2406.18587 (2024).
[19] O. Khattab, M. Zaharia, Colbert: Efficient and effective passage search via contextualized late
interaction over bert, in: Proceedings of the 43rd International ACM SIGIR conference on research
and development in Information Retrieval, 2020, pp. 39–48.
[20] Q. Macé, A. Loison, M. Faysse, Vidore benchmark v2: Raising the bar for visual retrieval, arXiv
preprint arXiv:2505.17166 (2025).
[21] A. Loison, Q. Macé, A. Edy, V. Xing, T. Balough, G. Moreira, B. Liu, M. Faysse, C. Hudelot, G. Viaud,
Vidore v3: A comprehensive evaluation of retrieval augmented generation in complex real-world
scenarios, arXiv preprint arXiv:2601.08620 (2026).
[22] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo,
Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu,
C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun,
J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang,
M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou,
J. Zhou, Y. Zhu, K. Zhu, Qwen3-vl technical report, 2025. URL: https://arxiv.org/abs/2511.21631.
arXiv:2511.21631.
[23] Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, M. Zhou, Layoutlm: Pre-training of text and layout
for document image understanding, in: Proceedings of the 26th ACM SIGKDD international
conference on knowledge discovery & data mining, 2020, pp. 1192–1200.
[24] Y. Huang, T. Lv, L. Cui, Y. Lu, F. Wei, Layoutlmv3: Pre-training for document ai with unified text
and image masking, in: Proceedings of the 30th ACM international conference on multimedia,
2022, pp. 4083–4091.
[25] X. Ma, S.-C. Lin, M. Li, W. Chen, J. Lin, Unifying multimodal retrieval via document screenshot
embedding, arXiv preprint arXiv:2406.11251 (2024).

[26] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin,
M. Tschannen, E. Bugliarello, et al., Paligemma: A versatile 3b vlm for transfer, arXiv preprint
arXiv:2407.07726 (2024).
[27] Q. Team, et al., Qwen2 technical report, arXiv preprint arXiv:2407.10671 2 (2024).
[28] M. Günther, S. Sturua, M. K. Akram, I. Mohr, A. Ungureanu, B. Wang, S. Eslami, S. Martens,
M. Werk, N. Wang, et al., jina-embeddings-v4: Universal embeddings for multimodal multilingual
retrieval, in: Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL
2025), 2025, pp. 531–550.
[29] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al., Qwen2.
5-vl technical report, arXiv preprint arXiv:2502.13923 (2025).
[30] M. Xu, G. Moreira, R. Ak, R. Osmulski, Y. Babakhin, Z. Yu, B. Schifferer, E. Oldridge, Llama nemore-
triever colembed: Top-performing text-image retrieval model, arXiv preprint arXiv:2507.05513
(2025).
[31] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, M. Zaharia, Colbertv2: Effective and effi-
cient retrieval via lightweight late interaction, in: Proceedings of the 2022 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, 2022, pp. 3715–3734.
[32] J. Lee, F. Chen, S. Dua, D. Cer, M. Shanbhogue, I. Naim, G. H. Ábrego, Z. Li, K. Chen, H. S.
Vera, X. Ren, S. Zhang, D. Salz, M. Boratko, J. Han, B. Chen, S. Huang, V. Rao, P. Suganthan,
F. Han, A. Doumanoglou, N. Gupta, F. Moiseev, C. Yip, A. Jain, S. Baumgartner, S. Shahi, F. P.
Gomez, S. Mariserla, M. Choi, P. Shah, S. Goenka, K. Chen, Y. Xia, K. Chen, S. M. K. Duddu,
Y. Chen, T. Walker, W. Zhou, R. Ghiya, Z. Gleicher, K. Gill, Z. Dong, M. Seyedhosseini, Y. Sung,
R. Hoffmann, T. Duerig, Gemini embedding: Generalizable embeddings from gemini, 2025. URL:
https://arxiv.org/abs/2503.07891.arXiv:2503.07891.
[33] H. S. Vera, S. Dua, B. Zhang, D. Salz, R. Mullins, S. R. Panyam, S. Smoot, I. Naim, J. Zou,
F. Chen, D. Cer, A. Lisak, M. Choi, L. Gonzalez, O. Sanseviero, G. Cameron, I. Ballantyne,
K. Black, K. Chen, W. Wang, Z. Li, G. Martins, J. Lee, M. Sherwood, J. Ji, R. Wu, J. Zheng,
J. Singh, A. Sharma, D. Sreepathihalli, A. Jain, A. Elarabawy, A. Co, A. Doumanoglou, B. Samari,
B. Hora, B. Potetz, D. Kim, E. Alfonseca, F. Moiseev, F. Han, F. P. Gomez, G. H. Ábrego, H. Zhang,
H. Hui, J. Han, K. Gill, K. Chen, K. Chen, M. Shanbhogue, M. Boratko, P. Suganthan, S. M. K.
Duddu, S. Mariserla, S. Ariafar, S. Zhang, S. Zhang, S. Baumgartner, S. Goenka, S. Qiu, T. Dabral,
T. Walker, V. Rao, W. Khawaja, W. Zhou, X. Ren, Y. Xia, Y. Chen, Y.-T. Chen, Z. Dong, Z. Ding,
F. Visin, G. Liu, J. Zhang, K. Kenealy, M. Casbon, R. Kumar, T. Mesnard, Z. Gleicher, C. Brick,
O. Lacombe, A. Roberts, Q. Yin, Y. Sung, R. Hoffmann, T. Warkentin, A. Joulin, T. Duerig,
M. Seyedhosseini, Embeddinggemma: Powerful and lightweight text representations, 2025. URL:
https://arxiv.org/abs/2509.20354.arXiv:2509.20354.
[34] Y. Babakhin, R. Osmulski, R. Ak, G. Moreira, M. Xu, B. Schifferer, B. Liu, E. Oldridge, Llama-embed-
nemotron-8b: A universal text embedding model for multilingual and cross-lingual tasks, arXiv
preprint arXiv:2511.07025 (2025).
[35] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, F. Wei, Text embeddings by
weakly-supervised contrastive pre-training, arXiv preprint arXiv:2212.03533 (2022).
[36] T. Chen, S. Kornblith, M. Norouzi, G. Hinton, A simple framework for contrastive learning of visual
representations, in: International conference on machine learning, PmLR, 2020, pp. 1597–1607.
[37] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang,
A. Fan, et al., The llama 3 herd of models, arXiv e-prints (2024) arXiv–2407.
[38] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al., Qwen3
technical report, arXiv preprint arXiv:2505.09388 (2025).
[39] C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, W. Ping, Nv-embed: Improved
techniques for training llms as generalist embedding models, arXiv preprint arXiv:2405.17428
(2024).
[40] G. d. S. P. Moreira, R. Ak, B. Schifferer, M. Xu, R. Osmulski, E. Oldridge, Enhancing q&a text
retrieval with ranking models: Benchmarking, fine-tuning and deploying rerankers for rag, arXiv

preprint arXiv:2409.07691 (2024).
[41] S. Diao, Y. Yang, Y. Fu, X. Dong, D. Su, M. Kliegl, Z. Chen, P. Belcak, Y. Suhara, H. Yin, et al.,
Nemotron-climb: Clustering-based iterative data mixture bootstrapping for language model pre-
training, in: The Thirty-ninth Annual Conference on Neural Information Processing Systems
Datasets and Benchmarks Track, ????
[42] R. Tibshirani, G. Walther, T. Hastie, Estimating the number of clusters in a data set via the gap
statistic, Journal of the royal statistical society: series b (statistical methodology) 63 (2001) 411–423.
[43] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, A. G. Wilson, Averaging weights
leads to wider optima and better generalization, 2019. URL: https://arxiv.org/abs/1803.05407.
arXiv:1803.05407.
[44] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong,
A. Farhadi, Y. Carmon, S. Kornblith, L. Schmidt, Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference time, 2022. URL: https://arxiv.
org/abs/2203.05482.arXiv:2203.05482.
[45] R. Osmulsk, G. d. S. P. Moreira, R. Ak, M. Xu, B. Schifferer, E. Oldridge, Miracl-vision: A large,
multilingual, visual document retrieval benchmark, arXiv preprint arXiv:2505.11651 (2025).
[46] X. Zhang, N. Thakur, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu, M. Reza-
gholizadeh, J. Lin, Miracl: A multilingual retrieval dataset covering 18 diverse languages, Transac-
tions of the Association for Computational Linguistics 11 (2023) 1114–1131.
[47] G. d. S. P. Moreira, R. Ak, B. Schifferer, M. Xu, R. Osmulski, E. Oldridge, Enhancing q&a text
retrieval with ranking models: Benchmarking, fine-tuning and deploying rerankers for rag, in:
Proceedings of the 1st Workshop on GenAI and RAG Systems for Enterprises, co-located with
CIKM, 2024.
[48] A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder,
K. Chen, S. Kakade, P. Jain, et al., Matryoshka representation learning, Advances in Neural
Information Processing Systems 35 (2022) 30233–30249.
[49] B. Clavié, A little pooling goes a long way for multi-vector representations, 2024. URL: https:
//www.answer.ai/posts/colbert-pooling.html.
[50] L. Dhulipala, M. Hadian, R. Jayaram, J. Lee, V. Mirrokni, Muvera: multi-vector retrieval via fixed
dimensional encodings, arXiv preprint arXiv:2405.19504 (2024).
# Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation

**Authors**: Mengdan Zhu, Senhao Cheng, Guangji Bai, Yifei Zhang, Liang Zhao

**Published**: 2025-05-28 04:09:49

**PDF URL**: [http://arxiv.org/pdf/2505.21956v2](http://arxiv.org/pdf/2505.21956v2)

## Abstract
Text-to-image generation increasingly demands access to domain-specific,
fine-grained, and rapidly evolving knowledge that pretrained models cannot
fully capture. Existing Retrieval-Augmented Generation (RAG) methods attempt to
address this by retrieving globally relevant images, but they fail when no
single image contains all desired elements from a complex user query. We
propose Cross-modal RAG, a novel framework that decomposes both queries and
images into sub-dimensional components, enabling subquery-aware retrieval and
generation. Our method introduces a hybrid retrieval strategy - combining a
sub-dimensional sparse retriever with a dense retriever - to identify a
Pareto-optimal set of images, each contributing complementary aspects of the
query. During generation, a multimodal large language model is guided to
selectively condition on relevant visual features aligned to specific
subqueries, ensuring subquery-aware image synthesis. Extensive experiments on
MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal
RAG significantly outperforms existing baselines in both retrieval and
generation quality, while maintaining high efficiency.

## Full Text


<!-- PDF content starts -->

arXiv:2505.21956v2  [cs.CV]  29 May 2025Cross-modal RAG: Sub-dimensional
Retrieval-Augmented Text-to-Image Generation
Mengdan Zhu1,Senhao Cheng2,Guangji Bai1,Yifei Zhang1,Liang Zhao1
1Emory University2University of Michigan
{mengdan.zhu, guangji.bai, yifei.zhang2, liang.zhao}@emory.edu
senhaoc@umich.edu
Abstract
Text-to-image generation increasingly demands access to domain-specific, fine-
grained, and rapidly evolving knowledge that pretrained models cannot fully cap-
ture. Existing Retrieval-Augmented Generation (RAG) methods attempt to address
this by retrieving globally relevant images, but they fail when no single image
contains all desired elements from a complex user query. We propose Cross-
modal RAG, a novel framework that decomposes both queries and images into
sub-dimensional components, enabling subquery-aware retrieval and generation.
Our method introduces a hybrid retrieval strategy—combining a sub-dimensional
sparse retriever with a dense retriever—to identify a Pareto-optimal set of images,
each contributing complementary aspects of the query. During generation, a multi-
modal large language model is guided to selectively condition on relevant visual
features aligned to specific subqueries, ensuring subquery-aware image synthesis.
Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-
LT demonstrate that Cross-modal RAG significantly outperforms existing baselines
in both retrieval and generation quality, while maintaining high efficiency.
1 Introduction
User query:A Cybertruckwith a third-generation Labubuon the roof is parked in front of a Tesla store onVenusground.PreviousRAGOurs(Cross-modalRAG)RetrievedImagesTop-kGlobalRetrievalMulti-objectiveJointRetrieval
Generationquery+retrievedimagesquery+retrievedimageswithsub-dimensionalsatisfiedinformation
Cybertruck,Tesla storeCybertruck,Tesla storeCybertruck,Tesla storeCybertruck,Tesla storeVenusgroundVenusgroundthird-generation Labubu
Figure 1: Visualization of retrieval and gen-
eration in Cross-modal RAG (ours) versus
previous RAG.Text-to-Image Generation (T2I-G) has witnessed
rapid progress in recent years, driven by advances
in diffusion models [ 1,2,3] and multimodal large
language models (MLLMs) [ 4,5,6,7,8], enabling
the synthesis of increasingly realistic and diverse im-
ages from natural language descriptions. However,
in many real-world applications, domain-specific im-
age generation requires knowledge that is not readily
encoded within pre-trained image generators, espe-
cially when such information is highly long-tailed,
fast-updated, and proprietary. To address this limi-
tation, Retrieval-Augmented Generation (RAG) has
emerged as a promising paradigm by incorporating an
exrternal image database to supply factual reference
during generation [ 9,10]. Notable RAG-based im-
age generation approaches such as Re-Imagen [ 11],
RDM [ 12], and KNN-Diffusion [ 13] integrate re-
trieved images with diffusion models to improve out-
put fidelity. However, these existing RAG methods
typically rely on off-the-shelf retrievers (e.g., those
based on CLIP [ 14]) which compute global image-text similarities and retrieve whole images based
on the full user query. This coarse-grained retrieval strategy often fails in complex scenarios where

the query involves multiple fine-grained entities or attributes [ 15] – especially when no single image
contains all required components in the query. In practice, it is extremely common that single images
in the retrieval database only satisfy a subset of the query. As in Fig. 1, no single image in the retrieval
database perfectly covers all four aspects in the query; instead, each covers different subsets of the
query. Existing RAG methods often retrieve top-k images based on the entire query, so images that
redundantly contain most aspects of the query tend to be retrieved (e.g., three images that all include
“Cybertruck” and “Tesla store”), while some aspects may be underweighted (e.g., “Venus ground”) or
even missed (e.g., “third-generation Labubu”), leading to the distortion in the missed aspects. Also,
during generation, existing RAG has not be precisely instructed about which aspects of each image
should be leveraged, resulting in the superfluous lightning in the generated image by previous RAG.
Therefore, instead of being restricted to considering whether each whole image is related to the
entirety of the query, it is desired to advance to pinpointing which aspects (i.e., sub-dimensions) of
which images can address which aspects of the query for image generation. Such desire is boiled
down to several open questions. First , how to precisely gauge which part of the queries match which
aspects of each image? Existing global embedding methods, like CLIP, do not naturally support
sub-dimensional alignment [ 14], and current fine-grained vision-language matching is limited to
region-level object patterns [ 15,16], which are computationally expensive and error-prone. Second ,
how to retrieve the smallest number of images that cover all necessary information? It is desired to
retrieve an optimal set of images such that each covers different aspects of the query while avoiding
redundancy in order to maximize the amount of relevant information under a limited context window
size. Third , how to precisely inform the image generator of what aspect of each image to refer to
when generating? Existing image generators typically can take in the input query or images, but here
it requires adding fine-grained instructions about how to leverage relevant aspects of images when
generating, which is not well explored.
To address these open problems, we propose Cross-modal Sub-dimensional Retrieval Augmented
Generation (Cross-modal RAG) , a novel text-to-image generation that can identify, retrieve, and
leverage image sub-dimensions to satisfy different query aspects. To decompose and identify key
image sub-dimensions, we decompose the user query into subqueries and candidate images into
sub-dimensional representations with respect to the subqueries, enabling accurate subquery-level
alignment. To retrieve comprehensive and complementary image sub-dimensions, we formulate the
retrieval goal as a multi-objective optimization problem and introduce an efficient hybrid retrieval
strategy – combining a lightweight sub-dimensional sparse retriever with a sub-dimensional dense
retriever – to retrieve a set of Pareto-optimal images that collectively cover all subqueries in the query
as in Fig. 1 (right). To effectively instruct image generators with the retrieved image sub-dimensions,
we present a model-agnostic and subquery-aware generation with MLLMs, which explicitly preserves
and composes the subquery-aligned components from the retrieved images into a coherent final image.
For instance, our method only preserves the “Venus ground” in the final image, while previous RAG
can also preserve the irrelevant lightning in Fig. 1. Extensive experiments demonstrate that Cross-
modal RAG achieves state-of-the-art performance in both text-to-image retrieval and text-to-image
generation tasks across multiple fine-grained, domain-specific, and long-tailed image benchmarks,
while maintaining excellent computational efficiency.
2 Related Work
2.1 Text-to-Image Generation
Text-to-Image Generation (T2I-G) has made significant strides, evolving through methodologies such
as Generative Adversarial Networks (GANs) [ 17,18], auto-regressive models [ 19,20], and diffusion
models [ 21,22]. Recent breakthroughs in diffusion models and multimodal large language models
(MLLMs), driven by scaling laws [ 23], have significantly advanced the capabilities of T2I-G. Notable
examples include the DALL-E series [ 20,24], the Imagen series [ 2], and the Stable Diffusion (SD)
series [ 1,25,26]. More recently, image generation functionalities have been integrated directly into
advanced MLLMs such as GPT Image [ 4] and Gemini 2.0 Flash Image Generation [ 5]. However,
despite these advancements, traditional T2I-G methods often struggle with knowledge-intensive,
long-tailed, and fine-grained image-generation tasks. These scenarios typically require additional
context to generate accurate images, necessitating RAG techniques.
2

Cli$ near Dieppe in the style of Claude MonetA Cli$ in the style of Pavel SvinyinDieppe in the style of Charles ConderCli$ at Dieppe in the style of Claude MonetStabble near Dieppe in the style of Paul GauguinSeascape with cow on the edge of a cli$ in the style of Paul Gauguin
“Cli% near Dieppe in the style of Paul Gauguin.”
User queryCli$DieppeThe style of Paul Gauguin
✔
✔
✔
✔
✖
✖
CLIP VisualEnoder
CLIP TextEnoder
Stage 1: Sub-dimensional Sparse RetrieverStage 2: Sub-dimensional Dense RetrieverStage 3: Multi-objective Joint RetrievalStage 4: Generation
111Pareto Front 
Pareto Optimal Image Set 
Cli;, Dieppe 
Dieppe, The style of Paul Gauguin Dieppe, The style of Paul Gauguin SatisﬁedSubqueries
MLLM
GeneratedImage
pairwise cosine similarityweighted meanuser query/visual/subquery tokensvisual embeddingstext embeddingsRetrievalDatabase
❄
❄
❄Learnable querytokens
Cross attention 
🔥Cross attention FeedForwardAdaptor
Figure 2: Overview of the Cross-modal RAG framework. The framework consists of four stages:
(1) Sub-dimensional Sparse Retriever, where images are filtered based on lexical subquery matches;
(2) Sub-dimensional Dense Retriever, where candidate images are re-ranked using the mean of
pairwise cosine similarities between sub-dimensional vision embeddings and subquery embeddings;
(3) Multi-objective Joint Retrieval, where a Pareto-optimal set of images is selected by Eq.6 to
collectively cover the subqueries. Pfis composed of three orange points(
 ) (solid points are on the
line while dashed points are off the line); and (4) Generation, where a MLLM composes a final image
by aligning subquery-level visual components from retrieved images.
2.2 Text-to-Image Retrieval
Text-to-Image Retrieval (T2I-R) has become a crucial subtask in supporting fine-grained image
understanding and generation. CLIP [ 14] is currently the most widely adopted approach, mapping
images and texts into a shared embedding space via contrastive learning. While CLIP excels at coarse-
grained alignment, it underperforms in fine-grained text-to-image retrieval, especially in scenes
involving multiple objects or nuanced attributes. ViLLA [ 15] explicitly highlights this limitation,
demonstrating that CLIP fails to capture detailed correspondences between image regions and textual
attributes. SigLIP [ 27], along with other refinements such as FILIP [ 28] and SLIP [ 29], improves
CLIP’s contrastive learning framework and achieves superior zero-shot classification performance.
However, these methods still rely on global image-text embeddings, which are inadequate for
resolving localized visual details required by fine-grained queries.
To address this, recent works on fine-grained text-to-image retrieval (e.g., ViLLA [ 15], Region-
CLIP [ 16]) have adopted region-based approaches that involve cropping image patches for localized
alignment. In contrast, our vision-based sub-dimensional dense retriever bypasses the need for
explicit cropping. By constructing sub-dimensional vision embeddings directly from the full image,
we enable more efficient and effective matching against subqueries.
2.3 Retrieval-Augmented Generation
Retrieval-Augmented Generation has demonstrated significant progress in improving factuality for
both natural language generation [ 30,31] and image generation [ 11,32]. Most RAG-based approaches
for image generation are built upon diffusion models (e.g., Re-Imagen [ 11], RDM [ 12], KNN-
Diffusion [ 13]), but these methods largely overlook fine-grained semantic alignment. FineRAG [ 33]
takes a step toward fine-grained image generation by decomposing the textual input into fine-grained
entities; however, it does not incorporate fine-grained decomposition on the visual side. In contrast,
our approach performs dual decomposition: (i) the query is parsed into subqueries that capture distinct
semantic components, and (ii) the candidate images are decomposed into sub-dimensional vision
embeddings aligned with the corresponding subqueries. Furthermore, while existing RAG-based
image models typically rely on off-the-shelf retrievers, we introduce a novel retrieval method that
combines a sub-dimensional sparse filtering stage with a sub-dimensional dense retriever. Finally,
with the recent surge of MLLM-based image generation, we explore how our fine-grained retrieval
information can be integrated to guide generation at the sub-dimensional level.
3 Proposed Method
We introduce Cross-modal RAG , as shown in Figure 2. The framework consists of four stages: (1) Sub-
dimensional sparse retriever based on lexical match on subqueries in Sec. 3.1.2; (2) Sub-dimensional
dense retriever based on semantic match on sub-dimensional vision embeddings and textual subquery
3

embeddings in Sec. 3.1.1; (3) Multi-objective joint retrieval to select a set of Pareto-optimal images
in Sec. 3.1.3; and (4) Subquery-aware image generation with retrieved images in Sec. 3.2.
The framework of Cross-modal RAG focuses on: 1) how to retrieve theoptimal images from the
retrieval database given multiple subqueries, and 2) how to guide the generator to generate images
preserving the satisfied subquery features in each retrieved image.
3.1 Multi-objective Retrieval for Image Generation
3.1.1 Sub-dimensional Dense Retriever
Given a user query Qand a candidate image Ij, we decompose Qinto a set of subqueries
{q1, q2, ..., q n}, where each subquery qicaptures a specific aspect of Q, such as object categories
or attributes, and we further compute the similarity scores between its normalized sub-dimensional
vision embeddings and textual subquery embeddings as follows:
S(Q, Ij) =1
nnX
i=1sim(vji, ti). (1)
Here, the similarity score sim is cosine similarity. The similarity scores are aggregated across n
subqueries to form an overall similarity metric S(Q, Ij). Images are ranked based on their similarity
S(Q, Ij)for the given query, and the top-ranked images are retrieved.
In terms of the sub-dimensional vision embeddings vji, after the image Ijis fed into a pretrained
CLIP vision encoder (Φclip-v(·)), a multi-head cross-attention module is introduced, functioning as
the vision adapter fa, to compute fine-grained sub-dimensional vision subembeddings:
vji=fa(Φclip-v(Ij), ti). (2)
The vision adapter consists of: 1) A multi-head vision cross-attention layer where the learnable
query tokens attend to the vision embeddings extracted from the frozen CLIP visual encoder; 2) A
multi-head text cross-attention layer where the output of the vision cross-attention further attends to
the subquery embeddings extracted from the frozen CLIP text encoder; 3) An MLP head that maps
the attended features to a shared multimodal embedding space, followed by layer normalization. The
output vjirepresents Ij’s ith-dimensional vision embedding corresponding to the subquery qi, which
is decomposed from Qand can be obtained by an off-the-shelf LLM (e.g. GPT-4o mini) using the
structured prompt in Appendix A.
The subquery embeddings tiwith respect to subquery qican be computed as:
ti= Φ clip-t( g(qi)), (3)
where g(·)denotes the tokenization, and Φclip-t(·)denotes the pre-trained CLIP text encoder.
The vision adapter fais optimized using the Info-NCE loss:
LInfo-NCE =−logP
(vji,ti)∈Pexp
⟨vT
ji,ti⟩
τ
P
(vji,ti)∈Pexp
⟨vT
ji,ti⟩
τ
+P
(v′
ji,t′
i)∼Nexp
⟨v′T
ji,t′
i⟩
τ, (4)
where Pis a set of positive pairs with all sub-dimensional vision embeddings and subquery embed-
dings,Nconversely refers to an associated set of negative pairs. τis a temperature parameter.
3.1.2 Sub-dimensional Sparse Retriever
Definition 3.1 (Sub-dimensional Sparse Retriever) .For each retrieval candidate image Ij, we define
a binary satisfaction score for the sub-dimensional sparse retriever:
si(Ij) =1,if the caption of Ijcontains qi
0,otherwise(5)
Hence, each image Ijyields an n-dimensional satisfaction vector [s1(Ij), . . . , s n(Ij)].
Definition 3.2 (Image Dominance) .Consider two images IaandIbfrom the retrieval database
D, with corresponding subquery satisfaction vectors s(Ia) = [ s1(Ia), . . . , s n(Ia)]and s(Ib) =
[s1(Ib), . . . , s n(Ib)]. We say Iadominates Ib, denoted Ia≻Ib, if:
4

Algorithm 1 MULTI -OBJECTIVE JOINT RETRIEVAL ALGORITHM
Require: Query Qdecomposed into subqueries {qi}n
i=1, image retrieval database D, weights
{αi}n
i=1withαi>0,P
iαi= 1, trade-off parameter βwith0< β < β max
Ensure: The set of Pareto optimal images P={I∗
j}
1:forIj∈ D do
2: compute s(Ij) = [s1(Ij), . . . , s n(Ij)]
3:end for
4:eD← {Ij|s(Ij)is not all zeros }
5:P ← ∅
6:forαin a discretized grid over the simplex do
7: P ← P ∪ arg maxIj∈eDPn
i=1αisi(Ij) +β·nS(Q, Ij)
8:end for
(1)si(Ia)≥si(Ib),∀i∈ {1, . . . , n }, and (2)∃j∈ {1, . . . , n }s.t.sj(Ia)> sj(Ib).
That is, Iais never worse in any subquery’s score and is strictly better in at least one subquery.
Ia∈ D is retrieved by the sub-dimensional sparse retriever if there exists no other image Ib∈ D
such that Ibdominates Ia. Formally, ∄Ib∈ D s.t.Ib≻Ia.
3.1.3 Multi-objective Optimization Formulation and Algorithm
Each subquery can be regarded as a distinct objective, giving rise to a multi-objective optimization
problem for retrieval. Our primary goal is to select images that collectively maximize text-based
subquery satisfaction (sub-dimensional sparse retrieval), while also maximizing fine-grained vision-
based similarity (sub-dimensional dense retrieval). Thus, the overall objective is formalized as:
I∗
j= arg max
IjnX
i=1αisi(Ij) +β·nS(Q, Ij),s.t.∀αi:αi>0,nX
i=1αi= 1, β∈(0, βmax),(6)
where αiis the relative importance of each subquery qiin the sub-dimensional sparse retrieval, and
the weight βtrades off between the sub-dimensional sparse and dense retrieval.
Definition 3.3 (Pareto Optimal Images) .The solution to Eq.6 is called the set of Pareto optimal
images , such that I∗
jis not dominated by by any other image Ik∈ D in terms of both s(I)and
S(Q, I). Formally,
P={I∗
j∈ D | ∄Ik∈ D s.t.F(Ik)> F(I∗
j)}, (7)
where F(Ij) =Pn
i=1αisi(Ij) +β·nS(Q, Ij),s.t.∀αi:αi>0,Pn
i=1αi= 1, β∈(0, βmax).
Definition 3.4 (Pareto Front of the Pareto Optimal Images) .Pis sometimes referred to as the Pareto
setin the decision space (here, the set of images). Pareto front Pfof the Pareto optimal image is the
corresponding set of non-dominated tuples in the objective space:
Pf={(s(I∗
j), S(Q, I∗
j)) :I∗
j∈ P} . (8)
Therefore, the Pareto optimal images in Prepresent the “best trade-offs” across all subqueries, since
no single image in Dcan strictly improve the Pareto front Pfon every subquery dimension.
We propose the multi-objective joint retrieval algorithm in Algorithm 1. If an image is Pareto optimal,
there exists at least one choice of { αi} for which it can maximizePn
i=1αisi(Ij). In particular, if
multiple images share the same subquery satisfaction vector s(Ij), we can use the sub-dimensional
dense retriever to further distinguish among images.
Theorem 3.1 (Retrieval Efficiency) .LetNbe the total number of images in D,eN≪Nbe the
number of images in eD,Kis a grid of α-values and nbe the number of subqueries. Also let Tclip
represent the cost of processing a single image with the CLIP vision encoder, and Tadaptor represent
the cost of the adaptor. The time complexity of Algorithm 1 is O(N) +O(K×eN) +O(K×
eN×n×(Tclip+Tadaptor ))and the time complexity of a pure sub-dimensional dense retriever is
O(N×n×(Tclip+Tadaptor )).
Proof. The formal proof can be found in Appendix B.
5

SinceeN≪NandKis a relatively small constant, the dominant term of Algorithm 1 is far less than
a pure sub-dimensional dense retriever. In terms of retrieval efficiency, we adopt Algorithm 1 - a
hybrid of sub-dimensional sparse and dense retriever.
Theorem 3.2 (Algorithm Optimality) .Letδmin= min {αi|αi>0}be the smallest nonzero
subquery weight, and Cmax= maxPn
i=1cos (vj,i, ti). For any 0< β < β max=δmin
Cmax, Algorithm 1
returns all Pareto-optimal solutions to Eq.6.
Proof. The formal proof can be found in Appendix C.
3.2 Image Generation with Retrieved Images
To generate an image from the user query Qwhile ensuring that the satisfied subquery features in R
are preserved, we utilize a pretrained MLLM with subquery-aware instructions.
Given the set of retrieved images is P={I∗
j}, each retrieved image I∗
jis associated with a subquery
satisfaction vector s(I∗
j). Let
Qr={qi|si(I∗
j) = 1} (9)
be the subset of subqueries from the user query QthatI∗
jactually satisfies.
For each image I∗
j∈ P, we construct an in-context example in a form: ⟨I∗
j⟩Use only [ Qr] in [I∗
j].
Here,⟨I∗
j⟩denotes the visual tokens for the r-th retrieved image, [ Qr] is the satisfied subqueries in
I∗
j, and [ I∗
j] is "the r-th retrieved image".
Next, we feed the in-context examples to a pretrained MLLM together with the original query Q. The
MLLM, which operates in an autoregressive manner, is thus guided to generate the final image ˆIas:
pθ ˆIQ,{I∗
j},{Qr}
=TY
t=1pθ
ˆItˆI<t, Q,{I∗
j},{Qr}
, (10)
where ˆItdenotes the t-th visual token in the generated image representation, and θrep-
resents the parameters of the pretrained MLLM. By referencing the full prompt: [Q]
⟨I∗
j⟩Use only [ Qr] in [ I∗
j], the MLLM learns to preserve the relevant subquery features that
each retrieved image I∗
jcontributes.
4 Experiments
4.1 Experiment Setup
Baselines and Evaluation Metrics We compare our proposed method with several baselines on
text-to-image retrieval and text-to-image generation.
•Text-to-Image Retrieval Baselines: CLIP(ViT-L/14) [ 14] is a widely adopted dual-encoder model
pretrained on large-scale image-text pairs and remains the most commonly used baseline for T2I
retrieval. SigLIP(ViT-SO400M/14@384) [ 27] improves retrieval precision over CLIP by replacing
the contrastive loss with a sigmoid-based loss. ViLLA [ 15] is a large-scale vision-language
pretraining model with multi-granularity objectives. GRACE (Structured ID) [ 34] is a recent
generative cross-modal retrieval model. IRGen [ 35] is a transformer-based image-text retriever.
We report GRACE and IRGen results as cited from [ 36]. GILL [ 37] is a unified framework that
combines generation and retrieval.
•Text-to-Image Generation Baselines: SDXL [ 25] is a widely used high-quality T2I diffusion model.
LaVIT [ 6] is a vision-language model that supports T2I generation. RDM [ 12] is a representative
retrieval-augmented diffusion model. UniRAG [ 38] is a recent retrieval-augmented vision-language
model. GILL [37] can perform both T2I-R and T2I-G.
For T2I-R, we adopt the standard retrieval metric Recall at K(R@K, K=1, 5, and 10). For T2I-G,
we evaluate the quality of generated images by computing the average pairwise cosine similarity
of generated and ground-truth images with CLIP(ViT-L/14) [ 14], DINOv2(ViT-L/14) [ 39], and
SigLIP(ViT-SO400M/14@384) [ 27] embeddings. We also employ style loss [ 40] to assess the artistic
style transfer in the WikiArt dataset [41].
6

Table 1: Evaluation of Text-to-Image Retrieval on MS-COCO and Flickr30K.
MethodMS-COCO (5K) Flickr30K (1K)
R@1 R@5 R@10 R@1 R@5 R@10
CLIP (ViT-L/14) 43.26 68.70 78.12 77.40 94.80 96.60
SigLIP (ViT-SO400M/14@384) 46.96 71.72 80.64 82.20 95.90 97.70
ViLLA 34.77 60.67 70.69 59.41 85.02 92.82
GRACE (Structured ID) 16.70 39.20 50.30 37.40 59.50 66.20
IRGen 29.60 50.70 56.30 49.00 68.90 72.50
GILL 32.12 57.73 66.55 55.41 81.94 89.77
Ours 80.78 97.00 99.16 97.50 100.00 100.00
Table 2: Evaluation of Text-to-Image Generation on WikiArt, CUB, and ImageNet-LT.
MethodWikiArt CUB ImageNet-LT
CLIP↑DINO ↑SigLIP ↑Style Loss ↓CLIP↑DINO ↑SigLIP ↑CLIP↑DINO ↑SigLIP ↑
SDXL 0.688 0.504 0.720 0.022 0.743 0.519 0.738 0.668 0.403 0.653
LaVIT 0.689 0.485 0.721 0.036 0.676 0.245 0.647 0.662 0.365 0.652
RDM 0.507 0.237 0.528 0.024 0.638 0.326 0.663 0.576 0.333 0.603
UniRAG 0.646 0.362 0.654 0.068 0.746 0.344 0.718 0.610 0.255 0.600
GILL 0.629 0.439 0.654 0.027 0.719 0.185 0.675 0.635 0.228 0.615
Ours 0.746 0.604 0.744 0.019 0.764 0.600 0.744 0.815 0.761 0.812
Dataset Construction We evaluate the text-to-image retrieval on the standard benchmark MS-
COCO [ 42] and Flickr30K [ 43] test sets. As for the text-to-image generation, we evaluate the model’s
image generation capabilities across different aspects and choose three datasets: artistic style transfer
in the WikiArt [ 41], fine-grained image generation in the CUB [ 44], and long-tailed image generation
in the ImageNet-LT [ 45]. For each genereation dataset, we select some test samples, and use the
remaining as the retrieval database. More details in Appendix D.
Implementation Details For T2I-R, our sub-dimensional dense retriever is composed of a pretrained
CLIP vision encoder (ViT-L/14) and an adaptor. We train the sub-dimensional dense retriever on the
COCO training set using the InfoNCE loss with a temperature of 0.07. The adaptor is optimized
using the Adam optimizer with an initial learning rate of 5e-5, and a StepLR scheduler with a step
size of 3 epochs and a decay factor of 0.6. For T2I-G, we use gpt-image-1 as our MLLM backbone
and set β= 0.015based on Therorem 3.2. The experiments1are conducted on a 64-bit machine with
24-core Intel 13th Gen Core i9-13900K@5.80GHz, 32GB memory and NVIDIA GeForce RTX 4090.
4.2 Quantitative Evaluation of Text-to-Image Retrieval
We test our sub-dimensional dense retriever model with various types of T2I-R models. As shown in
Tab. 1, our proposed sub-dimensional dense retriever achieves state-of-the-art performance across
all metrics and outperforms all baselines by a substantial margin on both MS-COCO and Flickr30K
datasets. On MS-COCO, our method achieves R@1 = 80.78%, R@5 = 97.00%, and R@10 = 99.16%,
which are significantly higher than the best-performing baseline SigLIP. The relative improvements
are 72% on R@1, 35% on R@5, and 20% on R@10, demonstrating our model’s superior capability
in the text-to-image retrieval. Notably, it exhibits strong zero-shot T2I-R performance on Flickr30K,
achieving near-perfect accuracy with R@1 = 97.50%, R@5 = 100.00%, and R@10 = 100.00%, and
surpassing SigLIP’s R@1 = 82.20% by nearly 20%. These results confirm that our proposed sub-
dimensional dense retriever significantly enhance fine-grained T2I-R compared to global embedding
alignment such as CLIP and SigLIP, generative retrieval methods like GRACE and IRGen, region-
based fine-grained match on ViLLA, and the retrieval and generation unified framework GILL.
4.3 Quantitative Evaluation of Text-to-Image Generation
We benchmark our Cross-modal RAG method against state-of-the-art text-to-image generation models,
including diffusion-based (SDXL), autoregressive (LaVIT), retrieval-augmented (RDM, UniRAG),
and retrieval and generation unified (GILL) baselines. The evaluation is conducted on three datasets
that span different generation challenges: WikiArt (artistic style transfer), CUB (fine-grained), and
ImageNet-LT (long-tailed). As shown in Tab. 2, Cross-modal RAG achieves the highest scores across
1The code is available at https://github.com/mengdanzhu/Cross-modal-RAG.
7

UserQueryDecomposedSubqueriesPareto Optimal Images with  Satisfied SubqueriesRetrievedimage1
GenerationRetrievedimage2Retrievedimage3Draw a Hooded Warbler. This bird has a black crown with black throat and yellow belly.
1.Hooded Warbler,2.black crown, 3.black throat,4.yellow belly
1.Hooded Warbler2.black crown4.yellow belly1.Hooded Warbler2.black crown3.black throat1.Hooded Warbler3.black throat4.yellow belly
UserQueryDecomposedSubqueries
Retrievedimage1A photo of [totempole].1.[totempole]
1.[totempole]
Pareto Optimal Images with  Satisfied Subqueries
Generation(b)CUB(c)ImageNet-LT(a)WikiArtUserQueryRoadin theforest ofFontainebleau in the style of Theodore Rousseau.1.road,2.forest, 3.Fontainebleau,4.the style of Theodore Rousseau
Retrievedimage1Retrievedimage2Retrievedimage3
1.road2.forest3.Fontainebleau2.forest3.Fontainebleau4.the style of Theodore Rousseau1.road4.the style of Theodore Rousseau
DecomposedSubqueriesPareto Optimal Images with  Satisfied Subqueries
Generation
Figure 3: The retrieved pareto optimal images with their corresponding satisfied subqueries in (a)
WikiArt, (b) CUB and (c) ImageNet-LT datasets and model generation results of Cross-modal RAG.
all models and datasets. On WikiArt, our method achieves the best performance in CLIP, DINO, and
SigLIP, along with the lowest style loss, indicating it can capture the particular artistic style specified
in the retrieved images effectively. On CUB, Cross-modal RAG also performs strongly across all
three metrics, because it can localize and leverage the specific visual details in the retrieved images to
facilitate generation. On ImageNet-LT, Cross-modal RAG improves CLIP similarity by 22%, DINO
by 89%, and SigLIP by 24% over the second-best SDXL. This indicates that our retrieval method
can retrieve images that best match the query and only use the relevant entity for generation, which
greatly benefits T2I-G in the long-tailed situation.
4.4 Qualitative Analysis
To qualitatively illustrate the effectiveness of our Cross-modal RAG model, we visualize some
examples of our retrieved pareto optimal images with their corresponding satisfied subqueries and
generated outputs across all datasets in Fig. 3. The satisfied subqueries of each retrieved Pareto-
optimal image are non-overlapping, and each retrieved image is optimal with respect to the sub-
dimensions it satisfies. Therefore, we can guarantee that the Pareto set Pcollectively covers images
with all satisfied subqueries in the retrieval database D. Moreover, since the model knows which
subqueries are satisfied by each retrieved image, MLLM can be guided to condition on the relevant
subquery-aligned parts of each retrieved image during generation. As shown Fig. 3(a), the model is
capable of style transfer , learning the artistic style of a certain artist ( e.g., Theodore Rousseau) while
preserving the details corresponding to each subquery ( e.g., road, forest, Fontainebleau). The model
is also able to retrieve accurate fine-grained information and compose the entities in subqueries ( e.g.,
black crown, black throat, yellow belly) to perform fine-grained image generation on the CUB dataset
in Fig. 3(b). Moreover, the model is good at long-tailed or knowledge-intensive image generation .
In Fig. 3(c), ImageNet-LT is a long-tailed distribution dataset with many rare entities ( e.g., totem
8

Table 3: Evaluation of the retrieval efficiency on the COCO. Our methods are denoted in gray .
Method GPU Memory (MB) # of Parameters (M) Query Latency (ms)
CLIP (ViT-L/14) 2172.19 427.62 8.68
dense (all) 2195.26 433.55 14.22
dense (adaptor) 23.07 7.31 4.35
sparse 0 0 2.17
pole). Retrieving such correct images can help improve generation fidelity. Baseline models without
retrieval capabilities tend to struggle in these scenarios. More comparisons of generated images with
other baselines are provided in Appendix F.
4.5 Efficiency Analysis
As shown in Tab. 3, we compare the retrieval efficiency of CLIP (ViT-L/14) with our sub-dimensional
dense and sparse retrievers on the COCO test set. Our sub-dimensional dense retriever is com-
posed of a frozen CLIP encoder (ViT-L/14) and a lightweight adaptor. As reported in Table 1, the
sub-dimensional dense retriever improves Recall@1 by +86.73% over CLIP on COCO. Despite
the adaptor’s minimal overhead – only 0.01 ×CLIP’s GPU memory usage, 0.017 ×its number of
parameters, and 0.5 ×its query latency – its performance gain is substantial. Our sub-dimensional
sparse retriever is text-based and operates solely on the CPU, requiring no GPU memory consumption,
no learnable parameters and achieving the lowest query latency. Our Cross-modal RAG method, a
hybrid of our sub-dimensional sparse and dense retriever, can leverage the complementary strengths
of both and achieve query latency that lies between the pure sparse and dense retrievers – closer to
that of the sparse. These results show Cross-modal RAG’s efficiency and scalability for large-scale
text-to-image retrieval tasks without compromising effectiveness.
4.6 Ablation Study
Ablation Study on Subquery Decomposition
Figure 4: Ablation Study on Subquery De-
composition on the WikiArt and CUB.We evaluate retrieval performance without subquery
decomposition on multi-subquery datasets, WikiArt
and CUB, by directly using a BM25 retriever to re-
trieve the top-1, top-2, and top-3 images based on
the full user query. Our multi-objective joint retrieval
method achieves a higher subquery coverage rate
compared to the conventional text-based BM25 re-
trieval on both WikiArt and CUB in Fig. 4. This
result indicates that our multi-objective joint retrieval
method retrieves a set of images Pthat collectively cover the largest number of subqueries from D,
demonstrating its superior ability to capture the full semantic intent of the user query.
Ablation Study on the Sub-dimensional Dense Retriever
We retain the sub-dimensional sparse retriever and
replace the sub-dimensional dense retriever in Cross-
modal RAG with a randomly selected image. The
results in Tab. 4 show that our dense retriever is
able to retrieve images that best match the entity in
the query in the ImageNet-LT. Notably, our dense
retriever, though trained only on the COCO, general-
izes well to unseen entities on the ImageNet-LT.Table 4: Ablation Study of our Cross-modal
RAG w/o dense retriever on the ImageNet-LT.
Method CLIP ↑DINO ↑SigLIP ↑
Ours 0.815 0.761 0.812
w/o dense 0.773 0.607 0.752
5 Conclusion
We proposed Cross-modal RAG, a novel sub-dimensional retrieval-augmented text-to-image gener-
ation framework addressing domain-specific, fine-grained, and long-tailed image generation. Our
method leverages a hybrid retrieval strategy combining sub-dimensional sparse filtering with dense
retrieval to precisely align subqueries with visual elements, guiding a MLLM to generate coherent
images on the subquery level. The Pareto-optimal image selection ensures the largest coverage of
various aspects in the query. Extensive experiments demonstrated Cross-modal RAG’s superior
performance over state-of-the-art baselines in T2I-R and T2I-G. The ablation study and efficiency
analysis highlight the effectiveness of each component and efficiency in the Cross-modal RAG.
9

References
[1]Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition , pages 10684–10695, 2022.
[2]Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton,
Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al.
Photorealistic text-to-image diffusion models with deep language understanding. Advances in
neural information processing systems , 35:36479–36494, 2022.
[3]Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew,
Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing
with text-guided diffusion models. arXiv preprint arXiv:2112.10741 , 2021.
[4]OpenAI. Addendum to gpt-4o system card: Native image generation, March 2025. Accessed:
2025-05-08.
[5]Google. Experiment with gemini 2.0 flash native image generation, April 2025. Accessed:
2025-05-08.
[6]Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Quzhe Huang, Bin Chen, Chenyi Lei,
An Liu, Chengru Song, et al. Unified language-vision pretraining in llm with dynamic discrete
visual tokenization. arXiv preprint arXiv:2309.04669 , 2023.
[7]Mengdan Zhu, Raasikh Kanjiani, Jiahui Lu, Andrew Choi, Qirui Ye, and Liang Zhao. La-
tentexplainer: Explaining latent representations in deep generative models with multi-modal
foundation models. arXiv preprint arXiv:2406.14862 , 2024.
[8]Guangji Bai, Zheng Chai, Chen Ling, Shiyu Wang, Jiaying Lu, Nan Zhang, Tingwei Shi,
Ziyang Yu, Mengdan Zhu, Yifei Zhang, et al. Beyond efficiency: A systematic survey of
resource-efficient large language models. arXiv preprint arXiv:2401.00625 , 2024.
[9]Xu Zheng, Ziqiao Weng, Yuanhuiyi Lyu, Lutao Jiang, Haiwei Xue, Bin Ren, Danda Paudel,
Nicu Sebe, Luc Van Gool, and Xuming Hu. Retrieval augmented generation and understanding
in vision: A survey and new outlook. arXiv preprint arXiv:2503.18016 , 2025.
[10] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated
content: A survey. arXiv preprint arXiv:2402.19473 , 2024.
[11] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-
augmented text-to-image generator. arXiv preprint arXiv:2209.14491 , 2022.
[12] Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas Müller, and Björn Ommer. Retrieval-
augmented diffusion models. Advances in Neural Information Processing Systems , 35:15309–
15324, 2022.
[13] Shelly Sheynin, Oron Ashual, Adam Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, and
Yaniv Taigman. Knn-diffusion: Image generation via large-scale retrieval. arXiv preprint
arXiv:2204.02849 , 2022.
[14] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning ,
pages 8748–8763. PmLR, 2021.
[15] Maya Varma, Jean-Benoit Delbrouck, Sarah Hooper, Akshay Chaudhari, and Curtis Langlotz.
Villa: Fine-grained vision-language representation learning from real-world data. In Proceedings
of the IEEE/CVF International Conference on Computer Vision , pages 22225–22235, 2023.
[16] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li,
Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-based language-image
pretraining. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition , pages 16793–16803, 2022.
10

[17] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural
information processing systems , 27, 2014.
[18] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity
natural image synthesis. arXiv preprint arXiv:1809.11096 , 2018.
[19] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural
networks. In International conference on machine learning , pages 1747–1756. PMLR, 2016.
[20] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea V oss, Alec Radford, Mark
Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on
machine learning , pages 8821–8831. Pmlr, 2021.
[21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances
in neural information processing systems , 33:6840–6851, 2020.
[22] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic
models. In International conference on machine learning , pages 8162–8171. PMLR, 2021.
[23] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models. arXiv preprint arXiv:2001.08361 , 2020.
[24] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang,
Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions.
Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf , 2(3):8, 2023.
[25] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe
Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image
synthesis. arXiv preprint arXiv:2307.01952 , 2023.
[26] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini,
Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow trans-
formers for high-resolution image synthesis. In Forty-first international conference on machine
learning , 2024.
[27] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for
language image pre-training. In Proceedings of the IEEE/CVF international conference on
computer vision , pages 11975–11986, 2023.
[28] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang,
Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image
pre-training. arXiv preprint arXiv:2111.07783 , 2021.
[29] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets
language-image pre-training. In European conference on computer vision , pages 529–544.
Springer, 2022.
[30] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models:
A survey. arXiv preprint arXiv:2312.10997 , 2:1, 2023.
[31] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan
Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen. Large language models for information
retrieval: A survey. arXiv preprint arXiv:2308.07107 , 2023.
[32] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang,
Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Retrieval-augmented multimodal language
modeling. arXiv preprint arXiv:2211.12561 , 2022.
[33] Huaying Yuan, Ziliang Zhao, Shuting Wang, Shitao Xiao, Minheng Ni, Zheng Liu, and Zhicheng
Dou. Finerag: Fine-grained retrieval-augmented text-to-image generation. In Proceedings of
the 31st International Conference on Computational Linguistics , pages 11196–11205, 2025.
11

[34] Yongqi Li, Wenjie Wang, Leigang Qu, Liqiang Nie, Wenjie Li, and Tat-Seng Chua. Generative
cross-modal retrieval: Memorizing images in multimodal language models for retrieval and
beyond. arXiv preprint arXiv:2402.10805 , 2024.
[35] Yidan Zhang, Ting Zhang, Dong Chen, Yujing Wang, Qi Chen, Xing Xie, Hao Sun, Weiwei
Deng, Qi Zhang, Fan Yang, et al. Irgen: Generative modeling for image retrieval. In European
Conference on Computer Vision , pages 21–41. Springer, 2024.
[36] Leigang Qu, Haochuan Li, Tan Wang, Wenjie Wang, Yongqi Li, Liqiang Nie, and Tat-Seng
Chua. Tiger: Unifying text-to-image generation and retrieval with large multimodal models. In
The Thirteenth International Conference on Learning Representations .
[37] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal
language models. Advances in Neural Information Processing Systems , 36:21487–21506, 2023.
[38] Sahel Sharifymoghaddam, Shivani Upadhyay, Wenhu Chen, and Jimmy Lin. Unirag: Universal
retrieval augmentation for multi-modal large language models. arXiv preprint arXiv:2405.10311 ,
2024.
[39] George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze,
Zhaoyan Liu, Anthony L Caterini, Eric Taylor, and Gabriel Loaiza-Ganem. Exposing flaws of
generative model evaluation metrics and their unfair treatment of diffusion models. Advances in
Neural Information Processing Systems , 36:3732–3784, 2023.
[40] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style.
arXiv preprint arXiv:1508.06576 , 2015.
[41] Asahi Ushio. Wikiart general dataset. https://huggingface.co/datasets/asahi417/
wikiart-all , 2024. Accessed: 2025-05-08.
[42] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár,
and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv
preprint arXiv:1504.00325 , 2015.
[43] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions
to visual denotations: New similarity metrics for semantic inference over event descriptions.
Transactions of the association for computational linguistics , 2:67–78, 2014.
[44] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011
dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.
[45] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-
scale long-tailed recognition in an open world. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition , pages 2537–2546, 2019.
[46] Scott Reed, Zeynep Akata, Honglak Lee, and Bernt Schiele. Learning deep representations of
fine-grained visual descriptions. In Proceedings of the IEEE conference on computer vision and
pattern recognition , pages 49–58, 2016.
12

A The Prompt Example to Decompose Queries into Subqueries
Decomposing Queries into Subqueries
Given an image caption, decompose the caption into an atomic entity. Each entity should
preserve descriptive details (e.g., size, color, material, location) together with the entity in a
natural, readable phrase. The entity should contain a noun and reserve noun modifiers in the
caption. Please ignore the entities like ‘a photo of’, ‘an image of’, ‘an overhead shot’, ‘the
window showing’ that are invisible in the image and ignore the entities like ’one’ and ’the
other’ that have duplicate entities before.
Caption: two cars are traveling on the road and waiting at the traffic light.
Entity: cars, road, traffic light
Caption: duplicate images of a girl with a blue tank top and black tennis skirt holding a tennis
racquet and swinging at a ball.
Entity: girl, blue tank top, black tennis skirt, tennis racqet, ball
Caption: the window showing a traffic signal is covered in droplets of rainwater.
Entity: traffic signal, droplets of rainwater
Caption: an overhead shot captures an intersection with a "go colts" sign.
Entity: intersection, "go colts" sign
Caption: a van with a face painted on its hood driving through street in china.
Entity: van, a face painted on its hood, street in china
Caption: two men, one with a black shirt and the other with a white shirt, are kicking each
other without making contact.
Entity: men, black shirt, white shirt
Caption: { caption }
Entity:
Figure 5: The Prompt Example for Decomposing User Queries into Subqueries on MS-COCO.
B Proof of the Time Complexity in Retrieval Efficiency
1. Proof of the time complexity for Algorithm 1
LetNbe the total number of images in D. We can score each image’s sparse textual match
inO(N). We discard images that do not satisfy any subquery, leaving a reduced set eD⊆D
of size eN.
We then discretize the simplex of subquery weights αintoKpossible combinations. Each
combination requires checkingP
iαisi(Ij)inO(eN)time , thus O(K×eN)in total.
Each adaptor pass handles both CLIP-based vision encoding (costing Tclipand the adaptor’s
own cross-attention (costing Tadaptor )). If we assume one pass per subquery set (of size n),
the total cost is eN×n×(Tclip+Tadaptor ). Multiplied by Kweight vectors, this yields
O
K×eN×n×(Tclip+Tadaptor )
.
Combining the steps above, total time is:
O(N) +O(K×eN) +O
K×eN×n×(Tclip+Tadaptor)
.
2. Proof of the time complexity for a pure sub-dimensional dense retriever
If we skip the sparse filter, we must embed all Nimages for each subquery. Thus, the pure
dense approach demands O(N×n×(Tclip+Tadaptor)).
13

Because eN≪NandKis small, this total is typically far lower than scanning all Nimages with the
sub-dimensional dense retriever.
C Proof of Algorithm Optimality
Because si(Ij)∈ {0,1}andP
iαi= 1,P
iαisi(Ij)lies in [0,1]. Since cos (vj,i, ti)∈[0,1],P
icos (vj,i, ti)≤n. Hence Cmax≤n.
Suppose Iadominates Ib. Then
∆sparse=X
iαisi(Ia)−X
iαisi(Ib)>0. (11)
Let
∆dense=X
icos (va,i, ti)−X
icos (vb,i, ti). (12)
We want ∆sparse+β∆dense>0. In the worst case for Ia,∆dense<0, potentially as low as −Cmax. A
sufficient condition for Iato stay preferred is
∆sparse−βCmax>0.
Because ∆sparse≥δminifIaindeed satisfies at least one more subquery and β >0is assumed by
definition, we obtain:
0< β <δmin
Cmax=βmax. (13)
We discretize the simplex {α:αi≥0,P
iαi= 1}. Because subqueries are strictly enumerated by
α, if an image satisfies a unique subquery set, it must appear as an arg maxP
iαisi(Ij)for some α.
Thus, no non-dominated s(Ij)is missed. ∆dense can be further used to find an optimal image among
those sharing the same s(Ij). Therefore, all Pareto-optimal solutions are obtained and s(Ij)in the
Pareto front Pfis unique.
D Datasets
For T2I-R on MS-COCO, we follow the Karpathy split using 82,783 training images, 5,000 validation
images, and 5,000 test images. For Flickr30K, we only use 1,000 images in the test set for evaluation.
Regarding T2I-G, WikiArt dataset is a comprehensive collection of fine art images sourced from
the WikiArt online encyclopedia. Our implementation is based on the version provided in [ 41]. To
construct the test set, we compare artwork titles across different images and identify pairs that differ
by at most three tokens. From each matched pair, we retain one sample, resulting in 2,619 distinct
test examples. The query for each test sample is formatted as: <title> in the style of <artistName>.
The retrieval database is composed of the remaining WikiArt images after excluding all test samples,
ensuring no overlap between ground-truth and retrieval candidates, as shown in Tab. 5. The Caltech-
UCSD Birds-200-2011 (CUB-200-2011) [ 44] is a widely used benchmark for fine-grained image
classification and generation tasks. It contains 11,788 images across 200 bird species. We use the
CUB dataset with 10 single-sentence visual descriptions per image collected by [ 46]. Similarly, to
construct the test set, we compare captions across different images and identify pairs that differ by
one token, resulting in 5,485 distinct test samples. The query for each test sample is formatted as:
Draw a <speciesName>. <caption>. For each test sample, the retrieval candidates consist of all
remaining images in the CUB dataset, excluding that test image. The ImageNet-LT dataset [ 45] is a
long-tailed version of the original ImageNet dataset. It contains 1,000 classes with 5 images per class.
We randomly choose one image from each class to construct the test samples. The retrieval database
is composed of the remaining ImageNet-LT images after excluding all test samples. The query for
each test sample is formatted as: A photo of <className>.
Table 5: Data construction of the T2I-G datasets
Dataset # of image in the dataset # of test samples # of images in the retrieval database
WikiArt 63,061 2,619 60,442
CUB 11,788 5,485 11,787
ImageNet-LT 50,000 1,000 49,000
14

E Limitations
While our multi-objective joint retrieval combining sub-dimensional sparse and dense retrievers is
efficient and achieves good granularity, when it comes to generation, the granularity of control is still
bounded by the capabilities of the underlying MLLM. If the underlying MLLM lacks the granularity
of control in generation, it becomes difficult to effectively leverage the subquery-level information
obtained from our retrieval method.
F More Visualizations of Our Method Compared with Other Baselines
More visualizations of our proposed method Cross-modal RAG compared with other baselines can
be found in Fig. 6 to 8. Across all three datasets, our method achieves superior image generation
capability. For the CUB in Fig. 6, our Cross-modal RAG can generate realistic images that align
with all subqueries, which are often ignored or distorted in GILL and UniRAG. Besides, SDXL,
LaVIT, and RDM tend to generate the sketch-like images rather than photo-realistic birds. On
the ImageNet-LT in Fig. 7, retrieving relevant images plays a crucial role in generating accurate
long-tailed objects. Our method successfully generates all three long-tailed objects with high visual
fidelity. In contrast, none of the baselines are able to generate all three correctly - UniRAG and GILL
even fail to produce a single accurate image. For WikiArt in Fig. 8, creative image generation poses a
unique challenge, as it is inherently difficult to reproduce the exact ground-truth image. However, our
method explicitly retrieve the images with satisfied subqueries and can capture the particular artistic
style specified in the query. As a result, all three generated images of Cross-modal RAG closely
resemble the style of the target artist in the query. In contrast, other RAG baselines can not guarantee
if the retrieved images grounded in the intended artist’s style. RDM even suffers from low visual
fidelity when generating human faces.
UserQueryPareto Optimal Images with  Satisfied SubqueriesRetrievedimage1Retrievedimage2Retrievedimage3Draw a Hooded Warbler. This bird has a blackcrown with black throat and yellow belly.
1.Hooded Warbler2.black crown4.yellow belly1.Hooded Warbler2.black crown3.black throat1.Hooded Warbler3.black throat4.yellow belly
Generation(Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation(GILL)
UserQueryPareto Optimal Images with  Satisfied SubqueriesRetrievedimage1Retrievedimage2Retrievedimage3Draw a White throated Sparrow. This bird has wings that are brownand has a white bellyand yellow crown.Generation(Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation(GILL)
UserQueryPareto Optimal Images with  Satisfied SubqueriesRetrievedimage1Retrievedimage2Retrievedimage3Generation(Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation(GILL)1.White throated Sparrow3.white belly4.yellow crown
1.White throated Sparrow2.Wingsthatarebrown3.Whitebelly
1.White throated Sparrow2.Wingsthatarebrown4.yellow crown
Draw a Vermilion Flycatcher. Asmall red bird with black wings and a small black beak.1.Vermilion Flycatcher 2.small red bird3.black wings
1.Vermilion Flycatcher 2.small red bird4.small black beak1.Vermilion Flycatcher 3.black wings4.small black beak
Figure 6: Visualizations on CUB compared with other baselines.
15

UserQueryPareto Optimal Images with  Satisfied Subqueries
Retrievedimage1A photo of [cardoon].Generation(Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation(GILL)
UserQueryPareto Optimal Images with  Satisfied Subqueries
Retrievedimage1A photo of [wire-haired fox terrier].Generation(Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation(GILL)
UserQueryPareto Optimal Images with  Satisfied Subqueries
Retrievedimage1A photo of [yurt].Generation(Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation(GILL)
[cardoon]
[wire-haired fox terrier]
[yurt]
Figure 7: Visualizations on ImageNet-LT compared with other baselines.
UserQueryPareto Optimal Images with  Satisfied SubqueriesRetrieved image1Retrievedimage2Retrievedimage3Fisherwomenon the Beach, Valenciain the style of Joaquín Sorolla.Generation(Ours)Generation (SDXL)Generation(LaVIT)Generation(RDM)Generation (UniRAG)Generation(GILL)
UserQueryPareto Optimal Images with  Satisfied SubqueriesRetrievedimage1Retrieved image2Retrievedimage3Roadnear Cagnesinthe style of Pierre-Auguste Renoir.Generation (Ours)Generation(SDXL)Generation(LaVIT)Generation(RDM)Generation(UniRAG)Generation (GILL)
UserQueryPareto Optimal Images with  Satisfied SubqueriesRetrieved image1Retrievedimage2Retrievedimage3Portraitof a Young Womanin the style of Alfred Stevens.Generation(Ours)Generation (SDXL)Generation(LaVIT)Generation(RDM)Generation (UniRAG)Generation(GILL)
2.Beach3.Valencia4.the style of Joaquín Sorolla
1.fisherwomen2.Beach1.Fisherwomen3.Valencia4.the style of Joaquín Sorolla
1. Road2. Cagnes2.Cagnes3.the style of Pierre-Auguste Renoir1.Road3.the style of Pierre-Auguste Renoir
1. Portrait 2. Young Woman  1.Portrait 3.the style of Alfred Stevens2.YoungWoman  3.the style of Alfred Stevens
Figure 8: Visualizations on WikiArt compared with other baselines.
16
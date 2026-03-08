# NICO-RAG: Multimodal Hypergraph Retrieval-Augmented Generation for Understanding the Nicotine Public Health Crisis

**Authors**: Manuel Serna-Aguilera, Raegan Anderes, Page Dobbs, Khoa Luu

**Published**: 2026-03-02 16:31:07

**PDF URL**: [https://arxiv.org/pdf/2603.02047v1](https://arxiv.org/pdf/2603.02047v1)

## Abstract
The nicotine addiction public health crisis continues to be pervasive. In this century alone, the tobacco industry has released and marketed new products in an aggressive effort to lure new and young customers for life. Such innovations and product development, namely flavored nicotine or tobacco such as nicotine pouches, have undone years of anti-tobacco campaign work. Past work is limited both in scope and in its ability to connect large-scale data points. Thus, we introduce the Nicotine Innovation Counter-Offensive (NICO) Dataset to provide public health researchers with over 200,000 multimodal samples, including images and text descriptions, on 55 tobacco and nicotine product brands. In addition, to provide public health researchers with factual connections across a large-scale dataset, we propose NICO-RAG, a retrieval-augmented generation (RAG) framework that can retrieve image features without incurring the high-cost of language models, as well as the added cost of processing image tokens with large-scale datasets such as NICO. At construction time, NICO-RAG organizes image- and text-extracted entities and relations into hypergraphs to produce as factual responses as possible. This joint multimodal knowledge representation enables NICO-RAG to retrieve images for query answering not only by visual similarity but also by the semantic similarity of image descriptions. Experimentals show that without needing to process additional tokens from images for over 100 questions, NICO-RAG performs comparably to the state-of-the-art RAG method adapted for images.

## Full Text


<!-- PDF content starts -->

NICO-RAG: Multimodal Hypergraph
Retrieval-Augmented Generation for
Understanding the Nicotine Public Health Crisis
Manuel Serna-Aguilera1, Raegan Anderes1, Page Daniel Dobbs2, and Khoa
Luu1
1University of Arkansas, Fayetteville, AR, 72701, USA
{mserna, rmandere, khoaluu}@uark.edu
2University of Arkansas for Medical Sciences, Little Rock, AR, 72205, USA
PDDobbs@uams.edu
Abstract.The nicotine addiction public health crisis continues to be
pervasive. In this century alone, the tobacco industry has released and
marketed new products in an aggressive effort to lure new and young
customers for life. Such innovations and product development, namely
flavored nicotine or tobacco such as nicotine pouches, have undone years
of anti-tobacco campaign work. Past work is limited both in scope and
in its ability to connect large-scale data points. Thus, we introduce
theNicotineInnovationCounter-Offensive (NICO) Dataset to provide
public health researchers with over 200,000 multimodal samples, includ-
ing images and text descriptions, on 55 tobacco and nicotine product
brands. In addition, to provide public health researchers with factual
connections across a large-scale dataset, we proposeNICO-RAG, a
retrieval-augmented generation (RAG) framework that can retrieve im-
age features without incurring the high-cost of language models, as well
as the added cost of processing image tokens with large-scale datasets
such as NICO. At construction time, NICO-RAG organizes image- and
text-extracted entities and relations into hypergraphs to produce as fac-
tual responses as possible. This joint multimodal knowledge represen-
tation enables NICO-RAG to retrieve images for query answering not
only by visual similarity but also by the semantic similarity of image
descriptions. Experimentals show that without needing to process addi-
tional tokens from images for over 100 questions, NICO-RAG performs
comparably to the state-of-the-art RAG method adapted for images.
Keywords:Retrieval-AugmentedGeneration·Hypergraph·Multimodal
·Vision-Language Learning·Nicotine·Public Health.
1 Introduction
In recent times, the tobacco industry has rapidly innovated in nicotine delivery
and is releasing many new products on a massive scale. By leveraging legal
loopholes,lobbying,courtbattles,slowlegalproceedings,andalackofawareness,arXiv:2603.02047v1  [cs.CV]  2 Mar 2026

2 Authors Suppressed Due to Excessive Length
Fig. 1.Samples of our NICO Dataset. (Left) Nicotine pouch samples with (a) mint-
based flavors, (b) fruit flavors, and (c) spice and coffee flavors. (Right) Other tobacco
and nicotine products with flavorings.
tobacco companies can aggressively market new nicotine products, particularly
towards youth, with little repercussions. Consequently, this rapid mass release
of new, increasingly popular nicotine pouch products makes it very difficult
for public health researchers and policy makers to address these innovations
and marketing effectively. Traditionally, researchers have had tomanuallysort,
organize, and analyze large-scale datasets. This slow manual processing prolongs
and exacerbates the nicotine addiction public health crisis. Therefore, there is
a critical need to process large-scale data, e.g., images and documents, and to
connect data points more quickly than current manual methods.
Current research that seeks to understand tobacco and nicotine products is
rather limited in scope and capacity. Datasets from works such as Vassey et
al. [12] analyze 6,999 Instagram images labeled for e-cigarette-related objects
for product detection using dynamic head attention [4]. Murthy et al. [8] anno-
tated 826 frames from TikTok videos and detected e-cigarette device use with a
YOLOv7 detector [13]. In larger-scale work, Chappa et al. [3] perform product
classification on various types of products using video frames from TikTok and
YouTube. Despite this progress, there remain gaps between what the AI side
can deliver and what the public health side needs. To our knowledge, no dataset
or methodology captures key relationships within large-scale data to enable in-
formed, factual retrieval of vital product information, e.g., brand identifiers,
distinctive visual features, flavors, advertising strategies, etc., for both the same
product type and new and upcoming products.
Contributions of this Work:We therefore address two large-scale prob-
lems, i.e., the need for a large dataset to build a unified knowledge base and,
consequently, a methodology to leverage the connections between data points
implicit in such a dataset. In this work, we make three main contributions. We
first introduce the large-scale Nicotine Innovation Counter–Offensive (NICO)
Dataset, comprising over 200,000 images of several product types across to-
bacco and nicotine product brands, with samples shown in Figure 1. Second, we
propose the Nicotine Innovation Counter-Offensive Retrieval-Augmented Gen-
eration (NICO-RAG), a novel multi-modal and multi-feature framework for

Title Suppressed Due to Excessive Length 3
Table 1.A comparison of the past dataset with our proposed dataset. Our dataset
contains 55 named tobacco or nicotine brands, 47 more than the PHAD [3]. We also
contribute not only images and labels but also textual, color, and shape descriptions.
Finally, our dataset contains more diverse images than previous works.
DatasetProduct
BrandsText
DescriptionsColor
DescriptionsShapeNumber of
Image Samples
Murthy et al. [8] 3✗ ✗ ✗826
Vassey et al. [12] 7✗ ✗ ✗6,999
PHAD [3] 8✗ ✗ ✗171,900
NICO (Ours) 55✓ ✓ ✓202,599
retrieval-augmentedgeneration(RAG)inpublichealthscenarios.WebuiltNICO-
RAG to handle combined text-and-image queries and not rely solely on image
embeddings, but instead on an enriched combination of multiple descriptors of
nicotine products—visual features, text embeddings, words on product pack-
aging, color, morphology, etc. Finally, these simple product descriptors enable
more holistic and diverse knowledge construction and retrieval processes while
avoiding reliance on large, expensive language models for large-scale work. It en-
ables public health researchers to more accurately link products across multiple
criteria at scale in a fraction of the time.
2 The Nicotine Innovation Counter-Offensive Dataset
The Nicotine Innovation Counter-Offensive Dataset is the first of our two main
contributions to advancing monitoring and understanding of tobacco and nico-
tine product innovation. With over 200,000 samples, it represents the largest
and most diverse dataset of tobacco and nicotine products assembled. NICO
comprises images and natural-language descriptors—coloring, text on packaging,
simple color descriptions, product shape, and a simple description. Our dataset
contributions are summarized in Table 1. In contrast, prior work by Vassey et
al. [12] and Murthy et al. [8] analyzes 6,999 and 826 e-cigarette images, respec-
tively. The PHAD [3] video dataset comprises 171,900 video frames across 8
brands, with no product descriptors and a smaller label set.
Image Data Collection.A significant portion of the images in NICO was
collected through a rigorous process of sampling product images from various
online resources and product catalogs. The image collection involved a combina-
tion of automated and manual processes. We first collected data using the Apify
platform [1], which enabled query-based categorization. This categorization ini-
tially assigned labels for tobacco type, product type, and brand. In addition,
we filtered out as many irrelevant samples as possible to ensure quality control,
yielding a large-scale image set of products, sorted by tobacco type, product
type, and brand hierarchical labels, which we use to organize every sample.
Data Preparation.With the images collected, we prepared our image mod-
ules to provide natural-language information about the products (if any) in the

4 Authors Suppressed Due to Excessive Length
Fig. 2.The NICO-RAG framework. We take in a query image and text, and via text
and image entity discovery, we create the multimodal hypergraph knowledgeK, giving
us image descriptors to capture all aspects desirable for public health in tobacco and
nicotine prevention research.
images, without relying solely on Large Language Models (LLMs) or Large Mul-
timodal Models (LMMs). We focus on the following image descriptors: natural
language, optical character recognition (OCR), color, and shape. Further details
for descriptor extraction formulation and implementations are provided in Sec-
tions 3 and 4.1. Additionally, in collaboration with our team of public health
researchers, we hand-crafted questions on hot topics in nicotine pouch product
research.Thereare11uniquequestionscoveringtopicssuchastheflavorsbrands
offer, the relationship between flavors and colors, the relationship between cur-
rent pouch flavors and other product types, etc. Further details on experimental
settings are given in Section 4.1.
NICO Dataset Statistics.TheNICOdatasetcontainsatotalof202,599image
samples spanning 55 tobacco or nicotine products. The images span five product
types that have undergone or are undergoing innovation, i.e., cigarettes, heated
tobacco, e-cigarettes, smokeless tobacco, and nicotine pouches. Our dataset con-
tains 50,882 cigarette images (13 brands), 48,261 heated tobacco images (14
brands), 61,243 e-cigarette images (11 brands), and 42,213 smokeless tobacco
and nicotine pouch images (17 brands).
3 Nicotine Innovation Counter-Offensive Retrieval
Augmented Generation (NICO-RAG) Approach
3.1 Preliminaries
We first provide preliminary definitions for NICO-RAG, following those in pre-
vious work [6]. The multimodal knowledge is denoted byKand is organized

Title Suppressed Due to Excessive Length 5
into text chunks (K chunk)—a graph or hypergraph representation. In a graph or
hypergraph setting,Kis defined by entities and edges in Eqn. (1).
K=
V(M), E(M)
H
=
V(I)∪V(T), E(I)∪E(T)
(1)
The setV(M)contains all images (I) and associated data (T, e.g., text de-
scriptions) inK, andv j∈Vis an entity containing relevant image-derived
information and corresponds to imageI j∈I. The setE(M)=E(I)∪E(T)con-
tains the relations (simple edges or hyperedges) from image and text-derived
information as inV(M). At retrieval time, we use our multimodal construction
ofKto retrieve the optimal subgraphK∗
qwith respect to a multimodal query
q= (I q, Tq), whereI qis an image andT qthe query text.
3.2 Multimodal Knowledge Construction
We can defineV(M)by decomposing it into image (I) and text (T) components,
as in Eqn. (2).
V(I)=ρV(I), V(T)=ϕV(π, T, p ext)(2)
The functionρis composed of image analysis modules, each returning different
aspects of a particularI j, e.g., embeddings, shape, color, detection, optical char-
acter recognition (OCR), etc. These qualities were chosen for their relevance in
tobacco product analysis. For instance, color provides clues to flavors, a highly
valuable data point. Legible words on product packaging can provide clues about
the contents of packages, e.g., promotions/rewards. It allows formulti-pronged
knowledge and retrieval. The functionϕ Vtakes care of extracting text descrip-
tors for eachI jgiven a LLM or LMMπ, textT, and extraction promptp extto
identify all entities withinT. Thus, we have text information by which we can
properly build a multimodal RAG framework.
The multimodal relationsE(M)are similarly defined as the entity extrac-
tion in Equation 3. It relates entities to natural-language descriptions, e.g., text
describing an image, and to other features such as color and shape.
E(I)=ρE(I), E(T)
H=ϕE(π, T, p ext)(3)
Forρ V,wedecomposeitintoseparateextractionfunctionsρ V(I) =∪ j=1∪l=1
(λl(Ij))whereλ lis a function that extracts one type of feature. Consequently,
we define image-feature relations asρ E(I) =∪ j=1∪l=1(Ij, λl(Ij)). In practice,
ϕuses the LMMπto extract the text entities and relations in a typical fashion;
queryingπfor allI jwould be prohibitively costly forϕ. Thus, our multimodal
knowledge construction, broken down, is defined in Eqn. (4).
K=
V(M), E(M)
H
=
ρV(I)∪ϕ V(π, T, p ext), ρE(I)∪ϕ E(π, T, p ext)
(4)

6 Authors Suppressed Due to Excessive Length
Fig. 3.Responses from NICO-RAG and a multimodal Hypergraph RAG [6] for two
nicotine pouch products, where a complex search for inter- and intra-product informa-
tion on image attributes is needed. Best viewed in zoom and in color. Green highlights
denote correct descriptions, while red denotes lower-quality or incorrect descriptions.
3.3 Multimodal Knowledge Retrieval for Question Answering
With our multi-pronged architecture, we can process multimodal queries ac-
cording to different criteria, leveraging features that address public health needs
(rather than merely text or image features). The image descriptors fromϕare
combined within the functionM. Our retrieval formulation is given in Eqn. (5).
K∗
q=M{v∈V(M)∪ρV(Iq), e∈E(M)∪ρE(Iq)|q, I q} ∪K chunk (5)
To grab entities fromKat query time, we perform top-kentity and relation
matching.Forimages,weretrievetop-kmatchesbasedon:(i)imageembeddings;
(ii) the image description; (iii) average color similarity; (iv) the shape description
of the object; and (v) OCR contents, i.e., the similarity of text that is present
in the image. Finally,π’s final response to the user comes in the formy∗=
π(q|p gen, K∗
q), wherep genis our response generation prompt.
4 Experiments and Results
4.1 Experimental Design
Implementation.We implement all our code using PyTorch in Python. As in
prior work, we use GPT-4o-mini [9] forπdue to its low cost; to process text
withinK, we usetext-embedding-3-small. Forρ’s extraction functionsλ l, we
use CLIP ViT-14 [11] for image embeddings, DocTR [7] for OCR, and Qwen3-
VL (4B model) [2] for image descriptions. Top-kretrieval is computed with

Title Suppressed Due to Excessive Length 7
cosine similarity. The image analysis modules were run on distributed computing
servers equipped with GPUs ranging from Quadro RTX 8000 to A100.
Test Data.Our experiments focus on question-answering centered around nico-
tine pouch products. We sample 9 images from Zyn, 8 from Velo, and 5 from
Klint, and 7 non-nicotine pouch products to simulate answering with existing
knowledge. For unseen brands, we exclude all images collected under the Griz-
zly and Goat brands fromK, which produce smokeless tobacco but have now
released pouch products. Furthermore, to mitigate the impact of noisy or misla-
beled images, we manually removed any pouch product images for Grizzly and
Goat that appeared under other brands, selecting the top 512 most similar sam-
ples. This amounts to approximately 3,700 images removed and 108 images in
our test set. In total,πwill return 197 responses per experiment.
RAG Methods.We use several RAG backbonesπ, adapted for the multi-
modal setting.Naive generation: We simply queryπwithqand return its
response as the answer.Standard multimodal RAG: In this case, we retrieve
the top-kimages, giving us corresponding image descriptions, which in prac-
tice are computed offline.Naive multimodal HyperGraphRAG: An adap-
tation of HypergraphRAG [6] where image descriptions are concatenated to-
gether and the entities and relations extracted, with images tied to correspond-
ing chunks. We chose the hypergraph representation because it provides a strong
natural-language representation of entities and relations in text-modality prob-
lems.NICO-RAG: Our proposed NICO-RAG as discussed in Section 3. We
use all image analysis modules for multi-image feature retrieval rather than just
embeddings. To connect image entities, we use the hypergraph structure from
HyperGraphRAG for text chunks, and we tie all image descriptors toK.
Experiments.In the first set of experiments, to measure response and golden
answer word-level similarity, we use the F1 score. To measure semantic similar-
ity, we follow [5,6] and use retrieval similarity (RS). To assess generation quality
across multiple aspects, we score responses using an LLM as the judge [6,10].
Evaluation ranges from inter-product questions (centered on pouches, with com-
parisons to non-pouch products) to intra-pouch- brand questions. In the second
set of experiments, we first conduct an ablation study on the different image
analysis modules by omitting certainλ lusingk= 4ork= 8top image descrip-
tions for eachλ l. For our experiments,λ 1gives color descriptions,λ 2gives shape
descriptions,λ 3gives OCR predictions, andλ 4gives image descriptions.
4.2 Experimental Results and Discussion
Table 2 shows the results of the generation evaluation across different RAG
methods. The multimodal HypergraphRAG, with three calls toπ(including two
that useI q), achieves similar performance to our NICO-RAG, which removes
one call toπentirely in favor ofλ l, and only one call involvesI q. This shows
that, in large-scale scenarios, we need not rely on the expressive power ofπto
obtain accurate image-level information. Meanwhile, Table 3 shows us the effects
of providing the top-ksample descriptions from image retrieval.λ 1returns color

8 Authors Suppressed Due to Excessive Length
Table 2.Multimodal response evaluation ofπ’s responses given different multimodal
RAG methods. Our results are inbold.
RAG Method F1 RS GE
Naive Generation 0.315 0.786 0.456
Standard RAG 0.295 0.807 0.473
Hypergraph RAG 0.283 0.789 0.467
NICO-RAG(Ours) 0.273 0.800 0.466
Table 3.Ablation study on NICO-RAG, where we showcase the contribution of each
of the image analysisλfunctions.
Includedk= 8k= 4
Functions F1 RS GE F1 RS GE
{λ1}0.261 0.784 0.438 0.253 0.764 0.411
{λ1, λ2}0.263 0.784 0.433 0.250 0.769 0.413
{λ1, λ2, λ3}0.269 0.784 0.446 0.257 0.782 0.432
{λ1, λ2, λ3, λ4}0.273 0.800 0.466 0.261 0.792 0.448
descriptors,λ 2returns shape descriptors,λ 3returns OCR terms, andλ 4returns
an image description of the correspondingI q.
5 Conclusion
In this work, we introduced the NICO dataset, a collection of nicotine and to-
bacco product samples of over 200,000 samples. To the best of our knowledge,
it is the largest and most diverse dataset of its kind. We also presented NICO-
RAG, a RAG framework that leverages diverse image descriptors to construct
a multimodal hypergraph-based knowledge base while reducing dependency on
heavy LMMs. From our experiments with NICO-RAG and the adaptation of the
state-of-the-art text RAG method, we observe comparable performance without
addinganadditionalLMMcallperquery.Atthesametime,NICO-RAGcanalso
return factual responses even with the removal of image-based entity extraction
fromπ. Despite its performance, NICO-RAG has avenues of improvement. The
heavierλ lfunctions, such as OCR and descriptions via Qwen3-VL, still require
relatively expensive non-consumer grade hardware to run at the scale of NICO.
We argue, however, that large models such asπcannot be run locally and require
monetary resources, unlike our proposedλ lfunctions. Furthermore, construct-
ingKfrom image and text descriptions can be prohibitively time-consuming
and costly if we rely solely onπto perform these tasks, necessitating the manual
insertion of relations and entities, as in our method (i.e., the image modules).
Despite our efforts, the NICO dataset contains images that are not relevant to
tobacco or nicotine product research, i.e., noisy samples. We argue that this is
mitigated by our multiple retrieval criteria per image, in which irrelevant im-

Title Suppressed Due to Excessive Length 9
ages are weighted less than relevant samples. This is especially the case with
OCR and image descriptions. We aim to foster further work to harmonize large-
scale datasets with the expressive power of RAG frameworks, with the goal of
resolving public health crises and yielding results more quickly.
References
1. Apify: Apify technologies, copyright @2024 (2024), http://apify.com
2. Bai, S., Cai, Y., Chen, R., et al., K.C.: Qwen3-vl technical report. arXiv preprint
arXiv:2511.21631 (2025)
3. Chappa, N.V., McCormick, C., Gongora, S.R., Dobbs, P.D., Luu, K.: Public health
advocacy dataset: A dataset of tobacco usage videos from social media. arXiv
preprint arXiv:2411.13572 (2024)
4. Dai, X., Chen, Y., Xiao, B., Chen, D., Liu, M., Yuan, L., Zhang, L.: Dynamic head:
Unifying object detection heads with attentions. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 7373–7382 (2021)
5. Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: Ragas: Automated evaluation
of retrieval augmented generation (2025), https://arxiv.org/abs/2309.15217
6. Luo, H., E, H., Chen, G., Zheng, Y., Wu, X., Guo, Y., Lin, Q., Feng,
Y., Kuang, Z., Song, M., Zhu, Y., Tuan, L.A.: Hypergraphrag: Retrieval-
augmented generation via hypergraph-structured knowledge representation (2025),
https://arxiv.org/abs/2503.21322
7. Mindee: doctr: Document text recognition. https://github.com/mindee/doctr
(2021)
8. Murthy, D., Ouellette, R.R., Anand, T., Radhakrishnan, S., Mohan, N.C., Lee,
J., Kong, G.: Using computer vision to detect e-cigarette content in tiktok videos.
Nicotine and Tobacco Research26(Supplement_1), S36–S42 (2024)
9. OpenAI, Achiam, J., Adler, S., et al., S.A.: Gpt-4 technical report (2024),
https://arxiv.org/abs/2303.08774
10. Que, H., Duan, F., He, L., Mou, Y., Zhou, W., Liu, J., Rong, W., Wang, Z.M.,
Yang, J., Zhang, G., et al.: Hellobench: Evaluating long text generation capabilities
of large language models. arXiv preprint arXiv:2409.16191 (2024)
11. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S.,
Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.:
Learning transferable visual models from natural language supervision (2021),
https://arxiv.org/abs/2103.00020
12. Vassey, J., Kennedy, C.J., Herbert Chang, H.C., Smith, A.S., Unger, J.B.: Scalable
surveillance of e-cigarette products on instagram and tiktok using computer vision.
Nicotine and Tobacco Research26(5), 552–560 (2024)
13. Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M.: Yolov7: Trainable bag-of-freebies
sets new state-of-the-art for real-time object detectors. In: Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. pp. 7464–7475
(2023)
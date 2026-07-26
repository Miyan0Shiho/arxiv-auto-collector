# Enhancing Relation Modeling with Social Attributes for Social Media Popularity Prediction

**Authors**: Bolun Zheng, Yuhao Luo, Wei Zhu, Ning Xu, An-An Liu, Lingyu Zhu, Canjin Wang

**Published**: 2026-07-21 15:34:11

**PDF URL**: [https://arxiv.org/pdf/2607.19200v1](https://arxiv.org/pdf/2607.19200v1)

## Abstract
Recent studies highlight the critical role of retrieval-augmented mechanisms in social media popularity prediction (SMPP). Although such frameworks have improved SMPP performance by leveraging historical posts, existing methods still suffer from the low retrieval accuracy due to the oversight of relative relationships among UGC instances. To address this limitation, we propose a novel Relation-Enhanced Retrieval-Augmented framework (RE-Rag) that models UGC similarity as a continuous relation jointly driven by semantic content and social attributes. Specifically, RE-Rag employs a Semantic-Attribute Retriever (SAR) to obtain instances aligned in both semantic and social-attribute distributions. Subsequently, we design a Relation-Guided Predictor (RGP): first, cross-attention encodes multimodal features of retrieved instances; then, a relative relation graph is introduced to guide attention weight allocation, forming a Relation-Guided Transformer (RGTs) that dynamically modulate attention weights based on relative attribute relations to capture the interplay between semantics and various social attributes. The refined features are fused with the target instance for popularity prediction. Experiments on three public benchmarks show that RE-Rag consistently outperforms state-of-the-art methods in both prediction accuracy and retrieval efficiency.

## Full Text


<!-- PDF content starts -->

Enhancing Relation Modeling with Social Attributes for Social
Media Popularity Prediction
Bolun Zheng
Yuhao Luo
Wei Zhu
Hangzhou Dianzi University
Hangzhou, China
blzheng@hdu.edu.cn
242060267@hdu.edu.cn
242050252@hdu.edu.cnNing Xu
Anan Liu
Tianjin University
Tianjin, China
ningxu@tju.edu.cn
anan0422@gmail.com
Lingyu Zhu
City University of Hong Kong
Hong Kong, China
lingyzhu-c@my.cityu.edu.hkCanjin Wang
Xinhua Zhiyun Technology Co., Ltd.
Hong Kong, China
CanjinWang@shuwen.com
Abstract
Recent studies highlight the critical role of retrieval-augmented
mechanisms in social media popularity prediction (SMPP). Although
such frameworks have improved SMPP performance by leveraging
historical posts, existing methods still suffer from the low retrieval
accuracy due to the oversight of relative relationships among UGC
instances. To address this limitation, we propose a novel Relation-
Enhanced Retrieval-Augmented framework (RE-Rag) that models
UGC similarity as a continuous relation jointly driven by seman-
tic content and social attributes. Specifically, RE-Rag employs a
Semantic-Attribute Retriever (SAR) to obtain instances aligned in
both semantic and social-attribute distributions. Subsequently, we
design a Relation-Guided Predictor (RGP): first, cross-attention en-
codes multimodal features of retrieved instances; then, a relative
relation graph is introduced to guide attention weight allocation,
forming a Relation-Guided Transformer (RGTs) that dynamically
modulate attention weights based on relative attribute relations
to capture the interplay between semantics and various social at-
tributes. The refined features are fused with the target instance for
popularity prediction. Experiments on three public benchmarks
show that RE-Rag consistently outperforms state-of-the-art meth-
ods in both prediction accuracy and retrieval efficiency. The code
and data are available at https://github.com/BBEC-opt/RE_RAG.
CCS Concepts
•Information systems →Multimedia information systems.;
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXKeywords
Multimedia popularity, retrieval augmentation
ACM Reference Format:
Bolun Zheng, Yuhao Luo, Wei Zhu, Ning Xu, Anan Liu, Lingyu Zhu, and Can-
jin Wang. 2026. Enhancing Relation Modeling with Social Attributes for
Social Media Popularity Prediction. InProceedings of Make sure to enter
the correct conference title from your rights confirmation email (Confer-
ence acronym ’XX).ACM, New York, NY, USA, 10 pages. https://doi.org/
XXXXXXX.XXXXXXX
1 Introduction
Social media platforms have enjoyed effective growth over the past
decade. Every day, hundreds of millions of users create information
on various social media platforms. Predicting potential popularity
of user-generated content (UGC) before its distribution is essential
for platforms to optimize resource allocation [ 2,21], improve ad-
vertising strategies [ 33,46], refine marketing approaches [ 25,30],
develop effective policies [ 23,35], and content ecosystem gover-
nance [ 32,39]. Due to the high potential commercial value, the
social media popularity prediction (SMPP) has emerged as a critical
research topic and attracts enormous academic attention [ 26]. Early
approaches (Figure 1(a)) extract diverse features from content, meta-
data, and social interactions, then use statistical models or neural
networks to directly predict popularity scores [ 4,9,15,29]. How-
ever, such models typically treated each post in isolation and relied
on limited contextual information, which hindered their ability to
capture underlying user behavior trends and patterns of informa-
tion diffusion. To address this limitation, recent studies (Figure 1(b))
introduce retrieval-enhanced frameworks by leveraging similar his-
torical posts as auxiliary features for popularity prediction [ 13,53].
Despite their effectiveness, existing retrieval-enhanced methods
often equate semantic similarity with instance relevance, overlook-
ing the fact that semantically similar UGCs may exhibit different
popularity patterns. Such discrepancies arise because similar con-
tent can be associated with different social contexts and propagation
processes, leading to diverse popularity outcomes. Even highly simi-
lar posts may be embedded in very different social contexts, exposed
arXiv:2607.19200v1  [cs.MM]  21 Jul 2026

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
UGC Predictor
UGC Predictor
Retrieval feature
Extractor
UGC
Semantic Content Attribute
Instances 
Relations    Popularity
    Popularity
   Popularity
   Popularity
    Popularity
    Popularity
InstancesInstancesInstancesInstancesInstancesInstances(a)Prior work
(b)Vanilla Retrieval
(c)OurMemory 
Bank
Memory 
Bank
Retrieval feature
ExtractorMemory 
Bank
Memory 
Bank
InstancesInstancesInstancesInstancesInstancesInstancesPredictorSemantic Content
Figure 1: Comparison among (a) non-retrieval models, (b)
vanilla retrieval models, and (c) our proposed model.
to different user groups, and activated through distinct diffusion
pathways, leading to drastically different popularity outcomes[5].
Beyond semantic-based retrieval, some studies [ 7,48] utilize
social attributes to enhance the RAG. However, they mainly treat
social attributes as auxiliary cues for similarity of contents, without
explicitly considering their role as indicators of the underlying
distribution mechanism. Thus, they struggle to capture the subtle
yet crucial distinctions among UGCs.
In this work, we rethink the mechanism of social media com-
munication and propose a dual-factor model, where social media
popularity is governed by two independent factors–content quality
and distribution mechanism. The former reflects how engaging the
content itself is, while the latter characterizes how many users the
platform exposes the content to, determined by social attributes
(author, category, language, etc.).
Based on this perspective, we propose a relation-enhanced re-
trieval augmentation framework (RE-Rag) that model relative at-
tribute relations among UGC instances for effective instance re-
trieval and accurate popularity prediction. Specifically, we first
introduce a Semantic-Attribute Retriever (SAR) to retrieve valu-
able UGC instances from the database basing on semantic cues
and attribute-based matching score, and build a relation map of
the target UGC and corresponding retrieved instances. Then two
relation-guided transformers (RGTs) are involved, constructing a
relation-guided predictor (RGP) to predict popularity score using
the target UGC and retrieved instances with the assist of the rela-
tion map obtained in SAR. Generally, our major contributions can
be summarized as follows:
•We propose a relation-enhanced retrieval augmentation frame-
work namely RE-Rag, that models UGC similarity as a continuous
relation jointly driven by semantic content and social attributes to
enhance popularity prediction.
•We design two core modules, semantic-attribute retriever and
relation-guided predictor, against distinguishing valuable UGCsbased on relation modeling driven by social attributes for retrieval
augmentation.
•Extensive experiments on three public benchmarks demonstrate
that the proposed relation modeling strategy is effective in retrieval
stage, and our RE-Rag achieves state-of-the-art performance with
better retrieval efficiency.
2 RELATED WORK
2.1 Popularity Prediction
Social Media Popularity Prediction aims to predict the popularity
of UGC based on multimodal information and has shown practical
value across domains. Early approaches relied on predefined fea-
tures and constructed predictive rules using statistical methods such
as linear regression [ 37], but these methods struggled to capture
complex nonlinear relationships. Subsequently, researchers com-
bined handcrafted features (e.g., content attributes, user profiles)
with machine learning models [ 12,16,18,24] to characterize diffu-
sion patterns. However, such methods remain sensitive to feature
quality, required extensive domain expertise, and lack scalability.
In recent years, deep neural networks have driven the develop-
ment of end-to-end representation learning, enabling the automatic
discovery of complex patterns from data and offering stronger fea-
ture modeling and scalability in SMPP tasks. For example, VSCNN
[1] employs convolutional neural networks to jointly model visual
content and social context features, DTCN [ 44] introduces temporal
coherence by learning continuous temporal context for predicting
popularity dynamics, CBAN [ 8] enhances multimodal feature inter-
action and expressiveness through bipolar attention, MASSL [ 52]
proposes a multimodal prediction framework that combines feature
extraction strategies with variational autoencoders, HMMVED [ 47]
adopts a hierarchical multimodal variational encoding–decoding
framework to capture cross-modal semantic structures, Liao et al.
[22] integrate article content with temporal features via deep fusion
and hierarchical feature extraction strategies to improve prediction
performance, and TGANN [ 31] employs text-guided attention for
multimodal fusion.
Nevertheless, most existing approaches focus on single-instance
popularity prediction, relying on limited modal cues. This hinders
capturing broader semantic associations and diffusion patterns,
thus limiting performance in dynamic social media environments.
2.2 Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) is a paradigm that integrates
neural network models with external retrieval modules, enabling
the model to dynamically acquire and incorporate relevant infor-
mation from large-scale knowledge bases, thereby improving both
prediction and generation performance [ 11,20]. This paradigm has
been widely applied in domains such as large language models
[3,19], recommender systems [ 51], and social network analysis
[36]. Its effectiveness has also been validated in the task of SMPP.
For example, AFRF [ 42] retrieves similar popularity time series
through angular features, while MMRA and NIPA [ 13,53] leverage
multimodal semantic content retrieval to obtain similar instances
and extract multimodal features for popularity prediction. However,
these approaches often overlook social relation information, which

Enhancing Relation Modeling with Social Attributes for Social Media Popularity Prediction Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Text 
EmbeddingImage 
Embedding
Text 
SimilarityImage 
Similarity
Retrieved Instances
QKVReference 
Sequence
Linear 
LayerQuery
Sequence
Softmax
Linear Layer
 OutputMatching
(a)Semantic-Attribute Retriever (b)Relation-Guided Predictor (c)Relation-Guided Transformer
So relaxed 
#cat ...Input  Target  UGC
Text Image
Semantic Encoder
MLPc-RGTTargetCategoty:17
AttributeUid:230
others...
popularityAttribute
Score
DatasetLinear 
Layer
EmbedMLP
s-RGT
Add&NormRelation
MapRelation Map weight matrix
calculate
weight(d)Relation Weight Computation 
.....
.....
Target/Retrievedrelation 
constraction......
Intra-
Relation 
Map
Cross-
Relation
Map
Relation Map
Figure 2: Overview of the proposed RE-Rag framework: (a) Semantic-Attribute Retriever combines multimodal semantic
similarity with attribute compensation to select relevant samples. (b) Relation-Guided Predictor, comprising a Semantic Encoder
and Relation-Guided Transformer, extracts retrieved sequence features for prediction. (c) Relation-Guided Transformer embeds
relative relation patterns into attention to highlight critical features. (d) Relation Weight Computation Details.
may lead to spurious correlations between retrieved and target
instances, thereby limiting the quality of the extracted knowledge.
Recently, RAGTrans [ 7] clusters UGC contents into virtual at-
tributes and aggregates them with social attributes through a hy-
pergraph [ 14], while SKAPP [ 48] enhances retrieval with BM25
algorithm and Selective Refiner. Although both leverage social at-
tributes, they regard them as auxiliary cues for sample retrieval.
RAGTrans further incorporates social attributes into hypergraph
modeling, but the predefined aggregation only captures coarse
associations and fails to characterize the underlying distribution
mechanism. Therefore, we argue that social attributes should serve
as proxy variables for modeling platform distribution mechanisms
rather than auxiliary features for content similarity measurement.
3 METHODOLOGY
In this section, we present the proposed RE-Rag framework (Fig-
ure 2). Given an input UGC instance, a Semantic-Attribute Re-
triever first identifies relevant references. These instances are then
processed by a retrieval feature extractor, comprising a semantic
encoder and a relation-guided transformer, to derive informative
representations for popularity prediction. Detailed descriptions of
each component are provided in the following subsections.3.1 Problem Definition
The goal of SMPP is to predict the popularity of UGCs using their
content and social attributes before they actually communicate on
the platform. Generally, we adopt normalized view counts as the
popularity [49], which can be written as:
𝑝=log2𝑣
𝑑+1
,(1)
where𝑝denotes UGC’s popularity score, 𝑣denotes its view count,
and𝑑denotes the number of days since its communication. This
definition is designed to eliminate the influence of time span on view
growth, thereby providing a fairer assessment of UGC popularity
on a unified time scale.
3.2 Semantic-Attribute Retriever
In this section, we propose a Semantic-Attribute Retriever, aiming
at retrieving instances with good consistency of popularity. The
UGC’s popularity is affected by both content semantics and social
attributes. Therefore, the proposed SAR involves both of them to
construct the database and multi-modal retriever.
3.2.1Database Construction.We constructed a multimodal re-
trieval database to retrieve UGC instances with similar popular-
ity. Each UGC entry is represented as < 𝑐𝑜𝑛𝑡𝑒𝑛𝑡 ,𝑎𝑡𝑡𝑟𝑖𝑏𝑢𝑡𝑒 ,𝑙𝑎𝑏𝑒𝑙 >.
Specifically, 𝑐𝑜𝑛𝑡𝑒𝑛𝑡=[𝒇𝑡𝑒𝑥𝑡,𝒇𝐼𝑚𝑔]denotes vector representations

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
derived from text features 𝑓𝑡𝑒𝑥𝑡and image features 𝒇𝐼𝑚𝑔through
pre-trained models. This label indicates the corresponding pop-
ularity. This 𝑎𝑡𝑡𝑟𝑖𝑏𝑢𝑡𝑒 represents the social attributes of a UGC,
including the category, author information, and language. Detailed
definitions of social attributes across different datasets are provided
in the supplementary material.
3.2.2Multimodal Retrieval.To retrieve UGC instances similar
to the target, we design a multi-modal retriever that fully considers
both content semantic and social attributes. The retriever consists
of two stages, semantic matching and attribute compensation. In
the semantic matching stage, we measure the textual and image
semantic similarities between the target and candidate using the
cosine similarity function:
𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖
𝐼𝑚𝑔=sim(𝒇𝑡𝑎𝑟𝑔𝑒𝑡
𝐼𝑚𝑔,𝒇𝑐𝑎𝑛𝑑𝑖
𝐼𝑚𝑔)(2)
𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖
𝑡𝑒𝑥𝑡=sim(𝒇𝑡𝑎𝑟𝑔𝑒𝑡
𝑡𝑒𝑥𝑡,𝒇𝑐𝑎𝑛𝑑𝑖
𝑡𝑒𝑥𝑡)(3)
where sim(·,·) denotes the cosine similarity function, 𝒇𝑡𝑎𝑟𝑔𝑒𝑡
∗ and
𝒇𝑐𝑎𝑛𝑑𝑖
∗ denote the features of the target UGC and candidate UGC.
Then we can have the semantic matching score as:
𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖=𝛼·𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖
𝐼𝑚𝑔+(1−𝛼)·𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖
𝑡𝑒𝑥𝑡(4)
where𝛼∈( 0,1)is a hyper-parameter that balances the contribution
of the text modal and image modal.
In the attribute compensation stage, we introduce social at-
tributes to compensate for the popularity gap among UGCs with
similar semantics. Social attributes are frequently associated with
specific audiences and delivery pathways [ 10], as rarer attributes
tend to carry more information and focusing on them helps distin-
guish propagation patterns.
Considering that there are 𝑚social attributes 𝑨={𝑎 1,𝑎2,...,𝑎𝑚}
involved in the dataset (The details of the social attributes in differ-
ent datasets are illustrated in the supplementary material), and each
social attribute gets 𝑚𝑖possible values, labeled as {𝑢𝑖
1,𝑢𝑖
2,...,𝑢𝑖
𝑛𝑖},
𝑖∈N[1,𝑚], the social attributes of a UGC instance can be denoted
as<𝑢1
𝑙1,𝑢2
𝑙2,...,𝑢𝑚
𝑙𝑚>,𝑙𝑖∈N[1,𝑛𝑖]. Then we can calculate its com-
pensation weight based on its rarity. Specifically, we evaluate the
rarity of each social attribute from global and local perspectives.
From the global sight, we adopt the inverse document frequency
(IDF) [34] as the global rarity, written as:
GR(𝑎𝑖)=log2𝑁𝑡𝑜𝑡𝑎𝑙
freq(𝑢𝑖
𝑙𝑖)+1+1(5)
where𝑁𝑡𝑜𝑡𝑎𝑙 denotes the totals of UGCs in the database, freq(𝑢𝑖
𝑙𝑖)re-
turns the occurred count of 𝑢𝑖
𝑙𝑖in the database. From the local sight,
we measure the local rarity within one social attribute following
Eq. 5, which can be expressed as:
LR(𝑎𝑖)=log2(freqmax(𝑎𝑖)
freq(𝑢𝑖
𝑙𝑖)+1+1)(6)
freqmax(𝑎𝑖)=Max(freq(𝑢𝑖
1),freq(𝑢𝑖
2),...,freq(𝑢𝑖
𝑛𝑖))(7)
Then, we can have the rarity of the social attribute 𝑎𝑖by calculating
the geometric average of GR(𝑎 𝑖)and LR(𝑎 𝑖), written as:
R(𝑎𝑖)=√︁
GR(𝑎𝑖)·LR(𝑎𝑖)(8)Subsequently, we introduce a matching function to calculate the
compensation weight for a candidate UGC, expressed as:
𝑤𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖=𝑚Ö
𝑖=1(1+Norm(R(𝑎 𝑖))·T(𝑎𝑐𝑎𝑛𝑑𝑖
𝑖,𝑎𝑡𝑎𝑟𝑔𝑒𝑡
𝑖))(9)
where Norm(·)denotes a min-max normalization, its detailed for-
mulation is provided in the supplementary material, 𝑎𝑐𝑎𝑛𝑑𝑖
𝑖and
𝑎𝑡𝑎𝑟𝑔𝑒𝑡
𝑖denotes the social attributes of the candidate UGC and tar-
get UGC, and T(·,·)denotes a matching function, written as:
T(𝑥,𝑦)=(
0, 𝑥=𝑦
1, 𝑥≠𝑦(10)
Finally, we can have the compensated matching score as:
ˆ𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖=𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖·𝑤𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖(11)
where ˆ𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑐𝑎𝑛𝑑𝑖denotes the compensated matching score, which will
finally be used to retrieve top-𝑁 𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙 similar UGC instances for
the following popularity prediction.
3.3 Relation-Guided Predictor
The RGP is supposed to predict the popularity of the input UGC
with the assistance of retrieved instances in the SAR. As shown
in Figure 2(b), we first introduce a semantic encoder to translate
the input UGC and retrieved instances to feature vectors, respec-
tively. Then, two RGTs, the self-relation-guided transformer (s-RGT)
and cross-relation-guided transformer (c-RGT) are sequentially in-
troduced to refine the features of retrieved instances. Finally, the
refined features and the input UGC features are jointly fed into an
MLP to predict popularity.
3.3.1Semantic Encoder.Taking a UGC as input, the semantic
encoder first uses pre-trained feature extractors to translate the
image content and textual content to corresponding feature vectors,
respectively. Assuming the translated image feature is 𝑓𝐼𝑚𝑔and
the translated textual feature is 𝑓𝑡𝑒𝑥𝑡, we adopt a cross-attention
mechanism [40] to fuse them, which can be expressed as:
𝒉𝐼𝑚𝑔=softmax𝒇𝐼𝑚𝑔·𝒇⊤
𝑡𝑒𝑥𝑡√𝑑𝑘
𝒇𝑡𝑒𝑥𝑡,(12)
𝒉𝑡𝑒𝑥𝑡=softmax𝒇𝑡𝑒𝑥𝑡·𝒇⊤
𝐼𝑚𝑔√𝑑𝑘
𝒇𝐼𝑚𝑔,(13)
𝒛=⟨𝒉𝐼𝑚𝑔,𝒉𝑡𝑒𝑥𝑡⟩,(14)
where𝑑𝑘denotes the feature dimension used for cross-modal at-
tention,⟨·,·⟩ denotes the concatenate operation, and 𝑧denotes the
fused feature produced by the semantic encoder.
3.3.2Relation Map with Social Attributes.Existing research
has demonstrated that incorporating a relation map between the tar-
get and retrieved instances can facilitate feature optimization [ 28].
To explicitly model relative attribute relations within the attention
mechanism, we construct a relation map that not only characterizes
the attribute pattern correspondences between arbitrary instance
pairs but also differentiates the attribute information of the target
instance, thereby highlighting the key relations most relevant to
the prediction task.

Enhancing Relation Modeling with Social Attributes for Social Media Popularity Prediction Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Given a set of UGC instances, containing 𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙 retrieved
UGC instances{𝐼1,𝐼2,...,𝐼𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙}and the corresponding tar-
get𝐼𝑡𝑎𝑟𝑔𝑒𝑡 , for any two elements 𝐼𝑖and𝐼𝑗in the collection, 𝑖,𝑗∈
{1,2,...,𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙,𝑡𝑎𝑟𝑔𝑒𝑡} , we can calculate their relation degree
under the social attribute𝑎 𝑘∈𝑨as:
𝑬𝑎𝑘(𝐼𝑖,𝐼𝑗)= 
0, 𝑎𝑘|𝐼𝑖≠𝑎𝑘|𝐼𝑗,
1, 𝑎𝑘|𝐼𝑖=𝑎𝑘|𝐼𝑗≠𝑎𝑘|𝐼𝑡𝑎𝑟𝑔𝑒𝑡,
2, 𝑎𝑘|𝐼𝑖=𝑎𝑘|𝐼𝑗=𝑎𝑘|𝐼𝑡𝑎𝑟𝑔𝑒𝑡.(15)
where𝑎𝑘|𝐼𝑖denotes the value of the social attribute 𝑎𝑘in the UGC
instance𝐼𝑖. Thus, we can obtain a relation map for 𝑎𝑘by traversing
all possible combinations of (𝐼𝑖,𝐼𝑗), denoted as 𝑬𝑎𝑘. Then, By stack-
ing the𝑚attribute-wise relation maps, we can obtain the overall
relation map 𝑬. We calculate two relation maps, the self-relation
map 𝑬𝑠𝑒𝑙𝑓and the cross-relation map 𝑬𝑐𝑟𝑜𝑠𝑠, for the subsequent s-
RGT and c-RGT respectively. Specifically, the 𝐼𝑡𝑎𝑟𝑔𝑒𝑡 is only included
in the calculation of For 𝑬𝑐𝑟𝑜𝑠𝑠, but not the For 𝑬𝑠𝑒𝑙𝑓. Therefore, the
shape of 𝑬𝑠𝑒𝑙𝑓is𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙×𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙×𝑚, while the shape of For
𝑬𝑐𝑟𝑜𝑠𝑠is1×𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙×𝑚.
3.3.3Relation-Guided Transformer.Though the relation map
is graph-like data, simply adopting graph convolution to handle the
relation map may suffer from its limited receptive fields, especially
when we get a large number of retrieved instances. Against this
limitation, we propose a relation-guided transformer to refine the
feature sequence of retrieved instances from a global sight.
Considering a group of <𝒒𝒖𝒆𝒓𝒚,𝒌𝒆𝒚,𝒗𝒂𝒍𝒖𝒆> group labeled
as<𝑸,𝑲,𝑽> , we first introduce a weight matrix 𝑀calculated
by the relation map 𝑬to refine the 𝐾, then use the refined 𝑲to
complete the rest of the calculation as a standard transformer does.
Specifically, we use an embedding layer to obtain embeddings of
𝑬, then introduce an MLP to further calculate 𝑀, which can be
expressed as:
𝑴=MLP(Embed(𝑬))(16)
where Embed(·) denotes the embedding operation. So that the
refined𝐾can be written as:
ˆ𝑲=𝑲·𝑴(17)
In practice, we design two RGTs, self-relation-guided transformer
(s-RGT) and cross-relation-guided transformer (c-RGT), that the
s-RGT is supposed to conduct a preliminary refinement using the
internal relationship among retrieved instances, while the c-RGT is
supposed to conduct a further refinement using the cross relation-
ship between the target and retrieved instances. Given 𝑁retrieved
instances{𝒛1,𝒛2,...,𝒛𝑁}, we concatenate each semantic feature 𝒛𝑖
with its popularity 𝑝𝑖and similarity score 𝑠𝑖, then map the concate-
nated vector through an MLP to obtain a fused representation ˜𝒛𝑖.
and then adding positional encodings [ 41] to formulate the input
of s-RGT𝒛 𝑖𝑛, which can be expressed as:
˜𝒛𝑖=MLP <𝒛𝑖,𝑝𝑖,ˆ𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑖>(18)
𝒛in={˜𝒛𝑖+PosEnc(𝑖)}𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙
𝑖=1(19)
where PosEnc(·) denotes the positional encoding operation, 𝑝𝑖
andˆ𝑠𝑡𝑎𝑟𝑔𝑒𝑡
𝑖denote the corresponding popularity and compensated
matching score of 𝑧𝑖. For the s-RGT, we use three linear layers togenerate <𝑸,𝑲,𝑽> from 𝒛𝑖𝑛, and obtain a self-refined feature
𝒛𝑠𝑒𝑙𝑓. For the c-RGT, we use one linear layer to generate 𝑸from
𝒛𝑡𝑎𝑟𝑔𝑒𝑡 and another two linear layers to generate <𝑲,𝑽> from
𝒛𝑠𝑒𝑙𝑓, where𝒛𝑡𝑎𝑟𝑔𝑒𝑡 denotes the features of the input UGC derived
by the semantic encoder. Thus, we can obtain the further refined
feature𝒛𝑐𝑟𝑜𝑠𝑠 through the c-RGT.
3.3.4 Predictor.In the final prediction stage, we combine the 𝒛𝑡𝑎𝑟𝑔𝑒𝑡
and𝒛𝑐𝑟𝑜𝑠𝑠 together, and send them to an MLP to predict the popu-
larity of the input UGC, which can be expressed as:
ˆ𝑝=MLP(<𝒛 𝑡𝑎𝑟𝑔𝑒𝑡,𝒛𝑐𝑟𝑜𝑠𝑠>)(20)
We measure the L2 loss between predicted popularity and ground-
truth popularity, providing full supervision for our RE-Rag.
4 EXPERIMENTS
In this section, we evaluate the effectiveness of the proposed RE-
Rag model. We conduct experiments on three datasets and compare
it with multiple strong baselines. Ablation studies are performed
to verify the contribution of key components. In addition, a se-
ries of experiments are carried out to further analyze the retrieval
capability, prediction performance, and stability of the model.
4.1 Implementation Details
4.1.1 Datasets and Metrics.We adopt three publicly available datasets
for SMPP to evaluate our method, including two widely used ICIP
[27] and SMPD (Image-300K) [ 45], and a newly proposed SMTPD
[49]. The details of three dataset are illustrated in Table 1. The
dataset split ratio is 8:1:1 for training, validation, and test sets, re-
spectively. Since ICIP and SMTPD provide temporal popularity
labels, we only predict the popularity of 30th-day as previous stud-
ies [38,45] did. Across all datasets, we use category information at
different hierarchical levels and author information as attributes,
and for the SMTPD dataset, due to its multilingual content, we
additionally incorporate language as an attribute.
Table 1: Dataset statistics used in our experiments.
Name #Samples #Users #Lang. Std (Pop.) Source
ICIP 20,337 17,302 1(EN) 1.54 Flickr
SMPD 305,595 38,307 1(EN) 2.47 Flickr
SMTPD 282,000 152,700 90+ 4.15 YouTube
4.1.2Settings and Metrics.We employ the Adam [ 17] optimizer
with an initial learning rate of 1e-5 and a batch size of 64 and a
weight decay of 1e-4. The learning rate is further adjusted using
PyTorch’s ReduceLROnPlateau scheduler at the end of each epoch.
An early stopping strategy is applied, terminating training if val-
idation performance does not improve for 5 consecutive epochs.
Each model is run five times, and all reported results are averaged
over these runs for robustness. We adopt three commonly used met-
rics: mean squared error (MSE), mean absolute error (MAE), and
Spearman’s rank correlation coefficient (SRC) to fairly evaluate the
performance of models compared. Details of the feature embedding
model are provided in the supplementary material.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Table 2: Performance comparison on three real-world datasets. The best results are in bold and the second-best are underlined .
Lower values of MSE and MAE, and higher values of SRC, indicate better performance.
ICIP SMPD SMTPD
Pub MSE MAE SRC MSE MAE SRC MSE MAE SRC
HyFea MM’20 2.0788 1.0460 0.3842 1.9032 0.9316 0.8324 8.0977 2.1585 0.6808
HMMVED IEEE TMM’23 2.0588 1.0328 0.4379 2.3585 1.1100 0.7866 6.2550 1.9414 0.7768
MASSL ICCIP’22 2.0621 0.9565 0.4602 1.7514 0.9045 0.8435 7.6974 2.1031 0.7062
UHAN WWW’18 2.1045 1.1047 0.5731 4.7491 1.6275 0.6698 7.2069 2.0726 0.6793
CBAN NEUCOM’22 1.7898 0.9239 0.4935 1.6421 0.8661 0.8560 7.2139 2.0534 0.7206
Jab ICWSM’22 2.2529 1.0845 0.4047 2.3673 1.0293 0.7862 9.7532 2.4804 0.6009
MMRA SIGIR’24 1.7549 0.8978 0.5049 1.6023 0.8455 0.8621 6.2997 1.8974 0.7803
ICPF AAAI’25 1.9031 0.9021 0.3874 2.7408 1.1589 0.7329 7.4978 1.9850 0.7327
RAGtrans KDD’24 1.5719 0.8332 0.6387 1.0318 0.6541 0.8927 5.2274 1.6516 0.8125
SKAPP AAAI’251.10750.6829 0.69751.1926 0.6487 0.9012 7.4017 2.0131 0.7796
RE-Rag (Ours) 1.1360 0.66170.6740 0.7263 0.4577 0.9355 4.2287 1.4697 0.8338
4.2 Main Results
We evaluate the effectiveness of our model by comparing it with
the following state-of-the-art approaches, including the feature
engineering-based methods: Hyfea [ 18]; the deep learning-based
methods: HMMVED [ 47], MASSL [ 52], UHAN [ 50], CBAN [ 8] Jab
[43]; the retrieval-augmented methods: MMRA [ 53], ICPF [ 6], RAG-
trans [ 7], SKAPP [ 48]. The comparison results are presented in
Table 2. The proposed model outperforms the comparison meth-
ods on the vast majority of metrics. Notably, our RE-Rag achieves
relative improvements of29 .61%,29.43%,3.80%in MSE, MAE and
SRC over the second-best method on the SMPD dataset. Besides,
our method also exhibits encouraging performance on the more
challenging SMTPD dataset. Although some comparison methods
also employ retrieval-enhanced strategies, their performance gains
remain limited due to constraints in retrieval quality and model-
ing approaches. SKAPP achieves competitive performance on the
smaller ICIP dataset, where weaker distribution-related attributes
(e.g., ispro and HasStats) and limited data scale reduce the diffi-
culty of modeling propagation mechanisms, resulting in similar
performance across methods. However, it exhibits clear degradation
on the more complex SMPD and SMTPD datasets, which provide
richer semantics, larger volumes, and attributes more closely corre-
lated with distribution mechanisms. This indicates that its multi-
stage retrieval and feature modeling strategies are insufficient to
capture distribution-related propagation differences. In contrast,
RE-Rag consistently maintains strong performance across datasets
of varying scales and semantic complexity, demonstrating superior
robustness and generalization capability (see Tables 4 and 5).
4.3 Ablation Study
We conduct extensive ablation studies to assess the contribution
of each key component in RE-Rag. The effects of individual social
attributes are reported in the supplementary material.
4.3.1Impact of Semantic-Attribute Retriever.We evaluated
two variant models: (1)w/o Retrieval: the retrieval mechanism wascompletely removed; (2)w/o Attribute: During the retrieval phase,
attribute compensation scores are eliminated, and the process relies
solely on semantic similarity; (3)w/o LR: Only Use GR; (4)w/o
GR: Only Use LR; The results are shown in Table 3.
From the results, removing the retrieval mechanism leads to a
sharp performance reduction, which confirms the effectiveness of
retrieval augmentation. Besides, eliminating attribute compensa-
tion scores also weakens the overall performance, high similarity in
semantic space alone does not guarantee consistency in propagation
mechanisms. Without attribute-based compensation, the number
of pseudo-correlated instances increases, thereby impairing overall
performance. Additionally, we find that removing either local rarity
or global rarity during retrieval leads to a performance drop, indi-
cating that the two play complementary roles in candidate selection.
Local rarity highlights the distinctiveness of an instance within
its attribute space, while global rarity reflects the overall informa-
tion value of the attribute across the dataset. Their combination
not only mitigates redundancy caused by high-frequency attribute
values but also preserves low-frequency yet representative features,
thereby enhancing the effectiveness of retrieval augmentation.
4.3.2Impact of Semantic Encoder.To evaluate the contribu-
tion of each feature component in the semantic encoder to overall
performance, we designed three investigation models: (1)w/o Text:
textual modality removed; (2)w/o Image: image modality removed;
(3)w/o Cross: cross-attention mechanism removed.
Removing either the textual feature or the image feature would
bring a visible performance reduction. The absence of the textual
feature is more serious than the image feature. This suggests that the
textual feature could provide more accurate semantic information
compared to the image feature. In addition, the cross-attention
mechanism also contributes to obtaining more effective feature
representation by fusing the textual and image features.
4.3.3Impact of the Relation-Guided Transformer.In this
part, we evaluate the contribution of components related to the
RGTs. Specifically, we design three experimental models: (1)w/o

Enhancing Relation Modeling with Social Attributes for Social Media Popularity Prediction Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 3: Ablation study of RE-Rag on the SMPD dataset.
Module Variant MSE MAE SRC
Retriever w/o Retrieval 1.5690 0.7923 0.8617
w/o Attribute 0.9101 0.5041 0.9233
w/o LR 0.7682 0.4855 0.9320
w/o GR 0.7498 0.4762 0.9325
Modal w/o Cross 0.7514 0.4751 0.9339
w/o Image 0.7631 0.4763 0.9331
w/o Text 0.7843 0.4889 0.9306
Relation-Guided
Transformerw/o Relation Map 0.7801 0.4867 0.9314
w/o s-RGT 0.7547 0.5125 0.9334
w/o c-RGT 0.7756 0.5123 0.9340
RE-Rag (Full Model) 0.7263 0.4577 0.9355
/uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013
/uni0000000b/uni00000044/uni0000000c/uni00000003/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003Nretrival/uni00000013/uni00000011/uni0000001b/uni00000013/uni00000011/uni0000001c/uni00000014/uni00000011/uni00000013/uni00000014/uni00000011/uni00000014/uni00000014/uni00000011/uni00000015/uni00000030/uni00000036/uni00000028/uni00000003/uni0000000b/uni0000002c/uni00000026/uni0000002c/uni00000033/uni00000003/uni00000012/uni00000003/uni00000036/uni00000030/uni00000033/uni00000027/uni0000000c
/uni00000030/uni0000004c/uni00000051/uni0000001d/uni00000003/uni00000014/uni00000011/uni00000014/uni00000016/uni00000019
/uni00000030/uni0000004c/uni00000051/uni0000001d/uni00000003/uni00000013/uni00000011/uni0000001a/uni00000015/uni00000019
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013
/uni0000000b/uni00000045/uni0000000c/uni00000003/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003
/uni00000013/uni00000011/uni0000001b/uni00000013/uni00000011/uni0000001c/uni00000014/uni00000011/uni00000013/uni00000014/uni00000011/uni00000014/uni00000014/uni00000011/uni00000015
/uni00000030/uni0000004c/uni00000051/uni0000001d/uni00000003/uni00000014/uni00000011/uni00000014/uni00000016/uni00000019
/uni00000030/uni0000004c/uni00000051/uni0000001d/uni00000003/uni00000013/uni00000011/uni0000001a/uni00000015/uni00000019
/uni00000017/uni00000011/uni00000015/uni00000013/uni00000013/uni00000017/uni00000011/uni00000015/uni00000015/uni00000018/uni00000017/uni00000011/uni00000015/uni00000018/uni00000013/uni00000017/uni00000011/uni00000015/uni0000001a/uni00000018/uni00000017/uni00000011/uni00000016/uni00000013/uni00000013/uni00000017/uni00000011/uni00000016/uni00000015/uni00000018/uni00000017/uni00000011/uni00000016/uni00000018/uni00000013/uni00000017/uni00000011/uni00000016/uni0000001a/uni00000018/uni00000017/uni00000011/uni00000017/uni00000013/uni00000013
/uni00000030/uni0000004c/uni00000051/uni0000001d/uni00000003/uni00000017/uni00000011/uni00000015/uni00000015/uni0000001c
/uni00000017/uni00000011/uni00000015/uni00000013/uni00000017/uni00000011/uni00000015/uni00000015/uni00000017/uni00000011/uni00000015/uni00000017/uni00000017/uni00000011/uni00000015/uni00000019/uni00000017/uni00000011/uni00000015/uni0000001b/uni00000017/uni00000011/uni00000016/uni00000013
/uni00000030/uni00000036/uni00000028/uni00000003/uni0000000b/uni00000036/uni00000030/uni00000037/uni00000033/uni00000027/uni0000000c
/uni00000030/uni0000004c/uni00000051/uni0000001d/uni00000003/uni00000017/uni00000011/uni00000015/uni00000015/uni0000001b/uni00000027/uni00000044/uni00000057/uni00000044/uni00000056/uni00000048/uni00000057/uni0000001d
/uni0000002c/uni00000026/uni0000002c/uni00000033 /uni00000036/uni00000030/uni00000033/uni00000027 /uni00000036/uni00000030/uni00000037/uni00000033/uni00000027
Figure 3: Parameter sensitivity analysis of RE-Rag on three
datasets: (a) Number of retrieved instances 𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙 , (b) Num-
ber of retrieval modality weight𝛼.
Relation Map: removing the relation map from the RGTs, which
means replacing the RGTs with a standard transformer; (2)w/o
s-RGT: removing the s-RGT; (3)w/o c-RGT: removing the c-RGT.
As shown in Table 3, removing the relation map leads to a notice-
able performance drop, indicating that incorporating a relation-
guided mechanism during context feature extraction helps the
model more effectively identify valuable information from retrieved
instances. Similarly, removing either s-RGT or c-RGT also degrades
performance, particularly increasing MAE, which suggests that con-
structing a relation network can effectively capture inter-instance
correlations and thereby reduce the overall average bias. Notably,
the MSE increase caused by removing c-RGT is substantially larger
than that from removing s-RGT, indicating that leveraging cross-
instance relations is more effective in mitigating extreme errors.
The complementary roles of s-RGT and c-RGT together contribute
to the optimal performance of the full RE-Rag model.
4.4 Parameter Sensitivity
4.4.1Number of retrieved instances 𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙 .Fig. 3 (a) illus-
trates the model’s performance across different retrieval set sizes
𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙 . The results show that variations in the number of re-
trieved instances exert only a minor effect on the accuracy of the
prediction. This robustness mainly benefits from the retriever’s
ability to prioritize the recall of key instances based on attribute
/uni00000013 /uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013
/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000048/uni00000047/uni00000003/uni00000033/uni00000052/uni00000056/uni0000004c/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000014/uni00000011/uni00000013/uni00000013/uni00000014/uni00000011/uni00000015/uni00000018/uni00000014/uni00000011/uni00000018/uni00000013/uni00000014/uni00000011/uni0000001a/uni00000018/uni00000015/uni00000011/uni00000013/uni00000013/uni00000015/uni00000011/uni00000015/uni00000018/uni00000015/uni00000011/uni00000018/uni00000013/uni00000024/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004a/uni00000048/uni00000003/uni0000002f/uni00000044/uni00000045/uni00000048/uni0000004f/uni00000003/uni00000027/uni0000004c/uni00000049/uni00000049/uni00000048/uni00000055/uni00000048/uni00000051/uni00000046/uni00000048
/uni00000035/uni00000028/uni00000010/uni00000035/uni00000044/uni0000004a
/uni00000030/uni00000030/uni00000035/uni00000024
/uni00000035/uni00000024/uni0000002a/uni00000037/uni00000055/uni00000044/uni00000051/uni00000056
/uni00000036/uni0000002e/uni00000024/uni00000033/uni00000033Figure 4: Average popularity gap between query samples and
retrieved results across ranking positions on the SMPD.
cues. Even if the retrieval results contain some samples with rel-
atively low relevance, the subsequent retrieval feature extraction
module can effectively filter out noise through relative relationship
modeling, thereby ensuring the stability and reliability of the pre-
diction results. Based on this, we set 𝑁𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙 to 40 on all datasets
to achieve a balance in overall performance.
4.4.2Impact of Retrieval Modality Weighting.Fig. 3(b) shows
the impact of retrieval modality weight 𝛼on model performance
across different datasets, where 𝛼∈[ 0,1]denotes the weight as-
signed to the image modality. We observe that extreme values of 𝛼
lead to degraded performance. When 𝛼is relatively small, the model
generally performs well, indicating that textual information often
carries more valuable features. Moreover, on the SMTPD dataset,
the optimal value of 𝛼is slightly higher, which we attribute to
platform differences. Considering the performance across datasets,
we finally fix 𝛼to 0.3 to achieve a balanced setting for all datasets.
4.5 Case Study
4.5.1Analysis of Retrieval Quality.To evaluate the effective-
ness of retrieval quality, we compared three retrieval methods
(MMRA, SKAPP and RAGtrans) and plotted the average popularity
difference curves between the target instances and their top 100 re-
trieved results. This metric can, to some extent, reflect the retrieval
model’s performance in similarity discrimination and ranking pre-
cision. As shown in Fig. 4, RE-Rag effectively outperforms other
baselines in terms of smaller average difference magnitude and
more stable fluctuations, indicating its ability to discover samples
with similar diffusion patterns while achieving finer-grained simi-
larity discrimination and ranking. In contrast, RAGtrans exhibits
the poorest performance, as it abstracts content semantics into
aspect-level representations and incorporates attribute features via
BM25-based weighting scheme. The lack of fine-grained interac-
tions leads to limited discriminative power in retrieval results.
4.5.2Analysis of retrieved instances.We randomly select a
UGC sample from the test set for retrieval visualization and com-
pare it with several strong baseline retrieval-augmented models. As
shown in Table 4, compared with other methods, our approach not

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Target Method Top1 Top2 Top3 Method Top1 Top2 Top3
RE-Rag
 MMRA
Bluebell WoodVine
#tree #artFalling: Crisp Old tree trunk WoodTrees
& Faces
11.82 12.64 10.42 9.86 6.86 4.17 7.13 7.81
RAGTrans
 SKAPP
Branch end
#camera #treesBluebell WoodAdult
Great TitInsects
lovenatureBluebell Wood ReflectionThe Love
tree
12.51 11.03 12.64 7.32 9.06 10.33 12.64 8.79 8.98
Table 4: Top-3 Retrieved Instances and Popularity under Different Variants
Table 5: Performance of Different Retrieval Feature Extrac-
tors under a Common Semantic-Attribute Retriever Input.
Variant MSE MAE SRC
Prediction via MMRA [53] 0.9549 0.5723 0.9103
Prediction via SKAPP [48] 0.9721 0.5667 0.9197
Prediction via RAGtrans [7] 0.8551 0.5815 0.9240
RE-Rag(RGTs) 0.7263 0.4577 0.9355
Table 6: Complexity analysis on SMTPD.
Model Retr. Inference Param. size
MMRA 1.6h2.2min18.12M
ICPF 2.9h 2.0h 213.97M
SKAPP 6.5h 2.3min 13.62M
RAGtrans 1.2h 3.4min 44.24M
RE-Rag72s2.8min 16.86M
only achieves broader coverage of retrieved instances but also ex-
hibits closer consistency with the target sample in terms of popular-
ity distribution, thereby validating the effectiveness of the Semantic-
Attribute retriever.
4.5.3Analysis of Retrieval Feature Extractor.To further evalu-
ate the effectiveness of the retrieval feature extractor, we conducted
comparative experiments by replacing the retrieval feature extrac-
tor module of RE-Rag with the corresponding components from
other retrieval-augmented paradigms (MMRA, SKAPP and RAG-
trans), while keeping the retrieval input consistent. Specifically,
MMRA adopts a bipolar attention mechanism, whereas RAGtrans
leverages a hypergraph transformer to extract auxiliary features
from retrieved instances for popularity prediction. As shown in Ta-
ble 5, our proposed RGTs achieves the best performance under this
setting, which confirms that modeling relative attribute relations
effectively enhances the retrieval feature extractor.
4.5.4Model robustness.We rigorously evaluated the robustness
of RE-Rag against four typical baseline methods. Fig. 5 illustrates
/uni00000014/uni00000013/uni00000013/uni00000008 /uni0000001b/uni00000013/uni00000008 /uni00000019/uni00000013/uni00000008 /uni00000017/uni00000013/uni00000008 /uni00000015/uni00000013/uni00000008/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013/uni00000014/uni00000011/uni00000018/uni00000015/uni00000011/uni00000013/uni00000015/uni00000011/uni00000018/uni00000016/uni00000011/uni00000013/uni00000030/uni00000036/uni00000028/uni00000035/uni00000028/uni00000010/uni00000035/uni00000044/uni0000004a /uni00000035/uni00000024/uni0000002a/uni00000057/uni00000055/uni00000051/uni00000044/uni00000056 /uni00000036/uni0000002e/uni00000024/uni00000033/uni00000033 /uni00000026/uni00000025/uni00000024/uni00000031 /uni0000002b/uni0000005c/uni00000049/uni00000048/uni00000044
/uni00000035/uni00000028/uni00000010/uni00000035/uni00000044/uni0000004a /uni00000035/uni00000024/uni0000002a/uni00000057/uni00000055/uni00000051/uni00000044/uni00000056 /uni00000036/uni0000002e/uni00000024/uni00000033/uni00000033 /uni00000026/uni00000025/uni00000024/uni00000031 /uni0000002b/uni0000005c/uni00000049/uni00000048/uni00000044
/uni00000013/uni00000008/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008
/uni00000033/uni00000048/uni00000055/uni00000049/uni00000052/uni00000055/uni00000050/uni00000044/uni00000051/uni00000046/uni00000048/uni00000003/uni00000027/uni00000048/uni00000046/uni00000044/uni0000005c/uni00000003/uni0000000b/uni00000008/uni0000000c
Figure 5: Impact of training set proportion on SMPD: bars
indicate MSE, line indicates performance degradation gain.
the performance of multiple baselines under varying training set
sizes. Notably, RE-Rag consistently achieves the highest perfor-
mance across all settings, substantially outperforming the com-
peting methods and demonstrating its exceptional stability and
generalization capabilities.
5 Conclusion
In this study, we propose a Relationship-Enhanced Retrieval-Augmented
Framework (RE-Rag) for media popularity prediction. Its core idea is
to model relative relationships among UGCs from a dual-factor per-
spective, considering both content quality and distribution mecha-
nism, and incorporate them into the retrieval and prediction process
to effectively leverage valuable historical instances. This enhances
the accuracy and generalization capability of popularity predictions.
Extensive experiments on three real-world datasets validate the
effectiveness and superior performance of this approach.
6 Acknowledgments
This work is supported by the Key R&D Program of Zhejiang under
Grant No. 2023C01044.

Enhancing Relation Modeling with Social Attributes for Social Media Popularity Prediction Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]Fatma S Abousaleh, Wen-Huang Cheng, Neng-Hao Yu, and Yu Tsao. 2020. Multi-
modal deep learning framework for image popularity prediction on social media.
IEEE Transactions on Cognitive and Developmental Systems13, 3 (2020), 679–692.
[2]Nesrine Ben Hassine, Pascale Minet, Dana Marinca, and Dominique Barth. 2019.
Popularity prediction–based caching in content delivery networks.Annals of
Telecommunications74, 5 (2019), 351–364.
[3]Sebastian Borgeaud and et al. 2022. Improving language models by retrieving
from trillions of tokens. InICML.
[4]Jingyuan Chen, Xuemeng Song, Liqiang Nie, Xiang Wang, Hanwang Zhang, and
Tat-Seng Chua. 2016. Micro tells macro: Predicting the popularity of micro-videos
via a transductive model. InProceedings of the 24th ACM international conference
on Multimedia. 898–907.
[5]Justin Cheng, Lada Adamic, P Alex Dow, Jon Michael Kleinberg, and Jure
Leskovec. 2014. Can cascades be predicted?. InProceedings of the 23rd inter-
national conference on World wide web. 925–936.
[6]Zhangtao Cheng, Jiao Li, Jian Lang, Ting Zhong, and Fan Zhou. 2025. In-context
Prompt-augmented Micro-video Popularity Prediction. InProceedings of the AAAI
Conference on Artificial Intelligence, Vol. 39. 11527–11535.
[7]Zhangtao Cheng, Jienan Zhang, Xovee Xu, Goce Trajcevski, Ting Zhong, and
Fan Zhou. 2024. Retrieval-augmented hypergraph for multimodal social media
popularity prediction. InProceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining. 445–455.
[8]Tsun-hin Cheung and Kin-man Lam. 2022. Crossmodal bipolar attention for
multimodal classification on social media.Neurocomputing514 (2022), 1–12.
[9]Minhwa Cho, Dahye Jeong, and Eunil Park. 2024. AMPS: Predicting popularity
of short-form videos using multi-modal attention mechanisms in social media
marketing environments.Journal of Retailing and Consumer Services78 (2024),
103778.
[10] Munmun De Choudhury, Hari Sundaram, Ajita John, Doree Duncan Seligmann,
and Aisling Kelliher. 2010. " Birds of a Feather": Does User Homophily Impact
Information Diffusion in Social Media?arXiv preprint arXiv:1006.1702(2010).
[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, Haofen Wang, et al .2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023), 32.
[12] Chih-Chung Hsu, Chia-Ming Lee, Xiu-Yu Hou, and Chi-Han Tsai. 2023. Gradient
boost tree network based on extensive feature analysis for popularity prediction of
social posts. InProceedings of the 31st ACM International Conference on Multimedia.
9451–9455.
[13] Liya Ji, Chan Ho Park, Zhefan Rao, and Qifeng Chen. 2023. Neural image pop-
ularity assessment with retrieval-augmented transformer. InProceedings of the
31st ACM International Conference on Multimedia. 2427–2436.
[14] Xinke Jiang, Rihong Qiu, Yongxin Xu, Yichen Zhu, Ruizhe Zhang, Yuchen Fang,
Chu Xu, Junfeng Zhao, and Yasha Wang. 2024. Ragraph: A general retrieval-
augmented graph learning framework.Advances in Neural Information Processing
Systems37 (2024), 29948–29985.
[15] Aditya Khosla, Atish Das Sarma, and Raffay Hamid. 2014. What makes an image
popular?. InProceedings of the 23rd international conference on World wide web.
867–876.
[16] Aditya Khosla, Atish Das Sarma, and Raffay Hamid. 2014. What makes an image
popular?. InProceedings of the 23rd international conference on World wide web.
867–876.
[17] Diederik P. Kingma and Jimmy Ba. 2017. Adam: A Method for Stochastic Opti-
mization. arXiv:1412.6980 [cs.LG] https://arxiv.org/abs/1412.6980
[18] Xin Lai, Yihong Zhang, and Wei Zhang. 2020. HyFea: Winning solution to social
media popularity prediction for multimedia grand challenge 2020. InProceedings
of the 28th ACM International Conference on Multimedia. 4565–4569.
[19] Patrick Lewis and et al. 2020. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks. InNeurIPS.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[21] Chen Li, Xiaoyu Wang, Tongyu Zong, Houwei Cao, and Yong Liu. 2023. Predic-
tive edge caching through deep mining of sequential patterns in user content
retrievals.Computer Networks233 (2023), 109866.
[22] Dongliang Liao, Jin Xu, Gongfu Li, Weijie Huang, Weiqing Liu, and Jing Li. 2019.
Popularity prediction on online articles with deep fusion of temporal process and
content features. InProceedings of the AAAI conference on artificial intelligence,
Vol. 33. 200–207.
[23] Yin Luo, Fangfang Wang, Feifei Zhao, Jianbin Guo, Lei Wang, Yanni Hao, and
Daniel Dajun Zeng. 2019. A framework for policy information popularity pre-
diction in new media. In2019 IEEE International Conference on Intelligence and
Security Informatics (ISI). IEEE, 209–211.
[24] Jinna Lv, Wu Liu, Meng Zhang, He Gong, Bin Wu, and Huadong Ma. 2017. Multi-
feature fusion for predicting social media popularity. InProceedings of the 25thACM international conference on Multimedia. 1883–1888.
[25] Mira Mayrhofer, Jörg Matthes, Sabine Einwiller, and Brigitte Naderer. 2020. User
generated content presenting brands on social media increases young adults’
purchase intention.International Journal of Advertising39, 1 (2020), 166–186.
[26] Nic Newman, Arguedas Ross Arguedas, Craig T Robertson, Rasmus Kleis Nielsen,
and Richard Fletcher. 2025.Digital news report 2025. Reuters Institute for the
study of Journalism.
[27] Alessandro Ortis, Giovanni Maria Farinella, and Sebastiano Battiato. 2019. Pre-
diction of social image popularity dynamics. InInternational Conference on Image
Analysis and Processing. Springer, 572–582.
[28] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey.
arXiv preprint arXiv:2408.08921(2024).
[29] Henrique Pinto, Jussara M Almeida, and Marcos A Gonçalves. 2013. Using early
view patterns to predict the popularity of youtube videos. InProceedings of the
sixth ACM international conference on Web search and data mining. 365–374.
[30] Yang Qian, Wang Xu, Xiao Liu, Haifeng Ling, Yuanchun Jiang, Yidong Chai,
and Yezheng Liu. 2022. Popularity prediction for marketer-generated content: A
text-guided attention neural network for multi-modal feature fusion.Information
Processing & Management59, 4 (2022), 102984.
[31] Yang Qian, Wang Xu, Xiao Liu, Haifeng Ling, Yuanchun Jiang, Yidong Chai,
and Yezheng Liu. 2022. Popularity prediction for marketer-generated content: A
text-guided attention neural network for multi-modal feature fusion.Information
Processing & Management59, 4 (2022), 102984.
[32] Yingdan Shang, Bin Zhou, Xiang Zeng, Ye Wang, Han Yu, and Zhong Zhang.
2022. Predicting the popularity of online content by modeling the social influence
and homophily features.Frontiers in Physics10 (2022), 915756.
[33] Fabian Spaeh and Alina Ene. 2023. Online ad allocation with predictions.Advances
in Neural Information Processing Systems36 (2023), 17265–17295.
[34] Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its
application in retrieval.Journal of documentation28, 1 (1972), 11–21.
[35] Mohd Suhairi Md Suhaimin, Mohd Hanafi Ahmad Hijazi, Ervin Gubin Moung,
Puteri Nor Ellyza Nohuddin, Stephanie Chua, and Frans Coenen. 2023. Social
media sentiment analysis and opinion mining in public security: Taxonomy, trend
analysis, issues and future directions.Journal of King Saud University-Computer
and Information Sciences35, 9 (2023), 101776.
[36] Dachun Sun, You Lyu, Jinning Li, Yizhuo Chen, Tianshi Wang, Tomoyoshi Kimura,
and Tarek Abdelzaher. 2025. SCRAG: Social Computing-Based Retrieval Aug-
mented Generation for Community Response Forecasting in Social Media Envi-
ronments. In2025 IEEE International Conference on Smart Computing (SMART-
COMP). IEEE, 170–177.
[37] Gabor Szabo and Bernardo A Huberman. 2010. Predicting the popularity of
online content.Commun. ACM53, 8 (2010), 80–88.
[38] Alexandru Tatar, Marcelo Dias De Amorim, Serge Fdida, and Panayotis Anto-
niadis. 2014. A survey on predicting the popularity of web content.Journal of
Internet Services and Applications5, 1 (2014), 8.
[39] Jing Tian, Huayin Fan, and Zengwen Hou. 2022. Research on the prediction of
popularity of news dissemination public opinion based on data mining.Compu-
tational Intelligence and Neuroscience2022, 1 (2022), 6512602.
[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszoreit, Llion Jones, Aidan N
Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need.
Advances in Neural Information Processing Systems30 (2017).
[41] Benyou Wang, Lifeng Shang, Christina Lioma, Xin Jiang, Hao Yang, Qun Liu,
and Jakob Grue Simonsen. 2020. On position embeddings in bert. InInternational
conference on learning representations.
[42] Haoyu Wang, Zongxia Xie, Meiyao Liu, and Canhua Guan. 2023. Afrf: angle
feature retrieval based popularity forecasting. InProceedings of the 32nd ACM
International Conference on Information and Knowledge Management. 2606–2615.
[43] Evan Weissburg, Arya Kumar, and Paramveer S Dhillon. 2022. Judging a book
by its cover: Predicting the marginal impact of title on Reddit post popularity. In
Proceedings of the International AAAI Conference on Web and Social Media, Vol. 16.
1098–1108.
[44] Bo Wu, Wen-Huang Cheng, Yongdong Zhang, Qiushi Huang, Jintao Li, and Tao
Mei. 2017. Sequential prediction of social media popularity with deep temporal
context networks.arXiv preprint arXiv:1712.04443(2017).
[45] Bo Wu, Peiye Liu, Qiushi Huang, Zhaoyang Zeng, Jia Wang, Bei Liu, Jiebo Luo,
and Wen-Huang Cheng. 2024. SMP Challenge Summary: Social Media Prediction
Challenge. InProceedings of the 32nd ACM International Conference on Multimedia.
11442–11444.
[46] Max Würfel, Qiwei Han, and Maximilian Kaiser. 2021. Online advertising revenue
forecasting: An interpretable deep learning approach. In2021 IEEE International
Conference on Big Data (Big Data). IEEE, 1980–1989.
[47] Jiayi Xie, Yaochen Zhu, and Zhenzhong Chen. 2021. Micro-video popularity
prediction via multimodal variational information bottleneck.IEEE Transactions
on Multimedia25 (2021), 24–37.
[48] Xovee Xu, Yifan Zhang, Fan Zhou, and Jingkuan Song. 2025. Improving Mul-
timodal Social Media Popularity Prediction via Selective Retrieval Knowledge
Augmentation. InProceedings of the AAAI Conference on Artificial Intelligence,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Vol. 39. 932–940.
[49] Yijie Xu, Bolun Zheng, Wei Zhu, Hangjia Pan, Yuchen Yao, Ning Xu, Anan Liu,
Quan Zhang, and Chenggang Yan. 2025. SMTPD: A New Benchmark for Temporal
Prediction of Social Media Popularity. InProceedings of the Computer Vision and
Pattern Recognition Conference. 18847–18857.
[50] Wei Zhang, Wen Wang, Jun Wang, and Hongyuan Zha. 2018. User-guided
hierarchical attention network for multi-modal social image popularity prediction.
InProceedings of the 2018 world wide web conference. 1277–1286.
[51] Y. Zhang and et al. 2021. RA-Rec: A Retrieval-Augmented Recommendation
Framework. InSIGIR.[52] Zhuoran Zhang, Shibiao Xu, Li Guo, and Wenke Lian. 2022. Multi-modal Varia-
tional Auto-Encoder Model for Micro-video Popularity Prediction. InProceedings
of the 8th International Conference on Communication and Information Processing.
9–16.
[53] Ting Zhong, Jian Lang, Yifan Zhang, Zhangtao Cheng, Kunpeng Zhang, and
Fan Zhou. 2024. Predicting micro-video popularity via multi-modal retrieval
augmentation. InProceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 2579–2583.
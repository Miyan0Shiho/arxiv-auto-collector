# CMRAG: Co-modality-based document retrieval and visual question answering

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li

**Published**: 2025-09-02 09:17:57

**PDF URL**: [http://arxiv.org/pdf/2509.02123v1](http://arxiv.org/pdf/2509.02123v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a core paradigm in document
question answering tasks. However, existing methods have limitations when
dealing with multimodal documents: one category of methods relies on layout
analysis and text extraction, which can only utilize explicit text information
and struggle to capture images or unstructured content; the other category
treats document segmentation as visual input and directly passes it to visual
language models (VLMs) for processing, yet it ignores the semantic advantages
of text, leading to suboptimal generation results. This paper proposes
co-modality-based RAG (CMRAG), which can simultaneously leverage text and
images for efficient retrieval and generation. Specifically, we first perform
structured parsing on documents to obtain co-modality representations of text
segments and image regions. Subsequently, in response to user queries, we
retrieve candidate evidence from text and image channels, respectively, and
aggregate the results at the cross-modal retrieval level. Finally, we prompt
the VLM to generate the final response based on the co-modality retrieval
results. Experiments demonstrate that our method significantly outperforms
pure-vision-based RAG in visual document question answering tasks. The findings
of this paper show that integrating co-modality information into the RAG
framework in a unified manner is an effective approach to improving the
performance of complex document visual question-answering (VQA) systems.

## Full Text


<!-- PDF content starts -->

CMRAG: Co-modality–based document retrieval and
visual question answering
Wang Chen
Baidu Inc.
The University of Hong Kong
wchen22@connect.hku.hkGuanqiang Qi
Baidu Inc.
qiguanqiang@baidu.comWeikang Li
Peking University
wavejkd@pku.edu.cn
Yang Li∗
Baidu Inc.
liyang164@baidu.com
Abstract
Retrieval-Augmented Generation (RAG) has become a core paradigm in docu-
ment question answering tasks. However, existing methods have limitations when
dealing with multimodal documents: one category of methods relies on layout
analysis and text extraction, which can only utilize explicit text information and
struggle to capture images or unstructured content; the other category treats docu-
ment segmentation as visual input and directly passes it to visual language models
(VLMs) for processing, yet it ignores the semantic advantages of text, leading to
suboptimal generation results. This paper proposes co-modality–based RAG (CM-
RAG), which can simultaneously leverage text and images for efficient retrieval
and generation. Specifically, we first perform structured parsing on documents
to obtain co-modality representations of text segments and image regions. Sub-
sequently, in response to user queries, we retrieve candidate evidence from text
and image channels, respectively, and aggregate the results at the cross-modal
retrieval level. Finally, we prompt the VLM to generate the final response based
on the co-modality retrieval results. Experiments demonstrate that our method
significantly outperforms pure-vision–based RAG in visual document question
answering tasks. The findings of this paper show that integrating co-modality
information into the RAG framework in a unified manner is an effective approach
to improving the performance of complex document visual question-answering
(VQA) systems.
1 Introduction
Large language models (LLMs) have received extensive attention in recent years (Touvron et al.,
2023; Achiam et al., 2023; Guo et al., 2025; Yang et al., 2025a), but they have inherent limitations
in handling out-of-domain knowledge (Ji et al., 2023). To address this issue, RAG integrates
external knowledge retrieval with the generation process (Lewis et al., 2020; Guu et al., 2020;
Karpukhin et al., 2020; Chen et al., 2025). RAG achieves wide success in open-domain question
answering, knowledge retrieval, and dialogue systems, and becomes an effective means of extending
the knowledge boundaries of LLMs (Ram et al., 2023; Gao et al., 2023). However, most external
∗Corresponding author
PreprintarXiv:2509.02123v1  [cs.CL]  2 Sep 2025

(a) Text–based RAG
Layout detectionParsed textPositive views of the performance of public health officials also have declined significantly: 63% now ………A smaller majority of Republicans (62%) say the primary reason is because more people are being tested. Parsing
LLMQueryAnsRetrieving
VLMQueryAnsRetrieving(b) Image–based RAG
Positive views of the performance of public health officials also have declined significantly: 63% now ………A smaller majority of Republicans (62%) say the primary reason is because more people are being tested. 
VLMParsing
VLMQueryAnsRetrieving
Retrieving(c) Co-modality–based RAG (ours)Figure 1: Comparison among (a) text–based RAG, (b) image–based RAG, and (c) co-modality–based
RAG.
data sources (e.g., documents) are essentially multimodal (Jeong et al., 2025; Faysse et al., 2025),
often containing natural language text, formulas, tables, images, and complex layout structures. How
to effectively leverage such multimodal information in question answering remains a challenging
problem that is not fully solved.
One line of approaches is text-based RAG, which typically relies on layout parsing and text extraction
(Xu et al., 2020; Dong et al., 2025; Yang et al., 2025b; Perez & Vizcaino, 2024). These methods first
detect document layouts and then extract textual information for subsequent retrieval and generation.
While stable at the semantic level, they struggle to handle content such as images and tables. Recently,
VLMs (Ghosh et al., 2024; Radford et al., 2021; Alayrac et al., 2022) enable RAG systems to process
documents directly as images (Faysse et al., 2025; Yu et al., 2025; Qi et al., 2024a; Wang et al.,
2025b), giving rise to vision-based RAG (Huang et al., 2022; Kim et al., 2022; Yu et al., 2024).
Specifically, these methods divide document pages into image segments and perform retrieval and
reasoning through visual understanding models. Although they capture non-textual information, they
often overlook the precise information carried by text, leading to performance bottlenecks.
To overcome these limitations, we propose a novel co-modality–based RAG framework, which unifies
text and image modalities, as illustrated in Fig. 1. In this framework, we first parse documents to
extract structured text and image segments. For a given query, we then perform retrieval in both text
and visual spaces, ensuring that semantic matching from text and perceptual grounding from images
are simultaneously leveraged. Finally, we feed the co-modality evidence into a VLM to integrate
information and generate answers. In this way, our method not only handles cross-modal reasoning
that single-modal RAG cannot cover but also achieves significant performance improvements on
standard visual document question answering benchmarks.
The main contributions of this paper are summarized as follows:
•We propose a novel RAG framework that jointly leverages text and image modalities to
enhance performance on complex visual document QA tasks.
•We design a co-modality–based retrieval mechanism that enables complementary use of text
and image evidence during retrieval, leading to more comprehensive knowledge grounding.
•We conduct extensive experiments on document VQA benchmarks, and the results demon-
strate that our approach significantly outperforms existing single-modal RAG methods in
both accuracy and robustness.
2

2 Related work
In this section, we discuss the recent studies related to multi-modal RAG (MMRAG). Specifically,
we primarily focus on three folds: (1) knowledge–based MMRAG, (2) video–based MMRAG, and
(3) document–based MMRAG.
Knowledge–based MMRAG refers to retrieving knowledge (text or image modality) from external
sources such as Wikipedia articles and websites to answer textual or visual queries (Talmor et al.,
2021; Marino et al., 2019; Chang et al., 2022; Schwenk et al., 2022; Mensink et al., 2023; Chen et al.,
2023; Ma et al., 2024a; Hu et al., 2025). Although the external knowledge database can enhance
the system performance (Caffagni et al., 2024), the key issue of knowledge–based MMRAG is the
inconsistency between textual and visual queries as well as the external knowledge database (Chen
et al., 2022; Lin et al., 2023; Zhang et al., 2024b). To address this issue, Lin & Byrne (2022) adopted
multiple algorithms, including object detection, image captioning, and OCR, to transform visual
queries into language space, and proposed a joint training scheme to optimize retrieval and generation
simultaneously. A similar training strategy was also used by (Adjali et al., 2024). Also, Yan & Xie
(2024) used a consistent modality for both retrieval and generation: visual modality for retrieval
(visual queries and Wikipedia article pages) and textual modality for generation (textual queries and
wiki articles). A similar strategy can also be found in RORA (Qi et al., 2024a). In addition, Tian
et al. (2025) proposed cross-source knowledge reconciliation for MMRAG, which could address the
inconsistency between textual and visual external knowledge.
Video–based MMRAG refers to retrieving videos from the corpus to help answer given queries
(Caba Heilbron et al., 2015; Xu et al., 2016; Anne Hendricks et al., 2017; Wang et al., 2019; Kriz et al.,
2025; Wan et al., 2025). Since encoding videos may incur high computational costs, a few studies
pre-processed videos using VLMs and converted videos to textual modality (Zhang et al., 2024a;
Arefeen et al., 2024; Ma et al., 2025). For example, Zhang et al. (2024a) first detected key information
in videos such as human faces, based on which a VLM was prompted to generate scene captions
for video frames. Consequently, the video modality can be converted to a text modality, which can
significantly reduce computational costs and facilitate retrieval and generation. Furthermore, a few
studies (Luo et al., 2024; Jeong et al., 2025; Reddy et al., 2025) processed videos by selecting or
clustering representative frames, facilitating video retrieval and final generation.
Document–based MMRAG refers to retrieving a few document pages from one or multiple docu-
ments to help generate answers for given questions (Methani et al., 2020; Masry et al., 2022; Tanaka
et al., 2023; Tito et al., 2023; Ma et al., 2024b; Li et al., 2024; Hui et al., 2024; Qi et al., 2024b;
Cho et al., 2024; Wang et al., 2025a; Li et al., 2025; Wasserman et al., 2025; Faysse et al., 2025).
Traditionally, documents were parsed using detection models (Ge et al., 2021) and OCR engines
(Smith, 2007), and the extracted components (e.g., text) were input to LLMs to generate answers
(Riedler & Langer, 2024; Faysse et al., 2025). With the proliferation of VLMs, a few studies (Faysse
et al., 2025; Yu et al., 2025; Wang et al., 2025b) processed document pages as images directly.
Specifically, they used VLMs to encode queries and document pages as text and images, respectively,
based on which the similarity scores between queries and visual document pages can be calculated.
This method paves the way for document–based MMRAG, as it does not need to parse documents and
can retrieve document pages directly. However, this method overlooks text modality in documents,
which may degrade the performance of RAG systems.
3 Methodology
In this section, we introduce the problem definition of document–based VQA and our proposed
CMRAG framework in detail.
3.1 Problem definition
In a document–based VQA system, a collection of visual documents provides evidence for answering
given queries. Formally, all pages are treated as candidate evidence D={p1, p2, . . . , p M}, where
3

Mdenotes the total number of pages across the document collection. For a given query q, the
retriever Ridentifies the top- krelevant pages Pk={p1, p2, . . . , p k}according to a similarity score
s(q, pi)(∀pi∈Pk). Once the top- kevidence pages are retrieved, they are combined with the
query into a prompt P(q, Pk), which is then passed to a generator model G, often instantiated as a
vision–language model (VLM). The generator integrates the multimodal evidence and produces the
final answer ˆa=G(P(q, Pk)).
3.2 VLM–based document parsing
Advanced vision–language models (VLMs) can directly parse document pages, simultaneously de-
tecting layouts, extracting text, and identifying localized visual regions. For each page pi, the parsing
process produces three complementary components: the entire image Ii, which preserves global
layout and contextual relationships; the parsed sub-images SIi, which capture localized elements
such as figures, tables, and diagrams; and the parsed text Ti, which provides precise semantic
grounding for fine-grained reasoning. Thus, each page can be represented as pi={Ii, SIi, Ti},
ensuring that both global structure and detailed content are available for retrieval and downstream
answer generation.
3.3 Co-modality late interaction
Late interaction. Given a query qand document page image p, the encoder Eproject them into a
high-dimension space RD, i.e.,q=E(q)∈RNq×Dandp=E(p)∈RNp×D, where NqandNp
denote the number of vectors in the query and document embeddings, respectively. Hence, the late
interaction (LI) (Khattab & Zaharia, 2020; Santhanam et al., 2022; Clavié, 2025; Faysse et al., 2025)
can be calculated as follows:
LI(q, p) =NqX
i=1max
j∈{1,2,...,N p}⟨qi,pj⟩, (1)
where ⟨·,·⟩denote dot production, and qiandpjareithandjthvector in the query and page
embeddings, respectively. Compared with the mean pooling method (Karpukhin et al., 2020; Yu
et al., 2025), LI can consider token-level similarity for detailed matching (Wan et al., 2025).
Co-modality late interaction. Furthermore, inspired by CLAMR (Wan et al., 2025), we consider
retrieving evidence based on similarity between the query qand the co-modality document content
IandT. Formally, we encode the entire page image Iand extracted text TaspI=E(I)and
pT=E(T), respectively. Therefore, we adopt co-modality late interaction (CoLI) to calculate the
similarity, as follows:
CoLI (q, p) = max
m∈{I,T}NqX
i=1max
j∈{1,2,...,N p}⟨qi,pm,j⟩, (2)
where pm,jisjthvector in the mthmodality document embedding.
Contrastive learning. Let{(qi, pi)}b
i=1denote a data batch where each query qi(∀i∈ {1,2, ..., b})
corresponds to a document page piandbdenotes the batch size. The loss (Oord et al., 2018) can be
represented as follows:
L=−1
bbX
i=1logexp(si,i/τ)
Pb
j=1exp(si,j/τ), (3)
where si,j(∀i, j∈ {1,2, ..., b}) denote the similarity score between ithquery and jthdocument page
andτthe temperature.
3.4 Co-modality–based generation
Recent advances in VLMs (Bai et al., 2025; Du et al., 2025) enable them to directly process naive-
resolution page images without the need for splitting each page into smaller sub-images, which often
4

risks losing global context. In addition, modern VLMs can accept multiple images simultaneously,
meaning that all retrieved pages can be jointly fed into the model. This design facilitates cross-page
reasoning, where evidence from different parts of the document can be integrated to produce more
accurate and comprehensive answers. Therefore, we combine the retrieved images and text into the
prompt and leverage the VLM to generate the final answer based on the co-modality input.
4 Experiments
4.1 Experiment setup
Datasets. We consider two datasets: MMLongBench (Ma et al., 2024b) and SlideVQA (Tanaka et al.,
2023). These datasets comprise various query types (multi-hop and single-hop) and answer types
(multi-span and single-span), which can be used to evaluate the capacities of VLMs. The statistics of
the datasets are summarized in Tab. 1. We follow the implementation of (Wang et al., 2025a) and use
refined queries of SlideVQA.
Method #Files #Images #QAs #Images/Q
MMLongBench 135 6,529 1,082 1.00
SlideVQA 400 8,000 2,020 1.26
Table 1: Statistics of evaluation datasets.
Baselines. To evaluate the proposed method, we first consider three evaluation scenarios: (1) Oracle
(Entire images) , (2) Oracle (Sub-images + parsed text) , and (3) Oracle (Entire images + parsed
text) . We consider using the ground truth document pages, with the entire images, parsed sub-images
plus text, and entire images plus parsed text as the input of the VLM, respectively.
LLM-as-a-Judge. The gold answers in the benchmark datasets consist of multiple spans or exhibit
diverse valid expressions. Traditional evaluation metrics, such as Exact Match (EM) and token-level
F1, struggle to capture the semantic equivalence between generated responses and these multi-span
gold answers. As a result, these metrics may underestimate the actual performance of models. To
address this limitation, we employ a large language model (LLM) as an automatic judge to assess
whether a generated answer is correct, providing a more flexible and reliable evaluation framework,
enabling fairer comparisons across different models.
Implementation details. We implemented our CMRAG framework using Qwen2.5-VL-7B-Instruct
Bai et al. (2025) as the backbone VLM for parsing documents and answering queries. In addition, we
adopted Qwen2.5-7B-Instruct (Yang et al., 2025a) as the judge. To ensure deterministic outputs and
eliminate variability due to random sampling, we set the temperature to 0.0 during decoding. We
present the prompts used in this study in the appendix.
Method MMLongBench SlideVQA
Oracle (Entire images) 40.39 81.58
Oracle (Sub-images + parsed text) 29.11 58.76
Oracle (Entire images + parsed text) 46.86 81.68
Table 2: Main results. Bold values represent the best scores.
4.2 Main results
Tab. 2 summarizes the performance of different oracle settings on MMLongBench and SlideVQA. The
results reveal several key findings. First, incorporating parsed text consistently enhances generation
accuracy, as shown by the improved scores of the oracle that integrates entire images with parsed
text. For instance, on MMLongBench, the combined setting achieves the best score of 46.86,
which significantly outperforms using only entire images (40.39). Similarly, on SlideVQA, the
5

combination yields 81.68, slightly higher than the image-only oracle. Second, directly using sub-
images with parsed text leads to degraded performance. This decline is likely due to incomplete visual
information within sub-images, which may omit contextual elements crucial for accurate reasoning.
On MMLongBench, the score drops to 29.11, while SlideVQA falls to 58.76—both substantially
lower than their entire image counterparts. These results suggest that incomplete visual segmentation
can hinder the integration of multimodal cues. Overall, the findings highlight that while parsed text
provides a complementary signal that improves multimodal reasoning, extracting sub-images without
preserving global context can be detrimental. The best results are achieved when both entire images
and parsed text are jointly considered.
How many % of Rep/Lean Rep people think cases have risen primally because of more testing and how many % of Rep/Lean Rep people think the federal government should be primarily responsible for COVID-19 control policy?QueryAnswer[62, 30]Oracle (Entire images)36%ofRep/LeanReppeoplethinkcaseshaverisenprimarilybecauseofmoretesting,and30%ofthemthinkthefederalgovernmentshouldbeprimarilyresponsibleforCOVID-19controlpolicy.
Oracle (Entire images + parsed text)62%ofRep/LeanReppeoplethinkcaseshaverisenprimarilybecauseofmoretesting,and30%ofRep/LeanReppeoplethinkthefederalgovernmentshouldbeprimarilyresponsibleforCOVID-19controlpolicy.Oracle (Sub-images + parsed text)36
cc
VLM failed to understand the imageParsed text could help generate accurate answersVLM failed to parse the sub-image,but extracted text only
Figure 2: Qualitative comparison among three baselines.
4.3 Case study
To further illustrate the effects observed in Table 2, we present a case study in Fig. 2. This example
highlights the strengths and limitations of different Oracle settings in handling complex queries. First,
when relying solely on entire images, the VLM misinterprets the numbers, producing an incorrect
6

output of 36 instead of 62. This demonstrates that VLMs may struggle to accurately ground numeric
reasoning based on purely visual inputs. Similarly, when only sub-images plus parsed text are used,
the model fails to capture the complete context, yielding partial and incomplete answers. The problem
arises because the VLM failed to accurately parse the sub-image but extracted textual numbers only.
However, incorporating entire images together with parsed text enables the model to generate the
correct multi-span answer, as the parsed text provides a reliable textual grounding that compensates
for the VLM’s difficulty in interpreting fine-grained visual details. This shows that parsed text can
serve as an essential complement, ensuring accurate reasoning across multiple evidence spans.
Another challenge revealed by this case is that the query requires multi-span answers, i.e., identifying
two distinct values from different textual locations. Conventional automatic evaluation metrics, such
as exact match (EM) or F1 score, cannot adequately capture the correctness of such answers, which
further validates the necessity of LLM-as-a-Judge.
5 Conclusion
In this paper, we introduced a co-modality–based RAG framework that unifies text and image
modalities to address the limitations of existing single-modal approaches in visual document question
answering. By jointly leveraging parsed text and entire images, our method enables complementary
retrieval and reasoning, where textual evidence provides precise grounding and visual context
preserves global completeness. Experimental results on MMLongBench and SlideVQA demonstrate
that the proposed framework consistently outperforms single-modality baselines, and case studies
further highlight its advantages in handling multi-span queries that traditional metrics cannot fully
capture. Overall, this work establishes co-modality RAG as an effective and robust paradigm for
document VQA, paving the way for future research on integrating structured parsing, fine-grained
retrieval, and LLM-based evaluation.
7

References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 , 2023.
Omar Adjali, Olivier Ferret, Sahar Ghannay, and Hervé Le Borgne. Multi-level information retrieval
augmented generation for knowledge-based visual question answering. In Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing , pp. 16499–16513. Association
for Computational Linguistics, 2024.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel
Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language
model for few-shot learning. Advances in neural information processing systems , 35:23716–23736,
2022.
Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell.
Localizing moments in video with natural language. In Proceedings of the IEEE international
conference on computer vision , pp. 5803–5812, 2017.
Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, and Srimat Chakradhar. irag:
Advancing rag for videos with an incremental approach. In Proceedings of the 33rd ACM
International Conference on Information and Knowledge Management , pp. 4341–4348, 2024.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet:
A large-scale video benchmark for human activity understanding. In Proceedings of the ieee
conference on computer vision and pattern recognition , pp. 961–970, 2015.
Davide Caffagni, Federico Cocchi, Nicholas Moratelli, Sara Sarto, Marcella Cornia, Lorenzo Baraldi,
and Rita Cucchiara. Wiki-llava: Hierarchical retrieval-augmented generation for multimodal llms.
InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp.
1818–1826, 2024.
Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk.
Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition , pp. 16495–16504, 2022.
Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Deguo Xia, and Jizhou Huang. Pairs:
Parametric-verified adaptive information retrieval and selection for efficient rag. arXiv preprint
arXiv:2508.04057 , 2025.
Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William Cohen. Murag: Multimodal retrieval-
augmented generator for open question answering over images and text. In Proceedings of the
2022 Conference on Empirical Methods in Natural Language Processing , pp. 5558–5570, 2022.
Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei
Chang. Can pre-trained vision and language models answer visual information-seeking questions?
arXiv preprint arXiv:2302.11713 , 2023.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-
modal retrieval is what you need for multi-page multi-document understanding. arXiv preprint
arXiv:2411.04952 , 2024.
Benjamin Clavié. Jacolbertv2. 5: Optimising multi-vector retrievers to create state-of-the-art japanese
retrievers with constrained resources. Journal of Natural Language Processing , 32(1):176–218,
2025.
8

Yuyang Dong, Nobuhiro Ueda, Kriszti ˘A ˛ An Boros, Daiki Ito, Takuya Sera, and Masafumi Oyamada.
Scan: Semantic document layout analysis for textual and visual retrieval-augmented generation.
arXiv preprint arXiv:2505.14381 , 2025.
Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang,
Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491 , 2025.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, CELINE HUDELOT, and
Pierre Colombo. Colpali: Efficient document retrieval with vision language models. In The
Thirteenth International Conference on Learning Representations , 2025.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A
survey. arXiv preprint arXiv:2312.10997 , 2(1), 2023.
Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021.
arXiv preprint arXiv:2107.08430 , 2021.
Akash Ghosh, Arkadeep Acharya, Sriparna Saha, Vinija Jain, and Aman Chadha. Exploring the
frontier of vision-language models: A survey of current methodologies and future directions. arXiv
preprint arXiv:2404.07214 , 2024.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented
language model pre-training. In International conference on machine learning , pp. 3929–3938.
PMLR, 2020.
Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang, and Nanyun Peng.
Mrag-bench: Vision-centric evaluation for retrieval-augmented multimodal models. In The Thir-
teenth International Conference on Learning Representations , 2025.
Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for
document ai with unified text and image masking. In Proceedings of the 30th ACM international
conference on multimedia , pp. 4083–4091, 2022.
Yulong Hui, Yao Lu, and Huanchen Zhang. Uda: A benchmark suite for retrieval augmented
generation in real-world document analysis. Advances in Neural Information Processing Systems ,
37:67200–67217, 2024.
Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. Videorag: Retrieval-augmented
generation over video corpus. In Findings of the Association for Computational Linguistics: ACL
2025 , pp. 21278–21298, 2025.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM
computing surveys , 55(12):1–38, 2023.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP
(1), pp. 6769–6781, 2020.
Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on
research and development in Information Retrieval , pp. 39–48, 2020.
9

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim,
Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document
understanding transformer. In European Conference on Computer Vision , pp. 498–517. Springer,
2022.
Reno Kriz, Kate Sanders, David Etter, Kenton Murray, Cameron Carpenter, Hannah Recknor, Jimena
Guallar-Blasco, Alexander Martin, Eugene Yang, and Benjamin Van Durme. Multivent 2.0: A
massive multilingual benchmark for event-centric video retrieval. In Proceedings of the Computer
Vision and Pattern Recognition Conference , pp. 24149–24158, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. Advances in neural information processing systems , 33:
9459–9474, 2020.
Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal
arxiv: A dataset for improving scientific comprehension of large vision-language models. In
Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers) , pp. 14369–14387, 2024.
Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao
Zheng, Fei Huang, Jingren Zhou, et al. Benchmarking multimodal retrieval augmented generation
with dynamic vqa dataset and self-adaptive planning agent. In The Thirteenth International
Conference on Learning Representations , 2025.
Weizhe Lin and Bill Byrne. Retrieval augmented visual question answering with outside knowledge.
InProceedings of the 2022 Conference on Empirical Methods in Natural Language Processing ,
pp. 11238–11254, 2022.
Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. Fine-grained late-
interaction multi-modal retrieval for retrieval augmented visual question answering. Advances in
Neural Information Processing Systems , 36:22820–22840, 2023.
Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo
Luo, and Rongrong Ji. Video-rag: Visually-aligned retrieval-augmented long video comprehension.
arXiv preprint arXiv:2411.13093 , 2024.
Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying multimodal
retrieval via document screenshot embedding. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pp. 6492–6505, 2024a.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan
Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding
with visualizations. Advances in Neural Information Processing Systems , 37:95963–96010, 2024b.
Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai.
Drvideo: Document retrieval based long video understanding. In Proceedings of the Computer
Vision and Pattern Recognition Conference , pp. 18936–18946, 2025.
Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual
question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf
conference on computer vision and pattern recognition , pp. 3195–3204, 2019.
Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark
for question answering about charts with visual and logical reasoning. In Findings of the Association
for Computational Linguistics: ACL 2022 , pp. 2263–2279, 2022.
10

Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha,
André Araujo, and Vittorio Ferrari. Encyclopedic vqa: Visual questions about detailed properties
of fine-grained categories. In Proceedings of the IEEE/CVF International Conference on Computer
Vision , pp. 3113–3124, 2023.
Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and Pratyush Kumar. Plotqa: Reasoning over
scientific plots. In Proceedings of the ieee/cvf winter conference on applications of computer
vision , pp. 1527–1536, 2020.
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
coding. arXiv preprint arXiv:1807.03748 , 2018.
Arnau Perez and Xavier Vizcaino. Advanced ingestion process powered by llm parsing for rag system.
arXiv preprint arXiv:2412.15262 , 2024.
Jingyuan Qi, Zhiyang Xu, Rulin Shao, Yang Chen, Jin Di, Yu Cheng, Qifan Wang, and Lifu Huang.
Rora-vlm: Robust retrieval-augmented vision language models. arXiv preprint arXiv:2410.08876 ,
2024a.
Zehan Qi, Rongwu Xu, Zhijiang Guo, Cunxiang Wang, Hao Zhang, and Wei Xu. Long2rag:
Evaluating long-context & long-form retrieval-augmented generation with key point recall. In
Findings of the Association for Computational Linguistics: EMNLP 2024 , pp. 4852–4872, 2024b.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning , pp.
8748–8763. PmLR, 2021.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and
Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association
for Computational Linguistics , 11:1316–1331, 2023.
Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray, Reno
Kriz, Celso M de Melo, Benjamin Van Durme, and Rama Chellappa. Video-colbert: Contextualized
late interaction for text-to-video retrieval. In Proceedings of the Computer Vision and Pattern
Recognition Conference , pp. 19691–19701, 2025.
Monica Riedler and Stefan Langer. Beyond text: Optimizing rag with multimodal inputs for industrial
applications. arXiv preprint arXiv:2410.21943 , 2024.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2:
Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022
Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies , pp. 3715–3734, 2022.
Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi.
A-okvqa: A benchmark for visual question answering using world knowledge. In European
conference on computer vision , pp. 146–162. Springer, 2022.
Ray Smith. An overview of the tesseract ocr engine. In Ninth international conference on document
analysis and recognition (ICDAR 2007) , volume 2, pp. 629–633. IEEE, 2007.
Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco,
Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: complex question answering over text,
tables and images. In International Conference on Learning Representations , 2021.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito.
Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings
of the AAAI Conference on Artificial Intelligence , volume 37, pp. 13636–13645, 2023.
11

Yang Tian, Fan Liu, Jingyuan Zhang, V . W., Yupeng Hu, and Liqiang Nie. CoRe-MMRAG: Cross-
source knowledge reconciliation for multimodal RAG. In Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers) , pp. 32967–32982, Vienna,
Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.
1583. URL https://aclanthology.org/2025.acl-long.1583/ .
Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for
multipage docvqa. Pattern Recognition , 144:109834, 2023.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée
Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
David Wan, Han Wang, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. Clamr: Contextualized
late-interaction for multimodal content retrieval. arXiv preprint arXiv:2506.06144 , 2025.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao.
Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents.
arXiv preprint arXiv:2502.18017 , 2025a.
Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei
Huang, and Feng Zhao. Vrag-rl: Empower vision-perception-based rag for visually rich in-
formation understanding via iterative reasoning with reinforcement learning. arXiv preprint
arXiv:2505.22019 , 2025b.
Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A
large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of
the IEEE/CVF international conference on computer vision , pp. 4581–4591, 2019.
Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay, and
Leonid Karlinsky. Real-mm-rag: A real-world multi-modal retrieval benchmark. arXiv preprint
arXiv:2502.12342 , 2025.
Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging
video and language. In Proceedings of the IEEE conference on computer vision and pattern
recognition , pp. 5288–5296, 2016.
Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. Layoutlm: Pre-training
of text and layout for document image understanding. In Proceedings of the 26th ACM SIGKDD
international conference on knowledge discovery & data mining , pp. 1192–1200, 2020.
Yibin Yan and Weidi Xie. Echosight: Advancing visual-language models with wiki knowledge. In
Findings of the Association for Computational Linguistics: EMNLP 2024 , pp. 1538–1551, 2024.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388 ,
2025a.
Jeff Yang, Duy-Khanh Vu, Minh-Tien Nguyen, Xuan-Quang Nguyen, Linh Nguyen, and Hung
Le. Superrag: Beyond rag with layout-aware graph modeling. arXiv preprint arXiv:2503.04790 ,
2025b.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality
documents. arXiv preprint arXiv:2410.10594 , 2024.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality
documents. In The Thirteenth International Conference on Learning Representations , 2025.
12

Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyusong Lee. Omagent: A multi-modal
agent framework for complex video understanding with task divide-and-conquer. In Proceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing , pp. 10031–10045,
2024a.
Tao Zhang, Ziqi Zhang, Zongyang Ma, Yuxin Chen, Zhongang Qi, Chunfeng Yuan, Bing Li, Junfu
Pu, Yuxuan Zhao, Zehua Xie, et al. mr2ag: Multimodal retrieval-reflection-augmented generation
for knowledge-based vqa. arXiv preprint arXiv:2411.15041 , 2024b.
13

A Prompt template
You are an AI specialized in recognizing and extracting text from images. Your mission is to analyze the image document and generate the result in QwenVLDocument Parser HTML format using specified tags while maintaining user privacy and data integrity.Image: {image_path}(a) Prompt for parsingimages
Your task is to answer the question based on the provided images. Please directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.Image: {image_path(s)}Question: {query}(b) Prompt for generating answers based on entire images
Your task is to answer the question based on the provided sub-images and their parsed text. Please directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.Image: {image_path(s)}Parsed Text: {parsed_text}Question: {query}(c) Prompt for generating answers based on sub-images and text
Your task is to answer the question based on the provided images and their parsed text. Please directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.Image: {image_path(s)}Parsed Text: {parsed_text}Question: {query}(d) Prompt for generating answers based on entire images and text
You are an expert evaluation system for a question answering chatbot.You are given the following information:-the query-a generated answer-a reference answerYour task is to evaluate the correctness of the generated answer.Query{q}Reference Answer{gold_ans}Generated Answer{gen_ans}Your response should be formatted as following:<judge>True or False</judge>If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.Please note that the generated answer may contain additional information beyond the reference answer.(e) Prompt for judging generated answers
Figure 3: Prompt templates for (a) parsing images, (b) generating answers based on entire images,
(c) generating answers based on sub-images and text, generating answers based on entire images
and text, and (e) judging generated answers. The first template can be found at https://github.
com/QwenLM/Qwen2.5VL/blob/main/cookbooks/document_parsing.ipynb and the rest can
be referred to (Wang et al., 2025b).
14
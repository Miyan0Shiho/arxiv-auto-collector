# Unlocking Multimodal Document Intelligence: From Current Triumphs to Future Frontiers of Visual Document Retrieval

**Authors**: Yibo Yan, Jiahao Huo, Guanbo Feng, Mingdong Ou, Yi Cao, Xin Zou, Shuliang Liu, Yuanhuiyi Lyu, Yu Huang, Jungang Li, Kening Zheng, Xu Zheng, Philip S. Yu, James Kwok, Xuming Hu

**Published**: 2026-02-23 15:27:41

**PDF URL**: [https://arxiv.org/pdf/2602.19961v1](https://arxiv.org/pdf/2602.19961v1)

## Abstract
With the rapid proliferation of multimodal information, Visual Document Retrieval (VDR) has emerged as a critical frontier in bridging the gap between unstructured visually rich data and precise information acquisition. Unlike traditional natural image retrieval, visual documents exhibit unique characteristics defined by dense textual content, intricate layouts, and fine-grained semantic dependencies. This paper presents the first comprehensive survey of the VDR landscape, specifically through the lens of the Multimodal Large Language Model (MLLM) era. We begin by examining the benchmark landscape, and subsequently dive into the methodological evolution, categorizing approaches into three primary aspects: multimodal embedding models, multimodal reranker models, and the integration of Retrieval-Augmented Generation (RAG) and Agentic systems for complex document intelligence. Finally, we identify persistent challenges and outline promising future directions, aiming to provide a clear roadmap for future multimodal document intelligence.

## Full Text


<!-- PDF content starts -->

Unlocking Multimodal Document Intelligence:
From Current Triumphs to Future Frontiers of Visual Document Retrieval
Yibo Yan1,2,3, Jiahao Huo1,2,4, Guanbo Feng1, Mingdong Ou2,*, Yi Cao2,
Xin Zou1,3,Shuliang Liu1,3,Yuanhuiyi Lyu1,3,Yu Huang1,2,Jungang Li1,Kening Zheng4,
Xu Zheng1,3,Philip S. Yu4,James Kwok3,Xuming Hu1,3,†
1Hong Kong University of Science and Technology (Guangzhou),2Alibaba Cloud Computing,
3Hong Kong University of Science and Technology,4University of Illinois Chicago
yanyibo70@gmail.com,xuminghu@hkust-gz.edu.cn
Abstract
With the rapid proliferation of multimodal in-
formation, Visual Document Retrieval (VDR)
has emerged as a critical frontier in bridging
the gap between unstructured visually rich data
and precise information acquisition. Unlike
traditional natural image retrieval, visual doc-
uments exhibit unique characteristics defined
by dense textual content, intricate layouts, and
fine-grained semantic dependencies. This pa-
per presents thefirst comprehensive survey
of the VDR landscape, specifically through
the lens of the Multimodal Large Language
Model (MLLM) era. We begin by examin-
ing the benchmark landscape, and subsequently
dive into the methodological evolution, cat-
egorizing approaches into three primary as-
pects: multimodalembedding models, multi-
modalreranker models, and the integration of
Retrieval-Augmented Generation(RAG) and
Agentic systemsfor complex document intel-
ligence. Finally, we identify persistent chal-
lenges and outline promising future directions,
aiming to provide a clear roadmap for future
multimodal document intelligence.
1 Introduction
Multimodal retrieval, the task of retrieving relevant
multimodal information from a large-scale collec-
tion using queries that span multiple modalities like
text and vision, has become a cornerstone of mod-
ern information retrieval (Mei et al., 2025; Zheng
et al., 2025a). Historically, research in this domain
has predominantly focused on natural image re-
trieval, targeting datasets of photographs and web
images where the primary goal is to match objects,
scenes, or holistic visual concepts (Wu et al., 2024a;
Arslan et al., 2024). However, both academia
and industry begin to turn their attention to a dis-
tinct yet ubiquitous data type:visual documents1.
*Project Lead
†Corresponding Author
1They are also commonly referred to asvisually rich docu-
ments,document images,etc. We use “visual documents” as a
CatSofa(Left) natural image
Query
A cat
sitting on 
a sofaChart
Notation(Right) visual document
Query
Method 
section of 
a paper
Main 
TextCaptionFigure 1:Comparison of retrieval of natural image (left) and
visual document (right), the focus of this survey.
These documents, ranging from scanned PDFs and
business reports to invoices and academic papers,
are characterized by a dense interplay of textual
content, complex layouts, and graphical elements
(Tang et al., 2023; Li et al., 2024d).
The pivot towardsVisual Document Retrieval
(VDR) is driven by three fundamental differences
that distinguish visual documents from natural
images, as illustrated in Figure 1. ❶Informa-
tion modality and density: unlike natural images
which convey semantic meaning through holistic
scenes, visual documents are hybrid entities where
meaning is co-determined by rich textual informa-
tion and a structured spatial layout. The informa-
tion is dense, hierarchical, and multi-modal by na-
ture.❷Semantic granularity: retrieval in natu-
ral images often targets high-level concepts (e.g.,
"a cat sitting on a sofa"), whereas VDR demands
a much finer-grained understanding. Users may
query for specific facts embedded within a table, a
particular sentence in a paragraph, or information
contingent on its document-level position (e.g.,"the
methodology section of a paper"). ❸User intent
and task complexity: VDR is typically geared
towards precise information-seeking, question an-
swering, and evidence-based reasoning, rather than
conceptual or aesthetic matching.
Furthermore, as the general capabilities of Multi-
modal Large Language Models (MLLMs) advance
(Song et al., 2025; Yan et al., 2025b,a), the VDR
unifying term. See more illustrative examples in Appendix A.
1arXiv:2602.19961v1  [cs.CL]  23 Feb 2026

field is increasingly focusing on their integration.
This includes the development of MLLM-based
embedding and reranker models to enhance se-
mantic matching (Tao et al., 2024; Zhang et al.,
2024b; Wang et al., 2025d). Beyond that, there
is active exploration into leveraging these mod-
els within more sophisticated frameworks like
Retrieval-Augmented Generation (RAG) pipelines
(Gao et al., 2023; Cheng et al., 2025; Gan et al.,
2025a) and Agentic systems (Singh et al., 2025) to
tackle complex document-based settings.
Scope.While several surveys have touched upon
related areas (as summarized in Table 1), a dedi-
cated, comprehensive analysis of VDR in the LLM
era has been conspicuously absent. Previous re-
views have largely concentrated on either tradi-
tional information retrieval (Alaei et al., 2016)
and general deep learning for document under-
standing (Subramani et al., 2020; Sassioui et al.,
2023; Ding et al., 2024), or retrieval for natural
images (Zhou et al., 2017; Hameed et al., 2021).
More recent surveys that acknowledge the rise
of MLLMs have continued this trend, focusing
on general document understanding (Huang et al.,
2024; Rombach and Fettke, 2025; Gao et al., 2025;
Ding et al., 2025; Ke et al., 2025) or applying
MLLMs to natural image retrieval (Zhao et al.,
2023; Zhang et al., 2025c). To the best of our
knowledge, no existing work provides a system-
atic overview of the VDR landscape through the
specific lens of retrieval-focused methodologies in
the age of LLMs, especially covering the emerging
paradigms of RAG and Agent-based systems. Our
survey aims to bridge this critical gap, offering the
first comprehensive treatise on VDR that syn-
thesizes foundational techniques with the latest
breakthroughs driven by (M)LLMs2.
Structure.We begin from abenchmark per-
spective, systematically organizing the field by
examining task formulations, basic settings, and
dataset characteristics such as multilingual sup-
port and the growing emphasis on reasoning-
intensive queries. Following this, we transition
to amethodology-centric analysis, categorizing
existing approaches into three primary paradigms:
❶embedding models that serve as the foundation
for retrieval, ❷reranker models designed to re-
fine initial retrieval results, and ❸the increasingly
prominent RAG pipelines and agentic systems. Fi-
2This version covers literature up to January 5, 2026, with
updates scheduled every 3-4 months. We invite authors of
relevant works to contact the first author via email.Survey Venue Scope SettingTechnical Trends
LLM RAG Agent
(Alaei et al., 2016) IJCNN’16 IR4Doc R
(Zhou et al., 2017) arxiv’17 IR4Img R
(Subramani et al., 2020) NeurIPS Workshop’20 DL4Doc U
(Hameed et al., 2021) CE’21 IR4Img R
(Cui et al., 2021) ICDAR’21 IR4Doc U
(Sassioui et al., 2023) WINCOM’23 IR4Doc U
(Zhao et al., 2023) EMNLP Finding’23 LLM4Img R ✔ ✔
(Ding et al., 2024) arxiv’24 DL4Doc U ✔
(Huang et al., 2024) IEEE TKDE’24 LLM4Doc U ✔ ✔
(Rombach and Fettke, 2025) ACM Computing Survey’24 DL4Doc U ✔
(Zhang et al., 2025c) arxiv’25 LLM4Img R ✔
(Gao et al., 2025) arxiv’25 LLM4Doc U ✔ ✔ ✔
(Ding et al., 2025) IJCAI Tutorial’25 LLM4Doc U ✔
(Ke et al., 2025) ACM TOIS’25 LLM4Doc U ✔ ✔ ✔
Ours-LLM4Doc R ✔ ✔ ✔
Table 1:Comparisons between relevant surveys & ours. We
denote R etrieval as Rand U nderstanding as U.
nally, we conclude by discussing the challenges
and outliningfuture frontiers, aiming to provide
valuable insights and inspire subsequent research in
the multimodal document intelligence community.
2 Benchmark Perspective
This section provides a systematic review of VDR
evaluation landscape. We first establish a formal
mathematical definition ( ▷Section 2.1), and then
analyze the current trends in terms of data scale and
metrics ( ▷Section 2.2), followed by a discussion
on the emerging frontiers of multilingual support
(▷Section 2.3) and reasoning-intensive retrieval
(▷Section 2.4), which reflect the shift from key-
word matching to complex document intelligence.
2.1 Formulation
VDR aims to identify the most relevant document
images from a large-scale corpus based on a given
query. Formally, let C={d 1, d2, . . . , d N}be a
corpus of Ndocument pages, where each dj∈ I
is a visually rich document image. Given a query
q, the task is to produce a ranked list of documents
fromCsuch that the top-ranked items maximize a
relevance scores(q, d).
Input and Modalities.In the standard VDR set-
ting, the query is typically a natural language string
q∈ T (text-to-image retrieval). However, the gen-
eralized VDR setting extends the query space to
multimodal inputs, including images or interleaved
text-and-image sequences q∈ {T ∪ I}∗. Unlike
traditional OCR-based retrieval that treats docu-
ments as plain text, VDR models process the doc-
ument ddirectly in the visual domain, often rep-
resenting it as a set of patch-level embeddings to
preserve layout and graphical information.
Mathematical Representation.Following
the late-interaction paradigm pioneered by Col-
Pali (Faysse et al., 2024), a document page can
be represented as a set of multiple patch-level
embeddings D={d j}Np
j=1, where dj∈RD,
2

Category BenchmarkPublication DatasetResource
Team Venue #Query* #Corpus* Multilingual Reasoning Retrieval Metric
VDR ViDoRe-V1 (Faysse et al., 2024) Illuin Tech ICLR’25 3.81k 8.31k en/fr - nDCG / Recall / MRR
VDR VisRAG (Yu et al., 2024) Tsinghua ICLR’25 3.61k 20.64k en - Recall / MRR
VDR SeaDoc (Xiao et al., 2025b) Alibaba NeurIPS’25 1.00k 5.06k 4 - nDCG
VDR Real-MM-RAG (Wasserman et al., 2025b) IBM ACL’25 4.55k 8.60k en - nDCG / Recall
VDR NL-DIR (Guo et al., 2025) CAS CVPR’25 205.00k 41.80k en - Recall / MRR
VDR OpenDocVQA (Tanaka et al., 2025) NTT CVPR’25 43.47k 206.27k en - nDCG
VDR ViDoSeek (Wang et al., 2025b) Alibaba EMNLP’25 1.14k 5.39k en - Recall / MRR
VDR MMDocIR (Dong et al., 2025a) Huawei EMNLP’25 1.66k 20.40k en - Recall
VDR VisDoMBench (Suri et al., 2025) UMD NAACL’25 2.27k 8.30k en - ANLCS
VDR ViDoRe-V2 (Macé et al., 2025) Illuin Tech arxiv’25 1.15k 4.54k 4 - nDCG
VDR Jina-VDR (Günther et al., 2025) Jina AI arxiv’25 57.33k 70.62k 20 - nDCG
VDR EVisRAG (Sun et al., 2025b) PKU arxiv’25 4.26k - en - Acc / F1
VDR MMDocRAG (Dong et al., 2025b) Huawei arxiv’25 4.06k 14.87k en - Recall
VDR Double-Bench (Shen et al., 2025) SCUT arxiv’25 5.17k 73.06k 6 - HR
VDR VisR-Bench (Chen et al., 2025c) UB arxiv’25 35.57k 35.63k 16 - Acc
VDR MR2-Bench (Zhou et al., 2025a) BAAI arxiv’25 1.31k 47.74k en✔nDCG / Recall
VDR MIRACL-VISION (Osmulski et al., 2025) NVIDIA arxiv’25 7.90k 338.73k 18 - nDCG
VDR UniDoc-Bench (Peng et al., 2025) Salesforce arxiv’25 1.74k 70.00k en - Recall / Precision
VDR MRMR (Zhang et al., 2025e) NTU arxiv’25 1.50k 42.80k en✔nDCG / HR
VDR ViMDoc (Kim et al., 2025a) KAIST arxiv’25 10.90k 76.35k en - Recall
VDR M4DocBench (Dong et al., 2025c) Huawei arxiv’25 158 6.18k en, zh✔Recall
VDR SDS KoPub VDR (Lee et al., 2025a) Samsung arxiv’25 600 40.78k en, ko - nDCG / Recall
VDR Nayana-IR (Kolavi and Jain, 2025) CognitiveLab arxiv’25 1.00k 5.40k 22 - nDCG / Recall / mAP / MRR
VDR ViDoRe-V3 (ILLUIN, 2025) Illuin Tech blog’25 3.10k 26.00k en/fr - nDCG
VDR MMLongBench-Doc (Ma et al., 2024b) NTU NeurIPS’24 1.08k 6.41k en - Acc / F1
Gen. MMEB (Jiang et al., 2024c) Waterloo & Salesforce ICLR’25 12.00k 12.00M en - Precision
Gen. Visual Haystacks (Wu et al., 2024b) UCB ICLR’25 20.10k 97.00k en - Acc
Gen. COCO-Facet (Li et al., 2025c) Washington NeurIPS’25 9.11k 911.20k en - Recall
Gen. UMRB (Zhang et al., 2024c) PolyU & Alibaba CVPR’25 200.00k 40.00M en - nDCG / Recall
Gen. MIEB (Xiao et al., 2025c) Durham University ICCV’25 1.25M 7.62M 38 - nDCG / Recall / mAP
Gen. MMNeedle (Wang et al., 2025a) Rutgers University NAACL’25 280.00k 40.00k en - Acc
Gen. MMMEB (Musacchio et al., 2025) Uniba arxiv’25 9.20k 9.20k 5 - Precision
Gen. MMEB-V2 (Meng et al., 2025) Salesforce arxiv’25 - - en - nDCG
Gen. MM-NIAH (Wang et al., 2024b) FDU & Shanghai AI Lab NeurIPS’24 5.58k - en - Acc
Gen. M2KR (Lin et al., 2024b) Cambridge ACL’24 39.79k 514.85k en - Recall
Gen. M-BEIR (Wei et al., 2024a) Waterloo ECCV’24 190k 5.6M en - Recall
Table 2:Comparison of VDR and general multimodal retrieval benchmarks. InResourcecolumn, we denote the corresponding
github codebase/huggingface/paper(e.g.,arxiv paper, technical report or blog) as
 /
/
. * indicates that #query and #corpus
corresponds to retrieval-related #valuation sample and #candidate, respectively, for general multimodal retrieval benchmarks.
and the query as a set of token-level embeddings
Q={q i}Nq
i=1. Relevance thus can be then com-
puted via a late-interaction mechanism like MaxSim
operation:s(q, d) =PNq
i=1maxNp
j=1q⊤
idj.
Evaluation Metrics.Performance is typically
measured using standard Information Retrieval met-
rics, including: (i)Recall@ k(R@k), proportion
of queries for which the relevant document is in the
top-kresults; (ii)nDCG, which measures ranking
quality by accounting for the position of relevant
documents; and (iii)MRR, which evaluates the
average of reciprocal ranks of the first relevant doc-
ument. See more details in Appendix B.
2.2 Basic Setting
Escalating Research Momentum.Over the past
two years, VDR has transitioned from a niche task
to a central focus in both industry and academia.
As shown in Table 2, the majority of specialized
VDR benchmarks, such as ViDoRe seires (Faysse
et al., 2024) and Real-MM-RAG (Wasserman et al.,
2025b), emerged in 2024 and 2025. This surge
is driven by the realization that (OCR-based) text-
only retrieval fails to capture the visual nuances
of documents, such as tables, charts, and spatial
hierarchies (Zhang et al., 2025b; Most et al., 2025).Diverse Dataset Scales.Current bench-
marks exhibit a wide range of data magni-
tudes (Cao et al., 2025). While expert-annotated
sets like SeaDoc (Xiao et al., 2025b) and
M4DocBench (Dong et al., 2025c) focus on special-
ized samples, recent large-scale efforts have pushed
boundaries. NL-DIR (Guo et al., 2025) provides
205k queries, and Jina-VDR (Macé et al., 2025) uti-
lizes over 70k documents, reflecting a trend toward
scaling both query volume and corpus diversity.
Standardized Evaluation.nDCG and Recall re-
main the primary metrics for performance measure-
ment. However, as VDR is increasingly integrated
into RAG, some benchmarks (e.g.,EVisRAG (Sun
et al., 2025b) and MMLongBench-Doc (Ma et al.,
2024b)) incorporate downstream Accuracy and F1
scores to evaluate how retrieval quality directly
impacts final multimodal generation.
2.3 Multilingual Support
The majority of early VDR benchmarks are pre-
dominantly English-centric. However, recent work
has begun to bridge this linguistic gap like multi-
lingual text embedding benchmarks (Enevoldsen
et al., 2025; Zhang et al., 2023). Jina-VDR (Macé
et al., 2025) and Nayana-IR (Kolavi and Jain, 2025)
3

represent a significant shift, supporting 20 and
22 languages respectively. Similarly, MIRACL-
VISION (Osmulski et al., 2025) introduces a large-
scale multilingual corpus covering 18 languages.
2.4 Reasoning-Intensive Setting
The rapid advancement of reasoning capabilities
in MLLM domain is inspiring a new frontier
for VDR benchmarks that moves beyond sim-
ple semantic matching toward complex reason-
ing: MR2-Bench (Zhou et al., 2025a) first moves
"beyond matching" by introducing tasks requir-
ing abstract, spatial, and analogical reasoning.
MRMR (Zhang et al., 2025e) then elevates the chal-
lenge by situating reasoning within expert-level
domains and introducing novel logical formula-
tions like Contradiction Retrieval. Most recently,
M4DocBench (Dong et al., 2025c) has expanded
the paradigm to assess multi-hop, multi-document
synthesis within agentic "deep research" workflows.
We leave more discussion in Appendix C.
3 Methodology Perspective
This section deconstructs the technical trends that
underpin modern VDR. We begin by examining
embedding models ( ▷Section 3.1), as summarized
in Table 3. Next, we analyze reranker models ( ▷
Section 3.2), as shown in Table 4. Finally, we
discuss how they are integrated into sophisticated
RAG and Agentic systems (▷Section 3.3).
3.1 Embedding Models
3.1.1 Formulation
In the context of VDR, an embedding model, de-
noted by an encoder function E(·), processes mul-
timodal inputs to produce vector representations.
For a given query q, typically tokenized into a
sequence Q={tq
1, tq
2, . . . , tq
M}, and a document
paged, represented as a sequence of image patches
D={pd
1, pd
2, . . . , pd
N}, the objective is to learn
an encoder that aligns their representations. While
early models produced a single vector for each item,
recent VDR models adopt a multi-vector paradigm.
The relevance score s(q, d) between the query em-
bedding set E(Q) ={eq
ti}M
i=1and the document
patch embedding set E(D) ={ed
pj}N
j=1is often
computed using a late-interaction mechanism like
MaxSim. The primary distinction between special-
ized VDR models and general multimodal embed-
ding models lies in their training data. We discuss
common VDR training set in Appendix D.3.1.2 Techinical Trends
Model Choices.The field has witnessed a clear
trend in both model scale and architecture. While
early models were based on smaller backbones
like BERT (Nussbaum et al., 2024), current VDR
embeddings are predominantly built on powerful
MLLMs, such as PaliGemma (Beyer et al., 2024)
and Qwen-VL series (Bai et al., 2025). parame-
ter counts also have escalated, with SOTA models
typically ranging from 2 to 8 billion parameters.
Multilingual Support.Unlike general-domain
image retrieval where this aspect is less empha-
sized, the prevalence of multilingual business re-
ports, academic papers, and official forms necessi-
tates cross-lingual understanding. Recent models
like jina-embeddings-v4 (Günther et al., 2025) and
Nemoretriever (Xu et al., 2025a) reflect this shift,
offering support for over 20 languages and demon-
strating strong performance on multilingual VDR
benchmarks such as Jina-VDR.
(a) OCR -based
Adaptive Pruning
(b) LVLM -based
Extraction via OCR tools
Cannot preserve layout 
& structural relationship
Page -Level
Cannot  capture  
fine-grained  infoPatch -Level
…
 …Query
MaxSim…
Offline  Phase Online  PhaseProhibitive storage  overhead
(c) DocPruner (Ours)
DocPrunerDiverse Documents TypesDocPruner
DocPrunerPruning%: 
High
Pruning%: 
Medium
Pruning%: 
Low
(Left) Single -Vec
Single -Vec Multi -Vec
Figure 2:Single-vec (left) and multi-vec (right) VDR.
Multi-Vector Representation.The multi-vector
paradigm, popularized in the VDR domain by Col-
Pali (Faysse et al., 2024), has become a dominant
approach for fine-grained retrieval, as shown in
Figure 2. This granular representation is partic-
ularly effective for VDR because it enables "late
interaction" matching (Khattab and Zaharia, 2020),
where specific phrases in a query can be precisely
aligned with corresponding visual or textual re-
gions in a document. ColPali (Faysse et al., 2024)
adapts PaliGemma to produce multi-vector outputs
by treating the embeddings of individual image
patches as distinct vectors for late interaction. Pre-
FLMR (Lin et al., 2024b) generates its multi-vector
representations by concatenating embeddings from
text tokens with both global and patch-level vi-
sual features, which are further refined through
cross-attention to be query-aware. Addressing effi-
ciency, ColModernVBERT (Teiletche et al., 2025)
demonstrates that a compact, bidirectional lan-
guage encoder can be effectively aligned with a
vision encoder to generate granular embeddings;
while MetaEmbed (Xiao et al., 2025d) introduces
4

Category EmbeddingPublication ModelNovelty Resource
Team Venue Para. Backbone Multilingual Multi-Vec Training-Free Generative
VDR ColPali/ColQwen (Faysse et al., 2024) Illuin Tech ICLR’25 3B PaliGemma en✔- - MDT
VDR Unveil (Sun et al., 2025a) PKU ACL’25 3B MiniCPM-V en - - - M
VDR ColMate (Masry et al., 2025) ServiceNow EMNLP’25 3B PaliGemma en/fr✔- - M
VDR ColFlor (Masry and Hoque, 2025) York Uni. arxiv’25 0.2B Florence-2 en/fr✔- - E
VDR ColModernVBERT (Teiletche et al., 2025) Illuin Tech arxiv’25 0.3B SigLIP en✔- - ME
VDR Nemoretriever Colembed (Xu et al., 2025a) NVIDIA arxiv’25 1B/3B Llama-3.2 18✔- - M
VDR jina-embeddings-v4 (Günther et al., 2025) Jina AI arxiv’25 4B Qwen2.5-VL 20✔- - MT
VDR Tomoro-ColQwen (Huang and Tan, 2025) Tomoro AI arxiv’25 4B/8B Qwen2.5/3 en/fr✔- - E
VDR ColNetraEmbed (Kolavi and Jain, 2025) CognitiveLab arxiv’25 4B Gemma 3 22✔- - D
VDR Granite-vision-embedding (Team et al., 2025a) IBM arxiv’25 2B Granite-3.1 en - - - DE
VDR vdr-2b-multi-v1 (LlamaIndex, 2025) LlamaIndex blog’25 2B Qwen2VL 5 - - - DE
VDR Eager Embed (Balarini, 2025) Eagerworks blog’25 4B Qwen2.5-VL en - - - -
VDR ColNomic Embed Multimodal (Team, 2025) Nomic AI blog’25 3B/7B Qwen2.5-VL 5✔- - DT
VDR DSE (Ma et al., 2024a) Waterloo EMNLP’24 4B Phi-3-Vision en - - - D
VDR Nomic Embed Vision (Nussbaum et al., 2024) Nomic AI arxiv’24 0.2B BERT en - - - -
Gen. UniME-V2 (Gu et al., 2025b) MiroMind AAAI’26 2B/7B Qwen2-VL/LLaV A en - - - DT
Gen. MM-Embed (Lin et al., 2024a) NVIDIA ICLR’25 7B LLaV A-NeXT en - - - T
Gen. VLM2Vec (Jiang et al., 2024c) Waterloo & Salesforce ICLR’25 8B Phi-3.5-V en - - - MD
Gen. B3 (Thirukovalluru et al., 2025) Duke NeurIPS’25 2B/7B/8B Qwen2-VL/InternVL3 en - - - TE
Gen. LCO-Embedding (Xiao et al., 2025b) Alibaba NeurIPS’25 3B/7B LLaV A/Qwen2.5-VL 38 - - - M
Gen. Retrv-R1 (Zhu et al., 2025b) CityU HK NeurIPS’25 3B/7B Qwen2.5-VL en - -✔ MDT
Gen. MMRet-MLLM (Zhou et al., 2025b) BUPT ACL’25 7B LLaV A-1.6 en - - - D
Gen. UniSE (Liu et al., 2025h) BAAI ACL’25 0.4/2B CLIP/Qwen2-VL en - - - MD
Gen. GME (Zhang et al., 2024c) PolyU & Alibaba CVPR’25 7B Qwen2-VL en - - - D
Gen. LamRA (Liu et al., 2025e) SJTU & Xiaohongshu CVPR’25 7B Qwen2-VL en - - - M
Gen. VladV A (Ouali et al., 2025) Samsung CVPR’25 7B LLaV A-1.5 en - -✔ M
Gen. CAFe (Yu et al., 2025a) Meta ICCV’25 0.5B/7B LLaV A-OV en - -✔ T
Gen. UniME-V1 (Gu et al., 2025a) Sydney MM’25 4B/7B Phi-3.5/LLaV A en - - - T
Gen. MetaEmbed (Xiao et al., 2025d) Meta arxiv’25 3-32B Qwen2.5/Llama-3.2 en/fr✔- - ME
Gen. CoMa (Li et al., 2025a) Kuaishou arxiv’25 3B/7B Qwen2.5-VL en - - - ME
Gen. FreeRet (Zhu et al., 2025c) NJU arxiv’25 2-32B Qwen2/2.5-VL/InternVL3 en -✔- M
Gen. LLaVE (Lan et al., 2025a) Tencent arxiv’25 0.5B/7B LLaV A/Aquila en - - - T
Gen. mmE5 (Chen et al., 2025b) RUC arxiv’25 11B Llama-3.2 93 - - - D
Gen. MoCa (Chen et al., 2025a) RUC arxiv’25 3B/7B Qwen2.5-VL en/fr - - - MT
Gen. PDF (Wang et al., 2025f) Alibaba arxiv’25 2B/7B LLaV A/Qwen2-VL en - - - MT
Gen. QQMM (Xue et al., 2025) Tencent arxiv’25 7B LLaV A/Qwen2-VL en - - - T
Gen. ReMatch (Liu et al., 2025d) Xiaohongshu arxiv’25 2B/7B Qwen2/2.5-VL en - - - M
Gen. RGE (Liu et al., 2025a) SenseTime arxiv’25 3B Qwen2.5-VL en - -✔ MT
Gen. RzenEmbed (Jian et al., 2025) 360 AI arxiv’25 2B/8B Qwen2-VL en - - - DT
Gen. TTE (Cui et al., 2025) Meta arxiv’25 2B/7B Qwen2-VL en/fr - -✔ M
Gen. U-MARVEL (Li et al., 2025d) NJU arxiv’25 7B Qwen2-VL en - - - MT
Gen. UME-R1 (Lan et al., 2025b) Tencent arxiv’25 2B/7B Qwen2-VL en - -✔ MT
Gen. Unite (Kong et al., 2025) Kuaishou arxiv’25 2B/7B Qwen2-VL en - - - DT
Gen. VIRTUE (Wang et al., 2025e) Sony arxiv’25 2B/7B Qwen2-VL en - - - MD
Gen. VLM2Vec-V2 (Meng et al., 2025) Salesforce arxiv’25 2B Qwen2-VL en/fr - - - D
Gen. xVLM2Vec (Musacchio et al., 2025) UBAM arxiv’25 4B Phi-3.5-V 5 - - - DT
Gen. SAIL-Embedding (Lin et al., 2025a) ByteDance arxiv’25 - SAIL-VL en - - - DT
Gen. Ops-MM-embedding (Alibaba, 2025b) Alibaba blog’25 2B/7B Qwen2-VL en - - - D
Gen. EvoQwen2.5-VL-Retriever (Alibaba, 2025a) Alibaba blog’25 3B/7B Qwen2.5-VL en✔- - D
Gen. Seed1.6-Embedding (ByteDance, 2025) ByteDance blog’25 - Seed1.6-flash en - - - D
Gen. PreFLMR (Lin et al., 2024b) Cambridge ACL’24 2B BERT en✔- - MD
Gen. Vista (Zhou et al., 2024) BUPT ACL’24 0.2B BGE-v1.5 en - - - MD
Gen. UniIR (Wei et al., 2024a) Waterloo ECCV’24 0.4B CLIP en - - - D
Gen. M-Solomon (Kim et al., 2025b) NC AI CIKM’24 7B Qwen2-VL en - - - M
Gen. E5-V (Jiang et al., 2024b) Beihang arxiv’24 8B LLaV A-NeXT en -✔- M
Table 3:Comparison of VDR and general multimodal embedding models. InNoveltycolumn, we denote Model-/ Data-
/Training-/ Efficiency-level contribution as M/D/T/E. InResourcecolumn, we denote the corresponding github
codebase/huggingface/paper(e.g.,arxiv paper, technical report or blog) as
 /
/
.
a fixed set of learnable "Meta Tokens" whose final
hidden states serve as a compact and scalable multi-
vector representation, enabling flexible trade-offs
between retrieval quality and efficiency at test-time.
Training Paradigm Exploration.The predomi-
nant training method for VDR embedding models
is end-to-end supervised fine-tuning using a con-
trastive loss, such as InfoNCE (Oord et al., 2018).
This objective pulls positive query-document pairs
closer together in the embedding space while push-
ing negative pairs apart. Beyond this standard ap-
proach, two novel paradigms are gaining traction.
The first paradigm explorestraining-free meth-
ods, which leverage the inherent knowledge of
pre-trained MLLMs. E5-V (Jiang et al., 2024b)
pioneers this by using carefully designed prompts
to elicit universal embeddings directly from the
MLLM’s vocabulary space. FreeRet (Zhu et al.,2025c) introduces a plug-and-play framework that
uses off-the-shelf MLLMs for both embedding-
based search and multiple-choice question based
reranking, all without any parameter updates.
The second paradigm explores how to harness
thegenerative capabilitiesof MLLMs to enhance
retrieval. Early works such as CAFe (Yu et al.,
2025a) and VladV A (Ouali et al., 2025) introduced
hybrid training frameworks; they jointly optimize
a contrastive loss for discriminative power with an
autoregressive, next-token prediction loss to pre-
serve and leverage the model’s inherent generative
abilities. More recent approaches explicitly inte-
grate reasoning as a precursor to embedding. For
example, RGE (Liu et al., 2025a) and TTE (Cui
et al., 2025) both propose a "think-then-embed"
process, where the model first generates an explicit
rationale or reasoning trace, and the final embed-
ding is conditioned on this generated context to
5

capture more nuanced semantics. Taking this a
step further, Retrv-R1 (Zhu et al., 2025b) employs
reinforcement learning to optimize a step-by-step
reasoning process, framing retrieval as a reasoning-
driven decision-making task.
3.1.3 Technical Innovations
Recent advancements in VDR embedding models
can be broadly categorized as follows:
❶Model-level:Innovations in this area focus on
designingnovel architectures and interaction
mechanisms. A pioneering example is ColPali
(Faysse et al., 2024), which first adapts the late-
interaction mechanism to operate directly on doc-
ument pageimages, enabling precise alignment
between query and visual patches without a brit-
tle OCR pipeline. Unveil (Sun et al., 2025a)
introduces a hybrid visual-textual teacher model
and then uses knowledge distillation to transfer
its comprehensive understanding to an efficient,
OCR-free visual-only student model.
❷Data-level:This includes not only creatinglarge-
scale, high-quality training datasetsbut also de-
velopingsophisticated data synthesis and hard
negative mining strategies. VLM2Vec (Jiang
et al., 2024c) introduces MMEB benchmark,
which unifies multimodal tasks into a univer-
sal ranking format. In terms of negative min-
ing strategies, UniME-V2 (Gu et al., 2025b) pro-
poses an "MLLM-as-a-Judge" mechanism that
leverages MLLMs to assess retrieved candidates
and generate soft semantic matching scores.
❸Training-level:Advancements here involve ex-
ploringnovel training objectivesandflexible
paradigms beyond standard contrastive loss.
jina-embedding-v4 (Günther et al., 2025) imple-
ments a unified multi-task learning framework
that simultaneously trains a model to produce
both single-vector and multi-vector embeddings,
while using LoRA adapters to optimize perfor-
mance for different retrieval scenarios. MM-
Embed (Lin et al., 2024a) introduces a modality-
aware training strategy, which explicitly samples
negatives that are semantically similar but have
the incorrect modality, complemented by a con-
tinuous fine-tuning schedule to balance multi-
modal and text-only retrieval.
❹Efficiency-level:This line of work challenges
the "bigger is better" assumption by developing
smaller yet powerful modelsandmore efficient
training methods. ModernVBERT (Teiletcheet al., 2025) demonstrates a compact (250M)
model based on a bidirectional encoder archi-
tecture can outperform much larger (e.g.,>3B)
decoder-based VLMs. On training efficiency
front, B3 (Thirukovalluru et al., 2025) introduces
a smart batch mining strategy that pre-processes
the entire dataset using graph-based detection to
construct batches rich in mutual hard negatives.
3.2 Reranker Models
3.2.1 Formulation
A reranker model, denoted as R(·,·) , operates as
a cross-encoder. It takes a query qand a sin-
gle candidate document dcfrom the initial re-
trieval stage as a combined input. By jointly
processing the query’s token sequence Q=
{tq
1, . . . , tq
M}and the document’s patch sequence
Dc={pdc
1, . . . , pdc
N}, the model can leverage deep
cross-attention mechanisms to capture fine-grained
inter-modal dependencies. The output is a sin-
gle scalar relevance score, srerank , which is used
to re-sort the candidate list: srerank =R(q, d c) =
R({tq
1, . . . , tq
M},{pdc
1, . . . , pdc
N}).This deep inter-
action allows the model to make more accurate
relevance judgments than the dot-product-based
similarity used in bi-encoder.
3.2.2 Technical Trends
Model Choices.Similar to embedding models,
the trend in rerankers is towards larger, MLLM-
based architectures. Models such as UniME-V2-
Reranker (Gu et al., 2025b) and LamRA-Rank (Liu
et al., 2025e) utilize powerful Qwen2.5-VL with
parameter counts reaching 7 billion.
Multilingual Support.In stark contrast to the
rapid adoption of multilingualism in VDR embed-
ding models, this capability remains largely under-
developed in the reranker space. The vast majority
of existing multimodal rerankers, including DocRe-
Rank (Wasserman et al., 2025a) and MonoQwen2-
VL-v0.1 (Chaffin and Lac, 2024), are English-only.
The only exception is jina-reranker-m0 (Jina AI,
2025), which supports 29 languages.
3.2.3 Reranking Paradigms
Reranker models are typically trained using one or
a combination of three main paradigms:
❶Pointwise:This approach evaluates each query-
document pair (q, d i)independently, predicting
an absolute relevance score. The model is trained
using a loss function like Binary Cross-Entropy
6

RerankerPublication ModelBenchmark* Resource
Team Venue Para. Backbone Ranking Multilingual
UniME-V2-Reranker (Gu et al., 2025b) MiroMind AI AAAI’26 7B Qwen2.5-VL PA LI en MMEB
LamRA-Rank (Liu et al., 2025e) SJTU & Xiaohongshu CVPR’25 7B Qwen2.5-VL PO LI en M-BEIR
DocReRank (Wasserman et al., 2025a) WIS EMNLP25 2B Qwen2-VL PO en ViDoRe-v2/Real-MM-RAG
RagVL (Chen et al., 2024) IDEA EMNLP Finding’25 1B/2B/4B/13B Qwen-VL/InternVL/LLaV A-v1.5 PO en -
Lychee-rerank-mm (Dai et al., 2025) HIT arxiv’25 7B Qwen2.5-VL PO en MRB/MRMR
MM-R5 (Xu et al., 2025b) DP Tech arxiv’25 7B Qwen2.5-VL LI en MMDocIR
jina-reranker-m0 (Jina AI, 2025) Jina AI blog’25 2.4B Qwen2-VL PO 29 ViDoRe-v1/M-BEIR
MonoQwen2-VL-v0.1 (Chaffin and Lac, 2024) LightOn blog’24 2B Qwen2-VL PO en ViDoRe-v1
Table 4:Comparison of multimodal (document) reranker models. * indicates that only VDR-related benchmarks evaluated.
InRankingcolumn, we denote POintwise/ PAirwise/ LIstwise design as PO/PA/LI. InResourcecolumn, we denote the
corresponding github codebase/huggingface/paper(e.g.,arxiv paper, technical report or blog) as
 /
/
.
(BCE) on the predicted score si=R(q, d i)
against a ground-truth label yi∈ {0,1} . It can
be formulated asL pointwise =P
iBCE(s i, yi).
❷Pairwise:This method learns relative relevance
by comparing pairs. Given a query qand a pair
of documents (di, dj)where diis more relevant
thandj, the model is trained with a margin-based
loss to ensure R(q, d i)> R(q, d j), formulated
asL pairwise =P
yi>yjmax(0, m−(s i−sj)).
❸Listwise:This paradigm considers the entire
list of candidate documents for a query simul-
taneously. It optimizes a loss function, such as
ListNet’s softmax cross-entropy, that directly cor-
responds to ranking metrics like nDCG, learning
to predict the optimal ordering of the full list,
formulated as Llistwise =−P
iP(di) log( ˆP(di)),
where P and ˆPare ground-truth and predicted
probability distributions over the list.
Recent rerankers have shown that combining mul-
tiple loss functions often yields superior perfor-
mance. LamRA-Rank (Liu et al., 2025e) and
UniME-V2-Reranker (Gu et al., 2025b) adopt hy-
brid training strategies that combine listwise op-
timization, where the model predicts the correct
item’s position, with pointwise or pairwise ob-
jectives that classify the relevance of individual
or paired candidates. Other approaches refine
a single paradigm to great effect; models like
Lychee-rerank-mm (Dai et al., 2025) and DocR-
eRank (Wasserman et al., 2025a) utilize super-
vised fine-tuning to frame reranking as a point-
wise classification task, predicting "yes" or "no"
to align with the generative nature of MLLMs.
Bsides, MM-R5 (Xu et al., 2025b) advances the
listwise paradigm by incorporating CoT reason-
ing and leveraging reinforcement learning with a
task-specific reward to optimize the ranked output.
3.3 RAG Pipeline & Agentic System
3.3.1 Formulation
The integration of a VDR embedding model E(·)
and a reranker R(·,·) into a RAG pipeline can beformulated as a multi-stage process. Given a query
qand a corpusC, the process unfolds as follows:
❶First-Stage Retrieval:An embedding model
Eis used to efficiently retrieve an initial set of
kcandidate documents Ckfrom the corpus C
based on embedding similarity. It can formu-
lated as:C k=Top-k
d∈C(s(E(q), E(d))).
❷Second-Stage Reranking:A reranker model
Rthen refines this candidate set by computing
a more accurate relevance score for each docu-
ment, producing a final ranked list C′
j(where
j≤k), formulated asC′
j=Top-j
d∈C k(R(q, d)).
❸Augmented Generation:Finally, a generative
MLLM, G(·), synthesizes an answer aby con-
ditioning on both the original query qand the
context provided by the retrieved and reranked
documentsC′
j, formulated asa=G(q, C′
j).
In anAgentic system, this process becomes dy-
namic and iterative. An agent Auses the VDR
model as a tool. At each step t, based on the query
qand internal state (or history) ht, the agent gen-
erates an action actt=A(q, h t). If the action is a
retrieval query qt, the VDR system is invoked to re-
trieve evidence Ct=Retrieve(q t). The agent then
updates its state ht+1=ht∪ {q t, Ct}and decides
on the next action.
3.3.2 Current Paradigms and Key Trends
The integration of VDR into RAG and agentic sys-
tems is rapidly evolving from static retrieval to
dynamic, multi-step reasoning. The foundational
paradigm shift involvedmoving from OCR-based
methods to end-to-end multimodal pipelinesthat
directly process document images, as exemplified
by M3DocRAG (Cho et al., 2024). Building on
this, the scope of interaction hasexpanded beyond
textto include more natural modalities like speech
in frameworks such as TextlessRAG (Xie et al.,
2025). The dominant trend, however, is the rise
of agentic systems thatmimic human research
7

workflows. These systems often feature a "soci-
ety of agents" for collaborative task decomposition,
where specialized agents for text, vision, and crit-
ical analysis work in concert, as seen in MDocA-
gent (Han et al., 2025b), or follow a coarse-to-fine
refinement loop with "seeker" and "inspector" roles,
as demonstrated by ViDoRAG (Wang et al., 2025b).
This concept is further scaled in "deep research"
systems like Doc-Researcher (Dong et al., 2025c),
which implement iterative planning and evidence
gathering across multiple documents and granulari-
ties. The most advanced systems empower agents
with active perception, where an agent performs
actions like "crop" and "zoom" on visual content,
as pioneered by VRAG-RL (Wang et al., 2025c).
We leave more in-depth discussion in Appendix E.
4 Challenges & Outlook
As shown in Figure 3, A trulyeffective,efficient,
andinteractivesystem is fraught with persistent
challenges. See more discussion in Appendix F.
Expanding the Data Frontier
EffectiveRethinking Architecture Paradigms
Performance Efficiency DilemmaTowards Interactive RetrievalUncovering Scaling Laws for VDR
EfficientInteractive
Figure 3:Big picture of future challenges in VDR domain.
❶Expanding the Data Frontier.Current bench-
marks, while growing, are often limited by lan-
guage (predominantly English-centric), domain
specificity, and document structure (mostly short,
single-page documents). They seldom capture real-
world complexities such as multi-hop reasoning,
cross-document,etc.Moreover, the reliance on
VLM-generated queries for training data risks cre-
ating a feedback loop where models are trained
on data that reflects their own inherent biases, and
the potential for data leakage between large-scale
web-crawled training sets and benchmarks remains
a critical concern for robust evaluation.
❷Rethinking Architectural Paradigms.The
VDR field has largely converged on contrastively-
trained embedding and reranker models. While
effective, this paradigm may not fully leverage
the generative power of MLLMs. A promising
future direction is to explore novel architectural
paradigms that reframe the retrieval task itself.For instance, an autoregressive retrieval framework
could be developed, where the model does not just
embed butgeneratesa unique identifier or a sum-
mary of the document. Another promising avenue
is the integration of MoE, where different experts
could specialize in distinct document domains, thus
enhancing domain generalization efficiently.
❸Performance-Efficiency Dilemma.High-
performance multi-vector VDR models, while ac-
curate, demand significant storage and compu-
tational resources, making them impractical for
many real-world applications. Future research
must pursue efficiency optimizations beyond ini-
tial efforts like clustering (e.g.,Light-ColPali (Ma
et al., 2025)) or pruning (e.g.,DocPruner (Yan
et al., 2025d)). Another direction is leveraging
Matryoshka representation learning, which enables
the embeddings that can be truncated to smaller
dimensions at inference time without retraining,
offering a flexible dial to balance quality & speed.
❹Towards Interactive Retrieval.Integrating
VDR with agentic systems holds immense poten-
tial, particularly for complex scenarios like Deep
Research (Zhang et al., 2025f), where a query re-
quires iterative evidence gathering from a vast can-
didate pool. The challenge is to move beyond
simple tool-use and achieve a more organic syn-
ergy. Future work should focus on the co-design of
agents and VDR tools, enabling agents to perform
sophisticated actions like decomposing a high-level
query into a multi-step retrieval plan or adaptively
selecting retrieval granularity.
❺Uncovering Scaling Laws for VDR.While
scaling laws are well-documented for general-
purpose (M)LLMs, their application to the VDR
domain remains underexplored. A crucial future
direction is to systematically investigate the scal-
ing dynamics of VDR models with respect to both
model size and data characteristics (Huo et al.,
2026). This includes developing sophisticated
document-specific data augmentation techniques
to increase data diversity. Besides, research can
explore the potential of synthetic data generation
not just for queries but for entire visual documents.
5 Conclusion
This survey systematically discusses VDR land-
scape, categorizing the evolution of benchmarks
and methodologies (embedding, reranker, and inte-
gration with RAG and Agents). Current triumphs
push the boundaries of fine-grained matching, but
8

the field’s trajectory is shifting towards more com-
plex document intelligence. Navigating the future
frontiers of data complexity, architectural innova-
tion, efficiency, and interactivity will be critical for
realizing the full potential of VDR as a cornerstone
of multimodal document intelligence.
Limitations
•The VDR field is evolving at an unprece-
dented pace, which means some concurrent or
very recent works may not be included in this
static snapshot. However, we have conducted
a best-effort search to provide a comprehen-
sive overview and plan to maintain a living
repository to ensure the survey remains a cur-
rent and valuable resource for the research
community.
•To maintain a coherent, high-level narrative
spanning benchmarks, models, and systems,
this survey deliberately prioritizes breadth
over exhaustive technical depth for each cited
work. While this means we do not dissect
the intricate implementation details of ev-
ery model, we believe this survey provides
a more accessible and structured roadmap of
the field’s overarching trends.
•Our scope is intentionally centered on VDR,
and as such, it does not offer an exhaustive
review of the broader Visual Document Un-
derstanding (VDU) field. This focused design,
however, allows us to provide the first dedi-
cated, in-depth treatise on VDR in the MLLM
era, offering a clear and uncluttered perspec-
tive on this specific and rapidly advancing
subfield.
References
Ibrahim M Alabdulmohsin, Behnam Neyshabur, and
Xiaohua Zhai. 2022. Revisiting neural scaling laws
in language and vision.Advances in Neural Informa-
tion Processing Systems, 35:22300–22312.
Fahimeh Alaei, Alireza Alaei, Michael Blumenstein,
and Umapada Pal. 2016. A brief review of document
image retrieval methods: Recent advances. In2016
International Joint Conference on Neural Networks
(IJCNN), pages 3500–3507. IEEE.
Alibaba. 2025a. ApsaraStackMaaS.
Alibaba. 2025b. OpenSearch-AI/Ops-MM-embedding-
v1.Andrea Apicella, Francesco Isgrò, and Roberto Prevete.
2025. Don’t push the button! exploring data leak-
age risks in machine learning and transfer learning.
Artificial Intelligence Review, 58(11):339.
Muhammad Arslan, Hussam Ghanem, Saba Munawar,
and Christophe Cruz. 2024. A survey on rag with
llms.Procedia computer science, 246:3781–3790.
Sumedha Arya and Nirmal Gaud. 2025. Advances
in retrieval-augmented generation (rag) and related
frameworks.IJSAT-International Journal on Science
and Technology, 16(3).
Tajamul Ashraf, Amal Saqib, Hanan Ghani, Muhra
AlMahri, Yuhao Li, Noor Ahsan, Umair Nawaz, Jean
Lahoud, Hisham Cholakkal, Mubarak Shah, and 1
others. 2025. Agent-x: Evaluating deep multimodal
reasoning in vision-centric agentic tasks.arXiv
preprint arXiv:2505.24876.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl
technical report.arXiv preprint arXiv:2502.13923.
Juan Pablo Balarini. 2025. Eager embed v1: Multi-
modal dense embeddings for retrieval.
Lucas Beyer, Andreas Steiner, André Susano Pinto,
Alexander Kolesnikov, Xiao Wang, Daniel Salz,
Maxim Neumann, Ibrahim Alabdulmohsin, Michael
Tschannen, Emanuele Bugliarello, and 1 others. 2024.
Paligemma: A versatile 3b vlm for transfer.arXiv
preprint arXiv:2407.07726.
Jing Bi, Susan Liang, Xiaofei Zhou, Pinxin Liu,
Junjia Guo, Yunlong Tang, Luchuan Song, Chao
Huang, Guangyu Sun, Jinxi He, and 1 others. 2025.
Why reasoning matters? a survey of advance-
ments in multimodal reasoning (v1).arXiv preprint
arXiv:2504.03151.
Bart Bussmann, Noa Nabeshima, Adam Karvonen, and
Neel Nanda. 2025. Learning multi-level features
with matryoshka sparse autoencoders.arXiv preprint
arXiv:2503.17547.
ByteDance. 2025. Seed1.6-Embedding.
Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae
Lee. 2024. Matryoshka multimodal models.arXiv
preprint arXiv:2405.17430.
Hongliu Cao. 2024. Recent advances in text embed-
ding: A comprehensive review of top-performing
methods on the mteb benchmark.arXiv preprint
arXiv:2406.01607.
Yixin Cao, Shibo Hong, Xinze Li, Jiahao Ying, Yubo
Ma, Haiyuan Liang, Yantao Liu, Zijun Yao, Xiaozhi
Wang, Dan Huang, and 1 others. 2025. Toward gen-
eralizable evaluation in the llm era: A survey beyond
benchmarks.arXiv preprint arXiv:2504.18838.
Antoine Chaffin and Aurélien Lac. 2024. Monoqwen:
Visual document reranking.
9

Haonan Chen, Hong Liu, Yuping Luo, Liang Wang, Nan
Yang, Furu Wei, and Zhicheng Dou. 2025a. Moca:
Modality-aware continual pre-training makes better
bidirectional multimodal embeddings.arXiv preprint
arXiv:2506.23115.
Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Zil-
iang Zhao, Furu Wei, and Zhicheng Dou. 2025b.
mme5: Improving multimodal multilingual embed-
dings via high-quality synthetic data.arXiv preprint
arXiv:2502.08468.
Jian Chen, Ming Li, Jihyung Kil, Chenguang Wang,
Tong Yu, Ryan Rossi, Tianyi Zhou, Changyou Chen,
and Ruiyi Zhang. 2025c. Visr-bench: An empirical
study on visual retrieval-augmented generation for
multilingual long document understanding.arXiv
preprint arXiv:2508.07493.
Jianlyu Chen, Junwei Lan, Chaofan Li, Defu Lian, and
Zheng Liu. 2025d. Reasonembed: Enhanced text em-
beddings for reasoning-intensive document retrieval.
arXiv preprint arXiv:2510.08252.
Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian
Guo. 2024. Mllm is a strong reranker: Advanc-
ing multimodal retrieval-augmented generation via
knowledge-enhanced reranking and noise-injected
training.arXiv preprint arXiv:2407.21439.
Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping
Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama
Patel, Ruoxi Meng, Mingyi Su, and 1 others. 2025e.
Browsecomp-plus: A more fair and transparent eval-
uation benchmark of deep-research agent.arXiv
preprint arXiv:2508.06600.
Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu,
Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei
Cao, Jie Ma, and 1 others. 2025. A survey on
knowledge-oriented retrieval-augmented generation.
arXiv preprint arXiv:2503.10677.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3docrag: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Isaac Chung, Imene Kerboua, Marton Kardos, Roman
Solomatin, and Kenneth Enevoldsen. 2025. Main-
taining mteb: Towards long term usability and repro-
ducibility of embedding benchmarks.arXiv preprint
arXiv:2506.21182.
Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei. 2021.
Document ai: Benchmarks, models and applications.
arXiv preprint arXiv:2111.08609.
Xuanming Cui, Jianpeng Cheng, Hong-you Chen,
Satya Narayan Shukla, Abhijeet Awasthi, Xi-
chen Pan, Chaitanya Ahuja, Shlok Kumar Mishra,
Yonghuan Yang, Jun Xiao, and 1 others. 2025. Think
then embed: Generative context improves multi-
modal embedding.arXiv preprint arXiv:2510.05014.Ziqi Dai, Xin Zhang, Mingxin Li, Yanzhao Zhang,
Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie
Li, and Min Zhang. 2025. Supervised fine-tuning or
contrastive learning? towards better multimodal llm
reranking.arXiv preprint arXiv:2510.14824.
Debrup Das, Sam O’Nuallain, and Razieh Rahimi. 2025.
Rader: Reasoning-aware dense retrieval models. In
Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, pages
19981–20008.
Jingcheng Deng, Zhongtao Jiang, Liang Pang, Zihao
Wei, Liwei Chen, Kun Xu, Yang Song, Huawei Shen,
and Xueqi Cheng. 2025a. Following the autoregres-
sive nature of llm embeddings via compression and
alignment. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 12672–12688.
Yong Deng, Guoqing Wang, Zhenzhe Ying, Xiaofeng
Wu, Jinzhen Lin, Wenwen Xiong, Yuqin Dai, Shuo
Yang, Zhanwei Zhang, Qiwen Wang, and 1 others.
2025b. Atom-searcher: Enhancing agentic deep re-
search via fine-grained atomic thought reward.arXiv
preprint arXiv:2508.12800.
Yihao Ding, Soyeon Caren Han, Jean Lee, and Ed-
uard Hovy. 2024. Deep learning based visually rich
document content understanding: A survey.arXiv
preprint arXiv:2408.01287.
Yihao Ding, Siwen Luo, Yue Dai, Yanbei Jiang,
Zechuan Li, Geoffrey Martin, and Yifan Peng. 2025.
A survey on mllm-based visually rich document un-
derstanding: Methods, challenges, and emerging
trends.arXiv preprint arXiv:2507.09861.
Kuicai Dong, Yujing Chang, Xin Deik Goh, Dexun
Li, Ruiming Tang, and Yong Liu. 2025a. Mmdocir:
Benchmarking multi-modal retrieval for long docu-
ments.arXiv preprint arXiv:2501.08828.
Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng
Wang, Ruiming Tang, and Yong Liu. 2025b. Bench-
marking retrieval-augmented multimomal generation
for document question answering.arXiv preprint
arXiv:2505.16470.
Kuicai Dong, Shurui Huang, Fangda Ye, Wei Han, Zhi
Zhang, Dexun Li, Wenjun Li, Qu Yang, Gang Wang,
Yichao Wang, and 1 others. 2025c. Doc-researcher:
A unified system for multimodal document parsing
and deep research.arXiv preprint arXiv:2510.21603.
Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang,
Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou,
Tong Lu, Hongsheng Li, and 1 others. 2025. Do-
copilot: Improving multimodal models for document-
level understanding. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages
4026–4037.
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua,
Márton Kardos, Ashwin Mathur, David Stap,
Jay Gala, Wissam Siblini, Dominik Krzemi ´nski,
10

Genta Indra Winata, and 1 others. 2025. Mmteb:
Massive multilingual text embedding benchmark.
arXiv preprint arXiv:2502.13595.
Dongyang Fan, Bettina Messmer, and Martin Jaggi.
2024. Towards an empirical understanding of moe
design choices.arXiv preprint arXiv:2402.13089.
Yongqi Fan, Xiaoyang Chen, Dezhi Ye, Jie Liu, Hai-
jin Liang, Jin Ma, Ben He, Yingfei Sun, and Tong
Ruan. 2025. Tfrank: Think-free reasoning enables
practical pointwise llm ranking.arXiv preprint
arXiv:2508.09539.
Tianqing Fang, Zhisong Zhang, Xiaoyang Wang, Rui
Wang, Can Qin, Yuxuan Wan, Jun-Yu Ma, Ce Zhang,
Jiaqi Chen, Xiyun Li, and 1 others. 2025. Cognitive
kernel-pro: A framework for deep research agents
and agent foundation models training.arXiv preprint
arXiv:2508.00414.
Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Wei-
hang Su, Jia Chen, and Yiqun Liu. 2024. Scaling
laws for dense retrieval. InProceedings of the 47th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, pages
1339–1349.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Om-
rani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. 2024. Colpali: Efficient document re-
trieval with vision language models.arXiv preprint
arXiv:2407.01449.
Chenghan Fu, Daoze Zhang, Yukang Lin, Zhanheng
Nie, Xiang Zhang, Jianyu Liu, Yueran Liu, Wanx-
ian Guan, Pengjie Wang, Jian Xu, and 1 others.
2025. Moon embedding: Multimodal representation
learning for e-commerce search advertising.arXiv
preprint arXiv:2511.11305.
Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu
Yan, Zhenya Huang, Shiwei Tong, and Guoping Hu.
2025a. Retrieval augmented generation evaluation in
the era of large language models: A comprehensive
survey.arXiv preprint arXiv:2504.14891.
Wensheng Gan, Zhenyao Ning, Zhenlian Qi, and
Philip S Yu. 2025b. Mixture of experts (moe): A big
data perspective.arXiv preprint arXiv:2501.16352.
Sensen Gao, Shanshan Zhao, Xu Jiang, Lunhao Duan,
Yong Xien Chng, Qing-Guo Chen, Weihua Luo,
Kaifu Zhang, Jia-Wang Bian, and Mingming Gong.
2025. Scaling beyond context: A survey of multi-
modal retrieval-augmented generation for document
understanding.arXiv preprint arXiv:2510.15253.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).Ziyu Gong, Chengcheng Mai, and Yihua Huang.
2025. Mhier-rag: Multi-modal rag for visual-
rich document question-answering via hierarchical
and multi-granularity reasoning.arXiv preprint
arXiv:2508.00579.
Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun
Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen,
Weidong Cai, and Jiankang Deng. 2025a. Breaking
the modality barrier: Universal embedding learning
with multimodal llms. InProceedings of the 33rd
ACM International Conference on Multimedia, pages
2860–2869.
Tiancheng Gu, Kaicheng Yang, Kaichen Zhang, Xi-
ang An, Ziyong Feng, Yueyi Zhang, Weidong Cai,
Jiankang Deng, and Lidong Bing. 2025b. Unime-v2:
Mllm-as-a-judge for universal multimodal embed-
ding learning.Preprint, arXiv:2510.13515.
Michael Günther, Saba Sturua, Mohammad Kalim
Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang,
Sedigheh Eslami, Scott Martens, Maximilian Werk,
Nan Wang, and 1 others. 2025. jina-embeddings-v4:
Universal embeddings for multimodal multilingual
retrieval.arXiv preprint arXiv:2506.18902.
Hao Guo, Xugong Qin, Jun Jie Ou Yang, Peng Zhang,
Gangyan Zeng, Yubo Li, and Hailun Lin. 2025. To-
wards natural language-based document image re-
trieval: New dataset and benchmark. InProceedings
of the Computer Vision and Pattern Recognition Con-
ference, pages 29722–29732.
Ibtihaal M Hameed, Sadiq H Abdulhussain, and
Basheera M Mahmmod. 2021. Content-based im-
age retrieval: A review of recent trends.Cogent
Engineering, 8(1):1927469.
Simeng Han, Frank Palma Gomez, Tu Vu, Zefei Li,
Daniel Cer, Hansi Zeng, Chris Tar, Arman Cohan,
and Gustavo Hernandez Abrego. 2025a. Ateb: Eval-
uating and improving advanced nlp tasks for text em-
bedding models.arXiv preprint arXiv:2502.16766.
Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li,
Hongtu Zhu, and Huaxiu Yao. 2025b. Mdocagent:
A multi-modal multi-agent framework for document
understanding.arXiv preprint arXiv:2503.13964.
Hans William Alexander Hanley and Zakir Durumeric.
2025. Hierarchical level-wise news article clustering
via multilingual matryoshka embeddings. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 2476–2492.
Wenbo Hu, Zi-Yi Dou, Liunian Li, Amita Kamath,
Nanyun Peng, and Kai-Wei Chang. 2024. Ma-
tryoshka query transformer for large vision-language
models.Advances in Neural Information Processing
Systems, 37:50168–50188.
Jerry Huang, Siddarth Madala, Cheng Niu, Julia Hock-
enmaier, and Tong Zhang. 2025a. Contextual rele-
vance and adaptive sampling for llm-based document
reranking.arXiv preprint arXiv:2511.01208.
11

Kung-Hsiang Huang, Hou Pong Chan, May Fung,
Haoyi Qiu, Mingyang Zhou, Shafiq Joty, Shih-Fu
Chang, and Heng Ji. 2024. From pixels to insights:
A survey on automatic chart understanding in the era
of large foundation models.IEEE Transactions on
Knowledge and Data Engineering, 37(5):2550–2568.
Xin Huang and Kye Min Tan. 2025. Beyond text: Un-
locking true multimodal, end-to-end rag with tomoro
colqwen3.
Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang
Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang
Li, Lifeng Shang, Songcen Xu, and 1 others. 2025b.
Deep research agents: A systematic examination and
roadmap.arXiv preprint arXiv:2506.18096.
Jiahao Huo, Yu Huang, Yibo Yan, Ye Pan, Yi Cao,
Mingdong Ou, Philip S Yu, and Xuming Hu. 2026.
Causalembed: Auto-regressive multi-vector genera-
tion in latent space for visual document embedding.
arXiv preprint arXiv:2601.21262.
ILLUIN. 2025. ViDoRe V3: a comprehensive evalua-
tion of retrieval for enterprise use-cases.
Abhinav Java, Ashmit Khandelwal, Sukruta Midi-
geshi, Aaron Halfaker, Amit Deshpande, Navin
Goyal, Ankur Gupta, Nagarajan Natarajan, and Amit
Sharma. 2025. Characterizing deep research: A
benchmark and formal definition.arXiv preprint
arXiv:2508.04183.
Daniel P Jeong, Zachary C Lipton, and Pradeep Raviku-
mar. 2024. Llm-select: Feature selection with large
language models.arXiv preprint arXiv:2407.02694.
Rohan Jha, Bo Wang, Michael Günther, Georgios Mas-
trapas, Saba Sturua, Isabelle Mohr, Andreas Kouk-
ounas, Mohammad Kalim Akram, Nan Wang, and
Han Xiao. 2024. Jina-colbert-v2: A general-purpose
multilingual late interaction retriever.arXiv preprint
arXiv:2408.16672.
Yuelyu Ji, Zhuochun Li, Rui Meng, and Daqing He.
2024. Reasoningrank: Teaching student models to
rank through reasoning-based knowledge distillation.
arXiv preprint arXiv:2410.05168.
Weijian Jian, Yajun Zhang, Dawei Liang, Chunyu Xie,
Yixiao He, Dawei Leng, and Yuhui Yin. 2025. Rzen-
embed: Towards comprehensive multimodal retrieval.
arXiv preprint arXiv:2510.27350.
Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing
Wang, and Fuzhen Zhuang. 2024a. Scaling sentence
embeddings with large language models. InFind-
ings of the association for computational linguistics:
EMNLP 2024, pages 3182–3196.
Ting Jiang, Minghui Song, Zihan Zhang, Haizhen
Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing
Wang, and Fuzhen Zhuang. 2024b. E5-v: Universal
embeddings with multimodal large language models.
arXiv preprint arXiv:2407.12580.Zi-Han Jiang, Chien-Wei Lin, Wei-Hua Li, Hsuan-
Tung Liu, Yi-Ren Yeh, and Chu-Song Chen. 2025.
Relation-rich visual document generator for visual
information extraction. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 14449–14459.
Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz,
Yingbo Zhou, and Wenhu Chen. 2024c. Vlm2vec:
Training vision-language models for massive
multimodal embedding tasks.arXiv preprint
arXiv:2410.05160.
Jina AI. 2025. jina-reranker-m0: Multilingual Multi-
modal Document Reranker.
Ashutosh Joshi, Sheikh Muhammad Sarwar, Samarth
Varshney, Sreyashi Nag, Shrivats Agrawal, and Juhi
Naik. 2024. Reaper: Reasoning based retrieval plan-
ning for complex rag systems. InProceedings of the
33rd ACM International Conference on Information
and Knowledge Management, pages 4621–4628.
Wenjun Ke, Yifan Zheng, Yining Li, Hengyuan Xu,
Dong Nie, Peng Wang, and Yao He. 2025. Large
language models in document intelligence: A com-
prehensive survey, recent advances, challenges, and
future trends.ACM Trans. Inf. Syst., 44(1).
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
Juyeon Kim, Geon Lee, Dongwon Choi, Taeuk Kim,
and Kijung Shin. 2025a. Hybrid-vector retrieval for
visually rich documents: Combining single-vector
efficiency and multi-vector accuracy.arXiv preprint
arXiv:2510.22215.
Wongyu Kim, Hochang Lee, Sanghak Lee, Yoon-
sung Kim, and Jaehyun Park. 2025b. Let mul-
timodal embedders learn when to augment query
via adaptive query augmentation.arXiv preprint
arXiv:2511.02358.
Adithya S Kolavi and Vyoman Jain. 2025. M3dr: To-
wards universal multilingual multimodal document
retrieval.arXiv preprint arXiv:2512.03514.
Fanheng Kong, Jingyuan Zhang, Yahui Liu, Hongzhi
Zhang, Shi Feng, Xiaocui Yang, Daling Wang,
Yu Tian, Fuzheng Zhang, Guorui Zhou, and 1 others.
2025. Modality curation: Building universal embed-
dings for advanced multimodal information retrieval.
arXiv preprint arXiv:2505.19650.
Aditya Kusupati, Gantavya Bhatt, Aniket Rege,
Matthew Wallingford, Aditya Sinha, Vivek Ramanu-
jan, William Howard-Snyder, Kaifeng Chen, Sham
Kakade, Prateek Jain, and 1 others. 2022. Ma-
tryoshka representation learning.Advances in Neural
Information Processing Systems, 35:30233–30249.
12

Riwei Lai, Li Chen, Weixin Chen, and Rui Chen. 2024.
Matryoshka representation learning for recommenda-
tion.arXiv preprint arXiv:2406.07432.
Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and
Jinsong Su. 2025a. Llave: Large language and vi-
sion embedding models with hardness-weighted con-
trastive learning.arXiv preprint arXiv:2503.04812.
Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and
Jinsong Su. 2025b. Ume-r1: Exploring reasoning-
driven generative multimodal embeddings.arXiv
preprint arXiv:2511.00405.
Carlos Lassance, Maroua Maachou, Joohee Park, and
Stéphane Clinchant. 2021. A study on token pruning
for colbert.arXiv preprint arXiv:2112.06540.
Carlos Lassance, Maroua Maachou, Joohee Park, and
Stéphane Clinchant. 2022. Learned token pruning
in contextualized late interaction over bert (colbert).
InProceedings of the 45th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval, pages 2232–2236.
Hugo Laurençon, Andrés Marafioti, Victor Sanh, and
Léo Tronchon. 2024. Building and better understand-
ing vision-language models: insights and future di-
rections.Preprint, arXiv:2408.12637.
Jaehoon Lee, Sohyun Kim, Wanggeun Park, Geon Lee,
Seungkyung Kim, and Minyoung Lee. 2025a. Sds
kopub vdr: A benchmark dataset for visual document
retrieval in korean public documents.arXiv preprint
arXiv:2511.04910.
Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel
Cer, Madhuri Shanbhogue, Iftekhar Naim, Gus-
tavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Hen-
rique Schechter Vera, and 1 others. 2025b. Gemini
embedding: Generalizable embeddings from gemini.
arXiv preprint arXiv:2503.07891.
Hongyang Lei, Xiaolong Cheng, Dan Wang, Kun Fan,
Qi Qin, Huazhen Huang, Yetao Wu, Qingqing Gu,
Zhonglin Jiang, Yong Chen, and 1 others. 2024. M3-
jepa: Multimodal alignment via multi-directional
moe based on the jepa framework.arXiv preprint
arXiv:2409.05929.
Da Li, Yuxiao Luo, Keping Bi, Jiafeng Guo, Wei Yuan,
Biao Yang, Yan Wang, Fan Yang, Tingting Gao, and
Guorui Zhou. 2025a. Compression then matching:
An efficient pre-training paradigm for multimodal
embedding.arXiv preprint arXiv:2511.08480.
Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong
Feng, Lingpeng Kong, and Qi Liu. 2024a. Mul-
timodal ArXiv: A dataset for improving scientific
comprehension of large vision-language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 14369–14387, Bangkok, Thai-
land. Association for Computational Linguistics.Lei Li, Xiao Zhou, and Zheng Liu. 2025b. R2med:
A benchmark for reasoning-driven medical retrieval.
arXiv preprint arXiv:2505.14558.
Siting Li, Xiang Gao, and Simon Shaolei Du. 2025c.
Highlighting what matters: Promptable embeddings
for attribute-focused image retrieval.arXiv preprint
arXiv:2505.15877.
Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia,
Yichun Yin, Hao Zhang, Yong Liu, Yasheng Wang,
and Ruiming Tang. 2024b. Coir: A comprehensive
benchmark for code information retrieval models.
Preprint, arXiv:2407.02883.
Xianming Li, Zongxi Li, Jing Li, Haoran Xie, and
Qing Li. 2024c. 2d matryoshka sentence embed-
dings.arXiv preprint arXiv:2402.14776.
Xiaojie Li, Chu Li, Shi-Zhe Chen, and Xi Chen. 2025d.
U-marvel: Unveiling key factors for universal multi-
modal retrieval via embedding learning with mllms.
arXiv preprint arXiv:2507.14902.
Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Ji-
ajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong
Wen, Yuan Lu, and 1 others. 2025e. Deepagent: A
general reasoning agent with scalable toolsets.arXiv
preprint arXiv:2510.21618.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian,
Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and
Zhicheng Dou. 2025f. Webthinker: Empowering
large reasoning models with deep research capability.
arXiv preprint arXiv:2504.21776.
Xin Li, Yunfei Wu, Xinghua Jiang, Zhihao Guo, Ming-
ming Gong, Haoyu Cao, Yinsong Liu, Deqiang
Jiang, and Xing Sun. 2024d. Enhancing visual doc-
ument understanding with contrastive learning in
large visual-language models. InProceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 15546–15555.
Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh
Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei,
Henry Peng Zou, Xiao Luo, Yusheng Zhao, and 1
others. 2025g. Towards agentic rag with deep rea-
soning: A survey of rag-reasoning systems in llms.
arXiv preprint arXiv:2507.09477.
Yunxin Li, Xinyu Chen, Shenyuan Jiang, Haoyuan Shi,
Zhenyu Liu, Xuanyu Zhang, Nanhao Deng, Zhen-
ran Xu, Yicheng Ma, Meishan Zhang, and 1 others.
2025h. Uni-moe-2.0-omni: Scaling language-centric
omnimodal large model with advanced moe, training
and data.arXiv preprint arXiv:2511.12609.
Yunxin Li, Shenyuan Jiang, Baotian Hu, Longyue Wang,
Wanqi Zhong, Wenhan Luo, Lin Ma, and Min Zhang.
2025i. Uni-moe: Scaling unified multimodal llms
with mixture of experts.IEEE Transactions on Pat-
tern Analysis and Machine Intelligence.
13

Zijian Li, Xin Guan, Bo Zhang, Shen Huang, Houquan
Zhou, Shaopeng Lai, Ming Yan, Yong Jiang, Pengjun
Xie, Fei Huang, and 1 others. 2025j. Webweaver:
Structuring web-scale evidence with dynamic out-
lines for open-ended deep research.arXiv preprint
arXiv:2509.13312.
Ziyue Li and Tianyi Zhou. 2024. Your mixture-of-
experts llm is secretly an embedding model for free.
arXiv preprint arXiv:2410.10814.
Wenhui Liao, Jiapeng Wang, Hongliang Li, Chengyu
Wang, Jun Huang, and Lianwen Jin. 2025. Do-
clayllm: An efficient multi-modal extension of large
language models for text-rich document understand-
ing. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 4038–4049.
Adam Lilja, Junsheng Fu, Erik Stenborg, and Lars Ham-
marstrand. 2024. Localization is all you evaluate:
Data leakage in online mapping datasets and how to
fix it. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages
22150–22159.
Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang,
Dingkang Yang, Shuang Yang, Yueyang Yao,
Xu Chen, Zirui Guo, Shengqiang Li, and 1 oth-
ers. 2025a. Sail-embedding technical report: Omni-
modal embedding foundation model.arXiv preprint
arXiv:2510.12709.
Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi,
Jimmy Lin, Bryan Catanzaro, and Wei Ping. 2024a.
Mm-embed: Universal multimodal retrieval with
multimodal llms.arXiv preprint arXiv:2411.02571.
Weizhe Lin, Jingbiao Mei, Jinghong Chen, and Bill
Byrne. 2024b. Preflmr: Scaling up fine-grained late-
interaction multi-modal retrievers.arXiv preprint
arXiv:2402.08327.
Zhiyu Lin, Yifei Gao, Xian Zhao, Yunfan Yang, and
Jitao Sang. 2025b. Mind with eyes: from language
reasoning to multimodal reasoning.arXiv preprint
arXiv:2503.18071.
Chunxu Liu, Jiyuan Yang, Ruopeng Gao, Yuhan Zhu,
Feng Zhu, Rui Zhao, and Limin Wang. 2025a. Rea-
soning guided embeddings: Leveraging mllm reason-
ing for improved multimodal retrieval.arXiv preprint
arXiv:2511.16150.
Keliang Liu, Zizhi Chen, Mingcheng Li, Jingqun Tang,
Dingkang Yang, and Lihua Zhang. 2025b. Resolv-
ing evidence sparsity: Agentic context engineering
for long-document understanding.arXiv preprint
arXiv:2511.22850.
Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan
Meng, Ding Wang, and Jun Ma. 2025c. Hm-rag:
Hierarchical multi-agent multimodal retrieval aug-
mented generation. InProceedings of the 33rd ACM
International Conference on Multimedia, pages 2781–
2790.Qi Liu, Gang Guo, Jiaxin Mao, Zhicheng Dou, Ji-Rong
Wen, Hao Jiang, Xinyu Zhang, and Zhao Cao. 2024.
An analysis on matching mechanisms and token prun-
ing for late-interaction models.ACM Transactions
on Information Systems, 42(5):1–28.
Qianying Liu, Xiao Liang, Zhiqiang Zhang, Yibo Chen,
Xu Tang, Zhongfei Qing, Fengfan Zhou, Yao Hu, and
Paul Henderson. 2025d. Rematch: Boosting repre-
sentation through matching for multimodal retrieval.
arXiv preprint arXiv:2511.19278.
Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang,
Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi
Xie. 2025e. Lamra: Large multimodal model as your
advanced retrieval assistant. InProceedings of the
Computer Vision and Pattern Recognition Confer-
ence, pages 4015–4025.
Yuxiang Liu, Tian Wang, Gourab Kundu, Tianyu Cao,
Guang Cheng, Zhen Ge, Jianshu Chen, Qingjun Cui,
and Trishul Chilimbi. 2025f. Exploring reasoning-
infused text embedding with large language models
for zero-shot dense retrieval. InProceedings of the
34th ACM International Conference on Information
and Knowledge Management, pages 4981–4985.
Zheng Liu, Chaofan Li, Shitao Xiao, Chaozhuo Li,
Defu Lian, and Yingxia Shao. 2025g. Matryoshka
re-ranker: A flexible re-ranking architecture with
configurable depth and width.arXiv preprint
arXiv:2501.16302.
Zheng Liu, Ze Liu, Zhengyang Liang, Junjie Zhou, Shi-
tao Xiao, Chao Gao, Chen Jason Zhang, and Defu
Lian. 2025h. Any information is just worth one sin-
gle screenshot: Unifying search with visualized in-
formation retrieval. InProceedings of the 63rd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 19238–
19261.
LlamaIndex. 2025. Visual Document Retrieval Goes
Multilingual.
Ka Man Lo, Zeyu Huang, Zihan Qiu, Zili Wang, and
Jie Fu. 2025. A closer look into mixture-of-experts
in large language models. InFindings of the Associ-
ation for Computational Linguistics: NAACL 2025,
pages 4427–4447.
Meixiu Long, Duolin Sun, Dan Yang, Junjie Wang,
Yue Shen, Jian Wang, Peng Wei, Jinjie Gu, and Ji-
ahai Wang. 2025. Diver: A multi-stage approach
for reasoning-intensive information retrieval.arXiv
preprint arXiv:2508.07995.
Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu
Chen, and Jimmy Lin. 2024a. Unifying multimodal
retrieval via document screenshot embedding. InPro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing, pages 6492–
6505, Miami, Florida, USA. Association for Compu-
tational Linguistics.
14

Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xi-
aoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan,
Jiaqi Wang, Yixin Cao, and 1 others. 2025. Towards
storage-efficient visual document retrieval: An empir-
ical study on reducing patch-level embeddings.arXiv
preprint arXiv:2506.04997.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, and 1 others. 2024b. Mmlongbench-
doc: Benchmarking long-context document under-
standing with visualizations.Advances in Neural
Information Processing Systems, 37:95963–96010.
Quentin Macé, António Loison, and Manuel Faysse.
2025. Vidore benchmark v2: Raising the bar for
visual retrieval.arXiv preprint arXiv:2505.17166.
Ahmed Masry and Enamul Hoque. 2025. Colflor: To-
wards bert-size vision-language document retrieval
models. In2025 IEEE 35th International Workshop
on Machine Learning for Signal Processing (MLSP),
pages 1–5. IEEE.
Ahmed Masry, Megh Thakkar, Patrice Bechard, Sath-
wik Tejaswi Madhusudhan, Rabiul Awal, Shambhavi
Mishra, Akshay Kalkunte Suresh, Srivatsava Daruru,
Enamul Hoque, Spandana Gella, and 1 others. 2025.
Colmate: Contrastive late interaction and masked
text for multimodal document retrieval. InProceed-
ings of the 2025 Conference on Empirical Methods in
Natural Language Processing: Industry Track, pages
2071–2080.
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
Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen.
2025. A survey of multimodal retrieval-augmented
generation.arXiv preprint arXiv:2504.08748.
Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang,
Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caim-
ing Xiong, and 1 others. 2025. Vlm2vec-v2: Advanc-
ing multimodal embedding for videos, images, and
visual documents.arXiv preprint arXiv:2507.04590.
Gavin Mischler, Yinghao Aaron Li, Stephan Bickel,
Ashesh D Mehta, and Nima Mesgarani. 2024. Con-
textual feature extraction hierarchies converge in
large language models and the brain.Nature Ma-
chine Intelligence, 6(12):1467–1477.
Alexander Most, Joseph Winjum, Manish Bhattarai,
Shawn Jones, Nishath Rajiv Ranasinghe, Ayan
Biswas, and Dan O’Malley. 2025. Lost in ocr trans-
lation? vision-based approaches to robust documentretrieval. InProceedings of the 2025 ACM Sympo-
sium on Document Engineering, pages 1–10.
Niklas Muennighoff, SU Hongjin, Liang Wang, Nan
Yang, Furu Wei, Tao Yu, Amanpreet Singh, and
Douwe Kiela. 2024. Generative representational in-
struction tuning. InThe Thirteenth International
Conference on Learning Representations.
Elio Musacchio, Lucia Siciliani, Pierpaolo Basile, and
Giovanni Semeraro. 2025. xvlm2vec: Adapting
lvlm-based embedding models to multilinguality
using self-knowledge distillation.arXiv preprint
arXiv:2503.09313.
Omer Nacar and Anis Koubaa. 2025. Enhancing seman-
tic similarity understanding in arabic nlp with nested
embedding learning. InGenerative AI and Large
Language Models: Opportunities, Challenges, and
Applications: Volume 1, pages 179–216. Springer.
Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz, Elad
Ben Avraham, Alona Golts, Yair Kittenplon, Shai
Mazor, and Ron Litman. 2025. Docvlm: Make your
vlm an efficient reader. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 29005–29015.
Pranav Nair, Puranjay Datta, Jeff Dean, Prateek Jain,
and Aditya Kusupati. 2025. Matryoshka quantization.
arXiv preprint arXiv:2502.06786.
Zach Nussbaum, Brandon Duderstadt, and Andriy Mul-
yar. 2024. Nomic embed vision: Expanding the latent
space.arXiv preprint arXiv:2406.18587.
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.
Representation learning with contrastive predictive
coding.arXiv preprint arXiv:1807.03748.
Radek Osmulski, Gabriel de Souza P Moreira, Ronay
Ak, Mengyao Xu, Benedikt Schifferer, and Even
Oldridge. 2025. Miracl-vision: A large, multilingual,
visual document retrieval benchmark.arXiv preprint
arXiv:2505.11651.
Yassine Ouali, Adrian Bulat, Alexandros Xenos, Anestis
Zaganidis, Ioannis Maniadis Metaxas, Brais Mar-
tinez, and Georgios Tzimiropoulos. 2025. Vladva:
Discriminative fine-tuning of lvlms. InProceedings
of the Computer Vision and Pattern Recognition Con-
ference, pages 4101–4111.
Tim Pearce, Tabish Rashid, Dave Bignell, Raluca
Georgescu, Sam Devlin, and Katja Hofmann. 2024.
Scaling laws for pre-training agents and world mod-
els.arXiv preprint arXiv:2411.04434.
Letian Peng, Yuwei Zhang, Zilong Wang, Jayanth Srini-
vasa, Gaowen Liu, Zihan Wang, and Jingbo Shang.
2024. Answer is all you need: Instruction-following
text embedding via answering the question.arXiv
preprint arXiv:2402.09642.
15

Xiangyu Peng, Can Qin, Zeyuan Chen, Ran Xu, Caim-
ing Xiong, and Chien-Sheng Wu. 2025. Unidoc-
bench: A unified benchmark for document-centric
multimodal rag.arXiv preprint arXiv:2510.03663.
Zile Qiao, Guoxin Chen, Xuanzhong Chen, Donglei
Yu, Wenbiao Yin, Xinyu Wang, Zhen Zhang, Baix-
uan Li, Huifeng Yin, Kuan Li, and 1 others. 2025.
Webresearcher: Unleashing unbounded reasoning
capability in long-horizon agents.arXiv preprint
arXiv:2509.13309.
Alexander Michael Rombach and Peter Fettke. 2025.
Deep learning based key information extraction from
business documents: Systematic literature review.
ACM Computing Surveys, 58(2):1–37.
Chris Samarinas and Hamed Zamani. 2025. Distilla-
tion and refinement of reasoning in small language
models for document re-ranking. InProceedings
of the 2025 International ACM SIGIR Conference
on Innovative Concepts and Theories in Information
Retrieval (ICTIR), pages 430–435.
Vinay Samuel, Yue Zhou, and Henry Peng Zou. 2025.
Towards data contamination detection for modern
large language models: Limitations, inconsistencies,
and oracle challenges. InProceedings of the 31st
International Conference on Computational Linguis-
tics, pages 5058–5070.
Abdellatif Sassioui, Rachid Benouini, Yasser El Ouar-
gui, Mohamed El Kamili, Meriyem Chergui, and
Mohammed Ouzzif. 2023. Visually-rich document
understanding: concepts, taxonomy and challenges.
In2023 10th International Conference on Wireless
Networks and Mobile Communications (WINCOM),
pages 1–7. IEEE.
Rahul Seetharaman, Kaustubh D Dhole, and Aman
Bansal. 2025. Insertrank: Llms can reason over
bm25 scores to improve listwise reranking.arXiv
preprint arXiv:2506.14086.
Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muen-
nighoff, Xi Victoria Lin, Daniela Rus, Bryan
Kian Hsiang Low, Sewon Min, Wen-tau Yih,
Pang Wei Koh, and 1 others. 2025. Reasonir: Train-
ing retrievers for reasoning tasks.arXiv preprint
arXiv:2504.20595.
Wenxuan Shen, Mingjia Wang, Yaochen Wang, Dong-
ping Chen, Junjie Yang, Yao Wan, and Weiwei Lin.
2025. Are we on the right way for assessing docu-
ment retrieval-augmented generation?arXiv preprint
arXiv:2508.03644.
Abhinav Shukla, Sai Vemprala, Aditya Kusupati, and
Ashish Kapoor. 2024. Matmamba: A matryoshka
state space model.arXiv preprint arXiv:2410.06718.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented gen-
eration: A survey on agentic rag.arXiv preprint
arXiv:2501.09136.Dingjie Song, Sicheng Lai, Shunian Chen, Lichao Sun,
and Benyou Wang. 2024. Both text and images
leaked! a systematic analysis of multimodal llm data
contamination.arXiv preprint arXiv:2411.03823.
Shezheng Song, Xiaopeng Li, Shasha Li, Shan Zhao,
Jie Yu, Jun Ma, Xiaoguang Mao, Weimin Zhang, and
Meng Wang. 2025. How to bridge the gap between
modalities: Survey on multimodal large language
model.IEEE Transactions on Knowledge and Data
Engineering.
Zhivar Sourati, Zheng Wang, Marianne Menglin Liu,
Yazhe Hu, Mengqing Guo, Sujeeth Bharadwaj, Kyu
Han, Tao Sheng, Sujith Ravi, Morteza Dehghani,
and 1 others. 2025. Lad-rag: Layout-aware dynamic
rag for visually-rich document understanding.arXiv
preprint arXiv:2510.07233.
Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram,
Michael Günther, Bo Wang, Markus Krimmel, Feng
Wang, Georgios Mastrapas, Andreas Koukounas,
Nan Wang, and 1 others. 2024. jina-embeddings-
v3: Multilingual embeddings with task lora.arXiv
preprint arXiv:2409.10173.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S Siegel, Michael Tang, and 1 others.
2024. Bright: A realistic and challenging bench-
mark for reasoning-intensive retrieval.arXiv preprint
arXiv:2407.12883.
Jiamin Su, Yibo Yan, Zhuoran Gao, Han Zhang,
Xiang Liu, and Xuming Hu. 2025a. Cafes:
A collaborative multi-agent framework for multi-
granular multimodal essay scoring.arXiv preprint
arXiv:2505.13965.
Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu,
Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide
Zeng, Zhengyuan Yang, and 1 others. 2025b. Think-
ing with images for multimodal reasoning: Founda-
tions, methods, and future frontiers.arXiv preprint
arXiv:2506.23918.
Nishant Subramani, Alexandre Matton, Malcolm
Greaves, and Adrian Lam. 2020. A survey of deep
learning approaches for ocr and document under-
standing.arXiv preprint arXiv:2011.13534.
Hao Sun, Yingyan Hou, Jiayan Guo, Bo Wang, Chunyu
Yang, Jinsong Ni, and Yan Zhang. 2025a. Unveil:
Unified visual-textual integration and distillation for
multi-modal document retrieval. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 23935–23945.
Yubo Sun, Chunyi Peng, Yukun Yan, Shi Yu, Zheng-
hao Liu, Chi Chen, Zhiyuan Liu, and Maosong Sun.
2025b. Visrag 2.0: Evidence-guided multi-image
reasoning in visual retrieval-augmented generation.
arXiv preprint arXiv:2510.09733.
16

Manan Suri, Puneet Mathur, Franck Dernoncourt,
Kanika Goswami, Ryan A Rossi, and Dinesh
Manocha. 2025. Visdom: Multi-document qa with
visually rich elements using multimodal retrieval-
augmented generation. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 6088–6109.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 24827–24837.
Yixuan Tang and Yi Yang. 2024. Do we need domain-
specific embedding models? an empirical investiga-
tion.arXiv preprint arXiv:2409.18511.
Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang,
Yang Liu, Chenguang Zhu, Michael Zeng, Cha
Zhang, and Mohit Bansal. 2023. Unifying vision,
text, and layout for universal document processing.
InProceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19254–
19264.
Chongyang Tao, Tao Shen, Shen Gao, Junshuo Zhang,
Zhen Li, Kai Hua, Wenpeng Hu, Zhengwei Tao, and
Shuai Ma. 2024. Llms are also effective embed-
ding models: An in-depth overview.arXiv preprint
arXiv:2412.12591.
Granite Vision Team, Leonid Karlinsky, Assaf Arbelle,
Abraham Daniels, Ahmed Nassar, Amit Alfassi,
Bo Wu, Eli Schwartz, Dhiraj Joshi, Jovana Kondic,
and 1 others. 2025a. Granite vision: a lightweight,
open-source multimodal model for enterprise intelli-
gence.arXiv preprint arXiv:2502.09927.
Nomic Team. 2025. Nomic embed multimodal: Inter-
leaved text, image, and screenshots for visual docu-
ment retrieval.
Tongyi DeepResearch Team, Baixuan Li, Bo Zhang,
Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin
Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, and 1
others. 2025b. Tongyi deepresearch technical report.
arXiv preprint arXiv:2510.24701.
Paul Teiletche, Quentin Macé, Max Conti, Anto-
nio Loison, Gautier Viaud, Pierre Colombo, and
Manuel Faysse. 2025. Modernvbert: Towards
smaller visual document retrievers.arXiv preprint
arXiv:2510.01149.
Raghuveer Thirukovalluru, Rui Meng, Ye Liu, Mingyi
Su, Ping Nie, Semih Yavuz, Yingbo Zhou, Wenhu
Chen, Bhuwan Dhingra, and 1 others. 2025. Break-
ing the batch barrier (b3) of contrastive learn-
ing via smart batch mining.arXiv preprint
arXiv:2505.11293.Anyang Tong, Xiang Niu, ZhiPing Liu, Chang Tian,
Yanyan Wei, Zenglin Shi, and Meng Wang. 2025.
Hkrag: Holistic knowledge retrieval-augmented gen-
eration over visually-rich documents.arXiv preprint
arXiv:2511.20227.
Yu-Che Tsai, Kuan-Yu Chen, Yuan-Chi Li, Yuan-Hao
Chen, Ching-Yu Tsai, and Shou-De Lin. 2025. Let
llms speak embedding languages: Generative text em-
beddings via iterative contrastive refinement.arXiv
preprint arXiv:2509.24291.
Henrique Schechter Vera, Sahil Dua, Biao Zhang,
Daniel Salz, Ryan Mullins, Sindhu Raghuram Pa-
nyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang
Chen, and 1 others. 2025. Embeddinggemma: Pow-
erful and lightweight text representations.arXiv
preprint arXiv:2509.20354.
Chetan Verma, Aditya Srinivas Timmaraju, Cho-Jui
Hsieh, Suyash Damle, Ngot Bui, Yang Zhang, Wen
Chen, Xin Liu, Prateek Jain, and Inderjit Dhillon.
2025. Matryoshka model learning for improved elas-
tic student models. InProceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and
Data Mining V . 2, pages 4935–4944.
Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin,
Wenyuan Wang, Tunyu Zhang, Akshay Nambi,
Tanuja Ganu, and Hao Wang. 2025a. Multimodal
needle in a haystack: Benchmarking long-context
capability of multimodal large language models. In
Proceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Compu-
tational Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 3221–3241.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024a. Improv-
ing text embeddings with large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 11897–11916.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu,
Shihang Wang, Pengjun Xie, and Feng Zhao. 2025b.
Vidorag: Visual document retrieval-augmented gen-
eration via dynamic iterative reasoning agents.arXiv
preprint arXiv:2502.18017.
Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen,
Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang,
and Feng Zhao. 2025c. Vrag-rl: Empower vision-
perception-based rag for visually rich information
understanding via iterative reasoning with reinforce-
ment learning.arXiv preprint arXiv:2505.22019.
Tianshi Wang, Fengling Li, Lei Zhu, Jingjing Li, Zheng
Zhang, and Heng Tao Shen. 2025d. Cross-modal
retrieval: a systematic review of methods and future
directions.Proceedings of the IEEE.
Wei-Yao Wang, Kazuya Tateishi, Qiyu Wu, Shusuke
Takahashi, and Yuki Mitsufuji. 2025e. Virtue: Visual-
interactive text-image universal embedder.arXiv
preprint arXiv:2510.00523.
17

Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen
Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe
Chen, Kaipeng Zhang, Lewei Lu, and 1 others. 2024b.
Needle in a multimodal haystack.Advances in
Neural Information Processing Systems, 37:20540–
20565.
Yueqi Wang, Zhenrui Yue, Huimin Zeng, Dong Wang,
and Julian McAuley. 2024c. Train once, de-
ploy anywhere: Matryoshka representation learn-
ing for multimodal recommendation.arXiv preprint
arXiv:2409.16627.
Zhicheng Wang, Chen Ju, Xu Chen, Shuai Xiao, Jinsong
Lan, Xiaoyong Zhu, Ying Chen, and Zhiguo Cao.
2025f. Explore more, learn better: Parallel mllm
embeddings under mutual information minimization.
arXiv preprint arXiv:2511.01588.
Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu,
Frank F. Xu, Yiqing Xie, Graham Neubig, and Daniel
Fried. 2024d. Coderag-bench: Can retrieval augment
code generation?Preprint, arXiv:2406.14497.
Navve Wasserman, Oliver Heinimann, Yuval Golbari,
Tal Zimbalist, Eli Schwartz, and Michal Irani. 2025a.
Docrerank: Single-page hard negative query gener-
ation for training multi-modal rag rerankers.arXiv
preprint arXiv:2505.22584.
Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz
Goldfarb, Eli Schwartz, Udi Barzelay, and Leonid
Karlinsky. 2025b. Real-mm-rag: A real-world
multi-modal retrieval benchmark.arXiv preprint
arXiv:2502.12342.
Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu,
Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen.
2024a. Uniir: Training and benchmarking univer-
sal multimodal information retrievers. InEuropean
Conference on Computer Vision, pages 387–404.
Springer.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024b. In-
structrag: Instructing retrieval-augmented genera-
tion via self-synthesized rationales.arXiv preprint
arXiv:2406.13629.
Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle
Lo, Arman Cohan, Benjamin Van Durme, Dawn
Lawrie, and Luca Soldaini. 2024. Followir: Eval-
uating and teaching information retrieval models to
follow instructions.Preprint, arXiv:2403.15246.
Orion Weller, Benjamin Chang, Eugene Yang, Mahsa
Yarmohammadi, Samuel Barham, Sean MacAvaney,
Arman Cohan, Luca Soldaini, Benjamin Van Durme,
and Dawn Lawrie. 2025a. mfollowir: A multilin-
gual benchmark for instruction following in retrieval.
InEuropean Conference on Information Retrieval,
pages 295–310. Springer.
Orion Weller, Kathryn Ricci, Eugene Yang, Andrew
Yates, Dawn Lawrie, and Benjamin Van Durme.
2025b. Rank1: Test-time compute for rerank-
ing in information retrieval.arXiv preprint
arXiv:2502.18418.Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can
Chen, Ye Yuan, Lianming Huang, Xue Liu, Tei-Wei
Kuo, Nan Guan, and 1 others. 2024a. Retrieval-
augmented generation for natural language process-
ing: A survey.arXiv preprint arXiv:2407.13193.
Tsung-Han Wu, Giscard Biamby, Jerome Quenum,
Ritwik Gupta, Joseph E Gonzalez, Trevor Dar-
rell, and David M Chan. 2024b. Visual haystacks:
A vision-centric needle-in-a-haystack benchmark.
arXiv preprint arXiv:2407.13766.
Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and
Hong Cheng. 2025. Molorag: Bootstrapping doc-
ument understanding via multi-modal logic-aware
retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 14035–14056.
Chaojun Xiao, Jie Cai, Weilin Zhao, Biyuan Lin,
Guoyang Zeng, Jie Zhou, Zhi Zheng, Xu Han,
Zhiyuan Liu, and Maosong Sun. 2025a. Densing
law of llms.Nature Machine Intelligence, pages
1–11.
Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen
Xu, Mahani Aljunied, and Yu Rong. 2025b. Scaling
language-centric omnimodal representation learning.
arXiv preprint arXiv:2510.11693.
Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie
Stirling, Xin Zhang, Márton Kardos, Roman Solo-
matin, Noura Al Moubayed, Kenneth Enevoldsen,
and Niklas Muennighoff. 2025c. Mieb: Massive im-
age embedding benchmark. InProceedings of the
IEEE/CVF International Conference on Computer
Vision, pages 22187–22198.
Chenghao Xiao, G Thomas Hudson, and Noura Al
Moubayed. 2024. Rar-b: Reasoning as retrieval
benchmark.arXiv preprint arXiv:2404.06347.
Zilin Xiao, Qi Ma, Mengting Gu, Chun-cheng Jason
Chen, Xintao Chen, Vicente Ordonez, and Vijai Mo-
han. 2025d. Metaembed: Scaling multimodal re-
trieval at test-time with flexible late interaction.arXiv
preprint arXiv:2509.18095.
Peijin Xie, Shun Qian, Bingquan Liu, Dexin Wang, Lin
Sun, and Xiangzheng Zhang. 2025. Textlessrag: End-
to-end visual document rag by speech without text.
arXiv preprint arXiv:2509.07538.
Jing Xiong, Gongye Liu, Lun Huang, Chengyue Wu,
Taiqiang Wu, Yao Mu, Yuan Yao, Hui Shen, Zhong-
wei Wan, Jinfa Huang, and 1 others. 2024. Autore-
gressive models in vision: A survey.arXiv preprint
arXiv:2411.05902.
Cheng Xu, Shuhao Guan, Derek Greene, M Kechadi,
and 1 others. 2024. Benchmark data contamination
of large language models: A survey.arXiv preprint
arXiv:2406.04244.
18

Haike Xu and Tong Chen. 2025. Beyond se-
quential reranking: Reranker-guided search im-
proves reasoning intensive retrieval.arXiv preprint
arXiv:2509.07163.
Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Os-
mulski, Yauhen Babakhin, Zhiding Yu, Benedikt
Schifferer, and Even Oldridge. 2025a. Llama
nemoretriever colembed: Top-performing text-image
retrieval model.arXiv preprint arXiv:2507.05513.
Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Si-
hang Li, Zhifeng Gao, Renxin Zhong, and Hengx-
ing Cai. 2025b. Mm-r5: Multimodal reasoning-
enhanced reranker via reinforcement learning for doc-
ument retrieval.arXiv preprint arXiv:2506.12364.
Youze Xue, Dian Li, and Gang Liu. 2025. Im-
prove multi-modal embedding learning via explicit
hard negative gradient amplifying.arXiv preprint
arXiv:2506.02020.
Yibo Yan, Jiamin Su, Jianxiang He, Fangteng Fu,
Xu Zheng, Yuanhuiyi Lyu, Kun Wang, Shen Wang,
Qingsong Wen, and Xuming Hu. 2025a. A survey
of mathematical reasoning in the era of multimodal
large language model: Benchmark, method & chal-
lenges. InFindings of the Association for Computa-
tional Linguistics: ACL 2025, pages 11798–11827.
Yibo Yan, Shen Wang, Jiahao Huo, Hang Li, Boyan
Li, Jiamin Su, Xiong Gao, Yi-Fan Zhang, Tianlong
Xu, Zhendong Chu, and 1 others. 2024. Errorradar:
Benchmarking complex mathematical reasoning of
multimodal large language models via error detection.
arXiv preprint arXiv:2410.04509.
Yibo Yan, Shen Wang, Jiahao Huo, Jingheng Ye, Zhen-
dong Chu, Xuming Hu, Philip S Yu, Carla Gomes,
Bart Selman, and Qingsong Wen. 2025b. Posi-
tion: Multimodal large language models can signifi-
cantly advance scientific reasoning.arXiv preprint
arXiv:2502.02871.
Yibo Yan, Shen Wang, Jiahao Huo, Philip S Yu, Xum-
ing Hu, and Qingsong Wen. 2025c. Mathagent:
Leveraging a mixture-of-math-agent framework for
real-world multimodal mathematical error detection.
arXiv preprint arXiv:2503.18132.
Yibo Yan, Guangwei Xu, Xin Zou, Shuliang Liu, James
Kwok, and Xuming Hu. 2025d. Docpruner: A
storage-efficient framework for multi-vector visual
document retrieval via adaptive patch-level embed-
ding pruning.arXiv preprint arXiv:2509.23883.
Eugene Yang, Andrew Yates, Kathryn Ricci, Orion
Weller, Vivek Chari, Benjamin Van Durme, and
Dawn Lawrie. 2025a. Rank-k: Test-time rea-
soning for listwise reranking.arXiv preprint
arXiv:2505.14432.
Zhaorui Yang, Bo Pan, Han Wang, Yiyao Wang, Xingyu
Liu, Luoxuan Weng, Yingchaojie Feng, Haozhe Feng,Minfeng Zhu, Bo Zhang, and 1 others. 2025b. Mul-
timodal deepresearcher: Generating text-chart inter-
leaved reports from scratch with agentic framework.
arXiv preprint arXiv:2506.02454.
Jinsung Yoon, Rajarishi Sinha, Sercan O Arik, and
Tomas Pfister. 2024. Matryoshka-adaptor: Unsuper-
vised and supervised tuning for smaller embedding
dimensions. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Process-
ing, pages 10318–10336.
Hao Yu, Zhuokai Zhao, Shen Yan, Lukasz Korycki,
Jianyu Wang, Baosheng He, Jiayi Liu, Lizhu
Zhang, Xiangjun Fan, and Hanchao Yu. 2025a.
Cafe: Unifying representation and generation with
contrastive-autoregressive finetuning.arXiv preprint
arXiv:2503.19900.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and 1 others. 2024. Vis-
rag: Vision-based retrieval-augmented generation
on multi-modality documents.arXiv preprint
arXiv:2410.10594.
Xinlei Yu, Zhangquan Chen, Yudong Zhang, Shilin Lu,
Ruolin Shen, Jiangning Zhang, Xiaobin Hu, Yan-
wei Fu, and Shuicheng Yan. 2025b. Visual docu-
ment understanding and question answering: A multi-
agent collaboration framework with test-time scaling.
arXiv preprint arXiv:2508.03404.
Biao Zhang, Lixin Chen, Tong Liu, and Bo Zheng.
2025a. Smec: Rethinking matryoshka representa-
tion learning for retrieval embedding compression.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
26220–26233.
Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Fi-
rat. 2024a. When scaling meets llm finetuning: The
effect of data, model and finetuning method.arXiv
preprint arXiv:2402.17193.
Charles Zhang, Benji Peng, Xintian Sun, Qian Niu,
Junyu Liu, Keyu Chen, Ming Li, Pohsun Feng, Ziqian
Bi, Ming Liu, and 1 others. 2024b. From word vec-
tors to multimodal embeddings: Techniques, applica-
tions, and future directions for large language models.
arXiv preprint arXiv:2411.05036.
Junyuan Zhang, Qintong Zhang, Bin Wang, Linke
Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Con-
ghui He, and Wentao Zhang. 2025b. Ocr hinders
rag: Evaluating the cascading impact of ocr on
retrieval-augmented generation. InProceedings of
the IEEE/CVF International Conference on Com-
puter Vision, pages 17443–17453.
Kun Zhang, Jingyu Li, Zhe Li, Jingjing Zhang, Fan Li,
Yandong Liu, Rui Yan, Zihang Jiang, Nan Chen, Lei
Zhang, and 1 others. 2025c. Composed multi-modal
retrieval: A survey of approaches and applications.
arXiv preprint arXiv:2503.01334.
19

Ruichen Zhang, Shunpu Tang, Yinqiu Liu, Dusit Niy-
ato, Zehui Xiong, Sumei Sun, Shiwen Mao, and Zhu
Han. 2025d. Toward agentic ai: generative informa-
tion retrieval inspired intelligent communications and
networking.arXiv preprint arXiv:2502.16866.
Siyue Zhang, Yuan Gao, Xiao Zhou, Yilun Zhao, Tingyu
Song, Arman Cohan, Anh Tuan Luu, and Chen Zhao.
2025e. Mrmr: A realistic and expert-level multidis-
ciplinary benchmark for reasoning-intensive multi-
modal retrieval.arXiv preprint arXiv:2510.09510.
Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue
Jia, Yichao Wang, Huifeng Guo, Yong Liu, and
Xiangyu Zhao. 2025f. Deep research: A survey
of autonomous research agents.arXiv preprint
arXiv:2508.12752.
Wenzheng Zhang, Xi Victoria Lin, Karl Stratos, Wen-
tau Yih, and Mingda Chen. 2025g. Imprag: Retrieval-
augmented generation with implicit queries.arXiv
preprint arXiv:2506.02279.
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi
Dai, Dingkun Long, Pengjun Xie, Meishan Zhang,
Wenjie Li, and Min Zhang. 2024c. Gme: Improving
universal multimodal retrieval by multimodal llms.
arXiv preprint arXiv:2412.16855.
Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo,
Ehsan Kamalloo, David Alfonso-Hermelo, Xi-
aoguang Li, Qun Liu, Mehdi Rezagholizadeh, and
Jimmy Lin. 2023. Miracl: A multilingual retrieval
dataset covering 18 diverse languages.Transactions
of the Association for Computational Linguistics,
11:1114–1131.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025h.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Yimeng Zhang, Jiri Gesi, Ran Xue, Tian Wang, Ziyi
Wang, Yuxuan Lu, Sinong Zhan, Huimin Zeng,
Qingjun Cui, Yufan Guo, and 1 others. 2025i. See,
think, act: Online shopper behavior simulation with
vlm agents.arXiv preprint arXiv:2510.19245.
Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai
Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding,
Xiaobao Guo, Minzhi Li, Xingxuan Li, and 1 oth-
ers. 2023. Retrieving multimodal information for
augmented generation: A survey.arXiv preprint
arXiv:2303.10868.
Xinping Zhao, Xinshuo Hu, Zifei Shan, Shouzheng
Huang, Yao Zhou, Xin Zhang, Zetian Sun, Zhenyu
Liu, Dongfang Li, Xinyuan Wei, and 1 others. 2025.
Kalm-embedding-v2: Superior training techniques
and data inspire a versatile embedding model.arXiv
preprint arXiv:2506.20923.
Xu Zheng, Ziqiao Weng, Yuanhuiyi Lyu, Lutao Jiang,
Haiwei Xue, Bin Ren, Danda Paudel, Nicu Sebe,Luc Van Gool, and Xuming Hu. 2025a. Re-
trieval augmented generation and understanding in
vision: A survey and new outlook.arXiv preprint
arXiv:2503.18016.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025b.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments.arXiv
preprint arXiv:2504.03160.
Junjie Zhou, Ze Liu, Lei Xiong, Jin-Ge Yao, Yueze
Wang, Shitao Xiao, Fenfen Lin, Miguel Hu Chen,
Zhicheng Dou, Siqi Bao, and 1 others. 2025a. Mr2-
bench: Going beyond matching to reasoning in mul-
timodal retrieval.arXiv preprint arXiv:2509.26378.
Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and
Yongping Xiong. 2024. Vista: Visualized text em-
bedding for universal multi-modal retrieval.arXiv
preprint arXiv:2406.04292.
Junjie Zhou, Yongping Xiong, Zheng Liu, Ze Liu, Shi-
tao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang,
and Defu Lian. 2025b. Megapairs: Massive data
synthesis for universal multimodal retrieval. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 19076–19095.
Wengang Zhou, Houqiang Li, and Qi Tian. 2017. Re-
cent advance in content-based image retrieval: A
literature survey.arXiv preprint arXiv:1706.06064.
Dawei Zhu, Rui Meng, Jiefeng Chen, Sujian Li,
Tomas Pfister, and Jinsung Yoon. 2025a. Doclens:
A tool-augmented multi-agent framework for long
visual document understanding.arXiv preprint
arXiv:2511.11552.
Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wen-
hao Wu, Furu Wei, and Sujian Li. 2024a. Longem-
bed: Extending embedding models for long context
retrieval.arXiv preprint arXiv:2404.12096.
Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang,
Haozhou Zhang, and Tat-Seng Chua. 2022. Towards
complex document understanding by discrete reason-
ing. InProceedings of the 30th ACM International
Conference on Multimedia, pages 4857–4866.
Lanyun Zhu, Deyi Ji, Tianrun Chen, Haiyang Wu, and
Shiqi Wang. 2025b. Retrv-r1: A reasoning-driven
mllm framework for universal and efficient multi-
modal retrieval.Advances in Neural Information
Processing Systems.
Yichen Zhu, Zhicai Ou, Xiaofeng Mou, and Jian Tang.
2024b. Retrieval-augmented embodied agents. In
Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 17985–
17995.
Yuhan Zhu, Xiangyu Zeng, Chenting Wang, Xinhao Li,
Yicheng Xu, Ziang Yan, Yi Wang, and Limin Wang.
2025c. Freeret: Mllms as training-free retrievers.
arXiv preprint arXiv:2509.24621.
20

Shengyao Zhuang, Shuai Wang, Fabio Zheng, Bevan
Koopman, and Guido Zuccon. 2024. Starbucks-v2:
Improved training for 2d matryoshka embeddings.
arXiv preprint arXiv:2410.13230.
21

Contents of Technical Appendices
AIllustrative Examples of Visual Docu-
ments 23
B More Details of Evaluation Metrics 23
CMore Details of Reasoning-Intensive
Benchmarks 25
D Common VDR Training Sets 26
EMore Discussion of Integration of VDR
with RAG and Agent 28
E.1 Current Paradigms and Key Trends 28
E.2 The Evolutionary Synergy of VDU
and VDR . . . . . . . . . . . . . 29
FVDR Challenges and Actionable Solu-
tions 29
F.1 Expanding the Data Frontier . . . 29
F.2 Rethiking Architectural Paradigms 30
F.3 Performance-Efficiency Dilemma 31
F.4 Towards Interactive Retrieval . . . 32
F.5 Uncovering Scaling Laws for VDR 33
G Usage of AI Assistant 33
22

Technical Appendices and Supplements
A Illustrative Examples of Visual
Documents
See Figures 4, 5, and 6 for illustrative examples
from three representative VDR benchmarks.
B More Details of Evaluation Metrics
This section provides detailed mathematical for-
mulations for the primary retrieval metrics used
in the benchmarks discussed in this survey. For a
given set of test queries Q, the goal of a retrieval
system is to return a ranked list of documents from
a corpusCfor each queryq∈ Q.
❶Recall@kRecall at k (Recall@k) measures
the fraction of relevant documents that are success-
fully retrieved within the top-k results. It is a metric
of coverage, evaluating how well the system is able
to find all ground-truth documents. For a given
query q, letRqbe the set of ground-truth relevant
documents and Lk(q)be the ranked list of the top-
kretrieved documents. The overall Recall@k is
the average score across all queries inQ.
Recall@k is calculated as:
Recall@k=1
|Q|X
q∈Q|Rq∩Lk(q)|
|Rq|(1)
where:
•|Q|is the total number of queries.
•|R q|is the total number of relevant documents
for queryq.
•|R q∩Lk(q)|is the number of relevant docu-
ments found in the top-kretrieved list.
❷Mean Reciprocal Rank (MRR)Mean Recip-
rocal Rank (MRR) evaluates the ranking of thefirst
correct document retrieved. It is particularly use-
ful for tasks where finding a single relevant item
quickly is the primary goal. For each query q, the
reciprocal rank is the inverse of the rank of the
first relevant document. If no relevant document is
retrieved, the reciprocal rank is 0.
MRR is calculated as:
MRR=1
|Q|X
q∈Q1
rank q(2)
where:
• rank qis the position (rank) of the first relevant
document for queryq.❸Mean Average Precision (mAP)Mean Av-
erage Precision (mAP) provides a comprehensive
evaluation of a ranked list by considering both pre-
cision and recall. It rewards retrieving relevant
documents at higher ranks. The Average Precision
(AP) for a single query is first calculated by av-
eraging the precision at each relevant document’s
position. mAP is then the mean of these AP scores
over all queries.
The Average Precision for a single queryqis:
APq=1
|Rq||C|X
k=1(P(k)×rel(k))(3)
And mAP is the mean of all AP scores:
mAP=1
|Q|X
q∈QAPq (4)
where:
•|C| is the total number of documents in the
corpus.
•P(k)is the precision at rankk.
• rel(k) is an indicator function that is 1 if the
document at rank kis relevant, and 0 other-
wise.
❹Normalized Discounted Cumulative Gain
(nDCG@k)Normalized Discounted Cumulative
Gain (nDCG@k) is a measure of ranking quality
that accounts for the graded relevance of docu-
ments. It assigns higher scores to more relevant
documents placed at top ranks, with a logarithmic
discount for documents at lower ranks. The score
is normalized by the ideal ranking to fall between
0 and 1.
The DCG@k is first calculated as:
DCG@k=kX
i=1reli
log2(i+ 1)(5)
The nDCG@k is then:
nDCG@k=DCG@k
IDCG@k(6)
where:
•kis the number of results being evaluated.
• rel iis the graded relevance score of the docu-
ment at positioni.
• IDCG@k is the Ideal Discounted Cumulative
Gain, representing the DCG score of the per-
fect ranking up to positionk.
23

ESG Report –ViDoRe-V2
Biomedical Slide –ViDoRe-V2
Economic Report –ViDoRe-V2Figure 4:Illustrative examples of ViDoRe-V2 (Macé et al., 2025).
German New –JinaVDR
Russian Beverage Catalogue –JinaVDR
Chinese Master Plan –JinaVDR
Arabic Infographics –JinaVDR
Figure 5:Illustrative examples of JinaVDR (Günther et al., 2025).
❺Hit Rate (HR@k)As used in benchmarks like
Double-Bench (Shen et al., 2025), Hit Rate at k
(HR@k) is a binary success metric that evaluates
whetherat least onerelevant document is found
within the top-k retrieved results. It is particularly
useful for assessing whether the retrieval systemcan find any correct evidence, which is a prerequi-
site for downstream tasks. For multi-hop queries,
a "hit" is only registered if evidence forallhops is
found within the top-k results.
24

Financial Report –Real-MM-RAGFinancial Slides –Real-MM-RAG
Technology Report –Real-MM-RAGTechnology Slides –Real-MM-RAG
Figure 6:Illustrative examples of Real-MM-RAG (Wasserman et al., 2025b).
For a single-hop queryq, the HR@k is:
HR@k(q) =I(L k(q)∩R q̸=∅)(7)
The overall HR@k is the average across all queries:
HR@k=1
|Q|X
q∈QHR@k(q)(8)
where:
•I(·) is the indicator function, which is 1 if the
condition is true and 0 otherwise.
•Lk(q)is the top-k retrieved list and Rqis the
set of ground-truth documents.
❻Averaged Normalized Longest Common Sub-
sequence (ANLCS)Used in VisDoMBench (Suri
et al., 2025) to evaluate evidence extraction, Aver-
aged Normalized Longest Common Subsequence
(ANLCS) measures the textual similarity between
retrieved content and ground-truth evidence. In-
stead of evaluating ranking, it assesses the quality
of the retrieved content itself. First, the Normal-
ized Longest Common Subsequence (NLCS) is
calculated for a pair of text strings, measuring their
shared content normalized by their lengths.
The NLCS between a ground-truth evidence
stringS gtand a retrieved chunk stringS retis:
NLCS(S gt, Sret) =2× |LCS(S gt, Sret)|
|Sgt|+|S ret|(9)The ANLCS score for a query is then computed
by finding the best-matching retrieved chunk for
each ground-truth evidence chunk and averaging
these scores. The final metric is the average over
all queries.
ANLCS@k=1
|Q|X
q∈Q 
1
|Gq|X
g∈G q
max
c∈C k(q)NLCS(g, c)! (10)
where:
• LCS(S gt, Sret)is the Longest Common Sub-
sequence between the two strings.
•|S|denotes the length of stringS.
•Gqis the set of ground-truth evidence texts
for queryq.
•Ck(q)is the set of retrieved text chunks in the
top-k results for queryq.
C More Details of Reasoning-Intensive
Benchmarks
The rapid advancement of MLLMs has funda-
mentally shifted the objective of information re-
trieval from surface-level semantic matching to
deep cognitive reasoning (Bi et al., 2025; Su
et al., 2025b; Lin et al., 2025b; Yan et al., 2024).
25

Whiletext-centric retrieval has already pio-
neered "reasoning-aware" capabilitiesthrough
benchmarks3(e.g.,BRIGHT (Su et al., 2024),
RAR-b (Xiao et al., 2024), ATEB (Han et al.,
2025a), R2MED (Li et al., 2025b)), embedding
models (e.g.,ReasonIR (Shao et al., 2025), Rea-
sonEmbed (Chen et al., 2025d), RaDeR (Das et al.,
2025), DIVER-Retriever (Long et al., 2025), RITE
(Liu et al., 2025f), REAPER (Joshi et al., 2024))
and reranker models (e.g.,Reason-to-Rank (Ji et al.,
2024), InteRank (Samarinas and Zamani, 2025),
Rank1 (Weller et al., 2025b), Rank-K (Yang et al.,
2025a), TFRank (Fan et al., 2025), TS-SetRank
(Huang et al., 2025a), InsertRank (Seetharaman
et al., 2025), RGS (Xu and Chen, 2025)), VDR is
currently witnessing a pivotal transition. In VDR,
the density of textual information combined with
intricate visual layouts (e.g.,charts, tables, and dia-
grams) necessitates a level of logical deduction that
extends far beyond traditional OCR-based match-
ing (Duan et al., 2025; Liao et al., 2025; Nacson
et al., 2025; Jiang et al., 2025).
Recent pioneering efforts have begun to formal-
ize this reasoning-intensive frontier through three
distinct perspectives:
❶Vision-Centric Logic and Abstract Reason-
ing.As highlighted byMR2-Bench(Zhou et al.,
2025a), current VDR models often suffer from
"shallow matching," where they succeed by iden-
tifying object-text correlations but fail at logical,
spatial, or causal inference. MR2-Bench introduces
vision-centric tasks such asVisual Puzzle(solv-
ing Raven-style matrices) andVisual Illustration
Search(e.g.,matching a mathematical formula to
its corresponding geometric proof). Their findings
reveal a massive "reasoning gap": models achiev-
ing high scores on general multimodal benchmarks
(e.g.,MMEB) exhibit a significant performance
drop when required to solve abstract visual analo-
gies or multi-image relational scenarios.
❷Expert-Level Multidisciplinary and Con-
tradiction Reasoning.Moving beyond general
knowledge,MRMR(Zhang et al., 2025e) intro-
ducesContradiction Retrieval, a novel task requir-
ing models to identify rules or requirements that
conflict with a given case description (e.g.,iden-
tifying a traffic violation in a visual scene based
3Instruction-following retrieval (e.g.,FollowIR (Weller
et al., 2024) and mFollowIR (Weller et al., 2025a)), long-
context retrieval (e.g.,LongEmbed (Zhu et al., 2024a)) and
code retrieval benchmarks (e.g.,CoIR (Li et al., 2024b) and
CodeRAG-Bench (Wang et al., 2024d)) are not within the
scope of reasoning-intensive retrieval in this survey.on a text-based rulebook). MRMR focuses on ex-
pert domains like medicine and engineering, where
visually rich documents (e.g.,pathological slides)
require specialized interpretation. Their study un-
derscores that text-only retrievers augmented with
high-quality captions often outperform native mul-
timodal models, suggesting that superior logical
deduction in LLMs still outweighs current visual-
semantic alignment in native multimodal embed-
ders.
❸Agentic, Multi-hop, and Process-Oriented
Reasoning.In complex enterprise scenarios,
M4DocBench(Dong et al., 2025c) redefines rea-
soning as an iterative, agentic process. It intro-
duces theM4 framework(Multi-modal, Multi-hop,
Multi-document, and Multi-turn), focusing on deep
research tasks where evidence is scattered across
dozens of documents. Unlike single-shot retrieval,
M4DocBench evaluates a system’s ability to per-
formQuery Decompositionand iterative refine-
ment. It underscores that deep document intelli-
gence requires "Strategic Planning"—the ability
to filter relevant documents from noisy collections
and adaptively select the optimal retrieval granular-
ity (e.g.,chunk, page, or summary) based on the
evolving state of the research workflow.
Future Frontiers.Despite these early triumphs,
several reasoning challenges remain largely unex-
plored. Future VDR research must addressimplicit
retrieval intents, where queries involve fuzzy con-
straints that can only be resolved through world-
knowledge synthesis (Zhang et al., 2025g; Wei
et al., 2024b). Furthermore, the development ofac-
tive retrieval agents—capable of self-correcting
their search path when initial reasoning leads to
dead ends—will be paramount to unlocking the
full potential of multimodal document intelligence
in open-domain, large-scale scenarios (Zhu et al.,
2024b; Liu et al., 2025c; Zhang et al., 2025d).
D Common VDR Training Sets
❶colpali-train-set.This dataset serves as the
training data for the ColPali model4and represents
a hybrid approach to data collection (Faysse et al.,
2024). It comprises approximately 127,000 query-
image pairs, strategically combining established
academic benchmarks (63%), such as DocVQA
(Mathew et al., 2021), InfoVQA (Mathew et al.,
2022), TAT-DQA (Zhu et al., 2022) and arXivQA
4https://huggingface.co/datasets/vidore/
colpali_train_set
26

(Li et al., 2024a), with a custom synthetic dataset
(37%). The synthetic component was created from
a diverse collection of web-crawled PDF docu-
ments, with pseudo-questions generated by Claude-
3 Sonnet. The dataset is intentionally English-only,
designed to facilitate research into zero-shot cross-
lingual generalization capabilities of VDR models.
❷VisRAG-Ret-Train-Synthetic-data.As the
synthetic training component for the VisRAG
model5(Yu et al., 2024; Sun et al., 2025b), this
dataset is composed entirely of VLM-generated
data, totaling around 239k query-document pairs.
The corpus was constructed from web-crawled
PDFs spanning diverse domains, including college-
level textbooks6, academic papers from premier
conferences like ICML’23 and NeurIPS’23, and
product manuals7. A powerful VLM, GPT-4o, was
leveraged to generate pseudo-queries for these doc-
ument pages, creating a large-scale resource tai-
lored for training retrieval models on a variety of
document layouts and topics.
❸vdr-multilingual-train.This dataset8marks
a significant step towards multilingual VDR, con-
taining nearly 500,000 query-image samples across
five languages (English, Spanish, Italian, German,
and French). Its construction involved a highly
sophisticated pipeline. First, a diverse corpus of
~50k documents was scraped from the internet us-
ing topic-based search queries for each language.
A key innovation was the use of layout analysis
to sample pages, ensuring an even distribution of
text-only, visual-only, and mixed-modality pages.
Synthetic queries were then generated using pow-
erful VLMs (Gemini-1.5-Pro and Qwen2-VL-72B)
with an advanced prompting technique that distin-
guished between general and specific questions to
improve query quality. The dataset underwent rig-
orous cleaning, filtering, and hard-negative mining
to enhance its utility for training robust retrieval
models.
❹VDR_MEGA_MultiDomain_DocRetrieval.
This dataset9represents the largest and most
comprehensive resource to date, amalgamat-
5https://huggingface.co/datasets/openbmb/
VisRAG-Ret-Train-Synthetic-data
6https://openstax.org/
7https://www.manualslib.com/
8https://huggingface.co/datasets/llamaindex/
vdr-multilingual-train
9https://huggingface.co/datasets/racineai/VDR_
MEGA_MultiDomain_DocRetrievaling approximately 1.09 million examples
across five languages. It functions as a meta-
dataset, strategically fusing the three previously
mentioned datasets ( colpali-train-set ,
VisRAG-Ret-Train-Synthetic-data , and
vdr-multilingual-train ) with several new,
domain-specific collections. These additions cover
specialized fields such as military10, energy11,
hydrogen technology12, and geotechnical engineer-
ing13. By unifying multiple datasets, it provides
unparalleled scale and diversity in both language
and subject matter, making it an ideal resource
for training highly generalized and robust VDR
systems.
❺docmatix.This is an interleaved multimodal
pre-training dataset14created for the modality-
aware continual pre-training of MoCa models
(Chen et al., 2025a). It is adapted from the orig-
inal Docmatix dataset (Laurençon et al., 2024),
a massive-scale Document Visual Question An-
swering resource containing approximately 2.4 mil-
lion images and 9.5 million question-answer pairs,
which was initially used to fine-tune the Idefics3
model. The adaptation process involves transform-
ing the original question-answer pairs by concate-
nating document screenshots with their correspond-
ing texts, thereby creating an interleaved format
suitable for continuous pre-training.
Common Characteristics.The current genera-
tion of VDR training sets reveals several unifying
trends. Firstly, there is a clear paradigm shift to-
wardsleveraging synthetic data generation at
scale. All major datasets heavily rely on powerful
VLMs to create pseudo-queries for large, unanno-
tated document corpora, effectively bypassing the
bottleneck of manual annotation. Secondly, these
corpora are primarily built fromweb-crawled PDF
documents, which provides a rich diversity of lay-
outs, domains, and styles that mirror real-world
scenarios. Thirdly, there is a growing emphasis
onsophisticated data curation, including tech-
niques like layout-aware page sampling and auto-
10https://huggingface.co/datasets/racineai/VDR_
Military
11https://huggingface.co/datasets/racineai/VDR_
Energy
12https://huggingface.co/datasets/racineai/VDR_
Hydrogen
13https://huggingface.co/datasets/racineai/VDR_
Geotechnie
14https://huggingface.co/datasets/moca-embed/
docmatix
27

mated hard-negative mining, to improve training
efficiency and model performance. Finally, a clear
trajectory exists towardsincreasing scale and mul-
tilingualism, evolving from English-only datasets
to massive, multi-language compilations that en-
able the development of globally competent mod-
els.
Future Directions.Looking ahead, the optimiza-
tion of VDR training sets can be advanced in sev-
eral key directions. The most critical frontier is
movingbeyond simple semantic matching to-
wards reasoning-intensive data. Future datasets
should include query-document pairs that necessi-
tate multi-hop, logical, or causal reasoning to find
the correct answer, mirroring the challenges posed
by benchmarks like MRMR and M4DocBench.
Secondly, there is a need forenhanced data au-
thenticity and complexity. While VLM-generated
queries are scalable, they can lack the nuance and
"messiness" of real user intent. Future work could
explore mining queries from anonymized user logs
or using agentic workflows to simulate more realis-
tic information-seeking behaviors. Lastly, training
sets could benefit fromfiner-grained structural
annotations. Instead of just matching a query to
a page, future datasets could provide explicit links
to specific sub-page elements (e.g.,a single row
in a table, a data point in a chart, or a specific
paragraph), which would be invaluable for train-
ing models that can perform precise, element-level
evidence retrieval.
EMore Discussion of Integration of VDR
with RAG and Agent
E.1 Current Paradigms and Key Trends
The integration of VDR into RAG and Agentic sys-
tems has moved beyond simple document fetching,
evolving into sophisticated frameworks that emu-
late human-like reasoning and interaction (Li et al.,
2025g; Arya and Gaud, 2025). This evolution is
characterized by several key trends, progressing
from foundational end-to-end pipelines to complex,
iterative, and deeply aware reasoning workflows.
❶Foundational Multimodal RAG Pipelines.
The most fundamental shift has been the move from
brittle OCR-based textual RAG to robust, end-to-
end multimodal pipelines that process documents
as visual inputs. This paradigm preserves critical
layout and graphical information often lost in text
extraction. A prime example is M3DocRAG (Choet al., 2024), which establishes a flexible frame-
work for multi-page and multi-document question
answering by directly retrieving relevant page im-
ages for a multimodal generator, forming a founda-
tional approach for subsequent innovations.
❷Expansion of Interaction Modalities.Build-
ing on the visual-centric pipeline, researchers are
expanding the interaction modalities beyond tra-
ditional text-based queries to create more natural
and accessible interfaces. A pioneering work in
this direction is TextlessRAG (Xie et al., 2025),
which introduces a fully "textless" pipeline that di-
rectly processes speech queries and generates spo-
ken answers without any explicit Automatic Speech
Recognition (ASR) or Text-to-Speech (TTS) steps,
showcasing the potential to significantly broaden
the application scenarios of VDR.
❸Emergence of Agentic and Iterative Reason-
ing Workflows.A dominant trend is the replace-
ment of static, linear RAG pipelines with dynamic,
agentic systems that perform multi-step, iterative
reasoning. These systems decompose complex
queries and progressively refine evidence, mim-
icking human research processes.
•Task Decomposition and Collaboration:Many
frameworks now employ a "society of agents,"
where specialized agents collaborate to solve a
problem. For instance, MDocAgent (Han et al.,
2025b) utilizes a team of five agents (e.g., Gen-
eral, Critical, Text, and Image agents) to syn-
thesize insights from different modalities. Simi-
larly, ViDoRAG (Wang et al., 2025b) introduces
a coarse-to-fine workflow where a "Seeker" agent
hunts for relevant images and an "Inspector"
agent provides detailed feedback, enabling itera-
tive evidence refinement.
•Iterative Deep Research Workflows:This agen-
tic concept is scaled further in systems designed
for "deep research." Doc-Researcher (Dong et al.,
2025c) implements a comprehensive multi-agent
framework with a "Planner" for query decomposi-
tion and a "Searcher-Refiner" loop that iteratively
gathers and filters evidence across multiple docu-
ments and granularities (e.g., chunks, pages, or
summaries).
❹Enhancing Core RAG Components with Ad-
vanced Mechanisms.Beyond structuring the
workflow, significant innovation is occurring within
28

the core retrieval and generation components them-
selves to make them more intelligent and aware.
•Holistic Knowledge Retrieval and Fusion:To
address the challenge that standard retrieval of-
ten misses nuanced information, new methods
are designed for more comprehensive knowl-
edge extraction. HKRAG (Tong et al., 2025)
explicitly models and retrieves both "salient" and
"fine-print" knowledge using a hybrid masking
retriever and an uncertainty-guided generator. In
a similar vein, VisDoMRAG (Suri et al., 2025)
runs parallel textual and visual RAG pipelines
and then employs a consistency-constrained fu-
sion mechanism to intelligently integrate their
outputs.
•Active Visual Perception and Learning:The
most advanced systems empower agents with the
ability to actively interact with retrieved visual
content. VRAG-RL (Wang et al., 2025c) pio-
neers this by defining a visual perception action
space that allows an agent to perform actions like
"crop" and "zoom" on retrieved images. This
interactive process is optimized using reinforce-
ment learning, enabling the agent to actively seek
out fine-grained details in a coarse-to-fine man-
ner, much like a human analyst.
In summary, the application of VDR in RAG
and Agentic systems is rapidly maturing from a
simple retrieval-and-generation process to highly
dynamic, interactive, and collaborative workflows.
The field is pushing towards systems that not only
find relevant documents but also intelligently rea-
son, synthesize, and interact with multimodal infor-
mation to solve complex problems (Ashraf et al.,
2025; Zhang et al., 2025i; Yan et al., 2025c; Su
et al., 2025a).
E.2 The Evolutionary Synergy of VDU and
VDR
The boundary between Visual Document Under-
standing (VDU) and VDR is rapidly dissolving in
the MLLM era. Modern RAG pipelines are evolv-
ing into autonomous agentic systems capable of
perception, strategic planning, and iterative refine-
ment.
VDU Trends in RAG and Agentic Systems.Re-
cent breakthroughs emphasize moving from single-
turn OCR-based retrieval to multi-step reasoning-
driven navigation. (Sourati et al., 2025) introducesLAD-RAG, which leverages an LLM agent to dy-
namically interact with a symbolic document graph
and neural indices, capturing structural dependen-
cies missed by dense embeddings. (Yu et al.,
2025b) proposes MACT, a collaborative frame-
work that decomposes document intelligence into
planning, execution, and judgment agents, imple-
menting a self-correction loop via procedural scal-
ing. To address evidence sparsity, MHier-RAG
(Gong et al., 2025) facilitates multi-granularity rea-
soning by retrieving parent pages and document
summaries, while MoLoRAG (Wu et al., 2025)
constructs page graphs to navigate logical con-
nections beyond surface-level semantic similarity.
For fine-grained localization, DocLens (Zhu et al.,
2025a) employs a tool-augmented “zoom-in” strat-
egy where agents locate specific visual elements
like tables or charts within retrieved pages. Finally,
SLEUTH (Liu et al., 2025b) optimizes the input
quality through context engineering, utilizing page-
screening agents to filter visual noise and construct
evidence-dense multimodal contexts.
Comparison between VDR and VDU.Table 5
illustrates the multi-dimensional differences and
the emerging convergence between retrieval and
understanding tasks. While VDR focuses on effi-
cient large-scale candidate localization, VDU em-
phasizes deep reasoning. The integration of the
two, as seen in recent agentic frameworks, enables
systems to perform “Retrieval as Understanding.”
Reflections on VDR: Toward Cognitive Discov-
ery.Future VDR systems should evolve from
passive semantic matchers to active cognitive nav-
igators. The integration of reinforcement learn-
ing (as in VRAG-RL (Wang et al., 2025c)) and
symbolic-neural fusion (as in LAD-RAG (Sourati
et al., 2025)) suggests that the next frontier of mul-
timodal document intelligence lies in the ability to
understand the “logical topology” of a document.
Rather than just returning snippets, future retrievers
will actively discover implicit contradictions and
synthesize cross-document insights, transforming
VDR into a tool for complex decision support in
expert domains.
F VDR Challenges and Actionable
Solutions
F.1 Expanding the Data Frontier
The foundation of robust VDR systems is high-
quality, diverse, and challenging data, yet the cur-
29

Dimension Visual Document Retrieval (VDR) Visual Document Understanding (VDU) Convergence Trend
Core ObjectiveCandidate page localization Information extraction & reasoning R→UIntegrated
GranularityCoarse (Document/Page-level) Fine (Element/Token-level) Hierarchical Indexing
ComputationHigh (Scanning entire corpus) Low (Deep processing of Top-K) Adaptive Resource Scaling
Key TechniqueLate Interaction, Multi-vector Indexing Visual CoT, Multimodal Fusion Agent-driven Seek-then-Verify
Main MetricnDCG, Recall@K Accuracy, F1-score Source Attribution Accuracy
Table 5:Multi-dimensional comparisons between VDR and VDU. We highlight that recent Agentic systems are bridging the
gap through iterative reasoning and dynamic filtering.
rent landscape of benchmarks and training sets
presents a significant bottleneck to progress. A
primary limitation is the lack of diversity in lan-
guage, domain, and document structure. While
recent benchmarks like Jina-VDR (Günther et al.,
2025) and MIRACL-VISION (Osmulski et al.,
2025) have made commendable strides in multi-
lingual support, the majority of available resources
remain predominantly English-centric and confined
to general-domain web documents. Furthermore,
most benchmarks focus on retrieving single, self-
contained pages, failing to capture the complex,
multi-hop, and cross-document reasoning required
for genuine information synthesis—a gap that pio-
neering benchmarks like MR2-Bench (Zhou et al.,
2025a) and MRMR (Zhang et al., 2025e) are only
beginning to explore. A more insidious issue lies
in data provenance; the heavy reliance on VLM-
generated queries for training data, a common prac-
tice in datasets, risks creating a hermetic feedback
loop where models are evaluated on the same kind
of logic they were trained on, inflating performance
metrics without guaranteeing real-world generaliza-
tion. This is compounded by the potential for data
leakage between large, web-crawled training sets
and evaluation benchmarks, which casts a shadow
over the true robustness of current models (Apicella
et al., 2025; Lilja et al., 2024).
Addressing these data frontiers requires a multi-
pronged, forward-looking strategy that moves be-
yond mere data scaling. A primary imperative is
the creation of large-scale benchmarks that arenot
only multilingual but also multi-domain, extend-
ing into specialized corpora like legal contracts,
financial reports, and medical records, which de-
mand expert-level nuance (Tang and Yang, 2024;
Cao, 2024; Chung et al., 2025). To cultivate
reasoning-intensive capabilities, future data cu-
ration can leverage agentic frameworks to automat-
ically generate complex, multi-hop query-evidence
chains that span multiple documents, simulating
the realistic research workflows envisioned by sys-
tems like Doc-Researcher (Dong et al., 2025c). Tobreak the cycle of synthetic data bias and enhance
authenticity, a pivot towardsincorporating real hu-
man queriesis essential; this could be achieved by
mining anonymized search logs or utilizing human-
in-the-loop annotation platforms to capture the am-
biguity and complexity of genuine user intent. Fi-
nally, to ensurerobust and fair evaluation, the
community must adopt stricter data governance
practices. This includes establishing "firewalled"
test sets derived from entirely held-out web crawls
or proprietary document collections and develop-
ing standardized protocols for data contamination
detection, thereby fostering a more reliable assess-
ment of model generalization and pushing the field
towards true document intelligence (Song et al.,
2024; Xu et al., 2024; Samuel et al., 2025).
F.2 Rethiking Architectural Paradigms
The multimodal retrieval field has largely
converged on a paradigm centered around
contrastively-trained discriminative models, typi-
cally bi-encoder architectures that learn to max-
imize the similarity between positive query-
document pairs while minimizing it for negative
ones. While highly effective for semantic matching,
this approach has inherent limitations. It primarily
treats powerful MLLMs as static feature extractors
(Mischler et al., 2024; Jeong et al., 2024), underuti-
lizing their sophisticated generative and reasoning
capabilities, which can lead to information bottle-
necks where fine-grained details or implicit docu-
ment semantics are lost during the compression into
a fixed-size embedding. Moreover, this "one-size-
fits-all" strategy forces a single, dense model to act
as a generalist for an incredibly diverse array of
document structures—from text-heavy reports and
dense tables to complex charts and forms—limiting
both performance specialization and computational
efficiency (Gan et al., 2025b; Fan et al., 2024; Lo
et al., 2025).
To overcome these limitations, future research is
pivoting towards two promising architectural shifts:
autoregressive retrieval and the integration of Mix-
30

Representative Works Modality Core Paradigm Generated Content Embedding Source Training Objective
InBedder(Peng et al., 2024) Text Embed-via-Answering Concise Answer Hidden state of the 1st generated token Instruction Tuning (QA)
GIRCSE(Tsai et al., 2025) Text Iterative Refinement Soft Tokens (Non-readable) Pooled hidden states of generated soft tokens Iterative Contrastive Refinement (ICR)
GritLM(Muennighoff et al., 2024) Text Task Unification Text Response or Embedding Mean pooling over input with bidirectional attention Joint Contrastive + LM Loss
RGE(Liu et al., 2025a) /TTE(Cui et al., 2025) Multimodal Think-then-Embed Reasoning Trace (CoT) Hidden state conditioned on generated reasoning Joint LM (on trace) + Contrastive Loss
CAFe(Yu et al., 2025a) /VladV A(Ouali et al., 2025) Multimodal Joint Training N/A (for retrieval) Last token’s hidden state Joint Contrastive + Autoregressive Loss
Retrv-R1(Zhu et al., 2025b) Multimodal Reasoning-driven Selection Reasoning trace leading to a selection Not an embedding model; selects best candidate SFT + Reinforcement Learning (GRPO)
Table 6:A multi-dimensional comparison of representative works in generative retrieval. These approaches move beyond static
encoding by leveraging the generative capabilities of (M)LLMs. They differ in their core paradigm, the nature of the generated
content (from concise answers to reasoning traces), and their training objectives, which range from joint contrastive-generative
losses to reinforcement learning.
ture of Experts (MoE). The first paradigmreframes
retrieval from a purely discriminative task to a
generative one(Deng et al., 2025a; Xiong et al.,
2024). For instance, future models could adopt an
"embed-via-answering" framework, where embed-
dings are derived from generating a concise answer
to an instruction, as pioneered by InBedder (Peng
et al., 2024). More advanced paradigms may
even involve generating sequences of non-human-
readable "soft tokens" that iteratively refine the em-
bedding, a concept introduced by GIRCSE (Tsai
et al., 2025) which learns to "speak an embed-
ding language" and shows improved quality with
more generation steps at test time. This generative-
representational synergy is also validated by frame-
works like GritLM (Muennighoff et al., 2024),
which successfully unifies generative and embed-
ding tasks within a single model, demonstrating
that these capabilities can coexist without perfor-
mance degradation. We summarize and compare
the representative works from text-only and multi-
modal generative retrieval in Table 6.
Concurrently,the MoE architecture offers a
path to efficient specialization. For instance, dif-
ferent experts can be trained to handle distinct doc-
ument modalities (e.g., text, tables, audio), a direc-
tion explored by Uni-MoE (Li et al., 2025i,h) and
M3-JEPA (Lei et al., 2024), which use multi-gate
MoE to disentangle modality-specific and shared
information. This architecture also unlocks novel
embedding sources, as demonstrated by MOEE (Li
and Zhou, 2024), which shows that the expert rout-
ing weights themselves can serve as a potent, com-
plementary embedding signal. Together, these ar-
chitectural innovations promise to evolve VDR
from static matching towards more dynamic, spe-
cialized, and generative systems.
F.3 Performance-Efficiency Dilemma
The high accuracy of state-of-the-art multi-vector
VDR models stems directly from their fine-grained
representation paradigm, where each document
page is encoded into hundreds or even thousandsof patch-level embeddings (Faysse et al., 2024).
While this enables precise query-to-patch match-
ing, it creates a significant performance-efficiency
dilemma. The primary issue is the prohibitive stor-
age overhead; for instance, a medium-sized docu-
ment of just 50 pages can require approximately
10 MB of storage for its embeddings alone (Ma
et al., 2025), rendering the large-scale deployment
of such models economically and practically chal-
lenging. Furthermore, this abundance of vectors
increases online computational costs during the
late-interaction scoring stage. This dilemma has
spurred research into embedding reduction tech-
niques to create more compact yet effective docu-
ment representations.
Future solutions to this dilemma are evolving
from post-hoc compression to built-in adaptability.
The two primary post-hoc strategies areembedding
pruning and merging. Pruning aims to discard re-
dundant embeddings. While early heuristic-based
methods proved unstable (Liu et al., 2024; Las-
sance et al., 2021, 2022), more sophisticated ap-
proaches like DocPruner (Yan et al., 2025d) use
attention scores to adaptively prune 50-60% of
patches per document with minimal performance
loss. Alternatively, merging or clustering aggre-
gates similar patches, a method that some argue
is more suitable as it retains information rather
than discarding it. For example, Light-ColPali
(Ma et al., 2025) employs hierarchical clustering
to merge embeddings, while HEA VEN (Kim et al.,
2025a) uses visually-summarized pages to create a
compact first-stage index.
A more forward-looking paradigm isMa-
tryoshka Representation Learning (MRL)(Kusu-
pati et al., 2022), which trains embeddings to be
inherently flexible. MRL ensures that a single
high-dimensional embedding contains a hierarchy
of smaller, nested embeddings that are also effec-
tive. The flexible Matryoshka embedding settings
have applied in top performing text embedding
models (e.g.,Qwen3 Embedding (Zhang et al.,
2025h), Gemini Embedding (Lee et al., 2025b),
31

Table 7:A multi-dimensional comparison of representative works in Matryoshka Representation Learning (MRL). The principle
of nested, adaptable representations has been extended from embedding dimensions to model depth, token counts, and even
quantization bit-widths, addressing a wide range of tasks from efficient retrieval to model interpretability.
Representative Works Modality Target Task Matryoshka On Core Innovation
MRL(Kusupati et al., 2022) Text / Vision Classification / Retrieval Embedding Dimension Foundational concept of nested embeddings.
2DMSE(Li et al., 2024c) Text STS / Retrieval Embedding Dimension & Model Depth Extends MRL to two dimensions for greater flexibility.
Starbucks(Zhuang et al., 2024) Text STS / Retrieval Embedding Dimension & Model Depth Improves 2DMSE to match individually trained models.
MatTA(Verma et al., 2025) Generic General Tasks Model Depth & Width (FFN) Teacher-TA-Student distillation for elastic student models.
M3(Cai et al., 2024) /MQT(Hu et al., 2024) Multimodal VQA / Reasoning Number of Visual Tokens Enables adaptive visual granularity in LVLMs.
Matryoshka Re-Ranker(Liu et al., 2025g) Text Re-ranking Model Depth & Sequence Length Creates flexible re-rankers with configurable architecture.
SMEC(Zhang et al., 2025a) Text / Vision Retrieval / Classification Embedding Dimension Sequential training to mitigate gradient variance.
Matryoshka-Adaptor(Yoon et al., 2024) Text / Multimodal Retrieval Embedding Dimension Post-hoc tuning to impart Matryoshka properties.
MatMamba(Shukla et al., 2024) Language / Vision LM / Classification Hidden Dimension & Heads Generalizes MRL to State Space Model (SSM) architectures.
Matryoshka SAE(Bussmann et al., 2025) Model Activations Interpretability SAE Dictionary Size Mitigates feature splitting in Sparse Autoencoders.
Matryoshka Quantization(Nair et al., 2025) Model Weights Model Compression Quantization Bit-width Leverages nested structure of integer data types.
KaLM-Embedding-V2 (Zhao et al., 2025), Embed-
dingGemma (Vera et al., 2025), jina-embeddings-
v3 (Sturua et al., 2024), jina-colbert-v2 (Jha et al.,
2024)) across diverse real-world tasks (Lai et al.,
2024; Wang et al., 2024c; Nacar and Koubaa, 2025;
Hanley and Durumeric, 2025; Fu et al., 2025). For-
mally, given an encoder E, a document page dis
mapped to a full embedding ed=E(d)∈RD.
MRL trains the model on a set of nested dimensions
M={m 1, m2, . . . , m k}where m1< m 2<
. . . < m k=D , such that each prefix e[mi]
d∈Rmi
(the first micomponents of ed) is a valid repre-
sentation. The training objective LMRL sums a
standard representation loss ℓ(·,·) over all nested
dimensions:
LMRL(q, d+) =X
mi∈Mwi·ℓ
E(q)[mi],E(d+)[mi]
(11)
whereqis a query,d+is a positive document, and
wiare loss weights. This allows practitioners to
truncate stored embeddings at inference time, pro-
viding a flexible dial to balance performance and
efficiency. Building on this, 2DMSE (Li et al.,
2024c) extends this elasticity to model depth, while
frameworks like Starbucks (Zhuang et al., 2024)
and MatTA (Verma et al., 2025) refine the training
process to bridge the performance gap with indi-
vidually trained models. The Matryoshka principle
has also been generalized to new granularities and
architectures: M3 (Cai et al., 2024) and MQT (Hu
et al., 2024) apply it to the number of visual to-
kens in LVLMs; Matryoshka Re-Ranker (Liu et al.,
2025g) enables flexible depth and sequence length
for re-ranking; and MatMamba (Shukla et al., 2024)
integrates it into State Space Models. The applica-
tion of MRL has further expanded to diverse tasks
such as improving feature hierarchy in Sparse Au-
toencoders (Bussmann et al., 2025) and enabling
multi-precision models via Matryoshka Quantiza-
tion (Nair et al., 2025). Finally, innovations inthe training paradigm itself, such as the sequen-
tial learning in SMEC (Zhang et al., 2025a) and
the post-hoc tuning of Matryoshka-Adaptor (Yoon
et al., 2024), are making the creation of these flexi-
ble embeddings more efficient and accessible. We
compare the works above in Table 7.
F.4 Towards Interactive Retrieval
Integrating VDR with agentic systems holds im-
mense potential, particularly for complex scenarios
likeDeep Research, where a query requires itera-
tive evidence gathering from a vast candidate pool
(Java et al., 2025; Chen et al., 2025e; Huang et al.,
2025b; Zhang et al., 2025f). The current challenge,
however, is that most agentic systems treat VDR
models as passive, reactive tools rather than active,
strategic partners. These agents typically operate
within a predefined, reactive loop of reasoning and
acting, which limits their ability to dynamically for-
mulate or adapt a high-level retrieval strategy. For
instance, while frameworks like WebThinker (Li
et al., 2025f) and DeepResearcher (Zheng et al.,
2025b) enable agents to interleave thought with
web search actions, they often function as sophis-
ticatedtool-executorsthat reactively process in-
formation rather than asresearch strategiststhat
proactively plan a multi-step information-seeking
campaign. This limitation becomes particularly ev-
ident in complex research tasks, such as generating
structured reports with dynamic outlines as demon-
strated by WebWeaver (Li et al., 2025j) or creat-
ing text-chart interleaved content as in Multimodal
DeepResearcher (Yang et al., 2025b), where the
retrieval process itself must be strategically guided
to gather diverse and structured evidence.
Future work must focus on theco-design of
agents and VDR tools to foster a more organic
and strategic synergy. A primary direction is to
empower agents with explicit retrieval planning
capabilities, enabling them to decompose a high-
level query into a multi-step, multi-granularity re-
32

trieval plan. This moves beyond simple tool use
towards strategic orchestration. For example, a
planner agent, inspired by the hierarchical structure
of Cognitive Kernel-Pro (Fang et al., 2025), could
adaptively selectretrieval granularity—choosing
between document summaries, full pages, or spe-
cific visual elements—and delegate these sub-tasks
to specialized agents. To handle the inevitable
"dead ends" in complex research, agents must
also develop self-correction mechanisms. Frame-
works like DeepAgent (Li et al., 2025e), with
its autonomous memory folding, offer a powerful
blueprint by allowing an agent to "take a breath,"
discard a failed exploration path, and restart from
a consolidated memory state. To effectively learn
such complex strategies, agents require more fine-
grained feedback. This can be achieved through
mechanisms like the Atomic Thought Rewards
(ATR) proposed in Atom-Searcher (Deng et al.,
2025b), which uses a reasoning reward model to
provide process-level supervision on the agent’s re-
trieval strategy. Finally, for massive-scale research,
parallel exploration frameworks like the Research-
Synthesis approach in WebResearcher (Qiao et al.,
2025) and Tongyi DeepResearch (Team et al.,
2025b), where multiple agents conduct concurrent
searches, can transform interactive retrieval into a
scalable, collaborative process. The ultimate goal
is to evolve the VDR system from a passive tool
into an active,strategic partnerin the knowledge
discovery process.
F.5 Uncovering Scaling Laws for VDR
While scaling laws—predictable power-law rela-
tionships between model performance, size, and
data volume—are well-documented for general-
purpose LLMs (Alabdulmohsin et al., 2022; Zhang
et al., 2024a; Xiao et al., 2025a; Pearce et al., 2024),
their application to the specialized domain of VDR
remains a largely unexplored and complex fron-
tier. The core challenge is that the scaling behavior
in VDR is not a straightforward extrapolation of
model size or data quantity. On the model axis,
naively increasing parameter count does not guar-
antee proportional performance gains. Larger mod-
els, without proper fine-tuning, can exhibit greater
embedding anisotropy, which paradoxically harms
retrieval performance in zero-shot settings by com-
pressing the effective embedding space, even as
they capture richer features for transfer tasks (Jiang
et al., 2024a). This complexity is mirrored on the
data axis, where simply increasing the volume ofraw documents is insufficient. The effectiveness
of VDR models is deeply tied to the quality and
diversity of the training pairs. Recent studies pow-
erfully illustrate this, showing that models trained
on smaller, high-quality synthetic datasets can sig-
nificantly outperform those trained on orders-of-
magnitude more, but less curated, data. For exam-
ple, both mmE5 (Chen et al., 2025b) and MegaPairs
(Zhou et al., 2025b) demonstrated that superior per-
formance can be achieved with a fraction of the
data if it is diverse and of high quality, underscoring
that scaling in VDR is a nuanced interplay between
model capacity and data sophistication, rather than
a simple numbers game.
To systematically navigate this challenge, future
research must pivot from ad-hoc scaling toestab-
lishing formal, VDR-specific scaling laws, adapt-
ing methodologies from the dense retrieval do-
main(Fang et al., 2024). A foundational step is to
adopt continuous and sensitive evaluation metrics,
such as contrastive entropy, which can more ac-
curately capture subtle performance changes com-
pared to discrete ranking metrics like nDCG, thus
enabling precise power-law curve fitting for model
size (N) and data volume ( D). Concurrently, ad-
vancing the "data" axis of the scaling law requires
moving beyond simple data augmentation to so-
phisticated, document-aware synthetic data gener-
ation. This involves leveraging MLLMs not just
for query generation but for creating entire ecosys-
tems of diverse training tasks (Wang et al., 2024a),
mining heterogeneous relationships between docu-
ments using multiple similarity models (e.g.,visual-
semantic and visual-pattern correlations) (Zhou
et al., 2025b), and implementing rigorous quality
control through mechanisms like single-pass multi-
aspect interpretation and self-refinement (Chen
et al., 2025b). By combining a formal scaling law
framework with advanced data synthesis, the VDR
community can quantitatively model the trade-offs
between investing in larger models versus more
extensive data annotation. This will enable more
efficient resource allocation and pave the way for
building maximally effective VDR systems under
practical budget constraints.
G Usage of AI Assistant
The language and expression throughout this
manuscript were polished with the assistance of
Gemini-2.5-Pro. This AI tool was utilized exclu-
sively for refining grammar, improving sentence
33

structure, and enhancing the overall clarity and
readability of the text. The conceptual framework,
structural organization, technical analyses, and all
core ideas presented in this survey are the original
work of the authors.
34
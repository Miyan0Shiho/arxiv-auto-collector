# MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System

**Authors**: Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang, Simin Niu, Bo Tang, Feiyu Xiong, Zhiyu Li

**Published**: 2025-03-12 17:59:42

**PDF URL**: [http://arxiv.org/pdf/2503.09600v1](http://arxiv.org/pdf/2503.09600v1)

## Abstract
Retrieval-Augmented Generation (RAG), while serving as a viable complement to
large language models (LLMs), often overlooks the crucial aspect of text
chunking within its pipeline. This paper initially introduces a dual-metric
evaluation method, comprising Boundary Clarity and Chunk Stickiness, to enable
the direct quantification of chunking quality. Leveraging this assessment
method, we highlight the inherent limitations of traditional and semantic
chunking in handling complex contextual nuances, thereby substantiating the
necessity of integrating LLMs into chunking process. To address the inherent
trade-off between computational efficiency and chunking precision in LLM-based
approaches, we devise the granularity-aware Mixture-of-Chunkers (MoC)
framework, which consists of a three-stage processing mechanism. Notably, our
objective is to guide the chunker towards generating a structured list of
chunking regular expressions, which are subsequently employed to extract chunks
from the original text. Extensive experiments demonstrate that both our
proposed metrics and the MoC framework effectively settle challenges of the
chunking task, revealing the chunking kernel while enhancing the performance of
the RAG system.

## Full Text


<!-- PDF content starts -->

MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented
Generation System
Jihao Zhao1Zhiyuan Ji1Zhaoxin Fan2Hanyu Wang1Simin Niu1Bo Tang2
Feiyu Xiong2Zhiyu Li2*
1School of Information, Renmin University of China, Beijing, China
2Institute for Advanced Algorithms Research, Shanghai, China
Abstract
Retrieval-Augmented Generation (RAG), while
serving as a viable complement to large lan-
guage models (LLMs), often overlooks the cru-
cial aspect of text chunking within its pipeline.
This paper initially introduces a dual-metric
evaluation method, comprising Boundary Clar-
ity and Chunk Stickiness, to enable the di-
rect quantification of chunking quality. Lever-
aging this assessment method, we highlight
the inherent limitations of traditional and se-
mantic chunking in handling complex contex-
tual nuances, thereby substantiating the neces-
sity of integrating LLMs into chunking pro-
cess. To address the inherent trade-off between
computational efficiency and chunking preci-
sion in LLM-based approaches, we devise the
granularity-aware Mixture-of-Chunkers (MoC)
framework, which consists of a three-stage pro-
cessing mechanism. Notably, our objective
is to guide the chunker towards generating a
structured list of chunking regular expressions,
which are subsequently employed to extract
chunks from the original text. Extensive ex-
periments demonstrate that both our proposed
metrics and the MoC framework effectively set-
tle challenges of the chunking task, revealing
the chunking kernel while enhancing the per-
formance of the RAG system1.
1 Introduction
Retrieval-augmented generation (RAG), as a
cutting-edge technological paradigm, aims to ad-
dress challenges faced by large language models
(LLMs), such as data freshness (He et al., 2022),
hallucinations (Bénédict et al., 2023; Chen et al.,
2023; Zuccon et al., 2023; Liang et al., 2024),
and the lack of domain-specific knowledge (Li
et al., 2023; Shen et al., 2023). This is particularly
relevant in knowledge-intensive tasks like open-
domain question answering (QA) (Lazaridou et al.,
*Corresponding author: lizy@iaar.ac.cn
1Our code is available at https://github.com/
IAAR-Shanghai/Meta-Chunking/tree/main/MoC .2022). By integrating two key components: the
retriever and the generator, this technology enables
more precise responses to input queries (Singh
et al., 2021; Lin et al., 2023). While the feasibil-
ity of the retrieval-augmentation strategy has been
widely demonstrated through practice, its effective-
ness heavily relies on the relevance and accuracy of
the retrieved documents (Li et al., 2022; Tan et al.,
2022). The introduction of excessive redundant or
incomplete information through retrieval not only
fails to enhance the performance of the generation
model but may also lead to a decline in answer
quality (Shi et al., 2023; Yan et al., 2024).
In response to the aforementioned challenges,
current research efforts mainly focus on two as-
pects: improving retrieval accuracy (Zhuang et al.,
2024; Sidiropoulos and Kanoulas, 2022; Guo et al.,
2023) and enhancing the robustness of LLMs
against toxic information (Longpre et al.; Kim et al.,
2024). However, in RAG systems, a commonly
overlooked aspect is the chunked processing of tex-
tual content, which directly impacts the quality of
dense retrieval for QA (Xu et al., 2023). This is
due to the significant “weakest link” effect in the
performance of RAG systems, where the quality
of text chunking constrains the retrieved content,
thereby influencing the accuracy of generated an-
swers (Ru et al., 2024). Despite advancements in
other algorithmic components, incremental flaws
in the chunking strategy can still detract from the
overall system performance to some extent.
Given the critical role of text chunking in RAG
systems, optimizing this process has emerged as
one of the key strategy to mitigate performance
bottlenecks. Traditional text chunking methods,
often based on rules or semantic similarity (Zhang
et al., 2021; Langchain, 2023; Lyu et al., 2024),
provide some structural segmentation but are in-
adequate in capturing subtle changes in logical re-
lationships between sentences. The LumberChun-
ker (Duarte et al., 2024) offers a novel solution byarXiv:2503.09600v1  [cs.CL]  12 Mar 2025

utilizing LLMs to receive a series of consecutive
paragraphs and accurately identify where content
begins to diverge. However, it demands a high
level of instruction-following ability from LLMs,
which incurs significant resource and time costs.
Additionally, the effectiveness of current chunk-
ing strategies is often evaluated indirectly through
downstream tasks, such as the QA accuracy in RAG
systems, with a lack of independent metrics for
evaluating the inherent rationality of the chunking
process itself. These challenges give rise to two
practical questions: This raises a practical question:
How can we fully utilize the powerful reasoning
capabilities of LLMs while accomplishing the text
chunking task at a lower cost? And how to devise
evaluation metrics that directly quantify the validity
of text chunking?
Inspired by these observations, we innovatively
propose two metrics, Boundary Clarity and
Chunk Stickiness , to independently and effec-
tively assess chunking quality. Concurrently, we
leverage these metrics to delve into the reasons
behind the suboptimal performance of semantic
chunking in certain scenarios, thereby highlighting
the necessity of LLM-based chunking. To miti-
gate the resource overhead of chunking without
compromising the inference performance of LLMs,
we introduce the Mixture-of-Chunkers (MoC)
framework. This framework primarily comprises
a multi-granularity-aware router, specialized meta-
chunkers, and a post-processing algorithm.
This mechanism adopts a divide-and-conquer
strategy, partitioning the continuous granularity
space into multiple adjacent subdomains, each cor-
responding to a lightweight, specialized chunker.
The router dynamically selects the most appropri-
ate chunker to perform chunking operation based
on the current input text. This approach not only
effectively addresses the “granularity generaliza-
tion dilemma” faced by traditional single-model
approaches but also maintains computational re-
source consumption at the level of a single small
language model (SLM) through sparse activation,
achieving an optimal balance between accuracy
and efficiency for the chunking system. It is crucial
to emphasize that our objective is not to require
the meta-chunker to generate each text chunk in
its entirety. Instead, we guide the model to gen-
erate a structured list of chunking regular expres-
sions used to extract chunks from the original text.
To address potential hallucination phenomena of
meta-chunker, we employ an edit distance recov-ery algorithm, which meticulously compares the
generated chunking rules with the original text and
subsequently rectifies the generated content.
The main contributions of this work are as fol-
lows:
•Breaking away from indirect evaluation
paradigms, we introduce the dual metrics of
Boundary Clarity and Chunk Stickiness to
achieve direct quantification of chunking qual-
ity. By deconstructing the failure mechanisms
of semantic chunking, we provide theoreti-
cal validation for the involvement of LLM in
chunking tasks.
•We devise the MoC architecture, a hy-
brid framework that dynamically orchestrates
lightweight chunking experts via a multi-
granularity-aware router. This architecture
innovatively integrates: a regex-guided chunk-
ing paradigm, a computation resource con-
straint mechanism based on sparse activation,
and a rectification algorithm driven by edit
distance.
•To validate the effectiveness of our pro-
posed metrics and chunking method, we con-
duct multidimensional experiments using five
different language models across four QA
datasets, accompanied by in-depth analysis.
2 Related Works
Text Segmentation It is a fundamental task in
NLP, aimed at breaking down text content into its
constituent parts to lay the foundation for subse-
quent advanced tasks such as information retrieval
(Li et al., 2020) and text summarization (Lukasik
et al., 2020; Cho et al., 2022). By conducting
topic modeling on documents, (Kherwa and Bansal,
2020) and (Barde and Bainwad, 2017) demonstrate
the identification of primary and sub-topics within
documents as a significant basis for text segmenta-
tion. (Zhang et al., 2021) frames text segmentation
as a sentence-level sequence labeling task, utiliz-
ing BERT to encode multiple sentences simulta-
neously. It calculates sentence vectors after mod-
eling longer contextual dependencies and finally
predicts whether to perform text segmentation after
each sentence. (Langchain, 2023) provides flexible
and powerful support for various text processing
scenarios by integrating multiple text segmenta-
tion methods, including character segmentation,

delimiter-based text segmentation, specific docu-
ment segmentation, and recursive chunk segmen-
tation. Although these methods better respect the
structure of the document, they have limitations in
deep contextual understanding. To address this is-
sue, semantic-based segmentation (Kamradt, 2024)
utilizes embeddings to aggregate semantically sim-
ilar text chunks and identifies segmentation points
by monitoring significant changes in embedding
distances.
Text Chunking in RAG By expanding the in-
put space of LLMs through introducing retrieved
text chunks (Guu et al., 2020; Lewis et al., 2020),
RAG significantly improves the performance of
knowledge-intensive tasks (Ram et al., 2023). Text
chunking allows information to be more concen-
trated, minimizing the interference of irrelevant
information, enabling LLMs to focus more on the
specific content of each text chunk and generate
more precise responses (Yu et al., 2023; Besta et al.,
2024; Su et al., 2024). LumberChunker (Duarte
et al., 2024) iteratively harnesses LLMs to identify
potential segmentation points within a continuous
sequence of textual content, showing some poten-
tial for LLMs chunking. However, this method
demands a profound capability of LLMs to follow
instructions and entails substantial consumption
when employing the Gemini model.
3 Methodology
3.1 Deep Reflection on Chunking Strategies
As pointed out by Qu et al. (2024), semantic chunk-
ing has not shown a significant advantage in many
experiments. This paper further explores this phe-
nomenon and proposes two key metrics, "Boundary
Clarity" and "Chunk Stickiness", to scientifically
explain the limitations of semantic chunking and
the effectiveness of LLM chunking. At the same
time, it also provides independent evaluation indi-
cators for the rationality of chunking itself.
3.1.1 Boundary Clarity (BC)
Boundary clarity refers to the effectiveness of
chunks in separating semantic units. Specifically, it
focuses on whether the structure formed by chunk-
ing can create clear boundaries between text units
at the semantic level. Blurred chunk boundaries
may lead to a decrease in the accuracy of subse-
quent tasks. Specifically, boundary clarity is calcu-lated utilizing the following formula:
BC(q, d) =ppl(q|d)
ppl(q)(1)
where ppl(q)represents the perplexity of sentence
sequence q, and ppl(q|d)denotes the contrastive
perplexity given the context d. Perplexity serves as
a critical metric for evaluating the predictive accu-
racy of language models (LMs) on specific textual
inputs, where lower perplexity values reveal su-
perior model comprehension of the text, whereas
higher values reflect greater uncertainty in seman-
tic interpretation. When the semantic relationship
between two text chunks is independent, ppl(q|d)
tends to be closer to ppl(q), resulting in the BC
metric approaching 1. Conversely, strong seman-
tic interdependence drives the BC metric toward
zero. Therefore, higher boundary clarity implies
that chunks can be effectively separated, whereas
a lower boundary clarity indicates blurred bound-
aries between chunks, which may potentially lead
to information confusion and comprehension diffi-
culties.
3.1.2 Chunk Stickiness (CS)
The objective of text chunking lies in achieving
adaptive partitioning of documents to generate log-
ically coherent independent chunks, ensuring that
each segmented chunk encapsulates a complete and
self-contained expression of ideas while prevent-
ing logical discontinuity during the segmentation
process. Chunk stickiness specifically focuses on
evaluating the tightness and sequential integrity of
semantic relationships between text chunks. This
is achieved by constructing a semantic association
graph among text chunks, where structural entropy
is introduced to quantify the network complexity.
Within this graph, nodes represent individual text
chunks, and edge weights are defined as follows:
Edge(q, d) =ppl(q)−ppl(q|d)
ppl(q)(2)
where the theoretical range of the Edge value is
defined as [0,1]. Specifically, we initially compute
the Edge value between any two text chunks within
a long document. Values approaching 1indicate
that ppl(q|d)tends towards 0, signifying a high
degree of inter-segment correlation. Conversely, an
Edge value approaching 0suggests that ppl(q|d)
converges to ppl(q), implying that text chunks are
mutually independent. We establish a threshold
parameter K∈(0,1)to retain edges exceeding

Figure 1: Overview of the entire process of granularity-aware MoC: Dataset construction, training of router and
meta-chunkers, as well as chunking inference.
this value. Subsequently, the chunk stickiness is
specifically calculated as:
CS(G) =−nX
i=1di
2m·log2di
2m(3)
where Gis the constructed semantic graph, direp-
resents the degree of node i, andmdenotes the total
number of edges. This methodology constructs a
complete graph, followed by redundancy reduction
based on the inter-segment relationships.
On the other hand, to enhance computational
efficiency, we construct a sequence-aware incom-
plete graph that preserves the original ordering of
text chunks, which constitutes a graph construc-
tion strategy governed by sequential positional
constraints. Specifically, given a long text parti-
tioned into an ordered sequence of text chunks
D={d1, d2, ..., d n}, each node in the graph corre-
sponds to a text chunk, while edge formation is sub-
ject to dual criteria: (1) Relevance Criterion: Edge
weight Edge(di, dj)> K , where Kdenotes a pre-
defined threshold; (2) Sequential Constraint: Con-
nections are permitted exclusively when j−i > δ ,
withδrepresenting the sliding window radius fixed
at 0. This dual-constraint mechanism strategi-
cally incorporates positional relationships, thereby
achieving a better equilibrium between semantic
relevance and textual coherence.The detailed design philosophy is elaborated in
Appendix A.2. To more intuitively demonstrate
the effectiveness of the two metrics, we construct a
“Dissimilarity” metric based on the current main-
stream semantic similarity, as detailed in Section
4.5. Stemming from the above analysis, we intro-
duce a LM-based training and reasoning framework
for text chunking, named granularity-aware MoC.
3.2 Granularity-Aware MoC
In response to the complex and variable granu-
larity of large-scale text chunking in real-world
scenarios, this paper proposes a multi-granularity
chunking framework based on MoC. Our approach,
whose overall architecture is illustrated in Figure
1, dynamically routes different granularity experts
through a scheduling mechanism and optimizes
the integrity of results with a post-processing algo-
rithm.
3.2.1 Dataset Construction
We instruct GPT-4o to generate text chunks from
raw long-form texts according to the following cri-
teria: (1) Segmentation: The given text should be
segmented according to its logical and semantic
structure, such that each resulting chunk maintains
a complete and independent logical expression. (2)
Fidelity: The segmentation outcome must remain

faithful to the original text, preserving its vocabu-
lary and content without introducing any fictitious
elements. However, extracting such data from GPT-
4o poses significant challenges, as the LLM does
not always follow instructions, particularly when
dealing with long texts that contain numerous spe-
cial characters. In preliminary experiments, we
also observed that GPT-4o tends to alter the ex-
pressions used in the original text and, at times,
generates fabricated content.
To address these challenges, we propose the fol-
lowing dataset distillation procedure. We enhance
chunking precision in GPT-4o through structured
instructions that enforce adherence to predefined
rules. A sliding window algorithm, coupled with a
chunk buffering mechanism, mitigates the impact
of input text length on performance, ensuring seam-
less transitions between text subsequences. Fur-
thermore, a rigorous data cleaning process, lever-
aging edit distance calculations and manual review,
addresses potential hallucination, while strategic
anchor point extraction and placeholder insertion
facilitate efficient processing. Detailed implemen-
tation and technical specifics are provided in Ap-
pendix A.3.
3.2.2 Multi-granularity-aware Router
After the dataset construction is completed, the
MoC architecture achieves efficient text processing
through the training of the routing decision mod-
ule and meta-chunkers. The router dynamically
evaluates the compatibility of each chunk granu-
larity level based on document features, thereby
activating the optimal chunk expert. A major chal-
lenge in training the routing module lies in the im-
plicit relationship between text features and chunk
granularity, where the goal is to infer the potential
granularity of the text without performing explicit
chunking operations.
In view of this, we propose a specialized fine-
tuning method for SLMs. Firstly, we truncate or
concatenate long and short texts respectively, en-
suring their lengths hover around 1024 characters.
Both operations are performed on text chunks as
the operational unit, preserving the semantic in-
tegrity of the training texts. By maintaining ap-
proximate text lengths, SLMs can better focus on
learning features that affect chunk granularity, thus
minimizing the impact of text length on route per-
formance. Subsequently, leveraging the segmented
data generated by GPT-4o, we assign granularity
labels ranging from 0 to 3 to the text, correspond-ing to average chunk length intervals such as (0,
120], (120, 150], (150, 180], and (180,+∞). The
loss function is formulated as:
L(θ) =−1
NNX
i=1yilog(p(yi|Xi;θ)) (4)
where θrepresents the set of trainable parameters
of the SLM, yidenotes the ground-truth granular-
ity label for the i-th sample, Nsignifies the total
number of samples, and p(yi|Xi;θ)represents the
probability of assigning granularity label yi, given
input Xiand current parameters θ.
During inference, we implement marginal sam-
pling over the probability distribution of the final
token generated by the SLM in its contextual se-
quence, selecting the granularity category with the
highest probability from the four available cate-
gories as the granularity for the corresponding text.
Afterwards, the text to be chunked is routed to the
corresponding chunking expert:
R(Xi) = arg max
kp(k|Xi;θ) (5)
where krepresents the category of chunking granu-
larity. Through this mechanism, the router enables
dynamic expert selection without explicit chunking
operations.
3.2.3 Meta-chunkers
Our objective is not to require meta-chunkers to
generate each text chunk in its entirety, but rather to
guide it in producing a structured list of segmented
regular expressions. Each element in this list con-
tains only the start Sand end Eof a text chunk C,
with a special character rreplacing the intervening
content. The regular expression is represented as:
Cregex=S⊕r⊕E, r ∈ R (6)
where ⊕denotes the string concatenation opera-
tion,R={“< omitted > ”,“< ellipsis >
”,“[MASK ]”,“[ELLIPSIS ]”,“.∗?”,“< ... >
”,“< .∗>”,“< pad > ”}is the set of eight
special characters we have defined to represent the
omitted parts in a text chunk. During the expert
training phase, we employ a full fine-tuning strat-
egy, utilizing datasets categorized by different seg-
mentation granularities to optimize the model pa-
rameters. The loss function remains consistent with
Equation 4. This design allows Meta-chunkers to
comprehensively understand the composition of
each chunk while significantly reducing the time
cost of generation.

3.2.4 Edit Distance Recovery Algorithm
Let string Adenote an element generated by a meta-
chunker and string Brepresent a segment within
the original text. The edit distance refers to the
minimum number of operations required to trans-
form AintoB, where the permissible operations
include the insertion, deletion, or substitution of a
single character. We then define a two-dimensional
array, ab[i][j], which represents the minimum num-
ber of operations needed to convert the substring
A[1. . . i]intoB[1. . . j]. By recursively deriving
the state transition formula, we can incrementally
construct this array.
Initially, the conditions are as follows: (1) When
i= 0,Ais an empty string, necessitating the inser-
tion of jcharacters to match B, thus ab[0][j] =j;
(2) When j= 0,Bis an empty string, requiring
the deletion of icharacters, hence ab[i][0] = i; (3)
When i=j= 0, the edit distance between two
empty strings is evidently ab[0][0] = 0 . Subse-
quently, the entire abarray is populated using the
following state transition formula:
ab[i][j] =

ab[i−1][j−1], ifA[i] =B[j]
1 + min( ab[i−1][j],
ab[i][j−1],
ab[i−1][j−1]),ifA[i]̸=B[j]
If the current characters are identical, no additional
operation is required, and the problem reduces to
a subproblem; if the characters differ, the opera-
tion with the minimal cost among insertion, dele-
tion, or substitution is selected. Ultimately, by
utilizing the minimum edit distance, we can ac-
curately pinpoint the field in the original text that
most closely matches the elements generated by
the meta-chunker, thereby ensuring the precision
of regular extraction.
4 Experiment
4.1 Datasets and Metrics
We conduct a comprehensive evaluation on three
datasets, and covering multiple metrics. The
CRUD benchmark (Lyu et al., 2024) contains
single-hop and two-hop questions, evaluated us-
ing metrics including BLEU series and ROUGE-L.
We utilize the DuReader dataset from LongBench
benchmark (Bai et al., 2023), evaluated based on
F1 metric. In addition, a dataset called WebCPM
(Qin et al., 2023) specifically designed for long-
text QA, is utilized to retrieve relevant facts andgenerate detailed paragraph-style responses, with
ROUGE-L as the metric.
4.2 Baselines
We primarily compare meta-chunker and MoC with
two types of baselines, namely rule-based chunking
and dynamic chunking, noting that the latter incor-
porates both semantic similarity models and LLMs.
The original rule-based method simply divides
long texts into fixed-length chunks, disregarding
sentence boundaries. However, the Llama_index
method (Langchain, 2023) offers a more nuanced
approach, balancing the maintenance of sentence
boundaries while ensuring that token counts in each
segment are close to a preset threshold. On the
other hand, semantic chunking (Xiao et al., 2023)
utilizes sentence embedding models to segment
text based on semantic similarity. LumberChun-
ker (Duarte et al., 2024) employs LLMs to predict
optimal segmentation points within the text.
4.3 Experimental Settings
Without additional annotations, all LMs used in
this paper adopt chat or instruction versions. When
chunking, we primarily employ LMs with the fol-
lowing hyperparameter settings: temperature at 0.1
and top-p at 0.1. For evaluation, Qwen2-7B is
applied with the following settings: top_p = 0.9,
top_k = 5, temperature = 0.1, and max_new_tokens
= 1280. When conducting QA, the system neces-
sitates dense retrievals from the vector database,
with top_k set to 8 for CRUD, 5 for DuReader
and WebCPM. To control variables, we maintain a
consistent average chunk length of 178 for various
chunking methods across each dataset. Detailed
experimental setup information can be found in
Appendix A.1.
4.4 Main Results
To comprehensively validate the effectiveness of
the proposed meta-chunker and MoC architectures,
we conducts experiments using three widely used
QA datasets. During dataset preparation, we cu-
rate 20,000 chunked QA pairs through rigorous
processing. Initially, we fine-tune the Qwen2.5-
1.5B model using this data. As shown in Table 1,
compared to traditional rule-based and semantic
chunking methods, as well as the state-of-the-art
LumberChunker approach based on Qwen2.5-14B,
the Meta-chunker-1.5B exhibits both improved and
more stable performance. Furthermore, we directly
perform chunking employing Qwen2.5-14B and

Chunking MethodsCRUD (Single-hop) CRUD (Two-hop) DuReader WebCPM
BLEU-1 BLEU-Avg ROUGE-L BLEU-1 BLEU-Avg ROUGE-L F1 ROUGE-L
Original 0.3515 0.2548 0.4213 0.2322 0.1133 0.2613 0.2030 0.2642
Llama_index 0.3620 0.2682 0.4326 0.2315 0.1133 0.2585 0.2220 0.2630
Semantic Chunking 0.3382 0.2462 0.4131 0.2223 0.1075 0.2507 0.2157 0.2691
LumberChunker 0.3456 0.2542 0.4160 0.2204 0.1083 0.2521 0.2178 0.2730
Qwen2.5-14B 0.3650 0.2679 0.4351 0.2304 0.1129 0.2587 0.2271 0.2691
Qwen2.5-72B 0.3721 0.2743 0.4405 0.2382 0.1185 0.2677 0.2284 0.2693
Meta-chunker-1.5B 0.3754 0.2760 0.4445 0.2354 0.1155 0.2641 0.2387 0.2745
Table 1: Main experimental results are presented in four QA datasets. The best result is in bold, and the second best
result is underlined.
Methods BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE-L
<pad> 0.3683 0.2953 0.2490 0.2132 0.4391
<omitted> 0.3725 0.2985 0.2523 0.2165 0.4401
<ellipsis> 0.3761 0.3025 0.2554 0.2193 0.4452
[MASK] 0.3754 0.3012 0.2545 0.2188 0.4445
[ELLIPSIS] 0.3699 0.2966 0.2510 0.2159 0.4380
.*? 0.3745 0.3015 0.2553 0.2195 0.4437
<...> 0.3716 0.2988 0.2526 0.2167 0.4412
<.*> 0.3790 0.3054 0.2583 0.2221 0.4470
MoC 0.3826 0.3077 0.2602 0.2234 0.4510
Table 2: Performance impact of special characters and
the effectiveness of granularity-aware MoC framework
in text chunking.
Qwen2.5-72B. The results demonstrate that these
LLMs, with their powerful context processing and
reasoning abilities, also deliver outstanding perfor-
mance in chunking tasks. However, Meta-chunker-
1.5B slightly underperforms the 72B model only
in the two-hop CRUD, while outperforming both
LLMs in other scenarios.
Upon validating the effectiveness of our pro-
posed chunking experts, we proceeded to inves-
tigate the impact of various special characters on
performance, and extended chunking within the
MoC framework. As illustrated in Table 2, we
design eight distinct special characters, each in-
ducing varying degrees of performance fluctuation
in the meta-chunker. Notably, all character con-
figurations demonstrate measurable performance
enhancements compared to baseline approaches,
with [Mask ]and< .∗>exhibiting particularly
remarkable efficacy. In our experiments, both the
Meta-chunker-1.5B and the MoC framework em-
ploy [Mask ]as an ellipsis to replace the middlesections of text chunks, while maintaining consis-
tent training data. The experimental results indi-
cate that the chunking method based on the MoC
architecture further enhances performance. Specif-
ically, when handling complex long texts, MoC
effectively differentiates the chunking granularity
of various sections. Moreover, the time complexity
of the MoC remains at the level of a single SLM,
showcasing a commendable balance between com-
putational efficiency and performance.
4.5 Exploring Chunking Based on Boundary
Clarity and Chunk Stickiness
To compare the effectiveness of the two metrics
we designed, we introduce the "Dissimilarity" (DS)
metric:
DS= 1−sim(q, d)
where sim(q, d)represents the semantic similarity
score between the text chunks qandd. With this
definition, the DS metric ranges from [0, 1], where
0 indicates perfect similarity and 1 indicates com-
plete dissimilarity. The design of the DS metric is
based on the following considerations: first, seman-
tic similarity measures are typically employed to
assess the degree of semantic proximity between
two text segments. By converting this to the dissim-
ilarity measure, we can more directly observe the
semantic differences between chunks. Second, the
linear transformation of DS preserves the mono-
tonicity of the original similarity measure without
losing any information. Figure 4(a) reveals the
QA performance of RAG using different chunking
strategies. It is important to note that, to ensure
the validity of the evaluation, we maintained the

Chunking MethodsQwen2.5-1.5B Qwen2.5-7B Qwen2.5-14B Internlm3-8B
BC CS c CSi BC CS c CSi BC CS c CSi BC CS c CSi
Original 0.8210 2.397 1.800 0.8049 2.421 1.898 0.7704 2.297 1.459 0.8054 2.409 1.940
Llama_index 0.8590 2.185 1.379 0.8455 2.250 1.483 0.8117 2.081 1.088 0.8334 2.107 1.303
Semantic Chunking 0.8260 2.280 1.552 0.8140 2.325 1.650 0.7751 2.207 1.314 0.8027 2.255 1.546
Qwen2.5-14B 0.8750 2.069 1.340 0.8641 2.125 1.438 0.8302 1.927 1.068 0.8444 1.889 1.181
Table 3: Performance of different chunking methods under various LMs, directly calculated using two metrics we
proposed: BC represents boundary clarity, which is preferable when higher; CScdenotes chunk stickiness utilizing
a complete graph, and CSiindicates chunk stickiness employing a incomplete graph, both of which are favorable
when lower.
same average text chunk length across all chunking
methods.
Why Does Semantic Chunking Underper-
form? As illustrated in Figure 4(b), while se-
mantic chunking scores are generally high, its per-
formance in QA tasks is suboptimal. Moreover,
there is no evident correlation between the scores
of semantic dissimilarity and the efficacy of QA.
This suggests that in the context of RAG, relying
solely on semantic similarity between sentences is
insufficient for accurately delineating the optimal
boundaries of text chunks.
Furthermore, it can be observed from Table 3
that the clarity of semantic chunking boundaries
is only marginally superior to fixed-length chunk-
ing. This implies that although semantic chunking
attempts to account for the degree of association
between sentences, its limited ability to distinguish
logically connected sentences often results in incor-
rect segmentation of content that should remain co-
herent. Additionally, Table 3 reveals that semantic
chunking also falls short in terms of capturing se-
mantic relationships, leading to higher chunk stick-
iness and consequently affecting the independence
of text chunks.
Why Does LLM-Based Chunking Work? As
shown in Table 3, the text chunks generated by
LLMs exhibit superior boundary clarity, indicating
the heightened ability to accurately identify seman-
tic shifts and topic transitions, thereby mitigating
the erroneous segmentation of related sentences.
Concurrently, the LLM-based chunking produces
text chunks with reduced chunk stickiness, signify-
ing that the internal semantics of chunks are more
tightly bound, while a greater degree of indepen-
dence is maintained between chunks. This combi-
nation of well-defined boundaries and diminished
stickiness contributes to enhanced retrieval effi-ciency and generation quality within RAG systems,
ultimately leading to superior overall performance.
4.6 Hyper-parameter Sensitivity Analysis
In calculating the chunk stickiness, we rely on the
Kto filter out edges with weaker associations be-
tween text chunks in the knowledge graph. As
presented in Table 4, an increase in the value of
Kleads to a gradual decrease in the metric. This
occurs because a larger Kvalue limits the number
of retained edges, resulting in a sparser connectiv-
ity structure within the graph. Notably, regardless
of the chosen Kvalue, the LLM-based chunking
method consistently maintains a low level of chunk
stickiness. This indicates that it more accurately
identifies semantic transition points between sen-
tences, effectively avoiding excessive cohesion be-
tween text chunks caused by interruptions within
paragraphs.
MethodsComplete Graph Incomplete Graph
0.7 0.8 0.9 0.7 0.8 0.9
Original 2.536 2.397 2.035 2.199 1.800 1.300
Llama_index 2.454 2.185 1.543 1.997 1.379 0.740
Semantic Chunking 2.455 2.280 1.733 2.039 1.552 0.835
Qwen2.5-14B 2.364 2.069 1.381 1.972 1.340 0.623
Table 4: Performance sensitivity of Kin chunk sticki-
ness.
We conduct experiments on the decoding sam-
pling hyperparameters of the meta-chunker within
the MoC framework, with specific results presented
in Table 2. Experimental data demonstrates that
higher values of temperature and top-k sampling
strategies introduce increased randomness, thereby
exerting a certain impact on the chunking effect.
Conversely, when these two hyperparameters are

set to lower values, the model typically provides
more stable and precise chunking, leading to a
more significant performance improvement.
Figure 2: Performance sensitivity to temperature and
top-k.
5 Conclusion
Addressing the current void in the independent as-
sessment of chunking quality, this paper introduces
two novel evaluation metrics: boundary clarity and
chunk stickiness. It systematically elucidates the
inherent limitations of semantic chunking in long-
text processing, which further leads to the neces-
sity of LLM-based chunking. Amidst the drive for
performance and efficiency optimization, we pro-
pose the MoC framework, which utilizes sparsely
activated meta-chunkers through multi-granularity-
aware router. It’s worth emphasizing that this study
guides meta-chunkers to generate a highly struc-
tured list of chunking regular expressions, precisely
extracting text chunks from the original text using
only a few characters from the beginning and end.
Our approach demonstrates superior performance
compared to strong baselines.
6 Limitations
Despite the superior performance demonstrated by
the proposed MoC framework for chunking tasks
on various datasets, there are still some limitations
that merit further exploration and improvement.
Although we have implemented multiple quality
control measures to ensure data quality and con-
structed a training set consisting of nearly 20,000
data entries, the current dataset size remains rela-
tively limited compared to the massive scale andcomplex diversity of real-world text data. We have
mobilized the power of the open-source community
to further enrich our chunking dataset utilizing pre-
training data from LMs. Additionally, while the
dataset construction process is flexible and theoret-
ically expandable to more scenarios, it has not yet
undergone adequate multi-language adaptation and
validation. We leave this aspect for future research.
References
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, et al. 2023. Longbench:
A bilingual, multitask benchmark for long context
understanding. arXiv preprint arXiv:2308.14508 .
Bhagyashree Vyankatrao Barde and Anant Madhavrao
Bainwad. 2017. An overview of topic modeling
methods and tools. In 2017 International Confer-
ence on Intelligent Computing and Control Systems
(ICICCS) , pages 745–750. IEEE.
Garbiel Bénédict, Ruqing Zhang, and Donald Metzler.
2023. Gen-ir@ sigir 2023: The first workshop on
generative information retrieval. In Proceedings of
the 46th International ACM SIGIR Conference on
Research and Development in Information Retrieval ,
pages 3460–3463.
Maciej Besta, Ales Kubicek, Roman Niggli, Robert
Gerstenberger, Lucas Weitzendorf, Mingyuan Chi,
Patrick Iff, Joanna Gajda, Piotr Nyczyk, Jürgen
Müller, et al. 2024. Multi-head rag: Solving
multi-aspect problems with llms. arXiv preprint
arXiv:2406.05085 .
Yuyan Chen, Qiang Fu, Yichen Yuan, Zhihao Wen,
Ge Fan, Dayiheng Liu, Dongmei Zhang, Zhixu Li,
and Yanghua Xiao. 2023. Hallucination detection:
Robustly discerning reliable answers in large lan-
guage models. In Proceedings of the 32nd ACM
International Conference on Information and Knowl-
edge Management , pages 245–255.
Sangwoo Cho, Kaiqiang Song, Xiaoyang Wang, Fei
Liu, and Dong Yu. 2022. Toward unifying text seg-
mentation and long document summarization. arXiv
preprint arXiv:2210.16422 .
André V Duarte, João Marques, Miguel Graça, Miguel
Freire, Lei Li, and Arlindo L Oliveira. 2024. Lum-
berchunker: Long-form narrative document segmen-
tation. arXiv preprint arXiv:2406.17526 .
Zhicheng Guo, Sijie Cheng, Yile Wang, Peng Li, and
Yang Liu. 2023. Prompt-guided retrieval augmen-
tation for non-knowledge-intensive tasks. arXiv
preprint arXiv:2305.17653 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In International confer-
ence on machine learning , pages 3929–3938. PMLR.

Hangfeng He, Hongming Zhang, and Dan Roth. 2022.
Rethinking with retrieval: Faithful large language
model inference. arXiv preprint arXiv:2301.00303 .
Greg Kamradt. 2024. Semantic chunk-
ing. https://github.com/FullStackRetrieval-
com/RetrievalTutorials .
P Kherwa and P Bansal. 2020. Topic modeling: A
comprehensive review. eai endorsed transactions on
scalable information systems, 7 (24), 1–16.
Youna Kim, Hyuhng Joon Kim, Cheonbok Park,
Choonghyun Park, Hyunsoo Cho, Junyeob Kim,
Kang Min Yoo, Sang-goo Lee, and Taeuk Kim.
2024. Adaptive contrastive decoding in retrieval-
augmented generation for handling noisy contexts.
arXiv preprint arXiv:2408.01084 .
Langchain. 2023. https://github.com/langchain-
ai/langchain .
Angeliki Lazaridou, Elena Gribovskaya, Wojciech
Stokowiec, and Nikolai Grigorev. 2022. Internet-
augmented language models through few-shot
prompting for open-domain question answering.
arXiv preprint arXiv:2203.05115 .
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and
Lemao Liu. 2022. A survey on retrieval-augmented
text generation. arXiv preprint arXiv:2202.01110 .
Jing Li, Billy Chiu, Shuo Shang, and Ling Shao. 2020.
Neural text segmentation and its application to sen-
timent analysis. IEEE Transactions on Knowledge
and Data Engineering , 34(2):828–842.
Xianzhi Li, Samuel Chan, Xiaodan Zhu, Yulong Pei,
Zhiqiang Ma, Xiaomo Liu, and Sameena Shah. 2023.
Are chatgpt and gpt-4 general-purpose solvers for
financial text analytics? a study on several typical
tasks. arXiv preprint arXiv:2305.05862 .
Xun Liang, Shichao Song, Zifan Zheng, Hanyu Wang,
Qingchen Yu, Xunkai Li, Rong-Hua Li, Feiyu Xiong,
and Zhiyu Li. 2024. Internal consistency and self-
feedback in large language models: A survey. arXiv
preprint arXiv:2407.14507 .
Weizhe Lin, Rexhina Blloshmi, Bill Byrne, Adrià
de Gispert, and Gonzalo Iglesias. 2023. Li-rage:
Late interaction retrieval augmented generation with
explicit signals for open-domain table question an-
swering. In Proceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 2: Short Papers) , pages 1557–1566.S Longpre, G Yauney, E Reif, K Lee, A Roberts, B Zoph,
D Zhou, J Wei, K Robinson, D Mimno, et al. A pre-
trainer’s guide to training data: Measuring the effects
of data age, domain coverage, quality, & toxicity,
may 2023. URL http://arxiv. org/abs/2305.13169 .
Michal Lukasik, Boris Dadachev, Gonçalo Simoes, and
Kishore Papineni. 2020. Text segmentation by cross
segment attention. arXiv preprint arXiv:2004.14535 .
Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong,
Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu,
Tong Xu, and Enhong Chen. 2024. Crud-rag:
A comprehensive chinese benchmark for retrieval-
augmented generation of large language models.
arXiv preprint arXiv:2401.17043 .
Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao
Liang, Kunlun Zhu, Yankai Lin, Xu Han, Ning Ding,
Huadong Wang, et al. 2023. Webcpm: Interactive
web search for chinese long-form question answering.
arXiv preprint arXiv:2305.06849 .
Renyi Qu, Ruixuan Tu, and Forrest Bao. 2024. Is seman-
tic chunking worth the computational cost? arXiv
preprint arXiv:2410.13070 .
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunx-
iang Wang, Shichao Sun, Huanyu Li, et al. 2024.
Ragchecker: A fine-grained framework for diagnos-
ing retrieval-augmented generation. arXiv preprint
arXiv:2408.08067 .
Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang
Zhang. 2023. In chatgpt we trust? measuring
and characterizing the reliability of chatgpt. arXiv
preprint arXiv:2304.08979 .
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context. In Inter-
national Conference on Machine Learning , pages
31210–31227. PMLR.
Georgios Sidiropoulos and Evangelos Kanoulas. 2022.
Analysing the robustness of dual encoders for dense
retrieval against misspellings. In Proceedings of the
45th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval ,
pages 2132–2136.
Devendra Singh, Siva Reddy, Will Hamilton, Chris
Dyer, and Dani Yogatama. 2021. End-to-end train-
ing of multi-document reader and retriever for open-
domain question answering. Advances in Neural
Information Processing Systems , 34:25968–25981.

Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. Dragin: Dynamic retrieval aug-
mented generation based on the real-time informa-
tion needs of large language models. arXiv preprint
arXiv:2403.10081 .
Chao-Hong Tan, Jia-Chen Gu, Chongyang Tao, Zhen-
Hua Ling, Can Xu, Huang Hu, Xiubo Geng, and
Daxin Jiang. 2022. Tegtok: Augmenting text gen-
eration via task-specific and open-world knowledge.
arXiv preprint arXiv:2203.08517 .
Shitao Xiao, Zheng Liu, Peitian Zhang, and N Muen-
nighof. 2023. C-pack: packaged resources to ad-
vance general chinese embedding. 2023. arXiv
preprint arXiv:2309.07597 .
Shicheng Xu, Liang Pang, Huawei Shen, and Xueqi
Cheng. 2023. Berm: Training the balanced and ex-
tractable representation for matching to improve gen-
eralization ability of dense retrieval. arXiv preprint
arXiv:2305.11052 .
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
arXiv preprint arXiv:2401.15884 .
Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin
Ma, Hongwei Wang, and Dong Yu. 2023. Chain-of-
note: Enhancing robustness in retrieval-augmented
language models. arXiv preprint arXiv:2311.09210 .
Qinglin Zhang, Qian Chen, Yali Li, Jiaqing Liu, and
Wen Wang. 2021. Sequence model with self-adaptive
sliding window for efficient spoken document seg-
mentation. In 2021 IEEE Automatic Speech Recog-
nition and Understanding Workshop (ASRU) , pages
411–418. IEEE.
Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai
Yang, Jia Liu, Shujian Huang, Qingwei Lin, Saravan
Rajmohan, Dongmei Zhang, and Qi Zhang. 2024. Ef-
ficientrag: Efficient retriever for multi-hop question
answering. arXiv preprint arXiv:2408.04259 .
Guido Zuccon, Bevan Koopman, and Razia Shaik. 2023.
Chatgpt hallucinates when attributing answers. In
Proceedings of the Annual International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval in the Asia Pacific Region , pages
46–51.
A Appendix
A.1 Main Experimental Details
All language models utilized in this paper em-
ploy the chat or instruct versions where multiple
versions exist, and are loaded in full precision
(Float32). The vector database is constructed using
Milvus, where the embedding model is bge-large-
zh-v1.5. In experiments, we utilize a total of four
benchmarks, and their specific configurations are
detailed as follows:(a)Rule-based Chunking Methods
•Original : This method divides long texts
into segments of a fixed length, such as
two hundred Chinese characters or words,
without considering sentence boundaries.
•Llama_index (Langchain, 2023): This
method considers both sentence com-
pleteness and token counts during seg-
mentation. It prioritizes maintain-
ing sentence boundaries while ensur-
ing that the number of tokens in each
chunk are close to a preset threshold.
We use the SimpleNodeParser func-
tion from Llama_index , adjusting the
chunk_size parameter to control seg-
ment length. Overlaps are handled by dy-
namically overlapping segments using the
chunk_overlap parameter, ensuring sen-
tence completeness during segmentation
and overlapping.
(b)Dynamic Chunking Methods
•Semantic Chunking (Xiao et al., 2023):
Utilizes pre-trained sentence embedding
models to calculate the cosine similarity
between sentences. By setting a simi-
larity threshold, sentences with lower
similarity are selected as segmentation
points, ensuring that sentences within
each chunk are highly semantically
related. This method employs the
SemanticSplitterNodeParser from
Llama_index , exploiting the bge-base-
zh-v1.5 model. The size of the text
chunks is controlled by adjusting the
similarity threshold.
•LumberChunker (Duarte et al., 2024):
Leverages the reasoning capabilities of
LLMs to predict suitable segmentation
points within the text. We utilize Qwen2.5
models with 14B parameters, set to full
precision.
A.2 Design Philosophy of Chunk Stickiness
In the context of network architecture, high struc-
tural entropy tends to exhibit greater challenges in
predictability and controllability due to its inherent
randomness and complexity. Our chunking strategy
aims to maximize semantic independence between
text chunks while maintaining a coherent semantic

expression. Consequently, a higher chunk stick-
iness implies greater interconnectedness among
these chunks, resulting in a more intricate and less
ordered semantic network. Furthermore, to ensure
a robust comparison between different chunking
methods, we enforce a uniform average chunking
length. This standardization provides a fair basis
for evaluation, mitigating potential biases arising
from discrepancies in chunking size. Ultimately, a
lower CS score signifies that the chunking method
is more accurate in identifying semantic transi-
tion points between sentences, thereby avoiding
the fragmentation of coherent passages and the
consequent excessive stickiness between resulting
chunks.
To more intuitively demonstrate the effective-
ness of the two metrics we designed, we construct
a “Dissimilarity” metric based on the current main-
stream semantic similarity, as detailed in Section
4.5. Furthermore, employing several chunking tech-
niques and LLMs, we conduct an in-depth inves-
tigation of boundary clarity and chunk stickiness,
conducting comparative experiments with the dis-
similarity metric. The experimental results clearly
show that the two proposed metrics exhibit a consis-
tent trend with RAG performance when evaluating
the quality of text chunking. In contrast, the dis-
similarity metric fail to display a similar variation.
This suggests that, even without relying on QA ac-
curacy, the two proposed metrics can independently
and effectively assess chunking quality.
A.3 Dataset Construction Process
Structured Instruction Design By explicitly
enumerating rules, GPT-4o is compelled to adhere
to predefined chunking regulations, such as ensur-
ing semantic unit integrity, enforcing punctuation
boundaries, and prohibiting content rewriting.
Sliding Window and Chunk Buffering Mech-
anism Drawing from the research conducted by
Duarte et al. (2024) and practical experience, we
observe that the length of the original text signif-
icantly influences the chunking performance of
LLMs. To address this problem, we initially apply
a sliding window algorithm to segment the input
text into subsequences, each below a threshold of
1024 tokens. Segmentation points are prioritized at
paragraph boundaries or sentence-ending positions.
These subsequences are then processed sequen-
tially by GPT-4o. To maintain continuity between
two consecutive subsequences, we implement a
chunk buffer mechanism by removing the last gen-erated text chunk of the preceding sequence and
using it as the prefix for the subsequent sequence,
thereby ensuring smooth information flow.
Data Cleaning and Annotation To identify
and eliminate hallucinated content during the gener-
ation process, we calculate the difference between
each chunk and the paragraphs in the original text
through the edit distance, as outlined in Section
3.2.4. If the minimum edit distance exceeds 10%
of the chunk length, we manually review the lo-
cation of the chunk error and make corrections
accordingly. Additionally, for a long text, we ex-
tract several characters at the beginning and end of
each text chunk as anchor points, while replacing
the intermediate content with eight preset special
placeholders, as demonstrated in Sections 3.2.2 and
3.2.3.
A.4 Another Perspective on Chunking
Performance Comparison
The performance evaluation of RAG systems pri-
marily focuses on the similarity between gener-
ated answers and reference answers. However, this
evaluation method introduces additional noise dur-
ing the decoding strategy in the generation stage,
making it difficult to distinguish whether the perfor-
mance defects originate from the retrieved chunk or
the generation module. To address this constraint,
we propose an evaluation approach based on infor-
mation support, which centers on quantifying the
supporting capability of retrieved text chunks for
the target answer through conditional probability
modeling.
Given a set of retrieved chunks C =
{c1, c2, ..., c n}and the reference answer A=
{a1, a2, ..., a m}, we employ a LLM to compute the
average conditional probability (CP) of the target
answer:
CP=−1
MMX
i=1logP(ai|c1, c2, . . . , c n) (7)
A smaller CP value indicates a higher likelihood
of the correct answer being inferred from the re-
trieved text chunks, signifying stronger support.
The results presented in Table 5 show that, even
when evaluated with different LMs, our chunking
method consistently exhibits high support. This
suggests that our chunking strategy, by optimiz-
ing the semantic integrity and independence of text
chunks, enhances the relevance of the retrieved text
to the question, thereby reducing the difficulty of
generating the correct answer.

Methods Qwen-1.5B Qwen-7B Qwen-14B Internlm-8B
Original 2.206 2.650 2.560 1.636
Llama_index 1.964 2.412 2.353 1.486
Semantic Chunking 1.865 2.331 2.238 1.411
LumberChunker 2.184 2.593 2.589 1.652
Qwen2.5-14B 1.841 2.313 2.209 1.373
Meta-chunker-1.5B 1.835 2.267 2.199 1.367
Table 5: Information-based performance evaluation for
the RAG system.
A.5 Prompt utilized in Chunking
When preparing datasets using GPT-4o and gener-
ating chunking rules with MoC, prompts are nec-
essary, as illustrated in Tables 6 and 7. The design
and implementation of these prompts are crucial,
as they directly influence the quality and character-
istics of the resulting datasets and chunking rules.
A.6 Details of MoC Training
During the model training phase, we adopt specific
parameter configurations. Specifically, the train-
ing batch size per device is set to 3, and model
parameters are updated every 16 steps through the
gradient accumulation strategy. The learning rate
is set to 1.0e−5to achieve fine-grained adjust-
ment of weights. The model underwent a total of
3 epochs of training. Additionally, we employ a
cosine annealing learning rate scheduling strategy
and set a warmup ratio of 0.1 to facilitate model
convergence. The variations in training loss are
recorded, namely, Figure 3 showcases the training
loss of the router, while Figure 5-8 individually
depict the training losses of chunking experts in
different intervals. The bf16 format is enabled dur-
ing training to balance memory consumption and
training speed. This training is conducted on two
NVIDIA A800 80G graphics cards, ensuring effi-
cient computing capabilities.Chunking Prompt
This is a text chunking task, and you are an expert
in text segmentation, responsible for dividing the
given text into text chunks. You must adhere to
the following four conditions:
1. Segment the text based on its logical and seman-
tic structure, ensuring each text chunk expresses a
complete logical thought.
2. Avoid making the text chunks too short, bal-
ancing the recognition of content transitions with
appropriate chunk length.
3. Do not alter the original vocabulary or content
of the text.
4. Do not add any new words or symbols.
If you understand, please segment the follow-
ing text into text chunks, with each chunk sep-
arated by "\n—\n". Output the complete set of
segmented chunks without omissions.
Document content: [Text to be segmented]
The segmented text chunks are:
Table 6: Prompt for direct chunking of GPT-4o.
Figure 3: Trend of loss change during router training.

Figure 4: Trends in evaluating chunking performance using different metrics.
Figure 5: Trend of loss change during meta-chunker
training with granularity range [0,120].
Figure 6: Trend of loss change during meta-chunker
training with granularity range (120,150].
Figure 7: Trend of loss change during meta-chunker
training with granularity range (150,180].
Figure 8: Trend of loss change during meta-chunker
training with granularity range (180,+∞).

Chunking Prompt
This is a text chunking task. As an expert in text segmentation, you are responsible
for segmenting the given text into text chunks. You must adhere to the following four
conditions:
1. Combine several consecutive sentences with related content into text chunks,
ensuring that each text chunk has a complete logical expression.
2. Avoid making the text chunks too short, and strike a good balance between
recognizing content transitions and chunk length.
3. The output of the chunking result should be in a list format, where each element
represents a text chunk in the document.
4. Each text chunk in the output should consist of the first few characters of the text
chunk, followed by "[MASK]" to replace the intermediate content, and end with the
last few characters of the text chunk. The output format is as follows:
[
"First few characters of text chunk [MASK] Last few characters of text chunk",
...
]
If you understand, please segment the following text into text chunks and output
them in the required list format.
Document content: [Text to be segmented]
Table 7: Prompt for chunking of MoC.
# MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework

**Authors**: Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, Bo Zheng

**Published**: 2025-04-14 10:19:47

**PDF URL**: [http://arxiv.org/pdf/2504.10074v2](http://arxiv.org/pdf/2504.10074v2)

## Abstract
Recent advancements in large language models (LLMs) and multi-modal LLMs have
been remarkable. However, these models still rely solely on their parametric
knowledge, which limits their ability to generate up-to-date information and
increases the risk of producing erroneous content. Retrieval-Augmented
Generation (RAG) partially mitigates these challenges by incorporating external
data sources, yet the reliance on databases and retrieval systems can introduce
irrelevant or inaccurate documents, ultimately undermining both performance and
reasoning quality. In this paper, we propose Multi-Modal Knowledge-Based
Retrieval-Augmented Generation (MMKB-RAG), a novel multi-modal RAG framework
that leverages the inherent knowledge boundaries of models to dynamically
generate semantic tags for the retrieval process. This strategy enables the
joint filtering of retrieved documents, retaining only the most relevant and
accurate references. Extensive experiments on knowledge-based visual
question-answering tasks demonstrate the efficacy of our approach: on the E-VQA
dataset, our method improves performance by +4.2% on the Single-Hop subset and
+0.4% on the full dataset, while on the InfoSeek dataset, it achieves gains of
+7.8% on the Unseen-Q subset, +8.2% on the Unseen-E subset, and +8.1% on the
full dataset. These results highlight significant enhancements in both accuracy
and robustness over the current state-of-the-art MLLM and RAG frameworks.

## Full Text


<!-- PDF content starts -->

MMKB-RAG: A Multi-Modal Knowledge-Based
Retrieval-Augmented Generation Framework
Zihan Ling
Peking University
China
lingzihan@stu.pku.edu.cnZhiyao Guo
Alibaba Group
China
guozhiyao45@gmail.comYixuan Huang
Alibaba Group
China
huangyixuan@sjtu.edu.cn
Yi An
Peking University
China
anyi@stu.pku.edu.cnShuai Xiao
Alibaba Group
China
shuai.xsh@gmail.comJinsong Lan
Alibaba Group
China
jinsonglan.ljs@taobao.com
Xiaoyong Zhu
Alibaba Group
China
xiaoyong.z@taobao.comBo Zheng
Alibaba Group
China
bozheng@alibaba-inc.com
Abstract
Recent advancements in large language models (LLMs) and multi-
modal LLMs have been remarkable. However, these models still
rely solely on their parametric knowledge, which limits their ability
to generate up-to-date information and increases the risk of pro-
ducing erroneous content. Retrieval-Augmented Generation (RAG)
partially mitigates these challenges by incorporating external data
sources, yet the reliance on databases and retrieval systems can
introduce irrelevant or inaccurate documents, ultimately under-
mining both performance and reasoning quality. In this paper, we
propose Multi-Modal Knowledge-Based Retrieval-Augmented Gen-
eration (MMKB-RAG), a novel multi-modal RAG framework that
leverages the inherent knowledge boundaries of models to dynami-
cally generate semantic tags for the retrieval process. This strategy
enables the joint filtering of retrieved documents, retaining only
the most relevant and accurate references. Extensive experiments
on knowledge-based visual question-answering tasks demonstrate
the efficacy of our approach: on the E-VQA dataset, our method im-
proves performance by +4.2% on the Single-Hop subset and +0.4%
on the full dataset, while on the InfoSeek dataset, it achieves gains
of +7.8% on the Unseen-Q subset, +8.2% on the Unseen-E subset,
and +8.1% on the full dataset. These results highlight significant
enhancements in both accuracy and robustness over the current
state-of-the-art MLLM and RAG frameworks.
1 Introduction
The rapid evolution of large language models (LLMs) and multi-
modal large language models (MLLMs) has revolutionized both
natural language processing and visual reasoning tasks. Trained
on extensive datasets, these models excel at leveraging intrinsic
parametric knowledge to generate coherent and contextually ap-
propriate responses. However, a fundamental challenge remains:
The reliance on static parametric knowledge (i.e., the knowledge
acquired during pre-training) often leads to errors or hallucinations
[23], particularly when addressing complex queries that demand
precise, domain-specific, or real-time information[ 14]. For example,
RETSRTCSToptimize
inherent knowledge boundariesQuery(Img+text)correctcorrectirrelevantredundantrag. doc.Wrong Resultscorrectcorrectfiltered doc.Correct Results
MLLMFigure 1: Examples illustrate that RAG may retrieve ir-
relevant and redundant documents, resulting in incorrect
outcomes. Our approach leverages the MLLM’s knowledge
boundaries to filter references, retaining essential evidence
for accurate answers.
in knowledge-based visual question-answering (VQA) tasks such
as Encyclopedic-VQA (E-VQA) [ 21] and InfoSeek [ 5], even state-
of-the-art MLLMs struggle to generate accurate responses because
of lacking the ability to retrieve and integrate external knowledge
effectively. This deficiency underscores a critical gap in current
MLLM architectures: the need for mechanisms that enable dynamic
and reliable access to external knowledge sources.
To overcome the limitations of static parametric knowledge, re-
searchers have developed Retrieval-Augmented Generation (RAG)
frameworks that enables models to incorporate the latest and most
relevant information during inference. These frameworks can be
broadly divided into retrieval-side mechanisms and generation-side
mechanisms. On the retrieval side, existing methods primarily focus
on aligning visual and textual modalities with external knowledge
sources. CLIP-based architectures leverage contrastive image-text
encoders to establish coarse-grained alignments between image-
question pairs and knowledge entries, while Dense Passage Re-
trieval (DPR)-based architectures [ 15,16] enhance precision by
incorporating fine-grained features. For instance, Wiki-LLaVA [ 4]
employs a CLIP-based multi-stage retrieval pipeline to improvearXiv:2504.10074v2  [cs.AI]  15 Apr 2025

, , Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, and Bo Zheng
alignment, and RoRA-VLM [ 22] utilizes adversarial training tech-
niques to enhance robustness against irrelevant content. Despite
these advancements, achieving fine-grained alignment between
visual and textual information remains challenging, often leading
to suboptimal retrieval results.
On the generation side, researchers have sought to improve RAG
systems by enabling MLLMs to evaluate the relevance and accuracy
of the retrieved content autonomously. For example, ReflectiVA
[7] uses specialized tokens to steer the retrieval process, while
EchoSight [ 31] incorporates fine-tuned re-ranking modules to filter
out noisy or irrelevant documents. However, these self-evaluation
strategies typically depend on external annotation pipelines or aux-
iliary models, which may not fully capture the intrinsic knowledge
boundaries of the MLLM. This observation highlights the demand
for an integrated framework that can use the inherent knowledge of
MLLMs to guide both retrieval and filtering processes dynamically.
Towards the problems above, we propose a novel framework
named Multi-Modal Knowledge-Based Retrieval-Augmented Gener-
ation (MMKB-RAG). Unlike traditional RAG systems that rely solely
on external retrieval strategies, MMKB-RAG leverages the inherent
knowledge boundaries of the target MLLM to dynamically generate
specialized tags for filtering retrieved documents. Through its en-
dogenous annotation system, the MLLM autonomously determines
when retrieval is necessary and verifies both the consistency and
relevance of the retrieved knowledge, all based on its own intrinsic
knowledge limits. The key innovation of MMKB-RAG is its ability
to bridge the gap between parametric and retrieved knowledge by
transitioning from exogenous, auxiliary-model-dependent annota-
tion pipelines to an intrinsic, capability-aware system, addressing
the shortcomings of both multi-modal retrieval and self-evaluation
approaches.
Overall, this paper presents the following main contributions:
•Token System Framework : MMKB-RAG introduces a three-
stage process to determine retrieval necessity, evaluate the
relevance of individual documents, and verify the consis-
tency among multiple documents, ensuring accurate and
robust reasoning.
•Internal Knowledge Utilization : By leveraging the in-
herent knowledge of MLLMs and their interactions with
datasets, MMKB-RAG autonomously defines knowledge bound-
aries and guides retrieval without relying on external anno-
tations.
•Superior Performance : MMKB-RAG outperforms state-of-
the-art models in knowledge-based VQA tasks, particularly
in handling fine-grained factual queries and complex multi-
modal reasoning challenges.
2 Related Work
Multi-modal LLM. The emergence of large language models
(LLMs)[ 9,11,32] has catalyzed significant progress in multi-modal
LLMs (MLLMs)[ 6,18,30], enabling basic visual comprehension and
commonsense reasoning capabilities. Notably, implementations
such as LLaVA [ 18], Qwen-VL[ 30], and InternVL[ 6] have demon-
strated strong performance on standard visual question answering
(VQA) benchmarks.Knowledge-based VQA. Knowledge-based VQA tasks require
MLLMs to integrate information beyond visual content by leverag-
ing external knowledge sources. Early benchmarks such as KVQA[ 26],
OK-VQA[ 20], and A-OKVQA[ 25] focused primarily on common-
sense reasoning, an area where large-scale pre-trained MLLMs
perform effectively thanks to their implicit knowledge representa-
tions. Recent datasets like E-VQA [ 21] and InfoSeek [ 5] have pushed
the field toward Wikipedia-scale knowledge integration, necessitat-
ing a comprehensive understanding of specific Wikipedia entities
and fine-grained details. Although these benchmarks highlight the
upper bounds of current systems, even state-of-the-art MLLMs re-
main constrained by their parameter-intensive architectures when
processing detailed factual queries. This fundamental limitation fre-
quently results in the generation of hallucinated content[ 19,24,29].
The RAG framework[ 12] addresses these limitations by dynami-
cally integrating external knowledge during inference. By employ-
ing multi-stage inference pipelines that merge parametric knowl-
edge with retrieved information, RAG architectures exhibit consid-
erable promise in mitigating these intrinsic constraints.
Advanced RAG. Within the RAG paradigm, multi-modal retrieval
mechanisms typically employ two principal strategies: 1) CLIP-
based architectures employ contrastive image-text encoders[ 27] to
establish coarse-grained alignment between image-question pairs
and knowledge entries, and 2) DPR-based architectures[ 15,16]
incorporate fine-grained visual features (e.g., Regions-of-Interest) to
enhance retrieval precision. The retrieved entries are subsequently
integrated into MLLM as contextual references.
Recently, Wiki-LLaVA[ 4] has integrated knowledge through
CLIP-based multi-stage retrieval process, while RoRA-VLM[ 22] em-
ploys adversarial training to improve robustness against irrelevant
retrieved content via query-oriented visual token pruning. Mean-
while, EchoSight[ 31] introduces multi-modal re-ranking modules
through fine-tuned Q-Former. Although existing approaches pri-
marily focus on optimizing multi-modal retrieval, they neglect the
intrinsic role of MLLMs in self-evaluating knowledge boundaries.
Inspired by self-RAG[ 2], we propose the MMKB-RAG framework
where the MLLM autonomously generates specialized tokens to
determine whether retrieval is necessary and verify both the con-
sistency and relevance of the retrieved knowledge. Although the
most recent work ReflectiVA[ 7] implements a similar specialized-
token mechanism, its annotation pipeline fundamentally differs by
leveraging different MLLMs, thereby neglecting the knowledge lim-
itations of the target model. In contrast, our framework establishes
an endogenous, knowledge boundary-aware annotation system
that directly aligns label generation with the target model. This par-
adigm shift from exogenous, model-dependent annotation process
to intrinsic, capability-aware labeling approach, yields significant
performance gains.
3 Proposed Method
In Knowledge-based Visual Question Answering (VQA), the sys-
tem processes an image-question pair (𝐼,𝑄)and leverages external
knowledge to generate accurate responses. Traditional Multi-Modal
Retrieval-Augmented Generation (MMRAG) approaches retrieve

MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework , ,
Step1:Retrieve on demandMLLM
RetrieveNotRetrieveGenerateAnswer
Retrieve K documentsStep3:consistency-accordingrerankingrelevantrelevantrelevant214MLLM
GenerateAnswer24High-Consistency doc. put topStep2:Relevance-according Filtering12345+Inputrelevantrelevantirrelevantrelevantirrelevant21435Generate tag in parallelMLLM
MMKB-RAGFramework
0.950.930.880.230.17Top m documentsContradictory doc. drop last1Question: What is the main feeding method used by this bird?Input
Figure 2: MMKB-RAG Pipeline. Given an input query, MMKB-RAG initially assesses retrieval necessity. For retrieval-dependent
queries, we employ a two-stage filtering process (Steps 2 and 3) to ensure only high-quality reference documents are preserved.
This curated context is then provided to the MLLM to generate comprehensive and accurate responses.
relevant documents from knowledge bases to support answer gen-
eration, formulated as:
𝑦ans=arg max
𝑦MLLM(𝑦|𝐼,𝑄,𝐷𝑘,𝑃𝑣𝑞𝑎). (1)
where𝐷𝑘={𝑑1,𝑑2,...,𝑑𝑘}denotes the set of top- 𝑘documents
retrieved by retriever 𝑅given the image-question pair (𝐼,𝑄), and
𝑃𝑣𝑞𝑎represents the prompt template designed for visual question
answering.
As illustrated in Fig.2, our proposed MMKB-RAG (Multi-Modal
Knowledge-Based Retrieval-Augmented Generation) extends be-
yond conventional MMRAG frameworks through three key innova-
tions: (1) leveraging the model’s internal knowledge to determine
when external retrieval is necessary, improving efficiency; (2) dy-
namically re-ranking retrieved documents based on their relevance
to the input query; and (3) employing consistency checking to filter
out inconsistent information, thereby improving answer reliability
and accuracy. In the following sections, we detail our approach:
Section 3.1 introduces the token system employed in MMKB-RAG,
while Section 3.2 describes the model’s training methodology.
3.1 Token System for MMKB-RAG
Retrieval Token (RET). Given an input pair (𝐼,𝑄), where𝐼repre-
sents the image and 𝑄represents the question text, we classify the
input into two categories: (1) cases where the MLLM can answer
using internal knowledge, represented by [𝑁𝑜𝑅𝑒𝑡], and (2) cases
that require external knowledge retrieval for an accurate response,
represented by[𝑅𝑒𝑡]. For inputs labeled as [𝑅𝑒𝑡], the retrieval 𝑅
fetches the top- 𝑁most relevant documents, allowing the MLLM
to incorporate external knowledge for a more accurate answer; for
[𝑁𝑜𝑅𝑒𝑡]inputs, the MLLM directly generates the answer, thereby
optimizing computational efficiency. A fine-tuned MLLM predicts
these tags with the designated prompt 𝑃𝑅𝐸𝑇.
𝑅𝐸𝑇=MLLM𝑓𝑡(𝐼,𝑄,𝑃𝑅𝐸𝑇) (2)we extract the logits corresponding to [𝑅𝑒𝑡]and[𝑁𝑜𝑅𝑒𝑡]from
the first generated token and apply a softmax function to convert
them into normalized probability scores. The hyperparameter 𝛾is
then used to modulate the final predicted class. By varying 𝛾, we
can control the model’s propensity to perform retrieval operations.
Typically,𝛾is set to 0.5.
𝑅𝐸𝑇=(
[𝑅𝑒𝑡], if𝑆𝑐𝑜𝑟𝑒𝑅𝐸𝑇>𝛾
[𝑁𝑜𝑅𝑒𝑡],otherwise(3)
𝑆𝑐𝑜𝑟𝑒𝑅𝐸𝑇=exp(𝑧𝑅𝑒𝑡)
exp(𝑧𝑅𝑒𝑡)+exp(𝑧𝑁𝑜𝑅𝑒𝑡)(4)
Single-Relevant Token Rerank (SRT). For inputs classified as
[𝑅𝑒𝑡], the initially retrieved documents 𝐷𝑘are selected based solely
on embedding similarity scores. However, this may not align opti-
mally with the internal knowledge of MLLM, potentially introduc-
ing irrelevant or contradictory information. We propose leveraging
the MLLM’s internal knowledge capabilities for relevance determi-
nation. To this end, we introduce [𝑅𝑒𝑙]and[𝑁𝑜𝑅𝑒𝑙]tags, which
enable the MLLM to evaluate each candidate document via a dedi-
cated prompt 𝑃𝑆𝑅𝑇. For a given pair (I, Q), the prediction for the
i-th document 𝑑𝑖is formulated as follows:
𝑆𝑅𝑇𝑖=MLLM𝑓𝑡(𝐼,𝑄,𝑑𝑖,𝑃𝑆𝑅𝑇) (5)
Similarly to the previous section, we compute the softmax probabili-
ties for the first-token logits of document 𝑑𝑖, where𝑧𝑅𝑒𝑙corresponds
to[𝑅𝑒𝑙]and𝑧𝑁𝑜𝑅𝑒𝑙 to[𝑁𝑜𝑅𝑒𝑙]. We re-rank all 𝑘retrieved docu-
ments based on the probabilities of [𝑅𝑒𝑙]and select the top- 𝐾𝑆𝑅𝑇
documents as our final result, where 𝑘𝑆𝑅𝑇≤𝑘. This process can be
formulated as:
𝐷𝑘𝑆𝑅𝑇={𝑑𝑖|𝑖∈argTop𝑘𝑆𝑅𝑇{Score𝑗
𝑆𝑅𝑇|𝑗∈[1,𝑘]}} (6)
𝑆𝑐𝑜𝑟𝑒𝑖
𝑆𝑅𝑇=exp(𝑧𝑖
𝑅𝑒𝑙)
exp(𝑧𝑖
𝑅𝑒𝑙)+exp(𝑧𝑖
𝑁𝑜𝑅𝑒𝑙)(7)

, , Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, and Bo Zheng
NotRetrieveMLLM
QuestionMLLM
QuestionRetrieveConsistencyMLLM
relevantrelevantrelevantirrelevantrelevantrelevantrelevantirrelevantConsistent docContradictory doc.relevantMLLM
Question
w/o ragw/ ragirrelevantMLLM
Question
w/o ragw/ rag
Figure 3: Training strategy for token systems. RET determine whether to invoke the retrieval mechanism based on the model’s
response accuracy; SRT assess reference text quality by evaluating the model’s performance when utilizing these documents as
context; and CST uniformly evaluate all retrieved documents to deliver consistent reference documents.
Multi-Consist Filter (MCT). After retrieving the 𝐷𝑘𝑆𝑅𝑇docu-
ments using[𝑅𝑒𝑙]token, it is important to note that this token
assesses the relevance of each document individually with respect
to the input. However, This per-document evaluation may lead to
issues such as a lack of global context, or a collection of documents
that, despite being individually relevant, are not collectively coher-
ent. To address these shortcomings, we introduce a consistency
check that examines 𝐷𝑘𝑆𝑅𝑇from a holistic perspective to filter out
inconsistencies and assemble a more reliable reference set. Specifi-
cally, we employ a fine-tuned MLLM with a dedicated prompt 𝑃𝐶𝑆𝑇.
This process refines the collection to a subset of 𝑘𝐶𝑆𝑇 documents,
providing robust external knowledge for MLLM.
𝐷𝑘𝐶𝑆𝑇=MLLM ft(𝐷𝑘𝑆𝑅𝑇,𝑃𝐶𝑆𝑇) (8)
After obtaining the external knowledge 𝐷𝑘𝐶𝑆𝑇through the to-
ken system , we could generate answers based on this reference
knowledge, similar to the standard MMRAG approach. The intro-
duction of the token system significantly enhances the relevance
and accuracy of retrieved knowledge, thereby enabling the MLLM
to produce more precise and reliable answers. This process can be
summarized as follows:
𝑦ans=arg max
𝑦MLLM(𝑦|𝐼,𝑄,𝐷𝑘𝐶𝑆𝑇,𝑃𝑣𝑞𝑎). (9)
3.2 Training with Token System
In the previous section, we introduced three distinct tags to filter
retrieved documents, significantly enhancing the relevance and
accuracy of the reference materials and thereby improving the
MLLM’s response quality. Our methodology represents a signifi-
cant departure from existing approaches [ 2,7] that rely on external
annotation pipelines or auxiliary models. As illustrated in Fig.3,
the principal innovation lies in our knowledge boundary-aware
paradigm, where all tag determinations are explicitly derived from
the target MLLM’s intrinsic knowledge scope and limitations. This
self-contained framework ensures that tagging decisions emergethrough the model’s self-assessment mechanisms rather than ex-
ternal supervision, thereby achieving precise alignment with the
model’s epistemic boundaries.
RET. Considering the capabilities of MLLM in addressing VQA
tasks, it is observed that these models inherently possess a signifi-
cant amount of knowledge. This intrinsic knowledge enables them
to accurately answer questions without the need for any external
information retrieval. our method uniquely leverages the existing
knowledge boundaries of the MLLM itself to determine whether
external retrieval is required for each query.
The detailed algorithmic process is illustrated in Algorithm 1,
which initializes an empty dataset and constructs a training dataset
based on the model’s ability to correctly respond to queries. For cor-
rectly answered queries, we label them as [𝑅𝑒𝑡]as the model’s inher-
ent knowledge suffices. Conversely, incorrectly answered queries
are labeled as[𝑁𝑜𝑅𝑒𝑡], indicating retrieval is necessary due to
knowledge gaps or hallucination. Following data collection, we
fine-tune the MLLM using prompt 𝑃𝑅𝐸𝑇to enable it to determine
whether the target MLLM requires retrieval support for a given
query.
SRT. Traditional RAG systems typically employ embedding-based
methods to retrieval reference documents that are semantically
most relevant to the input query. However, this approach presents
several limitations: (1) semantic similarity alone does not guarantee
that the retrieved document will contribute to answering the ques-
tion correctly, and (2) the retrieval process operates independently
of the downstream MLLM, potentially introducing distractor con-
tent that adversely affects response accuracy. To overcome these
limitations, we propose a novel model-centric retrieval paradigm
that performs relevance assessment through the lens of the tar-
get MLLM itself, explicitly evaluating each candidate document’s
capacity to support accurate response generation.
As outlined in Algorithm 2, we introduce a novel data collection
strategy for training relevance assessment MLLM. Our method iden-
tifies relevant documents as those that convert incorrect answers
to correct ones, while irrelevant documents are those that corrupt

MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework , ,
Algorithm 1 Data Construction of RET
Input: image-question pairs {(𝐼1,𝑄1),(𝐼2,𝑄2),...,(𝐼𝑁,𝑄𝑁)}, target
MLLM
Output: RET training dataset 𝐷RET
1:Initialize an empty dataset 𝐷RET=∅
2:for𝑖=1to𝑁do
3: Input(𝐼𝑖,𝑄𝑖)into the MLLM
4: Obtain the inference response 𝐴𝑖from MLLM
5: if𝐴𝑖is correct then
6: Set𝑅𝐸𝑇𝑖=[𝑅𝑒𝑡]
7: else
8: Set𝑅𝐸𝑇𝑖=[𝑁𝑜𝑅𝑒𝑡]
9: end if
10: Add(𝐼𝑖,𝑄𝑖,𝑅𝑖)to𝐷RET
11:end for
12:return𝐷RET
Algorithm 2 Data Construction of SRT
Input: image-question pairs {(𝐼1,𝑄1),(𝐼2,𝑄2),...,(𝐼𝑁,𝑄𝑁)}, target
MLLM, retriever 𝑅
Output: SRT training dataset 𝐷SRT
1:Initialize an empty dataset 𝐷SRT=∅
2:for𝑚=1to𝑀do
3: Input(𝐼𝑚,𝑄𝑚)into the MLLM
4: Obtain the inference response 𝐴𝑚,direct from MLLM
5: Use𝑅to retrieve top 𝐾most similar documents 𝐷𝑚
𝐾=
{𝑑𝑚
1,𝑑𝑚
2,...,𝑑𝑚
𝑘}for(𝐼𝑚,𝑄𝑚)
6: for𝑘=1to𝐾do
7: Input(𝐼𝑚,𝑄𝑚,𝑑𝑚
𝑘)into the MLLM
8: Obtain the inference response 𝐴𝑚,𝑘from MLLM
9: if𝐴𝑚,direct is incorrect and𝐴𝑚,𝑘is correct then
10: Set𝑆𝑅𝑇𝑚,𝑘=[𝑅𝑒𝑙]
11: else if𝐴𝑚,direct is correct and𝐴𝑚,𝑛is incorrect then
12: Set𝑆𝑅𝑇𝑚,𝑘=[𝑁𝑜𝑡𝑅𝑒𝑙]
13: end if
14: Add(𝐼𝑚,𝑄𝑚,𝑑𝑚
𝑘,𝑆𝑅𝑇𝑚,𝑘)to𝐷SRT
15: end for
16:end for
17:return𝐷SRT
initially correct answers. This approach creates natural contrastive
pairs that enable effective fine-tuning of MLLM with prompt 𝑃𝑆𝑅𝑇,
optimizing their ability to discern helpful reference documents for
accurate question answering.
MCT. Traditional methods [ 2,7] that rely solely on single-relevance
token can only determine the relevance between a query and in-
dividual reference documents. This approach fails to address con-
tradictions, redundancies, and inconsistencies that may arise when
multiple reference documents are used simultaneously, ultimately
compromising the accuracy of responses. To address this limitation,
we propose the Multi-Consist mechanism, which performs unified
filtering and cleaning across multiple documents, eliminating con-
tradictions and other negative elements while preserving clean,
consistent, and coherent reference content for the target MLLM,
thereby enhancing response accuracy.
Algorithm 3 outlines our approach for constructing MCT train-
ing data from 𝐷𝑆𝑅𝑇. We first leverage the MLLM to generate concise
summaries of 𝐷𝑆𝑅𝑇, effectively eliminating redundant information.Algorithm 3 Data Construction of MCT
Input:𝐷SRT, target MLLM, negative sample ratio 𝜏%
Output: MCT training dataset 𝐷𝑀𝐶𝑇
1:foreach unique(𝐼𝑚,𝑄𝑚)in𝐷SRTdo
2: Extract documents 𝑆𝑟𝑚={𝑑𝑚
𝑘|𝑆𝑅𝑇𝑚,𝑘=[𝑅𝑒𝑙]}and𝑆𝑖𝑟𝑚={𝑑𝑚
𝑘|
𝑆𝑅𝑇𝑚,𝑘=[𝑁𝑜𝑅𝑒𝑙]}
3: if|𝑆𝑟𝑚|>1then
4: Apply summary process: 𝑆𝑢𝑚 =𝑀𝐿𝐿𝑀(𝑆𝑟𝑚)
5: Merge with irrelevant: 𝑆𝑚𝑖𝑥𝑚,𝐼𝑑𝑥 =𝑀𝑖𝑥(𝑆𝑟𝑚,𝜏%𝑆𝑖𝑟𝑚)
6: Add(𝐼𝑚,𝑄𝑚,𝑆𝑢𝑚,𝑆𝑚𝑖𝑥𝑚,𝐼𝑑𝑥)to𝐷𝑀𝐶𝑇
7: end if
8:end for
9:return𝐷𝑀𝐶𝑇
To build robustness against noisy references, we intentionally con-
taminate𝑆𝑟𝑚by introducing 𝜏%of irrelevant documents 𝑆𝑖𝑟𝑚, then
identify indices of 𝑆𝑟𝑚entries ranked by [Single Relevance] proba-
bility. This approach trains the model to distinguish high-quality
information within mixed-quality inputs. Our objective is to enable
the MLLM, when presented with the mixed-quality corpus 𝑆𝑚𝑖𝑥𝑚,
to simultaneously identify high-quality document indices and gen-
erate comprehensive summaries derived exclusively from these
reliable sources.
4 Experiments
4.1 Dataset
Datasets. We evaluate our approach on four knowledge-intensive
VQA benchmarks: (1) OKVQA [20] contains 14,000 questions across
diverse knowledge categories, with 5,000 validation samples used
in our experiments, evaluated using the VQA score metric [ 1]; (2)
E-VQA [21] comprises 221,000 question-answer pairs linked to
16,700 fine-grained Wikipedia entities, featuring both single-hop
and two-hop reasoning questions, where we use the 5,800 test
samples evaluated with the BEM score [ 3]; (3) InfoSeek [5] in-
cludes 1.3 million image-question pairs connected to approximately
11,000 Wikipedia pages, where following prior work [ 31], we re-
port results on the 73,000 validation samples using the official VQA
score script; and (4) M2KR [16], a multi-modal knowledge retrieval
benchmark that processes the knowledge bases of InfoSeek and
E-VQA at paragraph granularity, where we follow the established
protocol using the PreFLMR retriever for comparative evaluation
against the RA-VQA with PreFLMR baseline.
External Knowledge Bases. Both the InfoSeek and E-VQA datasets
are supported by external knowledge bases derived from Wikipedia
documents. Specifically, the E-VQA dataset is accompanied by a
knowledge base consisting of 2 million Wikipedia pages. Each page
includes the Wikipedia title, associated textual sections, and related
images. In contrast, the InfoSeek dataset leverages a more extensive
knowledge base that comprises 6 million Wikipedia entities. In
our experiments, we utilize the full 2 million-document knowledge
base for E-VQA. For InfoSeek, following recent studies [ 4,31], we
extract a subset of 100,000 pages1from the original corpus of 6
million documents. This approach ensures efficient data processing
while preserving the quality and coverage of the knowledge base.
1The knowledge base used for InfoSeek contains the same entities as in [4].

, , Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, and Bo Zheng
For OKVQA, we employ a knowledge corpus based on Wikipedia
documents selected for their pseudo-relevance as determined by
M2KR [ 16]. Both the training and test passage corpora include all
passages from this knowledge corpus. Evaluation of OKVQA is
performed exclusively on the M2KR dataset.
4.2 Implementation details
Retrieved Knowledge. Our experiments are particularly aimed
at optimizing the knowledge injection and comprehension capabil-
ities of large models, focusing on how these models process and
understand external knowledge sources after retrieve.In our ex-
perimental design, we primarily utilized the EVA-CLIP-8B[ 28] for
image retrieval.In this experiment ,we choose the image-to-image
retrieval, where we evaluate the similarity between a query image
and images embedded within Wikipedia documents to retrieve the
corresponding Wikipedia pages. To align with the ReflectiVA[ 7],
we set the number of retrieved web pages to 5.
Model Architecture and Training detailed. We employ Qwen2-
VL-7B-Instruct [ 30] as our foundation model, which integrates a
675M-parameter Vision Transformer (ViT) for visual encoding with
the Qwen2-7B language model for text processing. The architecture
incorporates an MLP connector that inherently bridges image to-
kens and language representations, enabling effective multi-modal
information fusion without external modules. To optimize com-
putational efficiency, we implement LoRA [ 10] fine-tuning with a
batch size of 512.
Training Data Collection. In our study, we trained MMKB-RAG
using subsets of the official Infoseek and E-VQA training datasets.
To optimize computational efficiency, we randomly sampled 10%
of the available data for model training. Importantly, we employed
the answer model itself for generating fine-tuning data without
reliance on external models such as GPT-4o.
MCT type choose. During the MCT phase of our experiments,
we implemented three distinct refinement strategies: Filter : utiliz-
ing only documents indexed by MLLM outputs, preserving exclu-
sively high-quality documents; Merge : employing only the MLLM-
generated summary based on high-quality documents; and Re-
rank : prioritizing MLLM-identified high-quality documents while
retaining less consistent documents that might still contain valu-
able information. Comprehensive ablation studies validating the
effectiveness of these strategies are presented in subsequent sec-
tions.
4.3 Comparisons with SOTA
We evaluate our MMKB-RAG method against state-of-the-art mod-
els on the E-VQA and InfoSeek benchmarks, with results presented
in Table 1. The evaluation is organized into two categories: (1)
Zero-shot MLLMs without knowledge injection and (2) Retrieval-
Augmented Models with knowledge injection. For baseline compar-
isons, we include Qwen2-VL-7B-Instruct and its fine-tuned variant
Qwen2-VL-7B-Instruct(SFT) trained on the target dataset in both
tasks. To establish a fair comparison with Wiki-LLaVA, we employ
Contriever to retrieve the top-k documents with highest similarity
to the query.MMKB-RAG demonstrates superior performance across all set-
tings. In the zero-shot scenario, our approach outperforms mod-
els like BLIP-2 and InstructBLIP that rely solely on pre-trained
knowledge without external retrieval mechanisms. For the retrieval-
augmented scenario, MMKB-RAG significantly surpasses both Qwen2-
VL-7B-Instruct and its fine-tuned counterpart, despite these models
sharing similar architectural foundations. This performance gap
highlights the effectiveness of our proposed MMKB-RAG approach
in substantially improving response accuracy.
When utilizing the EVA-CLIP retriever for image-to-image re-
trieval, MMKB-RAG achieves even more substantial improvements
over competitive methods such as EchoSight and Reflective that
employ similar retrieval mechanisms. These results demonstrate
that our method can achieve superior performance even under
similar token system. Fig.4 provides a qualitative comparison on
different models.
4.4 Ablation Studies and Analyses
4.4.1 Effectiveness of MMKB-RAG Tokens. To evaluate each token’s
contribution, we conducted an ablation study by progressively en-
abling different components of our framework. Table 2 summarizes
these results, where 𝑘represents the number of retrieved docu-
ments. When only RET is enabled, the MLLM uses the default
setting (𝑘=5). In contrast, when SRT is utilized, paragraphs with
relevance scores above 0.5 are retained and ranked in descending
order of relevance ( 𝑘=auto).
Our experiments reveal distinctive contributions from each to-
ken: (1) RET improves performance across all metrics, though its
impact diminishes when other tokens are introduced; (2) SRT sig-
nificantly enhances all evaluation metrics, indicating that while
embedding-based retrieval can identify semantically related doc-
uments, it cannot guarantee the accuracy of the final answers; (3)
MCT further refines results by eliminating inconsistent or redun-
dant documents, substantially improving performance beyond SRT
alone.
Optimal performance is achieved when all three tokens work in
concert. This confirms the synergistic design of MMKB-RAG: RET
provides foundational knowledge, SRT ensures relevance, and con-
sistency filtering removes noise, collectively delivering state-of-the-
art performance across both datasets. Fig.5 provides a visualization
of how each token contributes to the performance.
4.4.2 Effectiveness of document quantity. In this section, we investi-
gate the impact of document numbers 𝑘on the performance of SRT
and MCT strategies. To isolate the effect of 𝑘, we fix the retrieval
module to be always enabled. The results are presented in Tables 3
and 4.
SRT. For E-VQA, 𝑘=5achieves peak performance, suggesting
that moderate document quantities effectively balance information
completeness and noise reduction. The InfoSeek dataset similarly
favors𝑘=5, though performance degrades sharply at 𝑘=20, indi-
cating that the automatic selection mechanism may inadvertently
discard critical information.
MCT. On InfoSeek, merge-based MCT demonstrates superior per-
formance as it emphasizes precise answers by more accurately
identifying consistent documents and eliminating redundant in-
formation. In contrast, rerank-based MCT excels on E-VQA under

MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework , ,
Table 1: VQA accuracy scores on E-VQA test set and InfoSeek validation set where all results from retrieval-augmented models
are reported without considering any re-ranking stage to reorder retrieved web pages. Bold indicates state-of-the-art, underline
denotes second-best, and †marks our reproduced results. Gray color indicates, results that are not directly comparable due to
different knowledge bases. All retrieval-augmented results exclude re-ranking.
E-VQA InfoSeek
Model LLM Retriever Feature Single-Hop All Unseen-Q Unseen-E All
Zero-shot MLLMs
BLIP-2 [13] Flan-T5 XL - - 12.6 12.4 12.7 12.3 12.5
InstructBLIP [8] Flan-T5 XL - - 11.9 12.0 8.9 7.4 8.1
LLaVA-v1.5 [17] Vicuna-7B - - 16.3 16.9 9.6 9.4 9.5
LLaVA-v1.5 [17] LLaMA-3.1-8B - - 16.0 16.9 8.3 8.9 7.8
Qwen2-VL-Instruct †[30] Qwen-2-7B - - 16.4 16.4 17.9 17.8 17.9
Qwen2-VL-Instruct(sft) †[30] Qwen-2-7B - - 25.0 23.8 22.7 20.6 21.6
Retrieval-Augmented Models
Wiki-LLaVA [4] Vicuna-7B CLIP ViT-L/14+Contriever Textual 17.7 20.3 30.1 27.8 28.9
Wiki-LLaVA [4] LLaMA-3.1-8B CLIP ViT-L/14+Contriever Textual 18.3 19.6 28.6 25.7 27.1
EchoSight [31] Mistral-7B/LLaMA-3-8B EVA-CLIP-8B Visual 19.4 - - - 27.7
EchoSight [31] LLaMA-3.1-8B EVA-CLIP-8B Visual 26.4 24.9 18.0 19.8 18.8
ReflectiVA [7] LLaMA-3.1-8B EVA-CLIP-8B Visual 35.5 35.5 28.6 28.1 28.3
Qwen2-VL-Instruct †[30] Qwen-2-7B EVA-CLIP-8B + Contriever Visual+Textual 25.9 23.6 21.6 21.3 21.4
Qwen2-VL-Instruct(sft) †[30] Qwen-2-7B EVA-CLIP-8B + Contriever Visual+Textual 32.6 29.9 23.3 23.8 23.6
MMKB-RAG (Ours) Qwen-2-7B EVA-CLIP-8B Visual 39.7 35.9 36.4 36.3 36.4
question : In which part(s) of the world does this fish live?
Qwen2-VL-Instruct :
Cuba✗
Qwen2-VL-Instruct(sft) :
Bahía de Cochinos ✗
MMKB-RAG (Ours):
Western Atlantic ✓question : Which architect designed this building?
Qwen2-VL-Instruct :
I don’t know ✗
Qwen2-VL-Instruct(sft) :
Gert Jan van Dijk ✗
MMKB-RAG (Ours):
César Pelli ✓question : What color is the inside of this plant?
Qwen2-VL-Instruct :
white✗
Qwen2-VL-Instruct(sft) :
white✗
MMKB-RAG (Ours):
purplish ✓
question : How did this bird fare against the australian white ibis?
Qwen2-VL-Instruct :
The Australian magpie has similar
plumage but has red eyes and is
found mainly on the ground. ✗
Qwen2-VL-Instruct(sft) :
well✗
MMKB-RAG (Ours):
narrowly ahead ✓question : On what street is this building located?
Qwen2-VL-Instruct :
M Street ✗
Qwen2-VL-Instruct(sft) :
M Street ✗
MMKB-RAG (Ours):
Rhode Island Avenue ✓question : Where did former residents of this district relocate to?
Qwen2-VL-Instruct :
Liverpool ✗
Qwen2-VL-Instruct(sft) :
central Liverpool ✗
MMKB-RAG (Ours):
Kirkby, Cantril Farm, and Netherley
✓
Figure 4: Sample qualitative results between MMKB-RAG and w/o MMKB-RAG.
Table 2: Ablation study on the token system.
RET SRT MCT 𝑘 E-VQA InfoSeek
Single-Hop All Unseen-Q Unseen-E All
5 32.57 29.88 23.33 23.82 23.57
✓ 5 32.72 30.01 24.55 24.59 24.57
✓ auto 38.09 34.63 34.07 33.55 33.81
✓ ✓ auto 38.13 34.66 34.07 33.56 33.81
✓ ✓ auto 39.62 35.89 36.43 36.32 36.37
✓ ✓ ✓ auto 39.66 35.91 36.44 36.34 36.37
BEM metrics, which emphasize semantic similarity rather than ex-
act matching. The rerank strategy proves advantageous in this con-
text as it prioritizes documents demonstrating higher consistency
while preserving information breadth, thus minimizing the risk of
omitting critical details that might occur with other approaches.
4.4.3 Effectiveness of Training Data Construction. To evaluate the
impact of different training data construction methods, we compare
our approach with a more powerful external model (GPT-4o) on the
SRT module, which has the most significant influence on systemperformance. Due to cost constraints, we limited GPT-4o-generated
samples to 2,000, while our method produced datasets ranging from
2,000 to 100,000 samples. Table 5 summarizes the results.
Our experiments reveal two key findings: First, increasing train-
ing data volume with our method does not yield performance im-
provements. This suggests that small, high-quality datasets are
sufficient for the model to learn effective preference patterns. Sec-
ond, with equal training data volume (2,000 samples), our method
significantly outperforms GPT-4o despite the latter’s superior gen-
eral capabilities. We hypothesize that our construction approach
produces training data that better aligns with the answering model’s
characteristics, resulting in more effective document re-ranking
and scoring.
4.5 Comparison on M2KR dataset
We further validate our approach on the M2KR dataset, which is
specifically designed to benchmark multimodal models on image+text-
to-text retrieval tasks. In this evaluation, models must generate
responses by jointly reasoning over visual and textual inputs. Ta-
ble 6 presents a comparative analysis of our MMKB-RAG method

, , Zihan Ling, Zhiyao Guo, Yixuan Huang, Yi An, Shuai Xiao, Jinsong Lan, Xiaoyong Zhu, and Bo Zheng
In which city is this square located?RetrievedDocument1:In2008,Hamburghadanareaof755.2km²(291.6sqmi),92%waslandand8%waterareas.Areaforthetrafficinfrastructurewas12%(9,183haor35.46sqmi).Thesewerenonbuilt-upareas.Munich...RetrievedDocument2:TheStuttgartStadtbahnoperatesfrom04:00-01:00.Monday-Friday:Servicefrequencyisevery10minutesbetween06:00-07:00and20:00-20:30.Saturday:Servicefrequencyisevery10minutesbetween09:30-10:30and20:00-20:30....Answer：Munichw/o[RET]Answer：Amsterdamretrieval score: 0.08w/[RET]
what airfield received supplies from kiplinhall during wwii?RetrievedDocument1:Hakeateretifoliaisapricklyshrubthatcanreach3\xa0m(10\xa0ft)inheight.Ithasspirallyarranged,thick,tough...RetrievedDocument1:SarahTalbotCarpentermarriedChristopherHattonTurnorfromStokeRochfordinLincolnshirein1907buttheyhadnochildren.ThecoupleneverresidedatKiplinandthehousewaslet....RAFCatterick,RAFCroftandRAFMiddletonStGeorge....w/o[SRT]w/[SRT]Answer：Ramsbottom&&PrestonAnswer：RAF Catterick, RAF Croft and RAF Middleton St George
In which city is this hotel located?RetrievedDocument1:HoteldelCoronado,alsoknownasTheDelandHotelDel,isahistoricbeachfronthotelinthecityofCoronado...RetrievedDocument2:Attheageof72,HindebegantoinvestinpropertyandminesinNewMexicoasahobby....HindeisburiedinMountHopeCemetery,SanDiego,California...foundedin1898andrunbyWillE.Keller."w/o[CST]w/o[CST]RetrievedDocument1:HoteldelCoronado,alsoknownasTheDelandHotelDel,isahistoricbeachfronthotelinthecityofCoronado...RetrievedDocument2:The2010UnitedStatesCensusreportedthattheCityofCoronadohadapopulationof24,697...w/[CST]w/[CST]Answer：San DiegoAnswer：Coronado
Figure 5: Illustration of our Token System, showcasing the effects of RET, SRT, and CST. Dashed lines indicate the operational
positions of different tokens, with red text highlighting erroneous information and green text denoting correct information.
Table 3: Impact of varying the number of retrieved docu-
ments on SRT performance.
𝑘 E-VQA InfoSeek
Single-Hop All Unseen-Q Unseen-E All
auto 38.09 34.63 34.07 33.55 33.81
1 37.30 33.79 33.39 30.77 32.03
5 39.56 35.89 35.05 34.91 34.98
10 38.90 35.33 33.68 33.18 33.43
15 37.39 33.84 30.16 29.82 29.99
20 35.20 32.08 25.32 25.09 25.09
against strong baselines, including RA-VQAv2 and various Qwen2-
VL-Instruct variants, all leveraging the PreFLMR retriever. The
results demonstrate that MMKB-RAG consistently outperforms
both RA-VQAv2 and the baseline knowledge-based models across
evaluation metrics, validating the effectiveness of our approach.
5 Conclusion
In this study, we introduce a new framework called multi-modal
Knowledge-Based Retrieval-Enhanced Generation (MMKB-RAG).
This framework leverages the knowledge boundary of the answer
model to dynamically generate tags for the RAG system, which
enables more efficient filtering of retrieved documents and retaining
only the most relevant and accurate references. By doing so, MMKB-
RAG significantly improves the accuracy and robustness of model
responses in multi-modal tasks.Table 4: Impact of Different MCT Types on Performance.
MCT type 𝑘 E-VQA InfoSeek
Single-Hop All Unseen-Q Unseen-E All
merge auto 34.84 32.14 36.43 36.32 36.37
rerank auto 39.62 35.90 34.53 34.96 34.74
filter auto 39.47 35.74 35.82 35.04 35.43
merge 5 35.77 33.00 36.10 35.80 36.00
rerank 5 39.16 35.43 35.17 34.98 35.07
filter 5 38.48 35.00 34.81 34.08 34.44
merge 10 34.61 31.79 34.38 34.48 34.43
rerank 10 38.61 35.08 32.83 32.96 32.89
filter 10 38.78 35.20 33.52 32.76 33.14
merge 15 33.47 30.57 35.91 35.65 35.78
rerank 15 37.16 33.72 30.09 29.90 29.99
filter 15 37.81 34.52 35.11 33.99 34.54
merge 20 31.98 29.64 35.61 35.04 35.32
rerank 20 35.32 32.14 25.20 25.17 25.18
filter 20 35.24 32.35 34.72 33.33 34.01
References
[1]Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence
Zitnick, Dhruv Batra, and Devi Parikh. 2016. VQA: Visual Question Answering.
arXiv:1505.00468 [cs.CL] https://arxiv.org/abs/1505.00468
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv
preprint arXiv:2310.11511 (2023).
[3]Jannis Bulian, Christian Buck, Wojciech Gajewski, Benjamin Boerschinger, and
Tal Schuster. 2022. Tomayto, Tomahto. Beyond Token-level Answer Equivalence
for Question Answering Evaluation. arXiv:2202.07654 [cs.CL] https://arxiv.org/

MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework , ,
Table 5: Comparison of SRT’s performance trained with GPT-
4o and our proposed method.
E-VQA InfoSeek
num𝑘 Single-Hop All Unseen-Q Unseen-E All
GPT-4o 2k auto 39.81 36.13 33.03 33.22 33.12
MMKB-RAG2k auto 40.06 36.24 35.22 36.65 35.92
5k auto 39.68 35.68 34.78 34.72 34.75
10k auto 39.22 35.50 33.73 33.93 33.83
50k auto 39.24 35.39 34.85 34.14 34.49
100k auto 38.09 34.63 34.07 33.55 33.81
Table 6: Performance comparison on the M2KR dataset.
Model OKVQA Infoseek E-VQA
Zero-shot MLLMs
RA-VQAv2 55.44 21.78 19.80
Qwen2-VL-Instruct 60.45 21.75 19.01
Qwen2-VL-Instruct(sft) 64.08 26.00 26.72
Retrieval-Augmented Models
RA-VQAv2 w/ FLMR 60.75 - -
RA-VQAv2 w/ PreFLMR 61.88 30.65 54.45
Qwen2-VL-Instruct w/ PreFLMR 46.99 24.68 51.81
Qwen2-VL-Instruct(sft) w/ PreFLMR 65.07 30.74 53.89
MMKB-RAG w/ PreFLMR 65.44 34.72 60.93
abs/2202.07654
[4]Davide Caffagni, Federico Cocchi, Nicholas Moratelli, Sara Sarto, Marcella Cornia,
Lorenzo Baraldi, and Rita Cucchiara. 2024. Wiki-LLaVA: Hierarchical Retrieval-
Augmented Generation for Multimodal LLMs. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition . 1818–1826.
[5]Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter,
and Ming-Wei Chang. 2023. Can Pre-trained Vision and Language Models Answer
Visual Information-Seeking Questions? arXiv:2302.11713 [cs.CV] https://arxiv.
org/abs/2302.11713
[6]Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan
Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al .2024. Internvl: Scaling
up vision foundation models and aligning for generic visual-linguistic tasks. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition .
24185–24198.
[7]Federico Cocchi, Nicholas Moratelli, Marcella Cornia, Lorenzo Baraldi, and Rita
Cucchiara. 2025. Augmenting Multimodal LLMs with Self-Reflective Tokens for
Knowledge-based Visual Question Answering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition .
[8]Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao,
Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. InstructBLIP:
Towards General-purpose Vision-Language Models with Instruction Tuning.
arXiv preprint arXiv:2305.06500 (2023).
[9]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
[10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, Weizhu Chen, et al .2022. Lora: Low-rank adaptation of large
language models. ICLR 1, 2 (2022), 3.
[11] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, et al .2023. Mistral 7B. arXiv preprint
arXiv:2310.06825 (2023).
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[13] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Boot-
strapping Language-Image Pre-training with Frozen Image Encoders and Large
Language Models.
[14] Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng,
Hui Wang, Hai-Tao Zheng, Fei Huang, Jingren Zhou, and Philip S. Yu. 2025.
Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA
Dataset and Self-adaptive Planning Agent. arXiv:2411.02937 [cs.CL] https:
//arxiv.org/abs/2411.02937
[15] Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, and Bill Byrne. 2023.
Fine-grained late-interaction multi-modal retrieval for retrieval augmented visualquestion answering. Advances in Neural Information Processing Systems 36 (2023),
22820–22840.
[16] Weizhe Lin, Jingbiao Mei, Jinghong Chen, and Bill Byrne. 2024. PreFLMR: Scal-
ing Up Fine-Grained Late-Interaction Multi-modal Retrievers. arXiv preprint
arXiv:2402.08327 (2024).
[17] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved Baselines
with Visual Instruction Tuning. In CVPR .
[18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruc-
tion tuning. Advances in neural information processing systems 36 (2024).
[19] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang,
Liping Hou, Rongjun Li, and Wei Peng. 2024. A survey on hallucination in large
vision-language models. arXiv preprint arXiv:2402.00253 (2024).
[20] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019.
Ok-vqa: A visual question answering benchmark requiring external knowledge.
InProceedings of the IEEE/cvf conference on computer vision and pattern recognition .
3195–3204.
[21] Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar,
Howard Zhou, Fei Sha, André Araujo, and Vittorio Ferrari. 2023. Encyclope-
dic VQA: Visual questions about detailed properties of fine-grained categories.
arXiv:2306.09224 [cs.CV] https://arxiv.org/abs/2306.09224
[22] Jingyuan Qi, Zhiyang Xu, Rulin Shao, Yang Chen, Jin Di, Yu Cheng, Qifan Wang,
and Lifu Huang. 2024. RoRA-VLM: Robust Retrieval-Augmented Vision Language
Models. arXiv preprint arXiv:2410.08876 (2024).
[23] Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, S.M
Towhidul Islam Tonmoy, Aman Chadha, Amit Sheth, and Amitava Das. 2023.
The Troubling Emergence of Hallucination in Large Language Models - An
Extensive Definition, Quantification, and Prescriptive Remediations. In Proceed-
ings of the 2023 Conference on Empirical Methods in Natural Language Processing ,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 2541–2573. doi:10.18653/v1/2023.emnlp-main.155
[24] Vipula Rawte, Amit Sheth, and Amitava Das. 2023. A survey of hallucination in
large foundation models. arXiv preprint arXiv:2309.05922 (2023).
[25] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and
Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering
using world knowledge. In European conference on computer vision . Springer,
146–162.
[26] Sanket Shah, Anand Mishra, Naganand Yadati, and Partha Pratim Talukdar. 2019.
Kvqa: Knowledge-aware visual question answering. In Proceedings of the AAAI
conference on artificial intelligence , Vol. 33. 8876–8884.
[27] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. 2023. Eva-clip:
Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389
(2023).
[28] Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang,
and Xinlong Wang. 2024. Eva-clip-18b: Scaling clip to 18 billion parameters.
arXiv preprint arXiv:2402.04252 (2024).
[29] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining
Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal
llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition . 9568–9578.
[30] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin
Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al .2024. Qwen2-vl: Enhancing
vision-language model’s perception of the world at any resolution. arXiv preprint
arXiv:2409.12191 (2024).
[31] Yibin Yan and Weidi Xie. 2024. EchoSight: Advancing Visual-Language Models
with Wiki Knowledge. In Findings of the Association for Computational Linguistics:
EMNLP 2024 . Association for Computational Linguistics, 1538–1551. doi:10.18653/
v1/2024.findings-emnlp.83
[32] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2024. Qwen2. 5
technical report. arXiv preprint arXiv:2412.15115 (2024).
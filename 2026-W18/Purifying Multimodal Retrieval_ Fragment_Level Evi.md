# Purifying Multimodal Retrieval: Fragment-Level Evidence Selection for RAG

**Authors**: Xihang Wang, Zihan Wang, Chengkai Huang, Cao Liu, Ke Zeng, Quan Z. Sheng, Lina Yao

**Published**: 2026-04-30 08:50:03

**PDF URL**: [https://arxiv.org/pdf/2604.27600v1](https://arxiv.org/pdf/2604.27600v1)

## Abstract
Multimodal Retrieval-Augmented Generation (MRAG) is widely adopted for Multimodal Large Language Models (MLLMs) with external evidence to reduce hallucinations. Despite its success, most existing MRAG frameworks treat retrieved evidence as indivisible documents, implicitly assuming that all content within a document is equally informative. In practice, however, sometimes only a small fraction of a document is relevant to a given query, while the remaining content introduces substantial noise that may lead to performance degradation. We address this fundamental limitation by reframing MRAG as a fine-grained evidence selection problem. We propose Fragment-level Evidence Selection for RAG (FES-RAG), a framework that selects atomic multimodal fragments rather than entire documents as grounding evidence. FES-RAG decomposes retrieved multimodal documents into sentence-level textual fragments and region-level visual fragments, enabling precise identification of evidence that directly supports generation. To guide fragment selection, we introduce Fragment Information Gain (FIG), a principled metric that measures the marginal contribution of each fragment to the MLLM's generation confidence. Based on FIG, we distill fragment-level utility judgments from a high-capacity MLLM into a lightweight selector, achieving accurate evidence selection with low inference overhead. Experiments on the M2RAG benchmark show that FES-RAG consistently outperforms state-of-the-art document-level MRAG methods, achieving up to 27 percent relative improvement in CIDEr. By selecting fewer yet more informative fragments, our approach substantially reduces context length while improving factual accuracy and generation coherence.

## Full Text


<!-- PDF content starts -->

Purifying Multimodal Retrieval: Fragment-Level Evidence
Selection for RAG
Xihang Wang∗
Zhejiang University
Hangzhou, China
xihangwang@163.comZihan Wang∗
Meituan LongCat Interaction Team
Beijing, China
wangzihan14@meituan.comChengkai Huang†
University of New South Wales,
Macquarie University
Sydney, Australia
chengkai.huang1@unsw.edu.au
Cao Liu
Meituan LongCat Interaction Team
Beijing, China
liucao@meituan.comKe Zeng
Meituan LongCat Interaction Team
Beijing, China
zengke02@meituan.comQuan Z. Sheng
Macquarie University
Sydney, Australia
michael.sheng@mq.edu.au
Lina Yao
University of New South Wales,
CSIRO’s Data61
Sydney, Australia
lina.yao@unsw.edu.au
Abstract
Multimodal Retrieval-Augmented Generation (MRAG) is widely
adopted for Multimodal Large Language Models (MLLMs) with ex-
ternal evidence to reduce hallucinations. Despite its success, most
existing MRAG frameworks treat retrieved evidence as indivisible
documents, implicitly assuming that all content within a docu-
ment is equally informative. In practice, however, sometimes only
a small fraction of a document is relevant to a given query, while
the remaining content introduces substantial noise that may lead
to performance degradation. We address this fundamental limi-
tation by reframing MRAG as a fine-grained evidence selection
problem. We propose Fragment-level Evidence Selection for RAG
(FES-RAG), a framework that selects atomic multimodal fragments
rather than entire documents as grounding evidence. FES-RAG
decomposes retrieved multimodal documents into sentence-level
textual fragments and region-level visual fragments, enabling pre-
cise identification of evidence that directly supports generation.
To guide fragment selection, we introduce Fragment Information
Gain (FIG), a principled metric that measures the marginal con-
tribution of each fragment to the MLLM’s generation confidence.
Based on FIG, we distill fragment-level utility judgments from a
high-capacity MLLM into a lightweight selector, achieving accurate
evidence selection with low inference overhead. Experiments on
the𝑀2RAG benchmark show that FES-RAG consistently outper-
forms state-of-the-art document-level MRAG methods, achieving
up to 27% relative improvement in CIDEr. By selecting fewer yet
∗These authors contributed equally to this work.
†Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809692more informative fragments, our approach substantially reduces
context length while improving factual accuracy and generation
coherence.
CCS Concepts
•Information systems→Information retrieval.
Keywords
Retrieval-Augmented Generation, Large Language Model, Multi-
modal Large Language Model
ACM Reference Format:
Xihang Wang, Zihan Wang, Chengkai Huang, Cao Liu, Ke Zeng, Quan
Z. Sheng, and Lina Yao. 2026. Purifying Multimodal Retrieval: Fragment-
Level Evidence Selection for RAG. InProceedings of the 49th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.ACM, New York, NY,
USA, 12 pages. https://doi.org/10.1145/3805712.3809692
1 Introduction
Multimodal Large Language Models (MLLMs) have shown remark-
able capability in understanding and generating content across
various multimodal tasks [ 10,15,34,36]. Despite these advances,
MLLMs are frequently constrained by inherent limitations such as
hallucinations and outdated knowledge [ 36,45]. To mitigate these
challenges, Multimodal Retrieval-Augmented Generation (MRAG)
has emerged as a critical paradigm [ 36]. By retrieving relevant
external multimodal evidence, such as specific text documents or
visual images, MRAG can enhance the accuracy and reliability of
the MLLM’s generation [1, 48].
The effectiveness of MRAG hinges fundamentally on the quality
of the retrieved context [ 14,19,53]. When the retrieved context is
contaminated with irrelevant, redundant, or contradictory informa-
tion, MLLMs are prone to being misled by such distracting noise,
which often precipitates factual hallucinations and undermines the
arXiv:2604.27600v1  [cs.IR]  30 Apr 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
Figure 1: Comparison of the traditional MRAG pipeline (a)
and our FES-RAG framework (b). Traditional methods only
perform document-level retrieval and reranking, while our
approach adds an additional document segmentation and
fragment selection stage to purify the evidence.
reliability of the output [ 2,37,50]. This vulnerability is further
compounded by the lost-in-the-middle effect, a phenomenon where
MLLMs struggle to identify and leverage critical evidence when
it is situated within the intermediate segments of a long context
window [ 11,13,26]. Consequently, even high-quality evidence can
fail to support effective reasoning if it is obscured by surrounding
noise or poorly positioned within the input.
To address this challenge, our research focuses on refining re-
trieved data at a granular level. The primary objective is to facilitate
a transition from the monolithic documents toward atomic, fine-
grained segments. We seek to investigate how the marginal utility
of evidence can be quantified at the atomic level, thereby enabling
the filtering of irrelevant, redundant, or contradictory noise that
typically compromises downstream reasoning. Ultimately, our goal
is to provide MLLMs with a compact context that contains only
essential evidence, so that generation is guided by more reliable
multimodal information.
Current relevant approaches, however, fall short of this aim due
to their reliance on coarse-grained document processing [ 3,49].
Methods such as InfoGain-RAG [ 49] and MM-R5 [ 51] focus on
document-level filtering and reranking, yet they still operate on
the premise of treating retrieved content as a monolithic document.
This limitation is further amplified in the multimodal domain: re-
trieved images often contain only a small Region of Interest (ROI)
that is pertinent to the query, while the majority of the visual
content consists of cluttered backgrounds, irrelevant objects, or
distracting scenes [ 43]. As shown in Figure 1 (a), the presence of
competing textual entities (e.g., the 2020 laureate) and irrelevant vi-
sual ROIs (e.g., unrelated political figures in the right part) distracts
the model, leading it to mistakenly conclude that the evidence for
the 2019 prize is insufficient.
To address these limitations, we propose Fragment-level Evi-
dence Selection for RAG (FES-RAG), a framework that moves be-
yond conventional document-level reranking toward fine-grainedfragment purification. FES-RAG introduces a multimodal segmen-
tation pipeline that decomposes retrieved documents into atomic
fragments. To quantify the utility of these fragments, we extend the
concept of Information Gain (IG) [ 17,49] from the document level
to the fragment level, introducing Fragment Information Gain (FIG).
This metric measures the marginal contribution of each atomic unit
to the MLLM’s generation confidence, providing a high-fidelity
supervision signal for training the fragment selector. To facilitate
real-time deployment, we propose a knowledge distillation (KD)
training strategy to bridge the gap between selection precision and
inference efficiency.
By feeding only high-information-density, purified evidence into
the generator, we ensure that the MLLM can focus its reasoning
attention exclusively on the fragments that directly support the
answer generation process. As shown in Figure 1 (b), this decompo-
sition enables the targeted extraction of critical information while
filtering out irrelevant text sentences and visual regions.
Our primary contributions are summarized as follows:
•We propose the FES-RAG framework, a novel paradigm shift-
ing from document-level to fragment-level evidence process-
ing for MRAG. It integrates systematic multimodal document
segmentation and fine-grained fragment selection to purify
retrieved evidence, effectively mitigating noise interference
in traditional coarse-grained systems.
•We develop a scalable pipeline combining multimodal seg-
mentation and FIG calculation, which enables precise quan-
tification of fragment utility and provides high-quality su-
pervision data for fine-grained evidence selection.
•We design a knowledge distillation-based training strategy
for the evidence selector, transferring the fragment utility-
judging capability of a high-capacity teacher MLLM to a
lightweight student model, balancing evidence extraction
precision and real-time inference efficiency.
•Extensive experiments on the M2RAG benchmark (MMQA,
Image Captioning, Fact Verification) show our FES-RAG out-
performs state-of-the-art coarse-grained baselines consis-
tently, achieving up to 27% relative CIDEr improvement
while reducing context token consumption.
2 Related Work
2.1 Multi-modal RAG
The integration of external knowledge into Large Language Mod-
els (LLMs) through Retrieval-Augmented Generation (RAG) has
proven highly effective in mitigating hallucinations and improv-
ing knowledge currency [ 9,16,20,22]. Building upon the success
of text-based RAG, MRAG extends this paradigm to MLLMs, en-
abling them to reason over cross-modal evidence such as images,
videos, and structured tables [ 35,46]. Early works in MRAG primar-
ily focused on aligning visual features with textual queries using
dual-encoder architectures like CLIP [ 41] or BLIP [ 24] to retrieve
relevant images that complement textual prompts.
Recent advancements have shifted toward more complex retrieve-
then-generate pipelines. For instance, frameworks like RoRA-VLM
[39] and Re-ViLM [ 53] demonstrate that retrieving pertinent vi-
sual context significantly improves performance on knowledge-
intensive tasks such as multimodal question answering (MMQA)

Purifying Multimodal Retrieval: Fragment-Level Evidence
Selection for RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
and image captioning. To benchmark these capabilities, the 𝑀2RAG
dataset was introduced, providing a rigorous evaluation of how
MLLMs handle long-context multimodal retrieval across diverse
real-world scenarios [35].
Most existing methods operate at the document level, treating a
retrieved item as a monolithic unit [ 51]. However, providing entire
documents often introduces substantial noise, such as irrelevant
background objects in images or irrelevant sentences in text, which
can distract the MLLM and lead to degraded reasoning quality
[7,52]. Our work addresses this gap by proposing a fine-grained
selection mechanism based on information gain, ensuring that only
information-dense evidence is presented to the generator. Notably,
to the best of our knowledge, there are currently no existing base-
lines that implement sentence-level reranking specifically within
the MRAG domain.
2.2 Information Gain
The concept of information gain, rooted in Shannon’s informa-
tion theory, has traditionally served as a cornerstone for feature
selection and decision tree induction by quantifying the expected
reduction in entropy after observing a feature [ 40]. In the context
of RAG, the paradigm has shifted from measuring simple semantic
relevance to evaluating the utilitarian value of retrieved evidence,
specifically regarding how much a piece of information contributes
to reducing the model’s uncertainty during generation [ 17]. Recent
studies have highlighted that high semantic similarity between
a query and a document does not always guarantee a boost in
generation accuracy [ 49]. To address this, researchers have intro-
duced confidence-based utility metrics. For instance, Self-RAG [ 3]
employs critique tokens to evaluate the relevance and support-
iveness of retrieved passages, essentially performing a form of
implicit information gain assessment. Similarly, FLARE [ 18] and
Iter-RetGen [ 42] utilize the model’s predictive entropy to decide
when to retrieve and which documents provide the most signifi-
cant knowledge update to the generator’s internal state. Building
upon these confidence-based approaches, InfoGain-RAG [ 49] intro-
duces Document Information Gain (DIG), a novel metric designed
to precisely quantify the contribution of retrieved documents to
correct answer generation. DIG measures a document’s value by
computing the difference in an LLM’s generation confidence with
and without the document augmented. This framework enables the
training of specialized rerankers that prioritize helpful documents
while effectively filtering out irrelevant or even misleading content.
However, document-level metrics such as DIG overlook fine-
grained evidence in long documents or cluttered images, introduc-
ing noise that distracts the generator. We therefore extend infor-
mation gain to the atomic level and propose FIG, which measures
the contribution of individual multimodal units, such as sentences
or visual regions, to maximize the information density of retrieved
context.
3 Task Definition
We consider the task of MRAG, where a model generates an answer
𝑔for a query𝑥by leveraging a large-scale multimodal database D.State-of-the-art MRAG frameworks employ a two-stage retrieve-
then-rerank pipeline to select the most relevant context. The end-
to-end process can be formally expressed as:
Dcand=𝑀 ret(𝑥,D),(1)
Dsorted=𝑀 rank(𝑥,D cand),(2)
𝑔=𝑀 gen(𝑥,D sorted),(3)
Here,𝑀retand𝑀rankdenote the initial retriever and the coarse
reranker, respectively, while 𝑀genrepresents the downstream MLLM
generator. First, 𝑀retefficiently recalls a broad candidate set Dcand
from the database. Subsequently, 𝑀rankrefines this set by perform-
ing a fine-grained relevance assessment, selecting a high-precision
subsetDsorted . This refined context is then prepended to the query
𝑥and fed into𝑀 gento produce the final answer𝑔.
4 Methodology
4.1 Overview of FES-RAG
Traditional MRAG systems typically operate at a coarse-grained
document level, where entire retrieved documents are provided
as context to an MLLM [ 39,53]. However, full documents often
contain significant amounts of redundant information, irrelevant
background noise, or contradictory signals that can mislead the
generation process [26, 36].
To mitigate these issues, we propose the FES-RAG framework,
which implements a “Retrieve–Rerank–Select–Generate” pipeline.
The introduced selection stage ensures that only informative evi-
dence is presented to the generator. By incorporating a dedicated
selector module, FES-RAG performs fine -grained evidence pruning,
extracting only the high -utility fragments (individual sentences
and visual regions) while discarding irrelevant content. This results
in a purified, information -rich context for the final MLLM, thereby
improving both factual grounding and computational efficiency.
4.2 Multimodal Document Segmentation
To align the granularity of evidence utilization with the atomic na-
ture of cross-modal reasoning in MLLMs, we propose a document
segmentation strategy that breaks down the retrieved document
into its most fundamental informative segments. By isolating in-
dividual sentences and specific visual ROIs, we can filter out the
inherent noise within documents and transform a collection of
coarse data into a refined pool of information-dense evidence.
4.2.1 Textual Segmentation.To identify the most informative text
fragments while filtering out distracting filler content, we propose
a Recursive Binary Segmentation strategy. Unlike traditional RAG,
which relies on fixed-length sliding windows, our approach dy-
namically determines the optimal granularity of evidence based on
semantic relevance scores.
As shown in Algorithm 1, given a query 𝑥and a retrieved text doc-
ument𝑡, we utilize a cross-encoder model (e.g., Jina-Reranker-m0
[21]) as a scoring function 𝑓𝜙(𝑥,𝑡) , which estimates the relevance
of a text segment 𝑡relative to𝑥. The process begins by calculating
the base score of the entire document 𝑠𝑝𝑎𝑟𝑒𝑛𝑡 =𝑓𝜙(𝑥,𝑡) . We then
bisect𝑡into two segments, 𝑠𝑙𝑒𝑓𝑡and𝑠𝑟𝑖𝑔ℎ𝑡, at the sentence boundary
nearest to the midpoint.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
TheGandhi Peace
Prize fortheyear
2019 is being
conferred on…Input
FIG Calculation
Teacher Supervision Signals
Student Selector TrainingRetrieve(Coarse)
Document Segmentation
Build Candidate Pool
Select Fragments
Generate Answer
Query x Retrieved Documents Ground -truth
Answer y
...
People familiar with developments
said oncondition ofanonymity that
the selection procedure does not
allow the Gandhi Peace Prize tobe
awarded posthumously .However,
given the contributions ofSultan
Qaboos and Rahman topeace, non-
violence and amelioration ofhuman
s u f f e r i n g s a n d t h e i r s p e c ia l
relationship with India, anexception
was made asaspecial gesture to
honour and celebrate them, the
people said.A: Support
Query x
RetrieverTop N Documents
...
People familiar with developments
said oncondition ofanonymity that
the selection procedure does not
allow the Gandhi Peace Prize tobe
awarded posthumously .However,
given the contributions ofSultan
Qaboos and Rahman topeace, non-
violence and amelioration ofhuman
s u f f e r i n g s a n d t h e i r s p e c ia l
relationship with India, anexception
was made asaspecial gesture to
honour and celebrate them, the
people said.
With Fragment Without Fragment
| | | |
1111( , ) log ( , , ) log ( , ),| | | |yy
j i i j i i
iiFIG f x y P y y x f P y y xyy
===− ∣ ∣ ∣
1, if ( | , )
0, otherwisej
jFIG f x yz =

Text Segmentation
Recursive Binary
Segmentation
𝑡1
𝑡2
𝑡𝑚…Sentence 1
Sentence 2
Sentence mImage Segmentation
Grounding DINO
𝑣1
𝑣2
𝑣𝑘…
Fine -Grained Fragments Remaining Coarse Documents
𝑡1…𝑡2𝑡𝑚𝑣1…𝑣2𝑣𝑘
…
Query x +Candidate
…Trained Selector
（Student）Top K Selected
Fragments
𝑡5𝑣3𝑡9
MLLM
Query x …
𝑡5𝑣3𝑡9…Selected Fragments
Answer
(a) FIG Calculation & Selector Training (b) FIG Calculation & Selector Training
t
Student Selector
InputTeacher Logits
T
jS
Teacher SupervisionDistillation Loss
2
1ˆ (1 ) ( , )
1S
BCE j j
N
TS
KD jj
jKLzz
ssN TTT
  =−+

 =    
           
( , )ijP y x f∣
()iP y x∣
Frozen Trainable Text Fragment Visual Fragment Document Student v
 Teacher
Teacher TeacherDocument Segmentation
Text Segmentation
Recursive Binary
Segmentation
𝑡1
𝑡2
𝑡𝑚…Sentence 1
Sentence 2
Sentence mImage Segmentation
Grounding DINO
𝑣1
𝑣2
𝑣𝑘…… …
Figure 2: Overview of the proposed FES-RAG framework. (a) FIG Calculation and Selector Training: During training, retrieved
multimodal documents are decomposed into textual fragments via recursive binary segmentation and visual fragments via
Grounding DINO. A frozen teacher MLLM estimates the Fragment Information Gain (FIG) of each fragment by comparing the
likelihood of the ground-truth answer with and without the fragment. The resulting hard labels and teacher logits are then
used to train a lightweight student selector through binary classification and knowledge distillation. (b) FES-RAG Inference
Pipeline: During inference, FES-RAG first retrieves top-ranked documents and segments them into fine-grained textual and
visual fragments. These fragments are combined with the remaining coarse-grained candidates to form a hybrid evidence
pool. The trained student selector scores this pool and selects the top- 𝐾high-utility evidence units, which are fed to the MLLM
generator as compact, purified context.
The segmentation step is considered gainful if either 𝑓𝜙(𝑥,𝑡𝑙𝑒𝑓𝑡)
or𝑓𝜙(𝑥,𝑡𝑟𝑖𝑔ℎ𝑡)exceeds𝑠𝑝𝑎𝑟𝑒𝑛𝑡 . If this condition is met, the docu-
ment is partitioned, and the process is applied recursively to the
partitioned segment with the higher score. The recursion termi-
nates when further subdivision no longer yields a higher relevance
score. This ensures that the selector receives highly informative
fragments, preventing the model from being misled by long, par-
tially irrelevant paragraphs.
4.2.2 Visual Segmentation.To mitigate the noise interference in
coarse-grained retrieved images, we implement a visual segmenta-
tion process designed to extract query-relevant local regions. We
utilize Grounding DINO [ 27] as the primary segmentation frame-
work to decompose monolithic images into atomic visual fragments.Grounding DINO is a cross-modal grounding framework that
unifies vision-language understanding with fine-grained object
granularity. The architecture comprises a dual-encoder utilizing
a Swin Transformer [ 30] for visual feature extraction and BERT
[8] for textual encoding—coupled with a feature enhancer for deep
cross-modality fusion. By mapping both modalities into a shared
latent space, the framework performs language-guided region se-
lection through a cross-modality decoder. This enables the model to
identify and segment image areas corresponding to arbitrary textual
descriptions in a zero-shot manner, providing a robust foundation
for fine-grained evidence extraction without being constrained by
predefined object categories.

Purifying Multimodal Retrieval: Fragment-Level Evidence
Selection for RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Algorithm 1Fine-grained Textual Segmentation via Score-Driven
Recursive Decomposition
Require:𝑥 : user query, 𝑡: retrieved text document, 𝑓𝜙: cross-
encoder relevance scorer
Ensure:𝑡∗: the optimal atomic text fragment
1:functionRecurSplit(𝑥,𝑡)
2:if𝑡is a single sentencethen
3:return𝑡
4:end if
5:𝑠 parent←𝑓𝜙(𝑥,𝑡)
6:(𝑡 left,𝑡right)←SplitDoc(𝑡)
7:𝑠 left←𝑓𝜙(𝑥,𝑡 left)// Left subsegment score
8:𝑠 right←𝑓𝜙(𝑥,𝑡 right)// Right subsegment score
9:ifmax(𝑠 left,𝑠right)>𝑠 parent then
10:if𝑠 left>𝑠rightthen
11:returnRecurSplit(𝑥,𝑡 left)
12:else
13:returnRecurSplit(𝑥,𝑡 right)
14:end if
15:else
16:return𝑡// No gain, keep intact
17:end if
18:end function
19:returnRecurSplit(𝑥,𝑡)
Formally, let 𝑓𝐺𝐷(𝑥,𝐼) denote the Grounding DINO, for a re-
trieved image 𝐼∈ D𝑐𝑎𝑛𝑑 and the user query 𝑥, the process of
generating filtered visual fragments is formulated as follows:
B={(𝑏𝑗,𝑠obj
𝑗,𝑠sem
𝑗)}𝑛
𝑗=1=𝑓𝐺𝐷(𝑥,𝐼),(4)
Bfiltered ={𝑏𝑗|(𝑏𝑗,𝑠obj
𝑗,𝑠sem
𝑗)∈B,subject toC},(5)
C={𝑠obj
𝑗>𝜏obj,𝑠sem
𝑗>𝜏sem,Area(𝑏𝑗)>𝜏 size},(6)
whereBrepresents the set of 𝑛candidate bounding boxes 𝑏𝑗. Each
box is associated with two distinct confidence measures: an object-
ness score𝑠𝑜𝑏𝑗
𝑗(representing the probability that the region contains
a distinct entity) and a semantic alignment score 𝑠𝑠𝑒𝑚
𝑗(representing
the relevance of the region to the query 𝑥). The set of constraints
Ccomprises three complementary criteria designed to ensure the
quality of the evidence. First, the objectness constraint ( 𝑠𝑜𝑏𝑗
𝑗>𝜏𝑜𝑏𝑗)
filters out spurious regions that the model perceives as non-entities
or background artifacts. Second, the semantic alignment constraint
(𝑠𝑠𝑒𝑚
𝑗>𝜏𝑠𝑒𝑚) ensures that the extracted regions are contextually
relevant to the specific user query. Finally, the geometric size con-
straint ( Area(𝑏𝑗)>𝜏𝑠𝑖𝑧𝑒) discards fragments whose pixel area is
too small to provide recognizable visual information.
The regions defined by B𝑓𝑖𝑙𝑡𝑒𝑟𝑒𝑑 are subsequently cropped from
the original image 𝐼. These purified image patches, denoted as
atomic visual fragments, effectively discard irrelevant background
pixels and non-target objects, thereby maximizing the marginal
utility of the visual context during the final generation phase.
4.3 FIG Quantification and Teacher Adaptation
To optimize a high-fidelity selector capable of identifying critical ev-
idence, it is imperative to establish a robust supervision signal thatquantifies the intrinsic utility of each candidate fragment. While
semantic relevance (e.g., cosine similarity) often serves as a proxy
for importance [ 6], it frequently fails to align with the actual re-
quirements of the generation process. Building upon the principles
of DIG [ 49], we propose FIG, a metric designed to measure the
marginal contribution of a fragment to the answer generation.
4.3.1 FIG Definition and Calculation Details.Formally, we define
Fragment Information Gain (FIG) as the improvement in the length-
normalized log-likelihood of the ground-truth answer 𝑦when the
MLLM is augmented with fragment𝑓 𝑗:
𝐹𝐼𝐺(𝑓𝑗|𝑥,𝑦)=1
|𝑦||𝑦|∑︁
𝑖=1log𝑃(𝑦𝑖|𝑦<𝑖,𝑥,𝑓𝑗)−1
|𝑦||𝑦|∑︁
𝑖=1log𝑃(𝑦𝑖|𝑦<𝑖,𝑥),
(7)
where𝑦𝑖is the𝑖-th token of the ground-truth answer of length |𝑦|,
and𝑦<𝑖represents the preceding generated tokens. Here, 𝑃(𝑦𝑖|
𝑦<𝑖,𝑥,𝑓𝑗)represents the probability of generating the correct token
𝑦𝑖given the multimodal query𝑥, the specific fragment𝑓 𝑗, and the
previous context, while 𝑃(𝑦𝑖|𝑦<𝑖,𝑥)denotes the model’s base
confidence derived from its internal parametric knowledge alone.
This length normalization is essential to prevent the metric from
being disproportionately biased towards shorter responses.
This metric effectively reflects the utility of fragments by filtering
out the MLLM’s prior biases, thereby providing a clear signal of
whether a fragment provides foundational evidence or introduces
distracting noise. Furthermore, to capture the model’s intrinsic
generative confidence without external prompt bias, we compute
these probabilities using a zero-shot, open-ended generation format
rather than a multi-reference few-shot setup.
In our framework, we employ a high-capacity MLLM [ 4] to
compute these values, leveraging its superior cross-modal reasoning
depth to ensure that the resulting utility scores serve as high-fidelity
targets for the selector training.
4.3.2 Selector Training via Hard Label Supervision.MLLMs inher-
ently function as generative models rather than discriminators,
making them unsuitable for direct fragment-level evidence selec-
tion. To address this gap, we first construct high-quality supervision
signals and then train a multimodal selector to identify critical evi-
dence fragments without relying on ground-truth answers during
inference.
We first compute FIG scores using the methodology described
above to quantify the marginal contribution of each fragment 𝑓𝑗to
the generation confidence of the ground-truth answer 𝑦, denoted
as𝐹𝐼𝐺(𝑓𝑗|𝑥,𝑦) . These continuous FIG scores are then converted
into discrete hard labels 𝑧𝑗using a threshold 𝜏, where𝑧𝑗denotes
whether a fragment is critical for reasoning:
𝑧𝑗=(
1,if𝐹𝐼𝐺(𝑓 𝑗|𝑥,𝑦)>𝜏
0,otherwise,(8)
A label𝑧𝑗=1indicates that the fragment 𝑓𝑗provides meaningful
evidence for answering query 𝑥, while𝑧𝑗=0marks the fragment
as noise or redundant content. This process yields a supervised
dataset where each sample is a tuple (𝑥,𝑓𝑗,𝑧𝑗)—consisting of a
query, a multimodal fragment (text sentence or visual ROIs), and
its corresponding binary utility label.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
After constructing this supervised dataset, we train the selector
on this dataset to map the input pair (𝑥,𝑓𝑗)to the hard label 𝑧𝑗.
The model takes only the query 𝑥and fragment 𝑓𝑗as input (no
ground-truth answer is provided), learning to distinguish between
informative fragments and noise purely from cross-modal context.
The optimization objective is the Binary Cross-Entropy (BCE) loss:
L𝑠𝑒𝑙𝑒𝑐𝑡𝑜𝑟 =−1
𝑁𝑁∑︁
𝑗=1
𝑧𝑗log(ˆ𝑧𝑗)+(1−𝑧 𝑗)log(1− ˆ𝑧𝑗)
,(9)
where𝑁is the total number of fragments in the training set, and
ˆ𝑧𝑗=𝜎(𝑓𝜃(𝑥,𝑓𝑗))(with𝜎as the sigmoid function) is the selector’s
predicted probability that fragment 𝑓𝑗can contribute to the answer
generation.
Unlike rerankers that optimize semantic relevance, minimizing
this loss enables the selector to approximate the generator’s internal
belief update, where it can reliably identify evidence that directly
supports accurate reasoning.
4.4 Student Selector Optimization via
Knowledge Distillation
While the fine-tuned MLLM Selector demonstrates strong reason-
ing capability in identifying high-utility atomic fragments, its large
parameter count incurs prohibitive inference latency—especially
when scoring dozens of atomic fragments (sentences and visual
ROIs) per query [ 54]. To address the need for efficient inference
in real-world MRAG systems, we propose a knowledge distillation
strategy [ 28,29,31] to transfer the fragment utility discrimina-
tion capability of a high-capacity teacher selector to a lightweight
student selector. We employ a lightweight student selector (initial-
ized with Jina-Reranker-2B [ 21]) and optimize it using a multi-task
knowledge distillation paradigm. This framework allows the stu-
dent to inherit the precise discrimination capability of the teacher
without incurring its computational cost, ensuring both high selec-
tion accuracy and efficient inference.
The student selector 𝑓𝜃is trained using a hybrid objective that
aligns its predictions with both the ground-truth hard labels and
the teacher’s soft probability distribution. The total distillation loss
L𝐾𝐷is defined as:
L𝐾𝐷=1
𝑁𝑁∑︁
𝑗=1"
(1−𝛼)L𝐵𝐶𝐸(𝑧𝑗,ˆ𝑧𝑆
𝑗)+𝛼𝑇2L𝐾𝐿 
𝜎 𝑠𝑇
𝑗
𝑇!
∥𝜎 𝑠𝑆
𝑗
𝑇!!#
,
(10)
where𝑁is the total number of fragments in the training set, 𝑠𝑇
𝑗
and𝑠𝑆
𝑗denote the raw output logits of the teacher and the student
model, respectively, while ˆ𝑧𝑆
𝑗=𝜎(𝑠𝑆
𝑗)represents the student’s
final predicted probability. The hyperparameter 𝛼balances two
complementary supervision signals, and 𝑇denotes the distillation
temperature. Specifically, the term LBCEemploys the binary hard
label𝑧𝑗to enforce hard-label supervision. By providing explicit
decision supervision, it guides the student toward the core objective
of determining whether a fragment should be selected or discarded,
thereby promoting high classification fidelity. However, relying
solely on discrete labels can lead to information collapse, where
fragments with substantially different utility scores are mapped
to the same binary label, and can introduce quantization noise
around the threshold 𝜏. To mitigate these limitations, the term LKLenables soft distribution-level distillation by distilling the teacher’s
calibrated probability distribution. Regulated by the temperature
parameter𝑇, these soft targets preserve fine-grained relative utility
information, allowing the student to capture nuanced evidence
rankings and providing dense, continuous gradient signals even
when the predicted class is already correct.
4.5 FES-RAG Inference
The inference process of FES-RAG follows a coarse-to-fine trajec-
tory designed to maximize information density while minimizing
context noise. The complete procedure is detailed in Algorithm 2
and consists of the following core phases:
•Coarse-grained Retrieval & Reranking:Given a query 𝑥, the
system first retrieves 𝑁retcandidates using a retriever 𝑀retand
reorders them via a coarse reranker 𝑀rank. This stage effectively
narrows the search space from the entire corpus to a manageable
set of documents.
•Document Segmentation & Selection:To eliminate intra-
document noise, the top-ranked items are decomposed into atomic
fragments (textual sentences and visual ROIs). To prevent text
redundancy, textual documents are entirely replaced by their
fragments, whereas original images are retained alongside their
visual ROIs to preserve potential global context. Crucially, these
fine-grained fragments are combined with the remaining coarse-
grained candidates to form a hybrid candidate pool. The distilled
selector𝑀distill then evaluates this unified set, identifying the
evidence with the highest predicted FIG.
•Purified Generation:Instead of feeding entire monolithic doc-
uments, only the top-𝑘purified fragments are prepended to the
query. The generator 𝑀genthen produces the final answer 𝑔based
on this information-dense context:
By filtering out irrelevant background objects and filler text at the
fragment level, this pipeline ensures the MLLM focuses on essential
evidence, effectively mitigating the lost-in-the-middle phenomenon
while reducing computational overhead.
5 Experiments
In this section, we conduct comprehensive experiments to evaluate
the effectiveness of our enhanced framework, which integrates
fine-grained segmentation (visual and textual segmentation) and
knowledge distillation. We aim to answer the following research
questions:
•(RQ1) Overall Performance:Does the proposed FES-RAG
consistently outperform existing document-level baselines
(e.g., zero-shot, naive retrieval, and coarse-grained rerank-
ing) across diverse multimodal tasks and various MLLM
architectures?
•(RQ2) Ablation & Validation:How do the core compo-
nents, particularly knowledge distillation and candidate depth
(𝑘), contribute to the overall performance, and how well does
the proposed Fragment Information Gain (FIG) align with
final generation correctness?
•(RQ3) Modality Analysis:How does fine-grained purifica-
tion impact performance across different modalities (visual
vs. textual), and does it inherently resolve the MLLMs’ sus-
ceptibility to noise better than brute-force truncation?

Purifying Multimodal Retrieval: Fragment-Level Evidence
Selection for RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Algorithm 2Inference pipeline of FES-RAG
Require:𝑥 : user query,D: multimodal database, 𝑀ret: retriever,
𝑀rank: coarse reranker, 𝑀distill: distilled selector, 𝑀gen: gener-
ator,𝑁ret: number of retrieved documents, 𝑁seg: number of
documents for segmentation, 𝑘: number of selected fragments
for generation
Ensure:𝑔: generated answer
1:Initialize:D aug←∅
2:{Phase 1: Coarse-grained Retrieval & Reranking}
3:D cand←𝑀 ret(𝑥,D,𝑁 ret)
4:D sorted←𝑀 rank(𝑥,D cand)
5:{Phase 2: Document Segmentation}
6:D top←D sorted[1 :𝑁 seg]
7:for𝑑∈D topdo
8:ifIsImage(𝑑)then
9:𝑉 frags←ImageSegment(𝑑)
10:D aug←D aug∪{𝑑}∪𝑉 frags
11:else ifIsText(𝑑)then
12:𝑇 frags←RecurSplit(𝑥,𝑑)
13:D aug←D aug∪𝑇frags
14:end if
15:end for
16:{Phase 3: Fine-grained Selection}
17:D aug←D aug∪D sorted[𝑁seg+1 :𝑁 ret]
18:𝑆 distill←𝑀 distill(𝑥,D aug)// Score the utility of the hybrid pool
19:D final←Sort(D aug,𝑆distill)
20:{Phase 4: Purified Generation}
21:D ctx←D final[1 :𝑘]
22:𝑔←𝑀 gen(𝑥,D ctx)
23:return𝑔
•(RQ4) System Efficiency:How does FES-RAG balance the
trade-off between reasoning performance, space efficiency
(context token consumption), and time efficiency (end-to-end
inference latency) compared to coarse-grained pipelines?
5.1 Experimental Setup
Datasets.We evaluate our framework on theM2RAGbench-
mark [ 32], a challenging large-scale dataset designed to assess
retrieval-augmented generation capabilities across heterogeneous
modalities. The benchmark simulates realistic open-domain scenar-
ios by including a mix of relevant, irrelevant, and partially relevant
multimodal documents. It encompasses three distinct tasks: (1)
Multi-Modal QA, requiring models to synthesize answers from
visual and textual evidence; (2)Image Captioning, which tests the
ability to generate descriptive text conditioned on retrieved visual
context; and (3)Fact Verification, a classification task determining
whether multimodal evidence supports a given claim.
Evaluation Metrics.We employ comprehensive metrics to eval-
uate generation quality and reasoning accuracy. For generative
tasks (MMQA and Image Captioning), we report BLEU-1/2/3/4 [ 38]
and ROUGE-1/2/L [ 25], which quantify n-gram precision and recall
against ground-truth references. Additionally, we utilize CIDEr [ 44],
a metric specifically designed for image captioning that capturesconsensus-based semantic similarity and has been shown to cor-
relate better with human judgment in vision-language tasks. For
the fact verification task, performance is assessed using standard
Accuracy and F1 Score to evaluate classification precision.
Baselines.To verify the universality of our approach, we first
employ diverse backbone MLLMs, including open-source MLLMs
(Qwen2.5-VL[ 5],LLaVA-NeXT[ 23],InternVL3.5[ 47]) as well as
MM-RAIT-Qwen2.5VL[ 32], a specialized baseline fine-tuned on
the M2RAG benchmark for retrieval-augmented tasks. Based on
these generators, we compare our proposed method against four
retrieval baselines: (1)None-MRAG(zero-shot inference without
retrieval); (2)Naïve-MRAG(Top-3 retrieval using raw embeddings);
(3)Jina-Reranker(off-the-shelf Top-3 without fine-tuning); and (4)
InfoGain-MM, which extends the InfoGain-RAG [ 49] to the multi-
modal domain (serving as our coarse-grained Top-3 baseline). Our
proposed method, denoted as FES-RAG, utilizes fine-grained seg-
mentation and distillation. Note that we compare the baseline’s
Top-3 documents against our Top-5 segments, as they incur compa-
rable token consumption, ensuring a fair evaluation under a strictly
controlled context window budget.
Implementation Details.Experiments are conducted on 8
NVIDIA A100 GPUs with 80GB of memory each. The training
dataset is approximately balanced, containing 10k positive samples
and 12k negative samples. For the initial retrieval stage ( 𝑀ret), we
employ Visualized BGE [ 55], a state-of-the-art multimodal dense
retriever, to encode both queries and documents into a shared
embedding space. We implement the knowledge distillation strat-
egy where a Qwen3-VL-32B [ 4] teacher guides the Jina-Reranker-
m0 [ 21] student. We select Qwen3-VL-32B as the teacher due to
its strong cross-modal reasoning capability and proven effective-
ness across various multimodal benchmarks. The student model
is optimized to minimize Eq. 10 with distillation weight 𝛼= 0.7
and temperature 𝑇= 2. Training is performed for 5 epochs with
a global batch size of 32, using the AdamW optimizer [ 33] with a
learning rate of2 𝑒−5. Regarding the inference hyperparameters, we
set the initial retrieval depth to 𝑁ret=100to ensure high recall. The
segmentation budget is set to 𝑁seg=15, prioritizing the top-ranked
candidates for document segmentation to balance efficiency and
coverage. For selector training, the hard-label generation threshold
𝜏is set to 0.2. For Grounding DINO segmentation, we adopt filtering
thresholds of𝜏 obj=0.40,𝜏 sem=0.35, and𝜏 size=2,500 px2
5.2 Overall Performance (RQ1)
Table 1 reports the comparative performance across three distinct
multimodal tasks and four MLLMs. The results reveal several crit-
ical insights into the limitations of current MRAG systems and
the efficacy of FES-RAG:(i) The Failure of Generic Reranking.
WhileNaïve-MRAGsignificantly boosts performance over the zero-
shot baseline (None-MRAG), the off-the-shelfJina-Rerankeryields
marginal or even negligible gains (e.g., only +1.5 CIDEr on MMQA
with MM-RAIT). This empirically confirms a misalignment in ex-
isting retrieval paradigms: generic rerankers optimize for semantic
similarity, which does not necessarily translate to reasoning utility.
A document can be semantically similar to a query yet lack the
specific visual or textual evidence required to answer it.(ii) Gran-
ularity Matters.While our coarse-grained baseline,InfoGain-MM,

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
Table 1: Main comparison of FES-RAG-top5 against baselines across multiple MLLMs. FES-RAG utilizes fine-grained segmenta-
tion and distillation. The results show that FES-RAG consistently achieves state-of-the-art performance. Best results are in
bold, second-best are underlined .
Model MethodMultimodal QA (MMQA) Image Captioning Fact Verification
B-1 B-2 B-3 B-4 R-1 R-2 R-L CIDEr B-1 B-2 B-3 B-4 R-1 R-2 R-L CIDEr ACC F1
MM-RAIT-
Qwen2.5VLNone-MRAG 33.88 27.39 23.15 20.20 48.31 30.52 46.25 206.42 16.18 10.46 7.56 5.68 34.16 15.43 32.58 79.41 59 61.15
Naïve-MRAG-top3 40.67 31.39 26.08 23.07 56.56 33.37 54.09 247.13 17.15 11.41 7.90 5.89 36.09 16.89 34.57 84.41 62 62.95
Jina-Reranker-top3 40.46 31.61 26.29 23.41 56.34 34.17 54.05 248.63 19.39 13.05 9.47 7.42 38.05 18.31 35.74 96.66 63 64.31
InfoGain-MM-top3 42.95 33.38 27.60 24.14 61.24 35.89 57.91 264.78 23.47 17.46 13.74 11.36 42.91 24.10 41.24 136.44 68 69.80
FES-RAG-top5 49.31 39.48 34.60 31.35 67.94 42.95 65.93 336.42 27.83 21.44 17.58 14.96 48.13 29.35 46.57 171.51 70 72.71
LLaVA-NeXTNone-MRAG 34.85 30.00 27.04 24.49 48.36 37.00 45.61 250.82 6.42 3.08 2.02 1.63 19.17 5.34 17.33 18.98 39 35.76
Naïve-MRAG-top3 35.73 30.81 27.65 24.90 49.81 37.83 46.93 250.52 9.30 5.52 3.85 2.84 23.99 9.74 21.87 35.48 47 31.14
Jina-Reranker-top3 36.19 31.37 28.32 25.78 50.08 38.37 47.44 259.41 10.05 5.98 4.10 3.02 25.79 10.68 22.91 44.60 50 34.26
InfoGain-MM-top3 37.89 33.16 30.21 27.71 52.66 41.01 50.22 278.43 13.17 8.18 5.95 4.43 29.85 12.84 25.96 57.78 55 38.25
FES-RAG-top5 43.26 35.12 31.18 28.42 60.72 44.50 57.79 299.29 16.16 11.26 8.33 6.87 36.23 18.53 34.01 92.82 65 65.69
InternVL3.5None-MRAG 25.34 19.78 16.75 14.44 38.82 26.67 34.92 114.24 11.76 6.69 4.37 3.07 23.66 9.42 21.47 45.09 41 38.96
Naïve-MRAG-top3 29.57 24.12 20.32 17.84 46.52 31.99 43.66 156.10 10.96 6.24 3.61 2.75 27.12 12.43 24.33 41.46 47 46.89
Jina-Reranker-top3 30.57 25.17 21.29 18.53 47.66 33.21 44.42 166.41 11.43 6.63 3.94 2.99 27.91 11.98 24.11 40.73 52 52.02
InfoGain-MM-top3 34.26 28.42 25.04 22.25 51.44 36.57 49.33 219.21 15.12 8.64 5.84 4.79 32.04 15.38 28.29 64.00 55 55.04
FES-RAG-top5 41.77 33.19 28.27 25.48 61.58 37.34 59.05 264.87 18.60 13.26 9.94 8.07 39.09 20.40 37.18 108.18 67 68.62
Qwen2.5-VLNone-MRAG 25.41 20.64 17.61 14.91 39.30 27.78 34.97 105.33 8.97 4.46 2.84 2.03 20.05 6.64 18.20 19.51 51 50.67
Naïve-MRAG-top3 30.67 25.45 22.14 19.55 45.04 32.63 41.42 185.94 11.75 6.92 4.93 3.70 26.99 10.77 24.55 46.64 60 62.26
Jina-Reranker-top3 29.89 25.19 22.16 19.68 45.41 33.50 41.39 174.55 12.62 7.16 4.56 3.49 27.82 11.86 25.43 45.73 63 65.96
InfoGain-MM-top3 33.37 28.11 24.79 22.02 47.94 36.03 44.73 218.62 15.31 9.77 7.10 5.58 31.24 16.29 28.50 73.66 70 73.75
FES-RAG-top5 40.29 32.11 27.38 24.73 60.43 40.23 58.18 256.36 20.51 14.77 11.30 8.99 39.89 21.23 37.02 110.99 73 74.97
surpasses standard rerankers by optimizing for utility, it remains
inferior to FES-RAG. The superiority of fine-grained retrieval is
evidenced by the substantial performance gap on the MMQA bench-
mark, where FES-RAG-top5 increases the CIDEr score from 264.78
to 336.42, representing a notable 27% relative improvement over the
baseline. This empirical evidence validates our core hypothesis that
document-level retrieval is insufficient for fine-grained reasoning.
By decomposing documents into atomic units, FES-RAG effectively
eliminates contextual clutter, such as irrelevant visual backgrounds
and filler sentences, thereby preventing the noise interference.(iii)
Universality Across Architectures.Crucially, the superiority of
FES-RAG is consistent across all tested backend generators, rang-
ing from the specialized MM-RAIT to general-purpose models like
LLaVA-NeXT and InternVL3.5. This suggests that susceptibility to
contextual noise is an intrinsic characteristic of MLLMs regard-
less of their architecture size, and providing purified, high-density
evidence serves as a universally effective strategy for enhancing
factual grounding.
5.3 Ablation Study (RQ2)
5.3.1 Effectiveness of Knowledge Distillation.To validate the ef-
ficacy of our supervision strategy, we compare identical student
selectors (Jina-Reranker-2B) trained under different objectives. As
illustrated in Figure 3, models trained solely with standard Cross-
Entropy (CE (No Teacher)) rely exclusively on hard-label supervi-
sion, which projects continuous utility scores into discrete binary
targets. This rigid quantization inevitably causes information col-
lapse, stripping away the fine-grained relative ranking information
among candidate fragments.
To isolate the intrinsic benefits of our distillation formulation
from the raw parameter advantage of a massive teacher model, we
introduce a same-size KD setup where the student is supervised bya teacher of identical capacity (Qwen3-VL-2B). Even without a pa-
rameter gap, the Same-Size KD student (CIDEr 242.60) consistently
outperforms the CE-only baseline (CIDEr 223.38). We attribute this
performance leap to the synergistic integration of soft distribution-
level distillation. By distilling the calibrated soft logits from the
teacher, the student captures nuanced utility assessments, enabling
it to distinguish subtle high-quality evidence from hard negatives
more effectively than binary supervision alone.
Naturally, scaling the teacher to the full Qwen3-VL-32B model
(FES-RAG) further elevates the upper bound of this capability,
achieving a peak CIDEr score of 256.36 on the MMQA task.
Figure 3: Impact of Knowledge Distillation. Comparison
among three identical Jina-Reranker-2B student selectors:
trained with hard labels (CE (No Teacher)), distilled from
a same-size teacher (Same-Size KD using Qwen3-VL-2B),
and distilled from the full teacher (FES-RAG using Qwen3-
VL-32B). The results confirm the intrinsic value of soft
distribution-level distillation independent of model scaling.

Purifying Multimodal Retrieval: Fragment-Level Evidence
Selection for RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
5.3.2 Robustness to Candidate Selection (Top-k).We further exam-
ine the scaling behavior of our method with respect to the number of
retrieved segments 𝑘, as presented in Table 2. The results highlight
the exceptional discriminative capability of our distilled student se-
lector, where FES-RAG-top1 alone achieves a CIDEr score of 237.49,
notably surpassing the coarse-grained InfoGain-MM-top3 baseline
of 218.62. This finding underscores that our selector effectively
prioritizes the most critical evidence, enabling superior reasoning
capabilities even with minimal context. Moreover, as 𝑘increases
from 1 to 5, the performance shows a robust upward trajectory. Al-
though expanding the context window in coarse-grained retrieval
also yields performance gains by accumulating more potential evi-
dence, it simultaneously introduces substantial noise, limiting the
reasoning accuracy. In contrast, FES-RAG mitigates this issue by
enforcing high information density at the fragment level. Unlike
coarse-grained approaches that rely on larger context windows
to capture sufficient evidence, our method ensures that each addi-
tional atomic fragment selected by the student selector provides
high-utility information, thereby maximizing the marginal gain of
the retrieval budget while minimizing noise interference.
Table 2: Comparison of Top-k scaling strategies on Qwen2.5-
VL. Best results are in bold.
MethodMMQA Image Captioning
B-4 R-L CIDEr B-4 R-L CIDEr
InfoGain-MM-top1 21.85 44.17 211.24 4.12 26.62 50.00
FES-RAG-top1 22.25 54.62 237.49 7.13 35.00 96.32
InfoGain-MM-top3 22.02 44.73 218.62 5.58 28.50 73.66
FES-RAG-top3 23.43 55.29 247.94 8.62 36.16 107.38
InfoGain-MM-top5 24.57 45.80 239.79 8.56 31.87 103.80
FES-RAG-top5 24.73 58.18 256.36 8.99 37.02 110.99
5.3.3 Correlation between FIG and Final Correctness.To empiri-
cally validate that Fragment Information Gain (FIG) monotonically
aligns with downstream correctness—rather than merely serving as
a proxy for ground-truth log-likelihood—we conduct a controlled
Top-1 injection sanity check.
To rigorously evaluate the final correctness of open-ended gen-
erative responses in the MMQA task, we employ GLM-4.6V [ 12]
as anLLM-as-a-Judgeevaluator. The evaluator assesses whether
the MLLM’s generated response semantically entails the ground
truth, yielding a semantic Accuracy (Acc) score. This metric ef-
fectively captures the true accuracy gain while avoiding the rigid
false-negative penalties associated with exact string matching.
Table 3: Sanity check showing the correlation between FIG
score intervals and final generative correctness (LLM Judge
Accuracy and CIDEr) on the MMQA task (Qwen2.5-VL).
FIG Interval LLM Judge Acc % CIDEr
𝐹𝐼𝐺≤0.0 32 102.45
0.0<𝐹𝐼𝐺≤0.2 65 182.95
𝐹𝐼𝐺>0.2 92 237.49
We force the Qwen2.5-VL generator to condition on exactly one
fragment from a specific FIG interval. As shown in Table 3, thegenerative accuracy strictly aligns with the FIG intervals. When
conditioned on noise fragments ( 𝐹𝐼𝐺≤ 0.0), the model’s accuracy
is restricted to 32% (CIDEr 102.45), confirming these fragments fail
to provide grounding and force the model to rely solely on internal
parametric guesses. For fragments with low to moderate utility
(0.0<𝐹𝐼𝐺≤ 0.2), the accuracy improves to 65%. Conversely, when
conditioned on high-utility fragments ( 𝐹𝐼𝐺> 0.2) successfully
isolated by FES-RAG, the LLM Judge Accuracy surges to an im-
pressive 92%, accompanied by a peak CIDEr score of 237.49. These
results provide empirical evidence that higher FIG scores are associ-
ated with stronger downstream generation correctness, supporting
FIG as an effective supervision signal for fragment-level evidence
selection.
5.4 Modality Analysis (RQ3)
To deconstruct the source of our performance gains, we partition
the test sets into image and text subsets and evaluate the backbone
Qwen2.5-VL across multiple tasks, as shown in Table 4.
The results highlight that FES-RAG’s purification is effective
across all scenarios. On theImage Subset, baseline scores are
relatively high, yet FES-RAG still achieves a significant boost, in-
creasing the MMQA CIDEr score from 427.49 to 439.59. Standard
MLLMs typically perform image downsampling to fit input resolu-
tion constraints, often causing small but critical objects to become
unrecognizable. By feeding explicit ROIs, FES-RAG preserves high-
resolution visual details, preventing resolution dilution and thereby
boosting FactVeri ACC from 72.73% to 75.76%.
The impact is even more pronounced on theText Subset, where
FES-RAG causes MMQA CIDEr scores to more than double (from
46.33 to 106.44) alongside a significant improvement in FactVeri
F1 score (from 65.10 to 66.41). To verify that this massive gain
stems from our preciseRecursive Binary Segmentationrather
than simply reducing the context window length, we introduce
aBrute-force Truncationbaseline. This baseline directly trun-
cates the originally retrieved documents to match the exact to-
ken budget of our selected fragments, entirely bypassing the FIG-
guided selection. As shown in Table 4, while brute-force truncation
yields a minor improvement (e.g., CIDEr 62.18) by casually filtering
some distal noise, it falls massively short of FES-RAG. This dis-
parity confirms that MLLMs are highly sensitive to textual noise.
Whereas coarse-grained or arbitrarily truncated documents obscure
the precise factual evidence within paragraphs of lengthy distrac-
tors, our fragment-level selection effectively isolates the critical
facts, thereby purifying the reasoning context for diverse tasks.
Table 4: Performance breakdown by retrieved modality sub-
sets across MMQA and Fact Verification (Qwen2.5-VL).
Subset MethodMMQA Fact Verification
B-4 R-L CIDEr ACC % F1
ImageInfoGain-MM-top3 42.52 73.78 427.49 72.73 78.21
FES-RAG-top5 45.09 74.57 439.59 75.76 79.38
TextInfoGain-MM-top3 5.24 20.95 46.33 64.71 65.10
Brute-force Truncation 6.15 27.40 62.18 64.71 65.82
FES-RAG-top5 8.07 44.76 106.44 67.65 66.41

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
5.5 Efficiency Analysis (RQ4)
A potential concern with introducing a multi-stage fragment-level
pipeline is the incremental computational overhead. To address this,
we evaluate FES-RAG’s efficiency across space (token consumption)
and time (inference latency).
As illustrated in Figure 4, FES-RAG demonstrates a significant
dual-efficiency advantage. First, in terms of space efficiency (Fig.
4a,b), our framework achieves substantially higher performance
while reducing context length by 20.4% in MMQA and 33.1% in Fact
Verification compared to the coarse-grained InfoGain-MM base-
line. This confirms that fragment-level segmentation successfully
purifies the context, maximizing information density.
Second, this reduction directly translates into a remarkable in-
ference acceleration (Fig. 4c). We decompose the End-to-End (E2E)
latency into four distinct phases: Vector Retrieval, Coarse Rerank-
ing, Segmentation & Selection, and GPU Generation. While the
coarse-grained InfoGain-MM proceeds directly to generation af-
ter reranking, FES-RAG introduces a fine-grained selection stage
(0.58s) to filter redundant information. Crucially, this purification
drastically lightens the computational load on the heavy MLLM gen-
erator. Since the self-attention cost during the prefill stage grows
quadratically with the input context length, reducing the retrieved
context can substantially lower generation latency. By filtering
redundant evidence before generation, FES-RAG reduces the gener-
ation time from 2.07s to 1.34s. The latency saved during generation
(+0.73s) outweighs the additional overhead introduced by the selec-
tor (-0.58s), leading to a lower overall end-to-end latency than the
coarse-grained baseline (2.38s vs. 2.53s). These results suggest that
fragment-level purification can serve as a system-level accelerator,
making FES-RAG suitable for latency-sensitive applications.
Figure 4: Efficiency and Performance Trade-offs. Comparison
of context token consumption (a, b) and end-to-end inference
latency (c).
6 Qualitative Results: Case Study
Figure 5 illustrates the selector’s ability to isolate utility-critical
evidence. InCase 1 (Visual), for the query “wooden item you
can sit on,” it assigns low scores (Label=0) to semantically unin-
formative park scenery while accurately highlighting thebench
(Label=1). This demonstrates preciseobject-level semantic ground-
ingover coarse visual relevance. Similarly, inCase 2 (Textual),
the selector suppresses a keyword-rich but irrelevant distractor
paragraph about “Richard Owen” (Label=0). Instead, it prioritizes
the exact sentence identifying “Georges Buffon” (Label=1), exhibit-
inganswer-bearing semantic precisionrather than relying on term
frequency.
Figure 5: Qualitative analysis of the FES-RAG framework.
The selector effectively filters out semantically related but
irrelevant fragments (Label=0) and prioritizes answer-critical
evidence (Label=1) across both visual and textual modalities.
Across modalities, FES-RAG consistently extracts atomic, answer-
critical fragments. By filtering out semantically adjacent noise, it
provides the MLLM with a compact, high-signal context, ensuring
downstream reasoning is well-grounded and robust to distraction.
7 Conclusion
This paper presents FES-RAG, a framework that advances MRAG
from monolithic document-level reranking to atomic fragment-level
selection. It decomposes retrieved content into sentences and visual
regions, enabling precise removal of redundant or noisy context that
can impair MLLM reasoning. FES-RAG uses Fragment Information
Gain (FIG) to estimate each fragment’s marginal utility, providing a
more informative supervision signal than document-level relevance
labels for training a fine-grained evidence selector. By distilling
this utility assessment into a lightweight student model, FES-RAG
achieves a 27% relative CIDEr improvement on the M2RAG bench-
mark while substantially reducing token consumption. These re-
sults show that fragment-level purification effectively improves
both factual grounding and inference efficiency in MLLMs.

Purifying Multimodal Retrieval: Fragment-Level Evidence
Selection for RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
References
[1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Moham-
madali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soley-
mani Baghshah, and Ehsaneddin Asgari. 2025. Ask in any modality: A compre-
hensive survey on multimodal retrieval-augmented generation.arXiv preprint
arXiv:2502.08826(2025).
[2]Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The
Distracting Effect: Understanding Irrelevant Passages in RAG. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers).
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InProceedings of the 12th International Conference on Learning Representations
(ICLR). Vienna, Austria.
[4]Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng,
Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al .2025. Qwen3-vl
technical report.arXiv preprint arXiv:2511.21631(2025).
[5]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai
Dang, Peng Wang, Shijie Wang, Jun Tang, et al .2025. Qwen2. 5-VL Technical
Report.arXiv preprint arXiv:2502.13923(2025).
[6]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text
embeddings through self-knowledge distillation.arXiv preprint arXiv:2402.03216
4, 5 (2024).
[7]Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare
Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. 2024. The
power of noise: Redefining retrieval for rag systems. InProceedings of the 47th
International ACM SIGIR Conference on Research and Development in Information
Retrieval. 719–729.
[8]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding. In
Proceedings of the 2019 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, Volume 1 (Long and
Short Papers). Minneapolis, Minnesota, 4171–4186.
[9]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards
Retrieval-Augmented Large Language Models. arXiv:2405.06211 https://arxiv.
org/abs/2405.06211
[10] Shutian Gu, Chengkai Huang, Ruoyu Wang, and Lina Yao. 2026. Learning to
Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation.
arXiv preprint arXiv:2602.15724(2026).
[11] Sil Hamilton, Rebecca M. M. Hicke, Matthew Wilkens, and David Mimno. 2025.
Too Long, Didn’t Model: Decomposing LLM Long-Context Understanding With
Novels. arXiv:2505.14925 https://arxiv.org/abs/2505.14925
[12] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao
Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al .2025. Glm-4.5 v and glm-4.1
v-thinking: Towards versatile multimodal reasoning with scalable reinforcement
learning.arXiv preprint arXiv:2507.01006(2025).
[13] Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long Le,
Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna,
and Tomas Pfister. 2024. Found in the middle: Calibrating Positional Atten-
tion Bias Improves Long Context Utilization. InFindings of the Association for
Computational Linguistics: ACL 2024.
[14] Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang,
and Nanyun Peng. 2025. MRAG-Bench: Vision-Centric Evaluation for Retrieval-
Augmented Multimodal Models. (2025).
[15] Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang,
Ryan A Rossi, Branislav Kveton, Dongruo Zhou, et al .2025. Towards agentic
recommender systems in the era of multimodal large language models.arXiv
preprint arXiv:2503.16734(2025).
[16] Chengkai Huang, Yu Xia, Rui Wang, Kaige Xie, Tong Yu, Julian McAuley, and
Lina Yao. 2025. Embedding-informed adaptive retrieval-augmented generation
of large language models. InProceedings of the 31st International Conference on
Computational Linguistics. 1403–1412.
[17] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, et al .
2022. Atlas: few-shot learning with retrieval augmented language models.The
Journal of Machine Learning Research24, 1 (2022), 11912–11954.
[18] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval Aug-
mented Generation. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing. Singapore, 7969–7992.
[19] Shuguang Jiao, Chengkai Huang, Shuhan Qi, Xuan Wang, Yifan Li, and Lina
Yao. 2026. Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented
Generation.arXiv preprint arXiv:2604.00865(2026).
[20] Shuguang Jiao, Xinyu Xiao, Yunfan Wei, Shuhan Qi, Chengkai Huang, Quan Z
Sheng, and Lina Yao. 2026. PruneRAG: Confidence-Guided Query Decomposition
Trees for Efficient Retrieval-Augmented Generation. InProceedings of the ACMWeb Conference 2026. 1923–1934.
[21] Jina AI. 2025. Jina Reranker M0: Multilingual & Multimodal Document Reranker.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
et al.2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
InAdvances in Neural Information Processing Systems (NeurIPS), Vol. 33. Virtual,
9459–9474.
[23] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma,
and Chunyuan Li. 2024. Llava-next-interleave: Tackling multi-image, video, and
3d in large multimodal models.arXiv preprint arXiv:2407.07895(2024).
[24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven C. H. Hoi. 2022. BLIP: Bootstrap-
ping Language-Image Pre-training for Unified Vision-Language Understanding
and Generation. InInternational Conference on Machine Learning, ICML 2022,
17-23 July Maryland, USA (Proceedings of Machine Learning Research, Vol. 162).
PMLR, 12888–12900.
[25] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries.
InText summarization branches out. 74–81.
[26] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
et al .2023. Lost in the Middle: How Language Models Use Long Contexts.
Transactions of the Association for Computational Linguistics (TACL)12 (2023),
157–173.
[27] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing
Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al .2024. Grounding dino: Marry-
ing dino with grounded pre-training for open-set object detection. InEuropean
conference on computer vision. Springer, 38–55.
[28] Zhiqiang Liu, Chengkai Huang, and Yanxia Liu. 2021. Improved knowledge
distillation via adversarial collaboration.arXiv preprint arXiv:2111.14356(2021).
[29] Zhiqiang Liu, Yuhong Li, Chengkai Huang, KunTing Luo, and Yanxia Liu. 2024.
Boosting fine-tuning via conditional online knowledge transfer.Neural Networks
169 (2024), 325–333.
[30] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
and Baining Guo. 2021. Swin transformer: Hierarchical vision transformer us-
ing shifted windows. InProceedings of the IEEE/CVF international conference on
computer vision. 10012–10022.
[31] Zhiqiang Liu, Yanxia Liu, and Chengkai Huang. 2021. Semi-online knowledge
distillation.arXiv preprint arXiv:2111.11747(2021).
[32] Zhenghao Liu, Xingsheng Zhu, Tianshuo Zhou, Xinyi Zhang, Xiaoyuan Yi, Yukun
Yan, Ge Yu, and Maosong Sun. 2025. Benchmarking retrieval-augmented gen-
eration in multi-modal contexts. InProceedings of the 33rd ACM International
Conference on Multimedia. 4817–4826.
[33] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization.
arXiv preprint arXiv:1711.05101(2017).
[34] Haowei Lou, Chengkai Huang, Hye-young Paik, Yongquan Hu, Aaron Quigley,
Wen Hu, and Lina Yao. 2025. SpeechAgent: An End-to-End Mobile Infrastructure
for Speech Impairment Assistance.arXiv preprint arXiv:2510.20113(2025).
[35] Zi-Ao Ma, Tian Lan, Rong-Cheng Tu, Yong Hu, Yu-Shi Zhu, Tong Zhang, Heyan
Huang, Zhijing Wu, and Xian-Ling Mao. 2024. Multi-modal retrieval augmented
multi-modal generation: Datasets, evaluation metrics and strong baselines.arXiv
preprint arXiv:2411.16365(2024).
[36] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. 2025. A survey of multimodal
retrieval-augmented generation.arXiv preprint arXiv:2504.08748(2025).
[37] Matin Mortaheb, Mohammad A. Amir Khojastepour, Srimat T. Chakradhar, and
Sennur Ulukus. 2025. RAG-Check: Evaluating Multimodal Retrieval Augmented
Generation Performance. arXiv:2501.03995 [cs.LG] https://arxiv.org/abs/2501.
03995
[38] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a
method for automatic evaluation of machine translation. InProceedings of the
40th annual meeting of the Association for Computational Linguistics. 311–318.
[39] Jingyuan Qi, Zhiyang Xu, Rulin Shao, Yang Chen, Jin Di, Yu Cheng, Qifan Wang,
and Lifu Huang. 2024. Rora-vlm: Robust retrieval-augmented vision language
models.arXiv preprint arXiv:2410.08876(2024).
[40] J. Ross Quinlan. 1986. Induction of decision trees.Machine learning1, 1 (1986),
81–106.
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models
From Natural Language Supervision. InProceedings of the 38th International Con-
ference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings
of Machine Learning Research, Vol. 139). PMLR, 8748–8763.
[42] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing Retrieval-Augmented Large Language Models with
Iterative Retrieval-Generation Synergy. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2023. Singapore, 9248–9274. https://aclanthology.
org/2023.findings-emnlp.620/
[43] Naoya Sogi, Takashi Shibata, and Makoto Terao. 2024. Object-aware query
perturbation for cross-modal image-text retrieval. InEuropean Conference on
Computer Vision. Springer, 447–464.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
[44] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider:
Consensus-based image description evaluation. InProceedings of the IEEE confer-
ence on computer vision and pattern recognition. 4566–4575.
[45] Chuhan Wang, Xintong Li, Jennifer Yuntong Zhang, Junda Wu, Chengkai Huang,
Lina Yao, Julian McAuley, and Jingbo Shang. 2026. SceneAlign: Aligning Mul-
timodal Reasoning to Scene Graphs in Complex Visual Scenes.arXiv preprint
arXiv:2601.05600(2026).
[46] Hongru Wang, Wenyu Huang, Yang Deng, Rui Wang, Zezhong Wang, Yufei
Wang, Fei Mi, Jeff Z Pan, and Kam-Fai Wong. 2024. Unims-rag: A unified multi-
source retrieval-augmented generation for personalized dialogue systems.arXiv
preprint arXiv:2401.13256(2024).
[47] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei,
Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al .2025. Internvl3. 5: Ad-
vancing open-source multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265(2025).
[48] Xihang Wang, Zihan Wang, Chengkai Huang, Quan Z. Sheng, and Lina Yao. 2026.
MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection
in RAG. arXiv:2604.24564
[49] Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Ling-
tao Mao, Chenyi Lei, Yuqing Ding, and Han Li. 2025. InfoGain-RAG: Boosting
Retrieval-Augmented Generation through Document Information Gain-based
Reranking and Filtering. InProceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing.[50] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang,
and Huaxiu Yao. 2024. RULE: Reliable Multimodal RAG for Factuality in Medical
Vision Language Models.
[51] Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin
Zhong, and Hengxing Cai. 2025. MM-R5: MultiModal Reasoning-Enhanced
ReRanker via Reinforcement Learning for Document Retrieval.arXiv preprint
arXiv:2506.12364(2025).
[52] Minglai Yang, Ethan Huang, Liang Zhang, Mihai Surdeanu, William Yang Wang,
and Liangming Pan. 2025. How is llm reasoning distracted by irrelevant context?
an analysis using a controlled benchmark. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing. 13340–13358.
[53] Zhuolin Yang, Wei Ping, Zihan Liu, Vijay Korthikanti, Weili Nie, De-An Huang,
Linxi Fan, Zhiding Yu, Shiyi Lan, Bo Li, et al .2023. Re-vilm: Retrieval-augmented
visual language model for zero and few-shot image captioning. InFindings of the
Association for Computational Linguistics: EMNLP 2023. 11844–11857.
[54] Yunzhu Zhang, Yu Lu, Tianyi Wang, Fengyun Rao, Yi Yang, and Linchao Zhu.
2025. FlexSelect: Flexible Token Selection for Efficient Long Video Understanding.
arXiv preprint arXiv:2506.00993(2025).
[55] Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. 2024. Vista:
Visualized text embedding for universal multi-modal retrieval.arXiv preprint
arXiv:2406.04292(2024).
# Query Routing for Retrieval-Augmented Language Models

**Authors**: Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu, Fan Wu, Guihai Chen

**Published**: 2025-05-29 03:44:56

**PDF URL**: [http://arxiv.org/pdf/2505.23052v1](http://arxiv.org/pdf/2505.23052v1)

## Abstract
Retrieval-Augmented Generation (RAG) significantly improves the performance
of Large Language Models (LLMs) on knowledge-intensive tasks. However, varying
response quality across LLMs under RAG necessitates intelligent routing
mechanisms, which select the most suitable model for each query from multiple
retrieval-augmented LLMs via a dedicated router model. We observe that external
documents dynamically affect LLMs' ability to answer queries, while existing
routing methods, which rely on static parametric knowledge representations,
exhibit suboptimal performance in RAG scenarios. To address this, we formally
define the new retrieval-augmented LLM routing problem, incorporating the
influence of retrieved documents into the routing framework. We propose
RAGRouter, a RAG-aware routing design, which leverages document embeddings and
RAG capability embeddings with contrastive learning to capture knowledge
representation shifts and enable informed routing decisions. Extensive
experiments on diverse knowledge-intensive tasks and retrieval settings show
that RAGRouter outperforms the best individual LLM by 3.61% on average and
existing routing methods by 3.29%-9.33%. With an extended score-threshold-based
mechanism, it also achieves strong performance-efficiency trade-offs under
low-latency constraints.

## Full Text


<!-- PDF content starts -->

arXiv:2505.23052v1  [cs.CL]  29 May 2025Query Routing for Retrieval-Augmented Language
Models
Jiarui Zhang1Xiangyu Liu2Yong Hu2Chaoyue Niu1∗Fan Wu1Guihai Chen1
1Shanghai Jiao Tong University2WeChat, Tencent Inc
Abstract
Retrieval-Augmented Generation (RAG) significantly improves the performance
of Large Language Models (LLMs) on knowledge-intensive tasks. However, vary-
ing response quality across LLMs under RAG necessitates intelligent routing
mechanisms, which select the most suitable model for each query from multiple
retrieval-augmented LLMs via a dedicated router model. We observe that external
documents dynamically affect LLMs’ ability to answer queries, while existing rout-
ing methods, which rely on static parametric knowledge representations, exhibit
suboptimal performance in RAG scenarios. To address this, we formally define
the new retrieval-augmented LLM routing problem, incorporating the influence
of retrieved documents into the routing framework. We propose RAGRouter, a
RAG-aware routing design, which leverages document embeddings and RAG capa-
bility embeddings with contrastive learning to capture knowledge representation
shifts and enable informed routing decisions. Extensive experiments on diverse
knowledge-intensive tasks and retrieval settings show that RAGRouter outper-
forms the best individual LLM by 3.61% on average and existing routing methods
by 3.29%–9.33%. With an extended score-threshold-based mechanism, it also
achieves strong performance-efficiency trade-offs under low-latency constraints.
1 Introduction
The rapid advancement of large language models (LLMs) has led to an increasingly diverse model
landscape, with significant heterogeneity in parametric knowledge stemming from variations in
training data, architectures, and learning objectives [ 47,37,38,3,4,59,56]. However, LLMs
remain limited by outdated knowledge, hallucinations, and insufficient domain coverage [ 16,24,
31]. Retrieval-Augmented Generation (RAG) [ 29,15] addresses these issues by injecting external
knowledge at inference time, effectively reducing hallucinations and improving performance on
knowledge-intensive tasks [5, 13, 49, 22, 43].
While RAG enhances LLM performance by incorporating external knowledge, different LLMs
exhibit substantial variation in their ability to utilize retrieved content. Prior studies [ 6,11] show
that, given identical documents, LLMs differ in information extraction, integration, and robustness
to noise—reflecting inherent heterogeneity in RAG capabilities stemming from differences in ar-
chitecture, training data, and optimization. Such diversity suggests that combining multiple models
can yield complementary strengths, enabling performance that surpasses any single model. LLM
routing, which pre-selects the most suitable model for each query from a pool of LLMs without
invoking them all, offers an efficient and effective fusion strategy [ 9]. Under dual heterogeneity in
parametric knowledge and RAG capability, intelligent query routing presents a promising direction
for leveraging RAG to achieve superior performance, motivating the central question: How can we
effectively route queries to the most capable LLM under the RAG paradigm?
∗Chaoyue Niu (rvince@sjtu.edu.cn) is the corresponding author.
Preprint. Under review.

Amram.
 Aaron.
Jacob.
 Amram.
RouterWho is the father of Moses?
Amram  In the Book  of Exodus,  Amram  is the 
husband  of Jochebed  and father  of Aaron,  Moses  
and Miriam . Alternative  spellings  of the name  
include  . In addition  to being  married  to ...
Retrieval
Route
Generation
Routing Failed
(w RAG)Routing succeeded
(w/o RAG)
Figure 1: Left: Accuracy of various LLMs on the PopQA task before and after RAG. Right : An
example query where retrieved documents improve Qwen’s response (unanswerable →answerable)
but impair Llama’s (answerable →unanswerable), illustrating how existing routing methods fail
under RAG due to their inability to capture such dynamic shifts.
Current multi-model routing methods primarily match queries to LLMs based on their inherent
parametric knowledge [ 42,21,33,45,40,10,7,35,60,12]. Several approaches [ 7,35,60,12]
construct compact vector representations of LLMs to enable efficient query-model compatibility
estimation. These methods all assume static knowledge representations for non-RAG scenarios.
However, these approaches face critical limitations in RAG settings, as they fail to account for the
dynamic impact of knowledge injection. As illustrated in Figure 1, RAG dramatically shifts the
distribution of response quality across input queries—external documents can reverse a model’s
ability to answer a question, rendering routing strategies designed for non-RAG scenarios obsolete.
The core issue lies in the Static Knowledge Assumption : existing approaches assume fixed LLM
knowledge, ignoring how retrieved content dynamically reshapes their capabilities. In practice,
RAG response quality depends on the interplay between a model’s internal knowledge and external
information. This leads to shortcomings: Missing Doc Interaction —existing methods focus on
queries and model embeddings, overlooking document features and their interaction with models; and
Ignoring RAG Capability —prior work captures only static knowledge differences, neglecting LLMs’
differing ability to leverage documents. These gaps highlight shortcomings of current LLM routing
strategies—they fail to adapt to the dynamic relationship between LLM and external knowledge.
To tackle this issue, we propose RAGRouter , a contrastive learning-based routing framework that
explicitly models knowledge shifts in RAG scenarios. RAGRouter is designed to route queries across
LLMs by modeling key factors that affect post-retrieval performance. At the architecture level ,
RAGRouter incorporates a document encoder and a cross encoder to capture document semantics
and query interactions, thereby addressing missing document interaction, and assigns each LLM a
RAG capability embedding—a learnable vector representing its proficiency in utilizing retrieved con-
tent—to mitigate ignoring RAG capability. However, directly optimizing such a router is challenging
due to inherent variations introduced by retrieval. To address this, at the optimization level , we
employ a contrastive learning objective, where positive and negative samples—i.e., representations of
LLMs that correctly or incorrectly respond to a query—are drawn from both Cross-Setting (between
non-RAG and RAG settings) and Intra-Setting (within each setting). Taking the query representation
as an anchor, the objective encourages alignment between answerable model-query pairs while
pushing apart unanswerable ones. This allows RAGRouter to effectively model retrieval-induced
behavior shifts, moving beyond the static knowledge assumption.
We evaluate RAGRouter on a suite of knowledge-intensive tasks [ 34,36,27,2,25] and retrieval
settings. Experimental results show that RAGRouter surpasses the performance of the best individual
LLM, highlighting its ability to leverage the complementary strengths of multiple models in retrieval-
augmented scenarios. Furthermore, RAGRouter substantially outperforms existing non-RAG-aware
routing methods, validating the effectiveness of modeling retrieval-induced knowledge shifts.
Our main contributions are summarized as follows: (i) To the best of our knowledge, this is the
first work exploring LLM routing in the RAG setting; (ii) We propose RAGRouter, a contrastive
learning-based routing mechanism that is aware of knowledge shifts, incorporating RAG capability
and document-aware representations to effectively address the failure modes of existing routing
strategies in RAG; (iii) We validate the effectiveness of our method on five knowledge-intensive
2

tasks and retrieval settings, using candidate LLMs from diverse families such as the Qwen and
LLaMA series, and covering a wide range of LLM scales from 0.5B to 72B. Results show that
RAGRouter outperforms the best individual LLM by 3.61% on average and existing non-RAG-aware
routing methods by 3.29%–9.33%; (iv) We apply an extended score-threshold-based mechanism
to RAGRouter, and results show that its accuracy–latency curve generally lies above those of all
baselines, indicating superior performance-efficiency trade-offs under low-latency constraints.
2 Related Work
Retrieval-Augmented Generation. RAG enhances language models by integrating retrieved in-
formation from external databases [ 29,15]. It typically follows a round of retrieval and sequential
generation pipelines, where documents are retrieved based on the input query and concatenated with
it for generation. Prior work has improved RAG by optimizing retrieval components [ 58,54,44] or
enhancing the generator’s ability to utilize retrieved content [ 23,52,50]. Recent studies highlight
heterogeneous LM capabilities in processing external information, both in utilizing retrieved content
[30,6] and tolerating retrieval noise [ 11,41]. These heterogeneous capabilities reveal optimiza-
tion opportunities for ensemble approaches that strategically leverage multiple LLMs within RAG
scenarios. In this work, we study the routing problem under the RAG setting.
LLM Routing. Existing LLM routing approaches [ 7,10,12,35,60,33] primarily focus on non-RAG
settings, where routing relies solely on the input query and each model’s parametric knowledge,
without incorporating external retrieved documents. For example, RouterDC [ 7] uses dual contrastive
learning to model query-model compatibility, while EmbedLLM [ 60] and RouteLLM [ 35] apply
matrix factorization to learn compact model embeddings for scalable routing. GraphRouter [ 12]
constructs a heterogeneous graph with nodes for tasks, queries, and LLMs, and encodes their
interactions as edges to capture contextual alignment between query needs and model capabilities.
However, in RAG scenarios, retrieved documents induce dynamic shifts in model knowledge, which
existing methods overlook. In contrast, our proposed RAGRouter models both the documents and
LLMs’ RAG capabilities, enabling more effective routing under retrieval-augmented settings.
3 Problem Formulation
RAG enhances LLMs by integrating external knowledge through a two-stage process: given a query q,
the retriever Ret(D, q)selects relevant documents dfrom an external corpus D, and the model M(q, d)
generates a response ybased on both the query qand the documents d, i.e., y=M(q,Ret(D, q)).
We formulate a LLM routing problem under RAG setting. Let M={M1, . . . , M N}be a set of
candidate LLMs. A routing policy R:Q × D → { 1, . . . , N }selects the most suitable model
MR(q,d)for each input pair (q, d). To evaluate response quality, we define an oracle scoring function
σ(Mi, q, d)∈ {0,1}, where σ(Mi, q, d) = 1 if the response from Migiven qanddmatches the
reference answer y∗. Importantly, using a fixed model can be suboptimal, as different LLMs excel on
different query-document pairs. The objective is to maximize the expected routing performance:
max
REq∼Q
σ(MR(q,d), q, d)
(1)
Notably, when no external documents are available (i.e., d=∅), the LLM routing problem under
RAG setting naturally degenerates into the conventional LLM routing problem. In this setting, the
routing policy simplifies to R:Q → { 1, . . . , N }, and the oracle scoring function becomes σ(Mi, q),
which assesses the response based solely on the query. The objective becomes:
max
REq∼Q
σ(MR(q), q)
(2)
Thus, the conventional routing problem can be seen as a special case of LLM routing under the RAG
setting, corresponding to the boundary condition where d=∅.
4 RAGRouter
4.1 Routing Model Architecture Design
We establish a conceptual framework by constructing an intuitive explanation of knowledge rep-
resentation and LLM-query matching under RAG and non-RAG settings. In non-RAG settings
3

[7,60], each LLM is typically associated with a compact knowledge representation vector vk∈Rdim,
which implicitly reflects its parametric knowledge; and meanwhile, a query is encoded as vq∈Rdim,
representing the knowledge needed to answer it. A proxy metric, like similarity sim(vq, vk)is then
used to gauge the LLM’s ability to respond, guiding non-RAG routing process.
However, in RAG settings, the LLM is augmented with retrieved documents that provide non-
parametric knowledge. This additional information influences the model’s response generation,
rendering the original knowledge representation vkinsufficient. The effective knowledge of the LLM
shifts due to the integration of external information, resulting in a new representation:
v′
k=vk+vf (3)
where vfis the fused knowledge representation derived from the documents. Consequently, in RAG
scenarios, the similarity between the query and the updated knowledge representation sim(vq, v′
k)
should serve as the new routing criterion, as it more accurately reflects the model’s ability to respond.
Muti -Head
Attention
Query
EncoderDocument
EncoderCross
EncoderLLM Knowledge 
Embedding LayerRAG Capability 
Embedding Layer
Query
…
LLMs
 Doc… ……
…
(a) Input Embedding(b) Knowledge Update (c) Similarity -Based Routing
⊙
+
𝑣𝑑 𝑣𝑞𝑣𝑐 𝑣𝑟 𝑣𝑘𝑣′𝑘
𝑣𝑓similarity
Figure 2: The inference pipeline of RAGRouter:
(a) Encode query, document, cross interaction,
LLM knowledge, and RAG capability; (b) Fuse
RAG capability, document, and cross embeddings
to update knowledge representation; (c) Route
based on similarity with the query embedding.RAGRouter is designed with this insight in mind
and explicitly models the fused knowledge vfto
dynamically update the LLM’s knowledge rep-
resentation. We identify three core factors that
influence vf: (1) the non-parametric knowledge
provided by the documents; (2) the LLM’s ability
to process external information, including knowl-
edge extraction and robustness to noise; and (3)
the query’s role in guiding knowledge retrieval.
Based on these, RAGRouter consists of the fol-
lowing modules, with its architecture illustrated
in Figure 2.
Representing Parametric Knowledge. To ob-
tain the original parametric knowledge represen-
tation vk, we introduce the LLM Knowledge Em-
bedding Layer ϕK, which takes the LLM ID M
and outputs vk=ϕK(M), capturing inter-model
variability in parametric knowledge. For query
representation, we employ a Query Encoder ϕQ,
which encodes the query qasvq=ϕQ(q).
Representing RAG-Aware Factors. To com-
pute the fused knowledge vf, RAGRouter inte-
grates signals from three perspectives. First, a Document Encoder ϕDis introduced to represent the
non-parametric knowledge in a document d, producing vd=ϕD(d). In practice, the document and
query encoders share parameters to ensure consistency in the embedding space. Second, to capture
the model’s capability in utilizing retrieved information, we design the RAG Capability Embedding
Layer ϕR, which maps Mtovr=ϕR(M), reflecting the LLM’s intrinsic ability to benefit from
RAG inputs. Third, to represent the alignment between the query and document, a Cross Encoder ϕC
takes the query-document pair (d, q)and outputs an interaction representation vc=ϕC(d, q).
Representation Update for Similarity-Based Routing. The fused knowledge representation vfis
then derived via a multi-head attention mechanism that integrates these signals:
vf=Attention (vr, vd, vc) (4)
With the fused knowledge computed, the RAG-aware knowledge representation of the model becomes
v′
k=vk+vf. The final routing decision is based on the similarity between the query and the updated
knowledge representations of candidate models:
R(q, d) = arg max
i∈{1,...,N}{sim(vq, v′
ki)} (5)
This formulation allows the routing policy to explicitly account for knowledge shifts introduced by
document retrieval, thus maintaining accurate assessment of each LLM’s ability in RAG settings.
4.2 Optimization
In RAG settings, the incorporation of retrieved documents often leads to significant changes in LLM
answerability—some LLMs become able to answer queries they previously could not, while others
4

Who plays Mary 
Jane in spiderman 2?
Non -RAG SettingRAG Setting
Cross -Setting Contrast
Intra -Setting Contrast
Pull Close
Push Away
(a) Construct Positive and Negative Samples for One Query (b) Contrastive LearningFigure 3: (a) CSC constructs positive and negative samples under different settings based on response
quality (e.g., Llama w/ RAG ( ✓) vs. Llama w/o RAG ( ✗)), while ISC constructs them under the same
setting (e.g., Llama w/ RAG ( ✓) vs. Qwen w/ RAG ( ✗)); (b) By combining CSC and ISC, contrastive
learning pulls positive samples closer to the query representation and pushes negative ones away.
fail after retrieval. These shifts in answerability, effectively label transitions, reflect corresponding
changes in the model’s knowledge representation. Such transitions naturally yield structured positive
and negative pairs across different knowledge states. This setting aligns well with the principles
of contrastive learning [ 8,17], which is particularly well-suited for capturing and optimizing the
knowledge representation shifts induced by external knowledge injection in RAGRouter.
To this end, we design the Cross-Setting Contrast (CSC) mechanism to model representation
differences between the non-RAG and RAG settings, and introduce the Intra-Setting Contrast
(ISC) mechanism to model representation differences within the same setting. As shown in Figure 3,
using the query representation vqas the anchor, CSC constructs positive and negative samples by
selecting knowledge representations with different response qualities from the non-RAG and RAG
settings (blue arrows). ISC, on the other hand, selects positive and negative samples from models
with different response qualities within the same setting (orange arrows). This enables CSC to help
RAGRouter distinguish between different knowledge transfer patterns induced by documents, while
ISC enhances the model’s discriminative ability across LLMs within the same setting.
Combining CSC and ISC, we construct a comprehensive set of positive and negative samples to train
the RAGRouter. For a given query q, we define the positive and negative sets as follows:(
V+={vki|σ(Mi, q) = 1} ∪ {v′
kj|σ(Mj, d, q) = 1}
V−={vki|σ(Mi, q) = 0} ∪ {v′
kj|σ(Mj, d, q) = 0}(6)
The corresponding contrastive loss is defined as:
LCT(q) =X
vk+∈V+−logexp( sim(vq, vk+)/τ)
exp( sim(vq, vk+)/τ) +P
vk−∈V−exp( sim(vq, vk−)/τ)(7)
where τis a temperature hyperparameter. This loss encourages the query embedding vqto be closer
to positive samples and further from negative ones, enabling the learning of representations that
are sensitive to both knowledge shifts and model heterogeneity. When retrieved documents alter a
model’s response ability—e.g., from unanswerable to answerable—the mechanism captures these
dynamic transitions, enhancing routing accuracy and knowledge adaptability in RAGRouter.
To further enhance LLM discrimination, we introduce a binary classification loss. For the original
model M, define sM,q=Sigmoid (sim(vk, vq)); for the RAG-enhanced model M′, define sM′,q=
Sigmoid (sim(v′
k, vq)). LetyM,q=σ(M, q)andyM′,q=σ(M, d, q )be the ground-truth labels. The
classification loss is:
LCLS(q) =−X
M∈M∪M′[yM,qlogsM,q+ (1−yM,q) log(1 −sM,q)] (8)
The total loss is the weighted sum of the contrastive loss and classification loss, with λ > 0as a
balancing hyperparameter:
L(q) =LCT(q) +λLCLS(q) (9)
4.3 Latency-Aware Extended Design
While RAGRouter does not explicitly model LLM’s latency, it outputs a relevance score for each can-
didate LLM given a query, which can be exploited to support flexible trade-offs between performance
5

and efficiency. To this end, we introduce a score-threshold-based routing mechanism. Concretely, we
pre-sort the Navailable LLMs as [M1, M 2, . . . , M N]based on their prior efficiency profiles, such as
smaller parameter sizes and lower latency—meaning that M1is the most efficient and MNthe least.
Given a query, suppose Mireceives the highest predicted score from RAGRouter (i.e., it is the
performance-optimal model). Instead of routing directly to Mi, we traverse the list from M1toMi
and select the first LLM Mjwhose score satisfies sMi,q−sMj,q≤θ, where θis a user-defined score
margin threshold. This mechanism sacrifices a small amount of accuracy for significantly improved
efficiency, making RAGRouter adaptable to latency-constrained or resource-limited scenarios.
5 Experiments
5.1 Experimental Setup
Datasets. We select queries from five different knowledge-intensive tasks: (i) PopQA [ 34] is an
open-domain question-answering benchmark covering diverse factual topics from broad knowledge
domains; (ii) MedMCQA [ 36] is a multiple-choice benchmark focused on biomedical knowledge
and clinical reasoning; (iii) Natural Questions (NQ) [ 27] is an open-domain benchmark based on
real-world search queries requiring span-level answer retrieval from Wikipedia; (iv) WebQuestions
(WebQ) [ 2] is a knowledge base-driven benchmark grounded in Freebase relations, designed to
evaluate entity-centric factual reasoning; and (v) TriviaQA (TQA) [ 25] is an open-domain benchmark
centered on factoid-style questions sourced from trivia enthusiasts and web documents. Following
[44], we adopt Cover Exact Match as the evaluation metric for PopQA, NQ, WebQ, and TriviaQA.
Further data processing details and dataset statistics are summarized in Appendix A.2.
Table 1: Statistics of different LLMs and
their latency.
LLM Params (B) Latency (ms)
Qwen2.5-0.5B-Instruct 0.494 24.54
Llama-3.2-1B-Instruct 1.240 20.47
Qwen2.5-1.5B-Instruct 1.500 24.79
gemma-2-2b-it 2.614 31.80
Llama-3.2-3B-Instruct 3.213 81.82
Qwen2.5-3B-Instruct 3.000 24.39
Yi-1.5-6B-Chat 6.061 142.67
Qwen2.5-7B-Instruct 7.616 80.83
Ministral-8B-Instruct-2410 8.020 26.13
Meta-Llama-3.1-8B-Instruct 8.030 177.37
Yi-1.5-9B-Chat 8.829 199.61
Qwen2.5-14B-Instruct 14.770 175.42
Qwen2.5-32B-Instruct 32.764 156.26
Qwen2.5-72B-Instruct 72.706 1610.00
Llama-3.3-70B-Instruct 70.554 1970.00Candidate LLMs. We selected 15 mainstream LLMs
[55,14,46,57,1] with parameter size ranging from
0.5B to 72B. Comprehensive statistics on model scales
and latency2are presented in Table 1, and implemen-
tation details are provided in Appendix A.1.
Retrieval Settings. Following [ 44], we adopt both
local and online retrieval strategies for PopQA and
MedMCQA to reflect realistic RAG scenarios. Local
retrieval uses the 2018 English Wikipedia dump [ 26]
with BGE-large-en-v1.5 [ 53] as the dense retriever. On-
line retrieval leverages the DuckDuckGo Web Search
API3to access up-to-date external content. For NQ,
WebQ, and TriviaQA, we follow [ 11] and construct re-
trieval contexts from Wikipedia passages augmented
with synthetic noise (e.g., irrelevant distractors, coun-
terfactual noise) to simulate imperfect retrieval. This
setting enables evaluate the effectiveness of the routing model under noisy conditions.
Baselines. We compare RAGRouter against a range of baselines. Existing routing methods were
not RAG-aware and exploited only query-LLM compatibility, ignoring the impact of retrieval
augmentation. This series of baselines include Prompt LLM [12], which employs GPT-4o4for
model selection via meta-prompts; GraphRouter [12], which models queries, tasks, and LLMs in
a heterogeneous graph; RouterDC [7], which aligns query-LLM embeddings through contrastive
learning; KNN Router , which [ 21] relies on historical performance of similar queries; and Matrix
Factorization (MF) [35,60], which reconstructs LLM correctness patterns via low-rank latent spaces.
We also introduce some rule-based routing methods, including a Single Fixed LLM for all queries;
Oracle Single Best that ideally selects the best-performing single LLM per dataset; Random LLM
assignment; Weighted routing to different LLMs according to RAGRouter’s empirical probability
distribution. To show the ideal upper bound of routing performance, we introduce the Oracle baseline
by routing each query to its optimal LLM using ground-truth performance data.
2Latency refers to the average time taken by an LLM to complete a single query, including both inference
time and potential network delays.
3https://duckduckgo.com
4https://platform.openai.com/docs/models/gpt-4o
6

Table 2: Performance comparison of RAGRouter with rule-based and non-RAG-aware baselines
across different knowledge-intensive tasks and retrieval settings. Testing accuracy (%) is reported.
"Ret." indicates whether the method is retrieval-aware ( ✓) or not ( ✗). The best results are shown in
bold , and the second-best are underlined .
PopQA MedMCQAMethod Ret.
Local Online Local OnlineNQ WQ TQA Avg
Qwen2.5-0.5B-Instruct - 45.19 51.11 25.93 34.81 27.08 38.75 39.17 37.43
Llama-3.2-1B-Instruct - 39.26 46.67 17.78 36.67 25.42 31.67 43.33 34.40
Qwen2.5-1.5B-Instruct - 44.44 48.89 30.00 42.59 30.00 35.42 46.67 39.72
gemma-2-2b-it - 41.11 50.74 20.37 37.04 22.92 27.92 40.83 34.42
Llama-3.2-3B-Instruct - 45.56 51.48 27.41 44.81 36.67 45.42 65.00 45.19
Qwen2.5-3B-Instruct - 41.11 47.41 40.37 49.26 30.42 36.25 53.33 42.59
Yi-1.5-6B-Chat - 46.67 51.48 31.11 40.00 35.83 43.33 56.67 43.58
Qwen2.5-7B-Instruct - 42.96 48.15 35.93 43.33 29.58 35.42 55.00 41.48
Ministral-8B-Instruct-2410 - 41.48 46.30 50.74 62.22 38.33 42.08 63.75 49.27
Meta-Llama-3.1-8B-Instruct - 46.67 51.85 41.85 52.59 39.58 46.25 69.58 49.77
Yi-1.5-9B-Chat - 46.67 52.59 50.74 57.78 38.33 47.08 58.75 50.28
Qwen2.5-14B-Instruct - 46.30 50.00 57.04 64.07 42.92 47.08 72.50 54.27
Qwen2.5-32B-Instruct - 45.93 48.52 43.33 49.63 44.58 50.42 80.42 51.83
Qwen2.5-72B-Instruct - 44.81 47.78 67.04 70.00 40.00 48.75 79.17 56.79
Llama-3.3-70B-Instruct - 46.30 50.37 68.89 70.37 51.67 50.42 87.92 60.85
Oracle Single Best - 46.67 52.59 68.89 70.37 51.67 50.42 87.92 61.22
Random - 44.30 49.56 40.57 50.35 35.56 41.75 60.81 46.13
Weighted - 46.35 50.53 68.31 70.18 46.87 48.39 86.09 59.53
Prompt LLM [12] ✗ 46.67 51.48 61.85 65.93 39.58 49.17 71.25 55.13
GraphRouter [12] ✗ 47.41 51.48 68.89 70.37 51.67 50.42 87.92 61.17
RouterDC [7] ✗ 44.81 50.37 67.04 68.89 40.00 48.33 77.50 56.71
KNN Router [21] ✗ 46.67 52.22 68.15 71.48 52.08 46.25 86.25 60.44
MF [35, 60] ✗ 46.30 52.59 68.89 71.48 49.17 50.42 82.92 60.25
RAGRouter (Ours) ✓ 48.52 52.59 71.48 74.44 56.67 56.67 90.83 64.46
Oracle - 54.44 57.41 91.85 90.37 69.17 77.92 96.25 76.77
Implementation Details. For the RAGRouter architecture, we use all-mpnet-base-v2 [ 39] as the
encoder for both queries and documents, and ms-marco-MiniLM-L12-v2 [ 48] as the cross-encoder,
resulting in a total parameter size of approximately 136M. Both the knowledge representation vector
and the RAG capability vector are set to a dimensionality of 768. To mitigate overfitting, all but the
last two transformer layers in the query/document encoder and the cross-encoder are frozen during
training. The classification loss weight λis set to 2.0, and the contrastive learning temperature τto
0.2. The router is optimized using AdamW [ 32] with a learning rate of 5e-5, batch size of 64, for 10
epochs. All experiments are conducted on a single NVIDIA RTX 4090D GPU.
5.2 Main Results
Comparison with Single Non-Routed LLMs. Table 2 presents the test accuracy across a variety of
knowledge-intensive tasks and retrieval settings. RAGRouter consistently achieves the highest perfor-
mance, with an average accuracy of 64.46%. It surpasses the best-performing single RAG-enabled
LLM, LLaMA-3.3-70B-Instruct (60.85%), by +3.61%, demonstrating the effectiveness of routing
in RAG. Notably, RAGRouter also outperforms the Oracle Single Best baseline (61.22%)—which
selects the optimal single model for each dataset—by +3.24%, indicating that routing enables the
integration of multiple RAG-enhanced LLMs to achieve performance beyond any individual model.
Comparison with Non-RAG-Aware and Other Baselines. RAGRouter significantly outperforms
all non-retrieval-aware routing baselines, including GraphRouter (61.17%, +3.29%), MF (60.25%,
+4.21%), KNN Router (60.44%, +4.02%), and RouterDC (56.71%, +7.75%). These results underscore
the limitations of methods that do not explicitly model the interaction between LLMs and retrieved
knowledge. Without capturing retrieval-induced capability shifts, such approaches struggle to make
effective routing decisions in RAG. RAGRouter also surpasses rule-based strategies such as Random
7

Table 3: Area (%), Peak Accuracy (PA, %), and Latency Gap-to-Match (G, s) for RAGRouter and
baselines using score-threshold-based routing on MedMCQA (Local/Online) and TriviaQA. "–"
indicates failure to match the best single-LLM performance; bold denotes the best result.
MedMCQA (Local) MedMCQA (Online) TQAMethod
Area ↑ PA↑ G (s) ↑Area ↑ PA↑ G (s) ↑Area ↑ PA↑ G (s) ↑
RouterDC 55.87 62.22 - 57.77 65.56 - 70.99 75.83 -
MF 46.42 59.63 0.10 54.85 65.56 0.45 66.84 80.00 -
GraphRouter 47.96 59.30 0.18 57.82 64.67 0.32 65.42 73.15 0.01
KNN Router 52.26 62.30 - 60.18 66.10 0.33 72.61 81.65 -
RAGRouter 57.12 62.59 0.24 63.12 67.78 0.51 73.78 87.50 0.76
0 500 1000 1500 2000
Latency per Query (ms)20406080Testing Accuracy (%)
MedMCQA (Local)
RAGRouter
0 500 1000 1500 2000
Latency per Query (ms)405060708090Testing Accuracy (%)
MedMCQA (Online)
RAGRouter
0 500 1000 1500 2000
Latency per Query (ms)405060708090Testing Accuracy (%)
TriviaQA
RAGRouterQwen2.5-0.5B-Instruct
Llama-3.2-1B-Instruct
Qwen2.5-1.5B-Instruct
gemma-2-2b-itLlama-3.2-3B-Instruct
Qwen2.5-3B-Instruct
Yi-1.5-6B-Chat
Qwen2.5-7B-InstructMinistral-8B-Instruct-2410
Meta-Llama-3.1-8B-Instruct
Yi-1.5-9B-Chat
Qwen2.5-14B-InstructQwen2.5-32B-Instruct
Qwen2.5-72B-Instruct
Llama-3.3-70B-Instruct
RandomWeighted
Prompt LLM
Oracle
EmbedLLMKNN Router
RouterDC
GraphRouter
RAGRouter (Ours)
Figure 4: Accuracy–latency curves on MedMCQA (Local), MedMCQA (Online), and TriviaQA.
(46.13%, +18.33%) and Weighted (59.53%, +4.93%). Together, these findings support our core claim:
modeling RAG-specific capabilities and knowledge shift is essential for accurate LLM routing.
5.3 Extended Results on Latency-Aware Routing
Metrics. Three metrics are used for evaluate latency-aware routing: Area , which measures the
proportion of the area under the accuracy–latency curve within 1 second, reflecting overall time-
efficiency; Peak Acc , the highest accuracy achieved within 1 second; and Latency Gap-to-Match ,
defined as the latency of the best-performing single LLM minus the minimum latency required by a
method to match its accuracy, indicating the efficiency margin obtained through routing.
Quantitative Results and Visualization. We apply the score-threshold-based routing mechanism to
RAGRouter and baselines across all tasks. Results on MedMCQA (Local/Online) and TriviaQA are
reported in Table 3 and Figure 4, with supplementary results in Appendix D. RAGRouter consistently
achieves the highest Area and Peak Accuracy across all tasks. Specifically, it outperforms baselines
by 4.86%–10.7%, 2.94%–8.27%, and 1.17%–8.36% in Area, and by 0.29%–3.29%, 1.68%–3.11%,
and 5.85%–14.35% in Peak Accuracy on MedMCQA (Local/Online) and TriviaQA, respectively. As
shown in Figure 4, its accuracy–latency curve consistently dominates those of baselines, indicating
superior accuracy under low-latency constraints. RAGRouter also achieves the largest positive
Latency Gap-to-Match margins, demonstrating it can match the accuracy of the best single LLM with
substantially lower latency. These results highlight RAGRouter’s ability to exploit the optimization
space enabled by retrieval augmentation, efficiently leveraging smaller, faster models to achieve the
performance of larger ones without incurring unnecessary latency.
5.4 Sensitivity Analysis and Ablation Study
Effects of RAGRouter Architecture. We conduct an ablation study to validate the effectiveness
of the Cross Encoder in RAGRouter. As shown in Table 4, removing the Cross Encoder reduces
performance by 0.98%, highlighting the importance of query-document interactions in deriving fused
knowledge representations. We also examine the impact of the dimensionality of both knowledge rep-
8

Table 4: Ablation study of Cross Encoder and contrastive learning.
PopQA MedMCQA Cross
EncoderISC CSC
Local Online Local OnlineNQ WQ TQA Avg ∆
✓ ✓ ✓ 48.52 52.59 71.48 74.44 56.67 56.67 90.83 64.46 0.00
✗ ✓ ✓ 47.78 52.22 70.74 74.44 55.42 55.00 88.75 63.48 -0.98
✓ ✓ ✗ 48.15 52.22 71.11 73.33 53.75 56.25 89.58 63.49 -0.97
✓ ✗ ✓ 48.52 51.85 71.48 74.07 55.00 56.25 89.58 63.82 -0.64
✓ ✗ ✗ 47.78 51.48 69.26 70.74 53.75 53.75 89.17 62.28 -2.18
Table 5: Effects of different candidate LLMs sets.
PopQA MedMCQACandidate Set Method
Local Online Local OnlineNQ WQ TQA Avg ∆
SmallOracle Single Best 45.56 51.48 40.37 49.26 36.67 45.42 65.00 47.68 0.00
RAGRouter 45.56 51.85 40.37 51.48 36.67 47.92 65.00 48.41 +0.73
LargeOracle Single Best 46.30 50.37 68.89 70.37 51.67 50.42 87.92 60.85 0.00
RAGRouter 47.41 50.74 71.11 73.33 54.58 53.33 90.42 62.99 +2.14
Small & LargeOracle Single Best 46.30 51.48 68.89 70.37 51.67 50.42 87.92 61.01 0.00
RAGRouter 48.15 52.22 71.48 73.33 53.33 55.00 89.58 63.30 +2.29
resentation and RAG capability vectors. We observe that performance improves as the dimensionality
increases, peaking at 768, after which it declines. Detailed results are provided in Appendix E.
Effects of λ.We investigate the impact of the classification loss weight λin the overall loss function
(Eq. 9) on test accuracy. We observe that combining contrastive and classification losses yields better
performance than using contrastive loss alone (i.e., λ= 0, achieving only 63.76% on average). As λ
increases, accuracy improves, peaking at λ= 2with 64.46% on average, before experiencing a slight
decline. Based on this, we set λ= 2in all experiments. Detailed results are provided in Appendix E.
Effects of Positive and Negative Sample Selection in Contrastive Learning. To assess the
effectiveness of contrastive learning, we perform an ablation study on the two positive–negative
sample construction strategies used in our method. As shown in Table 4, removing either Intra-Setting
Contrast or Cross-Setting Contrast results in performance drops of 0.64% and 0.97%, respectively.
When both components are removed—effectively disabling contrastive learning—accuracy decreases
by 2.18%. These findings underscore the importance of contrastive learning in capturing knowledge
representation shifts and LLM heterogeneity, which is essential for effective routing in RAG settings.
Effects of Different Candidate LLMs Sets. We investigate how the composition of candidate LLMs
affects routing performance. To this end, we form two subsets of models—Small ( ≤3B parameters)
and Large ( ≥32B)—and evaluate three configurations: Small only, Large only, and a heterogeneous
set combining both. As shown in Table 5, RAGRouter consistently outperforms the Oracle Single
Best in all settings, demonstrating its ability to coordinate models and achieve cumulative gains.
Two key insights emerge. First, the routing upper bound is largely determined by model strength:
the Large set yields a significantly higher Oracle Single Best average (60.85%) than the Small set
(47.68%), with RAGRouter following the same trend (62.99% vs. 48.41%). Second, combining
heterogeneous models yields the best performance: the Small & Large setting achieves the highest
RAGRouter average (63.30%) and the largest gain over its Oracle Single Best (+2.29%), suggesting
that model diversity improves complementarity and enables more effective routing.
6 Conclusion
In this paper, we have studied the problem of LLM routing in Retrieval-Augmented Generation
(RAG) for the first time and propose RAGRouter, the first RAG-aware routing method. By leveraging
contrastive learning, RAGRouter captures knowledge representation shifts induced by external
documents, enabling effective routing decisions. Experiments on diverse knowledge-intensive tasks
demonstrate that RAGRouter outperforms existing non-RAG-aware methods and achieves strong
performance-efficiency trade-offs under low-latency constraints.
9

References
[1]Q Jiang Albert, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, and Devendra Singh
Chaplot. Mistral 7b. arXiv , 2023.
[2]Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase
from question-answer pairs. In Proceedings of the 2013 conference on empirical methods in
natural language processing , pages 1533–1544, 2013.
[3]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in neural information processing systems , 33:1877–1901, 2020.
[4]Sébastien Bubeck, Varun Chadrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece
Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general
intelligence: Early experiments with gpt-4, 2023.
[5]Deng Cai, Yan Wang, Lemao Liu, and Shuming Shi. Recent advances in retrieval-augmented
text generation. In Proceedings of the 45th international ACM SIGIR conference on research
and development in information retrieval , pages 3417–3419, 2022.
[6]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language mod-
els in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 38, pages 17754–17762, 2024.
[7]Shuhao Chen, Weisen Jiang, Baijiong Lin, James T Kwok, and Yu Zhang. Routerdc: Query-
based router by dual contrastive learning for assembling large language models. arXiv preprint
arXiv:2409.19886 , 2024.
[8]Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework
for contrastive learning of visual representations. In International conference on machine
learning , pages 1597–1607. PmLR, 2020.
[9]Zhijun Chen, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun, Yuankai Luo, Qianren Mao,
Dingqi Yang, Hailong Sun, and Philip S Yu. Harnessing multiple large language models: A
survey on llm ensemble. arXiv preprint arXiv:2502.18036 , 2025.
[10] Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ruhle,
Laks VS Lakshmanan, and Ahmed Hassan Awadallah. Hybrid llm: Cost-efficient and quality-
aware query routing. arXiv preprint arXiv:2404.14618 , 2024.
[11] Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. Enhancing
noise robustness of retrieval-augmented language models with adaptive adversarial training.
arXiv preprint arXiv:2405.20978 , 2024.
[12] Tao Feng, Yanzhen Shen, and Jiaxuan You. Graphrouter: A graph-based router for llm selections.
InThe Thirteenth International Conference on Learning Representations , 2024.
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models:
A survey. arXiv preprint arXiv:2312.10997 , 2, 2023.
[14] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama
3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
[15] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval
augmented language model pre-training. In International conference on machine learning ,
pages 3929–3938. PMLR, 2020.
[16] Hangfeng He, Hongming Zhang, and Dan Roth. Rethinking with retrieval: Faithful large
language model inference. arXiv preprint arXiv:2301.00303 , 2022.
10

[17] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for
unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition , pages 9729–9738, 2020.
[18] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.
arXiv preprint arXiv:1503.02531 , 2015.
[19] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe,
Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning
for nlp. In International conference on machine learning , pages 2790–2799. PMLR, 2019.
[20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR ,
1(2):3, 2022.
[21] Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath,
Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm
routing system. arXiv preprint arXiv:2403.12031 , 2024.
[22] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering. arXiv preprint arXiv:2007.01282 , 2020.
[23] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick,
Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Few-shot learning
with retrieval augmented language models. arXiv preprint arXiv:2208.03299 , 1(2):4, 2022.
[24] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.
ACM computing surveys , 55(12):1–38, 2023.
[25] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large
scale distantly supervised challenge dataset for reading comprehension. arXiv preprint
arXiv:1705.03551 , 2017.
[26] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InEMNLP (1) , pages 6769–6781, 2020.
[27] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a
benchmark for question answering research. Transactions of the Association for Computational
Linguistics , 7:453–466, 2019.
[28] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large lan-
guage model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium
on Operating Systems Principles , 2023.
[29] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 33:9459–9474, 2020.
[30] Kuan Li, Liwen Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Shuai Wang, and Minhao Cheng.
Lara: Benchmarking retrieval-augmented generation and long-context llms-no silver bullet for
lc or rag routing. arXiv preprint arXiv:2502.09977 , 2025.
[31] Xianzhi Li, Samuel Chan, Xiaodan Zhu, Yulong Pei, Zhiqiang Ma, Xiaomo Liu, and Sameena
Shah. Are chatgpt and gpt-4 general-purpose solvers for financial text analytics? a study on
several typical tasks. arXiv preprint arXiv:2305.05862 , 2023.
[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint
arXiv:1711.05101 , 2017.
11

[33] Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren
Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models. arXiv
preprint arXiv:2311.08692 , 2023.
[34] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. When not to trust language models: Investigating effectiveness of parametric and
non-parametric memories. arXiv preprint arXiv:2212.10511 , 2022.
[35] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez,
M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms from preference data. In
The Thirteenth International Conference on Learning Representations , 2024.
[36] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale
multi-subject multi-choice dataset for medical domain question answering. In Conference on
health, inference, and learning , pages 248–260. PMLR, 2022.
[37] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language
understanding by generative pre-training. 2018.
[38] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.
[39] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-
networks. arXiv preprint arXiv:1908.10084 , 2019.
[40] Marija Šakota, Maxime Peyrard, and Robert West. Fly-swat or cannon? cost-effective language
model choice via meta-modeling. In Proceedings of the 17th ACM International Conference on
Web Search and Data Mining , pages 606–615, 2024.
[41] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael
Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context.
InInternational Conference on Machine Learning , pages 31210–31227. PMLR, 2023.
[42] Tal Shnitzer, Anthony Ou, Mírian Silva, Kate Soule, Yuekai Sun, Justin Solomon, Neil Thomp-
son, and Mikhail Yurochkin. Large language model routing with benchmark datasets. arXiv
preprint arXiv:2309.15789 , 2023.
[43] Devendra Singh, Siva Reddy, Will Hamilton, Chris Dyer, and Dani Yogatama. End-to-end
training of multi-document reader and retriever for open-domain question answering. Advances
in Neural Information Processing Systems , 34:25968–25981, 2021.
[44] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement
learning. arXiv preprint arXiv:2503.05592 , 2025.
[45] Dimitris Stripelis, Zijian Hu, Jipeng Zhang, Zhaozhuo Xu, Alay Dilipbhai Shah, Han Jin,
Yuhang Yao, Salman Avestimehr, and Chaoyang He. Tensoropera router: A multi-model router
for efficient llm inference. arXiv preprint arXiv:2408.12320 , 2024.
[46] Gemma Team. Gemma. 2024.
[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information
processing systems , 30, 2017.
[48] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep
self-attention distillation for task-agnostic compression of pre-trained transformers. Advances
in neural information processing systems , 33:5776–5788, 2020.
[49] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan
Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, et al. Searching for best practices in retrieval-
augmented generation. arXiv preprint arXiv:2407.01219 , 2024.
12

[50] Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot,
Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, et al. Speculative rag: Enhancing
retrieval augmented generation through drafting. arXiv preprint arXiv:2407.08223 , 2024.
[51] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems , 35:24824–24837, 2022.
[52] Zhepei Wei, Wei-Lin Chen, and Yu Meng. Instructrag: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. arXiv preprint arXiv:2406.13629 , 2024.
[53] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-
pack: Packed resources for general chinese embeddings. In Proceedings of the 47th international
ACM SIGIR conference on research and development in information retrieval , pages 641–649,
2024.
[54] Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Yanqiao Zhu, May D Wang, Joyce C Ho, Chao
Zhang, and Carl Yang. Bmretriever: Tuning large language models as better biomedical text
retrievers. arXiv preprint arXiv:2404.18443 , 2024.
[55] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115 , 2024.
[56] Haoran Yang, Yumeng Zhang, Jiaqi Xu, Hongyuan Lu, Pheng Ann Heng, and Wai Lam.
Unveiling the generalization power of fine-tuned large language models. arXiv preprint
arXiv:2403.09162 , 2024.
[57] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang,
Heng Li, Jiangcheng Zhu, Jianqun Chen, et al. Yi: Open foundation models by 01. ai. arXiv
preprint arXiv:2403.04652 , 2024.
[58] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. Retrieve anything to
augment large language models. arXiv preprint arXiv:2310.07554 , 2023.
[59] Yue Zhang, Ming Zhang, Haipeng Yuan, Shichun Liu, Yongyao Shi, Tao Gui, Qi Zhang, and
Xuanjing Huang. Llmeval: A preliminary study on how to evaluate large language models. In
Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 19615–19622,
2024.
[60] Richard Zhuang, Tianhao Wu, Zhaojin Wen, Andrew Li, Jiantao Jiao, and Kannan Ramchan-
dran. Embedllm: Learning compact representations of large language models. arXiv preprint
arXiv:2410.02223 , 2024.
13

A Experimental Setup Details
A.1 Details of Candidate LLMs
The responses of Qwen2.5-72B-Instruct and Llama-3-70B-Instruct were obtained via API calls, while
other open-source LLMs were locally deployed using the vLLM framework [ 28] for high-speed
inference from Huggingface5. Notably, latency calculations for API-based models incorporated both
average network latency and inference time, whereas latency measurements for locally deployed
models exclusively accounted for inference time.
A.2 Details of Datasets
Table 6: The statistics results for different tasks.
PopQA MedMCQAQuery Num
Local Online Local OnlineNQ WQ TQA
Train 2000 2000 2000 2000 1000 1000 1000
Test 270 270 270 270 240 240 240
As shown in Table 7, we randomly sampled queries from five knowledge-intensiv tasks (PopQA
[34], MedMCQA [ 36], NQ [ 27], WebQ [ 2], and TriviaQA [ 25]) and partitioned them into training
and test sets. For PopQA and MedMCQA, we retrieved documents through both local and online
search engines following [ 44], while for NQ, WebQ, and TriviaQA, we constructed artificially noised
documents based on [ 11], with an equal proportion of four types: Golden Context, Relevant Retrieval
Noise, Irrelevant Retrieval Noise, and Counterfactual Retrieval Noise. Each query-document pair
was processed by the 15 LLMs described in Section 5.1 to generate responses. These responses were
then evaluated against ground-truth answers to derive binary scores.
(a) (b) (c)
(e) (f) (g)
Figure 5: Accuracy of 15 LLMs before and after RAG under various tasks and retrieval settings: (a)
PopQA (Online), (b) MedMCQA (Local), (c) MedMCQA (Online), (d) NQ, (e) WebQ, (f) TQA.
5https://huggingface.co
14

B Extended Data Analysis
Performance Shift with and without RAG. We analyze how LLM performance changes before
and after RAG across different models and settings. As shown in Figure 5, on real retrieval settings
such as PopQA (Online), MedMCQA (Local), and MedMCQA (Online), RAG improves overall
performance and reduces the accuracy gap among models. Notably, small-scale models benefit more
(e.g., Qwen2.5-0.5B-Instruct achieves a 17.57% improvement on MedMCQA (Online)) compared
to larger models (e.g., Llama-3.3-70B-Instruct improves by only 0.88%). In contrast, under noisy
retrieval settings (NQ, WebQ, TriviaQA), the impact of RAG varies—some models improve while
others degrade (e.g., on NQ, Llama-3.3-70B-Instruct performs worse, while Qwen2.5-0.5B-Instruct
performs better). These results highlight inconsistent performance shifts across models, indicating
that RAG significantly alters the distribution of response quality, which undermines the assumptions
of non-RAG-aware routing strategies.
Qwen2.5-0.5B-Instruct Llama-3.2-1B-Instruct Qwen2.5-1.5B-Instruct gemma-2-2b-it Llama-3.2-3B-Instruct Qwen2.5-3B-Instruct Yi-1.5-6B-Chat Qwen2.5-7B-Instruct Ministral-8B-Instruct-2410 Meta-Llama-3.1-8B-Instruct Yi-1.5-9B-Chat Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct Llama-3.3-70B-Instruct Avg
PopQA (Local)
PopQA (Online)
MedMCQA (Local)
MedMCQA (Online)
NQ
WQ
TQA
Avg36.20 34.47 35.76 34.98 34.66 33.23 37.13 33.98 30.82 32.31 37.13 33.92 32.66 30.60 24.15 33.47
45.80 42.33 44.53 43.08 41.77 41.32 45.02 41.14 37.19 37.24 45.29 39.87 38.09 36.15 26.48 40.35
21.59 13.76 22.27 13.27 18.59 20.26 27.21 26.46 24.94 31.74 32.93 32.06 29.24 24.54 25.57 24.29
33.86 31.64 32.96 32.08 35.82 35.24 37.39 37.06 43.81 47.34 49.74 46.91 36.88 40.72 40.94 38.83
23.99 22.01 22.59 20.65 24.21 23.24 26.92 20.74 22.98 26.13 28.04 27.31 22.34 22.24 22.16 23.70
25.87 20.19 23.96 17.17 25.87 20.17 30.09 19.37 15.55 22.53 31.32 26.48 16.82 15.17 16.70 21.82
29.23 29.29 30.39 25.13 37.22 29.23 36.67 33.94 30.83 36.43 40.71 38.70 36.40 33.66 35.85 33.58
30.94 27.67 30.35 26.62 31.16 28.95 34.35 30.38 29.44 33.39 37.88 35.04 30.35 29.01 27.41 30.86
15 20 25 30 35 40 45
Positive Gain Rate (%)
Figure 6: Positive Gain Rates of 15 candidate LLMs across tasks.
Qwen2.5-0.5B-Instruct Llama-3.2-1B-Instruct Qwen2.5-1.5B-Instruct gemma-2-2b-it Llama-3.2-3B-Instruct Qwen2.5-3B-Instruct Yi-1.5-6B-Chat Qwen2.5-7B-Instruct Ministral-8B-Instruct-2410 Meta-Llama-3.1-8B-Instruct Yi-1.5-9B-Chat Qwen2.5-14B-Instruct Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct Llama-3.3-70B-Instruct Avg
PopQA (Local)
PopQA (Online)
MedMCQA (Local)
MedMCQA (Online)
NQ
WQ
TQA
Avg22.95 21.36 16.22 28.57 20.78 21.11 19.34 25.18 21.78 24.61 18.07 19.75 20.37 24.09 27.29 22.10
13.60 11.65 6.76 9.61 9.83 10.26 6.62 9.41 11.00 13.86 8.65 8.98 12.04 11.27 16.09 10.64
55.73 62.37 50.85 70.18 57.01 45.11 64.17 53.34 23.64 45.44 31.81 28.22 44.66 15.83 14.69 44.20
50.92 41.05 43.69 44.34 41.51 36.14 52.58 46.32 16.90 40.25 27.04 22.97 37.36 15.83 15.44 35.49
48.88 51.69 39.85 63.81 39.88 47.65 43.60 51.29 37.16 38.06 39.52 34.21 30.41 34.46 27.88 41.89
40.08 45.59 47.18 54.72 38.86 47.87 45.11 49.21 31.54 40.33 39.38 34.05 24.52 28.75 30.17 39.82
38.36 33.33 32.65 44.44 23.55 31.60 29.49 31.84 20.93 22.10 29.04 15.83 11.26 15.51 10.67 26.04
38.64 38.15 33.88 45.09 33.06 34.25 37.27 38.08 23.28 32.09 27.64 23.43 25.80 20.82 20.32 31.46
10 20 30 40 50 60 70
Negative Interference Rate (%)
Figure 7: Negative Interference Rates of 15 candidate LLMs across tasks.
Analysis of Response Quality Reversal. We further investigate how RAG causes quality reversals
for the same query. To quantify these reversals, we define two metrics at the task level. The positive
gain rate is the proportion of queries that were unanswerable without retrieval but became answerable
with retrieved documents, calculated as:
Positive Gain Rate =#(Incorrect w/o RAG →Correct w/ RAG)
#(Incorrect w/o RAG)(10)
The negative interference rate measures the opposite effect—queries that were answerable without
retrieval but became unanswerable due to retrieved content:
Negative Interference Rate =#(Correct w/o RAG →Incorrect w/ RAG)
#(Correct w/o RAG)(11)
15

Figures 6 and 7 report these metrics across 15 LLMs and multiple settings. On average, across all
models and tasks, the positive gain rate is 30.86%, negative interference rate is 31.46%. These results
confirm that response quality reversals induced by external documents are common suggests that
RAG-induced knowledge shifts are widespread among LLMs.
C Routing Performance under Noisy Retrieval
We further partition the TriviaQA dataset into four subsets with manually injected retrieval
noise—Golden Context, Relevant Noise, Irrelevant Noise, and Counterfactual Noise—to evalu-
ate the routing effectiveness of RAGRouter under different noise conditions, in comparison with
various baseline methods. As shown in Table 7, RAGRouter consistently achieves the best perfor-
mance across all subsets, outperforming both the Oracle Single Best and other routing strategies,
demonstrating strong robustness to different types of retrieval noise.
Notably, on the Relevant Noise, Irrelevant Noise, and Counterfactual Noise subsets, non-RAG-aware
baselines exhibit significant performance gaps compared to RAGRouter, highlighting their limitations
in high-noise retrieval scenarios. We hypothesize that this is due to knowledge shift induced by noisy
retrieval, which affects different LLMs in heterogeneous ways. As a result, routing methods that rely
on fixed model representations and ignore RAG capabilities struggle to accurately model routing
strategies under such conditions.
Table 7: Test accuracy (%) of RAGRouter and baselines on TriviaQA subsets with different types of
retrieval noise, bold indicates best results.
Method Golden Context Relevant Noise Irrelevant Noise Counterfactual Noise Avg
Oracle Single Best 95.00 83.33 90.00 83.33 87.92
KNN Router 96.67 78.33 83.33 86.67 86.25
GraphRouter 95.00 83.33 90.00 83.33 87.92
RouterDC 80.00 78.33 73.33 78.33 77.50
MF 95.00 76.67 80.00 80.00 82.92
RAGRouter 98.33 86.67 90.00 88.33 90.83
D Full Results on Latency-Aware Routing
Setting Details. Based on LLMs’ profiles, the 15 candidate models in Section 5.3 were ranked as
follows in ascending order: Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct,
gemma-2-2b-it, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Yi-1.5-6B-Chat, Qwen2.5-7B-Instruct,
Ministral-8B-Instruct-2410, Meta-Llama-3.1-8B-Instruct, Yi-1.5-9B-Chat, Qwen2.5-14B-Instruct,
Qwen2.5-32B-Instruct, Qwen2.5-72B-Instruct, Llama-3.3-70B-Instruct. Substitution models were
selected from immediate predecessors of the highest-routing-score model within the threshold. For
quantitative analysis, we discretized the threshold parameter θover [0, 1] with a step size of 1e-4 to
generate complete high-precision accuracy–latency trade-off curves.
Table 8: Area (%), Peak Accuracy (PA, %), and Latency Gap-to-Match (G, s) for RAGRouter and
baselines using score-threshold-based routing on PopQA (Local/Online), NQ and WebQ. "-" in
Latency Gap-to-Match indicates failure to match the best single-LLM performance; "-" in Area
denotes maximum routing latency below 1s, excluded from comparison; bold denotes the best result.
PopQA (Local) PopQA (Online) NQ WQMethod
Area ↑ PA↑ G (s) ↑Area ↑ PA↑ G (s) ↑Area ↑ PA↑ G (s) ↑Area ↑ PA↑ G (s) ↑
RouterDC 44.58 47.04 0.02 49.80 51.85 - 33.10 42.92 - 46.39 49.17 -
MF - 46.30 - 49.96 52.22 -0.91 37.34 44.17 - - 50.42 1.66
GraphRouter - 47.41 -0.50 - 51.48 - 35.68 38.35 0.02 45.67 48.18 0.01
KNN Router - 46.67 -0.06 - 52.22 - 41.13 50.63 0.72 - 50.42 1.72
RAGRouter 45.13 48.52 -0.48 50.13 52.59 0.15 43.63 55.83 1.13 - 58.33 1.84
Results. Figure 8 illustrates the accuracy–latency curves of RAGRouter and baseline methods on
PopQA (Local/Online), NQ, and WebQ, while Tables 8 present their quantitative results on Area,
16

0 250 500 750 1000 1250 1500 1750 2000
Latency per Query (ms)4042444648505254Testing Accuracy (%)
PopQA (Local)
RAGRouter
0 250 500 750 1000 1250 1500 1750 2000
Latency per Query (ms)464850525456Testing Accuracy (%)
PopQA (Online)
RAGRouter
0 250 500 750 1000 1250 1500 1750 2000
Latency per Query (ms)3040506070Testing Accuracy (%)
NQ
RAGRouter
0 250 500 750 1000 1250 1500 1750 2000
Latency per Query (ms)304050607080Testing Accuracy (%)
WQ
RAGRouterQwen2.5-0.5B-Instruct
Llama-3.2-1B-Instruct
Qwen2.5-1.5B-Instruct
gemma-2-2b-itLlama-3.2-3B-Instruct
Qwen2.5-3B-Instruct
Yi-1.5-6B-Chat
Qwen2.5-7B-InstructMinistral-8B-Instruct-2410
Meta-Llama-3.1-8B-Instruct
Yi-1.5-9B-Chat
Qwen2.5-14B-InstructQwen2.5-32B-Instruct
Qwen2.5-72B-Instruct
Llama-3.3-70B-Instruct
RandomWeighted
Prompt LLM
Oracle
EmbedLLMKNN Router
RouterDC
GraphRouter
RAGRouter (Ours)Figure 8: Accuracy–latency curves on PopQA (Local/Online), NQ and WebQ.
Peak Acc, and Latency Gap-to-Match metrics. RAGRouter achieves the highest scores in Area and
Peak Acc, with its accuracy–latency curve mostly surpassing the baselines, demonstrating strong
performance-efficiency trade-offs under low-latency constraints.
E Full Results of Sensitivity Analysis and Ablation Study
128 256 384 512 768 1024
DimensionPopQA (Local)
PopQA (Online)
MedMCQA (Local)
MedMCQA (Online)
NQ
WQ
TQA
Avg0.00 0.00 1.53 0.76 0.00 0.00
0.00 2.13 0.71 0.71 0.71 0.71
0.00 -1.04 -1.04 -1.04 0.52 0.00
0.00 -0.50 -1.51 -0.50 1.00 0.00
0.00 2.31 3.84 2.31 4.62 2.31
0.00 4.69 0.79 5.48 6.26 -3.11
0.00 0.94 1.42 1.42 3.31 2.37
0.00 1.03 0.68 1.14 2.32 0.46
−20246
Testing Accuracy Change Rate (%)
Figure 9: Effects of dimension (baseline: 128).
0.0 0.5 1.0 1.5 2.0 2.5 3.0
lPopQA (Local)
PopQA (Online)
MedMCQA (Local)
MedMCQA (Online)
NQ
WQ
TQA
Avg0.00 0.00 -0.76 0.00 -0.76 -0.76 -0.76
0.00 0.70 0.00 0.00 0.00 0.70 0.70
0.00 0.53 1.58 1.58 1.58 1.58 2.10
0.00 -0.50 -0.99 -0.50 0.00 0.00 0.00
0.00 0.00 0.00 0.74 2.26 3.00 0.74
0.00 0.00 2.31 2.31 4.62 3.84 3.06
0.00 0.00 0.00 0.45 0.45 0.00 -0.46
0.00 0.08 0.28 0.63 1.10 1.08 0.71
01234
Testing Accuracy Change Rate (%) Figure 10: Effects of λ(baseline: λ= 0).
Table 9 presents the impact of the dimensionality of LLM knowledge embeddings and RAG capability
embeddings on test accuracy in the RAGRouter architecture, with Figure 9 showing testing accuracy
change rates relative to the 128-dimensional baseline. The best average accuracy is observed at
dimension 768.
17

Table 9: Effects of dimension.
PopQA MedMCQADimension
Local Online Local OnlineNQ WQ TQA Avg
128 48.52 52.22 71.11 73.70 54.17 53.33 87.92 63.00
256 48.52 53.33 70.37 73.33 55.42 55.83 88.75 63.65
384 49.26 52.59 70.37 72.59 56.25 53.75 89.17 63.43
512 48.89 52.59 70.37 73.33 55.42 56.25 89.17 63.72
768 48.52 52.59 71.48 74.44 56.67 56.67 90.83 64.46
1024 48.52 52.59 71.11 73.70 55.42 51.67 90.00 63.29
Table 10 presents the impact of the loss weight λon test accuracy, with Figure 10 showing testing
accuracy change rates relative to the baseline of λ= 0. The best average accuracy is observed at
λ= 2.
Table 10: Effects of λ.
PopQA MedMCQAλ
Local Online Local OnlineNQ WQ TQA Avg
0.0 48.89 52.59 70.37 74.44 55.42 54.17 90.42 63.76
0.5 48.89 52.96 70.74 74.07 55.42 54.17 90.42 63.81
1.0 48.52 52.59 71.48 73.70 55.42 55.42 90.42 63.94
1.5 48.89 52.59 71.48 74.07 55.83 55.42 90.83 64.16
2.0 48.52 52.59 71.48 74.44 56.67 56.67 90.83 64.46
2.5 48.52 52.96 71.48 74.44 57.08 56.25 90.42 64.45
3.0 48.52 52.96 71.85 74.44 55.83 55.83 90.00 64.21
F Case Study
We further illustrate RAGRouter’s ability to perceive changes in LLM knowledge states under RAG
conditions through the two case studies shown in Table 11 and Table 12. In Table 11, the retrieved
document contains both correct answer information and certain distracting content. In this case,
RAGRouter identifies that the document provides significant performance gains for Qwen2.5-14B-
Instruct and accordingly selects it as the responder. Although this model produces an incorrect
response in the non-RAG setting due to confusion between two French departments, it successfully
corrects the answer when the document is incorporated. In contrast, Meta-Llama-3.1-8B-Instruct
exhibits greater sensitivity to distracting content in the document, where the introduction of RAG
leads to reverse interference and impairs its reasoning, and thus it is not prioritized by RAGRouter.
However, traditional routing strategies without RAG awareness, such as MF, rely solely on static
model performance and fail to perceive the document-induced performance shifts, ultimately routing
to Meta-Llama-3.1-8B-Instruct and resulting in routing failure for this sample.
As shown in Table 12, the query itself is challenging, and neither Llama-3.3-70B-Instruct nor
Qwen2.5-72B-Instruct is able to provide a correct answer without access to external documents.
However, when the retrieved document containing key information is provided, Llama-3.3-70B-
Instruct, benefiting from its stronger capabilities in information extraction and comprehension,
successfully identifies “Poreotics” as the correct answer. In contrast, Qwen2.5-72B-Instruct fails to
effectively utilize the document and still produces an incorrect response. In this scenario, RAGRouter
is able to sense the differential capabilities of the candidate LLMs in leveraging retrieved content
and routes the query to the model with stronger information extraction ability, leading to a correct
response. By contrast, non-RAG-aware routing methods based on static modeling assumptions, such
as RouterDC, fail to capture dynamic performance shifts induced by retrieval and result in incorrect
routing and response failure. This case further highlights the advantage of RAGRouter in perceiving
capability differences among LLMs in retrieval-augmented settings.
18

Table 11: A case from PopQA (Online), demonstrating RAGRouter’s ability to perceive LLMs’ RAG
capability and knowledge shift.
Query What is Agen the capital of?
Document Agen, located in the Nouvelle-Aquitaine region of Southwestern France, serves
as the prefecture of the Lot-et-Garonne department. Known for its geographical
positioning along the river Garonne, the city lies approximately 135 kilometers
southeast of Bordeaux. It has a rich cultural heritage, featuring various historical
buildings such as the twelfth-century Agen Cathedral and numerous museums,
including the Musée des Beaux Arts. Agen is also colloquially referred to as the
¨capital of the prune, ¨hosting a popular prune festival every August. The town,
with a population of 32,485 in 2021, has its own Roman Catholic diocese, adding
to its historical significance within the region.
Ground truth Lot-et-Garonne
Router RAGRouter MF
Selected LLM Qwen2.5-14B-Instruct Meta-Llama-3.1-8B-Instruct
Response w/o RAG Lot department. ( ✗) Lot-et-Garonne department. ( ✓)
Response w/ RAG Lot-et-Garonne department. ( ✓) The prune capital. ( ✗)
Table 12: A case from NQ, demonstrating RAGRouter’s ability to perceive LLMs’ RAG capability
and knowledge shift.
Query who are the dancers in the lazy song video?
Document tenth was the one chosen. The official video was directed by Mars and Cameron
Duddy, produced by Nick Tabri and Dara Siegel, and features Poreotics wearing
chimpanzee masks; it was released on April 15, 2011. The whole video is
presented in as a lone continuous and uninterrupted shot, it begins with Mars
singing and hanging out in a bedroom with five dancers, they all wear monkey
masks and Mars dresses in black sunglasses and a flannel shirt. While Mars
sings what he feels to do on a day off, he and the monkeys perform dance moves
typical of a boy-band,
Ground truth Poreotics
Router RAGRouter RouterDC
Selected LLM Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct
Response w/o RAG Bruno Mars and dancers. ( ✗) Five dancers. ( ✗)
Response w/ RAG Poreotics dancers. ( ✓) Bruno Mars and his backup dancers. ( ✗)
G Inference Cost of RAGRouter
We report the inference characteristics of RAGRouter on a single NVIDIA RTX 4090D GPU. The
peak GPU memory usage is 4147 MiB. With a batch size of 64, RAGRouter processes 270 instances
in 3 seconds over 5 batches, resulting in an average inference time of 0.011 seconds per instance.
These results demonstrate that RAGRouter is both lightweight and efficient during inference.
H Limitations and Future Works
Beyond RAG as the mainstream paradigm, recent techniques such as fine-tuning [ 20,19], knowledge
distillation [ 18], and Chain-of-Thought prompting [ 51] have also been widely adopted to enhance the
performance of LLM. These approaches inherently involve explicit or implicit modulation of model
knowledge, often entailing complex processes of knowledge transfer or shift. Although this work
focuses on capability modeling and dynamic routing within RAG settings, we believe the proposed
framework and analytical perspective offer valuable insights into the representation and utilization
of LLM capabilities in broader contexts. As this lies beyond the scope of this paper, we leave the
exploration to future work.
19
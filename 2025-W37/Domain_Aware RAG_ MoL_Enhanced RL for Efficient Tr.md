# Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval

**Authors**: Hao Lin, Peitong Xie, Jingxue Chen, Jie Lin, Qingkun Tang, Qianchun Lu

**Published**: 2025-09-08 13:04:07

**PDF URL**: [http://arxiv.org/pdf/2509.06650v1](http://arxiv.org/pdf/2509.06650v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems rely heavily on the retrieval
stage, particularly the coarse-ranking process. Existing coarse-ranking
optimization approaches often struggle to balance domain-specific knowledge
learning with query enhencement, resulting in suboptimal retrieval performance.
To address this challenge, we propose MoLER, a domain-aware RAG method that
uses MoL-Enhanced Reinforcement Learning to optimize retrieval. MoLER has a
two-stage pipeline: a continual pre-training (CPT) phase using a Mixture of
Losses (MoL) to balance domain-specific knowledge with general language
capabilities, and a reinforcement learning (RL) phase leveraging Group Relative
Policy Optimization (GRPO) to optimize query and passage generation for
maximizing document recall. A key innovation is our Multi-query Single-passage
Late Fusion (MSLF) strategy, which reduces computational overhead during RL
training while maintaining scalable inference via Multi-query Multi-passage
Late Fusion (MMLF). Extensive experiments on benchmark datasets show that MoLER
achieves state-of-the-art performance, significantly outperforming baseline
methods. MoLER bridges the knowledge gap in RAG systems, enabling robust and
scalable retrieval in specialized domains.

## Full Text


<!-- PDF content starts -->

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and
Scalable Retrieval
Hao Lin1
Southeast University
Nanjing, China
220230828@seu.edu.cnPeitong Xie1
Nanyang Technological University
Singapore, Singapore
xiep0004@e.ntu.edu.sgJingxue Chen
Wired Product Operation Division,
ZTE Corporation
Nanjing, China
chen.jingxue@zte.com.cn
Jie Lin
Wired Product Operation Division,
ZTE Corporation
Nanjing, China
lin.jie1@zte.com.cnQingkun Tang†
Wired Product Operation Division,
ZTE Corporation
Nanjing, China
tang.qingkun@zte.com.cnQianchun Lu
Wired Product Operation Division,
ZTE Corporation
Nanjing, China
ABSTRACT
Retrieval-Augmented Generation (RAG) systems rely heavily on
the retrieval stage, particularly the coarse-ranking process. Ex-
isting coarse-ranking optimization approaches often struggle to
balance domain-specific knowledge learning with query enhence-
ment, resulting in suboptimal retrieval performance. To address
this challenge, we propose MoLER, a domain-aware RAG method
that uses MoL-Enhanced Reinforcement Learning to optimize re-
trieval. MoLER has a two-stage pipeline: a continual pre-training
(CPT) phase using a Mixture of Losses (MoL) to balance domain-
specific knowledge with general language capabilities, and a rein-
forcement learning (RL) phase leveraging Group Relative Policy
Optimization (GRPO) to optimize query and passage generation
for maximizing document recall. A key innovation is our Multi-
query Single-passage Late Fusion (MSLF) strategy, which reduces
computational overhead during RL training while maintaining scal-
able inference via Multi-query Multi-passage Late Fusion (MMLF).
Extensive experiments on benchmark datasets show that MoLER
achieves state-of-the-art performance, significantly outperforming
baseline methods. MoLER bridges the knowledge gap in RAG sys-
tems, enabling robust and scalable retrieval in specialized domains.
Artifact Availability:
The source code, data, and/or other artifacts have been made available at
https://github.com/Jamoremore/MoLER.
1 INTRODUCTION
Retrieval-Augmented Generation (RAG) has emerged as a pivotal
framework in natural language processing (NLP), comprising three
core stages: retrieval, conditioning on retrieved documents and in-
put, and generation[ 15]. While RAG significantly enhances the
generation capabilities of large language model (LLM), its effi-
cacy is heavily contingent upon the quality of the retrieval phase.
During retrieval, coarse-ranking methods initially estimate query-
document similarity to generate an initial recall order, followed by
reranking to refine the results [ 18]. Given the critical dependency
of reranking on coarse-ranking outcomes [ 32], substantial research
1These authors contributed equally to this work.
†Corresponding author.has focused on improving coarse-ranking methodologies. This work
specifically targets advancements in coarse-ranking techniques to
address the limitations of existing approaches.
To enhance coarse-ranking performance, various query augmen-
tation methods have been proposed as critical strategies [ 3,27].
These methods, including query expansion [ 1], Query2Doc [ 25],
and multi-query generation [ 10], aim to improve the relevance and
diversity of retrieved documents by reformulating or expanding
the original query. For instance, query expansion replaces key-
words with synonyms to generate alternative queries [ 20,30], while
Query2Doc leverages LLM to pre-generate pseudo-passages for
retrieval. However, these approaches often overlook the LLM’s
contextual understanding of documents, a critical gap in RAG
scenarios where models typically operate in domains with lim-
ited prior knowledge . For example, synonym-based expansion
shows diminishing returns in dense semantic retrieval [ 13], and
Query2Doc’s pseudo-passage generation lacks explicit alignment
with the model’s comprehension of document content. These limita-
tions underscore the need for a methodology that directly optimizes
retrieval performance from the perspective of document relevance,
rather than relying on heuristic or template-based augmentation.
Jiang et al. [ 12] proposed the DeepRetrieval method, which employs
reinforcement learning (RL) to train an LLM capable of generating
and rewriting augmented queries with good retrieval performance.
However, the LLM was not explicitly trained to utilize stronger
query augmentation techniques such as instruction expansion or
pre-answering.
To address these challenges, we propose MoLER, a domain-aware
RAG framework that leverages MoL-Enhanced RL to optimize re-
trieval efficiency and scalability. MoLER integrates multi-query
expansion and LLM pre-answering techniques, grounded in three
core principles: (1) deepening the model’s comprehension of re-
trieved documents via continual learning [ 2]; (2) achieving more
efficient query enhancement through query expansion and question
pre-answering; and (3) further activating the LLM’s inherent knowl-
edge reservoir through RL to enable effective query augmentation.
MoLER systematically bridges the gap between conventional en-
hancement techniques and the evolving demands of RAG systems,
demonstrating significant advantages in scenarios where retrievalarXiv:2509.06650v1  [cs.CL]  8 Sep 2025

Hao Lin1, Peitong Xie1, Jingxue Chen, Jie Lin, Qingkun Tang†, and Qianchun Lu
General
Corpus
Domain
CorpusOutput Distribution
Output Distribution
Raw Query
Reinforcement 
Learning 
MQR 
generate n sub-queries 
by raw queryContinual 
 Pre-training
CQE-Train
CQE-Test generate 1 pasaage 
by n sub-queries & raw query  
Recall
RewardRetrievalRetrieved Doc
True Doc
Weight UpdateRRF Fusion
generate n pasaages 
by n sub-queries & raw query  
Figure 1: The MoLER framework’s training and inference pipeline. During training, the model first undergoes CPT using the
MoL approach, which balances domain-specific (CE loss) and general knowledge (KL divergence) corpora. In the RL phase, the
MQR generates diverse queries, which are consolidated into a single synthetic passage via CQE to optimize recall performance.
During inference, MQR generates multiple queries, each of which is processed independently to produce a passage. The retrieval
results for these passages are then fused using RRF to enhance final recall performance.
performance highly depends on the model’s domain-specific exper-
tise.
For the first requirement, we employ the MoL (Mixture of Losses)
training methodology, which distinguishes between domain-
specific and general corpora [ 2]. Domain corpora are trained using
cross-entropy (CE) loss, while general corpora utilize KL diver-
gence. Empirical results demonstrate that this approach effectively
enhances domain expertise while preserving general capabilities.
For the second and third requirements, we apply the Group
Relative Policy Optimization (GRPO) [ 6] based RL algorithm to op-
timize end-to-end recall. Although state-of-the-art methods such as
multi-query multi-passage late fusion (MMLF)[ 14] are assumed to
exhibit strong scalability in retrieval performance with the number
of generated queries and passages, this property is not formally
validated in the original MMLF work and remains an open research
question. The MMLF paper itself acknowledges this as a limitation,
noting that computational efficiency during training and inference
becomes prohibitive when generating multiple passages. To address
this critical trade-off, we propose Multi-query Single-passage Late
Fusion (MSLF), which reduces training complexity by consolidating
multiple queries into a single synthetic passage during RL optimiza-
tion. Crucially, our experiments demonstrate that the scalability
benefits of MMLF. Specifically, a log-linear relationship between
retrieval performance and the number of generated queries, are pre-
served in inference even after MSLF training. This strategy enables
us to verify the scalability hypothesis of MMLF while maintain-
ing training efficiency, ensuring the model retains MMLF’s query
expansion capabilities in deployment without incurring excessive
computational costs during RL training.
We conduct rigorous experiments on NFCORPUS and SCIFACT,
two representative retrieval datasets, to evaluate MoLER’s perfor-
mance. The results show that MoLER consistently outperforms
baseline methods across all datasets, achieving statistically signifi-
cant improvements in document recall. The superior recall perfor-
mance underscores the importance of end-to-end optimization fromthe perspective of document relevance, particularly in scenarios
where LLMs lack domain-specific knowledge.
Our contributions are as follows:
•We identify the critical limitation of existing query aug-
mentation methods: their failure to account for the LLM’s
contextual understanding of documents, which is particu-
larly vital in RAG scenarios where models often lack domain
expertise.
•A novel method (MoLER) that bridges the gap between tra-
ditional query augmentation techniques and dynamic RAG
requirements through RL-based end-to-end optimization.
•The MSLF method, which enables efficient RL training via
single-passage generation while maintaining MMLF-based
scalability during inference, achieving a critical balance
between training efficiency and inference effectiveness in
downstream domains.
•We demonstrate through extensive experiments on bench-
mark datasets (e.g., NFCORPUS, SCIFACT) that MoLER
achieves significant improvements in retrieval performance,
validating its effectiveness in bridging the knowledge gap
in RAG systems.
2 RELATED WORK
Query Agumentation.Query augmentation aims to reformulate
and agunent user’s input query which may be ambigous and cover
insufficient dimensions. Query expansion is an earlier proposed
method that replaces keywords in the original query with syn-
onyms [ 1]. Although it demonstrated good performance in early
sparse retrievers based on keyword matching [ 20,30], query expan-
sion has shown limited performance improvements with the rise of
dense retrievers based on semantic matching [ 13,29]. Query2Doc
leverages LLMs to firstly generate a pseudo-passage based on oring-
inal query and then uses this enriched context to retrieve [ 25].
Mill proposed a Query-Query-Document method which utilizess

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
Original Query qLLM 
MQR Promptq1
q2LLM
CQE PromptPRetrieverRetrieved 
Candidates
RRF Result
Retrieved 
CandidatesMulti-query Generation Query-to-passage Expansion Ranked List Fusion
qn
(a) Training: Multi-query Single-passage Late Fusion Retrieval
Original Query qq1
q2 P Retriever RRF ResultQuery-to-passage ExpansionRanked List Fusion
qnLLM
CQE Prompt
LLM
CQE Prompt
LLM
CQE PromptPPRetrieved 
Candidates
Retrieved 
Candidates
Retrieved 
CandidatesRetrieved 
CandidatesMulti-query Generation
LLM 
MQR Prompt
(b) Testing: Multi-query Multi-passage Late Fusion Retrieval
Figure 2: The figure illustrates the distinction between MSLF and MMLF in MoLER’s retrieval pipeline. (a) Training Phase (MSLF):
Multiple queries generated via MQR are consolidated into a single synthetic passage using CQE. (b) Inference Phase (MMLF):
Each query is processed independently to produce a distinct passage, and retrieval results are fused using RRF. This approach
maximizes recall during inference by leveraging diverse query-document interactions. The design balances training efficiency
(MSLF) and inference effectiveness (MMLF), ensuring optimal retrieval performance without compromising computational
resources.
LLMs to decomposes the original query into multi-dimensional
sub-queries, pre-answer each sub-queries and concatenate pseudo-
passages into a continuous text for retrieval [ 10]. MMLF employs
Multi Query Retriever (MQR) to generate diverse sub-queries and
Combined Query Expansion (CQE) to expand them into contex-
tual passages utilizes [ 14], and fusing rankings via reciprocal rank
fusion (RRF) instead of concatenate pseudo-passages [4].
Continuous Pre-training.Traditional CPT [ 5,8] methods typically
mix general and domain-specific corpora to train models to mitigate
catastrophic forgetting. However, determining the optimal mixing
ratio is challenging and requires extensive ratio experiments. Song
et al. propose a dual-objective optimization strategy during domain-
specific tuning to address catastrophic forgetting, using regularized
loss for general data and cross-entropy loss for domain data [ 23].
However, they did not provide the optimal ratio for general and
domain data. Chen et al. [ 2] introduce a novel dual-loss architecture,
MOL, to address these issues: general corpora are trained with KL
divergence loss to preserve foundational capabilities, while domain-
specific corpora are trained with CE loss to enhance specialized
knowledge. They explicitly propose that a nearly 1:1 ratio of domain
to general corpora effectively balances training outcomes, avoiding
the waste of computational resources.
RL for LLM Query Agumentation.Recent works leverage RL
to optimize query generation in retrieval tasks. Jiang et al.[ 12]
proposed DeepRetrieval, an RL framework that trains LLMs to
generate/rewrite queries using task-specific reward metrics. Xiao
et al.[ 28] introduced C-Pack, a three-stage pipeline (pre-training,Table 1: MQR Prompt
MQR prompt
You are an AI language model assistant. Your task is to generate
exactly {cnt} different versions of the given user question to re-
trieve relevant documents from a vector database. By generating
multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based
similarity search.
Original question: query
Format your response in plain text as:
{example}
contrastive learning, fine-tuning) to enhance Chinese text embed-
dings for coarse ranking. Shao et al. [ 22] developed REASONIR-
SYNTHESIZER, an automated data generation pipeline for train-
ing retrievers with reasoning-required queries. These approaches
demonstrate the potential of RL and data synthesis in improving
retrieval performance but focus on different aspects of the problem.
3 METHOD
For improving the performance of coarse ranking, this paper pro-
poses a novel training framework named MoLER. As shown in
Figure 1, MoLER consists of two core stages: a CPT stage based on
MoL and a retrieval optimization stage based on RL.

Hao Lin1, Peitong Xie1, Jingxue Chen, Jie Lin, Qingkun Tang†, and Qianchun Lu
Table 2: CQE Prompt
CQE prompt
Please write a passage to answer the following user questions
simultaneously.
Question 1: {original_query}
Question 2: {sub_query_1}
...
Question n: {sub_query_n+1}
Format your response in plain text as:
Passage:
(1)CPT with MoL.We adopt a dual-loss architecture in which
the training corpus is divided into two categories: domain
documents and general documents. For domain knowl-
edge, we apply CE loss to enhance domain-specific knowl-
edge learning based on retrievable documents. For general
knowledge, we use the KL divergence loss, treating the
LLMs reasoning corpus as general knowledge to preserve
the model’s general language capabilities.
(2)RL based fine-tuning.In this stage, RL is employed to
distill compressed knowledge into specialized document
retrieval capability. First, an MQR Prompt is used to seman-
tically expand the raw query; then, a CQE Prompt is used
to generate a preliminary answer. Finally, the preliminary
answer and the original query are combined using RRF
to further optimize the retrieval performance on private-
domain documents.
This section first introduces the document retrieval strategy
adopted by MoLER, followed by a detailed explanation of the algo-
rithmic design and process implementation of the CPT stage and
the RL stage.
3.1 Redesigned RAG Pipeline
MoLER enhances retrieval performance through the same three-
step process as [ 14]: instruction expansion, pre-answer guidance,
and reciprocal rank fusion.
(1)Instruction Expansion:Using a zero-shot prompting ap-
proach, the original query 𝑞is expanded into 𝑛semantically
related sub-queries {𝑞1,𝑞2,...,𝑞𝑛}with varied perspectives
and richer details. These form the expanded query set Q,
providing multi-view retrieval entry points.
(2)Pre-Answer Guidance:Traditional RAG methods directly
retrieve documents using the original query, which can
lead to retrieval bias when the query is ambiguous. MoLER
introduces a pre-answer mechanism. During the training
phase, all sub-queries are merged to generate a unified
pseudo-document 𝑝. During the testing phase, a pseudo-
document𝑝𝑖is generated for each 𝑞𝑖to serve as contextualenhancement for retrieval. Specifically, to enhance both in-
struction expansion and pre-answer generation capabilities
with similar token consumption while effectively reducing
the number of dialogues required for training, we adopt
an MSLF strategy during training and an MMLF strategy
during testing (as shown in Figure 2).
(3)Reciprocal Rank Fusion:The retrieval results from the
original query and the sub-queries are fused using RRF, as
follows:
•Compute similarity scores between 𝑞and the coarse-
ranked documents to obtain the similarity ranking list
𝐿0;
•Compute similarity scores between each pseudo-
document𝑝𝑖(corresponding to 𝑞𝑖) and the coarse-
ranked documents to obtain the ranking lists𝐿 𝑖;
•Compute the final RRF score and ranking for the orig-
inal query𝑞using:
𝑠(𝑞)=𝑛∑︁
𝑘=01
rank𝑘(𝑑)+𝐾(1)
where rank𝑘(𝑑)denotes the rank of document 𝑑in𝐿𝑘,
and𝐾=60is the standard RRF constant [14].
3.1.1 Key Prompt Strategies.To implement the redesigned RAG
pipeline, MoLER uses two critical prompt strategies, MQR and CQE,
that enable efficient query augmentation and contextual alignment.
Below, we detail their design and implementation.
The central idea of the MQR prompt is to generate multiple
reformulated queries that mitigate semantic drift issues inherent in
vector similarity search. Unlike the design in [ 14], we introduce two
custom parameters: 𝑐𝑛𝑡, denoting the number of expanded queries,
and𝑒𝑥𝑎𝑚𝑝𝑙𝑒 , which provides task-specific examples depending on
the query number. Table 1 shows the general format of the MQR
prompt.
Following [ 14], we also adopt the CQE prompt. Unlike simple
sub-query expansion, the CQE prompt generates a passage that si-
multaneously answers both the original query and the sub-queries.
The corresponding template is provided in Table 2. The key innova-
tion lies in its parameterized structure where the choice of 𝑛(num-
ber of sub-queries) determines the underlying learning framework:
when𝑛> 1, the CQE prompt operates under the MSLF strategy
during training, generating a pseudo-document that simultane-
ously addresses multiple reformulated queries to enhance contex-
tual alignment. Conversely, during inference, the prompt switches
to the MMLF strategy by iterating over each expanded query (i.e.,
𝑛=1per iteration) to generate a distinct pseudo-document for each
sub-query. This approach ensures that the system leverages diverse
query-document interactions during inference while maintaining
computational efficiency in training.
3.2 Continual Pre-training
When applying RL to LLMs, training from scratch proves highly
inefficient, whereas CPT mitigates this by reducing RL exploration
costs through domain-specific knowledge assimilation. Traditional
CPT often encounters the dilemma of "domain overfitting" and "gen-
eral capability dilution" when injecting domain-specific knowledge

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
Table 3: This table compares the retrieval performance of various methods on the NFCORPUS and SCIFACT datasets. The
proposed MoLER method (Qwen3-1.7B+MoL+GRPO) achieves state-of-the-art results, significantly outperforming baseline
methods such as Raw Query, Q2D, CoT, and LC-MQR.
MethodNFCORPUS SCIFACT
Recall@1k nDCG@10 Recall@10 nDCG@10
Raw Query 52.95 21.69 59.82 46.57
Qwen3-32B+Q2D 60.06 25.15 78.29 62.20
Qwen3-32B+CoT 59.30 24.17 71.98 57.89
Qwen3-32B+LC-MQR 58.10 24.48 73.36 57.97
Qwen3-32B+MMLF 60.87 24.97 79.26 62.55
Qwen3-0.6B+MoL+Dr.GRPO 59.72 23.18 72.96 55.43
Qwen3-0.6B+MoL+GRPO 60.52 23.59 77.73 61.00
Qwen3-1.7B+MoL+Dr.GRPO 60.19 24.38 77.47 60.90
Qwen3-1.7B+MoL+GRPO61.42 25.44 79.69 62.59
into LLMs, leading to a degradation in their general-purpose abili-
ties and impairing comprehension of user queries. Therefore, it is
crucial to develop CPT methods that enable LLMs to acquire domain
knowledge while preserving their general capabilities. To address
this, MoLER adopts the MoL dual-loss architecture [ 2], enabling col-
laborative optimization that strengthens domain knowledge while
maintaining general capabilities through differentiated loss design.
Specifically, the training corpus is divided into a domain-specific
corpus𝐶𝑑and a general-domain corpus 𝐶𝑔, optimized jointly with
CE loss and KL divergence loss, respectively.
For a sequence 𝑠from the domain-specific corpus 𝐶𝑑, the CE loss
is defined as:
L𝐶𝐸(𝑠)=−1
𝑛𝑠∑︁
𝑖log𝑝𝜃(𝑠𝑖)(2)
where𝑛𝑠is the sequence length, and 𝑝𝜃(𝑠𝑖)denotes the probability
of generating token𝑠 𝑖under parameters𝜃.
For a sequence 𝑠from the general-domain corpus 𝐶𝑔, the KL loss
is defined as:
L𝐾𝐿(𝑠)=1
𝑛𝑠∑︁
𝑖KL[𝑝𝜃∥𝑝0](𝑠𝑖)(3)
Following [ 2], we employ an optimal 1:1 corpus ratio in LoRA
fine-tuning to avoid degradation of generalization ability due to
excessive domain data, thereby ensuring balanced performance.
3.3 Reinforcement Learning
The core challenge in RAG tasks is to generate retrieval queries that
align closely with user intent, thereby maximizing the recall rate for
private-domain documents. Conventional supervised fine-tuning
(SFT) relies heavily on manually annotated "query–document" pairs,
which are scarce and costly to produce in practice. To overcome this,
MoLER employs an unsupervised RL approach using the GRPO-
based algorithm to directly optimize the probability distribution
of retrieval strategies, with retrieval recall serving as the reward
signal.
To assess the generalization capability of the proposed frame-
work across different RL paradigms, we conduct a comparative
analysis between GRPO and its representative variant, Dr.GRPO.3.3.1 GRPO and Dr.GRPO Algorithms.GRPO is a policy optimiza-
tion algorithm for multi-candidate output scenarios. It replaces
the independent critic model in PPO (Proximal Policy Optimiza-
tion) [ 21] with group-wise relative advantage estimation, signifi-
cantly reducing computational overhead. However, GRPO suffers
from two types of bias:
(1)Response length bias:The normalisation term1 /|𝑜𝑖|re-
duces the gradient penalty for longer incorrect responses
(negative advantage) while amplifying the positive gradi-
ent for shorter correct ones (positive advantage). This bias
encourages increasingly verbose outputs, especially when
responses are incorrect.
(2)Problem difficulty bias:The advantage estimate ˆ𝐴𝑖=
𝑅𝑖−mean(𝑅)
std(𝑅)assigns higher optimization weight to low-
variance problems (i.e., extremely easy or extremely dif-
ficult tasks).
The GRPO objective function is:
JGRPO(𝜋𝜃)=E𝑞,{𝑜}1
𝐺𝐺∑︁
𝑖=1∑︁
𝑐1
|𝑜𝑖,𝑐||𝑜𝑖,𝑐|∑︁
𝑡=1 
𝜋𝜃
𝜋𝜃oldˆ𝐴𝑖−𝛽𝐷 KL[𝜋𝜃∥𝜋ref]!
.(4)
The objective function is formulated to optimize the policy 𝜋𝜃
by maximizing the expected cumulative advantage across multi-
ple query generation and pre-answering. Here,𝑐indexes the com-
ponents of the retrieval pipeline, specifically the MQR and CQE
prompts, which correspond to distinct stages in the query augmen-
tation process. For each sample𝑖and component𝑐,𝑜 𝑖,𝑐represents
the model’s output generated under prompt 𝑐, such as sub-queries
(for MQR) or pseudo-passages (for CQE).
To mitigate the response length and problem difficulty biases in
GRPO and achieve unbiased optimization, Dr.GRPO [ 16] removes
the1/|𝑜𝑖|term and the std(𝑅) normalization term, thereby decou-
pling gradient updates from response length while maintaining

Hao Lin1, Peitong Xie1, Jingxue Chen, Jie Lin, Qingkun Tang†, and Qianchun Lu
/uni00000014/uni00000013 /uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013 /uni0000001a/uni00000013 /uni0000001b/uni00000013 /uni0000001c/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000014/uni00000014/uni00000013 /uni00000014/uni00000015/uni00000013 /uni00000014/uni00000016/uni00000013 /uni00000014/uni00000017/uni00000013 /uni00000014/uni00000018/uni00000013
/uni00000033/uni00000052/uni0000004f/uni0000004c/uni00000046/uni0000005c/uni00000003/uni0000004c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000056/uni00000057/uni00000048/uni00000053/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000015/uni00000018/uni00000013/uni00000016/uni00000013/uni00000013/uni00000016/uni00000018/uni00000013/uni00000017/uni00000013/uni00000013/uni00000017/uni00000018/uni00000013/uni00000018/uni00000013/uni00000013/uni00000018/uni00000018/uni00000013/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000048/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
(a) NFCORPUS
/uni00000014/uni00000013 /uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013 /uni0000001a/uni00000013 /uni0000001b/uni00000013 /uni0000001c/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000014/uni00000014/uni00000013 /uni00000014/uni00000015/uni00000013 /uni00000014/uni00000016/uni00000013 /uni00000014/uni00000017/uni00000013 /uni00000014/uni00000018/uni00000013
/uni00000033/uni00000052/uni0000004f/uni0000004c/uni00000046/uni0000005c/uni00000003/uni0000004c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000056/uni00000057/uni00000048/uni00000053/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000015/uni00000018/uni00000013/uni00000016/uni00000013/uni00000013/uni00000016/uni00000018/uni00000013/uni00000017/uni00000013/uni00000013/uni00000017/uni00000018/uni00000013/uni00000018/uni00000013/uni00000013/uni00000026/uni00000052/uni00000050/uni00000053/uni0000004f/uni00000048/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000045/uni0000000e/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c (b) SCIFACT
Figure 3: Comparison of response lengths between GRPO and Dr.GRPO after multi-step training using Qwen3-0.6B and
Qwen3-1.7B models on (a) NFCORPUS and (b) SCIFACT datasets. The results show that Dr.GRPO consistently generates shorter
completion lengths across both model scales and datasets.
consistent optimization weights across problems. The revised ob-
jective function is:
JDr.GRPO∼E𝑞,{𝑜}1
𝐺𝐺∑︁
𝑖=1∑︁
𝑐|𝑜𝑖,𝑐|∑︁
𝑡=1 
𝜋𝜃
𝜋𝜃oldˆ𝐴𝑖−𝛽𝐷 KL[𝜋𝜃∥𝜋ref]!
,(5)
where ˜𝐴𝑖=𝑅𝑖−mean(𝑅).
The framework employs the MSLF strategy during RL train-
ing to reduce computational costs. While MMLF is utilized during
inference to maximize retrieval effectiveness, the training phase
benefits from MSLF’s streamlined approach of generating a single
pseudo-passage from multiple sub-queries. This design reduces the
number of model interactions from 𝑛+1(where𝑛denotes the num-
ber of query expansions) to 2 during policy rollouts, significantly
accelerating training without compromising final performance. Em-
pirical results demonstrate that MSLF-based RL training can still
effectively enhance MMLF’s retrieval capabilities during inference.
4 EXPERIMENTS
This section systematically evaluates the performance of the pro-
posed MoLER framework on private retrieval tasks. We begin by
establishing the experimental setup, including the selection of base
models, datasets, and evaluation metrics that form the foundation
of our empirical analysis. Our evaluation then proceeds with the
main results, comparing MoLER against traditional RAG and vari-
ous prompting strategies to demonstrate its overall performance
advantages across different model scales. The analysis examines key
characteristics including comparative analysis of GRPO variants,scalability of query expansion, and the effectiveness of nonthinking
versus thinking modes. Subsequently, we conduct comprehensive
ablation studies to dissect the contributions of individual compo-
nents within the MoLER framework. These studies systematically
investigate: (1) the impact of different retrieval fusion strategies
(MSLF vs. MMLF); (2) the contribution of various retrieval augmen-
tation methods and their combinations; and (3) the enhancement of
retrieval capabilities brought by different continual pre-training ap-
proaches, specifically comparing our proposed MoL method against
conventional domain-specific training strategies.
4.1 Experimental Setup and Evaluation Metrics
4.1.1 Base Models and Embedding.Considering the dual require-
ments of efficiency and performance in RAG scenarios, we selected
the open-source Qwen3-0.6B and Qwen3-1.7B models as our base
models to compare performance across different parameter scales.
Furthermore, the embedding model used in all experiments is Ope-
nAI’s text-embedding-ada-002 [19].
4.1.2 Datasets and Evaluation Metrics.We selected two datasets
from the BEIR benchmark [ 24] for methodological evaluation: NF-
CORPUS and SCIFACT. NFCORPUS focuses on biomedical question-
answering retrieval, comprising 3,633 documents, 2,590 training
queries, and 323 test queries, with an average of 38.2 relevant docu-
ments per query and an average query length of 3.3 words. SCIFACT
is oriented toward scientific fact-checking, consisting of 5,183 paper
abstracts, 809 training queries, and 300 test queries, with an average
of only 1.1 relevant documents per query and an average query
length of 12.37 words. Following the setup in [ 14], NFCORPUS is
evaluated using Recall@1k and nDCG@10, while Recall@10 and

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
Table 4: Comparison of base models and MoLER-trained models under nonthinking and thinking Modes. The results demon-
strate that MoLER (Qwen3-0.6B/1.7B+MoL+Dr.GRPO) achieves comparable performance in both reasoning modes, with marginal
gains in thinking mode across all metrics.
MethodNFCORPUS SCIFACT
Recall@1k nDCG@10 Recall@10 nDCG@10
Qwen3-0.6B+nonthinking 57.73 22.02 63.21 49.80
Qwen3-0.6B+thinking 58.44 22.60 67.68 51.82
Qwen3-0.6B+MoL+Dr.GRPO+nonthinking 59.55 23.18 72.96 55.43
Qwen3-0.6B+MoL+Dr.GRPO+thinking59.72 23.61 73.10 56.30
Qwen3-1.7B+nothinking 59.29 23.36 72.26 57.76
Qwen3-1.7B+thinking 59.49 24.25 73.19 58.49
Qwen3-1.7B+MoL+Dr.GRPO+nothinking 60.19 24.38 77.47 60.90
Qwen3-1.7B+MoL+Dr.GRPO+thinking60.57 24.65 77.69 61.57
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000019 /uni0000001b
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni00000028/uni0000005b/uni00000053/uni00000044/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000018/uni0000001a/uni00000011/uni00000013/uni00000018/uni0000001a/uni00000011/uni00000018/uni00000018/uni0000001b/uni00000011/uni00000013/uni00000018/uni0000001b/uni00000011/uni00000018/uni00000018/uni0000001c/uni00000011/uni00000013/uni00000018/uni0000001c/uni00000011/uni00000018/uni00000019/uni00000013/uni00000011/uni00000013/uni00000019/uni00000013/uni00000011/uni00000018/uni00000019/uni00000014/uni00000011/uni00000013/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000025/uni00000010/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni00000014 /uni00000015/uni00000016/uni00000017/uni00000019/uni0000001b/uni00000018/uni0000001b/uni00000019/uni00000013
(a) NFCORPUS
/uni0000001a/uni00000013/uni0000001a/uni00000015/uni0000001a/uni00000017/uni0000001a/uni00000019/uni0000001a/uni0000001b
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000019 /uni0000001b
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni00000028/uni0000005b/uni00000053/uni00000044/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000019/uni00000015/uni00000019/uni00000016/uni00000019/uni00000017/uni00000019/uni00000018/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000014/uni00000011/uni0000001a/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni0000001a/uni00000013/uni0000001a/uni00000018
/uni00000014 /uni00000015 /uni00000016/uni00000017 /uni00000019/uni0000001b/uni00000019/uni00000015/uni00000011/uni00000018/uni00000019/uni00000018/uni00000011/uni00000013
 (b) SCIFACT
Figure 4: Performance comparison between MoLER-enhanced models and base models across different query expansion counts
on (a) NFCORPUS and (b) SCIFACT datasets. Results demonstrate that Qwen3-0.6B+MoLER achieves performance comparable
to Qwen3-1.7B base model on SCIFACT, highlighting MoLER’s effectiveness in enabling smaller models to achieve larger model
performance. The logarithmic scaling subplots (shown as insets in the main figure) reveal near-linear relationships between
recall improvements and query expansion counts, confirming adherence to scaling laws.
nDCG@10 are adopted for SCIFACT to achieve comparable recall
levels. These two datasets cover distinct domains and query com-
plexities, enabling an effective evaluation of the retrieval system’s
robustness in a zero-shot setting.
4.1.3 MoL Continual Pre-training.During the continual learning
phase, we employed the PEFT framework [ 17] to conduct MoL
training on the models using a domain-specific dataset and the
general-purpose dataset Light-R1 [ 26]. To align with LLM pre-
training principles, we transformed Light-R1 into unsupervised text
using a dialogue template (e.g., concatenating turns with speakerroles as natural text sequences). This approach adheres to the to-
ken prediction objective of LLMs, ensuring the training process
remains consistent with their pre-training paradigms. Moreover,
the inclusion of dialogue data helps preserve the model’s conversa-
tional understanding, which is critical for RAG systems involving
dynamic user interactions. To improve parameter efficiency, we
utilize Low-Rank Adaptation (LoRA) with a rank of 64 [ 7]. Through-
out the MoL training, the model’s context window is fixed at 8192
tokens to ensure effective coverage of document knowledge. Other
hyperparameters are detailed in Appendix A.1.

Hao Lin1, Peitong Xie1, Jingxue Chen, Jie Lin, Qingkun Tang†, and Qianchun Lu
Table 5: Comparison of MSLF and MMLF fusion strategies on NFCORPUS and SCIFACT datasets with fixed base models
(Qwen3-0.6B/Qwen3-1.7B) and training strategies (MoL+Dr.GRPO/MoL). The results demonstrate that MMLF significantly
outperforms MSLF in key metrics such as Recall@1k and Recall@10 across different model scales and training configurations,
validating the effectiveness of MMLF in enhancing retrieval performance.
MethodNFCORPUS SCIFACT
Recall@1k nDCG@10 Recall@10 nDCG@10
Qwen3-0.6B+MoL+MSLF 57.23 22.19 63.24 47.83
Qwen3-0.6B+MoL+MMLF 57.31 22.51 64.00 49.90
Qwen3-0.6B+MoL+Dr.GRPO+MSLF 58.02 23.6270.77 53.82
Qwen3-0.6B+MoL+Dr.GRPO+MMLF59.7223.18 72.96 55.43
Qwen3-1.7B+MoL+MSLF 58.37 23.82 70.34 54.19
Qwen3-1.7B+MoL+MMLF 59.41 24.14 73.22 57.29
Qwen3-1.7B+MoL+Dr.GRPO+MSLF 59.2724.5975.16 57.45
Qwen3-1.7B+MoL+Dr.GRPO+MMLF60.1924.38 77.47 60.90
/uni00000019/uni0000001b/uni00000019/uni0000001c/uni0000001a/uni00000013/uni0000001a/uni00000014/uni0000001a/uni00000015/uni0000001a/uni00000016/uni0000001a/uni00000017
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000019 /uni0000001b
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni00000028/uni0000005b/uni00000053/uni00000044/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000018/uni00000019/uni00000018/uni0000001a/uni00000018/uni0000001b/uni00000018/uni0000001c/uni00000019/uni00000013/uni00000019/uni00000014/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni0000000e/uni00000030/uni00000036/uni0000002f/uni00000029/uni0000000b/uni00000031/uni00000029/uni00000026/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni0000000b/uni00000031/uni00000029/uni00000026/uni0000000c/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni0000000e/uni00000030/uni00000036/uni0000002f/uni00000029/uni0000000b/uni00000036/uni00000026/uni0000002c/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni0000000b/uni00000036/uni00000026/uni0000002c/uni0000000c/uni0000001a/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000015/uni00000011/uni00000018
/uni00000014 /uni00000015/uni00000016/uni00000017/uni00000019/uni0000001b/uni00000018/uni0000001a/uni00000011/uni00000018/uni00000019/uni00000013/uni00000011/uni00000013
Figure 5: Performance comparison between MMLF and
MSLF fusion strategies across varying query expansion
counts on NFCORPUS and SCIFACT datasets using Qwen3-
0.6B+MoLER models. The results demonstrate that MMLF
exhibits logarithmic scaling in performance with increased
expansion counts, while MSLF shows minimal improve-
ment, indicating its limited scalability. Notably, training with
MSLF (which uses only 3 fixed expansions) does not degrade
MMLF’s scaling capability during inference. This demon-
strates the framework’s effectiveness: the MSLF-based train-
ing strategy maintains computational efficiency during RL
optimization while preserving MMLF’s end-to-end scalabil-
ity for optimal retrieval performance.
4.1.4 GRPO-based post-training.In the wake of MoL CPT, a GRPO-
based post-training phase is introduced with the objective of further
aligning the model’s generative priors with the final objectives
of the downstream retrieval task. The focal point of this stage isthe construction of a reward signal that directly reflects retrieval
performance. Specifically, the MSLF strategy (see Section 3.1 for
details) is employed to guide the model in generating a series of
query expansions for the original query, and further expanding
these expanded queries into a pre-answer passage for the question.
This passage, in conjunction with the original query, is then utilized
to compute the cosine similarity between their embeddings and
those of the original documents for the purpose of ranking. The
results are consolidated into a final document list via RRF. The
recall rate of relevant documents in this fused list is adopted as the
core metric for the reward function. Further hyperparameters of the
GRPO framework, incorporating the particular design and rollout
Configuration of the reward function, are outlined in Appendix
A.2. Furthermore, in the MSLF and MMLF strategies, the number of
query expansions 𝑛is explicitly set to 3 to balance computational
efficiency and retrieval effectiveness.
4.1.5 Evaluation Protocol and Implementation Details.To ensure a
fair comparison, enhancement techniques such as pseudo-relevance
feedback, mutual verification [ 11], or few-shot prompting are not
introduced during the evaluation. Unless otherwise specified, all
models are run in nonthinking mode during both the RL training
and testing phases, adopting the officially recommended decod-
ing hyperparameters for Qwen3: a temperature of 0.7, top_p of
0.8, and top_k of 20 [ 31]. Additionally, MMLF is used for query
augmentation Unless specified.
4.2 Main Results
4.2.1 Overall Performance Comparison.We validated the effective-
ness of MoLER by comparing it against four categories of baseline
methods: 1)Raw Query, which directly uses the initial query state-
ment; 2)Query2Doc (Q2D) [ 25], which expands the query into a
paragraph for retrieval; 3)Chain of Thought (CoT) [ 9], which
transforms the query into an answer and its reasoning chain; and 4)
LangChain Multi-Query Retriever (LC-MQR), which generates
multiple sub-queries and fuses the results using RRF.
Table 3 presents the overall experimental results. The data
show that MoLER significantly enhances retrieval performance

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
Table 6: Ablation study of retrieval enhancement strategies on base models and MoLER framework. This table presents a
comparative analysis of different retrieval augmentation methods (Raw Query, LC-MQR, Q2D) and their integration with
the MoLER framework (MoL+Dr.GRPO+MSLF) on Qwen3-0.6B and Qwen3-1.7B models across the NFCORPUS and SCIFACT
datasets. Key findings include: (1) The MoLER framework (MoL+Dr.GRPO) consistently improves retrieval metrics (Recall@1k,
Recall@10, nDCG@10) over baseline methods across both model scales; (2) Combining MoL+Dr.GRPO with MMLF fusion
achieves optimal performance, demonstrating that the synergistic design of "query expansion + question pre-answering"
effectively compensates for individual strategy limitations. The results validate MoLER’s superiority in enhancing retrieval
robustness through end-to-end optimization of multi-perspective query-document interactions.
MethodNFCORPUS SCIFACT
Recall@1k nDCG@10 Recall@10 nDCG@10
Raw Query 52.95 21.69 59.82 46.57
Qwen3-0.6B+LC-MQR 54.27 20.98 60.84 46.79
+MoL+Dr.GRPO 57.59 22.35 67.59 51.30
Qwen3-0.6B+Q2D 56.26 19.49 61.38 48.01
+MoL+Dr.GRPO 58.11 20.41 70.44 54.14
Qwen3-0.6B+MMLF 57.73 22.02 63.21 49.80
+MoL+Dr.GRPO59.72 23.18 72.96 55.43
Qwen3-1.7B+LC-MQR 53.93 22.29 66.89 51.08
+MoL+Dr.GRPO 55.16 22.51 68.97 53.16
Qwen3-1.7B+Q2D 58.70 22.58 73.41 59.10
+MoL+Dr.GRPO 59.31 23.32 75.20 59.21
Qwen3-1.7B+MMLF 59.10 23.59 73.52 57.10
+MoL+Dr.GRPO60.19 24.38 77.47 60.90
across Qwen3 models of different scales. Notably, Qwen3-
1.7B+MoL+GRPO achieves the best performance on both tasks, with
its recall and nDCG metrics ranking highest among all baselines.
This combination even surpasses the Qwen3-32B+MMLF model,
which has 18.82 times the number of parameters. Particularly on the
recall metric, MoLER shows an average improvement of 0.49% over
its closest competitor, Qwen3-32B+MMLF, indicating that applying
MoLER can produce substantial gains on smaller models.
4.2.2 GRPO vs Dr.GRPO Algorithm Comparison.As shown in Fig-
ure 3, we compared the token consumption of different model sizes
and RL algorithms during the training phase. After applying a mov-
ing average with a window size of 8, it is clearly observable that the
token lengths during the Dr.GRPO training process are shorter and
more stable. Given Dr.GRPO’s significant advantage in response
length, and although its recall and nDCG performance may not
consistently surpass that of GRPO, possibly due to the nature of our
tasks, and we ultimately select Dr.GRPO as the primary reference
algorithm for subsequent experiments.
4.2.3 Scalability Analysis of Query Expansion.Figure 4 presents the
performance on MoLER-enhanced models and base models across
different query expansion counts in MMLF. The results demonstrate
two key findings. First, when the x-axis is scaled logarithmically,
both Figure 4 (a) and Figure 4 (b) exhibit approximately linear
growth patterns in recall performance. This linear relationship un-
der log-scale suggests that recall improvement follows scaling laws
with respect to the number of query expansions, where performance
gains adhere to a power-law relationship. The scaling behavior is
/uni00000019/uni0000001c/uni0000001a/uni00000013/uni0000001a/uni00000014/uni0000001a/uni00000015/uni0000001a/uni00000016/uni0000001a/uni00000017
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000019 /uni0000001b
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni00000028/uni0000005b/uni00000053/uni00000044/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000056/uni00000018/uni0000001a/uni00000018/uni0000001b/uni00000018/uni0000001c/uni00000019/uni00000013/uni00000019/uni00000014/uni00000035/uni00000048/uni00000046/uni00000044/uni0000004f/uni0000004f
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000026/uni00000028/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000031/uni00000029/uni00000026/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni00000003/uni0000000b/uni00000031/uni00000029/uni00000026/uni0000000c/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni0000000e/uni00000026/uni00000028/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000036/uni00000026/uni0000002c/uni0000000c
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025/uni00000025/uni0000000e/uni00000030/uni00000052/uni0000002f/uni00000028/uni00000035/uni00000003/uni0000000b/uni00000036/uni00000026/uni0000002c/uni0000000c/uni0000001a/uni00000013/uni0000001a/uni00000015/uni0000001a/uni00000017
/uni00000014 /uni00000015 /uni00000016 /uni00000017 /uni00000019 /uni0000001b/uni00000018/uni0000001b/uni00000019/uni00000013
Figure 6: Impact of query expansion count on MoLER ver-
sus CE+Dr.GRPO performance across NFCORPUS and SCI-
FACT datasets. The figure compares Recall performance as
a function of the number of query expansions for Qwen3-
0.6B models trained with MoLER and CE+Dr.GRPO methods.
Results demonstrate that MoLER consistently outperforms
CE+Dr.GRPO across varying expansion counts, with both
methods showing logarithmic scaling behavior.

Hao Lin1, Peitong Xie1, Jingxue Chen, Jie Lin, Qingkun Tang†, and Qianchun Lu
Table 7: Comparison of pre-training methods with varying training epochs on retrieval performance. This table evaluates the
impact of excluding general-domain corpora versus domain-aware MoL training with 2/4 epochs on retrieval effectiveness
under Dr.GRPO. Key findings include: (1) Increasing MoL training from 2 to 4 epochs consistently improves Recall@1k and
nDCG@10; (2) The optimal performance is achieved with MoL+Dr.GRPO after 4 training epochs, demonstrating that MoLER
enhances the model’s ability to learn domain-specific patterns while maintaining robustness through the dual-loss architecture.
MethodNFCORPUS SCIFACT
Recall@1k nDCG@10 Recall@10 nDCG@10
Qwen3-0.6B 57.73 22.02 63.21 49.80
+Dr.GRPO 59.2223.6371.79 55.12
+CE(4 epoch) 57.02 22.44 64.00 49.90
+CE(4 epoch)+Dr.GRPO 59.32 22.84 72.17 55.19
+MoL(4 epoch) 57.31 22.51 64.83 50.76
+MoL(2 epoch)+Dr.GRPO 59.28 23.18 72.12 54.99
+MoL(4 epoch)+Dr.GRPO59.7223.18 72.96 55.43
consistent across both NFCORPUS and SCIFACT datasets, demon-
strating MoLER’s scalable characteristics in different retrieval sce-
narios.
Second, on the SCIFACT dataset shown in Figure 4 (b), Qwen3-
0.6B+MoLER achieves recall performance comparable to the Qwen3-
1.7B base model. This indicates that MoLER enables smaller param-
eter models to achieve performance levels similar to larger base
models.
4.2.4 Nonthinking vs Thinking Mode Analysis.The experimental
results in Table 4 demonstrate that both base models and MoLER-
trained models exhibit comparable performance in nonthinking
and thinking modes, with marginal improvements observed in the
latter. For instance, on the MoLER-trained Qwen3-0.6B model, the
Recall@1k metric increases by only 0.17% (from 59.55 to 59.72)
and nDCG@10 by 0.43% (from 23.18 to 23.61) when switching
from nothink to thinking mode. These findings suggest two key
insights: (1) CoT reasoning does not substantially enhance retrieval
performance for knowledge gaps in the LLM, as the marginal im-
provements fall well below the gains achieved by MoLER’s retrieval
optimization strategies. (2) Adopting nonthinking mode for train-
ing and inference is computationally efficient and practical, as the
minimal performance trade-off is outweighed by reduced latency
and resource consumption.
4.3 Ablation Study
4.3.1 Impact of Different Retrieval Fusion Strategies.To examine
the effectiveness of the proposed MMLF and MSLF strategies, we
conduct comparative experiments under Dr.GRPO optimization.
The results are presented in Table 5.
Experimental results confirm that employing the MSLF strategy
during the RL phase is an effective approach to enhancing the perfor-
mance of MMLF. Specifically, after applying the Dr.GRPO algorithm,
which uses the MSLF-derived recall as its optimization objective,
both the Qwen3-0.6B and Qwen3-1.7B models demonstrate signifi-
cant improvements in recall and nDCG metrics for both MSLF and
MMLF across the two datasets. This finding not only validates the
effectiveness of this RL strategy but also proves that MoLER can ef-
fectively boost the model’s performance in complex retrieval tasks.Furthermore, in line with expectations, MMLF’s performance met-
rics consistently surpass those of MSLF, both before and after the
RL phase. The reasoning is twofold: MSLF is selected during the RL
phase primarily to improve training efficiency, whereas MMLF, by
leveraging its "pre-answer fusion" strategy, is able to explore associ-
ated documents from more diverse perspectives, thereby achieving
superior final performance.
Figure 5 provides additional insights into the scaling behavior
of these fusion strategies across different query expansion counts.
The results reveal distinct scaling characteristics between MMLF
and MSLF approaches. MMLF demonstrates clear scaling law adher-
ence, with performance improving logarithmically as the number
of query expansions increases on both NFCORPUS and SCIFACT
datasets. In contrast, MSLF shows performance fluctuations without
a clear scaling trend across different expansion counts. This behav-
ioral difference can be attributed to MSLF’s architectural constraint
of utilizing only single-passage semantic similarity computation,
which inherently limits its sensitivity to variations in query ex-
pansion count. The observed scaling patterns reinforce our design
choice of using MSLF during the RL training phase while deploy-
ing MMLF for final inference, as this combination optimizes both
training efficiency and inference performance.
4.3.2 Comparison of Retrieval Augmentation Methods.To further
analyze the contributions of different retrieval augmentation strate-
gies, we compared the effects of Raw Query, LC-MQR, Q2D, and
their combinations with MoL+Dr.GRPO. The results are shown in
Table 6. The main findings are as follows.
(1)For the Qwen3-0.6B model on the NFCORPUS dataset, us-
ing either LC-MQR or Q2D alone resulted in a decrease in
nDCG, suggesting that relying on a single augmentation
strategy may introduce bias. In contrast, this phenomenon
is not observed with the Qwen3-1.7B model. We specu-
late this may be due to the smaller parameter count of the
Qwen3-0.6B model, leading to weaker generation capabili-
ties for query expansion and question pre-answering.
(2)Regardless of the retrieval augmentation method used, the
introduction of MoL+Dr.GRPO consistently led to signifi-
cant improvements in retrieval performance, manifesting

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
/uni00000014/uni00000013 /uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013 /uni0000001a/uni00000013 /uni0000001b/uni00000013 /uni0000001c/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000014/uni00000014/uni00000013 /uni00000014/uni00000015/uni00000013 /uni00000014/uni00000016/uni00000013 /uni00000014/uni00000017/uni00000013 /uni00000014/uni00000018/uni00000013
/uni00000033/uni00000052/uni0000004f/uni0000004c/uni00000046/uni0000005c/uni00000003/uni0000004c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000056/uni00000057/uni00000048/uni00000053/uni00000013/uni00000011/uni00000018/uni00000019/uni00000013/uni00000011/uni00000018/uni0000001b/uni00000013/uni00000011/uni00000019/uni00000013/uni00000013/uni00000011/uni00000019/uni00000015/uni00000013/uni00000011/uni00000019/uni00000017/uni00000037/uni00000055/uni00000044/uni0000004c/uni00000051/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000035/uni00000048/uni0000005a/uni00000044/uni00000055/uni00000047
/uni0000000e/uni00000030/uni00000052/uni0000002f/uni0000000b/uni00000028/uni00000033/uni00000017/uni0000000c/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni0000000e/uni00000030/uni00000052/uni0000002f/uni0000000b/uni00000028/uni00000033/uni00000015/uni0000000c/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni0000000e/uni00000026/uni00000028/uni0000000b/uni00000028/uni00000033/uni00000017/uni0000000c/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c
/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000052/uni00000055/uni0000004c/uni0000004a/uni0000004c/uni00000051/uni00000044/uni0000004f/uni0000000c/uni0000000e/uni00000030/uni00000052/uni0000002f/uni0000000b/uni00000028/uni00000033/uni00000017/uni0000000c/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni0000000e/uni00000030/uni00000052/uni0000002f/uni0000000b/uni00000028/uni00000033/uni00000015/uni0000000c/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni0000000e/uni00000026/uni00000028/uni0000000b/uni00000028/uni00000033/uni00000017/uni0000000c/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
/uni0000000e/uni00000027/uni00000055/uni00000011/uni0000002a/uni00000035/uni00000033/uni00000032/uni00000003/uni0000000b/uni00000056/uni00000050/uni00000052/uni00000052/uni00000057/uni0000004b/uni00000048/uni00000047/uni0000000c
Figure 7: Convergence trends of Dr.GRPO reward curves on
Qwen3-0.6B with varying pre-training strategies on NFCOR-
PUS. The plot compares the RL (Dr.GRPO) reward progres-
sion during training, focusing on the impact of removing
general corpora from MoL and adjusting the number of MoL
training epochs (2 vs. 4). A moving average with a window
size of 8 is applied for clarity. The results demonstrate that
MoL pre-training with 4 epochs achieves the most stable and
superior reward trajectory, highlighting its effectiveness in
balancing domain knowledge acquisition and generalization.
as higher Recall and nDCG, and demonstrating a consistent
advantage over Raw Query.
(3)When further combined with MMLF, performance reached
its optimum. This validates that the "query expansion +
question pre-answering" design, when used jointly, can
compensate for the shortcomings of individual strategies,
thereby achieving the best retrieval effectiveness.
4.3.3 Comparison of Different CPT Methods.We further compared
the impact of different CPT methods on the model’s retrieval ca-
pability. The experimental results, shown in Table 7, include com-
parisons of CE, MoL, and MoL with different training epochs. In
particular, the CE method utilizes only domain-specific documents
to be retrieved.
The experimental results in Table 7 demonstrate that MoLER
exhibits robustness to variations in pre-training strategies, consis-
tently achieving stable improvements across different CPT methods
and the proposed MoL framework outperforms CE in both recall and
nDCG metrics. This advantage is particularly pronounced when
training is constrained to the same 4 epochs, underscoring the
efficiency of MoL’s dual-loss design in optimizing retrieval perfor-
mance without compromising model versatility.
On the other hand, all models showed significant performance
improvements after the introduction of Dr.GRPO, validating the
effectiveness of the RL phase in improving retrieval capabilities. For
MoL, training for 4 epochs performed better than for 2 epochs,which further corroborates the rationality of our chosen pre-
training setup.
Additionally, Figure 6 provides further insights into the scal-
ing behavior of MoLER compared to CE+Dr.GRPO across different
numbers of query expansions. The results demonstrate that MoLER
maintains consistent performance advantages over CE+Dr.GRPO
across all expansion counts on both datasets. Notably, both methods
exhibit similar logarithmic scaling patterns, confirming that the per-
formance improvements follow theoretical scaling laws regardless
of the pre-training strategy employed. This finding reinforces that
MoLER’s superior performance stems not only from its domain-
aware pre-training approach but also from its inherent scalability
characteristics, making it a robust choice for retrieval enhancement
across different operational scales.
Figure 7 displays the convergence curves of different pre-training
methods on the NFCORPUS dataset. For ease of observation, we
compare the results after applying a moving average with a window
size of 8. It can be observed that applying Dr.GRPO directly to the
base model yields the most limited improvement. This is because,
without the support of domain-specific knowledge, the model needs
to spend more time exploring effective strategies. In contrast, after
introducing MoL, whether for 2 or 4 epochs, the model exhibits
a superior final performance. This indicates that MoL not only
preserves general-purpose capabilities but also provides a better
foundation for RL.
5 DISCUSSION AND LIMITATIONS
Our experimental results demonstrate that the proposed MoLER
framework significantly enhances retrieval performance in RAG
systems. The two-stage approach integrating CPT with MoL and
RL with GRPO proves highly effective.
Despite the significant achievements of MoLER, several criti-
cal limitations and avenues for future research warrant discus-
sion. Firstly, acquiring and curating high-quality, human-annotated
domain-specific data for the RL phase can be costly and time-
consuming, particularly in highly specialized or proprietary do-
mains. While MoLER effectively leverages such data, reducing re-
liance on extensive manual annotation or exploring semisuper-
vised/unsupervised methods for domain knowledge acquisition
will be important research directions.
Secondly, this study focuses mainly on text-based RAG systems.
Extending MoLER to handle multimodal information will open new
avenues for more comprehensive knowledge retrieval. Furthermore,
exploring the integration of MoLER with structured knowledge
bases or knowledge graphs could enhance its reasoning capabili-
ties and provide more precise and verifiable answers beyond pure
document retrieval. The principles of domain-aware learning and
RL-based optimization developed in MoLER can be adapted to these
complex data environments, thereby further solidifying its contri-
bution to large-scale information management and retrieval within
database systems.
6 CONCLUSION
In this paper, we address a critical limitation in existing RAG sys-
tems: the disconnect between query augmentation strategies and
the direct optimization of retrieval performance. To bridge this

Hao Lin1, Peitong Xie1, Jingxue Chen, Jie Lin, Qingkun Tang†, and Qianchun Lu
Table 8: Training Setup for MoL
Hyperparameter Value
global batch size 128
learning rate 2e-4
LoRA rank 64
weight decay 0.1
gap, we introduced MoLER, a novel two-stage framework that en-
hances informative retrieval for domain-specific tasks. MoLER first
employs MoL for CPT, enabling the model to acquire specialized
knowledge without suffering from catastrophic forgetting. Subse-
quently, it utilizes GRPO-based RL to fine-tune the model’s query
and passage generation policy, directly maximizing document re-
call.
Our extensive experiments on the NFCORPUS and SCIFACT
datasets demonstrate the effectiveness of the MoLER framework.
The results show that even a compact 1.7B parameter model
equipped with MoLER can significantly outperform strong base-
lines, including a much larger 32B parameter model using con-
ventional augmentation techniques. This underscores MoLER’s
parameter efficiency and its ability to unlock the full potential of
smaller models for complex retrieval tasks. Key contributions of
our work are threefold: (1) deepen the model’s comprehension of
retrieved documents through continual learning; (2) achieve more
efficient query enhancement via multi-query expansion and LLM
pre-answering; (3) further activate the LLM’s inherent knowledge
reservoir through RL to enable effective query augmentation. Ulti-
mately, MoLER provides a robust and scalable solution for building
highly effective and efficient domain-adaptive RAG systems, paving
the way for more powerful and knowledgeable AI applications.
A APPENDIX
A.1 MoL Continual Training Hyperparameters
Table 8 presents the hyperparameter configuration adopted during
the continual training of MoL. The choice of these hyperparame-
ters aims to ensure training stability while enhancing the model’s
generalization ability in cross-task transfer.
A.2 GRPO-based Post-Training
Hyperparameters
In the post-training phase, we applied the GRPO-based framework
to further refine the retrieval and generation performance of the
model. Table 9 summarizes the major hyperparameter settings,
emphasizing robustness under adversarial scenarios. Meanwhile,
Table 10 provides the rollout configuration during inference, cover-
ing parameters such as parallelism strategy, number of generations,
and temperature.
REFERENCES
[1] J. Bhogal, A. Macfarlane, and P. Smith. 2007. A review of ontology based query
expansion.Information Processing & Management43, 4 (2007), 866–886. https:
//doi.org/10.1016/j.ipm.2006.09.003
[2]Jingxue Chen, Qingkun Tang, Qianchun Lu, and Siyuan Fang. 2025. MoL for
LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving
General Capabilities. arXiv:2505.12043 [cs.CL] https://arxiv.org/abs/2505.12043Table 9: Training Setup for GRPO
Hyperparameter Value
global batch size 64
gradient learning rate 1e-4
LoRA rank 64
weight decay 0.1
Table 10: Rollout Configuration for GRPO
Parameter Value
rollout backend vLLM
tensor parallel size 1
data parallel size 1
num generation 8
max completion length 8192
temperature 0.9
[3]Shufan Chen, He Zheng, and Lei Cui. 2025. When and How to Augment Your
Input: Question Routing Helps Balance the Accuracy and Efficiency of Large
Language Models. InFindings of the Association for Computational Linguistics:
NAACL 2025, Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Association for
Computational Linguistics, Albuquerque, New Mexico, 3621–3634. https://doi.
org/10.18653/v1/2025.findings-naacl.200
[4] Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. 2009. Reciprocal
rank fusion outperforms condorcet and individual rank learning methods. In
Proceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval(Boston, MA, USA)(SIGIR ’09). Association
for Computing Machinery, New York, NY, USA, 758–759. https://doi.org/10.
1145/1571941.1572114
[5]Andrea Cossu, Tinne Tuytelaars, Antonio Carta, Lucia Passaro, Vincenzo
Lomonaco, and Davide Bacciu. 2022. Continual Pre-Training Mitigates For-
getting in Language and Vision. arXiv:2205.09357 [cs.LG] https://arxiv.org/abs/
2205.09357
[6]Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and et al. 2025. DeepSeek-
R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
arXiv preprint arXiv:2501.12948(2025). https://arxiv.org/abs/2501.12948 Code:
https://github.com/deepseek-ai/DeepSeek-R1.
[7] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, Weizhu Chen, et al .2022. Lora: Low-rank adaptation of large
language models.ICLR1, 2 (2022), 3.
[8] Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. 2019. ClinicalBERT: Model-
ing Clinical Notes and Predicting Hospital Readmission.ArXivabs/1904.05342
(2019). https://api.semanticscholar.org/CorpusID:119308351
[9] Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui Wang, and Michael Bender-
sky. 2023. Query expansion by prompting large language models.arXiv preprint
(2023). arXiv:2305.03653 https://arxiv.org/abs/2305.03653
[10] Pengyue Jia, Yiding Liu, Xiangyu Zhao, Xiaopeng Li, Changying Hao, Shuaiqiang
Wang, and Dawei Yin. 2024. MILL: Mutual Verification with Large Language
Models for Zero-Shot Query Expansion. InProceedings of the 2024 Conference of
the North American Chapter of the Association for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez,
and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico
City, Mexico, 2498–2518. https://doi.org/10.18653/v1/2024.naacl-long.138
[11] Pengyue Jia, Yiding Liu, Xiangyu Zhao, Xiaopeng Li, Changying Hao, Shuaiqiang
Wang, and Dawei Yin. 2024. MILL: Mutual Verification with Large Language
Models for Zero-Shot Query Expansion. InProceedings of the 2024 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies. 2498–2518. https://arxiv.org/abs/2310.19056
[12] Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng
Wang, Jimeng Sun, and Jiawei Han. 2025. DeepRetrieval: Hacking Real Search
Engines and Retrievers with Large Language Models via Reinforcement Learning.
arXiv:2503.00223 [cs.IR] https://arxiv.org/abs/2503.00223
[13] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval
for Open-Domain Question Answering.. InEMNLP (1). 6769–6781.

Domain-Aware RAG: MoL-Enhanced RL for Efficient Training and Scalable Retrieval
[14] Yuan-Ching Kuo, Yi Yu, Chih-Ming Chen, and Chuan-Ju Wang. 2025. MMLF:
Multi-query Multi-passage Late Fusion Retrieval. InFindings of the Association
for Computational Linguistics: NAACL 2025. 6587–6598.
[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks.CoRRabs/2005.11401 (2020).
arXiv:2005.11401 https://arxiv.org/abs/2005.11401
[16] Zichen Liu, Chen Chen, Wenhan Li, Tianyu Pang, Chengzhe Du, and Min Lin.
2025. Understanding R1-Zero-Like Training: A Critical Perspective.arXiv
preprint(2025). https://github.com/sail-sg/understand-r1-zero/blob/main/
understand-r1-zero.pdf Code: https://github.com/sail-sg/understand-r1-zero.
[17] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak
Paul, and Benjamin Bossan. 2022.PEFT: State-of-the-art Parameter-Efficient
Fine-Tuning Methods. Technical Report. Hugging Face. https://github.com/
huggingface/peft
[18] Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage Re-ranking with BERT.
arXiv:1901.04085 [cs.IR] https://arxiv.org/abs/1901.04085
[19] OpenAI. 2022.New and improved embedding model. https://openai.com/blog/
new-and-improved-embedding-model
[20] Dipasree Pal, Mandar Mitra, and Kalyankumar Datta. 2013. Improving Query
Expansion Using WordNet.CoRRabs/1309.4938 (2013). arXiv:1309.4938 http:
//arxiv.org/abs/1309.4938
[21] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal policy optimization algorithms.arXiv preprint arXiv:1707.06347
(2017).
[22] Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighoff, Xi Victoria Lin,
Daniela Rus, Bryan Kian Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei Koh,
et al.2025. ReasonIR: Training Retrievers for Reasoning Tasks.arXiv preprint
arXiv:2504.20595(2025).
[23] Shezheng Song, Hao Xu, Jun Ma, Shasha Li, Long Peng, Qian Wan, Xiaodong
Liu, and Jie Yu. 2025. How to Complete Domain Tuning while Keeping General
Ability in LLM: Adaptive Layer-wise and Element-wise Regularization. https://doi.org/10.48550/arXiv.2501.13669
[24] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of
information retrieval models.arXiv preprint arXiv:2104.08663(2021).
[25] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query expansion with
large language models.arXiv preprint arXiv:2303.07678(2023).
[26] Liang Wen, Fenrui Xiao, Xin He, Yunke Cai, Qi An, Zhenyu Duan, Yimin Du,
Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng
Jia, and Xiangzheng Zhang. 2025. Light-R1: Curriculum SFT, DPO and RL for
Long COT from Scratch and Beyond. arXiv:2503.10460 https://github.com/
Qihoo360/Light-R1
[27] Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A. Rossi, Haoliang Wang,
and Julian McAuley. 2025. Knowledge-Aware Query Expansion with Large
Language Models for Textual and Relational Retrieval. arXiv:2410.13765 [cs.CL]
https://arxiv.org/abs/2410.13765
[28] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023.
C-Pack: Packaged Resources To Advance General Chinese Embedding.
arXiv:2309.07597 [cs.CL]
[29] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett,
Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor
Negative Contrastive Learning for Dense Text Retrieval.CoRRabs/2007.00808
(2020). arXiv:2007.00808 https://arxiv.org/abs/2007.00808
[30] Jinxi Xu and W. Bruce Croft. 2017. Quary Expansion Using Local and Global
Document Analysis.SIGIR Forum51, 2 (Aug. 2017), 168–175. https://doi.org/10.
1145/3130348.3130364
[31] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[32] Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022. Curriculum Learning
for Dense Retrieval Distillation. InProceedings of the 45th International ACM
SIGIR Conference on Research and Development in Information Retrieval(Madrid,
Spain)(SIGIR ’22). Association for Computing Machinery, New York, NY, USA,
1979–1983. https://doi.org/10.1145/3477495.3531791
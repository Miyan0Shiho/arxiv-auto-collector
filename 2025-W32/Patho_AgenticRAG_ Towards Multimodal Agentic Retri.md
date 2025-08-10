# Patho-AgenticRAG: Towards Multimodal Agentic Retrieval-Augmented Generation for Pathology VLMs via Reinforcement Learning

**Authors**: Wenchuan Zhang, Jingru Guo, Hengzhe Zhang, Penghao Zhang, Jie Chen, Shuwan Zhang, Zhang Zhang, Yuhao Yi, Hong Bu

**Published**: 2025-08-04 10:03:08

**PDF URL**: [http://arxiv.org/pdf/2508.02258v2](http://arxiv.org/pdf/2508.02258v2)

## Abstract
Although Vision Language Models (VLMs) have shown strong generalization in
medical imaging, pathology presents unique challenges due to ultra-high
resolution, complex tissue structures, and nuanced clinical semantics. These
factors make pathology VLMs prone to hallucinations, i.e., generating outputs
inconsistent with visual evidence, which undermines clinical trust. Existing
RAG approaches in this domain largely depend on text-based knowledge bases,
limiting their ability to leverage diagnostic visual cues. To address this, we
propose Patho-AgenticRAG, a multimodal RAG framework with a database built on
page-level embeddings from authoritative pathology textbooks. Unlike
traditional text-only retrieval systems, it supports joint text-image search,
enabling direct retrieval of textbook pages that contain both the queried text
and relevant visual cues, thus avoiding the loss of critical image-based
information. Patho-AgenticRAG also supports reasoning, task decomposition, and
multi-turn search interactions, improving accuracy in complex diagnostic
scenarios. Experiments show that Patho-AgenticRAG significantly outperforms
existing multimodal models in complex pathology tasks like multiple-choice
diagnosis and visual question answering. Our project is available at the
Patho-AgenticRAG repository:
https://github.com/Wenchuan-Zhang/Patho-AgenticRAG.

## Full Text


<!-- PDF content starts -->

Patho-AgenticRAG: Towards Multimodal Agentic Retrieval-Augmented
Generation for Pathology VLMs via Reinforcement Learning
Wenchuan Zhang1,2*Jingru Guo3*Hengzhe Zhang4*Penghao Zhang5Jie Chen2
Shuwan Zhang6Zhang Zhang1Yuhao Yi1,2†Hong Bu1,2
1Department of Pathology, West China Hospital, Sichuan University
2Institute of Clinical Pathology, West China Hospital, Sichuan University
3University of Toronto4School of Engineering and Computer Science, Victoria University of Wellington
5Independent Researcher6Department of Pathology, Shengjing Hospital of China Medical University
zhangwenchuan@stu.scu.edu.cn, yuhaoyi@scu.edu.cn
Abstract
Although Vision Language Models (VLMs) have shown
strong generalization in medical imaging, pathology presents
unique challenges due to ultra-high resolution, complex tis-
sue structures, and nuanced clinical semantics. These factors
make pathology VLMs prone to hallucinations, i.e., gener-
ating outputs inconsistent with visual evidence, which un-
dermines clinical trust. Existing RAG approaches in this do-
main largely depend on text-based knowledge bases, limit-
ing their ability to leverage diagnostic visual cues. To ad-
dress this, we propose Patho-AgenticRAG, a multimodal
RAG framework with a database built on page-level embed-
dings from authoritative pathology textbooks. Unlike tradi-
tional text-only retrieval systems, it supports joint text–image
search, enabling direct retrieval of textbook pages that con-
tain both the queried text and relevant visual cues, thus
avoiding the loss of critical image-based information. Patho-
AgenticRAG also supports reasoning, task decomposition,
and multi-turn search interactions, improving accuracy in
complex diagnostic scenarios. Experiments show that Patho-
AgenticRAG significantly outperforms existing multimodal
models in complex pathology tasks like multiple-choice di-
agnosis and visual question answering. Our project is avail-
able at the Patho-AgenticRAG repository: https://github.com/
Wenchuan-Zhang/Patho-AgenticRAG.
1 Introduction
With the continuous development of large-scale vision-
language models (VLMs), multimodal learning has made
breakthrough progress in many fields such as natural im-
age understanding, image-text generation, and medical im-
age analysis. Compared with other medical images (such as
X-rays, CT, and MRI), pathological images, with their ultra-
high resolution, fine-grained structure, and complex seman-
tic relationships, have put forward higher requirements on
the perception, reasoning, and factual consistency capabili-
ties of the model.
In recent years, more and more studies have attempted
to introduce VLMs into digital pathology tasks, such as di-
agnostic assistance [1], risk stratification [2], and question-
*Equal contribution.
†Corresponding author.answering systems [3]. However, existing pathology VLMs
still face key challenges such as severe hallucinations and a
lack of structured semantic control of retrieval mechanisms,
especially in tasks that require factual support and traceable
evidence. Therefore, how to build a highly reliable, multi-
modally interpretable pathology VLM with a factual con-
sistency assurance mechanism has become an important is-
sue that needs to be solved in this field. Although many
works in recent years have attempted to apply the Retrieval-
Augmented Generation (RAG) framework to medical multi-
modal tasks to improve the accuracy and credibility of large
language models in medical reasoning, there are still many
limitations in pathological image analysis scenarios. MMed-
RAG proposed a general multimodal medical RAG system
[4], but the image vector library it constructed lacks fine-
grained annotation of the organization system and is only di-
vided according to the type of imaging modality (such as CT,
X-ray), which cannot meet the actual needs of “classification
of the organ or tissue system to which the image belongs” in
pathological diagnosis. For example, in clinical work, it is
necessary to clarify whether the image comes from a specific
system, such as breast or lung, to match contextual knowl-
edge and structured diagnostic paths. At the same time, some
studies completely ignore the image modality and rely only
on text retrieval, failing to give full play to the important role
of images in assisting VLM reasoning, especially in scenar-
ios facing image-text consistency and visual evidence sup-
port [5, 6]. In addition, recent methods such as MedRAG and
Medical Graph RAG introduce complex reasoning processes
guided by knowledge graphs [7, 8]. Although they enhance
reasoning capabilities, the system design is complicated, the
execution process lacks intelligence and scalability, and it
is difficult to adapt to a variety of actual clinical tasks. Al-
though Liu et al. emphasizes the importance of knowledge
augmentation, its RAG module has the problem of insuffi-
cient instruction following, making it difficult to stably ex-
tract relevant knowledge fragments in complex instruction
following tasks, affecting model performance [9]. This study
aims to build an intelligent retrieval augmented generation
framework for pathology VLMs, to improve the credibil-
ity and interpretability of the model in complex question-
answering and reasoning tasks.arXiv:2508.02258v2  [cs.CV]  7 Aug 2025

Filtered high quality textbook pagesVision encoderQwen2 LMDecoderProj.Partion1: BreastPartion2: Histology and EmbryologyPartion15: General and Comprehensive……
HNSW Index
OCRVLMTextCaptionText Embed.ModelTraditional Multimodal RAGPatho-AgenticRAG(ours): enables text-based querying over text-rich images without preprocessingMilvus Vector Database
ColQwen
7.22s/page
0.39s/page
Agent SFTCold Start + Reinforcement LearningMultimodal Fusion Formula for RetrieverData Distillation
Train<think>This question pertains to ... we need to understand ... it falls under the ...</think><tool_call> ... </tool_call>Lack of Generalizable Capabilities &Rigid Output Patterns
❌
➕A small amount of data for SFTas cold start
➕
✅
1. Format2. Retrieval QueryCorrectness3. Partition SelectCorrectnessReward Design for GRPOPerforms best in the ablation studyAccelerates the convergence of reinforcement learningEvaluate on the training set 
Modify the formula 
❌Pages consisting entirely of images are easily retrieved Accurately retrieves areas with high similarity concentration 
🌟𝑆!+𝜋(𝑆!,𝑆")𝑺𝒕+𝝅(𝜿(𝑺𝒕),𝜿(𝑺𝒕))Figure 1: Knowledge Base Construction and Agent Training Method in the Patho-AgenticRAG
Our method focuses on three dimensions: multimodal
knowledge retrieval, intelligent planning capabilities, and
adaptability to pathology scenarios. Different from the pre-
vious RAG framework based on static prompts or text re-
trieval, Patho-AgenticRAG introduces a multimodal image-
text retrieval module and an agent mechanism with plan-
ning capabilities, which enables the model to more effec-
tively retrieve target images and corresponding knowledge
content from the structured pathology knowledge base, and
to reasonably integrate and reason. In addition, we introduce
a reinforcement learning optimization strategy to make the
agent more robust and generalizable in the highly complex
and uncertain question-answering environment in the field
of pathology (see Figure 1). The main contributions are sum-
marized as follows:
• We proposed a novel Multimodal Retrieval Mech-
anism that combines multimodal (image-text) vector
space modeling with a tissue-aware retrieval strategy.
This significantly improves the recall rate of the target
knowledge fragment while ensuring accuracy, provid-
ing a guarantee for fine-grained knowledge alignment in
pathology diagnosis tasks.
• We built a planning-capable intelligent agent within
the Agentic RAG system , which autonomously plans
multi-round retrieval and reasoning trajectories in re-
sponse to complex natural language pathology questions.
It dynamically invokes relevant multimodal knowledge
and effectively supports long-term dependency modeling
and multi-hop reasoning in diagnostic tasks.
• We proposed a Tool-Integrated Reasoning training
paradigm tailored for medical diagnostics , built upon
GRPO. This paradigm enables the agent to make fine-
grained decisions, such as whether to invoke retrieval,
how to reformulate questions, and how to assign domain-specific tools or classifiers within complex pathology
question answering scenarios. It addresses the high-
stakes nature of medical reasoning by promoting robust
decision-making and reliable tool coordination.
2 Related Work
2.1 Multimodal Agentic RAG
Retrieval-Augmented Generation mitigates the limitations
of large language models in knowledge completeness and
factual consistency by incorporating external knowledge
sources [10, 11]. Traditional RAG systems, such as Na ¨ıve
RAG [12, 13] and Advanced RAG [14, 15], typically fol-
low a static, linear “retrieve-then-read” workflow. While ef-
fective for simple queries, they struggle with complex tasks
requiring multi-step reasoning, context-aware adaptation, or
tool use [16]. Agentic RAG extends the RAG paradigm by
embedding autonomous agents into the retrieval and rea-
soning process [17]. These agents can plan retrieval strate-
gies [18], invoke external tools [19], reflect and revise out-
puts [20], and collaborate across multiple roles or modal-
ities [21, 22, 23]. Depending on design, they may operate
as single decision-makers (e.g., router agents) or as special-
ized multi-agent systems for handling heterogeneous data
sources. Despite these advances, existing Agentic RAG re-
search in the medical domain has focused predominantly on
text or structured data [24, 25], overlooking the diagnostic
richness encoded in medical images. However, fields like
pathology rely heavily on visual features, patterns in tissue
morphology, staining, and spatial arrangements that cannot
be captured through text alone.
2.2 Reinforcement Learning for Medical VLMs
Reinforcement learning (RL) provides a promising
paradigm for aligning the outputs of VLMs with clinical

accuracy requirements, especially in high-risk medical
domains where hallucinated descriptions can lead to severe
consequences [26]. A core challenge in medical VLMs is
ensuring factual alignment between visual evidence (e.g.,
pathology slides, radiology images) and textual outputs
[27, 28, 29].
However, direct end-to-end RL of large VLMs remains
highly impractical. The main limitations include the scarcity
of high-quality reward data from physicians [30], the insta-
bility of RL training on large models, and the lack of inter-
pretability in the learned behaviors [31]. These issues make
it difficult to deploy RL-fine-tuned models safely in clinical
settings. To address this, recent approaches have turned to
agent-centric strategies, where RL is applied not to the inter-
nal parameters of VLMs, but to the decision-making policies
of external agents that interact with them. These agents may
learn how to craft better queries, validate initial answers,
or select relevant external knowledge iteratively [4, 22, 23].
This decoupled optimization process not only reduces risk
but also enhances transparency and controllability.
3 Methodology
3.1 Overall Framework
The overall architecture adopts a modular design and con-
tains four main components. 1. Multimodal pathology
knowledge base: This is a specialized vector database con-
taining a rich collection of pathology textbook pages. It
acts as an external storage for agent queries to collect rel-
evant evidence, such as images of similar cases and their
corresponding diagnostic descriptions; 2. Intelligent agentic
router: This is the central processing unit of our framework.
It accepts the initial diagnosis query, decomposes it into log-
ically sequential subtasks, and plans; 3. VRAG Agent [19]:
This module supports multi-turn retrieval and summariza-
tion. It interacts with the knowledge base and distills the
returned textbook images into concise, useful information.
4. Core vision language model (inference engine) [26]: We
use the pretrained pathology VLM as the basic inference en-
gine. With the contextual summaries provided by the VRAG
agent, it performs inference to address the diagnostic query.
3.2 Construction of a Multimodal Pathology
Knowledge Base
To support retrieval-augmented reasoning in pathology, we
construct a high-quality multimodal knowledge base that
integrates authoritative textual and visual information. We
curated a large corpus by collecting over 600 authoritative
pathology textbooks—approximately 300,000 pages in to-
tal and, after removing irrelevant content, retained more
than 200,000 high-quality pages, which were converted into
image-based samples for diagnostic relevance. Using the
ColQwen2 model [32], we embed image-text pairs into a
unified vector space that captures both visual and semantic
signals. The embeddings are indexed with the HNSW al-
gorithm [33] and stored in Milvus [34] to support efficient
high-dimensional retrieval during reasoning. Full construc-
tion details are provided in Appendix A.3.3 Multimodal Fusion
Method LetSt∈RNt×Nddenote the text–document sim-
ilarity matrix and Sv∈RNv×Ndthe image–document simi-
larity matrix, where Ntis the number of text tokens, Nvthe
number of image patches, and Ndthe number of document
tokens. These two modalities are fused by the following ex-
pression:
std 
std(St[i,:])
×
mean 
κ(St[i,:])2×mean 
κ(Sv[i,:])
+mean 
max( St[i,:])
,
(1)
where κ(St[i,:])denotes the kurtosis of the similarity scores
between all document tokens and text token i, andκ(Sv[i,:])
denotes the kurtosis of the similarity scores between all doc-
ument tokens and image patch i. The first term captures how
the standard deviation std (·)and kurtosis κ(·)reflect the
variation of similarity scores across tokens or image patches
with respect to the database document, represented as rows
inStorSv. This encourages the retrieval results to have dif-
ferent importance for various tokens in the document, in-
dexed by j= 1, . . . , N d. The reason for this is, in practice,
only a portion of a page is typically relevant to the database
document, meaning only some j-th elements in St[i, j]or
Sv[i, j]contribute meaningfully. When all parts of a docu-
ment exhibit high responsiveness, resulting in a low standard
deviation, this may indicate a noisy document with problem-
atic embeddings, which should be deprioritized. The second
term, mean 
max( St[i,:])
, as in CoPaLi [32], quantifies the
maximum relevance max( St[i,:])of each token to any token
in the document.
It is important to note that the similarity matrix of the
image modality with respect to the document embedding,
i.e.,Sv∈RNv×Nd, is used only to calculate the kurto-
sisκ(Sv[i,:]), and the average similarity information from
the image modality, such as mean (Sv[i,:]), is not incorpo-
rated. This design emphasizes attention to the most relevant
portions of a page, identified via large values of κ(Sv[i,:]).
When a token or patch shows high similarity to all parts of
a document, it is treated as a noisy retrieval result and is as-
signed a lower score during re-ranking.
Explanation To intuitively understand how Equation (1)
influences the re-ranking process, we present the similar-
ity matrices of the first- and second-ranked documents with
respect to the query text embedding and image embed-
ding in Figure 3. As shown in Figure 3, the second-ranked
document exhibits a more uniform similarity, or attention,
across the tokens of the document embedding. In contrast,
the first-ranked document demonstrates more concentrated
and higher similarity on fewer tokens of the document em-
bedding. In this case, the high similarity of the first-ranked
document to the query is meaningful, as it is caused by fo-
cused attention on the most informative tokens. On the other
hand, the high similarity of the second-ranked document to
the query results from more diffuse attention, likely caused
by noise. Thus, although the second-ranked feature initially
has the highest text similarity in the retrieval process, after
re-ranking by the fusion formula, it drops to second place,
while the original second-ranked document is promoted to

Final Answer:<think>The Indian-file pattern is a hallmark of invasive lobular carcinoma, where tumor cells infiltrate the stroma in single-file lines. Lobular carcinoma cells are typically small and uniform with scant cytoplasm and round nuclei. In contrast, ductal carcinoma (B) forms solid nests and trabeculae, papillary carcinoma (C) has fibrovascular cores in papillary structures, and mucinous carcinoma (D) shows tumor cells in mucin pools. The observation of small cells, uniform nuclei, and single-file infiltration aligns specifically with lobular carcinoma morphology.</think><answer>A</answer>Text Query:Small, uniform tumor cells infiltrate in an Indian-file pattern. Which breast cancer type is this?A. Lobular breast cancerB. Ductal breast cancerC. Papillary breast cancerD. Mucinous breast cancerExternal KnowledgeTool Call1: {"name": "rag", "parameters": {"query": "Histological features of invasive lobular carcinoma of the breast, including Indian-file pattern"}}{"name": ”classifier", "parameters": {”partition": ”Breast"}}Tool Call2: {"name": "rag", "parameters": {"query": "How to differentiate ductal, papillary, and mucinous breast carcinoma histologically"}}Text RetrievalMax Sim.
…
Top20 Pages for each query
Patho-R1Pathology Reasoning ModelRerank by the similiary of image queryCrop & summarizeinfo.
Database: over 200k pages of textbookover 150m vectorsVrag Agent
Multi-Turn IntercationThinkImage Query:
Ground Truth: A
Q1:Histological features of invasive lobular carcinoma of the breast, including Indian-file patternA: Histological features of invasive lobular carcinoma of the breast include uniform small round tumor cells infiltrating the stroma in a single-file (Indian-file) arrangement and circumferentially around ducts in a target-like pattern.Q2: How to differentiate ductal, papillary, and mucinous breast carcinoma histologicallyA: Histologically, ductal carcinoma shows glandular, papillary, cribriform, or diffuse growth patterns, often forming nests, trabeculae, or cords. Papillary carcinoma is characterized by prominent papillary structures with central fibrovascular cores. Mucinous carcinoma displays abundant extracellular mucin, with tumor cells floating loosely in mucin pools.Summarized Information
Invasive lobular carcinoma. Characteristic histologic features are: one cell wide files of round regular tumour cells (Indian file)Top1 for tool call 1
Ductal and lobular carcinomas represent75% and 25% of metastases…, papillary, cribriform or diffuse patterns…mucinous carcinoma … B Hybrid endometrioid-like/mucinous
Top1 for tool call 2
Agentic Router(ours)
Figure 2: An illustration of the multi-turn retrieval and summarization process.
OriginalText Query: This image shows monophasic synovial sarcoma. What are its key histologic features?A.Dense, uniform spindle cells with scant cytoplasm in short, intersecting fascicles.B.Biphasic: glandular epithelium with spindle cell stroma.C.Scattered adipocytes in fibrous stroma with atypia.D.Large polygonal cells with coarse eosinophilic granules.Rewrited Search Query:What are the histological features of monophasic synovial sarcoma?Each query token embedding is matched to the most similar image-patch embedding on indexed pages, and their similarity scores are summed to rank the pages.
Image Query:Retrieval 
Patho-Fusion Score:Text Similarity MatrixImage Similarity Matrix0.66380.6671Final Top1
Cytologic appearance of Ewing sarcoma/PNET ...
0.5878Top20 from text retrieval……Monophasic synovial sarcoma. The tumor ...
0.6558
… monophasic synovial sarcoma with ...
0.6190
0.6602
Lose
Synovial sarcoma with an adenocarcinoma-likeappearance of the epithelial component.Rerank
Figure 3: Illustration of the reranking process utilizing
modality fusion.the top rank due to its combination of high similarity and
concentrated attention.
3.4 Agentic Diagnostic Workflow with
Multimodal Evidence Tracing
Our system adopts an intelligent multi-agent workflow that
transforms a diagnostic query into an evidence-grounded
conclusion. Given a user input (e.g., a pathology image
and candidate diagnoses), the Agentic Router module first
parses the query, decomposes it into sub-tasks aligned with
each diagnostic candidate, and formulates a high-level re-
trieval plan. It delegates retrieval and evidence aggregation
to the VRAG Agent , which operates under its guidance.
The VRAG Agent executes a multi-turn retrieval pro-
cess against the multimodal knowledge base. It first con-
ducts text-based retrieval using candidate-specific keywords,
and then re-ranks the retrieved entries by evaluating image-
text similarity with the query. Through iterative evidence re-
finement and summarization, as illustrated in Figure 2, the
agent constructs a structured prompt containing the top 1-
ranked visual evidence for each candidate. This prompt is
then passed to a specialized vision-language model, which
performs contrastive reasoning to produce a diagnosis and
evidence-grounded report. Details of each module and step
are provided in Appendix B.
3.5 Tool Integrated RL for Agentic Router
Traditional RAG systems are often treated as static
pipelines, without adapting their behavior to each query’s
complexity. To overcome this limitation, we introduce a re-
inforcement learning (RL) framework that enables an agent
to learn dynamic invocation and routing strategies [35]. The
agent’s task is to generate a decision path specifying whether

and how to call the RAG system. Formally, given an input
query Qorig, the agent’s policy πoutputs a path P, optimized
to maximize the expected similarity to a ground-truth deci-
sion path Pgt
max
πEP∼π(Qorig)[Rfinal(P, P gt)] (2)
The hierarchical reward Rfinalcompares the generated path
to the target path step by step (see Algorithm 1). The agent
performs a sequence of decisions to construct the path:
•Decision 1: Whether to invoke RAG?
– Path A (No RAG Invocation): If the agent decides
False , the decision process terminates. This is for
simple queries that can be answered without external
knowledge. Final path: {rag :False }
– Path B (Invoke RAG): IfTrue , proceed to the next
decision.
•Decision 2: How to decompose the task? (Only applies
if RAG is invoked)
The agent may choose to rewrite the query one or more
times to better surface its core semantics and improve
retrieval quality. This is not a mechanical rewriting pro-
cess, but a targeted transformation to better align with the
retrieval engine.
•Decision 3: Whether to use a tissue-specific classifier?
(Only applies if RAG is invoked)
The agent decides whether to enable a classifier to restrict
retrieval to a relevant knowledge partition.
– Path B.1 (Global Retrieval): IfFalse , RAG
retrieves documents from the full corpus. Fi-
nal path: {rag :True ,rewrite count :
n,classifier :False }
– Path B.2 (Classifier-Based Retrieval): IfTrue , pro-
ceed to the next decision.
•Decision 4: Whether the classifier assigns the query
to the correct partition? (Only applies if a classifier is
enabled)
The agent selects a classifier from the available set
{C1, . . . , C m}and attempts to assign the query to
the correct partition. The effectiveness of this deci-
sion depends on both the selection of the classifier
and the correctness of the classification. Final path:
{rag :True ,rewrite count :n,classifier :
True ,partition :Cj}
RL Training with GRPO We use the GRPO algo-
rithm [36] to train the policy πθ. For each query Qorig, the
agent generates multiple decision paths:
GQ={(P1, r1),(P2, r2), . . . , (Pn, rn)} (3)
where each Piis a complete decision path and riis its corre-
sponding reward score ri∈[0,4]based on the hierarchical
reward function. We normalize the rewards within the group
to compute the advantage function Ai(Pi|Q):
Ai(Pi|Q) =ri−µQ
σQ+η(4)Algorithm 1: Hierarchical Reward Computation
Require: Agent path P, Ground Truth Pgt
1:Rfinal←0
2:ifP.rag̸=Pgt.rag then
3: return Rfinal ▷Incorrect Decision 1
4:end if
5:ifPgt.rag =False then
6: Rfinal←4 ▷Correct Decision 1 (Path A)
7:else
8: Rfinal←1 ▷Correct Decision 1 (Path B)
9: ifP.rewrite count =Pgt.rewrite count
then
10: Rfinal←Rfinal+ 1 ▷Correct Decision 2
11: end if
12: ifP.classifier =Pgt.classifier then
13: ifP.classifier =False then
14: Rfinal←Rfinal+ 2 ▷Correct Decision 3
(Path B.1)
15: else
16: Rfinal←Rfinal+ 1 ▷Correct Decision 3
(Path B.2)
17: ifP.partition =Pgt.partition then
18: Rfinal←Rfinal+ 1▷Correct Decision 4
19: end if
20: end if
21: end if
22:end if
23:return Rfinal
where µQandσQare the mean and standard deviation of
rewards within group GQ, and ηis a small constant for nu-
merical stability.
To update the policy, we apply the GRPO objective, which
extends PPO by group-wise normalized advantages and KL
regularization with a reference model. Specifically, for each
group of outputs {oi}G
i=1, from the same query, we optimize:
JGRPO(θ) =Eq∼P(Q),{oi}G
i=1∼πθold(O|q)1
GGX
i=11
|oi||oi|X
t=1
minπθ(oi,t|q, oi,<t)
πθold(oi,t|q, oi,<t)ˆAi,t,
clipπθ(oi,t|q, oi,<t)
πθold(oi,t|q, oi,<t),1−ϵ,1 +ϵ
ˆAi,t
−βD KL
πθ(oi,t|q, oi,<t)πref(oi,t|q, oi,<t)
(5)
Here, ˆAi,tis the advantage at step twithin each output,
computed relative to other outputs for the same query. This
group-wise comparison helps the agent learn from relative
improvements, leading to more effective decision-making in
complex reasoning paths. See Appendix C for details.

Method Rec@1 Rec@5 MRR@1 MRR@5 MRR@20 NDCG@1 NDCG@5 NDCG@20
CoPaLi (Text) 0.640 (2) 0.900 (1) 0.640 (2) 0.734 (2) 0.736 (2) 0.740 (2) 0.804 (2) 0.796 (2)
CoPaLi (Image) 0.060 (3) 0.220 (3) 0.060 (3) 0.112 (3) 0.170 (3) 0.080 (3) 0.174 (3) 0.359 (3)
WeiMoCIR 0.060 (3) 0.200 (4) 0.060 (3) 0.102 (4) 0.158 (4) 0.080 (3) 0.163 (4) 0.342 (4)
Patho-Fusion (ours) 0.720 (1) 0.880 (2) 0.720 (1) 0.777 (1) 0.784 (1) 0.820 (1) 0.824 (1) 0.827 (1)
Table 1: Comparison of retrieval methods. Numbers in parentheses denote rank.
4 Experiments
4.1 Multimodal Fusion
Baseline Methods For baseline comparisons, we evalu-
ate the proposed method against CoPaLi [32] and Weighted
Modality Fusion and Similarity for Composed Image Re-
trieval (WeiMoCIR) [37].
• For CoPaLi [32], which supports only a single modal-
ity, we apply retrieval separately for each modality. The
scoring function is
NqX
i=1max
j=1,...,N d
Eq(i), Ed(j)
, (6)
where Eqis either EtorEvsoNq=NtorNq=Nv,
and⟨·,·⟩denotes the inner product. Each query embed-
dingEq(i)is matched to the most relevant document em-
bedding Ed(j), where j= 1, . . . , N d, and the results are
aggregated over all query tokens or patches.
• In WeiMoCIR, the query embedding is computed as
q= (1−α)·ev+α·et, (7)
where ev=mean (Ev(i,:))is the average vision em-
bedding over all patches i= 1, . . . , N v, and et=
mean (Et(i,:))is the average text embedding over all to-
kensi= 1, . . . , N t. The parameter α= 0.1represents
the weighting coefficient for the text modality. The final
similarity score between the query and a database docu-
ment is computed using the inner product as
1
NdNdX
j=1⟨q,ed,j⟩, (8)
where ed,jis the j-th token embedding of the document.
Dataset and Evaluation Protocol The dataset consists of
100 pairs of images, questions, and answers curated by do-
main experts. We randomly split the dataset, using 50% for
training and 50% for testing. The modality fusion function
is optimized only on the training data to prevent potential
data leakage. All modality fusion methods are evaluated on
the test set, and recall, mean reciprocal rank (MRR), and
normalized discounted cumulative gain (NDCG) metrics are
reported.
Experimental Results The experimental results are
shown in Table 1. Recall@20 is omitted since there are only
20 results in the re-ranking stage and the Recall@20 is iden-
tical for all algorithms. The results demonstrate a clear ad-
vantage for the proposed modality fusion method over thebaseline approaches. While using only the text modality
for retrieval can already achieve good performance, the re-
trieval results remain suboptimal without the proposed fu-
sion strategy in Equation (1). Methods using only the image
modality or WeiMoCIR perform worse than the proposed fu-
sion by a significant margin. This is primarily because both
heavily rely on the image modality for retrieval, whereas
pathology images require strong domain expertise to inter-
pret and general-purpose embedding methods may not pro-
vide optimal representations for this task. Although WeiMo-
CIR achieves strong results on general retrieval benchmarks,
it underperforms in the medical multimodal retrieval setting.
Overall, these findings demonstrate that general multimodal
fusion strategies are not sufficient for the pathology domain
and the specifically designed fusion mechanism proposed
here offers superior effectiveness.
4.2 Patho-AgenticRAG Evaluation Results
Ablation Analysis We conducted three main ablation
studies to investigate the necessity and data proportion of
SFT before GRPO. The results show that skipping SFT
leads to poor convergence during GRPO. However, perform-
ing SFT with a large amount of data causes the model to
lack generalizable capabilities and exhibit rigid output pat-
terns. Therefore, using a small amount of SFT data as a
cold start before GRPO is the optimal strategy. Notably,
adopting a lightweight SFT phase (e.g., SFT400) before
GRPO achieves the best overall balance. This setting con-
sistently outperforms both the ”no-SFT” and ”large-SFT”
baselines across multiple datasets. For example, on the Path-
VQA benchmark, using SFT400+GRPO4k improves per-
formance from 77.51% (GRPO4k only) to 80.34%. Simi-
larly, on the Quilt-VQA dataset, performance improves from
60.93% (GRPO4k) to 75.80%, a +14.87% increase, indi-
cating that a small amount of supervised guidance before
preference optimization significantly enhances model capa-
bility. These results suggest that SFT400 provides an effec-
tive “cold start” that guides the policy initialization without
compromising flexibility or generalization, as shown by Fig-
ure 4.
Close-Ended Benchmarks Results Closed-ended ques-
tions play a crucial role in pathology-related tasks, partic-
ularly in diagnostic classification. To evaluate model perfor-
mance on such tasks, we consider two types of close-ended
question datasets: (1) Yes/No questions, selected from Path-
VQA [38] and Quilt-VQA [39]; and (2) multiple-choice
questions, sourced from PathMMU [40], MedXpertQA [41],
and OmniMedVQA [42].
The results on close-ended benchmarks are summarized

Accuracy(%)
0.00%25.00%50.00%75.00%100.00%
Pathmmu_test Pathmmu_test_tiny MedXpertQA OmniMedVQA 
BRIGHTChallengePath-VQA Quilt-VQAPatho-R1 +Qwen3 +GRPO4k +SFT4k-GRPO400 +SFT400-GRPO4kFigure 4: Ablation study results across multiple medical QA datasets.
ModelPathMMU-test PathMMU-test-tiny
Atlas EduContent PathCLS PubMed SocialPath Atlas EduContent PathCLS PubMed SocialPath
InternVL2-8B 43.68 44.86 23.77 44.56 45.40 46.63 50.59 21.47 49.11 51.38
InternVL2.5-8B 50.06 50.62 32.84 50.02 50.87 51.44 50.59 29.38 55.87 57.80
InternVL3-8B 54.07 50.80 39.09 54.04 53.32 58.17 54.90 42.94 57.65 60.55
Llama-3.2-11B-VI 41.05 37.49 26.72 38.82 39.21 45.19 38.04 29.38 39.50 41.74
Llama-3.2V-11B-cot 51.81 45.45 30.76 48.15 46.10 49.04 47.06 29.94 53.38 45.41
LLaV A-Onevision-7B 21.65 21.27 12.01 27.77 21.25 31.25 21.18 13.56 31.32 18.35
Qwen2.5VL-7B 41.18 43.20 24.82 42.77 39.67 44.23 49.41 24.86 44.84 40.83
Patho-R1-7B 75.34 66.43 45.40 66.06 67.93 81.73 75.29 44.63 72.24 67.89
Patho-AgenticRAG 78.32 70.96 53.16 69.69 71.06 79.33 76.47 57.22 72.24 74.70
Table 2: Comparison of model performance across multiple tasks. The left group shows results on PathMMU-test, and the
right group on PathMMU-test-tiny. Best and second-best performances are bolded and underlined respectively.
ModelYorN MedXpert OmniMed
Quilt Path Path Bright
InternVL2-8B 60.56 61.36 10.00 40.56
InternVL2.5-8B 60.06 64.78 22.22 49.78
InternVL3-8B 33.82 18.56 15.56 65.28
Llama-3.2-11B-VI 63.27 63.50 13.33 47.08
Llama-3.2V-11B-cot 54.81 56.42 21.11 54.83
LLaV A-Onevision-7B 24.20 52.38 16.67 31.46
Qwen2.5VL-7B 52.19 41.82 12.22 43.60
Patho-R1-7B 64.72 46.97 22.00 70.79
Patho-AgenticRAG 75.80 80.34 60.00 90.11
Table 3: Performance comparison on Quilt-VQA,
Path-VQA, MedXpert, and OmniMed.
in Tables 2 and 3. Patho-AgenticRAG achieves the best
overall performance across most tasks, significantly outper-
forming both general-purpose vision-language models (e.g.,
InternVL3, Qwen2.5VL) and domain-specialized baselines
such as Patho-R1-7B [26]. Specifically, Patho-AgenticRAG
achieves +13.37% improvement on Quilt-VQA (75.80%
vs. 64.72%) and +38.00% on MedXpertQA (60.00% vs.
22.00%) over Patho-R1. The largest margin appears on
MedXpertQA, highlighting the importance of retrieval-augmented reasoning in knowledge-intensive tasks. On Om-
niMedVQA Bright Challenge, the model improves from
70.79% (Patho-R1) to 90.11%, a +19.32% increase, demon-
strating substantial gains in both generalization and diagnos-
tic precision. Details are in Appendix D
5 Conclusion
We proposed Patho-AgenticRAG, a novel multimodal
retrieval-augmented generation framework tailored for
pathology diagnosis. By leveraging intelligent agents for dy-
namic querying of image-based vector databases, as well
as employing task decomposition, query planning, and ev-
idence aggregation, our approach significantly enhances the
reasoning capabilities of vision-language models in pathol-
ogy tasks. Our framework addresses the critical issue of
hallucination in pathology diagnosis by promoting knowl-
edge alignment, supporting evidence-based reasoning, and
improving factual consistency in generated outputs. Patho-
AgenticRAG demonstrates significant improvements over
existing state-of-the-art multimodal models in key metrics,
including answer precision and evidence traceability, rep-
resenting a notable advancement in the integration of image
content and reasoning for real-world pathology applications.

References
[1] C. Chen, L. L. Weishaupt, D. F. K. Williamson, R. J.
Chen, T. Ding, B. Chen, A. Vaidya, L. P. Le, G. Jaume,
M. Y . Lu et al. , “Evidence-based diagnostic reasoning
with multi-agent copilot for human pathology,” arXiv
preprint arXiv:2506.20964 , 2025.
[2] P. Liu, L. Ji, J. Gou, B. Fu, and M. Ye, “Interpretable
Vision-Language Survival Analysis with Ordinal In-
ductive Bias for Computational Pathology,” in The
Thirteenth International Conference on Learning Rep-
resentations (ICLR) , 2025.
[3] M. Y . Lu, B. Chen, D. F. K. Williamson, R. J. Chen,
M. Zhao, A. K. Chow, K. Ikemura, A. Kim, D. Pouli,
A. Patel et al. , “A multimodal generative AI copilot
for human pathology,” Nature , vol. 634, no. 8033, pp.
466–473, 2024.
[4] P. Xia, K. Zhu, H. Li, T. Wang, W. Shi, S. Wang,
L. Zhang, J. Zou, and H. Yao, “MMed-RAG: Versa-
tile Multimodal RAG System for Medical Vision Lan-
guage Models,” in The Thirteenth International Con-
ference on Learning Representations (ICLR) , 2025.
[5] M. S. Jabal, P. Warman, J. Zhang, K. Gupta, A. Jain,
M. Mazurowski, W. Wiggins, K. Magudia, and
E. Calabrese, “Language Models and Retrieval Aug-
mented Generation for Automated Structured Data
Extraction from Diagnostic Reports,” arXiv preprint
arXiv:2409.10576 , 2024.
[6] S. N. Cheetirala, G. Raut, D. Patel, F. Sanatana,
R. Freeman, M. A. Levin, G. N. Nadkarni, O. Dawkins,
R. Miller, R. M. Steinhagen et al. , “Less Context,
Same Performance: A RAG Framework for Resource-
Efficient LLM-Based Clinical NLP,” arXiv preprint
arXiv:2505.20320 , 2025.
[7] X. Zhao, S. Liu, S.-Y . Yang, and C. Miao,
“Medrag: Enhancing retrieval-augmented generation
with knowledge graph-elicited reasoning for health-
care copilot,” in Proceedings of the ACM on Web Con-
ference 2025 (WWW) , pp. 4442–4457, 2025.
[8] J. Wu, J. Zhu, Y . Qi, J. Chen, M. Xu, F. Menolascina,
and V . Grau, “Medical graph rag: Towards safe medical
large language model via graph retrieval-augmented
generation,” arXiv preprint arXiv:2408.04187 , 2024.
[9] H. Liu, K. Son, J. Yang, C. Liu, J. Gao, Y . J. Lee,
and C. Li, “Learning customized visual models with
retrieval-augmented knowledge,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR) , pp. 15148–15158, 2023.
[10] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih,
T. Rockt ¨aschel et al. , “Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks,” Advances
in Neural Information Processing Systems (NeurIPS) ,
vol. 33, pp. 9459–9474, 2020.
[11] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang,
Q. Chen, W. Peng, X. Feng, B. Qin et al. , “A survey onhallucination in large language models: Principles, tax-
onomy, challenges, and open questions,” ACM Trans-
actions on Information Systems (TOIS) , vol. 43, no. 2,
pp. 1–55, 2025.
[12] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer,
“Sigmoid loss for language image pre-training,” in
Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision (ICCV) , pp. 11975–11986,
2023.
[13] C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi,
B. Catanzaro, and W. Ping, “Nv-embed: Improved
techniques for training llms as generalist embedding
models,” arXiv preprint arXiv:2405.17428 , 2024.
[14] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu,
S. Wang, X. Han, Z. Liu et al. , “Visrag: Vision-based
retrieval-augmented generation on multi-modality doc-
uments,” arXiv preprint arXiv:2410.10594 , 2024.
[15] J. Cho, D. Mahata, O. Irsoy, Y . He, and M. Bansal,
“M3docrag: Multi-modal retrieval is what you need
for multi-page multi-document understanding,” arXiv
preprint arXiv:2411.04952 , 2024.
[16] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei,
“Agentic retrieval-augmented generation: A survey on
agentic rag,” arXiv preprint arXiv:2501.09136 , 2025.
[17] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, H. Wang, and H. Wang, “Retrieval-augmented
generation for large language models: A survey,” arXiv
preprint arXiv:2312.10997 , 2023.
[18] A. Joshi, S. M. Sarwar, S. Varshney, S. Nag,
S. Agrawal, and J. Naik, “Reaper: Reasoning based re-
trieval planning for complex rag systems,” in Proceed-
ings of the 33rd ACM International Conference on In-
formation and Knowledge Management (CIKM) , pp.
4621–4628, 2024.
[19] Y . Chen, Y . Shen, W. Huang, S. Zhou, Q. Lin, X. Cai,
Z. Yu, B. Shi, and Y . Qiao, “Learning Only with Im-
ages: Visual Reinforcement Learning with Reason-
ing, Rendering, and Visual Feedback,” arXiv preprint
arXiv:2507.20766 , 2025.
[20] C. Ravuru, S. S. Sakhinana, and V . Runkana, “Agentic
retrieval-augmented generation for time series analy-
sis,” arXiv preprint arXiv:2408.14484 , 2024.
[21] Q. Wang, R. Ding, Z. Chen, W. Wu, S. Wang, P. Xie,
and F. Zhao, “Vidorag: Visual document retrieval-
augmented generation via dynamic iterative reasoning
agents,” arXiv preprint arXiv:2502.18017 , 2025.
[22] B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang,
H. Zamani, and J. Han, “Search-r1: Training llms to
reason and leverage search engines with reinforcement
learning,” arXiv preprint arXiv:2503.09516 , 2025.
[23] J. Wu, Z. Deng, W. Li, Y . Liu, B. You, B. Li, Z. Ma,
and Z. Liu, “MMSearch-R1: Incentivizing LMMs to
Search,” arXiv preprint arXiv:2506.20670 , 2025.
[24] K. Thakrar, S. Basavatia, and A. Daftardar, “Culti-
vating Multimodal Intelligence: Interpretive Reason-
ing and Agentic RAG Approaches to Dermatological
Diagnosis,” arXiv preprint arXiv:2507.05520 , 2025.

[25] A. Zeggai, I. Traikia, A. Lakehal, and A. Boulesnane,
“AI-VaxGuide: An Agentic RAG-Based LLM for Vac-
cination Decisions,” arXiv preprint arXiv:2507.03493 ,
2025.
[26] W. Zhang, P. Zhang, J. Guo, T. Cheng, J. Chen,
S. Zhang, Z. Zhang, Y . Yi, and H. Bu, “Patho-R1: A
Multimodal Reinforcement Learning-Based Pathology
Expert Reasoner,” arXiv preprint arXiv:2505.11404 ,
2025.
[27] Y . Sun, Y . Si, C. Zhu, K. Zhang, Z. Shui, B. Ding,
T. Lin, and L. Yang, “CPathAgent: An Agent-based
Foundation Model for Interpretable High-Resolution
Pathology Image Analysis Mimicking Pathologists’
Diagnostic Logic,” arXiv preprint arXiv:2505.20510 ,
2025.
[28] J. Chen, R. Ouyang, A. Gao, S. Chen, G. H. Chen,
X. Wang, R. Zhang, Z. Cai, K. Ji, G. Yu, X. Wan,
and B. Wang, “HuatuoGPT-Vision, Towards Injecting
Medical Visual Knowledge into Multimodal LLMs at
Scale,” arXiv preprint arXiv:2406.19280 , 2024.
[29] Z. Xu, C. Jin, Y . Wang, Z. Liu, and H. Chen, “Dis-
covering Pathology Rationale and Token Allocation
for Efficient Multimodal Pathology Reasoning,” arXiv
preprint arXiv:2505.15687 , 2025.
[30] T.-H. Pham and C. Ngo, “RARL: Improving Medical
VLM Reasoning and Generalization with Reinforce-
ment Learning and LoRA under Data and Hardware
Constraints,” arXiv preprint arXiv:2506.06600 , 2025.
[31] W. Zhu, X. Dong, X. Li, P. Qiu, X. Chen, A. Razi,
A. Sotiras, Y . Su, and Y . Wang, “Toward Effec-
tive Reinforcement Learning Fine-Tuning for Medi-
cal VQA in Vision-Language Models,” arXiv preprint
arXiv:2505.13973 , 2025.
[32] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud,
C. Hudelot, and P. Colombo, “ColPali: Efficient Doc-
ument Retrieval with Vision Language Models,” in In-
ternational Conference on Learning Representations
(ICLR) , 2025.
[33] Y . A. Malkov and D. A. Yashunin, “Efficient and ro-
bust approximate nearest neighbor search using hier-
archical navigable small world graphs,” IEEE Trans-
actions on Pattern Analysis and Machine Intelligence
(TPAMI) , vol. 42, no. 4, pp. 824–836, 2018.
[34] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang,
X. Guo, C. Li, X. Xu et al. , “Milvus: A Purpose-Built
Vector Data Management System,” in Proceedings of
the 2021 International Conference on Management of
Data (SIGMOD) , pp. 2614–2627, 2021.
[35] C. Qian, E. C. Acikgoz, Q. He, H. Wang, X. Chen,
D. Hakkani-T ¨ur, G. Tur, and H. Ji, “Toolrl: Re-
ward is all tool learning needs,” arXiv preprint
arXiv:2504.13958 , 2025.
[36] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song,
X. Bi, H. Zhang, M. Zhang, Y . Li, Y . Wu et al. ,
“Deepseekmath: Pushing the limits of mathematical
reasoning in open language models,” arXiv preprint
arXiv:2402.03300 , 2024.[37] R.-D. Wu, Y .-Y . Lin, and H.-F. Yang, “Training-free
Zero-shot Composed Image Retrieval via Weighted
Modality Fusion and Similarity,” in International Con-
ference on Technologies and Applications of Artificial
Intelligence (TAAI) , pp. 77–90. Springer, 2024.
[38] X. He, Y . Zhang, L. Mou, E. Xing, and P. Xie,
“Pathvqa: 30000+ questions for medical visual ques-
tion answering,” arXiv preprint arXiv:2003.10286 ,
2020.
[39] W. Ikezogwo, S. Seyfioglu, F. Ghezloo, D. Geva,
F. Sheikh Mohammed, P. K. Anand, R. Krishna, and
L. Shapiro, “Quilt-1m: One million image-text pairs
for histopathology,” Advances in Neural Information
Processing Systems , vol. 36, pp. 37995–38017, 2023.
[40] Y . Sun, H. Wu, C. Zhu, S. Zheng, Q. Chen, K. Zhang,
Y . Zhang, D. Wan, X. Lan, M. Zheng et al. , “Pathmmu:
A massive multimodal expert-level benchmark for un-
derstanding and reasoning in pathology,” in European
Conference on Computer Vision (ECCV) , pp. 56–73,
2024.
[41] Y . Zuo, S. Qu, Y . Li, Z. Chen, X. Zhu, E. Hua,
K. Zhang, N. Ding, and B. Zhou, “MedXpertQA:
Benchmarking Expert-Level Medical Reasoning and
Understanding,” arXiv preprint arXiv:2501.18362 ,
2025.
[42] Y . Hu, T. Li, Q. Lu, W. Shao, J. He, Y . Qiao, and P. Luo,
“Omnimedvqa: A new large-scale comprehensive eval-
uation benchmark for medical lvlm,” in Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) , pp. 22170–22183, 2024.

A Multimodal Knowledge Base Construction
To construct a high-quality knowledge base tailored for pathology, we collected and processed over 600 authoritative pathology
textbooks that span a wide range of diagnostic domains. These sources were chosen for their depth, credibility, and relevance
to real-world diagnostic practice.
To improve retrieval accuracy and reduce noise in downstream reasoning, we removed sections that typically lack diagnostic
content, including covers, prefaces, tables of contents, and references. The remaining pages, primarily rich in textual explana-
tions and pathology figures, were preserved as the core knowledge units. These preserved pages were further categorized into
19 distinct anatomical and diagnostic classes to support systematized retrieval. Specifically, the categories are as follows:
1.Bone and soft tissue – including neoplastic and non-neoplastic lesions of bone, cartilage, and connective tissue.
2.Cytology – encompassing exfoliative and fine-needle aspiration cytopathology across body systems.
3.Gastrointestinal tract, liver, gallbladder, pancreas, and digestive system – covering both luminal and solid organ
pathology within the digestive system.
4.Hematology, lymphatic system, and bone marrow – focusing on hematopoietic and lymphoid neoplasms and marrow
disorders.
5.Infectious diseases – including pathology of viral, bacterial, fungal, and parasitic infections.
6.Oral and head and neck – covering salivary glands, oral mucosal lesions, larynx, pharynx, nasal cavity, and other ENT
(ear, nose, throat) structures.
7.Urinary and male reproductive system – involving kidney, bladder, prostate, testis, and related structures.
8.Breast – dedicated to benign, precursor, and malignant breast lesions.
9.Endocrine system – covering thyroid, parathyroid, adrenal glands, and neuroendocrine tumors.
10.General comprehensive pathology – consisting of cross-system concepts such as inflammation, necrosis, and neoplasia.
11.Histology and embryology – including normal tissue architecture and developmental processes.
12.Neonatal, pediatric, and childhood diseases – covering age-specific pathologies from birth to adolescence.
13.Skin and dermatopathology – addressing inflammatory, infectious, and neoplastic skin conditions.
14.Central nervous system – including brain, spinal cord, and peripheral nerve pathology.
15.Female reproductive system – encompassing ovary, uterus, cervix, and vulvovaginal tract pathology.
16.Gross specimen sampling – providing guidance on macroscopic examination and tissue sectioning.
17.Immunohistochemistry and molecular pathology – focused on biomarker detection and molecular diagnostics.
18.Ophthalmology and otolaryngology – covering diseases of the eye, ear, and related sensory organs.
19.Trachea, lung, pleura, respiratory system, and mediastinum – including neoplastic and inflammatory diseases of the
thoracic cavity.
Each cleaned PDF was converted page-wise into an image format, ensuring that the visual structure (including images,
captions, and layout) remained intact.
We used the ColQwen2 model [32], a powerful vision-language encoder, to generate unified high-dimensional embeddings
for each image-text pair. This model jointly encodes both the fine-grained visual features and the contextual diagnostic seman-
tics into a shared vector space, enabling effective multimodal similarity retrieval.
All embeddings were stored in Milvus [34], an open-source vector database optimized for large-scale similarity search.
To accelerate retrieval in high-dimensional space, we employed the hierarchical navigable small world (HNSW) indexing
algorithm [33], which is well suited for fast, approximate nearest neighbor search at scale. This infrastructure allows our
reasoning agents to retrieve highly relevant visual evidence from hundreds of thousands of multimodal entries in real time.
B Details of the Agentic Diagnostic Workflow
The intelligent diagnostic pipeline operates in four distinct stages, involving three key modules: the Agentic Router (task
planner), the VRAG Agent (retrieval and summarization), and the Pathology VLM (inference engine). Each step is described
in detail below.
Step 1: Query Understanding and Task Decomposition (Agentic Router) The Agentic Router receives the user’s in-
put and identifies the semantic structure of the task, typically involving image understanding and candidate differentiation. It
decomposes the query into subproblems mapped to each candidate diagnosis.
Step 2: Entity-Centric Multimodal Retrieval (within VRAG Agent) For each candidate diagnosis, the VRAG Agent
performs a two-step retrieval process:
•Textual Retrieval: Using candidate-specific keywords, the system retrieves the top 20 textual entries from the knowledge
base.

•Multimodal Re-ranking: Retrieved entries are then ranked by calculating joint similarity between the input image and as-
sociated image-text pairs using the ColQwen2 embedding [32]. The similarity score is computed according to Equation 1,
which captures the cross-modal relevance between the query image and each retrieved image-text pair in the embedding
space. The highest-ranked entry becomes the evidence for that candidate.
Step 3: Iterative Evidence Aggregation (VRAG Agent) The VRAG Agent [19] performs multi-turn retrieval and refine-
ment. It dynamically evaluates evidence sufficiency, performs focused re-retrieval when necessary, and aggregates the most
discriminative content into a structured prompt. The prompt includes: (1) the original user input, (2) top retrieved evidence for
each diagnosis, and (3) explicit comparative instructions.
Step 4: Reasoning and Report Generation (Pathology VLM) The structured prompt is passed to a specialized vision-
language model fine-tuned for diagnostic tasks [26]. The model conducts contrastive reasoning, constrained to the provided
evidence, and returns both a predicted diagnosis and an interpretable justification.
C Details of Models and Training
C.1 Training Dataset Construction
The training data for the agentic router was derived from the training sets of Quilt-VQA[39] and Path-VQA[38]. We first ran
inference using Patho-R1 and filtered the samples based on the correctness of its responses. Specifically, we removed questions
with overly simple reasoning or insufficient thought chains, as they were not useful for training our agent flow. From the
remaining samples, we randomly selected 2,200 questions that Patho-R1 answered incorrectly (indicating a need for RAG) and
2,200 that it answered correctly (indicating no need for RAG) to construct the training dataset. Among these selected questions,
we ensured an even split between MCQ and YORN formats. We then distilled the ground truth using QwenMax, ultimately
retaining 4,000 samples for RL training and an additional 400 samples for SFT training, as shown in Figure 5.
Specifically, to construct ground-truth decision paths Pgtfor reinforcement learning, we prompted QwenMax with carefully
designed instructions that emulate expert diagnostic reasoning. Specifically, given a user query Qorig, we provided QwenMax
with a prompt that specifies: (1) available tools (e.g., rag,classifier ), (2) a comprehensive list of anatomical partitions,
and (3) detailed behavioral guidelines for tool invocation.
The model was instructed to first output a <think> field capturing its reasoning, followed by one or more structured
<tool call> entries. For instance, it would decide whether to query external knowledge via the rag tool, invoke the
classifier tool to infer tissue system based on question content, or rely solely on internal knowledge when appropriate. A
sample prompt template and tool call schema are provided in Appendix D.
The resulting outputs serve as high-quality ground-truth action trajectories, reflecting both whether to invoke the retrieval
system and how to structure the tool usage. These trajectories form the supervision signal for training the agent’s policy π,
which aims to recover the expert-like decision plan Pgtfrom an input query.
Figure 5: Training Data Composition for Agentic Router
C.2 Training Details
All tool calls in the dataset are represented in JSON format. We use Qwen3-4B as the base model for all experiments, and
training was conducted on 8 NVIDIA RTX 4090 GPUs.
Supervised fine-tuning: We adopted the LLaMA-Factory1framework and froze the vision tower. We used a learning rate of
1e-5 and trained on 400 samples for 3 epochs.
Reinforcement learning: We adopted the verl2framework for reinforcement learning. We set the actor and critic learning
rates 1e-6 and 1e-5 respectively. For GRPO, we trained on 4k samples for 3 epochs
1https://github.com/hiyouga/LLaMA-Factory.git
2https://github.com/volcengine/verl

D Experimental Benchmarks, Evaluation Metrics, and Prompts
We presented the prompts used throughout our Patho-AgenticRAG workflow, along with the benchmarks and evaluation pro-
tocols adopted for assessment.
D.1 Multimodal Benchmarks for Pathology
We evaluate our approach using multiple multimodal benchmarks relevant to pathology. For the multiple-choice (MCQ) setting,
we adopt the PathMMU dataset, a pathology-specific benchmark comprising images and expert-annotated questions. Following
the dataset’s protocol, we downloaded pathology images originally shared via Twitter. However, due to post deletions, some
images were no longer accessible, and the corresponding questions were removed from evaluation.
The PathMMU-test-tiny split includes a total of 1,139 questions, distributed as follows: Atlas (208), EduContent (255),
PathCLS (177), PubMed (281), and SocialPath (218). The full PathMMU-test split contains 8,454 questions: Atlas (799),
EduContent (1,683), PathCLS (1,632), PubMed (2,787), and SocialPath (1,553).
To broaden evaluation coverage, we further curated pathology-focused subsets from general medical VQA datasets. Specifi-
cally, we selected 90 pathology-related examples from MedXpertQA, and used the BRIGHT Challenge subset (890 cases) from
OmniMedVQA, which focuses on diagnostic reasoning across medical specialties.
Finally, for the YORN dataset, we collected closed-ended questions from the test splits of Path-VQA and Quilt-VQA, result-
ing in 3,362 and 343 questions, respectively.
D.2 Evaluation Metrics for Multi-Model Fusion
We evaluate retrieval performance using three metrics: mean reciprocal rank at k(MRR@ k), recall@ k, and normalized dis-
counted cumulative gain at k(NDCG@ k), defined as follows:
•Recall@ kmeasures the fraction of queries for which the target textbook page is included among the top kretrieved results.
Recall@ kis given by
Recall@ k=1
|Q||Q|X
i=1I(rank i≤k), (9)
where Iis the indicator function.
•Mean Reciprocal Rank at k(MRR@ k)reports the average reciprocal rank of the target textbook page within the top k
results. Let |Q|be the number of queries and rank ithe position of the target textbook page for query i. The metric is defined
as
MRR@ k=1
|Q||Q|X
i=11
min (rank i, k+ 1), (10)
where the reciprocal rank is zero if the target textbook page does not appear in the top k.
•Normalized Discounted Cumulative Gain at k(NDCG@ k)evaluates ranking quality based on the graded relevance
of retrieved textbook pages. For NDCG calculation, the target textbook page is assigned a relevance score of 2, and the
previous and next textbook pages are assigned a score of 1, reflecting that neighboring pages may contain partially relevant
information. NDCG@ kis computed as
NDCG@ k=1
|Q||Q|X
i=1DCG i@k
IDCG i@k, (11)
where
DCG i@k=kX
j=12reli,j−1
log2(j+ 1)(12)
andreli,jis the relevance score of the j-th retrieved textbook page for query i. The ideal DCG, IDCG i@k, is computed with
pages sorted by the highest possible relevance.
D.3 Evaluation Metrics for Medical QA Tasks
To evaluate model performance on various pathology benchmarks, we adopt standard accuracy-based metrics. For closed-ended
visual question answering tasks, including multiple-choice and yes-or-no questions, we compute exact-match accuracy, where a
prediction is considered correct only if it exactly matches the ground-truth answer. In addition, we report per-category scores on
datasets such as PathMMU, whose subsets originate from diverse real-world sources (e.g., textbooks, educational slides, social
media, and literature). This allows us to assess the model’s generalization ability across heterogeneous pathology domains.

D.4 Prompts
Prompt for Agentic Router
You are a helpful dialogue assistant capable of leveraging tool calls to solve
user tasks and provide structured chat responses.
### User Query
{query }
### Available Tools
In your response, you can use the following tools: {’rag’, ’classifier’ }
### Partitions:
{"Bone andSoft Tissue", "Cytology",
"Gastrointestinal Tract Liver Gallbladder Pancreas Digestive System",
"Hematology Lymphatic System andBone Marrow",
"Infectious Diseases", "Oral andHead Neck",
"Urinary andMale Reproductive System",
"Breast", "Endocrine", "General Comprehensive", "Histology andEmbryology",
"Neonatal Pediatric andChild", "Skin Dermatology", "Central Nervous System",
"Female Reproductive System", "Gross Specimen Sampling",
"Immunohistochemistry andMolecular Pathology", "Ophthalmology Otolaryngology",
"Trachea Lung Pleura Respiratory System andMediastinum" }
### Chain of Thoughts
- You must always include the <think> field to outline your reasoning. Decide
whether to use <tool call> (possibly multiple times).
- Given a user query, first assess whether external knowledge is needed. If so,
call the rag tool.
- If the question is in multiple-choice or yes/no format, do not blindly
convert each option into a question. Instead, analyze what knowledge is
required to choose the correct answer | e.g., differential histological
features, diagnostic criteria, cellular morphologies | and formulate targeted
queries that reflect those specific informational needs.
- You may still use the options as anchors, but query construction should aim
for high signal-to-noise retrieval concise, informative, and focused.
- The rag tool should receive an informative, purpose-driven query as its
parameter.
- After determining that rag is needed, further inspect whether the
query includes indicators of tissue classification or pathology specimen
classification (e.g. mention of specific tissue, organ system, tissue
structure, etc.). If so, invoke the classifier tool in parallel.
- If the category cannot be determined from the question or option, there is no
need to call classifier, just call rag.
- The classifier tool must always be invoked with a partition parameter
whose value is exactly one of the predefined partition names. These are
case-sensitive and must come from the provided list of partitions.
- The output format for classifier must strictly follow this
structure: {"name": "classifier", "parameters": {"partition":
"<Exact Partition Name From List>" }}
- When the question and options rely mainly on an image to make a distinction,
just provide a <think> field to explore possible histological interpretations
based on general principles. Do not call any tools.
- Similarly, if the question reflects common knowledge in pathology or
histology, such as "What stain is used for nuclei?" or "Which cell secretes
collagen?", a <think> is sufficient, and no tool needs to be called.
### Output Format
<think> Your thoughts and reasoning </think>
<tool call>
{"name": "Tool name", "parameters": {""}
{"name": "... ...", "parameters": {"... ..." }}}
...
</tool call>

Prompt for Vrag Agent
Your task is **not**to directly answer the question, but to assist a
downstream system by **retrieving and summarizing factual information **
relevant to answering it.
You should focus on **describing the characteristics, criteria, or diagnostic
features **associated with the possible answer| **not**on concluding what the
answer is.
For example, if the question is whether an image shows invasive ductal
carcinoma, you should **retrieve facts about how invasive ductal carcinoma
typically presents **, especially in medical images or pathology reports.
Use reasoning steps enclosed in <think>...</think> to guide your process.
You may request external information using <search>...</search> or issue search
queries using <search> query </search>. The user will provide relevant results.
If a new image is retrieved, you may optionally crop it using <bbox>[x1, y1,
x2, y2]</bbox>.
Base your reasoning on the image, the question, and any supporting data you
collect. Use as many retrievals as necessary. Never rely solely on the initial
image.
Once you have gathered enough information, provide a concise summary of the
most **relevant evidence or diagnostic criteria **within <answer> and </answer>
tags.
Do not state whether the diagnosis or conclusion is correct|just provide the
features or findings that are relevant for making that decision.
Question:
{question }
Prompt for Patho-R1
You are a pathology expert, your task is to answer question step by step.
Use the following format:
<think> Your step-by-step reasoning, you could use the context below if needed
</think>
<answer> Your final answer, no explanation. </answer>
Question: {query }
Context: {information summrized by vrag }
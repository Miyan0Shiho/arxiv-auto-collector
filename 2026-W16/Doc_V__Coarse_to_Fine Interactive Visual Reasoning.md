# Doc-V*:Coarse-to-Fine Interactive Visual Reasoning for Multi-Page Document VQA

**Authors**: Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei, Zhenbo Luo, Jian Luan, Wei Chen, Xiang Bai

**Published**: 2026-04-15 11:12:27

**PDF URL**: [https://arxiv.org/pdf/2604.13731v1](https://arxiv.org/pdf/2604.13731v1)

## Abstract
Multi-page Document Visual Question Answering requires reasoning over semantics, layouts, and visual elements in long, visually dense documents. Existing OCR-free methods face a trade-off between capacity and precision: end-to-end models scale poorly with document length, while visual retrieval-based pipelines are brittle and passive. We propose Doc-$V^*$, an \textbf{OCR-free agentic} framework that casts multi-page DocVQA as sequential evidence aggregation. Doc-$V^*$ begins with a thumbnail overview, then actively navigates via semantic retrieval and targeted page fetching, and aggregates evidence in a structured working memory for grounded reasoning. Trained by imitation learning from expert trajectories and further optimized with Group Relative Policy Optimization, Doc-$V^*$ balances answer accuracy with evidence-seeking efficiency. Across five benchmarks, Doc-$V^*$ outperforms open-source baselines and approaches proprietary models, improving out-of-domain performance by up to \textbf{47.9\%} over RAG baseline. Other results reveal effective evidence aggregation with selective attention, not increased input pages.

## Full Text


<!-- PDF content starts -->

Doc-V∗: Coarse-to-Fine Interactive Visual Reasoning for Multi-Page
Document VQA
Yuanlei Zheng1*, Pei Fu2*, Hang Li2, Ziyang Wang1,
Yuyi Zhang1,Wenyu Ruan1,Xiaojin Zhang3,Zhongyu Wei4,
Zhenbo Luo2†,Jian Luan2,Wei Chen1†,Xiang Bai1
1School of Software Engineering, Huazhong University of Science and Technology,
2MiLM Plus, Xiaomi Inc.,
3School of Computer Science and Technology, Huazhong University of Science and Technology,
4School of Data Science, Fudan University
Abstract
Multi-page Document Visual Question Answer-
ing requires reasoning over semantics, layouts,
and visual elements in long, visually dense
documents. Existing OCR-free methods face
a trade-off between capacity and precision:
end-to-end models scale poorly with document
length, while visual retrieval-based pipelines
are brittle and passive. We proposeDoc- V∗,
anOCR-free agenticframework that casts
multi-page DocVQA as sequential evidence
aggregation.Doc- V∗begins with a thumbnail
overview, then actively navigates via semantic
retrieval and targeted page fetching, and aggre-
gates evidence in a structured working memory
for grounded reasoning. Trained by imitation
learning from expert trajectories and further
optimized with Group Relative Policy Opti-
mization,Doc- V∗balances answer accuracy
with evidence-seeking efficiency. Across five
benchmarks,Doc- V∗outperforms open-source
baselines and approaches proprietary models,
improving out-of-domain performance by up to
47.9%over RAG baseline. Other results reveal
effective evidence aggregation with selective
attention, not increased input pages.
1 Introduction
Understanding multi-page, visually rich docu-
ments—such as academic papers, financial re-
ports, and industrial manuals—remains a core chal-
lenge inDocument Visual Question Answering
(DocVQA) (Mathew et al., 2021; Tito et al., 2023).
Unlike plain text, such documents convey infor-
mation through a complex interplay of textual se-
mantics, spatial layouts, and visual elements (e.g.,
tables and figures) (Ding et al., 2025). Conven-
tionalOCR-basedpipelines linearize document
images into text before reasoning (Memon et al.,
2020; Wang et al., 2024; Appalaraju et al., 2021),
but inevitably lose fine-grained layout cues and
*Equal contribution
†Corresponding authorsuffer from cascading OCR errors. RecentOCR-
freeorpure-visionapproaches instead model doc-
uments directly as images using multimodal large
language models (MLLMs) (Lee et al., 2023; Kim
et al., 2022; Liu et al., 2024b), enabling joint vi-
sual–semantic reasoning and improved robustness.
However, existing pure-vision methods face a
fundamental trade-off betweencapacityandpre-
cision.End-to-endmodels process entire docu-
ments as long image sequences (Zhu et al., 2025;
Hu et al., 2025; Bai et al., 2025), but scale poorly to
long documents due to quadratic attention cost, con-
text length limits, and the "lost-in-the-middle" ef-
fect (Liu et al., 2024a). In contrast, visual retrieval-
augmented generation (RAG) systems reduce noise
by retrieving top- kpages before generation (Cho
et al., 2024; Faysse et al., 2025; Song et al., 2025),
yet suffer from retrieval errors, sensitivity to hyper-
parameters, and limited multi-hop reasoning. Criti-
cally, both paradigms remainpassive: they process
a fixed input without adapting their strategy as new
evidence emerges.
We argue that this limitation arises from a mis-
match with human document-reading behavior.
Guided byActive Vision Theory(Aloimonos et al.,
1988), human experts treat perception as a goal-
directed process: they first obtain a global struc-
tural overview, then iteratively seek, verify, and in-
tegrate evidence while maintaining working mem-
ory. Inspired by this cognitive process, we propose
Doc-V∗, anOCR-free agentic frameworkthat
formulates multi-page DocVQA as asequential ev-
idence aggregation process.Doc- V∗begins with
aGlobal Thumbnail Overviewthat provides a low-
cost structural prior, and then alternates between
structured visual reasoninganddocument naviga-
tion actions, including semantic retrieval and tar-
geted page fetching. This interactive reasoning
allows the agent toactive perceptionandpiece
together discontinuous visual evidencebefore an-
swering. Figure 1 shows the agent workflow ofarXiv:2604.13731v1  [cs.CL]  15 Apr 2026

Question:
How many signatures are in the 
document?
From the thumbnail,  pages 
7–8 show visibly handwritten 
signatures scribbles on lines, 
so I prioritize fetching 
pages 7 and 8 .
Answer:
Two signatures in 
the document.
Retrieve
[ pages with signatures ]Two signatures 
detected (pages 
7 & 8). Possible 
missing instances.
→ retrieve more 
pages.
No new signatures 
found. Existing two 
signatures are 
sufficient → 
finalize.Page 1 Page 2 Page 3
Page 4Document thumbnail
Page 5 Page 6
Page 7 Page 8
Figure 1:The Doc- V∗agent workflow for multi-page document VQA.It adopts anactive perceptionparadigm
by planning from a global thumbnail view and iteratively deciding when to fetch high-resolution pages or perform
semantic searches, aggregating evidence in a structured working memory for grounded answering.
Doc-V∗.
To trainDoc- V∗, we adopt a two-stage optimiza-
tion strategy. We first perform supervised fine-
tuning using high-quality interaction trajectories
synthesized by GPT-4o, providing a strong cold
start. We then apply Group Relative Policy Op-
timization (GRPO) (Guo et al., 2025) to jointly
optimize answer accuracy and evidence-seeking ef-
ficiency through reward signals that account for an-
swer quality, evidence discovery, and format com-
pliance. Extensive experiments on five benchmarks
demonstrate thatDoc- V∗consistently outperforms
existing open-source baselines and rivals propri-
etary models likeGPT-4o, particularly in out-of-
domain settings where it achieves up to a47.9%
improvementover static RAG baselines, as well
asrobustnessunder variations in retrieval tools
and hyperparameters. We also demonstrate that
long-document understanding hinges oneffective
aggregation of evidencewith selective attention
rather thansheer input pages, which is crucial to
the success ofDoc-V∗.
2 Related Work
Visual Document Question Answering (DocVQA)
has progressed from single-page inputs to long
and multi-page documents, driven by the increas-
ing demand for handling complex real-world doc-
ument understanding scenarios. Existing meth-
ods mainly follow two paradigms: 1)OCR-based
DocVQAOCR-based approaches first extract
textual and layout structures via OCR and doc-ument parsing, followed by reasoning over struc-
tured representations (Tito et al., 2023; Zhang et al.,
2024; Luo et al., 2024; Fujitake, 2024; Li et al.,
2024; Duan et al., 2025; Nacson et al., 2025; Xie
et al., 2024). While effective on clean and well-
formatted documents, these pipelines inevitably
suffer from cascading OCR and layout errors and
generalize poorly to noisy or out-of-domain scenar-
ios; 2)OCR-free Pure-Vision DocVQARecent
OCR-free methods leverage large vision–language
models to reason directly over document images,
preserving rich visual and spatial cues. However,
scaling to long documents remains challenging. Ex-
isting approaches include: (i)end-to-endmodels
that process all pages jointly (Hu et al., 2025; Zhu
et al., 2025), which scale poorly with document
length, as computational cost and memory con-
sumption grow rapidly with the number of pages;
(ii)retrieval-basedmethods that select top- kpages
before generation (Cho et al., 2024; Yu et al., 2024;
Chen et al., 2024; Tanaka et al., 2025; Wang et al.,
2025; Wu et al., 2025; Shi et al., 2025), improving
efficiency but remaining sensitive to retrieval errors
and fixed hyperparameters; and (iii)agent-based
systems that iteratively explore documents (Xu
et al., 2025; Yang et al., 2025; Yue et al., 2025),
which introduce interaction at the cost of increased
complexity. In contrast, our method formulates
DocVQA as a sequential evidence aggregation pro-
cess, enabling a single OCR-free agent to actively
and efficiently aggregate visual evidence over long
documents.

Retrieve
Tool
Fetch
Tool
Policy 
ModelReference 
Model
Env
T1,
T2,
T3,
...
TGReward 
Functionr1,
r2,
r3,
...
rGAdv 1,
Adv 2,
Adv 3,
...
Adv G(c) Reinforcement Learning With GRPO
Instruction
+ User Query
Policy 
ModelInstruction
+ User Query(b) Supervised Fine -Tuning
TT^
Pred
TrajectoryGT
Trajectory
Adv.KL
Cross -Entroy  Loss(a) Construct Train Dataset
Instruction
+ User Query
Thumbnail
+
GeneratorRaw Data
Too few 
images
TrajectoryAnswer ErrorFigure 2:Overview of the training pipeline for Doc- V∗.(a) Training data construction.Documents and queries
are paired to generate thumbnail-guided reasoning trajectories, followed by quality filtering.(b) Supervised fine-
tuning (SFT). (c) Reinforcement learning with GRPO.
3 Method
3.1 Formulation and Cognitive Motivation
Faced with lengthy, unfamiliar documents, human
experts exhibit pronouncedgoal-directednessand
proactivityrather than reading cover-to-cover: they
navigate using structural cues and keyword-like
searches, and iteratively update their strategy as
evidence is found. This behavior is consistent
withActive Vision(Aloimonos et al., 1988), which
views perception as goal-directed sampling to re-
duce uncertainty, andResource-Rational Cogni-
tion(Lieder and Griffiths, 2020), which trades off
information gain against processing costs. Mo-
tivated by these principles, we proposeDoc- V∗,
formulatingMulti-page Document VQAas aSe-
quential Decision Process: given a document
D={p 1, . . . , p N}and a question Q, anOCR-free
MLLM-based agent πθinteracts with the document
environment for up to Tsteps. At step t, the agent
receives its observation Ot, performs reasoning,
and selects an action at∈ A; the environment then
returns feedback Et+1, which is incorporated into
the next observation Ot+1. This closed-loop for-
mulation enablesselective evidence acquisition
and theintegration of scattered visual cues into
a coherent reasoning chain.
3.2 Environment Design
Document Visual RepresentationOur agent is
built upon the Qwen-2.5-VL (Bai et al., 2025) archi-
tecture, which comprises a visual encoder V(adopt-
ing ViT (Dosovitskiy, 2020) architecture), a multi-
layer perceptron projection module M, and a largelanguage model backbone L. We pre-compute and
cache the visual tokens vi=M(V(p i))∈RLi×d
for all pages {pi}N
i=1within Dat theirnative high
resolution(capped at 1024×768 ), where Liis the
token count and dis the hidden dimension. Cru-
cially, these visual tokens are not fed to the agent
all at once but are dynamically requested by the
agent based on its decisions.
Initial ObservationBefore interaction begins,
we design aGlobal Thumbnail Overview ˜Dfor
the document, inspired by the human behavior of
first "rapidly flipping through pages" to grasp the
overall structure when browsing a document. Con-
cretely, we partition the document into groups of
pages, resize each page to a thumbnail ( 256×256 ),
reorganize each group into a grid image and anno-
tate each thumbnail with itsabsolute page number.
While body text details become indiscernible at
this resolution, rich structural information remains
visible likedocument type,section layout,chart
distributionandlarger-font titles. This coarse-
grained global perception provides considerable
navigational priors for subsequent fine-grained ex-
ploration. Formally, the initial input fed to the
agent is denoted as: Oo={Q, ˜D}, where ˜Dpossi-
bly consisting of one or multiple grid images.
Please refer to Appendix A for the detail of the
Environment Design.
3.3 Action Space
We define three types of atomic actions for the
agent that capture common human document-
reading behaviors.
1. Retrieval ActionThe retrieval action is in-

tended to approximate the "Ctrl+F search within
document" behavior, but at the level of page images.
To trigger this, the agent need emits a structured
command: "<retrieval_page>q t", which signi-
fies a decision to retrieve document images using
the textual query qt. The query can differ from
the original question Q, allowing iterative refine-
ment as evidence accumulates. The environment
then calls an external multimodal retriever (e.g.,
ColQwen (Faysse et al., 2024)), ranking pages in
D \ P visited and returns the top- kunvisited pages,
where Pvisited is an external variable that maintains
a set of visited pages to avoid redundancy.
2. Fetch ActionThe fetch action requests spe-
cific pages by absolute indices via the command
"<fetch_page>[i 1, . . . , i m]". Upon receiving this,
the environment parses the index list and retrieves
the exact pages specified. This action facilitates
several common navigation strategies: 1) direct
page fetching based on visual features observed
in the thumbnail view (e.g., TOC and chart posi-
tions); 2) needing to view adjacent pages before
or after the current page for complete context af-
ter reading a certain page; 3) responding to page
numbers explicitly mentioned in the user question
(e.g.,"How many baselines are there in the table
on page three?").
For both actions, the environment returns the
cached high-resolution visual tokensof the re-
quested pages. Each page’s visual tokens are pre-
fixed with a textual page number identifier (e.g.,
"Page 5:" ) to ensure the agent can correctly asso-
ciate the visual content with its specific page num-
ber. If a requested page has already been visited,
the environment returns atext reminderinstead of
re-inputting the visual tokens. We denote Etas the
environment feedbackat interaction stept≥1.
3. Answer ActionWhen the agent determines
that sufficient evidence has been gathered, it termi-
nates the interaction by executing the answer action
by generating "<answer>y" , where yis the final
answer string.
3.4 Structured Visual Reasoning
To make the agent’s decision process explicit
and auditable, we enforce a fixedthink-acting
interaction protocol, a ReAct (Yao et al., 2022)
reasoning style with visual feedback. At each
step, the agent’s output must follow the format:
"<think>···</think><action>···</action>" ,
where <action> instantiates exactly one action
from §3.3 with the required arguments.We further structure <think> into 3 blocks,
with a slight distinction between the first turn and
later turns. At turn t=0, given the initial obser-
vation, i.e., document thumbnails with question,
<think> consists of: 1) <analysis> : a coarse
document-level inspection from thumbnails, iden-
tifying likely question-relevant regions/pages and
key visual cues; 2) <plan> : an explicit subgoal de-
composition and an interaction plan, which guides
subsequent actions under a limited step budget;
3)<summary> : a compact summary of the initial
inspection and plan. At turns t>0, given newly re-
turned high-resolution pages, <think> consists of:
1)<analysis> : Page-by-page content analysis of
newly returned pages, evaluating each page’s rele-
vance to the user question, determining whether the
evidence is sufficient to answer, and deciding on
the next optimal action that can reduce uncertainty;
2)<relevant_pages> : Explicitly outputs the list
of page numbers judged to be relevant among the
pages returned in the current turn. This compo-
nent forces the agent to make binary relevance
judgments, facilitating subsequent reward signal
computation and model evaluation; 3) <summary>
An incremental information summary for the cur-
rent turn, which together with historical summaries
constitutes the agent’sWorking Memory.
As interaction proceeds, image-text interleaved
tokens accumulate and pages may arrive out of
order, which can cause the agent to forget and
drift (e.g., forgetting resolved sub-questions or
repeatedly fetching a certain page). To mitigate
this, we feed the agent anaugmented observation
Ot=E t∪ {W t}, t≥1 , where theWorking Mem-
oryWt= Concat(S 0, . . . , S t−1)concatenates all
previous <summary> within <think> . Please refer
to Appendix B for the detail of theAgent Environ-
ment Interaction Protocol.
3.5 Training
We adopt a standard two-stage training pipeline
to obtain an agent that is both tool-competent and
exploration-efficient under a bounded interaction
budget. First, we perform supervised fine-tuning
with a cross-entropy objective on distilled closed-
loop interaction trajectories, where a strong teacher
interacts with the real environment and we compute
loss only on agent-generated tokens; we further
filter trajectories by format validity, answer correct-
ness, and evidence-page sanity, yielding9,019high-
quality trajectories constructed from MP-DocVQA
and DUDE. Second, we apply GRPO reinforce-

ment learning using only outcome supervision:
we filter2,048non-overlapping training examples,
stratify them into easy/medium/hard buckets esti-
mated by the SFT policy via multiple rollouts, and
train the agent by sampling groups of trajectories in
the same closed-loop environment and optimizing
a weighted reward that combines answer correct-
ness, evidence retrieval quality, and format validity.
Full training details are provided in Appendix C.
4 Experiments
4.1 Experimental Setup
DatasetsOur raw training data is sourced
fromMP-DocVQA(Tito et al., 2023) and
DUDE(Van Landeghem et al., 2023). Evaluation
is conducted under two settings. (1)In-Domain
evaluation is performed on the test splits of MP-
DocVQA and DUDE. (2)Out-of-Domain (OOD)
evaluation is carried out on three challenging bench-
marks:SlideVQA(Tanaka et al., 2023),Long-
DocURL(Deng et al., 2025), andMMLongBench-
Doc(Ma et al., 2024). These benchmarks cover
diverse document types and reasoning challenges,
enabling a comprehensive evaluation of generaliza-
tion beyond the training domain. Detailed statis-
tics and dataset characteristics are provided in Ap-
pendix E.
Evaluation MetricsAll methods are evaluated
using theofficial metrics and evaluation protocols
of each benchmark. Specifically, we report ANLS
for DUDE and MPDocVQA, F1 score for Slide-
VQA, and Accuracy for MMLongBench-Doc and
LongDocURL.
Agent and Environment SetupOur agent is
initialized fromQwen-2.5-VL-7B-Instruct(Bai
et al., 2025). For the retrieval_page , we em-
ployColQwen(Faysse et al., 2025) as the external
retriever. Retrieval budget is dynamically set to
k= min(⌈N/10⌉,4) to balance information cov-
erage and context efficiency, and the maximum
interaction horizon is fixed to T= 8 steps during
both training and inference. The optimization ob-
jective incorporates a composite reward function
balancing answer correctness ( ωans= 0.6 ), evi-
dence recall ( ωevi= 0.3 ), and structural validity
(ωstruct= 0.1 ). Specific training hyperparameters
and further implementation details are provided in
Appendix D.4.2 Main Results
We compareDoc- V∗with a broad suite of base-
lines spanning three paradigms: (i)End-to-End
(E2E)models including HiVT5 (Tito et al., 2023),
mPLUG-DocOwl2 (Hu et al., 2025), Docopi-
lot (Duan et al., 2025), DocVLM (Nacson et al.,
2025), and InternVL3 (Zhu et al., 2025); (ii)
Retrieval-Augmented Generation (RAG)meth-
ods including CREAM (Zhang et al., 2024),
M3DocRAG (Cho et al., 2024), VisRAG (Yu
et al., 2024), SV-RAG (Chen et al., 2024),
VDocRAG (Tanaka et al., 2025), MoLoRAG (Wu
et al., 2025), and URaG (Shi et al., 2025); and
(iii)Agent-basedapproaches including VRAG-
RL (Wang et al., 2025) and CogDoc (Xu et al.,
2025). We additionally report closed-source sys-
tems (Gemini-1.5-Pro (Team et al., 2024), GPT-
4o mini, GPT-4o (Hurst et al., 2024), GPT-4.1,
and Claude-3.7-Sonnet) as reference points, and
include Qwen2.5-VL (Bai et al., 2025) along with
its RAG Top-5 variant as direct backbone baselines.
Detailed descriptions of these baseline methods are
provided in Appendix F. As shown in Table 1, our
GRPO-enhanced model achieves the best overall
performance among open-source methods on four
of five benchmarks, while remaining competitive
on the remaining benchmark.
On the In-domain benchmarks (DUDE and MP-
DocVQA),Doc- V∗achieves strong accuracy. On
DUDE, it reaches64.5ANLS,outperforming all
open-source baselinesand alsosurpassing some
closed-source models reported, including GPT-
4o (54.1) and Claude-3.7-Sonnet (58.1). On MP-
DocVQA, our method attains86.2ANLS, remain-
ing highly competitive with URaG (88.2).
On the Out-of-Domain benchmarks,Doc- V∗
shows clear generalization advantages. On Slide-
VQA, our model achieves 77.2 F1, outperforming
SlideVQA-trained baselines CogDoc (67.9). It also
sets new open-source highs on long-context bench-
marks, scoring 42.1 accuracy on MMLongBench-
Doc and 56.3 accuracy on LongDocURL. These
results indicate thatDoc- V∗maintains robust long-
context evidence localization and aggregation abil-
ity when transferring to diverse document domains
and substantially longer inputs.
To isolate the effect of the agentic policy and
GRPO training, we compareDoc- V∗against
Qwen2.5-VL and Qwen2.5-VL (RAG Top-5)
under the same 7B scale. Static retrieval is
beneficial—Qwen2.5-VL (RAG Top-5) improves

Table 1:Comparison of different methods on five long-context and multi-page document understanding
benchmarks.The results are reported onDUDE(ANLS),MPDocVQA(ANLS),SlideVQA(F1),MMLongBench-
Doc(Acc), andLongDocURL(Acc). “Param.” denotes the parameter scale (referring specifically to theGenerator
for RAG methods). “Backbone” indicates the underlying LLM or LVLM used. “Paradigm” categorizes methods
into End-to-End (E2E), Retrieval-Augmented Generation (RAG), orAgent. The best and second-best results
among Open Source methodsare highlighted inboldand underlined , respectively. Scores marked with an asterisk
(∗) indicate that the method’s backbone was supervised fine-tuned on that specific benchmark’s training set. Red
subscripts in parentheses indicate the absolute performance gain over the baseline (Qwen2.5-VL).
Method Backbone Param ParadigmDUDE
(ANLS)MPDocVQA
(ANLS)SlideVQA
(F1)MMLong.
(Acc)LongDoc.
(Acc)
Closed Source
Gemini-1.5-Pro - - E2E 46.0 - - 28.2 50.9
GPT-4o mini - - E2E 46.5 - 60.7 28.6 -
GPT-4o - - E2E 54.1 67.4 65.8 42.8 64.5
GPT-4.1 - - E2E 50.2 - 74.7 45.6 -
Claude-3.7-Sonnet - - E2E 58.1 - 76.3 33.9 -
Open Source
HiVT5(PR)DiT / T5 0.3B E2E 23.1 62.0*- - -
CREAM(ACM MM’24)Pix2Struct / LLaMa2 7B RAG 52.5*74.3*- - -
mPLUG-DocOwl2(ACL’25)ViT / LLaMa 8B E2E 46.8*69.4*27.8 13.4 5.3
M3DocRAG(arXiv’24)Qwen2-VL 7B RAG 39.5 84.4 55.7 21.0 35.1
VisRAG(ICLR’25)MiniCPM-V 2.6 8B RAG 43.1 - 52.4 18.8 41.9
SV-RAG(ICLR’25)InternVL2 4B RAG 45.0 71.0 34.3*23.0 -
VDocRAG(CVPR’25)Phi3-Vision 4B RAG 44.0*62.6 42.0 18.4 39.8
Docopilot(CVPR’25)InternVL2 8B E2E 40.7*81.3*43.1 28.8 -
DocVLM(CVPR’25)Qwen2-VL 7B E2E 47.4 84.5 - - -
InternVL3(arXiv’25)InternViT / Qwen2.5 8B E2E 47.4 80.8 64.4 24.1 38.7
VRAG-RL(NeurIPS’25)Qwen2.5-VL 7B Agent - - - 26.6 44.9
MoLoRAG(EMNLP’25)Qwen2.5-VL 7B RAG - - - 41.0 51.9
CogDoc(arXiv’25)Qwen2.5-VL 7B Agent 46.2*75.0 67.9*33.0 -
URaG(AAAI’26)Qwen2.5-VL 7B RAG 57.6*88.2*- 33.8 52.2
Ours
Qwen2.5-VL (Baseline) Qwen2.5-VL 7B E2E 51.9 75.2 55.2 28.0 32.9
Qwen2.5-VL (RAG Top-5) Qwen2.5-VL 7B RAG 52.2 (+0.3) 77.4 (+2.2) 62.9 (+7.7) 36.1 (+8.1) 37.8 (+4.9)
Doc-V∗(SFT) Qwen2.5-VL 7B Agent 58.1 (+6.2) 81.3 (+6.1) 73.8 (+18.6) 39.8 (+11.8) 53.0 (+20.1)
Doc-V∗(GRPO) Qwen2.5-VL 7B Agent64.5 (+12.6) 86.2 (+11.0) 77.2 (+22.0) 42.1 (+14.1) 56.3 (+23.4)
over the vanilla backbone, e.g., 28.0 →36.1 on
MMLongBench-Doc and 32.9 →37.8 on Long-
DocURL. Nevertheless, our proposed method
yields substantially larger gains at the same param-
eter scale, improving over RAG Top-5 by +12.3 on
DUDE (52.2 →64.5) and +18.5 on LongDocURL
(37.8→56.3). These results demonstrate that op-
timizing a multi-step evidence-seeking policy via
GRPO offers superior robustness compared to fixed
top-kretrieval, allowing small open-source models
to rival powerful closed-source models in complex
document understanding.
4.3 Analysis of Page-Level Retrieval
Figure 3 analyzes the trade-off between the av-
erage number of input pages and both page-level
evidence quality and downstream task performance.
This analysis directly probes how different methods
handle evidence under constrained budgets.
Formultimodal RAG, increasing the number
of retrieved pages exhibits a characteristic non-
monotonic trend: performance initially improvesas more relevant pages are included, but degrades
once additional pages introduce noise. This behav-
ior highlights two structural limitations of RAG-
style pipelines. First, performance is highly sen-
sitive to the choice ofTop- K. Second, evidence
selection and reasoning are loosely coupled—the
generator must attend over a fixed, noisy context
without explicit mechanisms for evidence valida-
tion or revision.
In contrast,Doc- V∗frames long-document un-
derstanding as aprogressive evidence aggrega-
tion process. Instead of consuming all pages at
once, the agent incrementally explores the doc-
ument, extracts candidate evidence, and explic-
itly decides which pages are relevant at each step.
This difference is reflected in thePage-F1 metric,
which measures the alignment between the pages
ultimately selected by the model and the ground-
truth evidence pages.
Under comparable average input budgets,Doc-
V∗consistently achieves substantially higher Page-

10305070
0 510 15 20 25 30F1 (Page)
Avg. Input Pages1030507090
0 5 10 15 20F1 (Page)
Avg. Input Pages10305070
051015202530354045F1 (Page)
Avg. Input Pages
50607080
0 5 10 15 20Score (F1)
Avg. Input Pages30405060
0 5 10 15 20 25 30Score (Acc)
Avg. Input Pages2530354045
051015202530354045Score (Acc)
Avg. Input PagesLower Bound Lower Bound Lower Bound
Lower Bound
Lower BoundLower Bound
(a)SlideVQA (b)LongDocURL (c)MMLongBench -DocDoc-V*All PagesRAG
Doc-V*All PagesRAG
Doc-V*All PagesRAG
Doc-V*All PagesRAG
Doc-V*All PagesRAG
Doc-V*All PagesRAGFigure 3: Efficiency–effectiveness trade-off acrossSlideVQA,LongDocURL, andMMLongBench-Doc.The
top row reports Page-F1, measuring the quality of page selection under different input budgets, while the bottom
row shows downstream task performance.For Doc- V∗, Page-F1 is computed based on the pages that the model
explicitly predicts as relevant, i.e., the model outputs a set of relevant_pages , which are then compared against
the ground-truth evidence pages to compute F1.
F1 than RAG across SlideVQA, LongDocURL,
and MMLongBench-Doc. Importantly, this im-
provement does not arise from retrieving more
pages, but fromselectively consolidating evidence
across multiple interaction steps. Early observa-
tions guide hypothesis formation, while later page
accesses serve to verify, refine, or reject these hy-
potheses.
These results suggest thatlong-document un-
derstanding is not limited by insufficient con-
text, but by the model’s ability to organize and
integrate evidence. Revisiting the behavior of
multimodal RAG, increasing the number of input
pages primarily amplifies irrelevant or weakly re-
lated signals, while lacking explicit mechanisms
for evidence consolidation. As a result, evidence
becomes diluted rather than reinforced, leading to
degraded reasoning performance.
4.4 Robustness Analysis
In this section, we analyze the robustness of our
framework regarding the number of reasoning steps
and the efficiency trade-off compared to traditional
retrieval methods. More analysis see Appendix G
Impact of Document LengthFigure 4 shows
performance across different document length
ranges. BothAll PagesandRAGexhibit a clear
performance degradation as document length in-
creases, whereasDoc- V∗maintains consistentlyTable 2:Comparison of different retrievers on
MMLongBench-Doc.
Retriever Model Avg. Pages Page-F1 Overall SIN MUL UNA
ColQwenQwen2.5-VL 6.0 30.9 35.5 37.0 13.470.4
Doc-V∗5.649.7 42.1 54.6 23.545.7
BGE-LargeQwen2.5-VL 9.0 17.6 33.0 31.2 9.877.1
Doc-V∗8.434.0 36.3 45.7 18.545.7
BM25Qwen2.5-VL 10.0 20.5 32.9 32.7 11.371.3
Doc-V∗9.236.8 37.5 48.4 20.443.0
strong results across all ranges. BothAll Pagesand
RAGsuffer from substantial performance degra-
dation as document length increases, whileDoc-
V∗remains consistently strong. In the longest-
document regime ( >80 pages),Doc- V∗outper-
forms RAG by31.7%(40.7 vs. 30.9) and exceeds
theAll Pagessetting by a large margin of85.8%
(40.7 vs. 21.9), demonstrating itseffectivenessand
robustnessfor long-document understanding.
Efficiency and CostTo evaluate the efficiency
and computational cost of different document pro-
cessing strategies, a subset of samples with long
documents is randomly selected for analysis. These
samples are characterized by a large number of
pages, with an average document length of107.3
pages, which provides a representative setting for
assessing scalability under realistic long-document
scenarios. Figure 5 presents a comparative analysis
of inference latency and GPU memory consump-
tion across different methods. The results indicate

46.2
37.739.4
27.630.946.5
4048
38.840.7
37.1
28.6 28.3
21.4 21.9 20253035404550
<20 20-40 40-60 60-80 >80Acc
Page Num2.38.6
11.29.8
Doc-V*RAG All PagesFigure 4: Accuracy vs. document length under different
methods (RAG uses top-k = 5 retrieval).
that processing the entire document at once leads
to substantially higher inference latency and GPU
memory consumption, as all pages must be loaded
and processed simultaneously. By contrast, the
standard RAG baseline significantly reduces both
latency and memory footprint by restricting compu-
tation to a small subset of retrieved pages.Doc- V∗
occupies a middle ground between these two ex-
tremes: while incurring higher cost than RAG due
to iterative page access and multi-step reasoning, it
avoids the prohibitive overhead of full-document
processing and achieves a more favorable balance
between efficiency and document coverage.
Impact of Different RetrieversTable 2 shows
thatDoc- V∗maintains strong overall performance
across retrievers with substantially different ca-
pabilities. Even when coupled with weak text-
based retrievers (BM25 (Robertson et al., 2009),
BGE-Large (Xiao et al., 2023)), which suffer from
low Page-F1 and increased noise due to OCR and
layout loss, Doc- V∗incurs only moderate perfor-
mance degradation, indicating limited dependence
on high-quality retrieval. Unlike conventional RAG
pipelines where downstream performance is tightly
coupled with retrieval recall, this robustness stems
fromDoc- V∗’s active compensation mechanism:
when initial retrieval misses critical evidence, the
model detects contextual insufficiency and proac-
tively recovers missing pages via browsing actions
(e.g.,fetch_page ), effectively acting as an intel-
ligent correction layer rather than a passive con-
sumer.
4.5 Ablation Study
To validate the design choices of the proposed agen-
tic framework, we conduct ablation experiments on
MMLongBench-Doc, focusing on both the cog-
nitive modules that govern the agent’s reasoning
process and the navigation actions that support evi-
/uni00000013/uni00000011/uni00000013 /uni00000015/uni00000011/uni00000018 /uni00000018/uni00000011/uni00000013 /uni0000001a/uni00000011/uni00000018 /uni00000014/uni00000013/uni00000011/uni00000013 /uni00000014/uni00000015/uni00000011/uni00000018 /uni00000014/uni00000018/uni00000011/uni00000013 /uni00000014/uni0000001a/uni00000011/uni00000018 /uni00000015/uni00000013/uni00000011/uni00000013
/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000035/uni00000024/uni0000002a/uni00000027/uni00000052/uni00000046/uni00000010/uni00000039/uni0000000d/uni00000024/uni0000004f/uni0000004f/uni00000003/uni00000033/uni00000044/uni0000004a/uni00000048
/uni00000018/uni00000011/uni0000001b/uni00000014/uni0000001a/uni00000011/uni0000001c/uni00000014/uni0000001c/uni00000011/uni00000013/uni0000000b/uni00000044/uni0000000c
/uni00000013 /uni00000014/uni00000013 /uni00000015/uni00000013 /uni00000016/uni00000013 /uni00000017/uni00000013 /uni00000018/uni00000013 /uni00000019/uni00000013 /uni0000001a/uni00000013
/uni0000002a/uni00000033/uni00000038/uni00000003/uni00000033/uni00000048/uni00000044/uni0000004e/uni00000003/uni00000030/uni00000048/uni00000050/uni00000052/uni00000055/uni0000005c/uni00000003/uni0000000b/uni0000002a/uni00000025/uni0000000c/uni00000035/uni00000024/uni0000002a/uni00000027/uni00000052/uni00000046/uni00000010/uni00000039/uni0000000d/uni00000024/uni0000004f/uni0000004f/uni00000003/uni00000033/uni00000044/uni0000004a/uni00000048
/uni00000015/uni00000013/uni00000011/uni00000015/uni00000016/uni00000014/uni00000011/uni0000001a/uni00000019/uni00000018/uni00000011/uni00000016/uni0000000b/uni00000045/uni0000000cFigure 5:(a): Comparison of average inference latency
per sample across different methods.(b): Comparison
of average peak GPU memory consumption per sample
under different methods.
dence acquisition.
Importance of Multi-granularity Page Under-
standingRemoving either the global thumb-
nail overview or the page-by-page analysis module
causes significant performance drops of 4.9 and 4.7
accuracy points, respectively (Table 3), indicating
that effective long-document reasoning relies on
multi-granularity page understanding. The global
overview provides structural cues for efficient navi-
gation, while fine-grained analysis enables precise
evidence extraction; using only one level of per-
ception leads to either inefficient exploration or
insufficient evidence recovery.
Complementary Roles of Retrieval and Fetch
ActionsWe analyze the agent’s navigation be-
havior using both page-level metrics (Table 5) and
action ablation on MMLongBench-Doc (Table 4).
Theretrieval_page action achieves higher recall
but lower precision, serving as a coarse semantic
filter, while fetch_page provides higher precision
for fine-grained evidence grounding. Ablation re-
sults further confirm their complementarity: remov-
ing retrieval leads to inefficient exploration (more
pages), while removing fetch degrades accuracy.
Combining both yields the best accuracy–efficiency
trade-off, forming a coarse-to-fine evidence aggre-
gation strategy.
Conclusion
This paper introducesDoc-V∗, an OCR-free agen-
tic framework for multi-page document VQA via
active evidence aggregation. Experiments on five
benchmarks show gains over strong open-source
baselines and competitive results against propri-
etary models, particularly on long and OOD docu-

Table 3: Ablation study on the cognitive modules of
theDoc- V∗agent.T: Global Thumbnail Overview;A:
Page-by-page content analyis;M: Memory.
Cognitive Modules MMLong. LongDoc. SlideVQA
T A M (Acc) (Acc) (F1)
✓✓ ✓ 39.8 53.0 73.8
✗✓ ✓ 34.9 (-4.9) 46.3 (-6.7) 68.3 (-5.5)
✓✗✓ 35.9 (-4.7) 49.5 (-3.5) 71.8 (-2.0)
✓ ✓✗ 36.4 (-3.4) 47.1 (-5.9) 69.8 (-4.0)
Table 4: Action ablation study on MMLongBench-Doc.
Removing either retrieval or fetch leads to clear perfor-
mance degradation.
Setting Acc↑Avg. Pages↓
Doc-V∗39.86.4
w/o Retrieval 34.9 14.2
w/o Fetch 35.25.9
ments. These findings position selective evidence
aggregation as a robust alternative to fixed-context
and retrieval-augmented methods.
Limitations
This work is subject to several limitations. First,
all experiments are conducted with a single back-
bone (Qwen2.5-VL), and the effectiveness of the
proposed agentic framework across different vi-
sion–language backbones is not systematically
evaluated. Although the method is conceptually
backbone-agnostic, architectural differences may
affect evidence aggregation and tool usage behav-
iors. Second, Doc- V∗is evaluated only in the
single-document setting; its performance on multi-
document scenarios, where evidence must be ag-
gregated across multiple heterogeneous documents,
remains unexplored and requires further study.
Ethical Considerations
Most datasets used in this work are publicly avail-
able benchmarks for document visual question an-
swering and are utilized in accordance with their
respective licenses. The proposed framework does
not introduce new data collection or annotation pro-
cesses involving human subjects. Similar to exist-
ing vision–language models, Doc- V∗may produce
incorrect or incomplete answers due to hallucina-
tion or imperfect evidence aggregation, particularly
on complex or ambiguous documents. As with
prior work, its outputs are intended to support doc-Table 5:Page-level analysis of agent tool usage and
retrieval quality across three benchmarks. RPde-
notes pages retrieved by the retrieval_page , while
FPdenotes pages obtained via the fetch_page .Ratio
indicates the proportion of samples in which the corre-
sponding tool is invoked.Recall,Precision, andF1are
computed at the page level.
MetricSlideVQA LongDoc. MMLong.
RP FP RP FP RP FP
Ratio 97.6 4.1 99.8 3.6 94.0 14.7
Recall 95.7 70.9 83.4 37.3 75.4 55.9
Precision 39.0 81.2 32.7 36.6 33.1 49.9
F1 54.1 72.9 44.4 31.9 42.1 49.6
ument understanding and analysis, rather than to
serve as authoritative or final interpretations.
Acknowledgments
This work is supported by the NSFC (62225603).
References
John Aloimonos, Isaac Weiss, and Amit Bandyopad-
hyay. 1988. Active vision.International journal of
computer vision, 1(4):333–356.
Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota,
Yusheng Xie, and R Manmatha. 2021. Docformer:
End-to-end transformer for document understanding.
InProceedings of the IEEE/CVF international con-
ference on computer vision, pages 993–1003.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl
technical report.arXiv preprint arXiv:2502.13923.
Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu,
Franck Dernoncourt, Jiuxiang Gu, Ryan A Rossi,
Changyou Chen, and Tong Sun. 2024. Sv-
rag: Lora-contextualizing adaptation of mllms for
long document understanding.arXiv preprint
arXiv:2411.01106.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3docrag: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-
Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song,
Bo Zheng, and 1 others. 2025. Longdocurl: a com-
prehensive multimodal long document benchmark
integrating understanding, reasoning, and locating.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 1135–1159.

Yihao Ding, Soyeon Caren Han, Jean Lee, and Eduard
Hovy. 2025. Deep learning based visually rich doc-
ument content understanding: A survey.Preprint,
arXiv:2408.01287.
Alexey Dosovitskiy. 2020. An image is worth 16x16
words: Transformers for image recognition at scale.
arXiv preprint arXiv:2010.11929.
Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang,
Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou,
Tong Lu, Hongsheng Li, and 1 others. 2025. Do-
copilot: Improving multimodal models for document-
level understanding. InProceedings of the Computer
Vision and Pattern Recognition Conference, pages
4026–4037.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2024. Colpali: Efficient document retrieval with vi-
sion language models.Preprint, arXiv:2407.01449.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. Colpali: Efficient document retrieval with vi-
sion language models.Preprint, arXiv:2407.01449.
Masato Fujitake. 2024. Layoutllm: Large language
model instruction tuning for visually rich document
understanding.arXiv preprint arXiv:2403.14252.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming
Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou.
2025. mplug-docowl2: High-resolution compressing
for ocr-free multi-page document understanding. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 5817–5834.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Geewook Kim, Teakgyu Hong, Moonbin Yim,
JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Won-
seok Hwang, Sangdoo Yun, Dongyoon Han, and Se-
unghyun Park. 2022. Ocr-free document understand-
ing transformer. InComputer Vision – ECCV 2022:
17th European Conference, Tel Aviv, Israel, Octo-
ber 23–27, 2022, Proceedings, Part XXVIII, page
498–517, Berlin, Heidelberg. Springer-Verlag.
Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexi-
ang Hu, Fangyu Liu, Julian Martin Eisenschlos, Ur-
vashi Khandelwal, Peter Shaw, Ming-Wei Chang,and Kristina Toutanova. 2023. Pix2struct: Screen-
shot parsing as pretraining for visual language under-
standing. InInternational Conference on Machine
Learning, pages 18893–18912. PMLR.
Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo
Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and
Xiang Bai. 2024. Monkey: Image resolution and
text label are important things for large multi-modal
models. Inproceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages
26763–26773.
Falk Lieder and Thomas L Griffiths. 2020. Resource-
rational analysis: Understanding human cognition as
the optimal use of limited computational resources.
Behavioral and brain sciences, 43:e1.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts.Transactions of the Asso-
ciation for Computational Linguistics, 12:157–173.
Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li,
Zhiyin Ma, Shuo Zhang, and Xiang Bai. 2024b.
Textmonkey: An ocr-free large multimodal model
for understanding document.arXiv preprint
arXiv:2403.04473.
Chuwei Luo, Yufan Shen, Zhaoqing Zhu, Qi Zheng, Zhi
Yu, and Cong Yao. 2024. Layoutllm: Layout instruc-
tion tuning with large language models for document
understanding. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition,
pages 15630–15640.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, and 1 others. 2024. Mmlongbench-doc:
Benchmarking long-context document understanding
with visualizations.Advances in Neural Information
Processing Systems, 37:95963–96010.
Minesh Mathew, Dimosthenis Karatzas, and CV Jawa-
har. 2021. Docvqa: A dataset for vqa on document
images. InProceedings of the IEEE/CVF winter con-
ference on applications of computer vision, pages
2200–2209.
Jamshed Memon, Maira Sami, Rizwan Ahmed Khan,
and Mueen Uddin. 2020. Handwritten optical charac-
ter recognition (ocr): A comprehensive systematic lit-
erature review (slr).IEEE Access, 8:142642–142668.
Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz, Elad
Ben Avraham, Alona Golts, Yair Kittenplon, Shai
Mazor, and Ron Litman. 2025. Docvlm: Make your
vlm an efficient reader. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 29005–29015.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.

Yongxin Shi, Jiapeng Wang, Zeyu Shan, Dezhi Peng,
Zening Lin, and Lianwen Jin. 2025. Urag: Unified
retrieval and generation in multimodal llms for effi-
cient long document understanding.arXiv preprint
arXiv:2511.10552.
Yulun Song, Long Yan, Lina Qin, Gongju Wang, Xingru
Huang, Luzhe Hu, and Weixin Liu. 2025. Urag: Uni-
fied retrieval-augmented generation. InProceedings
of the 2024 10th International Conference on Com-
munication and Information Processing, ICCIP ’24,
page 660–667, New York, NY , USA. Association for
Computing Machinery.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 24827–24837.
Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku
Hasegawa, Itsumi Saito, and Kuniko Saito. 2023.
Slidevqa: A dataset for document visual question
answering on multiple images. InProceedings of
the AAAI Conference on Artificial Intelligence, vol-
ume 37, pages 13636–13645.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, and 1
others. 2024. Gemini 1.5: Unlocking multimodal
understanding across millions of tokens of context.
arXiv preprint arXiv:2403.05530.
Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny.
2023. Hierarchical multimodal transformers for mul-
tipage docvqa.Pattern Recognition, 144:109834.
Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski,
Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anck-
aert, Ernest Valveny, and 1 others. 2023. Document
understanding dataset and evaluation (dude). InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision, pages 19528–19540.
Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan
Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao
Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and
Conghui He. 2024. Mineru: An open-source solution
for precise document content extraction.Preprint,
arXiv:2409.18839.
Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen,
Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang,
and Feng Zhao. 2025. Vrag-rl: Empower vision-
perception-based rag for visually rich information
understanding via iterative reasoning with reinforce-
ment learning.arXiv preprint arXiv:2505.22019.
Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and
Hong Cheng. 2025. Molorag: Bootstrapping doc-
ument understanding via multi-modal logic-aware
retrieval. InProceedings of the 2025 Conference onEmpirical Methods in Natural Language Processing,
pages 14035–14056.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.Preprint,
arXiv:2309.07597.
Xudong Xie, Hao Yan, Liang Yin, Yang Liu, Jing Ding,
Minghui Liao, Yuliang Liu, Wei Chen, and Xiang
Bai. 2024. Wukong: A large multimodal model for
efficient long pdf reading with end-to-end sparse sam-
pling.arXiv preprint arXiv:2410.05970.
Qixin Xu, Haozhe Wang, Che Liu, Fangzhen Lin, and
Wenhu Chen. 2025. Cogdoc: Towards unified think-
ing in documents.arXiv preprint arXiv:2512.12658.
Dayu Yang, Antoine Simoulin, Xin Qian, Xiaoyi Liu,
Yuwei Cao, Zhaopu Teng, and Grey Yang. 2025.
Docagent: A multi-agent system for automated
code documentation generation.arXiv preprint
arXiv:2504.08725.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models. InThe eleventh international conference on
learning representations.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and 1 others. 2024. Vis-
rag: Vision-based retrieval-augmented generation
on multi-modality documents.arXiv preprint
arXiv:2410.10594.
Shengbin Yue, Siyuan Wang, Wei Chen, Xuanjing
Huang, and Zhongyu Wei. 2025. Synergistic
multi-agent framework with trajectory learning for
knowledge-intensive tasks. InProceedings of the
AAAI Conference on Artificial Intelligence, vol-
ume 39, pages 25796–25804.
Jinxu Zhang, Yongqi Yu, and Yu Zhang. 2024. Cream:
coarse-to-fine retrieval and multi-modal efficient tun-
ing for document vqa. InProceedings of the 32nd
ACM International Conference on Multimedia, pages
925–934.
Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu,
Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan,
Weijie Su, Jie Shao, and 1 others. 2025. Internvl3:
Exploring advanced training and test-time recipes
for open-source multimodal models.arXiv preprint
arXiv:2504.10479.
A Detail of Environment Design
This subsection provides the detailed construction
of theGlobal Thumbnail Overview ˜Dreferenced
in ourEnvironment Initialization. Given a docu-
ment with Npages D={I 1, . . . , I N}, we build
˜Das a small set of tiled overview images that to-
gether cover all pages while maintaining a very low

initial visual budget compared to all image with
high-resolution. We set G= 36 to be the maxi-
mum number of pages allowed per overview image.
We first partition the page indices into consecutive
groups in sequential order
Gk={(k−1)G+ 1, . . . ,min(kG, N)},
where k= 1, . . . , K , and the number of overview
imagesKis
K=N
G
, n k=|G k| ≤G.
Each page Iiis resized to a fixed thumbnail Ti∈
R256×256(aspect-ratio handling follows standard
padding/letterboxing so that all thumbnails share
identical canvas size). For each group Gk, we
pack its nkthumbnails into a single composite im-
age˜I(k)using an adaptive near-square grid. Con-
cretely, we choose grid dimensions (Rk, Ck)such
thatRkCk≥nkand the grid is as close to square
as possible; in practice, we set
Rk=⌈√nk⌉, C k=nk
Rk
,
which guarantees RkCk≥n kand yields a com-
pact layout. If RkCk> nk, the remaining cells are
left empty (blank padding) to preserve a regular
grid geometry.
To ensure unambiguous visual indexing, each
grid cell includes a thin blank header band of height
hpixels above the thumbnail region; we render
the absolute page index i(for the corresponding
thumbnail Ti) inside this header band. Thus, a cell
is a(h+ 256)×256 block consisting of a header
strip for the index and a 256×256 thumbnail area
below it. The resulting overview image ˜I(k)is
obtained by tiling these blocks into an Rk×C k
array, with empty cells rendered as blank blocks.
This construction yields the global overview set
˜D={ ˜I(1), . . . , ˜I(K)}, K=N
G
,
which is then used in the initial observation O1=
{Q,˜D}as described in the main paper.
For intuition, consider several document lengths.
When N= 40 , we obtain K=⌈40/36⌉= 2
overview images: the first group has n1= 36
pages and forms a 6×6 grid, while the second
group has n2= 4 pages and forms a 2×2 grid.
When N= 50 , we again have K= 2 : the firstoverview remains 6×6 (36 pages), and the second
overview contains n2= 14 pages, which under the
near-square rule becomes a 4×4 grid with two
empty cells. In the appendix H, we visualizes these
overviews images, it illustrates that these low-cost
overviews provide strong initial navigational sig-
nals, especially for counting-style user questions.
In summary, a critical advantage of the proposed
Global Thumbnail Overviewis the substantial re-
duction in visual token consumption compared to
full-resolution ingestion. Empirical analysis using
the Qwen-2.5-VL (Bai et al., 2025) vision encoder
demonstrates that our method achieves a compres-
sion ratio of approximately 10× to12×. For in-
stance, a 100-page document processed at a stan-
dard high resolution of 1024×768 typically gener-
ates over 100,000 visual tokens. In contrast, repre-
senting the same document via our tiled overview
construction (resulting in K= 3 composite im-
ages) yields only ≈8,000 visual tokens. While
further downscaling of individual page thumbnails
Tiis theoretically possible, our chosen resolution
strikes a balance betweenlegibility and efficiency.
Consequently, this approach functions as a strategic
compromise between full-document input—which
preserves global context but incurs prohibitive com-
putational costs—and Visual Retrieval-Augmented
Generation (RAG), which optimizes for cost but
often fragments global coherence. By retaining
a macro-level visual representation, we preserve
structural and semantic continuity while leveraging
external tools for fine-grained details.
B Agent–Environment Interaction
Protocol
This section provides a complete, implementation
oriented description of how theDoc- V∗agent in-
teracts with a multi-page document environment.
Our goal is to make the interaction loop explicit
and reproducible: what the agentreceivesat each
turn, what itmust output, how the environment
responds, and how state (e.g., visited pages and
working memory) is maintained. Please refer to
Algorithm 1 for the complete pseudocode.
Given a document D={p 1, . . . , p N}(each piis
a page image) and a question Q, we cast multi-page
Document VQA as a sequential decision process
with a maximum budget of Tinteraction turns. At
each turn t: (i) the agent receives an observation
Ot, (ii) it performs reasoning and emits exactly one
atomic action at∈ A, (iii) the environment exe-

cutes the action and returns feedback Et+1, (iv) the
feedback is incorporated into the next observation.
Crucially, the agent isnotgiven the full docu-
ment at high resolution upfront. Instead, the envi-
ronment pre-computes and caches high-resolution
visual tokens for each page and only reveals
the requested pages on demand, enabling se-
lective evidence acquisition under limited con-
text/computation budgets.
Algorithm 1 Doc- V∗Agent–Environment Interac-
tion (Inference-Time Loop)
Require: Document pages D={p 1, . . . , p N}, question Q,
turn limit T, retrieval top- k, Global Thumbnail Overview
˜D, high-res visual tokensv i← M(V(p i))
Initialization:
1:P visited← ∅▷tracks pages already revealed to the agent
2:W← ∅▷working memory: concatenated per-turn
summaries
3:O← {Q, ˜D}▷initial observationO 0
4:fort←0toT−1do
5:u t←π θ(O)▷must follow
<think>...</think><action>...</action>
6: Parse utto obtain (i) one atomic action atand (ii)
summaryS t
7:W←W⊕S t ▷append summary to working
memory
8:ifa tis<answer>with stringythen
9:returny ▷terminate interaction
10:else ifa tis<retrieval_page>with queryq tthen
11:I ←RETRIEVER(q t,D \ P visited, k)▷rank
unvisitedpages using an external multimodal retriever
12: else if atis<fetch_page> with indices [i1, . . . , i m]
then
13:I ←[i 1, . . . , i m]▷direct request by absolute
page indices
14:else
15:I ← ∅▷invalid action; environment may return
a format reminder
16:end if
Environment feedback construction:
17:E← ∅
18:for alli∈ Ido
19:ifi∈ P visited then
20:E←E∪ {“Pageialready visited.”}
▷avoid redundant visual tokens
21:else
22:E←E∪ {“Pagei:”,v i}▷prefix page id
+ cached high-res tokens
23:P visited← P visited∪ {i}
24:end if
25:end for
26:O←E∪ {W}▷augmented observation for next
turn:O t+1
27:end for
28:returnNoAnswer▷ optional fallback when turn budget is
exhausted
Cached high-resolution page tokensFor each
pagepi, the environment caches its high-resolution
visual tokens vi=M(V(p i))∈RLi×d, computed
at the page’s native resolution (capped at 1024×
768).Initial Observation ( t=0).Before any interac-
tion, the environment constructs aGlobal Thumb-
nail Overview ˜Dby resizing pages to thumbnails
(e.g., 256×256 ), arranging them into one or more
grid images, and annotating each thumbnail with its
absolute page number. While fine text is typically
unreadable at this scale, it preserves strong struc-
tural cues (document type, section layout, chart
distribution, large-font titles). The initial observa-
tion is
O0={Q, ˜D}.
Visited Page SetThe environment maintains
an external set Pvisited to prevent redundant page
inputs. If the agent requests an already visited page,
the environment returns a shorttext reminderrather
than re-sending visual tokens.
Working MemoryTo reduce forgetting and
repetitive behaviors during multi-turn interaction,
we maintain aWorking Memory Wtformed by con-
catenating the agent’s per-turn summaries:
Wt= Concat(S 0, . . . , S t−1),
where Stis the content of the agent’s <summary>
block at turnt.
Augmented Observation ( t≥1).At turn t≥1 ,
the agent receives an augmented observation:
Ot=E t∪ {W t},
where Etis the environment feedback produced by
executing the previous action.
At each turn, the agent must outputexactly one
atomic action from the following set:
•Retrieval action: <retrieval_page>q t.
This action mimics a “Ctrl+F”-like search but
over page images. The query qtmay differ
from the original question Qand can be itera-
tively refined.
•Fetch action: <fetch_page>[i 1, . . . , i m].
This action requests pages by absolute indices
(e.g., based on thumbnail cues, adjacency ex-
ploration, or explicit page references in the
question).
•Answer action: <answer>y . This action
terminates the interaction and outputs the final
answer stringy.
To make decision-making auditable, we enforce
a fixed ReAct-style output schema:
<think>...</think><action>...</action>.

Att=0, the<think> section should include (i)
<analysis> based on thumbnails, (ii) <plan> for
a turn-budgeted strategy, and (iii) <summary> to
be appended to working memory. At t>0, the
<think> section should include (i) <analysis> of
newly returned pages, (ii) <relevant_pages> list-
ing the page numbers judged relevant among the
newly returned pages, and (iii)<summary>.
Environment Response SemanticsFor retrieval
or fetch actions, the environment returns the cached
high-resolution visual tokens of the requested
pages. Each page’s tokens are preceded by a tex-
tual page identifier (e.g., “Page 5:” ) to maintain
an unambiguous mapping between content and ab-
solute page index, especially when pages arrive
out of order. For already-visited pages, the envi-
ronment returns a short reminder string instead of
re-injecting tokens.
C Detail of Training
ExistingMulti-page Document Visual Question An-
swering (VQA)benchmarks usually annotate only
the final supervision tuple (D, Q, y, P gt), i.e., the
document, question, final answer, and (optionally)
evidence pages, but they do not provide the multi-
step interaction traces required by our agent. To
train the behavior model described in the main
text, we adopt atwo-stage recipe: first supervised
fine-tuning (SFT) on distilled closed-loop interac-
tion trajectories, and then GRPO-based (Guo et al.,
2025) reinforcement learning to further optimize
answer correctness and evidence discovery under
a bounded interaction budget. In both stages, all
environment feedback (returned page images and
working memorys) is used only as conditioning
context; training losses are applied only to tokens
generated by the agent itself.
SFT: Closed-loop Interaction Trajectory Distil-
lationWe distill interaction trajectories from a
strong teacher model (GPT-4o (Hurst et al., 2024))
by running it in a closed-loop environment that
executes real actions and returns real page im-
ages. Each teacher turn must follow our proto-
col: one <think> block plus exactly one <action>
among <retrieval_page> ,<fetch_page> , and
<answer> . The environment executes the ac-
tion and returns the corresponding visual obser-
vation (thumbnail overview at the beginning; high-
resolution pages thereafter) and working memory
as feedback for the next turn. This closed-loop dis-tillation is essential because retrieval and fetching
change subsequent observations, so the distilled
traces reflect realistic exploration dynamics rather
than offline labels.
SFT: Trajectory FilteringWe keep only reliable
trajectories for imitation: 1)Format validity:the
full trace must be parseable; every turn contains ex-
actly one valid action with valid arguments and re-
quired fields in <think> ; 2)Answer correctness:
we compare the teacher final answer ˆywith the
ground-truth y. For free-form textual answers, we
compute ANLS and require ANLS(ˆy, y)≥τ anls
(we use τanls= 0.7 ). For identifier-like answers
(dates, counts, phone numbers, emails), we re-
quire exact match I[ˆy=y] = 1 . When ANLS
is low (may be due to benign formatting differ-
ences), we additionally use a judge model (GPT-
4o) to verify semantic equivalence; 3)Evidence
sanity:the teacher outputs <relevant_pages> in-
side <think> . Let Prelbe the union of all pages
listed in <relevant_pages> across turns. We re-
quire Prel∩Pgt̸=∅; if not, we keep the trajectory
only if another judge model (GPT-4o) verifies that
the selected pages support the answer (to mitigate
incomplete evidence annotations).
We build long-document training samples by
selecting examples with more than 10 pages
from MP-DocVQA (Tito et al., 2023) and
DUDE (Van Landeghem et al., 2023) dataset. We
keep DUDE not-answerable cases to improve
abstention when evidence is insufficient. After dis-
tillation and filtering, our SFT set contains 9,019
trajectories in total (5,969 from MP-DocVQA
and 3,050 from DUDE). Each distilled trajec-
tory is serialized into a single sequence that in-
terleaves environment observations and agent out-
puts across multiple turns. Observations include
the current visual feedback (thumbnail overview
or returned page images) and the accumulated
working-memory summaries from previous turns.
Agent outputs include structured <think> con-
tent (analysis/plan/summaryin the first turn;anal-
ysis/relevant_pages/summaryin later turns) fol-
lowed by exactly one action tag. During training,
the model is conditioned on the entire serialized
prefix, but only the agent-generated tokens con-
tribute to the loss.
SFT: ObjectiveLet a serialized trajectory be the
token sequence x1:L. We define a mask mℓ∈
{0,1} indicating whether token xℓbelongs to the
agent-generated part ( <think> and<action> ) or

to the environment observation. The SFT objective
is the masked negative log-likelihood:
LSFT(θ) =−L−1X
ℓ=1mℓ+1logπ θ(xℓ+1|x1:ℓ).(1)
GRPO: Training DataWhile SFT enables ef-
fective imitation, it inherits teacher biases and does
not explicitly optimize exploration efficiency under
the interaction budget. We therefore further train
the agent with GRPO (Guo et al., 2025), which
optimizes expected trajectory-level reward using
group-wise sampled rollouts. GRPO training uses
only raw dataset-level supervision (D, Q, y, P gt)
without intermediate traces. We select 2,048 train-
ing examples from MP-DocVQA and DUDE that
do not overlap with the SFT training set. To ensure
a balanced difficulty distribution, we estimate per-
example difficulty using the SFT model: for each
(D, Q) we run 4 independent rollouts and count the
number of successes (ANLS ≥0.7 ). We then strat-
ify samples into easy/medium/hard buckets and ran-
domly draw them with proportions 10%/70%/20%,
respectively.
For each training sample (D, Q) , we run the
current policy πθin the same closed-loop environ-
ment to sample a group of Gcomplete trajectories
{T1, . . . , T G}(stochastic decoding). Each trajec-
tory terminates when the agent outputs <answer>
or reaches the interaction budget. Each sampled tra-
jectory can be represented as a pair (ci, ai), where
cidenotes all conditioning context tokens (all ob-
servations, including page images and working
memory) and aidenotes the concatenated agent-
generated tokens (all <think> and<action> to-
kens) in that trajectory.
GRPO: RewardFor a trajectory T, we compute
R(T) =w aRa(T) +w eRe(T) +w fRf(T).
Rameasures answer correctness. For free-form
textual answers, we use thresholdedAverage Nor-
malized Levenshtein Similarity (ANLS):
Ra(T) =I[ANLS(ˆy, y)≥τ] ANLS(ˆy, y),
where τ= 0.5 and for identifier-like answers we
useExact Match (EM):
Ra(T) =I[ˆy=y]
For evidence, letP rel(T)be the union of all pages
listed in <relevant_pages> across turns. We com-
pute a recall-weighted F-score:
Re(T) =(1 +β2)pr
β2p+r, β2= 2,where
p=|Prel∩Pgt|
|Prel|+ϵ, r=|Prel∩Pgt|
|Pgt|+ϵ,
where ϵis a small constant. Finally Rfpenalizes
binary invalid outputs (unparseable format, invalid
action arguments, or budget violation), with a re-
ward of 1 for valid output and 0 for invalid output.
GRPO: ObjectiveGRPO optimizes relative per-
formance within a sampled group. For each group
{Ti}G
i=1, letRi=R(T i). We compute the group-
normalized advantage:
Ai=Ri−µ
σ+ϵ
µ=1
GGX
j=1Rj, σ=vuut1
GGX
j=1(Rj−µ)2
We then update the policy by maximizing the log-
likelihood of sampled actions weighted by Ai, us-
ing a PPO-style clipped objective at the token level.
Letπθolddenote the policy used to sample the
group. For each trajectory iand each agent token
positiont, define the ratio
ρi,t(θ) =πθ(ai,t|ci, ai,<t)
πθold(ai,t|ci, ai,<t).(2)
The GRPO loss is defined as:
LGRPO (θ) =−1
GGX
i=1|ai|X
t=1min
ρi,t(θ)A i,
clip 
ρi,t(θ),1−ϵ c,1 +ϵ c
Ai
,(3)
where ϵcis the clip range. Importantly, the loss
is applied only on agent-generated tokens ai; all
environment observation tokens are used only as
conditioning context.
Inference: Coarse-to-Fine Evidence Acquisition
At test time, we use greedy decoding (tempera-
ture= 0) and enforce a maximum of Tinteraction
steps. Starting from the global thumbnail overview,
the agent follows a coarse-to-fine strategy: it uses
structural cues in ˜Dto propose candidate pages,
employsretrieval_page with refined queries to
localize evidence, uses fetch_page for targeted
reading and cross-page completion when needed,
updates its working memory via summaries, and
terminates with answer once evidence suffices.
The essential idea is not to increase context indis-
criminately, but to keep the input in a high signal-
to-noise regime by actively selecting what to read.

D Training and Inference Configuration
In this section, we provide the comprehensive hy-
perparameter settings and configuration details for
the training and inference ofDoc- V∗. All exper-
iments were conducted on a computational node
equipped with 8 NVIDIA A100 (80GB) GPUs, im-
plemented in PyTorch using BF16 mixed precision
to optimize memory efficiency.
Stage I: Supervised Fine-Tuning (SFT)The pri-
mary goal of the SFT stage is to initialize the agent
with stable tool usage capabilities and reasoning
behaviors.
•Data:We utilize a filtered dataset comprising
9,019 high-quality interaction trajectories.
•Optimization:The model is trained for 3
epochs using the AdamW optimizer with a co-
sine learning rate scheduler. The initial learn-
ing rate is set to3×10−6.
•Loss Masking:To focus the model’s adap-
tation on reasoning and planning, the loss is
computed exclusively on agent-generated to-
kens (specifically the contents within <think>
and<action> blocks), masking out the user
instructions and environment observations.
Stage II: Group Relative Policy Optimization
(GRPO)Following SFT, the agent undergoes re-
inforcement learning alignment to further refine its
decision-making logic.
•Hyperparameters:We employ a group size
ofG= 8 with a sampling temperature of 1.0
to encourage exploration during the genera-
tion phase. The training proceeds for 3 epochs
with a reduced learning rate of2×10−6.
•Reward Configuration:As outlined in the
main text, the composite reward function
is defined as R=ω ansRans+ω eviRevi+
ωstructRstruct . The specific coefficients are
set to ωans= 0.6 (Correctness), ωevi= 0.3
(Evidence Recall), and ωstruct = 0.1 (Format
Validity).
Inference ConfigurationDuring the evaluation
phase, to ensure deterministic and reproducible
results, we employ greedy decoding (temperature
= 0). The maximum interaction horizon is fixed at
T= 8 steps, consistent with the constraints applied
during the training phase.E Details of Datasets
MP-DocVQA (Tito et al., 2023)a multi-page
document visual question answering benchmark
that focuses on fine-grained information extraction
from scanned documents. Questions often require
precise localization of textual or visual elements
within a document and explicit reasoning over page
indices. The dataset emphasizes accurate page nav-
igation and localized evidence grounding.
DUDE (Van Landeghem et al., 2023)con-
sists of document images paired with questions
that demand detailed visual-textual understanding.
Compared to MP-DocVQA, DUDE places stronger
emphasis on structured layouts such as forms and
tables, and requires robust cross-page navigation to
retrieve relevant evidence scattered across multiple
pages.
SlideVQA (Tanaka et al., 2023)a document
visual question answering dataset focused on un-
derstanding presentation slides. It contains slide
documents with diverse visual layouts, including
figures, charts, bullet lists, and sparsely distributed
text. Documents typically span around 20 pages,
and the associated questions require complex rea-
soning over non-linear reading orders and spatial
arrangements, rather than relying solely on sequen-
tial textual flow.
LongDocURL (Deng et al., 2025)composed
of web-based multi-modal documents with rich
structural diversity, such as headings, hyperlinks,
images, and embedded tables. With an average
document length of approximately 30 pages, the
dataset evaluates long-range retrieval and the ability
to locate and synthesize information across distant
document sections.
MMLongBench-Doc (Ma et al., 2024)de-
signed for long-context multi-modal document un-
derstanding. Documents in this benchmark are sub-
stantially longer, extending up to 468 pages. The
dataset poses significant challenges for scalable
page selection, efficient navigation, and multi-hop
reasoning over large multi-modal contexts.
F Details of Baseline
This section provides detailed specifications for the
open-source baselines compared in our study. Ta-
ble 6 summarizes their key configurations and train-
ing settings, followed by comprehensive descrip-
tions of each method’s architecture and paradigm.

Table 6:Detailed configurations of Open Source baselines.“Retriever” denotes the model used for page retrieval.
“Param” refers to the parameter size of the LLM backbone. “Paradigm” categorizes methods into End-to-End (E2E),
Retrieval-Augmented Generation (RAG), orAgent. The columns under “Trained on Dataset?” indicate whether the
backbonewas supervised fine-tuned ( ✓) on the corresponding benchmark’s training set or evaluated in a zero-shot
setting (×).
Method Retriever Backbone Param OCR-Free ParadigmTrained on Dataset?
DUDE MPDocVQA SlideVQA
HiVT5(PR)- DiT / T5 0.3B×E2E×✓×
CREAM(ACM MM’24)bge-large Pix2Struct / LLaMa2 7B×RAG✓ ✓×
mPLUG-DocOwl2(ACL’25)- ViT / LLaMa 8B✓E2E✓ ✓×
M3DocRAG(arXiv’24)Colpali Qwen2-VL 7B✓RAG× × ×
VisRAG(ICLR’25)VisRAG-Ret MiniCPM-V 2.6 8B✓RAG× × ×
SV-RAG(ICLR’25)SV-RAG-InternVL2 InternVL2 4B✓RAG× ×✓
VDocRAG(CVPR’25)VDocRetriever Phi3-Vision 4B✓RAG✓× ×
Docopilot(CVPR’25)- InternVL2 8B×E2E✓ ✓×
DocVLM(CVPR’25)- Qwen2-VL 7B×E2E× × ×
InternVL3(arXiv’25)- InternViT / Qwen2.5 8B✓E2E× × ×
VRAG-RL(NeurIPS’25)ColQwen2 Qwen2.5-VL 7B✓Agent× ×✓
MoLoRAG(EMNLP’25)Colpali+Qwen2.5-VL Qwen2.5-VL 7B✓RAG× × ×
CogDoc(arXiv’25)- Qwen2.5-VL 7B✓Agent✓×✓
URaG(AAAI’26)Qwen2.5-VL (Early Layers) Qwen2.5-VL 7B✓RAG✓ ✓ ✓
Ours Colqwen2.5 Qwen2.5-VL 7B ✓ Agent ✓ ✓ ×
/uni00000027/uni00000038/uni00000027/uni00000028/uni00000030/uni00000033/uni00000027/uni00000052/uni00000046/uni00000039/uni00000034/uni00000024 /uni00000036/uni0000004f/uni0000004c/uni00000047/uni00000048/uni00000039/uni00000034/uni00000024/uni0000002f/uni00000052/uni00000051/uni0000004a/uni00000027/uni00000052/uni00000046/uni00000038/uni00000035/uni0000002f/uni00000030/uni00000030/uni0000002f/uni00000052/uni00000051/uni0000004a/uni00000025/uni00000048/uni00000051/uni00000046/uni0000004b/uni00000010/uni00000027/uni00000052/uni00000046/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013/uni00000017/uni00000013/uni00000024/uni00000059/uni0000004a/uni00000011/uni00000003/uni00000033/uni00000044/uni0000004a/uni00000048/uni00000056/uni00000003/uni00000053/uni00000048/uni00000055/uni00000003/uni00000036/uni00000044/uni00000050/uni00000053/uni0000004f/uni00000048 /uni00000018/uni00000011/uni00000019 /uni00000018/uni00000011/uni00000019/uni00000015/uni00000013/uni00000011/uni00000013/uni00000016/uni00000013/uni00000011/uni00000013/uni00000017/uni00000016/uni00000011/uni00000016
Figure 6:Average document length across datasets.
The figure reports the average number of pages per
document for MP-DocVQA, DUDE, SlideVQA, Long-
DocURL, and MMLongBench-Doc, illustrating the in-
creasing document length and context complexity from
standard document QA benchmarks to long-context
multi-modal settings.
HiVT5HiVT5 (Tito et al., 2023) proposes a hi-
erarchical multimodal transformer to extend Docu-
ment VQA to multi-page scenarios, addressing the
quadratic complexity of standard attention mecha-
nisms. Relying on an off-the-shelf OCR engine for
text and bounding box extraction, it employs a T5-
based encoder to process each page independently.
The model fuses OCR tokens, layout embeddings,
and visual features into learned [PAGE] tokens,
which summarize page content conditioned on the
query. These summaries are concatenated for the
decoder to generate the answer, supported by a
module predicting evidence page indices. Training
involves a hierarchical layout-aware pre-training
task followed by fine-tuning on MP-DocVQA.CREAMCREAM (Zhang et al., 2024) presents a
framework integrating coarse-to-fine retrieval with
multimodal efficient tuning to handle token limita-
tions in multi-page documents. It first utilizes an
OCR engine to extract and chunk text, followed
by a two-stage retrieval process: a coarse ranking
via text embedding similarity and a fine-grained re-
ranking where an LLM recursively groups chunks
to select the top-k candidates. To incorporate vi-
sual context, a multi-page vision encoder employs
attention pooling to merge features into a unified
representation. Based on LLaMA-Adapter V2, the
model undergoes multimodal instruction tuning (us-
ing LoRA and prefix tuning) to jointly optimize the
LLM with retrieved chunks and visual embeddings.
mPLUG-DocOwl2mPLUG-DocOwl2 (Hu
et al., 2025) introduces a modularized Multimodal
Large Language Model (MLLM) specialized for
OCR-free document understanding. Improving
upon the mPLUG-Owl architecture, it employs a
visual abstractor to bridge the pre-trained visual
encoder and the LLM, directly aligning visual
features with textual semantics to eliminate
external OCR dependency. The model is optimized
via a unified instruction tuning strategy on a
diverse document instruction dataset (covering
tables, charts, and webpages), enhancing its
capability to comprehend fine-grained visual text
and complex structures.
M3DocRAGM3DocRAG (Cho et al., 2024) pro-
poses a multimodal Retrieval-Augmented Gener-
ation (RAG) framework to overcome the limita-

tions of text-based pipelines in visually rich, open-
domain tasks. Diverging from OCR-dependent
methods, it adopts an all-multimodal paradigm us-
ing a vision-language retriever (e.g., ColPali) to
encode page images into visual embeddings. This
enables precise retrieval via late interaction mecha-
nisms that preserve layout semantics. The retrieved
top-k raw page images are then fed into an MLLM
(e.g., Qwen2-VL) for end-to-end question answer-
ing. The authors also introduce M3DocVQA, a
benchmark requiring cross-document retrieval and
multi-hop reasoning.
VisRAGVisRAG (Yu et al., 2024) presents a
vision-based RAG framework that treats document
pages purely as images, mitigating information loss
from OCR extraction. It employs a dual-encoder
architecture (VisRAG-Ret) where queries and doc-
ument images are encoded into a shared embedding
space using position-weighted mean pooling. Gen-
eration (VisRAG-Gen) is handled by a generative
VLM that synthesizes answers directly from the
retrieved visual context. The retriever is fine-tuned
via contrastive learning on a mixture of public VQA
datasets and synthetic query-document pairs to en-
sure robust generalization.
SV-RAGSV-RAG (Chen et al., 2024) leverages
a single MLLM backbone equipped with two dis-
tinct Low-Rank Adaptation (LoRA) adapters to
handle both retrieval and generation without exter-
nal parsers. It employs a retrieval adapter using
contextualized late interaction to identify evidence
pages, and a QA adapter for answer generation.
The adapters are optimized via contrastive learn-
ing for retrieval and autoregressive generation for
QA, enabling efficient, unified visual retrieval and
reasoning within a single model architecture.
VDocRAGVDocRAG (Tanaka et al., 2025) in-
troduces a visual RAG framework designed to pro-
cess visually rich documents by leveraging visual
features directly. It employs a dual-component ar-
chitecture: VDocRetriever, which retrieves relevant
page images using dense token representations, and
VDocGenerator, which synthesizes answers from
these inputs. To align visual and textual informa-
tion, the authors utilize self-supervised pre-training
tasks that adapt Large Vision-Language Models
(LVLMs) for retrieval by compressing visual rep-
resentations into dense tokens, facilitating open-
domain document reasoning.DocopilotDocopilot (Duan et al., 2025) pro-
poses a native multimodal framework that eschews
external retrieval in favor of scaling the model’s in-
trinsic context processing. Centered on a "retrieval-
free" paradigm, the model ingests entire documents
as concatenated high-resolution image sequences.
It leverages engineering optimizations like Ring At-
tention and Liger Kernel to manage long contexts
(up to 32k tokens). The capability is supported
by "Doc-750K," a large-scale dataset with diverse
proxy tasks. Training involves Supervised Fine-
Tuning (SFT) with multimodal data-packing, al-
lowing the model to process full document contexts
in a single forward pass to resolve long-distance
dependencies.
DocVLMDocVLM (Nacson et al., 2025)
presents a model-agnostic framework to enhance
VLMs by efficiently integrating OCR-derived text
and layout information. It utilizes an OCR encoder
to capture textual and spatial details, compressing
them into a compact set of learned queries (typi-
cally 64) which are projected into the LLM along-
side visual features. This approach preserves the
original VLM weights. Training follows a two-
stage process: aligning the OCR encoder with
the frozen VLM via captioning, followed by fine-
tuning on DocVQA datasets, achieving high per-
formance with reduced visual token usage.
InternVL3InternVL3 (Zhu et al., 2025) is a
state-of-the-art multimodal large language model
(MLLM) developed by OpenGVLab that advances
the field through a native multimodal pre-training
paradigm, jointly acquiring visual and linguistic ca-
pabilities rather than adapting a text-only backbone.
By incorporating variable visual position encoding
(V2PE) for extended contexts and advanced post-
training techniques like mixed preference optimiza-
tion, the model achieves superior performance on
diverse benchmarks, including MMMU and OCR-
related tasks. In this study, InternVL3 is utilized as
a strong baseline due to its robust optical character
recognition (OCR) and document understanding
capabilities, serving as a high-standard reference
for evaluating the efficacy of the proposed method
in visually rich environments.
VRAG-RLVRAG-RL (Wang et al., 2025) intro-
duces an agentic framework empowering VLMs
with iterative reasoning. It defines a unified ac-
tion space integrating search queries with fine-
grained visual perception actions, specifically pre-

dicting coordinates for cropping and zooming
into information-dense regions to handle resolu-
tion bottlenecks. Operating in a "Thought-Action-
Observation" loop, the model generates reasoning
chains, executes actions to update observations, and
iterates until evidence is gathered. The policy is
optimized via Group Relative Policy Optimization
(GRPO) with a reward function incentivizing both
retrieval precision and answer accuracy.
MoLoRAGMoLoRAG (Wu et al., 2025) pro-
poses a logic-aware retrieval framework capturing
both semantic and logical dependencies. It con-
structs a document-level "page graph" where edges
represent semantic similarities. A lightweight
VLM acts as a retrieval engine, traversing this
graph by evaluating "logical relevance"—the in-
ferential necessity of a page—alongside semantic
alignment. This allows the model to uncover logi-
cally connected but semantically distant evidence.
The framework supports both a training-free mode
and a fine-tuned mode where the engine is opti-
mized on synthesized "question-image-relevance"
triplets.
CogDocCogDoc (Xu et al., 2025) proposes a
unified, two-stage cognitive framework mimicking
human reading patterns to balance scalability and
fidelity. It decomposes reasoning into two phases
executed by a single VLM: a "Fast Reading" phase
(Localization Mode), scanning the document at
low resolution to predict page indices based on
structural cues; and a "Focused Thinking" phase
(Reasoning Mode), processing localized pages at
high resolution for grounded reasoning. To avoid
policy conflicts in supervised training, it employs
Direct Reinforcement Learning (RL from scratch),
enabling the model to autonomously learn to alter-
nate between global scanning and local reasoning.
URaGURaG (Shi et al., 2025) introduces a uni-
fied framework integrating retrieval and generation
within a single MLLM to handle long documents
efficiently. Based on the observation that MLLMs
exhibit a "coarse-to-fine" attention pattern, the
method inserts a lightweight cross-modal retrieval
module into the model’s early layers (e.g., layer 6).
This module acts as an internal evidence selector,
computing relevance via late interaction and retain-
ing only the top-k pages while discarding irrelevant
tokens from subsequent layers. This "early-exit"
mechanism reduces computational overhead for
deeper reasoning layers. Training involves pre-Table 7:Impact of K on MMLongBench-Doc.“Adap-
tive” denotes the document-adaptive setting K=
min(⌈N/10⌉,4) , where Nis the total number of pages.
K Avg. Pages OverallBreakdown
SIN MUL UNA
Adaptive 5.6 42.1 54.6 23.5 45.7
1 3.0 40.5 51.6 17.0 56.1
2 4.3 39.7 54.4 20.7 39.9
3 5.4 40.1 53.3 23.0 40.8
4 6.5 41.1 53.3 24.0 43.5
5 8.1 41.7 52.9 23.5 48.4
Table 8:Impact of maximum interaction steps on
MMLongBench-Doc.
Iteration Avg. Pages OverallBreakdown
SIN MUL UNA
3 4.2 41.1 54.0 22.5 44.4
4 4.6 41.2 54.3 23.0 43.5
5 4.9 41.4 54.3 23.4 43.5
6 5.2 41.5 54.4 24.0 44.0
7 5.6 42.1 54.6 23.5 45.7
8 5.8 41.4 54.3 24.0 44.0
9 6.0 41.5 54.8 24.0 44.0
10 6.2 41.5 54.8 24.0 44.0
training the retrieval module followed by joint fine-
tuning of both components.
G Robustness of Iteration & K
We investigate how the number of interaction turns
(iterations) affects the agent’s performance on the
MMLongBench-Docdataset. As the agent op-
erates in a recursive “Observe-Think-Act” loop,
the number of steps determines the depth of ex-
ploration. As shown in Table 7, performance im-
proves consistently as the maximum iteration limit
increases from 3 to 7. The model achieves peak per-
formance at7 iterationswith an overall accuracy
of42.1%. This suggests that for complex long-
document tasks, the agent requires approximately
5–7 steps to effectively locate evidence and synthe-
size answers. Beyond 7 iterations, the performance
plateaus and slightly fluctuates, indicating that the
agent has converged and further exploration yields
diminishing returns.
We further analyze the effect of the page se-
lection budget KonMMLongBench-Doc, as re-
ported in Table 8. Overall performance exhibits
a clear non-monotonic trend with respect to K.
When Kis small (e.g., K= 1 or2), the agent is
restricted to a limited number of pages, leading to

insufficient evidence coverage and degraded overall
accuracy. As Kincreases, performance improves
steadily and reaches its peak under theAdaptive
setting, where K= min(⌈N/10⌉,4) . This adap-
tive strategy achieves the best overall accuracy of
42.1%while maintaining a moderate average page
count of 5.6.
Further increasing Kbeyond the adaptive range
does not result in consistent gains. Although larger
Kvalues introduce more pages, the additional con-
text also brings redundant or irrelevant information,
which weakens evidence aggregation and slightly
hurts performance. This effect is particularly ev-
ident in the SIN and MUL subsets, where accu-
racy saturates or fluctuates as Kgrows. These
results indicate that effective long-document rea-
soning depends on selecting a well-calibrated num-
ber of pages rather than aggressively expanding the
context. The adaptive strategy strikes a favorable
balance between evidence sufficiency and noise
control, highlighting the importance of dynamic,
document-aware page budgeting.
H Case Study
See Fiugre 7, 8, 9

Question: What is the difference in percentage from Year 0 to Year 3 
regarding cumulative abnormal returns?
Input All Pages
··· ···<answer> 
4% increase 
</answer>
RAG -Top 5
Page 14 Page 12 Page 13 Page 11 Page 9
<answer> -4.3</answer>
Doc-V* (SFT & GRPO)
<think> \\boxed{analysis} { Page Index 14: Relevant . It shows 
the chart titled “Cumulative Abnormal Returns” with bars 
labeled Year -3 through Year +3. The value at Year 0 is 1.60% 
and at Year +3 is -6.80%. The difference between Year 0 and 
Year +3 is 1.60% - (-6.80%) = 8.40%. } 
\\boxed{relevant_pages} { 14 }
\\boxed{summary}{ The difference is 8.40%. }
</think> <answer> difference is 8.40%  </answer>
<think> \\boxed{thumbnail} {The thumbnail shows a slide 
deck with 20 slides. Several slides contain charts: Page Index 
12 looks like a bar chart titled “Net Profit Margin,”…… Page 
Index 14 a bar chart titled “Cumulative Abnormal 
Returns,” and Page Index 20 a diagram titled “Fraud 
Triangle.”}
\\boxed{plan} { 
 1) Locate the slide titled “Cumulative Abnormal Returns.”  
2) Read the values for Year 0 and Year 3 on that chart.  
3) Compute the difference between Year 3 and Year 0 
percentages.}
</think> <fetch_page > [14] </fetch_page >
Figure 7: Case 1 in SlideVQA between different methods

Question: What is the difference in percentage from Year 0 to Year 3 
regarding cumulative abnormal returns?
Input All Pages
··· ···<answer> 
The World's 
Conflicts Focus 
2009
</answer>
RAG -Top 5
Page 1 Page 3 Page 6 Page 13 Page 18
<answer> The World's Conflicts Focus 2009 </answer>
Doc-V* (SFT & GRPO)
<think> \\boxed{analysis} { Page Index 1 is the cover. It 
clearly states “Purchase now! Send a mail to: 
info@nordstromfrank.se €2,500 Contains facts, maps, flags 
and templates for use in InDesign.” This directly answers 
what can be purchased for 2,500 euro: the World Conflicts 
2009 Editorial Toolbox Focus. } \\boxed{relevant_pages} { 1 }
\\boxed{summary} { From Page Index 1 (cover), we learn 
that for 2,500 euro, one can purchase the World Conflicts 
2009 Editorial Toolbox Focus. }
</think> <answer> The World's conflicts 2009 Editorial 
Toolbox Focus </answer><think> \\boxed{thumbnail} {The thumbnail shows a 20 -page 
booklet titled “World Conflicts 2009” by Nordstrom Frank. 
The first page appears to be a cover with pricing 
information ; subsequent pages include maps, symbols, country 
profiles, and conflict data. The price “2,500” likely appears 
prominently on the cover.}
\\boxed{plan} { 
 1) Locate where the price “2,500 euro” is mentioned. 
 2) Identify what item is available for that price. 
 3) Provide the concise answer. }
</think> <fetch_page > [1] </fetch_page >
Doc-V* (SFT)
Doc-V* (GRPO )
<think> \\boxed{analysis} { …… This directly answers what can be 
purchased for 2,500 euro: a supplement containing facts, maps, 
flags, and templates for InDesign. } \\boxed{relevant_pages} { 1 }
\\boxed{summary} {From Page Index 1, we learned that for 
2,500 euro, you can purchase a supplement containing facts, 
maps, flags, and templates for use in InDesign . }
</think> <answer> a supplement containing facts, maps, flags, and 
templates for InDesign  </answer>
Figure 8: Case 2 in SlideVQA between different methods

Question: What specific word is the red arrow pointing toward?
Input All Pages
··· ···<answer> 
Problem
</answer>
RAG -Top 5
Page 2 Page 3 Page 5 Page 6 Page 15
<answer> Problem </answer>
Doc-V* (SFT & GRPO)
<think >  
\\boxed{analysis} {
Page Index 6: Shows a “DRAMATIC STORY CANVAS: 
Storyboard Template.” There is a red arrow pointing to a red 
box labeled “Problem” with the subtitle “Pain: (Un)Known.” 
This directly answers the question about what the red arrow 
points toward .
Page Index 11: Another “DRAMATIC STORY CANVAS FOR 
DEVELOPING ‘GOOD STRATEGY’” with a red arrow pointing 
to the same red box labeled “Problem,” confirming the label.
Page Index 2: “DRAMATIC STORY CANVAS: Space -Timeline 
Template” also has a red arrow pointing to the red box 
labeled “Problem .”All three pages provide consistent evidence; 
the specific word is “ Problem .”} 
\\boxed{relevant_pages } {6, 11, 2 } 
\\boxed{summary } {Across Page Index 6, 11, and 2, the red 
arrow clearly points to a red box labeled “Problem.” This 
resolves the question .}
 </think > <answer >Problem </answer ><think>
 \\boxed{thumbnail} {The document appears to be a slide deck. 
Each slide has a red arrow pointing to a specific field on the 
template. The question likely refers to the target of one of 
these red arrows on a template slide .}
\\boxed{plan} { 
1) Identify which page contains a red arrow pointing to a 
specific labeled field.
2) Read the label/word the red arrow points toward.}
</think> 
<page_retrieval >Find the page(s) where a red arrow points to a 
specific labeled field; identify the word the red arrow 
targets. </page_retrieval >
Page 2
Page 6
Page 11
Figure 9: Case 3 in SlideVQA between different methods
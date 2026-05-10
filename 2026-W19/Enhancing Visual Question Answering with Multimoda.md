# Enhancing Visual Question Answering with Multimodal LLMs via Chain-of-Question Guided Retrieval-Augmented Generation

**Authors**: Quanxing Xu, Ling Zhou, Xian Zhong, Xiaohua Huang, Rubing Huang, Chia-Wen Lin

**Published**: 2026-05-05 14:18:11

**PDF URL**: [https://arxiv.org/pdf/2605.03790v1](https://arxiv.org/pdf/2605.03790v1)

## Abstract
With advances in multimodal research and deep learning, Multimodal Large Language Models (MLLMs) have emerged as a powerful paradigm for a wide range of multimodal tasks. As a core problem in vision-language research, Visual Question Answering (VQA) has increasingly employed MLLMs to improve performance, particularly in open-domain settings where external knowledge is essential. In this work, we aim to further enhance retrieval-based VQA by more effectively integrating MLLMs with structured reasoning and knowledge acquisition. We introduce a logical prompting strategy that fuses Chain-of-Thought (CoT) reasoning with Visual Question Decomposition (VQD), termed CoVQD, to guide retrieval toward more accurate and relevant knowledge for MLLM inference. Building on this idea, we propose a new framework, CoVQD-guided RAG (CgRAG), which enables MLLMs to access more comprehensive and coherent external knowledge while benefiting from structured visual-text reasoning guidance, thereby improving generalization and reliability in complex cross-domain VQA scenarios. Extensive experiments on E-VQA, InfoSeek, and OKVQA benchmarks demonstrate the effectiveness of the proposed method.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 1
Enhancing Visual Question Answering with
Multimodal LLMs via Chain-of-Question Guided
Retrieval-Augmented Generation
Quanxing Xu , Ling Zhou , Xian Zhong ,Senior Member, IEEE, Xiaohua Huang ,Senior Member, IEEE,
Rubing Huang ,Senior Member, IEEE, and Chia-Wen Lin ,Fellow, IEEE
Abstract—With advances in multimodal research and deep
learning, Multimodal Large Language Models (MLLMs) have
emerged as a powerful paradigm for a wide range of multimodal
tasks. As a core problem in vision-language research, Visual
Question Answering (VQA) has increasingly employed MLLMs to
improve performance, particularly in open-domain settings where
external knowledge is essential. In this work, we aim to further
enhance retrieval-based VQA by more effectively integrating
MLLMs with structured reasoning and knowledge acquisition.
We introduce a logical prompting strategy that fuses Chain-of-
Thought (CoT) reasoning with Visual Question Decomposition
(VQD), termed CoVQD, to guide retrieval toward more accurate
and relevant knowledge for MLLM inference. Building on this idea,
we propose a new framework, CoVQD- guided RAG (CgRAG),
which enables MLLMs to access more comprehensive and coherent
external knowledge while benefiting from structured visual-
text reasoning guidance, thereby improving generalization and
reliability in complex cross-domain VQA scenarios. Extensive
experiments on E-VQA, INFOSEEK, and OKVQA benchmarks
demonstrate the effectiveness of the proposed method.
Index Terms—Multimodal large language models, open-domain
visual question answering, question decomposition, retrieval-
augmented generation
I. INTRODUCTION
VISION–language tasks serve as representative benchmarks
for evaluating models’ capabilities in multimodal learning
and visual-linguistic understanding, including visual story-
telling [1], video captioning [2], [3], and Visual Question
Manuscript received February 25, 2026. This work was supported in
part by the Science and Technology Development Fund of Macau, Macao
SAR, under Grants 0035/2023/ITP1 and 0021/2023/RIA1, the National
Natural Science Foundation of China under Grant 62271361, and the Hubei
Provincial Key Research and Development Program under Grant 2024BAB039.
(Corresponding authors: Ling Zhou and Xian Zhong.)
Quanxing Xu and Ling Zhou are with the School of Computer Sci-
ence and Engineering, Macau University of Science and Technology,
Macao SAR 999078, China (e-mail: 3230002299@student.must.edu.mo;
lzhou@must.edu.mo).
Xian Zhong is with the Hubei Key Laboratory of Transportation Internet
of Things, School of Computer Science and Artificial Intelligence, Wuhan
University of Technology, Wuhan, Hubei 430070, China, and also with the
State Key Laboratory of Maritime Technology and Safety, Wuhan University
of Technology, Wuhan 430063, China (e-mail: zhongx@whut.edu.cn).
Xiaohua Huang is with the Oulu School, Nanjing Institute of Technology,
Nanjing 210096, China (e-mail: xiaohuahwang@gmail.com).
Rubing Huang is with the School of Computer Science and Engineering,
Macau University of Science and Technology, Macao SAR 999078, China,
and also with the Zhuhai MUST Science and Technology Research Institute,
Macau University of Science and Technology, Zhuhai, Guangdong 519099,
China (e-mail: rbhuang@must.edu.mo).
Chia-Wen Lin is with the Department of Electrical Engineering, National
Tsing Hua University, Hsinchu 30013, Taiwan (e-mail: cwlin@ee.nthu.edu.tw).
(a)
(b)yellow box
MLLM
MLLM
MLLM
MLLM
CoVQDVQD CoT
CoVQDVQD CoTRAG
MLLM
MLLM
 RAG
ImageQuestion
ImageQuestion
AnswerAnswerAnswerAnswerEncoderEncoder
VQD CoT
CoVQD
Fig. 1:Comparison between prior retrieval-based VQA
with MLLMs and the proposed CgRAG framework.With
an MLLM fine-tuned by liDPO, VQD and CoT are fused to
guide fine-grained RAG, yielding an enhanced MLLM-based
VQA framework.
Answering (VQA) [4]. As one of the most fundamental vision-
language tasks, VQA requires generating accurate natural
language answers given an image and a question, and has
therefore attracted substantial attention in recent years. This
growing interest reflects a broader shift in the image processing
community from conventional “bucketed” recognition problems
toward more complex multimodal reasoning challenges [4],
[5].
Knowledge-based VQA (KBVQA) [6], an important subtask
of VQA, explicitly requires information beyond an image’s
visual content. Early KBVQA approaches were predominantly
retrieval-based, leveraging structured knowledge bases such as
Wikipedia or ConceptNet. With the advent of large language
models (LLMs), more recent methods increasingly rely on
frozen LLMs as implicit repositories of world knowledge.
Alongside advances in multimodal research, Multimodal LLMs
(MLLMs) have further expanded this paradigm, ushering in a
new era of multimodal generation and reasoning. Nevertheless,
how to more fully unlock the reasoning potential of large
models, improve domain adaptability, and mitigate predictive
hallucinations remains a pressing challenge in open-domain
VQA.
Although existing methods [7], [8], [9], [10] have achieved
strong performance in open-domain VQA by integrating
external knowledge through multimodal Retrieval-Augmented
Generation (RAG) or reorganizing knowledge in a fine-grained
manner, they often overlook two critical factors inherent in
question-image (QI) pairs, as illustrated in Fig. 1(a): (1) the
progressive logical structure implied by the question, and (2)
the controllability of the retrieval process. Meanwhile, prior
studies [11], [12], [13], [14] have shown that Visual Question
Decomposition (VQD) and Chain-of-Thought (CoT) reasoningarXiv:2605.03790v1  [cs.CV]  5 May 2026

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 2
What is the year 
this aircraft took 
the last flight?
Ground Truth   
2003
Direct answering2003
Visual question decompsition1. What model is this aircraft?
3. When  was this aircraft produced?2. Where is this aircraft  made?
...
Fig. 2:The illustration of the VQD.The generation of Chain-
of-Question via VQD on the input question can benefit the
MLLM’s exploration of the knowledge behind the given image.
can substantially enhance comprehension and inference in
large models. These observations motivate the development of
a framework that supports hierarchical question decomposition
and logic-guided retrieval for MLLM-based VQA.
In this work, we propose a CoVQD- guided RAG (CgRAG)
framework to boost the performance of MLLM-based KBVQA.
As illustrated in Fig. 1(b), this framework integrates CoT
and VQD into a multi-grained retrieval strategy to achieve
the aforementioned performance improvement. Specifically,
CgRAG introduces Chain-of-VQD (CoVQD) to extract fine-
grained multimodal information from input QI pairs, which
is then leveraged to guide a structured RAG process. To
effectively incorporate the retrieved knowledge, we design
a flexible prompting scheme for MLLM inference. Moreover,
as illustrated in Fig. 2, compelling the MLLM to perform VQD
on the input question and output a chain of questions could
enhance understanding of knowledge relevant to the given
image. Therefore, to further enhance the analytical capability
of MLLMs on QI pairs, we propose a new fine-tuning strategy
termed logical implication Direct Preference Optimization
(liDPO).
The proposed CgRAG framework consists of three stages:
1) Dissecting Chain Generation (DCG), which constructs CoT
enriched with fine-grained sub-questions for deeper analysis;
2) Elaborate Knowledge Retrieval (EKR), which leverages
DCG outputs to obtain relevant external knowledge; and
3) Comprehensive Prompt Construction (CPC), which fuses
implicit and explicit knowledge into a unified prompt to
guide progressive reasoning. These stages are sequentially
connected, enabling MLLMs to produce more accurate answers
accompanied by coherent explanations. Notably, the proposed
framework is model-agnostic and can be readily integrated
with different MLLMs, thereby unleashing their potential for
explanatory VQA.
Our key contributions are summarized in three folds:
•We fuse CoT and VQD to construct a logical multi-
question chain for knowledge retrieval, and propose anew framework, CoVQD-guided RAG (CgRAG), which
enables more abundant and accurate knowledge acquisition
to build robust prompts for open-domain VQA with
MLLMs.
•We design a novel fine-tuning strategy for MLLMs, termed
liDPO, which further enhances their VQD capability by
explicitly encouraging correct logical relations among
decomposed sub-questions.
•Extensive experiments on E-VQA, INFOSEEK, and
OKVQA demonstrate the effectiveness of the proposed
method, achieving competitive performance compared
with existing approaches.
II. RELATEDWORK
A. Knowledge-Based Visual Question Answering
Knowledge-based Visual Question Answering (KBVQA)
requires models to jointly reason about visual content, textual
questions, and external knowledge sources to generate accurate
answers. According to the modality used during knowledge
retrieval, existing retrieval-based KBVQA methods can be
broadly categorized into three types: textual-only methods
(e.g., Wiki-LLaV A [15]), visual-only methods (e.g., RORA-
VLM [16], EchoSight [17], ReflectiV A [7], MMKB-RAG [9]),
and methods that combine visual and textual modalities (e.g.,
DPR_V+T, KU-RAG [8], VLM-PRF [10]). These approaches
demonstrate that incorporating external knowledge through
retrieval can substantially improve open-domain VQA per-
formance. In contrast to prior work that relies on a single
or loosely coupled modality, our work aims to retrieve and
exploit fused multimodal information to better support MLLM
reasoning. By leveraging hybrid, multi-level knowledge that
spans fine-grained visual details, contextual linguistic cues, and
external factual resources, the proposed approach facilitates
more coherent step-by-step reasoning.
B. Question Decomposition
Question decomposition has proven effective in enhancing
the reasoning capability of LLMs by breaking complex queries
into simpler, more manageable sub-questions. Recently, this
paradigm has been extended to multimodal scenarios. For in-
stance, Youet al.[11] propose an iterative decomposition frame-
work that enables models to tackle complex vision–language
reasoning tasks through progressive steps, while Qiet al.[12]
demonstrate that Socratic-style recursive questioning can guide
LLMs toward deeper and more interpretable reasoning. Inspired
by these advances, our work introduces Visual Question De-
composition (VQD) into MLLM-based VQA, enabling complex
visual–textual queries to be decomposed into structured sub-
questions and to form a chain of questions. This design
improves reasoning transparency, enhances retrieval precision,
and ultimately leads to more accurate and interpretable answers.
C. Reinforcement Learning for Large Models
Reinforcement learning (RL) [18] is a machine learning
paradigm that optimizes its policy to maximize long-term

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 3
cumulative reward through interaction with the environment, re-
ceiving corresponding reward or penalty feedback for its actions.
Recently, RL has been widely applied to optimize the behavior
of large models and improve their reasoning ability [19],
[20]. Moreover, with the emergence of Direct Preference
Optimization (DPO) [21], a mainstream RL from Preference
Feedback (RLPF)-based preference alignment method, the
optimization paradigm can not only preserve the model’s
inherent generative and inferential capacity but also aligns its
output with human-like reasoning norms, factual consistency,
and step-by-step logical rigor. Building on this line of research,
our work develops a DPO strategy that explicitly promotes
MLLM’s understanding of QI pairs and supports logical
question decomposition.
III. PROPOSEDMETHOD
In this section, we elaborate on the proposed CgRAG system,
including its overall architecture and key components. An
overview of the framework is presented in Section III-A ,
highlighting the core design principles. The CgRAG framework
is organized into three main components: 1) the generation of
a detailed reasoning chain based on Visual Question Decompo-
sition (VQD), described in Section III-B ; 2) refined Retrieval-
Augmented Generation (RAG), detailed in Section III-C ; and
3) comprehensive prompt construction that integrates multi-
granular knowledge, outlined in Section III-D . Together, these
components form a cohesive pipeline that effectively enhances
open-domain explanatory VQA performance.
A. Overview
Advancing the reasoning capability of LLMs has long
been a central goal in artificial intelligence. Recent years
have witnessed substantial progress in both unimodal and
multimodal LLMs. Although prior studies have improved
explanatory VQA through causal reasoning [22], contrastive
learning [23], or the use of frozen LLMs [24], the reasoning
potential of MLLMs in open-domain explanatory VQA remains
insufficiently explored. Existing MLLM-based approaches
primarily emphasize enriching fine-grained knowledge through
retrieval, while the role of structured question decomposition
in guiding retrieval has received less attention. We posit that
retrieval explicitly guided by detailed VQD can further improve
the effectiveness of reasoning.
The proposed CgRAG framework integrates the strengths
of existing augmentation strategies by introducing a multi-
granular, step-by-step reasoning process based on CoVQD,
together with a fine-grained and accurate RAG mechanism.
Specifically, to transform the input question into a chain of
questions, we fuse CoT reasoning with VQD to form CoVQD,
which extracts detailed multimodal information from the input
image and question and serves as structured guidance for
retrieval. On this basis, we develop an MLLM-based pipeline
that can be seamlessly integrated with different backbone
MLLMs. A flexible prompt construction strategy is further
adopted to effectively organize retrieved knowledge and align
it with MLLM inference.As shown in Fig. 3, the proposed framework comprises
three stages: Dissecting Chain Generation (DCG), Elaborate
Knowledge Retrieval (EKR), and Comprehensive Prompt
Construction (CPC). DCG constructs CoT enriched with fine-
grained sub-questions derived from VQD, enabling detailed
analysis of both visual content and question semantics. EKR
subsequently retrieves external knowledge guided by the DCG
outputs, providing relevant factual support for reasoning. CPC
then integrates the outputs of DCG and EKR into a unified
prompt, guiding the MLLM toward progressive and coherent
reasoning. Each stage builds upon the previous one, enabling
the generation of more accurate answers accompanied by
detailed explanations.
B. Dissecting Chain Generation
The DCG module constitutes the first stage of the CgRAG
pipeline and integrates CoT reasoning with VQD to produce
CoVQD, which plays a central role in subsequent retrieval and
reasoning. The predictive mechanism of LLMs in VQA [25]
and VQD [26] is rooted in in-context learning [27], which
enables frozen models to perform zero-shot inference. Given
a constructed prompt p, a frozen model generates an answer
sequencey= (y 1, . . . , y z)via autoregressive decoding:
ˆyt= arg max
ytpθ 
yt|p,ˆy<t
,(1)
where pθ(·)denotes the conditional token distribution parame-
terized by model weights θ,ˆy<t={ˆy1, . . . ,ˆyt−1}represents
previously generated tokens, and tindexes the decoding step.
This objective corresponds to the Next Token Prediction (NTP)
loss.
Although existing MLLMs (e.g., LLaV A-1.5 [28] and Qwen-
VL [29]) achieve strong performance in VQA, their VQD
capability is often limited [30]. To address this issue, we first
fine-tune the MLLM using a dedicated dataset to improve its
VQD ability. Specifically, we adopt the SelectiveVQD loss [30],
which combines NTP loss with Binary Cross-Entropy (BCE)
loss to supervise question-decomposition decisions. The BCE
loss is defined as
LBCE=−[ylog (ˆy) + (1−y) log (1−ˆy)],(2)
where y∈ {0,1} is the ground-truth label indicating whether
decomposition is required, and ˆy∈(0,1) is the predicted
probability. The overall SelectiveVQD loss is given by:
LSelectiveVQD =NX
i=1(λLNTP,i +βL BCE,i ),(3)
where iindexes training samples, Nis the total number of
samples, and λ, β are weighting hyperparameters balancing
the generative and decomposition objectives.
To further enhance the model’s ability to capture logical
relations among sub-questions, we introduce logical implication
Direct Preference Optimization (liDPO) during fine-tuning. As
illustrated in Fig. 4, following [31], we employ a pre-trained
BERT [32] with a visual encoder to predict logical relations
among sub-questions conditioned on both visual and textual
inputs. These relations are then used to construct preference
pairs for liDPO, as shown in Fig. 5.

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 4
DCG
             Multi -modal Large Language Model
IO :
IO :Decomposing the 
input question...P :
Decomposing the 
input question...P :CoVQD Knowledge Fine -grained RAGPrompt
ConstructionAnswering the 
question according 
to...P :
A :
school busA :
school busWhat is in the tow zone?QO :
What is in the tow zone?QO :E :
The bus top on the area
with a sigh of tow zone... E :
The bus top on the area
with a sigh of tow zone... 
EKR CPC
DCG
Prompt
ConstructionAnswering the 
question according 
to...P EKR CPC
KnowledgeKnowledge Fine -grained RAG
 Fine -grained RAG
             Multimodal Large Language Model
              Multimodal Large Language Model
What is in 
the tow 
zone?
CoVQD
CoVQD
A “school bus ” E “The vehicle  on the area with a sigh of tow zone              has a plate  of school bus           . ” 
 “The vehicle  on the area with a sigh of tow zone              has a plate  of school bus           . ” 
 A “school bus ” E “The vehicle  on the area with a sigh of tow zone              has a plate  of school bus           . ” 
Input Generate
OutputInput
Fig. 3:Overall architecture of the proposed CgRAG framework.The pipeline consists of three components: Dissecting
Chain Generation (DCG), which constructs CoT guided by VQD from the input image and question; Elaborate Knowledge
Retrieval (EKR), which retrieves external knowledge under the guidance of CoVQD; and Comprehensive Prompt Construction
(CPC), which aggregates implicit and explicit knowledge for inference.
What kind of athlete was the man in the 
black and white image most likely?QO
Does the man's appearance suggest 
that he is a football player?subQ1
Does the black and white image show 
the man participating in cycling?subQ2
Does the sign underneath the image 
indicate that the man is associated with 
the Tour de France bike race?subQ3
VEBERT
BERT
BERTVisual Input Question Input
subQ1    subQ2    subQ3→ → subQ1    subQ2    subQ3→ →
Logical relation 
(Decomposition order)
Logical relation 
(Decomposition order)
Fig. 4:Prediction of logical relations among sub-questions.
A pre-trained BERT is employed to infer logical implications
between sub-questions for liDPO.
In the VQD setting, preference optimization aims to align
the model with desired outputs under a reward function while
constraining deviation from a reference policy. Given the input
question Q, image V, and an output sub-question sequence O,
the policy πθinduces a conditional distribution πθ(O|Q, V) .
To avoid over-optimization [33], the optimization objective
regularizes the divergence between πθand a reference policy
πref(initialized from the same checkpoint). The corresponding
preference-optimization loss is written as:
LPO=−logσ
r(Q, V, O)−βlogπθ(O|Q, V)
πref(O|Q, V)
,(4)
where r(Q, V, O) denotes the reward associated with O,σ(·) is
the sigmoid function, and βcontrols the regularization strength.
Direct Preference Optimization (DPO) [21] further simplifies
this alignment by directly maximizing the reward gap between
a preferred output Owand a rejected output Olunder theBradley-Terry model [34]. Its objective is the following.
LDPO =−logσ
βlogπθ(Ow|Q, V)
πref(Ow|Q, V)
−βlogπθ(Ol|Q, V)
πref(Ol|Q, V)
,(5)
where OwandOlare the preferred and rejected sub-question
sequences, respectively.
As a multimodal optimization objective, the proposed liDPO
is implemented as a two-stage procedure. In the first stage, we
construct preference pairs using the logical relations predicted
by BERT: the sub-question order consistent with the desired
logical relation serves as Ow, while inconsistent orders are
treated as Ol. In the second stage, we optimize the MLLM
with the DPO objective. Moreover, inspired by [35], [36], we
incorporate an anchor-based objective to consistently reinforce
high-quality outputs:
LAncPO =−logσ
βlogπθ(O|Q, V)
πref(O|Q, V)
,(6)
where Odenotes an anchor output with high quality. The final
liDPO loss is:
LliDPO =L DPO+γL AncPO ,(7)
whereγweights the anchored objective.
During inference, given an image-question pair (IO, QO), the
fine-tuned MLLM decomposes QOinto an ordered set of sub-
question-answer pairs {qa1, . . . , qa n}, forming a structured
CoVQD. Here, ndenotes the number of decomposed sub-
questions. This structure provides interpretable intermediate
reasoning steps and serves as explicit guidance for downstream
retrieval.
Formally, the generated CoVQD is organized as an ordered
chain that preserves both the visual context and the sequential
dependency among sub-question-answer pairs:
Head:Please decompose the given question into sub-
questions for easier answering according to the given image.
Context:I O
Question:Q O

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 5
① Construct Rejected data
Desired set:
Undesired set:subQ1 → subQ2 → subQ3
subQ2 → subQ3 → subQ1
subQ1 → subQ3 → subQ2
subQ2 → subQ1 → subQ3
subQ3 → subQ2 → subQ1
subQ3 → subQ1 → subQ2② Preference Optimization
r(V,Q,O) r(V,Q,O r)＞
MLLMQ:Q:V:V:
O:subQ1
subQ2
subQ3Q:V:
O:subQ1
subQ2
subQ3Q:Q:V:V:
Or:subQ3
subQ2
subQ1Q:V:
Or:subQ3
subQ2
subQ1Q:Q:V:V:
O:subQ2
subQ3
subQ1Q:V:
O:subQ2
subQ3
subQ1
＞(O)
(Or)
r(V,Q,O) 0＞ r(V,Q,O) 0＞AnchorLogical 
Relation
Preference
Optimization
Fig. 5:Overview of logical implication Direct Preference Optimization (liDPO).The two-stage procedure includes rejected
data construction and preference optimization, where the desired set ( O) contains logically correct sub-question sequences and
the undesired set (O r) contains incorrect ones.
IO
CoVQD
1. Original Image Retrieval
“A child in a 
"Rangers" 
baseball uniform 
kneels on a 
grassy lawn , 
ready to play 
with bat in 
hand.”
CO IP
COQO
IPProcessing
2. Multimodal Retrieval
COCoVQD
IP
3. CoVQD -guided 
RetrievalIs this kid practicing?QO
Filtering and IntegrationE
Prompt
Instruction
CRCR
IP
QOQOK
subQAE 1
subQAE 2
subQAE 3
subQAE 4K
subQAE 1
subQAE 2
subQAE 3
subQAE 4
IS
IOPrompt
Instruction
CR
IP
QOK
subQAE 1
subQAE 2
subQAE 3
subQAE 4
IS
IO
CR“A child wears  a baseball 
uniform , holds a baseball bat .”
IS
E...“Because a baseball bat …”
“Because child is smiling …”“Because a baseball bat …”
“Because child is smiling …” ...“Because a baseball bat …”
“Because child is smiling …”CR“A child wears  a baseball 
uniform , holds a baseball bat .”
IS
E...“Because a baseball bat …”
“Because child is smiling …”What is the kid holding in his hands?   
He is holding a baseball bat .subQA1
Is the kid playing baseball right now?
No, the bat is touching the ground 
and the tip of the bat is facing down.
What does the kid wear?
He wears a baseball uniform.
What is the kid doing?
The kid is posing for a picture.subQA2
subQA3
subQA4What is the kid holding in his hands?   
He is holding a baseball bat .subQA1
Is the kid playing baseball right now?
No, the bat is touching the ground 
and the tip of the bat is facing down.
What does the kid wear?
He wears a baseball uniform.
What is the kid doing?
The kid is posing for a picture.subQA2
subQA3
subQA4
Fig. 6:Illustration of the Elaborate Knowledge Retrieval (EKR) module.Three retrieval processes are involved: original
image retrieval, multimodal retrieval, and CoVQD-guided retrieval. I,C,E, and Kdenote Image, Caption, Explanation, and
Knowledge, respectively. And QOand CoVQD serve as a supervisor for filtering out visual information irrelevant to reasoning.
IO\n qa 1\n . . .\n qa n (8)
where\ndenotes a delimiter separating consecutive elements in
the chain. This explicit structure serves as a unified intermediate
representation that is subsequently used to guide fine-grained
knowledge retrieval during the EKR stage.
C. Elaborate Knowledge Retrieval
Following DCG, the EKR module performs multi-level
external knowledge retrieval. As illustrated in Fig. 6, it consists
of three retrieval processes: original image retrieval, multi-
modal retrieval, and CoVQD-guided retrieval, corresponding
to increasing levels of granularity. The original image IO,
the question QO, and the generated CoVQD jointly serveas retrieval inputs. It is worth noting that in multimodal
retrieval and CoVQD-guided retrieval, QOand CoVQD act
as a supervisor for filtering out visual information irrelevant
to reasoning, respectively. After the three retrieval and the
filtering method following [10], the more comprehensive and
fine-grained multimodal information is obtained, which contains
the Refined Caption CR, Searched Image IS, and Explanation
E. And this information would be integrated into the final
prompt for the MLLM’s prediction.
Concretely, original image retrieval provides coarse visual
context. Multimodal retrieval leverages VinVL [37] and BLIP-
2 [38] to generate refined image patches IPand a global
caption CO, which are retrieved under the supervision of
QO. CoVQD-guided retrieval further refines this process by
sequentially conditioning retrieval on each sub-question-answer

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 6
pair, ensuring logical consistency across nretrieval steps.
Retrieved visual and textual candidates are projected into a
shared embedding space using vision-language encoders such as
CLIP [39] or BLIP [40]. For each sub-question (qi, ai), cosine
similarity is computed to select the most aligned explanation ei,
yielding a set of Question-Answer-Explanation (QAE) triples
{(qi, ai, ei)}n
i=1.
D. Comprehensive Prompt Construction
The CPC stage integrates the outputs of DCG and EKR
to guide MLLM inference. Unlike existing approaches, our
prompt jointly incorporates refined visual content and struc-
tured logical knowledge, reducing irrelevant information and
mitigating hallucinations. The constructed prompt consists of
five components: instruction head H, refined caption CR, image
patches IP, logical knowledge K={(q i, ai, ei)}n
i=1, and the
original questionQ O:
H\nC R\nIP\nK\nQ O.(9)
By integrating these components into a unified prompt, the
proposed CgRAG framework provides a comprehensive, multi-
granular context for inference, improving answer accuracy and
explanation reliability in open-domain explanatory VQA.
IV. EXPERIMENTALRESULTS
In this section, we design explicit experiments to validate
the effectiveness of the proposed CgRAG. Specifically, the
experimental settings and research questions are presented
in Section IV-A ; the main quantitative comparisons against
existing methods and evaluations with different backbone
models are reported in Section IV-B ; exploratory studies
that analyze key components and other influential factors
are provided in Section IV-C ; and a qualitative analysis of
CgRAG’s out-of-domain (OOD) behavior is presented in
Section IV-D.
A. Experimental Settings
1) Datasets and Metrics:To evaluate the reasoning capacity
of CgRAG in zero-shot open-ended KBVQA, we adopt OK-
VQA [41], ENCYCLOPEDIC-VQA (E-VQA) [42], and INFOS-
EEK[43] in the main experiments. Specifically, OK-VQA is a
widely used KBVQA benchmark containing 9K and 5K image-
question pairs for training and testing, respectively. E-VQA
is a large-scale dataset that focuses on fine-grained categories
and instance-level properties. It contains approximately 221K
unique question-answer pairs, each associated with up to five
images, resulting in nearly 1M VQA samples. Moreover, E-
VQA provides a controlled knowledge base derived from
Wikipedia, which supplies explicit evidence for each annotated
answer and enables knowledge-grounded evaluation. INFOSEEK
is a large-scale benchmark for information-seeking VQA, in
which answering requires external knowledge beyond common
sense. It comprises approximately 1.3M VQA pairs aligned with
11K images sourced from OVEN [44]. The data are split into a
training set (934K pairs) and a validation set (73K pairs), with
strict separation by both entities and questions. The validationset is further divided into Unseen Entity and Unseen Question
subsets, facilitating a fine-grained evaluation of generalization
to novel concepts and query forms. In addition, we employ
GQA-REX [45], which extends the GQA benchmark [46]
with multimodal annotations that capture the visual reasoning
process, and we evaluate on GQA-OOD [47], a benchmark
specifically designed for assessing OOD robustness.
For VQA, we use VQA accuracy [5] to evaluate answer
correctness. For explanation generation, we adopt standard
automatic metrics, including BLEU-4, METEOR, ROUGE-L,
CIDEr, and SPICE, to assess explanation quality, and we use
the Grounding metric [45] to evaluate whether the generated
explanation correctly localizes relevant visual regions.
2) Methods and MLLMs:To assess the proposed method
objectively, we compare two categories of MLLM-based VQA
approaches: zero-shot MLLMs and retrieval-augmented models.
For zero-shot MLLMs, we evaluate BLIP-2 [38], Instruct-
BLIP [48], LLaV A-1.5 [28], GPT-4V [49], Qwen2-VL [50],
Qwen2.5-VL [51], and LLaV A-NeXT [52]. For retrieval-
augmented models, we compare against DPR_V+T [53],
RORA-VLM [16], EchoSight [17], Wiki-LLaV A [15], Reflec-
tiV A [7], KU-RAG [8], MMKB-RAG [9], and VLM-PRF [10].
To explore generalization within the same pipeline, we further
instantiate CgRAG with different backbone MLLMs (Qwen2-
VL, Qwen2.5-VL, LLaV A-NeXT, and InternVL3 [54]).
3) Implementation:For MLLM fine-tuning, we follow the
training strategy of [30]: all MLLMs are fine-tuned on De-
coVQD+ using the SelectiveVQD loss. For BERT pretraining,
we follow [31]: BERT is pretrained on SNLI [55] for five
epochs, initialized withbert-base-uncased, using a batch size
of 16, weight decay of 0.01, and AdamW with a learning rate
of2×10−5. The same configuration is then used to fine-tune
the model on a subset of 2,000 manually annotated proposition
pairs from INTROSPECT[56]. For liDPO fine-tuning, we set β
as 0.5, and the learning rate is set up to 1×10−7, employ a
cosine scheduler with a warmup ratio of 0.03, and set γ= 1
by default. All MLLMs are trained for one epoch.
4) Research Question:To evaluate CgRAG comprehensively,
we design experiments around four research questions (RQs):
1) assessing the effectiveness of CgRAG relative to MLLM-
based methods in zero- and few-shot settings; 2) testing the
generalization of CgRAG across different backbone MLLMs; 3)
quantifying the contribution of each component and analyzing
the impact of key factors; and 4) qualitatively assessing
robustness across diverse domains. The RQs are summarized
as follows:
•RQ_1: How does CgRAG compare fairly with zero-shot
and few-shot methods?
•RQ_2: How does CgRAG’s performance vary across
different backbone MLLMs within the same pipeline?
•RQ_3: What are the contributions of each component, and
how do other factors influence performance?
•RQ_4: How does CgRAG perform across evaluation cases
from different domains?
The corresponding experiments are presented in the following
sections.

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 7
TABLE I:Overall performance comparison on E-VQA and INFOSEEKdatasets.Bold and underlined values indicate the
best and second-best results, respectively. “V” and “T” denote visual and textual features.
Method LLM/MLLM Retriever FeatureE-VQA INFOSEEK
Single-Hop All Unseen-Q Unseen-E All
Zero-shot MLLMs
InstructBLIP [48](NeurIPS’23)Flan-T5 XL - - 11.9 12.0 8.9 7.4 8.1
BLIP-2 [38](ICML’23)Flan-T5 XL - - 12.6 12.4 12.7 12.3 12.5
GPT-4V [49](arxiv’24)- - - 26.8 28.0 15.0 14.3 14.6
LLaV A-1.5 [28](CVPR’24)LLaMA-3.1-8B - - 16.0 16.9 8.3 8.9 7.8
LLaV A-1.5 [28](CVPR’24)Vicuna-7B - - 16.3 16.9 9.6 9.4 9.5
LLaV A-NeXT-7B [52](arxiv’24)- - - 22.1 20.8 23.2 23.4 23.1
LLaV A-NeXT-8B [52](arxiv’24)- - - 22.5 21.4 23.7 24.0 23.6
Qwen2-VL-7B [50](arxiv’24)- - - 16.2 17.0 15.4 16.8 16.1
Qwen2.5-VL-3B [51](arxiv’25)- - - 17.9 19.6 20.2 21.7 20.7
Qwen2.5-VL-7B [51](arxiv’25)- - - 21.7 20.3 22.8 24.0 23.2
Retrieval-Augmented Models
Wiki-LLaV A [15](CVPRW’24)LLaMA-3.1-8B CLIP ViT-L/14 + Contriever T 18.3 19.6 28.6 25.7 27.1
Wiki-LLaV A [15](CVPRW’24)Vicuna-7B CLIP ViT-L/14 + Contriever T 17.7 20.3 30.1 27.8 28.9
RORA-VLM [16](arxiv’24)Vicuna-7B CLIP + Google Search V - 20.3 25.1 27.3 26.0
EchoSight [17](EMNLP’24)Mistral-7B/LLaMA-3-8B EV A-CLIP-8B V 19.4 - - - 27.5
EchoSight [17](EMNLP’24)LLaMA-3.1-8B EV A-CLIP-8B V 26.4 24.9 18.0 19.8 18.8
ReflectiV A [7](CVPR’25)LLaMA-3.1-8B EV A-CLIP-8B V 35.5 35.5 28.6 28.1 28.3
MMKB-RAG [9](arxiv’25)Qwen2-VL-7B EV A-CLIP-8B V 39.7 35.9 36.4 36.3 36.4
DPR_V+T [53](ECIR’24)Multi-passage BERT CLIP ViT-B/32 V+T 29.1 - - - 12.4
KU-RAG [8](arxiv’25)GPT-4o EV A-CLIP-8B V+T 38.3 - - - 26.1
VLM-PRF [10](NeurIPS’25)Qwen2.5-VL-7B EV A-CLIP-8B V+T 28.9 28.6 40.0 39.4 39.5
VLM-PRFw/RL [10](NeurIPS’25)InternVL3-8B EV A-CLIP-8B V+T 40.1 39.2 43.5 42.142.5
CgRAG (Ours) Qwen2-VL-7B EV A-CLIP-8B V+T 39.6 38.3 40.8 39.2 36.3
CgRAG (Ours) Qwen2.5-VL-7B EV A-CLIP-8B V+T 39.8 38.6 41.3 39.5 36.6
CgRAG (Ours) LLaV A-NeXT-7B EV A-CLIP-8B V+T 39.9 39.0 41.7 40.5 39.5
CgRAG (Ours) LLaV A-NeXT-8B EV A-CLIP-8B V+T 39.9 39.2 41.9 40.9 39.7
CgRAG (Ours) InternVL3-8B EV A-CLIP-8B V+T 40.4 39.5 43.5 42.0 43.0
MLLM -based VQAPerformance on InfoSeek ( All)
2024 20252025303540Textual -only Visual -only Visual+Textual
202645
15
DPR V+TRORA -VLMEchoSightWiki -LLaVAReflectiVA
KU-RAGMMKB -RAGVLM -PRFCgRAG
(Ours)MLLM -based VQAPerformance on InfoSeek ( All)
2024 20252025303540Textual -only Visual -only Visual+Textual
202645
15
DPR V+TRORA -VLMEchoSightWiki -LLaVAReflectiVA
KU-RAGMMKB -RAGVLM -PRFCgRAG
(Ours)
Fig. 7:Performance comparison of MLLM-based VQA on
INFOSEEK-All.The best result of each method is reported,
and different shapes indicate distinct retrieval features.
B. Quantitative Results
To addressRQ_1andRQ_2, we compare CgRAG with two
categories of MLLM-based VQA methods: zero-shot MLLMs
and retrieval-augmented models. We instantiate CgRAG with
multiple backbone MLLMs to examine its generalization within
a unified pipeline. Table I reports results on E-VQA andTABLE II:Performance comparison on OK-VQA dataset.
Bold values indicate the best results.
Method LLM/MLLM Accuracy
Qwen2.5-VL-3B [51] - 62.0
Qwen2.5-VL-7B [51] - 72.4
KU-RAG [8] GPT-4o 77.2
CgRAG (Ours) InternVL3-8B 77.8
INFOSEEK, and Fig. 7 provides an intuitive comparison on
INFOSEEK-All. Overall, CgRAG with InternVL3 [54] achieves
a new state-of-the-art on E-VQA and INFOSEEK(Unseen-Q
andAll), while CgRAG paired with other MLLMs also yields
strong accuracy.
Specifically, CgRAG with InternVL3 attains 40.4% and
39.5% onSingle-HopandAllof E-VQA, and 43.5% and 43.0%
onUnseen-QandAllof INFOSEEK, which are the best results
among all compared methods. It achieves 42.0% onUnseen-
Eof INFOSEEK, which is slightly below the best-performing
model. Compared with methods that use both visual and textual
features, CgRAG improves over DPR_V+T by 11.3% onSingle-
Hopof E-VQA and by 30.6% on INFOSEEK; surpasses KU-
RAG by 2.1% onSingle-Hopof E-VQA and by 16.9% on
INFOSEEK; and exceeds VLM-PRF by 11.5% and 10.9% on
Single-HopandAllof E-VQA, and by 3.5%, 2.6%, and 3.5%
onUnseen-Q,Unseen-E, andAllof INFOSEEK, respectively.
Compared with the latest VLM-PRFw/RL, CgRAG achieves
slightly higher accuracy onSingle-HopandAllof E-VQA

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 8
TABLE III:Results on explanation generation and question answering for explanatory VQA.GQA- and OOD- denote
answer accuracy on GQA-REX and GQA-OOD, respectively. Bold values indicate the best results.
Method BLEU-4 METEOR ROUGE-L CIDEr SPICE Grounding GQA-val GQA-test OOD-val OOD-test
VQA-E [57](ECCV’18) 42.5 34.5 73.5 358.1 40.3 31.2 65.1 57.2 49.1 46.2
EXP [58](ACLW’19) 42.4 34.4 73.4 357.0 40.3 33.5 65.1 56.9 49.4 47.6
REX [45](CVPR’22) 54.7 39.5 79.3 465.9 49.9 70.7 78.1 58.1 71.2 52.1
VCIN [59](ICCV’23) 58.6 41.5 81.4 519.2 54.6 77.3 81.7 60.6 74.7 54.2
MRVQA [60](CVPRW’25) 16.9 23.0 44.2 107.0 23.6 18.4 38.2 32.7 28.0 26.5
CgRAG (Ours) 61.4 44.1 83.8 590.2 57.9 82.3 84.3 64.4 78.0 57.5
Fig. 8:Comparison of different methods on Explanatory
VQA.For visualization clarity, CIDEr scores are scaled to
one-tenth of their original values.
(both +0.3%) and onAllof INFOSEEK(+0.5%), while being
marginally lower onUnseen-Eof INFOSEEK(-0.1%).
These results indicate that CoVQD-guided refined retrieval
effectively strengthens MLLMs for open-domain KBVQA
by improving grounding and maintaining semantic coherence
across modalities. By using a chain-of-questions as a structured
retrieval signal, the model is encouraged to retrieve and
integrate evidence aligned with the decomposed reasoning steps,
thereby enabling robust cross-modal inference on knowledge-
intensive questions.
To further validate CgRAG for open-ended KBVQA, we
compare it with existing methods on OK-VQA. The results are
summarized in Table II. CgRAG consistently outperforms prior
retrieval-augmented baselines, and CgRAG with InternVL3
achieves the best performance among the MLLM-based meth-
ods compared. This demonstrates that CgRAG effectively
combines multimodal reasoning with external knowledge
retrieval, supporting accurate answers to knowledge-intensive
questions beyond surface-level visual recognition.
In addition, to evaluate the quality of explanations produced
within CgRAG, we adapt the framework to generate multimodal
explanations and assess performance on GQA-REX. Following
Chen and Zhao [45], we evaluate both VQA accuracy andTABLE IV:Consistency evaluation between predicted
answers and explanations on GQA-REX.“Con.”, “Vis.”,
and “Tex.” denote consistency, visual relevance, and textual
relevance, respectively. Bold values indicate the best results.
Method Con. Vis. Tex. Ave.
REX [45] 84.8 3.1 4.1 3.6
VCIN [59] 93.4 3.5 4.5 3.9
CgRAG (Ours) 97.0 3.9 4.9 4.5
TABLE V:Ablation study of DCG, EKR, and CPC
components in the CgRAG framework.Bold values indicate
the best results.
DCG EKR CPC E-VQA (Single-Hop) INFOSEEK(All)
# # # 23.1 24.0
 # # 29.6 30.3
#  # 37.5 35.4
  # 39.6 36.1
#   38.0 35.9
   40.4 43.0
explanation quality. To evaluate answer-explanation consistency,
we additionally report Visual Consistency (Vis.) and Textual
Consistency (Tex.) [59]. Table III and Fig. 8 shows comparisons
with VQA-E [57], EXP [58], REX [45], VCIN [59], and
MRVQA [60]. Overall, CgRAG achieves the best scores
across all explanation metrics, indicating that it produces
fluent, informative, and well-grounded explanations. In terms of
question answering, CgRAG also achieves the highest accuracy
on both in-distribution (GQA-REX validation/test) and out-of-
distribution (GQA-OOD validation/test) splits, demonstrating
robustness and generalization.
Table IV further compares answer-explanation consistency
on GQA-REX. CgRAG achieves the highest Consistency score
(97.0) and also leads on Vis. and Tex. scores (3.9 and 4.9),
resulting in the best average (4.5). These results suggest that
CgRAG improves not only answer accuracy and explanation
quality, but also the alignment between predicted answers and
their supporting explanations, thereby enhancing reliability in
explanatory VQA.
C. Ablation Study
To addressRQ_3, we conduct ablation studies to quantify
the contributions of the three components in CgRAG and to
analyze the impact of key factors.
1) Components:We first evaluate modular variants to assess
the contribution of DCG, EKR, and CPC. As shown in Table V,

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 9
TABLE VI:Ablation study of different retrieval modes within the CgRAG framework.Bold and underlined values indicate
the best and second-best results, “V” and “T” denote visual and textual features, respectively. And their results are highlighted
in different colors.
MLLM FeatureE-VQA INFOSEEK
Single-Hop All Unseen-Q Unseen-E All
Qwen2-VL-7B [50](arxiv’24)T 22.1 23.4 25.6 25.5 25.5
V 37.8 36.2 34.9 34.6 34.7
V+T 39.6 38.3 40.8 39.2 36.3
Qwen2.5-VL-7B [51](arxiv’25)T 22.6 24.0 27.7 27.0 27.3
V 38.3 36.9 35.5 35.7 35.5
V+T 39.8 38.6 41.3 39.5 36.6
LLaV A-NeXT-7B [52](arxiv’24)T 22.3 23.6 27.2 26.3 26.9
V 37.4 35.6 33.8 34.5 34.3
V+T 39.9 39.0 41.7 40.5 39.5
LLaV A-NeXT-8B [52](arxiv’24)T 23.0 24.1 28.6 27.1 27.6
V 38.1 36.7 35.0 35.2 35.1
V+T 39.9 39.2 41.9 40.9 39.7
InternVL3-8B [54](arxiv’25)T 23.5 25.2 29.6 27.6 28.8
V 38.5 34.3 35.6 35.6 35.5
V+T 40.4 39.5 43.5 42.0 43.0
Textual -only Visual -only Visual+Textual
Fig. 9:Ablation comparison of different retrieval modes across MLLMs.Results are reported on E-VQA-Alland INFOSEEK-
All, where “V” and “T” denote visual and textual features, respectively.
removing all modules yields the lowest performance (23.1 on
E-VQA-Single-Hopand 24.0 on INFOSEEK-All), highlighting
the necessity of these enhancements. Adding DCG alone
substantially improves performance, supporting its role in
promoting structured decomposition and CoT-style reasoning.
Introducing EKR further boosts accuracy, confirming the
importance of CoVQD-guided retrieval for acquiring relevant
external evidence. CPC also contributes both independently
and synergistically: while EKR+CPC improves upon EKR
alone, the full model achieves the best results (40.4 on E-
VQA-Single-Hopand 43.0 on INFOSEEK-All). This progressive
improvement indicates that each module plays a distinct role,
and their combination yields the strongest overall performance.
2) Factors:We further examine the effects of retrieval
mode, the number of QA pairs in CoVQD, and the final
prompt structure. As shown in Table VI and Fig. 9, textual-
only (T), visual-only (V), and combined visual-textual (V+T)
retrieval are evaluated. V+T consistently performs best across
all backbones and datasets, indicating that visual and textual
evidence are complementary for knowledge-intensive VQA.
Text-only retrieval is markedly weaker, suggesting that textualTABLE VII:Effect of the number of QA pairs ( K) in
CoVQD.Bold values indicate the best results.
K E-VQA (Single-Hop) INFOSEEK(All)
2 39.2 42.5
4 40.4 43.0
6 40.442.9
evidence alone is insufficient for complex visual reasoning,
while visual-only retrieval improves performance but remains
inferior to V+T fusion.
As shown in Table VII, increasing the number of QA pairs
Kfrom 2 to 4 improves performance (39.2 →40.4 on E-VQA-
Single-Hopand 42.5 →43.0 on INFOSEEK-All), indicating that
a moderate increase in decomposed context strengthens retrieval
alignment and reasoning depth. However, increasing Kto 6
does not bring further gains and slightly degrades INFOSEEK-
All(43.0 →42.9), suggesting diminishing returns and potential
noise from excessive context. These results support K=4 as a
practical trade-off between effectiveness and stability.
As shown in Table VIII, we compare two prompt structures

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 10
(e) (f)(a) (b)
(g)
Which country is this animal 
from?
Australia
Australia
Australia
Which country is this animal 
from?
Australia
Australia
Australia
(h)
What is the name of this place ?
Zaanse SchansZaanse SchansZaanse Schans
What is the name of this place ?
Zaanse SchansZaanse SchansZaanse Schans
Who invented the object in the 
picture?
Richard Trevithick
George StephensonGeorge Stephenson
Who invented the object in the 
picture?
Richard Trevithick
George StephensonGeorge Stephenson
1800
In which year was this building 
completed?
18001800
1800
In which year was this building 
completed?
18001800
 Sparrow
Tree SparrowWhat was the first subspecies of 
this bird?
Aleutian Savannah 
Sparrow
Sparrow
Tree SparrowWhat was the first subspecies of 
this bird?
Aleutian Savannah 
Sparrow
Dominikus Zimmermann
Johan DientzenhoferWho designed this palace ?
Balthasar  Neumann
Dominikus Zimmermann
Johan DientzenhoferWho designed this palace ?
Balthasar  Neumann
AmericaAmericaIn what country has this bee 
been  found?
Canada
AmericaAmericaIn what country has this bee 
been  found?
Canada
Philip JohnsonPhilip JohnsonWhich architect designed this 
building?
Cé sar Pelli
Philip JohnsonPhilip JohnsonWhich architect designed this 
building?
Cé sar Pelli
(c) (d)CgRAG
(Ours)Qwen2.5 -VL
LLaVA -NeXT
CgRAG
(Ours)Qwen2.5 -VL
LLaVA -NeXT
CgRAG
(Ours)Qwen2.5 -VL
LLaVA -NeXT
CgRAG
(Ours)Qwen2.5 -VL
LLaVA -NeXT
Fig. 10:Qualitative comparison across different knowledge domains.Performance differences between Qwen2.5-VL [51],
LLaV A-NeXT [52], and their CgRAG-enhanced variants are shown on cases from (a-b) commonsense, (c-d) animal, (e)
geography, (f) architecture, (g) history, and (h) art domains.
TABLE VIII:Effect of prompt structure on final perfor-
mance. C,I, and Kdenote caption, image, and knowledge,
respectively. Bold values indicate the best results.
Prompt Structure E-VQA (Single-Hop) INFOSEEK(All)
C-I-K 40.4 43.0
I-C-K 40.1 39.9
produced by CPC. The C-I-Korder achieves higher accuracy
thanI-C-K, indicating that providing a global caption before
image patches is more effective for structuring the subsequent
integration of retrieved knowledge.
D. Qualitative Analysis
To addressRQ_4, we present qualitative comparisons across
different domains. Fig. 10 shows results for Qwen2.5-VL [51]
and LLaV A-NeXT [52] with and without CgRAG on eight
cases spanning (a-b) commonsense, (c-d) animal, (e) geography,
(f) architecture, (g) history, and (h) art. In commonsense
cases (a-b) and simple animal case (c), all methods produce
the correct answers. However, across the cases (d-h) that
require deep domain expertise, CgRAG enables backbone
MLLMs to consistently produce correct answers. In contrast,
the plain MLLMs are more prone to superficial cues and noisy
knowledge integration: in cases (d-e), they rely on shallow
textual or visual triggers that lead to incorrect inferences(e.g., “country” inducing “America”, or recognizing a bird
and answering “sparrow”), which hinders evidence-seeking
retrieval; in cases (f-h), they produce confused answers due to
crude integration of retrieved content (e.g., mismatching “Philip
Johnson” with the building in the picture despite he was a core
figure in the American architectural, or misunderstanding the
“inventor” as “reformer” of the steam locomotive, or confusing
the famous work of three Baroque and Rococo artists). These
cases illustrate that CgRAG improves fine-grained knowledge
integration and supports more reliable inference across diverse
domains.
V. CONCLUSION
In this work, we propose CoVQD- guided RAG (CgRAG) to
improve MLLM-based VQA on KBVQA tasks. CgRAG unifies
Chain-of-Thought (CoT) and Visual Question Decomposition
(VQD) into a multi-grained retrieval strategy and introduces
Chain-of-VQD (CoVQD) to extract fine-grained multimodal
signals from question-image (QI) pairs that guide a precise
retrieval-augmented generation process. To effectively integrate
the retrieved knowledge within this stepwise guidance, we de-
sign a flexible prompting strategy to support MLLM inference.
Moreover, to enhance the MLLM’s analytical capability for
QI pairs, we develop a fine-tuning paradigm, termed logical
implication Direct Preference Optimization (liDPO), within
the CgRAG framework. The experiment conducted on various

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 11
relevant datasets demonstrated the effectiveness of the proposed
method.
REFERENCES
[1]L. Yang, Z. Xiao, W. Huang, and X. Zhong, “StoryLLaV A: enhancing
visual storytelling with multi-modal large language models,” inProc. Int.
Conf. Comput. Linguistics, pp. 3936–3951, 2025.
[2]X. Zhong, Z. Li, S. Chen, K. Jiang, C. Chen, and M. Ye, “Refined
semantic enhancement towards frequency diffusion for video captioning,”
inProc. AAAI Conf. Artif. Intell., pp. 3724–3732, 2023.
[3]S. Chen, X. Zhong, Y . Zhang, L. Zhu, P. Li, X. Yang, and
B. Sheng, “Action-aware linguistic skeleton optimization network for
non-autoregressive video captioning,”ACM Trans. Multimedia Comput.
Commun. Appl., vol. 20, no. 10, pp. 326:1–326:24, 2024.
[4]S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and
D. Parikh, “VQA: visual question answering,” inProc. IEEE/CVF Int.
Conf. Comput. Vis., pp. 2425–2433, 2015.
[5]Y . Goyal, T. Khot, A. Agrawal, D. Summers-Stay, D. Batra, and D. Parikh,
“Making the V in VQA matter: Elevating the role of image understanding
in visual question answering,” inProc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit., pp. 6904–6913, 2017.
[6]S. Shah, A. Mishra, N. Yadati, and P. P. Talukdar, “KVQA: knowledge-
aware visual question answering,” inProc. AAAI Conf. Artif. Intell.,
pp. 8876–8884, 2019.
[7]F. Cocchi, N. Moratelli, M. Cornia, L. Baraldi, and R. Cucchiara,
“Augmenting multimodal LLMs with self-reflective tokens for knowledge-
based visual question answering,” inProc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit., pp. 9199–9209, 2025.
[8]Z. Zhang, Y . Wu, Y . Luo, and N. Tang, “Fine-grained retrieval-
augmented generation for visual question answering,”arXiv preprint
arXiv:2502.20964, 2025.
[9]Z. Ling, Z. Guo, Y . Huang, Y . An, S. Xiao, J. Lan, X. Zhu, and
B. Zheng, “MMKB-RAG: a multi-modal knowledge-based retrieval-
augmented generation framework,”arXiv preprint arXiv:2504.10074,
2025.
[10] Y . Hong, J. Gu, Q. Yang, L. Fan, Y . Wu, Y . Wang, K. Ding, S. Xiang,
and J. Ye, “Knowledge-based visual question answer with multimodal
processing, retrieval and filtering,”arXiv preprint arXiv:2510.14605,
2025.
[11] H. You, R. Sun, Z. Wang, L. Chen, G. Wang, H. A. Ayyubi, K.-W.
Chang, and S.-F. Chang, “Idealgpt: Iteratively decomposing vision and
language reasoning via large language models,” inFindings EMNLP,
pp. 11289–11303, 2023.
[12] J. Qi, Z. Xu, Y . Shen, M. Liu, D. Jin, Q. Wang, and L. Huang, “The art of
SOCRATIC questioning: Recursive thinking with large language models,”
inProc. Conf. Empir. Methods Nat. Lang. Process., pp. 4177–4199, 2023.
[13] Y . Tian, T. Ma, L. Xie, and Q. Ye, “Chatterbox: Multimodal referring
and grounding with chain-of-questions,” inProc. AAAI Conf. Artif. Intell.,
pp. 7401–7409, 2025.
[14] M. Bigverdi, Z. Luo, C.-Y . Hsieh, E. Shen, D. Chen, L. G. Shapiro, and
R. Krishna, “Perception tokens enhance visual reasoning in multimodal
language models,” inProc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., pp. 3836–3845, 2025.
[15] D. Caffagni, F. Cocchi, N. Moratelli, S. Sarto, M. Cornia, L. Baraldi, and
R. Cucchiara, “Wiki-llava: Hierarchical retrieval-augmented generation
for multimodal LLMs,” inProc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. Workshops, pp. 1818–1826, 2024.
[16] J. Qi, Z. Xu, R. Shao, Y . Chen, D. Jin, Y . Cheng, Q. Wang, and L. Huang,
“Rora-vlm: Robust retrieval-augmented vision language models,”arXiv
preprint arXiv:2410.08876, 2024.
[17] Y . Yan and W. Xie, “Echosight: Advancing visual-language models with
wiki knowledge,” inFindings EMNLP, pp. 1538–1551, 2024.
[18] V . Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wier-
stra, and M. A. Riedmiller, “Playing atari with deep reinforcement
learning,”arXiv preprint arXiv:1312.5602, 2013.
[19] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in LLMs
via reinforcement learning,”arXiv preprint arXiv:2501.12948, 2025.
[20] Y . Yang, X. He, H. Pan, X. Jiang, Y . Deng, X. Yang, H. Lu, D. Yin,
F. Rao, M. Zhu, B. Zhang, and W. Chen, “R1-onevision: Advancing
generalized multimodal reasoning through cross-modal formalization,”
arXiv preprint arXiv:2503.10615, 2025.
[21] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and
C. Finn, “Direct preference optimization: Your language model is secretly
a reward model,” inProc. Adv. Neural Inf. Process. Syst., 2023.
[22] D. Xue, S. Qian, and C. Xu, “Variational causal inference network for
explanatory visual question answering,” inProc. IEEE/CVF Int. Conf.
Comput. Vis., pp. 2515–2525, 2023.[23] C. Lai, S. Song, S. Meng, J. Li, S. Yan, and G. Hu, “Towards more faithful
natural language explanation using multi-level contrastive learning in
VQA,” inProc. AAAI Conf. Artif. Intell., pp. 2849–2857, 2024.
[24] D. Xue, S. Qian, and C. Xu, “Few-shot multimodal explanation for visual
question answering,” inProc. ACM Int. Conf. Multimedia, pp. 1875–1884,
2024.
[25] Z. Yang, Z. Gan, J. Wang, X. Hu, Y . Lu, Z. Liu, and L. Wang, “An
empirical study of GPT-3 for few-shot knowledge-based VQA,” inProc.
AAAI Conf. Artif. Intell., pp. 3081–3089, 2022.
[26] Z. Khan, V . K. B. G, S. Schulter, M. Chandraker, and Y . Fu, “Exploring
question decomposition for zero-shot VQA,” inProc. Int. Conf. Neural
Inf. Process. Syst., 2023.
[27] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-
V oss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler,
J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever,
and D. Amodei, “Language models are few-shot learners,” inProc. Adv.
Neural Inf. Process. Syst., 2020.
[28] H. Liu, C. Li, Y . Li, and Y . J. Lee, “Improved baselines with visual
instruction tuning,” inProc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., pp. 26286–26296, 2024.
[29] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and
J. Zhou, “Qwen-vl: A versatile vision-language model for understanding,
localization, text reading, and beyond,”arXiv preprint arXiv:2308.12966,
2023.
[30] H. Zhang, J. Liu, Z. Han, S. Chen, B. He, V . Tresp, Z. Xu, and J. Gu,
“Visual question decomposition on multimodal large language models,”
inFindings EMNLP, pp. 1926–1949, 2024.
[31] S. Tascon-Morales, P. Márquez-Neila, and R. Sznitman, “Logical impli-
cations for visual question answering consistency,” inProc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., pp. 6725–6735, 2023.
[32] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: pre-training
of deep bidirectional transformers for language understanding,” inProc.
Conf. North Amer. Chapter Assoc. Comput. Linguistics: Hum. Lang.
Technol., pp. 4171–4186, 2019.
[33] L. Gao, J. Schulman, and J. Hilton, “Scaling laws for reward model
overoptimization,” inProc. Int. Conf. Mach. Learn., pp. 10835–10866,
2023.
[34] R. A. Bradley and M. E. Terry, “Rank analysis of incomplete block
designs: I. the method of paired comparisons,”Biometrika, vol. 39,
no. 3/4, pp. 324–345, 1952.
[35] F. Wang, W. Zhou, J. Y . Huang, N. Xu, S. Zhang, H. Poon, and
M. Chen, “mDPO: conditional preference optimization for multimodal
large language models,” inProc. Conf. Empir. Methods Nat. Lang.
Process., pp. 8078–8088, 2024.
[36] Y . He, H. Sun, P. Ren, J. Wang, H. Wang, Q. Qi, Z. Zhuang, and
J. Wang, “Evaluating and mitigating object hallucination in large vision-
language models: Can they still see removed objects?,” inProc. Conf.
North Amer. Chapter Assoc. Comput. Linguistics: Hum. Lang. Technol.,
pp. 6841–6858, 2025.
[37] P. Zhang, X. Li, X. Hu, J. Yang, L. Zhang, L. Wang, Y . Choi, and J. Gao,
“Vinvl: Revisiting visual representations in vision-language models,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 5579–5588,
2021.
[38] J. Li, D. Li, S. Savarese, and S. C. H. Hoi, “BLIP-2: bootstrapping
language-image pre-training with frozen image encoders and large
language models,” inProc. Int. Conf. Mach. Learn., pp. 19730–19742,
2023.
[39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervision,”
inProc. Int. Conf. Mach. Learn., pp. 8748–8763, 2021.
[40] J. Li, D. Li, C. Xiong, and S. C. H. Hoi, “BLIP: bootstrapping
language-image pre-training for unified vision-language understanding
and generation,” inProc. Int. Conf. Mach. Learn., vol. 162, pp. 12888–
12900, 2022.
[41] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi, “OK-VQA: a
visual question answering benchmark requiring external knowledge,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 3195–3204,
2019.
[42] T. Mensink, J. R. R. Uijlings, L. Castrejón, A. Goel, F. Cadar, H. Zhou,
F. Sha, A. Araújo, and V . Ferrari, “Encyclopedic VQA: visual questions
about detailed properties of fine-grained categories,” inProc. IEEE/CVF
Int. Conf. Comput. Vis., pp. 3090–3101, 2023.

IEEE TRANSACTIONS ON IMAGE PROCESSING, 2026 12
[43] Y . Chen, H. Hu, Y . Luan, H. Sun, S. Changpinyo, A. Ritter, and M.-
W. Chang, “Can pre-trained vision and language models answer visual
information-seeking questions?,” inProc. Conf. Empir. Methods Nat.
Lang. Process., pp. 14948–14968, 2023.
[44] H. Hu, Y . Luan, Y . Chen, U. Khandelwal, M. Joshi, K. Lee, K. Toutanova,
and M.-W. Chang, “Open-domain visual entity recognition: Towards
recognizing millions of wikipedia entities,” inProc. IEEE/CVF Int. Conf.
Comput. Vis., pp. 12031–12041, 2023.
[45] S. Chen and Q. Zhao, “REX: reasoning-aware and grounded explanation,”
inProc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 15565–
15574, 2022.
[46] D. A. Hudson and C. D. Manning, “GQA: a new dataset for real-
world visual reasoning and compositional question answering,” inProc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit., pp. 6700–6709, 2019.
[47] C. Kervadec, G. Antipov, M. Baccouche, and C. Wolf, “Roses are red,
violets are blue... but should VQA expect them to?,” inProc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., pp. 2776–2785, 2021.
[48] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung,
and S. C. H. Hoi, “Instructblip: Towards general-purpose vision-language
models with instruction tuning,” inProc. Adv. Neural Inf. Process. Syst.,
2023.
[49] OpenAI, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L.
Aleman, D. Almeida, J. Altenschmidt, S. Altman,et al., “Gpt-4 technical
report,”arXiv preprint arXiv:2303.08774, 2024.
[50] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang,
W. Ge, Y . Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou,
and J. Lin, “Qwen2-VL: enhancing vision-language model’s perception
of the world at any resolution,”arXiv preprint arXiv:2409.12191, 2024.
[51] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang,
S. Wang, J. Tang,et al., “Qwen2.5-vl technical report,”arXiv preprintarXiv:2502.13923, 2025.
[52] F. Li, R. Zhang, H. Zhang, Y . Zhang, B. Li, W. Li, Z. Ma, and C. Li,
“LLaV A-NeXT-Interleave: tackling multi-image, video, and 3D in large
multimodal models,”arXiv preprint arXiv:2407.07895, 2024.
[53] P. Lerner, O. Ferret, and C. Guinaudeau, “Cross-modal retrieval for
knowledge-based visual question answering,” inProc. Eur. Conf. Inf.
Retr., pp. 421–438, 2024.
[54] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y . Duan,
W. Su, J. Shao,et al., “Internvl3: Exploring advanced training and
test-time recipes for open-source multimodal models,”arXiv preprint
arXiv:2504.10479, 2025.
[55] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning, “A large
annotated corpus for learning natural language inference,” inProc. Conf.
Empir. Methods Nat. Lang. Process., pp. 632–642, 2015.
[56] R. R. Selvaraju, P. Tendulkar, D. Parikh, E. Horvitz, M. T. Ribeiro,
B. Nushi, and E. Kamar, “Squinting at VQA models: Introspecting VQA
models with sub-questions,” inProc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit., pp. 10000–10008, 2020.
[57] Q. Li, Q. Tao, S. R. Joty, J. Cai, and J. Luo, “VQA-E: explaining,
elaborating, and enhancing your answers for visual questions,” inProc.
Eur. Conf. Comput. Vis., vol. 11211, pp. 570–586, 2018.
[58] J. Wu and R. J. Mooney, “Faithful multimodal explanation for visual
question answering,” inProc. ACL Workshop BlackboxNLP: Analyzing
and Interpreting Neural Networks for NLP, pp. 103–112, 2019.
[59] D. Xue, S. Qian, and C. Xu, “Variational causal inference network for
explanatory visual question answering,” inProc. IEEE/CVF Int. Conf.
Comput. Vis., pp. 2515–2525, 2023.
[60] K. Li, G. V osselman, and M. Y . Yang, “Multimodal rationales for
explainable visual question answering,” inProc. IEEE/CVF Conf. Comput.
Vis. Pattern Recognit. Workshops, pp. 191–201, 2025.
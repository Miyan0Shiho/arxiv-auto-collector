# M2Note: Continual Evolution of Vision Language Models via Mistake Notebook Learning

**Authors**: Haiwen Li, Jing Tang, Rui Chen, Lei Sun, Xiangxiang Chu

**Published**: 2026-07-01 09:30:10

**PDF URL**: [https://arxiv.org/pdf/2607.00685v1](https://arxiv.org/pdf/2607.00685v1)

## Abstract
Vision Language Models (VLMs) have demonstrated remarkable capabilities in multimodal reasoning tasks, yet they still suffer from recurring failures, such as skipping key visual checks, misapplying domain rules, and hallucinating unsupported concepts. Most existing solutions rely on supervised fine-tuning (SFT) and reinforcement learning (RL), which are expensive to iterate and can be brittle under distribution shift. To this end, we propose Multimodal Mistake Notebook Learning (M2Note), a training-free continual evolution framework that externalizes learning into an editable memory. M2Note transforms failed trajectories into compact subject-guidance notes: the subject summarizes the underlying domain and concept, while the guidance provides actionable verification steps that can be reused in future inference. At test time, M2Note retrieves relevant notes via multimodal retrieval-augmented generation (RAG) and appends them to the model context, steering reasoning away from previously observed pitfalls. To stabilize continual evolution, we adopt batch-level post-verification with rollback, which commits notebook edits only if they improve performance on the same batch, reducing noisy updates and preventing regressions. M2Note supports both self-evolving, where the same VLM acts as solver and supervisor, and cross-model evolving, where a stronger supervisor guides a weaker solver, enabling capability transfer without weight updates. Experiments on six multimodal reasoning benchmarks show consistent improvements across domains and backbones, while achieving strong cost and sample efficiency and remaining complementary to Chain-of-Thought (CoT) prompting.

## Full Text


<!-- PDF content starts -->

M2Note: Continual Evolution of Vision Language Models via Mistake
Notebook Learning
Haiwen Li*Jing Tang†Rui Chen Lei Sun Xiangxiang Chu
AMAP, Alibaba Group
Abstract
Vision Language Models (VLMs) have demon-
strated remarkable capabilities in multimodal
reasoning tasks, yet they still suffer from re-
curring failures, such as skipping key visual
checks, misapplying domain rules, and hallu-
cinating unsupported concepts. Most exist-
ing solutions rely on supervised fine-tuning
(SFT) and reinforcement learning (RL), which
are expensive to iterate and can be brittle un-
der distribution shift. To this end, we pro-
poseMultimodalMistakeNotebook Learning
(M2Note), a training-free continual evolution
framework that externalizes learning into an ed-
itable memory. M2Note transforms failed tra-
jectories into compact subject–guidance notes:
the subject summarizes the underlying domain
and concept, while the guidance provides ac-
tionable verification steps that can be reused
in future inference. At test time, M2Note re-
trieves relevant notes via multimodal retrieval-
augmented generation (RAG) and appends
them to the model context, steering reasoning
away from previously observed pitfalls. To sta-
bilize continual evolution, we adopt batch-level
post-verification with rollback, which commits
notebook edits only if they improve perfor-
mance on the same batch, reducing noisy up-
dates and preventing regressions. M2Note sup-
ports bothself-evolving, where the same VLM
acts as solver and supervisor, andcross-model
evolving, where a stronger supervisor guides
a weaker solver, enabling capability transfer
without weight updates. Experiments on six
multimodal reasoning benchmarks show con-
sistent improvements across domains and back-
bones, while achieving strong cost and sam-
ple efficiency and remaining complementary to
Chain-of-Thought (CoT) prompting.
*Work done during the internship at AMAP, Alibaba.
†Project Lead.1 Introduction
Vision Language Models (VLMs) (Bordes et al.,
2024; Hurst et al., 2024; Bai et al., 2025; Guo et al.,
2025; Liu et al., 2023) have become a general in-
terface for multimodal tasks, including STEM rea-
soning (Yue et al., 2024; Lu et al., 2023; Wang
et al., 2024; Qiao et al., 2025), chart and diagram
understanding (Masry et al., 2022; Kembhavi et al.,
2016), document OCR (Liu et al., 2024c; Yang
et al., 2025), visual question answering (Liu et al.,
2024b; Chen et al., 2024; xAI), and video under-
standing (Li et al., 2024; Zhao et al., 2025; Yuan
et al., 2026). Despite strong progress, VLMs in re-
alistic settings still exhibit repeatable failure modes,
such as missing key visual evidence or over-relying
on superficial cues. Correcting such behaviors effi-
ciently, without sacrificing robustness or requiring
costly retraining, remains a practical challenge.
The dominant adaptation paradigm is parameter-
based post-training, such as supervised fine-tuning
(SFT) (Liu et al., 2023; Wei et al., 2022a) and re-
inforcement learning (RL) (Schulman et al., 2017;
Rafailov et al., 2023; Shao et al., 2024; Yu et al.,
2025). Although effective, these methods are ex-
pensive, slow to iterate, and prone to regression.
More importantly, once model weights are updated,
test-time behavior becomes fixed, making contin-
ual improvement difficult in dynamic environments.
This motivates growing interest in training-free
adaptation through in-context steering.
Existing training-free approaches mainly fall
into two categories: ❶prompt optimization meth-
ods (Zhou et al., 2022; Yang et al., 2023; Pryzant
et al., 2023) refine global instructions, but often
provide advice that is too coarse to target recurring
errors across diverse multimodal tasks; ❷memory-
based methods (Zhao et al., 2024; Zhang et al.,
2024a; Shinn et al., 2023; Zhang et al., 2025a)
store instance-level experiences for retrieval, but
they often lack abstraction, leading to redundant
1arXiv:2607.00685v1  [cs.MA]  1 Jul 2026

Question: How many molecules of the sweetener saccharin can be prepared from 30 $C$ atoms, 25 $H$ atoms, 12 $O$ atoms, 8 $S$ atoms, and 14 $N$ atoms? <image>
Qwen3-VL-8B-Instruct: (A) 3EvolvingviaM2Note: (B) 4
I will continuously summarize the mistakes, and update my memory.Raw reasoning → Recall on experience1. BaselineResponse:Do not verify the molecular formula element by element ...2. Summarize Mistake: Failed to confirm the molecular formula and take the minimum of “available vs. required” across all elements ...3. Retrieve Guidance: Compute “available vs. required” for each element, and answer with the minimum (the limiting element) …
Reasoning Benchmarks (Qwen3-VL-8B)
RL (8B)Ours (271B)Ours (271B)RL (8B)Figure 1:Overview of M2Note. (Left) The system updates an external mistake notebook from incorrect responses
and retrieves task-relevant guidance at inference time to refine reasoning, for instance, limiting-reagent counting for
saccharin. (Right) Accuracy gains on different benchmarks (e.g., MMMU (Yue et al., 2024), MathVista (Lu et al.,
2023), and AI2D (Kembhavi et al., 2016)) via M²Note, together with cost- and sample-efficiency comparisons.
memories and limited generalization beyond super-
ficially similar cases. These limitations are espe-
cially pronounced in multimodal settings, where
visual inputs are highly diverse and sparse.
To address this, we proposeMultimodalMistake
Notebook Learning (M2Note), a tuning-free frame-
work, which extends the learning-from-mistakes
paradigm (Zhang et al., 2024a; Su et al., 2025)
to VLMs. M2Note maintains an external mistake
notebook that stores structuredsubject-guidance
pairs and retrieves relevant notes via multimodal
embeddings (Li et al., 2026; Zhang et al., 2024b)
to augment the model context. Given a multimodal
query, the VLM first recalls relevant notes and pro-
duces an answer. If the answer is incorrect, a re-
flective supervisor summarizes the failure into a
high-level subject and concise actionable guidance,
then writes or merges the resulting note into the
notebook. In this way, recurring errors are distilled
into compact, reusable experience, shifting adap-
tation from model parameter updates to semantic
updates of an external notebook.
Beyond being training-free, M2Note is designed
as a closed-loop evolution protocol:generate →
reflect →update →verify. Whenever notebook
updates are proposed, the system re-evaluates the
same batch with the updated notebook and accepts
the change only if performance improves; other-
wise, it rolls back. This conservativeaccept-if-
improvesmechanism stabilizes continual evolution
and suppresses noisy updates. The framework sup-
ports bothself-evolving, where the same VLM acts
as solver and supervisor, andcross-model evolving,
where a stronger VLM supervises a weaker one to
transfer knowledge without any parameter updates.Experiments on six multimodal reasoning bench-
marks show that M2Note consistently improves
VLM performance. Compared with RL-based self-
evolving methods such as VisPlay (He et al., 2025),
EvoLMM (Thawakar et al., 2025), and Vision-
Zero (Wang et al., 2025a), M2Note achieves strong
gains with substantially lower training cost and
fewer samples, since it evolves only a compact ex-
ternal notebook rather than model parameters. We
introduce the following technical components:
•We present M2Note, a multimodal mistake
notebook learning framework that improves
VLM reasoning by storing and retrieving
subject-level guidance in an external memory.
•We propose a stable closed-loop evolution pro-
tocol with batch-level verification, supporting
both self-evolving and cross-model evolving
settings while remaining tuning-free.
•We deliver consistent gains across six multi-
modal reasoning benchmarks, with strong cost
and sample efficiency compared with SFT-
and RL-based methods.
2 Related Work
VLMs and Multimodal Reasoning.Multimodal
large language models (MLLMs) (Radford et al.,
2021; Bai et al., 2025; Liu et al., 2024a; Wang
et al., 2025b; Hurst et al., 2024; Comanici et al.,
2025) extend LLMs with visual perception and
have achieved strong performance on multimodal
understanding and reasoning tasks. Benchmarks
such as MMMU (Yue et al., 2024), MathVista (Lu
et al., 2023), AI2D (Kembhavi et al., 2016),
2

ChartQA (Masry et al., 2022), MMStar (Chen et al.,
2024), and RealworldQA (xAI) evaluate challeng-
ing abilities including STEM reasoning, diagram
understanding, and real-world visual reasoning.
Self-Evolving and Training-Free Adaptation.
Recent self-evolving methods improve models via
closed-loop interaction, often by assigning co-
evolving roles (e.g., a challenger and a solver)
to the same base model. Representative exam-
ples (Huang et al., 2025; Wang et al., 2025a; He
et al., 2025; Xu et al., 2026; Jia et al., 2026) use
self-play to generate increasingly informative train-
ing signals and typically internalize the resulting
improvements through parameter updates, such as
supervised fine-tuning (Wei et al., 2022a; Zhang
et al., 2026) and reinforcement learning (Schul-
man et al., 2017; Shao et al., 2024; Dai et al.,
2026; Ji et al., 2025). While effective, these meth-
ods are costly and can be brittle under distribu-
tion shift (Chen et al., 2023; Zeng et al., 2024;
Zhai et al., 2025). Training-free alternatives in-
stead adapt models through inference-time con-
text, primarily including prompt optimization (Wei
et al., 2022b; Pryzant et al., 2023; Zhou et al.,
2022; Yang et al., 2023) and memory-augmented
inference (Shinn et al., 2023; Zhao et al., 2024;
Zhou et al., 2025; Zhang et al., 2025a; Cai et al.,
2025; Ouyang et al., 2025). Within the latter,
learning-from-mistakes methods (Madaan et al.,
2023; Zhang et al., 2024a; Zhao et al., 2024; Su
et al., 2025) convert failures into reusable vexpe-
rience for future inference. Our work extends this
paradigm to VLMs by distilling multimodal fail-
ures into compact subject-level guidance and up-
dating memory with batch-level verification.
3 Methodology
3.1 Overview
We proposeMultimodalMistakeNotebook Learn-
ing (M2Note), a training-free framework that im-
proves VLMs by maintaining an external mistake
notebook. Unlike prior context-optimization meth-
ods mainly developed for text-only settings (Su
et al., 2025; Zheng et al.; Wang et al., 2025c; Zhang
et al., 2024a; Zhou et al., 2025), M2Note extends
thelearning-from-mistakes(Su et al., 2025; Zhang
et al., 2024a) paradigm to multimodal tasks.
As illustrated in Figure 2, M2Note involves two
roles: aTuning Model( πθ), which solves multi-
modal problems with retrieved notebook guidance,
and aTuner Model( πtuner ), which analyzes theTuning Model’s errors and writes reusable guid-
ance back into the notebook. The notebook is ini-
tialized as an external memory Mthat stores struc-
tured triplets (s, g, e) , where sdenotes the subject,
summarizing the underlying task and domain, g
denotes the corresponding guidance, i.e., concise
recommendations associated with that subject, and
e∈Rdrepresents the embedding produced by a
multimodal embedding model for retrieval. Exam-
ples of the notes are provided in Appendix B.
The framework runs in a closed loop with three
steps:❶the Tuning Model retrieves relevant notes
fromMand generates baseline responses; ❷the
Tuner Model summarizes failed cases into new
subject-guidance notes, and decides whether to add
or merge them; ❸the system reruns the same batch
with the updated memory and accepts the update
only if the new memory improves batch-level per-
formance; otherwise, it rolls back.
3.2 Problem Formulation
M2Note can be formulated as a context optimiza-
tion problem: instead of updating the parameters
ofπθ, we refine only the external memory to max-
imize the expected reward, ensuring that the re-
trieved guidance provides effective assistance. For-
mally, let Ddenote a data distribution over mul-
timodal queries (q, i) , optionally paired with y
(when the training split is available), where qde-
notes the text question, ithe associated image, and
ythe optional ground-truth answer when available.
We seek an optimal memoryM∗:
M∗= arg max
ME(q,i)∼Dh
R 
πθ(z)i
,
wherez= (q, i)⊕Ret((q, i),M),(1)
where⊕is context concatenation, and Ret(·,·) rep-
resents retrieving guidance from the notebook. We
consider two instantiations of the reward function
R(·):❶Supervised setting.If ground-truth labels
yare available, the model response is first parsed
into a final prediction, and R(·) is instantiated as a
binary reward by comparing the parsed prediction
against the ground truth, i.e., it returns 1if they
match and 0otherwise; ❷Label-free setting.If
labels are unavailable, R(·) adopts an LLM-as-a-
Judge (Gu et al., 2024), which evaluates whether
the response is correct given the input (and task
instruction if available), and outputs a binary judg-
ment as reward. In both cases, the optimization pro-
cess is delegated to πtuner , which identifies failures,
summarizes guidance, and updates the memory.
3

Baseline vs. Updated(Batch-Level)
Leighton was a keen traveler and during his trips he would make small landscape studies of the places he visited. Which city is shown in <image>?Answer: Damascus(i)Multimodal RAG-based Guidance Retrieval(ii)Batch-Level Memory Refinement
①
Subject&Guidance: Art History & Architectural Identification·Entity-Driven Research: Use the artist's name to narrow down the search space …·Architectural Fingerprint: Analyze the specific structure …
Tuning Model
Mistake Notebook ℳ!•Subject 1•Guidance
•Subject 2•Guidance
…ℬ~{(𝑞",𝑖")}"#$%RetrieveGuidance
Multimodal Embedding {(𝑠",𝑔")}"#$%
Tuner Model
Baseline GenerationFailed responses: -ℛ!Correct responses: ℛ!
Subject & Guidance: Art History and Architectural Identification·Entity-Driven Research: Use the artist's name to narrow down the search space …·Architectural Fingerprint: Analyze the specific structure …
Tuning Model
②Subject “𝑠”Question “𝑞,𝑖”Error “𝜋!(𝑧)”SynthesizeNew Guidance for Subject
Merge Subject & Guidance
Add New GuidanceUpdated Mistake Notebook to ℳ!&
If baseline wins, roll back, ℳ"#$=ℳ"Else, updated wins, enforce ℳ"#$=ℳ"%
I can learn from mistakes and summarize guidance into my memory so that I won’t make similar mistakes in the future.Updated GenerationFailed responses: -ℛ!&Correct responses: ℛ!&
Question: Leighton was a keen traveler and during his trips he would make small landscape studies of the places. Which city is shown in <image>?Answer: Damascus (Optional)Figure 2:The M2Note evolving protocol.M2Note improves a VLM through a closed loop with two roles: a
Tuning Model for solving multimodal queries with retrieved notebook guidance, and a Tuner Model for analyzing
mistakes and refining the notebook. The process consists of (i)Multimodal RAG-based Guidance Retrieval,
where the Tuning Model retrieves relevant subject-guidance notes from memory to generate responses, and (ii)
Batch-level Memory Refinement, where the Tuner Model summarizes failures into new or merged notes, after
which the update is verified and accepted only if it improves batch-level performance.
3.3 The M2Note Evolution Protocol
As illustrated in Figure 2, M2Note follows a closed-
loop protocol that iteratively retrieves guidance, up-
dates memory from mistakes, and verifies whether
the update should be kept.
Multimodal RAG-based Guidance Retrieval.
The Tuning Model πθuses the current memory Mi
in a multimodal RAG manner (Lewis et al., 2020;
Mei et al., 2025). For the i-th iteration, given an in-
put batch Bi={(q k, tk)}N
k=1, each query (qk, tk)
is fed into a multimodal embedding model Eto ob-
tain an embedding ek=E(q k, ik)∈Rd. The sys-
tem then computes the cosine similarity between
ekand each note embedding in memory, ranks all
notes by cosine similarity, and returns the top- K
results (the following showsK= 1):
(sk, gk) = arg max
(s,g)⟨ek, e⟩
∥ek∥∥e∥(s, g, e)∈ M i,
⟨ek, e⟩
∥ek∥∥e∥> α
(2)
where ⟨ek, e⟩denotes the inner product; ∥·∥de-
notes the ℓ2normalization; and αcontrols the rele-
vance of the retrieved content.
The retrieved notes are concatenated as advisory
context to the VLM input, i.e., zk= (q k, ik)⊕
(sk, gk). In addition, the Tuning Model is in-
structed to incorporate the extra context critically
rather than follow it blindly, which helps to reduce
hallucinations (Appendix A.2, Table 5 and Fig-
ure 5). The initial response, denoted as πθ(zk), isfed into the reward function R(·). Aggregating and
partitioning the responses within the batch yields
two mutually exclusive sets: correct responses are
collected into Ri={k|R(π θ(zk)) = 1} , while
failed trajectories form ˜Ri={k|R(π θ(zk)) = 0} .
The batch accuracy is then computed as the perfor-
mance baseline for the current iteration.
Batch-Level Memory Refinement. Given the
failure set ˜Ri, the Tuner Model πtunersummarizes
reusable guidance and proposes notebook updates.
An update is accepted only if it improves batch-
level performance; otherwise, it is rolled back.
(1)Subject Extraction σ.For each failed sample
index k∈ ˜Ri, the system infers asubjectthat
summarizes the underlying concept of the query.
Concretely, as shown below, we apply a prompted
operatorσ(·)implemented byπ tuner:
s′
k=π tuner(σ(q k, ik)), k∈ ˜Ri.(3)
The subject is expected to be domain-aware (e.g.,
math, chemistry, or diagram understanding), spe-
cific enough to capture recurring error patterns, but
not tied to instance-specific details (e.g., exact enti-
ties or numbers).
(2)Guidance Synthesis ϕ.Given a failed tra-
jectory, πtunerfurther generates a guidance that is
directly actionable at inference time. The guidance
is distilled from the failed response πθ(zk)and the
available feedback, abstracting the cause of failure
into verification-oriented steps:
g′
k=π tuner(ϕ((q k, ik), πθ(zk))), k∈ ˜Ri,(4)
4

where ϕ(·) is a prompted “mistake-to-guidance”
operator. In practice, g′
kis constrained to be con-
cise, structured, and oriented toward checks and
invariants (e.g., “use the artist’s name to narrow
down the search space” in Figure 2).
(3)Subject Merging and Memory Update µ.To
avoid fragmentation, M2Note optionally merges
new notes with existing ones. For each candidate
note(s′
k, g′
k), the tuner predicts an edit action:
ak=π tuner(µ(s′
k, sk))∈ {add,merge}.(5)
Ifak=Merge ,πtunersynthesizes the merged sub-
jects∗
kand the merged guidance g∗
k; otherwise,
they are set to s′
kandg′
k, respectively. For each
newly created or merged note, we compute its em-
bedding as e∗
k=E(s∗
k⊕g∗
k). The updated mem-
oryM′
iis obtained by incorporating all entries
{(s∗
k, g∗
k, e∗
k)}N
k=1derived from this batch of data.
(4)Batch-Level Post-Verification.After propos-
ingM′
i, we rerun the same batch using the updated
memory. The batch-level accuracy before and after
the update is computed by the formula:
Acc(B i|M) =1
NNX
k=1R(π θ(zk|M)).(6)
We accept the update if it improves performance:
Mi+1=(
M′
i,if Acc(B i|M′
i)≥Acc(B i|Mi),
Mi,otherwise (rollback).
(7)
This “accept-if-improves” rule enforces monotonic
non-degradation at the batch level, making note-
book evolution robust to occasional low-quality
reflections. It also acts as an implicit regularizer
that curbs uncontrolled notebook growth, since
only useful notes survive. The pseudo-code of the
overall procedure is shown in Algorithm 1, and all
prompt templates are provided in Appendix A.
4 Experiments
Experimental Settings.We adopt two experimen-
tal settings: ❶Supervised manner.We hold out
the test set for evaluation, and use data sampled
from the training split to update and save the exter-
nal mistake notebook, which is then used during
evaluation. ❷Test-time scaling (TTS).We directly
evaluate on the test set while continuously updating
the external memory online, allowing the model to
improve progressively as testing proceeds.Algorithm 1:M2Note evolving protocol
Input:Tuning Modelπ θ; Tuner Modelπ tuner ; Data
CorpusD; Reward functionR(·); RAG
top-Kandα
Output:Mistake NotebookM
1InitializeM 0← ∅;
2foreachB i={(q k, ik)}N
k=1sampled fromDdo
// (1) Retrieve guidance and run baseline
3foreachk∈ B ido
4(s k, gk)←Ret((q k, ik),M i;K, α);
5z k= (q k, ik)⊕(s k, gk);
6Acc base←1
NPN
k=1R(πθ(zk|Mi));
// (2) Propose notebook edits from failures
7Backup memoryM′
i← M i;
8foreachksuch thatR(π θ(zk|Mi) = 0do
9π tuner generates a subject-guidance pair
(s′
k, g′
k)and comparess kands′
kto make
an actiona k;
10ifa k=addthen
11M′
i← M′
i∪(s′
k, g′
k,E(s′
k⊕g′
k))
12else
13π tuner mergess k⊕s′
kintos∗
kand
gk⊕g′
kintog∗
k;
14M′
i← M′
i∪(s∗
k, g∗
k,E(s∗
k⊕g∗
k))
// (3) Batch-level post-verification with rollback
15Rerun the batchB iwithM′
iand compute
Accnew←1
NPN
k=1R(πθ(zk|M′
i));
16M i+1← M′
iifAcc new≥Acc base elseroll
backM i+1=M i;
Evaluation Datasets and Metrics.We evalu-
ate on six benchmarks covering three vision lan-
guage domains: STEM/Math (MMMU val(Yue
et al., 2024), MathVista (Lu et al., 2023)), Gen-
eral VQA (MMStar (Chen et al., 2024), Real-
worldQA (xAI)), and Document OCR (AI2D (Kem-
bhavi et al., 2016), ChartQA (Masry et al., 2022)).
MMMU and MathVista emphasize multi-discipline
reasoning and mathematical problem solving; MM-
Star and RealworldQA focus on open-world vi-
sual understanding; AI2D and ChartQA evaluate
diagram/chart comprehension and text-grounded
reasoning. Following prior work, we report accu-
racy (%) as the primary evaluation metric for all
datasets. Among them, MMMU and MathVista
are trained in a supervised manner, while the other
benchmarks are evaluated using test-time scaling
(TTS); please refer to Appendix C.2 for details.
Implementation Details.Our M2Note does not
require training model parameters; it only updates a
mistake notebook stored as external memory in the
JSONL format. We evaluate three VLMs: the open-
sourceQwen3-VL-8B-Instruct(Bai et al., 2025),
the closed-sourceQwen3-VL-Plus(Bai et al., 2025)
andGPT-5.4(Hurst et al., 2024). For multimodal
5

Table 1:Main results.We report accuracy and average score, together with the estimated training/API cost, the
average number of entries in the memory (Mem), and the average number of tokens in the guidance (Len). ‡
denotes results reproduced by us with a specific answer-parsing strategy, and the best performance is inbold.
Method Cost Mem LenSTEM Puzzle General VQA Document OCRAvg.
MMMU valMathVista MMStar RealworldQA AI2D ChartQA
Self-Evolving: Compare with SFT- and RL-based methods
Vision-Zero (Wang et al.)
Basically– – 58.8 72.6 65.2 68.5 84.5 86.3 72.65
EvoLMM (Thawakar et al.)
>$500– – 52.0 70.5 – – 83.4 86.7 –
iReasoner (Sunil et al.) – – 52.4 69.7 – – 83.9 85.8 –
VisPlay (He et al.) – – 54.9 68.2 65.1 69.0 – 86.2 –
Self-Evolving: Qwen3-VL-8B-Instruct (Open-source)
Vanilla†– – – 67.1 73.2 62.1 73.2 80.8 89.6 74.3
w/ DPE (Jia et al.)>$500– – 69.1(↑2.0 ) 76.2(↑3.0 ) 62.1 72.1(↓1.1 ) – 84.8(↓4.8 ) –
w/M2Note≈$350 26168.4(↑1.3 )77.3(↑4.1 ) 63.9(↑1.8 )75.4(↑2.2 ) 82.1(↑1.3 )89.9(↑0.3 )76.2(↑1.9 )
w/ CoT – – – 68.8 82.9 62.9 78.0 81.3 91.7 77.6
w/ CoT+M2Note≈$432 31569.4(↑0.6 )83.9(↑1.0 ) 64.9(↑2.0 )78.8(↑0.8 ) 81.7(↑0.4 )91.7 78.4(↑0.8 )
Self-Evolving: Qwen3-VL-Plus (Proprietary)
Vanilla†– – – 77.9 78.0 68.3 78.2 85.9 88.5 79.5
w/M2Note≈$728 46279.7(↑1.8 )80.1(↑2.1 ) 69.4(↑1.1 )79.7(↑1.5 ) 86.5(↑0.6 )89.5(↑1.0 )80.8(↑1.3 )
Self-Evolving: GPT-5.4 (Proprietary)
Vanilla†– – – 77.1 77.2 66.5 80.8 85.6 85.6 78.8
w/M2Note≈$728 46277.9(↑0.8 )78.0(↑0.8 ) 67.8(↑1.3 )81.8(↑1.0 ) 86.1(↑0.5 )85.8(↑0.2 )79.6(↑0.8 )
Cross-Model Evolving: Qwen3-VL-8B-Instruct (Tuning) and Qwen3-VL-Plus (Tuner)
Vanilla†– – – 67.1 73.2 62.1 73.2 80.8 89.6 74.3
w/M2Note≈$543 44568.9(↑1.8 )78.3(↑5.1 ) 64.0(↑1.9 )77.1(↑3.9 ) 81.0(↑0.2 )90.1(↑0.5 )76.6(↑2.3 )
RAG, we useQwen3-VL-Embedding(Li et al.,
2026) as the embedding model. In addition to these
models, all ablation variants are also implemented
through API requests. Unless otherwise specified,
we set the RAG top- KtoK= 1 and the RAG
threshold to α= 0.4 . The batch size for both
training and inference is 16. For supervised experi-
ments on MMMU and MathVista, we run 10and
20training steps, respectively. More details can be
found in Appendix C.1.
4.1 Main Results
Table 1 reports the main results of M2Note on six
multimodal reasoning benchmarks: ❶Consistent
gains across domains and backbones.M2Note
yields consistent improvements on different do-
mains, and the gains hold for both an open-source
backbone and a stronger proprietary backbone. ❷
Cost- and sample-efficiency.Compared to RL-
based self-evolving approaches, M2Note achieves
competitive or even better performance with dra-
matically lower cost via pure API requests. Under
the same backbone model (Qwen3-VL-8B-Instruct),
M2Note matches or slightly outperforms DPE (Jia
et al., 2026) while avoiding RL-style training over-
head.❸Multiple evolving modes.M2Note sup-
ports both self-evolving (single model acts as both
the tuning and tuner models) and cross-modelevolving (a stronger tuner model supervises a
weaker tuning model), enabling capability transfer
without any weight updates. ❹CoT compatibility.
M2Note can work together with Chain-of-Thought
(CoT) prompting (Brown et al., 2020; Wei et al.,
2022b), leading to further gains over either CoT
or M2Note alone. ❺Compact notebook with inter-
pretable scaling trends.We observe that stronger
backbones make fewer mistakes and require fewer
notebook entries (Mem), while a stronger tuner
(Qwen3-VL-Plus) tends to produce more detailed
guidance, resulting in longer length (Len).
4.2 Ablation Study
Sensitivity Analysis of Hyperparameters.Re-
sults in Figure 3a and 3b reveal two trends: ❶
M2Note is relatively insensitive to batch size.
Across a wide range (1–32), test accuracy remains
stable at around 76%–77.4%, suggesting that the
closed-loop update does not depend heavily on a
specific batch size. Moderate batch sizes (e.g., 8 or
16) perform best, likely because they support more
robust batch-level memory refinement and make
theverifystage more reliable. Meanwhile, note-
book size does not increase monotonically with
batch size (about 41–61 entries), indicating that
subject-level merging prevents memory from scal-
ing directly with the batch. ❷As training pro-
6

1 2 4 8 16 32
Batch Size01020304050607080Size of Mistake Notebook50
4152575161Mistake Notebook Size
Accuracy
0.700.720.740.760.780.80
Accuracy
76.6%76.9%
76.0%77.4% 77.3%76.9%(a) Batch size sensitivity.
Base Step 5 Step 10 Step 15 Step 20 Step 25 Step 30
Training Step01020304050607080Size of Mistake Notebook
153038505767Mistake Notebook Size
Test Acc.
0.700.720.740.760.780.80
Accuracy
73.2%75.0%75.4%
74.7%75.7%76.0%76.6% (b) Evolving as training progresses.
0.0 0.2 0.4 0.6 0.8
Retrieval Threshold ( )
020406080100120Notebook Size 8674
5232
0Notebook Size
Retrieval Threshold
Retrieval TopK
72747678
Accuracy (%)
K=1 K=2 K=3 K=4 K=5Retrieval TopK (c) Ablation study of the RAG settings.
Figure 3:Effects of different hyperparameters on performance and notebook size.Experiments are conducted
on MathVista withQwen3-VL-8B-Instruct.
gresses, the notebook gradually expands (15 to 67
entries), while accuracy improves from 73.2% to
76.6%. Although there is some fluctuation around
step 15, the overall upward trend suggests that con-
tinued evolution accumulates reusable guidance
and improves test generalization.
Multimodal RAG Settings.Figure 3c stud-
ies the effect of the retrieval threshold αand
the retrieval top- Kon MathVista (Qwen3-VL-8B-
Instruct). Increasing αmakes the system more
conservative, reducing the number of queries that
trigger note insertion; as fewer queries retrieve any
guidance, batch performance remains unchanged
before and after the update, thus shrinking the fi-
nal notebook size. In terms of accuracy, a moder-
ate threshold performs best: α=0.4 achieves the
highest accuracy, while overly strict retrieval (e.g.,
α=0.8 ) disables the notebook and degrades perfor-
mance. We also observe that the system is robust to
the retrieval top- K, with performance varying only
slightly. Overall, we prefer a smaller value ( K=1
by default) to reduce the risk of hallucinations.
Effectiveness of Batch-Level Post-Verification.
Table 2 examines the batch-level post-verification,
namelyaccept-if-improves. ForQwen3-VL-8B-
Instruct, enabling post verification yields the best
overall results, notably improving MathVista (73.2
→77.3) and MMMU val(67.1→68.4), showing
that filtering noisy updates stabilizes evolution and
makes the accumulated notes more reliable. With-
out post verification, performance becomes less
consistent (e.g., a drop on MMMU val), indicating
that blindly committing proposed notes may intro-
duce harmful or noisy guidance. For the stronger
Qwen3-VL-Plusbackbone, the effect is smaller but
still positive on average, suggesting that as the base
model makes fewer mistakes, the notebook receives
fewer high-impact updates, yet conservative verifi-Table 2:Effectiveness of batch-level post-verification.
We report results on three benchmarks across two back-
bones. Best results are marked inbold.
Method MathVista MMMU val MMStar
Self-Evolving: Qwen3-VL-8B-Instruct (Open-source)
Vanilla 73.2 67.1 62.1
w/o Batch-Level Post-Verification 76.2 65.7 62.7
w/ Batch-Level Post-Verification 77.3 68.4 63.9
Self-Evolving: Qwen3-VL-Plus (Proprietary)
Vanilla 78.0 77.9 68.3
w/o Batch-Level Post-Verification80.477.069.6
w/ Batch-Level Post-Verification 80.1 79.7 69.4
Table 3:Impact of different embedding models and
retrieval modes.We report results usingQwen3-VL-
Instruct-8B. Best results are marked inbold.
Embedding Model Image Text MathVista MMMU valMMStar
Self-Evolving: Qwen3-VL-8B-Instruct (Open-source)
Vanilla - - 73.2 67.1 62.1
Qwen3-Embedding(Zhang et al.) -✓75.5 66.6 59.1
Qwen2.5-VL-Embedding(Li et al.)✓ ✓77.366.2 63.2
Qwen3-VL-Embedding(Li et al.)✓ ✗76.3 66.0 63.0
Qwen3-VL-Embedding(Li et al.)✗ ✓77.1 66.2 62.3
Qwen3-VL-Embedding(Li et al.) ✓✓ 77.3 68.4 63.9
cation remains useful for preventing degradation.
Ablation on retrieval embeddings.We study
how the embedding model and input modality af-
fect M2Note as shown in Table 3. Using a text-
only embedding model (Qwen3-Embedding) can
improve MathVista, but tends to generalize worse
on other benchmarks such as MMStar. In contrast,
VLM-based embedding models (Qwen3/2.5-VL-
Embedding) that encode both image and text con-
sistently achieve better performance, suggesting
that multimodal indexing better captures the visual
cues needed to retrieve relevant notes. This is fur-
ther supported by the modality ablation: using a
single modality, either image or text, is inferior to
using both, indicating that visual and textual signals
are complementary for robust note retrieval.
7

What kind of sugar is this, aldose or ketose? <image> Baseline: To determine whether the sugar is an aldose or a ketose, we need to examine the functional group present in the molecule. -Aldose: Contains an aldehyde group (–CHO) at carbon 1. -Ketose: Contains a ketone group (C=O) at carbon 2 or higher. Looking at the structure provided: The top carbon (carbon 1) has a double bond to oxygen and a single bond to hydrogen — this is characteristic of a carbonyl group.Since the carbonyl group is attached to two carbon atoms (not H and a carbon), it’s a ketone, not an aldehyde.The guidance is applicable because the question asks to classify an open-chain sugar structure based on the position of its carbonyl group. - The structure shows a carbonyl group (C=O) at the first carbon (C1). - This carbon is bonded to a hydrogen atom (H–C=O), which defines an aldehyde functional group. - According to the CORE strategy: “If the carbonyl group is at C1 → classify as aldose.” Therefore, this is an aldose.
The image shows Belbin’s Team Roles model. These roles can be grouped into three directions. Which direction isincorrect?<image>
Number of entries: 52Average length (tokens): 375 MathPhysicsArtHistoryChemistryDocumentSubject 1:Biochemistry — Identifying thetypeofopen-chain sugarsGuidance: Locate the carbonyl and assign numbering:C1 with H–C=O → aldose (aldehyde);C2+ with C=O bonded to two carbons → ketose (ketone). Don’t apply directly tocyclicforms where the carbonyl is hidden/anomeric…Subject 2: Diagram — validating options against a given diagramGuidance: For “which option is wrong” questions, eliminate choices thatintroduce elements not present in the referenced model(external conceptual imports), and verify the remaining options bydirect category mappingto the model’s stated clusters.
Baseline:  (Summarized) The response selects an option based on “it sounds like a valid management direction,” without strictly matching each choice to Belbin’s three cluster directions in the model. This makes it prone to incorrectly accepting anexternal concept(e.g., “catering to customer personas”) as if it were part of Belbin’s taxonomy. Belbin team roles are clustered intothree directions:Action-oriented,Thought-oriented, andSocially-oriented. Options(A),(B), and(C)correspond to these clusters, while(D) catering to customer personasis not part of the Belbin model. Answer: (D).
(A) action oriented role (B) thought role  (C) socially oriented role (D) catering to customer personas
Figure 4:Qualitative results.For a STEM (biochemistry) question and a document understanding question, the
baseline tends to rely on superficial cues and makes incorrect selections, while M2Note retrieves subject-specific
guidance from the external mistake notebook and corrects the reasoning by enforcing key structural checks (e.g.,
carbonyl position for aldose/ketose; option-to-cluster mapping for Belbin roles).
Table 4:Notebook transfer across different VLMs.
We report results using three different VLMs.
Tuning Model Mem Len MathVista MMMU valMMStar
Notebook obtained via self-evolving Qwen3-VL-8B-Instruct
Qwen3-VL-Plus(Bai et al.) – – 78.0 77.9 68.3
w/ Specific Notebook 61 261 78.7(↑0.7 ) 76.8(↓1.1 ) 68.6(↑0.3 )
Qwen3-VL-32B-Instruct(Bai et al.) – – 81.1 73.1 67.8
w/ Specific Notebook 61 261 81.4(↑0.4 ) 74.1(↑1.0 ) 68.1(↑0.3 )
Notebook obtained via self-evolving Qwen3-VL-Plus
Qwen3-VL-8B-Instruct(Bai et al.) – – 73.2 67.1 62.1
w/ Specific Notebook 33 469 74.7(↑1.5 ) 67.4(↑0.3 ) 62.7(↑0.6 )
Notebook transfer across models.We evaluate
whether an evolved mistake notebook can trans-
fer across different VLMs (Table 4). Specifically,
we first self-evolve a notebook with one backbone
(e.g.,Qwen3-VL-8B-InstructorQwen3-VL-Plus),
then freeze it and use it as external guidance for a
different tuning model, without further evolution
or parameter updates. Overall, the transferred note-
books remain effective, indicating that M2Note cap-
tures partially model-agnostic and reusable guid-
ance. However, cross-model transfer is not always
risk-free, and mismatched guidance can hurt per-
formance in some cases. Moreover, compared with
self-evolved notebooks in Table 1, transferred note-
books generally yield smaller gains, suggesting that
the strongest improvements come from adapting to
each model’s own failure patterns.
4.3 Qualitative Results
Fig. 4 presents two representative cases illustrating
how M2Note improves multimodal reasoning. In
the STEM (biochemistry) example on the left, the
baseline gives a plausible explanation but misiden-tifies key structural cues, leading to an incorrect
classification. With M2Note, the model retrieves
subject-level guidance from the external mistake
notebook (e.g., locate the carbonyl and verify its po-
sition), which enforces essential structural checks
and corrects the conclusion. In the diagram under-
standing example on the right, the baseline tends
to pick an option that sounds reasonable without
strictly aligning choices to the diagram’s defined
taxonomy, making it vulnerable to introducing hal-
lucinations. M2Note retrieves guidance that explic-
itly requires option-to-cluster mapping and elimi-
nates elements not present in the reference model,
yielding the correct answer. More visualizations,
including case analyses and mistake-notebook visu-
alizations, are provided in Appendix B (Figure 5).
5 Conclusion
We introduce M2Note, a training-free framework
for continually improving VLMs through multi-
modal mistake notebook learning. Instead of up-
dating model weights, M2Note distills model fail-
ures into reusable subject-specific guidance and
retrieves relevant notes at inference time via multi-
modal RAG, helping the model verify its reasoning
and avoid repeated errors. A key mechanism is
batch-levelaccept-if-improvesverification, which
filters noisy notebook updates and stabilizes contin-
ual evolution by retaining only beneficial changes.
Across six benchmarks covering STEM/math rea-
soning, general VQA, and document understand-
ing, M2Note yields consistent gains in bothself-
evolvingandcross-model evolvingsettings, while
also combining effectively with CoT prompting.
8

Overall, these results suggest that reusing verified
failure cases provides a practical and effective way
to enhance VLM robustness at deployment.
6 Limitation
M2Note evolves in a way that resembles how hu-
mans learn from experience: instead of rewriting
one’s “brain” (model weights), it repeatedly sum-
marizes failures into reusable rules and consults
them when facing similar situations. This mecha-
nism is most effective when mistakes exhibit recur-
ring structure, so that a distilled note can be reliably
reused. As a result, M2Note tends to work best in
relatively narrow domains (e.g., math reasoning),
where tasks share stable abstractions and retrieved
guidance is more likely to transfer. In broad do-
mains with long-tail visual diversity, relevant notes
can be harder to retrieve, and mismatched guidance
may introduce misleading context or amplify hal-
lucinations, making continual evolution less stable.
These limitations suggest two directions: ❶build-
ing more generalizable notebooks through stronger
abstraction and a more reliable post-update ver-
ification strategy, so that notes transfer beyond
near-duplicate failures and harmful guidance is less
likely to be retrieved; and ❷enriching notes be-
yond text (e.g., reference images, tool-use traces,
structured checklists) to provide more grounded
and actionable guidance.
References
Shuai Bai, Yuxuan Cai, Ruizhe Chen, and 1 oth-
ers. 2025. Qwen3-vl technical report.Preprint,
arXiv:2511.21631.
Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay,
and 1 others. 2024. An introduction to vision-
language modeling.Preprint, arXiv:2405.17247.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao
Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi
Li, Haojia Lin, and 1 others. 2025. Training-free
group relative policy optimization.arXiv preprint
arXiv:2510.08191.
Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier,
Karthik Narasimhan, and Shunyu Yao. 2023. Fireact:
Toward language agent fine-tuning.arXiv preprint
arXiv:2310.05915.Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang
Zang, Zehui Chen, Haodong Duan, Jiaqi Wang,
Yu Qiao, Dahua Lin, and 1 others. 2024. Are we
on the right way for evaluating large vision-language
models?Advances in Neural Information Processing
Systems, 37:27056–27087.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and
1 others. 2025. Gemini 2.5: Pushing the frontier with
advanced reasoning, multimodality, long context, and
next generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Yanqi Dai, Yuxiang Ji, Xiao Zhang, Yong Wang, Xi-
angxiang Chu, and Zhiwu Lu. 2026. Harder is better:
Boosting mathematical reasoning via difficulty-aware
grpo and multi-aspect question reformulation.arXiv
preprint arXiv:2601.20614.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, and 1 others. 2024. A
survey on llm-as-a-judge.The Innovation.
Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng,
Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang,
Jianyu Jiang, Jiawei Wang, and 1 others. 2025.
Seed1. 5-vl technical report.arXiv preprint
arXiv:2505.07062.
Yicheng He, Chengsong Huang, Zongxia Li, Ji-
axin Huang, and Yonghui Yang. 2025. Visplay:
Self-evolving vision-language models from images.
Preprint, arXiv:2511.15661.
Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hong-
ming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang,
Haitao Mi, and Dong Yu. 2025. R-zero: Self-
evolving reasoning llm from zero data.arXiv
preprint arXiv:2508.05004.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xi-
angxiang Chu, and Liaoni Wu. 2025. Tree search
for llm agent reinforcement learning.arXiv preprint
arXiv:2509.21240.
Hongrui Jia, Chaoya Jiang, Shikun Zhang, and Wei
Ye. 2026. From blind spots to gains: Diagnostic-
driven iterative training for large multimodal models.
Preprint, arXiv:2602.22859.
Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min-
joon Seo, Hannaneh Hajishirzi, and Ali Farhadi.
2016. A diagram is worth a dozen images. InEuro-
pean conference on computer vision, pages 235–251.
Springer.
9

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Kunchang Li, Yali Wang, Yinan He, Yizhuo Li,
Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen,
Ping Luo, and 1 others. 2024. Mvbench: A com-
prehensive multi-modal video understanding bench-
mark. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages
22195–22206.
Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin
Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Jun-
yang Lin. 2026. Qwen3-vl-embedding and qwen3-
vl-reranker: A unified framework for state-of-the-
art multimodal retrieval and ranking.Preprint,
arXiv:2601.04720.
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae
Lee. 2024a. Improved baselines with visual instruc-
tion tuning. InProceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition,
pages 26296–26306.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning.Advances in
neural information processing systems, 36:34892–
34916.
Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li,
Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi
Wang, Conghui He, Ziwei Liu, and 1 others. 2024b.
Mmbench: Is your multi-modal model an all-around
player? InEuropean conference on computer vision,
pages 216–233. Springer.
Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang,
Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-
Lin Liu, Lianwen Jin, and Xiang Bai. 2024c. Ocr-
bench: on the hidden mystery of ocr in large multi-
modal models.Science China Information Sciences,
67(12):220102.
Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chun-
yuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-
Wei Chang, Michel Galley, and Jianfeng Gao. 2023.
Mathvista: Evaluating mathematical reasoning of
foundation models in visual contexts.arXiv preprint
arXiv:2310.02255.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
and 1 others. 2023. Self-refine: Iterative refinement
with self-feedback.Advances in neural information
processing systems, 36:46534–46594.
Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. 2022. Chartqa: A benchmark
for question answering about charts with visual andlogical reasoning. InFindings of the association for
computational linguistics: ACL 2022, pages 2263–
2279.
Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen.
2025. A survey of multimodal retrieval-augmented
generation.arXiv preprint arXiv:2504.08748.
Niklas Muennighoff, Zitong Yang, Weijia Shi, Xi-
ang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke
Zettlemoyer, Percy Liang, Emmanuel Candès, and
Tatsunori B Hashimoto. 2025. s1: Simple test-time
scaling. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 20286–20332.
Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang,
Zifeng Wang, Rujun Han, Long T Le, Samira Daruki,
Xiangru Tang, and 1 others. 2025. Reasoningbank:
Scaling agent self-evolving with reasoning memory.
arXiv preprint arXiv:2509.25140.
Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang
Zhu, and Michael Zeng. 2023. Automatic prompt op-
timization with “gradient descent” and beam search.
InProceedings of the 2023 conference on empiri-
cal methods in natural language processing, pages
7957–7968.
Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu
MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng
Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang,
and 1 others. 2025. We-math: Does your large multi-
modal model achieve human-like mathematical rea-
soning? InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 20023–20070.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark, and
1 others. 2021. Learning transferable visual models
from natural language supervision. InInternational
conference on machine learning, pages 8748–8763.
PmLR.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Prox-
imal policy optimization algorithms.Preprint,
arXiv:1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
10

Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2023. Re-
flexion: Language agents with verbal reinforcement
learning.Advances in neural information processing
systems, 36:8634–8652.
Xuanbo Su, Yingfang Zhang, Hao Luo, Xiaoteng Liu,
and Leo Huang. 2025. Mistake notebook learning:
Batch-clustered failures for training-free agent adap-
tation.arXiv preprint arXiv:2512.11485.
Meghana Sunil, Manikandarajan Venmathimaran, and
Muthu Subash Kavitha. 2026. ireasoner: Trajectory-
aware intrinsic reasoning supervision for self-
evolving large multimodal models.Preprint,
arXiv:2601.05877.
Omkar Thawakar, Shravan Venkatraman, Ritesh
Thawkar, Abdelrahman Shaker, Hisham Cholakkal,
Rao Muhammad Anwer, Salman Khan, and Fahad
Khan. 2025. Evolmm: Self-evolving large multi-
modal models with continuous rewards.Preprint,
arXiv:2511.16672.
Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing
Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li.
2024. Measuring multimodal mathematical reason-
ing with math-vision dataset.Advances in Neural
Information Processing Systems, 37:95095–95169.
Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi, Yueqian
Lin, Yiran Chen, Hai Helen Li, Kun Wan, and
Wentian Zhao. 2025a. Vision-zero: Scalable vlm
self-improvement via strategic gamified self-play.
Preprint, arXiv:2509.25541.
Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu,
Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin
Jing, Shenglong Ye, Jie Shao, and 1 others. 2025b. In-
ternvl3. 5: Advancing open-source multimodal mod-
els in versatility, reasoning, and efficiency.arXiv
preprint arXiv:2508.18265.
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and
Graham Neubig. 2025c. Agent workflow memory.
InInternational Conference on Machine Learning,
pages 63897–63911. PMLR.
Jason Wei, Maarten Bosma, Vincent Y . Zhao, Kelvin
Guu, Adams Wei Yu, Brian Lester, Nan Du, An-
drew M. Dai, and Quoc V . Le. 2022a. Finetuned
language models are zero-shot learners.Preprint,
arXiv:2109.01652.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022b. Chain-of-thought prompting
elicits reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
xAI. Realworldqa: A benchmark for real-world
spatial understanding. https://huggingface.
co/datasets/xai-org/RealworldQA . Accessed:
2025-04-26.Chaoran Xu, Yingmao Miao, Pengfei Zhang, Hao Dou,
Lei Sun, and Xiangxiang Chu. 2026. Rise: Reli-
able improvement in self-evolving vision-language
models.arXiv preprint arXiv:2605.20914.
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao
Liu, Quoc V Le, Denny Zhou, and Xinyun Chen.
2023. Large language models as optimizers. In
The Twelfth International Conference on Learning
Representations.
Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jian-
qiang Wan, Humen Zhong, Xuejing Liu, Mingkun
Yang, Peng Wang, Shuai Bai, and 1 others. 2025.
Cc-ocr: A comprehensive and challenging ocr bench-
mark for evaluating large multimodal models in lit-
eracy. InProceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 21744–
21754.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin,
Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan
Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and
16 others. 2025. Dapo: An open-source llm re-
inforcement learning system at scale.Preprint,
arXiv:2503.14476.
Zhenlong Yuan, Xiangyan Qu, Chengxuan Qian, Rui
Chen, Jing Tang, Lei Sun, Xiangxiang Chu, Dapeng
Zhang, Yiwei Wang, Yujun Cai, and Shuo Li. 2026.
Video-STAR: Reinforcing open-vocabulary action
recognition with tools. InThe Fourteenth Interna-
tional Conference on Learning Representations.
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng,
Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang,
Weiming Ren, Yuxuan Sun, and 1 others. 2024.
Mmmu: A massive multi-discipline multimodal un-
derstanding and reasoning benchmark for expert agi.
InProceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 9556–
9567.
Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao
Liu, Yuxiao Dong, and Jie Tang. 2024. Agenttun-
ing: Enabling generalized agent abilities for llms. In
Findings of the Association for Computational Lin-
guistics: ACL 2024, pages 3053–3077.
Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou,
Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji
Deng, Zouying Cao, and 1 others. 2025. Agente-
volver: Towards efficient self-evolving agent system.
arXiv preprint arXiv:2511.10395.
Qizheng Zhang, Changran Hu, Shubhangi Upasani,
Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru,
Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li,
and 1 others. 2025a. Agentic context engineering:
Evolving contexts for self-improving language mod-
els.arXiv preprint arXiv:2510.04618.
11

Tianjun Zhang, Aman Madaan, Luyu Gao, Steven
Zheng, Swaroop Mishra, Yiming Yang, Niket Tan-
don, and Uri Alon. 2024a. In-context principle learn-
ing from mistakes. InProceedings of the 41st In-
ternational Conference on Machine Learning, pages
59520–59558.
Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi
Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and
Jingren Zhou. 2026. On-policy rl meets off-policy ex-
perts: Harmonizing supervised fine-tuning and rein-
forcement learning via dynamic weighting.Preprint,
arXiv:2508.11408.
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi
Dai, Dingkun Long, Pengjun Xie, Meishan Zhang,
Wenjie Li, and Min Zhang. 2024b. Gme: improving
universal multimodal retrieval by multimodal llms.
arXiv preprint arXiv:2412.16855.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025b. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu
Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel:
Llm agents are experiential learners. InProceedings
of the AAAI Conference on Artificial Intelligence,
volume 38, pages 19632–19642.
Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu,
Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen,
Chuhan Li, Zhijian Xu, and 1 others. 2025. Mmvu:
Measuring expert-level multi-discipline video under-
standing. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 8475–8489.
Longtao Zheng, Rundong Wang, Xinrun Wang, and
Bo An. Synapse: Trajectory-as-exemplar prompting
with memory for computer control. InThe Twelfth
International Conference on Learning Representa-
tions.
Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan,
Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun
Zhang, Kun Shao, Linyi Yang, and 1 others. 2025.
Memento: Fine-tuning llm agents without fine-tuning
llms.arXiv preprint arXiv:2508.16153.
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han,
Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy
Ba. 2022. Large language models are human-level
prompt engineers. InThe eleventh international con-
ference on learning representations.A Prompts Used in the M2Note
Framework
A.1 Tuner Model Prompts
Guidance Extraction
You are a Cognitive Strategy Expert. Your
goal is to inducegeneralizable rulesfrom
specific model errors, not just correct them.
Subject: {subject}
Error Data: {error_context}
Task: Analyze the underlying logical flaws
in the Error Data and formulate a reusable
problem-solving heuristic.
Constraints (Total≤350 words):
1.Abstracted Error Pattern(≤60 words)
- Describe thetypeof situation where this
error occurs (strip specific numbers/names).
- Do NOT simply rewrite the original ques-
tions.
2.Root Cause Analysis(≤50 words)
- Identify the cognitive gap (e.g., confusing
correlation with causation, ignoring boundary
conditions).
3.Generalizable Strategy (CORE)(≤120
words)
- Provide a step-by-step heuristic or checklist
applicable to ANY similar future problem.
- Use “If [condition], Then [action]” format
where possible.
- Must work beyond the provided examples.
4.Anti-Patterns & Boundary( ≤80 words)
- List 1-2 specific scenarios where this strat-
egy should NOT be applied.
- List 1-2 common misinterpretations of this
guidance.
Output Style:
- High-level, structured, knowledge-focused,
and reusable for similar future questions.
- Avoid referencing specific details from the
provided Error Data unless illustrating a pat-
tern.
12

Subject Classification
You are an expert in categorizing ques-
tions into precise, high-relevance subjects for
Retrieval-Augmented Generation (RAG).
Your goal is to assign each question a subject
label (about 10-50 words) that:
- Maximizes retrieval relevance by precisely
describing the problem type and solution
method.
- Groups only genuinely similar questions to-
gether (same domain AND same approach).
- Avoids over-broad categories that would
match unrelated problems.
- Reuses the same subject name for closely
related questions.
CRITICAL: The subject must be specific
enough to prevent irrelevant retrieval. In-
clude, when applicable:
1. Primary Domain (e.g., Combinatorics,
Complex Analysis, Linear Algebra, Physics,
Programming, Document Understanding)
2. Problem Type (e.g., counting with con-
straints, roots of unity products, debugging
API parameters, OCR table extraction)
3. Solution Method (e.g., stars and bars,
polynomial/root identities, Hensel’s lemma,
reproduce-minimize-fix)
Examples of GOOD subjects (specific):
✓Complex Analysis: Evaluating products
over roots of unity using polynomial eval-
uation and complex identities
✓Document Understanding: Extracting ta-
bles from scanned PDFs using layout detec-
tion + OCR + row/column reconstruction
Examples of BAD subjects (too broad):
✗number theory (too broad - could match any
modulo problem)
✗physics (too broad - could match any
physics problem)
Key principle: If a problem spans multiple
domains, label it by the primary one.
Output only the finalized subject label(s).Guidence Merge
Synthesize guidance for subject: {subject}
Existing guidance from related subjects: {ex-
isting_guidance}
New guidance: {new_guidance}
Merge into a single coherent guidance (max
2048 chars) that:
- Combines insights from related subjects
with new guidance
- Eliminates redundancy while preserving key
information and examples of the mistakes
-Preserves and emphasizes applicability
conditions
- clearly state when each method applies
- Focuses on actionable advice
- Maintains consistent style
-Includes warnings about when NOT to
apply the guidanceto avoid misapplication
Merged guidance:
Subject Merge
The following subjects are related: {subjects}
Provide a single, concise subject name (≤10
words) that best represents all of these. Only
output the subject name, nothing else.
A.2 Tuning Model Prompts
User Prompt
Question: {question}
Options: {options}
Please select the correct answer from the op-
tions above.
13

RAG-Enhanced System Prompt
The following mistake notes are not necessar-
ily tied to the current question, but you may
use them to deepen your analytical approach
IMPORTANT: Before applying any guid-
ance below, carefully evaluate:
1. Does the current problem match the appli-
cability conditions stated in the guidance?
2. Is the problem type and context similar to
the examples in the guidance?
3. If the problem is totally different (e.g.,
combinatorics vs modulo arithmetic, complex
numbers vs number theory), do NOT force-fit
the guidance.
4. Only use guidance that is clearly relevant
to the current problem.
Gudience Subject: {subject} - {guidance}
Before solving, review the guidance. State
whether it is: applicable, partially applicable,
or irrelevant. Use only applicable parts.
Since the guidance introduced via multimodal
RAG (Lewis et al., 2020; Mei et al., 2025) may not
necessarily apply to the current question, in addi-
tion to the top- Kand threshold settings in the RAG
configuration, we further emphasize in the system
prompt that the model should judge whether the
retrieved content is applicable to the current ques-
tion. This is crucial for reducing hallucinations. As
shown in Fig. 5, several cases are presented where
the model’s response explicitly includes this judg-
ment: the first two cases are applicable, while the
last one illustrates an inapplicable scenario. Table 5
compares blindly following the guidance (naive)
with introducing a judgment mechanism (judge-
ment); the latter effectively improves the model’s
problem-solving accuracy.
B More Visualizations
Below we present several entries from the mis-
take notebook, obtained by trainingQwen3-VL-8B-
Instructas theTuning Modelon MMMU (Yue
et al., 2024). As shown, the structured guidance is
produced using theTuner Modelwith the “Guid-
ance Extraction” prompt, while theNoteand the
Taskfield are derived using the “Subject Classifi-
cation” prompt.Table 5:Effect of applicability judgment for retrieved
guidance. Results on three benchmarks and two back-
bones.Naiveuses retrieved guidance directly, while
Judgmentverifies relevance first, consistently improv-
ing performance. Best results are inbold.
Method MathVista MMMU val MMStar
Self-Evolving: Qwen3-VL-8B-Instruct (Open-source)
Vanilla 73.2 67.1 62.1
w/ M2Note (Naive) 76.2 67.864.1
w/ M2Note (Judgment) 77.3 68.4 63.9
Self-Evolving: Qwen3-VL-Plus (Proprietary)
Vanilla 78.0 77.9 68.3
w/ M2Note (Naive) 79.5 76.9 69.0
w/ M2Note (Judgment) 80.1 79.7 69.4
Note 1: Document Understanding
TaskExtract structured brainstorming work-
flow steps from a quadrant diagram (visual
category mapping + label extraction).
Abstracted Error PatternOptions that
sound reasonable are mistaken as valid steps
even when they contradict the diagram’s col-
laborative workflow (e.g., silent reading, dur-
ing disagreement).
Root CausePlausibility is prioritized over
fidelity to the visual framework; behav-
ioral/structural constraints are ignored.
Generalizable Strategy (CORE)
•Reject options introducing elements ab-
sent from the visual model (silence, con-
flict resolution, external tools).
•Reject behaviors inconsistent with the in-
teraction pattern (solo reading vs group
discussion).
•If ambiguous, validate against core stages:
categorization →discussion →selection.
•Prefer behavioral alignment over fluent
wording.
BoundaryNot suitable when silent reflection
is explicitly part of the process, or when the
model allows multiple valid interpretations.
14

Note 2: Medical Imaging
TaskIdentify anatomical landmarks in fun-
dus photographs (visual cue detection + retinal
structure recognition).
Abstracted Error PatternStructural spar-
sity is overgeneralized into complete absence
across a region without checking the exact
boundary/layer.
Root CauseConfusing minimal presence
with absence; weak boundary checking for lay-
ered anatomy and region-specific exceptions.
Generalizable Strategy (CORE)
•Verify absence at the exact indicated loca-
tion (not nearby areas).
•Check layers independently: photorecep-
tors→bipolar →ganglion →blood ves-
sels.
•Choose “all of the above” only if every
listed condition holds.
•Cross-reference standard anatomy for
region-specific exceptions.
BoundaryNot for dynamic processes (dis-
ease progression) or non-structural properties
(function).Note 3: Chemistry
TaskDetermine the maximum number of sac-
charin molecules from an atomic inventory
(limiting-reactant style stoichiometry).
Abstracted Error PatternUsers misidentify
the limiting element by ignoring stoichiometric
ratios or miscounting atoms in the molecular
formula.
Root CauseAtom availability is mistaken for
molecule yield; constraints are not computed
per element to find the true bottleneck.
Generalizable Strategy (CORE)
1.Derive the exact molecular formula
(count all atoms).
2.For each element: available atoms ÷re-
quired per molecule⇒max molecules.
3.The smallest quotient determines the max-
imum yield (the limiting element).
4.Sanity-check: if quotients look inconsis-
tent, recheck formula/inventory.
5.Confirm the final answer satisfies the lim-
iting element’s requirement exactly.
BoundaryNot for non-stoichiometric set-
tings (catalysis, equilibrium-limited yield) or
inventories with impurities/intermediates not
modeled.
15

Note 4: Education
TaskIdentify a classroom activity for literary
analysis via role-play and multi-perspective
retelling (media simulation).
Abstracted Error PatternThe activity is
misidentified by matching the image’s media
appearance (e.g., a news set) rather than its
pedagogical function.
Root CauseSurface-level visual similarity
overrides functional interpretation (real media
format vs simulated classroom role-play).
Generalizable Strategy (CORE)
•Eliminate options that are literal media
formats (e.g., News Program).
•Prefer terms describing pedagogical func-
tion: interviewing, role assumption, per-
spective simulation.
•Check definitions for student agency, dia-
logue, and interpretive framing.
•Treat broadcast-studio visuals as distrac-
tors; focus on the learning objective.
•Differentiate close terms: Hot Seat (struc-
tured questioning) vs Readers Theatre
(scripted reading).
BoundaryNot for purely observational tasks
(watching media) or when the explicit goal is
media production.Note 5: Document Understanding
TaskCount pathogens on leaf images (visual
segmentation + pattern recognition).
Abstracted Error PatternMultiple symp-
tom patterns are incorrectly treated as multiple
pathogens, ignoring overlap and single-agent
variability.
Root CauseSymptoms (markers) are con-
flated with causal entities; overlap/co-infection
and environmental confounds are not checked.
Generalizable Strategy (CORE)
•Check whether different patterns overlap
spatially/temporally; overlap may indi-
cate one agent.
•Compare with known symptom profiles
to validate distinct causes.
•If distinctness is supported, count inde-
pendent infection zones (not symptom
types).
•Consider context (treatment, stressors)
that can mimic/mask infections.
•If evidence is weak, prefer “unknown”
over forcing a precise count.
BoundaryNot when symptoms clearly indi-
cate unrelated agents (e.g., fungal + viral), or
when the question asks for types rather than
counts.
16

Note 6: Combinatorics
TaskSchedule exams under conflict con-
straints using graph coloring (minimum num-
ber of time slots).
Abstracted Error PatternMinimum slots
are underestimated by ignoring constraint prop-
agation or assuming a locally good coloring is
globally feasible.
Root CauseChromatic number is conflated
with clique size; indirect (non-local) conflicts
are not fully enforced/validated.
Generalizable Strategy (CORE)
1.Build a conflict graph: exams = vertices;
edges = shared students/resources.
2.Color the graph (greedy or exact), enforc-
ing all pairwise conflicts.
3.Validate the schedule: no student may
have two exams in the same slot/day.
4.If violations occur, re-solve using back-
tracking/constraint propagation.
5.Re-check feasibility against real-world
rules (concurrency constraints).
BoundaryNot for dynamic/time-dependent
constraints beyond a static conflict graph, or
when intra-day sequencing/rooms must be
modeled.
C Experimental Settings
C.1 Implementation Details
Our supervised training settings on MMMU (Yue
et al., 2024) and MathVista (Lu et al., 2023) are as
follows: ❶On MMMU, we train on the officialdev
split (150 STEM questions) to obtain a tuned mem-
ory, which is then used as an external notebook
during inference on the MMMUvalidationsplit,
where we report accuracy. ❷On MathVista, we
randomly sample 320 examples from thetestsplit
to train the notebook, and evaluate on thetext-mini
split, following common practice in prior work.
For the other benchmarks (MMStar (Chen
et al., 2024), AI2D (Kembhavi et al., 2016), Real-
worldQA (xAI), and ChartQA (Masry et al., 2022)),
we adopt a test-time scaling (TTS) (Muennighoff
et al., 2025) setup, where the memory is updated
online during evaluation to progressively enhance
the model’s capability.C.2 Benchmark Details
For comprehensive evaluation across diverse vi-
sual modalities, reasoning skills, and knowledge
domains, we conduct experiments on six widely-
used multimodal benchmarks: MMMU (Yue et al.,
2024), MathVista (Lu et al., 2023), MMStar (Chen
et al., 2024), RealworldQA (xAI), AI2D (Kem-
bhavi et al., 2016), and ChartQA (Masry et al.,
2022). Together, these benchmarks cover broad
subject understanding, visual mathematical reason-
ing, real-world recognition and commonsense, dia-
gram interpretation, and chart comprehension.
MMMU(Yue et al., 2024) is a large-scale bench-
mark designed to measure multimodal understand-
ing across a wide range of academic disciplines
and professional fields. Questions typically require
jointly interpreting images (e.g., figures, tables, di-
agrams, screenshots) and text, and then perform-
ing domain-specific reasoning. These questions
span 30 subjects and 183 subfields, comprising 30
highly heterogeneous image types, such as charts,
diagrams, maps, tables, music sheets, and chem-
ical structures. MMMU is challenging due to its
breadth of subjects and its emphasis on knowledge-
intensive and reasoning-intensive problems rather
than purely perceptual recognition.
MathVista(Lu et al., 2023) focuses on visual
mathematical reasoning, where models must solve
math problems grounded in visual context. Inputs
often contain charts, diagrams, geometry figures,
or real-world scenes with quantitative cues. The
benchmark evaluates capabilities such as extract-
ing numerical information from images, mapping
visual elements to formal quantities, performing
multi-step reasoning, and producing the final math-
ematical answer. It is well-suited for analyzing er-
ror patterns related to perception-to-symbol ground-
ing and step-by-step quantitative reasoning.
MMStar(Chen et al., 2024) is a general mul-
timodal evaluation benchmark aimed at assessing
robust vision-language understanding across a vari-
ety of everyday and document-style visual inputs.
Questions span recognition, attribute reasoning,
spatial reasoning, and higher-level comprehension.
Compared with subject-focused benchmarks, MM-
Star is commonly used to test overall multimodal
competence and generalization, making it useful
for verifying whether our method improves broad
visual-text reasoning rather than overfitting to a
narrow domain.
RealworldQA(xAI) targets question answering
17

in real-world visual scenarios, emphasizing practi-
cal knowledge and reasoning grounded in natural
images. Questions often rely on recognizing ob-
jects, scenes, affordances, and context, and may
require commonsense or everyday knowledge be-
yond simple identification. This benchmark is help-
ful for evaluating whether the model’s “mistake-
driven” updates improve robustness in realistic set-
tings where visual ambiguity and implicit assump-
tions are common.
AI2D(Kembhavi et al., 2016) is a benchmark
for science diagram understanding. It contains el-
ementary and middle-school style diagrams (e.g.,
life cycles, physics illustrations, anatomy/biology
schematics) paired with questions that require in-
terpreting labeled components, arrows, processes,
and relationships. AI2D is particularly challenging
because correct answers often depend on under-
standing diagram structure and semantics, not just
recognizing visual entities.
ChartQA(Masry et al., 2022) evaluates chart
and plot understanding, including bar charts, line
charts, pie charts, and other visualizations com-
monly seen in reports. Questions require reading
values from axes/legends, comparing trends, per-
forming simple arithmetic, or synthesizing infor-
mation across multiple chart elements. ChartQA is
valuable for analyzing errors related to visual-text
alignment (e.g., legend-to-series mapping), numer-
ical extraction, and compositional reasoning over
structured visual data.
18

Updated System Prompt (injected subject & guidance):·Topic:Solving linear circuits (superposition + Laplace)with dependent sources.·Core rule:During superposition,do not deactivate dependent sources; treat them asactive constraints. If a dependent source’s output equals a node voltage, explicitly substitute it into KCL/KVL and solve consistently.Updated Response (key excerpt to visualize):·Applicability decision:The model ultimately concludes the guidance isapplicablebecause the circuit includes dependent sources (notably a VCVS proportional to𝑣!, which must remain active under superposition).·Outcome:SelectsOption (D), matching the standard answer.Question:Using superposition and Laplace transform techniques, calculate the voltage𝑣!𝑡in the circuit.<image>Answer:(D) 𝑣!𝑡=[2−2.4𝑒"#/!−1/5𝑠𝑖𝑛𝑡−63.44∘]𝑢𝑡
Question: The lesion depicted here. <image>Ground-truth answer:(E): a, b and c are correctUpdated System Prompt (injected subject & guidance):·Topic:Histopathology image interpretation with multiple-choice options.·Core rule (composite-option check):For answers like “A+B+C,”verify each component independently; select a composite only ifalllisted statements are supported. Avoid “feature aggregation” bias (assuming partial match ⇒ whole option correct).Updated Response (key excerpt to visualize):·Applicability decision:The model explicitly labels the guidanceApplicablebecause the task ishistology-basedand includescomposite optionsrequiring component-wise validation.·Outcome:SelectsOption (E), matching the standard answer.
Question:Using a finite summation, compute the initial deflection at midspan for the beam in Figure P8.42. Given𝐸=3000kips/in2. Use 3-ft segments. Assume𝐼=0.5𝐼&. Answer using a single word. <image>Ground-truth answer:1.06(units per problem context, typically inches)Updated System Prompt (injected subject & guidance):·Topic:Truss displacementusingvirtual workormethod of consistent deformations.·Core rule:apply aunit virtual loadat the target point → compute real/virtualmember forces: "()%&'(); Or solve redundant via compatibility (consistent deformations); Emphasis: don’t guess; ensure equilibrium closure; keep units consistent.Updated Response (key excerpt to visualize): ·Applicability decision:Irrelevant. The guidance is fortrusses(axial member forces,AE, virtual work), but the question is abeam deflectionproblem requiringfinite summation / numerical integration of curvatureusingEIover 3-ft segments—different structural model, different governing equations.
Figure 5:Qualitative examples of applicability judgment for retrieved guidance. The model explicitly decides
whether the injected subject-guidance pair is applicable: the first two cases areApplicableand lead to correct
answers, while the last case isIrrelevantand is rejected.
19
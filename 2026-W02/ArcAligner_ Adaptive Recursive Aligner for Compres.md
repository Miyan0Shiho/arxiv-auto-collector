# ArcAligner: Adaptive Recursive Aligner for Compressed Context Embeddings in RAG

**Authors**: Jianbo Li, Yi Jiang, Sendong Zhao, Bairui Hu, Haochun Wang, Bing Qin

**Published**: 2026-01-08 15:44:52

**PDF URL**: [https://arxiv.org/pdf/2601.05038v1](https://arxiv.org/pdf/2601.05038v1)

## Abstract
Retrieval-Augmented Generation (RAG) helps LLMs stay accurate, but feeding long documents into a prompt makes the model slow and expensive. This has motivated context compression, ranging from token pruning and summarization to embedding-based compression. While researchers have tried ''compressing'' these documents into smaller summaries or mathematical embeddings, there is a catch: the more you compress the data, the more the LLM struggles to understand it. To address this challenge, we propose ArcAligner (Adaptive recursive context *Aligner*), a lightweight module integrated into the language model layers to help the model better utilize highly compressed context representations for downstream generation. It uses an adaptive ''gating'' system that only adds extra processing power when the information is complex, keeping the system fast. Across knowledge-intensive QA benchmarks, ArcAligner consistently beats compression baselines at comparable compression rates, especially on multi-hop and long-tail settings. The source code is publicly available.

## Full Text


<!-- PDF content starts -->

ArcAligner: Adaptive Recursive Aligner for Compressed Context
Embeddings in RAG
Jianbo Li*, Yi Jiang*, Sendong Zhao†, Bairui Hu, Haochun Wang, Bing Qin
Harbin Institute of Technology, China
{jianboli,yjiang,sdzhao,brhu,hcwang,qinb}@ir.hit.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) helps
LLMs stay accurate, but feeding long docu-
ments into a prompt makes the model slow
and expensive. This has motivated context
compression, ranging from token pruning and
summarization to embedding-based compres-
sion. While researchers have tried “compress-
ing” these documents into smaller summaries
or mathematical embeddings, there is a catch:
the more you compress the data, the more the
LLM struggles to understand it. To address this
challenge, we propose ArcAligner (Adaptive
recursivecontextAligner), a lightweight mod-
ule integrated into the language model layers to
help the model better utilize highly compressed
context representations for downstream gen-
eration. It uses an adaptive “gating” system
that only adds extra processing power when
the information is complex, keeping the system
fast. Across knowledge-intensive QA bench-
marks, ArcAligner consistently beats compres-
sion baselines at comparable compression rates,
especially on multi-hop and long-tail settings.
The source code is publicly available1.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Guu et al., 2020) is now the stan-
dard approach for keeping Large Language Models
(LLMs) (Achiam et al., 2023; Touvron et al., 2023)
factually accurate. By grounding answers in ex-
ternal documents rather than relying on memory
alone, RAG helps models avoid “hallucinations”
and answer difficult, niche questions more reliably.
However, the standard way of doing RAG–simply
pasting long documents into the prompt–creates
a bottleneck. As we ask models to handle longer
histories, deeper user profiles, or massive libraries
of evidence, the input quickly becomes too long
*Equal Contribution.
†Corresponding Author.
1https://github.com/liunian-Jay/ArcAligner.git
Figure 1: Comparison of different methods.
for the model to process efficiently. This makes
effective context compression no longer just an op-
timization, but a necessity for building responsive,
long-memory LLM agents.
To solve this problem, recent studies explore con-
text compression–essentially shrinking the content
before the LLM reads it (Liu et al., 2025). Cur-
rent research generally follows one of two paths
to achieve this. The first istext-level compres-
sion, which involves pruning unnecessary words or
summarizing documents into a shorter prompt (Pan
et al., 2024). The second isembedding-based
compression, which translates long text into a com-
pact “shorthand” of mathematical signals (embed-
dings) that the model can digest quickly (Cheng
et al., 2024; Ge et al.; Rau et al., 2025). These meth-
ods suggest a compelling possibility: if we can im-
prove an LLM’s in-model alignment to compressed
context, RAG could maintain answer quality while
significantly reducing time and cost.
However, as shown in Figure 1, these approaches
still face significant limitations (Nagle et al., 2024;
Deng et al., 2025; Chen et al., 2025). Text-level
compression is often unpredictable; because it re-
lies on picking and choosing specific words to keep,
1arXiv:2601.05038v1  [cs.CL]  8 Jan 2026

Figure 2: Illustration of the ArcAligner framework. The left part is an illustration of our training and inference
process, where the context is compressed and progressively refined within the LLM while the input query remains
unchanged. The right part is a flow of the token through each layer, passing through the LoRA and the gate, where
the context tokens are adaptively and recursively refined.
even a small error can result in the loss of crucial
evidence. This makes it particularly unreliable for
complex, multi-step questions where every detail
matters. On the other hand, embedding-based com-
pression avoids deleting text but creates a “lan-
guage barrier.” The compressed signals are mathe-
matically different from what the LLM is used to
seeing, making it difficult for the model to actu-
ally use that information to generate a grounded
answer. Despite progress on compression objec-
tives and encoder–decoder interfaces, performance
under strong compression still hinges on whether
the generator can consistently exploit and progres-
sively decode compressed evidence for answering.
We proposeArcAligner, a lightweight module
that enables LLMs to better read and reason over
compressed context representations. ArcAligner
operates within the model layers to progressively
align compressed signals with the representations
used during generation, reducing the mismatch be-
tween compressed inputs and the model’s native
processing space. It further introduces an adap-
tive alignment mechanism with a learned gate that
controls, layer by layer, how much additional re-
finement is applied to each piece of compressed in-
formation. This design helps preserve task-relevant
evidence and supports multi-step reasoning on com-
plex questions. Across a range of knowledge-
intensive benchmarks, ArcAligner consistently im-
proves answer quality and reliability over prior
compression-based baselines under the same con-
text budget. Our main contributions are:
•The ArcAligner Framework:We propose
ArcAligner, a parameter-efficient alignment
framework that performs deep semantic align-
ment from within the language model throughmulti-stage training.
•Adaptive Recursive Alignment:We pro-
pose a novel “gating” mechanism that selec-
tively processes information. By applying ad-
ditional refinement where needed, ArcAligner
improves the use of compressed context with-
out uniformly increasing computation.
•Robust Performance Gains:We provide ex-
tensive experiments and ablations showing
consistent gains over strong compression base-
lines, with clear improvements in “difficult”
scenarios, such as multi-hop and long-tail QA.
2 Related Work
2.1 Retrieval-Augmented Generation
RAG improves factuality by conditioning LLM on
retrieved evidence. The standard pipeline concate-
nates top- kpassages into the prompt and relies
on attention to integrate evidence. Existing ap-
proaches enhance evidence utilization with fusion-
in-decoder readers (Jiang et al., 2025c) and end-
to-end retrieval–generation training (Izacard and
Grave, 2021; Izacard et al., 2023; Jiang et al.,
2025a), and also investigate reliability and con-
trollable reliance on retrieved signals (Asai et al.,
2024). However, long prompts incur substantial
inference overhead, motivating methods that re-
duce the context burden while preserving evidence
usability.
2.2 Compressing Retrieved Evidence
Context compression reduces context before it
reaches the generator, viahardcompression in text
space orsoftcompression in embedding space.
2

Hard Compression (Text Space).Hard com-
pression shortens the prompt by selecting/prun-
ing tokens or sentences (Li et al., 2023; Jiang
et al., 2023b; Pan et al., 2024) or rewriting/distilling
passages into shorter textual evidence (Xu et al.).
These methods preserve a text-only interface but
require query-time decisions and can be sensitive
to downstream alignment; under aggressive bud-
gets, small mistakes may remove answer-critical
evidence and disproportionately hurt difficult or
long-tail questions (Pan et al., 2024).
Soft Compression (Embedding Space).Soft
compression replaces retrieved text with compact
continuous representations, e.g., encoding passages
into a small set of projected embedding token-
s/slots (Cheng et al., 2024), or injecting learned
context/memory representations via reconstruction-
style objectives and downstream fine-tuning (Ge
et al.; Rau et al., 2025; Pilchen et al., 2025). While
avoiding long prompts, it shifts the bottleneck to
alignment and usability: compressed embeddings
are heterogeneous to the generator’s hidden space
and typically require a projector, while training
must ensure reliable decoding under strong com-
pression (Rau et al., 2025; Pilchen et al., 2025).
2.3 From Compression to Usable Evidence
A key issue is thatcompressing or preserving in-
formationdoes not guaranteeeffective usedur-
ing decoding. Related work tackles adjacent as-
pects: parameter-efficient adaptation (adapters, pre-
fix/prompt tuning, low-rank updates) improves
behavior under constrained or non-standard in-
puts (Houlsby et al., 2019; Li and Liang, 2021;
Hu et al., 2022), while RAG-specific designs reg-
ulate reliance on retrieved evidence (e.g., self-
checking/controlled use and reducing over-reliance
via preference alignment) (Asai et al., 2024; Jiang
et al., 2025b). However, strong compression still
poses a distinct challenge—whether compact re-
trieval representations can be consistently inter-
preted and exploited throughout decoding. Un-
like embedding-based compression methods that
mainly optimize representations or compression
objectives (Cheng et al., 2024; Ge et al.; Rau et al.,
2025; Pilchen et al., 2025), ArcAligner targets
decode-time usability, strengthening how the gen-
erator integrates compressed retrieval information
for grounded generation.3 Methodology
We proposeArcAligner, a parameter-efficient
alignment framework (Figure 2) that improves the
usability of compressed embeddings by: (i) en-
forcinglayer-wise mandatory alignmentof context
slots, and (ii) enablingadaptive recursive align-
mentwhose depth is controlled by a learnedgate.
A three-stage training strategy progressively equips
the model with reconstruction ability, task compe-
tence, and adaptive refinement.
3.1 Problem Setup
Given a query q, a retriever returns the top- kpas-
sages{pi}k
i=1from a corpus P. Instead of con-
catenating retrieved passages as text, we compress
them into a short sequence of densecompressed
context embeddings
E∈Rm×d r,
where mis the number of context slots, drdenotes
the embedding dimension of each compressed con-
text embedding, and |pi|denotes the number of
tokens in passage pi, with m≪P
i|pi|. These
embeddings are provided to the language model as
a fixed set of designatedcontext slots, forming a
compact interface between retrieved evidence and
the base model.
3.2 Context Slot Interface
Let the hidden size of the LLM be d. We map com-
pressed context embeddings into the model space
using a lightweight projector Wψ(·), obtaining the
projected context-slot embeddings
˜E=W ψ(E)∈Rm×d,(1)
where mis the number of context slots and ˜Ej∈
Rddenotes the embedding assigned to the j-th slot.
We construct an input sequence of length nand
reserve a fixed set of context-slot positions r=
{r1, . . . , r m} ⊂ {1, . . . , n} . Let H(0,0)∈Rn×d
denote the initial token-wise hidden states fed into
the first transformer layer, defined as
H(0,0)
i=(˜Ej,ifi=r j,
Emb(u i),otherwise,(2)
where Emb(·) is the standard token embedding
function and {ui}n
i=1denotes the textual input se-
quence (e.g., the query and prompt tokens).
3

3.3 Selective LoRA on Context Slots
Inspired by multimodal models (Dong et al., 2024),
we treat context as a special modality. Specifically,
we design context slots and apply LoRA ( (Hu
et al., 2022)) updates only to context tokens while
keeping the prompt tokens on the frozen base path.
LetH(ℓ,t)∈Rn×ddenote the hidden states at
layer ℓand recursive step t. At the block level,
letF(ℓ)
θ(·)denote the ℓ-th transformer layer with
frozen base parameters θ, and let ∆F(ℓ)
ϕ(·)be the
LoRA-induced residual.
We define a binary broadcast mask Mr∈
{0,1}n×1, where (Mr)i=I[i∈r] . The result-
ingselective-LoRAlayer is defined as
A(ℓ)(H) =F(ℓ)
θ(H) +M r⊙∆F(ℓ)
ϕ(H),(3)
where ⊙denotes element-wise multiplication
broadcast along the hidden dimension.
3.4 Adaptive Recursive Alignment with Gate
We introduce anAdaptive Recursive Alignerthat
dynamically controls the refinement depth of con-
text slots across transformer layers. A subset of
layers is equipped with slot-wise gates, denoted
byLgate, while all other layers perform a single
refinement step. Let Ndenote the total number of
transformer layers.
Mandatory refinement.At every layer ℓ∈
{0, . . . , N−1} , an initial refinement step is always
applied:
H(ℓ+1,0)=A(ℓ)
H(ℓ,0)
.(4)
Gate-controlled recursive refinement.At gated
layers ( ℓ∈ L gate), we allow up to Tadditional
refinement steps indexed by t= 1, . . . , T . Let
H(ℓ+1,t−1)
r ∈Rm×ddenote the hidden states re-
stricted to context-slot positions at step t−1 . A
lightweight gating network predicts a slot-wise re-
finement decision:
g(ℓ,t)=σ
MLP(ℓ) 
H(ℓ+1,t−1)
r
,(5)
where g(ℓ,t)∈[0,1]m×1andσ(·) denotes the sig-
moid function.
Given the current state, we compute a candidate
refinement by reapplying the same selective-LoRA:
eH(ℓ+1,t)=A(ℓ) 
H(ℓ+1,t−1)
.(6)
We define stopgrad(·) as the stop-gradient op-
erator, which blocks gradient propagation throughits argument. During training, we adopt a straight-
through estimator (STE) (Bengio et al., 2013) to
enable discrete refinement decisions while pre-
serving gradient flow. Specifically, the hard gate
g(ℓ,t)
hard=I[g(ℓ,t)≥0.5] is used in the forward pass,
while gradients are backpropagated through the
continuous probability:
g(ℓ,t)
STE=g(ℓ,t)+ stopgrad
g(ℓ,t)
hard−g(ℓ,t)
.(7)
The context-slot update is then given by
H(ℓ+1,t)
r =H(ℓ+1,t−1)
r
+g(ℓ,t)
STE⊙
eH(ℓ+1,t)
r −H(ℓ+1,t−1)
r
,
(8)
while non-slot positions remain fixed across recur-
sive steps:
H(ℓ+1,t)
¯r =H(ℓ+1,0)
¯r .(9)
At inference time, we replace g(ℓ,t)
STEwith the
strictly binary gate g(ℓ,t)
infer=g(ℓ,t)
hard∈ {0,1}m×1,
so that only slots with g(ℓ,t)
infer= 1are further refined,
yielding an exact conditional update.
3.5 Three-Stage Training Strategy
We adopt a three-stage training strategy that pro-
gressively equips the model with (i) alignment to
compressed embeddings, (ii) task grounding under
compressed-slot inputs, and (iii) adaptive refine-
ment via the gate.
Stage I: Reconstruction Alignment Pretraining.
We first warm up the model to interpret compressed
embeddings by reconstructing the original text.
Specifically, we optimize the projector parameters
ψinWψ(·)and the selective LoRA parameters ϕ,
while keeping the base LLM frozen. Given an input
where the designated slot positions Rare filled with
compressed embeddings ˜E, the model is trained to
autoregressively reconstruct the context yusing the
negative log-likelihood (NLL) objective:
Lrec=−|y|X
i=1logp θ
yi|H(0,0), y<i
.(10)
Stage II: Task-Grounded RAG Finetuning.
Starting from Stage I, we finetune the model on
downstream RAG tasks so that it can answer
queries by grounding on compressed slot inputs.
During this stage, we continue updating ψandϕ
but disable the gate, so that each layer performs
4

only the mandatory single-step refinement. The
training target is the ground-truth answer and is op-
timized using the same NLL objective as in Stage I.
Stage III: Gate-Aware Adaptive Finetuning.In
the final stage, we enable the gating mechanism and
train the full adaptive recursive aligner. We jointly
optimize the projector parameters ψ, the selective
LoRA parameters ϕ, and the gate parameters. The
forward pass follows § 3.4, using STE-based gat-
ing during training and hard binary gating at infer-
ence. The optimization in this stage still uses NLL
loss. Through training in this stage, ArcAligner is
trained more thoroughly and can learn slot refine-
ment depth according to task requirements, thereby
supporting more effective inference under strong
compression.
4 Experiments
4.1 Implementation Details
Base Models.We useMistral-7B-Instruct-
v0.2(Jiang et al., 2023a) as the base language
model in all main experiments. We useSFR-
Embedding-Mistral(Meng et al., 2024) as the sen-
tence encoder to encode passages into dense vec-
tors. All parameters of the base model are frozen
during training, except for the lightweight modules
introduced by our method.
Passage Segmentation.We segment retrieved
passages at the sentence level using spaCy (Honni-
bal et al., 2020). These sentence-based segments
are used as the basic units for compression.
Training Data.Our training follows a three-
stage training strategy.Stage I (Pretraining).Pre-
training data is built from a large-scale Wikipedia
corpus. We follow standard preprocessing by seg-
menting the dump into fixed-length passages, then
sample about 200k passages and train for one
epoch. This stage learns a robust alignment be-
tween embeddings and the language model’s repre-
sentation space.Stage II & III (RAG Finetuning).
Both stages are trained on the HotpotQA (Yang
et al., 2018) dataset, using approximately 90K train-
ing samples. Stage II enables only the projector
and LoRA parameters, while Stage III further en-
ables the gate for adaptive refinement. Details can
be found in Appendix C.
4.2 Datasets and Evaluation Metrics
Evaluation Datasets.We evaluate effectiveness
and generalization on multi-hop and open-domainQA benchmarks. For multi-hop QA, we use
2WikiMultiHopQA (Ho et al., 2020) and Hot-
potQA (Yang et al., 2018); for open-domain
QA, we use NaturalQA (Kwiatkowski et al.,
2019), PopQA_longtail (Mallen et al., 2023), Trivi-
aQA* (Joshi et al., 2017), and WebQuestions (Be-
rant et al., 2013), whereTriviaQA*denotes a sub-
sampled evaluation set consisting of 500 randomly
selected questions. For each dataset, we retrieve
the top-20 passages withContriever(Izacard et al.,
2021) and rerank them withRankZephyr(Pradeep
et al., 2023). The reranked top passage is used as
the context. All datasets share the same retrieval
and inference settings, with details in Appendix A.
Evaluation Metrics.We report Exact Match
(EM),F1, and Accuracy (Acc). Following prior
work (Asai et al., 2024; Mallen et al., 2023), we
usenon-strictEM, counting a prediction as correct
if it contains the gold answer. F1 is computed as
token-level overlap with the gold answer, comple-
menting EM since longer outputs can increase EM
but lower F1. We additionally report LLM-judged
accuracy; details are in Appendix B.
4.3 Baselines
We compare against representative baselines:
(1)Naive, which answers the question without any
retrieval context; (2)StandardRAG, the classic
retrieve-then-read setup that concatenates retrieved
passages with the query; (3)xRAG(Cheng et al.,
2024), which replaces text passages with projected
dense retrieval embeddings for extreme compres-
sion;(4)COCOM(Rau et al., 2025), which injects
learned context embeddings into the decoder; and
(5)LLMLingua-2(Pan et al., 2024), which com-
presses retrieved contexts by retaining only infor-
mative tokens before generation.
4.4 Main Results
Table 1 and 2 report the main results. Some key
findings are as follows.
(1) Compression vs. No Compression.Stan-
dardRAG achieves the highest EM and accuracy
on most datasets, setting a performance upper
bound by providing the full textual context. In
contrast, embedding-based compression methods
(e.g., xRAG and COCOM) show a notable drop,
highlighting the limitations of current compression
techniques. Our approach offers a trade-off and
performs well on multi-hop and long-tail tasks.
(2) Language space vs. Embedding space.
5

Method Comp. 2WikiMultiHopQA HotpotQA NaturalQA
rateEM F1 Acc EM F1 Acc EM F1 Acc
w/o retrieval
Naive∇–22.20 14.63 19.40 21.20 13.39 23.20 27.20 16.50 28.25
w/ retrieval
StandardRAG△–31.60 18.55 22.60 37.40 19.72 36.60 43.49 23.56 44.57
xRAG×12848.808.66 22.00 30.40 6.00 27.8044.133.39 34.04
COCOM×4 26.80 29.38 27.20 25.20 32.45 31.18 36.59 39.54 41.50
COCOM×16 27.80 31.02 28.80 24.40 31.77 31.20 35.1841.04 41.94
LLMLingua-2×3 33.40 15.78 23.2031.4016.41 29.40 37.73 18.98 36.95
ArcAligner (Ours)×24 31.8036.00 33.4026.6034.92 33.4031.72 37.82 37.04
Table 1: EM/F1/Acc on 2WikiMultiHopQA, HotpotQA, and NaturalQA. Comp. rate denotes the context compression
ratio. Best and second-best scores are highlighted inboldand underlined , respectively.Italicdenote excluded
reference settings: Naive∇(no-retrieval) and StandardRAG△(full-context).
Method Comp. PopQA_longtail TriviaQA* WebQuestions
rateEM F1 Acc EM F1 Acc EM F1 Acc
w/o retrieval
Naive∇–20.01 8.90 13.30 56.60 33.25 44.80 38.34 25.32 45.96
w/ retrieval
StandardRAG△–45.96 23.10 33.95 69.40 35.63 60.20 39.67 23.96 45.23
xRAG×128 33.95 4.94 28.9569.8011.53 55.8055.865.14 47.79
COCOM×4 24.59 25.15 24.37 60.80 63.60 58.80 42.72 36.43 47.44
COCOM×16 24.16 25.79 24.59 58.00 61.91 58.40 37.84 40.45 48.77
LLMLingua-2×340.1715.6631.0966.60 30.62 51.80 37.75 21.04 42.77
ArcAligner (Ours)×24 30.3832.8030.74 62.8065.65 62.8035.1443.1946.80
Table 2: EM/F1/Acc on PopQA_longtail, TriviaQA*, and WebQuestions. Comp. rate denotes the context compres-
sion ratio. Best and second-best scores are highlighted inboldand underlined , respectively.Italicdenote excluded
reference settings: Naive∇(no-retrieval) and StandardRAG△(full-context).
Method HotpotQA NaturalQA TriviaQA*
EM F1 Acc EM F1 Acc EM F1 Acc
ArcAligner (Ours) 26.60 34.92 33.40 31.72 37.82 37.0462.80 65.65 62.80
w/o Recursion 26.00 34.48 32.60 31.39 37.64 36.90 62.00 65.17 62.00
w/o Gate (Max Loop)27.2031.69 31.00 31.25 32.57 32.7464.0062.66 61.80
w/o LoRA & Recursion 25.00 32.02 32.00 31.58 37.50 36.76 63.40 66.2062.20
Table 3: Ablation results on HotpotQA, NaturalQA, and TriviaQA*. We report non-strict EM, F1, and LLM-judged
accuracy (Acc).w/o Recursiondisables gate-controlled refinement loops at inference.w/o Gate (Max Loop)
disables the gate while keeping a fixed number of refinement loops for all context slots.w/o LoRA & Recursion
disables the selective-adaptation weights (LoRA) and refinement loops, but keeps the context slots.
Overall, language-based methods showed mixed re-
sults, suggesting they require unique and complex
designs to handle different tasks. Embedding-based
methods outperformed LLMlingua-2 overall and
demonstrated stronger robustness across various
tasks. Our ArcAligner exhibits even greater robust-
ness and learn more task-oriented information.
(3) Shallow Alignment vs. Deep alignment
Shallow alignment methods such as xRAG exhibit
high EM but low F1, resulting in excessively long
outputs, poor task orientation, and even worse per-formance. In contrast, deep alignment methods
like COCOM and ArcAligner not only ensure in-
formation availability but also provide greater task
orientation.
4.5 Ablation Study
We conduct ablations on two core components of
ArcAligner, with results reported in Table 3.
Disabling refinement loops (w/o Recursion) con-
sistently degrades performance, indicating that re-
cursive refinement provides larger gains than a sin-
6

Figure 3: Accuracy across different backbone language
models on four QA datasets. Grouped bars compare
Naive,StandardRAG, andArcAligner(Ours)under
two backbones:Llama-3.1-8B-InstructandMistral-
7B-Instruct-v0.2.
gle pass. Further removing the gating mechanism
while keeping a fixed number of refinement loops
for all context slots (w/o Gate (Max Loop)) leads
to more pronounced drops on NaturalQA and Trivi-
aQA*, suggesting that indiscriminate refinement is
suboptimal and that the gate is crucial for selecting
which context slots to refine.
Finally, removing selective adaptive weights
while retaining projection-generated context slots
(w/o LoRA & Recursion) performs worse than the
full model, highlighting the importance of deep
semantic alignment.
Overall, layer-wise selective adaptation and
learned gated recursion are key to aligning com-
pressed context embeddings.
4.6 Analysis Across Different Backbones
Figure 3 reports the accuracy of different methods
under two backbone language models on four rep-
resentative QA datasets. Across datasets, the rela-
tive performance ordering amongNaive,Standard-
RAG, and ArcAligner remains largely consistent
when switching between Llama-3.1 and Mistral-
v0.2, although the absolute accuracy varies across
backbones. In particular,ArcAlignerexhibits com-
parable behavior under both backbone models and
maintains competitive performance across datasets.
These results suggest that the observed method-
level trends are not specific to a particular backbone
language model.
4.7 Reconstruction Ability After Pretraining
To investigate reconstruction pre-training, we com-
pare the reconstruction perplexity (PPL) on the test
set after the pre-training stage.
Figure 4: Reconstruction perplexity (PPL) after the pre-
training stage, evaluated on a held-out pretraining test
set with 10k samples. Results are reported for two
backbone language models,Llama-3.1-8B-Instruct
andMistral-7B-Instruct-v0.2. Lower PPL indicates
better reconstruction fidelity.
Figure 4 reports the results.Baserepresents the
model directly reciting the context without com-
pression, and can be considered as the upper limit
of reconstruction performance.xRAGyields sub-
stantially higher PPL for both backbone language
models, indicating that directly applying retrieval-
based compression markedly reduces reconstruc-
tion fidelity during pre-training. Our model attains
lower PPL thanxRAGandw/o LoRA, showing that
the proposed alignment design mitigates the degra-
dation introduced by compression. Removing the
deep alignment module (w/o LoRA) also lowers
PPL relative toxRAG, but remains significantly
worse thanBase. This suggests that shallow projec-
tion alone is insufficient for the LLM to interpret
compressed embeddings.
4.8 Analysis of Gate Behavior
To investigate the layer-by-layer behavior of gates,
we compute the average number of refinement
loops triggered per layer.
As shown in Figure 5, the gates exhibit a con-
Figure 5: Gate behavior analysis across layers. The fig-
ure shows the token-level average number of refinement
loops triggered by the gate at each layer evaluated on
four datasets.
7

sistent pattern across all four datasets: most layers
operate with a single forward pass, while a few lay-
ers selectively activate additional refinement loops.
This suggests that the gate allocates extra compu-
tation at specific stages of the network, rather than
uniformly increasing depth across layers. Notably,
higher loop counts are concentrated in the early
and late layers, whereas the middle layers remain
relatively stable, approaching a single forward pass.
We hypothesize that early layers are crucial for
embedding alignment, while later layers support
semantic fusion, which benefits QA. More inter-
pretable mechanisms remain an important direction
for further study.
4.9 4-way Error Category Analysis
To better understand where ArcAligner improves,
we partition questions into four categories based on
whether theNaiveandStandardRAGsucceed:TT,
TF,FT, andFF. The results are shown in Figure 6.
We observe several consistent trends.
First, in theTTcategory, where both the naive
prompt and StandardRAG succeed, ArcAligner
achieves strong performance across all datasets.
This indicates that ArcAligner does not introduce
harmful behavior on simpler or well-supported
questions. In theFTcategory, ArcAligner consis-
tently outperformsxRAG, suggesting that its com-
pression preserves useful information better than
comparable approaches.
Notably, ArcAligner shows clear advantages in
theTFcategory, which typically corresponds to
cases where retrieved passages are insufficient or
Figure 6:4-way category analysison four datasets. We
partition questions into four groups based on whether
naive promptingandstandard RAGanswer correctly:
TT,TF,FT, andFF, whereT/Fdenote correct/incor-
rect predictions. We report accuracy (Acc) of xRAG
and ArcAligner within each category. Category size is
shown under each label (e.g.,TTwithn=41).
Figure 7: Performance under different compression
ratios on 2WikiMultiHopQA and PopQA_longtail.
EM/F1 scores are used as evaluation metrics.
contain misleading signals. Compared withxRAG,
ArcAligner produces a much higher proportion of
correct responses, indicating that it can better com-
pensate for imperfect retrieval rather than merely
relying on retrieved content. TheFFcategory is the
most challenging. ArcAligner still yields a small
number of correct cases, which we attribute to the
later training stages that encourage the LLM to at-
tend more effectively to informative signals in the
compressed context.
Overall, benefiting from adaptive recursion and
three-stage training, ArcAligner improves perfor-
mance and exhibits robustness for RAG tasks.
4.10 Effect of Compression Ratio
We further analyse the impact of different compres-
sion ratios on ArcAligner. For each ratio, we train a
separate model with the same training settings, and
vary only the sentence segmentation granularity
that controls the ratio: 12 ×is obtained by splitting
each sentence into two segments, whereas 72 ×is
obtained by merging three consecutive sentences
into one, with all other settings fixed.
As shown in Figure 7, at a higher compression ra-
tio of 72x, performance decreased slightly, but not
significantly. This indicates that the compressed
representation retains the key evidence required
for multi-hop inference. Meanwhile, we found
some performance improvement as we reduce the
compression ratio, suggesting that a smaller com-
pression ratio can retain more key information and
mitigate excessive information loss. This suggests
that in practice, we need to flexibly adjust the com-
pression ratio to achieve a trade-off between per-
formance and efficiency.
5 Conclusion
In this paper, we propose ArcAligner, a parameter-
efficient framework that enhances the usability of
compressed context embeddings for RAG. By in-
8

troducing an adaptive recursive alignment mecha-
nism with gating, ArcAligner ensures deep seman-
tic alignment between compressed retrieval signals
and the language model’s internal representations.
Extensive experiments on knowledge-intensive QA
benchmarks show that ArcAligner consistently out-
performs existing compression baselines, espe-
cially in multi-hop and long-tail QA tasks, making
it a valuable tool for efficient RAG systems.
Limitations
We discuss the limitations of ArcAligner as follows.
Our current setup assumes a fixed context slot in-
terface and a specific compression pipeline, and
we do not systematically study how design choices
such as passage segmentation, compression ratio,
or slot budget affect the accuracy–efficiency trade-
off. We also focus on a constrained retrieval setting
and do not fully explore Top- Kmulti-document ev-
idence, where complementary information across
documents can be crucial; how to allocate slots and
refinement budget across multiple retrieved pas-
sages remains open. Finally, ArcAligner relies on
hard, slot-wise gating with a maximum recursion
depth Tat selected layers; while efficient in typical
cases, the gate placement and Tare tuned heuris-
tically and may be suboptimal under strict com-
pute/latency constraints, motivating future work on
budget-aware gating or adaptive stopping.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Yoshua Bengio, Nicholas Léonard, and Aaron Courville.
2013. Estimating or propagating gradients through
stochastic neurons for conditional computation.
arXiv preprint arXiv:1308.3432.
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on freebase from
question-answer pairs. InProceedings of the 2013
conference on empirical methods in natural language
processing, pages 1533–1544.
Shaoshen Chen, Yangning Li, Zishan Xu, Yongqin Zeng,
Shunlong Wu, Xinshuo Hu, Zifei Shan, Xin Su, Jiwei
Tang, Yinghui Li, and 1 others. 2025. Dast: Context-
aware compression in llms via dynamic allocationof soft tokens. InFindings of the Association for
Computational Linguistics: ACL 2025, pages 20544–
20552.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-
Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan
Zhao. 2024. xrag: Extreme context compression
for retrieval-augmented generation with one token.
Advances in Neural Information Processing Systems,
37:109487–109516.
Chenlong Deng, Zhisong Zhang, Kelong Mao, Shuaiyi
Li, Xinting Huang, Dong Yu, and Zhicheng Dou.
2025. A silver bullet or a compromise for full at-
tention? a comprehensive study of gist token-based
context compression. InProceedings of the 63rd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 4861–
4879.
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao,
Bin Wang, Linke Ouyang, Xilin Wei, Songyang
Zhang, Haodong Duan, Maosong Cao, and 1 oth-
ers. 2024. Internlm-xcomposer2: Mastering free-
form text-image composition and comprehension
in vision-language large model.arXiv preprint
arXiv:2401.16420.
Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen,
and Furu Wei. In-context autoencoder for context
compression in a large language model. InThe
Twelfth International Conference on Learning Repre-
sentations.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Matthew Honnibal, Ines Montani, Sofie Van Lan-
deghem, and Adriane Boyd. 2020. spaCy: Industrial-
strength Natural Language Processing in Python.
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski,
Bruna Morrone, Quentin De Laroussilhe, Andrea
Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.
Parameter-efficient transfer learning for nlp. InIn-
ternational conference on machine learning, pages
2790–2799. PMLR.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models.ICLR, 1(2):3.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
9

Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
conference of the european chapter of the association
for computational linguistics: main volume, pages
874–880.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023a. Mistral 7b.Preprint,
arXiv:2310.06825.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023b. Llmlingua: Compressing
prompts for accelerated inference of large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 13358–13376.
Yi Jiang, Lei Shen, Lujie Niu, Sendong Zhao, Wenbo
Su, and Bo Zheng. 2025a. Qagent: A modular search
agent with interactive query understanding.arXiv
preprint arXiv:2510.08383.
Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, and
Bing Qin. 2025b. GainRAG: Preference alignment in
retrieval-augmented generation through gain signal
synthesis. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 10746–10757, Vi-
enna, Austria. Association for Computational Lin-
guistics.
Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang,
Lizhe Zhang, Yan Liu, and Bing Qin. 2025c. Co-
coa: Collaborative chain-of-agents for parametric-
retrieved knowledge synergy.arXiv preprint
arXiv:2508.01696.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang Woo Lee, Minjoon Seo, Jung Woo Ha,
and Jinwoo Shin. 2024. Sure: Summarizing re-
trievals using answer candidates for open-domain qa
of llms. In12th International Conference on Learn-
ing Representations, ICLR 2024.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning:
Optimizing continuous prompts for generation. In
Proceedings of the 59th Annual Meeting of the Asso-
ciation for Computational Linguistics and the 11th
International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), pages 4582–
4597.
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 6342–6353, Singa-
pore. Association for Computational Linguistics.
Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng
He, Huanxuan Liao, Haoran Que, Zekun Wang,
Chenchen Zhang, Ge Zhang, Jiebin Zhang, and
1 others. 2025. A comprehensive survey on
long context language modeling.arXiv preprint
arXiv:2503.17407.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822.
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming
Xiong, Yingbo Zhou, and Semih Yavuz. 2024.
Sfrembedding-mistral: enhance text retrieval with
transfer learning.Salesforce AI Research Blog, 3:6.
Alliot Nagle, Adway Girish, Marco Bondaschi, Michael
Gastpar, Ashok Vardhan Makkuva, and Hyeji Kim.
2024. Fundamental limits of prompt compression:
A rate-distortion framework for black-box language
models.Advances in Neural Information Processing
Systems, 37:94934–94970.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin
Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor
Rühle, Yuqing Yang, Chin-Yew Lin, and 1 others.
2024. Llmlingua-2: Data distillation for efficient and
faithful task-agnostic prompt compression. InACL
(Findings).
Hippolyte Pilchen, Edouard Grave, and Patrick Pérez.
2025. Arc-encoder: learning compressed text repre-
sentations for large language models.arXiv preprint
arXiv:2510.20535.
10

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023. Rankzephyr: Effective and robust zero-
shot listwise reranking is a breeze!arXiv preprint
arXiv:2312.02724.
David Rau, Shuai Wang, Hervé Déjean, Stéphane Clin-
chant, and Jaap Kamps. 2025. Context embeddings
for efficient answer generation in retrieval-augmented
generation. InProceedings of the Eighteenth ACM
International Conference on Web Search and Data
Mining, WSDM ’25, page 493–502, New York, NY ,
USA. Association for Computing Machinery.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, and 1 others. 2023. Llama: Open and effi-
cient foundation language models.arXiv preprint
arXiv:2302.13971.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp:
Improving retrieval-augmented lms with context com-
pression and selective augmentation. InThe Twelfth
International Conference on Learning Representa-
tions.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
A Datasets
Here, we introduce in detail the datasets we used,
which are seven datasets on four tasks.
2WikiMultiHopQA(Ho et al., 2020) andHot-
potQA(Yang et al., 2018): Both are multi-hop
question answering benchmarks constructed from
Wikipedia articles. To control experimental cost,
we follow prior work (Trivedi et al., 2023; Kim
et al., 2024) and evaluate on the released sub-
sampled splits, each containing 500 questions
drawn from the original validation set.
NaturalQA(Kwiatkowski et al., 2019): A large-
scale benchmark designed to support comprehen-sive question answering. The questions are col-
lected from real Google search queries, and the
answers are annotated as text spans from corre-
sponding Wikipedia articles by human annotators.
PopQA-longtail(Mallen et al., 2023): An
open-domain QA dataset targeting long-tail factual
knowledge. The questions are entity-centric and
automatically generated from Wikidata knowledge
triples using relation-specific templates. Entity
popularity is quantified via Wikipedia page views,
enabling controlled evaluation on low-popularity
(long-tail) facts.
TriviaQA*(Joshi et al., 2017): A collection
of open-domain trivia questions paired with an-
swer annotations, originally sourced from online
trivia websites. We useTriviaQA*to denote a sub-
sampled evaluation set of 500 randomly selected
questions.
WebQuestions(Berant et al., 2013): A factoid
question answering dataset constructed from real
user queries issued to the Google Suggest API,
where answers correspond to specific entities in
Freebase.
Dataset statistics are summarized in Table 4
Task Type Datasets # Samples
Multi-HopQA2WikiMultiHopQA 500
HotpotQA 500
OpenQANaturalQA 3610
PopQA_longtail 1399
TriviaQA* 500
WebQuestions 2032
Table 4: Description of tasks and evaluation datasets.
B Evaluation
Evaluation Metrics.We report three metrics:
Exact Match (EM), F1, and Accuracy (Acc).
For EM and F1, we follow the standard evalua-
tion protocol used in prior RAG work. Exact Match
measures whether the model’s prediction contains
the gold answer. Following Self-RAG (Asai et al.,
2024) and When Not to Trust LMs (Mallen et al.,
2023), we adopt anon-strictEM metric, where a
prediction is considered correct if it includes the
gold answer, rather than requiring an exact string
match. F1 measures the token-level overlap be-
tween the predicted answer and the gold answer.
As noted in prior work, longer responses may
artificially improve EM due to higher matching
probability, while often reducing F1 due to the
11

Hyperparameter Assignment
optimizer AdamW
learning rate 2.0e-4
lr scheduler type linear
warmup ratio 0.03
weight decay 0.0
LoRA r 128
LoRA alpha 32
LoRA dropout 0.05
max loops 3
loop layers all
epochs 1
batch size 8
gradient accumulation steps 8
num GPUs 4
max train samples 200,000
Table 5: Hyperparameters for Stage I.
inclusion of irrelevant content. Therefore, EM and
F1 provide complementary perspectives on answer
quality.
In addition, we report Accuracy (Acc) based on
large language model judgment. Specifically, we
useQwen3-30B-A3B-Instruct-2507(Yang et al.,
2025) as an evaluator to assess whether the gener-
ated answer correctly addresses the question given
the reference answer. The evaluator is prompted to
produce a binary correctness judgment, which is
then averaged over the dataset to compute Acc.
C Implementation Details
C.1 Hyperparameters
For all experiments, we adopt instruction-tuned
variants of the base language models. All train-
ing and evaluation are conducted on a cluster of4
NVIDIA H20-3e GPUs.
Unless otherwise specified, all models are
trained using the AdamW optimizer with linear
learning rate scheduling. We report the detailed hy-
perparameter configurations for different training
stages in Tables 5 and 6.
C.2 Prompts for Stage I
The list of instructions for paraphrase pretrain-
ing is shown in Table 7. They present the same
meaning with natural language variance. Follow-
ing prior work on embedding-based context com-
pression, these paraphrase-style instructions are
adapted from the instruction templates used inHyperparameter Assignment
optimizer AdamW
learning rate 2.0e-5
lr scheduler type linear
warmup ratio 0.03
weight decay 0.0
LoRA r 128
LoRA alpha 32
LoRA dropout 0.05
max loops 3
loop layers all
epochs 1
batch size 8
gradient accumulation steps 2
num GPUs 4
max train samples 90,447
Table 6: Hyperparameters for Stage II & III.
xRAG (Cheng et al., 2024).
C.3 Prompt for Stage II & III
During task fine-tuning, we represent the com-
pressed background document using a short in-
struction line composed of repeated background
placeholders. This line is inserted at a designated
location in the prompt and marks the positions of
the injected compressed segments:
Refer to the background document:[B]
[B] . . . [B]
Question:[Q]
The number of background placeholders is set
toN= max(1,|sentences|) , and the instruction
line is instantiated withNcopies of [B].
D Algorithm
This section provides the detailed pseudocode of
the ArcAligner forward computation, correspond-
ing to the description in Section 3. The algorithm
specifies how gated recursive updates are applied
at selected layers and context-slot positions during
both training and inference, as detailed in Algo-
rithm 1.
E Case Studies
We present several representative cases to further
analyze the behavior of our method (Figures 8–10).
12

Instruction intent Template (with placeholders)
Equivalence statementBackground: [B]. This is equivalent to: [T].
Direct rewriteRewrite the background in your own words: [B]→[T].
Restatement requestProvide a restatement of the background: [B]. Return: [T].
Paraphrase-as-question[B] is a paraphrase of what? Answer with: [T].
Two-formulation alignmentThese two expressions convey the same meaning: (1) [B] (2) [T].
Minimal-output constraintRestate [B] using a single sentence. Output only [T].
Table 7: Stage-I instruction templates. [B] and [T] denote the compressed background representation and the
target textual reconstruction, respectively. We sample from multiple intent categories to diversify supervision while
keeping a consistent I/O format.
Notation.We use traj to visualize the per-slot
recursion depth determined by the gate. A token L
corresponds to one recursion step, and the number
of repeated L’s matches the loop count reported
inloops . Thus, L0. means the default single
pass (loop count = 1), whereas LLLmeans three
recursive passes (loop count= 3). Algorithm 1ArcAligner Forward
Require: Initial states H(0,0), blocks {A(ℓ)}N−1
ℓ=0,
gated layers Lgate, max recursion T, mode ∈
{train,infer}
Ensure:Final statesH(N,0)
H|r: context-slot positions; H|¯r: non-slot
positions
1:forℓ= 0toN−1do
2:H(ℓ+1,0)← A(ℓ)(H(ℓ,0))
3:ifℓ∈ L gatethen
4:fort= 1toTdo
5:g←σ(MLP(ℓ)(H(ℓ+1,t−1)|r))
6:ifmode=trainthen
7:g←g+ stopgrad(I[g≥
0.5]−g)
8:else
9:g←I[g≥0.5]
10:end if
11: eH← A(ℓ)(H(ℓ+1,t−1))
12:H(ℓ+1,t)|r←H(ℓ+1,t−1)|r+g⊙
(eH|r−H(ℓ+1,t−1)|r)
13:H(ℓ+1,t)|¯r←H(ℓ+1,0)|¯r
14:end for
15:H(ℓ+1,0)←H(ℓ+1,T)
16:end if
17:end for
18:returnH(N,0)
13

Question.In what country is Toronto Northwest?
Compressed Sentence Passages.
1. was redistributed between Davenport, Spadina, Trinity and York West ridings.
2.Toronto Northwest was a federal electoral district represented in the House of Commons of
Canada from 1925 to 1935.
3. It was located in the city of Toronto in the province of Ontario.
4. This riding was created in 1924 from parts of Parkdale, Toronto North and York South ridings.
5.It consisted of the part of the city of Toronto north of Bloor Street, west of Bathurst St. and east
of the Northern Division of the Canadian National Railway.
Answer.Canada.
Routing Behavior.At layer 31:
traj= [L0.,LLL,L0.,L0.,L0.]
The answer evidence appears in thesecond sentence, which is repeatedly updated via recursive routing
(LLL).
Figure 8: Case 1: the answer evidence appears in the second sentence, which is repeatedly updated via recursive
routing.
Question.What sport does Radik Zhaparov play?
Compressed Sentence Passages.
1.Radik Zhaparov (born February 29, 1984) is a Kazakh ski jumper who has competed since
2003.
2.At the 2006 Winter Olympics in Turin, he finished 11th in the team large hill and 26th in the
individual normal hill events.
3.At the FIS Nordic World Ski Championships, Zhaparov has finished 11th in team events three
times and 24th in the individual normal hill events.
4. Zharparov’s best individual World Cup finish was 11th in a large hill event in Finland in 2007.
5. His best individual career finish was second in an FIS Cup normal.
Answer.ski jumping.
Routing Behavior.At layer 31:
traj= [LLL,L0.,L0.,L0.,L0.]
The recursive routing ( LLL, loop count = 3) is assigned to the first slot, which corresponds to thefirst
sentencecontaining the explicit evidence.
Figure 9: Case 2: the answer evidence appears in the first sentence, which receives repeated recursive updates ( LLL)
at layer 31.
14

Question.What nationality is the director of filmAstronauts Gone Wild?
Compressed Sentence Passages.
1.Astronauts Gone Wildis a 2004 documentary video produced and directed by Bart Sibrel, a
Nashville, Tennessee-based video maker.
2.Sibrel made this video as a follow-up to his 2001 videoA Funny Thing Happened on the Way to
the Moon.
3. The title of the presentation is a wordplay on theGirls Gone Wildvideo series.
4. InAstronauts Gone Wild, Sibrel confronts nine Apollo . . .
Answer.American.
Routing Behavior.At layer 14:
traj= [LLL,L0.,L0.,L0.]
The recursive routing ( LLL, loop count = 3) is assigned to the first slot, which aligns with thefirst
sentencedescribing the director and location cues used to infer nationality.
Figure 10: Case 3: the director evidence appears in the first sentence, which receives repeated recursive updates
(LLL) at layer 14.
15
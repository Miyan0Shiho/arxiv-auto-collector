# Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference

**Authors**: Wenxuan Xie, Yujia Wang, Xin Tan, Chaochao Lu, Xia Hu, Xuhong Wang

**Published**: 2026-02-10 17:42:31

**PDF URL**: [https://arxiv.org/pdf/2602.10021v1](https://arxiv.org/pdf/2602.10021v1)

## Abstract
The integration of extensive, dynamic knowledge into Large Language Models (LLMs) remains a significant challenge due to the inherent entanglement of factual data and reasoning patterns. Existing solutions, ranging from non-parametric Retrieval-Augmented Generation (RAG) to parametric knowledge editing, are often constrained in practice by finite context windows, retriever noise, or the risk of catastrophic forgetting. In this paper, we propose DRIFT, a novel dual-model architecture designed to explicitly decouple knowledge extraction from the reasoning process. Unlike static prompt compression, DRIFT employs a lightweight knowledge model to dynamically compress document chunks into implicit fact tokens conditioned on the query. These dense representations are projected into the reasoning model's embedding space, replacing raw, redundant text while maintaining inference accuracy. Extensive experiments show that DRIFT significantly improves performance on long-context tasks, outperforming strong baselines among comparably sized models. Our approach provides a scalable and efficient paradigm for extending the effective context window and reasoning capabilities of LLMs. Our code is available at https://github.com/Lancelot-Xie/DRIFT.

## Full Text


<!-- PDF content starts -->

Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model
Framework for Efficient Long-Context Inference
Wenxuan Xie1,2,3, Yujia Wang4, Xin Tan1, Chaochao Lu1, Xia Hu1, Xuhong Wang1,*
1Shanghai Artificial Intelligence Laboratory, Shanghai, China
2Fudan University, Shanghai, China
3Shanghai Innovation Institute, Shanghai, China
4Tongji University, Shanghai, China
wxxie25@m.fudan.edu.cn,wangxuhong@pjlab.org.cn
Abstract
The integration of extensive, dynamic knowl-
edge into Large Language Models (LLMs) re-
mains a significant challenge due to the inher-
ent entanglement of factual data and reason-
ing patterns. Existing solutions, ranging from
non-parametric Retrieval-Augmented Gener-
ation (RAG) to parametric knowledge edit-
ing, are often constrained in practice by finite
context windows, retriever noise, or the risk
of catastrophic forgetting. In this paper, we
proposeDRIFT, a novel dual-model architec-
ture designed to explicitly decouple knowledge
extraction from the reasoning process. Un-
like static prompt compression, DRIFT em-
ploys a lightweight knowledge model to dy-
namically compress document chunks into im-
plicit fact tokens conditioned on the query.
These dense representations are projected into
the reasoning model’s embedding space, re-
placing raw, redundant text while maintain-
ing inference accuracy. Extensive experiments
show that DRIFT significantly improves per-
formance onlong-context tasks, outperform-
ing strong baselines among comparably sized
models. Our approach provides a scalable
and efficient paradigm for extending the effec-
tive context window and reasoning capabili-
ties of LLMs. Our code is available at https:
//github.com/Lancelot-Xie/DRIFT.
1 Introduction
The applicability of Large Language Models
(LLMs) to knowledge-intensive tasks is limited
by the static nature of their pre-training data. To
address this limitation, prior work has explored
two complementary strategies.The first focuses on
augmenting the input context, while the second
emphasizes knowledge parameter editing.
Traditional input augmentation via RAG or long-
context prompting is increasingly constrained by
*Corresponding author.the “retriever’s ceiling” and the quadratic computa-
tional costs of processing long sequences. Neural
compression methods (e.g., COCOM (Rau et al.,
2024), C3 (Liu and Qiu, 2025)) attempt to mitigate
this by distilling text into latent representations;
however, as they primarily focus on static com-
pression, task-critical information relevant to the
query is frequently lost. Conversely, internaliz-
ing knowledge through direct parametric updates,
such as fine-tuning or knowledge editing, often
disrupts the inherent coupling between a model’s
internal knowledge and its reasoning logic, while
also risking catastrophic forgetting. While modular
approaches like MLP Memory (Wei et al., 2025) of-
fer a plug-and-play alternative, they remain bound
to pre-indexed resources and struggle to handle
instantaneous, unseen long-context inputs in real-
time.
To address these challenges, we introduce
DRIFT, a dual-model architecture that decouples
context processing from core reasoning. In this
framework, a lightweightknowledge modelex-
tracts query-relevant information from document
chunks and compresses it into high-densityim-
plicit fact tokenswithin a latent space. These
tokens serve as concise, knowledge-rich represen-
tations that are projected into a largerreasoning
model’sembedding space. The reasoning model
can perform sophisticated inference efficiently
based on this compact factual context instead of raw
text, even in long-context or knowledge-intensive
scenarios. By delegating the processing of redun-
dant background knowledge to the knowledge mod-
ule, this design allows the reasoning model to re-
main unburdened by raw context, focusing instead
on "clean" and deep inference through the distilled
factual representations.
Our main contributions are as follows:
•A Decoupled Inference Paradigm for Large
Language Models:We propose a dual-model
1arXiv:2602.10021v1  [cs.CL]  10 Feb 2026

framework that decouples knowledge extrac-
tion from reasoning. A lightweight knowl-
edge model encodes query-relevant informa-
tion into compact fact tokens, which are con-
sumed by a larger reasoning model for in-
ference. Compared with directly processing
long contexts, DRIFT improves performance
while substantially reducing inference latency,
achievingan average 7× speedup on 256k-
token documents.
•Expanding Effective Context Window with
High-Ratio Compression:By encoding ex-
tensive textual knowledge into compact fact
tokens, our framework significantly extends
the model’s usable context window. Specifi-
cally, ourDRIFTmodel (based on Mistral 7B)
achieves a32 ×compression ratiowhileim-
proving accuracy from 20.87% to 29.22%
on the LongBench v2 benchmark, demon-
strating superior reasoning capabilities with
substantially reduced time overhead. Further-
more, even under more aggressive compres-
sion settings (64 ×and128 ×), DRIFT re-
mains highly competitive, indicating strong
robustness to extreme compression ratios.
•Comprehensive Empirical Analysis and Re-
source Contribution:We construct a large-
scale Document–QA–Evidence dataset with
fine-grained supervision, comprising over
300Kinstances and documents ranging from
1Kto8Ktokens. We further conduct exten-
sive ablation studies and quantitative evalua-
tions, rigorously validating the framework and
demonstrating its robustness.
2 Related Work
2.1 Prompt Compression
Directly feeding new knowledge as context into
Large Language Models (LLMs) is constrained
by limited context windows and incurs significant
overhead in both memory and computation. To
mitigate these issues, various prompt compression
methods have been proposed, which generally fall
into two paradigms:Hard CompressionandSoft
Compression.
Hard Compression (Token Selection).Hard
compression methods, also known as token prun-
ing or selection, aim to reduce input length by dis-
carding tokens deemed less informative. Early ap-
proaches like Selective Context (Li et al., 2023b)utilize self-information or perplexity metrics to fil-
ter out redundancy. More advanced frameworks,
such as LongLLMLingua (Jiang et al., 2024) and
LLMLingua-2 (Pan et al., 2024), perform prompt
compression via task-aware coarse-to-fine filtering
and distillation-based token selection, respectively.
Despite these advancements, hard compression ap-
proaches inherently limit the model’s reasoning
potential. Theyirreversibly discard information
and rely on rigid, locally made retention decisions
that often fail to preserve theglobal semantic
structurerequired for reliable complex reasoning.
Soft Compression (Latent Representation).
Early soft compression approaches, such as Au-
toCompressor (Chevalier et al., 2023), Gist Tokens
(Mu et al., 2024), and ICAE (Ge et al., 2024), in-
tegrate compression and reasoning within a single
language model. While effective for moderate con-
text reduction, this tightly coupled design makes
it difficult for a single model to simultaneously
excel at both high-fidelity compression and com-
plex reasoning, particularly under extreme com-
pression ratios. To address this limitation, sub-
sequent work explores decoupling compression
from reasoning. Methods such as xRAG (Cheng
et al., 2024) and COCOM (Rau et al., 2024) build
upon the Retrieval-Augmented Generation (RAG)
paradigm by compressing retrieved documents into
compact latent representations before passing them
to the language model. Although effective in re-
ducing input length, these approaches remain fun-
damentally constrained by the retrieval stage and
inherit the upper-bound limitations of RAG sys-
tems. Beyond RAG-based compression, E2LLM
(Liao et al., 2025) employs a lightweight encoder
to compress long inputs into latent representations
for downstream LLM reasoning, though the model
weights are not publicly released, limiting repro-
ducibility and practical adoption. Context Cascade
Compression (C3) (Liu and Qiu, 2025) demon-
strates strong compression fidelity but lacks task-
specific adaptation for downstream reasoning. As
a result, a gap remains between compressed repre-
sentation understanding and effective reasoning.
However, most existing soft compression meth-
ods operate in astaticand query-agnostic manner,
producing generic compressed representations that
often fail to preserve task-critical information un-
der high compression ratios. In contrast,DRIFT
adopts adynamic, query-conditioned compres-
sion strategythat selectively encodes relevant in-
2

formation, enabling effective reasoning even un-
derextreme compression ratiosin knowledge-
intensive scenarios.
2.2 Learned Memory
nother line of work introduces learned paramet-
ric memory modules, such as Memory Decoder
(Cao et al., 2025) and MLP Memory (Wei et al.,
2025), which store knowledge in trainable param-
eters and are often pretrained to emulate retrieval
behavior. These methods reduce inference latency
without modifying the underlying model parame-
ters, thereby preserving general reasoning capabil-
ities. However, these architectures are inherently
static-resource bound: they rely on pre-indexing
or offline training on fixed knowledge bases, ren-
dering them incapable of handling instantaneous,
long-context inputs in real-time. Moreover, as these
modules are often pre-trained to emulate or com-
press retriever behaviors, their inference dynamics
remain tethered to the limitations of the original
retrieval paradigm.
3 Methodology
Our proposed strategy encourages the knowledge
model to perform information-adaptive compres-
sion, rather than static compression. This compels
the model to first identify and abstract core informa-
tional content before encoding it into a compressed
semantic representation. As a result, the model
learns to prioritize semantically meaningful content
over superficial token patterns, leading to improved
generalization and robustness in downstream tasks.
3.1 Bucketed Compression: Beyond
Fixed-Ratio Compression
Most existing context compression methods adopt
a fixed-ratio strategy, where the number of com-
pressed tokens is strictly proportional to the input
length (e.g., compressing 128 input tokens into 16
output tokens for an 8:1 ratio). However, such
a design implicitly assumes that informative con-
tent is evenly distributed across the input, which
rarely holds in real-world tasks. The amount of
query-relevant information within a given context
is always unknown and varies in token length. Cer-
tain tokens (such as numbers, named entities, main
verbs, and constraint-related words) carry a large
amount of task-relevant information, whereas re-
dundant modifiers and generic, content-free sen-
tences contribute little to understanding the context.For example, in document-grounded QA or long-
form reasoning, a few critical sentences may carry
the majority of the answer-relevant information.
This fixed-ratio compression can thus become
fragile in scenarios where the input contains sparse
but crucial evidence. Moreover, training a model
to uniformly compress variable-length sequences
may encourage shortcut learning (e.g., positional
bias, over-averaging), hindering semantic abstrac-
tion. To address this, we propose a Bucketed Com-
pression strategy that shifts from ratio-based com-
pression to range-based compression. Instead of
computing the number of output tokens as a fixed
fraction of the input, we predefine token-length
buckets (e.g., 64–128, 128–256), and map each in-
put in a bucket to a fixed-size output based on the
upper bound of that range.
To make the distinction clear, we present a direct
comparison of the two strategies in formulaic form,
under the assumption of a compression ratioc:
ξuniform =n
c
, ξ bucket =l
b(n)
cm
where b(n) denotes the upper bound of the bucket
containingntokens.
3.2 DRIFT: Decoupled Reasoning with
Implicit Fact Tokens
The core idea of DRIFT is to modularize and ex-
plicitly separate knowledge reading and reason-
ing. Specifically, a small-scale knowledge model
(ψkno) is responsible for reading long documents
and compressing them into query-relevant infor-
mation, while a large-scale reasoning model ψkno
focuses on utilizing this compressed knowledge to
perform complex reasoning and generate answers.
The interaction between the two models is real-
ized in the latent space, which reduces redundancy
and mitigates the risk of irrelevant noise interfering
with the reasoning process.
To facilitate a clear understanding of our method,
the overall workflow of DRIFT is depicted in
Figure 1. We first define a document as X=
(x1, . . . , x n), where nis the number of tokens
in the document. For long input contexts, a
systematic document chunking strategy was em-
ployed within the DRIFT framework. We utilized
theRecursiveCharacterTextSplitter from the
LangChain framework (Contributors, 2024) for this
purpose. This recursive approach was adopted to
ensure optimal semantic coherence by prioritizing
3

Chunking Parallel Knowledge Compression
Query··· ···
···ψkno
······
···
··· ···
···ψkno
······
···
··· ···
···ψkno
······
···Parallel processing of C3 to CK-1 T1
T2
TKChunk C1
··· Long
Documents
Concatenation & Projectionlast-layer hidden stateslast-layer hidden stateslast-layer hidden states
···
···
···T1
T2
TK···T1T2 TK ··· Projector
Implicit Fact TokensReasoning & Answer
···Embedding 
LayerθreaOther
Layers
Response···
···Chunk C2
Chunk CK
Query Embeddings
Implicit Fact EmbeddingsCompression 
Token
Trainable
ψknoSmall
Knowledge 
Model
θreaLarge 
Reasoning
Model
ConcatenationFigure 1:The overall workflow of DRIFT.DRIFT implements knowledge compression and decoupled reasoning
in four steps.Step 1:The long document Xis recursively partitioned into semantically coherent chunks to preserve
structural integrity.Step 2:The small knowledge model ψknocompresses query-relevant information from each
chunk in parallel into latent implicit fact tokens TJ.Step 3:The latent tokens are concatenated and mapped by an
MLP projector πto align with the reasoning model’s embedding space.Step 4:The large reasoning model θrea
generates the final response by performing efficient inference on the concatenated embeddings.
natural delimiters (e.g., paragraphs and sentences).
XSplit− − →C= (C 1, C2, . . . , C K).(1)
Given a query Q, we append a fixed number of
<|CPS|> tokens to each chunk Cjand process them
in parallel with the knowledge model ψkno, using
the last-layer hidden states of these compression
tokens as the latent representationT j.
ψkno: (Cj, Q)→T j∈Rξj×d.(2)
Finally, the outputs from all chunks are concate-
nated in the original order to yield the global se-
quence of implicit fact tokens, denotedT.
T= Concat(T 1, . . . , T K) = [t 1, . . . , t ξ]∈Rξ×d,
(3)
whereξ=PK
j=1ξj≪N.
Implementation Note: During concatenation, a
double newline separator ( \n\n ) is inserted be-
tween adjacent sub-sequences TjandTj+1to mark
chunk boundaries. We define Ti∈Rdas implicit
fact tokens, and d denotes the hidden state dimen-
sionality. These tokens encapsulate essential in-
formation from the input and serve as continuous
latent units for the reasoning model.In the next step, a three-layer MLP projector
πmaps the implicit fact tokens into implicit fact
embeddings E= (e 1, e2, . . . , e ξ), thereby aligning
them with the embedding space of the reasoning
model. These embeddings, together with the query
embeddings E(Q) , are subsequently fed into the
reasoning model for downstream inference.
θrea: Concat(E, E(Q))→Response (4)
The Response denotes the thoughts and the final
answer generated by the reasoning model.
This approach not only substantially reduces
GPU memory consumption but also frees the rea-
soning model from processing lengthy documents
filled with irrelevant information.
3.3 Task Definition
To more effectively achieve the decoupling of
knowledge and reasoning, we decompose the train-
ing of DRIFT into three distinct stages, each opti-
mized for a different objective, as shown in Figure
2.
4

Stage 1：LFRP
Ista
Projector
θreaIrecStage 2：QAFT - DC
Idyn
Projector
θreaIrecStage 3：QAFT - QA
Idyn
Projector
θreaIansDocument Document Document Query Query
Evidence Answer
Document 
…Marie Curie was a renowned scientist known for her groundbreaking research on 
radioactivity. She discovered two new chemical elements, polonium and radium, through
years of work. Her achievements earned her two Nobel Prizes in different scientific fields…Query
Which elements did Marie Curie discover?
Evidence
She discovered two new chemical elements, polonium and radium.Answer
polonium and radium.
Trainable Frozen Small Knowledge Model Large Reasoning Model ψkno θreaψkno ψkno ψkno
𝓛 𝓛 𝓛Figure 2: Three different trainging tasks for DRIFT. The instructions in the figure include the dynamic compression
instruction, reconstruct instruction, answer instruction, and static compression instructio.
3.3.1 Latent Fact Reconstruction Pretraining
(LFRP)
We redefine the pretraining objective such that the
reasoning model is used only as a frozen decoder
to provide a reconstruction signal, while the knowl-
edge model is optimized to generate latent factual
representations that best support document recon-
struction.
For the knowledge model, this compression is
static because it is query-independent. We set the
static compression ratio cstato 8. After applying
the bucketed compression strategy, we obtain ξsta,
denoting the number of implicit fact tokens.
Given a document xconsisting of ntokens,
X= (x 1, x2, . . . , x n), the knowledge model pro-
duces latent fact tokens conditioned on a static com-
pression instructionI sta:
Tsta= [t 1, t2, . . . , t ξsta] =ψ kno(Ista, x1, x2, . . . , x N).
(5)
These implicit fact tokens are projected into fact
embeddings via a projector moduleπ:
Esta=π(T sta) = (e 1, e2, . . . , e ξsta).(6)
The reasoning model, parameterized by θr, re-
mains frozen during this pretraining stage. It re-
ceives a reconstruction instruction Irecand predicts
each token conditioned on the fact embeddings and
previously generated tokens:
L(ψ kno, π) =−X
xt∈XlogP θrea(xt|Irec, E, x <t),(7)Thus, although the loss is computed using the
frozen reasoning model, gradients are only back-
propagated through Estainto the projector πand
the knowledge model ψkno. This teaches the knowl-
edge model to produce latent factual representa-
tions that are maximally useful for reconstructing
the original document.
3.3.2 Query-Aware Fine-Tuning (QAFT) with
Single-Context
Through the pretraining objective described above,
the knowledge model acquires the ability to encode
factual knowledge into a latent space, while the rea-
soning model learns to understand the implicit fact
embeddings. To further adapt the framework for
downstream tasks, we introduce an additional train-
ing objective that incorporates query inputs. This
objective is designed to encourage the knowledge
model to implicitly extract and compress query-
relevant knowledge into the latent space, and to
enable the reasoning model to perform question an-
swering by leveraging the information encoded in
the implicit fact embeddings. We train the models
on QA datasets, including reading comprehension,
fact verification and open-domain question answer-
ing.
To better train toward this objective, we per-
form two fine-tuning tasks in sequence: first a
dynamic compression task, followed by a question-
answering task.
5

Dynamic Compression TaskIn the pretraining
task, the knowledge model learns to perform static
compression of the context. Now, we aim to train
the model to acquire query-aware dynamic com-
pression capability.
In training dataset, each question-answer pair is
annotated with supporting evidence. We design a
dynamic compression task using the context and
question-evidence pairs: a knowledge model ex-
tracts query-specific information from the context
into a latent space (implicit fact tokens), while a rea-
soning model reconstructs the evidence from these
tokens. By leveraging the sparsity of query-relevant
information within the context, dynamic compres-
sion can safely achieve a significantly higher com-
pression ratio than static approaches. The default
dynamic compression ratio is set to 32.
We use the instruction Idynto ask the knowledge
model to extract question-relevant knowledge from
the document and encodes it into a set of implicit
fact tokens in the latent space. Conditioned on the
input query Q, the knowledge model ψknogenerates
a sequence of question-aware implicit fact tokens
Tdyn.
The reconstruction performed by the reasoning
model under instruction Irecremains analogous to
the pre-training stage. However, we diverge by re-
stricting the reconstruction target to the evidence in-
stead of the complete original text. This mechanism
employs the evidence as the supervisory signal, ex-
plicitly training the knowledge model to develop
dynamic compression capabilities.
Given a document X= (x 1, x2, . . . , x n)and
a question Q= (q 1, q2, . . . , q m), the knowledge
model ψkperforms dynamic compression on X
conditioned onQ.:
Tdyn= (t 1, t2, . . . , t ξdyn) =ψ kno(Idyn, X, Q)(8)
These latent tokens are then projected into the
final fact embeddings Edyn= (e 1, e2, . . . , e ξdyn)
via the projector module π. The reasnoning model
still keeps fixed and is asked to reconstruct the
evidenceX evi:
L(ψ kno, π) =−X
xk∈XevilogP θrea(xk|Irec, Edyn, x<k)
(9)
Question-answering TaskThis task is the only
one in which the reasoning model is not frozen dur-
ing training, enabling it to better exploit the com-
pressed context for downstream tasks. We performfine-tuning by updating the models solely based on
the target answers.
The generation of Edynleverages the same mech-
anism as the Dynamic Compression Task, imple-
mented by the Knowledge model and projector.
Then we instruct the reasoning model with Iansto
generate the answer of the question based on the
fact embeddings. We denote the answer sequence
asA= (a 1, a2, . . . , a l).
We define the training objective as the standard
language modeling loss. Our formulation is largely
analogous to instruction tuning (Wei et al., 2022),
with the key distinction that the context presented
to the reasoning model is transformed from the ex-
plicit text tokens to the implicit fact embeddings
produced by the knowledge model. The question
embedding is represented by E(Q) . We minimize
the following loss to optimize the model parame-
ters:
L(θrea, ψkno) =−X
aj∈AlogP θrea 
aj|Ians, Edyn, E(Q), a <j
(10)
3.4 Multi-Context Inference without
Fine-Tuning
We find that DRIFT, fine-tuned solely in the
single-context setting via QAFT, generalizes ef-
fectively to multi-context inference without requir-
ing any additional multi-context training. Specif-
ically, to process extensive contexts, we em-
ploy an overlapping chunking strategy utilizing
RecursiveCharacterTextSplitter . We set the
chunk size to 8,192 tokens, aligning with the maxi-
mum document length used during training. These
chunks are then dynamically compressed in parallel
by the knowledge model.
3.5 Training Data
We use the English Wikipedia snapshot dated
November 1, 2023, treating each entry as a doc-
ument and its text field as the raw training content.
Since longer input documents make model train-
ing and convergence more challenging, we adopt
a token-level curriculum learning strategy across
all three training tasks, with phases defined by in-
put document length. Details of the construction
procedures for LFRP and QAFT, as well as the
curriculum learning partitions, are provided in Ap-
pendix A.
6

4 Experiments
4.1 Implement Details
For the DRIFT setup, we fix
Qwen2.5-Instruct-3B as the knowledge
model and evaluate reasoning models from
different families, including Mistral(Jiang et al.,
2023) and Qwen2.5(Qwen et al., 2025), with
Mistral-7B-Instruct-v0.2 as default. We
further explore additional configurations, such
as smaller knowledge models (1.5B) and larger
reasoning models (14B). All models are trained
using parameter-efficient fine-tuning with LoRA
(Hu et al., 2021). The results for broader model
combinations, along with detailed training
hyperparameters, are provided in the appendix.
RQ1: To what extent do the latent representa-
tions capture and preserve the essential infor-
mation of the original context?
To assess whether the knowledge model can
compress long contexts with minimal information
loss, we evaluate the reconstruction fidelity of the
learned latent representations before downstream
reasoning. Specifically, after training on the LFRP
task, we conduct a compression–reconstruction ex-
periment on a test subset with input lengths rang-
ing from 512 to 1024 tokens, where the reasoning
model reconstructs the original text conditioned
solely on the compressed representations produced
by the knowledge model. Reconstruction quality
is measured using BLEU and ROUGE (ROUGE-
1/2/L) scores.
Models BLEU R-1 R-2 R-L
DRIFT after LFRPMistral-7B 90.01 94.95 93.93 94.75
Qwen2.5-7B 83.43 86.60 84.13 88.80
DRIFT before LFRPMistral-7B 0.01 1.63 0.07 1.46
Qwen2.5-7B 0.00 6.32 0.65 4.52
Note:R-1/2/L stands for ROUGE-1/2/L scores. Scores are scaled by 100.
Table 1: Performance evaluation of the reconstruction
task across different model configurations.
As illustrated in Table 1, the compres-
sion–reconstruction task achieves strong perfor-
mance across all model combinations following the
LFRP training phase. This indicates that after train-
ing, the knowledge model effectively compresses
the input content and the reasoning model accu-
rately interprets the compressed representations.
RQ 2: How does DRIFT compare to representa-
tive baselines in terms of overall effectiveness in
long-context reasoning scenarios?BenchmarksTo evaluate DRIFT across diverse
long-context scenarios, we employ several repre-
sentative benchmarks.BAMBOO(Dong et al.,
2024) serves as a comprehensive suite for testing
extended context capabilities through tasks like
question answering and code completion.L-Eval
(An et al., 2023) provides a balanced evaluation
of single-hop and multi-hop reasoning across vary-
ing document lengths up to 256K tokens while
minimizing knowledge leakage.LongBench-v2
(Bai et al., 2025) consists of challenging multiple-
choice questions requiring multi-document under-
standing and structured data reasoning across con-
texts reaching millions of words. Finally,LoCoMo
(Maharana et al., 2024) focuses on long-term con-
versational memory, probing the model’s ability to
perform temporal reasoning and event summariza-
tion over multi-session dialogue histories.
MetricsThere are two primary metrics. For
datasets involving multiple-choice, closed-ended,
and measurable open-ended tasks, we utilize
Qwen-2.5-72B-Instruct as anLLM-Judgeto
compute accuracy. For summarization-oriented
tasks, we reportROUGE-Lscores to measure the
similarity between generated responses and ground-
truth references based on the Longest Common
Subsequence.
BaselinesWe compareDRIFTagainst a diverse
set of baseline methods categorized by their com-
pression and retrieval paradigms. Forhard com-
pression, we selectLLMLingua-2to represent
lexical-level token pruning. Forsoft compression,
we evaluate several latent-space models includ-
ingICAE,COCOM, andxRAG. Additionally,
we implement aNaiveRAGbaseline utilizing the
BGE-M3embedding model for document retrieval.
TheMistral-7B-v0.2model serves as our primary
vanilla backbone to provide a performance lower
bound without external enhancements.
Results and AnalysisTable 2 demonstrates
that DRIFT consistently outperforms existing
compression-based baselines across diverse long-
context benchmarks, particularly under high com-
pression ratios. With the same Mistral backbone,
DRIFT sets a new state of the art over existing
compression-based methods. Notably, on task-
oriented summarization benchmarks (QMSUM
andSPACE) and theLoCoMoconversational
memory benchmark—where prior compression ap-
proaches largely fail—DRIFT remains effective,
7

ModelComp.
RatioBAMBOO (16k) L-Eval (QA Subset) L-Eval (Sum Subset) LoCoMo LongBench-v2
AltQA Meet Paper Avg NQ NarQA CrseAvgQMS SPCAvgT1 T2 T3 T4AvgShort Medium LongAvg
Baseline Methods based on Mistral-7B-v0.2
LLMLingua-2 3×40.00 75.00 76.77 57.89 77.98 33.18 64.53 53.94 19.37 18.86 19.14 46.81 27.41 48.96 76.93 59.35 26.11 13.95 25.00 20.68
NaiveRAG – 34.50 79.00 75.76 55.89 72.48 21.50 63.37 47.27 19.61 18.11 18.94 45.04 24.61 41.67 73.48 56.10 17.78 26.05 25.00 22.86
xRAG 128×25.00 68.00 57.58 43.86 56.88 12.62 50.58 35.56 18.06 12.34 15.49 20.57 6.23 34.38 32.10 24.74 31.11 25.58 25.00 27.43
ICAE 4×21.50 19.00 19.19 20.30 36.70 5.14 15.70 15.76 17.62 13.27 15.67 17.38 3.43 23.96 15.10 13.64 13.89 14.88 10.56 13.12
COCOM 4× 30.50 60.00 47.47 42.11 61.47 9.81 53.49 36.36 9.15 7.79 8.54 9.93 1.87 27.08 10.34 9.55 27.22 23.26 34.26 27.04
16× 25.50 49.00 41.41 35.34 53.21 8.88 55.81 34.95 7.78 5.37 6.70 11.35 1.87 25.00 10.11 9.55 27.78 23.26 34.26 27.24
128× 15.00 40.00 36.36 26.57 40.37 5.61 39.53 25.05 8.71 4.79 6.95 12.06 1.56 30.21 8.32 8.96 29.44 21.40 34.26 27.04
DRIFT based on Mistral-7B-v0.2
DRIFT (Ours) 32×41.50 80.00 77.78 60.15 69.72 27.10 61.05 48.2821.59 19.75 20.7636.88 23.99 33.33 80.38 57.73 32.22 28.84 25.00 29.22
64×36.00 82.00 82.83 59.15 70.64 24.77 58.72 46.6721.95 19.49 20.8534.75 16.82 42.71 74.67 53.31 26.67 29.77 19.44 26.44
128×36.00 77.00 78.79 56.89 71.56 17.29 60.47 44.2420.69 19.14 19.9930.14 17.13 36.46 72.06 50.71 32.22 24.65 19.44 26.24
DRIFT based on Qwen2.5-Instruct-7B
DRIFT (Ours) 32×37.00 81.00 84.85 59.90 80.73 27.10 76.16 55.9622.66 19.01 21.0242.55 27.73 40.62 85.49 62.79 36.67 28.37 33.33 32.41
Vanilla LLM
Mistral-7B-v0.2 1×40.00 75.00 76.77 57.89 79.82 31.78 65.12 53.94 19.32 18.90 19.13 46.81 27.73 48.96 76.81 59.35 25.00 16.28 23.15 20.87
Qwen2.5-Instruct-7B 1×36.00 78.00 82.83 58.15 80.73 33.18 75.58 58.38 20.74 18.50 19.74 48.23 38.01 46.88 85.14 66.17 41.11 26.05 24.07 31.01
Table 2: Main results of DRIFT and other baseline methods on long context datasets.
highlighting its superior ability to preserve and ex-
ploit long-range contextual information.
RQ3: How does each training objective con-
tribute to the overall performance of DRIFT?
DRIFT incorporates three distinct training objec-
tives: LFRP, QAFT-DC, and QAFT-QA. To justify
the necessity of this multi-stage design and quantify
the contribution of each component, we conduct
a comprehensive ablation study across three long-
context benchmarks.
Method Bamboo LongBenchv2 LoCoMo
DRIFT (32×)60.15 29.22 57.73
w/o LFRP 57.64↓2.5126.84↓2.3852.68↓5.05
w/o QAFT-DC 56.64↓3.5125.84↓3.3857.36↓0.37
w/o QAFT-QA 45.14↓15.0118.05↓11.1736.89↓20.84
Note:QAFT-DC denotes theQAFT Dynamic Compression
objective, and QAFT-QA denotes theQAFT Question Answer-
ingobjective.
Table 3: DRIFT ablation results (avg. accuracy).
As shown in Table 3, each training objective in
DRIFT serves a unique purpose: QAFT-QA pro-
vides the fundamental reasoning backbone, while
LFRP and QAFT-DC further optimize the latent
space efficiency and robustness.
RQ4: Does DRIFT maintain competitive infer-
ence efficiency compared to classical baselines?
Efficiency is a critical bottleneck for long-
context reasoning. We conduct an end-to-end Time-
to-First-Token (TTFT) analysis to evaluate whetherDRIFT maintains its performance advantages with-
out incurring prohibitive computational costs as
the input scale grows. The end-to-end Token-to-
First-Token (TTFT) measures the latency from pro-
viding both the input documents and the query to
the system until the first output token is generated.
This metric directly captures the practical respon-
siveness of long-context reasoning systems under
realistic inference settings.
Figure 3 shows that DRIFT maintains competi-
tive efficiency across all baselines. Notably, the
performance gap between DRIFT and theFull-
Contextbaseline widens significantly as the input
length increases, demonstrating DRIFT’s superior
scalability for ultra-long sequences.
Figure 3: End-to-end TTFT as a function of input length
for different baselines.
8

5 Conclusion
In this paper, we introducedDRIFT, a dual-model
architecture designed to decouple factual knowl-
edge acquisition from general reasoning in LLMs.
Input documents are segmented into chunks, com-
pressed into high-density fact embeddings by a
lightweight knowledge model, and interpreted by a
larger reasoning model. Experimental results and
ablation studies demonstrate the effectiveness of
DRIFT and the necessity of each training task. The
method also generalizes well across different com-
pression ratios and backbone models, highlighting
its robustness and practical applicability.
6 Limitations
Despite its effectiveness, our work has certain limi-
tations that suggest directions for future research.
First, due to computational resource constraints,
our experiments were primarily validated on mod-
els with up to 14B parameters; the performance
and scaling laws of DRIFT on larger-scale mod-
els remain to be further explored. Second, the
use of latent compression introduces challenges re-
garding interpretability, as the implicit fact tokens
are not as human-readable as raw text snippets in
traditional RAG. Finally, our current framework
is primarily optimized through Supervised Fine-
Tuning (SFT). We anticipate that integrating Rein-
forcement Learning (RL) could further enhance the
model’s decision-making in knowledge selection
and lead to new breakthroughs in performance.
References
Chenxin An, Shansan Gong, Ming Zhong, Xingjian
Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and
Xipeng Qiu. 2023. L-eval: Instituting standard-
ized evaluation for long context language models.
Preprint, arXiv:2307.11088.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng,
Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu,
Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li.
2025. Longbench v2: Towards deeper understanding
and reasoning on realistic long-context multitasks.
Preprint, arXiv:2412.15204.
Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai
Chen, Bowen Zhou, and Zhouhan Lin. 2025. Mem-
ory decoder: A pretrained, plug-and-play memory for
large language models.Preprint, arXiv:2508.09874.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Ponde de Oliveira Pinto, Jared Kaplan,
Harri Edwards, Yuri Burda, Nicholas Joseph, GregBrockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela
Mishkin, Brooke Chan, Scott Gray, and 39 others.
2021. Evaluating large language models trained on
code.Preprint, arXiv:2107.03374.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-
Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan
Zhao. 2024. xrag: Extreme context compression
for retrieval-augmented generation with one token.
Preprint, arXiv:2405.13792.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and
Danqi Chen. 2023. Adapting language models to
compress contexts.Preprint, arXiv:2305.14788.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems.Preprint, arXiv:2110.14168.
LangChain Contributors. 2024. Langchain: A frame-
work for developing applications powered by large
language models. GitHub repository. Used the
text-splitting utilities (originally from langchain-text-
splitters, Version 0.3.8).
Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao,
and Ji-Rong Wen. 2024. Bamboo: A compre-
hensive benchmark for evaluating long text model-
ing capacities of large language models.Preprint,
arXiv:2309.13345.
Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen,
and Furu Wei. 2024. In-context autoencoder for con-
text compression in a large language model.Preprint,
arXiv:2307.06945.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2021. Lora: Low-rank adaptation of
large language models.Preprint, arXiv:2106.09685.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b.Preprint,
arXiv:2310.06825.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng
Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024.
Longllmlingua: Accelerating and enhancing llms
in long context scenarios via prompt compression.
Preprint, arXiv:2310.06839.
Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rod-
kin, Dmitry Sorokin, Artyom Sorokin, and Mikhail
Burtsev. 2024. Babilong: Testing the limits of llms
with long context reasoning-in-a-haystack.Preprint,
arXiv:2406.10149.
9

Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun
Nie, and Ji-Rong Wen. 2023a. Halueval: A large-
scale hallucination evaluation benchmark for large
language models.Preprint, arXiv:2305.11747.
Yucheng Li, Bo Dong, Chenghua Lin, and Frank
Guerin. 2023b. Compressing context to enhance in-
ference efficiency of large language models.Preprint,
arXiv:2310.06201.
Zihan Liao, Jun Wang, Hang Yu, Lingxiao Wei, Jian-
guo Li, Jun Wang, and Wei Zhang. 2025. E2llm:
Encoder elongated large language models for long-
context understanding and reasoning.Preprint,
arXiv:2409.06679.
Fanfan Liu and Haibo Qiu. 2025. Context cascade com-
pression: Exploring the upper limits of text compres-
sion.Preprint, arXiv:2511.15244.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of llm agents.Preprint, arXiv:2402.17753.
Jesse Mu, Xiang Lisa Li, and Noah Goodman. 2024.
Learning to compress prompts with gist tokens.
Preprint, arXiv:2304.08467.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia,
Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle,
Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu,
and Dongmei Zhang. 2024. LLMLingua-2: Data dis-
tillation for efficient and faithful task-agnostic prompt
compression. InFindings of the Association for Com-
putational Linguistics: ACL 2024, pages 963–981,
Bangkok, Thailand. Association for Computational
Linguistics.
Valentina Pyatkin, Saumya Malik, Victoria Graf,
Hamish Ivison, Shengyi Huang, Pradeep Dasigi,
Nathan Lambert, and Hannaneh Hajishirzi. 2025.
Generalizing verifiable instruction following.
Preprint, arXiv:2507.02833.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
David Rau, Shuai Wang, Hervé Déjean, and Stéphane
Clinchant. 2024. Context embeddings for ef-
ficient answer generation in rag.Preprint,
arXiv:2407.09252.
David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien Di-
rani, Julian Michael, and Samuel R. Bowman. 2023.
Gpqa: A graduate-level google-proof q&a bench-
mark.Preprint, arXiv:2311.12022.Mirac Suzgun, Nathan Scales, Nathanael Schärli, Se-
bastian Gehrmann, Yi Tay, Hyung Won Chung,
Aakanksha Chowdhery, Quoc V . Le, Ed H. Chi,
Denny Zhou, and Jason Wei. 2022. Challenging
big-bench tasks and whether chain-of-thought can
solve them.Preprint, arXiv:2210.09261.
Jason Wei, Maarten Bosma, Vincent Y . Zhao, Kelvin
Guu, Adams Wei Yu, Brian Lester, Nan Du, An-
drew M. Dai, and Quoc V . Le. 2022. Finetuned
language models are zero-shot learners.Preprint,
arXiv:2109.01652.
Rubin Wei, Jiaqi Cao, Jiarui Wang, Jushi Kai, Qipeng
Guo, Bowen Zhou, and Zhouhan Lin. 2025. Mlp
memory: A retriever-pretrained memory for large
language models.Preprint, arXiv:2508.01832.
10

A Data Generation Details
We compute token lengths for Wikipedia docu-
ments and sample text segments for each length
bucket. Specifically, for the LFRP task, we sample
200,000 segments per bucket, ensuring full cover-
age including the 4k8k range. For the QA-FT task,
we sample 100,000 segments per bucket, with the
exception of the 4k–8k range where data remains
insufficient. Each stage is then split into training,
validation, and test sets with an 8:1:1 ratio.
A.1 LFRP Data
For the unlabeled data used in pre-training, we de-
liberately refrain from any data cleaning. Since
documents or knowledge bases in real-world sce-
narios are often noisy and heterogeneous in format,
we retain the raw text to improve the robustness
of the pre-trained model. For each stage, we use
160,000 samples to train the model.
A.2 QAFT Data
Based on raw Wikipedia documents corresponding
to each length bucket, we employed Qwen2.5-72B-
Instruct to generate QA pairs, which were subse-
quently used for fine-tuning in each training stage.
In general, each document corresponds to one QA
pair; however, for the 4k–8k stage, the available
raw documents were insufficient, so some docu-
ments were reused. To avoid positional bias in the
generated QA data, we segment each document and
randomly sample one slice as the input for QA gen-
eration. To avoid positional bias in the generated
QA data, we segment each document and randomly
sample one slice as the input for QA generation.
The workflow of data generation is shown in Figure
4.
Data Sampling StrategyTo ensure that relevant
information is uniformly distributed across the gen-
erated dataset, we adopt the following sampling
strategy. Each Wikipedia document is first divided
into multiple slices of equal length. For every doc-
ument, we then randomly select one slice as the
context for QA generation. This prevents the model
from always encountering answers concentrated in
specific regions (e.g., the beginning of documents)
and ensures that relevant content appears at ran-
dom positions. As a result, the generated QA pairs
cover diverse locations within documents, leading
to a more balanced and robust training signal. In
addition to generating the question–answer pairs,
the model is also required to provide the supportingevidence from the original text corresponding to
each answer.
Question Type DiversificationTo better train
the reasoning model to utilize information com-
pressed by the knowledge model, we randomly
sample one of three formats—multiple-choice,
true/false, or short-answer questions—for QA gen-
eration. This exposes the model to diverse reason-
ing scenarios, thereby strengthening its ability to
integrate knowledge-derived information for effec-
tive problem solving, and enhancing its generaliza-
tion across different task settings.
Data FilteringTo ensure the quality of the auto-
matically generated QA pairs, we employ Qwen2.5-
72B-Instruct as the judger model to filter out low-
quality instances. Specifically, the filtering pro-
cess is guided by five criteria: (i)Relevance: the
question must be grounded in the given context;
(ii)Correctness: the provided answer should be
factually consistent with the context; (iii)Clarity:
the question and answer must be well-formed and
unambiguous; (iv)Fidelity: the evidence must ac-
curately reflect content from the original document
without introducing external information; (v)Suf-
ficiency: the evidence must provide enough infor-
mation to answer the question. Only QA pairs that
satisfy all conditions are retained for subsequent
training.
A.3 Training Strategy: Token-Level
Curriculum Learning
During model training, we observed that longer
input documents lead to increased training diffi-
culty. In particular, the convergence of both the
knowledge and reasoning models becomes more
challenging as the number of tokens grows. There-
fore, we apply atoken-level curriculum learning
strategyacross all three training tasks, where train-
ing stages are defined according to the token length
of the input documents. Table 4 summarizes the
token-level stage configuration for each of the three
training tasks.
Task Stage 1 Stage 2 Stage 3 Stage 4
LFRP64-128 128-256 256-512 512-1k
QAFT Tasks
Dynamic Comp. 1k-2k 2k-4k 4k-8k –
Question Ans. 1k-2k 2k-4k 4k-8k –
Table 4: Curriculum learning stages. Note that QAFT
tasks conclude at Stage 3.
11

Module 1: Data
InitializationModule 2: Parallel
SchedulingModule 3: LLM Gener ation & Quality
Verifica tion
🔄 Retry Loop (Max 10 times)Module 4: Result
ProcessingModule 5: Output Saving
Start
Parse Command Line
Arguments
Load Parquet File
sample_size
< 1.0?
Random Sample
DataExtract
contexts List &
Initialize Result
Array🔀
ThreadPool
Executor
(Parallel Processing)
💡 For each context
dispatch to thread pool🎯 Start Processing
Single Context
🎲 Randomly Select
Position & Q-T ype
✂ split_context()
Segment Text
🤖 Call LLM API
Generate Q&A&E Pair
📝 parse_json()
Parse JSON Response
question/
answer/evidence
field empty?
✅
judge_answer_quality()
Quality AssessmentQuality
Check
Passed?❌ Retry
❌ Retry📦 Collect All
Parallel Results
🔍 Filter Empty Results
Any
Valid Data?
⚠ Print No-Output
Warning Message💾 Save JSONL
Main Data File
📋 Generate Example
JSON File
📊 Print Success Info
and Statistics
✓ End
YesNoDispatch T asks
No✓ Success
NoYesFigure 4: An Automated Pipeline for Contextual Question-Answering Data Synthesis
A.4 Construction Prompt
Two distinct prompts serve data construction: one
for generation and one for fine-grained evaluation.
Prompt 1“ Please generate a question that can be an-
swered based on the provided context. The question
should be highly relevant to the context, and the answer
must be directly inferable from the given information.
Avoid asking questions that cannot be answered using
the context. The question should be of the type: {ques-
tion_type}.
Your response should consist of three parts:
1. Question – the generated question. (a string)
2. Answer – the answer, including how it is reasoned out
from the relevant information in the context. (a string)
3. Evidence – the specific part(s) of the original text
that support the answer. (a string)
Attention: Evidence must be quoted directly from the
original text and must include all the information
needed to answer the question. If some parts of the
evidence involve unclear references (e.g., ambiguous
subjects), include the related sentences that clarify them,
so that the evidence alone is sufficient for answering the
question. Ensure that every sentence remains complete,
without the use of ellipses.
Your output format should be:
json
{{
"question": "<the generated question (include options
if the question type is multiple choice)>",
"answer": "<the corresponding answer, including how
it is inferred from the relevant information in the con-
text>",
"evidence": "<the specific part(s) taken directly from
the original text that support the answer>"
}}
Context: {context}
Your output: ”,Prompt 2“You are a judge evaluating the quality
of question-answer pairs. Your task is to determine
whether the given answer can be reasonably inferred
from the provided evidence.
Please evaluate based on the following criteria:
1. Can the answer be directly supported by the evi-
dence?
2. Is the evidence sufficient to answer the question?
3. Is the answer logically consistent with the evidence?
4. Are there any contradictions between the answer and
evidence?
Question: {question}
Evidence: {evidence}
Answer: {answer}
Please respond with only "true" if the answer can be
reasonably inferred from the evidence, or "false" if it
cannot.
Your judgment:”,
B Dataset Statistics
Dataset Statistics and Configuration: Table 5 sum-
marizes the statistics of our cleaned raw dataset.
The dataset comprises approximately 2.13 million
samples with a total of 1.61 billion tokens, cover-
ing a wide range of sequence lengths from 64 to
4,096. This diverse distribution ensures the model’s
robustness across various context windows. Our
final datasets were constructed by extracting and
processing samples from this original dataset. For
the training pipeline, we strategically allocated the
data across different phases: the LFRP stage uti-
lized 640,000 samples to establish solid feature
representations, while the QAFT stage employed
240,000 specifically constructed samples to refine
the model’s task-specific performance.
12

Table 5: Statistics of the dataset across different token
length ranges.
Token Range Split Total Tokens Samples
64 – 128Train 30,511,507 320,000
Val 3,813,711 40,000
Test 3,813,120 40,000
Total 38,138,338 400,000
128 – 256Train 61,175,314 320,000
Val 7,645,806 40,000
Test 7,644,977 40,000
Total 76,466,097 400,000
256 – 512Train 122,436,827 320,000
Val 15,301,762 40,000
Test 15,303,893 40,000
Total 153,042,482 400,000
512 – 1,024Train 244,714,180 320,000
Val 30,584,627 40,000
Test 30,586,768 40,000
Total 305,885,575 400,000
1,024 – 2,048Train 464,065,287 303,724
Val 58,032,940 37,964
Test 58,040,040 37,968
Total 580,138,267 379,656
2,048 – 4,096Train 361,195,148 118,272
Val 45,147,828 14,784
Test 45,127,513 14,784
Total 451,470,489 147,840
4,096 – 8,192Train 246,548,120 40,392
Val 30,841,105 5,048
Test 30,832,237 5,052
Total 308,221,462 50,492
C Training Details
C.1 Hyperparameter Settings
The specific hyperparameter configurations for our
model architectures and training environment are
summarized here. We conduct the training using
Low-Rank Adaptation (LoRA) to ensure parameter
efficiency. Specifically, we set the LoRA rank to
r=16 and the scaling factor to α=32, incorporating
a dropout rate of 0.05 to mitigate overfitting. The
optimization is performed with a learning rate of
0.0001 and a total effective batch size of 128.
C.2 Training Loss Curves Across Three
Stages
To evaluate the optimization stability and conver-
gence of DRIFT, we illustrate the training loss tra-
jectories across its three sequential stages below.C.2.1 Stage 1: Latent Fact Reconstruction
Pretraining (LFRP)
The training loss for the LFRP stage is presented
in Figure 5. The curve shows a steady decline
and eventual plateau, indicating that the knowledge
model successfully learned to reconstruct factual
content into the latent space with high fidelity.
Figure 5: Training loss trajectory of Stage 1
C.2.2 Stage 2: Query-Aware Fine-Tuning
(QAFT) with Single-Context Dynamic
Compression
During this stage, we fine-tune the model on the
Dynamic Compression Task. The training objec-
tive focuses on the model’s ability to compress and
project query-relevant knowledge into the reason-
ing model’s embedding space. The loss reflects
the efficiency of information bottlenecking under
query guidance.
Figure 6: Training loss trajectory of Stage 2
13

C.2.3 Stage 3: Query-Aware Fine-Tuning
(QAFT) with Single-Context Question
Answering
The final stage involves end-to-end optimization
for the Question-Answering (QA) task. We report
the cross-entropy loss during this phase, which
demonstrates how the reasoning model effectively
utilizes the distilled latent facts to generate accurate
and context-grounded answers.
Figure 7: Training loss trajectory of Stage 3
D Additional Experimental Results
RQ5: Does DRIFT outperform the reasoning
model with direct input across varied context
lengths?
To compare the performance of DRIFT against
the reasoning model with direct input across vari-
ous context lengths, we evaluated a sampled subset
from the BABILong (Kuratov et al., 2024) bench-
mark. The resulting comparison is illustrated in the
bar chart below:
Figure 8: Comparison of accuracy between Mistral-7B-
Instruct-v0.2 and DRIFT on a BABILong subset across
context lengths from 8k to 1M tokens.
Experimental results indicate that DRIFT con-
sistently outperforms the vanilla reasoning modelacross all tested scales, maintaining a substantial
performance margin even as the context length ex-
tends to 1M tokens. Notably, at the 64k context
length, DRIFT achieves nearly a 4×accuracy im-
provement (50.0% vs. 13.0%) compared to the
direct inference approach.
RQ6: Does fine-tuning degrade the reasoning
model’s general-purpose capabilities?
To assess whether the reasoning-oriented fine-
tuning (QAFT-QA) affects the model’s broad util-
ity, we compare the general-purpose capabilities of
the reasoning model before and after our proposed
training. The results indicate that the degradation
in generic competencies remains minimal. This
suggests that our fine-tuning strategy effectively
enhances reasoning performance while preserving
overall versatility.
Logical ReasoningTo test this capability, we uti-
lized the BBH (Big-Bench Hard) dataset (Suzgun
et al., 2022), which consists of 23 challenging tasks
from the BIG-bench suite that require sophisticated
multi-step reasoning where previous language mod-
els often underperformed.
Math ReasoningTo test this capability, we eval-
uate models on the standard GSM8K test set (1,319
problems) (Cobbe et al., 2021), which is drawn
from a benchmark of over eight thousand high-
quality grade school math word problems requiring
multi-step arithmetic reasoning and logical deduc-
tion.
Scientific KnowledgeTo test this capability, we
utilized the GPQA Diamond dataset (Rein et al.,
2023), which is a subset of the Graduate-Level
Google-Proof Q&A benchmark. It contains 198
high-quality questions in biology, physics, and
chemistry that have been meticulously vetted by
experts to ensure they are exceptionally challeng-
ing even for highly skilled non-experts with access
to the internet.
Instruction FollowingTo test this capability,
we evaluate on the IFBench benchmark (Pyatkin
et al., 2025), which measures precise instruction-
following generalization using a set of verifiable
constraints that models must satisfy in their gener-
ated outputs.
Code GenerationTo test this capability, we uti-
lized the HumanEval dataset (Chen et al., 2021),
comprising 164 manually crafted Python program-
ming problems used to evaluate the functional cor-
14

Figure 9: Placeholder radar chart illustrating general-
purpose capabilities before and after reasoning-oriented
fine-tuning.
rectness of code synthesized from natural language
function docstrings.
FactualityTo test this capability, we utilized the
HaluEval dataset (Li et al., 2023a), a large-scale
benchmark designed to assess hallucination levels
in large language models by testing their ability to
recognize and avoid generating factually incorrect
information.
RQ7: Does the DRIFT method remain effective
across different combinations of model sizes?
What would be the impact of substituting a
smaller knowledge model or a larger reasoning
model?Experimental analysis of the performance
impact when using different size combinations for
the knowledge and reasoning models.
Table 6: Performance of different model combinations
on LongBenchv2 across different lengths.
Combination LongBenchv2
Short Medium Long Overall
Qwen-7B×3B 36.67 28.37 33.33 32.41
Qwen-7B×1.5B 27.78 29.30 37.22 31.81
Qwen-14B×3B 37.03 31.63 36.11 34.39
RQ8: How does the model align its latent rea-
soning with explicit evidence during training?
A critical question in our framework is whether
the reasoning model merely replicates explicit tex-
tual evidence or instead develops distinct and effi-
cient reasoning strategies within the latent space.
This distinction is crucial for determining whetherthe implicit context functions as a complementary
modality rather than a redundant compression.
To examine this behavior, we introduce a di-
agnostic metric,Reasoning Consistency( MED),
which is monitored during the question-answering
task in theQAFT stage. Specifically, MEDmea-
sures the Kullback–Leibler (KL) divergence be-
tween the output distributions of the reasoning
model when conditioned on compressed latent rep-
resentations versus explicit textual evidence:
MED=D KL
Pθrea(· |I ans, Edyn,E(Q))
Pθrea(· |I ans, X evi, Q) (11)
Importantly, MEDis used solely as a non-intrusive
diagnostic probe to analyze reasoning behavior and
does not participate in gradient backpropagation.
0 100 200 300 400 500 600
Training Steps0.81.01.21.41.61.82.02.2MED (nats)Raw Metric
Smoothed Trend
Figure 10: Evolution of the reasoning consistency met-
ricM EDduring traiwoqning.
The trajectory of MEDin Figure 10 exhibits a
clearGrounding-then-Specializinglearning pattern
with two phases. InStage I (Rapid Grounding),
MEDdecreases sharply, indicating that the reason-
ing model initially aligns its latent representations
closely with explicit textual evidence to ensure
faithful reasoning. InStage II (Emergent Special-
ization), MEDgradually increases and stabilizes,
suggesting that the model moves beyond strict tex-
tual alignment and develops more specialized and
efficient reasoning strategies in the latent space.
Overall, this behavior shows that the implicit con-
text is not a mere compressed copy of the input,
but a complementary modality that supports task-
oriented reasoning beyond surface-level text.
15

E Other Details
E.1 Instructions for DRIFT
Ista“Given a text passage, condense its core concepts
into a set of words. The number of these compressed
words is {num}. The placeholder of compressed word is
‘{COMPRESSION_TOKEN}‘. The text you need to con-
dense is: <context>context</context> The compressed
words are: “
Irec “Background: <background> {com-
pressed_information} </background>. Please
restate the background information above in your own
words to convey the same meaning: “
Irec“Given several documents and a question, you
need to extract the information from the Documents
that is relevant to the Question, and condense the core
concepts of this knowledge into a set of words. Please
note that you are only responsible for extracting infor-
mation relevant to answering the question. You are not
required to reason out the answer yourself. You are
not allowed to fabricate information. You may only
extract and compress relevant information contained
in the documents. Please ensure the completeness
and understandability of the compressed knowledge.
The number of these compressed words is {num}." +
f"The placeholder of compressed word is ‘{COMPRES-
SION_TOKEN}‘ The documents are: <Documents>
{document}</Documents> The question is: <Ques-
tion>{question}</Question>The compressed words of
useful information are: . “
Ians “You will be provided with a background
consisting of {num} different paragraphs. Back-
ground: <background> {compressed_information}
</background>. Please answer the following
question based on the background. <Ques-
tion>{question}</Question>{answer_prefix} “
E.2 Text Reconstruction Case
A representative case of the source text and the cor-
responding reconstruction is illustrated in Figure
11, demonstrating the model’s ability to preserve
semantic coherence from highly compressed repre-
sentations.
16

Arthur Edgar Winston (12 March 1887– 4 November 1965) was a British historian, linguist, and cultural theorist best known for h is pioneering work on the "discursive margins" of
19th and early 20th -century Europe. His interdisciplinary approach to cultural history, emphasizing the recovery and analysis of non-canonical texts, profoundly influenced the 
development of microhistory and the linguistic turn in historical studies during the mid -20th century. Winston is most celebrate d for his seminal work The Silent Text: Forgotten 
Narratives of the Everyday (1938), which redefined how historians interpret marginal voices in historical discourse.
Born in Bristol, England, to a family of modest means —his father was a railway clerk and his mother a schoolteacher —Winston displayed an early aptitude for languages and 
classical literature. He attended Bristol Grammar School on scholarship before winning a place at Balliol College, Oxford, in 1905, where he studied Classics and Modern History. 
At Oxford, he came under the influence of historian Lord Acton and philologist J.R. Mayor, whose emphasis on textual precisio n and moral historiography shaped Winston’s 
intellectual trajectory. He graduated with first -class honours in 1909 and remained at Oxford for postgraduate research, focusing on the vernacular pamphlets of the Chartist
movement.
Winston’s academic career was briefly interrupted by the outbreak of the First World War. Commissioned as a lieutenant in the Royal Engineers, he served in France and Belgium, 
where his linguistic skills were employed in intelligence and translation. The war had a transformative effect on his worldvi ew; he later described the trenches as "a laboratory of
human expression under extremity," where he began collecting soldiers’ letters, graffiti, and field notes —materials that would l ater inform his methodological innovations. Injured 
during the Battle of the Somme in 1916, he was discharged and returned to academia.
In 1920, Winston was appointed lecturer in Historical Linguistics at the University of Manchester, where he began developing his theory of "peripheral discourse." He argued that
official historical records —government documents, chronicles, and elite memoirs —often obscured the lived experiences of ordinary people. Instead, he advocated for the systematic 
study of "residual texts": private correspondence, shopkeepers’ ledgers, parish records, and oral testimonies. His 1927 monog raph V oices from the Margin: Language and Class in 
Victorian England was among the first scholarly works to treat informal writing as a legitimate object of historical analysis .
Winston’s magnum opus, The Silent Text, published in 1938, synthesized two decades of archival research across Britain, Franc e, and Germany. The book examined how 
marginalized communities —industrial workers, rural women, colonial migrants —used language to assert identity and resist dominant narratives. He introduced the concept of 
"linguistic latency," the idea that meaning in subaltern texts is often encoded through metaphor, silence, or grammatical dev iation, requiring specialized interpretive frameworks. The 
work received both acclaim and criticism: while praised for its methodological rigor, some contemporaries dismissed it as ove rly speculative or lacking in political clarity.
During the Second World War, Winston worked with the BBC’s European Service, analyzing propaganda and monitoring public senti ment in occupied territories. After the war, he 
returned to Oxford as Professor of Cultural History, a newly established chair reflecting the growing recognition of his fiel d. He mentored a generation of scholars, including E.P. 
Thompson and Raymond Williams, who acknowledged his influence on the development of British cultural studies.
Winston retired in 1957 but continued writing and lecturing. He published Echoes of the Unwritten in 1962, a reflective work on the ethics of historical interpretation. He died in 
Oxford on 4 November 1965, survived by his wife, Margaret (née Carr), whom he married in 1913, and their two children. Though not a public intellectual in the traditional sense, 
Winston’s legacy endures in the methodologies of digital humanities, oral history, and postcolonial studies. In 2005, the Int ernational Society for Cultural History established the
annual Winston Prize for outstanding contributions to the study of everyday discourse.
Arthur Edgar Winston (12 March 1887 – 4 November 1965) was a British historian, linguist, and cultural theorist best known for h is pioneering work on the "discursive margins" of 
19th and early 20th century Europe. His interdisciplinary approach to cultural history, emphasizing the recovery and analysis of non-canonical texts, profoundly influenced the
development of microhistory and the linguistic turn in historical studies during the late 20th century. Winston is most celeb rated for his seminal work The Silent Text: Forgotten 
Narratives of the 1930s (which redefined how historians interpret modern voices in historical discourse).
Born in Bristol, England to a modest family —his father was a railway clerk and his mother —a schoolteacher showed early aptitudefor languages and classics. He attended Bristol
Grammar School on scholarship before winning a place at Balliol College, Oxford, in 1905, where he studied Classics and Moder n History. At Oxford, he came under the influence 
of historian Lord Acton and philosopher J.R. Pole, whose emphasis on textual precision and moral historiography shaped Winsto n’s intellectual trajectory. He graduated with first -
class honours in 1909 and remained at Oxford for postgraduate research, focusing on the vernacular pamphlets of the Chartist movement.
Winston's academic career was briefly interrupted by the outbreak of the First World War. Commissioned as a lieutenant in the Royal Engineers, he served in France and Belgium, 
where his skills in linguistic intelligence were employed in the war. A transformative effect had on his worldview; later he described the trenches as "a laboratory of human 
expression under extremity," where he began collecting soldiers’ letters, graffiti, and field notes —materials that would later i nform his methodological innovations. Injured during the 
Battle of the Somme in 1916, he was discharged and returned to academia.
In 1920, Winston was appointed lecturer in Historical Linguistics at the University of Manchester, where he began developing his theory of "peripheral discourse." He argued that 
official historical records —government documents, chronicles, and elite memoirs —often obscured the lived experiences of ordinary people. Instead, he advocated for the systematic 
study of "resurgent texts": private correspondence, shopkeepers’ ledgers, parish records, and oral testimonies. His 1927 mono graph V oices from the Margin: Language and Class in
Victorian England was among the first scholarly works to treat informal writing as a legitimate object of historical analysis .
Winston's magnum opus, The Silent Text, published in 1938, synthesized two decades of archival research across Britain, Franc e, and Germany. The book examined how the 
marginalized communities —industrial workers, rural women, colonial migrants —used language to assert identity and resist dominant narratives. He introduced the concept of "
"linguistic latency, the idea that meaning in modern texts is often encoded through metaphor, silence, or grammatical deviati on, requiring specialized interpretive frameworks. The 
work received both acclaim and criticism: while praised for its methodological rigor, some contemporaries dismissed it as ove rly speculative or lacking in political clarity.
During the Second World War, Winston worked with the BBC’s European Service, analyzing propaganda and monitoring public opini on in occupied territories. After the war, he
returned to Oxford as Professor of Historical Linguistics, a newly established chair reflecting the growing recognition of hi s field. He mentored a generation of scholars, including 
E.P. Thompson, Raymond Williams, and who acknowledged his influence on the development of British cultural studies.
Winston retired in 1957 but continued writing and lecturing. He published Echoes of the Unwritten in 1962, a reflective work on the ethics of historical interpretation. He died in
Oxford on 4 November 1965, survived by his wife, Margaret (née Carr), whom he married in 1913, and their two children. Though not a public intellectual in the traditional sense, 
Winston's legacy endures in the methodologies of digital humanities, oral history, and postcolonial studies. In 2005, the Int ernational Society for Cultural Studies established the 
annual Winston Prize for outstanding contributions to the study of everyday discourse.MoM Reconstruction (8x Compression)Source TextFigure 11: Illustration of the source text alongside its reconstructed counterpart generated by the MoM model.
17
# ImpRAG: Retrieval-Augmented Generation with Implicit Queries

**Authors**: Wenzheng Zhang, Xi Victoria Lin, Karl Stratos, Wen-tau Yih, Mingda Chen

**Published**: 2025-06-02 21:38:21

**PDF URL**: [http://arxiv.org/pdf/2506.02279v1](http://arxiv.org/pdf/2506.02279v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems traditionally treat retrieval
and generation as separate processes, requiring explicit textual queries to
connect them. This separation can limit the ability of models to generalize
across diverse tasks. In this work, we propose a query-free RAG system, named
ImpRAG, which integrates retrieval and generation into a unified model. ImpRAG
allows models to implicitly express their information needs, eliminating the
need for human-specified queries. By dividing pretrained decoder-only language
models into specialized layer groups, ImpRAG optimizes retrieval and generation
tasks simultaneously. Our approach employs a two-stage inference process, using
the same model parameters and forward pass for both retrieval and generation,
thereby minimizing the disparity between retrievers and language models.
Experiments on 8 knowledge-intensive tasks demonstrate that ImpRAG achieves
3.6-11.5 improvements in exact match scores on unseen tasks with diverse
formats, highlighting its effectiveness in enabling models to articulate their
own information needs and generalize across tasks. Our analysis underscores the
importance of balancing retrieval and generation parameters and leveraging
generation perplexities as retrieval training objectives for enhanced
performance.

## Full Text


<!-- PDF content starts -->

arXiv:2506.02279v1  [cs.CL]  2 Jun 2025
ImpRAG: Retrieval-Augmented Generation with Implicit Queries
Wenzheng Zhang1*Xi Victoria Lin2Karl Stratos1Wen-tau Yih2Mingda Chen2
1Rutgers University2FAIR, Meta
{wenzheng.zhang, karl.stratos}@rutgers.edu
{victorialin,scottyih,mingdachen}@meta.com
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems traditionally treat retrieval and generation
as separate processes, requiring explicit textual
queries to connect them. This separation can
limit the ability of models to generalize across
diverse tasks. In this work, we propose a query-
free RAG system, named ImpRAG, which in-
tegrates retrieval and generation into a unified
model. ImpRAG allows models to implicitly
express their information needs, eliminating the
need for human-specified queries. By dividing
pretrained decoder-only language models into
specialized layer groups, ImpRAG optimizes
retrieval and generation tasks simultaneously.
Our approach employs a two-stage inference
process, using the same model parameters and
forward pass for both retrieval and generation,
thereby minimizing the disparity between re-
trievers and language models. Experiments on
8 knowledge-intensive tasks demonstrate that
ImpRAG significantly enhances both retrieval
and generation performance, with exact match
scores increasing by 3.6-11.5 points and re-
trieval recalls improving by 5.0-23.2 points for
unseen tasks with diverse formats, highlighting
its effectiveness in enabling models to articu-
late their own information needs and generalize
across tasks. Our analysis underscores the im-
portance of balancing retrieval and generation
parameters and leveraging generation perplexi-
ties as retrieval training objectives for enhanced
performance.
1 Introduction
Retrieval-Augmented Generation (RAG; Guu et al.,
2020; Lewis et al., 2020; Shi et al., 2024) typi-
cally involves two key operations: retrieval and
generation. RAG systems retrieve relevant infor-
mation to enhance generation models, enabling
them to respond more effectively to prompts by
providing long-tail knowledge or up-to-date infor-
mation. While effective, traditional approaches
*Work done during an internship at Meta
P a s s a g e  
K e y - V a l u e  S t a t e sT o p - k  P a s s a g e sR e t r i e v eL a t e n t  Q u e r y  V e c t o rC r o s s  A t t e n t i o n
E U  r e j e c t s  G e r m a n  c a l l  t o  b o y c o t t  [ S T A R T ]  B r i t i s h  [ E N D ]  l a m b   . . .  
P l e a s e  o u t p u t  t h e  W i k i p e d i a  t i t l e  o f  t h e  e n t i t y  m e n t i o n e d  b e t w e e n  
[ S T A R T ]  a n d  [ E N D ]  i n  t h e  g i v e n  t e x t  L a r g e  L a n g u a g e  M o d e lU n i t e d  K i n g d o mFigure 1: Diagram illustrating the inference process of
ImpRAG on the entity linking task. We divide decoder-
only LLMs into three layer groups for specialized fine-
tuning: bottom (green), middle (red), and top (blue).
The bottom layers are optimized for retrieval tasks. The
middle and top layers handle the reading of retrieved
passages, with cross-attention disabled in the top layers
to reduce memory consumption. Standard RAG sys-
tems would require a task-specific design of queries
(e.g., use the substring “British” as the query in the
shown example). In contrast, ImpRAG uses implicit
queries, eliminating the need for explicit specification
of queries and allowing models to generalize across un-
seen tasks with varied formats.
often treat retrieval and generation as separate pro-
cesses, connected by queries.1Consequently, these
approaches usually require explicit specification of
textual queries. By definition, queries express one’s
uncertainties; however, in RAG systems, instead of
models expressing their information needs, humans
must do this for them. This separation can lead to
a disconnect between what large language models
1In this work, we use the term “queries” to refer to textual
queries used in an information retrieval setup, unless otherwise
specified. This is distinct from queries in the context of self-
attention within Transformer architectures.
1

(LLMs) require and what retrievers assume is nec-
essary. More importantly, it restricts the models’
ability to generalize across diverse, unseen tasks
during testing. Therefore, in this work, we explore
the development of a query-free RAG system, en-
abling models to articulate their own information
needs without additional human intervention.
To achieve this, we introduce ImpRAG, a novel
approach that integrates retrieval and generation
into a unified model and process. This allows mod-
els to convey their own information needs implic-
itly, reducing the need for prior knowledge of test
tasks and for humans to formulate explicit textual
queries in advance. At its core, ImpRAG aims to
enable retrieval capabilities through retrieval heads
in self-attention. Building upon pretrained decoder-
only language models, ImpRAG divides the layers
into three groups: the bottom group for retrieval
and the middle and top groups for reading and gen-
eration.
Figure 1 illustrates an example of applying Im-
pRAG to the entity linking task, where models are
tasked with linking the mention "British" to an en-
tity in Wikipedia, given the context paragraph. A
typical RAG model would require the design of
a separate query template, such as using only the
mention text, to achieve reasonable retrieval perfor-
mance. In contrast, ImpRAG uses implicit queries
and can perform retrieval and generation jointly
without the need for additional template design,
making it more generalizable.
During training, we optimize two objectives si-
multaneously: generation loss and retrieval loss.
The generation loss is the standard causal language
modeling loss, while the retrieval loss first utilizes
pseudo labels generated by trained retrievers to
warm up the retrieval ability and then self-improves
using its own generation log likelihood for the re-
mainder of the training.
At inference time, we employ a two-stage pro-
cess. First, we embed passages using the bottom
layer for retrieval, and then utilize the top layer
group to read the retrieved passages and generate
the final responses. By leveraging the same forward
pass and model parameters for both retrieval and
generation, ImpRAG reduces the disparity between
retrievers and LLMs.
In experiments, we train models on datasets that
either require retrieval or do not. The datasets re-
quiring retrieval are used to enhance retrieval per-
formance, while those not requiring retrieval are
used to improve models’ instruction-following ca-pabilities. We evaluate the models on 8 knowledge-
intensive tasks, focusing on different aspects: ba-
sic question answering, multihop reasoning, and
instruction following. We also establish strong
baselines that perform RAG in the retrieve-then-
generate paradigm, including RA-DIT (Lin et al.,
2024), a method that iteratively updates LLMs and
retrievers to better align the two.
Our experiments demonstrate that ImpRAG
achieves slightly better performance on 4 tasks with
formats similar to the training tasks, with an im-
provement of 0.2-0.6 points in exact match scores,
all without the need for additional model parame-
ters. Moreover, it significantly outperforms previ-
ous approaches on unseen test tasks with more di-
verse formats, achieving improvements of 3.6-11.5
points in exact match scores and 5.0-23.2 points in
retrieval recalls. This highlights the effectiveness
of enabling models to articulate their own informa-
tion needs. Our analysis indicates that carefully
selecting layer group boundaries that balance the
parameters used for retrieval and generation, using
both trained retrievers for warmup and then self-
improve by leveraging generation perplexities as
retrieval training objectives, and instruction tun-
ing training datasets is crucial for achieving supe-
rior performance in ImpRAG. Our analysis also
reveals that ImpRAG is effective in transferring
supervision from generation tasks to retrieval tasks,
showing the potential of using an unified model ar-
chitecture for performing retrieval and generation
jointly.
2 Related Work
There has been a lot of work on using the retrieve-
then-generate paradigm for RAG (Lewis et al.,
2020; Shi et al., 2024, inter alia ). Many efforts
in this line of work have focused on optimizing re-
trievers using training signals from generation mod-
els, and optionally, the reverse (Guu et al., 2020;
Lewis et al., 2020; Izacard et al., 2023; Shi et al.,
2024). Although the specifics can differ, these ap-
proaches generally utilize distinct models and input
templates for the retrieval and generation phases.
A closely related study is that of Jiang et al. (2022),
which seek to use the same model for retrieval and
generation. However, their research primarily fo-
cuses on encoder-decoder style models and their
models still rely on separate input templates for
retrieval and generation. Another related work by
Zhang et al. (2024a) explores the use of special
2

tokens for retrieval, but their study emphasizes in-
domain task performance rather than unseen task
generalization.
This work is also related to research on query
formulation in the context of multihop question
answering, where previous studies typically gener-
ate textual queries by prompting LLMs, followed
by retrieval using a separate retriever (Lazaridou
et al., 2022; Khattab et al., 2022; Press et al., 2023;
Trivedi et al., 2023; Jiang et al., 2023, inter alia ).
Chen et al. (2024) enable LLMs to generate textual
queries through synthetic data generation. Addi-
tionally, this work is connected to memory archi-
tectures in RAG (Yang et al., 2024; Lu et al., 2024),
which aim to utilize the key-value (KV) caches of
LLMs to reduce computational costs, rather than
focusing on minimizing the disparities between
generation and retrieval.
Another relevant area of research is instruction
tuning for RAG. Lin et al. (2024) perform instruc-
tion tuning for both retrievers and LLMs and then
align them through iterative updates. Wang et al.
(2024) conduct instruction tuning for RETRO-like
models (Borgeaud et al., 2022; Wang et al., 2023).
Zhang et al. (2024b) align retrievers with LLMs
using synthetic data generated by LLMs. Unlike
our work, these studies still treat retrieval and gen-
eration as separate processes. In a similar vein,
researchers have tried to teach retrievers to follow
instructions for building general-purpose informa-
tion retrieval systems (Asai et al., 2023; Lee et al.,
2024; Oh et al., 2024; Weller et al., 2025). Since
ImpRAG enables its retrieval capabilities by using
self-attention, it is related to research on investigat-
ing retrieval heads in the context of long context
LLMs (Wu et al., 2024).
3 Method
We build on an autoregressive pretrained language
model and enable it to perform retrieval and gen-
eration jointly. Our model, ImpRAG , is based on
the LLaMA 3 family (Grattafiori et al., 2024), with
architectural modifications to support retrieval and
retrieval-augmented generation. At a high level,
the layer grouping strategy of ImpRAG is inspired
by the observation that LLMs learn distinct func-
tions at different layers (Zhao et al., 2024). Con-
sequently, we have designed the layer groups to
align with the capabilities required for retrieval-
augmented generation, i.e., retrieval and genera-
tion.3.1 Architecture
Layer Slicing. We partition an N-layer language
model vertically into three groups, as illustrated in
Figure 1. The bottom group, spanning layers 0tob,
is denoted as LB. The middle group, from layer bto
t, is denoted as LM, and the top group, from layer
t+ 1toN−1, asLT. Note that LBandLMshare
layer b, while LMandLTare disjoint. The layer
boundaries bandtare treated as hyperparameters
and can be tuned to optimize performance across
different model configurations.
Bottom Layers as Retriever. We repurpose the
bottom group LBto act as a retriever , in addition to
its standard decoder functionality. Specifically, we
apply pooling last-token pooling over the attention
query or key states at the final layer binLB. Unlike
prior work (Muennighoff et al., 2024), we retain
the original causal attention in the bottom layers
rather than enabling bidirectional attention, as we
do not observe any performance improvement from
this modification.
Lethkbe the number of key attention heads,
gthe number of query attention groups (as in
Grouped-Query Attention (Ainslie et al., 2023)),
anddhthe head dimension. For a query input, we
apply last-token pooling by taking the query atten-
tion state of its final token, resulting in a grouped
query embedding Eg
q∈R(hkg)dh. We then average
the attention heads within each group to obtain the
final query embedding Eq∈Rhkdh.2Similarly,
for each corpus passage, we extract the key atten-
tion state of its last token to compute the passage
embedding Ep∈Rhkdh. Similarity between
query and passage embeddings is computed via dot
product:
s(q, p) =Eq·Ep (1)
We choose to pool over query and key attention
states based on the intuition that their dot prod-
uct underlies the attention mechanism and is pre-
trained to capture token-wise relevance. By ag-
gregating these signals across tokens, we aim to
capture query-passage-level semantic relevance.
Middle Layers as Reader. The middle layer
groupLMfunctions as a reader by enabling cross-
attention from the input query tokens to the re-
trieved passage tokens, thereby incorporating ex-
ternal information into the query representation.
2Our preliminary results show that taking average heads
works slightly better than using individual heads.
3

Given kretrieved passages, we jointly encode the
concatenation of all kpassages to form the key and
value states for layers bthrough t. Cross-attention
is then performed from the query’s attention states
to these key and value states, allowing the model
to read and integrate relevant content from the pas-
sages. This aligns with prior findings that middle
layers of language models are particularly effective
at attending to and integrating long-range contex-
tual information (Fang et al., 2024; Yang et al.,
2024).
Top Layers Disable Cross-Attention. In the
top layer group LT, we optionally disable cross-
attention from the input query tokens to the re-
trieved passage tokens solely to reduce computa-
tional and memory overhead. This design choice
is made for efficiency purposes; empirically, we
find it results in only a minor performance drop
when the layer boundary tis properly tuned as a
hyperparameter.
Position IDs. Language models using RoPE (Su
et al., 2024) are highly sensitive to position IDs. To
prevent interference between the query and passage
position encodings during reading, we shift the
query’s position IDs to the right rather than starting
from zero. Let lmaxdenote the maximum passage
length and kthe number of retrieved passages. We
shift the query position IDs by k·lmaxtokens to
account for the total length.
3.2 Training
We train ImpRAG using a multi-task objective that
jointly optimizes generation and retrieval:
J=Jgen(r|q,C) +λ·Jret(q,C) (2)
Here, Jgen(r|q,C)denotes the generation loss,
implemented as the standard causal language mod-
eling loss over the response tokens r, conditioned
on the input query qand a set of sampled candi-
date passages C. The term Jret(q,C)denotes the
retrieval loss, computed over the query qand the
same set of candidate passages C, and is further
detailed in the two-stage formulation described in
Section 3.2.1. The hyperparameter λbalances the
relative importance of the retrieval loss, allowing us
to control the trade-off between retrieval accuracy
and generation quality during training.3.2.1 Retrieval Objective
While the overall training objective remains consis-
tent across both stages—combining generation and
retrieval losses as in (2)—the retrieval loss compo-
nentJretvaries depending on the training phase. In
this section, we describe the two-stage training pro-
cess used to endow ImpRAG with strong retrieval
capabilities.
Warmup. Since the pretrained language model
is not inherently optimized for retrieval, we begin
with a warmup stage that introduces basic retrieval
ability. We adopt a Multi-Label NCE loss (Zhang
et al., 2022) as the retrieval objective and con-
struct supervision using pseudo-labeled data gener-
ated by a strong off-the-shelf retriever, Contriever-
MSMARCO (Izacard et al., 2022). For each query
q, we retrieve the top-5 passages as pseudo-positive
examples, denoted by P(q). We then sample a
small set of pseudo hard negatives, denoted by
Nh(q)(e.g.,|Nh(q)|<10), from passages ranked
10–50.3While these passages may still be some-
what relevant, they are less likely to contain the key
information necessary to answer the query. This
selection introduces meaningful retrieval difficulty.
We also use in-batch negatives across devices as ad-
ditional random negatives Nr(q). The full negative
set isN(q) =Nh(q)∪ Nr(q), and the candidate
set isC=P(q)∪ N(q). The retrieval loss for this
stage is defined as:
Jret(q,C) =−P
p∈P(q)log
exp(s(q,p))
exp(s(q,p))+P
p′∈N(q)exp(s(q,p′))
(3)
Self-Distillation. To further enhance retrieval
performance, we employ language model perplex-
ity distillation (Izacard et al., 2023), which as-
sesses how much each candidate passage improves
the language model’s likelihood of generating the
ground-truth response, conditioned on the query.
Specifically, for each candidate passage p∈ C, we
compute the log-likelihood of the gold response
rgiven the concatenation of pandq, denoted as
logPLM(r|p, q). This defines a soft target distri-
bution over candidate passages:
PT(p|q, r) =exp(log PLM(r|p, q))P
p′∈Cexp(log PLM(r|p′, q))
(4)
3We find this approach effective in preliminary experi-
ments, though we did not perform extensive hyperparameter
tuning.
4

We also define the retrieval model’s predicted
distribution based on the similarity scores:
PR(p|q) =exp(s(q, p))P
p′∈Cexp(s(q, p′))(5)
The retrieval loss is then computed as the KL
divergence between the target and predicted distri-
butions:
Jret(q,C) = KL
PT(p|q, r)∥PR(p|q)
(6)
Here, PT(p|q, r)indicates that gradients are
not backpropagated through the target distribution.
Note that this stage also involves joint training; the
only difference from the warmup phase lies in the
retrieval loss Jret.
3.3 Inference
At inference time, we first embed all passages in the
knowledge corpus using the bottom layer group LB
of the model. These embeddings are stored in an
approximate nearest neighbor (ANN) index (e.g.,
FAISS (Douze et al., 2024)) hosted on a remote
server for efficient retrieval.
As illustrated in Figure 1, given a query, the
ImpRAG model performs the following steps to
generate a response:
1.The bottom layers LBencode the input query
and generate a query embedding, which is
sent to the remote ANN search server.
2.The ANN server retrieves the top- kmost rele-
vant passages based on the query embedding
and returns their passage IDs.
3.The middle layers LMcontinue processing
the information by applying cross-attention to
the KV states of the retrieved passages.
4.The top layers LTcomplete the encoding and
decoding process without cross-attention, gen-
erating the next token.
5.The above steps are repeated at each decoding
step. Notably, the query embeddings are com-
puted only once at the end of the input prompt,
and passage retrieval is not re-triggered there-
after.4In subsequent decoding steps, cross-
attention continues to use the cached key-
value states, and this process repeats until the
4While ImpRAG is general and can be adapted for iterative
retrieval, we intend to focus this work on the single retrieval
setup and will leave iterative retrieval for future work.model reaches a stopping criterion (e.g., an
end-of-sequence token).
4 Experiment
4.1 Experimental Setup
Training. For training, we consider two types
of datasets: (1) datasets requiring retrieval knowl-
edge: NaturalQuestions (NQ; Kwiatkowski et al.,
2019) and HotpotQA (Hopo; Yang et al., 2018);
and (2) datasets without requiring retrieval knowl-
edge, where we use the instruction tuning datasets
from Lin et al. (2024) (see Appendix D for a com-
plete list of these datasets). Inspired by Chen
et al. (2022), we also incorporate two synthetic,
retrieval-free tasks into the training to enhance
instruction-following capabilities: phrase denois-
ing, and next/previous sentence generation. The
training data for phrase denoising is generated by
prompting LLMs (we use Llama-3.1 70B) with a
paragraph from Wikipedia. For the sentence gener-
ation task, we construct it randomly using content
from Wikipedia.
For all these datasets, we use a subset of 5,000
examples from their training splits in each dataset.
In addition, we use 1,000 examples from the NQ
dev split as the development set. We use the De-
cember 2021 Wikipedia from Izacard et al. (2023)
as our knowledge corpus. Additionally, we spend
approximately 10% of training on plain text from
Wikipedia to prevent models from overfitting to the
downstream tasks.
Evaluation. We evaluate models on 8 different
knowledge-intensive tasks to assess their various
capabilities, specifically:
•Basic question answering: NQ, Sim-
pleQA (SQA; Wei et al., 2024);
•Multihop reasoning: Hopo, 2WikiMulti-
HopQA (2WQA; Ho et al., 2020);
•Instruction following: (1) relation extraction: T-
Rex (Elsahar et al., 2018), ZsRE (Levy et al.,
2017), (2) fact checking: FEVER (FEV; Thorne
et al., 2018), and (3) entity linking: AIDA (Hof-
fart et al., 2011).
For all these datasets, we report exact matches
as the evaluation metric for generation tasks and
recall rates for retrieval tasks. The retrieval recall
is measured by the percentage of instances where
the top-retrieved results contain the answers as sub-
strings. We omit retrieval recall for FEV as it is
5

Task Template
Knowledge-Intensive Tasks
NQ, Hopo, SQA, 2WQA Q: {question} A: {answer}
AIDA {context} Output the Wikipedia page title of the entity mentioned between
[START] and [END] in the given text A: {answer}
FEV Is this statement true? {statement} A: {answer}
T-Rex, ZsRE {entity} [SEP] {relation} Provide the answer corresponding to the relation
specified after [SEP] for the entity mentioned before [SEP] A: {answer}
Instruction-Tuning Tasks
Dialogue Completion {turn 1} {turn 2} {turn 3} ...
Reading Comprehension {context} Q: {question} A: {answer}
Summarization {context} Summarize this article: {summary}
Phrase Denoising {context} Recover the original phrases marked between [START] and [END]
in the given text A: {answer}
Sentence Generation {context} [SEP] next/previous sentence Generate a sentence corresponding to
the relation specified after [SEP] for the context mentioned before [SEP] A:
{sentence}
Table 1: Prompt templates. We only use retrieval for knowledge-intensive tasks. For simplicity, we list task
categories for a subset of the instruction tuning datasets. See Appendix D for more detailed description.
NQ SQA Hopo 2WQA T-Rex ZsRE FEV AIDA avg
Llama-3.2 3B
+RA-IT 43.2 (77.0) 38.1 (48.2) 35.9 (48.8) 33.4 (43.3) 54.2 (84.3) 58.1 (86.6) 79.2 (-) 40.1 (38.1) 47.8 (60.9)
+RA-DIT 43.4 (77.5) 38.8 (48.7) 36.4 (49.3) 34.0 (43.5) 55.0 (85.0) 59.0 (87.2) 80.5 (-) 41.0 (38.2) 48.5 (61.3)
+RA-DIT-Llama 43.9 (78.0) 39.8(49.9) 37.0 (49.8) 35.1 (44.0) 55.8 (85.9) 60.0 (87.9) 80.2 (-) 41.1 (38.4) 50.4 (64.0)
+ImpRAG 44.1 (78.4) 40.3 (50.0) 37.3 (50.2) 35.5 (44.5) 60.8 (90.2) 65.4 (93.2) 83.8 (-) 52.6 (58.3) 52.5 (66.4)
Llama-3.1 8B
+RA-IT 45.1 (77.0) 39.0 (48.2) 36.9 (48.8) 34.4 (43.3) 55.0 (84.3) 59.1 (86.6) 83.2 (-) 41.1 (38.1) 49.2 (60.9)
+RA-DIT 45.7 (77.7) 38.9 (48.9) 37.2 (49.1) 34.9 (44.0) 56.1 (85.4) 60.1 (87.8) 85.1 (-) 41.5 (38.8) 49.9 (61.7)
+RA-DIT-Llama 46.1 (78.7) 40.7 (50.3) 37.9 (50.2) 35.6 (44.8) 57.0 (86.1) 61.2 (88.1) 86.2 (-) 42.1 (39.2) 50.9 (62.5)
+ImpRAG 46.4 (79.1) 41.3 (51.2) 38.4 (50.9) 36.0 (45.2) 62.5 (92.7) 67.1 (94.0) 89.2 (-) 54.2 (62.4) 54.4 (67.9)
Table 2: Evaluation results for 8 knowledge-intensive tasks. We report exact match scores for generation tasks
and retrieval recall (shown in parentheses) for retrieval tasks. Retrieval recall is not reported for FEV, as it is a
classification task. All these methods use retrieval augmentation.
a classification task where the answer strings are
either “True” or “False”.
For Hopo, T-Rex, ZsRE, FEV, and AIDA,
we use development sets from the KILT bench-
mark (Lewis et al., 2020). For SQA and NQ, we
use the official test set. For 2WQA, we use their
development set. For all datasets, we utilize the
entire input prompts as queries for the retrievers.
We describe our task templates in Table 1.
Baselines. We consider 3 baseline models:
•Retrieval Augmented Instruction Tuning (RA-
IT): This approach involves directly incor-
porating retrieved passages from Contriever-
MSMARCO into the context and fine-tuning the
language models (LMs) on the training data;
•Retrieval Augmented Dual Instruction Tuning
(RA-DIT; Lin et al., 2024): In this method, we
first fine-tune the Contriever-MSMARCO on
the training subsets of NQ and HotpotQA us-
ing Equation 6. Subsequently, we perform fine-
tuning as in RA-IT, utilizing the fine-tuned re-
triever;•RA-DIT with Llama as the Retriever (RA-DIT-
Llama): Here, we replace the Contriever used in
RA-DIT with the first 8 layers from the Llama
models.5To ensure effective retrieval perfor-
mance, we initially warm up the Llama retriev-
ers with pseudo labels generated by Contriever-
MSMARCO using Equation 3.
Hyperparameters. We use Llama-3.2 3B and
Llama-3.1 8B as the base models for ImpRAG. For
both models, the layer boundary bis set to 7.6For
Llama-3.2 3B, the layer boundary tis 19, while for
Llama-3.1 8B, it is 23. We train for 10 epochs and
perform the retrieval warmup in the first 3 epochs.
When retrieving passages, we take the top 10 most
relevant documents.
See Appendix E for more details on the baselines
and computational resources.
5We choose to use first 8 layers for fair comparison as
ImpRAG uses the same layers for retrieval.
6Since we label the first layer of a LLM as layer 0, a layer
boundary bof 7 means that the bottom layer group contains
the first 8 layers.
6

2 4 6 8 10 12
Layer Boundary b304050607080
27.449.1
29.555.1
30.562.4
30.761.9
36.369.0
40.275.0
45.181.0
44.581.5
44.082.1
44.182.2
43.281.6
42.180.5
Contriever Retrieval Recall
Exact Match
Retrieval Recall
10 12 14 16 18
Layer Boundary t304050607080
30.270.2
32.571.4
34.274.3
35.574.9
37.976.2
41.377.2
41.177.1
42.578.3
44.080.7
44.580.9
45.181.0
Exact Match
Retrieval Recall
Contriever Retrieval RecallFigure 2: Exact match and retrieval recall on the NQ dev set using Llama-3.2 3B with different values of b(left
side) and t(right side). When varying one layer boundary, we keep the other constant.
NQ SQA Hopo 2WQA T-Rex ZsRE FEV AIDA avg
self-distillation only 29.9 (61.2) 30.1 (39.9) 29.8 (41.9) 27.5 (37.4) 35.6 (64.9) 40.9 (65.9) 67.7 (-) 28.3 (22.5) 36.2 (47.7)
warmup only 44.0 (78.3) 39.9(50.0) 37.1 (50.0) 35.1 (44.2) 56.5 (87.0) 61.2 (88.3) 81.0 (-) 45.2 (42.9) 50.0 (63.0)
warmup+self-distillation 44.1 (78.4) 40.3 (50.0) 37.3 (50.2) 35.5 (44.5) 60.8 (90.2) 65.4 (93.2) 83.8 (-) 52.6 (58.3) 52.5 (66.4)
Table 3: Exact match scores and retrieval recall (shown in parentheses) for ImpRAG using Llama-3.2 3B as the base
model, trained with different retrieval objectives.
4.2 Experimental Result
Table 2 presents our main evaluation results. Each
model variant—RA-IT, RA-DIT, RA-DIT-Llama,
and ImpRAG —exhibits different performance lev-
els, with ImpRAG consistently achieving the high-
est scores across all tasks. For Llama-3.2 3B, the
average exact match score increases from 47.8 with
RA-IT to 52.5 with ImpRAG, while for Llama-3.1
8B, the score rises from 49.2 to 54.4. Although RA-
DIT shows improvements over RA-IT, ImpRAG
further enhances performance. Notably, ImpRAG
significantly outperforms RA-DIT-Llama, indicat-
ing that the improvements are not merely due to
using a more powerful base model (i.e., the first
8 layers of Llama models) for retrieval. Impor-
tantly, the enhancements are evident in both exact
match scores and retrieval recalls, demonstrating
that ImpRAG improves both generation quality and
retrieval performance. It is worth noting that com-
pared to the baseline approaches, the most substan-
tial improvements with ImpRAG are seen in tasks
that queries need to be formulated more differently
from input prompts, such as T-Rex, ZsRE, FEVER,
and AIDA. Among these tasks, AIDA shows the
most significant improvements, with over a 20-
point increase in retrieval recall and more than a 10-
point rise in exact match scores for both Llama-3.1
3B and Llama-3.1 8B, likely due to the inadequacy
of directly using input prompts as queries in AIDA.
This underscores ImpRAG’s effectiveness in formu-lating implicit queries and embedding instruction-
following capabilities into retrievers. Overall, these
results demonstrate that ImpRAG significantly en-
hances the models’ ability to accurately retrieve
and apply knowledge, with improvements more
significant in tasks requiring diverse formats.
5 Analysis
5.1 Layer Group Boundary Ablation
In this section, we examine the effects of the layer
boundaries bandt. The findings are presented in
Figure 2. To facilitate comparison, we vary one
layer boundary while keeping the other constant.
We note that increasing breduces the number of
layers allocated to the middle layer group, which
includes layers for reading and generation. Con-
versely, increasing tdoes not affect the retrieval lay-
ers. Overall, we find that increasing benhances re-
trieval recall, with improvements leveling off once
breaches 7. This plateau is likely due to dimin-
ished generation performance, which results in less
precise training signals for self-distillation. This
underscores the importance of balancing parame-
ters between retrieval and generation. On the other
hand, as expected, increasing tconsistently yields
improvements. Although these improvements seem
to plateau at 19, we refrain from further increasing
tprimarily due to memory constraints. We plan to
leave more memory-efficient training of ImpRAG
for future exploration.
7

T-Rex ZsRE FEV AIDA avg
No templates 55.8 (85.9) 60.0 (87.9) 80.2 (-) 41.1 (38.4) 59.3 (70.7)
Oracle templates 61.4 (90.7) 66.0 (93.6) 83.9 (-) 66.1 (72.3) 69.4 (85.5)
ImpRAG 60.8 (90.2) 65.9 (93.5) 83.8 (-) 52.6 (58.3) 65.8 (80.7)
Table 4: Exact match scores and retrieval recall (shown in parentheses) for RA-DIT-Llama using Llama-3.2 3B as
the base model, evaluated with various query templates. In the case of “no templates”, the inputs to the LLMs are
used directly as queries.
NQ SQA Hopo 2WQA T-Rex ZsRE FEV AIDA avg
ImpRAG 44.1 (78.4) 40.3 (50.0) 37.3 (50.2) 35.5 (44.5) 60.8 (90.2) 65.4 (93.2) 83.8 (-) 52.6 (58.3) 52.5 (66.4)
w/o all IT tasks 42.9 (76.4) 38.1 (48.2) 35.2 (47.7) 33.7 (42.0) 43.5 (69.3) 49.5 (70.2) 76.2 (-) 25.4 (20.5) 43.1 (53.5)
w/o PD and SG 44.0 (78.5) 40.1 (50.1) 37.4 (50.3) 35.4 (44.4) 53.3 (82.8) 57.1 (84.9) 81.2 (-) 40.5 (41.2) 48.6 (61.7)
Table 5: Exact match scores and retrieval recall (shown in parentheses) for ImpRAG using Llama-3.2 3B as the base
model, trained with different combinations of instruction tuning datasets. IT tasks refer to instruction tuning tasks,
PD stands for phrase denoising, and SG denotes sentence generation.
5.2 Retrieval Objective Ablation
We conduct experiments to compare the effects of
different retrieval training objectives. The results
are presented in Table 3. During training, we con-
sistently apply each retrieval objective throughout
the entire process. For instance, in the "warmup
only" experiment, we extend the use of the warmup
objective to 10 epochs instead of limiting it to
the initial 3 epochs. Our findings indicate that
the warmup objective provides a baseline perfor-
mance across all tasks and is particularly beneficial
for tasks with direct supervision. Self-distillation
builds on this baseline, further enhancing model
performance on unseen test tasks. Overall, the two
training objectives complement each other effec-
tively.
5.3 Effect of Query Templates
We also examine the impact of using different
query templates for the baseline approach, RA-
DIT-Llama. The results are detailed in Table 4.
In these experiments, we omit QA tasks because
their “no templates” and “oracle templates” setups
are almost the same. Overall, “oracle templates”
still provides the best performance. The improve-
ments are particularly notable on AIDA. However,
it is important to highlight that ImpRAG achieves
highly competitive performance on 3 out of 4 tasks
and already shows significant improvement on the
remaining task compared to using “no templates.”
5.4 Effect of Instruction Tuning for Retrieval
In Table 5, we explore the effects of training on
instruction tuning datasets. The table shows that
omitting all instruction tuning datasets leads to adecline in model performance on both in-domain
tasks (NQ, SQA, Hopo, and 2WQA) and out-of-
domain tasks. Notably, removing only phrase de-
noising and sentence generation has a minimal
impact on in-domain tasks but causes more pro-
nounced negative effects on out-of-domain tasks,
except for FEV. This exception likely arises be-
cause FEV’s task format is more similar to the
in-domain tasks than other tasks. This suggests
that instruction tuning tasks aid models in under-
standing task formats, and ImpRAG can transfer
this knowledge from generation to retrieval due to
its unified model architecture.
6 Conclusion
We present ImpRAG, a query-free retrieval-
augmented generation (RAG) system that implic-
itly captures information needs without requiring
human-specified queries. Unlike prior work that
treats retrieval and generation as separate com-
ponents with independently trained models, Im-
pRAG unifies them within a single decoder-only
language model by partitioning it into specialized
layer groups and jointly optimizing for both re-
trieval and generation. The same model parame-
ters and forward pass are shared across retrieval
and generation, effectively minimizing the mis-
match between the retriever and the generator. Im-
pRAG demonstrates strong performance across
eight knowledge-intensive tasks, outperforming tra-
ditional RAG systems and delivering substantial
gains on unseen tasks with diverse formats.
8

7 Limitations
One limitation of this work is its focus on a single-
pass retrieval setup; we do not explore iterative or
multi-hop retrieval, which could further enhance
performance on complex reasoning tasks. Adapting
ImpRAG to iterative retrieval remains an important
direction for future work.
Our method is also evaluated exclusively using
the LLaMA 3 family of models. While the ap-
proach is broadly applicable, its generalizability to
other architectures and model sizes has yet to be
validated.
Additionally, the warmup stage relies on pseudo-
labeled data generated by Contriever-MSMARCO.
Although this provides a strong starting point,
we expect that using more powerful retrievers or
human-labeled data could lead to further gains by
offering higher-quality supervision early in train-
ing.
References
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury
Zemlyanskiy, Federico Lebron, and Sumit Sanghai.
2023. GQA: Training generalized multi-query trans-
former models from multi-head checkpoints. In Pro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing , pages 4895–
4901, Singapore. Association for Computational Lin-
guistics.
Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen,
Gautier Izacard, Sebastian Riedel, Hannaneh Ha-
jishirzi, and Wen-tau Yih. 2023. Task-aware retrieval
with instructions. In Findings of the Association for
Computational Linguistics: ACL 2023 , pages 3650–
3675, Toronto, Canada. Association for Computa-
tional Linguistics.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, Diego
De Las Casas, Aurelia Guy, Jacob Menick, Roman
Ring, Tom Hennigan, Saffron Huang, Loren Mag-
giore, Chris Jones, Albin Cassirer, and 9 others. 2022.
Improving language models by retrieving from tril-
lions of tokens. In Proceedings of the 39th Interna-
tional Conference on Machine Learning , volume 162
ofProceedings of Machine Learning Research , pages
2206–2240. PMLR.
Danqi Chen, Jason Bolton, and Christopher D. Manning.
2016. A thorough examination of the CNN/Daily
Mail reading comprehension task. In Proceedings
of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 2358–2367, Berlin, Germany. Association for
Computational Linguistics.Mingda Chen, Xilun Chen, and Wen-tau Yih. 2024.
Few-shot data synthesis for open domain multi-hop
question answering. In Proceedings of the 18th Con-
ference of the European Chapter of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers) , pages 190–208, St. Julian’s, Malta. Associa-
tion for Computational Linguistics.
Mingda Chen, Jingfei Du, Ramakanth Pasunuru, Todor
Mihaylov, Srini Iyer, Veselin Stoyanov, and Zor-
nitsa Kozareva. 2022. Improving in-context few-shot
learning via self-supervised training. In Proceedings
of the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies , pages 3558–3573,
Seattle, United States. Association for Computational
Linguistics.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The faiss library. arXiv preprint
arXiv:2401.08281 .
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel
Stanovsky, Sameer Singh, and Matt Gardner. 2019.
DROP: A reading comprehension benchmark requir-
ing discrete reasoning over paragraphs. In Proceed-
ings of the 2019 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, Volume 1
(Long and Short Papers) , pages 2368–2378, Min-
neapolis, Minnesota. Association for Computational
Linguistics.
Hady Elsahar, Pavlos V ougiouklis, Arslen Remaci,
Christophe Gravier, Jonathon Hare, Frederique Lafor-
est, and Elena Simperl. 2018. T-REx: A large scale
alignment of natural language with knowledge base
triples. In Proceedings of the Eleventh International
Conference on Language Resources and Evaluation
(LREC 2018) , Miyazaki, Japan. European Language
Resources Association (ELRA).
Junjie Fang, Likai Tang, Hongzhe Bi, Yujia Qin, Si Sun,
Zhenyu Li, Haolun Li, Yongjian Li, Xin Cong,
Yankai Lin, Yukun Yan, Xiaodong Shi, Sen Song,
Zhiyuan Liu, and Maosong Sun. 2024. Unimem: To-
wards a unified view of long-context large language
models. In First Conference on Language Modeling .
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. In Proceedings of the
37th International Conference on Machine Learning ,
volume 119 of Proceedings of Machine Learning
Research , pages 3929–3938. PMLR.
9

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. In Proceedings of the 28th Inter-
national Conference on Computational Linguistics ,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bordino,
Hagen Fürstenau, Manfred Pinkal, Marc Spaniol,
Bilyana Taneva, Stefan Thater, and Gerhard Weikum.
2011. Robust disambiguation of named entities in
text. In Proceedings of the 2011 Conference on Em-
pirical Methods in Natural Language Processing ,
pages 782–792, Edinburgh, Scotland, UK. Associa-
tion for Computational Linguistics.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense informa-
tion retrieval with contrastive learning. Transactions
on Machine Learning Research .
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine
Learning Research , 24(251):1–43.
Zhengbao Jiang, Luyu Gao, Zhiruo Wang, Jun Araki,
Haibo Ding, Jamie Callan, and Graham Neubig.
2022. Retrieval as attention: End-to-end learning
of retrieval and reading within a single transformer.
InProceedings of the 2022 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2336–2349, Abu Dhabi, United Arab Emirates. As-
sociation for Computational Linguistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. PubMedQA: A
dataset for biomedical research question answering.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the
9th International Joint Conference on Natural Lan-
guage Processing (EMNLP-IJCNLP) , pages 2567–
2577, Hong Kong, China. Association for Computa-
tional Linguistics.
Omar Khattab, Keshav Santhanam, Xiang Lisa
Li, David Hall, Percy Liang, Christopher Potts,
and Matei Zaharia. 2022. Demonstrate-search-
predict: Composing retrieval and language mod-
els for knowledge-intensive nlp. arXiv preprint
arXiv:2212.14024 .
Andreas Köpf, Yannic Kilcher, Dimitri von Rütte,
Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens,Abdullah Barhoum, Duc Minh Nguyen, Oliver
Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri,
David Alexandrovich Glushkov, Arnav Varma Dan-
tuluri, Andrew Maguire, Christoph Schuhmann, Huu
Nguyen, and Alexander Julian Mattick. 2023. Ope-
nassistant conversations - democratizing large lan-
guage model alignment. In Thirty-seventh Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Angeliki Lazaridou, Elena Gribovskaya, Wojciech
Stokowiec, and Nikolai Grigorev. 2022. Internet-
augmented language models through few-shot
prompting for open-domain question answering.
arXiv preprint arXiv:2203.05115 .
Yoonsang Lee, Minsoo Kim, and Seung-won Hwang.
2024. Disentangling questions from query genera-
tion for task-adaptive retrieval. In Findings of the
Association for Computational Linguistics: EMNLP
2024 , pages 4775–4785, Miami, Florida, USA. Asso-
ciation for Computational Linguistics.
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke
Zettlemoyer. 2017. Zero-shot relation extraction via
reading comprehension. In Proceedings of the 21st
Conference on Computational Natural Language
Learning (CoNLL 2017) , pages 333–342, Vancouver,
Canada. Association for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Advances in Neural Infor-
mation Processing Systems , volume 33, pages 9459–
9474. Curran Associates, Inc.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia
Shi, Maria Lomeli, Richard James, Pedro Rodriguez,
Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. 2024. RA-DIT:
Retrieval-augmented dual instruction tuning. In The
Twelfth International Conference on Learning Repre-
sentations .
Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and
Yaohua Tang. 2024. Turborag: Accelerating retrieval-
augmented generation with precomputed kv caches
for chunked text. arXiv preprint arXiv:2410.07590 .
Niklas Muennighoff, SU Hongjin, Liang Wang, Nan
Yang, Furu Wei, Tao Yu, Amanpreet Singh, and
10

Douwe Kiela. 2024. Generative representational in-
struction tuning. In ICLR 2024 Workshop: How Far
Are We From AGI .
Hanseok Oh, Hyunji Lee, Seonghyeon Ye, Haebin Shin,
Hansol Jang, Changwook Jun, and Minjoon Seo.
2024. Instructir: A benchmark for instruction follow-
ing of information retrieval models. arXiv preprint
arXiv:2402.14334 .
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.
Know what you don‘t know: Unanswerable ques-
tions for SQuAD. In Proceedings of the 56th Annual
Meeting of the Association for Computational Lin-
guistics (Volume 2: Short Papers) , pages 784–789,
Melbourne, Australia. Association for Computational
Linguistics.
Siva Reddy, Danqi Chen, and Christopher D. Manning.
2019. CoQA: A conversational question answering
challenge. Transactions of the Association for Com-
putational Linguistics , 7:249–266.
Anna Rogers, Olga Kovaleva, Matthew Downey, and
Anna Rumshisky. 2020. Getting closer to ai complete
question answering: A set of prerequisite real tasks.
Proceedings of the AAAI Conference on Artificial
Intelligence , 34(05):8722–8731.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-
augmented black-box language models. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers) , pages 8371–8384, Mexico City,
Mexico. Association for Computational Linguistics.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,
Wen Bo, and Yunfeng Liu. 2024. Roformer: En-
hanced transformer with rotary position embedding.
Neurocomputing , 568:127063.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. In Proceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers) , pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Adam Trischler, Tong Wang, Xingdi Yuan, Justin Har-
ris, Alessandro Sordoni, Philip Bachman, and Kaheer
Suleman. 2017. NewsQA: A machine comprehen-
sion dataset. In Proceedings of the 2nd Workshop
on Representation Learning for NLP , pages 191–200,Vancouver, Canada. Association for Computational
Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers) ,
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Boxin Wang, Wei Ping, Lawrence Mcafee, Peng Xu,
Bo Li, Mohammad Shoeybi, and Bryan Catanzaro.
2024. InstructRetro: Instruction tuning post retrieval-
augmented pretraining. In Proceedings of the 41st
International Conference on Machine Learning , vol-
ume 235 of Proceedings of Machine Learning Re-
search , pages 51255–51272. PMLR.
Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee,
Zihan Liu, Mohammad Shoeybi, Yi Dong, Oleksii
Kuchaiev, Bo Li, Chaowei Xiao, Anima Anandku-
mar, and Bryan Catanzaro. 2023. Shall we pretrain
autoregressive language models with retrieval? a
comprehensive study. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7763–7786, Singapore. As-
sociation for Computational Linguistics.
Jason Wei, Nguyen Karina, Hyung Won Chung,
Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John
Schulman, and William Fedus. 2024. Measuring
short-form factuality in large language models. arXiv
preprint arXiv:2411.04368 .
Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle
Lo, Arman Cohan, Benjamin Van Durme, Dawn
Lawrie, and Luca Soldaini. 2025. FollowIR: Eval-
uating and teaching information retrieval models to
follow instructions. In Proceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) ,
pages 11926–11942, Albuquerque, New Mexico. As-
sociation for Computational Linguistics.
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao
Peng, and Yao Fu. 2024. Retrieval head mechanisti-
cally explains long-context factuality. arXiv preprint
arXiv:2404.15574 .
Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu,
Zhiyu Li, Bo Tang, Wenqiang Wei, Jinbo Wang,
Zeyun Tang, Shichao Song, and 1 others. 2024. Mem-
ory3: Language modeling with explicit memory.
arXiv preprint arXiv:2407.01178 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
11

Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Long-
context language modeling with parallel context en-
coding. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 2588–2610.
Jintian Zhang, Cheng Peng, Mengshu Sun, Xiang Chen,
Lei Liang, Zhiqiang Zhang, Jun Zhou, Huajun Chen,
and Ningyu Zhang. 2024a. OneGen: Efficient one-
pass unified generation and retrieval for LLMs. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2024 , pages 4088–4119, Miami,
Florida, USA. Association for Computational Lin-
guistics.
LingXi Zhang, Yue Yu, Kuan Wang, and Chao Zhang.
2024b. ARL2: Aligning retrievers with black-box
large language models via self-guided adaptive rele-
vance labeling. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 3708–3719,
Bangkok, Thailand. Association for Computational
Linguistics.
Wenzheng Zhang, Wenyue Hua, and Karl Stratos. 2022.
EntQA: Entity linking as question answering. In In-
ternational Conference on Learning Representations .
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, and 1 oth-
ers. 2023. H2o: Heavy-hitter oracle for efficient
generative inference of large language models. Ad-
vances in Neural Information Processing Systems ,
36:34661–34710.
Zheng Zhao, Yftah Ziser, and Shay B Cohen. 2024.
Layer by layer: Uncovering where multi-task learn-
ing happens in instruction-tuned large language mod-
els. In Proceedings of the 2024 Conference on Empir-
ical Methods in Natural Language Processing , pages
15195–15214, Miami, Florida, USA. Association for
Computational Linguistics.
A Passage Encoding
Given kretrieved passages, we must obtain the key-
value (KV) states from the middle layer group LM
to enable cross-attention. We explore three passage
encoding strategies, summarized in Table 6.
First, we consider Independent Encoding, where
each passage is encoded separately using position
IDs starting from zero, following the parallel en-
coding strategy in Yen et al. (2024). The resulting
KV states are then concatenated across passages.
Second, we examine Concatenated Encoding
(Segmented), in which passages are concatenated
into a single sequence, but attention across pas-
sages is blocked to prevent inter-passage interac-
tion.
Third, we evaluate Concatenated Encoding (Full
Attention), where passages are concatenated andfull cross-passage attention is allowed throughout
the encoding.
We conduct these experiments by finetuning
Llama-3.1 8B model on the Natural Questions
(NQ) dataset using the top-10 passages retrieved by
Contriever-MSMARCO, and report Exact Match
(EM) scores on the development set. As shown in
Table 6, the two simpler strategies—Independent
Encoding and Segmented Concatenation—perform
similarly, while Full Attention Concatenation
yields a clear performance improvement, highlight-
ing the benefit of modeling inter-passage dependen-
cies.
Encoding Method Dev EM
Independent Encoding 51.7
Segmented Concatenation 51.4
Full Attention Concatenation 53.3
Table 6: Performance of different passage encoding
strategies.
B Freezing Passage Representations
We investigate the impact of freezing passage rep-
resentations—either hidden states or key-value
(KV) states—during inference with a fixed retriever.
All experiments are conducted using a fine-tuned
LLaMA-3.1 8B model and the top-10 passages re-
trieved by Contriever-MSMARCO on the Natural
Questions (NQ) dataset. Results are reported in
Table 7.
We explore two freezing strategies, both using
the Independent Encoding approach described in
Appendix A. In the first variant, Frozen Hidden
States, we freeze the hidden representations of re-
trieved passages as produced by the initial (un-
trained) LLaMA-3.1 8B model, and pass them
through the trained key/value projection layers to
generate the KV states used in cross-attention.
In the second variant, Frozen KV States, we
directly freeze the key and value attention states of
the passages, also obtained from the initial LLama-
3.1 8B model.
We observe that both freezing methods yield
comparable performance, slightly underperforming
the fully dynamic setting where passage KV states
are computed using the trained model.
C Passage KV States Compression
When we use independent encoding strategy in
Appendix A, one benefit will be that we can save
12

Method Dev EM
No Freezing 51.4
Frozen Hidden States 50.8
Frozen KV States 50.7
Table 7: Performance of freezing different passage
representations on NQ dev set with top-10 Contriever-
MSMARCO retrieved passages.
the middle layer group key value states for all the
passages in knowledge corpus in disk and during
inference after retrieval we can load the key value
states from disk without recomputation. However,
this will result in a large amount of disk spaces.
Thus, we consider two compression strategies: to-
ken compression and product quantization and we
conduct experiments following the same setting
as the Frozen KV states in Appendix B. Specif-
ically, take for token compression, we use the
Heavy Hitter (Zhang et al., 2023) and only keep
half number of tokens for each passage. For pro-
duction quantization, we use FAISS codec with in-
dex type OPQ32x128-PQ32x8 for each key value
head, which is trained on 500k randomly sampled
wikipedia passages. The compression rate with
this quantization is128×2
32= 8for original bfloat16
state vector of each attention head. We report the
results in Table 8. We can see that both strategies
don’t hurt the performance much.
Compression Dev EM
No Compression 50.7
Heavy Hitter 49.9
Product Quantization 50.3
Table 8: The results for various compression techniques.
D Instruction-Tuning Datasets
We use OpenAssistant Conversations
Dataset (oasst1; Köpf et al., 2023), Conversational
Question Answering (CoQA; Reddy et al., 2019),
Discrete Reasoning Over Paragraphs (DROP; Dua
et al., 2019), NewsQA (Trischler et al., 2017),
PubMedQA (Jin et al., 2019), QA for Artificial
Intelligence (Quail; Rogers et al., 2020), SQuAD
v2 (Rajpurkar et al., 2018),7and CNN Daily-
Mail (Chen et al., 2016) The templates for these
datasets are shown in Table 9.
7We only use answerable questions from SQuAD v2.Task Template
Instruction-Tuning Tasks
oasst1 {turn 1} {turn 2} {turn 3} ...
CoQA, DROP, NewsQA, PubMedQA,
SQuAD{context} Q: {question} A: {answer}
CNN DailyMail {context} Summarize this article: {summary}
Table 9: Prompt templates. We only use retrieval for
knowledge-intensive tasks.
E Baselines and Computational
Resources
Discussions on Baselines. For all these baselines,
we use the retrieve-then-generate paradigm, i.e., be-
gin by retrieving candidates using the retrievers and
then incorporate them into the context for training
and inference. This implies that these baselines
require an additional retriever, leading to increased
computational costs and a higher number of model
parameters compared to ImpRAG. However, since
this is a standard practice for retrieval-augmented
models, we continue to use them in the baselines
to establish stronger comparisons.
Computational Resources. We use NVIDIA
H100 GPUs. Each training session requires 8 H100
GPUs, and hosting the index also demands an ad-
ditional 8 GPUs. Training the baseline approaches
takes roughly 96 GPU hours, whereas our models
require approximately 160 GPU hours.
13
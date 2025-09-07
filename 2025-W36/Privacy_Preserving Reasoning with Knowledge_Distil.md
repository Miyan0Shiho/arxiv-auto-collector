# Privacy-Preserving Reasoning with Knowledge-Distilled Parametric Retrieval Augmented Generation

**Authors**: Jinwen Chen, Hainan Zhang, Liang Pang, Yongxin Tong, Haibo Zhou, Yuan Zhan, Wei Lin, Zhiming Zheng

**Published**: 2025-09-01 03:23:57

**PDF URL**: [http://arxiv.org/pdf/2509.01088v1](http://arxiv.org/pdf/2509.01088v1)

## Abstract
The current RAG system requires uploading plaintext documents to the cloud,
risking private data leakage. Parametric RAG (PRAG) addresses this by encoding
documents as LoRA within LLMs, enabling reasoning without exposing raw content.
However, it still faces two issues: (1) PRAG demands synthesizing QA pairs and
fine-tuning LLM for each individual document to create its corresponding LoRA,
leading to unacceptable inference latency. (2) The performance of PRAG relies
solely on synthetic QA data, lacking internal alignment with standard RAG,
resulting in poor generalization on out-of-distribution(OOD) inputs. Therefore,
achieving high-efficiency parameterization while maintaining RAG-level
performance remains a critical challenge for privacy-preserving reasoning. In
this paper, we propose DistilledPRAG, a generalizable knowledge-distilled
parametric RAG model aligned with standard RAG in document structure and
parameter activation. We first synthesize QA pairs from single and
multi-documents to enhance cross-document reasoning. Then, we mask the
plaintext documents with a special token and translate them to LoRA via a
parameter generator, maintaining the standard RAG document structure. Finally,
guided by synthetic QA data, we train the parameter generator to match standard
RAG's hidden states and output logits, enabling RAG-style reasoning without
original documents. Experiments on four QA datasets show that DistilledPRAG
outperforms baselines in accuracy and generalizes well on OOD data.

## Full Text


<!-- PDF content starts -->

Privacy-Preserving Reasoning with Knowledge-Distilled
Parametric Retrieval Augmented Generation
Jinwen Chen1,2Hainan Zhang1,2∗Liang Pang3Yongxin Tong1
Haibo Zhou4Yuan Zhan4Wei Lin4Zhiming Zheng1,2
1Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing
2School of Artificial Intelligence, Beihang University
3Institute of Computing Technology, Chinese Academy of Sciences
4Meituan
{jwkami, zhanghainan}@buaa.edu.cn
Abstract
The current RAG system requires upload-
ing plaintext documents to the cloud, risk-
ing private data leakage. Parametric RAG
(PRAG) addresses this by encoding docu-
ments as LoRA within LLMs, enabling rea-
soning without exposing raw content. How-
ever, it still faces two issues: (1) PRAG
demands synthesizing QA pairs and fine-
tuning LLM for each individual document
to create its corresponding LoRA, leading
tounacceptable inference latency . (2) The
performance of PRAG relies solely on syn-
thetic QA data, lacking internal alignment
with standard RAG, resulting in poor gen-
eralization on out-of-distribution(OOD) in-
puts. Therefore, achieving high-efficiency
parameterization while maintaining RAG-
level performance remains a critical chal-
lenge for privacy-preserving reasoning. In
this paper, we propose DistilledPRAG, a
generalizable knowledge-distilled paramet-
ric RAG model aligned with standard RAG
in document structure and parameter activa-
tion. We first synthesize QA pairs from sin-
gle and multi-documents to enhance cross-
document reasoning. Then, we mask the
plaintext documents with a special token
and translate them to LoRA via a parame-
ter generator, maintaining the standard RAG
document structure. Finally, guided by
synthetic QA data, we train the parameter
generator to match standard RAG’s hidden
states and output logits, enabling RAG-style
reasoning without original documents. Ex-
periments on four QA datasets show that
DistilledPRAG outperforms baselines in ac-
curacy and generalizes well on OOD data1.
1 Introduction
Retrieval-Augmented Generation (RAG) enables
real-time integration of external knowledge into
∗Corresponding author
1https://github.com/JWQZ/DistilledPRAG-arxiv
⊕
QuestionQA···
QA···
QA···Synthesis SFT
Sum
Translator
AvgIn-Context Injection
<|doc_mask |>
Distilled 
PRAG
RAG
PRAG
⊕
Question
Question
Question
DyPRAG
Translator
···
LoRA1
LoRA2
LoRA3Figure 1: Inference Paradigms for standard RAG,
PRAG, DyPRAG, and our DistilledPRAG. (1)
Standard RAG inputs the plaintext documents and
question. (2) PRAG generates QA pairs per docu-
ment to fine-tune LoRA adapters, and sums them
to obtain document aggregated representations for
LLM injection. (3) DyPRAG translates individ-
ual documents to its LoRA and averages them
to achieve document aggregation for LLM injec-
tion. (4) DistilledPRAG concatenates documents
to parameter generator to create cross-document
LoRA, and masks documents with question as in-
put, more similar to standard RAG.
LLMs and is widely used across fields like fi-
nance (Zhang et al., 2023), law (Louis et al.,
2024), and materials science (Buehler, 2024).
However, current RAG systems require uploading
local documents to the cloud, raising privacy con-
cerns when handling sensitive information such as
corporate contracts, medical records, or personal
notes (Zeng et al., 2024; Wang et al., 2025; ZhengarXiv:2509.01088v1  [cs.CL]  1 Sep 2025

et al., 2024). Therefore, a new RAG paradigm
without plaintext is necessary to support privacy-
preserving reasoning.
Recently, parametric RAG (PRAG) (Su et al.,
2025) is proposed to encode documents into
LoRA2parameters (Hu et al., 2022) and upload
to a cloud server, thereby avoiding the transmis-
sion of plaintext documents. As shown in Fig-
ure 1, PRAG first synthesizes QA pairs for an in-
dividual document and then fine-tunes LLM with
them to obtain its corresponding LoRA parame-
ters. At inference, PRAG aggregates the LoRAs
of retrieved documents to answer questions. How-
ever, the need for both data synthesis and fine-
tuning per document leads to unacceptable infer-
ence latency in real-world scenarios.
To reduce this latency, DyPRAG (Tan et al.,
2025) proposes training a dynamic parameter
translator to replace data synthesis and fine-tuning
at test-time. It still synthesizes QA pairs and fine-
tunes LLM to obtain the corresponding LoRA like
PRAG, but differs by training a linear parameter
translator to map each document to its LoRA. As
shown in Figure 1, during inference, each retrieved
document is processed by the parameter translator
to generate its LoRA, which is then averaged and
injected into the LLM to generate the final answer.
However, both PRAG and DyPRAG rely solely
on synthetic QA pairs to activate knowledge
learning, lacking internal alignment with standard
RAG, which may lead to poor generalization on
out-of-distribution(OOD) inputs. This misalign-
ment appears in two ways: (1) Document struc-
ture: training LoRA on individual documents and
aggregating them disrupts cross-document reason-
ing. For instance, in Figure 1, summing or av-
eraging LoRA1, LoRA2, and LoRA3 will miss
cross-document reasoning cues, leading to incor-
rect answers(see in Section 5.2). (2) Parameter
activation: performing inference only with query
alters internal parameter activations compared to
standard RAG, losing original reasoning capabil-
ity. Therefore, efficient parameterization with
RAG-level reasoning remains a major challenge
for parametric RAG.
In this paper, we introduce DistilledPRAG, a
knowledge-distilled parametric RAG model by
aligning the document structure and parameter ac-
2LoRA (Low-Rank Adaptation) is a technique for effi-
ciently fine-tuning LLMs by adding small trainable low-rank
matrices to frozen pretrained weights, significantly reducing
computational cost.tivations with a standard RAG for more robust
generalization on OOD data. Specifically, we
first use DeepSeek-V3 to synthesize QA pairs
for individual documents and concatenated multi-
documents to enhance cross-document reasoning.
Then, we replace plaintext document tokens with
a special token and translate them to LoRA by
our LongT5-based parameter generator, forming
the student model with the same document struc-
ture as standard RAG(teacher). Finally, guided by
the synthetic QA data, we train the parameter gen-
erator by minimizing the differences between the
student and teacher in both hidden states and out-
put logits, enabling it to learn RAG-style reason-
ing without access to the original documents. In
this setting, the student model inherits the teacher
model’s structure and activations, allowing rapid
learning of standard RAG reasoning.
Experiments on four public QA datasets
demonstrate that DistilledPRAG, trained only on
2WQA, outperforms baselines on three OOD
datasets, validating its strong generalization ca-
pabilities. Further analysis on synthetic data and
alignment functions confirms the effectiveness of
our internal alignment mechanism. Our main con-
tributions are:
• We identify that internal alignment between
parametric RAG and standard RAG is cru-
cial, facilitating efficient and highly general-
izable document-specific parameter learning.
• We propose a knowledge-distilled paramet-
ric RAG by internal alignment with standard
RAG in terms of both document structure and
activation parameters, thereby enabling ro-
bust generalization to OOD data.
• Experiments show that DistilledPRAG deliv-
ers strong QA performance without upload-
ing plaintext documents, and exhibits a supe-
rior ability to convert unseen documents into
reasoning-capable LoRA modules.
2 Related Work
2.1 Document Compression
Document compression in RAG enables privacy-
preserving inference by encoding documents
into input embeddings for RAG reasoning.
xRAG (Cheng et al., 2024) compresses each doc-
ument to a single dense embedding, integrating
it into LLM via a lightweight projection, and

requires no retriever or LLM fine-tuning. CO-
COM (Rau et al., 2025) jointly trains a compressor
and LLM decoder to represent multiple contexts
with a few embeddings, supporting flexible com-
pression rates and efficient multi-document RAG.
PISCO (Louis et al., 2025) further streamlines
compression by using only sequence-level knowl-
edge distillation, requiring neither pretraining nor
annotated data, and achieves high compression
rates with minimal accuracy loss. However, com-
pressing knowledge-rich documents into embed-
dings often loses fine-grained semantics, while
larger document parameters is necessary to better
preserve full content.
2.2 Parametric RAG
Recently, Parametric Retrieval-Augmented Gen-
eration (PRAG) (Su et al., 2025) is proposed, en-
coding retrieved documents into model parame-
ters (e.g., via LoRA) instead of appending them
as context. This approach injects external knowl-
edge directly into the LLMs, treating retrieval as
parameter updates rather than input expansion. To
obtain the document’s parameters, PRAG requires
synthesizing QA pairs and fine-tuning LLMs for
each retrieved document, which has unacceptable
inference latency in real-world testing scenarios.
To address this, Dynamic PRAG (DyPRAG) (Tan
et al., 2025) is proposed to introduce a parame-
ter translator that maps documents to parameters
at test-time, enabling document-specific parame-
terization without additional fine-tuning or storage
overhead. Although DyPRAG improves flexibility
but often fails to generalize across OOD inputs,
limiting its ability to replace traditional RAG un-
der privacy constraints fully. To overcome these
issues, we propose a knowledge-distilled paramet-
ric RAG method by cross-document data augmen-
tation and aligning the model’s behavior across the
student and teacher models, significantly enhanc-
ing its generalization ability to OOD inputs.
3 Backgrounds
Now, we will introduce the process and notions
of standard RAG and PRAG. Let qdenote a user
query and D={d1,···, dN} ∈ C denote the top-
N documents retrieved from a large corpus Cvia a
retriever R. The standard RAG constructs the in-
put of LLMs as a concatenation of the documents
and the query:
x= [d1,···, dN;q], (1)and generates yvia LLM:
y= arg max
y′P(y′|x;θ), (2)
where θis the parameter of LLM. However, this
setup inevitably exposes private documents at in-
ference time, raising serious privacy concerns in
many practical scenarios.
In PRAG, given the query qand the retrieved
documents D, it synthesizes QA pairs from indi-
vidual document di∈Dand fine-tuning LLM to
obtain its LoRA ∆θi. Then, the resulting LoRAs
are summed to inject into LLM for reasoning y:
y= arg max
y′P(y′|q;θ+ ∆θ), (3)
∆θ=NX
i=1∆θi. (4)
4 Model
This section introduces the DistilledPRAG model,
covering synthetic data construction, knowledge
distillation design, and online inference proce-
dures, as shown in Figure 2.
4.1 Synthetic Data Construction
To supervise the parametric RAG model in an-
swering questions, we construct a large-scale
dataset with 289,079 synthetic QA pairs by
DeepSeek-V3 (Liu et al., 2024) as the training
dataset D. An example is: “Document: [his-
tory of Ada Lovelace] Question: What was Ada
Lovelace’s main contribution to computer sci-
ence? Answer: She is credited as the first com-
puter programmer...” The proceeds are as follows:
1. We randomly sample 30,000 documents from
the 2WikiMultihopQA (Ho et al., 2020) train-
ing set as Dwiki, covering diverse topics and
question styles.
2. For each document di∈ D wiki, we
use DeepSeek-V3 with carefully designed
prompts(see in Appendix) to automatically
generate 2–5 high-quality question–answer
pairs, donated as {(di, qij, aij)}where 2≤
j≤5, totally 139,723 single-document QA
pairs into our training dataset D.
3. To simulate realistic multi-document re-
trieval, for each di∈ D wiki, we ran-
domly sample a document d′
ifrom the cor-
pusDwiki and concatenate them. Then we

Document: Edson was born in 1996, when...
Question: What is the name of Edson's wife?
Answer: …Document: <| doc_mask |><|doc_mask |>…
Question: What is the name of Edson's wife?
Answer: …
Teacher LLM Student LLMEdson was born in 1996...
ℒ𝑐𝑜𝑠ℒ𝑔𝑒𝑛
deepseek -v3
𝑞𝑖1…
...…𝐷𝑖;𝑞𝑖;𝑎𝑖 <𝑑𝑜𝑐_𝑚𝑎𝑠𝑘>𝐷𝑖;𝑞𝑖;𝑎𝑖 𝐷𝑖
𝒟𝑤𝑖𝑘𝑖
𝑑𝑖𝑎𝑖1
𝑞𝑖j𝑎𝑖j𝑞𝑖j′
𝑎𝑖j′𝑑𝑖
𝑞𝑖1′
𝑎𝑖1′T5 Encoder
CrossAttention Query Emb
TransformerEncoder
FFN
Reshape
Merge ℒ𝐾𝐿ℎ𝑡 ℎ𝑠
𝑧𝑡 𝑧𝑠
𝑑𝑖′⊕Figure 2: The Architecture of DistilledPRAG Model. 1⃝Use DeepSeek-V3 to mine knowledge from a
single document and augmented cross-documents by random concatenation. 2⃝Train a parameter gener-
ator to map documents to a LoRA for student LLM, enabling it to mimic a teacher RAG’s reasoning by
minimizing differences in hidden states and logits on synthetic data.
use the same DeepSeek-V3 with carefully de-
signed prompts (see in Appendix) to auto-
matically generate at least 5 additional cross-
document QA pairs, focusing on reasoning
that spans both documents, donated as {(di+
d′
i, q′
ij, a′
ij)}where j≥5, totally 149,356
cross-document QA pairs into D.
This construction yields a challenging multi-hop
QA dataset with broad coverage, which encour-
ages the model to encode document semantics into
adapter parameters deeply. For the convenience of
formal definition, we uniformly define the training
set asD={(Di, qi, ai)}N
i=1, which includes both
single-document and cross-document QA pairs,
4.2 Knowledge Distillation
We define the teacher and the student mod-
els’ inputs and outputs for each training triple
(Di, qi, ai)∈ D as:
• The original input xi= [Di;qi]is passed
to the standard LLMs fθ, producing aias
teacher model’s output.
• The masked input ˜xi =
[<|doc_mask|>|Di|;qi]is passed to
the student’s LLMs fθ+∆θi, producing
aias output. Here, <|doc_mask|>|Di|
is formed by repeating the special token
<|doc_mask|> to match the length of
|Di|,∆θi=Tϕ(Di), where Tϕis a parameter
generator network to map documents Dito
its corresponding LoRA parameters ∆θi.
4.2.1 Special Token Initialization
To prevent raw document content from being
leaked during inference, we introduce a special to-
ken<|doc_mask|> which replaces all tokensin the raw document. Naive initialization (e.g.,
random values) often leads to unstable training
or degraded performance, as they are mismatched
with the distribution of the pretrained embedding
space( see in Experiments 5.3.3). Therefore, we
propose a special token initialization that aligns
with the first-order and second-order statistics of
the model’s pretrained vocabulary, thereby pro-
moting stable training.
Formally, let E∈RV×hbe the embedding ma-
trix of LLMs, where Vis the vocabulary size and
his the hidden state size. We compute the mean
and variance of the embedding distribution as:
µ=1
VVX
i=1Ei,σ=vuut1
VVX
i=1(Ei−µ)2.(5)
Then, we sample the special token’s embedding
from this distribution:
emask=µ+ϵ,ϵ∼ N(0,diag(σ2)),(6)
where diag (.)is the diagonal matrix.
After initialization, emask is frozen mask and
serves as a stable, information-free placeholder
used only for document masking, never appearing
in queries or answers.
4.2.2 Parameter Generator
Here,Tϕdenotes a parameter generator network
that transforms the full documents Diinto LoRA
weights ∆θi. Specifically, it consists of two com-
ponents: a document embedding model Encψand
a parameter generator Gen ω, such that:
Tϕ(Di) = Gen ω(Enc ψ(Di)), (7)

where Encψis a pretrained LongT5 encoder (Guo
et al., 2022) to encode documents Diinto em-
beddings EDi∈RL×d, where Lis the sequence
length of Dianddis the embedding dimension.
Parameter generator Gen ωmaps the document
embeddings EDito the target LoRA. Specifically,
given EDias key and value, a learnable query
embeddings Q∈R˜L×d, where ˜Lis the number
of hidden layers in LLM, we utilize the multi-
head cross-attention mechanism to selectively ag-
gregate information from the documents:
H0=CrossAttention (Q,EDi,EDi). (8)
Next, the output H0undergoes a self-attention-
based Transformer encoder, which models de-
pendencies and integrates information within the
query sequence itself:
H1=SelfAttentionEncoder (H0). (9)
Finally, a feed-forward network (FFN) maps the
encoded representation to the desired LoRA pa-
rameter space:
Tϕ(Di) = FFN( H1). (10)
This architecture enables efficient extraction,
aggregation, and projection of document knowl-
edge into model parameters. Importantly, both the
base LLM parameters θand the document encoder
parameters ψare frozen throughout training, and
only the generator Gen ωis updated.
4.2.3 Training Objectives
The training objectives of parameter generator are:
generative loss on synthetic QA data, alignment
loss of internal representations, and KL divergence
of output logits between the student model and the
teacher model.
Generative Loss: Given the documents Diand
the query qi, we minimize the negative log-
likelihood for generating the answer aifrom the
masked input ˜xi:
Lgen=−logP(ai|˜xi;θ+ ∆θi), (11)
∆θi=Tϕ(Di). (12)
Internal Alignment Loss: To align the internal
reasoning process, we match their hidden repre-
sentations at each layer between the student model
and the teacher model. Let h(i)
tandh(i)
sdenote thehidden states from the i-th hidden layer of teacher
model fθ(xi)and student model fθ+∆θi(˜xi), re-
spectively. The cosine similarity alignment loss
for the i-th layer is defined as:
L(i)
cos= 1−cos(h(i)
t, h(i)
s). (13)
The overall alignment loss is then computed as
the weighted average of the losses from all layers,
where the weight of the i-th layer is proportional
toi:
Lcos=PM
i=1iL(i)
cosPM
i=1i(14)
where Mis the total number of hidden layers in
LLM. We believe that the layers closer to the out-
put are more important.
KL Divergence of Output Logits: To fur-
ther ensure output consistency, we match the
token-level output distributions by minimizing
the Kullback-Leibler divergence (Kullback and
Leibler, 1951) between the logits from the student
model and the teacher model. Let ztandzsbe
the pre-softmax logits of teacher model and stu-
dent model for answer positions:
LKL=KL(softmax (zt)∥softmax (zs)).(15)
Finally, we optimize only the generator param-
etersωvia gradient descent over the above objec-
tives:Lgen,LcosandLKL. The base model and
document encoder remain frozen during training.
The overall objective can be written as:
min
ωE(Di,qi,ai)∼D
Lgen+λ1Lcos+λ2LKL
,
(16)
where λ1, λ2are hyperparameters that balance
hidden-states and output-logits alignment.
4.3 Online Inference
At inference time, our approach diverges from
previous methods such as PRAG and DyPRAG,
which require explicit fusion of LoRA parameters
from multiple documents. In PRAG, the LoRA A
and B matrices generated on each document are
concatenated along the LoRA rank dimension. In
contrast, DyPRAG directly averages the LoRA A
and B weights generated on each document. In-
stead, we adhere to the same input organization
paradigm as in our training phase. During the fu-
sion of LoRA weights, there may be loss or cor-
ruption of knowledge and information, as this fu-
sion process is not explicitly modeled during train-
ing in their methods. In contrast, our inference

paradigm remains consistent with training, help-
ing enhance generalization performance.
Given a user query qtest, we first employ a
retriever Rto obtain the top- krelevant docu-
ments {d1, . . . , d k}. These documents are con-
catenated in their original order to form a com-
posite document dinf= [d1;. . .;dk]. To preserve
privacy, we mask the entire dinfusing our spe-
cial token, resulting in the masked input ˜xinf=
[<|doc_mask|>|dinf|;qtest]. The composite doc-
ument is then passed to the parameter genera-
tor network Tϕ, which computes LoRA adapter
weights ∆θinf=Tϕ(dinf)conditioned on the com-
posite document. The LoRA-augmented model
fθ+∆θinfprocesses the masked input and generates
the final answer (prompts in the Appendix):
gtest=fθ+∆θinf(˜xinf). (17)
In summary, our online inference procedure is
fully aligned with the data format during training
and obviates the need for complex parameter fu-
sion schemes commonly required in prior works.
During inference, the document is never exposed
in plaintext, and only the dynamically generated
∆θiis used to condition the model.
5 Experiments
To demonstrate the performance of Distilled-
PRAG, we define the in-domain benchmark and
OOD datasets across four datasets and conduct
comparisons with baselines.
5.1 Experimental Setup
5.1.1 Datasets and Metrics
We evaluate question-answering performance us-
ing the F1 score(%) for our DistilledPRAG
method and baselines on four open-domain QA
datasets: 2WikiMultihopQA(2WQA) (Ho et al.,
2020), HotpotQA(HQA) (Yang et al., 2018),
PopQA(PQA) (Mallen et al., 2023), and Com-
plexWebQuestions(CWQ) (Talmor and Berant,
2018). 2WQA and HQA evaluate multi-hop rea-
soning by requiring integration of information
from multiple sources. PQA tests factual re-
call and entity disambiguation. CWQ assesses
multi-step reasoning over web-based content. The
2WQA and HQA datasets categorize questions by
reasoning type, with four sub-tasks for 2WQA and
two for HQA.
For a fair comparison between DistilledPRAG
and baselines, followed PRAG and DyPRAG, weuse the same first 300 questions from the dev split
of each sub-task dataset as the test set. But their
training sets are different. PRAG requires no
training and only loads offline-generated LoRA
parameters at test time. DyPRAG proposes to
train its parameter translator on questions 301–600
from each sub-task’s dev split, which has a distri-
bution similar to the test set. Notably, our Distilled
PRAG only use the training split of 2WQA as a
training dataset, and test on the 2WQA dev set as
the in-domain benchmark, while HQA, PQA, and
CWQ dev sets serve as OOD datasets for gener-
alization evaluation. Therefore, the generaliza-
tion evaluation of our model is more rigorous ,
demonstrating the effectiveness of our method.
5.1.2 Implementation Details
Our main experiments are conducted using two
LLM backbones: Llama-3.2-1B-Instruct and
LLaMA3-8B-Instruct . To compare with PISCO,
we use the same backbone LLM Mistral-7B-
Instruct-v0.2 . All experiments are implemented
in PyTorch and performed on NVIDIA A100
80GB PCIE and NVIDIA RTX PRO 6000 Black-
well Workstation Edition. We choose the encoder
of LongT53(Guo et al., 2022) as the embedding
model of the parameter generator. During train-
ing, we set the learning rate to 10−4, with 10%
of the total training steps used for warm-up. We
employ a polynomial scheduler, with the sched-
uler’s ending learning rate set to 10−6. The batch
size is set to 4, the number of training epochs
is 1, LoRA rank is 2, and the LoRA alpha pa-
rameter is set to 32. We use the AdamW opti-
mizer (Loshchilov and Hutter, 2019). We set λ1=
0.5andλ2= 0.1in our training loss. All other un-
specified hyperparameters follow their default set-
tings. During inference, we disable sampling by
setting do_sample=False and use greedy de-
coding. This ensures stable and deterministic out-
put across all baselines.
5.1.3 Baselines
For DistilledPRAG and all baselines, we adopt a
standard BM25 retriever (Robertson et al., 2009)
to fetch the top-3 documents for each question.
Our method and all baselines are as follows:
•Standard RAG : Using the retrieved docu-
ments as in-context input for backbone LLM.
3https://huggingface.co/google/long-t5-tglobal-base

Base LLM Method2WQA HQA
PQA CWQ Avg
Compare Bridge Inference Compose Bridge Compare
LLaMA-1BStandard RAG 34.9 32.7 28.2 7.9 18.9 27.5 17.8 29.1 24.6
PRAG 41.2 41.0 18.9 5.5 13.9 42.2 21.3 31.7 27.0
DyPRAG 31.3 20.9 22.3 7.2 10.4 21.3 9.7 23.2 18.3
DistilledPRAG 42.0 43.1 26.3 14.5 14.7 32.6 14.3 38.9 28.3
LLaMA-8BStandard RAG 28.7 37.1 31.9 8.9 33.6 55.2 33.0 41.9 33.8
PRAG 45.1 37.0 19.8 6.9 18.6 44.7 17.6 36.2 28.2
DyPRAG 44.7 41.9 21.6 12.4 14.9 47.1 12.4 41.4 29.6
DistilledPRAG 41.3 45.6 30.1 16.2 26.9 54.2 25.6 49.0 36.1
Mistral-7BStandard RAG 26.7 28.1 26.7 6.8 16.2 23.6 18.2 18.4 20.6
PISCO 28.9 27.9 27.9 6.6 16.8 24.0 14.6 26.0 21.6
DistilledPRAG 34.0 35.9 25.8 10.2 15.4 26.4 12.6 24.7 23.1
Table 1: Overall F1(%) performance of DistilledPRAG and baselines on 2WQA, HQA, PQA and CWQ
datasets. Bold indicates the best performance, and underlined indicates the second best.
•PRAG (Su et al., 2025): Parameterizing
each document from the corpus into LoRA
weights by offline synthesizing QA pairs
and fine-tuning LLM. During inference, they
retrieve the corresponding LoRAs for each
query-relevant document and sum them ac-
cording to the LoRA rank dimension.
•DyPRAG (Tan et al., 2025): During in-
ference, dynamically generating document-
specific LoRA using a translator network for
each document, and averaging them as the
aggregated document representation.
•PISCO (Louis et al., 2025): Optimized from
COCOM (Rau et al., 2025), compressing
each retrieved document into a few embed-
dings and concatenating them with the in-
put’s embedding for backbone LLM.
•DistilledPRAG : The parameter generator di-
rectly encodes the retrieved top-3 documents
to the unified multi-document LoRA, without
additional LoRA aggregation.
5.2 Main Results
Table 1 shows the QA performance of Distilled-
PRAG and baselines across multiple datasets and
backbone LLMs. We can see that DistilledPRAG
achieves the best or second-best performance in
the vast majority of sub-tasks and consistently ob-
tains the highest average F1 score across all back-
bone models. Taking LLaMA-8B backbone as
an example, DistilledPRAG achieves the highest
average F1 score, outperforming standard RAG,PRAG, and DyPRAG, i.e., 6.8%, 28% and 22%,
respectively.
Notably, DistilledPRAG is trained solely on
the 2WQA training set, whereas the baseline
DyPRAG is trained on a similar distribution
data with test data4. Nevertheless, Distilled-
PRAG demonstrates strong generalization capa-
bilities on OOD datasets, including HQA, PQA,
and CWQ. On the CWQ dataset, DistilledPRAG
achieves a leading score of 49.0%. Similar trends
are observed with LLaMA-1B and Mistral-7B,
where DistilledPRAG consistently outperforms or
matches the strongest baselines in both in-domain
and OOD scenarios. These results highlight that
DistilledPRAG excels not only in-domain but also
exhibits strong transferability and generalization
across diverse OOD datasets, underscoring its
practical value for real-world open-domain ques-
tion answering tasks.
The failure of PRAG and DyPRAG mainly
stems from: the knowledge coverage of the syn-
thesized QA is insufficient, as most documents
cannot have their key information adequately rep-
resented by just three QA pairs. Additionally,
DyPRAG suffers from a significant mismatch be-
tween its inference data paradigm and the pre-
training paradigm of the base LLM, resulting in
poor generalization. As for PISCO, its document
compression process greatly reduces the document
embedding representations, inevitably discarding
a substantial amount of information, including
knowledge required to answer QA tasks.
4DyPRAG proposes to train its parameter translator on
questions 301–600 from each sub-task’s dev split, which has
a distribution similar to the test set.

Method2WQA HQA
PQA CWQ AvgCompare Bridge Inference Compose Bridge Compare
DistilledPRAG 41.3 45.6 30.1 16.2 26.9 54.2 25.6 49.0 36.1
w/oLcos 41.0 46.3 29.2 15.7 25.8 51.2 22.8 46.0 34.7
w/oLKL 33.1 33.9 29.5 16.5 24.5 42.5 24.5 44.6 31.1
w/oLcos,LKL 30.5 30.5 30.2 17.2 24.3 39.8 23.3 44.5 30.0
Table 2: Ablation study of alignment losses based on LLaMA3-8B-Instruct backbone model.
Method2WQA HQA
PQA CWQ AvgCompare Bridge Inference Compose Bridge Compare
DistilledPRAG 42.0 43.1 26.3 14.5 14.7 32.6 14.3 38.9 28.3
Llama-Synthesis 39.5 42.6 21.6 13.6 15.1 30.5 14.9 37.7 26.9
Single-Document 33.8 42.4 27.0 12.9 14.7 29.7 12.7 38.2 26.4
Table 3: The impact of QA synthesis on model performance using Llama-3.2-1B-Instruct. Llama-
Synthesis uses Llama-3.2-1B-Instruct (vs. DeepSeek-V3) to generate QA data. Single-Document uses
only single-document QA pairs of training dataset D(see in Section 4.1).
5.3 Analysis
5.3.1 Alignment Loss
We conducted ablation studies on the alignment
losses LcosandLKL, as shown in Table 2. The
results demonstrate that removing either of these
losses leads to a noticeable drop in performance
across most tasks. Specifically, eliminating the
cosine similarity loss Lcosresults in an average
performance decrease of 3.9%, highlighting its
contribution to semantic alignment. Similarly,
discarding the KL divergence loss LKLcauses a
larger drop of 13.9%, indicating its importance in
maintaining distributional consistency. When both
losses are removed, the performance further de-
clines by 16.9%. Aligning the hidden states and
output logits further enables the model to match
the teacher’s internal representation and decision
boundaries. This alignment improves the model’s
sensitivity to the nuanced content of the docu-
ments and significantly enhances its generalization
ability to unseen or out-of-distribution data.
5.3.2 QA Synthesis
In Table 3, we investigate the benefits of DeepSeek
synthetic data and cross-document augmentation.
From the results, when using QA pairs synthesized
by Llama-3.2-1B, we observe an average perfor-
mance drop of 4.9%, indicating that the quality of
synthetic QA pairs is crucial for training the pa-
rameter generator. High-quality QA pairs better
capture the key facts and reasoning patterns withinthe document, providing more effective supervi-
sion during parameter learning. If the QA genera-
tion process only produces superficial or incom-
plete questions, the resulting LoRA parameters
may fail to encode critical document knowledge,
leading to suboptimal downstream performance.
Additionally, when restricting training to only
single-document QA pairs, the average perfor-
mance decreases by 6.7%. In this case, the diver-
sity and coverage of knowledge in training data are
significantly limited. Cross-document QA syn-
thesis introduces questions that require reason-
ing over multiple documents, thereby encourag-
ing the model to learn broader and more trans-
ferable representations. Without such augmenta-
tion, the model is more likely to overfit to narrow
document-specific patterns and generalize poorly
to complex or open-domain queries.
5.3.3 Special Token
In Table 4, we explore several configurations of
the document’s mask token, including no token,
random token, and trainable token. (1) Removing
documents without mask tokens obtains a substan-
tial drop in F1 score, i.e., 17.5%, indicating that
maintaining consistency with the model’s struc-
tural paradigm facilitates effective training of the
parameter generator. (2) We initialize the special
token’s embedding using the default initialization
function of Transformer package and keep it fixed:
emask∼ N 
0, σ2I
,where σis typically set to

Method2WQA HQA
PQA CWQ AvgCompare Bridge Inference Compose Bridge Compare
DistilledPRAG 42.0 43.1 26.3 14.5 14.7 32.6 14.3 38.9 28.3
No-token 35.5 29.8 25.3 12.5 13.6 23.3 13.3 33.5 23.3
Random-token 39.2 41.6 22.6 9.0 9.9 39.3 9.2 35.5 25.8
Trainable-token 31.9 39.2 22.8 8.4 13.7 25.3 15.3 31.7 23.5
Table 4: The impact of special tokens on masked documents using Llama-3.2-1B-Instruct: No-token
deletes content without masking; Random-token uses default Transformer initialization for special token;
Trainable-token has a learnable special token embedding.
MethodLatency
LLaMA-1B LLaMA-8B Mistral-7B
Standard RAG 0.14 0.52 0.76
PRAG 0.6(+18) 1.18(+100) -
DyPRAG 0.52 0.8 -
PISCO - - 0.92
DistilledPRAG 0.3 0.81 1
Table 5: Average inference latency (s). For
PRAG, extra time (in brackets) is added for offline
LoRA synthesis and training when new documents
are introduced.
0.02, and Iis the identity matrix matching the em-
bedding dimension. Results show that a randomly
initialized token leads to an 8.8% lower F1 score
compared to ours. This is because random initial-
ization without semantic grounding makes it diffi-
cult for the model to interpret the token. In con-
trast, we sample the mask token embedding using
the mean and variance computed from all vocab-
ulary embeddings, ensuring that the token’s distri-
bution aligns with that of the pre-trained embed-
ding space, thereby achieving more stable training
and better performance. (3) We include the mask
token in the training loop, but observe a significant
performance drop. This may be due to the con-
stantly changing mask token disrupting the docu-
ment representation.5
5.3.4 Inference Latency
Table 5 reports the average inference latency of
different methods under the same hardware condi-
tions. All experiments are performed on RTX PRO
6000, with an Intel(R) Xeon(R) Gold 5318Y CPU.
The results show that our method achieves the
5Document-specific mask tokens are not feasible under
the current framework, as each document contains no more
than 15 QA pairs, which is insufficient to effectively train
such tokens.lowest latency with small models, except for RAG.
This is because our approach requires only a single
round of parameter generation, whereas DyPRAG
involves three rounds of parameter generation and
aggregation, and PRAG incurs additional over-
head from repeatedly loading offline document pa-
rameters, leading to reduced inference efficiency.
However, as model size increases, the latency of
our method becomes more pronounced due to the
involvement of special tokens, equally in length
to the original documents. It is worth noting that
our inference procedure is identical to that of stan-
dard RAG, and our primary objective is to attain
RAG-level inference performance without plain-
text documents, rather than optimizing for infer-
ence speed.
5.3.5 Reconstruction Attack
Although our approach prevents plaintext expo-
sure of user documents during cloud-based infer-
ence, there remains a potential risk that an attacker
could attempt to reconstruct documents from gen-
erated LoRA weights. We consider a worst-case
scenario in which an attacker obtains a set of docu-
ment–LoRA weight pairs. To simulate such an at-
tack, we use 30,000 documents, selecting 100 as a
test set and the rest for training a T5 decoder6(Guo
et al., 2022) to reconstruct documents from LoRA
parameters. We use ROUGE-2 recall to measure
reconstruction quality. As shown in Figure 3, even
after extensive training (over 10 epochs), the re-
construction recall does not exceed 9%.
Figure 4 presents an example of document re-
construction. This represents one of the best re-
construction cases observed. In most instances,
the reconstructed content consists primarily of
repetitive or meaningless phrases such as "Count
of the Count of the. . . ". Even in this rela-
6https://huggingface.co/google/long-t5-tglobal-base

Figure 3: Reconstruction attack results of Rouge-
2 recall values for documents reconstructed at
multiple checkpoints.
Come and Find Me is a 2016 American drama film directed 
and written by Zack Whedon .
The film stars Aaron Paul, Annabelle Wallis, Enver Gjokaj 
and Garret Dillahunt .
The film was released in a limited release and through 
video on demand on November 11, 2016 , by Saban Films.Original Document
The A. as : The film is a 2012 American comedy film
written and directed by Roberts. The film stars generosity , 
and starring Johngre , and Johngre . The film was released on 
July 26, $200 .Reconstructed Document
Figure 4: An example of a document reconstruc-
tion. Red indicates successful reconstruction, and
blue indicates incorrect reconstruction.
2WQA HQA PQA CWQ
DyPRAG 19.9 17.1 17.5 25.4
Ours 19.8 17.5 17.8 16.6
Table 6: The overlap (%) between documents
used in training and those retrieved from test set.
tively successful example, key information such
as dates, names, and titles is completely incor-
rect. This demonstrates that our parameter gen-
erator is trained to encode abstract knowledge,
rather than to directly reconstruct the original
text. As a result, recovering the original docu-
ment content—especially precise private informa-
tion—from LoRA weights is extremely difficult.
This further confirms the reliability of our method.
5.3.6 Overlap Between Train and Test
To further validate the generalization of Distilled-
PRAG and eliminate the risk of test set leak-age, we estimated the maximum Jaccard similar-
ity (Jaccard, 1901) between each test document
and all training documents using an efficient Min-
Hash and Locality-Sensitive Hashing(LSH) ap-
proach. For each test document, we recorded the
highest similarity with any training document and
averaged these scores. As shown in Table 6, the
average maximum similarity across four datasets
remains below 20%. This demonstrates limited
textual overlap and confirms the robustness and
authenticity of our method’s generalization.
6 Conclusion
In this work, we present DistilledPRAG, a novel
parametric RAG model that addresses the gener-
alization and efficiency limitations of existing ap-
proaches such as PRAG and DyPRAG. By align-
ing both document structure and parameter acti-
vations with a standard RAG teacher model, Dis-
tilledPRAG leverages knowledge distillation to
achieve robust RAG-style reasoning without ac-
cess to plaintext documents. Experimental results
on four public QA datasets demonstrate that Dis-
tilledPRAG significantly outperforms strong base-
lines in OOD settings, even when trained on a sin-
gle dataset. In future work, we plan to extend Dis-
tilledPRAG to support multi-modal inputs and ex-
plore its applicability in settings with more com-
plex reasoning and open-domain generation tasks.
7 Limitations
Despite its promising performance, Distilled-
PRAG remains an approximation of standard
RAG. While our internal alignment strategy im-
proves reasoning consistency, the parametric na-
ture of LoRA representations may still fall short
in capturing nuanced cross-document interactions,
especially in complex multi-hop scenarios. Fur-
thermore, the privacy-preserving design by avoid-
ing transmission of plaintext documents inevitably
introduces a privacy-utility tradeoff. Encod-
ing documents into LoRA may lose fine-grained
contextual details, limiting answer accuracy on
knowledge-intensive queries. Future work may
explore richer alignment objectives and more ex-
pressive methods to further close the gap between
parametric and standard RAG performance.

References
Markus J Buehler. 2024. Generative retrieval-
augmented ontologic graph and multiagent
strategies for interpretive large language model-
based materials design. ACS Engineering Au ,
4(2):241–277.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge,
Si-Qing Chen, Furu Wei, Huishuai Zhang, and
Dongyan Zhao. 2024. xRAG: Extreme con-
text compression for retrieval-augmented gen-
eration with one token. In The Thirty-eighth
Annual Conference on Neural Information Pro-
cessing Systems .
Mandy Guo, Joshua Ainslie, David C Uthus, San-
tiago Ontanon, Jianmo Ni, Yun-Hsuan Sung,
and Yinfei Yang. 2022. Longt5: Efficient text-
to-text transformer for long sequences. In Find-
ings of the Association for Computational Lin-
guistics: NAACL 2022 , pages 724–736.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Constructing
a multi-hop qa dataset for comprehensive eval-
uation of reasoning steps. In Proceedings of
the 28th International Conference on Compu-
tational Linguistics , pages 6609–6625.
Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu,
Yuanzhi Li, Shean Wang, Lu Wang, Weizhu
Chen, et al. 2022. Lora: Low-rank adaptation
of large language models. In International Con-
ference on Learning Representations .
Paul Jaccard. 1901. Etude de la distribution florale
dans une portion des alpes et du jura. Bulletin
de la Societe Vaudoise des Sciences Naturelles ,
37:547–579.
Solomon Kullback and Richard A Leibler. 1951.
On information and sufficiency. The annals of
mathematical statistics , 22(1):79–86.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan,
et al. 2024. Deepseek-v3 technical report.
arXiv preprint arXiv:2412.19437 .
Ilya Loshchilov and Frank Hutter. 2019. Decou-
pled weight decay regularization. In Interna-
tional Conference on Learning Representations .Antoine Louis, Gijs van Dijck, and Gerasimos
Spanakis. 2024. Interpretable long-form legal
question answering with retrieval-augmented
large language models. In Proceedings of the
AAAI Conference on Artificial Intelligence , vol-
ume 38, pages 22266–22275.
Maxime Louis, Hervé Déjean, and Stéphane Clin-
chant. 2025. Pisco: Pretty simple compres-
sion for retrieval-augmented generation. arXiv
preprint arXiv:2501.16075 .
Alex Mallen, Akari Asai, Victor Zhong, Ra-
jarshi Das, Daniel Khashabi, and Hannaneh Ha-
jishirzi. 2023. When not to trust language mod-
els: Investigating effectiveness of parametric
and non-parametric memories. In Proceedings
of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long
Papers) , pages 9802–9822.
David Rau, Shuai Wang, Hervé Déjean, Stéphane
Clinchant, and Jaap Kamps. 2025. Context
embeddings for efficient answer generation in
retrieval-augmented generation. In Proceed-
ings of the Eighteenth ACM International Con-
ference on Web Search and Data Mining , pages
493–502.
Stephen Robertson, Hugo Zaragoza, et al. 2009.
The probabilistic relevance framework: Bm25
and beyond. Foundations and Trends® in In-
formation Retrieval , 3(4):333–389.
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan,
Changyue Wang, Hongning Wang, Ziyi Ye, Yu-
jia Zhou, and Yiqun Liu. 2025. Parametric re-
trieval augmented generation. In Proceedings
of the 48th International ACM SIGIR Confer-
ence on Research and Development in Informa-
tion Retrieval , pages 1240–1250.
Alon Talmor and Jonathan Berant. 2018. The web
as a knowledge-base for answering complex
questions. In Proceedings of the 2018 Confer-
ence of the North American Chapter of the As-
sociation for Computational Linguistics: Hu-
man Language Technologies, Volume 1 (Long
Papers) , pages 641–651.
Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun
Zhao, and Kang Liu. 2025. Dynamic para-
metric retrieval augmented generation for test-
time knowledge enhancement. arXiv preprint
arXiv:2503.23895 .

Yujing Wang, Hainan Zhang, Liang Pang,
Yongxin Tong, Binghui Guo, Hongwei Zheng,
and Zhiming Zheng. 2025. Learning to
erase private knowledge from multi-documents
for retrieval-augmented large language models.
arXiv preprint arXiv:2504.09910 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. 2018. Hotpotqa:
A dataset for diverse, explainable multi-hop
question answering. In Proceedings of the 2018
Conference on Empirical Methods in Natural
Language Processing , pages 2369–2380.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yid-
ing Liu, Yue Xing, Han Xu, Jie Ren, Yi Chang,
Shuaiqiang Wang, Dawei Yin, et al. 2024. The
good and the bad: Exploring privacy issues in
retrieval-augmented generation (rag). In Find-
ings of the Association for Computational Lin-
guistics ACL 2024 , pages 4505–4524.
Boyu Zhang, Hongyang Yang, Tianyu Zhou,
Muhammad Ali Babar, and Xiao-Yang Liu.
2023. Enhancing financial sentiment analysis
via retrieval augmented large language models.
InProceedings of the fourth ACM international
conference on AI in finance , pages 349–356.
Jia-Ying Zheng, Hainan Zhang, Lingxiang Wang,
Wangjie Qiu, Hong-Wei Zheng, and Zhi-Ming
Zheng. 2024. Safely learning with private data:
A federated learning framework for large lan-
guage model. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Lan-
guage Processing , pages 5293–5306.
A Prompts Used in Our Experiments
Figures 5, 6, and 7 show the prompts we use for
single-document QA synthesis, cross-document
QA synthesis, and inference.
I will provide a passage of text, and you need to generate two to five 
different questions based on the content of this passage. Each question 
should be answerable using the information provided in the passage. 
Additionally, please provide an appropriate answer for each question derived 
from the passage .
You need to generate the question and answer in the following format :
[
{
"question": " What is the capital of France ?",
"answer": "Paris"
"full_ answer ": "The capital of France is Paris ."
}, 
]
This list should have at least two elements. As long as the passage 
information is sufficient, you should try to generate five elements as much as 
possible. You only need to output this list in the above format .
Passage :
{passage}Prompt for Single -Document QA SynthesisFigure 5: Prompt for single-document QA syn-
thesis.
I will provide multiple passages of text. Your task is to generate five different questions that can 
only be answered by synthesizing information from multiple passages. Each question must require 
combining information across all passages. Do not create questions that can be answered using 
only a single passage .
To help you think more broadly, here are some types of questions you can consider :
-**Comparison questions** (e.g., comparing entities or events mentioned in different passages )
-**Cause -and-effect questions** (e.g., identifying how one event described in one passage 
influences or results in another event in a different passage )
-**Temporal reasoning questions** (e.g., constructing a timeline or understanding 
chronological dependencies )
-**Entity synthesis questions** (e.g., asking about a person/organization/concept that appears 
in multiple passages with different attributes )
-**Theme or topic integration** (e.g., identifying common themes, contrasting perspectives, or 
overarching insights across documents )
Each question should be answerable using the combined information provided in the set of 
passages. For each question, provide both a short answer and a full sentence answer .
You must follow this format strictly :
[
{
"question": " What is the capital of France ?",
"answer": "Paris",
"full_ answer ": "The capital of France is Paris ."
},
... (at least five in total )
]
Only output the list in this exact format .
Passages:
{passage }Prompt for Cross -Document QA Synth esis
Figure 6: Prompt for single-document QA syn-
thesis.
user_ template :
You should answer the question by referring to the knowledge provided 
below and integrating your own knowledge. You must answer in a concise 
manner without any additional explanation .
Passages:
{passage }
Question : {question }
assistant_ template :
The answer isPrompt for Inference
Figure 7: Prompt for inference.
# ECoRAG: Evidentiality-guided Compression for Long Context RAG

**Authors**: Yeonseok Jeong, Jinsu Kim, Dohyeon Lee, Seung-won Hwang

**Published**: 2025-06-05 15:43:49

**PDF URL**: [http://arxiv.org/pdf/2506.05167v1](http://arxiv.org/pdf/2506.05167v1)

## Abstract
Large Language Models (LLMs) have shown remarkable performance in Open-Domain
Question Answering (ODQA) by leveraging external documents through
Retrieval-Augmented Generation (RAG). To reduce RAG overhead, from longer
context, context compression is necessary. However, prior compression methods
do not focus on filtering out non-evidential information, which limit the
performance in LLM-based RAG. We thus propose Evidentiality-guided RAG, or
\textbf{ECoRAG} framework. ECoRAG improves LLM performance by compressing
retrieved documents based on evidentiality, ensuring whether answer generation
is supported by the correct evidence. As an additional step, ECoRAG reflects
whether the compressed content provides sufficient evidence, and if not,
retrieves more until sufficient. Experiments show that ECoRAG improves LLM
performance on ODQA tasks, outperforming existing compression methods.
Furthermore, ECoRAG is highly cost-efficient, as it not only reduces latency
but also minimizes token usage by retaining only the necessary information to
generate the correct answer. Code is available at
https://github.com/ldilab/ECoRAG.

## Full Text


<!-- PDF content starts -->

arXiv:2506.05167v1  [cs.CL]  5 Jun 2025ECoRAG: Evidentiality-guided Compression for Long Context RAG
Yeonseok Jeong1, Jinsu Kim2, Dohyeon Lee3, Seung-won Hwang3*
IPAI, Seoul National University1, Korea University2, Seoul National University3
{jys3136, waylight3, seungwonh}@snu.ac.kr
tonmmy222@korea.ac.kr
Abstract
Large Language Models (LLMs) have shown
remarkable performance in Open-Domain
Question Answering (ODQA) by leverag-
ing external documents through Retrieval-
Augmented Generation (RAG). To reduce RAG
overhead, from longer context, context com-
pression is necessary. However, prior com-
pression methods do not focus on filtering out
non-evidential information, which limit the
performance in LLM-based RAG. We thus
propose Evidentiality-guided RAG, or EC-
oRAG framework. ECoRAG improves LLM
performance by compressing retrieved docu-
ments based on evidentiality, ensuring whether
answer generation is supported by the cor-
rect evidence. As an additional step, EC-
oRAG reflects whether the compressed con-
tent provides sufficient evidence, and if not,
retrieves more until sufficient. Experiments
show that ECoRAG improves LLM perfor-
mance on ODQA tasks, outperforming existing
compression methods. Furthermore, ECoRAG
is highly cost-efficient, as it not only reduces
latency but also minimizes token usage by re-
taining only the necessary information to gen-
erate the correct answer. Code is available at
https://github.com/ldilab/ECoRAG.
1 Introduction
LLMs (OpenAI, 2023; Touvron et al., 2023) have
excelled in tasks such as ODQA by leveraging ex-
ternal knowledge through RAG (Lewis et al., 2020;
Ram et al., 2023). However, RAG inevitably in-
creases context length, which incurs higher com-
putational cost and also hinders generation qual-
ity (Liu et al., 2024; Hsieh et al., 2024; Li et al.,
2024).
While adopting existing context compression (Li
et al., 2023) may look promising, such a baseline
presents two main challenges. First, LLMs are
known to be vulnerable to irrelevant contents that
*Corresponding Author
36.037.038.039.040.041.042.043.044.0
3 6 9 12Exact Match (EM)
the number of documents
Raw Prepend RECOMP ECoRAG (ours)
41.042.043.044.045.0Figure 1: Comparison of performance between prepend-
ing retrieved documents (standard RAG) (Karpukhin
et al., 2020), applying RECOMP (Xu et al., 2024),
and applying ECoRAG on the Natural Questions
(Kwiatkowski et al., 2019) test set. Experiments were
conducted using Flan-UL2 (Tay et al., 2023).
cannot provide evidence for answer generation (Shi
et al., 2023; Qian et al., 2024; Wu et al., 2024), and
existing compression methods (Xu et al., 2024;
Jiang et al., 2024; Yoon et al., 2024) do not effec-
tively filter them out. As a result, a naive baseline
simply prepending retrieved documents, ‘standard
RAG’ in Figure 1, outperforms a baseline compres-
sor RECOMP (Xu et al., 2024). As the number of
documents increases, a baseline compressor fails
to filter out increasing irrelevant contents, causing
performance to decline.
Second, it is challenging to determine the desir-
able compression ratio for each question. Failure
to do so may lead to compressing too much, which
results in losing crucial information, or compress-
ing too little, which produces overly long contexts
that degrade generation quality (Liu et al., 2024;
1

Hsieh et al., 2024; Li et al., 2024) and increase com-
putational costs. Thus, it is necessary to find the
desirable compression ratio that enables the LLM
to generate the correct answer for each question.
Our distinction is using evidentiality to ad-
dress both challenges and proposing Evidentiality-
guided Compression and Retrieval- Augmented
Generation ( ECoRAG ) framework: Ours com-
presses retrieved documents to retain only the in-
formation necessary to support the answer. To
overcome the first challenge, evidentiality (Lee
et al., 2021; Asai et al., 2022) is used to determine
whether each sentence in the retrieved documents
supports the correct answer to a question. It can
be quantified for each sentence by measuring how
much it contributes to the model to generate the
correct answer. We train the compressor using this
as training signals.
To address the second challenge, ECoRAG re-
flects on compression as a collective, where it con-
tains sufficient evidence. We begin by forming the
smallest possible collective unit of compression
and assess whether it is evidential. If not, it means
that it is compressed too much, which we adjust
adaptively by collecting more, until it is sufficient.
Through this reflection process, ECoRAG finds the
desirable compression ratio that enables the LLM
to generate the correct answer with minimal tokens.
By applying these methods, ECoRAG has two
advantages when dealing with long contexts as
the number of documents increases. First, EC-
oRAG improves performance by retaining only the
information necessary for generating the correct
answer and removing distracting content. This re-
sults in gains on ODQA datasets such as Natural
Questions (NQ) (Kwiatkowski et al., 2019), Triv-
iaQA (TQA) (Joshi et al., 2017), WebQuestions
(WQ) (Berant et al., 2013). Second, by compress-
ing the long context to only what is needed, it re-
duces computational costs.
Our contributions to this work can be summa-
rized as follows: (1) Evidentiality-guided Com-
pression: We developed a method that compresses
retrieved documents based on evidentiality. (2) Ev-
identiality Reflection for Adaptive Compression:
Our framework evaluates compressed content for
evidentiality and adaptively adjusts the length of
compression. (3) Experiments show that our ap-
proach significantly improves retrieval-augmented
LLM performance on ODQA datasets. (4) Our
approach is also cost-efficient, as it quickly com-
presses long context, reducing latency and tokens.2 Related Work
2.1 Evidentiality-guided RAG
Dense retrievers (Karpukhin et al., 2020; Izacard
et al., 2022) focus on lexical answerability, but may
mislabel documents as relevant when they lack con-
textual evidence, leading to the need for evidential-
ity. In prior work (Lee et al., 2021), evidentiality
refers to whether a document supports generating
the correct answer to a question. Unlike answer-
ability, evidentiality is more challenging to mine
directly as it reflects the contextual relationship
between a question and a document. To measure
evidentiality, previous work checks whether the
removal of the document is critical for answering
the question (Asai et al., 2022), utilizes attention
scores (Niu et al., 2020), or considers the change
in confidence scores (Song et al., 2024). Our work
introduces evidentiality in LLMs, enhancing RAG
by prioritizing contextually rich documents for gen-
erating correct answers.
2.2 Prompt Compression
Numerous studies (Mu et al., 2024; Li et al., 2023;
Kim et al., 2024) have focused on prompt compres-
sion to address both cost and performance chal-
lenges, as shown in prior research (Shi et al., 2023;
Liu et al., 2024; Hsieh et al., 2024). RECOMP (Xu
et al., 2024) provides both extractive and generative
summaries of documents, considering whether the
summaries helped answer the given question. LLM-
Lingua (Jiang et al., 2023b) uses conditional proba-
bilities of LLMs to guide fine-grained prompt com-
pression. Building on this, LongLLMLingua (Jiang
et al., 2024) compresses prompts in long context
scenarios by using a question-aware coarse-to-fine
compression and document reordering mechanism.
Similarly, CompAct (Yoon et al., 2024) employs an
adaptive compression strategy to iteratively com-
press documents while retaining key information
relevant to the query. However, existing methods
struggle to compress long context, which prevents
them from fully utilizing the retrieval results.
2.3 Retrieval Evaluation for RAG
LLMs may evaluate the quality of retrieved re-
sults for enhancing RAG, as seen in Madaan et al.
(2024), where models iteratively improve their re-
sponses; this concept has been applied to RAG.
Self-RAG (Asai et al., 2024) trains LLM to evalu-
ate retrieved documents and its output by predicting
reflection tokens that assess the need for retrieval
2

and the quality of the generated text. Labruna et al.
(2024) dynamically determines whether to retrieve
additional context when needed by using a trained
reader LLM. CRAG (Yan et al., 2024) employs a re-
trieval evaluator to assess document relevance and
triggers corrective actions to refine retrieved infor-
mation, by using lexical overlap between questions
and documents. In our ECoRAG framework, we
evaluate whether the evidence is sufficient to gen-
erate the correct answer by leveraging evidentiality
as defined by the LLM.
3 Proposed Method
In this section, we describe how ECoRAG adap-
tively adjusts the compression length to ensure
that the LLM generates the correct answer. To
achieve this, we focus on: (1) compressing re-
trieved documents by sorting them based on evi-
dentiality (Section 3.1), and (2) evaluating whether
the compressed documents is sufficiently eviden-
tial, and if not, adaptively incorporating more in-
formation (Section 3.2), and Figure 2 provides an
overview.
3.1 Evidentiality-guided Compressor
This section explains how retrieved documents are
compressed while preserving the evidence that en-
ables the LLM to generate the correct answer. We
decompose documents into sentences inspired by
Xu et al. (2024) and compress them guided by ev-
identiality. To retain the necessary content and
remove irrelevant parts during the compression pro-
cess, we first extract evidential sentences from the
retrieved documents (Section 3.1.1) and then use
them to train the compressor (Section 3.1.2).
3.1.1 Definition of Evidentiality
We define the evidentiality of a sentence based on
its contribution to generating the correct answer
while penalizing distractors that interfere with this
process. The degree of evidentiality is categorized
hierarchically based on two conditions. We find
sentences that enable the LLM to generate the cor-
rect answer. If a sentence does not, we then check
if it interferes with other evidence.
First, when assessing whether each sentence
helps generate the correct answer, it is important to
consider that the LLM contains parametric knowl-
edge (Wang et al., 2020; Yu et al., 2023; Luo et al.,
2023). Prior work (Lee et al., 2021; Asai et al.,
2022) has focused on whether the language model
could contribute to generating the correct answerusing given document. However, it is challeng-
ing to distinguish whether the correct answer was
generated using the document or parametric knowl-
edge, especially in larger models. If the correct an-
swer was generated solely using parametric knowl-
edge, regardless of the given document, it is unclear
to determine whether the document serves as key
evidence. Therefore, we propose the following first
condition: 1Without the sentence the LLM can-
not generate the correct answer alone, but with the
sentence it can.
Second, it is also crucial for the compressor to
filter out distractors that hinder the evidence from
generating the correct answer. While robustness
to distractors can be improved through fine-tuning
(Liu et al., 2024), training LLMs often requires sub-
stantial costs for training and closed LLMs often
impossible to train. If the compressor can remove
distractors, it can be applied to any LLM without re-
quiring additional training. To identify distractors,
we introduce a second condition for sentences that
do not satisfy 1:2The sentence does not inter-
fere with the evidence defined in 1in generating
the correct answer.
Based on the aforementioned conditions, we hi-
erarchically define evidentiality as depicted in Fig-
ure 3. Sentences satisfying condition 1are labeled
asstrong evidence . Sentences failing to meet con-
dition 1are further classified based on condition
2: those satisfying condition 2are labeled as
weak evidence , while those that do not are classi-
fied as distractor . Following these conditions, we
use an LLM to label sentences in retrieved docu-
ments for each question in the training data.
3.1.2 Learning Objective for Compressor
Given labeled sentences D={d1, d2,···, d|D|},
for a question q, we train our compressor based
on dual encoders (Izacard et al., 2022) to differen-
tiate between strong and weak evidence, as well
as distractor. Using dual encoders, EQfor ques-
tions and EDfor sentences, we calculate the sim-
ilarity score between qand sentences in D(i.e.,
sim(q, di) =EQ(q)·ED(di)). Sentences are cate-
gorized into strong ( d∗) or weak ( d+) evidence, and
distractor ( d−) based on our hierarchical definition.
We define similarity scores as s∗=sim(q, d∗),
s+=sim(q, d+), ands−=sim(q, d−). The sim-
ilarity scores are utilized to train two inequalities:
(s+> s−),(s∗> s+, s−) (1)
3

⋮𝑑𝑖
𝑑𝑖+1
𝑑|𝐷|
Evidentiality -
guided
Compressor
decompose
𝑞: Who got the first nobel prize?
Evidentiality
Evaluator
𝑑2′𝑑1′
𝑑|𝐷|′⋮retrieval
ECoRAG
𝐶Final 
compressionFinal compression
LLM
yes
sort sentences
by evidentiality⋮
𝑑𝑘′Collective
𝐶 𝑑𝑘+1′Retrieved
documents
(n = 100)
⋮𝑑2′𝑑1′
𝑑|𝐷|′⋮⋮
𝑑𝑘′
𝑑𝑘+1′
no collect more evidence ( 𝑑𝑘+1′)Evidentiality Reflection Evidentiality -guided Compression
Is 𝐶 
evidential
to 𝑞?
[End]Figure 2: This figure illustrates the overall framework of ECoRAG. First, the evidentiality-guided compressor
compresses the retrieved documents by sorting decomposed sentences based on evidentiality, producing an ordered
set of evidences d′
1, d′
2, . . . , d′
|D|. Second, evidentiality reflection starts with the top-ranked sentence ( n= 1, i.e.,
C=d′
1), and the evidentiality evaluator determines whether Cis evidential. If not, more evidence is added
iteratively ( n=k→n=k+ 1) until the evaluator judges Cas evidential. Once evidential, it is used for final
compression (green line); otherwise, additional evidence is collected (red line).
These inequalities ensure that strong evidence is
ranked above weak evidence, which in turn is
ranked above distractor, guiding the training of
our compressor.
The weak evidentiality loss Lweuses the In-
foNCE loss to distinguish weak evidence d+from
distractor d−. The loss function is formulated as:
Lwe=−logexp(s+/τ)
exp(s+/τ) +P
d−
j∈D−exp(s−
j/τ)
(2)
Here, s−
j=sim(q, d−
j)represents the similarity
score for each distractor in the set D−, and τis a
temperature parameter.
The strong evidentiality loss Lsealso utilizes the
InfoNCE loss to prioritize strong evidence d∗. The
loss function is formulated as:
Lse=−logexp(s∗/τ)
exp(s∗/τ) +P
d±
j∈D−∪D+exp(s±
j/τ)
(3)
Here, s±
j=sim(q, d−
j)is the similarity score for
each sentence in the combined sets of distractors
D−and weak evidences D+.
The final loss Lis defined as the sum of the
strong and weak evidentiality losses:
L=Lse+Lwe (4)
Our compressor is trained using this loss L, and
ranks sentences d′
1, d′
2, . . . , d′
|D|by evidentiality,selecting high-scoring ones for compression. The
number of sorted evidence required can vary de-
pending on the difficulty of each question. How-
ever, providing too little evidence may omit impor-
tant information, while too much increases com-
putational costs for each question. Thus, balanced
compression ratio is necessary for each question to
address both issues.
3.2 Evidentiality Reflection for Adaptive
Compression
Once a collective of evidential sentences is formed,
we need to determine whether the compression ra-
tio is appropriate. To achieve this, we reflect on the
evidentiality of compressed documents using a lan-
guage model (Section 3.2.1). Then, if compressed
too much, we adaptively adjust the compression
ratio by collecting more (Section 3.2.2).
3.2.1 Training Evidentiality Evaluator
We develop an effective evidentiality evaluator
Mevalthat assesses whether the compressed docu-
ments are strong evidence enough to generate the
correct answer. In prior work, CompAct (Yoon
et al., 2024) trained the evaluator by prompting
GPT-4o (OpenAI, 2023) to determine if the evi-
dence is sufficient to answer the question. However,
this approach can introduce bias (Chiang and Lee,
2023) when GPT-4o evaluates through prompting,
leading to inaccurate supervision. Accurate super-
vision requires verifying if the document actually
enables the reader LLM to generate the correct
4

𝑞: When is the next deadpool  movie being released?
𝑎: May 18, 2018
𝑑1: “Deadpool 2” was released on May 18, 2018. 
𝑑2: “Deadpool 2” is the next movie of Deadpool.
𝑞
𝑞
𝑞
𝑞𝑑12017
May 18, 2018
2017
𝑑1strong evidence𝑑3: Spider -Man and Deadpool often team up in Marvel.
𝑑2
𝑑3 2017
𝑑2weak evidence 𝑑3distractor𝑞
𝑞 𝑑3𝑑2 May 18, 2018 𝑑1
𝑑1 2017
LLM
𝑑2,𝑑3sent to (c)
(b) strong evidentiality mining (a) Example of question, answer, 
and sentences for evidentiality mining
LLM
(c) weak evidentiality mining Figure 3: This figure illustrates the evidentiality mining
strategy of ECoRAG.
answer. To achieve this, we reuse our evidential-
ity labels obtained from the LLM in Section 3.1.1
and distill them from our reader LLM into smaller
model, Flan-T5-large (Chung et al., 2022), to build
the evaluator. Comparison between CompAct and
our evaluator is discussed in Section 5.2.
We train Mevalusing our evidentiality labeled
dataset (d∗, d+, d−)to determine if compressed
documents are sufficient for correct answer gener-
ation. The evaluator is trained to classify whether
the given compressed documents is strong evi-
dence. To facilitate this, we add 2 special tokens
t∈[<EVI> ,<NOT> ]and train Mevalto gener-
ate ‘<EVI>’ for strong evidence d∗, and ‘<NOT>’
for other sentences d+, d−. Subsequently, next-
token prediction loss Levalis used for this training
stage to predict whether compressed documents are
strong evidence.
Leval=−log pMeval(t|q, d) (5)
3.2.2 Adaptive Compression
In adaptive compression, the compression ratio is
adaptively adjusted by our evaluator, which reflects
on whether the current compression is evidential,as described in Figure 2. Initially, our evaluator as-
sesses the evidentiality of compressed documents
Ccontaining only the first evidence, d′
1, from our
ordered evidences d′
1, d′
2, . . . , d′
|D|. If the evalua-
tor determines that Cis evidential, it becomes the
final compression provided to LLM. If Cis not
evidential, we add the next piece of evidence d′
2
tod′
1to build new compressed documents. If the
k-th iteration fails, d′
k+1is added to the previously
compressed documents. This process is repeated
until the desirable compression is found, with a to-
ken limit set to avoid infinite loop. Since retrieved
documents do not always include gold evidence for
all queries, a token limit is necessary to prevent
infinite loops from continuously adding evidence.
The final compression is then used as input for the
LLM, which generates the final answer.
Although iterative adjustment can increase la-
tency compared to using raw documents, ECoRAG
reduces it efficiently. Prior work (Yoon et al., 2024),
each iteration required LLM (7B) to generate a new
compression by using the previous compression
and the next piece of evidence. Thus, with each
iteration, LLM reads different contents and gen-
erates compression of multiple tokens, increasing
latency time. However, ECoRAG reduces redun-
dancy by ordering evidence just once and adding
it iteratively. Moreover, our framework utilized
a lightweight evaluator (0.77B) that adjusts com-
pression length by generating just a single special
token, resulting in rapid compression speed; the
actual results are shown in Section 5.4.
4 Experiments
4.1 Experimental Settings
Datasets We evaluate our framework through
NQ (Kwiatkowski et al., 2019), TQA (Joshi et al.,
2017), and WQ (Berant et al., 2013), which are
ODQA datasets. We use the 100 documents re-
trieved from DPR (Karpukhin et al., 2020)1.
Models We initialize our evidentiality compres-
sor from Contriever (Izacard et al., 2022) and use
it to compare its performance with RECOMP (Xu
et al., 2024). For evidentiality evaluator, we utilize
Flan-T5-large (Chung et al., 2022), because pre-
vious RAG and document-assessment work (Han
et al., 2023; Yan et al., 2024) have successfully em-
ployed it. Detailed justification for this choice can
1Since enhancing the retriever is beyond the scope of this
study, we conduct our experiments under the assumption that
the retrieved documents are already provided.
5

MethodsNQ TQA WQ
#tokens ↓ EM F1 #tokens ↓ EM F1 #tokens ↓ EM F1
RAG without compression
closed-book 0 31.88 44.10 0 64.78 73.10 0 24.51 42.73
standard RAG (100 documents) 13905 36.09 50.18 14167 56.21 64.22 13731 21.11 38.72
RAG with 100 documents compressed
LLMLingua (Jiang et al., 2023b) 635 26.84 38.30 630 50.81 57.91 641 22.98 39.77
LLMLingua-2 (Pan et al., 2024) 1315 30.11 42.52 1324 53.19 60.46 1113 23.52 40.61
LongLLMLingua (Jiang et al., 2024) 1370 32.96 45.32 1402 55.75 63.75 1355 21.51 39.13
RECOMP (extractive) (Xu et al., 2024) 662 32.85 44.54 672 51.66 59.08 658 19.54 36.83
RECOMP (abstractive) (Xu et al., 2024) 14 27.59 39.19 26 39.95 46.68 19 20.47 36.90
CompAct (Yoon et al., 2024) 106 35.71 47.14 96 63.96 73.87 75 29.77 44.25
ECoRAG (ours) 632 36.48 49.81 441 65.34 75.37 560 30.17 46.13
Table 1: Compression methods performance comparison on NQ, TQA, and WQ. The table shows the results using
GPT-4o-mini as the reader model, given 100 retrieved documents (Karpukhin et al., 2020). It reports the number of
tokens after compression, along with EM and F1-score, illustrating the impact of different compression methods on
model performance.
be found in Section B.2. For the reader model, we
use GPT-4o-mini (OpenAI, 2023), as it supports a
context length of 128K tokens, sufficient to process
all 100 retrieved documents.
Evaluation Metrics We report results on the test
sets of NQ, TQA, and WQ using EM and word-
level F1-score to assess the question-answering
task performance. We also report the average num-
ber of input tokens given to the reader LLM to
evaluate the efficiency of our compression step.
Baseline We report two types of baselines.
RAG without compression : As a baseline, we
report the results using only the question and raw
retrieved documents. The ‘closed-book’ setting,
where no retrieval is used, shows that the model
relies solely on its internal knowledge. In the ‘stan-
dard RAG’ setting, we simply concatenate the top
100 retrieved documents without any compression
for evaluation.2This is the approach used in con-
ventional RAG without compression.
RAG with 100 compressed documents : We
also reproduce several retrieval augmentation
methods for comparison. To better understand
the effect of different compression methods, we
evaluated several baselines including LLMLin-
gua (Jiang et al., 2023b), LLMLingua-2 (Pan et al.,
2024), LongLLMLingua (Jiang et al., 2024), Com-
pAct (Yoon et al., 2024), and RECOMP which
offers both extractive and abstractive variants.
In addition to our compression and non-
compression baselines, we include BGE-M3 (Chen
et al., 2024) and BGE-reranker (Xiao et al., 2024)
2We also evaluated the effect of reducing the number of
retrieved documents ( k= 5,10,20) for both DPR and Con-
triever in Table 16; results are explained in Section A.9.under equal token budgets. However, since these
are not compression methods, their comparison
results are addressed in Section A.4.
4.2 Results
In this section, we report the results of our model
and compare them with both compression-based
and non-compression baselines for ODQA in Table
1. Accuracy, such as EM and F1-score, is a more
important metric than token reduction for evaluat-
ing compression quality because simply reducing
tokens without preserving necessary information
is meaningless. A method is more efficient if it
reduces more tokens while maintaining higher ac-
curacy than another.
In terms of accuracy, ECoRAG outperforms all
baselines, including standard RAG, where the LLM
reads all retrieved information. In the long context
setting, retrieving many documents often brings in
those with low relevance scores, introducing noise.
However, previous compression methods fail to
filter out this noise, leading to performance degra-
dation compared to uncompressed approaches. No-
tably, ECoRAG surpasses all these methods, even
with fewer tokens than some of them. The strength
of ECoRAG lies in compressing only the necessary
content, focusing solely on the information essen-
tial for generating the correct answer. As a result,
ECoRAG outperforms the strongest compression
baseline in NQ (+0.77%p), TQA (+1.38%p), and
WQ (+0.40%p) in EM. As further detailed in Sec-
tion A.10, ECoRAG maintains this advantage even
on much longer retrieved documents, confirming
its robustness in another long context setting (Bai
et al., 2024).
From a token efficiency perspective, ECoRAG
6

Methods NDCG@1 NDCG@10
Answerability (baseline) 67.82 79.20
Leave-One-Out (Asai et al., 2022) 70.67 80.80
ECoRAG (ours) 75.53 81.92
Table 2: Comparison of NDCG@1 and NDCG@10 on
HotpotQA dataset using different training signals
uses more tokens than RECOMP (abstractive) and
CompAct but still outperforms them, while com-
pressing with fewer tokens than other methods.
According to Xu et al. (2024), abstractive RE-
COMP performs well in the 5-document setting
but struggles in long contexts due to input size limi-
tations. CompAct suffers from inaccurate compres-
sion evaluation, failing to retain essential informa-
tion, which lowers performance. In contrast, EC-
oRAG can handle long context and retain only the
necessary content to generate the correct answer,
which results in superior performance across differ-
ent datasets. Excluding the two compressors that
fail to preserve necessary information, ECoRAG
achieves higher accuracy with fewer tokens than
other methods, demonstrating its token efficiency.
5 Analysis
In addition to the main results, we verified the ef-
fectiveness of our framework by addressing the
following research questions:
•RQ1 : Does our compressor effectively cap-
ture human-annotated evidence?
•RQ2 : How accurately does our evaluator pre-
dict evidentiality?
•RQ3 : What is the impact of each component
in ECoRAG?
•RQ4 : Is ECoRAG efficient compression?
5.1 RQ1: Alignment with Human-annotated
Evidentiality
In this section, we assess whether our compres-
sor can effectively sort sentences by evidentiality
for next step. Although our compressor improves
LLM performance by learning LLM-defined ev-
identiality, it is essential to verify whether it ef-
fectively captures ground-truth evidence. Thus,
we conducted experiments using HotpotQA (Yang
et al., 2018), which provides human-annotated evi-
dence. We compared how well prior methods and
our compressor assign higher scores to ground-
truth evidence. For evaluation, we use NormalizedDiscounted Cumulative Gain (NDCG) as a metric
to evaluate how effectively evidentiality-focused
methods, including ours, rank evidence higher.
As shown in Table 2, ECoRAG achieved the
highest performance, demonstrating strong align-
ment with human-annotated evidentiality. The ‘An-
swerability’ baseline trains the compressor by treat-
ing passages containing the correct answer as pos-
itive and those without as negative. The ‘Leave-
One-Out’ (Asai et al., 2022) considers a passage
as positive if removing it prevents the model from
generating the correct answer, and negative if the
model still succeeds. ECoRAG outperforms prior
evidentiality baselines, achieving improvements in
NDCG@1 (+4.86%p) and NDCG@10 (+1.12%p)
This result indicates that our compressor effectively
captures evidence and aligns well with human anno-
tations. Thus, our compressor provides well-sorted
evidences to our evaluator, then we need to verify
the evaluator, the other component of ECoRAG.
5.2 RQ2: Evaluator Performance on
Evidentiality Prediction
30405060708090100
Accuracy Precision Recall F1TQA
CompAct Flan-UL2 Ours
Figure 4: Evidentiality evaluation metrics using differ-
ent evaluator, including ours, measured on the TQA.
We also need to verify the evidentiality evalua-
tor to accurately evaluate whether the compressed
documents enable the LLM to generate the correct
answer. To assess its accuracy, we conducted exper-
iments on the TQA test set. For each question, we
define ground-truth labels for retrieved documents
as either <EVI>, which lead to generating the cor-
rect answer as in Section 3.2.1, or <NOT>. We
then measured how well our evaluator and other
evaluators predicted these labels using accuracy,
precision, recall, and F1-score. The results are
shown in Figure 4.
Across all metrics, our evidentiality evalua-
tor effectively predicts evidentiality, even though
it has significantly fewer parameters than other
7

NQ TQA
EM R20 EM R20
(A) ECoRAG (ours) 36.48 75.18 65.43 80.38
Compressor
(B) w/o answerability 31.25 49.53 63.86 70.84
(C) w/o evidentiality 35.46 74.93 64.90 80.59
Adaptive Compression
(D) w/o evaluator 35.71 - 63.63 -
Table 3: Ablation study of ECoRAG, showing the im-
pact of compressor and adaptive compression methods.
evaluators. It outperforms the CompAct evalua-
tor (7B) (Yoon et al., 2024) by +13.96%p in F1
score. The CompAct evaluator is based on Mistral-
7B (Jiang et al., 2023a) and trained with supervi-
sion from GPT-4o. As Asai et al. (2024) noted,
the reader LLM evaluates whether documents sup-
port the correct answer, making it a strong baseline.
We used Flan-UL2 (Tay et al., 2023) (20B) as our
reader LLM, as described in Section B.3. Notably,
our evidentiality evaluator, despite its much smaller
size (770M), closely approximates the performance
of Flan-UL2 (-0.08p%).
5.3 RQ3: Ablation Study
In Table 3, we present the results of our ablation
study, assessing the impact of each component in
our framework by comparing EM across different
settings. We also report R20, checking if the gold
answer is in the top 20 sentences.
ForCompressor , we compare (A) ECoRAG with
two inferior compressors, (B) and (C). In (B), the
compressor uses a pretrained Contriever check-
point without additional training, while in (C), it is
trained with answerability labels. As shown, our
compressor trained with evidentiality labels out-
performs both alternatives. Comparing (A) and
(C) shows that evidentiality labels increase EM
(+1.02%p, +0.53%p) while maintaining R20 at a
comparable level. Since R20 measures lexical over-
lap, (C), trained with answerability, performs simi-
larly to or better than (A). The results demonstrate
the superiority of our evidentiality labels over an-
swerability labels, as they prioritize contextually
rich information.
ForEvaluator , we consider a no-evaluator set-
ting (D), where the initial compression from the
compressor is used without evaluating its eviden-
tiality. The EM gap between (A) and (D) (+0.77%p,
+1.80%p) highlights the impact of the evidential-
ity evaluator. These results highlight the impor-
tance of adaptively adjusting the amount of evi-MethodsCompression
TimeInference
TimeTotal
TimeThroughput
(example/sec)
closed-book - 3.79h 3.79h 0.26
standard RAG - 12.28h 12.28h 0.08
RECOMP 0.27h 4.08h 4.35h 0.23
CompAct 10.10h 4.83h 14.94h 0.07
ECoRAG (ours) 0.73h 4.23h 4.96h 0.20
Table 4: Inference time and compression time for NQ
test.
dence through evidentiality evaluation.
5.4 RQ4: Total Latency
ECoRAG is cost-efficient not only because it re-
duces the number of tokens but also because it de-
creases total latency in the RAG process. In RAG
without compression, computational costs increase
as more documents are retrieved. By applying com-
pression and retaining only the necessary informa-
tion, ECoRAG reduces total processing time.
Table 4 presents the total latency3, including
both compression and inference time, to show the
efficiency of our approach. For long context, the
LLM-based abstractive compressor CompAct took
longer than the ‘standard RAG’ setting, whereas the
extractive compressors RECOMP and ECoRAG
were faster. ECoRAG uses the lightweight eval-
uator that generates only a single token per itera-
tion, stopping the reflection process once the com-
pressed document is evidential or the token limit
is reached, thereby preventing excessive compres-
sion time. While ECoRAG had similar speed to
RECOMP, it achieved better performance by re-
taining only the information necessary to generate
the correct answer, as described in Table 1. Thus,
ECoRAG is effective in handling long contexts in
terms of both performance and efficiency.
ECoRAG is a two-step design that achieves both
speed and performance. Single-step aggregation
with LLMs, as demonstrated by CompAct in Ta-
ble 1, struggles with length dependency for list-
wise evaluation due to the “lost-in-the-middle” is-
sue (Liu et al., 2024). In contrast, ECoRAG sep-
arates the process by first assessing sentences in-
dividually with an extractive compressor and then
evaluating them collectively. This separation over-
comes challenges in handling long contexts and im-
proves compression effectiveness. Our lightweight
components ensure efficiency while achieving ef-
fective compression.
3Since GPT-4o-mini does not provide latency measure-
ments, we conducted the latency experiments using Flan-UL2.
8

6 Conclusion
ECoRAG is a framework designed to compress
long context by focusing on evidentiality in LLMs,
defined as whether information supports generating
the correct answer. Evidentiality-guided compres-
sion effectively filters out irrelevant content and
retains necessary evidence. Through adaptive com-
pression, ECoRAG determines the optimal com-
pression length for each question, ensuring efficient
use of context. As a result, ECoRAG demonstrates
both superior performance and efficiency in han-
dling long context, outperforming other compres-
sion methods.
7 Limitation
Evidentiality provides an effective indicator for
determining whether information is necessary for
an LLM to generate the correct answer. However,
mining evidentiality labels is computationally ex-
pensive, leading to increased costs. Since multiple
inferences are required for each question, it results
in significant time consumption. Nevertheless, as
more time is spent, more evidentiality labels can
be obtained, which can contribute to the training
of the compressor. Evidentiality labels can also be
reused to train the evidentiality evaluator, optimiz-
ing resource usage. Once the compressor is fully
trained and applied, the LLM inference process
becomes faster.
Building upon this efficiency improvement, the
application of this system can be extended beyond
ODQA to address broader real-world scenarios.
Extending it to tasks like summarization may be
necessary due to context length limits when pro-
cessing full content with LLMs. Selecting and
summarizing only the most important parts can
improve performance (Saxena and Keller, 2024;
Jeong et al., 2025), requiring evidentiality to be
redefined based on summarization metrics. Investi-
gating such adaptations is a potential direction for
future work.
Acknowledgements
This work was supported by the National Research
Foundation of Korea(NRF) grant funded by the Ko-
rea government(MSIT) (No. RS-2024-00414981),
Institute of Information & communications Tech-
nology Planning & Evaluation (IITP) grant funded
by the Korea government (MSIT) (No. 2022-0-
00077/RS-2022-II220077, AI Technology Devel-
opment for Commonsense Extraction, Reasoning,and Inference from Heterogeneous Data), and Insti-
tute of Information & communications Technology
Planning & Evaluation (IITP) grant funded by the
Korea government(MSIT) [NO.RS-2021-II211343,
Artificial Intelligence Graduate School Program
(Seoul National University)].
References
Akari Asai, Matt Gardner, and Hannaneh Ha-
jishirzi. 2022. Evidentiality-guided generation for
knowledge-intensive nlp tasks. In Proceedings of
the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies , pages 2226–2243.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations .
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, et al. 2024. Longbench:
A bilingual, multitask benchmark for long context
understanding. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 3119–3137.
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on freebase from
question-answer pairs. In Proceedings of the 2013
conference on empirical methods in natural language
processing , pages 1533–1544.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. In Findings of the Asso-
ciation for Computational Linguistics: ACL 2024 ,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Cheng-Han Chiang and Hung-yi Lee. 2023. Can large
language models be an alternative to human evalua-
tions? In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 15607–15631, Toronto,
Canada. Association for Computational Linguistics.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret
Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi
Wang, Mostafa Dehghani, Siddhartha Brahma, Al-
bert Webson, Shixiang Shane Gu, Zhuyun Dai,
Mirac Suzgun, Xinyun Chen, Aakanksha Chowdh-
ery, Alex Castro-Ros, Marie Pellat, Kevin Robinson,
Dasha Valter, Sharan Narang, Gaurav Mishra, Adams
Yu, Vincent Zhao, Yanping Huang, Andrew Dai,
Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Ja-
cob Devlin, Adam Roberts, Denny Zhou, Quoc V . Le,
and Jason Wei. 2022. Scaling instruction-finetuned
language models. Preprint , arXiv:2210.11416.
9

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 .
Sang-eun Han, Yeonseok Jeong, Seung-won Hwang,
and Kyungjae Lee. 2023. On monotonic aggrega-
tion for open-domain qa. In Proc. Interspeech 2023 ,
pages 3432–3436.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
tanu Acharya, Dima Rekesh, Fei Jia, and Boris Gins-
burg. 2024. RULER: What’s the real context size of
your long-context language models? In First Confer-
ence on Language Modeling .
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense informa-
tion retrieval with contrastive learning. Transactions
on Machine Learning Research .
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. In Proceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume ,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Yeonseok Jeong, Minsoo Kim, Seung-won Hwang, and
Byung-Hak Kim. 2025. Agent-as-judge for factual
summarization of long narratives. arXiv preprint
arXiv:2501.09993 .
Albert Q Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, et al. 2023a. Mistral
7b.arXiv preprint arXiv:2310.06825 .
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023b. LLMLingua: Compress-
ing prompts for accelerated inference of large lan-
guage models. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing , pages 13358–13376, Singapore. Association
for Computational Linguistics.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dong-
sheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu.
2024. LongLLMLingua: Accelerating and enhanc-
ing LLMs in long context scenarios via prompt com-
pression. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1658–1677, Bangkok,
Thailand. Association for Computational Linguistics.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781,
Online. Association for Computational Linguistics.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha,
and Jinwoo Shin. 2024. Sure: Improving open-
domain question answering of LLMs via summarized
retrieval. In The Twelfth International Conference on
Learning Representations .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: A benchmark
for question answering research. Transactions of the
Association for Computational Linguistics , 7:452–
466.
Tiziano Labruna, Jon Ander Campos, and Gorka
Azkune. 2024. When to retrieve: Teaching llms to
utilize information retrieval effectively. Preprint ,
arXiv:2404.19705.
Kyungjae Lee, Seung-won Hwang, Sang-eun Han, and
Dohyeon Lee. 2021. Robustifying multi-hop qa
through pseudo-evidentiality training. In Proceed-
ings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing
(Volume 1: Long Papers) , pages 6110–6119.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue,
and Wenhu Chen. 2024. Long-context llms strug-
gle with long in-context learning. arXiv preprint
arXiv:2404.02060 .
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 6342–6353, Singa-
pore. Association for Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.
Ziyang Luo, Can Xu, Pu Zhao, Xiubo Geng, Chongyang
Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023.
Augmented large language models with parametric
knowledge guiding. Preprint , arXiv:2305.04757.
10

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
et al. 2024. Self-refine: Iterative refinement with
self-feedback. Advances in Neural Information Pro-
cessing Systems , 36.
Jesse Mu, Xiang Li, and Noah Goodman. 2024. Learn-
ing to compress prompts with gist tokens. Advances
in Neural Information Processing Systems , 36.
Yilin Niu, Fangkai Jiao, Mantong Zhou, Ting Yao, Jing-
fang Xu, and Minlie Huang. 2020. A self-training
method for machine reading comprehension with soft
evidence extraction. In Proceedings of the 58th An-
nual Meeting of the Association for Computational
Linguistics , pages 3916–3927, Online. Association
for Computational Linguistics.
OpenAI. 2023. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin
Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor
Ruhle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao,
Lili Qiu, and Dongmei Zhang. 2024. LLMLingua-
2: Data distillation for efficient and faithful task-
agnostic prompt compression. In Findings of the
Association for Computational Linguistics ACL 2024 ,
pages 963–981, Bangkok, Thailand and virtual meet-
ing. Association for Computational Linguistics.
Cheng Qian, Xinran Zhao, and Tongshuang Wu. 2024.
”merge conflicts!”’ exploring the impacts of exter-
nal knowledge distractors to parametric knowledge
graphs. In First Conference on Language Modeling .
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Rohit Saxena and Frank Keller. 2024. Select and sum-
marize: Scene saliency for movie script summariza-
tion. In Findings of the Association for Computa-
tional Linguistics: NAACL 2024 , pages 3439–3455.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H Chi, Nathanael Schärli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context. In Inter-
national Conference on Machine Learning , pages
31210–31227. PMLR.
Yongho Song, Dahyun Lee, Myungha Jang, Seung-won
Hwang, Kyungjae Lee, Dongha Lee, and Jinyoung
Yeo. 2024. Evidentiality-aware retrieval for overcom-
ing abstractiveness in open-domain question answer-
ing. In Findings of the Association for Computa-
tional Linguistics: EACL 2024 , pages 1930–1943.
Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia,
Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara
Bahri, Tal Schuster, Steven Zheng, Denny Zhou, Neil
Houlsby, and Donald Metzler. 2023. UL2: Unifyinglanguage learning paradigms. In The Eleventh Inter-
national Conference on Learning Representations .
Gemma Team, Morgane Riviere, Shreya Pathak,
Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati-
raju, Léonard Hussenot, Thomas Mesnard, Bobak
Shahriari, Alexandre Ramé, et al. 2024. Gemma 2:
Improving open language models at a practical size.
arXiv preprint arXiv:2408.00118 .
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. ♪musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Chenguang Wang, Xiao Liu, and Dawn Song. 2020.
Language models are open knowledge graphs. arXiv
preprint arXiv:2010.11967 .
Yizhong Wang, Swaroop Mishra, Pegah Alipoormo-
labashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva
Naik, Arjun Ashok, Arut Selvan Dhanasekaran, An-
jana Arunkumar, David Stap, et al. 2022. Super-
naturalinstructions: Generalization via declarative
instructions on 1600+ nlp tasks. In Proceedings of
the 2022 Conference on Empirical Methods in Natu-
ral Language Processing , pages 5085–5109.
Guillaume Wenzek, Marie-Anne Lachaux, Alexis Con-
neau, Vishrav Chaudhary, Francisco Guzmán, Ar-
mand Joulin, and Edouard Grave. 2020. CCNet:
Extracting high quality monolingual datasets from
web crawl data. In Proceedings of the Twelfth Lan-
guage Resources and Evaluation Conference , pages
4003–4012, Marseille, France. European Language
Resources Association.
Kevin Wu, Eric Wu, and James Zou. 2024. How faith-
ful are rag models? quantifying the tug-of-war be-
tween rag and llms’ internal prior. arXiv preprint
arXiv:2404.10198 .
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th international ACM SIGIR
conference on research and development in informa-
tion retrieval , pages 641–649.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. Re-
comp: Improving retrieval-augmented lms with con-
text compression and selective augmentation. In The
Twelfth International Conference on Learning Repre-
sentations .
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
Preprint , arXiv:2401.15884.
11

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang, Min-
byul Jeong, and Jaewoo Kang. 2024. CompAct:
Compressing retrieved documents actively for ques-
tion answering. In Proceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing , pages 21424–21439, Miami, Florida, USA.
Association for Computational Linguistics.
Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu,
Mingxuan Ju, Soumya Sanyal, Chenguang Zhu,
Michael Zeng, and Meng Jiang. 2023. Generate
rather than retrieve: Large language models are
strong context generators. In The Eleventh Inter-
national Conference on Learning Representations .
12

Appendices
A Further Analysis
A.1 Comparative Analysis of Compression
Methods
In this section, we will provide a more detailed
comparison of our approach with other baselines
based on Table 1. Table 5 provides an overview of
how each method differs. Based on this compari-
son, we discuss how large-scale documents can be
compressed efficiently and effectively.
In ODQA, since the model must provide an an-
swer to a given question, the compression process
needs to consider the question. LLMLingua (Jiang
et al., 2023b) and LLMLingua-2 (Pan et al., 2024),
which do not consider the question during compres-
sion, often include irrelevant information, leading
to suboptimal performance. On the other hand, the
methods other than LLMLingua and LLMLingua-2
are question-aware, allowing them to more effec-
tively capture the necessary content, resulting in
higher performance compared to question-agnostic
methods.
The amount of evidence needed varies for each
question, and one solution to address this is adap-
tive compression, where the compression ratio
is adjusted for each question. By applying this
method, only the necessary tokens are used, lead-
ing to high performance with fewer tokens. As seen
in Table 1, both CompAct (Yoon et al., 2024) and
ECoRAG achieve high performance with a reduced
number of tokens.
However, there are two main challenges when
dealing with long context. First, while using nu-
merous retrieval results increases the amount of
necessary information available, it also includes
documents with lower relevance scores, resulting
in considerable noise. Second, the overall length
of the documents is too long, which makes the
compression process time-consuming.
To address the first challenge mentioned above,
the concept of evidentiality is necessary. As dis-
cussed in Section 3.1.1, by prioritizing strong evi-
dence for correct answer generation and penalizing
distractors, we have been able to create a compres-
sor that is robust against noise. Consequently, this
approach allows ECoRAG to demonstrate the high-
est performance in large-scale document settings.
To address the second challenge, the compres-
sor must be an extractive compressor that evalu-
ates each content pointwise and extracts only thenecessary information. Language model-based ab-
stractive compressor is hindered by limited context
length, which leads to truncation and fails to han-
dle entire large-scale documents. Moreover, LLM-
based abstractive compressor often requires sub-
stantial time for inference and may suffer from po-
sitional biases (Liu et al., 2024), which can lead to
inaccurate assessments of evidentiality. However,
extractive compressors such as ECoRAG and RE-
COMP (extractive) (Xu et al., 2024) are lightweight
models that can quickly calculate scores, as seen in
Table 4, and process each document in parallel for
each document, thus avoiding positional biases.
Based on these observations, we conclude that
ECoRAG, which combines all the characteristics
from Table 5, is appropriate for compressing large-
scale documents effectively.
A.2 Evaluator Performance on NQ
30405060708090100
Accuracy Precision Recall F1NQ
CompAct Flan-UL2 Ours
Figure 5: Evidentiality evaluation metrics using differ-
ent evaluator, including ours, measured on the NQ.
We conducted same experiments on
NQ (Kwiatkowski et al., 2019), as described in
Section 5.2, observed similar trends to those in
TQA (Joshi et al., 2017). As shown in Figure 5, our
evidentiality evaluator consistently outperforms
CompAct and demonstrates comparable results
to Flan-UL2, further validating its effectiveness
across different datasets.
A.3 Compression Effectiveness with More
Long Context
To explore performance of ECoRAG with more
documents, we conducted additional experiments
using 1000 retrieved documents in Table 6. Previ-
ous compression work, such as CompAct, focused
on up to 30 documents, while our experiments used
100 documents, a common setting in RAG mod-
els like FiD (Izacard and Grave, 2021). To verify
13

MethodsQuestion
-awareAdaptive
CompressionEvidentiality
-guidedExtractive
Compression
LLMLingua, LLMLingua-2 ✗ ✗ ✗ ✓
LongLLMLingua ✓ ✗ ✗ ✓
RECOMP (extractive) ✓ ✗ ✗ ✓
RECOMP (abstractive) ✓ ✗ ✗ ✗
CompAct ✓ ✓ ✗ ✗
ECoRAG (ours) ✓ ✓ ✓ ✓
Table 5: The table compares different methods based on their key characteristics. Our approach, ECoRAG, integrates
all these features for fast and effective large-scale document compression.
Methods #tokens ↓ EM F1
RAG without compression
closed-book 0 21.33 28.71
standard RAG (1000 documents) 127,880 0.44 0.63
RAG with 1000 documents compressed
RECOMP (extractive) 661 31.39 42.29
ECoRAG (ours) 659 35.51 48.63
Table 6: Experimental results on the NQ test dataset
using GPT-4o-mini, comparing performance with
and without compression for 1000 retrieved docu-
ments (Karpukhin et al., 2020).
whether our method consistently improves perfor-
mance even with more documents, we tested with
1000 documents. Due to limited budget, we used
documents already retrieved by a DPR setting that
was searched, differing from our top-100 DPR set-
ting. We compared ECoRAG with RECOMP, an
extractive method with a similar structure, and ex-
cluded abstractive compressors such as CompAct
due to its too high latency in longer context com-
pression.
With 1000 documents, ECoRAG remained
highly effective in compressing and preserving es-
sential information. The context length became too
long for GPT-4o-mini to effectively utilize the in-
formation (Hsieh et al., 2024), as shown in Table 6.
However, our compression effectively reduced the
length, maintaining high performance. Addition-
ally, ECoRAG outperformed other extractive com-
pressors, demonstrating its superiority in handling
extensive document sets.
ECoRAG remains the most effective compressor
even for extremely long contexts. Without compres-
sion, excessive context length can degrade perfor-
mance or exceed the context limit. In contrast, our
retriever-based compressor efficiently compresses
extended inputs regardless of length.A.4 A Comparative Study with Reranker
ECoRAG fundamentally differs from reranking
methods like BGE-M3 (Chen et al., 2024) and
RECOMP by adaptively determining the rank and
compression ratio needed for each query. While
reranking models focus on relevance, they lack our
ability to iteratively refine compression based on
evidentiality. To ensure a fair comparison with our
approach in terms of token usage, we conducted ad-
ditional experiments with both BGE-M34and BGE-
reranker5(Xiao et al., 2024) by using its reranked
top-10 and top-20 sentences. As shown in Table 7,
ECoRAG achieves better performance, demonstrat-
ing the importance of selecting the appropriate con-
text over simply increasing or reducing the amount
of information.
Unlike other sentence reranking methods, EC-
oRAG evaluates the initial compression and adap-
tively adjusts the compression ratio through a re-
flection process to determine how much infor-
mation is required. This capability moves EC-
oRAG closer to true compression rather than sim-
ple reranking. Furthermore, our research extends
beyond proposing a compressor—it introduces a
complete framework. While we used Contriever to
ensure fair comparisons with RECOMP, our frame-
work is flexible and capable of training models like
BGE-M3 and BGE-reranker to learn LLM-based
evidentiality, further enhancing performance.
A.5 Adaptive Compression Ratio Analysis
To validate the claim of our adaptive compression
capabilities, we analyzed the distribution of com-
pression ratios across datasets. The compression
ratio is defined as the number of compressed tokens
divided by the number of original tokens. Table 8
summarizes the minimum, maximum, mean, me-
4BAAI/bge-m3
5BAAI/bge-reranker-large
14

MethodsNQ TQA WQ
#tokens EM F1 #tokens EM F1 #tokens EM F1
BGE-M3 (top 10) 330 33.02 45.47 370 64.12 74.34 322 20.77 38.27
BGE-M3 (top 20) 670 33.99 46.82 746 65.15 75.14 645 20.77 38.00
BGE-reranker (top 10) 436 34.16 47.73 – – – – – –
BGE-reranker (top 20) 838 34.82 47.95 – – – – – –
ECoRAG (ours) 632 36.48 49.81 441 65.34 75.37 560 30.17 46.13
Table 7: Performance on NQ, TQA and WQ using GPT-4o-mini, comparing dense retriever BGE-M3, BGE-reranker
(results shown only for NQ), and our ECoRAG.
DatasetMin Compression
RatioMax Compression
RatioMean Compression
RatioMedian Compression
RatioStandard
Deviation
NQ 0.0036 1 0.0401 0.0446 0.0247
TQA 0.0034 1 0.0267 0.0161 0.0221
Table 8: Compression ratio statistics for NQ and TQA datasets.
Compressor Evaluator Reader
VRAM usage 110M 770M ≥8B
Latency 0.70h 0.03h 4.23h
Table 9: VRAM usage and latency for each component
in ECoRAG on the NQ test set.
dian, and standard deviation of compression ratios
for the NQ and TQA datasets.
The results highlight differences between
datasets, with higher mean and median compres-
sion ratios observed for NQ. This reflects complex-
ity of dataset, requiring the extraction of answers
from lengthy Wikipedia documents through reason-
ing and comprehensive understanding. In contrast,
TQA involves documents with explicitly positioned
answers, making the task primarily about filtering
irrelevant information. Consequently, ECoRAG re-
trieves more evidence for NQ to address its higher
information needs, demonstrating its ability to ad-
just compression ratios adaptively based on dataset
complexity and information requirements.
A.6 Further Analysis on Efficiency
ECoRAG has demonstrated efficiency over tradi-
tional RAG, as shown in Table 1 and 4, but further
analysis is required to verify its resource and la-
tency efficiency. To compare resource usage, we
refer to Table 9. While traditional RAG requires at
least 8B VRAM in our experiments, ECoRAG only
adds additional 880M VRAM. Furthermore, since
the compressor and evaluator can operate sequen-
tially as well as simultaneously with the reader,
ECoRAG remains feasible in traditional RAG envi-
ronments.MethodsCompression
TimeInference
TimeTotal
TimeThroughput
(example/sec)
standard RAG - 8.55h 8.55h 0.08
ECoRAG (ours) 0.51h 2.94h 3.45h 0.20
Table 10: Inference time and compression time for NQ
test under worst case scenarios.
In terms of latency, Table 4 shows that ECoRAG
is more efficient than traditional RAG, but addi-
tional verification is needed across different cases.
The additional modules—compressor and evalu-
ator—may seem to increase system complexity.
However, traditional RAG must process the entire
long context, while ECoRAG reduces latency by
7.32h, as shown in Table 4. Table 9 shows that EC-
oRAG requires little time for compression, reduc-
ing the risk of bottleneck as the preceding modules
process efficiently. In the worst case, ECoRAG
evaluates compression multiple times, leading to
longer latency than in the best case. However, even
in the worst case, Table 10 demonstrates that EC-
oRAG is still faster than traditional RAG.
A.7 Case study of evidentiality-guided
compression
Table 11 illustrates an example of evidentiality-
guided compression. For the given question, who
dies at the end of Den of Thieves? with the correct
answer Merrimen , the initial document set before
compression includes the correct answer. But it
also contains irrelevant information, which mis-
leads the LLM into generating the wrong answer,
Donnie. After compression, irrelevant content con-
taining the word Donnie is effectively suppressed,
leaving only the evidential (highlighted) sentences.
15

Question Gold answers
who dies at the end of den of thieves Merrimen
Type In-context documents Prediction
None Donnie
retrieved
documentsDen of Thieves (film) Nick, forcing Nick to shoot him. AsMerrimen liesontheground
dying,Nick kneels andconsoles him. When Nick inspects Merrimen ’s SUV , he only finds
bags with shredded paper; he also finds that Donnie has escaped custody. Nick later goes to
Donnie ’s bar and sees pictures of him with some of the crew members from the heist. It is
revealed Donnie masterminded the heist to keep all of the stolen cash for himself in a second
garbage truck. After the passage of some time, Donnie is working in a London bar, planning
a new heist. The film was in Den of Thieves (film) is currently in development. In Los
Angeles, a team of robbers led by Ray Merrimen make a violent armed attack and hijack an
armored truck. Police officers arrive on the scene and engage in a shootout with the robbers.
Eventually, Merrimen and his crew escape with the empty armored truck. In the morning,
Detective Nick O’Brien investigates the crime scene, having been monitoring Merrimen and
his crew for a while. Suspecting a local bartender named Donnie for involvement, Nick finds
him at the bar and kidnaps him for interrogation. Donnie reveals Merrimen is planning
to rob the Federal Reserve on Den of Thieves (film) garbage truck that removes shredded
bills. Nick’s team catches up to Donnie and seizes him, beating him until he tells them
where Merrimen is going. Merrimen , Bosco, and Levi try to make their escape with the
money bags from the waste truck but hit a traffic jam and are blocked. Nick’s team spots
them and attempt to shoot them as the robbers try to escape. A shootout occurs initiated
byMerrimen , killing one of Nick’s men. Levi and Bosco are eventually shot dead, but
Merrimen gets away. Nick chases and shoots Merrimen , wounding him. Merrimen raises
an empty gun to Den of Thieves (film) is currently in development. In Los Angeles, a team
of robbers led by Ray Merrimen make a violent armed attack and hijack an armored truck.
Police officers arrive on the scene and engage in a shootout with the robbers. Eventually,
Merrimen and his crew escape with the empty armored truck. In the morning, Detective
Nick O’Brien investigates the crime scene, having been monitoring Merrimen and his crew
for a while. Suspecting a local bartender named Donnie for involvement, Nick finds him
at the bar and kidnaps him for interrogation. Donnie reveals Merrimen is planning to rob
the Federal Reserve on Den of Thieves (film) Friday of that week by covertly removing
about $30 million in old bills which are scheduled to be shredded after their serial numbers
are deleted from computer records. At their hideout, Merrimen has one of his crew, Levi,
roughly interrogate Donnie to ensure he didn’t disclose anything about the plan. Meanwhile,
Nick goes to a strip club and finds Merrimen ’s stripper girlfriend, hiring her for the night to
find out where the heist is going to happen. The next morning, Nick makes an effort to see
his daughter at her school. As the day of the heist comes, Merrimen andDonnie
Compression Den of Thieves (film) AsMerrimen liesontheground dying,Nick kneels andconsoles him.
Den of Thieves (film) Eventually, Merrimen and his crew escape with the empty armored
truck. Den of Thieves (film) Merrimen , Bosco, and Levi try to make their escape with
the money bags from the waste truck but hit a traffic jam and are blocked. Den of Thieves
(film) In the morning, Detective Nick O’Brien investigates the crime scene, having been
monitoring Merrimen and his crew for a while. Den of Thieves (film) Meanwhile, Nick
goes to a strip club and finds Merrimen ’s stripper girlfriend, hiring her for the night to find
out where the heist is going to happen.Merrimen
Table 11: Case study of how the compression of the retrieved documents helps the model to identify the correct
answer from NQ test set. The highlighted part is the evidential sentence that directly gives useful information for
generating the correct answer Merrimen , rather than the incorrect answer Donnie .
A.8 Generalizability across Readers
To evaluate the generalizability of our compres-
sion framework, we conducted experiments using
Flan-UL2 (Tay et al., 2023) (20B), Llama3 (Dubey
et al., 2024) (8B), and Gemma2 (Team et al., 2024)
(9B) as the reader LLMs. These models were cho-
sen to investigate how our method performs across
diverse architectures and parameter sizes.
Flan-UL2 was selected because RECOMP also
utilizes it, as we intend to directly compare with
it. Furthermore, additional experiments were con-
ducted with Llama3 and Gemma2 to extend theevaluation. Since Llama3 has large context length,
it can conduct ‘standard RAG’ experiment, unlike
Flan-UL2 and Gemma2.
Results show that our evidentiality-guided com-
pression method consistently outperforms other
compression baselines on all three models. Specifi-
cally, with Flan-UL2 in Table 12, which was used
to define evidentiality during training, the model
demonstrated a clear improvement across all met-
rics. Similarly, as shown in Table 13. Gemma2,
despite being trained without its own evidentiality
mining, also showed improved performance with
16

MethodsNQ TQA WQ
#tokens ↓ EM F1 #tokens ↓ EM F1 #tokens ↓ EM F1
RAG without compression
closed-book 0 21.33 28.71 0 46.48 52.47 0 32.97 42.33
standard RAG (100 documents) 15456 - - 15943 - - 15135 - -
RAG with 100 documents compressed
LLMLingua 725 19.17 25.48 726 42.97 48.93 868 31.10 40.87
LLMLingua-2 1475 24.63 32.19 1518 53.07 59.42 1580 30.61 41.76
LongLLMLingua 1516 38.03 46.94 1570 65.79 73.88 1629 32.78 45.27
RECOMP (extractive) 727 38.06 46.18 750 62.49 69.68 857 31.25 43.18
RECOMP (abstractive) 16 22.22 29.56 30 43.50 49.88 157 38.15 38.56
CompAct 252 42.16 51.05 253 64.37 72.25 218 33.07 44.45
ECoRAG (ours) 693 44.38 53.56 501 66.45 74.02 671 33.71 46.08
Table 12: Comparison of compression methods on NQ, TQA, and WQ using Flan-UL2 (Tay et al., 2023) with 100
retrieved documents (Karpukhin et al., 2020).
MethodsNQ TQA WQ
#tokens ↓ EM F1 #tokens ↓ EM F1 #tokens ↓ EM F1
RAG without compression
closed-book 0 27.84 38.35 0 57.11 66.39 0 26.77 43.24
standard RAG (100 documents) 14260 - - - - - 14075 - -
RAG with 100 documents compressed
LLMLingua 643 26.90 37.90 638 60.71 68.09 649 25.04 42.08
LLMLingua-2 1403 28.56 38.95 1393 59.95 67.84 1401 24.36 40.52
LongLLMLingua 1411 37.67 49.40 1436 63.17 70.28 1399 27.02 44.23
RECOMP (extractive) 165 37.65 48.24 687 63.19 70.38 680 26.03 42.22
RECOMP (abstractive) 17 27.98 38.00 28 58.78 65.74 21 25.20 41.60
CompAct 111 38.67 49.87 100 65.88 73.29 78 26.67 43.04
ECoRAG (ours) 684 39.20 50.24 448 66.32 74.25 504 27.41 44.00
Table 13: Comparison of compression methods on NQ, TQA, and WQ using Gemma2 (Team et al., 2024) with 100
retrieved documents (Karpukhin et al., 2020).
MethodsNQ TQA WQ
#tokens ↓ EM F1 #tokens ↓ EM F1 #tokens ↓ EM F1
RAG without compression
closed-book 0 22.16 32.36 0 60.89 67.80 0 21.79 35.81
standard RAG (100 documents) 14263 0.27 0.97 14574 0.24 2.70 14147 0.25 4.48
RAG with 100 documents compressed
LLMLingua 641 15.20 22.31 636 52.11 59.23 646 17.62 30.92
LLMLingua-2 1346 3.91 7.19 1366 48.08 55.91 1337 4.28 11.44
LongLLMLingua 1388 20.30 28.85 1423 58.34 68.49 1372 18.70 32.12
RECOMP (extractive) 160 22.33 31.12 683 36.69 44.08 667 16.19 27.80
RECOMP (abstractive) 16 18.75 27.85 27 42.73 50.94 21 18.80 33.25
CompAct 107 28.01 38.52 99 56.01 64.69 76 21.41 35.21
ECoRAG (ours) 519 30.22 42.55 445 59.25 69.32 588 21.60 35.43
Table 14: Comparison of compression methods on NQ, TQA, and WQ using Llama3 (Dubey et al., 2024) with 100
retrieved documents (Karpukhin et al., 2020).
17

Methods #tokens ↓ EM F1
RAG without compression
closed-book 0 31.88 44.10
standard RAG (100 documents) 13,847 37.11 50.82
RAG with 100 documents compressed
LLMLingua 645 25.79 37.56
LLMLingua-2 1,319 29.95 44.40
LongLLMLingua 1,364 33.44 46.20
RECOMP (extractive) 659 33.21 45.98
RECOMP (abstractive) 16 30.45 43.01
CompAct 75 37.69 51.65
ECoRAG (ours) 641 41.43 54.02
Table 15: Experimental results on the NQ dataset using
GPT-4o-mini, comparing performance with and without
compression for documents retrieved by Contriever.
our compression method, further validating its ef-
fectiveness.
In the case of Llama3, as presented in Ta-
ble 14, our compression approach outperformed
other baselines, including naive prepend. However,
in certain instances, it was outperformed by the
‘closed book’ approach. This suggests that paramet-
ric knowledge embedded within the reader LLM
can occasionally align well with specific datasets,
leading to variations in performance across models.
Nonetheless, our framework ECoRAG is model-
agnostic, as we have excluded the influence of the
parametric knowledge of the reader LLM in mining
evidentiality labels. These results emphasize that
our compression method consistently outperforms
other compression approaches, further validating
its effectiveness across diverse models and configu-
rations.
A.9 Generalizability across Retrievers
To verify that our compression approach general-
izes beyond DPR, we conducted additional experi-
ments using another retriever. Our initial choice of
DPR was intentional, in order to demonstrate the
robustness of our compression approach even un-
der challenging conditions where a weaker retriever
could introduce significant noise. In Table 15, we
then evaluated our method with Contriever (Izacard
et al., 2022), a stronger dense retriever. The results
show an even larger performance gain when paired
with Contriever than with DPR, indicating that
ECoRAG synergizes especially well with higher-
quality retrieval.
To compare against a simple baseline of re-
trieving fewer documents, we evaluated ECoRAG
against varying retrieval sizes. We initially chose
100 documents to align with standard practice in
prior work, such as Fusion-in-Decoder (IzacardMethods#tokens
(DPR)EM
(DPR)#tokens
(Contriever)EM
(Contriever)
Reduced retrieval size
# docs (k) = 5 693 35.53 690 33.48
# docs (k) = 10 1,386 35.95 1,381 34.76
# docs (k) = 20 2,774 36.33 2,762 36.47
Adaptive compression
ECoRAG (ours) 632 36.48 641 41.43
Table 16: Experimental results on the NQ dataset us-
ing GPT-4o-mini, comparing reduced retrieval sizes for
DPR and Contriever against adaptive compression via
ECoRAG.
Methods #tokens ↓ EM F1
RAG without compression
closed-book 0 26.19 36.71
standard RAG (100 documents) 14,313 34.52 44.69
RAG with 100 documents compressed
LLMLingua 636 22.57 31.54
LLMLingua-2 1,330 26.66 37.00
LongLLMLingua 1,406 27.45 38.07
RECOMP (extractive) 688 28.05 38.87
RECOMP (abstractive) 12 24.27 33.88
CompAct 74 31.21 42.42
ECoRAG (ours) 647 34.69 45.13
Table 17: Experimental results on the HotpotQA dataset
using GPT-4o-mini, comparing performance with and
without compression for 100 documents (Karpukhin
et al., 2020).
and Grave, 2021). In Table 16, we then conducted
experiments on varying numbers of retrieved docu-
ments ( k= 5, 10, and 20) for both DPR and Con-
triever, measuring token count and EM without
compression. Our results show that reducing k
does not improve accuracy as effectively as adap-
tive compression via ECoRAG, emphasizing the
benefit of our evidentiality-guided approach.
A.10 Evaluation in Multi-hop QA
To assess the effectiveness of ECoRAG in multi-
hop QA tasks requiring multiple evidence sources,
we conducted experiments in Table 17. ECoRAG
classifies evidentiality into three categories and
defines weak evidence that supports the correct
answer without directly generating the answer.
This enables ECoRAG to perform effectively in
tasks requiring partial evidence, such as multi-hop
QA. Furthermore, according to CompAct, adap-
tively adjusting evidence can collect the partial evi-
dence needed for multi-hop QA, ECoRAG achieves
through Evidentiality Reflection.
Table 17 shows that ECoRAG outperformed
both non-compressed and other compression base-
lines in HotpotQA (Yang et al., 2018). CompAct
and other baselines did not outperform the “stan-
dard RAG” approach, which uses all 100 docu-
18

Method HotpotQA MusiQue
standard RAG 49.76 23.91
RECOMP 49.12 23.08
ECoRAG (ours) 52.29 24.60
Table 18: Experiments on LongBench multi-hop
datasets (HotpotQA and MusiQue) evaluating F1-score
performance of standard RAG, RECOMP, and ECoRAG
using Llama3-8B.
ments without compression. In contrast, ECoRAG
improved performance by removing distractors and
keeping necessary evidence. These results show
that ECoRAG is effective for complex scenarios
such as multi-hop QA.
To evaluate ECoRAG in scenarios where the
challenge lies not only in the number but also in
the length of retrieved documents, we applied our
method to the LongBench (Bai et al., 2024) bench-
mark. LongBench is a long-context understand-
ing benchmark covering tasks such as HotpotQA6
and MuSiQue (Trivedi et al., 2022). In Table 18,
we compared standard RAG, RECOMP, and EC-
oRAG (using Llama3-8B) across these tasks within
LongBench. Consistent with our multi-hop results,
ECoRAG outperformed both non-compressed and
compression baselines in this long-document set-
ting, further demonstrating its robustness and effec-
tiveness.
B Experimental Details
B.1 Implementation Details
We used 8 Nvidia RTX3090 GPUs to train all mod-
els. For mining evidentiality labels for all sentences
in retrieved documents, we used the NLTK library7
to split DPR (Karpukhin et al., 2020) retrieved top-
100 documents into sentences. To reduce costs,
we used the open LLM Flan-UL28(Tay et al.,
2023), which was also used in our experiments and
RECOMP (Xu et al., 2024), to label evidentiality
based on the definition in Section 3.1.1.
Our evidentiality compressor was trained from
Contriever (Izacard et al., 2022) checkpoint pre-
trained on CC-net (Wenzek et al., 2020) and En-
glish Wikipedia (Izacard et al., 2022).. We trained
it using the AdamW optimizer with a batch size of
6LongBench does not include the full original datasets, so
our result in HotpotQA results may differ from those reported
in Table 17.
7www.nltk.org
8google/flan-ul2Subset Accuracy Precision Recall F1-score
exposed subset 68.52 81.47 64.17 72.13
non-exposed subset 71.77 78.64 76.22 77.42
Table 19: Evaluation results of the Flan-T5-based evi-
dentiality evaluator on TQA, comparing performance
between the exposed and non-exposed subsets.
64 and a learning rate of 5·10−5for 4 epochs on
NQ (Kwiatkowski et al., 2019) and WQ (Berant
et al., 2013), and 2 epochs on TQA (Joshi et al.,
2017). While training with LweandLselosses,
we we used 8 positive contexts and 56 negative
contexts per batch. When calculating the Lseloss,
we used negative set with weak evidence to dis-
tractor ratio of 0.15:0.85, treating weak evidence
as hard negative. We set the temperature τfor the
contrastive loss to 1.0.
Our evidentiality evaluator was trained from
a pretrained Flan-T5-large checkpoint9using the
AdamW optimizer. We trained it with a batch size
of 40 and a learning rate of 1·10−5for 4 epochs
with all datasets. We included ‘<NOT>’ sentences
with high compressor scores in the training stage
to make the evidentiality evaluator distinguish only
the genuinely strong evidence ‘<EVI>’ from the
seemingly plausible ones. We constructed the train-
ing data for the evaluator with a ratio of 1:3 be-
tween ‘<EVI>’ and ‘<NOT>’ sentences. For adap-
tive compression, a limit on the number of evidence
pieces was necessary to avoid infinite loops, which
we set at 20. We set this limit to 20 to achieve
a compression level similar to RECOMP, but it
can be increased for tasks that require more evi-
dence. Additionally, to prevent high latency due
to overly frequent evaluations, we incrementally
added 4 evidence pieces at a time. For experiments
on the test set, we used GPT-4o-mini10, Flan-UL2,
Gemma211, and Llama312.
B.2 Selection for Evidentiality Evaluator
We chose Flan-T5-large as the basis for our Evi-
dentiality Evaluator due to its strong instruction-
finetuning and robust performance on classifica-
tion tasks. T5-large has been widely used in prior
RAG research (Han et al., 2023; Yan et al., 2024)
for document-based evaluation. For our eviden-
tiality scoring task, we employ Flan-T5-large as it
demonstrates enhanced instruction-following capa-
9google/flan-t5-large
10gpt-4o-mini-2024-07-18
11google/gemma-2-9b-it
12meta-llama/Meta-Llama-3-8B-Instruct
19

bilities that are well-suited for this classification
task. However, a potential concern arises regard-
ing Flan-T5-large’s prior exposure to datasets such
as NQ, TQA, and WQ, which might lead to mem-
orization rather than genuine evidentiality learn-
ing. As reported in research related to Flan-T5-
large (Chung et al., 2022; Wang et al., 2022), its
exposure was clearly separated into exposed and
non-exposed subsets, and our comparison experi-
ments (Section 5.2) demonstrate negligible perfor-
mance differences between these groups. When
we applied a two-proportion Z test to the results in
Table 19, the analysis at the 0.05 significance level
confirmed that the observed differences are not sta-
tistically significant. Therefore, these observation
indicates that the model has learned evidentiality
principles rather than simply memorizing evidence
from prior exposure.
B.3 Input Prompts for LLM
We report two examples of input prompts for reader
LLMs. In Figure 6, we report the input prompt
used for evidentiality mining and test set experi-
ments to answer a given question when provided
with the question and the compressed documents.
This prompt was also utilized during the evidential-
ity mining process, as described in Section 3.1.1.
Figure 7 presents the input prompt for mining the
ground truth label of compressed documents us-
ing Flan-UL2 as the evidentiality evaluator in the
experiments detailed in Section 5.2.
C Usage of AI Assistants
We utilized ChatGPT to improve the clarity and
grammatical accuracy of my writing. It provided
suggestions for rephrasing sentences and correct-
ing grammatical errors to make the text flow more
naturally.
20

Question Answering Prompt
who won a million on deal or no deal
Answer: Tomorrow Rodriguez
who is the woman washing the car in cool hand luke
Answer: Joy Harmon
who is the actor that plays ragnar on vikings
Answer: Travis Fimmel
who said it’s better to have loved and lost
Answer: Alfred , Lord Tennyson
name the first indian woman to be crowned as miss world
Answer: Reita Faria
Documents
Question
Answer:
Figure 6: An input prompt for LLM for question answering, including few-shot examples, input documents, and a
question.
21

Evidentiality Evaluation Prompt
You are an expert at determining whether a document provides evidential support for a given
question. You will receive a question and a document, and your task is to evaluate whether the
document is evidential, partially evidential, or non-evidential in relation to the question.
Assess the support provided by the document using the following scale:
- [Evidential] - The document fully supports the question, providing clear and direct evidence that
answers or addresses the query completely.
- [Non-Evidential] - The document does not provide relevant information or evidence related to the
question, making it unrelated or insufficient to support the query.
Please provide your assessment and briefly justify your reasoning based on the content of the
document in relation to the question.
Question: what is the temperature of dry ice in kelvin?
Evidence: At atmospheric pressure, sublimation/deposition occurs at or 194.65 K. The density of
dry ice varies, but usually ranges between about.
Score: [Evidential]
Question: when did north vietnam unify with the south?
Evidence: The distinctive synthesizer theme was performed by the then-little-known Thomas
Dolby, and this song also marked a major departure from their earlier singles because their
previous singles were mid to upper tempo rock songs while this song was a softer love song with
the energy of a power ballad.
Score: [Non-Evidential]
Question: who played all the carly ’s on general hospital?
Evidence: Throughout the 2000s, Carly, then Tamara Braun (2001–05) goes on to become one of
the
Score: [Non-Evidential]
Question: who sang the original blinded by the light?
Evidence: Light of Day (song) "Light of Day", sometimes written as "(Just Around the Corner to
the) Light of Day", is a song written by Bruce Springsteen and performed initially by Joan Jett and
Michael J.
Score: [Non-Evidential]
Question: who was the rfc editor until 1998 just provide the family name?
Evidence: Perhaps his most famous legacy is from RFC 760, which includes a robustness principle
often called "Postel’s law": "an implementation
Score: [Non-Evidential]
Question: Question
Evidence: Compressed Documents
Score:
Figure 7: An input prompt for LLM for evidentiality evaluation, including few-shot examples, compressed
documents, and a question.
22
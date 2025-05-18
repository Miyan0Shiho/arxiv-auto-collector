# CL-RAG: Bridging the Gap in Retrieval-Augmented Generation with Curriculum Learning

**Authors**: Shaohan Wang, Licheng Zhang, Zheren Fu, Zhendong Mao

**Published**: 2025-05-15 16:53:04

**PDF URL**: [http://arxiv.org/pdf/2505.10493v1](http://arxiv.org/pdf/2505.10493v1)

## Abstract
Retrieval-Augmented Generation (RAG) is an effective method to enhance the
capabilities of large language models (LLMs). Existing methods focus on
optimizing the retriever or generator in the RAG system by directly utilizing
the top-k retrieved documents. However, the documents effectiveness are various
significantly across user queries, i.e. some documents provide valuable
knowledge while others totally lack critical information. It hinders the
retriever and generator's adaptation during training. Inspired by human
cognitive learning, curriculum learning trains models using samples progressing
from easy to difficult, thus enhancing their generalization ability, and we
integrate this effective paradigm to the training of the RAG system. In this
paper, we propose a multi-stage Curriculum Learning based RAG system training
framework, named CL-RAG. We first construct training data with multiple
difficulty levels for the retriever and generator separately through sample
evolution. Then, we train the model in stages based on the curriculum learning
approach, thereby optimizing the overall performance and generalization of the
RAG system more effectively. Our CL-RAG framework demonstrates consistent
effectiveness across four open-domain QA datasets, achieving performance gains
of 2% to 4% over multiple advanced methods.

## Full Text


<!-- PDF content starts -->

arXiv:2505.10493v1  [cs.CL]  15 May 2025CL-RAG: Bridging the Gap in Retrieval-Augmented Generation with
Curriculum Learning
Shaohan Wang, Licheng Zhang , Zheren Fu, Zhendong Mao
University of Science and Technology of China, Hefei, China
{wsh2000, zlczlc, fzr}@mail.ustc.edu.cn zdmao@ustc.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) is an
effective method to enhance the capabilities
of large language models (LLMs). Existing
methods focus on optimizing the retriever or
generator in the RAG system by directly uti-
lizing the top-kretrieved documents. How-
ever, the documents effectiveness are various
significantly across user queries, i.e.some doc-
uments provide valuable knowledge while oth-
ers totally lack critical information. It hinders
the retriever and generator’s adaptation during
training. Inspired by human cognitive learn-
ing, curriculum learning trains models using
samples progressing from easy to difficult, thus
enhancing their generalization ability, and we
integrate this effective paradigm to the training
of the RAG system. In this paper, we propose a
multi-stage Curriculum Learning based RAG
system training framework, named CL-RAG .
We first construct training data with multiple
difficulty levels for the retriever and genera-
tor separately through sample evolution. Then,
we train the model in stages based on the cur-
riculum learning approach, thereby optimizing
the overall performance and generalization of
the RAG system more effectively. Our CL-
RAG framework demonstrates consistent effec-
tiveness across four open-domain QA datasets,
achieving performance gains of 2% to 4% over
multiple advanced methods.
1 Introduction
Large language models (LLMs) have demonstrated
remarkable capabilities in a wide range of Natu-
ral Language Processing (NLP) tasks (Brown et al.,
2020; Anil et al., 2023; Dubey et al., 2024), but they
still constrained by the limitations of the knowledge
embedded within their internal parameters (Roberts
et al., 2020; Kandpal et al., 2023; Gao et al., 2023).
Retrieval-Augmented Generation (RAG) addresses
this limitation by leveraging additional knowledge
retrieved from external knowledge bases. By re-
trieving relevant documents and incorporating them
Q
Q
QRetrieve Results…~………~
~
Training Randomly
Retriever
&
GeneratorQ
Q
QRetrieve Results
…~………~
~Rewrite
Training Stage by StageEasy
Hard
Reconstruct
Retriever
&
GeneratorRerank
Easy 
DocumentsCommon 
DocumentsHard 
DocumentsPrevious Studies Our Framework
(a) Previous Studies (b) Our FrameworkFigure 1: Comparison between our method and pre-
vious studies, where green represents documents that
are conducive to model responses, while red indicates
documents that are useless or even detrimental to model
responses. Figure 1(a) illustrates the previous method,
which randomly samples documents of varying quality
without considering the difficulty order of the docu-
ments. Figure 1(b), in contrast, presents our approach,
which focuses on reordering and combining documents
based on their difficulty levels for stage-by-stage train-
ing.
as contextual input, RAG has significantly en-
hanced the capabilities of existing large models
in tasks such as Open-Domain Question Answer-
ing (Izacard et al., 2023; Shi et al., 2023; Yoran
et al., 2023; Lin et al., 2023; Fang et al., 2024)
and Natural Language Modeling (Borgeaud et al.,
2022; Ram et al., 2023; Zhang et al., 2024). The
overall performance of the RAG system depends
on the quality of the retrieved documents and the
LLMs’ ability to utilize these documents. There-
fore, how to retrieve better documents and how to
better utilize the retrieved documents are of vital
importance in RAG research.
Existing efforts to enhance RAG systems have
1

focused on: (1) Utilizing LLMs to evaluate docu-
ment quality and guide retriever training (Shi et al.,
2023; Zhang et al., 2024); (2) Training Retrieval-
Augmented Language Models (RALMs) that can
better utilize documents (Izacard and Grave, 2020;
Yoran et al., 2023; Fang et al., 2024), or a combina-
tion of both (Lin et al., 2023) to enhance the overall
performance of the RAG system. However, given
a specific query, these researches directly use the
top-kretrieved documents as training samples and
randomly input them into the model for training, as
shown in Figure 1(a). This approach neglects the
substantial quality variations among retrieved doc-
uments, where the top-kdocuments for different
queries may vary greatly in quality: some provide
valuable knowledge while others lack critical infor-
mation or even contain misleading content. These
discrepancies pose varying learning challenges for
the retriever and generator during training.
Inspired by cognitive science studies (Elman,
1993; Rohde and Plaut, 1999) that humans can
benefit from a easy-to-difficult learning sequence,
the curriculum learning (Bengio et al., 2009) sug-
gests that deep learning models trained by the grad-
ual introduction of more challenging samples in
subsequent stages. Previous studies (Guo et al.,
2018; Hacohen and Weinshall, 2019; Xu et al.,
2020) have demonstrated that this easy-to-difficult
strategy can effectively enhance the generalization
ability of models. Therefore, we propose that the
training of the RAG system should also adhere to
this paradigm, which has been neglected in previ-
ous works. To bridge this gap, we propose a sample
evolution strategy to construct document data with
different levels of difficulty and further train the
retriever and generator based on the idea of curricu-
lum learning, as demonstrated in Figure 1(b).
In this work, we propose a Curriculum Learning
based training framework for the RAG system,
named CL-RAG . To the best of our knowledge,
we are the first to integrate the idea of human im-
itation learning with RAG training, which can ef-
fectively enhance the generalization and stability
of the RAG system. Specifically, we first focus on
the training of the generator, constructing multiple
difficulty levels of documents by rewriting query-
enhanced golden documents or counterfactual dis-
tractor documents. Then we finetune a RALM in a
stage-by-stage manner according to the order of dif-
ficulty. For the training of the retriever, we propose
to use the well-trained RALM to assess the quality
of documents and rerank them. We then constructdocument data from easy to difficult by gradually
reducing the ranking gap between sampled docu-
ments, and propose a hierarchical training strategy
to finetune the retriever.
As a result, CL-RAG bring two advantages to
RAG system: (1) For the generator, CL-RAG sys-
tematically transition it from merely extracting in-
formation to effectively countering potential dis-
tracting noise within documents. (2) For the re-
triever, CL-RAG train it in stages to progressively
distinguish documents with obvious differences
and then distinguish those with only slight differ-
ences. This progressive training framework has led
to an overall performance and generalization im-
provement of the RAG system. Our contributions
can be concluded as follows:
•We propose the CL-RAG training framework
for the RAG system based on the concept of
curriculum learning. To the best of our knowl-
edge, this is one of the first times that CL
strategy has been applied to optimize RAG
systems.
•We defined the document difficulty levels for
the retriever and generator based on the con-
cept of curriculum learning and designed a
complete method for constructing enhanced
documents.
•We evaluated our CL-RAG framework on
four popular datasets, demonstrating consis-
tent performance gains of 2% to 4% over mul-
tiple advanced methods, thereby highlighting
the superiority of our approach.
2 Related Work
2.1 Retrieval-augmented Generation
Using documents retrieved from extended knowl-
edge bases to enhance the capabilities of large
language models (LLMs) has been proven effec-
tive in NLP tasks, including language modeling
(Borgeaud et al., 2022; Ram et al., 2023; Zhang
et al., 2024) and question answering (Izacard et al.,
2023; Shi et al., 2023; Yoran et al., 2023; Lin et al.,
2023; Fang et al., 2024). Specifically, a Retrieval-
Augmented Generator (RAG) system takes a query
as input and uses a retriever to retrieve relevant doc-
uments from an external knowledge base. Then, it
combines the documents with the query and feeds
them into the LLM to make up for the LLM’s own
lack of knowledge.
2

Training Retriever Retriever
（Raw）
Query: Who was the director of 
More (1969)?(Ground truth: Barbet Schroeder)Documents Prepare[1] More (1969 film) is … directed by Barbet Schroeder…
[2] More (1998 film) was written and directed by Mark Osborne…
······Common documents
[1] More (1969 film) is … directed by Barbet Schroeder…
[2] More is a…written and directed by Pierre Cardin …
······Hard documents
Q
Q𝑟𝑟𝑎𝑎=8Prior
𝑟𝑟𝑎𝑎=1
Q 𝑟𝑟𝑎𝑎=10+
+
………RALM preferenceRetriever
（Raw）
rerank sample
𝐷𝐷𝐷𝐷𝑐𝑐1
𝐷𝐷𝐷𝐷𝑐𝑐𝑛𝑛Golden Answer 
Rank  Training Stages
Generator
(RALM)Training Generator···
Generator
(RALM)[1] More (1969 film) is … directed by Barbet Schroeder…
[2] The 1969 film “More”… Directed by Barbet Schroeder in…
······Easy documentsTraining Stages
Retriever
(LLM preferred)Generator
(Raw LLM)
Common
HardEasyQuery -enhance
rewrite
Counterfactual
rewrite
Figure 2: The overview of our CL-RAG training framework, which has into two continuous phases: (1) Training
Generator: We construct multiple difficulty levels of documents and then finetune a RALM in a stage-by-stage
manner. (2) Training Retriever: We use the well-trained RALM to assess documents and rerank them. We then
construct document data from easy to difficult, and finetune the retriever in a stage-by-stage manner.
Optimization of the RAG system focuses on two
main areas: improving the retriever and enhancing
the generator (LLM) to the RALM. Replug (Shi
et al., 2023) uses KL divergence to align retriever
results with LLM preferences. LLM-Embedder
(Zhang et al., 2024) employs a distillation objec-
tive based on LLM rankings. These methods train
the retriever to better match LLM preferences. For
generator, FiD (Izacard and Grave, 2020) finetunes
LLM to handle retrieved documents and queries,
addressing irrelevant information. Other studies
introduce noise to improve LLM robustness (Yoran
et al., 2023; Fang et al., 2024). Combining the
strengths of both approaches, RA-DIT (Lin et al.,
2023) uses modular training to optimize the re-
triever and LLM separately, enhancing overall
RAG system performance.
2.2 Curriculum Learning
Curriculum Learning (CL) is a machine learning
strategy that mimics human learning by training
models gradually, starting with simpler tasks and
progressing to more complex ones. It aims to im-
prove the model’s generalization ability and accel-
erate its convergence speed. Early studies have
extensively investigated CL in computer vision do-
main and have demonstrated its advantages in train-ing deep models (Guo et al., 2018; Hacohen and
Weinshall, 2019). In the NLP domain, Xu et al.
(2020) first explore and validate the effectiveness
of CL in the context of finetuning LMs on Natu-
ral Language Understanding (NLU) tasks. Their
framework achieved general performance improve-
ments across various NLU tasks, including Ma-
chine Reading Comprehension (MRC) and Natural
Language Inference (NLI). In terms of retrievers,
recent research has shown that by gradually in-
creasing the difficulty of sampled data, CL can also
bring significant performance improvements in the
training of embedders (Zhu et al., 2022; Zeng et al.,
2022; He et al., 2023).
3 Methodology
In this section, we will provide a detailed introduc-
tion to our CL-RAG training framework. We first
briefly introduce the RAG pipeline in Section 3.1.
Then, in Sections 3.2 and 3.3, we present the cur-
riculum learning methods we propose for RALM
training and retriever training, respectively. An
overview of our CL-RAG training framework is
given in Figure 2.
3

3.1 Preliminary
The RAG system combines a Retriever that re-
trieves query-relevant documents from external
data bases with a Generator (LLM) that synthe-
sizes responses from retrieved documents.
Retriever Given a query q, the Retriever aims to
retrieve documents {d1, d2, ..., d n}relevant to the
query from an external knowledge base D. In this
work, we employ a dense retriever. Specifically,
we use a dual encoder to encode the input query q
and the documents dintoE(q)andE(d). The sim-
ilarity between them is defined by cosine similarity,
which serves as the score for document retrieval:
score i=cos(E(q), E(di)).
Typically, we select the top-kdocuments with the
highest scores as the input to the generator.
Generator Given the top-kretrieved documents,
the goal of the Generator (LLM) is to utilize these
external documents to better answer the question.
Generally, the retrieved documents are concate-
nated with the query qas contextual information,
and then fed into the Generator to produce the an-
swer:
Output =LLM (d1⊕d2⊕...⊕dk, q).
In this paper, we employ a raw retriever to retrieve
nrelevant documents. For generator (LLM) train-
ing, we construct curriculum-ordered data from
easy to difficult examples by applying data augmen-
tation and document rewriting to top-kretrieved
documents. Subsequently, for retriever training,
we leverage the finetuned generator (RALM) to
assess the quality of nretrieved documents. We
then perform reranking and sampling to construct
curriculum-ordered training data to train the re-
triever to align with RALM’s preference.
3.2 Curriculum Learning for RALM
In this section, we will present the data construc-
tion process for training the RALM, along with the
corresponding curriculum learning framework.
To construct data with increasing difficulty for
curriculum learning, we first categorize and con-
struct documents of various difficulty levels. Pre-
vious work (Yoran et al., 2023; Fang et al., 2024)
has considered data augmentation and constructed
documents with relevant noise ,irrelevant noise ,
andcounterfactual noise . However, they neglected
Rewrite Documents
Instruction: Please rewrite the given text to make it better assist
in answering the provided question. 
Question: Who was the director of More (1969)?
Original text (100+ tokens) :More (1969 film) More is an 
English -language drama-romance film written and directed by 
Barbet Schroeder, in his theatrical feature film directorial debut, released in 1969……
Rewritten Text (20 tokens ): The 1969 film "More" is directed 
by Barbet Schroeder in his directorial debut.
Instruction: Make false modifications to the facts(country, region, 
year, etc.) in the given text and rewrite the text while maintaining the similar quotation marks. You don't need to consider whether the rewritten object is a real person or object. 
Original text: More (1969 film) More is an English -language 
drama -romance film written and directed by Barbet Schroeder, in 
his theatrical feature film directorial debut, released in 1969…
Rewritten Text: More is a French -language drama-romance film 
written and directed by Pierre Cardin , in his theatrical feature 
film directorial debut…Rewrite Golden Documents
Rewrite Counterfactual DocumentsFigure 3: Case of document rewriting, including
query-enhanced document rewriting and counter-
factual document rewriting. The highlighted green
text represents the keywords within the document that
aid in addressing the question, while the highlighted red
text signifies wrong knowledge.
to include simple documents that definitely con-
tain the correct answers, which is essential during
the initial training stage for the model to learn the
fundamental ability to answer questions from doc-
uments that contain the answers.
We first classify documents into three levels of
difficulty: Easy, Common, and Hard. The Easy
level contains at least two documents with correct
answers to ensure the model can acquire prelim-
inary answer extraction capabilities. The Com-
mon level follows the original retrieved documents,
which may not contain the exact correct answers,
thereby training the model’s ability to infer answers
using documents of average quality. The Hard level
includes the most challenging documents for the
model, potentially containing irrelevant or even
harmful noise, which trains the model’s ability to
counteract disruptive information.
Since the retriever’s results may not always in-
clude the correct answers, to construct the docu-
ments for Easy level, we use the documents pro-
vided for the corresponding questions in the MRQA
reading comprehension task (Fisch et al., 2019) as
the golden documents to ensure that they contain
the exact correct answers. Meanwhile, we rewrite
the golden document as shown in Figure 3 to obtain
document that better meet the model’s needs. Com-
4

pared to using only the golden document, using
both the golden document and its rewritten version
ensures that there are at least two documents con-
taining answers within the top-kdocuments, which
helps train the model’s ability to extract consistent
information from multiple documents. We com-
plete the total number of documents to kusing the
original retrieved documents.
For Common level, we directly use the top-k
documents from the retriever’s results. Unlike doc-
uments in Easy level, these may not contain the
correct answers and have more noise related to the
questions, which may increase the difficulty for the
model to answer the questions.
For Hard level, we randomly select a document
from the top-kand replace it with retrieved docu-
ment from other questions or perform counterfac-
tual rewriting as shown in Figure 3. This introduces
irrelevant and counterfactual noise, which repre-
sents the most challenging type of documents for
the model, aiming to train the model’s robustness
against various types of real-world data.
During the training phase, data from the three
difficulty levels are fed into the model in sequence
to conduct Supervised Fine-Tuning (SFT). Through
the curriculum learning approach that progresses
from easy to difficult, the model is gradually trained
to extract answers from documents and to resist the
interference of distracting documents.
3.3 Curriculum Learning for Retriever
In this section, we will introduce our retriever train-
ing strategy. It initially employs a well-trained
RALM to rerank documents and constructs data
through stratified sampling. Subsequently, the re-
triever is trained in stages using a curriculum learn-
ing approach to align with the RALM.
We first employ the trained RALM to assess doc-
uments, thereby eliciting its preferences for them.
Differs from that in Lin et al. (2023), which relies
on the raw LLM, utilizing RALM for evaluation
provides a more effective means of differentiating
between high-quality and low-quality documents.
Previous researchers (Shi et al., 2023; Zhang
et al., 2024) assess document quality based on the
probability of the model generating the correct an-
swer and the improvement in decoding rank, re-
spectively. However, using either method alone
has its limitation: the probability of generating the
correct answer can be unstable when the input doc-
uments are diverse, and the decoding rank cannot
finely distinguish between good and bad documentswhen the model itself is capable of generating the
correct answer, i.e., when most of the input doc-
uments have a decoding rank of 1 for the correct
answer.
To address these limitations, we propose using
the improvement in decoding rank as a coarse mea-
sure of document quality, and then using the prob-
ability of generating the correct answer to finely
rank documents with the same decoding rank im-
provement. This helps us obtain a reranked list of
retrieved documents:
Drerank ={d1, d2, ..., d n}.
Next, we construct data with different difficulty
levels for curriculum learning, aiming to first
train the retriever to distinguish documents with
obvious quality gaps and then train it to distin-
guish documents with slightly quality differences.
Specifically, we divide Drerank into three groups
{D1,D2,D3}based on the ranking order from
RALM. These groups contain documents in the
ranges [1, n1],[n1, n2], and [n2, n], respectively,
representing good documents, sub-optimal docu-
ments, and hard negative documents. During train-
ing stages, we sample k1,k2, and k3documents
from each of the three groups, while k1+k2+k3=
k, and gradually increase k1and decrease k2and
k3, narrowing the quality gap between the sampled
documents. We have also designed a tiered loss
function to better fit our approach:
L(q,Dk) =−X
di,j∈DkX
j>ij−i
n−1log(esi
esi+esj),
whereDkdenotes a subset of kdocuments selected
fromD,nrepresents the total number of docu-
ments involved in the ranking, and i,jare the ranks
of the documents, sidenotes the similarity score
between the query qand the document di. We start
by distinguishing documents with obvious quality
differences and gradually move to distinguishing
documents with slight quality differences. This
stage-by-stage approach helps bridge the gap be-
tween the original retriever and the retriever aligned
with RALM preferences.
4 Experiments
4.1 Experimental Setting
4.1.1 Datasets
We first evaluated our complete training frame-
work under the standard open-domain question-
answering (QA) setting. Subsequently, we assessed
5

MethodNQ TriviaQA PopQA HotpotQA Avg.
EM F1 EM F1 EM F1 EM F1 EM F1
Llama 38B 39.51 53.37 72.74 81.80 46.07 52.16 25.23 39.46 45.89 56.70
With RALM Training
RAAT 31.65 40.39 71.84 77.02 38.49 41.87 22.16 30.26 41.04 47.39
RALM golden 42.90 52.30 79.55 83.84 50.06 52.45 31.25 41.96 50.94 57.64
RALM top5 48.15 58.95 80.32 85.05 57.58 60.13 36.57 48.46 55.66 63.15
RetRobust 47.83 58.60 81.25 85.96 57.51 60.31 37.02 48.99 55.90 63.47
RALM CL(Ours )51.53 61.19 82.93 87.59 59.71 62.28 38.14 50.22 58.08 65.32
With Retriever Training
RA-DIT 49.52 59.94 76.62 81.10 61.73 64.47 36.14 47.82 56.00 63.33
Replug 52.52 62.28 76.86 81.59 63.75 66.35 37.75 49.22 57.72 64.86
LLM -Emebedder 51.85 61.41 82.60 87.20 64.24 66.69 38.80 50.51 59.37 66.45
CL-RAG (Ours ) 53.01 62.51 82.97 87.57 66.63 69.12 39.11 51.19 60.43 67.60
Table 1: Experimental results for EM and F1 scores(%) on four open-domain QA datasets compared with
multiple baselines in RALM or Retriever training. "RALM CL" refers to the RALM trained with CL strategy.
the robustness of the RALM within our training
pipeline under a setting with added noise.
Open-domain QA We considered four open-
domain QA datasets: single-hop question datasets
Natural Questions (NQ) (Kwiatkowski et al.,
2019), TriviaQA (Joshi et al., 2017), and PopQA
(Mallen et al., 2023), as well as the multi-hop ques-
tion dataset HotpotQA (Yang et al., 2018). For
each dataset, we employed Contriever-MSMARCO
(Izacard et al., 2021) as the retriever. Following pre-
vious settings (Asai et al., 2024), for NQ, TriviaQA,
and HotpotQA, we retrieved the corresponding doc-
uments from the 2018 Wikipedia corpus provided
by Izacard et al. (2023). For PopQA, we retrieved
the documents from the 2020 Wikipedia corpus.
Noting that PopQA only contains a test set, which
we split into two non-overlapping parts for training
and testing, respectively. We utilized datasets from
the KILT benchmark (Petroni et al., 2021), more
detailed information about the datasets is provided
in Appendix A.
Robustness Test To evaluate the robustness of
the RALM in our framework, we artificially in-
troduced irrelevant and counterfactual documents
into the retrieved document sets. Specifically, for
each question and its retrieved documents in the
test set, we generated test data with irrelevant noise
by randomly replacing one of the retrieved doc-
uments with a document retrieved for a different
question. Additionally, we created test data with
counterfactual noise by randomly rewriting one of
the retrieved documents in a counterfactual manner.
The test sets obtained through these two methodswill be used separately to assess the model’s robust-
ness against irrelevant and counterfactual noise.
4.1.2 Evaluation Metrics
We employ Exact Match (EM) and F1 score as eval-
uation metrics. Specifically, EM assesses whether
the model-generated answers are identical to the
correct answers. Meanwhile, the F1 score inte-
grates Precision and Recall to measure the accuracy
and coverage of the model in generating answers.
4.1.3 Baselines
We utilized several training methods for RALM
and Retriever as baselines, and firstly evaluated
the performance of all the RALM training meth-
ods: (1) RAAT : Fang et al. (2024) enhances the
robustness of RALMs through adversarial training.
(2)RALM top5: The most basic approach in RALM
training by directly employing the top-5retrieved
documents. (3) RALM golden : Lin et al. (2023)
added the golden document to the retrieved docu-
ments to train the RALM. (4) RetRobust : Yoran
et al. (2023) randomly injected irrelevant and coun-
terfactual documents into the retrieved documents
to expose the model to diverse types of noise during
training.
Then, we continued to assess the performance
of retrievers on the best-performing RALM:
(1)Replug : Shi et al. (2023) minimized the KL
divergence between the score of retriever’s and the
model’s preference distribution to train a retriever
that better aligns with the LLM. (2) RA-DIT : Lin
et al. (2023) combined the method of finetuning
the retriever using KL divergence and finetuning
6

MethodIrrelevant Counterfactual
EM F1 EM F1
Llama 38B 44.10 54.79 44.32 54.55
RAAT 39.17 45.38 39.27 45.65
RALM golden 49.36 56.07 48.25 54.73
RALM top5 53.48 61.05 53.63 61.13
RetRobust 53.85 61.43 54.16 61.77
RALM CL 56.17 63.42 56.21 63.47
Table 2: Experimental results on robustness test(%)
over irrelevant and counterfactual documents. We
report the average results on each dataset here, and the
complete results are provided in Appendix C.
the RALM with retrieved documents that include
golden documents as context. (3) LLM-Embedder :
Zhang et al. (2024) defines the quality of document
dbased on how much it improves the correct an-
swer’s ranking in LLM’s response and proposes a
fine-grained hierarchical loss function for training.
4.1.4 Implementation Details
RLAM Training We trained our model based on
LLaMA3-8B-Instruct (Dubey et al., 2024), using
five retrieved documents. We randomly selected
1500 samples from the training sets of four datasets,
totaling 6000 samples, and conducted LoRA fine-
tuning based on LLaMA Factory (Zheng et al.,
2024), with the LoRA rank set to 8, a learning
rate of 1e-4, a gradient accumulation step of 8, and
a warmup ratio of 0.1. All experiments were con-
ducted on a A800 80G GPU card.
Retriever Training We trained our model based
on Contriever-MSMARCO (Izacard et al., 2021).
By randomly sampling from the training sets of
four datasets, we constructed a retriever training set
comprising 100,000 samples. We set the number of
input retrieved documents kto 5, the learning rate
of 1e-5, the number of epochs to 3, and the batch
size to 64. For our three-stage curriculum learning
settings, the number of reranked documents nis set
to 20. n1= [1,3,5]for each training epoch and n2
is set consistently to 15. The sampling parameters
for each stage are k1= [1,3,5],k2= [2,2,0], and
k3= [2,0,0].
4.2 Main Results
The overall results are shown in Table 1, where the
underlined andboldface items indicate the best re-
sults under each setting. We notify that the training
of all methods adhered to a unified model and set-
tings, and were tested under a 0-shot setting, with
the prompts provided in the Appendix B.With RALM Training The results in the up-
per half of Table 1 demonstrate that our training
method achieves better performance than previ-
ous methods across all test sets, thereby validat-
ing the effectiveness of our approach. It is noted
that the model trained with the RAAT method
performs worse than the baseline model. This
is because its training data follows the setting of
{dgolden⊕dnoise}. While this setting trains the
model to be robust against noise, the inclusion of
golden documents in all data deviates from the
conventional RAG setup, resulting in poorer perfor-
mance under the standard setting. Training using
documents that contain a golden document can
enhance the model’s basic performance, i.e., its
ability to extract correct answers from documents.
However, due to the limitations of the retriever,
the retrieved documents may not always contain
the necessary information to answer the question.
Therefore, training solely for answer extraction is
insufficient. It is also necessary to further train
the model to infer answers from more complex
documents. RetRobust considers a more complex
setting by accounting for both irrelevant document
noise and counterfactual document noise. However,
it neglects the incorporation of golden documents,
which are beneficial in the early stages of model
training. In contrast, our proposed method consid-
ers all types of document combinations, thereby
enabling the model to adapt to the diverse docu-
ments generated by retrieval. Therefore, it achieves
the best performance, with improvements of 2.18%
in the average EM score and 1.85% in the average
F1 score.
With Retriever Training In the settings that
incorporate retriever training, our method also
achieves the best average performance. As shown
in the lower half of Table 1, RA-DIT utilizes
the model from its experimental setting, namely
RALM top5, while the other methods all employ
the best-performing RALM CL. We found that on
datasets where LLM itself performs well, such as
TriviaQA, the gains from further improving the
retriever’s performance are minimal. However,
for datasets like PopQA, which involve questions
about specific entities, enhancing the retriever en-
ables it to more accurately fetch entity-related doc-
uments, resulting in a significant boost in perfor-
mance.
7

MethodAvg.
R@1 R@5 R@10
Baseline 44.35 67.40 74.26
Replug 49.10 72.32 78.43
w/o RALM 43.82 67.25 74.25
LLM -Embedder 49.45 71.48 77.78
w/o RALM 47.78 69.78 76.29
CL-RAG 50.40 73.29 79.55
w/o RALM 48.55 70.71 77.33
Table 3: Experimental results on Recall rates(%) for
retrievers trained with different methods. "Baseline"
refers to the retrieval results of the raw retriever and
"w/o RALM" indicates using the raw LLM without
employing the well-trained RALM. Detailed results for
each dataset are provided in Appendix D.
4.3 Robustness Test
To demonstrate the superiority of our RALM in
combating noise, we also conducted tests under
document settings that included noise. The re-
sults are shown in Table 2. Under the influence
of document noise, the performance of all models
declined to some extent. The impact of irrelevant
document noise was comparable to that of counter-
factual noise. The experimental results show that
our model maintained its leading performance even
when different types of noise were added.
4.4 Further Study of Retriever Training
To demonstrate that using the trained RALM to
guide the finetuning of the retriever is superior,
we compared the retrieval results of the retriever
finetuned with the raw LLM preferences to those
finetuned with the preferences of RALM. We re-
ported the recall rate of correct answers in the re-
trieved documents in Table 3. The experimental
results show that finetuning with the preferences
of RALM generally outperforms finetuning with
the raw LLM preferences. We observed that Re-
plug, which assesses document quality based on
answer probability, led to a significant drop in re-
call rate. This is because when the LLM itself
is capable of providing the correct answer, using
answer probability as a criterion to judge the qual-
ity of documents becomes unstable. In contrast,
the trained RALM has a higher ability to discern
documents and can more accurately differentiate
between good and bad ones. The experiments on fi-
nal answer generation are provided in Appendix D.
It is worth emphasizing that even when the same
trained RALM is used for guidance, our methodstage 3 stage 2 stage 154565860
w/o CLw/o CLEM(%)
With RALM Training
With Retriever Training
Figure 4: Ablation studies for our CL-RAG frame-
work(%). We report the average EM scores and the
complete results are provided in Appendix E.
still maintains a leading position.
4.5 Ablation Study
To systematically elucidate the contribution of each
training stage to the overall performance, we con-
ducted an ablation study. Specifically, we evaluated
the intermediate models generated at each stage of
our CL-RAG framework in a stage-by-stage man-
ner. This approach allows us to more explicitly
discern the contribution of each training stage to
the overall performance. The experimental results
are presented in Figure 4, where stage 1,stage 2,
andstage 3refer to the training phases involving
simple, common, and hard samples, respectively.
We observed that the performance of both the re-
triever and the generator steadily improves as the
training stages progress. Furthermore, the removal
of the curriculum learning strategy resulted in a
significant performance decline, indicating that ne-
glecting the difficulty order of training samples
leads to sub-optimal training outcomes.
5 Conclusion
In this study, we introduce CL-RAG, a curriculum
learning-based training framework designed for the
RAG system. To the best of our knowledge, it’s one
of the first time that the idea of human imitation
learning is integrated with RAG training, which can
effectively enhance the generalization and stability
of the RAG system. Experiments on four open-
domain question answering datasets provide sub-
stantial evidence of the framework’s effectiveness.
Additionally, separate experiments conducted on
the retriever and the generator have demonstrated
the significant enhancements our method brings to
each individual part.
8

Limitations
Despite its effectiveness, CL-RAG still has cer-
tain limitations. First, the LLM preference mea-
surement based on the probability or ranking of
decoding correct answers may have advantages
in open-domain question answering data, but it
may be unstable in long text generation. Second,
our framework only involves a single iteration of
training between the RALM and the retriever, that
is, training the RALM first and then the retriever.
More refined iterative methods, such as iteration
after each difficulty level training stage, still need
further exploration.
References
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin John-
son, Dmitry Lepikhin, Alexandre Passos, Siamak
Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng
Chen, et al. 2023. Palm 2 technical report. arXiv
preprint arXiv:2305.10403 .
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations .
Yoshua Bengio, Jérôme Louradour, Ronan Collobert,
and Jason Weston. 2009. Curriculum learning. In
Proceedings of the 26th Annual International Confer-
ence on Machine Learning , ICML ’09, page 41–48,
New York, NY , USA. Association for Computing
Machinery.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.
Improving language models by retrieving from tril-
lions of tokens. In International conference on ma-
chine learning , pages 2206–2240. PMLR.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners. Advances in neural information processing
systems , 33:1877–1901.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 .
Jeffrey L. Elman. 1993. Learning and development in
neural networks: the importance of starting small.
Cognition , 48(1):71–99.Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiao-
jun Chen, and Ruifeng Xu. 2024. Enhancing noise
robustness of retrieval-augmented language models
with adaptive adversarial training. In Annual Meeting
of the Association for Computational Linguistics .
Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eun-
sol Choi, and Danqi Chen. 2019. MRQA 2019 shared
task: Evaluating generalization in reading compre-
hension. In Proceedings of 2nd Machine Reading
for Reading Comprehension (MRQA) Workshop at
EMNLP .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. ArXiv , abs/2312.10997.
Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan
Zhuang, Dengke Dong, Matthew R Scott, and Din-
glong Huang. 2018. Curriculumnet: Weakly super-
vised learning from large-scale web images. In Pro-
ceedings of the European conference on computer
vision (ECCV) , pages 135–150.
Guy Hacohen and Daphna Weinshall. 2019. On the
power of curriculum learning in training deep net-
works. In International conference on machine learn-
ing, pages 2535–2544. PMLR.
Xingwei He, Yeyun Gong, A-Long Jin, Hang Zhang,
Anlei Dong, Jian Jiao, Siu Yiu, and Nan Duan. 2023.
Capstone: Curriculum sampling for dense retrieval
with document expansion. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 10531–10541.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2021. Unsupervised dense informa-
tion retrieval with contrastive learning. Transactions
on Machine Learning Research .
Gautier Izacard and Edouard Grave. 2020. Leverag-
ing passage retrieval with generative models for
open domain question answering. arXiv preprint
arXiv:2007.01282 .
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine
Learning Research , 24(251):1–43.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric
Wallace, and Colin Raffel. 2023. Large language
9

models struggle to learn long-tail knowledge. In In-
ternational Conference on Machine Learning , pages
15696–15707. PMLR.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:453–466.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Richard James, Pedro Rodriguez, Ja-
cob Kahn, Gergely Szilvasy, Mike Lewis, et al. 2023.
Ra-dit: Retrieval-augmented dual instruction tuning.
InThe Twelfth International Conference on Learning
Representations .
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
Vassilis Plachouras, Tim Rocktäschel, and Sebastian
Riedel. 2021. KILT: a benchmark for knowledge
intensive language tasks. In Proceedings of the 2021
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies , pages 2523–2544, Online.
Association for Computational Linguistics.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Adam Roberts, Colin Raffel, and Noam Shazeer. 2020.
How much knowledge can you pack into the param-
eters of a language model? In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 5418–5426.
Douglas L.T Rohde and David C Plaut. 1999. Lan-
guage acquisition in the absence of explicit negative
evidence: how important is starting small? Cogni-
tion, 72(1):67–109.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2023. Replug: Retrieval-
augmented black-box language models. arXiv
preprint arXiv:2301.12652 .
Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan
Wang, Hongtao Xie, and Yongdong Zhang. 2020.Curriculum learning for natural language understand-
ing. In Proceedings of the 58th Annual Meeting of
the Association for Computational Linguistics , pages
6095–6104.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2023. Making retrieval-augmented language
models robust to irrelevant context. In The Twelfth
International Conference on Learning Representa-
tions .
Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022.
Curriculum learning for dense retrieval distillation.
InProceedings of the 45th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval , pages 1979–1983.
Peitian Zhang, Zheng Liu, Shitao Xiao, Zhicheng Dou,
and Jian-Yun Nie. 2024. A multi-task embedder for
retrieval augmented llms. In Proceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 3537–
3553.
Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan
Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma.
2024. Llamafactory: Unified efficient fine-tuning
of 100+ language models. In Proceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 3: System Demonstra-
tions) , Bangkok, Thailand. Association for Computa-
tional Linguistics.
Yutao Zhu, Jianyun Nie, Yixuan Su, Haonan Chen,
Xinyu Zhang, and Zhicheng Dou. 2022. From easy
to hard: A dual curriculum learning framework for
context-aware document ranking. Proceedings of the
31st ACM International Conference on Information
& Knowledge Management .
A Datasets Details
Table 4 shows details of the datasets we used,
where the data volumes for RALM and Retriever
represent the amounts used for training each respec-
tive component.
B Inference Prompt
During the inference stage of all models, we used
the same prompt to ensure fairness. As shown in
Table 5, paragraph represents the retrieved docu-
ment, and instruction represents the query. Finally,
we extracted the content following "Answer:" in
the model’s response as the final answer.
10

DatasetTrain Test
Total RALM Retriever Total
NQ 87372 1500 30000 2837
TriviaQA 61844 1500 30000 5359
PopQA 10000 1500 10000 4267
HotpotQA 88869 1500 30000 5600
Table 4: Details of the datasets we used.
Prompt
System Prompt: Answer the following questions with two to three words. Your answer must be
formatted as follows: Answer: <your answer>
User Prompt: The following contexts will help you answer the question.
{paragraph}
Question: {instruction}
Table 5: Prompt for LLM inference.
C Complete Results of Robustness Test
Complete results are shown in Table 6.
D Further Study of Retriever Training
D.1 Complete Results of Retriever Training
Complete document retrieval results are shown
in Table 7 and Table 8. The final results on QA
datasets are provided in Table 9.
D.2 Case Study of Documents Evaluation
Table 10 presents the case study comparing the
document evaluation result of the base LLM and
the RALM. The result indicate that for some docu-
ments containing implicit knowledge required for
answering questions, the base LLM may generate
erroneous judgments, whereas the trained RALM
is more capable of distinguishing the quality of
documents. This also highlights the necessity of
the iterative process of training the RALM first and
then training the retriever, as the trained RALM
and the base LLM may no longer share the same
preferences for documents. Separate training may
lead to suboptimal final outcomes.
E Complete Results of Ablation Study
Complete results of the ablation study are shown
in Table 11.
11

MethodNQ TriviaQA PopQA HotpotQA Avg.
EM F1 EM F1 EM F1 EM F1 EM F1
With irrelevant documents
Llama 38B 38.56 51.72 71.28 80.61 42.68 48.85 23.89 37.99 44.10 54.79
RAAT 30.31 38.65 69.34 74.61 35.90 39.29 21.13 28.96 39.17 45.38
RALM golden 41.49 50.92 78.52 83.10 46.97 49.44 30.46 40.80 49.36 56.07
RALM top5 46.32 57.35 78.66 83.52 53.60 56.33 35.32 47.01 53.48 61.05
RetRobust 46.56 57.17 79.57 84.42 54.28 57.17 35.00 46.95 53.85 61.43
RALM CL 50.30 59.89 81.48 86.31 56.13 59.05 36.75 48.43 56.17 63.42
With counterfactual documents
Llama 38B 38.10 50.78 72.53 80.93 42.49 48.52 24.16 37.95 44.32 54.55
RAAT 30.60 39.16 70.23 75.40 34.92 38.59 21.32 29.43 39.27 45.65
RALM golden 40.08 49.03 77.52 81.90 45.58 48.07 29.82 39.92 48.25 54.73
RALM top5 46.18 56.88 79.10 83.88 53.88 56.66 35.34 47.10 53.63 61.13
RetRobust 46.56 57.47 80.39 85.13 53.90 57.04 35.80 47.42 54.16 61.77
RALM CL 50.16 59.82 81.74 86.55 55.89 58.68 37.04 48.82 56.21 63.47
Table 6: Complete results on robustness test(%).
MethodNQ TriviaQA
R@1 R@5 R@10 R@1 R@5 R@10
Baseline 43.64 72.75 81.00 65.56 88.25 93.30
Replug 48.43 75.54 83.22 66.50 88.59 93.13
w/o RALM 45.29 73.60 81.32 56.13 80.26 86.29
LLM -Embedder 46.60 73.56 82.16 68.20 88.68 92.47
w/o RALM 44.77 72.01 80.12 67.41 88.27 92.90
CL−RAG 48.82 75.33 83.33 68.59 89.86 93.91
w/o RALM 46.11 72.26 80.61 69.26 89.00 93.22
Table 7: Retrieve results of Natural Questions and TriviaQA.
MethodPopQA HotpotQA
R@1 R@5 R@10 R@1 R@5 R@10
Baseline 43.03 66.51 73.56 25.16 42.07 49.14
Replug 54.79 80.50 86.15 26.68 44.63 51.23
w/o RALM 50.55 76.49 83.90 23.30 38.64 45.48
LLM -Embedder 54.68 79.17 85.66 28.30 44.52 50.82
w/o RALM 51.54 75.72 82.05 27.41 43.13 50.07
CL−RAG 55.26 81.53 87.18 28.91 46.45 53.79
w/o RALM 51.84 78.04 84.77 26.98 43.55 50.73
Table 8: Retrieve results of PopQA and HotpotQA.
MethodNQ TriviaQA PopQA HotpotQA Avg.
EM F1 EM F1 EM F1 EM F1 EM F1
LLM -Emebedder 52.10 61.53 83.03 87.72 65.24 67.88 38.86 50.91 59.81 67.01
Replug 52.73 62.42 83.17 87.67 66.16 68.75 38.73 50.33 60.20 67.29
CL-RAG 53.01 62.51 82.97 87.57 66.63 69.12 39.11 51.19 60.43 67.60
Table 9: Final results on QA datasets when all retrievers are trained to align with the preferences of RALM CL.(%)
12

Question : Who scored a film based on a 1961 science fiction novel by Stanislaw Lem?
Answer : Cliff Martinez
Retrieved Document
Stanisław Lem
(Poland, Germany, and the Soviet Union). Franz Rottensteiner, Lem’s former agent abroad, had
this to say about Lem’s reception on international markets: His best-known novels include "Solaris"
(1961) , "His Master’s V oice" ("Głos pana", 1968), and the late "Fiasco" ("Fiasko", 1987). "Solaris"
was made into a film in 1968 by Russian director Boris Nirenburg, a film in 1972 by Russian director
Andrei Tarkovsky—which won a Special Jury Prize at the Cannes Film Festival in 1972—and
an American film in 2002 by Steven Soderbergh. "Solaris" is not the only work of Lem’s
to be filmed.
Raw LLM preference
Answer probably: 0.002
Answer rank: 1417
Well-trained RALM preference
Answer probably: 0.73
Answer rank: 1
Table 10: An example of document evaluation.
MethodNQ TriviaQA PopQA HotpotQA Avg.
EM F1 EM F1 EM F1 EM F1 EM F1
With RALM Training
w/o CL 49.31 59.37 80.34 85.38 57.65 60.47 35.11 46.94 55.60 63.04
stage 1 46.32 56.26 81.77 85.99 53.29 55.79 32.77 43.79 53.54 60.46
stage 2 51.22 60.73 83.12 87.50 58.54 61.52 37.36 49.13 57.56 64.72
RALM CL 51.53 61.19 82.93 87.59 59.71 62.28 38.14 50.22 58.08 65.32
With Retriever Training
w/o CL 52.13 61.42 82.48 87.11 64.26 66.70 38.20 50.13 59.27 66.34
stage 1 52.66 62.11 82.96 87.67 63.67 66.14 38.94 50.89 59.56 66.70
stage 2 52.98 62.39 83.04 87.49 65.10 67.53 39.05 51.26 60.04 67.17
CL-RAG 53.01 62.51 82.97 87.57 66.63 69.12 39.11 51.19 60.43 67.60
Table 11: Complete results of ablation study(%).
13
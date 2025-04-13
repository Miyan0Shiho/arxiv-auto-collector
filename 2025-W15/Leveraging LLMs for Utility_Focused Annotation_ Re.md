# Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng

**Published**: 2025-04-07 16:05:52

**PDF URL**: [http://arxiv.org/pdf/2504.05220v2](http://arxiv.org/pdf/2504.05220v2)

## Abstract
Retrieval models typically rely on costly human-labeled query-document
relevance annotations for training and evaluation. To reduce this cost and
leverage the potential of Large Language Models (LLMs) in relevance judgments,
we aim to explore whether LLM-generated annotations can effectively replace
human annotations in training retrieval models. Retrieval usually emphasizes
relevance, which indicates "topic-relatedness" of a document to a query, while
in RAG, the value of a document (or utility) depends on how it contributes to
answer generation. Recognizing this mismatch, some researchers use LLM
performance on downstream tasks with documents as labels, but this approach
requires manual answers for specific tasks, leading to high costs and limited
generalization. In another line of work, prompting LLMs to select useful
documents as RAG references eliminates the need for human annotation and is not
task-specific. If we leverage LLMs' utility judgments to annotate retrieval
data, we may retain cross-task generalization without human annotation in
large-scale corpora. Therefore, we investigate utility-focused annotation via
LLMs for large-scale retriever training data across both in-domain and
out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of
low-quality positives labeled by LLMs, we design a novel loss function, i.e.,
Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on
utility-focused annotations significantly outperform those trained on human
annotations in the out-of-domain setting on both tasks, demonstrating superior
generalization capabilities. (2) LLM annotation does not replace human
annotation in the in-domain setting. However, incorporating just 20%
human-annotated data enables retrievers trained with utility-focused
annotations to match the performance of models trained entirely with human
annotations.

## Full Text


<!-- PDF content starts -->

Leveraging LLMs for Utility-Focused Annotation: Reducing
Manual Effort for Retrieval and RAG
Hengran Zhang*
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
zhanghengran22z@ict.ac.cnMinghao Tang*
Nankai University
Tianjin, China
tangminghao@mail.nankai.edu.cnKeping Bi, Jiafeng Guo
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
bikeping@ict.ac.cn
guojiafeng@ict.ac.cn
Shihao Liu, Daiting Shi
Baidu Inc
Beijing, China
liushihao02@baidu.com
shidaiting01@baidu.comDawei Yin
Baidu Inc
Beijing, China
yindawei@acm.orgXueqi Cheng
CAS Key Lab of Network Data
Science and Technology, ICT, CAS
University of Chinese Academy of
Sciences
Beijing, China
cxq@ict.ac.cn
Abstract
Retrieval models typically rely on costly human-labeled query-
document relevance annotations for training and evaluation. To
reduce this cost and leverage the potential of Large Language Mod-
els (LLMs) in relevance judgments, we aim to explore whether
LLM-generated annotations can effectively replace human anno-
tations in training retrieval models. Retrieval usually emphasizes
relevance, which indicates “topic-relatedness” of a document to a
query, while in RAG, the value of a document (or utility), depends
on how it contributes to answer generation. Recognizing this mis-
match, some researchers use LLM performance on downstream
tasks with documents as labels, but this approach requires man-
ual answers for specific tasks, leading to high costs and limited
generalization. In another line of work, prompting LLMs to select
useful documents as RAG references eliminates the need for human
annotation and is not task-specific. If we leverage LLMs’ utility
judgments to annotate retrieval data, we may retain cross-task
generalization without human annotation in large-scale corpora.
Therefore, we investigate utility-focused annotation via LLMs
for large-scale retriever training data across both in-domain and
out-of-domain settings on the retrieval and RAG tasks. To reduce
the impact of low-quality positives labeled by LLMs, we design a
novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that:
(1) Retrievers trained on utility-focused annotations significantly
*Contributed equally.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06
https://doi.org/XXXXXXX.XXXXXXXoutperform those trained on human annotations in the out-of-do-
main setting on both tasks, demonstrating superior generalization
capabilities. (2) LLM annotation does not replace human annota-
tion in the in-domain setting. However, incorporating just 20% hu-
man-annotated data enables retrievers trained with utility-focused
annotations to match the performance of models trained entirely
with human annotations, while adding 100% human annotations fur-
ther significantly enhances performance on both tasks. We hope our
work inspires others to design automated annotation solutions us-
ing LLMs, especially when human annotations are unavailable. The
code and models are available on https://github.com/Trustworthy-
Information-Access/utility-focused-annotation.
CCS Concepts
•Information systems →Language models ;Novelty in infor-
mation retrieval .
Keywords
First-stage retrieval, utility, retrieval-augmented generation
ACM Reference Format:
Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Dait-
ing Shi, Dawei Yin, and Xueqi Cheng. 2018. Leveraging LLMs for Utility-
Focused Annotation: Reducing Manual Effort for Retrieval and RAG. In
Proceedings of Make sure to enter the correct conference title from your rights
confirmation email (Conference acronym ’XX). ACM, New York, NY, USA,
12 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Information retrieval (IR) has long been a critical method for in-
formation seeking, and retrieval-augmented generation (RAG) is
increasingly recognized as a key strategy for reducing hallucina-
tions in large language models (LLMs) in the modern landscape of
information access [ 48,58,77]. Typically, retrieval models rely onarXiv:2504.05220v2  [cs.IR]  8 Apr 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, and Xueqi Cheng
human annotations of query-document relevance to train and eval-
uate. Given the high cost of human annotation and the promising
potential of LLMs for relevance judgments [ 41], we aim to explore
whether LLM-generated annotations can effectively replace human
annotations in training models for retrieval and RAG. This is espe-
cially important when question-answering (QA) systems are built
with a reference corpus that has no annotation to train a retrieval
model, and the service provider has limited budgets.
Retrieval usually emphasizes relevance, which indicates “about-
ness”, “pertinence”, or “topic-relatedness” of a document to a query
[54], while in RAG, the value of a document (or utility), depends
on how it contributes to answer generation. There is an evident
gap between these two. In other words, a relevant document from
a retriever is not necessarily of utility (or useful) for RAG. Utility
has also been proposed as an important counterpart measure of
relevance by IR researchers several decades ago [ 7,55]. It refers to
the usefulness of a retrieval item to an information seeker, its value,
appropriateness in resolution of a problem, etc. [ 53–56]. Relevance
and utility characterize the goal of target documents for retrieval
and RAG well respectively.
Aware of this mismatch between the retrieval objective of stan-
dard retrieval and RAG, researchers have resorted to LLM perfor-
mance on downstream tasks given a document as its label [ 17,22,30,
35,57,76], e.g., the likelihood of the ground-truth answers [ 57] or
exact match (EM) between the generated answer and ground-truth
answer [ 76]. The other thread of work prompts LLMs to select
documents with utility from the input as the final reference for
RAG [ 79,80]. Studies from both paths have shown enhanced RAG
performance.
Despite their effectiveness, they have notable limitations. Specifi-
cally, downstream task performance requires manual-labeled ground-
truth answers to evaluate, which still incurs huge manual annota-
tion costs. Moreover, the retriever trained with a specific task has
difficulty generalizing on other downstream tasks or even other
evaluation metrics of the same task. When the questions are non-
factoid, precise evaluation itself is challenging, limiting its use as
training objectives for retrieval. In contrast, the other approach, i.e.,
leveraging LLMs to select useful documents [ 79,80], does not need
human annotation and is not limited to specific tasks and metrics.
However, it cannot scale to the entire corpus due to the prohibitive
inference cost.
If we leverage LLMs’ capability of utility judgments for anno-
tating training data to learn retrieval models, we may retain the
advantages of generalization on various tasks without human anno-
tation in the large-scale corpus. So, in this paper, we leverage LLMs
with utility-focused annotation to train effective retrievers for re-
trieval and RAG. In concrete, we study several groups of research
questions: ( RQ1 )Can LLM-annotated data replace human-annotated
data for retrieval and RAG and to what extent human annotation
can be saved? (RQ2 )How do retrievers trained with LLM annota-
tions generalize under the in-domain (performance on MS MARCO
dev) and out-of-domain(performance on BEIR benchmarks) settings?
(RQ3 ) Regarding training effective models with LLM annotations
for retrieval and downstream tasks: Will utility-focused annotation
produce better retrieval and RAG performance? andWhat training
objectives are effective for LLM-annotated data?
Our empirical work leads to the following interesting results:ForRQ1 , the answer is PARTIAL. Our experimental results in-
dicate that retrievers trained with different LLM-generated anno-
tations perform slightly worse than those trained with human an-
notations. We further explore the integration of LLM-annotated
and human-annotated data using the curriculum learning, which
is first trained on weak supervision generated by LLMs and then
trained on high-quality labels generated by humans. Our findings
show that incorporating 20% human-annotated data in curriculum
learning allows models trained with utility-focused annotation to
achieve performance comparable to those trained exclusively with
human annotations. Additionally, when 100% human annotations
are used in curriculum learning, the resulting models significantly
outperform those trained solely with human annotations.
ForRQ2 , considering the in-domain setting and out-of-domain
setting, there are different findings. Although the retriever trained
on human-annotated labels has better performance on both tasks
compared to LLMs annotated labels in the in-domain setting, the re-
triever trained with utility-focused annotations significantly outper-
forms those trained with human annotations in the out-of-domain
setting on both tasks, suggesting that LLM-generated annotations
offer better generalization capabilities.
ForRQ3 : For the first question, the answer is YES. Experiments
show that retrievers trained on labels from relevance selection
perform poorly. Building upon relevance selection by applying
utility selection or ranking further improves the retriever’s perfor-
mance. Retrievers trained with utility-focused annotations have
better performance on retrieval and RAG tasks than those using
the performance on downstream tasks given a document as its
label. For the second question, LLMs typically generate multiple
positive instances for each query, which, compared to human an-
notations, can be seen as weak supervision and may lead to false
or low-quality positive instances. To address this, we propose a
novel loss function, i.e., Disj-InfoNCE, that aggregates all positive
instances for each query during optimization, reducing the impact
of low-quality positives.
We summarize our contributions as follows:
•We provide a large LLM-annotated dataset suitable for training
retrieval models on nearly 500K queries.
•We propose a comprehensive solution for data annotation using
LLMs in first-stage retrieval, along with corresponding training
strategies.
•Our approach achieves strong performance in both retrieval and
generation tasks without relying on human annotations, demon-
strating excellent generalization. Additionally, when combined
with human annotations using the curriculum learning method,
our method outperforms human-only annotations in both re-
trieval and generation tasks.
We hope that our work can inspire others to design automated
annotation solutions using LLMs, especially in scenarios where
human annotations are unavailable.
2 Related Work
In this section, we briefly introduce first-stage retrieval, utility-
focused retrieval-augmented generation (RAG), and automatic an-
notation using LLMs.

Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
2.1 First-Stage Retrieval
Modern search systems utilize a multi-stage ranking pipeline to
balance efficiency and effectiveness, starting with a first-stage re-
trieval, followed by multiple re-ranking stages to refine the results
[23]. We mainly focus on the first-stage retrieval, which aims to
retrieve all potentially relevant documents from the whole collec-
tion that contains documents on a million scale or even higher.
To achieve millisecond-level latency for querying the corpus [ 18],
first-stage retrieval indexes the entire corpus offline and then per-
forms retrieval using the approximate nearest neighbor (ANN) [ 32]
search method. Initially, the first-stage retrieval models were pre-
dominantly classical term-based models, such as BM25 [ 50], which
combines term matching with TF-IDF weighting. Subsequently,
large-scale pre-trained language models (PLMs) like BERT [ 11]
have been widely applied to various NLP tasks [ 5,20,70], including
first stage retrieval [ 33,37]. PLM-based retrievers have been exten-
sively explored, including the design of pre-training tasks tailored
for retrieval [ 29,38,67,71], mining dynamic hard negative samples
for the retriever [ 15,46,72,78], and the introduction of rankers for
knowledge distillation training [49, 71].
2.2 Utility-Focused RAG
Retrieval-augmented generation (RAG), amalgamating an informa-
tion retrieval component with a text generator model, is commonly
used to mitigate the issues of hallucination and knowledge obso-
lescence in LLMs [ 24,35,45]. However, the goals of the retriever
(retrieving more relevant information) and generator (extracting
useful information to produce precise and coherent responses)
in RAG are different and can be mismatched. To address this is-
sue, current research focuses mainly on two approaches: (1) Util-
ity judgments, which directly entails utilizing LLMs to identify
useful retrieved information based on its utility for downstream
tasks [ 79,80,82]. Utility judgments typically serve as post-pro-
cessing steps for retrieval results and do not directly influence
the retriever. (2) Utility-optimized retriever, which involves trans-
ferring the capability of LLMs to evaluate the utility of retrieved
information to the retriever. Specifically, two primary optimization
functions are commonly employed: (a) calculating the likelihood of
the ground truth answers given the query and retrieval information
[2,16,22,27,30,35,52,57,74]; (b) directly using evaluation met-
rics of the downstream generation tasks [ 17,66,76], such as exact
match (EM), and ROUGE [ 36], and computing the performance dif-
ference between the generated answer and the ground truth answer.
However, this approach relies on ground truth answers for specific
downstream tasks and limits generalization.
2.3 Automatic Annotation with LLMs
Large language models (LLMs) demonstrate strong general capa-
bilities and are increasingly utilized to annotate a wide range of
tasks, such as named entity recognition [ 69], sentiment analysis
[51], and recommendation systems [ 1]. Wang et al . [68] are among
the early users of LLMs for data annotation in classification and
natural language generation tasks, and their findings show that
LLM-based annotation can considerably reduce annotation costs.
In the field of information retrieval, many studies [ 42,47,59,61,79]have also explored the annotation capabilities of LLMs. For ex-
ample, Thomas et al . [61] examined how LLMs can be leveraged
for relevance judgments, with their results suggesting that LLMs
can perform at levels comparable to human annotators in finding
the best systems. However, these studies predominantly focus on
the construction of evaluation datasets to assess retrieval perfor-
mance, lacking a comprehensive investigation into the annotation
capabilities of LLMs for training datasets in retrieval-related tasks.
3 Preliminary
In this section, we will briefly introduce typical dense retrieval
models and how to use downstream performance as utility label.
3.1 Typical Dense Retrieval Models
Dense retrieval models primarily employ a two-tower architec-
ture of pre-trained language models (PLMs), i.e., R𝑞(·)andR𝑑(·),
to encode query and passage, into fix-length dense vectors. The
relevance between the query 𝑞and passage 𝑑is𝑠(𝑞,𝑑), i.e.,
𝑠(𝑞,𝑑)=𝑓<R𝑞(𝑞),R𝑑(𝑑)>, (1)
where𝑓<·>is usually implemented as a simple metric, e.g., dot
product and cosine similarity. R𝑞(·)andR𝑑(·)are usually share
the parameters. The traditional way for training dense retrievers
uses contrastive loss, also referred to as InfoNCE [43] loss, i.e.,
L𝑠(𝑞,𝑑+,𝐷−)=−logexp(𝑠(𝑞,𝑑+))Í
𝑑∈{𝑑+,𝐷−}exp(𝑠(𝑞,𝑑)), (2)
where𝑑+and𝐷−represent the positive and negative instances for
the query𝑞.
3.2 Downstream Performance as Utility Label
Considering the downstream task for the retriever, i.e., RAG, the
goals of the retriever and generator in RAG are different and can
be mismatched. To alleviate this issue, the utility of retrieval in-
formation𝑓𝑢(𝑞,𝑑,𝑎), where𝑎is the ground truth answer, enables
the retriever to be more effectively alignment with the generator.
𝑓𝑢(𝑞,𝑑,𝑎)mainly has two ways: directly model how likely the can-
didate passages can generate the ground truth answer [ 57], i.e.,
𝑓𝐿𝐿𝑀(𝑎|𝑞,𝑑), which computes the likelihood of the ground truth
answer; and measure the divergence of model output 𝐿𝐿𝑀(𝑞,𝑑)
and the answer 𝑎using evaluation metrics [ 76], e.g., exact match
(EM), i.e.,𝐸𝑀(𝑎,𝐿𝐿𝑀(𝑞,𝑑)). Given the query 𝑞and candidate pas-
sage list𝐷=[𝑑1,𝑑2,...,𝑑𝑛], where𝑛=|𝐷|. The optimization of
the retriever is to minimize the KL divergence between the rel-
evance distribution 𝑅={𝑠′(𝑞,𝑑𝑖)}𝑁
𝑖=1, where𝑠′(𝑞,𝑑𝑖)is the rele-
vance𝑠(𝑞,𝑑𝑖)from retriever after softmax operation, and utility
distribution 𝑈={𝑓′𝑢(𝑞,𝑑𝑖,𝑎)}𝑁
𝑖=1, where𝑓′𝑢(·)is the utility function
𝑓𝑢(·)from generator after softmax operation:
𝐾𝐿(𝑈||𝑅)=𝑁∑︁
𝑖=1𝑈(𝑑𝑖)𝑙𝑜𝑔(𝑈(𝑑𝑖)
𝑅(𝑑𝑖)). (3)
4 Utility-Focused Annotation Using LLMs
Retrieval usually emphasizes relevance, which indicates “about-
ness”, “pertinence”, or “topic-relatedness” of a document to a query,
while in RAG, the value of a document (or utility), depends on how
it contributes to answer generation. Recognizing this mismatch,
researchers have resorted to LLM performance on downstream

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, and Xueqi Cheng
✅
❌Positive
Negative
QuestionQuestion: 𝑞
Ground Truth Answer: 𝑎
𝑓𝐿𝐿𝑀𝑎|𝑑𝑖,𝑞
Candidate Pool
…
Candidate Pool
…
Directly output thepassages youselected
thatarerelevant tothequestion .
Relevance
Selection
Directly output the passages you 
selected that have utility in generating 
the reference answer to the question. Reference answer generation ：
Answer the following question based 
on the given information.
Utility
Selection
Rank the 3 passages above based on 
their utility in generating the 
reference answer to the question.Reference answer generation: 
Answer the following question based 
on the given information.
>
Utility
Ranking
>𝑑𝑖
Candidate Pool
…
(a) (b) (c)
Figure 1: (a) Human annotation, (b) Using downstream performance as utility score, (c) Our utility-focused annotation pipeline.
0 10 20 300.0000.0250.0500.0750.1000.1250.150FrequencyLlama-3.1-8B
Relevance Selection
Utility Selection
0 10 20 300.000.050.100.150.200.250.30Qwen-2.5-32B-Int8
Relevance Selection
Utility Selection
Figure 2: Frequency distribution of different annotators at
various stages.
tasks given a document as its label to optimize the retriever rather
than the relevance labels generated by humans in Figure 1 (a).
As shown in Figure 1 (b), this method requires manual-labeled
ground-truth answers to evaluate, which still incurs huge manual
annotation costs. Moreover, the retriever trained with a specific task
has difficulty generalizing on other downstream tasks. In another
line of work, prompting LLMs to select useful documents as RAG
references eliminates the need for human annotation and isn’t task-
specific. Therefore, given the effectiveness of utility judgments via
LLMs [ 80], we analyze LLM on utility-focused annotation without
relying on ground truth answers. Zhang et al . [79] proposed that
iteratively applying the relevance-answer-utility can effectively im-
prove utility judgments performance, inspired by Schutz’s theory.
Therefore, we also introduce a relevance-then-utility pipeline in
our annotation, as shown in Figure 1 (c).
4.1 Annotation Pipeline
Annotation Pool Construction. We utilized the representative
retrieval dataset, MS MARCO (as show in section 5.1), for annota-
tion. The construction of an annotation pool was necessary for the
annotated queries. Since the dataset does not provide specific de-
tails on which passages constitute the manually annotated pooling,
the quality and quantity of passages within the annotation pool
could potentially affect the quality of the annotations. To mitigate
the influence of the annotation pool, all annotation methods were
applied within the same pool. Given that the training of current re-
trieval systems involves each query comprising the positive passage𝑑+and hard-negative passages {𝑑−
𝑖}𝑁
𝑖=1, we consider a combina-
tion of hard negatives generated by BM25 and CoCondenser [ 19],
to enhance the diversity of hard negative samples, which is the
same as Ma et al . [39] . We constructed an annotation pool by shuf-
fling and mixing positive and hard-negative passages, i.e., {𝑑𝑖}𝑁+1
𝑖=1.
The original labels of the dataset served as the results of human
annotation.
Annotation Details. Since annotation requires selecting positive
examples directly from the annotation pool for training, using rele-
vance ranking necessitates setting a threshold to determine positive
examples. Moreover, relevance ranking needs to rank all passages
from the entire annotation pool, which increases annotation costs
and potentially affects the quality of annotations. Therefore, we only
utilized relevance selection ( RelSel ), allowing the LLMs to directly
select passages relevant to the query from the candidate annotation
pool, instead of employing relevance ranking. The instruction of
relevance selection is “I will provide you with {K} passages, each
indicated by number identifier []. Select the passages that are rele-
vant to the question: {query}. ” . Due to the input limitation of LLMs,
relevance selection was employed for most 𝑚(𝑚=16) passages at
once as input. When annotating for utility, the number of passages
to be annotated was reduced. We explored both utility selection
(UtilSel ) and utility ranking ( UtilRank ), with the input of the
query, all relevance-selected passages, and pseudo answer 𝑎, which
is generated by LLMs based on the relevance selection results. The
instructions of pseudo answer generation and utility selection are
“Given the information: {all passages} Answer the following question
based on the given information with one or few sentences without
the source. ” ,‘’The requirements for judging whether a passage has
utility in answering the question are: The passage has utility in an-
swering the question, meaning that the passage not only be relevant
to the question, but also be useful in generating a correct, reasonable,
and perfect answer to the question. Directly output the passages you
selected that have utility in generating the reference answer to the
question. ” , respectively. We employ the top 𝑘% (𝑘=10) cutoff for util-
ity ranking as the final annotation, and more details on the different
thresholds are shown in Figure 4. The instruction of utility ranking
is“Rank the K passages above based on their utility in generating
the reference answer to the question. The passages should be listed
in utility descending order using identifiers. The passages that have
utility in generating the reference answer to the question should be

Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 1: Recall and precision performance (%) of human pos-
itive passage of different annotators. “RS”, “US”, “UR” means
“RelSel”, “UtilSel”, “UtilRank”, respectively.
LLMPrecision Recall Avg Number
RS US UR RS US UR RS US UR
Llama-3.1-8B 7.1 11.9 36.5 97.6 91.6 41.0 13.8 7.7 1.2
Qwen-2.5-32B-Int8 15.1 29.5 71.3 92.8 84.8 72.0 6.2 2.9 1.0
listed first. ” . All annotations utilize listwise input, and the overall
pipeline of utility-focused annotation is shown in Figure 1 (c).
4.2 Statistics of LLM Annotations
In this section, we conducted a detailed analysis of the annotated
dataset. We employ two of the latest and high-performing open-
source LLMs with different parameter counts for our annotation
task: (1) LlaMa-3.1-8B-Instruct [ 13] (Llama-3.1-8B), which is op-
timized for multilingual dialogue scenarios and surpasses many
existing open-source and proprietary chat models on standard in-
dustry benchmarks. (2) Qwen-2.5-32B-Instruct [ 73], which is the
latest iteration of the Qwen series of large language models. Due
to the extensive hardware resources required by the 32B model, we
employ its GPTQ-quantized [ 14] 8-bit version (Qwen-2.5-32B-Int8).
Selection Frequency Distribution. The number of positive in-
stances generated through automated annotation is crucial for the
training of a retriever. Figure 2 illustrates the number of annotations
at various stages by different LLMs (utility ranking with 𝑘%has an
average of one and no need to compute frequency): (1) LLMs with
larger scales tend to retrieve fewer positive instances, which could
alleviate the issue of false positives. (2) The number of positive
instances decreases progressively from the relevance selection to
the utility selection. This aligns with the transition from relevance
to utility, as utility indicates a high standard of relevance [79].
Annotation Evaluation. We evaluated the precision, recall, and
average number of annotations at each stage for both LLMs us-
ing human-annotated labels as a standard. From Table 1, we can
observe that: (1) Both LLMs exhibit commendable recall rates for
human-annotated positive passages. Llama-3.1-8B has a slightly
higher recall rate and higher average number, which is expected
given that it retrieves a larger quantity of annotations. (2) Preci-
sion for human-positive passages is lower in relevance selection
than in utility selection, suggesting a high rate of false positives
in relevance selection. Additionally, the lower average number in
utility selection compared to relevance selection typically results in
a lower recall rate for human-positive instances in utility selection.
4.3 Loss Function
The retriever is typically trained using InfoNCE loss [ 43], which
maximizes the probability of positive pairs overall negative pairs.
Retrieval dataset like MS MARCO only has one positive passage, so
InfoNCE loss typically employs one positive instance for each query.
For datasets annotated by LLMs, each query may have multiple
positive instances. The simplest approach is to randomly sample
one positive instance per query in each epoch and train using the
standard InfoNCE loss .
Alternatively, multiple positive instances can be used simultane-
ously, with the optimization logic for handling multiple positivesbeing conjunctive ( Conj-InfoNCE loss ), i.e.,
L𝑠(𝑞,𝐷+,𝐷−)=−∑︁
𝑑+∈𝐷+logexp(𝑠(𝑞,𝑑+))Í
𝑑∈{𝐷+,𝐷−}exp(𝑠(𝑞,𝑑))(4)
=−logÎ
𝑑+∈𝐷+exp(𝑠(𝑞,𝑑+))Í
𝑑∈{𝐷+,𝐷−}exp(𝑠(𝑞,𝑑)), (5)
where each positive sample’s probability is multiplied within the
logarithm and requires that each positive sample’s predicted prob-
ability be optimized towards the highest. If low-quality positive
samples are included, they can negatively impact the training of
other high-quality positives, thus degrading retriever performance.
To relieve this, we proposed shifting the optimization logic to a
disjunctive relationship ( Disj-InfoNCE loss ), i.e.,
L𝑠(𝑞,𝐷+,𝐷−)=−logÍ
𝑑+∈𝐷+exp(𝑠(𝑞,𝑑+))Í
𝑑∈{𝐷+,𝐷−}exp(𝑠(𝑞,𝑑)). (6)
In contrast to the Conj-InfoNCE loss, the difference here is that each
positive sample’s probability is summed within the logarithm. This
approach does not need all positive samples’ predicted probabilities
strictly optimized to the highest and reduces the impact of false
positives during retriever training.
4.4 Combination of Human Annotations and
LLM Annotations
LLM annotations, compared to human annotations, act as a form of
weak supervision, while human annotations provide high-quality
labels. In our work, we investigated whether LLM annotations can
replace human annotations. If human annotations cannot be sub-
stituted entirely, we then explored how LLM annotations could be
integrated with a minimal amount of human annotations to achieve
performance comparable to that of human-annotated retrievers. We
examined two methods of integration, i.e., (1) Interleave, mixing the
two sets of labels together for one-stage training; (2) Curriculum
learning ( CL), where labels of different quality are learned in two
stages—starting with the weakly supervised labels generated by
LLMs (we directly used utility selection/ranking annotation for the
first stage training), followed by the high-quality human labels.
4.5 Positive Sampling
A crucial aspect of retriever training is the selection of positive ex-
amples for each query. LLMs’ annotation might yield multiple posi-
tive instances. If the loss function is Disj-InfoNCE or Conj-InfoNCE,
for their positive selection during training for each query, we de-
vised three strategies: (1) Pos-one : choosing at least one annotated
positive example, with others selected randomly; (2) Pos-avg : com-
puting the average number of LLM-annotated positive examples
and selecting up to this average number during training; (3) Pos-all :
attempting to include all annotated positive whenever possible.
5 Experimental Setup
5.1 Datasets
MS MARCO Passage Ranking. We used the MS MARCO passage
ranking dataset [ 41] to train retrievers, which is derived from Bing’s
search query logs. The training set comprises approximately 8.8M
passages and 503K queries, annotated with shallow relevance labels,
averaging 1.1 relevant passages per query. For retrieval evaluation,

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, and Xueqi Cheng
we evaluated our methods using the development set of the MS
MARCO passage ranking task, i.e., MS MARCO Dev set, comprising
6980 queries. In addition, we also evaluated our methods on TREC
DL 2019 [ 10], and TREC DL 2020 [ 9], which include 43 and 54
queries and have graded human relevance judgments, respectively.
BEIR. BEIR [ 60] is a heterogeneous benchmark, which encom-
passes 18 diverse retrieval datasets from various fields (e.g., medical,
Wikipedia), and different downstream tasks (e.g., fact-checking and
question answering). Since some datasets are not publicly available,
we evaluated our methods on 14 publicly accessible datasets from
BEIR, including T-COVID [ 63], NFCorpus [ 4], NQ [ 34], HotpotQA
[75], FiQA [ 40], ArguAna [ 64], Touche [ 3], Quora, DBPedia [ 25],
SCIDOCS [ 6], FEVER [ 62], C-FEVER [ 12], SciFact [ 65], and CQA
[26]. We trained retrievers on the MS MARCO dataset [ 41] and
tested them on these datasets to evaluate zero-shot performance.
5.2 Baselines
We employed RetroMAE [ 71] as our backbone, a state-of-the-art
PLM for IR that uses a masked auto-encoder architecture. It features
an asymmetric encoder-decoder structure and varying masking ra-
tios, which make the reconstruction task more challenging and
enhance the encoder’s ability to learn effective representations.
RetroMAE was then trained on MS MARCO passage training data
annotated with various methods using the same candidate pas-
sage pool. We made comparisons with a wide variety of baseline
annotation methods, i.e.,
•Human labels: the original dataset labels generated by humans;
•REPLUG [57], which used the likelihood of the ground truth
answer as the utility score for information retrieval, annotating
the passage pool in a pointwise manner. The retriever was trained
using KL divergence loss;
•REPLUG (Human) , which used a warm-up retriever trained on
human labels and then fine-tuned using KL divergence loss;
•REPLUG (CL 20%) , which employed a curriculum learning
method: first trained with KL divergence and then retrained
on a randomly selected 20% of human-annotated labels;
•REPLUG (CL 100%) , which also used a curriculum learning
method, but was retrained on 100% of human annotated labels
after the initial KL divergence training.
5.3 Evalaution Settings
We conducted two types of evaluation: retrieval performance and
RAG performance in both the in-domain and out-of-domain settings.
For the retrieval performance evaluation :
•In-domain setting: (1) Human test collection : We used three
human-annotated test collections: MS MARCO Dev, and TREC
DL19/DL20 passage ranking [ 9,10]. (2) LLM test collection :
There may be distribution biases when using LLM-based an-
notations training while testing on human test collection, we
constructed an LLM-based test set to mitigate this effect. We
randomly selected 200 queries, and for each query, we employed
GPT-4o-mini [ 28] to re-annotate the top 20 retrieval results from
four retrievers using utility selection prompt (see Figure 1 (c))
based on the ground truth answers.•Out-of-domain setting: We leveraged the BEIR benchmark [ 60]
to evaluate the ability of the retriever fine-tuned on MS MARCO
in a zero-shot setting to generalize to unseen data.
For the RAG performance :
•In-domain setting: We used MS MARCO-dev to evaluate the end-
to-end RAG performance. The ground truth answers for queries
were obtained from the MS MARCO-QA dataset [41].
•Out-of-domain setting: We used two factoid QA datasets, i.e., NQ
[34] and HotpotQA [75].
To evaluate retrieval performance, we employed three standard
metrics: Mean Reciprocal Rank (MRR) [ 8], Recall and Normalized
Discounted Cumulative Gain (NDCG) [ 31]. For evaluating RAG
performation, we adopted two different approaches based on the
nature of the datasets: (1) For datasets that include non-factoid QA,
such as MS MARCO, we evaluated answer generation performance
using ROUGE [ 36], BLEU [ 44], and BERT-Score [ 81]. (2) For factoid
QA datasets, such as NQ and HotpotQA, we used Exact Match (EM)
and F1 score as main metrics.
5.4 Implementation Details
The hyperparameters of the retriever trained on human annotations
were the same as the original work [ 71]. The retriever was trained
for 2 epochs, with AdamW optimizer, batch-size 16(per device), and
learning rate 3e-5. The training was on a machine with 8 ×Nvidia
A800 (80GB) GPUs. The models were implemented with PyTorch
2.4 and Hugging Face transformers=4.40. For the second stage of
curriculum learning, the retriever was then trained for 1 epoch, the
learning rate of 3e-5, others were the same as the first stage. In our
experiments, the temperature hyperparameter was uniformly set
to 1.0, following [71].
6 Experimental Results
In this section, we conducted a comparative analysis of the perfor-
mance of the retrievers trained with different annotations to analyze
whether LLM-annotated data can replace human-annotated data
and to which extent human annotation can be saved ( RQ1 ). Sub-
sequently, we showed a detailed analysis of various strategies for
automated LLM annotation and what training objectives will yield
better performance ( RQ3 ). Moreover, we investigated retrievers
trained using different annotations on retrieval and RAG under the
out-of-domain setting ( RQ2 ). By default, the annotator is Qwen-
2.5-32B-Int8 in all experiments, if not specified otherwise.
6.1 In-Domain Performance
We compared the retriever performance trained with our utility-
focused annotations, REPLUG labels, and human labels on the
performance of retrieval and RAG tasks. All retrievers are trained
on the MS MARCO passage dataset and evaluated on both retrieval
and RAG tasks. For the RAG task, the query and retrieved passages
are directly fed into LLMs to generate answers.
Retrieval Performance. The results on different evaluation test
sets are shown in Table 2. From the results of human test collection,
we can observe that: (1) Retrievers trained on human annotations
have better performance compared to different LLM annotations.
Our utility-focused annotation has better performance than RE-
PLUG. For example, using the utility ranking approach improves
MRR@10 by 5.6% compared to REPLUG on the MS MARCO-dev.

Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 2: Retrieval performance (%) of different annotation methods. “R@k”, means “Recall@k”.+,−indicate significant
improvements and decline over human annotation, respectively (p<0.05) using a two-sided paired t-test.†indicates significant
improvements over REPLUG when using the same proportion of human annotations (p<0.05) using a two-sided paired t-test.
AnnotationHuman test collection LLM test collection
MS MARCO-Dev DL19 DL20 MS MARCO-Dev
MRR@10 R@50 R@100 R@1000 NDCG@10 NDCG@10 MRR@10 NDCG@10 R@5 R@10
Human 38.6 87.3 91.7 98.6 68.2 71.6 83.7 63.1 31.5 49.5
REPLUG 33.8−79.2−84.0−94.7−65.5 58.7 75.7−54.3−27.6−43.1−
REPLUG (Human) 34.5−80.6−85.8−94.8−62.9 62.7 76.8−54.3−26.8−42.2−
UtilSel 35.3−†83.6−†88.9−†97.7−†68.0 71.0 87.5+†65.8+†31.8†51.2†
UtilRank 35.7−†83.9−†89.2−†97.8−†67.1 71.0 86.1†66.1+†32.0†52.0+†
REPLUG (CL 20%) 36.6−84.9−90.0−98.3−69.5 67.8 81.7 60.2−30.2 47.3−
UtilSel (CL 20%) 38.2†86.7−†91.4†98.5†69.6 71.4 83.4 65.5+†32.9+†52.0+†
UtilRank (CL 20%) 38.3†86.4−†91.4†98.4 70.5 70.0 84.3 64.6†32.1†51.4+†
REPLUG (CL 100%) 38.7 86.8−91.3−98.6 69.5 69.7 83.7 63.1 30.7 50.0
UtilSel (CL 100%) 39.3+†87.3†92.1+†98.6 70.5 70.9 84.7 64.7+†31.5 50.9+
UtilRank (CL 100%) 39.2+†87.1 91.9†98.7 69.6 69.9 84.2 64.2 31.6 50.7
Table 3: RAG performance (%) of different retrievers trained with different annotation data on MS MARCO dev. The symbols
+,−, and†are defined in Table 2. The official BLEU evaluation for MS MARCO QA targets the entire queries, not individual
queries, thus no significance tests are conducted.
Top-k Annotation RecallGenerator: Llama-3.1-8B Generator: Qwen-2.5-32B-Int8
BLUE-3 BLUE-4 ROUGE-L BERT-score(F1) BLUE-3 BLUE-4 ROUGE-L BERT-score(F1)
Top-1Human 24.7 17.2 14.2 35.7 67.8 15.8 12.6 34.3 67.4
REPLUG 21.7−15.7 12.9 33.8−66.7−14.7 11.6 32.4−66.2−
REPLUG (Human) 21.7−16.1 13.3 34.4−66.9−15.1 12.0 32.6−66.3−
UtilSel 22.3−16.3 13.4 34.7−†67.4−†14.9 11.7 33.5−†67.1−†
UtilRank 22.6−16.6 13.6 35.1−†67.5−†15.2 12.0 33.9−†67.3−†
REPLUG (CL 20%) 23.2−16.7 13.7 34.9−67.4−15.2 12.1 33.6−67.1−
UtilSel (CL 20%) 24.6†17.4 14.3 35.4†67.7†15.8 12.6 34.2†67.4†
UtilRank (CL 20%) 24.6†17.4 14.4 35.6†67.8†15.8 12.6 34.3†67.5†
REPLUG (CL 100%) 25.0 17.2 14.2 35.8 67.8 15.8 12.6 34.4 67.5
UtilSel (CL 100%) 25.6+17.8 14.8 36.0 68.0+†16.2 12.9 34.6+†67.7+†
UtilRank (CL 100%) 25.5+17.7 14.7 35.9 68.0+†16.2 12.9 34.6+†67.7+†
(2) After using curriculum learning, all the retrievers trained on dif-
ferent LLM annotations have performance improvement, indicating
the effectiveness of curriculum learning on the combination of dif-
ferent annotations. For example, REPLUG and utility ranking have
improvements of 8.3% and 7.3% in terms of MRR@10 compared to
those without curriculum learning, respectively. For utility-focused
annotation, using 20% of human annotations achieves performance
comparable to using the full set of human-annotated data. If 100%
of the human annotations are used, the retriever’s performance
gains a significant improvement of 1.8% in MRR@10 compared to
training solely on human data. More details on curriculum learning
can be found in Figure 3. From the results of LLMs test collection,
we can observe that (1) Retrievers trained on our utility-focused
annotations perform better on the LLMs test collection than the
retriever trained on human annotations. For instance, the utility
selection annotated retriever outperforms the human-annotated re-
triever by 4.5% in terms of MRR@10. (2) Following the application of
curriculum learning and the addition of human-annotated training
data, the model that includes human annotations exhibits declinesin the LLMs test collection across multiple metrics compared to the
retriever without human annotations.
RAG Performance. To further evaluate the performance of the
retriever, we assessed the end-to-end performance of RAG. We
evaluated the answer generation performance for top-1 retrieval
results. The results are shown in Table 3. We observe the follow-
ing: (1) Similar to the retrieval performance, retrievers trained on
human-annotated data generally produce better downstream task
performance than those trained on different LLM-annotated data.
Our utility-focused annotation has better generation performance
than REPLUG on different generators. (2) Utility-focused annota-
tion with curriculum learning (100%) achieves the best generation
performance.
6.1.1 Exploration of Annotation Strategies. We analyzed the im-
pact of the following factors: (1) LLMs’ capabilities on annotation,
(2) utility selection and utility ranking, (3) loss function, (4) positive

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, and Xueqi Cheng
Table 4: Different retrieval performance (%) with various
strategies on automated LLM annotation on the MS MARCO
dev. By default, the annotator is Qwen-2.5-32B-Int8, loss func-
tion is Disj-InfoNCE, positive sampling is Pos-all and utility
selection, if not specified otherwise.
Annotation Strategy MRR@10 R@50 R@100 R@1000
LLMs
Llama-3.1-8B 33.0 81.7 87.7 97.4
Qwen-2.5-32B-Int8 35.3 83.6 88.9 97.7
Relevance vs Utility
Relevance selection 33.5 83.0 88.5 97.9
Utility selection 35.3 83.6 88.9 97.7
Utility ranking (InfoNCE) 35.7 83.9 89.2 97.8
Loss Function
InfoNCE 34.5 82.9 88.8 97.9
Conj-InfoNCE 34.0 82.0 88.0 97.5
Disj-InfoNCE 35.3 83.6 88.9 97.7
Positive Sampling
Pos-one 35.1 83.4 88.8 97.7
Pos-avg 35.1 83.2 88.8 97.7
Pos-all 35.3 83.6 88.9 97.7
Combination of Human and LLM Annotations
Interleave ( Pos-one ) 33.2 81.2 87.5 97.2
Curriculum learning ( Pos-one ) 38.2 86.7 91.4 98.5
Curriculum Learning ( Pos-all ) 37.8 86.5 91.2 98.5
sampling method, (5) combination of human annotations and our
utility-focused annotations.
Different LLMs. Annotation quality improves with the increasing
capability of the LLM.
Relevance vs Utility. As depicted in Table 4, annotations based
on utility demonstrate better retrieval performance compared to
those based on relevance selection. Thus, utility annotation may
be more suitable for the automatic annotation of retrievers than
direct relevance annotation. The performance difference between
retrievers trained with annotations from utility selection and utility
ranking is not significant.
Different Loss Function. As seen in Table 4, the results of Disj-
InfoNCE demonstrate better retrieval performance compared to the
other two loss functions. A possible explanation is that automatic
annotations may include poor-quality positive instances. If these
instances of varying quality are optimized independently, as in Conj-
InfoNCE or InfoNCE, the poor-quality positives could negatively
affect model optimization.
Positive Selection Strategies. Table 4 indicates that different pos-
itive instance selection strategies have minimal impact on retriever
training. This suggests that the number of positive instances does
not significantly affect training with Disj-InfoNCE.
Combination of Human and LLMs Annotations. We randomly
selected 20% of the human annotations and considered two methods
to integrate different annotations. From Table 4, we can see that
interleaving different labels during training causes interference
between them, leading to decreased performance. In contrast, using
curriculum learning enhances performance. Additionally, since
the high-quality human labels contain fewer positive examples,
the pos-one method aligns better with the distribution of human
0.0 0.5 1.03536373839MRR@10
Utility Selection
Utility Ranking
Human
0.0 0.5 1.097.7598.0098.2598.50Recall@1000
 Utility Selection
Utility Ranking
HumanFigure 3: Different retrieval performance (%) of using cur-
riculum learning with different ratios of human annotation
upon the retriever trained on LLM-annotated labels.
annotations than the pos-all method. Therefore, using pos-one
during curriculum learning yields better results.
6.2 Out-of-Domain Performance
We further evaluated the generalization capabilities of retrievers
trained with different annotations.
Retrieval Performance. Table 6 shows the zero-shot performance
of different retrievers. We can observe that (1) Utility-focused anno-
tation achieves the best performance among all annotation meth-
ods, with a NDCG@10 of 45.3. This indicates that models trained
with utility annotations possess strong generalization ability. More
impressively, without relying on human annotations, utility annota-
tion attains optimal (bold) or near-optimal performance (underline)
on seven datasets. This indicates utility-focused annotation’s su-
perior versatility across different scenarios. (2) REPLUG performs
worst, which illustrates the vulnerability of the retriever in this
method. (3) Human annotation performs well on in-domain re-
trieval results but poorly on out-of-domain datasets. And the re-
triever trained on utility-focused annotations experiences a slight
decline in zero-shot retrieval performance and a significant improve-
ment in in-domain retrieval performance after using curriculum
learning. This suggests that (a) relying solely on human supervision
may compromise the robustness of the retriever in the out-of-do-
main setting; (b) curriculum learning not only improves in-domain
retrieval performance but also maintains robustness on out-of-do-
main data to some extent, which has better balance performance
and robustness. Surprisingly, using curriculum learning led to a
further enhancement of REPLUG’s performance on both in-domain
and out-of-domain. The reason might be that the retriever trained
on the REPLUG labels relies too heavily on in-domain downstream
task annotations, resulting in poor out-of-domain performance. In-
corporating curriculum learning can alleviate this issue. Therefore,
applying curriculum learning in out-of-domain scenarios further
enhances performance.
RAG Performance. Retrievers were trained on different annota-
tion data and we directly used the top-5 retrieval results (HotpotQA
is a multi-hop dataset, requiring multiple pieces of evidence to
obtain the answer) for answer generation in RAG. The answer
generation performance of different retrievers is shown in Table
5. We observe the following: (1) Similar to retrieval performance,
retrievers trained on our utility-focused annotations achieve the
best RAG performance using different generators, especially on the

Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 5: RAG performance (%) of different retrievers trained using different annotated data on NQ and HotpotQA. The symbols
+,−, and†are defined in Table 2. “Llama” and “Qwen” are “Llama-3.1-8B“ and “Qwen-2.5-32B-Int8”, respectively.
AnnotationNQ (Top 5) HotpotQA (Top 5)
RecallGenerator: Llama Generator: Qwen
RecallGenerator: Llama Generator: Qwen
EM F1 EM F1 EM F1 EM F1
Human 56.7 42.8 56.4 43.6 57.9 54.8 31.5 42.6 38.6 50.7
REPLUG 46.2−41.1−53.7−41.6−55.0−53.3−30.6−41.6−38.0 50.0−
REPLUG (Human) 45.7−39.2−52.4−40.5−53.6−52.2−30.7 41.8−37.8−49.8−
UtilSel 61.1+†44.4+†58.8+†44.9†59.8+†55.8+†31.9†43.2†39.0†51.1†
UtilRank 62.0+†45.4+†59.8+†45.9+†60.0+†55.9+†31.4†43.0†38.7 51.0†
REPLUG (CL 20%) 55.0−43.3 56.9 44.7 58.4 56.5+31.3 42.6 38.6 50.7
UtilSel (CL 20%) 59.8+†43.4 58.0+44.9+59.3+56.2+31.9 43.0 38.8 51.0
UtilRank (CL 20%) 59.7+†44.7+58.9+†45.6+59.7+†56.2+31.5 42.9 39.0 51.3
REPLUG (CL 100%) 58.2+43.5 57.2 45.3+59.2+57.1+31.8 43.3+38.8 51.1
UtilSel (CL 100%) 59.9+†43.7 57.5 45.4+59.8+56.6+31.7 43.2 38.7 50.8
UtilRank (CL 100%) 59.4+†43.8 57.8+45.0+59.10+56.0+31.4 42.9 38.4 50.7
Table 6: Zero-shot retrieval performance (%) of different retrievers trained on different annotation data (NDCG@10).
Method BM25 Human REPLUG REPLUG(Human) UtilRank UtilSelCurriculum Learning, 20% Curriculum Learning, 100%
REPLUG UtilSel UtilRank REPLUG UtilSel UtilRank
DBPedia 31.8 36.0 29.1 29.8 37.9 38.0 35.9 37.4 37.4 36.1 37.1 37.5
FiQA 23.6 29.7 24.9 24.5 31.6 32.6 30.8 32.1 31.3 31.3 31.6 30.4
NQ 30.6 49.2 41.2 39.9 53.9 53.5 48.0 51.4 51.9 50.1 51.9 51.7
HotpotQA 63.3 58.4 57.4 55.5 59.6 59.6 60.2 60.0 59.8 60.5 60.1 59.5
NFCorpus 32.2 32.8 30.3 31.7 34.0 33.9 33.9 34.2 33.8 33.7 34.0 33.4
T-COVID 59.5 63.4 54.2 54.8 64.5 66.1 68.5 65.0 67.5 71.8 64.8 68.0
Touche 44.2 24.2 18.9 17.3 26.6 28.5 27.0 24.7 28.0 25.4 22.6 25.7
CQA 32.5 32.2 29.2 28.5 30.7 32.3 33.2 33.9 33.0 32.8 32.9 32.8
ArguAna 39.7 30.5 22.7 24.2 25.0 34.1 32.9 36.4 29.3 29.0 30.8 28.1
C-FEVER 16.5 18.0 13.2 13.8 16.4 19.5 17.9 16.5 15.3 18.4 18.5 16.8
FEVER 65.1 66.6 66.1 56.1 73.1 73.8 72.3 69.9 72.4 71.1 70.1 71.0
Quora 78.9 86.2 76.9 75.4 85.3 85.4 85.3 86.1 85.9 85.7 86.4 86.5
SCIDOCS 14.1 13.4 13.5 12.8 13.6 14.3 14.5 14.4 13.9 13.9 13.7 13.6
SciFact 67.9 63.1 59.3 63.0 63.2 62.8 63.2 64.2 63.8 63.6 64.1 64.9
Avg 42.9 43.1 38.4 37.7 43.9 45.3 44.5 44.7 44.5 44.5 44.2 44.3
NQ dataset, indicating the superiority of our annotation. (2) The
retriever trained on human annotations also does not perform as
well in out-of-domain RAG evaluations. Moreover, when we use
curriculum learning to incorporate different proportions of in-do-
main human annotations, the out-of-domain RAG performance
decrease, especially on the NQ dataset. However, compared to the
retriever trained purely on human annotations, it still maintains
the performance advantage in RAG, further demonstrating that
automated annotation contributes to the robustness of retriever
training. (3) REPLUG performs the worst in out-of-domain RAG
evaluation with different generators. A possible reason is that it
heavily relies on in-domain downstream task annotations, leading
to poorer robustness in out-of-domain settings.
7 Further Analyses
7.1 Curriculum Learning
In the second stage of curriculum learning, we conducted experi-
ments by training the retrievers with different proportions of man-
ually annotated labels. The retrieval performance achieved is il-
lustrated in Figure 3. The results indicate that, following weaklysupervised training, increasing the proportion of manual annota-
tions leads to a continuous improvement in performance.
7.2 Different Thresholds for Utility Ranking
Under the condition that at least one positive instance is present,
we used the top 10% - 50% of the ranked results as annotations,
recall, and precision of human labels, and the corresponding re-
trieval performance are shown in Table 4. We can observe that
a smaller threshold or high precision of human labels results in
better retrieval performance of the model. The results indicate a
significant impact of the number of positive instances on retrieval
performance, potentially due to the limited ability of the annotation
model, which can introduce false-positive annotations.
7.3 Efficiency and Cost
According to [ 21], the cost of human annotation is approximately
$0.09 per annotation on MTurk, a crowd-sourcing platform. Each
query requires annotations for 31 passages, and there are a total
of 491,007 queries, leading to a total human annotation cost of
$1,369,910. We utilize cloud computing resources, where the cost

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, and Xueqi Cheng
10% 20% 30% 40% 50%50607080Recall and Precision
34.434.634.835.035.235.435.6
MRR@10
Recall Precision MRR@10
Figure 4: Different retrieval performance (%) for utility rank-
ing annotation on the MS MARCO-dev and different anno-
tated recall (%) and precision (%) of human labels.
of using an A100 80GB GPU is assumed to be $0.8 per hour1. Our
utility-focused annotation process requires a total of 53 hours on an
8xA800 GPU machine using the Qwen 32B, resulting in a GPU com-
puting cost of $339. For the REPLUG method, the annotation process
takes 70 hours, costing $448 in GPU computing. However, REPLUG
requires human-annotated answers for each query, bringing the
total to $44,639. More details are provided in Table 7. Although hu-
man annotation achieves superior performance on the in-domain
dataset, the cost of such annotation is substantial. In contrast, the
utility-focused annotation offers the lowest annotation cost, with
performance second only to that of human annotation.
Table 7: Different retrieval performance (%) on the MS
MARCO-dev and corresponding annotation cost and time.
Annotation Cost($) Time MRR@10 Recall@100 Recall@1000
Human 1,369,910 - 38.6 91.7 98.6
REPLUG 44,639 53h 33.8 84.0 94.7
UtilSel 339 70h 35.3 88.9 97.7
UtilSel (CL 20%) 274,321 - 38.2 91.4 98.5
8 Conclusion and Future Work
In this work, we explored the use of LLMs to annotate large-scale
retrieval training datasets with a focus on utility. For different anno-
tation labels, experiments show that retrievers trained with utility
annotations perform worse in-domain than retrievers trained with
human annotations. However, they outperform retrievers trained
with human annotations in out-of-domain settings on both retrieval
and RAG tasks. For the combination of human annotations and
LLM annotations, experiments demonstrate that curriculum learn-
ing requires only 20% of human labels to achieve retrieval and
RAG performance comparable to that of human annotation. Using
100% human labels in curriculum learning can even surpass human
annotation and still exceed it in out-of-domain performance, high-
lighting the robustness of LLM automated annotation across differ-
ent datasets. Moreover, experiments show utility selection/ranking
has better performance than relevance selection on retrieval per-
formance and we propose a novel loss function that aggregates
all positive instances during optimization to reduce the impact of
low-quality positives annotated by LLMs. Due to the limitations
in obtaining human-annotated labels, our current annotation pool
1https://vast.ai/pricing/gpu/A800-PCIEuses positive examples and hard negative passages from the train-
ing of the retriever. This may not completely align with the actual
annotation process. In the future, we can analyze the performance
of LLM annotation separately based on a more realistic annotation
scenario, such as using pools composed of results from multiple
retrievers. Moreover, exploring better annotation techniques to
achieve human-level performance without human involvement is
a matter that requires further consideration in the future.

Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]Arkadeep Acharya, Brijraj Singh, and Naoyuki Onoe. 2023. Llm based generation
of item-description for recommendation system. In Proceedings of the 17th ACM
Conference on Recommender Systems . 1204–1207.
[2]Andrea Bacciu, Florin Cuconasu, Federico Siciliano, Fabrizio Silvestri, Nicola
Tonellotto, and Giovanni Trappolini. 2023. RRAML: reinforced retrieval aug-
mented machine learning. arXiv preprint arXiv:2307.12798 (2023).
[3]Alexander Bondarenko, Maik Fröbe, Meriem Beloucif, Lukas Gienapp, Yamen
Ajjour, Alexander Panchenko, Chris Biemann, Benno Stein, Henning Wachsmuth,
Martin Potthast, et al .2020. Overview of Touché 2020: argument retrieval.
InExperimental IR Meets Multilinguality, Multimodality, and Interaction: 11th
International Conference of the CLEF Association, CLEF 2020, Thessaloniki, Greece,
September 22–25, 2020, Proceedings 11 . Springer, 384–395.
[4]Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A
full-text learning to rank dataset for medical information retrieval. In Advances in
Information Retrieval: 38th European Conference on IR Research, ECIR 2016, Padua,
Italy, March 20–23, 2016. Proceedings 38 . Springer, 716–722.
[5]Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei
Huang, Luo Si, and Huajun Chen. 2022. Knowprompt: Knowledge-aware prompt-
tuning with synergistic optimization for relation extraction. In Proceedings of the
ACM Web conference 2022 . 2778–2788.
[6]Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld.
2020. Specter: Document-level representation learning using citation-informed
transformers. arXiv preprint arXiv:2004.07180 (2020).
[7]William S Cooper. 1973. On selecting a measure of retrieval effectiveness. Journal
of the American Society for Information Science 24, 2 (1973), 87–100.
[8]Nick Craswell. 2009. Mean reciprocal rank. Encyclopedia of database systems
(2009), 1703–1703.
[9]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview
of the TREC 2020 deep learning track. CoRR abs/2102.07662 (2021). arXiv preprint
arXiv:2102.07662 (2021).
[10] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M
Voorhees. 2020. Overview of the TREC 2019 deep learning track. arXiv preprint
arXiv:2003.07820 (2020).
[11] Jacob Devlin. 2018. Bert: Pre-training of deep bidirectional transformers for
language understanding. arXiv preprint arXiv:1810.04805 (2018).
[12] Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bulian, Massimiliano Ciaramita,
and Markus Leippold. 2020. Climate-fever: A dataset for verification of real-world
climate claims. arXiv preprint arXiv:2012.00614 (2020).
[13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
[14] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. Gptq:
Accurate post-training quantization for generative pre-trained transformers.
arXiv preprint arXiv:2210.17323 (2022).
[15] P Moreira Gabriel de Souza, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt
Schifferer, and Even Oldridge. 2024. Nv-retriever: Improving text embedding
models with effective hard-negative mining. arXiv preprint arXiv:2407.15831 1
(2024).
[16] Chunjing Gan, Dan Yang, Binbin Hu, Hanxiao Zhang, Siyuan Li, Ziqi Liu, Yue
Shen, Lin Ju, Zhiqiang Zhang, Jinjie Gu, et al .2024. Similarity is Not All You
Need: Endowing Retrieval Augmented Generation with Multi Layered Thoughts.
arXiv preprint arXiv:2405.19893 (2024).
[17] Jingsheng Gao, Linxu Li, Weiyuan Li, Yuzhuo Fu, and Bin Dai. 2024. SmartRAG:
Jointly Learn RAG-Related Tasks From the Environment Feedback. arXiv preprint
arXiv:2410.18141 (2024).
[18] Luyu Gao and Jamie Callan. 2021. Unsupervised corpus aware language model
pre-training for dense passage retrieval. arXiv preprint arXiv:2108.05540 (2021).
[19] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. Rethink training of BERT
rerankers in multi-stage retrieval pipeline. In Advances in Information Retrieval:
43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28–April
1, 2021, Proceedings, Part II 43 . Springer, 280–286.
[20] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. Simcse: Simple contrastive
learning of sentence embeddings. arXiv preprint arXiv:2104.08821 (2021).
[21] Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. ChatGPT outperforms
crowd workers for text-annotation tasks. Proceedings of the National Academy of
Sciences 120, 30 (2023), e2305016120.
[22] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Rajaram
Naik, Pengshan Cai, and Alfio Gliozzo. 2022. Re2G: Retrieve, rerank, generate.
arXiv preprint arXiv:2207.06300 (2022).
[23] Jiafeng Guo, Yinqiong Cai, Yixing Fan, Fei Sun, Ruqing Zhang, and Xueqi Cheng.
2022. Semantic models for the first-stage retrieval: A comprehensive review.
ACM Transactions on Information Systems (TOIS) 40, 4 (2022), 1–42.
[24] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.
Retrieval augmented language model pre-training. In International conference on
machine learning . PMLR, 3929–3938.[25] Faegheh Hasibi, Fedor Nikolaev, Chenyan Xiong, Krisztian Balog, Svein Erik
Bratsberg, Alexander Kotov, and Jamie Callan. 2017. DBpedia-entity v2: a test
collection for entity search. In Proceedings of the 40th International ACM SIGIR
Conference on Research and Development in Information Retrieval . 1265–1268.
[26] Doris Hoogeveen, Karin M Verspoor, and Timothy Baldwin. 2015. Cqadupstack: A
benchmark data set for community question-answering research. In Proceedings
of the 20th Australasian document computing symposium . 1–8.
[27] Xuming Hu, Zhaochen Hong, Zhijiang Guo, Lijie Wen, and Philip Yu. 2023. Read
it twice: Towards faithfully interpretable fact verification by revisiting evidence.
InProceedings of the 46th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 2319–2323.
[28] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
[29] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv preprint arXiv:2112.09118
(2021).
[30] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models.
Journal of Machine Learning Research 24, 251 (2023), 1–43.
[31] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation
of IR techniques. ACM Transactions on Information Systems (TOIS) 20, 4 (2002),
422–446.
[32] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
[33] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. arXiv preprint arXiv:2004.04906 (2020).
[34] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics 7 (2019), 453–466.
[35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[36] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries.
InText summarization branches out . 74–81.
[37] Xinyu Ma, Jiafeng Guo, Ruqing Zhang, Yixing Fan, Xiang Ji, and Xueqi Cheng.
2021. Prop: Pre-training with representative words prediction for ad-hoc retrieval.
InProceedings of the 14th ACM international conference on web search and data
mining . 283–291.
[38] Xinyu Ma, Jiafeng Guo, Ruqing Zhang, Yixing Fan, Yingyan Li, and Xueqi Cheng.
2021. B-PROP: bootstrapped pre-training with representative words prediction
for ad-hoc retrieval. In Proceedings of the 44th International ACM SIGIR Conference
on Research and Development in Information Retrieval . 1513–1522.
[39] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-
tuning llama for multi-stage text retrieval. In Proceedings of the 47th International
ACM SIGIR Conference on Research and Development in Information Retrieval .
2421–2425.
[40] Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott,
Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: financial
opinion mining and question answering. In Companion proceedings of the the web
conference 2018 . 1941–1942.
[41] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan
Majumder, and Li Deng. 2016. Ms marco: A human-generated machine reading
comprehension dataset. (2016).
[42] Jingwei Ni, Tobias Schimanski, Meihong Lin, Mrinmaya Sachan, Elliott Ash, and
Markus Leippold. 2024. DIRAS: Efficient LLM-Assisted Annotation of Document
Relevance in Retrieval Augmented Generation. arXiv preprint arXiv:2406.14162
(2024).
[43] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning
with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
[44] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a
method for automatic evaluation of machine translation. In Proceedings of the
40th annual meeting of the Association for Computational Linguistics . 311–318.
[45] Fabio Petroni, Patrick Lewis, Aleksandra Piktus, Tim Rocktäschel, Yuxiang Wu,
Alexander H Miller, and Sebastian Riedel. 2020. How context affects language
models’ factual predictions. arXiv preprint arXiv:2005.04611 (2020).
[46] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxi-
ang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training
Approach to Dense Passage Retrieval for Open-Domain Question Answering. In
Proceedings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies . Association for
Computational Linguistics.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Hengran Zhang*, Minghao Tang*, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, and Xueqi Cheng
[47] Hossein A Rahmani, Emine Yilmaz, Nick Craswell, Bhaskar Mitra, Paul Thomas,
Charles LA Clarke, Mohammad Aliannejadi, Clemencia Siro, and Guglielmo Faggi-
oli. 2024. Llmjudge: Llms for relevance judgments. arXiv preprint arXiv:2408.08896
(2024).
[48] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language
models. Transactions of the Association for Computational Linguistics 11 (2023),
1316–1331.
[49] Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu,
Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A Joint Training Method
for Dense Passage Retrieval and Passage Re-ranking. In Proceedings of the 2021
Conference on Empirical Methods in Natural Language Processing . 2825–2835.
[50] Stephen Robertson, Hugo Zaragoza, et al .2009. The probabilistic relevance
framework: BM25 and beyond. Foundations and Trends ®in Information Retrieval
3, 4 (2009), 333–389.
[51] Egil Rønningstad, Erik Velldal, and Lilja Øvrelid. 2024. A GPT among Annota-
tors: LLM-based Entity-Level Sentiment Annotation. In Proceedings of The 18th
Linguistic Annotation Workshop (LAW-XVIII) . 133–139.
[52] Alireza Salemi and Hamed Zamani. 2024. Learning to Rank for Multiple Retrieval-
Augmented Models through Iterative Utility Maximization. arXiv preprint
arXiv:2410.09942 (2024).
[53] Tefko Saracevic. 1975. Relevance: A review of and a framework for the thinking on
the notion in information science. Journal of the American Society for information
science 26, 6 (1975), 321–343. https://asistdl.onlinelibrary.wiley.com/doi/abs/10.
1002/asi.4630260604
[54] Tefko Saracevic. 1996. Relevance reconsidered. In Proceedings of the second
conference on conceptions of library and information science (CoLIS 2) . 201–218.
[55] Tefko Saracevic, Paul Kantor, Alice Y Chamis, and Donna Trivison. 1988. A
study of information seeking and retrieving. I. Background and methodol-
ogy. Journal of the American Society for Information science 39, 3 (1988),
161–176. https://www.researchgate.net/publication/245088184_A_Study_in_
Information_Seeking_and_Retrieving_I_Background_and_Methodology
[56] Linda Schamber and Michael Eisenberg. 1988. Relevance: The Search for a
Definition. (1988).
[57] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented
Black-Box Language Models. In Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers) . 8364–8377.
[58] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval augmentation reduces hallucination in conversation. arXiv preprint
arXiv:2104.07567 (2021).
[59] Rikiya Takehi, Ellen M Voorhees, and Tetsuya Sakai. 2024. LLM-Assisted Rel-
evance Assessments: When Should We Ask LLMs for Help? arXiv preprint
arXiv:2411.06877 (2024).
[60] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. [n. d.]. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
of Information Retrieval Models. In Thirty-fifth Conference on Neural Information
Processing Systems Datasets and Benchmarks Track (Round 2) .
[61] Paul Thomas, Seth Spielman, Nick Craswell, and Bhaskar Mitra. 2024. Large
language models can accurately predict searcher preferences. In Proceedings of
the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 1930–1940.
[62] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
2018. FEVER: a large-scale dataset for fact extraction and VERification. arXiv
preprint arXiv:1803.05355 (2018).
[63] Ellen Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman,
William R Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. 2021.
TREC-COVID: constructing a pandemic information retrieval test collection. In
ACM SIGIR Forum , Vol. 54. ACM New York, NY, USA, 1–12.
[64] Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the
best counterargument without prior topic knowledge. In Proceedings of the 56th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers) . 241–251.
[65] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific
claims. arXiv preprint arXiv:2004.14974 (2020).
[66] Dingmin Wang, Qiuyuan Huang, Matthew Jackson, and Jianfeng Gao. 2024.
Retrieve What You Need: A Mutual Learning Framework for Open-domain
Question Answering. Transactions of the Association for Computational Linguistics
12 (2024), 247–263.
[67] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2023. SimLM: Pre-training with Representa-
tion Bottleneck for Dense Passage Retrieval. In The 61st Annual Meeting Of The
Association For Computational Linguistics .
[68] Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, and Michael Zeng. 2021.
Want To Reduce Labeling Cost? GPT-3 Can Help. In Findings of the Association
for Computational Linguistics: EMNLP 2021 . 4195–4205.[69] Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang,
Jiwei Li, and Guoyin Wang. 2023. Gpt-ner: Named entity recognition via large
language models. arXiv preprint arXiv:2304.10428 (2023).
[70] Yequan Wang, Hengran Zhang, Aixin Sun, and Xuying Meng. 2022. Cort: A new
baseline for comparative opinion classification by dual prompts. In Findings of
the Association for Computational Linguistics: EMNLP 2022 . 7064–7075.
[71] Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. 2022. RetroMAE: Pre-
Training Retrieval-oriented Language Models Via Masked Auto-Encoder. In
Proceedings of the 2022 Conference on Empirical Methods in Natural Language
Processing . 538–548.
[72] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett,
Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor nega-
tive contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808
(2020).
[73] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2024. Qwen2. 5
Technical Report. arXiv preprint arXiv:2412.15115 (2024).
[74] Sohee Yang and Minjoon Seo. 2020. Is retriever merely an approximator of
reader? arXiv preprint arXiv:2010.10999 (2020).
[75] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing . 2369–2380.
[76] Hamed Zamani and Michael Bendersky. 2024. Stochastic rag: End-to-end retrieval-
augmented generation through expected utility maximization. In Proceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 2641–2646.
[77] Hamed Zamani, Fernando Diaz, Mostafa Dehghani, Donald Metzler, and Michael
Bendersky. 2022. Retrieval-enhanced machine learning. In Proceedings of the 45th
International ACM SIGIR Conference on Research and Development in Information
Retrieval . 2875–2886.
[78] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping
Ma. 2021. Optimizing dense retrieval model training with hard negatives. In
Proceedings of the 44th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 1503–1512.
[79] Hengran Zhang, Keping Bi, Jiafeng Guo, and Xueqi Cheng. 2024. Iterative Utility
Judgment Framework via LLMs Inspired by Relevance in Philosophy. arXiv
preprint arXiv:2406.11290 (2024).
[80] Hengran Zhang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and
Xueqi Cheng. 2024. Are Large Language Models Good at Utility Judgments?.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval . 1941–1951.
[81] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav
Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint
arXiv:1904.09675 (2019).
[82] Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha, Shicheng Tan, Yuxiao
Dong, and Jie Tang. 2024. LongRAG: A Dual-Perspective Retrieval-Augmented
Generation Paradigm for Long-Context Question Answering. In Proceedings
of the 2024 Conference on Empirical Methods in Natural Language Processing .
22600–22632.
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
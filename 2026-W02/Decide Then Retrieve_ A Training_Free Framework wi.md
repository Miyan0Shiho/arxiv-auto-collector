# Decide Then Retrieve: A Training-Free Framework with Uncertainty-Guided Triggering and Dual-Path Retrieval

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang

**Published**: 2026-01-07 13:20:59

**PDF URL**: [https://arxiv.org/pdf/2601.03908v1](https://arxiv.org/pdf/2601.03908v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but existing approaches indiscriminately trigger retrieval and rely on single-path evidence construction, often introducing noise and limiting performance gains. In this work, we propose Decide Then Retrieve (DTR), a training-free framework that adaptively determines when retrieval is necessary and how external information should be selected. DTR leverages generation uncertainty to guide retrieval triggering and introduces a dual-path retrieval mechanism with adaptive information selection to better handle sparse and ambiguous queries. Extensive experiments across five open-domain QA benchmarks, multiple model scales, and different retrievers demonstrate that DTR consistently improves EM and F1 over standard RAG and strong retrieval-enhanced baselines, while reducing unnecessary retrievals. The code and data used in this paper are available at https://github.com/ChenWangHKU/DTR.

## Full Text


<!-- PDF content starts -->

Decide Then Retrieve: A Training–Free Framework with
Uncertainty–Guided Triggering and Dual-Path Retrieval
Wang Chen1,2, Guanqiang Qi1, Weikang Li3,
Yang Li1,Deguo Xia1,Jizhou Huang1,
1Baidu Inc,2The University of Hong Kong,3Peking University
Correspondence:liyang164@baidu.com
Abstract
Retrieval–augmented generation (RAG) en-
hances large language models (LLMs) by incor-
porating external knowledge, but existing ap-
proaches indiscriminately trigger retrieval and
rely on single-path evidence construction, of-
ten introducing noise and limiting performance
gains. In this work, we proposeDecide Then
Retrieve(DTR), a training-free framework that
adaptively determineswhenretrieval is nec-
essary andhowexternal information should
be selected. DTR leverages generation uncer-
tainty to guide retrieval triggering and intro-
duces a dual-path retrieval mechanism with
adaptive information selection to better han-
dle sparse and ambiguous queries. Exten-
sive experiments across five open-domain QA
benchmarks, multiple model scales, and dif-
ferent retrievers demonstrate that DTR con-
sistently improves EM and F1 over standard
RAG and strong retrieval-enhanced baselines,
while reducing unnecessary retrievals. The
code and data used in this paper are available
athttps://github.com/ChenWangHKU/DTR.
1 Introduction
Large language models (LLMs) have dramati-
cally advanced natural language processing (NLP),
achieving comparable or even better performance
than human beings (Touvron et al., 2023; Achiam
et al., 2023; Guo et al., 2025; Yang et al., 2025).
However, previous studies (Ji et al., 2023) found
that LLMs can only answer questions or accom-
plish tasks by leveraging their parametric knowl-
edge, which cannot update the latest information
nor private datasets. To address this issue, a
retrieval–augmented generation (RAG) framework
was designed to enhance the generation perfor-
mance of LLMs using the retrieved information
(Lewis et al., 2020; Guu et al., 2020; Karpukhin
et al., 2020; Chen et al., 2025b). The effectiveness
of RAG has been validated across various NLPtasks, achieving impressive improvement com-
pared with pure LLM systems (Ram et al., 2023;
Gao et al., 2023b).
Despite its widespread adoption, conventional
RAG systems suffer from two critical shortcomings.
First, they trigger retrieval for every query, includ-
ing straightforward questions that could be resolved
solely using the LLM’s internal parametric knowl-
edge, resulting in unnecessary noise, which may
degrade the final generation accuracy. Second, they
exhibit vulnerability to sparse queries—concise
user inputs with limited contextual signals—which
often yield irrelevant or low-quality retrievals due
to inadequate semantic cues, ultimately degrading
answer accuracy and reliability (Wang et al., 2023a;
Gao et al., 2023a; Zhao et al., 2024; Singh et al.,
2025; Chen et al., 2025a). Please refer to the Ap-
pendix C for detailed analysis.
Existing work (Wang et al., 2023b; Jeong et al.,
2024a; Zhang et al., 2025c) trains a model to de-
termine whether to retrieve, which may not be
generalized to other scenarios, e.g., with differ-
ent generators. In addition, existing approaches
use query augmentation (e.g., appending LLM–
generated pseudo-documents or external data to en-
rich sparse queries) (Wang et al., 2023a; Jagerman
et al., 2023; Buss et al., 2023; Jeong et al., 2024b)
to enhance retrieval accuracy. This may induce
unnecessary noise due to irrelevant retrievals, de-
grading the performance of RAG systems in some
scenarios.
To address these issues, we introduceDecide
Then Retrieve(DTR), a unified, training-free frame-
work that rethinks retrieval as a conditional and
adaptive process. DTR is built on two key ideas.
First,uncertainty-guided triggering(UGT) lever-
ages the LLM’s own generation uncertainty to
decide whether retrieval is beneficial, allowing
confident queries to bypass retrieval and prevent-
ing noise from irrelevant evidence. Second,dual-
path retrieval with adaptive information selection
1arXiv:2601.03908v1  [cs.CL]  7 Jan 2026

𝑠(𝑑)=𝑠𝑑,𝑞+𝑠(𝑑,𝑝)
dQueryContextqpAdaptive Information Selection (AIS)
Query
ContextDual-Path Retrieval (DPR)
Retrieve
…Corpus
LLM
Query+AnsGeneration(c) DPR-AIS Module(b) Uncertainty–GuidedTriggering(UGT)(a) DecideThenRetrieve(DTR)
LLMQuery:Which NFL team represented the NFC at Super Bowl 50?Carolina Panthersu=0.03DPR-AISQuery
LLMAns
Not to retrieveTo retrieve
AIS𝑢<𝑢,	?Ans:CarolinaPanthersYesNo𝑢=−!"log∏𝑃𝑎*#		𝑞,𝑎*$#)"#%!
LLMPseudoContextFigure 1: Overview of the proposed decide then retrieve (DTR) framework. (a) DTR can adaptively determine
whether to retrieve and how to select external information. (b) DTR guides whether to activate retrievals based on
the uncertainty score. (c) DTR adaptively selects effective information based on the dual-path retrieval for the final
generation.
(DPR-AIS) treats the original query and an LLM-
generated pseudo-context as complementary sig-
nals, retrieving evidence from both perspectives
and selecting documents that are jointly relevant.
To summarize, this paper makes the following
key contributions:
•We proposeDecide Then Retrieve(DTR), a
training-free and model-agnostic RAG frame-
work that can be easily plugged into existing
systems, enabling seamless deployment while
consistently improving RAG performance.
•We introduce uncertainty-guided triggering
to selectively activate retrieval based on gen-
eration uncertainty and a dual-path retrieval
mechanism with adaptive information selec-
tion to improve evidence quality for sparse
queries.
•We provide a principled accuracy and uncer-
tainty analysis that explains when retrieval
is beneficial and how retrieval noise affects
generation.
•Extensive experiments across five QA
datasets, various model sizes, multiple re-
trieval depths, and different retrievers demon-
strate consistent and robust improvements
over strong RAG baselines.2 Related Work
In this section, we review the related works in re-
cent years, primally focusing on adaptive retrieval
and query augmentation.
2.1 Adaptive retrieval
Jeong et al. (2024a) trained a classifier (i.e., a small
LM) to predict the complexity level of incoming
queries, achieving a dynamic RAG system. Specif-
ically, the classifier splits queries into three com-
plexity levels: A, B, and C levels, which corre-
spond toNon Retrieval,Single-step Retrieval, and
Multi-step Retrieval, respectively, offering an adap-
tive retrieval approach. However, the classification
results may differ from the capabilities of LLMs,
which may not extend to other RAG systems. Sim-
ilarly, a few studies also trained a classifier to pre-
dict whether LLMs know the given queries (Wang
et al., 2023b; Zhang et al., 2025c) or each unit’s
importance of queries (Jia et al., 2025). Also, a
few studies dynamically activated web searching
engines when LLMs generated low-confidence to-
kens (Jiang et al., 2023; Su et al., 2022). However,
these methods may not be generalized to other sce-
narios, where the classifier or generator is different.
2.2 Query augmentation
A query from users may be concise and contain
relatively sparse information, which can result in
suboptimal retrievals and poor answers. To ad-
dress this issue, many studies have proposed to
2

augment the query by appending additional terms
extracted from retrieved documents (Lavrenko and
Croft, 2017; Lv and Zhai, 2009) or generated by
neural models (Zheng et al., 2020; Mao et al.,
2021). Recently, many studies leveraged advanced
LLMs to augment queries and achieved notable im-
provement (Buss et al., 2023; Wang et al., 2023a;
Jagerman et al., 2023; Gao et al., 2023a; Lin et al.,
2023). For example. Wang et al. (2023a) first
prompted LLMs to generate a pseudo-document of
the query, which was then concatenated with the
original query to enhance the retrieval. In addition,
a few studies (Jeong et al., 2022, 2024b) used an
external database, such as relevant tabular data, to
augment original queries, which can enhance query
representations but require additional data. Fur-
thermore, a few studies (Chan et al., 2024; Zhang
et al., 2025a) trained models to refine or encode
queries to enhance the performance of information
retrieval and question answering. However, these
3 Preliminaries
In this section, we first define the RAG system, and
then, we analyze the primary facts that affect the
generation accuracy of a RAG system.
3.1 Problem setup
In a retrieval–augmented generation (RAG) system,
a collection of documents is first split into small
chunks D, which are then embedded into vectors
using an embedding model f. Given a query q,
the retriever Rfirst searches top krelevant chunks
Dk={d 1, d2, ..., d k} ⊂D from the collection
according to the similarity between the query and
chunks, i.e., Dk=R(D, q) . The similarity sis
calculated using the inner product (IP) or L2 norm
between the query embedding q=f(q) and chunk
embeddings d=f(d),∀d∈D . Finally, the re-
trievals and query are incorporated into a prompt
P(q, D k), which is used as the input of the gen-
erator model gto generate the final answer, i.e.,
ˆa=G(P(q, D k)). We list all notations and abbre-
viations in Appendix B.
3.2 Accuracy analysis of RAG
Given a query q, we analyze the probability that
a retrieval-augmented generation (RAG) system
produces an accurate answer by decomposing the
process into parametric answering, retrieval, and
generation stages, as illustrated in Figure 2. We
denote by Pparam the probability that the large lan-
guage model (LLM) can directly generate a correct
Query
LLMCorrectAns
WrongAnsCorrectAnsWrongAns
LLM
CorrectAnsWrongAns
LLMWrongPassageCorrectPassageWrongPassageCorrectPassage𝑝!"#"$𝑝#%&𝑝'%(𝑝#%&𝑝'%(Figure 2: Overview of RAG accuracy analysis.
answer based solely on its parametric knowledge,
without relying on external evidence. When Pparam
is low, the system resorts to external retrieval. Let
Pretdenote the probability that the retriever returns
correct and relevant passages for the query, captur-
ing the effectiveness of the retrieval module. Given
correct retrievals, the probability that the LLM suc-
cessfully generates an accurate answer from the re-
trieved passages is denoted as Pgen, which reflects
the model’s robustness to noise and its long-context
reasoning capability.
We assume that the LLM cannot generate an
accurate answer when conditioned on incorrect re-
trievals. Under this assumption, the overall proba-
bility of obtaining a correct answer via retrieval is
Pret·Pgen. Therefore, external retrieval should be
triggered only when retrieval-based answering is
expected to outperform parametric answering, i.e.,
Pret·Pgen> P param.(1)
This formulation yields several insights. First,
when Pparam is sufficiently high, incorporating re-
trieval may degrade performance due to imperfect
retrieval accuracy. Second, for a fixed LLM, both
Pparam andPgenare primarily determined by the
model’s intrinsic capability, whereas Pretdepends
on the retrieval system. Consequently, we identify
two key research questions:(1)how to determine,
for a given query, whether retrieval should be in-
voked;and (2)how to improve retrieval accuracy to
maximize the overall effectiveness of RAG systems.
4 Decide Then Retrieve
In this section, we introduce the proposed de-
cide then retrieve (DTR) method, including (1)
uncertainty–guided triggering (UGT) and (2) dual-
3

path retrieval with adaptive information selection
(DPR-AIS).
4.1 Uncertainty–Guided Triggering
Given a query q, the LLM generates an answer
ˆa={ˆa 1, . . . ,ˆa T}in an autoregressive manner,
where the generation probability is
P(ˆa|q) =TY
t=1P(ˆat|q,ˆa <t).(2)
Following recent work (Kang et al., 2025; Fu et al.,
2025), we define the uncertainty of the generated
answer as the normalized negative log-likelihood:
u=−1
TlogP(ˆa|q).(3)
This uncertainty score measures the model’s confi-
dence in its own prediction and can be computed
directly from next-token probabilities.
Figure 3 analyzes how uncertainty relates
to answer accuracy and retrieval triggering
across HotpotQA (Yang et al., 2018) and Natu-
ralQA (Kwiatkowski et al., 2019) (please refer to
Appendix E for more results). As the uncertainty
threshold increases, the exact match (EM) score
of answers generatedwithoutretrieval monotoni-
cally decreases, indicating that higher uncertainty
is strongly associated with lower parametric ac-
curacy. This indicates that the proposed uncer-
tainty score can be used to guide the decision of re-
trievals. At the same time, the query ratio—defined
as the proportion of queries whose uncertainty is
no greater than the threshold—monotonically in-
creases, implying that a higher threshold allows
more queries to bypass retrieval.
Figure 3: Generation Uncertainty vs. Parametric Accu-
racy and Query Coverage.Qwen2.5series models are
used as the generators.4.2 Dual-path retrieval
Unlike prior approaches that first generate a pseudo-
context (e.g., an answer or rationale) and then
merge it with the original query into a single re-
trieval signal—an operation that can amplify noise
due to LLM hallucinations—we adopt adual-
path retrieval(DPR) mechanism that treats the
query and the self-generated context as comple-
mentary but independent sources of information.
Specifically, DPR executes two parallel retrieval
operations—one conditioned on the query and the
other on the pseudo-context. Specifically, as shown
in Figure 4 (a), the system retrieves top nrele-
vant documents using the embeddings of the query
q=f(q) and the generated context p=f(p) ,
respectively, and the retrieved documents can be
denoted as D2n={d 1, d2, ..., d 2n} ⊂D . This
dual-path strategy enhances the relevance and di-
versity of the retrieved documents, which is espe-
cially beneficial for complex or ambiguous queries
where either source alone may fall short.
rqprqpd𝜃!𝜃"𝜃#Retrievalembeddingsa.Dual-path retrievalb.Similarity calculation
Figure 4: Illustration of (a) dual-path retrieval mecha-
nism and (b) similarity calculation of retrievals.
Adaptive information selection.Once the dual-
path retrieval yields a set of 2ncandidate docu-
ments based on the original query and the LLM-
generated pseudo-context, we introduce an adap-
tive information selection (AIS) mechanism to re-
fine this candidate set. The goal is to prioritize
documents that are simultaneously relevant to q
andp, ensuring that the final context passed to the
LLM is not only topically aligned but also seman-
tically coherent from different perspectives. For-
mally, we compute the score for each document
s(d)as follows:
s(d) =s 1(d, q) +s 2(d, p),∀d∈D 2n,(4)
where s1(d, q) ands2(d, p) denote the relevance
between the document dand the query qand LLM–
generated contextp, respectively.
As shown in Figure 4 (b), a straightforward
method is to calculate the relevance ( s1ands2)
4

as the IP between their corresponding embeddings,
as follows:
s1(d, q) = cos(θ 1) =⟨d,q⟩,(5)
s2(d, p) = cos(θ 2) =⟨d,p⟩,(6)
where q=f(q) ,p=f(p) , and d=f(d),∀d∈
D2n. Remind that q,p, and dare normalized vec-
tors. However, such aggregation may not fully cap-
ture the joint relevance, especially if either qorpin-
troduces semantic noise. Therefore, AIS leverages
a geometric intuition: if both qandpalign well
with a given document d, then the combined angle
θ=θ 1+θ 2should be minimized. This motivates
the maximization of s(d) = cos(θ) = cos(θ 1+θ2),
which expands to:
s(d) = cos(θ 1+θ 2)
=s 1·s2−p
1−(s 1)2·p
1−(s 2)2.(7)
This formulation implicitly rewards documents that
are jointly aligned with both qandpwhile penal-
izing those that diverge from either direction. The
DPR-AIS algorithm, i.e., dual-path retrieval–based
adaptive information selection, is formulated as
Algorithm 1.
Algorithm 1DPR-AIS
Input: Query q, LLM–generated pseudo-context
p, encoder f(·), retriever R, corpus D=
{d1, d2, . . . , d N}, retrieval size n, and selection
sizek
Output: Filtered relevant documentsD k⊂D
1:Compute embeddings: q←f(q),p←
f(p),d←f(d)
2:Retrieve top- ndocuments using q:Dq←
R(D,q)
3:Retrieve top- ndocuments using p:Dp←
R(D,p)
4:Merge retrieved sets:D 2n←D q∪Dp
5:foreachd∈D 2ndo
6:Computes 1=⟨q,d⟩ands 2=⟨p,d⟩
7: Compute joint relevance score using Eq. (7)
8:end for
9:Select top- kdocuments from D2nby descend-
ings(d)values
10:returnselected document setD k
5 Experiments
5.1 Experiment setup
Evaluation datasets.We evaluate the effective-
ness of the proposed method on open domainQA datasets: (1) NaturalQA (Kwiatkowski et al.,
2019), (2) WebQuestions (Berant et al., 2013), (3)
SQuAD (Rajpurkar et al., 2016), and (4) Trivi-
aQA (Joshi et al., 2017). In addition, we further
test the method using one complex multi-hop QA
dataset—HotpotQA (Yang et al., 2018). For the
open domain datasets, we use the 21M English
Wikipedia dump as the corpus for retrieval, while
for HotpotQA, we use its original corpus. The
statistics of all datasets are listed in Table 1.
Datasets #Questions #Documents
NaturalQA 3,610 21 M
WebQuestions 2,032 21 M
SQuAD 10,570 21 M
TriviaQA 11,313 21 M
HotpotQA 7,405 5 M
Table 1: Statistics of evaluation datasets.
Baselines.We compare the proposed method
with other training–free baselines. (1)No Re-
trievalleverages the parametric knowledge of
LLM to directly generate a response to the given
query without retrieval. (2)Standard RAGin-
corporates the retrieved top- kdocuments with the
query as the input of LLM through prompting. (3)
LLM Judge(Wang et al., 2023b; Jeong et al.,
2024a; Zhang et al., 2025c) determines whether
to trigger retrievals based on the LLM. (4)HyDE
(Gao et al., 2023a) leverages the LLM–generated
passage to enhance the retrieval. (5)Q2D(Wang
et al., 2023a) concatenates multiple queries and
the LLM–generated pseudo-document to perform
the retrieval. (6)CoT(Jagerman et al., 2023) first
prompts the LLM to generate the answer as well
as the rationale to the given query, and then com-
bines multiple queries and the LLM outputs into
the retrieval signal.
Evaluation metrics.To comprehensively assess
the quality of generated answers, we adopt two
widely used metrics in open-domain question an-
swering: Exact Match (EM) and F1 score. The EM
metric quantifies the proportion of predictions that
exactly match any one of the ground-truth answers,
serving as a strict measure of correctness. On the
contrary, the F1 score provides a more forgiving
evaluation by computing the token-level overlap
between the predicted and reference answers. We
normalize the predicted and ground truth answers
following the implementation of Fang et al. (2025).
5

MethodHotpotQA NaturalQA TriviaQA SQuAD WebQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen2.5-7B-Instruct
No Retrieval 18.73 25.38 17.40 24.95 43.24 49.12 12.20 18.63 22.59 35.03 22.83 30.62
Standard RAG 36.18 46.56 34.29 44.00 56.30 64.09 27.57 36.12 24.70 38.27 35.81 45.81
LLM Judge (7B) 18.80 25.49 17.40 24.95 43.27 49.15 12.21 18.64 22.59 35.03 22.85 30.65
Trigger Ratio (0.5%) (0.0%) (0.1%) (0.0%) (0.0%) (0.1%)
LLM Judge (72B) 34.56 44.60 26.79 35.64 49.92 56.38 25.14 33.21 23.67 36.23 32.02 41.21
Trigger Ratio (88.7%) (43.7%) (27.8%) (70.0%) (26.9%) (51.4%)
HyDE 32.64 42.40 35.32 44.92 55.31 63.17 24.52 32.62 25.69 39.40 34.70 44.50
Q2D 34.54 44.44 35.93 45.87 57.48 65.29 27.34 35.93 25.39 39.62 36.14 46.23
CoT 34.44 44.57 36.18 45.89 57.39 65.26 27.09 35.57 25.84 39.99 36.19 46.25
DTR (u= 0.001) 36.53 46.95 38.03 47.83 59.13 67.12 29.38 37.84 26.28 40.65 37.87 48.08
Trigger Ratio (87.7%) (94.8%) (90.8%) (96.8%) (91.8%) (92.4%)
DTR (u= 0.005) 36.16 46.55 37.42 47.35 59.41 67.23 29.27 37.69 26.77 41.17 37.81 48.00
Trigger Ratio (83.5%) (90.4%) (83.0%) (93.6%) (84.5%) (87.0%)
DTR (u= 0.01) 36.02 46.34 37.15 47.13 59.51 67.24 29.10 37.45 27.17 41.28 37.79 47.89
Trigger Ratio (81.0%) (87.5%) (78.7%) (91.2%) (81.1%) (83.9%)
Qwen2.5-72B-Instruct
No Retrieval 25.32 33.54 27.42 36.70 62.51 68.99 18.72 25.64 25.39 39.61 31.87 40.90
Standard RAG 39.53 51.00 37.56 49.27 64.27 72.56 29.86 39.85 22.93 40.95 38.83 50.73
LLM Judge (7B) 25.42 33.66 27.42 36.70 62.51 68.99 18.73 25.66 25.39 39.61 31.90 40.92
Trigger Ratio (0.5%) (0.0%) (0.1%) (0.0%) (0.0%) (0.1%)
LLM Judge (72B) 39.24 50.54 33.91 44.90 65.51 72.61 28.52 37.63 25.54 40.78 38.54 49.29
Trigger Ratio (88.7%) (43.7%) (27.8%) (70.0%) (26.9%) (51.4%)
HyDE 35.08 45.62 37.40 49.13 61.42 69.71 26.03 35.66 24.51 42.12 36.89 48.45
Q2D 36.95 47.93 37.73 49.72 63.28 71.71 29.09 38.98 23.97 41.70 38.20 50.01
CoT 36.87 47.96 37.56 49.54 64.02 72.47 29.22 38.92 24.06 42.12 38.35 50.20
DTR (u= 0.001) 40.68 52.28 38.92 51.28 66.15 73.98 31.04 40.97 24.56 42.26 40.27 52.16
Trigger Ratio (83.9%) (88.1%) (71.0%) (91.9%) (82.7%) (83.5%)
DTR (u= 0.005) 40.27 51.74 38.56 50.93 67.06 74.61 30.75 40.49 25.64 42.95 40.46 52.14
Trigger Ratio (77.1%) (79.2%) (59.8%) (85.6%) (73.4%) (75.0%)
DTR (u= 0.01) 39.89 51.27 38.42 50.71 67.24 74.72 30.51 40.15 25.64 43.01 40.34 51.97
Trigger Ratio (72.7%) (74.5%) (54.4%) (81.5%) (69.5%) (70.5%)
Table 2: Main results across five QA datasets withQwen2.5-7B-InstructorQwen2.5-72B-Instructas the generator
andbgeas the retriever, respectively. Except for experiments that require no retrievals, all results are generated
based ontop-3 retrievals.Trigger Ratioindicates the proportion of queries for which the retriever was triggered.
Boldand underlined values represent the best and second-best scores, respectively.
Implementation details.We implement our
DTR framework using Qwen2.5 series mod-
els (QwenTeam, 2024) as the backbone LLM for
answer and context generation and retrieval trigger-
ing judge. To ensure deterministic outputs and elim-
inate variability due to random sampling, we set
the temperature to 0.0 during decoding (Kim et al.,
2024). For dense retrieval, we use the bge-large-
en-v1.5 embedding model (Xiao et al., 2023) and
e5 (Wang et al., 2022), employing the inner prod-
uct (IP) as the similarity metric. In the dual-path
retrieval step, we retrieve the top 5 most relevant
documents independently for both the query and
the generated pseudo-context, resulting in a com-
bined candidate pool of 10 documents ( 2n= 10 ).
All remaining experimental configurations of base-lines follow the implementations reported in their
respective original papers. We present the prompts
used in this study in Appendix D.
5.2 Main results
Overall performance.Table 2 summarizes the
main results on five QA benchmarks using
Qwen2.5-7B-InstructandQwen2.5-72B-Instruct
as generators. Across both model scales and all
datasets, DTR consistently achieves the best or
second-best performance in terms of both EM and
F1. For the 7B model, DTR improves the aver-
age EM/F1 from 35.81/45.81 (standard RAG) to
up to 37.87/48.08, demonstrating clear gains over
static retrieval. Similar improvements are observed
for the 72B model, where DTR reaches an aver-
6

MethodHotpotQA NaturalQA TriviaQA SQuAD WebQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Qwen2.5-7B-Instruct
DTR (u= 0.001) 36.53 46.95 38.03 47.83 59.13 67.12 29.38 37.84 26.28 40.65 37.87 48.08
w/oUGT37.27 47.71 38.09 47.8858.82 66.94 29.19 37.82 25.84 40.30 37.84 48.13
w/oDPR 35.46 45.77 34.76 44.41 56.80 64.44 27.80 36.20 25.10 38.49 35.98 45.86
w/oAIS (1q+ 2p) 30.38 40.20 30.36 39.18 51.90 59.16 22.15 29.54 20.96 33.53 31.15 40.32
w/oAIS (2q+ 1p) 34.30 44.24 33.30 42.75 55.10 62.56 26.08 34.05 23.28 36.93 34.41 44.10
Qwen2.5-72B-Instruct
DTR (u= 0.001) 40.68 52.28 38.92 51.28 66.15 73.98 31.04 40.97 24.56 42.26 40.27 52.16
w/oUGT 40.54 52.12 39.47 51.5164.62 73.16 30.63 40.85 23.92 41.82 39.84 51.89
w/oDPR 39.99 51.44 37.56 49.40 66.03 73.66 30.37 40.06 23.97 41.58 39.58 51.23
Table 3: Ablation results across five QA datasets withQwen2.5-7B-InstructorQwen2.5-72B-Instructas the
generator.w/oAIS (w/ 1 q+ 2p) andw/oAIS (w/ 2 q+ 1p) denote retrieving documents using fixed proportions, i.e.,
two from the query and one from the pseudo-context, or vice versa.Boldand underlined values represent the best
and second-best scores, respectively.
age EM/F1 of 40.46/52.14, outperforming standard
RAG and all competing baselines. We list more
evaluation results with top-5 retrievals or e5 as the
retriever in Appendix F.
Effectiveness of uncertainty-guided triggering.
Compared to standard RAG, which triggers re-
trieval for all queries, DTR selectively activates re-
trieval based on uncertainty, which yields higher ac-
curacy, confirming that unnecessary retrievals can
introduce noise and degrade generation quality. On
the contrary, LLM Judge with a small model (7B)
rarely triggers retrieval, resulting in performance
nearly identical to the no-retrieval baseline, while
LLM Judge (72B) triggers retrieval frequently but
remains inferior to DTR, highlighting the advan-
tage of uncertainty-based signals over heuristic or
judge-based decisions. In addition, varying the
uncertainty threshold uallows DTR to trade off
retrieval coverage and accuracy. Smaller thresh-
olds lead to higher trigger ratios and slightly better
performance on reasoning-intensive datasets (e.g.,
HotpotQA, NaturalQA), while larger thresholds re-
duce retrieval frequency with marginal accuracy
loss. This flexibility enables DTR to adapt to dif-
ferent deployment constraints.
Comparison with retrieval-enhanced prompting.
Methods such as HyDE, Q2D, and CoT improve
over standard RAG in some cases but are consis-
tently outperformed by DTR. This indicates that
the proposed dual-path retrieval mechanism with
adaptive information selection is more robust to
noise and can retrieve more relevant evidence, up-
grading the overall performance of RAG systems.5.3 Ablation results
Table 3 presents ablation studies that analyze the
contribution of each component in DTR under both
Qwen2.5-7B-InstructandQwen2.5-72B-Instruct.
Effect of uncertainty-guided triggering (UGT).
Removing UGT results in comparable or slightly
lower scores, indicating that UGT effectively
avoids activating unnecessary retrievals without
sacrificing overall accuracy. Notably, the perfor-
mance gap is more evident for the 72B model, sug-
gesting that stronger parametric capability benefits
more from uncertainty-guided triggering, as ac-
curate answers can be generated directly without
relying on external evidence.
Effect of dual-path retrieval (DPR).Disabling
DPR (w/o DPR) consistently degrades performance
across datasets and model scales. For the 7B
model, the average EM/F1 drops from 37.87/48.08
to 35.98/45.86, and similar declines are observed
for the 72B model. This confirms that the dual-
path retrieval mechanism is a critical component,
enabling complementary evidence acquisition that
improves answer generation.
Effect of adaptive information selection (AIS).
We further ablate AIS by fixing the composition
of retrieved information. When AIS is removed,
and the system uses static combinations (1 q+2p
or 2q+1p), performance drops substantially, es-
pecially for the 7B model. This demonstrates
that adaptively selecting and balancing different
retrieval paths is essential for effectively leveraging
retrieved evidence.
7

MethodHotpotQA
bge e5
Standard RAG 61.9% 59.3%
HyDE 49.9% 49.9%
Q2D 54.1% 55.2%
CoT 53.6% 54.7%
DPR62.7% 62.6%
Table 4: Comparison of retrieval performance between
our proposed dual-path retrieval method and the other
query expansion methods. The measurement metric is
recall@3.
5.4 Analysis
Retrieval accuracy.Table 4 compares retrieval
accuracy on HotpotQA using two retrievers,bge
ande5. Among all methods, DPR achieves the
highest retrieval accuracy under both retrievers.
Specifically, DPR improves retrieval accuracy from
61.9% to 62.7% withbge, and from 59.3% to 62.6%
withe5. On the contrary, query expansion meth-
ods such as HyDE, Q2D, and CoT consistently
underperform standard RAG, indicating that naive
expansion may introduce noise and dilute retrieval
relevance. These results demonstrate that the pro-
posed dual-path retrieval strategy is more effective
at identifying relevant evidence than conventional
single-path retrieval or query expansion methods,
providing a stronger foundation for downstream
generation. It should be noted that we only evalu-
ate retrieval accuracy on HotpotQA, as it is paired
with a well-defined corresponding corpus.
Uncertainty scaling.We evaluate the EM im-
provement obtained by activating retrieval relative
to no retrieval across different uncertainty thresh-
olds. As shown in Figure 5, retrieval degrades
accuracy when uncertainty is low, indicating that
unnecessary retrieval introduces noise. As uncer-
tainty increases, the improvement ratio consistently
rises, showing that retrieval becomes increasingly
helpful when parametric knowledge is insufficient.
Model scale significantly affects this trend.
The 72B model exhibits slower gains due to its
strong parametric capability, while the 32B model
achieves the largest improvements, benefiting from
a favorable balance between parametric knowledge
and robustness to retrieval noise. In addition, more
accurate retrieval mechanisms yield larger improve-
ments under the same uncertainty thresholds, as
increased retrieval accuracy (higher Pret) amplifies
Figure 5: Uncertainty scaling results across various
model sizes and different retrieval mechanisms.
Are Ganzhou and Jimo District both located in China?QueryAnswerYesUncertaintyscore:0.0008Ans YesStandardRAGRetrievals1.Ganzhou(disambiguation):Ganzhouisaprefecture-levelcityinJiangxi,China.2.Gantian,Zhuzhou:GanzhouTown()isanurbantowninZhuzhouCounty,ZhuzhouCity,HunanProvince,People'sRepublicofChina.3.GanzhouDistrict:GanzhouDistrict,formerlytheseparatecityofGanzhouorKanchow,isadistrictinandtheseatoftheprefecture-levelcityofZhangyeinGansuProvince…AnsNo
Figure 6: Comparison between uncertainty–guided trig-
gering and standard RAG.
the effectiveness of retrieval. We report all scaling
results in Appendix G.
Case study.Fig. 6 demonstrates a representative
example. The query is a simple factual compar-
ison that lies well within the LLM’s parametric
knowledge, and thus, the model accurately gen-
erates answers correctly with a lower uncertainty
score. However, irrelevant documents retrieved us-
ing the query can lead to an inaccurate answer. We
present more cases in the appendix H.
6 Conclusion
This paper presents DTR, a training-free frame-
work that enhances retrieval–augmented generation
through uncertainty-guided triggering and dual-
path retrieval with adaptive information selection.
By explicitly deciding when retrieval is necessary
and how to leverage complementary retrieval sig-
nals, DTR improves both accuracy and robust-
ness over conventional RAG pipelines. Extensive
experiments demonstrate that DTR consistently
outperforms strong baselines across model scales,
datasets, and retrievers, establishing it as a practical
and scalable solution for adaptive RAG systems.
8

Limitations
While DTR is effective and lightweight, it relies on
the quality of uncertainty estimates derived from
token probabilities, which may vary across de-
coding strategies or model families. In addition,
generating pseudo-contexts introduces extra infer-
ence steps, increasing latency compared to single-
pass retrieval. Finally, although DTR generalizes
well across retrievers, its performance still depends
on the underlying retrieval corpus and embedding
quality.
Ethical Considerations
DTR does not introduce new data sources or train-
ing procedures beyond standard RAG setups, and
thus inherits existing ethical considerations related
to LLMs and information retrieval, such as poten-
tial biases in corpora and misinformation in re-
trieved documents. By reducing unnecessary re-
trievals, DTR may mitigate exposure to irrelevant
or misleading content, but it does not eliminate the
need for careful dataset curation and responsible
deployment.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.Preprint, arXiv:2303.08774.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on freebase from
question-answer pairs. InProceedings of the 2013
conference on empirical methods in natural language
processing, pages 1533–1544.
Christopher Buss, Jasmin Mosavi, Mikhail Tokarev,
Arash Termehchy, David Maier, and Stefan Lee. 2023.
Generating data augmentation queries using large lan-
guage models. InVLDB Workshops.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. Rq-rag: Learn-
ing to refine queries for retrieval augmented genera-
tion.Preprint, arXiv:2404.00610.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Ma-
hashweta Das, and Na Zou. 2025. MAIN-RAG:Multi-agent filtering retrieval-augmented generation.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 2607–2622, Vienna, Austria.
Association for Computational Linguistics.
Wang Chen, Guanqiang Qi, Weikang Li, Yang Li,
Deguo Xia, and Jizhou Huang. 2025a. Pairs:
Parametric-verified adaptive information retrieval
and selection for efficient rag.arXiv preprint
arXiv:2508.04057.
Wang Chen, Wenhan Yu, Guanqiang Qi, Weikang Li,
Yang Li, Lei Sha, Deguo Xia, and Jizhou Huang.
2025b. Cmrag: Co-modality-based visual document
retrieval and question answering.arXiv preprint
arXiv:2509.02123.
Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald.
2025. Kirag: Knowledge-driven iterative retriever for
enhancing retrieval-augmented generation.Preprint,
arXiv:2502.18397.
Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei
Zhao. 2025. Deep think with confidence.arXiv
preprint arXiv:2508.15260.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023a. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. 2023b. Retrieval-
augmented generation for large language models: A
survey.Preprint, arXiv:2312.10997.
Michael Glass, Gaetano Rossiello, Md Faisal Mahbub
Chowdhury, Ankita Rajaram Naik, Pengshan Cai,
and Alfio Gliozzo. 2022. Re2g: Retrieve, rerank,
generate.Preprint, arXiv:2207.06300.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-
r1: Incentivizing reasoning capability in llms via
reinforcement learning.Preprint, arXiv:2501.12948.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui
Wang, and Michael Bendersky. 2023. Query expan-
sion by prompting large language models.Preprint,
arXiv:2305.03653.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2022. Augmenting doc-
ument representations for dense retrieval with in-
terpolation and perturbation. In60th Annual Meet-
ing of the Association-for-Computational-Linguistics
(ACL), pages 442–452. ASSOC COMPUTATIONAL
LINGUISTICS-ACL.
9

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024a. Adaptive-rag:
Learning to adapt retrieval-augmented large lan-
guage models through question complexity.Preprint,
arXiv:2403.14403.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024b. Database-
augmented query representation for information re-
trieval.Preprint, arXiv:2406.16013.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of hal-
lucination in natural language generation.ACM com-
puting surveys, 55(12):1–38.
Mingyi Jia, Junwen Duan, Yan Song, and Jianxin Wang.
2025. Find: Fine-grained information density guided
adaptive retrieval-augmented generation for disease
diagnosis.Preprint, arXiv:2502.14614.
Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, and
Bing Qin. 2025. Gainrag: Preference alignment in
retrieval-augmented generation through gain signal
synthesis.Preprint, arXiv:2505.18710.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611.
Zhewei Kang, Xuandong Zhao, and Dawn Song.
2025. Scalable best-of-n selection for large lan-
guage models via self-certainty.arXiv preprint
arXiv:2502.18581.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha,
and Jinwoo Shin. 2024. Sure: Summarizing re-
trievals using answer candidates for open-domain
qa of llms. InThe Twelfth International Conference
on Learning Representations, Vienna, Austria.Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Victor Lavrenko and W Bruce Croft. 2017. Relevance-
based language models. InACM SIGIR Forum, vol-
ume 51, pages 260–267. ACM New York, NY , USA.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models.Preprint, arXiv:2501.05366.
Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz,
Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun
Chen. 2023. How to train your dragon: Diverse
augmentation towards generalizable dense retrieval.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 6385–6400.
Yuanhua Lv and ChengXiang Zhai. 2009. A compara-
tive study of methods for estimating query language
models with pseudo feedback. InProceedings of the
18th ACM conference on Information and knowledge
management, pages 1895–1898.
Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong
Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen.
2021. Generation-augmented retrieval for open-
domain question answering. InJoint Conference
of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing,
ACL-IJCNLP 2021, pages 4089–4100. Association
for Computational Linguistics (ACL).
QwenTeam. 2024. Qwen2 technical report.Preprint,
arXiv:2407.10671.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev,
and Percy Liang. 2016. Squad: 100,000+ ques-
tions for machine comprehension of text.Preprint,
arXiv:1606.05250.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models.Transactions of the Association for
Computational Linguistics, 11:1316–1331.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented
generation: A survey on agentic rag.Preprint,
arXiv:2501.09136.
10

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
Preprint, arXiv:2503.05592.
W Su, Y Tang, Q Ai, Z Wu, and Y Liu. 2022. Dragin:
Dynamic retrieval augmented generation based on
the real-time information needs of large language
models. arxiv 2024.Preprint, arXiv:2403.10081.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, and 1 others. 2023. Llama: Open and
efficient foundation language models.Preprint,
arXiv:2302.13971.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Liang Wang, Nan Yang, and Furu Wei. 2023a.
Query2doc: Query expansion with large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 9414–9423.
Yile Wang, Peng Li, Maosong Sun, and Yang Liu.
2023b. Self-knowledge guided retrieval augmenta-
tion for large language models. InThe 2023 Con-
ference on Empirical Methods in Natural Language
Processing.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.Preprint,
arXiv:2309.07597.
Shi-Qi Yan and Zhen-Hua Ling. 2025. Rpo: Re-
trieval preference optimization for robust retrieval-
augmented generation.Preprint, arXiv:2501.13726.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 oth-
ers. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms.Advances in
Neural Information Processing Systems, 37:121156–
121184.Wenzheng Zhang, Xi Victoria Lin, Karl Stratos,
Wen-tau Yih, and Mingda Chen. 2025a. Imprag:
Retrieval-augmented generation with implicit queries.
Preprint, arXiv:2506.02279.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025b.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.Preprint,
arXiv:2506.05176.
Zhen Zhang, Xinyu Wang, Yong Jiang, Zile Qiao,
Zhuo Chen, Guangyu Li, Feiteng Mu, Mengting Hu,
Pengjun Xie, and Fei Huang. 2025c. Kbm: Delin-
eating knowledge boundary for adaptive retrieval in
large language models. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 21771–21782.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024.
Retrieval-augmented generation for ai-generated con-
tent: A survey.Preprint, arXiv:2402.19473.
Zhi Zheng, Kai Hui, Ben He, Xianpei Han, Le Sun,
and Andrew Yates. 2020. Bert-qe: Contextualized
query expansion for document re-ranking. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2020, pages 4718–4728.
11

A Additional related work
How to retrieve and select relevant information is
a key issue in RAG systems. With the remarkable
success of advancing LLMs using reinforcement
learning (RL), many studies have adopted RL to
train LLMs to enhance information retrieval and
reranking (Asai et al., 2024; Jin et al., 2025; Li
et al., 2025; Song et al., 2025; Yu et al., 2024). In
addition, several studies (Jiang et al., 2025; Yan
and Ling, 2025) have aligned the retriever’s pref-
erences with those of LLMs to retrieve more rel-
evant information and enhance generation perfor-
mance. These methods could achieve significant
improvement, but may suffer from high computa-
tional costs. In addition, with the accessibility to
advanced reranking models, such as bge-reranker
(Xiao et al., 2023) and Qwen3-reranker (Zhang
et al., 2025b), a straightforward method is first to
rerank the retrievals using a reranker and then se-
lect the top- krelevant retrievals (Glass et al., 2022;
Chang et al., 2025). This method could further
select retrievals with a higher semantic similarity
with the query and thus enhance the generation per-
formance. However, it may not perform well or
even degrade answer accuracy in QA tasks due to
the sparse query or large corpus (Kim et al., 2024;
Jiang et al., 2025).
B Notation list
The main notations and abbreviations used in this
paper are listed in Table 5.
C Retrieval analysis
In this section, we further analyze the properties
of retrievals using the query or LLM–generated
pseudo-context. Specifically, we first compare
the documents retrieved using the query with the
ground-truth (GT) documents. We found that the
retrieved documents may be significantly different
from the GT documents. In addition, we analyze
the spatial distribution of query–based, pseudo-
context–based, and GT documents. We discov-
ered that the documents retrieved using the pseudo-
context may compensate for this shortcoming.
C.1 Similarity between query and GT
documents
To figure out how the query–based documents dif-
fer from the GT documents, we first calculated
the similarity scores between the ground-truth(GT)
documents and the queries in the HotpotQA dataset.Notation Explanation
Pparam Probability of accurate parametric generation
Pret Probability of accurate retrieval
Pgen Probability of accurate final generation
DChunks of Documents
Dk Retrievedkchunks
fEmbedding model
qQuery
qQuery embedding
dA chunk
dChunk embedding
RRetriever
PPrompt
GLLM generator
ˆaGenerated answer
pLLM–generated pseudo-context
pPseudo-context embedding
θ0 The angle betweenqandp
θ1 The angle betweenqandd
θ2 The angle betweenpandd
s(d)Score ofd
Abbreviation Explanation
DTR Decide then retrieve
UGT Uncertainty–guided triggering
DPR Dual-path retrieval
AIS Adaptive information selction
IP Inner product
EM Exact match
Table 5: Explanation of main notations and abbrevia-
tions used in this study.
As shown in Fig. 7, the similarity scores of the most
GT documents are below 0.8, indicating that the
query may differ from the GT documents, which
can lead to irrelevant retrievals.
0.3 0.4 0.5 0.6 0.7 0.8 0.9
GT Document Scores02004006008001000FrequencyGT Document Scores Distribution
Figure 7: Similarity scores between queries and ground
truth documents.
Furthermore, we ranked the GT documents in
the query–based retrievals. As shown in Fig. 8, a
significant proportion of the GT documents rank
20+ in the documents retrieved using the query.
This further demonstrates that using the query to re-
trieve documents only can lead to irrelevant results,
which in turn result in inaccurate final answers.
12

01234567891011121314151617181920+
Ranking010002000300040005000FrequencyGT Document RankingFigure 8: Ranking of ground truth documents in query–
based retrievals.
C.2 Spatial distribution of query, context, and
GT documents
To figure out the relationship between the query,
context, and GT documents, we projected these
documents into the 2D space. Figs. 9 and 10 illus-
trate their relationship in xy and polar coordinates,
respectively. Remind that we only visualize 200
samples, and we set the origin of the coordinates as
the query. Most of the query documents are located
besides the origin (i.e., the query), while the GT
documents are kind of far from the origin, demon-
strating that the GT documents may differ from
the query documents in the vector space. How-
ever, the context documents can compensate for
this gap as their distribution is similar to that of the
GT documents when they are far away from the
origin. These results further demonstrate that the
LLM–generated pseudo-context can compensate
for the sparse queries, enhancing the performance
of RAG systems.
1.0
 0.5
 0.0 0.5 1.01.0
0.5
0.00.51.0PCA in XY Coordinates
Query docs
Context docs
GT docs
Query
Figure 9: Relationship between query, context, and GT
documents in the XY coordinate.
D Prompt template
In this section, we list all prompt templates used in
this study.
1.0
 0.5
 0.0 0.5 1.01.00
0.75
0.50
0.25
0.000.250.500.751.00PCA in Polar Coordinates
Query docs
Context docs
GT docs
QueryFigure 10: Relationship between query, context, and GT
documents in the polar coordinate.
E Additional Uncertainty Measures
Beyond threshold-based analysis, we further in-
vestigate uncertainty by ranking queries accord-
ing to their uncertainty scores and progressively
selecting subsets of queries with the lowest uncer-
tainty. Specifically, thequery ratiodenotes the pro-
portion of queries whose uncertainty is no greater
than a given uncertainty threshold. Figure 16 re-
ports the Exact Match (EM) and F1 scoreswith-
out retrievalacross five benchmark datasets as the
query ratio increases from 10% to 100%. Across
all datasets and model scales, a consistent trend
emerges: queries with lower uncertainty achieve
substantially higher EM and F1 scores, while in-
cluding more high-uncertainty queries gradually
degrades performance. This observation holds for
both factoid (e.g., TriviaQA, SQuAD) and multi-
hop or open-domain settings (e.g., HotpotQA, We-
bQA).
Moreover, larger models consistently dominate
smaller ones at the same query ratio, indicating that
stronger LLMs not only achieve higher parametric
accuracy but also produce more reliable uncertainty
estimates. Importantly, even when answering only
a small fraction of low-uncertainty queries (e.g.,
top 20%–30%), the models can retain a large por-
tion of their maximum achievable accuracy, sug-
gesting that uncertainty effectively captures answer
correctness.
F Additional evaluation results
In this section, we report more evaluation results,
including those with top-5 retrievals or with e5 as
the retriever.
13

Prompt for answer generation without retrievals
{Question}
Answer the question using a single word or phrase.
Figure 11: Prompt used for generating final answers without retrievals.
Prompt for answer generation with retrievals
{Question}
<Passage_1>
<Passage_2>
···
<Passage_k>
Answer the question based on the above context using a single word or phrase.
Figure 12: Prompt used for generating final answers with retrievals.
Prompt for pseudo context generation
{Question}
Write a passage to answer this question.
Figure 13: Prompt used for generating pseudo context.
Prompt for CoT generation
Answer the following question:
{Question}
Give the rationale before answering
Figure 14: Prompt used for generating CoT.
Prompt for retrieval judge
{Question}
Determine whether external information is needed to answer the question accurately. Respond
with "Yes" if additional information is required, or "No" if the question can be answered without it.
Figure 15: Prompt used for determining whether to retrieve.
F.1 Results with top-5 retrievals.
Table 6 reports the results when increasing the num-
ber of retrieved passages to top-5, usingQwen2.5-
7B-Instructas the generator. Overall, similar trends
to the top-3 setting are observed, while perfor-
mance is consistently improved across most meth-
ods due to the increased evidence coverage.Overall performance.DTR remains the best-
performing method under top-5 retrievals, achiev-
ing the highest average EM/F1 across all un-
certainty thresholds. Compared with standard
RAG, DTR improves the average EM/F1 from
37.30/47.60 to up to 38.78/49.28, demonstrating
that uncertainty-guided triggering continues to be
effective even when more retrieved passages are
14

Figure 16: Evaluation results (EM and F1) across five datasets withQwen2.5series models as the generators. The
percentage of questions is divided by the uncertainty scores of the generated answers. Without external retrievals, a
lower uncertainty score can lead to higher generation accuracy.
provided.
Effect of increased retrieval depth.While stan-
dard RAG benefits from additional retrieved pas-
sages, it also becomes more susceptible to noise in-
troduced by less relevant evidence. On the contrary,
DTR consistently yields higher accuracy, indicat-
ing that selectively triggering retrieval and adap-
tively leveraging retrieved information is crucial
for mitigating noise in long-context settings.
Impact of uncertainty thresholds.Varying the
uncertainty threshold ucontrols the trade-off be-
tween retrieval frequency and accuracy. Smaller
thresholds result in higher trigger ratios and slightly
better performance on reasoning-intensive datasets
such as HotpotQA and NaturalQA, whereas larger
thresholds reduce retrieval usage with minimal per-
formance degradation. These results further con-firm the robustness of DTR across different re-
trieval depths.
Overall, the top-5 retrieval results reinforce our
main findings: dynamic, uncertainty-guided trig-
gering with dual-path retrieval remains effective
and robust as the retrieval depth increases.
F.2 Results with e5 as the retriever.
Table 7 reports the main results when replacingbge
withe5as the retriever, while keepingQwen2.5-7B-
Instructas the generator and using top-3 retrieved
passages. Overall, the performance trends are con-
sistent with those observed usingbge, demonstrat-
ing that DTR generalizes well across different re-
trieval models.
Overall performance.DTR consistently
achieves the best or second-best performance
across all datasets. In particular, DTR improves
15

MethodHotpotQA NaturalQA TriviaQA SQuAD WebQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
No Retrieval 18.73 25.38 17.40 24.95 43.24 49.12 12.20 18.63 22.59 35.03 22.83 30.62
Standard RAG 37.54 48.4036.45 46.23 57.33 65.21 30.01 38.84 25.15 39.29 37.30 47.60
LLM Judge (7B) 18.77 25.48 17.40 24.95 43.27 49.15 12.21 18.64 22.59 35.03 22.85 30.65
Trigger Ratio (0.5%) (0.0%) (0.1%) (0.0%) (0.0%) (0.1%)
LLM Judge (72B) 35.80 46.32 27.89 36.78 50.45 56.92 26.98 35.21 24.11 36.73 33.05 42.39
Trigger Ratio (88.7%) (43.7%) (27.8%) (70.0%) (26.9%) (51.4%)
HyDE 34.17 43.86 36.70 46.35 56.25 64.15 26.27 34.83 27.07 41.00 36.09 46.04
Q2D 35.81 45.89 36.87 47.15 58.48 66.34 29.40 38.39 26.03 40.18 37.32 47.59
CoT 35.92 45.94 37.17 47.46 58.61 66.42 29.21 37.89 26.18 40.54 37.42 47.65
DTR (u= 0.001) 37.61 48.30 38.50 48.63 60.30 68.29 30.99 39.92 26.48 41.23 38.78 49.28
Trigger Ratio (87.7%) (94.8%) (90.8%) (96.8%) (91.8%) (92.4%)
DTR (u= 0.005) 37.25 47.88 37.95 48.13 60.55 68.36 30.85 39.76 27.21 41.90 38.76 49.20
Trigger Ratio (83.5%) (90.4%) (83.0%) (93.6%) (84.5%) (87.0%)
DTR (u= 0.01) 37.12 47.70 37.67 47.95 60.70 68.39 30.69 39.53 27.56 42.10 38.75 49.13
Trigger Ratio (81.0%) (87.5%) (78.7%) (91.2%) (81.1%) (83.9%)
Table 6: Main results across five QA datasets withQwen2.5-7B-Instructas the generator andbgeas the retriever,
respectively. Except for experiments that require no retrievals, all results are generated based ontop-5 retrievals.
Trigger Ratioindicates the proportion of queries for which the retriever was triggered.Boldand underlined values
represent the best and second-best scores, respectively.
MethodHotpotQA NaturalQA TriviaQA SQuAD WebQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
No Retrieval 18.73 25.38 17.40 24.95 43.24 49.12 12.20 18.63 22.59 35.03 22.83 30.62
Standard RAG 35.18 45.77 37.56 47.29 59.00 66.99 28.84 37.43 23.72 38.43 36.86 47.18
LLM Judge (7B) 18.77 25.46 17.40 24.95 43.27 49.15 12.21 18.65 22.59 35.03 22.85 30.65
Trigger Ratio (0.5%) (0.0%) (0.1%) (0.0%) (0.0%) (0.1%)
LLM Judge (72B) 33.60 43.83 28.53 37.66 51.26 57.82 26.37 34.44 24.16 36.93 32.78 42.14
Trigger Ratio (88.7%) (43.7%) (27.8%) (70.0%) (26.9%) (51.4%)
HyDE 32.64 42.40 35.32 44.92 55.31 63.17 24.52 32.62 25.69 39.40 34.70 44.50
Q2D 34.54 44.44 35.93 45.87 57.48 65.29 27.34 35.93 25.39 39.62 36.14 46.23
CoT 34.44 44.57 36.18 45.89 57.39 65.26 27.09 35.57 25.84 39.99 36.19 46.25
DTR (u= 0.001) 36.73 47.34 38.84 49.22 61.26 69.40 29.11 37.68 26.18 40.65 38.42 48.86
Trigger Ratio (87.7%) (94.8%) (90.8%) (96.8%) (91.8%) (92.4%)
DTR (u= 0.005) 36.29 46.87 38.12 48.64 61.54 69.49 29.13 37.65 26.97 41.41 38.41 48.81
Trigger Ratio (83.5%) (90.4%) (83.0%) (93.6%) (84.5%) (87.0%)
DTR (u= 0.01) 36.21 46.71 37.92 48.45 61.61 69.46 29.02 37.47 27.36 41.57 38.42 48.73
Trigger Ratio (81.0%) (87.5%) (78.7%) (91.2%) (81.1%) (83.9%)
Table 7: Main results across five QA datasets withQwen2.5-7B-Instructas the generator ande5as the retriever,
respectively. Except for experiments that require no retrievals, all results are generated based ontop-3 retrievals.
Trigger Ratioindicates the proportion of queries for which the retriever was triggered.Boldand underlined values
represent the best and second-best scores, respectively.
the average EM/F1 from 36.86/47.18 (standard
RAG) to up to 38.42/48.86, confirming that
uncertainty-guided dynamic triggering remains
effective even with a different retriever.
Robustness across datasets.Across reasoning-
intensive datasets such as HotpotQA and Natu-
ralQA, DTR yields clear gains over standard RAG
and retrieval-enhanced prompting methods. On
TriviaQA and WebQA, where retrieval noise is
more prominent, DTR maintains strong perfor-
mance by selectively activating retrieval, highlight-ing its robustness to retriever variability.
Effect of uncertainty thresholds.As withbge,
varying the uncertainty threshold uallows DTR to
balance retrieval frequency and accuracy. Smaller
thresholds lead to higher trigger ratios and slightly
better overall performance, while larger thresholds
reduce retrieval usage with minimal degradation.
This further confirms that uncertainty-based trigger-
ing provides a stable and retriever-agnostic control
mechanism.
Overall, the results withe5demonstrate that
16

Figure 17: Uncertainty scaling results across various model sizes.
DTR is not tied to a specific retriever and can con-
sistently improve RAG performance across differ-
ent retrieval backbones.
G Additional uncertainty scaling results
In this section, we present all uncertainty scaling
results across five benchmarks, various model sizes,
and different retrieval mechanisms, as shown in
Figures 17 and 18.H Additional case studies
Figure 19 compares the retrievals and answers
achieved by standard RAG and DPR-AIS, respec-
tively. Using the query to retrieve documents only
can lead to similar but irrelevant results, which
in turn results in a wrong final answer. However,
our proposed dual-path retrieval mechanism can
compensate for sparse queries and retrieve more
relevant information, ultimately leading to accurate
answers.
17

Figure 18: Uncertainty scaling results across different retrieval mechanisms.
I Use of LLMs
In the preparation of this paper, large language
models (LLMs) were used solely for the purpose
of polishing the writing, including grammar cor-
rection, improving sentence fluency, and ensuring
a consistent academic tone. All core intellectual
content—including the conceptualization of the
proposed method, the design and execution of ex-
periments, the analysis and interpretation of results,
and the conclusions drawn—is the original work ofthe authors. The authors take full responsibility for
the entire content of this paper, including any text
generated with the assistance of LLMs.
18

Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?QueryAnswer2000StandardRAGRetrievals1.SevenBriefLessonsonPhysics:SevenBriefLessonsonPhysics(Italian:"")isashortbookbytheItalianphysicistCarloRovelli.OriginallypublishedinItalianin2014,…2.EttoreMajorana:EttoreMajorana(bornon5August1906\u2013probablydiedafter1959)wasanItaliantheoreticalphysicistwhoworkedonneutrinomasses.…3.CarloBecchi:CarloMariaBecchi(born20October1939)isanItaliantheoreticalphysicist.Ans 2014DPR-AIS Retrievals1.SevenBriefLessonsonPhysics:SevenBriefLessonsonPhysics(Italian:"")isashortbookbytheItalianphysicistCarloRovelli.OriginallypublishedinItalianin2014,…2.CarloRovelli:CarloRovelli(born3May1956)isanItaliantheoreticalphysicistandwriterwhohasworkedinItaly,theUnitedStatesandsince2000,inFrance.…3.GabrieleVeneziano:GabrieleVeneziano(born7September1942)isanItaliantheoreticalphysicistandoneofthepioneersofstringtheory.…Ans2000
Figure 19: Comparison between standard RAG and DPR-AIS.
19
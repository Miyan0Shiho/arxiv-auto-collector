# POLYRAG: Integrating Polyviews into Retrieval-Augmented Generation for Medical Applications

**Authors**: Chunjing Gan, Dan Yang, Binbin Hu, Ziqi Liu, Yue Shen, Zhiqiang Zhang, Jian Wang, Jun Zhou

**Published**: 2025-04-21 07:35:24

**PDF URL**: [http://arxiv.org/pdf/2504.14917v1](http://arxiv.org/pdf/2504.14917v1)

## Abstract
Large language models (LLMs) have become a disruptive force in the industry,
introducing unprecedented capabilities in natural language processing, logical
reasoning and so on. However, the challenges of knowledge updates and
hallucination issues have limited the application of LLMs in medical scenarios,
where retrieval-augmented generation (RAG) can offer significant assistance.
Nevertheless, existing retrieve-then-read approaches generally digest the
retrieved documents, without considering the timeliness, authoritativeness and
commonality of retrieval. We argue that these approaches can be suboptimal,
especially in real-world applications where information from different sources
might conflict with each other and even information from the same source in
different time scale might be different, and totally relying on this would
deteriorate the performance of RAG approaches. We propose PolyRAG that
carefully incorporate judges from different perspectives and finally integrate
the polyviews for retrieval augmented generation in medical applications. Due
to the scarcity of real-world benchmarks for evaluation, to bridge the gap we
propose PolyEVAL, a benchmark consists of queries and documents collected from
real-world medical scenarios (including medical policy, hospital & doctor
inquiry and healthcare) with multiple tagging (e.g., timeliness,
authoritativeness) on them. Extensive experiments and analysis on PolyEVAL have
demonstrated the superiority of PolyRAG.

## Full Text


<!-- PDF content starts -->

POLYRAG: Integrating Polyviews into Retrieval-Augmented Generation
for Medical Applications
Chunjing Gan Dan Yang Binbin Hu Ziqi Liu Yue Shen
Zhiqiang Zhang Jian Wang Jun Zhou†
Ant Group
jun.zhoujun@antgroup.com
Abstract
Large language models (LLMs) have become
a disruptive force in the industry, introduc-
ing unprecedented capabilities in natural lan-
guage processing, logical reasoning and so on.
However, the challenges of knowledge updates
and hallucination issues have limited the appli-
cation of LLMs in medical scenarios, where
retrieval-augmented generation (RAG) can of-
fer significant assistance. Nevertheless, exist-
ing retrieve-then-read approaches generally di-
gest the retrieved documents, without consider-
ing the timeliness, authoritativeness and com-
monality of retrieval. We argue that these ap-
proaches can be suboptimal, especially in real-
world applications where information from dif-
ferent sources might conflict with each other
and even information from the same source
in different time scale might be different, and
totally relying on this would deteriorate the per-
formance of RAG approaches. We propose
POLYRAGthat carefully incorporate judges
from different perspectives and finally inte-
grate the polyviews for retrieval augmented
generation in medical applications. Due to the
scarcity of real-world benchmarks for evalu-
ation, to bridge the gap we propose POLYE-
VAL, a benchmark consists of queries and doc-
uments collected from real-world medical sce-
narios (including medical policy, hospital &
doctor inquiry and healthcare) with multiple
tagging ( e.g., timeliness, authoritativeness) on
them. Extensive experiments and analysis on
POLYEVAL have demonstrated the superiority
of P OLYRAG1.
1 Introduction
Recently, large language models (LLMs) such as
GPT4 (OpenAI, 2023), Llama3 (Grattafiori et al.,
2024), Qwen (Yang et al., 2024), Deepseek-R1
(DeepSeek-AI et al., 2025) have become a disrup-
tive force in the industry, which introduces mar-
velous capabilities in natural language process-
1We will release the data of P OLYEVAL soon.
Can Sodium Hyaluronate and Pranoprofen Eye Drops be Used Together?Based on my personal experience over the past two days: No, using both eye drops together causes a noticeable stinging sensation, but using them separately does not.www.xxonlinesbs.com 2019The symptoms may be caused by dry eye syndrome or ocular inflammation. It is recommended to use Sodium Hyaluronate and Pranoprofen eye drops for treatment.www.xx_commercial.comWhen using multiple types of eye drops simultaneously, apply the less irritating ones first, followed by the more irritating medications.www.xx.gov.cn 2020No, using both eye drops together would cause a noticeable stinging sensation, but using them separately does not.If the discomfort is caused by dry eye syndrome or ocular inflammation, Sodium Hyaluronate and Pranoprofen eye drops can be used for treatment, but they should be administered with a time interval between them to avoid interactions.www.xx.med.com 2020If it is allergic conjunctivitis, it is recommended to primarily use Sodium Hyaluronate eye drops and secondarily use Pranoprofen eye drops for inflammation. You can discontinue Pranoprofen once the redness subsides. If you experience any discomfort, please seek medical attention promptly.www.xx_medqa.com 2023When using multiple types of eye drops simultaneously, apply the less irritating ones first, followed by the more irritating medications.www.xx.gov.cn 2020For conditions such as dry eye syndrome, ocular inflammation, and allergic conjunctivitis, both medications can be used together, but it is important to ensure an interval between applications to avoid interactions. If you experience any discomfort, please seek medical attention promptly.
Figure 1: A toy example illustrating the difference be-
tween traditional retrieval and our retrieval strategy,
where beyond relevance of a document, we also takes
other perspectives such as its authoritativeness into con-
sideration.
ing (Mallen et al., 2023), logical reasoning (Patel
et al., 2024), multi-modal processing (Zhang et al.,
2024a) and so on. However, the heavy costs of
knowledge updates (Shi et al., 2024a) and the long-
standing hallucination issues (Gao et al., 2023a)
have limited the application of LLMs in medical
scenarios where incorrect answers may result in se-
vere consequences, in this case retrieval-augmented
generation (RAG) can be of help. Nevertheless,
existing retrieve-then-read approaches generally
directly digest the documents from the retrieval
stages (Asai et al., 2024), without considering other
perspectives such as timeliness, authoritativeness
and commonality of retrieval.
Here, we argue that oftentimes these approaches
can be suboptimal, especially in real-world appli-
cations ( e.g., medical applications) where not only
information from different sources with respect to
the same fact might conflict with each other but
also information from the same source in different
time scale might be different, and directly rely-
ing on them for generation would deteriorate the
performance of RAG approaches. As the toy ex-
ample shown in Figure 1, when a user types inarXiv:2504.14917v1  [cs.LG]  21 Apr 2025

the query “Can Sodium Hyaluronate and Pranopro-
fen Eye Drops be Used Together?”, a traditional
RAG system would search and rank documents
according to its relevance to the query (Shi et al.,
2024a). Though the retrieved documents comes
from non-authoritative websites and even contra-
dicts with each other such that the LLM used for
generation struggles in incorporating the retrieved
information, e.g., the first document just states they
cannot be used together but separately without fur-
ther context, the second document states they can
be used for treating dry eye syndrome or ocular
inflammation while the third document states the
order of usage, however, various discussions held
on this topic do not result in a definitive conclusion
which finally hinders its effectiveness for question
answering. Not to mention that for some complex
queries that contains multiple factors, the top re-
trieved documents may only contains facts focusing
on one factor and ignores documents with respect
to other factors, which would severely hinder the
performance.
Given the above limitations in current ap-
proaches, instead of solely relying on the relevance
of documents for generation, we aim to integrate
polyviews ( i.e.,multiple views w.r.t. retrieval such
as utility, complement, authoritativeness, timeli-
ness and composibility) into consideration so as
to promote its application in medical applications.
However, the solution is quite non-trivial, which
needs to tackle the following challenges: ( C1)
With multiple views to evaluate, how to measure
them and its feasibility in real-world applications
remains unknown. ( C2) With the evaluated re-
sults of multiple views, in real-world applications
what we needed is actually an integrated scoring
strategy that comprehensively evaluates each view,
how to develop a reasonable and applicable ranking
strategy to combine the precedent views remains
unanswered. ( C3) The lack of benchmark data that
evaluates the retrieval performance of a model from
multiple views prohibits us from further developing
our model.
To this end, we propose POLYRAG. In particu-
lar, given that there are many available small but
performant models, we carefully allocate storage
to make this modeling feasible. ( C1) To compre-
hensively integrate the results of each view, we
transform the modeling of ranking strategy to a
multi-reward problem and find the mixture of dif-
ferent views. ( C2) Due to the scarcity of real-world benchmarks for evaluation, to bridge the
gap we propose POLYEVAL, which is a benchmark
consists of queries and documents collected from
real-world healthcare scenarios (including medi-
cal policy, hospital recommendation and medical
care) with multiple tagging ( e.g., timeliness, au-
thoritativeness) on them ( C3). With the polyviews
gained from the precedent procedures, we apply
the retrieved top-k documents and call an LLM
for knowledge-augmented generation. We evalu-
ate the proposed POLYRAGon multiple tasks and
extensive experiments and analysis on POLYEVAL
have demonstrated the superiority of the proposed
POLYRAG.
2 Related Work
Retrieval-augmented generation (RAG) approaches
which empower large language models (LLMs)
with additional knowledge and henceforth less need
for additional training (Gao et al., 2023b; Fan et al.,
2024; Gupta et al., 2024; Nguyen et al., 2024) have
been successfully applied to various fields(Sun
et al., 2023; Zhang et al., 2024b; Shi et al., 2024b;
Golatkar et al., 2024; Zhao et al., 2024) includ-
ing recommender systems(Contal and McGoldrick,
2024; Rao and Lin, 2024; Zeng et al., 2024), ques-
tion answering(Asai et al., 2024; Wang et al., 2025)
and so on. Among them, question answering in
medical applications poses significant challenges
due to their high professionalism and low fault-
tolerance characteristics. Existing approaches for
medical-based RAG have been studying additional
knowledge acquisition(Jin et al., 2023; Wang et al.,
2024), query construction(Chen et al., 2025; Sohn
et al., 2024), complex retrieval strategy(Wu et al.,
2024; Xiong et al., 2024; Tang et al., 2025), com-
plex reasoning(Verma et al., 2025; Li et al., 2024;
Zafar et al., 2025) and so on with focus on better
retrieval strategy from external source and better
utilization strategy when employ LLMs for answer
generation.
Open issues. Few research works consider multi-
ple perspectives of the retrieval results and in this
work we delve into a direction that can be directly
integrated into these existing pipelines where we
investigate on how to incorporate retrieval from
polyviews for downstream tasks and henceforth
promoting retrieval.

Online SearchNewsKnowledge BaseExpert KnowledgeUser Query
Topic 1
Topic 2Topic 3Topic 4Topic n
User QueryTop-k Docs from Different Topics
+Answer GenerationOutput
Figure 2: The proposed P OLYRAGframework.
3 The Proposed Approach
3.1 Overview
The task of retrieving top critical documents from
previous searching and filtering stage is equiva-
lent to comprehensively evaluate the input docu-
ments, i.e.,evaluate the retrieved document from
mpolyviews V. For simplicity, with the assump-
tion that multiple polyviews are independent, given
an input query q, a document d(d∈ D =
{d1, d2, ..., d n}), where we first evaluate each doc-
ument independently as follows:
P(dj| V1,j, . . . ,Vm,j) =mY
i=1(P(dj| Vi,j))wi,
(1)
where Vi,j,widenote the jth document evaluate
from the ith view regarding the input query q, the
weight of ith view respectively. Given some pre-
defined constraints C, we can obtain top-ranking
documents DTop:
DTop={d∈ D s.t.C} (2)
In this work, we propose POLYRAG, as shown in
Figure 2. With the multi-source searching and filter-
ing results, POLYRAGfirstly embrace varied views
for evaluation of each retrieved document (detailed
in Section 3.2) and further pursuing integrated
polyviews via a multi-rewards based view-mixture
mechanism (detailed in Section 3.3), then incorpo-
rating the derived polyview-grounded knowledge
for answer generation (detailed in Section 3.4).
3.2 Through Different Lenses: A Document
Evaluated via Polyviews
In this paper, we pre-define 6polyviews, i.e., Rel-
evance (R),Utility (U),Supplement (S),Authori-
tativeness (A),Timeliness (T),Composibility (C,which is used as a retrieval constraint) and detail
the estimation of each in the following.
Relevance View is a case of symmetric retrieval,
which is designed to be direction-agnostic. With
an off-the-shelf model E(which could be a dense
retriever followed by a predefined metric Msuch
as cosine similarity for simplicity or large language
models by designing instruction INSR), we can
efficiently obtain the Relevance score between the
query and document as follows:
R(q, d) =PLLM(d|q,INSR),with LLM;
M(E(q),E(d)),otherwise.
(3)
However, Relevance cannot guarantee usefulness,
where we introduce asymmetric retrieval i.e., Util-
ityView that measures the extent that one document
is useful for assisting an LLM to answer the given
query, which is modelled by the probability of gen-
erating correct answer awith a specific LLM, by
designing an appropriate instruction INSUto guide
the LLM, we can calculate the Utility of a docu-
ment w.r.t. the input query as follows:
U(d|q, a) =PLLM(a|q, d,INSU). (4)
Oftentimes there are documents that do not directly
answer the query but they can provide additional
knowledge, background information, or alterna-
tives that help users to make more informed deci-
sions or better understand the treatment process,
where we define it as the Supplement View of a
document w.r.t. the input query, with a carefully
designed INSSto guide the LLM for estimating
Supplement , we can formalize it as follows:
S(d|q) =PLLM(d|q,INSS). (5)
Besides, given the retrieved documents from pre-
vious stage, it is of great significance to take

into account the Authoritativeness andTimeliness
Views of them, since that for scenarios with strong
professionalism, i.e.,medical applications in our
case, medical treatments recommended by different
sources, such as professional doctors and individ-
ual accounts, can vary greatly. Additionally, med-
ical policies and practices may evolve over time.
Therefore, keeping track of these two dimensions
is crucial and here we denote these two dimensions
of document dasA(d)andT(d)2. Moreover, the
retrieved documents might cover multiple topics
w.r.t. the input query and directly ranking may lead
to top documents focusing on partial topics, there-
fore, we introduce Composibility View to account
for the difference of topics among them, where
the topic of each document can be assigned via
an LLM or clustering algorithms to maximize its
assigning probability as follows:
Cd= arg max
kP(Ck|di)≈arg max
kP(di|Ck)P(Ck).
(6)
3.3 A Cord of Three Strands is Not Quickly
Broken: Multi-rewards Boosted Polyview
Integration
Given the polyview evaluation results, to efficiently
incorporate them for downstream generation, mo-
tivated by the idea and marvelous performance in
simple rewards-driven reinforcement learning, here
we model the integration as multi-rewards integra-
tion to obtain an effective mixture of polyviews, for
each document dfromD, the polyview integration
score can be formalized as follows:
yd=α1dR+α2dU+α3dS+α4dA+α5dT,(7)
where the coefficients can be obtained either by ex-
pertise designation or learning from models. With
the polyview integrated score, we can obtain the
top-ranking documents DTopunder the Composibil-
ityconstraints so that top-ranking documents can
cover different topics w.r.t. the input query:
Cd, d∈ D Top≈ ∥C d, d∈ D∥ . (8)
3.4 Polyview-grounded Generation
With the input query qand the polyview-grounded
knowledge Pthat scatter across different topics
related to the query, we can directly call an LLM
2We approximate A(d)viaA(dsource )for simplicity to
reduce tagging costs, where the A(dsource )is annotated by
human annotators. For T(d), we employ efficient tool for date
extraction.(it can also be fine-tuned in a supervised manner),
where its knowledge-augmented generation output
ocan be formalized as follows:
o∗= arg max
oP(o|q,P), (9)
whereP(o|q,P)is the probability of the output
ogiven the query qand the external documents
P, and arg max denotes the argument of the max-
imum, i.e., the answer ofor which P(o|q,P)is
maximized.
4 Benchmark
We will first describe the characteristics of POLYE-
VAL and then delve into its creation process.
4.1 Characteristics
To ensure that POLYEVAL can be representatives
of real-world medical application user cases, we
carefully design it to be diverse in the following
three perspectives.
•Domain Type :POLYEVAL contains questions
from diverse domains including Medical Policy,
Healthcare, Hospital & Doctor Inquiry in order
to cover different real-world medical scenarios.
•Query Intent : Given questions in each domain,
they encompass various types of real user intents,
e.g., Medical Insurance Balance in Medical Pol-
icy domain, Medication Inquiry in Healthcare
domain in order to comprehensively represent
user needs.
•Annotation Dimension : Given a query, for each
retrieved document, it is annotated with tags on
relevance ,complement ,utility ,publish date and
authority level .
4.2 Benchmark Creation
4.2.1 Data Collection
We collect 1,447 real-world user queries from
a large-scale online platform that offers medical-
related services in China, where its distribution of
domain type and query intent is illustrated in Fig-
ure 33. Given each query, we perform multi-source
(including expert knowledge, online search engine,
knowledge bases and news) documents searching
to find relevant documents for annotation. In sum,
we have collected 21,276documents, making 14.7
documents for each query on average.
3Due to space limit, we only used the first-level categories
when drawing the query intent distribution. In total, there are
40 labels when considering the second-level categories.

36.35%
39.60%24.05%Domain Type
Hospital & Doct or Inquiry
Healthcare
Health Insurance
31.59%
28.80%16.58%10.82%6.63%5.58%Healthcare
Disease Education
Symptom Consultation
Medication Inquiry
Indicator Interpretation
Surgical Procedure Education
Vaccine Information
65.78%13.88%10.46%9.70%0.19%Hospital & Doctor inquiry
Doctor Information Inquiry
Doctor Recommendation
Hospital Information Inquiry
Healthcare Rec ommendation
Department Guidance
32.76%
20.69%16.09%10.63%5.46%3.16%2.59%2.30%2.30%1.72%1.15%0.57%0.57%Medical Policy
Medical Insurance Reimbursement
Cross-region Medical Care
Enrollment and Payment
Family A ssistance
Transfer and Continuation of Coverage
Others
Designated Medical Institutions
Family A ccount
Medical Insurance Balance
Medical Insurance Suspension/Cancellation
Electronic Medical Insurance
Cross-region Medical Reimbursement
Medical Assistance
 (a)                                                                                            (b)                           (c)                                                                   (d)                   Figure 3: Data distribution of POLYEVAL, where Figure (a) denotes the domain type distribution and Figure (b-d)
denote the query intent distribution within each domain.
4.2.2 Annotation Details
Overall, POLYEVALis annotated by human annota-
tors or automated tools. For each query and its asso-
ciated documents, three highly-skilled annotators
who have received professional medical training
are involved for document relevance ,complement
andutility annotation and the annotation result is
“accepted” if at least two annotators reach an agree-
ment unless it is “rejected”. For authority level
of document, we approximate it via the authority
level of its source, i.e.,we firstly collect abundant
information from multiple sources such as medical-
related websites and random sample information
from them, and then ask human annotators to judge
the overall authority of these sources and finally
come up with the authority level . For publish date
of document, we employ efficient automated tools
for date extraction.
5 Experiments
5.1 Experimental Setup
5.1.1 Tasks
We evaluate our proposed POLYRAGand multiple
baselines for retrieval and generation on POLYE-
VAL and evaluate the performance of retrieval via
metrics HIT, NDCG, and generation via judge
model ( e.g., GPT 4). To better demonstrate the
difference between domains in POLYRAG, we de-
note data of domain Healthcare, Hospital & Doctor
Inquiry, Medical Policy as CARE ,INQUIRY and
POLICY respectively for simplicity.
5.1.2 Baselines
We evaluate models augmented with retrieval via
publicly available retrieval model including BM25,
GTE (Li et al., 2023), BGE-M3 (Chen et al.,
2024), jina embedding v3 (Sturua et al., 2024).
With the top- kretrieved documents, we directly
call strong publicly available pre-trained LLMs,Qwen2.5 7B,14B,32B(Yang et al., 2024) for genera-
tion.
5.1.3 Training, Generation and Evaluation
Details.
Our training data includes randomly sampled
<query,document,label> triples (which are ex-
cluded from POLYEVAL)4from a large-scale medi-
cal service platform in China to train our model
for evaluating polyviews. All experiments are
conducted using 4 NVIDIA A100 GPUs. For
Relevance andSupplement evaluation, we utilize
open-source Llama Factory5to finetune small-scale
Qwen2.5 1.5B and adopt Lora tuning for 5 epoch
with a learning rate of 5e-5, a batch size of 4 and
a cosine learning rate scheduler. As for Utility
evaluation, we incorporate BGE-M3 owing to its
superior performance in a variety of benchmark
leaderboards and distill the marvelous power of
LLM in evaluating utility into it, where M(·)is
defined as cosine similarity. We train the utility
model for 5 epochs with a learning rate of 1e-5, a
batch size of 16 for each device, a warm-up ratio of
0.2, the passage window size of 50 and the temper-
ature parameter τset to 0.05 following (Gan et al.,
2024). For Composibility evaluation, we borrow
the embedding from Utility and conduct clustering
via DBSCAN (Ester et al., 1996). For all gener-
ation tasks, we utilize vLLM (Kwon et al., 2023)
for inference speed-up and set the temperature to 0
for reproducibility and max token parameter to 1.
We set [α1, α2, α3, α4, α5]is set to [0.35, 0.35, 0.1,
0.1, 0.1] for INQUIRY andPOLICY and [0.35,
0.35, 0.1, 0.2, 0.0] for CARE for simplicity. For
generation evaluation, we directly call private com-
mercial LLM GPT4 to conduct answer statement
generation and the judgement ( i.e.,circumstances
4ForRelevance andSupplement evaluation, the label is
binary, i.e.,0 or 1 while for Utility evaluation the label is a
float number generated by a powerful LLM.
5https://github.com/hiyouga/LLaMA-Factory

Table 1: Overall retrieval performance (%) evaluation
on P OLYEVAL, here kis set to 3 for simplicity.
RetrievalCARE INQUIRY POLICY
HIT NDCG HIT NDCG HIT NDCG
BM25 26.6 26.6 22.3 22.7 28.2 28.6
GTE 38.7 39.4 31.1 31.7 31.4 32.0
BGE-M3 40.0 40.8 34.8 35.7 33.7 34.6
jina 42.8 43.7 33.5 34.7 35.9 36.9
POLYRAG 47.1 48.3 38.1 39.1 42.8 44.5
Table 2: Generation performance (%) evaluation on
CARE using Top-3 Documents for Retrieval.
Retrieval Generation Rc↑Ri↓RnNc↑Ni↓Nn
BM25Qwen2.5 7B 35.7 8.23 53.4 3.02 0.62 4.72
Qwen2.5 14B 54.6 6.97 35.3 4.39 0.55 3.01
Qwen2.5 32B 57.5 6.31 33.2 4.62 0.50 2.80
GTEQwen2.5 7B 34.5 7.70 55.2 3.16 0.72 5.03
Qwen2.5 14B 53.3 7.02 36.3 4.56 0.60 3.26
Qwen2.5 32B 55.3 6.31 35.3 4.65 0.52 3.09
BGE-M3Qwen2.5 7B 36.2 8.55 51.3 3.25 0.72 5.06
Qwen2.5 14B 54.8 6.93 34.9 4.48 0.56 2.97
Qwen2.5 32B 57.3 6.89 32.6 4.75 0.56 2.83
jinaQwen2.5 7B 38.4 6.96 51.5 3.50 0.62 5.02
Qwen2.5 14B 55.5 6.67 34.6 4.53 0.54 3.01
Qwen2.5 32B 57.0 6.87 33.0 4.65 0.57 2.84
POLYRAGQwen2.5 7B 60.9 7.65 27.5 4.71 0.52 2.23
Qwen2.5 14B 69.2 4.80 22.0 5.32 0.36 1.72
Qwen2.5 32B 71.6 5.40 20.5 5.39 0.38 1.53
correct, incorrect and not mentioned) between an-
swer statement and ground truth and Nc,Rcdenote
the count and ratio of the given circumstance c. Fi-
nally, we have listed all prompt templates in the
Appendix.
5.2 Results and Analysis
5.2.1 Main Results
From the empirical results on retrieval and genera-
tion tasks (Table 1 and Table 2), we can summarize
the major findings as follows:
•POLYRAGlargely improves the performance
of retrieval and generation for knowledge-
intensive tasks. We only list the retrieval results
due to the fact that refusal rate is high when with-
out retrieval ( e.g., forINQUIRY the refusal rate
is as high as 59.7% for Qwen2.5 7B)). By com-
prehensively combining retrieval and generation
metrics defining the correct count, correct ratio,
incorrect count, incorrect ratio, we can find that
POLYRAGconsistently performs well in different
tasks and metrics.
•Both time-evolving and authoritative-sensitive
tasks benefit more from POLYRAG.A
large margin of improvement can be found inPOLICY as it is more sensitive to timeliness
and authoritativeness compared to task such as
CARE , which depends more on the authorita-
tiveness since the improvement of the treatment
takes a lot of time.
•More customization of POLYRAGw.r.t. down-
stream tasks deserves more attention. We take
a trivial step to assign weights to tasks in POLYE-
VAL, however, the ablation study demonstrates
the importance of different views varies across
different tasks, hence more attention should be
devoted to its customization since each task
comes with areas of emphasis.
5.2.2 Feasibility Analysis and Broader Impact
For industrial platform that directly serves user
queries, low-latency inference is of great signif-
icance. In POLYRAG, we utilize polyviews for
a more comprehensive way of information inte-
gration that incorporate multiple models in this
progress, where the overall procedure is illustrated
in the upper part of Figure 2. By flexibly incor-
porating multiple small-scale models and the con-
currency and GPU Segmentation mechanisms, the
polyview-based integration stage can be deployed
using a L20 GPU with latency around 200ms given
an user query with an average of 15 documents
where the total length exceeds 8k tokens. Besides
medical applications, for the broader application,
the idea of POLYRAGcan also be applied to other
domains such as finance where the authoritative-
ness and timeliness of information greatly matters.
6 Conclusion and Future Work
In this work, we propose POLYRAGthat incorpo-
rates varied views for evaluation of each retrieved
document and then pursues integrated polyviews
via a multi-reward based view-mixture mechanism,
which finally incorporates the derived polyview-
grounded knowledge for answer generation. To
bridge the evaluation gap we also propose POLYE-
VAL, a benchmark consists of queries and docu-
ments collected from real-world medical scenarios
with multiple annotation on them. Experiments
and analysis on POLYEVAL have demonstrated the
superiority of POLYRAG. Nevertheless, we take
a trivial step for the multi-rewards mixture and
more complicated approaches requires further re-
search. In the future, we would like to explore
multi-modal retrieval integration and apply the pro-
posed POLYRAGto other scenarios such as finance.

References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InICLR .
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Zhe Chen, Yusheng Liao, Shuyang Jiang, Pingjie Wang,
Yiqiu Guo, Yanfeng Wang, and Yu Wang. 2025. To-
wards omni-rag: Comprehensive retrieval-augmented
generation for large language models in medical ap-
plications. arXiv preprint arXiv:2501.02460 .
Emile Contal and Garrin McGoldrick. 2024. Ragsys:
Item-cold-start recommender as rag system.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong
Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue,
Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu,
Chenggang Zhao, Chengqi Deng, Chenyu Zhang,
Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji,
Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo,
Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
Han Bao, Hanwei Xu, Haocheng Wang, Honghui
Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li,
Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L.
Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai
Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai
Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong
Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan
Zhang, Minghua Zhang, Minghui Tang, Meng Li,
Miaojun Wang, Mingming Li, Ning Tian, Panpan
Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen,
Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan,
Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen,
Shanghao Lu, Shangyan Zhou, Shanhuang Chen,
Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng
Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing
Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun,
T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu,
Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao
Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin
Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li,
Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin,
Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxi-
ang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang,
Xinxia Shan, Y . K. Li, Y . Q. Wang, Y . X. Wei, Yang
Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng
Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi,
Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang,
Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo,
Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yu-
jia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You,
Yuxuan Liu, Yuyang Zhou, Y . X. Zhu, Yanhong Xu,
Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu,Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan,
Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean
Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao,
Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zi-
jia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song,
Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu
Zhang, and Zhen Zhang. 2025. Deepseek-r1: Incen-
tivizing reasoning capability in llms via reinforce-
ment learning. arXiv preprint arXiv:2501.12948 .
Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xi-
aowei Xu. 1996. A density-based algorithm for dis-
covering clusters in large spatial databases with noise.
InKDD , pages 226–231.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on RAG meeting llms: Towards
retrieval-augmented large language models. In KDD ,
pages 6491–6501.
Chunjing Gan, Dan Yang, Binbin Hu, Hanxiao Zhang,
Siyuan Li, Ziqi Liu, Yue Shen, Lin Ju, Zhiqiang
Zhang, Jinjie Gu, Lei Liang, and Jun Zhou. 2024.
Similarity is not all you need: Endowing retrieval
augmented generation with multi layered thoughts.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023a. Retrieval-
augmented generation for large language models: A
survey. arXiv preprint arXiv:2312.10997 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023b. Retrieval-
augmented generation for large language models: A
survey. arXiv preprint arXiv:2312.10997 .
Aditya Golatkar, Alessandro Achille, Luca Zancato,
Yu-Xiang Wang, Ashwin Swaminathan, and Stefano
Soatto. 2024. CPR: retrieval augmented generation
for copyright protection. In CVPR , pages 12374–
12384.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, Arun Rao, Aston Zhang, Aurelien Ro-
driguez, Austen Gregerson, Ava Spataru, Baptiste
Roziere, Bethany Biron, Binh Tang, Bobbie Chern,
Charlotte Caucheteux, Chaya Nayak, Chloe Bi,
Chris Marra, Chris McConnell, Christian Keller,
Christophe Touret, Chunyang Wu, Corinne Wong,
Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Al-
lonsius, Daniel Song, Danielle Pintz, Danny Livshits,
Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino,
Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy,
Elina Lobanova, Emily Dinan, Eric Michael Smith,
Filip Radenovic, Francisco Guzmán, Frank Zhang,

Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis An-
derson, Govind Thattai, Graeme Nail, Gregoire Mi-
alon, Guan Pang, Guillem Cucurell, Hailey Nguyen,
Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan
Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Is-
han Misra, Ivan Evtimov, Jack Zhang, Jade Copet,
Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park,
Jay Mahadeokar, Jeet Shah, Jelmer van der Linde,
Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu,
Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang,
Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park,
Joseph Rocca, Joshua Johnstun, Joshua Saxe, Jun-
teng Jia, Kalyan Vasuden Alwala, Karthik Prasad,
Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth
Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer,
Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal
Lakhotia, Lauren Rantala-Yeary, Laurens van der
Maaten, Lawrence Chen, Liang Tan, Liz Jenkins,
Louis Martin, Lovish Madaan, Lubo Malo, Lukas
Blecher, Lukas Landzaat, Luke de Oliveira, Madeline
Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar
Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew
Oldham, Mathieu Rita, Maya Pavlova, Melanie Kam-
badur, Mike Lewis, Min Si, Mitesh Kumar Singh,
Mona Hassan, Naman Goyal, Narjes Torabi, Niko-
lay Bashlykov, Nikolay Bogoychev, Niladri Chatterji,
Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick
Alrassy, Pengchuan Zhang, Pengwei Li, Petar Va-
sic, Peter Weng, Prajjwal Bhargava, Pratik Dubal,
Praveen Krishnan, Punit Singh Koura, Puxin Xu,
Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj
Ganapathy, Ramon Calderer, Ricardo Silveira Cabral,
Robert Stojnic, Roberta Raileanu, Rohan Maheswari,
Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ron-
nie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan
Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sa-
hana Chennabasappa, Sanjay Singh, Sean Bell, Seo-
hyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sha-
ran Narang, Sharath Raparthy, Sheng Shen, Shengye
Wan, Shruti Bhosale, Shun Zhang, Simon Van-
denhende, Soumya Batra, Spencer Whitman, Sten
Sootla, Stephane Collot, Suchin Gururangan, Syd-
ney Borodinsky, Tamar Herman, Tara Fowler, Tarek
Sheasha, Thomas Georgiou, Thomas Scialom, Tobias
Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal
Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh
Ramanathan, Viktor Kerkez, Vincent Gonguet, Vir-
ginie Do, Vish V ogeti, Vítor Albiero, Vladan Petro-
vic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whit-
ney Meers, Xavier Martinet, Xiaodong Wang, Xi-
aofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xin-
feng Xie, Xuchao Jia, Xuewei Wang, Yaelle Gold-
schlag, Yashesh Gaur, Yasmine Babaei, Yi Wen,
Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao,
Zacharie Delpierre Coudert, Zheng Yan, Zhengxing
Chen, Zoe Papakipos, Aaditya Singh, Aayushi Sri-
vastava, Abha Jain, Adam Kelsey, Adam Shajnfeld,
Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand,
Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei
Baevski, Allie Feinstein, Amanda Kallet, Amit San-
gani, Amos Teo, Anam Yunus, Andrei Lupu, An-
dres Alvarado, Andrew Caples, Andrew Gu, Andrew
Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchan-dani, Annie Dong, Annie Franco, Anuj Goyal, Apara-
jita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,
Ashwin Bharambe, Assaf Eisenman, Azadeh Yaz-
dan, Beau James, Ben Maurer, Benjamin Leonhardi,
Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi
Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Han-
cock, Bram Wasti, Brandon Spence, Brani Stojkovic,
Brian Gamido, Britt Montalvo, Carl Parker, Carly
Burton, Catalina Mejia, Ce Liu, Changhan Wang,
Changkyu Kim, Chao Zhou, Chester Hu, Ching-
Hsiang Chu, Chris Cai, Chris Tindal, Christoph Fe-
ichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty,
Daniel Kreymer, Daniel Li, David Adkins, David
Xu, Davide Testuggine, Delia David, Devi Parikh,
Diana Liskovich, Didem Foss, Dingkang Wang, Duc
Le, Dustin Holland, Edward Dowling, Eissa Jamil,
Elaine Montgomery, Eleonora Presani, Emily Hahn,
Emily Wood, Eric-Tuan Le, Erik Brinkman, Este-
ban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun,
Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat
Ozgenel, Francesco Caggioni, Frank Kanayet, Frank
Seide, Gabriela Medina Florez, Gabriella Schwarz,
Gada Badeer, Georgia Swee, Gil Halpern, Grant
Herman, Grigory Sizov, Guangyi, Zhang, Guna
Lakshminarayanan, Hakan Inan, Hamid Shojanaz-
eri, Han Zou, Hannah Wang, Hanwen Zha, Haroun
Habeeb, Harrison Rudolph, Helen Suk, Henry As-
pegren, Hunter Goldman, Hongyuan Zhan, Ibrahim
Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis,
Irina-Elena Veliche, Itai Gat, Jake Weissman, James
Geboski, James Kohli, Janice Lam, Japhet Asher,
Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jen-
nifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy
Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe
Cummings, Jon Carvill, Jon Shepard, Jonathan Mc-
Phie, Jonathan Torres, Josh Ginsburg, Junjie Wang,
Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khan-
delwal, Katayoun Zand, Kathy Matosich, Kaushik
Veeraraghavan, Kelly Michelena, Keqian Li, Ki-
ran Jagadeesh, Kun Huang, Kunal Chawla, Kyle
Huang, Lailin Chen, Lakshya Garg, Lavender A,
Leandro Silva, Lee Bell, Lei Zhang, Liangpeng
Guo, Licheng Yu, Liron Moshkovich, Luca Wehrst-
edt, Madian Khabsa, Manav Avalani, Manish Bhatt,
Martynas Mankus, Matan Hasson, Matthew Lennie,
Matthias Reso, Maxim Groshev, Maxim Naumov,
Maya Lathi, Meghan Keneally, Miao Liu, Michael L.
Seltzer, Michal Valko, Michelle Restrepo, Mihir Pa-
tel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark,
Mike Macey, Mike Wang, Miquel Jubert Hermoso,
Mo Metanat, Mohammad Rastegari, Munish Bansal,
Nandhini Santhanam, Natascha Parks, Natasha
White, Navyata Bawa, Nayan Singhal, Nick Egebo,
Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich
Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz,
Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin
Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pe-
dro Rittner, Philip Bontrager, Pierre Roux, Piotr
Dollar, Polina Zvyagina, Prashant Ratanchandani,
Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel
Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu
Nayani, Rahul Mitra, Rangaprabhu Parthasarathy,
Raymond Li, Rebekkah Hogan, Robin Battey, Rocky

Wang, Russ Howes, Ruty Rinott, Sachin Mehta,
Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara
Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov,
Satadru Pan, Saurabh Mahajan, Saurabh Verma,
Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lind-
say, Shaun Lindsay, Sheng Feng, Shenghao Lin,
Shengxin Cindy Zha, Shishir Patil, Shiva Shankar,
Shuqiang Zhang, Shuqiang Zhang, Sinong Wang,
Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala,
Stephanie Max, Stephen Chen, Steve Kehoe, Steve
Satterfield, Sudarshan Govindaprasad, Sumit Gupta,
Summer Deng, Sungmin Cho, Sunny Virk, Suraj
Subramanian, Sy Choudhury, Sydney Goldman, Tal
Remez, Tamar Glaser, Tamara Best, Thilo Koehler,
Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim
Matthews, Timothy Chou, Tzook Shaked, Varun
V ontimitta, Victoria Ajayi, Victoria Montanez, Vijai
Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad
Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu,
Vladimir Ivanov, Wei Li, Wenchen Wang, Wen-
wen Jiang, Wes Bouaziz, Will Constable, Xiaocheng
Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo
Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia,
Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi,
Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao,
Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary
DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang,
Zhiwei Zhao, and Zhiyu Ma. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (RAG): evolution, current
landscape and future directions. arXiv preprint
arXiv:2410.12837 .
Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau,
Lana Yeganova, W. John Wilbur, and Zhiyong Lu.
2023. Medcpt: Contrastive pre-trained transformers
with large-scale pubmed search logs for zero-shot
biomedical information retrieval. Bioinform. , 39(10).
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. In SOSP , pages 611–626.
Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng
Ding, Shafiq Joty, Soujanya Poria, and Lidong Bing.
2024. Chain-of-knowledge: Grounding large lan-
guage models via dynamic knowledge adapting over
heterogeneous sources. In ICLR .
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In ACL, pages 9802–9822.Xuan-Phi Nguyen, Shrey Pandit, Senthil Purushwalkam,
Austin Xu, Hailin Chen, Yifei Ming, Zixuan Ke, Sil-
vio Savarese, Caiming Xong, and Shafiq Joty. 2024.
Sfr-rag: Towards contextually faithful llms.
OpenAI. 2023. GPT-4 technical report. arXiv preprint
arXiv:2303.08774 .
Nisarg Patel, Mohith Kulkarni, Mihir Parmar, Aashna
Budhiraja, Mutsumi Nakamura, Neeraj Varshney, and
Chitta Baral. 2024. Multi-logieval: Towards eval-
uating multi-step logical reasoning ability of large
language models. In EMNLP , pages 20856–20879.
Jiarui Rao and Jionghao Lin. 2024. Ramo: Retrieval-
augmented generation for enhancing moocs recom-
mendations.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024a. REPLUG: retrieval-
augmented black-box language models. In NAACL ,
pages 8371–8384.
Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao,
Pengjie Ren, Zhumin Chen, and Zhaochun Ren.
2024b. Generate-then-ground in retrieval-augmented
generation for multi-hop question answering. In ACL,
pages 7339–7353.
Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon
Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim,
and Jaewoo Kang. 2024. Rationale-guided retrieval
augmented generation for medical question answer-
ing.arXiv preprint arXiv:2411.00300 .
Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram,
Michael Günther, Bo Wang, Markus Krimmel, Feng
Wang, Georgios Mastrapas, Andreas Koukounas,
Nan Wang, and Han Xiao. 2024. jina-embeddings-
v3: Multilingual embeddings with task lora.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is chatgpt good at search?
investigating large language models as re-ranking
agents. In EMNLP , pages 14918–14937.
Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and
Sihong Xie. 2025. MBA-RAG: a bandit approach
for adaptive retrieval-augmented generation through
question complexity. In COLING , pages 3248–3254.
Prakhar Verma, Sukruta Prakash Midigeshi, Gaurav
Sinha, Arno Solin, Nagarajan Natarajan, and Amit
Sharma. 2025. Plan*rag: Efficient test-time planning
for retrieval augmented generation.
Shuting Wang, Xin Yu, Mang Wang, Weipeng Chen, Yu-
tao Zhu, and Zhicheng Dou. 2025. Richrag: Crafting
rich responses for multi-faceted queries in retrieval-
augmented generation. In COLING , pages 11317–
11333.

Yubo Wang, Xueguang Ma, and Wenhu Chen. 2024.
Augmenting black-box llms with medical textbooks
for biomedical question answering. In EMNLP Find-
ings, pages 1754–1770.
Junde Wu, Jiayuan Zhu, and Yunli Qi. 2024. Med-
ical graph RAG: towards safe medical large lan-
guage model via graph retrieval-augmented gener-
ation. arXiv preprint arXiv:2408.04187 .
Guangzhi Xiong, Qiao Jin, Xiao Wang, Minjia Zhang,
Zhiyong Lu, and Aidong Zhang. 2024. Improv-
ing retrieval-augmented generation in medicine
with iterative follow-up questions. arXiv preprint
arXiv:2408.00727 .
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu,
Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng
Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tian-
hao Li, Tingyu Xia, Xingzhang Ren, Xuancheng
Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan
Qiu. 2024. Qwen2.5 technical report. arXiv preprint
arXiv:2412.15115 .
Aizan Zafar, Kshitij Mishra, and Asif Ekbal. 2025.
Medex: Enhancing medical question-answering with
first-order logic based reasoning and knowledge in-
jection. In COLING , pages 9701–9720.
Huimin Zeng, Zhenrui Yue, Qian Jiang, and Dong
Wang. 2024. Federated recommendation via hy-
brid retrieval augmented generation. arXiv preprint
arXiv:2403.04256 .
Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li,
Dan Su, Chenhui Chu, and Dong Yu. 2024a. Mm-
llms: Recent advances in multimodal large language
models. In ACL Findings , pages 12401–12430.
Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E.
Gonzalez. 2024b. RAFT: adapting language
model to domain specific RAG. arXiv preprint
arXiv:2403.10131 .
Qingfei Zhao, Ruobing Wang, Yukuo Cen, Daren Zha,
Shicheng Tan, Yuxiao Dong, and Jie Tang. 2024.
Longrag: A dual-perspective retrieval-augmented
generation paradigm for long-context question an-
swering. In EMNLP , pages 22600–22632.

A Appendix
A.1 Prompt Template
This section presents the prompt templates used
during training, inference, and evaluation in our
proposed P OLYRAG6.
A.1.1 Model Training Prompt
Utility Training Prompt. When training the utility
model, we design different prompts so that an LLM
can output its perplexity as our supervision sig-
nal for embedding model under following circum-
stances: i) answering the question with retrieved
document, which demonstrates the utility of the
document towards the input question; ii) answer-
ing the question directly, which means that if the
perplexity for answering this question is lower than
the perplexity when with retrieved document, then
the retrieved document is considered to be useless
by the LLM and it could be utilized to achieve se-
lective retrieval. Here we present the prompts in
Table 3.
Table 3: Utility Model Training Prompt.
w/ retrieved documents
Please answer the question based on the given context. Question:
[QUESTION] The context related to the question is as follows:
[CONTEXT]. Answer: [ANSWER]
w/o retrieved documents
Please answer the question. Question: [QUESTION] Answer: [ANSWER]
Relevance Training and Inference Prompt. We
evaluate the relevance of the retrieved document
w.r.t. the input query by prompting an LLM with
few-shot demonstrations and present the prompts
in Table 4.
Supplement Training and Inference Prompt. We
evaluate the supplement of the retrieved document
w.r.t. the input query by prompting an LLM with
few-shot demonstrations and present the prompts
in Table 5.
A.1.2 Generation Stage Prompt
To prompt an LLM such that it can generate output
for domains INQUIRY ,POLICY andCARE as
we required, we utilize the different prompts when
(not) incorporating retrieved documents in different
domains and the detailed prompts can be found in
Table 6, Table 7 and Table 8.
6Note that since our primary application scenario involves
the Chinese language, the initial prompts are provided in Chi-
nese. For your convenience and reference, each prompt tem-
plate has been translated into English.Table 4: Relevance Training and Inference Prompt.
Your task is to assess the degree of relevance between the Content and the
Query. The Query consists of a user 's question, and the Content contains the
title and some excerpts from a webpage retrieved online. These Queries and
Content mainly involve medical knowledge and medical insurance knowledge.
Below are some examples. After reading these examples, I will give you a
Query and Content. Please assess the relevance of the Content in answering
the Query and assign a score between A-E (A represents that the Query can be
fully answered directly by referencing the Content. B represents that the
Query can still be answered directly by the Content, but the Content contains
some redundant information or lacks minor details. C represents that the
Query cannot be directly answered by the Content, but there’s some degree of
relevance. D represents that the Content cannot directly answer the Query and
contains only scattered keywords related to the Query. E represents that the
Content cannot answer the Query at all, and the Content is either meaningless
or off-topic).
<omited examples>
Example 3:
Query: Pediatric massage
Content: Which department should a child with unexplained fever see?
Pediatric internal medicine or a fever clinic.
Judge: E
<omited examples>
Now I will provide a Query and Content. Please strictly adhere to the Judge
format above when providing your judgment and avoid outputting any additional
content.
Query:{QUESTION}
Content:{CONTEXT}
A.1.3 Auto-evaluation Prompt
We evaluate each generation result by incorporat-
ing GPT4 as the judge model, we first generate
different statements in the answer (please refer to
Table 9 for details) and then check the ratio of state-
ments of the generation result that has been cor-
rectly mentioned in the ground truth from human
experts (please refer to Table 10 for details).

Table 5: Supplement Training and Inference Prompt.
Your task is to determine whether a piece of Content can serve as
supplementary information to aid in answering a Query. The Query consists of
a user 's question, and the Content contains the title and some excerpts from
a webpage retrieved online. These Queries and Content mainly involve medical
knowledge and medical insurance knowledge.
Regarding supplementary information, here’s a description of the distinction
between "supplementary information" and "direct answers," using "how to treat
diabetes" as an example:
(1) Directly answering the Query: Information is considered unable to
directly answer the Query if the retrieved data is entirely irrelevant or
provides little to no help in answering "how to treat diabetes." For
instance, if a user asks about diabetes treatment methods and the returned
information describes the definition, causes of diabetes, or completely
unrelated health advice (e.g., general fitness tips that are not specifically
tailored for diabetic patients), these details cannot help the user
understand how to treat diabetes and would therefore be deemed irrelevant.
(2) Supplementary information: On the other hand, Content that "provides
supplementary information" may not directly answer "how to treat diabetes,"
but could contribute additional knowledge, context, or alternative approaches
that help the user better understand the treatment process or make a more
informed decision. Examples include:
i. Diet recommendations: Introducing dietary plans for people with diabetes,
which, while not pharmacological treatments, are critical for managing blood
sugar levels.
ii. Lifestyle changes: Providing advice on moderate exercise, smoking
cessation, or limiting alcohol intake, which are beneficial for diabetes
management.
iii. Psychological support: Discussing mental health maintenance for diabetic
patients, which, while not a direct physiological treatment, is essential for
overall patient well-being.
Although such information does not explicitly list specific treatment steps
or medications, it plays an important role in providing users with a broader
perspective and support in diabetes management.
In short, whether information is deemed "irrelevant" or "providing
supplementary information" depends on whether it positively aids the user in
understanding, deciding, or carrying out actions related to the core question
(e.g., diabetes treatment). Even indirect information that facilitates the
user in achieving their query objective can be regarded as supplementary.
Below are some examples. After reading these examples, I will give you a
Query and Content. Please assess the degree to which the Content provides
supplementary information for answering the Query and assign a score of 0/1
(1 represents that the Content provides supplementary information, while 0
represents that it does not provide supplementary information).
Example 1:
Query: How to reverse mild fatty liver disease?
Content: What are the stages of fatty liver disease? Simple steatosis:
Symptoms include fatigue and upper right abdominal discomfort, with normal
liver function. Ultrasound or (and) CT scans indicate mild to moderate fatty
liver. Steatohepatitis: Symptoms include fatigue and upper right abdominal
discomfort, with liver function exceeding the upper normal limit by 1-5 times
for over four weeks. Ultrasound or (and) CT scans indicate fatty liver.
Hepatic fibrosis or (and) cirrhosis: Symptoms include fatigue and upper right
abdominal discomfort, with liver function and blood indicators of fibrosis
being normal or abnormal. Ultrasound or (and) CT, MRI, liver stiffness
testing, etc., suggest fatty liver with fibrosis or cirrhosis confirmed by
liver biopsy.
Judge: 1
<omitted examples>
Now I will provide a Query and Content. Please strictly adhere to the Judge
format above when providing your judgment and avoid outputting any additional
content.
Query:{QUESTION}
Content:{CONTEXT}Table 6: Generation Prompt for INQUIRY .
w/ retrieved documents
system:
Please answer the following question based on the "Reference Materials,"
adhering to the requirements below:
1. Provide an answer that is as concise, polite, and logical as possible,
under 300 words.
2. Use the "general-specific-general" format and markdown structure in
your response.
3. If it is not possible to answer based on the content in the Reference
Materials, reply with: "Sorry, I do not have the relevant knowledge yet."
4. Do not forget that you are a medical assistant. Offer positive and
constructive advice or educational explanations related to the issue
without providing definitive diagnostic opinions like a doctor.
5. Do not use <|Reason|> to start your reasoning. Begin your final answer
with the tag <|ANSWER|> and end your response in the format <|ANSWER|>:
\$answer.
user:
Question:
[QUESTION]
Reference Materials
[CONTEXTS]
w/o retrieved documents
system:
Please answer the following questions with the following requirements:
1. Provide answers that are as concise, polite, logical, and under 300
words as possible.
2. Use the "general-specific-general" structure and markdown format for
answering.
3. If unable to answer, respond with: "Sorry, I do not have the relevant
knowledge yet."
4. Do not forget that you are a medical assistant. Offer positive and
constructive advice or scientific explanations related to the issue
without providing definitive diagnostic opinions like a doctor.
5. Do not begin thinking with <|Reason|>; instead, start your final
answer with the tag <|ANSWER|> and conclude your reply in the format
<|ANSWER|>: \$answer.
user:
Question:
[QUESTION]
Table 7: Generation Prompt for POLICY .
w/ retrieved documents
system:
Please answer the question based on the "Reference Materials" with the
following requirements:
1. Ensure that your response is polite, logical, and no more than 300
words.
2. If the answer requires providing detailed steps, include all details
as mentioned in the original text, and do not omit any steps.
3. If the reference materials mention specific regions, do not omit them
in your response. You can specify by saying “For example, in [region].”
4. Avoid using terms like "New Rural Cooperative Medical Scheme" (also
called NCMS, cooperative medical care, rural cooperative healthcare, or
rural medical insurance), as they no longer exist. Inform users that it
has been merged into the Urban and Rural Resident Basic Medical Insurance.
5. Do not begin with <|Reason|> when reasoning. Start your final answer
with the tag <|ANSWER|> and end your response in the format <|ANSWER|>: \$answer.
user:
Question:
[QUESTION]
Reference Materials
[CONTEXTS]
w/o retrieved documents
system:
Please answer the following questions with the requirements below:
1. Ensure that your response is polite, logical, and no more than 300
words.
2. If you have relevant professional knowledge and there are detailed
steps available, provide the steps in full without omitting them.
3. If the response requires mentioning specific regions, do not omit the
locations. You can specify by saying “For example, in [region].”
4. Avoid using terms like "New Rural Cooperative Medical Scheme" (also
known as NCMS, cooperative medical care, rural cooperative healthcare, or
rural medical insurance), as they no longer exist. Instead, inform users
that it has been merged into the Urban and Rural Resident Basic Medical
Insurance.
5. Do not begin with <|Reason|> when reasoning. Start your final answer
with the tag <|ANSWER|> and end your response in the format <|ANSWER|>: \$answer.
user:
Question:
[QUESTION]

Table 8: Generation Prompt for CARE .
w/ retrieved documents
system:
You are a medical expert with professional healthcare knowledge and excel
at using plain and understandable language to provide educational
explanations for patients. Please base your answers on the following
execution steps and respond to the patient 's question step by step:
Execution Steps:
1. Understand the patient 's question and consider the key information
points the patient is most eager to learn when asking the question.
2. Think about the specific content that should be included in those key
information points. You may use your professional knowledge or consult
the reference materials to answer. If the content from the reference
materials is incorrect, do not use it. If you lack the relevant
expertise, reply with: "Sorry, I do not have the relevant knowledge yet."
3. Organize the information from steps 1 and 2 logically, such as by
using categorization or progressive relationships.
4. Provide a comprehensive and logical answer, and include a risk warning
at the end to help avoid potential medical disputes.
5. For "yes or no" type questions, clearly state your conclusion upfront,
such as: "Yes," "Not recommended," or "No."
6. If the patient’s condition appears to be dangerous, advise the patient
to seek medical attention promptly.
Output Requirements:
1. Use plain and simple language, avoiding overly technical terms.
2. Keep the response brief but thorough, with a clear and easy-to-read
format. Do not omit key points, avoid wordiness, and ensure brevity, as
users may not have the patience for lengthy responses.
3. Answers must adhere to medical facts; no fabricated information is
allowed.
4. Provide only the final answer; do not display your reasoning process.
5. The response should not exceed 250 words.
user:
Question:
[QUESTION]
Reference Materials
[CONTEXTS]
w/o retrieved documents
system:
You are a medical expert with professional healthcare knowledge and excel
at using plain and understandable language to provide educational
information to patients. Please base your answers on the following
execution steps and answer the patient 's question step by step:
Execution Steps:
1. Understand the patient 's question and consider the key information
points the patient is most eager to learn when asking the question.
2. Think about the specific content that should be included in those key
information points. Use your professional knowledge to answer; if you
lack the relevant knowledge, respond with "Sorry, I do not have the
relevant expertise.”
3. Organize the information from steps 1 and 2 logically, such as using
categorization or progressive relationships.
4. Provide a comprehensive and logical answer, and include a risk warning
at the end of the answer to help avoid medical disputes.
5. For "yes or no" type questions, clearly state your conclusion upfront,
e.g., "Yes," "Not recommended," or "No."
6. For situations where the patient 's condition may be dangerous, suggest
that they seek medical attention promptly.
Output Requirements:
1. Use plain and simple language, avoiding overly technical terms.
2. Keep the response brief but thorough, with a clear and easy-to-read
format. Avoid omitting key points or being excessively wordy, as users
may not have the patience to read overly long responses.
3. Answers must align with medical facts; absolutely no fabricated
information is allowed.
4. Provide the final answer only; do not display your thinking process.
5. The overall response should not exceed 250 words.
user:
Question:
[QUESTION]Table 9: Answer Statement Generation Prompt.
[Instruction] You are a medical insurance expert. Given a question and an
answer, generate one or more factual statements from each sentence of the
answer.
[Requirements]:
The generated statements must not contain pronouns. If necessary, pronouns
can be rewritten using the overall context of the answer or the question.
The generated statements must be complete. If necessary, the cause and effect
can be supplemented based on the context.
The generated statements must be entirely derived from the answer and must
not alter the original meaning.
If a specific procedure is mentioned, the entire procedure must be included
in one statement. If there are multiple procedures, they must be included in
separate statements.
[Here is an example]:
Question
How can I use my medical insurance balance for my family members who are part
of the shared account?
Answer
To use your medical insurance balance for your family members who are part of
the shared account, you can follow these steps:
Setting up Family Sharing:
First, you need to set up the family sharing binding.
On the Alipay homepage, click on [Medical Health] or search for “Medical
Health,” enter the Alipay [Medical Health] mini-program, search for [Family
Sharing], click [Use Now], click [Apply Now], and follow the operation
prompts to complete the setup.
Using the Medical Insurance Electronic Voucher:
During payment, display the medical insurance electronic voucher for scanning
and settlement.
The system will prioritize deducting from the balance in your account.
When your account balance is insufficient, the system will automatically use
the personal account balance of the family members in the shared account.
Handling Special Cases:
For infants or elderly family members without mobile devices, the family
member can use the family account feature in the National Medical Insurance
Bureau APP to display the electronic voucher for settlement.
Please note that the use and management of family sharing funds must comply
with local medical insurance regulations. Violating these regulations will
result in corresponding legal liabilities.
Statements
To use the medical insurance balance for family members, first, set up the
family sharing binding.
The procedure for setting up family sharing is: On the Alipay homepage, click
on [Medical Health] or search for “Medical Health,” enter the Alipay [Medical
Health] mini-program, search for [Family Sharing], click [Use Now], click
[Apply Now], and follow the operation prompts to complete the setup.
When using the medical insurance balance for family members, display the
medical insurance electronic voucher for scanning and settlement.
When using the medical insurance balance for family members, the system
prioritizes deducting from the balance in the account.
When using the medical insurance balance for family members, if the account
balance is insufficient, the system will automatically use the personal
account balance of the family members in the shared account.
When using the medical insurance balance for family members, if there are
special cases such as infants or elderly family members without mobile
devices, the family member can use the family account feature in the National
Medical Insurance Bureau APP.
[Please generate the following results based on the requirements and example]:
Question
${QUESTION}
Answer
${ANSWER}
Statements

Table 10: Answer Statement Judgement Prompt.
[Instruction] You are an expert in the field of medical insurance.
Considering the given question, the real answer, and multiple statements,
judge whether each statement is incorrect, not mentioned, or correct, and
provide the reason.
[Requirements]:
1. Combine the question to understand the overall meaning of the real
answer, understand each reference relationship in the answer, and understand
each logical relationship of and, or, not, before judging each statement.
2. The criteria for judging "not mentioned" are as follows:
2.1 If the argument mentioned in the statement does not exist in the real
answer or cannot be inferred, it is considered not mentioned.
2.2 If the statement answers from multiple perspectives, but the real answer
only covers one perspective, it is considered not mentioned.
2.3 If the correctness of the statement cannot be verified based on the real
answer, it is considered not mentioned.
3. The criteria for judging "incorrect" are as follows:
3.1 If the statement mentions "related app," "related application," "medical
insurance app," or other vague expressions, it is considered incorrect.
3.2 If the argument mentioned in the statement is also mentioned or can be
inferred from the real answer, and you can verify that the argument in the
statement is incorrect using the real answer, it is considered incorrect. If
you cannot prove the argument is incorrect based on the real answer, do not
consider it incorrect.
3.3 For statements about the process, only judge that the process exists in
the real answer. It is considered incorrect only when the process does not
exist in the real answer.
4. The criteria for judging "correct" are as follows:
4.1 If the argument in the statement is also mentioned or can be inferred
from the real answer, and there is no contradiction, it is considered
correct.
4.2 If none of the situations in 2 and 3 apply, it is considered correct.
After indicating the judgment result with "not mentioned" / "incorrect" /
"correct," use a semicolon to separate the reason.
[Here is an example]:
Question
How can I use my medical insurance balance for my family members who are part
of the shared account?
Answer
To use your medical insurance balance for your family members who are part of
the shared account, you can follow these steps:
1. Set up Family Sharing:
On the Alipay homepage, click on [Healthcare] or search for “Healthcare,”
enter the Alipay [Healthcare] mini-program, search for [Family Sharing],
click [Use Now], click [Apply Now], and follow the prompts to complete the
setup.
2. Use the Electronic Medical Insurance Card:
When making a payment, show the electronic medical insurance card for
scanning.
The system will prioritize deducting from the balance of the current user 's
electronic medical insurance card.
If the user 's account balance is insufficient, it will automatically use the
personal account balance of the authorized person.
3. Special Case Handling:
For infants or elderly family members without mobile devices, you can use the
Alipay family account feature to display the user 's electronic card to
complete the transaction.
Please note that the use and management of family sharing funds must comply
with local medical insurance regulations. Misuse of funds will result in
corresponding legal responsibilities.
Statements
1. To use for family members, you need to set up family sharing.
3. The setup path is: On the Alipay homepage, click on [Healthcare] or search
for “Healthcare,” enter the Alipay [Healthcare] mini-program, search for
[Family Sharing], click [Use Now], click [Apply Now], and follow the prompts
to complete the setup.
3. When using for family members, you need to show the electronic medical
insurance card for scanning.
4. When using for family members, the system will prioritize deducting from
your balance.
5. When using for family members, if your account balance is insufficient, it
will automatically use the personal account balance of the family member.
6. When using for family members, if there are special cases such as infants
or elderly family members without mobile devices, you can use the family
account feature of the National Medical Insurance Bureau app.
Judgment
1. Correct; The real answer mentions following the steps, the first step is
to set up family sharing, which can be inferred from the statement, and there
is no contradiction.
2. Correct; The real answer mentions the setup path for family sharing, which
is consistent with the statement.
3. Correct; The real answer mentions that when using, you need to show the
electronic medical insurance card for scanning, which is consistent with the
statement.
4. Incorrect; The real answer mentions that when using, the system
prioritizes deducting from the user 's account balance. Based on the question,
the user refers to the family member, which is inconsistent with the
deduction subject mentioned in the statement.
5. Incorrect; The real answer mentions that when using, if the user 's account
balance is insufficient, it will automatically use the personal account
balance of the authorized person. The user refers to the family member, and
the authorized person is you, which is opposite to the subject mentioned in
the statement.
6. Not mentioned; The statement mentions that it can be used through the
National Medical Insurance Bureau app, but the real answer does not mention
this, only stating that it can be used through the Alipay app, and it is
unclear whether the National Medical Insurance Bureau app can be used, so it
cannot be verified as correct or incorrect, hence it is not mentioned.
Question
${QUESTION}
Real Answer
${GROUNDTRUTH}
Statements
${STATEMENT}
Judgment
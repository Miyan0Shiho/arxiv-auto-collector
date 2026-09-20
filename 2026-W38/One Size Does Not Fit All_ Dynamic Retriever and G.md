# One Size Does Not Fit All! Dynamic Retriever and Generator Selection for RAG

**Authors**: Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, Sumit Bhatia

**Published**: 2026-09-15 18:20:38

**PDF URL**: [https://arxiv.org/pdf/2609.17709v1](https://arxiv.org/pdf/2609.17709v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems typically employ fixed retriever and generator configurations across queries, despite substantial differences in query complexity and information needs, leading to inefficient allocation of computational resources. While retrieval and generation adaptivity have been studied independently, their joint effect on end-to-end RAG performance remains underexplored. We systematically analyze how retriever and generator complexity interacts across factoid and multi-hop question answering (QA), including bridge and composition reasoning tasks. Our analysis shows that stronger retrieval generally yields larger gains than increased generation effort, but both exhibit diminishing and non-monotonic returns, indicating that higher-complexity configurations are not uniformly better across queries. Motivated by these findings, we introduce DRAG, a query-adaptive framework for selecting retriever-generator configurations. We first propose DRAG$_\text{QPP}$, a training-free routing approach that uses Query Performance Prediction (QPP) signals to guide retriever selection and perplexity-based measures over retrieved context to guide generator selection. We further introduce DRAG$_\text{SFT}$, a supervised routing approach that fine-tunes an LLM to jointly predict retriever-generator configurations. Across three LLM families and four QA benchmarks, \qpprag~achieves performance comparable to strong static RAG baselines while substantially reducing inference latency, whereas DRAG$_\text{SFT}$ consistently improves effectiveness over static and training-free adaptive baselines. Overall, DRAG demonstrates that jointly adapting retrieval and generation achieves a more favorable effectiveness-efficiency trade-off than static RAG pipelines.

## Full Text


<!-- PDF content starts -->

One Size Does Not Fit All!
Dynamic Retriever and Generator Selection for RAG
Neeraj Anand
Media and Data Science Research Lab
Adobe Systems, Noida, India
neeraja@adobe.comPayel Santra
IACS
Kolkata, India
payel.iacs@gmail.comPartha Basuchowdhuri
IACS
Kolkata, India
partha.basuchowdhuri@iacs.res.in
Debasis Ganguly
University of Glasgow
Glasgow, United Kingdom
debasis.ganguly@glasgow.ac.ukSumit Bhatia
Media and Data Science Research Lab
Adobe Systems, Noida, India
sumit.bhatia@adobe.com
Abstract
Retrieval-Augmented Generation (RAG) systems typically employ
fixed retriever and generator configurations across queries, despite
substantial differences in query complexity and information needs,
leading to inefficient allocation of computational resources. While
retrieval and generation adaptivity have been studied indepen-
dently, their joint effect on end-to-end RAG performance remains
underexplored. We systematically analyze how retriever and gen-
erator complexity interacts across factoid and multi-hop question
answering (QA), including bridge and composition reasoning tasks.
Our analysis shows that stronger retrieval generally yields larger
gains than increased generation effort, but both exhibit diminish-
ing and non-monotonic returns, indicating that higher-complexity
configurations are not uniformly better across queries.
Motivated by these findings, we introduce DRAG, a query-adaptive
framework for selecting retriever-generator configurations. We
first propose DRAG QPP, a training-free routing approach that uses
Query Performance Prediction (QPP) signals to guide retriever se-
lection and perplexity-based measures over retrieved context to
guide generator selection. We further introduce DRAG SFT, a super-
vised routing approach that fine-tunes an LLM to jointly predict
retriever-generator configurations. Across three LLM families and
four QA benchmarks, DRAG QPPachieves performance compara-
ble to strong static RAG baselines while substantially reducing
inference latency, whereas DRAG SFTconsistently improves effec-
tiveness over static and training-free adaptive baselines. Overall,
DRAG demonstrates that jointly adapting retrieval and generation
achieves a more favorable effectiveness-efficiency trade-off than
static RAG pipelines.
CCS Concepts
•Information systems →Query intent;Information retrieval
query processing.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference’17, Washington, DC, USA
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXKeywords
Adaptive RAG, Retriever Selection, Generator Selection
ACM Reference Format:
Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit
Bhatia. 2018. One Size Does Not Fit All! Dynamic Retriever and Gener-
ator Selection for RAG . In.ACM, New York, NY, USA, 12 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Large Language Models (LLMs) have demonstrated strong capa-
bilities in text generation; however, their responses remain con-
strained by the knowledge encoded in their parameters during
pretraining [ 35]. As a result, LLMs often struggle to reliably gen-
erate or reason over information that is absent from, outdated
in, or only sparsely represented in their training data. Retrieval-
Augmented Generation (RAG) addresses this limitation by explicitly
incorporating external knowledge sources into the generation pro-
cess [ 27,35,73]. A typical RAG pipeline consists of two components:
(i) aretriever(such as BM25 [ 48], or a dense ranker such as E5 [ 66])
that selects contextually relevant documents from an external cor-
pus, and (ii) agenerator, which is typically a decoder-based LLM
(e.g., LLaMA [ 63] or GPT [ 6]), which conditions on both the input
query and the retrieved contexts to produce a task-specific response.
Most existing RAG systems rely on a fixed retriever–generator con-
figuration that is applied uniformly across all queries [ 15,43,67,79],
thus leading to computational inefficiency and suboptimal effective-
ness. However, prior work in IR has long established that queries
differ substantially in the complexity of their underlying informa-
tion needs, which in turn affects retrieval difficulty and downstream
generation quality [7, 14, 26].
In a RAG setting, query complexity spans a broad spectrum:
from simple factoid queries answerable using a single relevant doc-
ument [ 25,32], to compositional multi-hop queries requiring rea-
soning across multiple evidence sources [ 64,70], as well as queries
involving temporal reasoning [ 23], numerical aggregation [ 11], or
multi-constraint satisfaction [ 57]. For simpler queries, standard
sparse or dense retrieval models (e.g., sparse such as BM25 or bi-
encoders such as E5 [ 66]) are often sufficient without requiring com-
putationally expensive re-ranking strategies [ 13,71]. In contrast,
more complex queries require stronger retrieval and generation
pipelines, achieved either through enhanced reasoning capabilities
1
arXiv:2609.17709v1  [cs.IR]  15 Sep 2026

Conference’17, July 2017, Washington, DC, USA Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit Bhatia
Figure 1:Performance (F1) and efficiency (time/query) on the Hot-
potQA dataset across12retriever-generator configurations (4re-
trievers×3generators) of increasing complexity. The DRAG QPP
and DRAG SFTmethods adaptively select query-specific retriever-
generator pairs, achieving a favorable effectiveness-efficiency trade-
off by attaining strong downstream performance at lower latency
than static RAG with high-complexity configurations.
in the generator, such as multi-step reasoning [ 44], query decompo-
sition [ 1], increasing thethinkingcapability of the generator, struc-
tured generation strategies such as chain-of-thought [ 30], or by
improving the retrieval pipeline using tools such asretrieve-rerank
pipelines employing cross-encoders [ 27,29,45] and LLM-based
listwise rankers [ 55,58,76,80,81]. However, these improvements
often come at the cost of increased computational overhead and
latency [ 46]. This suggests that aquery-adaptive RAG workflow
that dynamically allocates retrieval and generation capacity based
on the requirements of each query can potentially improve the overall
effectiveness–efficiency trade-off.
While promising, implementing such an adaptive framework
presents two primary challenges: (i)generalizability, since rout-
ing decisions must remain robust across diverse query types and
adapt to evolving model configurations without relying on rele-
vance judgments or answer quality signals at inference time; and (ii)
thetrade-off between effectiveness and efficiency, since increased re-
trieval or reasoning effort improves answer quality only selectively
while substantially increasing latency. To address these challenges,
we first conduct asystematic empirical analysis of retriever-
generator configurations of varying complexity(Section 3),
examining how their interactions affect downstream RAG effec-
tiveness and efficiency. This analysis provides insights into when
additional computational capacity is beneficial and motivates query-
adaptive configuration selection.
To this end, we introduceDRAG(DynamicRAG), a query-
adaptive framework for selecting retriever-generator configura-
tions. We instantiate DRAG through two complementary approaches
(Section 4): atraining-freedynamic approach,DRAG QPP, which
routes queries through a cascaded sequence of models of increasing
complexity, and asupervisedapproachDRAG SFT, that learns to
jointly predicts retriever-generator configurations from query sig-
nals. Together, these approaches explore the effectiveness-efficiencytrade-off (EET) of adaptive RAG under different levels of supervi-
sion and computational overhead [39].
We study both DRAG QPPand DRAG SFTunder the common
DRAG framework to analyze their respective strengths and limita-
tions across standard benchmarks as well as evaluate them in out-
of-domain (OOD) scenarios. We study OOD robustness using the
FRAMESdataset, where routing models are applied without dataset-
specific tuning and we show that adaptive retriever-generator se-
lection can generalize beyond the training distribution.
The contributions of our work can be summarized as :
•We present the first comprehensive empirical study analyzing the
joint impact of retriever-generator pair selection on end-to-end
RAG performance.
•We proposeDRAG QPP, a training-free, query-adaptive frame-
work that leverages unsupervised retrieval performance pre-
diction (QPP-based [ 78]) and generator performance prediction
(GPP-based [52]) to dynamically balance the EET.
•We introduceDRAG SFT, a supervised framework that treats con-
figuration selection as a predictive task, which improves selection
effectiveness over DRAG QPP, albeit at cost of latency.
•We conduct extensive evaluations across three diverse LLM fam-
ilies and four benchmarks – including out-of-domain evaluation
onFRAMES– demonstrating the robustness and generalizability
of our adaptive paradigms across full test sets.
•To facilitate reproducible research, we release our dataset and
code at https://anonymous.4open.science/r/DynamicRAG-2A01/.
2 Related Works
Retrieval Augmented Generation.Retrieval-Augmented Gen-
eration (RAG) improves text generation by grounding LLM out-
puts in external knowledge, thereby mitigating hallucinations and
knowledge limitations. Prior work has explored several forms of re-
trieval augmentation, including query-based retrieval and retrieval-
conditioned generation [ 17,51], latent and logit-level integration
of retrieved evidence into language modeling [ 28,77], retrieval
refinement through reranking, active, and iterative retrieval strate-
gies [ 16,24,75], as well as architectures that improve reasoning
over retrieved documents and multi-document fusion [ 19,34,50].
Another line of work explores supervised fine-tuning of LLMs, that
retrofit any LLM with retrieval capabilities [ 36,65], distill reader
signal back into the retriever [ 20], or fine-tune the generator to
reason over retrieved context [3, 36].
Adaptive Retrieval.Existing studies on adaptive RAG focused
on binary decisions, i.e., whether to retrieve or not [ 38]. This method
is sufficient for simple queries, but generally fails multi-hop difficult
queries. Thus, some studies introduces a method with a variable
number of iterative retrieval steps, i.e., it jointly retrieve-and-rerank
until sufficient evidence is accumulated for correct prediction[ 47].
Furthermore, SELF-RAG [ 3,21] introduces training strategy to
retrieve, self-critique and refine the outputs. Unlike our method
DRAG, it does not explicitly address the problem of selecting among
multiple retriever–generator configurations. In contrast, several
works explore adaptivity from reasoning point of view, while assum-
ing a fixed retrieval pipeline. Some existing studies varies the num-
ber of reasoning steps [ 68] or samples multiple chain-of-thought
2

Dynamic Retriever and Generator Selection for RAG Conference’17, July 2017, Washington, DC, USA
reasoning paths to dynamically select the generation path for im-
proving robustness [ 72]. Different from previous approaches, this
paper jointly analyzes the impact of retriever and generator choices
with different capabilities on RAG framework.
Query Performance Prediction.Query performance predic-
tors can be classified into two major categories: pre-retrieval and
post-retrieval. Pre-retrieval prediction technique aims to predict
query difficulty by analyzing the prior distribution to querying the
given collection of documents [ 10,18,41]. The post-retrieval predic-
tion methodology, on the contrary, relies on statistical approaches
on retrieval scores [ 5,54,59,78], learning to rank features inspired
predictors [ 4] and robustness of rank-list perturbation based predic-
tors [ 49,53]. Some recent studies explored supervised and neural
approaches for QPP by considering both the content and the score
features [ 7,74]. In this work, we use these pre-retrieval predictors
in our downstream pipeline to guide the selection of an appropriate
retriever based on query complexity.
3 RAG Configuration Analysis and Problem
Formulation
We first formalize the adaptive RAG task and analyze the behavior
ofstaticRAG systems, where a fixed retriever and generator are
used uniformly across all queries. This analysis serves two purposes:
(i) characterizing the EET across retriever-generator configurations,
and (ii) motivating the need for query-adaptive frameworks.
Task Formulation.Given a query 𝑞, a set of candidate retriev-
ersΘ={𝜃𝑖}𝑚
𝑖=1and a set of candidate generators Φ={𝜙𝑖}𝑛
𝑖=1, the
objective of adaptive RAG is to select a query-specific retriever-
generator configuration (𝜃,𝜙)∈Θ×Φ that balances downstream
answer quality and computational cost. Unlike static RAG, adaptive
RAG seeks to select the optimal retriever and generator based on
the query’s specific information need to maximize effectiveness
while minimizing latency.
Since the effectiveness of a retriever-generator pair cannot be
directly observed at inference time (i.e., the ground-truth answer
𝑎𝑞is unavailable), we need an estimator, 𝑓:(𝑞,𝜃,𝜙)↦→R , that
predicts the likelihood of generating a correct answer for a given
query𝑞. The objective is then to select the most efficient configura-
tion that maximizes downstream effectiveness while minimizing
computational overhead.
Retriever and Generator Configuration Choice.The adap-
tive setup requires a discrete action space of retrievers Θand gen-
erators Φ, ordered by increasing computational cost and capability.
Following the conventional view in IR, we maintain the progres-
sion of retriever complexity fromlexical retrievaltodense semantic
retrievaland further to sophisticatedretrieve-and-rerankpipelines
employing query decomposition and cross-encoder rerankers. Sim-
ilarly, generator complexity scales from lightweight decoding with
no or minimal reasoning to longer, multi-step reasoning i.e., extend-
ing the reasoning effort or thinking modes of LLMs.Importantly,
our proposed framework is entirely agnostic to the specific number of
retriever or generator choices, allowing it to scale seamlessly across
finer- or coarser-grained complexity modes.
In our experiments for GPT-OSS base-model, we consider four
retrieval levels, i.e.,low(L; sparse retriever),medium(M; denseGen / Ret Low (%) Medium (%) High (%) Extra-High (%)
Low 11.2 5.6 12.5 6.9
Medium 5.1 3.8 5.7 4.5
High 13.0 7.1 15.6 9.0
Table 1:Distribution (%) of correctly answered HotpotQA queries
under an efficiency-aware oracle RAG system across total 12 config-
urations, with four retriever levels (L, M, H, XH) and three GPT-OSS
generator reasoning levels (L, M, H). The results show that a sub-
stantial fraction of queries can be answered using low and medium
complexity configurations, indicating that considerable latency re-
ductions are achievable through adaptive framework.
retriever),high(H; retrieve-and-rerank pipeline), andextra-high
(XH; query decomposition with retrieve-and-rerank pipeline), re-
flecting complex retrieval pipelines with increasing latency and
three cascaded levels for retriever component, i.e.,low(L),medium
(M) andhigh(H) reflecting the mode of thinking, which means
total12configurations (4 ×3). Similarly, for Qwen and Gemma, we
consider two generator modes depending on thinking availability
(thinking-off/on), resulting in8configurations (4×2) per model.
Efficiency-aware Oracle configuration.To understand the
query-level EET in static RAG frameworks, we define anefficiency-
aware oracle setupthat assigns each query to the least computation-
ally expensive configuration of retriever-generator pair to correctly
answer the query 𝑞. Intuitively, the oracle represents an ideal setup
that has access to ground-truth effectiveness and therefore reveals
how retrieval and generation effort should ideally be allocated
across queries. Formally, given a query𝑞, the oracle selects
(𝜃∗,𝜙∗)=arg min
𝜃,𝜙∈Θ×ΦL(𝜃,𝜙)s.t.P 𝑎𝑞,𝜙(𝑞,𝜃𝑘(𝑞))≥𝛿,(1)
where i)𝜃𝑘(𝑞)denotes the top- 𝑘documents retrieved by the
ranker𝜃, ii)𝜙∈Φ :(𝑞,𝜃𝑞)↦→ ˆ𝑎denotes a generator which takes 𝑞
along with its retrieved documents 𝜃𝑞as input to yield a predicted
answer ˆ𝑎𝑞, iii)𝑎𝑞is the ground-truth answer for query 𝑞, iv)P:
𝑎𝑞×ˆ𝑎𝑞↦→R denotes downstream effectiveness (measured by
metrics such as Exact Match (EM) and F1) between ˆ𝑎𝑞and𝑎𝑞, v)
L:𝜃×𝜙↦→R is the total latency and vi) 𝛿represents a effectiveness
threshold. When multiple configurations satisfy the threshold, the
oracle selects the least expensive one.
In practice, if no configuration satisfies the target criterion (e.g.,
EM=1), we map to the least expensive configuration achieving high
partial correctness (F1 ≥0.8). Following prior works that applied
thresholds on answer-level F1 to account for minor lexical varia-
tions in partial correct responses [ 3,24], we use0.8as a threshold .
If no configuration satisfies this criterion, the query is assigned to
the highest-complexity setting(𝜃 𝑋𝐻,𝜙𝐻)as a safe fallback.
Discussion.Table 1 reports the oracle distribution across all
retriever-generator configurations on the HotpotQA dataset when
GPT-OSS is the base model. Note that a substantial fraction of
queries are correctly answered using low- and medium-complexity
configurations, indicating that uniformly invoking the most expen-
sive retrieval and reasoning pipeline is often unnecessary.
This data underscores a fundamental reality of RAG optimiza-
tion: query complexity dictates architectural requirements, meaning
3

Conference’17, July 2017, Washington, DC, USA Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit Bhatia
Query :  What is the name for the adventure in "T unnels and Trolls", a game designed by Ken St. Andre?
QueryCorpus
   Sparse
    RetrieverSparse
Index
 Corpus
   Dense   
RetrieverDense
Index
Query
Corpus
   Dense   
RetrieverDense
Index
QueryCross-Encoder
Ranker
Corpus
   Dense   
RetrieverDense
Index
Cross-Encoder
RankerQuerySub-query 1
Sub-query 2
Sub-query 3Sparse Retrieval (Low) Dense Retrieval (Medium)
Retrieve-and-Rerank (High)
Query-Docomposition >> Retrieve-and-Rerank (Extra-High)
Retrievers/Rankers with Increasing Ef fectiveness and LatencyIncreasing Thinking Ef fort of Generators
Low Medium HighRPPQuery
QPP
EstimatorCorpus
Sparse
Perplexity
Score
Perplexity
Score
MediumThresholding
GPP
Train
Dataset
 (Sparse,High))
 (Dense,Low))
SFT Trainer
Test
Query
Saved CheckpointSparse 
Retriever
+
 Medium
GeneratorAdaptive Workflow of Retriever and Generator 
(Our Proposal)
Training Phase
Inference PhaseSparse Retriever
+
Medium ef fort of Generator
Arena of Khazan
Predicted Downstream
AnswerPredicted Configuration1 2
3
4Predicted Retriever
Predicted Generator
PredictionLLM
LLM
Figure 2:Overview of the proposed query-adaptive framework (DRAG) for GPT-OSS. The left panel shows retriever and generator configurations
of progressively increasing complexity, ranging from sparse to query decomposition with retrieve-rerank pipelines, and from low to high
thinking effort. Given an input query, DRAG dynamically selects a query-specific retriever-generator configuration instead of uniformly
invoking a fixed pipeline. The top-right panel illustrates the training-free DRAG QPPworkflow with 2 modules,retrieval performance prediction
(RPP) followed bygenerator performance prediction(GPP). The bottom-right panel shows DRAG SFT, a supervised framework that learns
retriever–generator selection from query–configuration supervision and predicts the optimal configuration during inference. The selected
configuration is then executed to produce the downstream answer. While illustrated using GPT-OSS with four retrievers and three generator
levels (4×3=12configurations), the framework naturally generalizes to other model families (e.g., Qwen and Gemma with4 ×2configurations).
For our proposed adaptive DRAG QPPapproach, we used training-free RPP and GPP-based methods as explained in Section 4.
some queries are adequately resolved by lightweight configurations
while others strictly demand advanced retrieval and reasoning
pipelines. For example, a simple factoid question from HotpotQA,
“Alfie Allen played Theon Greyjoy on which show?”is correctly an-
swered acrossallretriever-generator configurations; for such sim-
ple queries, employing expensive retrieval and reasoning pipelines
incurs unnecessary computational cost without yielding any per-
formance gains. Conversely, the oracle distribution (Table 1) also
reveals that there are a substantial number of queries that are cor-
rectly answered only under an intermediate retrieval setting paired
with complex reasoner (e.g., 𝜃𝑀with𝜙𝐻), while other modes of
retrievers fail irrespective of generator choice. For instance, the
query“Belle Gold is a fictional character portrayed by an actress of
what nationality?”requires resolving a semantic entity relationship
(Belle Gold→portraying actress→nationality), where dense re-
trieval (𝜃𝑀) successfully retrieves semantically relevant evidence,
whereas other retrievers either miss relevant context or retrieved
noisy evidence. Thus, higher-complexity configurations are not
strict supersets of lower ones, and we needadaptive per-query
routing frameworkover the retriever-generator space Θ×Φ to
better balance downstream effectiveness and inference latency.
4 Dynamic RAG Workflow (DRAG)
Motivated by the observations in Section 3, which demonstrate
that no single configuration is optimal across queries, we intro-
duce DRAG, an adaptive framework that approximates the oraclerouting policy of Equation 1 when the ground-truth answer 𝑎𝑞
is unavailable at inference time. Depending on the availability of
supervision, we introduce two complementary approaches under
this paradigm: (i)DRAG QPP, a training-free framework that se-
quentially selects retrievers and generators using unsupervised per-
formance signals, and (ii)DRAG SFT, a supervised framework that
directly learns optimal, query-specific retriever-generator configu-
rations. An overview of the proposed DRAG workflow, including
both DRAG QPPand DRAG SFTframeworks, is illustrated in Figure 2.
4.1 DRAG QPP
As the initial step towards realizing an optimized adaptive frame-
work, we introduce DRAG QPP, a training-free routing framework
that approximates the oracle routing policy using unsupervised
techniques. Rather than directly predicting a joint configuration,
DRAG QPPdecomposes the routing process into two sequential pre-
diction stages: (i) Retriever Performance Prediction and (ii) Genera-
tor Performance Prediction. The selected retriever and generator
are then used for downstream answer generation.
4.1.1Retriever Performance Prediction (RPP).The first mod-
ule in DRAG QPPframework aims to select the most suitable re-
triever for each query by analyzing retrieval difficulty [ 62] without
requiring relevance judgments at inference time. Because adaptive
routing must decide required retriever complexitybeforeretrieval
is executed, we need aretriever-independentsignal that relies solely
the query text and corpus statistics. Query Performance Prediction
4

Dynamic Retriever and Generator Selection for RAG Conference’17, July 2017, Washington, DC, USA
(QPP) directly addresses this need by estimating retrieval quality
in the absence of relevance judgments [ 8,12]. Motivated by this,
we leverage apre-retrievalQPP predictor 𝜎(𝑞) :𝑞↦→R as a signal
to guide per-query retriever selection [ 9,40]; the larger values of
𝜎indicate queries expected to have relevant documents after re-
trieval. Intuitively, queries that are easier to retrieve can be handled
by lightweight retrievers, whereas harder queries are escalated to
progressively stronger retrieval pipelines.
QPP-aware Retriever Routing.Rather than evaluating the
effectiveness of all retrievers for each query, which would incur sub-
stantial computational overhead and defeat the purpose of adaptive
routing, we adopt a lightweight query-specific routing strategy that
sequentially selects the retrievers with increasing complexity. Since
retriever selection must be performed without actually perform-
ing each retrieval , we usedpre-retrievalQPP predictors, which are
retriever-independentand depend on the query and corpus statistics.
Formally, given a query 𝑞, we compute a QPP score 𝜎(𝑞) and
select the least complex retriever satisfying
𝜃∗(𝑞)=𝜃𝑖,where𝑖=min
𝑗𝜎(𝑞)≥𝜏𝑗	
,(2)
where{𝜏𝑗}𝑚−1
𝑗=1is a set of𝑚−1thresholds (where, 𝜏𝑗<𝜏𝑗+1,∀𝑗)
dividing the score range into 𝑚intervals. A query interval with
lower value of 𝜎indicate the need of more complex retrievers,
while easier queries computer higher 𝜎and are handled by simpler,
cheaper retrievers .
In our experiments, we employAverage IDF(Avg-IDF) [ 33] as our
lightweight, pre-retrieval QPP signal to capture query specificity
with respect to the underlying corpus. Queries dominated by rare,
highly informative content terms typically receive higher Avg-IDF
scores and are inherently easier to retrieve. Conversely, queries
with ambiguous or common terms yield lower scores, indicating
they would benefit from more advanced retrieval pipelines. We
select Avg-IDF aligned with recent literature demonstrating its
robustness [ 2], a choice further validated by our initial experiments
against alternative pre-retrieval predictors.
Threshold Selection.We derive the routing thresholds {𝜏𝑖}𝑚−1
𝑖=1
from the Avg-IDF score distribution computed over the training
set. Concretely, these thresholds are defined via percentile-based
partitioning of the predictor values. We select a base percentile 𝑝
as a tunable hyperparameter and determine subsequent thresholds
using uniform offsets, i.e., {𝑝,𝑝+Δ,𝑝+ 2Δ,...} , where Δcontrols
the interval spacing between adjacent complexity levels. The score
values corresponding to these specific percentiles are then fixed
as the routing thresholds {𝜏𝑖}, which are applied to the test data
to map incoming queries to their respective retriever complexity
levels based on their online QPP scores.
4.1.2Generator Performance Prediction (GPP).As the sec-
ond component of our framework, DRAG QPPselects an appropriate
generator for each query, conditioned on the specific retriever 𝜃∗se-
lected in the previous stage. To capture the generator’s uncertainty
regarding the retrieved evidence prior to answer synthesis, we eval-
uate the perplexity of the retrieved context sequence itself under
the base language model [ 52]. Perplexity is computed as the expo-
nential of the mean negative log-likelihood of the tokens within
the retrieved context documents. Notably, a lower perplexity scoreindicates that the retrieved evidence is highly consistent with the
model’s internal language distribution and parametric expectations.
Conversely, a high perplexity score signals that the context is noisy,
out-of-distribution, or poorly aligned, which drastically escalates
the risk of downstream hallucination [ 22,69]. Consequently, queries
paired with low-perplexity context are routed to lightweight gener-
ators, while queries yielding high-perplexity context are escalated
to more complex, reasoning-intensive models capable of handling
noisy evidence. Formally, for a given query 𝑞and retrieved set 𝜃∗
𝑘(𝑞),
we compute the generator-specific perplexity score ℎ(𝜃∗
𝑘(𝑞)|𝑞 ;𝜙𝑗)
and select the generator according to,
𝜙∗(𝑞)=𝜙𝑖,where𝑖=min
𝑗ℎ(𝜃∗
𝑘(𝑞)|𝑞;𝜙𝑗)≤𝜈𝑗	
.(3)
Threshold on Posterior.For each generator 𝜙𝑖, we estimate
the distribution of perplexity scores over training queries and de-
fine a routing threshold 𝜈𝑖as the score corresponding to the 𝑝-th
percentile (𝑝∈{ 10,20,...} ), where𝑝is treated as a hyperparameter
selected on train data. During inference, we select generators se-
quentially from low to high complexity. We compute the perplexity
of the retrieved context and compare it against the corresponding
threshold𝜈𝑖. If the perplexity exceeds 𝜈𝑖– implying that the query
can be answered by the current generation tier’s capacity to pro-
cess the retrieved context’s informational density – the current
generator is selected. Otherwise, the query is escalated to the next
higher-complexity tier. This process continues until a generator
satisfies this threshold criterion or the highest-complexity model is
deployed by default.
4.2 DRAG SFT
While DRAG QPPprovides a lightweight and training-free mecha-
nism for adaptive routing using query and corpus derived signals,
its effectiveness relies heavily on data-specific threshold selection
and the sequential estimation of generation complexity. This lim-
itation motivates a more generalizable supervised approach that
directly models retriever and generator selection jointly within a
unified framework.
To this end, DRAG SFTcasts routing as a supervised mapping ˆP:
𝑞↦→(𝜃,𝜙)from an input query to its optimal retriever–generator
configuration. We instantiate ˆPby fine-tuning Qwen as the base
model on our constructed training dataset Doracle (described in Sec-
tion 3), where each query is paired with its optimal configuration
label. The input context consists of the query, the instruction tem-
plate, and the available configurations, and the model is trained
to produce a structured JSON output of the form {"retriever":𝜃,
"generator":𝜙}, specifying the selected configuration.
At inference time, given a query 𝑞, the fine-tuned model predicts
the retriever–generator configuration (ˆ𝜃,ˆ𝜙)in a single decoding
step. The selected retriever ˆ𝜃retrieves the top- 𝑘contexts, which
are subsequently passed to the generator ˆ𝜙to produce the final
answer ˆ𝑎𝑞. By jointly predicting both components directly from
the query, DRAG SFTavoids the sequential routing and threshold
calibration required by DRAG QPP, while introducing negligible
upfront inference overhead.
5

Conference’17, July 2017, Washington, DC, USA Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit Bhatia
Dataset Task #Train #Test |𝑄|Domain
TriviaQA Single-hop 78,785 11,313 14.2 Wiki/Web
HotpotQA
Multi-hop90,447 7,405 17.8
Wiki MuSiQue 19,938 2,417 26.4
FRAMES - 824 16.0
Table 2:Characteristics of the datasets used in our experiments. We
used the entire test set for all evaluations. FRAMES is used as an
OOD benchmark while the remaining datasets are in-domain.
5 Experimental Setup
Our empirical evaluation is designed to answer the following core
research questions about the behavior of RAG pipelines.
•RQ1 (Retriever vs. Generator Capacity):How do retriever
and generator complexity individually contribute to end-to-end
RAG performance, and does scaling either component in isolation
consistently lead to downstream quality gains?
•RQ2 (Dynamic vs. Static RAG Trade-offs):Can dynamically
selecting the retriever and generator configuration optimize the
effectiveness-efficiency trade-off (EET) relative to conventional
static, fixed-complexity RAG pipelines?
•RQ3 (Framework Generalization):How effectively do our
proposed adaptive routing strategies generalize when evaluated
across out-of-domain datasets and unseen model configurations?
5.1 Datasets
We evaluate our framework on four question answering (QA) bench-
marks selected to represent a spectrum of retrieval and reasoning
complexities.TriviaQA[ 25] consists primarily of open-domain
factoid questions that are typically answerable using a single sup-
porting document, serving as a baseline for settings with limited
evidence dispersion. To capture increasing retrieval and reason-
ing difficulty, we consider three multi-hop QA benchmarks:Hot-
potQA[ 70], which requires bridging information across indirectly
connected documents,MuSiQue[ 64], which enforces stricter com-
positional reasoning (2-4 hops) with widely dispersed evidence
and, finally we includeFRAMES[ 31] as our multi-hop OOD eval-
uation benchmark, since, unlike the previous datasets, it involves
more diverse reasoning requirements (e.g., numerical, temporal,
and multi-constraint reasoning), often requiring aggregation across
a larger set of documents. Thus, FRAMES provides a challenging
testbed for evaluating the robustness and generalizability of the
proposed adaptive routing methodologies . Dataset statistics are
reported in Table 2.
5.2 Retrievers Investigated
The action space Θshould span retrieval pipelines that differ signif-
icantly in cost and capability, so the routing policy has substantial
trade-offs to learn. In this study, the granularity of |Θ|is itself a
design choice that can vary from simple binary routing to more
fine-grained adaptive configurations.
In this work, motivated by the evolution of retrievers and rankers
in IR, we consider Θas a spectrum of retrieval configurations with
progressively increasing complexity: lexical retrieval, dense seman-
tic retrieval, retrieve-rerank pipelines employing cross-encoders,and query decomposition followed by reranking. These config-
urations have a different spectrum of latency and effectiveness.
Although we focus on these dimensions, the framework is general
and can accommodate other configuration choices. Particularly, we
consider the following:
•Low (𝜃𝐿) (BM25): We use BM25 [ 48], a widely adopted lexical
model that relies on exact term matching. It is highly efficient
but has no semantic awareness.
•Medium (𝜃𝑀) (E5): We employe5-large-v2[ 66], a dense bi-
encoder trained using weakly supervised contrastive learning,
which captures semantic relationships under vocabulary mis-
match and is more computationally expensive than lexical one.
•High (𝜃𝐻) (E5%50>>BGE ): As the next level of complexity, we
employ a retrieve-and-rank model, where we use E5 to retrieve
the top-50documents, which are then re-ranked against the query
using BGE, a cross-encoder model [37].
•Extra-High ( 𝜃𝑋𝐻) (QD>>E5 %50>>BGE ): To address multi-
hop information needs, we implement a decompose-and-rerank
pipeline. The query is broken into 𝑠atomic sub-queries [ 1]. We
retrieve the top-50documents for each sub-query and the orig-
inal query using E5, creating a candidate pool of (𝑠+ 1)×50
documents, which are then re-ranked against the original query
using BGE [ 37]. This incurs the highest computational cost but
maximizes relevance for complex tasks.
5.3 Generators Investigated
The space of generator configurations should span varying levels of
reasoning capability and computational cost, enabling the routing
policy to adapt generation effort based on query complexity. We
evaluate three open-source LLM families that provide distinct mech-
anisms for scaling test-time compute [ 56]. Each backbone exposes
configurablereasoning-effortsettings that control the extent of in-
ternal chain-of-thought reasoning and inference-time computation.
Increasing the reasoning effort generally improves performance on
compositional and multi-hop queries, albeit at the cost of higher
token usage and increased inference latency.
•GPT-OSS[ 42] considers thereasoning effortparameter with
three settings,Low(minimal internal thought, lowest latency),
Medium(moderate reasoning budget), andHigh(extensive rea-
soning paths, highest cost but boost in downstream performance).
•Qwen-3[ 61] considers a binary thinking-mode switch,Low
(Thinking-Off) generates a direct response as a standard instruction-
tuned LLM, whileHigh(Thinking-On) emits an explicit step-by-
step reasoning trace before the final answer, improving multi-hop
performance at the cost of a linearly scaled token budget.
•Gemma-4[ 60] similar to the previous setup, it also considers
binary thinking control withLow(direct response) andHigh
(reasoning-trace-enabled) modes.
5.4 Baselines
We compare our adaptive methodology against the following base-
lines to evaluate the benefits of dynamic RAG configurations:
•0-shot: In this baseline, we directly predict the answer to a par-
ticular query without any supporting document, serves as lower
bound. It shows the impact of generator’s parametric knowledge.
6

Dynamic Retriever and Generator Selection for RAG Conference’17, July 2017, Washington, DC, USA
GPT-OSS (20B) Qwen (8B) Gemma (26B)
EM F1 Accuracy EM F1 Accuracy EM F1 Accuracy
Ret / Gen L M H L M H L M H L H L H L H L H L H L H
TriviaQA
L 53.3∗†55.1∗†56.7∗†63.5∗†64.7∗†64.8∗†65.4∗†66.4∗†67.4∗†57.4∗†57.1∗†65.5∗†64.9∗†66.0∗†64.8∗†49.6∗†52.1∗†59.1∗†61.4∗†61.6∗†63.4∗†
M 59.6∗†62.3∗†64.4∗†71.4∗†73.4∗†73.7†74.4∗†76.0∗†77.1†64.0∗†64.4†73.5∗†73.4∗†74.3∗†74.1∗†56.3∗†58.8†66.8∗†69.0†70.8∗†72.4∗†
H 61.5∗†64.2∗†66.1†73.3∗†75.4∗†75.7∗†76.3∗†77.9∗†78.4∗66.1∗†66.8∗75.7∗†75.9∗†76.5∗76.5∗58.6†60.1∗†69.3∗†71.1∗73.9∗†76.2∗
XH 60.8∗†63.2∗†65.2†72.6∗†74.3†74.8∗†75.6∗†77.1∗†77.9∗†64.9†65.6∗†74.3†74.1∗†75.4†74.7∗†57.6∗†59.7∗†68.3†70.5∗†72.6∗†74.1†
DRAG QPP 65.4 73.9 76.7 64.9 74.3 75.1 58.5 68.4 74.3
DRAG SFT 68.1 77.1 79.6 67.1 76.2 76.9 61.2 71.9 76.8
HotpotQA
L 25.5∗†28.8∗†30.2∗†35.6∗†38.1∗†38.9∗†35.8∗†36.4∗†37.4∗†26.7∗†27.8∗†35.5∗†35.4∗†33.4∗†33.5∗†25.2∗†26.8∗†33.1∗†34.4∗†33.1∗†33.9∗†
M 38.5∗†43.9∗†45.5∗†53.4∗†57.4∗†57.9∗†53.0∗†54.2∗†54.7∗†41.49∗†45.4∗†53.3∗†57.0∗†49.5∗†52.3∗†41.6∗†44.3∗†53.5∗†56.1∗†52.3∗†54.0∗†
H 44.0∗†49.9†51.6∗†60.1∗†64.5∗†65.2†59.5∗†60.9†61.5†46.4∗†50.9∗59.6∗†63.9∗55.2†58.7∗46.9∗†50.5∗†60.4∗†63.8∗†59.2†61.8∗
XH 41.2∗†47.4∗†48.8∗†57.3∗†61.6∗†62.3∗†57.0∗†57.7∗†58.6∗†43.9∗†48.1†56.5∗†60.3∗†52.5∗†53.3∗†44.6∗†50.5∗†57.5∗†62.7†56.6∗†60.1∗†
DRAG QPP 49.2 63.1 60.8 48.2 61.3 55.9 48.4 61.2 59.4
DRAG SFT 53.2 67.6 63.8 50.5 63.9 58.2 51.1 64.1 62.2
MuSiQue
L 7.1∗†7.2∗†7.6∗†11.4∗†11.4∗†11.7∗†10.1∗†10.6∗†12.0∗†3.9∗†5.5∗†7.9∗†9.2∗†6.5∗†8.2∗†4.4∗†4.5∗†7.6∗†7.3∗†7.5∗†7.1∗†
M 14.3∗†17.8∗†18.5∗†22.1∗†25.0∗†25.3∗†20.3∗†23.3∗†25.8∗†9.6∗†15.8∗†15.7∗†22.1∗†13.4∗†19.8†11.8∗†14.8∗†17.3∗†20.0∗†16.7∗†20.0∗†
H 19.5†21.1∗†22.8∗†27.7∗†29.7∗†30.1∗†26.0∗†28.4†31.1∗11.3∗†18.9∗†18.1∗†25.4∗†15.9∗†23.7∗†15.4†17.3∗20.9∗†22.9†21.0∗†23.9†
XH 18.6∗†21.6∗†22.1∗†26.8∗†29.8†29.3∗†25.1∗†28.1∗†30.5∗†11.0∗†17.9∗†17.6∗†24.0∗†15.3∗†22.2∗†14.3∗†17.7∗†19.8∗†23.2∗19.5∗†25.0∗
DRAG QPP 19.5 28.4 28.4 17.1 23.3 20.8 15.9 22.8 23.0
DRAG SFT 24.5 31.9 32.3 19.3 26.2 24.4 19.2 24.6 25.4
FRAMES
L14.2∗†13.2∗†12.1∗†22.9∗†20.6∗†18.5∗†18.2∗†17.5∗†17.7∗†8.1∗†12.6∗†13.9∗†19.5∗†9.8∗†13.8∗†6.3∗†9.5∗†11.2∗†15.0∗†8.1∗†11.3∗†
M18.3∗†18.1∗†18.2∗†28.4∗†27.2∗†27.1∗†21.5∗†22.2∗†25.1∗†12.6∗†18.0∗†20.2∗26.5∗†14.4∗†18.9†10.6∗†15.3†17.9∗†23.3∗†12.6∗†17.6†
H 19.4†19.5†17.6∗†28.6†28.8†26.3∗†22.9∗†23.3∗†23.5∗†10.7∗†17.0†19.1∗24.9†12.1∗†18.0†11.7∗†16.8†18.4∗†24.0∗†13.8∗†19.0∗
XH21.8∗†21.1∗20.4∗†32.1∗30.6∗†29.6†26.5∗25.2∗†26.0∗11.3∗†18.6∗†19.6∗†27.6∗13.1∗†20.3∗12.0∗†17.8∗†19.0∗†24.3∗†13.8∗†18.3∗†
DRAG QPP 19.3 29.5 23.9 15.8 24.8 18.4 15.3 22.9 16.3
DRAG SFT 23.3 33.9 27.2 19.5 28.1 21.3 18.6 25.8 18.9
Table 3:End-to-end RAG performance of three LLMs across four QA benchmarks under all retriever-generator configurations. Rows correspond
to retriever complexity levels ( 𝜃∈{L,M,H,XH} ), while columns correspond to generator complexity levels ( 𝜙∈{L,M,H} for GPT-OSS and
𝜙∈{L,H} for Qwen and Gemma). We report Exact Match (EM), F1, Accuracy, and average per-query inference time. Among static configurations,
boldnumbers denote the best result within each retriever block (best generator for a fixed retriever), while underlined numbers denote the
best result within each generator block (best retriever for a fixed generator). We additionally bold the best of the two proposed adaptive
methods. Asterisks (∗) and Dagger (†) indicate statistically significant differences compared to DRAG QPPand DRAG SFTrespectively ( 𝑝< 0.05).
All reported results use 5 contextual examples.
•RAG max: This represents a standard static RAG setup, where
a fixed configuration is uniformly applied to all queries, and
performance is reported for the best-performing configuration
over the dataset.
•RAG Orac: This represents the theoretical upper bound of an ideal
adaptive RAG system as mentioned in Equation 1 of Section 3
(Efficiency-aware Oracle Configuration).
•DRAG AP: Our routing policy of RPP in DRAG QPP(in Equation 2)
is dependent on pre-retrieval QPP, thus we define an upper-bound
by replacing QPP with the ground-truth retrieval effectiveness
(AP@50) of each retriever. This yields a retrieval-aware, sequen-
tial routing strategy that estimates the maximum achievable gain
under perfect effectiveness signals.
•DRAG PPT: In this case, given a query, the generator is provided
with a natural language description of the available retriever-
generator configurations and is asked to select the most suitable
one, relying solely on parametric knowledge rather than retrieved
evidence. The chosen configuration is then used to generate the
final answer. This shows whether an LM can infer appropriate
configurations from the query’s information need. This is a basic
baseline from DRAG’s perspective.5.5 Experimental Details
Evaluation Metrics.We assess theeffectiveness of the down-
stream taskusing three standard metrics: Exact Match (EM), F1,
and Accuracy (Acc), which capture exact correctness, partial lexical
overlap, and answer containment, respectively. We reportstatis-
tical significanceat the query level using McNemar’s test for
EM and Accuracy (due to comparison between binary outcomes),
and a paired 𝑡-test for F1, which captures the comparison between
systems. Theretrieval qualityis evaluated using AP@10 to assess
the relevance of the retrieved contexts provided to the generator.
Hyperparameter Details.For Extra-High mode of retriever
we use same the target LLM to decompose each query into atomic
sub-queries (temperature =0.1). In this paper, for RPP module
in DRAG QPPwe employ Avg-IDF as we pre-retrieved QPP which
is computed over query terms with respect to the corpus. It uses
thresholds set of75th,50th, and25thpercentiles for distinguishing
all four retriever levels. For GPP, we use threshold set of perplexity
scores at the20thpercentile of the training distribution (where lower
is better). Additionally, for DRAG SFTwe used Qwen3-4B-Instruct .
We train for 5 epochs with a learning rate of 2e-5, a cosine decay
schedule with warmup ratio 0.15, weight decay 0.05, and an effective
batch size of 256. The maximum sequence length is set to 1,280
tokens. The best checkpoint is selected based on combined (retrieval
7

Conference’17, July 2017, Washington, DC, USA Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit Bhatia
TriviaQA HotpotQA MuSiQue FRAMES (OOD)
Model EM F1 Acc t(s) EM F1 Acc t(s) EM F1 Acc t(s) EM F1 Acc t(s)GPT-OSSRAG Orac 77.5 86.1 85.52.5663.9 76.9 71.91.8337.1 44.5 43.35.3638.9 52.4 44.34.89
RAG max 66.1 75.7 78.4 10.73 51.6 65.2 61.5 4.36 22.8 30.1 31.1 10.2 21.8 32.1 26.5 5.14
0-shot 42.5 52.3 53.0 1.02 20.7 27.9 34.8 0.83 5.4 10.4 21.6 1.73 10.9 17.6 23.8 1.72
DRAG AP 67.3 76.2 78.9 5.74 52.3 67.9 64.2 2.67 24.6 32.2 32.5 6.43 22.8 33.7 27.8 3.20
DRAG PPT 52.7 65.8 65.2 5.12 43.7 57.9 55.3 2.87 16.8 26.7 26.9 6.45 16.4 27.3 21.1 3.22
DRAG QPP 65.4 73.9 76.7 5.14 49.2 63.1 60.8 2.18 19.5 28.4 28.4 6.36 19.3 29.5 23.9 3.15
DRAG SFT 68.1 77.1 79.6 5.37 53.2 67.6 63.8 3.11 24.5 31.9 32.3 6.85 23.3 33.9 27.2 3.28QwenRAG Orac 76.0 84.3 83.32.3461.8 75.3 68.91.1329.6 39.8 35.31.6531.8 45.1 33.70.78
RAG max 66.8 75.9 76.5 8.76 50.9 63.9 58.7 3.16 18.9 25.4 23.7 8.64 18.6 27.5 20.3 4.39
0-shot 50.6 58.3 56.5 0.82 21.3 29.8 24.1 0.86 4.9 11.6 7.5 1.07 10.4 17.9 11.5 1.41
DRAG AP 67.2 75.6 76.3 4.37 50.8 63.9 58.9 1.28 19.9 26.5 24.9 4.87 18.8 27.8 21.1 2.30
DRAG PPT 54.7 67.8 66.5 4.35 46.3 58.4 54.4 1.18 15.3 21.6 19.2 4.76 12.7 20.5 16.8 1.93
DRAG QPP 64.9 74.3 75.1 4.10 48.2 61.3 55.9 1.21 17.1 23.3 20.8 4.23 15.8 24.8 18.4 1.84
DRAG SFT 67.1 76.2 76.9 4.46 50.5 63.9 58.2 1.37 19.3 26.2 24.4 5.10 19.5 28.1 21.3 2.14GemmaRAG Orac 76.5 85.4 84.22.8663.3 76.7 72.21.3227.7 36.1 36.16.8727.3 38.2 29.84.36
RAG max 60.1 71.1 76.2 10.86 50.4 63.8 61.8 4.87 17.7 23.2 24.9 11.13 17.7 24.3 18.3 5.68
0-shot 55.8 64.3 66.9 1.32 27.1 36.8 30.3 1.09 9.4 18.8 11.9 1.04 16.3 26.1 18.0 1.55
DRAG AP 61.4 71.7 75.8 5.7 50.8 63.4 60.5 2.7 17.4 24.4 25.1 7.7 18.2 25.5 18.6 3.4
DRAG PPT 51.8 61.6 64.5 3.77 45.8 59.3 57.3 2.1 15.2 22.3 22.8 7.3 15.1 21.2 16.5 3.1
DRAG QPP 58.5 68.4 74.3 5.34 48.4 61.2 59.4 2.2 15.9 22.8 23.0 7.16 15.3 22.9 16.3 3.35
DRAG SFT 61.2 71.9 76.8 5.67 51.1 64.1 62.2 2.30 19.2 24.6 25.4 7.34 18.6 25.8 18.9 3.51
Table 4:We compare static, oracle, and adaptive RAG configurations across TriviaQA, HotpotQA, MuSiQue, and FRAMES. Our adaptive
methods, DRAG QPP(training-free) and DRAG SFT(supervised), select retriever–generator from available configurations. We report EM, F1-score,
Accuracy, and average inference latency (time/second, represented in table as t(s)). The number of contexts used for few-shot experiments is
5. All evaluations are conducted on the full test sets to ensure statistical robustness.Bolddenotes the best result among static and adaptive
methods, while underline indicates the best over our proposed methods (i.e., grey portion).
+ generator) macro-F1. For generation, we use GPT-OSS (20B1),
Qwen3 (8B2) and Gemma (26B3) both deployed using the vLLM
inference engine. To accommodate the necessary reasoning tokens
required by these strategies, we set the maximum generation budget
(max_new_tokens ) to512,2048, and4096for theLow,Medium,
andHighmodes, respectively. All experiments are conducted on
NVIDIA A100 GPUs with a batch size of12.
6 Results and Discussion
We present our experimental results to answer the proposed re-
search questions in Section 5.
6.1 Effect of Retriever and Generator
Complexity
Table 3 reports the performance of each retriever-generator pair
in terms of EM, F1-score, and Accuracy across all four QA bench-
marks using three different families of LLMs. To addressRQ1, we
examine how varying the complexityindividuallyinfluences end-
to-end RAG performance, and whether scaling either component
separately consistently yields gains.
Overall, we observe thatincreasing component complexity
leads to performance improvement, but the gains are not
monotonic. When we fix the choice of generators and increase
1https://huggingface.co/openai/gpt-oss-20b
2https://huggingface.co/Qwen/Qwen3-8B
3google/gemma-4-26B-A4B-itthe retriever complexity, performance generally improves from lex-
ical retrieval ( 𝜃𝐿) to dense (𝜃𝑀) and reranking-based pipelines ( 𝜃𝐻),
with gains in all datasets, particularly higher gains for multi-hop
benchmarks such as HotpotQA, MuSiQue, and FRAMES, where
evidence is dispersed across multiple documents and stronger re-
trieval is needed. However, it saturates at higher retrieval levels
(e.g.,𝜃𝐻→𝜃𝑋𝐻) for all datasets except FRAMES, i.e.,diminish-
ing returns from retrieval scalingdespite substantially higher
computational cost . A likely reason of the effect is less evident in
FRAMES is that its queries impose multiple constraints (numerical,
temporal, multi-entity) and require evidence assembled from gen-
uinely distinct sub-aspects, so atomic sub-query retrieval supplies
a signal that single-pass retrievers miss.
In contrast, increasing generator reasoning effort under a fixed
retriever (particularly, for weak retrieval) shows comparatively
smaller and inconsistent gains for all datasets. For example, for
multi-hop dataset, MuSiQue, increasing reasoning effort under
weak retrieval ( 𝜃𝐿) yields only a7% improvementin terms of
EM, but improves performance by16.9%in terms of EM for 𝜃𝐻. It
shows thatreasoning cannot compensate for poor retrieval
qualityandgenerator effectiveness is fundamentally con-
strained by retrieval quality, with stronger retrieval typically
yielding larger gains than increased reasoning effort alone.
Effectiveness–Efficiency Trade-off.Table 3 reports down-
stream effectiveness, while Table 5 shows per-query latency for
static RAG configurations. We observe a clear trade-off between
effectiveness and computational cost, i.e., increasing retriever or
8

Dynamic Retriever and Generator Selection for RAG Conference’17, July 2017, Washington, DC, USA
(a) Static RAG Latency
𝜙\𝜃L M H XH
L 0.11 0.33 1.04 3.51
M 0.38 0.67 2.34 4.10
H 1.02 1.70 4.36 5.87(b) DRAG Latency Breakdown
Time DRAG QPP DRAG SFT
Decision 0.28 0.36
RAG 1.90 2.75
Total 2.18 3.11
Table 5:Per-query latency (seconds) on HotpotQA using GPT-OSS.
(a) Static retriever-generator configurations across retrieval (𝜃) and
generator (𝜙) complexity levels. (b) Latency breakdown of adaptive
methods, where decision time denotes routing overhead (IDF com-
putation and perplexity estimation) and RAG time corresponds to
downstream retrieval and generation. Retrieval indexing is excluded,
as it is a one-time offline preprocessing cost. Green and redcells
indicate, respectively, lower and higher total latency w.r.t DRAG SFT.
generator complexity incurs substantially higher latency while pro-
viding only marginal downstream gains. For example, on HotpotQA
(GPT-OSS), increasing generator complexity from 𝜙𝑀to𝜙𝐻under
𝜃𝐻improves EM only marginally (49.9 →51.6), while increasing
latency (2.34s→4.36s, per query). Similarly, under 𝜃𝑋𝐻, the same in-
crease very marginally (47.4 →48.8) despite a high latency increase
(4.10s→5.87s, per query). Moreover, stronger configurations are not
always strictly better (e.g., 𝜃𝐻outperforming 𝜃𝑋𝐻), souniformly
allocating maximum retrieval and reasoning effort is ineffi-
cient. These findings motivate anadaptive routing strategythat
dynamically allocates retrieval and generation complexity based
on query requirements.
6.2 Dynamic vs. Static RAG Performance
To address RQ2, Table 4 compares our adaptive methods against
oracle setup, RAG Oracand other baselines across downstream QA
metrics for three LLM families. Being an oracle setup, RAG Orac
consistently outperforms all adaptive methods and baselines. This
performance gap highlights the difficulty of routing under real-
istic settings, where configuration decisions must be made with
or without supervision. Among the baselines, DRAG PPTconsis-
tently outperforms 0-shot across nearly all datasets (except Triv-
iaQA with Gemma), suggesting that directly prompting an LLM
to predict the answer is insufficient, whereas explicitly predicting
the retriever-generator configuration provides a more appropri-
ate downstream prediction. Furthermore, DRAG APconsistently im-
proves over DRAG QPPacross datasets, indicating that retrieval qual-
ity estimation remains a key bottleneck for unsupervised adaptive
routing. Notably, both DRAG SFTand DRAG APoutperform the static
configuration, RAG max, thus query-adaptive selection is more effec-
tive than uniformly assigning a fixed high-complexity pipeline to
all queries. Finally, DRAG SFTconsistently improves over DRAG QPP,
suggesting that jointly learning query-specific retriever-generator
interactions is more effective than sequential threshold-based rout-
ing, albeit at the cost of supervision.
Performance vs Latency.Figure 1 and Tables 4 illustrate the
effectiveness and latency (sec/query) values of adaptive and static
RAG systems. Overall, increasing retriever and generator complex-
ity improves downstream performance but at substantially higher
inference cost, with marginal gains at stronger configurations. We
Figure 3:Effect of change in retrieved document count ( 𝑘) on the per-
formance of HotpotQA dataset. We analyze F1 scores under varying
Retriever (𝜃) and Generator ( 𝜙) complexities. The results demon-
strate how increasing the number of documents impacts perfor-
mance across different system configurations. Each line corresponds
to a retrieval complexity. The shared legend applies to all subplots.
Figure 4:Impact of retriever effectiveness on downstream QA per-
formance using GPT-OSS on HotpotQA and MuSiQue. MAP@10
measures retrieval effectiveness, while F1 measures end-to-end QA
performance under static RAG configurations. The bars correspond
to retriever modes.
also observea clear trade-off emerges between our two pro-
posed DRAG methods, DRAG SFTconsistently achieves better
downstream effectiveness than DRAG QPP, typically improving per-
formance by approximately 2-12%across datasets, but incurs a
modest increase in latency (approximately 5-40%). While DRAG QPP
remains training-free and lightweight, DRAG SFTleverages supervi-
sion to model query-specific interactions, resulting in better down-
stream performance with higher inference cost.
9

Conference’17, July 2017, Washington, DC, USA Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit Bhatia
RAG Orac DRAG SFT
Query and Gold Answer Config Output Config Output Failure Type
What is Mohamed Atta’s car, manufactured by the com-
pany that includes Datsun, an instance of?Ans:Nissan
AltimaXH / La 2001 Nissan AltimaXH / Ha blue rental carOverestimation
What is the oldest college in the state that contains Bor-
ough House Plantation?Ans:College of Charles TowneH / HCollege of Charles TowneL / MColumbia UniversityUnderestimation
What is the tallest building in the state where KZAR is
located?Ans:JPMorgan Chase TowerH / HJPMorgan Chase TowerL / MTransamerica TowerOverestimation
What is the position of the first governor general of India?
Ans:Governor-General of IndiaL / HGovernor-General of British IndiaL / LWarren HastingsWrong Entity
What team does the person who hit the first home run at
the new South Bronx stadium play for?Ans:New York
YankeesM / LNew York YankeesH / HIndiansWrong Entity
Table 6:Qualitative failure cases on MuSiQue dataset. Each row shows a query where DRAG SFTselects the failure/suboptimal cases relative to
the oracle RAG Orac, along with the resulting output error. Config columns represent the mode ofretriever/ generator.never use vspace; the
publishers won’t allow this anyway
Significance Testing.Table 3 further highlights the statisti-
cal behavior of the DRAG framework. Across in-domain datasets,
DRAG SFTconsistently yields statistically significant gains over
static configurations, whereas DRAG QPPoften remains statistically
comparable to static RAG with medium/high-complexity genera-
tors. Under the OOD setting (FRAMES), although DRAG SFTgener-
ally maintains stronger performance, improvements are not always
statistically significant, suggesting that transferring routing deci-
sions across datasets with different query types remains challeng-
ing.
Latency.Table 5 reports the per-query latency breakdown of
adaptive RAG systems. We observe that the routing cost, i.e., model
configuration prediction cost, is relatively less. Instead, most of the
latency difference arises from retrieval and downstream generation
(1.90s vs.2.75s RAG time), here DRAG SFTpicks higher-complexity
configurations, trading latency for effectiveness, while DRAG QPP
favors cheaper settings at the cost of a modest accuracy drop.
6.3 Model Generalization
To address RQ3, we evaluate the robustness of our adaptive meth-
ods under an OOD setting using FRAMES, where models don’t
rely on dataset-specific training (i.e., use training data of MuSiQue
and are evaluated directly on FRAMES). From Tables 4 and 3, we
observe that both DRAG QPPand DRAG SFTprovide better results
w.r.t all the baselines, including DRAG APacross all three LLM fam-
ilies. Interestingly, Table 3 shows that for GPT-OSS on FRAMES
dataset, the best static RAG performance is achieved under low or
medium-reasoning generator settings with a fixed retriever choice,
thus uniformly increasing reasoning effort is not always beneficial
under distribution shift. In contrast, our adaptive methods not only
achieve better downstream effectiveness than RAG maxand other
baselines, but also incur a lower inference cost. Thus, our proposed
framework, DRAG is robust and generalizable across OOD datasets.
6.4 Ablation Studies
Impact of Increasing Context Size.Figure 3 analyzes the sen-
sitivity of RAG performance to the number of retrieved documents
𝑘across retrieval complexity levels and generator reasoning modes.
While low-complexity retrieval ( 𝜃𝐿) shows steady improvementwith increasing 𝑘, reflecting its lower per-document precision and
reliance on broader context coverage, higher-complexity retrievers
(𝜃𝑀,𝜃𝐻,𝜃𝑋𝐻) plateau early, as their stronger relevance signals are
largely captured within a compact candidate set. This saturation
behavior is consistent across all three generator reasoning levels,
suggesting it is driven by retrieval quality rather than generation
capacity, and collectively justifies our choice of 𝑘=5as a balance
between evidence coverage and computational efficiency.
Correlation between retrieval and downstream perfor-
mance.Figure 4 examines the relationship between retrieval effec-
tiveness (MAP@10) and downstream QA performance (F1) across
retrieval complexity levels on HotpotQA and MuSiQue. A clear pos-
itive correlation is observed: as retrieval complexity increases from
lexical (𝜃𝐿) to dense (𝜃𝑀) and reranking-based pipelines ( 𝜃𝐻,𝜃𝑋𝐻),
both MAP@10 and F1 improve consistently and proportionally
across both benchmarks, confirming that stronger retrieval directly
translates to better answer generation, particularly for multi-hop
queries where evidence is dispersed across multiple documents.
6.5 Per-Query Analysis
Table 6 presents qualitative failure cases of our best performing
adaptive method DRAG SFT(as shown in Table 3) on MuSiQue, re-
vealing three primary error patterns.Overestimationoccurs when
DRAG SFTroutes simple queries to unnecessarily expensive config-
urations (e.g., 𝜃𝐻/𝜙𝑋𝐻instead of𝜃𝐿/𝜙𝑋𝐻), introducing noisy evi-
dence that degrades answer quality.Underestimationoccurs when
DRAG SFTassigns insufficient capacity to genuinely complex queries
(e.g.,𝜃𝑀/𝜙𝐿instead of𝜃𝐻/𝜙𝐻), leading to factually incorrect an-
swers. Finally,wrong entityerrors occur when suboptimal config-
uration selection (e.g., 𝜃𝐻/𝜙𝐻instead of𝜃𝐿/𝜙𝑀) causes the model
to retrieve incorrect evidence, leading to entity confusion during
multi-hop reasoning. Together, these cases highlight that routing
errors are systematically tied to query complexity estimation, and
that better query representations could further improve adaptive
configuration selection.
7 Conclusions and Future Work
We presented a systematic study of how retrieval complexity and
generation effort jointly affect end-to-end RAG performance. Our
10

Dynamic Retriever and Generator Selection for RAG Conference’17, July 2017, Washington, DC, USA
results show that increasing complexity does not consistently im-
prove accuracy, and that high-complexity configurations are not
strict supersets of simpler ones. This highlights the non-monotonic,
query-dependent nature of RAG and the limitations of fixed re-
triever–generator settings. To address this, we introduced a training-
free approach (DRAG QPP) and a supervised approach (DRAG SFT) un-
der a common framework DRAG for per-query retriever-generator
selection. Across multiple QA benchmarks, DRAG QPPmatches
strong static baselines while reducing latency, and DRAG SFTfurther
improves the effectiveness–efficiency trade-off, achieving perfor-
mance close to oracle routing.
Future work will focus on developing more expressive routing
policies that better capture the non-separable interactions between
retrieval and generation, including richer supervision and reinforce-
ment learning for optimizing cost–quality trade-offs.
GenAI Usage Disclosure
Generative AI tools were not used for core idea generation or exper-
imental design. Its use was limited to minor writing and formatting.
References
[1]Paul J. L. Ammann, Jonas Golde, and Alan Akbik. 2025. Question Decomposition
for Retrieval-Augmented Generation. InACL (4). Association for Computational
Linguistics, 497–507.
[2]Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, and Matei Zaharia. 2026.
Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection
for RAG Pipelines.CoRRabs/2604.22661 (2026).
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InICLR. OpenReview.net.
[4]Adrian-Gabriel Chifu, Léa Laporte, Josiane Mothe, and Md. Zia Ullah. 2018. Query
Performance Prediction Focused on Summarized Letor Features. InSIGIR. ACM,
1177–1180.
[5]Ronan Cummins, Joemon M. Jose, and Colm O’Riordan. 2011. Improved query
performance prediction using standard deviation. InSIGIR. ACM, 1089–1090.
[6]Robert Dale. 2021. GPT-3: What’s it good for?Nat. Lang. Eng.27, 1 (2021),
113–118.
[7]Suchana Datta, Debasis Ganguly, Derek Greene, and Mandar Mitra. 2022. Deep-
QPP: A Pairwise Interaction-based Deep Learning Model for Supervised Query
Performance Prediction. InWSDM. ACM, 201–209.
[8]Suchana Datta, Debasis Ganguly, Sean MacAvaney, and Derek Greene. 2024. A
Deep Learning Approach for Selective Relevance Feedback. InECIR (2) (Lecture
Notes in Computer Science). Springer, 189–204.
[9]Suchana Datta, Debasis Ganguly, Mandar Mitra, and Derek Greene. 2023. A
Relative Information Gain-based Query Performance Prediction Framework with
Generated Query Variants.ACM Trans. Inf. Syst.41, 2 (2023), 38:1–38:31.
[10] Claude De Loupy and Patrice Bellot. 2000. Evaluation of document retrieval
systems and query difficulty. InLREC Workshop, Athens. 32–39.
[11] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh,
and Matt Gardner. 2019. DROP: A Reading Comprehension Benchmark Requiring
Discrete Reasoning Over Paragraphs. InProceedings of the 2019 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and Short Papers), Jill Burstein, Christy
Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics,
Minneapolis, Minnesota, 2368–2378. https://doi.org/10.18653/v1/N19-1246
[12] Sajad Ebrahimi, Maryam Khodabakhsh, Negar Arabzadeh, and Ebrahim Bagheri.
2024. Estimating Query Performance Through Rich Contextualized Query Rep-
resentations. InECIR (4) (Lecture Notes in Computer Science). Springer, 49–58.
[13] Matteo Gabburo, Nicolaas Paul Jedema, Siddhant Garg, Leonardo F. R. Ribeiro,
and Alessandro Moschitti. 2024. Measuring Retrieval Complexity in Question
Answering Systems. InACL (Findings) (Findings of ACL). Association for Compu-
tational Linguistics, 14636–14650.
[14] Debasis Ganguly, Johannes Leveling, Walid Magdy, and Gareth J. F. Jones. 2011.
Patent query reduction using pseudo relevance feedback. InCIKM. 1953–1956.
[15] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi,
Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2023.
Retrieval-Augmented Generation for Large Language Models: A Survey.CoRR
abs/2312.10997 (2023).
[16] Michael R. Glass, Gaetano Rossiello, Md. Faisal Mahbub Chowdhury, Ankita
Naik, Pengshan Cai, and Alfio Gliozzo. 2022. Re2G: Retrieve, Rerank, Generate.InNAACL-HLT. Association for Computational Linguistics, 2701–2715.
[17] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. Retrieval Augmented Language Model Pre-Training. InICML (Proceedings
of Machine Learning Research). PMLR, 3929–3938.
[18] Ben He and Iadh Ounis. 2006. Query performance prediction.Inf. Syst.31, 7
(2006), 585–594.
[19] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. InEACL. Association
for Computational Linguistics, 874–880.
[20] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo
Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave.
2023. Atlas: Few-shot Learning with Retrieval Augmented Language Models.J.
Mach. Learn. Res.24 (2023), 251:1–251:43.
[21] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024.
Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models
through Question Complexity. InNAACL-HLT. Association for Computational
Linguistics, 7036–7050.
[22] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in
Natural Language Generation.ACM Comput. Surv.55, 12 (2023), 248:1–248:38.
[23] Zhen Jia, Abdalghani Abujabal, Rishiraj Saha Roy, Jannik Strötgen, and Gerhard
Weikum. 2018. TempQuestions: A Benchmark for Temporal Question Answering.
InCompanion Proceedings of the The Web Conference 2018(Lyon, France)(WWW
’18). International World Wide Web Conferences Steering Committee, Republic
and Canton of Geneva, CHE, 1057–1062. https://doi.org/10.1145/3184558.3191536
[24] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InEMNLP. Association for Computational Linguistics,
7969–7992.
[25] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. TriviaQA:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehen-
sion. InACL (1). Association for Computational Linguistics, 1601–1611.
[26] Evangelos Kanoulas, Ben Carterette, Paul D. Clough, and Mark Sanderson. 2011.
Evaluating multi-query sessions. InSIGIR. ACM, 1053–1062.
[27] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InEMNLP (1). Association for Computational
Linguistics, 6769–6781.
[28] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike
Lewis. 2020. Generalization through Memorization: Nearest Neighbor Language
Models. InICLR. OpenReview.net.
[29] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT. InProceedings of the 43rd
International ACM SIGIR conference on research and development in Information Re-
trieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy Huang, Yi Chang,
Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.).
ACM, 39–48. https://doi.org/10.1145/3397271.3401075
[30] Michael Kleinman, Matthew Trager, Alessandro Achille, Wei Xia, and Ste-
fano Soatto. 2025. e1: Learning Adaptive Control of Reasoning Effort.CoRR
abs/2510.27042 (2025).
[31] Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam
Stambler, Shyam Upadhyay, and Manaal Faruqui. 2025. Fact, Fetch, and Reason: A
Unified Evaluation of Retrieval-Augmented Generation. InNAACL (Long Papers).
Association for Computational Linguistics, 4745–4759.
[32] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics7 (2019), 453–466.
[33] K. L. Kwok. 1996. A New Method of Weighting Query Terms for Ad-Hoc Retrieval.
InSIGIR. ACM, 187–195.
[34] Tian Lan, Deng Cai, Yan Wang, Heyan Huang, and Xian-Ling Mao. 2023. Copy
is All You Need. InICLR. OpenReview.net.
[35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InNeurIPS.
[36] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard
James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. RA-DIT: Retrieval-Augmented Dual Instruction
Tuning. InICLR. OpenReview.net.
[37] Kun Luo, Zheng Liu, Shitao Xiao, and Kang Liu. 2024. BGE Landmark Embedding:
A Chunking-Free Embedding Method For Retrieval Augmented Long-Context
Large Language Models.CoRRabs/2402.11573 (2024).
[38] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Han-
naneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. InACL (1). Associa-
tion for Computational Linguistics, 9802–9822.
11

Conference’17, July 2017, Washington, DC, USA Neeraj Anand, Payel Santra, Partha Basuchowdhuri, Debasis Ganguly, and Sumit Bhatia
[39] Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and
Maarten de Rijke. 2024. Ranked List Truncation for Large Language Model-based
Re-Ranking. InSIGIR. ACM, 141–151.
[40] Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and
Maarten de Rijke. 2025. Query Performance Prediction Using Relevance Judg-
ments Generated by Large Language Models.ACM Trans. Inf. Syst.43, 4 (2025),
106:1–106:35.
[41] Josiane Mothe and Ludovic Tanguy. 2005. Linguistic features to predict query
difficulty. InSIGIR, Predicting query difficulty-methods and applications workshop.
7–10.
[42] OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card.CoRRabs/2508.10925
(2025).
[43] Andrew Parry, Debasis Ganguly, and Manish Chandra. 2024. "In-Context Learn-
ing" or: How I learned to stop worrying and love "Applied Information Retrieval".
InSIGIR. ACM, 14–25.
[44] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and
Thomas Bäck. 2026. Multi-Step Reasoning with Large Language Models, a Survey.
ACM Comput. Surv.58, 6 (2026), 160:1–160:35.
[45] Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. 2021. The Expando-Mono-
Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence
Models.CoRRabs/2101.05667 (2021).
[46] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankZephyr: Ef-
fective and Robust Zero-Shot Listwise Reranking is a Breeze!CoRRabs/2312.02724
(2023).
[47] Peng Qi, Haejun Lee, Tg Sido, and Christopher D. Manning. 2021. Answering
Open-Domain Questions of Varying Reasoning Steps from Text. InEMNLP (1).
Association for Computational Linguistics, 3599–3614.
[48] Stephen Robertson. 2004. Understanding inverse document frequency: on theo-
retical arguments for IDF.J. Documentation60, 5 (2004), 503–520.
[49] Haggai Roitman, Shai Erera, and Bar Weiner. 2017. Robust Standard Deviation
Estimation for Query Performance Prediction. InICTIR. ACM, 245–248.
[50] Payel Santra, Madhusudan Ghosh, Debasis Ganguly, Partha Basuchowdhuri,
and Sudip Kumar Naskar. 2025. HF-RAG: Hierarchical Fusion-based RAG with
Multiple Sources and Rankers. InCIKM. ACM, 5202–5207.
[51] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented
Black-Box Language Models. InNAACL-HLT. Association for Computational
Linguistics, 8371–8384.
[52] Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z. Ren, and Anirudha Majumdar.
2026. A Survey on Uncertainty Quantification of Large Language Models: Tax-
onomy, Open Research Challenges, and Future Directions.ACM Comput. Surv.
58, 3 (2026), 63:1–63:38.
[53] Anna Shtok, Oren Kurland, and David Carmel. 2010. Using statistical decision
theory and relevance models for query-performance prediction. InSIGIR. ACM,
259–266.
[54] Anna Shtok, Oren Kurland, David Carmel, Fiana Raiber, and Gad Markovits. 2012.
Predicting Query Performance by Query-Drift Estimation.ACM Trans. Inf. Syst.
30, 2 (2012), 11:1–11:35.
[55] Nilanjan Sinhababu, Andrew Parry, Debasis Ganguly, Debasis Samanta, and
Pabitra Mitra. 2024. Few-shot Prompting for Pairwise Ranking: An Effective Non-
Parametric Retrieval Model. InEMNLP (Findings) (Findings of ACL). Association
for Computational Linguistics, 12363–12377.
[56] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling LLM Test-
Time Compute Optimally can be More Effective than Scaling Model Parameters.
CoRRabs/2408.03314 (2024).
[57] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long Papers), Marilyn Walker, Heng Ji, and
Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans,
Louisiana, 641–651. https://doi.org/10.18653/v1/N18-1059
[58] Manveer Singh Tamber, Ronak Pradeep, and Jimmy Lin. 2023. Scaling Down, LiT-
ting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder
Models.CoRRabs/2312.16098 (2023).
[59] Yongquan Tao and Shengli Wu. 2014. Query Performance Prediction By Consid-
ering Score Magnitude and Variance Together. InCIKM. ACM, 1891–1894.
[60] Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev,
Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon,
et al. 2026. Gemma 4 technical report.arXiv preprint arXiv:2607.02770(2026).
[61] Qwen Team. 2025. Qwen3 Technical Report.CoRRabs/2505.09388 (2025).
[62] Fangzheng Tian, Debasis Ganguly, and Craig Macdonald. 2026. Predicting Re-
trieval Utility and Answer Quality in Retrieval-Augmented Generation. InECIR
(1) (Lecture Notes in Computer Science). Springer, 368–385.
[63] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models.ArXiv
preprintabs/2302.13971 (2023). https://arxiv.org/abs/2302.13971[64] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. ♪MuSiQue: Multihop Questions via Single-hop Question Composition.
Trans. Assoc. Comput. Linguistics10 (2022), 539–554.
[65] Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi,
and Bryan Catanzaro. 2024. InstructRetro: Instruction Tuning post Retrieval-
Augmented Pretraining. InICML (Proceedings of Machine Learning Research).
PMLR / OpenReview.net, 51255–51272.
[66] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised
Contrastive Pre-training.CoRRabs/2212.03533 (2022).
[67] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and
Furu Wei. 2024. Improving Text Embeddings with Large Language Models. In
ACL (1). Association for Computational Linguistics, 11897–11916.
[68] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain
of Thought Reasoning in Language Models. InICLR. OpenReview.net.
[69] Shicheng Xu, Liang Pang, Huawei Shen, and Xueqi Cheng. 2025. A Theory
for Token-Level Harmonization in Retrieval-Augmented Generation. InICLR.
OpenReview.net.
[70] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InEMNLP. Association for
Computational Linguistics, 2369–2380.
[71] Mohammad Yani and Adila Alfa Krisnadhi. 2021. Challenges, Techniques, and
Trends of Simple Knowledge Graph Question Answering: A Survey.Inf.12, 7
(2021), 271.
[72] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InThe Eleventh International Conference on Learning Representations,
ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview.
net/pdf?id=WE_vluYUL-X
[73] Qinyuan Ye, Iz Beltagy, Matthew E. Peters, Xiang Ren, and Hannaneh Hajishirzi.
2023. FiD-ICL: A Fusion-in-Decoder Approach for Efficient In-Context Learning.
InACL’23 (1). 8158–8185.
[74] Hamed Zamani, W. Bruce Croft, and J. Shane Culpepper. 2018. Neural Query Per-
formance Prediction using Weak Supervision from Multiple Signals. InSIGIR’18.
105–114.
[75] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi
Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-Level
Code Completion Through Iterative Retrieval and Generation. InEMNLP’23.
2471–2484.
[76] Longhui Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang,
and Min Zhang. 2024. A Two-Stage Adaptation of Large Language Models for
Text Ranking. InACL’24 (Findings). 11880–11891.
[77] Zexuan Zhong, Tao Lei, and Danqi Chen. 2022. Training Language Models with
Memory Augmentation. InEMNLP. Association for Computational Linguistics,
5657–5673.
[78] Yun Zhou and W. Bruce Croft. 2007. Query performance prediction in web search
environments. InSIGIR. ACM, 543–550.
[79] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and
Tat-Seng Chua. 2021. Retrieving and Reading: A Comprehensive Survey on
Open-domain Question Answering.CoRRabs/2101.00774 (2021).
[80] Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni,
Xuanhui Wang, and Michael Bendersky. 2023. RankT5: Fine-Tuning T5 for Text
Ranking with Ranking Losses. InSIGIR. ACM, 2308–2313.
[81] Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon.
2025. Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via
Reinforcement Learning.CoRRabs/2503.06034 (2025).
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
12
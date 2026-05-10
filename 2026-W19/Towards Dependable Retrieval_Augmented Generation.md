# Towards Dependable Retrieval-Augmented Generation Using Factual Confidence Prediction

**Authors**: Florian Geissler, Francesco Carella, Laura Fieback, Jakob Spiegelberg

**Published**: 2026-05-04 11:28:19

**PDF URL**: [https://arxiv.org/pdf/2605.05244v1](https://arxiv.org/pdf/2605.05244v1)

## Abstract
Incorporating specific knowledge into large language models via retrieval-augmented generation (RAG) is a widespread technique that fuels many of today's industry AI applications. A fundamental problem is to assess if the context retrieved by some similarity search provides indeed supporting facts, or instead misguides the generator with irrelevant information. It is critical to associate meaningful confidence measures about the factuality of the retrieval process with the generated answers. We present a new, two-staged approach to predict fact faithfulness of the output of retrieval-augmented generations. First, we employ conformal prediction to select only those retrieved chunks who have a high chance to come from the correct source. This approach in itself can improve answer quality by up to 6% in some of the studied datasets, however, the associated statistical guarantees do not hold generally, since the assumption of sample exchangeability depends on the retriever setup. We present diagnostic metrics to assess whether a setup is suitable. Second, we quantify confidence in the consistency of a generated final answer with a given retrieved context, using an attention-based factuality classifier. This approach can detect inconsistent answers with a chance of up to 77%. Our work helps to establish a novel type of certified RAG systems for a broad range of natural language industry applications.

## Full Text


<!-- PDF content starts -->

Towards Dependable Retrieval-Augmented
Generation Using Factual Confidence Prediction
Florian Geissler1⋆, Francesco Carella1, Laura Fieback2, and Jakob Spiegelberg2
1Fraunhofer Institute for Cognitive Systems (IKS), Munich, Germany
2Volkswagen AG, Wolfsburg, Germany
Abstract.Incorporating specific knowledge into large language models
viaretrieval-augmentedgeneration(RAG)isawidespreadtechniquethat
fuels many of today’s industry AI applications. A fundamental problem
is to assess if the context retrieved by some similarity search provides in-
deed supporting facts, or instead misguides the generator with irrelevant
information. It is critical to associate meaningful confidence measures
about the factuality of the retrieval process with the generated answers.
We present a new, two-staged approach to predict fact faithfulness of the
output of retrieval-augmented generations. First, we employ conformal
prediction to select only those retrieved chunks who have a high chance
to come from the correct source. This approach in itself can improve
answer quality by up to6%in some of the studied datasets, however,
the associated statistical guarantees do not hold generally, since the as-
sumption of sample exchangeability depends on the retriever setup. We
present diagnostic metrics to assess whether a setup is suitable. Second,
wequantifyconfidenceintheconsistencyofageneratedfinalanswerwith
a given retrieved context, using an attention-based factuality classifier.
This approach can detect inconsistent answers with a chance of up to
77%. Our work helps to establish a novel type of certified RAG systems
for a broad range of natural language industry applications.
Keywords:Retrieval-augmented generation·Large language models·
Conformal Prediction·Hallucination detection
1 Introduction
Retrieval augmented generation (RAG) has become a de-facto standard to em-
power large language models (LLMs) with domain knowledge, such as a curated
corpus of source documents. The leap to industry applications has long hap-
pened, as frameworks to set up RAG bots in a production environment have
become simple and resource-efficient [1,19]. In modern agentic AI applications,
RAG is commonly used as a subroutine [14], which creates a dependency of the
final outcome on the reliable operation of this mechanism. Certifiable guarantees
of the RAG pipeline must be established, which serve as stepping stones in the
safety assurance of the application.
⋆Email: florian.geissler@iks.fraunhofer.dearXiv:2605.05244v1  [cs.IR]  4 May 2026

2 F. Geissler et al.
User Question EmbeddingContext
retrieval
Documents PreprocessingVector store
EmbeddingSelection and 
rerankingResponse 
synthesis
LLM
Output 
generationBlock 1 Block 2
Answer
Fig.1: Diagram of a generic RAG workflow, augmented by two certifications as pro-
posed in this article. For the retrieval block 1 (blue), we ensure confidence in retrieving
context from the original source. In the generation block 2 (grey), attention monitoring
protects the LLM output against potential hallucinations based on certified context.
Previous works on reliable RAG systems have focused on extending the basic
RAG pipeline - see Fig. 1 - by various additional mechanisms to improve perfor-
mance or robustness heuristically, or by analyzing and quantifying failure modes,
see Sec. 2. In this work, we pursue instead an architecture-agnostic certification
protocol, that does not rely on modifications of individual components but is
applicable to any RAG instantiation. By conformalizing the retrieval process,
we construct certificates for the retrieval quality and analyze the ability of such
certificates to generalize (see Sec. 3.2). Hallucinated generations from the certi-
fied chunks are identified by a detector trained on chunk attentions (Sec. 3.3).
Our approach is not limited by the choice of retriever or open-source model,
and can be combined with any modern, extended RAG workflow. In detail, we
demonstrate
■a certifiable RAG pipeline featuring conformalized retrieval and chunk-based
hallucination detection,
■an analysis of the pipeline’s performance on several data sets. Caveats for
real world system designs are discussed,
■an analysis of the generalisation capability of the proposed method across
various datasets, to validate empirical performance in real-world deploy-
ments.
2 Related work
A major barrier for the adoption of LLMs in safety-critical applications is the
possibility of factually incorrect answers, so-called hallucinations [4,18]. RAG
models ground the available information in external sources [22,15], but provide
no factuality guarantees. Hallucinations originate typically from retrieved con-
text, parametric knowledge, or conflicts between external context and internal
knowledge [32], for which different hallucination detection strategies have been

Towards Dependable Retrieval-Augmented Generation 3
introduced. These methods can be categorized intoclosed-bookdetection meth-
ods, which tackle parametric knowledge hallucinations, andopen-bookmeth-
ods aiming to detect context hallucinations [32]. While closed-book detection
methods rely on internal model representations [5,35] or consistency checks [26],
open-book methods measure the importance of context tokens during genera-
tion by leveraging attention maps [10] or comparing output distributions with
and without external knowledge [21]. The factuality of RAG systems can further
be improved via advanced processing pipelines, involving additional steps like
self-verification [3], knowledge consolidation, grading [33,16,12], or uncertainty
estimation[29,23].Adaptiveretrievalandrisk-awaregenerationarestrengthened
by semantic entropy-guided retrieval [36] and counterfactual prompting to pro-
mote abstention under low confidence [9]. Recent works also explore how answer
reliability can be boosted by agent consensus [8]. Lastly, security-focused eval-
uations highlight the need for pipeline-level defenses against adversarial attacks
[24].
3 Methodology
3.1 Concept
A generic RAG workflow, as depicted in Fig. 1, can be divided into two blocks:
The first block retrieves relevant context from a database, typically a vector
store. We conformalize scores and identify chunks with a statistically guarantee
of relevance. The second block generates an answer for the user question, given
thetrustedcontextretrievedinthefirstblock.Toconfideintheoutputgenerated
by the LLM, it is key to understand whether the output was obtained based
on the provided context and is free of hallucinations. We quantify this using
intermediate attention values and a factuality classifier, as explained in Sec. 3.3.
3.2 Retrieval confidence estimation
Conformalprediction(CP)isadistribution-freeuncertaintyquantificationmethod
[31,6,2] based on the assumption of the exchangeability of data. In a first cal-
ibration step, a set of training data is reserved to calculate a non-conformity
threshold with a user-given significance level. During inference, this threshold is
used to form prediction sets that meet statistical guarantees. We here apply this
principle to retrieve document chunks with guaranteed source fidelity.
Calibration ground truth:Our datasets consist of triplets of questions(q),
reference answers(r)and document indicesd(r), where the respective answer is
located. A retriever selects for a given questionithe topKcandidate chunks,
whereKis the retrieval depth. A retrieval scoresis assigned to each retrieved
chunkc, based on the estimated relevance of that chunk for the given query. This
leads to a set of calibration scoresS cal={s i,j;i∈(1, . . . N cal), j∈(1, . . . K)},
withN calbeing the number of calibration samples. To facilitate thresholding,
we perform a global normalization step to ensures∈[0,1]. Withn 1= min(S cal)

4 F. Geissler et al.
andn 2= max(S cal), we rescaleS cal→(S cal−n 1)/(n 2−n 1). To establish ground
truth, we compare a chunk candidatecwith the reference answerrfor a given
question, using a selected similarity metrich. The source document associated
with the candidate chunk isd(c). We first define a score thresholds thresas the
β-percentileP βof retrieval scores from those chunks that donothave the cor-
rect document affiliation. This serves as a minimal scoring threshold for correct
chunks,
sthres =Pβ({h(c i,j, ri)|d(c i,j)̸=d(r i)}).(1)
Dependingonthenatureofadataset,thesimilaritymeasure(h)canbeadjusted.
For the NQ dataset, where a source document is rather long, we use a high
β= 0.99, while for Ragbench, where source documents are compact, we found
thatcalibrationworksbetterbydocumentID only,i.e.β= 0.Acandidate chunk
is consideredcorrectif it passes the thresholdandhas the correct affiliation,
leading to boolean chunk ground truth labels
Ycal={h(c i,j, ri)≥sthres∧d(c i,j) =d(r i)}.
We takehto be theRougeLscore, which calculates the longest common subse-
quence using n-grams of varying length [25]. An example of the distributions of
chunk scores and affiliations is given in Fig. 2.
0.0 0.2 0.4 0.6 0.8 1.0
Normalized Retrieval Score (s)0.00.20.40.60.81.0RougeL Score (h)sthres1q
Source (d)
correct
incorrect
(a) BM25
0.0 0.2 0.4 0.6 0.8 1.0
Normalized Retrieval Score (s)0.00.20.40.60.81.0RougeL Score (h)sthres1q
Source (d)
correct
incorrect (b) BM25 with cross-encoder reranking
Fig.2:CharacteristicsfortheBM25retrieverwithout(a)andwithasubsequentrerank-
ing using a cross-encoder (b) withK= 10and chunk size512. Retrieval scores(s)of
each chunk are displayed against their rougeL scores(h)and the document source IDs
(d), using5000calibration samples of the NQ dataset. For calibration, first, the green
chunks withh > s thresdefine a set of correct ground truth, see Eq. 2. The retrieval
scores of this subset then determinebqfor a given error rateα(here0.1for illustration),
see Eq. 2. During inference,bqserves for conformal prediction of a trust label with sta-
tistical guarantees (Eq. 3).
Prediction threshold:During inference, estimates of the correctness of
chunk labels are derived with a user-chosen error rateα∈[0,1]. Given a cali-

Towards Dependable Retrieval-Augmented Generation 5
bration set sizen, we use the adjusted confidence rate,δ=⌈(n+ 1)(1−α)/n⌉,
to calculate theδ-percentile,bq, of correct scores [2],
bq=P δ({(1−s)|y=true;s∈S cal, y∈Y cal}).(2)
Connecting to the concept of conformal prediction, we interpret the retriever
as a pre-trained classifier function bfwith a single class (saytrusted), and the
retrieval score of a given chunk,s= bf(c)as a normalized prediction of this
class. Ifs≥1−bq, the prediction set is the trusted class, otherwise it is empty.
Undertheassumptionofdatasampleexchangeability,thisgivesusthestatistical
guarantee, that we include the true chunks in the candidate set, with a chance
of almost exactly1−α[2],
1−α≤ P(y=true|s≥1−bq)≤1−α+1
n+ 1,(3)
wherePdenotes a probability. We thus obtain a retriever with a1−αguarantee
that a chunk scored above the conformality is from the right source, given that
sample exchangeability holds.
Confidence metrics at inference:LetCbe the subset of trusted chunks
retrieved for a given question,
C={c j|sj≥1−bq;j∈(1, . . . K)},(4)
and|C|=k≤Kthe cardinality of this subset. To estimate the quality of the
retriever for a given question, we define the following two metrics
m1=k >0,(5)
m2=k/K.(6)
In words,m 1is a binary metric stating whether or not there isat least one
trustworthy chunk among all retrieved chunks, whereasm 2indicates the average
portion of trusted chunks among all retrieved chunks. As we show in Sec. 4.2,
the rates m1andm2averaged over a calibration set can serve as diagnostic tools
for the suitability of a retriever on a dataset.
3.3 Response confidence estimation
Attention extraction:The attention associated with the LLM output for a
given question is a tensorαl,h
t,p, with indiceslfor attention layers,hfor attention
heads,t∈ {1, . . . T}runs over the newly generated tokens of the output, and
pis the considered token index in the prompt. The overall prompt consists of
the context and the already generated output tokens (if any), meaning possi-
ble segmentsx∈ {context, output}. The context can again be separated into
individual semantic pieces,context={pre, c 1, ..., c k, qu}, wherepredenotes a
system preamble,ca retrieved context chunks with the highest retrieval score

6 F. Geissler et al.
Add & NormFeed -forwardAdd & NormTransformer
Multi -head
attention𝑐2qu pre
output𝑐1𝑐𝑘
context𝑡
𝑡+1
𝑡+2… …Attentions
tokens𝑁×𝐻×𝐿×𝐾
Logistic regression
unit
𝑃(𝑓𝑎𝑐𝑡𝑢𝑎𝑙)Classification
𝑣
Classifier
Fig.3: Prediction of factuality, based on the lookback lens concept [10]. Intermediate
attention values are extracted with a distillation of the relative attention on individ-
ual chunks via lookback ratios. A classifier learns to interpret those ratios to predict
whether a generated answer was sufficiently factual.
first, andquthe final user question (see Fig. 3). We track the attentions averaged
over tokens inx,
Al,h
t(x) =1
NxX
p∈xαl,h
t,p.(7)
Here,N xis the length of the respective piecexin tokens. The quantityAcan
be seen as the average attention the model pays to the text blockxat a given
time steptduring generation.
Lookbackratios:Itistheninformativetodefinechunk-wiselookbackratios
LR∈[0,1]for a chunkc iwithi∈ {1, . . . , K}
LRl,h
t(ci) =K·Al,h
t(ci)
Al,h
t(context) +Al,h
t(output),(8)
LRl,h(ci) =1
TX
tLRl,h
t(ci).(9)
Purposefully, theLRhere do not include the preamble or the questions in the
context, as those parts should not matter for assessment of factuality. This is to
be contrasted with the lookback ratio defined in the original work of [10], which
includes also the static context portions or preamble and question.
Factuality classifier:A small classifier is trained to predict hallucinations
from those lookback ratios, following [10]. Its inputs are time-averaged lookback
ratios (Eq. 9) which are unrolled into a single feature vectorv. If, for a given
input, less thanKtrusted chunks are identified, we padvwith zeros at those
sections, in order to assure a fixed size ofL×H×K. If the full-context lookback
ratios are used, this simplifies tov= [LR1,1(context), . . . , LRL,H(context)]of
sizeL×H.
The classifier estimates a factuality scorep∈[0,1]from the input feature
vectorv, which serves as confidence estimate for the entire answer. It is trained

Towards Dependable Retrieval-Augmented Generation 7
using theanswer consistency(γ ac) metric defined in [10], established with a
separate LLM judge as ground truth. A predicted factuality is considered correct
if its rounded value matches the provided answer consistency,⌊p⌉=γ ac. Across
a dataset, the results are evaluated as the AUROC [28] for training and testing.
In our setup, the classifier architecture consists of a single logistic regression unit
[28], see also Fig. 3.
4 Results
4.1 Experimental setup
We use theNatural Questions(NQ) [20] dataset by Google as well as theRag-
bench(RB) [13] collection which in turn contains12datasets of question-and-
answer pairs from different domains. For each dataset, a random selection of
thetrainsplit is chosen, with a maximum cap of10Ksamples for practical
feasibility. Half of those samples are used for calibration, half for testing. As a
representative of a popular family of open-source model architectures, we select
Llama-3-8B-Instruct [17,34], while GPT3.5 [7] is used as LLM judge. To learn
factuality classification, we split the lookback ratio data in a ratio of3 : 2for
training and validation of the classifier, respectively.
Furthermore, thetext-ada-embedding-002-2embedding model [27] is used with
a chunk size of512. For retrieval, we deploy the BM25 retriever from theLlama-
indexlibrary[11],whileexperimentswithrerankingasdescribedinSec.3.2make
useofthestsb-roberta-basepair-wisecross-encoderfromthesentencetransformer
library [30]. Note that we want to examine certification in RAG, not strive for
state of the art performance. Throughout the experiments in Sec. 4, we use
K= 10andα= 0.1.
For the classification of lookback ratios, we train a simple logistic regression
unit fromsklearn[28]. As hyperparameters, we set a maximum iteration of1000
and class weightbalanced. The lookback ratios are normalized by linear scaling
to a range of[0,1]for each individual data sample.
4.2 Retrieval confidence estimation
Sample exchangeability:The suitability of a retriever for a given setup can
be evaluated by the averagem 1andm 2metrics defined in Sec. 3.2. The retrieval
score of an arbitrary,correct(according to ground truth) chunk is - by construc-
tion ofbqin Eq. 2, assuming full exchangeability of samples - above the trust
threshold with a chance of1−α, and given that there exists at least one correct
source chunk for each question. Importantly, however, the retrieval depthKis
always finite in practice. The assumption of exchangeability of samples is only
given within the pool of the respective topKchunks. If the retriever is strug-
gling to find the correct document source, and instead assigns high retrieval
scores only to incorrect chunks, correct chunks may not appear anymore in the
top-K, meaning that they are under-represented andm 1will fall short of the

8 F. Geissler et al.
0.0 0.2 0.4 0.6 0.8 1.0
0.00.20.40.60.81.0m1
BM25
Cross-encoder
0.0 0.2 0.4 0.6 0.8 1.0
0.00.20.40.60.81.0m2BM25
Cross-encoder
Fig.4: Them 1andm 2metrics averaged over a calibration set under variation of the
error rateαfor the BM25 retriever, with and without subsequent reranking with the
cross-encoder. We used5000samples of NQ andK= 10.
expectation1−α. On the other hand, if the retriever assigns high retrieval scores
only to correct source chunks and finds so many of them that incorrect chunks
are cut off from the topK, almost all retrieved content will be accepted and
m1is unexpectedly high. As a consequence, the bounds in Eq. 3 do not hold
strictly,butapproximately,ifthetop-Ksamplepoolissufficientlyrepresentative
of correct chunks.
Tab. 1a shows the averagem 1andm 2values across the studied datasets.
We see that most results are close to the expected bounds of conformal predic-
tion (see Eq. 3). Notable exceptions areRB-tatqaandRB-finqa. Both datasets
contain financial questions and the BM25 retriever fails to identify the correct
chunk in the topK: Almost always either zero or a maximum amount ofK
chunks ( m2≈m1) is retrieved, indicating its struggle to assign a meaningful
score. Since calibration scores are evaluated over true chunks in the calibration
data, absence of true chunks in the topKbreaks exchangeability of calibration
data and test data. This is a crucial pitfall to consider when designing practical
application using CP.
To further elaborate on this important observation, we study the variation of
m1,m2with the significance levelαfor different ranking techniques. An example
usingNQdataisshowninFig.4.WeseethattheBM25retrievercharacteristicis
close to the expectation of exchangeability. With reranking, we see in Fig. 4 that
theacceptanceofcorrectchunkincreasessignificantly.Thebalanceofcorrectand
incorrect chunks gets skewed towards more correct chunks with higher scores.
Unfortunately, however, this result is overly optimistic at finite retrieval depth,
as incorrect source chunks get cut off alltogether. We choose the base retriever
without cross-encoder reranking for the further experiments of Sec. 4.
Impact of chunk filtering:We next evaluate, whether the filtering for
trusted chunks directly affects the quality of LLM answers during RAG. Results
for LLama-3 are shown in Tab. 1b and Fig. 5, where the answer consistency is
measured asγ ac. We observe variations between the data sets. ForRB-expertqa,

Towards Dependable Retrieval-Augmented Generation 9
Dataset m1(%)m2(%)
NQ 91.1±0.4 74.9±0.5
RB-covidqa 90.4±1.2 66.4±1.5
RB-cuad 94.8±0.8 94.8±0.8
RB-delucionqa 89.3±1.6 77.5±1.9
RB-emanual 82.7±1.6 77.8±1.7
RB-expertqa 93.7±0.9 66.9±1.3
RB-finqa 75.8±0.6 72.6±0.6
RB-hagrid 92.4±0.7 40.7±0.8
RB-hotpotqa 97.9±0.5 27.0±0.9
RB-msmarco 94.9±0.7 37.9±1.0
RB-pubmedqa 91.3±0.4 59.2±0.5
RB-tatqa 61.4±0.7 61.4±0.7
RB-techqa 93.8±1.0 87.5±1.2
(a) Averagedm 1andm 2characteristic of the
retriever on the studied datasets, with standard
error. Parameters:α= 0.1,K= 10. For RB we
usedβ= 0, for NQβ= 0.99, as explained in
the text.Datasetγac
filter no filter
NQ 65.1±0.7 65.3±0.7
RB-covidqa 73.9±1.8 72.3±1.8
RB-cuad 57.1±1.9 51.6±1.9
RB-delucionqa 77.7±2.2 77.6±2.2
RB-emanual 69.2±2.0 78.4±1.8
RB-expertqa 59.5±1.7 53.0±1.8
RB-finqa 49.5±0.7 59.7±0.7
RB-hagrid 74.1±1.2 70.0±1.2
RB-hotpotqa 78.8±1.3 74.5±1.4
RB-msmarco 74.1±1.4 68.8±1.5
RB-pubmedqa 78.6±0.6 77.9±0.6
RB-tatqa 49.7±0.7 65.3±0.7
RB-techqa 50.9±1.2 49.3±2.1
(b) Averaged answer consistency γacfor Llama3
with and without chunk filtering based on con-
fidence estimation. With filtering, untrusted
chunks are removed to for the subsequent LLM
call.
Table 1: Caption
for example, a boost in answer consistency of more than6%emerges. On the
other hand,RB-finqashows the opposite effect, and context filtering actually
degrades the answer consistency by roughly10%. More generally, the change in
answer consistency performance is correlated with them 1retriever performance,
as illustrated in Fig. 5. We here visualize the difference from the CP expectation,
m1−(1−α), against the delta in answer quality. A low retriever performance
in this metric leads to a negative effect of chunk filtering and vice versa. This
is due to the fact that correct chunk sources are no longer strictly associated
with high retrieval scores. In our experiments, this lacking retriever performance
is found forRB-finqa,RB-tatqa, andRB-emanual. For strong retrievers, chunk
filtering benefits the answer quality of the LLMs positively, i.e., m1≥1−α.
We conclude that poor retrieval not only breaks CP guarantees, it also de-
grades subsequent generation performance. Them 1andm 2metrics devised in
this article therefore serve as a metric to characterize the suitability of a RAG
setup for a given dataset and retrieval depth. If such a suitability is established,
test time confidence guarantees from CP can be constructed.
4.3 Response confidence estimation
In this section, we derive a confidence estimation for the factuality of a gener-
ated LLM answer, given a previously retrieved context (Sec. 3.3 or block 2 of the
RAG pipeline). To that end, we train a classifier of answer consistency. Since the
individual datasets vary significantly in magnitude, we chose to group them in

10 F. Geissler et al.
30
 20
 10
 0 10
m1 (1 )
15
10
5
05ac(filtered)ac(unfiltered)
RB_hotpotqa
RB_msmarco
RB_cuad
RB_techqa
RB_expertqa
RB_hagrid
RB_pubmedqa
nq
RB_covidqa
RB_delucionqa
RB_emanual
RB_finqa
RB_tatqa
Fig.5: Retriever performance on a dataset, as measured by deviation of the average
m1from the expected1−α, versus the impact of chunk filtering on the consistency
of the subsequent LLM output, as given in Tabs. 1a-1b. We see that chunk filtering is
beneficial if the retriever performance is high, and vice versa.
clusters of comparable size (see Tab. 2) with group 0: (RB-tatqa, RB-expertqa,
RB-cuad),group1:(RB-finqa,RB-msmarco,RB-techqa,RB-delucionqa),group
2: (RB-pubmedqa, RB-hotpotqa, RB-covidqa), and group 3: (NQ, RB-hagrid,
RB-emanual). This also ensures that hallucination detection does not learn pat-
terns that are too specific to one given dataset.
As classifier input, we compare chunk-wise (CW) lookback ratios and those aver-
aged across the full context (FC). The latter importantly includes system pream-
ble and question, see Fig. 3 and discussion in Sec. 3.3. From Tab. 2, we first ob-
serve that both strategies result in classifiers with similar training performance
for each group. Fig. 6 shows that AUROCs are highest for validation within the
same dataset group - up to77%and82%, respectively - while cross-validation
with other groups exhibit performance drops of up to16%.
Interestingly, we observe that the FC LR classifiers outperform the CW LR
ones by up to12%, especially in cross-validation. As an intermediate experiment,
we average out the chunk dimension of the lookback ratios in Eq. 8, capturing
all context attention except from preamble and question in the feature vector.
The resulting classifiers show some individual improvements in off-diagonal val-
idation, yet do not lead to any test AUROC>77%, hence still fall short of the
classifiers trained on FC LR, where the only difference is the inclusion of prompt
and question attention.
Our study in this section hence reveals the following key finding: The supe-
rior performance of the full-context hallucination detection over the one based
on chunk context only (Fig. 6) is attributed to relevant attention on the sys-
tem prompt and question. This implies that hallucination patterns are partially

Towards Dependable Retrieval-Augmented Generation 11
Group_0
Group_1
Group_2
Group_3
T argetGroup_0
Group_1
Group_2
Group_3Source77.2 72.9 62.9 64.1
73.7 76.3 61.2 65.7
70.6 71.5 70.0 63.7
73.6 71.7 62.7 70.7
556065707580
T est AUROC
(a) CW LR
Group_0
Group_1
Group_2
Group_3
T argetGroup_0
Group_1
Group_2
Group_3Source81.8 76.9 67.0 75.5
78.2 79.1 70.5 76.6
66.8 71.3 76.7 74.0
75.9 74.7 71.9 78.1
556065707580
T est AUROC (b) FC LR
Fig.6: Test AUROCs for classifiers trained on the training split of thesourcedataset,
and tested on the test portion of thetargetdataset. Off-diagonal elements represent
cross-validationscenarios.Wecompareclassifierstrainedonchunk-wise(CW)LR(left)
and lookback ratios using the average full context (FC) (right) as in [10].
Group ID SizeTrain AUROC
CW LRFC LR
06447 83.7 85.8
16810 86.5 85.0
26463 87.5 84.1
36840 86.3 83.6
Table 2: Training AUROC of dataset groups, using clusters of similar size for Chunk-
wise (CW) and Full-context (FC) LR.
learned from generic information that is unrelated to factual correctness. For
example, the classifier might learn that questions from the domain of finance are
more likely to elicit incorrect answers. Full-context hallucination detectors can
thus be misleading, instead we propose to use CW LR in reliable RAG systems
despite the lower test AUROC for confidence estimation, as non-retrieval related
sources of attention are excluded by design. Therefore, our predictor provides a
better factual grounding. Our result may inspire further systematic studies on
the impact of domain-specific system prompts and questions on model attentions
in the future.
5 Conclusion
We present a comprehensive workflow to quantify factual confidence in RAG
systems. Our approach provides separate metrics for the stages of context re-
trieval and LLM-based answer generation, respectively, and therefore allows to
pinpoint potential weaknesses in practical RAG pipelines. We demonstrate that

12 F. Geissler et al.
conformal prediction is an effective way to provide confidence guarantees for the
correctness of chunks, but also showcase that the assumption of exchangeability
does not apply a priori to retrieval scores at finite retrieval depth, such that
combinations of retrievers and datasets have to be chosen with care. The cardi-
nalities of the trusted chunk sets serve as an indicator metric for the suitability of
such combinations. To predict factuality of LLM answers from a given context,
we train a classifier on the chunk-wise lookback ratio. This custom quantity rep-
resents the attention the model spends on the respective context chunks during
answer generation and leads to high hallucination detection rates. In contrast to
previously proposed work on lookback ratios, our approach excludes attentions
on prompt segments unrelated to retrieval, which can possibly have misleading
effects on the classifier. Our findings guide the development towards dependable
RAG applications in safety-critical systems.
Acknowledgment
Parts of this work have been funded by the Free State of Bavaria in the DS-
genAI project (Grant Nr.: RMF-SG20-3410-2-18-4). The results, opinions and
conclusions expressed in this publication are not necessarily those of Volkswagen
Aktiengesellschaft.
References
1. Amazon Web Services: Develop advanced generative ai
chat-based assistants by using rag and react prompt-
ing.https://docs.aws.amazon.com/prescriptive-guidance/
latest/patterns/develop-advanced-generative-ai-chat-based-
assistants-by-using-rag-and-react-prompting.html(2023), accessed:
2023-10-01
2. Angelopoulos, A.N., Bates, S.: A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification (December 2022).https:
//doi.org/10.48550/arXiv.2107.07511,http://arxiv.org/abs/2107.07511,
arXiv:2107.07511 [cs]
3. Asai, A., Wu, Z., Wang, Y., et al.: Self-rag: Learning to retrieve, generate, and cri-
tique through self-reflection. In: The Twelfth International Conference on Learning
Representations (2023)
4. Azamfirei, R., Kudchadkar, S.R., Fackler, J.: Large language models and the
perils of their hallucinations. Critical Care27(1), 120 (March 2023).https://
doi.org/10.1186/s13054-023-04393-x,https://ccforum.biomedcentral.com/
articles/10.1186/s13054-023-04393-x
5. Azaria, A., Mitchell, T.: The Internal State of an LLM Knows When It’s Lying. In:
Findings of the Association for Computational Linguistics: EMNLP 2023. pp. 967–
976. Association for Computational Linguistics, Singapore (2023).https://doi.
org/10.18653/v1/2023.findings-emnlp.68,https://aclanthology.org/2023.
findings-emnlp.68

Towards Dependable Retrieval-Augmented Generation 13
6. Bates, S., Angelopoulos, A., Lei, L., et al.: Distribution-free, Risk-controlling Pre-
diction Sets. Journal of the ACM68(6), 1–34 (December 2021).https://doi.org/
10.1145/3478535,https://dl.acm.org/doi/10.1145/3478535
7. Brown, T.B., Mann, B., Ryder, N., et al.: Language Models are Few-Shot Learners
(July 2020).https://doi.org/10.48550/arXiv.2005.14165,http://arxiv.org/
abs/2005.14165, arXiv:2005.14165 [cs]
8. Chang, C.Y., Jiang, Z., Rakesh, V., et al.: MAIN-RAG: Multi-agent filtering
retrieval-augmented generation. In: Proceedings of the 63rd Annual Meeting of
the ACL (2025)
9. Chen, L., Zhang, R., Guo, J., et al.: Controlling risk of retrieval-augmented gener-
ation: A counterfactual prompting framework. In: Findings of the Association for
Computational Linguistics: EMNLP 2024. Association for Computational Linguis-
tics (Nov 2024).https://doi.org/10.18653/v1/2024.findings-emnlp.133, code
& RC-RAG benchmark available
10. Chuang,Y.S.,Qiu,L.,Hsieh,C.Y.,etal.:LookbackLens:DetectingandMitigating
Contextual Hallucinations in Large Language Models Using Only Attention Maps
(October 2024).https://doi.org/10.48550/arXiv.2407.07071,http://arxiv.
org/abs/2407.07071, arXiv:2407.07071 [cs]
11. Contributors, L.: Llamaindex: A data framework for llms (2023),https://github.
com/jerryjliu/llama_index, accessed: 2023-10-03
12. Fang, Y., Thomas, S., Zhu, X.: HGOT: Hierarchical graph of thoughts for retrieval-
augmented in-context learning in factuality evaluation. In: Proceedings of the 4th
Workshop on Trustworthy NLP (TrustNLP@ACL). pp. 118–144 (2024)
13. Friel, R., Belyi, M., Sanyal, A.: RAGBench: Explainable Benchmark
for Retrieval-Augmented Generation Systems (January 2025).https:
//doi.org/10.48550/arXiv.2407.11005,http://arxiv.org/abs/2407.11005,
http://arxiv.org/abs/2407.11005
14. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M.,
Wang, H.: Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 (2024),https://arxiv.org/abs/2312.10997v5
15. Gao, Y., Xiong, Y., Gao, X., et al.: Retrieval-Augmented Generation for Large
Language Models: A Survey (March 2024).https://doi.org/10.48550/arXiv.
2312.10997,http://arxiv.org/abs/2312.10997, arXiv:2312.10997
16. Ge, Z., Wu, Y., Chin, D.W.K., et al.: Resolving conflicting evidence in automated
fact-checking: A study on retrieval-augmented llms. Proc. of IJCAI-25 (2025)
17. Grattafiori, A., Dubey, A., Jauhri, A., et al.: The Llama 3 Herd of Models (Novem-
ber 2024).https://doi.org/10.48550/arXiv.2407.21783,http://arxiv.org/
abs/2407.21783, arXiv:2407.21783
18. Huang, L., Yu, W., Ma, W., et al.: A Survey on Hallucination in Large Language
Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Transac-
tions on Information Systems43(2), 1–55 (March 2025).https://doi.org/10.
1145/3703155,https://dl.acm.org/doi/10.1145/3703155
19. IBM Developer: Awb scenarios: Options for rag da.https://developer.ibm.com/
articles/awb-scenarios-options-for-rag-da/(2023), accessed: 2023-10-01
20. Kwiatkowski, T., Palomaki, J., Redfield, O., et al.: Natural Questions: A Bench-
mark for Question Answering Research
21. Lee, D., Yu, H.: REFIND at SemEval-2025 Task 3: Retrieval-Augmented Fac-
tuality Hallucination Detection in Large Language Models (April 2025).https:
//doi.org/10.48550/arXiv.2502.13622,http://arxiv.org/abs/2502.13622,
arXiv:2502.13622

14 F. Geissler et al.
22. Lewis, P., Perez, E., Piktus, A., et al.: Retrieval-Augmented Genera-
tion for Knowledge-Intensive NLP Tasks. In: Advances in Neural In-
formation Processing Systems. vol. 33, pp. 9459–9474. Curran Asso-
ciates, Inc. (2020),https://proceedings.neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.html
23. Li, Z., Xiong, J., Ye, F., et al.: Uncertaintyrag: Span-level uncertainty enhanced
long-context modeling for retrieval-augmented generation. In: arXiv preprint
arXiv:2410.02719 (Oct 2024)
24. Liang, X., Niu, S., Li, Z., et al.: SafeRAG: Benchmarking security in retrieval-
augmented generation of large language models. arXiv:2501.18636 (2025)
25. Lin, C.Y.: ROUGE: A Package for Automatic Evaluation of Summaries. In: Text
Summarization Branches Out. pp. 74–81. Association for Computational Linguis-
tics, Barcelona, Spain (July 2004),https://aclanthology.org/W04-1013/
26. Manakul, P., Liusie, A., Gales, M.: SelfCheckGPT: Zero-Resource Black-Box Hal-
lucination Detection for Generative Large Language Models. In: Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Process-
ing. pp. 9004–9017. Association for Computational Linguistics, Singapore (2023).
https://doi.org/10.18653/v1/2023.emnlp-main.557,https://aclanthology.
org/2023.emnlp-main.557
27. OpenAI: OpenAI Embeddings,https://platform.openai.com/docs/guides/
embeddings
28. Pedregosa, F., Varoquaux, G., Gramfort, A., et al.: Scikit-learn: Machine Learning
in Python. Journal of Machine Learning Research12(85), 2825–2830 (2011),http:
//jmlr.org/papers/v12/pedregosa11a.html
29. Perez-Beltrachini, L., Lapata, M.: Uncertainty quantification in retrieval aug-
mented question answering (Feb 2025), code available athttps://github.com/
lauhaide/ragu
30. Reimers, N., Gurevych, I.: Sentence transformers: Multilingual sentence and image
embeddings (2019),https://www.sbert.net, accessed: 2023-10-03
31. Shafer, G., Vovk, V.: A Tutorial on Conformal Prediction. Journal of Ma-
chine Learning Research9(12), 371–421 (2008),http://jmlr.org/papers/v9/
shafer08a.html
32. Simhi, A., Herzig, J., Szpektor, I., et al.: Constructing Benchmarks and Interven-
tions for Combating Hallucinations in LLMs (July 2024).https://doi.org/10.
48550/arXiv.2404.09971,http://arxiv.org/abs/2404.09971, arXiv:2404.09971
33. Wang,F.,Wan,X.,Sun,R.,etal.:ASTUTERAG:OvercomingImperfectRetrieval
AugmentationandKnowledgeConflictsforLargeLanguageModels(2025),https:
//arxiv.org/abs/2410.07176
34. Wolf, T., Debut, L., Sanh, V., et al.: HuggingFace’s Transformers: State-of-the-
art Natural Language Processing (July 2020).https://doi.org/10.48550/arXiv.
1910.03771,http://arxiv.org/abs/1910.03771, arXiv:1910.03771 [cs]
35. Xiao, Y., Wang, W.Y.: On Hallucination and Predictive Uncertainty in Condi-
tional Language Generation. In: Proceedings of the 16th Conference of the Euro-
pean Chapter of the Association for Computational Linguistics: Main Volume. pp.
2734–2744. Association for Computational Linguistics, Online (2021).https://
doi.org/10.18653/v1/2021.eacl-main.236,https://aclanthology.org/2021.
eacl-main.236
36. Zubkova, H., Park, J.H., Lee, S.W.: Sugar: Leveraging contextual confidence for
smarter retrieval (Jan 2025), semantic entropy guides adaptive retrieval
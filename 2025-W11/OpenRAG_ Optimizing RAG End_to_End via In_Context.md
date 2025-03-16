# OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning

**Authors**: Jiawei Zhou, Lei Chen

**Published**: 2025-03-11 13:04:05

**PDF URL**: [http://arxiv.org/pdf/2503.08398v1](http://arxiv.org/pdf/2503.08398v1)

## Abstract
In this paper, we analyze and empirically show that the learned relevance for
conventional information retrieval (IR) scenarios may be inconsistent in
retrieval-augmented generation (RAG) scenarios. To bridge this gap, we
introduce OpenRAG, a RAG framework that is optimized end-to-end by tuning the
retriever to capture in-context relevance, enabling adaptation to the diverse
and evolving needs. Extensive experiments across a wide range of tasks
demonstrate that OpenRAG, by tuning a retriever end-to-end, leads to a
consistent improvement of 4.0% over the original retriever, consistently
outperforming existing state-of-the-art retrievers by 2.1%. Additionally, our
results indicate that for some tasks, an end-to-end tuned 0.2B retriever can
achieve improvements that surpass those of RAG-oriented or instruction-tuned 8B
large language models (LLMs), highlighting the cost-effectiveness of our
approach in enhancing RAG systems.

## Full Text


<!-- PDF content starts -->

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
Jiawei Zhou1Lei Chen1
1Hong Kong University of Science and Technology
Abstract
In this paper, we analyze and empirically show
that the learned relevance for conventional in-
formation retrieval (IR) scenarios may be incon-
sistent in retrieval-augmented generation (RAG)
scenarios. To bridge this gap, we introduce
Open-Rag ,aRAGframeworkthatis OPtimized
ENd-to-end by tuning the retriever to capture in-
context relevance, enabling adaptation to the di-
verseandevolvingneeds. Extensiveexperiments
across a wide range of tasks demonstrate that
Open-Rag ,bytuningaretrieverend-to-end,leads
toaconsistentimprovementof4.0%overtheorig-
inal retriever, consistently outperforming exist-
ing state-of-the-artretrievers by 2.1%. Addition-
ally, our results indicate that for some tasks, an
end-to-end tuned 0.2B retriever can achieve im-
provements that surpass those of RAG-oriented
or instruction-tuned 8B large language models
(LLMs), highlighting the cost-effectiveness of
our approach in enhancing RAG systems.
1. Introduction
As large language models (LLMs) (Zhao et al., 2023; Mi-
naeeetal.,2024)scale,theyfaceadatabottleneckwherethe
high-quality internet data unable to meet growing training
demands. Meanwhile,thevolumeofdownstreamdataisex-
pandingrapidlybutoftenremainsunusableforpre-training
due to their real-time availability (Wang et al., 2024b; Liu
etal.,2023),privacyconcerns(Aroraetal.,2023),licensing
restrictions(Minetal.,2024),andethicalconcern(Serouis
& Sèdes, 2024; Ayyamperumal & Ge, 2024).
Retrieval-augmentedgeneration(RAG)(Lewisetal.,2020;
Guu et al., 2020; Gao et al., 2023) emerges as a promis-
ing solution to this challenge. Rather than relying solely
on well-curated internet data, RAG leverages information
retrieval (IR) to fetch relevant data from external sources
and incorporates it as context to enhance generation qual-
ity. This is valuable as RAG enables the use of rapidly
expanding yet often inaccessible downstream data, which
aremorescalableandup-to-datethantheheavilyprocessed
and regulated internet data used in pre-training.
Figure 1. Comparisonofquery-documentrelevanceinIRscenario
and RAG scenario.
Despite their success, existing RAG frameworks typically
relyonoff-the-shelfretrieverstrainedonQAdatasets,which
can lead to inconsistencies between the learned retrieval
relevance and the needs of downstream tasks. This dis-
crepancy highlights key relevance gaps between IR and
RAG scenarios. We explore these gaps in detail below,
drawing on insights from prior research. First, there is the
broadeningoftasks : traditionalIRdatasets(Kwiatkowski
et al., 2019; Bajaj et al., 2016) are designed mainly for
open-domain question-answering (OpenQA), while RAG
framework are applied to a wider range of tasks, such as
recommendation (Manzoor & Jannach, 2022), dialog sys-
tems(Liuetal.,2024),androle-playing(Wangetal.,2023),
where task requirements can be flexibly written as instruc-
tions. We refer to relevance in these two cases as QA
relevance andin-context relevance , respectively, as shown
in Figure 1. Second, the role of retrieved documents has
shifted: inIR,retrieveddocumentsarethefinaloutputpro-
vided to users, whereas in RAG, they are fed into the LLM
to generate a response. Recent studies (Cuconasu et al.,
2024a;b; Wu et al., 2024) have shown that including more
answer-containing documents, which align with QA rele-
vance in IR scenarios, can harm RAG performance, while
documentswithoutdirectanswersmayactuallyhelp. These
findings challenge traditional IR assumptions in the RAG
setting. Finally, the complexity of queries has increased:
unlike traditional IR, where queries are typically simple
questions, RAG queries tend to be more diverse and noisy,
reflecting varying levels of task complexity. Several stud-
1arXiv:2503.08398v1  [cs.CL]  11 Mar 2025

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
ieshighlightthechallengesofcomplexqueriesandsuggest
that refining queries (Chan et al., 2024) or generating task-
specific queries (Wu & Cao, 2024; Koo et al., 2024) based
ondocumentscansignificantlyenhanceRAGperformance.
Toaddressthisgap,weintroduce Open-Rag ,aRAGframe-
work that is OPtimizedENd-to-end by tuning the retriever
to capture in-context relevance. Unlike existing retrievers,
which are constrained to training on specific corpora and
tasks with human annotations provided, our framework is
OPENto training on any task, with any corpus and any
LLM. During training, Open-Rag retrieves documents on-
the-flyandidentifiesthemaspositivesornegativesforcon-
trastive learning.To reduce training costs, we use approxi-
mation techniques to bypass the autoregressive generation
process and employ semi-parametric retrieval to avoid the
needforre-indexing. OurtrainingrequiresonlyfourGPUs
and can be completed within a day. Extensive experiments
demonstrate that our method leads to significant improve-
ments, consistently outperforming state-of-the-art (SOTA)
retrievers. For certain tasks, our improvements surpass
thoseachievedbytuningan8BLLM,showcasingthatend-
to-end retrieval learning is a cost-effective approach for
enhancing RAG systems.
Our contribution can be summarized as follows:
•We investigate the relevance gap between IR and RAG
scenarios, providing empirical evidence of when and how
this gap negatively impacts RAG performance.
•Through our experiments, we identify potential biases in
priorresearchthatmayimpedeprogressinthisfield. These
findings provide critical insights to guide future research
directions.
•We introduce Open-Rag , an end-to-end optimized RAG
frameworkthatlearnsin-contextretrievalforvariousdown-
stream tasks without requiring query-document human an-
notations, facilitating broader real-world deployment and
applications.
•Extensive experiments show that Open-Rag achieves su-
perior performance across diverse tasks compared to RAG
systemsusingSOTAretrieversorfine-tunedLLMs,under-
scoring its effectiveness as a reliable and versatile solution
for improving RAG systems.
2. Preliminary
2.1. Transferring from IR to RAG Scenarios
In Table 1, we examine the performance of off-the-shelf
retrieversacrossdifferentdatasetsinIRandRAGscenarios.
Details about the datasets and retrievers can be found in
AppendixAandB,whiletheevaluationmetricisdescribed
in Section 4.1. Key findings are summarized below.Table 1.AccuracyinIRandRAGscenariosusingLlama3-8bwith
top-1 retrieved document in-context; Bold: best performance; ∆:
improvementordeclinecomparedto SiDR MS;§: hasaccessedthe
training split of the dataset.
Dataset ( →) NQ TriviaQA PubHealth ARC-C
Retriever ( ↓) IR ∆ RAG ∆ IR ∆RAG ∆RAG ∆RAG ∆
Unsupervised Pre-training
Contriever 23.6 -15.5 30.9 -3.5 37.2 -18.9 56.6 -5.4 61.8 -1.758.6+1.7
E5-unsup 30.8 -8.3 33.4 -1.0 39.5 -16.6 54.3 -7.7 62.9 -0.6 58.3 +1.4
Supervised on MSMARCO
DPRMS38.9-0.2 34.9 +0.5 43.7 -12.4 55.2 -6.8 64.5 +1.0 56.3 -0.6
SiDRMS39.1– 34.4 – 56.1 – 62.0 – 63.5 – 56.9 –
Supervised on NQ
DPRNQ ‡43.5+4.4‡38.5+4.1 39.4 -16.7 55.9 -6.1 62.9 -0.6 56.6 -0.3
SiDRNQ ‡49.5+10.4‡42.7+8.3 47.4 -8.7 59.8 -2.2 63.5 – 57.1 +0.2
Supervised on TQA
DPRTQA32.1-7.0 32.9 -1.5‡55.4-0.7‡61.1-0.9 63.1 -0.4 56.7 -0.2
SiDRTQA30.6-8.5 32.9 -1.5‡56.9+0.8‡63.6+1.661.1-2.458.6+1.7
Pre-training + Supervised on Multiple Datasets
Contriever MS41.5+2.4 36.5 +2.1 53.5 -2.6 60.7 -1.3 63.1 -0.4 58.1 +1.2
E5 ‡58.0+18.9‡43.2+8.8 58.7 +2.663.2+1.264.7+1.258.0+1.1
Potential Improvement of IR vs. Improvement of LLMs
Best-of-8 – – 77.6 – – – 80.3 – 92.1 – 71.5 –
E5+ 8B-Instruct – – 54.4 – – – 66.7 – 72.4 – 74.1 –
E5+ 70B – – 51.4 – – – 68.0 – 63.2 – 81.9 –
Finding1: Trainingretrieversin-domainiseffectivefor
both IR and RAG. As shown, with comparable training
complexity, SiDR NQexcels on the NQ dataset relative to
other SiDRandDPRmodels. Additionally, SiDR TQAout-
performsthestate-of-the-artretriever E5inRAGscenarios
on the TriviaQA dataset.
Finding 2: Superiority of retrievers in IR scenarios can
transfer to RAG scenarios cross-domain but not cross-
task.For QA tasks, retrievers with higher accuracy in IR
scenariostendstoperformbetterinRAGscenarios,asevi-
dencedbyNQandTQAdatasets. However,thistrenddoes
notextendtonon-QAtasks. Forinstance,onthePubHealth
dataset,therelativelyweakerretrieverDPR MSoutperforms
others,whileontheARCdataset,theunsupervisedretriever
Contriever surpasses all advanced retrievers.
Finding3: RetrievalhasgreatpotentialtoimproveRAG
asmuchasusinginstruction-tunedorlargerLLMs. We
usetheBest-of-8 metrictomeasuretheproportionofqueries
thatcanbeaddressedinRAGscenariosbyanyoftheabove
eight retrievers. Best-of-8 substantially outperforms SOTA
retrieverE5acrossthesedatasets. Notably,formosttasks,it
evensurpassesthecombinationofE5withinstruction-tuned
LLMs(Llama3-8B-Instruct)orlargerLLMs(Llama3-70B).
For example, on NQ dataset, 77% of test queries have
a searchable document in the datastore that can serve as
context to generate a correct answer. However, combin-
ing E5 with instruction-tuned LLMs addresses 54% while
larger LLMs address 51%. These results highlight the
largely untapped potential of million-scale datastores and
in-contextexamplesforenhancingLLMinference,wherea
well-optimized retrieval model could unlock this potential.
Motivated by these observations, our work aims to learns
task-specificin-contextrelevanceforRAGinanend-to-end
manner, moving beyond the traditional QA relevance.
2

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
2.2. Problem Setup
A RAG framework typically consists of:
•A retriever Rθparameterized by θ
•A large language model Gϕparameterized by ϕ
•A task Tpresented as an instruction prompt
•A datastore Dwith a vast number of documents d
•A user query q
•The answers ato the query
•An evaluation metric Evaldetermining whether the
output generation addresses the query
The downstream RAG pipeline generally follows:
1. Retrieve the top- krelevant documents from the D
based on q, with a relevance function fθ:
{ˆd}k=Rθ(q,D, k)≜argmaxk
d∈Dfθ(q, d)
2. Formulate the task-specific prompt xusing the query
qand the retrieved documents {ˆd}k:
x=PromptT(q,{ˆd}k)
3. Generate response ˆyfrom input xvia LLM:
ˆy=Gϕ(x)
4. Evaluate if the generation ˆyreflects the answer a:
Eval (ˆy) =(
1ifˆyreflects a,
0otherwise.
TheGoalof Open-Rag : InaRAGsystem,givenanLLM,
adatastore,andatask, Open-Rag aimstotraintheretriever
component to maximize the likelihood of generating a re-
sponse ˆythatoptimallysatisfiesthedownstreamevaluation
metric. This can be formulated as:
ˆθ= argmax
θX
∀qEval (ˆy| Rθ,Gϕ,T,D, q)
2.3. Challenges and Prior Work
Major Challenges. There are two major challenges in
training a RAG framework end-to-end via tuning retriever.
(i) The primary challenge involves the extreme computa-
tional costs associated with deploying such a pipeline in
training. These costs mainly arise from two sources: first,
the LLMs generate sequences autoregressively, which is
inherently resource-intensive; secondly, as θupdates, the
retrieval index need to be rebuilt accordingly, adding fur-
ther computational demands. (ii) The second challenge is
ensuring stable and effective back-propagation of supervi-
sion signals from the final outcome of the RAG pipeline to
the retriever.Prior Practices. Prior research (Guu et al., 2020; Xu
etal.,2023;Shietal.,2023)hasexploredthejointtraining
ofretrieverswithLLMsforRAG.Despiteextensiveefforts,
they often default to learning a universal relevance, where
the retrieved document aids in generating the continuation
of a natural language input, while neglecting the specific
downstream components T,D,Gϕ(x)andEval. These
general approaches lead to a significant discrepancy as the
components used during training do not align with those
employed during inference. As a result, these methods of-
ten fall short in meeting the specific, nuanced relevance
needs of various downstream tasks.
3. Methodology
In this section, we introduce Open-Rag , anOPtimized
ENd-to-end RAGframework designed to fine-tune a re-
triever to capture in-context, open-ended relevance, opti-
mizing it for the downstream RAG pipeline.
To summarize, Open-Rag training comprises two stages:
offline RAG and online RAG. The primary goal is to on-
the-fly identify positive and negative documents for the
contrastive learning of the retriever. An illustration of our
framework is depicted in Figure 2.
3.1. Preliminary Concepts
Continuation yand Generation ˆy.For knowledge-
intensive generative tasks, information is aggregated and
promptedasinput xtoaLLMforgeneration. Theexpected
output could be an answer string ain question-answering
tasks or might be a choice label cin reasoning and fact-
checking tasks. Here, we refer to the expected output as
the ground truth continuation, denoted as y, and the actual
output generated by the LLM as ˆy. In a well-performing
RAGframework,itisgenerallyexpectedthat ˆy=yorthat
ˆycontain or reflect y.
RAG Label. Given a query q, the RAG label Lq
dfor a
document dis a binary value that indicates whether the
RAG outcome, when dis used in the context, meets the
evaluation metric. The computation involves the following
steps:
x=PromptT(q, d); ˆy=Gϕ(x)
Lq
d≜Eval (ˆy)
Thisassessmentistypicallybasedonwhetherthegenerated
response contains the answers. The computation of RAG
labels aligns with downstream inference, which involves
autoregressive generation. For a clearer understanding, we
provide examples in Appendix G.
RAG Score. Given a query q, the RAG score Sq
dof ad
is the joint probability that LLM generates continuation y
3

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
IR Contrastive LearningApproximate RAG LabelCompute RAG Score
0.42
Autoregressively generate for evaluationRAG LabelRAG ScoreOne forward pass to compute probability0.650.71To approximate Label via Score Go through RAG pipeline to obtain Score and LabelThresholdOﬄine RAG RetrievalPrompt Formulation XForward Pass
Query
Docs
Datastore
Instruction
OR
Use oﬄine threshold to identify online pos and negQRAG ScoreRAG Label
         No need to rebuild index.Semi-parametric retrieval allow using ﬁxed tokenization as index and reranking on-the-ﬂy.Prompt
         Consistent prompt as downstream.
Semi-parametricRetrieval·········Q1D1Q2D1Q2·D2Q2·D1+D2+-D2-Q1·D1+Q1·D2+++Dn+···Q1·Dn+Dn-Qn···Q2·Dn+Q1·D1-Q1·D2-
Qn·D1+Qn·D2+Qn·Dn+···············Q2·D1-Q2·D2-···Q1·Dn-Q2·Dn-Qn·Dn-············Qn·D1-Qn·D2-·········NegativesPositivesPositivePool
Contrastive Learning of Retriever       .SamplingNegativePool
Figure 2. Illustration of the Open-Rag training process.
withdin context:
x=PromptT(q, d)
Sq
d≜Pϕ(y|x) =Y
∀ti∈yPϕ(ti|t<i, x)
Here, y= (t1, . . . , t n)is a sequence of ntokens and Pϕis
thefunctionmeasurestheprobabilityofgeneratingthenext
token or spans. Unlike the RAG label, the computation of
the RAG score requires only a single forward pass of the
LLM.
3.2. Offline RAG
ForofflineRAG,wefollowthetraditionalRAGpipelineas
mentionedinSection2.2. Givenaquery q,weretrievetop-
kdocuments and denote this retrieved subset as Dq⊂ D
where |Dq|=k. We then compute the RAG label and
score for each retrieved document di, resulting in the set
{(q, di,Lq
di,Sq
di)}k
i=1. Based on their RAG labels, Dqis
furtherdividedintoapositivepool D+
qandanegativepool
D−
q. In our experiments, we set kto 100 and discard any
sample where either pool is empty.
These RAG offline preparation serve two purposes. First,
they establish initial positive and negative query-document
pairs to warm up the retriever for tasks. Second, they pro-
vide insights into the relationship between the RAG score
andtheRAGlabel. Specifically,wewanttodeterminewhen
theRAGscoreisaboveacertainthreshold,theRAGlabelis
1,andwhentheRAGscoreisbelowathreshold,thelabelis
0. This relationship will be used to approximate labels via
scoresduringonlineRAGtraining,enablingmoreefficient
online construction of positive and negative pairs.
3.3. Online RAG
In-trainingRetrieval. Duringretrievertraining,asitspa-
rameters update, the index needs to be rebuilt accordingly,
which incurs significant costs. To address this challenge,weemploythesemi-parametricretriever SiDR(Zhouetal.,
2024a). Specifically, SiDRincorporates both a paramet-
ric and a non-parametric encoder. The parametric encoder
embedstextinput xintoasparserepresentationwith |V|di-
mensions,whereeachdimensionsignifiestheimportanceof
atokenwithinthelanguagemodel’svocabulary V,denoted
asVθ(x). Conversely,thenon-parametricencoderconverts
xintobag-of-tokensrepresentation,referredtoas VBoT(x),
whichisconstructedviaatokenizerandisindependentof θ.
SiDRis strategically trained to allow the embedded query
Vθ(q)to search on both an embedding-based index Vθ(D)
and a bag-of-tokens index VBoT(D).
We adopt the late parametric mechanism of SiDR, which
firstlyretrievethetop- mdocumentsusingthebag-of-tokens
index VBoT(D), denoted as:
{ˆd}m=Rθ(Vθ(q), VBoT(D), m)
These retrieved documents are then embedded and re-
rankedon-the-flytoyieldthetop- kwell-rankeddocuments,
where k < m:
{ˆd}k=Rθ(Vθ(q), Vθ({ˆd}m), k)
In this case, our in-training retrieval does not require index
updates,andtherelevanceisbasedontheup-to-dateparam-
eters. For late parametric mechanism, we set m=k= 20
toreducetrainingcost. Moredetailsof SiDRcanbefound
in Appendix D.
Identifying Positives and Negatives On-the-fly. During
training, we denote the pool of top- kretrieved documents
asˆDq. Our goal is to divide ˆDqinto a positive pool ˆD+
q
andanegativepool ˆD−
qwithouttheneedforautoregressive
generation. Wepresenthowtoachievethisidentificationin
two generation scenarios.
Forfree-form generation , such as in question answering
tasks,thecontinuation ytypicallyconsistsofamulti-token
4

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
answer string. We identify a retrieved document ˆdas pos-
itive if its RAG score surpasses the highest RAG score in
the offline negative pool D−
qand as negative if it is be-
low the lowest RAG score in the offline positive pool D+
q;
otherwise, it is excluded:
ˆLq
ˆd=

1,ifSq
ˆd>max{Sq
d| ∀d∈ D−
q}
0,ifSq
ˆd<min{Sq
d| ∀d∈ D+
q}
None,otherwise
Here, we use ˆLto denote the online RAG label, as it in-
volves certain approximation. The approximation is based
on theassumption that a higher RAG score correlates with
an increased probability that the generated output ˆywill
match or reflect the target y. This strategy aims to reduce
computationalcosts,enablinglow-resourceinstitutionsand
individuals to conduct retriever training. If computational
resources are not a limitation, ideally, one could perform
autoregressive generation and evaluation on-the-fly or em-
ploy a larger LLM for identification purposes. We provide
further discussion and verification of this assumption in
Appendix C.
Forclosed-set generation , such as in multiple-choice rea-
soningorfact-checkingtasks,thecontinuation yistypically
a single-token choice label or can be prompted as such. In
this case, we can relax the assumptions:
ˆLq
ˆd=(
1,ifPϕ(ci|x)>max{Pϕ(cj|x)| ∀j̸=i}
0,otherwise.
Here, xistheinputpromptand ciisthecorrectsingle-token
choicewhile cjaretheincorrectchoices. Thissetupchecks
whetherLLMismorelikelytogenerate ciasthenexttoken
following xinstead of cj, when ˆdis used in context.
Forbothscenarios,ifaqueryhasmultiplecorrectcontinua-
tiony(answersorchoices),each yistreatedasanindividual
entry. If ˆdsucceedsonatleastoneoftheseentries,welabel
it as positive; if it fails all of them, we label it as negative.
Sampling and Cache. During the online phase, we re-
trieve the top- kdocuments and compute their RAG scores
toapproximateRAGlabels,processingthemindescending
orderofretrievalrelevance. Westopthisprocessatthefirst
document classified as negative. We then use this highest
relevant negative, denoted as ˆd−, and randomly select one
positive ˆd+from the pool ˆD+
q. If either is unavailable, we
fallbacktorandomsamplingfromofflinepositivepool D+
q
ornegative D−
q. Toavoidredundantcalculations,wecache
alltheonlinescoresandlabels {(q,ˆdi,ˆLq
ˆdi,ˆSq
ˆdi)}forreuse.
3.4. Contrastive Learning
Throughoutourofflineandonlineefforts,ourobjectiveisto
acquirehigh-qualitypositiveandnegativequery-documentpairs for the contrastive learning (Jaiswal et al., 2020) of
theretriever Rθ. Positivesandnegativesaredeterminedby
their impact on the RAG output; specifically, their ability
to enable the RAG framework to generate the correct con-
tinuation that meets the criteria of the evaluation metric.
This ensures that supervision signals are propagated from
the end of the RAG pipeline back to the retriever.
Ourtrainingobjectiveremainsthesameas SiDRtomaintain
its ability for late parametric. Given a batch Bthat consist
ofNsamples,eachsampleconsistsofaquery qi,apositive
document d+
i, and a negative document d−
i. Our training
objectiveaimstomaximizethesimilarityofpositivequery-
documentpairs f(qi, d+
i)forallinstances i,whileminimize
the similarity of all negative pairs, denoted as f(qi, d)for
alld̸=d+
i. Thecontrastivelosscanbedefinedasfollows:
L(q, d) =−NX
i=1(logef(qi,d+
i)
P
∀d∈Bef(qi,d)
|{z }
q-to-d+ logef(d+
i,qi)
P
∀q∈Bef(d+
i,qi)
|{z }
d-to-q)
Thefinallossintegratescontrastivelossofbothparametric
and semi-parametric components:
Lpara(q, d) =L(Vθ(q), Vθ(d))
Lsemi-para (q, d) =L(Vθ(q), VBoT(d))/2 +L(VBoT(q), Vθ(d))/2
Lfinal(q, d) =Lpara(q, d) +Lsemi-para (q, d)
4. Experiments
4.1. Experimental Setup
Tasks and Datasets. We evaluate Open-Rag on four pub-
lic RAG benchmarks. For free-form generation, we uti-
lize Natural Questions (NQ; Kwiatkowski et al., 2019) and
TriviaQA (TQA; Joshi et al., 2017), two well-established
open-domain QA datasets. For closed-set generation, we
employ the PubHealth (Kotonya & Toni, 2020) dataset for
fact-checking tasks, and the ARC-Challenge (Clark et al.,
2018) dataset for multiple-choice reasoning. More infor-
mation about the datasets can be found in Appendix A.
We exclude long-form generation datasets as we use the
probability of continuation to approximate RAG perfor-
mance, which may not align well with such tasks. Ad-
ditionally, certain datasets, such as PopQA (Mallen et al.,
2023), which only offer a test split, are also excluded.
EvaluationMetrics. Followingpreviousworks(Asaietal.,
2023;Mallenetal.,2023),weuseaccuracyastheevaluation
metric and report results on the test set. In IR scenarios,
accuracy is measured by whether the retrieved documents
containtheexpectedanswers,whileinRAGscenarios,itis
assessed based on the generated output. Since our training
uses1documentincontextwhileexistingresearchgenerally
uses 10 for RAG, we report accuracy with both 1 and 10
documents in context for comparison.
5

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
Table 2.Mainresultsof Open-Rag andotherRAGbaselineson4datasets,usingtop-1andtop-10retrieveddocumentsincontext. Bold:
best RAG method that does not involve LLM tuning. ∆: improvement or decline; ▲: baseline that below methods compare with;
†: reproduction from other works; ‡: our reproduction; §: has accessed the training split of the dataset.
Task Type ( →) Free-form Closed-set
Dataset ( →) NQ TriviaQA PubHealth ARC-C
Method ( ↓) Metrics ( →) 1-doc ∆10-doc ∆1-doc ∆10-doc ∆1-doc ∆10-doc ∆1-doc ∆10-doc ∆
Standard RAG
Baseline IR
Llama3 8B+SiDR MS 34.4 ▲37.6 ▲62.0 ▲62.5 ▲63.5 ▲64.9 ▲56.9 ▲57.5 ▲
Llama3 8B+SiDR NQ §42.7+8.3§41.6+4.0–– – ––– – ––– – –
Advanced IR
Llama3 8B+Contriever MS 36.5+2.1 38.3 +0.760.7-1.3 60.6 -1.963.1-0.4 62.9 -2.058.1+1.2 58.9 +1.4
Llama3 8B+E5 §43.2+8.8§41.8+4.263.2+1.2 61.4 -1.164.7+1.2 63.7 -1.258.0+1.1 58.1 +0.6
RAG with IR tuning
†RePlug Llama2-7B(3-doc, Yue et al. (2024)) – – – ––– – ––– 41.7 ––– 47.2 –
Ours
Open-Rag (SiDR MS) 39.8 +5.4 40.9 +3.365.8+3.8 66.2 +3.769.5+6.0 69.3 +4.458.1+1.258.3+0.8
Open-Rag (SiDR NQ) §44.1+9.7§44.7+7.1–– – ––– – ––– – –
RAG with LLM tuning
Llama3-Instruct 8B+SiDR MS 41.2+6.8 52.1 +14.565.2+3.2 73.3 +10.867.2+3.7 71.8 +6.972.1+15.2 75.5 +18.0
Self-RAG Llama2-7B(Asai et al., 2023) – – – ––– 66.4 +3.9–– 72.4 +7.5–– 67.3 +9.8
†Self-RAG Mistral-7B(Wang et al., 2024d) – – – ––– 64.8 +2.3–– 72.4 +7.5–– 74.9 +17.4
†Self-RAG Llama3-8B(Zhang et al., 2024) – – – ––– 56.4 -6.1–– 67.8 +2.9–– 58.0 +0.5
‡Self-RAG Llama3-8B+SiDR MS 30.8-3.6 37.0 -0.651.0-11.0 57.7 -4.864.2+0.7 64.0 -0.958.9+2.0 59.1 +1.6
Transferring Open-Rag to other LLM
Llama3-Instruct 8B+SiDR MS 41.2 ▲52.1 ▲65.2 ▲73.3 ▲67.2 ▲71.8 ▲72.1 ▲75.5 ▲
Llama3-Instruct 8B+Open-Rag (SiDR MS) 43.6 +2.4 54.7 +2.665.6+0.4 73.8 +0.565.2-2.0 66.1 -5.771.9-0.2 75.0 -0.5
Phi-3-mini-4k-instruct 3.8B+SiDR MS 40.6 ▲49.2 ▲64.6 ▲69.2 ▲48.2 ▲57.6 ▲84.9 ▲84.3 ▲
Phi-3-mini-4k-instruct 3.8B+Open-Rag (SiDR MS) 43.4+2.8 50.3 +1.165.6+1.0 70.4 +1.245.3-2.9 54.4 -3.285.1+0.2 84.6 +0.3
Mistral-Instruct 7B+SiDR MS 37.5 ▲48.0 ▲58.2 ▲57.1 ▲50.1 ▲57.4 ▲69.7 ▲71.5 ▲
Mistral-Instruct 7B+Open-Rag (SiDR MS) 40.5 +3.0 49.4 +1.459.8+1.6 57.6 +0.546.7-3.4 54.6 -2.869.2-0.5 70.6 -0.9
Implementation Details. Our RAG system employs the
LLM Llama3-8b (Dubey et al., 2024) with the retriever
SiDR MS(Zhou et al., 2024a) that trained on MS MARCO
dataset (Bajaj et al., 2016). We use the same English
Wikipedia datastore and prompt as those open-sourced by
Self-RAG , detailed in Appendix H. During training, we
train the retriever for each dataset for 80 epochs, aligning
withthetrainingdurationusedfor SiDR MS. Weuseabatch
sizeof128andanAdamWoptimizer(Loshchilov&Hutter,
2018)withalearningrateof 2×10−5. Thetrainingprocess
isdividedintotwophases: thefirsthalfinvolvesawarm-up
phase using offline positives and negatives, while the sec-
ond half transitions to in-training retrieval, primarily using
the positives and negatives identified on-the-fly. During
inference, we set the maximum number of generated token
to be 100 for free-form generation while 20 for closed-set
generation.
Training Costs. Our experiments are conducted with 4
NVIDIA A100 GPUs. Both offline RAG preparation and
online RAG training take less than one day, depending
on the number of queries in the datasets. We leverage
vLLM (Kwon et al., 2023) to accelerate offline generation.
Baselines. We consider the baselines detailed below, with
additional model information provided in Appendix B.
(1)Standard RAG with advanced IR : RAGframeworksus-
ing Llama3-8b and state-of-the-art retrievers E5(Wang
et al., 2022) and Contriever MS(Izacard et al., 2021). Werefer to Open-Rag (SiDR NQ) and Open-Rag (SiDR MS) as
our framework utilizing SiDR NQandSiDR MSas the ini-
tial retriever, respectively. Unless explicitly stated other-
wise, Open-Rag refers to Open-Rag (SiDR MS). For a fair
comparison, we compare E5with Open-Rag (SiDR NQ),
both of which have access to the query-document pairs
from the NQ training split. (2) RAG with IR tuning :
RAG frameworks that incorporate a tunable IR compo-
nent. We compare against RePlug(Shi et al., 2023),
which uses part of a sequence as query to retrieve doc-
uments which maximize the generation likelihood of the
remaining part. Since the model weights are not pub-
licly available, we reference a reproduction by (Yue et al.,
2024) that uses the top-3 retrieved documents in context.
(3)RAG with LLM tuning : RAG frameworks that incorpo-
rate RAG-oriented or instruction-tuned LLMs, which typ-
ically require more resources for tuning an 8B LLM. We
comparewith Self-RAG (Asaietal.,2023)usingLlama2-
7B, along with some reproductions (Zhang et al., 2024;
Wang et al., 2024d) employing more recent LLMs. Our
primary comparison with Self-RAG and its variants is
designed to ensure a controlled and fair evaluation, as
we adhere to the same prompts and downstream evalua-
tion pipeline. (4) Transferring to other LLMs : We com-
pare the RAG framework using different LLMs, such as
Llama3-Instruct 8B(Dubey et al., 2024), Phi-3-mini-4k-
instruct 3.8B(Abdin et al., 2024), Mistral-Instruct 7B(Jiang
et al., 2023), along with SiDR MSbefore and after tuning.
6

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
This setup is designed to evaluate whether the learned in-
context relevance transfers across different LLMs.
4.2. Main Experiments
Table 2 presents results of Open-Rag and other baselines.
The key findings are summarized as follows:
End-to-end tuning effectively improves the retriever in
RAG scenarios, surpassing existing SOTA retrievers.
Unlike E5andContriever MS, which require both exten-
sivepre-trainingandhuman-labeledquery-documentpairs,
Open-Rag improves the initial retriever using only down-
stream queries, achieving better automation and training
efficiency. Our approach leads to a notable 4.0% enhance-
mentinperformancebeyondtheoriginal SiDR MSandcon-
sistently achieves a 2.1% better outcome than the SOTA
retrievers. For PubHealth, the improvement reaches up to
6%, a significant value that even using instruction-tuned
LLMs cannot achieve. For ARC, the modest improvement
can be attributed to its limited number of training samples,
only a few hundred, compared to other datasets containing
tens of thousands. These results demonstrate that, despite
approximation,thelearnedin-contextrelevanceismoreef-
fectivethantheinconsistentrelevancederivedfromexisting
datasets. InAppendixF,weshowthatimprovetheretriever
for RAG scenarios may degrade its performance in tradi-
tional IR scenarios, further reinforcing this inconsistency.
Relevancelearningconstitutesavaluableyetoverlooked
dimension for improving the RAG system. Reproduc-
tionsof Self-RAG usingLlama3-8Bbyotherworks(Zhang
et al., 2024; Wang et al., 2024d) and ourselves have not
yieldedconsistentimprovements. Thissuggeststhatdespite
the substantial training expenses, enhancing RAG through
tuning LLM requires extensive customization and does not
reliably generalize. In contrast, tuning a smaller-sized re-
triever can lead to comparable, or in some cases, supe-
rior improvements over those achieved by RAG-oriented
or instruction-tuned 8B LLMs on specific datasets. Impor-
tantly,learninganin-contextretrieverdoesnotconflictwith
LLM enhancements, offering a complementary avenue for
improving the RAG system.
The learned in-context retriever can be transferred to
other LLMs for free-form generation tasks. Our results
show that Open-Rag , initially co-trained with Llama3-8b,
enhances other LLMs such as Llama3-Instruct-8B, Phi-3-
mini-4k-instruct, and Mistral-Instruct in free-form genera-
tion tasks. However, for closed-set generation tasks, this
transferability does not consistently hold. Despite the lim-
itations, Open-Rag significantly enhances performance of
PubHealth by a large margin. We hypothesize that closed-
settasks,wherethecontinuationisasingletoken,areeasier
to optimize due to less approximation involved. Conse-
quently, the retriever learns a very specific relevance tai-lored to the particular LLM prediction of the next token,
complicating its transferability. Therefore, we recommend
end-to-end tuning on a LLM-by-LLM basis to potentially
improve outcomes for these tasks.
4.3. Ablation Study
Compared to prior works, our main differences include (i)
employingcontrastivelearninginsteadofKLdivergenceto
induce supervision signals from the LLM to the IR, and
(ii)usinglateparametrictoavoidperiodicre-indexing. We
systematically analyze these factors in this section.
Figure 3. Ablation studies on NQ and Pubhealth datasets.
As shown in Figure 3, we conducted an ablation study on
NQ and PubHealth with several setup: our method is la-
beled as [offline+online] , where[offline-only] represents
usingonlytheofflinepositivesandnegativesforcontrastive
learning, and [online-only] indicates that we do not use
any warmup. We also explore using KL divergence [of-
fline+online(KL)] instead of contrastive learning.
Offline versus Online. During the warmup stage, docu-
ments are retrieved using the initial parameters θ. Dur-
ing the in-training retrieval stage, they are retrieved using
the up-to-date parameters θ′. We assess the improvements
provided by the in-training retrieval stage. As shown in
Figure 3, relying solely on either [offline-only] or[online-
only]can lead to suboptimal improvements, proving to be
less effective than a combination of a warmup phase fol-
lowed by online in-training retrieval [offline+online] . This
observationechoestheconclusionsofpriorresearch(Zhou
etal.,2024a),whichindicatesthatwarminguptheretriever
to initially capture the in-task relevance, followed by in-
trainingretrievaltocontinuouslyexplorepotentialpositives
andchallengingnegativesinthedatastore,cansignificantly
enhance performance.
Contrastive Learning versus KL-Divergence. Prior
works (Shi et al., 2023; Guu et al., 2020) have employed
KLdivergence toalignquery-documentrelevance withthe
distributionofgenerationlikelihood. Ourexperimentsindi-
catethatwhileKLdivergenceleadstoimprovements,these
benefits quickly stabilize and the overall enhancement falls
short of our method. Unlike our approach, which employs
contrastive learning requiring efforts to identify positives
7

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
and negatives, KL divergence alignment offers a straight-
forward but potentially overly restrictive solution. On one
hand,inRAGscenarios,documentsaredeliveredtoLLMs,
differingfromIRscenarioswheredocumentsmustbewell-
ranked before being presented to users. For a proficient
LLM, including even a single useful document in the con-
text window should suffice (Cuconasu et al., 2024a). On
theotherhand,similarworksinknowledgedistillation(Gou
et al., 2021), which uses cross-encoder scores to guide bi-
encoder training, demonstrate that improvements for bi-
encoders are limited and cannot match the performance of
cross-encoder rerankers. Consequently, the prevalent in-
dustry practice of retrieve-then-rerank (Gupta et al., 2018)
underscoresthecurrentlimitationsofretrieversincapturing
complex relationships. We believe that the distribution of
generation likelihood from LLMs is too complex for these
small-sizedretrievertoaccuratelycapture,therebyresulting
in less improvement.
Late Parametric versus Periodic Re-indexing. Due to
page limitations, we detail our comparison of different in-
trainingretrievalmethodsinAppendixE.Thiscomparison
particularly focuses on the late parametric method versus
prior solutions that utilize an embedding index and require
periodicre-indexing. Ourresultsindicatethatthelatepara-
metric method not only leads to better improvements but
also reduces training costs and simplifies the implementa-
tion. Webelievethatthehighcostsandcompleximplemen-
tation associated with periodic re-indexing have prevented
previous research from effectively training retrievers on a
task-by-taskbasis,usingconsistentinstructions,LLMs,and
datastores tailored to downstream tasks, ultimately leading
to less effective results.
4.4. Cost-Effectiveness Analysis
Regarding training costs, the primary expense comes from
computing the RAG scores using the LLM. In Table 3, we
report the number of documents required to compute RAG
scores on-the-fly during training.
NQ TriviaQA PubHealth ARC
nDoc 20 18 128 15
Improv. +5.4% +3.8% +6.0% +1.2%
Table 3.Number of documents required on-the-fly RAG score
computation and the improvement for each task.
Throughout training, each query encounters between 15 to
128 unscored documents, depending on the task, requiring
LLM forward passes to compute RAG scores on-the-fly.
This process incurs a manageable cost, typically amount-
ing to hours rather than days. We also observe a positive
correlation between the number of documents processed
andtheperformanceimprovementsof Open-Rag . Notably,
thePubHealthdatasetrequiresmoredocumentstocompute
the RAG score online, resulting in the most significant im-provement. Thissuggeststhatencounteringmoreunscored
documents indicates a larger gap in relevance between the
initial and the learned retriever, highlighting the presence
of more potentially useful documents in the datastore that
could be leveraged by in-context retrieval learning.
5. Related Works
Retrieval-augmented Generation (RAG). The RAG sys-
tem combines LLMs, retrievers, and datastores, each con-
tributingtoperformanceimprovement. Significantresearch
hasfocusedonimprovingRAGbytuningLLMstoaddress
challenges such as enhancing on-demand retrieval (Asai
et al., 2023; Jeong et al., 2024), optimizing response effi-
ciency (Wang et al., 2024d), and enabling self-reasoning
capabilities (Li et al., 2024). Additional efforts have ex-
plored building domain-specific (Wang et al., 2024e) or
large datastores (Shao et al., 2024). While some stud-
ies focus on retrieval, exploring adaptive retrieval strate-
gies(Wangetal.,2024a;c)andleveragingLLMstodevelop
stronger retrievers (Guu et al., 2020; Shi et al., 2023), re-
searchonend-to-endrelevancelearningforRAGscenarios
remains limited. Our work addresses this gap, paving the
way for new advancements in RAG systems.
Relevance Learning. Relevance learning is an important
andlong-establishedareaofresearch. Traditionally,textrel-
evancehasbeenmeasuredbyheuristicrulesbasedonterm
overlap,asseeninthewidely-usedBM25(Robertsonetal.,
2009). With advances in deep learning, neural retrievers
have emerged (Karpukhin et al., 2020), learning relevance
fromhuman-annotateddatasets(Kwiatkowskietal.,2019).
Further research has explored pre-training retrievers using
weakly supervised text pairs, such as cropped text spans
within documents (Izacard et al., 2021) and relational text
pairsextractedfromwebdata(Zhouetal.,2022;Wangetal.,
2022), to enable retrievers to learn general relevance. This
general relevance can then be refined to task-specific and
domain-specificrelevancethroughdownstreamfine-tuning,
resultinginimprovedperformance. Ourmethodfallswithin
these advancements, where the LLM acts as a container of
general relevance, providing on-the-fly supervision of spe-
cific in-context relevance for relevance learning.
6. Conclusion
In this work, we show that traditional retrieval relevance
derived from QA datasets can be inconsistent in RAG sce-
narios. To bridge this gap, we introduce Open-Rag , a
RAG framework that learns in-context retrieval end-to-end
for downstream tasks. Our framework consistently outper-
formsRAGframeworksusingSOTAretrieversandseveral
thattunean8BLLM.Thishighlightsthesignificantpoten-
tial of retrieval learning to improve RAG performance.
8

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
References
Abdin, M., Aneja, J., Awadalla, H., Awadallah, A., Awan,
A. A., Bach, N., Bahree, A., Bakhtiari, A., Bao, J.,
Behl, H., et al. Phi-3 technical report: A highly capable
language model locally on your phone. arXiv preprint
arXiv:2404.14219 , 2024.
Arora,S.,Lewis,P.,Fan,A.,Kahn,J.,andRé,C.Reasoning
over public and private data in retrieval-based systems.
Transactions of the Association for Computational Lin-
guistics, 11:902–921, 2023.
Asai,A.,Wu,Z.,Wang,Y.,Sil,A.,andHajishirzi,H. Self-
rag: Learning to retrieve, generate, and critique through
self-reflection. arXiv preprint arXiv:2310.11511 , 2023.
Ayyamperumal, S. G. and Ge, L. Current state of llm
risksandaiguardrails. arXivpreprintarXiv:2406.12934 ,
2024.
Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao,
J., Liu, X., Majumder, R., McNamara, A., Mitra, B.,
Nguyen, T., et al. Ms marco: A human generated ma-
chine reading comprehension dataset. arXiv preprint
arXiv:1611.09268 , 2016.
Chan, C.-M., Xu, C., Yuan, R., Luo, H., Xue, W., Guo, Y.,
andFu,J. Rq-rag: Learningtorefinequeriesforretrieval
augmentedgeneration. arXivpreprintarXiv:2404.00610 ,
2024.
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,
Schoenick, C., and Tafjord, O. Think you have solved
questionanswering? tryarc,theai2reasoningchallenge.
arXiv preprint arXiv:1803.05457 , 2018.
Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Cam-
pagnano, C., Maarek, Y., Tonellotto, N., and Silvestri, F.
Thepowerofnoise: Redefiningretrievalforragsystems.
InProceedings of the 47th International ACM SIGIR
Conference on Research and Development in Informa-
tion Retrieval , pp. 719–729, 2024a.
Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Cam-
pagnano,C.,Maarek,Y.,Tonellotto,N.,Silvestri,F.,etal.
Rethinking relevance: How noise and distractors impact
retrieval-augmented generation. In CEUR WORKSHOP
PROCEEDINGS , volume 3802, pp. 95–98. CEUR-WS,
2024b.
Devlin,J.,Chang,M.-W.,Lee,K.,andToutanova,K. Bert:
Pre-training of deep bidirectional transformers for lan-
guage understanding. In Proceedings of the 2019 Con-
ference of the North American Chapter of the Associa-
tion for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers) , pp.
4171–4186, 2019.Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,
A.,Letman,A.,Mathur,A.,Schelten,A.,Yang,A.,Fan,
A., et al. The llama 3 herd of models. arXiv preprint
arXiv:2407.21783 , 2024.
Gao,Y.,Xiong,Y.,Gao,X.,Jia,K.,Pan,J.,Bi,Y.,Dai,Y.,
Sun, J., and Wang, H. Retrieval-augmented generation
for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2023.
Gou, J., Yu, B., Maybank, S. J., and Tao, D. Knowledge
distillation: Asurvey. InternationalJournalofComputer
Vision, 129:1789–1819, 2021.
Gupta, V., Chinnakotla, M., and Shrivastava, M. Retrieve
and re-rank: A simple and effective ir approach to sim-
ple question answering over knowledge graphs. In Pro-
ceedings of the First Workshop on Fact Extraction and
VERification (FEVER) , pp. 22–27, 2018.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.
Retrieval augmented language model pre-training. In
Internationalconferenceonmachinelearning ,pp.3929–
3938. PMLR, 2020.
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bo-
janowski, P., Joulin, A., and Grave, E. Towards un-
supervised dense information retrieval with contrastive
learning. arXiv preprint arXiv:2112.09118 , 2021.
Jaiswal, A., Babu, A. R., Zadeh, M. Z., Banerjee, D., and
Makedon, F. A survey on contrastive self-supervised
learning. Technologies , 9(1):2, 2020.
Jeong, S., Baek, J., Cho, S., Hwang, S. J., and Park, J. C.
Adaptive-rag: Learning to adapt retrieval-augmented
large language models through question complexity.
arXiv preprint arXiv:2403.14403 , 2024.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot,D.S.,Casas,D.d.l.,Bressand,F.,Lengyel,G.,
Lample,G.,Saulnier,L.,etal. Mistral7b. arXivpreprint
arXiv:2310.06825 , 2023.
Joshi,M.,Choi,E.,Weld,D.S.,andZettlemoyer,L. Trivi-
aqa: A large scale distantly supervised challenge dataset
for reading comprehension. In Proceedings of the 55th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pp. 1601–1611,
2017.
Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. In Pro-
ceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP) , pp. 6769–
6781, 2020.
9

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
Ke, Z., Kong, W., Li, C., Zhang, M., Mei, Q., and Bender-
sky, M. Bridging the preference gap between retrievers
and LLMs. In Ku, L.-W., Martins, A., and Srikumar,
V. (eds.), Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers) , pp. 10438–10451, Bangkok, Thailand,
August2024.AssociationforComputationalLinguistics.
doi: 10.18653/v1/2024.acl-long.562. URL https://
aclanthology.org/2024.acl-long.562/ .
Koo, H., Kim, M., and Hwang, S. J. Optimizing query
generationforenhanceddocumentretrievalinrag. arXiv
preprint arXiv:2407.12325 , 2024.
Kotonya, N. and Toni, F. Explainable automated fact-
checking for public health claims. In Proceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pp. 7740–7754, 2020.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., De-
vlin,J.,Lee,K.,etal. Naturalquestions: abenchmarkfor
questionansweringresearch. TransactionsoftheAssoci-
ation for Computational Linguistics , 7:453–466, 2019.
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu,
C.H.,Gonzalez,J.E.,Zhang,H.,andStoica,I. Efficient
memory management for large language model serving
withpagedattention.In ProceedingsoftheACMSIGOPS
29thSymposiumonOperatingSystemsPrinciples ,2023.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin,
V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rock-
täschel, T., et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in Neural In-
formation Processing Systems , 33:9459–9474, 2020.
Li,H.,Verga,P.,Sen,P.,Yang,B.,Viswanathan,V.,Lewis,
P., Watanabe, T., and Su, Y. Alr: A retrieve-then-reason
framework for long-context question answering. arXiv
preprint arXiv:2410.03227 , 2024.
Liu, X.-Y., Wang, G., and Zha, D. Fingpt: Democratizing
internet-scale data for financial large language models.
arXiv preprint arXiv:2307.10485 , 2023.
Liu, Z., Ping, W., Roy, R., Xu, P., Lee, C., Shoeybi, M.,
and Catanzaro, B. Chatqa: Surpassing gpt-4 on conver-
sational qa and rag. arXiv preprint arXiv:2401.10225 ,
2024.
Loshchilov, I. and Hutter, F. Decoupled weight decay reg-
ularization. In International Conference on Learning
Representations , 2018.Mallen, A. T., Asai, A., Zhong, V., Das, R., Khashabi,
D., and Hajishirzi, H. When not to trust language mod-
els: Investigating effectiveness of parametric and non-
parametric memories. In The 61st Annual Meeting Of
The Association For Computational Linguistics , 2023.
Manzoor,A.andJannach,D. Towardsretrieval-basedcon-
versational recommendation. Information Systems , 109:
102083, 2022.
Min,S.,Gururangan,S.,Wallace,E.,Shi,W.,Hajishirzi,H.,
Smith,N.A.,andZettlemoyer,L.SILOlanguagemodels:
Isolating legal risk in a nonparametric datastore. In The
TwelfthInternationalConferenceonLearningRepresen-
tations, 2024. URL https://openreview.net/
forum?id=ruk0nyQPec .
Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M.,
Socher, R., Amatriain, X., and Gao, J. Large language
models: A survey. arXiv preprint arXiv:2402.06196 ,
2024.
Nian, J., Peng, Z., Wang, Q., and Fang, Y. W-rag: Weakly
supervised dense retrieval in rag for open-domain ques-
tionanswering. arXivpreprint arXiv:2408.08444 , 2024.
Robertson, S., Zaragoza, H., et al. The probabilistic rele-
vance framework: Bm25 and beyond. Foundations and
Trends®in Information Retrieval , 3(4):333–389, 2009.
Serouis,I.M.andSèdes,F. Exploringlargelanguagemod-
els for bias mitigation and fairness. In 1st International
Workshop on AI Governance (AIGOV) in conjunction
with the Thirty-Third International Joint Conference on
Artificial Intelligence , 2024.
Shao, R., He, J., Asai, A., Shi, W., Dettmers, T., Min, S.,
Zettlemoyer, L., and Koh, P. W. Scaling retrieval-based
language models with a trillion-token datastore. arXiv
preprint arXiv:2407.12854 , 2024.
Shi,W.,Min,S.,Yasunaga,M.,Seo,M.,James,R.,Lewis,
M., Zettlemoyer, L., and Yih, W.-t. Replug: Retrieval-
augmented black-box language models. arXiv preprint
arXiv:2301.12652 , 2023.
Wang,F.,Wan,X.,Sun,R.,Chen,J.,andArık,S.Ö. Astute
rag: Overcoming imperfect retrieval augmentation and
knowledge conflicts for large language models. arXiv
preprint arXiv:2410.07176 , 2024a.
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L.,
Jiang, D., Majumder, R., and Wei, F. Text embeddings
by weakly-supervised contrastive pre-training. arXiv
preprint arXiv:2212.03533 , 2022.
10

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang,
J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A sur-
vey on large language model based autonomous agents.
Frontiers of Computer Science , 18(6):1–26, 2024b.
Wang, X., Fei, Y., Leng, Z., and Li, C. Does role-playing
chatbots capture the character personalities? assessing
personalitytraitsforrole-playingchatbots. arXivpreprint
arXiv:2310.17976 , 2023.
Wang, X., Wang, Z., Gao, X., Zhang, F., Wu, Y., Xu, Z.,
Shi, T., Wang, Z., Li, S., Qian, Q., et al. Searching
for best practices in retrieval-augmented generation. In
Proceedingsofthe2024ConferenceonEmpiricalMeth-
ods in Natural Language Processing , pp. 17716–17736,
2024c.
Wang,Z.,Wang,Z.,Le,L.,Zheng,H.S.,Mishra,S.,Perot,
V., Zhang, Y., Mattapalli, A., Taly, A., Shang, J., et al.
Speculative rag: Enhancing retrieval augmented genera-
tion through drafting. arXiv preprint arXiv:2407.08223 ,
2024d.
Wang,Z.Z.,Asai,A.,Yu,X.V.,Xu,F.F.,Xie,Y.,Neubig,
G., and Fried, D. Coderag-bench: Can retrieval aug-
mentcodegeneration? arXivpreprintarXiv:2406.14497 ,
2024e.
Wenzek,G.,Lachaux,M.-A.,Conneau,A.,Chaudhary,V.,
Guzmán, F., Joulin, A., and Grave, E. Ccnet: Extracting
high quality monolingual datasets from web crawl data.
arXiv preprint arXiv:1911.00359 , 2019.
Wu, M. and Cao, S. Llm-augmented retrieval: Enhancing
retrieval models through language models and doc-level
embedding. arXiv preprint arXiv:2404.05825 , 2024.
Wu, S., Xie, J., Chen, J., Zhu, T., Zhang, K., and
Xiao, Y. How easily do irrelevant inputs skew the re-
sponses of large language models? arXiv preprint
arXiv:2404.03302 , 2024.
Xiong,L.,Xiong,C.,Li,Y.,Tang,K.-F.,Liu,J.,Bennett,P.,
Ahmed,J.,andOverwĳk,A.Approximatenearestneigh-
bor negative contrastive learning for dense text retrieval.
arXiv preprint arXiv:2007.00808 , 2020.
Xu,P.,Ping,W.,Wu,X.,McAfee,L.,Zhu,C.,Liu,Z.,Sub-
ramanian, S., Bakhturina, E., Shoeybi, M., and Catan-
zaro, B. Retrieval meets long context large language
models.arXiv preprint arXiv:2310.03025 , 2023.
Yu, Y., Ping, W., Liu, Z., Wang, B., You, J., Zhang, C.,
Shoeybi,M.,andCatanzaro,B. Rankrag: Unifyingcon-
textrankingwithretrieval-augmentedgenerationinllms.
arXiv preprint arXiv:2407.02485 , 2024.Yue, S., Wang, S., Chen, W., Huang, X., and Wei,
Z. Synergistic multi-agent framework with trajectory
learning for knowledge-intensive tasks. arXiv preprint
arXiv:2407.09893 , 2024.
Zhang, X., Song, Y.-Z., Wang, Y., Tang, S., Li, X., Zeng,
Z., Wu, Z., Ye, W., Xu, W., Zhang, Y., et al. Raglab:
A modular and research-oriented unified framework for
retrieval-augmented generation. In Proceedings of the
2024 Conference on Empirical Methods in Natural Lan-
guageProcessing: SystemDemonstrations ,pp.408–418,
2024.
Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y.,
Min,Y.,Zhang,B.,Zhang,J.,Dong,Z.,etal.Asurveyof
largelanguagemodels. arXivpreprintarXiv:2303.18223 ,
2023.
Zhou, J., Li, X., Shang, L., Luo, L., Zhan, K., Hu, E.,
Zhang, X., Jiang, H., Cao, Z., Yu, F., et al. Hyperlink-
inducedpre-trainingforpassageretrievalinopen-domain
question answering. In Proceedings of the 60th Annual
MeetingoftheAssociationforComputationalLinguistics
(Volume 1: Long Papers) , pp. 7135–7146, 2022.
Zhou, J., Dong, L., Wei, F., and Chen, L. Semi-
parametric retrieval via binary token index. arXiv
preprint arXiv:2405.01924 , 2024a.
Zhou, J., Li, X., Shang, L., Jiang, X., Liu, Q., and
Chen, L. Retrieval-based disentangled representation
learning with natural language supervision. In The
TwelfthInternationalConferenceonLearningRepresen-
tations, 2024b. URL https://openreview.net/
forum?id=ZlQRiFmq7Y .
11

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
A. Details of Datasets
We present details of datasets as follows.
•Natural Questions (NQ; Kwiatkowski et al., 2019) is a widely used open-domain QA dataset constructed from
Wikipedia. The questions originate from Google search queries, and the answers are text spans within Wikipedia
passages. This dataset consists of queries with one or more answer strings, requiring RAG systems to generate
responses based on factual knowledge.
•TriviaQA(TQA;Joshietal.,2017)isachallengingQAdatasetthatcomprisesquestion-answerpairscuratedbytrivia
enthusiasts along with independently gathered evidence documents.
•PubHealth (Kotonya & Toni, 2020) is a fact-checking task that focuses on verifying health claims across a variety of
biomedical topics.
•ARC-Challenge (Clark et al., 2018) is a multiple-choice reasoning dataset consisting of science exam questions for
grades 3 to 9.
B. Details of Baseline Models
The information for baseline models are listed as follows.
B.1. Retrieval Model (IR)
•E5(Wangetal.,2022)isastate-of-the-artdenseretrieverthatpre-trainedonmillionsofweaklyrelatedtextpairsfrom
the Web. The unsupervised version of this model is denoted as E5-unsup. This model undergoes further fine-tuning
on natural language inference (NLI) datasets, as well as the Natural Questions and MS MARCO datasets, to enhance
its capabilities in downstream applications. The fine-tuned version is denoted as E5.
•Contriever (Izacard et al., 2021) is a widely-used dense retriever pre-trained unsupervised on Wikipedia data and
CCNet (Wenzek et al., 2019). The unsupervised version of this model is denoted as Contriever . It is further
fine-tuned on the MS MARCO dataset to enhance its retrieval performance, with the fine-tuned version denoted as
Contriever MS.
•DPR (Karpukhin et al., 2020) is a widely used dense passage retriever initialized with a BERT-based uncased
encoder (Devlin et al., 2019), and fine-tuned on downstream dataset. Specifically, DPR MSis fine-tuned on the MS
MARCO dataset, DPR NQon the NQ dataset, and DPR TQAon the TriviaQA dataset.
•SiDR(Zhouetal.,2024a)isasemi-parametricsparseretrieverthatsupportsusingbothembeddingsandtokenization
as index. This nature allows for in-training retrieval, where the model’s parameters dynamically update while the
retrieval index remains fixed. The model is initialized with a BERT-based uncased encoder (Devlin et al., 2019) and
fine-tuned exclusively on single dataset depending on the variant: SiDR MSis fine-tuned on the MS MARCO dataset,
SiDR NQon the NQ dataset, and SiDR TQAon the TriviaQA dataset.
All the above retrieval methods are initialized with a BERT-based encoder, which contains approximately 200 million
(0.2B) parameters.
B.2. Large Language Model (LLM)
•Llama3 8B(Dubey et al., 2024) is a variant of the latest Llama3 model series with 8 billion parameters.
•Llama3-Instruct 8B(Dubey et al., 2024) builds upon the Llama3 8Bby undergoing a post-training stage in which the
model is specifically tuned to follow instructions and align with human preferences to improve specific capabilities.
•Phi-3-mini-4k-instruct 3.8B(Abdin et al., 2024) is a lightweight widely-used LLM with 3.8 billion parameters, trained
on the Phi-3 dataset featuring synthetic and high-quality filtered web data, focused on reasoning and quality.
•Mistral-Instruct 7B(Jiang et al., 2023). We use Mistral-7B-Instruct-v0.3 LLM which is an instruct fine-tuned version
of the Mistral-7B-v0.3.
12

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
B.3. Retrieval-augmented Generation Framework (RAG)
•RePlug(Shi et al., 2023) is a RAG framework using GPT-3 and Contriever . The retriever is specifically trained to
use the first 128 tokens of a sequence as queries, with the goal of retrieving documents that maximize the probability
of generating the subsequent 128 tokens when these retrieved documents are prepended to the query.
•Self-RAG (Asai et al., 2023) is a RAG framework designed to improve response quality by enabling on-demand
retrieval and incorporating self-reflection mechanisms.
ThereproductionsbyWangetal.(2024d)andZhangetal.(2024), Self-RAG Mistral-7BandSelf-RAG Llama3-8Brespec-
tively, involve tuning Mistral-7B and Llama3-8B as base language models using the open-source data provided by
Self-RAG .
Our reproduction, Self-RAG Llama3-8B +SiDR MS, utilizes the Self-RAG Llama3-8Bcheckpoint from Zhang et al. (2024)
as LLM, while employing the same retriever SiDR MSand adapting it to our downstream setup.
C. Effectiveness of RAG Scores on Task Accuracy
Table 4.Results of RAG framework using top-1 and top-10 documents in context, sorted by retrieval relevance and RAG scores.
Task Type ( →) Free-form Closed-set
Dataset ( →) NQ TriviaQA PubHealth ARC-C
Method ( ↓) Metrics ( →) 1-doc 10-doc 1-doc 10-doc 1-doc 10-doc 1-doc 10-doc
Llama3 8B+SiDR MS(doc with top relevance) 49.1 51.4 65.3 67.2 65.2 67.4 58.1 57.3
Llama3 8B+SiDR MS(doc with top RAG scores) 85.1 76.2 88.7 84.2 87.4 77.4 95.6 83.6
GiventhatourlearningisbasedonusingtheRAGscoreasanindicatortoidentifypositiveandnegativedocuments,wenow
investigate whether using documents with higher RAG scores leads to improved RAG response quality. For each dataset,
we sample 1k samples from training split. For each query, we retrieve the top 100 documents, and then perform the RAG
pipelineusingonlythetop-1andtop-10documents,sortedbyretrievalrelevanceandRAGscores,respectively. Theresults,
shown in Table 4, indicate that RAG scores are indicative of the final accuracy of the RAG framework. Furthermore, the
high accuracy achieved using top RAG scores documents suggests that the datastore holds significant untapped potential,
which current retrieval strategies have not yet fully exploited.
Toourknowledge,usingRAGscorestoidentifypositivesandnegativesisaroughyetresource-efficientsolutionthatcould
cover most existing knowledge-intensive tasks, aligning with their evaluation metrics that often utilize string matching.
However, it may not be suitable for long-form generation, which requires different evaluation strategies. We believe it is
possibletocustomizetheidentificationofpositiveandnegativeexamplesbasedonthespecificneedsofeachtask. Ideally,
if computational cost is not a concern or resources are sufficient, a strong proprietary LLM like GPT-4 can be used for
contrastive identification on-the-fly.
Here are some additional observations: RAG scores are generally more indicative when using single document in context,
likely because they are computed in this manner, ensuring more consistent evaluations. Furthermore, the improved
performance seen in Table 4 compared to our main experiments may be attributed to the LLM having been pretrained on
the training split of these datasets.
D. Revisiting Semi-parametric Disentangled Retriever ( SiDR)
Our work adopts the recently proposed retriever SiDRas the backbone for two main reasons. First, it supports the
use of a non-parametric index, which enables in-training retrieval when the retriever’s parameters change dynamically.
Second, evaluating retriever checkpoints can be resource-intensive, as it requires embedding a large datastore with each
newcheckpoint. SiDRofferslateparametrictechniquesthatreducethisevaluationprocessfromafulldayonourresource
to just a few minutes, significantly accelerating our research.
SiDR(Zhou et al., 2024b;a) is a sparse disentangled retriever (also known as a sparse lexical retriever) that encodes text
chunks into a |V|-dimensional sparse representation, where each dimension represents the importance of a token within
the language model vocabulary V.SiDRis then trained to align the |V|-dimensional parametric embedding, denoted as
Vθ(x), with the |V|-dimensional bag-of-tokens representation, denoted as VBoT(x).
13

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
Figure 4. Illustration of semi-parametric disentangled retriever ( SiDR) framework, adapted from Zhou et al. (2024a).
At downstream, a parametric query embedding Vθ(q)can perform search on both an embedding-based index Vθ(D)and a
bag-of-tokens index VBoT(D), which leads to three distinct search schemes:
•Full parametric search utilizes a parametric index Vθ(D), which relies on embeddings derived from a neural
encoderforthedatastore. Therelevanceisdefinedastheinnerproductoftheembededqueryandembededdatastore:
fθ(q,D) =⟨Vθ(q), Vθ(D)⟩
This is the common indexing process for neural retrieval systems, which are effective but involve higher costs and
longer latency for embedding the entire Dto obtain the index Vθ(D).
•Semi-parametric beta search leverages a non-parametric index VBoT(D)based on BoT representations of the
datastore, which are constructed solely by a tokenizer. The relevance is defined as:
fβ(q,D) =⟨Vθ(q), VBoT(D)⟩
•Late parametric with top-m re-rank is a search pipeline that starts search with a non-parametric index to retrieve
top-mpassages, denote as Dm, and then on-the-fly embeds them for re-ranking:
fβ(q,D) =⟨Vθ(q), VBoT(D)⟩;fθ(q,Dm) =⟨Vθ(q), Vθ(Dm)⟩
Inourframework,weprimarilyutilizethelateparametrictechniquesprovidedby SiDR.Forin-trainingretrieval,weuselate
parametric with top-20 re-ranking. For checkpoint evaluation and inspection in the ablation study, we use late parametric
with top-100 re-ranking to accelerate results while managing limited resources. In our main experiments, we use full
parametric search.
E. Late Parametric vs. Periodic Re-indexing
Akeydistinctionbetweenourworkandpriorpracticesliesinouruseofthelateparametricmechanismtoavoidre-indexing
during training. In this section, we systematically evaluate these in-training retrieval approaches.
Baseline. Wepresentablationstudiesondifferentin-trainingretrievalapproaches: (i) Open-Rag employsthelateparametric
method as proposed in SiDR, which uses a bag-of-token index for first-stage retrieval and re-ranks the top-20 documents
on-the-flyusingup-to-dateparameters. (ii) Open-Rag (w/o re-rank) employsthebag-of-tokenindexforretrieval,similarto
thelateparametricmethodbutwithoutthere-rankingprocess. Thissetupaimstoassessthecostsassociatedwithre-ranking
duringtraining. (iii) Open-Rag (w/ re-index) involvesperiodicre-indexingusingthemostrecentlybuiltbutoutdatedindex
for retrieval, an in-training retrieval method that commonly used in prior studies. In this setup, we employ DPR MSas the
initial retriever. We avoid using SiDR MS, which has high-dimensional embeddings of 30,522, in stark contrast to DPR’s
768 dimensions. This significant discrepancy prevents our GPU cards from allocating the parametric index for SiDR MS,
although they manage DPR effectively.
14

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
Training. All models undergo the similar training pipeline: they are trained for 80 epochs with the first 40 epochs as
a warm-up and the last 40 conducting in-training retrieval. They differ only in their in-training retrieval strategies: both
Open-Rag andOpen-Rag (w/o re-rank) do not require re-indexing; Open-Rag (w/ re-index) requires rebuilding index at
every15epochs(around5ksteps),arebuildintervalcommonlyusedinpreviousresearch(Xiongetal.,2020),resultingin
a total of three rebuilds.
Results.We present the RAG accuracy on NQ and PubHealth test splits during in-training retrieval, with results reported
everyfourepochs,asdepictedinFigure5. Forthere-rankingsetup,significantimprovementsareobservedinthePubHealth
datawhenre-rankingisemployed,whereastheNQdatasetshowsonlyminorimprovements. Giventhatthecostsassociated
withre-rankingaremanageableinoursetup,wecontinuetoimplementit. Regardingre-indexing,ourfindingsindicatethat
despite requiring significant time and resources, it fails to yield improvements comparable to those of the late parametric
approachandsignificantlylagsbehind. Weattributethistoindexstaleness,wherequeryembeddingsmustoptimizeagainst
outdateddocumentembeddings,renderingthelearningprocesslesseffective. Ontheotherhand,aspresentedinthestudy
by Zhou et al. (2024a), by re-ranking the top-20 retrieved documents, the late parametric method can recover more than
90% of the performance of a full parametric search across different tasks, representing a minor compromise. This also
partially explains why the late parametric approach outperforms periodic re-indexing.
Figure 5. RAG accuracy of different in-training retrieval approaches.
F. Inconsistencies between IR and RAG Scenarios
F.1. Performance Changes in IR Scenarios after Tuning
Table 5.Performance changes before and after tuning the retriever using the Open-Rag approach.
Dataset ( →) NQ TriviaQA
Method ( ↓) Metrics ( →) IR RAG IR RAG
Llama3 8B+SiDR MS 39.1 34.4 56.1 62.0
Llama3 8B+Open-Rag (SiDR MS) 40.8 (+1.7) 39.8 (+5.4) 53.9 (-2.2) 65.8 (+3.8)
Llama3 8B+SiDR NQ 49.5 42.7 – –
Llama3 8B+Open-Rag (SiDR NQ) 47.1 (-2.4) 44.1 (+1.4) – –
We evaluate the performance of our retriever in both IR and RAG scenarios before and after tuning. In IR scenarios, we
measuretop-1retrievalaccuracybycheckingwhetherthetop-1retrieveddocumentcontainstheanswer. InRAGscenarios,
we measure accuracy using a single document in the context window, evaluating whether the generated response contains
the correct answer.
Our results indicate that while Open-Rag tunes the retriever to improve RAG performance, it results in inconsistent
performance on traditional IR performance, with some degradation observed on certain datasets. This highlights a long-
standingissueintheIRevaluationpipeline: adocumentcontainingtheanswerdoesnotnecessarilyimplythatiteffectively
addresses the query, and conversely, a document not containing the answer does not mean it is irrelevant or unhelpful.
Ourconclusionalsoalignswiththefindingsandobservationsofotherresearch. Cuconasuetal.(2024a)findthatincluding
more answer-containing documents in the context negatively impacts RAG performance. Similarly, Nian et al. (2024)
observe that traditional relevance definitions for IR tasks do not enhance RAG response quality. Additional research
15

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
emphasizes the need for further learning to bridge the preference gap (Ke et al., 2024) or re-ranking (Yu et al., 2024) for
off-the-shelf retrievers to improve RAG performance.
F.2. Case Study
Inthissection,wepresentacasestudyusingtheNQdatasetwhereeachqueryhasalistofanswerstrings. Thiscasestudy
is designed to further explore the inconsistency issues inherent in RAG implementations. We specifically examine two
scenarios: (i)caseswheretheretrieveddocumentcontainsthecorrectanswerbutfailstoproducethecorrectRAGoutput,
and(ii)instanceswheretheretrieveddocumentdoesnotdirectlyaddressthequery,yettheRAGmodelmanagestogenerate
the correct answer nonetheless. To enhance our analysis, we also ask GPT-4 to judge whether the documents address the
question, helping readers quickly grasp the key issue.
=================================================== Question ===================================================
Who plays Big Momma in Big Momma’s House?
=================================================== Answers ====================================================
[’Ella Mitchell’, ’Martin Lawrence’]
=================================================== Document ===================================================
Bounce with Me
Jermaine Dupri, Jagged Edge and Da Brat. Brief clips from "Big Momma’s House" are also included. Bounce with Me
"Bounce with Me" is a single by American rapper Lil’ Bow Wow featuring Xscape. It is Lil’ Bow Wow’s debut single
, taken from his debut album "Beware of Dog". The single samples "Love Serenade (Part II)" by Barry White. It
spent nine weeks at number 1 on the U.S. Rap Chart and number 20 on "Billboard" Hot 100. The song was featured
in the film "Big Momma’s House" and the trailer of "Hardball".
================================================ RAG Generation ================================================
[1] Martin Lawrence
================================================ GPT4 Judgement ================================================
No, the document does not address the question. The document talks about the song "Bounce with Me" and its
connection to the movie Big Momma’s House, but it doesn’t mention who plays Big Momma in the movie. The answer
to the question is Martin Lawrence, who plays the character of Big Momma in Big Momma’s House.
=================================================== Question ===================================================
Who sang "Buddy, Can You Spare a Dime?"
=================================================== Answers ====================================================
[’Rudy Vallee’, ’Bing Crosby’]
=================================================== Document ===================================================
Brother, Can You Spare a Dime?
Brother, Can You Spare a Dime? "Brother, Can You Spare a Dime?", also sung as "Buddy, Can You Spare a Dime?", is
one of the best-known American songs of the Great Depression. Written in 1930 by lyricist E. Y. "Yip" Harburg
and composer Jay Gorney, "Brother, Can You Spare a Dime?" was part of the 1932 musical revue "Americana"; the
melody is based on a Russian-Jewish lullaby Gorney’s mother had sung to him as a child. It was considered by
Republicans to be anti-capitalist propaganda, and almost dropped from the show; attempts were made to ban it
from the radio.
================================================ RAG Generation ================================================
The song "Brother, Can You Spare a Dime?" was written by E. Y. "Yip" Harburg and Jay Gorney. It was first
performed by Bing Crosby in 1932.
================================================ GPT4 Judgement ================================================
The document provides information about the song "Brother, Can You Spare a Dime?" (also referred to as "Buddy,
Can You Spare a Dime?") and its origins, but it does not directly answer the question of who sang it. The
document focuses more on the history of the song and its cultural significance.
To answer your question: The song has been performed by several artists, including Bing Crosby, Al Jolson, and
Barbra Streisand, among others.
Figure 6. Example of RAG output correct answers while the retrieved document does not contain the correct answer, nor does GPT-4
consider the document address the question.
In Figure 6, we present examples where RAG outputs the correct answer, even though the retrieved document neither
contains the answer nor is considered to address the question by GPT-4. In both cases, the document fails to provide
the correct answer or relevant clues, yet RAG is still able to generate the correct response. We believe this is a common
phenomenon,asLLMspossessawealthofinternalknowledge,particularlyforpublicknowledgequestions. Ingeneral,an
incorrect or imperfect retrieved document is insufficient to mislead the LLM into producing an incorrect output.
InFigure7,wepresentexampleswhereRAGfailstooutputthecorrectanswer,eventhoughtheretrieveddocumentcontains
the correct answer or GPT-4 considers the document as addressing the question. In the first case, the document does not
address the query, and the LLM tends to extract key phrases, such as the title, as the response, ignoring the query. In the
16

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
=================================================== Question ===================================================
When was The Hitchhiker’s Guide to the Galaxy made?
=================================================== Answers ====================================================
[’1978’, ’2005’]
=================================================== Document ===================================================
The Hitchhiker’s Guide to the Galaxy (radio series)
The original series was the recipient of a number of awards including the Imperial Tobacco Award (1978), The
Sony Award (1979), The Society of Authors/Pye Awards Best Programme for Young People (1980) and the Mark Time
Awards ’Grand Master Award’ (Adams) and ’Hall of Fame’ (1998). It was the only radio show ever to be nominated
for the Hugo science fiction awards, in 1979, in the ’Best Dramatic Presentation’ category. As a result of the
series, Douglas Adams was inducted into the Radio Academy’s Hall of Fame. In November 2003, two years after
Adams’s death and 23 years after the ...
================================================ RAG Generation ================================================
[1] The Hitchhiker’s Guide to the Galaxy (radio series)
================================================ GPT4 Judgement ================================================
No, the document does not directly address the query of when The Hitchhiker’s Guide to the Galaxy was made. It
provides information about the radio series, its awards, and recognition but does not specify the year of its
creation or first broadcast.
The original radio series of The Hitchhiker’s Guide to the Galaxy first aired in 1978.
=================================================== Question ===================================================
Who ordered the Forbidden City to be built?
=================================================== Answers ====================================================
[’Zhu Di’]
=================================================== Document ===================================================
History of the Forbidden City
The site of the Forbidden City was situated on the Imperial city during the Mongol Yuan Dynasty. After the
collapse of the Yuan Dynasty, the Hongwu Emperor of the Ming Dynasty moved the capital from Beijing in the north
to Nanjing in the south, and in 1369 ordered that the Yuan palaces be razed. His son Zhu Di was created Prince
of Yan with his seat in Beijing. In 1402, Zhu Di usurped the throne and became the Yongle Emperor. He made
Beijing a secondary capital of the Ming empire ...
================================================ RAG Generation ================================================
The Yongle Emperor ordered the forbidden city to be built.
================================================ GPT4 Judgement ================================================
Yes, the document addresses the query. It mentions that Zhu Di, who later became the Yongle Emperor, ordered the
construction of the Forbidden City. The passage describes how he moved the capital to Beijing and commissioned
the building of the Forbidden City in 1406.
Figure 7. Example of RAG failing to output the correct answer while the retrieved document contains the correct answer or GPT-4
considers the document as addressing the question.
secondcase,thedocumentcontainsinformationthataddressesthequery,andtheLLMgeneratesthecorrectanswer,butthe
answer’s alias name is not included in the pre-defined answer candidates, leading to a failure in the RAG scenario. These
inconsistencies can be driven by many factors, including the LLM, instruction prompt, evaluation metrics, and relevance.
All of these factors are intertwined, and we believe that end-to-end data-driven learning is more effective than analyzing
their interplay in isolation.
17

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
G. Case Study of RAG Labels
For free-form generation tasks, we assess whether the generation contains any of the given answers. For closed-set
generation tasks, we measure whether the generation contains the label. Below are examples that illustrate how different
generations lead to different RAG labels given the same question and answers.
================================================== Question ====================================================
Who is the sister of for king and country?
=================================================== Answers ====================================================
[’Rebecca St. James’]
================================================= Generation 1 =================================================
Rebecca St. James
### Explanation:
Rebecca St. James is the sister of for king and country. She is a Christian recording artist and speaker. The
Smallbones are the younger brothers of Rebecca St. James, and the brothers-in-law of Jacob Fink, former bassist
for the band Foster the People. They were raised with another sister, Libby, and three other brothers, Ben, Dan,
and Josh. On
=================================================== Output 1 ===================================================
RAG Label: 1
================================================= Generation 2 =================================================
Courtney Helm
### Explanation:
Courtney Helm is the sister of for king and country.
=================================================== Output 2 ===================================================
RAG Label: 0
Figure 8. Examples of RAG labels for free-form generation.
================================================== Question ====================================================
Study: Vaccine for Breast, Ovarian Cancer Has Potential.
=================================================== Answers ====================================================
True
================================================= Generation 1 =================================================
True
### Explanation:
The study is about a vaccine for breast and ovarian cancer. The vaccine is
=================================================== Output 1 ===================================================
RAG Label: 1
================================================= Generation 2 =================================================
False
### Explanation:
The statement is false because the vaccine is not for breast and ovarian cancer.
=================================================== Output 2 ===================================================
RAG Label: 0
Figure 9. Examples of RAG labels for close-set generation.
H. Prompt Formats
WedemonstrateourpromptsfordifferenttaskssuchasOpenQA,fact-checking,andmulti-choicereasoninginFigures10,
11, and 12, respectively.
18

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
=============================================== Prompt (OpenQA) ================================================
### Paragraph:
[1] For King & Country (band)
Their father was a music promoter, and Joel recalls, "going to these rock concerts, sitting on my father\’s
shoulders, plugging my ears...Honestly, in a lot of ways I feel like music chose me and as I grew older, I made
a clear decision to fully lean into it." The Smallbones are the younger brothers of Christian recording artist
and speaker Rebecca St. James, and the brothers-in-law of Jacob Fink, former bassist for the band Foster the
People. They were raised with another sister, Libby, and three other brothers, Ben, Dan, and Josh. On ...
### Instruction:
Who is the sister of for king and country?
### Response:
================================================= Continuation =================================================
Rebecca St. James
================================================== Generation ==================================================
Rebecca St. James
### Explanation:
Rebecca St. James is the sister of for king and country. She is a Christian recording artist and speaker. The
Smallbones are the younger brothers of Rebecca St. James, and the brothers-in-law of Jacob Fink, former bassist
for the band Foster the People. They were raised with another sister, Libby, and three other brothers, Ben, Dan,
and Josh. On
==================================================== Output ====================================================
RAG Score: P(‘Rebecca St. James’|prompt) = 0.595
RAG Label: 1
Figure 10. Example prompt and outcomes of each step for NQ and TQA dataset.
============================================ Prompt (Fact Checking) ============================================
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Paragraph:
[1] Gustav Gaudernack
potential of dendritic cells (DCs) and in 2005, Gaudernack’s group published results from a phase I/II clinical
trial in prostate cancer patients using autologous DCs loaded with tumor mRNA as a vaccine. This study
demonstrated that vaccination with autologous DCs transfected with mRNA derived from three prostate cancer cell
lines was safe and an improved clinical outcome was significantly related to immune responses against the
vaccine. Furthermore, Gaudernack and colleagues initiated a phase I/II clinical trial for treatment of malignant
melanoma with autologous tumor-mRNA transfected DC vaccines. These data clearly demonstrated vaccine-specific
immune responses with a broad specter of ...
### Instruction:
Is the following statement correct or not? Say true if it’s correct; otherwise say false.
### Input:
Study: Vaccine for Breast, Ovarian Cancer Has Potential
### Response:
================================================= Continuation =================================================
True
================================================== Generation ==================================================
true
### Explanation:
The study is about a vaccine for breast and ovarian cancer. The study has ...
==================================================== Output ====================================================
P(‘true’ |prompt) = 0.116
P(‘false’|prompt) = 0.109
RAG Label: 1
Figure 11. Example prompt and outcomes of each step for the Pubhealth dataset.
19

OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning
======================================= Prompt (Multi-choice Reasoning) ========================================
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Paragraph:
[1] Rheumatic fever
Rheumatic fever may occur following an infection of the throat by the bacterium "Streptococcus pyogenes". If the
infection is untreated rheumatic fever can occur in up to three percent of people. The underlying mechanism is
believed to involve the production of antibodies against a person\’s own tissues. Due to their genetics, some
people are more likely to get the disease when exposed to the bacteria than others. Other risk factors include
malnutrition and poverty. Diagnosis of RF is often based on the presence of signs and symptoms in combination
with evidence of a recent streptococcal infection. Treating people who have strep ...
### Instruction:
Given four answer candidates, A, B, C and D, choose the best answer choice.
### Input:
Which factor will most likely cause a person to develop a fever?
A: a leg muscle relaxing after exercise
B: a bacterial population in the bloodstream
C: several viral particles on the skin
D: carbohydrates being digested in the stomach
### Response:
================================================= Continuation =================================================
B
================================================== Generation ==================================================
B
### Explanation:
The bacteria Streptococcus pyogenes is a common cause of throat
==================================================== Output ====================================================
P(‘A’|prompt) = 0.121
P(‘B’|prompt) = 0.309
P(‘C’|prompt) = 0.061
P(‘D’|prompt) = 0.100
RAG Label: 1
Figure 12. Example prompt and outcomes of each step for the ARC-Challenge dataset.
20
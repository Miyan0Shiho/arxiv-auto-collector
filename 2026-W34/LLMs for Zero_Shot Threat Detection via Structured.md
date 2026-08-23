# LLMs for Zero-Shot Threat Detection via Structured Risk Indicators

**Authors**: Abdullah Alghamdi, Siamak Layeghy, Marius Portmann

**Published**: 2026-08-17 12:47:38

**PDF URL**: [https://arxiv.org/pdf/2608.16508v1](https://arxiv.org/pdf/2608.16508v1)

## Abstract
We propose a two-stage large language model (LLM) framework for zero-shot detection of insider threats and advanced persistent threats (APTs) from heterogeneous security logs. The framework models user activity as chronological timelines and incorporates retrieval-augmented generation (RAG) to provide personalised behavioural context from each user's historical activity. Rather than performing end-to-end classification directly from raw logs, it first generates structured, interpretable sets of threat-specific risk indicators, which are then classified jointly across temporal sequences to capture attack patterns spanning multiple windows.The framework is evaluated on two benchmark datasets, CERT r5.2 for insider threat detection and PicoDomain for APT detection, using four combinations of two open-weight LLMs under both retrieval and non-retrieval settings. All configurations outperform the previous state-of-the-art LLM-based framework (GABM), with the best configuration improving the F1-score by 11.40 percentage points on CERT r5.2 and 31.50 percentage points on PicoDomain. Results further show that retrieval mainly benefits weaker LLMs by generating more discriminative risk indicators, whereas stronger models achieve comparable performance without retrieved context. The most effective assignment of LLMs to the two stages depends on the dataset. These findings show that the quality of the generated risk indicators is the main driver of zero-shot cyber threat detection performance.

## Full Text


<!-- PDF content starts -->

LLMs for Zero-Shot Threat Detection via Structured Risk Indicators
Abdullah Alghamdia,∗, Siamak Layeghyaand Marius Portmanna
aUniversity of Queensland, Brisbane, Australia
ARTICLE INFO
Keywords:
Intrusion Detection
Insider Threat Detection
Advanced Persistent Threat
Large Language Models
Zero-Shot Learning
Retrieval-Augmented Generation
Anomaly DetectionABSTRACT
We propose a two-stage large language model (LLM) framework for zero-shot detection of insider
threatsandadvancedpersistentthreats(APTs)fromheterogeneoussecuritylogs.Theframeworkmod-
elsuseractivityaschronologicaltimelinesandincorporatesretrieval-augmentedgeneration(RAG)to
providepersonalisedbehaviouralcontextfromeachuser’shistoricalactivity.Ratherthanperforming
end-to-end classification directly from raw logs, it first generates structured, interpretable sets of
threat-specific risk indicators, which are then classified jointly across temporal sequences to capture
attack patterns spanning multiple windows. The framework is evaluated on two benchmark datasets,
CERTr5.2forinsiderthreatdetectionandPicoDomainforAPTdetection,usingfourcombinationsof
twoopen-weightLLMsunderbothretrievalandnon-retrievalsettings.Allconfigurationsoutperform
the previous state-of-the-art LLM-based framework (GABM), with the best configuration improving
the F1-score by 11.40 percentage points on CERT r5.2 and 31.50 percentage points on PicoDomain.
Results further show that retrieval mainly benefits weaker LLMs by generating more discriminative
risk indicators, whereas stronger models achieve comparable performance without retrieved context.
ThemosteffectiveassignmentofLLMstothetwostagesdependsonthedataset.Thesefindingsshow
that the quality of the generated risk indicators is the main driver of zero-shot cyber threat detection
performance.
1. Introduction
Host-based intrusion detection systems (HIDS) play a
critical role in modern cybersecurity by monitoring activity
within an organisation’s endpoints, including logon events,
file access, device usage, and application-level activities
such as email exchanges and host-recorded network in-
teractions (Bridges, Glass-Vanderlan, Iannacone, Vincent
and Chen, 2019). Unlike network-based intrusion detection
systems (NIDS), which analyse network traffic in transit,
HIDSprovidedirectvisibilityintoendpointactivityandare
particularly effective for detecting insider threats and host-
level stages of advanced persistent threats (APTs), includ-
ing lateral movement, credential misuse, and data exfiltra-
tion(PrabhuandThompson,2022;Georgiadou,Mouzakitis
and Askounis, 2022; Laprade, Bowman and Huang, 2020).
Because adversaries who obtain valid credentials, through
phishing, credential theft, or insider collusion, can closely
mimic legitimate users, endpoint-level detection often con-
stitutes the last line of defence before significant damage
occurs (Glasser and Lindauer, 2013). In practice, malicious
behaviour is rarely confined to a single log source, and
the same adversary may leave traces across endpoint and
network logs. Effective detection therefore benefits from
integrating these heterogeneous observations into a unified
view of user behaviour.
Detecting such threats remains challenging because in-
dividual malicious actions frequently appear benign in iso-
lation. Effective detection therefore requires contextual rea-
soning, including comparing current behaviour against his-
torical baselines, correlating activity across heterogeneous
log sources, and recognising attack stages that unfold over
∗Corresponding author
as.alghamdi@uq.edu.au(A. Alghamdi)
ORCID(s):0009-0008-7235-4488(A. Alghamdi)extended periods. Although both insider threats and APTs
require contextual reasoning, the nature of that context dif-
fers substantially across threat classes. Insider threats typ-
ically emerge as gradual behavioural drift in user activity
logs, such as unusual file access patterns, after-hours lo-
gons, or atypical communication behaviour (Glasser and
Lindauer,2013),whereasAPTactivityisreflectedprimarily
through network-level patterns, including anomalous proto-
col sequences, command-and-control communication, and
lateral movement (Mandiant, 2013).
Traditional machine learning approaches to intrusion
detection, including Random Forest, Support Vector Ma-
chines,gradient-boostedtrees(AlzaabiandMehmood,2024),
and deep learning models such as recurrent and graph
neural networks (Yuan and Wu, 2021), have shown strong
performancebutdependonsubstantiallabelledtrainingdata
andoftenstruggletogeneralisetopreviouslyunseenattacks.
Large language models (LLMs) have recently emerged as
a promising alternative because they can analyse semi-
structured security logs in a zero-shot setting without task-
specific training (Kojima, Gu, Reid, Matsuo and Iwasawa,
2022). However, current LLM-based approaches still per-
form end-to-end classification directly from raw logs, re-
quiringthemodeltosimultaneouslyinterpretheterogeneous
events,reasonaboutuserbehaviour,andmakeathreatdeci-
sion. The current state-of-the-art LLM framework, GABM
(Ferraro, Orlando and Russo, 2025), illustrates this limita-
tion by achieving perfect recall but low precision, resulting
in a false-positive rate that limits practical deployment.
We hypothesise that this limitation arises not from the
reasoning capability of LLMs themselves, but from the
absence of an explicit behavioural abstraction between raw
security logs and the final detection decision. Rather than
asking an LLM to classify heterogeneous logs directly, we
argue that it is more effective to first generate structured,
First Author et al.:Preprint submitted to ElsevierPage 1 of 15
arXiv:2608.16508v1  [cs.CR]  17 Aug 2026

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
interpretable sets of threat-specific risk indicators that sum-
marise behavioural evidence, and then reason over their
temporal evolution to detect multi-step attacks. We further
hypothesisethatgroundingthisprocessineachuser’shistor-
icalbehaviourenablesthegeneratedriskindicatorstobetter
distinguish normal behavioural variation from genuine ma-
licious activity while preserving the zero-shot setting.
Toinvestigatethesehypotheses,weproposeatwo-stage
LLM framework for zero-shot cyber threat detection across
heterogeneoussecuritylogs.Theframeworkcombinesthree
complementary design principles. First, heterogeneous se-
curity logs are organised into chronological user timelines,
providing a unified behavioural view across multiple log
sources. Second, retrieval-augmented generation (RAG)
provides personalised behavioural context by retrieving
semantically similar historical activity from the same user.
Third, LLMs generate structured, interpretable risk indica-
tors that summarise each activity window before a second
stage classifies their temporal evolution to identify attack
patterns spanning multiple windows. To the best of our
knowledge, combining personalised behavioural retrieval
withLLM-generatedstructuredriskindicatorsforzero-shot
intrusion detection has not previously been investigated.
We evaluate the proposed framework on two comple-
mentary benchmark datasets: CERT r5.2 (Glasser and Lin-
dauer,2013),representinginsiderthreatdetectionfromhost
activitylogs,andPicoDomain(Lapradeetal.,2020),repre-
sentingAPTdetectionfromZeeknetworklogs.Acrossfour
combinationsoftwoopen-weightLLMs,evaluatedwithand
without retrieval augmentation, every configuration outper-
forms the previous state-of-the-art LLM-based framework
(GABM). The best configuration improves the F1-score by
11.40percentagepointsonCERTr5.2and31.50percentage
points on PicoDomain, while substantially improving pre-
cision without sacrificing high recall. Beyond these perfor-
mance gains, the evaluation provides new insights into how
retrieval augmentation and model capability interact during
zero-shot threat detection.
The main contributions of this work are as follows.
•Weproposeazero-shotLLMframeworkthatreplaces
direct classification of heterogeneous security logs
with structured risk-indicator generation followed by
temporal classification, providing a unified approach
to insider threat and APT detection.
•We present a comprehensive empirical analysis of
retrievalaugmentationandmodelcapability,showing
that retrieval primarily benefits weaker LLMs by im-
provingthequalityofgeneratedriskindicators,while
the most effective assignment of LLMs to the two
stages depends on the characteristics of the dataset.
The remainder of this paper is organised as follows.
Section2reviewsrelatedworkontraditional,deeplearning,
andLLM-basedapproachestointrusiondetection.Section3
presents the proposed methodology. Section 4 describes
the experimental setup. Section 5 presents the experimentalresultsandablationstudies.Finally,Section6concludesthe
paper and outlines future research directions.
2. Related Work
Recent advances in LLMs have shifted cybersecurity
log analysis from supervised learning towards zero-shot
reasoning over semi-structured security data. This section
reviews existing LLM-based approaches to cyber threat de-
tection and RAG, and positions the proposed work within
this literature.
2.1. Large Language Models for Cyber Threat
Detection
Large language models have recently emerged as a
promising alternative to traditional supervised intrusion
detection by enabling zero-shot reasoning over heteroge-
neous security logs without task-specific training (Kojima
et al., 2022). Existing approaches explore this direction
throughpromptengineering(Li,Zhu,HeandZhang,2025),
parameter-efficientfine-tuning(Kong,Liu,Jin,Geng,Liand
Weng,2025;Song,ZhangandGao,2025b),andmulti-view
behavioural modelling (Song, Zheng, Ma, Liao, Kuang and
Yang, 2025a). A broader survey by Xu, Wang, Li, Wang,
Zhao, Chen, Yu, Liu and Wang (2024) identifies low pre-
cision under high-volume telemetry as one of the principal
challenges facing LLM-based cybersecurity systems.
Despite their methodological differences, most existing
approaches rely on end-to-end classification directly from
raw or lightly processed log data. This requires the LLM
to simultaneously interpret heterogeneous security events,
reason about user behaviour, and make a threat decision
within a single inference step. Our preliminary experiments
(Section 4.5) show that this design leads to unstable pre-
dictions and low precision, consistent with the observations
reported by Xu et al. (2024). A separate line of research
therefore limits the role of LLMs to post-hoc investigation
and explanation rather than detection itself. For example,
eX-NIDS (Houssel, Layeghy, Singh and Portmann, 2026)
employs LLMs to explain alerts generated by an external
intrusiondetectionsystem,whilePletzerandMottok(2026)
useLLMstoanalysecandidateattacksidentifiedbyastatis-
tical anomaly detector. Although these approaches improve
interpretability, the LLM no longer contributes directly to
the detection process.
The closest LLM-based detection frameworks to this
work are GABM (Ferraro et al., 2025) and Audit-LLM
(Song, Ma, Zheng, Liao, Kuang and Yang, 2024), both
of which decompose detection into multiple cooperating
LLM agents. GABM represents the current state of the art
on the benchmark datasets considered in this work, but
achieves high recall at the expense of low precision. Audit-
LLMsimilarlyemploysmultipleagentstocoordinatethreat
analysisbutreportsuser-levelevaluationmetrics,preventing
direct comparison with window-level detection. Although
these frameworks demonstrate the potential of collabora-
tive LLM reasoning, they continue to exchange free-form
First Author et al.:Preprint submitted to ElsevierPage 2 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Sliding-
window
segmentation
Overlapping
fixed-size
windows
(stride 1)
Window t ... t+w
Window t+1 ... t+w+1
Window t+2 ... t+w+2Per-user
grouping
Merges logs
into one
chronological
timeline
Embedding &
retrieval 
Embed window →
cosine similarity over
the user's full timeline
→ top-k similar past
windows + 
5 similarity features
No-RAG Path
skips embedding/similarity → LLM ₁
input: current window only .RAG PathLLM ₁  input (RAG): 
current window + top-k
retrieved past
windows + similarity
features .
Temporal-context
sampling
(Selects windows for inference)DeepSeek-R1-
Distill-32B
as a risk indicator
generatorLLM1: Risk Indicator
GenerationLLM2- T ermporal
Classification
Receives the ordered risk
vectors and labels each
window benign or maliciousMaliciousBenign
Llama 3.1 8B
as a classifier
Transforms each window into a
structured  risk vector . 
All scores normalised to [0,1],
one risk vector per window
1        2, 3, ..... 20Ordered sequence 
- Chunks of 20
-  overlap 5Threat-
specific
encoding
Windows →
text for LLM &
embedder
 OR
Llama 3.1 8B
as a risk indicator
generator
Heterogenous 
Logs
16, 17,     ..... 35
ORDeepSeek-R1-
Distill-32B
as a classifier
Figure 1:Overview of the proposed retrieval-augmented dual-LLM framework. Heterogeneous logs are grouped into per-user
chronological timelines, segmented into overlapping windows, and encoded into a threat-specific textual representation. Under
the retrieval-augmented (RAG) configuration, each window is embedded and compared against the user’s full history to retrieve
the top-𝑘most similar past windows and five similarity features; the No-RAG configuration omits this step. LLM1generates a
threat-specific structured feature vector per window (17insider-threat or20APT indicators), and LLM2classifies the resulting
ordered feature sequences as benign or malicious. Both stages are instantiated with DeepSeek-R1-Distill-Qwen 32B or Llama 3.1
8B, yielding four model combinations under each retrieval setting.
natural-languagereasoningbetweenstagesratherthanstruc-
tured behavioural evidence, and neither explicitly models
userbehaviourrelativetoapersonalisedhistoricalbaseline.
2.2. Retrieval-Augmented Generation for
Cybersecurity
Retrieval-Augmented Generation extends LLM reason-
ingbyincorporatingexternallyretrievedinformation(Lewis,
Perez, Piktus, Petroni, Karpukhin, Goyal, Küttler, Lewis,
Yih,Rocktäscheletal.,2020).Existingcybersecurityappli-
cations primarily retrieve threat intelligence from external
knowledgesources,suchasMITREATT&CK,CVErepos-
itories, and vendor advisories, to improve reasoning about
emerging attacks (Paul, Alemi and Macwan, 2025; Borah,
AlamandRastogi,2025).Inthesesystems,retrievalenriches
the model with domain knowledge that is not contained
within its parameters.
Our use of retrieval addresses a different problem. In-
steadofretrievingexternalthreatintelligence,retrievalpro-
vides personalised behavioural context by comparing each
activitywindowagainstthesameuser’shistoricalbehaviour.
This shifts the role of retrieval from knowledge augmenta-
tion to behavioural grounding, enabling risk indicators to
be generated relative to an individual’s established activity
patternsratherthangenericnotionsofsuspiciousbehaviour.
To the best of our knowledge, combining personalised be-
haviouralretrievalwithstructuredLLM-generatedriskindi-
catorsforzero-shotcyberthreatdetectionhasnotpreviously
been investigated.
3. Methodology
The proposed framework consists of three main com-
ponents: (i) data preparation and temporal windowing,(ii) behaviour-aware retrieval for contextual augmentation,
and (iii) dual-LLM inference for feature generation and
classification over ordered windows. Given the scale of
the windowed datasets, a deterministic temporal-context
sampling step (Section 3.2.5) selects the windows that
undergo LLM inference. We evaluate the framework under
two retrieval configurations, RAG and No-RAG, to isolate
the contribution of retrieval across both models. Figure 1
providesanoverviewofthepipeline.Thesamearchitectural
backboneisappliedtobothdatasetsconsideredinthiswork,
with the windowing, window representation, and LLM1
feature set instantiated according to the threat type.
3.1. Data Preparation and Temporal Windowing
3.1.1. Per-user log grouping.
Wegroupbyuserbecausemaliciousbehaviourtypically
spans several log sources rather than appearing within any
single one: an insider’s file access, logon timing, and email
activity(oranAPTactor’sbeaconingandlateralmovement)
form a recognisable pattern only when a user’s activity is
viewed as a whole. Grouping all of a user’s records into
one chronological timeline therefore gives each inference
windowthefullcontextofthatuser’sbehaviour,ratherthan
a fragment confined to a single log type or connection.
Concretely,logsaregroupedbyuserandsortedchrono-
logically. For PicoDomain, each log entry is associated
with a user through the host-to-user mapping provided in
the dataset ground truth (e.g.BDUCK,JDOE). For CERT r5.2,
the five log sources used in this work (logon, device, file,
email,HTTP)aremergedintoasinglechronologicalactivity
sequence per user.
First Author et al.:Preprint submitted to ElsevierPage 3 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Table 1
Dataset overview and sampling summary. For CERT r5.2,
windowing and all subsequent stages are applied to the 99
malicious users’ logs;Raw entriesreports the full corpus.
Benign windows are therefore drawn from the benign activity
within these users’ own timelines.
Dataset Raw entries Users (Total) Mal. users Windows Sampled Mal. / Ben.
PicoDomain 537,840 7 5 460,033 2,741 1,000 / 1,741
CERT r5.2∼79.9M 2,000 99 3,400,220 9,571 4,410 / 5,161
3.1.2. Sliding-window construction.
Each user’s chronologically sorted log sequence is seg-
mented into sliding windows of fixed size𝑤with stride 1,
where𝑤=10for CERT r5.2 and𝑤=5for PicoDomain.
A window is labelled malicious if any of its constituent
log entries carries a malicious label; otherwise it is la-
belledbenign.Understride-1windowing,asinglemalicious
log entry propagates across up to𝑤consecutive windows,
all receiving a malicious label; this any-positive labelling
rule is applied consistently across all experimental condi-
tions, ensuring that observed performance differences are
attributable to pipeline configuration rather than labelling
variation. The larger window for CERT r5.2 reflects the
gradual,multi-stepnatureofinsiderthreatbehaviour,where
ameaningfulactivitysequencespansmultiplelogtypes;the
smallerwindowforPicoDomainissufficientgiventhatAPT
activity manifests in short, concentrated bursts.
Each window is encoded for two purposes. For the
embedding model (Section 3.2), the window is represented
as a compact protocol-and-destination token sequence such
ashttp:internal→ssl:c2→dns:external→kerberos:
internal→conn:external, where destinations are cate-
gorisedasinternal(10.99.99.0/24),c2(knownC2addresses
3.3.3.5 and 1.1.1.11), orexternal.
LLM1(Section 3.3) receives a richer representation:
the full Zeek log records for the window across all log
types,accompaniedbythisprotocol-destinationsequenceas
acompactstructuralsummary.ForCERTr5.2,eacheventin
thewindowisrenderedasastructuredeventobject:aJSON
recordretainingthethreat-relevantfieldsforitsactivitytype
(actionandPCforlogon,recipientsandattachmentmetadata
for email, filename and removable-media flag for file, URL
for HTTP, and file-tree metadata for device), with events
preservedintemporalorder.Thisrepresentationservesboth
the embedding model and LLM1. After windowing, Pi-
coDomain yields 460,033 windows from 537,840 raw log
entries,whileCERTr5.2yields3,400,220windowsfromthe
logs of its 99 malicious users, drawn from a full corpus of
approximately 79.9 million entries (Table 1).
3.2. Behaviour-Aware Retrieval for Contextual
Augmentation
Weincorporatearetrieval-augmentedgeneration(RAG)
mechanism to provide each window with behavioural con-
text drawn from the same user’s history. For each window,
the most similar past windows are retrieved and attached ascontextforLLM1,enablingcomparisonofcurrentbehaviour
against a personalised historical baseline.
3.2.1. Window embedding.
Each window’s textual representation is encoded into
a 384-dimensional dense vector usingall-MiniLM-L6-v2
(ReimersandGurevych,2019),alightweightsentencetrans-
former model. Embeddings are computed once per window
and stored for reuse across all downstream steps.
3.2.2. Similarity computation and features.
Foreachwindow𝑖,cosinesimilarityiscomputedagainst
all prior windows within a lookback horizon of 500 win-
dows from the same user. Five deterministic similarity
features are derived from the resulting similarity distribu-
tion:sim_max(maximum similarity to any past window),
sim_topk_avg(mean similarity over the top five matches),
sim_topk_std(standarddeviationoverthetopfivematches),
sim_global_mean(mean similarity over all windows in
the lookback horizon), andsim_gap(difference between
sim_maxandsim_topk_avg).Thesefeaturesprovideaquan-
titative signal of how anomalous the current window is
relativetotheuser’srecenthistory,andarepassedtoLLM1
alongside the log content.
3.2.3. Retrieval.
The top𝑘=3most similar past windows, by cosine
similarity over the embeddings, are retrieved from the pre-
ceding500-windowlookbackhorizonwithinthesameuser’s
history and attached as context for LLM1. For CERT r5.2,
eachretrievedwindowcontainsthestructuredeventfieldsof
its constituent log entries. For PicoDomain, each retrieved
window contains the full Zeek log entries across all log
types, with unique connection identifiers and sensor names
removed. The first window of each user’s sequence has no
retrievable history and is processed with an empty context.
Labels are removed from all retrieved windows before they
are presented to LLM1, preventing label leakage.
3.2.4. No-RAG configuration.
To isolate the contribution of retrieval, we define a No-
RAGconfigurationinwhichLLM1receivesonlythecurrent
window’s log content, with no retrieved past windows and
no deterministic similarity features. The LLM1prompt is
trimmed accordingly: the instruction to compare the cur-
rent window against past windows is removed, and the
model must generate risk indicators from the current win-
dow alone. All other pipeline components remain identical;
the No-RAG configuration differs from the RAG configu-
ration solely in the removal of the retrieval mechanism: the
retrieved past windows and the similarity features derived
from them.
3.2.5. Temporal-context sampling.
Giventhelargescaleofthewindoweddatasets,applying
LLM inference to all windows is computationally impracti-
cal. We therefore adopt a sampling strategy that preserves
First Author et al.:Preprint submitted to ElsevierPage 4 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
malicious activity while retaining sufficient benign context
for temporal analysis.
Because embedding, similarity computation, and re-
trieval(Section3.2)areperformedovereachuser’scomplete
windowed timeline, sampling determines only which win-
dowsundergoLLMinference;retrievedcontextistherefore
drawn from the user’s full history rather than from the
sampled subset.
For each user, contiguous sequences of malicious win-
dows (i.e.malicious bursts) are identified. A symmetric
context of benign windows is retained before and after each
burst, preserving the temporal transition between normal
and anomalous behaviour. For PicoDomain, all malicious
bursts are retained with a context of±10benign windows,
yielding2,741sampledwindows.ForCERTr5.2,thelargest
malicious burst per user is selected with a context of±30
benign windows, yielding 9,571 sampled windows across
99 users; selecting the largest burst bounds the sampled
set while capturing each user’s principal malicious episode
together with its surrounding normal behaviour. The larger
context for CERT r5.2 reflects the gradual nature of insider
threat behaviour, where the transition between normal and
anomalous activity spans many windows; the smaller con-
text for PicoDomain is sufficient given the concentrated,
abrupt nature of APT beaconing bursts.
The sampling procedure is deterministic and depends
onlyonwindowlabelsandsequentialpositions.Thisensures
that the exact same subset of windows is evaluated across
allexperimentalconditions(includingallsecond-stageclas-
sifier variants) so that observed performance differences
are attributable solely to the pipeline configuration rather
than to variation in the evaluation data. The resulting class
distributions are summarised in Table 1.
Because sampling concentrates on malicious bursts and
theirimmediatecontext,thereportedprecisionandF1reflect
this evaluation distribution rather than the full-corpus base
rate.Asthesampledsubsetisidenticalacrossallconditions,
thisdoesnotaffecttherelativecomparisonsthatarethefocus
of this work.
3.3. Feature Generation (LLM1)
The proposed framework employs a two-stage infer-
ence process in a zero-shot setting. Its central component is
LLM1,athreat-awarefeaturegeneratorthattransformseach
raw log window into a structured vector of normalised risk
indicators, replacing the hand-engineered feature extractors
usedinconventionalpipelines.Thesecondstage,LLM2,isa
downstreamsequenceclassifierthatreasonsovertheordered
featurevectorsLLM1produces.Theframeworkinstantiates
both stages using two open-weight LLMs of differing capa-
bility (DeepSeek-R1-Distill-Qwen 32B and Llama 3.1 8B),
both quantised to 4-bit precision (Q4_K_M) and deployed
viallama.cpp.
LLM1transformseachsampledwindowintoastructured
feature representation: a set of normalised risk indicators
tailored to the threat type of each dataset. Its core input
is the current window’s log content (with labels removed).In the retrieval-augmented configuration, LLM1addition-
ally receives up to three retrieved past windows from the
sameusertogetherwiththedeterministicsimilarityfeatures
defined in Section 3.2, which quantify how anomalous the
current window is relative to the user’s recent history; the
contribution of this retrieved context is analysed in Sec-
tion5.1(Table4).Inthisconfiguration,thepromptinstructs
the model to act as a cybersecurity analyst and to compare
the current window against the provided past windows,
identifying deviations in destinations, protocol or activity
sequences,timingpatterns,andanyescalationofsuspicious
behaviour.IntheNo-RAGconfiguration(Section3.2.4),the
current window’s log content is the model’s sole input, and
the prompt omits the comparison instruction, requiring the
risk indicators to be generated from the window alone. The
model returns a JSON object containing normalised risk
scores, each bounded to [0,1]. The discriminative quality
of the generated features is assessed using SHAP analysis,
reportedinSection5.2.Thefullsetofriskindicatorsislisted
inTables3and2,andtheLLM1andLLM2systemprompts
are shown in Figures 2 and 3.
Dataset-specific feature sets.The feature sets are tai-
lored to the threat type of each dataset. For each dataset,
the feature set is constructed to satisfy two constraints:
(i)coverage:everybehaviouraldimensionrepresentedinthe
dataset’s documented threat scenarios must be observable
through at least one feature; and (ii)extractability: each
feature must be derivable from the log types the dataset
actually provides. Scenario documentation is used only to
define the feature schema (analogous to the threat-model
knowledge that informs detection rules in operational de-
ployments); no ground-truth labels inform feature design,
and the same fixed feature set is applied uniformly across
all users and experimental configurations. For CERT r5.2,
the 17 features capture behavioural indicators of insider
threat across the scenarios documented in the dataset (data
leakage, intellectual property theft, and IT sabotage), or-
ganised into seven behavioural categories: deviation from
the user’s behavioural baseline, temporal anomalies, data
access,dataexfiltrationacrossremovablemedia,email,and
filechannels,privilegemisuse,networkactivity,andasingle
aggregaterisksummary.Eachfeatureisextractablefromthe
five CERT log types used in this work (logon, device, file,
email,andHTTP),aligningthefeaturesetwithbothMITRE
ATT&CK tactics (MITRE Corporation, 2026) and User
and Entity Behaviour Analytics (UEBA) indicators. SHAP
analysis (Section 5.2) shows that both models rank file-
access and exfiltration indicators most highly, though they
differinthesingledominantfeature;therelativeimportance
reflects the attack characteristics of the dataset, with USB-
basedexfiltrationscenarioselevatingusb_usage_riskforthe
stronger model.
For PicoDomain, the 20 features target network-level
indicators of APT activity aligned with the stages of the
First Author et al.:Preprint submitted to ElsevierPage 5 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
LLM1 System Prompt — CERT r5.2
You are a cybersecurity analyst specialising in insider threat detection.
You will be given a JSON object containing:
- current_window: the user activity window to analyse (10 events) with similarity features
- past_similar_windows: the most similar past windows from the same user
Your job is to compare the current window against past windows and detect:
- New destinations that did not appear in past windows
- Changes in activity sequences
- Unusual timing or behaviour patterns
- Any escalation or suspicious activity
Input: {current_window, past_similar_windows}
Return ONLY a JSON object with 17 normalised risk scores [0,1].
No explanation. No markdown.
LLM1 System Prompt — PicoDomain
You are a cybersecurity analyst specialising in APT and network intrusion detection.
You will be given a JSON object containing:
- current_window: a sequence of 5 Zeek network log entries from a single host
- past_similar_windows: the most similar past windows from the same host
Your job is to compare the current window against past windows and detect:
- New destinations that did not appear in past windows
- Changes in protocol sequences
- Authentication anomalies (Kerberos, NTLM)
- Lateral movement patterns
- Data exfiltration indicators
- Any escalation or suspicious activity
Key context: Internal network 10.99.99.0/24;
Known C2 servers: 3.3.3.5 (icecream.inet), 1.1.1.11 (kali.inet)
Input: {current_window, past_similar_windows}
Return ONLY a JSON object with 20 normalised risk scores [0,1].
No explanation. No markdown.
Figure 2:LLM1system prompts: CERT r5.2 (top) and Pi-
coDomain (bottom).
APT kill chain, organised into seven categories: sequence-
level anomalies in the host’s connection patterns, authen-
tication abuse (Kerberos and NTLM), lateral movement
between internal hosts, protocol-level irregularities, data
exfiltration,attack-stageprogression,andasingleaggregate
risksummary.EachfeatureisextractablefromtheZeeklog
types in the dataset (CONN, DNS, DCE_RPC, SSL, and
Kerberos). Unlike CERT r5.2, individual-window feature
signal on PicoDomain is inherently weak (Section 5.2), a
consequenceofthestealthynatureofAPTbeaconingrather
than feature design; the feature set nonetheless provides a
consistent structured representation for LLM2to classify.
This dataset-specific design allows LLM1to focus on the
most discriminative signals for each threat type rather than
relying on a generic feature set.
LLM1receives a JSON object containing the current
activity window and its retrieved similar past windows, and
returnsafixed-lengthvectorofriskscorescorrespondingto
the features in Tables 3 and 2.
3.4. Sequence Classification (LLM2)
LLM2receives the unlabelled feature representations
produced by LLM1for a given user, arranged in chronolog-
ical order, and produces a binary classification (benign or
malicious) for each window. In both configurations, LLM2
receives only the risk-indicator vectors produced by LLM1,
never raw log content; the deterministic similarity features
of Section 3.2 serve solely as input context for feature
generation and are not propagated downstream. LLM2is
identical across the RAG and No-RAG configurations, so
performance differences between the two settings are at-
tributable solely to the features LLM1produces.
Chunkedprocessing.Duetocontext-windowconstraints,
the ordered feature sequence for each user is divided into
chunksof20windowswithanoverlapof5windowsbetween
LLM2 System Prompt — CERT r5.2
You are an expert cybersecurity AI specialising in insider threat detection
in enterprise user activity logs. You will receive a sequence of activity
windows for a SINGLE user in chronological order. Each window contains
17 risk feature scores [0,1], higher = more suspicious.
Task: classify EVERY window as 0 (Benign) or 1 (Malicious).
PRIMARY indicators: behavior_deviation_score, email_exfiltration_risk,
file_exfiltration_risk, usb_usage_risk
BASELINE: normal = overall_risk_score 0.0-0.2; sustained >0.5 across
3+ consecutive windows = confirmed insider threat.
RULES:
1. Analyse the FULL sequence - not each window alone
2. Malicious windows appear in BURSTS of consecutive elevated windows
3. A low-risk window CAN be malicious if surrounded by high-risk windows
4. Attack combinations:
- email + file_exfiltration + external_transfer -> data exfiltration
- usb + sensitive_file + volume -> USB theft
- privilege_escalation + role_mismatch -> privilege abuse
5. Watch for LOW AND SLOW attacks: sustained mid-tier risk (0.4-0.5)
OUTPUT: JSON array only. 0=Benign, 1=Malicious.
No explanation. No text. No markdown. Example: [0, 0, 1, 1, 0]
LLM2 System Prompt — PicoDomain
You are an expert cybersecurity AI specialising in APT detection
in enterprise network traffic. You will receive a sequence of activity
windows for a SINGLE host in chronological order. Each window contains
20 network risk feature scores [0,1], higher = more suspicious.
Task: classify EVERY window as 0 (Benign) or 1 (Malicious).
PRIMARY indicators: sequence_anomaly_score, lateral_movement_risk,
external_connection_risk, data_exfiltration_indicator, attack_stage_indicator
BASELINE: normal = overall_sequence_risk 0.0-0.2; sustained >0.5 across
3+ consecutive windows = confirmed attack.
RULES:
1. Analyse the FULL sequence - not each window alone
2. Malicious windows appear in BURSTS of consecutive elevated windows
3. Attack combinations:
- lateral_movement + host_traversal + kerberos_anomaly -> lateral movement
- data_exfiltration + large_transfer + external_connection -> exfiltration
- reconnaissance + sequence_anomaly -> reconnaissance
4. Watch for LOW AND SLOW attacks: sustained mid-tier risk (0.4-0.5)
OUTPUT: JSON array only. 0=Benign, 1=Malicious.
No explanation. No text. No markdown. Example: [0, 0, 1, 1, 0]Figure 3:LLM2system prompts: CERT r5.2 (top) and Pi-
coDomain (bottom).
consecutive chunks. The overlap ensures that burst bound-
ariesarenotmissedatchunkedges.Whenawindowappears
in multiple chunks, the later chunk’s prediction is retained,
since there the window has its subsequent context in view.
Burst-aware reasoning.The prompt instructs LLM2to
reasonoverthefulltemporalsequenceratherthanevaluating
each window in isolation. The model identifies regions of
consecutive windows with elevated risk scores, recognises
coordinated attack patterns such as data exfiltration, privi-
lege abuse, and suspicious access based on related feature
combinations (pattern descriptions instantiated per dataset,
Figure 3), and uses surrounding context to classify ambigu-
ous windows that may appear benign in isolation but fall
within a broader malicious burst. This prompt design is
motivatedbythetendencyofmaliciousactivityinbothAPT
andinsiderthreatscenariostomanifestasburstsofsustained
elevated risk rather than isolated anomalous windows.
LLM2receives a chronological sequence of risk-score
vectors produced by LLM1for a single user and assigns a
binary label to every window (0=benign,1=malicious).
4. Experimental Setup
4.1. Datasets
We evaluate the proposed framework on two publicly
availablecybersecuritydatasets:PicoDomainandCERTr5.2.
PicoDomain.The PicoDomain (Laprade et al., 2020) is
a high-fidelity Zeek network log dataset simulating an en-
terprise environment under an advanced persistent threat
First Author et al.:Preprint submitted to ElsevierPage 6 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Table 2
LLM1risk indicators for PicoDomain, grouped by the category of behaviour they characterise. All scores are normalised to[0,1].
Feature Description
Sequence
sequence_anomaly_score Overall degree to which the window’s protocol and connection sequence departs from the user’s established patterns.
rare_sequence_indicator Presence of protocol or connection sequences rarely or never seen in the host’s prior history, flagging novel network
behaviour.
sequence_pattern_deviation Extent of deviation in the ordering of events relative to the host’s normal sequential behaviour.
Authentication
kerberos_anomaly_score Anomalous Kerberos authentication activity, such as unusual ticket-granting requests. Kerberos abuse is a common
signature of credential-based attacks in Windows domains.
ntlm_anomaly_score Suspicious NTLM authentication activity inconsistent with the host’s normal authentication profile.
auth_failure_pattern Patterns of repeated or distributed authentication failures, indicative of credential probing or brute-force attempts.
ticket_misuse_indicator Evidence of Kerberos ticket abuse or reuse, a signature of attacks such as pass-the-ticket.
credential_access_pattern Suspicious patterns of credential access or harvesting, capturing attempts to obtain account secrets.
Lateral movement
lateral_movement_risk Risk of movement between internal hosts, a core stage of APT campaigns as adversaries pivot toward their objective.
host_traversal_pattern Patterns of connections traversing multiple internal hosts beyond the host’s normal communication footprint.
multi_hop_connection_indicator Presence of multi-hop connection chains characteristic of pivoting through a network.
Protocol
protocol_transition_anomaly Unusual transitions between network protocols within a connection sequence, which can indicate covert channels.
unusual_protocol_sequence Protocol orderings within the window that diverge from expected service behaviour.
service_access_anomaly Anomalous access to network services relative to the host’s normal service-usage profile, capturing service enumeration.
Exfiltration
data_exfiltration_indicator Signs that data is being transferred outside the network boundary, the defining objective of an APT campaign.
large_transfer_anomaly Unusually large data transfers inconsistent with the host’s normal traffic volume.
external_connection_risk Risk associated with connections to external or command-and-control destinations.
Attack progression
attack_stage_indicator Evidence positioning the window within a recognised stage of an APT campaign, from reconnaissance through to
exfiltration.
reconnaissance_pattern Patterns of port scanning or service discovery, characteristic of the reconnaissance stage of an attack.
Aggregate
overall_sequence_risk Holistic risk assessment for the window, integrating all preceding indicators into a single summary signal.
(APT) campaign. The dataset covers a three-day period and
contains 537,840 log rows across multiple Zeek log types,
including CONN, DNS, DCE_RPC, SSL, and Kerberos.
Logs are associated with users based on the host-to-user
mapping provided in the dataset ground truth. Following
preprocessing, the dataset contains 7 users, of which 2 ex-
hibitexclusivelybenignactivityandareexcluded,leaving5
users (all associated with malicious activity) for evaluation.
The ground truth contains 80 red team events, of which
79 correspond to malicious actions; the remaining event
marks the end of the attack campaign and is not considered
a malicious activity in this work. Labels are derived by
matching log entries against the 79 malicious events using
atemporalwindowof±5secondsandIPaddressmatching.
CERTr5.2.TheCERTInsiderThreatDatasetr5.2(Glasser
and Lindauer, 2013) is a synthetic dataset capturing user
activity logs for a simulated organisation over an 18-month
period. It contains 2,000 employees, of whom 99 are asso-
ciated with labelled malicious activity across four insider
threat scenarios (data leakage, two variants of intellectual
propertytheft,andITsabotage).Thedatasetprovidesseven
logtypes,ofwhichfiveareusedinthiswork:logon,device,
file,email,andHTTP.Theremainingtwo(decoyfileaccess
andpsychometricdata)areexcludedastheydonotrepresent
genuine user activity. Labels are derived by matching event
identifiers against the ground truth from scenario folders.4.2. Implementation Details
DeepSeek-R1-Distill-Qwen-32Bisrunlocallyviallama.
cppon NVIDIA A100 and L40S GPU nodes of the Bunya
HPC cluster (University of Queensland). Inference is per-
formed with temperature set to 0. For the cross-model
analysis (Section 5.1), Llama 3.1 8B is run under the
sameconfiguration.Allmodelsarequantisedto4-bitpreci-
sion (Q4_K_M). Sentence embeddings are computed using
all-MiniLM-L6-v2on the same GPU nodes. For the RAG
setting, retrieval is performed with𝑘=3nearest windows
andalookbackof500priorwindows,usingcosinesimilarity
overthecomputedembeddings.Alldataset-specificchoices
(thefeatureschema,thewindowsize,andthebenign-context
span)aresetapriorifromthedocumentedcharacteristicsof
each threat type rather than tuned on the evaluation labels;
theframeworkthereforeremainszero-shot,usingnolabelled
training data at any stage.
4.3. Baselines
We compare the proposed framework against the Gen-
erativeAgent-BasedModelling(GABM)methodofFerraro
etal.(2025),arecentLLM-basedmulti-agentframeworkfor
insider threat detection evaluated on the same two bench-
markdatasetsusedinthiswork.GABMemploysspecialised
LLM agents for each log type, whose analyses are syn-
thesised by a supervisor agent for final classification using
LLaMA-3.1-8B. It classifies per activity identifier rather
First Author et al.:Preprint submitted to ElsevierPage 7 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Table 3
LLM1risk indicators for CERT r5.2, grouped by the category of behaviour they characterise. All scores are normalised to[0,1].
Feature Description
Behavioural baseline
behavior_deviation_score Overall degree to which the window departs from the user’s established activity profile. Insider threats are fundamentally
deviations from a user’s own behavioural baseline.
Temporal
after_hours_activity Degree to which actions fall outside the user’s normal working hours. Insider data theft is frequently timed to off-hours
to reduce the chance of observation.
time_anomaly_score Irregularity of event timing relative to the user’s historical rhythm, capturing atypical intervals and weekend activity
not explained by working hours alone.
activity_burstiness Concentration of many actions into a short interval. Bulk data collection or staging produces dense activity bursts
atypical of routine work.
Data access
file_access_volume Volume of file-access operations relative to the user’s norm. An elevated access count is a common precursor to
large-scale exfiltration.
sensitive_file_access Degree of interaction with sensitive, restricted, or high-value files, flagging direct contact with the assets most likely
to be targeted.
file_access_pattern_change Shift in the type, location, or breadth of files accessed relative to prior behaviour, indicating a move from role-consistent
to anomalous access.
Data exfiltration
usb_usage_risk Risk of removable-media operations, scaled by the frequency of device use and the sensitivity of files transferred.
Removable storage is the principal physical exfiltration channel in CERT r5.2.
email_exfiltration_risk Likelihood that email activity constitutes data leakage, such as large or unusual attachments sent to external addresses.
file_exfiltration_risk Likelihood that file operations represent exfiltration, such as copying sensitive files toward staging or external
destinations.
external_transfer_risk Risk of data crossing the organisational boundary to external recipients or systems, aggregating signals across email,
web, and file channels.
Privilege misuse
role_mismatch_score Extent to which observed actions diverge from those typical of the user’s job role. Insider misuse often surfaces as
access unrelated to legitimate responsibilities.
policy_violation Presence of actions that contravene organisational security policy, a direct indicator of intentional misconduct.
privilege_escalation_indicator Evidence of attempts to obtain or exercise access beyond the user’s authorised permissions.
Network
unusual_web_activity Anomaly in web-browsing behaviour relative to the user’s norm, capturing reconnaissance or unsanctioned upload
activity.
external_domain_access Degree of connection to external or previously unseen domains, flagging communication with untrusted destinations.
Aggregate
overall_risk_score Holistic risk assessment for the window, integrating all preceding indicators into a single summary signal.
than per fixed-length window, and does not specify the
proceduremappinggroundtruthtolabelledinstancesorhow
itsbenignclassissampled(constrainedto5,000of537,840
entries on PicoDomain). GABM achieves perfect recall on
both datasets but suffers from low precision, indicating a
high false positive rate. Reported results from the original
publication are used for comparison. GABM uses LLaMA-
3.1-8Basitsbasemodel,whereasourprimaryconfiguration
uses the larger DeepSeek-R1-Distill-Qwen 32B. To isolate
thecontributionofthearchitecturefromthatofmodelscale,
we hold the base model fixed at Llama 3.1 8B, the same
model GABM uses. Applied directly to logs without the
dual-LLM architecture, Llama 3.1 8B performs well below
the framework (Section 4.5); embedded in the proposed
framework, the same model performs substantially better
(Section5.1),indicatingthatthedual-LLMarchitecture,not
model capacity, drives the difference.
Songetal.(2024)isarelatedmulti-agentLLMapproach
for insider threat detection but is not used as a direct nu-
mericalbaseline;webenchmarkagainstGABMasthemost
directly comparable LLM-based method evaluated on these
two datasets, and discuss Audit-LLM as related work.4.4. Evaluation Metrics
Performance is evaluated using precision, recall, and
F1-score. Metrics are computed by aggregating predictions
acrossalluserswithineachdataset(5usersforPicoDomain
and 99 users for CERT r5.2), producing a single overall
evaluation per dataset. The F1-score provides a balanced
measure of precision and recall, which is the primary basis
for comparison in this work.
4.5. Preliminary Experiment: Direct LLM
Classification
The state-of-the-art baseline (GABM) decomposes de-
tection across multiple LLM agents, separating per-log-
typereasoningfromasupervisorclassificationstage.Before
adoptingadecompositionofourown,wetestwhetheritcan
be avoided: whether a single LLM can carry reasoning and
classificationinonepass.Themodelreceivesindividualraw
logrecords,withalllabelandground-truthannotationfields
removed, and must reason over each event and emit aBE-
NIGN/MALICIOUSdecisionwithinthesameprompt,without
windowing,retrieval,oradedicatedfeature-generationstage
(LLM1). We sweep this design on both datasets over the
numberoffew-shotexamplesperclass𝑁∈ {0,1,2,3}and
First Author et al.:Preprint submitted to ElsevierPage 8 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
0 1 2 30.050.100.150.20F1 score
(a) PicoDomain, Llama-3.1-8B
0 1 2 30.050.100.150.20
(b) PicoDomain, DeepSeek-R1-Distill-Qwen-32B
0 1 2 3
Few-shot examples per class (N)0.200.250.300.350.40F1 score
(c) CERT r5.2, Llama-3.1-8B
0 1 2 3
Few-shot examples per class (N)0.200.250.300.350.40
(d) CERT r5.2, DeepSeek-R1-Distill-Qwen-32B
b=10 b=20 b=30 b=50
Figure 4:Direct LLM classification (reasoning and classification in a single pass, without windowing, retrieval, or a dedicated
feature-generation stage): F1 against the number of few-shot examples per class𝑁, by batch size𝑏, for (a, b) PicoDomain and
(c, d) CERT r5.2, using Llama-3.1-8B and DeepSeek-R1-Distill-Qwen-32B. All configurations remain well below the proposed
framework.
the batch size𝑏∈ {10,20,30,50}, using both DeepSeek-
R1-Distill-Qwen 32B and Llama 3.1 8B. Results are shown
in Figure 4.
Additional supervision yields only small and inconsis-
tent gains. Llama 3.1 8B improves up to two-shot and then
declines, while DeepSeek improves slowly with shot count
on CERT r5.2. No combination of shot count and batch
size lifts direct classification past a low ceiling. At matched
zero-shot supervision, the setting in which the framework
itselfoperates,thebestdirectconfigurationreachesanF1of
only 30.98% (DeepSeek-R1-Distill-Qwen 32B) and 23.10%
(Llama 3.1 8B) on CERT r5.2, and 10.60% and 10.47% on
PicoDomain, all far below the framework’s best of 64.14%
and50.87%(Table4).Becausethesamemodelsareusedin
bothsettings,thisgapreflectsthearchitecturaldesignrather
than model capacity or supervision level.
Across both models and datasets, a single LLM cannot
performthetaskend-to-end,andadditionalpromptingdoes
not close the gap. We attribute this to the demands of a
single prompt, which must parse heterogeneous log fields,
reason about threat behaviour, and commit to a decision
at once. This motivates the structured, multi-stage design
of the proposed framework, in which a feature-generation
stage (LLM1) produces an intermediate representation that
aseparatesequenceclassifier(LLM2)consumes;thecontri-
bution of retrieval and of the model assigned to each stage
isanalysedinSection5.1,andtheremainingdesignchoices
in the ablation study (Section 5.4).5. Results and Discussion
5.1. Classification Performance
To examine how model capability and retrieval interact
across the two stages, we evaluate four model combina-
tions: (i) DeepSeek as both feature generator and classifier;
(ii) Llama 3.1 8B as both; (iii) Llama 3.1 8B as feature
generator with DeepSeek as classifier; and (iv) DeepSeek
as feature generator with Llama 3.1 8B as classifier. Each
configurationisevaluatedunderboththeRAGandNo-RAG
settingsdefinedabove,yieldingeightconditionsperdataset
(16acrossthetwodatasets).Thiscross-modelanalysisforms
a central investigation of the framework (Section 5.1).
The overall performance of the framework across the
four model combinations is presented in Table 4. Two pat-
terns are consistent across datasets. First, the choice of
feature-generating model (LLM1) has a larger effect on
performancethanthechoiceofclassifier(LLM2):swapping
LLM2while holding LLM1fixed moves F1 by less than
swappingLLM1.Second,thevalueofretrievalisconditional
on feature-generator capacity. When LLM1is the weaker
model (Llama 3.1 8B), retrieval supplies the comparative
context it cannot generate internally, lifting F1 by up to
13.4 percentage points on PicoDomain; when LLM1is the
stronger model (DeepSeek-R1 32B), retrieval is neutral or
slightly negative.
Beyond F1, retrieval shifts the operating point. On
CERT r5.2 under the DeepSeek/DeepSeek combination,
the No-RAG configuration attains its higher F1 through
a recall-biased regime: it produces 4,490 false positives
versus 3,447 with retrieval, and correctly identifies only
First Author et al.:Preprint submitted to ElsevierPage 9 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
671 of 5,161 benign windows (13.0% specificity) versus
1,714(33.2%)withretrieval.Retrievedper-userbehavioural
context thus calibrates LLM1risk scores against historical
baselines, yielding a more selective detector. The false-
positivedisciplineiswhatdistinguishestheframeworkfrom
the perfect-recall, low-precision GABM baseline.
Table5comparesthehighest-F1configurationperdataset
against the GABM baseline. On PicoDomain, the frame-
work reaches an F1-score of 50.87% (DeepSeek as LLM1,
Llama3.18BasLLM2,No-RAG),animprovementof31.50
percentage points over the GABM baseline (F1=19.37%).
While the baseline attains perfect recall, it does so at the
cost of extremely low precision (10.72%). The framework
provides a substantially more balanced trade-off, improving
precision by 25.85 percentage points while maintaining a
recall of 83.50%.
On CERT r5.2, the framework attains an F1-score of
64.14%(Llama3.18BasLLM1,DeepSeekasLLM2,RAG),
withprecision48.55%andrecall94.47%.Toverifythatthis
reflectsgenuinedetectionratherthanthepositive-heavybias
that F1 can reward on enriched evaluation sets, we report
the Matthews correlation coefficient (MCC), which is zero
foranytrivialconstantclassifier.Theproposedconfiguration
attains MCC=0.146 (𝜒2≈ 204,𝑝 <10−10), confirming a
statistically significant association between predictions and
labels. This pattern is consistent across the configurations
in Table 4, indicating that it stems from the framework’s
architecture rather than from any single model choice.
Across both datasets, the proposed framework consis-
tentlyoutperformstheGABMbaselineonF1andprecision,
offering a more balanced precision-recall operating point.
The dominant driver of these gains is the structured feature
representation produced by LLM1, supported by per-user
behaviouralgrounding;LLM2aggregatestheseper-window
indicators into window-level decisions. The remaining de-
signdecisionsareisolatedintheablationstudy(Section5.4).
While high recall is desirable in security operations to
minimise missed threats, the framework’s higher precision
than GABM’s reported results indicates better discrimina-
tion between benign and malicious windows. We do not
claim a specific operational false-positive rate: because this
distribution is enriched for malicious activity, precision on
a realistic, predominantly benign stream would be lower
(Section 5.5).
5.2. Feature Analysis
The cross-model results in Table 4 show that the stage
at which model capability matters most differs by dataset:
on CERT r5.2 the strongest configuration places the larger
model at classification (Llama 8B/DeepSeek, RAG, F1
=64.14%), whereas on the stealthier PicoDomain logs it is
required at feature generation (DeepSeek/Llama 8B, No-
RAG, F1=50.87%). To explain this, we examine the dis-
criminative signal carried by the LLM1feature vectors, us-
ingSHAPfeatureimportanceLundbergandLee(2017)and
distribution analysis, for DeepSeek-R1-Distill-Qwen 32B
and Llama 3.1 8B on each dataset. The Random ForestTable 4
Detection performance (F1-score, %) across the four
LLM1/LLM2model combinations, under retrieval (RAG) and
no retrieval (No-RAG), on both datasets. Rows are grouped by
the feature-generating model (LLM1); best F1 per dataset in
bold.
CERT r5.2 PicoDomain
LLM1/ LLM2 RAG No-RAG RAG No-RAG
DeepSeek / DeepSeek 58.36 61.72 45.74 46.08
DeepSeek / Llama 8B 60.53 62.31 50.1750.87
Llama 8B / Llama 8B 63.88 60.88 48.18 34.73
Llama 8B / DeepSeek64.1460.08 47.24 34.85
Table 5
Comparison of the proposed framework against the prior
LLM-based baseline (%). The proposed row reports the best
configuration per dataset (PicoDomain: DeepSeek/Llama 8B,
No-RAG; CERT r5.2: Llama 8B/DeepSeek, RAG). Best F1 per
dataset inbold.
Dataset Method Precision Recall F1
PicoDomainGABM (Ferraro et al., 2025) 10.72 100.00 19.37
Proposed framework 36.57 83.5050.87
CERT r5.2GABM (Ferraro et al., 2025) 35.82 100.00 52.74
Proposed framework 48.55 94.4764.14
underlying the SHAP analysis is used solely as a probe
of the intrinsic discriminative content of the LLM1feature
vectors; it is not part of the proposed pipeline, and the
resultingimportancescharacterisethefeaturesetratherthan
the behaviour of LLM2.
CERT r5.2.To validate LLM1feature quality on CERT
r5.2, we analyse feature score distributions (Figure 5) and
SHAP feature importance (Figure 6). DeepSeek-R1 32B
ranksusb_usage_riskas the dominant contributor (20.2%
of total importance), consistent with USB-based exfiltra-
tion scenarios in CERT r5.2. Llama 3.1 8B produces a
different ranking, prioritisingsensitive_file_access(16.9%)
andfile_exfiltration_risk(15.3%). Although the single top-
ranked feature differs, both models concentrate importance
onthesamefamilyoffile-accessandexfiltrationindicators,
the features that also show the clearest malicious/benign
separation in Figure 5. The separation is substantial: the
mean absolute difference between class averages across the
17 features is 0.057, withusb_usage_riskreaching 0.210
(malicious mean 0.295 versus benign 0.085). The agree-
ment between the two analyses indicates the discrimina-
tive signal is robust to model choice even where the ex-
act ranking is not. Under No-RAG, DeepSeek’s anomaly
featuressharpenrelativetoRAG(e.g.sensitive_file_access,
time_anomaly_score), indicating that retrieval redistributes
featureimportanceratherthanaddingdiscriminativepower,
consistent with the substitution pattern in Table 4.
PicoDomain.OnPicoDomain,theLLM1riskscoresshow
very weak malicious/benign separation: the mean absolute
First Author et al.:Preprint submitted to ElsevierPage 10 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
difference between class averages across the 20 indicators
is 0.008 (maximum 0.030), roughly seven-fold smaller than
on CERT r5.2; per-window feature variance is essentially
identical between classes (0.043 vs 0.044 for DeepSeek-R1
32B); and both classes activate a near-identical number of
features per window (9.4 vs 9.5 of 20). The largest differ-
ences are moreover mildly inverted: benign windows score
higherthan malicious ones (e.g.large_transfer_anomaly,
0.130 vs 0.100), reflecting that stealthy beaconing pro-
duces per-window records less conspicuous than ordinary
traffic, a characteristic of the threat rather than a failure
of feature design. SHAP analysis (Figure 7) shows the
same pattern, with maximum mean|SHAP|below 0.035,
roughly threefold smaller than the dominant CERT r5.2
features (≈0.089). The two models also differ in where
they place importance: DeepSeek-R1 32B spreads it across
network-level indicators (external_connection_risk,recon-
naissance_pattern,overall_sequence_risk) with no single
dominant feature, whereas Llama 3.1 8B under RAG con-
centrates onkerberos_anomaly_score(a feature that shows
no malicious/benign separation in Figure 5), indicating
reliance on a non-discriminative signal. This weak per-
window signal (Figure 7) is reflected in the prevalence-
robustmetric:onPicoDomainthebestconfigurationattains
MCC=0.004, statistically indistinguishable from chance,
in contrast to the significant association on CERT r5.2
(MCC=0.146).Wethereforetreatper-windowdetectionon
PicoDomain as a boundary case for this class of method,
attributable to the near-absence of extractable signal in
stealthy beaconing traffic rather than to a failure of feature
design, and not as evidence of reliable detection.
5.3. Per-User Classification Performance
Table 6 presents per-user results on PicoDomain for the
best configuration (DeepSeek as LLM1, Llama 3.1 8B as
LLM2, No-RAG). Performance varies considerably across
users, with F1-scores ranging from 40.00% to 61.37%. Per-
user results for CERT r5.2 are omitted due to the large
numberofusers(99);combinedresultsforallconfigurations
are reported in Table 4.
Recall is uniformly high across the four larger users
(79.65%–86.82%), indicating the framework detects most
malicious windows regardless of user; precision is the
binding constraint throughout (29.30%–47.46%). JSNAKE
achieves the strongest result (F1=61.37%), combining the
highest precision (47.46%) with high recall, followed by
BDUCK (52.99%) and RMOLE (52.08%). JDOE records
the lowest precision (29.30%): its traffic is predominantly
benign(520of751windows),sofalsepositivesaccumulate
disproportionately. ADMINISTRATOR scores lowest over-
all (40.00%) on a very small sample (11 malicious and 20
benign windows), where individual misclassifications shift
the score considerably.
5.4. Ablation Study
Weanalysethecontributionoftwodesigndecisions:the
window representation fed to LLM1, and the log-grouping
strategy. The window-representation ablation is conductedTable 6
Per-user results on PicoDomain for the best configuration
(DeepSeek/Llama 8B, No-RAG).
User Prec. Recall F1 Mal. Ben.
ADMINISTRATOR 31.58 54.55 40.00 11 20
BDUCK 38.96 82.82 52.99 326 513
JDOE 29.30 79.65 42.84 231 520
JSNAKE 47.46 86.82 61.37 129 158
RMOLE 37.20 86.80 52.08 303 530
Combined 36.57 83.50 50.87 1,000 1,741
on CERT r5.2, whose five log sources carry rich and varied
per-event fields; how each window is encoded materially
affects the signal available to LLM1. PicoDomain’s dis-
criminative signal is instead concentrated in the protocol
and destination of each connection, already captured by the
sequence representation, and a comparable representation
studyisthereforeleftforfuturework.Thegrouping-strategy
ablation is conducted on PicoDomain because connection-
pairgroupingisanetwork-specificalternativewithnocoun-
terpart in CERT r5.2’s host activity logs, where per-user
grouping instead serves to merge the five log sources into
a single timeline. The contribution of retrieval and of the
model assigned to each stage is analysed jointly in Sec-
tion 5.1 (Table 4).
5.4.1. Effect of window representation
Table 7 isolates the effect of window representation on
CERT r5.2, holding the model, retrieval method, and fea-
ture set constant (DeepSeek-R1-32B, embedding RAG, 17
features).Wecomparethreerepresentationsofthesameun-
derlyinglogwindow:(i)verboseeventtext:therawnatural-
languageloglinesofthewindowconcatenatedintoasingle
text block; (ii)sequence + destination tokens: a compact
summary listing the ordered sequence of activity types and
a deduplicated set of all destinations referenced (recipients,
URLs, file paths, hosts, attachment names); and (iii)struc-
turedeventobjects:eachlogentryrenderedasaJSONobject
retaining the fields relevant to threat reasoning (timestamp,
action, sender, recipients, URL, filename, file tree, host,
size, attachments, and USB flags), with events preserved in
temporal order.
Verbosefull-textrepresentationunderperformsthecom-
pact sequence-and-destination format, which is in turn out-
performed by structured event objects, the representation
adoptedintheproposedsystem.Structured,temporallyrich
input provides the most discriminative signal for LLM1
feature extraction and is the dominant controllable factor in
detection performance on CERT r5.2.
5.4.2. Effect of grouping strategy
A core component of the framework is per-user log
grouping(Section3),whichmergesalllogentriesassociated
with a single user into a unified chronological timeline. To
validate this design, we compare it against connection-pair
groupingonPicoDomain,inwhichwindowsareconstructed
First Author et al.:Preprint submitted to ElsevierPage 11 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Figure 5:LLM1feature score distributions (malicious vs benign) for CERT r5.2 (top) and PicoDomain (bottom). On CERT r5.2,
several features show clear separation (notablyusb_usage_risk,file_exfiltration_risk, andfile_access_pattern_change),
consistent with known insider threat behaviours. On PicoDomain, the malicious and benign distributions of the LLM1scores
are near-identical across all 20 features.
Table 7
Effect of window representation on CERT r5.2. All configu-
rations use DeepSeek-R1-32B with embedding RAG and 17
dataset-specific features.
Window Representation Prec. (%) Recall (%) F1 (%)
Verbose event text 45.84 62.52 52.89
Sequence + destination tokens 47.41 66.83 55.47
Structured event objects (proposed) 48.43 73.4058.36
from logs grouped by source–destination host pair rather
than by user. Here each unique pair forms its own log
sequence, segmented into windows by the same sliding-
window procedure used for per-user grouping. Both con-
figurations use identical features (20 network indicators),
DeepSeek-R1-32BwithembeddingRAG,andinferenceset-
tings; only the grouping unit differs. As shown in Table 8,
per-usergroupingsubstantiallyoutperformsconnection-pairTable 8
Effect of log-grouping strategy on PicoDomain. Both con-
figurations use identical features (20 network indicators),
DeepSeek-R1-32B with embedding RAG, and inference set-
tings; only the grouping unit differs.
Grouping Prec. (%) Recall (%) F1 (%)
Connection-pair (source–destination) 20.45 55.41 29.87
Per-user (proposed) 36.16 62.2045.74
grouping (F1=45.74% vs 29.87%), with the largest gain
inprecision(36.16%vs20.45%).Connection-pairgrouping
produces three times more windows (8,320 vs 2,741) by
fragmentingeachuser’sactivityacross83connectionpairs,
diluting the behavioural context available to LLM1. User-
leveltimelinesinsteadpreservethefullscopeofeachuser’s
activityacrossallconnectiontypes,providingmorecoherent
context for threat-aware feature extraction.
First Author et al.:Preprint submitted to ElsevierPage 12 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Figure 6:LLM1feature importance on CERT r5.2: mean|SHAP|from TreeSHAP (Lundberg and Lee, 2017) over a Random
Forest trained on the LLM1feature vectors, for both feature-generation models under RAG and No-RAG. Features appear in
extraction order with a shared x-axis across all four panels.
5.5. Limitations
First, the temporal-context sampling evaluates attack-
proximal windows, not a continuous stream. The subset is
enriched for malicious activity, so the reported precision
reflects this distribution rather than an operational false-
positive rate, which would be lower on a predominantly
benign stream. As malicious windows appear in bursts,
part of LLM2’s performance may derive from this structure
rather than per-window content.
Second, LLM1features carry almost no per-window
signal on PicoDomain, and detection there is at chance
(MCC=0.004)againstasignificantassociationonCERTr5.2
(MCC=0.146). We treat PicoDomain as a boundary case
forthisclassofmethod,reflectingstealthybeaconingrather
than a feature-design flaw (Section 5.2).
Third, we do not compare absolute metrics against
GABM: it classifies per activity identifier rather than per
window, and specifies neither its labelling nor its benign-
sampling procedure (5,000 of 537,840 entries, a prevalence
near 1.6% against our 36.5%), so a like-for-like compari-
son is not definable. We rely instead on prevalence-robust
metrics and internal comparisons on an identical subset.Generalisation beyond these two datasets, and to other
prompts and embedding models, remains to be established.
6. Conclusion
We presented a retrieval-augmented dual-LLM frame-
workforzero-shotintrusiondetectionacrossheterogeneous
security logs. Logs are grouped into per-user chronological
timelines against which each window is assessed. LLM1
transformsrawlogwindowsintostructuredvectorsofinter-
pretableriskindicatorstailoredtothethreattype,andLLM2
classifiesorderedsequencesofthesevectorstodetectmulti-
window attack patterns.
Evaluated on CERT r5.2 and PicoDomain, the frame-
workachievesbestF1-scoresof64.14%and50.87%respec-
tively, improvements of 11.40 and 31.50 percentage points
over the GABM baseline, and every one of the eight eval-
uated configurations exceeds the baseline on both datasets,
indicating the gains stem from the architecture rather than
anysinglemodelchoice.Thecross-modelanalysissuggests
two patterns. First, retrieval and model capability appear to
act as substitutes: retrieval improves the features produced
by the less capable model, especially on weak-signal data,
First Author et al.:Preprint submitted to ElsevierPage 13 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Figure 7:LLM1feature importance on PicoDomain; same method and layout as Figure 6.
whereasthemorecapablemodelattainscomparablefeature
quality without retrieved context. Second, the best config-
urations assign different models to the two stages, placing
themorecapablemodelatfeaturegenerationonweak-signal
data and at classification on strong-signal data.
FeatureanalysisshowsthatLLM1featurequalityreflects
the nature of each threat type: insider threat activity on
CERTr5.2producesclearbehaviouraltraces(meanabsolute
class difference 0.057 across the 17 features), while APT
beaconing on PicoDomain is inherently stealthy (0.008,
with the largest differences mildly inverted). The structured
representation nonetheless provides a consistent basis for
LLM2’swindow-levelclassification,despitethelimiteddis-
criminative signal at the individual window level.
Future work includes evaluating on additional datasets,
investigating cross-user retrieval to surface shared C2 bea-
coning patterns across hosts, and exploring fine-tuning
LLMs for the feature generation stage.References
Alzaabi, F.R., Mehmood, A., 2024. A review of recent advances, chal-
lenges, and opportunities in malicious insider threat detection using
machine learning methods. IEEE Access 12, 30907–30927.
Borah,A.,Alam,M.T.,Rastogi,N.,2025. Adaptinglargelanguagemodels
to emerging cybersecurity using retrieval augmented generation. arXiv
preprint arXiv:2510.27080 .
Bridges, R.A., Glass-Vanderlan, T.R., Iannacone, M.D., Vincent, M.S.,
Chen,Q.,2019. Asurveyofintrusiondetectionsystemsleveraginghost
data. ACM computing surveys (CSUR) 52, 1–35.
Ferraro, A., Orlando, G.M., Russo, D., 2025. Generative agent-based
modelingwithlargelanguagemodelsforinsiderthreatdetection. Engi-
neering Applications of Artificial Intelligence 157, 111343.
Georgiadou, A., Mouzakitis, S., Askounis, D., 2022. Detecting insider
threat via a cyber-security culture framework. Journal of Computer
Information Systems 62, 706–716.
Glasser, J., Lindauer, B., 2013. Bridging the gap: A pragmatic approach
to generating insider threat data, in: 2013 IEEE Security and Privacy
Workshops, IEEE. pp. 98–104.
Houssel, P.R., Layeghy, S., Singh, P., Portmann, M., 2026. ex-nids: A
framework for explainable network intrusion detection leveraging large
language models. Computers and Electrical Engineering 129, 110826.
Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., Iwasawa, Y., 2022. Large lan-
guage models are zero-shot reasoners. Advances in Neural Information
Processing Systems 35, 22199–22213.
First Author et al.:Preprint submitted to ElsevierPage 14 of 15

Dual-LLM Framework for Zero-Shot Cyber Threat Detection
Kong, K., Liu, D., Jin, X., Geng, G., Li, Z., Weng, J., 2025. Dmfi: A dual-
modality log analysis framework for insider threat detection with lora-
tunedlanguagemodels,in:2025IEEEInternationalConferenceonData
Mining (ICDM), IEEE. pp. 387–396.
Laprade,C.,Bowman,B.,Huang,H.H.,2020.Picodomain:acompacthigh-
fidelity cybersecurity dataset. arXiv preprint arXiv:2008.09192 .
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler,H.,Lewis,M.,Yih,W.t.,Rocktäschel,T.,etal.,2020. Retrieval-
augmented generation for knowledge-intensive nlp tasks. Advances in
neural information processing systems 33, 9459–9474.
Li, C., Zhu, Z., He, J., Zhang, X., 2025. Redchronos: A large language
model-based log analysis system for insider threat detection in enter-
prises. arXiv preprint arXiv:2503.02702 .
Lundberg, S.M., Lee, S.I., 2017. A unified approach to interpreting model
predictions, in: Advances in Neural Information Processing Systems.
Mandiant, 2013. APT1: Exposing One of China’s Cyber Espionage
Units. Technical Report. Mandiant Intelligence Center. URL:https:
//services.google.com/fh/files/misc/mandiant-apt1-report.pdf.
MITRE Corporation, 2026. MITRE ATT&CK Knowledge Base. URL:
https://attack.mitre.org/. accessed: 2026-04-20.
Paul, S., Alemi, F., Macwan, R., 2025. Llm-assisted proactive threat
intelligence for automated reasoning. arXiv preprint arXiv:2504.00428
.
Pletzer, B., Mottok, J., 2026. From anomaly to attack path: Llm-based
networktrafficinvestigationforaptdetection,in:Proceedingsofthe19th
European Workshop on Systems Security, Association for Computing
Machinery, New York, NY, USA. p. 10–16. URL:https://doi.org/10.
1145/3803525.3804991, doi:10.1145/3803525.3804991.
Prabhu,S.,Thompson,N.,2022. Aprimeroninsiderthreatsincybersecu-
rity. Information Security Journal: A Global Perspective 31, 602–611.
Reimers, N., Gurevych, I., 2019. Sentence-bert: Sentence embeddings
using siamese bert-networks, in: Proceedings of EMNLP.
Song,C.,Ma,L.,Zheng,J.,Liao,J.,Kuang,H.,Yang,L.,2024. Audit-llm:
Multi-agent collaboration for log-based insider threat detection. arXiv
preprint arXiv:2408.08902 .
Song, C., Zheng, J., Ma, L., Liao, J., Kuang, H., Yang, L., 2025a. Insight-
llm: Llm-enhanced multi-view fusion in insider threat detection. arXiv
preprint arXiv:2509.01509 .
Song, S., Zhang, Y., Gao, N., 2025b. Confront insider threat: Precise
anomalydetectioninbehaviorlogsbasedonllmfine-tuning,in:Proceed-
ings of the 31st international conference on computational linguistics,
pp. 8589–8601.
Xu, H., Wang, S., Li, N., Wang, K., Zhao, Y., Chen, K., Yu, T., Liu, Y.,
Wang,H.,2024. Largelanguagemodelsforcybersecurity:Asystematic
literature review. ACM Transactions on Software Engineering and
Methodology .
Yuan,S.,Wu,X.,2021. Deeplearningforinsiderthreatdetection:Review,
challenges and opportunities. Computers & Security 104, 102221.
First Author et al.:Preprint submitted to ElsevierPage 15 of 15
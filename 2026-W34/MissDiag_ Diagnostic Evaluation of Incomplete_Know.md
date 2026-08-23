# MissDiag: Diagnostic Evaluation of Incomplete-Knowledge Robustness in KGQA and KG-RAG

**Authors**: Hang Wang, Hang Dong, Lu Liu, Chuanru Ren

**Published**: 2026-08-19 03:28:11

**PDF URL**: [https://arxiv.org/pdf/2608.18489v1](https://arxiv.org/pdf/2608.18489v1)

## Abstract
Knowledge graph question answering (KGQA) and knowledge-graph-based retrieval-augmented generation (KG-RAG) aim to ground answers in explicit graph evidence, but real-world knowledge graphs are often sparse, outdated, and incomplete. Existing robustness evaluations usually report aggregate changes in answer quality after evidence is removed or perturbed, which measures sensitivity to incomplete support but leaves the source of degradation under-specified: the same score change can conflate the type of missing evidence, the response of the evaluated system, and the sensitivity of the answer-matching protocol. To address this gap, we propose \textbf{MissDiag}, a diagnostic evaluation framework for incomplete-knowledge robustness in KGQA and KG-RAG. MissDiag keeps the question and gold answer fixed while applying structurally typed missingness interventions to benchmark-provided support graphs, enabling paired comparisons that decompose robustness changes by evidence type, system response, and evaluation protocol rather than reducing them to a single aggregate score drop. Experiments across multiple system families show that incomplete-knowledge robustness is better understood as a typed degradation phenomenon than as a uniform property: answer-adjacent evidence loss produces the largest observed degradation, source-context removal is often neutral and can be beneficial, and semantic answer matching changes absolute scores while preserving the main typed degradation patterns. By transforming aggregate robustness measurement into typed diagnostic attribution, MissDiag provides a more interpretable basis for comparing, diagnosing, and stress-testing KGQA and KG-RAG systems under incomplete knowledge.

## Full Text


<!-- PDF content starts -->

MissDiag: Diagnostic Evaluation of Incomplete-Knowledge Robustness
in KGQA and KG-RAG
Hang Wang, Hang Dong, Lu Liu, Chuanru Ren
Abstract
Knowledge graph question answering (KGQA) and
knowledge-graph-basedretrieval-augmentedgeneration(KG-
RAG) aim to ground answers in explicit graph evidence, but
real-world knowledge graphs are often sparse, outdated, and
incomplete.Existingrobustnessevaluationsusuallyreportag-
gregatechangesinanswerqualityafterevidenceisremovedor
perturbed, which measures sensitivity to incomplete support
butleavesthesourceofdegradationunder-specified:thesame
score change can conflate the type of missing evidence, the
response of the evaluated system, and the sensitivity of the
answer-matching protocol. To address this gap, we propose
MissDiag,adiagnosticevaluationframeworkforincomplete-
knowledge robustness in KGQA and KG-RAG. MissDiag
keeps the question and gold answer fixed while applying
structurally typed missingness interventions to benchmark-
provided support graphs, enabling paired comparisons that
decompose robustness changes by evidence type, system re-
sponse,andevaluationprotocolratherthanreducingthemtoa
singleaggregatescoredrop.Experimentsacrossmultiplesys-
tem families show that incomplete-knowledge robustness is
betterunderstoodasatypeddegradationphenomenonthanas
a uniform property: answer-adjacent evidence loss produces
thelargestobserveddegradation,source-contextremovalisof-
tenneutralandcanbebeneficial,andsemanticanswermatch-
ing changes absolute scores while preserving the main typed
degradation patterns. By transforming aggregate robustness
measurementintotypeddiagnosticattribution,MissDiagpro-
vides a more interpretable basis for comparing, diagnosing,
andstress-testingKGQAandKG-RAGsystemsunderincom-
plete knowledge.
Code will be released after the review process.
Introduction
Knowledge graph question answering (KGQA) and
knowledge-graph-based retrieval-augmented generation
(KG-RAG)aimtogroundanswersinexplicitgraphevidence
rather than relying only on parametric memory. This makes
themimportanttestbedsforevaluatingstructuredreasoning,
factual grounding, and answer trustworthiness. Prior work
has advanced this goal through semantic parsing and query-
graph construction (Berant et al. 2013; Yih et al. 2015),
embedding-based and graph-based QA (Huang et al. 2019;
Sun, Bedrax-Weiss, and Cohen 2019; Shi et al. 2021; Ye
etal.2022;GuandSu2022;MavromatisandKarypis2022),
Figure 1: Motivation for typed degradation analysis. The
same QA instance can show similar aggregate degradation
under structurally different missingness conditions. Aggre-
gate score drops alone do not reveal which evidence type
was removed, whether answer-local evidence was affected,
or whether the effect is systematic.
retrieve-and-reason systems over knowledge bases and text
(Guu et al. 2020; Lewis et al. 2020), LLM-based KBQA
(Xiong, Bao, and Zhao 2024), and increasingly challenging
benchmarks (Dubey et al. 2019; Gu et al. 2021; Cao et al.
2022; Zhang et al. 2025). More recently, graph-grounded
LLM studies have examined whether KG augmentation im-
proves factuality, reasoning quality, retrieval efficiency, and
trustworthiness in open-ended generation settings (Wang
et al. 2024; Sun et al. 2024; Sui et al. 2025; Zhou et al.
2026).Acrosstheselinesofwork,however,apersistentdiffi-
culty remains: real-world knowledge graphs are rarely com-
plete. They are often sparse, outdated, and unevenly pop-
ulated across entities and relations, making robustness un-
der incomplete knowledge a central requirement for reliable
KGQA and KG-RAG systems.
Existingresearchhasstudiedincompleteknowledgefrom
several complementary perspectives, including contextual
reasoning over partial graph structures (Mai et al. 2019), in-
tegrationoftextualevidence(Xiongetal.2019;Han,Cheng,
and Wang 2020; Sun et al. 2023), KG embeddings and re-
lation prediction (Trouillon et al. 2016; Schlichtkrull et al.
2018; Sun et al. 2019; Huang et al. 2019; Saxena, Tripathi,
and Talukdar 2020; Zhao et al. 2022; Zan et al. 2022; Sax-
ena, Kochsiek, and Gemulla 2022; Guo et al. 2023), knowl-
edge graph completion (Zhao et al. 2022; Guo et al. 2023),
completion-aware reasoning pipelines (Liu et al. 2022; Ye
arXiv:2608.18489v1  [cs.CL]  19 Aug 2026

etal.2024;Hanetal.2025),andLLM-basedormulti-agent
reasoning over incomplete graphs (Xu et al. 2024; Liu et al.
2026). In parallel, evaluation studies have asked whether
completion methods improve downstream QA and whether
currentbenchmarksandmetricssupportreliableconclusions
under incomplete or imperfect knowledge (Yu et al. 2023;
Perevalovetal.2022;SteinmetzandSattler2021;Zhangetal.
2025,2026;Zhouetal.2026;Suietal.2025).Thesestudies
have substantially improved our understanding of how sys-
temsrecovermissingfacts,exploitauxiliaryevidence,orrea-
son over partial structures. Nevertheless, current evaluation
practice still has a basic attribution limitation: most evalua-
tionsremove,corrupt,orrecoverevidenceandthenreportthe
resulting change in answer quality. Such degradation-based
evaluationcanmeasurewhetherperformancechanges,butit
does not explain why the change occurs.
This ambiguity is illustrated in Figure 1: the same
question–answer instance may exhibit similar aggregate
score drops under structurally different evidence-removal
conditions, even though the underlying failure mechanisms
are different. A lower score may indicate that essential sup-
porting evidence has become unavailable; it may also indi-
cate that the system fails to exploit retained evidence, that
graph conversion or candidate construction changes the ef-
fective search space, or that the evaluation protocol fails to
recognizeasemanticallyacceptableanswer.Conversely,ap-
parent robustness or even improvement under incomplete
evidence may reflect support pruning or metric behavior
rather than stronger reasoning. This matters because ro-
bustness under incomplete knowledge is often used as evi-
denceofreasoningcapabilityandsystemreliability.Ifsimilar
scorechangescanarisefromdifferentcauses,thenaggregate
degradation alone is insufficient for interpreting robustness
claims. What is needed is not only a performance measure-
ment protocol, but a diagnostic evaluation framework that
can attribute degradation to different forms of missing ev-
idence, compare how system families respond to the same
intervention, and expose how much the conclusion depends
on the evaluation metric.
To address this gap, we proposeMissDiag, a diagnostic
evaluation framework for incomplete-knowledge robustness
in KGQA and KG-RAG. As shown in Figure 2, MissDiag
starts from a benchmark-provided evaluation instance con-
sisting of a question, a gold answer set, and a local sup-
port graph. It keeps the question and gold answer fixed
while transforming the support graph into structurally dis-
tinctincomplete-evidenceconditions.Theprimarymissing-
ness operators include random support loss, source-context
loss, relation-level removal, and answer-adjacent removal.
Byevaluatingthesamesystemonthesamequestion–answer
pairbeforeandaftereachtypedintervention,MissDiagsep-
aratesthreefactorsthatareusuallyentangledinincomplete-
knowledge evaluation: missingness type, system response,
and evaluation sensitivity.
Rather than treating incomplete knowledge as a single
robustness condition, MissDiag represents it as a typed
degradation phenomenon. The framework reports robust-
ness through degradation profiles indexed by missingness
type,severity,system,andevaluationmetric,withstructuralslices used for further analysis. This design enables paired
andinterpretablecomparisonacrosstrainedKGQAmodels,
graph-structured prompting methods, iterative KG agents,
and direct LLM baselines. It therefore allows robustness
claimstobeexaminedintermsofwheredegradationcomes
from, when it reflects genuine evidence loss, and when it
shouldinsteadbeattributedtosystembehaviororevaluation
sensitivity.
Our contributions are as follows:
•We formulate incomplete-knowledge robustness evalu-
ation as a diagnostic attribution problem, showing that
aggregate score changes conflate evidence availability,
system behavior, and evaluation protocol, and therefore
cannot by themselves support reliable conclusions about
reasoning robustness.
•We introduce MissDiag, a controlled diagnostic frame-
work that applies structurally typed missingness inter-
ventions to benchmark support graphs while preserving
paired question–answer comparisons, enabling degrada-
tiontobeanalyzedbymissingnesssourceratherthanonly
by overall performance loss.
•WeinstantiateMissDiagacrossmultiplesystemfamilies,
severity levels, structural slices, and answer-matching
metrics, demonstrating that the same missing-evidence
intervention can lead to degradation, near invariance, or
improvementdependingonsystemdesignandevaluation
protocol. This reveals robustness patterns that aggregate
scores obscure.
Related Work
KGQA and Graph-Grounded QA
KGQA aims to answer natural-language questions using
structured graph evidence. Early work often relied on se-
mantic parsing or query-graph construction to map ques-
tions into executable logical forms or graph queries (Berant
et al. 2013; Yih et al. 2015). Later graph-based models re-
trieveandreasonoverlocalevidencesubgraphs,makingthe
support structure itself part of the answering process (Sun,
Bedrax-Weiss,andCohen2019;Heetal.2021;Mavromatis
and Karypis 2022). In parallel, retrieval-centered QA and
graph-grounded LLM methods use retrieved knowledge to
support generation and reasoning beyond purely parametric
memory(Guuetal.2020;Lewisetal.2020;Wangetal.2024;
Xiong, Bao, and Zhao 2024; Sun et al. 2024). Benchmarks
such as LC-QuAD 2.0, GrailQA, KQA Pro, and KGQAGen
broadenevaluationbeyondsimplefactlookupbyemphasiz-
ing compositionality, generalization, and dataset reliability
(Dubey et al. 2019; Gu et al. 2021; Cao et al. 2022; Zhang
etal.2025).Thesestudiesprovidethesystemandbenchmark
context for our work. MissDiag differs by treating existing
systemsasdiagnosticsubjectsundercontrolledevidencema-
nipulationratherthanproposinganotherKGQAarchitecture.
Incomplete-Knowledge Question Answering
Incomplete knowledge is a persistent challenge for KGQA
because real-world graphs are sparse, unevenly populated,
andoftenmissingfactsneededformulti-hopreasoning.Prior

Figure2:OverviewofMissDiag.(1)Input:aninstancecontainsaquestion,goldanswerset,andlocalsupportgraph.(2)Typed
MissingnessOperators:typedoperatorsselectremovablesupportedgesforrandom,source-context,relation-level,andanswer-
adjacent missingness. (3)Severity-Controlled Missingness: a shared severity budget produces an incomplete support graph.
(4)Paired Evaluation: the same system is evaluated under complete and incomplete support to compute paired degradation.
(5)Typed Degradation Profile: degradation values are summarized across missingness type, severity, system, and metric.
workhasaddressedthisissuebyreasoningoverpartialgraph
structures (Mai et al. 2019), incorporating auxiliary textual
evidence (Xiong et al. 2019; Han, Cheng, and Wang 2020),
using graph completion or completion-aware QA pipelines
(Yuetal.2023;Yeetal.2024),andapplyingLLM-centered
reasoning to incomplete graph evidence (Xu et al. 2024;
Zhou et al. 2026). These approaches mainly ask how to re-
coverormaintainanswerqualitywhenknowledgeismissing.
Our work asks a complementary evaluation question: when
answer quality changes under incomplete knowledge, how
should the change be attributed? Instead of treating miss-
ingness as a single condition, MissDiag separates different
structural forms of evidence loss and compares their effects
under a paired protocol.
Benchmark Reliability and Evaluation Sensitivity
Evaluation conclusions in KGQA and graph-grounded QA
are sensitive to dataset construction, evidence availability,
and answer-matching protocols. Dataset audits and leader-
board analyses show that KGQA benchmarks can contain
annotation issues, heterogeneous difficulty, and inconsistent
reporting practices (Steinmetz and Sattler 2021; Perevalov
et al. 2022; Zhang et al. 2025). More broadly, adversar-
ial evaluation and behavioral testing show that aggregate
metrics can hide distinct failure mechanisms behind simi-
lar score changes (Jia and Liang 2017; Ribeiro et al. 2020;
Gardner et al. 2020). Recent KG-RAG and trustworthiness
studies further suggest that graph augmentation and incom-
plete evidence require careful evaluation design (Sui et al.
2025; Zhang et al. 2026; Zhou et al. 2026). This literature
motivates our view that evaluation is not a neutral report-
ing layer. MissDiag extends this line of work by decompos-
ingincomplete-knowledgeevaluationintomissingnesstype,
system response, severity, and metric sensitivity, so that ro-bustnessclaimscanbeinterpretedbeyondasingleaggregate
degradation score.
Method
This section introduces MissDiag, a diagnostic evaluation
framework for incomplete-knowledge robustness in KGQA
and KG-RAG. As shown in Figure 2, the framework keeps
thequestion,goldanswer,andevaluatedsystemfixed,trans-
formsthesupportgraphwithseverity-controlledtypedmiss-
ingnessoperators,andcomparestheresultingoutputsagainst
thecomplete-supportcondition.Thisdesignturnsaggregate
robustness changes into paired degradation profiles indexed
by missingness type, severity, system, and metric. The sec-
tion first formulates the diagnostic evaluation problem, then
describes support graph construction, defines the missing-
ness operators, and presents the paired degradation profile
used as the main diagnostic output.
Diagnostic Formulation
Each evaluation instance is represented as
xi= (q i, A∗
i, Gi),(1)
whereq iis the question,A∗
iis the gold answer set, and
Gi= (V i, Ei)is the local support graph. The source enti-
ties linked to the question are denoted byS i⊆Vi, and the
gold-answer entities aligned to graph nodes are denoted by
Yi⊆Vi. Operators requiring unavailable alignments mark
theinstanceinfeasibleforthecorrespondingoperator.Given
an evaluated systemf, MissDiag compares the complete-
support predictionf(q i, Gi)with predictions obtained after
transforming only the support edges. Across conditions, the
question, gold answer, source entities, and node inventory
remain fixed; only the retained edge evidence changes.

Support Graph Construction
ThelocalsupportgraphG iisconstructedfromtheevidence
associated with instancex i. When the evidence is given as
triples, proof paths, or a retrieved subgraph, it is converted
into a labeled directed graph. Graph distances used by the
missingness operators are computed on the undirected pro-
jectionofthisgraph.ForanodeuandnodesetB,dist(u, B)
denotes the shortest-path distance fromuto any node inB
on the undirected projection.
The edge set is written as
Ei=Esup
i∪Ectx
i,(2)
whereEsup
idenotes the original support evidence andEctx
i
denotes a possibly empty set of optional source-anchored
context edges. The context edges are constructed before any
missingness intervention, follow a deterministic selection
rule, and remain fixed across all conditions. This ensures
that all missingness operators start from the same support
graph.
After an intervention, the evaluated system receives only
the retained support evidence. The node inventory may be
kept fixed internally for alignment and paired comparison,
butisolatednodesarenotexposedasadditionalanswerhints
ingenerativeKG-RAGprompts.Additionalconstructionde-
tails are provided in the supplementary material.
Severity-Controlled Missingness
Letα∈[0,1]denotethemissingnessseverityandn i=|E i|
thenumberofsupportedges.Foreachinstance,thenominal
removal budget is defined as
κi(α) =0, n i≤1orα= 0,
min 
ni−1, η i(α)
,otherwise,(3)
whereη i(α) = max(1,round(αn i)).Thisbudgetpreserves
complete support atα= 0and keeps at least one support
edge when removal is applied.
Eachmissingnessoperatormdefinesanorderedcandidate
edge listL(m)
i. Given the shared budget, the removed edge
setR(m,α)
iconsists of the firstmin(κ i(α),|L(m)
i|)edges in
L(m)
i. The retained support graph is
eG(m,α)
i = (V i, Ei\R(m,α)
i),(4)
whereR(m,α)
iistheremovededgeset.Ifκ i(α)>0butL(m)
i
is empty, the instance is marked infeasible for operatorm.
Typed Missingness Operators
MissDiag uses four typed missingness operators: random,
source-context, relation-level, and answer-adjacent missing-
ness. Each operator first defines a candidate edge set and
thenordersitintoL(m)
i,whichisusedbythesharedremoval
budget. The operators are designed to probe different struc-
turalformsofevidencelossunderthesameremovalbudget,
rather than to define mutually exclusive edge categories.
For an edgee= (u, r, v), two structural scores are used:
τi(e) = max{dist(u, S i),dist(v, S i)},(5)ρi(e) = min{dist(u, Y i),dist(v, Y i)}.(6)
Here,τ i(e)measures source depth andρ i(e)measures an-
swer distance.
Random missingness.Random missingness uses all sup-
port edges as candidates,C(rand)
i =E i. The ordered list
L(rand)
iis a uniform random permutation ofC(rand)
i. This
operator serves as a quantity-matched baseline for generic
support loss.
Source-contextmissingness.Source-contextmissingness
targets edges incident to source entities but not incident to
aligned answer entities:
C(src)
i={e= (u, r, v)∈E i:{u, v} ∩S i̸=∅,
{u, v} ∩Y i=∅}.(7)
The ordered listL(src)
iis obtained by sorting candidates by
increasingτ i(e),withdeterministictie-breaking.Thisopera-
torprobessensitivitytosource-sidecontextwhileexcluding
answer-incident evidence from its candidate set.
Relation-level missingness.Relation-level missingness
usesallsupportedgesascandidates,C(rel)
i=E i,andgroups
thembyrelationtype.Relationblocksareorderedbydecreas-
ing frequency inE i, and edges are removed block by block
under the shared budget. This operator probes sensitivity to
relation-family evidence.
Answer-adjacentmissingness.Answer-adjacentmissing-
ness targets edges incident to, or one hop from, aligned an-
swer entities:
C(ans)
i ={e∈E i:ρi(e)≤1}.(8)
The ordered listL(ans)
iis obtained by sorting candidates by
increasingρ i(e), with deterministic tie-breaking. This oper-
ator probes sensitivity to answer-local grounding evidence.
Paired Degradation Profiles
Letµbe the answer-set evaluation metric used for a given
comparison. MissDiag treats the metric as an explicit axis
ofthediagnosticprofile.Forsystemf,thecomplete-support
score of instanceiisµ(f(q i, Gi), A∗
i), and the score under
missingness typemand severityαisµ(f(q i,eG(m,α)
i), A∗
i).
The paired degradation is defined as
δ(m,α,µ)
i =µ(f(q i, Gi), A∗
i)−µ(f(q i,eG(m,α)
i), A∗
i).(9)
Positive values indicate performance loss after evidence re-
moval, while negative values indicate improvement.
LetI(m,α)be the feasible instance set for operatormat
severityα. The dataset-level degradation of systemfis
∆(m,α,µ)
f=1
|I(m,α)|X
i∈I(m,α)δ(m,α,µ)
i .(10)
Themaindiagnosticoutputisthetypeddegradationprofile
D(α,µ)
f=
∆(rand,α,µ)
f
∆(src,α,µ)
f
∆(rel,α,µ)
f
∆(ans,α,µ)
f
.(11)

This profile summarizes how a fixed system degrades under
different missingness types at a given severity and metric.
Instances infeasible for an operator are excluded from that
operator’s aggregation, with feasibility details provided in
the supplementary material.
Experiments
WeevaluatewhetherMissDiagprovidesamoreinformative
view of incomplete-knowledge robustness than a single ag-
gregate score. Following the method design, the evaluation
firstcomparestypeddegradationprofilesacrosssystemfam-
ilies, then examines severity effects, metric sensitivity, and
structuralslices.Theexperimentsareorganizedaroundfour
research questions:
•RQ1:Domissingnesstypesproducedistinctdegradation
profiles?
•RQ2:How do profiles change as severity increases?
•RQ3:Do profiles depend on the evaluation metric?
•RQ4:When is answer-adjacent degradation strongest?
Experimental Setup
Data.We use KGQAGen-10k (Zhang et al. 2025) as the
main evaluation benchmark. Its instances provide question–
answer pairs and local evidence structures, which are con-
verted into the support-graph format used by MissDiag. We
link sourceentities from thequestion and aligngold-answer
entitiestographnodeswhenavailable.Themainexperiments
usethedevelopmentsplit.Dataset-specificsupportconstruc-
tion details are provided in the supplementary material.
Evaluation protocol.We use a paired fixed-input proto-
col. For each instance, the question, gold answer set, source
entities, and node inventory are kept fixed across complete
and incomplete-support conditions; only the retained sup-
port edges change. For a fixed instance, missingness type,
and severity, all systems are evaluated on the same trans-
formed support graph. The main cross-system comparison
uses 1,050 examples for which the complete-support condi-
tion and all four missingness conditions are feasible. LLM-
focusedanalysesusethecorrespondingfeasiblesubset,with
sample sizes reported in the relevant tables.
Systems.We evaluate four system families: (1)Trained
KGQA models, including ReaRev (Mavromatis and Karypis
2022),NuTrea(Choietal.2023),andNSM(Heetal.2021);
(2)Graph prompting methods, including MindMap (Wen,
Wang,andSun2024),StructGPT(Jiangetal.2023),andKG-
GPT (Kim et al. 2023); (3)KG agents, including ToG (Sun
et al. 2024), PoG (Chen et al. 2024), and GoG (Xu et al.
2024);and(4)DirectLLMbaselines,includingQwen2.5-7B-
Instruct1, Qwen2.5-14B-Instruct2, Llama-3.1-8B-Instruct3,
and Mistral-7B-Instruct-v0.34. For graph prompting meth-
ods and KG agents, all methods use Qwen2.5-7B-Instruct
1https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
2https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
3https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
4https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3as the shared LLM backbone across complete-support and
missingness conditions. This controls for backbone capabil-
ityandfocusesthecomparisononprompting,planning,and
agentworkflow.ForallLLM-basedsystems,graphaccessis
restricted to the transformed local support graph.
Missingness conditions.We compare complete support
with four typed missingness conditions: random, source-
context,relation-level,andanswer-adjacentmissingness.Un-
less otherwise stated, the main comparison uses severity
α= 0.3.Theseverityanalysisevaluatesα∈ {0.1,0.3,0.5}.
Metricsandreporting.Theprimarymetricisexactmacro
set-level F1. Results are reported as complete-support F1
and paired degradation∆F1, where positive values indicate
performance loss after evidence removal and negative val-
ues indicate improvement. Metric sensitivity is evaluated
by comparing exact F1 with semantic F1 for direct LLM
baselines. Semantic F1 replaces exact string matching with
semantic equivalence matching before computing set-level
precision and recall. Additional low-level details, including
prompt templates and feasibility bookkeeping, are provided
in the supplementary material.
Implementation Details
Random missingness uses a fixed global seed, with per-
example seeds derived from the sample identifier, severity,
and operator. Non-random operators use deterministic tie-
breaking when ordering candidate edges. ReaRev, NuTrea,
and NSM are trained once on complete-support data us-
ingpinnedofficialimplementations,andtheirbest-F1check-
points are reused across all missingness conditions without
condition-specific retraining. For LLM-based systems, re-
tained graph edges are serialized as textual triples and pro-
vided as the only graph evidence in the prompt. Inference
uses deterministic greedy decoding without sampling, with
fixedgenerationandreasoningbudgetsacrossevidencecon-
ditions. Experiments are conducted using NVIDIA GH200
GPUs; LLM-based experiments use PyTorch 2.9.1, CUDA
12.8,Transformers4.46.2,andbfloat16inference.Additional
implementation details are provided in the supplementary
material.
RQ1: Typed Degradation Profiles Across Systems
Table 1 reports the main cross-system comparison on 1,050
paireddevelopmentexamplesatα= 0.3.Eachrowgivesthe
complete-supportF1ofasystemanditspaired∆F1underthe
four missingness conditions, directly instantiating the typed
degradationprofiledefinedinthemethod.Threepatternsare
clear: First, answer-adjacent removal is the dominant degra-
dation condition for every system, with drops ranging from
10.3 to 21.3 F1 points. This shows that answer-local evi-
dence loss is consistently more damaging than generic sup-
port removal across trained KGQA models, graph prompt-
ingmethods,KGagents,anddirectLLMbaselines.Second,
source-context removal behaves differently: it is small for
severalpromptinganddirectLLMsystems,andnegativefor
several trained KGQA models. This indicates that remov-
ing source-side context can sometimes reduce distracting
evidence rather than harm prediction. Third, random and

Category SystemComplete Paired degradation∆F1
F1 Random Source-context Relation-level Answer-adjacent
Trained KGQA
ModelsReaRev 80.8 4.5 -1.2 2.7 11.9
NuTrea 77.2 1.9 -9.9 -2.7 12.2
NSM 46.7 3.0 -24.1 -5.2 12.1
Graph Prompting
MethodsMindMap 46.2 10.1 0.6 11.2 12.7
StructGPT 52.9 9.6 -0.7 8.0 13.8
KG-GPT 45.7 7.0 4.2 5.3 10.3
KG AgentsToG 79.5 7.0 3.9 6.0 11.4
PoG 79.9 6.8 2.7 5.8 11.6
GoG 81.7 12.1 2.3 8.9 19.8
Direct LLM
BaselinesQwen2.5-7B-Instruct 85.3 11.1 3.0 7.4 21.3
Qwen2.5-14B-Instruct 90.0 6.7 1.2 5.5 17.0
Llama-3.1-8B-Instruct 84.4 6.1 -0.5 4.0 15.8
Mistral-7B-Instruct-v0.3 81.2 5.4 1.6 3.8 13.3
Table1:Maintypeddegradationprofilesacrosssystemfamiliesatα= 0.3.Scoresarereportedascomplete-supportmacroF1
and paired∆F1 under each missingness type. Positive∆F1 indicates degradation; negative values indicate improvement.
relation-level removal usually produce intermediate degra-
dation, suggesting that support quantity and relation-family
loss affect performance but do not explain the full degrada-
tion pattern.
Together,theseresultsprovidethemainempiricalsupport
for typed degradation profiles: under the same paired proto-
col, incomplete-support degradation depends on what kind
of evidence is missing and how each system uses the re-
maining graph. A single aggregate degradation score would
collapse these distinct effects and obscure the difference be-
tweenharmfulanswer-localloss,neutralorbeneficialsource-
context removal, and intermediate random or relation-level
loss.
RQ2: Severity Effects
Figure3reportsseverity-dependentdegradationforrepresen-
tative direct LLM baselines. The severity parameter varies
overα∈ {0.1,0.3,0.5}, while the question, gold answer,
metric, and missingness operators remain fixed. The curves
show that increasing severity amplifies degradation, but not
uniformly across missingness types. Answer-adjacent re-
moval remains the strongest degradation condition at every
severityandgrowsmostsharplyasαincreases.Randomand
relation-level removal also increase with severity, but their
degradation remains consistently below answer-adjacent re-
moval. Source-context removal stays comparatively small
and changes little across severity levels. These results show
that the typed degradation pattern is not an artifact of the
mainα= 0.3setting.Severitycontrolsthescaleofevidence
loss,whiletherelativebehaviorofmissingnesstypesremains
structurally distinct.
RQ3: Metric Sensitivity
Table 2 compares paired degradation under exact F1 and
semantic F1 for direct LLM baselines atα= 0.3. This
analysis tests whether the typed degradation profile changes
0.1 0.3 0.5
Severity 
0102030Paired F1
Qwen2.5-7B
0.1 0.3 0.5
Severity 
Llama-3.1-8BRandom Source-context Relation-level Answer-adjacentFigure 3: Severity effects on paired degradation. Curves
show paired∆F1 across missingness types for Qwen2.5-
7B-Instruct and Llama-3.1-8B-Instruct asαincreases.
when answer matching allows semantic equivalence rather
than exact surface matching. The results show that metric
choice changes degradation magnitudes but not the main
typed pattern. For all four direct LLM baselines, answer-
adjacent removal remains the largest degradation condition
under both exact F1 and semantic F1. Semantic matching
slightly changes individual∆F1 values, but it does not re-
versetheorderingofmissingnesseffects.Thisindicatesthat
themaindiagnosticconclusiondependsprimarilyonthetype
of missing evidence rather than on the particular answer-
matching rule.
RQ4: Structural Slices
Table 3 examines when answer-adjacent degradation is
strongest. The analysis compares random and answer-
adjacentdegradation acrossanswer cardinalityandsupport-
graphsize,averagedoverthefourdirectLLMbaselines.The
gap is defined as answer-adjacent∆F1 minus random∆F1.

System MetricPaired degradation∆F1
Rand Src Rel Ans-adj
Qwen2.5-7BExact F1 11.1 3.0 7.4 21.3
Semantic F1 11.6 3.2 8.0 21.4
Qwen2.5-14BExact F1 6.7 1.2 5.5 17.0
Semantic F1 6.9 1.2 5.4 16.1
Llama-3.1-8BExact F1 6.1 -0.5 4.0 15.8
Semantic F1 5.1 -1.0 3.9 15.1
Mistral-7BExact F1 5.4 1.6 3.8 13.3
Semantic F1 5.1 0.9 3.2 13.5
Table 2: Metric sensitivity of typed degradation profiles for
directLLMbaselinesatα= 0.3.Rand,Src,Rel,andAns-adj
denote random, source-context, relation-level, and answer-
adjacent missingness.
Slice Group N Full Rand Ans-adj Gap
Answer cardinalitySingle-answer 888 90.8 6.8 14.4 7.6
Multi-answer 163 54.5 10.4 30.2 19.9
Support sizeSmall support 353 80.5 12.6 26.7 14.2
Medium support 527 87.0 5.2 13.3 8.0
Large support 171 89.2 2.9 7.4 4.5
Table 3: Structural slices for direct LLM baselines atα=
0.3.RandandAns-adjreportpaired∆F1underrandomand
answer-adjacent missingness. Gap is the difference between
Ans-adj and Rand degradation.
Answer-adjacent degradation is most pronounced for
multi-answer questions and small-support graphs. Multi-
answer examples show a 19.9-point gap over random re-
moval, compared with 7.6 points for single-answer exam-
ples.Thegapalsodecreasesassupportsizegrows,from14.2
pointsonsmall-supportgraphsto4.5pointsonlarge-support
graphs. These results show that answer-local evidence loss
isespeciallyharmfulwhenanswersarestructurallyharderto
recover or when alternative support paths are limited.
Discussion
Taken together, the experiments show that incomplete-
knowledgerobustnessisbettercharacterizedbytypeddegra-
dation profiles than by a single aggregate score. The domi-
nanteffectcomesfromanswer-adjacentevidenceloss,while
severity, metric, and structural analyses clarify when this
effect is amplified or preserved. A system can appear ro-
bust under one missingness type while being highly sensi-
tive to another. This is most visible in the contrast between
source-contextandanswer-adjacentremoval.Source-context
removal is often small or even beneficial, suggesting that
some source-side evidence may introduce distraction or in-
creasetheburdenofevidenceselection.Incontrast,answer-
adjacent removal consistently produces the largest degrada-
tion, indicating that systems depend strongly on evidencenear the aligned answer entities. These two effects would
be collapsed by an aggregate missing-evidence score, even
though they imply different failure mechanisms.
Typed profiles also make cross-system comparison more
interpretable.Highercomplete-supportF1doesnotnecessar-
ilyimplystrongerrobustnessunderallformsofmissingness.
Forexample,directLLMbaselinesachievestrongcomplete-
supportperformancebutcanstillshowlargeanswer-adjacent
degradation. Conversely, some trained KGQA models show
negativedegradationundersource-contextremoval,indicat-
ingthattheirbehaviorunderincompleteevidenceisnotcap-
tured by complete-support accuracy alone. This highlights
MissDiag’sdiagnosticroleinrevealinghowsystemsrespond
todifferentevidencelosses,notonlyhowmuchtheiraverage
score changes.
Conclusion
This paper studies incomplete-knowledge KGQA and KG-
RAGevaluationasadiagnosticproblem.Ascoredropunder
incompleteknowledgeisnotdirectlyinterpretablebecauseit
canconflatemissingevidence,systemresponse,andevalua-
tionprotocol.MissDiagaddressesthisambiguitybykeeping
thequestion,goldanswer,sourceentities,andnodeinventory
fixed while applying severity-controlled typed missingness
operators to support edges. Across trained KGQA models,
graphpromptingmethods,KGagents,anddirectLLMbase-
lines, answer-adjacent evidence loss produces the most con-
sistent degradation, whereas source-context removal can be
neutralorbeneficial.Severity,metric,andstructuralanalyses
further show when this typed pattern is preserved or ampli-
fied.Thecentralimplicationismethodological:incomplete-
knowledge robustness should be reported as typed degrada-
tion profiles, rather than as one aggregate score drop.
References
Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013. Se-
manticParsingonFreebasefromQuestion-AnswerPairs. In
Proceedings of the 2013 Conference on Empirical Methods
in Natural Language Processing, 1533–1544.
Cao,S.;Shi,J.;Pan,L.;Nie,L.;Xiang,Y.;Hou,L.;Li,J.;He,
B.; and Zhang, H. 2022. KQA Pro: A Dataset with Explicit
Compositional Programs for Complex Question Answering
over Knowledge Base. InProceedings of the 60th Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 6101–6119.
Chen, L.; Tong, P.; Jin, Z.; Sun, Y.; Ye, J.; and Xiong, H.
2024. Plan-on-Graph:Self-CorrectingAdaptivePlanningof
LargeLanguageModelonKnowledgeGraphs. InAdvances
in Neural Information Processing Systems, volume 37.
Choi, H. K.; Lee, S.; Chu, J.; and Kim, H. J. 2023. Nu-
trea: Neural tree search for context-guided multi-hop kgqa.
Advances in Neural Information Processing Systems, 36:
35954–35965.
Dubey, M.; Banerjee, D.; Abdelkawi, A.; and Lehmann, J.
2019.LC-QuAD2.0:ALargeDatasetforComplexQuestion
Answering over Wikidata and DBpedia. InThe Semantic
Web – ISWC 2019, 69–78.

Gardner, M.; Artzi, Y.; Basmov, V.; Berant, J.; Bogin, B.;
Chen, S.; Dasigi, P.; Dua, D.; Elazar, Y.; Gottumukkala, A.;
et al. 2020. Evaluating Models’ Local Decision Boundaries
viaContrastSets. InFindingsoftheAssociationforCompu-
tational Linguistics: EMNLP 2020, 1307–1323.
Gu, Y.; Kase, S.; Vanni, M.; Sadler, B.; Liang, P.; Yan, X.;
and Su, Y. 2021. Beyond I.I.D.: Three Levels of General-
ization for Question Answering on Knowledge Bases. In
Proceedings of The Web Conference 2021, 3477–3488.
Gu, Y.; and Su, Y. 2022. ArcaneQA: Dynamic Program In-
duction and Contextualized Encoding for Knowledge Base
Question Answering. InProceedings of the 29th Inter-
national Conference on Computational Linguistics, 1718–
1731.
Guo, Q.; Wang, X.; Zhu, Z.; Liu, P.; and Xu, L. 2023. A
Knowledge Inference Model for Question Answering on an
Incomplete Knowledge Graph.Applied Intelligence, 53(7):
7634–7646.
Guu,K.;Lee,K.;Tung,Z.;Pasupat,P.;andChang,M.2020.
RetrievalAugmentedLanguageModelPre-Training. InPro-
ceedings of the 37th International Conference on Machine
Learning, 3929–3938.
Han,J.;Cheng,B.;andWang,X.2020. OpenDomainQues-
tion Answering based on Text Enhanced Knowledge Graph
withHyperedgeInfusion. InFindingsoftheAssociationfor
Computational Linguistics: EMNLP 2020, 1475–1481.
Han, R.; Liu, J.; Bi, H.; Peng, T.; and Liu, L. 2025. SCR:
A Completion-then-Reasoning Framework for Multi-hop
Question Answering over Incomplete Knowledge Graph.
Neurocomputing, 131027.
He,G.;Lan,Y.;Jiang,J.;Zhao,W.X.;andWen,J.-R.2021.
Improving Multi-hop Knowledge Base Question Answering
by Learning Intermediate Supervision Signals. InProceed-
ingsoftheFourteenthACMInternationalConferenceonWeb
SearchandDataMining,553–561.AssociationforComput-
ing Machinery.
Huang, X.; Zhang, J.; Li, D.; and Li, P. 2019. Knowledge
Graph Embedding Based Question Answering. InProceed-
ings of the Twelfth ACM International Conference on Web
Search and Data Mining, 105–113.
Jia, R.; and Liang, P. 2017. Adversarial Examples for Eval-
uating Reading Comprehension Systems. InProceedings
of the 2017 Conference on Empirical Methods in Natural
Language Processing, 2021–2031.
Jiang,J.;Zhou,K.;Dong,Z.;Ye,K.;Zhao,W.X.;andWen,
J.-R. 2023. StructGPT: A General Framework for Large
Language Model to Reason over Structured Data. InPro-
ceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, 9237–9251.
Kim, J.; Kwon, Y.; Jo, Y.; and Choi, E. 2023. KG-GPT: A
GeneralFrameworkforReasoningonKnowledgeGraphsUs-
ing Large Language Models. InFindings of the Association
for Computational Linguistics: EMNLP 2023, 9410–9421.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;etal.2020.Retrieval-AugmentedGenerationforKnowledge-
Intensive NLP Tasks.Advances in Neural Information Pro-
cessing Systems, 33: 9459–9474.
Liu, J.; Shao, P.; Qin, W.; Liu, F.; Yang, Y.; and Hong, R.
2026. DebateoverMixed-knowledge:ARobustMulti-Agent
Reasoning Framework for Incomplete Knowledge Graph
QuestionAnswering.InProceedingsoftheAAAIConference
on Artificial Intelligence, volume 40, 15333–15341.
Liu, L.; Du, B.; Xu, J.; Xia, Y.; and Tong, H. 2022. Joint
Knowledge Graph Completion and Question Answering.
InProceedings of the 28th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, 1098–1108.
Mai, G.; Janowicz, K.; Yan, B.; Zhu, R.; Cai, L.; and Lao,
N.2019. ContextualGraphAttentionforAnsweringLogical
QueriesoverIncompleteKnowledgeGraphs.InProceedings
ofthe10thInternationalConferenceonKnowledgeCapture,
171–178.
Mavromatis, C.; and Karypis, G. 2022. ReaRev: Adaptive
ReasoningforQuestionAnsweringoverKnowledgeGraphs.
InFindingsoftheAssociationforComputationalLinguistics:
EMNLP 2022, 2447–2458.
Perevalov, A.; Yan, X.; Kovriguina, L.; Jiang, L.; Both, A.;
andUsbeck,R.2022.KnowledgeGraphQuestionAnswering
Leaderboard: A Community Resource to Prevent a Repli-
cation Crisis. InProceedings of the Thirteenth Language
Resources and Evaluation Conference, 2998–3007.
Ribeiro, M. T.; Wu, T.; Guestrin, C.; and Singh, S. 2020.
Beyond Accuracy: Behavioral Testing of NLP Models with
CheckList. InProceedingsofthe58thAnnualMeetingofthe
Association for Computational Linguistics, 4902–4912.
Saxena,A.;Kochsiek,A.;andGemulla,R.2022. Sequence-
to-Sequence Knowledge Graph Completion and Question
Answering. InProceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), 2814–2828.
Saxena, A.; Tripathi, A.; and Talukdar, P. 2020. Improv-
ing Multi-hop Question Answering over Knowledge Graphs
using Knowledge Base Embeddings. InProceedings of the
58th Annual Meeting of the Association for Computational
Linguistics, 4498–4507.
Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; van den Berg, R.;
Titov, I.; and Welling, M. 2018. Modeling Relational Data
with Graph Convolutional Networks. InThe Semantic Web,
593–607.
Shi,J.;Cao,S.;Hou,L.;Li,J.;andZhang,H.2021.Transfer-
Net:AnEffectiveandTransparentFrameworkforMulti-hop
Question Answering over Relation Graph. InProceedings
of the 2021 Conference on Empirical Methods in Natural
Language Processing, 4149–4158.
Steinmetz,N.;andSattler,K.-U.2021. WhatisintheKGQA
Benchmark Datasets? Survey on Challenges in Datasets for
QuestionAnsweringonKnowledgeGraphs.JournalonData
Semantics, 10(3): 241–265.
Sui,Y.;He,Y.;Ding,Z.;andHooi,B.2025. CanKnowledge
Graphs Make Large Language Models More Trustworthy?
AnEmpiricalStudyoverOpen-EndedQuestionAnswering.

InProceedings of the 63rd Annual Meeting of the Associa-
tionforComputationalLinguistics(Volume1:LongPapers),
12685–12701.
Sun, H.; Bedrax-Weiss, T.; and Cohen, W. 2019. PullNet:
Open Domain Question Answering with Iterative Retrieval
on Knowledge Bases and Text. InProceedings of the 2019
ConferenceonEmpiricalMethodsinNaturalLanguagePro-
cessingandthe9thInternationalJointConferenceonNatu-
ral Language Processing (EMNLP-IJCNLP), 2380–2390.
Sun, J.; Xu, C.; Tang, L.; Wang, S.; Lin, C.; Gong, Y.; Ni,
L.; Shum, H.-Y.; and Guo, J. 2024. Think-on-Graph: Deep
and Responsible Reasoning of Large Language Model on
KnowledgeGraph. InTheTwelfthInternationalConference
on Learning Representations.
Sun,Q.;Zhang,C.;Hu,Z.;Jin,Z.;Yu,J.;andLiu,L.2023.
Multi-hopQuestionAnsweringoverIncompleteKnowledge
Graph with Abstract Conceptual Evidence.Applied Intelli-
gence, 53(21): 25731–25751.
Sun, Z.; Deng, Z.-H.; Nie, J.-Y.; and Tang, J. 2019. Ro-
tatE: Knowledge Graph Embedding by Relational Rotation
in Complex Space.arXiv preprint arXiv:1902.10197.
Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, É.; and
Bouchard, G. 2016. Complex Embeddings for Simple Link
Prediction. InProceedingsofthe33rdInternationalConfer-
ence on Machine Learning, 2071–2080.
Wang, Y.; Lipka, N.; Rossi, R. A.; Siu, A.; Zhang, R.; and
Derr, T. 2024. Knowledge Graph Prompting for Multi-
DocumentQuestionAnswering. InProceedingsoftheAAAI
Conference on Artificial Intelligence, volume 38, 19206–
19214.
Wen,Y.;Wang,Z.;andSun,J.2024. MindMap:Knowledge
Graph Prompting Sparks Graph of Thoughts in Large Lan-
guage Models. InProceedings of the 62nd Annual Meeting
oftheAssociationforComputationalLinguistics(Volume1:
LongPapers),10370–10388.AssociationforComputational
Linguistics.
Xiong, G.; Bao, J.; and Zhao, W. 2024. Interactive-KBQA:
Multi-turn Interactions for Knowledge Base Question An-
sweringwithLargeLanguageModels. InProceedingsofthe
62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), 10561–10582.
Xiong, W.; Yu, M.; Chang, S.; Guo, X.; and Wang, W. Y.
2019. ImprovingQuestionAnsweringoverIncompleteKBs
with Knowledge-Aware Reader. InProceedings of the 57th
Annual Meeting of the Association for Computational Lin-
guistics, 4258–4264.
Xu,Y.;He,S.;Chen,J.;Wang,Z.;Song,Y.;Tong,H.;Liu,G.;
Zhao, J.; and Liu, K. 2024. Generate-on-Graph: Treat LLM
as both Agent and KG for Incomplete Knowledge Graph
Question Answering. InProceedings of the 2024 Confer-
enceonEmpiricalMethodsinNaturalLanguageProcessing,
18410–18430.
Ye, X.; Xiao, L.; Zhang, C.; and Yamasaki, T. 2024. E-
ReaRev: Adaptive Reasoning for Question Answering over
Incomplete Knowledge Graphs by Edge and Meaning Ex-
tensions. InNatural Language Processing and Information
Systems, 85–95.Ye, X.; Yavuz, S.; Hashimoto, K.; Zhou, Y.; and Xiong, C.
2022. RNG-KBQA: Generation Augmented Iterative Rank-
ing for Knowledge Base Question Answering. InProceed-
ingsofthe60thAnnualMeetingoftheAssociationforCom-
putationalLinguistics(Volume1:LongPapers),6032–6043.
Yih, W.-t.; Chang, M.-W.; He, X.; and Gao, J. 2015. Se-
mantic Parsing via Staged Query Graph Generation: Ques-
tion Answering with Knowledge Base. InProceedings of
the 53rd Annual Meeting of the Association for Computa-
tionalLinguisticsandthe7thInternationalJointConference
on Natural Language Processing (Volume 1: Long Papers),
1321–1331.
Yu, D.; Gu, Y.; Xiong, C.; and Yang, Y. 2023. CompleQA:
Benchmarking the Impacts of Knowledge Graph Comple-
tion Methods on Question Answering. InFindings of the
Association for Computational Linguistics: EMNLP 2023,
12748–12755.
Zan,D.;Wang,S.;Zhang,H.;Zhou,K.;Wu,W.;Zhao,W.X.;
Wu, B.; Guan, B.; and Wang, Y. 2022. Complex Question
AnsweringoverIncompleteKnowledgeGraphasN-aryLink
Prediction.In2022InternationalJointConferenceonNeural
Networks (IJCNN), 1–8.
Zhang,L.;Jiang,Z.;Chi,H.;Chen,H.;ElKoumy,M.;Wang,
F.; Wu, Q.; Zhou, Z.; Pan, S.; Wang, S.; and Ma, Y. 2025.
Diagnosing and Addressing Pitfalls in KG-RAG Datasets:
Toward More Reliable Benchmarking. InNeurIPS 2025
Datasets and Benchmarks Track.
Zhang,L.;Jiang,Z.;Chi,H.;Chen,H.;Elkoumy,M.;Wang,
F.; Wu, Q.; Zhou, Z.; Pan, S.; Wang, S.; et al. 2026. Diag-
nosingandAddressingPitfallsinKG-RAGDatasets:Toward
MoreReliableBenchmarking.AdvancesinNeuralInforma-
tion Processing Systems, 38.
Zhao,F.;Li,Y.;Hou,J.;andBai,L.2022. ImprovingQues-
tion Answering over Incomplete Knowledge Graphs with
Relation Prediction.Neural Computing and Applications,
34(8): 6331–6348.
Zhou, D.; Zhu, Y.; Wang, X.; Zhou, H.; He, Y.; Chen, J.;
Staab,S.;andKharlamov,E.2026. WhatBreaksKnowledge
Graph based RAG? Benchmarking and Empirical Insights
into Reasoning under Incomplete Knowledge. InProceed-
ings of the 19th Conference of the European Chapter of the
Association for Computational Linguistics (Volume 1: Long
Papers), 2522–2538.
# Context-Picker: Dynamic context selection using multi-stage reinforcement learning

**Authors**: Siyuan Zhu, Chengdong Xu, Kaiqiang Ke, Chao Yu

**Published**: 2025-12-16 14:52:11

**PDF URL**: [https://arxiv.org/pdf/2512.14465v1](https://arxiv.org/pdf/2512.14465v1)

## Abstract
In long-context question answering (LCQA), determining the optimal amount of context for a given query is a significant challenge. Including too few passages may omit critical information, while including too many can introduce noise and reduce the quality of the answer. Traditional approaches, such as fixed Top-$K$ retrieval and single-stage reranking, face the dilemma of selecting the right number of passages. This problem is particularly pronounced for factoid questions, which often require only a few specific pieces of evidence. To address this issue, we introduce \emph{Context-Picker}, a reasoning-aware framework that shifts the paradigm from similarity-based ranking to minimal sufficient subset selection. Context-Picker treats context selection as a decision-making process optimized via a human-inspired, two-stage reinforcement learning schedule: a \emph{recall-oriented} stage that prioritizes the coverage of reasoning chains, followed by a \emph{precision-oriented} stage that aggressively prunes redundancy to distill a compact evidence set. To resolve reward sparsity, we propose an offline evidence distillation pipeline that mines "minimal sufficient sets" via a Leave-One-Out (LOO) procedure, providing dense, task-aligned supervision. Experiments on five long-context and multi-hop QA benchmarks demonstrate that Context-Picker significantly outperforms strong RAG baselines, achieving superior answer accuracy with comparable or reduced context lengths. Ablation studies indicate that the coarse-to-fine optimization schedule, the redundancy-aware reward shaping, and the rationale-guided format all contribute substantially to these gains.

## Full Text


<!-- PDF content starts -->

Context-Picker: Dynamic context selection using multi-stage
reinforcement learning
Siyuan Zhu
School of Computer Science and Engineering
Sun Yat-sen University
zhusy58@mail2.sysu.edu.cnChengdong Xu
School of Computer Science and Engineering
Sun Yat-sen University
xuchd6@mail2.sysu.edu.cn
Kaiqiang Ke
School of Computer Science and Engineering
Sun Yat-sen University
kekq@mail2.sysu.edu.cnChao Yu∗
School of Computer Science and Engineering
Sun Yat-sen University
yuchao3@mail.sysu.edu.cn
Abstract
In long-context question answering (LCQA), determining the optimal amount of context for a given
query is a significant challenge. Including too few passages may omit critical information, while
including too many can introduce noise and reduce the quality of the answer. Traditional approaches,
suchasfixedTop- Kretrievalandsingle-stagereranking,facethedilemmaofselectingtherightnumber
ofpassages. Thisproblemisparticularlypronouncedforfactoidquestions,whichoftenrequireonlya
fewspecificpiecesofevidence. Toaddressthisissue,weintroduceContext-Picker,areasoning-aware
frameworkthatshiftstheparadigmfromsimilarity-basedrankingtominimalsufficientsubsetselection.
Context-Picker treats context selection as a decision-making process optimized via a human-inspired,
two-stage reinforcement learning schedule: arecall-orientedstage that prioritizes the coverage of
reasoning chains, followed by aprecision-orientedstage that aggressively prunes redundancy to
distill a compact evidence set. To resolve reward sparsity, we propose an offline evidence distillation
pipeline that mines "minimal sufficient sets" via a Leave-One-Out (LOO) procedure, providing
dense,task-alignedsupervision. Experimentsonfivelong-contextandmulti-hopQAbenchmarks
demonstrate that Context-Picker significantly outperforms strong RAG baselines, achieving superior
answer accuracy with comparable or reduced context lengths. Ablation studies indicate that the
coarse-to-fine optimization schedule, the redundancy-aware reward shaping, and the rationale-guided
format all contribute substantially to these gains.
1 Introduction
Retrieval-AugmentedGeneration(RAG)hasbecomeastandardparadigmforextendingLargeLanguageModels(LLMs)
beyondtheirparametricknowledge,especiallyonknowledge-intensiveandlong-contextquestionanswering(LCQA)
tasks[Lewisetal.,2021,Guuetal.,2020,Izacardetal.,2022]. Byretrievingpassagesfromanexternalcorpusand
conditioning generation on them, RAG mitigates hallucination and enables access to up-to-date or domain-specific
information. Inpractice,mostsystemsadoptasimplefixedTop- Kstrategy: aretrieverrankscandidatepassagesand
thetop- Kareconcatenated andfed tothe generator. However,the coredesign questionofhowmuchexternalcontext
should be retrieved for a given query remains largely underexplored. When Kis too small, the model may miss critical
evidence and break multi-hop reasoning chains, while an overly large Kintroduces many weakly related passages,
increasinginferencecostanddegradinganswerqualitythroughdistractors,attentiondilution,andthe“lost-in-the-middle”
phenomenon where LLMs under-utilize information placed in the middle of long prompts [Liu et al., 2023]. Moreover,
our experiments in Figure 1 show that increasing retrieval depth monotonically improves recall but leaves accuracy
almost unchanged, which consistent with recent observations on long-context limitations in RAG [Jin et al., 2024]. This
∗Corresponding author.arXiv:2512.14465v1  [cs.AI]  16 Dec 2025

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
suggeststhatcontexthandlinginLCQAshouldbeviewednotpurelyasarankingproblem,butasasubsetselection
problem: for each query, the system should construct a compact, query-specific evidence set that is sufficient for
answering the question, rather than a long prefix of a ranked list.
0 25 50 75 100 125 150 175 200
Top K0.40.50.60.70.80.91.0Score
Accuracy and Recall vs Top K
Accuracy
Recall
Figure 1: Accuracy vs. retrieval depth (Top- K) in
a standard RAG pipeline. Recall increases with K,
butansweraccuracydoesnotimprove,whichisalso
reportedinrecentlongcontextstudies[Jinetal.,2024].Recent works on retrieval-augmented generation solve this
problem from two main directions. One line of methods
strengthens theretrieval pipelinewhile keeping the context
size essentially fixed. Classical sparse retrievers and dense
dual-encoderretrieversimprovetherecallandcoarserankingof
candidatepassages,andareoftencoupledwithcross-encoder
orsequence-to-sequencererankersthatrefinethefine-grained
orderingofdocuments[Robertsonetal.,2009,Karpukhinetal.,
2020, Xiong et al., 2020, Nogueira et al., 2020]. More recently,
LLM-based rerankers score and prune contexts using query-
aware, list-aware,orgenerator-awaresignals[Sunetal.,2024,
Chen et al., 2025, Drozdov et al., 2023, Wang et al., 2024,
Dengetal.,2025]. Theseapproachesareeffectiveatpromoting
highlyrelevantpassagesintherankedlistanddemotingobvious
distractors, but the generator still typically consumes either a
fixed top- Kprefix or a set obtained by hand-crafted thresholds,
so the fundamental trade-off between missing evidence and
accumulating noise remains.
A complementary line of work explicitlyadapts the number
of retrieved passages. Adaptive-RAG routes each query to
no-retrieval, single-step, or iterative RAG pipelines based on a learned complexity classifier [Jeong et al., 2024],
while adaptive- kmethods select the cutoff Kfrom the similarity-score distribution of the retrieved candidates without
additional model tuning or extra LLM calls [Taguchi et al., 2025]. Although such methods alleviate the mismatch
betweensimpleandcomplexqueries,theystillrelyonheuristicdecisionrulesoverper-passagesimilarityscores,anddo
not directly optimize for aminimal sufficientevidence subset under a given token budget.
To move beyond fixed heuristics, reinforcement learning (RL) has recently been explored as a way to optimize retrieval
and selection policies directly from task feedback while keeping test-time inference to a single policy forward pass.
DynamicRAG models thererankeras an RLagentoverdocument sequences and usesLLM-judgedanswer quality as
reward to jointly adjust both the order and the number of retrieved documents [Sun et al., 2025]. Beyond reranking,
recentRL-basedsystemssuchasMemory-R1 andrelatedmemoryagents framelong-termmemorymanagementand
retrieval decisions as RL problems, training policies to decide what to store, update, or retrieve in order to support
downstream QA and dialogue [Yan et al., 2025]. RL has also been applied to conversational query reformulation
and retrieval alignment and to broader agentic RAG frameworks that optimize multi-step retrieval and reasoning
trajectories [Zhu et al., 2025, Xiong et al., 2025, Jiang et al., 2025]. However, existing RL-style approaches still suffer
from largelytrajectory-level and sparserewards, which makes it difficult to assign credit to individual passages or
penalizeredundancy,andtheyaretypicallytrainedtoimprovelist-wiserankingqualityormemoryoperationsrather
than to identify a minimal evidence subset that preserves answerability under a fixed input budget.
Toaddressthesechallenges,weintroduceContext-Picker,areasoning-awareframeworkthatfundamentallyshiftsthe
context selection paradigm from similarity-based ranking to minimal sufficient subset selection. Instead of treating
retrieval as a sorting problem, Context-Picker formulates it as a decision-making process, learning to construct a
variable-length evidence set that is strictly necessary for answering the query. Central to our approach is a human-
inspired Coarse-to-Fine optimization strategy implemented via a two-stage reinforcement learning schedule. In Stage I
(Recall-Oriented),thepickeristrainedtomaximizeinformationwitharelaxedredundancymargin,sothatallpotentially
relevant reasoning chains—especially those spanning multiple passages—are captured. In Stage II (Precision-Oriented),
theobjectiveshiftstorefinement: thepolicylearnstopruneredundantorweaklyrelevantpassages,distillingthecontext
intoacompact,noise-freesubsetwithoutcompromisinganswerability. Tostabilizetrainingandalleviaterewardsparsity,
we introduce an offline evidence distillation pipeline that uses a generator–judge loop with greedy Leave-One-Out
(LOO) pruning to mine “minimal sufficient” evidence sets from raw documents. These distilled sets provide dense,
task-alignedsupervision,enablingthepolicytolearnthecontributionofeachevidencepiece. Extensiveexperimentson
five long-context and multi-hop QA benchmarks demonstrate that Context-Picker significantly outperforms strong RAG
baselines, achieving superior answer accuracy with comparable or reduced context lengths.
Contributions.Our main contributions are summarized as follows:
2

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
•We proposeContext-Picker, a reasoning-aware context picker trained with a two-stage reinforcement learning
schemeandredundancy-awarerewardshaping. Thepickerjointlydecideswhichpassagestokeepandhow
manyto include, with a recall-oriented stage for high-coverage picking and a precision-oriented stage for
aggressive compression, explicitly addressing the limitations of fixed top-Kselection in long-context QA.
•Weintroduceanofflineevidenceminingpipelinethatminesgreedilyminimalsufficientevidencesetsviaa
generator–judge loop and a leave-one-out pruning procedure, providing high-quality, task-aligned supervision
for training the picker.
•We conduct extensive experiments on five long-context and multi-hop QA benchmarks, showing that Context-
PickerimprovesLLM-as-judgeaccuracyoverstrongRAGbaselinesonfourdatasetsandachievesfavorable
accuracy–efficiencytrade-offsontheremainingone,withablationsvalidatingtheimpactofeachkeycomponent.
2 Preliminaries
2.1 Retrieval-Augmented Generation
We follow the standard retrieval-augmented generation (RAG) formulation [Lewis et al., 2021, Guu et al., 2020, Izacard
etal.,2022]. Let Ddenotealargenon-parametriccorpus(e.g.,Wikipediaoralong-termmemorystore). Inlong-context
QA, each document in Dis first segmented into shorter passages (“chunks”), which serve as the retrieval units. Given a
queryq, a retriever operates over these passages and returns acandidate poolof at mostK maxpassages
C(q) ={c 1, c2, . . . , c N}, N≤K max,(1)
optionally refined by a reranker that reorders C(q)according to query-specific relevance [Karpukhin et al., 2020, Xiong
et al., 2020, Nogueira et al., 2020]. Unless otherwise stated, we use C(q), or simply Cwhen the query is clear from
context, to denote this (re)ranked candidate pool. Later, when formulating Context-Picker, we additionally attach a
unique identifier to each passagec jand writeC={(c j,idj)}N
j=1for convenience.
Context selection.Given C(q), the system must choose a variable-lengthsupport set S ⊆ C(q) to feed into the
generator under an input budgetB. We view this as a subset selection problem that trades off task utility and brevity:
S⋆∈arg max
S⊆C(q)
U(q,S)−λ·Len(S)
s.t.Tok(q,S)≤B,(2)
where U(q,S)isataskutility, Len(S)measuresthesizeofthesupportset, λ≥0controlsthequality–brevitytrade-off,
andTok(q,S) counts input tokens. A common baseline uses a fixed top- KprefixS={˜c 1, . . . ,˜c K}for all queries,
which under- or over-includes context depending on query difficulty and can suffer from “lost in the middle” effects in
longprompts[Liuetal.,2023,Jinetal.,2024]. Adaptivestrategiesinsteadlearnapolicy πϕ(S |q,C(q)) thatjointly
decideswhichpassages to keep andhow manyto include [Sun et al., 2025, Deng et al., 2025]. Context-Picker builds on
this formulation and learns a reasoning-aware policy under a token budget.
Response generation and utility.Given a support set S, we construct a prompt x= Tpl(q,S) by concatenating
instructions,thequery,andtheselectedpassages,anduseagenerator Gtodefineaconditionaldistributionoveranswers:
pθ(y|x) =G(x), from which we decode an answer ˆy. We instantiate the utility U(q,S)in Eq.(2)either with
exact-matchaccuracyorwithanLLM-as-judgescorethatevaluatesthesemanticcorrectnessof ˆyw.r.t.thereference
answer.
2.2 Group Relative Policy Optimization (GRPO)
We view evidence picking as a policy optimization problem. Let odenote an observation which consists of a query
qand its candidate pool C(q), and let adenote a discrete action (a set of picked passage IDs). A stochastic policy
πϕ(a|o)with parametersϕinduces the objective
J(ϕ) =E o∼Dtrain, a∼π ϕ(·|o)
R(a, o)
,(3)
whereR(a, o)is a task-specific reward.
We used Group Relative Policy Optimization (GRPO) [Shao et al., 2024] to optimize our training goal. For each
observation o(e.g.,aqueryanditscandidatepool),thepolicy πϕ(withafrozenreferencepolicy πϕold)samplesagroup
ofGcandidate actions{a i}G
i=1∼πϕold(· |o),and each action receives a scalar rewardR i=R(a i, o).
3

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
Figure2: OverviewoftheContext-Pickerframework. Thepipelineconsistsoftwoparts: (1)OfflineEvidenceMining,
whereagenerator–judgeloopemploysaLeave-One-Out(LOO)strategytomineminimalsufficientevidencesets( Sgold)
as supervision; and (2)Context-Picker Pipeline, where the picker policy ( πθ) learns to select evidence from retrieved
candidates ( C). The training follows aCoarse-to-Fineschedule: Stage I optimizes for high recall to capture reasoning
chains, while Stage II tightens the redundancy penalty to distill a compact support set, guided by GRPO updates.
Thegroup-normalizedadvantageforthe i-thactionis ˆAi=Ri−mean 
{Rj}G
j=1
std 
{Rj}G
j=1
+ϵ,where ϵisasmallconstantfornumerical
stability.
The probability ratio is defined as
ri(ϕ) =πϕ(ai|o)
πϕold(ai|o).(4)
Our GRPO objective with decoupled, asymmetric clipping is
JGRPO (ϕ) =Eo∼D,{a i}G
i=1∼πϕold(·|o)"
1
GGX
i=1min
ri(ϕ)ˆAi,
clip 
ri(ϕ),1−ϵ low,1 +ϵ highˆAi#
−β·KL(π ϕ∥πϕold),(5)
whereϵ low, ϵhigh>0control the asymmetric clipping range andβ≥0controls the KL regularization strength.
3 Context-Picker
In this section, we presentContext-Picker, our reinforcement learning–based context picker. An overview of the
frameworkisshowninFigure2. Wefirstformulatecontextpickingasasingle-stepMarkovdecisionprocess(MDP)
in Section 3.1. We then describe the overall framework, which consists of two components: (i) anoffline evidence
miningpipeline(Section3.2)thatdistillsminimalsufficientevidencesetsfromrawdocuments,and(ii)amulti-stage
reinforcement learningprocedure (Section 3.3) that trains a picker policy with a recall-oriented stage followed by a
precision-oriented stage. Finally, we detail how the learned picker is integrated with the downstream generator at
inference time, and summarize the resulting inference pipeline in Algorithm 3.3.
4

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
3.1 Problem Formulation
We cast context picking as a single-step decision problem. For each query, a retriever first returns a candidate pool
of passages C={(c 1,id1),(c 2,id2), . . . ,(c N,idN)},where cjis a candidate passage and idjis its unique identifier.
Together with the queryqand a stage-specific instruction promptp i, this defines the observationo=⟨p i, q,C⟩.
The action space consists of subsets of candidate identifiers. Concretely, the policy outputs a structured response
output=⟨r, a⟩, where ris a rubric-guided natural-language rationale and a={id i1,idi2, . . . ,id ik} ⊆ {id 1, . . . ,id N}
is the selected subset of IDs. The corresponding support set fed to the downstream generator is
S={c j|(cj,idj)∈ C,id j∈a},
which ensures end-to-end consistency between the picker and the generator.
Weconstraintheactionspacesothat aisavalid,duplicate-freesubsetofcandidateIDs;malformedorout-of-range
selectionsaretreatedasinvalidactionsandreceiveformatpenaltiesinthereward. Thisdiscretesubset-basedformulation
matchesthenatureofevidencepickingandservesastheMDPonwhichweapplyGRPO-basedtraininginthefollowing
subsections.
Given an observation–action pair (o, a)with support set Sand an offline-mined golden evidence set Sgold(Section 3.2),
the stage-ireward takes the abstract form
Ri(o, a) = Cov(S,S gold)|{z}
coverage−Redun i(S,Sgold)| {z }
redundancy penalty−γI
¬format_valid(S)
,(6)
where Covmeasureshowwell Scovers thegolden evidence, Redunpenalizesover-longor redundantselections ina
stage-dependent manner, and the last term discourages invalid outputs. In Section 3.3 we instantiate (6)with a concrete
design based onCovand a normalized redundancy penalty (cf. Eq. (8)).
3.2 Data Curation
Offline evidence mining.To construct high-quality training data for Context-Picker, we introduce an offline evidence
distillation pipeline. Eachdocument Disfirstsegmentedintosemanticallycoherentchunks viasemanticchunking,
whichensuresthateachchunkformsalocallyconsistentunitwhilepreservingcontextualcontinuity. Thiscorrespondsto
theOfflineEvidenceMiningmoduleontheleftsideofFigure2,andtheoverallprocedureissummarizedinAlgorithm1.
For each query–answer pair (q, a), we perform retrieval using BM25 on the concatenation of the query and answer, i.e.,
on[q;a]againstthechunkeddocument. Thetop- kretrievedchunksconstituteaninitialcandidateset Scand. Wethenrun
ananswer-judgepipelineon Scand: agenerator Gproducesaresponse ˆaconditionedon (q,Scand),andanLLM-based
judgeJdecideswhether ˆasemanticallymatchesthegoldanswer a. IfJdeems Scandinsufficient(i.e.,theansweris
judgedasincorrect),wediscardthispair,sincetheretrievedevidencedoesnotsupportacorrectanswerevenbefore
pruning.
For the remaining pairs, we greedily prune redundant chunks via a leave-one-out (LOO) procedure. We initialize
Ssuf← Scandand iterate over chunks c∈ Ssuf. For each c, we temporarily remove it to form S′=Ssuf\ {c}, run the
same answer-judge pipeline on (q,S′), and obtain a new judge decision. If Jstill marks the answer as correct, we
treatcasredundantandpermanentlydropit,updating Ssuf← S′. WerepeatthisLOOpruninguntilnochunkcanbe
removed without flipping the judge decision from correct to incorrect. The resulting set Ssufis thus a greedily minimal
sufficient evidence set with respect to the judge: every remaining chunk is empirically necessary in the sense that
removing any of them would cause the model to fail the judge. We treat Ssufas the golden evidence supervision for
training Context-Picker.
Dataaugmentation.Consideredthatmostlong-contextQAorretrievaldatasetscontainrelativelyfewuniquequeries,
we introduce lightweight query rewriting to enhance data diversity. For each original query q, we generate five
semanticallyequivalentbutlexicallydiversereformulations {q′
i}5
i=1usingalanguagemodel. Theserewritespreserve
themeaningoftheoriginalquerywhilevaryinginphrasingandfocus,whichhelpsimprovelinguisticdiversityand
reduces overfitting during RL training. During data partitioning, all rewrites of the same query are assigned to the same
subset to prevent data leakage between training and evaluation data.
Thiscurated dataset,consistingofgolden evidencepicksand diversequeryformulations, servesasthe foundationfor
the reinforcement learning phase of Context-Picker.
3.3 Multi-stage Reinforcement Learning
5

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
Algorithm 1:Offline Evidence Mining
Input:DocumentD; queryq; gold answera; retriever
R; encoderf emb; generatorG; answer judgeJ;
top-k.
Output:Minimal sufficient setS suf.
C ←Chunk(D;f emb)
x←[q;a]
Scand←RetrieveTopK(x,C;R, k)
ˆa← G(q,S cand)
rfull← J(q,ˆa, a)
ifrfull= 0then
return∅
end
Ssuf← Scand
changed←True
whilechangeddo
changed←False
foreachc∈ S sufdo
S′← Ssuf\ {c}
ˆa′← G(q,S′)
r′← J(q,ˆa′, a)
ifr′= 1then
Ssuf← S′
changed←True
end
end
end
returnS sufUsing the curated training set from Section 3.2, this
subsection describes the two-stage policy optimization
componentofContext-Picker,whichcorrespondstothe
rightpartofFigure 2;thedetailedGRPO-basedtraining
loop is given in Algorithm 2.
In long-context question answering (LCQA), the chal-
lengesofwhatevidencetopickandhowmuchevidence
to pickare essentially a coupled combinatorial optimiza-
tionproblem. Selectingtoofewpiecesofevidencerisks
missing key reasoning hops, while selecting too many
introducesnoiseandattentiondilution. Staticstrategies,
such asfixed Top- Ksamplingor single-stage reranking,
struggletosimultaneouslyensurerecallsufficiencyand
input compactness.
As is shown in Figure 2, we decouple this problem into
two training stages:a recall-oriented stagethat empha-
sizescomprehensiveevidencecoverage,andarefinement-
orientedstagethatfocusesonminimalsufficientselection.
StageI:Recall-orientedstrategyoptimization.StageI
is designed to learn ahigh-recallpicking behavior that
prioritizesinformationcompleteness. Inoursetting,the
downstream generator can answer a query correctly as
long as the selected context set contains the key evidence
that supports the reasoning chain. We formalize this
notionviatheoffline-minedminimalsufficientevidence
setSgold(Section3.2),whichapproximatesthesmallest
subset that preserves answerability under an LLM-based judge.
StageIthusencouragesthepolicytomaximize Cov(S,S gold)witharelaxedredundancytolerance red1(Eq.8),allowing
moderate over-selection. This is crucial for multi-hop QA: missing even a single hop in the evidence chain can cause a
failure,whereasincludingafewextrapassagesisoftenharmlessatthisstage. Byemphasizingcoverageandusinga
looseredundancymargin, StageIpreventsprematurepruning andimprovesexplorationoverthecombinatorial subset
space, yielding a robust high-recall initialization for later compression.
StageII:Refinement-orientedstrategyoptimization.StageIItargetsinputconcisenesswhilepreservingsufficiency,
i.e., convergingtoaminimalsufficient evidenceset. Starting fromthe high-recallpolicylearnedinStage I,we tighten
theredundancymarginto red2<red 1andstrengthentheredundancypenaltyinthereward(Eq.8),sothepolicyis
explicitly discouraged from keeping passages that do not improve answerability. Intuitively, Stage II pushes the picker
toward solving a constrained compression problem:
min
S⊆C(q)|S|s.t.U(q,S) = 1,(7)
where U(q,S)is approximated during training by the distilled supervision Sgoldand instantiated as coverage-plus-
redundancy shaping in Eq. (8). Operationally, Stage II encourages the policy to keep setsthat (i) retain near-complete
coverage of Sgold(high recall), yet (ii) eliminate redundant, repetitive, or weakly relevant passages so as to reduce
distractors and mitigate long-context degradation. As a result, the learned picker progressively shifts from arecall-
sufficientregime to aprecision-sufficientregime, producing compact evidence subsets that maximize informativeness
under a fixed token budget.
The reward function is defined as:
Ri=

Cov(S,S gold)−γ·max
0,|S|−|S gold|−red i
|Sgold|+red i
,if format_valid(S)and|S| ≤ |S gold|+red i,
0,if format_valid(S)and|S|>|S gold|+red i,
−1.0,if not format_valid(S),(8)
whereiis training stage andCov(S,S gold) =|S∩Sgold|
|Sgold|. The reward logic follows three principles:
6

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
Algorithm 2:Two-stage GRPO training of Context-Picker
Input:Training setD={(q,C,S gold)}; initial policyπ θ; stage prompts{p 1, p2}; redundancy margins
{red 1,red 2}; group sizeK; iterations{T 1, T2}.
Output:Trained picker policyπ θ.
Initialize reference policyπ θold←π θ.
fori∈ {1,2}do
fort= 1toT ido
Sample a mini-batchB ⊂ D.
Initialize an empty set of GRPO training examplesG ←∅.
foreach(q,C,S gold)∈ Bdo
Construct observationo← ⟨p i, q,C⟩.
Sample a group ofKactions{S 1, . . . ,S K}fromπ θold(· |o).
For eachS j, compute rewardr j←R 
Sj, o;Sgold,redi
using Eq. (8).
Add the group 
o,{S j, rj}K
j=1
intoG.
Update policy parametersθusing GRPO onGaccording to Eq. (5).
πθold←π θ
returnπ θ
•When the output format is valid and the number of selected items does not exceed the “gold standard +
redundancy margin,” the reward is determined by recall rate with a redundancy penalty proportional to
oversampling.
•When the selection exceeds the redundancy margin, the reward is set to zero, discouraging excessive evidence
inclusion.
•When the output format is invalid, a fixed penalty of−1.0is applied to enforce structural correctness.
Progressive redundancy compression.The key distinction between the two stages lies in the dynamic compression
of the redundancy margin red. Stage I employs a relaxed margin red1to tolerate redundancy for completeness, whereas
Stage II tightens the threshold to red2, forcing the policy to eliminate redundant evidence while maintaining high recall.
This“loose-to-tight”marginadaptationachievesasmoothoptimizationfromrecallsufficiencytoinputcompactness,
enabling a Pareto-optimal trade-off between comprehensiveness and efficiency in LCQA.
Stagetransitionandschedule.WeimplementthetwostagesasconsecutiveGRPOphasesoverthesamecurated
dataset. In Algorithm 2, the hyperparameters T1andT2control the number of GRPO update steps spent in the
recall-orientedStageIandtherefinement-orientedStageII,respectively. Inpractice,wefirsttrainthepickerwiththe
StageIreward(largerredundancymargin red1)untilthevalidationrewardcurveplateaus,andthenswitchtoStageIIby
continuing training from the Stage I checkpoint with the tighter margin red2. We found that Context-Picker is robust to
the exact split between T1andT2as long as Stage I is given enough updates to learn a high-recall policy; the resulting
training dynamics for both stages are shown in Figure 3.
Inference.At test time, Context-Picker runs in a single-pass retrieve–pick–generate pipeline. Given a question
qand a long document D, we first segment Dinto semantically coherent chunks using the same chunker as in
training: C ←Chunk(D;f emb).Wethen builda candidatepool byretrieving themost relevantchunks to q, optionally
truncatingthepoolsothatthepickerinputfitswithinabudget: Ccand←TopSim(q,C;B). Next,weconstructthepicker
observation o=⟨p test, q,Ccand⟩and sample the picker output from the learned policy πθ:{r, S} ∼π θ(· |o),where ris
a rubric-guided rationale and Sis the set of selected chunk identifiers. The final evidence set is obtained by filtering the
candidate pool by the selected IDs, Cpick← {c j∈ Ccand:idj∈S},and the downstream generator produces the answer
conditioned on the picked evidence: ˆa← G(q,C pick).For evaluation, we additionally report an LLM-as-judge score by
comparingˆaagainst the reference answer (Section 4).
4 Experiments
4.1 Experimental Setup
Datasets.We evaluate Context-Picker on five knowledge-intensive QA benchmarks that require reasoning over long
ormulti-hopcontexts: (1)LoCoMo[Maharanaetal.,2024],whichcontainsextremelylongmulti-sessionconversations
7

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
0 5 10 15 20 25 30
Training Step0.60.70.80.91.01.1Reward
Stage 1 Training
Train Reward
Validation Reward
0 5 10 15 20 25 30
Training Step0.40.50.60.70.80.91.0Reward
Stage 2 Training
Train Reward
Validation Reward
Figure3:TrainingdynamicsofContext-PickerusingGRPO.Thecurvesvisualizetheaveragerewardtrajectories
ontrainingandvalidationsetsduring(Left)theRecall-OrientedStageIand(Right)thePrecision-OrientedStageII.
Bothstagesexhibitstableconvergenceandanarrowgapbetweentrainingandvalidationrewards,indicatingthatthe
policy effectively learns to balance evidence coverage and compactness without overfitting.
and tests long-term conversational memory; (2)MultiFieldQA[Jiang et al., 2024], a long-context QA dataset with
diverse domains and relatively factoid-style questions; (3)HotpotQA[Yang et al., 2018], a classic multi-hop QA
benchmarkoverWikipedia;(4)2WikiMQA[Hoetal.,2020],amulti-hopQAdatasetrequiringreasoningacrosstwo
Wikipedia articles; and (5)MuSiQue[Trivedi et al., 2022], which decomposes multi-hop questions into compositional
single-hop subquestions. For datasets in LongBench [Bai et al., 2024] that do not come with ground-truth evidence
annotations, we apply the offline evidence mining procedure in Algorithm 1 to construct training labels. Concretely, we
firstperformsemanticchunkingovereachlongdocumentusing text-embedding-ada-002 withasimilaritythreshold
of0.75, and then mine sufficient and golden evidence sets for each(q, a)pair.
Models and baselines.Unless otherwise specified, Context-Picker is instantiated withQwen3-8Bas the picker
backbone. Foranswergeneration,weuseQwen3-32Basthegeneratormodel,andadoptGPT-4o-miniasanLLM-
as-judgeevaluator. Concretely,givenaquestion qanditscandidatecontexts,thepickerselectsasubsetofevidence;
thegeneratorthenproduces ananswerconditionedon qandtheselectedevidence; finally,thejudgemodel scoresthe
predictedansweragainstthereference. Wedeliberatelyusedifferentmodelfamiliesforgenerationandevaluationto
mitigate overestimation bias when a model family evaluates its own outputs [Panickssery et al., 2024].
As baselines, we consider: (i) anon-retrieval LLM(Qwen3-8B) that directly consumes the raw document by
concatenating qwith as much of the context as fits into its input window, without any retrieval or selection module; and
(ii)avanillaRAGpipeline,wherearetrieverreturnstop- Kpassagesthataredirectlyconcatenatedandfedintothe
generator. For RAG we employ a strong dense retriever and report results for K∈ {5,10} (andK= 100 on LoCoMo),
which roughly match the average number of passages selected by Context-Picker under our token budget.
Evaluation protocol.Traditional metrics such as exact match (EM) and F1 are known to be brittle for free-form
answers. Forexample,theanswers“Thecatisonthemat.” and“Acatrestsonamat.” conveyessentiallythesame
meaningbutwouldreceivealowEM/F1scoreduetolexicaldifferences,whereas“Thecatisonthemat.” and“The
dog is on the mat.” share substantial n-gram overlap while being factually incompatible. Following recent work on
LLM-based evaluation [Gu et al., 2025], we thus adopt anLLM-as-judgeprotocol as our primary metric. Given a
questionq, a reference answera⋆, and a predicted answerˆa, a judge model returns a binary correctness label:
Judgeans(q, a⋆,ˆa)∈ {0,1},
based on a rubric that checks semantic equivalence to a⋆and penalizes hallucinations or contradictions. We report the
fraction of examples for which the judge predicts correctness, referred to asJudge Acc.
4.2 Main Results
Table 1 summarizes the main results across the five benchmarks.
ComparisonwithLLM-onlyandRAGbaselines.Acrossalldatasets,bothstagesofContext-Pickersubstantially
outperform the non-retrieval LLM baseline, confirming that external evidence is crucial for long-context and multi-hop
8

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
Method LoCoMo MultiFieldQA HotpotQA 2WikiMQA MuSiQue
LLM (Qwen3-8B, no retrieval) 0.566 0.833 0.661 0.389 0.280
RAG (Qwen3-8B)0.622
(TopK=100)0.857
(TopK=5)
0.857
(TopK=10)0.597
(TopK=5)
0.700
(TopK=10)0.525
(TopK=5)
0.560
(TopK=10)0.340
(TopK=5)
0.390
(TopK=10)
Context-Picker, Stage 1 (Qwen3-8B) 0.6810.8730.741 0.621 0.476
Context-Picker, Stage 2 (Qwen3-8B)0.7060.8250.747 0.702 0.522
Table1:Mainresultsonknowledge-intensiveQAbenchmarks.WereportJudgeAcc(higherisbetter). Bestper
column is inbold.
QAandthatsimplyrelyingonparametricknowledgeisinsufficientinthesesettings. Eventherecall-orientedStageI,
which tolerates some redundancy, already yields sizable gains over the plain LLM.
When compared under comparable evidence budgets, Context-Picker also brings consistent improvements over the
vanilla RAG pipeline on most datasets. On LoCoMo, HotpotQA, 2WikiMQA, and MuSiQue, Stage 2 delivers the best
overall performance, exceeding the strongest RAG baseline by +4–18points in Judge Acc. On MultiFieldQA, the
recall-oriented Stage 1 slightly surpasses RAG (0.873 vs. 0.857), while Stage 2 trades a small drop in accuracy (0.825)
for more compact inputs. These results suggest that, beyond a strong retriever, adaptively decidingwhichpassages
to keep andhow manyto include per query is beneficial: Context-Picker improves answer quality without simply
increasing the number of passages and often reduces prompt overhead.
Effect of the two-stage schedule.The two-stage training scheme yields a clear pattern. Stage I, which uses a relaxed
redundancy margin and emphasizes high recall, is particularly helpful on datasets where evidence is dispersed or
conversationsarelong. StageII,whichtightenstheredundancypenaltytofavorminimalsufficientsets,furtherimproves
accuracy on four out of five benchmarks while also shortening the selected contexts. This supports our hypothesis that
gradually shifting the objective from recall to precision leads to a better quality–efficiency trade-off than optimizing a
single-stage objective.
Trainingstability.Reinforcementlearningondiscretetextselectionisoftencharacterizedbyinstability. However,
thanks to our dense reward supervision mined via LOO and the GRPO algorithm, Context-Picker demonstrates robust
training dynamics. As illustrated in Figure 3, the reward curves for both the Recall-Oriented Stage I (Left) and
Precision-Oriented Stage II (Right) show steady convergence. The minimal gap between training and validation
performance further validates the generalization capability of our offline evidence mining strategy.
4.3 Ablation Studies
To better understand which components of Context-Picker drive the gains, we conduct ablations on theLoCoModataset.
We focus on three aspects: rationale generation, redundancy-aware reward shaping, and the recall-oriented Stage I.
Rationalegeneration.Inthefullmodel,thepickeroutputsbothashortnatural-languagerationaleandasetofselected
IDs. Removing the rationale branch (“w/o rationale”) leads to a 6.5-point drop in Judge Acc and noticeably higher
varianceacrossruns. Wehypothesizethatrequiringthemodeltoverbalizewhycertainpassagesareselectedactsas
a structural regularizer: it encourages more stable reasoning over evidence interactions and reduces the tendency to
over-select loosely related passages.
Redundancy-aware reward shaping.When we remove the redundancy term in the reward (“w/o redundancy”), the
picker no longer receives explicit penalties for overshooting the golden set size. Under the same token budget, this
varianttendstokeepmorepassagesandaccumulatesnoise,resultingina 4.6-pointdroponLoCoMo. Thisconfirms
that explicitly modeling length/redundancy in the reward is important for achieving a good balance between recall and
precision, rather than relying solely on an implicit budget constraint.
Role of the recall-oriented Stage I.Finally, we examine a variant trained only with the Stage II objective (“w/o
Stage 1”), i.e., directly optimizing the refinement-oriented reward from scratch. This leads to a substantial degradation
to56.5%Judge Acc, 14.1points below the full two-stage Context-Picker. Qualitatively, this variant tends to converge
to over-pruned policies that miss key evidence, suggesting that the recall-oriented warm-up in Stage I is crucial for
9

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
Method Judge Acc (%)∆vs. full
Context-Picker (full) 70.6 –
w/o rationale 64.1−6.5
w/o redundancy 66.0−4.6
w/o Stage 1 56.5−14.1
Table 2: Ablation study of Context-Picker onLoCoMo. ∆denotes absolute drops in Judge Acc (percentage points)
compared to the full model.
exploring a diverse evidence space before learning to compress it. Taken together, the ablations show that both the
redundancy-awarerewardandthestagedoptimizationschemearenecessarytorealizethefullbenefitsofContext-Picker.
5 Related Works
5.1 Adaptive Retrieval and Context Optimization
Standard RAG systems typically retrieve a fixed top- Kset of passages using sparse or dense retrievers [Robertson
et al., 2009, Karpukhin et al., 2020, Lewis et al., 2021, Izacard et al., 2022], often combined with cross-encoder or
sequence-to-sequencererankerssuchasmonoT5toimproveorderingquality[Nogueiraetal.,2020,Sunetal.,2024,
Drozdov et al., 2023, Chen et al., 2025]. While this “retrieve-then-rerank” architecture substantially improves recall and
ranking, it still reliesona static Kfordownstream generation. Asa result, complexmulti-hop questions may suffer
frommissing evidence when Kissmall, whereas simplefactoid queriesincur unnecessary noiseand costwhen Kis
large, exacerbating long-context issues suchas distractor accumulation andthe “lost-in-the-middle” effect [Liuet al.,
2023, Jin et al., 2024]. Recent analyses of long-context RAG pipelines further show that simply increasing the number
of retrieved passages often yields higher recall but only marginal or even negative gains in answer accuracy [Jiang et al.,
2024, Jin et al., 2024].
Toovercometherigidityoffixed-sizeretrieval,aseriesofworkshaveexploredmoreadaptivestrategies. Self-RAG
[Asai et al., 2023] trains a single LM augmented with reflection tokens to decide, segment by segment, when to
retrieve, when to critique evidence, and when to continue generation. FLARE [Jiang et al., 2023a] performs active
retrieval by monitoring low-confidence tokens and issuing retrieval queries only when the model anticipates future
uncertainty. Adaptive-RAG [Jeong et al., 2024] introduces a query-complexity classifier that routes questions to
no-retrieval, single-step, or iterative RAG pipelines, and Adaptive- kchooses the number of selected passages from the
similarity-score distribution of candidates without additional tuning or iteration [Taguchi et al., 2025]. These methods
show that adjustingwhenandhow muchto retrieve can improve overall QA performance, but they typically require
multiple rounds of retrieval and generation or rely on hand-crafted decision rules rather than an explicitly learned
selection policy under a token budget.
Another line of work targets the context side of the pipeline viacompression. LLMLingua [Jiang et al., 2023b] uses a
smaller model to score and remove non-essential tokens inside prompts, yielding substantial speedups while preserving
taskperformance. RECOMP[Xuetal.,2023]compressesretrieveddocumentsintoconcisetextualsummariesbefore
feedingthemtothegenerator,reducingbothpromptlengthandtheburdenontheLMtolocaterelevantinformation.
These approaches operate primarily at the token or sentence level and focus on shrinking a given context, without
explicitly reasoning about whichsubset of passagesis minimally sufficient for answering the query.
Complementary to these advances, several works study context selection from a scoring perspective. Query-aware
and list-aware rerankers use LLMs or specialized models to assign relevance scores to passages individually or jointly,
sometimes with list-wise prompting that considers redundancy and coverage [Sun et al., 2024, Chen et al., 2025].
Generator-aware metrics evaluate how well candidate contexts align with the generator’s internal knowledge or its
preferencesviarewardmodelstrainedfromLLMfeedback[Wangetal.,2024]. Influence-guidedselectiongoesone
step further and defines a leave-one-out style Contextual Influence Value that measures performance degradation when
removing each passage [Deng et al., 2025]. However, most of these methods still operate at the level of per-passage
utilities plus thresholding, and are not designed to directly optimize for aminimal sufficientsubset under a strict input
budget.
5.2 Reinforcement Learning for Evidence Selection
Reinforcement learning (RL) has been widely used to align retrieval-augmented systems with downstream tasks. Early
work applied RL to optimize query reformulation, where an agent learns to rewrite user queries to better exploit a fixed
10

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
retrieval module [Zhu et al., 2025], or to train retrievers end-to-end with task feedback, as in REALM-style frameworks
thatupdateboththeencoderandretrievertomaximizeQAreward[Guuetal.,2020]. Theseapproachesimprovethe
quality of retrieved candidates, but still leave the final context selection to static Top- Kheuristics or simple truncation.
More recent approaches bring RL or RL-inspired feedback closer to the evidence and memory selection step itself.
DynamicRAG[Sunetal.,2025]modelsthererankerasanagentoverdocumentsequencesandtrainsitwithacombination
of supervised fine-tuning and RL, using LLM-judged answer quality as reward to adjust both the order and the number
ofretrieveddocuments. Memory-R1andrelatedmemoryagentsframelong-termmemorymanagementandretrieval
decisions as RL problems, training policies to decide what to store, update, or retrieve in order to support downstream
QAanddialogueoververylongconversationalhistories[Yanetal.,2025,Maharanaetal.,2024]. RLhasalsobeen
applied to conversational query reformulation and retrieval alignment and to broader agentic RAG frameworks such as
RAG-GymandREX-RAG,whichoptimizemulti-stepretrievalandreasoningtrajectorieswithpolicy-gradient–style
updates [Zhu et al., 2025, Xiong et al., 2025, Jiang et al., 2025]. Influence-guided context selection [Deng et al., 2025]
employs a generator–judge loop to estimate each passage’s marginal influence via leave-one-out utilities and then trains
a surrogate selector to approximate these influence scores.
WhiletheseRL-styleorRL-adjacentapproachesintroducevaluabletask-alignedsignals,theystillfacetwokeychallenges
for context selection: (i) rewards are largelytrajectory-level and sparse, as the agent receives only a scalar signal after
producing a full list or trajectory, making credit assignment to individual passages and redundancy penalties difficult;
and (ii) policies are typically optimized to improve list-wise ranking quality, to include all positively-scored contexts, or
tomanagememoryoperations,ratherthantoidentifyaminimalevidencesubsetthatpreservesanswerabilityundera
fixed inputbudget. Incontrast,Context-Pickeristrained onoffline-mined minimalsufficientevidence setsand usesa
two-stage,redundancy-awareGRPOobjectivetoexplicitlytradeoffcoverageandcompactnessatthepassagesubset
level.
6 Conclusion
In this work, we presentedContext-Picker, a reasoning-aware framework that learns a variable-length evidence set
underatokenbudget. Context-Pickercombines(i)anofflineevidenceminingpipelinethatdistillsgreedilyminimal
sufficientevidencesetsviaagenerator–judgeloopwithLeave-One-Out(LOO)pruning,providingdenseandtask-aligned
supervision;and(ii)atwo-stagereinforcementlearningscheduleoptimizedwithGRPO,whereStageI(recall-oriented)
emphasizes coverage of reasoningchains with arelaxed redundancy margin, andStage II (precision-oriented)tightens
redundancypenaltiestoprunedistractorsanddistillcompactsupportsets. Thepickerfurtheroutputsarubric-guided
rationale together with selected passage IDs, enabling structured, end-to-end consistent evidence selection.
Experimentson fivelong-context andmulti-hopQA benchmarksdemonstratethat Context-Pickeroutperforms strong
RAG baselines under comparable evidence budgets, achieving higher LLM-as-judge accuracy while often with
comparableorreducedcontextlengths. Ablationstudiesfurtherverifythatthecoarse-to-fineschedule,redundancy-
aware reward shaping, and rationale-guided output format each contribute substantially to the gains, and that removing
StageIleadstosevereover-pruninganddegradedperformance. FutureworkmayincludeextendingContext-Picker
tomoreopen-endedgenerationtasks, exploringalternativerewardsignalsbeyondLLM-as-judge, andintegratingthe
picker with token- or KV-level compression inside the generator to further reduce inference cost.
7 Acknowledgement
Wegratefullyacknowledge thesupportfromtheDistinguishedYoung ScholarsProjectfundedbytheNatural Science
FoundationofGuangdongProvince(No. 2025B1515020060),theBasicandAppliedBasicResearchProgramofthe
Guangzhou Science and Technology Plan (No. 2025A04J7141).
References
PatrickLewis,EthanPerez,AleksandraPiktus,FabioPetroni,VladimirKarpukhin,NamanGoyal,HeinrichKüttler,
Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for
knowledge-intensive nlp tasks, 2021. URLhttps://arxiv.org/abs/2005.11401.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrieval-augmented language
model pre-training, 2020. URLhttps://arxiv.org/abs/2002.08909.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand
Joulin,SebastianRiedel,andEdouardGrave. Atlas: Few-shotlearningwithretrievalaugmentedlanguagemodels,
2022. URLhttps://arxiv.org/abs/2208.03299.
11

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in
the middle: How language models use long contexts, 2023. URLhttps://arxiv.org/abs/2307.03172.
BowenJin,JinsungYoon,JiaweiHan,andSercanO.Arik. Long-contextllmsmeetrag: Overcomingchallengesfor
long inputs in rag, 2024. URLhttps://arxiv.org/abs/2410.05983.
Robertson, Stephen, and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and beyond.Foundations and
Trends in Information Retrieval, 3(4):333–389, 2009. doi:10.1561/1500000019.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-
tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan
He, and Yang Liu, editors,Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP), pages 6769–6781, Online, November 2020. Association for Computational Linguistics.
doi:10.18653/v1/2020.emnlp-main.550. URLhttps://aclanthology.org/2020.emnlp-main.550/.
LeeXiong,ChenyanXiong,YeLi,Kwok-FungTang,JialinLiu,PaulBennett,JunaidAhmed,andArnoldOverwijk.
Approximatenearestneighbornegativecontrastivelearningfordensetextretrieval,2020. URL https://arxiv.
org/abs/2007.00808.
RodrigoNogueira,ZhiyingJiang,andJimmyLin. Documentrankingwithapretrainedsequence-to-sequencemodel,
2020. URLhttps://arxiv.org/abs/2003.06713.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun
Ren. Is chatgpt good at search? investigating large language models as re-ranking agents, 2024. URL https:
//arxiv.org/abs/2304.09542.
HaotianChen,QingqingLong,MengXiao,XiaoLuo,WeiJu,ChengruiWang,XuezhiWang,YuanchunZhou,and
Hengshu Zhu. Scirerankbench: Benchmarking rerankers towards scientific retrieval-augmented generated llms, 2025.
URLhttps://arxiv.org/abs/2508.08742.
AndrewDrozdov,HongleiZhuang,ZhuyunDai,ZhenQin,RaziehRahimi,XuanhuiWang,DanaAlon,MohitIyyer,
AndrewMcCallum,DonaldMetzler,andKaiHui. Parade: Passagerankingusingdemonstrationswithlargelanguage
models, 2023. URLhttps://arxiv.org/abs/2310.14408.
Liang Wang, Nan Yang, and Furu Wei. Learning to retrieve in-context examples for large language models, 2024. URL
https://arxiv.org/abs/2307.07164.
Jiale Deng, Yanyan Shen, Ziyuan Pei, Youmin Chen, and Linpeng Huang. Influence guided context selection for
effective retrieval-augmented generation, 2025. URLhttps://arxiv.org/abs/2509.21359.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-rag: Learning to adapt
retrieval-augmentedlargelanguagemodelsthroughquestioncomplexity,2024. URL https://arxiv.org/abs/
2403.14403.
Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. Efficient context selection for long-context qa: No tuning, no
iteration, just adaptive-k, 2025. URLhttps://arxiv.org/abs/2506.08479.
Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, and Jiawei Han. Dynamicrag: Leveraging outputs of large language model as
feedbackfordynamicrerankinginretrieval-augmentedgeneration,2025. URL https://arxiv.org/abs/2505.
07233.
Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Kristian Kersting,
Jeff Z. Pan, Hinrich Schütze, Volker Tresp, and Yunpu Ma. Memory-r1: Enhancing large language model agents to
manage and utilize memories via reinforcement learning, 2025. URLhttps://arxiv.org/abs/2508.19828.
ChangtaiZhu,SiyinWang,RuijunFeng,KaiSong,andXipengQiu. Convsearch-r1: Enhancingqueryreformulation
for conversational search with reasoning via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.
15776.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang, Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing Song, Dengyu
Wang,MinjiaZhang,ZhiyongLu,andAidongZhang. Rag-gym: Systematicoptimizationoflanguageagentsfor
retrieval-augmented generation, 2025. URLhttps://arxiv.org/abs/2502.13957.
Wentao Jiang, Xiang Feng, Zengmao Wang, Yong Luo, Pingbo Xu, Zhe Chen, Bo Du, and Jing Zhang. Rex-rag:
Reasoning exploration with policy correction in retrieval-augmented generation, 2025. URL https://arxiv.org/
abs/2508.08149.
ZhihongShao,PeiyiWang,QihaoZhu,RunxinXu,JunxiaoSong,XiaoBi,HaoweiZhang,MingchuanZhang,Y.K.Li,
Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.
URLhttps://arxiv.org/abs/2402.03300.
12

Context-Picker: Dynamic context selection using multi-stage reinforcement learning
AdyashaMaharana,Dong-HoLee,SergeyTulyakov,MohitBansal,FrancescoBarbieri,andYuweiFang. Evaluating
very long-term conversational memory of llm agents, 2024. URLhttps://arxiv.org/abs/2402.17753.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen. Longrag: Enhancing retrieval-augmented generation with long-context
llms, 2024. URLhttps://arxiv.org/abs/2406.15319.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D.
Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018. URL https:
//arxiv.org/abs/1809.09600.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for
comprehensive evaluation of reasoning steps, 2020. URLhttps://arxiv.org/abs/2011.01060.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via
single-hop question composition, 2022. URLhttps://arxiv.org/abs/2108.00573.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng,
Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context
understanding, 2024. URLhttps://arxiv.org/abs/2308.14508.
Arjun Panickssery, Samuel R. Bowman, and Shi Feng. Llm evaluators recognize and favor their own generations, 2024.
URLhttps://arxiv.org/abs/2404.13076.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie
Ma,HonghaoLiu,SaizhuoWang,KunZhang,YuanzhuoWang,WenGao,LionelNi,andJianGuo. Asurveyon
llm-as-a-judge, 2025. URLhttps://arxiv.org/abs/2411.15594.
AkariAsai,ZeqiuWu,YizhongWang,AvirupSil,andHannanehHajishirzi. Self-rag: Learningtoretrieve,generate,
and critique through self-reflection, 2023. URLhttps://arxiv.org/abs/2310.11511.
ZhengbaoJiang,FrankF.Xu,LuyuGao,ZhiqingSun,QianLiu,JaneDwivedi-Yu,YimingYang,JamieCallan,and
Graham Neubig. Active retrieval augmented generation, 2023a. URLhttps://arxiv.org/abs/2305.06983.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for
accelerated inference of large language models, 2023b. URLhttps://arxiv.org/abs/2310.05736.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and
selective augmentation, 2023. URLhttps://arxiv.org/abs/2310.04408.
13
# When Confidence Takes the Wrong Path: Diagnosing Retrieval-State Lock-In in RAG

**Authors**: Sahib Julka

**Published**: 2026-06-22 00:08:00

**PDF URL**: [https://arxiv.org/pdf/2606.22728v1](https://arxiv.org/pdf/2606.22728v1)

## Abstract
The trustworthiness of a retrieval-augmented generation (RAG) system depends on more than the answer it returns, yet many black-box uncertainty methods still read agreement among sampled answers as confidence. That inference fails when repeated samples condition on the same defective retrieval state. The state may be empty, with the model falling back on parametric memory, or populated by a coherent but wrong neighbourhood. In either case, the answers agree because the error is stable. The problem is recognised in deployed RAG, but it has lacked a name, a measurable signature, and a prevalence bound. We supply all three. We name the failure retrieval-state lock-in and diagnose it by separating the three objects a single confidence score conflates: the answer surface, the retrieved evidence, and the retrieval state itself. In an inspectable, ontology-guided knowledge-graph RAG (KG-RAG) system across six question-answering snapshots, we measure the agreement blind spot directly: at five samples per question, 42% of KG-RAG errors and 59% of dense-retrieval errors carry zero answer dispersion, so agreement has nothing to rank, while evidence- and retrieval-state checks still flag most of them. The decomposition supports an auditable decision rule: accepting an answer only when answer, evidence, and retrieval checks all agree that it is low-risk reaches 91.9% pooled precision against a 69.7% accept-all rate. The cost is coverage: it certifies only 7.7% of answers as low-risk. On the clinical calibration domain it reaches 100% precision under an automated judge; this is an in-domain automated-label upper bound, not a clinical safety claim, and still needs human validation. Confidence in RAG is object-specific: when answers agree, the useful question is which part of the pipeline to distrust.

## Full Text


<!-- PDF content starts -->

When Confidence Takes the Wrong Path:
Diagnosing Retrieval-State Lock-In in RAG
Sahib Julka
LMU University Hospital, LMU Munich
Marchioninistr. 15, 81377 Munich
Abstract
Thetrustworthinessofaretrieval-augmentedgeneration(RAG)systemdependsonmorethantheansweritreturns,yet
many black-box uncertainty methods still read agreement among sampled answers as confidence. That inference fails when
repeated samples condition on the same defective retrieval state. The state may be empty, with the model falling back
on parametric memory, or populated by a coherent but wrong neighbourhood. In either case, the answers agree because
the error is stable. The problem is recognised in deployed RAG, but it has lacked a name, a measurable signature, and a
prevalencebound. Wesupplyallthree. Wenamethefailureretrieval-statelock-inanddiagnoseitbyseparatingthethree
objects a single confidence score conflates: the answer surface, the retrieved evidence, and the retrieval state itself. In
an inspectable, ontology-guided knowledge-graph RAG (KG-RAG) system across six question-answering snapshots, we
measure the agreement blind spot directly: at five samples per question, 42%of KG-RAG errors and 59%of dense-retrieval
errorscarryzeroanswerdispersion,soagreementhasnothingtorank,whileevidence-andretrieval-statechecksstillflag
mostofthem. Thedecompositionsupportsanauditabledecisionrule: acceptinganansweronlywhenanswer,evidence,
and retrievalchecks all agreethat it islow-risk reaches 91.9%pooled precisionagainst a 69.7%accept-all rate. Thecost is
coverage: itcertifiesonly7.7%of answersaslow-risk. On theclinicalcalibration domainit reaches100%precision under
anautomatedjudge;thisisanin-domainautomated-labelupperbound,notaclinicalsafetyclaim,andstillneedshuman
validation. ConfidenceinRAGisobject-specific: whenanswersagree,theusefulquestioniswhichpartofthepipelineto
distrust.
1 Introduction
Ask a retrieval-augmented generation (RAG) system (Lewis
etal.,2020)whichpharaohthetempleinfrontoftheOsireion
honours. The gold answer isSeti I, but the retriever anchors
toRamessesII,thepharaohoftheadjacentAbydostemple.
SampletheanswerfivetimesanditreturnsRamessesIIevery
time,withzerodisagreementandacoherentsupportinggraph
trace(cf.Figure1). Agreementisperfect;theansweriswrong.
A black-box uncertainty estimator that reads agreement as
confidence sees nothing to flag. Only one signal fires: the
retrieved passages contradict the confident answer.
Confidently stable wrong answers are a recognised risk in
deployedRAG,yetthefailurelacksastandardname,ameasur-
able signature, and a prevalence bound. This paper supplies
allthree. Manyblack-boxuncertaintyestimators(semantic
entropy (Farquhar et al., 2024; Kuhn et al., 2023), Self-
CheckGPT(Manakuletal.,2023),andtheirRAG-integrated
variants (Jiang et al., 2023b; Su et al., 2024; Zimmerman
et al., 2024)) ask whether sampled answers agree, then read
agreement as confidence. That inference breaks down when
retrievalkeepsreturningthesamedefectivestate: theanswer
is stable for the wrong reason, and drawing more samples
cannot recover the missing signal.Retrieval-statelock-in.Wecallthisfailuremoderetrieval-
state lock-in: the retrieval state is degenerate and near-
identical across repeated samples, so resampling cannot
surface the error. Two variants matter. Inabsencelock-in,
retrieval repeatedly returns no usable graph state, so the
modelanswersfromparametricmemory. Inpresencelock-in
(theOsireioncase),retrievalrepeatedlyreturnsacoherentbut
wrong graph neighbourhood. Both count as lock-in because
the fixed retrieval state, empty or wrong, is what defeats
sampling. We call the observable signature asilent error: a
wronganswerwithzeroobservedanswer-statedispersion(all
𝑁samples in one semantic cluster). Silence is a signature,
notproof: parametricoverconfidence,benchmarkambiguity,
andanswernormalisationcanallleavethesamefootprint,so
silent-errorratesareanupperboundonlock-inprevalence.
We then decompose those silent errors by retrieval-side
mechanism, empty versus demonstrably wrong route, to
separatelock-infromlook-alikeconfoundersandshowthe
signature is not predominantly artefactual (Section 7.4).
Lock-in is easier to diagnose in knowledge-graph-
augmented RAG (KG-RAG), where a symbolic retriever
can repeatedly anchor to the same entity and traverse the
samerelationpath. Knowledgegraphscanimprovemulti-hop
orprovenance-sensitivetasks(Edgeetal.,2024;Gutiérrez
et al., 2024; Hu et al., 2025; Shen et al., 2025; Zhu et al.,
1arXiv:2606.22728v1  [cs.CL]  22 Jun 2026

Question
Wrong entity anchor
Retrieved graph
neighbourhood
Repeated retrieval state
Repeated wrong answer
Near-zero answer
uncertainty×𝑁samplesretrieval-state
GPS0.44(supported)evidence-state
SEU=1.0firesanswer-state
silent“Osireion ... named after
which pharaoh?” (gold: Seti I)
Ramesses II
coherent, wrong builder
same evidence every sample
“Ramesses II”×5
DSE=0,SD-UQ≈0
Figure1: Workedpresence-lock-intrace(theHotpotQAOsireion
case, also in Appendix Table 32). The retriever anchors to a
stronglyassociatedbutwrongentity;theresultingneighbourhoodis
internallycoherent,everysamplereceivesthesameevidence,and
therepeatedwronganswercarriesnear-zeroanswer-state(sampled-
answer) uncertainty. The left rail shows where each diagnostic
familyattaches: answer-stateagreementissilent(allsamplesagree),
graph-pathsupport(GPS)stayslowbecausethewrongansweris
still reachable in the graph, and only the evidence-state check fires
(the retrieved passages contradict the answer). In theabsence
variant (Section 7.4) the chain instead repeats an empty retrieval
state, moving the warning to a retrieval-state abstention.
2025a; Wu et al., 2025), though recent comparisons find
graphretrievaltask-dependentandoftencomplementaryto
dense retrieval (Han et al., 2025; Xiang et al., 2025; Chen
etal., 2025;Pengetal., 2024). Thesymptomisnot limited
to graphs: any system that repeatedly returns a concentrated
evidence set can produce stable wrong answers, and the risk
compounds in agentic pipelines where successive queries re-
inforce a locked-in state. KG-RAG is useful here because its
retrieval state is visible: matched entities, triples, paths, and
anchorsmakethemechanismobservable. Ourexperiments
useOntoGraphRAG,1anontology-guided KG-RAGframe-
workwithentity-firstrouting,densefallback,andgraph-state
logging.
Threedistinctuncertaintyobjects.WearguethatRAG
confidence has three distinct objects:
1.Answer-state uncertaintymeasures variation across
sampled answers.
2.Evidence-stateuncertaintymeasureswhetherthere-
trieved passages locally support the generated answer.
3.Retrieval-stateuncertaintymeasureswhether there-
trieval state itself supplies graph support.
1https://github.com/julka01/OntoGraphRAGEach attaches to a different pipeline object and maps to a
practicaltrustquestion: calibratedconfidence,faithfulness
to evidence, and auditable provenance.
Contributions.First, wemeasure how prevalent the
answer-state blind spot isacross five QA families under
deployablepolicies.2At𝑁=5thesilent-errorfootprintpools
to42%of adaptive-KG and 59%of dense errors ( 8–55%by
dataset).
Second, we introduce adiagnostic decompositionsep-
arating answer-state, evidence-state, and retrieval-state un-
certainty. Theirrepresentativescoresareweaklycorrelated
(pooled Spearman 𝜌=0.03 between answer- and evidence-
state), so they carry complementary signal.
Third, we introduceSD-UQ, a low-cost question-
conditioned embedding-dispersion score within the answer-
state family: a practical convenience, competitive in these
low-sample runs, rather than a central claim.
Fourth, weshow the decomposition is actionable: a
conjunctiverulethatcertifiesanansweronlywhenanswer-
, evidence-, and retrieval-state checks all agree that it is
low-riskreaches 91.9%pooledprecisionat 7.7%coverage
(81.6%out-of-calibration), against a69.7%accept-all rate.
WecomparedenseRAG,adaptiveKG-RAG,andastrict
graph-onlystresstestacrossclinical,biomedical,andopen-
domainmulti-hopQAsnapshots. WedonotclaimKG-RAG
ismoreaccurate: itshowsnostatisticallydetectableaccuracy
gap with dense retrieval (paired McNemar, all 𝑝≥0.12).
Its value here is to expose the failure states answer-only
uncertainty cannot see.
Scope.The decomposition is graph-native: it reads On-
toGraphRAG’ssymbolicretrievalstatetoseparateabsence
from presence, though the silent-error phenomenon itself
is general (and larger under dense retrieval). The strict
stress test changes several pipeline components at once, so it
probesacompositeworst-caseregimeratherthanisolating
one causal variable.
2 Related Work
2.1 Answer-State Uncertainty for LLMs
AlargepartofLLMuncertaintyestimationstaysinoutput
space: if sampled answers agree, the answer is treated as
morelikelytobecorrect. Semanticentropyclusterssampled
responses into meaning-equivalent groups and computes
entropyovertheresultingdistribution(Kuhnetal.,2023;Far-
quharetal.,2024);relatedmethodsapproximateorsubstitute
forthatideabypredictingitfromhiddenstates(Kossenetal.,
2025),measuringcross-samplecontradiction(Manakuletal.,
2023),elicitingverbalisedconfidence(Kadavathetal.,2022;
Tian et al., 2023; Lin et al., 2022), or scoring agreement
2PubMedQA, RealMedQA, HotpotQA, 2WikiMultiHopQA, and
MuSiQue; HotpotQA contributes two snapshots (bundle and FullWiki).
2

across sampled reasoning paths (Wang et al., 2023). White-
boxvariantssuchashidden-stateprobes(Kossenetal.,2025)
andinduction-awareentropygating(Bazarovaetal.,2026)
sit outside the black-box setting studied here. They are or-
thogonal to the retrieval-state view: where hidden states are
available,theycouldbefusedwiththesediagnosticsrather
than treated as competitors.
A second line replaces hard semantic clustering with
response-embedding geometry: von Neumann entropy on
asimilaritykernel(Nikitinetal.,2024),concentrationand
dispersioninembeddingspace(QiuandMiikkulainen,2024;
Lietal.,2025b),andspectraldecompositionofrepresentation
uncertainty (Walha et al., 2025). For this paper they are
thenearestanswer-statebaselines. Theirfocus,however,is
answervariationitself,notthecasewhereretrievalstabilises
around an explicit graph trace.
2.2 Uncertainty in RAG
In RAG, uncertainty is asked to do several jobs. Adaptive
retrievalmethodsuseconfidenceorentropytodecidewhento
retrieve (Jiang et al., 2023b; Su et al., 2024; Yao et al., 2025;
Liuetal.,2025;Zubkovaetal.,2025;Moskvoretskiietal.,
2025),linkingtoselectivepredictionandabstention(Geifman
and El-Yaniv, 2017; Tomani et al., 2024), conformal NLP
(Camposetal.,2024),andtrustworthinessworkthattreats
reliabilityasapipelineproperty(Huangetal.,2024;Nietal.,
2025). Mostofthisworkusesanswer-sidegates: sampled-
answer uncertainty decides whether retrieval, critique, or
abstention is needed. That is precisely the signal lock-in
can silence. These methods sit naturally alongside the
evidence- and retrieval-state diagnostics studied here, rather
thanreplacingthem. Conformalmethodsareadjacentforthe
samereason: theyreadcoveragefromasinglenonconformity
score, whereas our conjunctive rule asks three families to
agreeandtargetsauditableabstentionratherthancoverage
calibration.
Two neighbouring lines come closer to the evidence- and
retrieval-state arms. Perez-Beltrachini and Lapata (2025)
predict evidence quality directly with a lightweight passage-
utilitymodelthatapproximatessampling-baseduncertainty
withoutsampling. Self-reflectivemethods(Self-RAG,cor-
rectiveRAG(Asaietal.,2024;Yanetal.,2024))letthemodel
decide when to retrieve, critique, or abstain through control
tokensorfine-tuning,butneedwhite-boxaccess. Thepresent
design instead stays black-box, a constraint that matters in
clinical and biomedical RAG, where recent reviews report
heavyuseofproprietarymodelsandunresolvedevaluation
and governance gaps (Amugongo et al., 2025).
Theclosestneighbourssharpentheclaim. Anaxiomatic
analysisshowsthatstandarduncertaintyestimatorscannot
reliablyassesscorrectnessinRAGandproposesacalibration
framework(Soudanietal.,2025a);thispapersuppliestheem-
piricalcounterpart,locatingwhereanswer-stateagreement
collapses and which retrieval- and evidence-state signals
still respond. FRANQ separates factuality from faithful-ness to retrieved context, but operates on flat-text RAG
and does not expose three-way graph-side decomposition
(Fadeevaetal.,2025). SURE-RAGscoresevidence-setsuffi-
ciency for selective answering (Qiu et al., 2026), posing the
evidence-state question effectively but not assessing whether
retrieval has concentrated around a wrong symbolic trace.
SURE-RAGandFRANQarebetterreadasmorespecialised
evidence-state diagnostics for flat-text RAG. We use SEU as
alightweightblack-boxrepresentative,leavingsubstitution
withlearnedsufficiencymodelstofuturework. R2Cperturbs
theretrieval–reasoningloopsothatdispersionreflectsboth
retrieverandgeneratoruncertainty(Soudanietal.,2025b);
itiscomplementary,measuringmovementunderperturba-
tion, while lock-in concerns unperturbed retrieval that barely
moves at all.
Broader retrieval-stability and context-selection work
makes the same background point: retrieval composition
mattersforreliability. Long-contextandcontext-balancing
methodsaskwhichspanstokeep(Lietal.,2024b;Voloshyn,
2026),noise-awarecalibrationstudieshowirrelevantcontext
affects confidence (Liu et al., 2026a), and retriever bench-
marks study how much information the retrieved set carries
(Zheng et al., 2026). Our claim is narrower: when repeated
samples see the same defective state, answer variance can
collapse even if the generator is sampled. On the graph
side, Ca2KG studies overconfidence in KG-RAG through
counterfactual prompting, but it does not report a structured
multi-family diagnostic that separates retrieval-state from
evidence-state uncertainty. Nor does it provide a prevalence
estimate,aroute-levelabsence/presencedecomposition,oran
auditrule(Renetal.,2026),preciselythediagnosticsthispa-
peradds. BRINKshowsthatKG-RAGsystemscanfallback
onparametricmemorywhengraphsareincomplete(Zhou
etal.,2026). Ourabsencevariantlargelyoperationalisesthat
phenomenon at retrieval time: the strict-clinical silent errors
are empty-retrieval rows answered from parametric memory
(Section 7.4). The additional case ispresencelock-in, where
the system retrieves a populated but wrong-and-coherent
neighbourhood. The three-family framing then separates the
two empirically. Table 1 places this contribution against the
five closest neighbours; it should be read compositionally,
not as a contest.
Threefurtherliteraturessettheremainingboundarycondi-
tions. First,RAGevaluationframeworkssuchasRAGAsand
ARES already separate context relevance and faithfulness
from answer quality (Es et al., 2024; Saad-Falcon et al.,
2024); their context-relevance/faithfulness split motivates
the analogous evidence-state check here (SEU, Table 4). We
tie that separation to explicit graph retrieval state, where an-
sweragreementandevidencesupportcandecouple. Second,
external selective-RAG systems, including passage-utility
predictors, FRANQ, SURE-RAG, R2C, and ARES-style
frameworks, target flat-text RAG, so a like-for-like compari-
sonwouldmeanreimplementingtheminsidetheKG-RAG
pipeline;becausethesilent-errorphenomenonislargerunder
denseretrieval(Section7.2),ourowndense-RAGrunsarethe
3

Table1: Positioningagainsttheclosest relatedwork. ✓: directobject ofstudy; ∼: partialorindirect; –: notaddressed. Among these
fiverecentneighbours,onlythepresentstudyjointlyinspectstheanswer,evidence,andretrievalstate(thethreeleftcolumns). Thetwo
right columns, set off by the rule, are context rather than comparison axes:Trace exposedreflects the inspectable KG-RAG substrate, and
Lock-in studiedis definitional, since the concept is introduced here.
Answer Evidence Retrieval Trace Lock-in
state state state exposed studied
FRANQ (Fadeeva et al., 2025)✓ ✓– – –
SURE-RAG (Qiu et al., 2026)∼✓∼ – –
R2C (Soudani et al., 2025b)✓∼ ∼ – –
Ca2KG (Ren et al., 2026)✓∼ ∼ ∼–
BRINK (Zhou et al., 2026) – –∼ ∼–
This paper✓ ✓ ✓ ✓ ✓
naturalsubstratewheresuchabaselinecouldbecompared.
Third, knowledge-conflict work models the tension between
retrievedevidenceandthemodelpriorthroughsource-aware
synthesisorKG-mediatedreconciliation(Wuetal.,2024b;
Wang et al., 2025; Zhang et al., 2025; Liu et al., 2026b)
and through attribution and citation-faithfulness checks that
catch unsupported LLM references, including in medicine
(Wallat et al., 2025; Li et al., 2024a; Wu et al., 2024a). The
presentstudyaskswhichofthesesignalsstayusefulwhen
retrieval is visible and partially stabilised.
2.3 KG-RAG and Graph-Side Uncertainty
Graph-augmented retrieval has become a prominent alter-
native to flat-text RAG. Systems such as RoG, Think-on-
Graph, HippoRAG, SubgraphRAG, GraphRAG, StructGPT,
G-Retriever,andGNN-RAGusegraphstomakemulti-hop
reasoning more explicit and auditable (Luo et al., 2024; Sun
et al., 2024; Ma et al., 2025; Gutiérrez et al., 2024; Li et al.,
2025a; Edge et al., 2024; Jiang et al., 2023a; He et al., 2024;
Mavromatis and Karypis, 2025), with benefits concentrated
onmulti-hopretrieval,bridgepreservation,networkedevi-
dence,orstrongerprovenance(Edgeetal.,2024;Gutiérrez
et al., 2024; Shen et al., 2025; Hu et al., 2025; Zhu et al.,
2025a; Wu et al., 2025); topology-aware retrieval adds se-
lection by graph proximity and structural role rather than
similarity alone (Wang et al., 2024). This study treats the
graphasasource-linkedsummaryofpassages,wheretriples,
paths,andrelationanchorstracebacktothetextthatlicensed
them. Comparative work correspondingly finds graph and
dense retrievers often complementary rather than globally
ordered (Han et al., 2025; Xiang et al., 2025; Peng et al.,
2024; Chen et al., 2025).
Less developed is the graph-side confidence question:
whether the retrieved graph state itself, including matched
entities, traversal paths, and relation anchors, supports the
generatedanswer. Thatquestionisdifferentfromchecking
answer variation or text-evidence consistency. Ca2KG (Ren
et al., 2026) and BRINK (Zhou et al., 2026), both discussed
above, are the nearest graph-side efforts, but neither does
this, while a separate literature models uncertainty inside
knowledgegraphsthemselvesthroughconfidence-weightedtriples or distribution shift in KG embeddings (Takahashi
et al., 2026; Lee, 2025; Zhu et al., 2025b). Once a hybrid
KG-RAG system exposes all three objects, the diagnostic
problem is to score answer variation, evidence support, and
graph support without pretending they are the same signal.
3Background and Problem Formula-
tion
3.1 Sampling-Based Uncertainty in RAG
ARAGmodeldoesnotdrawanswersinavacuum: eachsam-
pleis conditionedon whatever contextthe retrieverreturns.
Whenretrievalkeepsreturningthesamecontext,resampling
explores decoder variation but not evidence variation, so
agreement reports decoder stability, not correctness.
Let𝑞be a question, 𝑐a retrieved context bundle, and 𝑟
a model response. Sampling-based uncertainty estimators
operate on repeated draws from 𝑝(𝑟|𝑞). In RAG, the
response distribution is mediated by retrieval:
𝑝(𝑟|𝑞)=∑︁
𝑐∈C(𝑞)𝑝(𝑟|𝑞,𝑐)𝑝(𝑐|𝑞),(1)
whereC(𝑞)is the finite set of candidate context bundles.
𝑝(𝑐|𝑞)can be sharply peaked (near-degenerate, the lock-in
regime) or spread across several bundles.
3.2WhyKG-RAGChangestheInterpretation
KG-RAGchangeswhatdisagreement,andagreement,can
mean. Vanilla RAG returns a ranked list of passages. KG-
RAG returns a structured state: anchors, relation labels,
triples,andpathswhosenamesandtypescanbeinspected
ratherthaninferredafterthefact. Intheentity-firstregime,
retrievalismediatedbyentitymatchingandgraphexpansion
over a fixed graphG. When that process is highly stabilised,
one context dominates:
𝑐★=arg max
𝑐∈C(𝑞)𝑝(𝑐|𝑞).(2)
4

As the residual mass 1−𝑝(𝑐★|𝑞)shrinks, the marginal
collapses:
𝑝(𝑟|𝑞)≈𝑝(𝑟|𝑞,𝑐★),(3)
sorepeatedsamplesmostlyexposedecodervariabilityunder
fixed evidence (Appendix A gives the total-variation bound).
This is the regime of interest, and also the dangerous one.
Once retrieval has stabilised, low answer variance can mean
good evidence, but it can also mean the same retrieval-
side mistake has been repeated on every sample. In the
deployedpolicy, iterativedecomposition and densefallback
keep the residual from fully collapsing, so Equation (3)is
a stylised diagnostic lens rather than a full description of
every call. Write 𝑠for the graph retrieval state: the matched
entities,paths, andanchorsbehind aretrievedcontext. The
empirical counterpart is read from the saved route logs: a
silenterror(Definition1)isclassifiedbyitsretrievalrouteas
absence-compatible(anemptyroute)orpresence-compatible
(apopulatedroutethatdoesnotreachthegoldanswerentity),
per Definition 2.
Definition 1 (Silent error) A wrong answer with zero ob-
servedanswer-statedispersionunderthesamplingbudget:
all𝑁samples fall in one semantic cluster (DSE =0) and
theembedding-dispersionscoreSD-UQisatitsfloor(both
answer-dispersionscoresaredefinedbelow,Section5.1). The
floor is implementation-defined (encoder- and 𝜀-dependent),
so silent-error rates are reported as an upper bound.
Definition 2 (Retrieval-state lock-in) A silent error for
which the repeated samples condition on the samedefective
retrieval state 𝑠, either empty (absence) or a coherent but
wrong neighbourhood (presence). “Defective” is opera-
tionalised from route logs as an empty route, or a populated
routewhoseneighbourhooddoesnotcontainthegoldanswer
entity where checkable; rows with missing route metadata
are reported separately.
Bothvariantsareretrieval-statedefects. Thismattersbecause
parametricoverconfidencedespiteadequateretrievalcanlook
similar at the answer surface, but it is not lock-in.
3.3A Three-Family Analysis Frame for KG-
RAG
We use three families because a KG-RAG run exposes three
different objects: the answer, the assembled evidence, and
theretrievalstate(Table2). Recallthegraphretrievalstate 𝑠;
let𝑐bethe textualevidence and 𝑟theanswer. Thepipeline
factorises:
𝑝(𝑟,𝑐,𝑠|𝑞)=𝑝(𝑠|𝑞)𝑝(𝑐|𝑞,𝑠)𝑝(𝑟|𝑞,𝑐,𝑠).(4)
Answer-state uncertaintyprobes variability in𝑝(𝑟|𝑞,𝑐,𝑠);
evidence-statemeasuresaskwhether 𝑐islocallyconsistent
with𝑟; retrieval-state measures ask whether 𝑠supplies co-
herent graph support. A system can produce low-varianceTable2: Uncertaintyasanauditofthreepipelineobjects,not
one scalar.Answer-state metrics see only the sampled answers;
evidence-state metrics compare the answer against the retrieved
passages; retrieval-state diagnostics inspect the graph trace itself
(anchors,triples,paths,andabstentions). Thepaperreportsthem
separately because they answer different questions. This table
definestheobjects;Table5stateshoweachbehavesacrossretrieval
regimes,andTable8mapstheirjointoutcomesintoa 2×2failure
taxonomy.
Observed
objectSignal family Diagnostic question
Answer sam-
plesAnswer-state un-
certaintyDo repeated generations dis-
agree, or has the answer sur-
facebecomeartificiallysta-
ble?
Retrieved
passagesEvidence-stateun-
certaintyDoes the retrieved text lo-
cally entail, contradict, or
fail to support the generated
answer?
Graph trace Retrieval-state un-
certaintyWhich anchors, typed
triples, and paths made
the answer reachable, and
where does the graph
abstain?
Thethreefamiliesarenotindependentpredictorsofcorrectness.Theyare
threeloggedviewsofthesameKG-RAGrun: generatedtext,retrieved
evidence, and symbolic retrieval state.
answersfromthewrongsubgraph,retrieveaplausiblepath
whose passages do not entail the answer, or retrieve good
evidence while the answer remains unstable.
4 System Description
The experimental platform is OntoGraphRAG, an open-
source framework that exposes both vanilla RAG and KG-
RAGunderasharedinterface. Thetwosystemsusethesame
corpora, embeddings, and generation model; in the adaptive
dense-vs-KG comparison they differ only in retrieval and
evidence organisation.
4.1 OntoGraphRAG Overview
OntoGraphRAGusesgraphretrievaltoaddstructurewithout
discarding dense evidence (Figure 2). The graph turns re-
trievalcommitmentsintovisibleobjects: recognisedentities,
accepted relations, triples, paths, and the passages that li-
cencethem. DuringKGconstruction,passagesareconverted
intoentityandrelationnodeswithontology-guidedtyping,
anchor grounding, confidence scores, provenance links, and
contradiction flags when a schema is available. During
retrieval, the system routes the question through matched
entitiesandlocalneighbourhoods,fallingbacktodensere-
trieval and retriever-first expansion when the anchor is weak.
5

Domain
corpusPassage
chunkingEntity & relation
extractionKnowledge
graph
Question𝑞Entity
linkingSubgraph
retrievalContext
assemblyLLM
generationAnswerˆ𝑎build time query timegraph state
×𝑁samples: entity-first routing returns nearly the same subgraph
GPS
(retrieval-state)SEU
(evidence-state)DSE, SD-UQ, VN-Ent.
(answer-state, over𝑁samples)
Figure 2: OntoGraphRAG pipeline with the three measurement taps.Build time(top): a domain corpus is chunked and converted via
ontology-guided extraction into a typed knowledge graph with chunk-level provenance.Query time(bottom): the question is linked to seed
entities,andthegraphstateconditionsretrievalatthreepoints(entitylinking,subgraphtraversal,andcontextassembly)beforetheLLM
producesthefinalanswer. Thedashedprobesmarkwhereeachdiagnosticfamilyattaches: GPSscoresgraphsupportinriskorientation,
SEUscores theassembledevidence againsttheanswer,andthe answer-stateestimators seeonlythe 𝑁sampledanswers. Whenrepeated
samples re-enter retrieval but entity-first routing returns nearly the same subgraph, answer-state uncertainty mainly reads decoder variation
under almost-fixed evidence (Section 5).
The resulting context contains both textual passages and
explicitgraphpaths. Thatiswhatmakesthethree-levelaudit
possible: answerstate,evidencestate,andretrievalstate. On-
toGraphRAGlogsanchors,paths,andsupportingpassages
withthefinalanswer,sotheprovenancetraceispartofthe
retrieval output rather than a private prompt-construction
detail.
4.2 Vanilla RAG
Vanilla RAG performs direct vector retrieval over chunk
embeddings. For each question, it retrieves the top- 𝑘chunks
above a similarity threshold and optionally appends adjacent
chunks to capture answers split across chunk boundaries.
TheLLMreceivesflattextonly. Therearenoexplicitentities,
relations,ormulti-hoppaths,andhencenograph-sideobject
on which to define retrieval-state diagnostics.
4.3 KG-RAG
KG-RAG is a routed hybrid pipeline rather than a single
graph walk.
Stage1: seedselection.Thepreferredpathisentity-first
anchoring: the system identifies seed entities with symbolic
matching and embedding lookup over short mentions. If no
reliable anchor is found, the system does not immediately
reduce to vector-only retrieval; instead it can route to the
retriever-firstgraphpassofStage3,bypassingtheentity-first
expansion of Stage 2.
Stage 2: local graph expansion and scoring.Starting
from the seed entities, the system traverses the graph up to a
dataset-specifichoplimit,producinganexplicitretrievalstateof linked chunks, entities, relations, and readable traversal
paths. Traversal is provenance-aware: on question-bundle
datasets,pathsandedgesarerestrictedtothecurrentques-
tion’slocalevidenceratherthanallowedtoborrowsupport
from other questions in the same dataset graph. Chunks are
scored by hop distance and local diffusion over the retrieved
entity subgraph.
Stage3 (fallback): retriever-first graphexpansionwhen
anchoringisweak.Whenentity-firstretrievalisweakor
empty,thesystemfirstretrievesdensetextchunks,extracts
theirlinkedentities,expandsthegraphfromthosepassage-
derived seeds, and re-scores the resulting context with a
combined vector-plus-graph signal.
Stage 4: evidence organisation.The final prompt is or-
ganised rather than simply concatenated. Retrieved graph
paths are grouped with overlapping supporting passages into
explicit reasoning chains, with remaining passages listed
separatelyasadditionalevidence. Thegeneratorseesboth
structured multi-hop support and the local text that grounds
each chain.
Diagnostic role.OntoGraphRAG makes the retrieval state
explicit rather than tacit. The graph is not an oracle: its
triples and paths may be helpful or wrong. Their value here
is that they can be inspected. Logged anchors, paths, and
routesturnretrievalfromaninternalvariableintoaqueryable
object,richerfordiagnosisthanalistofnearest-neighbour
chunks alone.
Retrievalstabilisation.Thediagnosticproblemappears
when the retrieval state is stable across repeated calls.
6

Table 3: Quick reference for the four scores used in the results.
All are oriented so thathigher means more risk: a high value flags
a potentially untrustworthy answer, and a low score is low-risk
andpassestheauditgate. Fulldefinitionsarebelow;thisboxisa
reading aid for Sections 7 onward.
Score Object Plainmeaning(higher =riskier)
DSE answer sampled answers disagree
SD-UQ answer sampledanswersaredispersed
in embedding space
SEU evidence retrieved text contradicts the an-
swer
GPS retrieval answer entity is weakly sup-
ported in the graph
Retriever-first expansion, vector fallback, and iterative de-
composition can weaken that stabilisation when the graph
anchor is brittle, so the experiments compare diagnostics
across retrieval regimes. Entity-first retrieval can be espe-
cially prone to stabilising: once the system commits to an
anchorentity,thesubsequentgraphexpansionislargelyde-
termined, so the same neighbourhood, right or wrong, tends
to recur across samples.
5 Uncertainty Measures
Table 3 orients the four headline scores used in the results.
The full suite contains eight diagnostic measures, but the
organisingunitisnotthemeasure;itistheobjectobserved
in a QA run: the sampled answer surface, the retrieved
evidence,ortheretrievalstate(Table4). Theeightmeasures
shouldthereforenotbereadaseightindependentestimates
ofonehidden confidencevariable. Some dependenciesare
built in: the P(True)-style proxy is monotone in discrete
semantic entropy (DSE) under black-box sampling, and
VN-Entropy and SD-UQ are both geometric statistics of
the same response matrix. For that reason, the headline
tables use one representative per object: SD-UQ for answer
state (introduced here), SEU for evidence state, and GPS
forretrievalstate. DSE,P(True),SelfCheckGPT,SRE-UQ,
and VN-Entropy remain useful answer-side controls and
ablations(Table4);Section7specifieswhichappearsineach
table. The answer-state scores mostly come from standalone
LLM uncertainty estimation (Kuhn et al., 2023; Manakul
et al., 2023; Kadavath et al., 2022; Moskvoretskii et al.,
2025). Let 𝒓={𝑟 1,...,𝑟𝑁}denote𝑁responses sampled
from the LLM for a question 𝑞with retrieved context 𝑐.
Let𝒗𝑖∈R𝑑denote theℓ2-normalised embedding of 𝑟𝑖,
and𝑽=[𝒗 1|···|𝒗𝑁]⊤∈R𝑁×𝑑the stacked response
embedding matrix. All estimators return a non-negative
scalar: GPS and SEU are bounded in [0,1], DSE and VN-
Entropyhavemaximum log𝑁,andSRE-UQandSD-UQare
unbounded.5.1 Answer-State Uncertainty Estimators
Borrowed answer-state estimators.The five additional
answer-state controls are standard estimators, used here
without modification; their formal definitions are deferred
to Appendix C.1. In brief, DSE is the count-weighted black-
box semantic entropy of Kuhn et al. (2023); the P(True)
proxy (Kadavath etal., 2022) isrisk-scoredas one minusthe
cluster-agreement fraction, monotone in DSE under black-
boxsampling;SelfCheckGPT(Manakuletal.,2023)isthe
NLI-contradiction rate over sampled response pairs; SRE-
UQ (Vipulanandan et al., 2026) is a perturbation-sensitivity
statistic of the response-embedding distribution; and VN-
Entropy (Nikitin et al., 2024) is the von Neumann entropy
ofthenormalisedresponse-embeddingGrammatrix. Only
SD-UQ, the answer-state score introduced here, is defined in
full below.
SD-UQ (introduced here).SD-UQ is a question-
conditioned embedding-dispersion statistic, distinct from
QiuandMiikkulainen(2024)’sSemanticDensity. Itislow
when sampled answers point in nearly the same direction,
whichisthesignatureofastabilised(andpossiblylocked-in)
answer surface. Two design choices separate it from VN-
Entropy, the other geometric score. First, SD-UQ projects
out the question direction, so answers that merely restate the
questionregisteraslow-dispersion. Second,itsummarises
the residual spectrum by the geometric mean of the top- 𝑘
singular values, rather than by the von Neumann entropy of
the full Gram matrix, making collapse onto a single residual
modeeasiertosee. Giventheunit-normquestionembedding
ˆ𝒒∈R𝑑, define the orthogonal projector
𝑷⊥
𝑞=𝑰− ˆ𝒒ˆ𝒒⊤.(5)
The projected response matrix is
𝑽⊥=𝑽𝑷⊥
𝑞.(6)
The thin SVD of the centred projected matrix is
𝑯𝑽⊥√
𝑁=𝑼𝚺𝑾⊤,(7)
where 𝑯=𝑰−1
𝑁11⊤is the row-centering matrix. Let
𝜂1≥···≥𝜂𝑘be the top-𝑘singular values on the diagonal
of𝚺.
SD-UQ(𝒓)=exp 
1
𝑘𝑘∑︁
𝑖=1log(𝜂𝑖+𝜀)!
,(8)
where𝑘=min(𝑁−1, 𝑘 max)with𝑘max=8(so𝑘=4at
the reported 𝑁=5budget) and 𝜀=10−12is a numerical
stabilityconstant. Becauseallembeddingsare ℓ2-normalised
and the encoder and sampling budget are fixed, SD-UQ is
usedasanoperationallycomparablewithin-runrankingscore
rather than an encoder-invariant uncertainty estimate; at the
reported𝑁=5budgetonly 𝑘=4residualmodesenterthe
7

Table 4: Taxonomy of the eight uncertainty measures.The first six rows are answer-state estimators; the last two are the evidence-state
(SEU) and retrieval-state (GPS) diagnostics. Measured per-question costs are reported in Appendix Table 21. †: metric or KG-RAG
operationalisation defined here.‡: prior estimator first benchmarked here for KG-RAG.
Object family Subtype Measure Formal sketch Required inputs Cost at inference
Answer-stateEntropy DSE−Í
𝑘ˆ𝑝𝑘log ˆ𝑝𝑘, withˆ𝑝𝑘=|𝐶𝑘|/𝑁sampled answers,
NLI𝑁samples;𝑂(𝑁2)
NLI calls
CalibrationP(True)1−|{𝑖: cl(𝑟 𝑖)=cl(𝑟 1)}|/𝑁; black-
box P(True)-style proxysampled answers,
NLI𝑁samples; shares
DSE clustering
Similarity SelfCheckGPT1
|P|Í
(𝑖,𝑗)∈P[NLI(𝑟𝑖,𝑟𝑗)=
contradiction]sampled answers,
NLI𝑁samples;𝑂(𝑁2)
NLI calls
Perturbation SRE-UQ1
𝑀Í
𝑖∈T|Δ𝑖|; perturbation sensitiv-
ity of the kernel mean embeddingsampled answer em-
beddings𝑁samples; embed-
dings only
Geometric VN-Entropy‡−tr(𝝆log𝝆), with𝝆=𝑽𝑽⊤/𝑁 sampled answer em-
beddings𝑁samples; embed-
dings only
Geometric SD-UQ†exp 1
𝑘Í𝑘
𝑖=1log(𝜂𝑖+𝜀)onquestion-
orthogonal residualsquestion embed-
ding, sampled
answer embeddings𝑁samples; embed-
dings only
Evidence-state NLI support SEU† 1−(𝑛𝐸−𝑛𝐶)/𝐾
2; entailment–
contradiction deficit over retrieved
chunksretrievedchunks,an-
swer, NLI1 sample;𝐾NLI
calls
Retrieval-state graph support GPS†1−Í
𝑒:reach𝑒𝑤𝑒𝛾|𝐿𝑒−ˆ𝐿(𝑞)|
Í
𝑒𝑤𝑒; soft-
linked, depth-matched answer-entity
supportretrieved KG, entity
embeddings,linked
question/answeren-
tities1 sample; per-entity
graph queries, no
LLM calls
geometric mean, so it is a low-resolution ranking diagnostic,
not a high-resolution distribution estimate. SD-UQ needs
onlyquestionandanswerembeddings;thiskeepsitsmarginal
cost low.
5.2 Retrieval-State Support Diagnostics
The retrieval-state family (Section 1) asks whether the re-
trievedgraphcontainsausablepathfromthequestionentities
totheassertedanswerentity. Itmatterswhen 𝑝(𝑐|𝑞)≈𝛿 𝑐∗
(Section 3.2): repeated samples then see the same context,
soanswer-stateagreementcanbehighforbothcorrectand
wrong answers. GPS is used here as a graph-support di-
agnostic in risk orientation, where higher means weaker
support. It is not a truth label or entailment test. If retrieval
anchors to a wrong but coherent neighbourhood that sup-
ports the generated answer, GPS rates it low-risk; this is the
behaviour behind its weakness on the open-domain bridge
tasks (Section 7.5).
LetE𝑞andE𝑎denote the sets of KG entities linked to the
question𝑞and answer𝑎, respectively, filtered to E𝑎\E𝑞
to exclude trivial self-loops. Question entities are linked
by surface and fuzzy name matching. Answer entities are
linkedbythesamesurfacematcherplusanembedding-based
soft matcher: candidate answer spans are embedded with
the retrieval encoder and linked to any entity whose name
embeddinghascosinesimilarityatleast 𝜏(theentity-linking
threshold, calibrated to 𝜏=0.60 and distinct from the SD-
UQnumericalconstant 𝜀)tosomespan. Eachlinkedanswerentitycarriesalinkweight 𝑤𝑒: thematchedcosineforsoft
links and𝑤 𝑒=1for surface matches.
For GPS, answer entity extraction is restricted to the
primary answer span: the first sentence of the response,
truncatedto150characters,whichpreventslaterexplanatory
text from inflating the reachable entity count. If no usable
answer entity remains after filtering and soft linking, GPS
abstains rather than returning a spurious support score.
Graph Path Support (GPS; retrieval-state diagnostic).
GPS scores how strongly the answer is reachable from the
question entities within the retrieved KG. Each linked an-
swerentitycontributesdistance-weightedsupport 𝛾|𝐿𝑒−ˆ𝐿(𝑞)|,
where𝐿𝑒is the shortest qualifying path length (under the
sameedge-confidenceandprovenancefiltersusedatretrieval
time), ˆ𝐿(𝑞)is the expected reasoning depth for the question,
and unreachable entities contribute zero:
GPS(𝑞,𝑎)=1−Í
𝑒∈E𝑎𝑤𝑒𝛾|𝐿𝑒−ˆ𝐿(𝑞)|⊮[𝑒reachable]Í
𝑒∈E𝑎𝑤𝑒.
(9)
GPS=0 when every linked answer entity is reachable at
the expected reasoning depth (full structural support, low
uncertainty); GPS=1 when none are reachable within
theconfiguredhoplimit(nostructuralsupport,highuncer-
tainty); with 𝛾=1the score reduces to the unweighted
unreachable-entity fraction. The linking threshold and de-
cay were calibrated on the RealMedQA development run
(𝜏=0.60,𝛾=0.4) and applied frozen elsewhere; a 5×5
8

sweep over𝜏and𝛾(Appendix D.8) shows the held-out AU-
ROCs are stable across the grid, so the reported numbers are
notanartefactofthecalibratedcell. For2WikiMultiHopQA
and MuSiQue, ˆ𝐿(𝑞)is the logged per-question hop count;
forHotpotQAvariantsitisthenominaltwo-hopdepth;for
RealMedQAitis 1,whichmakesGPSidenticaltothecali-
brated one-hop-decay score on the clinical domain. When
no answer entities are found in the primary span (or when
the question is a direct-choice comparison), the estimator
abstainsandtherowisexcludedfromretrieval-stateAUROC
while being counted in the linking-failure rate (Section 5.4).
5.3 Evidence-State Uncertainty Measures
Evidence-state scores instead test whether the retrieved pas-
sages support the answer, so they can remain informative
when sampled answers agree. These are local answer–
evidence consistency scores, not truth labels.
SupportEntailmentUncertainty(SEU;evidence-support
score).SEU is the normalised entailment–contradiction
deficitoverretrievedchunks,closetoanswer-supportscoring
in RAGAs (Es et al., 2024).
Let𝑐1,...,𝑐𝐾denote the𝐾retrieved chunks and 𝑎the
generated answer. Each chunk is classified by an NLI
model(microsoft/deberta-large-mnli ;allmodelver-
sions are listed in Appendix C): ℓ𝑘∈{𝐸,𝑁,𝐶} (entailment,
neutral, contradiction) where 𝐸=NLI(𝑐 𝑘⇒𝑎). De-
fine the support score 𝑠=(𝑛𝐸−𝑛𝐶)/𝐾∈[−1,1] , where
𝑛𝐸=|{𝑘:ℓ𝑘=𝐸}|and𝑛 𝐶=|{𝑘:ℓ𝑘=𝐶}|.
SEU(𝑞,𝑎,c)=1−𝑠
2∈[0,1].(10)
SEU=0.0 when all chunks entail the answer (maximum
support,lowestuncertainty), SEU=0.5 whenchunksareall
neutral(undefinedsupport), SEU=1.0 whenallchunkscon-
tradicttheanswer(maximumconflict,highestuncertainty).
Neutral chunks create a plateau at 0.5, especially when a
general-domain NLI model declines to commit on specialist
biomedicalpassages(Section7.5). BecauseNLIneutrality
canreflectmodeluncertaintyratherthantrueabsenceofsup-
port,SEUisreadasalocalsupportheuristic,notacalibrated
evidence score. Unlike answer-state scores, SEU needs only
one generation.
Faithfulness to structured provenance.SEU does not
prove that the decoder followed the retrieved graph path.
A stricter path-faithfulness diagnostic would align answer
claims to entities, relation labels, and relation-anchor text
on the retrieved paths. The runs log those ingredients, but
not sentence-level claim-to-path alignment for every answer;
pathfaithfulnessisthereforeleftasafutureevidence-state
diagnostic.5.4 Entity-Linking Quality
GPS requires matched question entities and at least one
linkableanswerentity. Softanswerlinkingreducesabstention
relative to surface matching (Section 7.5); yes/no answers
remainapredictablefailurecase. Forbackwardcompatibility
the scalar API emits a 0.5sentinel, but AUROC, expected
calibrationerror(ECE)(Guoetal.,2017),andprecision-at- 𝑘
drop sentinel rows using the recorded null reason.
Entity-linkingsuccessrate 𝑔∈[0,1] measuresthefraction
ofcontent-bearingquerytokenscoveredbymatchedentity
names:
𝑔(𝑞)=min 
1,|ˆTcov
𝑞|
|ˆT𝑞|!
,(11)
where ˆT𝑞isthesetofdistinctalphanumericcontenttokensof
length≥4in𝑞(excluding stopwords), and ˆTcov
𝑞⊆ˆT𝑞is the
subset covered by at least one matched entity name (a token
𝑡iscoveredwhenitappearsasasub-tokenofanentityname,
or the entity name contains 𝑡as a substring). This token-
coverage score is a lightweight routing and stratification
heuristic, not an uncertainty estimator.
Table 5 is the behaviour map for the three families across
KG-RAG retrieval regimes: it states what each family can
observeasretrievalmovesfromanchorfailuretostabilised,
wrong-coherent, and incomplete-graph states, including
where a family goes silent or abstains.
6 Experimental Setup
6.1 Datasets
Thesuitecontainssixfixedevaluationsnapshotsspanning
three QA domains: biomedical control, clinical QA, and
open-domain multi-hop QA (Table 6). It deliberately mixes
corpus contracts: per-question bundles, per-question source
documents,andsharedcorpora. Theprimarygraphobject
isapassage-provenanceKGbuiltfromtheretrievalcorpus
itself,sincethequestioniswhetherOntoGraphRAGpreserves
dense-RAGanswerqualitywhileexposingfacts,passages,
relations, and retrieval state. PubMedQA and MuSiQue use
𝑛=100; RealMedQAusesits fullevaluableset ( 𝑛=230);
HotpotQA,HotpotQAFullWiki,and2WikiMultiHopQAuse
fixed𝑛=250subsets. PubMedQA (yes/no) is a boundary
control: itslabel-spaceanswersexposenolinkableanswer
entity,soGPSisundefinedandtheauditruleselectsnothing,
marking the regime where the retrieval-state arm of the
decompositiondoesnotapplywhiledenseretrievalisalready
strong.
RealMedQAis the main shared-corpus clinical setting
andHotpotQA FullWikia controlled Wikipedia shared-
corpus stress snapshot; the rest are closed source-document
orbundlesettings. ThesetupisconservativeforKG-RAG:
dense retrieval sees small, benchmark-filtered candidate sets
and provides a demanding accuracy baseline. That curation
flattersdenseretrieval,andtheaccuracycomparisonsshould
9

Table 5: Expected diagnostic behaviour across KG-RAG retrieval regimes.Symbols: ✓informative;∼partially informative;
↓compressed and potentially silent; ⊘abstains;×locally consistent with a wrong state. Each cell states what a family can observe, not a
ranking; Table 2 defines the families and Table 8 gives the resulting2×2taxonomy.
Retrieval regime State of the context Answer-state SEU GPS Reading
Anchor failure Entity linking fails; system falls back
to text retrieval.∼variation may
persist∼fallback pas-
sages⊘abstains Absence of a graph
object is itself diag-
nostic.
Stabilised retrieval Sameornear-samegraphneighbour-
hood across samples.↓compressed✓ support /
contradiction✓consistency Agreementshouldbe
readagainstevidence
and graph.
Wrong coherent an-
chorAnchoring succeeds on the wrong
neighbourhood; paths locally coher-
ent.↓same wrong an-
swer repeats∼can warn ×supports
wrong stateHardest lock-in case;
trace inspection
needed.
KG incompleteness Plausibleanchor,butbridgesorrela-
tions missing from the graph.∼fallbackrestores
variation✓if text re-
mains⊘/∼reduced Coverage or con-
struction error, not
decoder uncertainty.
Table 6:Evaluation datasets, corpus contracts, KG scale, and primary roles. ℎis the KG traversal depth;corpusdenotes the retrieval
contract;𝑛is the evaluated subset size. |𝐸|and|𝑅|count entity nodes and typed entity–entity relations in the dataset-scoped KG, queried
from the persistent Neo4j stores used in the reported runs.
Dataset Domain Answer typeℎCorpus𝑛|𝐸| |𝑅|Primary role
PubMedQA Biomedical Yes/No/Maybe 2 abstract 100 2,354 3,023 Binarycontrol;strong
dense baseline
RealMedQA Clinical Free-text 2 shared (143) 230 537 359 Clinical shared-
corpus grounding
HotpotQA (Yang et al., 2018) Wikipedia Free-text 2 bundle 250 13,546 12,355 Open-domain bridge
diagnostic
HotpotQA FullWiki (Yang et al., 2018) Wikipedia Free-text 2 shared subset 250 13,498 12,012 Shared-corpus bridge
stress test
2WikiMultiHopQA (Ho et al., 2020) Wikipedia Free-text 2 bundle 250 8,328 7,187 Bridge and compari-
son analysis
MuSiQue (Trivedi et al., 2022) Wikipedia Free-text 4 bundle 100 15,440 24,858 Hard multi-hop stress
test
be read with that in mind: a dilution probe in Appendix H.1
shows gold-passage recall falling sharply as the candidate
corpus grows toward deployment scale.PubMedQAis
retained as a control; the open-domain datasets stress bridge
preservation, shared-corpus evidence selection, comparison
questions, and deeper compositional chains.
Throughout, hop count refers to the reasoning chain im-
pliedbythequestion,notthenumberofpassagessuppliedby
the benchmark. Explicit decomposition metadata are used
where available, and dataset-specific conventions otherwise.
Hop-stratified analyses are reported as diagnostics rather
than the main table.
Scope.Theexperimentsarefixed-subsetdiagnosticruns,
not leaderboard submissions or web-scale throughput bench-
marks. They support the paper’s coarse family-level claims;
fine-grained ranking of neighbouring AUROC or AUREC
values would require larger multi-seed sweeps.6.2 Model and Sampling
Allanswer-generationexperimentsuseGPT-4o-mini(Ope-
nAI,2024). Foranswer-stateuncertainty, 𝑁=5responses
aresampledattemperature 𝑇=1.0. Fixedevaluationsubsets
are drawn with seed 42; model versions and the per-run sub-
setseedsarelistedinAppendixC(Table13). Weuse 𝑁=5
because it is the low-budget setting that a deployed API user
can afford per question, and the regime in which lock-in’s
structuralceilingonanswer-statescoresismostbinding. This
separates an empirical claim from a mechanistic one.Empir-
ically,at𝑁=5alargefractionofwronganswersshowno
observedanswerdispersion(Section7.2).Mechanistically,a
genuinely fixed defective retrieval state yields a fixed answer
distribution, so answer-only uncertainty cannot rank those
errorsatany 𝑁;larger𝑁wouldsoftentheempiricalfootprint
but not the mechanism. These are low-budget black-box
diagnostics: theAPIprovidesnoreliabletokenlikelihoods,
so entropy-style scores use sampled responses rather than
10

decoder probabilities, and the DSE/P(True)/VN-Entropy
clusterandeigen-spectraarenecessarilycoarseat 𝑁=5(the
reported scores are operational ranking diagnostics rather
than high-resolution distribution estimates).
KGconstructionislargelydeterministic: entityandrela-
tionextractionusetemperature 0.0,andbenchmarkpassages
areingestedoneat atimewithchunk-levelprovenance. Ex-
traction is ontology-guided on the biomedical and clinical
sets (PubMedQA, RealMedQA), which supply a domain
type schema, and schema-free on the open-domain sets; the
lock-indiagnosticsdonotdependontheschema. Thestudied
regimeisstabilisedentity-firstretrievalwithexplicitfallback.
Scope of causal claims.The comparison holds the corpus,
embeddings, and generation model fixed, but it does not
fully isolate graph structure from retrieval stabilisation, fall-
backrouting,promptorganisation,orprovenancegrouping
(Section 8.1). The runs report mean pairwise chunk overlap
across the𝑁=5calls (Appendix Table 17); an archived
2WikiMultiHopQAtracealsosupportsthestabilitysplitin
Table26. Theheadlinerunsdonotretainper-questionentity-
or path-overlap statistics for every dataset, but a dedicated
graph-statediversityrunonHotpotQA-FullWikilogsthefull
per-sample seed-entity, path, subgraph, and chunk overlap
and is analysed in Appendix D.5.
6.3 Correctness Labels
Correctness labels are binary. Label-space tasks are nor-
malised totheiranswercontract( yes/no/maybe). Free-text
andfactoidtasksuseareference-basedsemanticjudgegiven
thequestion,goldanswer,aliaseswhereavailable,andmodel
response;bydefaultthejudgeisGPT-4o-mini(AppendixC).
The judge performs benchmark answer normalisation, not
open-endedclinicaldiagnosis: itdecidesreference-grounded
semantic equivalence, for which exact match would under-
count aliases, paraphrases, and short explanatory answers.
Same-model judging is a known limitation (Zheng et al.,
2023; Panickssery et al., 2024), so the labels are practical
reference-grounded judgements rather than an oracle. An in-
dependentre-judgewithadifferentmodelfamily(Llama-3.3-
70B) on the saved answers of every free-text run agrees with
the original labels on 92–99%of answers ( 𝜅=0.52–0.97,
the low end being the near-ceiling-accuracy RealMedQA
adaptiverunwherefewwronganswersmake 𝜅unstable)and
leaves every central contrast unchanged (Section 8.1).
Two conventions matter. Provider-side failures are
recorded explicitly, so results report bothraw accuracy
andclean accuracy, the latter excluding generation failures
(AppendixTable14). Forfree-textdatasets,promptselicit
short explanatory answers rather than extractive spans; se-
manticcorrectnessisthereforetheheadlineanswermetric,
with EM/F1 retained only in the artefacts.6.4 Evaluation Metrics
Each uncertainty score is evaluated mainly byAUROC:
how well the score ranks incorrect above correct answers.
AUREC(areaundertherisk-excess-coveragecurve;loweris
better)isreportedonlyforthedense-sideselective-prediction
frontier (Appendix H.4) (Geifman and El-Yaniv, 2017), not
as a parallel metric in the main AUROC tables. Raw ac-
curacy, clean accuracy, retrieval overlap, and per-metric
compute time are also logged; the marginal cost of each
diagnostic,includingSEU’sper-chunkNLIcallsandGPS,is
reportedinAppendixTable21. Allmainfiguresreportpoint
estimates from one fixed subset per dataset. With mixed
sample sizes ( 𝑛=100,𝑛=230,𝑛=250), small AUROC
gaps are descriptive unless they recur across signal families
or match the qualitative traces. No multiple-comparison
correction is applied because the heatmap is not used for
family-wise significance testing. The two primary head-
linecontrastsinsteadcarrypaired-bootstrapintervals. The
largest (the adaptive-versus-strict SD-UQ collapse of +0.52
AUROC at𝑛=196) is far from zero. Gaps near 0.05are
not treated as significant: at these sample sizes the bootstrap
cannot reliably resolve effects that small, so the intervals
are used descriptively rather than as a formal power anal-
ysis. Appendix Tables 13, 14, 15, 22, and 25 document
run configurations, answered and failed counts, GPS usable
counts,theHotpotQAFullWikisnapshot,andbootstrapin-
tervals. Retrieval-stateAUROCwithfewerthanroughlyfifty
usable rows is treated as trace-level evidence. GPS AUROC
is always conditional on rows for which GPS is defined;
linking failures and other abstentions are reported through
usable/answereddenominatorsandaudit-rulecoverage,not
hidden inside the AUROC.
The headline tables report the family representatives of
Section 5 (SD-UQ, SEU, GPS), with DSE, SRE-UQ, and
VN-Entropyas within-familycontrols;the P(True)proxyis
omitted as monotone inDSE,and SelfCheckGPT is kept in
theartefacts. GPSvaluesarecomputedbypost-hocreplay
on saved answer logs and persistent KGs without rerun-
ning answer generation or document retrieval; primary GPS
numbers reuse stored linked entities and path lengths where
available,whilethecoverage-raisingandgold-reachability
replays recompute entity alignment and path support against
the persistent KG (Appendix C).
6.5 Retrieval and KG Configuration
Both systems use all-MiniLM-L6-v2 . Vanilla RAG per-
formsdirectvectorretrievaloverchunks. KG-RAGperforms
entity-first retrieval, graph expansion, and context assembly
from graph-linked chunks, paths, and entities, with dense
retrievalandretriever-firstgraphexpansionasfallbackswhen
entityanchoringisweak. EachKG-RAGresponserecords
its route ( entity_first ,rfge, orsemantic_only ) and
route reason.
Dataset-scoped KGs are built passage-wise, preserving
11

questionandpassageprovenanceandavoidingartificialcross-
question chunking. Each artefact retains chunk identifiers,
passage titles where available, relation paths, route labels,
andrelation-anchortextwhensuppliedbythe KGextractor.
Dense retrieval can log passages and scores; the KG layer
additionally exposes matched entities, typed triples, graph
paths,relationanchors,andabstentionevents,sotheKGtrace
is structured provenance. Traversal depth is dataset-specific:
ℎ=2for PubMedQA, RealMedQA, HotpotQA, HotpotQA
FullWiki,and2Wiki; ℎ=4forMuSiQue. Duringiterative
decomposition,eachsub-questionusesaboundedlocalgraph
expansioncapsothatoverallchainlengthandper-stepsearch
breadth are not confounded.
7 Results
The results follow the lock-in mechanism, not a leaderboard.
We first check whether KG-RAG shows any detectable accu-
racy gap against dense retrieval (Section 7.1), then measure
how often answers go silently wrong under deployable poli-
cies (Section 7.2), show that answer-state ranking collapses
onceretrievalisforcedtoconcentrate(Sections7.3–7.4),and
finallytestwhetherevidence-andretrieval-statesignalsfill
the gap, distilling the decomposition into a conjunctive audit
rule (Sections 7.5–7.7). Table 7 gives the running summary.
7.1Accuracy: nostatisticallydetectableKG–
dense gap at these sample sizes
KG-RAG point estimates sit slightly below dense retrieval
on all six snapshots ( 0.004to0.084), but no per-snapshot
differenceisstatisticallyreliable(pairedMcNemar,all 𝑝≥
0.12; Appendix Table 35). This is a no-detectable-gap
resultatthesesamplesizes,notanequivalenceclaim. The
comparison is deliberately conservative: we do not claim
KG-RAG is more accurate, only that it exposes answer-,
evidence-, and retrieval-state diagnostics the dense baseline
neverlogs. Comparableaccuracymakesthecomparisonfair,
but it is not the contribution. The strict graph-only stress
test forces retrieval to concentrate, a synthetic composite
worst-caseproberatherthanacontrolledablation: itdisables
dense fallback, augmentation, reranking, and decomposition
together,sothecollapseisnotattributedtographstructure
alone. It is a diagnostic contrast, not evidence that removing
any one component would cause the same failure.
7.2Finding 1: Answer-state uncertainty
ranks errors but is silent on a large slice
Answer-state uncertainty ranks errors well in ordinary adap-
tive runs, yet a non-trivial share of wrong answers carry no
answer-statesignalatall.Table9reportsthesilent-failure
rate: thefractionofwronganswerswithzeroanswer-state
uncertainty (DSE =0and SD-UQ at its numerical floor),
so no within-question disagreement signal remains. Theblind spot belongs to the whole family: identical samples
contain no signal for DSE, SD-UQ, VN-Entropy, or any
answer-dispersion variant. The footprint is strongly dataset-
dependent: 8%ontheclinicaldomain(RealMedQA)to 55%
on2WikiMultiHopQA,poolingto 42%(adaptiveKG)and
59%(dense).
Thestrictprobereaches 84%,butthatfigureisnotcom-
parable to the deployable-policy rates: it comes from the
multi-component stress test above, so its elevation reflects
engineered absence of retrieval, not graph structure alone.
Dense retrieval producesmoresilent errors than adaptive
KG-RAG,sothephenomenonisnotgraph-specific. Norisita
low-budgetartefact: an 𝑁=20probestillleaves 68%ofdense
2WikiMultiHopQA errors strictly silent (Appendix D.1).
TheKG-specificriskisauditablewrongness: asilenterror
arriveswithmatchedentities,typedpaths,andprovenance
anchors,makingthesystemlookstructurallyauditablewhile
it is wrong.
At the operational 𝑁=5budget, a method that only ob-
serves sampled-answer disagreement can recall at most 41%
of dense errors, 58%of adaptive KG errors, and 16%of
strict-stress-testerrors;theremainingwronganswersprovide
no within-question disagreement signal to rank, whatever
weighting it applies to the samples. That missing observable
motivates evidence-state and retrieval-state diagnostics.
7.3Answer-state metrics are strong but not
sufficient
Thenextquestionishowmuchanswer-stateestimatorsstill
contribute when retrieval remains adaptive. Of the eight
measures (Table 4), the additional answer-side controls do
not outperform SD-UQ in this low-sample regime, so the
narrative tracks the three family representatives (SD-UQ,
SEU,GPS)andtheappendixcarriestherest. Intheadaptive
runs they remain the strongest default: the AUROC heatmap
(Figure 3) shows that answer-side scores have the clearest
overall association with error, especially when retrieval and
decodingstillproducemeaningfulanswervariation. When
graph routing stabilises the context around a wrong entity
orpath,answerdisagreementcanbelowevenforincorrect
answers; the score then measures the smoothness of the
answersurfaceratherthanthereliabilityoftheretrievedstate.
Answer-stateuncertaintyremainsastrongdefault,butitis
not a lock-in diagnostic.
Within that family, the embedding-geometric scores VN-
Entropy and SD-UQ are the most dependable answer-side
baselines in these low-sample runs, matching or beating
the imported semantic-entropy and perturbation scores on
every KG snapshot but MuSiQue (per-dataset AUROCs and
intervals in Appendix Table 25). They differ mainly in
how they degradeas retrieval concentrates: hard-clustering
scoressuchasDSEloseresolutionanddrifttowardchance
once overlap rises, whereas SD-UQ retains discriminative
power by first projecting out the question direction and then
measuring the residual spread among answers (cf. Appendix
12

Table 7: KG-RAG shows no statistically detectable accuracy gap with dense retrieval, while exposing per-family diagnostics dense
retrieval cannot.Accuracy is clean semantic accuracy (provider failures excluded). The displayed 𝑛counts answered KG rows; Δis
KG−denseaccuracy. DenseandKGcolumnsreportcleanaccuracyoneachsystem’sownansweredrows. PaireddeltasandMcNemar
testsuserowsansweredbybothsystems(AppendixTable35). GPSisthegraph-supportdiagnostic(lowerisstrongersupport)withits
usable/answered denominator; the per-family answer-, evidence-, and retrieval-state AUROCs are in Figure 3. TheMain readingcolumn
summarises each row; intervals and denominators are in Appendix C. GPS was calibrated on RealMedQA ( 𝜏=0.60,𝛾=0.40) and applied
frozen elsewhere; the Main reading column flags where GPS is in its calibration domain, near chance, or weak.
Dataset𝑛Dense acc. KG acc.ΔGPS (usable) Main reading
PubMedQA 100 0.750 0.730−0.020– Binary control; GPS abstains
(no answer entities).
RealMedQA 223 0.950 0.946−0.0040.76 (195/223) Clinical near-match; GPS cali-
bration domain.
HotpotQA 238 0.655 0.605−0.0500.51 (158/238) Dense stronger; KG adds an-
chors and paths.
HotpotQA FullWiki 218 0.721 0.665−0.0560.54 (185/218) Shared-corpusstress;GPSnear
chance.
2WikiMHQA 244 0.712 0.689−0.0230.38 (182/244) Near-match; GPS weak on
bridges.
MuSiQue 94 0.478 0.394−0.0840.68 (83/94) Hardmulti-hop; auditable,not
competitive.
Table 8: Failure taxonomy for retrieval-augmented generation.
Retrieval-statelock-in(topright)isthefocalregime: theretrieval
state is defective while answer-state uncertainty is low. Absence
lock-inisflaggedbyretrieval-stateabstention;presencelock-in,a
coherent wrong neighbourhood, remains the hard case. At 𝑁=5,
42%of adaptive-KG errors fall in this silent cell ( 59%dense, 84%
strict).
Retrieval state
correctRetrieval state
wrong / missing
Low answer-state Certified low-risk Retrieval-state
lock-in
uncertainty all three families
concuranswer-state silent;
absence variant
flagged by
retrieval-state
abstention,
presence variant
open(this paper)
High answer-state Ordinary
uncertaintyRetrieval failure
uncertainty answer-state flags it answer-state flags it;
retrieval-state
abstains or shows
no support
Table 26). This robustness costs little: SD-UQ needs only
question andanswer embeddings,with no log-probabilities,
NLI calls, or model internals. One caveat matters for this
comparison: SD-UQ and VN-Entropy read the geometry of
anexternalembeddingspace thatthecluster-andNLI-based
scores(DSE,SelfCheckGPT)donotuse,sotheanswer-state
contrast is not strictly like-for-like.Table9: Silent-failurerateswithWilson95%intervals.Fraction
of wrong answers with zero answer-state uncertainty (DSE =0and
SD-UQ at its numerical floor), per retrieval policy. 𝑛is the number
of wrong answered rows.
Dataset Dense Adaptive KG Strict KG
PubMedQA .84 [.65,.94] (25) .67 [.48,.81] (27) –
RealMedQA .09 [.02,.38] (11) .08 [.01,.35] (12) .73 [.61,.82] (66)
HotpotQA .59 [.48,.69] (78) .46 [.36,.56] (94) –
HotpotQA-FW .45 [.33,.58] (58) .18 [.11,.28] (73) –
2WikiMHQA .79 [.67,.87] (61) .55 [.44,.66] (76) .90 [.84,.94] (145)
MuSiQue .51 [.37,.65] (47) .42 [.30,.55] (57) –
Pooled .59 [.53,.65] (280) .42 [.36,.47] (339) .84 [.79,.89] (211)
7.4Finding 2: Strict graph-only stress col-
lapses answer-state ranking
Forcingretrievaltoconcentratedriveswronganswersinto
the low-dispersion corner, so answer-state ranking degrades
toorbelowchance.Thesharpeststress-testsignatureappears
in the strict graph-only regime, where answer-state uncer-
tainty becomes a poor guide to correctness. That answer
dispersionfallsunderforcedconcentrationisexpected;the
informativepartiswhatthecollapseismadeof,whichthe
route decomposition below recovers.
Thestresstestmoveserrorsintothelow-dispersioncorner:
43/196RealMedQAand 87/2422WikiMultiHopQAques-
tions shift from correct-and-adaptive to wrong-and-silent
under strict retrieval (Appendix Figure 7). Clean KG ac-
curacy falls to 0.670and chunk overlap rises from 0.558
to0.661(cf. Appendix Table 17). Answer-state ranking
collapses: DSE AUROC 0.463, VN-Entropy 0.248, SD-UQ
0.233, at or below chance, actively misleading (Figure 4;
Appendix Table 24). The 0.233reflects engineered absence
ofretrieval,notagenericfailureonpopulatedlock-in(where
ranking stays healthy). The adaptive-to-strict SD-UQ gap
13

DSE
SRE-UQ VN-Ent. SD-UQSEU GPS
CombinedPubMedQA
RealMedQA
HotpotQA
HotpotQA-FW
2WikiMHQA
MuSiQue0.64 0.64 0.61 0.70 0.59 n/a 0.81
0.70 0.66 0.81 0.81 0.50 0.76 0.73
0.66 0.63 0.70 0.64 0.48 0.51 0.57
0.67 0.69 0.70 0.67 0.56 0.54 0.66
0.60 0.65 0.67 0.69 0.64 0.38 0.62
0.66 0.66 0.67 0.63 0.63 0.68 0.74Answer-state Evidence Retrieval CompositeAdaptive KG diagnostic AUROC
0.00.20.40.60.81.0
AUROCFigure 3: Answer-state scores rank errors most consistently; GPS discriminates only in its calibration domain (RealMedQA).
Adaptive KG-side diagnostic AUROC across the six snapshots, grouped by family; incorrect answers are the positive (risk) class and the
diverging colour scale is neutral at chance ( 0.50), red below, blue above. Vertical rules separate the answer-, evidence-, and retrieval-state
families;theCombinedcolumnisthemeanofwithin-datasetpercentileranksofSD-UQ,SEU,andGPS-risk(abstentionsimputedhigh-risk,
so it differs from the usable-row GPS column). PubMedQA GPS is undefined (yes/no answers expose no entities). Thin-denominator cells
(notably MuSiQue) carry wide intervals and are read qualitatively; denominators and bootstrap 95%CIs are in Appendix Tables 15 and 25.
on the paired subset is +0.52AUROC (bootstrap 95% CI
[0.28,0.73] ,𝑛=196). SEU ( 0.721) and GPS ( 0.746) still
rank errors on this strict run (GPS in its calibration domain);
thecontrastholdsunderanindependentLlama-3.3-70Bjudge
(𝜅=0.89).
Averbalised-confidenceproberecoversmuchof theclin-
ical mass (AUROC 0.89vs. SD-UQ’s 0.233) at one call,
but sits near chance on 2WikiMultiHopQA and returns no
auditable provenance (Appendix D.6). We keep SD-UQ
as the answer-state representative for consistency with the
adaptivetables;theverbalisedprobedominatesitinthestrict
regime but not on the multi-hop tasks the decomposition
targets.
Routedecomposition.Thecollapsedecomposesintothree
components.
First, it is not a chunk-level artefact.An interventional
dose-response reduces chunk overlap from 0.67to0.29, but
leaves SD-UQ AUROC flat ( 0.18) and the silent-error count
unchanged (Appendix D.2).
Second, route logging shows where the silent state lives.
Everysilenterrorinthestrictclinicalrunisanempty-retrieval
row ( 48/48at𝑛=230): strict anchoring found no usable
context,thesameemptystaterecurredacrossallfivesamples,
and the generator answered unanimously from parametric
memory. The nearestexplanation istheremoval offallback
andvectoraugmentationinthisstressrun,butthedesignstill
changesmultiplecomponentsanddoesnotisolateoneasthe
cause. On the populated-route rows, answer-state ranking is
healthy (SD-UQ AUROC 0.79). Much of SEU’s apparent
survivalismechanical: empty-retrievalwrongrowsreceivethe all-neutral default 0.5, which ranks above confidently
supported correct rows; populated-row SEU is only 0.56.
This isabsencelock-in, whose simplest reliable flag is the
empty-route/abstention observable, not answer dispersion or
evidence entailment.
Third, the strict 2WikiMultiHopQA run decomposes the
samewaybutleavesaremainder.Ofits 130silenterrors, 110
areempty-retrieval,while 20ridepopulatedentity-firstroutes
intowrong-but-coherentneighbourhoodsoftheOsireiontype
(presencelock-in). Populated-row SD-UQ stays degraded
there ( 0.65). Presence lock-in is the harder case: the answer
surfaceandtherouteobservablearebothsilent,leavingonly
evidence-levelchecks. ItisalsoasettingwhereSEUitself
provedunreliable(thestrict2-hopslice),whichiswhythe
composite recommendation does not reduce to any single
surviving family.
Adaptive-policy footprint.Under the deployable adaptive
policy the silent errors are mostly presence-like. Of the
141pooled adaptive silent errors, 89carry recorded route
metadata, all on populated routes with none on the empty-
retrievalroute, so the deployable-policy footprint is not an
empty-retrieval artefact. The logging gap does not carry
that claim: the remaining 52of the 141silent errors have
no route metadata, so their absence/presence mix cannot be
readfromthesavedlogs. Butevenundertheworstcasefor
our reading (all 52unlogged rows being empty-retrieval),
thepopulated-routeshareisstillatleast 89/141≈63% ,so
populated-route, presence-compatible silent errors dominate
the deployable-policy footprint under any imputation.
Decomposingthose 141adaptivesilenterrorsbyfamily,
14

Dense Adaptive KG Strict KG12
10
8
6
4
2
log10(SD−UQ+10−12)
208/11 211/12 134/66wrong answers
collapse to zeroAUROC 0.63
9% of wrong at floorAUROC 0.81
8% of wrong at floorAUROC 0.23
73% of wrong at floorAnswer-state dispersion (SD-UQ)
Dense Adaptive KG Strict KG0.00.20.40.60.81.0SEU uncertainty
208/11 211/12 134/66SEU stays defined
(partly neutral default;
see caption)AUROC 0.63 AUROC 0.50 AUROC 0.72Evidence-state support (SEU)
Numbers beneath each policy give correct/wrong answered counts.Correct WrongFigure 4: Under strict retrieval the answer surface collapses while evidence support stays separated (RealMedQA: dense, adaptive,
strict graph-only).Violin/strip distributions split by correctness, with AUROC per panel. 73%of wrong answers fall to the SD-UQ floor
under strict retrieval ( 48/66vs.1/12adaptive; Fisher exact 𝑝=4×10−5); SEU stays separated, but mainly because empty-retrieval wrong
rowstakeitsneutraldefault(Table10). Thisclinicalcollapseisdrivenbyempty-retrieval(absence)rows,distinctfromthepopulated-route,
presence-compatiblesilenterrorsthatdominatetheadaptive-policyfootprint. Countsbeneatheachpolicyarecorrect/wrongansweredrows.
134(95%)carryatleastonenon-answerflagandonly 7(5%)
remainunflagged. Thetwonon-answerfamiliesoverlapon
the flagged rows: 67show an evidence-state contradiction
(SEU>0.5 ) and 127weak or absent graph support (GPS
>0.5or abstention).
Amanualauditoftheseseven(AppendixD.9)findsmostly
benchmark-ambiguous questions, wrong-answer-slot errors,
andparametricconflationsratherthanconfirmabledeeppres-
encelock-inwhoseevidenceentailsthewronganswer(the
BalticCuptypeofTable32);chunktextisre-retrievablefrom
theKG,butconfirmingthatdeceptivetypeneedsthecross-
passage entailment and path-faithfulness replay described
below, not run for this log-only audit. This decomposition
helps separate lock-in from look-alike confounders: those
134flaggedrowscarryaconcreteevidence-orretrieval-state
mechanism,whiletheresidueisconfoundersratherthancon-
firmablelock-in,whichiswhythesilent-errorrateisreported
throughout as an upper bound on lock-in prevalence.
Flagging the cases where all three diagnostic arms fail,
withSEUandGPSbothnearzeroonawronganswer,requires
twochecksthislog-onlyanalysisdoesnotrun: cross-passage
contradiction mining over the re-retrieved chunk text, and
relation-levelpathfaithfulnesstocheckthattheconnecting
path uses the relation the question asked for rather than a
locally coherent but wrong one. Both are concrete, retrieval-
replay-only extensions, not new theory. Table 10 makes the
accounting explicit.
Which signal survives depends on the variant: route
or abstention observables for absence, and evidence-level
checks, imperfectly, for presence.Cross-datasetcorroboration.Thesamepatternappears
on the 2WikiMultiHopQA 𝑛=250run (cf. Appendix Ta-
ble30). TheadaptiveKGpolicystaysclosetodenseretrieval
overall ( 0.689vs.0.712) and matches it on the 4-hop slice.
Onthestrict4-hopslice,accuracyiszeroandbothSD-UQ
and VN-Entropy collapse to near-zero: the answer surface is
stablepreciselywhentheretrievedgraphstatehasbecome
wrong enough to make every answer fail. The collapse
signature is thejointcollapse of SD-UQ and VN-Entropy
together with failed answers, not low SD-UQ alone (the
dense 4-hop row has low SD-UQ but healthy VN-Entropy
and high accuracy).
Per-questionsupportcomesfromadedicateddiversityrun
onHotpotQA-FullWiki( 𝑛=221,74wrong;AppendixTa-
ble 27). Across 𝑁=5samples the retrieval state is highly sta-
ble(seed-entityJaccard 0.99;path/subgraph/chunkoverlap
0.58–0.65),andamongwronganswersmoreoverlappredicts
lower dispersion ( 𝜌=−0.30 /−0.23/−0.24, all𝑝<0.05; the
2WikiMHQA trace is consistent: 𝜌=−0.32 ,𝑝=0.01 ).
Theseanalysessupportthemechanismwithoutidentifyingit
causally (Appendix D.5, D.3; Appendix Figure 6).
Boundary case (HotpotQA).HotpotQA supplies a useful
counterweight: denseretrievalleads on accuracy ( 0.655vs.
0.605) while answer-state AUROCs stay healthy on both
systemsandGPSisonlyweaklydiscriminative( 0.51/0.54;
Appendix Table 22). These runs bound the claim: when
denseretrievalalreadyhasacompact,well-indexedcandidate
set,graphorganisationaddsprovenancebutdoesnotimprove
answer quality or retrieval-state ranking.
15

Table 10: Silent-error accounting from saved logs.Buckets are mutually exclusive and apply only to wrong answered rows. “Max
answerrecall”isthestructuralupperboundforanyanswer-dispersion-onlydetectoratthe 𝑁=5budget: silentwronganswerscontainno
sampled-answer disagreementsignal. Empty denotes absence lock-in; Pop.+SEU and Pop.+GPSare populated-route silenterrorsflagged
by evidence or graph-support signals. Unflagged is the residue with no non-answer-family flag.
Run Wrong Silent Max answer recall Empty Pop.+SEU Pop.+GPS Route-unk.+flag Unflagged
Adaptive KG pooled 339 141 .58 0 40 44 50 7
Strict KG pooled 211 178 .16 158 13 6 0 1
RealMedQA strict 66 48 .27 48 0 0 0 0
2WikiMHQA strict 145 130 .10 110 13 6 0 1
Pop. buckets are route-known; because GPS can be defined without a recorded route label, 2of the Adaptive-KG Unflagged rows are route-unknown, so the
89route-known /52route-unknown split in Section 7.4 is not a direct column sum.
Selectiveprediction.Figure5reframestheresultsasse-
lective prediction: if a diagnostic score is high, should the
systemanswer,broadenretrieval,orsendthecasetoreview?
Under adaptive runs, answer-side scores give the cleanest
rejection signal; under strict lock-in, SEU and GPS must be
read as provenance checks rather than weaker versions of
answer-stateentropy. Onthestrict4-hopslice( 𝑛=58)every
answer is wrong, so AUROC is undefined.
7.5Finding 3: Evidence-state helps unevenly;
retrieval-state is auditable but brittle
Once the answer surface goes silent, evidence-state and
retrieval-statesignalsrecoverpartofthemissingfootprint,
butnotthehardestpresence-lock-incases.Whentheanswer
surface goes silent, the evidence-side question is whether
the retrieved text betrays the error. This matters most in
high-accuracy or near-match settings, where the deployment
questioniswhethertheanswercanbejustifiedbyretrieved
evidence.
SEU’ssignalisdataset-dependent. Itreaches 0.721AU-
ROConthestrictRealMedQAstresstest,butmuchofthat
comes from the all-neutral default over empty-retrieval rows
rather than genuine entailment (Section 7.4). On the strict
2WikiMultiHopQA2-hopsliceitfallsbelowchance. Bridge
questions retrieve passages about the wrong-but-coherent
anchor,andthosepassagesweaklyentailthewronganswer
fromthatneighbourhoodratherthancontradictingit. Onthe
adaptiveRealMedQArun, 74%ofrowshittheall-neutral 0.5
plateaubecausenochunkentailsorcontradictstheanswer,
giving a chance-level AUROC. Evidence support helps in
someregimes,butitsutilityvarieswiththeretrievalpolicy
and the NLI model’s domain competence.
A domain-NLI ablation on PubMedQA (a proxy for
the clinical plateau, not a direct RealMedQA rerun; Ap-
pendix H.5) tests whether the neutral plateau is an NLI-
domain artefact. We recompute SEU with a biomedical-
aware LLM-NLI judge ( gpt-4o-mini ) in place of the pa-
per’sdeberta-large-mnli . AUROC rises from 0.55to
0.64on the dense run and from 0.60to0.63on the KG
run,sopartoftheplateauisindeedageneral-domain-NLI
artefact. The improvement is modest, however, and the
neutral rate does not fall under the LLM judge; it rises. Theplateau is notentirelyan artefact: the retriever genuinely
returns evidence that neither entails nor contradicts many
answers. Sentence-level entailmentscoringis theplausible
refinement, but it requires re-running retrieval rather than
replayinglogs. Theselective-predictioncurvesinFigure5
andthefixed-coverageoperatingpointsinAppendixTable18
show the same regime switch as an abstention decision.
Undertheadaptivepolicy,gatingonSD-UQaloneisthe
strongest simple abstention rule. Under the strict clinical
stress test that rule becomes uninformative: 0.338error
at80%coverage against a 0.330no-abstention base. SEU
gatinginsteadcutstheresidualerrorto 0.269,an18%relative
reductionintheregimewheretheanswersurfaceissilent. A
single global ranking hides this regime-dependent switch.
Retrieval-state diagnostics: calibrated but brittle.If
evidence-state (SEU) is hit-or-miss, retrieval-state (GPS)
offers a different, graph-specific lens, but its utility depends
entirelyonwhethertheretrievalpolicyexposesthegoldentity.
Soft linking fixes coverage, but it cannot fix ranking; the
held-outAUROCsremainpoor. GPSbehaviourisdomain-
dependent. Ontheclinicalshared-corpussettingwhereits
two hyperparameters were calibrated, it is a useful error
ranker ( 0.76adaptive, 0.75strict). On the held-out suites
transfer is uneven: MuSiQue reaches 0.68with per-question
depth,whileHotpotQA,FullWiki,and2WikiMultiHopQA
hovernearorbelowchance(exactAUROCsinthebreakdown
below). Asensitivityreplayover 𝜏∈{0.50,...,0.70} and
𝛾∈{0.2,...,1.0} leaves the held-out open-domain GPS
AUROCs essentially unchanged (SD ≤0.018for HotpotQA,
FullWiki, 2WikiMultiHopQA, and MuSiQue; Appendix
Table 29). The weak 2WikiMultiHopQA ranking is not a
threshold/decayartefact. GPSscoresgraphsupportforthe
generatedanswer,notcorrectness. Itcanrateawrong-but-
coherent neighbourhood as low risk. In the Baltic Cup case
ofTable32,thewrongentityisreachable,GPSis 0.00,SEU
is0.00,andtheanswersurfacehasnear-zerodispersion. The
decomposition makes that failure visiblebut does not solve
it. Retrieval-state AUROC is computed only on rows with
linkable answer entities, with linking failures reported as
abstentions (Table 7).
Evenwherethescalarfails,GPSstillprovidesanauditable
retrieval-state trace of matched entities, paths, and reacha-
16

0% 50% 100%
Accepted coverage π0%20%40%60%80%100%Accuracy after abstentionRealMedQA
0% 50% 100%
Accepted coverage π0%20%40%60%80%100%2WikiMHQA
0% 50% 100%
Accepted coverage π0%20%40%60%80%100%HotpotQA
0% 50% 100%
Accepted coverage π0%20%40%60%80%100%MuSiQue
0% 50% 100%
Accepted coverage π0%20%40%60%80%100%RealMedQA strictSD-UQ (answer-state) SEU (evidence-state) GPS (retrieval-state, where defined) Combined audit No abstentionFigure 5: The useful diagnostic changes with the retrieval regime.Selective prediction with different KG-side diagnostic families:
panels are the four adaptive datasets (RealMedQA, 2WikiMultiHopQA, HotpotQA, MuSiQue) and the RealMedQA strict stress test
(rightmost). Inthestrictstresstest,SD-UQgatinginverts,acceptingthesilentwrongmassfirst,whileSEUgatingclimbstoward 90%.
Curves sort answered KG-RAG queries by increasing uncertainty and report the accuracy retained at each coverage level; the dotted line is
no-abstention accuracyand the shaded bandmarks the noisy <20%-coverage region. GPS andthe combined auditare omitted from the
strict panel because GPS comes from post-hoc replay rather than row-level logs.
bility on every dataset where it is defined, which is useful
forprovenanceregardlessofrankingquality. Theheld-out
weaknessisapropertyoftheretrievalpolicy: onthesesuites
thepolicyoftenleavesthegoldentityoutsidetheretrieved
subgraph even when the answeris correct, so GPS-risk and
correctnessdecouple(wetestthispredictioninAppendixTa-
ble 31). GPS is most reliable as an abstention record, usable
denominator,andtrace-levelsupportcheck;itisonescalarin-
stantiation of the retrieval-state class (anchors, paths, routes,
abstention),notthefamilyitself. Afixed-binreliabilitycheck
makesthesamepointwithoutAUROC:RealMedQAerror
rates rise from 0/39in the lowest GPS-risk bin to 6/42in
the highest, but the open-domain bins are not monotone,
especially on 2WikiMultiHopQA (cf. Appendix Table 16).
Softanswer-entitylinkingreducestheabstentioncost(e.g.
185/218onHotpotQAFullWiki,upfrom 69undersurface
matching); PubMedQA still abstains completely because
binary answers expose no answer entities.
GPS can mis-rank when the retrieval policy makes cor-
rect answers harder to reach than wrong ones: apositive
gold-unreachability gap would tend to invert the signal to-
wardbelow-chanceGPS.Thisisconsistentwithbothcases
examined: on the regenerated HotpotQA FullWiki KG a
negativegap(−20points)goeswithnear-chanceGPS( 0.54),
and on an earlier 2WikiMultiHopQA build a positive gap
(+26points) with below-chance GPS ( 0.38; cf. Appendix
Table 31).
The post-hoc replay on the full 𝑛=230answer log raises
KG-side coverage from 97/223to195/223 and AUROC
from 0.47to0.76(cf.AppendixTable23);thesamereplay
reaches 0.89onthedenseside. RealMedQAisthecalibrated
case: itshowswhatretrieval-statescoringcandeliverona
clinicalsharedcorpus,whiletheheld-outrowsaboveshowthe generalisation limit.
7.6 Concrete cases
TheaggregateAUROCsandroutestatisticssummarisebe-
haviour across questions; the presence-lock-in case traced in
Figure1makesthemechanismconcreteinscores. Answer-
state uncertainty sits at its floor ( DSE=0,SD-UQ≈0 ) and
GPSstayslow( 0.44)becausethewrongentityisgenuinely
reachable,soonlytheevidence-statesignalfires( SEU=1.0
fromcontradictingchunks). Thecaseshowsthatthegraph
can make a wrong reasoning chain visible even when the
scalar retrieval-state score reports low risk, provided the
evidence-support head is read alongside it; the full per-case
audit readout is in Appendix Table 33.
Absencelock-inisthesamelogicwithadifferentface: the
48strict-clinical silent errors leave no graph trace to inspect,
visible only as the empty-retrieval route. Presence hides a
wronganswerinsideacoherenttrace;absenceleavesnotrace
atall,whichiswhythetwovariantsneeddifferentflags. A
consolidatedcasegallery,withsevenverifiedlock-infailures,
four KG successes, and per-case diagnostic readouts, is in
Appendix E (Table 32).
7.7 Finding4: A conjunctivegateyields high
precision but low coverage
Requiringanswer-,evidence-,andretrieval-statechecksto
concur certifies a small slice of answers at precision well
abovethebaserate,atthecostoflowcoverage.Nosingle
non-answer arm proved reliable enough to act alone. The
decisionlayerhastorespectthedifferencebetweentheobjects
being measured, rather than collapse them into one score
17

too early. Empirically, the scores barely move together; that
is the quantitative reason to keep the decomposition visible.
Across the six-snapshot heatmap suite ( 𝑛=803 pooled
non-abstained KG rows), answer-state and evidence-state
dispersion are essentially uncorrelated (Spearman 𝜌=0.03
forSD-UQversusSEU;SEUversusGPS 𝜌=−0.10 ). The
one non-trivial coupling, 𝜌=−0.27 between SD-UQ and
GPS, is largely a sign artefact of GPS’s risk orientation, not
auniversalrisklaw. Thethreescorescarrypartlyseparate
diagnostic signal, even if low correlation alone cannot prove
distinctlatentcauses;thepooledfamily-disagreementscatter
for the adaptive KG setting is in Appendix Figure 8.
Alowanswer-statescoreshouldnotoverridearetrieval-
stateabstention,afragilegraphpath,orhighevidencecontra-
diction. Adeliberatelysimplepercentile-rankcompositecan
capture this family disagreement (Appendix D.7, Table 28);
here we propose a stricter, more auditable policy.
Aconjunctivelow-riskauditrule.Themostconservative
use of the suite turns the decomposition into a single gate,
treating a response as low-risk only when all three checks
concur: lowanswer-stateuncertainty(SD-UQinthelower
halfofitsdatasetdistribution),non-contradictoryevidence
support ( SEU≤0.5 ), and defined, present retrieval-state
support( GPS≤0.5 ). The SEU≤0.5 armactsasahalluci-
nation filter, not a relevance filter: the threshold sits at the
neutral point to catchactivecontradiction in the retrieved ev-
idence, leaving provenance (the GPS arm) to handle missing
or neutral evidence. A neutral, all- 0.5verdict counts as a
pass.
The gate is narrow, admitting only 7.7%of answers ( 86
of1,117), but those it admits are correct 91.9%of the
time against a 69.7%base rate. This pooled figure spans
real per-dataset heterogeneity (from 1.000on the clinical
calibrationdomainto 0.808onthelargerHotpotQA-FullWiki
contribution,Table11),andisreadhereasasummary,not
a single operating point. This is unlikely to be threshold
overfitting: no gate cutoff is fit to correctness labels, since
SEUusesitsneutralentailmentpoint( 0.5)andtheGPSgate
themidpointofthefrozengraph-supportriskscale( 0.5). The
GPS score itself still depends on the RealMedQA-calibrated
linkinganddecayhyperparametersreportedabove. Theonly
data-dependent gate threshold is the SD-UQ cutoff (the per-
datasetmedian,settoholdrelativeanswer-stateriskconstant
acrossdomains). Recomputingthatmedianonarandomhalf
ofeachdatasetandapplyingtheruletotheotherhalfgives
91.8%meanprecision( 5th–95thpercentile 86.7–97.4%over
200splits) at unchanged coverage (Appendix H.3).
Ontheclinicaltargetdomaintheruleselects 22%ofan-
swersat 100%automated-judgeprecision( 48/48). Thisis
an in-domain automated-label upper bound, not evidence
of clinical safety or readiness, and it needs human-expert
validation before any clinical use (Section 8.3). The in-
dependent Llama-3.3-70B judge confirms all 48of these
certified answers with zero label flips, so the cell is robust to
the judge family despite RealMedQA’s low dataset-wide 𝜅Table 11: The conjunctive audit rule is high-precision and low-
coverage,strongestontheclinicaldomain.Per-datasetbehaviour
ontheadaptiveKGruns.Recallisthefractionofallcorrectanswers
the rule selects; Wilson intervals expose the small denominators.
Cells with fewer than roughly ten selected answers (HotpotQA,
2WikiMHQA, MuSiQue) are trace-level, not reliable per-dataset
precision estimates. The rule presumes unselected answers are
routed to ordinary confidence handling or review, not discarded.
Dataset Selected Prec. 95% CI Recall
PubMedQA 0/100 (0%) – – 0.00
RealMedQA 48/223 (22%) 1.000 [0.926, 1.000] 0.23
HotpotQA 6/238 (3%) 0.833 [0.436, 0.970] 0.04
HotpotQA-FW 26/218 (12%) 0.808 [0.621, 0.915] 0.14
2WikiMHQA 4/244 (2%) 1.000 [0.510, 1.000] 0.02
MuSiQue†2/94 (2%) 0.500[0.095, 0.905] 0.03
Pooled 86/1117 (7.7%) 0.919 [0.841, 0.960] 0.10
†Denominator too small to carry statistical weight; reported for completeness, not as
evidence.
(Appendix H.6). The same policy form is applied to every
dataset (Table 11, the cross-dataset generalisation view),
with recall reported because a high-precision selection rule
istrivialwithoutit. ExcludingRealMedQA,itscalibration
domain,therulestillreaches 81.6%pooledprecision( 31/38)
on the five out-of-calibration datasets, so the high-precision
behaviour is not solely an in-domain artefact.
Coverage is structurally limited: the rule cannot select
answers with no linkable answer entity (binary, numeric, or
label-spaceanswers,whereGPSabstainsbyconstruction),
so on those formats it defers rather than selects, and a
deployment would need a separate fallback. On the adaptive
RealMedQA run, SEU is largely at the 0.5neutral plateau
(74%of rows), so the conjunctive rule there effectively
operatesonSD-UQandGPSalone. Theconjunctionbeats
matched-coveragescalargates: SD-UQalonegives 81.4%
precision at the same coverage, and the percentile-rank
combined score gives 88.4%. A dataset-stratified Mantel–
Haenszel check gives the same direction (odds ratio 3.39,
bootstrap 95% CI[1.64,10.99] ; the per-dataset 2×2cells
are in Appendix Table 36).
The rule is exploratory and domain-conditional: strongest
ontheclinicaltargetdomain(partlyanin-domainadvantage,
since GPS was calibrated there at 𝜏=0.60,𝛾=0.4), thin-
coverage on the harder multi-hop sets, and undefined for
label-space answers. The combined audit score beats the
bestsingle-familyscoreononlyhalfofthesixdatasets(cf.
Table28);theadvantageisdataset-dependentandshouldnot
be read as universal.
Per-dataset behaviour exposes both cautionary cases:
2WikiMultiHopQA selects only four answers ( 2%cover-
age) despite weak GPS ranking, while MuSiQue selects
two answers at only 50%precision on a denominator too
small to weigh. Cells with fewer than roughly ten selections
(HotpotQA, 2WikiMultiHopQA, MuSiQue) should be read
as trace-level, not as reliable per-dataset precision estimates.
Theruleisademonstrationofcertifiability: thethreediag-
18

nostic arms can jointly certify a small, high-precision subset.
Itisnotaselect-to-answerpolicythatdiscardstherest. Its
valueismakingthedecompositionactionable;coverageisthe
bindingconstraint,andper-domainvalidationismandatory
beforeanydeployment. CoverageisboundedmainlybyGPS
definability and SEU neutrality, so the refinements above
(soft linking already; sentence-level entailment next) are the
most direct route to higher audit-rule coverage. A simple
learnedalternativedoesnottransferbetterinthisfixed-subset
test: leave-one-dataset-out logistic gating selects 7.5%of
answers at 85.7%precision. With three raw standardised
scalar features (SD-UQ, SEU, and GPS-risk, not their bi-
narised gate decisions) and substantial cross-dataset shift,
conjunctivegatingremainsthemoreauditablepolicy. The
dense substrate gives the same pattern, although this is only
an internal scalar-gate control, not a reimplementation of
SURE-RAG or FRANQ: it asks whether saved dense-side
scalarsignalscanreproducethehigh-precisionrejectionfron-
tier. A learned logistic gate over all seven dense uncertainty
scalars never attains the lowest risk–excess–coverage and
invertsbelowchanceonthelow-errorbiomedicalsets,while
the conjunctive rule attains the lowest AUREC on three of
six datasets (including the clinical target) and the best single
signal wins the other three (Appendix H.4).
8 Conclusion
Retrieval-state lock-in turnsagreement intothe wrongkind
of reassurance. When the retriever repeatedly returns the
sameemptyorwrongstate,moresamplesneednottestmore
possibilities; they may simply repeat the same condition.
Theconfidencequestionchangesfromwhethertheanswer
is stable to which object is stable: the answer surface, the
retrieved evidence, or the retrieval state itself.
We find that answer-state uncertainty remains a useful
default,butaboundedone. Inthedeployable-policyruns, 42–
59%of errors already have zero observed answer dispersion
at𝑁=5, so an answer-only method has no within-question
disagreementsignaltorecover. Evidence-statediagnostics
catchsomeofthesefailuresbytestingsupportintheretrieved
text. GPSaddsagraph-specificviewthatfaithfullyreports
the retrieval state: strong where retrieval exposes the gold
entity (the clinical corpus), weak where the policy does
not (open-domain multi-hop). That off-domain weakness
survives re-calibration (Appendix D.8), so it reflects the
retrievalpolicyratherthanatuninggap,anditsvaluethere
lies in the auditable trace it exposes, not a universal ranking.
Agreement, support, and retrieval grounding are different
observables; a single confidence scalar hides that difference.
Inpractice,thismeansreportinganswer-,evidence-,and
retrieval-statesignalsseparately,andcallingananswerlow-
risk only when all three agree: low answer uncertainty,
adequate evidence support, and present graph support. In
OntoGraphRAG, this conjunctive rule selects a small sub-
set at 91.9%pooled precision against a 69.7%base rate,reachingnear-ceilingautomated-labelprecision( 100%un-
dertheautomatedjudge)ontheclinicaltargetdomainat 22%
coverage. That clinical cell is an in-domain upper bound,
notasafetyclaim,andstillneedshuman-expertvalidation
(Section 8.3). A learned gate does not transfer better here
(Section7.7,AppendixH.4);ahead-to-headagainstexternal
baselines (SURE-RAG, FRANQ) remains an informative
next test. Unselected answers are not thereby wrong, only
uncertified: the answers outside the 7.7%certified subset
areneithercertifiednorrejectedbythisauditrule,andinthe
reported runs they retain the system’s pooled base accuracy
(69.7%). When overall trust is low (the rule does not certify
the answer and the evidence- or retrieval-state diagnostics
alsodisagree),thecaseshouldinsteadbeescalatedtohuman
review or a retrieval-perturbation retry.
Thecasethediagnosticsleaveopenisthehardestpresence
lock-in: thesystemretrievesacoherentbutwrongneighbour-
hood that locally supports the answer, so even theevidence-
andretrieval-statechecksstaycalm. Thatisthesettingwhere
relation-level path faithfulness matters most. Future systems
should test not only whether an answer entity is reachable,
butwhetherthepathusestherelationthequestionactually
asks for.
The evidence is not equally strong throughout. The ob-
served silent-error rates and the answer-state collapse under
strict stress are the firmest pieces; the route decomposi-
tiongivesmoderateevidenceforabsencelock-in;presence
lock-in rests on populated-route cases and a smaller set
of confirmed examples; and reliable detection of a coher-
ent wrong neighbourhood when both SEU and GPS are
calm remains open. The silent-error phenomenon itself is
substrate-general(denseretrievalshowsittoo),whilethefull
three-family audit is easiest on an inspectable graph, where
theretrievalstate isaqueryableobject. Sothecontribution
is not that we solve lock-in, but that we make it nameable,
measurable, and diagnosable: retrieval-state lock-in does
not make RAG uncertainty hopeless, it makes confidence
object-specific. Once the answer, evidence, and retrieval
state are separated, stable agreement stops being a blanket
certificate and becomes a diagnostic clue.
8.1 Limitations
The strict graph-only setting is a mechanism probe, not a
deployablepolicyoraone-componentablation. Itchanges
fusion,reranking,decomposition,andfallbacktogether,so
itdemonstratesaregimeinwhichanswer-stateuncertainty
fails rather than estimating production prevalence.
Each representative score is local or conditional. GPS
is strongest on the calibrated clinical corpus, weak on sev-
eral open-domain multi-hop suites despite the sensitivity
replay (Appendix D.8), and leaves yes/no and other label-
space answers outside its design; the retrieval-state family is
bestreadasanobservableclass(anchors,paths,subgraphs,
routes)ofwhichGPSisonescalarinstantiation. SEUtests
answer–evidence consistency, not whether the model used
19

theevidence;itsneutralplateaucanmakeempty-contextfail-
ureseasiertorankthanpopulatedpresence-lock-infailures,
andwhilethedomain-NLIablation(AppendixH.5)shows
part of that plateau is a general-domain-NLI artefact, the
neutral rate does not fall under a biomedical-aware judge, so
a retriever-side component remains. SD-UQ depends on the
responseembeddingmodel,andallreportednumbersusea
single encoder.
The empirical scale supports the central contrasts but
notfineleaderboardclaimsbetweenneighbouringmetrics.
Correctness labels come froman automated semantic judge
andsurviveanindependentLlama-familyre-judge,buthu-
man or domain-expert adjudication would be needed before
treating silent errors as clinical risk. The study also uses
fixedsmallsubsets,oneKG-RAGframework,onegeneration
model (GPT-4o-mini), and one primary random seed for
subsetconstruction. Theseconstraintsboundthepaperasa
diagnostic methodology and fixed-system empirical audit;
a cross-model survey of lock-in prevalence would require
additional retrievers and generators.
The dense–KG accuracy comparison is measured on
curated benchmark corpora, which flatter dense retrieval:
a retrieval-only dilution probe on the shared HotpotQA-
FullWiki corpus shows gold-passage recall@10 falling from
1.00at a curated 10-candidate pool to 0.55at the full 2,489-
passagecorpus(AppendixH.1). Thenear-matchisaproperty
ofthecuratedregime,notaguaranteeatdeploymentscale,
where an auditable retrieval state matters more rather than
less.
Theretrieval-statediagnosticsaregraph-native,thoughthe
silent-error phenomenon they target is not: dense retrieval
shows it too.
8.2 Future Work
Extendingthesamedecompositiontodense-RAGretrieval
states, for example through citation-path or passage-link
plausibility, is the most direct route to broader generality.
ThedecompositionitselfisnottiedtoOntoGraphRAG:the
answer- and evidence-state arms read only sampled answers
and retrieved text, so they port with minimal adaptation to
otherKG-RAGsystemssuchasGraphRAGorHippoRAG,
whileonlytheretrieval-statearmmustbere-targetedtoeach
system’s trace. A second direction is agentic and iterative-
retrievalRAG,wheretheper-stepdiagnosticslargelycarry
overandcanflagastatethatsuccessivequeriesdeepenrather
than break. The decomposition also points to a control
policy: when answer samples agree but the evidence- or
retrieval-statediagnosticsdisagree,asystemcouldperturb
the retrieval state (for example by masking the dominant
intermediate anchor) and regenerate: if the answer changes,
itwasanchoredtothatstate;ifnot,theevidencewasstable
acrosstheperturbation. Whichanchortomask,howmany
alternatives to test, and when to stop are open questions,
so we report the diagnostics here rather than a validated
controller.8.3 Ethical and Safety Considerations
Nothing in these results licenses autonomous clinical an-
swering. The audit rule is deliberately high precision and
low coverage; unselected answers must not be treated as
audited. The three diagnostic arms differ in cost: SD-UQ is
embedding-only, while SEU and GPS add NLI and graph
queries(AppendixTable21),soalatency-sensitivedeploy-
mentcancascade, screeningwithSD-UQand invokingthe
evidence-andretrieval-statearmsonlyonborderlinecases.
Lock-in also has an adversarial analogue: presence lock-
in is the benign counterpart of retrieval-corpus poisoning,
where a corpus or query is deliberately shaped to anchor
retrieval in a plausible but wrong neighbourhood (Zou et al.,
2025; Zhong et al., 2023). Retrieval diversity constraints
andanchor-maskingcontrolsareplausibledefences,butthey
remain future work. Adversarial robustness lies outside this
evaluation;thediagnosticdecompositionisstillaprerequisite
forit,sinceafailuremodethatcannotbeobservedcannotbe
defended against.
Takeaway for practitioners.Deployments should not gate
reliability on answer-agreement alone. Monitor three distinct
objectsratherthanasingleconfidencescalar: answerdispersion
(SD-UQ), evidence contradiction (SEU), and the retrieval trace
(GPS).WhenanswersagreebutSEUexceedsitsneutralpoint,
GPSabstains,orGPSreportsweaksupport,triggeraretrieval
perturbation or route the case to human review.
References
Lameck Mbangula Amugongo, Pietro Mascheroni,
StevenGeoffreyBrooks,StefanDoering,andJanSeidel.
Retrievalaugmentedgenerationforlargelanguagemodels
in healthcare: A systematic review.PLOS Digital Health,
4(6):e0000877, 2025. doi: 10.1371/journal.pdig.0000877.
URLhttps://journals.plos.org/digitalhealt
h/article?id=10.1371/journal.pdig.0000877.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
HannanehHajishirzi. Self-RAG:Learningtoretrieve,gen-
erate,andcritiquethroughself-reflection. InInternational
Conference on Learning Representations (ICLR), 2024.
Alexandra Bazarova, Andrei Volodichev, Daria Kotova, and
Alexey Zaytsev. INTRYGUE: Induction-aware entropy
gating for reliable RAG uncertainty estimation.arXiv
preprint arXiv:2603.21607, 2026.
Margarida M. Campos, António Farinhas, Chrysoula Zerva,
Mário A. T. Figueiredo, and André F. T. Martins. Confor-
mal prediction for natural language processing: A survey.
TransactionsoftheAssociationforComputationalLinguis-
tics, 12:1619–1638, 2024. doi: 10.1162/tacl_a_00715.
Eason Chen, Chuangji Li, Shizhuo Li, Zimo Xiao, Jionghao
Lin, and Kenneth R. Koedinger. Comparing RAG and
GraphRAG for page-level retrieval question answering on
math textbook.arXiv preprint arXiv:2509.16780, 2025.
20

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley,
AlexChao,ApurvaMody,StevenTruitt,DashaMetropoli-
tansky,RobertOsazuwaNess,andJonathanLarson. From
local to global: A graph RAG approach to query-focused
summarization.arXiv preprint arXiv:2404.16130, 2024.
Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven
Schockaert. RAGAs: Automated evaluation of retrieval
augmented generation. InProceedings of the 18th Con-
ference of the European Chapter of the Association for
ComputationalLinguistics: SystemDemonstrations,pages
150–158, 2024. doi: 10.18653/v1/2024.eacl-demo.16.
URLhttps://aclanthology.org/2024.eacl-dem
o.16/.
Ekaterina Fadeeva, Aleksandr Rubashevskii, Dzianis Pia-
trashyn,RomanVashurin,ShehzaadDhuliawala,Artem
Shelmanov, Timothy Baldwin, Preslav Nakov, Mrinmaya
Sachan,andMaximPanov. Faithfulness-awareuncertainty
quantificationforfact-checkingtheoutputofretrievalaug-
mented generation.arXiv preprint arXiv:2505.21072,
2025.
Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin
Gal. Detecting hallucinations in large language models
using semantic entropy.Nature, 630:625–630, 2024.
Yonatan Geifman and Ran El-Yaniv. Selective classifica-
tion for deep neural networks. InAdvances in Neural
Information Processing Systems (NeurIPS), 2017.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger.
Oncalibrationofmodernneuralnetworks. InProceedings
ofthe34thInternationalConferenceonMachineLearning
(ICML), 2017.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro
Yasunaga, and Yu Su. HippoRAG: Neurobiologically
inspired long-term memory for large language models.
InAdvances in Neural Information Processing Systems
(NeurIPS), 2024.
HaoyuHan, LiMa, YuWang, HarryShomer, YongjiaLei,
ZhishengQi,KaiGuo,ZhigangHua,BoLong,HuiLiu,
CharuC.Aggarwal,andJiliangTang.RAGvs.GraphRAG:
Asystematicevaluationandkeyinsights.arXivpreprint
arXiv:2502.11371, 2025.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu
Chen. DeBERTa: Decoding-enhanced BERT with dis-
entangled attention. InInternational Conference on
Learning Representations (ICLR), 2021. URL https:
//openreview.net/forum?id=XPZIaotutsD.
XiaoxinHe,YijunTian,YifeiSun,NiteshV.Chawla,Thomas
Laurent,YannLeCun,XavierBresson,andBryanHooi.
G-Retriever: Retrieval-augmentedgenerationfortextual
graphunderstandingandquestionanswering. InAdvances
in Neural Information Processing Systems (NeurIPS),
2024.XanhHo,Anh-KhoaDuongNguyen,SakuSugawara,and
AkikoAizawa. Constructingamulti-hopQAdatasetfor
comprehensive evaluation of reasoning steps. InProceed-
ings of the 28th International Conference on Computa-
tional Linguistics (COLING), 2020.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling,
and Liang Zhao. GRAG: Graph retrieval-augmented gen-
eration. InFindings of the Association for Computational
Linguistics: NAACL2025,pages4145–4157,2025. doi:
10.18653/v1/2025.findings-naacl.232. URL https://
aclanthology.org/2025.findings-naacl.232/.
Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qi-
huiZhang,YuanLi,ChujieGao,YixinHuang,Wenhan
Lyu, Yixuan Zhang, Xiner Li, et al. TrustLLM: Trust-
worthiness in Large Language Models.arXiv preprint
arXiv:2401.05561, 2024.
JinhaoJiang,KunZhou,ZicanDong,KemingYe,XinZhao,
and Ji-Rong Wen. StructGPT: A general framework for
large language model to reason over structured data. In
Proceedingsofthe2023ConferenceonEmpiricalMethods
in Natural Language Processing (EMNLP), 2023a.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian
Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and
GrahamNeubig. Activeretrievalaugmentedgeneration.
InProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages 7969–
7992,Singapore,2023b.AssociationforComputational
Linguistics. doi: 10.18653/v1/2023.emnlp-main.495.
URLhttps://aclanthology.org/2023.emnlp-mai
n.495/.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer,
ZacHatfield-Dodds,NovaDasSarma,EliTran-Johnson,
Scott Johnston, Sheer El-Showk, Andy Jones, Nelson
Elhage,TristanHume,AnnaChen,YuntaoBai,SamBow-
man, Stanislav Fort, Deep Ganguli, Danny Hernandez,
Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane
Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer,
Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph,
BenMann,SamMcCandlish,ChrisOlah,andJaredKa-
plan. Language models (mostly) know what they know.
arXiv preprint arXiv:2207.05221, 2022.
JannikKossen,JiatongHan,MuhammedRazzak,LisaSchut,
Shreshth Malik, and Yarin Gal. Semantic entropy probes:
Robust and cheap hallucination detection in LLMs. In
International Conference on Learning Representations
(ICLR), 2025.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic
uncertainty: Linguistic invariances for uncertainty esti-
mation in natural language generation. InInternational
Conference on Learning Representations (ICLR), 2023.
21

Chorok Lee. Decomposing uncertainty in probabilistic
knowledge graph embeddings: Why entity variance is
not enough.arXiv preprint arXiv:2512.22318, 2025.
PatrickLewis,EthanPerez,AleksandraPiktus,FabioPetroni,
Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian
Riedel,andDouweKiela. Retrieval-augmentedgeneration
forknowledge-intensiveNLPtasks. InAdvancesinNeural
Information Processing Systems (NeurIPS), 2020.
Mufei Li, Siqi Miao, and Pan Li. Simple is effective: The
rolesofgraphsandlargelanguagemodelsinknowledge-
graph-basedretrieval-augmentedgeneration. InInterna-
tional Conference on Learning Representations (ICLR),
2025a.
Weitao Li, Junkai Li, Weizhi Ma, and Yang Liu. Citation-
enhanced generation for LLM-based chatbots. InPro-
ceedingsofthe62ndAnnualMeetingoftheAssociation
forComputationalLinguistics(Volume1: LongPapers),
pages 1451–1466, 2024a. doi: 10.18653/v1/2024.acl-l
ong.79. URL https://aclanthology.org/2024.ac
l-long.79/.
Xiaomin Li, Zhou Yu, Ziji Zhang, Yingying Zhuang, Swair
Shah, Narayanan Sadagopan, and Anurag Beniwal. Se-
mantic volume: Quantifying and detecting both exter-
nal and internal uncertainty in LLMs.arXiv preprint
arXiv:2502.21239, 2025b.
Zixuan Li, Jing Xiong, Fanghua Ye, Chuanyang Zheng,
Xun Wu, Jianqiao Lu, Zhongwei Wan, Xiaodan Liang,
Chengming Li, Zhenan Sun, Lingpeng Kong, and Ngai
Wong. UncertaintyRAG:Span-leveluncertaintyenhanced
long-contextmodelingforretrieval-augmentedgeneration.
arXiv preprint arXiv:2410.02719, 2024b.
Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching
models to express their uncertainty in words. InTrans-
actionsoftheAssociationforComputationalLinguistics
(TACL), 2022.
HuanshuoLiu,HaoZhang,ZhijiangGuo,JingWang,Kuicai
Dong,XiangyangLi,YiQuanLee,CongZhang,andYong
Liu. CtrlA:Adaptiveretrieval-augmentedgenerationvia
inherentcontrol. InFindingsoftheAssociationforCom-
putational Linguistics: ACL 2025, pages 12592–12618,
Vienna, Austria, 2025. Association for Computational
Linguistics. doi: 10.18653/v1/2025.findings-acl.652.
URLhttps://aclanthology.org/2025.findings
-acl.652/.
Jiayu Liu, Rui Wang, Qing Zong, Yumeng Wang, Cheng
Qian, Qingcheng Zeng, Tianshi Zheng, Haochen Shi,
Dadi Guo, Baixuan Xu, Chunyang Li, and Yangqiu Song.
NAACL: Noise-AwAre verbal confidence calibration for
robust large language models in RAG systems.arXiv
preprint arXiv:2601.11004, 2026a.Shuyi Liu, Yuming Shang, and Xi Zhang. TruthfulRAG:
Resolving factual-level conflicts in retrieval-augmented
generation with knowledge graphs. InProceedings of the
AAAI Conference on Artificial Intelligence, 2026b.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui
Pan. Reasoningongraphs: Faithfulandinterpretablelarge
language model reasoning. InInternational Conference
on Learning Representations (ICLR), 2024.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren
Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. Think-
on-graph 2.0: Deep and faithful large language model
reasoning with knowledge-guided retrieval augmented
generation. InInternational Conference on Learning
Representations (ICLR), 2025.
PotsaweeManakul,AdianLiusie,andMarkJ.F.Gales. Self-
CheckGPT:Zero-resourceblack-boxhallucinationdetec-
tion for generative large language models. InProceedings
of the 2023 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2023.
Costas Mavromatis and George Karypis. GNN-RAG: Graph
neural retrieval for large language model reasoning. In
Findings of the Association for Computational Linguistics
(ACL), 2025.
ViktorMoskvoretskii,MariaMarina,MikhailSalnikov,Niko-
layIvanov,SergeyPletenev,DariaGalimzianova,Nikita
Krayko,VasilyKonovalov,IrinaNikishina,andAlexander
Panchenko. Adaptive retrieval without self-knowledge?
bringing uncertainty back home. InProceedings of the
63rdAnnualMeetingoftheAssociationforComputational
Linguistics(Volume1: LongPapers),pages6355–6384,
Vienna, Austria, 2025. Association for Computational
Linguistics. doi: 10.18653/v1/2025.acl-long.319. URL
https://aclanthology.org/2025.acl-long.31
9/.
Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying
Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Yinglong
Xia,KrishnaramKenthapadi,etal. Towardstrustworthy
retrieval augmented generation for large language models:
A survey.arXiv preprint arXiv:2502.06872, 2025.
Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka
Marttinen. Kernel language entropy: Fine-grained uncer-
tainty quantification for LLMs from semantic similarities.
InAdvances in Neural Information Processing Systems
(NeurIPS), 2024.
OpenAI. GPT-4omini: Advancingcost-efficientintelligence.
Technical report, OpenAI, 2024. URL https://openai
.com/index/gpt-4o-mini-advancing-cost-eff
icient-intelligence/.
ArjunPanickssery,SamuelR.Bowman,andShiFeng. LLM
evaluatorsrecognizeandfavortheirowngenerations. In
22

Advances in Neural Information Processing Systems 37
(NeurIPS), 2024.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou
Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph
retrieval-augmented generation: A survey.ACM Transac-
tions on Information Systems, 2024.
Laura Perez-Beltrachini and Mirella Lapata. Uncertainty
quantification in retrieval augmented question answering.
Transactions on Machine Learning Research (TMLR),
2025.
Jingxi Qiu, Zeyu Han, and Cheng Huang. SURE-RAG:
Sufficiencyanduncertainty-awareevidenceverificationfor
selectiveretrieval-augmentedgeneration.arXivpreprint
arXiv:2605.03534, 2026.
Xin QiuandRistoMiikkulainen. Semantic density: Uncer-
tainty quantification for large language models through
confidencemeasurementinsemanticspace.InAdvancesin
Neural Information Processing Systems (NeurIPS), 2024.
NilsReimersandIrynaGurevych.Sentence-BERT:Sentence
embeddings using siamese BERT-networks. InProceed-
ings of the 2019 Conference on Empirical Methods in
Natural Language Processing (EMNLP), 2019.
Jing Ren, BowenLi, Ziqi Xu, Xikun Zhang, Haytham Fayek,
andXiaodongLi. Whentotrust: Acausality-awarecali-
bration framework for accurate knowledge graph retrieval-
augmented generation.arXiv preprint arXiv:2601.09241,
2026.
JonSaad-Falcon,OmarKhattab,ChristopherPotts,andMatei
Zaharia. ARES: An automated evaluation framework for
retrieval-augmentedgenerationsystems. InProceedings
of the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies, pages 338–354, Mexico City,
Mexico, 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.naacl-long.20. URL https:
//aclanthology.org/2024.naacl-long.20/.
Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual
Merita, Shriram Piramanayagam, Enting Chen, Damien
Graux, Andre Melo, Ruofei Lai, Zeren Jiang, Zhongyang
Li, Ye Qi, Yang Ren, Dandan Tu, and Jeff Z. Pan. GeAR:
Graph-enhanced agent for retrieval-augmented genera-
tion. InFindings of the Association for Computational
Linguistics: ACL 2025, pages 12049–12072, 2025. doi:
10.18653/v1/2025.findings-acl.624. URL https:
//aclanthology.org/2025.findings-acl.624/.
Heydar Soudani, Evangelos Kanoulas, and Faegheh Ha-
sibi. Why uncertainty estimation methods fall short in
RAG: An axiomatic analysis. InFindings of the As-
sociation for Computational Linguistics: ACL 2025,pages 16596–16616, Vienna, Austria, 2025a. Associ-
ation for Computational Linguistics. URL https:
//aclanthology.org/2025.findings-acl.852/.
HeydarSoudani,HamedZamani,andFaeghehHasibi. Un-
certainty quantification for retrieval-augmented reasoning.
arXiv preprint arXiv:2510.11483, 2025b.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and
Yiqun Liu. DRAGIN: Dynamic retrieval augmented gen-
eration based on the real-time information needs of large
language models. InProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 12991–13013, Bangkok,
Thailand,2024.AssociationforComputationalLinguis-
tics. doi: 10.18653/v1/2024.acl-long.702. URL https:
//aclanthology.org/2024.acl-long.702/.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Shengjie
Wang,ChenLin,YeyunGong,Heung-YeungShum,and
Jian Guo. Think-on-graph: Deep and responsible rea-
soningoflargelanguagemodelonknowledgegraph. In
International Conference on Learning Representations
(ICLR), 2024.
Yu Takahashi, Shun Takeuchi, Kexuan Xin, Guillaume
Pelat,YoshiakiIkai,JunyaSaito,JonathanVitale,Shlomo
Berkovsky, and Amin Beheshti. Uncertainty-aware dy-
namicknowledgegraphsforreliablequestionanswering.
arXiv preprint arXiv:2601.09720, 2026.
KatherineTian,EricMitchell,HuaxiuYao,ChristopherD.
Manning, and Chelsea Finn. Does my LLM need a better
evaluator? Just ask for calibration. InarXiv preprint
arXiv:2310.02415, 2023.
Christian Tomani, Kamalika Chaudhuri, Ivan Evtimov,
Daniel Cremers, and Mark Ibrahim. Uncertainty-based
abstentioninLLMsimprovessafetyandreduceshalluci-
nations.arXiv preprint arXiv:2404.10960, 2024.
Harsh Trivedi, Niranjan Bauer, Tushar Khot, and Ashish
Sabharwal. MuSiQue: Multihop questions via single-hop
question composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
Pragatheeswaran Vipulanandan, Kamal Premaratne, and
DilipSarkar. Semanticuncertaintyquantificationofhal-
lucinations in LLMs: A quantum tensor network based
method.arXiv preprint arXiv:2601.20026, 2026.
Sergii Voloshyn. L-RAG: Balancing context and re-
trieval with entropy-based lazy loading.arXiv preprint
arXiv:2601.06551, 2026.
NassimWalha,SebastianG.Gruber,ThomasDecker,Yin-
chongYang,AlirezaJavanmardi,EykeHüllermeier,and
FlorianBuettner. Fine-graineduncertaintydecomposition
in large language models: A spectral approach.arXiv
preprint arXiv:2509.22272, 2025.
23

JonasWallat,MariaHeuss,MaartendeRijke,andAvishek
Anand. Correctness is not faithfulness in retrieval aug-
mented generation attributions. InProceedings of the
2025 International ACM SIGIR Conference on Innovative
Concepts and Theories in Information Retrieval, pages
22–32. Association for Computing Machinery, 2025. doi:
10.1145/3731120.3744592.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and
Sercan O. Arik. Astute RAG: Overcoming imperfect
retrievalaugmentationandknowledgeconflictsforlarge
language models. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics,
2025.
XuezhiWang,JasonWei,DaleSchuurmans,QuocLe,EdChi,
Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.
Self-consistency improves chain of thought reasoning
in language models. InProceedings of the Eleventh
International Conference on Learning Representations
(ICLR), 2023.
Yu Wang, Nedim Lipka, Ruiyi Zhang, Alexa Siu, Yuying
Zhao, Bo Ni, Xin Wang, Ryan Rossi, and Tyler Derr.
Augmenting textual generation via topology aware re-
trieval.arXiv preprint arXiv:2405.17602, 2024. doi:
10.48550/arXiv.2405.17602.
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min
Xu, Filippo Menolascina, Yueming Jin, and Vicente
Grau. Medical graph RAG: Evidence-based medical
large languagemodel via graphretrieval-augmented gen-
eration. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 28443–28467, 2025. doi:
10.18653/v1/2025.acl-long.1381. URL https:
//aclanthology.org/2025.acl-long.1381/.
KevinWu,EricWu,AllyCassasola,AngelaZhang,Kevin
Wei, Teresa Nguyen, Sith Riantawan, Patricia Shi Ri-
antawan, Daniel E. Ho, and James Zou. How well do
LLMs cite relevant medical references? An evaluation
frameworkandanalyses.arXivpreprintarXiv:2402.02008,
2024a.
Kevin Wu, Eric Wu, and James Zou. ClashEval: Quan-
tifying the tug-of-war between an LLM’s internal prior
andexternal evidence.arXiv preprintarXiv:2404.10198,
2024b.
ZhishangXiang,ChuanjieWu,QinggangZhang,Shengyuan
Chen, Zijin Hong, Xiao Huang, and Jinsong Su. When
to use graphs in RAG: A comprehensive analysis for
graph retrieval-augmented generation.arXiv preprint
arXiv:2506.05690, 2025.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
Correctiveretrievalaugmentedgeneration.arXivpreprint
arXiv:2401.15884, 2024.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William W. Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. HotpotQA: A dataset for diverse, ex-
plainablemulti-hopquestionanswering. InProceedings
of the 2018 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2018.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao, Linmei
Hu, Liu Weichuan, Lei Hou, and Juanzi Li. SeaKR: Self-
awareknowledgeretrievalforadaptiveretrievalaugmented
generation. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume
1: Long Papers), pages 27022–27043, Vienna, Austria,
2025. Association for Computational Linguistics. doi:
10.18653/v1/2025.acl-long.1312. URL https://acla
nthology.org/2025.acl-long.1312/.
Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang,
Junhui Li, Xinrun Wang, and Jinsong Su. Faithful-
RAG: Fact-level conflict modeling for context-faithful
retrieval-augmented generation. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics, pages 21863–21882. Association
for Computational Linguistics, 2025. URL https:
//aclanthology.org/2025.acl-long.1062/.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang,ZhanghaoWu,YonghaoZhuang,ZiLin,Zhuo-
hanLi,DachengLi,EricP.Xing,HaoZhang,JosephE.
Gonzalez,andIonStoica. JudgingLLM-as-a-judgewith
MT-benchandchatbotarena. InAdvancesinNeuralInfor-
mation Processing Systems 36 (NeurIPS), Datasets and
Benchmarks Track, 2023.
WenqingZheng,DmitriKalaev,NoahFatsi,DanielBarcklow,
OwenReinert,IgorMelnyk,SenthilKumar,andC.Bayan
Bruss. Revisiting RAG retrievers: An information theo-
retic benchmark.arXiv preprint arXiv:2602.21553, 2026.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi
Chen. Poisoning retrieval corpora by injecting adversarial
passages. InConferenceonEmpiricalMethodsinNatural
Language Processing (EMNLP), 2023.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
HongkuanZhou,YuanHe,JiaoyanChen,SteffenStaab,
and Evgeny Kharlamov. What breaks knowledge graph
based RAG? benchmarking and empirical insights into
reasoningunderincompleteknowledge. InProceedings
of the 19th Conference of the European Chapter of the
Association for Computational Linguistics (EACL), 2026.
XiangrongZhu, YuexiangXie, YiLiu,Yaliang Li,andWei
Hu. Knowledge graph-guided retrieval augmented gen-
eration. InProceedings of the 2025 Conference of the
Nations of the Americas Chapter of the Association for
ComputationalLinguistics: HumanLanguageTechnolo-
gies(Volume1: LongPapers),pages8912–8924,2025a.
24

doi: 10.18653/v1/2025.naacl-long.449. URL https:
//aclanthology.org/2025.naacl-long.449/.
Yuqicheng Zhu, Jingcheng Wu, Yizhen Wang, Hongkuan
Zhou, Jiaoyan Chen, Evgeny Kharlamov, and Steffen
Staab. Certainty in uncertainty: Reasoning over uncertain
knowledge graphs with statistical guarantees. InProceed-
ings of the 2025 Conference on Empirical Methods in
NaturalLanguageProcessing,pages8730–8752,2025b.
doi: 10.18653/v1/2025.emnlp-main.441. URL https:
//aclanthology.org/2025.emnlp-main.441/.
IlanaZimmerman,JadinTredup,EthanSelfridge,andJoseph
Bradley. Two-tieredencoder-basedhallucinationdetection
for retrieval-augmented generation in the wild. InPro-
ceedingsofthe2024ConferenceonEmpiricalMethodsin
NaturalLanguageProcessing: IndustryTrack,pages8–22,
Miami, Florida, US, 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.emnlp-industry.2.
URLhttps://aclanthology.org/2024.emnlp-ind
ustry.2/.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia.
PoisonedRAG: Knowledge corruption attacks to retrieval-
augmented generation of large language models. In
USENIX Security Symposium, 2025.
Hanna Zubkova, Ji-Hoon Park, and Seong-Whan Lee.
SUGAR: Leveraging contextual confidence for smarter
retrieval.arXiv preprint arXiv:2501.04899, 2025.
25

A Residual-Mass Bound for Stabilised Retrieval
Thecollapsed-contextapproximationinSection3canbereadasasimplemixtureargument. Let 𝛼=𝑝(𝑐★|𝑞)anddefine
the residual context mixture
𝜋res(𝑟)=1
1−𝛼∑︁
𝑐∈C(𝑞)\{𝑐★}𝑝(𝑟|𝑞,𝑐)𝑝(𝑐|𝑞),
whenever𝛼<1. Then
𝑝(𝑟|𝑞)=𝛼𝑝(𝑟|𝑞,𝑐★)+(1−𝛼)𝜋 res(𝑟),
and therefore𝑝(𝑟|𝑞)−𝑝(𝑟|𝑞,𝑐★)
TV=(1−𝛼)𝜋res(𝑟)−𝑝(𝑟|𝑞,𝑐★)
TV≤1−𝛼.
Theboundisnotanestimator;itonlystateswhenthestylisedapproximationismeaningful: theresidualroutingmassmustbe
small.
B KG Construction and System Configuration
B.1 Knowledge Graph Construction Pipeline
The KG is built passage-wise with chunk-level provenance, so every triple and path traces back to the text that licensed it.
Text chunking.Input documents are split into overlapping passages using a token-aware recursive splitter (tiktoken-based,
with a character-based fallback when the tokenizer is unavailable) with chunk size 𝐿=1,500 tokens and overlap 𝛿=200
tokens. Each chunk receives an embedding at construction time using all-MiniLM-L6-v2 (384 dimensions) from Sentence
Transformers(ReimersandGurevych,2019). ChunksarestoredasChunknodesinNeo4jwithaSHA-1contenthashas
identifier and aPart_Ofedge to their parentDocumentnode.
Entityandrelationextraction.EntitiesandtypedrelationsareextractedfromeachchunkbyapromptedGPT-4o-mini
call (OpenAI, 2024) returning a JSON object with an entities array (fields id,type,name,description ) and a
relationshipsarray (source,target,type,description). Two extraction modes are supported:
1.Ontology-guidedextraction.AprovideddomainschemaisparsedfromJSONorOWL/RDFintoonetypedinternal
representation; the extraction prompt then carries the ontology entity types (with descriptions and typed properties) and
relationship types (with domain/range/cardinality constraints). After extraction, entity types are normalised against
ontology class labels by embedding cosine similarity (threshold 𝜏=0.50), with substring matching and keyword
heuristics as fallback, and relationship labels are canonicalised to the closest schema-compatible type by domain/range
compatibility and fuzzy matching. Biomedical and clinical types includeDisease,Treatment,Symptom,Biomarker,
andAnatomy.
2.Open extraction.Without a type schema, the LLM extracts entities and relations freely. Used for the open-domain
datasets (HotpotQA, HotpotQA FullWiki, 2WikiMultiHopQA, and MuSiQue).
Thesystemdoesnotinduceanontologyfromthedocuments;itloadsauser-providedschemaandusesittoguideextraction,
typing, and relationship normalisation.
Neo4jschema.EntitiesareEntitynodeswithcompoundlabels :__Entity__:Type (e.g.:__Entity__:Disease ),each
storing an embedding of its name and description and linked to originating chunks viaMentioned_Inedges. Extracted
relationtypes(e.g.Treats,Causes,Has_Complication)aretypededgesbetweenentitynodes. Threeindexesaremaintained:
vector indexes over chunk and entity embeddings, and a full-text keyword index over chunk content.
KG scope.Each dataset’s KG is a named graph (a kgNameattribute on all nodes), so multiple datasets coexist in one Neo4j
instancewithout interference. The experimentpipelinesupports twobuild-scopemodes selectedby –dataset-kg-scope :
evaluation_subset(default) builds the KG from the same question slice being evaluated, which minimises KG size and
keepsretrievalandknowledgeinthe sameclosedcorpus;full_datasetbuildsfrom allavailable passagesinthe normalised
datasetbefore evaluatingtherequestedsubset. Unless explicitlynotedasa diagnostic,thereportedfixed-snapshot runs use
evaluation_subset so that the KG and dense retriever see the same closed corpus slice. This improves experimental
control, but it also suppresses some of the graph-incompleteness and corpus-drift failures that would matter more in a
deployment-scale setting.
26

Table 12:System configuration for all experiments.
Parameter Vanilla RAG KG-RAG
Chunk size (tokens) 1,500 1,500
Chunk overlap (tokens) 200 200
Embedding modelsentence-transformers/all-MiniLM-L6-v2(via sentence-transformers, 384-dimensional)
Top-𝑘chunks 10 10
Adjacent chunk expansion yes (pos±1) –
Similarity threshold 0.10 0.10
Entity matching –mention ANN (≥0.72) when query entities are extracted;
otherwise query ANN (≥0.55); plus exact/synonym matching
Chunk sufficiency threshold – 2 chunks
Graph scoring –local PPR over the retrieved entity subgraph;
fallback to hop prior when no graph edges are available
Graph expansion hops – 2 (4 for MuSiQue)
Iterative local hop cap –min(ℎ,3)per sub-question
Neighbours per seed entity – 30
Max seed entities in prompt – 15
Max neighbour entities – 10
Max direct relationships – 25
Iterative decomposition – enabled when the hop target isℎ≥2
Fallback cascade vector entity-first→graph expansion→vector→text-keyword
LLM (generation) GPT-4o-mini
Samples per question (𝑁) 5
Relationshipverification.Eachtypedrelationshipisverifiedagainstthechunksthatcontributedatleastoneofitsendpoint
entities (tracked via provenance positions stored at build time), so verification NLI runs only over the source chunks, not the
full corpus; this reduces false-positive edges from entity-name collisions across unrelated passages.
Builderprofilesandgraphrepair.TherunnerexposesthreeKG-builderprofiles. full(usedforallfull-metricsrunsunless
statedotherwise)enablesanchor-constrainedextraction,self-reflectionovermissingentities,anchor-coveragesupplementation,
cross-passage relation recovery, low-confidence triple reverification, soft entity linking, fragmentation repair, optional graph
summaries, and claim extraction; on biomedical and clinical datasets it also attempts UMLS-backed normalisation when the
local SciSpaCy/UMLS stack is available, continuing without UMLS metadata otherwise rather than failing. lightweight
disablestheexpensiveanchor,reflection,andcross-passageextrasforquickaccuracy-onlysweeps; autoselectsitonlyfor
those sweeps and otherwise resolves to full. Soft linking and fragmentation repair are deliberately conservative: the builder
canonicalisesobviousaliasesandnear-duplicates,thenaddsbridgerecordsonlyforhigh-similarityfragmentsratherthan
freely rewiring the graph. Claim extraction and graph summaries are stored as additional artefacts that enrich inspection, but
the eight headline uncertainty measures do not depend on them as separate scoring heads.
B.2 System Configuration
Table 12 summarises the hyperparameters used across all experiments.
Configuration note.All reported runs use the 𝑘=10and similarity-threshold 0.10entries of Table 12, under the
final_pair retrieval-study profile: vanilla RAG runs only dense_floor and KG-RAG only kg_entity_first , avoiding
theearlierfour-waycrossingofdenseandgraphvariants. ThestrictRealMedQAstresstestusestheseparate strict_entity
profile, which disables query fusion, late interaction, dense fallback, vector augmentation, decomposition, and the KG-RAG
runtimeguardrail. Becauseiterativedecomposition isitselfa sourceofretrievalvariability, Equation (3)isa stylisedlimit,
not a claim that every KG-RAG call is deterministic; the stabilised regime is strongest when decomposition and entity
anchorsconvergetothesameneighbourhoodacrossrepeatedcalls. Thereleasedmanifestsrecordthereranking,fallback,and
query-fusion flags, but this paper does not sweep𝑘, similarity threshold, or routing flags as independent factors.
27

C Implementation Details and Reproducibility
Questionsampling.PubMedQAandMuSiQueretaintheoriginalfixed 𝑛=100subsets;RealMedQAisreportedonits
full𝑛=230evaluableset (smallenoughtoruninfull andthemainshared-corpusclinical diagnostic); HotpotQA, HotpotQA
FullWiki, and 2WikiMultiHopQA use fixed 𝑛=250subsets drawn with the same seed to reduce denominator volatility.
Clean-accuracy analyses can have smaller effective denominators after excluding provider-side generation failures; those
answered counts are in the run artefactsand summarised in the main text. Subset sampling is deterministic: given a dataset,
samplesize𝑛,andseed𝑠,thepipelinedrawsafixedsetofquestionIDs,persistsit,andreusesitonlaterrunsunlessanew
subset is requested, keeping KG construction and evaluation aligned. Exact IDs are stored in the run artefacts and subset
manifests.
KG construction LLM.Reported KG builds use GPT-4o-mini (via OpenRouter in the latest local runs) for extraction,
temperature-locked at 𝑇=0.0even when answer sampling uses a higher temperature. Biomedical and clinical datasets (Pub-
MedQA,RealMedQA)useontology-guidedextractionwithadomaintypeschema(Disease,Treatment,Symptom,Biomarker,
Anatomy); open-domain datasets (HotpotQA, HotpotQA FullWiki, 2WikiMultiHopQA, MuSiQue) use unconstrained open
extraction.
Training and fine-tuning configuration.No task-specific training, fine-tuning, or gradient-based calibration is performed:
the generation, judge, NLI, and embedding models are all frozen off-the-shelf components, and the only object that changes
acrossexperimentsisthedataset-specificKG.Unlessarunsets –judge-provider or–judge-model ,thecorrectnessjudge
uses the same GPT-4o-mini backend as generation. Uncertainty samples use 𝑇=1.0with𝑁=5responses per question;
accuracy generation and KG extraction use 𝑇=0.0. Final-stage retrieval selection uses retrieval temperature 0.0, with a
shortlist factor of4retained in the code for stochastic sweeps.
NLI model.DSE and the P(True)-style proxy cluster responses by bidirectional NLI entailment using
microsoft/deberta-large-mnli (He et al., 2021), run locally; SelfCheckGPT uses roberta-large-mnli for pairwise
contradictionscoring. Entailmentdecisionsarelabel-based(argmaxoverthethreeclasses): tworesponsesshareacluster
when at least one direction predicts entailment and neither predicts contradiction.
Embeddings for geometric metrics.VN-Entropy, SD-UQ, and SRE-UQ embed sampled responses with
all-MiniLM-L6-v2 (384-d), the same model used for chunk and entity indexing; response embeddings are computed at
evaluation time, not cached.
Metric hyperparameters.GPS: maximum path length ℎ=2hops (4 for MuSiQue), matching the retrieval hop depth; soft
answer-entitylinking with all-MiniLM-L6-v2 cosinethreshold 𝜏=0.60anddistance decay 𝛾=0.4, bothselected onthe
RealMedQA development run over a small grid and applied frozen to all other datasets. Reported GPS values come from a
post-hoc replay on the saved answer logs and the persistent dataset KGs: no answer generation, document retrieval, or LLM
callisrerun. FortheprimaryGPSnumbersthestoredlinkedentitiesandpathlengthsarereused;thecoverage-raisingand
gold-reachabilityreplaysinsteadrecomputeentityalignmentandpathsupportagainstthepersistentKG.Replayartefacts
arerecordedunder results/analyses/ . SD-UQ:𝑘max=min(𝑁−1,8) principalcomponents(effective 𝑘max=4forthe
reported𝑁=5samples),𝜀=10−12.
Compute environment.All experiments ran on a MacBook Pro (Apple M4, macOS 15.5 ARM64, 14 cores, 52GB RAM),
with NLI and embeddings on CPU. Because KG extraction and generation call GPT-4o-mini through an API, wall-clock time
isdominatedbyremotecalls: roughly8–12hoursforthecore-suitepass,longerforthelargershared-corpusandablation
runs(theHotpotQAFullWiki 𝑛=250stressruntookaboutaday). Thesetimingsarespecifictothelaptop-and-APIsetup
and would improve with local GPU serving.
Effective denominators and bootstrap intervals.The main text reports point estimates because the analysis is organised
around family-level behaviour and qualitative traces, not hypothesis tests over small AUROC gaps. To make the scale
explicit, the appendix reports effective answered counts (after provider-failure filtering), usable GPS denominators (after
fallback-abstention filtering), and percentile-bootstrap 95% intervals for the headline AUROCs. All intervals use 𝐵=2000
resamplesoveransweredquestionswithineachdatasetandsystem,atafixedseedof42. ForHotpotQAand2WikiMultiHopQA
the remaining GPS denominators are small enough to make the corresponding AUROCs descriptive.
28

Table13:Summaryofreportedrunartefacts: dataset,subsetseed,samplesize,evaluationmode,andKGbuildscopeforeachreported
snapshot. All runs use GPT-4o-mini as the generation model and𝑁=5uncertainty samples.
Dataset Subset seed𝑛Evaluation mode KG build scope
PubMedQA 42 100 full_metrics evaluation_subset
RealMedQA (adaptive) 42 230 full_metrics evaluation_subset
RealMedQA (strict) 42 230 full_metrics evaluation_subset
HotpotQA 42 250 full_metrics evaluation_subset
HotpotQA FullWiki 42 250 full_metrics evaluation_subset
2WikiMultiHopQA (adaptive) 42 250 full_metrics evaluation_subset
2WikiMultiHopQA (strict) 42 250 full_metrics evaluation_subset
MuSiQue 42 100 full_metrics evaluation_subset
Table 14:Answered and failed rows in the reported saved runs. Failures are provider- or pipeline-side generation failures and are excluded
fromcleanaccuracybutretainedinrawaccounting. TheRealMedQAadaptive–strictcomparisonisanalysedonthe 196questionsanswered
by both KG policies.
Dataset Dense ans. Dense fail KG ans. KG fail Strict ans. Strict fail
PubMedQA 100/100 0 100/100 0 – –
RealMedQA 219/230 11 223/230 7 200/230 30
HotpotQA 226/250 24 238/250 12 – –
HotpotQA FullWiki 208/250 42 218/250 32 – –
2WikiMHQA 212/250 38 244/250 6 245/250 5
MuSiQue 90/100 10 94/100 6 – –
Table 16 gives a descriptive reliability view: the clinical calibration domain has the expected monotone shape, several
open-domain rows do not, which is why GPS is treated as a conditional graph-support diagnostic rather than a calibrated
probability.
Depth-matchedGPSscoring.GPSusesadepth-matcheddecay, 𝛾|𝐿𝑒−ˆ𝐿|,where ˆ𝐿istheexpectedreasoningdepth: the
question’sloggedhopcountwhereavailable(2WikiMultiHopQA,MuSiQue),thedataset’snominaldepthbyconstruction
(HotpotQA variants, ˆ𝐿=2), and ˆ𝐿=1otherwise. Because ˆ𝐿=1on RealMedQA, the calibration domain is untouched
andthefrozen(𝜏,𝛾)carryover;allotherrowsaresingle-shotevaluationsofthesamedeclareddefinition. ThisistheGPS
definition used in all main tables.
C.1 Answer-State Baseline Metric Definitions
These are the five borrowed answer-state estimators summarised in Section 5.1. Each is applied unchanged; only SD-UQ, the
answer-state score introduced here, is defined in the main text.
SemanticentropyandDSE.Kuhnetal.(2023)clustersampledresponsesintomeaning-equivalentsets C={𝐶 1,...,𝐶𝐾}
using bidirectional NLI entailment. The API setting exposes no reliable per-sample likelihoods, so the reported score is the
count-weighted black-box variant,
DSE(𝒓)=−𝐾∑︁
𝑘=1|𝐶𝑘|
𝑁log|𝐶𝑘|
𝑁.(12)
DSE is semantic entropy with uniform sample weights; likelihood-weighted SE is not reported separately.
P(True). P(True)(Kadavathetal.,2022)isimplementedasablack-boxcluster-agreementproxy: thefractionofsamplesin
thesamesemanticclusterasthefirstresponse( P(True)proxy=|{𝑖: cl(𝑟 𝑖)=cl(𝑟 1)}|/𝑁);uncertaintyis 1−P(True)proxy. It
is monotone in DSE here and is not treated as independent evidence.
SelfCheckGPT.Manakul et al. (2023) measure the fraction of response pairs flagged as contradictions by NLI:
SCG=1
|P|∑︁
(𝑖,𝑗)∈P
NLI(𝑟𝑖,𝑟𝑗)=contradiction
,(13)
29

Table 15:Effective denominators for the current reported runs. “Van answered” and “KG answered” count questions remaining after
generation-failure filtering. “GPS usable” counts rows remaining after GPS abstention filtering. The HotpotQA FullWiki stress snapshot is
also expandedin Table 22. GPScolumns usethe finalsoft-linked, depth-matchedestimatorreplayed onthe saved runs(Table 23reports
thecorrespondingcoveragebefore/after);RealMedQAistheGPScalibrationdomain. Retrieval-stateAUROCisonlyshownwherethe
resulting denominator leaves an interpretable ranking problem.
Dataset Van answered KG answered GPS usable GPS AUROC [95% CI]
PubMedQA 100 100 – –
RealMedQA 219 223 195 0.76 [0.59, 0.90]
HotpotQA 226 238 158 0.51 [0.42, 0.61]
HotpotQA FullWiki 208 218 185 0.54 [0.45, 0.63]
2WikiMHQA 212 244 182 0.38 [0.31, 0.47]
MuSiQue 90 94 83 0.68 [0.56, 0.80]
Table 16: GPS reliability bins.GPS abstention and fixed-bin empirical error rates on adaptive KG runs. The three right columns report
wrong/total rates within GPS-risk bins; lower GPS means stronger local graph support. This is a descriptive reliability check, not an
expected-calibration-error (ECE) calculation, because GPS is a graph-support score rather than a calibrated probability.
Dataset Answered Abstained Usable GPS<.33.33–.67GPS>.67
PubMedQA 100 100 (100.0%) 0 – – –
RealMedQA 223 28 (12.6%) 195 0.0% (0/39) 4.4% (5/114) 14.3% (6/42)
HotpotQA 238 80 (33.6%) 158 44.4% (4/9) 38.1% (16/42) 33.6% (36/107)
HotpotQA FullWiki 218 33 (15.1%) 185 30.0% (21/70) 30.8% (16/52) 36.5% (23/63)
2WikiMHQA 244 62 (25.4%) 182 33.3% (6/18) 50.0% (10/20) 27.8% (40/144)
MuSiQue 94 11 (11.7%) 83 33.3%(3/9) 50.0% (10/20) 70.4% (38/54)
wherePis the set of ordered response pairs. For these short-answer tasks, the implementation uses whole-response NLI.
SRE-UQ.Vipulanandan et al. (2026) measure perturbation sensitivity of the response embedding distribution. Let Δ𝑖
denote the first-order perturbation score of response embedding 𝝓𝑖around the weighted kernel mean embedding, using
the published bandwidth rule. This analysis uses SRE-UQ(𝒓)=1
𝑀∑︁
𝑖∈T|Δ𝑖|, whereTindexes the𝑀highest-amplitude
modes. The mode cap is 𝑀=8, so at the𝑁=5budget all responses are retained, and the Gaussian kernel bandwidth is
𝜎=std𝑖∥𝝓𝑖−¯𝝓∥2, the standard deviation of response distances to the kernel mean embedding ¯𝝓; both settings are listed
in Appendix C. The estimator is imported as written; the contribution is its KG-RAG benchmark, not a new perturbation
objective.
VN-Entropy (KG-RAG instantiation).VN-Entropy measures spectral diversity of the normalised response-embedding
Gram matrix. With𝑮=𝑽𝑽⊤and𝝆=𝑮/𝑁,
VN-Entropy(𝒓)=𝑆(𝝆)=−tr(𝝆log𝝆)=−Í𝑁
𝑖=1𝜆𝑖log𝜆𝑖,(14)
where{𝜆𝑖}𝑁
𝑖=1are the eigenvalues of 𝝆.𝑆(𝝆)=0 when all responses are identical and approaches log𝑁when responses are
mutuallyorthogonal. This isa cosine-Gram instantiation of theVNE familyintroduced by Kernel Language Entropy (Nikitin
et al., 2024); the novelty lies in the KG-RAG benchmark.
D Extended Results and Robustness Analyses
ThissectionreportstherobustnessandmechanismchecksreferencedfromSection7. SectionsD.1–D.5testwhethersilent
errors persist under alternative sampling budgets and retrieval-stability views; Sections D.6 and D.7 compare auxiliary
confidence and family-disagreement signals; Sections D.8 and D.9 check GPS hyperparameter sensitivity and audit the fully
unflagged residue; and Section D.10 holds two relocated diagnostic tables.
30

Table17:Meanpairwiseretrieval-overlapsummariesfromthesavedreportedruns. OverlapisthemeanJaccardsimilarityofretrieved
chunk-text sets across the 𝑁=5uncertainty-sampling calls for each question, averaged over the run. The current artefacts log chunk
overlapatrunlevel;theydonotyetretainper-questionentity-overlaporpath-overlapfields. Thestrictgraph-onlyrowsaremechanism
stress tests, not deployment policies.
Dataset Dense Adaptive KG Strict KG Reading
PubMedQA 1.000 0.994 – Single-abstract control; both retrievers
are nearly deterministic.
RealMedQA 0.952 0.558 0.661 Strict KG concentrates the graph con-
text relative to adaptive KG, but dense
retrieval is also highly stable.
HotpotQA 0.904 0.677 – Both systems retrieve stable chunk sets;
the KG trace adds provenance but not an
accuracy win.
HotpotQA FullWiki 0.832 0.516 – Theshared-corpusstressrunintroduces
more KG route and fallback variation.
2WikiMHQA 0.840 0.802 0.540 Adaptive KG is close to dense at chunk
level;strictgraph-onlyfailuresaremore
visibleinhop-wiseoutputcollapsethan
in aggregate chunk overlap.
MuSiQue 0.900 0.684 – Deepchainsarestableenoughforaudit,
butearlywronganchorsstilldamagean-
swer quality.
D.1 Silent-Error Threshold Sensitivity
The exact-floor silent-error definition used in Table 9 is conservative: requiring only unanimous sampled answers (DSE =0,
any SD-UQ) raises the pooled rates to 67%/52%/88%(dense / adaptive KG / strict), and admitting one dissenting sample out
of five raises them to 84%/69%/92%(Table 19), so the headline definition understates rather than inflates the blind spot.
Unanimity at 𝑁=5is already informative: if the modal answer held only 0.7probability, five identical samples would occur
with probability 0.75≈0.17. Two 20-sample probes confirm the mass is not a budget artefact in the analysed snapshots.
Onthestrictclinicalsubset,all 24empty-routewronganswersstayperfectlyunanimousat 𝑁=20(DSE=0 foreveryrow;
SD-UQ AUROC 0.17, unchanged from 𝑁=5). Ondense2WikiMultiHopQA(a purenon-graphretriever), 68%of wrong
answers remainstrictly silent and 86%stay within one dissenting sample oftwenty, against 79%strictly silent at 𝑁=5on
the headline run (the 𝑁=20probe uses an 𝑛=100subset and the 𝑁=5headline an𝑛=250subset, so the level at 𝑁=20is
therobuststatisticratherthantheprecise 79→68% delta). Quadruplingthebudgetbarelymovesthesilentrate,including
on a dense system with no graph state. These are narrow probes: they show the headline failure mode persists, not a full
sampling-budget sweep across every KG setting.
D.2 Interventional Dose-Response
The strict-clinical collapse summarised in Section 7.4 is not a chunk-level stochasticity artefact. An interventional dose-
response holds the graph, policy, prompts, and questions fixed ( 𝑛=100, seed 42) and varies only final-stage retrieval
determinism, sampling the final 𝑘=10chunks from a 4𝑘shortlist at retrieval temperature {0,0.5,1.0} . The manipulation
works: mean within-question chunk overlap falls from 0.67to0.29, a57%reduction. Yet the collapse does not move:
SD-UQAUROCis 0.18/0.18/0.16 acrossdosesandthesilent-errorcountisidentical( 24of29–30wrong,withthesame
24/24empty-retrieval rows in each arm). Chunk-level stochasticity does not dislodge the lock-in; the route decomposition in
Section 7.4 locates the silent mass in the empty-retrieval state instead.
D.3 Archived Concentration–Dispersion Traces
Directper-questionevidencecomesfromarchivedtracesthatretainedper-questionretrievaloverlap(2WikiMultiHopQA
𝑛=100, MuSiQue𝑛=66; Figure 6). AmongwrongKG answers the pooled coupling is negative (Spearman 𝜌=−0.33 ,
𝑝=0.003 ,𝑛=82): themoreconcentratedtheretrievalstate,thelowerthewrong-answerdispersion. Thecouplingiscarried
by2WikiMultiHopQA( 𝜌=−0.32 ,𝑝=0.01,𝑛=60),thebridge-styledatasetwherelock-inissharpest,whileMuSiQue’s
smallwrong-answersettrendstheotherway( 𝜌=+0.25 ,𝑛=22,notsignificant). Onthedensesidethecouplingisabsent
whereoverlapisnear-deterministicandpositiveonMuSiQue( 𝜌=+0.55 ),nothingliketheKG-sidebridgepattern. These
31

12
10
8
6
4
2
Entity-first KGlog10(SD-UQ+10−12)
ρ=−0.32 (p=0.01, n=60)2WikiMHQA
ρ=0.25 (p=0.26, n=22)MuSiQue
0.0 0.2 0.4 0.6 0.8 1.0
Retrieval overlap12
10
8
6
4
2
Denselog10(SD-UQ+10−12)
n=64 wrong
0.0 0.2 0.4 0.6 0.8 1.0
Retrieval overlap
ρ=0.55 (p<0.01, n=33)
wrong correct wrong-answer binned mean (95% CI)Figure 6: Full per-dataset, per-system view of the archived concentration–dispersion traces, with per-dataset Spearman correlations
among wrong answers and binned means with bootstrap 95%CIs. The KG-side coupling is carried by the bridge-style 2WikiMHQA trace,
while MuSiQue’s small wrong set trends positive. These details keep the pooled 𝜌=−0.33 from being read as a between-dataset artefact
or as a universal law.
archived traces support, but do not identify, the mechanism.
D.4 Within-Adaptive Stability Check
An archived 2WikiMultiHopQA 𝑛=100trace, unlike the newer 𝑛=250run, retained per-question chunk-overlap fields;
Table 26 uses it for a small within-adaptive stability check. The high-overlap band does not prove collapse monotonically, but
it is a useful warning case: accuracy falls to 0.294, DSE AUROC is near chance at 0.467, and VN-Entropy AUROC falls to
0.583. SD-UQ still carriesmoderate signal (0.633). High-stability stratathus exist inside anadaptive policy, but the logger
must persist per-question chunk, entity, and path overlap before this becomes a headline claim.
D.5 Dedicated Per-Sample Graph-State Diversity Run
Table 27 comes from a run built to close the logging gap above, where archived traces persisted chunk overlap alone. It was
runonHotpotQA-FullWiki( 𝑛=221answeredquestionswithcompleteper-sampletraces, 74wrong),reusingtheexisting
FullWiki KG with no rebuild. For every question it records the mean pairwise Jaccard overlap across the 𝑁=5samples for
four retrieval-state families (matched seed entities, traversed paths, assembled subgraph, and retrieved chunks), together with
the per-question SD-UQ used for the coupling test. It confirms two pieces of the argument, both reported in the main text: a
near-stable retrieval state across samples (the premise Equation (3)assumes) and a negative overlap–dispersion coupling
amongwronganswers. Theseed-entitycouplingthereisnullonlybecauseseedoverlapsaturatesnear 1.0,leavingalmost
32

Table 18:Selective risk at fixed coverage on the KG side: error rate among the accepted questions when the most uncertain 20%or10%
areabstained,usingasingleuncertaintysignalasthegate. “Base”istheno-abstentionerrorrate. Undertheadaptivepolicy,gatingon
SD-UQaloneisthestrongestsimplepolicy;SEUgatingismediocretherebecause 25–74%ofadaptiverowssitattheall-neutral SEU=0.5
plateau,sothe 80%thresholdfallsinsideatiemass. Underthestrict clinicalstresstesttheorderingreverses: SD-UQgatingisnobetter
than acceptingeverything, while SEUgating removes roughlyone in fiveresidual errorsat80%coverage. The2WikiMultiHopQA strict
run is the regime where no scalar gate helps, consistent with the dataset-dependent survival reported in Section 7.5.
Run𝑛Base err. SD@90% SD@80% SEU@80% Combined@80%
PubMedQA adaptive 100 0.270 0.222 0.175 0.243 0.200
RealMedQA adaptive 223 0.054 0.030 0.022 0.054 0.034
HotpotQA adaptive 238 0.395 0.393 0.384 0.405 0.395
HotpotQA FullWiki adaptive 218 0.335 0.296 0.282 0.312 0.293
2WikiMHQA adaptive 244 0.311 0.282 0.256 0.261 0.267
MuSiQue adaptive 94 0.606 0.600 0.573 0.547 0.533
RealMedQA strict 200 0.330 0.328 0.338 0.269 0.300
2WikiMHQA strict 245 0.592 0.600 0.633 0.668 0.617
Adaptive SEU cells are means over200randomised tie-breaking draws; the spread across draws is at most0.006.
Table 19:Silent-failure sensitivity ladder: pooled fraction of wrong answers classified as silent under progressively relaxed definitions of
answer-state dispersion. With 𝑁=5samples DSE is quantised, so DSE≤0.51 admits at most one dissenting sample (a 4-of-5 agreement
givesDSE=0.5004). The headline definition in Table 9 is the most conservative rung.
Silent-error definition Dense Adaptive KG Strict KG
DSE=0and SD-UQ at floor (headline) 0.59 0.42 0.84
DSE=0, any SD-UQ (unanimous) 0.67 0.52 0.88
DSE≤0.51(≤1 dissenting sample) 0.84 0.69 0.92
no variance to correlate against. The sign and magnitude of the path, subgraph, and chunk couplings match the archived
2WikiMultiHopQAtraceofSectionD.3onindependentdataandacrossthefullsetofoverlapfamilies,sothecouplingno
longer rests on chunk overlap from a single archived run.
D.6 Verbalised-Confidence Baseline
The verbalised-confidence baseline summarised in Section 7.3 uses the original P(True) protocol, which asks the model for
a probability rather than counting cluster agreement and is not blind to silent errors by construction. Querying the same
backbone once per saved answer (Table 20) yields pooled adaptive-KG AUROC 0.69: competitive on the HotpotQA variants
(0.72/0.77, above SD-UQ there), weaker than SD-UQ on RealMedQA ( 0.65vs.0.81), and near chance precisely where
bridge-style lock-in lives (2WikiMultiHopQA: 0.49adaptive, 0.58strict). On the strict clinical stress test it reaches 0.89
while sampled dispersion collapses to 0.233, so part of the clinical silent-error mass is recoverable from the generator’s own
self-estimatewithoutanyretrieval-sidesignal. Ifaone-callself-estimatecanflagclinicalsilenterrors,whydecomposeat
all? First, it fails on bridge-style lock-in ( 0.49–0.58), the regime the decomposition targets: where the wrong answer is
parametrically plausible,nooutput-side signal tested here, sampled or verbalised, ranks the errors. Second, a verbalised
probabilityisanuncalibratedself-reportwithknownself-preferencepathologies,whereasSEUandGPSaregroundedin
visible evidence and graph state. Third, the self-estimate returns a number without a reason: when it disagrees with the
answer there is no provenance to audit, which is precisely the evidence a clinical review workflow would need.
D.7 Family Disagreement and the Simple Composite Audit Score
Beyond the conjunctive audit rule of Section 7.7, Table 28 gives a deliberately simple audit score: the mean of within-dataset
percentile ranks for SD-UQ, SEU, and final GPS-risk, with GPS abstention treated as high risk. It is uneven and not claimed
to dominate the best answer-state metric. The PubMedQA row shows where the combination earns its keep: GPS abstains on
everyyes/norow(flat 0.5),yetCombinedreaches 0.813againstSD-UQ’s 0.699,becauseSEUcontributescomplementary
support-levelrankingexactlywhereanswerdispersioniscoarse,notbecausetheabstainingfamilyaddssignal. Thescore
shows how the reported families can feed a decision layer; a deployable policy would still need held-out calibration.
33

Table 20:Verbalised-confidence baseline (original P(True) protocol): the generator backbone is asked once per saved answer for the
probability that the answer is correct, and 1−𝑝is scored as uncertainty against the paper’s correctness labels. No retrieval or generation is
rerun. AUROCtreatsincorrectanswersasthepositiveclass;SD-UQcolumnsrepeatthesampled-dispersionvaluesforcomparison. The
probe is competitive on the HotpotQA variants, weaker than SD-UQ on RealMedQA, near chance on bridge-style 2WikiMultiHopQA
under both policies, and, unlike sampled dispersion, survives the strict clinical stress test.
Run Verbalised AUROC SD-UQ AUROC
PubMedQA adaptive 0.55 0.70
RealMedQA adaptive 0.65 0.81
HotpotQA adaptive 0.72 0.64
HotpotQA FullWiki adaptive 0.77 0.67
2WikiMHQA adaptive 0.49 0.69
MuSiQue adaptive 0.65 0.63
RealMedQA strict 0.89 0.23
2WikiMHQA strict 0.58 0.52
Pooled adaptive KG 0.69 –
Table21:MeanmarginalcomputetimeperquestionforeachdiagnosticontheKGside,averagedoverthesixheadlineruns(MacBookPro
M4;NLIonCPU).Timesaremarginal: theyexcludetheshared 𝑁=5answer-samplingcallsthateveryanswer-stateestimatorrequires
(the dominant latency and token cost) and the cached NLI clustering shared by the entropy-family scores. GPS varies with graph depth (its
maximum,12.3s, is theℎ=4MuSiQue configuration; the clinical graph costs0.06s).
Diagnostic Extra inputs Mean time (s)
DSE / P(True) proxy cached NLI clusters<10−4
SelfCheckGPT𝑂(𝑁2)NLI calls 1.68
SRE-UQ embeddings 0.040
VN-Entropy embeddings 0.007
SD-UQ embeddings 0.012
SEU𝐾NLI calls 1.65
GPS graph queries 2.43 (0.06–12.3)
D.8 GPS Hyperparameter Sensitivity
To test whether the headline GPS AUROCs depend on the two calibrated hyperparameters, the soft-link threshold 𝜏and the
distance decay 𝛾are swept over 𝜏∈{0.50,0.55,0.60,0.65,0.70} and𝛾∈{0.2,0.4,0.6,0.8,1.0} (25cells), recomputed by
offline replay from the saved GPS replay stores. No generation, retrieval, linking, or KG build is rerun. Table 29 reports
thecalibratedcell(whichreproducestheheadlinenumbers),theAUROCrangeacrossthegrid,itsstandarddeviation,and
theusable-rowrange. Theheld-outopen-domainAUROCsarerobusttothehyperparameters(standarddeviation ≤0.018
on HotpotQA, HotpotQA FullWiki, 2WikiMultiHopQA, and MuSiQue), so the open-domain weakness, including the
below-chance 2WikiMultiHopQA score, is a genuine retrieval limitation rather than a tuning artefact. The clinical calibration
domain is the most sensitive (0.038adaptive,0.091strict), as expected for the dataset the hyperparameters were tuned on.
D.9 Audit of the Fully Unflagged Residue
Of the 141pooled adaptive-KG silent errors, 7(5%) are unflagged by every non-answer family (Section 7.4): not on an
empty route, SEU≤0.5 , and GPSdefined and ≤0.5. Amanual audit of allseven from the saved logs (question, gold answer,
generated answer) characterises them as two benchmark-ambiguous or contested-label questions (e.g. the founding year of an
institutionwithadisputedpredecessordate;asubjective“moreknownfor”comparison),twowrong-answer-sloterrorswhere
the model returns a question anchor rather than the asked attribute (e.g. naming a song’s performer instead of their cause of
death), and three parametric conflations or specificity mismatches (e.g. returning a film character’s love interest instead of the
leadactress’srealspouse). SEUsitsatitsneutral 0.5defaultonfourofthesevenandbelowitontheotherthree,soitabstains
or mildly supports rather than flagging; by construction none exceed0.5.
Crucially,thismanualpassreadsthesavedlogs(question,goldanswer,generatedanswer,scores,routemetadata)only;
chunk text is re-retrievable from the KG, but verifying whether the evidence itselfentailsthe wrong answer needs the
cross-passage entailment and path-faithfulness replay left as future work, so none is confirmed as deep presence lock-in, and
the residue is better read as a mixture of label ambiguity, answer-slot errors, and parametric overconfidence than as evidence
34

Table22:HotpotQAFullWiki 𝑛=250shared-corpusstressrun. Vanillausesdenseretrieval;KG-RAGusesthereportedentity-firstpolicy
withfallback. Cleanaccuracyexcludesprovider-sidegenerationfailures. GPSAUROCisreportedonlyforKG-RAGbecauseGPSisa
graph-state diagnostic; the GPS null rate is the fraction of answered KG rows for which the metric is unavailable or falls back.
Policy Ans. Clean acc. Raw acc. DSE SRE-UQ VN-Ent. SD-UQ SEU GPS / null
Dense vanilla 208 0.721 0.600 0.650 0.630 0.666 0.665 0.563 –
Entity-first KG 218 0.665 0.580 0.666 0.688 0.701 0.672 0.565 0.543 / 0.151
Table 23:Post-hoc RealMedQA GPS replay on the full 𝑛=230answer logs. No generation, retrieval, or LLM call was rerun; only GPS
was recomputed against the persistent dataset KG. “Old” is the original surface-matched, binary-reachability GPS; “new” is the final
soft-linked, depth-matched estimator of Section 5.2. The linking threshold and decay ( 𝜏=0.60,𝛾=0.4) were selected on this dataset and
applied frozen everywhere else, so this table shows calibrated in-domain behaviour; the open-domain tables provide the held-out estimate.
Setting Answered GPS usable (old→new) GPS AUROC (old→new)
Dense + vanilla 219 100→192 0.72→0.89
Entity-first + KG 223 97→195 0.47→0.76
that undetectable lock-in dominates.
D.10 Relocated Diagnostic Tables
Table 30, collected here to keep the results flow tight, gives the full hop-stratified 2WikiMultiHopQA comparison across the
three retrieval policies; its decisive row, the strict graph-only 4-hop slice with zero clean accuracy and collapsed SD-UQ and
VN-Entropy,isdiscussedinSection7.4. Itusescleanansweredrowsratherthanrawhop-bucketsizes,so 𝑛differsacross
policies when provider failures or skipped systems remove rows, the same convention used for clean accuracy.
Table 31, also relocated here, reports the gold-reachability gap behind the falsifiable GPS prediction of Section 7.5: a
positive gap(correctanswersharder toreachthan wrongones) wouldtend toward below-chanceGPS, adiagnostic pattern
consistent with the regenerated HotpotQA FullWiki KG and an earlier 2WikiMultiHopQA build.
35

Table 24:RealMedQA no-rebuild strict graph-only stress test on the full 𝑛=230question set. The adaptive row is the reported KG policy
fromthefullRealMedQArun. ThestrictrowreusesthesameRealMedQAgraphanddisablesdensefallback,vectoraugmentation,and
decomposition, testing whether a more concentrated entity-first retrieval regime compresses answer-state uncertainty. All metric columns
report AUROC.
KG policy Ans. Acc. Overlap DSE VN SD-UQ SEU GPS
Adaptive + fallback 223 0.946 0.558 0.699 0.815 0.808 0.501 0.76†
Strict graph-only 200 0.670 0.661 0.463 0.248 0.233 0.721 0.746†
†GPS values come from the post-hoc replay (Table 23): adaptive 195/223usable rows, strict 142/200. RealMedQA is the calibration domain for the GPS
hyperparameters, so these cells show calibrated in-domain behaviour; Table 16 gives the corresponding binned error-rate view.
Table25:Percentile-bootstrap95%intervalsfortheheadlineanswer-stateandevidence-stateAUROCs. Thesearethemainanswer-side
metrics used in the narrative: SRE-UQ and SD-UQ as the practical answer-state baselines, and SEU as the evidence-state signal. The table
is intended to calibrate scale, not to claim significance testing for every pairwise gap.
Dataset SRE-UQ Van. SRE-UQ KG SD-UQ Van. SD-UQ KG SEU Van. SEU KG
PubMedQA 0.56 [0.49, 0.64] 0.64 [0.55, 0.73] 0.60 [0.46, 0.74] 0.70 [0.57, 0.81] 0.55 [0.44, 0.67] 0.59 [0.47, 0.71]
RealMedQA 0.43 [0.24, 0.62] 0.66 [0.51, 0.80] 0.63 [0.39, 0.86] 0.81 [0.64, 0.94] 0.63 [0.47, 0.79] 0.50 [0.35, 0.63]
HotpotQA 0.61 [0.54, 0.68] 0.63 [0.57, 0.70] 0.64 [0.56, 0.72] 0.64 [0.56, 0.70] 0.46 [0.39, 0.54] 0.48 [0.40, 0.55]
HotpotQA FullWiki 0.63 [0.55, 0.71] 0.69 [0.60, 0.76] 0.66 [0.58, 0.75] 0.67 [0.60, 0.75] 0.56 [0.47, 0.65] 0.56 [0.49, 0.64]
2WikiMHQA 0.59 [0.52, 0.67] 0.65 [0.58, 0.72] 0.61 [0.52, 0.69] 0.69 [0.61, 0.75] 0.62 [0.53, 0.70] 0.64 [0.57, 0.72]
MuSiQue 0.64 [0.53, 0.75] 0.66 [0.56, 0.77] 0.67 [0.55, 0.78] 0.63 [0.52, 0.75] 0.63 [0.52, 0.75] 0.63 [0.51, 0.74]
Table 26: Within-adaptive retrieval-stability check.Mechanism check from an archived 2WikiMultiHopQA 𝑛=100trace that retained
per-question chunk-overlap fields. The newer 𝑛=250run is the headline result; this table is a supporting diagnostic. Bands are population
tertiles of per-question chunkoverlap; because a large fraction ofquestions sit at exactly 1.0overlap, ties spill acrossthe tertile boundary.
AUROC treats incorrect answers as the positive class.
Overlap band𝑛Range Acc. DSE AUC SD-UQ AUC VN AUC DSE mean
Low 33 0.400–0.840 0.364 0.415 0.714 0.750 0.277
Medium 33 0.850–1.000 0.545 0.667 0.800 0.752 0.103
High 34 1.000–1.000 0.294 0.467 0.633 0.583 0.157
Table 27: Per-sample graph-state diversity on the dedicated HotpotQA-FullWiki run( 𝑛=221answered, 74wrong).Stabilityis
themean(median)pairwiseJaccardoverlapacrossthe 𝑁=5samplesperquestion,byretrieval-statefamily.Couplingisthecorrelation
between that per-question overlap and answer-state dispersion (SD-UQ) among wrong answers ( 𝑛=74), reported as Spearman’s 𝜌with its
two-sided𝑝-value; a negative value means more overlap goes with less dispersion, the lock-in prediction.
Stability (Jaccard) Coupling vs SD-UQ
Family Mean Median𝜌 𝑝
Seed entity0.99 1.00+0.09 0.47
Path0.58 0.58−0.30 0.009
Subgraph0.64 0.63−0.23 0.048
Chunk0.65 0.65−0.24 0.037
Table28: Adaptive-KGcompositeauditscore.Thecombinedscoreisthemeanofwithin-datasetpercentileranksforSD-UQ,SEU,
and final GPS-risk, with GPS abstention treated as high risk. AUROC treats incorrect answers as the positive class; “Base err.” is the
no-abstention error rate, and lower area under the risk-excess-coverage curve (AUREC) is better within a dataset. “Best single” is the
largest per-family AUROC in the row; bold cells mark datasets where the combined score exceeds the best single-family metric.
Dataset𝑛Base err. SD-UQ AUC SEU AUC GPS-risk AUC Best single Combined AUC Combined AUREC
PubMedQA 100 0.270 0.699 0.594 0.500 0.6990.8130.096
RealMedQA 223 0.054 0.808 0.501 0.7030.8080.729 0.022
HotpotQA 238 0.395 0.635 0.477 0.5510.6350.573 0.326
HotpotQA FullWiki 218 0.335 0.672 0.565 0.5490.6720.661 0.228
2WikiMHQA 244 0.311 0.685 0.643 0.4140.6850.619 0.225
MuSiQue 94 0.606 0.631 0.626 0.640 0.6400.7450.426
36

2WikiMHQA
12
 10
 8
 6
 4
 2
log10(SD-UQ+10−12)0.00.20.40.60.81.0SEU (evidence-state
uncertainty)
output
floorAdaptive KG (wrong answers)
Strict KG (wrong answers)
same question, both wrong (random sample of 12)Figure7: Lock-inasmigrationintothelow-dispersioncorner(2WikiMultiHopQA).EachpointisawrongKGanswer,plottedby
answer-statedispersion(SD-UQ,logscale)andevidence-stateuncertainty(SEU),undertheadaptivepolicy(greycircles)andthestrict
graph-onlystresstest(redtriangles);greyarrowsconnectarandomsampleof 12questionswrongunderboth. Understrictretrievalthe
SD-UQ mass collapses onto the floor while SEU stays spread. Small jitter is added for legibility.
12
 10
 8
 6
 4
 2
answer-state dispersion: log10(SD−UQ+10−12)0.00.20.40.60.81.0evidence-state uncertainty (SEU)
low-dispersion + contradicted
n=244, 31% wrong
low-dispersion + supported
n=139, 20% wrongFamily disagreement, pooled adaptive KGall answers (density) wrong answers
Figure 8: Family disagreement in the pooled adaptive KG runs.Grey hexagons show the density of all answered questions with
a non-neutral SEU verdict; red points overlay the wrong answers. The 𝑥-axis is answer-state residual dispersion (SD-UQ, log scale)
and the𝑦-axis is evidence-state uncertainty (SEU). The all-neutral SEU=0.5 rows, where the NLI head said nothing ( 𝑛=464,24%
wrong), are excluded. Dashed lines mark the pooled SD-UQ median and SEU=0.5 ; the two low-answer-uncertainty quadrants carry the
decision-relevant contrast: low answer uncertainty with contradicted evidence is 31%wrong against 20%for low answer uncertainty with
support, so answer-surface agreement is not a proxy for evidence support.
37

Table29:GPS AUROCsensitivity to (𝜏,𝛾)byoffline replay. “Calib.” isthepaper’sfrozen 𝜏=0.60,𝛾=0.40cell;range,SD,andusable
counts are over the5×5grid.
Run Calib. Grid range SD Usable
RealMedQA adaptive 0.76 0.60–0.76 0.038 191–197
RealMedQA strict 0.75 0.53–0.90 0.091 139–171
HotpotQA 0.51 0.50–0.55 0.014 154–169
HotpotQA FullWiki 0.54 0.53–0.56 0.008 182–187
2WikiMHQA adaptive 0.38 0.35–0.40 0.012 177–188
2WikiMHQA strict 0.48 0.40–0.53 0.035 94–104
MuSiQue 0.68 0.64–0.70 0.018 80–85
Table 30: Hop-wise 2WikiMultiHopQA diagnostics.Computed on the same fixed 𝑛=250subset. The table uses clean answered rows;
𝑛wis the number of wrong answered rows (the AUROC positive class). AUROC is omitted when a hop slice has only one correctness class.
The≥5-hop bucket contains four questions and is kept only in the released JSON artefact.
Policy Hop𝑛 𝑛 wAcc. SD mean VN mean SEU mean SD AUC VN AUC SEU AUC GPS AUC
Dense vanilla 2-hop 150 42 0.720 0.000 0.092 0.612 0.615 0.639 0.615 0.491
Dense vanilla 4-hop 58 16 0.724 0.000 0.198 0.458 0.628 0.571 0.656 0.410
Adaptive KG 2-hop 182 57 0.687 0.003 0.191 0.544 0.702 0.668 0.611 0.335
Adaptive KG 4-hop 58 16 0.724 0.002 0.248 0.466 0.600 0.638 0.646 0.452
Strict KG 2-hop 183 83 0.546 0.003 0.127 0.569 0.560 0.680 0.378 0.353
Strict KG 4-hop 58 58 0.000 0.000 0.000 0.500 – – – –
Table31: Gold-reachabilitygapandGPS’spositionrelativetochance.“Unreach.” isthefractionofrowswhosegoldanswerentity
is unreachable from the question entities, split by correctness; “Gap” is correct minus wrong. A positive gap would tend to predict
below-chance GPS. The HotpotQA-FW row regenerates from the released KG; the 2WikiMHQA row (†) does not.
Dataset𝑛Unreach. corr. Unreach. wrong Gap
2WikiMHQA†2440.71 0.45+0.26
HotpotQA-FW 1770.31 0.51−0.20
GPSAUROC: 0.38(2WikiMHQA,belowchance); 0.54(HotpotQA-FW,nearchance).†FromanearlierKGbuild,notregeneratedfromthecurrentreleased
KG.
38

Table 32: Consolidated case gallery from the reported runs.×rows are verified lock-in failures: the KG answer is wrong yet repeated
acrossallsamples( DSE=0,SD-UQatornearthefloor);thediagnosticcolumnsshowwhichfamily,ifany,stillfires. ✓rowsareKG
successeson bridgequestions, wherethesame retrievalstability ishelpfulbecause thepathreaches theright entityandrelation, plusone
correct answer despite GPS abstention. The Baltic Cup row is the hardest failure: the retrieved evidenceentailsthe wrong answer, so every
scalarfamilyreportslowriskandonlytheprovenancetraceremainsauditable. GPS =0.5denotesabstention. Thecasesareselectedfor
illustration, not sampled; prevalence claims rest on Tables 9 and 8, not on this gallery.
KG Question (dataset) Gold Anchor / path KG answer SEU GPS
×Osireion is behind the temple of
which pharaoh? (HotpotQA)Seti I Ramesses IIneighbourhood of
the Abydos templeRamesses II1.00 0.44
×Who is the paternal grandfather of
Uskhal Khan? (2Wiki)Khutughtu KhanToghon Temür(father, one
hop; bridge skipped)Toghon Temür0.60 0.00
×Who created Mickey Mouse’s
spouse? (MuSiQue)Walt DisneyUb Iwerksvia Mickey Mouse
→creatorUb Iwerks0.75 0.00
×Who is Ieuan ab Owain Glynd ˆwr’s
paternal grandfather? (2Wiki)Gruffudd Fychan IIOwain Glynd ˆwr(father, one
hop; bridge skipped)Owain Glynd ˆwr0.43 0.24
×Which country skipped the 1991
Baltic Cup? (HotpotQA)BelarusEstoniavia Baltic Cup
participantsEstonia0.00 0.00
×Are TEC-1 and Dubna 48K based
onthesameprocessor? (HotpotQA)yes weak graph state; no usable
anchor, GPS abstainsno0.75 0.50
×2010 population of the city popular
with tourists? (MuSiQue)8.005 million wrong city; only linkable
answer string four hops away2,2170.67 0.94
✓Who is the spouse of the director of
Fire-Eater? (2Wiki)Pirkko Saisio film→director→spouse
bridge preservedPirkko Saisio0.60 0.44
✓Who is the father of the director of
The Cup(1999)? (2Wiki)Thinley Norbu film→director→father
bridge preservedThinley Norbu0.69 0.35
✓Birthplace of the director ofDollar
(1938)? (2Wiki)Helsingfors film→director→birthplace
bridge preservedHelsingfors0.50 0.60
✓Dow Jones fall at the highest US
unemployment rate? (MuSiQue)54.7% chain resolved correctly; GPS
still abstains54.7%0.72 0.50
E Additional Qualitative Cases
Table32isthecanonicalcasegallery. Thefailurerowsfollowtheformwronganchor →stableretrievedcontext →stable
wronganswer→silentanswer surface,showing percase whichdiagnosticfamily stillresponds; thesuccessrowsshowthe
sameretrievalstabilityworkingasintendedon2WikiMHQAbridgequestions,wherethegraphpreservestheintermediate
entity that vanilla retrievalloses, plus one MuSiQue chain resolved correctly despite GPS abstention. The main text uses
onepath-faithfulnessexample(theOsireionlock-inofFigure1)tokeeptheexpositionfocusedonthelock-inmechanism;
thegalleryprovidestraceevidence,notadditionalleaderboardclaims. Table33givesthefullper-familyauditreadoutfor
that Osireion case (question, retrieved graph state, and the DSE/SD-UQ/GPS/SEU verdicts), expanding the summary in
Section 7.6.
39

Table 33: Path-faithfulness failure under retrieval lock-in (HotpotQA).A wrong-but-graph-coherent case from the fixed- 𝑛=250run.
TheretrieveranchorstoastronglyassociatedbutincorrectNineteenth-Dynastypharaoh,andeverysampledanswerrepeatsit. Answer-state
metrics are silent and GPS reports support; only SEU fires.
Question and gold answer Retrieved graph state Diagnostic reading
Osireion is located to the rear of the
temple named after which New Kingdom
Nineteenth Dynasty of Egypt pharaoh?
Gold answer:Seti I.
The intended chain attributes the Great
Temple of Abydos, directly in front of the
Osireion, to its builderSeti I.The retrieved subgraph anchors instead to
Ramesses II, the adjacent
Nineteenth-Dynasty pharaoh who
completed the temple and built his own
nearby:
Osireion / Abydos temple
↓associated pharaoh
Ramesses II
The generated KG answer is therefore
Ramesses II(×), and dense retrieval
makes the same error. The graph state is
locallycoherentbutattachedtothewrong
builder.All sampled KG answers repeat the same
wrong entity:
DSE=0,SD-UQ≈0.
Answer-state uncertainty is therefore at its
floor.
GPS is low (0.44), so it does not flag the
error:Ramesses IIis reachable in the
retrieved graph, and GPS is reported in risk
orientation, where lower means stronger
graph support for the generated answer.
SEU is maximal (1.0): every retrieved chunk
contradicts the generated answer, the one
family that fires.
F Prompts
Thegenerationandjudgingpromptsusedintheexperimentsarereproducedverbatimbelow;theentity-andrelation-extraction
prompts are in the code release (Appendix I).
F.1 Correctness Judge Prompt
Used to produce the binary correctness labels on which AUROC and AUREC are computed.
System message:
You are a strict answer evaluator for a factoid question answering
task. Your job is to decide if a model’s response is CORRECT.
Rules:
- Reply with exactly one word: ’correct’ or ’incorrect’.
- The response is CORRECT only if it contains an answer semantically
equivalent to the expected answer (minor spelling/accent differences
are ok).
- The response is INCORRECT if the model says it doesn’t know, cannot
determine, or provides a factually different answer.
User message (template):
Question: {question}
Expected answer: {expected_answer}
Model response: {model_response}
Is the model response correct? Reply with one word only:
correct or incorrect.
F.2 Vanilla RAG Generation Prompt
Used by the vanilla RAG system to generate all𝑁=5sampled responses per question.
System message (template):
You are an AI assistant that provides accurate, factual answers
based on the provided context.
40

Context Information:
{context}
Guidelines:
- Read all context passages carefully -- the answer is often present
but may require connecting two passages.
- For multi-hop questions: explicitly chain your reasoning step by
step (e.g. "The film starred X -> X later held position Y").
- Base your answer on the provided context; do not invent facts.
- If the answer is not directly stated but can be inferred by
connecting two pieces of evidence, make the inference explicitly
and state your reasoning chain.
- Only say the context is insufficient if you genuinely cannot find
any relevant evidence after carefully reading all passages.
- Be concise but comprehensive; include specific facts to support
your answer.
- IMPORTANT: For yes/no questions (questions starting with: Is, Are,
Does, Do, Can, Should, Was, Were, Has, Have), you MUST begin your
response with either "Yes" or "No" as the very first word,
followed by your explanation.
User Query: {question}
F.3 KG-RAG Generation Prompt
Used by the KG-RAG system to generate all 𝑁=5sampled responses per question. The instruction to prefer evidence
appearing inbothtext chunks and graph paths is a plausible contributor to consistent, confident generation under KG context,
and may amplify graph-induced overconfidence.
System message (template):
You are a knowledgeable AI assistant. Answer the question using
the provided context.
The context has three parts:
1. TEXT CHUNKS -- document passages retrieved by semantic similarity.
2. GRAPH TRAVERSAL PATHS -- multi-hop chains discovered by walking
the knowledge graph from seed entities.
Each path shows how concepts connect:
Entity A --RELATIONSHIP--> Entity B --RELATIONSHIP--> Entity C
3. ENTITIES -- individual concepts found in the graph.
HOW TO REASON:
- Read all text chunks carefully -- the answer is often present but
requires connecting two passages.
- Start from the entities most relevant to the question, then follow
graph paths to discover indirect relationships.
- For multi-hop questions: explicitly chain your reasoning step by
step (e.g. "The film starred X -> X later held position Y").
- Prefer evidence that appears in BOTH a text chunk AND a graph path
-- that is the strongest signal.
IMPORTANT:
- For yes/no questions, begin with "Yes" or "No" followed by your
explanation.
- Ground every claim in the provided context; do not invent facts.
- If the answer is not directly stated but can be inferred by
connecting two pieces of evidence in the context, make the
inference explicitly and state your reasoning chain.
- Only say the context is insufficient if you genuinely cannot find
any relevant evidence after carefully reading all passages.
41

Text Chunks:
{context}
Knowledge Graph Traversal Paths:
{graph_paths}
Entities:
{entities}
Question: {question}
F.4 Entity Extraction Prompts
The prompts used for LLM-based entity and relation extraction during KG construction (open and ontology-guided modes)
follow the JSON schema described in Appendix B.1. They are available in the supplementary code release (Appendix I).
G OntoGraphRAG System Description
OntoGraphRAG (v1.0.0) is the open-source platform used for all reported experiments. It exposes a FastAPI application on
port 8004 ( uvicorn ontographrag.api.app:app –host 0.0.0.0 –port 8004 , or the console script ontograph );
Neo4j is checked at startup, and endpoints requiring the graph database returnHTTP 503if unavailable.
G.1 Package Structure
ontographrag.kg KGconstruction. builders/ containstheopenextractionandontology-guidedextractionpipelines.
loaders/handles Neo4j serialisation, andutils/contains chunking, source-node, and graph-query helpers.
ontographrag.ragEnhancedRAGSystem (KG-RAGpipeline)and VanillaRAGSystem (dense-retrievalbaseline),with
auxiliary modules for reranking, retrieval sampling, and answer guardrails.
ontographrag.providers UnifiedLLMinterfacesupportingOpenAI,Anthropic,GoogleGemini,VertexAI,OpenRouter,
and Ollama.
Persistence: Neo4j ≥5.25forallgraph,vector,andfull-textindexes. ChunkandentityembeddingsarestoredasNeo4j
native vector indexes; no external vector store is required. Install with pip install . ; deployment requires Python ≥3.11
and Neo4j connection environment variables.
G.2 Serving Interface
ThefullRESTAPIisdocumentedinthesoftwarerelease. Therelevantinterfacehereisthequestion-answeringendpoint,
which returns the generated answer, retrieved text chunks, matched entities, graph paths, route labels, relation anchors,
and per-query diagnostic scores; the manuscript treats the API as infrastructure for trace logging, not a separate systems
contribution.
H Additional Robustness Checks
H.1 Corpus Dilution: How Curation Flatters Dense Retrieval
The accuracy comparisons in the main text run on curated benchmark corpora, where each question’s answer-bearing passage
sitsinasmallcandidatepool. Thisprobeisolateshowmuchthatcurationhelpsdenseretrieval,ontheretrievalstepalone(no
generation, no API). Using the persisted HotpotQA-FullWiki chunk corpus ( 2,489passages, one per document), we take the
143free-textquestionswhosegoldanswerstringislocatableinsomechunkandtreatthatchunkasthegoldtarget. Fora
pool of size𝑃we score the gold chunk against 𝑃−1distractor chunks sampled from the rest of the shared corpus (seed 42),
rank byquestion–chunk cosineunder the deployed MiniLM encoder,and record gold-passagerecall@ 𝑘. Growing𝑃from a
curated10to the full corpus simulates moving from benchmark to deployment retrieval.
42

Table 34: Gold-passage retrieval recall as the candidate corpus grows(HotpotQA-FullWiki, 143locatable free-text questions, MiniLM
cosine, seed 42).𝑃is the candidate-pool size (gold passage plus 𝑃−1sampled distractors). Recall collapses as curation is removed, with
questions and encoder fixed.
Pool𝑃Recall@1 Recall@5 Recall@10
100.69 0.92 1.00
250.66 0.83 0.88
500.62 0.76 0.79
1000.56 0.69 0.74
2500.49 0.65 0.68
5000.43 0.63 0.65
10000.34 0.56 0.61
24890.27 0.49 0.55
Table 35: Paired dense-vs-adaptive-KG accuracy with exact McNemar test. 𝑛is rows answered by both systems; “D-only”/“KG-only”
are discordant pairs (one system correct, the other wrong); Δis KG−dense clean accuracy on the paired set; 𝑝is the exact two-sided
McNemar value. No snapshot shows a significant accuracy difference.
Dataset𝑛D-only KG-onlyΔMcNemar𝑝
PubMedQA 100 5 3−0.020 0.73
RealMedQA 218 4 4+0.000 1.00
HotpotQA 225 14 7−0.031 0.19
HotpotQA-FW 196 18 9−0.046 0.12
2WikiMHQA 212 29 26−0.014 0.79
MuSiQue 86 14 7−0.081 0.19
TheeffectinTable34islargeandmonotone: recall@ 10fallsfrom 1.00at𝑃=10to0.55at𝑃=2489,andrecall@ 1from
0.69to0.27. Since the questions, encoder, and gold passages are held fixed, the drop estimates the cost of removing curated
candidatesetsunderaprimarystring-matchrecallcriterion(astricterall-gold-chunkcriterionwouldbounditdifferently).
Thedense–KGnear-matchreportedinSection7.1ismeasuredintheregimemostfavourabletodenseretrieval;atdeployment
scale, retrieval misses are far more frequent, and a retrieval layer that exposes what was retrieved and whether it supports the
answer matters more, not less. The probe bounds the retrieval step only; it does not re-estimate end-to-end accuracy, which
would require rerunning generation over each diluted pool.
H.2 Accuracy: Paired McNemar Test
Table 35 gives the exact paired McNemar test behind the no-detectable-gap claim in Section 7.1. For each snapshot the test is
computed on the rows answered by both systems, counting the discordant pairs (dense correct while KG wrong, and the
reverse);theexacttwo-sidedbinomial 𝑝-valueonthosediscordantpairsistheappropriatetestbecausethetwosystemsseethe
same questions. No snapshot reaches significance: the smallest 𝑝is0.12(HotpotQA FullWiki), and even MuSiQue, whose
−0.081point gap is the largest, has only 14versus 7discordant pairs ( 𝑝=0.19). The point deltas are within paired sampling
noise, not statistically resolved by this fixed-subset test, so the diagnostic comparison is made without a measured accuracy
penalty. Numbers regenerate from the saved per-question labels ( accuracy_parity_mcnemar.json ); no generation or
retrieval is rerun.
H.3 Audit-Rule Generalisation: Stratified Effect and Split-Half
Table 36 reproduces the per-dataset 2×2contingency cells behind the Mantel–Haenszel analysis cited in Section 7.7, so
thestratifiedoddsratiocanberecomputedfromthemanuscriptratherthanonlyfromthereleasedJSON.Eachrowcounts
answered KG rows as certified-or-not (the conjunctive rule) crossed with correct-or-wrong. The Mantel–Haenszel common
oddsratioacrossthesixstratais 3.39(bootstrap 95%CI[1.64,10.99] ,𝐵=5000resamplingquestionswithindataset),against
apooled(unstratified)oddsratioof 5.04;thestratifiedvalueisthehonestonebecauseitcontrolsfortheeasy-domainconfound
(the rule selects more, and more accurately, on the high-accuracy clinical domain). The single data-dependent threshold (the
per-dataset SD-UQ median) survives held-out evaluation: recomputing it on a random half of each dataset and applying the
ruletotheotherhalfover 𝐵=200splitsgivesmeanprecision 0.918(5th–95thpercentile [0.867,0.974] )atmeancoverage
0.078, essentially unchanged from the in-sample0.919at0.077, so the headline precision is not threshold overfitting.
43

Table36: Per-dataset 2×2cellsfortheconjunctiveauditrule(adaptiveKGruns),feedingtheMantel–HaenszeloddsratioofSection7.7.
“Cert.” is certified (selected as low-risk by the rule); “Corr.”/“Wr.” are correct/wrong answered rows. PubMedQA selects nothing (binary
answers expose no linkable answer entity), so it contributes no stratum.
Cert. Corr. Cert. Wr. Uncert. Corr. Uncert. Wr.
PubMedQA 0 0 73 27
RealMedQA 48 0 163 12
HotpotQA 5 1 139 93
HotpotQA-FW 21 5 124 68
2WikiMHQA 4 0 164 76
MuSiQue 1 1 36 56
Pooled 79 7 699 332
Table37: Dense-sideselective-predictionfrontier.AUREC(areaunderrisk-excess-coverage;lowerisbetter)forthefourgates,withbase
errorandthelearnedgate’sout-of-foldAUROC;thelowestAURECineachrowisbold. “Best”isthebestsinglesignalperdataset;the
winningsignalis,inroworder: SD-UQ,DSE,SD-UQ,VN-Entropy,SEU,VN-Entropy. Thelogisticgatecollapsesbelowchance(AUROC
<0.5) on the two low-error biomedical sets and never attains the lowest AUREC on any dataset.
Base AUREC (lower better) Log.
Dataset err. Best Log. Conj. Mean AUROC
PubMedQA .250.057.185 .078 .193 .453
RealMedQA .050 .006 .076.003.018 .312
HotpotQA .345.085.107 .113 .118 .615
HotpotQA-FW .279.047.097 .073 .071 .626
2WikiMHQA .288 .084 .093.048.100 .628
MuSiQue .522 .194 .159.084.128 .684
H.4 Dense-Side Selective-Prediction Frontier
Thedense-sidefrontiercomparisonreferencedinSection7.7reportstheAURECforfourgatingstrategies,computedasa
pure post-hoc replay from the saved dense-run uncertainty scalars (no retrieval, generation, or API calls). The four gates are:
best_single(the single signal with the highest dense-side AUROC on that dataset),logistic(a leave-one-fold-out logistic gate
fitonallsevendenseuncertaintyscalars,standardisedandsanitisedagainstnon-finitevalues),conjunctive(thedense-analogue
of the KG audit rule: SD-UQ ≤dataset medianandSEU ≤0.5; GPS is KG-only so it is dropped), andmean_percentile(the
paper’s composite score applied to the dense run). AUREC is the area under the risk-excess-coverage curve (lower is better);
the logistic gate’s AUROC is reported to expose where it overfits the low-error biomedical sets.
The logistic gate’s below-chance AUROC on RealMedQA ( 0.31) and PubMedQA ( 0.45) is not a numerical artefact:
features are standardised and non-finite values are median-imputed before fitting. It reflects genuine overfitting in the
low-positive-fractionregime ( 5%and25%error), wherea five-fold logisticmodel withsevenfeatures hastoo few positives
per fold to generalise and inverts the ranking on held-out data. On the larger, higher-error multi-hop sets (HotpotQA,
2WikiMHQA, MuSiQue) the logistic gate recovers to 0.61–0.68AUROC but still does not attain the lowest AUREC on any
dataset (Table 37); the conjunctive rule and the best single signal split the six datasets between them.
H.5 SEU Domain-NLI Ablation
One remaining question is whether SEU’s neutral plateau ( 74%of RealMedQA rows at the all-neutral SEU=0.5 default) is
anartefactofthegeneral-domainNLImodel( microsoft/deberta-large-mnli )ratherthanapropertyoftheretrieved
evidence. We test this by recomputing SEU on PubMedQA under two NLI conditions (Table 38): (A) the paper’s
deberta-large-mnli ,and(C)abiomedical-awareLLM-NLIjudge( gpt-4o-mini ,promptedasadomainexpert;zero-
shot,temperature 0). BecausetherunlogskeepchunkidentifiersratherthantheSEUper-chunkinputs,retrievalisre-run
against the existing KG (no KG rebuild) to recover the per-question chunks; the saved answer is used as the SEU hypothesis,
matching the production path. A reproduction check confirms that the re-retrieved chunks reproduce the paper’s SEU:
recomputed deberta-large AUROC matches the saved value to within 0.001on both policies, so there is no retrieval drift.
The LLM judge improves AUROC by +0.030(KG) and+0.090(dense), so part of the neutral plateau is indeed a
general-domain-NLI artefact: the LLM judge resolves some biomedical paraphrase that deberta-large labels neutral.
Theimprovementismodest,however,andtheneutralrateparadoxicallyrisesundertheLLMjudge( 0.73→0.89 onKG),
becausegpt-4o-mini ismoreconservativeaboutcommittingtoentailmentonspecialistpassagesthantheMNLI-fine-tuned
model. Theplateaureflectstwoeffects: (i)ageneral-domain-NLIlabellinggapthatadomain-awarejudgepartiallycloses,
44

Table38: SEUdomain-NLIablationonPubMedQA( 𝑛=100). Condition(A)reproducesthepaper;(C)replacestheNLIheadwitha
biomedical-awareLLMjudge. “Neutralrate”isthefractionofrowswherenochunkentailsorcontradicts the answer; “Plateau@0.5”isthe
fraction sitting at the all-neutral SEU=0.5 default. The LLM judge improves AUROC on both policies, confirming that part of the neutral
plateau is a general-domain-NLI artefact, but the neutral rate does not fall, so the plateau is not entirely an artefact.
Policy NLI condition AUROC Mean SEU Neutral rate Plateau@0.5
KG(A) deberta-large 0.595 0.619 0.732 0.59
(C) gpt-4o-mini 0.625 0.460 0.887 0.71
Dense(A) deberta-large 0.551 0.591 0.788 0.62
(C) gpt-4o-mini 0.641 0.467 0.902 0.72
and (ii) genuinely non-committal retrieved evidence that neither entails nor contradicts the answer. RealMedQA was not
included because its KG was not loaded for the same re-retrieval here (PubMedQA’s was), so the result is PubMedQA-only;
the direction is consistent with the RealMedQA plateau being partly artefactual, but the magnitude on the clinical target
domain remains to be confirmed.
H.6 Independent Re-Judge on the Certified Subset
Theclinicalheadline(theconjunctiverule’s 48/48selectiononadaptiveRealMedQA)isthecellmostexposedtosame-model
judging, and RealMedQA also carries the lowest inter-judge 𝜅(0.52) because near-ceiling accuracy leaves few wrong
answers for 𝜅to score (a marginal-imbalance effect, not broad disagreement: raw agreement is 92.8%). We re-judged
the48certified answers with the independent Llama-3.3-70B judge:all 48are confirmed correct, with zero label flips
(certificate_rejudge.json ). The certified cell is robust to the judge family even though dataset-wide 𝜅is low, because
theruleselectshigh-agreement,near-ceiling-confidenceclinicalanswersratherthanthecontestedwrong-answerrowsthat
depress𝜅. More broadly, re-ranking every free-text run under the independent labels leaves each central contrast intact:
onadaptiveRealMedQASD-UQAUROCis 0.86undertheindependentjudge(versus 0.81originally),whileonthestrict
clinicalrunSD-UQstayscollapsed( 0.31)andSEUandGPScontinuetoranktheerrors( 0.71and0.67);thelock-insignature
isnotanartefactofsame-modelself-preference,whichwouldhaveinflatedanswer-stateagreement,nottheevidence-and
retrieval-state signals.
I Code and Data Availability
The system implementation, experiment harness, and scripts that generate every table and figure are available at https:
//github.com/julka01/OntoGraphRAG , with the run artefacts in the same repository. Each run log is a per-query JSON
record storing the verbatim model responses for all 𝑁=5uncertainty samples (and the single accuracy-generation response),
retrieved chunk identifiers and graph paths, route labels, both judges’ correctness labels, the raw per-query metric scores
and uncertainty estimates, and the fixed sampled question IDs for each snapshot. The GPS replay stores for the post-hoc
recomputations are also released (the RealMedQA replay, the 𝜏×𝛾sensitivity sweep, and the HotpotQA-FullWiki diversity
run). Thereleasebundleis reproducibility/arxiv_v1/ ,whoseMANIFEST.json recordstheexactrepositorycommit
andfile hashesforthecached logs. Everyfixed-subsetnumberand post-hocreplayiscomputed fromtheseartefacts, sothe
results reproduce without rerunning generation or retrieval; the verification bundle replays cached JSONL logs (metrics,
confidenceintervals,audit-rulecells)butexcludesthemainexperimentrunner,soend-to-endregenerationoftheper-query
logs requires cloning the full repository and incurring API costs. None of these tiers stores the full retrieved chunktext:
therunlogskeepchunkidentifiers,theverificationbundleholdscachedmetricinputs,andthechunktextitselflivesinthe
persistentdatasetKG.Thefewanalysesthatneedchunktext(theSEUdomain-NLIablation,AppendixH.5,andconfirmation
of deep presence lock-in) re-retrieve it from the KG.
Reproducibility checklist.Table 39 consolidates the exact configuration values scattered through this appendix.
Mechanism-probe artefacts.The additional mechanism analyses reported in Section 7.4 have their own artefacts, released
alongsidetherunlogs: theretrieval-temperaturedose-responseruns(RealMedQA,strictprofile, 𝑛=100,seed42,retrievaltem-
peratures{0,0.5,1.0} ,withper-questionchunkoverlapretained),the 𝑁=20resamplingprobeonthesamestrictsubset,thein-
dependentre-judgeofallfree-textruns( rejudge_independent.json ,rejudge_wave1.json ),theverbalised-confidence
baseline(verbalized_confidence.json ),theaudit-ruleevaluationwithmatched-coveragebaselinesandsplit-halfvalida-
tion(certificate_eval.json ),thesilent-ratesensitivityandroute-decompositionnumbers( wave1_analysis.json ),
45

Table 39:Reproducibility checklist (verified against the released run artefacts and code).
Item Value
Generation modelgpt-4o-mini via OpenRouter ( openai/gpt-4o-mini-2024-07-18 ;
OpenAI snapshot 2024-07-18, accessed via OpenRouter; all runs use
this fixed snapshot); 𝑇=1.0for the𝑁=5uncertainty samples, 𝑇=0.0for
accuracy generation and KG extraction
Judge model (main) samegpt-4o-minibackbone as generation
Judge model (re-judge)meta-llama/llama-3.3-70b-instructvia OpenRouter,𝑇=0.0
Embedding modelall-MiniLM-L6-v2 (384-d,SentenceTransformers),sharedbychunks,
entities, and response embeddings
NLI modelsmicrosoft/deberta-large-mnli (DSE / P(True) clustering, SEU);
roberta-large-mnli(SelfCheckGPT pairwise)
Subset seeds42for every dataset and run (including dose-response and𝑁=20probes)
Sampling budget𝑁=5headline;𝑁=20resampling probe
GPS calibration and sensitivity𝜏=0.60 ,𝛾=0.40selectedontheRealMedQAreplayandappliedfrozenelse-
where;AppendixD.8replays 𝜏∈{0.50,...,0.70} and𝛾∈{0.2,...,1.0}
Cached reproduction headline tables and figures regenerate from released saved artefacts; no
API calls, KG rebuild, or generation rerun is required for the reported
analyses
Retrieval temperature0.0 inall reportedruns; {0,0.5,1.0} inthe dose-responsearms (shortlist
factor4)
Provider failures per-rungenerationfailuresexcludedfromcleanaccuracy: KGside 0–32
and dense side0–42per run (Table 14)
Code versionOntoGraphRAG v1.0.0,publicrepository;exactcommitandartefacthashes
recorded in the released manifest
the reporting-robustness bookkeeping for failure counts and the audit rule odds ratio ( reporting_robustness.json ), and
the KG scale statistics queried from the persistent Neo4j stores (kg_scale_stats.json).
46
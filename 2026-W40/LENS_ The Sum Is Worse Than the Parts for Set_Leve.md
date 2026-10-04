# LENS: The Sum Is Worse Than the Parts for Set-Level Poisoning in Retrieval-Augmented Generation

**Authors**: Kaisheng Fan, Yishu Gao, Xunzhu Tang, Tegawend'e F. Bissyand'e, Weizhe Zhang

**Published**: 2026-09-28 13:33:32

**PDF URL**: [https://arxiv.org/pdf/2609.35155v1](https://arxiv.org/pdf/2609.35155v1)

## Abstract
Retrieval-augmented generation (RAG) aggregates evidence from multiple external documents, yet this joint integration creates an underexamined vulnerability: attack effects absent in individual documents can emerge through set-level composition. Existing coordinated attacks do not explicitly enforce that every proper subset remains insufficient in frozen single-round RAG. We formalize set-level compositional poisoning, where documents designed to remain individually plausible jointly redirect RAG outputs to a target answer, while proper subsets fail to induce the target on their own. To construct such attacks, we propose LENS, a generator-black-box multi-agent framework that casts construction as constrained evidence composition. LENS factorizes target inference into a query-conditioned interpretation lens and complementary facts, then uses a nested dual-loop workflow to concentrate steering in the full set while suppressing subset leakage. The outer loop plans the interpretation lens and semantic roles; the inner loop synthesizes documents and applies counterexample-guided repair. Across four benchmarks and three generators, returned packets achieve 0.852 full-set ASR and 0.784 post-retrieval ASR@5, while their strongest proper subsets reach only 0.069. Against construction baselines evaluated on the same frozen manifest, LENS improves all-attempt E2E-Strict@5 from 0.244 to 0.363, a 48.8% relative gain. A blinded human audit finds that 68.3% of returned packets combine an incorrect target, a definite answer-criterion shift, and no target entailment under the original semantics. Across four published defenses, LENS attains the highest defended all-attempt ASR@5, exceeding the strongest baseline by 0.141 on average. Together, these results establish evidence composition as a distinct RAG security boundary and position LENS as a stress test for defenses that reason over document sets.

## Full Text


<!-- PDF content starts -->

LENS: The Sum Is Worse Than the Parts for Set-Level Poisoning
in Retrieval-Augmented Generation
Kaisheng Fan1, Yishu Gao1, Xunzhu Tang2,
Tegawend’e F. Bissyand’e2, Weizhe Zhang1,3∗
1School of Cyber Science and Technology, Harbin Institute of Technology, Harbin, China
2SnT, University of Luxembourg, Luxembourg City, Luxembourg
3Department of New Networks, Peng Cheng Laboratory, Shenzhen, China
{fankaisheng, gaoyishu}@stu.hit.edu.cn, wzzhang@hit.edu.cn
{xunzhu.tang, tegawende.bissyande}@uni.lu
Abstract
Retrieval-augmentedgeneration(RAG)aggregatesevidence
from multiple external documents, yet this joint integration
createsanunderexaminedvulnerability:attackeffectsabsent
in individual documents can emerge through set-level compo-
sition.Existingcoordinatedattacksdonotexplicitlyenforce
thateveryproper subsetremains insufficientin frozensingle-
roundRAG.Weformalizeset-levelcompositionalpoisoning,
where documents designed to remain individually plausible
jointlyredirectRAGoutputstoatargetanswer,whileproper
subsetsfailtoinducethetargetontheirown.Toconstructsuch
attacks, we propose LENS, a generator-black-box multi-agent
framework that casts construction as constrained evidence
composition. LENS factorizes target inference into a query-
conditioned interpretation lens and complementary facts, then
usesanesteddual-loopworkflowtoconcentratesteeringinthe
fullsetwhilesuppressingsubsetleakage.Theouterloopplans
the interpretation lens and semantic roles; the inner loop syn-
thesizes documents and applies counterexample-guided repair.
Acrossfourbenchmarksandthreegenerators,returnedpackets
achieve0.852full-setASRand0.784post-retrievalASR@5,
while their strongest proper subsets reach only 0.069. Against
construction baselines evaluated on the same frozen mani-
fest, LENS improves all-attempt E2E-Strict@5 from 0.244
to0.363,a48.8%relativegain.Ablindedhumanauditfinds
that 68.3% of returned packets combine an incorrect target, a
definite answer-criterion shift, and no target entailment under
the original semantics. Across four published defenses, LENS
attains the highest defended all-attempt ASR@5, exceeding
the strongest baseline by 0.141 on average. Together, these
results establish evidence composition as a distinct RAG secu-
rityboundaryandpositionLENSasaconcretestresstestfor
defenses that reason over document sets.
Introduction
Retrieval-augmentedgeneration(RAG)improveslargelan-
guage models (LLMs) by conditioning generation on a small
top-Ksetofpassagesretrievedfromexternalcorpora(Lewis
et al. 2020; Ram et al. 2023; Asai et al. 2024). This external-
memoryinterfaceletsLLMsusefresh,domain-specific,or
user-providedinformationwithoutchangingmodelparame-
ters, but it also turns corpus content into a security-critical
input.Whendocumentsareuser-uploaded,weaklycurated,
∗Corresponding author.or drawn from open sources, poisoned documents that enter
the retrievablecorpus may beselected atinference time and
shape the final answer.
Many RAG poisoning attacks expose locally sufficient
orindividuallydetectablecuesbydirectlystatingthetarget
answer,addinganswer-shapedscaffolding(Zouetal.2025),
optimizing adversarial strings (Ben-Tov and Sharif 2025;
Wangetal.2026b),orusingtrigger-stylepoisoneddocuments
(Chaudhari et al. 2026). Several defenses screen passages
throughdocument-localabnormality,conflict,answersupport,
or isolation tests (Xiang et al. 2026; Shen et al. 2025; Si et al.
2025). This leaves a different failure mode less explored:
individuallyplausibledocumentscanremaininsufficientin
isolation while their composition redirectsanswer selection.
Becausegenerationconditionsonthecomposedretrievedset,
passage-wise inspection evaluates a different unit from the
one that produces the answer.
Weshowthatthismismatchenablesset-levelcompositional
poisoning: individually plausible documents occupy comple-
mentary semantic roles, so their composition installs an
attacker-selectedcriterionforanswerselectionwhileproper
subsets remain insufficient. LENS separates this criterion
fromthe factsthat support it,placing controlinthe induced
evidence-set decision rule instead of a passage-local payload.
TheattackoperatesatinferencetimeagainstafrozenRAG
pipeline,makingevidencecompositionthesecurityboundary.
Figure1contraststhissettingwithsingle-documentpoisoning,
where one passage is locally sufficient.
Toconstructsuchattacks,weintroduceLENS(LocalEvi-
dence,Non-localSteering),aquery-aware,generator-black-
box multi-agent pipeline with nested planning and repair
loops. The outer loop mines a typed, query-conditioned inter-
pretationlens.Thislensspecifieshowevidenceshouldberead
according to role, scope, time, category, naming convention,
orreferent.Itthenfixesadocumentplanthatassignsthelens,
true or locally supported complementary facts, and claims to
avoidforsubsetsafety.Theinnerloopsynthesizeslens-setting
and fact-completion documents, evaluates full-set success
andsubsetleakagewithanattacker-controlledlocalsurrogate
reader, and uses typed counterexamples to repair candidates
toward strong full-set steering with low subset leakage.
This compositional attack surface creates a new defense
objective: identify suspicious cross-document dependence
1
arXiv:2609.35155v1  [cs.CR]  28 Sep 2026

Single -document Poisoning
(risk inside onedocument)
User
Query
Retriever
Retrieval Corpus
Clean Doc
Poisoned DocLocal Sufficient
LLM
Generator
Wrong
Answer
User
Query
Retriever
Retrieval Corpus
Clean Doc
LENS insertJointly Sufficient
(Risk Emerges)
LLM
Generator
Wrong
Answer
LENS insert
Set-level Compositional Poisoning
(risk emerges from adocument set)Figure1:Single-documentversusset-levelcompositionalpoisoning.Left:onepoisoneddocumentislocallysufficienttosteer
generation. Right: plausible LENS documents rarely induce the target in proper subsets, but the full set shifts generation toward
the preselected non-gold target.
whilepreserving theevidence integrationthatenables legiti-
matemulti-hopreasoning(Yangetal.2018;Hoetal.2020;
Trivedi et al. 2022). We therefore evaluate attack suppression
andcleanmulti-hoputilitytogether,usingtheirtrade-offto
characterize composition-aware defenses.
Thispapermakesthreecontributions.First,weformalize
set-level compositional poisoning with an explicit all-proper-
subsetconstraint:thefullinsertedsetinducesapreselected
non-goldtarget,whileeverypropersubsetisconstrainedto
maintain a low target rate. Second, we introduce LENS, a
generator-black-box construction method that mines interpre-
tation lenses, synthesizes lens-setting and fact-completion
documents,andusescounterexample-guidedrepairwithac-
tive subset checking and exhaustive final verification. Third,
constructionbaselinesunderasharedmanifestandevaluation,
semanticcontrols,cross-generatorevaluation,andall-attempt
defense experiments establish the distinctive value of full-set
dependence. LENS improves end-to-end strict set success by
48.8%overthestrongestconstructionbaselineandretainsthe
highestattacksuccessundereverytesteddefense.Effective
interventionsmustdistinguishsuspiciousdependencefrom
legitimate multi-hop composition.
Problem Setup and Formalization
Attack model.Let C={D 1, . . . , D n}be a corpus and
X=R K(q;C)itscleantop- Kcontextforquery q.Theadver-
saryinsertsapacket D={d 1, . . . , d k},yielding C′=C ∪D,
but cannot modify the query, original documents, retrieval
pipeline, or target generators. Construction uses only an
attacker-controlled local surrogate, and target generators pro-
vide no attacked-query feedback before packet freezing. We
study index-admitted corpus poisoning in frozen, benchmark-
scale RAG.
Letgbe the benchmark gold and ta type-compatible
target fixed before construction and alias-distinct from g.
Benchmark-intended query semantics are fixed before in-
sertion; frozen filters admit only candidates classified as
incorrectunderthosesemantics,andablindedhumanauditindependently measures residual ambiguity or compatibility.
We studytargeted benchmark-answer displacement, where
corpusinsertionredirectsacleansystemthatoutputs gtoward
t. This defines an attacker-directed output-integrity violation:
an untrusted contributor preselectstand attempts to replace
the clean-system answer with it.
ForS⊆ D, define
TR(x)(S) = Pr[G x(q,X ∪S) =t],(1)
where x= surdenotes the construction surrogate and x=
ma target generator. We write dTR(x)
20(S)for its observed
frequency over 20 generations. Construction uses dTR(sur)
20,
while final evaluation uses dTR(m)
20.
Subset dependence.A packet is subset-safe for target gener-
atormat toleranceϵwhen
∀S⊊D: TR(m)(S)≤ϵ.(2)
This is a population property. We report the corresponding
empiricalall-subsetpassrateunderthe20-generationprotocol,
not a probability certificate.
We quantify full-set dependence by
CJL(m)(D) = TR(m)(D)−max
S⊊DTR(m)(S).(3)
Reported values replace each probability with its observed
frequency. CJL measures a full-set dependence gap, while
token- and slot-matched controls test alternative context-size
explanations.
Retrieval and validity.End-to-end evaluation reports
JRR@K, the fraction of packets retrieved in full, and Ret.
ASR@K,thetarget-outputrateafterretrievalfrom C′.Doc-
ument validity is evaluated separately from attack success
and subset safety. A verifier filters candidates judged fabri-
cated,speculative,answer-shaped,orconflictingandfavors
documents that remain plausible and consistent with the
surrounding evidence.
2

(b) Multi-Agent Construction: Lens Search and Document Synthesis-RepairOuter Loop: Lens Search & PlanningLenses Candidate Queue!!...!"Select !! Search LensQuestionWhat nationality is athlete AHow to read  'nationality'
"#$%: nationality = sporting/offical&!: nationality = country of birthsame question, different criterion -> different answer
LLM Planner Agents...Evidence Analyzer...Len Enumerator...Len Scorer...Planner Composer
Inner Loop: Synthesis & Repair
LLM Proposer Agents...Drafter...Humanizer...Reviser
LLM Verifier Agents...SuccessChecker...SubsetChecker...AlignmentChecker...ValidityCheckerTR(%)≥()*+'⊂)TR(,)≤.Align(ℓ,6,%)=1Valid(<,%)=1
Selected Policy *	(ℓ,%,&"#$,'%&'())Nationality may denote birth country, not the country represented in competition.[A]was born in Country [Y].[A]represented Country [X]in international competition....ReflectionCounterSampleLens-setting DocSupported Doc(a) Setup & Threat
Sporting NationalityX(benchmark gold)Birth nationality Y(preselected target)
target ) = Country YBlack-box RAG
LLMsQuestionWhat nationality is athlete A
clean passageAthlete [A]officially competes for [Country X]!#(c) Set-LevelCompositional FailureRetrieval Top-K Results...JRR@K: full set co-retrieved "!"""#Subset-Safety no proper subset induces =TR(,)≤>for	all	,⊆%{E+}
{E+,E,}
{E+,...,E,}|⋅|<0Full Set Successonly the full set induces =TR(%)≥(
{E!,...,E1}Figure 2: Overview of LENS. The Planner defines the document plan, the Proposer writes lens-setting and fact-completion
documents,andtheVerifierreturnstypedcounterexamplesfromfull-set,subset,alignment,andvaliditychecks.Construction
uses a local surrogate, while attacked target-side runs occur only after packet freezing. Exhausted plans return to the Planner.
Method: LENS
Overview: Constraint-Maintaining Construction
LENS treats constraint violations as counterexamples for
local document repair or, when a plan is exhausted, for
outer-loop replanning. Figure 2 summarizes three functional
roles. ThePlannerselects an interpretation lens and assigns
complementary document roles, required facts, and claims
toavoid.TheProposerrealizesthisplanaslens-settingand
fact-completion documents. TheVerifierevaluates full-set
success,subsetleakage,interfacealignment,anddocument
validity using the local surrogate, then returns a typed repair
signal.
Interpretation-Lens Planning
Aninterpretation lensis a typed, query-conditioned reading
criterion,suchasrole,time,category,naming,orscope,rather
than the inserted conclusion. For example, a role lens may
map a creator query to a local creator-entry rule; separate
documents supply the role and date facts needed under it.
Usingthecleanevidencepath,answerdimension,andlocally
supportabletarget-relevantfacts,thePlannerrankslensand
role decompositions by semantic fit, closure, validity, and
leakage risk. This structured search rejects plans that require
one document to state a decisive target relation. The selected
plan is
π= (ℓ, ρ, F req, Cavoid, Rrisk),(4)
where ℓisthelens, ρassignsdocumentroles, Freqlistsclosure
facts, Cavoidlistsleakage-proneclaims,and Rrisklistsproper
subsets of planned document slots predicted to have high
leakage risk. Inner-loop repair preserves ℓandρ; changing
either starts a new outer-loop plan.
The key planning decision is to separate interpretation
from factual completion. Lens-setting documents make anon-defaultreadingcriterionavailable,whilefact-completion
documents provide the missing support needed under that
criterion. At the role level, the Planner seeks D=D lens∪
Dfactsuch that
TR(sur)(Dlens)≤ϵ,TR(sur)(Dfact)≤ϵ,TR(sur)(D)≥τ.
(5)
This is a population-level planning objective. Construction
instead uses 20-generation empiricalrates over an active set
Aof proper subsets, initialized with all singletons and the
subsets specified by Rrisk, then expanded when new leakage
is found.
Document Synthesis and Diagnostic Verification
The Proposer writes lens-setting documents that make the
selected criterion available without instantiating the target
relation. Fact-completion documents supply the remaining
supportwithoutrestatingthelens,comparingcandidatean-
swers,orstatingthetarget.Bothrolesmustresembleordinary
reference prose and avoid answer-shaped scaffolding.
The Verifier distinguishes two common failures.Subset
leakageoccurs when a proper subset makes the target lo-
callydecisive.Interfacemismatchoccurswhenthelensand
complementary facts operate over different answer dimen-
sions. Accordingly, Align(ℓ, π,D) checks whether the query,
cleansupport,lens,andtarget-relevantfactssharethesame
selection criterion. Diagnostic contexts include clean-only,
lens-only,fact-only,clean-plus-single-role,andfull-setinputs,
allowing the Verifier to separate genuine complementarity
from a locally sufficient insert or target behavior already
present in the clean reader. For a candidate packet D, plan π,
3

Diagnostic view Test / comparison Failure signal Repair objective
Full-set closure dTR(sur)
20(D)≥τFull-set rate is too low Supply the missing relation
Tracked subset safetymax S∈AdTR(sur)
20(S)≤ϵA tracked subset leaks Remove the decisive cue
Interface alignmentAlign(ℓ, π,D)Lens and facts use different criteria Align the document roles
Document validityValid(X,D)An insert is unsupported or conflicting Revoice, qualify, or remove
Tracked-set update FindS v⊊DwithdTR(sur)
20(Sv)> ϵA new subset leaks AddS vtoA
Table 1: Verifier diagnostic matrix. Each failed check returns a typed counterexample and a repair target. Newly discovered
leaking subsets are added to the tracked constraints.
lensℓ, and active subset set A, the current diagnostic gate is
Pass(D;ℓ, π,A)⇐⇒

dTR(sur)
20(D)≥τ,
max S∈AdTR(sur)
20(S)≤ϵ,
Align(ℓ, π,D) = 1,
Valid(X,D) = 1.
(6)
Table 1 summarizes the resulting counterexamples and repair
targets.
Counterexample-Guided Repair
For a fixed lens, plan, and active subset set, the inner loop
uses the tracked surrogate dependence gap
[CJL(sur)
A,20(D) =dTR(sur)
20(D)−max
S∈AdTR(sur)
20(S).(7)
This score prioritizes repairs, while Eq. 6 remains the accep-
tance criterion. When Acontains every proper subset, the
scoreequalsthe20-generationempiricalsurrogate-sidede-
pendencegap.Duringconstruction,itcoversonlythesubsets
currently tracked by the Verifier.
Each counterexample identifies both the violated condi-
tion and the responsible document role. The Proposer then
rewritesonlythecorrespondingdocumentwhilepreserving
the clean context, lens, and role assignment. Low full-set clo-
suretriggerstheadditionorclarificationofmissingsupport
rather than direct insertion of the conclusion. Subset leakage
triggers removalor qualificationof thedecisivecue. Align-
mentfailuresrevisehowadocumentrealizesitsassignedrole,
andvalidityfailuresrevoiceorremoveunsupportedcontent.
This localized repair preserves the connection between each
failure signal and the change intended to correct it.
WhenevertheVerifierdiscoversaviolating Sv⊊D,itadds
SvtoAforsubsequentrepairs.Thesetrackedchecksguide
construction, but they do not define the final reported subset
result. Before a packet is frozen, the Verifier exhaustively
evaluateseverypropersubsetunderthesame20-generation
surrogateprotocolandreappliesthefull-set,alignment,and
validity checks. Target-side evaluation then independently
repeatstheexhaustivesubsetanalysisforeachtargetgenerator
Gm. If repair repeatedly alternates between subset leakage
andinsufficientfull-setclosure,LENSexhauststhecurrent
planandreturnstothePlannerratherthanassumingthatlocal
improvements will compose.Experiments
Experimental Setup
Benchmarks,models,andpipeline.WeevaluateHotpotQA,
2WikiMultihopQA, MuSiQue, and Natural Questions (Yang
et al. 2018; Ho et al. 2020; Trivedi et al. 2022; Kwiatkowski
etal.2019)onafixedmanifestof1,000query–targetattempts
per benchmark. Each retained query is answered correctly
by all three clean target RAGs; targets are type-compatible
and fixed before construction, and failures are never replaced.
LENS uses Qwen3.6-27B for construction and surrogate
reading, while Qwen3.6-35B-A3B, Llama-3.1-70B-Instruct,
andDeepSeek-V4-Proserveastargetgenerators(QwenTeam
2026; Grattafiori et al. 2024; DeepSeek-AI 2026). LENS and
the LLM-based baselines follow the assigned k∈ {2,3}
budget; Semantic Chameleon always uses its native two-
documentsleeper–triggerpair.ALENSattemptsucceedsonly
if its packet passes the exhaustive surrogate gate at τ= 0.50
andϵ= 0.10 . Frozen packets are evaluated on all three
target generators without target-side selection. Conditional
evaluation supplies X ∪S; end-to-end evaluation indexes
returnedpacketsusingBM25/BGE-M3reciprocal-rankfusion
and a Qwen3 reranker (Robertson and Zaragoza 2009; Chen
et al. 2024; Cormack, Clarke, and Buettcher 2009; Zhang
et al. 2025b).
Metrics and evaluation.Each context uses 20 stochastic
generations. Packet metrics use all returned packet–generator
pairs,withJRRcomputedonceperpacket;all-attemptmetrics
retainthefixedmanifestandscoreconstructionfailuresaszero.
Full ASR is the conditionalfull-packettarget rate,Max-sub.
TRisthelargestproper-subsetrate,andCJListheirdifference.
SubsetPass@20acceptsapaironlywheneverypropersubset
producesthetargetatmosttwicein20generations.Ret.ASR
measurespost-retrievaltargetadoption.E2E-Strict@5further
requires complete packet retrieval, target rate at least 0.50
in the actual top-5 context, and target rate at most 0.10 for
every controlled proper subset. Cond-Strict@5 applies the
full-set threshold in the conditional interface while retaining
the same controlled subset test. Only unhedged matches to
torgcount. Clean Acc. uses a separate held-out utility set.
Mechanism and mitigation analyses condition on returned
artifacts; defense comparisons retain all attempts. Results are
benchmark-macroaverageswithbenchmark-stratifiedconfi-
dence intervals; method comparisons use paired resampling.
Comparisons and checks.Construction baselines in-
clude direct multi-document synthesis, best-of- Nsynthesis,
genericself-refinement,splitevidence/payloadsynthesis,and
4

Metric Hotpot 2Wiki MuSiQue NQ Avg.
All attempts
Yield↑0.655 0.640 0.705 0.500 0.625
Ret. ASR@5↑0.517 0.490 0.611 0.357 0.494
E2E-Strict@5↑0.401 0.364 0.477 0.209 0.363
Returned packets
Full ASR↑0.862 0.841 0.910 0.795 0.852
Max-sub. TR↓0.036 0.029 0.069 0.141 0.069
JRR@5↑0.939 0.913 0.958 0.898 0.927
Ret. ASR@5↑0.790 0.765 0.866 0.713 0.784
E2E-Strict@5↑0.612 0.568 0.677 0.419 0.569
Table2:Mainresults.Theupperpanelusesall4,000attempts
and scores construction failures as zero. The lower uses all
returnedpacketswithouttarget-sidefiltering.E2E-Strict@5
requires complete packet retrieval, target adoption in the
actual top-5 context, and the controlled all-proper-subset
check.
Metric Value
Returned target-specific packets
Intended Ret. ASR@5 0.735
Other-target Ret. ASR@5 0.042
Selectivity gap 0.693
All 720 attempts
Yield 0.588
Intended Ret. ASR@5 0.434
Other-target Ret. ASR@5 0.024
All 240 queries
≥2selective targets 0.538
All 3 selective targets 0.158
Table3:Targetcontrollabilitywiththreepreselectedtargets
perquery.Packetrowsconditiononreturnedpackets,attempt
rowsscorefailuresaszero,andselectivereachabilityrequires
a gap of at least 0.30.
protocol-faithful Semantic Chameleon (Thornton 2026). All
share the fixed manifest, corpus, retrieval pipeline, target-
agnostic admission validator, and target-side evaluation; can-
didate selection uses frozen construction-side signals only.
LLM-based baselines additionally share the Qwen construc-
tionmodelandgenerationbudget,whileSemanticChameleon
uses its published native optimizer. We also compare Poi-
sonedRAG, GASLITE, and BadRAG under RobustRAG, Re-
liabilityRAG,SeCon-RAG, andTrustRAG(Zouetal. 2025;
Ben-Tov and Sharif 2025; Xue et al. 2024; Xiang et al. 2026;
Shen et al. 2025; Si et al. 2025; Zhou et al. 2025). Additional
checkscoverthree-targetcontrollability,Llama-familycon-
struction transfer, and independent 100-draw confirmation.
Main Results
Attack effectiveness.Table 2 shows that LENS returns pack-
etsfor62.5%offixedattempts,yielding0.494all-attemptRet.
ASR@5 and 0.363 E2E-Strict@5 (95% CI: [0.348,0.378] ).
Returned packets exhibit strong set dependence: conditionalAttack None SeCon Reliab. Robust Trust
PoisonedRAG.507.219 .286 .157 .274
GASLITE .454 .167 .274 .118 .255
BadRAG .473 .194 .259 .141 .291
LENS.494.401 .448 .260 .408
Table4:All-attemptdefendedASR@5ontheshared4,000-
attempt manifest. Construction failures and invalid outputs
are retained and scored as zero. Each cell aggregates three
target generators and 20 paired decoding seeds per attempt.
Human and joint validity outcome Result
Document and target validity
Locally supported inserts (n= 600) 552/600 (92.0%)
Plausible reference prose (n= 600) 535/600 (89.2%)
Incorrect frozen targets (n= 240) 211/240 (87.9%)
Joint semantic and behavioral validity
SemValid returned packets (n= 240) 164/240 (68.3%)
SemValid∧E2E-Strict pairs (n= 720) 331/720 (46.0%)
E2E-Strict|SemValid 331/492 (67.3%)
E2E-Strict|non-SemValid 78/228 (34.2%)
Table 5: Blinded human validation. Document judgments
use 600 inserts from 240 returned packets; target correctness
uses a separate pre-construction sample of 240 frozen targets.
SemValid is evaluated on the returned-packet audit pool
and requires an incorrect target, a definite criterion shift,
andnoentailmentundertheoriginalquerysemantics.Joint
behavioralratesattachthethreetarget-generatorE2E-Strict
outcomes to each of the 240 audited returned packets.
Full ASR is 0.852, Max-sub. TR is 0.069, and 79.7% pass
everyproper-subsetcheck.Moreover,89.7%ofCond-Strict
successes remain strict in theactual retrieved context, show-
ingthattheconstructeddependencetransfersfromcontrolled
evaluation to the deployed retrieval path.
Humanvalidation.Annotatorsareindependentofthecon-
struction validators and blind to their acceptance decisions
and model-side outcomes. Table 5 shows that inserted doc-
uments are predominantly locally supported and plausible,
while 87.9% of targets sampled before construction are incor-
rectundertheintendedquerysemantics.Moreimportantly,
68.3% of returned packets satisfy SemValid, and 46.0% of
theirgeneratorevaluationsarebothSemValidandE2E-Strict.
Strict success is nearly twice as frequent within SemValid
packets as outside them, linking the behavioral effect to
attacker-induced answer criteria.
Construction baselines.Table 6 shows that LENS raises
all-attemptE2E-Strict@5from0.244forthestrongestbase-
line to 0.363, a 48.8% relative gain. This advantage targets
the capability studied here: isolating attack success to the
complete document set. LENS lowers Max-sub. TR from
0.153 to 0.069, and the same conclusion holds under exact
two-document matching, where it reaches 0.397 versus 0.247
for native Semantic Chameleon.
Statistical robustness.The main conclusion is stable across
5

Figure3:Mechanismcontrolson240matchedinstances.Bars
show Ret. ASR@K and lines show JRR@K. Each condition
uses 720 realization–generator pairs, with JRR computed
once per realization. We use K= 5fork∈ {2,3} and
K= 8for the held-outk= 4extension.
construction thresholds: increasing τtrades coverage for
packet strength, while the nested ϵsweep changes subset
acceptancewithoutalteringthefull-setdependencepattern.
Independent 100-draw evaluation confirms all-subset control
for68.6%ofsampledpairs.All-attemptE2E-Strictremains
stable across target generators (0.357–0.370) and transfers to
Llama-guided construction.
Target specificity and spillover.Table 3 reports a 0.693
intended-versus-alternate target gap, with at least two targets
selectively reachable for 53.8% of queries. Across 1,200
unseen off-target query–packet pairs, full-packet retrieval
is 1.7%, original-target spillover is 0.8% versus 0.3% in
pairedcleanruns,andgoldaccuracychangesby1.5points.
Evenwhensame-entityqueriesretrieveindividualdocuments
more often (29.0%), the composed packet rarely transfers its
behavioral effect.
Robustness under defenses.Table 4 shows that LENS at-
Figure 4: Component ablations on the fixed 1,000-attempt
ablation manifest. Natural runs use each variant’s executed
budget; fixed-budget runs apply the same call, generation,
and token caps. Points report the primary all-attempt E2E-
Strict@5 metric.
tains the highest all-attempt ASR@5 under all four defenses,
exceeding the strongest baseline by 0.103–0.182 (0.141 on
average); paired intervals exclude zero. Its advantage com-
binesconstructioncoveragewithset-leveleffectsthatsurvive
defenses built around document-local cues.
Mechanism and Ablation
Component ablations.Figure 4 evaluates every compo-
nentonanindependent1,000-attemptmanifest.FullLENS
reaches 0.403 all-attempt E2E-Strict@5. Under a shared cap
of15proposal-or-repaircalls,800diagnosticgenerations,and
1.05Mtokens,itretains0.327,twicethestrongestcomponent
removal at 0.163. Removing subset updates preserves high
full-set success but reduces fixed-budget E2E-Strict@5 to
0.134, identifying subset-aware repair as the source of set
isolation.
Mechanismcontrols.Figure3showsthatmatchedpackets
achieve 0.785 Ret. ASR@K, compared with 0.265 after
rolescramblingand0.127aftercross-lenssubstitution.JRR
remains nearly unchanged across these conditions (0.863–
0.867), localizing the attack effect to the planned lens–fact
correspondence. The larger residual under role scrambling is
consistentwithretainingallsemanticatoms,whereascross-
lenssubstitutionreplacesthecriterionthatmakestherecipient
facts jointly decisive.
Composition-Aware Mitigation Probes
Local passage screening leaves LENS largely intact, whereas
defenses that evaluate the retrieved set expose its composi-
tional dependence. A set-level judge reduces Ret. ASR from
0.784to0.318,andcombiningjudgingwithsubsetablation
reaches 0.164. Development-frozen selective auditing pro-
vides the strongest operating point: it reduces Ret. ASR to
0.284whileretaining91.3%ofbaselinecleanaccuracy.These
results establish a practical defense principle: allocate set-
level inspection to answers whose support is both distributed
6

Method Yield FullMax-
sub.CJLPacket
Ret.Attempt
Ret.Attempt
E2E-Strict@5
Direct multi-doc .338 .754 .241 .513 .671 .227 .088
Best-of-Ndirect .447 .823 .198 .625 .741 .331 .154
Generic self-refine .472 .812 .146 .666 .735 .347 .190
Split evidence/payload .503 .842 .172 .670 .766 .385 .216
Semantic Chameleon .546.872.153 .719.798.436 .244
LENS.625.852.069 .783.784.494 .363
Table 6: Construction baselines on the fixed 4,000-attempt manifest. All methods share query–target pairs, corpus, retrieval
pipeline, target-agnostic admission, and target-side evaluation. LENS and the LLM-based baselines follow the assigned k= 2/3
budget;SemanticChameleonretainsitsnativetwo-documentsleeper–triggerGCGprotocol.Packetmetricsconditiononreturned
packets; attempt metrics score construction failures as zero.
and unstable under document removal.
Operational implications.The experiments identify joint
retrieval as both the operational bottleneck and the defense
lever. Incomplete packets rarely activate the target (0.028),
whiletargetedhardnegativesreduceRet.ASR@5to0.519,
showing that retrieval competition directly limits composi-
tional steering. Attack construction and defense therefore
meet at the same control point: which evidence relations
surviveretrievalandbecomejointlyavailabletothegenerator.
Extendingthisanalysisacrosslargerretrievalecosystemsand
competitive corpus settings is a natural next step.
Security implications.Together, these results redefine three
units of RAG security analysis. Attack models should treat
retrieved evidence sets as units of control; evaluation should
test whether target adoption depends on the complete packet;
and defenses should examine how passages jointly determine
an answer. This makes evidence-set dependence measurable
andframesdefensesarounddistinguishingattacker-induced
rules from legitimate multi-hop support.
Related Work
RAG corpus poisoning.RAG corpus poisoning uses
answer-bearingpassages,triggers,gradientoptimization,or
black-boxdocumentconstructiontosteerretrievedgeneration
(Lewis et al. 2020; Ram et al. 2023; Izacard and Grave 2021;
Asaietal.2024).Representativemethodsspanthesethreat
models(Zouetal.2025;Ben-TovandSharif2025;Chaudhari
et al. 2026; Wang et al. 2026b; Xian et al. 2025; Chang et al.
2025b;Zhangetal.2025a;Nazary,Deldjoo,anddiNoia2025;
Chen et al. 2025b). SilentRetrieval improves the fluency and
retrieval transfer of such poisons (Qian 2026). Prior methods
optimize attack success, retrievability, or stealth without con-
straining every proper subset. LENS makes this constraint its
constructionobjectiveandplacesthetargeteffectinevidence
composition.
Coordinatedandcompositeattacks.Priorworkstudies
sleeper–trigger pairs, agentic trajectories, prompt-injection-
plus-databasepoisoning,adversarialKGinferencechains,and
competingattackers(Thornton2026;Choietal.2026;Pan
etal.2026;Wangetal.2026a;Zhaoetal.2025;Chenetal.
2025a;Huangetal.2024).LENSisolatesacomplementary
regime: frozen single-round text RAG, where one natural-
language packet realizes an answer-level effect while everyproper subset remains insufficient. This all-proper-subset
objective distinguishes set-level composition from coordi-
natedretrieval,promptinjection,structuredinference-chain
poisoning, and attacker competition.
Detection and defense.RAG defenses and transferable
input-screeningmethods inspectorisolatepassages anduse
filtering,answeraggregation,clean-evidencerecovery,con-
sistencygraphs,clustering,orself-assessment(Qietal.2021;
Robey et al. 2023; Xiang et al. 2026; Tan et al. 2025; Yao
et al. 2025; Edemacu et al. 2025; Cheng et al. 2025; Kim,
Lee, and Koo 2025; Chang et al. 2025a; Shen et al. 2025;
Si et al. 2025; Zhou et al. 2025). These signals suit locally
abnormal or conflicting poisons. LENS reduces both cues
through plausible, locally insufficient documents. It therefore
motivates defenses that evaluate how passages jointly sup-
portananswer,extendingsecurityinspectionfromdocument
content to evidence composition.
Conclusion
WeintroducedLENS,aset-levelpoisoningframeworkthat
shiftsfrozenRAG’sattacksurfacefromindividualdocuments
totheircomposition.LENScombinesinterpretation-lensplan-
ning, role-separated synthesis, counterexample-guided repair,
and empirical proper-subset verification to construct pack-
ets whose full set redirects answers while proper subsets
maintain lowtarget rates.This exposes asecurity blind spot:
plausible, locally inconclusive documents can jointly induce
an attacker-chosen decision rule. Mechanism controls and
blinded audits localize this effect to the planned lens–fact
interface and establish it as a distinct form of evidence-set
control.Acrosstargetgenerators,LENSoutperformsstrong
construction baselines on strict set-level success and sur-
vives published defenses in the retrieved context. LENS also
provides a reusable stress test for evidence-set dependence,
allowing RAG models, retrieval pipelines, and defenses to be
evaluatedagainstanswercontrolthatemergesonlythrough
cross-documentcomposition.RAGsecuritymusttherefore
treatevidencecompositionasafirst-classattacksurfacewhile
preserving the cross-document reasoningthat gives retrieval
augmentation its value.
GenerativeAIusedisclosure.GenerativeAItoolswereused
forlanguageediting,LaTeXassistance,andfigureandcode
7

drafting.Theauthorsverifiedallscientificclaims,experimen-
tal results, analyses, citations, and final text.
References
Asai, A.;Wu,Z.; Wang, Y.; Sil, A.;and Hajishirzi, H.2024.
Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. InInternational Conference on
Learning Representations.
Ben-Tov,M.;andSharif,M.2025.GASLITEingtheRetrieval:
ExploringVulnerabilitiesinDenseEmbedding-BasedSearch.
InProceedingsofthe2025ACMSIGSACConferenceonCom-
puterandCommunicationsSecurity,4364–4378.Association
for Computing Machinery.
Chang, C.-Y.;Jiang, Z.; Rakesh,V.; Pan, M.;Yeh, C.-C.M.;
Wang, G.; Hu, M.; Xu, Z.; Zheng, Y.; Das, M.; and Zou,
N. 2025a. MAIN-RAG: Multi-Agent Filtering Retrieval-
AugmentedGeneration. InProceedingsofthe63rdAnnual
Meeting of the Association for Computational Linguistics
(Volume1:LongPapers),2607–2622.Vienna,Austria:Asso-
ciation for Computational Linguistics.
Chang, Z.; Li, M.; Jia, X.; Wang, J.; Huang, Y.; Jiang, Z.;
Liu, Y.; and Wang, Q. 2025b. One Shot Dominance: Knowl-
edgePoisoningAttackonRetrieval-AugmentedGeneration
Systems. InFindingsoftheAssociationforComputational
Linguistics: EMNLP 2025, 18811–18825. Suzhou, China:
Association for Computational Linguistics.
Chaudhari,H.;Severi,G.;Abascal,J.;Suri,A.;Jagielski,M.;
Choquette-Choo,C.A.;Nasr,M.;Nita-Rotaru,C.;andOprea,
A.2026. Phantom:GeneralBackdoorAttacksonRetrieval
Augmented Language Generation.ACM Transactions on AI
Security and Privacy.
Chen,J.;Xiao,S.;Zhang,P.;Luo,K.;Lian,D.;andLiu,Z.
2024. M3-Embedding:Multi-Linguality,Multi-Functionality,
Multi-GranularityTextEmbeddingsThroughSelf-Knowledge
Distillation. InFindingsoftheAssociationforComputational
Linguistics: ACL 2024, 2318–2335. Bangkok, Thailand: As-
sociation for Computational Linguistics.
Chen,L.;Yang,X.;Lu,Y.;Zhang,J.;Sun,X.;Liu,Q.;Wu,
S.; Dong, J.; and Wang, L. 2025a. PoisonArena: Uncover-
ing Competing Poisoning Attacks in Retrieval-Augmented
Generation.arXiv preprint arXiv:2505.12574.
Chen, Z.; Gong, Y.; Liu, J.; Chen, M.; Liu, H.; Cheng,
Q.; Zhang, F.; Lu, W.; and Liu, X. 2025b. FlippedRAG:
Black-Box Opinion Manipulation Adversarial Attacks to
Retrieval-Augmented Generation Models.arXiv preprint
arXiv:2501.02968.
Cheng, Z.; Sun, J.; Gao, A.; Quan, Y.; Liu, Z.; Hu, X.; and
Fang, M. 2025. Secure Retrieval-Augmented Generation
Against Poisoning Attacks. In2025 IEEE International
Conference on Big Data, 1799–1806. ArXiv:2510.25025.
Choi, C.; Kim, E.; Lee, K.; Chun, Y.; Jeong, J.; Kim,
E.; Oh, M.; Jang, J.; and Chang, B. 2026. KidnapRAG:
A Black-Box Attack for Hijacking Reasoning in Agentic
Retrieval-Augmented Generation Systems.arXiv preprint
arXiv:2607.00422.Cormack, G. V.; Clarke, C. L. A.; and Buettcher, S. 2009.
Reciprocal Rank Fusion Outperforms Condorcet and Indi-
vidualRank LearningMethods. InProceedingsofthe 32nd
InternationalACMSIGIRConferenceonResearchandDe-
velopmentinInformationRetrieval,758–759.Associationfor
Computing Machinery.
DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Efficient
Million-Token Context Intelligence.
Edemacu,K.;Shashidhar,V.M.;Tuape,M.;Abudu,D.;Jang,
B.; and Kim, J. W. 2025. Defending Against Knowledge
Poisoning AttacksDuring Retrieval-AugmentedGeneration.
arXiv preprint arXiv:2508.02835.
Grattafiori,A.;Dubey,A.;Jauhri,A.;Pandey,A.;Kadian,A.;
Al-Dahle,A.;Letman,A.;Mathur,A.;Schelten,A.;Vaughan,
A.; et al. 2024. The Llama 3 Herd of Models.arXiv preprint
arXiv:2407.21783.
Ho,X.;Nguyen,A.-K.D.;Sugawara,S.;andAizawa,A.2020.
Constructing a Multi-hop QA Dataset for Comprehensive
Evaluation of Reasoning Steps. InProceedings of the 28th
InternationalConferenceonComputationalLinguistics,6609–
6625. Barcelona, Spain (Online): International Committee
on Computational Linguistics.
Huang, H.; Zhao, Z.; Backes, M.; Shen, Y.; and Zhang, Y.
2024. Composite Backdoor Attacks Against Large Language
Models. InFindings of the Association for Computational
Linguistics: NAACL 2024, 1459–1472. Mexico City, Mexico:
Association for Computational Linguistics.
Izacard, G.; and Grave, E. 2021. Leveraging Passage Re-
trievalwithGenerativeModelsforOpenDomainQuestion
Answering. InProceedings of the 16th Conference of the
European Chapter of the Association for Computational Lin-
guistics: Main Volume, 874–880. Online: Association for
Computational Linguistics.
Kim, M.; Lee, H.; and Koo, H. 2025. Rescuing the Un-
poisoned:EfficientDefenseagainstKnowledgeCorruption
Attacks on RAG Systems.arXiv preprint arXiv:2511.01268.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang,
M.-W.; Dai, A. M.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019.
Natural Questions: A Benchmark for Question Answering
Research.Transactions of the Association for Computational
Linguistics, 7: 453–466.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal, N.; Kuttler, H.; Lewis, M.; Yih, W.-t.; Rocktaschel,
T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented
GenerationforKnowledge-IntensiveNLPTasks. InAdvances
in Neural Information Processing Systems, volume 33, 9459–
9474.
Nazary, F.; Deldjoo, Y.; and di Noia, T. 2025. Poison-
RAG: Adversarial Data Poisoning Attacks on Retrieval-
Augmented Generation in Recommender Systems.arXiv
preprint arXiv:2501.11759.
Pan, Y.;Zhang, Z.;Lei, J.;Jia, C.;Si,Q.; andGuo, H.2026.
FORGE: Research-Trajectory Hijacking Attacks on Deep
Research Agents.arXiv preprint arXiv:2607.04718.
8

Qi, F.; Chen, Y.; Li, M.; Yao, Y.; Liu, Z.; and Sun, M.
2021. ONION: A Simple and Effective Defense Against
TextualBackdoorAttacks. InProceedingsofthe2021Confer-
ence on Empirical Methods in Natural Language Processing,
9558–9566. Online and Punta Cana, Dominican Republic:
Association for Computational Linguistics.
Qian, J. 2026. SilentRetrieval: Hijacking Retrieval-
AugmentedGenerationviaSemantically-PreservingAdver-
sarial Data Poisoning.arXiv preprint arXiv:2605.28074.
Qwen Team. 2026. Qwen3.6 Model Collection. Hugging
Face model collection.
Ram, O.; Levine, Y.; Dalmedigos, I.; Muhlgay, D.; Shashua,
A.; Leyton-Brown, K.; and Shoham, Y. 2023. In-Context
Retrieval-Augmented Language Models.Transactions of the
Association for Computational Linguistics, 11: 1316–1331.
Robertson, S.; and Zaragoza, H. 2009. The Probabilistic
Relevance Framework: BM25 and Beyond.Foundations and
Trends in Information Retrieval, 3(4): 333–389.
Robey, A.; Wong, E.; Hassani, H.; and Pappas, G. J. 2023.
SmoothLLM: Defending Large Language Models Against
Jailbreaking Attacks.arXiv preprint arXiv:2310.03684.
Shen, Z.; Imana, B.; Wu, T.; Xiang, C.; Mittal, P.; and
Korolova,A.2025. ReliabilityRAG:EffectiveandProvably
Robust Defense for RAG-based Web-Search.Advances in
Neural Information Processing Systems, 38: 45662–45702.
Si, X.; Zhu, M.; Qin, S.; Yu, L.; Zhang, L.; Liu, S.; Li, X.;
Duan,R.;Liu,Y.;andJia,X.2025.SeCon-RAG:ATwo-Stage
Semantic Filtering and Conflict-Free Framework for Trust-
worthy RAG.Advances in Neural Information Processing
Systems, 38: 70652–70681.
Tan, X.; Luan, H.; Luo, M.; Sun, X.; Chen, P.; and Dai, J.
2025. RevPRAG:RevealingPoisoningAttacksinRetrieval-
AugmentedGenerationthroughLLMActivationAnalysis. In
Findings of the Association for Computational Linguistics:
EMNLP2025,12999–13011.Suzhou,China:Associationfor
Computational Linguistics.
Thornton,S.2026. SemanticChameleon:Corpus-Dependent
Poisoning Attacks and Defenses in RAG Systems.arXiv
preprint arXiv:2603.18034.
Trivedi,H.;Balasubramanian,N.;Khot,T.;andSabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
QuestionComposition.TransactionsoftheAssociationfor
Computational Linguistics, 10: 539–554.
Wang, H.; Liu, H.; Zhu, J.; Wang, Z.; Guo, Y.; and Tang,
X. 2026a. PIDP-Attack: Combining Prompt Injection with
Database Poisoning Attacks on Retrieval-Augmented Gener-
ation Systems.arXiv preprint arXiv:2603.25164.
Wang, H.; Zhang, R.; Wang, J.; Li, M.; Huang, Y.; Wang,
D.; and Wang, Q. 2026b. Joint-GCG: Unified Gradient-
BasedPoisoningAttacksonRetrieval-AugmentedGeneration
Systems. InProceedingsoftheAAAIConferenceonArtificial
Intelligence, volume 40, 35793–35801.
Xian, X.; Wang, G.; Bi, X.; Zhang, R.; Srinivasa, J.; Kundu,
A.;Fleming,C.;Hong,M.;andDing,J.2025. OntheVulner-
abilityofApplyingRetrieval-AugmentedGenerationwithin
Knowledge-Intensive Application Domains. InProceedingsof the 42nd International Conference on Machine Learning,
volume267ofProceedingsofMachineLearningResearch,
68292–68315. PMLR.
Xiang, C.; Wu, T.; Zhong, Z.; Wagner, D.; Chen, D.; and
Mittal,P.2026. CertifiablyRobustRAGagainstRetrievalCor-
ruption. InConferenceonSecureandTrustworthyMachine
Learning (SaTML).
Xue, J.; Zheng, M.; Hu, Y.; Liu, F.; Chen, X.; and Lou,
Q.2024. BadRAG:IdentifyingVulnerabilitiesinRetrieval
Augmented Generation of Large Language Models.arXiv
preprint arXiv:2406.00083.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W.; Salakhut-
dinov,R.;andManning,C.D.2018. HotpotQA:ADataset
for Diverse, Explainable Multi-hop Question Answering. In
Proceedings of the 2018 Conference on Empirical Meth-
odsinNaturalLanguageProcessing,2369–2380.Brussels,
Belgium: Association for Computational Linguistics.
Yao, R.; Zhang, Y.; Song, S.; Gao, N.; and Tu, C. 2025.
EcoSafeRAG: Efficient Security through Context Analysis in
Retrieval-AugmentedGeneration. InFindingsoftheAssocia-
tionforComputationalLinguistics:EMNLP2025,4034–4050.
Suzhou, China: Association for Computational Linguistics.
Zhang, C.; Zhang, X.; Lou, J.; Wu, K.; Wang, Z.; and
Chen, X. 2025a. PoisonedEye: Knowledge Poisoning At-
tackonRetrieval-AugmentedGenerationbasedLargeVision-
Language Models. InProceedings of the 42nd International
ConferenceonMachineLearning,volume267ofProceedings
of Machine Learning Research, 76811–76830. PMLR.
Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou,
J. 2025b. Qwen3 Embedding: Advancing Text Embedding
andRerankingThroughFoundationModels.arXiv preprint
arXiv:2506.05176.
Zhao, T.; Chen, J.; Ru, Y.; Zhu, H.; Hu, N.; Liu, J.; and
Lin, Q. 2025. RAG Safety: Exploring Knowledge Poisoning
AttackstoRetrieval-AugmentedGeneration.arXivpreprint
arXiv:2507.08862.
Zhou, H.; Lee, K.-H.; Zhan, Z.; Chen, Y.; Li, Z.; Wang,
Z.; Haddadi, H.; and Yilmaz, E. 2025. TrustRAG: Enhanc-
ing Robustness and Trustworthiness in Retrieval-Augmented
Generation.arXiv preprint arXiv:2501.00879.
Zou,W.;Geng,R.;Wang,B.;andJia,J.2025. PoisonedRAG:
Knowledge Corruption Attacks to Retrieval-Augmented Gen-
erationofLargeLanguageModels. In34thUSENIXSecurity
Symposium (USENIX Security 25), 3827–3844. Seattle, WA:
USENIX Association.
9
# GuideSkill: Evolving Executable LLM Agent Skills for Guideline-Grounded Clinical Reasoning

**Authors**: Lang Cao, Yuhao Shen, Tianyang Luo, Simo Du, Hao Peng, Yue Guo

**Published**: 2026-07-28 18:10:33

**PDF URL**: [https://arxiv.org/pdf/2607.26160v1](https://arxiv.org/pdf/2607.26160v1)

## Abstract
Clinical practice guidelines (CPGs) encode diagnostic criteria, but LLM systems typically retrieve guideline text or absorb it through training rather than execute its rules. We introduce GuideSkill, an external reasoning layer that compiles disease-specific criteria into executable functions returning ordinal diagnostic-support scores. GuideSkill-Zero is initialized from guidelines, while GuideSkill-Evo uses case--diagnosis pairs to refine covered skills and add missing diagnoses. At inference, an LLM proposes a differential diagnosis, grounds the features required by each matched skill, and fuses its ranking with the executed skill scores. Across four benchmarks and four backbones, GuideSkill-Zero improves macro-average accuracy over guideline RAG by 13.45% on average. GuideSkill-Evo achieves the highest macro-average for every backbone, improves over direct inference by 18.49% relatively, and increases gold-label skill coverage from 56.5% to 99.5%. On Qwen3.5-9B, it also exceeds the strongest parameter-update baseline by 11.16% without updating the backbone. Expert evaluation further indicates that GuideSkill produces clinically sound and broadly acceptable skills, suggesting that its initialized and evolved rules are reliable and practically meaningful. These results support executable skills as a model-agnostic mechanism for combining guideline-derived procedures with case-derived diagnostic patterns.

## Full Text


<!-- PDF content starts -->

GuideSkill: Evolving Executable LLM Agent Skills for
Guideline-Grounded Clinical Reasoning
Lang Cao Yuhao Shen✦Tianyang Luo Simo Du✧Hao Peng Yue Guo
University of Illinois Urbana-Champaign
✦The Chinese University of Hong Kong, Shenzhen
✧Albert Einstein College of Medicine
Abstract
Clinical practice guidelines (CPGs) encode diagnostic crite-
ria,butLLMsystemstypicallyretrieveguidelinetextorabsorb
it through training rather than execute its rules. We introduce
GuideSkill,anexternalreasoninglayerthatcompilesdisease-
specific criteria into executable functions returning ordinal
diagnostic-supportscores.GuideSkill-Zeroisinitializedfrom
guidelines,whileGuideSkill-Evousescase–diagnosispairsto
refinecoveredskillsandaddmissingdiagnoses.Atinference,
an LLM proposes a differential diagnosis, grounds the fea-
tures required by each matched skill, and fuses its ranking
with the executed skill scores. Across four benchmarks and
fourbackbones,GuideSkill-Zeroimprovesmacro-averageac-
curacyoverguidelineRAGby13.45%onaverage.GuideSkill-
Evoachieves the highest macro-average for every backbone,
improves over direct inference by 18.49% relatively, and in-
creases gold-label skill coverage from 56.5% to 99.5%. On
Qwen3.5-9B, it also exceeds the strongest parameter-update
baseline by 11.16% without updating the backbone. Expert
evaluationfurtherindicatesthatGuideSkillproducesclinically
soundandbroadlyacceptableskills,suggestingthatitsinitial-
izedandevolvedrulesarereliableandpracticallymeaningful.
These results support executable skills as a model-agnostic
mechanismforcombiningguideline-derivedprocedureswith
case-derived diagnostic patterns.
1 Introduction
Clinical diagnosis is not only a knowledge-recall problem.
It is a multistep process that requires gathering and synthe-
sizing patient information, generating and comparing plau-
sible diagnoses, determining which findings and thresholds
support or weaken each candidate, excluding alternatives,
and identifying decisive tests (McDuff et al. 2025; Hager
et al. 2024; Cao, Chen, and Guo 2026; You et al. 2026).
Clinical practice guidelines (CPGs) encode evidence-based
recommendations and conditional decision logic that can
guidethesedecisions(InstituteofMedicine2011;Shenetal.
2026b). However, providing guideline-derived criteria to an
LLM does not guarantee that they will be applied correctly
tothepatient:evenwhenguideline-basedchecklistsaresup-
plied,criterion-levelevaluationremainsimperfect(Schubert
et al. 2025). LLMs can generate ranked differential diag-
noses from free-text cases (McDuff et al. 2025), but may
omit clinician-identified reasoning evidence and fail to fol-
lowdiagnosticguidelines(Wuetal.2025;Hageretal.2024).We therefore ask whether CPGs can be transformed from
passivereferencesintoanevolvablelibraryofexecutabledi-
agnostic skills that combines flexible candidate generation
with explicit disease-specific rule application.
Prior work incorporates CPGs by supplying guideline
text or checklists at inference (Schubert et al. 2025), adapt-
ing model parameters with guideline-containing corpora or
guideline-derived supervision (Chen et al. 2023; Staniek,
Sokolov,andRiezler2025;Shenetal.2026b),andtranslating
guidelines into structured decision trees or program-aided
pathways(Onianietal.2024;Lietal.2023;Dengetal.2026).
Theseapproachesdemonstrateseveralwaystooperationalize
guideline knowledge in LLM systems, but they leave open
a complementary systems question: can guideline-derived
proceduresbeorganizedasanexternal,disease-indexedskill
library that compares evidence across candidate diagnoses,
transfersacrossLLMbackbones,andexpandswithoutupdat-
ing model parameters? This question is particularly relevant
todifferentialdiagnosis,whereseveralconditionsmayplau-
sibly explain the same presentation and must be compared
against disease-specific evidence (McDuff et al. 2025). Be-
cause the initial guideline corpus covers only a finite set of
diagnoses, we further examine whether labeled cases can
extend the library to diagnoses absent from that corpus, in-
cludingtherareandcomplexconditionsrepresentedincase-
report benchmarks (Wu et al. 2025).
We introduceGuideSkill, which transforms CPGs from
passive references into an external library of executable,
disease-specific skills. Externalizing this knowledge makes
diagnostic criteria reusable across LLM backbones and al-
lows the library to be updated without modifying model pa-
rameters.GuideSkill-Zeroinitializes the library from guide-
lines, providing explicit supporting and contradictory rules
for covered diagnoses.GuideSkill-Evothen uses labeled
casestorefineexistingskillsandadddiagnosesabsentfrom
the guideline corpus, expanding coverage while updating
disease-specific decision rules outside the backbone. At in-
ference, the LLM and the skill library address complemen-
tary limitations. The LLM generates a ranked differential
from the full patient narrative, including diagnoses beyond
the finite skill library, while matched skills apply explicit
disease-specific criteria to covered candidates. Their fusion
preserves open-ended candidate generation while allowing
reusable evidence rules to refine the ranking, rather than re-
1
arXiv:2607.26160v1  [cs.AI]  28 Jul 2026

lying entirely on either an LLM-only ranking or incomplete
skill coverage. Because the skills remain external, the same
diagnosticlogiccanbeinspected,updated,andreusedacross
backbones.
WeevaluateGuideSkillonfourheterogeneousdiagnostic-
reasoning benchmarks—MedCaseReasoning (Wu et al.
2025), ER-Reason (Mehandru et al. 2025), MIMIC-CDM-
FI (Hager et al. 2024), and MedThink-Bench (Zhou et al.
2025)—using four proprietary and open-weight LLM back-
bones. Using only guideline-derived executable skills,
GuideSkill-Zeroachieves higher macro-average accuracy
than guideline RAG for every backbone. After evolu-
tion,GuideSkill-Evooutperforms direct inference in all 16
dataset–backbone comparisons, with a mean relative im-
provement of 18.49%, while increasing pooled gold-label
skillcoveragefrom56.50%to99.50%.OnMedThink-Bench,
which is excluded from evolution training, it achieves the
bestortied-bestaccuracyacrossallfourbackbones,demon-
strating transfer to an unseen benchmark. On Qwen3.5-9B,
it also outperforms the evaluated supervised fine-tuning,
reinforcement-learning, and guideline-decision-tree base-
lines without updating the backbone. Together, these results
demonstrate the benefits of external executable skills for di-
agnostic accuracy and coverage.
Our contributions are threefold:
•We formulate guideline-grounded diagnosis as agentic
skill execution and develop a disease-indexed compila-
tionpipelinethatconvertsCPGrecommendationsintoex-
ecutable ordinal scorers.
•Weintroduceacase-conditionedevolutionmechanismthat
refinescoveredskillsandaddspreviouslyuncovereddiag-
noses outside the LLM parameters.
•Wedevelopacandidate-levelinferenceprocedureandeval-
uateitacrossfourbackbonesandfourbenchmarks,separat-
ing the contribution of guideline-only initialization from
the additional coverage and downstream gains obtained
through evolution.
2 Related Work
Clinical Guidelines as Reasoning Resources.Because
CPGs synthesize reviewed clinical evidence into recom-
mendations, they are a valuable foundation for clinical
decision support (Institute of Medicine 2011). LLM-based
systems have incorporated them as retrieved prompt
context (Schubert et al. 2025; Oniani et al. 2024), super-
vised examples (Staniek, Sokolov, and Riezler 2025), or
reinforcement-learning signals (Tziakouri and Menolascina
2025). More structured approaches operationalize their
decision logic: MedDM represents clinical pathways as
LLM-executable guidance trees (Li et al. 2023), CPG-
Prompt translates CPGs into decision trees traversed during
inference (Deng et al. 2026), and MedGuideX executes
guideline logic to generate factual and counterfactual
supervision for post-training (Shen et al. 2026b). However,
the guideline-derived component of these systems remains
bounded by the source corpus: it provides no procedure
for diagnoses outside that corpus and no mechanism to
learn additional diagnostic patterns from labeled casepresentations.GuideSkilladdresses this gap by initializing
an external skill library from guidelines and using labeled
cases to refine covered skills and add uncovered diagnoses.
It thereby retains guideline-derived procedures while
extending beyond the initial corpus.
Evolving Skills in Healthcare and Biomedicine.Agent
skillsexternalizereusableproceduressothatcapabilitiescan
accumulatewithoutchangingmodelparameters.Trace2Skill
induces transferable operating procedures from recurring
patternsinexecutiontrajectories(Nietal.2026),whileSkill-
Claw aggregates cross-user trajectories to refine and extend
a shared skill repository (Ma et al. 2026). In healthcare,
an empirical study of public skills finds that they primarily
supportpatient-facingworkflowautomationandmonitoring,
with limited coverage of diagnosis and treatment (Xu et al.
2026). Related biomedical systems target scientific work-
flows:SkillFoundryminesheterogeneousresourcesintoval-
idatedexecutableskillsanddemonstratesthemongenomics
tasks (Shen et al. 2026a), whereas STELLA evolves rea-
soning templates and tool use for biomedical research and
experimental discovery (Jin et al. 2025). These studies cap-
tureexperienceoroperationalproceduresbutdonotaddress
evolving evidence-based knowledge for patient-level diag-
nosis.GuideSkillfills this gap by initializing skills from
CPGs, expanding them with labeled cases, and executing
them against patient evidence.
3 Methodology
Figure1providesanoverviewofGuideSkill.Theframework
first constructs an initial skill library,GuideSkill-Zero, from
clinical guidelines, then evolves the library with real patient
casestoobtainGuideSkill-Evo,andfinallyappliesthelearned
disease-specific skills during inference.
3.1 Task Formulation
Given an undiagnosed patient casex, our goal is to predict
the final diagnosisd⋆∈ D. Here,xincludes the patient’s
demographics,chiefcomplaint,medicalhistory,physicalex-
amination findings, and available test results. The diagnosis
spaceDisstandardizedusingthree-characterWHOICD-10
categories(WorldHealthOrganization2019),whichprovide
normalized disease identifiers across datasets.
During inference, an LLMMfirst generates a ranked
differential diagnosis set:
C(x) ={d 1, . . . , d K},C(x)⊆ D,(1)
whereK= 5by default. For each candidate diagnosis
d∈ C(x), we convert its rank into an LLM ranking score
sLLM(x, d). Specifically, ifdis ranked at positionr dwith
zero-based indexing, then
sLLM(x, d) =1
rd+ 1.(2)
GuideSkillthen retrieves the corresponding disease-
specific skillg dfrom the skill libraryGand executes it on
the patient case to obtain a skill-based evidence score:
sskill(x, d) =g d(x).(3)
2

Skill InitializationClinical Practice GuidelineFiltering& Curation
1
Executable SkillLibraryGuideSkill-ZeroLLM-basedConversionDepressive Episode(F32):In a terminally ill patient, diagnose a depressive episode when depressed mood persists for at least two weeks and is accompanied by symptoms of hopelessness, helplessness, worthlessness, guilt, lack of reactivity, or suicidal ideation.……
Depressive_Episode_F32_skill.py
Skill Evolution2Skill Execution3Real Patient CaseGround-truth DiagnosisLLM-generated Rationale
Disease-wise Aggregation & ExperienceDistillationDisease-specific Clinical Rules
GuideSkill-EvoQuality& CoverageSkill-Available PathSkill-Missing Path
Skill GenerationSkill OptimizationIntegration
UndiagnosedPatient CaseLLM-basedDifferential Diagnosis
Diagnosis A
Diagnosis B
Diagnosis CLLM Ranking ScoreSkill EvidenceScore
GuideSkill-Evo
FusedScoreGuideline-GroundedExecution0.610.30.30.510.450.750.65Final Diagnosis DeterminationbyMax-ScoreSelection
Because his interest and pleasure are gone, won‘t eat ……Figure1:OverviewofGuideSkill.Guidelinerecommendationsarecompiledintotheinitialexecutablelibrary(GuideSkill-Zero);
labeledcasesrefinecoveredskillsandaddmissingdiagnoses(GuideSkill-Evo);and,duringinference,executedskillscoresare
fused with an LLM-generated differential ranking.
The final diagnosis score combines the LLM ranking score
and the skill-based evidence score:
S(x, d) =αs LLM(x, d) + (1−α)s skill(x, d),(4)
whereα∈[0,1]controlstherelativecontributionofthetwo
signals. The final prediction is selected from the candidate
set:
ˆd= arg max
d∈C(x)S(x, d).(5)
3.2 Skill Initialization
The first stage constructs the initial skill library, denoted
asGuideSkill-Zero. Given a collection of clinical practice
guidelines,P={p 1, . . . , p M},we first filter and curate the
raw guideline documents to obtain a high-quality guideline
setP′. Details of the guideline preprocessing are provided
in Appendix B. Each curated guideline is then parsed into
disease-specific clinical recommendations, where each rec-
ommendation captures a clinically relevant diagnostic crite-
rion,finding,exclusionrule,ordecisionrulethatsupportsor
refutes a specific diagnosis.
SinceGuideSkilloperatesattheICD-10categorylevel,we
group recommendations by their mapped ICD-10 diagnosis
category.Foradiagnosiscategoryd,wedenotetheextracted
recommendation set as:R d={r d,1, rd,2, . . . , r d,m d},
wherem dis the number of recommendations extracted and
mapped to diagnosis categoryd.
EachrecommendationsetR disthenconvertedbyanLLM
into executable diagnostic logic, yielding an initial disease-
specific skill:
g(0)
d= LLM Compile (Rd).(6)
The initialized skill library is therefore defined as:
G(0)={g(0)
d|d∈ D guide},(7)
whereD guidedenotesthesetofICD-10diagnosiscategories
covered by the curated guidelines.Each skillg(0)
dis implemented as an executable Python
function. It takes the relevant features of a patient casexas
input and returns a skill-based evidence score:
sskill(x, d) =g(0)
d(x).(8)
The score is computed by matching the clinical evidence in
xagainst the executable diagnostic rules encoded ing(0)
d.
Specifically, each skill assigns the evidence to one of four
support levels:Confirmed,Strongly Suggestive,
Compatible, andNot Supported. Contradictory evi-
dence is further used to downgrade the support level when
applicable. The final evidence score is normalized to the
range[0,1]. Since each skill is executable, it can be directly
invoked by an LLM agent to obtain a structured evidence
score for diagnosisd. In this way,GuideSkill-Zeroprovides
aguideline-groundedandinterpretablemechanismfordiag-
nosis scoring.
3.3 Skill Evolution
AlthoughGuideSkill-Zeroisgroundedinclinicalguidelines,
guidelines mainly capture general diagnostic principles and
key decision points, while real-world cases often contain
heterogeneous presentations and atypical clinical patterns.
We therefore evolve the skill library using labeled patient
cases. Let the training set beT={(x j, dj)}n
j=1, wherex j
is a patient case andd jis the ground-truth diagnosis.
For each case(x j, dj), the LLM generates a diagnostic
rationalez jthat summarizes the clinical evidence support-
ingd j. Rationales associated with the same diagnosis are
thenaggregatedanddistilledintoadditionaldisease-specific
rules:
∆Rd= LLM Distill 
{zj|dj=d}
.(9)
Thenewlydistilledrulesareusedtoupdatetheskilllibrary.
Ifdiagnosisdalreadyhasanexistingskill,therulesareused
to optimize that skill; otherwise, they are used to generate a
3

new skill:
g(t+1)
d=(
Optimize 
g(t)
d,∆R d
,ifd∈ DG(t),
Generate 
∆Rd
,otherwise.(10)
Here,DG(t)denotes the set of diagnoses currently covered
by the skill libraryG(t). After applying this update across
training diagnoses, we obtain the evolved skill libraryGEvo.
Thisprocessimprovesbothskillqualityandskillcoverage:
existing skills become better aligned with real patient cases,
while diagnoses missing from the initial guideline-derived
library can be newly added.
3.4 Skill Execution
During inference, given an unseen patient casex, the LLM
first generates a ranked differential diagnosis setC(x)⊆ D.
Foreachcandidatediagnosisd i∈ C(x),GuideSkillretrieves
the corresponding skill from the evolved skill library by
matching either the disease name or the ICD-10 code:
gdi←Retrieve(GEvo, di).(11)
Because different skills may require different clinical fea-
tures, the LLM extracts the skill-specific inputs needed by
each retrieved skill:
ϕdi(x) = LLM feat(x, g di),(12)
whereϕ di(x)denotesthesubsetofpatientfeaturesrequired
to executeg di. The retrieved skill is then executed on these
extracted features to produce a skill-based evidence score:
sskill(x, d i) =g di 
ϕdi(x)
.(13)
Meanwhile, the LLM-generated differential diagnosis list
provides a ranking over the candidate set. We convert this
rank into an LLM ranking score:
sLLM(x, d i) =1
rank i+ 1,(14)
whererank iis the zero-based rank of candidated i. The
final fused score combines the LLM ranking score with the
skill-based evidence score:
sfuse(x, d i) =αs LLM(x, d i) + (1−α)s skill(x, d i),(15)
whereα∈[0,1]controlstherelativecontributionofthetwo
signals. By default, we setα= 0.5. The final diagnosis is
selected as:
ˆd= arg max
di∈C(x)sfuse(x, d i).(16)
Compared with direct LLM inference, this process
grounds the final decision in executable clinical rules while
preserving the broad diagnostic capability of the LLM.
4 Experiments
Data.WeevaluateGuideSkill-ZeroandGuideSkill-Evoon
fourheterogeneousdiagnostic-reasoningbenchmarks:Med-
CaseReasoning (Wu et al. 2025), ER-Reason (Mehandru
et al. 2025), MIMIC-CDM-FI (Hager et al. 2024), and
MedThink-Bench (Zhou et al. 2025), spanning published
0 20 40 60 80 100
share of gold diagnoses in the test split (%)MedCase
ER-Reason
MIMIC
MedThink34 7 6 9 7 7 10
9 15 9 12 7 8 10 17
100
7 11 9 11 15 7 7 22ICD-10 chapter
Neoplasms
Digestive
Circulatory
Infectious
Musculoskeletal
Endocrine/Metabolic
Genitourinary
Blood/Immune
Skin
Congenital
Respiratory
OtherFigure2:Test-splitdiagnosisdistributionbyICD-10chapter
across the four benchmarks. Each horizontal bar shows the
percentage of gold diagnoses belonging to each chapter.
Statistic MedCase ER-Reason MIMIC MedThink Total
Training split for evolution (#)
Cases 11,598 1,235 219 - 13,052
Distinct ICD-10 categories 463 232 5 - 473
New skills added 263 102 0 - 267
Test split for inference (#)
Cases 894 360 94 55 1,403
Distinct ICD-10 categories 389 130 5 49 448
Skill coverage on test cases (%)
GuideSkill-Zero43.1 79.2 100.0 50.9 56.5
GuideSkill-Evo100.0 100.0 100.0 87.3 99.5 (+43.0)
Table1:Benchmarkstatisticsandskillcoverage.GuideSkill-
Evosubstantially improves test-case skill coverage over
GuideSkill-Zero, increasing overall coverage by 43.0 points.
casenarratives,sequentialemergency-departmentreasoning,
full-information acute-abdominal diagnosis, and multistep
medical QA. Training splits from the first three benchmarks
areusedforskillevolution,whereasMedThink-Benchisre-
served for external evaluation and excluded from evolution.
For cross-benchmark comparison, we normalize reference
diagnosestothree-characterICD-10categories,suchasK35
foracuteappendicitis,andevaluatecategory-levelratherthan
subtype-level diagnosis. We useClaude-Sonnet-4.6for skill
initialization and evolution. The four test sets contain 1,403
cases across 448 ICD-10 categories (Table 1), with their
chapter-level diagnosis distributions shown in Figure 2.
BaselinesTo assessGuideSkillagainst representative
inference-time alternatives under a common evaluation pro-
tocol, we compare six baselines: direct prompting, chain-
of-thought prompting (Wei et al. 2022), 3-shot in-context
learning (Brown et al. 2020), guideline RAG (Lewis et al.
2020), LLM-generated differential diagnosis (DDx), and
LLM-generated DDx with guideline retrieval. These base-
lines test whether performance gains can be explained by
explicit reasoning, case demonstrations, access to guide-
linetext,broadercandidategeneration,ortheircombination.
We evaluate every method usingGPT-5.4,Claude-Sonnet-
4.6,MedGemma-27B, andQwen3.5-9B, covering propri-
etary and open-weight, general-purpose and medically spe-
cializedbackbones.WefurthercompareGuideSkillwithrep-
resentative parameter-update and structured-guideline alter-
nativesonQwen3.5-9B:fine-tuningwithguidelines(Staniek,
Sokolov, and Riezler 2025), cases (Wu et al. 2025), or both;
RL with cases (Chen et al. 2024); guideline fine-tuning fol-
lowed by case-based RL; and guidelines represented as de-
cision trees (Deng et al. 2026). These methods span the
4

Base Model Method MedCaseReasoning ER-Reason MIMIC-CDM-FI MedThink-Bench Average
GPT-5.4Direct 25.73 42.2293.6229.09 47.67
CoT 25.17 43.06 89.3636.3648.49
3-shot ICL28.97 43.8992.55 34.5549.99
RAG 23.71 43.0693.6232.73 48.28
LLM DDx 25.28 40.00 90.43 34.55 47.57
LLM DDx + RAG 25.39 39.72 90.43 27.27 45.70
GuideSkill-Zero 35.01 53.06 95.74 34.55 54.59
GuideSkill-Evo 39.71+37.07% 55.00+25.31% 96.81+3.41% 36.36+0.00% 56.97+13.96%
Claude-Sonnet-4.6Direct 23.27 41.1192.55 34.5547.87
CoT 23.94 39.17 89.36 32.73 46.30
3-shot ICL25.9539.9492.55 34.55 48.25
RAG 23.8343.0691.49 29.09 46.87
LLM DDx 22.04 34.4492.5530.91 44.99
LLM DDx + RAG 22.15 35.56 91.49 27.27 44.12
GuideSkill-Zero 32.33 47.22 94.68 25.45 49.92
GuideSkill-Evo 40.27+55.18% 50.56+17.42% 95.74+3.45% 40.00+15.77% 56.64+17.39%
MedGemma 27BDirect 19.02 32.22 91.49 25.45 42.05
CoT 20.02 33.8992.5525.45 42.98
3-shot ICL20.2530.28 91.4927.2742.32
RAG 14.54 30.83 91.49 21.82 39.67
LLM DDx 18.57 39.1792.5521.8243.03
LLM DDx + RAG 17.9039.44 92.5518.18 42.02
GuideSkill-Zero 23.60 44.72 92.55 27.27 48.08
GuideSkill-Evo 25.95+28.15% 48.89+23.96% 93.62+1.16% 30.91+13.35% 49.84+15.83%
Qwen3.5 9BDirect 19.69 39.44 92.55 21.82 43.38
CoT 19.57 36.1193.6223.64 43.24
3-shot ICL20.69 39.9492.55 23.6444.21
RAG 16.55 33.33 90.43 23.64 40.99
LLM DDx 17.67 33.3393.6220.00 41.16
LLM DDx + RAG 19.13 35.28 92.55 21.82 42.20
GuideSkill-Zero 23.71 41.67 94.68 25.45 46.38
GuideSkill-Evo 26.17+26.49% 46.39+16.15% 96.81+3.41% 34.55+46.15% 50.98+15.31%
Table2:Mainresultsacrossfourbenchmarksandfourbasemodels,reportedasaccuracy(%).Foreachbasemodelandcolumn,
boldindicates the best score, anditalicindicates the strongest baseline. Green numbers show the relative improvement of
GuideSkill-Evoover the strongest baseline.GuideSkill-Evoachieves the best accuracy across all benchmarks and base models,
with the largest gains over the strongest baselines andGuideSkill-Zero.
principal supervision sources, optimization strategies, and
guideline representations relevant to our setting. To ensure
comparability,weimplementthemusingthesamebackbone,
datasplits,evaluationprotocol,andcuratedguidelinecorpus
whenever applicable, rather than comparing with published
results obtained on different tasks. The cited methods there-
foremotivatematchedbaselineadaptationsratherthanexact
reproductions. For case-based training, we reserve 20% of
the training split for validation and use the remainder for
optimization.
Evaluation.We report accuracy on each benchmark and
the macro-average across the four benchmarks. Gold diag-
noses are normalized to three-character WHO ICD-10 cat-
egories during preprocessing, and cases without a reliable
mapping to a single diagnostic category are excluded. Be-
causeallmethodsareinstructedtoreturnanICD-10category
code,weuseexactequalitybetweenthepredictedandrefer-
ence codes as the primary correctness criterion; malformed
or code-free outputs are counted as incorrect. Appendix H
provides the model and inference settings, baseline imple-mentations,trainingconfigurations,andcompleteevaluation
protocol. Appendix I describes ICD-10 normalization, data
filtering, and benchmark splits.
5 Results
MainResultsTable2reportsresultsacrossfourbackbone
models.GuideSkill-Evoachieves the highest macro-average
accuracy for every backbone: 56.97 withGPT-5.4, 56.64
withClaude-Sonnet-4.6, 49.84 withMedGemma-27B, and
50.98 withQwen3.5-9B. ForGPT-5.4andClaude-Sonnet-
4.6,thesescoresexceedthestrongestnon-GuideSkillbaseline
by 6.98 and 8.39 percentage points, respectively. The gains
therefore extend across proprietary, open-weight, general-
purpose, and medically specialized backbones, indicating
that the improvement is not tied to a particular LLM.
Evenwithoutcase-basedevolution,GuideSkill-Zerooften
ranks second within each backbone block and outperforms
guideline RAG for every backbone. This is notable because
itsguideline-derivedlibrarycoversonlyasubsetoftheeval-
uated diagnoses; candidates without a matching skill rely
primarily on the LLM ranking. Its competitive performance
5

0255075100accuracy (%)Existing skills New skills
+2.1-0.7+1.1
+7.1
+12.4+18.7
+22.2Claude-Sonnet-4.6
Zero
EvoExisting skills New skills
+1.8-1.4+1.1
-3.6
+6.9+14.7
+7.4GPT-5.4
MedCase ER-ReasonMIMICMedThink MedCase ER-Reason MedThink0255075100accuracy (%)Existing skills New skills
-1.0-1.1+0.0
+0.0
+4.9+4.0
+7.4MedGemma-27B
MedCase ER-ReasonMIMICMedThink MedCase ER-Reason MedThinkExisting skills New skills
+1.3+1.8+1.1
+3.6
+3.3+16.0
+14.8Qwen3.5-9BFigure 3: Accuracy ofGuideSkill-ZeroandGuideSkill-Evoon initially covered and newly added ICD-10 categories. Evolution
improves all new-skill settings while preserving or improving existing-skill performance in 11 of 16 settings.
Method MedCaseReasoning ER-Reason MIMIC-CDM-FI MedThink-Bench Average
Qwen3.5 9B
Direct Inference 19.69 39.44 92.55 21.82 43.38
Fine-tuning w/ Guidelines (Staniek, Sokolov, and Riezler 2025) 23.94 43.33 92.55 18.18 44.50
Fine-tuning w/ Cases (Wu et al. 2025) 23.2746.9492.55 18.18 45.24
Fine-tuning w/ Guidelines + Cases 24.50 45.2893.6216.36 44.94
RL w/ Cases (Chen et al. 2024)25.9542.50 92.55 16.36 44.34
Fine-tuning w/ Guidelines + RL w/ Cases 24.94 43.0693.6221.8245.86
Guidelines as Decision Trees (Deng et al. 2026) 20.58 29.17 56.3832.7334.72
GuideSkill-Zero (Ours) 23.71 41.67 94.68 25.45 46.38
GuideSkill-Evo (Ours) 26.17 46.39 96.81 34.55 50.98
Table 3: Training results on Qwen3.5-9B across four benchmarks, reported as accuracy (%). For each column,boldindicates
the best score.GuideSkill-Evoachieves the best performance across all benchmarks and attains the highest average accuracy,
consistently outperforming fine-tuning and RL-based training variants.
shows that guideline-derived executable skills provide sub-
stantial diagnostic value before using benchmark training
cases. Rather than creating this benefit from scratch, evo-
lution builds on it by expanding the library from 349 to
473 ICD-10 categories and increasing gold-label skill cov-
erage from 56.5% to 99.5%. The resultingGuideSkill-Evo
improves overGuideSkill-Zeroon MedCaseReasoning, ER-
Reason,andMedThink-Bench.OnMedThink-Bench,which
is excluded from evolution, it outperformsGuideSkill-Zero
across all four backbones, suggesting that case-derived up-
dates can transfer to an unseen benchmark.
The advantage over guideline RAG further highlights
thevalueofoperationalizingguidelineknowledge.Whereas
RAGappendsretrievedpassagesandreliesontheLLMtoin-
terpret them,GuideSkillexecutes disease-specific criteria to
produce candidate-level support scores. To examine robust-
ness, efficiency, and mechanism, additional analyses show
thatα= 0.5isacompetitivefusionsettingandthataccuracy
largelysaturatesatK= 5(Figures4and5).Batchedfeature
groundinginGuideSkill-EffireducesestimatedGPT-5.4API
cost by 72.7% with a 0.34-point accuracy decrease on Med-
CaseReasoning (Appendix F; Table 6). A case study further
illustrates how candidate-specific skill scores can correct an
initially incorrect LLM ranking (Appendix K).Skill EvolutionWe analyze evolution from two comple-
mentary perspectives: skill coverage and downstream di-
agnostic utility. Gold-label skill coverage is the percent-
age of test cases whose reference ICD-10 category has a
corresponding executable skill. As shown in Table 1, evo-
lution expands the library from 349 to 473 ICD-10 cate-
gories and increases coverage from 56.5% withGuideSkill-
Zeroto 99.5% withGuideSkill-Evo, filling nearly all gaps
in the initial guideline-derived library. Figure 3 examines
whether this expansion improves newly covered diagnoses
withoutdegradingtheoriginalskills.Fornewlycoveredcat-
egories,GuideSkill-Evoimprovesaccuracyinall12available
backbone–benchmark comparisons, with gains of 3.3–22.2
percentage points. For initially covered categories, accuracy
improves or remains unchanged in 11 of 16 comparisons;
gains reach 7.1 points, whereas declines are limited to 3.6
points.Evolutionthereforesubstantiallyimprovesnewlycov-
ered diagnoses while preserving or strengthening the initial
skillsinmostsettings.Toidentifytheremainingbottlenecks,
weconductanerroranalysiswithClaude-Sonnet-4.6.Ofthe
749errors,456(60.9%)occurbecausethereferenceICD-10
categoryisabsentfromthecandidatesetandthereforecannot
berecoveredbydownstreamskillexecution.Candidaterecall
is thus the dominant remaining failure mode; Appendix G
6

Dataset Executable Skill Textual Skill
Claude-Sonnet-4.6
MedCaseReasoning 45.00±0.0044.60±0.80
ER-Reason 63.00±0.0061.40±0.80
MIMIC-CDM-FI 95.74±0.0096.38±0.52
MedThink-Bench 40.00±0.0040.00±0.00
Average 60.94±0.0060.60±0.13
Table 4: Comparison between executable and textual skill
representations using the sameGuideSkill-Evocandidate
sets. Executable skills provide deterministic tier scoring af-
terfeaturegrounding,whiletextualskillsrelyondirectLLM
interpretation of plain-text rubrics.
and Table 7 provide the complete error breakdown.
Comparison with Training- and Guideline-Based Base-
linesTable 3 comparesGuideSkillwith parameter-
update and structured-guideline baselines using the same
Qwen3.5-9Bbackbone.GuideSkill-Evoachievesthehighest
macro-average accuracy of 50.98, exceeding the strongest
parameter-update baseline by 5.12 percentage points and
GuidelinesasDecisionTreesby16.26points.Itranksfirston
three of four benchmarks; the only exception is ER-Reason,
where case fine-tuning exceeds it by 0.55 points. Notably,
GuideSkill-Zeroalready achieves a macro-average of 46.38,
surpassing the strongest parameter-update baseline before
case-based skill evolution. On MedThink-Bench, which is
excludedfromskillevolutionandcase-basedparametertrain-
ing,GuideSkill-Evoachieves34.55,comparedwith21.82for
thestrongestparameter-updatebaselineand32.73forGuide-
linesasDecisionTrees.Thisindicatesstrongertransfertoan
unseen benchmark. By storing diagnostic knowledge in ex-
ternalfunctionsthatdirectlyexecutedisease-specificcriteria,
GuideSkillimprovesaccuracywithoutupdatingthebackbone
while keeping its decision logic explicit.
ComparisonBetweenExecutableandTextualSkillsUn-
der identicalGuideSkill-Evocandidate sets and fusion pa-
rameters, executable and textual skills achieve similar mean
accuracy (60.94 versus 60.60), but only the executable vari-
antshowsnorun-to-runaccuracyvariationoverfiveevalua-
tions,providingamorerepeatablescoringinterface(Table4;
Appendix C).
ClinicianAssessmentofSkillQualityOneclinicianeval-
uated ten disease skills, five guideline-initialized and five
case-derived, covering 42 initial rules, 100 evolution cases,
120 final rules, and 51 held-out cases. For each skill, the
clinician first assigned an expected support tier (0–3) and
confidence score to held-out cases without seeing the skill
output,enablingcomparisonwiththemethod-assignedtiers.
For guideline-initialized skills, the clinician then assessed
whether each initial rule was consistent with its cited guide-
line passage and whether clinically important criteria were
omitted or incorrectly encoded. Next, the clinician rated
whether each evolution case was suitable for skill evolu-
tion and whether it supported a new or revised rule. Finally,Measure Guideline Case-derived Overall
Initial rule and skill validity
Guideline-consistent rules 95.2% – –
Skills without major clinical errors 5/5 – –
Evolution case utility
Cases suitable for skill evolution 90.0% 80.0% 85.0%
supporting a new or revised rule 60.0% 70.0% 64.7%
Blinded held-out case assessment
Correct diagnoses assigned tier≥2 12/12 11/11 23/23
Agreement within one tier 88.0% 84.6% 86.3%
Exact tier agreement 52.0% 34.6% 43.1%
Quadratic-weightedκ0.64 0.55 0.60
Final skill acceptability
Skills accepted unchanged/minor 4/5 5/5 9/10
Table 5: Clinician assessment of guideline-initialized and
case-derivedskills.Tier-agreementmetricscomparetheclin-
ician’s blinded judgments with the corresponding skill out-
puts.
after reviewing the rule changes and final skill, the clini-
cianrecommendedaccepting,revising,orrejectingtheskill.
Thisstagedprotocolreducesanchoringfromtheskilloutput;
Appendix J provides the complete questionnaire.
As shown in Table 5, 95.2% of the initial rules were
guideline-consistent, and all five guideline-initialized skills
were judged to have no major clinical errors. Among evolu-
tion cases, 85.0% were rated as suitable for skill evolution,
with 64.7% supporting a new or revised rule. On blinded
held-out cases, all correct diagnoses received a support tier
of at least 2, and 86.3% of skill outputs were within one tier
of the clinician judgment. The overall quadratic-weightedκ
was0.60,indicatingmoderateclinician–skillagreement.Fi-
nally,9/10finalskillswereacceptedunchangedorwithonly
minorrevisions.Adetailedcasereviewshowedthattheonly
finalskillrequiringmajorrevisionresultedfromanevolution
case that changed an originally correct rule into an inappro-
priate one by overgeneralizing a case-specific pattern rather
thancapturingageneralizablediagnosticrule.Overall,these
results suggest thatGuideSkillproduces clinically validated
and largely acceptable skills through both skill initialization
and evolution, while highlighting the need for careful vali-
dation of case-derived updates.
6 Conclusion
We introducedGuideSkill, an external reasoning layer that
compiles guideline criteria into executable disease-specific
skills and evolves them from labeled cases without updat-
ing the backbone. Across four benchmarks and four LLMs,
guideline-initializedskillsoutperformguidelineRAG,while
evolution improves all 16 direct-inference comparisons and
raises gold-label skill coverage from 56.5% to 99.5%.
OnQwen3.5-9B,GuideSkill-Evoalso exceeds the strongest
matched parameter-update baseline. These results show that
clinical knowledge can be maintained as a reusable, in-
spectable, and extensible execution layer across LLM back-
bones;broaderclinicianvalidationremainsnecessarybefore
clinical use.
7

References
Anthropic. 2026. Claude Sonnet 4.6. Accessed July 21,
2026.
Brown,T.B.;Mann,B.;Ryder,N.;Subbiah,M.;Kaplan,J.;
Dhariwal,P.;Neelakantan,A.;Shyam,P.;Sastry,G.;Askell,
A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan,
T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter,
C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.;
Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford,
A.; Sutskever, I.; and Amodei, D. 2020. Language Models
are Few-Shot Learners. InAdvances in Neural Information
Processing Systems, volume 33, 1877–1901.
Cao, L.; Chen, Q.; and Guo, Y. 2026. EHR-RAG: Bridg-
ingLong-HorizonStructuredElectronicHealthRecordsand
LargeLanguageModelsviaEnhancedRetrieval-Augmented
Generation.arXiv preprint arXiv:2601.21340.
Chen,J.;Cai,Z.;Ji,K.;Wang,X.;Liu,W.;Wang,R.;Hou,J.;
andWang,B.2024.Huatuogpt-o1,towardsmedicalcomplex
reasoning with llms.arXiv preprint arXiv:2412.18925.
Chen, Z.; Cano, A. H.; Romanou, A.; Bonnet, A.; Matoba,
K.;Salvi,F.;Pagliardini,M.;Fan,S.;Köpf,A.;Mohtashami,
A.; et al. 2023. Meditron-70b: Scaling medical pretraining
forlargelanguagemodels.arXivpreprintarXiv:2311.16079.
Deng,R.;Martin,G.;Wang,T.;Zhang,G.;Liu,Y.;Weng,C.;
Wang, Y.; Rousseau, J. F.; and Peng, Y. 2026. CPGPrompt:
Translating Clinical Guidelines into LLM-Executable Deci-
sion Support.arXiv preprint arXiv:2601.03475.
Google. 2026. MedGemma 27B Instruction-Tuned Model
Card. Accessed July 21, 2026.
Hager, P.; Jungmann, F.; Holland, R.; Bhagat, K.; Hubrecht,
I.; Knauer, M.; Vielhauer, J.; Makowski, M.; Braren, R.;
Kaissis, G.; and Rueckert, D. 2024. Evaluation and mitiga-
tion of the limitations of large language models in clinical
decision-making.Nature medicine, 30(9): 2613–2622.
Hu,E.J.;Shen,Y.;Wallis,P.;Allen-Zhu,Z.;Li,Y.;Wang,S.;
Wang,L.;andChen,W.2022. LoRA:Low-RankAdaptation
of Large Language Models. InInternational Conference on
Learning Representations.
InstituteofMedicine.2011.ClinicalPracticeGuidelinesWe
CanTrust. Washington,DC:TheNationalAcademiesPress.
Jin, R.; Xu, M.; Meng, F.; Wan, G.; Cai, Q.; Jiang, Y.; Han,
J.; Chen, Y.; Lu, W.; Wang, M.; Lan, Z.; Jiang, Y.; Liu, J.;
Wang,D.;Cong,L.;andZhang,Z.2025. STELLA:Towards
a Biomedical World Model with Self-Evolving Multimodal
Agents.bioRxiv.
Kwon,W.;Li,Z.;Zhuang,S.;Sheng,Y.;Zheng,L.;Yu,C.H.;
Gonzalez, J. E.; Zhang, H.; and Stoica, I. 2023. Efficient
Memory Management for Large Language Model Serving
withPagedAttention. InProceedingsofthe29thSymposium
on Operating Systems Principles, 611–626.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;K"uttler,H.;Lewis,M.;Yih,W.-t.;Rockt"aschel,
T.;Riedel,S.;andKiela,D.2020.Retrieval-AugmentedGen-
erationforKnowledge-IntensiveNLPTasks. InAdvancesin
Neural Information Processing Systems, volume 33, 9459–
9474.Li,B.;Meng,T.;Shi,X.;Zhai,J.;andRuan,T.2023.Meddm:
Llm-executable clinical guidance tree for clinical decision-
making.arXiv preprint arXiv:2312.02441.
Loshchilov,I.;andHutter,F.2019.DecoupledWeightDecay
Regularization. InInternational Conference on Learning
Representations.
Ma, Z.; Yang, S.; Ji, Y.; Wang, X.; Wang, Y.; Hu, Y.;
Huang, T.; and Chu, X. 2026. SkillClaw: Let Skills
Evolve Collectively with Agentic Evolver.arXiv preprint
arXiv:2604.08377.
McDuff, D.; Schaekermann, M.; Tu, T.; Palepu, A.; Wang,
A.;Garrison,J.;Singhal,K.;Sharma,Y.;Azizi,S.;Kulkarni,
K.; et al. 2025. Towards accurate differential diagnosis with
large language models.Nature, 642: 451–457.
Mehandru,N.;Golchini,N.;Garg,N.;LeSaint,K.T.;Nash,
C. J.; Ramachandran, A.; Zack, T.; McCoy, L. G.; Rodman,
A.;Bamman,D.;Molina,M.;andAlaa,A.2025. Er-reason:
A benchmark dataset for llm-based clinical reasoning in the
emergency room.arXiv preprint arXiv:2505.22919.
Ni,J.;Liu,Y.;Liu,X.;Sun,Y.;Zhou,M.;Cheng,P.;Wang,
D.;Zhao,E.;Jiang,X.;andJiang,G.2026. Trace2Skill:Dis-
tillTrajectory-LocalLessonsintoTransferableAgentSkills.
arXiv preprint arXiv:2603.25158.
Oniani, D.; Wu, X.; Visweswaran, S.; Kapoor, S.; Koora-
gayalu, S.; Polanska, K.; and Wang, Y. 2024. Enhancing
large language models for clinical decision support by in-
corporating clinical practice guidelines. In2024 IEEE 12th
InternationalConferenceonHealthcareInformatics(ICHI),
694–702. IEEE.
OpenAI. 2026a. GPT-5.4 Model Documentation. OpenAI
API documentation. Accessed July 21, 2026.
OpenAI. 2026b. text-embedding-3-small Model Documen-
tation. OpenAIAPIdocumentation. AccessedJuly21,2026.
QwenTeam.2026. Qwen3.5-9BModelCard. AccessedJuly
21, 2026.
Schubert,M.C.;Soyka,S.;Wick,W.;andVenkataramani,V.
2025. Guideline-incorporated large language model-driven
evaluation of medical records using MedCheckLLM.JMIR
Formative Research, 9: e53335.
Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.;
Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D.
2024. DeepSeekMath: Pushing the Limits of Mathemati-
cal Reasoning in Open Language Models.arXiv preprint
arXiv:2402.03300.
Shen, S.; Cheng, W.; Ma, M.; Turcan, A.; Zhang, M. J.; and
Ma, J. 2026a. SKILLFOUNDRY: Building Self-Evolving
Agent Skill Libraries from Heterogeneous Scientific Re-
sources.arXiv preprint arXiv:2604.03964.
Shen, Y.; Cao, L.; Du, S.; Wang, Y.; Zhou, J.; Peng, H.; and
Guo, Y. 2026b. MedGuideX: Internalizing Decision Logic
fromExecutableGuidelinesintoLargeLanguageModelsfor
Clinical Reasoning.arXiv preprint arXiv:2605.26567.
Staniek,M.;Sokolov,A.;andRiezler,S.2025. Trainingand
EvaluationofGuideline-BasedMedicalReasoninginLLMs.
arXiv preprint arXiv:2512.03838.
8

Tziakouri, A.; and Menolascina, F. 2025. Reinforcement
Learning for Clinical Reasoning: Aligning LLMs with
ACR Imaging Appropriateness Criteria.arXiv preprint
arXiv:2510.05194.
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.;
Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chain-
of-ThoughtPromptingElicits ReasoninginLargeLanguage
Models. InAdvancesinNeuralInformationProcessingSys-
tems, volume 35, 24824–24837.
World Health Organization. 2019. International Statistical
ClassificationofDiseasesandRelatedHealthProblems,10th
Revision. Version 2019.
Wu, K.; Wu, E.; Thapa, R.; Wei, K.; Zhang, A.; Suresh, A.;
Tao, J. J.; Sun, M. W.; Lozano, A.; and Zou, J. 2025. Med-
casereasoning:Evaluatingandlearningdiagnosticreasoning
fromclinicalcasereports.arXivpreprintarXiv:2505.11733.
Xu, G.; Tang, N.; Li, X.; Li, T. J.-J.; Zheng, Z.; Jin, W.;
and Shi, Y. 2026. An Empirical Study of Agent Skills for
Healthcare: Practice, Gaps, and Governance.arXivpreprint
arXiv:2605.02709.
You,Z.;Chen,X.;Vashishtha,A.;Du,S.;Erion-Barner,G.;
Mei, H.; Peng, H.; and Guo, Y. 2026. Improving clinical
diagnosis with counterfactual multi-agent reasoning.arXiv
preprint arXiv:2603.27820.
Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.;
Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.;
Gonzalez, J. E.; and Stoica, I. 2023. Judging LLM-as-a-
Judge with MT-Bench and Chatbot Arena.arXiv preprint
arXiv:2306.05685.
Zhou, S.; Xie, W.; Li, J.; Zhan, Z.; Song, M.; Yang, H.;
Espinoza,C.;Welton,L.;Mai,X.;Jin,Y.;Xu,Z.;Chung,Y.-
H.; Xing, Y.; Tsai, M.-H.; Schaffer, E.; Shi, Y.; Liu, N.; Liu,
Z.; and Zhang, R. 2025. Automating expert-level medical
reasoning evaluation of large language models.npj Digital
Medicine.
9

A Limitations
AlthoughGuideSkillsubstantially improves clinical reason-
ing accuracy by integrating executable clinical skills with
LLM-based reasoning, it still has several limitations. First,
the current skill library is built from a limited set of clin-
ical guidelines. Many high-quality guidelines remain to be
collected, curated, and incorporated, which may further im-
provethecoverageandreliabilityoftheskilllibrary.Second,
for scalability, the current framework performs skill initial-
izationandskillevolutioninafullyautomatedmannerusing
LLMs. While this design makesGuideSkilleasy to scale,
involvingphysiciansintheverificationprocesscouldfurther
improve the quality and clinical validity of the generated
skills, though at the cost of additional time and annotation
effort. Finally,GuideSkillis still a research prototype and is
not intended for direct deployment in real clinical settings.
Our goal is to provide an initial step toward combining exe-
cutable clinical knowledge with LLM-based reasoning, and
further clinical validation is required before practical use.
B Preprocessing of Clinical Practice
Guidelines
Our preprocessing pipeline serves two goals: (A) curating a
corpusofusableclinicalpracticeguidelinesfromalargeand
heterogeneoussource,and(B)compilingthisfreetextintoa
library ofexecutabledisease-level diagnostic skills.
B.1 Guideline Corpus Curation
We start from the publicly availableepfl-llm/guidelinescor-
pus(Chenetal.2023),whichiscommonlyusedasapretrain-
ing corpus for medical LLMs. The corpus contains37,970
clinical practice guideline (CPG) documents drawn from
ninepublicsources.However,becausethesedocumentswere
scraped from online sources, the raw collection is highly
noisy: some documents have missing or empty body text,
someareextremelyshortorexcessivelylong,andmanycon-
tain content that is not directly relevant to actionable clini-
cal guidance. In addition, the corpus varies substantially in
length,rangingfrom5toover255,000words,withamedian
of618words. We therefore apply a four-stage cascade filter
to curate a usable guideline corpus.
Source filtering.We first retain only seven authoritative
clinical guideline sources: NICE, PubMed, CMA, CDC,
SPOR, WHO, and CCO. We discard the crowd-sourced and
quality-inconsistentwikidocsubset (33,058documents), as
well as the length-extremeicrcsubset (49documents). This
reduces the corpus from37,970to4,863documents.
Length filtering.We remove documents with missing
body text or fewer than100words, yielding4,675docu-
ments.
Usability filtering.Even authoritative sources contain ta-
bles of contents, reference lists, methodology sections, dis-
claimers, announcements, and abstracts without actionable
guidance.Toremovesuchdocuments,weuseanLLM-as-a-
judge,Claude-Sonnet-4.6, to determine whether each docu-
ment,truncatedtoitsfirst12,000characters,containsaction-
ableclinicalguidance,definedasconcreterecommendationson diagnosis, treatment, screening, management, dosing, or
eligibility.Onlydocumentsjudgedusableareretained,leav-
ing4,145documents.
Length-outliertruncation.Finally,weremovethelongest
5%of documents using a p95word-count cutoff. This pre-
ventsoverlongdocumentsfromdominatingdownstreamcon-
text and diluting relevant content. The final curated corpus
contains3,938clinical practice guidelines, with a median
length of4,030words, a mean length of5,046words, and a
range of101–18,662words. By source, it comprises1,585
NICE,1,236PubMed,377CMA,374CDC,184SPOR,108
WHO, and74CCO documents.
B.2 From Guidelines to Executable Diagnostic
Skills
Free-textguidelinescannotbeexecuteddirectly.Wetherefore
compile the curated corpus into a disease-indexed library of
executable diagnostic skills through three steps.
Recommendation extraction.UsingClaude-Opus-4.8,
we extractdisease-diagnosisrecommendations from each
guideline, namely rules of the form clinical findings or di-
agnostic criteria→confirm or rule out a specific disease
X. Here, a recommendation refers to an actionable diag-
nostic rule that links observable patient evidence, such as
symptoms,signs,laboratoryresults,imagingfindings,ordi-
agnostic criteria, to a disease-level conclusion. To ensure
executability,weimposethreestrictconstraints.First,were-
tain only rules whose output is a disease-level conclusion,
explicitlyexcludingtest-appropriatenessstatements,suchas
whether to order a scan, even when they mention diagno-
sis or staging. Second, each recommendation must map to
athree-characterICD-10category,suchasK35;recommen-
dations that do not map to a valid disease category, such
as non-disease findings codable only in the Z chapter, are
discarded. Third, each recommendation is annotated with a
diseasename,itsICD-10category,andanis-usflagindi-
catingwhetherthecontentisspecifictoUnitedStatesclinical
practice.Thisstepyields1,200diagnosticrecommendations
spanning349unique ICD-10 disease categories, of which
193are marked as US-specific.
Merging by disease.Because the same disease is of-
ten covered by multiple guidelines, we merge all recom-
mendations within each ICD-10 category into a single
self-contained, complementary, and de-duplicated diagnos-
ticstatement.Duringmerging,wereconcileUS-specificand
non-US content using theis-usflag. This produces one
merged diagnostic statement for each ICD-10 category, re-
sulting in349statements in total.
Skill synthesis and indexing.Each merged statement is
compiled into an executable Python diagnostic function,
orskill, that takes structured features extracted from a pa-
tient case as input and returns a diagnostic confidence tier
for the corresponding disease. The result is an ICD-10-
indexed library of349executable diagnostic skills. Overall,
the pipeline distills37,970heterogeneous documents into
3,938curatedguidelinesandfurthercompilestheminto349
disease-organized,deterministicallyexecutableskills,which
10

0.0 0.2 0.4 0.6 0.8 1.0
α (1 = LLM-only, 0 = skill-only)3032343638accuracy (%)
MedCaseReasoning
best α=0.5 (36.8%)
α=0.5 (default)
0.0 0.2 0.4 0.6 0.8 1.0
α (1 = LLM-only, 0 = skill-only)48495051525354accuracy (%)
ER-Reason
best α=0.5 (53.1%)
α=0.5 (default)
0.0 0.2 0.4 0.6 0.8 1.0
α (1 = LLM-only, 0 = skill-only)80859095100accuracy (%)
MIMIC-CDM-FI
best α=0.35 (95.7%)
α=0.5 (default)
0.0 0.2 0.4 0.6 0.8 1.0
α (1 = LLM-only, 0 = skill-only)1520253035accuracy (%)
MedThink-Bench
best α=0.0 (32.7%)
α=0.5 (default)Figure 4: Sensitivity analysis of the fusion weightαon four clinical reasoning benchmarks.α= 1corresponds to LLM-only
ranking, whileα= 0corresponds to skill-only scoring. The vertical dashed line marks the default settingα= 0.5. Across
datasets,GuideSkillremains relatively stable over a broad range ofα, showing that fusing LLM ranking with executable skill
scores provides robust diagnostic performance.
1 2 3 4 5 6 7 8 9 10
top-k candidates45505560657075accuracy (%)
recall@k (ceiling)
GuideSkill (fused, α=0.5)
Figure 5: Sensitivity analysis of candidate set size. We vary
thenumberofcandidatediagnosesKgeneratedbytheLLM
and report the average accuracy. The dotted curve shows
recall@K, which serves as an upper-bound ceiling. AsK
increases,thefinalaccuracyofGuideSkillquicklysaturates,
indicating that a moderate candidate set provides a practical
balancebetweencandidatecoverageandselectiondifficulty.
serveastheknowledgebasefortheretrievalandfusionstages
of our method.
C Details of the Textual-Skill Comparison
We compare executable and textual representations using
identicalGuideSkill-Evocandidate sets and the same fu-
sionrule.ExecutableskillsapplyPythonfunctionstoLLM-
groundedfeaturestoproducedeterministicsupporttiers(0–
3),whereastextualskillsasktheLLMtoassigntiersdirectly
from equivalent plain-text rubrics. We evaluate 349 cases
over five runs withClaude-Sonnet-4.6at temperature 0 and
report mean accuracy and run-to-run standard deviation.
D Sensitivity Analysis of Score Fusion
Figure 2 analyzes the sensitivity ofGuideSkillto the fusion
weightαacross the four benchmarks. Recall thatαcontrols
the relative contribution of the LLM ranking score and the
executable skill score:α= 1corresponds to relying only
on the LLM ranking, whileα= 0corresponds to relyingonly on the skill score. In our main experiments, we use the
default settingα= 0.5.
Overall, the fused scoring mechanism is robust across a
broadrangeofαvalues,butthebestsettingvariesslightlyby
benchmark.OnMedCaseReasoningandER-Reason,perfor-
mancepeaksaroundα= 0.5,indicatingthatboththeLLM’s
ranking prior and the skill-based evidence contribute useful
information. On MIMIC-CDM-FI, the best performance is
obtained aroundα= 0.35, suggesting that the executable
skills provide particularly strong diagnostic signal in this
more structured setting. On MedThink-Bench, performance
is highest nearα= 0, indicating that skill scores are more
reliable than the LLM ranking for these compact but chal-
lenging reasoning cases.
These results show that neither pure LLM ranking nor
pure skill scoring is uniformly optimal across benchmarks.
Instead,combiningthetwosourcesofevidenceyieldsstable
performance, withα= 0.5serving as a simple default that
performs competitively across datasets without benchmark-
specific tuning.
E Effect of Top-K Candidate Diagnoses
Figure 3 analyzes the effect of the candidate set sizeK
on diagnostic accuracy based onClaude-Sonnet-4.6. In our
main experiments, we setK= 5by default. As expected,
recall@Kconsistently increases as more candidate diag-
nosesareincluded,sincealargercandidatesetismorelikely
tocontaintheground-truthdiagnosis.However,thefinalac-
curacy ofGuideSkilldoes not always increase at the same
rate,becausethemodelmuststillselectthecorrectdiagnosis
from a larger set of plausible candidates. On MedCaseRea-
soning and ER-Reason,GuideSkillquickly reaches a stable
performance after a small number of candidates, indicat-
ing that most useful diagnostic evidence is already captured
in the top-ranked candidates. On MIMIC-CDM-FI, perfor-
mance improves with largerKand then saturates, closely
following the high recall ceiling. On MedThink-Bench, in-
creasingKbrings more noticeable gains, suggesting that
challenging reasoning cases benefit from a broader candi-
date set. Overall, this analysis shows thatK= 5provides
a good trade-off between candidate coverage and decision
11

Method Accuracy Input Tokens / Case Output Tokens / Case Cost / 10k Cases
GPT-5.4
CoT 25.17 332.87 1287.27 201.41
3-shot ICL 28.97 1178.87 1007.13 180.54
RAG 23.71 771.70 841.33 145.49
LLM DDx 25.28 775.10 1808.27 290.62
LLM DDx + RAG 25.39 1244.03 1988.57 329.39
GuideSkill (Ours) 39.71 9722.60 1106.30 409.01
GuideSkill-Effi (Ours) 39.37 3207.70 208.80 111.51
Table6:Accuracy,tokenusage,andestimatedinferencecostofdifferentpromptingandreasoningmethods.GuideSkillachieves
thehighestaccuracy,whileGuideSkill-Effimaintainsnearlythesameaccuracywithsubstantiallyfeweroutputtokensandmuch
lower inference cost.
complexity, and supports its use as the default setting.
F Efficiency Analysis
We evaluate the efficiency ofGuideSkillagainst baseline
methods to assess its potential for future deployment. The
originalGuideSkillframeworkperformsskill-levelevidence
extraction independently for each candidate skill, which en-
ables accurate and fine-grained reasoning but incurs rela-
tivelyhighinferencecost.Toimprovedeploymentefficiency,
GuideSkill-Effiperformsfeatureextractioninaunifiedpass,
allowing shared case-level evidence to be extracted once
and reused across skills. Specifically,GuideSkill-Effiis an
efficiency-oriented variant that reduces the number of LLM
calls per case from1 +Nto exactly two, whereNis the
numberofmatchedskills.InGuideSkill,inferenceconsistsof
one differential-diagnosis proposal followed by one feature-
extraction call for each matched skill. InGuideSkill-Effi, the
first call proposes the top-Kdifferential diagnosis as above,
while the second call extracts the required features forall
matched skills in a single pass. The model is instructed to
returnonlyfeaturekeysthatareexplicitlypresentinthecase,
withallotherfeaturesdefaultingtoabsent.Eachskillisthen
executed locally without further LLM calls, and fusion pro-
ceedsidenticallytoGuideSkill.Becausefeatureextractionis
a form-filling task rather than a reasoning task,GuideSkill-
Effiruns both calls without reasoning.
Thisdesignpreservesthemainbenefitofskill-guidedrea-
soningwhilegreatlyreducinggenerationoverhead.Weeval-
uatebothGuideSkillanditsefficientvariantagainstbaseline
methodsontheMedCaseReasoningbenchmark,whereboth
GuideSkillvariantsusetheevolvedskilllibrary.Asshownin
Table3,GuideSkill-Effiachieves39.37%accuracy,only0.34
points lower thanGuideSkill, while reducing the estimated
cost from 409.01 to 111.51 per 10k cases.
Importantly,GuideSkill-Effistillsubstantiallyoutperforms
allbaselinemethodsinaccuracy,whileachievingthelowest
inferencecostamongallcomparedmethods.Thisshowsthat
theefficientdesigndoesnotsimplytradeaccuracyforlower
cost,butprovidesabetteraccuracy-costbalance.Thisreduc-
tionisparticularlymeaningfulbecause,inmanycommercial
LLMAPIs,outputtokensarepricedhigherthaninputtokens.
GPT-5.4 follows this common pricing pattern: the official
API price is $2.50 per 1M input tokens and $15.00 per 1M
output tokens, so output tokens are 6×more expensive thanStage Clinical Type %
Candidate omission (60.9%)Coding granularity 74.8
Common diagnosis that should be listed 14.3
Rare or atypical diagnosis 11.0
Skill under-scoring (12.6%)Adjacent same-system diagnosis 40.4
Cross-system mimic 36.2
Causal chain 22.3
Distractor selection (26.6%)Adjacent same-system diagnosis 47.2
Causal chain 27.6
Cross-system mimic 24.6
Table 7: Error analysis ofGuideSkill-EvowithClaude-
Sonnet-4.6across all test cases. Stage percentages indicate
each error stage’s share among all analyzed errors; clinical-
type percentages are computed within each stage.
input tokens1. As a result, inference cost is often dominated
by output length.GuideSkill-Effidirectly addresses this bot-
tleneck by transferring much of the reasoning process from
free-form LLM generation to structured executable skills,
yielding much shorter outputs with little accuracy degrada-
tion.
G Error Analysis ofGuideSkill-Evo
AlthoughGuideSkill-Evoachieves the best overall perfor-
mance, it still leaves substantial room for improvement. We
conduct an error analysis on all test cases usingClaude-
Sonnet-4.6as the backbone and categorize the remaining
errorsintothreestages.Theanalysisisjudgeandcateogirze
also byClaude-Sonnet-4.6 Recall-misserrors occur when
the gold diagnosis is not included in the LLM-generated
candidateset.Skill-underscoringerrorsoccurwhenthegold
diagnosisisrecalledbutitsexecutableskillassignsaninsuf-
ficient score.Lost-to-distractorerrors occur when the gold
diagnosis is recalled and scored, but another plausible can-
didate receives a higher fused score.
As shown in Table 6, the largest source of error is recall
failure:in456cases,thegoldICD-10categoryisnotincluded
inthecandidateset,makingitimpossibleforskillexecution
torecoverthecorrectanswer.Mostrecall-misserrorsaredue
to coding granularity, where the LLM proposes a clinically
relateddiagnosisbutnottheexactICD-10categoryrequired
by evaluation. Among cases where the gold diagnosis is re-
called, errors often arise from fine-grained clinical distinc-
1https://developers.openai.com/api/docs/models/gpt-5.4
12

tions. Skill-underscoring errors are dominated by adjacent
same-systemdiagnosesandcross-systemmimics,suggesting
that some skills still underweight discriminative evidence.
Lost-to-distractor errors show a similar pattern: the correct
diagnosis is present, but a nearby diagnosis, causal down-
stream condition, or cross-system mimic receives a stronger
fused score. These results suggest that future improvements
should target both candidate recall at the ICD-10 category
levelandfiner-grainedskillcalibrationamongclinicallysim-
ilar diagnoses.
H Experiment Details
Models.We evaluate all methods using four backbone
LLMs spanning both proprietary and open-weight mod-
els. The proprietary models are accessed through Microsoft
Azure2:Claude-Sonnet-4.6(claude-sonnet-4-6)(An-
thropic 2026) and GPT-5.4 (gpt-5.4) (OpenAI
2026a). The open-weight models are served locally
with vLLM (Kwon et al. 2023): MedGemma-27B
(google/medgemma-27b-it) (Google 2026) and
Qwen3.5-9B (Qwen/Qwen3.5-9B) (Qwen Team 2026).
We use Claude-Sonnet-4.6 for skill initialization and skill
evolution inGuideSkill. For experiments involving non-
public clinical data from PhysioNet, we follow the Phys-
ioNetresponsible-useguidanceforMIMICdatawithLLMs3.
Specifically,restrictedclinicaldataareusedwithproprietary
models only through an institutionally approved Azure de-
ployment that ensures zero data retention, no training on
submitted data, and no human review of prompts or out-
puts.Thelocallyservedopen-weightmodelsprovideafully
controlled deployment path.
LLM Inference.Proprietary models are queried through
theAzureAPI.ForClaudebackbones,wesettemperatureto
0and do not enable extended thinking. GPT-5.4 is queried
with the API default settings, including the default temper-
ature of1and the default reasoning configuration. Open-
weight models are served through vLLM with a32,768-
token context window. For Qwen3.5-9B, we set tempera-
ture to0and explicitly disable thinking mode by setting
enable_thinking=False, so that the final answer is
returned directly rather than embedded in a long reasoning
trace. For MedGemma-27B, we use temperature0; since it
is not a reasoning model, no thinking mode is used. Unless
otherwise noted, generation is capped at4,096tokens.
Skill Evolution.We evolve the skill library using a
merged training set from MedCaseReasoning, ER-Reason,
and MIMIC-CDM-FI, with no overlap with the test splits.
Each case is labeled with its three-character ICD-10 cate-
gory. The evolution pipeline mirrors skill initialization and
also usesclaude-sonnet-4-6. First, inrationale generation,
the model is given each training case and its gold diagno-
sis, and is asked to extract the key clinical variables and
explain why they support that diagnosis. Second, incase-
to-recommendation, cases are grouped by ICD-10 label and
2https://azure.microsoft.com/en-us
3https://physionet.org/news/post/llm-responsible-use/distilledintoasinglerecommendationthatcapturesonlycri-
teria recurring across cases, while discarding case-specific
incidentals.Whenamatchingguidelinerecommendational-
readyexists,itisusedasthebackboneandisneverweakened.
We process cases in batches of10using a map-reduce strat-
egy: the model first drafts recommendations for each batch
andtheniterativelyrefinesthem,preventingpromptoverflow
and reducing specificity loss. Third, inrecommendation-to-
skill, the resulting recommendations are compiled using the
same generator and tier contract as in skill initialization.
Skillevolutionyields473case-derivedskills.Aftermerging
them with the initialized skill library, we obtain the evolved
skill library covering473ICD-10 categories, including206
categories shared with the initialized library and267newly
added categories from training cases.
Baselines.We compareGuideSkillwith both prompting-
basedandtraining-basedbaselines,usingthesamebackbone
modelswheneverapplicable.Forprompting-basedbaselines,
we include: (i)Direct, which directly prompts the model to
producethefinaldiagnosis;(ii)CoT,whichperformschain-
of-thought diagnostic reasoning before producing the final
answer (Wei et al. 2022); (iii)3-shot ICL, which uses three
fixed in-domain demonstrations from each dataset’s train-
ing split (Brown et al. 2020). Since MedThink-Bench does
not provide a training split, we use demonstrations from
MedCaseReasoning;(iv)RAG,whichretrievesfromcurated
guidelinerecommendationsandprependsthetop-5passages
retrievedbytext-embedding-3-small(OpenAI2026b)ascon-
text before answering (Lewis et al. 2020); (v)LLM DDx, a
two-pass differential-diagnosis baseline that first proposes
the top-5candidate diagnoses and then selects one final an-
swer without using external knowledge; and (vi)LLM DDx
+ RAG, which extends the two-pass differential-diagnosis
baseline by conditioning the final selection on the same re-
trieved guideline context used in RAG.
For training-based baselines, we conduct comparisons on
Qwen-3.5and include: (vii)Fine-tuning w/ Guidelines,
which fine-tunes the model on all curated guideline recom-
mendations; (viii)Fine-tuning w/ Cases, which fine-tunes
the model on training cases with gold diagnoses; (ix)Fine-
tuningw/Guidelines+Cases,whichfine-tunesontheunion
ofguideline-derivedsupervisionandcase-levelsupervision;
(x)RL w/ Cases, which applies reinforcement learning on
trainingcasesusingdiagnosiscorrectnessastherewardsig-
nal; and (xi)Fine-tuning w/ Guidelines + RL w/ Cases,
which first fine-tunes the model with guideline supervision
and then further optimizes it with case-level reinforcement
learning. All training baselines are implemented with verl4
usingthedefaulttrainingconfiguration.Wealsoinclude(xii)
GuidelinesasDecisionTrees,whichconvertsguidelinerec-
ommendations into decision-tree-style reasoning structures
and uses them as explicit diagnostic guidance. We use the
officialimplementationreleasedbytheoriginalpaper(Deng
et al. 2026).
Details of Training Baselines.For all trainable variants,
weapplyLoRA(Huetal.2022)toalllinearlayerswithrank
4https://github.com/verl-project/verl
13

16 andα= 32. Training uses bfloat16 precision, gradient
checkpointing,AdamWoptimization(LoshchilovandHutter
2019), a learning rate of1×10−5, a constant learning-rate
schedulewith10warmupsteps,aglobalbatchsizeof512,a
maximumsequencelengthof8,192tokens,andthreetraining
epochs.Forcase-basedlearning,werandomlysplittheorig-
inaltrainingsetintotrainingandvalidationsubsetsusingan
8:2 ratio. Guideline SFT is performed on the clinical guide-
linecorpus,whereascaseSFTisconductedonstandardized
diagnosis cases using answer-only supervision. TheGuide-
lines+Casessettingperformsthesetwostagessequentially.
For RL, we adopt GRPO (Shao et al. 2024) with a learning
rateof5×10−6,twotrainingepochs,abatchsizeof32,and
24 rollouts per prompt. The maximum prompt and response
lengthsaresetto8,192and512tokens,respectively.Weuse
one policy-update epoch, a KL coefficient of 0.005, an en-
tropycoefficientof0,arollouttemperatureof1.0,andtop-p
of1.0.CheckpointsaresavedaftereverySFTepochandev-
ery10RLupdates,andthebestcheckpointisselectedbased
on validation performance. All experiments are conducted
on a single server equipped with eight NVIDIA RTX 5090
GPUs.
EvaluationProtocol.Weevaluatepredictionsatthethree-
character ICD-10 category level: a prediction is counted as
correct if it maps to the same ICD-10 category as the gold
diagnosis. Semantic matching between the predicted diag-
nosis and the gold label is performed using an LLM-as-a-
judge,claude-haiku-4-5,withtemperaturesetto0.Thejudge
treats synonyms, abbreviations, subtype-level differences,
and clinically equivalent paraphrases as matches when they
fall within the same ICD-10 category. LLM judges can ex-
hibitsystematicbiasesandreasoninglimitations(Zhengetal.
2023);wethereforeidentifyjudge-basedsemanticmatching
as a limitation of the evaluation. The full LLM-as-a-judge
prompt is shown below.
LLM Judge Prompt for ICD-10 Category Matching
Decide whether the predicted diagnosis and the
ground-truth diagnosis refer to the same disease at
the ICD-10 3-character category level (e.g., "K35").
Judge at the category granularity: they match if they
fall under the same ICD-10 3-character category,
even if the subtype or wording differs.
Predicted diagnosis: {pred}
Ground-truth diagnosis: {gold}
Please answer directly with "yes" or "no".
I Benchmark Details and Examples
The four benchmarks originate from different data sources
and represent case reports, emergency-department records,
structured clinical decision-making cases, and medical QA
vignettes. Training splits from MedCaseReasoning, ER-
Reason, and MIMIC-CDM-FI are used for skill evolution
and parameter-update baselines; the corresponding held-out
testsplitsandtheMedThink-Benchtestsetareusedforeval-
uation. The benchmark data are separate from the guideline
corpus used to initializeGuideSkill-Zero.•MedCaseReasoning(Wu et al. 2025): A long-form open
diagnostic reasoning benchmark based on open-access
clinical case reports from the New England Journal of
Medicine Clinicopathological Conferences (NEJM CPC).
It evaluates case-based differential diagnosis over broad
and long-tailed clinical conditions.
Example: MedCaseReasoning
Case: A 30-year-old man presented with several
months of pain, tenderness, and swelling on the
leftsideofhispalate.T2-weightedMRIshoweda
hyperintense lesion, and biopsy suggested a soft-
tissue tumor. Given the case, what is the final di-
agnosis? [case report excerpt; approximately 890
words]
Diagnosis: Other disorders of nerve roots and
plexuses. ICD-10: G54.
•ER-Reason(Mehandru et al. 2025): An emergency-
department diagnosis prediction benchmark derived from
realemergency-roompatientrecords.Itevaluateswhether
a model can infer the final diagnosis from noisy, time-
sensitive clinical presentations in the emergency setting.
Example: ER-Reason
Case: Age: 26; Sex: Female; Chief Complaint:
hematuria. The record includes a full emergency-
department note with pregnancy and delivery his-
tory,normalspontaneousvaginaldelivery,second-
degree perineal laceration repair, hematuria with
dysuria,lowerabdominalpain,andsexualhistory.
[long EHR excerpt; approximately 15,400 words]
Diagnosis: Tubulo-interstitial nephritis, not speci-
fied as acute or chronic. ICD-10: N12.
•MIMIC-CDM-FI(Hageretal.2024):Afull-information
open clinical decision-making benchmark derived from
MIMIC-IV, which is based on electronic health records
from Beth Israel Deaconess Medical Center. It evaluates
full-information clinical decision making.
Example: MIMIC-CDM-FI
Case:Patienthistorydescribes10hoursofabdomi-
nalpainthatbeganneartheumbilicusandmigrated
totherightlowerquadrant,withoneepisodeofdi-
arrhea and no nausea, vomiting, or fever. Physical
examination shows right-lower-quadrant tender-
ness, positive obturator sign, and negative Rovs-
ing sign. Laboratory tests include urea nitrogen,
sodium, potassium, lipase, creatinine, and other
values. [structured history, exam, and laboratory
excerpt; approximately 4,000 words]
Diagnosis: Acute appendicitis. ICD-10: K35.
•MedThink-Bench(Zhou et al. 2025): An expert-curated
medical reasoning benchmark constructed from ten pub-
14

licly available medical QA datasets. The benchmark fil-
ters for complex questions requiring multi-step reasoning
across ten medical domains. In our setting, we use it as
a compact but challenging open-ended diagnosis-oriented
reasoning benchmark.
Example: MedThink-Bench
Case: A 23-year-old female weightlifter presents
with neck and right shoulder pain. Osteopathic
structuralexaminationrevealsrestrictedmotionof
Sibson’sfascia,andtheclaviclesappearasymmet-
ric. What is the most likely diagnosis? [medical
QA excerpt; approximately 780 words]
Diagnosis: Other acquired deformities of muscu-
loskeletal system. ICD-10: M95.
Preprocessing and Data Split.To normalize answer la-
bels across benchmarks, we first map all gold diagnoses to
ICD-10 categories. Specifically, we useClaude-Sonnet-4.6
to convert each dataset’s original answer label into an ICD-
10 code. We discard examples for which no reliable ICD-10
mapping can be obtained. We also remove examples whose
goldanswerdoesnotcorrespondtoasinglediseasediagno-
sis, since our evaluation requires one normalized diagnostic
target per case.
After preprocessing, we randomly split MedCaseReason-
ing, ER-Reason, and MIMIC-CDM-FI into training and test
sets.MedThink-Benchdoesnotprovideatrainingsplitinour
setting, so we use it only for evaluation. The final processed
data contain 11,598 training and 894 test cases for Med-
CaseReasoning, 1,235 training and 360 test cases for ER-
Reason,219trainingand94testcasesforMIMIC-CDM-FI,
and 55 test cases for MedThink-Bench. In total, our exper-
iments use 13,052 training cases and 1,403 test cases after
ICD-10 normalization and filtering.
J Clinician Assessment Questionnaire
This appendix gives the complete instrument used for the
clinician assessment reported in Table 5, together with the
orderinwhichmaterialwaspresented.Thestudywasdeliv-
eredasasingle-pagewebapplication;eachpacketcoversone
disease skill, and the clinician worked through the screens
in a fixed order without being able to see later material in
advance.
J.1 Presentation Order
Eachpacketispresentedasfourscreens.Critically,theheld-
outscreencomesfirst:theclinicianassignsanexpectedsup-
porttiertounseencasesbeforeanypartoftheskill—guide-
line passages, rule tables, or code — is revealed. Presenting
the skill first would let its output anchor the expected tiers
and inflate agreement.
1.Held-out cases(Q5). Blinded tier assignment on cases
never used to build the skill. The skill’s own output stays
hidden throughout.
2.Guidelineandinitialskill(Q1,Q2).Numberedguideline
passagesG1–Gn, a human-readable rule tableR1–Rnlinkingeachruletoitssupportingpassage,andtheinitial
Python function in an expandable panel.
3.Evolution cases(Q3). Ten cases per skill, unlabeled and
in fixed order, with the final skill still hidden.
4.Initial-to-final update(Q4). The change log with per-
change provenance, the final rule table, and the final
Python function; the disposition is recorded last.
Forcase-derivedskills there is no guideline-initialized
predecessor, so Screen 2 and questions Q1–Q2 are omitted
(threescreens,Q3–Q5).Thesepacketsareexplicitlylabeled
“case-derived skill; no guideline-derived initialization.” In
both tracks, “cannot assess / outside expertise” is available
on every question and is excluded from the corresponding
denominator rather than treated as a negative rating.
J.2 Questions
Q1 — Guideline support (per rule; guideline-initialized
skillsonly).Foreachruleintheinitialskill,howwellisit
supported by the provided guideline passages?
•Fully supported
•Mostly supported (minor interpretation)
•Partially supported (substantial interpretation)
•Unsupported or contradictory
•Cannot assess
Table5reportstwothresholdsonthisscale:rulesratedfully
ormostlysupported,andrulesnotratedunsupportedorcon-
tradictory.
Q2 — Errors or omissions in the initial skill (per skill).
Basedonlyontheprovidedguidelinepassages,doestheini-
tialskillomitorincorrectlyencodeanyclinicallyimportant
criterion,threshold,exclusion,exception,orlogicalrelation-
ship?
•No clinically important problem
•Minor problem (unlikely to change the support tier)
•Major problem (could change the support tier)
•Potentially dangerous / seriously misleading
•Cannot assess
Q1 and Q2 are locked before the clinician proceeds to the
evolution cases.
Q3 — Contribution of an evolution case (per case).If a
rule for this skill were written from this case, what would it
do to future patients?
A Add or change a rule — shows a criterion the skill needs
and might otherwise miss
B Keep the rules as they are — textbook presentation; con-
firms what a skill would already check
C Wouldapplytoalmostnooneelse—arulefromthiscase
would rarely fire again; harmless but useless
D Would fire on the wrong patients — a rule from this case
would misjudge future patients (scores a mimic as this
disease, drops a needed requirement, or relies on a non-
specific feature)
E Cannot assess
15

ThedecidingtestbetweenCandDisstatedintheinstrument:
would a rule taken from this case ever fire on a patient who
does not have this disease? No→C; yes→D.An optional
free-text field records the finding or omission that drove the
rating.CasesratedA,B,orCarecountedasgeneralizablein
Table 5; D marks a case that should not influence the skill.
Q4 — Final update and disposition (per skill).After
reviewing the guideline, the cases, and the initial-to-final
diff, what is your recommendation for the final skill?
•Accept unchanged
•Accept with minor edits
•Major revision required
•Reject
•Cannot assess
If the recommendation is not “accept unchanged”, the clin-
ician selects all applicable concerns from: insufficient case
support; case- or dataset-specific pattern; conflicts with the
guideline;weakensaguideline-supportedrequirement;omits
animportantcriterion;treatsanunreportedfeatureasabsent;
treatsanunexcludedmimicasexcluded;incorrectthreshold,
unit,negation,orAND/ORlogic;inappropriatesupporttier;
other. A free-text field records the single most important
required change.
Q5 —Expected support tier(per held-outcase).Based
on the available evidence, what diagnostic-support tier
should a skill for this disease assign to this case?
•3 — Confirmed / highest support
•2 — Strongly suggestive
•1 — Compatible
•0 — Not supported
•Insufficient information to assign a tier
•Cannot assess
Aconfidencerating(low/moderate/high)accompanieseach
tier.Thesearethesamefourtierstheskillsthemselvesemit,
so the clinician’s blinded judgment and the skill’s executed
tier are directly comparable; agreement is computed only
after all held-out cases for a packet are submitted.
J.3 Held-Out Case Construction
Held-out cases are drawn from the MedCaseReasoning test
split and are disjoint from every case used to build the skill.
Each packet mixes two kinds, and the kind isnotshown to
the clinician:
•Correct diagnoses(23 cases): the reviewed category is
the gold diagnosis.
•Look-alikes(28 cases): the gold diagnosis is a differ-
ent category, but the reviewed category appeared in the
model’s proposed differential — an operational, repro-
ducible definition of clinical similarity.
Becausebothkindsarepresentedidentically,Q5measures
whethertheskill’stiertracksclinicaljudgmentoncasesthat
doanddonotwarrantsupport,ratherthanonlyonpositives.
16

K Case Study
We present a representative MedCaseReasoning example as
a case study ofGuideSkill. In this example,GuideSkillcor-
rects the base LLM’s plausible but incorrect top-ranked di-
agnosis. The patient is a 60-year-old woman with dyspep-
sia, unintentional weight loss, postprandial vomiting, gas-
tric ulcers on endoscopy, and biopsy showing large conflu-
ent non-caseating epithelioid granulomas. The base LLM
initially ranks Crohn’s disease as the most likely diagno-
sis,likelybecausegastrointestinalulcersandgranulomasare
common cues for Crohn’s disease. However, after executing
candidate-specific skills,GuideSkillassigns different diag-
nostic strengths to the same evidence. The Crohn’s disease
skill treats granulomas as a specific but insufficient feature
and stops at tier 2, while the sarcoidosis skill treats non-
caseating granulomas as confirmatory histologic evidence
and assigns tier 3. The fusion step therefore overturns the
LLM ranking and selects sarcoidosis, matching the gold di-
agnosis.
The key mechanism is that the two executed skills as-
signdifferentevidentialstatustothesamegroundedfinding.
Forsarcoidosis,non-caseatinggranulomaswithunsupported
mimics enter a direct tier-3 branch. For Crohn’s disease,
granulomas alone are only a specific tier-2 feature unless
additional Crohn’s-specific criteria or mimic-exclusion re-
quirements are satisfied. Thus, the same clinical evidence
is sufficient to confirm sarcoidosis but only suggestive for
Crohn’s disease, making the tier-based fusion interpretable.
17

Field Content
Case A 60-year-old woman with no medical comorbidities presented with a 2-month history of dyspepsia,
unintentionalweightloss,andanorexia.Shenotedpostprandialfullness,earlysatiety,andmultipleepisodes
of non-bilious, non-projectile vomiting occurring 20–30 minutes after meals, containing undigested food.
Shehadnofever,abdominalpain,gastrointestinalbleeding,orhistoryofmedicationuse.Onexamination,she
was dehydrated and tachycardic. Abdominal examination revealed a distended and tender upper abdomen;
the liver edge was palpable 2 cm below the right costal margin. Laboratory tests showed a low hemoglobin
level with otherwise normal routine studies. Upper endoscopy demonstrated an oval pre-pyloric ulcer with
erythematous, everted margins and a whitish exudate at the base, surrounded by normal mucosa, and an
irregular fundal ulcer with inverted margins and whitish exudate, surrounded by normal mucosa. Multiple
gastric biopsies revealed patchy chronic inflammation, mild crypt architectural disarray, and several large
confluent non-caseating epithelioid granulomas.
Ground Truth Diagnosis Sarcoidosis (D86)
LLM Top-5 Differential
Diagnosis(1) Crohn’s disease [regional enteritis] (K50)
(2) Sarcoidosis (D86)
(3) Malignant neoplasm of stomach (C16)
(4) Tuberculosis of other organs (A18)
(5) Gastric ulcer (K25)
Skill Execution Trace Crohn’s disease (K50) tier 2 / 0.667 Granulomas and GI ulcers are specific evidence,
but the case lacks decisive Crohn’s evidence such
as transmural or terminal ileal involvement, and
direct-confirmation branches require mimics to be
excluded.
Sarcoidosis (D86) tier 3 / 1.000 Non-caseatingepithelioidgranulomasprovidecon-
firmatory histologic evidence for sarcoidosis when
competing mimics are not supported.
Malignant neoplasm of stomach
(C16)tier 0 / 0.000 Endoscopic ulcers raise concern, but biopsy does
not report malignant cells.
Tuberculosis of other organs (A18) tier 1 / 0.333 Granulomas are compatible with tuberculosis, but
the non-caseating pattern and lack of TB-specific
evidence weakens this diagnosis.
Gastric ulcer (K25) tier 1 / 0.333 Gastric ulcers are present, but ulcer disease alone
does not explain the granulomatous pathology as
the final diagnosis.
GuideSkillPrediction Sarcoidosis (D86), selected after fusion. The tier-3 sarcoidosis evidence overrides the LLM’s top-ranked
Crohn’s disease prediction.
Table 8: Case study on MedCaseReasoning. The table includes the patient case, ground-truth diagnosis, the LLM’s top-5
differential diagnosis, and the executable skill trace used byGuideSkill.
18

Executed Skill: Sarcoidosis (D86)
1def score_sarcoidosis(case):
2"""id: 148 | icd10: D86 | disease: Sarcoidosis"""
3
4# DIRECT (tier 3): biopsy non−caseating granuloma + mimics excluded,
5# OR three−pillar clinical/radiographic/pathologic support,
6# OR cardiac/neuro imaging support with mimics excluded.
7direct_confirmed = (
8(biopsy_noncaseating_granuloma is True and mimics_excluded is True)
9or (three_pillars_met is True)
10or (cardiac_neuro_imaging_supported is True and mimics_excluded is True)
11)
12
13# CONTRADICTING evidence: caseating granuloma, positive AFB/fungal testing,
14# malignant cells, or another confirmed mimic.
15if direct_confirmed and not any_contradiction:
16tier = 3
17elif any_specific:
18tier = 1 if any_contradiction else 2
19elif any_generic:
20tier = 0 if any_contradiction else 1
21else:
22tier = 0
23
24return ("Sarcoidosis", "D86", tier / 3.0)
Executed Skill: Crohn’s Disease (K50)
1def score_crohns_disease(case):
2"""id: 313 | icd10: K50 | disease: Crohn’s disease [regional enteritis]"""
3
4# DIRECT (tier 3) requires Crohn’s−specific evidence and mimics_excluded.
5direct_findings = [
6(noncaseating_granulomas_with_giant_cells is True) and mimics_excluded,
7(segmental_skip_transmural_terminal_ileum is True) and mimics_excluded,
8(biopsy_proven_crohns is True) and penetrating_or_perianal and mimics_excluded,
9]
10
11# SPECIFIC (tier 2): granulomas, skip/transmural pattern, perianal disease,
12# or other characteristic Crohn’s features without full direct confirmation.
13if any(direct_findings):
14tier = 3
15elif any_specific:
16tier = 2
17elif any_generic:
18tier = 1
19else:
20tier = 0
21
22return ("Crohn’s disease [regional enteritis]", "K50", tier / 3.0)
19

L Prompt Design
The three stages ofGuideSkillinvolve multiple LLM calls,
withdifferentpromptsdesignedforpreprocessing,skillcon-
struction, and inference-time reasoning.
IntheSkillInitializationstage,fourpromptsareused:ICD-
10LabelNormalizationmapsfree-textdiagnosestoICD-10
categories;GuidelinetoRecommendationextractsatomicdi-
agnosticrecommendationsfromclinicalguidelines;Recom-
mendation Mergeconsolidates recommendations that map
to the same ICD-10 category; andRecommendation to Exe-
cutableSkillcompileseachmergedrecommendationintoan
executable diagnostic skill.
In theSkill Evolutionstage, three prompts are used:Ra-
tionale Generationderives diagnosis-supporting rationales
from training cases;Case-to-Recommendation Distillation
distillsrecurringdiagnosticcriteriaacrosscasesintodisease-
level recommendations; andCase-to-Recommendation Re-
finementupdates existing recommendations with new
batches of case evidence.
In theSkill Executionstage, three prompts are used:Dif-
ferentialDiagnosisProposalproposesarankedICD-10dif-
ferential diagnosis;Skill Feature Groundingmaps a patient
case into the input features required by a selected skill; and
Efficient Skill Feature Groundinggrounds features for mul-
tiple candidate skills in a single pass, which is used by the
efficient variant ofGuideSkill.
The full prompt templates are shown below. Placeholders
to be replaced at runtime are enclosed in curly braces ({ }).
20

Skill Initialization: ICD-10 Label Normalization
1# Task
2You map a free−text clinical diagnosis to a single standard ICD−10 disease CATEGORY.
3
4# Diagnosis
5{answer}
6
7# Rules
8−Map the diagnosis to its standard ICD−10 category and respond with ONLY a JSON object:
9{"answer": "<category title>", "icd10": "<CODE>"}.
10−"answer" is the official ICD−10 category title, written as a disease name with NO code in it.
11−"icd10" MUST be the 3−character ICD−10 CATEGORY code: one letter + two digits, NO decimal.
12Use "K35", never "K35.2".
13−If the diagnosis does NOT map to any valid ICD−10 disease category, respond with exactly:
14{"answer": null, "icd10": null}.
15−If the diagnosis refers to multiple diseases, symptoms only, procedures, treatments, social history,
16or non−disease concepts, respond with exactly:
17{"answer": null, "icd10": null}.
18
19# Examples
20{"answer": "Acute appendicitis", "icd10": "K35"}
21{"answer": "Pulmonary tuberculosis", "icd10": "A15"}
Skill Initialization: Guideline to Recommendation
1# Task
2Extract atomic diagnostic recommendations from the given clinical practice guideline.
3
4# Clinical Practice Guideline
5{guideline}
6
7# Rules
8−Extract only recommendations that support disease identification, diagnosis, differential diagnosis,
9or clinically meaningful diagnostic decision−making.
10−Each recommendation should describe a finding−to−disease or evidence−to−disease rule.
11−Keep concrete clinical criteria, thresholds, tests, signs, symptoms, imaging findings, laboratory values,
12pathology findings, risk factors, and exclusion criteria when they affect diagnosis.
13−Do NOT extract general background, epidemiology, treatment−only advice, administrative guidance,
14or recommendations that do not help diagnose a disease.
15−Each extracted recommendation must be self−contained and understandable without the full guideline.
16−Stay faithful to the source. Do not add, weaken, strengthen, or invent any criterion.
17
18# Output Format
19Return ONLY a JSON array. Each item must have:
20{
21"disease": "<diagnosed disease name>",
22"icd10": "<3−character ICD−10 category if available, otherwise null>",
23"recommendation": "<self−contained diagnostic recommendation>"
24}
21

Skill Initialization: Recommendation Merge
1# Role
2You merge several diagnostic recommendations that all diagnose the SAME disease
3(the same ICD−10 category) into ONE coherent, self−contained diagnostic statement.
4
5# Disease
6{disease}
7
8# ICD−10 Category
9{icd10}
10
11# Recommendations
12{recommendations}
13
14# Instructions
15−Synthesize ALL diagnostic criteria into one coherent recommendation.
16−Combine overlapping criteria, keep every distinct diagnostic criterion, and remove pure repetition.
17−Preserve thresholds, test names, signs, symptoms, imaging findings, laboratory findings, pathology,
18exclusion criteria, and differential clues when present.
19−Stay faithful: do not add, strengthen, weaken, or invent any criterion or threshold.
20−The merged recommendation should be concise but complete enough to support diagnosis.
21
22# Output Format
23Return ONLY the merged diagnostic recommendation text.
22

Skill Initialization: Recommendation to Executable Skill
1# Task
2Convert the diagnostic recommendation into an executable Python diagnostic skill.
3
4# Disease
5{disease}
6
7# ICD−10 Category
8{icd10}
9
10# Diagnostic Recommendation
11{recommendation}
12
13# Requirements
14Write a Python function named diagnose(case) that takes one dictionary case as input
15and returns a dictionary:
16{
17"disease": "<disease name>",
18"icd10": "<3−character ICD−10 category>",
19"tier": <0, 1, 2, or 3>,
20"score": <tier divided by 3>,
21"rationale": "<brief explanation>"
22}
23
24# Tier Contract
25−tier = 0: The case provides no meaningful support, contradicts the disease, or lacks the required evidence.
26−tier = 1: The case is compatible with the disease but only has nonspecific or weak supporting evidence.
27−tier = 2: The case is strongly suggestive of the disease based on characteristic findings.
28−tier = 3: The case confirms the disease using decisive evidence such as definitive imaging,
29pathology, microbiology, diagnostic test, or explicit gold−standard criterion.
30
31# Instructions
32−Encode the recommendation as explicit, executable decision logic.
33−Use only variables that can be extracted from a patient case.
34−Handle missing values safely. Missing evidence should not be treated as positive evidence.
35−Preserve the clinical thresholds and diagnostic criteria from the recommendation.
36−Include a docstring that documents every expected key in the case dictionary.
37−Do not call external APIs, import non−standard libraries, or use hidden state.
38−The skill should be comparable across diseases through the same 0−−3 tier scale.
39
40# Output Format
41Return ONLY valid Python code.
23

Skill Evolution: Rationale Generation
1# Task
2You are given a patient case and its CONFIRMED final diagnosis.
3Produce a diagnostic rationale.
4
5# Patient Case
6{question}
7
8# Confirmed Final Diagnosis
9{disease} (ICD−10 category {icd10})
10
11# Instructions
12Produce:
131. key_variables: the clinically IMPORTANT variables from the case. For each variable,
14give its name and VALUE exactly as stated in the case whenever possible.
152. rationale: FIRST state the necessary variables and their values; THEN explain why they
16establish {disease}.
17
18# Output Format
19Return ONLY a JSON object:
20{
21"key_variables": [
22{"name": "<variable name>", "value": "<value from case>"}
23],
24"rationale": "<diagnostic rationale>"
25}
Skill Evolution: Case-to-Recommendation Distillation
1# Task
2Distill recurring diagnostic criteria for the same disease from multiple case rationales.
3
4# Disease
5{disease}
6
7# ICD−10 Category
8{icd10}
9
10# Case Rationales
11{rationales}
12
13# Instructions
14−Identify recurring criteria that reliably support the diagnosis across cases.
15−Keep clinically meaningful symptoms, signs, labs, imaging findings, pathology findings,
16history, risk factors, and exclusion criteria.
17−Drop incidental, case−specific, demographic, or noisy details that do not define the disease.
18−Add discriminating features that help distinguish this disease from plausible alternatives
19when they are supported by the rationales.
20−Do not invent criteria not supported by the cases.
21−Write the result as a reusable diagnostic recommendation for future cases.
22
23# Output Format
24Return ONLY the distilled diagnostic recommendation.
24

Skill Evolution: Case-to-Recommendation Refinement
1# Task
2Refine an existing diagnostic recommendation using a new batch of case rationales.
3
4# Disease
5{disease}
6
7# ICD−10 Category
8{icd10}
9
10# Existing Recommendation
11{current_recommendation}
12
13# New Case Rationales
14{new_rationales}
15
16# Instructions
17−Preserve the useful diagnostic criteria from the existing recommendation.
18−Add new recurring criteria only if they are clinically meaningful and supported by the new batch.
19−Remove or soften criteria that appear overly specific, incidental, or unsupported.
20−Avoid bloating the recommendation with one−off details.
21−Keep the recommendation coherent, self−contained, and faithful to the evidence.
22−Do not add thresholds, tests, or claims that are not supported by the rationales.
23
24# Output Format
25Return ONLY the refined diagnostic recommendation.
Skill Execution: Differential Diagnosis Proposal
1# Task
2Give EXACTLY the {k} most likely diagnoses for this patient as ICD−10 3−character categories,
3ranked from most to least likely.
4
5# Patient Case
6{question}
7
8# Rules
9−Each candidate is one ICD−10 3−character category.
10−Diagnose at the category level, not the decimal subcode level.
11−You MUST return EXACTLY {k} candidates with {k} DIFFERENT categories.
12−Prefer specific disease categories that best explain the patient case.
13−Do not include symptoms, procedures, treatments, or non−disease concepts as diagnoses.
14
15# Output Format
16Reply with ONLY a JSON array of exactly {k} objects:
17[
18{"disease": "<ICD−10 category title>", "icd10": "<3−character ICD−10 code>"}
19]
25

Skill Execution: Skill Feature Grounding
1# Task
2You are given a patient case and a diagnostic skill function.
3Extract the function’s input as a JSON object matching the case dict described in the docstring.
4
5# Patient Case
6{question}
7
8# Skill Function
9{skill_content}
10
11# Instructions
12−Read the docstring to identify every key the case dict expects.
13−Extract each value from the patient case as faithfully as possible.
14−Use null if the case does not state the value.
15−Booleans must be true, false, or null.
16−Numeric values should be numbers when possible.
17−Do not infer unsupported facts.
18−Do not add keys that are not requested by the skill docstring.
19
20# Output Format
21Return ONLY a JSON object that can be passed directly as the case argument to diagnose(case).
Skill Execution: Efficient Skill Feature Grounding
1# Task
2You are given a patient case and several diagnostic skill functions, one per candidate diagnosis.
3Each skill lists its input features and their meaning in the docstring.
4In a SINGLE pass, identify which features are clearly PRESENT (true) in the patient case.
5
6# Patient Case
7{question}
8
9# Diagnostic Skills
10{skills_block}
11
12# Instructions
13−For each candidate diagnosis, list ONLY the feature keys that the case shows to be TRUE (present).
14−Omit every feature that is false, absent, or not mentioned; any feature you do not list is assumed false or unknown.
15−Do not infer unsupported facts.
16−Do not add keys that are not requested by the skill docstrings.
17
18# Output Format
19Return ONLY a JSON object mapping each candidate’s ICD−10 code to the list of its true features,
20i.e. {{"<icd10>": ["<true_feature>", ...], ...}} (use [] if none are true),
21directly usable to construct the case argument for each diagnose(case).
26